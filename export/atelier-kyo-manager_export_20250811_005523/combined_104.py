
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\internal\encoder.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Code for encoding protocol message primitives.

Contains the logic for encoding every logical protocol field type
into one of the 5 physical wire types.

This code is designed to push the Python interpreter's performance to the
limits.

The basic idea is that at startup time, for every field (i.e. every
FieldDescriptor) we construct two functions:  a "sizer" and an "encoder".  The
sizer takes a value of this field's type and computes its byte size.  The
encoder takes a writer function and a value.  It encodes the value into byte
strings and invokes the writer function to write those strings.  Typically the
writer function is the write() method of a BytesIO.

We try to do as much work as possible when constructing the writer and the
sizer rather than when calling them.  In particular:
* We copy any needed global functions to local variables, so that we do not need
  to do costly global table lookups at runtime.
* Similarly, we try to do any attribute lookups at startup time if possible.
* Every field's tag is encoded to bytes at startup, since it can't change at
  runtime.
* Whatever component of the field size we can compute at startup, we do.
* We *avoid* sharing code if doing so would make the code slower and not sharing
  does not burden us too much.  For example, encoders for repeated fields do
  not just call the encoders for singular fields in a loop because this would
  add an extra function call overhead for every loop iteration; instead, we
  manually inline the single-value encoder into the loop.
* If a Python function lacks a return statement, Python actually generates
  instructions to pop the result of the last statement off the stack, push
  None onto the stack, and then return that.  If we really don't care what
  value is returned, then we can save two instructions by returning the
  result of the last statement.  It looks funny but it helps.
* We assume that type and bounds checking has happened at a higher level.
"""

__author__ = 'kenton@google.com (Kenton Varda)'

import struct

from google.protobuf.internal import wire_format


# This will overflow and thus become IEEE-754 "infinity".  We would use
# "float('inf')" but it doesn't work on Windows pre-Python-2.6.
_POS_INF = 1e10000
_NEG_INF = -_POS_INF


def _VarintSize(value):
  """Compute the size of a varint value."""
  if value <= 0x7f: return 1
  if value <= 0x3fff: return 2
  if value <= 0x1fffff: return 3
  if value <= 0xfffffff: return 4
  if value <= 0x7ffffffff: return 5
  if value <= 0x3ffffffffff: return 6
  if value <= 0x1ffffffffffff: return 7
  if value <= 0xffffffffffffff: return 8
  if value <= 0x7fffffffffffffff: return 9
  return 10


def _SignedVarintSize(value):
  """Compute the size of a signed varint value."""
  if value < 0: return 10
  if value <= 0x7f: return 1
  if value <= 0x3fff: return 2
  if value <= 0x1fffff: return 3
  if value <= 0xfffffff: return 4
  if value <= 0x7ffffffff: return 5
  if value <= 0x3ffffffffff: return 6
  if value <= 0x1ffffffffffff: return 7
  if value <= 0xffffffffffffff: return 8
  if value <= 0x7fffffffffffffff: return 9
  return 10


def _TagSize(field_number):
  """Returns the number of bytes required to serialize a tag with this field
  number."""
  # Just pass in type 0, since the type won't affect the tag+type size.
  return _VarintSize(wire_format.PackTag(field_number, 0))


# --------------------------------------------------------------------
# In this section we define some generic sizers.  Each of these functions
# takes parameters specific to a particular field type, e.g. int32 or fixed64.
# It returns another function which in turn takes parameters specific to a
# particular field, e.g. the field number and whether it is repeated or packed.
# Look at the next section to see how these are used.


def _SimpleSizer(compute_value_size):
  """A sizer which uses the function compute_value_size to compute the size of
  each value.  Typically compute_value_size is _VarintSize."""

  def SpecificSizer(field_number, is_repeated, is_packed):
    tag_size = _TagSize(field_number)
    if is_packed:
      local_VarintSize = _VarintSize
      def PackedFieldSize(value):
        result = 0
        for element in value:
          result += compute_value_size(element)
        return result + local_VarintSize(result) + tag_size
      return PackedFieldSize
    elif is_repeated:
      def RepeatedFieldSize(value):
        result = tag_size * len(value)
        for element in value:
          result += compute_value_size(element)
        return result
      return RepeatedFieldSize
    else:
      def FieldSize(value):
        return tag_size + compute_value_size(value)
      return FieldSize

  return SpecificSizer


def _ModifiedSizer(compute_value_size, modify_value):
  """Like SimpleSizer, but modify_value is invoked on each value before it is
  passed to compute_value_size.  modify_value is typically ZigZagEncode."""

  def SpecificSizer(field_number, is_repeated, is_packed):
    tag_size = _TagSize(field_number)
    if is_packed:
      local_VarintSize = _VarintSize
      def PackedFieldSize(value):
        result = 0
        for element in value:
          result += compute_value_size(modify_value(element))
        return result + local_VarintSize(result) + tag_size
      return PackedFieldSize
    elif is_repeated:
      def RepeatedFieldSize(value):
        result = tag_size * len(value)
        for element in value:
          result += compute_value_size(modify_value(element))
        return result
      return RepeatedFieldSize
    else:
      def FieldSize(value):
        return tag_size + compute_value_size(modify_value(value))
      return FieldSize

  return SpecificSizer


def _FixedSizer(value_size):
  """Like _SimpleSizer except for a fixed-size field.  The input is the size
  of one value."""

  def SpecificSizer(field_number, is_repeated, is_packed):
    tag_size = _TagSize(field_number)
    if is_packed:
      local_VarintSize = _VarintSize
      def PackedFieldSize(value):
        result = len(value) * value_size
        return result + local_VarintSize(result) + tag_size
      return PackedFieldSize
    elif is_repeated:
      element_size = value_size + tag_size
      def RepeatedFieldSize(value):
        return len(value) * element_size
      return RepeatedFieldSize
    else:
      field_size = value_size + tag_size
      def FieldSize(value):
        return field_size
      return FieldSize

  return SpecificSizer


# ====================================================================
# Here we declare a sizer constructor for each field type.  Each "sizer
# constructor" is a function that takes (field_number, is_repeated, is_packed)
# as parameters and returns a sizer, which in turn takes a field value as
# a parameter and returns its encoded size.


Int32Sizer = Int64Sizer = EnumSizer = _SimpleSizer(_SignedVarintSize)

UInt32Sizer = UInt64Sizer = _SimpleSizer(_VarintSize)

SInt32Sizer = SInt64Sizer = _ModifiedSizer(
    _SignedVarintSize, wire_format.ZigZagEncode)

Fixed32Sizer = SFixed32Sizer = FloatSizer  = _FixedSizer(4)
Fixed64Sizer = SFixed64Sizer = DoubleSizer = _FixedSizer(8)

BoolSizer = _FixedSizer(1)


def StringSizer(field_number, is_repeated, is_packed):
  """Returns a sizer for a string field."""

  tag_size = _TagSize(field_number)
  local_VarintSize = _VarintSize
  local_len = len
  assert not is_packed
  if is_repeated:
    def RepeatedFieldSize(value):
      result = tag_size * len(value)
      for element in value:
        l = local_len(element.encode('utf-8'))
        result += local_VarintSize(l) + l
      return result
    return RepeatedFieldSize
  else:
    def FieldSize(value):
      l = local_len(value.encode('utf-8'))
      return tag_size + local_VarintSize(l) + l
    return FieldSize


def BytesSizer(field_number, is_repeated, is_packed):
  """Returns a sizer for a bytes field."""

  tag_size = _TagSize(field_number)
  local_VarintSize = _VarintSize
  local_len = len
  assert not is_packed
  if is_repeated:
    def RepeatedFieldSize(value):
      result = tag_size * len(value)
      for element in value:
        l = local_len(element)
        result += local_VarintSize(l) + l
      return result
    return RepeatedFieldSize
  else:
    def FieldSize(value):
      l = local_len(value)
      return tag_size + local_VarintSize(l) + l
    return FieldSize


def GroupSizer(field_number, is_repeated, is_packed):
  """Returns a sizer for a group field."""

  tag_size = _TagSize(field_number) * 2
  assert not is_packed
  if is_repeated:
    def RepeatedFieldSize(value):
      result = tag_size * len(value)
      for element in value:
        result += element.ByteSize()
      return result
    return RepeatedFieldSize
  else:
    def FieldSize(value):
      return tag_size + value.ByteSize()
    return FieldSize


def MessageSizer(field_number, is_repeated, is_packed):
  """Returns a sizer for a message field."""

  tag_size = _TagSize(field_number)
  local_VarintSize = _VarintSize
  assert not is_packed
  if is_repeated:
    def RepeatedFieldSize(value):
      result = tag_size * len(value)
      for element in value:
        l = element.ByteSize()
        result += local_VarintSize(l) + l
      return result
    return RepeatedFieldSize
  else:
    def FieldSize(value):
      l = value.ByteSize()
      return tag_size + local_VarintSize(l) + l
    return FieldSize


# --------------------------------------------------------------------
# MessageSet is special: it needs custom logic to compute its size properly.


def MessageSetItemSizer(field_number):
  """Returns a sizer for extensions of MessageSet.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """
  static_size = (_TagSize(1) * 2 + _TagSize(2) + _VarintSize(field_number) +
                 _TagSize(3))
  local_VarintSize = _VarintSize

  def FieldSize(value):
    l = value.ByteSize()
    return static_size + local_VarintSize(l) + l

  return FieldSize


# --------------------------------------------------------------------
# Map is special: it needs custom logic to compute its size properly.


def MapSizer(field_descriptor, is_message_map):
  """Returns a sizer for a map field."""

  # Can't look at field_descriptor.message_type._concrete_class because it may
  # not have been initialized yet.
  message_type = field_descriptor.message_type
  message_sizer = MessageSizer(field_descriptor.number, False, False)

  def FieldSize(map_value):
    total = 0
    for key in map_value:
      value = map_value[key]
      # It's wasteful to create the messages and throw them away one second
      # later since we'll do the same for the actual encode.  But there's not an
      # obvious way to avoid this within the current design without tons of code
      # duplication. For message map, value.ByteSize() should be called to
      # update the status.
      entry_msg = message_type._concrete_class(key=key, value=value)
      total += message_sizer(entry_msg)
      if is_message_map:
        value.ByteSize()
    return total

  return FieldSize

# ====================================================================
# Encoders!


def _VarintEncoder():
  """Return an encoder for a basic varint value (does not include tag)."""

  local_int2byte = struct.Struct('>B').pack

  def EncodeVarint(write, value, unused_deterministic=None):
    bits = value & 0x7f
    value >>= 7
    while value:
      write(local_int2byte(0x80|bits))
      bits = value & 0x7f
      value >>= 7
    return write(local_int2byte(bits))

  return EncodeVarint


def _SignedVarintEncoder():
  """Return an encoder for a basic signed varint value (does not include
  tag)."""

  local_int2byte = struct.Struct('>B').pack

  def EncodeSignedVarint(write, value, unused_deterministic=None):
    if value < 0:
      value += (1 << 64)
    bits = value & 0x7f
    value >>= 7
    while value:
      write(local_int2byte(0x80|bits))
      bits = value & 0x7f
      value >>= 7
    return write(local_int2byte(bits))

  return EncodeSignedVarint


_EncodeVarint = _VarintEncoder()
_EncodeSignedVarint = _SignedVarintEncoder()


def _VarintBytes(value):
  """Encode the given integer as a varint and return the bytes.  This is only
  called at startup time so it doesn't need to be fast."""

  pieces = []
  _EncodeVarint(pieces.append, value, True)
  return b"".join(pieces)


def TagBytes(field_number, wire_type):
  """Encode the given tag and return the bytes.  Only called at startup."""

  return bytes(_VarintBytes(wire_format.PackTag(field_number, wire_type)))

# --------------------------------------------------------------------
# As with sizers (see above), we have a number of common encoder
# implementations.


def _SimpleEncoder(wire_type, encode_value, compute_value_size):
  """Return a constructor for an encoder for fields of a particular type.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      encode_value:  A function which encodes an individual value, e.g.
        _EncodeVarint().
      compute_value_size:  A function which computes the size of an individual
        value, e.g. _VarintSize().
  """

  def SpecificEncoder(field_number, is_repeated, is_packed):
    if is_packed:
      tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
      local_EncodeVarint = _EncodeVarint
      def EncodePackedField(write, value, deterministic):
        write(tag_bytes)
        size = 0
        for element in value:
          size += compute_value_size(element)
        local_EncodeVarint(write, size, deterministic)
        for element in value:
          encode_value(write, element, deterministic)
      return EncodePackedField
    elif is_repeated:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeRepeatedField(write, value, deterministic):
        for element in value:
          write(tag_bytes)
          encode_value(write, element, deterministic)
      return EncodeRepeatedField
    else:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeField(write, value, deterministic):
        write(tag_bytes)
        return encode_value(write, value, deterministic)
      return EncodeField

  return SpecificEncoder


def _ModifiedEncoder(wire_type, encode_value, compute_value_size, modify_value):
  """Like SimpleEncoder but additionally invokes modify_value on every value
  before passing it to encode_value.  Usually modify_value is ZigZagEncode."""

  def SpecificEncoder(field_number, is_repeated, is_packed):
    if is_packed:
      tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
      local_EncodeVarint = _EncodeVarint
      def EncodePackedField(write, value, deterministic):
        write(tag_bytes)
        size = 0
        for element in value:
          size += compute_value_size(modify_value(element))
        local_EncodeVarint(write, size, deterministic)
        for element in value:
          encode_value(write, modify_value(element), deterministic)
      return EncodePackedField
    elif is_repeated:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeRepeatedField(write, value, deterministic):
        for element in value:
          write(tag_bytes)
          encode_value(write, modify_value(element), deterministic)
      return EncodeRepeatedField
    else:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeField(write, value, deterministic):
        write(tag_bytes)
        return encode_value(write, modify_value(value), deterministic)
      return EncodeField

  return SpecificEncoder


def _StructPackEncoder(wire_type, format):
  """Return a constructor for an encoder for a fixed-width field.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      format:  The format string to pass to struct.pack().
  """

  value_size = struct.calcsize(format)

  def SpecificEncoder(field_number, is_repeated, is_packed):
    local_struct_pack = struct.pack
    if is_packed:
      tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
      local_EncodeVarint = _EncodeVarint
      def EncodePackedField(write, value, deterministic):
        write(tag_bytes)
        local_EncodeVarint(write, len(value) * value_size, deterministic)
        for element in value:
          write(local_struct_pack(format, element))
      return EncodePackedField
    elif is_repeated:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeRepeatedField(write, value, unused_deterministic=None):
        for element in value:
          write(tag_bytes)
          write(local_struct_pack(format, element))
      return EncodeRepeatedField
    else:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeField(write, value, unused_deterministic=None):
        write(tag_bytes)
        return write(local_struct_pack(format, value))
      return EncodeField

  return SpecificEncoder


def _FloatingPointEncoder(wire_type, format):
  """Return a constructor for an encoder for float fields.

  This is like StructPackEncoder, but catches errors that may be due to
  passing non-finite floating-point values to struct.pack, and makes a
  second attempt to encode those values.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      format:  The format string to pass to struct.pack().
  """

  value_size = struct.calcsize(format)
  if value_size == 4:
    def EncodeNonFiniteOrRaise(write, value):
      # Remember that the serialized form uses little-endian byte order.
      if value == _POS_INF:
        write(b'\x00\x00\x80\x7F')
      elif value == _NEG_INF:
        write(b'\x00\x00\x80\xFF')
      elif value != value:           # NaN
        write(b'\x00\x00\xC0\x7F')
      else:
        raise
  elif value_size == 8:
    def EncodeNonFiniteOrRaise(write, value):
      if value == _POS_INF:
        write(b'\x00\x00\x00\x00\x00\x00\xF0\x7F')
      elif value == _NEG_INF:
        write(b'\x00\x00\x00\x00\x00\x00\xF0\xFF')
      elif value != value:                         # NaN
        write(b'\x00\x00\x00\x00\x00\x00\xF8\x7F')
      else:
        raise
  else:
    raise ValueError('Can\'t encode floating-point values that are '
                     '%d bytes long (only 4 or 8)' % value_size)

  def SpecificEncoder(field_number, is_repeated, is_packed):
    local_struct_pack = struct.pack
    if is_packed:
      tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
      local_EncodeVarint = _EncodeVarint
      def EncodePackedField(write, value, deterministic):
        write(tag_bytes)
        local_EncodeVarint(write, len(value) * value_size, deterministic)
        for element in value:
          # This try/except block is going to be faster than any code that
          # we could write to check whether element is finite.
          try:
            write(local_struct_pack(format, element))
          except SystemError:
            EncodeNonFiniteOrRaise(write, element)
      return EncodePackedField
    elif is_repeated:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeRepeatedField(write, value, unused_deterministic=None):
        for element in value:
          write(tag_bytes)
          try:
            write(local_struct_pack(format, element))
          except SystemError:
            EncodeNonFiniteOrRaise(write, element)
      return EncodeRepeatedField
    else:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeField(write, value, unused_deterministic=None):
        write(tag_bytes)
        try:
          write(local_struct_pack(format, value))
        except SystemError:
          EncodeNonFiniteOrRaise(write, value)
      return EncodeField

  return SpecificEncoder


# ====================================================================
# Here we declare an encoder constructor for each field type.  These work
# very similarly to sizer constructors, described earlier.


Int32Encoder = Int64Encoder = EnumEncoder = _SimpleEncoder(
    wire_format.WIRETYPE_VARINT, _EncodeSignedVarint, _SignedVarintSize)

UInt32Encoder = UInt64Encoder = _SimpleEncoder(
    wire_format.WIRETYPE_VARINT, _EncodeVarint, _VarintSize)

SInt32Encoder = SInt64Encoder = _ModifiedEncoder(
    wire_format.WIRETYPE_VARINT, _EncodeVarint, _VarintSize,
    wire_format.ZigZagEncode)

# Note that Python conveniently guarantees that when using the '<' prefix on
# formats, they will also have the same size across all platforms (as opposed
# to without the prefix, where their sizes depend on the C compiler's basic
# type sizes).
Fixed32Encoder  = _StructPackEncoder(wire_format.WIRETYPE_FIXED32, '<I')
Fixed64Encoder  = _StructPackEncoder(wire_format.WIRETYPE_FIXED64, '<Q')
SFixed32Encoder = _StructPackEncoder(wire_format.WIRETYPE_FIXED32, '<i')
SFixed64Encoder = _StructPackEncoder(wire_format.WIRETYPE_FIXED64, '<q')
FloatEncoder    = _FloatingPointEncoder(wire_format.WIRETYPE_FIXED32, '<f')
DoubleEncoder   = _FloatingPointEncoder(wire_format.WIRETYPE_FIXED64, '<d')


def BoolEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a boolean field."""

  false_byte = b'\x00'
  true_byte = b'\x01'
  if is_packed:
    tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
    local_EncodeVarint = _EncodeVarint
    def EncodePackedField(write, value, deterministic):
      write(tag_bytes)
      local_EncodeVarint(write, len(value), deterministic)
      for element in value:
        if element:
          write(true_byte)
        else:
          write(false_byte)
    return EncodePackedField
  elif is_repeated:
    tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_VARINT)
    def EncodeRepeatedField(write, value, unused_deterministic=None):
      for element in value:
        write(tag_bytes)
        if element:
          write(true_byte)
        else:
          write(false_byte)
    return EncodeRepeatedField
  else:
    tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_VARINT)
    def EncodeField(write, value, unused_deterministic=None):
      write(tag_bytes)
      if value:
        return write(true_byte)
      return write(false_byte)
    return EncodeField


def StringEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a string field."""

  tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
  local_EncodeVarint = _EncodeVarint
  local_len = len
  assert not is_packed
  if is_repeated:
    def EncodeRepeatedField(write, value, deterministic):
      for element in value:
        encoded = element.encode('utf-8')
        write(tag)
        local_EncodeVarint(write, local_len(encoded), deterministic)
        write(encoded)
    return EncodeRepeatedField
  else:
    def EncodeField(write, value, deterministic):
      encoded = value.encode('utf-8')
      write(tag)
      local_EncodeVarint(write, local_len(encoded), deterministic)
      return write(encoded)
    return EncodeField


def BytesEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a bytes field."""

  tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
  local_EncodeVarint = _EncodeVarint
  local_len = len
  assert not is_packed
  if is_repeated:
    def EncodeRepeatedField(write, value, deterministic):
      for element in value:
        write(tag)
        local_EncodeVarint(write, local_len(element), deterministic)
        write(element)
    return EncodeRepeatedField
  else:
    def EncodeField(write, value, deterministic):
      write(tag)
      local_EncodeVarint(write, local_len(value), deterministic)
      return write(value)
    return EncodeField


def GroupEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a group field."""

  start_tag = TagBytes(field_number, wire_format.WIRETYPE_START_GROUP)
  end_tag = TagBytes(field_number, wire_format.WIRETYPE_END_GROUP)
  assert not is_packed
  if is_repeated:
    def EncodeRepeatedField(write, value, deterministic):
      for element in value:
        write(start_tag)
        element._InternalSerialize(write, deterministic)
        write(end_tag)
    return EncodeRepeatedField
  else:
    def EncodeField(write, value, deterministic):
      write(start_tag)
      value._InternalSerialize(write, deterministic)
      return write(end_tag)
    return EncodeField


def MessageEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a message field."""

  tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
  local_EncodeVarint = _EncodeVarint
  assert not is_packed
  if is_repeated:
    def EncodeRepeatedField(write, value, deterministic):
      for element in value:
        write(tag)
        local_EncodeVarint(write, element.ByteSize(), deterministic)
        element._InternalSerialize(write, deterministic)
    return EncodeRepeatedField
  else:
    def EncodeField(write, value, deterministic):
      write(tag)
      local_EncodeVarint(write, value.ByteSize(), deterministic)
      return value._InternalSerialize(write, deterministic)
    return EncodeField


# --------------------------------------------------------------------
# As before, MessageSet is special.


def MessageSetItemEncoder(field_number):
  """Encoder for extensions of MessageSet.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """
  start_bytes = b"".join([
      TagBytes(1, wire_format.WIRETYPE_START_GROUP),
      TagBytes(2, wire_format.WIRETYPE_VARINT),
      _VarintBytes(field_number),
      TagBytes(3, wire_format.WIRETYPE_LENGTH_DELIMITED)])
  end_bytes = TagBytes(1, wire_format.WIRETYPE_END_GROUP)
  local_EncodeVarint = _EncodeVarint

  def EncodeField(write, value, deterministic):
    write(start_bytes)
    local_EncodeVarint(write, value.ByteSize(), deterministic)
    value._InternalSerialize(write, deterministic)
    return write(end_bytes)

  return EncodeField


# --------------------------------------------------------------------
# As before, Map is special.


def MapEncoder(field_descriptor):
  """Encoder for extensions of MessageSet.

  Maps always have a wire format like this:
    message MapEntry {
      key_type key = 1;
      value_type value = 2;
    }
    repeated MapEntry map = N;
  """
  # Can't look at field_descriptor.message_type._concrete_class because it may
  # not have been initialized yet.
  message_type = field_descriptor.message_type
  encode_message = MessageEncoder(field_descriptor.number, False, False)

  def EncodeField(write, value, deterministic):
    value_keys = sorted(value.keys()) if deterministic else value
    for key in value_keys:
      entry_msg = message_type._concrete_class(key=key, value=value[key])
      encode_message(write, entry_msg, deterministic)

  return EncodeField

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\vector.py ===
from sympy import (S, sympify, expand, sqrt, Add, zeros, acos,
                   ImmutableMatrix as Matrix, simplify)
from sympy.simplify.trigsimp import trigsimp
from sympy.printing.defaults import Printable
from sympy.utilities.misc import filldedent
from sympy.core.evalf import EvalfMixin

from mpmath.libmp.libmpf import prec_to_dps


__all__ = ['Vector']


class Vector(Printable, EvalfMixin):
    """The class used to define vectors.

    It along with ReferenceFrame are the building blocks of describing a
    classical mechanics system in PyDy and sympy.physics.vector.

    Attributes
    ==========

    simp : Boolean
        Let certain methods use trigsimp on their outputs

    """

    simp = False
    is_number = False

    def __init__(self, inlist):
        """This is the constructor for the Vector class. You should not be
        calling this, it should only be used by other functions. You should be
        treating Vectors like you would with if you were doing the math by
        hand, and getting the first 3 from the standard basis vectors from a
        ReferenceFrame.

        The only exception is to create a zero vector:
        zv = Vector(0)

        """

        self.args = []
        if inlist == 0:
            inlist = []
        if isinstance(inlist, dict):
            d = inlist
        else:
            d = {}
            for inp in inlist:
                if inp[1] in d:
                    d[inp[1]] += inp[0]
                else:
                    d[inp[1]] = inp[0]

        for k, v in d.items():
            if v != Matrix([0, 0, 0]):
                self.args.append((v, k))

    @property
    def func(self):
        """Returns the class Vector. """
        return Vector

    def __hash__(self):
        return hash(tuple(self.args))

    def __add__(self, other):
        """The add operator for Vector. """
        if other == 0:
            return self
        other = _check_vector(other)
        return Vector(self.args + other.args)

    def dot(self, other):
        """Dot product of two vectors.

        Returns a scalar, the dot product of the two Vectors

        Parameters
        ==========

        other : Vector
            The Vector which we are dotting with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dot
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> dot(N.x, N.x)
        1
        >>> dot(N.x, N.y)
        0
        >>> A = N.orientnew('A', 'Axis', [q1, N.x])
        >>> dot(N.y, A.y)
        cos(q1)

        """

        from sympy.physics.vector.dyadic import Dyadic, _check_dyadic
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Vector(0)
            for v in other.args:
                ol += v[0] * v[2] * (v[1].dot(self))
            return ol
        other = _check_vector(other)
        out = S.Zero
        for v1 in self.args:
            for v2 in other.args:
                out += ((v2[0].T) * (v2[1].dcm(v1[1])) * (v1[0]))[0]
        if Vector.simp:
            return trigsimp(out, recursive=True)
        else:
            return out

    def __truediv__(self, other):
        """This uses mul and inputs self and 1 divided by other. """
        return self.__mul__(S.One / other)

    def __eq__(self, other):
        """Tests for equality.

        It is very import to note that this is only as good as the SymPy
        equality test; False does not always mean they are not equivalent
        Vectors.
        If other is 0, and self is empty, returns True.
        If other is 0 and self is not empty, returns False.
        If none of the above, only accepts other as a Vector.

        """

        if other == 0:
            other = Vector(0)
        try:
            other = _check_vector(other)
        except TypeError:
            return False
        if (self.args == []) and (other.args == []):
            return True
        elif (self.args == []) or (other.args == []):
            return False

        frame = self.args[0][1]
        for v in frame:
            if expand((self - other).dot(v)) != 0:
                return False
        return True

    def __mul__(self, other):
        """Multiplies the Vector by a sympifyable expression.

        Parameters
        ==========

        other : Sympifyable
            The scalar to multiply this Vector with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> b = Symbol('b')
        >>> V = 10 * b * N.x
        >>> print(V)
        10*b*N.x

        """

        newlist = list(self.args)
        other = sympify(other)
        for i in range(len(newlist)):
            newlist[i] = (other * newlist[i][0], newlist[i][1])
        return Vector(newlist)

    def __neg__(self):
        return self * -1

    def outer(self, other):
        """Outer product between two Vectors.

        A rank increasing operation, which returns a Dyadic from two Vectors

        Parameters
        ==========

        other : Vector
            The Vector to take the outer product with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> outer(N.x, N.x)
        (N.x|N.x)

        """

        from sympy.physics.vector.dyadic import Dyadic
        other = _check_vector(other)
        ol = Dyadic(0)
        for v in self.args:
            for v2 in other.args:
                # it looks this way because if we are in the same frame and
                # use the enumerate function on the same frame in a nested
                # fashion, then bad things happen
                ol += Dyadic([(v[0][0] * v2[0][0], v[1].x, v2[1].x)])
                ol += Dyadic([(v[0][0] * v2[0][1], v[1].x, v2[1].y)])
                ol += Dyadic([(v[0][0] * v2[0][2], v[1].x, v2[1].z)])
                ol += Dyadic([(v[0][1] * v2[0][0], v[1].y, v2[1].x)])
                ol += Dyadic([(v[0][1] * v2[0][1], v[1].y, v2[1].y)])
                ol += Dyadic([(v[0][1] * v2[0][2], v[1].y, v2[1].z)])
                ol += Dyadic([(v[0][2] * v2[0][0], v[1].z, v2[1].x)])
                ol += Dyadic([(v[0][2] * v2[0][1], v[1].z, v2[1].y)])
                ol += Dyadic([(v[0][2] * v2[0][2], v[1].z, v2[1].z)])
        return ol

    def _latex(self, printer):
        """Latex Printing method. """

        ar = self.args  # just to shorten things
        if len(ar) == 0:
            return str(0)
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            for j in 0, 1, 2:
                # if the coef of the basis vector is 1, we skip the 1
                if v[0][j] == 1:
                    ol.append(' + ' + v[1].latex_vecs[j])
                # if the coef of the basis vector is -1, we skip the 1
                elif v[0][j] == -1:
                    ol.append(' - ' + v[1].latex_vecs[j])
                elif v[0][j] != 0:
                    # If the coefficient of the basis vector is not 1 or -1;
                    # also, we might wrap it in parentheses, for readability.
                    arg_str = printer._print(v[0][j])
                    if isinstance(v[0][j], Add):
                        arg_str = "(%s)" % arg_str
                    if arg_str[0] == '-':
                        arg_str = arg_str[1:]
                        str_start = ' - '
                    else:
                        str_start = ' + '
                    ol.append(str_start + arg_str + v[1].latex_vecs[j])
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def _pretty(self, printer):
        """Pretty Printing method. """
        from sympy.printing.pretty.stringpict import prettyForm

        terms = []

        def juxtapose(a, b):
            pa = printer._print(a)
            pb = printer._print(b)
            if a.is_Add:
                pa = prettyForm(*pa.parens())
            return printer._print_seq([pa, pb], delimiter=' ')

        for M, N in self.args:
            for i in range(3):
                if M[i] == 0:
                    continue
                elif M[i] == 1:
                    terms.append(prettyForm(N.pretty_vecs[i]))
                elif M[i] == -1:
                    terms.append(prettyForm("-1") * prettyForm(N.pretty_vecs[i]))
                else:
                    terms.append(juxtapose(M[i], N.pretty_vecs[i]))

        if terms:
            pretty_result = prettyForm.__add__(*terms)
        else:
            pretty_result = prettyForm("0")

        return pretty_result

    def __rsub__(self, other):
        return (-1 * self) + other

    def _sympystr(self, printer, order=True):
        """Printing method. """
        if not order or len(self.args) == 1:
            ar = list(self.args)
        elif len(self.args) == 0:
            return printer._print(0)
        else:
            d = {v[1]: v[0] for v in self.args}
            keys = sorted(d.keys(), key=lambda x: x.index)
            ar = []
            for key in keys:
                ar.append((d[key], key))
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            for j in 0, 1, 2:
                # if the coef of the basis vector is 1, we skip the 1
                if v[0][j] == 1:
                    ol.append(' + ' + v[1].str_vecs[j])
                # if the coef of the basis vector is -1, we skip the 1
                elif v[0][j] == -1:
                    ol.append(' - ' + v[1].str_vecs[j])
                elif v[0][j] != 0:
                    # If the coefficient of the basis vector is not 1 or -1;
                    # also, we might wrap it in parentheses, for readability.
                    arg_str = printer._print(v[0][j])
                    if isinstance(v[0][j], Add):
                        arg_str = "(%s)" % arg_str
                    if arg_str[0] == '-':
                        arg_str = arg_str[1:]
                        str_start = ' - '
                    else:
                        str_start = ' + '
                    ol.append(str_start + arg_str + '*' + v[1].str_vecs[j])
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def __sub__(self, other):
        """The subtraction operator. """
        return self.__add__(other * -1)

    def cross(self, other):
        """The cross product operator for two Vectors.

        Returns a Vector, expressed in the same ReferenceFrames as self.

        Parameters
        ==========

        other : Vector
            The Vector which we are crossing with

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame, cross
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> cross(N.x, N.y)
        N.z
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, q1, N.x)
        >>> cross(A.x, N.y)
        N.z
        >>> cross(N.y, A.x)
        - sin(q1)*A.y - cos(q1)*A.z

        """

        from sympy.physics.vector.dyadic import Dyadic, _check_dyadic
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Dyadic(0)
            for i, v in enumerate(other.args):
                ol += v[0] * ((self.cross(v[1])).outer(v[2]))
            return ol
        other = _check_vector(other)
        if other.args == []:
            return Vector(0)

        def _det(mat):
            """This is needed as a little method for to find the determinant
            of a list in python; needs to work for a 3x3 list.
            SymPy's Matrix will not take in Vector, so need a custom function.
            You should not be calling this.

            """

            return (mat[0][0] * (mat[1][1] * mat[2][2] - mat[1][2] * mat[2][1])
                    + mat[0][1] * (mat[1][2] * mat[2][0] - mat[1][0] *
                    mat[2][2]) + mat[0][2] * (mat[1][0] * mat[2][1] -
                    mat[1][1] * mat[2][0]))

        outlist = []
        ar = other.args  # For brevity
        for v in ar:
            tempx = v[1].x
            tempy = v[1].y
            tempz = v[1].z
            tempm = ([[tempx, tempy, tempz],
                      [self.dot(tempx), self.dot(tempy), self.dot(tempz)],
                      [Vector([v]).dot(tempx), Vector([v]).dot(tempy),
                       Vector([v]).dot(tempz)]])
            outlist += _det(tempm).args
        return Vector(outlist)

    __radd__ = __add__
    __rmul__ = __mul__

    def separate(self):
        """
        The constituents of this vector in different reference frames,
        as per its definition.

        Returns a dict mapping each ReferenceFrame to the corresponding
        constituent Vector.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> R1 = ReferenceFrame('R1')
        >>> R2 = ReferenceFrame('R2')
        >>> v = R1.x + R2.x
        >>> v.separate() == {R1: R1.x, R2: R2.x}
        True

        """

        components = {}
        for x in self.args:
            components[x[1]] = Vector([x])
        return components

    def __and__(self, other):
        return self.dot(other)
    __and__.__doc__ = dot.__doc__
    __rand__ = __and__

    def __xor__(self, other):
        return self.cross(other)
    __xor__.__doc__ = cross.__doc__

    def __or__(self, other):
        return self.outer(other)
    __or__.__doc__ = outer.__doc__

    def diff(self, var, frame, var_in_dcm=True):
        """Returns the partial derivative of the vector with respect to a
        variable in the provided reference frame.

        Parameters
        ==========
        var : Symbol
            What the partial derivative is taken with respect to.
        frame : ReferenceFrame
            The reference frame that the partial derivative is taken in.
        var_in_dcm : boolean
            If true, the differentiation algorithm assumes that the variable
            may be present in any of the direction cosine matrices that relate
            the frame to the frames of any component of the vector. But if it
            is known that the variable is not present in the direction cosine
            matrices, false can be set to skip full reexpression in the desired
            frame.

        Examples
        ========

        >>> from sympy import Symbol
        >>> from sympy.physics.vector import dynamicsymbols, ReferenceFrame
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> t = Symbol('t')
        >>> q1 = dynamicsymbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'Axis', [q1, N.y])
        >>> A.x.diff(t, N)
        - sin(q1)*q1'*N.x - cos(q1)*q1'*N.z
        >>> A.x.diff(t, N).express(A).simplify()
        - q1'*A.z
        >>> B = ReferenceFrame('B')
        >>> u1, u2 = dynamicsymbols('u1, u2')
        >>> v = u1 * A.x + u2 * B.y
        >>> v.diff(u2, N, var_in_dcm=False)
        B.y

        """

        from sympy.physics.vector.frame import _check_frame

        _check_frame(frame)
        var = sympify(var)

        inlist = []

        for vector_component in self.args:
            measure_number = vector_component[0]
            component_frame = vector_component[1]
            if component_frame == frame:
                inlist += [(measure_number.diff(var), frame)]
            else:
                # If the direction cosine matrix relating the component frame
                # with the derivative frame does not contain the variable.
                if not var_in_dcm or (frame.dcm(component_frame).diff(var) ==
                                      zeros(3, 3)):
                    inlist += [(measure_number.diff(var), component_frame)]
                else:  # else express in the frame
                    reexp_vec_comp = Vector([vector_component]).express(frame)
                    deriv = reexp_vec_comp.args[0][0].diff(var)
                    inlist += Vector([(deriv, frame)]).args

        return Vector(inlist)

    def express(self, otherframe, variables=False):
        """
        Returns a Vector equivalent to this one, expressed in otherframe.
        Uses the global express method.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The frame for this Vector to be described in

        variables : boolean
            If True, the coordinate symbols(if present) in this Vector
            are re-expressed in terms otherframe

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> q1 = dynamicsymbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = N.orientnew('A', 'Axis', [q1, N.y])
        >>> A.x.express(N)
        cos(q1)*N.x - sin(q1)*N.z

        """
        from sympy.physics.vector import express
        return express(self, otherframe, variables=variables)

    def to_matrix(self, reference_frame):
        """Returns the matrix form of the vector with respect to the given
        frame.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            The reference frame that the rows of the matrix correspond to.

        Returns
        -------
        matrix : ImmutableMatrix, shape(3,1)
            The matrix that gives the 1D vector.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.vector import ReferenceFrame
        >>> a, b, c = symbols('a, b, c')
        >>> N = ReferenceFrame('N')
        >>> vector = a * N.x + b * N.y + c * N.z
        >>> vector.to_matrix(N)
        Matrix([
        [a],
        [b],
        [c]])
        >>> beta = symbols('beta')
        >>> A = N.orientnew('A', 'Axis', (beta, N.x))
        >>> vector.to_matrix(A)
        Matrix([
        [                         a],
        [ b*cos(beta) + c*sin(beta)],
        [-b*sin(beta) + c*cos(beta)]])

        """

        return Matrix([self.dot(unit_vec) for unit_vec in
                       reference_frame]).reshape(3, 1)

    def doit(self, **hints):
        """Calls .doit() on each term in the Vector"""
        d = {}
        for v in self.args:
            d[v[1]] = v[0].applyfunc(lambda x: x.doit(**hints))
        return Vector(d)

    def dt(self, otherframe):
        """
        Returns a Vector which is the time derivative of
        the self Vector, taken in frame otherframe.

        Calls the global time_derivative method

        Parameters
        ==========

        otherframe : ReferenceFrame
            The frame to calculate the time derivative in

        """
        from sympy.physics.vector import time_derivative
        return time_derivative(self, otherframe)

    def simplify(self):
        """Returns a simplified Vector."""
        d = {}
        for v in self.args:
            d[v[1]] = simplify(v[0])
        return Vector(d)

    def subs(self, *args, **kwargs):
        """Substitution on the Vector.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> s = Symbol('s')
        >>> a = N.x * s
        >>> a.subs({s: 2})
        2*N.x

        """

        d = {}
        for v in self.args:
            d[v[1]] = v[0].subs(*args, **kwargs)
        return Vector(d)

    def magnitude(self):
        """Returns the magnitude (Euclidean norm) of self.

        Warnings
        ========

        Python ignores the leading negative sign so that might
        give wrong results.
        ``-A.x.magnitude()`` would be treated as ``-(A.x.magnitude())``,
        instead of ``(-A.x).magnitude()``.

        """
        return sqrt(self.dot(self))

    def normalize(self):
        """Returns a Vector of magnitude 1, codirectional with self."""
        return Vector(self.args + []) / self.magnitude()

    def applyfunc(self, f):
        """Apply a function to each component of a vector."""
        if not callable(f):
            raise TypeError("`f` must be callable.")

        d = {}
        for v in self.args:
            d[v[1]] = v[0].applyfunc(f)
        return Vector(d)

    def angle_between(self, vec):
        """
        Returns the smallest angle between Vector 'vec' and self.

        Parameter
        =========

        vec : Vector
            The Vector between which angle is needed.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame("A")
        >>> v1 = A.x
        >>> v2 = A.y
        >>> v1.angle_between(v2)
        pi/2

        >>> v3 = A.x + A.y + A.z
        >>> v1.angle_between(v3)
        acos(sqrt(3)/3)

        Warnings
        ========

        Python ignores the leading negative sign so that might give wrong
        results. ``-A.x.angle_between()`` would be treated as
        ``-(A.x.angle_between())``, instead of ``(-A.x).angle_between()``.

        """

        vec1 = self.normalize()
        vec2 = vec.normalize()
        angle = acos(vec1.dot(vec2))
        return angle

    def free_symbols(self, reference_frame):
        """Returns the free symbols in the measure numbers of the vector
        expressed in the given reference frame.

        Parameters
        ==========
        reference_frame : ReferenceFrame
            The frame with respect to which the free symbols of the given
            vector is to be determined.

        Returns
        =======
        set of Symbol
            set of symbols present in the measure numbers of
            ``reference_frame``.

        """

        return self.to_matrix(reference_frame).free_symbols

    def free_dynamicsymbols(self, reference_frame):
        """Returns the free dynamic symbols (functions of time ``t``) in the
        measure numbers of the vector expressed in the given reference frame.

        Parameters
        ==========
        reference_frame : ReferenceFrame
            The frame with respect to which the free dynamic symbols of the
            given vector is to be determined.

        Returns
        =======
        set
            Set of functions of time ``t``, e.g.
            ``Function('f')(me.dynamicsymbols._t)``.

        """
        # TODO : Circular dependency if imported at top. Should move
        # find_dynamicsymbols into physics.vector.functions.
        from sympy.physics.mechanics.functions import find_dynamicsymbols

        return find_dynamicsymbols(self, reference_frame=reference_frame)

    def _eval_evalf(self, prec):
        if not self.args:
            return self
        new_args = []
        dps = prec_to_dps(prec)
        for mat, frame in self.args:
            new_args.append([mat.evalf(n=dps), frame])
        return Vector(new_args)

    def xreplace(self, rule):
        """Replace occurrences of objects within the measure numbers of the
        vector.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule.

        Returns
        =======

        Vector
            Result of the replacement.

        Examples
        ========

        >>> from sympy import symbols, pi
        >>> from sympy.physics.vector import ReferenceFrame
        >>> A = ReferenceFrame('A')
        >>> x, y, z = symbols('x y z')
        >>> ((1 + x*y) * A.x).xreplace({x: pi})
        (pi*y + 1)*A.x
        >>> ((1 + x*y) * A.x).xreplace({x: pi, y: 2})
        (1 + 2*pi)*A.x

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> ((x*y + z) * A.x).xreplace({x*y: pi})
        (z + pi)*A.x
        >>> ((x*y*z) * A.x).xreplace({x*y: pi})
        x*y*z*A.x

        """

        new_args = []
        for mat, frame in self.args:
            mat = mat.xreplace(rule)
            new_args.append([mat, frame])
        return Vector(new_args)


class VectorTypeError(TypeError):

    def __init__(self, other, want):
        msg = filldedent("Expected an instance of %s, but received object "
                         "'%s' of %s." % (type(want), other, type(other)))
        super().__init__(msg)


def _check_vector(other):
    if not isinstance(other, Vector):
        raise TypeError('A Vector must be supplied')
    return other

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\accumulationbounds.py ===
from sympy.core import Add, Mul, Pow, S
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.numbers import _sympifyit, oo, zoo
from sympy.core.relational import is_le, is_lt, is_ge, is_gt
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.logic.boolalg import And
from sympy.multipledispatch import dispatch
from sympy.series.order import Order
from sympy.sets.sets import FiniteSet


class AccumulationBounds(Expr):
    r"""An accumulation bounds.

    # Note AccumulationBounds has an alias: AccumBounds

    AccumulationBounds represent an interval `[a, b]`, which is always closed
    at the ends. Here `a` and `b` can be any value from extended real numbers.

    The intended meaning of AccummulationBounds is to give an approximate
    location of the accumulation points of a real function at a limit point.

    Let `a` and `b` be reals such that `a \le b`.

    `\left\langle a, b\right\rangle = \{x \in \mathbb{R} \mid a \le x \le b\}`

    `\left\langle -\infty, b\right\rangle = \{x \in \mathbb{R} \mid x \le b\} \cup \{-\infty, \infty\}`

    `\left\langle a, \infty \right\rangle = \{x \in \mathbb{R} \mid a \le x\} \cup \{-\infty, \infty\}`

    `\left\langle -\infty, \infty \right\rangle = \mathbb{R} \cup \{-\infty, \infty\}`

    ``oo`` and ``-oo`` are added to the second and third definition respectively,
    since if either ``-oo`` or ``oo`` is an argument, then the other one should
    be included (though not as an end point). This is forced, since we have,
    for example, ``1/AccumBounds(0, 1) = AccumBounds(1, oo)``, and the limit at
    `0` is not one-sided. As `x` tends to `0-`, then `1/x \rightarrow -\infty`, so `-\infty`
    should be interpreted as belonging to ``AccumBounds(1, oo)`` though it need
    not appear explicitly.

    In many cases it suffices to know that the limit set is bounded.
    However, in some other cases more exact information could be useful.
    For example, all accumulation values of `\cos(x) + 1` are non-negative.
    (``AccumBounds(-1, 1) + 1 = AccumBounds(0, 2)``)

    A AccumulationBounds object is defined to be real AccumulationBounds,
    if its end points are finite reals.

    Let `X`, `Y` be real AccumulationBounds, then their sum, difference,
    product are defined to be the following sets:

    `X + Y = \{ x+y \mid x \in X \cap y \in Y\}`

    `X - Y = \{ x-y \mid x \in X \cap y \in Y\}`

    `X \times Y = \{ x \times y \mid x \in X \cap y \in Y\}`

    When an AccumBounds is raised to a negative power, if 0 is contained
    between the bounds then an infinite range is returned, otherwise if an
    endpoint is 0 then a semi-infinite range with consistent sign will be returned.

    AccumBounds in expressions behave a lot like Intervals but the
    semantics are not necessarily the same. Division (or exponentiation
    to a negative integer power) could be handled with *intervals* by
    returning a union of the results obtained after splitting the
    bounds between negatives and positives, but that is not done with
    AccumBounds. In addition, bounds are assumed to be independent of
    each other; if the same bound is used in more than one place in an
    expression, the result may not be the supremum or infimum of the
    expression (see below). Finally, when a boundary is ``1``,
    exponentiation to the power of ``oo`` yields ``oo``, neither
    ``1`` nor ``nan``.

    Examples
    ========

    >>> from sympy import AccumBounds, sin, exp, log, pi, E, S, oo
    >>> from sympy.abc import x

    >>> AccumBounds(0, 1) + AccumBounds(1, 2)
    AccumBounds(1, 3)

    >>> AccumBounds(0, 1) - AccumBounds(0, 2)
    AccumBounds(-2, 1)

    >>> AccumBounds(-2, 3)*AccumBounds(-1, 1)
    AccumBounds(-3, 3)

    >>> AccumBounds(1, 2)*AccumBounds(3, 5)
    AccumBounds(3, 10)

    The exponentiation of AccumulationBounds is defined
    as follows:

    If 0 does not belong to `X` or `n > 0` then

    `X^n = \{ x^n \mid x \in X\}`

    >>> AccumBounds(1, 4)**(S(1)/2)
    AccumBounds(1, 2)

    otherwise, an infinite or semi-infinite result is obtained:

    >>> 1/AccumBounds(-1, 1)
    AccumBounds(-oo, oo)
    >>> 1/AccumBounds(0, 2)
    AccumBounds(1/2, oo)
    >>> 1/AccumBounds(-oo, 0)
    AccumBounds(-oo, 0)

    A boundary of 1 will always generate all nonnegatives:

    >>> AccumBounds(1, 2)**oo
    AccumBounds(0, oo)
    >>> AccumBounds(0, 1)**oo
    AccumBounds(0, oo)

    If the exponent is itself an AccumulationBounds or is not an
    integer then unevaluated results will be returned unless the base
    values are positive:

    >>> AccumBounds(2, 3)**AccumBounds(-1, 2)
    AccumBounds(1/3, 9)
    >>> AccumBounds(-2, 3)**AccumBounds(-1, 2)
    AccumBounds(-2, 3)**AccumBounds(-1, 2)

    >>> AccumBounds(-2, -1)**(S(1)/2)
    sqrt(AccumBounds(-2, -1))

    Note: `\left\langle a, b\right\rangle^2` is not same as `\left\langle a, b\right\rangle \times \left\langle a, b\right\rangle`

    >>> AccumBounds(-1, 1)**2
    AccumBounds(0, 1)

    >>> AccumBounds(1, 3) < 4
    True

    >>> AccumBounds(1, 3) < -1
    False

    Some elementary functions can also take AccumulationBounds as input.
    A function `f` evaluated for some real AccumulationBounds `\left\langle a, b \right\rangle`
    is defined as `f(\left\langle a, b\right\rangle) = \{ f(x) \mid a \le x \le b \}`

    >>> sin(AccumBounds(pi/6, pi/3))
    AccumBounds(1/2, sqrt(3)/2)

    >>> exp(AccumBounds(0, 1))
    AccumBounds(1, E)

    >>> log(AccumBounds(1, E))
    AccumBounds(0, 1)

    Some symbol in an expression can be substituted for a AccumulationBounds
    object. But it does not necessarily evaluate the AccumulationBounds for
    that expression.

    The same expression can be evaluated to different values depending upon
    the form it is used for substitution since each instance of an
    AccumulationBounds is considered independent. For example:

    >>> (x**2 + 2*x + 1).subs(x, AccumBounds(-1, 1))
    AccumBounds(-1, 4)

    >>> ((x + 1)**2).subs(x, AccumBounds(-1, 1))
    AccumBounds(0, 4)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Interval_arithmetic

    .. [2] https://fab.cba.mit.edu/classes/S62.12/docs/Hickey_interval.pdf

    Notes
    =====

    Do not use ``AccumulationBounds`` for floating point interval arithmetic
    calculations, use ``mpmath.iv`` instead.
    """

    is_extended_real = True
    is_number = False

    def __new__(cls, min, max) -> Expr: # type: ignore

        min = _sympify(min)
        max = _sympify(max)

        # Only allow real intervals (use symbols with 'is_extended_real=True').
        if not min.is_extended_real or not max.is_extended_real:
            raise ValueError("Only real AccumulationBounds are supported")

        if max == min:
            return max

        # Make sure that the created AccumBounds object will be valid.
        if max.is_number and min.is_number:
            bad = max.is_comparable and min.is_comparable and max < min
        else:
            bad = (max - min).is_extended_negative
        if bad:
            raise ValueError(
                "Lower limit should be smaller than upper limit")

        return Basic.__new__(cls, min, max)

    # setting the operation priority
    _op_priority = 11.0

    def _eval_is_real(self):
        if self.min.is_real and self.max.is_real:
            return True

    @property
    def min(self):
        """
        Returns the minimum possible value attained by AccumulationBounds
        object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).min
        1

        """
        return self.args[0]

    @property
    def max(self):
        """
        Returns the maximum possible value attained by AccumulationBounds
        object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).max
        3

        """
        return self.args[1]

    @property
    def delta(self):
        """
        Returns the difference of maximum possible value attained by
        AccumulationBounds object and minimum possible value attained
        by AccumulationBounds object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).delta
        2

        """
        return self.max - self.min

    @property
    def mid(self):
        """
        Returns the mean of maximum possible value attained by
        AccumulationBounds object and minimum possible value
        attained by AccumulationBounds object.

        Examples
        ========

        >>> from sympy import AccumBounds
        >>> AccumBounds(1, 3).mid
        2

        """
        return (self.min + self.max) / 2

    @_sympifyit('other', NotImplemented)
    def _eval_power(self, other):
        return self.__pow__(other)

    @_sympifyit('other', NotImplemented)
    def __add__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                return AccumBounds(
                    Add(self.min, other.min),
                    Add(self.max, other.max))
            if other is S.Infinity and self.min is S.NegativeInfinity or \
                    other is S.NegativeInfinity and self.max is S.Infinity:
                return AccumBounds(-oo, oo)
            elif other.is_extended_real:
                if self.min is S.NegativeInfinity and self.max is S.Infinity:
                    return AccumBounds(-oo, oo)
                elif self.min is S.NegativeInfinity:
                    return AccumBounds(-oo, self.max + other)
                elif self.max is S.Infinity:
                    return AccumBounds(self.min + other, oo)
                else:
                    return AccumBounds(Add(self.min, other), Add(self.max, other))
            return Add(self, other, evaluate=False)
        return NotImplemented

    __radd__ = __add__

    def __neg__(self):
        return AccumBounds(-self.max, -self.min)

    @_sympifyit('other', NotImplemented)
    def __sub__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                return AccumBounds(
                    Add(self.min, -other.max),
                    Add(self.max, -other.min))
            if other is S.NegativeInfinity and self.min is S.NegativeInfinity or \
                    other is S.Infinity and self.max is S.Infinity:
                return AccumBounds(-oo, oo)
            elif other.is_extended_real:
                if self.min is S.NegativeInfinity and self.max is S.Infinity:
                    return AccumBounds(-oo, oo)
                elif self.min is S.NegativeInfinity:
                    return AccumBounds(-oo, self.max - other)
                elif self.max is S.Infinity:
                    return AccumBounds(self.min - other, oo)
                else:
                    return AccumBounds(
                        Add(self.min, -other),
                        Add(self.max, -other))
            return Add(self, -other, evaluate=False)
        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rsub__(self, other):
        return self.__neg__() + other

    @_sympifyit('other', NotImplemented)
    def __mul__(self, other):
        if self.args == (-oo, oo):
            return self
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                if other.args == (-oo, oo):
                    return other
                v = set()
                for a in self.args:
                    vi = other*a
                    v.update(vi.args or (vi,))
                return AccumBounds(Min(*v), Max(*v))
            if other is S.Infinity:
                if self.min.is_zero:
                    return AccumBounds(0, oo)
                if self.max.is_zero:
                    return AccumBounds(-oo, 0)
            if other is S.NegativeInfinity:
                if self.min.is_zero:
                    return AccumBounds(-oo, 0)
                if self.max.is_zero:
                    return AccumBounds(0, oo)
            if other.is_extended_real:
                if other.is_zero:
                    if self.max is S.Infinity:
                        return AccumBounds(0, oo)
                    if self.min is S.NegativeInfinity:
                        return AccumBounds(-oo, 0)
                    return S.Zero
                if other.is_extended_positive:
                    return AccumBounds(
                        Mul(self.min, other),
                        Mul(self.max, other))
                elif other.is_extended_negative:
                    return AccumBounds(
                        Mul(self.max, other),
                        Mul(self.min, other))
            if isinstance(other, Order):
                return other
            return Mul(self, other, evaluate=False)
        return NotImplemented

    __rmul__ = __mul__

    @_sympifyit('other', NotImplemented)
    def __truediv__(self, other):
        if isinstance(other, Expr):
            if isinstance(other, AccumBounds):
                if other.min.is_positive or other.max.is_negative:
                    return self * AccumBounds(1/other.max, 1/other.min)

                if (self.min.is_extended_nonpositive and self.max.is_extended_nonnegative and
                    other.min.is_extended_nonpositive and other.max.is_extended_nonnegative):
                    if self.min.is_zero and other.min.is_zero:
                        return AccumBounds(0, oo)
                    if self.max.is_zero and other.min.is_zero:
                        return AccumBounds(-oo, 0)
                    return AccumBounds(-oo, oo)

                if self.max.is_extended_negative:
                    if other.min.is_extended_negative:
                        if other.max.is_zero:
                            return AccumBounds(self.max / other.min, oo)
                        if other.max.is_extended_positive:
                            # if we were dealing with intervals we would return
                            # Union(Interval(-oo, self.max/other.max),
                            #       Interval(self.max/other.min, oo))
                            return AccumBounds(-oo, oo)

                    if other.min.is_zero and other.max.is_extended_positive:
                        return AccumBounds(-oo, self.max / other.max)

                if self.min.is_extended_positive:
                    if other.min.is_extended_negative:
                        if other.max.is_zero:
                            return AccumBounds(-oo, self.min / other.min)
                        if other.max.is_extended_positive:
                            # if we were dealing with intervals we would return
                            # Union(Interval(-oo, self.min/other.min),
                            #       Interval(self.min/other.max, oo))
                            return AccumBounds(-oo, oo)

                    if other.min.is_zero and other.max.is_extended_positive:
                        return AccumBounds(self.min / other.max, oo)

            elif other.is_extended_real:
                if other in (S.Infinity, S.NegativeInfinity):
                    if self == AccumBounds(-oo, oo):
                        return AccumBounds(-oo, oo)
                    if self.max is S.Infinity:
                        return AccumBounds(Min(0, other), Max(0, other))
                    if self.min is S.NegativeInfinity:
                        return AccumBounds(Min(0, -other), Max(0, -other))
                if other.is_extended_positive:
                    return AccumBounds(self.min / other, self.max / other)
                elif other.is_extended_negative:
                    return AccumBounds(self.max / other, self.min / other)
            if (1 / other) is S.ComplexInfinity:
                return Mul(self, 1 / other, evaluate=False)
            else:
                return Mul(self, 1 / other)

        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rtruediv__(self, other):
        if isinstance(other, Expr):
            if other.is_extended_real:
                if other.is_zero:
                    return S.Zero
                if (self.min.is_extended_nonpositive and self.max.is_extended_nonnegative):
                    if self.min.is_zero:
                        if other.is_extended_positive:
                            return AccumBounds(Mul(other, 1 / self.max), oo)
                        if other.is_extended_negative:
                            return AccumBounds(-oo, Mul(other, 1 / self.max))
                    if self.max.is_zero:
                        if other.is_extended_positive:
                            return AccumBounds(-oo, Mul(other, 1 / self.min))
                        if other.is_extended_negative:
                            return AccumBounds(Mul(other, 1 / self.min), oo)
                    return AccumBounds(-oo, oo)
                else:
                    return AccumBounds(Min(other / self.min, other / self.max),
                                       Max(other / self.min, other / self.max))
            return Mul(other, 1 / self, evaluate=False)
        else:
            return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __pow__(self, other):
        if isinstance(other, Expr):
            if other is S.Infinity:
                if self.min.is_extended_nonnegative:
                    if self.max < 1:
                        return S.Zero
                    if self.min > 1:
                        return S.Infinity
                    return AccumBounds(0, oo)
                elif self.max.is_extended_negative:
                    if self.min > -1:
                        return S.Zero
                    if self.max < -1:
                        return zoo
                    return S.NaN
                else:
                    if self.min > -1:
                        if self.max < 1:
                            return S.Zero
                        return AccumBounds(0, oo)
                    return AccumBounds(-oo, oo)

            if other is S.NegativeInfinity:
                return (1/self)**oo

            # generically true
            if (self.max - self.min).is_nonnegative:
                # well defined
                if self.min.is_nonnegative:
                    # no 0 to worry about
                    if other.is_nonnegative:
                        # no infinity to worry about
                        return self.func(self.min**other, self.max**other)

            if other.is_zero:
                return S.One  # x**0 = 1

            if other.is_Integer or other.is_integer:
                if self.min.is_extended_positive:
                    return AccumBounds(
                        Min(self.min**other, self.max**other),
                        Max(self.min**other, self.max**other))
                elif self.max.is_extended_negative:
                    return AccumBounds(
                        Min(self.max**other, self.min**other),
                        Max(self.max**other, self.min**other))

                if other % 2 == 0:
                    if other.is_extended_negative:
                        if self.min.is_zero:
                            return AccumBounds(self.max**other, oo)
                        if self.max.is_zero:
                            return AccumBounds(self.min**other, oo)
                        return (1/self)**(-other)
                    return AccumBounds(
                        S.Zero, Max(self.min**other, self.max**other))
                elif other % 2 == 1:
                    if other.is_extended_negative:
                        if self.min.is_zero:
                            return AccumBounds(self.max**other, oo)
                        if self.max.is_zero:
                            return AccumBounds(-oo, self.min**other)
                        return (1/self)**(-other)
                    return AccumBounds(self.min**other, self.max**other)

            # non-integer exponent
            # 0**neg or neg**frac yields complex
            if (other.is_number or other.is_rational) and (
                    self.min.is_extended_nonnegative or (
                    other.is_extended_nonnegative and
                    self.min.is_extended_nonnegative)):
                num, den = other.as_numer_denom()
                if num is S.One:
                    return AccumBounds(*[i**(1/den) for i in self.args])

                elif den is not S.One:  # e.g. if other is not Float
                    return (self**num)**(1/den)  # ok for non-negative base

            if isinstance(other, AccumBounds):
                if (self.min.is_extended_positive or
                        self.min.is_extended_nonnegative and
                        other.min.is_extended_nonnegative):
                    p = [self**i for i in other.args]
                    if not any(i.is_Pow for i in p):
                        a = [j for i in p for j in i.args or (i,)]
                        try:
                            return self.func(min(a), max(a))
                        except TypeError:  # can't sort
                            pass

            return Pow(self, other, evaluate=False)

        return NotImplemented

    @_sympifyit('other', NotImplemented)
    def __rpow__(self, other):
        if other.is_real and other.is_extended_nonnegative and (
                self.max - self.min).is_extended_positive:
            if other is S.One:
                return S.One
            if other.is_extended_positive:
                a, b = [other**i for i in self.args]
                if min(a, b) != a:
                    a, b = b, a
                return self.func(a, b)
            if other.is_zero:
                if self.min.is_zero:
                    return self.func(0, 1)
                if self.min.is_extended_positive:
                    return S.Zero

        return Pow(other, self, evaluate=False)

    def __abs__(self):
        if self.max.is_extended_negative:
            return self.__neg__()
        elif self.min.is_extended_negative:
            return AccumBounds(S.Zero, Max(abs(self.min), self.max))
        else:
            return self


    def __contains__(self, other):
        """
        Returns ``True`` if other is contained in self, where other
        belongs to extended real numbers, ``False`` if not contained,
        otherwise TypeError is raised.

        Examples
        ========

        >>> from sympy import AccumBounds, oo
        >>> 1 in AccumBounds(-1, 3)
        True

        -oo and oo go together as limits (in AccumulationBounds).

        >>> -oo in AccumBounds(1, oo)
        True

        >>> oo in AccumBounds(-oo, 0)
        True

        """
        other = _sympify(other)

        if other in (S.Infinity, S.NegativeInfinity):
            if self.min is S.NegativeInfinity or self.max is S.Infinity:
                return True
            return False

        rv = And(self.min <= other, self.max >= other)
        if rv not in (True, False):
            raise TypeError("input failed to evaluate")
        return rv

    def intersection(self, other):
        """
        Returns the intersection of 'self' and 'other'.
        Here other can be an instance of :py:class:`~.FiniteSet` or AccumulationBounds.

        Parameters
        ==========

        other : AccumulationBounds
            Another AccumulationBounds object with which the intersection
            has to be computed.

        Returns
        =======

        AccumulationBounds
            Intersection of ``self`` and ``other``.

        Examples
        ========

        >>> from sympy import AccumBounds, FiniteSet
        >>> AccumBounds(1, 3).intersection(AccumBounds(2, 4))
        AccumBounds(2, 3)

        >>> AccumBounds(1, 3).intersection(AccumBounds(4, 6))
        EmptySet

        >>> AccumBounds(1, 4).intersection(FiniteSet(1, 2, 5))
        {1, 2}

        """
        if not isinstance(other, (AccumBounds, FiniteSet)):
            raise TypeError(
                "Input must be AccumulationBounds or FiniteSet object")

        if isinstance(other, FiniteSet):
            fin_set = S.EmptySet
            for i in other:
                if i in self:
                    fin_set = fin_set + FiniteSet(i)
            return fin_set

        if self.max < other.min or self.min > other.max:
            return S.EmptySet

        if self.min <= other.min:
            if self.max <= other.max:
                return AccumBounds(other.min, self.max)
            if self.max > other.max:
                return other

        if other.min <= self.min:
            if other.max < self.max:
                return AccumBounds(self.min, other.max)
            if other.max > self.max:
                return self

    def union(self, other):
        # TODO : Devise a better method for Union of AccumBounds
        # this method is not actually correct and
        # can be made better
        if not isinstance(other, AccumBounds):
            raise TypeError(
                "Input must be AccumulationBounds or FiniteSet object")

        if self.min <= other.min and self.max >= other.min:
            return AccumBounds(self.min, Max(self.max, other.max))

        if other.min <= self.min and other.max >= self.min:
            return AccumBounds(other.min, Max(self.max, other.max))


@dispatch(AccumulationBounds, AccumulationBounds) # type: ignore # noqa:F811
def _eval_is_le(lhs, rhs): # noqa:F811
    if is_le(lhs.max, rhs.min):
        return True
    if is_gt(lhs.min, rhs.max):
        return False


@dispatch(AccumulationBounds, Basic) # type: ignore # noqa:F811
def _eval_is_le(lhs, rhs): # noqa: F811

    """
    Returns ``True `` if range of values attained by ``lhs`` AccumulationBounds
    object is greater than the range of values attained by ``rhs``,
    where ``rhs`` may be any value of type AccumulationBounds object or
    extended real number value, ``False`` if ``rhs`` satisfies
    the same property, else an unevaluated :py:class:`~.Relational`.

    Examples
    ========

    >>> from sympy import AccumBounds, oo
    >>> AccumBounds(1, 3) > AccumBounds(4, oo)
    False
    >>> AccumBounds(1, 4) > AccumBounds(3, 4)
    AccumBounds(1, 4) > AccumBounds(3, 4)
    >>> AccumBounds(1, oo) > -1
    True

    """
    if not rhs.is_extended_real:
            raise TypeError(
                "Invalid comparison of %s %s" %
                (type(rhs), rhs))
    elif rhs.is_comparable:
        if is_le(lhs.max, rhs):
            return True
        if is_gt(lhs.min, rhs):
            return False


@dispatch(AccumulationBounds, AccumulationBounds)
def _eval_is_ge(lhs, rhs): # noqa:F811
    if is_ge(lhs.min, rhs.max):
        return True
    if is_lt(lhs.max, rhs.min):
        return False


@dispatch(AccumulationBounds, Expr)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa: F811
    """
    Returns ``True`` if range of values attained by ``lhs`` AccumulationBounds
    object is less that the range of values attained by ``rhs``, where
    other may be any value of type AccumulationBounds object or extended
    real number value, ``False`` if ``rhs`` satisfies the same
    property, else an unevaluated :py:class:`~.Relational`.

    Examples
    ========

    >>> from sympy import AccumBounds, oo
    >>> AccumBounds(1, 3) >= AccumBounds(4, oo)
    False
    >>> AccumBounds(1, 4) >= AccumBounds(3, 4)
    AccumBounds(1, 4) >= AccumBounds(3, 4)
    >>> AccumBounds(1, oo) >= 1
    True
    """

    if not rhs.is_extended_real:
        raise TypeError(
            "Invalid comparison of %s %s" %
            (type(rhs), rhs))
    elif rhs.is_comparable:
        if is_ge(lhs.min, rhs):
            return True
        if is_lt(lhs.max, rhs):
            return False


@dispatch(Expr, AccumulationBounds)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811
    if not lhs.is_extended_real:
        raise TypeError(
            "Invalid comparison of %s %s" %
            (type(lhs), lhs))
    elif lhs.is_comparable:
        if is_le(rhs.max, lhs):
            return True
        if is_gt(rhs.min, lhs):
            return False


@dispatch(AccumulationBounds, AccumulationBounds)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811
    if is_ge(lhs.min, rhs.max):
        return True
    if is_lt(lhs.max, rhs.min):
        return False

# setting an alias for AccumulationBounds
AccumBounds = AccumulationBounds

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\rde.py ===
"""
Algorithms for solving the Risch differential equation.

Given a differential field K of characteristic 0 that is a simple
monomial extension of a base field k and f, g in K, the Risch
Differential Equation problem is to decide if there exist y in K such
that Dy + f*y == g and to find one if there are some.  If t is a
monomial over k and the coefficients of f and g are in k(t), then y is
in k(t), and the outline of the algorithm here is given as:

1. Compute the normal part n of the denominator of y.  The problem is
then reduced to finding y' in k<t>, where y == y'/n.
2. Compute the special part s of the denominator of y.   The problem is
then reduced to finding y'' in k[t], where y == y''/(n*s)
3. Bound the degree of y''.
4. Reduce the equation Dy + f*y == g to a similar equation with f, g in
k[t].
5. Find the solutions in k[t] of bounded degree of the reduced equation.

See Chapter 6 of "Symbolic Integration I: Transcendental Functions" by
Manuel Bronstein.  See also the docstring of risch.py.
"""

from operator import mul
from functools import reduce

from sympy.core import oo
from sympy.core.symbol import Dummy

from sympy.polys import Poly, gcd, ZZ, cancel

from sympy.functions.elementary.complexes import (im, re)
from sympy.functions.elementary.miscellaneous import sqrt

from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
    splitfactor, NonElementaryIntegralException, DecrementLevel, recognize_log_derivative)

# TODO: Add messages to NonElementaryIntegralException errors


def order_at(a, p, t):
    """
    Computes the order of a at p, with respect to t.

    Explanation
    ===========

    For a, p in k[t], the order of a at p is defined as nu_p(a) = max({n
    in Z+ such that p**n|a}), where a != 0.  If a == 0, nu_p(a) = +oo.

    To compute the order at a rational function, a/b, use the fact that
    nu_p(a/b) == nu_p(a) - nu_p(b).
    """
    if a.is_zero:
        return oo
    if p == Poly(t, t):
        return a.as_poly(t).ET()[0][0]

    # Uses binary search for calculating the power. power_list collects the tuples
    # (p^k,k) where each k is some power of 2. After deciding the largest k
    # such that k is power of 2 and p^k|a the loop iteratively calculates
    # the actual power.
    power_list = []
    p1 = p
    r = a.rem(p1)
    tracks_power = 1
    while r.is_zero:
        power_list.append((p1,tracks_power))
        p1 = p1*p1
        tracks_power *= 2
        r = a.rem(p1)
    n = 0
    product = Poly(1, t)
    while len(power_list) != 0:
        final = power_list.pop()
        productf = product*final[0]
        r = a.rem(productf)
        if r.is_zero:
            n += final[1]
            product = productf
    return n


def order_at_oo(a, d, t):
    """
    Computes the order of a/d at oo (infinity), with respect to t.

    For f in k(t), the order or f at oo is defined as deg(d) - deg(a), where
    f == a/d.
    """
    if a.is_zero:
        return oo
    return d.degree(t) - a.degree(t)


def weak_normalizer(a, d, DE, z=None):
    """
    Weak normalization.

    Explanation
    ===========

    Given a derivation D on k[t] and f == a/d in k(t), return q in k[t]
    such that f - Dq/q is weakly normalized with respect to t.

    f in k(t) is said to be "weakly normalized" with respect to t if
    residue_p(f) is not a positive integer for any normal irreducible p
    in k[t] such that f is in R_p (Definition 6.1.1).  If f has an
    elementary integral, this is equivalent to no logarithm of
    integral(f) whose argument depends on t has a positive integer
    coefficient, where the arguments of the logarithms not in k(t) are
    in k[t].

    Returns (q, f - Dq/q)
    """
    z = z or Dummy('z')
    dn, ds = splitfactor(d, DE)

    # Compute d1, where dn == d1*d2**2*...*dn**n is a square-free
    # factorization of d.
    g = gcd(dn, dn.diff(DE.t))
    d_sqf_part = dn.quo(g)
    d1 = d_sqf_part.quo(gcd(d_sqf_part, g))

    a1, b = gcdex_diophantine(d.quo(d1).as_poly(DE.t), d1.as_poly(DE.t),
        a.as_poly(DE.t))
    r = (a - Poly(z, DE.t)*derivation(d1, DE)).as_poly(DE.t).resultant(
        d1.as_poly(DE.t))
    r = Poly(r, z)

    if not r.expr.has(z):
        return (Poly(1, DE.t), (a, d))

    N = [i for i in r.real_roots() if i in ZZ and i > 0]

    q = reduce(mul, [gcd(a - Poly(n, DE.t)*derivation(d1, DE), d1) for n in N],
        Poly(1, DE.t))

    dq = derivation(q, DE)
    sn = q*a - d*dq
    sd = q*d
    sn, sd = sn.cancel(sd, include=True)

    return (q, (sn, sd))


def normal_denom(fa, fd, ga, gd, DE):
    """
    Normal part of the denominator.

    Explanation
    ===========

    Given a derivation D on k[t] and f, g in k(t) with f weakly
    normalized with respect to t, either raise NonElementaryIntegralException,
    in which case the equation Dy + f*y == g has no solution in k(t), or the
    quadruplet (a, b, c, h) such that a, h in k[t], b, c in k<t>, and for any
    solution y in k(t) of Dy + f*y == g, q = y*h in k<t> satisfies
    a*Dq + b*q == c.

    This constitutes step 1 in the outline given in the rde.py docstring.
    """
    dn, ds = splitfactor(fd, DE)
    en, es = splitfactor(gd, DE)

    p = dn.gcd(en)
    h = en.gcd(en.diff(DE.t)).quo(p.gcd(p.diff(DE.t)))

    a = dn*h
    c = a*h
    if c.div(en)[1]:
        # en does not divide dn*h**2
        raise NonElementaryIntegralException
    ca = c*ga
    ca, cd = ca.cancel(gd, include=True)

    ba = a*fa - dn*derivation(h, DE)*fd
    ba, bd = ba.cancel(fd, include=True)

    # (dn*h, dn*h*f - dn*Dh, dn*h**2*g, h)
    return (a, (ba, bd), (ca, cd), h)


def special_denom(a, ba, bd, ca, cd, DE, case='auto'):
    """
    Special part of the denominator.

    Explanation
    ===========

    case is one of {'exp', 'tan', 'primitive'} for the hyperexponential,
    hypertangent, and primitive cases, respectively.  For the
    hyperexponential (resp. hypertangent) case, given a derivation D on
    k[t] and a in k[t], b, c, in k<t> with Dt/t in k (resp. Dt/(t**2 + 1) in
    k, sqrt(-1) not in k), a != 0, and gcd(a, t) == 1 (resp.
    gcd(a, t**2 + 1) == 1), return the quadruplet (A, B, C, 1/h) such that
    A, B, C, h in k[t] and for any solution q in k<t> of a*Dq + b*q == c,
    r = qh in k[t] satisfies A*Dr + B*r == C.

    For ``case == 'primitive'``, k<t> == k[t], so it returns (a, b, c, 1) in
    this case.

    This constitutes step 2 of the outline given in the rde.py docstring.
    """
    # TODO: finish writing this and write tests

    if case == 'auto':
        case = DE.case

    if case == 'exp':
        p = Poly(DE.t, DE.t)
    elif case == 'tan':
        p = Poly(DE.t**2 + 1, DE.t)
    elif case in ('primitive', 'base'):
        B = ba.to_field().quo(bd)
        C = ca.to_field().quo(cd)
        return (a, B, C, Poly(1, DE.t))
    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'base'}, not %s." % case)

    nb = order_at(ba, p, DE.t) - order_at(bd, p, DE.t)
    nc = order_at(ca, p, DE.t) - order_at(cd, p, DE.t)

    n = min(0, nc - min(0, nb))
    if not nb:
        # Possible cancellation.
        from .prde import parametric_log_deriv
        if case == 'exp':
            dcoeff = DE.d.quo(Poly(DE.t, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                alphaa, alphad = frac_in(-ba.eval(0)/bd.eval(0)/a.eval(0), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    Q, m, z = A
                    if Q == 1:
                        n = min(n, m)

        elif case == 'tan':
            dcoeff = DE.d.quo(Poly(DE.t**2+1, DE.t))
            with DecrementLevel(DE):  # We are guaranteed to not have problems,
                                      # because case != 'base'.
                alphaa, alphad = frac_in(im(-ba.eval(sqrt(-1))/bd.eval(sqrt(-1))/a.eval(sqrt(-1))), DE.t)
                betaa, betad = frac_in(re(-ba.eval(sqrt(-1))/bd.eval(sqrt(-1))/a.eval(sqrt(-1))), DE.t)
                etaa, etad = frac_in(dcoeff, DE.t)

                if recognize_log_derivative(Poly(2, DE.t)*betaa, betad, DE):
                    A = parametric_log_deriv(alphaa*Poly(sqrt(-1), DE.t)*betad+alphad*betaa, alphad*betad, etaa, etad, DE)
                    if A is not None:
                        Q, m, z = A
                        if Q == 1:
                            n = min(n, m)
    N = max(0, -nb, n - nc)
    pN = p**N
    pn = p**-n

    A = a*pN
    B = ba*pN.quo(bd) + Poly(n, DE.t)*a*derivation(p, DE).quo(p)*pN
    C = (ca*pN*pn).quo(cd)
    h = pn

    # (a*p**N, (b + n*a*Dp/p)*p**N, c*p**(N - n), p**-n)
    return (A, B, C, h)


def bound_degree(a, b, cQ, DE, case='auto', parametric=False):
    """
    Bound on polynomial solutions.

    Explanation
    ===========

    Given a derivation D on k[t] and ``a``, ``b``, ``c`` in k[t] with ``a != 0``, return
    n in ZZ such that deg(q) <= n for any solution q in k[t] of
    a*Dq + b*q == c, when parametric=False, or deg(q) <= n for any solution
    c1, ..., cm in Const(k) and q in k[t] of a*Dq + b*q == Sum(ci*gi, (i, 1, m))
    when parametric=True.

    For ``parametric=False``, ``cQ`` is ``c``, a ``Poly``; for ``parametric=True``, ``cQ`` is Q ==
    [q1, ..., qm], a list of Polys.

    This constitutes step 3 of the outline given in the rde.py docstring.
    """
    # TODO: finish writing this and write tests

    if case == 'auto':
        case = DE.case

    da = a.degree(DE.t)
    db = b.degree(DE.t)

    # The parametric and regular cases are identical, except for this part
    if parametric:
        dc = max(i.degree(DE.t) for i in cQ)
    else:
        dc = cQ.degree(DE.t)

    alpha = cancel(-b.as_poly(DE.t).LC().as_expr()/
        a.as_poly(DE.t).LC().as_expr())

    if case == 'base':
        n = max(0, dc - max(db, da - 1))
        if db == da - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)

    elif case == 'primitive':
        if db > da:
            n = max(0, dc - db)
        else:
            n = max(0, dc - da + 1)

        etaa, etad = frac_in(DE.d, DE.T[DE.level - 1])

        t1 = DE.t
        with DecrementLevel(DE):
            alphaa, alphad = frac_in(alpha, DE.t)
            if db == da - 1:
                from .prde import limited_integrate
                # if alpha == m*Dt + Dz for z in k and m in ZZ:
                try:
                    (za, zd), m = limited_integrate(alphaa, alphad, [(etaa, etad)],
                        DE)
                except NonElementaryIntegralException:
                    pass
                else:
                    if len(m) != 1:
                        raise ValueError("Length of m should be 1")
                    n = max(n, m[0])

            elif db == da:
                # if alpha == Dz/z for z in k*:
                    # beta = -lc(a*Dz + b*z)/(z*lc(a))
                    # if beta == m*Dt + Dw for w in k and m in ZZ:
                        # n = max(n, m)
                from .prde import is_log_deriv_k_t_radical_in_field
                A = is_log_deriv_k_t_radical_in_field(alphaa, alphad, DE)
                if A is not None:
                    aa, z = A
                    if aa == 1:
                        beta = -(a*derivation(z, DE).as_poly(t1) +
                            b*z.as_poly(t1)).LC()/(z.as_expr()*a.LC())
                        betaa, betad = frac_in(beta, DE.t)
                        from .prde import limited_integrate
                        try:
                            (za, zd), m = limited_integrate(betaa, betad,
                                [(etaa, etad)], DE)
                        except NonElementaryIntegralException:
                            pass
                        else:
                            if len(m) != 1:
                                raise ValueError("Length of m should be 1")
                            n = max(n, m[0].as_expr())

    elif case == 'exp':
        from .prde import parametric_log_deriv

        n = max(0, dc - max(db, da))
        if da == db:
            etaa, etad = frac_in(DE.d.quo(Poly(DE.t, DE.t)), DE.T[DE.level - 1])
            with DecrementLevel(DE):
                alphaa, alphad = frac_in(alpha, DE.t)
                A = parametric_log_deriv(alphaa, alphad, etaa, etad, DE)
                if A is not None:
                    # if alpha == m*Dt/t + Dz/z for z in k* and m in ZZ:
                        # n = max(n, m)
                    a, m, z = A
                    if a == 1:
                        n = max(n, m)

    elif case in ('tan', 'other_nonlinear'):
        delta = DE.d.degree(DE.t)
        lam = DE.d.LC()
        alpha = cancel(alpha/lam)
        n = max(0, dc - max(da + delta - 1, db))
        if db == da + delta - 1 and alpha.is_Integer:
            n = max(0, alpha, dc - db)

    else:
        raise ValueError("case must be one of {'exp', 'tan', 'primitive', "
            "'other_nonlinear', 'base'}, not %s." % case)

    return n


def spde(a, b, c, n, DE):
    """
    Rothstein's Special Polynomial Differential Equation algorithm.

    Explanation
    ===========

    Given a derivation D on k[t], an integer n and ``a``,``b``,``c`` in k[t] with
    ``a != 0``, either raise NonElementaryIntegralException, in which case the
    equation a*Dq + b*q == c has no solution of degree at most ``n`` in
    k[t], or return the tuple (B, C, m, alpha, beta) such that B, C,
    alpha, beta in k[t], m in ZZ, and any solution q in k[t] of degree
    at most n of a*Dq + b*q == c must be of the form
    q == alpha*h + beta, where h in k[t], deg(h) <= m, and Dh + B*h == C.

    This constitutes step 4 of the outline given in the rde.py docstring.
    """
    zero = Poly(0, DE.t)

    alpha = Poly(1, DE.t)
    beta = Poly(0, DE.t)

    while True:
        if c.is_zero:
            return (zero, zero, 0, zero, beta)  # -1 is more to the point
        if (n < 0) is True:
            raise NonElementaryIntegralException

        g = a.gcd(b)
        if not c.rem(g).is_zero:  # g does not divide c
            raise NonElementaryIntegralException

        a, b, c = a.quo(g), b.quo(g), c.quo(g)

        if a.degree(DE.t) == 0:
            b = b.to_field().quo(a)
            c = c.to_field().quo(a)
            return (b, c, n, alpha, beta)

        r, z = gcdex_diophantine(b, a, c)
        b += derivation(a, DE)
        c = z - derivation(r, DE)
        n -= a.degree(DE.t)

        beta += alpha * r
        alpha *= a

def no_cancel_b_large(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) large enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with ``b != 0`` and either D == d/dt or
    deg(b) > max(0, deg(D) - 1), either raise NonElementaryIntegralException, in
    which case the equation ``Dq + b*q == c`` has no solution of degree at
    most n in k[t], or a solution q in k[t] of this equation with
    ``deg(q) < n``.
    """
    q = Poly(0, DE.t)

    while not c.is_zero:
        m = c.degree(DE.t) - b.degree(DE.t)
        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        p = Poly(c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC()*DE.t**m, DE.t,
            expand=False)
        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


def no_cancel_b_small(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) small enough.

    Explanation
    ===========

    Given a derivation D on k[t], ``n`` either an integer or +oo, and ``b``,``c``
    in k[t] with deg(b) < deg(D) - 1 and either D == d/dt or
    deg(D) >= 2, either raise NonElementaryIntegralException, in which case the
    equation Dq + b*q == c has no solution of degree at most n in k[t],
    or a solution q in k[t] of this equation with deg(q) <= n, or the
    tuple (h, b0, c0) such that h in k[t], b0, c0, in k, and for any
    solution q in k[t] of degree at most n of Dq + bq == c, y == q - h
    is a solution in k of Dy + b0*y == c0.
    """
    q = Poly(0, DE.t)

    while not c.is_zero:
        if n == 0:
            m = 0
        else:
            m = c.degree(DE.t) - DE.d.degree(DE.t) + 1

        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        if m > 0:
            p = Poly(c.as_poly(DE.t).LC()/(m*DE.d.as_poly(DE.t).LC())*DE.t**m,
                DE.t, expand=False)
        else:
            if b.degree(DE.t) != c.degree(DE.t):
                raise NonElementaryIntegralException
            if b.degree(DE.t) == 0:
                return (q, b.as_poly(DE.T[DE.level - 1]),
                    c.as_poly(DE.T[DE.level - 1]))
            p = Poly(c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC(), DE.t,
                expand=False)

        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


# TODO: better name for this function
def no_cancel_equal(b, c, n, DE):
    """
    Poly Risch Differential Equation - No cancellation: deg(b) == deg(D) - 1

    Explanation
    ===========

    Given a derivation D on k[t] with deg(D) >= 2, n either an integer
    or +oo, and b, c in k[t] with deg(b) == deg(D) - 1, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c has
    no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n, or the tuple (h, m, C) such that h
    in k[t], m in ZZ, and C in k[t], and for any solution q in k[t] of
    degree at most n of Dq + b*q == c, y == q - h is a solution in k[t]
    of degree at most m of Dy + b*y == C.
    """
    q = Poly(0, DE.t)
    lc = cancel(-b.as_poly(DE.t).LC()/DE.d.as_poly(DE.t).LC())
    if lc.is_Integer and lc.is_positive:
        M = lc
    else:
        M = -1

    while not c.is_zero:
        m = max(M, c.degree(DE.t) - DE.d.degree(DE.t) + 1)

        if not 0 <= m <= n:  # n < 0 or m < 0 or m > n
            raise NonElementaryIntegralException

        u = cancel(m*DE.d.as_poly(DE.t).LC() + b.as_poly(DE.t).LC())
        if u.is_zero:
            return (q, m, c)
        if m > 0:
            p = Poly(c.as_poly(DE.t).LC()/u*DE.t**m, DE.t, expand=False)
        else:
            if c.degree(DE.t) != DE.d.degree(DE.t) - 1:
                raise NonElementaryIntegralException
            else:
                p = c.as_poly(DE.t).LC()/b.as_poly(DE.t).LC()

        q = q + p
        n = m - 1
        c = c - derivation(p, DE) - b*p

    return q


def cancel_primitive(b, c, n, DE):
    """
    Poly Risch Differential Equation - Cancellation: Primitive case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.
    """
    # Delayed imports
    from .prde import is_log_deriv_k_t_radical_in_field
    with DecrementLevel(DE):
        ba, bd = frac_in(b, DE.t)
        A = is_log_deriv_k_t_radical_in_field(ba, bd, DE)
        if A is not None:
            n, z = A
            if n == 1:  # b == Dz/z
                raise NotImplementedError("is_deriv_in_field() is required to "
                    " solve this problem.")
                # if z*c == Dp for p in k[t] and deg(p) <= n:
                #     return p/z
                # else:
                #     raise NonElementaryIntegralException

    if c.is_zero:
        return c  # return 0

    if n < c.degree(DE.t):
        raise NonElementaryIntegralException

    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException
        with DecrementLevel(DE):
            a2a, a2d = frac_in(c.LC(), DE.t)
            sa, sd = rischDE(ba, bd, a2a, a2d, DE)
        stm = Poly(sa.as_expr()/sd.as_expr()*DE.t**m, DE.t, expand=False)
        q += stm
        n = m - 1
        c -= b*stm + derivation(stm, DE)

    return q


def cancel_exp(b, c, n, DE):
    """
    Poly Risch Differential Equation - Cancellation: Hyperexponential case.

    Explanation
    ===========

    Given a derivation D on k[t], n either an integer or +oo, ``b`` in k, and
    ``c`` in k[t] with Dt/t in k and ``b != 0``, either raise
    NonElementaryIntegralException, in which case the equation Dq + b*q == c
    has no solution of degree at most n in k[t], or a solution q in k[t] of
    this equation with deg(q) <= n.
    """
    from .prde import parametric_log_deriv
    eta = DE.d.quo(Poly(DE.t, DE.t)).as_expr()

    with DecrementLevel(DE):
        etaa, etad = frac_in(eta, DE.t)
        ba, bd = frac_in(b, DE.t)
        A = parametric_log_deriv(ba, bd, etaa, etad, DE)
        if A is not None:
            a, m, z = A
            if a == 1:
                raise NotImplementedError("is_deriv_in_field() is required to "
                    "solve this problem.")
                # if c*z*t**m == Dp for p in k<t> and q = p/(z*t**m) in k[t] and
                # deg(q) <= n:
                #     return q
                # else:
                #     raise NonElementaryIntegralException

    if c.is_zero:
        return c  # return 0

    if n < c.degree(DE.t):
        raise NonElementaryIntegralException

    q = Poly(0, DE.t)
    while not c.is_zero:
        m = c.degree(DE.t)
        if n < m:
            raise NonElementaryIntegralException
        # a1 = b + m*Dt/t
        a1 = b.as_expr()
        with DecrementLevel(DE):
            # TODO: Write a dummy function that does this idiom
            a1a, a1d = frac_in(a1, DE.t)
            a1a = a1a*etad + etaa*a1d*Poly(m, DE.t)
            a1d = a1d*etad

            a2a, a2d = frac_in(c.LC(), DE.t)

            sa, sd = rischDE(a1a, a1d, a2a, a2d, DE)
        stm = Poly(sa.as_expr()/sd.as_expr()*DE.t**m, DE.t, expand=False)
        q += stm
        n = m - 1
        c -= b*stm + derivation(stm, DE)  # deg(c) becomes smaller
    return q


def solve_poly_rde(b, cQ, n, DE, parametric=False):
    """
    Solve a Polynomial Risch Differential Equation with degree bound ``n``.

    This constitutes step 4 of the outline given in the rde.py docstring.

    For parametric=False, cQ is c, a Poly; for parametric=True, cQ is Q ==
    [q1, ..., qm], a list of Polys.
    """
    # No cancellation
    if not b.is_zero and (DE.case == 'base' or
            b.degree(DE.t) > max(0, DE.d.degree(DE.t) - 1)):

        if parametric:
            # Delayed imports
            from .prde import prde_no_cancel_b_large
            return prde_no_cancel_b_large(b, cQ, n, DE)
        return no_cancel_b_large(b, cQ, n, DE)

    elif (b.is_zero or b.degree(DE.t) < DE.d.degree(DE.t) - 1) and \
            (DE.case == 'base' or DE.d.degree(DE.t) >= 2):

        if parametric:
            from .prde import prde_no_cancel_b_small
            return prde_no_cancel_b_small(b, cQ, n, DE)

        R = no_cancel_b_small(b, cQ, n, DE)

        if isinstance(R, Poly):
            return R
        else:
            # XXX: Might k be a field? (pg. 209)
            h, b0, c0 = R
            with DecrementLevel(DE):
                b0, c0 = b0.as_poly(DE.t), c0.as_poly(DE.t)
                if b0 is None:  # See above comment
                    raise ValueError("b0 should be a non-Null value")
                if c0 is  None:
                    raise ValueError("c0 should be a non-Null value")
                y = solve_poly_rde(b0, c0, n, DE).as_poly(DE.t)
            return h + y

    elif DE.d.degree(DE.t) >= 2 and b.degree(DE.t) == DE.d.degree(DE.t) - 1 and \
            n > -b.as_poly(DE.t).LC()/DE.d.as_poly(DE.t).LC():

        # TODO: Is this check necessary, and if so, what should it do if it fails?
        # b comes from the first element returned from spde()
        if not b.as_poly(DE.t).LC().is_number:
            raise TypeError("Result should be a number")

        if parametric:
            raise NotImplementedError("prde_no_cancel_b_equal() is not yet "
                "implemented.")

        R = no_cancel_equal(b, cQ, n, DE)

        if isinstance(R, Poly):
            return R
        else:
            h, m, C = R
            # XXX: Or should it be rischDE()?
            y = solve_poly_rde(b, C, m, DE)
            return h + y

    else:
        # Cancellation
        if b.is_zero:
            raise NotImplementedError("Remaining cases for Poly (P)RDE are "
            "not yet implemented (is_deriv_in_field() required).")
        else:
            if DE.case == 'exp':
                if parametric:
                    raise NotImplementedError("Parametric RDE cancellation "
                        "hyperexponential case is not yet implemented.")
                return cancel_exp(b, cQ, n, DE)

            elif DE.case == 'primitive':
                if parametric:
                    raise NotImplementedError("Parametric RDE cancellation "
                        "primitive case is not yet implemented.")
                return cancel_primitive(b, cQ, n, DE)

            else:
                raise NotImplementedError("Other Poly (P)RDE cancellation "
                    "cases are not yet implemented (%s)." % DE.case)

        if parametric:
            raise NotImplementedError("Remaining cases for Poly PRDE not yet "
                "implemented.")
        raise NotImplementedError("Remaining cases for Poly RDE not yet "
            "implemented.")


def rischDE(fa, fd, ga, gd, DE):
    """
    Solve a Risch Differential Equation: Dy + f*y == g.

    Explanation
    ===========

    See the outline in the docstring of rde.py for more information
    about the procedure used.  Either raise NonElementaryIntegralException, in
    which case there is no solution y in the given differential field,
    or return y in k(t) satisfying Dy + f*y == g, or raise
    NotImplementedError, in which case, the algorithms necessary to
    solve the given Risch Differential Equation have not yet been
    implemented.
    """
    _, (fa, fd) = weak_normalizer(fa, fd, DE)
    a, (ba, bd), (ca, cd), hn = normal_denom(fa, fd, ga, gd, DE)
    A, B, C, hs = special_denom(a, ba, bd, ca, cd, DE)
    try:
        # Until this is fully implemented, use oo.  Note that this will almost
        # certainly cause non-termination in spde() (unless A == 1), and
        # *might* lead to non-termination in the next step for a nonelementary
        # integral (I don't know for certain yet).  Fortunately, spde() is
        # currently written recursively, so this will just give
        # RuntimeError: maximum recursion depth exceeded.
        n = bound_degree(A, B, C, DE)
    except NotImplementedError:
        # Useful for debugging:
        # import warnings
        # warnings.warn("rischDE: Proceeding with n = oo; may cause "
        #     "non-termination.")
        n = oo

    B, C, m, alpha, beta = spde(A, B, C, n, DE)
    if C.is_zero:
        y = C
    else:
        y = solve_poly_rde(B, C, m, DE)

    return (alpha*y + beta, hn*hs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\format.py ===
from ._format_impl import (  # noqa: F401
    ARRAY_ALIGN,
    BUFFER_SIZE,
    EXPECTED_KEYS,
    GROWTH_AXIS_MAX_DIGITS,
    MAGIC_LEN,
    MAGIC_PREFIX,
    __all__,
    __doc__,
    descr_to_dtype,
    drop_metadata,
    dtype_to_descr,
    header_data_from_array_1_0,
    isfileobj,
    magic,
    open_memmap,
    read_array,
    read_array_header_1_0,
    read_array_header_2_0,
    read_magic,
    write_array,
    write_array_header_1_0,
    write_array_header_2_0,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\style.py ===
import sys
from functools import lru_cache
from marshal import dumps, loads
from random import randint
from typing import Any, Dict, Iterable, List, Optional, Type, Union, cast

from . import errors
from .color import Color, ColorParseError, ColorSystem, blend_rgb
from .repr import Result, rich_repr
from .terminal_theme import DEFAULT_TERMINAL_THEME, TerminalTheme

# Style instances and style definitions are often interchangeable
StyleType = Union[str, "Style"]


class _Bit:
    """A descriptor to get/set a style attribute bit."""

    __slots__ = ["bit"]

    def __init__(self, bit_no: int) -> None:
        self.bit = 1 << bit_no

    def __get__(self, obj: "Style", objtype: Type["Style"]) -> Optional[bool]:
        if obj._set_attributes & self.bit:
            return obj._attributes & self.bit != 0
        return None


@rich_repr
class Style:
    """A terminal style.

    A terminal style consists of a color (`color`), a background color (`bgcolor`), and a number of attributes, such
    as bold, italic etc. The attributes have 3 states: they can either be on
    (``True``), off (``False``), or not set (``None``).

    Args:
        color (Union[Color, str], optional): Color of terminal text. Defaults to None.
        bgcolor (Union[Color, str], optional): Color of terminal background. Defaults to None.
        bold (bool, optional): Enable bold text. Defaults to None.
        dim (bool, optional): Enable dim text. Defaults to None.
        italic (bool, optional): Enable italic text. Defaults to None.
        underline (bool, optional): Enable underlined text. Defaults to None.
        blink (bool, optional): Enabled blinking text. Defaults to None.
        blink2 (bool, optional): Enable fast blinking text. Defaults to None.
        reverse (bool, optional): Enabled reverse text. Defaults to None.
        conceal (bool, optional): Enable concealed text. Defaults to None.
        strike (bool, optional): Enable strikethrough text. Defaults to None.
        underline2 (bool, optional): Enable doubly underlined text. Defaults to None.
        frame (bool, optional): Enable framed text. Defaults to None.
        encircle (bool, optional): Enable encircled text. Defaults to None.
        overline (bool, optional): Enable overlined text. Defaults to None.
        link (str, link): Link URL. Defaults to None.

    """

    _color: Optional[Color]
    _bgcolor: Optional[Color]
    _attributes: int
    _set_attributes: int
    _hash: Optional[int]
    _null: bool
    _meta: Optional[bytes]

    __slots__ = [
        "_color",
        "_bgcolor",
        "_attributes",
        "_set_attributes",
        "_link",
        "_link_id",
        "_ansi",
        "_style_definition",
        "_hash",
        "_null",
        "_meta",
    ]

    # maps bits on to SGR parameter
    _style_map = {
        0: "1",
        1: "2",
        2: "3",
        3: "4",
        4: "5",
        5: "6",
        6: "7",
        7: "8",
        8: "9",
        9: "21",
        10: "51",
        11: "52",
        12: "53",
    }

    STYLE_ATTRIBUTES = {
        "dim": "dim",
        "d": "dim",
        "bold": "bold",
        "b": "bold",
        "italic": "italic",
        "i": "italic",
        "underline": "underline",
        "u": "underline",
        "blink": "blink",
        "blink2": "blink2",
        "reverse": "reverse",
        "r": "reverse",
        "conceal": "conceal",
        "c": "conceal",
        "strike": "strike",
        "s": "strike",
        "underline2": "underline2",
        "uu": "underline2",
        "frame": "frame",
        "encircle": "encircle",
        "overline": "overline",
        "o": "overline",
    }

    def __init__(
        self,
        *,
        color: Optional[Union[Color, str]] = None,
        bgcolor: Optional[Union[Color, str]] = None,
        bold: Optional[bool] = None,
        dim: Optional[bool] = None,
        italic: Optional[bool] = None,
        underline: Optional[bool] = None,
        blink: Optional[bool] = None,
        blink2: Optional[bool] = None,
        reverse: Optional[bool] = None,
        conceal: Optional[bool] = None,
        strike: Optional[bool] = None,
        underline2: Optional[bool] = None,
        frame: Optional[bool] = None,
        encircle: Optional[bool] = None,
        overline: Optional[bool] = None,
        link: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ):
        self._ansi: Optional[str] = None
        self._style_definition: Optional[str] = None

        def _make_color(color: Union[Color, str]) -> Color:
            return color if isinstance(color, Color) else Color.parse(color)

        self._color = None if color is None else _make_color(color)
        self._bgcolor = None if bgcolor is None else _make_color(bgcolor)
        self._set_attributes = sum(
            (
                bold is not None,
                dim is not None and 2,
                italic is not None and 4,
                underline is not None and 8,
                blink is not None and 16,
                blink2 is not None and 32,
                reverse is not None and 64,
                conceal is not None and 128,
                strike is not None and 256,
                underline2 is not None and 512,
                frame is not None and 1024,
                encircle is not None and 2048,
                overline is not None and 4096,
            )
        )
        self._attributes = (
            sum(
                (
                    bold and 1 or 0,
                    dim and 2 or 0,
                    italic and 4 or 0,
                    underline and 8 or 0,
                    blink and 16 or 0,
                    blink2 and 32 or 0,
                    reverse and 64 or 0,
                    conceal and 128 or 0,
                    strike and 256 or 0,
                    underline2 and 512 or 0,
                    frame and 1024 or 0,
                    encircle and 2048 or 0,
                    overline and 4096 or 0,
                )
            )
            if self._set_attributes
            else 0
        )

        self._link = link
        self._meta = None if meta is None else dumps(meta)
        self._link_id = (
            f"{randint(0, 999999)}{hash(self._meta)}" if (link or meta) else ""
        )
        self._hash: Optional[int] = None
        self._null = not (self._set_attributes or color or bgcolor or link or meta)

    @classmethod
    def null(cls) -> "Style":
        """Create an 'null' style, equivalent to Style(), but more performant."""
        return NULL_STYLE

    @classmethod
    def from_color(
        cls, color: Optional[Color] = None, bgcolor: Optional[Color] = None
    ) -> "Style":
        """Create a new style with colors and no attributes.

        Returns:
            color (Optional[Color]): A (foreground) color, or None for no color. Defaults to None.
            bgcolor (Optional[Color]): A (background) color, or None for no color. Defaults to None.
        """
        style: Style = cls.__new__(Style)
        style._ansi = None
        style._style_definition = None
        style._color = color
        style._bgcolor = bgcolor
        style._set_attributes = 0
        style._attributes = 0
        style._link = None
        style._link_id = ""
        style._meta = None
        style._null = not (color or bgcolor)
        style._hash = None
        return style

    @classmethod
    def from_meta(cls, meta: Optional[Dict[str, Any]]) -> "Style":
        """Create a new style with meta data.

        Returns:
            meta (Optional[Dict[str, Any]]): A dictionary of meta data. Defaults to None.
        """
        style: Style = cls.__new__(Style)
        style._ansi = None
        style._style_definition = None
        style._color = None
        style._bgcolor = None
        style._set_attributes = 0
        style._attributes = 0
        style._link = None
        style._meta = dumps(meta)
        style._link_id = f"{randint(0, 999999)}{hash(style._meta)}"
        style._hash = None
        style._null = not (meta)
        return style

    @classmethod
    def on(cls, meta: Optional[Dict[str, Any]] = None, **handlers: Any) -> "Style":
        """Create a blank style with meta information.

        Example:
            style = Style.on(click=self.on_click)

        Args:
            meta (Optional[Dict[str, Any]], optional): An optional dict of meta information.
            **handlers (Any): Keyword arguments are translated in to handlers.

        Returns:
            Style: A Style with meta information attached.
        """
        meta = {} if meta is None else meta
        meta.update({f"@{key}": value for key, value in handlers.items()})
        return cls.from_meta(meta)

    bold = _Bit(0)
    dim = _Bit(1)
    italic = _Bit(2)
    underline = _Bit(3)
    blink = _Bit(4)
    blink2 = _Bit(5)
    reverse = _Bit(6)
    conceal = _Bit(7)
    strike = _Bit(8)
    underline2 = _Bit(9)
    frame = _Bit(10)
    encircle = _Bit(11)
    overline = _Bit(12)

    @property
    def link_id(self) -> str:
        """Get a link id, used in ansi code for links."""
        return self._link_id

    def __str__(self) -> str:
        """Re-generate style definition from attributes."""
        if self._style_definition is None:
            attributes: List[str] = []
            append = attributes.append
            bits = self._set_attributes
            if bits & 0b0000000001111:
                if bits & 1:
                    append("bold" if self.bold else "not bold")
                if bits & (1 << 1):
                    append("dim" if self.dim else "not dim")
                if bits & (1 << 2):
                    append("italic" if self.italic else "not italic")
                if bits & (1 << 3):
                    append("underline" if self.underline else "not underline")
            if bits & 0b0000111110000:
                if bits & (1 << 4):
                    append("blink" if self.blink else "not blink")
                if bits & (1 << 5):
                    append("blink2" if self.blink2 else "not blink2")
                if bits & (1 << 6):
                    append("reverse" if self.reverse else "not reverse")
                if bits & (1 << 7):
                    append("conceal" if self.conceal else "not conceal")
                if bits & (1 << 8):
                    append("strike" if self.strike else "not strike")
            if bits & 0b1111000000000:
                if bits & (1 << 9):
                    append("underline2" if self.underline2 else "not underline2")
                if bits & (1 << 10):
                    append("frame" if self.frame else "not frame")
                if bits & (1 << 11):
                    append("encircle" if self.encircle else "not encircle")
                if bits & (1 << 12):
                    append("overline" if self.overline else "not overline")
            if self._color is not None:
                append(self._color.name)
            if self._bgcolor is not None:
                append("on")
                append(self._bgcolor.name)
            if self._link:
                append("link")
                append(self._link)
            self._style_definition = " ".join(attributes) or "none"
        return self._style_definition

    def __bool__(self) -> bool:
        """A Style is false if it has no attributes, colors, or links."""
        return not self._null

    def _make_ansi_codes(self, color_system: ColorSystem) -> str:
        """Generate ANSI codes for this style.

        Args:
            color_system (ColorSystem): Color system.

        Returns:
            str: String containing codes.
        """

        if self._ansi is None:
            sgr: List[str] = []
            append = sgr.append
            _style_map = self._style_map
            attributes = self._attributes & self._set_attributes
            if attributes:
                if attributes & 1:
                    append(_style_map[0])
                if attributes & 2:
                    append(_style_map[1])
                if attributes & 4:
                    append(_style_map[2])
                if attributes & 8:
                    append(_style_map[3])
                if attributes & 0b0000111110000:
                    for bit in range(4, 9):
                        if attributes & (1 << bit):
                            append(_style_map[bit])
                if attributes & 0b1111000000000:
                    for bit in range(9, 13):
                        if attributes & (1 << bit):
                            append(_style_map[bit])
            if self._color is not None:
                sgr.extend(self._color.downgrade(color_system).get_ansi_codes())
            if self._bgcolor is not None:
                sgr.extend(
                    self._bgcolor.downgrade(color_system).get_ansi_codes(
                        foreground=False
                    )
                )
            self._ansi = ";".join(sgr)
        return self._ansi

    @classmethod
    @lru_cache(maxsize=1024)
    def normalize(cls, style: str) -> str:
        """Normalize a style definition so that styles with the same effect have the same string
        representation.

        Args:
            style (str): A style definition.

        Returns:
            str: Normal form of style definition.
        """
        try:
            return str(cls.parse(style))
        except errors.StyleSyntaxError:
            return style.strip().lower()

    @classmethod
    def pick_first(cls, *values: Optional[StyleType]) -> StyleType:
        """Pick first non-None style."""
        for value in values:
            if value is not None:
                return value
        raise ValueError("expected at least one non-None style")

    def __rich_repr__(self) -> Result:
        yield "color", self.color, None
        yield "bgcolor", self.bgcolor, None
        yield "bold", self.bold, None,
        yield "dim", self.dim, None,
        yield "italic", self.italic, None
        yield "underline", self.underline, None,
        yield "blink", self.blink, None
        yield "blink2", self.blink2, None
        yield "reverse", self.reverse, None
        yield "conceal", self.conceal, None
        yield "strike", self.strike, None
        yield "underline2", self.underline2, None
        yield "frame", self.frame, None
        yield "encircle", self.encircle, None
        yield "link", self.link, None
        if self._meta:
            yield "meta", self.meta

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Style):
            return NotImplemented
        return self.__hash__() == other.__hash__()

    def __ne__(self, other: Any) -> bool:
        if not isinstance(other, Style):
            return NotImplemented
        return self.__hash__() != other.__hash__()

    def __hash__(self) -> int:
        if self._hash is not None:
            return self._hash
        self._hash = hash(
            (
                self._color,
                self._bgcolor,
                self._attributes,
                self._set_attributes,
                self._link,
                self._meta,
            )
        )
        return self._hash

    @property
    def color(self) -> Optional[Color]:
        """The foreground color or None if it is not set."""
        return self._color

    @property
    def bgcolor(self) -> Optional[Color]:
        """The background color or None if it is not set."""
        return self._bgcolor

    @property
    def link(self) -> Optional[str]:
        """Link text, if set."""
        return self._link

    @property
    def transparent_background(self) -> bool:
        """Check if the style specified a transparent background."""
        return self.bgcolor is None or self.bgcolor.is_default

    @property
    def background_style(self) -> "Style":
        """A Style with background only."""
        return Style(bgcolor=self.bgcolor)

    @property
    def meta(self) -> Dict[str, Any]:
        """Get meta information (can not be changed after construction)."""
        return {} if self._meta is None else cast(Dict[str, Any], loads(self._meta))

    @property
    def without_color(self) -> "Style":
        """Get a copy of the style with color removed."""
        if self._null:
            return NULL_STYLE
        style: Style = self.__new__(Style)
        style._ansi = None
        style._style_definition = None
        style._color = None
        style._bgcolor = None
        style._attributes = self._attributes
        style._set_attributes = self._set_attributes
        style._link = self._link
        style._link_id = f"{randint(0, 999999)}" if self._link else ""
        style._null = False
        style._meta = None
        style._hash = None
        return style

    @classmethod
    @lru_cache(maxsize=4096)
    def parse(cls, style_definition: str) -> "Style":
        """Parse a style definition.

        Args:
            style_definition (str): A string containing a style.

        Raises:
            errors.StyleSyntaxError: If the style definition syntax is invalid.

        Returns:
            `Style`: A Style instance.
        """
        if style_definition.strip() == "none" or not style_definition:
            return cls.null()

        STYLE_ATTRIBUTES = cls.STYLE_ATTRIBUTES
        color: Optional[str] = None
        bgcolor: Optional[str] = None
        attributes: Dict[str, Optional[Any]] = {}
        link: Optional[str] = None

        words = iter(style_definition.split())
        for original_word in words:
            word = original_word.lower()
            if word == "on":
                word = next(words, "")
                if not word:
                    raise errors.StyleSyntaxError("color expected after 'on'")
                try:
                    Color.parse(word)
                except ColorParseError as error:
                    raise errors.StyleSyntaxError(
                        f"unable to parse {word!r} as background color; {error}"
                    ) from None
                bgcolor = word

            elif word == "not":
                word = next(words, "")
                attribute = STYLE_ATTRIBUTES.get(word)
                if attribute is None:
                    raise errors.StyleSyntaxError(
                        f"expected style attribute after 'not', found {word!r}"
                    )
                attributes[attribute] = False

            elif word == "link":
                word = next(words, "")
                if not word:
                    raise errors.StyleSyntaxError("URL expected after 'link'")
                link = word

            elif word in STYLE_ATTRIBUTES:
                attributes[STYLE_ATTRIBUTES[word]] = True

            else:
                try:
                    Color.parse(word)
                except ColorParseError as error:
                    raise errors.StyleSyntaxError(
                        f"unable to parse {word!r} as color; {error}"
                    ) from None
                color = word
        style = Style(color=color, bgcolor=bgcolor, link=link, **attributes)
        return style

    @lru_cache(maxsize=1024)
    def get_html_style(self, theme: Optional[TerminalTheme] = None) -> str:
        """Get a CSS style rule."""
        theme = theme or DEFAULT_TERMINAL_THEME
        css: List[str] = []
        append = css.append

        color = self.color
        bgcolor = self.bgcolor
        if self.reverse:
            color, bgcolor = bgcolor, color
        if self.dim:
            foreground_color = (
                theme.foreground_color if color is None else color.get_truecolor(theme)
            )
            color = Color.from_triplet(
                blend_rgb(foreground_color, theme.background_color, 0.5)
            )
        if color is not None:
            theme_color = color.get_truecolor(theme)
            append(f"color: {theme_color.hex}")
            append(f"text-decoration-color: {theme_color.hex}")
        if bgcolor is not None:
            theme_color = bgcolor.get_truecolor(theme, foreground=False)
            append(f"background-color: {theme_color.hex}")
        if self.bold:
            append("font-weight: bold")
        if self.italic:
            append("font-style: italic")
        if self.underline:
            append("text-decoration: underline")
        if self.strike:
            append("text-decoration: line-through")
        if self.overline:
            append("text-decoration: overline")
        return "; ".join(css)

    @classmethod
    def combine(cls, styles: Iterable["Style"]) -> "Style":
        """Combine styles and get result.

        Args:
            styles (Iterable[Style]): Styles to combine.

        Returns:
            Style: A new style instance.
        """
        iter_styles = iter(styles)
        return sum(iter_styles, next(iter_styles))

    @classmethod
    def chain(cls, *styles: "Style") -> "Style":
        """Combine styles from positional argument in to a single style.

        Args:
            *styles (Iterable[Style]): Styles to combine.

        Returns:
            Style: A new style instance.
        """
        iter_styles = iter(styles)
        return sum(iter_styles, next(iter_styles))

    def copy(self) -> "Style":
        """Get a copy of this style.

        Returns:
            Style: A new Style instance with identical attributes.
        """
        if self._null:
            return NULL_STYLE
        style: Style = self.__new__(Style)
        style._ansi = self._ansi
        style._style_definition = self._style_definition
        style._color = self._color
        style._bgcolor = self._bgcolor
        style._attributes = self._attributes
        style._set_attributes = self._set_attributes
        style._link = self._link
        style._link_id = f"{randint(0, 999999)}" if self._link else ""
        style._hash = self._hash
        style._null = False
        style._meta = self._meta
        return style

    @lru_cache(maxsize=128)
    def clear_meta_and_links(self) -> "Style":
        """Get a copy of this style with link and meta information removed.

        Returns:
            Style: New style object.
        """
        if self._null:
            return NULL_STYLE
        style: Style = self.__new__(Style)
        style._ansi = self._ansi
        style._style_definition = self._style_definition
        style._color = self._color
        style._bgcolor = self._bgcolor
        style._attributes = self._attributes
        style._set_attributes = self._set_attributes
        style._link = None
        style._link_id = ""
        style._hash = None
        style._null = False
        style._meta = None
        return style

    def update_link(self, link: Optional[str] = None) -> "Style":
        """Get a copy with a different value for link.

        Args:
            link (str, optional): New value for link. Defaults to None.

        Returns:
            Style: A new Style instance.
        """
        style: Style = self.__new__(Style)
        style._ansi = self._ansi
        style._style_definition = self._style_definition
        style._color = self._color
        style._bgcolor = self._bgcolor
        style._attributes = self._attributes
        style._set_attributes = self._set_attributes
        style._link = link
        style._link_id = f"{randint(0, 999999)}" if link else ""
        style._hash = None
        style._null = False
        style._meta = self._meta
        return style

    def render(
        self,
        text: str = "",
        *,
        color_system: Optional[ColorSystem] = ColorSystem.TRUECOLOR,
        legacy_windows: bool = False,
    ) -> str:
        """Render the ANSI codes for the style.

        Args:
            text (str, optional): A string to style. Defaults to "".
            color_system (Optional[ColorSystem], optional): Color system to render to. Defaults to ColorSystem.TRUECOLOR.

        Returns:
            str: A string containing ANSI style codes.
        """
        if not text or color_system is None:
            return text
        attrs = self._ansi or self._make_ansi_codes(color_system)
        rendered = f"\x1b[{attrs}m{text}\x1b[0m" if attrs else text
        if self._link and not legacy_windows:
            rendered = (
                f"\x1b]8;id={self._link_id};{self._link}\x1b\\{rendered}\x1b]8;;\x1b\\"
            )
        return rendered

    def test(self, text: Optional[str] = None) -> None:
        """Write text with style directly to terminal.

        This method is for testing purposes only.

        Args:
            text (Optional[str], optional): Text to style or None for style name.

        """
        text = text or str(self)
        sys.stdout.write(f"{self.render(text)}\n")

    @lru_cache(maxsize=1024)
    def _add(self, style: Optional["Style"]) -> "Style":
        if style is None or style._null:
            return self
        if self._null:
            return style
        new_style: Style = self.__new__(Style)
        new_style._ansi = None
        new_style._style_definition = None
        new_style._color = style._color or self._color
        new_style._bgcolor = style._bgcolor or self._bgcolor
        new_style._attributes = (self._attributes & ~style._set_attributes) | (
            style._attributes & style._set_attributes
        )
        new_style._set_attributes = self._set_attributes | style._set_attributes
        new_style._link = style._link or self._link
        new_style._link_id = style._link_id or self._link_id
        new_style._null = style._null
        if self._meta and style._meta:
            new_style._meta = dumps({**self.meta, **style.meta})
        else:
            new_style._meta = self._meta or style._meta
        new_style._hash = None
        return new_style

    def __add__(self, style: Optional["Style"]) -> "Style":
        combined_style = self._add(style)
        return combined_style.copy() if combined_style.link else combined_style


NULL_STYLE = Style()


class StyleStack:
    """A stack of styles."""

    __slots__ = ["_stack"]

    def __init__(self, default_style: "Style") -> None:
        self._stack: List[Style] = [default_style]

    def __repr__(self) -> str:
        return f"<stylestack {self._stack!r}>"

    @property
    def current(self) -> Style:
        """Get the Style at the top of the stack."""
        return self._stack[-1]

    def push(self, style: Style) -> None:
        """Push a new style on to the stack.

        Args:
            style (Style): New style to combine with current style.
        """
        self._stack.append(self._stack[-1] + style)

    def pop(self) -> Style:
        """Pop last style and discard.

        Returns:
            Style: New current style (also available as stack.current)
        """
        self._stack.pop()
        return self._stack[-1]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\unitofwork.py ===
# orm/unitofwork.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""The internals for the unit of work system.

The session's flush() process passes objects to a contextual object
here, which assembles flush tasks based on mappers and their properties,
organizes them in order of dependency, and executes.

"""

from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Set
from typing import TYPE_CHECKING

from . import attributes
from . import exc as orm_exc
from . import util as orm_util
from .. import event
from .. import util
from ..util import topological


if TYPE_CHECKING:
    from .dependency import DependencyProcessor
    from .interfaces import MapperProperty
    from .mapper import Mapper
    from .session import Session
    from .session import SessionTransaction
    from .state import InstanceState


def track_cascade_events(descriptor, prop):
    """Establish event listeners on object attributes which handle
    cascade-on-set/append.

    """
    key = prop.key

    def append(state, item, initiator, **kw):
        # process "save_update" cascade rules for when
        # an instance is appended to the list of another instance

        if item is None:
            return

        sess = state.session
        if sess:
            if sess._warn_on_events:
                sess._flush_warning("collection append")

            prop = state.manager.mapper._props[key]
            item_state = attributes.instance_state(item)

            if (
                prop._cascade.save_update
                and (key == initiator.key)
                and not sess._contains_state(item_state)
            ):
                sess._save_or_update_state(item_state)
        return item

    def remove(state, item, initiator, **kw):
        if item is None:
            return

        sess = state.session

        prop = state.manager.mapper._props[key]

        if sess and sess._warn_on_events:
            sess._flush_warning(
                "collection remove"
                if prop.uselist
                else "related attribute delete"
            )

        if (
            item is not None
            and item is not attributes.NEVER_SET
            and item is not attributes.PASSIVE_NO_RESULT
            and prop._cascade.delete_orphan
        ):
            # expunge pending orphans
            item_state = attributes.instance_state(item)

            if prop.mapper._is_orphan(item_state):
                if sess and item_state in sess._new:
                    sess.expunge(item)
                else:
                    # the related item may or may not itself be in a
                    # Session, however the parent for which we are catching
                    # the event is not in a session, so memoize this on the
                    # item
                    item_state._orphaned_outside_of_session = True

    def set_(state, newvalue, oldvalue, initiator, **kw):
        # process "save_update" cascade rules for when an instance
        # is attached to another instance
        if oldvalue is newvalue:
            return newvalue

        sess = state.session
        if sess:
            if sess._warn_on_events:
                sess._flush_warning("related attribute set")

            prop = state.manager.mapper._props[key]
            if newvalue is not None:
                newvalue_state = attributes.instance_state(newvalue)
                if (
                    prop._cascade.save_update
                    and (key == initiator.key)
                    and not sess._contains_state(newvalue_state)
                ):
                    sess._save_or_update_state(newvalue_state)

            if (
                oldvalue is not None
                and oldvalue is not attributes.NEVER_SET
                and oldvalue is not attributes.PASSIVE_NO_RESULT
                and prop._cascade.delete_orphan
            ):
                # possible to reach here with attributes.NEVER_SET ?
                oldvalue_state = attributes.instance_state(oldvalue)

                if oldvalue_state in sess._new and prop.mapper._is_orphan(
                    oldvalue_state
                ):
                    sess.expunge(oldvalue)
        return newvalue

    event.listen(
        descriptor, "append_wo_mutation", append, raw=True, include_key=True
    )
    event.listen(
        descriptor, "append", append, raw=True, retval=True, include_key=True
    )
    event.listen(
        descriptor, "remove", remove, raw=True, retval=True, include_key=True
    )
    event.listen(
        descriptor, "set", set_, raw=True, retval=True, include_key=True
    )


class UOWTransaction:
    session: Session
    transaction: SessionTransaction
    attributes: Dict[str, Any]
    deps: util.defaultdict[Mapper[Any], Set[DependencyProcessor]]
    mappers: util.defaultdict[Mapper[Any], Set[InstanceState[Any]]]

    def __init__(self, session: Session):
        self.session = session

        # dictionary used by external actors to
        # store arbitrary state information.
        self.attributes = {}

        # dictionary of mappers to sets of
        # DependencyProcessors, which are also
        # set to be part of the sorted flush actions,
        # which have that mapper as a parent.
        self.deps = util.defaultdict(set)

        # dictionary of mappers to sets of InstanceState
        # items pending for flush which have that mapper
        # as a parent.
        self.mappers = util.defaultdict(set)

        # a dictionary of Preprocess objects, which gather
        # additional states impacted by the flush
        # and determine if a flush action is needed
        self.presort_actions = {}

        # dictionary of PostSortRec objects, each
        # one issues work during the flush within
        # a certain ordering.
        self.postsort_actions = {}

        # a set of 2-tuples, each containing two
        # PostSortRec objects where the second
        # is dependent on the first being executed
        # first
        self.dependencies = set()

        # dictionary of InstanceState-> (isdelete, listonly)
        # tuples, indicating if this state is to be deleted
        # or insert/updated, or just refreshed
        self.states = {}

        # tracks InstanceStates which will be receiving
        # a "post update" call.  Keys are mappers,
        # values are a set of states and a set of the
        # columns which should be included in the update.
        self.post_update_states = util.defaultdict(lambda: (set(), set()))

    @property
    def has_work(self):
        return bool(self.states)

    def was_already_deleted(self, state):
        """Return ``True`` if the given state is expired and was deleted
        previously.
        """
        if state.expired:
            try:
                state._load_expired(state, attributes.PASSIVE_OFF)
            except orm_exc.ObjectDeletedError:
                self.session._remove_newly_deleted([state])
                return True
        return False

    def is_deleted(self, state):
        """Return ``True`` if the given state is marked as deleted
        within this uowtransaction."""

        return state in self.states and self.states[state][0]

    def memo(self, key, callable_):
        if key in self.attributes:
            return self.attributes[key]
        else:
            self.attributes[key] = ret = callable_()
            return ret

    def remove_state_actions(self, state):
        """Remove pending actions for a state from the uowtransaction."""

        isdelete = self.states[state][0]

        self.states[state] = (isdelete, True)

    def get_attribute_history(
        self, state, key, passive=attributes.PASSIVE_NO_INITIALIZE
    ):
        """Facade to attributes.get_state_history(), including
        caching of results."""

        hashkey = ("history", state, key)

        # cache the objects, not the states; the strong reference here
        # prevents newly loaded objects from being dereferenced during the
        # flush process

        if hashkey in self.attributes:
            history, state_history, cached_passive = self.attributes[hashkey]
            # if the cached lookup was "passive" and now
            # we want non-passive, do a non-passive lookup and re-cache

            if (
                not cached_passive & attributes.SQL_OK
                and passive & attributes.SQL_OK
            ):
                impl = state.manager[key].impl
                history = impl.get_history(
                    state,
                    state.dict,
                    attributes.PASSIVE_OFF
                    | attributes.LOAD_AGAINST_COMMITTED
                    | attributes.NO_RAISE,
                )
                if history and impl.uses_objects:
                    state_history = history.as_state()
                else:
                    state_history = history
                self.attributes[hashkey] = (history, state_history, passive)
        else:
            impl = state.manager[key].impl
            # TODO: store the history as (state, object) tuples
            # so we don't have to keep converting here
            history = impl.get_history(
                state,
                state.dict,
                passive
                | attributes.LOAD_AGAINST_COMMITTED
                | attributes.NO_RAISE,
            )
            if history and impl.uses_objects:
                state_history = history.as_state()
            else:
                state_history = history
            self.attributes[hashkey] = (history, state_history, passive)

        return state_history

    def has_dep(self, processor):
        return (processor, True) in self.presort_actions

    def register_preprocessor(self, processor, fromparent):
        key = (processor, fromparent)
        if key not in self.presort_actions:
            self.presort_actions[key] = Preprocess(processor, fromparent)

    def register_object(
        self,
        state: InstanceState[Any],
        isdelete: bool = False,
        listonly: bool = False,
        cancel_delete: bool = False,
        operation: Optional[str] = None,
        prop: Optional[MapperProperty] = None,
    ) -> bool:
        if not self.session._contains_state(state):
            # this condition is normal when objects are registered
            # as part of a relationship cascade operation.  it should
            # not occur for the top-level register from Session.flush().
            if not state.deleted and operation is not None:
                util.warn(
                    "Object of type %s not in session, %s operation "
                    "along '%s' will not proceed"
                    % (orm_util.state_class_str(state), operation, prop)
                )
            return False

        if state not in self.states:
            mapper = state.manager.mapper

            if mapper not in self.mappers:
                self._per_mapper_flush_actions(mapper)

            self.mappers[mapper].add(state)
            self.states[state] = (isdelete, listonly)
        else:
            if not listonly and (isdelete or cancel_delete):
                self.states[state] = (isdelete, False)
        return True

    def register_post_update(self, state, post_update_cols):
        mapper = state.manager.mapper.base_mapper
        states, cols = self.post_update_states[mapper]
        states.add(state)
        cols.update(post_update_cols)

    def _per_mapper_flush_actions(self, mapper):
        saves = SaveUpdateAll(self, mapper.base_mapper)
        deletes = DeleteAll(self, mapper.base_mapper)
        self.dependencies.add((saves, deletes))

        for dep in mapper._dependency_processors:
            dep.per_property_preprocessors(self)

        for prop in mapper.relationships:
            if prop.viewonly:
                continue
            dep = prop._dependency_processor
            dep.per_property_preprocessors(self)

    @util.memoized_property
    def _mapper_for_dep(self):
        """return a dynamic mapping of (Mapper, DependencyProcessor) to
        True or False, indicating if the DependencyProcessor operates
        on objects of that Mapper.

        The result is stored in the dictionary persistently once
        calculated.

        """
        return util.PopulateDict(
            lambda tup: tup[0]._props.get(tup[1].key) is tup[1].prop
        )

    def filter_states_for_dep(self, dep, states):
        """Filter the given list of InstanceStates to those relevant to the
        given DependencyProcessor.

        """
        mapper_for_dep = self._mapper_for_dep
        return [s for s in states if mapper_for_dep[(s.manager.mapper, dep)]]

    def states_for_mapper_hierarchy(self, mapper, isdelete, listonly):
        checktup = (isdelete, listonly)
        for mapper in mapper.base_mapper.self_and_descendants:
            for state in self.mappers[mapper]:
                if self.states[state] == checktup:
                    yield state

    def _generate_actions(self):
        """Generate the full, unsorted collection of PostSortRecs as
        well as dependency pairs for this UOWTransaction.

        """
        # execute presort_actions, until all states
        # have been processed.   a presort_action might
        # add new states to the uow.
        while True:
            ret = False
            for action in list(self.presort_actions.values()):
                if action.execute(self):
                    ret = True
            if not ret:
                break

        # see if the graph of mapper dependencies has cycles.
        self.cycles = cycles = topological.find_cycles(
            self.dependencies, list(self.postsort_actions.values())
        )

        if cycles:
            # if yes, break the per-mapper actions into
            # per-state actions
            convert = {
                rec: set(rec.per_state_flush_actions(self)) for rec in cycles
            }

            # rewrite the existing dependencies to point to
            # the per-state actions for those per-mapper actions
            # that were broken up.
            for edge in list(self.dependencies):
                if (
                    None in edge
                    or edge[0].disabled
                    or edge[1].disabled
                    or cycles.issuperset(edge)
                ):
                    self.dependencies.remove(edge)
                elif edge[0] in cycles:
                    self.dependencies.remove(edge)
                    for dep in convert[edge[0]]:
                        self.dependencies.add((dep, edge[1]))
                elif edge[1] in cycles:
                    self.dependencies.remove(edge)
                    for dep in convert[edge[1]]:
                        self.dependencies.add((edge[0], dep))

        return {
            a for a in self.postsort_actions.values() if not a.disabled
        }.difference(cycles)

    def execute(self) -> None:
        postsort_actions = self._generate_actions()

        postsort_actions = sorted(
            postsort_actions,
            key=lambda item: item.sort_key,
        )
        # sort = topological.sort(self.dependencies, postsort_actions)
        # print "--------------"
        # print "\ndependencies:", self.dependencies
        # print "\ncycles:", self.cycles
        # print "\nsort:", list(sort)
        # print "\nCOUNT OF POSTSORT ACTIONS", len(postsort_actions)

        # execute
        if self.cycles:
            for subset in topological.sort_as_subsets(
                self.dependencies, postsort_actions
            ):
                set_ = set(subset)
                while set_:
                    n = set_.pop()
                    n.execute_aggregate(self, set_)
        else:
            for rec in topological.sort(self.dependencies, postsort_actions):
                rec.execute(self)

    def finalize_flush_changes(self) -> None:
        """Mark processed objects as clean / deleted after a successful
        flush().

        This method is called within the flush() method after the
        execute() method has succeeded and the transaction has been committed.

        """
        if not self.states:
            return

        states = set(self.states)
        isdel = {
            s for (s, (isdelete, listonly)) in self.states.items() if isdelete
        }
        other = states.difference(isdel)
        if isdel:
            self.session._remove_newly_deleted(isdel)
        if other:
            self.session._register_persistent(other)


class IterateMappersMixin:
    __slots__ = ()

    def _mappers(self, uow):
        if self.fromparent:
            return iter(
                m
                for m in self.dependency_processor.parent.self_and_descendants
                if uow._mapper_for_dep[(m, self.dependency_processor)]
            )
        else:
            return self.dependency_processor.mapper.self_and_descendants


class Preprocess(IterateMappersMixin):
    __slots__ = (
        "dependency_processor",
        "fromparent",
        "processed",
        "setup_flush_actions",
    )

    def __init__(self, dependency_processor, fromparent):
        self.dependency_processor = dependency_processor
        self.fromparent = fromparent
        self.processed = set()
        self.setup_flush_actions = False

    def execute(self, uow):
        delete_states = set()
        save_states = set()

        for mapper in self._mappers(uow):
            for state in uow.mappers[mapper].difference(self.processed):
                (isdelete, listonly) = uow.states[state]
                if not listonly:
                    if isdelete:
                        delete_states.add(state)
                    else:
                        save_states.add(state)

        if delete_states:
            self.dependency_processor.presort_deletes(uow, delete_states)
            self.processed.update(delete_states)
        if save_states:
            self.dependency_processor.presort_saves(uow, save_states)
            self.processed.update(save_states)

        if delete_states or save_states:
            if not self.setup_flush_actions and (
                self.dependency_processor.prop_has_changes(
                    uow, delete_states, True
                )
                or self.dependency_processor.prop_has_changes(
                    uow, save_states, False
                )
            ):
                self.dependency_processor.per_property_flush_actions(uow)
                self.setup_flush_actions = True
            return True
        else:
            return False


class PostSortRec:
    __slots__ = ("disabled",)

    def __new__(cls, uow, *args):
        key = (cls,) + args
        if key in uow.postsort_actions:
            return uow.postsort_actions[key]
        else:
            uow.postsort_actions[key] = ret = object.__new__(cls)
            ret.disabled = False
            return ret

    def execute_aggregate(self, uow, recs):
        self.execute(uow)


class ProcessAll(IterateMappersMixin, PostSortRec):
    __slots__ = "dependency_processor", "isdelete", "fromparent", "sort_key"

    def __init__(self, uow, dependency_processor, isdelete, fromparent):
        self.dependency_processor = dependency_processor
        self.sort_key = (
            "ProcessAll",
            self.dependency_processor.sort_key,
            isdelete,
        )
        self.isdelete = isdelete
        self.fromparent = fromparent
        uow.deps[dependency_processor.parent.base_mapper].add(
            dependency_processor
        )

    def execute(self, uow):
        states = self._elements(uow)
        if self.isdelete:
            self.dependency_processor.process_deletes(uow, states)
        else:
            self.dependency_processor.process_saves(uow, states)

    def per_state_flush_actions(self, uow):
        # this is handled by SaveUpdateAll and DeleteAll,
        # since a ProcessAll should unconditionally be pulled
        # into per-state if either the parent/child mappers
        # are part of a cycle
        return iter([])

    def __repr__(self):
        return "%s(%s, isdelete=%s)" % (
            self.__class__.__name__,
            self.dependency_processor,
            self.isdelete,
        )

    def _elements(self, uow):
        for mapper in self._mappers(uow):
            for state in uow.mappers[mapper]:
                (isdelete, listonly) = uow.states[state]
                if isdelete == self.isdelete and not listonly:
                    yield state


class PostUpdateAll(PostSortRec):
    __slots__ = "mapper", "isdelete", "sort_key"

    def __init__(self, uow, mapper, isdelete):
        self.mapper = mapper
        self.isdelete = isdelete
        self.sort_key = ("PostUpdateAll", mapper._sort_key, isdelete)

    @util.preload_module("sqlalchemy.orm.persistence")
    def execute(self, uow):
        persistence = util.preloaded.orm_persistence
        states, cols = uow.post_update_states[self.mapper]
        states = [s for s in states if uow.states[s][0] == self.isdelete]

        persistence.post_update(self.mapper, states, uow, cols)


class SaveUpdateAll(PostSortRec):
    __slots__ = ("mapper", "sort_key")

    def __init__(self, uow, mapper):
        self.mapper = mapper
        self.sort_key = ("SaveUpdateAll", mapper._sort_key)
        assert mapper is mapper.base_mapper

    @util.preload_module("sqlalchemy.orm.persistence")
    def execute(self, uow):
        util.preloaded.orm_persistence.save_obj(
            self.mapper,
            uow.states_for_mapper_hierarchy(self.mapper, False, False),
            uow,
        )

    def per_state_flush_actions(self, uow):
        states = list(
            uow.states_for_mapper_hierarchy(self.mapper, False, False)
        )
        base_mapper = self.mapper.base_mapper
        delete_all = DeleteAll(uow, base_mapper)
        for state in states:
            # keep saves before deletes -
            # this ensures 'row switch' operations work
            action = SaveUpdateState(uow, state)
            uow.dependencies.add((action, delete_all))
            yield action

        for dep in uow.deps[self.mapper]:
            states_for_prop = uow.filter_states_for_dep(dep, states)
            dep.per_state_flush_actions(uow, states_for_prop, False)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.mapper)


class DeleteAll(PostSortRec):
    __slots__ = ("mapper", "sort_key")

    def __init__(self, uow, mapper):
        self.mapper = mapper
        self.sort_key = ("DeleteAll", mapper._sort_key)
        assert mapper is mapper.base_mapper

    @util.preload_module("sqlalchemy.orm.persistence")
    def execute(self, uow):
        util.preloaded.orm_persistence.delete_obj(
            self.mapper,
            uow.states_for_mapper_hierarchy(self.mapper, True, False),
            uow,
        )

    def per_state_flush_actions(self, uow):
        states = list(
            uow.states_for_mapper_hierarchy(self.mapper, True, False)
        )
        base_mapper = self.mapper.base_mapper
        save_all = SaveUpdateAll(uow, base_mapper)
        for state in states:
            # keep saves before deletes -
            # this ensures 'row switch' operations work
            action = DeleteState(uow, state)
            uow.dependencies.add((save_all, action))
            yield action

        for dep in uow.deps[self.mapper]:
            states_for_prop = uow.filter_states_for_dep(dep, states)
            dep.per_state_flush_actions(uow, states_for_prop, True)

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.mapper)


class ProcessState(PostSortRec):
    __slots__ = "dependency_processor", "isdelete", "state", "sort_key"

    def __init__(self, uow, dependency_processor, isdelete, state):
        self.dependency_processor = dependency_processor
        self.sort_key = ("ProcessState", dependency_processor.sort_key)
        self.isdelete = isdelete
        self.state = state

    def execute_aggregate(self, uow, recs):
        cls_ = self.__class__
        dependency_processor = self.dependency_processor
        isdelete = self.isdelete
        our_recs = [
            r
            for r in recs
            if r.__class__ is cls_
            and r.dependency_processor is dependency_processor
            and r.isdelete is isdelete
        ]
        recs.difference_update(our_recs)
        states = [self.state] + [r.state for r in our_recs]
        if isdelete:
            dependency_processor.process_deletes(uow, states)
        else:
            dependency_processor.process_saves(uow, states)

    def __repr__(self):
        return "%s(%s, %s, delete=%s)" % (
            self.__class__.__name__,
            self.dependency_processor,
            orm_util.state_str(self.state),
            self.isdelete,
        )


class SaveUpdateState(PostSortRec):
    __slots__ = "state", "mapper", "sort_key"

    def __init__(self, uow, state):
        self.state = state
        self.mapper = state.mapper.base_mapper
        self.sort_key = ("ProcessState", self.mapper._sort_key)

    @util.preload_module("sqlalchemy.orm.persistence")
    def execute_aggregate(self, uow, recs):
        persistence = util.preloaded.orm_persistence
        cls_ = self.__class__
        mapper = self.mapper
        our_recs = [
            r for r in recs if r.__class__ is cls_ and r.mapper is mapper
        ]
        recs.difference_update(our_recs)
        persistence.save_obj(
            mapper, [self.state] + [r.state for r in our_recs], uow
        )

    def __repr__(self):
        return "%s(%s)" % (
            self.__class__.__name__,
            orm_util.state_str(self.state),
        )


class DeleteState(PostSortRec):
    __slots__ = "state", "mapper", "sort_key"

    def __init__(self, uow, state):
        self.state = state
        self.mapper = state.mapper.base_mapper
        self.sort_key = ("DeleteState", self.mapper._sort_key)

    @util.preload_module("sqlalchemy.orm.persistence")
    def execute_aggregate(self, uow, recs):
        persistence = util.preloaded.orm_persistence
        cls_ = self.__class__
        mapper = self.mapper
        our_recs = [
            r for r in recs if r.__class__ is cls_ and r.mapper is mapper
        ]
        recs.difference_update(our_recs)
        states = [self.state] + [r.state for r in our_recs]
        persistence.delete_obj(
            mapper, [s for s in states if uow.states[s][0]], uow
        )

    def __repr__(self):
        return "%s(%s)" % (
            self.__class__.__name__,
            orm_util.state_str(self.state),
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\puiseux.py ===
"""
Puiseux rings. These are used by the ring_series module to represented
truncated Puiseux series. Elements of a Puiseux ring are like polynomials
except that the exponents can be negative or rational rather than just
non-negative integers.
"""

# Previously the ring_series module used PolyElement to represent Puiseux
# series. This is problematic because it means that PolyElement has to support
# negative and non-integer exponents which most polynomial representations do
# not support. This module provides an implementation of a ring for Puiseux
# series that can be used by ring_series without breaking the basic invariants
# of polynomial rings.
#
# Ideally there would be more of a proper series type that can keep track of
# not just the leading terms of a truncated series but also the precision
# of the series. For now the rings here are just introduced to keep the
# interface that ring_series was using before.

from __future__ import annotations

from sympy.polys.domains import QQ
from sympy.polys.rings import PolyRing, PolyElement
from sympy.core.add import Add
from sympy.core.mul import Mul
from sympy.external.gmpy import gcd, lcm


from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from typing import Any, Unpack
    from sympy.core.expr import Expr
    from sympy.polys.domains import Domain
    from collections.abc import Iterable, Iterator


def puiseux_ring(
    symbols: str | list[Expr], domain: Domain
) -> tuple[PuiseuxRing, Unpack[tuple[PuiseuxPoly, ...]]]:
    """Construct a Puiseux ring.

    This function constructs a Puiseux ring with the given symbols and domain.

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> R, x, y = puiseux_ring('x y', QQ)
    >>> R
    PuiseuxRing((x, y), QQ)
    >>> p = 5*x**QQ(1,2) + 7/y
    >>> p
    7*y**(-1) + 5*x**(1/2)
    """
    ring = PuiseuxRing(symbols, domain)
    return (ring,) + ring.gens # type: ignore


class PuiseuxRing:
    """Ring of Puiseux polynomials.

    A Puiseux polynomial is a truncated Puiseux series. The exponents of the
    monomials can be negative or rational numbers. This ring is used by the
    ring_series module:

    >>> from sympy.polys.domains import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> from sympy.polys.ring_series import rs_exp, rs_nth_root
    >>> ring, x, y = puiseux_ring('x y', QQ)
    >>> f = x**2 + y**3
    >>> f
    y**3 + x**2
    >>> f.diff(x)
    2*x
    >>> rs_exp(x, x, 5)
    1 + x + 1/2*x**2 + 1/6*x**3 + 1/24*x**4

    Importantly the Puiseux ring can represent truncated series with negative
    and fractional exponents:

    >>> f = 1/x + 1/y**2
    >>> f
    x**(-1) + y**(-2)
    >>> f.diff(x)
    -1*x**(-2)

    >>> rs_nth_root(8*x + x**2 + x**3, 3, x, 5)
    2*x**(1/3) + 1/12*x**(4/3) + 23/288*x**(7/3) + -139/20736*x**(10/3)

    See Also
    ========

    sympy.polys.ring_series.rs_series
    PuiseuxPoly
    """
    def __init__(self, symbols: str | list[Expr], domain: Domain):

        poly_ring = PolyRing(symbols, domain)

        domain = poly_ring.domain
        ngens = poly_ring.ngens

        self.poly_ring = poly_ring
        self.domain = domain

        self.symbols = poly_ring.symbols
        self.gens = tuple([self.from_poly(g) for g in poly_ring.gens])
        self.ngens = ngens

        self.zero = self.from_poly(poly_ring.zero)
        self.one = self.from_poly(poly_ring.one)

        self.zero_monom = poly_ring.zero_monom # type: ignore
        self.monomial_mul = poly_ring.monomial_mul # type: ignore

    def __repr__(self) -> str:
        return f"PuiseuxRing({self.symbols}, {self.domain})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PuiseuxRing):
            return NotImplemented
        return self.symbols == other.symbols and self.domain == other.domain

    def from_poly(self, poly: PolyElement) -> PuiseuxPoly:
        """Create a Puiseux polynomial from a polynomial.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.rings import ring
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R1, x1 = ring('x', QQ)
        >>> R2, x2 = puiseux_ring('x', QQ)
        >>> R2.from_poly(x1**2)
        x**2
        """
        return PuiseuxPoly(poly, self)

    def from_dict(self, terms: dict[tuple[int, ...], Any]) -> PuiseuxPoly:
        """Create a Puiseux polynomial from a dictionary of terms.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.from_dict({(QQ(1,2),): QQ(3)})
        3*x**(1/2)
        """
        return PuiseuxPoly.from_dict(terms, self)

    def from_int(self, n: int) -> PuiseuxPoly:
        """Create a Puiseux polynomial from an integer.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.from_int(3)
        3
        """
        return self.from_poly(self.poly_ring(n))

    def domain_new(self, arg: Any) -> Any:
        """Create a new element of the domain.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.domain_new(3)
        3
        >>> QQ.of_type(_)
        True
        """
        return self.poly_ring.domain_new(arg)

    def ground_new(self, arg: Any) -> PuiseuxPoly:
        """Create a new element from a ground element.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring, PuiseuxPoly
        >>> R, x = puiseux_ring('x', QQ)
        >>> R.ground_new(3)
        3
        >>> isinstance(_, PuiseuxPoly)
        True
        """
        return self.from_poly(self.poly_ring.ground_new(arg))

    def __call__(self, arg: Any) -> PuiseuxPoly:
        """Coerce an element into the ring.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> R(3)
        3
        >>> R({(QQ(1,2),): QQ(3)})
        3*x**(1/2)
        """
        if isinstance(arg, dict):
            return self.from_dict(arg)
        else:
            return self.from_poly(self.poly_ring(arg))

    def index(self, x: PuiseuxPoly) -> int:
        """Return the index of a generator.

        >>> from sympy.polys.domains import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x y', QQ)
        >>> R.index(x)
        0
        >>> R.index(y)
        1
        """
        return self.gens.index(x)


def _div_poly_monom(poly: PolyElement, monom: Iterable[int]) -> PolyElement:
    ring = poly.ring
    div = ring.monomial_div
    return ring.from_dict({div(m, monom): c for m, c in poly.terms()})


def _mul_poly_monom(poly: PolyElement, monom: Iterable[int]) -> PolyElement:
    ring = poly.ring
    mul = ring.monomial_mul
    return ring.from_dict({mul(m, monom): c for m, c in poly.terms()})


def _div_monom(monom: Iterable[int], div: Iterable[int]) -> tuple[int, ...]:
    return tuple(mi - di for mi, di in zip(monom, div))


class PuiseuxPoly:
    """Puiseux polynomial. Represents a truncated Puiseux series.

    See the :class:`PuiseuxRing` class for more information.

    >>> from sympy import QQ
    >>> from sympy.polys.puiseux import puiseux_ring
    >>> R, x, y = puiseux_ring('x, y', QQ)
    >>> p = 5*x**2 + 7*y**3
    >>> p
    7*y**3 + 5*x**2

    The internal representation of a Puiseux polynomial wraps a normal
    polynomial. To support negative powers the polynomial is considered to be
    divided by a monomial.

    >>> p2 = 1/x + 1/y**2
    >>> p2.monom # x*y**2
    (1, 2)
    >>> p2.poly
    x + y**2
    >>> (y**2 + x) / (x*y**2) == p2
    True

    To support fractional powers the polynomial is considered to be a function
    of ``x**(1/nx), y**(1/ny), ...``. The representation keeps track of a
    monomial and a list of exponent denominators so that the polynomial can be
    used to represent both negative and fractional powers.

    >>> p3 = x**QQ(1,2) + y**QQ(2,3)
    >>> p3.ns
    (2, 3)
    >>> p3.poly
    x + y**2

    See Also
    ========

    sympy.polys.puiseux.PuiseuxRing
    sympy.polys.rings.PolyElement
    """

    ring: PuiseuxRing
    poly: PolyElement
    monom: tuple[int, ...] | None
    ns: tuple[int, ...] | None

    def __new__(cls, poly: PolyElement, ring: PuiseuxRing) -> PuiseuxPoly:
        return cls._new(ring, poly, None, None)

    @classmethod
    def _new(
        cls,
        ring: PuiseuxRing,
        poly: PolyElement,
        monom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> PuiseuxPoly:
        poly, monom, ns = cls._normalize(poly, monom, ns)
        return cls._new_raw(ring, poly, monom, ns)

    @classmethod
    def _new_raw(
        cls,
        ring: PuiseuxRing,
        poly: PolyElement,
        monom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> PuiseuxPoly:
        obj = object.__new__(cls)
        obj.ring = ring
        obj.poly = poly
        obj.monom = monom
        obj.ns = ns
        return obj

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, PuiseuxPoly):
            return (
                self.poly == other.poly
                and self.monom == other.monom
                and self.ns == other.ns
            )
        elif self.monom is None and self.ns is None:
            return self.poly.__eq__(other)
        else:
            return NotImplemented

    @classmethod
    def _normalize(
        cls,
        poly: PolyElement,
        monom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> tuple[PolyElement, tuple[int, ...] | None, tuple[int, ...] | None]:
        if monom is None and ns is None:
            return poly, None, None

        if monom is not None:
            degs = [max(d, 0) for d in poly.tail_degrees()]
            if all(di >= mi for di, mi in zip(degs, monom)):
                poly = _div_poly_monom(poly, monom)
                monom = None
            elif any(degs):
                poly = _div_poly_monom(poly, degs)
                monom = _div_monom(monom, degs)

        if ns is not None:
            factors_d, [poly_d] = poly.deflate()
            degrees = poly.degrees()
            monom_d = monom if monom is not None else [0] * len(degrees)
            ns_new = []
            monom_new = []
            inflations = []
            for fi, ni, di, mi in zip(factors_d, ns, degrees, monom_d):
                if di == 0:
                    g = gcd(ni, mi)
                else:
                    g = gcd(fi, ni, mi)
                ns_new.append(ni // g)
                monom_new.append(mi // g)
                inflations.append(fi // g)

            if any(infl > 1 for infl in inflations):
                poly_d = poly_d.inflate(inflations)

            poly = poly_d

            if monom is not None:
                monom = tuple(monom_new)

            if all(n == 1 for n in ns_new):
                ns = None
            else:
                ns = tuple(ns_new)

        return poly, monom, ns

    @classmethod
    def _monom_fromint(
        cls,
        monom: tuple[int, ...],
        dmonom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> tuple[Any, ...]:
        if dmonom is not None and ns is not None:
            return tuple(QQ(mi - di, ni) for mi, di, ni in zip(monom, dmonom, ns))
        elif dmonom is not None:
            return tuple(QQ(mi - di) for mi, di in zip(monom, dmonom))
        elif ns is not None:
            return tuple(QQ(mi, ni) for mi, ni in zip(monom, ns))
        else:
            return tuple(QQ(mi) for mi in monom)

    @classmethod
    def _monom_toint(
        cls,
        monom: tuple[Any, ...],
        dmonom: tuple[int, ...] | None,
        ns: tuple[int, ...] | None,
    ) -> tuple[int, ...]:
        if dmonom is not None and ns is not None:
            return tuple(
                int((mi * ni).numerator + di) for mi, di, ni in zip(monom, dmonom, ns)
            )
        elif dmonom is not None:
            return tuple(int(mi.numerator + di) for mi, di in zip(monom, dmonom))
        elif ns is not None:
            return tuple(int((mi * ni).numerator) for mi, ni in zip(monom, ns))
        else:
            return tuple(int(mi.numerator) for mi in monom)

    def itermonoms(self) -> Iterator[tuple[Any, ...]]:
        """Iterate over the monomials of a Puiseux polynomial.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x, y', QQ)
        >>> p = 5*x**2 + 7*y**3
        >>> list(p.itermonoms())
        [(2, 0), (0, 3)]
        >>> p[(2, 0)]
        5
        """
        monom, ns = self.monom, self.ns
        for m in self.poly.itermonoms():
            yield self._monom_fromint(m, monom, ns)

    def monoms(self) -> list[tuple[Any, ...]]:
        """Return a list of the monomials of a Puiseux polynomial."""
        return list(self.itermonoms())

    def __iter__(self) -> Iterator[tuple[tuple[Any, ...], Any]]:
        return self.itermonoms()

    def __getitem__(self, monom: tuple[int, ...]) -> Any:
        monom = self._monom_toint(monom, self.monom, self.ns)
        return self.poly[monom]

    def __len__(self) -> int:
        return len(self.poly)

    def iterterms(self) -> Iterator[tuple[tuple[Any, ...], Any]]:
        """Iterate over the terms of a Puiseux polynomial.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x, y', QQ)
        >>> p = 5*x**2 + 7*y**3
        >>> list(p.iterterms())
        [((2, 0), 5), ((0, 3), 7)]
        """
        monom, ns = self.monom, self.ns
        for m, coeff in self.poly.iterterms():
            mq = self._monom_fromint(m, monom, ns)
            yield mq, coeff

    def terms(self) -> list[tuple[tuple[Any, ...], Any]]:
        """Return a list of the terms of a Puiseux polynomial."""
        return list(self.iterterms())

    @property
    def is_term(self) -> bool:
        """Return True if the Puiseux polynomial is a single term."""
        return self.poly.is_term

    def to_dict(self) -> dict[tuple[int, ...], Any]:
        """Return a dictionary representation of a Puiseux polynomial."""
        return dict(self.iterterms())

    @classmethod
    def from_dict(
        cls, terms: dict[tuple[Any, ...], Any], ring: PuiseuxRing
    ) -> PuiseuxPoly:
        """Create a Puiseux polynomial from a dictionary of terms.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring, PuiseuxPoly
        >>> R, x = puiseux_ring('x', QQ)
        >>> PuiseuxPoly.from_dict({(QQ(1,2),): QQ(3)}, R)
        3*x**(1/2)
        >>> R.from_dict({(QQ(1,2),): QQ(3)})
        3*x**(1/2)
        """
        ns = [1] * ring.ngens
        mon = [0] * ring.ngens
        for mo in terms:
            ns = [lcm(n, m.denominator) for n, m in zip(ns, mo)]
            mon = [min(m, n) for m, n in zip(mo, mon)]

        if not any(mon):
            monom = None
        else:
            monom = tuple(-int((m * n).numerator) for m, n in zip(mon, ns))

        if all(n == 1 for n in ns):
            ns_final = None
        else:
            ns_final = tuple(ns)

        terms_p = {cls._monom_toint(m, monom, ns_final): coeff for m, coeff in terms.items()}

        poly = ring.poly_ring.from_dict(terms_p)

        return cls._new(ring, poly, monom, ns_final)

    def as_expr(self) -> Expr:
        """Convert a Puiseux polynomial to :class:`~sympy.core.expr.Expr`.

        >>> from sympy import QQ, Expr
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x = puiseux_ring('x', QQ)
        >>> p = 5*x**2 + 7*x**3
        >>> p.as_expr()
        7*x**3 + 5*x**2
        >>> isinstance(_, Expr)
        True
        """
        ring = self.ring
        dom = ring.domain
        symbols = ring.symbols
        terms = []
        for monom, coeff in self.iterterms():
            coeff_expr = dom.to_sympy(coeff)
            monoms_expr = []
            for i, m in enumerate(monom):
                monoms_expr.append(symbols[i] ** m)
            terms.append(Mul(coeff_expr, *monoms_expr))
        return Add(*terms)

    def __repr__(self) -> str:

        def format_power(base: str, exp: int) -> str:
            if exp == 1:
                return base
            elif exp >= 0 and int(exp) == exp:
                return f"{base}**{exp}"
            else:
                return f"{base}**({exp})"

        ring = self.ring
        dom = ring.domain

        syms = [str(s) for s in ring.symbols]
        terms_str = []
        for monom, coeff in sorted(self.terms()):
            monom_str = "*".join(format_power(s, e) for s, e in zip(syms, monom) if e)
            if coeff == dom.one:
                if monom_str:
                    terms_str.append(monom_str)
                else:
                    terms_str.append("1")
            elif not monom_str:
                terms_str.append(str(coeff))
            else:
                terms_str.append(f"{coeff}*{monom_str}")

        return " + ".join(terms_str)

    def _unify(
        self, other: PuiseuxPoly
    ) -> tuple[
        PolyElement, PolyElement, tuple[int, ...] | None, tuple[int, ...] | None
    ]:
        """Bring two Puiseux polynomials to a common monom and ns."""
        poly1, monom1, ns1 = self.poly, self.monom, self.ns
        poly2, monom2, ns2 = other.poly, other.monom, other.ns

        if monom1 == monom2 and ns1 == ns2:
            return poly1, poly2, monom1, ns1

        if ns1 == ns2:
            ns = ns1
        elif ns1 is not None and ns2 is not None:
            ns = tuple(lcm(n1, n2) for n1, n2 in zip(ns1, ns2))
            f1 = [n // n1 for n, n1 in zip(ns, ns1)]
            f2 = [n // n2 for n, n2 in zip(ns, ns2)]
            poly1 = poly1.inflate(f1)
            poly2 = poly2.inflate(f2)
            if monom1 is not None:
                monom1 = tuple(m * f for m, f in zip(monom1, f1))
            if monom2 is not None:
                monom2 = tuple(m * f for m, f in zip(monom2, f2))
        elif ns2 is not None:
            ns = ns2
            poly1 = poly1.inflate(ns)
            if monom1 is not None:
                monom1 = tuple(m * n for m, n in zip(monom1, ns))
        elif ns1 is not None:
            ns = ns1
            poly2 = poly2.inflate(ns)
            if monom2 is not None:
                monom2 = tuple(m * n for m, n in zip(monom2, ns))
        else:
            assert False

        if monom1 == monom2:
            monom = monom1
        elif monom1 is not None and monom2 is not None:
            monom = tuple(max(m1, m2) for m1, m2 in zip(monom1, monom2))
            poly1 = _mul_poly_monom(poly1, _div_monom(monom, monom1))
            poly2 = _mul_poly_monom(poly2, _div_monom(monom, monom2))
        elif monom2 is not None:
            monom = monom2
            poly1 = _mul_poly_monom(poly1, monom2)
        elif monom1 is not None:
            monom = monom1
            poly2 = _mul_poly_monom(poly2, monom1)
        else:
            assert False

        return poly1, poly2, monom, ns

    def __pos__(self) -> PuiseuxPoly:
        return self

    def __neg__(self) -> PuiseuxPoly:
        return self._new_raw(self.ring, -self.poly, self.monom, self.ns)

    def __add__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, PuiseuxPoly):
            if self.ring != other.ring:
                raise ValueError("Cannot add Puiseux polynomials from different rings")
            return self._add(other)
        domain = self.ring.domain
        if isinstance(other, int):
            return self._add_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._add_ground(other)
        else:
            return NotImplemented

    def __radd__(self, other: Any) -> PuiseuxPoly:
        domain = self.ring.domain
        if isinstance(other, int):
            return self._add_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._add_ground(other)
        else:
            return NotImplemented

    def __sub__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, PuiseuxPoly):
            if self.ring != other.ring:
                raise ValueError(
                    "Cannot subtract Puiseux polynomials from different rings"
                )
            return self._sub(other)
        domain = self.ring.domain
        if isinstance(other, int):
            return self._sub_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._sub_ground(other)
        else:
            return NotImplemented

    def __rsub__(self, other: Any) -> PuiseuxPoly:
        domain = self.ring.domain
        if isinstance(other, int):
            return self._rsub_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._rsub_ground(other)
        else:
            return NotImplemented

    def __mul__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, PuiseuxPoly):
            if self.ring != other.ring:
                raise ValueError(
                    "Cannot multiply Puiseux polynomials from different rings"
                )
            return self._mul(other)
        domain = self.ring.domain
        if isinstance(other, int):
            return self._mul_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._mul_ground(other)
        else:
            return NotImplemented

    def __rmul__(self, other: Any) -> PuiseuxPoly:
        domain = self.ring.domain
        if isinstance(other, int):
            return self._mul_ground(domain.convert_from(QQ(other), QQ))
        elif domain.of_type(other):
            return self._mul_ground(other)
        else:
            return NotImplemented

    def __pow__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, int):
            if other >= 0:
                return self._pow_pint(other)
            else:
                return self._pow_nint(-other)
        elif QQ.of_type(other):
            return self._pow_rational(other)
        else:
            return NotImplemented

    def __truediv__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, PuiseuxPoly):
            if self.ring != other.ring:
                raise ValueError(
                    "Cannot divide Puiseux polynomials from different rings"
                )
            return self._mul(other._inv())
        domain = self.ring.domain
        if isinstance(other, int):
            return self._mul_ground(domain.convert_from(QQ(1, other), QQ))
        elif domain.of_type(other):
            return self._div_ground(other)
        else:
            return NotImplemented

    def __rtruediv__(self, other: Any) -> PuiseuxPoly:
        if isinstance(other, int):
            return self._inv()._mul_ground(self.ring.domain.convert_from(QQ(other), QQ))
        elif self.ring.domain.of_type(other):
            return self._inv()._mul_ground(other)
        else:
            return NotImplemented

    def _add(self, other: PuiseuxPoly) -> PuiseuxPoly:
        poly1, poly2, monom, ns = self._unify(other)
        return self._new(self.ring, poly1 + poly2, monom, ns)

    def _add_ground(self, ground: Any) -> PuiseuxPoly:
        return self._add(self.ring.ground_new(ground))

    def _sub(self, other: PuiseuxPoly) -> PuiseuxPoly:
        poly1, poly2, monom, ns = self._unify(other)
        return self._new(self.ring, poly1 - poly2, monom, ns)

    def _sub_ground(self, ground: Any) -> PuiseuxPoly:
        return self._sub(self.ring.ground_new(ground))

    def _rsub_ground(self, ground: Any) -> PuiseuxPoly:
        return self.ring.ground_new(ground)._sub(self)

    def _mul(self, other: PuiseuxPoly) -> PuiseuxPoly:
        poly1, poly2, monom, ns = self._unify(other)
        if monom is not None:
            monom = tuple(2 * e for e in monom)
        return self._new(self.ring, poly1 * poly2, monom, ns)

    def _mul_ground(self, ground: Any) -> PuiseuxPoly:
        return self._new_raw(self.ring, self.poly * ground, self.monom, self.ns)

    def _div_ground(self, ground: Any) -> PuiseuxPoly:
        return self._new_raw(self.ring, self.poly / ground, self.monom, self.ns)

    def _pow_pint(self, n: int) -> PuiseuxPoly:
        assert n >= 0
        monom = self.monom
        if monom is not None:
            monom = tuple(m * n for m in monom)
        return self._new(self.ring, self.poly**n, monom, self.ns)

    def _pow_nint(self, n: int) -> PuiseuxPoly:
        return self._inv()._pow_pint(n)

    def _pow_rational(self, n: Any) -> PuiseuxPoly:
        if not self.is_term:
            raise ValueError("Only monomials can be raised to a rational power")
        [(monom, coeff)] = self.terms()
        domain = self.ring.domain
        if not domain.is_one(coeff):
            raise ValueError("Only monomials can be raised to a rational power")
        monom = tuple(m * n for m in monom)
        return self.ring.from_dict({monom: domain.one})

    def _inv(self) -> PuiseuxPoly:
        if not self.is_term:
            raise ValueError("Only terms can be inverted")
        [(monom, coeff)] = self.terms()
        domain = self.ring.domain
        if not domain.is_Field and not domain.is_one(coeff):
            raise ValueError("Cannot invert non-unit coefficient")
        monom = tuple(-m for m in monom)
        coeff = 1 / coeff
        return self.ring.from_dict({monom: coeff})

    def diff(self, x: PuiseuxPoly) -> PuiseuxPoly:
        """Differentiate a Puiseux polynomial with respect to a variable.

        >>> from sympy import QQ
        >>> from sympy.polys.puiseux import puiseux_ring
        >>> R, x, y = puiseux_ring('x, y', QQ)
        >>> p = 5*x**2 + 7*y**3
        >>> p.diff(x)
        10*x
        >>> p.diff(y)
        21*y**2
        """
        ring = self.ring
        i = ring.index(x)
        g = {}
        for expv, coeff in self.iterterms():
            n = expv[i]
            if n:
                e = list(expv)
                e[i] -= 1
                g[tuple(e)] = coeff * n
        return ring(g)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\sqfreetools.py ===
"""Square-free decomposition algorithms and related tools. """


from sympy.polys.densearith import (
    dup_neg, dmp_neg,
    dup_sub, dmp_sub,
    dup_mul, dmp_mul,
    dup_quo, dmp_quo,
    dup_mul_ground, dmp_mul_ground)
from sympy.polys.densebasic import (
    dup_strip,
    dup_LC, dmp_ground_LC,
    dmp_zero_p,
    dmp_ground,
    dup_degree, dmp_degree, dmp_degree_in, dmp_degree_list,
    dmp_raise, dmp_inject,
    dup_convert)
from sympy.polys.densetools import (
    dup_diff, dmp_diff, dmp_diff_in,
    dup_shift, dmp_shift,
    dup_monic, dmp_ground_monic,
    dup_primitive, dmp_ground_primitive)
from sympy.polys.euclidtools import (
    dup_inner_gcd, dmp_inner_gcd,
    dup_gcd, dmp_gcd,
    dmp_resultant, dmp_primitive)
from sympy.polys.galoistools import (
    gf_sqf_list, gf_sqf_part)
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    DomainError)


def _dup_check_degrees(f, result):
    """Sanity check the degrees of a computed factorization in K[x]."""
    deg = sum(k * dup_degree(fac) for (fac, k) in result)
    assert deg == dup_degree(f)


def _dmp_check_degrees(f, u, result):
    """Sanity check the degrees of a computed factorization in K[X]."""
    degs = [0] * (u + 1)
    for fac, k in result:
        degs_fac = dmp_degree_list(fac, u)
        degs = [d1 + k * d2 for d1, d2 in zip(degs, degs_fac)]
    assert tuple(degs) == dmp_degree_list(f, u)


def dup_sqf_p(f, K):
    """
    Return ``True`` if ``f`` is a square-free polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_p(x**2 - 2*x + 1)
    False
    >>> R.dup_sqf_p(x**2 - 1)
    True

    """
    if not f:
        return True
    else:
        return not dup_degree(dup_gcd(f, dup_diff(f, 1, K), K))


def dmp_sqf_p(f, u, K):
    """
    Return ``True`` if ``f`` is a square-free polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqf_p(x**2 + 2*x*y + y**2)
    False
    >>> R.dmp_sqf_p(x**2 + y**2)
    True

    """
    if dmp_zero_p(f, u):
        return True

    for i in range(u+1):

        fp = dmp_diff_in(f, 1, i, u, K)

        if dmp_zero_p(fp, u):
            continue

        gcd = dmp_gcd(f, fp, u, K)

        if dmp_degree_in(gcd, i, u) != 0:
            return False

    return True


def dup_sqf_norm(f, K):
    r"""
    Find a shift of `f` in `K[x]` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x)=f(x-sa)`, `r(x)=\text{Norm}(g(x))` and
    `r` is a square-free polynomial over `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\mathbb{Q}(\sqrt{3})`
    and rings `K[x]` and `k[x]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import sqrt

    >>> K = QQ.algebraic_field(sqrt(3))
    >>> R, x = ring("x", K)
    >>> _, X = ring("x", QQ)

    We can now find a square free norm for a shift of `f`:

    >>> f = x**2 - 1
    >>> s, g, r = R.dup_sqf_norm(f)

    The choice of shift `s` is arbitrary and the particular values returned for
    `g` and `r` are determined by `s`.

    >>> s == 1
    True
    >>> g == x**2 - 2*sqrt(3)*x + 2
    True
    >>> r == X**4 - 8*X**2 + 4
    True

    The invariants are:

    >>> g == f.shift(-s*K.unit)
    True
    >>> g.norm() == r
    True
    >>> r.is_squarefree
    True

    Explanation
    ===========

    This is part of Trager's algorithm for factorizing polynomials over
    algebraic number fields. In particular this function is algorithm
    ``sqfr_norm`` from [Trager76]_.

    See Also
    ========

    dmp_sqf_norm:
        Analogous function for multivariate polynomials over ``k(a)``.
    dmp_norm:
        Computes the norm of `f` directly without any shift.
    dup_ext_factor:
        Function implementing Trager's algorithm that uses this.
    sympy.polys.polytools.sqf_norm:
        High-level interface for using this function.
    """
    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    s, g = 0, dmp_raise(K.mod.to_list(), 1, 0, K.dom)

    while True:
        h, _ = dmp_inject(f, 0, K, front=True)
        r = dmp_resultant(g, h, 1, K.dom)

        if dup_sqf_p(r, K.dom):
            break
        else:
            f, s = dup_shift(f, -K.unit, K), s + 1

    return s, f, r


def _dmp_sqf_norm_shifts(f, u, K):
    """Generate a sequence of candidate shifts for dmp_sqf_norm."""
    #
    # We want to find a minimal shift if possible because shifting high degree
    # variables can be expensive e.g. x**10 -> (x + 1)**10. We try a few easy
    # cases first before the final infinite loop that is guaranteed to give
    # only finitely many bad shifts (see Trager76 for proof of this in the
    # univariate case).
    #

    # First the trivial shift [0, 0, ...]
    n = u + 1
    s0 = [0] * n
    yield s0, f

    # Shift in multiples of the generator of the extension field K
    a = K.unit

    # Variables of degree > 0 ordered by increasing degree
    d = dmp_degree_list(f, u)
    var_indices = [i for di, i in sorted(zip(d, range(u+1))) if di > 0]

    # Now try [1, 0, 0, ...], [0, 1, 0, ...]
    for i in var_indices:
        s1 = s0.copy()
        s1[i] = 1
        a1 = [-a*s1i for s1i in s1]
        f1 = dmp_shift(f, a1, u, K)
        yield s1, f1

    # Now try [1, 1, 1, ...], [2, 2, 2, ...]
    j = 0
    while True:
        j += 1
        sj = [j] * n
        aj = [-a*j] * n
        fj = dmp_shift(f, aj, u, K)
        yield sj, fj


def dmp_sqf_norm(f, u, K):
    r"""
    Find a shift of ``f`` in ``K[X]`` that has square-free norm.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Returns `(s,g,r)`, such that `g(x_1,x_2,\cdots)=f(x_1-s_1 a, x_2 - s_2 a,
    \cdots)`, `r(x)=\text{Norm}(g(x))` and `r` is a square-free polynomial over
    `k`.

    Examples
    ========

    We first create the algebraic number field `K=k(a)=\mathbb{Q}(i)` and rings
    `K[x,y]` and `k[x,y]`:

    >>> from sympy.polys import ring, QQ
    >>> from sympy import I

    >>> K = QQ.algebraic_field(I)
    >>> R, x, y = ring("x,y", K)
    >>> _, X, Y = ring("x,y", QQ)

    We can now find a square free norm for a shift of `f`:

    >>> f = x*y + y**2
    >>> s, g, r = R.dmp_sqf_norm(f)

    The choice of shifts ``s`` is arbitrary and the particular values returned
    for ``g`` and ``r`` are determined by ``s``.

    >>> s
    [0, 1]
    >>> g == x*y - I*x + y**2 - 2*I*y - 1
    True
    >>> r == X**2*Y**2 + X**2 + 2*X*Y**3 + 2*X*Y + Y**4 + 2*Y**2 + 1
    True

    The required invariants are:

    >>> g == f.shift_list([-si*K.unit for si in s])
    True
    >>> g.norm() == r
    True
    >>> r.is_squarefree
    True

    Explanation
    ===========

    This is part of Trager's algorithm for factorizing polynomials over
    algebraic number fields. In particular this function is a multivariate
    generalization of algorithm ``sqfr_norm`` from [Trager76]_.

    See Also
    ========

    dup_sqf_norm:
        Analogous function for univariate polynomials over ``k(a)``.
    dmp_norm:
        Computes the norm of `f` directly without any shift.
    dmp_ext_factor:
        Function implementing Trager's algorithm that uses this.
    sympy.polys.polytools.sqf_norm:
        High-level interface for using this function.
    """
    if not u:
        s, g, r = dup_sqf_norm(f, K)
        return [s], g, r

    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    g = dmp_raise(K.mod.to_list(), u + 1, 0, K.dom)

    for s, f in _dmp_sqf_norm_shifts(f, u, K):

        h, _ = dmp_inject(f, u, K, front=True)
        r = dmp_resultant(g, h, u + 1, K.dom)

        if dmp_sqf_p(r, u, K.dom):
            break

    return s, f, r


def dmp_norm(f, u, K):
    r"""
    Norm of ``f`` in ``K[X]``, often not square-free.

    The domain `K` must be an algebraic number field `k(a)` (see :ref:`QQ(a)`).

    Examples
    ========

    We first define the algebraic number field `K = k(a) = \mathbb{Q}(\sqrt{2})`:

    >>> from sympy import QQ, sqrt
    >>> from sympy.polys.sqfreetools import dmp_norm
    >>> k = QQ
    >>> K = k.algebraic_field(sqrt(2))

    We can now compute the norm of a polynomial `p` in `K[x,y]`:

    >>> p = [[K(1)], [K(1),K.unit]]                  # x + y + sqrt(2)
    >>> N = [[k(1)], [k(2),k(0)], [k(1),k(0),k(-2)]] # x**2 + 2*x*y + y**2 - 2
    >>> dmp_norm(p, 1, K) == N
    True

    In higher level functions that is:

    >>> from sympy import expand, roots, minpoly
    >>> from sympy.abc import x, y
    >>> from math import prod
    >>> a = sqrt(2)
    >>> e = (x + y + a)
    >>> e.as_poly([x, y], extension=a).norm()
    Poly(x**2 + 2*x*y + y**2 - 2, x, y, domain='QQ')

    This is equal to the product of the expressions `x + y + a_i` where the
    `a_i` are the conjugates of `a`:

    >>> pa = minpoly(a)
    >>> pa
    _x**2 - 2
    >>> rs = roots(pa, multiple=True)
    >>> rs
    [sqrt(2), -sqrt(2)]
    >>> n = prod(e.subs(a, r) for r in rs)
    >>> n
    (x + y - sqrt(2))*(x + y + sqrt(2))
    >>> expand(n)
    x**2 + 2*x*y + y**2 - 2

    Explanation
    ===========

    Given an algebraic number field `K = k(a)` any element `b` of `K` can be
    represented as polynomial function `b=g(a)` where `g` is in `k[x]`. If the
    minimal polynomial of `a` over `k` is `p_a` then the roots `a_1`, `a_2`,
    `\cdots` of `p_a(x)` are the conjugates of `a`. The norm of `b` is the
    product `g(a1) \times g(a2) \times \cdots` and is an element of `k`.

    As in [Trager76]_ we extend this norm to multivariate polynomials over `K`.
    If `b(x)` is a polynomial in `k(a)[X]` then we can think of `b` as being
    alternately a function `g_X(a)` where `g_X` is an element of `k[X][y]` i.e.
    a polynomial function with coefficients that are elements of `k[X]`. Then
    the norm of `b` is the product `g_X(a1) \times g_X(a2) \times \cdots` and
    will be an element of `k[X]`.

    See Also
    ========

    dmp_sqf_norm:
        Compute a shift of `f` so that the `\text{Norm}(f)` is square-free.
    sympy.polys.polytools.Poly.norm:
        Higher-level function that calls this.
    """
    if not K.is_Algebraic:
        raise DomainError("ground domain must be algebraic")

    g = dmp_raise(K.mod.to_list(), u + 1, 0, K.dom)
    h, _ = dmp_inject(f, u, K, front=True)

    return dmp_resultant(g, h, u + 1, K.dom)


def dup_gf_sqf_part(f, K):
    """Compute square-free part of ``f`` in ``GF(p)[x]``. """
    f = dup_convert(f, K, K.dom)
    g = gf_sqf_part(f, K.mod, K.dom)
    return dup_convert(g, K.dom, K)


def dmp_gf_sqf_part(f, u, K):
    """Compute square-free part of ``f`` in ``GF(p)[X]``. """
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_sqf_part(f, K):
    """
    Returns square-free part of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_sqf_part(x**3 - 3*x - 2)
    x**2 - x - 2

    See Also
    ========

    sympy.polys.polytools.Poly.sqf_part
    """
    if K.is_FiniteField:
        return dup_gf_sqf_part(f, K)

    if not f:
        return f

    if K.is_negative(dup_LC(f, K)):
        f = dup_neg(f, K)

    gcd = dup_gcd(f, dup_diff(f, 1, K), K)
    sqf = dup_quo(f, gcd, K)

    if K.is_Field:
        return dup_monic(sqf, K)
    else:
        return dup_primitive(sqf, K)[1]


def dmp_sqf_part(f, u, K):
    """
    Returns square-free part of a polynomial in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> R.dmp_sqf_part(x**3 + 2*x**2*y + x*y**2)
    x**2 + x*y

    """
    if not u:
        return dup_sqf_part(f, K)

    if K.is_FiniteField:
        return dmp_gf_sqf_part(f, u, K)

    if dmp_zero_p(f, u):
        return f

    if K.is_negative(dmp_ground_LC(f, u, K)):
        f = dmp_neg(f, u, K)

    gcd = f
    for i in range(u+1):
        gcd = dmp_gcd(gcd, dmp_diff_in(f, 1, i, u, K), u, K)
    sqf = dmp_quo(f, gcd, u, K)

    if K.is_Field:
        return dmp_ground_monic(sqf, u, K)
    else:
        return dmp_ground_primitive(sqf, u, K)[1]


def dup_gf_sqf_list(f, K, all=False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[x]``. """
    f_orig = f

    f = dup_convert(f, K, K.dom)

    coeff, factors = gf_sqf_list(f, K.mod, K.dom, all=all)

    for i, (f, k) in enumerate(factors):
        factors[i] = (dup_convert(f, K.dom, K), k)

    _dup_check_degrees(f_orig, factors)

    return K.convert(coeff, K.dom), factors


def dmp_gf_sqf_list(f, u, K, all=False):
    """Compute square-free decomposition of ``f`` in ``GF(p)[X]``. """
    raise NotImplementedError('multivariate polynomials over finite fields')


def dup_sqf_list(f, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Uses Yun's algorithm from [Yun76]_.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list(f)
    (2, [(x + 1, 2), (x + 2, 3)])
    >>> R.dup_sqf_list(f, all=True)
    (2, [(1, 1), (x + 1, 2), (x + 2, 3)])

    See Also
    ========

    dmp_sqf_list:
        Corresponding function for multivariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.

    References
    ==========

    [Yun76]_
    """
    if K.is_FiniteField:
        return dup_gf_sqf_list(f, K, all=all)

    f_orig = f

    if K.is_Field:
        coeff = dup_LC(f, K)
        f = dup_monic(f, K)
    else:
        coeff, f = dup_primitive(f, K)

        if K.is_negative(dup_LC(f, K)):
            f = dup_neg(f, K)
            coeff = -coeff

    if dup_degree(f) <= 0:
        return coeff, []

    result, i = [], 1

    h = dup_diff(f, 1, K)
    g, p, q = dup_inner_gcd(f, h, K)

    while True:
        d = dup_diff(p, 1, K)
        h = dup_sub(q, d, K)

        if not h:
            result.append((p, i))
            break

        g, p, q = dup_inner_gcd(p, h, K)

        if all or dup_degree(g) > 0:
            result.append((g, i))

        i += 1

    _dup_check_degrees(f_orig, result)

    return coeff, result


def dup_sqf_list_include(f, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> f = 2*x**5 + 16*x**4 + 50*x**3 + 76*x**2 + 56*x + 16

    >>> R.dup_sqf_list_include(f)
    [(2, 1), (x + 1, 2), (x + 2, 3)]
    >>> R.dup_sqf_list_include(f, all=True)
    [(2, 1), (x + 1, 2), (x + 2, 3)]

    """
    coeff, factors = dup_sqf_list(f, K, all=all)

    if factors and factors[0][1] == 1:
        g = dup_mul_ground(factors[0][0], coeff, K)
        return [(g, 1)] + factors[1:]
    else:
        g = dup_strip([coeff])
        return [(g, 1)] + factors


def dmp_sqf_list(f, u, K, all=False):
    """
    Return square-free decomposition of a polynomial in `K[X]`.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list(f)
    (1, [(x + y, 2), (x, 3)])
    >>> R.dmp_sqf_list(f, all=True)
    (1, [(1, 1), (x + y, 2), (x, 3)])

    Explanation
    ===========

    Uses Yun's algorithm for univariate polynomials from [Yun76]_ recursively.
    The multivariate polynomial is treated as a univariate polynomial in its
    leading variable. Then Yun's algorithm computes the square-free
    factorization of the primitive and the content is factored recursively.

    It would be better to use a dedicated algorithm for multivariate
    polynomials instead.

    See Also
    ========

    dup_sqf_list:
        Corresponding function for univariate polynomials.
    sympy.polys.polytools.sqf_list:
        High-level function for square-free factorization of expressions.
    sympy.polys.polytools.Poly.sqf_list:
        Analogous method on :class:`~.Poly`.
    """
    if not u:
        return dup_sqf_list(f, K, all=all)

    if K.is_FiniteField:
        return dmp_gf_sqf_list(f, u, K, all=all)

    f_orig = f

    if K.is_Field:
        coeff = dmp_ground_LC(f, u, K)
        f = dmp_ground_monic(f, u, K)
    else:
        coeff, f = dmp_ground_primitive(f, u, K)

        if K.is_negative(dmp_ground_LC(f, u, K)):
            f = dmp_neg(f, u, K)
            coeff = -coeff

    deg = dmp_degree(f, u)
    if deg < 0:
        return coeff, []

    # Yun's algorithm requires the polynomial to be primitive as a univariate
    # polynomial in its main variable.
    content, f = dmp_primitive(f, u, K)

    result = {}

    if deg != 0:

        h = dmp_diff(f, 1, u, K)
        g, p, q = dmp_inner_gcd(f, h, u, K)

        i = 1

        while True:
            d = dmp_diff(p, 1, u, K)
            h = dmp_sub(q, d, u, K)

            if dmp_zero_p(h, u):
                result[i] = p
                break

            g, p, q = dmp_inner_gcd(p, h, u, K)

            if all or dmp_degree(g, u) > 0:
                result[i] = g

            i += 1

    coeff_content, result_content = dmp_sqf_list(content, u-1, K, all=all)

    coeff *= coeff_content

    # Combine factors of the content and primitive part that have the same
    # multiplicity to produce a list in ascending order of multiplicity.
    for fac, i in result_content:
        fac = [fac]
        if i in result:
            result[i] = dmp_mul(result[i], fac, u, K)
        else:
            result[i] = fac

    result = [(result[i], i) for i in sorted(result)]

    _dmp_check_degrees(f_orig, u, result)

    return coeff, result


def dmp_sqf_list_include(f, u, K, all=False):
    """
    Return square-free decomposition of a polynomial in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    >>> f = x**5 + 2*x**4*y + x**3*y**2

    >>> R.dmp_sqf_list_include(f)
    [(1, 1), (x + y, 2), (x, 3)]
    >>> R.dmp_sqf_list_include(f, all=True)
    [(1, 1), (x + y, 2), (x, 3)]

    """
    if not u:
        return dup_sqf_list_include(f, K, all=all)

    coeff, factors = dmp_sqf_list(f, u, K, all=all)

    if factors and factors[0][1] == 1:
        g = dmp_mul_ground(factors[0][0], coeff, u, K)
        return [(g, 1)] + factors[1:]
    else:
        g = dmp_ground(coeff, u)
        return [(g, 1)] + factors


def dup_gff_list(f, K):
    """
    Compute greatest factorial factorization of ``f`` in ``K[x]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x = ring("x", ZZ)

    >>> R.dup_gff_list(x**5 + 2*x**4 - x**3 - 2*x**2)
    [(x, 1), (x + 2, 4)]

    """
    if not f:
        raise ValueError("greatest factorial factorization doesn't exist for a zero polynomial")

    f = dup_monic(f, K)

    if not dup_degree(f):
        return []
    else:
        g = dup_gcd(f, dup_shift(f, K.one, K), K)
        H = dup_gff_list(g, K)

        for i, (h, k) in enumerate(H):
            g = dup_mul(g, dup_shift(h, -K(k), K), K)
            H[i] = (h, k + 1)

        f = dup_quo(f, g, K)

        if not dup_degree(f):
            return H
        else:
            return [(f, 1)] + H


def dmp_gff_list(f, u, K):
    """
    Compute greatest factorial factorization of ``f`` in ``K[X]``.

    Examples
    ========

    >>> from sympy.polys import ring, ZZ
    >>> R, x,y = ring("x,y", ZZ)

    """
    if not u:
        return dup_gff_list(f, K)
    else:
        raise MultivariatePolynomialError(f)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\install.py ===
import errno
import json
import operator
import os
import shutil
import site
from optparse import SUPPRESS_HELP, Values
from typing import List, Optional

from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.requests.exceptions import InvalidProxyURL
from pip._vendor.rich import print_json

# Eagerly import self_outdated_check to avoid crashes. Otherwise,
# this module would be imported *after* pip was replaced, resulting
# in crashes if the new self_outdated_check module was incompatible
# with the rest of pip that's already imported, or allowing a
# wheel to execute arbitrary code on install by replacing
# self_outdated_check.
import pip._internal.self_outdated_check  # noqa: F401
from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.req_command import (
    RequirementCommand,
    with_cleanup,
)
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.exceptions import CommandError, InstallationError
from pip._internal.locations import get_scheme
from pip._internal.metadata import get_environment
from pip._internal.models.installation_report import InstallationReport
from pip._internal.operations.build.build_tracker import get_build_tracker
from pip._internal.operations.check import ConflictDetails, check_install_conflicts
from pip._internal.req import install_given_reqs
from pip._internal.req.req_install import (
    InstallRequirement,
    check_legacy_setup_py_options,
)
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.filesystem import test_writable_dir
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import (
    check_externally_managed,
    ensure_dir,
    get_pip_version,
    protect_pip_from_modification_on_windows,
    warn_if_run_as_root,
    write_output,
)
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.virtualenv import (
    running_under_virtualenv,
    virtualenv_no_global,
)
from pip._internal.wheel_builder import build, should_build_for_install_command

logger = getLogger(__name__)


class InstallCommand(RequirementCommand):
    """
    Install packages from:

    - PyPI (and other indexes) using requirement specifiers.
    - VCS project urls.
    - Local project directories.
    - Local or remote source archives.

    pip also supports installing from "requirements files", which provide
    an easy way to specify a whole environment to be installed.
    """

    usage = """
      %prog [options] <requirement specifier> [package-index-options] ...
      %prog [options] -r <requirements file> [package-index-options] ...
      %prog [options] [-e] <vcs project url> ...
      %prog [options] [-e] <local project path> ...
      %prog [options] <archive url/path> ..."""

    def add_options(self) -> None:
        self.cmd_opts.add_option(cmdoptions.requirements())
        self.cmd_opts.add_option(cmdoptions.constraints())
        self.cmd_opts.add_option(cmdoptions.no_deps())
        self.cmd_opts.add_option(cmdoptions.pre())

        self.cmd_opts.add_option(cmdoptions.editable())
        self.cmd_opts.add_option(
            "--dry-run",
            action="store_true",
            dest="dry_run",
            default=False,
            help=(
                "Don't actually install anything, just print what would be. "
                "Can be used in combination with --ignore-installed "
                "to 'resolve' the requirements."
            ),
        )
        self.cmd_opts.add_option(
            "-t",
            "--target",
            dest="target_dir",
            metavar="dir",
            default=None,
            help=(
                "Install packages into <dir>. "
                "By default this will not replace existing files/folders in "
                "<dir>. Use --upgrade to replace existing packages in <dir> "
                "with new versions."
            ),
        )
        cmdoptions.add_target_python_options(self.cmd_opts)

        self.cmd_opts.add_option(
            "--user",
            dest="use_user_site",
            action="store_true",
            help=(
                "Install to the Python user install directory for your "
                "platform. Typically ~/.local/, or %APPDATA%\\Python on "
                "Windows. (See the Python documentation for site.USER_BASE "
                "for full details.)"
            ),
        )
        self.cmd_opts.add_option(
            "--no-user",
            dest="use_user_site",
            action="store_false",
            help=SUPPRESS_HELP,
        )
        self.cmd_opts.add_option(
            "--root",
            dest="root_path",
            metavar="dir",
            default=None,
            help="Install everything relative to this alternate root directory.",
        )
        self.cmd_opts.add_option(
            "--prefix",
            dest="prefix_path",
            metavar="dir",
            default=None,
            help=(
                "Installation prefix where lib, bin and other top-level "
                "folders are placed. Note that the resulting installation may "
                "contain scripts and other resources which reference the "
                "Python interpreter of pip, and not that of ``--prefix``. "
                "See also the ``--python`` option if the intention is to "
                "install packages into another (possibly pip-free) "
                "environment."
            ),
        )

        self.cmd_opts.add_option(cmdoptions.src())

        self.cmd_opts.add_option(
            "-U",
            "--upgrade",
            dest="upgrade",
            action="store_true",
            help=(
                "Upgrade all specified packages to the newest available "
                "version. The handling of dependencies depends on the "
                "upgrade-strategy used."
            ),
        )

        self.cmd_opts.add_option(
            "--upgrade-strategy",
            dest="upgrade_strategy",
            default="only-if-needed",
            choices=["only-if-needed", "eager"],
            help=(
                "Determines how dependency upgrading should be handled "
                "[default: %default]. "
                '"eager" - dependencies are upgraded regardless of '
                "whether the currently installed version satisfies the "
                "requirements of the upgraded package(s). "
                '"only-if-needed" -  are upgraded only when they do not '
                "satisfy the requirements of the upgraded package(s)."
            ),
        )

        self.cmd_opts.add_option(
            "--force-reinstall",
            dest="force_reinstall",
            action="store_true",
            help="Reinstall all packages even if they are already up-to-date.",
        )

        self.cmd_opts.add_option(
            "-I",
            "--ignore-installed",
            dest="ignore_installed",
            action="store_true",
            help=(
                "Ignore the installed packages, overwriting them. "
                "This can break your system if the existing package "
                "is of a different version or was installed "
                "with a different package manager!"
            ),
        )

        self.cmd_opts.add_option(cmdoptions.ignore_requires_python())
        self.cmd_opts.add_option(cmdoptions.no_build_isolation())
        self.cmd_opts.add_option(cmdoptions.use_pep517())
        self.cmd_opts.add_option(cmdoptions.no_use_pep517())
        self.cmd_opts.add_option(cmdoptions.check_build_deps())
        self.cmd_opts.add_option(cmdoptions.override_externally_managed())

        self.cmd_opts.add_option(cmdoptions.config_settings())
        self.cmd_opts.add_option(cmdoptions.global_options())

        self.cmd_opts.add_option(
            "--compile",
            action="store_true",
            dest="compile",
            default=True,
            help="Compile Python source files to bytecode",
        )

        self.cmd_opts.add_option(
            "--no-compile",
            action="store_false",
            dest="compile",
            help="Do not compile Python source files to bytecode",
        )

        self.cmd_opts.add_option(
            "--no-warn-script-location",
            action="store_false",
            dest="warn_script_location",
            default=True,
            help="Do not warn when installing scripts outside PATH",
        )
        self.cmd_opts.add_option(
            "--no-warn-conflicts",
            action="store_false",
            dest="warn_about_conflicts",
            default=True,
            help="Do not warn about broken dependencies",
        )
        self.cmd_opts.add_option(cmdoptions.no_binary())
        self.cmd_opts.add_option(cmdoptions.only_binary())
        self.cmd_opts.add_option(cmdoptions.prefer_binary())
        self.cmd_opts.add_option(cmdoptions.require_hashes())
        self.cmd_opts.add_option(cmdoptions.progress_bar())
        self.cmd_opts.add_option(cmdoptions.root_user_action())

        index_opts = cmdoptions.make_option_group(
            cmdoptions.index_group,
            self.parser,
        )

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, self.cmd_opts)

        self.cmd_opts.add_option(
            "--report",
            dest="json_report_file",
            metavar="file",
            default=None,
            help=(
                "Generate a JSON file describing what pip did to install "
                "the provided requirements. "
                "Can be used in combination with --dry-run and --ignore-installed "
                "to 'resolve' the requirements. "
                "When - is used as file name it writes to stdout. "
                "When writing to stdout, please combine with the --quiet option "
                "to avoid mixing pip logging output with JSON output."
            ),
        )

    @with_cleanup
    def run(self, options: Values, args: List[str]) -> int:
        if options.use_user_site and options.target_dir is not None:
            raise CommandError("Can not combine '--user' and '--target'")

        # Check whether the environment we're installing into is externally
        # managed, as specified in PEP 668. Specifying --root, --target, or
        # --prefix disables the check, since there's no reliable way to locate
        # the EXTERNALLY-MANAGED file for those cases. An exception is also
        # made specifically for "--dry-run --report" for convenience.
        installing_into_current_environment = (
            not (options.dry_run and options.json_report_file)
            and options.root_path is None
            and options.target_dir is None
            and options.prefix_path is None
        )
        if (
            installing_into_current_environment
            and not options.override_externally_managed
        ):
            check_externally_managed()

        upgrade_strategy = "to-satisfy-only"
        if options.upgrade:
            upgrade_strategy = options.upgrade_strategy

        cmdoptions.check_dist_restriction(options, check_target=True)

        logger.verbose("Using %s", get_pip_version())
        options.use_user_site = decide_user_install(
            options.use_user_site,
            prefix_path=options.prefix_path,
            target_dir=options.target_dir,
            root_path=options.root_path,
            isolated_mode=options.isolated_mode,
        )

        target_temp_dir: Optional[TempDirectory] = None
        target_temp_dir_path: Optional[str] = None
        if options.target_dir:
            options.ignore_installed = True
            options.target_dir = os.path.abspath(options.target_dir)
            if (
                # fmt: off
                os.path.exists(options.target_dir) and
                not os.path.isdir(options.target_dir)
                # fmt: on
            ):
                raise CommandError(
                    "Target path exists but is not a directory, will not continue."
                )

            # Create a target directory for using with the target option
            target_temp_dir = TempDirectory(kind="target")
            target_temp_dir_path = target_temp_dir.path
            self.enter_context(target_temp_dir)

        global_options = options.global_options or []

        session = self.get_default_session(options)

        target_python = make_target_python(options)
        finder = self._build_package_finder(
            options=options,
            session=session,
            target_python=target_python,
            ignore_requires_python=options.ignore_requires_python,
        )
        build_tracker = self.enter_context(get_build_tracker())

        directory = TempDirectory(
            delete=not options.no_clean,
            kind="install",
            globally_managed=True,
        )

        try:
            reqs = self.get_requirements(args, options, finder, session)
            check_legacy_setup_py_options(options, reqs)

            wheel_cache = WheelCache(options.cache_dir)

            # Only when installing is it permitted to use PEP 660.
            # In other circumstances (pip wheel, pip download) we generate
            # regular (i.e. non editable) metadata and wheels.
            for req in reqs:
                req.permit_editable_wheels = True

            preparer = self.make_requirement_preparer(
                temp_build_dir=directory,
                options=options,
                build_tracker=build_tracker,
                session=session,
                finder=finder,
                use_user_site=options.use_user_site,
                verbosity=self.verbosity,
            )
            resolver = self.make_resolver(
                preparer=preparer,
                finder=finder,
                options=options,
                wheel_cache=wheel_cache,
                use_user_site=options.use_user_site,
                ignore_installed=options.ignore_installed,
                ignore_requires_python=options.ignore_requires_python,
                force_reinstall=options.force_reinstall,
                upgrade_strategy=upgrade_strategy,
                use_pep517=options.use_pep517,
                py_version_info=options.python_version,
            )

            self.trace_basic_info(finder)

            requirement_set = resolver.resolve(
                reqs, check_supported_wheels=not options.target_dir
            )

            if options.json_report_file:
                report = InstallationReport(requirement_set.requirements_to_install)
                if options.json_report_file == "-":
                    print_json(data=report.to_dict())
                else:
                    with open(options.json_report_file, "w", encoding="utf-8") as f:
                        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)

            if options.dry_run:
                would_install_items = sorted(
                    (r.metadata["name"], r.metadata["version"])
                    for r in requirement_set.requirements_to_install
                )
                if would_install_items:
                    write_output(
                        "Would install %s",
                        " ".join("-".join(item) for item in would_install_items),
                    )
                return SUCCESS

            try:
                pip_req = requirement_set.get_requirement("pip")
            except KeyError:
                modifying_pip = False
            else:
                # If we're not replacing an already installed pip,
                # we're not modifying it.
                modifying_pip = pip_req.satisfied_by is None
            protect_pip_from_modification_on_windows(modifying_pip=modifying_pip)

            reqs_to_build = [
                r
                for r in requirement_set.requirements_to_install
                if should_build_for_install_command(r)
            ]

            _, build_failures = build(
                reqs_to_build,
                wheel_cache=wheel_cache,
                verify=True,
                build_options=[],
                global_options=global_options,
            )

            if build_failures:
                raise InstallationError(
                    "Failed to build installable wheels for some "
                    "pyproject.toml based projects ({})".format(
                        ", ".join(r.name for r in build_failures)  # type: ignore
                    )
                )

            to_install = resolver.get_installation_order(requirement_set)

            # Check for conflicts in the package set we're installing.
            conflicts: Optional[ConflictDetails] = None
            should_warn_about_conflicts = (
                not options.ignore_dependencies and options.warn_about_conflicts
            )
            if should_warn_about_conflicts:
                conflicts = self._determine_conflicts(to_install)

            # Don't warn about script install locations if
            # --target or --prefix has been specified
            warn_script_location = options.warn_script_location
            if options.target_dir or options.prefix_path:
                warn_script_location = False

            installed = install_given_reqs(
                to_install,
                global_options,
                root=options.root_path,
                home=target_temp_dir_path,
                prefix=options.prefix_path,
                warn_script_location=warn_script_location,
                use_user_site=options.use_user_site,
                pycompile=options.compile,
                progress_bar=options.progress_bar,
            )

            lib_locations = get_lib_location_guesses(
                user=options.use_user_site,
                home=target_temp_dir_path,
                root=options.root_path,
                prefix=options.prefix_path,
                isolated=options.isolated_mode,
            )
            env = get_environment(lib_locations)

            # Display a summary of installed packages, with extra care to
            # display a package name as it was requested by the user.
            installed.sort(key=operator.attrgetter("name"))
            summary = []
            installed_versions = {}
            for distribution in env.iter_all_distributions():
                installed_versions[distribution.canonical_name] = distribution.version
            for package in installed:
                display_name = package.name
                version = installed_versions.get(canonicalize_name(display_name), None)
                if version:
                    text = f"{display_name}-{version}"
                else:
                    text = display_name
                summary.append(text)

            if conflicts is not None:
                self._warn_about_conflicts(
                    conflicts,
                    resolver_variant=self.determine_resolver_variant(options),
                )

            installed_desc = " ".join(summary)
            if installed_desc:
                write_output(
                    "Successfully installed %s",
                    installed_desc,
                )
        except OSError as error:
            show_traceback = self.verbosity >= 1

            message = create_os_error_message(
                error,
                show_traceback,
                options.use_user_site,
            )
            logger.error(message, exc_info=show_traceback)

            return ERROR

        if options.target_dir:
            assert target_temp_dir
            self._handle_target_dir(
                options.target_dir, target_temp_dir, options.upgrade
            )
        if options.root_user_action == "warn":
            warn_if_run_as_root()
        return SUCCESS

    def _handle_target_dir(
        self, target_dir: str, target_temp_dir: TempDirectory, upgrade: bool
    ) -> None:
        ensure_dir(target_dir)

        # Checking both purelib and platlib directories for installed
        # packages to be moved to target directory
        lib_dir_list = []

        # Checking both purelib and platlib directories for installed
        # packages to be moved to target directory
        scheme = get_scheme("", home=target_temp_dir.path)
        purelib_dir = scheme.purelib
        platlib_dir = scheme.platlib
        data_dir = scheme.data

        if os.path.exists(purelib_dir):
            lib_dir_list.append(purelib_dir)
        if os.path.exists(platlib_dir) and platlib_dir != purelib_dir:
            lib_dir_list.append(platlib_dir)
        if os.path.exists(data_dir):
            lib_dir_list.append(data_dir)

        for lib_dir in lib_dir_list:
            for item in os.listdir(lib_dir):
                if lib_dir == data_dir:
                    ddir = os.path.join(data_dir, item)
                    if any(s.startswith(ddir) for s in lib_dir_list[:-1]):
                        continue
                target_item_dir = os.path.join(target_dir, item)
                if os.path.exists(target_item_dir):
                    if not upgrade:
                        logger.warning(
                            "Target directory %s already exists. Specify "
                            "--upgrade to force replacement.",
                            target_item_dir,
                        )
                        continue
                    if os.path.islink(target_item_dir):
                        logger.warning(
                            "Target directory %s already exists and is "
                            "a link. pip will not automatically replace "
                            "links, please remove if replacement is "
                            "desired.",
                            target_item_dir,
                        )
                        continue
                    if os.path.isdir(target_item_dir):
                        shutil.rmtree(target_item_dir)
                    else:
                        os.remove(target_item_dir)

                shutil.move(os.path.join(lib_dir, item), target_item_dir)

    def _determine_conflicts(
        self, to_install: List[InstallRequirement]
    ) -> Optional[ConflictDetails]:
        try:
            return check_install_conflicts(to_install)
        except Exception:
            logger.exception(
                "Error while checking for conflicts. Please file an issue on "
                "pip's issue tracker: https://github.com/pypa/pip/issues/new"
            )
            return None

    def _warn_about_conflicts(
        self, conflict_details: ConflictDetails, resolver_variant: str
    ) -> None:
        package_set, (missing, conflicting) = conflict_details
        if not missing and not conflicting:
            return

        parts: List[str] = []
        if resolver_variant == "legacy":
            parts.append(
                "pip's legacy dependency resolver does not consider dependency "
                "conflicts when selecting packages. This behaviour is the "
                "source of the following dependency conflicts."
            )
        else:
            assert resolver_variant == "resolvelib"
            parts.append(
                "pip's dependency resolver does not currently take into account "
                "all the packages that are installed. This behaviour is the "
                "source of the following dependency conflicts."
            )

        # NOTE: There is some duplication here, with commands/check.py
        for project_name in missing:
            version = package_set[project_name][0]
            for dependency in missing[project_name]:
                message = (
                    f"{project_name} {version} requires {dependency[1]}, "
                    "which is not installed."
                )
                parts.append(message)

        for project_name in conflicting:
            version = package_set[project_name][0]
            for dep_name, dep_version, req in conflicting[project_name]:
                message = (
                    "{name} {version} requires {requirement}, but {you} have "
                    "{dep_name} {dep_version} which is incompatible."
                ).format(
                    name=project_name,
                    version=version,
                    requirement=req,
                    dep_name=dep_name,
                    dep_version=dep_version,
                    you=("you" if resolver_variant == "resolvelib" else "you'll"),
                )
                parts.append(message)

        logger.critical("\n".join(parts))


def get_lib_location_guesses(
    user: bool = False,
    home: Optional[str] = None,
    root: Optional[str] = None,
    isolated: bool = False,
    prefix: Optional[str] = None,
) -> List[str]:
    scheme = get_scheme(
        "",
        user=user,
        home=home,
        root=root,
        isolated=isolated,
        prefix=prefix,
    )
    return [scheme.purelib, scheme.platlib]


def site_packages_writable(root: Optional[str], isolated: bool) -> bool:
    return all(
        test_writable_dir(d)
        for d in set(get_lib_location_guesses(root=root, isolated=isolated))
    )


def decide_user_install(
    use_user_site: Optional[bool],
    prefix_path: Optional[str] = None,
    target_dir: Optional[str] = None,
    root_path: Optional[str] = None,
    isolated_mode: bool = False,
) -> bool:
    """Determine whether to do a user install based on the input options.

    If use_user_site is False, no additional checks are done.
    If use_user_site is True, it is checked for compatibility with other
    options.
    If use_user_site is None, the default behaviour depends on the environment,
    which is provided by the other arguments.
    """
    # In some cases (config from tox), use_user_site can be set to an integer
    # rather than a bool, which 'use_user_site is False' wouldn't catch.
    if (use_user_site is not None) and (not use_user_site):
        logger.debug("Non-user install by explicit request")
        return False

    if use_user_site:
        if prefix_path:
            raise CommandError(
                "Can not combine '--user' and '--prefix' as they imply "
                "different installation locations"
            )
        if virtualenv_no_global():
            raise InstallationError(
                "Can not perform a '--user' install. User site-packages "
                "are not visible in this virtualenv."
            )
        logger.debug("User install by explicit request")
        return True

    # If we are here, user installs have not been explicitly requested/avoided
    assert use_user_site is None

    # user install incompatible with --prefix/--target
    if prefix_path or target_dir:
        logger.debug("Non-user install due to --prefix or --target option")
        return False

    # If user installs are not enabled, choose a non-user install
    if not site.ENABLE_USER_SITE:
        logger.debug("Non-user install because user site-packages disabled")
        return False

    # If we have permission for a non-user install, do that,
    # otherwise do a user install.
    if site_packages_writable(root=root_path, isolated=isolated_mode):
        logger.debug("Non-user install because site-packages writeable")
        return False

    logger.info(
        "Defaulting to user installation because normal site-packages "
        "is not writeable"
    )
    return True


def create_os_error_message(
    error: OSError, show_traceback: bool, using_user_site: bool
) -> str:
    """Format an error message for an OSError

    It may occur anytime during the execution of the install command.
    """
    parts = []

    # Mention the error if we are not going to show a traceback
    parts.append("Could not install packages due to an OSError")
    if not show_traceback:
        parts.append(": ")
        parts.append(str(error))
    else:
        parts.append(".")

    # Spilt the error indication from a helper message (if any)
    parts[-1] += "\n"

    # Suggest useful actions to the user:
    #  (1) using user site-packages or (2) verifying the permissions
    if error.errno == errno.EACCES:
        user_option_part = "Consider using the `--user` option"
        permissions_part = "Check the permissions"

        if not running_under_virtualenv() and not using_user_site:
            parts.extend(
                [
                    user_option_part,
                    " or ",
                    permissions_part.lower(),
                ]
            )
        else:
            parts.append(permissions_part)
        parts.append(".\n")

    # Suggest to check "pip config debug" in case of invalid proxy
    if type(error) is InvalidProxyURL:
        parts.append(
            'Consider checking your local proxy configuration with "pip config debug"'
        )
        parts.append(".\n")

    # Suggest the user to enable Long Paths if path length is
    # more than 260
    if (
        WINDOWS
        and error.errno == errno.ENOENT
        and error.filename
        and len(error.filename) > 260
    ):
        parts.append(
            "HINT: This error might have occurred since "
            "this system does not have Windows Long Path "
            "support enabled. You can find information on "
            "how to enable this at "
            "https://pip.pypa.io/warnings/enable-long-paths\n"
        )

    return "".join(parts).strip() + "\n"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\indexed.py ===
r"""Module that defines indexed objects.

The classes ``IndexedBase``, ``Indexed``, and ``Idx`` represent a
matrix element ``M[i, j]`` as in the following diagram::

       1) The Indexed class represents the entire indexed object.
                  |
               ___|___
              '       '
               M[i, j]
              /   \__\______
              |             |
              |             |
              |     2) The Idx class represents indices; each Idx can
              |        optionally contain information about its range.
              |
        3) IndexedBase represents the 'stem' of an indexed object, here `M`.
           The stem used by itself is usually taken to represent the entire
           array.

There can be any number of indices on an Indexed object.  No
transformation properties are implemented in these Base objects, but
implicit contraction of repeated indices is supported.

Note that the support for complicated (i.e. non-atomic) integer
expressions as indices is limited.  (This should be improved in
future releases.)

Examples
========

To express the above matrix element example you would write:

>>> from sympy import symbols, IndexedBase, Idx
>>> M = IndexedBase('M')
>>> i, j = symbols('i j', cls=Idx)
>>> M[i, j]
M[i, j]

Repeated indices in a product implies a summation, so to express a
matrix-vector product in terms of Indexed objects:

>>> x = IndexedBase('x')
>>> M[i, j]*x[j]
M[i, j]*x[j]

If the indexed objects will be converted to component based arrays, e.g.
with the code printers or the autowrap framework, you also need to provide
(symbolic or numerical) dimensions.  This can be done by passing an
optional shape parameter to IndexedBase upon construction:

>>> dim1, dim2 = symbols('dim1 dim2', integer=True)
>>> A = IndexedBase('A', shape=(dim1, 2*dim1, dim2))
>>> A.shape
(dim1, 2*dim1, dim2)
>>> A[i, j, 3].shape
(dim1, 2*dim1, dim2)

If an IndexedBase object has no shape information, it is assumed that the
array is as large as the ranges of its indices:

>>> n, m = symbols('n m', integer=True)
>>> i = Idx('i', m)
>>> j = Idx('j', n)
>>> M[i, j].shape
(m, n)
>>> M[i, j].ranges
[(0, m - 1), (0, n - 1)]

The above can be compared with the following:

>>> A[i, 2, j].shape
(dim1, 2*dim1, dim2)
>>> A[i, 2, j].ranges
[(0, m - 1), None, (0, n - 1)]

To analyze the structure of indexed expressions, you can use the methods
get_indices() and get_contraction_structure():

>>> from sympy.tensor import get_indices, get_contraction_structure
>>> get_indices(A[i, j, j])
({i}, {})
>>> get_contraction_structure(A[i, j, j])
{(j,): {A[i, j, j]}}

See the appropriate docstrings for a detailed explanation of the output.
"""

#   TODO:  (some ideas for improvement)
#
#   o test and guarantee numpy compatibility
#      - implement full support for broadcasting
#      - strided arrays
#
#   o more functions to analyze indexed expressions
#      - identify standard constructs, e.g matrix-vector product in a subexpression
#
#   o functions to generate component based arrays (numpy and sympy.Matrix)
#      - generate a single array directly from Indexed
#      - convert simple sub-expressions
#
#   o sophisticated indexing (possibly in subclasses to preserve simplicity)
#      - Idx with range smaller than dimension of Indexed
#      - Idx with stepsize != 1
#      - Idx with step determined by function call
from collections.abc import Iterable

from sympy.core.numbers import Number
from sympy.core.assumptions import StdFactKB
from sympy.core import Expr, Tuple, sympify, S
from sympy.core.symbol import _filter_assumptions, Symbol
from sympy.core.logic import fuzzy_bool, fuzzy_not
from sympy.core.sympify import _sympify
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.multipledispatch import dispatch
from sympy.utilities.iterables import is_sequence, NotIterable
from sympy.utilities.misc import filldedent


class IndexException(Exception):
    pass


class Indexed(Expr):
    """Represents a mathematical object with indices.

    >>> from sympy import Indexed, IndexedBase, Idx, symbols
    >>> i, j = symbols('i j', cls=Idx)
    >>> Indexed('A', i, j)
    A[i, j]

    It is recommended that ``Indexed`` objects be created by indexing ``IndexedBase``:
    ``IndexedBase('A')[i, j]`` instead of ``Indexed(IndexedBase('A'), i, j)``.

    >>> A = IndexedBase('A')
    >>> a_ij = A[i, j]           # Prefer this,
    >>> b_ij = Indexed(A, i, j)  # over this.
    >>> a_ij == b_ij
    True

    """
    is_Indexed = True
    is_symbol = True
    is_Atom = True

    def __new__(cls, base, *args, **kw_args):
        from sympy.tensor.array.ndim_array import NDimArray
        from sympy.matrices.matrixbase import MatrixBase

        if not args:
            raise IndexException("Indexed needs at least one index.")
        if isinstance(base, (str, Symbol)):
            base = IndexedBase(base)
        elif not hasattr(base, '__getitem__') and not isinstance(base, IndexedBase):
            raise TypeError(filldedent("""
                The base can only be replaced with a string, Symbol,
                IndexedBase or an object with a method for getting
                items (i.e. an object with a `__getitem__` method).
                """))
        args = list(map(sympify, args))
        if isinstance(base, (NDimArray, Iterable, Tuple, MatrixBase)) and all(i.is_number for i in args):
            if len(args) == 1:
                return base[args[0]]
            else:
                return base[args]

        base = _sympify(base)

        obj = Expr.__new__(cls, base, *args, **kw_args)

        IndexedBase._set_assumptions(obj, base.assumptions0)

        return obj

    def _hashable_content(self):
        return super()._hashable_content() + tuple(sorted(self.assumptions0.items()))

    @property
    def name(self):
        return str(self)

    @property
    def _diff_wrt(self):
        """Allow derivatives with respect to an ``Indexed`` object."""
        return True

    def _eval_derivative(self, wrt):
        from sympy.tensor.array.ndim_array import NDimArray

        if isinstance(wrt, Indexed) and wrt.base == self.base:
            if len(self.indices) != len(wrt.indices):
                msg = "Different # of indices: d({!s})/d({!s})".format(self,
                                                                       wrt)
                raise IndexException(msg)
            result = S.One
            for index1, index2 in zip(self.indices, wrt.indices):
                result *= KroneckerDelta(index1, index2)
            return result
        elif isinstance(self.base, NDimArray):
            from sympy.tensor.array import derive_by_array
            return Indexed(derive_by_array(self.base, wrt), *self.args[1:])
        else:
            if Tuple(self.indices).has(wrt):
                return S.NaN
            return S.Zero

    @property
    def assumptions0(self):
        return {k: v for k, v in self._assumptions.items() if v is not None}

    @property
    def base(self):
        """Returns the ``IndexedBase`` of the ``Indexed`` object.

        Examples
        ========

        >>> from sympy import Indexed, IndexedBase, Idx, symbols
        >>> i, j = symbols('i j', cls=Idx)
        >>> Indexed('A', i, j).base
        A
        >>> B = IndexedBase('B')
        >>> B == B[i, j].base
        True

        """
        return self.args[0]

    @property
    def indices(self):
        """
        Returns the indices of the ``Indexed`` object.

        Examples
        ========

        >>> from sympy import Indexed, Idx, symbols
        >>> i, j = symbols('i j', cls=Idx)
        >>> Indexed('A', i, j).indices
        (i, j)

        """
        return self.args[1:]

    @property
    def rank(self):
        """
        Returns the rank of the ``Indexed`` object.

        Examples
        ========

        >>> from sympy import Indexed, Idx, symbols
        >>> i, j, k, l, m = symbols('i:m', cls=Idx)
        >>> Indexed('A', i, j).rank
        2
        >>> q = Indexed('A', i, j, k, l, m)
        >>> q.rank
        5
        >>> q.rank == len(q.indices)
        True

        """
        return len(self.args) - 1

    @property
    def shape(self):
        """Returns a list with dimensions of each index.

        Dimensions is a property of the array, not of the indices.  Still, if
        the ``IndexedBase`` does not define a shape attribute, it is assumed
        that the ranges of the indices correspond to the shape of the array.

        >>> from sympy import IndexedBase, Idx, symbols
        >>> n, m = symbols('n m', integer=True)
        >>> i = Idx('i', m)
        >>> j = Idx('j', m)
        >>> A = IndexedBase('A', shape=(n, n))
        >>> B = IndexedBase('B')
        >>> A[i, j].shape
        (n, n)
        >>> B[i, j].shape
        (m, m)
        """

        if self.base.shape:
            return self.base.shape
        sizes = []
        for i in self.indices:
            upper = getattr(i, 'upper', None)
            lower = getattr(i, 'lower', None)
            if None in (upper, lower):
                raise IndexException(filldedent("""
                    Range is not defined for all indices in: %s""" % self))
            try:
                size = upper - lower + 1
            except TypeError:
                raise IndexException(filldedent("""
                    Shape cannot be inferred from Idx with
                    undefined range: %s""" % self))
            sizes.append(size)
        return Tuple(*sizes)

    @property
    def ranges(self):
        """Returns a list of tuples with lower and upper range of each index.

        If an index does not define the data members upper and lower, the
        corresponding slot in the list contains ``None`` instead of a tuple.

        Examples
        ========

        >>> from sympy import Indexed,Idx, symbols
        >>> Indexed('A', Idx('i', 2), Idx('j', 4), Idx('k', 8)).ranges
        [(0, 1), (0, 3), (0, 7)]
        >>> Indexed('A', Idx('i', 3), Idx('j', 3), Idx('k', 3)).ranges
        [(0, 2), (0, 2), (0, 2)]
        >>> x, y, z = symbols('x y z', integer=True)
        >>> Indexed('A', x, y, z).ranges
        [None, None, None]

        """
        ranges = []
        sentinel = object()
        for i in self.indices:
            upper = getattr(i, 'upper', sentinel)
            lower = getattr(i, 'lower', sentinel)
            if sentinel not in (upper, lower):
                ranges.append((lower, upper))
            else:
                ranges.append(None)
        return ranges

    def _sympystr(self, p):
        indices = list(map(p.doprint, self.indices))
        return "%s[%s]" % (p.doprint(self.base), ", ".join(indices))

    @property
    def free_symbols(self):
        base_free_symbols = self.base.free_symbols
        indices_free_symbols = {
            fs for i in self.indices for fs in i.free_symbols}
        if base_free_symbols:
            return {self} | base_free_symbols | indices_free_symbols
        else:
            return indices_free_symbols

    @property
    def expr_free_symbols(self):
        from sympy.utilities.exceptions import sympy_deprecation_warning
        sympy_deprecation_warning("""
        The expr_free_symbols property is deprecated. Use free_symbols to get
        the free symbols of an expression.
        """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-expr-free-symbols")

        return {self}


class IndexedBase(Expr, NotIterable):
    """Represent the base or stem of an indexed object

    The IndexedBase class represent an array that contains elements. The main purpose
    of this class is to allow the convenient creation of objects of the Indexed
    class.  The __getitem__ method of IndexedBase returns an instance of
    Indexed.  Alone, without indices, the IndexedBase class can be used as a
    notation for e.g. matrix equations, resembling what you could do with the
    Symbol class.  But, the IndexedBase class adds functionality that is not
    available for Symbol instances:

      -  An IndexedBase object can optionally store shape information.  This can
         be used in to check array conformance and conditions for numpy
         broadcasting.  (TODO)
      -  An IndexedBase object implements syntactic sugar that allows easy symbolic
         representation of array operations, using implicit summation of
         repeated indices.
      -  The IndexedBase object symbolizes a mathematical structure equivalent
         to arrays, and is recognized as such for code generation and automatic
         compilation and wrapping.

    >>> from sympy.tensor import IndexedBase, Idx
    >>> from sympy import symbols
    >>> A = IndexedBase('A'); A
    A
    >>> type(A)
    <class 'sympy.tensor.indexed.IndexedBase'>

    When an IndexedBase object receives indices, it returns an array with named
    axes, represented by an Indexed object:

    >>> i, j = symbols('i j', integer=True)
    >>> A[i, j, 2]
    A[i, j, 2]
    >>> type(A[i, j, 2])
    <class 'sympy.tensor.indexed.Indexed'>

    The IndexedBase constructor takes an optional shape argument.  If given,
    it overrides any shape information in the indices. (But not the index
    ranges!)

    >>> m, n, o, p = symbols('m n o p', integer=True)
    >>> i = Idx('i', m)
    >>> j = Idx('j', n)
    >>> A[i, j].shape
    (m, n)
    >>> B = IndexedBase('B', shape=(o, p))
    >>> B[i, j].shape
    (o, p)

    Assumptions can be specified with keyword arguments the same way as for Symbol:

    >>> A_real = IndexedBase('A', real=True)
    >>> A_real.is_real
    True
    >>> A != A_real
    True

    Assumptions can also be inherited if a Symbol is used to initialize the IndexedBase:

    >>> I = symbols('I', integer=True)
    >>> C_inherit = IndexedBase(I)
    >>> C_explicit = IndexedBase('I', integer=True)
    >>> C_inherit == C_explicit
    True
    """
    is_symbol = True
    is_Atom = True

    @staticmethod
    def _set_assumptions(obj, assumptions):
        """Set assumptions on obj, making sure to apply consistent values."""
        tmp_asm_copy = assumptions.copy()
        is_commutative = fuzzy_bool(assumptions.get('commutative', True))
        assumptions['commutative'] = is_commutative
        obj._assumptions = StdFactKB(assumptions)
        obj._assumptions._generator = tmp_asm_copy  # Issue #8873

    def __new__(cls, label, shape=None, *, offset=S.Zero, strides=None, **kw_args):
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.tensor.array.ndim_array import NDimArray

        assumptions, kw_args = _filter_assumptions(kw_args)
        if isinstance(label, str):
            label = Symbol(label, **assumptions)
        elif isinstance(label, Symbol):
            assumptions = label._merge(assumptions)
        elif isinstance(label, (MatrixBase, NDimArray)):
            return label
        elif isinstance(label, Iterable):
            return _sympify(label)
        else:
            label = _sympify(label)

        if is_sequence(shape):
            shape = Tuple(*shape)
        elif shape is not None:
            shape = Tuple(shape)

        if shape is not None:
            obj = Expr.__new__(cls, label, shape)
        else:
            obj = Expr.__new__(cls, label)
        obj._shape = shape
        obj._offset = offset
        obj._strides = strides
        obj._name = str(label)

        IndexedBase._set_assumptions(obj, assumptions)
        return obj

    @property
    def name(self):
        return self._name

    def _hashable_content(self):
        return super()._hashable_content() + tuple(sorted(self.assumptions0.items()))

    @property
    def assumptions0(self):
        return {k: v for k, v in self._assumptions.items() if v is not None}

    def __getitem__(self, indices, **kw_args):
        if is_sequence(indices):
            # Special case needed because M[*my_tuple] is a syntax error.
            if self.shape and len(self.shape) != len(indices):
                raise IndexException("Rank mismatch.")
            return Indexed(self, *indices, **kw_args)
        else:
            if self.shape and len(self.shape) != 1:
                raise IndexException("Rank mismatch.")
            return Indexed(self, indices, **kw_args)

    @property
    def shape(self):
        """Returns the shape of the ``IndexedBase`` object.

        Examples
        ========

        >>> from sympy import IndexedBase, Idx
        >>> from sympy.abc import x, y
        >>> IndexedBase('A', shape=(x, y)).shape
        (x, y)

        Note: If the shape of the ``IndexedBase`` is specified, it will override
        any shape information given by the indices.

        >>> A = IndexedBase('A', shape=(x, y))
        >>> B = IndexedBase('B')
        >>> i = Idx('i', 2)
        >>> j = Idx('j', 1)
        >>> A[i, j].shape
        (x, y)
        >>> B[i, j].shape
        (2, 1)

        """
        return self._shape

    @property
    def strides(self):
        """Returns the strided scheme for the ``IndexedBase`` object.

        Normally this is a tuple denoting the number of
        steps to take in the respective dimension when traversing
        an array. For code generation purposes strides='C' and
        strides='F' can also be used.

        strides='C' would mean that code printer would unroll
        in row-major order and 'F' means unroll in column major
        order.

        """

        return self._strides

    @property
    def offset(self):
        """Returns the offset for the ``IndexedBase`` object.

        This is the value added to the resulting index when the
        2D Indexed object is unrolled to a 1D form. Used in code
        generation.

        Examples
        ==========
        >>> from sympy.printing import ccode
        >>> from sympy.tensor import IndexedBase, Idx
        >>> from sympy import symbols
        >>> l, m, n, o = symbols('l m n o', integer=True)
        >>> A = IndexedBase('A', strides=(l, m, n), offset=o)
        >>> i, j, k = map(Idx, 'ijk')
        >>> ccode(A[i, j, k])
        'A[l*i + m*j + n*k + o]'

        """
        return self._offset

    @property
    def label(self):
        """Returns the label of the ``IndexedBase`` object.

        Examples
        ========

        >>> from sympy import IndexedBase
        >>> from sympy.abc import x, y
        >>> IndexedBase('A', shape=(x, y)).label
        A

        """
        return self.args[0]

    def _sympystr(self, p):
        return p.doprint(self.label)


class Idx(Expr):
    """Represents an integer index as an ``Integer`` or integer expression.

    There are a number of ways to create an ``Idx`` object.  The constructor
    takes two arguments:

    ``label``
        An integer or a symbol that labels the index.
    ``range``
        Optionally you can specify a range as either

        * ``Symbol`` or integer: This is interpreted as a dimension. Lower and
          upper bounds are set to ``0`` and ``range - 1``, respectively.
        * ``tuple``: The two elements are interpreted as the lower and upper
          bounds of the range, respectively.

    Note: bounds of the range are assumed to be either integer or infinite (oo
    and -oo are allowed to specify an unbounded range). If ``n`` is given as a
    bound, then ``n.is_integer`` must not return false.

    For convenience, if the label is given as a string it is automatically
    converted to an integer symbol.  (Note: this conversion is not done for
    range or dimension arguments.)

    Examples
    ========

    >>> from sympy import Idx, symbols, oo
    >>> n, i, L, U = symbols('n i L U', integer=True)

    If a string is given for the label an integer ``Symbol`` is created and the
    bounds are both ``None``:

    >>> idx = Idx('qwerty'); idx
    qwerty
    >>> idx.lower, idx.upper
    (None, None)

    Both upper and lower bounds can be specified:

    >>> idx = Idx(i, (L, U)); idx
    i
    >>> idx.lower, idx.upper
    (L, U)

    When only a single bound is given it is interpreted as the dimension
    and the lower bound defaults to 0:

    >>> idx = Idx(i, n); idx.lower, idx.upper
    (0, n - 1)
    >>> idx = Idx(i, 4); idx.lower, idx.upper
    (0, 3)
    >>> idx = Idx(i, oo); idx.lower, idx.upper
    (0, oo)

    """

    is_integer = True
    is_finite = True
    is_real = True
    is_symbol = True
    is_Atom = True
    _diff_wrt = True

    def __new__(cls, label, range=None, **kw_args):

        if isinstance(label, str):
            label = Symbol(label, integer=True)
        label, range = list(map(sympify, (label, range)))

        if label.is_Number:
            if not label.is_integer:
                raise TypeError("Index is not an integer number.")
            return label

        if not label.is_integer:
            raise TypeError("Idx object requires an integer label.")

        elif is_sequence(range):
            if len(range) != 2:
                raise ValueError(filldedent("""
                    Idx range tuple must have length 2, but got %s""" % len(range)))
            for bound in range:
                if (bound.is_integer is False and bound is not S.Infinity
                        and bound is not S.NegativeInfinity):
                    raise TypeError("Idx object requires integer bounds.")
            args = label, Tuple(*range)
        elif isinstance(range, Expr):
            if range is not S.Infinity and fuzzy_not(range.is_integer):
                raise TypeError("Idx object requires an integer dimension.")
            args = label, Tuple(0, range - 1)
        elif range:
            raise TypeError(filldedent("""
                The range must be an ordered iterable or
                integer SymPy expression."""))
        else:
            args = label,

        obj = Expr.__new__(cls, *args, **kw_args)
        obj._assumptions["finite"] = True
        obj._assumptions["real"] = True
        return obj

    @property
    def label(self):
        """Returns the label (Integer or integer expression) of the Idx object.

        Examples
        ========

        >>> from sympy import Idx, Symbol
        >>> x = Symbol('x', integer=True)
        >>> Idx(x).label
        x
        >>> j = Symbol('j', integer=True)
        >>> Idx(j).label
        j
        >>> Idx(j + 1).label
        j + 1

        """
        return self.args[0]

    @property
    def lower(self):
        """Returns the lower bound of the ``Idx``.

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).lower
        0
        >>> Idx('j', 5).lower
        0
        >>> Idx('j').lower is None
        True

        """
        try:
            return self.args[1][0]
        except IndexError:
            return

    @property
    def upper(self):
        """Returns the upper bound of the ``Idx``.

        Examples
        ========

        >>> from sympy import Idx
        >>> Idx('j', 2).upper
        1
        >>> Idx('j', 5).upper
        4
        >>> Idx('j').upper is None
        True

        """
        try:
            return self.args[1][1]
        except IndexError:
            return

    def _sympystr(self, p):
        return p.doprint(self.label)

    @property
    def name(self):
        return self.label.name if self.label.is_Symbol else str(self.label)

    @property
    def free_symbols(self):
        return {self}


@dispatch(Idx, Idx)
def _eval_is_ge(lhs, rhs): # noqa:F811

    other_upper = rhs if rhs.upper is None else rhs.upper
    other_lower = rhs if rhs.lower is None else rhs.lower

    if lhs.lower is not None and (lhs.lower >= other_upper) == True:
        return True
    if lhs.upper is not None and (lhs.upper < other_lower) == True:
        return False
    return None


@dispatch(Idx, Number)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811

    other_upper = rhs
    other_lower = rhs

    if lhs.lower is not None and (lhs.lower >= other_upper) == True:
        return True
    if lhs.upper is not None and (lhs.upper < other_lower) == True:
        return False
    return None


@dispatch(Number, Idx)  # type:ignore
def _eval_is_ge(lhs, rhs): # noqa:F811

    other_upper = lhs
    other_lower = lhs

    if rhs.upper is not None and (rhs.upper <= other_lower) == True:
        return True
    if rhs.lower is not None and (rhs.lower > other_upper) == True:
        return False
    return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\sansio\scaffold.py ===
from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import typing as t
from collections import defaultdict
from functools import update_wrapper

from jinja2 import BaseLoader
from jinja2 import FileSystemLoader
from werkzeug.exceptions import default_exceptions
from werkzeug.exceptions import HTTPException
from werkzeug.utils import cached_property

from .. import typing as ft
from ..helpers import get_root_path
from ..templating import _default_template_ctx_processor

if t.TYPE_CHECKING:  # pragma: no cover
    from click import Group

# a singleton sentinel value for parameter defaults
_sentinel = object()

F = t.TypeVar("F", bound=t.Callable[..., t.Any])
T_after_request = t.TypeVar("T_after_request", bound=ft.AfterRequestCallable[t.Any])
T_before_request = t.TypeVar("T_before_request", bound=ft.BeforeRequestCallable)
T_error_handler = t.TypeVar("T_error_handler", bound=ft.ErrorHandlerCallable)
T_teardown = t.TypeVar("T_teardown", bound=ft.TeardownCallable)
T_template_context_processor = t.TypeVar(
    "T_template_context_processor", bound=ft.TemplateContextProcessorCallable
)
T_url_defaults = t.TypeVar("T_url_defaults", bound=ft.URLDefaultCallable)
T_url_value_preprocessor = t.TypeVar(
    "T_url_value_preprocessor", bound=ft.URLValuePreprocessorCallable
)
T_route = t.TypeVar("T_route", bound=ft.RouteCallable)


def setupmethod(f: F) -> F:
    f_name = f.__name__

    def wrapper_func(self: Scaffold, *args: t.Any, **kwargs: t.Any) -> t.Any:
        self._check_setup_finished(f_name)
        return f(self, *args, **kwargs)

    return t.cast(F, update_wrapper(wrapper_func, f))


class Scaffold:
    """Common behavior shared between :class:`~flask.Flask` and
    :class:`~flask.blueprints.Blueprint`.

    :param import_name: The import name of the module where this object
        is defined. Usually :attr:`__name__` should be used.
    :param static_folder: Path to a folder of static files to serve.
        If this is set, a static route will be added.
    :param static_url_path: URL prefix for the static route.
    :param template_folder: Path to a folder containing template files.
        for rendering. If this is set, a Jinja loader will be added.
    :param root_path: The path that static, template, and resource files
        are relative to. Typically not set, it is discovered based on
        the ``import_name``.

    .. versionadded:: 2.0
    """

    cli: Group
    name: str
    _static_folder: str | None = None
    _static_url_path: str | None = None

    def __init__(
        self,
        import_name: str,
        static_folder: str | os.PathLike[str] | None = None,
        static_url_path: str | None = None,
        template_folder: str | os.PathLike[str] | None = None,
        root_path: str | None = None,
    ):
        #: The name of the package or module that this object belongs
        #: to. Do not change this once it is set by the constructor.
        self.import_name = import_name

        self.static_folder = static_folder  # type: ignore
        self.static_url_path = static_url_path

        #: The path to the templates folder, relative to
        #: :attr:`root_path`, to add to the template loader. ``None`` if
        #: templates should not be added.
        self.template_folder = template_folder

        if root_path is None:
            root_path = get_root_path(self.import_name)

        #: Absolute path to the package on the filesystem. Used to look
        #: up resources contained in the package.
        self.root_path = root_path

        #: A dictionary mapping endpoint names to view functions.
        #:
        #: To register a view function, use the :meth:`route` decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.view_functions: dict[str, ft.RouteCallable] = {}

        #: A data structure of registered error handlers, in the format
        #: ``{scope: {code: {class: handler}}}``. The ``scope`` key is
        #: the name of a blueprint the handlers are active for, or
        #: ``None`` for all requests. The ``code`` key is the HTTP
        #: status code for ``HTTPException``, or ``None`` for
        #: other exceptions. The innermost dictionary maps exception
        #: classes to handler functions.
        #:
        #: To register an error handler, use the :meth:`errorhandler`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.error_handler_spec: dict[
            ft.AppOrBlueprintKey,
            dict[int | None, dict[type[Exception], ft.ErrorHandlerCallable]],
        ] = defaultdict(lambda: defaultdict(dict))

        #: A data structure of functions to call at the beginning of
        #: each request, in the format ``{scope: [functions]}``. The
        #: ``scope`` key is the name of a blueprint the functions are
        #: active for, or ``None`` for all requests.
        #:
        #: To register a function, use the :meth:`before_request`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.before_request_funcs: dict[
            ft.AppOrBlueprintKey, list[ft.BeforeRequestCallable]
        ] = defaultdict(list)

        #: A data structure of functions to call at the end of each
        #: request, in the format ``{scope: [functions]}``. The
        #: ``scope`` key is the name of a blueprint the functions are
        #: active for, or ``None`` for all requests.
        #:
        #: To register a function, use the :meth:`after_request`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.after_request_funcs: dict[
            ft.AppOrBlueprintKey, list[ft.AfterRequestCallable[t.Any]]
        ] = defaultdict(list)

        #: A data structure of functions to call at the end of each
        #: request even if an exception is raised, in the format
        #: ``{scope: [functions]}``. The ``scope`` key is the name of a
        #: blueprint the functions are active for, or ``None`` for all
        #: requests.
        #:
        #: To register a function, use the :meth:`teardown_request`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.teardown_request_funcs: dict[
            ft.AppOrBlueprintKey, list[ft.TeardownCallable]
        ] = defaultdict(list)

        #: A data structure of functions to call to pass extra context
        #: values when rendering templates, in the format
        #: ``{scope: [functions]}``. The ``scope`` key is the name of a
        #: blueprint the functions are active for, or ``None`` for all
        #: requests.
        #:
        #: To register a function, use the :meth:`context_processor`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.template_context_processors: dict[
            ft.AppOrBlueprintKey, list[ft.TemplateContextProcessorCallable]
        ] = defaultdict(list, {None: [_default_template_ctx_processor]})

        #: A data structure of functions to call to modify the keyword
        #: arguments passed to the view function, in the format
        #: ``{scope: [functions]}``. The ``scope`` key is the name of a
        #: blueprint the functions are active for, or ``None`` for all
        #: requests.
        #:
        #: To register a function, use the
        #: :meth:`url_value_preprocessor` decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.url_value_preprocessors: dict[
            ft.AppOrBlueprintKey,
            list[ft.URLValuePreprocessorCallable],
        ] = defaultdict(list)

        #: A data structure of functions to call to modify the keyword
        #: arguments when generating URLs, in the format
        #: ``{scope: [functions]}``. The ``scope`` key is the name of a
        #: blueprint the functions are active for, or ``None`` for all
        #: requests.
        #:
        #: To register a function, use the :meth:`url_defaults`
        #: decorator.
        #:
        #: This data structure is internal. It should not be modified
        #: directly and its format may change at any time.
        self.url_default_functions: dict[
            ft.AppOrBlueprintKey, list[ft.URLDefaultCallable]
        ] = defaultdict(list)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.name!r}>"

    def _check_setup_finished(self, f_name: str) -> None:
        raise NotImplementedError

    @property
    def static_folder(self) -> str | None:
        """The absolute path to the configured static folder. ``None``
        if no static folder is set.
        """
        if self._static_folder is not None:
            return os.path.join(self.root_path, self._static_folder)
        else:
            return None

    @static_folder.setter
    def static_folder(self, value: str | os.PathLike[str] | None) -> None:
        if value is not None:
            value = os.fspath(value).rstrip(r"\/")

        self._static_folder = value

    @property
    def has_static_folder(self) -> bool:
        """``True`` if :attr:`static_folder` is set.

        .. versionadded:: 0.5
        """
        return self.static_folder is not None

    @property
    def static_url_path(self) -> str | None:
        """The URL prefix that the static route will be accessible from.

        If it was not configured during init, it is derived from
        :attr:`static_folder`.
        """
        if self._static_url_path is not None:
            return self._static_url_path

        if self.static_folder is not None:
            basename = os.path.basename(self.static_folder)
            return f"/{basename}".rstrip("/")

        return None

    @static_url_path.setter
    def static_url_path(self, value: str | None) -> None:
        if value is not None:
            value = value.rstrip("/")

        self._static_url_path = value

    @cached_property
    def jinja_loader(self) -> BaseLoader | None:
        """The Jinja loader for this object's templates. By default this
        is a class :class:`jinja2.loaders.FileSystemLoader` to
        :attr:`template_folder` if it is set.

        .. versionadded:: 0.5
        """
        if self.template_folder is not None:
            return FileSystemLoader(os.path.join(self.root_path, self.template_folder))
        else:
            return None

    def _method_route(
        self,
        method: str,
        rule: str,
        options: dict[str, t.Any],
    ) -> t.Callable[[T_route], T_route]:
        if "methods" in options:
            raise TypeError("Use the 'route' decorator to use the 'methods' argument.")

        return self.route(rule, methods=[method], **options)

    @setupmethod
    def get(self, rule: str, **options: t.Any) -> t.Callable[[T_route], T_route]:
        """Shortcut for :meth:`route` with ``methods=["GET"]``.

        .. versionadded:: 2.0
        """
        return self._method_route("GET", rule, options)

    @setupmethod
    def post(self, rule: str, **options: t.Any) -> t.Callable[[T_route], T_route]:
        """Shortcut for :meth:`route` with ``methods=["POST"]``.

        .. versionadded:: 2.0
        """
        return self._method_route("POST", rule, options)

    @setupmethod
    def put(self, rule: str, **options: t.Any) -> t.Callable[[T_route], T_route]:
        """Shortcut for :meth:`route` with ``methods=["PUT"]``.

        .. versionadded:: 2.0
        """
        return self._method_route("PUT", rule, options)

    @setupmethod
    def delete(self, rule: str, **options: t.Any) -> t.Callable[[T_route], T_route]:
        """Shortcut for :meth:`route` with ``methods=["DELETE"]``.

        .. versionadded:: 2.0
        """
        return self._method_route("DELETE", rule, options)

    @setupmethod
    def patch(self, rule: str, **options: t.Any) -> t.Callable[[T_route], T_route]:
        """Shortcut for :meth:`route` with ``methods=["PATCH"]``.

        .. versionadded:: 2.0
        """
        return self._method_route("PATCH", rule, options)

    @setupmethod
    def route(self, rule: str, **options: t.Any) -> t.Callable[[T_route], T_route]:
        """Decorate a view function to register it with the given URL
        rule and options. Calls :meth:`add_url_rule`, which has more
        details about the implementation.

        .. code-block:: python

            @app.route("/")
            def index():
                return "Hello, World!"

        See :ref:`url-route-registrations`.

        The endpoint name for the route defaults to the name of the view
        function if the ``endpoint`` parameter isn't passed.

        The ``methods`` parameter defaults to ``["GET"]``. ``HEAD`` and
        ``OPTIONS`` are added automatically.

        :param rule: The URL rule string.
        :param options: Extra options passed to the
            :class:`~werkzeug.routing.Rule` object.
        """

        def decorator(f: T_route) -> T_route:
            endpoint = options.pop("endpoint", None)
            self.add_url_rule(rule, endpoint, f, **options)
            return f

        return decorator

    @setupmethod
    def add_url_rule(
        self,
        rule: str,
        endpoint: str | None = None,
        view_func: ft.RouteCallable | None = None,
        provide_automatic_options: bool | None = None,
        **options: t.Any,
    ) -> None:
        """Register a rule for routing incoming requests and building
        URLs. The :meth:`route` decorator is a shortcut to call this
        with the ``view_func`` argument. These are equivalent:

        .. code-block:: python

            @app.route("/")
            def index():
                ...

        .. code-block:: python

            def index():
                ...

            app.add_url_rule("/", view_func=index)

        See :ref:`url-route-registrations`.

        The endpoint name for the route defaults to the name of the view
        function if the ``endpoint`` parameter isn't passed. An error
        will be raised if a function has already been registered for the
        endpoint.

        The ``methods`` parameter defaults to ``["GET"]``. ``HEAD`` is
        always added automatically, and ``OPTIONS`` is added
        automatically by default.

        ``view_func`` does not necessarily need to be passed, but if the
        rule should participate in routing an endpoint name must be
        associated with a view function at some point with the
        :meth:`endpoint` decorator.

        .. code-block:: python

            app.add_url_rule("/", endpoint="index")

            @app.endpoint("index")
            def index():
                ...

        If ``view_func`` has a ``required_methods`` attribute, those
        methods are added to the passed and automatic methods. If it
        has a ``provide_automatic_methods`` attribute, it is used as the
        default if the parameter is not passed.

        :param rule: The URL rule string.
        :param endpoint: The endpoint name to associate with the rule
            and view function. Used when routing and building URLs.
            Defaults to ``view_func.__name__``.
        :param view_func: The view function to associate with the
            endpoint name.
        :param provide_automatic_options: Add the ``OPTIONS`` method and
            respond to ``OPTIONS`` requests automatically.
        :param options: Extra options passed to the
            :class:`~werkzeug.routing.Rule` object.
        """
        raise NotImplementedError

    @setupmethod
    def endpoint(self, endpoint: str) -> t.Callable[[F], F]:
        """Decorate a view function to register it for the given
        endpoint. Used if a rule is added without a ``view_func`` with
        :meth:`add_url_rule`.

        .. code-block:: python

            app.add_url_rule("/ex", endpoint="example")

            @app.endpoint("example")
            def example():
                ...

        :param endpoint: The endpoint name to associate with the view
            function.
        """

        def decorator(f: F) -> F:
            self.view_functions[endpoint] = f
            return f

        return decorator

    @setupmethod
    def before_request(self, f: T_before_request) -> T_before_request:
        """Register a function to run before each request.

        For example, this can be used to open a database connection, or
        to load the logged in user from the session.

        .. code-block:: python

            @app.before_request
            def load_user():
                if "user_id" in session:
                    g.user = db.session.get(session["user_id"])

        The function will be called without any arguments. If it returns
        a non-``None`` value, the value is handled as if it was the
        return value from the view, and further request handling is
        stopped.

        This is available on both app and blueprint objects. When used on an app, this
        executes before every request. When used on a blueprint, this executes before
        every request that the blueprint handles. To register with a blueprint and
        execute before every request, use :meth:`.Blueprint.before_app_request`.
        """
        self.before_request_funcs.setdefault(None, []).append(f)
        return f

    @setupmethod
    def after_request(self, f: T_after_request) -> T_after_request:
        """Register a function to run after each request to this object.

        The function is called with the response object, and must return
        a response object. This allows the functions to modify or
        replace the response before it is sent.

        If a function raises an exception, any remaining
        ``after_request`` functions will not be called. Therefore, this
        should not be used for actions that must execute, such as to
        close resources. Use :meth:`teardown_request` for that.

        This is available on both app and blueprint objects. When used on an app, this
        executes after every request. When used on a blueprint, this executes after
        every request that the blueprint handles. To register with a blueprint and
        execute after every request, use :meth:`.Blueprint.after_app_request`.
        """
        self.after_request_funcs.setdefault(None, []).append(f)
        return f

    @setupmethod
    def teardown_request(self, f: T_teardown) -> T_teardown:
        """Register a function to be called when the request context is
        popped. Typically this happens at the end of each request, but
        contexts may be pushed manually as well during testing.

        .. code-block:: python

            with app.test_request_context():
                ...

        When the ``with`` block exits (or ``ctx.pop()`` is called), the
        teardown functions are called just before the request context is
        made inactive.

        When a teardown function was called because of an unhandled
        exception it will be passed an error object. If an
        :meth:`errorhandler` is registered, it will handle the exception
        and the teardown will not receive it.

        Teardown functions must avoid raising exceptions. If they
        execute code that might fail they must surround that code with a
        ``try``/``except`` block and log any errors.

        The return values of teardown functions are ignored.

        This is available on both app and blueprint objects. When used on an app, this
        executes after every request. When used on a blueprint, this executes after
        every request that the blueprint handles. To register with a blueprint and
        execute after every request, use :meth:`.Blueprint.teardown_app_request`.
        """
        self.teardown_request_funcs.setdefault(None, []).append(f)
        return f

    @setupmethod
    def context_processor(
        self,
        f: T_template_context_processor,
    ) -> T_template_context_processor:
        """Registers a template context processor function. These functions run before
        rendering a template. The keys of the returned dict are added as variables
        available in the template.

        This is available on both app and blueprint objects. When used on an app, this
        is called for every rendered template. When used on a blueprint, this is called
        for templates rendered from the blueprint's views. To register with a blueprint
        and affect every template, use :meth:`.Blueprint.app_context_processor`.
        """
        self.template_context_processors[None].append(f)
        return f

    @setupmethod
    def url_value_preprocessor(
        self,
        f: T_url_value_preprocessor,
    ) -> T_url_value_preprocessor:
        """Register a URL value preprocessor function for all view
        functions in the application. These functions will be called before the
        :meth:`before_request` functions.

        The function can modify the values captured from the matched url before
        they are passed to the view. For example, this can be used to pop a
        common language code value and place it in ``g`` rather than pass it to
        every view.

        The function is passed the endpoint name and values dict. The return
        value is ignored.

        This is available on both app and blueprint objects. When used on an app, this
        is called for every request. When used on a blueprint, this is called for
        requests that the blueprint handles. To register with a blueprint and affect
        every request, use :meth:`.Blueprint.app_url_value_preprocessor`.
        """
        self.url_value_preprocessors[None].append(f)
        return f

    @setupmethod
    def url_defaults(self, f: T_url_defaults) -> T_url_defaults:
        """Callback function for URL defaults for all view functions of the
        application.  It's called with the endpoint and values and should
        update the values passed in place.

        This is available on both app and blueprint objects. When used on an app, this
        is called for every request. When used on a blueprint, this is called for
        requests that the blueprint handles. To register with a blueprint and affect
        every request, use :meth:`.Blueprint.app_url_defaults`.
        """
        self.url_default_functions[None].append(f)
        return f

    @setupmethod
    def errorhandler(
        self, code_or_exception: type[Exception] | int
    ) -> t.Callable[[T_error_handler], T_error_handler]:
        """Register a function to handle errors by code or exception class.

        A decorator that is used to register a function given an
        error code.  Example::

            @app.errorhandler(404)
            def page_not_found(error):
                return 'This page does not exist', 404

        You can also register handlers for arbitrary exceptions::

            @app.errorhandler(DatabaseError)
            def special_exception_handler(error):
                return 'Database connection failed', 500

        This is available on both app and blueprint objects. When used on an app, this
        can handle errors from every request. When used on a blueprint, this can handle
        errors from requests that the blueprint handles. To register with a blueprint
        and affect every request, use :meth:`.Blueprint.app_errorhandler`.

        .. versionadded:: 0.7
            Use :meth:`register_error_handler` instead of modifying
            :attr:`error_handler_spec` directly, for application wide error
            handlers.

        .. versionadded:: 0.7
           One can now additionally also register custom exception types
           that do not necessarily have to be a subclass of the
           :class:`~werkzeug.exceptions.HTTPException` class.

        :param code_or_exception: the code as integer for the handler, or
                                  an arbitrary exception
        """

        def decorator(f: T_error_handler) -> T_error_handler:
            self.register_error_handler(code_or_exception, f)
            return f

        return decorator

    @setupmethod
    def register_error_handler(
        self,
        code_or_exception: type[Exception] | int,
        f: ft.ErrorHandlerCallable,
    ) -> None:
        """Alternative error attach function to the :meth:`errorhandler`
        decorator that is more straightforward to use for non decorator
        usage.

        .. versionadded:: 0.7
        """
        exc_class, code = self._get_exc_class_and_code(code_or_exception)
        self.error_handler_spec[None][code][exc_class] = f

    @staticmethod
    def _get_exc_class_and_code(
        exc_class_or_code: type[Exception] | int,
    ) -> tuple[type[Exception], int | None]:
        """Get the exception class being handled. For HTTP status codes
        or ``HTTPException`` subclasses, return both the exception and
        status code.

        :param exc_class_or_code: Any exception class, or an HTTP status
            code as an integer.
        """
        exc_class: type[Exception]

        if isinstance(exc_class_or_code, int):
            try:
                exc_class = default_exceptions[exc_class_or_code]
            except KeyError:
                raise ValueError(
                    f"'{exc_class_or_code}' is not a recognized HTTP"
                    " error code. Use a subclass of HTTPException with"
                    " that code instead."
                ) from None
        else:
            exc_class = exc_class_or_code

        if isinstance(exc_class, Exception):
            raise TypeError(
                f"{exc_class!r} is an instance, not a class. Handlers"
                " can only be registered for Exception classes or HTTP"
                " error codes."
            )

        if not issubclass(exc_class, Exception):
            raise ValueError(
                f"'{exc_class.__name__}' is not a subclass of Exception."
                " Handlers can only be registered for Exception classes"
                " or HTTP error codes."
            )

        if issubclass(exc_class, HTTPException):
            return exc_class, exc_class.code
        else:
            return exc_class, None


def _endpoint_from_view_func(view_func: ft.RouteCallable) -> str:
    """Internal helper that returns the default endpoint for a given
    function.  This always is the function name.
    """
    assert view_func is not None, "expected view func if endpoint is not provided."
    return view_func.__name__


def _find_package_path(import_name: str) -> str:
    """Find the path that contains the package or module."""
    root_mod_name, _, _ = import_name.partition(".")

    try:
        root_spec = importlib.util.find_spec(root_mod_name)

        if root_spec is None:
            raise ValueError("not found")
    except (ImportError, ValueError):
        # ImportError: the machinery told us it does not exist
        # ValueError:
        #    - the module name was invalid
        #    - the module name is __main__
        #    - we raised `ValueError` due to `root_spec` being `None`
        return os.getcwd()

    if root_spec.submodule_search_locations:
        if root_spec.origin is None or root_spec.origin == "namespace":
            # namespace package
            package_spec = importlib.util.find_spec(import_name)

            if package_spec is not None and package_spec.submodule_search_locations:
                # Pick the path in the namespace that contains the submodule.
                package_path = pathlib.Path(
                    os.path.commonpath(package_spec.submodule_search_locations)
                )
                search_location = next(
                    location
                    for location in root_spec.submodule_search_locations
                    if package_path.is_relative_to(location)
                )
            else:
                # Pick the first path.
                search_location = root_spec.submodule_search_locations[0]

            return os.path.dirname(search_location)
        else:
            # package with __init__.py
            return os.path.dirname(os.path.dirname(root_spec.origin))
    else:
        # module
        return os.path.dirname(root_spec.origin)  # type: ignore[type-var, return-value]


def find_package(import_name: str) -> tuple[str | None, str]:
    """Find the prefix that a package is installed under, and the path
    that it would be imported from.

    The prefix is the directory containing the standard directory
    hierarchy (lib, bin, etc.). If the package is not installed to the
    system (:attr:`sys.prefix`) or a virtualenv (``site-packages``),
    ``None`` is returned.

    The path is the entry in :attr:`sys.path` that contains the package
    for import. If the package is not installed, it's assumed that the
    package was imported from the current working directory.
    """
    package_path = _find_package_path(import_name)
    py_prefix = os.path.abspath(sys.prefix)

    # installed to the system
    if pathlib.PurePath(package_path).is_relative_to(py_prefix):
        return py_prefix, package_path

    site_parent, site_folder = os.path.split(package_path)

    # installed to a virtualenv
    if site_folder.lower() == "site-packages":
        parent, folder = os.path.split(site_parent)

        # Windows (prefix/lib/site-packages)
        if folder.lower() == "lib":
            return parent, package_path

        # Unix (prefix/lib/pythonX.Y/site-packages)
        if os.path.basename(parent).lower() == "lib":
            return os.path.dirname(parent), package_path

        # something else (prefix/site-packages)
        return site_parent, package_path

    # not installed
    return None, package_path

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\polyoptions.py ===
"""Options manager for :class:`~.Poly` and public API functions. """

from __future__ import annotations

__all__ = ["Options"]

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.sympify import sympify
from sympy.polys.polyerrors import GeneratorsError, OptionError, FlagError
from sympy.utilities import numbered_symbols, topological_sort, public
from sympy.utilities.iterables import has_dups, is_sequence

import sympy.polys

import re

class Option:
    """Base class for all kinds of options. """

    option: str | None = None

    is_Flag = False

    requires: list[str] = []
    excludes: list[str] = []

    after: list[str] = []
    before: list[str] = []

    @classmethod
    def default(cls):
        return None

    @classmethod
    def preprocess(cls, option):
        return None

    @classmethod
    def postprocess(cls, options):
        pass


class Flag(Option):
    """Base class for all kinds of flags. """

    is_Flag = True


class BooleanOption(Option):
    """An option that must have a boolean value or equivalent assigned. """

    @classmethod
    def preprocess(cls, value):
        if value in [True, False]:
            return bool(value)
        else:
            raise OptionError("'%s' must have a boolean value assigned, got %s" % (cls.option, value))


class OptionType(type):
    """Base type for all options that does registers options. """

    def __init__(cls, *args, **kwargs):
        @property
        def getter(self):
            try:
                return self[cls.option]
            except KeyError:
                return cls.default()

        setattr(Options, cls.option, getter)
        Options.__options__[cls.option] = cls


@public
class Options(dict):
    """
    Options manager for polynomial manipulation module.

    Examples
    ========

    >>> from sympy.polys.polyoptions import Options
    >>> from sympy.polys.polyoptions import build_options

    >>> from sympy.abc import x, y, z

    >>> Options((x, y, z), {'domain': 'ZZ'})
    {'auto': False, 'domain': ZZ, 'gens': (x, y, z)}

    >>> build_options((x, y, z), {'domain': 'ZZ'})
    {'auto': False, 'domain': ZZ, 'gens': (x, y, z)}

    **Options**

    * Expand --- boolean option
    * Gens --- option
    * Wrt --- option
    * Sort --- option
    * Order --- option
    * Field --- boolean option
    * Greedy --- boolean option
    * Domain --- option
    * Split --- boolean option
    * Gaussian --- boolean option
    * Extension --- option
    * Modulus --- option
    * Symmetric --- boolean option
    * Strict --- boolean option

    **Flags**

    * Auto --- boolean flag
    * Frac --- boolean flag
    * Formal --- boolean flag
    * Polys --- boolean flag
    * Include --- boolean flag
    * All --- boolean flag
    * Gen --- flag
    * Series --- boolean flag

    """

    __order__ = None
    __options__: dict[str, type[Option]] = {}

    gens: tuple[Expr, ...]
    domain: sympy.polys.domains.Domain

    def __init__(self, gens, args, flags=None, strict=False):
        dict.__init__(self)

        if gens and args.get('gens', ()):
            raise OptionError(
                "both '*gens' and keyword argument 'gens' supplied")
        elif gens:
            args = dict(args)
            args['gens'] = gens

        defaults = args.pop('defaults', {})

        def preprocess_options(args):
            for option, value in args.items():
                try:
                    cls = self.__options__[option]
                except KeyError:
                    raise OptionError("'%s' is not a valid option" % option)

                if issubclass(cls, Flag):
                    if flags is None or option not in flags:
                        if strict:
                            raise OptionError("'%s' flag is not allowed in this context" % option)

                if value is not None:
                    self[option] = cls.preprocess(value)

        preprocess_options(args)

        for key in dict(defaults):
            if key in self:
                del defaults[key]
            else:
                for option in self.keys():
                    cls = self.__options__[option]

                    if key in cls.excludes:
                        del defaults[key]
                        break

        preprocess_options(defaults)

        for option in self.keys():
            cls = self.__options__[option]

            for require_option in cls.requires:
                if self.get(require_option) is None:
                    raise OptionError("'%s' option is only allowed together with '%s'" % (option, require_option))

            for exclude_option in cls.excludes:
                if self.get(exclude_option) is not None:
                    raise OptionError("'%s' option is not allowed together with '%s'" % (option, exclude_option))

        for option in self.__order__:
            self.__options__[option].postprocess(self)

    @classmethod
    def _init_dependencies_order(cls):
        """Resolve the order of options' processing. """
        if cls.__order__ is None:
            vertices, edges = [], set()

            for name, option in cls.__options__.items():
                vertices.append(name)

                edges.update((_name, name) for _name in option.after)

                edges.update((name, _name) for _name in option.before)

            try:
                cls.__order__ = topological_sort((vertices, list(edges)))
            except ValueError:
                raise RuntimeError(
                    "cycle detected in sympy.polys options framework")

    def clone(self, updates={}):
        """Clone ``self`` and update specified options. """
        obj = dict.__new__(self.__class__)

        for option, value in self.items():
            obj[option] = value

        for option, value in updates.items():
            obj[option] = value

        return obj

    def __setattr__(self, attr, value):
        if attr in self.__options__:
            self[attr] = value
        else:
            super().__setattr__(attr, value)

    @property
    def args(self):
        args = {}

        for option, value in self.items():
            if value is not None and option != 'gens':
                cls = self.__options__[option]

                if not issubclass(cls, Flag):
                    args[option] = value

        return args

    @property
    def options(self):
        options = {}

        for option, cls in self.__options__.items():
            if not issubclass(cls, Flag):
                options[option] = getattr(self, option)

        return options

    @property
    def flags(self):
        flags = {}

        for option, cls in self.__options__.items():
            if issubclass(cls, Flag):
                flags[option] = getattr(self, option)

        return flags


class Expand(BooleanOption, metaclass=OptionType):
    """``expand`` option to polynomial manipulation functions. """

    option = 'expand'

    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return True


class Gens(Option, metaclass=OptionType):
    """``gens`` option to polynomial manipulation functions. """

    option = 'gens'

    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return ()

    @classmethod
    def preprocess(cls, gens):
        if isinstance(gens, Basic):
            gens = (gens,)
        elif len(gens) == 1 and is_sequence(gens[0]):
            gens = gens[0]

        if gens == (None,):
            gens = ()
        elif has_dups(gens):
            raise GeneratorsError("duplicated generators: %s" % str(gens))
        elif any(gen.is_commutative is False for gen in gens):
            raise GeneratorsError("non-commutative generators: %s" % str(gens))

        return tuple(gens)


class Wrt(Option, metaclass=OptionType):
    """``wrt`` option to polynomial manipulation functions. """

    option = 'wrt'

    requires: list[str] = []
    excludes: list[str] = []

    _re_split = re.compile(r"\s*,\s*|\s+")

    @classmethod
    def preprocess(cls, wrt):
        if isinstance(wrt, Basic):
            return [str(wrt)]
        elif isinstance(wrt, str):
            wrt = wrt.strip()
            if wrt.endswith(','):
                raise OptionError('Bad input: missing parameter.')
            if not wrt:
                return []
            return list(cls._re_split.split(wrt))
        elif hasattr(wrt, '__getitem__'):
            return list(map(str, wrt))
        else:
            raise OptionError("invalid argument for 'wrt' option")


class Sort(Option, metaclass=OptionType):
    """``sort`` option to polynomial manipulation functions. """

    option = 'sort'

    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return []

    @classmethod
    def preprocess(cls, sort):
        if isinstance(sort, str):
            return [ gen.strip() for gen in sort.split('>') ]
        elif hasattr(sort, '__getitem__'):
            return list(map(str, sort))
        else:
            raise OptionError("invalid argument for 'sort' option")


class Order(Option, metaclass=OptionType):
    """``order`` option to polynomial manipulation functions. """

    option = 'order'

    requires: list[str] = []
    excludes: list[str] = []

    @classmethod
    def default(cls):
        return sympy.polys.orderings.lex

    @classmethod
    def preprocess(cls, order):
        return sympy.polys.orderings.monomial_key(order)


class Field(BooleanOption, metaclass=OptionType):
    """``field`` option to polynomial manipulation functions. """

    option = 'field'

    requires: list[str] = []
    excludes = ['domain', 'split', 'gaussian']


class Greedy(BooleanOption, metaclass=OptionType):
    """``greedy`` option to polynomial manipulation functions. """

    option = 'greedy'

    requires: list[str] = []
    excludes = ['domain', 'split', 'gaussian', 'extension', 'modulus', 'symmetric']


class Composite(BooleanOption, metaclass=OptionType):
    """``composite`` option to polynomial manipulation functions. """

    option = 'composite'

    @classmethod
    def default(cls):
        return None

    requires: list[str] = []
    excludes = ['domain', 'split', 'gaussian', 'extension', 'modulus', 'symmetric']


class Domain(Option, metaclass=OptionType):
    """``domain`` option to polynomial manipulation functions. """

    option = 'domain'

    requires: list[str] = []
    excludes = ['field', 'greedy', 'split', 'gaussian', 'extension']

    after = ['gens']

    _re_realfield = re.compile(r"^(R|RR)(_(\d+))?$")
    _re_complexfield = re.compile(r"^(C|CC)(_(\d+))?$")
    _re_finitefield = re.compile(r"^(FF|GF)\((\d+)\)$")
    _re_polynomial = re.compile(r"^(Z|ZZ|Q|QQ|ZZ_I|QQ_I|R|RR|C|CC)\[(.+)\]$")
    _re_fraction = re.compile(r"^(Z|ZZ|Q|QQ)\((.+)\)$")
    _re_algebraic = re.compile(r"^(Q|QQ)\<(.+)\>$")

    @classmethod
    def preprocess(cls, domain):
        if isinstance(domain, sympy.polys.domains.Domain):
            return domain
        elif hasattr(domain, 'to_domain'):
            return domain.to_domain()
        elif isinstance(domain, str):
            if domain in ['Z', 'ZZ']:
                return sympy.polys.domains.ZZ

            if domain in ['Q', 'QQ']:
                return sympy.polys.domains.QQ

            if domain == 'ZZ_I':
                return sympy.polys.domains.ZZ_I

            if domain == 'QQ_I':
                return sympy.polys.domains.QQ_I

            if domain == 'EX':
                return sympy.polys.domains.EX

            r = cls._re_realfield.match(domain)

            if r is not None:
                _, _, prec = r.groups()

                if prec is None:
                    return sympy.polys.domains.RR
                else:
                    return sympy.polys.domains.RealField(int(prec))

            r = cls._re_complexfield.match(domain)

            if r is not None:
                _, _, prec = r.groups()

                if prec is None:
                    return sympy.polys.domains.CC
                else:
                    return sympy.polys.domains.ComplexField(int(prec))

            r = cls._re_finitefield.match(domain)

            if r is not None:
                return sympy.polys.domains.FF(int(r.groups()[1]))

            r = cls._re_polynomial.match(domain)

            if r is not None:
                ground, gens = r.groups()

                gens = list(map(sympify, gens.split(',')))

                if ground in ['Z', 'ZZ']:
                    return sympy.polys.domains.ZZ.poly_ring(*gens)
                elif ground in ['Q', 'QQ']:
                    return sympy.polys.domains.QQ.poly_ring(*gens)
                elif ground in ['R', 'RR']:
                    return sympy.polys.domains.RR.poly_ring(*gens)
                elif ground == 'ZZ_I':
                    return sympy.polys.domains.ZZ_I.poly_ring(*gens)
                elif ground == 'QQ_I':
                    return sympy.polys.domains.QQ_I.poly_ring(*gens)
                else:
                    return sympy.polys.domains.CC.poly_ring(*gens)

            r = cls._re_fraction.match(domain)

            if r is not None:
                ground, gens = r.groups()

                gens = list(map(sympify, gens.split(',')))

                if ground in ['Z', 'ZZ']:
                    return sympy.polys.domains.ZZ.frac_field(*gens)
                else:
                    return sympy.polys.domains.QQ.frac_field(*gens)

            r = cls._re_algebraic.match(domain)

            if r is not None:
                gens = list(map(sympify, r.groups()[1].split(',')))
                return sympy.polys.domains.QQ.algebraic_field(*gens)

        raise OptionError('expected a valid domain specification, got %s' % domain)

    @classmethod
    def postprocess(cls, options):
        if 'gens' in options and 'domain' in options and options['domain'].is_Composite and \
                (set(options['domain'].symbols) & set(options['gens'])):
            raise GeneratorsError(
                "ground domain and generators interfere together")
        elif ('gens' not in options or not options['gens']) and \
                'domain' in options and options['domain'] == sympy.polys.domains.EX:
            raise GeneratorsError("you have to provide generators because EX domain was requested")


class Split(BooleanOption, metaclass=OptionType):
    """``split`` option to polynomial manipulation functions. """

    option = 'split'

    requires: list[str] = []
    excludes = ['field', 'greedy', 'domain', 'gaussian', 'extension',
        'modulus', 'symmetric']

    @classmethod
    def postprocess(cls, options):
        if 'split' in options:
            raise NotImplementedError("'split' option is not implemented yet")


class Gaussian(BooleanOption, metaclass=OptionType):
    """``gaussian`` option to polynomial manipulation functions. """

    option = 'gaussian'

    requires: list[str] = []
    excludes = ['field', 'greedy', 'domain', 'split', 'extension',
        'modulus', 'symmetric']

    @classmethod
    def postprocess(cls, options):
        if 'gaussian' in options and options['gaussian'] is True:
            options['domain'] = sympy.polys.domains.QQ_I
            Extension.postprocess(options)


class Extension(Option, metaclass=OptionType):
    """``extension`` option to polynomial manipulation functions. """

    option = 'extension'

    requires: list[str] = []
    excludes = ['greedy', 'domain', 'split', 'gaussian', 'modulus',
        'symmetric']

    @classmethod
    def preprocess(cls, extension):
        if extension == 1:
            return bool(extension)
        elif extension == 0:
            raise OptionError("'False' is an invalid argument for 'extension'")
        else:
            if not hasattr(extension, '__iter__'):
                extension = {extension}
            else:
                if not extension:
                    extension = None
                else:
                    extension = set(extension)

            return extension

    @classmethod
    def postprocess(cls, options):
        if 'extension' in options and options['extension'] is not True:
            options['domain'] = sympy.polys.domains.QQ.algebraic_field(
                *options['extension'])


class Modulus(Option, metaclass=OptionType):
    """``modulus`` option to polynomial manipulation functions. """

    option = 'modulus'

    requires: list[str] = []
    excludes = ['greedy', 'split', 'domain', 'gaussian', 'extension']

    @classmethod
    def preprocess(cls, modulus):
        modulus = sympify(modulus)

        if modulus.is_Integer and modulus > 0:
            return int(modulus)
        else:
            raise OptionError(
                "'modulus' must a positive integer, got %s" % modulus)

    @classmethod
    def postprocess(cls, options):
        if 'modulus' in options:
            modulus = options['modulus']
            symmetric = options.get('symmetric', True)
            options['domain'] = sympy.polys.domains.FF(modulus, symmetric)


class Symmetric(BooleanOption, metaclass=OptionType):
    """``symmetric`` option to polynomial manipulation functions. """

    option = 'symmetric'

    requires = ['modulus']
    excludes = ['greedy', 'domain', 'split', 'gaussian', 'extension']


class Strict(BooleanOption, metaclass=OptionType):
    """``strict`` option to polynomial manipulation functions. """

    option = 'strict'

    @classmethod
    def default(cls):
        return True


class Auto(BooleanOption, Flag, metaclass=OptionType):
    """``auto`` flag to polynomial manipulation functions. """

    option = 'auto'

    after = ['field', 'domain', 'extension', 'gaussian']

    @classmethod
    def default(cls):
        return True

    @classmethod
    def postprocess(cls, options):
        if ('domain' in options or 'field' in options) and 'auto' not in options:
            options['auto'] = False


class Frac(BooleanOption, Flag, metaclass=OptionType):
    """``auto`` option to polynomial manipulation functions. """

    option = 'frac'

    @classmethod
    def default(cls):
        return False


class Formal(BooleanOption, Flag, metaclass=OptionType):
    """``formal`` flag to polynomial manipulation functions. """

    option = 'formal'

    @classmethod
    def default(cls):
        return False


class Polys(BooleanOption, Flag, metaclass=OptionType):
    """``polys`` flag to polynomial manipulation functions. """

    option = 'polys'


class Include(BooleanOption, Flag, metaclass=OptionType):
    """``include`` flag to polynomial manipulation functions. """

    option = 'include'

    @classmethod
    def default(cls):
        return False


class All(BooleanOption, Flag, metaclass=OptionType):
    """``all`` flag to polynomial manipulation functions. """

    option = 'all'

    @classmethod
    def default(cls):
        return False


class Gen(Flag, metaclass=OptionType):
    """``gen`` flag to polynomial manipulation functions. """

    option = 'gen'

    @classmethod
    def default(cls):
        return 0

    @classmethod
    def preprocess(cls, gen):
        if isinstance(gen, (Basic, int)):
            return gen
        else:
            raise OptionError("invalid argument for 'gen' option")


class Series(BooleanOption, Flag, metaclass=OptionType):
    """``series`` flag to polynomial manipulation functions. """

    option = 'series'

    @classmethod
    def default(cls):
        return False


class Symbols(Flag, metaclass=OptionType):
    """``symbols`` flag to polynomial manipulation functions. """

    option = 'symbols'

    @classmethod
    def default(cls):
        return numbered_symbols('s', start=1)

    @classmethod
    def preprocess(cls, symbols):
        if hasattr(symbols, '__iter__'):
            return iter(symbols)
        else:
            raise OptionError("expected an iterator or iterable container, got %s" % symbols)


class Method(Flag, metaclass=OptionType):
    """``method`` flag to polynomial manipulation functions. """

    option = 'method'

    @classmethod
    def preprocess(cls, method):
        if isinstance(method, str):
            return method.lower()
        else:
            raise OptionError("expected a string, got %s" % method)


def build_options(gens, args=None):
    """Construct options from keyword arguments or ... options. """
    if args is None:
        gens, args = (), gens

    if len(args) != 1 or 'opt' not in args or gens:
        return Options(gens, args)
    else:
        return args['opt']


def allowed_flags(args, flags):
    """
    Allow specified flags to be used in the given context.

    Examples
    ========

    >>> from sympy.polys.polyoptions import allowed_flags
    >>> from sympy.polys.domains import ZZ

    >>> allowed_flags({'domain': ZZ}, [])

    >>> allowed_flags({'domain': ZZ, 'frac': True}, [])
    Traceback (most recent call last):
    ...
    FlagError: 'frac' flag is not allowed in this context

    >>> allowed_flags({'domain': ZZ, 'frac': True}, ['frac'])

    """
    flags = set(flags)

    for arg in args.keys():
        try:
            if Options.__options__[arg].is_Flag and arg not in flags:
                raise FlagError(
                    "'%s' flag is not allowed in this context" % arg)
        except KeyError:
            raise OptionError("'%s' is not a valid option" % arg)


def set_defaults(options, **defaults):
    """Update options with default values. """
    if 'defaults' not in options:
        options = dict(options)
        options['defaults'] = defaults

    return options

Options._init_dependencies_order()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\matrices\linalg.py ===
"""
Linear algebra
--------------

Linear equations
................

Basic linear algebra is implemented; you can for example solve the linear
equation system::

      x + 2*y = -10
    3*x + 4*y =  10

using ``lu_solve``::

    >>> from mpmath import *
    >>> mp.pretty = False
    >>> A = matrix([[1, 2], [3, 4]])
    >>> b = matrix([-10, 10])
    >>> x = lu_solve(A, b)
    >>> x
    matrix(
    [['30.0'],
     ['-20.0']])

If you don't trust the result, use ``residual`` to calculate the residual ||A*x-b||::

    >>> residual(A, x, b)
    matrix(
    [['3.46944695195361e-18'],
     ['3.46944695195361e-18']])
    >>> str(eps)
    '2.22044604925031e-16'

As you can see, the solution is quite accurate. The error is caused by the
inaccuracy of the internal floating point arithmetic. Though, it's even smaller
than the current machine epsilon, which basically means you can trust the
result.

If you need more speed, use NumPy, or ``fp.lu_solve`` for a floating-point computation.

    >>> fp.lu_solve(A, b)   # doctest: +ELLIPSIS
    matrix(...)

``lu_solve`` accepts overdetermined systems. It is usually not possible to solve
such systems, so the residual is minimized instead. Internally this is done
using Cholesky decomposition to compute a least squares approximation. This means
that that ``lu_solve`` will square the errors. If you can't afford this, use
``qr_solve`` instead. It is twice as slow but more accurate, and it calculates
the residual automatically.


Matrix factorization
....................

The function ``lu`` computes an explicit LU factorization of a matrix::

    >>> P, L, U = lu(matrix([[0,2,3],[4,5,6],[7,8,9]]))
    >>> print(P)
    [0.0  0.0  1.0]
    [1.0  0.0  0.0]
    [0.0  1.0  0.0]
    >>> print(L)
    [              1.0                0.0  0.0]
    [              0.0                1.0  0.0]
    [0.571428571428571  0.214285714285714  1.0]
    >>> print(U)
    [7.0  8.0                9.0]
    [0.0  2.0                3.0]
    [0.0  0.0  0.214285714285714]
    >>> print(P.T*L*U)
    [0.0  2.0  3.0]
    [4.0  5.0  6.0]
    [7.0  8.0  9.0]

Interval matrices
-----------------

Matrices may contain interval elements. This allows one to perform
basic linear algebra operations such as matrix multiplication
and equation solving with rigorous error bounds::

    >>> a = iv.matrix([['0.1','0.3','1.0'],
    ...             ['7.1','5.5','4.8'],
    ...             ['3.2','4.4','5.6']])
    >>>
    >>> b = iv.matrix(['4','0.6','0.5'])
    >>> c = iv.lu_solve(a, b)
    >>> print(c)
    [   [5.2582327113062568605927528666, 5.25823271130625686059275702219]]
    [[-13.1550493962678375411635581388, -13.1550493962678375411635540152]]
    [  [7.42069154774972557628979076189, 7.42069154774972557628979190734]]
    >>> print(a*c)
    [  [3.99999999999999999999999844904, 4.00000000000000000000000155096]]
    [[0.599999999999999999999968898009, 0.600000000000000000000031763736]]
    [[0.499999999999999999999979320485, 0.500000000000000000000020679515]]
"""

# TODO:
# *implement high-level qr()
# *test unitvector
# *iterative solving

from copy import copy

from ..libmp.backend import xrange

class LinearAlgebraMethods(object):

    def LU_decomp(ctx, A, overwrite=False, use_cache=True):
        """
        LU-factorization of a n*n matrix using the Gauss algorithm.
        Returns L and U in one matrix and the pivot indices.

        Use overwrite to specify whether A will be overwritten with L and U.
        """
        if not A.rows == A.cols:
            raise ValueError('need n*n matrix')
        # get from cache if possible
        if use_cache and isinstance(A, ctx.matrix) and A._LU:
            return A._LU
        if not overwrite:
            orig = A
            A = A.copy()
        tol = ctx.absmin(ctx.mnorm(A,1) * ctx.eps) # each pivot element has to be bigger
        n = A.rows
        p = [None]*(n - 1)
        for j in xrange(n - 1):
            # pivoting, choose max(abs(reciprocal row sum)*abs(pivot element))
            biggest = 0
            for k in xrange(j, n):
                s = ctx.fsum([ctx.absmin(A[k,l]) for l in xrange(j, n)])
                if ctx.absmin(s) <= tol:
                    raise ZeroDivisionError('matrix is numerically singular')
                current = 1/s * ctx.absmin(A[k,j])
                if current > biggest: # TODO: what if equal?
                    biggest = current
                    p[j] = k
            # swap rows according to p
            ctx.swap_row(A, j, p[j])
            if ctx.absmin(A[j,j]) <= tol:
                raise ZeroDivisionError('matrix is numerically singular')
            # calculate elimination factors and add rows
            for i in xrange(j + 1, n):
                A[i,j] /= A[j,j]
                for k in xrange(j + 1, n):
                    A[i,k] -= A[i,j]*A[j,k]
        if ctx.absmin(A[n - 1,n - 1]) <= tol:
            raise ZeroDivisionError('matrix is numerically singular')
        # cache decomposition
        if not overwrite and isinstance(orig, ctx.matrix):
            orig._LU = (A, p)
        return A, p

    def L_solve(ctx, L, b, p=None):
        """
        Solve the lower part of a LU factorized matrix for y.
        """
        if L.rows != L.cols:
            raise RuntimeError("need n*n matrix")
        n = L.rows
        if len(b) != n:
            raise ValueError("Value should be equal to n")
        b = copy(b)
        if p: # swap b according to p
            for k in xrange(0, len(p)):
                ctx.swap_row(b, k, p[k])
        # solve
        for i in xrange(1, n):
            for j in xrange(i):
                b[i] -= L[i,j] * b[j]
        return b

    def U_solve(ctx, U, y):
        """
        Solve the upper part of a LU factorized matrix for x.
        """
        if U.rows != U.cols:
            raise RuntimeError("need n*n matrix")
        n = U.rows
        if len(y) != n:
            raise ValueError("Value should be equal to n")
        x = copy(y)
        for i in xrange(n - 1, -1, -1):
            for j in xrange(i + 1, n):
                x[i] -= U[i,j] * x[j]
            x[i] /= U[i,i]
        return x

    def lu_solve(ctx, A, b, **kwargs):
        """
        Ax = b => x

        Solve a determined or overdetermined linear equations system.
        Fast LU decomposition is used, which is less accurate than QR decomposition
        (especially for overdetermined systems), but it's twice as efficient.
        Use qr_solve if you want more precision or have to solve a very ill-
        conditioned system.

        If you specify real=True, it does not check for overdeterminded complex
        systems.
        """
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A nor b
            A, b = ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy()
            if A.rows < A.cols:
                raise ValueError('cannot solve underdetermined system')
            if A.rows > A.cols:
                # use least-squares method if overdetermined
                # (this increases errors)
                AH = A.H
                A = AH * A
                b = AH * b
                if (kwargs.get('real', False) or
                    not sum(type(i) is ctx.mpc for i in A)):
                    # TODO: necessary to check also b?
                    x = ctx.cholesky_solve(A, b)
                else:
                    x = ctx.lu_solve(A, b)
            else:
                # LU factorization
                A, p = ctx.LU_decomp(A)
                b = ctx.L_solve(A, b, p)
                x = ctx.U_solve(A, b)
        finally:
            ctx.prec = prec
        return x

    def improve_solution(ctx, A, x, b, maxsteps=1):
        """
        Improve a solution to a linear equation system iteratively.

        This re-uses the LU decomposition and is thus cheap.
        Usually 3 up to 4 iterations are giving the maximal improvement.
        """
        if A.rows != A.cols:
            raise RuntimeError("need n*n matrix") # TODO: really?
        for _ in xrange(maxsteps):
            r = ctx.residual(A, x, b)
            if ctx.norm(r, 2) < 10*ctx.eps:
                break
            # this uses cached LU decomposition and is thus cheap
            dx = ctx.lu_solve(A, -r)
            x += dx
        return x

    def lu(ctx, A):
        """
        A -> P, L, U

        LU factorisation of a square matrix A. L is the lower, U the upper part.
        P is the permutation matrix indicating the row swaps.

        P*A = L*U

        If you need efficiency, use the low-level method LU_decomp instead, it's
        much more memory efficient.
        """
        # get factorization
        A, p = ctx.LU_decomp(A)
        n = A.rows
        L = ctx.matrix(n)
        U = ctx.matrix(n)
        for i in xrange(n):
            for j in xrange(n):
                if i > j:
                    L[i,j] = A[i,j]
                elif i == j:
                    L[i,j] = 1
                    U[i,j] = A[i,j]
                else:
                    U[i,j] = A[i,j]
        # calculate permutation matrix
        P = ctx.eye(n)
        for k in xrange(len(p)):
            ctx.swap_row(P, k, p[k])
        return P, L, U

    def unitvector(ctx, n, i):
        """
        Return the i-th n-dimensional unit vector.
        """
        assert 0 < i <= n, 'this unit vector does not exist'
        return [ctx.zero]*(i-1) + [ctx.one] + [ctx.zero]*(n-i)

    def inverse(ctx, A, **kwargs):
        """
        Calculate the inverse of a matrix.

        If you want to solve an equation system Ax = b, it's recommended to use
        solve(A, b) instead, it's about 3 times more efficient.
        """
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A
            A = ctx.matrix(A, **kwargs).copy()
            n = A.rows
            # get LU factorisation
            A, p = ctx.LU_decomp(A)
            cols = []
            # calculate unit vectors and solve corresponding system to get columns
            for i in xrange(1, n + 1):
                e = ctx.unitvector(n, i)
                y = ctx.L_solve(A, e, p)
                cols.append(ctx.U_solve(A, y))
            # convert columns to matrix
            inv = []
            for i in xrange(n):
                row = []
                for j in xrange(n):
                    row.append(cols[j][i])
                inv.append(row)
            result = ctx.matrix(inv, **kwargs)
        finally:
            ctx.prec = prec
        return result

    def householder(ctx, A):
        """
        (A|b) -> H, p, x, res

        (A|b) is the coefficient matrix with left hand side of an optionally
        overdetermined linear equation system.
        H and p contain all information about the transformation matrices.
        x is the solution, res the residual.
        """
        if not isinstance(A, ctx.matrix):
            raise TypeError("A should be a type of ctx.matrix")
        m = A.rows
        n = A.cols
        if m < n - 1:
            raise RuntimeError("Columns should not be less than rows")
        # calculate Householder matrix
        p = []
        for j in xrange(0, n - 1):
            s = ctx.fsum(abs(A[i,j])**2 for i in xrange(j, m))
            if not abs(s) > ctx.eps:
                raise ValueError('matrix is numerically singular')
            p.append(-ctx.sign(ctx.re(A[j,j])) * ctx.sqrt(s))
            kappa = ctx.one / (s - p[j] * A[j,j])
            A[j,j] -= p[j]
            for k in xrange(j+1, n):
                y = ctx.fsum(ctx.conj(A[i,j]) * A[i,k] for i in xrange(j, m)) * kappa
                for i in xrange(j, m):
                    A[i,k] -= A[i,j] * y
        # solve Rx = c1
        x = [A[i,n - 1] for i in xrange(n - 1)]
        for i in xrange(n - 2, -1, -1):
            x[i] -= ctx.fsum(A[i,j] * x[j] for j in xrange(i + 1, n - 1))
            x[i] /= p[i]
        # calculate residual
        if not m == n - 1:
            r = [A[m-1-i, n-1] for i in xrange(m - n + 1)]
        else:
            # determined system, residual should be 0
            r = [0]*m # maybe a bad idea, changing r[i] will change all elements
        return A, p, x, r

    #def qr(ctx, A):
    #    """
    #    A -> Q, R
    #
    #    QR factorisation of a square matrix A using Householder decomposition.
    #    Q is orthogonal, this leads to very few numerical errors.
    #
    #    A = Q*R
    #    """
    #    H, p, x, res = householder(A)
    # TODO: implement this

    def residual(ctx, A, x, b, **kwargs):
        """
        Calculate the residual of a solution to a linear equation system.

        r = A*x - b for A*x = b
        """
        oldprec = ctx.prec
        try:
            ctx.prec *= 2
            A, x, b = ctx.matrix(A, **kwargs), ctx.matrix(x, **kwargs), ctx.matrix(b, **kwargs)
            return A*x - b
        finally:
            ctx.prec = oldprec

    def qr_solve(ctx, A, b, norm=None, **kwargs):
        """
        Ax = b => x, ||Ax - b||

        Solve a determined or overdetermined linear equations system and
        calculate the norm of the residual (error).
        QR decomposition using Householder factorization is applied, which gives very
        accurate results even for ill-conditioned matrices. qr_solve is twice as
        efficient.
        """
        if norm is None:
            norm = ctx.norm
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A nor b
            A, b = ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy()
            if A.rows < A.cols:
                raise ValueError('cannot solve underdetermined system')
            H, p, x, r = ctx.householder(ctx.extend(A, b))
            res = ctx.norm(r)
            # calculate residual "manually" for determined systems
            if res == 0:
                res = ctx.norm(ctx.residual(A, x, b))
            return ctx.matrix(x, **kwargs), res
        finally:
            ctx.prec = prec

    def cholesky(ctx, A, tol=None):
        r"""
        Cholesky decomposition of a symmetric positive-definite matrix `A`.
        Returns a lower triangular matrix `L` such that `A = L \times L^T`.
        More generally, for a complex Hermitian positive-definite matrix,
        a Cholesky decomposition satisfying `A = L \times L^H` is returned.

        The Cholesky decomposition can be used to solve linear equation
        systems twice as efficiently as LU decomposition, or to
        test whether `A` is positive-definite.

        The optional parameter ``tol`` determines the tolerance for
        verifying positive-definiteness.

        **Examples**

        Cholesky decomposition of a positive-definite symmetric matrix::

            >>> from mpmath import *
            >>> mp.dps = 25; mp.pretty = True
            >>> A = eye(3) + hilbert(3)
            >>> nprint(A)
            [     2.0      0.5  0.333333]
            [     0.5  1.33333      0.25]
            [0.333333     0.25       1.2]
            >>> L = cholesky(A)
            >>> nprint(L)
            [ 1.41421      0.0      0.0]
            [0.353553  1.09924      0.0]
            [0.235702  0.15162  1.05899]
            >>> chop(A - L*L.T)
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]

        Cholesky decomposition of a Hermitian matrix::

            >>> A = eye(3) + matrix([[0,0.25j,-0.5j],[-0.25j,0,0],[0.5j,0,0]])
            >>> L = cholesky(A)
            >>> nprint(L)
            [          1.0                0.0                0.0]
            [(0.0 - 0.25j)  (0.968246 + 0.0j)                0.0]
            [ (0.0 + 0.5j)  (0.129099 + 0.0j)  (0.856349 + 0.0j)]
            >>> chop(A - L*L.H)
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]
            [0.0  0.0  0.0]

        Attempted Cholesky decomposition of a matrix that is not positive
        definite::

            >>> A = -eye(3) + hilbert(3)
            >>> L = cholesky(A)
            Traceback (most recent call last):
              ...
            ValueError: matrix is not positive-definite

        **References**

        1. [Wikipedia]_ http://en.wikipedia.org/wiki/Cholesky_decomposition

        """
        if not isinstance(A, ctx.matrix):
            raise RuntimeError("A should be a type of ctx.matrix")
        if not A.rows == A.cols:
            raise ValueError('need n*n matrix')
        if tol is None:
            tol = +ctx.eps
        n = A.rows
        L = ctx.matrix(n)
        for j in xrange(n):
            c = ctx.re(A[j,j])
            if abs(c-A[j,j]) > tol:
                raise ValueError('matrix is not Hermitian')
            s = c - ctx.fsum((L[j,k] for k in xrange(j)),
                absolute=True, squared=True)
            if s < tol:
                raise ValueError('matrix is not positive-definite')
            L[j,j] = ctx.sqrt(s)
            for i in xrange(j, n):
                it1 = (L[i,k] for k in xrange(j))
                it2 = (L[j,k] for k in xrange(j))
                t = ctx.fdot(it1, it2, conjugate=True)
                L[i,j] = (A[i,j] - t) / L[j,j]
        return L

    def cholesky_solve(ctx, A, b, **kwargs):
        """
        Ax = b => x

        Solve a symmetric positive-definite linear equation system.
        This is twice as efficient as lu_solve.

        Typical use cases:
        * A.T*A
        * Hessian matrix
        * differential equations
        """
        prec = ctx.prec
        try:
            ctx.prec += 10
            # do not overwrite A nor b
            A, b = ctx.matrix(A, **kwargs).copy(), ctx.matrix(b, **kwargs).copy()
            if A.rows !=  A.cols:
                raise ValueError('can only solve determined system')
            # Cholesky factorization
            L = ctx.cholesky(A)
            # solve
            n = L.rows
            if len(b) != n:
                raise ValueError("Value should be equal to n")
            for i in xrange(n):
                b[i] -= ctx.fsum(L[i,j] * b[j] for j in xrange(i))
                b[i] /= L[i,i]
            x = ctx.U_solve(L.T, b)
            return x
        finally:
            ctx.prec = prec

    def det(ctx, A):
        """
        Calculate the determinant of a matrix.
        """
        prec = ctx.prec
        try:
            # do not overwrite A
            A = ctx.matrix(A).copy()
            # use LU factorization to calculate determinant
            try:
                R, p = ctx.LU_decomp(A)
            except ZeroDivisionError:
                return 0
            z = 1
            for i, e in enumerate(p):
                if i != e:
                    z *= -1
            for i in xrange(A.rows):
                z *= R[i,i]
            return z
        finally:
            ctx.prec = prec

    def cond(ctx, A, norm=None):
        """
        Calculate the condition number of a matrix using a specified matrix norm.

        The condition number estimates the sensitivity of a matrix to errors.
        Example: small input errors for ill-conditioned coefficient matrices
        alter the solution of the system dramatically.

        For ill-conditioned matrices it's recommended to use qr_solve() instead
        of lu_solve(). This does not help with input errors however, it just avoids
        to add additional errors.

        Definition:    cond(A) = ||A|| * ||A**-1||
        """
        if norm is None:
            norm = lambda x: ctx.mnorm(x,1)
        return norm(A) * norm(ctx.inverse(A))

    def lu_solve_mat(ctx, a, b):
        """Solve a * x = b  where a and b are matrices."""
        r = ctx.matrix(a.rows, b.cols)
        for i in range(b.cols):
            c = ctx.lu_solve(a, b.column(i))
            for j in range(len(c)):
                r[j, i] = c[j]
        return r

    def qr(ctx, A, mode = 'full', edps = 10):
        """
        Compute a QR factorization $A = QR$ where
        A is an m x n matrix of real or complex numbers where m >= n

        mode has following meanings:
        (1) mode = 'raw' returns two matrixes (A, tau) in the
            internal format used by LAPACK
        (2) mode = 'skinny' returns the leading n columns of Q
            and n rows of R
        (3) Any other value returns the leading m columns of Q
            and m rows of R

        edps is the increase in mp precision used for calculations

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> mp.pretty = True
            >>> A = matrix([[1, 2], [3, 4], [1, 1]])
            >>> Q, R = qr(A)
            >>> Q
            [-0.301511344577764   0.861640436855329   0.408248290463863]
            [-0.904534033733291  -0.123091490979333  -0.408248290463863]
            [-0.301511344577764  -0.492365963917331   0.816496580927726]
            >>> R
            [-3.3166247903554  -4.52267016866645]
            [             0.0  0.738548945875996]
            [             0.0                0.0]
            >>> Q * R
            [1.0  2.0]
            [3.0  4.0]
            [1.0  1.0]
            >>> chop(Q.T * Q)
            [1.0  0.0  0.0]
            [0.0  1.0  0.0]
            [0.0  0.0  1.0]
            >>> B = matrix([[1+0j, 2-3j], [3+j, 4+5j]])
            >>> Q, R = qr(B)
            >>> nprint(Q)
            [     (-0.301511 + 0.0j)   (0.0695795 - 0.95092j)]
            [(-0.904534 - 0.301511j)  (-0.115966 + 0.278318j)]
            >>> nprint(R)
            [(-3.31662 + 0.0j)  (-5.72872 - 2.41209j)]
            [              0.0       (3.91965 + 0.0j)]
            >>> Q * R
            [(1.0 + 0.0j)  (2.0 - 3.0j)]
            [(3.0 + 1.0j)  (4.0 + 5.0j)]
            >>> chop(Q.T * Q.conjugate())
            [1.0  0.0]
            [0.0  1.0]

        """

        # check values before continuing
        assert isinstance(A, ctx.matrix)
        m = A.rows
        n = A.cols
        assert n >= 0
        assert m >= n
        assert edps >= 0

        # check for complex data type
        cmplx = any(type(x) is ctx.mpc for x in A)

        # temporarily increase the precision and initialize
        with ctx.extradps(edps):
            tau = ctx.matrix(n,1)
            A = A.copy()

            # ---------------
            # FACTOR MATRIX A
            # ---------------
            if cmplx:
                one = ctx.mpc('1.0', '0.0')
                zero = ctx.mpc('0.0', '0.0')
                rzero = ctx.mpf('0.0')

                # main loop to factor A (complex)
                for j in xrange(0, n):
                    alpha = A[j,j]
                    alphr = ctx.re(alpha)
                    alphi = ctx.im(alpha)

                    if (m-j) >= 2:
                        xnorm = ctx.fsum( A[i,j]*ctx.conj(A[i,j]) for i in xrange(j+1, m) )
                        xnorm = ctx.re( ctx.sqrt(xnorm) )
                    else:
                        xnorm = rzero

                    if (xnorm == rzero) and (alphi == rzero):
                        tau[j] = zero
                        continue

                    if alphr < rzero:
                        beta = ctx.sqrt(alphr**2 + alphi**2 + xnorm**2)
                    else:
                        beta = -ctx.sqrt(alphr**2 + alphi**2 + xnorm**2)

                    tau[j] = ctx.mpc( (beta - alphr) / beta, -alphi / beta )
                    t = -ctx.conj(tau[j])
                    za = one / (alpha - beta)

                    for i in xrange(j+1, m):
                        A[i,j] *= za

                    A[j,j] = one
                    for k in xrange(j+1, n):
                        y = ctx.fsum(A[i,j] * ctx.conj(A[i,k]) for i in xrange(j, m))
                        temp = t * ctx.conj(y)
                        for i in xrange(j, m):
                            A[i,k] += A[i,j] * temp

                    A[j,j] = ctx.mpc(beta, '0.0')
            else:
                one = ctx.mpf('1.0')
                zero = ctx.mpf('0.0')

                # main loop to factor A (real)
                for j in xrange(0, n):
                    alpha = A[j,j]

                    if (m-j) > 2:
                        xnorm = ctx.fsum( (A[i,j])**2 for i in xrange(j+1, m) )
                        xnorm = ctx.sqrt(xnorm)
                    elif (m-j) == 2:
                        xnorm = abs( A[m-1,j] )
                    else:
                        xnorm = zero

                    if xnorm == zero:
                        tau[j] = zero
                        continue

                    if alpha < zero:
                        beta = ctx.sqrt(alpha**2 + xnorm**2)
                    else:
                        beta = -ctx.sqrt(alpha**2 + xnorm**2)

                    tau[j] = (beta - alpha) / beta
                    t = -tau[j]
                    da = one / (alpha - beta)

                    for i in xrange(j+1, m):
                        A[i,j] *= da

                    A[j,j] = one
                    for k in xrange(j+1, n):
                        y = ctx.fsum( A[i,j] * A[i,k] for i in xrange(j, m) )
                        temp = t * y
                        for i in xrange(j,m):
                            A[i,k] += A[i,j] * temp

                    A[j,j] = beta

            # return factorization in same internal format as LAPACK
            if (mode == 'raw') or (mode == 'RAW'):
                return A, tau

            # ----------------------------------
            # FORM Q USING BACKWARD ACCUMULATION
            # ----------------------------------

            # form R before the values are overwritten
            R = A.copy()
            for j in xrange(0, n):
                for i in xrange(j+1, m):
                    R[i,j] = zero

            # set the value of p (number of columns of Q to return)
            p = m
            if (mode == 'skinny') or (mode == 'SKINNY'):
                p = n

            # add columns to A if needed and initialize
            A.cols += (p-n)
            for j in xrange(0, p):
                A[j,j] = one
                for i in xrange(0, j):
                    A[i,j] = zero

            # main loop to form Q
            for j in xrange(n-1, -1, -1):
                t = -tau[j]
                A[j,j] += t

                for k in xrange(j+1, p):
                    if cmplx:
                        y = ctx.fsum(A[i,j] * ctx.conj(A[i,k]) for i in xrange(j+1, m))
                        temp = t * ctx.conj(y)
                    else:
                        y = ctx.fsum(A[i,j] * A[i,k] for i in xrange(j+1, m))
                        temp = t * y
                    A[j,k] = temp
                    for i in xrange(j+1, m):
                        A[i,k] += A[i,j] * temp

                for i in xrange(j+1, m):
                    A[i, j] *= t

            return A, R[0:p,0:n]

        # ------------------
        # END OF FUNCTION QR
        # ------------------