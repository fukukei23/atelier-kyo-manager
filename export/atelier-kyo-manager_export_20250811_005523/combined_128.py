
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\circuitutils.py ===
"""Primitive circuit operations on quantum circuits."""

from functools import reduce

from sympy.core.sorting import default_sort_key
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.utilities import numbered_symbols
from sympy.physics.quantum.gate import Gate

__all__ = [
    'kmp_table',
    'find_subcircuit',
    'replace_subcircuit',
    'convert_to_symbolic_indices',
    'convert_to_real_indices',
    'random_reduce',
    'random_insert'
]


def kmp_table(word):
    """Build the 'partial match' table of the Knuth-Morris-Pratt algorithm.

    Note: This is applicable to strings or
    quantum circuits represented as tuples.
    """

    # Current position in subcircuit
    pos = 2
    # Beginning position of candidate substring that
    # may reappear later in word
    cnd = 0
    # The 'partial match' table that helps one determine
    # the next location to start substring search
    table = []
    table.append(-1)
    table.append(0)

    while pos < len(word):
        if word[pos - 1] == word[cnd]:
            cnd = cnd + 1
            table.append(cnd)
            pos = pos + 1
        elif cnd > 0:
            cnd = table[cnd]
        else:
            table.append(0)
            pos = pos + 1

    return table


def find_subcircuit(circuit, subcircuit, start=0, end=0):
    """Finds the subcircuit in circuit, if it exists.

    Explanation
    ===========

    If the subcircuit exists, the index of the start of
    the subcircuit in circuit is returned; otherwise,
    -1 is returned.  The algorithm that is implemented
    is the Knuth-Morris-Pratt algorithm.

    Parameters
    ==========

    circuit : tuple, Gate or Mul
        A tuple of Gates or Mul representing a quantum circuit
    subcircuit : tuple, Gate or Mul
        A tuple of Gates or Mul to find in circuit
    start : int
        The location to start looking for subcircuit.
        If start is the same or past end, -1 is returned.
    end : int
        The last place to look for a subcircuit.  If end
        is less than 1 (one), then the length of circuit
        is taken to be end.

    Examples
    ========

    Find the first instance of a subcircuit:

    >>> from sympy.physics.quantum.circuitutils import find_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> find_subcircuit(circuit, subcircuit)
    1

    Find the first instance starting at a specific position:

    >>> find_subcircuit(circuit, subcircuit, start=1)
    1

    >>> find_subcircuit(circuit, subcircuit, start=2)
    -1

    >>> circuit = circuit*subcircuit
    >>> find_subcircuit(circuit, subcircuit, start=2)
    4

    Find the subcircuit within some interval:

    >>> find_subcircuit(circuit, subcircuit, start=2, end=2)
    -1
    """

    if isinstance(circuit, Mul):
        circuit = circuit.args

    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args

    if len(subcircuit) == 0 or len(subcircuit) > len(circuit):
        return -1

    if end < 1:
        end = len(circuit)

    # Location in circuit
    pos = start
    # Location in the subcircuit
    index = 0
    # 'Partial match' table
    table = kmp_table(subcircuit)

    while (pos + index) < end:
        if subcircuit[index] == circuit[pos + index]:
            index = index + 1
        else:
            pos = pos + index - table[index]
            index = table[index] if table[index] > -1 else 0

        if index == len(subcircuit):
            return pos

    return -1


def replace_subcircuit(circuit, subcircuit, replace=None, pos=0):
    """Replaces a subcircuit with another subcircuit in circuit,
    if it exists.

    Explanation
    ===========

    If multiple instances of subcircuit exists, the first instance is
    replaced.  The position to being searching from (if different from
    0) may be optionally given.  If subcircuit cannot be found, circuit
    is returned.

    Parameters
    ==========

    circuit : tuple, Gate or Mul
        A quantum circuit.
    subcircuit : tuple, Gate or Mul
        The circuit to be replaced.
    replace : tuple, Gate or Mul
        The replacement circuit.
    pos : int
        The location to start search and replace
        subcircuit, if it exists.  This may be used
        if it is known beforehand that multiple
        instances exist, and it is desirable to
        replace a specific instance.  If a negative number
        is given, pos will be defaulted to 0.

    Examples
    ========

    Find and remove the subcircuit:

    >>> from sympy.physics.quantum.circuitutils import replace_subcircuit
    >>> from sympy.physics.quantum.gate import X, Y, Z, H
    >>> circuit = X(0)*Z(0)*Y(0)*H(0)*X(0)*H(0)*Y(0)
    >>> subcircuit = Z(0)*Y(0)
    >>> replace_subcircuit(circuit, subcircuit)
    (X(0), H(0), X(0), H(0), Y(0))

    Remove the subcircuit given a starting search point:

    >>> replace_subcircuit(circuit, subcircuit, pos=1)
    (X(0), H(0), X(0), H(0), Y(0))

    >>> replace_subcircuit(circuit, subcircuit, pos=2)
    (X(0), Z(0), Y(0), H(0), X(0), H(0), Y(0))

    Replace the subcircuit:

    >>> replacement = H(0)*Z(0)
    >>> replace_subcircuit(circuit, subcircuit, replace=replacement)
    (X(0), H(0), Z(0), H(0), X(0), H(0), Y(0))
    """

    if pos < 0:
        pos = 0

    if isinstance(circuit, Mul):
        circuit = circuit.args

    if isinstance(subcircuit, Mul):
        subcircuit = subcircuit.args

    if isinstance(replace, Mul):
        replace = replace.args
    elif replace is None:
        replace = ()

    # Look for the subcircuit starting at pos
    loc = find_subcircuit(circuit, subcircuit, start=pos)

    # If subcircuit was found
    if loc > -1:
        # Get the gates to the left of subcircuit
        left = circuit[0:loc]
        # Get the gates to the right of subcircuit
        right = circuit[loc + len(subcircuit):len(circuit)]
        # Recombine the left and right side gates into a circuit
        circuit = left + replace + right

    return circuit


def _sympify_qubit_map(mapping):
    new_map = {}
    for key in mapping:
        new_map[key] = sympify(mapping[key])
    return new_map


def convert_to_symbolic_indices(seq, start=None, gen=None, qubit_map=None):
    """Returns the circuit with symbolic indices and the
    dictionary mapping symbolic indices to real indices.

    The mapping is 1 to 1 and onto (bijective).

    Parameters
    ==========

    seq : tuple, Gate/Integer/tuple or Mul
        A tuple of Gate, Integer, or tuple objects, or a Mul
    start : Symbol
        An optional starting symbolic index
    gen : object
        An optional numbered symbol generator
    qubit_map : dict
        An existing mapping of symbolic indices to real indices

    All symbolic indices have the format 'i#', where # is
    some number >= 0.
    """

    if isinstance(seq, Mul):
        seq = seq.args

    # A numbered symbol generator
    index_gen = numbered_symbols(prefix='i', start=-1)
    cur_ndx = next(index_gen)

    # keys are symbolic indices; values are real indices
    ndx_map = {}

    def create_inverse_map(symb_to_real_map):
        rev_items = lambda item: (item[1], item[0])
        return dict(map(rev_items, symb_to_real_map.items()))

    if start is not None:
        if not isinstance(start, Symbol):
            msg = 'Expected Symbol for starting index, got %r.' % start
            raise TypeError(msg)
        cur_ndx = start

    if gen is not None:
        if not isinstance(gen, numbered_symbols().__class__):
            msg = 'Expected a generator, got %r.' % gen
            raise TypeError(msg)
        index_gen = gen

    if qubit_map is not None:
        if not isinstance(qubit_map, dict):
            msg = ('Expected dict for existing map, got ' +
                   '%r.' % qubit_map)
            raise TypeError(msg)
        ndx_map = qubit_map

    ndx_map = _sympify_qubit_map(ndx_map)
    # keys are real indices; keys are symbolic indices
    inv_map = create_inverse_map(ndx_map)

    sym_seq = ()
    for item in seq:
        # Nested items, so recurse
        if isinstance(item, Gate):
            result = convert_to_symbolic_indices(item.args,
                                                 qubit_map=ndx_map,
                                                 start=cur_ndx,
                                                 gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)

        elif isinstance(item, (tuple, Tuple)):
            result = convert_to_symbolic_indices(item,
                                                 qubit_map=ndx_map,
                                                 start=cur_ndx,
                                                 gen=index_gen)
            sym_item, new_map, cur_ndx, index_gen = result
            ndx_map.update(new_map)
            inv_map = create_inverse_map(ndx_map)

        elif item in inv_map:
            sym_item = inv_map[item]

        else:
            cur_ndx = next(gen)
            ndx_map[cur_ndx] = item
            inv_map[item] = cur_ndx
            sym_item = cur_ndx

        if isinstance(item, Gate):
            sym_item = item.__class__(*sym_item)

        sym_seq = sym_seq + (sym_item,)

    return sym_seq, ndx_map, cur_ndx, index_gen


def convert_to_real_indices(seq, qubit_map):
    """Returns the circuit with real indices.

    Parameters
    ==========

    seq : tuple, Gate/Integer/tuple or Mul
        A tuple of Gate, Integer, or tuple objects or a Mul
    qubit_map : dict
        A dictionary mapping symbolic indices to real indices.

    Examples
    ========

    Change the symbolic indices to real integers:

    >>> from sympy import symbols
    >>> from sympy.physics.quantum.circuitutils import convert_to_real_indices
    >>> from sympy.physics.quantum.gate import X, Y, H
    >>> i0, i1 = symbols('i:2')
    >>> index_map = {i0 : 0, i1 : 1}
    >>> convert_to_real_indices(X(i0)*Y(i1)*H(i0)*X(i1), index_map)
    (X(0), Y(1), H(0), X(1))
    """

    if isinstance(seq, Mul):
        seq = seq.args

    if not isinstance(qubit_map, dict):
        msg = 'Expected dict for qubit_map, got %r.' % qubit_map
        raise TypeError(msg)

    qubit_map = _sympify_qubit_map(qubit_map)
    real_seq = ()
    for item in seq:
        # Nested items, so recurse
        if isinstance(item, Gate):
            real_item = convert_to_real_indices(item.args, qubit_map)

        elif isinstance(item, (tuple, Tuple)):
            real_item = convert_to_real_indices(item, qubit_map)

        else:
            real_item = qubit_map[item]

        if isinstance(item, Gate):
            real_item = item.__class__(*real_item)

        real_seq = real_seq + (real_item,)

    return real_seq


def random_reduce(circuit, gate_ids, seed=None):
    """Shorten the length of a quantum circuit.

    Explanation
    ===========

    random_reduce looks for circuit identities in circuit, randomly chooses
    one to remove, and returns a shorter yet equivalent circuit.  If no
    identities are found, the same circuit is returned.

    Parameters
    ==========

    circuit : Gate tuple of Mul
        A tuple of Gates representing a quantum circuit
    gate_ids : list, GateIdentity
        List of gate identities to find in circuit
    seed : int or list
        seed used for _randrange; to override the random selection, provide a
        list of integers: the elements of gate_ids will be tested in the order
        given by the list

    """
    from sympy.core.random import _randrange

    if not gate_ids:
        return circuit

    if isinstance(circuit, Mul):
        circuit = circuit.args

    ids = flatten_ids(gate_ids)

    # Create the random integer generator with the seed
    randrange = _randrange(seed)

    # Look for an identity in the circuit
    while ids:
        i = randrange(len(ids))
        id = ids.pop(i)
        if find_subcircuit(circuit, id) != -1:
            break
    else:
        # no identity was found
        return circuit

    # return circuit with the identity removed
    return replace_subcircuit(circuit, id)


def random_insert(circuit, choices, seed=None):
    """Insert a circuit into another quantum circuit.

    Explanation
    ===========

    random_insert randomly chooses a location in the circuit to insert
    a randomly selected circuit from amongst the given choices.

    Parameters
    ==========

    circuit : Gate tuple or Mul
        A tuple or Mul of Gates representing a quantum circuit
    choices : list
        Set of circuit choices
    seed : int or list
        seed used for _randrange; to override the random selections, give
        a list two integers, [i, j] where i is the circuit location where
        choice[j] will be inserted.

    Notes
    =====

    Indices for insertion should be [0, n] if n is the length of the
    circuit.
    """
    from sympy.core.random import _randrange

    if not choices:
        return circuit

    if isinstance(circuit, Mul):
        circuit = circuit.args

    # get the location in the circuit and the element to insert from choices
    randrange = _randrange(seed)
    loc = randrange(len(circuit) + 1)
    choice = choices[randrange(len(choices))]

    circuit = list(circuit)
    circuit[loc: loc] = choice
    return tuple(circuit)

# Flatten the GateIdentity objects (with gate rules) into one single list


def flatten_ids(ids):
    collapse = lambda acc, an_id: acc + sorted(an_id.equivalent_ids,
                                        key=default_sort_key)
    ids = reduce(collapse, ids, [])
    ids.sort(key=default_sort_key)
    return ids

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_utils\_pep440.py ===
"""Utility to compare pep440 compatible version strings.

The LooseVersion and StrictVersion classes that distutils provides don't
work; they don't recognize anything like alpha/beta/rc/dev versions.
"""

# Copyright (c) Donald Stufft and individual contributors.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.

#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import collections
import itertools
import re

__all__ = [
    "parse", "Version", "LegacyVersion", "InvalidVersion", "VERSION_PATTERN",
]


# BEGIN packaging/_structures.py


class Infinity:
    def __repr__(self):
        return "Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not isinstance(other, self.__class__)

    def __gt__(self, other):
        return True

    def __ge__(self, other):
        return True

    def __neg__(self):
        return NegativeInfinity


Infinity = Infinity()


class NegativeInfinity:
    def __repr__(self):
        return "-Infinity"

    def __hash__(self):
        return hash(repr(self))

    def __lt__(self, other):
        return True

    def __le__(self, other):
        return True

    def __eq__(self, other):
        return isinstance(other, self.__class__)

    def __ne__(self, other):
        return not isinstance(other, self.__class__)

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def __neg__(self):
        return Infinity


# BEGIN packaging/version.py


NegativeInfinity = NegativeInfinity()

_Version = collections.namedtuple(
    "_Version",
    ["epoch", "release", "dev", "pre", "post", "local"],
)


def parse(version):
    """
    Parse the given version string and return either a :class:`Version` object
    or a :class:`LegacyVersion` object depending on if the given version is
    a valid PEP 440 version or a legacy version.
    """
    try:
        return Version(version)
    except InvalidVersion:
        return LegacyVersion(version)


class InvalidVersion(ValueError):
    """
    An invalid version was found, users should refer to PEP 440.
    """


class _BaseVersion:

    def __hash__(self):
        return hash(self._key)

    def __lt__(self, other):
        return self._compare(other, lambda s, o: s < o)

    def __le__(self, other):
        return self._compare(other, lambda s, o: s <= o)

    def __eq__(self, other):
        return self._compare(other, lambda s, o: s == o)

    def __ge__(self, other):
        return self._compare(other, lambda s, o: s >= o)

    def __gt__(self, other):
        return self._compare(other, lambda s, o: s > o)

    def __ne__(self, other):
        return self._compare(other, lambda s, o: s != o)

    def _compare(self, other, method):
        if not isinstance(other, _BaseVersion):
            return NotImplemented

        return method(self._key, other._key)


class LegacyVersion(_BaseVersion):

    def __init__(self, version):
        self._version = str(version)
        self._key = _legacy_cmpkey(self._version)

    def __str__(self):
        return self._version

    def __repr__(self):
        return f"<LegacyVersion({str(self)!r})>"

    @property
    def public(self):
        return self._version

    @property
    def base_version(self):
        return self._version

    @property
    def local(self):
        return None

    @property
    def is_prerelease(self):
        return False

    @property
    def is_postrelease(self):
        return False


_legacy_version_component_re = re.compile(
    r"(\d+ | [a-z]+ | \.| -)", re.VERBOSE,
)

_legacy_version_replacement_map = {
    "pre": "c", "preview": "c", "-": "final-", "rc": "c", "dev": "@",
}


def _parse_version_parts(s):
    for part in _legacy_version_component_re.split(s):
        part = _legacy_version_replacement_map.get(part, part)

        if not part or part == ".":
            continue

        if part[:1] in "0123456789":
            # pad for numeric comparison
            yield part.zfill(8)
        else:
            yield "*" + part

    # ensure that alpha/beta/candidate are before final
    yield "*final"


def _legacy_cmpkey(version):
    # We hardcode an epoch of -1 here. A PEP 440 version can only have an epoch
    # greater than or equal to 0. This will effectively put the LegacyVersion,
    # which uses the defacto standard originally implemented by setuptools,
    # as before all PEP 440 versions.
    epoch = -1

    # This scheme is taken from pkg_resources.parse_version setuptools prior to
    # its adoption of the packaging library.
    parts = []
    for part in _parse_version_parts(version.lower()):
        if part.startswith("*"):
            # remove "-" before a prerelease tag
            if part < "*final":
                while parts and parts[-1] == "*final-":
                    parts.pop()

            # remove trailing zeros from each series of numeric parts
            while parts and parts[-1] == "00000000":
                parts.pop()

        parts.append(part)
    parts = tuple(parts)

    return epoch, parts


# Deliberately not anchored to the start and end of the string, to make it
# easier for 3rd party code to reuse
VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""


class Version(_BaseVersion):

    _regex = re.compile(
        r"^\s*" + VERSION_PATTERN + r"\s*$",
        re.VERBOSE | re.IGNORECASE,
    )

    def __init__(self, version):
        # Validate the version and parse it into pieces
        match = self._regex.search(version)
        if not match:
            raise InvalidVersion(f"Invalid version: '{version}'")

        # Store the parsed out pieces of the version
        self._version = _Version(
            epoch=int(match.group("epoch")) if match.group("epoch") else 0,
            release=tuple(int(i) for i in match.group("release").split(".")),
            pre=_parse_letter_version(
                match.group("pre_l"),
                match.group("pre_n"),
            ),
            post=_parse_letter_version(
                match.group("post_l"),
                match.group("post_n1") or match.group("post_n2"),
            ),
            dev=_parse_letter_version(
                match.group("dev_l"),
                match.group("dev_n"),
            ),
            local=_parse_local_version(match.group("local")),
        )

        # Generate a key which will be used for sorting
        self._key = _cmpkey(
            self._version.epoch,
            self._version.release,
            self._version.pre,
            self._version.post,
            self._version.dev,
            self._version.local,
        )

    def __repr__(self):
        return f"<Version({str(self)!r})>"

    def __str__(self):
        parts = []

        # Epoch
        if self._version.epoch != 0:
            parts.append(f"{self._version.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self._version.release))

        # Pre-release
        if self._version.pre is not None:
            parts.append("".join(str(x) for x in self._version.pre))

        # Post-release
        if self._version.post is not None:
            parts.append(f".post{self._version.post[1]}")

        # Development release
        if self._version.dev is not None:
            parts.append(f".dev{self._version.dev[1]}")

        # Local version segment
        if self._version.local is not None:
            parts.append(
                f"+{'.'.join(str(x) for x in self._version.local)}"
            )

        return "".join(parts)

    @property
    def public(self):
        return str(self).split("+", 1)[0]

    @property
    def base_version(self):
        parts = []

        # Epoch
        if self._version.epoch != 0:
            parts.append(f"{self._version.epoch}!")

        # Release segment
        parts.append(".".join(str(x) for x in self._version.release))

        return "".join(parts)

    @property
    def local(self):
        version_string = str(self)
        if "+" in version_string:
            return version_string.split("+", 1)[1]

    @property
    def is_prerelease(self):
        return bool(self._version.dev or self._version.pre)

    @property
    def is_postrelease(self):
        return bool(self._version.post)


def _parse_letter_version(letter, number):
    if letter:
        # We assume there is an implicit 0 in a pre-release if there is
        # no numeral associated with it.
        if number is None:
            number = 0

        # We normalize any letters to their lower-case form
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        if letter == "alpha":
            letter = "a"
        elif letter == "beta":
            letter = "b"
        elif letter in ["c", "pre", "preview"]:
            letter = "rc"
        elif letter in ["rev", "r"]:
            letter = "post"

        return letter, int(number)
    if not letter and number:
        # We assume that if we are given a number but not given a letter,
        # then this is using the implicit post release syntax (e.g., 1.0-1)
        letter = "post"

        return letter, int(number)


_local_version_seperators = re.compile(r"[\._-]")


def _parse_local_version(local):
    """
    Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
    """
    if local is not None:
        return tuple(
            part.lower() if not part.isdigit() else int(part)
            for part in _local_version_seperators.split(local)
        )


def _cmpkey(epoch, release, pre, post, dev, local):
    # When we compare a release version, we want to compare it with all of the
    # trailing zeros removed. So we'll use a reverse the list, drop all the now
    # leading zeros until we come to something non-zero, then take the rest,
    # re-reverse it back into the correct order, and make it a tuple and use
    # that for our sorting key.
    release = tuple(
        reversed(list(
            itertools.dropwhile(
                lambda x: x == 0,
                reversed(release),
            )
        ))
    )

    # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
    # We'll do this by abusing the pre-segment, but we _only_ want to do this
    # if there is no pre- or a post-segment. If we have one of those, then
    # the normal sorting rules will handle this case correctly.
    if pre is None and post is None and dev is not None:
        pre = -Infinity
    # Versions without a pre-release (except as noted above) should sort after
    # those with one.
    elif pre is None:
        pre = Infinity

    # Versions without a post-segment should sort before those with one.
    if post is None:
        post = -Infinity

    # Versions without a development segment should sort after those with one.
    if dev is None:
        dev = Infinity

    if local is None:
        # Versions without a local segment should sort before those with one.
        local = -Infinity
    else:
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alphanumeric segments sort before numeric segments
        # - Alphanumeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        local = tuple(
            (i, "") if isinstance(i, int) else (-Infinity, i)
            for i in local
        )

    return epoch, release, pre, post, dev, local

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\filters.py ===
# Copyright (c) 2010-2024 openpyxl

import re

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Alias,
    Typed,
    Set,
    Float,
    DateTime,
    NoneSet,
    Bool,
    Integer,
    String,
    Sequence,
    MinMax,
)
from openpyxl.descriptors.excel import ExtensionList, CellRange
from openpyxl.descriptors.sequence import ValueSequence
from openpyxl.utils import absolute_coordinate


class SortCondition(Serialisable):

    tagname = "sortCondition"

    descending = Bool(allow_none=True)
    sortBy = NoneSet(values=(['value', 'cellColor', 'fontColor', 'icon']))
    ref = CellRange()
    customList = String(allow_none=True)
    dxfId = Integer(allow_none=True)
    iconSet = NoneSet(values=(['3Arrows', '3ArrowsGray', '3Flags',
                           '3TrafficLights1', '3TrafficLights2', '3Signs', '3Symbols', '3Symbols2',
                           '4Arrows', '4ArrowsGray', '4RedToBlack', '4Rating', '4TrafficLights',
                           '5Arrows', '5ArrowsGray', '5Rating', '5Quarters']))
    iconId = Integer(allow_none=True)

    def __init__(self,
                 ref=None,
                 descending=None,
                 sortBy=None,
                 customList=None,
                 dxfId=None,
                 iconSet=None,
                 iconId=None,
                ):
        self.descending = descending
        self.sortBy = sortBy
        self.ref = ref
        self.customList = customList
        self.dxfId = dxfId
        self.iconSet = iconSet
        self.iconId = iconId


class SortState(Serialisable):

    tagname = "sortState"

    columnSort = Bool(allow_none=True)
    caseSensitive = Bool(allow_none=True)
    sortMethod = NoneSet(values=(['stroke', 'pinYin']))
    ref = CellRange()
    sortCondition = Sequence(expected_type=SortCondition, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('sortCondition',)

    def __init__(self,
                 columnSort=None,
                 caseSensitive=None,
                 sortMethod=None,
                 ref=None,
                 sortCondition=(),
                 extLst=None,
                ):
        self.columnSort = columnSort
        self.caseSensitive = caseSensitive
        self.sortMethod = sortMethod
        self.ref = ref
        self.sortCondition = sortCondition


    def __bool__(self):
        return self.ref is not None



class IconFilter(Serialisable):

    tagname = "iconFilter"

    iconSet = Set(values=(['3Arrows', '3ArrowsGray', '3Flags',
                           '3TrafficLights1', '3TrafficLights2', '3Signs', '3Symbols', '3Symbols2',
                           '4Arrows', '4ArrowsGray', '4RedToBlack', '4Rating', '4TrafficLights',
                           '5Arrows', '5ArrowsGray', '5Rating', '5Quarters']))
    iconId = Integer(allow_none=True)

    def __init__(self,
                 iconSet=None,
                 iconId=None,
                ):
        self.iconSet = iconSet
        self.iconId = iconId


class ColorFilter(Serialisable):

    tagname = "colorFilter"

    dxfId = Integer(allow_none=True)
    cellColor = Bool(allow_none=True)

    def __init__(self,
                 dxfId=None,
                 cellColor=None,
                ):
        self.dxfId = dxfId
        self.cellColor = cellColor


class DynamicFilter(Serialisable):

    tagname = "dynamicFilter"

    type = Set(values=(['null', 'aboveAverage', 'belowAverage', 'tomorrow',
                        'today', 'yesterday', 'nextWeek', 'thisWeek', 'lastWeek', 'nextMonth',
                        'thisMonth', 'lastMonth', 'nextQuarter', 'thisQuarter', 'lastQuarter',
                        'nextYear', 'thisYear', 'lastYear', 'yearToDate', 'Q1', 'Q2', 'Q3', 'Q4',
                        'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'M10', 'M11',
                        'M12']))
    val = Float(allow_none=True)
    valIso = DateTime(allow_none=True)
    maxVal = Float(allow_none=True)
    maxValIso = DateTime(allow_none=True)

    def __init__(self,
                 type=None,
                 val=None,
                 valIso=None,
                 maxVal=None,
                 maxValIso=None,
                ):
        self.type = type
        self.val = val
        self.valIso = valIso
        self.maxVal = maxVal
        self.maxValIso = maxValIso


class CustomFilter(Serialisable):

    tagname = "customFilter"

    val = String()
    operator = Set(values=['equal', 'lessThan', 'lessThanOrEqual',
                           'notEqual', 'greaterThanOrEqual', 'greaterThan'])

    def __init__(self, operator="equal", val=None):
        self.operator = operator
        self.val = val


    def _get_subtype(self):
        if self.val == " ":
            subtype = BlankFilter
        else:
            try:
                float(self.val)
                subtype = NumberFilter
            except ValueError:
                subtype = StringFilter
        return subtype


    def convert(self):
        """Convert to more specific filter"""
        typ = self._get_subtype()
        if typ in (BlankFilter, NumberFilter):
            return typ(**dict(self))

        operator, term = StringFilter._guess_operator(self.val)
        flt = StringFilter(operator, term)
        if self.operator == "notEqual":
            flt.exclude = True
        return flt


class BlankFilter(CustomFilter):
    """
    Exclude blanks
    """

    __attrs__ = ("operator", "val")

    def __init__(self, **kw):
        pass


    @property
    def operator(self):
        return "notEqual"


    @property
    def val(self):
        return " "


class NumberFilter(CustomFilter):


    operator = Set(values=
                   ['equal', 'lessThan', 'lessThanOrEqual',
                    'notEqual', 'greaterThanOrEqual', 'greaterThan'])
    val = Float()

    def __init__(self, operator="equal", val=None):
        self.operator = operator
        self.val = val


string_format_mapping = {
    "contains": "*{}*",
    "startswith": "{}*",
    "endswith": "*{}",
    "wildcard":  "{}",
}


class StringFilter(CustomFilter):

    operator = Set(values=['contains', 'startswith', 'endswith', 'wildcard']
                   )
    val = String()
    exclude = Bool()


    def __init__(self, operator="contains", val=None, exclude=False):
        self.operator = operator
        self.val = val
        self.exclude = exclude


    def _escape(self):
        """Escape wildcards ~, * ? when serialising"""
        if self.operator == "wildcard":
            return self.val
        return re.sub(r"~|\*|\?", r"~\g<0>", self.val)


    @staticmethod
    def _unescape(value):
        """
        Unescape value
        """
        return re.sub(r"~(?P<op>[~*?])", r"\g<op>", value)


    @staticmethod
    def _guess_operator(value):
        value = StringFilter._unescape(value)
        endswith = r"^(?P<endswith>\*)(?P<term>[^\*\?]*$)"
        startswith = r"^(?P<term>[^\*\?]*)(?P<startswith>\*)$"
        contains = r"^(?P<contains>\*)(?P<term>[^\*\?]*)\*$"
        d = {"wildcard": True, "term": value}
        for pat in [contains, startswith, endswith]:
            m = re.match(pat, value)
            if m:
                d = m.groupdict()

        term = d.pop("term")
        op = list(d)[0]
        return op, term


    def to_tree(self, tagname=None, idx=None, namespace=None):
        fmt = string_format_mapping[self.operator]
        op = self.exclude and "notEqual" or "equal"
        value = fmt.format(self._escape())
        flt = CustomFilter(op, value)
        return flt.to_tree(tagname, idx, namespace)


class CustomFilters(Serialisable):

    tagname = "customFilters"

    _and = Bool(allow_none=True)
    customFilter = Sequence(expected_type=CustomFilter) # min 1, max 2

    __elements__ = ('customFilter',)

    def __init__(self,
                 _and=None,
                 customFilter=(),
                ):
        self._and = _and
        self.customFilter = customFilter


class Top10(Serialisable):

    tagname = "top10"

    top = Bool(allow_none=True)
    percent = Bool(allow_none=True)
    val = Float()
    filterVal = Float(allow_none=True)

    def __init__(self,
                 top=None,
                 percent=None,
                 val=None,
                 filterVal=None,
                ):
        self.top = top
        self.percent = percent
        self.val = val
        self.filterVal = filterVal


class DateGroupItem(Serialisable):

    tagname = "dateGroupItem"

    year = Integer()
    month = MinMax(min=1, max=12, allow_none=True)
    day = MinMax(min=1, max=31, allow_none=True)
    hour = MinMax(min=0, max=23, allow_none=True)
    minute = MinMax(min=0, max=59, allow_none=True)
    second = Integer(min=0, max=59, allow_none=True)
    dateTimeGrouping = Set(values=(['year', 'month', 'day', 'hour', 'minute',
                                    'second']))

    def __init__(self,
                 year=None,
                 month=None,
                 day=None,
                 hour=None,
                 minute=None,
                 second=None,
                 dateTimeGrouping=None,
                ):
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.dateTimeGrouping = dateTimeGrouping


class Filters(Serialisable):

    tagname = "filters"

    blank = Bool(allow_none=True)
    calendarType = NoneSet(values=["gregorian","gregorianUs",
                                   "gregorianMeFrench","gregorianArabic", "hijri","hebrew",
                                   "taiwan","japan", "thai","korea",
                                   "saka","gregorianXlitEnglish","gregorianXlitFrench"])
    filter = ValueSequence(expected_type=str)
    dateGroupItem = Sequence(expected_type=DateGroupItem, allow_none=True)

    __elements__ = ('filter', 'dateGroupItem')

    def __init__(self,
                 blank=None,
                 calendarType=None,
                 filter=(),
                 dateGroupItem=(),
                ):
        self.blank = blank
        self.calendarType = calendarType
        self.filter = filter
        self.dateGroupItem = dateGroupItem


class FilterColumn(Serialisable):

    tagname = "filterColumn"

    colId = Integer()
    col_id = Alias('colId')
    hiddenButton = Bool(allow_none=True)
    showButton = Bool(allow_none=True)
    # some elements are choice
    filters = Typed(expected_type=Filters, allow_none=True)
    top10 = Typed(expected_type=Top10, allow_none=True)
    customFilters = Typed(expected_type=CustomFilters, allow_none=True)
    dynamicFilter = Typed(expected_type=DynamicFilter, allow_none=True)
    colorFilter = Typed(expected_type=ColorFilter, allow_none=True)
    iconFilter = Typed(expected_type=IconFilter, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('filters', 'top10', 'customFilters', 'dynamicFilter',
                    'colorFilter', 'iconFilter')

    def __init__(self,
                 colId=None,
                 hiddenButton=False,
                 showButton=True,
                 filters=None,
                 top10=None,
                 customFilters=None,
                 dynamicFilter=None,
                 colorFilter=None,
                 iconFilter=None,
                 extLst=None,
                 blank=None,
                 vals=None,
                ):
        self.colId = colId
        self.hiddenButton = hiddenButton
        self.showButton = showButton
        self.filters = filters
        self.top10 = top10
        self.customFilters = customFilters
        self.dynamicFilter = dynamicFilter
        self.colorFilter = colorFilter
        self.iconFilter = iconFilter
        if blank is not None and self.filters:
            self.filters.blank = blank
        if vals is not None and self.filters:
            self.filters.filter = vals


class AutoFilter(Serialisable):

    tagname = "autoFilter"

    ref = CellRange()
    filterColumn = Sequence(expected_type=FilterColumn, allow_none=True)
    sortState = Typed(expected_type=SortState, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('filterColumn', 'sortState')

    def __init__(self,
                 ref=None,
                 filterColumn=(),
                 sortState=None,
                 extLst=None,
                ):
        self.ref = ref
        self.filterColumn = filterColumn
        self.sortState = sortState


    def __bool__(self):
        return self.ref is not None


    def __str__(self):
        return absolute_coordinate(self.ref)


    def add_filter_column(self, col_id, vals, blank=False):
        """
        Add row filter for specified column.

        :param col_id: Zero-origin column id. 0 means first column.
        :type  col_id: int
        :param vals: Value list to show.
        :type  vals: str[]
        :param blank: Show rows that have blank cell if True (default=``False``)
        :type  blank: bool
        """
        self.filterColumn.append(FilterColumn(colId=col_id, filters=Filters(blank=blank, filter=vals)))


    def add_sort_condition(self, ref, descending=False):
        """
        Add sort condition for cpecified range of cells.

        :param ref: range of the cells (e.g. 'A2:A150')
        :type  ref: string, is the same as that of the filter
        :param descending: Descending sort order (default=``False``)
        :type  descending: bool
        """
        cond = SortCondition(ref, descending)
        if self.sortState is None:
            self.sortState = SortState(ref=self.ref)
        self.sortState.sortCondition.append(cond)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\string_arrow.py ===
from __future__ import annotations

import operator
import re
from typing import (
    TYPE_CHECKING,
    Callable,
    Union,
)
import warnings

import numpy as np

from pandas._libs import (
    lib,
    missing as libmissing,
)
from pandas.compat import (
    pa_version_under10p1,
    pa_version_under13p0,
    pa_version_under16p0,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_scalar,
    pandas_dtype,
)
from pandas.core.dtypes.missing import isna

from pandas.core.arrays._arrow_string_mixins import ArrowStringArrayMixin
from pandas.core.arrays.arrow import ArrowExtensionArray
from pandas.core.arrays.boolean import BooleanDtype
from pandas.core.arrays.floating import Float64Dtype
from pandas.core.arrays.integer import Int64Dtype
from pandas.core.arrays.numeric import NumericDtype
from pandas.core.arrays.string_ import (
    BaseStringArray,
    StringDtype,
)
from pandas.core.strings.object_array import ObjectStringArrayMixin

if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc


if TYPE_CHECKING:
    from collections.abc import Sequence

    from pandas._typing import (
        ArrayLike,
        Dtype,
        Self,
        npt,
    )

    from pandas import Series


ArrowStringScalarOrNAT = Union[str, libmissing.NAType]


def _chk_pyarrow_available() -> None:
    if pa_version_under10p1:
        msg = "pyarrow>=10.0.1 is required for PyArrow backed ArrowExtensionArray."
        raise ImportError(msg)


def _is_string_view(typ):
    return not pa_version_under16p0 and pa.types.is_string_view(typ)


# TODO: Inherit directly from BaseStringArrayMethods. Currently we inherit from
# ObjectStringArrayMixin because we want to have the object-dtype based methods as
# fallback for the ones that pyarrow doesn't yet support


class ArrowStringArray(ObjectStringArrayMixin, ArrowExtensionArray, BaseStringArray):
    """
    Extension array for string data in a ``pyarrow.ChunkedArray``.

    .. warning::

       ArrowStringArray is considered experimental. The implementation and
       parts of the API may change without warning.

    Parameters
    ----------
    values : pyarrow.Array or pyarrow.ChunkedArray
        The array of data.

    Attributes
    ----------
    None

    Methods
    -------
    None

    See Also
    --------
    :func:`pandas.array`
        The recommended function for creating a ArrowStringArray.
    Series.str
        The string methods are available on Series backed by
        a ArrowStringArray.

    Notes
    -----
    ArrowStringArray returns a BooleanArray for comparison methods.

    Examples
    --------
    >>> pd.array(['This is', 'some text', None, 'data.'], dtype="string[pyarrow]")
    <ArrowStringArray>
    ['This is', 'some text', <NA>, 'data.']
    Length: 4, dtype: string
    """

    # error: Incompatible types in assignment (expression has type "StringDtype",
    # base class "ArrowExtensionArray" defined the type as "ArrowDtype")
    _dtype: StringDtype  # type: ignore[assignment]
    _storage = "pyarrow"
    _na_value: libmissing.NAType | float = libmissing.NA

    def __init__(self, values) -> None:
        _chk_pyarrow_available()
        if isinstance(values, (pa.Array, pa.ChunkedArray)) and (
            pa.types.is_string(values.type)
            or _is_string_view(values.type)
            or (
                pa.types.is_dictionary(values.type)
                and (
                    pa.types.is_string(values.type.value_type)
                    or pa.types.is_large_string(values.type.value_type)
                    or _is_string_view(values.type.value_type)
                )
            )
        ):
            values = pc.cast(values, pa.large_string())

        super().__init__(values)
        self._dtype = StringDtype(storage=self._storage, na_value=self._na_value)

        if not pa.types.is_large_string(self._pa_array.type):
            raise ValueError(
                "ArrowStringArray requires a PyArrow (chunked) array of "
                "large_string type"
            )

    @classmethod
    def _box_pa_scalar(cls, value, pa_type: pa.DataType | None = None) -> pa.Scalar:
        pa_scalar = super()._box_pa_scalar(value, pa_type)
        if pa.types.is_string(pa_scalar.type) and pa_type is None:
            pa_scalar = pc.cast(pa_scalar, pa.large_string())
        return pa_scalar

    @classmethod
    def _box_pa_array(
        cls, value, pa_type: pa.DataType | None = None, copy: bool = False
    ) -> pa.Array | pa.ChunkedArray:
        pa_array = super()._box_pa_array(value, pa_type)
        if pa.types.is_string(pa_array.type) and pa_type is None:
            pa_array = pc.cast(pa_array, pa.large_string())
        return pa_array

    def __len__(self) -> int:
        """
        Length of this array.

        Returns
        -------
        length : int
        """
        return len(self._pa_array)

    @classmethod
    def _from_sequence(cls, scalars, *, dtype: Dtype | None = None, copy: bool = False):
        from pandas.core.arrays.masked import BaseMaskedArray

        _chk_pyarrow_available()

        if dtype and not (isinstance(dtype, str) and dtype == "string"):
            dtype = pandas_dtype(dtype)
            assert isinstance(dtype, StringDtype) and dtype.storage == "pyarrow"

        if isinstance(scalars, BaseMaskedArray):
            # avoid costly conversion to object dtype in ensure_string_array and
            # numerical issues with Float32Dtype
            na_values = scalars._mask
            result = scalars._data
            result = lib.ensure_string_array(result, copy=copy, convert_na_value=False)
            return cls(pa.array(result, mask=na_values, type=pa.large_string()))
        elif isinstance(scalars, (pa.Array, pa.ChunkedArray)):
            return cls(pc.cast(scalars, pa.large_string()))

        # convert non-na-likes to str
        result = lib.ensure_string_array(scalars, copy=copy)
        return cls(pa.array(result, type=pa.large_string(), from_pandas=True))

    @classmethod
    def _from_sequence_of_strings(
        cls, strings, dtype: Dtype | None = None, copy: bool = False
    ):
        return cls._from_sequence(strings, dtype=dtype, copy=copy)

    @property
    def dtype(self) -> StringDtype:  # type: ignore[override]
        """
        An instance of 'string[pyarrow]'.
        """
        return self._dtype

    def insert(self, loc: int, item) -> ArrowStringArray:
        if self.dtype.na_value is np.nan and item is np.nan:
            item = libmissing.NA
        if not isinstance(item, str) and item is not libmissing.NA:
            raise TypeError(
                f"Invalid value '{item}' for dtype 'str'. Value should be a "
                f"string or missing value, got '{type(item).__name__}' instead."
            )
        return super().insert(loc, item)

    def _convert_bool_result(self, values, na=lib.no_default, method_name=None):
        if na is not lib.no_default and not isna(na) and not isinstance(na, bool):
            # GH#59561
            warnings.warn(
                f"Allowing a non-bool 'na' in obj.str.{method_name} is deprecated "
                "and will raise in a future version.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
            na = bool(na)

        if self.dtype.na_value is np.nan:
            if na is lib.no_default or isna(na):
                # NaN propagates as False
                values = values.fill_null(False)
            else:
                values = values.fill_null(na)
            return values.to_numpy()
        else:
            if na is not lib.no_default and not isna(
                na
            ):  # pyright: ignore [reportGeneralTypeIssues]
                values = values.fill_null(na)
        return BooleanDtype().__from_arrow__(values)

    def _maybe_convert_setitem_value(self, value):
        """Maybe convert value to be pyarrow compatible."""
        if is_scalar(value):
            if isna(value):
                value = None
            elif not isinstance(value, str):
                raise TypeError(
                    f"Invalid value '{value}' for dtype 'str'. Value should be a "
                    f"string or missing value, got '{type(value).__name__}' instead."
                )
        else:
            value = np.array(value, dtype=object, copy=True)
            value[isna(value)] = None
            for v in value:
                if not (v is None or isinstance(v, str)):
                    raise TypeError(
                        "Invalid value for dtype 'str'. Value should be a "
                        "string or missing value (or array of those)."
                    )
        return super()._maybe_convert_setitem_value(value)

    def isin(self, values: ArrayLike) -> npt.NDArray[np.bool_]:
        value_set = [
            pa_scalar.as_py()
            for pa_scalar in [pa.scalar(value, from_pandas=True) for value in values]
            if pa_scalar.type in (pa.string(), pa.null(), pa.large_string())
        ]

        # short-circuit to return all False array.
        if not len(value_set):
            return np.zeros(len(self), dtype=bool)

        result = pc.is_in(
            self._pa_array, value_set=pa.array(value_set, type=self._pa_array.type)
        )
        # pyarrow 2.0.0 returned nulls, so we explicily specify dtype to convert nulls
        # to False
        return np.array(result, dtype=np.bool_)

    def astype(self, dtype, copy: bool = True):
        dtype = pandas_dtype(dtype)

        if dtype == self.dtype:
            if copy:
                return self.copy()
            return self
        elif isinstance(dtype, NumericDtype):
            data = self._pa_array.cast(pa.from_numpy_dtype(dtype.numpy_dtype))
            return dtype.__from_arrow__(data)
        elif isinstance(dtype, np.dtype) and np.issubdtype(dtype, np.floating):
            return self.to_numpy(dtype=dtype, na_value=np.nan)

        return super().astype(dtype, copy=copy)

    @property
    def _data(self):
        # dask accesses ._data directlys
        warnings.warn(
            f"{type(self).__name__}._data is a deprecated and will be removed "
            "in a future version, use ._pa_array instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._pa_array

    # ------------------------------------------------------------------------
    # String methods interface

    _str_isalnum = ArrowStringArrayMixin._str_isalnum
    _str_isalpha = ArrowStringArrayMixin._str_isalpha
    _str_isdecimal = ArrowStringArrayMixin._str_isdecimal
    _str_isdigit = ArrowStringArrayMixin._str_isdigit
    _str_islower = ArrowStringArrayMixin._str_islower
    _str_isnumeric = ArrowStringArrayMixin._str_isnumeric
    _str_isspace = ArrowStringArrayMixin._str_isspace
    _str_istitle = ArrowStringArrayMixin._str_istitle
    _str_isupper = ArrowStringArrayMixin._str_isupper

    _str_map = BaseStringArray._str_map
    _str_startswith = ArrowStringArrayMixin._str_startswith
    _str_endswith = ArrowStringArrayMixin._str_endswith
    _str_pad = ArrowStringArrayMixin._str_pad
    _str_match = ArrowStringArrayMixin._str_match
    _str_fullmatch = ArrowStringArrayMixin._str_fullmatch
    _str_lower = ArrowStringArrayMixin._str_lower
    _str_upper = ArrowStringArrayMixin._str_upper
    _str_strip = ArrowStringArrayMixin._str_strip
    _str_lstrip = ArrowStringArrayMixin._str_lstrip
    _str_rstrip = ArrowStringArrayMixin._str_rstrip
    _str_removesuffix = ArrowStringArrayMixin._str_removesuffix
    _str_get = ArrowStringArrayMixin._str_get
    _str_capitalize = ArrowStringArrayMixin._str_capitalize
    _str_title = ArrowStringArrayMixin._str_title
    _str_swapcase = ArrowStringArrayMixin._str_swapcase
    _str_slice_replace = ArrowStringArrayMixin._str_slice_replace
    _str_len = ArrowStringArrayMixin._str_len
    _str_slice = ArrowStringArrayMixin._str_slice

    def _str_contains(
        self,
        pat,
        case: bool = True,
        flags: int = 0,
        na=lib.no_default,
        regex: bool = True,
    ):
        if flags:
            return super()._str_contains(pat, case, flags, na, regex)

        return ArrowStringArrayMixin._str_contains(self, pat, case, flags, na, regex)

    def _str_replace(
        self,
        pat: str | re.Pattern,
        repl: str | Callable,
        n: int = -1,
        case: bool = True,
        flags: int = 0,
        regex: bool = True,
    ):
        if isinstance(pat, re.Pattern) or callable(repl) or not case or flags:
            return super()._str_replace(pat, repl, n, case, flags, regex)

        return ArrowStringArrayMixin._str_replace(
            self, pat, repl, n, case, flags, regex
        )

    def _str_repeat(self, repeats: int | Sequence[int]):
        if not isinstance(repeats, int):
            return super()._str_repeat(repeats)
        else:
            return ArrowExtensionArray._str_repeat(self, repeats=repeats)

    def _str_removeprefix(self, prefix: str):
        if not pa_version_under13p0:
            return ArrowStringArrayMixin._str_removeprefix(self, prefix)
        return super()._str_removeprefix(prefix)

    def _str_count(self, pat: str, flags: int = 0):
        if flags:
            return super()._str_count(pat, flags)
        result = pc.count_substring_regex(self._pa_array, pat)
        return self._convert_int_result(result)

    def _str_find(self, sub: str, start: int = 0, end: int | None = None):
        if (
            pa_version_under13p0
            and not (start != 0 and end is not None)
            and not (start == 0 and end is None)
        ):
            # GH#59562
            return super()._str_find(sub, start, end)
        return ArrowStringArrayMixin._str_find(self, sub, start, end)

    def _str_get_dummies(self, sep: str = "|"):
        dummies_pa, labels = ArrowExtensionArray(self._pa_array)._str_get_dummies(sep)
        if len(labels) == 0:
            return np.empty(shape=(0, 0), dtype=np.int64), labels
        dummies = np.vstack(dummies_pa.to_numpy())
        return dummies.astype(np.int64, copy=False), labels

    def _convert_int_result(self, result):
        if self.dtype.na_value is np.nan:
            if isinstance(result, pa.Array):
                result = result.to_numpy(zero_copy_only=False)
            else:
                result = result.to_numpy()
            if result.dtype == np.int32:
                result = result.astype(np.int64)
            return result

        return Int64Dtype().__from_arrow__(result)

    def _convert_rank_result(self, result):
        if self.dtype.na_value is np.nan:
            if isinstance(result, pa.Array):
                result = result.to_numpy(zero_copy_only=False)
            else:
                result = result.to_numpy()
            return result.astype("float64", copy=False)

        return Float64Dtype().__from_arrow__(result)

    def _reduce(
        self, name: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
    ):
        if self.dtype.na_value is np.nan and name in ["any", "all"]:
            if not skipna:
                nas = pc.is_null(self._pa_array)
                arr = pc.or_kleene(nas, pc.not_equal(self._pa_array, ""))
            else:
                arr = pc.not_equal(self._pa_array, "")
            result = ArrowExtensionArray(arr)._reduce(
                name, skipna=skipna, keepdims=keepdims, **kwargs
            )
            if keepdims:
                # ArrowExtensionArray will return a length-1 bool[pyarrow] array
                return result.astype(np.bool_)
            return result

        if name in ("min", "max", "sum", "argmin", "argmax"):
            result = self._reduce_calc(name, skipna=skipna, keepdims=keepdims, **kwargs)
        else:
            raise TypeError(f"Cannot perform reduction '{name}' with string dtype")

        if name in ("argmin", "argmax") and isinstance(result, pa.Array):
            return self._convert_int_result(result)
        elif isinstance(result, pa.Array):
            return type(self)(result)
        else:
            return result

    def value_counts(self, dropna: bool = True) -> Series:
        result = super().value_counts(dropna=dropna)
        if self.dtype.na_value is np.nan:
            res_values = result._values.to_numpy()
            return result._constructor(
                res_values, index=result.index, name=result.name, copy=False
            )
        return result

    def _cmp_method(self, other, op):
        result = super()._cmp_method(other, op)
        if self.dtype.na_value is np.nan:
            if op == operator.ne:
                return result.to_numpy(np.bool_, na_value=True)
            else:
                return result.to_numpy(np.bool_, na_value=False)
        return result

    def __pos__(self) -> Self:
        raise TypeError(f"bad operand type for unary +: '{self.dtype}'")


class ArrowStringArrayNumpySemantics(ArrowStringArray):
    _na_value = np.nan

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\assume.py ===
"""A module which implements predicates and assumption context."""

from contextlib import contextmanager
import inspect
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import Boolean, false, true
from sympy.multipledispatch.dispatcher import Dispatcher, str_signature
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.source import get_class


class AssumptionsContext(set):
    """
    Set containing default assumptions which are applied to the ``ask()``
    function.

    Explanation
    ===========

    This is used to represent global assumptions, but you can also use this
    class to create your own local assumptions contexts. It is basically a thin
    wrapper to Python's set, so see its documentation for advanced usage.

    Examples
    ========

    The default assumption context is ``global_assumptions``, which is initially empty:

    >>> from sympy import ask, Q
    >>> from sympy.assumptions import global_assumptions
    >>> global_assumptions
    AssumptionsContext()

    You can add default assumptions:

    >>> from sympy.abc import x
    >>> global_assumptions.add(Q.real(x))
    >>> global_assumptions
    AssumptionsContext({Q.real(x)})
    >>> ask(Q.real(x))
    True

    And remove them:

    >>> global_assumptions.remove(Q.real(x))
    >>> print(ask(Q.real(x)))
    None

    The ``clear()`` method removes every assumption:

    >>> global_assumptions.add(Q.positive(x))
    >>> global_assumptions
    AssumptionsContext({Q.positive(x)})
    >>> global_assumptions.clear()
    >>> global_assumptions
    AssumptionsContext()

    See Also
    ========

    assuming

    """

    def add(self, *assumptions):
        """Add assumptions."""
        for a in assumptions:
            super().add(a)

    def _sympystr(self, printer):
        if not self:
            return "%s()" % self.__class__.__name__
        return "{}({})".format(self.__class__.__name__, printer._print_set(self))

global_assumptions = AssumptionsContext()


class AppliedPredicate(Boolean):
    """
    The class of expressions resulting from applying ``Predicate`` to
    the arguments. ``AppliedPredicate`` merely wraps its argument and
    remain unevaluated. To evaluate it, use the ``ask()`` function.

    Examples
    ========

    >>> from sympy import Q, ask
    >>> Q.integer(1)
    Q.integer(1)

    The ``function`` attribute returns the predicate, and the ``arguments``
    attribute returns the tuple of arguments.

    >>> type(Q.integer(1))
    <class 'sympy.assumptions.assume.AppliedPredicate'>
    >>> Q.integer(1).function
    Q.integer
    >>> Q.integer(1).arguments
    (1,)

    Applied predicates can be evaluated to a boolean value with ``ask``:

    >>> ask(Q.integer(1))
    True

    """
    __slots__ = ()

    def __new__(cls, predicate, *args):
        if not isinstance(predicate, Predicate):
            raise TypeError("%s is not a Predicate." % predicate)
        args = map(_sympify, args)
        return super().__new__(cls, predicate, *args)

    @property
    def arg(self):
        """
        Return the expression used by this assumption.

        Examples
        ========

        >>> from sympy import Q, Symbol
        >>> x = Symbol('x')
        >>> a = Q.integer(x + 1)
        >>> a.arg
        x + 1

        """
        # Will be deprecated
        args = self._args
        if len(args) == 2:
            # backwards compatibility
            return args[1]
        raise TypeError("'arg' property is allowed only for unary predicates.")

    @property
    def function(self):
        """
        Return the predicate.
        """
        # Will be changed to self.args[0] after args overriding is removed
        return self._args[0]

    @property
    def arguments(self):
        """
        Return the arguments which are applied to the predicate.
        """
        # Will be changed to self.args[1:] after args overriding is removed
        return self._args[1:]

    def _eval_ask(self, assumptions):
        return self.function.eval(self.arguments, assumptions)

    @property
    def binary_symbols(self):
        from .ask import Q
        if self.function == Q.is_true:
            i = self.arguments[0]
            if i.is_Boolean or i.is_Symbol:
                return i.binary_symbols
        if self.function in (Q.eq, Q.ne):
            if true in self.arguments or false in self.arguments:
                if self.arguments[0].is_Symbol:
                    return {self.arguments[0]}
                elif self.arguments[1].is_Symbol:
                    return {self.arguments[1]}
        return set()


class PredicateMeta(type):
    def __new__(cls, clsname, bases, dct):
        # If handler is not defined, assign empty dispatcher.
        if "handler" not in dct:
            name = f"Ask{clsname.capitalize()}Handler"
            handler = Dispatcher(name, doc="Handler for key %s" % name)
            dct["handler"] = handler

        dct["_orig_doc"] = dct.get("__doc__", "")

        return super().__new__(cls, clsname, bases, dct)

    @property
    def __doc__(cls):
        handler = cls.handler
        doc = cls._orig_doc
        if cls is not Predicate and handler is not None:
            doc += "Handler\n"
            doc += "    =======\n\n"

            # Append the handler's doc without breaking sphinx documentation.
            docs = ["    Multiply dispatched method: %s" % handler.name]
            if handler.doc:
                for line in handler.doc.splitlines():
                    if not line:
                        continue
                    docs.append("    %s" % line)
            other = []
            for sig in handler.ordering[::-1]:
                func = handler.funcs[sig]
                if func.__doc__:
                    s = '    Inputs: <%s>' % str_signature(sig)
                    lines = []
                    for line in func.__doc__.splitlines():
                        lines.append("    %s" % line)
                    s += "\n".join(lines)
                    docs.append(s)
                else:
                    other.append(str_signature(sig))
            if other:
                othersig = "    Other signatures:"
                for line in other:
                    othersig += "\n        * %s" % line
                docs.append(othersig)

            doc += '\n\n'.join(docs)

        return doc


class Predicate(Boolean, metaclass=PredicateMeta):
    """
    Base class for mathematical predicates. It also serves as a
    constructor for undefined predicate objects.

    Explanation
    ===========

    Predicate is a function that returns a boolean value [1].

    Predicate function is object, and it is instance of predicate class.
    When a predicate is applied to arguments, ``AppliedPredicate``
    instance is returned. This merely wraps the argument and remain
    unevaluated. To obtain the truth value of applied predicate, use the
    function ``ask``.

    Evaluation of predicate is done by multiple dispatching. You can
    register new handler to the predicate to support new types.

    Every predicate in SymPy can be accessed via the property of ``Q``.
    For example, ``Q.even`` returns the predicate which checks if the
    argument is even number.

    To define a predicate which can be evaluated, you must subclass this
    class, make an instance of it, and register it to ``Q``. After then,
    dispatch the handler by argument types.

    If you directly construct predicate using this class, you will get
    ``UndefinedPredicate`` which cannot be dispatched. This is useful
    when you are building boolean expressions which do not need to be
    evaluated.

    Examples
    ========

    Applying and evaluating to boolean value:

    >>> from sympy import Q, ask
    >>> ask(Q.prime(7))
    True

    You can define a new predicate by subclassing and dispatching. Here,
    we define a predicate for sexy primes [2] as an example.

    >>> from sympy import Predicate, Integer
    >>> class SexyPrimePredicate(Predicate):
    ...     name = "sexyprime"
    >>> Q.sexyprime = SexyPrimePredicate()
    >>> @Q.sexyprime.register(Integer, Integer)
    ... def _(int1, int2, assumptions):
    ...     args = sorted([int1, int2])
    ...     if not all(ask(Q.prime(a), assumptions) for a in args):
    ...         return False
    ...     return args[1] - args[0] == 6
    >>> ask(Q.sexyprime(5, 11))
    True

    Direct constructing returns ``UndefinedPredicate``, which can be
    applied but cannot be dispatched.

    >>> from sympy import Predicate, Integer
    >>> Q.P = Predicate("P")
    >>> type(Q.P)
    <class 'sympy.assumptions.assume.UndefinedPredicate'>
    >>> Q.P(1)
    Q.P(1)
    >>> Q.P.register(Integer)(lambda expr, assump: True)
    Traceback (most recent call last):
      ...
    TypeError: <class 'sympy.assumptions.assume.UndefinedPredicate'> cannot be dispatched.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Predicate_%28mathematical_logic%29
    .. [2] https://en.wikipedia.org/wiki/Sexy_prime

    """

    is_Atom = True

    def __new__(cls, *args, **kwargs):
        if cls is Predicate:
            return UndefinedPredicate(*args, **kwargs)
        obj = super().__new__(cls, *args)
        return obj

    @property
    def name(self):
        # May be overridden
        return type(self).__name__

    @classmethod
    def register(cls, *types, **kwargs):
        """
        Register the signature to the handler.
        """
        if cls.handler is None:
            raise TypeError("%s cannot be dispatched." % type(cls))
        return cls.handler.register(*types, **kwargs)

    @classmethod
    def register_many(cls, *types, **kwargs):
        """
        Register multiple signatures to same handler.
        """
        def _(func):
            for t in types:
                if not is_sequence(t):
                    t = (t,)  # for convenience, allow passing `type` to mean `(type,)`
                cls.register(*t, **kwargs)(func)
        return _

    def __call__(self, *args):
        return AppliedPredicate(self, *args)

    def eval(self, args, assumptions=True):
        """
        Evaluate ``self(*args)`` under the given assumptions.

        This uses only direct resolution methods, not logical inference.
        """
        result = None
        try:
            result = self.handler(*args, assumptions=assumptions)
        except NotImplementedError:
            pass
        return result

    def _eval_refine(self, assumptions):
        # When Predicate is no longer Boolean, delete this method
        return self


class UndefinedPredicate(Predicate):
    """
    Predicate without handler.

    Explanation
    ===========

    This predicate is generated by using ``Predicate`` directly for
    construction. It does not have a handler, and evaluating this with
    arguments is done by SAT solver.

    Examples
    ========

    >>> from sympy import Predicate, Q
    >>> Q.P = Predicate('P')
    >>> Q.P.func
    <class 'sympy.assumptions.assume.UndefinedPredicate'>
    >>> Q.P.name
    Str('P')

    """

    handler = None

    def __new__(cls, name, handlers=None):
        # "handlers" parameter supports old design
        if not isinstance(name, Str):
            name = Str(name)
        obj = super(Boolean, cls).__new__(cls, name)
        obj.handlers = handlers or []
        return obj

    @property
    def name(self):
        return self.args[0]

    def _hashable_content(self):
        return (self.name,)

    def __getnewargs__(self):
        return (self.name,)

    def __call__(self, expr):
        return AppliedPredicate(self, expr)

    def add_handler(self, handler):
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. Predicate.add_handler()
            should be replaced with the multipledispatch handler of Predicate.
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
        )
        self.handlers.append(handler)

    def remove_handler(self, handler):
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. Predicate.remove_handler()
            should be replaced with the multipledispatch handler of Predicate.
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
        )
        self.handlers.remove(handler)

    def eval(self, args, assumptions=True):
        # Support for deprecated design
        # When old design is removed, this will always return None
        sympy_deprecation_warning(
            """
            The AskHandler system is deprecated. Evaluating UndefinedPredicate
            objects should be replaced with the multipledispatch handler of
            Predicate.
            """,
            deprecated_since_version="1.8",
            active_deprecations_target='deprecated-askhandler',
            stacklevel=5,
        )
        expr, = args
        res, _res = None, None
        mro = inspect.getmro(type(expr))
        for handler in self.handlers:
            cls = get_class(handler)
            for subclass in mro:
                eval_ = getattr(cls, subclass.__name__, None)
                if eval_ is None:
                    continue
                res = eval_(expr, assumptions)
                # Do not stop if value returned is None
                # Try to check for higher classes
                if res is None:
                    continue
                if _res is None:
                    _res = res
                else:
                    # only check consistency if both resolutors have concluded
                    if _res != res:
                        raise ValueError('incompatible resolutors')
                break
        return res


@contextmanager
def assuming(*assumptions):
    """
    Context manager for assumptions.

    Examples
    ========

    >>> from sympy import assuming, Q, ask
    >>> from sympy.abc import x, y
    >>> print(ask(Q.integer(x + y)))
    None
    >>> with assuming(Q.integer(x), Q.integer(y)):
    ...     print(ask(Q.integer(x + y)))
    True
    """
    old_global_assumptions = global_assumptions.copy()
    global_assumptions.update(assumptions)
    try:
        yield
    finally:
        global_assumptions.clear()
        global_assumptions.update(old_global_assumptions)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\solvers\ode\nonhomogeneous.py ===
r"""
This File contains helper functions for nth_linear_constant_coeff_undetermined_coefficients,
nth_linear_euler_eq_nonhomogeneous_undetermined_coefficients,
nth_linear_constant_coeff_variation_of_parameters,
and nth_linear_euler_eq_nonhomogeneous_variation_of_parameters.

All the functions in this file are used by more than one solvers so, instead of creating
instances in other classes for using them it is better to keep it here as separate helpers.

"""
from collections import Counter
from sympy.core import Add, S
from sympy.core.function import diff, expand, _mexpand, expand_mul
from sympy.core.relational import Eq
from sympy.core.sorting import default_sort_key
from sympy.core.symbol import Dummy, Wild
from sympy.functions import exp, cos, cosh, im, log, re, sin, sinh, \
    atan2, conjugate
from sympy.integrals import Integral
from sympy.polys import (Poly, RootOf, rootof, roots)
from sympy.simplify import collect, simplify, separatevars, powsimp, trigsimp # type: ignore
from sympy.utilities import numbered_symbols
from sympy.solvers.solvers import solve
from sympy.matrices import wronskian
from .subscheck import sub_func_doit
from sympy.solvers.ode.ode import get_numbered_constants


def _test_term(coeff, func, order):
    r"""
    Linear Euler ODEs have the form  K*x**order*diff(y(x), x, order) = F(x),
    where K is independent of x and y(x), order>= 0.
    So we need to check that for each term, coeff == K*x**order from
    some K.  We have a few cases, since coeff may have several
    different types.
    """
    x = func.args[0]
    f = func.func
    if order < 0:
        raise ValueError("order should be greater than 0")
    if coeff == 0:
        return True
    if order == 0:
        if x in coeff.free_symbols:
            return False
        return True
    if coeff.is_Mul:
        if coeff.has(f(x)):
            return False
        return x**order in coeff.args
    elif coeff.is_Pow:
        return coeff.as_base_exp() == (x, order)
    elif order == 1:
        return x == coeff
    return False


def _get_euler_characteristic_eq_sols(eq, func, match_obj):
    r"""
    Returns the solution of homogeneous part of the linear euler ODE and
    the list of roots of characteristic equation.

    The parameter ``match_obj`` is a dict of order:coeff terms, where order is the order
    of the derivative on each term, and coeff is the coefficient of that derivative.

    """
    x = func.args[0]
    f = func.func

    # First, set up characteristic equation.
    chareq, symbol = S.Zero, Dummy('x')

    for i in match_obj:
        if i >= 0:
            chareq += (match_obj[i]*diff(x**symbol, x, i)*x**-symbol).expand()

    chareq = Poly(chareq, symbol)
    chareqroots = [rootof(chareq, k) for k in range(chareq.degree())]
    collectterms = []

    # A generator of constants
    constants = list(get_numbered_constants(eq, num=chareq.degree()*2))
    constants.reverse()

    # Create a dict root: multiplicity or charroots
    charroots = Counter(chareqroots)
    gsol = S.Zero
    ln = log
    for root, multiplicity in charroots.items():
        for i in range(multiplicity):
            if isinstance(root, RootOf):
                gsol += (x**root) * constants.pop()
                if multiplicity != 1:
                    raise ValueError("Value should be 1")
                collectterms = [(0, root, 0)] + collectterms
            elif root.is_real:
                gsol += ln(x)**i*(x**root) * constants.pop()
                collectterms = [(i, root, 0)] + collectterms
            else:
                reroot = re(root)
                imroot = im(root)
                gsol += ln(x)**i * (x**reroot) * (
                    constants.pop() * sin(abs(imroot)*ln(x))
                    + constants.pop() * cos(imroot*ln(x)))
                collectterms = [(i, reroot, imroot)] + collectterms

    gsol = Eq(f(x), gsol)

    gensols = []
    # Keep track of when to use sin or cos for nonzero imroot
    for i, reroot, imroot in collectterms:
        if imroot == 0:
            gensols.append(ln(x)**i*x**reroot)
        else:
            sin_form = ln(x)**i*x**reroot*sin(abs(imroot)*ln(x))
            if sin_form in gensols:
                cos_form = ln(x)**i*x**reroot*cos(imroot*ln(x))
                gensols.append(cos_form)
            else:
                gensols.append(sin_form)
    return gsol, gensols


def _solve_variation_of_parameters(eq, func, roots, homogen_sol, order, match_obj, simplify_flag=True):
    r"""
    Helper function for the method of variation of parameters and nonhomogeneous euler eq.

    See the
    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffVariationOfParameters`
    docstring for more information on this method.

    The parameter are ``match_obj`` should be a dictionary that has the following
    keys:

    ``list``
    A list of solutions to the homogeneous equation.

    ``sol``
    The general solution.

    """
    f = func.func
    x = func.args[0]
    r = match_obj
    psol = 0
    wr = wronskian(roots, x)

    if simplify_flag:
        wr = simplify(wr)  # We need much better simplification for
                        # some ODEs. See issue 4662, for example.
        # To reduce commonly occurring sin(x)**2 + cos(x)**2 to 1
        wr = trigsimp(wr, deep=True, recursive=True)
    if not wr:
        # The wronskian will be 0 iff the solutions are not linearly
        # independent.
        raise NotImplementedError("Cannot find " + str(order) +
        " solutions to the homogeneous equation necessary to apply " +
        "variation of parameters to " + str(eq) + " (Wronskian == 0)")
    if len(roots) != order:
        raise NotImplementedError("Cannot find " + str(order) +
        " solutions to the homogeneous equation necessary to apply " +
        "variation of parameters to " +
        str(eq) + " (number of terms != order)")
    negoneterm = S.NegativeOne**(order)
    for i in roots:
        psol += negoneterm*Integral(wronskian([sol for sol in roots if sol != i], x)*r[-1]/wr, x)*i/r[order]
        negoneterm *= -1

    if simplify_flag:
        psol = simplify(psol)
        psol = trigsimp(psol, deep=True)
    return Eq(f(x), homogen_sol.rhs + psol)


def _get_const_characteristic_eq_sols(r, func, order):
    r"""
    Returns the roots of characteristic equation of constant coefficient
    linear ODE and list of collectterms which is later on used by simplification
    to use collect on solution.

    The parameter `r` is a dict of order:coeff terms, where order is the order of the
    derivative on each term, and coeff is the coefficient of that derivative.

    """
    x = func.args[0]
    # First, set up characteristic equation.
    chareq, symbol = S.Zero, Dummy('x')

    for i in r.keys():
        if isinstance(i, str) or i < 0:
            pass
        else:
            chareq += r[i]*symbol**i

    chareq = Poly(chareq, symbol)
    # Can't just call roots because it doesn't return rootof for unsolveable
    # polynomials.
    chareqroots = roots(chareq, multiple=True)
    if len(chareqroots) != order:
        chareqroots = [rootof(chareq, k) for k in range(chareq.degree())]

    chareq_is_complex = not all(i.is_real for i in chareq.all_coeffs())

    # Create a dict root: multiplicity or charroots
    charroots = Counter(chareqroots)
    # We need to keep track of terms so we can run collect() at the end.
    # This is necessary for constantsimp to work properly.
    collectterms = []
    gensols = []
    conjugate_roots = [] # used to prevent double-use of conjugate roots
    # Loop over roots in theorder provided by roots/rootof...
    for root in chareqroots:
        # but don't repoeat multiple roots.
        if root not in charroots:
            continue
        multiplicity = charroots.pop(root)
        for i in range(multiplicity):
            if chareq_is_complex:
                gensols.append(x**i*exp(root*x))
                collectterms = [(i, root, 0)] + collectterms
                continue
            reroot = re(root)
            imroot = im(root)
            if imroot.has(atan2) and reroot.has(atan2):
                # Remove this condition when re and im stop returning
                # circular atan2 usages.
                gensols.append(x**i*exp(root*x))
                collectterms = [(i, root, 0)] + collectterms
            else:
                if root in conjugate_roots:
                    collectterms = [(i, reroot, imroot)] + collectterms
                    continue
                if imroot == 0:
                    gensols.append(x**i*exp(reroot*x))
                    collectterms = [(i, reroot, 0)] + collectterms
                    continue
                conjugate_roots.append(conjugate(root))
                gensols.append(x**i*exp(reroot*x) * sin(abs(imroot) * x))
                gensols.append(x**i*exp(reroot*x) * cos(    imroot  * x))

                # This ordering is important
                collectterms = [(i, reroot, imroot)] + collectterms
    return gensols, collectterms


# Ideally these kind of simplification functions shouldn't be part of solvers.
# odesimp should be improved to handle these kind of specific simplifications.
def _get_simplified_sol(sol, func, collectterms):
    r"""
    Helper function which collects the solution on
    collectterms. Ideally this should be handled by odesimp.It is used
    only when the simplify is set to True in dsolve.

    The parameter ``collectterms`` is a list of tuple (i, reroot, imroot) where `i` is
    the multiplicity of the root, reroot is real part and imroot being the imaginary part.

    """
    f = func.func
    x = func.args[0]
    collectterms.sort(key=default_sort_key)
    collectterms.reverse()
    assert len(sol) == 1 and sol[0].lhs == f(x)
    sol = sol[0].rhs
    sol = expand_mul(sol)
    for i, reroot, imroot in collectterms:
        sol = collect(sol, x**i*exp(reroot*x)*sin(abs(imroot)*x))
        sol = collect(sol, x**i*exp(reroot*x)*cos(imroot*x))
    for i, reroot, imroot in collectterms:
        sol = collect(sol, x**i*exp(reroot*x))
    sol = powsimp(sol)
    return Eq(f(x), sol)


def _undetermined_coefficients_match(expr, x, func=None, eq_homogeneous=S.Zero):
    r"""
    Returns a trial function match if undetermined coefficients can be applied
    to ``expr``, and ``None`` otherwise.

    A trial expression can be found for an expression for use with the method
    of undetermined coefficients if the expression is an
    additive/multiplicative combination of constants, polynomials in `x` (the
    independent variable of expr), `\sin(a x + b)`, `\cos(a x + b)`, and
    `e^{a x}` terms (in other words, it has a finite number of linearly
    independent derivatives).

    Note that you may still need to multiply each term returned here by
    sufficient `x` to make it linearly independent with the solutions to the
    homogeneous equation.

    This is intended for internal use by ``undetermined_coefficients`` hints.

    SymPy currently has no way to convert `\sin^n(x) \cos^m(y)` into a sum of
    only `\sin(a x)` and `\cos(b x)` terms, so these are not implemented.  So,
    for example, you will need to manually convert `\sin^2(x)` into `[1 +
    \cos(2 x)]/2` to properly apply the method of undetermined coefficients on
    it.

    Examples
    ========

    >>> from sympy import log, exp
    >>> from sympy.solvers.ode.nonhomogeneous import _undetermined_coefficients_match
    >>> from sympy.abc import x
    >>> _undetermined_coefficients_match(9*x*exp(x) + exp(-x), x)
    {'test': True, 'trialset': {x*exp(x), exp(-x), exp(x)}}
    >>> _undetermined_coefficients_match(log(x), x)
    {'test': False}

    """
    a = Wild('a', exclude=[x])
    b = Wild('b', exclude=[x])
    expr = powsimp(expr, combine='exp')  # exp(x)*exp(2*x + 1) => exp(3*x + 1)
    retdict = {}

    def _test_term(expr, x) -> bool:
        r"""
        Test if ``expr`` fits the proper form for undetermined coefficients.
        """
        if not expr.has(x):
            return True
        if expr.is_Add:
            return all(_test_term(i, x) for i in expr.args)
        if expr.is_Mul:
            if expr.has(sin, cos):
                foundtrig = False
                # Make sure that there is only one trig function in the args.
                # See the docstring.
                for i in expr.args:
                    if i.has(sin, cos):
                        if foundtrig:
                            return False
                        else:
                            foundtrig = True
            return all(_test_term(i, x) for i in expr.args)
        if expr.is_Function:
            return expr.func in (sin, cos, exp, sinh, cosh) and \
                   bool(expr.args[0].match(a*x + b))
        if expr.is_Pow and expr.base.is_Symbol and expr.exp.is_Integer and \
                expr.exp >= 0:
            return True
        if expr.is_Pow and expr.base.is_number:
            return bool(expr.exp.match(a*x + b))
        return expr.is_Symbol or bool(expr.is_number)

    def _get_trial_set(expr, x, exprs=set()):
        r"""
        Returns a set of trial terms for undetermined coefficients.

        The idea behind undetermined coefficients is that the terms expression
        repeat themselves after a finite number of derivatives, except for the
        coefficients (they are linearly dependent).  So if we collect these,
        we should have the terms of our trial function.
        """
        def _remove_coefficient(expr, x):
            r"""
            Returns the expression without a coefficient.

            Similar to expr.as_independent(x)[1], except it only works
            multiplicatively.
            """
            term = S.One
            if expr.is_Mul:
                for i in expr.args:
                    if i.has(x):
                        term *= i
            elif expr.has(x):
                term = expr
            return term

        expr = expand_mul(expr)
        if expr.is_Add:
            for term in expr.args:
                if _remove_coefficient(term, x) in exprs:
                    pass
                else:
                    exprs.add(_remove_coefficient(term, x))
                    exprs = exprs.union(_get_trial_set(term, x, exprs))
        else:
            term = _remove_coefficient(expr, x)
            tmpset = exprs.union({term})
            oldset = set()
            while tmpset != oldset:
                # If you get stuck in this loop, then _test_term is probably
                # broken
                oldset = tmpset.copy()
                expr = expr.diff(x)
                term = _remove_coefficient(expr, x)
                if term.is_Add:
                    tmpset = tmpset.union(_get_trial_set(term, x, tmpset))
                else:
                    tmpset.add(term)
            exprs = tmpset
        return exprs

    def is_homogeneous_solution(term):
        r""" This function checks whether the given trialset contains any root
            of homogeneous equation"""
        return expand(sub_func_doit(eq_homogeneous, func, term)).is_zero

    retdict['test'] = _test_term(expr, x)
    if retdict['test']:
        # Try to generate a list of trial solutions that will have the
        # undetermined coefficients. Note that if any of these are not linearly
        # independent with any of the solutions to the homogeneous equation,
        # then they will need to be multiplied by sufficient x to make them so.
        # This function DOES NOT do that (it doesn't even look at the
        # homogeneous equation).
        temp_set = set()
        for i in Add.make_args(expr):
            act = _get_trial_set(i, x)
            if eq_homogeneous is not S.Zero:
                while any(is_homogeneous_solution(ts) for ts in act):
                    act = {x*ts for ts in act}
            temp_set = temp_set.union(act)

        retdict['trialset'] = temp_set
    return retdict


def _solve_undetermined_coefficients(eq, func, order, match, trialset):
    r"""
    Helper function for the method of undetermined coefficients.

    See the
    :py:meth:`~sympy.solvers.ode.single.NthLinearConstantCoeffUndeterminedCoefficients`
    docstring for more information on this method.

    The parameter ``trialset`` is the set of trial functions as returned by
    ``_undetermined_coefficients_match()['trialset']``.

    The parameter ``match`` should be a dictionary that has the following
    keys:

    ``list``
    A list of solutions to the homogeneous equation.

    ``sol``
    The general solution.

    """
    r = match
    coeffs = numbered_symbols('a', cls=Dummy)
    coefflist = []
    gensols = r['list']
    gsol = r['sol']
    f = func.func
    x = func.args[0]

    if len(gensols) != order:
        raise NotImplementedError("Cannot find " + str(order) +
        " solutions to the homogeneous equation necessary to apply" +
        " undetermined coefficients to " + str(eq) +
        " (number of terms != order)")

    trialfunc = 0
    for i in trialset:
        c = next(coeffs)
        coefflist.append(c)
        trialfunc += c*i

    eqs = sub_func_doit(eq, f(x), trialfunc)

    coeffsdict = dict(list(zip(trialset, [0]*(len(trialset) + 1))))

    eqs = _mexpand(eqs)

    for i in Add.make_args(eqs):
        s = separatevars(i, dict=True, symbols=[x])
        if coeffsdict.get(s[x]):
            coeffsdict[s[x]] += s['coeff']
        else:
            coeffsdict[s[x]] = s['coeff']

    coeffvals = solve(list(coeffsdict.values()), coefflist)

    if not coeffvals:
        raise NotImplementedError(
            "Could not solve `%s` using the "
            "method of undetermined coefficients "
            "(unable to solve for coefficients)." % eq)

    psol = trialfunc.subs(coeffvals)

    return Eq(f(x), gsol.rhs + psol)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\whisper\whisper_helper.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
from pathlib import Path

import numpy as np
import torch
from convert_generation import add_cache_indirection_to_mha, add_output_qk_to_mha, fix_past_sequence_length
from optimizer import optimize_model
from transformers import WhisperConfig, WhisperForConditionalGeneration, WhisperProcessor
from whisper_decoder import WhisperDecoder
from whisper_encoder import WhisperEncoder
from whisper_encoder_decoder_init import WhisperEncoderDecoderInit
from whisper_jump_times import WhisperJumpTimes

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)

PRETRAINED_WHISPER_MODELS = [
    "whisper-tiny",
    "whisper-tiny.en",
    "whisper-base",
    "whisper-base.en",
    "whisper-small",
    "whisper-small.en",
    "whisper-medium",
    "whisper-medium.en",
    "whisper-large",
    "whisper-large-v2",
    "whisper-large-v3",
    "whisper-large-v3-turbo",
]


class WhisperHelper:
    @staticmethod
    def get_onnx_path(
        output_dir: str,
        model_name_or_path: str,
        suffix: str = "",
        new_folder: bool = False,
    ) -> str:
        """Build onnx path

        Args:
            output_dir (str): output directory
            model_name_or_path (str): pretrained model name, or path to the model checkpoint
            suffix (str, optional): suffix like "_encoder" or "_decoder_fp16" will be appended to file name. Defaults to None.
            new_folder (bool, optional): create a new directory for the model. Defaults to False.
        Returns:
            str: path of onnx model
        """
        model_name = model_name_or_path
        if os.path.isdir(model_name_or_path):
            model_name = Path(model_name_or_path).parts[-1]
        else:
            model_name = model_name.split("/")[-1]

        model_name += suffix

        directory = os.path.join(output_dir, model_name) if new_folder else output_dir
        return os.path.join(directory, model_name + ".onnx")

    @staticmethod
    def load_model(
        model_name_or_path: str,
        model_impl: str,
        cache_dir: str,
        device: torch.device,
        dtype: torch.dtype,
        merge_encoder_and_decoder_init: bool = True,
        no_beam_search_op: bool = False,
        output_qk: bool = False,
    ) -> dict[str, torch.nn.Module]:
        """Load model given a pretrained name or path, then build models for ONNX conversion.

        Args:
            model_name_or_path (str): pretrained model name or path
            model_impl (str): library to load model from
            cache_dir (str): cache directory
            device (torch.device): device to run the model
            dtype (torch.dtype): dtype to run the model
            merge_encoder_and_decoder_init (bool, optional): Whether merge encoder and decoder initialization into one ONNX model. Defaults to True.
            no_beam_search_op (bool, optional): Whether to use beam search op or not. Defaults to False.
            output_qk (bool, optional): Whether to output QKs to calculate batched jump times for word-level timestamps. Defaults to False.
        Returns:
            Dict[str, torch.nn.Module]: mapping from name to modules for ONNX conversion.
        """
        # Load PyTorch model
        if model_impl == "hf":
            # Load from Hugging Face
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name_or_path, cache_dir=cache_dir, attn_implementation="eager"
            )
        else:
            # Load from OpenAI
            import whisper

            if not os.path.exists(model_name_or_path):
                name_or_path = model_name_or_path.split("/")[-1][8:]
            else:
                name_or_path = model_name_or_path
            model = whisper.load_model(name_or_path, device, download_root=cache_dir, in_memory=True)

        # Set PyTorch model properties
        model.eval().to(device=device)
        if model_impl == "hf":
            model.to(dtype=dtype)
        config = WhisperConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)

        # Load each component of PyTorch model
        decoder = WhisperDecoder(config, model, model_impl, no_beam_search_op).eval()
        components = {"decoder": decoder}
        if merge_encoder_and_decoder_init:
            encoder_decoder_init = WhisperEncoderDecoderInit(config, model, model_impl, no_beam_search_op).eval()
            components.update({"encoder": encoder_decoder_init})
        else:
            encoder = WhisperEncoder(config, model, model_impl).eval()
            components.update({"encoder": encoder, "decoder_init": decoder})

        if output_qk:
            batched_jump_times = WhisperJumpTimes(config, device, cache_dir).eval()
            components.update({"jump_times": batched_jump_times})
        return components

    @staticmethod
    def export_onnx(
        model: WhisperEncoder | WhisperEncoderDecoderInit | WhisperDecoder,
        onnx_model_path: str,
        provider: str,
        verbose: bool,
        use_external_data_format: bool,
        use_fp16_inputs: bool,
        use_int32_inputs: bool,
        use_encoder_hidden_states: bool,
        use_kv_cache_inputs: bool,
    ):
        """Export model component to ONNX

        Args:
            model (class): PyTorch class to export
            onnx_model_path (str): path to save ONNX model
            provider (str): provider to use for verifying parity on ONNX model
            verbose (bool): print verbose information.
            use_external_data_format (bool): use external data format or not.
            use_fp16_inputs (bool): use float16 inputs for the audio_features, encoder_hidden_states, logits, and KV caches.
            use_int32_inputs (bool): use int32 inputs for the decoder_input_ids.
            use_encoder_hidden_states (bool): use encoder_hidden_states as model input for decoder-init/decoder-without-past models.
            use_kv_cache_inputs (bool): use KV caches as model inputs for decoder-with-past models.
        """
        if isinstance(model, WhisperEncoder):
            model.export_onnx(
                onnx_model_path,
                provider,
                verbose,
                use_external_data_format,
                use_fp16_inputs,
            )
        elif isinstance(model, WhisperEncoderDecoderInit):
            model.export_onnx(
                onnx_model_path,
                provider,
                verbose,
                use_external_data_format,
                use_fp16_inputs,
                use_int32_inputs,
            )
        elif isinstance(model, WhisperDecoder):
            model.export_onnx(
                onnx_model_path,
                provider,
                verbose,
                use_external_data_format,
                use_fp16_inputs,
                use_int32_inputs,
                use_encoder_hidden_states,
                use_kv_cache_inputs,
            )
        elif isinstance(model, WhisperJumpTimes):
            model.export_onnx(
                onnx_model_path,
                provider,
                verbose,
                use_external_data_format,
                use_fp16_inputs,
                use_int32_inputs,
            )
        else:
            raise ValueError(f"Unknown instance for model detected: {type(model)}")

    @staticmethod
    def optimize_onnx(
        onnx_model_path: str,
        optimized_model_path: str,
        is_float16: bool,
        num_attention_heads: int,
        hidden_size: int,
        num_layers: int,
        use_external_data_format: bool = False,
        use_gpu: bool = False,
        provider: str = "cpu",
        is_decoder: bool = False,
        no_beam_search_op: bool = False,
        output_qk: bool = False,
    ):
        """Optimize ONNX model with an option to convert it to use mixed precision."""

        from fusion_options import FusionOptions

        optimization_options = FusionOptions("bart")
        optimization_options.use_multi_head_attention = True
        optimization_options.disable_multi_head_attention_bias = provider == "rocm"

        m = optimize_model(
            onnx_model_path,
            model_type="bart",
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            opt_level=0,
            optimization_options=optimization_options,
            use_gpu=use_gpu,
            only_onnxruntime=False,
        )

        # Add `past_sequence_length`, `cache_indirection`, and `output_qk` to `MultiHeadAttention` ops
        if is_decoder and no_beam_search_op:
            if provider == "cuda":  # FP32 CPU can be supported here once the DMMHA CPU kernel bugs are fixed
                # FP16 CUDA, FP32 CUDA, and FP32 CPU use the `DecoderMaskedMultiHeadAttention` kernel
                # via `MultiHeadAttention`, which requires the `past_sequence_length` and
                # `cache_indirection` inputs
                m, past_seq_len_name = fix_past_sequence_length(m)
                m = add_cache_indirection_to_mha(m, past_seq_len_name)

            if output_qk:
                m = add_output_qk_to_mha(m, skip_node_idxs=list(range(0, 2 * num_layers, 2)))

        m.save_model_to_file(optimized_model_path, use_external_data_format, all_tensors_to_one_file=True)

    @staticmethod
    def pt_transcription_for_verify_onnx(
        processor: WhisperProcessor,
        pt_model: torch.nn.Module,
        device: torch.device,
        batch_size: int = 1,
        prompt_mode: bool = False,
    ):
        # Try to import `datasets` pip package
        try:
            from datasets import load_dataset
        except Exception as e:
            logger.error(f"An error occurred while importing `datasets`: {e}", exc_info=True)  # noqa: G201
            install_cmd = "pip install datasets"
            logger.warning(f"Could not import `datasets`. Attempting to install `datasets` via `{install_cmd}`.")
            os.system(install_cmd)

        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        input_features_ = []
        if batch_size == 1:
            input_features = processor([ds[0]["audio"]["array"]], return_tensors="pt").input_features
        else:
            input_features_ = [
                processor([ds[3]["audio"]["array"]], return_tensors="pt").input_features,
                processor([ds[3]["audio"]["array"]], return_tensors="pt").input_features,
            ]
            assert len(input_features_) == batch_size
            input_features = torch.cat((input_features_[0], input_features_[1]))

        max_length, min_length, num_beams, num_return_sequences = 30, 0, 1, 1
        length_penalty, repetition_penalty = 1.0, 1.0
        inputs = {
            "input_features": input_features.to(device),
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "num_return_sequences": num_return_sequences,
            "length_penalty": length_penalty,
            "repetition_penalty": repetition_penalty,
            "early_stopping": True,
            "use_cache": True,
        }

        if prompt_mode:
            prompts = ["John has doubts", "Maria has grave doubts"]
            prompt_ids = [processor.get_prompt_ids(p) for p in prompts]
            pt_transcription = []
            pt_outputs = []
            # The looping for model.generate is necessary here due to the limitation as per
            # https://huggingface.co/docs/transformers/model_doc/whisper#transformers.WhisperForConditionalGeneration.generate.prompt_ids
            # prompt_ids input requires a tensor of rank 1
            for i in range(batch_size):
                inputs["prompt_ids"] = torch.from_numpy(prompt_ids[i]).to(device=device)
                inputs["input_features"] = input_features_[i].to(device)
                pt_output = pt_model.generate(**inputs).detach().cpu().numpy()
                pt_outputs.append(pt_output)
                pt_transcription.append(processor.batch_decode(pt_output, skip_special_tokens=True)[0])
            inputs["input_features"] = input_features
            del inputs["prompt_ids"]
        else:
            prompt_ids = []
            pt_outputs = pt_model.generate(**inputs).detach().cpu().numpy()
            pt_transcription = [processor.batch_decode(pt_outputs, skip_special_tokens=True)[0]]
            pt_outputs = list(pt_outputs)
        del inputs["early_stopping"]
        del inputs["use_cache"]
        return inputs, pt_transcription, pt_outputs, prompt_ids

    @staticmethod
    def select_transcription_options(
        batch_size: int,
        prompt_mode: bool,
    ):
        if batch_size > 1 and prompt_mode:
            expected_transcription_no_comma_prompt1 = " John has doubts whether Sir Frederick Layton's work is really Greek after all and can discover in it but little of Rocky I"
            expected_transcription_misspelled_prompt1 = " John has doubts whether Sir Frederick Latins work is really Greek after all and can discover in it but little of Rocky I"
            expected_transcription_no_comma_prompt2 = " Maria has grave doubts whether Sir Frederick Layton's work is really Greek after all and can discover in it but little of Rocky"
            expected_transcription_misspelled_prompt2 = " Maria has grave doubts whether Sir Frederick Latins work is really Greek after all and can discover in it but little of Rocky I"
            expected_transcription_options = {
                expected_transcription_no_comma_prompt1,
                expected_transcription_no_comma_prompt2,
                expected_transcription_misspelled_prompt1,
                expected_transcription_misspelled_prompt2,
            }
        else:
            expected_transcription_no_comma = (
                " Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel."
            )
            expected_transcription_with_comma = (
                " Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel."
            )
            expected_transcription_with_quote_and_comma = (
                ' "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel.'
            )
            expected_transcription_options = {
                expected_transcription_no_comma,
                expected_transcription_with_comma,
                expected_transcription_with_quote_and_comma,
            }
        return expected_transcription_options

    @staticmethod
    def get_outputs(
        pt_outputs: np.ndarray,
        ort_outputs: np.ndarray,
        i: int,
    ):
        """Get PyTorch and ONNX Runtime output token ids at index i"""
        pt_output, ort_output = pt_outputs[i], ort_outputs[i]
        pt_shape, ort_shape = pt_output.shape, ort_output.shape

        # Hugging Face impl. + Beam Search op: PyTorch = (26,) and ORT = (30,)
        # OpenAI impl. + Beam Search op: PyTorch = (1, 30) and ORT = (30,)
        if pt_shape != ort_shape:
            if len(pt_shape) > 1:
                pt_output = pt_output[0]
                pt_shape = pt_output.shape
            if len(ort_shape) > 1:
                ort_output = ort_output[0]
                ort_shape = ort_output.shape
            if pt_shape[0] != ort_shape[0]:
                min_len = min(pt_shape[0], ort_shape[0])
                pt_output = pt_output[:min_len]
                ort_output = ort_output[:min_len]

        assert pt_output.shape == ort_output.shape
        return pt_output, ort_output

    @staticmethod
    def verify_onnx(
        model_name_or_path: str,
        cache_dir: str,
        ort_session: InferenceSession,
        device: torch.device,
        batch_size: int = 1,
        prompt_mode: bool = False,
    ):
        """Compare the result from PyTorch and ONNX Runtime to verify the ONNX model is good."""
        pt_model = WhisperForConditionalGeneration.from_pretrained(
            model_name_or_path, cache_dir=cache_dir, attn_implementation="eager"
        ).to(device)
        processor = WhisperProcessor.from_pretrained(model_name_or_path, cache_dir=cache_dir)
        config = WhisperConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)

        inputs, pt_transcription, pt_outputs, decoder_prompt_ids = WhisperHelper.pt_transcription_for_verify_onnx(
            processor,
            pt_model,
            device,
            batch_size=batch_size,
            prompt_mode=prompt_mode,
        )

        start_id = [config.decoder_start_token_id]  # ex: [50258]
        prompt_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
        prompt_ids = [token[1] for token in prompt_ids]  # ex: [50259, 50358, 50363]
        forced_decoder_ids = start_id + prompt_ids  # ex: [50258, 50259, 50358, 50363]

        ort_names = [entry.name for entry in ort_session.get_inputs()]
        ort_dtypes = [entry.type for entry in ort_session.get_inputs()]
        ort_to_np = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
        }

        use_extra_decoding_ids = "extra_decoding_ids" in ort_names
        for name, dtype in zip(ort_names, ort_dtypes, strict=False):
            if name == "input_features":
                inputs[name] = inputs[name].detach().cpu().numpy()
            elif name == "vocab_mask":
                inputs[name] = np.ones(config.vocab_size, dtype=ort_to_np[dtype])
            elif name == "prefix_vocab_mask":
                inputs[name] = np.ones((batch_size, config.vocab_size), dtype=ort_to_np[dtype])
            elif name == "decoder_input_ids":
                if not prompt_mode:
                    raw_input_ids = [start_id] if use_extra_decoding_ids else [forced_decoder_ids]
                    inputs[name] = np.array(raw_input_ids, dtype=ort_to_np[dtype])
                else:
                    # This logic handles the scenario for when prompts are not of the same size
                    # For example if our prompt ids are [p1_id_1, p1_id_2] and [p2_id_1]
                    # The final decoder_input_ids will look as such after padding
                    # [prev_token, p1_id_1, p1_id_2, start_token, lang_token, transcribe_token]
                    # [prev_token, p2_id_1, PAD_TOKEN, start_token, lang_token, transcribe_token]
                    ort_prompts = []
                    for i in range(batch_size):
                        ort_prompts.append(decoder_prompt_ids[i].tolist())
                    max_len = max(len(p) for p in ort_prompts)
                    padded_prompts = []
                    for p in ort_prompts:
                        padded_prompt = [*p, *([config.pad_token_id] * (max_len - len(p)))]
                        padded_prompts.append(padded_prompt + forced_decoder_ids)
                    inputs[name] = np.array(padded_prompts, dtype=ort_to_np[dtype])
            elif name == "logits_processor":
                inputs[name] = np.array([1], dtype=ort_to_np[dtype])
            elif name == "cross_qk_layer_head":
                inputs[name] = np.array([[0, 0]], dtype=ort_to_np[dtype])
            elif name == "extra_decoding_ids":
                inputs[name] = np.repeat(np.array([prompt_ids], dtype=ort_to_np[dtype]), batch_size, 0)
            elif name == "temperature":
                inputs[name] = np.array([1.0], dtype=ort_to_np[dtype])
            else:
                inputs[name] = np.array([inputs[name]], dtype=ort_to_np[dtype])

        ort_outputs = ort_session.run(None, inputs)[0][:, 0, :]
        ort_transcription = processor.batch_decode(ort_outputs, skip_special_tokens=True)
        expected_transcription_options = WhisperHelper.select_transcription_options(batch_size, prompt_mode)

        parity = 1
        for i in range(batch_size):
            pt_output, ort_output = WhisperHelper.get_outputs(pt_outputs, ort_outputs, i)

            # Check if token ids match
            parity *= np.allclose(pt_output, ort_output)

            # Check if transcribed outputs match
            parity *= (
                pt_transcription[i] in expected_transcription_options
                and ort_transcription[i] in expected_transcription_options
            )
        max_diff = 0

        if not parity:
            for i in range(batch_size):
                pt_output, ort_output = WhisperHelper.get_outputs(pt_outputs, ort_outputs, i)
                diff = pt_output - ort_output

                max_diff_i = max(diff.min(), diff.max(), key=abs)
                max_diff = max(max_diff, max_diff_i)

        if max_diff != 0:
            logger.warning(f"PyTorch outputs: {pt_transcription}")
            logger.warning(f"ONNX Runtime outputs: {ort_transcription}")

        return 0

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_spinners.py ===
"""
Spinners are from:
* cli-spinners:
    MIT License
    Copyright (c) Sindre Sorhus <sindresorhus@gmail.com> (sindresorhus.com)
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights to
    use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
    the Software, and to permit persons to whom the Software is furnished to do so,
    subject to the following conditions:
    The above copyright notice and this permission notice shall be included
    in all copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
    INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
    PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
    FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
    IN THE SOFTWARE.
"""

SPINNERS = {
    "dots": {
        "interval": 80,
        "frames": "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏",
    },
    "dots2": {"interval": 80, "frames": "⣾⣽⣻⢿⡿⣟⣯⣷"},
    "dots3": {
        "interval": 80,
        "frames": "⠋⠙⠚⠞⠖⠦⠴⠲⠳⠓",
    },
    "dots4": {
        "interval": 80,
        "frames": "⠄⠆⠇⠋⠙⠸⠰⠠⠰⠸⠙⠋⠇⠆",
    },
    "dots5": {
        "interval": 80,
        "frames": "⠋⠙⠚⠒⠂⠂⠒⠲⠴⠦⠖⠒⠐⠐⠒⠓⠋",
    },
    "dots6": {
        "interval": 80,
        "frames": "⠁⠉⠙⠚⠒⠂⠂⠒⠲⠴⠤⠄⠄⠤⠴⠲⠒⠂⠂⠒⠚⠙⠉⠁",
    },
    "dots7": {
        "interval": 80,
        "frames": "⠈⠉⠋⠓⠒⠐⠐⠒⠖⠦⠤⠠⠠⠤⠦⠖⠒⠐⠐⠒⠓⠋⠉⠈",
    },
    "dots8": {
        "interval": 80,
        "frames": "⠁⠁⠉⠙⠚⠒⠂⠂⠒⠲⠴⠤⠄⠄⠤⠠⠠⠤⠦⠖⠒⠐⠐⠒⠓⠋⠉⠈⠈",
    },
    "dots9": {"interval": 80, "frames": "⢹⢺⢼⣸⣇⡧⡗⡏"},
    "dots10": {"interval": 80, "frames": "⢄⢂⢁⡁⡈⡐⡠"},
    "dots11": {"interval": 100, "frames": "⠁⠂⠄⡀⢀⠠⠐⠈"},
    "dots12": {
        "interval": 80,
        "frames": [
            "⢀⠀",
            "⡀⠀",
            "⠄⠀",
            "⢂⠀",
            "⡂⠀",
            "⠅⠀",
            "⢃⠀",
            "⡃⠀",
            "⠍⠀",
            "⢋⠀",
            "⡋⠀",
            "⠍⠁",
            "⢋⠁",
            "⡋⠁",
            "⠍⠉",
            "⠋⠉",
            "⠋⠉",
            "⠉⠙",
            "⠉⠙",
            "⠉⠩",
            "⠈⢙",
            "⠈⡙",
            "⢈⠩",
            "⡀⢙",
            "⠄⡙",
            "⢂⠩",
            "⡂⢘",
            "⠅⡘",
            "⢃⠨",
            "⡃⢐",
            "⠍⡐",
            "⢋⠠",
            "⡋⢀",
            "⠍⡁",
            "⢋⠁",
            "⡋⠁",
            "⠍⠉",
            "⠋⠉",
            "⠋⠉",
            "⠉⠙",
            "⠉⠙",
            "⠉⠩",
            "⠈⢙",
            "⠈⡙",
            "⠈⠩",
            "⠀⢙",
            "⠀⡙",
            "⠀⠩",
            "⠀⢘",
            "⠀⡘",
            "⠀⠨",
            "⠀⢐",
            "⠀⡐",
            "⠀⠠",
            "⠀⢀",
            "⠀⡀",
        ],
    },
    "dots8Bit": {
        "interval": 80,
        "frames": "⠀⠁⠂⠃⠄⠅⠆⠇⡀⡁⡂⡃⡄⡅⡆⡇⠈⠉⠊⠋⠌⠍⠎⠏⡈⡉⡊⡋⡌⡍⡎⡏⠐⠑⠒⠓⠔⠕⠖⠗⡐⡑⡒⡓⡔⡕⡖⡗⠘⠙⠚⠛⠜⠝⠞⠟⡘⡙"
        "⡚⡛⡜⡝⡞⡟⠠⠡⠢⠣⠤⠥⠦⠧⡠⡡⡢⡣⡤⡥⡦⡧⠨⠩⠪⠫⠬⠭⠮⠯⡨⡩⡪⡫⡬⡭⡮⡯⠰⠱⠲⠳⠴⠵⠶⠷⡰⡱⡲⡳⡴⡵⡶⡷⠸⠹⠺⠻"
        "⠼⠽⠾⠿⡸⡹⡺⡻⡼⡽⡾⡿⢀⢁⢂⢃⢄⢅⢆⢇⣀⣁⣂⣃⣄⣅⣆⣇⢈⢉⢊⢋⢌⢍⢎⢏⣈⣉⣊⣋⣌⣍⣎⣏⢐⢑⢒⢓⢔⢕⢖⢗⣐⣑⣒⣓⣔⣕"
        "⣖⣗⢘⢙⢚⢛⢜⢝⢞⢟⣘⣙⣚⣛⣜⣝⣞⣟⢠⢡⢢⢣⢤⢥⢦⢧⣠⣡⣢⣣⣤⣥⣦⣧⢨⢩⢪⢫⢬⢭⢮⢯⣨⣩⣪⣫⣬⣭⣮⣯⢰⢱⢲⢳⢴⢵⢶⢷"
        "⣰⣱⣲⣳⣴⣵⣶⣷⢸⢹⢺⢻⢼⢽⢾⢿⣸⣹⣺⣻⣼⣽⣾⣿",
    },
    "line": {"interval": 130, "frames": ["-", "\\", "|", "/"]},
    "line2": {"interval": 100, "frames": "⠂-–—–-"},
    "pipe": {"interval": 100, "frames": "┤┘┴└├┌┬┐"},
    "simpleDots": {"interval": 400, "frames": [".  ", ".. ", "...", "   "]},
    "simpleDotsScrolling": {
        "interval": 200,
        "frames": [".  ", ".. ", "...", " ..", "  .", "   "],
    },
    "star": {"interval": 70, "frames": "✶✸✹✺✹✷"},
    "star2": {"interval": 80, "frames": "+x*"},
    "flip": {
        "interval": 70,
        "frames": "___-``'´-___",
    },
    "hamburger": {"interval": 100, "frames": "☱☲☴"},
    "growVertical": {
        "interval": 120,
        "frames": "▁▃▄▅▆▇▆▅▄▃",
    },
    "growHorizontal": {
        "interval": 120,
        "frames": "▏▎▍▌▋▊▉▊▋▌▍▎",
    },
    "balloon": {"interval": 140, "frames": " .oO@* "},
    "balloon2": {"interval": 120, "frames": ".oO°Oo."},
    "noise": {"interval": 100, "frames": "▓▒░"},
    "bounce": {"interval": 120, "frames": "⠁⠂⠄⠂"},
    "boxBounce": {"interval": 120, "frames": "▖▘▝▗"},
    "boxBounce2": {"interval": 100, "frames": "▌▀▐▄"},
    "triangle": {"interval": 50, "frames": "◢◣◤◥"},
    "arc": {"interval": 100, "frames": "◜◠◝◞◡◟"},
    "circle": {"interval": 120, "frames": "◡⊙◠"},
    "squareCorners": {"interval": 180, "frames": "◰◳◲◱"},
    "circleQuarters": {"interval": 120, "frames": "◴◷◶◵"},
    "circleHalves": {"interval": 50, "frames": "◐◓◑◒"},
    "squish": {"interval": 100, "frames": "╫╪"},
    "toggle": {"interval": 250, "frames": "⊶⊷"},
    "toggle2": {"interval": 80, "frames": "▫▪"},
    "toggle3": {"interval": 120, "frames": "□■"},
    "toggle4": {"interval": 100, "frames": "■□▪▫"},
    "toggle5": {"interval": 100, "frames": "▮▯"},
    "toggle6": {"interval": 300, "frames": "ဝ၀"},
    "toggle7": {"interval": 80, "frames": "⦾⦿"},
    "toggle8": {"interval": 100, "frames": "◍◌"},
    "toggle9": {"interval": 100, "frames": "◉◎"},
    "toggle10": {"interval": 100, "frames": "㊂㊀㊁"},
    "toggle11": {"interval": 50, "frames": "⧇⧆"},
    "toggle12": {"interval": 120, "frames": "☗☖"},
    "toggle13": {"interval": 80, "frames": "=*-"},
    "arrow": {"interval": 100, "frames": "←↖↑↗→↘↓↙"},
    "arrow2": {
        "interval": 80,
        "frames": ["⬆️ ", "↗️ ", "➡️ ", "↘️ ", "⬇️ ", "↙️ ", "⬅️ ", "↖️ "],
    },
    "arrow3": {
        "interval": 120,
        "frames": ["▹▹▹▹▹", "▸▹▹▹▹", "▹▸▹▹▹", "▹▹▸▹▹", "▹▹▹▸▹", "▹▹▹▹▸"],
    },
    "bouncingBar": {
        "interval": 80,
        "frames": [
            "[    ]",
            "[=   ]",
            "[==  ]",
            "[=== ]",
            "[ ===]",
            "[  ==]",
            "[   =]",
            "[    ]",
            "[   =]",
            "[  ==]",
            "[ ===]",
            "[====]",
            "[=== ]",
            "[==  ]",
            "[=   ]",
        ],
    },
    "bouncingBall": {
        "interval": 80,
        "frames": [
            "( ●    )",
            "(  ●   )",
            "(   ●  )",
            "(    ● )",
            "(     ●)",
            "(    ● )",
            "(   ●  )",
            "(  ●   )",
            "( ●    )",
            "(●     )",
        ],
    },
    "smiley": {"interval": 200, "frames": ["😄 ", "😝 "]},
    "monkey": {"interval": 300, "frames": ["🙈 ", "🙈 ", "🙉 ", "🙊 "]},
    "hearts": {"interval": 100, "frames": ["💛 ", "💙 ", "💜 ", "💚 ", "❤️ "]},
    "clock": {
        "interval": 100,
        "frames": [
            "🕛 ",
            "🕐 ",
            "🕑 ",
            "🕒 ",
            "🕓 ",
            "🕔 ",
            "🕕 ",
            "🕖 ",
            "🕗 ",
            "🕘 ",
            "🕙 ",
            "🕚 ",
        ],
    },
    "earth": {"interval": 180, "frames": ["🌍 ", "🌎 ", "🌏 "]},
    "material": {
        "interval": 17,
        "frames": [
            "█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "██▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "███▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "████▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "██████▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "██████▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "███████▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "████████▁▁▁▁▁▁▁▁▁▁▁▁",
            "█████████▁▁▁▁▁▁▁▁▁▁▁",
            "█████████▁▁▁▁▁▁▁▁▁▁▁",
            "██████████▁▁▁▁▁▁▁▁▁▁",
            "███████████▁▁▁▁▁▁▁▁▁",
            "█████████████▁▁▁▁▁▁▁",
            "██████████████▁▁▁▁▁▁",
            "██████████████▁▁▁▁▁▁",
            "▁██████████████▁▁▁▁▁",
            "▁██████████████▁▁▁▁▁",
            "▁██████████████▁▁▁▁▁",
            "▁▁██████████████▁▁▁▁",
            "▁▁▁██████████████▁▁▁",
            "▁▁▁▁█████████████▁▁▁",
            "▁▁▁▁██████████████▁▁",
            "▁▁▁▁██████████████▁▁",
            "▁▁▁▁▁██████████████▁",
            "▁▁▁▁▁██████████████▁",
            "▁▁▁▁▁██████████████▁",
            "▁▁▁▁▁▁██████████████",
            "▁▁▁▁▁▁██████████████",
            "▁▁▁▁▁▁▁█████████████",
            "▁▁▁▁▁▁▁█████████████",
            "▁▁▁▁▁▁▁▁████████████",
            "▁▁▁▁▁▁▁▁████████████",
            "▁▁▁▁▁▁▁▁▁███████████",
            "▁▁▁▁▁▁▁▁▁███████████",
            "▁▁▁▁▁▁▁▁▁▁██████████",
            "▁▁▁▁▁▁▁▁▁▁██████████",
            "▁▁▁▁▁▁▁▁▁▁▁▁████████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁███████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁██████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████",
            "█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁████",
            "██▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁███",
            "██▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁███",
            "███▁▁▁▁▁▁▁▁▁▁▁▁▁▁███",
            "████▁▁▁▁▁▁▁▁▁▁▁▁▁▁██",
            "█████▁▁▁▁▁▁▁▁▁▁▁▁▁▁█",
            "█████▁▁▁▁▁▁▁▁▁▁▁▁▁▁█",
            "██████▁▁▁▁▁▁▁▁▁▁▁▁▁█",
            "████████▁▁▁▁▁▁▁▁▁▁▁▁",
            "█████████▁▁▁▁▁▁▁▁▁▁▁",
            "█████████▁▁▁▁▁▁▁▁▁▁▁",
            "█████████▁▁▁▁▁▁▁▁▁▁▁",
            "█████████▁▁▁▁▁▁▁▁▁▁▁",
            "███████████▁▁▁▁▁▁▁▁▁",
            "████████████▁▁▁▁▁▁▁▁",
            "████████████▁▁▁▁▁▁▁▁",
            "██████████████▁▁▁▁▁▁",
            "██████████████▁▁▁▁▁▁",
            "▁██████████████▁▁▁▁▁",
            "▁██████████████▁▁▁▁▁",
            "▁▁▁█████████████▁▁▁▁",
            "▁▁▁▁▁████████████▁▁▁",
            "▁▁▁▁▁████████████▁▁▁",
            "▁▁▁▁▁▁███████████▁▁▁",
            "▁▁▁▁▁▁▁▁█████████▁▁▁",
            "▁▁▁▁▁▁▁▁█████████▁▁▁",
            "▁▁▁▁▁▁▁▁▁█████████▁▁",
            "▁▁▁▁▁▁▁▁▁█████████▁▁",
            "▁▁▁▁▁▁▁▁▁▁█████████▁",
            "▁▁▁▁▁▁▁▁▁▁▁████████▁",
            "▁▁▁▁▁▁▁▁▁▁▁████████▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁███████▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁███████▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁███████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁███████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁████",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁███",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁███",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁██",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁██",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁██",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
            "▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁",
        ],
    },
    "moon": {
        "interval": 80,
        "frames": ["🌑 ", "🌒 ", "🌓 ", "🌔 ", "🌕 ", "🌖 ", "🌗 ", "🌘 "],
    },
    "runner": {"interval": 140, "frames": ["🚶 ", "🏃 "]},
    "pong": {
        "interval": 80,
        "frames": [
            "▐⠂       ▌",
            "▐⠈       ▌",
            "▐ ⠂      ▌",
            "▐ ⠠      ▌",
            "▐  ⡀     ▌",
            "▐  ⠠     ▌",
            "▐   ⠂    ▌",
            "▐   ⠈    ▌",
            "▐    ⠂   ▌",
            "▐    ⠠   ▌",
            "▐     ⡀  ▌",
            "▐     ⠠  ▌",
            "▐      ⠂ ▌",
            "▐      ⠈ ▌",
            "▐       ⠂▌",
            "▐       ⠠▌",
            "▐       ⡀▌",
            "▐      ⠠ ▌",
            "▐      ⠂ ▌",
            "▐     ⠈  ▌",
            "▐     ⠂  ▌",
            "▐    ⠠   ▌",
            "▐    ⡀   ▌",
            "▐   ⠠    ▌",
            "▐   ⠂    ▌",
            "▐  ⠈     ▌",
            "▐  ⠂     ▌",
            "▐ ⠠      ▌",
            "▐ ⡀      ▌",
            "▐⠠       ▌",
        ],
    },
    "shark": {
        "interval": 120,
        "frames": [
            "▐|\\____________▌",
            "▐_|\\___________▌",
            "▐__|\\__________▌",
            "▐___|\\_________▌",
            "▐____|\\________▌",
            "▐_____|\\_______▌",
            "▐______|\\______▌",
            "▐_______|\\_____▌",
            "▐________|\\____▌",
            "▐_________|\\___▌",
            "▐__________|\\__▌",
            "▐___________|\\_▌",
            "▐____________|\\▌",
            "▐____________/|▌",
            "▐___________/|_▌",
            "▐__________/|__▌",
            "▐_________/|___▌",
            "▐________/|____▌",
            "▐_______/|_____▌",
            "▐______/|______▌",
            "▐_____/|_______▌",
            "▐____/|________▌",
            "▐___/|_________▌",
            "▐__/|__________▌",
            "▐_/|___________▌",
            "▐/|____________▌",
        ],
    },
    "dqpb": {"interval": 100, "frames": "dqpb"},
    "weather": {
        "interval": 100,
        "frames": [
            "☀️ ",
            "☀️ ",
            "☀️ ",
            "🌤 ",
            "⛅️ ",
            "🌥 ",
            "☁️ ",
            "🌧 ",
            "🌨 ",
            "🌧 ",
            "🌨 ",
            "🌧 ",
            "🌨 ",
            "⛈ ",
            "🌨 ",
            "🌧 ",
            "🌨 ",
            "☁️ ",
            "🌥 ",
            "⛅️ ",
            "🌤 ",
            "☀️ ",
            "☀️ ",
        ],
    },
    "christmas": {"interval": 400, "frames": "🌲🎄"},
    "grenade": {
        "interval": 80,
        "frames": [
            "،   ",
            "′   ",
            " ´ ",
            " ‾ ",
            "  ⸌",
            "  ⸊",
            "  |",
            "  ⁎",
            "  ⁕",
            " ෴ ",
            "  ⁓",
            "   ",
            "   ",
            "   ",
        ],
    },
    "point": {"interval": 125, "frames": ["∙∙∙", "●∙∙", "∙●∙", "∙∙●", "∙∙∙"]},
    "layer": {"interval": 150, "frames": "-=≡"},
    "betaWave": {
        "interval": 80,
        "frames": [
            "ρββββββ",
            "βρβββββ",
            "ββρββββ",
            "βββρβββ",
            "ββββρββ",
            "βββββρβ",
            "ββββββρ",
        ],
    },
    "aesthetic": {
        "interval": 80,
        "frames": [
            "▰▱▱▱▱▱▱",
            "▰▰▱▱▱▱▱",
            "▰▰▰▱▱▱▱",
            "▰▰▰▰▱▱▱",
            "▰▰▰▰▰▱▱",
            "▰▰▰▰▰▰▱",
            "▰▰▰▰▰▰▰",
            "▰▱▱▱▱▱▱",
        ],
    },
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_matplotlib\misc.py ===
from __future__ import annotations

import random
from typing import TYPE_CHECKING

from matplotlib import patches
import matplotlib.lines as mlines
import numpy as np

from pandas.core.dtypes.missing import notna

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.style import get_standard_colors
from pandas.plotting._matplotlib.tools import (
    create_subplots,
    do_adjust_figure,
    maybe_adjust_figure,
    set_ticks_props,
)

if TYPE_CHECKING:
    from collections.abc import Hashable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from pandas import (
        DataFrame,
        Index,
        Series,
    )


def scatter_matrix(
    frame: DataFrame,
    alpha: float = 0.5,
    figsize: tuple[float, float] | None = None,
    ax=None,
    grid: bool = False,
    diagonal: str = "hist",
    marker: str = ".",
    density_kwds=None,
    hist_kwds=None,
    range_padding: float = 0.05,
    **kwds,
):
    df = frame._get_numeric_data()
    n = df.columns.size
    naxes = n * n
    fig, axes = create_subplots(naxes=naxes, figsize=figsize, ax=ax, squeeze=False)

    # no gaps between subplots
    maybe_adjust_figure(fig, wspace=0, hspace=0)

    mask = notna(df)

    marker = _get_marker_compat(marker)

    hist_kwds = hist_kwds or {}
    density_kwds = density_kwds or {}

    # GH 14855
    kwds.setdefault("edgecolors", "none")

    boundaries_list = []
    for a in df.columns:
        values = df[a].values[mask[a].values]
        rmin_, rmax_ = np.min(values), np.max(values)
        rdelta_ext = (rmax_ - rmin_) * range_padding / 2
        boundaries_list.append((rmin_ - rdelta_ext, rmax_ + rdelta_ext))

    for i, a in enumerate(df.columns):
        for j, b in enumerate(df.columns):
            ax = axes[i, j]

            if i == j:
                values = df[a].values[mask[a].values]

                # Deal with the diagonal by drawing a histogram there.
                if diagonal == "hist":
                    ax.hist(values, **hist_kwds)

                elif diagonal in ("kde", "density"):
                    from scipy.stats import gaussian_kde

                    y = values
                    gkde = gaussian_kde(y)
                    ind = np.linspace(y.min(), y.max(), 1000)
                    ax.plot(ind, gkde.evaluate(ind), **density_kwds)

                ax.set_xlim(boundaries_list[i])

            else:
                common = (mask[a] & mask[b]).values

                ax.scatter(
                    df[b][common], df[a][common], marker=marker, alpha=alpha, **kwds
                )

                ax.set_xlim(boundaries_list[j])
                ax.set_ylim(boundaries_list[i])

            ax.set_xlabel(b)
            ax.set_ylabel(a)

            if j != 0:
                ax.yaxis.set_visible(False)
            if i != n - 1:
                ax.xaxis.set_visible(False)

    if len(df.columns) > 1:
        lim1 = boundaries_list[0]
        locs = axes[0][1].yaxis.get_majorticklocs()
        locs = locs[(lim1[0] <= locs) & (locs <= lim1[1])]
        adj = (locs - lim1[0]) / (lim1[1] - lim1[0])

        lim0 = axes[0][0].get_ylim()
        adj = adj * (lim0[1] - lim0[0]) + lim0[0]
        axes[0][0].yaxis.set_ticks(adj)

        if np.all(locs == locs.astype(int)):
            # if all ticks are int
            locs = locs.astype(int)
        axes[0][0].yaxis.set_ticklabels(locs)

    set_ticks_props(axes, xlabelsize=8, xrot=90, ylabelsize=8, yrot=0)

    return axes


def _get_marker_compat(marker):
    if marker not in mlines.lineMarkers:
        return "o"
    return marker


def radviz(
    frame: DataFrame,
    class_column,
    ax: Axes | None = None,
    color=None,
    colormap=None,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    def normalize(series):
        a = min(series)
        b = max(series)
        return (series - a) / (b - a)

    n = len(frame)
    classes = frame[class_column].drop_duplicates()
    class_col = frame[class_column]
    df = frame.drop(class_column, axis=1).apply(normalize)

    if ax is None:
        ax = plt.gca()
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)

    to_plot: dict[Hashable, list[list]] = {}
    colors = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )

    for kls in classes:
        to_plot[kls] = [[], []]

    m = len(frame.columns) - 1
    s = np.array(
        [(np.cos(t), np.sin(t)) for t in [2 * np.pi * (i / m) for i in range(m)]]
    )

    for i in range(n):
        row = df.iloc[i].values
        row_ = np.repeat(np.expand_dims(row, axis=1), 2, axis=1)
        y = (s * row_).sum(axis=0) / row.sum()
        kls = class_col.iat[i]
        to_plot[kls][0].append(y[0])
        to_plot[kls][1].append(y[1])

    for i, kls in enumerate(classes):
        ax.scatter(
            to_plot[kls][0],
            to_plot[kls][1],
            color=colors[i],
            label=pprint_thing(kls),
            **kwds,
        )
    ax.legend()

    ax.add_patch(patches.Circle((0.0, 0.0), radius=1.0, facecolor="none"))

    for xy, name in zip(s, df.columns):
        ax.add_patch(patches.Circle(xy, radius=0.025, facecolor="gray"))

        if xy[0] < 0.0 and xy[1] < 0.0:
            ax.text(
                xy[0] - 0.025, xy[1] - 0.025, name, ha="right", va="top", size="small"
            )
        elif xy[0] < 0.0 <= xy[1]:
            ax.text(
                xy[0] - 0.025,
                xy[1] + 0.025,
                name,
                ha="right",
                va="bottom",
                size="small",
            )
        elif xy[1] < 0.0 <= xy[0]:
            ax.text(
                xy[0] + 0.025, xy[1] - 0.025, name, ha="left", va="top", size="small"
            )
        elif xy[0] >= 0.0 and xy[1] >= 0.0:
            ax.text(
                xy[0] + 0.025, xy[1] + 0.025, name, ha="left", va="bottom", size="small"
            )

    ax.axis("equal")
    return ax


def andrews_curves(
    frame: DataFrame,
    class_column,
    ax: Axes | None = None,
    samples: int = 200,
    color=None,
    colormap=None,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    def function(amplitudes):
        def f(t):
            x1 = amplitudes[0]
            result = x1 / np.sqrt(2.0)

            # Take the rest of the coefficients and resize them
            # appropriately. Take a copy of amplitudes as otherwise numpy
            # deletes the element from amplitudes itself.
            coeffs = np.delete(np.copy(amplitudes), 0)
            coeffs = np.resize(coeffs, (int((coeffs.size + 1) / 2), 2))

            # Generate the harmonics and arguments for the sin and cos
            # functions.
            harmonics = np.arange(0, coeffs.shape[0]) + 1
            trig_args = np.outer(harmonics, t)

            result += np.sum(
                coeffs[:, 0, np.newaxis] * np.sin(trig_args)
                + coeffs[:, 1, np.newaxis] * np.cos(trig_args),
                axis=0,
            )
            return result

        return f

    n = len(frame)
    class_col = frame[class_column]
    classes = frame[class_column].drop_duplicates()
    df = frame.drop(class_column, axis=1)
    t = np.linspace(-np.pi, np.pi, samples)
    used_legends: set[str] = set()

    color_values = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )
    colors = dict(zip(classes, color_values))
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(-np.pi, np.pi)
    for i in range(n):
        row = df.iloc[i].values
        f = function(row)
        y = f(t)
        kls = class_col.iat[i]
        label = pprint_thing(kls)
        if label not in used_legends:
            used_legends.add(label)
            ax.plot(t, y, color=colors[kls], label=label, **kwds)
        else:
            ax.plot(t, y, color=colors[kls], **kwds)

    ax.legend(loc="upper right")
    ax.grid()
    return ax


def bootstrap_plot(
    series: Series,
    fig: Figure | None = None,
    size: int = 50,
    samples: int = 500,
    **kwds,
) -> Figure:
    import matplotlib.pyplot as plt

    # TODO: is the failure mentioned below still relevant?
    # random.sample(ndarray, int) fails on python 3.3, sigh
    data = list(series.values)
    samplings = [random.sample(data, size) for _ in range(samples)]

    means = np.array([np.mean(sampling) for sampling in samplings])
    medians = np.array([np.median(sampling) for sampling in samplings])
    midranges = np.array(
        [(min(sampling) + max(sampling)) * 0.5 for sampling in samplings]
    )
    if fig is None:
        fig = plt.figure()
    x = list(range(samples))
    axes = []
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.set_xlabel("Sample")
    axes.append(ax1)
    ax1.plot(x, means, **kwds)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.set_xlabel("Sample")
    axes.append(ax2)
    ax2.plot(x, medians, **kwds)
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.set_xlabel("Sample")
    axes.append(ax3)
    ax3.plot(x, midranges, **kwds)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.set_xlabel("Mean")
    axes.append(ax4)
    ax4.hist(means, **kwds)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.set_xlabel("Median")
    axes.append(ax5)
    ax5.hist(medians, **kwds)
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.set_xlabel("Midrange")
    axes.append(ax6)
    ax6.hist(midranges, **kwds)
    for axis in axes:
        plt.setp(axis.get_xticklabels(), fontsize=8)
        plt.setp(axis.get_yticklabels(), fontsize=8)
    if do_adjust_figure(fig):
        plt.tight_layout()
    return fig


def parallel_coordinates(
    frame: DataFrame,
    class_column,
    cols=None,
    ax: Axes | None = None,
    color=None,
    use_columns: bool = False,
    xticks=None,
    colormap=None,
    axvlines: bool = True,
    axvlines_kwds=None,
    sort_labels: bool = False,
    **kwds,
) -> Axes:
    import matplotlib.pyplot as plt

    if axvlines_kwds is None:
        axvlines_kwds = {"linewidth": 1, "color": "black"}

    n = len(frame)
    classes = frame[class_column].drop_duplicates()
    class_col = frame[class_column]

    if cols is None:
        df = frame.drop(class_column, axis=1)
    else:
        df = frame[cols]

    used_legends: set[str] = set()

    ncols = len(df.columns)

    # determine values to use for xticks
    x: list[int] | Index
    if use_columns is True:
        if not np.all(np.isreal(list(df.columns))):
            raise ValueError("Columns must be numeric to be used as xticks")
        x = df.columns
    elif xticks is not None:
        if not np.all(np.isreal(xticks)):
            raise ValueError("xticks specified must be numeric")
        if len(xticks) != ncols:
            raise ValueError("Length of xticks must match number of columns")
        x = xticks
    else:
        x = list(range(ncols))

    if ax is None:
        ax = plt.gca()

    color_values = get_standard_colors(
        num_colors=len(classes), colormap=colormap, color_type="random", color=color
    )

    if sort_labels:
        classes = sorted(classes)
        color_values = sorted(color_values)
    colors = dict(zip(classes, color_values))

    for i in range(n):
        y = df.iloc[i].values
        kls = class_col.iat[i]
        label = pprint_thing(kls)
        if label not in used_legends:
            used_legends.add(label)
            ax.plot(x, y, color=colors[kls], label=label, **kwds)
        else:
            ax.plot(x, y, color=colors[kls], **kwds)

    if axvlines:
        for i in x:
            ax.axvline(i, **axvlines_kwds)

    ax.set_xticks(x)
    ax.set_xticklabels(df.columns)
    ax.set_xlim(x[0], x[-1])
    ax.legend(loc="upper right")
    ax.grid()
    return ax


def lag_plot(series: Series, lag: int = 1, ax: Axes | None = None, **kwds) -> Axes:
    # workaround because `c='b'` is hardcoded in matplotlib's scatter method
    import matplotlib.pyplot as plt

    kwds.setdefault("c", plt.rcParams["patch.facecolor"])

    data = series.values
    y1 = data[:-lag]
    y2 = data[lag:]
    if ax is None:
        ax = plt.gca()
    ax.set_xlabel("y(t)")
    ax.set_ylabel(f"y(t + {lag})")
    ax.scatter(y1, y2, **kwds)
    return ax


def autocorrelation_plot(series: Series, ax: Axes | None = None, **kwds) -> Axes:
    import matplotlib.pyplot as plt

    n = len(series)
    data = np.asarray(series)
    if ax is None:
        ax = plt.gca()
        ax.set_xlim(1, n)
        ax.set_ylim(-1.0, 1.0)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0

    x = np.arange(n) + 1
    y = [r(loc) for loc in x]
    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.axhline(y=z95 / np.sqrt(n), color="grey")
    ax.axhline(y=0.0, color="black")
    ax.axhline(y=-z95 / np.sqrt(n), color="grey")
    ax.axhline(y=-z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.plot(x, y, **kwds)
    if "label" in kwds:
        ax.legend()
    ax.grid()
    return ax


def unpack_single_str_list(keys):
    # GH 42795
    if isinstance(keys, list) and len(keys) == 1:
        keys = keys[0]
    return keys

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\box.py ===
import sys
from typing import TYPE_CHECKING, Iterable, List

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from pip._vendor.typing_extensions import Literal  # pragma: no cover


from ._loop import loop_last

if TYPE_CHECKING:
    from pip._vendor.rich.console import ConsoleOptions


class Box:
    """Defines characters to render boxes.

    ┌─┬┐ top
    │ ││ head
    ├─┼┤ head_row
    │ ││ mid
    ├─┼┤ row
    ├─┼┤ foot_row
    │ ││ foot
    └─┴┘ bottom

    Args:
        box (str): Characters making up box.
        ascii (bool, optional): True if this box uses ascii characters only. Default is False.
    """

    def __init__(self, box: str, *, ascii: bool = False) -> None:
        self._box = box
        self.ascii = ascii
        line1, line2, line3, line4, line5, line6, line7, line8 = box.splitlines()
        # top
        self.top_left, self.top, self.top_divider, self.top_right = iter(line1)
        # head
        self.head_left, _, self.head_vertical, self.head_right = iter(line2)
        # head_row
        (
            self.head_row_left,
            self.head_row_horizontal,
            self.head_row_cross,
            self.head_row_right,
        ) = iter(line3)

        # mid
        self.mid_left, _, self.mid_vertical, self.mid_right = iter(line4)
        # row
        self.row_left, self.row_horizontal, self.row_cross, self.row_right = iter(line5)
        # foot_row
        (
            self.foot_row_left,
            self.foot_row_horizontal,
            self.foot_row_cross,
            self.foot_row_right,
        ) = iter(line6)
        # foot
        self.foot_left, _, self.foot_vertical, self.foot_right = iter(line7)
        # bottom
        self.bottom_left, self.bottom, self.bottom_divider, self.bottom_right = iter(
            line8
        )

    def __repr__(self) -> str:
        return "Box(...)"

    def __str__(self) -> str:
        return self._box

    def substitute(self, options: "ConsoleOptions", safe: bool = True) -> "Box":
        """Substitute this box for another if it won't render due to platform issues.

        Args:
            options (ConsoleOptions): Console options used in rendering.
            safe (bool, optional): Substitute this for another Box if there are known problems
                displaying on the platform (currently only relevant on Windows). Default is True.

        Returns:
            Box: A different Box or the same Box.
        """
        box = self
        if options.legacy_windows and safe:
            box = LEGACY_WINDOWS_SUBSTITUTIONS.get(box, box)
        if options.ascii_only and not box.ascii:
            box = ASCII
        return box

    def get_plain_headed_box(self) -> "Box":
        """If this box uses special characters for the borders of the header, then
        return the equivalent box that does not.

        Returns:
            Box: The most similar Box that doesn't use header-specific box characters.
                If the current Box already satisfies this criterion, then it's returned.
        """
        return PLAIN_HEADED_SUBSTITUTIONS.get(self, self)

    def get_top(self, widths: Iterable[int]) -> str:
        """Get the top of a simple box.

        Args:
            widths (List[int]): Widths of columns.

        Returns:
            str: A string of box characters.
        """

        parts: List[str] = []
        append = parts.append
        append(self.top_left)
        for last, width in loop_last(widths):
            append(self.top * width)
            if not last:
                append(self.top_divider)
        append(self.top_right)
        return "".join(parts)

    def get_row(
        self,
        widths: Iterable[int],
        level: Literal["head", "row", "foot", "mid"] = "row",
        edge: bool = True,
    ) -> str:
        """Get the top of a simple box.

        Args:
            width (List[int]): Widths of columns.

        Returns:
            str: A string of box characters.
        """
        if level == "head":
            left = self.head_row_left
            horizontal = self.head_row_horizontal
            cross = self.head_row_cross
            right = self.head_row_right
        elif level == "row":
            left = self.row_left
            horizontal = self.row_horizontal
            cross = self.row_cross
            right = self.row_right
        elif level == "mid":
            left = self.mid_left
            horizontal = " "
            cross = self.mid_vertical
            right = self.mid_right
        elif level == "foot":
            left = self.foot_row_left
            horizontal = self.foot_row_horizontal
            cross = self.foot_row_cross
            right = self.foot_row_right
        else:
            raise ValueError("level must be 'head', 'row' or 'foot'")

        parts: List[str] = []
        append = parts.append
        if edge:
            append(left)
        for last, width in loop_last(widths):
            append(horizontal * width)
            if not last:
                append(cross)
        if edge:
            append(right)
        return "".join(parts)

    def get_bottom(self, widths: Iterable[int]) -> str:
        """Get the bottom of a simple box.

        Args:
            widths (List[int]): Widths of columns.

        Returns:
            str: A string of box characters.
        """

        parts: List[str] = []
        append = parts.append
        append(self.bottom_left)
        for last, width in loop_last(widths):
            append(self.bottom * width)
            if not last:
                append(self.bottom_divider)
        append(self.bottom_right)
        return "".join(parts)


# fmt: off
ASCII: Box = Box(
    "+--+\n"
    "| ||\n"
    "|-+|\n"
    "| ||\n"
    "|-+|\n"
    "|-+|\n"
    "| ||\n"
    "+--+\n",
    ascii=True,
)

ASCII2: Box = Box(
    "+-++\n"
    "| ||\n"
    "+-++\n"
    "| ||\n"
    "+-++\n"
    "+-++\n"
    "| ||\n"
    "+-++\n",
    ascii=True,
)

ASCII_DOUBLE_HEAD: Box = Box(
    "+-++\n"
    "| ||\n"
    "+=++\n"
    "| ||\n"
    "+-++\n"
    "+-++\n"
    "| ||\n"
    "+-++\n",
    ascii=True,
)

SQUARE: Box = Box(
    "┌─┬┐\n"
    "│ ││\n"
    "├─┼┤\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "└─┴┘\n"
)

SQUARE_DOUBLE_HEAD: Box = Box(
    "┌─┬┐\n"
    "│ ││\n"
    "╞═╪╡\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "└─┴┘\n"
)

MINIMAL: Box = Box(
    "  ╷ \n"
    "  │ \n"
    "╶─┼╴\n"
    "  │ \n"
    "╶─┼╴\n"
    "╶─┼╴\n"
    "  │ \n"
    "  ╵ \n"
)


MINIMAL_HEAVY_HEAD: Box = Box(
    "  ╷ \n"
    "  │ \n"
    "╺━┿╸\n"
    "  │ \n"
    "╶─┼╴\n"
    "╶─┼╴\n"
    "  │ \n"
    "  ╵ \n"
)

MINIMAL_DOUBLE_HEAD: Box = Box(
    "  ╷ \n"
    "  │ \n"
    " ═╪ \n"
    "  │ \n"
    " ─┼ \n"
    " ─┼ \n"
    "  │ \n"
    "  ╵ \n"
)


SIMPLE: Box = Box(
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n"
)

SIMPLE_HEAD: Box = Box(
    "    \n"
    "    \n"
    " ── \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
    "    \n"
)


SIMPLE_HEAVY: Box = Box(
    "    \n"
    "    \n"
    " ━━ \n"
    "    \n"
    "    \n"
    " ━━ \n"
    "    \n"
    "    \n"
)


HORIZONTALS: Box = Box(
    " ── \n"
    "    \n"
    " ── \n"
    "    \n"
    " ── \n"
    " ── \n"
    "    \n"
    " ── \n"
)

ROUNDED: Box = Box(
    "╭─┬╮\n"
    "│ ││\n"
    "├─┼┤\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "╰─┴╯\n"
)

HEAVY: Box = Box(
    "┏━┳┓\n"
    "┃ ┃┃\n"
    "┣━╋┫\n"
    "┃ ┃┃\n"
    "┣━╋┫\n"
    "┣━╋┫\n"
    "┃ ┃┃\n"
    "┗━┻┛\n"
)

HEAVY_EDGE: Box = Box(
    "┏━┯┓\n"
    "┃ │┃\n"
    "┠─┼┨\n"
    "┃ │┃\n"
    "┠─┼┨\n"
    "┠─┼┨\n"
    "┃ │┃\n"
    "┗━┷┛\n"
)

HEAVY_HEAD: Box = Box(
    "┏━┳┓\n"
    "┃ ┃┃\n"
    "┡━╇┩\n"
    "│ ││\n"
    "├─┼┤\n"
    "├─┼┤\n"
    "│ ││\n"
    "└─┴┘\n"
)

DOUBLE: Box = Box(
    "╔═╦╗\n"
    "║ ║║\n"
    "╠═╬╣\n"
    "║ ║║\n"
    "╠═╬╣\n"
    "╠═╬╣\n"
    "║ ║║\n"
    "╚═╩╝\n"
)

DOUBLE_EDGE: Box = Box(
    "╔═╤╗\n"
    "║ │║\n"
    "╟─┼╢\n"
    "║ │║\n"
    "╟─┼╢\n"
    "╟─┼╢\n"
    "║ │║\n"
    "╚═╧╝\n"
)

MARKDOWN: Box = Box(
    "    \n"
    "| ||\n"
    "|-||\n"
    "| ||\n"
    "|-||\n"
    "|-||\n"
    "| ||\n"
    "    \n",
    ascii=True,
)
# fmt: on

# Map Boxes that don't render with raster fonts on to equivalent that do
LEGACY_WINDOWS_SUBSTITUTIONS = {
    ROUNDED: SQUARE,
    MINIMAL_HEAVY_HEAD: MINIMAL,
    SIMPLE_HEAVY: SIMPLE,
    HEAVY: SQUARE,
    HEAVY_EDGE: SQUARE,
    HEAVY_HEAD: SQUARE,
}

# Map headed boxes to their headerless equivalents
PLAIN_HEADED_SUBSTITUTIONS = {
    HEAVY_HEAD: SQUARE,
    SQUARE_DOUBLE_HEAD: SQUARE,
    MINIMAL_DOUBLE_HEAD: MINIMAL,
    MINIMAL_HEAVY_HEAD: MINIMAL,
    ASCII_DOUBLE_HEAD: ASCII2,
}


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich.columns import Columns
    from pip._vendor.rich.panel import Panel

    from . import box as box
    from .console import Console
    from .table import Table
    from .text import Text

    console = Console(record=True)

    BOXES = [
        "ASCII",
        "ASCII2",
        "ASCII_DOUBLE_HEAD",
        "SQUARE",
        "SQUARE_DOUBLE_HEAD",
        "MINIMAL",
        "MINIMAL_HEAVY_HEAD",
        "MINIMAL_DOUBLE_HEAD",
        "SIMPLE",
        "SIMPLE_HEAD",
        "SIMPLE_HEAVY",
        "HORIZONTALS",
        "ROUNDED",
        "HEAVY",
        "HEAVY_EDGE",
        "HEAVY_HEAD",
        "DOUBLE",
        "DOUBLE_EDGE",
        "MARKDOWN",
    ]

    console.print(Panel("[bold green]Box Constants", style="green"), justify="center")
    console.print()

    columns = Columns(expand=True, padding=2)
    for box_name in sorted(BOXES):
        table = Table(
            show_footer=True, style="dim", border_style="not dim", expand=True
        )
        table.add_column("Header 1", "Footer 1")
        table.add_column("Header 2", "Footer 2")
        table.add_row("Cell", "Cell")
        table.add_row("Cell", "Cell")
        table.box = getattr(box, box_name)
        table.title = Text(f"box.{box_name}", style="magenta")
        columns.add_renderable(table)
    console.print(columns)

    # console.save_svg("box.svg")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\_collections.py ===
from __future__ import annotations

import typing
from collections import OrderedDict
from enum import Enum, auto
from threading import RLock

if typing.TYPE_CHECKING:
    # We can only import Protocol if TYPE_CHECKING because it's a development
    # dependency, and is not available at runtime.
    from typing import Protocol

    from typing_extensions import Self

    class HasGettableStringKeys(Protocol):
        def keys(self) -> typing.Iterator[str]: ...

        def __getitem__(self, key: str) -> str: ...


__all__ = ["RecentlyUsedContainer", "HTTPHeaderDict"]


# Key type
_KT = typing.TypeVar("_KT")
# Value type
_VT = typing.TypeVar("_VT")
# Default type
_DT = typing.TypeVar("_DT")

ValidHTTPHeaderSource = typing.Union[
    "HTTPHeaderDict",
    typing.Mapping[str, str],
    typing.Iterable[tuple[str, str]],
    "HasGettableStringKeys",
]


class _Sentinel(Enum):
    not_passed = auto()


def ensure_can_construct_http_header_dict(
    potential: object,
) -> ValidHTTPHeaderSource | None:
    if isinstance(potential, HTTPHeaderDict):
        return potential
    elif isinstance(potential, typing.Mapping):
        # Full runtime checking of the contents of a Mapping is expensive, so for the
        # purposes of typechecking, we assume that any Mapping is the right shape.
        return typing.cast(typing.Mapping[str, str], potential)
    elif isinstance(potential, typing.Iterable):
        # Similarly to Mapping, full runtime checking of the contents of an Iterable is
        # expensive, so for the purposes of typechecking, we assume that any Iterable
        # is the right shape.
        return typing.cast(typing.Iterable[tuple[str, str]], potential)
    elif hasattr(potential, "keys") and hasattr(potential, "__getitem__"):
        return typing.cast("HasGettableStringKeys", potential)
    else:
        return None


class RecentlyUsedContainer(typing.Generic[_KT, _VT], typing.MutableMapping[_KT, _VT]):
    """
    Provides a thread-safe dict-like container which maintains up to
    ``maxsize`` keys while throwing away the least-recently-used keys beyond
    ``maxsize``.

    :param maxsize:
        Maximum number of recent elements to retain.

    :param dispose_func:
        Every time an item is evicted from the container,
        ``dispose_func(value)`` is called.  Callback which will get called
    """

    _container: typing.OrderedDict[_KT, _VT]
    _maxsize: int
    dispose_func: typing.Callable[[_VT], None] | None
    lock: RLock

    def __init__(
        self,
        maxsize: int = 10,
        dispose_func: typing.Callable[[_VT], None] | None = None,
    ) -> None:
        super().__init__()
        self._maxsize = maxsize
        self.dispose_func = dispose_func
        self._container = OrderedDict()
        self.lock = RLock()

    def __getitem__(self, key: _KT) -> _VT:
        # Re-insert the item, moving it to the end of the eviction line.
        with self.lock:
            item = self._container.pop(key)
            self._container[key] = item
            return item

    def __setitem__(self, key: _KT, value: _VT) -> None:
        evicted_item = None
        with self.lock:
            # Possibly evict the existing value of 'key'
            try:
                # If the key exists, we'll overwrite it, which won't change the
                # size of the pool. Because accessing a key should move it to
                # the end of the eviction line, we pop it out first.
                evicted_item = key, self._container.pop(key)
                self._container[key] = value
            except KeyError:
                # When the key does not exist, we insert the value first so that
                # evicting works in all cases, including when self._maxsize is 0
                self._container[key] = value
                if len(self._container) > self._maxsize:
                    # If we didn't evict an existing value, and we've hit our maximum
                    # size, then we have to evict the least recently used item from
                    # the beginning of the container.
                    evicted_item = self._container.popitem(last=False)

        # After releasing the lock on the pool, dispose of any evicted value.
        if evicted_item is not None and self.dispose_func:
            _, evicted_value = evicted_item
            self.dispose_func(evicted_value)

    def __delitem__(self, key: _KT) -> None:
        with self.lock:
            value = self._container.pop(key)

        if self.dispose_func:
            self.dispose_func(value)

    def __len__(self) -> int:
        with self.lock:
            return len(self._container)

    def __iter__(self) -> typing.NoReturn:
        raise NotImplementedError(
            "Iteration over this class is unlikely to be threadsafe."
        )

    def clear(self) -> None:
        with self.lock:
            # Copy pointers to all values, then wipe the mapping
            values = list(self._container.values())
            self._container.clear()

        if self.dispose_func:
            for value in values:
                self.dispose_func(value)

    def keys(self) -> set[_KT]:  # type: ignore[override]
        with self.lock:
            return set(self._container.keys())


class HTTPHeaderDictItemView(set[tuple[str, str]]):
    """
    HTTPHeaderDict is unusual for a Mapping[str, str] in that it has two modes of
    address.

    If we directly try to get an item with a particular name, we will get a string
    back that is the concatenated version of all the values:

    >>> d['X-Header-Name']
    'Value1, Value2, Value3'

    However, if we iterate over an HTTPHeaderDict's items, we will optionally combine
    these values based on whether combine=True was called when building up the dictionary

    >>> d = HTTPHeaderDict({"A": "1", "B": "foo"})
    >>> d.add("A", "2", combine=True)
    >>> d.add("B", "bar")
    >>> list(d.items())
    [
        ('A', '1, 2'),
        ('B', 'foo'),
        ('B', 'bar'),
    ]

    This class conforms to the interface required by the MutableMapping ABC while
    also giving us the nonstandard iteration behavior we want; items with duplicate
    keys, ordered by time of first insertion.
    """

    _headers: HTTPHeaderDict

    def __init__(self, headers: HTTPHeaderDict) -> None:
        self._headers = headers

    def __len__(self) -> int:
        return len(list(self._headers.iteritems()))

    def __iter__(self) -> typing.Iterator[tuple[str, str]]:
        return self._headers.iteritems()

    def __contains__(self, item: object) -> bool:
        if isinstance(item, tuple) and len(item) == 2:
            passed_key, passed_val = item
            if isinstance(passed_key, str) and isinstance(passed_val, str):
                return self._headers._has_value_for_header(passed_key, passed_val)
        return False


class HTTPHeaderDict(typing.MutableMapping[str, str]):
    """
    :param headers:
        An iterable of field-value pairs. Must not contain multiple field names
        when compared case-insensitively.

    :param kwargs:
        Additional field-value pairs to pass in to ``dict.update``.

    A ``dict`` like container for storing HTTP Headers.

    Field names are stored and compared case-insensitively in compliance with
    RFC 7230. Iteration provides the first case-sensitive key seen for each
    case-insensitive pair.

    Using ``__setitem__`` syntax overwrites fields that compare equal
    case-insensitively in order to maintain ``dict``'s api. For fields that
    compare equal, instead create a new ``HTTPHeaderDict`` and use ``.add``
    in a loop.

    If multiple fields that are equal case-insensitively are passed to the
    constructor or ``.update``, the behavior is undefined and some will be
    lost.

    >>> headers = HTTPHeaderDict()
    >>> headers.add('Set-Cookie', 'foo=bar')
    >>> headers.add('set-cookie', 'baz=quxx')
    >>> headers['content-length'] = '7'
    >>> headers['SET-cookie']
    'foo=bar, baz=quxx'
    >>> headers['Content-Length']
    '7'
    """

    _container: typing.MutableMapping[str, list[str]]

    def __init__(self, headers: ValidHTTPHeaderSource | None = None, **kwargs: str):
        super().__init__()
        self._container = {}  # 'dict' is insert-ordered
        if headers is not None:
            if isinstance(headers, HTTPHeaderDict):
                self._copy_from(headers)
            else:
                self.extend(headers)
        if kwargs:
            self.extend(kwargs)

    def __setitem__(self, key: str, val: str) -> None:
        # avoid a bytes/str comparison by decoding before httplib
        if isinstance(key, bytes):
            key = key.decode("latin-1")
        self._container[key.lower()] = [key, val]

    def __getitem__(self, key: str) -> str:
        val = self._container[key.lower()]
        return ", ".join(val[1:])

    def __delitem__(self, key: str) -> None:
        del self._container[key.lower()]

    def __contains__(self, key: object) -> bool:
        if isinstance(key, str):
            return key.lower() in self._container
        return False

    def setdefault(self, key: str, default: str = "") -> str:
        return super().setdefault(key, default)

    def __eq__(self, other: object) -> bool:
        maybe_constructable = ensure_can_construct_http_header_dict(other)
        if maybe_constructable is None:
            return False
        else:
            other_as_http_header_dict = type(self)(maybe_constructable)

        return {k.lower(): v for k, v in self.itermerged()} == {
            k.lower(): v for k, v in other_as_http_header_dict.itermerged()
        }

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __len__(self) -> int:
        return len(self._container)

    def __iter__(self) -> typing.Iterator[str]:
        # Only provide the originally cased names
        for vals in self._container.values():
            yield vals[0]

    def discard(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            pass

    def add(self, key: str, val: str, *, combine: bool = False) -> None:
        """Adds a (name, value) pair, doesn't overwrite the value if it already
        exists.

        If this is called with combine=True, instead of adding a new header value
        as a distinct item during iteration, this will instead append the value to
        any existing header value with a comma. If no existing header value exists
        for the key, then the value will simply be added, ignoring the combine parameter.

        >>> headers = HTTPHeaderDict(foo='bar')
        >>> headers.add('Foo', 'baz')
        >>> headers['foo']
        'bar, baz'
        >>> list(headers.items())
        [('foo', 'bar'), ('foo', 'baz')]
        >>> headers.add('foo', 'quz', combine=True)
        >>> list(headers.items())
        [('foo', 'bar, baz, quz')]
        """
        # avoid a bytes/str comparison by decoding before httplib
        if isinstance(key, bytes):
            key = key.decode("latin-1")
        key_lower = key.lower()
        new_vals = [key, val]
        # Keep the common case aka no item present as fast as possible
        vals = self._container.setdefault(key_lower, new_vals)
        if new_vals is not vals:
            # if there are values here, then there is at least the initial
            # key/value pair
            assert len(vals) >= 2
            if combine:
                vals[-1] = vals[-1] + ", " + val
            else:
                vals.append(val)

    def extend(self, *args: ValidHTTPHeaderSource, **kwargs: str) -> None:
        """Generic import function for any type of header-like object.
        Adapted version of MutableMapping.update in order to insert items
        with self.add instead of self.__setitem__
        """
        if len(args) > 1:
            raise TypeError(
                f"extend() takes at most 1 positional arguments ({len(args)} given)"
            )
        other = args[0] if len(args) >= 1 else ()

        if isinstance(other, HTTPHeaderDict):
            for key, val in other.iteritems():
                self.add(key, val)
        elif isinstance(other, typing.Mapping):
            for key, val in other.items():
                self.add(key, val)
        elif isinstance(other, typing.Iterable):
            other = typing.cast(typing.Iterable[tuple[str, str]], other)
            for key, value in other:
                self.add(key, value)
        elif hasattr(other, "keys") and hasattr(other, "__getitem__"):
            # THIS IS NOT A TYPESAFE BRANCH
            # In this branch, the object has a `keys` attr but is not a Mapping or any of
            # the other types indicated in the method signature. We do some stuff with
            # it as though it partially implements the Mapping interface, but we're not
            # doing that stuff safely AT ALL.
            for key in other.keys():
                self.add(key, other[key])

        for key, value in kwargs.items():
            self.add(key, value)

    @typing.overload
    def getlist(self, key: str) -> list[str]: ...

    @typing.overload
    def getlist(self, key: str, default: _DT) -> list[str] | _DT: ...

    def getlist(
        self, key: str, default: _Sentinel | _DT = _Sentinel.not_passed
    ) -> list[str] | _DT:
        """Returns a list of all the values for the named field. Returns an
        empty list if the key doesn't exist."""
        try:
            vals = self._container[key.lower()]
        except KeyError:
            if default is _Sentinel.not_passed:
                # _DT is unbound; empty list is instance of List[str]
                return []
            # _DT is bound; default is instance of _DT
            return default
        else:
            # _DT may or may not be bound; vals[1:] is instance of List[str], which
            # meets our external interface requirement of `Union[List[str], _DT]`.
            return vals[1:]

    def _prepare_for_method_change(self) -> Self:
        """
        Remove content-specific header fields before changing the request
        method to GET or HEAD according to RFC 9110, Section 15.4.
        """
        content_specific_headers = [
            "Content-Encoding",
            "Content-Language",
            "Content-Location",
            "Content-Type",
            "Content-Length",
            "Digest",
            "Last-Modified",
        ]
        for header in content_specific_headers:
            self.discard(header)
        return self

    # Backwards compatibility for httplib
    getheaders = getlist
    getallmatchingheaders = getlist
    iget = getlist

    # Backwards compatibility for http.cookiejar
    get_all = getlist

    def __repr__(self) -> str:
        return f"{type(self).__name__}({dict(self.itermerged())})"

    def _copy_from(self, other: HTTPHeaderDict) -> None:
        for key in other:
            val = other.getlist(key)
            self._container[key.lower()] = [key, *val]

    def copy(self) -> Self:
        clone = type(self)()
        clone._copy_from(self)
        return clone

    def iteritems(self) -> typing.Iterator[tuple[str, str]]:
        """Iterate over all header lines, including duplicate ones."""
        for key in self:
            vals = self._container[key.lower()]
            for val in vals[1:]:
                yield vals[0], val

    def itermerged(self) -> typing.Iterator[tuple[str, str]]:
        """Iterate over all headers, merging duplicate ones together."""
        for key in self:
            val = self._container[key.lower()]
            yield val[0], ", ".join(val[1:])

    def items(self) -> HTTPHeaderDictItemView:  # type: ignore[override]
        return HTTPHeaderDictItemView(self)

    def _has_value_for_header(self, header_name: str, potential_value: str) -> bool:
        if header_name in self:
            return potential_value in self._container[header_name.lower()][1:]
        return False

    def __ior__(self, other: object) -> HTTPHeaderDict:
        # Supports extending a header dict in-place using operator |=
        # combining items with add instead of __setitem__
        maybe_constructable = ensure_can_construct_http_header_dict(other)
        if maybe_constructable is None:
            return NotImplemented
        self.extend(maybe_constructable)
        return self

    def __or__(self, other: object) -> Self:
        # Supports merging header dicts using operator |
        # combining items with add instead of __setitem__
        maybe_constructable = ensure_can_construct_http_header_dict(other)
        if maybe_constructable is None:
            return NotImplemented
        result = self.copy()
        result.extend(maybe_constructable)
        return result

    def __ror__(self, other: object) -> Self:
        # Supports merging header dicts using operator | when other is on left side
        # combining items with add instead of __setitem__
        maybe_constructable = ensure_can_construct_http_header_dict(other)
        if maybe_constructable is None:
            return NotImplemented
        result = type(self)(maybe_constructable)
        result.extend(self)
        return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\horizontal_shard.py ===
# ext/horizontal_shard.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Horizontal sharding support.

Defines a rudimental 'horizontal sharding' system which allows a Session to
distribute queries and persistence operations across multiple databases.

For a usage example, see the :ref:`examples_sharding` example included in
the source distribution.

.. deepalchemy:: The horizontal sharding extension is an advanced feature,
   involving a complex statement -> database interaction as well as
   use of semi-public APIs for non-trivial cases.   Simpler approaches to
   refering to multiple database "shards", most commonly using a distinct
   :class:`_orm.Session` per "shard", should always be considered first
   before using this more complex and less-production-tested system.



"""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .. import event
from .. import exc
from .. import inspect
from .. import util
from ..orm import PassiveFlag
from ..orm._typing import OrmExecuteOptionsParameter
from ..orm.interfaces import ORMOption
from ..orm.mapper import Mapper
from ..orm.query import Query
from ..orm.session import _BindArguments
from ..orm.session import _PKIdentityArgument
from ..orm.session import Session
from ..util.typing import Protocol
from ..util.typing import Self

if TYPE_CHECKING:
    from ..engine.base import Connection
    from ..engine.base import Engine
    from ..engine.base import OptionEngine
    from ..engine.result import IteratorResult
    from ..engine.result import Result
    from ..orm import LoaderCallableStatus
    from ..orm._typing import _O
    from ..orm.bulk_persistence import BulkUDCompileState
    from ..orm.context import QueryContext
    from ..orm.session import _EntityBindKey
    from ..orm.session import _SessionBind
    from ..orm.session import ORMExecuteState
    from ..orm.state import InstanceState
    from ..sql import Executable
    from ..sql._typing import _TP
    from ..sql.elements import ClauseElement

__all__ = ["ShardedSession", "ShardedQuery"]

_T = TypeVar("_T", bound=Any)


ShardIdentifier = str


class ShardChooser(Protocol):
    def __call__(
        self,
        mapper: Optional[Mapper[_T]],
        instance: Any,
        clause: Optional[ClauseElement],
    ) -> Any: ...


class IdentityChooser(Protocol):
    def __call__(
        self,
        mapper: Mapper[_T],
        primary_key: _PKIdentityArgument,
        *,
        lazy_loaded_from: Optional[InstanceState[Any]],
        execution_options: OrmExecuteOptionsParameter,
        bind_arguments: _BindArguments,
        **kw: Any,
    ) -> Any: ...


class ShardedQuery(Query[_T]):
    """Query class used with :class:`.ShardedSession`.

    .. legacy:: The :class:`.ShardedQuery` is a subclass of the legacy
       :class:`.Query` class.   The :class:`.ShardedSession` now supports
       2.0 style execution via the :meth:`.ShardedSession.execute` method.

    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert isinstance(self.session, ShardedSession)

        self.identity_chooser = self.session.identity_chooser
        self.execute_chooser = self.session.execute_chooser
        self._shard_id = None

    def set_shard(self, shard_id: ShardIdentifier) -> Self:
        """Return a new query, limited to a single shard ID.

        All subsequent operations with the returned query will
        be against the single shard regardless of other state.

        The shard_id can be passed for a 2.0 style execution to the
        bind_arguments dictionary of :meth:`.Session.execute`::

            results = session.execute(stmt, bind_arguments={"shard_id": "my_shard"})

        """  # noqa: E501
        return self.execution_options(_sa_shard_id=shard_id)


class ShardedSession(Session):
    shard_chooser: ShardChooser
    identity_chooser: IdentityChooser
    execute_chooser: Callable[[ORMExecuteState], Iterable[Any]]

    def __init__(
        self,
        shard_chooser: ShardChooser,
        identity_chooser: Optional[IdentityChooser] = None,
        execute_chooser: Optional[
            Callable[[ORMExecuteState], Iterable[Any]]
        ] = None,
        shards: Optional[Dict[str, Any]] = None,
        query_cls: Type[Query[_T]] = ShardedQuery,
        *,
        id_chooser: Optional[
            Callable[[Query[_T], Iterable[_T]], Iterable[Any]]
        ] = None,
        query_chooser: Optional[Callable[[Executable], Iterable[Any]]] = None,
        **kwargs: Any,
    ) -> None:
        """Construct a ShardedSession.

        :param shard_chooser: A callable which, passed a Mapper, a mapped
          instance, and possibly a SQL clause, returns a shard ID.  This id
          may be based off of the attributes present within the object, or on
          some round-robin scheme. If the scheme is based on a selection, it
          should set whatever state on the instance to mark it in the future as
          participating in that shard.

        :param identity_chooser: A callable, passed a Mapper and primary key
         argument, which should return a list of shard ids where this
         primary key might reside.

          .. versionchanged:: 2.0  The ``identity_chooser`` parameter
             supersedes the ``id_chooser`` parameter.

        :param execute_chooser: For a given :class:`.ORMExecuteState`,
          returns the list of shard_ids
          where the query should be issued.  Results from all shards returned
          will be combined together into a single listing.

          .. versionchanged:: 1.4  The ``execute_chooser`` parameter
             supersedes the ``query_chooser`` parameter.

        :param shards: A dictionary of string shard names
          to :class:`~sqlalchemy.engine.Engine` objects.

        """
        super().__init__(query_cls=query_cls, **kwargs)

        event.listen(
            self, "do_orm_execute", execute_and_instances, retval=True
        )
        self.shard_chooser = shard_chooser

        if id_chooser:
            _id_chooser = id_chooser
            util.warn_deprecated(
                "The ``id_chooser`` parameter is deprecated; "
                "please use ``identity_chooser``.",
                "2.0",
            )

            def _legacy_identity_chooser(
                mapper: Mapper[_T],
                primary_key: _PKIdentityArgument,
                *,
                lazy_loaded_from: Optional[InstanceState[Any]],
                execution_options: OrmExecuteOptionsParameter,
                bind_arguments: _BindArguments,
                **kw: Any,
            ) -> Any:
                q = self.query(mapper)
                if lazy_loaded_from:
                    q = q._set_lazyload_from(lazy_loaded_from)
                return _id_chooser(q, primary_key)

            self.identity_chooser = _legacy_identity_chooser
        elif identity_chooser:
            self.identity_chooser = identity_chooser
        else:
            raise exc.ArgumentError(
                "identity_chooser or id_chooser is required"
            )

        if query_chooser:
            _query_chooser = query_chooser
            util.warn_deprecated(
                "The ``query_chooser`` parameter is deprecated; "
                "please use ``execute_chooser``.",
                "1.4",
            )
            if execute_chooser:
                raise exc.ArgumentError(
                    "Can't pass query_chooser and execute_chooser "
                    "at the same time."
                )

            def _default_execute_chooser(
                orm_context: ORMExecuteState,
            ) -> Iterable[Any]:
                return _query_chooser(orm_context.statement)

            if execute_chooser is None:
                execute_chooser = _default_execute_chooser

        if execute_chooser is None:
            raise exc.ArgumentError(
                "execute_chooser or query_chooser is required"
            )
        self.execute_chooser = execute_chooser
        self.__shards: Dict[ShardIdentifier, _SessionBind] = {}
        if shards is not None:
            for k in shards:
                self.bind_shard(k, shards[k])

    def _identity_lookup(
        self,
        mapper: Mapper[_O],
        primary_key_identity: Union[Any, Tuple[Any, ...]],
        identity_token: Optional[Any] = None,
        passive: PassiveFlag = PassiveFlag.PASSIVE_OFF,
        lazy_loaded_from: Optional[InstanceState[Any]] = None,
        execution_options: OrmExecuteOptionsParameter = util.EMPTY_DICT,
        bind_arguments: Optional[_BindArguments] = None,
        **kw: Any,
    ) -> Union[Optional[_O], LoaderCallableStatus]:
        """override the default :meth:`.Session._identity_lookup` method so
        that we search for a given non-token primary key identity across all
        possible identity tokens (e.g. shard ids).

        .. versionchanged:: 1.4  Moved :meth:`.Session._identity_lookup` from
           the :class:`_query.Query` object to the :class:`.Session`.

        """

        if identity_token is not None:
            obj = super()._identity_lookup(
                mapper,
                primary_key_identity,
                identity_token=identity_token,
                **kw,
            )

            return obj
        else:
            for shard_id in self.identity_chooser(
                mapper,
                primary_key_identity,
                lazy_loaded_from=lazy_loaded_from,
                execution_options=execution_options,
                bind_arguments=dict(bind_arguments) if bind_arguments else {},
            ):
                obj2 = super()._identity_lookup(
                    mapper,
                    primary_key_identity,
                    identity_token=shard_id,
                    lazy_loaded_from=lazy_loaded_from,
                    **kw,
                )
                if obj2 is not None:
                    return obj2

            return None

    def _choose_shard_and_assign(
        self,
        mapper: Optional[_EntityBindKey[_O]],
        instance: Any,
        **kw: Any,
    ) -> Any:
        if instance is not None:
            state = inspect(instance)
            if state.key:
                token = state.key[2]
                assert token is not None
                return token
            elif state.identity_token:
                return state.identity_token

        assert isinstance(mapper, Mapper)
        shard_id = self.shard_chooser(mapper, instance, **kw)
        if instance is not None:
            state.identity_token = shard_id
        return shard_id

    def connection_callable(
        self,
        mapper: Optional[Mapper[_T]] = None,
        instance: Optional[Any] = None,
        shard_id: Optional[ShardIdentifier] = None,
        **kw: Any,
    ) -> Connection:
        """Provide a :class:`_engine.Connection` to use in the unit of work
        flush process.

        """

        if shard_id is None:
            shard_id = self._choose_shard_and_assign(mapper, instance)

        if self.in_transaction():
            trans = self.get_transaction()
            assert trans is not None
            return trans.connection(mapper, shard_id=shard_id)
        else:
            bind = self.get_bind(
                mapper=mapper, shard_id=shard_id, instance=instance
            )

            if isinstance(bind, Engine):
                return bind.connect(**kw)
            else:
                assert isinstance(bind, Connection)
                return bind

    def get_bind(
        self,
        mapper: Optional[_EntityBindKey[_O]] = None,
        *,
        shard_id: Optional[ShardIdentifier] = None,
        instance: Optional[Any] = None,
        clause: Optional[ClauseElement] = None,
        **kw: Any,
    ) -> _SessionBind:
        if shard_id is None:
            shard_id = self._choose_shard_and_assign(
                mapper, instance=instance, clause=clause
            )
            assert shard_id is not None
        return self.__shards[shard_id]

    def bind_shard(
        self, shard_id: ShardIdentifier, bind: Union[Engine, OptionEngine]
    ) -> None:
        self.__shards[shard_id] = bind


class set_shard_id(ORMOption):
    """a loader option for statements to apply a specific shard id to the
    primary query as well as for additional relationship and column
    loaders.

    The :class:`_horizontal.set_shard_id` option may be applied using
    the :meth:`_sql.Executable.options` method of any executable statement::

        stmt = (
            select(MyObject)
            .where(MyObject.name == "some name")
            .options(set_shard_id("shard1"))
        )

    Above, the statement when invoked will limit to the "shard1" shard
    identifier for the primary query as well as for all relationship and
    column loading strategies, including eager loaders such as
    :func:`_orm.selectinload`, deferred column loaders like :func:`_orm.defer`,
    and the lazy relationship loader :func:`_orm.lazyload`.

    In this way, the :class:`_horizontal.set_shard_id` option has much wider
    scope than using the "shard_id" argument within the
    :paramref:`_orm.Session.execute.bind_arguments` dictionary.


    .. versionadded:: 2.0.0

    """

    __slots__ = ("shard_id", "propagate_to_loaders")

    def __init__(
        self, shard_id: ShardIdentifier, propagate_to_loaders: bool = True
    ):
        """Construct a :class:`_horizontal.set_shard_id` option.

        :param shard_id: shard identifier
        :param propagate_to_loaders: if left at its default of ``True``, the
         shard option will take place for lazy loaders such as
         :func:`_orm.lazyload` and :func:`_orm.defer`; if False, the option
         will not be propagated to loaded objects. Note that :func:`_orm.defer`
         always limits to the shard_id of the parent row in any case, so the
         parameter only has a net effect on the behavior of the
         :func:`_orm.lazyload` strategy.

        """
        self.shard_id = shard_id
        self.propagate_to_loaders = propagate_to_loaders


def execute_and_instances(
    orm_context: ORMExecuteState,
) -> Union[Result[_T], IteratorResult[_TP]]:
    active_options: Union[
        None,
        QueryContext.default_load_options,
        Type[QueryContext.default_load_options],
        BulkUDCompileState.default_update_options,
        Type[BulkUDCompileState.default_update_options],
    ]

    if orm_context.is_select:
        active_options = orm_context.load_options

    elif orm_context.is_update or orm_context.is_delete:
        active_options = orm_context.update_delete_options
    else:
        active_options = None

    session = orm_context.session
    assert isinstance(session, ShardedSession)

    def iter_for_shard(
        shard_id: ShardIdentifier,
    ) -> Union[Result[_T], IteratorResult[_TP]]:
        bind_arguments = dict(orm_context.bind_arguments)
        bind_arguments["shard_id"] = shard_id

        orm_context.update_execution_options(identity_token=shard_id)
        return orm_context.invoke_statement(bind_arguments=bind_arguments)

    for orm_opt in orm_context._non_compile_orm_options:
        # TODO: if we had an ORMOption that gets applied at ORM statement
        # execution time, that would allow this to be more generalized.
        # for now just iterate and look for our options
        if isinstance(orm_opt, set_shard_id):
            shard_id = orm_opt.shard_id
            break
    else:
        if active_options and active_options._identity_token is not None:
            shard_id = active_options._identity_token
        elif "_sa_shard_id" in orm_context.execution_options:
            shard_id = orm_context.execution_options["_sa_shard_id"]
        elif "shard_id" in orm_context.bind_arguments:
            shard_id = orm_context.bind_arguments["shard_id"]
        else:
            shard_id = None

    if shard_id is not None:
        return iter_for_shard(shard_id)
    else:
        partial = []
        for shard_id in session.execute_chooser(orm_context):
            result_ = iter_for_shard(shard_id)
            partial.append(result_)
        return partial[0].merge(*partial[1:])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\whisper\whisper_jump_times.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import tempfile
import textwrap
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F
import torch.utils.cpp_extension
from onnx_model import OnnxModel
from transformers import WhisperConfig
from whisper_inputs import convert_inputs_for_ort, get_model_dynamic_axes, get_sample_jump_times_inputs

from onnxruntime import InferenceSession
from onnxruntime.tools import pytorch_export_contrib_ops

logger = logging.getLogger(__name__)

##################################################
# Functions that have to be outside of the class
# for torch.jit.script_if_tracing to work
##################################################


@torch.jit.script_if_tracing
def index_QKs(alignment_heads: torch.Tensor, QKs: list[torch.Tensor]):  # noqa: N802
    """
    Compute the following to get stacked QK tensor that has been indexed for the desired attention heads:
    weights = torch.stack([QKs[_l][:, _h] for _l, _h in alignment_heads], dim=1)
    """
    indexed_QKs = []  # noqa: N806
    for pair in alignment_heads:
        # Each QK is of shape (batch_size, num_heads, sequence_length, num_frames // 2)
        # The `QKs[_l]` selects the right QK from the list of QKs
        # The `QKs[_l][:, _h]` selects the right attention heads from the chosen QK. The `:` is to do this for the batch dim.
        #
        # PyTorch:
        # QKs[_l] is of shape (batch_size, num_heads, sequence_length, num_frames // 2)
        # QKs[_l][:, _h] is of shape (batch_size, sequence_length, num_frames // 2)
        #
        # ONNX:
        # QKs[_l] is of shape (batch_size, num_heads, sequence_length, num_frames // 2)
        # QKs[_l][:, _h] is of shape (batch_size, 1, sequence_length, num_frames // 2) because
        # the `[:, _h]` operation maps to a Gather op and that op does not reduce dimensions
        _l, _h = pair[0], pair[1]
        indexed_QKs.append(QKs[_l][:, _h])

    # PyTorch:
    # torch.stack will return a tensor of shape (batch_size, num_alignment_heads, sequence_length, num_frames // 2).
    #
    # ONNX:
    # torch.stack will return a tensor of shape (batch_size, num_alignment_heads, 1, sequence_length, num_frames // 2)
    # because the Gather op does not reduce dimensions. To remove the unneeded dimension, torch.squeeze with a specified
    # dim (dim = 2) is added. The torch.squeeze op with a specified dim only runs if the specified dim has a size of 1.
    # Since the dim won't be of size 1 in the PyTorch tensor but it is of size 1 in the ONNX tensor, it will be a no-op
    # in PyTorch and an op in ONNX. Thus, the Squeeze op will only affect the ONNX model.
    weights = torch.stack(indexed_QKs, dim=1)
    weights = torch.squeeze(weights, dim=2)
    return weights


def jump_timings(text_indices, time_indices):
    """
    Calculate jump times from text_indices and time_indices where
    text_indices and time_indices are both 1d vectors
    """
    TOKENS_PER_SECOND = 50.0  # noqa: N806
    diff = text_indices[1:] - text_indices[:-1]
    padding = torch.tensor([1], dtype=torch.int32)
    jumps = torch.cat((padding, diff)).to(torch.bool)
    jump_times = time_indices[jumps].to(torch.float) / TOKENS_PER_SECOND
    return jump_times


def padded_jump_from_dtw(matrix_2d: torch.Tensor, max_length: torch.Tensor):
    """
    Run Dynamic Time Warping (DTW) on batched tensor
    """
    trace = torch.ops.onnxruntime.DynamicTimeWarping(matrix_2d)
    text_indices = trace[0, :]
    time_indices = trace[1, :]
    jump_times = jump_timings(text_indices, time_indices)
    return F.pad(jump_times, [0, int((max_length - jump_times.size(-1)).item())], mode="constant", value=-1.0)


@torch.jit.script_if_tracing
def batch_jump_times(matrix: torch.Tensor, max_decoded_length: torch.Tensor):
    """
    Compute the following to calculate jump times for all batches:
    batched_jump_times = torch.stack([self.padded_jump_from_dtw(matrix[b], max_decoded_length) for b in range(matrix.size(0))])
    """
    list_of_jump_times = []
    for b in range(matrix.size(0)):
        jump_times = padded_jump_from_dtw(matrix[b], max_decoded_length)
        list_of_jump_times.append(jump_times)
    batched_jump_times = torch.stack(list_of_jump_times)
    return batched_jump_times


class WhisperJumpTimes(torch.nn.Module):
    """Whisper jump times component"""

    def __init__(self, config: WhisperConfig, device: torch.device, cache_dir: str | os.PathLike):
        super().__init__()
        self.config = config
        self.device = device
        self.cache_dir = cache_dir

        self.filter_width = 7
        self.qk_scale = 1.0

    def median_filter(self, weights: torch.Tensor):
        """
        Apply a median filter of width `filter_width` along the last dimension of `weights`
        """
        pad_width = self.filter_width // 2
        x = F.pad(weights, (pad_width, pad_width, 0, 0), mode="reflect")
        x_unfolded = torch.ops.onnxruntime.UnfoldTensor(x, -1, self.filter_width, 1)
        result = torch.select(x_unfolded.sort()[0], dim=-1, index=pad_width)
        return result

    def forward(
        self,
        alignment_heads: torch.Tensor,
        sot_sequence_length: torch.Tensor,
        segment_length: torch.Tensor,
        QKs: list[torch.Tensor],
    ):
        # Get stacked QKs tensor
        weights = index_QKs(alignment_heads, QKs)
        weights = weights[:, :, : segment_length // 2]
        weights = weights.to(torch.float32)

        weights = (weights * self.qk_scale).softmax(dim=-1)
        std, mean = torch.std_mean(weights, dim=-2, keepdim=True, unbiased=False)
        weights = (weights - mean) / std
        weights = self.median_filter(weights)

        matrix = torch.mean(weights, 1)
        matrix = -matrix[:, sot_sequence_length:-1]

        max_decoded_length = torch.tensor([matrix.size(1)], dtype=torch.int64)
        batched_jump_times = batch_jump_times(matrix, max_decoded_length)
        return batched_jump_times

    def input_names(self):
        input_names = [
            "alignment_heads",
            "sot_sequence_length",
            "segment_length",
            *[f"cross_qk_{i}" for i in range(self.config.num_hidden_layers)],
        ]
        return input_names

    def output_names(self):
        output_names = ["jump_times"]
        return output_names

    def inputs(self, use_fp16_inputs: bool, use_int32_inputs: bool, return_dict: bool = False):
        inputs = get_sample_jump_times_inputs(
            self.config,
            self.device,
            batch_size=2,
            sequence_length=8,
            num_alignment_heads=6,
            sot_sequence_length=3,
            segment_length=1332,
            use_fp16=use_fp16_inputs,
            use_int32=use_int32_inputs,
        )
        if return_dict:
            return inputs
        return (
            inputs["alignment_heads"],
            inputs["sot_sequence_length"],
            inputs["segment_length"],
            inputs["QKs"],
        )

    def create_torch_ops(self):
        """
        1) Create UnfoldTensor and DynamicTimeWarping as torch ops
        3) Provide a symbolic mapping from torch ops to ORT contrib ops

        See https://pytorch.org/tutorials/advanced/torch_script_custom_ops.html#building-with-jit-compilation
        for more details on how this works.
        """
        # Set torch extensions directory to cache directory
        os.environ["TORCH_EXTENSIONS_DIR"] = self.cache_dir

        # Try to import `jinja` pip package
        try:
            assert torch.utils.cpp_extension.verify_ninja_availability()
        except Exception as e:
            logger.error(f"An error occurred while verifying `ninja` is available: {e}", exc_info=True)  # noqa: G201
            install_cmd = "pip install ninja"
            logger.warning(f"Could not import `ninja`. Attempting to install `ninja` via `{install_cmd}`.")
            os.system(install_cmd)

        # Create UnfoldTensor torch op
        unfold_op_source = textwrap.dedent("""\
        #include "torch/script.h"

        torch::Tensor UnfoldTensor(torch::Tensor input, int64_t dim, int64_t size, int64_t step) {
          return input.unfold(dim, size, step);
        }

        // namespace is onnxruntime
        static auto registry = torch::RegisterOperators("onnxruntime::UnfoldTensor", &UnfoldTensor);
        """)

        torch.utils.cpp_extension.load_inline(
            name="UnfoldTensor",
            cpp_sources=unfold_op_source,
            is_python_module=False,
            verbose=True,
        )

        # Create DynamicTimeWarping torch op
        dtw_op_source = textwrap.dedent("""\
        #include "torch/script.h"
        #include "torch/torch.h"
        #include <stdexcept>
        #include <tuple>
        #include <vector>

        torch::Tensor Backtrace(torch::Tensor trace) {
          int64_t i = trace.size(0) - 1;
          int64_t j = trace.size(1) - 1;
          trace.index({0, torch::indexing::Slice()}) = 2;
          trace.index({torch::indexing::Slice(), 0}) = 1;

          std::vector<int32_t> result_vec;
          while (i > 0 || j > 0) {
            result_vec.push_back(static_cast<int32_t>(i - 1));
            result_vec.push_back(static_cast<int32_t>(j - 1));
            int value = trace[i][j].item<int>();

            if (value == 0) {
              i--;
              j--;
            } else if (value == 1) {
              i--;
            } else if (value == 2) {
              j--;
            } else {
              throw std::runtime_error("Unexpected trace[i, j]");
            }
          }

          // Compute result[::-1, :].T
          torch::Tensor result = torch::from_blob(result_vec.data(), {static_cast<long int>(result_vec.size() / 2), 2}, torch::kInt32).clone();
          torch::Tensor reversed = result.flip(0); // result[::-1, :]
          torch::Tensor transposed = reversed.transpose(0, 1); // .T
          return transposed;
        }

        torch::Tensor DynamicTimeWarping(torch::Tensor x) {
          int64_t N = x.size(0);
          int64_t M = x.size(1);
          torch::Tensor cost = torch::full({N + 1, M + 1}, std::numeric_limits<float>::infinity(), torch::dtype(torch::kFloat32));
          torch::Tensor trace = torch::full({N + 1, M + 1}, -1, torch::dtype(torch::kFloat32));

          cost[0][0] = 0;
          for (int j = 1; j < M + 1; j++) {
            for (int i = 1; i < N + 1; i++) {
              float c0 = cost[i - 1][j - 1].item<float>();
              float c1 = cost[i - 1][j].item<float>();
              float c2 = cost[i][j - 1].item<float>();

              float c = 0;
              float t = 0;

              if (c0 < c1 && c0 < c2) {
                c = c0;
                t = 0;
              } else if (c1 < c0 && c1 < c2) {
                c = c1;
                t = 1;
              } else {
                c = c2;
                t = 2;
              }

              cost[i][j] = x[i - 1][j - 1].item<float>() + c;
              trace[i][j] = t;
            }
          }

          return Backtrace(trace);
        }

        // namespace is onnxruntime
        static auto registry = torch::RegisterOperators("onnxruntime::DynamicTimeWarping", &DynamicTimeWarping);
        """)

        torch.utils.cpp_extension.load_inline(
            name="DynamicTimeWarping",
            cpp_sources=dtw_op_source,
            is_python_module=False,
            verbose=True,
        )

        # Create symbolic mapping from torch ops to ORT contrib ops
        pytorch_export_contrib_ops.register()

    def export_onnx(
        self,
        onnx_model_path: str,
        provider: str,
        verbose: bool = True,
        use_external_data_format: bool = False,
        use_fp16_inputs: bool = False,
        use_int32_inputs: bool = True,
    ):
        """Export word-level timestamps to ONNX

        Args:
            onnx_model_path (str): path to save ONNX model
            provider (str): provider to use for verifying parity on ONNX model
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
            use_fp16_inputs (bool, optional): use float16 inputs for the audio_features. Defaults to False.
            use_int32_inputs (bool, optional): use int32 inputs for the decoder_input_ids. Defaults to True.
        """
        # Shape of timestamps's tensors:
        # Inputs:
        #    alignment_heads: (num_alignment_heads, 2)
        #    sot_sequence_length: (1)
        #    segment_length: (1)
        #    cross_qk_*: (batch_size, num_heads, sequence_length, num_frames // 2)
        # Outputs:
        #    jump_times: (batch_size, max_length)

        # Definitions:
        # alignment_heads: the attention head indices where the Q*K values are highly correlated with word-level timestamps
        # (i.e. the alignment between audio and text tokens)
        # This is calculated as follows:
        #
        # ```
        # import base64
        # import gzip
        # import numpy as np
        # import torch
        #
        # # base85-encoded (n_layers, n_heads) boolean arrays indicating the cross-attention heads that are
        # # highly correlated to the word-level timing, i.e. the alignment between audio and text tokens.
        # _ALIGNMENT_HEADS = {
        #     "tiny.en": b"ABzY8J1N>@0{>%R00Bk>$p{7v037`oCl~+#00",
        #     "tiny": b"ABzY8bu8Lr0{>%RKn9Fp%m@SkK7Kt=7ytkO",
        #     "base.en": b"ABzY8;40c<0{>%RzzG;p*o+Vo09|#PsxSZm00",
        #     "base": b"ABzY8KQ!870{>%RzyTQH3`Q^yNP!>##QT-<FaQ7m",
        #     "small.en": b"ABzY8>?_)10{>%RpeA61k&I|OI3I$65C{;;pbCHh0B{qLQ;+}v00",
        #     "small": b"ABzY8DmU6=0{>%Rpa?J`kvJ6qF(V^F86#Xh7JUGMK}P<N0000",
        #     "medium.en": b"ABzY8usPae0{>%R7<zz_OvQ{)4kMa0BMw6u5rT}kRKX;$NfYBv00*Hl@qhsU00",
        #     "medium": b"ABzY8B0Jh+0{>%R7}kK1fFL7w6%<-Pf*t^=N)Qr&0RR9",
        #     "large-v1": b"ABzY8r9j$a0{>%R7#4sLmoOs{s)o3~84-RPdcFk!JR<kSfC2yj",
        #     "large-v2": b"ABzY8zd+h!0{>%R7=D0pU<_bnWW*tkYAhobTNnu$jnkEkXqp)j;w1Tzk)UH3X%SZd&fFZ2fC2yj",
        #     "large-v3": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
        #     "large": b"ABzY8gWO1E0{>%R7(9S+Kn!D~%ngiGaR?*L!iJG9p-nab0JQ=-{D1-g00",
        #     "large-v3-turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
        #     "turbo": b"ABzY8j^C+e0{>%RARaKHP%t(lGR*)0g!tONPyhe`",
        # }
        #
        # model_name = "large-v3-turbo"
        # array = np.frombuffer(
        #     gzip.decompress(base64.b85decode(_ALIGNMENT_HEADS[model_name])), dtype=bool
        # ).copy()
        # mask = torch.from_numpy(array).reshape(
        #     self.dims.n_text_layer, self.dims.n_text_head
        # )
        # self.alignment_heads = mask.to_sparse().indices().T
        # ```
        #
        # sot_sequence_length: the length of the start-of-transcription sequence before the first token is generated
        # Typically the start-of-transcription sequence is [<|startoftranscription|>, <|language_token|>, <|task_token|>]
        # so its length is 3.
        #
        # segment_length: the length (in frames) of the audio segment that is being transcribed
        #
        # cross_qk_*: the Q*K values for the cross-attention blocks in the decoder
        # Every decoder layer has a self-attention block and a cross-attention block so there are `n` cross-attention blocks
        # where `n` is the number of decoder layers.
        #
        # jump_times: the timings where jumps occur in speech
        # This allows us to detect when a word began to be spoken by the speaker (start_times) and when a word was finished
        # being spoken by the speaker (end_times).

        inputs = self.inputs(use_fp16_inputs=use_fp16_inputs, use_int32_inputs=use_int32_inputs)
        input_names = self.input_names()
        output_names = self.output_names()
        dynamic_axes = get_model_dynamic_axes(self.config, input_names, output_names)

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            temp_onnx_model_path = os.path.join(tmp_dir_name, "encoder.onnx")
            Path(temp_onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
            out_path = temp_onnx_model_path if use_external_data_format else onnx_model_path

            # Create torch ops and map them to ORT contrib ops before export
            self.create_torch_ops()
            torch.onnx.export(
                self,
                args=inputs,
                f=out_path,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
                verbose=verbose,
                custom_opsets={"com.microsoft": 1},
            )

            if use_external_data_format:
                model = onnx.load_model(out_path, load_external_data=use_external_data_format)
                OnnxModel.save(
                    model,
                    onnx_model_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                )

        self.verify_onnx(onnx_model_path, provider, use_fp16_inputs, use_int32_inputs)

    def verify_onnx(
        self,
        onnx_model_path: str,
        provider: str,
        use_fp16_inputs: bool,
        use_int32_inputs: bool,
    ):
        """Verify ONNX model outputs and PyTorch model outputs match

        Args:
            onnx_model_path (str): path to save ONNX model
            provider (str): execution provider for ONNX model
            use_fp16_inputs (bool, optional): use float16 inputs for the cross_qk_{i}
            use_int32_inputs (bool, optional): use int32 inputs for the alignment_heads and sot_sequence_length
        """
        # Shape of jump times's tensors:
        # Inputs:
        #    alignment_heads: (num_alignment_heads, 2)
        #    sot_sequence_length: (1)
        #    segment_length: (1)
        #    cross_qk_*: (batch_size, num_heads, sequence_length, num_frames // 2)
        # Outputs:
        #    jump_times: (batch_size, max_length)
        inputs = self.inputs(use_fp16_inputs=use_fp16_inputs, use_int32_inputs=use_int32_inputs, return_dict=True)

        # Run PyTorch model
        pt_outputs = (
            self.forward(
                inputs["alignment_heads"], inputs["sot_sequence_length"], inputs["segment_length"], inputs["QKs"]
            )
            .detach()
            .cpu()
            .numpy()
        )

        # Run ONNX model
        sess = InferenceSession(onnx_model_path, providers=[provider])
        ort_outputs = sess.run(None, convert_inputs_for_ort(inputs, sess))

        # Calculate output difference
        diff = np.abs(pt_outputs - ort_outputs)
        print("Comparing batched jump_times...", flush=True)
        print(f"Max diff: {np.max(diff)}", flush=True)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\finite_diff.py ===
"""
Finite difference weights
=========================

This module implements an algorithm for efficient generation of finite
difference weights for ordinary differentials of functions for
derivatives from 0 (interpolation) up to arbitrary order.

The core algorithm is provided in the finite difference weight generating
function (``finite_diff_weights``), and two convenience functions are provided
for:

- estimating a derivative (or interpolate) directly from a series of points
    is also provided (``apply_finite_diff``).
- differentiating by using finite difference approximations
    (``differentiate_finite``).

"""

from sympy.core.function import Derivative
from sympy.core.singleton import S
from sympy.core.function import Subs
from sympy.core.traversal import preorder_traversal
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable



def finite_diff_weights(order, x_list, x0=S.One):
    """
    Calculates the finite difference weights for an arbitrarily spaced
    one-dimensional grid (``x_list``) for derivatives at ``x0`` of order
    0, 1, ..., up to ``order`` using a recursive formula. Order of accuracy
    is at least ``len(x_list) - order``, if ``x_list`` is defined correctly.

    Parameters
    ==========

    order: int
        Up to what derivative order weights should be calculated.
        0 corresponds to interpolation.
    x_list: sequence
        Sequence of (unique) values for the independent variable.
        It is useful (but not necessary) to order ``x_list`` from
        nearest to furthest from ``x0``; see examples below.
    x0: Number or Symbol
        Root or value of the independent variable for which the finite
        difference weights should be generated. Default is ``S.One``.

    Returns
    =======

    list
        A list of sublists, each corresponding to coefficients for
        increasing derivative order, and each containing lists of
        coefficients for increasing subsets of x_list.

    Examples
    ========

    >>> from sympy import finite_diff_weights, S
    >>> res = finite_diff_weights(1, [-S(1)/2, S(1)/2, S(3)/2, S(5)/2], 0)
    >>> res
    [[[1, 0, 0, 0],
      [1/2, 1/2, 0, 0],
      [3/8, 3/4, -1/8, 0],
      [5/16, 15/16, -5/16, 1/16]],
     [[0, 0, 0, 0],
      [-1, 1, 0, 0],
      [-1, 1, 0, 0],
      [-23/24, 7/8, 1/8, -1/24]]]
    >>> res[0][-1]  # FD weights for 0th derivative, using full x_list
    [5/16, 15/16, -5/16, 1/16]
    >>> res[1][-1]  # FD weights for 1st derivative
    [-23/24, 7/8, 1/8, -1/24]
    >>> res[1][-2]  # FD weights for 1st derivative, using x_list[:-1]
    [-1, 1, 0, 0]
    >>> res[1][-1][0]  # FD weight for 1st deriv. for x_list[0]
    -23/24
    >>> res[1][-1][1]  # FD weight for 1st deriv. for x_list[1], etc.
    7/8

    Each sublist contains the most accurate formula at the end.
    Note, that in the above example ``res[1][1]`` is the same as ``res[1][2]``.
    Since res[1][2] has an order of accuracy of
    ``len(x_list[:3]) - order = 3 - 1 = 2``, the same is true for ``res[1][1]``!

    >>> res = finite_diff_weights(1, [S(0), S(1), -S(1), S(2), -S(2)], 0)[1]
    >>> res
    [[0, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [0, 1/2, -1/2, 0, 0],
     [-1/2, 1, -1/3, -1/6, 0],
     [0, 2/3, -2/3, -1/12, 1/12]]
    >>> res[0]  # no approximation possible, using x_list[0] only
    [0, 0, 0, 0, 0]
    >>> res[1]  # classic forward step approximation
    [-1, 1, 0, 0, 0]
    >>> res[2]  # classic centered approximation
    [0, 1/2, -1/2, 0, 0]
    >>> res[3:]  # higher order approximations
    [[-1/2, 1, -1/3, -1/6, 0], [0, 2/3, -2/3, -1/12, 1/12]]

    Let us compare this to a differently defined ``x_list``. Pay attention to
    ``foo[i][k]`` corresponding to the gridpoint defined by ``x_list[k]``.

    >>> foo = finite_diff_weights(1, [-S(2), -S(1), S(0), S(1), S(2)], 0)[1]
    >>> foo
    [[0, 0, 0, 0, 0],
     [-1, 1, 0, 0, 0],
     [1/2, -2, 3/2, 0, 0],
     [1/6, -1, 1/2, 1/3, 0],
     [1/12, -2/3, 0, 2/3, -1/12]]
    >>> foo[1]  # not the same and of lower accuracy as res[1]!
    [-1, 1, 0, 0, 0]
    >>> foo[2]  # classic double backward step approximation
    [1/2, -2, 3/2, 0, 0]
    >>> foo[4]  # the same as res[4]
    [1/12, -2/3, 0, 2/3, -1/12]

    Note that, unless you plan on using approximations based on subsets of
    ``x_list``, the order of gridpoints does not matter.

    The capability to generate weights at arbitrary points can be
    used e.g. to minimize Runge's phenomenon by using Chebyshev nodes:

    >>> from sympy import cos, symbols, pi, simplify
    >>> N, (h, x) = 4, symbols('h x')
    >>> x_list = [x+h*cos(i*pi/(N)) for i in range(N,-1,-1)] # chebyshev nodes
    >>> print(x_list)
    [-h + x, -sqrt(2)*h/2 + x, x, sqrt(2)*h/2 + x, h + x]
    >>> mycoeffs = finite_diff_weights(1, x_list, 0)[1][4]
    >>> [simplify(c) for c in  mycoeffs] #doctest: +NORMALIZE_WHITESPACE
    [(h**3/2 + h**2*x - 3*h*x**2 - 4*x**3)/h**4,
    (-sqrt(2)*h**3 - 4*h**2*x + 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,
    (6*h**2*x - 8*x**3)/h**4,
    (sqrt(2)*h**3 - 4*h**2*x - 3*sqrt(2)*h*x**2 + 8*x**3)/h**4,
    (-h**3/2 + h**2*x + 3*h*x**2 - 4*x**3)/h**4]

    Notes
    =====

    If weights for a finite difference approximation of 3rd order
    derivative is wanted, weights for 0th, 1st and 2nd order are
    calculated "for free", so are formulae using subsets of ``x_list``.
    This is something one can take advantage of to save computational cost.
    Be aware that one should define ``x_list`` from nearest to furthest from
    ``x0``. If not, subsets of ``x_list`` will yield poorer approximations,
    which might not grand an order of accuracy of ``len(x_list) - order``.

    See also
    ========

    sympy.calculus.finite_diff.apply_finite_diff

    References
    ==========

    .. [1] Generation of Finite Difference Formulas on Arbitrarily Spaced
            Grids, Bengt Fornberg; Mathematics of computation; 51; 184;
            (1988); 699-706; doi:10.1090/S0025-5718-1988-0935077-0

    """
    # The notation below closely corresponds to the one used in the paper.
    order = S(order)
    if not order.is_number:
        raise ValueError("Cannot handle symbolic order.")
    if order < 0:
        raise ValueError("Negative derivative order illegal.")
    if int(order) != order:
        raise ValueError("Non-integer order illegal")
    M = order
    N = len(x_list) - 1
    delta = [[[0 for nu in range(N+1)] for n in range(N+1)] for
             m in range(M+1)]
    delta[0][0][0] = S.One
    c1 = S.One
    for n in range(1, N+1):
        c2 = S.One
        for nu in range(n):
            c3 = x_list[n] - x_list[nu]
            c2 = c2 * c3
            if n <= M:
                delta[n][n-1][nu] = 0
            for m in range(min(n, M)+1):
                delta[m][n][nu] = (x_list[n]-x0)*delta[m][n-1][nu] -\
                    m*delta[m-1][n-1][nu]
                delta[m][n][nu] /= c3
        for m in range(min(n, M)+1):
            delta[m][n][n] = c1/c2*(m*delta[m-1][n-1][n-1] -
                                    (x_list[n-1]-x0)*delta[m][n-1][n-1])
        c1 = c2
    return delta


def apply_finite_diff(order, x_list, y_list, x0=S.Zero):
    """
    Calculates the finite difference approximation of
    the derivative of requested order at ``x0`` from points
    provided in ``x_list`` and ``y_list``.

    Parameters
    ==========

    order: int
        order of derivative to approximate. 0 corresponds to interpolation.
    x_list: sequence
        Sequence of (unique) values for the independent variable.
    y_list: sequence
        The function value at corresponding values for the independent
        variable in x_list.
    x0: Number or Symbol
        At what value of the independent variable the derivative should be
        evaluated. Defaults to 0.

    Returns
    =======

    sympy.core.add.Add or sympy.core.numbers.Number
        The finite difference expression approximating the requested
        derivative order at ``x0``.

    Examples
    ========

    >>> from sympy import apply_finite_diff
    >>> cube = lambda arg: (1.0*arg)**3
    >>> xlist = range(-3,3+1)
    >>> apply_finite_diff(2, xlist, map(cube, xlist), 2) - 12 # doctest: +SKIP
    -3.55271367880050e-15

    we see that the example above only contain rounding errors.
    apply_finite_diff can also be used on more abstract objects:

    >>> from sympy import IndexedBase, Idx
    >>> x, y = map(IndexedBase, 'xy')
    >>> i = Idx('i')
    >>> x_list, y_list = zip(*[(x[i+j], y[i+j]) for j in range(-1,2)])
    >>> apply_finite_diff(1, x_list, y_list, x[i])
    ((x[i + 1] - x[i])/(-x[i - 1] + x[i]) - 1)*y[i]/(x[i + 1] - x[i]) -
    (x[i + 1] - x[i])*y[i - 1]/((x[i + 1] - x[i - 1])*(-x[i - 1] + x[i])) +
    (-x[i - 1] + x[i])*y[i + 1]/((x[i + 1] - x[i - 1])*(x[i + 1] - x[i]))

    Notes
    =====

    Order = 0 corresponds to interpolation.
    Only supply so many points you think makes sense
    to around x0 when extracting the derivative (the function
    need to be well behaved within that region). Also beware
    of Runge's phenomenon.

    See also
    ========

    sympy.calculus.finite_diff.finite_diff_weights

    References
    ==========

    Fortran 90 implementation with Python interface for numerics: finitediff_

    .. _finitediff: https://github.com/bjodah/finitediff

    """

    # In the original paper the following holds for the notation:
    # M = order
    # N = len(x_list) - 1

    N = len(x_list) - 1
    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list not equal in length.")

    delta = finite_diff_weights(order, x_list, x0)

    derivative = 0
    for nu in range(len(x_list)):
        derivative += delta[order][N][nu]*y_list[nu]
    return derivative


def _as_finite_diff(derivative, points=1, x0=None, wrt=None):
    """
    Returns an approximation of a derivative of a function in
    the form of a finite difference formula. The expression is a
    weighted sum of the function at a number of discrete values of
    (one of) the independent variable(s).

    Parameters
    ==========

    derivative: a Derivative instance

    points: sequence or coefficient, optional
        If sequence: discrete values (length >= order+1) of the
        independent variable used for generating the finite
        difference weights.
        If it is a coefficient, it will be used as the step-size
        for generating an equidistant sequence of length order+1
        centered around ``x0``. default: 1 (step-size 1)

    x0: number or Symbol, optional
        the value of the independent variable (``wrt``) at which the
        derivative is to be approximated. Default: same as ``wrt``.

    wrt: Symbol, optional
        "with respect to" the variable for which the (partial)
        derivative is to be approximated for. If not provided it
        is required that the Derivative is ordinary. Default: ``None``.

    Examples
    ========

    >>> from sympy import symbols, Function, exp, sqrt, Symbol
    >>> from sympy.calculus.finite_diff import _as_finite_diff
    >>> x, h = symbols('x h')
    >>> f = Function('f')
    >>> _as_finite_diff(f(x).diff(x))
    -f(x - 1/2) + f(x + 1/2)

    The default step size and number of points are 1 and ``order + 1``
    respectively. We can change the step size by passing a symbol
    as a parameter:

    >>> _as_finite_diff(f(x).diff(x), h)
    -f(-h/2 + x)/h + f(h/2 + x)/h

    We can also specify the discretized values to be used in a sequence:

    >>> _as_finite_diff(f(x).diff(x), [x, x+h, x+2*h])
    -3*f(x)/(2*h) + 2*f(h + x)/h - f(2*h + x)/(2*h)

    The algorithm is not restricted to use equidistant spacing, nor
    do we need to make the approximation around ``x0``, but we can get
    an expression estimating the derivative at an offset:

    >>> e, sq2 = exp(1), sqrt(2)
    >>> xl = [x-h, x+h, x+e*h]
    >>> _as_finite_diff(f(x).diff(x, 1), xl, x+h*sq2)
    2*h*((h + sqrt(2)*h)/(2*h) - (-sqrt(2)*h + h)/(2*h))*f(E*h + x)/((-h + E*h)*(h + E*h)) +
    (-(-sqrt(2)*h + h)/(2*h) - (-sqrt(2)*h + E*h)/(2*h))*f(-h + x)/(h + E*h) +
    (-(h + sqrt(2)*h)/(2*h) + (-sqrt(2)*h + E*h)/(2*h))*f(h + x)/(-h + E*h)

    Partial derivatives are also supported:

    >>> y = Symbol('y')
    >>> d2fdxdy=f(x,y).diff(x,y)
    >>> _as_finite_diff(d2fdxdy, wrt=x)
    -Derivative(f(x - 1/2, y), y) + Derivative(f(x + 1/2, y), y)

    See also
    ========

    sympy.calculus.finite_diff.apply_finite_diff
    sympy.calculus.finite_diff.finite_diff_weights

    """
    if derivative.is_Derivative:
        pass
    elif derivative.is_Atom:
        return derivative
    else:
        return derivative.fromiter(
            [_as_finite_diff(ar, points, x0, wrt) for ar
             in derivative.args], **derivative.assumptions0)

    if wrt is None:
        old = None
        for v in derivative.variables:
            if old is v:
                continue
            derivative = _as_finite_diff(derivative, points, x0, v)
            old = v
        return derivative

    order = derivative.variables.count(wrt)

    if x0 is None:
        x0 = wrt

    if not iterable(points):
        if getattr(points, 'is_Function', False) and wrt in points.args:
            points = points.subs(wrt, x0)
        # points is simply the step-size, let's make it a
        # equidistant sequence centered around x0
        if order % 2 == 0:
            # even order => odd number of points, grid point included
            points = [x0 + points*i for i
                      in range(-order//2, order//2 + 1)]
        else:
            # odd order => even number of points, half-way wrt grid point
            points = [x0 + points*S(i)/2 for i
                      in range(-order, order + 1, 2)]
    others = [wrt, 0]
    for v in set(derivative.variables):
        if v == wrt:
            continue
        others += [v, derivative.variables.count(v)]
    if len(points) < order+1:
        raise ValueError("Too few points for order %d" % order)
    return apply_finite_diff(order, points, [
        Derivative(derivative.expr.subs({wrt: x}), *others) for
        x in points], x0)


def differentiate_finite(expr, *symbols,
                         points=1, x0=None, wrt=None, evaluate=False):
    r""" Differentiate expr and replace Derivatives with finite differences.

    Parameters
    ==========

    expr : expression
    \*symbols : differentiate with respect to symbols
    points: sequence, coefficient or undefined function, optional
        see ``Derivative.as_finite_difference``
    x0: number or Symbol, optional
        see ``Derivative.as_finite_difference``
    wrt: Symbol, optional
        see ``Derivative.as_finite_difference``

    Examples
    ========

    >>> from sympy import sin, Function, differentiate_finite
    >>> from sympy.abc import x, y, h
    >>> f, g = Function('f'), Function('g')
    >>> differentiate_finite(f(x)*g(x), x, points=[x-h, x+h])
    -f(-h + x)*g(-h + x)/(2*h) + f(h + x)*g(h + x)/(2*h)

    ``differentiate_finite`` works on any expression, including the expressions
    with embedded derivatives:

    >>> differentiate_finite(f(x) + sin(x), x, 2)
    -2*f(x) + f(x - 1) + f(x + 1) - 2*sin(x) + sin(x - 1) + sin(x + 1)
    >>> differentiate_finite(f(x, y), x, y)
    f(x - 1/2, y - 1/2) - f(x - 1/2, y + 1/2) - f(x + 1/2, y - 1/2) + f(x + 1/2, y + 1/2)
    >>> differentiate_finite(f(x)*g(x).diff(x), x)
    (-g(x) + g(x + 1))*f(x + 1/2) - (g(x) - g(x - 1))*f(x - 1/2)

    To make finite difference with non-constant discretization step use
    undefined functions:

    >>> dx = Function('dx')
    >>> differentiate_finite(f(x)*g(x).diff(x), points=dx(x))
    -(-g(x - dx(x)/2 - dx(x - dx(x)/2)/2)/dx(x - dx(x)/2) +
    g(x - dx(x)/2 + dx(x - dx(x)/2)/2)/dx(x - dx(x)/2))*f(x - dx(x)/2)/dx(x) +
    (-g(x + dx(x)/2 - dx(x + dx(x)/2)/2)/dx(x + dx(x)/2) +
    g(x + dx(x)/2 + dx(x + dx(x)/2)/2)/dx(x + dx(x)/2))*f(x + dx(x)/2)/dx(x)

    """
    if any(term.is_Derivative for term in list(preorder_traversal(expr))):
        evaluate = False

    Dexpr = expr.diff(*symbols, evaluate=evaluate)
    if evaluate:
        sympy_deprecation_warning("""
        The evaluate flag to differentiate_finite() is deprecated.

        evaluate=True expands the intermediate derivatives before computing
        differences, but this usually not what you want, as it does not
        satisfy the product rule.
        """,
            deprecated_since_version="1.5",
             active_deprecations_target="deprecated-differentiate_finite-evaluate",
        )
        return Dexpr.replace(
            lambda arg: arg.is_Derivative,
            lambda arg: arg.as_finite_difference(points=points, x0=x0, wrt=wrt))
    else:
        DFexpr = Dexpr.as_finite_difference(points=points, x0=x0, wrt=wrt)
        return DFexpr.replace(
            lambda arg: isinstance(arg, Subs),
            lambda arg: arg.expr.as_finite_difference(
                    points=points, x0=arg.point[0], wrt=arg.variables[0]))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\builder\_htmlparser.py ===
# encoding: utf-8
"""Use the HTMLParser library to parse HTML files that aren't too bad."""
from __future__ import annotations

# Use of this source code is governed by the MIT license.
__license__ = "MIT"

__all__ = [
    "HTMLParserTreeBuilder",
]

from html.parser import HTMLParser

from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    TYPE_CHECKING,
    Tuple,
    Type,
    Union,
)

from bs4.element import (
    AttributeDict,
    CData,
    Comment,
    Declaration,
    Doctype,
    ProcessingInstruction,
)
from bs4.dammit import EntitySubstitution, UnicodeDammit

from bs4.builder import (
    DetectsXMLParsedAsHTML,
    HTML,
    HTMLTreeBuilder,
    STRICT,
)

from bs4.exceptions import ParserRejectedMarkup

if TYPE_CHECKING:
    from bs4 import BeautifulSoup
    from bs4.element import NavigableString
    from bs4._typing import (
        _Encoding,
        _Encodings,
        _RawMarkup,
    )

HTMLPARSER = "html.parser"

_DuplicateAttributeHandler = Callable[[Dict[str, str], str, str], None]


class BeautifulSoupHTMLParser(HTMLParser, DetectsXMLParsedAsHTML):
    #: Constant to handle duplicate attributes by ignoring later values
    #: and keeping the earlier ones.
    REPLACE: str = "replace"

    #: Constant to handle duplicate attributes by replacing earlier values
    #: with later ones.
    IGNORE: str = "ignore"

    """A subclass of the Python standard library's HTMLParser class, which
    listens for HTMLParser events and translates them into calls
    to Beautiful Soup's tree construction API.

        :param on_duplicate_attribute: A strategy for what to do if a
            tag includes the same attribute more than once. Accepted
            values are: REPLACE (replace earlier values with later
            ones, the default), IGNORE (keep the earliest value
            encountered), or a callable. A callable must take three
            arguments: the dictionary of attributes already processed,
            the name of the duplicate attribute, and the most recent value
            encountered.
    """

    def __init__(
        self,
        soup: BeautifulSoup,
        *args: Any,
        on_duplicate_attribute: Union[str, _DuplicateAttributeHandler] = REPLACE,
        **kwargs: Any,
    ):
        self.soup = soup
        self.on_duplicate_attribute = on_duplicate_attribute
        self.attribute_dict_class = soup.builder.attribute_dict_class
        HTMLParser.__init__(self, *args, **kwargs)

        # Keep a list of empty-element tags that were encountered
        # without an explicit closing tag. If we encounter a closing tag
        # of this type, we'll associate it with one of those entries.
        #
        # This isn't a stack because we don't care about the
        # order. It's a list of closing tags we've already handled and
        # will ignore, assuming they ever show up.
        self.already_closed_empty_element = []

        self._initialize_xml_detector()

    on_duplicate_attribute: Union[str, _DuplicateAttributeHandler]
    already_closed_empty_element: List[str]
    soup: BeautifulSoup

    def error(self, message: str) -> None:
        # NOTE: This method is required so long as Python 3.9 is
        # supported. The corresponding code is removed from HTMLParser
        # in 3.5, but not removed from ParserBase until 3.10.
        # https://github.com/python/cpython/issues/76025
        #
        # The original implementation turned the error into a warning,
        # but in every case I discovered, this made HTMLParser
        # immediately crash with an error message that was less
        # helpful than the warning. The new implementation makes it
        # more clear that html.parser just can't parse this
        # markup. The 3.10 implementation does the same, though it
        # raises AssertionError rather than calling a method. (We
        # catch this error and wrap it in a ParserRejectedMarkup.)
        raise ParserRejectedMarkup(message)

    def handle_startendtag(
        self, name: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        """Handle an incoming empty-element tag.

        html.parser only calls this method when the markup looks like
        <tag/>.
        """
        # `handle_empty_element` tells handle_starttag not to close the tag
        # just because its name matches a known empty-element tag. We
        # know that this is an empty-element tag, and we want to call
        # handle_endtag ourselves.
        self.handle_starttag(name, attrs, handle_empty_element=False)
        self.handle_endtag(name)

    def handle_starttag(
        self,
        name: str,
        attrs: List[Tuple[str, Optional[str]]],
        handle_empty_element: bool = True,
    ) -> None:
        """Handle an opening tag, e.g. '<tag>'

        :param handle_empty_element: True if this tag is known to be
            an empty-element tag (i.e. there is not expected to be any
            closing tag).
        """
        # TODO: handle namespaces here?
        attr_dict: AttributeDict = self.attribute_dict_class()
        for key, value in attrs:
            # Change None attribute values to the empty string
            # for consistency with the other tree builders.
            if value is None:
                value = ""
            if key in attr_dict:
                # A single attribute shows up multiple times in this
                # tag. How to handle it depends on the
                # on_duplicate_attribute setting.
                on_dupe = self.on_duplicate_attribute
                if on_dupe == self.IGNORE:
                    pass
                elif on_dupe in (None, self.REPLACE):
                    attr_dict[key] = value
                else:
                    on_dupe = cast(_DuplicateAttributeHandler, on_dupe)
                    on_dupe(attr_dict, key, value)
            else:
                attr_dict[key] = value
        # print("START", name)
        sourceline: Optional[int]
        sourcepos: Optional[int]
        if self.soup.builder.store_line_numbers:
            sourceline, sourcepos = self.getpos()
        else:
            sourceline = sourcepos = None
        tag = self.soup.handle_starttag(
            name, None, None, attr_dict, sourceline=sourceline, sourcepos=sourcepos
        )
        if tag and tag.is_empty_element and handle_empty_element:
            # Unlike other parsers, html.parser doesn't send separate end tag
            # events for empty-element tags. (It's handled in
            # handle_startendtag, but only if the original markup looked like
            # <tag/>.)
            #
            # So we need to call handle_endtag() ourselves. Since we
            # know the start event is identical to the end event, we
            # don't want handle_endtag() to cross off any previous end
            # events for tags of this name.
            self.handle_endtag(name, check_already_closed=False)

            # But we might encounter an explicit closing tag for this tag
            # later on. If so, we want to ignore it.
            self.already_closed_empty_element.append(name)

        if self._root_tag_name is None:
            self._root_tag_encountered(name)

    def handle_endtag(self, name: str, check_already_closed: bool = True) -> None:
        """Handle a closing tag, e.g. '</tag>'

        :param name: A tag name.
        :param check_already_closed: True if this tag is expected to
           be the closing portion of an empty-element tag,
           e.g. '<tag></tag>'.
        """
        # print("END", name)
        if check_already_closed and name in self.already_closed_empty_element:
            # This is a redundant end tag for an empty-element tag.
            # We've already called handle_endtag() for it, so just
            # check it off the list.
            # print("ALREADY CLOSED", name)
            self.already_closed_empty_element.remove(name)
        else:
            self.soup.handle_endtag(name)

    def handle_data(self, data: str) -> None:
        """Handle some textual data that shows up between tags."""
        self.soup.handle_data(data)

    def handle_charref(self, name: str) -> None:
        """Handle a numeric character reference by converting it to the
        corresponding Unicode character and treating it as textual
        data.

        :param name: Character number, possibly in hexadecimal.
        """
        # TODO: This was originally a workaround for a bug in
        # HTMLParser. (http://bugs.python.org/issue13633) The bug has
        # been fixed, but removing this code still makes some
        # Beautiful Soup tests fail. This needs investigation.
        if name.startswith("x"):
            real_name = int(name.lstrip("x"), 16)
        elif name.startswith("X"):
            real_name = int(name.lstrip("X"), 16)
        else:
            real_name = int(name)

        data = None
        if real_name < 256:
            # HTML numeric entities are supposed to reference Unicode
            # code points, but sometimes they reference code points in
            # some other encoding (ahem, Windows-1252). E.g. &#147;
            # instead of &#201; for LEFT DOUBLE QUOTATION MARK. This
            # code tries to detect this situation and compensate.
            for encoding in (self.soup.original_encoding, "windows-1252"):
                if not encoding:
                    continue
                try:
                    data = bytearray([real_name]).decode(encoding)
                except UnicodeDecodeError:
                    pass
        if not data:
            try:
                data = chr(real_name)
            except (ValueError, OverflowError):
                pass
        data = data or "\N{REPLACEMENT CHARACTER}"
        self.handle_data(data)

    def handle_entityref(self, name: str) -> None:
        """Handle a named entity reference by converting it to the
        corresponding Unicode character(s) and treating it as textual
        data.

        :param name: Name of the entity reference.
        """
        character = EntitySubstitution.HTML_ENTITY_TO_CHARACTER.get(name)
        if character is not None:
            data = character
        else:
            # If this were XML, it would be ambiguous whether "&foo"
            # was an character entity reference with a missing
            # semicolon or the literal string "&foo". Since this is
            # HTML, we have a complete list of all character entity references,
            # and this one wasn't found, so assume it's the literal string "&foo".
            data = "&%s" % name
        self.handle_data(data)

    def handle_comment(self, data: str) -> None:
        """Handle an HTML comment.

        :param data: The text of the comment.
        """
        self.soup.endData()
        self.soup.handle_data(data)
        self.soup.endData(Comment)

    def handle_decl(self, data: str) -> None:
        """Handle a DOCTYPE declaration.

        :param data: The text of the declaration.
        """
        self.soup.endData()
        data = data[len("DOCTYPE ") :]
        self.soup.handle_data(data)
        self.soup.endData(Doctype)

    def unknown_decl(self, data: str) -> None:
        """Handle a declaration of unknown type -- probably a CDATA block.

        :param data: The text of the declaration.
        """
        cls: Type[NavigableString]
        if data.upper().startswith("CDATA["):
            cls = CData
            data = data[len("CDATA[") :]
        else:
            cls = Declaration
        self.soup.endData()
        self.soup.handle_data(data)
        self.soup.endData(cls)

    def handle_pi(self, data: str) -> None:
        """Handle a processing instruction.

        :param data: The text of the instruction.
        """
        self.soup.endData()
        self.soup.handle_data(data)
        self._document_might_be_xml(data)
        self.soup.endData(ProcessingInstruction)


class HTMLParserTreeBuilder(HTMLTreeBuilder):
    """A Beautiful soup `bs4.builder.TreeBuilder` that uses the
    :py:class:`html.parser.HTMLParser` parser, found in the Python
    standard library.

    """

    is_xml: bool = False
    picklable: bool = True
    NAME: str = HTMLPARSER
    features: Iterable[str] = [NAME, HTML, STRICT]
    parser_args: Tuple[Iterable[Any], Dict[str, Any]]

    #: The html.parser knows which line number and position in the
    #: original file is the source of an element.
    TRACKS_LINE_NUMBERS: bool = True

    def __init__(
        self,
        parser_args: Optional[Iterable[Any]] = None,
        parser_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        """Constructor.

        :param parser_args: Positional arguments to pass into
            the BeautifulSoupHTMLParser constructor, once it's
            invoked.
        :param parser_kwargs: Keyword arguments to pass into
            the BeautifulSoupHTMLParser constructor, once it's
            invoked.
        :param kwargs: Keyword arguments for the superclass constructor.
        """
        # Some keyword arguments will be pulled out of kwargs and placed
        # into parser_kwargs.
        extra_parser_kwargs = dict()
        for arg in ("on_duplicate_attribute",):
            if arg in kwargs:
                value = kwargs.pop(arg)
                extra_parser_kwargs[arg] = value
        super(HTMLParserTreeBuilder, self).__init__(**kwargs)
        parser_args = parser_args or []
        parser_kwargs = parser_kwargs or {}
        parser_kwargs.update(extra_parser_kwargs)
        parser_kwargs["convert_charrefs"] = False
        self.parser_args = (parser_args, parser_kwargs)

    def prepare_markup(
        self,
        markup: _RawMarkup,
        user_specified_encoding: Optional[_Encoding] = None,
        document_declared_encoding: Optional[_Encoding] = None,
        exclude_encodings: Optional[_Encodings] = None,
    ) -> Iterable[Tuple[str, Optional[_Encoding], Optional[_Encoding], bool]]:
        """Run any preliminary steps necessary to make incoming markup
        acceptable to the parser.

        :param markup: Some markup -- probably a bytestring.
        :param user_specified_encoding: The user asked to try this encoding.
        :param document_declared_encoding: The markup itself claims to be
            in this encoding.
        :param exclude_encodings: The user asked _not_ to try any of
            these encodings.

        :yield: A series of 4-tuples: (markup, encoding, declared encoding,
             has undergone character replacement)

            Each 4-tuple represents a strategy for parsing the document.
            This TreeBuilder uses Unicode, Dammit to convert the markup
            into Unicode, so the ``markup`` element of the tuple will
            always be a string.
        """
        if isinstance(markup, str):
            # Parse Unicode as-is.
            yield (markup, None, None, False)
            return

        # Ask UnicodeDammit to sniff the most likely encoding.

        known_definite_encodings: List[_Encoding] = []
        if user_specified_encoding:
            # This was provided by the end-user; treat it as a known
            # definite encoding per the algorithm laid out in the
            # HTML5 spec. (See the EncodingDetector class for
            # details.)
            known_definite_encodings.append(user_specified_encoding)

        user_encodings: List[_Encoding] = []
        if document_declared_encoding:
            # This was found in the document; treat it as a slightly
            # lower-priority user encoding.
            user_encodings.append(document_declared_encoding)

        dammit = UnicodeDammit(
            markup,
            known_definite_encodings=known_definite_encodings,
            user_encodings=user_encodings,
            is_html=True,
            exclude_encodings=exclude_encodings,
        )

        if dammit.unicode_markup is None:
            # In every case I've seen, Unicode, Dammit is able to
            # convert the markup into Unicode, even if it needs to use
            # REPLACEMENT CHARACTER. But there is a code path that
            # could result in unicode_markup being None, and
            # HTMLParser can only parse Unicode, so here we handle
            # that code path.
            raise ParserRejectedMarkup(
                "Could not convert input to Unicode, and html.parser will not accept bytestrings."
            )
        else:
            yield (
                dammit.unicode_markup,
                dammit.original_encoding,
                dammit.declared_html_encoding,
                dammit.contains_replacement_characters,
            )

    def feed(self, markup: _RawMarkup) -> None:
        args, kwargs = self.parser_args

        # HTMLParser.feed will only handle str, but
        # BeautifulSoup.markup is allowed to be _RawMarkup, because
        # it's set by the yield value of
        # TreeBuilder.prepare_markup. Fortunately,
        # HTMLParserTreeBuilder.prepare_markup always yields a str
        # (UnicodeDammit.unicode_markup).
        assert isinstance(markup, str)

        # We know BeautifulSoup calls TreeBuilder.initialize_soup
        # before calling feed(), so we can assume self.soup
        # is set.
        assert self.soup is not None
        parser = BeautifulSoupHTMLParser(self.soup, *args, **kwargs)

        try:
            parser.feed(markup)
            parser.close()
        except AssertionError as e:
            # html.parser raises AssertionError in rare cases to
            # indicate a fatal problem with the markup, especially
            # when there's an error in the doctype declaration.
            raise ParserRejectedMarkup(e)
        parser.already_closed_empty_element = []

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_bert_keras.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

import onnx
from onnx import numpy_helper
from onnx_model_bert_tf import BertOnnxModelTF

logger = logging.getLogger(__name__)


class BertOnnxModelKeras(BertOnnxModelTF):
    def __init__(self, model, num_heads, hidden_size):
        super().__init__(model, num_heads, hidden_size)

    def match_mask_path(self, add_or_sub_before_softmax):
        mask_nodes = self.match_parent_path(
            add_or_sub_before_softmax,
            ["Mul", "Sub", "Reshape", "Cast"],
            [1, None, 1, 0],
        )
        if mask_nodes is not None:
            return mask_nodes

        mask_nodes = self.match_parent_path(
            add_or_sub_before_softmax,
            ["Mul", "Sub", "Cast", "Slice", "Unsqueeze"],
            [1, 1, 1, 0, 0],
        )
        if mask_nodes is not None:
            return mask_nodes

        mask_nodes = self.match_parent_path(
            add_or_sub_before_softmax,
            ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze"],
            [1, None, 1, 0, 0],
        )
        return mask_nodes

    def check_attention_input(self, matmul_q, matmul_k, matmul_v, parent, output_name_to_node):
        reshape_nodes = []

        for x in [matmul_q, matmul_k, matmul_v]:
            root_input = x.input[0]
            root_node = output_name_to_node[root_input]
            if root_node == parent:
                continue
            if root_node.op_type == "Reshape" and root_node.input[0] == parent.output[0]:
                reshape_nodes.append(root_node)
                continue
            logger.debug(f"Check attention input failed:{root_input}, {parent.output[0]}")
            return False, []

        return True, reshape_nodes

    def fuse_attention(self):
        self.input_name_to_nodes()
        output_name_to_node = self.output_name_to_node()

        nodes_to_remove = []
        attention_count = 0

        skip_layer_norm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        for normalize_node in skip_layer_norm_nodes:
            # SkipLayerNormalization has two inputs, and one of them is the root input for attention.
            parent = self.get_parent(normalize_node, 0)
            if parent is None or parent.op_type not in [
                "SkipLayerNormalization",
                "EmbedLayerNormalization",
            ]:
                if parent.op_type == "Add":
                    parent = self.get_parent(normalize_node, 1)
                    if parent is None or parent.op_type not in [
                        "SkipLayerNormalization",
                        "EmbedLayerNormalization",
                    ]:
                        logger.debug(f"First input for skiplayernorm: {parent.op_type if parent is not None else None}")
                        continue
                else:
                    logger.debug(f"First input for skiplayernorm: {parent.op_type if parent is not None else None}")
                    continue
            else:
                # TODO: shall we add back the checking of children op types.
                pass

            qkv_nodes = self.match_parent_path(
                normalize_node,
                ["Add", "Reshape", "MatMul", "Reshape", "Transpose", "MatMul"],
                [None, 0, 0, 0, 0, 0],
            )
            if qkv_nodes is None:
                logger.debug("Failed to match qkv nodes")
                continue
            (
                add,
                extra_reshape_0,
                matmul,
                reshape_qkv,
                transpose_qkv,
                matmul_qkv,
            ) = qkv_nodes
            logger.debug("Matched qkv nodes")

            v_nodes = self.match_parent_path(
                matmul_qkv,
                ["Transpose", "Reshape", "Add", "Reshape", "MatMul"],
                [1, 0, 0, 0, 0],
            )
            if v_nodes is None:
                logger.debug("Failed to match v path")
                continue
            (transpose_v, reshape_v, add_v, extra_reshape_1, matmul_v) = v_nodes

            qk_nodes = self.match_parent_path(matmul_qkv, ["Softmax", "Sub", "MatMul"], [0, 0, 0])
            if qk_nodes is not None:
                (softmax_qk, sub_qk, matmul_qk) = qk_nodes
                q_nodes = self.match_parent_path(
                    matmul_qk,
                    ["Mul", "Transpose", "Reshape", "Add", "Reshape", "MatMul"],
                    [0, None, 0, 0, 0, 0],
                )
                if q_nodes is not None:
                    (
                        mul_q,
                        transpose_q,
                        reshape_q,
                        add_q,
                        extra_reshape_2,
                        matmul_q,
                    ) = q_nodes

            else:
                qk_nodes = self.match_parent_path(matmul_qkv, ["Softmax", "Add", "Mul", "MatMul"], [0, 0, 0, None])
                if qk_nodes is None:
                    qk_nodes = self.match_parent_path(matmul_qkv, ["Softmax", "Add", "Div", "MatMul"], [0, 0, 0, None])
                    if qk_nodes is None:
                        logger.debug("Failed to match qk path")
                        continue
                (softmax_qk, add_qk, mul_qk, matmul_qk) = qk_nodes

                q_nodes = self.match_parent_path(
                    matmul_qk,
                    ["Transpose", "Reshape", "Add", "Reshape", "MatMul"],
                    [0, 0, 0, 0, 0],
                )
                if q_nodes is not None:
                    (transpose_q, reshape_q, add_q, extra_reshape_2, matmul_q) = q_nodes

            if q_nodes is None:
                logger.debug("Failed to match q path")
                continue

            k_nodes = self.match_parent_path(
                matmul_qk,
                ["Transpose", "Reshape", "Add", "Reshape", "MatMul"],
                [1, 0, 0, 0, 0],
            )
            if k_nodes is None:
                logger.debug("Failed to match k path")
                continue
            (transpose_k, reshape_k, add_k, extra_reshape_3, matmul_k) = k_nodes

            mask_nodes = self.match_mask_path(qk_nodes[1])
            if mask_nodes is None:
                logger.debug("Failed to match mask path")
                continue
            if not self.has_constant_input(mask_nodes[1], 1):
                logger.debug("Sub node expected to have an input with constant value 1.0.")
                continue

            is_same_root, reshape_nodes = self.check_attention_input(
                matmul_q, matmul_k, matmul_v, parent, output_name_to_node
            )
            if is_same_root:
                mask_index = self.attention_mask.process_mask(mask_nodes[-1].input[0])
                logger.debug("Create an Attention node.")
                attention_node = self.attention_fusion.create_attention_node(
                    mask_index=mask_index,
                    q_matmul=matmul_q,
                    k_matmul=matmul_k,
                    v_matmul=matmul_v,
                    q_add=add_q,
                    k_add=add_k,
                    v_add=add_v,
                    num_heads=self.num_heads,
                    hidden_size=self.hidden_size,
                    first_input=parent.output[0],
                    output=reshape_qkv.output[0],
                )
                if attention_node is None:
                    continue

                self.add_node(attention_node)
                attention_count += 1

                nodes_to_remove.extend([reshape_qkv, transpose_qkv, matmul_qkv])
                nodes_to_remove.extend(qk_nodes)
                nodes_to_remove.extend(q_nodes)
                nodes_to_remove.extend(k_nodes)
                nodes_to_remove.extend(v_nodes)
                nodes_to_remove.extend(mask_nodes)
                nodes_to_remove.extend(reshape_nodes)
                nodes_to_remove.append(extra_reshape_0)
                self.replace_node_input(add, extra_reshape_0.output[0], matmul.output[0])
            else:
                logger.debug("Root node not matched.")
                continue
        self.remove_nodes(nodes_to_remove)
        self.update_graph()
        logger.info(f"Fused Attention count:{attention_count}")

    def preprocess(self):
        self.process_embedding()
        self.fuse_mask()
        self.skip_reshape()

    def skip_reshape(self):
        self.input_name_to_nodes()
        self.output_name_to_node()

        count = 0
        reshape_nodes = self.get_nodes_by_op_type("Reshape")
        for reshape_node in reshape_nodes:
            parent = self.get_parent(reshape_node, 0)
            if parent is not None and parent.op_type == "Reshape":
                reshape_node.input[0] = parent.input[0]
                count += 1

        if count > 0:
            logger.info(f"Skip consequent Reshape count: {count}")

    def fuse_embedding(self, node, output_name_to_node):
        assert node.op_type == "LayerNormalization"
        logger.debug(f"start fusing embedding from node with output={node.output[0]}...")
        word_embed_path = self.match_parent_path(node, ["Add", "Add", "Gather"], [0, 0, 0], output_name_to_node)
        if word_embed_path is None:
            logger.debug("failed to match word_embed_path")
            return False

        skip_node, add_node, gather_node = word_embed_path

        word_initializer = self.get_initializer(gather_node.input[0])
        if word_initializer is None:
            logger.debug("failed to get word initializer")
            return False

        temp = numpy_helper.to_array(word_initializer)
        if len(temp.shape) == 2:
            logger.info(f"Found word embedding. name:{word_initializer.name}, shape:{temp.shape}")
            word_embedding = word_initializer.name
        else:
            logger.info(f"Failed to find word embedding. name:{word_initializer.name}, shape:{temp.shape}")
            return False

        pos_initializer = self.get_initializer(add_node.input[1])
        if pos_initializer is not None:
            temp = numpy_helper.to_array(pos_initializer)
            if len(temp.shape) == 3 and temp.shape[0] == 1:
                tensor = numpy_helper.from_array(temp.reshape((temp.shape[1], temp.shape[2])), "position_embedding")
                self.add_initializer(tensor)
                logger.info(f"Found position embedding. name:{pos_initializer.name}, shape:{temp.shape[1:]}")
                position_embedding = "position_embedding"
            else:
                logger.info(f"Failed to find position embedding. name:{pos_initializer.name}, shape:{temp.shape}")
                return False
        else:
            pos_embed_path = self.match_parent_path(add_node, ["Gather", "Slice"], [1, 1], output_name_to_node)
            if pos_embed_path is None:
                logger.debug("failed to match pos_embed_path")
                return False

            pos_gather, pos_slice = pos_embed_path
            pos_initializer = self.get_initializer(pos_gather.input[0])
            if pos_initializer is None:
                logger.debug("failed to get pos initializer")
                return False

            temp = numpy_helper.to_array(pos_initializer)
            if len(temp.shape) == 2:
                logger.info(f"Found word embedding. name:{pos_initializer.name}, shape:{temp.shape}")
                position_embedding = pos_initializer.name
            else:
                logger.info(f"Failed to find position embedding. name:{pos_initializer.name}, shape:{temp.shape}")
                return False

        gather = self.get_parent(skip_node, 1, output_name_to_node)
        if gather is None or gather.op_type != "Gather":
            logger.debug("failed to get gather")
            return False

        segment_initializer = self.get_initializer(gather.input[0])
        if segment_initializer is None:
            logger.debug("failed to get segment initializer")
            return False

        temp = numpy_helper.to_array(segment_initializer)
        if len(temp.shape) == 2:
            logger.info(f"Found segment embedding. name:{segment_initializer.name}, shape:{temp.shape}")
            segment_embedding = segment_initializer.name
        else:
            logger.info(f"Failed to find segment embedding. name:{segment_initializer.name}, shape:{temp.shape}")
            return False

        logger.info("Create Embedding node")
        self.create_embedding_subgraph(node, word_embedding, segment_embedding, position_embedding)
        return True

    def process_embedding(self):
        """
        Automatically detect word, segment and position embeddings.
        """
        logger.info("start processing embedding layer...")
        output_name_to_node = self.output_name_to_node()
        for node in self.nodes():
            if node.op_type == "LayerNormalization":
                if self.fuse_embedding(node, output_name_to_node):
                    return
                break

    def fuse_mask(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Mul" and self.has_constant_input(node, -10000):
                mask_path = self.match_parent_path(node, ["Sub", "Cast", "Slice", "Unsqueeze"], [0, 1, 0, 0])
                if mask_path is None:
                    continue
                sub_node, cast_node, slice_node, unsqueeze_node = mask_path

                mask_input_name = self.attention_mask.get_first_mask()
                if unsqueeze_node.input[0] != mask_input_name:
                    print(f"Cast input {unsqueeze_node.input[0]} is not mask input {mask_input_name}")
                    continue

                unsqueeze_added_1 = onnx.helper.make_node(
                    "Unsqueeze",
                    inputs=[mask_input_name],
                    outputs=["mask_fuse_unsqueeze1_output"],
                    name="Mask_UnSqueeze_1",
                    axes=[1],
                )

                unsqueeze_added_2 = onnx.helper.make_node(
                    "Unsqueeze",
                    inputs=["mask_fuse_unsqueeze1_output"],
                    outputs=["mask_fuse_unsqueeze2_output"],
                    name="Mask_UnSqueeze_2",
                    axes=[2],
                )

                # self.replace_node_input(cast_node, cast_node.input[0], 'mask_fuse_unsqueeze2_output')
                cast_node_2 = onnx.helper.make_node(
                    "Cast",
                    inputs=["mask_fuse_unsqueeze2_output"],
                    outputs=["mask_fuse_cast_output"],
                )
                cast_node_2.attribute.extend([onnx.helper.make_attribute("to", 1)])
                self.replace_node_input(sub_node, sub_node.input[1], "mask_fuse_cast_output")

                nodes_to_remove.extend([slice_node, unsqueeze_node, cast_node])
                self.add_node(unsqueeze_added_1)
                self.add_node(unsqueeze_added_2)
                self.add_node(cast_node_2)

        self.remove_nodes(nodes_to_remove)

        # Prune graph is done after removing nodes to remove island nodes.
        if len(nodes_to_remove) > 0:
            self.prune_graph()

        logger.info("Fused mask" if len(nodes_to_remove) > 0 else "Failed to fuse mask")

    def remove_extra_reshape(self):
        skiplayernorm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        reshape_removed = 0
        for skiplayernorm_node in skiplayernorm_nodes:
            path = self.match_parent_path(
                skiplayernorm_node,
                [
                    "Add",
                    "Reshape",
                    "MatMul",
                    "Reshape",
                    "Gelu",
                    "Add",
                    "Reshape",
                    "MatMul",
                    "SkipLayerNormalization",
                ],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            if path is None:
                continue

            (
                add_1,
                reshape_1,
                matmul_1,
                reshape_2,
                gelu,
                add_2,
                reshape_3,
                matmul_2,
                skiplayernorm,
            ) = path
            add_2.input[0] = matmul_2.output[0]
            self.remove_node(reshape_3)
            matmul_1.input[0] = gelu.output[0]
            self.remove_node(reshape_2)
            add_1.input[0] = matmul_1.output[0]
            self.remove_node(reshape_1)
            reshape_removed += 3

        return reshape_removed

    def remove_extra_reshape_2(self):
        skiplayernorm_nodes = self.get_nodes_by_op_type("SkipLayerNormalization")
        reshape_removed = 0
        for skiplayernorm_node in skiplayernorm_nodes:
            path = self.match_parent_path(
                skiplayernorm_node,
                [
                    "Add",
                    "Reshape",
                    "MatMul",
                    "Reshape",
                    "Gelu",
                    "Add",
                    "Reshape",
                    "MatMul",
                    "Reshape",
                    "SkipLayerNormalization",
                ],
                [None, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            )
            if path is None:
                continue

            (
                add_1,
                reshape_1,
                matmul_1,
                reshape_2,
                gelu,
                add_2,
                reshape_3,
                matmul_2,
                reshape_4,
                skiplayernorm,
            ) = path

            matmul_2.input[0] = skiplayernorm.output[0]
            self.remove_node(reshape_4)

            add_2.input[0] = matmul_2.output[0]
            self.remove_node(reshape_3)

            matmul_1.input[0] = gelu.output[0]
            self.remove_node(reshape_2)

            add_1.input[0] = matmul_1.output[0]
            self.remove_node(reshape_1)

            reshape_removed += 4

        return reshape_removed

    def postprocess(self):
        reshape_removed = self.remove_extra_reshape() + self.remove_extra_reshape_2()
        logger.info(f"Remove {reshape_removed} Reshape nodes.")

        self.prune_graph()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\engines.py ===
# testing/engines.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


from __future__ import annotations

import collections
import re
import typing
from typing import Any
from typing import Dict
from typing import Optional
import warnings
import weakref

from . import config
from .util import decorator
from .util import gc_collect
from .. import event
from .. import pool
from ..util import await_only
from ..util.typing import Literal


if typing.TYPE_CHECKING:
    from ..engine import Engine
    from ..engine.url import URL
    from ..ext.asyncio import AsyncEngine


class ConnectionKiller:
    def __init__(self):
        self.proxy_refs = weakref.WeakKeyDictionary()
        self.testing_engines = collections.defaultdict(set)
        self.dbapi_connections = set()

    def add_pool(self, pool):
        event.listen(pool, "checkout", self._add_conn)
        event.listen(pool, "checkin", self._remove_conn)
        event.listen(pool, "close", self._remove_conn)
        event.listen(pool, "close_detached", self._remove_conn)
        # note we are keeping "invalidated" here, as those are still
        # opened connections we would like to roll back

    def _add_conn(self, dbapi_con, con_record, con_proxy):
        self.dbapi_connections.add(dbapi_con)
        self.proxy_refs[con_proxy] = True

    def _remove_conn(self, dbapi_conn, *arg):
        self.dbapi_connections.discard(dbapi_conn)

    def add_engine(self, engine, scope):
        self.add_pool(engine.pool)

        assert scope in ("class", "global", "function", "fixture")
        self.testing_engines[scope].add(engine)

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:
            warnings.warn(
                "testing_reaper couldn't rollback/close connection: %s" % e
            )

    def rollback_all(self):
        for rec in list(self.proxy_refs):
            if rec is not None and rec.is_valid:
                self._safe(rec.rollback)

    def checkin_all(self):
        # run pool.checkin() for all ConnectionFairy instances we have
        # tracked.

        for rec in list(self.proxy_refs):
            if rec is not None and rec.is_valid:
                self.dbapi_connections.discard(rec.dbapi_connection)
                self._safe(rec._checkin)

        # for fairy refs that were GCed and could not close the connection,
        # such as asyncio, roll back those remaining connections
        for con in self.dbapi_connections:
            self._safe(con.rollback)
        self.dbapi_connections.clear()

    def close_all(self):
        self.checkin_all()

    def prepare_for_drop_tables(self, connection):
        # don't do aggressive checks for third party test suites
        if not config.bootstrapped_as_sqlalchemy:
            return

        from . import provision

        provision.prepare_for_drop_tables(connection.engine.url, connection)

    def _drop_testing_engines(self, scope):
        eng = self.testing_engines[scope]
        for rec in list(eng):
            for proxy_ref in list(self.proxy_refs):
                if proxy_ref is not None and proxy_ref.is_valid:
                    if (
                        proxy_ref._pool is not None
                        and proxy_ref._pool is rec.pool
                    ):
                        self._safe(proxy_ref._checkin)

            if hasattr(rec, "sync_engine"):
                await_only(rec.dispose())
            else:
                rec.dispose()
        eng.clear()

    def after_test(self):
        self._drop_testing_engines("function")

    def after_test_outside_fixtures(self, test):
        # don't do aggressive checks for third party test suites
        if not config.bootstrapped_as_sqlalchemy:
            return

        if test.__class__.__leave_connections_for_teardown__:
            return

        self.checkin_all()

        # on PostgreSQL, this will test for any "idle in transaction"
        # connections.   useful to identify tests with unusual patterns
        # that can't be cleaned up correctly.
        from . import provision

        with config.db.connect() as conn:
            provision.prepare_for_drop_tables(conn.engine.url, conn)

    def stop_test_class_inside_fixtures(self):
        self.checkin_all()
        self._drop_testing_engines("function")
        self._drop_testing_engines("class")

    def stop_test_class_outside_fixtures(self):
        # ensure no refs to checked out connections at all.

        if pool.base._strong_ref_connection_records:
            gc_collect()

            if pool.base._strong_ref_connection_records:
                ln = len(pool.base._strong_ref_connection_records)
                pool.base._strong_ref_connection_records.clear()
                assert (
                    False
                ), "%d connection recs not cleared after test suite" % (ln)

    def final_cleanup(self):
        self.checkin_all()
        for scope in self.testing_engines:
            self._drop_testing_engines(scope)

    def assert_all_closed(self):
        for rec in self.proxy_refs:
            if rec.is_valid:
                assert False


testing_reaper = ConnectionKiller()


@decorator
def assert_conns_closed(fn, *args, **kw):
    try:
        fn(*args, **kw)
    finally:
        testing_reaper.assert_all_closed()


@decorator
def rollback_open_connections(fn, *args, **kw):
    """Decorator that rolls back all open connections after fn execution."""

    try:
        fn(*args, **kw)
    finally:
        testing_reaper.rollback_all()


@decorator
def close_first(fn, *args, **kw):
    """Decorator that closes all connections before fn execution."""

    testing_reaper.checkin_all()
    fn(*args, **kw)


@decorator
def close_open_connections(fn, *args, **kw):
    """Decorator that closes all connections after fn execution."""
    try:
        fn(*args, **kw)
    finally:
        testing_reaper.checkin_all()


def all_dialects(exclude=None):
    import sqlalchemy.dialects as d

    for name in d.__all__:
        # TEMPORARY
        if exclude and name in exclude:
            continue
        mod = getattr(d, name, None)
        if not mod:
            mod = getattr(
                __import__("sqlalchemy.dialects.%s" % name).dialects, name
            )
        yield mod.dialect()


class ReconnectFixture:
    def __init__(self, dbapi):
        self.dbapi = dbapi
        self.connections = []
        self.is_stopped = False

    def __getattr__(self, key):
        return getattr(self.dbapi, key)

    def connect(self, *args, **kwargs):
        conn = self.dbapi.connect(*args, **kwargs)
        if self.is_stopped:
            self._safe(conn.close)
            curs = conn.cursor()  # should fail on Oracle etc.
            # should fail for everything that didn't fail
            # above, connection is closed
            curs.execute("select 1")
            assert False, "simulated connect failure didn't work"
        else:
            self.connections.append(conn)
            return conn

    def _safe(self, fn):
        try:
            fn()
        except Exception as e:
            warnings.warn("ReconnectFixture couldn't close connection: %s" % e)

    def shutdown(self, stop=False):
        # TODO: this doesn't cover all cases
        # as nicely as we'd like, namely MySQLdb.
        # would need to implement R. Brewer's
        # proxy server idea to get better
        # coverage.
        self.is_stopped = stop
        for c in list(self.connections):
            self._safe(c.close)
        self.connections = []

    def restart(self):
        self.is_stopped = False


def reconnecting_engine(url=None, options=None):
    url = url or config.db.url
    dbapi = config.db.dialect.dbapi
    if not options:
        options = {}
    options["module"] = ReconnectFixture(dbapi)
    engine = testing_engine(url, options)
    _dispose = engine.dispose

    def dispose():
        engine.dialect.dbapi.shutdown()
        engine.dialect.dbapi.is_stopped = False
        _dispose()

    engine.test_shutdown = engine.dialect.dbapi.shutdown
    engine.test_restart = engine.dialect.dbapi.restart
    engine.dispose = dispose
    return engine


@typing.overload
def testing_engine(
    url: Optional[URL] = None,
    options: Optional[Dict[str, Any]] = None,
    asyncio: Literal[False] = False,
    transfer_staticpool: bool = False,
) -> Engine: ...


@typing.overload
def testing_engine(
    url: Optional[URL] = None,
    options: Optional[Dict[str, Any]] = None,
    asyncio: Literal[True] = True,
    transfer_staticpool: bool = False,
) -> AsyncEngine: ...


def testing_engine(
    url=None,
    options=None,
    asyncio=False,
    transfer_staticpool=False,
    share_pool=False,
    _sqlite_savepoint=False,
):
    if asyncio:
        assert not _sqlite_savepoint
        from sqlalchemy.ext.asyncio import (
            create_async_engine as create_engine,
        )
    else:
        from sqlalchemy import create_engine
    from sqlalchemy.engine.url import make_url

    if not options:
        use_reaper = True
        scope = "function"
        sqlite_savepoint = False
    else:
        use_reaper = options.pop("use_reaper", True)
        scope = options.pop("scope", "function")
        sqlite_savepoint = options.pop("sqlite_savepoint", False)

    url = url or config.db.url

    url = make_url(url)

    if (
        config.db is None or url.drivername == config.db.url.drivername
    ) and config.db_opts:
        use_options = config.db_opts.copy()
    else:
        use_options = {}

    if options is not None:
        use_options.update(options)

    engine = create_engine(url, **use_options)

    if sqlite_savepoint and engine.name == "sqlite":
        # apply SQLite savepoint workaround
        @event.listens_for(engine, "connect")
        def do_connect(dbapi_connection, connection_record):
            dbapi_connection.isolation_level = None

        @event.listens_for(engine, "begin")
        def do_begin(conn):
            conn.exec_driver_sql("BEGIN")

    if transfer_staticpool:
        from sqlalchemy.pool import StaticPool

        if config.db is not None and isinstance(config.db.pool, StaticPool):
            use_reaper = False
            engine.pool._transfer_from(config.db.pool)
    elif share_pool:
        engine.pool = config.db.pool

    if scope == "global":
        if asyncio:
            engine.sync_engine._has_events = True
        else:
            engine._has_events = (
                True  # enable event blocks, helps with profiling
            )

    if (
        isinstance(engine.pool, pool.QueuePool)
        and "pool" not in use_options
        and "pool_timeout" not in use_options
        and "max_overflow" not in use_options
    ):
        engine.pool._timeout = 0
        engine.pool._max_overflow = 0
    if use_reaper:
        testing_reaper.add_engine(engine, scope)

    return engine


def mock_engine(dialect_name=None):
    """Provides a mocking engine based on the current testing.db.

    This is normally used to test DDL generation flow as emitted
    by an Engine.

    It should not be used in other cases, as assert_compile() and
    assert_sql_execution() are much better choices with fewer
    moving parts.

    """

    from sqlalchemy import create_mock_engine

    if not dialect_name:
        dialect_name = config.db.name

    buffer = []

    def executor(sql, *a, **kw):
        buffer.append(sql)

    def assert_sql(stmts):
        recv = [re.sub(r"[\n\t]", "", str(s)) for s in buffer]
        assert recv == stmts, recv

    def print_sql():
        d = engine.dialect
        return "\n".join(str(s.compile(dialect=d)) for s in engine.mock)

    engine = create_mock_engine(dialect_name + "://", executor)
    assert not hasattr(engine, "mock")
    engine.mock = buffer
    engine.assert_sql = assert_sql
    engine.print_sql = print_sql
    return engine


class DBAPIProxyCursor:
    """Proxy a DBAPI cursor.

    Tests can provide subclasses of this to intercept
    DBAPI-level cursor operations.

    """

    def __init__(self, engine, conn, *args, **kwargs):
        self.engine = engine
        self.connection = conn
        self.cursor = conn.cursor(*args, **kwargs)

    def execute(self, stmt, parameters=None, **kw):
        if parameters:
            return self.cursor.execute(stmt, parameters, **kw)
        else:
            return self.cursor.execute(stmt, **kw)

    def executemany(self, stmt, params, **kw):
        return self.cursor.executemany(stmt, params, **kw)

    def __iter__(self):
        return iter(self.cursor)

    def __getattr__(self, key):
        return getattr(self.cursor, key)


class DBAPIProxyConnection:
    """Proxy a DBAPI connection.

    Tests can provide subclasses of this to intercept
    DBAPI-level connection operations.

    """

    def __init__(self, engine, conn, cursor_cls):
        self.conn = conn
        self.engine = engine
        self.cursor_cls = cursor_cls

    def cursor(self, *args, **kwargs):
        return self.cursor_cls(self.engine, self.conn, *args, **kwargs)

    def close(self):
        self.conn.close()

    def __getattr__(self, key):
        return getattr(self.conn, key)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\tensor_functions.py ===
from math import prod

from sympy.core import S, Integer
from sympy.core.function import DefinedFunction
from sympy.core.logic import fuzzy_not
from sympy.core.relational import Ne
from sympy.core.sorting import default_sort_key
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.piecewise import Piecewise
from sympy.utilities.iterables import has_dups

###############################################################################
###################### Kronecker Delta, Levi-Civita etc. ######################
###############################################################################


def Eijk(*args, **kwargs):
    """
    Represent the Levi-Civita symbol.

    This is a compatibility wrapper to ``LeviCivita()``.

    See Also
    ========

    LeviCivita

    """
    return LeviCivita(*args, **kwargs)


def eval_levicivita(*args):
    """Evaluate Levi-Civita symbol."""
    n = len(args)
    return prod(
        prod(args[j] - args[i] for j in range(i + 1, n))
        / factorial(i) for i in range(n))
    # converting factorial(i) to int is slightly faster


class LeviCivita(DefinedFunction):
    """
    Represent the Levi-Civita symbol.

    Explanation
    ===========

    For even permutations of indices it returns 1, for odd permutations -1, and
    for everything else (a repeated index) it returns 0.

    Thus it represents an alternating pseudotensor.

    Examples
    ========

    >>> from sympy import LeviCivita
    >>> from sympy.abc import i, j, k
    >>> LeviCivita(1, 2, 3)
    1
    >>> LeviCivita(1, 3, 2)
    -1
    >>> LeviCivita(1, 2, 2)
    0
    >>> LeviCivita(i, j, k)
    LeviCivita(i, j, k)
    >>> LeviCivita(i, j, i)
    0

    See Also
    ========

    Eijk

    """

    is_integer = True

    @classmethod
    def eval(cls, *args):
        if all(isinstance(a, (SYMPY_INTS, Integer)) for a in args):
            return eval_levicivita(*args)
        if has_dups(args):
            return S.Zero

    def doit(self, **hints):
        return eval_levicivita(*self.args)


class KroneckerDelta(DefinedFunction):
    """
    The discrete, or Kronecker, delta function.

    Explanation
    ===========

    A function that takes in two integers $i$ and $j$. It returns $0$ if $i$
    and $j$ are not equal, or it returns $1$ if $i$ and $j$ are equal.

    Examples
    ========

    An example with integer indices:

        >>> from sympy import KroneckerDelta
        >>> KroneckerDelta(1, 2)
        0
        >>> KroneckerDelta(3, 3)
        1

    Symbolic indices:

        >>> from sympy.abc import i, j, k
        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)
        >>> KroneckerDelta(i, i)
        1
        >>> KroneckerDelta(i, i + 1)
        0
        >>> KroneckerDelta(i, i + 1 + k)
        KroneckerDelta(i, i + k + 1)

    Parameters
    ==========

    i : Number, Symbol
        The first index of the delta function.
    j : Number, Symbol
        The second index of the delta function.

    See Also
    ========

    eval
    DiracDelta

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Kronecker_delta

    """

    is_integer = True

    @classmethod
    def eval(cls, i, j, delta_range=None):
        """
        Evaluates the discrete delta function.

        Examples
        ========

        >>> from sympy import KroneckerDelta
        >>> from sympy.abc import i, j, k

        >>> KroneckerDelta(i, j)
        KroneckerDelta(i, j)
        >>> KroneckerDelta(i, i)
        1
        >>> KroneckerDelta(i, i + 1)
        0
        >>> KroneckerDelta(i, i + 1 + k)
        KroneckerDelta(i, i + k + 1)

        # indirect doctest

        """

        if delta_range is not None:
            dinf, dsup = delta_range
            if (dinf - i > 0) == True:
                return S.Zero
            if (dinf - j > 0) == True:
                return S.Zero
            if (dsup - i < 0) == True:
                return S.Zero
            if (dsup - j < 0) == True:
                return S.Zero

        diff = i - j
        if diff.is_zero:
            return S.One
        elif fuzzy_not(diff.is_zero):
            return S.Zero

        if i.assumptions0.get("below_fermi") and \
                j.assumptions0.get("above_fermi"):
            return S.Zero
        if j.assumptions0.get("below_fermi") and \
                i.assumptions0.get("above_fermi"):
            return S.Zero
        # to make KroneckerDelta canonical
        # following lines will check if inputs are in order
        # if not, will return KroneckerDelta with correct order
        if default_sort_key(j) < default_sort_key(i):
            if delta_range:
                return cls(j, i, delta_range)
            else:
                return cls(j, i)

    @property
    def delta_range(self):
        if len(self.args) > 2:
            return self.args[2]

    def _eval_power(self, expt):
        if expt.is_positive:
            return self
        if expt.is_negative and expt is not S.NegativeOne:
            return 1/self

    @property
    def is_above_fermi(self):
        """
        True if Delta can be non-zero above fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_above_fermi
        True
        >>> KroneckerDelta(p, i).is_above_fermi
        False
        >>> KroneckerDelta(p, q).is_above_fermi
        True

        See Also
        ========

        is_below_fermi, is_only_below_fermi, is_only_above_fermi

        """
        if self.args[0].assumptions0.get("below_fermi"):
            return False
        if self.args[1].assumptions0.get("below_fermi"):
            return False
        return True

    @property
    def is_below_fermi(self):
        """
        True if Delta can be non-zero below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_below_fermi
        False
        >>> KroneckerDelta(p, i).is_below_fermi
        True
        >>> KroneckerDelta(p, q).is_below_fermi
        True

        See Also
        ========

        is_above_fermi, is_only_above_fermi, is_only_below_fermi

        """
        if self.args[0].assumptions0.get("above_fermi"):
            return False
        if self.args[1].assumptions0.get("above_fermi"):
            return False
        return True

    @property
    def is_only_above_fermi(self):
        """
        True if Delta is restricted to above fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, a).is_only_above_fermi
        True
        >>> KroneckerDelta(p, q).is_only_above_fermi
        False
        >>> KroneckerDelta(p, i).is_only_above_fermi
        False

        See Also
        ========

        is_above_fermi, is_below_fermi, is_only_below_fermi

        """
        return ( self.args[0].assumptions0.get("above_fermi")
                or
                self.args[1].assumptions0.get("above_fermi")
                ) or False

    @property
    def is_only_below_fermi(self):
        """
        True if Delta is restricted to below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, i).is_only_below_fermi
        True
        >>> KroneckerDelta(p, q).is_only_below_fermi
        False
        >>> KroneckerDelta(p, a).is_only_below_fermi
        False

        See Also
        ========

        is_above_fermi, is_below_fermi, is_only_above_fermi

        """
        return ( self.args[0].assumptions0.get("below_fermi")
                or
                self.args[1].assumptions0.get("below_fermi")
                ) or False

    @property
    def indices_contain_equal_information(self):
        """
        Returns True if indices are either both above or below fermi.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> p = Symbol('p')
        >>> q = Symbol('q')
        >>> KroneckerDelta(p, q).indices_contain_equal_information
        True
        >>> KroneckerDelta(p, q+1).indices_contain_equal_information
        True
        >>> KroneckerDelta(i, p).indices_contain_equal_information
        False

        """
        if (self.args[0].assumptions0.get("below_fermi") and
                self.args[1].assumptions0.get("below_fermi")):
            return True
        if (self.args[0].assumptions0.get("above_fermi")
                and self.args[1].assumptions0.get("above_fermi")):
            return True

        # if both indices are general we are True, else false
        return self.is_below_fermi and self.is_above_fermi

    @property
    def preferred_index(self):
        """
        Returns the index which is preferred to keep in the final expression.

        Explanation
        ===========

        The preferred index is the index with more information regarding fermi
        level. If indices contain the same information, 'a' is preferred before
        'b'.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> j = Symbol('j', below_fermi=True)
        >>> p = Symbol('p')
        >>> KroneckerDelta(p, i).preferred_index
        i
        >>> KroneckerDelta(p, a).preferred_index
        a
        >>> KroneckerDelta(i, j).preferred_index
        i

        See Also
        ========

        killable_index

        """
        if self._get_preferred_index():
            return self.args[1]
        else:
            return self.args[0]

    @property
    def killable_index(self):
        """
        Returns the index which is preferred to substitute in the final
        expression.

        Explanation
        ===========

        The index to substitute is the index with less information regarding
        fermi level. If indices contain the same information, 'a' is preferred
        before 'b'.

        Examples
        ========

        >>> from sympy import KroneckerDelta, Symbol
        >>> a = Symbol('a', above_fermi=True)
        >>> i = Symbol('i', below_fermi=True)
        >>> j = Symbol('j', below_fermi=True)
        >>> p = Symbol('p')
        >>> KroneckerDelta(p, i).killable_index
        p
        >>> KroneckerDelta(p, a).killable_index
        p
        >>> KroneckerDelta(i, j).killable_index
        j

        See Also
        ========

        preferred_index

        """
        if self._get_preferred_index():
            return self.args[0]
        else:
            return self.args[1]

    def _get_preferred_index(self):
        """
        Returns the index which is preferred to keep in the final expression.

        The preferred index is the index with more information regarding fermi
        level. If indices contain the same information, index 0 is returned.

        """
        if not self.is_above_fermi:
            if self.args[0].assumptions0.get("below_fermi"):
                return 0
            else:
                return 1
        elif not self.is_below_fermi:
            if self.args[0].assumptions0.get("above_fermi"):
                return 0
            else:
                return 1
        else:
            return 0

    @property
    def indices(self):
        return self.args[0:2]

    def _eval_rewrite_as_Piecewise(self, *args, **kwargs):
        i, j = args
        return Piecewise((0, Ne(i, j)), (1, True))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\linearize.py ===
__all__ = ['Linearizer']

from sympy import Matrix, eye, zeros
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import flatten
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics.functions import msubs, _parse_linear_solver

from collections import namedtuple
from collections.abc import Iterable


class Linearizer:
    """This object holds the general model form for a dynamic system. This
    model is used for computing the linearized form of the system, while
    properly dealing with constraints leading to  dependent coordinates and
    speeds. The notation and method is described in [1]_.

    Attributes
    ==========

    f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a : Matrix
        Matrices holding the general system form.
    q, u, r : Matrix
        Matrices holding the generalized coordinates, speeds, and
        input vectors.
    q_i, u_i : Matrix
        Matrices of the independent generalized coordinates and speeds.
    q_d, u_d : Matrix
        Matrices of the dependent generalized coordinates and speeds.
    perm_mat : Matrix
        Permutation matrix such that [q_ind, u_ind]^T = perm_mat*[q, u]^T

    References
    ==========

    .. [1] D. L. Peterson, G. Gede, and M. Hubbard, "Symbolic linearization of
           equations of motion of constrained multibody systems," Multibody
           Syst Dyn, vol. 33, no. 2, pp. 143-161, Feb. 2015, doi:
           10.1007/s11044-014-9436-5.

    """

    def __init__(self, f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i=None,
                 q_d=None, u_i=None, u_d=None, r=None, lams=None,
                 linear_solver='LU'):
        """
        Parameters
        ==========

        f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a : array_like
            System of equations holding the general system form.
            Supply empty array or Matrix if the parameter
            does not exist.
        q : array_like
            The generalized coordinates.
        u : array_like
            The generalized speeds
        q_i, u_i : array_like, optional
            The independent generalized coordinates and speeds.
        q_d, u_d : array_like, optional
            The dependent generalized coordinates and speeds.
        r : array_like, optional
            The input variables.
        lams : array_like, optional
            The lagrange multipliers
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.

        """
        self.linear_solver = _parse_linear_solver(linear_solver)

        # Generalized equation form
        self.f_0 = Matrix(f_0)
        self.f_1 = Matrix(f_1)
        self.f_2 = Matrix(f_2)
        self.f_3 = Matrix(f_3)
        self.f_4 = Matrix(f_4)
        self.f_c = Matrix(f_c)
        self.f_v = Matrix(f_v)
        self.f_a = Matrix(f_a)

        # Generalized equation variables
        self.q = Matrix(q)
        self.u = Matrix(u)
        none_handler = lambda x: Matrix(x) if x else Matrix()
        self.q_i = none_handler(q_i)
        self.q_d = none_handler(q_d)
        self.u_i = none_handler(u_i)
        self.u_d = none_handler(u_d)
        self.r = none_handler(r)
        self.lams = none_handler(lams)

        # Derivatives of generalized equation variables
        self._qd = self.q.diff(dynamicsymbols._t)
        self._ud = self.u.diff(dynamicsymbols._t)
        # If the user doesn't actually use generalized variables, and the
        # qd and u vectors have any intersecting variables, this can cause
        # problems. We'll fix this with some hackery, and Dummy variables
        dup_vars = set(self._qd).intersection(self.u)
        self._qd_dup = Matrix([var if var not in dup_vars else Dummy() for var
                               in self._qd])

        # Derive dimension terms
        l = len(self.f_c)
        m = len(self.f_v)
        n = len(self.q)
        o = len(self.u)
        s = len(self.r)
        k = len(self.lams)
        dims = namedtuple('dims', ['l', 'm', 'n', 'o', 's', 'k'])
        self._dims = dims(l, m, n, o, s, k)

        self._Pq = None
        self._Pqi = None
        self._Pqd = None
        self._Pu = None
        self._Pui = None
        self._Pud = None
        self._C_0 = None
        self._C_1 = None
        self._C_2 = None
        self.perm_mat = None

        self._setup_done = False

    def _setup(self):
        # Calculations here only need to be run once. They are moved out of
        # the __init__ method to increase the speed of Linearizer creation.
        self._form_permutation_matrices()
        self._form_block_matrices()
        self._form_coefficient_matrices()
        self._setup_done = True

    def _form_permutation_matrices(self):
        """Form the permutation matrices Pq and Pu."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Compute permutation matrices
        if n != 0:
            self._Pq = permutation_matrix(self.q, Matrix([self.q_i, self.q_d]))
            if l > 0:
                self._Pqi = self._Pq[:, :-l]
                self._Pqd = self._Pq[:, -l:]
            else:
                self._Pqi = self._Pq
                self._Pqd = Matrix()
        if o != 0:
            self._Pu = permutation_matrix(self.u, Matrix([self.u_i, self.u_d]))
            if m > 0:
                self._Pui = self._Pu[:, :-m]
                self._Pud = self._Pu[:, -m:]
            else:
                self._Pui = self._Pu
                self._Pud = Matrix()
        # Compute combination permutation matrix for computing A and B
        P_col1 = Matrix([self._Pqi, zeros(o + k, n - l)])
        P_col2 = Matrix([zeros(n, o - m), self._Pui, zeros(k, o - m)])
        if P_col1:
            if P_col2:
                self.perm_mat = P_col1.row_join(P_col2)
            else:
                self.perm_mat = P_col1
        else:
            self.perm_mat = P_col2

    def _form_coefficient_matrices(self):
        """Form the coefficient matrices C_0, C_1, and C_2."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Build up the coefficient matrices C_0, C_1, and C_2
        # If there are configuration constraints (l > 0), form C_0 as normal.
        # If not, C_0 is I_(nxn). Note that this works even if n=0
        if l > 0:
            f_c_jac_q = self.f_c.jacobian(self.q)
            self._C_0 = (eye(n) - self._Pqd *
                         self.linear_solver(f_c_jac_q*self._Pqd,
                                            f_c_jac_q))*self._Pqi
        else:
            self._C_0 = eye(n)
        # If there are motion constraints (m > 0), form C_1 and C_2 as normal.
        # If not, C_1 is 0, and C_2 is I_(oxo). Note that this works even if
        # o = 0.
        if m > 0:
            f_v_jac_u = self.f_v.jacobian(self.u)
            temp = f_v_jac_u * self._Pud
            if n != 0:
                f_v_jac_q = self.f_v.jacobian(self.q)
                self._C_1 = -self._Pud * self.linear_solver(temp, f_v_jac_q)
            else:
                self._C_1 = zeros(o, n)
            self._C_2 = (eye(o) - self._Pud *
                         self.linear_solver(temp, f_v_jac_u))*self._Pui
        else:
            self._C_1 = zeros(o, n)
            self._C_2 = eye(o)

    def _form_block_matrices(self):
        """Form the block matrices for composing M, A, and B."""

        # Extract dimension variables
        l, m, n, o, s, k = self._dims
        # Block Matrix Definitions. These are only defined if under certain
        # conditions. If undefined, an empty matrix is used instead
        if n != 0:
            self._M_qq = self.f_0.jacobian(self._qd)
            self._A_qq = -(self.f_0 + self.f_1).jacobian(self.q)
        else:
            self._M_qq = Matrix()
            self._A_qq = Matrix()
        if n != 0 and m != 0:
            self._M_uqc = self.f_a.jacobian(self._qd_dup)
            self._A_uqc = -self.f_a.jacobian(self.q)
        else:
            self._M_uqc = Matrix()
            self._A_uqc = Matrix()
        if n != 0 and o - m + k != 0:
            self._M_uqd = self.f_3.jacobian(self._qd_dup)
            self._A_uqd = -(self.f_2 + self.f_3 + self.f_4).jacobian(self.q)
        else:
            self._M_uqd = Matrix()
            self._A_uqd = Matrix()
        if o != 0 and m != 0:
            self._M_uuc = self.f_a.jacobian(self._ud)
            self._A_uuc = -self.f_a.jacobian(self.u)
        else:
            self._M_uuc = Matrix()
            self._A_uuc = Matrix()
        if o != 0 and o - m + k != 0:
            self._M_uud = self.f_2.jacobian(self._ud)
            self._A_uud = -(self.f_2 + self.f_3).jacobian(self.u)
        else:
            self._M_uud = Matrix()
            self._A_uud = Matrix()
        if o != 0 and n != 0:
            self._A_qu = -self.f_1.jacobian(self.u)
        else:
            self._A_qu = Matrix()
        if k != 0 and o - m + k != 0:
            self._M_uld = self.f_4.jacobian(self.lams)
        else:
            self._M_uld = Matrix()
        if s != 0 and o - m + k != 0:
            self._B_u = -self.f_3.jacobian(self.r)
        else:
            self._B_u = Matrix()

    def linearize(self, op_point=None, A_and_B=False, simplify=False):
        """Linearize the system about the operating point. Note that
        q_op, u_op, qd_op, ud_op must satisfy the equations of motion.
        These may be either symbolic or numeric.

        Parameters
        ==========
        op_point : dict or iterable of dicts, optional
            Dictionary or iterable of dictionaries containing the operating
            point conditions for all or a subset of the generalized
            coordinates, generalized speeds, and time derivatives of the
            generalized speeds. These will be substituted into the linearized
            system before the linearization is complete. Leave set to ``None``
            if you want the operating point to be an arbitrary set of symbols.
            Note that any reduction in symbols (whether substituted for numbers
            or expressions with a common parameter) will result in faster
            runtime.
        A_and_B : bool, optional
            If A_and_B=False (default), (M, A, B) is returned and of
            A_and_B=True, (A, B) is returned. See below.
        simplify : bool, optional
            Determines if returned values are simplified before return.
            For large expressions this may be time consuming. Default is False.

        Returns
        =======
        M, A, B : Matrices, ``A_and_B=False``
            Matrices from the implicit form:
                ``[M]*[q', u']^T = [A]*[q_ind, u_ind]^T + [B]*r``
        A, B : Matrices, ``A_and_B=True``
            Matrices from the explicit form:
                ``[q_ind', u_ind']^T = [A]*[q_ind, u_ind]^T + [B]*r``

        Notes
        =====

        Note that the process of solving with A_and_B=True is computationally
        intensive if there are many symbolic parameters. For this reason, it
        may be more desirable to use the default A_and_B=False, returning M, A,
        and B. More values may then be substituted in to these matrices later
        on. The state space form can then be found as A = P.T*M.LUsolve(A), B =
        P.T*M.LUsolve(B), where P = Linearizer.perm_mat.

        """

        # Run the setup if needed:
        if not self._setup_done:
            self._setup()

        # Compose dict of operating conditions
        if isinstance(op_point, dict):
            op_point_dict = op_point
        elif isinstance(op_point, Iterable):
            op_point_dict = {}
            for op in op_point:
                op_point_dict.update(op)
        else:
            op_point_dict = {}

        # Extract dimension variables
        l, m, n, o, s, k = self._dims

        # Rename terms to shorten expressions
        M_qq = self._M_qq
        M_uqc = self._M_uqc
        M_uqd = self._M_uqd
        M_uuc = self._M_uuc
        M_uud = self._M_uud
        M_uld = self._M_uld
        A_qq = self._A_qq
        A_uqc = self._A_uqc
        A_uqd = self._A_uqd
        A_qu = self._A_qu
        A_uuc = self._A_uuc
        A_uud = self._A_uud
        B_u = self._B_u
        C_0 = self._C_0
        C_1 = self._C_1
        C_2 = self._C_2

        # Build up Mass Matrix
        #     |M_qq    0_nxo   0_nxk|
        # M = |M_uqc   M_uuc   0_mxk|
        #     |M_uqd   M_uud   M_uld|
        if o != 0:
            col2 = Matrix([zeros(n, o), M_uuc, M_uud])
        if k != 0:
            col3 = Matrix([zeros(n + m, k), M_uld])
        if n != 0:
            col1 = Matrix([M_qq, M_uqc, M_uqd])
            if o != 0 and k != 0:
                M = col1.row_join(col2).row_join(col3)
            elif o != 0:
                M = col1.row_join(col2)
            else:
                M = col1
        elif k != 0:
            M = col2.row_join(col3)
        else:
            M = col2
        M_eq = msubs(M, op_point_dict)

        # Build up state coefficient matrix A
        #     |(A_qq + A_qu*C_1)*C_0       A_qu*C_2|
        # A = |(A_uqc + A_uuc*C_1)*C_0    A_uuc*C_2|
        #     |(A_uqd + A_uud*C_1)*C_0    A_uud*C_2|
        # Col 1 is only defined if n != 0
        if n != 0:
            r1c1 = A_qq
            if o != 0:
                r1c1 += (A_qu * C_1)
            r1c1 = r1c1 * C_0
            if m != 0:
                r2c1 = A_uqc
                if o != 0:
                    r2c1 += (A_uuc * C_1)
                r2c1 = r2c1 * C_0
            else:
                r2c1 = Matrix()
            if o - m + k != 0:
                r3c1 = A_uqd
                if o != 0:
                    r3c1 += (A_uud * C_1)
                r3c1 = r3c1 * C_0
            else:
                r3c1 = Matrix()
            col1 = Matrix([r1c1, r2c1, r3c1])
        else:
            col1 = Matrix()
        # Col 2 is only defined if o != 0
        if o != 0:
            if n != 0:
                r1c2 = A_qu * C_2
            else:
                r1c2 = Matrix()
            if m != 0:
                r2c2 = A_uuc * C_2
            else:
                r2c2 = Matrix()
            if o - m + k != 0:
                r3c2 = A_uud * C_2
            else:
                r3c2 = Matrix()
            col2 = Matrix([r1c2, r2c2, r3c2])
        else:
            col2 = Matrix()
        if col1:
            if col2:
                Amat = col1.row_join(col2)
            else:
                Amat = col1
        else:
            Amat = col2
        Amat_eq = msubs(Amat, op_point_dict)

        # Build up the B matrix if there are forcing variables
        #     |0_(n + m)xs|
        # B = |B_u        |
        if s != 0 and o - m + k != 0:
            Bmat = zeros(n + m, s).col_join(B_u)
            Bmat_eq = msubs(Bmat, op_point_dict)
        else:
            Bmat_eq = Matrix()

        # kwarg A_and_B indicates to return  A, B for forming the equation
        # dx = [A]x + [B]r, where x = [q_indnd, u_indnd]^T,
        if A_and_B:
            A_cont = self.perm_mat.T * self.linear_solver(M_eq, Amat_eq)
            if Bmat_eq:
                B_cont = self.perm_mat.T * self.linear_solver(M_eq, Bmat_eq)
            else:
                # Bmat = Matrix([]), so no need to sub
                B_cont = Bmat_eq
            if simplify:
                A_cont.simplify()
                B_cont.simplify()
            return A_cont, B_cont
        # Otherwise return M, A, B for forming the equation
        # [M]dx = [A]x + [B]r, where x = [q, u]^T
        else:
            if simplify:
                M_eq.simplify()
                Amat_eq.simplify()
                Bmat_eq.simplify()
            return M_eq, Amat_eq, Bmat_eq


def permutation_matrix(orig_vec, per_vec):
    """Compute the permutation matrix to change order of
    orig_vec into order of per_vec.

    Parameters
    ==========

    orig_vec : array_like
        Symbols in original ordering.
    per_vec : array_like
        Symbols in new ordering.

    Returns
    =======

    p_matrix : Matrix
        Permutation matrix such that orig_vec == (p_matrix * per_vec).
    """
    if not isinstance(orig_vec, (list, tuple)):
        orig_vec = flatten(orig_vec)
    if not isinstance(per_vec, (list, tuple)):
        per_vec = flatten(per_vec)
    if set(orig_vec) != set(per_vec):
        raise ValueError("orig_vec and per_vec must be the same length, "
                         "and contain the same symbols.")
    ind_list = [orig_vec.index(i) for i in per_vec]
    p_matrix = zeros(len(orig_vec))
    for i, j in enumerate(ind_list):
        p_matrix[i, j] = 1
    return p_matrix

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\utilities.py ===
"""Utilities for algebraic number theory. """

from sympy.core.sympify import sympify
from sympy.ntheory.factor_ import factorint
from sympy.polys.domains.rationalfield import QQ
from sympy.polys.domains.integerring import ZZ
from sympy.polys.matrices.exceptions import DMRankError
from sympy.polys.numberfields.minpoly import minpoly
from sympy.printing.lambdarepr import IntervalPrinter
from sympy.utilities.decorator import public
from sympy.utilities.lambdify import lambdify

from mpmath import mp


def is_rat(c):
    r"""
    Test whether an argument is of an acceptable type to be used as a rational
    number.

    Explanation
    ===========

    Returns ``True`` on any argument of type ``int``, :ref:`ZZ`, or :ref:`QQ`.

    See Also
    ========

    is_int

    """
    # ``c in QQ`` is too accepting (e.g. ``3.14 in QQ`` is ``True``),
    # ``QQ.of_type(c)`` is too demanding (e.g. ``QQ.of_type(3)`` is ``False``).
    #
    # Meanwhile, if gmpy2 is installed then ``ZZ.of_type()`` accepts only
    # ``mpz``, not ``int``, so we need another clause to ensure ``int`` is
    # accepted.
    return isinstance(c, int) or ZZ.of_type(c) or QQ.of_type(c)


def is_int(c):
    r"""
    Test whether an argument is of an acceptable type to be used as an integer.

    Explanation
    ===========

    Returns ``True`` on any argument of type ``int`` or :ref:`ZZ`.

    See Also
    ========

    is_rat

    """
    # If gmpy2 is installed then ``ZZ.of_type()`` accepts only
    # ``mpz``, not ``int``, so we need another clause to ensure ``int`` is
    # accepted.
    return isinstance(c, int) or ZZ.of_type(c)


def get_num_denom(c):
    r"""
    Given any argument on which :py:func:`~.is_rat` is ``True``, return the
    numerator and denominator of this number.

    See Also
    ========

    is_rat

    """
    r = QQ(c)
    return r.numerator, r.denominator


@public
def extract_fundamental_discriminant(a):
    r"""
    Extract a fundamental discriminant from an integer *a*.

    Explanation
    ===========

    Given any rational integer *a* that is 0 or 1 mod 4, write $a = d f^2$,
    where $d$ is either 1 or a fundamental discriminant, and return a pair
    of dictionaries ``(D, F)`` giving the prime factorizations of $d$ and $f$
    respectively, in the same format returned by :py:func:`~.factorint`.

    A fundamental discriminant $d$ is different from unity, and is either
    1 mod 4 and squarefree, or is 0 mod 4 and such that $d/4$ is squarefree
    and 2 or 3 mod 4. This is the same as being the discriminant of some
    quadratic field.

    Examples
    ========

    >>> from sympy.polys.numberfields.utilities import extract_fundamental_discriminant
    >>> print(extract_fundamental_discriminant(-432))
    ({3: 1, -1: 1}, {2: 2, 3: 1})

    For comparison:

    >>> from sympy import factorint
    >>> print(factorint(-432))
    {2: 4, 3: 3, -1: 1}

    Parameters
    ==========

    a: int, must be 0 or 1 mod 4

    Returns
    =======

    Pair ``(D, F)``  of dictionaries.

    Raises
    ======

    ValueError
        If *a* is not 0 or 1 mod 4.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Prop. 5.1.3)

    """
    if a % 4 not in [0, 1]:
        raise ValueError('To extract fundamental discriminant, number must be 0 or 1 mod 4.')
    if a == 0:
        return {}, {0: 1}
    if a == 1:
        return {}, {}
    a_factors = factorint(a)
    D = {}
    F = {}
    # First pass: just make d squarefree, and a/d a perfect square.
    # We'll count primes (and units! i.e. -1) that are 3 mod 4 and present in d.
    num_3_mod_4 = 0
    for p, e in a_factors.items():
        if e % 2 == 1:
            D[p] = 1
            if p % 4 == 3:
                num_3_mod_4 += 1
            if e >= 3:
                F[p] = (e - 1) // 2
        else:
            F[p] = e // 2
    # Second pass: if d is cong. to 2 or 3 mod 4, then we must steal away
    # another factor of 4 from f**2 and give it to d.
    even = 2 in D
    if even or num_3_mod_4 % 2 == 1:
        e2 = F[2]
        assert e2 > 0
        if e2 == 1:
            del F[2]
        else:
            F[2] = e2 - 1
        D[2] = 3 if even else 2
    return D, F


@public
class AlgIntPowers:
    r"""
    Compute the powers of an algebraic integer.

    Explanation
    ===========

    Given an algebraic integer $\theta$ by its monic irreducible polynomial
    ``T`` over :ref:`ZZ`, this class computes representations of arbitrarily
    high powers of $\theta$, as :ref:`ZZ`-linear combinations over
    $\{1, \theta, \ldots, \theta^{n-1}\}$, where $n = \deg(T)$.

    The representations are computed using the linear recurrence relations for
    powers of $\theta$, derived from the polynomial ``T``. See [1], Sec. 4.2.2.

    Optionally, the representations may be reduced with respect to a modulus.

    Examples
    ========

    >>> from sympy import Poly, cyclotomic_poly
    >>> from sympy.polys.numberfields.utilities import AlgIntPowers
    >>> T = Poly(cyclotomic_poly(5))
    >>> zeta_pow = AlgIntPowers(T)
    >>> print(zeta_pow[0])
    [1, 0, 0, 0]
    >>> print(zeta_pow[1])
    [0, 1, 0, 0]
    >>> print(zeta_pow[4])  # doctest: +SKIP
    [-1, -1, -1, -1]
    >>> print(zeta_pow[24])  # doctest: +SKIP
    [-1, -1, -1, -1]

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*

    """

    def __init__(self, T, modulus=None):
        """
        Parameters
        ==========

        T : :py:class:`~.Poly`
            The monic irreducible polynomial over :ref:`ZZ` defining the
            algebraic integer.

        modulus : int, None, optional
            If not ``None``, all representations will be reduced w.r.t. this.

        """
        self.T = T
        self.modulus = modulus
        self.n = T.degree()
        self.powers_n_and_up = [[-c % self for c in reversed(T.rep.to_list())][:-1]]
        self.max_so_far = self.n

    def red(self, exp):
        return exp if self.modulus is None else exp % self.modulus

    def __rmod__(self, other):
        return self.red(other)

    def compute_up_through(self, e):
        m = self.max_so_far
        if e <= m: return
        n = self.n
        r = self.powers_n_and_up
        c = r[0]
        for k in range(m+1, e+1):
            b = r[k-1-n][n-1]
            r.append(
                [c[0]*b % self] + [
                    (r[k-1-n][i-1] + c[i]*b) % self for i in range(1, n)
                ]
            )
        self.max_so_far = e

    def get(self, e):
        n = self.n
        if e < 0:
            raise ValueError('Exponent must be non-negative.')
        elif e < n:
            return [1 if i == e else 0 for i in range(n)]
        else:
            self.compute_up_through(e)
            return self.powers_n_and_up[e - n]

    def __getitem__(self, item):
        return self.get(item)


@public
def coeff_search(m, R):
    r"""
    Generate coefficients for searching through polynomials.

    Explanation
    ===========

    Lead coeff is always non-negative. Explore all combinations with coeffs
    bounded in absolute value before increasing the bound. Skip the all-zero
    list, and skip any repeats. See examples.

    Examples
    ========

    >>> from sympy.polys.numberfields.utilities import coeff_search
    >>> cs = coeff_search(2, 1)
    >>> C = [next(cs) for i in range(13)]
    >>> print(C)
    [[1, 1], [1, 0], [1, -1], [0, 1], [2, 2], [2, 1], [2, 0], [2, -1], [2, -2],
     [1, 2], [1, -2], [0, 2], [3, 3]]

    Parameters
    ==========

    m : int
        Length of coeff list.
    R : int
        Initial max abs val for coeffs (will increase as search proceeds).

    Returns
    =======

    generator
        Infinite generator of lists of coefficients.

    """
    R0 = R
    c = [R] * m
    while True:
        if R == R0 or R in c or -R in c:
            yield c[:]
        j = m - 1
        while c[j] == -R:
            j -= 1
        c[j] -= 1
        for i in range(j + 1, m):
            c[i] = R
        for j in range(m):
            if c[j] != 0:
                break
        else:
            R += 1
            c = [R] * m


def supplement_a_subspace(M):
    r"""
    Extend a basis for a subspace to a basis for the whole space.

    Explanation
    ===========

    Given an $n \times r$ matrix *M* of rank $r$ (so $r \leq n$), this function
    computes an invertible $n \times n$ matrix $B$ such that the first $r$
    columns of $B$ equal *M*.

    This operation can be interpreted as a way of extending a basis for a
    subspace, to give a basis for the whole space.

    To be precise, suppose you have an $n$-dimensional vector space $V$, with
    basis $\{v_1, v_2, \ldots, v_n\}$, and an $r$-dimensional subspace $W$ of
    $V$, spanned by a basis $\{w_1, w_2, \ldots, w_r\}$, where the $w_j$ are
    given as linear combinations of the $v_i$. If the columns of *M* represent
    the $w_j$ as such linear combinations, then the columns of the matrix $B$
    computed by this function give a new basis $\{u_1, u_2, \ldots, u_n\}$ for
    $V$, again relative to the $\{v_i\}$ basis, and such that $u_j = w_j$
    for $1 \leq j \leq r$.

    Examples
    ========

    Note: The function works in terms of columns, so in these examples we
    print matrix transposes in order to make the columns easier to inspect.

    >>> from sympy.polys.matrices import DM
    >>> from sympy import QQ, FF
    >>> from sympy.polys.numberfields.utilities import supplement_a_subspace
    >>> M = DM([[1, 7, 0], [2, 3, 4]], QQ).transpose()
    >>> print(supplement_a_subspace(M).to_Matrix().transpose())
    Matrix([[1, 7, 0], [2, 3, 4], [1, 0, 0]])

    >>> M2 = M.convert_to(FF(7))
    >>> print(M2.to_Matrix().transpose())
    Matrix([[1, 0, 0], [2, 3, -3]])
    >>> print(supplement_a_subspace(M2).to_Matrix().transpose())
    Matrix([[1, 0, 0], [2, 3, -3], [0, 1, 0]])

    Parameters
    ==========

    M : :py:class:`~.DomainMatrix`
        The columns give the basis for the subspace.

    Returns
    =======

    :py:class:`~.DomainMatrix`
        This matrix is invertible and its first $r$ columns equal *M*.

    Raises
    ======

    DMRankError
        If *M* was not of maximal rank.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory*
       (See Sec. 2.3.2.)

    """
    n, r = M.shape
    # Let In be the n x n identity matrix.
    # Form the augmented matrix [M | In] and compute RREF.
    Maug = M.hstack(M.eye(n, M.domain))
    R, pivots = Maug.rref()
    if pivots[:r] != tuple(range(r)):
        raise DMRankError('M was not of maximal rank')
    # Let J be the n x r matrix equal to the first r columns of In.
    # Since M is of rank r, RREF reduces [M | In] to [J | A], where A is the product of
    # elementary matrices Ei corresp. to the row ops performed by RREF. Since the Ei are
    # invertible, so is A. Let B = A^(-1).
    A = R[:, r:]
    B = A.inv()
    # Then B is the desired matrix. It is invertible, since B^(-1) == A.
    # And A * [M | In] == [J | A]
    #  => A * M == J
    #  => M == B * J == the first r columns of B.
    return B


@public
def isolate(alg, eps=None, fast=False):
    """
    Find a rational isolating interval for a real algebraic number.

    Examples
    ========

    >>> from sympy import isolate, sqrt, Rational
    >>> print(isolate(sqrt(2)))  # doctest: +SKIP
    (1, 2)
    >>> print(isolate(sqrt(2), eps=Rational(1, 100)))
    (24/17, 17/12)

    Parameters
    ==========

    alg : str, int, :py:class:`~.Expr`
        The algebraic number to be isolated. Must be a real number, to use this
        particular function. However, see also :py:meth:`.Poly.intervals`,
        which isolates complex roots when you pass ``all=True``.
    eps : positive element of :ref:`QQ`, None, optional (default=None)
        Precision to be passed to :py:meth:`.Poly.refine_root`
    fast : boolean, optional (default=False)
        Say whether fast refinement procedure should be used.
        (Will be passed to :py:meth:`.Poly.refine_root`.)

    Returns
    =======

    Pair of rational numbers defining an isolating interval for the given
    algebraic number.

    See Also
    ========

    .Poly.intervals

    """
    alg = sympify(alg)

    if alg.is_Rational:
        return (alg, alg)
    elif not alg.is_real:
        raise NotImplementedError(
            "complex algebraic numbers are not supported")

    func = lambdify((), alg, modules="mpmath", printer=IntervalPrinter())

    poly = minpoly(alg, polys=True)
    intervals = poly.intervals(sqf=True)

    dps, done = mp.dps, False

    try:
        while not done:
            alg = func()

            for a, b in intervals:
                if a <= alg.a and alg.b <= b:
                    done = True
                    break
            else:
                mp.dps *= 2
    finally:
        mp.dps = dps

    if eps is not None:
        a, b = poly.refine_root(a, b, eps=eps, fast=fast)

    return (a, b)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\arrays\arrow\accessors.py ===
"""Accessors for arrow-backed data."""

from __future__ import annotations

from abc import (
    ABCMeta,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    cast,
)

from pandas.compat import (
    pa_version_under10p1,
    pa_version_under11p0,
)

from pandas.core.dtypes.common import is_list_like

if not pa_version_under10p1:
    import pyarrow as pa
    import pyarrow.compute as pc

    from pandas.core.dtypes.dtypes import ArrowDtype

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas import (
        DataFrame,
        Series,
    )


class ArrowAccessor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, data, validation_msg: str) -> None:
        self._data = data
        self._validation_msg = validation_msg
        self._validate(data)

    @abstractmethod
    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        pass

    def _validate(self, data):
        dtype = data.dtype
        if pa_version_under10p1 or not isinstance(dtype, ArrowDtype):
            # Raise AttributeError so that inspect can handle non-struct Series.
            raise AttributeError(self._validation_msg.format(dtype=dtype))

        if not self._is_valid_pyarrow_dtype(dtype.pyarrow_dtype):
            # Raise AttributeError so that inspect can handle invalid Series.
            raise AttributeError(self._validation_msg.format(dtype=dtype))

    @property
    def _pa_array(self):
        return self._data.array._pa_array


class ListAccessor(ArrowAccessor):
    """
    Accessor object for list data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow list data.
    """

    def __init__(self, data=None) -> None:
        super().__init__(
            data,
            validation_msg="Can only use the '.list' accessor with "
            "'list[pyarrow]' dtype, not {dtype}.",
        )

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        return (
            pa.types.is_list(pyarrow_dtype)
            or pa.types.is_fixed_size_list(pyarrow_dtype)
            or pa.types.is_large_list(pyarrow_dtype)
        )

    def len(self) -> Series:
        """
        Return the length of each list in the Series.

        Returns
        -------
        pandas.Series
            The length of each list.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(
        ...         pa.int64()
        ...     ))
        ... )
        >>> s.list.len()
        0    3
        1    1
        dtype: int32[pyarrow]
        """
        from pandas import Series

        value_lengths = pc.list_value_length(self._pa_array)
        return Series(value_lengths, dtype=ArrowDtype(value_lengths.type))

    def __getitem__(self, key: int | slice) -> Series:
        """
        Index or slice lists in the Series.

        Parameters
        ----------
        key : int | slice
            Index or slice of indices to access from each list.

        Returns
        -------
        pandas.Series
            The list at requested index.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(
        ...         pa.int64()
        ...     ))
        ... )
        >>> s.list[0]
        0    1
        1    3
        dtype: int64[pyarrow]
        """
        from pandas import Series

        if isinstance(key, int):
            # TODO: Support negative key but pyarrow does not allow
            # element index to be an array.
            # if key < 0:
            #     key = pc.add(key, pc.list_value_length(self._pa_array))
            element = pc.list_element(self._pa_array, key)
            return Series(element, dtype=ArrowDtype(element.type))
        elif isinstance(key, slice):
            if pa_version_under11p0:
                raise NotImplementedError(
                    f"List slice not supported by pyarrow {pa.__version__}."
                )

            # TODO: Support negative start/stop/step, ideally this would be added
            # upstream in pyarrow.
            start, stop, step = key.start, key.stop, key.step
            if start is None:
                # TODO: When adding negative step support
                #  this should be setto last element of array
                # when step is negative.
                start = 0
            if step is None:
                step = 1
            sliced = pc.list_slice(self._pa_array, start, stop, step)
            return Series(sliced, dtype=ArrowDtype(sliced.type))
        else:
            raise ValueError(f"key must be an int or slice, got {type(key).__name__}")

    def __iter__(self) -> Iterator:
        raise TypeError(f"'{type(self).__name__}' object is not iterable")

    def flatten(self) -> Series:
        """
        Flatten list values.

        Returns
        -------
        pandas.Series
            The data from all lists in the series flattened.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         [1, 2, 3],
        ...         [3],
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.list_(
        ...         pa.int64()
        ...     ))
        ... )
        >>> s.list.flatten()
        0    1
        1    2
        2    3
        3    3
        dtype: int64[pyarrow]
        """
        from pandas import Series

        flattened = pc.list_flatten(self._pa_array)
        return Series(flattened, dtype=ArrowDtype(flattened.type))


class StructAccessor(ArrowAccessor):
    """
    Accessor object for structured data properties of the Series values.

    Parameters
    ----------
    data : Series
        Series containing Arrow struct data.
    """

    def __init__(self, data=None) -> None:
        super().__init__(
            data,
            validation_msg=(
                "Can only use the '.struct' accessor with 'struct[pyarrow]' "
                "dtype, not {dtype}."
            ),
        )

    def _is_valid_pyarrow_dtype(self, pyarrow_dtype) -> bool:
        return pa.types.is_struct(pyarrow_dtype)

    @property
    def dtypes(self) -> Series:
        """
        Return the dtype object of each child field of the struct.

        Returns
        -------
        pandas.Series
            The data type of each child field.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.struct(
        ...         [("version", pa.int64()), ("project", pa.string())]
        ...     ))
        ... )
        >>> s.struct.dtypes
        version     int64[pyarrow]
        project    string[pyarrow]
        dtype: object
        """
        from pandas import (
            Index,
            Series,
        )

        pa_type = self._data.dtype.pyarrow_dtype
        types = [ArrowDtype(struct.type) for struct in pa_type]
        names = [struct.name for struct in pa_type]
        return Series(types, index=Index(names))

    def field(
        self,
        name_or_index: list[str]
        | list[bytes]
        | list[int]
        | pc.Expression
        | bytes
        | str
        | int,
    ) -> Series:
        """
        Extract a child field of a struct as a Series.

        Parameters
        ----------
        name_or_index : str | bytes | int | expression | list
            Name or index of the child field to extract.

            For list-like inputs, this will index into a nested
            struct.

        Returns
        -------
        pandas.Series
            The data corresponding to the selected child field.

        See Also
        --------
        Series.struct.explode : Return all child fields as a DataFrame.

        Notes
        -----
        The name of the resulting Series will be set using the following
        rules:

        - For string, bytes, or integer `name_or_index` (or a list of these, for
          a nested selection), the Series name is set to the selected
          field's name.
        - For a :class:`pyarrow.compute.Expression`, this is set to
          the string form of the expression.
        - For list-like `name_or_index`, the name will be set to the
          name of the final field selected.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.struct(
        ...         [("version", pa.int64()), ("project", pa.string())]
        ...     ))
        ... )

        Extract by field name.

        >>> s.struct.field("project")
        0    pandas
        1    pandas
        2     numpy
        Name: project, dtype: string[pyarrow]

        Extract by field index.

        >>> s.struct.field(0)
        0    1
        1    2
        2    1
        Name: version, dtype: int64[pyarrow]

        Or an expression

        >>> import pyarrow.compute as pc
        >>> s.struct.field(pc.field("project"))
        0    pandas
        1    pandas
        2     numpy
        Name: project, dtype: string[pyarrow]

        For nested struct types, you can pass a list of values to index
        multiple levels:

        >>> version_type = pa.struct([
        ...     ("major", pa.int64()),
        ...     ("minor", pa.int64()),
        ... ])
        >>> s = pd.Series(
        ...     [
        ...         {"version": {"major": 1, "minor": 5}, "project": "pandas"},
        ...         {"version": {"major": 2, "minor": 1}, "project": "pandas"},
        ...         {"version": {"major": 1, "minor": 26}, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.struct(
        ...         [("version", version_type), ("project", pa.string())]
        ...     ))
        ... )
        >>> s.struct.field(["version", "minor"])
        0     5
        1     1
        2    26
        Name: minor, dtype: int64[pyarrow]
        >>> s.struct.field([0, 0])
        0    1
        1    2
        2    1
        Name: major, dtype: int64[pyarrow]
        """
        from pandas import Series

        def get_name(
            level_name_or_index: list[str]
            | list[bytes]
            | list[int]
            | pc.Expression
            | bytes
            | str
            | int,
            data: pa.ChunkedArray,
        ):
            if isinstance(level_name_or_index, int):
                name = data.type.field(level_name_or_index).name
            elif isinstance(level_name_or_index, (str, bytes)):
                name = level_name_or_index
            elif isinstance(level_name_or_index, pc.Expression):
                name = str(level_name_or_index)
            elif is_list_like(level_name_or_index):
                # For nested input like [2, 1, 2]
                # iteratively get the struct and field name. The last
                # one is used for the name of the index.
                level_name_or_index = list(reversed(level_name_or_index))
                selected = data
                while level_name_or_index:
                    # we need the cast, otherwise mypy complains about
                    # getting ints, bytes, or str here, which isn't possible.
                    level_name_or_index = cast(list, level_name_or_index)
                    name_or_index = level_name_or_index.pop()
                    name = get_name(name_or_index, selected)
                    selected = selected.type.field(selected.type.get_field_index(name))
                    name = selected.name
            else:
                raise ValueError(
                    "name_or_index must be an int, str, bytes, "
                    "pyarrow.compute.Expression, or list of those"
                )
            return name

        pa_arr = self._data.array._pa_array
        name = get_name(name_or_index, pa_arr)
        field_arr = pc.struct_field(pa_arr, name_or_index)

        return Series(
            field_arr,
            dtype=ArrowDtype(field_arr.type),
            index=self._data.index,
            name=name,
        )

    def explode(self) -> DataFrame:
        """
        Extract all child fields of a struct as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            The data corresponding to all child fields.

        See Also
        --------
        Series.struct.field : Return a single child field as a Series.

        Examples
        --------
        >>> import pyarrow as pa
        >>> s = pd.Series(
        ...     [
        ...         {"version": 1, "project": "pandas"},
        ...         {"version": 2, "project": "pandas"},
        ...         {"version": 1, "project": "numpy"},
        ...     ],
        ...     dtype=pd.ArrowDtype(pa.struct(
        ...         [("version", pa.int64()), ("project", pa.string())]
        ...     ))
        ... )

        >>> s.struct.explode()
           version project
        0        1  pandas
        1        2  pandas
        2        1   numpy
        """
        from pandas import concat

        pa_type = self._pa_array.type
        return concat(
            [self.field(i) for i in range(pa_type.num_fields)], axis="columns"
        )