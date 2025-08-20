
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\circuitplot.py ===
"""Matplotlib based plotting of quantum circuits.

Todo:

* Optimize printing of large circuits.
* Get this to work with single gates.
* Do a better job checking the form of circuits to make sure it is a Mul of
  Gates.
* Get multi-target gates plotting.
* Get initial and final states to plot.
* Get measurements to plot. Might need to rethink measurement as a gate
  issue.
* Get scale and figsize to be handled in a better way.
* Write some tests/examples!
"""

from __future__ import annotations

from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS


__all__ = [
    'CircuitPlot',
    'circuit_plot',
    'labeller',
    'Mz',
    'Mx',
    'CreateOneQubitGate',
    'CreateCGate',
]

np = import_module('numpy')
matplotlib = import_module(
    'matplotlib', import_kwargs={'fromlist': ['pyplot']},
    catch=(RuntimeError,))  # This is raised in environments that have no display.

if np and matplotlib:
    pyplot = matplotlib.pyplot
    Line2D = matplotlib.lines.Line2D
    Circle = matplotlib.patches.Circle

#from matplotlib import rc
#rc('text',usetex=True)

class CircuitPlot:
    """A class for managing a circuit plot."""

    scale = 1.0
    fontsize = 20.0
    linewidth = 1.0
    control_radius = 0.05
    not_radius = 0.15
    swap_delta = 0.05
    labels: list[str] = []
    inits: dict[str, str] = {}
    label_buffer = 0.5

    def __init__(self, c, nqubits, **kwargs):
        if not np or not matplotlib:
            raise ImportError('numpy or matplotlib not available.')
        self.circuit = c
        self.ngates = len(self.circuit.args)
        self.nqubits = nqubits
        self.update(kwargs)
        self._create_grid()
        self._create_figure()
        self._plot_wires()
        self._plot_gates()
        self._finish()

    def update(self, kwargs):
        """Load the kwargs into the instance dict."""
        self.__dict__.update(kwargs)

    def _create_grid(self):
        """Create the grid of wires."""
        scale = self.scale
        wire_grid = np.arange(0.0, self.nqubits*scale, scale, dtype=float)
        gate_grid = np.arange(0.0, self.ngates*scale, scale, dtype=float)
        self._wire_grid = wire_grid
        self._gate_grid = gate_grid

    def _create_figure(self):
        """Create the main matplotlib figure."""
        self._figure = pyplot.figure(
            figsize=(self.ngates*self.scale, self.nqubits*self.scale),
            facecolor='w',
            edgecolor='w'
        )
        ax = self._figure.add_subplot(
            1, 1, 1,
            frameon=True
        )
        ax.set_axis_off()
        offset = 0.5*self.scale
        ax.set_xlim(self._gate_grid[0] - offset, self._gate_grid[-1] + offset)
        ax.set_ylim(self._wire_grid[0] - offset, self._wire_grid[-1] + offset)
        ax.set_aspect('equal')
        self._axes = ax

    def _plot_wires(self):
        """Plot the wires of the circuit diagram."""
        xstart = self._gate_grid[0]
        xstop = self._gate_grid[-1]
        xdata = (xstart - self.scale, xstop + self.scale)
        for i in range(self.nqubits):
            ydata = (self._wire_grid[i], self._wire_grid[i])
            line = Line2D(
                xdata, ydata,
                color='k',
                lw=self.linewidth
            )
            self._axes.add_line(line)
            if self.labels:
                init_label_buffer = 0
                if self.inits.get(self.labels[i]): init_label_buffer = 0.25
                self._axes.text(
                    xdata[0]-self.label_buffer-init_label_buffer,ydata[0],
                    render_label(self.labels[i],self.inits),
                    size=self.fontsize,
                    color='k',ha='center',va='center')
        self._plot_measured_wires()

    def _plot_measured_wires(self):
        ismeasured = self._measurements()
        xstop = self._gate_grid[-1]
        dy = 0.04 # amount to shift wires when doubled
        # Plot doubled wires after they are measured
        for im in ismeasured:
            xdata = (self._gate_grid[ismeasured[im]],xstop+self.scale)
            ydata = (self._wire_grid[im]+dy,self._wire_grid[im]+dy)
            line = Line2D(
                xdata, ydata,
                color='k',
                lw=self.linewidth
            )
            self._axes.add_line(line)
        # Also double any controlled lines off these wires
        for i,g in enumerate(self._gates()):
            if isinstance(g, (CGate, CGateS)):
                wires = g.controls + g.targets
                for wire in wires:
                    if wire in ismeasured and \
                           self._gate_grid[i] > self._gate_grid[ismeasured[wire]]:
                        ydata = min(wires), max(wires)
                        xdata = self._gate_grid[i]-dy, self._gate_grid[i]-dy
                        line = Line2D(
                            xdata, ydata,
                            color='k',
                            lw=self.linewidth
                            )
                        self._axes.add_line(line)
    def _gates(self):
        """Create a list of all gates in the circuit plot."""
        gates = []
        if isinstance(self.circuit, Mul):
            for g in reversed(self.circuit.args):
                if isinstance(g, Gate):
                    gates.append(g)
        elif isinstance(self.circuit, Gate):
            gates.append(self.circuit)
        return gates

    def _plot_gates(self):
        """Iterate through the gates and plot each of them."""
        for i, gate in enumerate(self._gates()):
            gate.plot_gate(self, i)

    def _measurements(self):
        """Return a dict ``{i:j}`` where i is the index of the wire that has
        been measured, and j is the gate where the wire is measured.
        """
        ismeasured = {}
        for i,g in enumerate(self._gates()):
            if getattr(g,'measurement',False):
                for target in g.targets:
                    if target in ismeasured:
                        if ismeasured[target] > i:
                            ismeasured[target] = i
                    else:
                        ismeasured[target] = i
        return ismeasured

    def _finish(self):
        # Disable clipping to make panning work well for large circuits.
        for o in self._figure.findobj():
            o.set_clip_on(False)

    def one_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a single qubit gate."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        self._axes.text(
            x, y, t,
            color='k',
            ha='center',
            va='center',
            bbox={"ec": 'k', "fc": 'w', "fill": True, "lw": self.linewidth},
            size=self.fontsize
        )

    def two_qubit_box(self, t, gate_idx, wire_idx):
        """Draw a box for a two qubit gate. Does not work yet.
        """
        # x = self._gate_grid[gate_idx]
        # y = self._wire_grid[wire_idx]+0.5
        print(self._gate_grid)
        print(self._wire_grid)
        # unused:
        # obj = self._axes.text(
        #     x, y, t,
        #     color='k',
        #     ha='center',
        #     va='center',
        #     bbox=dict(ec='k', fc='w', fill=True, lw=self.linewidth),
        #     size=self.fontsize
        # )

    def control_line(self, gate_idx, min_wire, max_wire):
        """Draw a vertical control line."""
        xdata = (self._gate_grid[gate_idx], self._gate_grid[gate_idx])
        ydata = (self._wire_grid[min_wire], self._wire_grid[max_wire])
        line = Line2D(
            xdata, ydata,
            color='k',
            lw=self.linewidth
        )
        self._axes.add_line(line)

    def control_point(self, gate_idx, wire_idx):
        """Draw a control point."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.control_radius
        c = Circle(
            (x, y),
            radius*self.scale,
            ec='k',
            fc='k',
            fill=True,
            lw=self.linewidth
        )
        self._axes.add_patch(c)

    def not_point(self, gate_idx, wire_idx):
        """Draw a NOT gates as the circle with plus in the middle."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        radius = self.not_radius
        c = Circle(
            (x, y),
            radius,
            ec='k',
            fc='w',
            fill=False,
            lw=self.linewidth
        )
        self._axes.add_patch(c)
        l = Line2D(
            (x, x), (y - radius, y + radius),
            color='k',
            lw=self.linewidth
        )
        self._axes.add_line(l)

    def swap_point(self, gate_idx, wire_idx):
        """Draw a swap point as a cross."""
        x = self._gate_grid[gate_idx]
        y = self._wire_grid[wire_idx]
        d = self.swap_delta
        l1 = Line2D(
            (x - d, x + d),
            (y - d, y + d),
            color='k',
            lw=self.linewidth
        )
        l2 = Line2D(
            (x - d, x + d),
            (y + d, y - d),
            color='k',
            lw=self.linewidth
        )
        self._axes.add_line(l1)
        self._axes.add_line(l2)

def circuit_plot(c, nqubits, **kwargs):
    """Draw the circuit diagram for the circuit with nqubits.

    Parameters
    ==========

    c : circuit
        The circuit to plot. Should be a product of Gate instances.
    nqubits : int
        The number of qubits to include in the circuit. Must be at least
        as big as the largest ``min_qubits`` of the gates.
    """
    return CircuitPlot(c, nqubits, **kwargs)

def render_label(label, inits={}):
    """Slightly more flexible way to render labels.

    >>> from sympy.physics.quantum.circuitplot import render_label
    >>> render_label('q0')
    '$\\\\left|q0\\\\right\\\\rangle$'
    >>> render_label('q0', {'q0':'0'})
    '$\\\\left|q0\\\\right\\\\rangle=\\\\left|0\\\\right\\\\rangle$'
    """
    init = inits.get(label)
    if init:
        return r'$\left|%s\right\rangle=\left|%s\right\rangle$' % (label, init)
    return r'$\left|%s\right\rangle$' % label

def labeller(n, symbol='q'):
    """Autogenerate labels for wires of quantum circuits.

    Parameters
    ==========

    n : int
        number of qubits in the circuit.
    symbol : string
        A character string to precede all gate labels. E.g. 'q_0', 'q_1', etc.

    >>> from sympy.physics.quantum.circuitplot import labeller
    >>> labeller(2)
    ['q_1', 'q_0']
    >>> labeller(3,'j')
    ['j_2', 'j_1', 'j_0']
    """
    return ['%s_%d' % (symbol,n-i-1) for i in range(n)]

class Mz(OneQubitGate):
    """Mock-up of a z measurement gate.

    This is in circuitplot rather than gate.py because it's not a real
    gate, it just draws one.
    """
    measurement = True
    gate_name='Mz'
    gate_name_latex='M_z'

class Mx(OneQubitGate):
    """Mock-up of an x measurement gate.

    This is in circuitplot rather than gate.py because it's not a real
    gate, it just draws one.
    """
    measurement = True
    gate_name='Mx'
    gate_name_latex='M_x'

class CreateOneQubitGate(type):
    def __new__(mcl, name, latexname=None):
        if not latexname:
            latexname = name
        return type(name + "Gate", (OneQubitGate,),
            {'gate_name': name, 'gate_name_latex': latexname})

def CreateCGate(name, latexname=None):
    """Use a lexical closure to make a controlled gate.
    """
    if not latexname:
        latexname = name
    onequbitgate = CreateOneQubitGate(name, latexname)
    def ControlledGate(ctrls,target):
        return CGate(tuple(ctrls),onequbitgate(target))
    return ControlledGate

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\h11\_events.py ===
# High level events that make up HTTP/1.1 conversations. Loosely inspired by
# the corresponding events in hyper-h2:
#
#     http://python-hyper.org/h2/en/stable/api.html#events
#
# Don't subclass these. Stuff will break.

import re
from abc import ABC
from dataclasses import dataclass
from typing import List, Tuple, Union

from ._abnf import method, request_target
from ._headers import Headers, normalize_and_validate
from ._util import bytesify, LocalProtocolError, validate

# Everything in __all__ gets re-exported as part of the h11 public API.
__all__ = [
    "Event",
    "Request",
    "InformationalResponse",
    "Response",
    "Data",
    "EndOfMessage",
    "ConnectionClosed",
]

method_re = re.compile(method.encode("ascii"))
request_target_re = re.compile(request_target.encode("ascii"))


class Event(ABC):
    """
    Base class for h11 events.
    """

    __slots__ = ()


@dataclass(init=False, frozen=True)
class Request(Event):
    """The beginning of an HTTP request.

    Fields:

    .. attribute:: method

       An HTTP method, e.g. ``b"GET"`` or ``b"POST"``. Always a byte
       string. :term:`Bytes-like objects <bytes-like object>` and native
       strings containing only ascii characters will be automatically
       converted to byte strings.

    .. attribute:: target

       The target of an HTTP request, e.g. ``b"/index.html"``, or one of the
       more exotic formats described in `RFC 7320, section 5.3
       <https://tools.ietf.org/html/rfc7230#section-5.3>`_. Always a byte
       string. :term:`Bytes-like objects <bytes-like object>` and native
       strings containing only ascii characters will be automatically
       converted to byte strings.

    .. attribute:: headers

       Request headers, represented as a list of (name, value) pairs. See
       :ref:`the header normalization rules <headers-format>` for details.

    .. attribute:: http_version

       The HTTP protocol version, represented as a byte string like
       ``b"1.1"``. See :ref:`the HTTP version normalization rules
       <http_version-format>` for details.

    """

    __slots__ = ("method", "headers", "target", "http_version")

    method: bytes
    headers: Headers
    target: bytes
    http_version: bytes

    def __init__(
        self,
        *,
        method: Union[bytes, str],
        headers: Union[Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]]],
        target: Union[bytes, str],
        http_version: Union[bytes, str] = b"1.1",
        _parsed: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(headers, Headers):
            object.__setattr__(self, "headers", headers)
        else:
            object.__setattr__(
                self, "headers", normalize_and_validate(headers, _parsed=_parsed)
            )
        if not _parsed:
            object.__setattr__(self, "method", bytesify(method))
            object.__setattr__(self, "target", bytesify(target))
            object.__setattr__(self, "http_version", bytesify(http_version))
        else:
            object.__setattr__(self, "method", method)
            object.__setattr__(self, "target", target)
            object.__setattr__(self, "http_version", http_version)

        # "A server MUST respond with a 400 (Bad Request) status code to any
        # HTTP/1.1 request message that lacks a Host header field and to any
        # request message that contains more than one Host header field or a
        # Host header field with an invalid field-value."
        # -- https://tools.ietf.org/html/rfc7230#section-5.4
        host_count = 0
        for name, value in self.headers:
            if name == b"host":
                host_count += 1
        if self.http_version == b"1.1" and host_count == 0:
            raise LocalProtocolError("Missing mandatory Host: header")
        if host_count > 1:
            raise LocalProtocolError("Found multiple Host: headers")

        validate(method_re, self.method, "Illegal method characters")
        validate(request_target_re, self.target, "Illegal target characters")

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class _ResponseBase(Event):
    __slots__ = ("headers", "http_version", "reason", "status_code")

    headers: Headers
    http_version: bytes
    reason: bytes
    status_code: int

    def __init__(
        self,
        *,
        headers: Union[Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]]],
        status_code: int,
        http_version: Union[bytes, str] = b"1.1",
        reason: Union[bytes, str] = b"",
        _parsed: bool = False,
    ) -> None:
        super().__init__()
        if isinstance(headers, Headers):
            object.__setattr__(self, "headers", headers)
        else:
            object.__setattr__(
                self, "headers", normalize_and_validate(headers, _parsed=_parsed)
            )
        if not _parsed:
            object.__setattr__(self, "reason", bytesify(reason))
            object.__setattr__(self, "http_version", bytesify(http_version))
            if not isinstance(status_code, int):
                raise LocalProtocolError("status code must be integer")
            # Because IntEnum objects are instances of int, but aren't
            # duck-compatible (sigh), see gh-72.
            object.__setattr__(self, "status_code", int(status_code))
        else:
            object.__setattr__(self, "reason", reason)
            object.__setattr__(self, "http_version", http_version)
            object.__setattr__(self, "status_code", status_code)

        self.__post_init__()

    def __post_init__(self) -> None:
        pass

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class InformationalResponse(_ResponseBase):
    """An HTTP informational response.

    Fields:

    .. attribute:: status_code

       The status code of this response, as an integer. For an
       :class:`InformationalResponse`, this is always in the range [100,
       200).

    .. attribute:: headers

       Request headers, represented as a list of (name, value) pairs. See
       :ref:`the header normalization rules <headers-format>` for
       details.

    .. attribute:: http_version

       The HTTP protocol version, represented as a byte string like
       ``b"1.1"``. See :ref:`the HTTP version normalization rules
       <http_version-format>` for details.

    .. attribute:: reason

       The reason phrase of this response, as a byte string. For example:
       ``b"OK"``, or ``b"Not Found"``.

    """

    def __post_init__(self) -> None:
        if not (100 <= self.status_code < 200):
            raise LocalProtocolError(
                "InformationalResponse status_code should be in range "
                "[100, 200), not {}".format(self.status_code)
            )

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class Response(_ResponseBase):
    """The beginning of an HTTP response.

    Fields:

    .. attribute:: status_code

       The status code of this response, as an integer. For an
       :class:`Response`, this is always in the range [200,
       1000).

    .. attribute:: headers

       Request headers, represented as a list of (name, value) pairs. See
       :ref:`the header normalization rules <headers-format>` for details.

    .. attribute:: http_version

       The HTTP protocol version, represented as a byte string like
       ``b"1.1"``. See :ref:`the HTTP version normalization rules
       <http_version-format>` for details.

    .. attribute:: reason

       The reason phrase of this response, as a byte string. For example:
       ``b"OK"``, or ``b"Not Found"``.

    """

    def __post_init__(self) -> None:
        if not (200 <= self.status_code < 1000):
            raise LocalProtocolError(
                "Response status_code should be in range [200, 1000), not {}".format(
                    self.status_code
                )
            )

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(init=False, frozen=True)
class Data(Event):
    """Part of an HTTP message body.

    Fields:

    .. attribute:: data

       A :term:`bytes-like object` containing part of a message body. Or, if
       using the ``combine=False`` argument to :meth:`Connection.send`, then
       any object that your socket writing code knows what to do with, and for
       which calling :func:`len` returns the number of bytes that will be
       written -- see :ref:`sendfile` for details.

    .. attribute:: chunk_start

       A marker that indicates whether this data object is from the start of a
       chunked transfer encoding chunk. This field is ignored when when a Data
       event is provided to :meth:`Connection.send`: it is only valid on
       events emitted from :meth:`Connection.next_event`. You probably
       shouldn't use this attribute at all; see
       :ref:`chunk-delimiters-are-bad` for details.

    .. attribute:: chunk_end

       A marker that indicates whether this data object is the last for a
       given chunked transfer encoding chunk. This field is ignored when when
       a Data event is provided to :meth:`Connection.send`: it is only valid
       on events emitted from :meth:`Connection.next_event`. You probably
       shouldn't use this attribute at all; see
       :ref:`chunk-delimiters-are-bad` for details.

    """

    __slots__ = ("data", "chunk_start", "chunk_end")

    data: bytes
    chunk_start: bool
    chunk_end: bool

    def __init__(
        self, data: bytes, chunk_start: bool = False, chunk_end: bool = False
    ) -> None:
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "chunk_start", chunk_start)
        object.__setattr__(self, "chunk_end", chunk_end)

    # This is an unhashable type.
    __hash__ = None  # type: ignore


# XX FIXME: "A recipient MUST ignore (or consider as an error) any fields that
# are forbidden to be sent in a trailer, since processing them as if they were
# present in the header section might bypass external security filters."
# https://svn.tools.ietf.org/svn/wg/httpbis/specs/rfc7230.html#chunked.trailer.part
# Unfortunately, the list of forbidden fields is long and vague :-/
@dataclass(init=False, frozen=True)
class EndOfMessage(Event):
    """The end of an HTTP message.

    Fields:

    .. attribute:: headers

       Default value: ``[]``

       Any trailing headers attached to this message, represented as a list of
       (name, value) pairs. See :ref:`the header normalization rules
       <headers-format>` for details.

       Must be empty unless ``Transfer-Encoding: chunked`` is in use.

    """

    __slots__ = ("headers",)

    headers: Headers

    def __init__(
        self,
        *,
        headers: Union[
            Headers, List[Tuple[bytes, bytes]], List[Tuple[str, str]], None
        ] = None,
        _parsed: bool = False,
    ) -> None:
        super().__init__()
        if headers is None:
            headers = Headers([])
        elif not isinstance(headers, Headers):
            headers = normalize_and_validate(headers, _parsed=_parsed)

        object.__setattr__(self, "headers", headers)

    # This is an unhashable type.
    __hash__ = None  # type: ignore


@dataclass(frozen=True)
class ConnectionClosed(Event):
    """This event indicates that the sender has closed their outgoing
    connection.

    Note that this does not necessarily mean that they can't *receive* further
    data, because TCP connections are composed to two one-way channels which
    can be closed independently. See :ref:`closing` for details.

    No fields.
    """

    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\yacctab.py ===

# yacctab.py
# This file is automatically generated. Do not edit.
_tabversion = '3.10'

_lr_method = 'LALR'

_lr_signature = 'translation_unit_or_emptyleftLORleftLANDleftORleftXORleftANDleftEQNEleftGTGELTLEleftRSHIFTLSHIFTleftPLUSMINUSleftTIMESDIVIDEMODAUTO BREAK CASE CHAR CONST CONTINUE DEFAULT DO DOUBLE ELSE ENUM EXTERN FLOAT FOR GOTO IF INLINE INT LONG REGISTER OFFSETOF RESTRICT RETURN SHORT SIGNED SIZEOF STATIC STRUCT SWITCH TYPEDEF UNION UNSIGNED VOID VOLATILE WHILE __INT128 _BOOL _COMPLEX _NORETURN _THREAD_LOCAL _STATIC_ASSERT _ATOMIC _ALIGNOF _ALIGNAS _PRAGMA ID TYPEID INT_CONST_DEC INT_CONST_OCT INT_CONST_HEX INT_CONST_BIN INT_CONST_CHAR FLOAT_CONST HEX_FLOAT_CONST CHAR_CONST WCHAR_CONST U8CHAR_CONST U16CHAR_CONST U32CHAR_CONST STRING_LITERAL WSTRING_LITERAL U8STRING_LITERAL U16STRING_LITERAL U32STRING_LITERAL PLUS MINUS TIMES DIVIDE MOD OR AND NOT XOR LSHIFT RSHIFT LOR LAND LNOT LT LE GT GE EQ NE EQUALS TIMESEQUAL DIVEQUAL MODEQUAL PLUSEQUAL MINUSEQUAL LSHIFTEQUAL RSHIFTEQUAL ANDEQUAL XOREQUAL OREQUAL PLUSPLUS MINUSMINUS ARROW CONDOP LPAREN RPAREN LBRACKET RBRACKET LBRACE RBRACE COMMA PERIOD SEMI COLON ELLIPSIS PPHASH PPPRAGMA PPPRAGMASTRabstract_declarator_opt : empty\n| abstract_declaratorassignment_expression_opt : empty\n| assignment_expressionblock_item_list_opt : empty\n| block_item_listdeclaration_list_opt : empty\n| declaration_listdeclaration_specifiers_no_type_opt : empty\n| declaration_specifiers_no_typedesignation_opt : empty\n| designationexpression_opt : empty\n| expressionid_init_declarator_list_opt : empty\n| id_init_declarator_listidentifier_list_opt : empty\n| identifier_listinit_declarator_list_opt : empty\n| init_declarator_listinitializer_list_opt : empty\n| initializer_listparameter_type_list_opt : empty\n| parameter_type_liststruct_declarator_list_opt : empty\n| struct_declarator_listtype_qualifier_list_opt : empty\n| type_qualifier_list direct_id_declarator   : ID\n         direct_id_declarator   : LPAREN id_declarator RPAREN\n         direct_id_declarator   : direct_id_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET\n         direct_id_declarator   : direct_id_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET\n                                    | direct_id_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET\n         direct_id_declarator   : direct_id_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET\n         direct_id_declarator   : direct_id_declarator LPAREN parameter_type_list RPAREN\n                                    | direct_id_declarator LPAREN identifier_list_opt RPAREN\n         direct_typeid_declarator   : TYPEID\n         direct_typeid_declarator   : LPAREN typeid_declarator RPAREN\n         direct_typeid_declarator   : direct_typeid_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET\n         direct_typeid_declarator   : direct_typeid_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET\n                                    | direct_typeid_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET\n         direct_typeid_declarator   : direct_typeid_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET\n         direct_typeid_declarator   : direct_typeid_declarator LPAREN parameter_type_list RPAREN\n                                    | direct_typeid_declarator LPAREN identifier_list_opt RPAREN\n         direct_typeid_noparen_declarator   : TYPEID\n         direct_typeid_noparen_declarator   : direct_typeid_noparen_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET\n         direct_typeid_noparen_declarator   : direct_typeid_noparen_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET\n                                    | direct_typeid_noparen_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET\n         direct_typeid_noparen_declarator   : direct_typeid_noparen_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET\n         direct_typeid_noparen_declarator   : direct_typeid_noparen_declarator LPAREN parameter_type_list RPAREN\n                                    | direct_typeid_noparen_declarator LPAREN identifier_list_opt RPAREN\n         id_declarator  : direct_id_declarator\n         id_declarator  : pointer direct_id_declarator\n         typeid_declarator  : direct_typeid_declarator\n         typeid_declarator  : pointer direct_typeid_declarator\n         typeid_noparen_declarator  : direct_typeid_noparen_declarator\n         typeid_noparen_declarator  : pointer direct_typeid_noparen_declarator\n         translation_unit_or_empty   : translation_unit\n                                        | empty\n         translation_unit    : external_declaration\n         translation_unit    : translation_unit external_declaration\n         external_declaration    : function_definition\n         external_declaration    : declaration\n         external_declaration    : pp_directive\n                                    | pppragma_directive\n         external_declaration    : SEMI\n         external_declaration    : static_assert\n         static_assert           : _STATIC_ASSERT LPAREN constant_expression COMMA unified_string_literal RPAREN\n                                    | _STATIC_ASSERT LPAREN constant_expression RPAREN\n         pp_directive  : PPHASH\n         pppragma_directive      : PPPRAGMA\n                                    | PPPRAGMA PPPRAGMASTR\n                                    | _PRAGMA LPAREN unified_string_literal RPAREN\n         pppragma_directive_list : pppragma_directive\n                                    | pppragma_directive_list pppragma_directive\n         function_definition : id_declarator declaration_list_opt compound_statement\n         function_definition : declaration_specifiers id_declarator declaration_list_opt compound_statement\n         statement   : labeled_statement\n                        | expression_statement\n                        | compound_statement\n                        | selection_statement\n                        | iteration_statement\n                        | jump_statement\n                        | pppragma_directive\n                        | static_assert\n         pragmacomp_or_statement     : pppragma_directive_list statement\n                                        | statement\n         decl_body : declaration_specifiers init_declarator_list_opt\n                      | declaration_specifiers_no_type id_init_declarator_list_opt\n         declaration : decl_body SEMI\n         declaration_list    : declaration\n                                | declaration_list declaration\n         declaration_specifiers_no_type  : type_qualifier declaration_specifiers_no_type_opt\n         declaration_specifiers_no_type  : storage_class_specifier declaration_specifiers_no_type_opt\n         declaration_specifiers_no_type  : function_specifier declaration_specifiers_no_type_opt\n         declaration_specifiers_no_type  : atomic_specifier declaration_specifiers_no_type_opt\n         declaration_specifiers_no_type  : alignment_specifier declaration_specifiers_no_type_opt\n         declaration_specifiers  : declaration_specifiers type_qualifier\n         declaration_specifiers  : declaration_specifiers storage_class_specifier\n         declaration_specifiers  : declaration_specifiers function_specifier\n         declaration_specifiers  : declaration_specifiers type_specifier_no_typeid\n         declaration_specifiers  : type_specifier\n         declaration_specifiers  : declaration_specifiers_no_type type_specifier\n         declaration_specifiers  : declaration_specifiers alignment_specifier\n         storage_class_specifier : AUTO\n                                    | REGISTER\n                                    | STATIC\n                                    | EXTERN\n                                    | TYPEDEF\n                                    | _THREAD_LOCAL\n         function_specifier  : INLINE\n                                | _NORETURN\n         type_specifier_no_typeid  : VOID\n                                      | _BOOL\n                                      | CHAR\n                                      | SHORT\n                                      | INT\n                                      | LONG\n                                      | FLOAT\n                                      | DOUBLE\n                                      | _COMPLEX\n                                      | SIGNED\n                                      | UNSIGNED\n                                      | __INT128\n         type_specifier  : typedef_name\n                            | enum_specifier\n                            | struct_or_union_specifier\n                            | type_specifier_no_typeid\n                            | atomic_specifier\n         atomic_specifier  : _ATOMIC LPAREN type_name RPAREN\n         type_qualifier  : CONST\n                            | RESTRICT\n                            | VOLATILE\n                            | _ATOMIC\n         init_declarator_list    : init_declarator\n                                    | init_declarator_list COMMA init_declarator\n         init_declarator : declarator\n                            | declarator EQUALS initializer\n         id_init_declarator_list    : id_init_declarator\n                                       | id_init_declarator_list COMMA init_declarator\n         id_init_declarator : id_declarator\n                               | id_declarator EQUALS initializer\n         specifier_qualifier_list    : specifier_qualifier_list type_specifier_no_typeid\n         specifier_qualifier_list    : specifier_qualifier_list type_qualifier\n         specifier_qualifier_list  : type_specifier\n         specifier_qualifier_list  : type_qualifier_list type_specifier\n         specifier_qualifier_list  : alignment_specifier\n         specifier_qualifier_list  : specifier_qualifier_list alignment_specifier\n         struct_or_union_specifier   : struct_or_union ID\n                                        | struct_or_union TYPEID\n         struct_or_union_specifier : struct_or_union brace_open struct_declaration_list brace_close\n                                      | struct_or_union brace_open brace_close\n         struct_or_union_specifier   : struct_or_union ID brace_open struct_declaration_list brace_close\n                                        | struct_or_union ID brace_open brace_close\n                                        | struct_or_union TYPEID brace_open struct_declaration_list brace_close\n                                        | struct_or_union TYPEID brace_open brace_close\n         struct_or_union : STRUCT\n                            | UNION\n         struct_declaration_list     : struct_declaration\n                                        | struct_declaration_list struct_declaration\n         struct_declaration : specifier_qualifier_list struct_declarator_list_opt SEMI\n         struct_declaration : SEMI\n         struct_declaration : pppragma_directive\n         struct_declarator_list  : struct_declarator\n                                    | struct_declarator_list COMMA struct_declarator\n         struct_declarator : declarator\n         struct_declarator   : declarator COLON constant_expression\n                                | COLON constant_expression\n         enum_specifier  : ENUM ID\n                            | ENUM TYPEID\n         enum_specifier  : ENUM brace_open enumerator_list brace_close\n         enum_specifier  : ENUM ID brace_open enumerator_list brace_close\n                            | ENUM TYPEID brace_open enumerator_list brace_close\n         enumerator_list : enumerator\n                            | enumerator_list COMMA\n                            | enumerator_list COMMA enumerator\n         alignment_specifier  : _ALIGNAS LPAREN type_name RPAREN\n                                 | _ALIGNAS LPAREN constant_expression RPAREN\n         enumerator  : ID\n                        | ID EQUALS constant_expression\n         declarator  : id_declarator\n                        | typeid_declarator\n         pointer : TIMES type_qualifier_list_opt\n                    | TIMES type_qualifier_list_opt pointer\n         type_qualifier_list : type_qualifier\n                                | type_qualifier_list type_qualifier\n         parameter_type_list : parameter_list\n                                | parameter_list COMMA ELLIPSIS\n         parameter_list  : parameter_declaration\n                            | parameter_list COMMA parameter_declaration\n         parameter_declaration   : declaration_specifiers id_declarator\n                                    | declaration_specifiers typeid_noparen_declarator\n         parameter_declaration   : declaration_specifiers abstract_declarator_opt\n         identifier_list : identifier\n                            | identifier_list COMMA identifier\n         initializer : assignment_expression\n         initializer : brace_open initializer_list_opt brace_close\n                        | brace_open initializer_list COMMA brace_close\n         initializer_list    : designation_opt initializer\n                                | initializer_list COMMA designation_opt initializer\n         designation : designator_list EQUALS\n         designator_list : designator\n                            | designator_list designator\n         designator  : LBRACKET constant_expression RBRACKET\n                        | PERIOD identifier\n         type_name   : specifier_qualifier_list abstract_declarator_opt\n         abstract_declarator     : pointer\n         abstract_declarator     : pointer direct_abstract_declarator\n         abstract_declarator     : direct_abstract_declarator\n         direct_abstract_declarator  : LPAREN abstract_declarator RPAREN  direct_abstract_declarator  : direct_abstract_declarator LBRACKET assignment_expression_opt RBRACKET\n         direct_abstract_declarator  : LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET\n         direct_abstract_declarator  : direct_abstract_declarator LBRACKET TIMES RBRACKET\n         direct_abstract_declarator  : LBRACKET TIMES RBRACKET\n         direct_abstract_declarator  : direct_abstract_declarator LPAREN parameter_type_list_opt RPAREN\n         direct_abstract_declarator  : LPAREN parameter_type_list_opt RPAREN\n         block_item  : declaration\n                        | statement\n         block_item_list : block_item\n                            | block_item_list block_item\n         compound_statement : brace_open block_item_list_opt brace_close  labeled_statement : ID COLON pragmacomp_or_statement  labeled_statement : CASE constant_expression COLON pragmacomp_or_statement  labeled_statement : DEFAULT COLON pragmacomp_or_statement  selection_statement : IF LPAREN expression RPAREN pragmacomp_or_statement  selection_statement : IF LPAREN expression RPAREN statement ELSE pragmacomp_or_statement  selection_statement : SWITCH LPAREN expression RPAREN pragmacomp_or_statement  iteration_statement : WHILE LPAREN expression RPAREN pragmacomp_or_statement  iteration_statement : DO pragmacomp_or_statement WHILE LPAREN expression RPAREN SEMI  iteration_statement : FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN pragmacomp_or_statement  iteration_statement : FOR LPAREN declaration expression_opt SEMI expression_opt RPAREN pragmacomp_or_statement  jump_statement  : GOTO ID SEMI  jump_statement  : BREAK SEMI  jump_statement  : CONTINUE SEMI  jump_statement  : RETURN expression SEMI\n                            | RETURN SEMI\n         expression_statement : expression_opt SEMI  expression  : assignment_expression\n                        | expression COMMA assignment_expression\n         assignment_expression : LPAREN compound_statement RPAREN  typedef_name : TYPEID  assignment_expression   : conditional_expression\n                                    | unary_expression assignment_operator assignment_expression\n         assignment_operator : EQUALS\n                                | XOREQUAL\n                                | TIMESEQUAL\n                                | DIVEQUAL\n                                | MODEQUAL\n                                | PLUSEQUAL\n                                | MINUSEQUAL\n                                | LSHIFTEQUAL\n                                | RSHIFTEQUAL\n                                | ANDEQUAL\n                                | OREQUAL\n         constant_expression : conditional_expression  conditional_expression  : binary_expression\n                                    | binary_expression CONDOP expression COLON conditional_expression\n         binary_expression   : cast_expression\n                                | binary_expression TIMES binary_expression\n                                | binary_expression DIVIDE binary_expression\n                                | binary_expression MOD binary_expression\n                                | binary_expression PLUS binary_expression\n                                | binary_expression MINUS binary_expression\n                                | binary_expression RSHIFT binary_expression\n                                | binary_expression LSHIFT binary_expression\n                                | binary_expression LT binary_expression\n                                | binary_expression LE binary_expression\n                                | binary_expression GE binary_expression\n                                | binary_expression GT binary_expression\n                                | binary_expression EQ binary_expression\n                                | binary_expression NE binary_expression\n                                | binary_expression AND binary_expression\n                                | binary_expression OR binary_expression\n                                | binary_expression XOR binary_expression\n                                | binary_expression LAND binary_expression\n                                | binary_expression LOR binary_expression\n         cast_expression : unary_expression  cast_expression : LPAREN type_name RPAREN cast_expression  unary_expression    : postfix_expression  unary_expression    : PLUSPLUS unary_expression\n                                | MINUSMINUS unary_expression\n                                | unary_operator cast_expression\n         unary_expression    : SIZEOF unary_expression\n                                | SIZEOF LPAREN type_name RPAREN\n                                | _ALIGNOF LPAREN type_name RPAREN\n         unary_operator  : AND\n                            | TIMES\n                            | PLUS\n                            | MINUS\n                            | NOT\n                            | LNOT\n         postfix_expression  : primary_expression  postfix_expression  : postfix_expression LBRACKET expression RBRACKET  postfix_expression  : postfix_expression LPAREN argument_expression_list RPAREN\n                                | postfix_expression LPAREN RPAREN\n         postfix_expression  : postfix_expression PERIOD ID\n                                | postfix_expression PERIOD TYPEID\n                                | postfix_expression ARROW ID\n                                | postfix_expression ARROW TYPEID\n         postfix_expression  : postfix_expression PLUSPLUS\n                                | postfix_expression MINUSMINUS\n         postfix_expression  : LPAREN type_name RPAREN brace_open initializer_list brace_close\n                                | LPAREN type_name RPAREN brace_open initializer_list COMMA brace_close\n         primary_expression  : identifier  primary_expression  : constant  primary_expression  : unified_string_literal\n                                | unified_wstring_literal\n         primary_expression  : LPAREN expression RPAREN  primary_expression  : OFFSETOF LPAREN type_name COMMA offsetof_member_designator RPAREN\n         offsetof_member_designator : identifier\n                                         | offsetof_member_designator PERIOD identifier\n                                         | offsetof_member_designator LBRACKET expression RBRACKET\n         argument_expression_list    : assignment_expression\n                                        | argument_expression_list COMMA assignment_expression\n         identifier  : ID  constant    : INT_CONST_DEC\n                        | INT_CONST_OCT\n                        | INT_CONST_HEX\n                        | INT_CONST_BIN\n                        | INT_CONST_CHAR\n         constant    : FLOAT_CONST\n                        | HEX_FLOAT_CONST\n         constant    : CHAR_CONST\n                        | WCHAR_CONST\n                        | U8CHAR_CONST\n                        | U16CHAR_CONST\n                        | U32CHAR_CONST\n         unified_string_literal  : STRING_LITERAL\n                                    | unified_string_literal STRING_LITERAL\n         unified_wstring_literal : WSTRING_LITERAL\n                                    | U8STRING_LITERAL\n                                    | U16STRING_LITERAL\n                                    | U32STRING_LITERAL\n                                    | unified_wstring_literal WSTRING_LITERAL\n                                    | unified_wstring_literal U8STRING_LITERAL\n                                    | unified_wstring_literal U16STRING_LITERAL\n                                    | unified_wstring_literal U32STRING_LITERAL\n         brace_open  :   LBRACE\n         brace_close :   RBRACE\n        empty : '
    
_lr_action_items = {'$end':([0,1,2,3,4,5,6,7,8,9,10,14,15,64,90,91,127,208,251,262,267,355,499,],[-340,0,-58,-59,-60,-62,-63,-64,-65,-66,-67,-70,-71,-61,-90,-72,-76,-339,-77,-73,-69,-221,-68,]),'SEMI':([0,2,4,5,6,7,8,9,10,12,13,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,69,70,71,72,73,74,75,76,77,78,79,81,83,84,85,86,87,88,89,90,91,97,98,99,100,101,102,103,104,105,106,107,108,110,111,112,117,118,119,121,122,123,124,127,128,130,132,139,140,143,144,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,203,204,205,206,207,208,209,210,211,212,214,220,221,222,223,224,225,226,227,228,229,230,231,232,233,236,239,242,245,246,247,248,249,250,251,252,253,254,255,262,263,267,291,292,293,295,296,297,300,301,302,303,311,312,326,327,330,333,334,335,336,337,338,339,340,341,342,343,344,345,346,348,349,353,354,355,356,357,358,360,361,369,370,371,372,373,374,375,376,377,403,404,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,439,440,459,460,463,464,465,468,469,470,471,473,475,479,480,481,482,483,484,485,486,493,494,497,499,501,502,505,506,508,509,522,523,524,525,526,527,529,530,531,535,536,538,552,553,554,555,556,558,561,563,570,571,574,579,580,582,584,585,586,],[9,9,-60,-62,-63,-64,-65,-66,-67,-340,90,-70,-71,-52,-340,-340,-340,-128,-102,-340,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,-340,-340,-129,-134,-181,-98,-99,-100,-101,-104,-88,-134,-19,-20,-135,-137,-182,-54,-37,-90,-72,-53,-93,-9,-10,-340,-94,-95,-103,-89,-129,-15,-16,-139,-141,-97,-96,-169,-170,-338,-149,-150,210,-76,-340,-181,-55,-328,-30,-306,-255,-256,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,210,210,210,-152,-159,-339,-340,-162,-163,-145,-147,-13,-340,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-315,361,-14,-340,374,375,377,-238,-242,-277,-77,-38,-136,-138,-196,-73,-329,-69,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-35,-36,-140,-142,-171,210,-154,210,-156,-151,-160,465,-143,-144,-148,-25,-26,-164,-166,-146,-130,-177,-178,-221,-220,-13,-340,-340,-237,-340,-87,-74,-340,483,-233,-234,484,-236,-43,-44,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,-275,-276,-295,-296,-297,-298,-299,-31,-34,-172,-173,-153,-155,-161,-168,-222,-340,-224,-240,-239,-86,-75,529,-340,-232,-235,-243,-197,-39,-42,-278,-68,-293,-294,-284,-285,-32,-33,-165,-167,-223,-340,-340,-340,-340,559,-198,-40,-41,-257,-225,-87,-74,-227,-228,572,-302,-309,-340,580,-303,-226,-229,-340,-340,-231,-230,]),'PPHASH':([0,2,4,5,6,7,8,9,10,14,15,64,90,91,127,208,251,262,267,355,499,],[14,14,-60,-62,-63,-64,-65,-66,-67,-70,-71,-61,-90,-72,-76,-339,-77,-73,-69,-221,-68,]),'PPPRAGMA':([0,2,4,5,6,7,8,9,10,14,15,64,90,91,121,124,127,128,203,204,205,207,208,210,211,221,222,223,224,225,226,227,228,229,230,231,232,242,251,262,267,333,335,338,355,356,358,360,361,369,370,371,374,375,377,465,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[15,15,-60,-62,-63,-64,-65,-66,-67,-70,-71,-61,-90,-72,-338,15,-76,15,15,15,15,-159,-339,-162,-163,15,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,15,-77,-73,-69,15,15,-160,-221,-220,15,15,-237,15,-87,-74,-233,-234,-236,-161,-222,15,-224,-86,-75,-232,-235,-68,-223,15,15,15,-225,-87,-74,-227,-228,15,-226,-229,15,15,-231,-230,]),'_PRAGMA':([0,2,4,5,6,7,8,9,10,14,15,64,90,91,121,124,127,128,203,204,205,207,208,210,211,221,222,223,224,225,226,227,228,229,230,231,232,242,251,262,267,333,335,338,355,356,358,360,361,369,370,371,374,375,377,465,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[16,16,-60,-62,-63,-64,-65,-66,-67,-70,-71,-61,-90,-72,-338,16,-76,16,16,16,16,-159,-339,-162,-163,16,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,16,-77,-73,-69,16,16,-160,-221,-220,16,16,-237,16,-87,-74,-233,-234,-236,-161,-222,16,-224,-86,-75,-232,-235,-68,-223,16,16,16,-225,-87,-74,-227,-228,16,-226,-229,16,16,-231,-230,]),'_STATIC_ASSERT':([0,2,4,5,6,7,8,9,10,14,15,64,90,91,121,127,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,251,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[18,18,-60,-62,-63,-64,-65,-66,-67,-70,-71,-61,-90,-72,-338,-76,18,-339,18,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,18,-77,-73,-69,-221,-220,18,18,-237,18,-87,-74,-233,-234,-236,-222,18,-224,-86,-75,-232,-235,-68,-223,18,18,18,-225,-87,-74,-227,-228,18,-226,-229,18,18,-231,-230,]),'ID':([0,2,4,5,6,7,8,9,10,12,14,15,17,20,21,22,23,24,25,26,27,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,62,63,64,69,70,71,72,74,75,76,77,78,80,81,82,90,91,94,95,96,98,99,100,101,102,103,104,106,112,113,114,115,116,117,118,119,120,121,122,123,126,127,128,134,135,136,137,141,147,148,149,150,153,154,155,156,160,161,182,183,184,192,194,195,196,197,198,199,206,208,209,212,214,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,244,247,251,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,294,298,306,309,310,314,318,322,323,330,331,332,334,336,337,340,341,342,347,348,349,353,354,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,398,400,401,402,405,448,449,452,455,457,459,460,463,464,466,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,507,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,564,565,570,572,579,580,582,584,585,586,],[28,28,-60,-62,-63,-64,-65,-66,-67,28,-70,-71,28,28,-340,-340,-340,-128,-102,28,-340,-107,-340,-125,-126,-127,-129,-241,118,122,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-157,-158,-61,28,28,-129,-134,-98,-99,-100,-101,-104,28,-134,28,-90,-72,159,-340,159,-93,-9,-10,-340,-94,-95,-103,-129,-97,-183,-27,-28,-185,-96,-169,-170,202,-338,-149,-150,159,-76,233,28,159,-340,159,159,-287,-288,-289,-286,159,159,159,159,-290,-291,159,-340,-28,28,28,159,-184,-186,202,202,-152,-339,28,-145,-147,233,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,159,159,233,373,159,-77,-340,159,-340,-28,-73,-69,159,159,159,159,159,159,159,159,159,159,159,159,159,159,159,159,159,159,159,159,159,431,433,159,159,-287,159,159,159,28,28,-340,-171,202,159,-154,-156,-151,-143,-144,-148,159,-146,-130,-177,-178,-221,-220,233,233,-237,159,159,159,159,233,-87,-74,159,-233,-234,-236,159,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,159,-12,159,159,-287,159,159,159,-340,159,28,159,159,-172,-173,-153,-155,28,159,-222,233,-224,159,-86,-75,159,-232,-235,-340,-201,-340,-68,159,159,159,159,-340,-28,-287,-223,233,233,233,159,159,159,-11,-287,159,159,-225,-87,-74,-227,-228,159,-340,159,159,233,159,-226,-229,233,233,-231,-230,]),'LPAREN':([0,2,4,5,6,7,8,9,10,12,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,64,69,70,71,72,74,75,76,77,78,80,81,82,88,89,90,91,94,95,97,98,99,100,101,102,103,104,106,109,112,113,114,115,116,117,118,119,121,122,123,126,127,128,132,134,135,136,139,140,141,143,147,148,149,150,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,192,194,195,196,197,206,208,209,212,214,216,221,222,223,224,225,226,227,228,229,230,231,232,233,234,237,238,240,241,242,243,247,251,252,256,257,258,259,262,263,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,291,292,294,298,300,301,302,303,306,309,310,311,312,318,319,322,323,324,325,330,332,334,336,337,340,341,342,347,348,349,351,352,353,354,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,403,404,405,406,429,431,432,433,434,439,440,446,447,448,452,455,457,459,460,463,464,466,467,469,470,471,474,478,479,480,482,483,484,487,489,493,494,498,499,500,501,502,503,508,509,510,511,512,515,516,518,520,524,525,526,527,528,529,532,533,535,536,543,544,545,546,547,548,549,550,551,552,553,554,555,556,559,561,562,563,565,566,567,570,572,574,577,578,579,580,582,584,585,586,],[17,17,-60,-62,-63,-64,-65,-66,-67,82,-70,-71,92,17,94,96,17,-340,-340,-340,-128,-102,17,-340,-29,-107,-340,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,125,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,126,-61,82,17,-129,125,-98,-99,-100,-101,-104,82,-134,82,137,-37,-90,-72,141,-340,96,-93,-9,-10,-340,-94,-95,-103,-129,125,-97,-183,-27,-28,-185,-96,-169,-170,-338,-149,-150,141,-76,238,137,82,238,-340,-328,-30,238,-306,-287,-288,-289,-286,288,294,294,141,298,299,-292,-315,-290,-291,-304,-305,-307,304,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,238,-340,-28,322,82,238,-184,-186,-152,-339,82,-145,-147,351,238,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-315,141,362,238,366,367,238,372,238,-77,-38,-340,238,-340,-28,-73,-329,-69,238,141,141,141,141,141,141,141,141,141,141,141,141,141,141,141,141,141,141,238,238,-300,-301,238,238,-334,-335,-336,-337,-287,238,238,-35,-36,322,449,322,-340,-45,458,-171,141,-154,-156,-151,-143,-144,-148,141,-146,-130,351,351,-177,-178,-221,-220,238,238,-237,238,238,238,238,238,-87,-74,238,-233,-234,-236,238,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,238,-12,141,-287,238,238,-43,-44,141,-308,-295,-296,-297,-298,-299,-31,-34,449,458,-340,322,238,238,-172,-173,-153,-155,82,141,-222,238,-224,141,528,-86,-75,238,-232,-235,-340,-201,-39,-42,-340,-68,141,-293,-294,238,-32,-33,238,-340,-28,-210,-216,-214,-287,-223,238,238,238,238,238,238,-11,-40,-41,-287,238,238,-50,-51,-212,-211,-213,-215,-225,-87,-74,-227,-228,238,-302,-340,-309,238,-46,-49,238,238,-303,-47,-48,-226,-229,238,238,-231,-230,]),'TIMES':([0,2,4,5,6,7,8,9,10,12,14,15,17,21,22,23,24,25,26,27,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,69,70,71,72,74,75,76,77,78,81,82,90,91,94,95,98,99,100,101,102,103,104,106,112,113,114,115,116,117,118,119,121,122,123,126,127,128,134,135,136,139,141,143,145,146,147,148,149,150,151,152,153,154,155,156,158,159,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,192,194,195,197,206,208,209,212,214,216,221,222,223,224,225,226,227,228,229,230,231,232,233,234,238,242,247,250,251,256,257,258,259,262,263,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,291,292,293,294,295,296,297,298,300,301,302,303,306,309,310,322,323,330,332,334,336,337,340,341,342,347,348,349,351,353,354,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,448,455,457,459,460,463,464,466,467,469,470,471,474,479,480,482,483,484,487,489,497,498,499,500,501,502,503,505,506,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,561,562,563,565,570,572,574,579,580,582,584,585,586,],[30,30,-60,-62,-63,-64,-65,-66,-67,30,-70,-71,30,-340,-340,-340,-128,-102,30,-340,-107,-340,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,30,30,-129,-134,-98,-99,-100,-101,-104,-134,30,-90,-72,147,-340,-93,-9,-10,-340,-94,-95,-103,-129,-97,30,-27,-28,-185,-96,-169,-170,-338,-149,-150,147,-76,147,30,147,-340,-328,147,-306,269,-258,-287,-288,-289,-286,-277,-279,147,147,147,147,-292,-315,-290,-291,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,306,-340,-28,30,30,147,-186,-152,-339,30,-145,-147,30,147,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-315,147,147,147,147,-277,-77,-340,400,-340,-28,-73,-329,-69,147,147,147,147,147,147,147,147,147,147,147,147,147,147,147,147,147,147,147,147,147,-300,-301,-280,147,-281,-282,-283,147,-334,-335,-336,-337,-287,147,147,30,456,-171,147,-154,-156,-151,-143,-144,-148,147,-146,-130,30,-177,-178,-221,-220,147,147,-237,147,147,147,147,147,-87,-74,147,-233,-234,-236,147,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,147,-12,147,-287,147,147,147,-308,-259,-260,-261,269,269,269,269,269,269,269,269,269,269,269,269,269,269,269,-295,-296,-297,-298,-299,-340,147,520,-172,-173,-153,-155,30,147,-222,147,-224,147,-86,-75,147,-232,-235,-340,-201,-278,-340,-68,147,-293,-294,147,-284,-285,543,-340,-28,-287,-223,147,147,147,147,147,147,-11,-287,147,147,-225,-87,-74,-227,-228,147,-302,-340,-309,147,147,147,-303,-226,-229,147,147,-231,-230,]),'TYPEID':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,62,63,64,67,68,69,70,71,72,73,74,75,76,77,78,80,81,82,90,91,96,97,98,99,100,101,102,103,104,106,112,113,114,115,116,117,118,119,121,122,123,124,125,126,127,128,129,134,137,140,141,192,193,194,196,197,203,204,205,206,207,208,209,210,211,212,213,214,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,289,290,294,298,299,304,311,312,313,318,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,466,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[35,35,-60,-62,-63,-64,-65,-66,-67,35,89,-70,-71,-52,-340,-340,-340,-128,-102,35,-340,-29,-107,-340,-125,-126,-127,-129,-241,119,123,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-157,-158,-61,35,-91,89,35,-129,-134,35,-98,-99,-100,-101,-104,89,-134,89,-90,-72,35,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-183,-27,-28,-185,-96,-169,-170,-338,-149,-150,35,35,35,-76,35,-92,89,35,-30,35,324,35,89,-184,-186,35,35,35,-152,-159,-339,89,-162,-163,-145,35,-147,35,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,35,-77,-73,-69,432,434,35,35,35,35,-35,-36,35,324,35,-171,35,-154,35,-156,-151,-160,-143,-144,-148,-146,-130,35,-177,-178,-221,-220,-237,-87,-84,35,-233,-234,-236,-31,-34,35,35,-172,-173,-153,-155,-161,89,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'ENUM':([0,2,4,5,6,7,8,9,10,11,14,15,19,21,22,23,26,27,28,29,34,50,51,52,53,54,55,56,57,58,59,60,64,67,68,70,71,72,73,90,91,96,97,98,99,100,101,102,103,112,116,117,121,124,125,126,127,128,129,137,140,141,193,197,203,204,205,207,208,210,211,213,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,333,335,338,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[36,36,-60,-62,-63,-64,-65,-66,-67,36,-70,-71,-52,-340,-340,-340,36,-340,-29,-107,-340,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,36,-91,36,-340,-134,36,-90,-72,36,-53,-93,-9,-10,-340,-94,-95,-97,-185,-96,-338,36,36,36,-76,36,-92,36,-30,36,36,-186,36,36,36,-159,-339,-162,-163,36,36,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,36,-77,-73,-69,36,36,36,36,-35,-36,36,36,36,36,-160,-130,36,-177,-178,-221,-220,-237,-87,-84,36,-233,-234,-236,-31,-34,36,36,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'VOID':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[38,38,-60,-62,-63,-64,-65,-66,-67,38,38,-70,-71,-52,-340,-340,-340,-128,-102,38,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,38,-91,38,38,-129,-134,38,-98,-99,-100,-101,-104,-134,-90,-72,38,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,38,38,38,-76,38,-92,38,-30,38,38,38,-186,38,38,38,-152,-159,-339,38,-162,-163,-145,38,-147,38,38,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,38,-77,-73,-69,38,38,38,38,-35,-36,38,38,-171,38,-154,38,-156,-151,-160,-143,-144,-148,-146,-130,38,-177,-178,-221,-220,-237,-87,-84,38,-233,-234,-236,-31,-34,38,38,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'_BOOL':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[39,39,-60,-62,-63,-64,-65,-66,-67,39,39,-70,-71,-52,-340,-340,-340,-128,-102,39,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,39,-91,39,39,-129,-134,39,-98,-99,-100,-101,-104,-134,-90,-72,39,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,39,39,39,-76,39,-92,39,-30,39,39,39,-186,39,39,39,-152,-159,-339,39,-162,-163,-145,39,-147,39,39,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,39,-77,-73,-69,39,39,39,39,-35,-36,39,39,-171,39,-154,39,-156,-151,-160,-143,-144,-148,-146,-130,39,-177,-178,-221,-220,-237,-87,-84,39,-233,-234,-236,-31,-34,39,39,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'CHAR':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[40,40,-60,-62,-63,-64,-65,-66,-67,40,40,-70,-71,-52,-340,-340,-340,-128,-102,40,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,40,-91,40,40,-129,-134,40,-98,-99,-100,-101,-104,-134,-90,-72,40,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,40,40,40,-76,40,-92,40,-30,40,40,40,-186,40,40,40,-152,-159,-339,40,-162,-163,-145,40,-147,40,40,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,40,-77,-73,-69,40,40,40,40,-35,-36,40,40,-171,40,-154,40,-156,-151,-160,-143,-144,-148,-146,-130,40,-177,-178,-221,-220,-237,-87,-84,40,-233,-234,-236,-31,-34,40,40,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'SHORT':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[41,41,-60,-62,-63,-64,-65,-66,-67,41,41,-70,-71,-52,-340,-340,-340,-128,-102,41,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,41,-91,41,41,-129,-134,41,-98,-99,-100,-101,-104,-134,-90,-72,41,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,41,41,41,-76,41,-92,41,-30,41,41,41,-186,41,41,41,-152,-159,-339,41,-162,-163,-145,41,-147,41,41,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,41,-77,-73,-69,41,41,41,41,-35,-36,41,41,-171,41,-154,41,-156,-151,-160,-143,-144,-148,-146,-130,41,-177,-178,-221,-220,-237,-87,-84,41,-233,-234,-236,-31,-34,41,41,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'INT':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[42,42,-60,-62,-63,-64,-65,-66,-67,42,42,-70,-71,-52,-340,-340,-340,-128,-102,42,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,42,-91,42,42,-129,-134,42,-98,-99,-100,-101,-104,-134,-90,-72,42,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,42,42,42,-76,42,-92,42,-30,42,42,42,-186,42,42,42,-152,-159,-339,42,-162,-163,-145,42,-147,42,42,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,42,-77,-73,-69,42,42,42,42,-35,-36,42,42,-171,42,-154,42,-156,-151,-160,-143,-144,-148,-146,-130,42,-177,-178,-221,-220,-237,-87,-84,42,-233,-234,-236,-31,-34,42,42,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'LONG':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[43,43,-60,-62,-63,-64,-65,-66,-67,43,43,-70,-71,-52,-340,-340,-340,-128,-102,43,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,43,-91,43,43,-129,-134,43,-98,-99,-100,-101,-104,-134,-90,-72,43,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,43,43,43,-76,43,-92,43,-30,43,43,43,-186,43,43,43,-152,-159,-339,43,-162,-163,-145,43,-147,43,43,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,43,-77,-73,-69,43,43,43,43,-35,-36,43,43,-171,43,-154,43,-156,-151,-160,-143,-144,-148,-146,-130,43,-177,-178,-221,-220,-237,-87,-84,43,-233,-234,-236,-31,-34,43,43,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'FLOAT':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[44,44,-60,-62,-63,-64,-65,-66,-67,44,44,-70,-71,-52,-340,-340,-340,-128,-102,44,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,44,-91,44,44,-129,-134,44,-98,-99,-100,-101,-104,-134,-90,-72,44,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,44,44,44,-76,44,-92,44,-30,44,44,44,-186,44,44,44,-152,-159,-339,44,-162,-163,-145,44,-147,44,44,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,44,-77,-73,-69,44,44,44,44,-35,-36,44,44,-171,44,-154,44,-156,-151,-160,-143,-144,-148,-146,-130,44,-177,-178,-221,-220,-237,-87,-84,44,-233,-234,-236,-31,-34,44,44,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'DOUBLE':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[45,45,-60,-62,-63,-64,-65,-66,-67,45,45,-70,-71,-52,-340,-340,-340,-128,-102,45,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,45,-91,45,45,-129,-134,45,-98,-99,-100,-101,-104,-134,-90,-72,45,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,45,45,45,-76,45,-92,45,-30,45,45,45,-186,45,45,45,-152,-159,-339,45,-162,-163,-145,45,-147,45,45,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,45,-77,-73,-69,45,45,45,45,-35,-36,45,45,-171,45,-154,45,-156,-151,-160,-143,-144,-148,-146,-130,45,-177,-178,-221,-220,-237,-87,-84,45,-233,-234,-236,-31,-34,45,45,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'_COMPLEX':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[46,46,-60,-62,-63,-64,-65,-66,-67,46,46,-70,-71,-52,-340,-340,-340,-128,-102,46,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,46,-91,46,46,-129,-134,46,-98,-99,-100,-101,-104,-134,-90,-72,46,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,46,46,46,-76,46,-92,46,-30,46,46,46,-186,46,46,46,-152,-159,-339,46,-162,-163,-145,46,-147,46,46,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,46,-77,-73,-69,46,46,46,46,-35,-36,46,46,-171,46,-154,46,-156,-151,-160,-143,-144,-148,-146,-130,46,-177,-178,-221,-220,-237,-87,-84,46,-233,-234,-236,-31,-34,46,46,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'SIGNED':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[47,47,-60,-62,-63,-64,-65,-66,-67,47,47,-70,-71,-52,-340,-340,-340,-128,-102,47,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,47,-91,47,47,-129,-134,47,-98,-99,-100,-101,-104,-134,-90,-72,47,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,47,47,47,-76,47,-92,47,-30,47,47,47,-186,47,47,47,-152,-159,-339,47,-162,-163,-145,47,-147,47,47,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,47,-77,-73,-69,47,47,47,47,-35,-36,47,47,-171,47,-154,47,-156,-151,-160,-143,-144,-148,-146,-130,47,-177,-178,-221,-220,-237,-87,-84,47,-233,-234,-236,-31,-34,47,47,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'UNSIGNED':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[48,48,-60,-62,-63,-64,-65,-66,-67,48,48,-70,-71,-52,-340,-340,-340,-128,-102,48,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,48,-91,48,48,-129,-134,48,-98,-99,-100,-101,-104,-134,-90,-72,48,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,48,48,48,-76,48,-92,48,-30,48,48,48,-186,48,48,48,-152,-159,-339,48,-162,-163,-145,48,-147,48,48,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,48,-77,-73,-69,48,48,48,48,-35,-36,48,48,-171,48,-154,48,-156,-151,-160,-143,-144,-148,-146,-130,48,-177,-178,-221,-220,-237,-87,-84,48,-233,-234,-236,-31,-34,48,48,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'__INT128':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,96,97,98,99,100,101,102,103,104,106,112,116,117,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[49,49,-60,-62,-63,-64,-65,-66,-67,49,49,-70,-71,-52,-340,-340,-340,-128,-102,49,-340,-29,-107,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,49,-91,49,49,-129,-134,49,-98,-99,-100,-101,-104,-134,-90,-72,49,-53,-93,-9,-10,-340,-94,-95,-103,-129,-97,-185,-96,-169,-170,-338,-149,-150,49,49,49,-76,49,-92,49,-30,49,49,49,-186,49,49,49,-152,-159,-339,49,-162,-163,-145,49,-147,49,49,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,49,-77,-73,-69,49,49,49,49,-35,-36,49,49,-171,49,-154,49,-156,-151,-160,-143,-144,-148,-146,-130,49,-177,-178,-221,-220,-237,-87,-84,49,-233,-234,-236,-31,-34,49,49,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'_ATOMIC':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,70,71,72,73,74,75,76,77,78,81,90,91,95,96,97,98,99,100,101,102,103,104,106,112,115,116,117,118,119,121,122,123,124,125,126,127,128,129,136,137,140,141,183,184,192,193,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,258,259,262,267,294,298,299,304,311,312,313,322,323,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,448,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,511,512,524,552,553,554,555,556,579,580,585,586,],[50,50,-60,-62,-63,-64,-65,-66,-67,72,81,-70,-71,-52,72,72,72,-128,-102,109,72,-29,-107,81,-125,-126,-127,72,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,72,-91,81,109,72,-134,72,-98,-99,-100,-101,-104,-134,-90,-72,81,50,-53,-93,-9,-10,72,-94,-95,-103,-129,-97,81,-185,-96,-169,-170,-338,-149,-150,50,50,50,-76,72,-92,81,50,-30,50,81,81,81,109,-186,50,50,50,-152,-159,-339,81,-162,-163,-145,72,-147,81,72,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,50,-77,81,81,-73,-69,50,50,50,50,-35,-36,50,50,81,-171,50,-154,50,-156,-151,-160,-143,-144,-148,-146,-130,50,-177,-178,-221,-220,-237,-87,-84,72,-233,-234,-236,-31,-34,81,50,50,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,81,81,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'CONST':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,95,96,97,101,104,106,115,116,118,119,121,122,123,124,125,126,127,128,129,136,137,140,141,183,184,192,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,258,259,262,267,294,298,299,304,311,312,313,322,323,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,448,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,511,512,524,552,553,554,555,556,579,580,585,586,],[51,51,-60,-62,-63,-64,-65,-66,-67,51,51,-70,-71,-52,51,51,51,-128,-102,51,-29,-107,51,-125,-126,-127,51,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,51,-91,51,51,-134,51,-98,-99,-100,-101,-104,-134,-90,-72,51,51,-53,51,-103,-129,51,-185,-169,-170,-338,-149,-150,51,51,51,-76,51,-92,51,51,-30,51,51,51,51,-186,51,51,51,-152,-159,-339,51,-162,-163,-145,51,-147,51,51,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,51,-77,51,51,-73,-69,51,51,51,51,-35,-36,51,51,51,-171,51,-154,51,-156,-151,-160,-143,-144,-148,-146,-130,51,-177,-178,-221,-220,-237,-87,-84,51,-233,-234,-236,-31,-34,51,51,51,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,51,51,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'RESTRICT':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,95,96,97,101,104,106,115,116,118,119,121,122,123,124,125,126,127,128,129,136,137,140,141,183,184,192,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,258,259,262,267,294,298,299,304,311,312,313,322,323,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,448,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,511,512,524,552,553,554,555,556,579,580,585,586,],[52,52,-60,-62,-63,-64,-65,-66,-67,52,52,-70,-71,-52,52,52,52,-128,-102,52,-29,-107,52,-125,-126,-127,52,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,52,-91,52,52,-134,52,-98,-99,-100,-101,-104,-134,-90,-72,52,52,-53,52,-103,-129,52,-185,-169,-170,-338,-149,-150,52,52,52,-76,52,-92,52,52,-30,52,52,52,52,-186,52,52,52,-152,-159,-339,52,-162,-163,-145,52,-147,52,52,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,52,-77,52,52,-73,-69,52,52,52,52,-35,-36,52,52,52,-171,52,-154,52,-156,-151,-160,-143,-144,-148,-146,-130,52,-177,-178,-221,-220,-237,-87,-84,52,-233,-234,-236,-31,-34,52,52,52,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,52,52,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'VOLATILE':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,95,96,97,101,104,106,115,116,118,119,121,122,123,124,125,126,127,128,129,136,137,140,141,183,184,192,197,203,204,205,206,207,208,209,210,211,212,213,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,258,259,262,267,294,298,299,304,311,312,313,322,323,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,448,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,511,512,524,552,553,554,555,556,579,580,585,586,],[53,53,-60,-62,-63,-64,-65,-66,-67,53,53,-70,-71,-52,53,53,53,-128,-102,53,-29,-107,53,-125,-126,-127,53,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,53,-91,53,53,-134,53,-98,-99,-100,-101,-104,-134,-90,-72,53,53,-53,53,-103,-129,53,-185,-169,-170,-338,-149,-150,53,53,53,-76,53,-92,53,53,-30,53,53,53,53,-186,53,53,53,-152,-159,-339,53,-162,-163,-145,53,-147,53,53,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,53,-77,53,53,-73,-69,53,53,53,53,-35,-36,53,53,53,-171,53,-154,53,-156,-151,-160,-143,-144,-148,-146,-130,53,-177,-178,-221,-220,-237,-87,-84,53,-233,-234,-236,-31,-34,53,53,53,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,53,53,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'AUTO':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,96,97,101,104,106,118,119,121,122,123,127,128,129,137,140,192,206,208,221,222,223,224,225,226,227,228,229,230,231,232,251,262,267,311,312,313,322,330,334,336,337,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[54,54,-60,-62,-63,-64,-65,-66,-67,54,54,-70,-71,-52,54,54,54,-128,-102,54,-29,-107,-125,-126,-127,54,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,54,-91,54,54,-134,54,-98,-99,-100,-101,-104,-134,-90,-72,54,-53,54,-103,-129,-169,-170,-338,-149,-150,-76,54,-92,54,-30,54,-152,-339,54,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-77,-73,-69,-35,-36,54,54,-171,-154,-156,-151,-130,54,-177,-178,-221,-220,-237,-87,-84,54,-233,-234,-236,-31,-34,54,54,-172,-173,-153,-155,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'REGISTER':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,96,97,101,104,106,118,119,121,122,123,127,128,129,137,140,192,206,208,221,222,223,224,225,226,227,228,229,230,231,232,251,262,267,311,312,313,322,330,334,336,337,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[55,55,-60,-62,-63,-64,-65,-66,-67,55,55,-70,-71,-52,55,55,55,-128,-102,55,-29,-107,-125,-126,-127,55,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,55,-91,55,55,-134,55,-98,-99,-100,-101,-104,-134,-90,-72,55,-53,55,-103,-129,-169,-170,-338,-149,-150,-76,55,-92,55,-30,55,-152,-339,55,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-77,-73,-69,-35,-36,55,55,-171,-154,-156,-151,-130,55,-177,-178,-221,-220,-237,-87,-84,55,-233,-234,-236,-31,-34,55,55,-172,-173,-153,-155,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'STATIC':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,95,96,97,101,104,106,116,118,119,121,122,123,127,128,129,136,137,140,184,192,197,206,208,221,222,223,224,225,226,227,228,229,230,231,232,251,259,262,267,311,312,313,322,330,334,336,337,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,448,449,458,459,460,463,464,469,471,479,480,483,484,499,508,509,512,524,552,553,554,555,556,579,580,585,586,],[29,29,-60,-62,-63,-64,-65,-66,-67,29,29,-70,-71,-52,29,29,29,-128,-102,29,-29,-107,-125,-126,-127,29,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,29,-91,29,29,-134,29,-98,-99,-100,-101,-104,-134,-90,-72,183,29,-53,29,-103,-129,-185,-169,-170,-338,-149,-150,-76,29,-92,258,29,-30,310,29,-186,-152,-339,29,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-77,402,-73,-69,-35,-36,29,29,-171,-154,-156,-151,-130,29,-177,-178,-221,-220,-237,-87,-84,29,-233,-234,-236,-31,-34,511,29,29,-172,-173,-153,-155,-222,-224,-86,-84,-232,-235,-68,-32,-33,545,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'EXTERN':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,96,97,101,104,106,118,119,121,122,123,127,128,129,137,140,192,206,208,221,222,223,224,225,226,227,228,229,230,231,232,251,262,267,311,312,313,322,330,334,336,337,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[56,56,-60,-62,-63,-64,-65,-66,-67,56,56,-70,-71,-52,56,56,56,-128,-102,56,-29,-107,-125,-126,-127,56,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,56,-91,56,56,-134,56,-98,-99,-100,-101,-104,-134,-90,-72,56,-53,56,-103,-129,-169,-170,-338,-149,-150,-76,56,-92,56,-30,56,-152,-339,56,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-77,-73,-69,-35,-36,56,56,-171,-154,-156,-151,-130,56,-177,-178,-221,-220,-237,-87,-84,56,-233,-234,-236,-31,-34,56,56,-172,-173,-153,-155,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'TYPEDEF':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,96,97,101,104,106,118,119,121,122,123,127,128,129,137,140,192,206,208,221,222,223,224,225,226,227,228,229,230,231,232,251,262,267,311,312,313,322,330,334,336,337,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[57,57,-60,-62,-63,-64,-65,-66,-67,57,57,-70,-71,-52,57,57,57,-128,-102,57,-29,-107,-125,-126,-127,57,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,57,-91,57,57,-134,57,-98,-99,-100,-101,-104,-134,-90,-72,57,-53,57,-103,-129,-169,-170,-338,-149,-150,-76,57,-92,57,-30,57,-152,-339,57,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-77,-73,-69,-35,-36,57,57,-171,-154,-156,-151,-130,57,-177,-178,-221,-220,-237,-87,-84,57,-233,-234,-236,-31,-34,57,57,-172,-173,-153,-155,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'_THREAD_LOCAL':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,96,97,101,104,106,118,119,121,122,123,127,128,129,137,140,192,206,208,221,222,223,224,225,226,227,228,229,230,231,232,251,262,267,311,312,313,322,330,334,336,337,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[58,58,-60,-62,-63,-64,-65,-66,-67,58,58,-70,-71,-52,58,58,58,-128,-102,58,-29,-107,-125,-126,-127,58,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,58,-91,58,58,-134,58,-98,-99,-100,-101,-104,-134,-90,-72,58,-53,58,-103,-129,-169,-170,-338,-149,-150,-76,58,-92,58,-30,58,-152,-339,58,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-77,-73,-69,-35,-36,58,58,-171,-154,-156,-151,-130,58,-177,-178,-221,-220,-237,-87,-84,58,-233,-234,-236,-31,-34,58,58,-172,-173,-153,-155,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'INLINE':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,96,97,101,104,106,118,119,121,122,123,127,128,129,137,140,192,206,208,221,222,223,224,225,226,227,228,229,230,231,232,251,262,267,311,312,313,322,330,334,336,337,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[59,59,-60,-62,-63,-64,-65,-66,-67,59,59,-70,-71,-52,59,59,59,-128,-102,59,-29,-107,-125,-126,-127,59,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,59,-91,59,59,-134,59,-98,-99,-100,-101,-104,-134,-90,-72,59,-53,59,-103,-129,-169,-170,-338,-149,-150,-76,59,-92,59,-30,59,-152,-339,59,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-77,-73,-69,-35,-36,59,59,-171,-154,-156,-151,-130,59,-177,-178,-221,-220,-237,-87,-84,59,-233,-234,-236,-31,-34,59,59,-172,-173,-153,-155,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'_NORETURN':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,96,97,101,104,106,118,119,121,122,123,127,128,129,137,140,192,206,208,221,222,223,224,225,226,227,228,229,230,231,232,251,262,267,311,312,313,322,330,334,336,337,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[60,60,-60,-62,-63,-64,-65,-66,-67,60,60,-70,-71,-52,60,60,60,-128,-102,60,-29,-107,-125,-126,-127,60,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,60,-91,60,60,-134,60,-98,-99,-100,-101,-104,-134,-90,-72,60,-53,60,-103,-129,-169,-170,-338,-149,-150,-76,60,-92,60,-30,60,-152,-339,60,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-77,-73,-69,-35,-36,60,60,-171,-154,-156,-151,-130,60,-177,-178,-221,-220,-237,-87,-84,60,-233,-234,-236,-31,-34,60,60,-172,-173,-153,-155,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'_ALIGNAS':([0,2,4,5,6,7,8,9,10,11,12,14,15,19,21,22,23,24,25,27,28,29,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,64,67,68,69,71,72,73,74,75,76,77,78,81,90,91,96,97,101,104,106,118,119,121,122,123,124,125,126,127,128,129,137,140,141,192,203,204,205,206,207,208,209,210,211,212,214,216,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,330,333,334,335,336,337,338,340,341,342,348,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,459,460,463,464,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[61,61,-60,-62,-63,-64,-65,-66,-67,61,61,-70,-71,-52,61,61,61,-128,-102,61,-29,-107,-125,-126,-127,61,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,61,-91,61,61,-134,61,-98,-99,-100,-101,-104,-134,-90,-72,61,-53,61,-103,-129,-169,-170,-338,-149,-150,61,61,61,-76,61,-92,61,-30,61,61,61,61,61,-152,-159,-339,61,-162,-163,-145,-147,61,61,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,61,-77,-73,-69,61,61,61,61,-35,-36,61,61,-171,61,-154,61,-156,-151,-160,-143,-144,-148,-146,-130,61,-177,-178,-221,-220,-237,-87,-84,61,-233,-234,-236,-31,-34,61,61,-172,-173,-153,-155,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'STRUCT':([0,2,4,5,6,7,8,9,10,11,14,15,19,21,22,23,26,27,28,29,34,50,51,52,53,54,55,56,57,58,59,60,64,67,68,70,71,72,73,90,91,96,97,98,99,100,101,102,103,112,116,117,121,124,125,126,127,128,129,137,140,141,193,197,203,204,205,207,208,210,211,213,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,333,335,338,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[62,62,-60,-62,-63,-64,-65,-66,-67,62,-70,-71,-52,-340,-340,-340,62,-340,-29,-107,-340,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,62,-91,62,-340,-134,62,-90,-72,62,-53,-93,-9,-10,-340,-94,-95,-97,-185,-96,-338,62,62,62,-76,62,-92,62,-30,62,62,-186,62,62,62,-159,-339,-162,-163,62,62,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,62,-77,-73,-69,62,62,62,62,-35,-36,62,62,62,62,-160,-130,62,-177,-178,-221,-220,-237,-87,-84,62,-233,-234,-236,-31,-34,62,62,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'UNION':([0,2,4,5,6,7,8,9,10,11,14,15,19,21,22,23,26,27,28,29,34,50,51,52,53,54,55,56,57,58,59,60,64,67,68,70,71,72,73,90,91,96,97,98,99,100,101,102,103,112,116,117,121,124,125,126,127,128,129,137,140,141,193,197,203,204,205,207,208,210,211,213,221,222,223,224,225,226,227,228,229,230,231,232,238,251,262,267,294,298,299,304,311,312,313,322,333,335,338,349,351,353,354,355,356,361,370,371,372,374,375,377,439,440,449,458,465,469,471,479,480,483,484,499,508,509,524,552,553,554,555,556,579,580,585,586,],[63,63,-60,-62,-63,-64,-65,-66,-67,63,-70,-71,-52,-340,-340,-340,63,-340,-29,-107,-340,-134,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-61,63,-91,63,-340,-134,63,-90,-72,63,-53,-93,-9,-10,-340,-94,-95,-97,-185,-96,-338,63,63,63,-76,63,-92,63,-30,63,63,-186,63,63,63,-159,-339,-162,-163,63,63,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,63,-77,-73,-69,63,63,63,63,-35,-36,63,63,63,63,-160,-130,63,-177,-178,-221,-220,-237,-87,-84,63,-233,-234,-236,-31,-34,63,63,-161,-222,-224,-86,-84,-232,-235,-68,-32,-33,-223,-225,-87,-84,-227,-228,-226,-229,-231,-230,]),'LBRACE':([11,15,19,28,36,37,62,63,65,66,67,68,73,90,91,97,118,119,121,122,123,128,129,131,135,140,195,208,221,222,223,224,225,226,227,228,229,230,231,232,238,242,256,262,267,311,312,355,356,358,360,361,369,370,371,374,375,377,392,393,394,405,439,440,469,470,471,474,479,480,483,484,487,489,498,499,504,505,508,509,524,525,526,527,532,533,552,553,554,555,556,562,570,579,580,582,584,585,586,],[-340,-71,-52,-29,121,121,-157,-158,121,-7,-8,-91,-340,-90,-72,-53,121,121,-338,121,121,121,-92,121,121,-30,121,-339,121,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,121,121,-340,-73,-69,-35,-36,-221,-220,121,121,-237,121,-87,-74,-233,-234,-236,-11,121,-12,121,-31,-34,-222,121,-224,121,-86,-75,-232,-235,-340,-201,-340,-68,121,121,-32,-33,-223,121,121,121,121,-11,-225,-87,-74,-227,-228,-340,121,-226,-229,121,121,-231,-230,]),'RBRACE':([15,90,91,121,124,128,139,143,144,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,200,201,202,203,204,205,207,208,210,211,219,220,221,222,223,224,225,226,227,228,229,230,231,232,249,250,255,256,262,263,267,291,292,293,295,296,297,300,301,302,303,328,329,331,333,335,338,355,356,361,370,371,374,375,377,390,391,392,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,461,462,465,469,471,473,479,480,483,484,485,486,487,488,497,499,501,502,505,506,524,531,537,538,552,553,554,555,556,560,561,562,563,574,579,580,585,586,],[-71,-90,-72,-338,208,-340,-328,-306,-255,-256,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,208,-174,-179,208,208,208,-159,-339,-162,-163,208,-5,-6,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-242,-277,-196,-340,-73,-329,-69,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,208,208,-175,208,208,-160,-221,-220,-237,-87,-84,-233,-234,-236,208,-22,-21,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,-275,-276,-295,-296,-297,-298,-299,-176,-180,-161,-222,-224,-240,-86,-84,-232,-235,-243,-197,208,-199,-278,-68,-293,-294,-284,-285,-223,-198,208,-257,-225,-87,-84,-227,-228,-200,-302,208,-309,-303,-226,-229,-231,-230,]),'CASE':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,234,-339,234,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,234,-73,-69,-221,-220,234,234,-237,234,-87,-74,-233,-234,-236,-222,234,-224,-86,-75,-232,-235,-68,-223,234,234,234,-225,-87,-74,-227,-228,234,-226,-229,234,234,-231,-230,]),'DEFAULT':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,235,-339,235,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,235,-73,-69,-221,-220,235,235,-237,235,-87,-74,-233,-234,-236,-222,235,-224,-86,-75,-232,-235,-68,-223,235,235,235,-225,-87,-74,-227,-228,235,-226,-229,235,235,-231,-230,]),'IF':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,237,-339,237,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,237,-73,-69,-221,-220,237,237,-237,237,-87,-74,-233,-234,-236,-222,237,-224,-86,-75,-232,-235,-68,-223,237,237,237,-225,-87,-74,-227,-228,237,-226,-229,237,237,-231,-230,]),'SWITCH':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,240,-339,240,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,240,-73,-69,-221,-220,240,240,-237,240,-87,-74,-233,-234,-236,-222,240,-224,-86,-75,-232,-235,-68,-223,240,240,240,-225,-87,-74,-227,-228,240,-226,-229,240,240,-231,-230,]),'WHILE':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,368,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,241,-339,241,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,241,-73,-69,-221,-220,241,241,-237,478,241,-87,-74,-233,-234,-236,-222,241,-224,-86,-75,-232,-235,-68,-223,241,241,241,-225,-87,-74,-227,-228,241,-226,-229,241,241,-231,-230,]),'DO':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,242,-339,242,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,242,-73,-69,-221,-220,242,242,-237,242,-87,-74,-233,-234,-236,-222,242,-224,-86,-75,-232,-235,-68,-223,242,242,242,-225,-87,-74,-227,-228,242,-226,-229,242,242,-231,-230,]),'FOR':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,243,-339,243,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,243,-73,-69,-221,-220,243,243,-237,243,-87,-74,-233,-234,-236,-222,243,-224,-86,-75,-232,-235,-68,-223,243,243,243,-225,-87,-74,-227,-228,243,-226,-229,243,243,-231,-230,]),'GOTO':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,244,-339,244,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,244,-73,-69,-221,-220,244,244,-237,244,-87,-74,-233,-234,-236,-222,244,-224,-86,-75,-232,-235,-68,-223,244,244,244,-225,-87,-74,-227,-228,244,-226,-229,244,244,-231,-230,]),'BREAK':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,245,-339,245,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,245,-73,-69,-221,-220,245,245,-237,245,-87,-74,-233,-234,-236,-222,245,-224,-86,-75,-232,-235,-68,-223,245,245,245,-225,-87,-74,-227,-228,245,-226,-229,245,245,-231,-230,]),'CONTINUE':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,246,-339,246,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,246,-73,-69,-221,-220,246,246,-237,246,-87,-74,-233,-234,-236,-222,246,-224,-86,-75,-232,-235,-68,-223,246,246,246,-225,-87,-74,-227,-228,246,-226,-229,246,246,-231,-230,]),'RETURN':([15,90,91,121,128,208,221,222,223,224,225,226,227,228,229,230,231,232,242,262,267,355,356,358,360,361,369,370,371,374,375,377,469,470,471,479,480,483,484,499,524,525,526,527,552,553,554,555,556,570,579,580,582,584,585,586,],[-71,-90,-72,-338,247,-339,247,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,247,-73,-69,-221,-220,247,247,-237,247,-87,-74,-233,-234,-236,-222,247,-224,-86,-75,-232,-235,-68,-223,247,247,247,-225,-87,-74,-227,-228,247,-226,-229,247,247,-231,-230,]),'PLUSPLUS':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,139,141,143,147,148,149,150,152,153,154,155,156,158,159,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,233,234,238,242,247,256,257,258,259,262,263,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,291,292,294,298,300,301,302,303,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,406,429,431,432,433,434,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,501,502,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,561,562,563,565,570,572,574,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,153,-340,-27,-28,-185,-338,153,153,153,-340,-328,153,-306,-287,-288,-289,-286,291,153,153,153,153,-292,-315,-290,-291,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,153,-340,-28,153,-186,-339,153,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-315,153,153,153,153,-340,153,-340,-28,-73,-329,-69,153,153,153,153,153,153,153,153,153,153,153,153,153,153,153,153,153,153,153,153,153,-300,-301,153,153,-334,-335,-336,-337,-287,153,153,-340,153,153,-221,-220,153,153,-237,153,153,153,153,153,-87,-74,153,-233,-234,-236,153,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,153,-12,153,-287,153,153,153,-308,-295,-296,-297,-298,-299,-340,153,153,153,-222,153,-224,153,-86,-75,153,-232,-235,-340,-201,-340,-68,153,-293,-294,153,153,-340,-28,-287,-223,153,153,153,153,153,153,-11,-287,153,153,-225,-87,-74,-227,-228,153,-302,-340,-309,153,153,153,-303,-226,-229,153,153,-231,-230,]),'MINUSMINUS':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,139,141,143,147,148,149,150,152,153,154,155,156,158,159,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,233,234,238,242,247,256,257,258,259,262,263,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,291,292,294,298,300,301,302,303,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,406,429,431,432,433,434,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,501,502,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,561,562,563,565,570,572,574,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,154,-340,-27,-28,-185,-338,154,154,154,-340,-328,154,-306,-287,-288,-289,-286,292,154,154,154,154,-292,-315,-290,-291,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,154,-340,-28,154,-186,-339,154,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-315,154,154,154,154,-340,154,-340,-28,-73,-329,-69,154,154,154,154,154,154,154,154,154,154,154,154,154,154,154,154,154,154,154,154,154,-300,-301,154,154,-334,-335,-336,-337,-287,154,154,-340,154,154,-221,-220,154,154,-237,154,154,154,154,154,-87,-74,154,-233,-234,-236,154,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,154,-12,154,-287,154,154,154,-308,-295,-296,-297,-298,-299,-340,154,154,154,-222,154,-224,154,-86,-75,154,-232,-235,-340,-201,-340,-68,154,-293,-294,154,154,-340,-28,-287,-223,154,154,154,154,154,154,-11,-287,154,154,-225,-87,-74,-227,-228,154,-302,-340,-309,154,154,154,-303,-226,-229,154,154,-231,-230,]),'SIZEOF':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,156,-340,-27,-28,-185,-338,156,156,156,-340,156,-287,-288,-289,-286,156,156,156,156,-290,-291,156,-340,-28,156,-186,-339,156,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,156,156,156,156,-340,156,-340,-28,-73,-69,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,-287,156,156,-340,156,156,-221,-220,156,156,-237,156,156,156,156,156,-87,-74,156,-233,-234,-236,156,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,156,-12,156,-287,156,156,156,-340,156,156,156,-222,156,-224,156,-86,-75,156,-232,-235,-340,-201,-340,-68,156,156,156,-340,-28,-287,-223,156,156,156,156,156,156,-11,-287,156,156,-225,-87,-74,-227,-228,156,-340,156,156,156,-226,-229,156,156,-231,-230,]),'_ALIGNOF':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,157,-340,-27,-28,-185,-338,157,157,157,-340,157,-287,-288,-289,-286,157,157,157,157,-290,-291,157,-340,-28,157,-186,-339,157,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,157,157,157,157,-340,157,-340,-28,-73,-69,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,157,-287,157,157,-340,157,157,-221,-220,157,157,-237,157,157,157,157,157,-87,-74,157,-233,-234,-236,157,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,157,-12,157,-287,157,157,157,-340,157,157,157,-222,157,-224,157,-86,-75,157,-232,-235,-340,-201,-340,-68,157,157,157,-340,-28,-287,-223,157,157,157,157,157,157,-11,-287,157,157,-225,-87,-74,-227,-228,157,-340,157,157,157,-226,-229,157,157,-231,-230,]),'AND':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,139,141,143,145,146,147,148,149,150,151,152,153,154,155,156,158,159,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,233,234,238,242,247,250,256,257,258,259,262,263,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,291,292,293,294,295,296,297,298,300,301,302,303,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,497,498,499,500,501,502,503,505,506,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,561,562,563,565,570,572,574,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,150,-340,-27,-28,-185,-338,150,150,150,-340,-328,150,-306,282,-258,-287,-288,-289,-286,-277,-279,150,150,150,150,-292,-315,-290,-291,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,150,-340,-28,150,-186,-339,150,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-315,150,150,150,150,-277,-340,150,-340,-28,-73,-329,-69,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,-300,-301,-280,150,-281,-282,-283,150,-334,-335,-336,-337,-287,150,150,-340,150,150,-221,-220,150,150,-237,150,150,150,150,150,-87,-74,150,-233,-234,-236,150,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,150,-12,150,-287,150,150,150,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,282,282,282,282,-295,-296,-297,-298,-299,-340,150,150,150,-222,150,-224,150,-86,-75,150,-232,-235,-340,-201,-278,-340,-68,150,-293,-294,150,-284,-285,150,-340,-28,-287,-223,150,150,150,150,150,150,-11,-287,150,150,-225,-87,-74,-227,-228,150,-302,-340,-309,150,150,150,-303,-226,-229,150,150,-231,-230,]),'PLUS':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,139,141,143,145,146,147,148,149,150,151,152,153,154,155,156,158,159,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,233,234,238,242,247,250,256,257,258,259,262,263,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,291,292,293,294,295,296,297,298,300,301,302,303,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,497,498,499,500,501,502,503,505,506,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,561,562,563,565,570,572,574,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,148,-340,-27,-28,-185,-338,148,148,148,-340,-328,148,-306,272,-258,-287,-288,-289,-286,-277,-279,148,148,148,148,-292,-315,-290,-291,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,148,-340,-28,148,-186,-339,148,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-315,148,148,148,148,-277,-340,148,-340,-28,-73,-329,-69,148,148,148,148,148,148,148,148,148,148,148,148,148,148,148,148,148,148,148,148,148,-300,-301,-280,148,-281,-282,-283,148,-334,-335,-336,-337,-287,148,148,-340,148,148,-221,-220,148,148,-237,148,148,148,148,148,-87,-74,148,-233,-234,-236,148,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,148,-12,148,-287,148,148,148,-308,-259,-260,-261,-262,-263,272,272,272,272,272,272,272,272,272,272,272,272,272,-295,-296,-297,-298,-299,-340,148,148,148,-222,148,-224,148,-86,-75,148,-232,-235,-340,-201,-278,-340,-68,148,-293,-294,148,-284,-285,148,-340,-28,-287,-223,148,148,148,148,148,148,-11,-287,148,148,-225,-87,-74,-227,-228,148,-302,-340,-309,148,148,148,-303,-226,-229,148,148,-231,-230,]),'MINUS':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,139,141,143,145,146,147,148,149,150,151,152,153,154,155,156,158,159,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,233,234,238,242,247,250,256,257,258,259,262,263,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,291,292,293,294,295,296,297,298,300,301,302,303,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,497,498,499,500,501,502,503,505,506,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,561,562,563,565,570,572,574,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,149,-340,-27,-28,-185,-338,149,149,149,-340,-328,149,-306,273,-258,-287,-288,-289,-286,-277,-279,149,149,149,149,-292,-315,-290,-291,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,149,-340,-28,149,-186,-339,149,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,-315,149,149,149,149,-277,-340,149,-340,-28,-73,-329,-69,149,149,149,149,149,149,149,149,149,149,149,149,149,149,149,149,149,149,149,149,149,-300,-301,-280,149,-281,-282,-283,149,-334,-335,-336,-337,-287,149,149,-340,149,149,-221,-220,149,149,-237,149,149,149,149,149,-87,-74,149,-233,-234,-236,149,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,149,-12,149,-287,149,149,149,-308,-259,-260,-261,-262,-263,273,273,273,273,273,273,273,273,273,273,273,273,273,-295,-296,-297,-298,-299,-340,149,149,149,-222,149,-224,149,-86,-75,149,-232,-235,-340,-201,-278,-340,-68,149,-293,-294,149,-284,-285,149,-340,-28,-287,-223,149,149,149,149,149,149,-11,-287,149,149,-225,-87,-74,-227,-228,149,-302,-340,-309,149,149,149,-303,-226,-229,149,149,-231,-230,]),'NOT':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,160,-340,-27,-28,-185,-338,160,160,160,-340,160,-287,-288,-289,-286,160,160,160,160,-290,-291,160,-340,-28,160,-186,-339,160,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,160,160,160,160,-340,160,-340,-28,-73,-69,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,160,-287,160,160,-340,160,160,-221,-220,160,160,-237,160,160,160,160,160,-87,-74,160,-233,-234,-236,160,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,160,-12,160,-287,160,160,160,-340,160,160,160,-222,160,-224,160,-86,-75,160,-232,-235,-340,-201,-340,-68,160,160,160,-340,-28,-287,-223,160,160,160,160,160,160,-11,-287,160,160,-225,-87,-74,-227,-228,160,-340,160,160,160,-226,-229,160,160,-231,-230,]),'LNOT':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,161,-340,-27,-28,-185,-338,161,161,161,-340,161,-287,-288,-289,-286,161,161,161,161,-290,-291,161,-340,-28,161,-186,-339,161,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,161,161,161,161,-340,161,-340,-28,-73,-69,161,161,161,161,161,161,161,161,161,161,161,161,161,161,161,161,161,161,161,161,161,161,161,-287,161,161,-340,161,161,-221,-220,161,161,-237,161,161,161,161,161,-87,-74,161,-233,-234,-236,161,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,161,-12,161,-287,161,161,161,-340,161,161,161,-222,161,-224,161,-86,-75,161,-232,-235,-340,-201,-340,-68,161,161,161,-340,-28,-287,-223,161,161,161,161,161,161,-11,-287,161,161,-225,-87,-74,-227,-228,161,-340,161,161,161,-226,-229,161,161,-231,-230,]),'OFFSETOF':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,165,-340,-27,-28,-185,-338,165,165,165,-340,165,-287,-288,-289,-286,165,165,165,165,-290,-291,165,-340,-28,165,-186,-339,165,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,165,165,165,165,-340,165,-340,-28,-73,-69,165,165,165,165,165,165,165,165,165,165,165,165,165,165,165,165,165,165,165,165,165,165,165,-287,165,165,-340,165,165,-221,-220,165,165,-237,165,165,165,165,165,-87,-74,165,-233,-234,-236,165,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,165,-12,165,-287,165,165,165,-340,165,165,165,-222,165,-224,165,-86,-75,165,-232,-235,-340,-201,-340,-68,165,165,165,-340,-28,-287,-223,165,165,165,165,165,165,-11,-287,165,165,-225,-87,-74,-227,-228,165,-340,165,165,165,-226,-229,165,165,-231,-230,]),'INT_CONST_DEC':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,166,-340,-27,-28,-185,-338,166,166,166,-340,166,-287,-288,-289,-286,166,166,166,166,-290,-291,166,-340,-28,166,-186,-339,166,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,166,166,166,166,-340,166,-340,-28,-73,-69,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,166,-287,166,166,-340,166,166,-221,-220,166,166,-237,166,166,166,166,166,-87,-74,166,-233,-234,-236,166,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,166,-12,166,-287,166,166,166,-340,166,166,166,-222,166,-224,166,-86,-75,166,-232,-235,-340,-201,-340,-68,166,166,166,-340,-28,-287,-223,166,166,166,166,166,166,-11,-287,166,166,-225,-87,-74,-227,-228,166,-340,166,166,166,-226,-229,166,166,-231,-230,]),'INT_CONST_OCT':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,167,-340,-27,-28,-185,-338,167,167,167,-340,167,-287,-288,-289,-286,167,167,167,167,-290,-291,167,-340,-28,167,-186,-339,167,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,167,167,167,167,-340,167,-340,-28,-73,-69,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,167,-287,167,167,-340,167,167,-221,-220,167,167,-237,167,167,167,167,167,-87,-74,167,-233,-234,-236,167,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,167,-12,167,-287,167,167,167,-340,167,167,167,-222,167,-224,167,-86,-75,167,-232,-235,-340,-201,-340,-68,167,167,167,-340,-28,-287,-223,167,167,167,167,167,167,-11,-287,167,167,-225,-87,-74,-227,-228,167,-340,167,167,167,-226,-229,167,167,-231,-230,]),'INT_CONST_HEX':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,168,-340,-27,-28,-185,-338,168,168,168,-340,168,-287,-288,-289,-286,168,168,168,168,-290,-291,168,-340,-28,168,-186,-339,168,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,168,168,168,168,-340,168,-340,-28,-73,-69,168,168,168,168,168,168,168,168,168,168,168,168,168,168,168,168,168,168,168,168,168,168,168,-287,168,168,-340,168,168,-221,-220,168,168,-237,168,168,168,168,168,-87,-74,168,-233,-234,-236,168,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,168,-12,168,-287,168,168,168,-340,168,168,168,-222,168,-224,168,-86,-75,168,-232,-235,-340,-201,-340,-68,168,168,168,-340,-28,-287,-223,168,168,168,168,168,168,-11,-287,168,168,-225,-87,-74,-227,-228,168,-340,168,168,168,-226,-229,168,168,-231,-230,]),'INT_CONST_BIN':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,169,-340,-27,-28,-185,-338,169,169,169,-340,169,-287,-288,-289,-286,169,169,169,169,-290,-291,169,-340,-28,169,-186,-339,169,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,169,169,169,169,-340,169,-340,-28,-73,-69,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,169,-287,169,169,-340,169,169,-221,-220,169,169,-237,169,169,169,169,169,-87,-74,169,-233,-234,-236,169,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,169,-12,169,-287,169,169,169,-340,169,169,169,-222,169,-224,169,-86,-75,169,-232,-235,-340,-201,-340,-68,169,169,169,-340,-28,-287,-223,169,169,169,169,169,169,-11,-287,169,169,-225,-87,-74,-227,-228,169,-340,169,169,169,-226,-229,169,169,-231,-230,]),'INT_CONST_CHAR':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,170,-340,-27,-28,-185,-338,170,170,170,-340,170,-287,-288,-289,-286,170,170,170,170,-290,-291,170,-340,-28,170,-186,-339,170,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,170,170,170,170,-340,170,-340,-28,-73,-69,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,170,-287,170,170,-340,170,170,-221,-220,170,170,-237,170,170,170,170,170,-87,-74,170,-233,-234,-236,170,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,170,-12,170,-287,170,170,170,-340,170,170,170,-222,170,-224,170,-86,-75,170,-232,-235,-340,-201,-340,-68,170,170,170,-340,-28,-287,-223,170,170,170,170,170,170,-11,-287,170,170,-225,-87,-74,-227,-228,170,-340,170,170,170,-226,-229,170,170,-231,-230,]),'FLOAT_CONST':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,171,-340,-27,-28,-185,-338,171,171,171,-340,171,-287,-288,-289,-286,171,171,171,171,-290,-291,171,-340,-28,171,-186,-339,171,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,171,171,171,171,-340,171,-340,-28,-73,-69,171,171,171,171,171,171,171,171,171,171,171,171,171,171,171,171,171,171,171,171,171,171,171,-287,171,171,-340,171,171,-221,-220,171,171,-237,171,171,171,171,171,-87,-74,171,-233,-234,-236,171,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,171,-12,171,-287,171,171,171,-340,171,171,171,-222,171,-224,171,-86,-75,171,-232,-235,-340,-201,-340,-68,171,171,171,-340,-28,-287,-223,171,171,171,171,171,171,-11,-287,171,171,-225,-87,-74,-227,-228,171,-340,171,171,171,-226,-229,171,171,-231,-230,]),'HEX_FLOAT_CONST':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,172,-340,-27,-28,-185,-338,172,172,172,-340,172,-287,-288,-289,-286,172,172,172,172,-290,-291,172,-340,-28,172,-186,-339,172,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,172,172,172,172,-340,172,-340,-28,-73,-69,172,172,172,172,172,172,172,172,172,172,172,172,172,172,172,172,172,172,172,172,172,172,172,-287,172,172,-340,172,172,-221,-220,172,172,-237,172,172,172,172,172,-87,-74,172,-233,-234,-236,172,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,172,-12,172,-287,172,172,172,-340,172,172,172,-222,172,-224,172,-86,-75,172,-232,-235,-340,-201,-340,-68,172,172,172,-340,-28,-287,-223,172,172,172,172,172,172,-11,-287,172,172,-225,-87,-74,-227,-228,172,-340,172,172,172,-226,-229,172,172,-231,-230,]),'CHAR_CONST':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,173,-340,-27,-28,-185,-338,173,173,173,-340,173,-287,-288,-289,-286,173,173,173,173,-290,-291,173,-340,-28,173,-186,-339,173,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,173,173,173,173,-340,173,-340,-28,-73,-69,173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,173,-287,173,173,-340,173,173,-221,-220,173,173,-237,173,173,173,173,173,-87,-74,173,-233,-234,-236,173,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,173,-12,173,-287,173,173,173,-340,173,173,173,-222,173,-224,173,-86,-75,173,-232,-235,-340,-201,-340,-68,173,173,173,-340,-28,-287,-223,173,173,173,173,173,173,-11,-287,173,173,-225,-87,-74,-227,-228,173,-340,173,173,173,-226,-229,173,173,-231,-230,]),'WCHAR_CONST':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,174,-340,-27,-28,-185,-338,174,174,174,-340,174,-287,-288,-289,-286,174,174,174,174,-290,-291,174,-340,-28,174,-186,-339,174,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,174,174,174,174,-340,174,-340,-28,-73,-69,174,174,174,174,174,174,174,174,174,174,174,174,174,174,174,174,174,174,174,174,174,174,174,-287,174,174,-340,174,174,-221,-220,174,174,-237,174,174,174,174,174,-87,-74,174,-233,-234,-236,174,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,174,-12,174,-287,174,174,174,-340,174,174,174,-222,174,-224,174,-86,-75,174,-232,-235,-340,-201,-340,-68,174,174,174,-340,-28,-287,-223,174,174,174,174,174,174,-11,-287,174,174,-225,-87,-74,-227,-228,174,-340,174,174,174,-226,-229,174,174,-231,-230,]),'U8CHAR_CONST':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,175,-340,-27,-28,-185,-338,175,175,175,-340,175,-287,-288,-289,-286,175,175,175,175,-290,-291,175,-340,-28,175,-186,-339,175,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,175,175,175,175,-340,175,-340,-28,-73,-69,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,175,-287,175,175,-340,175,175,-221,-220,175,175,-237,175,175,175,175,175,-87,-74,175,-233,-234,-236,175,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,175,-12,175,-287,175,175,175,-340,175,175,175,-222,175,-224,175,-86,-75,175,-232,-235,-340,-201,-340,-68,175,175,175,-340,-28,-287,-223,175,175,175,175,175,175,-11,-287,175,175,-225,-87,-74,-227,-228,175,-340,175,175,175,-226,-229,175,175,-231,-230,]),'U16CHAR_CONST':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,176,-340,-27,-28,-185,-338,176,176,176,-340,176,-287,-288,-289,-286,176,176,176,176,-290,-291,176,-340,-28,176,-186,-339,176,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,176,176,176,176,-340,176,-340,-28,-73,-69,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,176,-287,176,176,-340,176,176,-221,-220,176,176,-237,176,176,176,176,176,-87,-74,176,-233,-234,-236,176,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,176,-12,176,-287,176,176,176,-340,176,176,176,-222,176,-224,176,-86,-75,176,-232,-235,-340,-201,-340,-68,176,176,176,-340,-28,-287,-223,176,176,176,176,176,176,-11,-287,176,176,-225,-87,-74,-227,-228,176,-340,176,176,176,-226,-229,176,176,-231,-230,]),'U32CHAR_CONST':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,177,-340,-27,-28,-185,-338,177,177,177,-340,177,-287,-288,-289,-286,177,177,177,177,-290,-291,177,-340,-28,177,-186,-339,177,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,177,177,177,177,-340,177,-340,-28,-73,-69,177,177,177,177,177,177,177,177,177,177,177,177,177,177,177,177,177,177,177,177,177,177,177,-287,177,177,-340,177,177,-221,-220,177,177,-237,177,177,177,177,177,-87,-74,177,-233,-234,-236,177,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,177,-12,177,-287,177,177,177,-340,177,177,177,-222,177,-224,177,-86,-75,177,-232,-235,-340,-201,-340,-68,177,177,177,-340,-28,-287,-223,177,177,177,177,177,177,-11,-287,177,177,-225,-87,-74,-227,-228,177,-340,177,177,177,-226,-229,177,177,-231,-230,]),'STRING_LITERAL':([15,51,52,53,81,90,91,92,94,95,114,115,116,121,126,128,135,136,138,139,141,143,147,148,149,150,153,154,155,156,160,161,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,263,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,407,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,139,139,-340,-27,-28,-185,-338,139,139,139,-340,263,-328,139,263,-287,-288,-289,-286,139,139,139,139,-290,-291,139,-340,-28,139,-186,-339,139,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,139,139,139,139,-340,139,-340,-28,-73,-329,139,-69,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,139,-287,139,139,-340,139,139,-221,-220,139,139,-237,139,139,139,139,139,-87,-74,139,-233,-234,-236,139,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,139,-12,139,-287,139,139,139,263,-340,139,139,139,-222,139,-224,139,-86,-75,139,-232,-235,-340,-201,-340,-68,139,139,139,-340,-28,-287,-223,139,139,139,139,139,139,-11,-287,139,139,-225,-87,-74,-227,-228,139,-340,139,139,139,-226,-229,139,139,-231,-230,]),'WSTRING_LITERAL':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,164,178,179,180,181,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,300,301,302,303,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,178,-340,-27,-28,-185,-338,178,178,178,-340,178,-287,-288,-289,-286,178,178,178,178,-290,-291,300,-330,-331,-332,-333,178,-340,-28,178,-186,-339,178,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,178,178,178,178,-340,178,-340,-28,-73,-69,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,178,-334,-335,-336,-337,-287,178,178,-340,178,178,-221,-220,178,178,-237,178,178,178,178,178,-87,-74,178,-233,-234,-236,178,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,178,-12,178,-287,178,178,178,-340,178,178,178,-222,178,-224,178,-86,-75,178,-232,-235,-340,-201,-340,-68,178,178,178,-340,-28,-287,-223,178,178,178,178,178,178,-11,-287,178,178,-225,-87,-74,-227,-228,178,-340,178,178,178,-226,-229,178,178,-231,-230,]),'U8STRING_LITERAL':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,164,178,179,180,181,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,300,301,302,303,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,179,-340,-27,-28,-185,-338,179,179,179,-340,179,-287,-288,-289,-286,179,179,179,179,-290,-291,301,-330,-331,-332,-333,179,-340,-28,179,-186,-339,179,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,179,179,179,179,-340,179,-340,-28,-73,-69,179,179,179,179,179,179,179,179,179,179,179,179,179,179,179,179,179,179,179,179,179,179,179,-334,-335,-336,-337,-287,179,179,-340,179,179,-221,-220,179,179,-237,179,179,179,179,179,-87,-74,179,-233,-234,-236,179,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,179,-12,179,-287,179,179,179,-340,179,179,179,-222,179,-224,179,-86,-75,179,-232,-235,-340,-201,-340,-68,179,179,179,-340,-28,-287,-223,179,179,179,179,179,179,-11,-287,179,179,-225,-87,-74,-227,-228,179,-340,179,179,179,-226,-229,179,179,-231,-230,]),'U16STRING_LITERAL':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,164,178,179,180,181,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,300,301,302,303,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,180,-340,-27,-28,-185,-338,180,180,180,-340,180,-287,-288,-289,-286,180,180,180,180,-290,-291,302,-330,-331,-332,-333,180,-340,-28,180,-186,-339,180,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,180,180,180,180,-340,180,-340,-28,-73,-69,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,180,-334,-335,-336,-337,-287,180,180,-340,180,180,-221,-220,180,180,-237,180,180,180,180,180,-87,-74,180,-233,-234,-236,180,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,180,-12,180,-287,180,180,180,-340,180,180,180,-222,180,-224,180,-86,-75,180,-232,-235,-340,-201,-340,-68,180,180,180,-340,-28,-287,-223,180,180,180,180,180,180,-11,-287,180,180,-225,-87,-74,-227,-228,180,-340,180,180,180,-226,-229,180,180,-231,-230,]),'U32STRING_LITERAL':([15,51,52,53,81,90,91,94,95,114,115,116,121,126,128,135,136,141,147,148,149,150,153,154,155,156,160,161,164,178,179,180,181,182,183,184,195,197,208,221,222,223,224,225,226,227,228,229,230,231,232,234,238,242,247,256,257,258,259,262,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,300,301,302,303,306,309,310,323,332,347,355,356,358,360,361,362,365,366,367,369,370,371,372,374,375,377,378,379,380,381,382,383,384,385,386,387,388,389,392,393,394,397,400,401,402,405,448,455,457,467,469,470,471,474,479,480,482,483,484,487,489,498,499,500,503,510,511,512,520,524,525,526,527,528,529,532,533,543,544,545,552,553,554,555,556,559,562,565,570,572,579,580,582,584,585,586,],[-71,-131,-132,-133,-134,-90,-72,181,-340,-27,-28,-185,-338,181,181,181,-340,181,-287,-288,-289,-286,181,181,181,181,-290,-291,303,-330,-331,-332,-333,181,-340,-28,181,-186,-339,181,-219,-217,-218,-78,-79,-80,-81,-82,-83,-84,-85,181,181,181,181,-340,181,-340,-28,-73,-69,181,181,181,181,181,181,181,181,181,181,181,181,181,181,181,181,181,181,181,181,181,181,181,-334,-335,-336,-337,-287,181,181,-340,181,181,-221,-220,181,181,-237,181,181,181,181,181,-87,-74,181,-233,-234,-236,181,-244,-245,-246,-247,-248,-249,-250,-251,-252,-253,-254,-11,181,-12,181,-287,181,181,181,-340,181,181,181,-222,181,-224,181,-86,-75,181,-232,-235,-340,-201,-340,-68,181,181,181,-340,-28,-287,-223,181,181,181,181,181,181,-11,-287,181,181,-225,-87,-74,-227,-228,181,-340,181,181,181,-226,-229,181,181,-231,-230,]),'ELSE':([15,91,208,225,226,227,228,229,230,232,262,267,355,361,370,371,374,375,377,469,471,479,480,483,484,499,524,552,553,554,555,556,579,580,585,586,],[-71,-72,-339,-78,-79,-80,-81,-82,-83,-85,-73,-69,-221,-237,-87,-84,-233,-234,-236,-222,-224,-86,-84,-232,-235,-68,-223,-225,570,-84,-227,-228,-226,-229,-231,-230,]),'PPPRAGMASTR':([15,],[91,]),'EQUALS':([19,28,73,86,87,88,89,97,111,130,132,139,140,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,202,208,233,250,252,263,291,292,293,295,296,297,300,301,302,303,311,312,395,396,403,404,406,429,431,432,433,434,439,440,490,492,493,494,497,501,502,505,506,508,509,534,535,536,561,563,574,],[-52,-29,-181,135,-182,-54,-37,-53,195,-181,-55,-328,-30,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,332,-339,-315,379,-38,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-35,-36,489,-202,-43,-44,-308,-295,-296,-297,-298,-299,-31,-34,-203,-205,-39,-42,-278,-293,-294,-284,-285,-32,-33,-204,-40,-41,-302,-309,-303,]),'COMMA':([19,24,25,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,60,73,74,75,76,77,78,81,84,85,86,87,88,89,97,104,106,108,110,111,113,114,115,116,118,119,122,123,130,132,139,140,142,143,144,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,187,189,190,191,192,196,197,200,201,202,206,208,212,214,216,233,239,248,249,250,252,253,254,255,263,265,291,292,293,295,296,297,300,301,302,303,311,312,315,316,317,318,319,320,321,324,325,326,327,328,329,330,331,334,336,337,340,341,342,344,345,346,348,349,350,352,353,354,376,391,403,404,406,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,428,429,430,431,432,433,434,438,439,440,444,445,446,447,459,460,461,462,463,464,468,472,473,475,476,477,485,486,488,493,494,497,501,502,505,506,508,509,515,516,518,522,523,531,535,536,537,538,539,546,547,548,549,550,551,557,560,561,563,566,567,574,576,577,578,],[-52,-128,-102,-29,-107,-340,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-181,-98,-99,-100,-101,-104,-134,134,-135,-137,-182,-54,-37,-53,-103,-129,194,-139,-141,-183,-27,-28,-185,-169,-170,-149,-150,-181,-55,-328,-30,266,-306,-255,-256,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,313,314,-189,-194,-340,-184,-186,331,-174,-179,-152,-339,-145,-147,-340,-315,365,-238,-242,-277,-38,-136,-138,-196,-329,365,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-35,-36,-191,-192,-193,-207,-56,-1,-2,-45,-209,-140,-142,331,331,-171,-175,-154,-156,-151,-143,-144,-148,466,-164,-166,-146,-130,-206,-207,-177,-178,365,487,-43,-44,-308,365,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,-275,-276,365,503,-295,-313,-296,-297,-298,-299,507,-31,-34,-190,-195,-57,-208,-172,-173,-176,-180,-153,-155,-168,365,-240,-239,365,365,-243,-197,-199,-39,-42,-278,-293,-294,-284,-285,-32,-33,-210,-216,-214,-165,-167,-198,-40,-41,562,-257,-314,-50,-51,-212,-211,-213,-215,365,-200,-302,-309,-46,-49,-303,365,-47,-48,]),'RPAREN':([19,24,25,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,60,74,75,76,77,78,81,88,89,93,96,97,104,106,113,114,115,116,118,119,122,123,132,133,137,138,139,140,142,143,144,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,185,186,187,188,189,190,191,192,196,197,206,208,212,214,215,216,217,218,239,248,249,250,252,260,261,263,264,265,288,291,292,293,295,296,297,300,301,302,303,311,312,315,316,317,318,319,320,321,322,324,325,330,334,336,337,340,341,342,348,349,350,351,352,353,354,355,357,363,364,403,404,406,407,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,428,429,430,431,432,433,434,435,436,437,439,440,443,444,445,446,447,449,450,451,452,453,454,458,459,460,463,464,472,473,475,476,477,485,493,494,497,501,502,505,506,508,509,513,514,515,516,518,521,535,536,538,539,540,541,546,547,548,549,550,551,557,559,561,563,566,567,572,573,574,575,577,578,581,583,],[-52,-128,-102,-29,-107,-340,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-98,-99,-100,-101,-104,-134,-54,-37,140,-340,-53,-103,-129,-183,-27,-28,-185,-169,-170,-149,-150,-55,252,-340,262,-328,-30,267,-306,-255,-256,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,311,312,-187,-17,-18,-189,-194,-340,-184,-186,-152,-339,-145,-147,349,-340,353,354,-14,-238,-242,-277,-38,403,404,-329,405,406,429,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-35,-36,-191,-192,-193,-207,-56,-1,-2,-340,-45,-209,-171,-154,-156,-151,-143,-144,-148,-146,-130,-206,-340,-207,-177,-178,-221,-13,473,474,-43,-44,-308,499,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,-275,-276,502,-295,-313,-296,-297,-298,-299,504,505,506,-31,-34,-188,-190,-195,-57,-208,-340,515,516,-207,-23,-24,-340,-172,-173,-153,-155,525,-240,-239,526,527,-243,-39,-42,-278,-293,-294,-284,-285,-32,-33,546,547,-210,-216,-214,551,-40,-41,-257,-314,563,-310,-50,-51,-212,-211,-213,-215,571,-340,-302,-309,-46,-49,-340,582,-303,-311,-47,-48,584,-312,]),'COLON':([19,24,28,31,32,33,35,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,81,87,88,89,97,106,118,119,122,123,130,132,139,140,143,144,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,206,208,209,212,214,233,235,248,249,250,252,263,291,292,293,295,296,297,300,301,302,303,311,312,330,334,336,337,340,341,342,346,348,349,353,354,359,403,404,406,408,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,439,440,459,460,463,464,466,473,475,485,493,494,497,501,502,505,506,508,509,535,536,538,561,563,574,],[-52,-128,-29,-125,-126,-127,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-131,-132,-133,-134,-182,-54,-37,-53,-129,-169,-170,-149,-150,-181,-55,-328,-30,-306,-255,-256,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-152,-339,347,-145,-147,358,360,-238,-242,-277,-38,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-35,-36,-171,-154,-156,-151,-143,-144,-148,467,-146,-130,-177,-178,470,-43,-44,-308,500,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,-275,-276,-295,-296,-297,-298,-299,-31,-34,-172,-173,-153,-155,347,-240,-239,-243,-39,-42,-278,-293,-294,-284,-285,-32,-33,-40,-41,-257,-302,-309,-303,]),'LBRACKET':([19,24,25,28,29,30,31,32,33,34,35,38,39,40,41,42,43,44,45,46,47,48,49,51,52,53,54,55,56,57,58,59,60,74,75,76,77,78,81,88,89,97,104,106,113,114,115,116,118,119,121,122,123,132,139,140,143,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,192,196,197,206,208,212,214,216,233,252,256,263,291,292,300,301,302,303,311,312,318,319,322,324,325,330,334,336,337,340,341,342,348,349,351,352,353,354,395,396,403,404,406,429,431,432,433,434,439,440,446,447,452,459,460,463,464,487,490,492,493,494,498,501,502,508,509,515,516,518,534,535,536,540,541,546,547,548,549,550,551,561,562,563,566,567,574,575,577,578,583,],[95,-128,-102,-29,-107,-340,-125,-126,-127,-129,-241,-113,-114,-115,-116,-117,-118,-119,-120,-121,-122,-123,-124,-131,-132,-133,-105,-106,-108,-109,-110,-111,-112,-98,-99,-100,-101,-104,-134,136,-37,95,-103,-129,-183,-27,-28,-185,-169,-170,-338,-149,-150,136,-328,-30,-306,287,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,323,-184,-186,-152,-339,-145,-147,323,-315,-38,397,-329,-300,-301,-334,-335,-336,-337,-35,-36,323,448,323,-45,457,-171,-154,-156,-151,-143,-144,-148,-146,-130,323,323,-177,-178,397,-202,-43,-44,-308,-295,-296,-297,-298,-299,-31,-34,448,457,323,-172,-173,-153,-155,397,-203,-205,-39,-42,397,-293,-294,-32,-33,-210,-216,-214,-204,-40,-41,565,-310,-50,-51,-212,-211,-213,-215,-302,397,-309,-46,-49,-303,-311,-47,-48,-312,]),'RBRACKET':([51,52,53,81,95,114,115,116,136,139,143,144,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,184,197,208,248,249,250,257,259,263,291,292,293,295,296,297,300,301,302,303,305,306,307,308,323,399,400,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,427,429,431,432,433,434,441,442,448,455,456,457,473,475,485,491,495,496,497,501,502,505,506,510,512,517,519,520,538,542,543,561,563,568,569,574,576,],[-131,-132,-133,-134,-340,-27,-28,-185,-340,-328,-306,-255,-256,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-340,-28,-186,-339,-238,-242,-277,-340,-28,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,439,440,-3,-4,-340,493,494,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,-275,-276,501,-295,-296,-297,-298,-299,508,509,-340,-340,518,-340,-240,-239,-243,534,535,536,-278,-293,-294,-284,-285,-340,-28,548,549,550,-257,566,567,-302,-309,577,578,-303,583,]),'PERIOD':([121,139,143,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,256,263,291,292,300,301,302,303,395,396,406,429,431,432,433,434,487,490,492,498,501,502,534,540,541,561,562,563,574,575,583,],[-338,-328,-306,289,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,398,-329,-300,-301,-334,-335,-336,-337,398,-202,-308,-295,-296,-297,-298,-299,398,-203,-205,398,-293,-294,-204,564,-310,-302,398,-309,-303,-311,-312,]),'ARROW':([139,143,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,263,291,292,300,301,302,303,406,429,431,432,433,434,501,502,561,563,574,],[-328,-306,290,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-329,-300,-301,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-293,-294,-302,-309,-303,]),'CONDOP':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,268,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,-275,-276,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'DIVIDE':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,270,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,270,270,270,270,270,270,270,270,270,270,270,270,270,270,270,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'MOD':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,271,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,271,271,271,271,271,271,271,271,271,271,271,271,271,271,271,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'RSHIFT':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,274,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,274,274,274,274,274,274,274,274,274,274,274,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'LSHIFT':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,275,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,275,275,275,275,275,275,275,275,275,275,275,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'LT':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,276,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,276,276,276,276,276,276,276,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'LE':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,277,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,277,277,277,277,277,277,277,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'GE':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,278,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,278,278,278,278,278,278,278,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'GT':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,279,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,279,279,279,279,279,279,279,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'EQ':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,280,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,280,280,280,280,280,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'NE':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,281,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,281,281,281,281,281,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'OR':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,283,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,283,283,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'XOR':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,284,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,284,-274,284,284,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'LAND':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,285,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,-275,285,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'LOR':([139,143,145,146,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,286,-258,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,-277,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-259,-260,-261,-262,-263,-264,-265,-266,-267,-268,-269,-270,-271,-272,-273,-274,-275,-276,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'XOREQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,380,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'TIMESEQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,381,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'DIVEQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,382,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'MODEQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,383,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'PLUSEQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,384,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'MINUSEQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,385,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'LSHIFTEQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,386,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'RSHIFTEQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,387,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'ANDEQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,388,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'OREQUAL':([139,143,151,152,158,159,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,208,233,250,263,291,292,293,295,296,297,300,301,302,303,406,429,431,432,433,434,497,501,502,505,506,561,563,574,],[-328,-306,-277,-279,-292,-315,-304,-305,-307,-316,-317,-318,-319,-320,-321,-322,-323,-324,-325,-326,-327,-330,-331,-332,-333,-339,-315,389,-329,-300,-301,-280,-281,-282,-283,-334,-335,-336,-337,-308,-295,-296,-297,-298,-299,-278,-293,-294,-284,-285,-302,-309,-303,]),'ELLIPSIS':([313,],[443,]),}

_lr_action = {}
for _k, _v in _lr_action_items.items():
   for _x,_y in zip(_v[0],_v[1]):
      if not _x in _lr_action:  _lr_action[_x] = {}
      _lr_action[_x][_k] = _y
del _lr_action_items

_lr_goto_items = {'translation_unit_or_empty':([0,],[1,]),'translation_unit':([0,],[2,]),'empty':([0,11,12,21,22,23,26,27,30,34,69,70,71,73,95,96,101,128,136,137,182,183,192,209,216,221,242,256,257,258,322,323,351,358,360,369,372,448,449,455,457,458,470,482,487,498,510,511,525,526,527,529,559,562,570,572,582,584,],[3,66,83,99,99,99,107,99,114,99,83,107,99,66,114,188,99,220,114,188,307,114,320,343,320,357,357,392,307,114,453,114,453,357,357,357,357,114,188,307,307,453,357,357,533,533,307,114,357,357,357,357,357,533,357,357,357,357,]),'external_declaration':([0,2,],[4,64,]),'function_definition':([0,2,],[5,5,]),'declaration':([0,2,11,67,73,128,221,372,],[6,6,68,129,68,223,223,482,]),'pp_directive':([0,2,],[7,7,]),'pppragma_directive':([0,2,124,128,203,204,205,221,242,333,335,358,360,369,470,525,526,527,570,582,584,],[8,8,211,231,211,211,211,231,371,211,211,371,371,480,371,554,371,371,371,371,371,]),'static_assert':([0,2,128,221,242,358,360,369,470,525,526,527,570,582,584,],[10,10,232,232,232,232,232,232,232,232,232,232,232,232,232,]),'id_declarator':([0,2,12,17,26,69,70,82,134,192,194,209,322,466,],[11,11,73,93,111,130,111,93,130,315,130,130,93,130,]),'declaration_specifiers':([0,2,11,67,73,96,128,137,221,313,322,351,372,449,458,],[12,12,69,69,69,192,69,192,69,192,192,192,69,192,192,]),'decl_body':([0,2,11,67,73,128,221,372,],[13,13,13,13,13,13,13,13,]),'direct_id_declarator':([0,2,12,17,20,26,69,70,80,82,134,192,194,209,318,322,452,466,],[19,19,19,19,97,19,19,19,97,19,19,19,19,19,97,19,97,19,]),'pointer':([0,2,12,17,26,69,70,82,113,134,192,194,209,216,322,351,466,],[20,20,80,20,20,80,20,80,196,80,318,80,80,352,452,352,80,]),'type_qualifier':([0,2,11,12,21,22,23,27,30,34,67,69,71,73,95,96,101,115,124,125,126,128,136,137,141,183,184,192,203,204,205,209,213,216,221,238,258,259,294,298,299,304,313,322,323,333,335,351,372,448,449,458,511,512,],[21,21,21,74,21,21,21,21,116,21,21,74,21,21,116,21,21,197,116,116,116,21,116,21,116,116,197,74,116,116,116,341,197,341,21,116,116,197,116,116,116,116,21,21,116,116,116,21,21,116,21,21,116,197,]),'storage_class_specifier':([0,2,11,12,21,22,23,27,34,67,69,71,73,96,101,128,137,192,221,313,322,351,372,449,458,],[22,22,22,75,22,22,22,22,22,22,75,22,22,22,22,22,22,75,22,22,22,22,22,22,22,]),'function_specifier':([0,2,11,12,21,22,23,27,34,67,69,71,73,96,101,128,137,192,221,313,322,351,372,449,458,],[23,23,23,76,23,23,23,23,23,23,76,23,23,23,23,23,23,76,23,23,23,23,23,23,23,]),'type_specifier_no_typeid':([0,2,11,12,26,67,69,70,73,96,124,125,126,128,137,141,192,193,203,204,205,209,213,216,221,238,294,298,299,304,313,322,333,335,351,372,449,458,],[24,24,24,77,24,24,77,24,24,24,24,24,24,24,24,24,77,24,24,24,24,340,24,340,24,24,24,24,24,24,24,24,24,24,24,24,24,24,]),'type_specifier':([0,2,11,26,67,70,73,96,124,125,126,128,137,141,193,203,204,205,213,221,238,294,298,299,304,313,322,333,335,351,372,449,458,],[25,25,25,104,25,104,25,25,212,212,212,25,25,212,104,212,212,212,348,25,212,212,212,212,212,25,25,212,212,25,25,25,25,]),'declaration_specifiers_no_type':([0,2,11,21,22,23,27,34,67,71,73,96,101,128,137,221,313,322,351,372,449,458,],[26,26,70,100,100,100,100,100,70,100,70,193,100,70,193,70,193,193,193,70,193,193,]),'alignment_specifier':([0,2,11,12,21,22,23,27,34,67,69,71,73,96,101,124,125,126,128,137,141,192,203,204,205,209,216,221,238,294,298,299,304,313,322,333,335,351,372,449,458,],[27,27,27,78,27,27,27,27,27,27,78,27,27,27,27,214,214,214,27,27,214,78,214,214,214,342,342,27,214,214,214,214,214,27,27,214,214,27,27,27,27,]),'typedef_name':([0,2,11,26,67,70,73,96,124,125,126,128,137,141,193,203,204,205,213,221,238,294,298,299,304,313,322,333,335,351,372,449,458,],[31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,31,]),'enum_specifier':([0,2,11,26,67,70,73,96,124,125,126,128,137,141,193,203,204,205,213,221,238,294,298,299,304,313,322,333,335,351,372,449,458,],[32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,]),'struct_or_union_specifier':([0,2,11,26,67,70,73,96,124,125,126,128,137,141,193,203,204,205,213,221,238,294,298,299,304,313,322,333,335,351,372,449,458,],[33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,33,]),'atomic_specifier':([0,2,11,21,22,23,26,27,34,67,70,71,73,96,101,124,125,126,128,137,141,193,203,204,205,213,221,238,294,298,299,304,313,322,333,335,351,372,449,458,],[34,34,71,101,101,101,106,101,101,71,106,101,71,34,101,106,106,106,71,34,106,106,106,106,106,106,71,106,106,106,106,106,34,34,106,106,34,71,34,34,]),'struct_or_union':([0,2,11,26,67,70,73,96,124,125,126,128,137,141,193,203,204,205,213,221,238,294,298,299,304,313,322,333,335,351,372,449,458,],[37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,37,]),'declaration_list_opt':([11,73,],[65,131,]),'declaration_list':([11,73,],[67,67,]),'init_declarator_list_opt':([12,69,],[79,79,]),'init_declarator_list':([12,69,],[84,84,]),'init_declarator':([12,69,134,194,],[85,85,253,326,]),'declarator':([12,69,134,194,209,466,],[86,86,86,86,346,346,]),'typeid_declarator':([12,69,82,134,194,209,466,],[87,87,133,87,87,87,87,]),'direct_typeid_declarator':([12,69,80,82,134,194,209,466,],[88,88,132,88,88,88,88,88,]),'declaration_specifiers_no_type_opt':([21,22,23,27,34,71,101,],[98,102,103,112,117,117,117,]),'id_init_declarator_list_opt':([26,70,],[105,105,]),'id_init_declarator_list':([26,70,],[108,108,]),'id_init_declarator':([26,70,],[110,110,]),'type_qualifier_list_opt':([30,95,136,183,258,323,448,511,],[113,182,257,309,401,455,510,544,]),'type_qualifier_list':([30,95,124,125,126,136,141,183,203,204,205,238,258,294,298,299,304,323,333,335,448,511,],[115,184,213,213,213,259,213,115,213,213,213,213,115,213,213,213,213,115,213,213,512,115,]),'brace_open':([36,37,65,118,119,122,123,128,131,135,195,221,238,242,358,360,369,393,405,470,474,504,505,525,526,527,532,570,582,584,],[120,124,128,198,199,203,204,128,128,256,256,128,128,128,128,128,128,256,498,128,498,498,498,128,128,128,256,128,128,128,]),'compound_statement':([65,128,131,221,238,242,358,360,369,470,525,526,527,570,582,584,],[127,227,251,227,363,227,227,227,227,227,227,227,227,227,227,227,]),'unified_string_literal':([92,94,126,128,135,141,153,154,155,156,182,195,221,234,238,242,247,257,266,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,405,455,457,467,470,474,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[138,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,407,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,143,]),'constant_expression':([94,126,234,332,347,397,467,],[142,218,359,462,468,491,523,]),'conditional_expression':([94,126,128,135,141,182,195,221,234,238,242,247,257,268,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,455,457,467,470,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[144,144,249,249,249,249,249,249,144,249,249,249,249,249,249,249,249,249,249,249,144,144,249,249,249,249,249,249,249,249,249,249,144,249,249,249,249,144,249,249,538,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,249,]),'binary_expression':([94,126,128,135,141,182,195,221,234,238,242,247,257,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,455,457,467,470,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[145,145,145,145,145,145,145,145,145,145,145,145,145,145,409,410,411,412,413,414,415,416,417,418,419,420,421,422,423,424,425,426,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,145,]),'cast_expression':([94,126,128,135,141,155,182,195,221,234,238,242,247,257,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,405,455,457,467,470,474,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[146,146,146,146,146,296,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,497,146,146,146,146,497,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,146,]),'unary_expression':([94,126,128,135,141,153,154,155,156,182,195,221,234,238,242,247,257,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,405,455,457,467,470,474,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[151,151,250,250,250,293,295,151,297,250,250,250,151,250,250,250,250,250,151,151,151,151,151,151,151,151,151,151,151,151,151,151,151,151,151,151,250,250,250,250,250,250,151,151,250,250,250,250,250,250,250,250,250,250,151,250,250,151,250,250,151,250,151,250,151,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,]),'postfix_expression':([94,126,128,135,141,153,154,155,156,182,195,221,234,238,242,247,257,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,405,455,457,467,470,474,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,152,]),'unary_operator':([94,126,128,135,141,153,154,155,156,182,195,221,234,238,242,247,257,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,405,455,457,467,470,474,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,155,]),'primary_expression':([94,126,128,135,141,153,154,155,156,182,195,221,234,238,242,247,257,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,405,455,457,467,470,474,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,158,]),'identifier':([94,96,126,128,135,137,141,153,154,155,156,182,195,221,234,238,242,247,257,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,314,332,347,358,360,362,365,366,367,369,372,378,393,397,398,401,402,405,449,455,457,467,470,474,482,500,503,507,510,525,526,527,528,529,532,544,545,559,564,565,570,572,582,584,],[162,191,162,162,162,191,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,162,445,162,162,162,162,162,162,162,162,162,162,162,162,162,492,162,162,162,191,162,162,162,162,162,162,162,162,541,162,162,162,162,162,162,162,162,162,162,575,162,162,162,162,162,]),'constant':([94,126,128,135,141,153,154,155,156,182,195,221,234,238,242,247,257,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,405,455,457,467,470,474,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,163,]),'unified_wstring_literal':([94,126,128,135,141,153,154,155,156,182,195,221,234,238,242,247,257,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,294,298,309,310,332,347,358,360,362,365,366,367,369,372,378,393,397,401,402,405,455,457,467,470,474,482,500,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,164,]),'parameter_type_list':([96,137,322,351,449,458,],[185,260,454,454,513,454,]),'identifier_list_opt':([96,137,449,],[186,261,514,]),'parameter_list':([96,137,322,351,449,458,],[187,187,187,187,187,187,]),'identifier_list':([96,137,449,],[189,189,189,]),'parameter_declaration':([96,137,313,322,351,449,458,],[190,190,444,190,190,190,190,]),'enumerator_list':([120,198,199,],[200,328,329,]),'enumerator':([120,198,199,331,],[201,201,201,461,]),'struct_declaration_list':([124,203,204,],[205,333,335,]),'brace_close':([124,200,203,204,205,219,328,329,333,335,390,487,537,562,],[206,330,334,336,337,355,459,460,463,464,486,531,561,574,]),'struct_declaration':([124,203,204,205,333,335,],[207,207,207,338,338,338,]),'specifier_qualifier_list':([124,125,126,141,203,204,205,238,294,298,299,304,333,335,],[209,216,216,216,209,209,209,216,216,216,216,216,209,209,]),'type_name':([125,126,141,238,294,298,299,304,],[215,217,264,364,435,436,437,438,]),'block_item_list_opt':([128,],[219,]),'block_item_list':([128,],[221,]),'block_item':([128,221,],[222,356,]),'statement':([128,221,242,358,360,369,470,525,526,527,570,582,584,],[224,224,370,370,370,479,370,553,370,370,370,370,370,]),'labeled_statement':([128,221,242,358,360,369,470,525,526,527,570,582,584,],[225,225,225,225,225,225,225,225,225,225,225,225,225,]),'expression_statement':([128,221,242,358,360,369,470,525,526,527,570,582,584,],[226,226,226,226,226,226,226,226,226,226,226,226,226,]),'selection_statement':([128,221,242,358,360,369,470,525,526,527,570,582,584,],[228,228,228,228,228,228,228,228,228,228,228,228,228,]),'iteration_statement':([128,221,242,358,360,369,470,525,526,527,570,582,584,],[229,229,229,229,229,229,229,229,229,229,229,229,229,]),'jump_statement':([128,221,242,358,360,369,470,525,526,527,570,582,584,],[230,230,230,230,230,230,230,230,230,230,230,230,230,]),'expression_opt':([128,221,242,358,360,369,372,470,482,525,526,527,529,559,570,572,582,584,],[236,236,236,236,236,236,481,236,530,236,236,236,558,573,236,581,236,236,]),'expression':([128,141,221,238,242,247,268,287,294,298,358,360,362,366,367,369,372,470,482,525,526,527,528,529,559,565,570,572,582,584,],[239,265,239,265,239,376,408,427,265,265,239,239,472,476,477,239,239,239,239,239,239,239,557,239,239,576,239,239,239,239,]),'assignment_expression':([128,135,141,182,195,221,238,242,247,257,268,287,288,294,298,309,310,358,360,362,365,366,367,369,372,378,393,401,402,455,457,470,482,503,510,525,526,527,528,529,532,544,545,559,565,570,572,582,584,],[248,255,248,308,255,248,248,248,248,308,248,248,430,248,248,441,442,248,248,248,475,248,248,248,248,485,255,495,496,308,308,248,248,539,308,248,248,248,248,248,255,568,569,248,248,248,248,248,248,]),'initializer':([135,195,393,532,],[254,327,488,560,]),'assignment_expression_opt':([182,257,455,457,510,],[305,399,517,519,542,]),'typeid_noparen_declarator':([192,],[316,]),'abstract_declarator_opt':([192,216,],[317,350,]),'direct_typeid_noparen_declarator':([192,318,],[319,446,]),'abstract_declarator':([192,216,322,351,],[321,321,450,450,]),'direct_abstract_declarator':([192,216,318,322,351,352,452,],[325,325,447,325,325,447,447,]),'struct_declarator_list_opt':([209,],[339,]),'struct_declarator_list':([209,],[344,]),'struct_declarator':([209,466,],[345,522,]),'pragmacomp_or_statement':([242,358,360,470,525,526,527,570,582,584,],[368,469,471,524,552,555,556,579,585,586,]),'pppragma_directive_list':([242,358,360,470,525,526,527,570,582,584,],[369,369,369,369,369,369,369,369,369,369,]),'assignment_operator':([250,],[378,]),'initializer_list_opt':([256,],[390,]),'initializer_list':([256,498,],[391,537,]),'designation_opt':([256,487,498,562,],[393,532,393,532,]),'designation':([256,487,498,562,],[394,394,394,394,]),'designator_list':([256,487,498,562,],[395,395,395,395,]),'designator':([256,395,487,498,562,],[396,490,396,396,396,]),'argument_expression_list':([288,],[428,]),'parameter_type_list_opt':([322,351,458,],[451,451,521,]),'offsetof_member_designator':([507,],[540,]),}

_lr_goto = {}
for _k, _v in _lr_goto_items.items():
   for _x, _y in zip(_v[0], _v[1]):
       if not _x in _lr_goto: _lr_goto[_x] = {}
       _lr_goto[_x][_k] = _y
del _lr_goto_items
_lr_productions = [
  ("S' -> translation_unit_or_empty","S'",1,None,None,None),
  ('abstract_declarator_opt -> empty','abstract_declarator_opt',1,'p_abstract_declarator_opt','plyparser.py',43),
  ('abstract_declarator_opt -> abstract_declarator','abstract_declarator_opt',1,'p_abstract_declarator_opt','plyparser.py',44),
  ('assignment_expression_opt -> empty','assignment_expression_opt',1,'p_assignment_expression_opt','plyparser.py',43),
  ('assignment_expression_opt -> assignment_expression','assignment_expression_opt',1,'p_assignment_expression_opt','plyparser.py',44),
  ('block_item_list_opt -> empty','block_item_list_opt',1,'p_block_item_list_opt','plyparser.py',43),
  ('block_item_list_opt -> block_item_list','block_item_list_opt',1,'p_block_item_list_opt','plyparser.py',44),
  ('declaration_list_opt -> empty','declaration_list_opt',1,'p_declaration_list_opt','plyparser.py',43),
  ('declaration_list_opt -> declaration_list','declaration_list_opt',1,'p_declaration_list_opt','plyparser.py',44),
  ('declaration_specifiers_no_type_opt -> empty','declaration_specifiers_no_type_opt',1,'p_declaration_specifiers_no_type_opt','plyparser.py',43),
  ('declaration_specifiers_no_type_opt -> declaration_specifiers_no_type','declaration_specifiers_no_type_opt',1,'p_declaration_specifiers_no_type_opt','plyparser.py',44),
  ('designation_opt -> empty','designation_opt',1,'p_designation_opt','plyparser.py',43),
  ('designation_opt -> designation','designation_opt',1,'p_designation_opt','plyparser.py',44),
  ('expression_opt -> empty','expression_opt',1,'p_expression_opt','plyparser.py',43),
  ('expression_opt -> expression','expression_opt',1,'p_expression_opt','plyparser.py',44),
  ('id_init_declarator_list_opt -> empty','id_init_declarator_list_opt',1,'p_id_init_declarator_list_opt','plyparser.py',43),
  ('id_init_declarator_list_opt -> id_init_declarator_list','id_init_declarator_list_opt',1,'p_id_init_declarator_list_opt','plyparser.py',44),
  ('identifier_list_opt -> empty','identifier_list_opt',1,'p_identifier_list_opt','plyparser.py',43),
  ('identifier_list_opt -> identifier_list','identifier_list_opt',1,'p_identifier_list_opt','plyparser.py',44),
  ('init_declarator_list_opt -> empty','init_declarator_list_opt',1,'p_init_declarator_list_opt','plyparser.py',43),
  ('init_declarator_list_opt -> init_declarator_list','init_declarator_list_opt',1,'p_init_declarator_list_opt','plyparser.py',44),
  ('initializer_list_opt -> empty','initializer_list_opt',1,'p_initializer_list_opt','plyparser.py',43),
  ('initializer_list_opt -> initializer_list','initializer_list_opt',1,'p_initializer_list_opt','plyparser.py',44),
  ('parameter_type_list_opt -> empty','parameter_type_list_opt',1,'p_parameter_type_list_opt','plyparser.py',43),
  ('parameter_type_list_opt -> parameter_type_list','parameter_type_list_opt',1,'p_parameter_type_list_opt','plyparser.py',44),
  ('struct_declarator_list_opt -> empty','struct_declarator_list_opt',1,'p_struct_declarator_list_opt','plyparser.py',43),
  ('struct_declarator_list_opt -> struct_declarator_list','struct_declarator_list_opt',1,'p_struct_declarator_list_opt','plyparser.py',44),
  ('type_qualifier_list_opt -> empty','type_qualifier_list_opt',1,'p_type_qualifier_list_opt','plyparser.py',43),
  ('type_qualifier_list_opt -> type_qualifier_list','type_qualifier_list_opt',1,'p_type_qualifier_list_opt','plyparser.py',44),
  ('direct_id_declarator -> ID','direct_id_declarator',1,'p_direct_id_declarator_1','plyparser.py',126),
  ('direct_id_declarator -> LPAREN id_declarator RPAREN','direct_id_declarator',3,'p_direct_id_declarator_2','plyparser.py',126),
  ('direct_id_declarator -> direct_id_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET','direct_id_declarator',5,'p_direct_id_declarator_3','plyparser.py',126),
  ('direct_id_declarator -> direct_id_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET','direct_id_declarator',6,'p_direct_id_declarator_4','plyparser.py',126),
  ('direct_id_declarator -> direct_id_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET','direct_id_declarator',6,'p_direct_id_declarator_4','plyparser.py',127),
  ('direct_id_declarator -> direct_id_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET','direct_id_declarator',5,'p_direct_id_declarator_5','plyparser.py',126),
  ('direct_id_declarator -> direct_id_declarator LPAREN parameter_type_list RPAREN','direct_id_declarator',4,'p_direct_id_declarator_6','plyparser.py',126),
  ('direct_id_declarator -> direct_id_declarator LPAREN identifier_list_opt RPAREN','direct_id_declarator',4,'p_direct_id_declarator_6','plyparser.py',127),
  ('direct_typeid_declarator -> TYPEID','direct_typeid_declarator',1,'p_direct_typeid_declarator_1','plyparser.py',126),
  ('direct_typeid_declarator -> LPAREN typeid_declarator RPAREN','direct_typeid_declarator',3,'p_direct_typeid_declarator_2','plyparser.py',126),
  ('direct_typeid_declarator -> direct_typeid_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET','direct_typeid_declarator',5,'p_direct_typeid_declarator_3','plyparser.py',126),
  ('direct_typeid_declarator -> direct_typeid_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET','direct_typeid_declarator',6,'p_direct_typeid_declarator_4','plyparser.py',126),
  ('direct_typeid_declarator -> direct_typeid_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET','direct_typeid_declarator',6,'p_direct_typeid_declarator_4','plyparser.py',127),
  ('direct_typeid_declarator -> direct_typeid_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET','direct_typeid_declarator',5,'p_direct_typeid_declarator_5','plyparser.py',126),
  ('direct_typeid_declarator -> direct_typeid_declarator LPAREN parameter_type_list RPAREN','direct_typeid_declarator',4,'p_direct_typeid_declarator_6','plyparser.py',126),
  ('direct_typeid_declarator -> direct_typeid_declarator LPAREN identifier_list_opt RPAREN','direct_typeid_declarator',4,'p_direct_typeid_declarator_6','plyparser.py',127),
  ('direct_typeid_noparen_declarator -> TYPEID','direct_typeid_noparen_declarator',1,'p_direct_typeid_noparen_declarator_1','plyparser.py',126),
  ('direct_typeid_noparen_declarator -> direct_typeid_noparen_declarator LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET','direct_typeid_noparen_declarator',5,'p_direct_typeid_noparen_declarator_3','plyparser.py',126),
  ('direct_typeid_noparen_declarator -> direct_typeid_noparen_declarator LBRACKET STATIC type_qualifier_list_opt assignment_expression RBRACKET','direct_typeid_noparen_declarator',6,'p_direct_typeid_noparen_declarator_4','plyparser.py',126),
  ('direct_typeid_noparen_declarator -> direct_typeid_noparen_declarator LBRACKET type_qualifier_list STATIC assignment_expression RBRACKET','direct_typeid_noparen_declarator',6,'p_direct_typeid_noparen_declarator_4','plyparser.py',127),
  ('direct_typeid_noparen_declarator -> direct_typeid_noparen_declarator LBRACKET type_qualifier_list_opt TIMES RBRACKET','direct_typeid_noparen_declarator',5,'p_direct_typeid_noparen_declarator_5','plyparser.py',126),
  ('direct_typeid_noparen_declarator -> direct_typeid_noparen_declarator LPAREN parameter_type_list RPAREN','direct_typeid_noparen_declarator',4,'p_direct_typeid_noparen_declarator_6','plyparser.py',126),
  ('direct_typeid_noparen_declarator -> direct_typeid_noparen_declarator LPAREN identifier_list_opt RPAREN','direct_typeid_noparen_declarator',4,'p_direct_typeid_noparen_declarator_6','plyparser.py',127),
  ('id_declarator -> direct_id_declarator','id_declarator',1,'p_id_declarator_1','plyparser.py',126),
  ('id_declarator -> pointer direct_id_declarator','id_declarator',2,'p_id_declarator_2','plyparser.py',126),
  ('typeid_declarator -> direct_typeid_declarator','typeid_declarator',1,'p_typeid_declarator_1','plyparser.py',126),
  ('typeid_declarator -> pointer direct_typeid_declarator','typeid_declarator',2,'p_typeid_declarator_2','plyparser.py',126),
  ('typeid_noparen_declarator -> direct_typeid_noparen_declarator','typeid_noparen_declarator',1,'p_typeid_noparen_declarator_1','plyparser.py',126),
  ('typeid_noparen_declarator -> pointer direct_typeid_noparen_declarator','typeid_noparen_declarator',2,'p_typeid_noparen_declarator_2','plyparser.py',126),
  ('translation_unit_or_empty -> translation_unit','translation_unit_or_empty',1,'p_translation_unit_or_empty','c_parser.py',509),
  ('translation_unit_or_empty -> empty','translation_unit_or_empty',1,'p_translation_unit_or_empty','c_parser.py',510),
  ('translation_unit -> external_declaration','translation_unit',1,'p_translation_unit_1','c_parser.py',518),
  ('translation_unit -> translation_unit external_declaration','translation_unit',2,'p_translation_unit_2','c_parser.py',524),
  ('external_declaration -> function_definition','external_declaration',1,'p_external_declaration_1','c_parser.py',534),
  ('external_declaration -> declaration','external_declaration',1,'p_external_declaration_2','c_parser.py',539),
  ('external_declaration -> pp_directive','external_declaration',1,'p_external_declaration_3','c_parser.py',544),
  ('external_declaration -> pppragma_directive','external_declaration',1,'p_external_declaration_3','c_parser.py',545),
  ('external_declaration -> SEMI','external_declaration',1,'p_external_declaration_4','c_parser.py',550),
  ('external_declaration -> static_assert','external_declaration',1,'p_external_declaration_5','c_parser.py',555),
  ('static_assert -> _STATIC_ASSERT LPAREN constant_expression COMMA unified_string_literal RPAREN','static_assert',6,'p_static_assert_declaration','c_parser.py',560),
  ('static_assert -> _STATIC_ASSERT LPAREN constant_expression RPAREN','static_assert',4,'p_static_assert_declaration','c_parser.py',561),
  ('pp_directive -> PPHASH','pp_directive',1,'p_pp_directive','c_parser.py',569),
  ('pppragma_directive -> PPPRAGMA','pppragma_directive',1,'p_pppragma_directive','c_parser.py',580),
  ('pppragma_directive -> PPPRAGMA PPPRAGMASTR','pppragma_directive',2,'p_pppragma_directive','c_parser.py',581),
  ('pppragma_directive -> _PRAGMA LPAREN unified_string_literal RPAREN','pppragma_directive',4,'p_pppragma_directive','c_parser.py',582),
  ('pppragma_directive_list -> pppragma_directive','pppragma_directive_list',1,'p_pppragma_directive_list','c_parser.py',592),
  ('pppragma_directive_list -> pppragma_directive_list pppragma_directive','pppragma_directive_list',2,'p_pppragma_directive_list','c_parser.py',593),
  ('function_definition -> id_declarator declaration_list_opt compound_statement','function_definition',3,'p_function_definition_1','c_parser.py',600),
  ('function_definition -> declaration_specifiers id_declarator declaration_list_opt compound_statement','function_definition',4,'p_function_definition_2','c_parser.py',618),
  ('statement -> labeled_statement','statement',1,'p_statement','c_parser.py',633),
  ('statement -> expression_statement','statement',1,'p_statement','c_parser.py',634),
  ('statement -> compound_statement','statement',1,'p_statement','c_parser.py',635),
  ('statement -> selection_statement','statement',1,'p_statement','c_parser.py',636),
  ('statement -> iteration_statement','statement',1,'p_statement','c_parser.py',637),
  ('statement -> jump_statement','statement',1,'p_statement','c_parser.py',638),
  ('statement -> pppragma_directive','statement',1,'p_statement','c_parser.py',639),
  ('statement -> static_assert','statement',1,'p_statement','c_parser.py',640),
  ('pragmacomp_or_statement -> pppragma_directive_list statement','pragmacomp_or_statement',2,'p_pragmacomp_or_statement','c_parser.py',688),
  ('pragmacomp_or_statement -> statement','pragmacomp_or_statement',1,'p_pragmacomp_or_statement','c_parser.py',689),
  ('decl_body -> declaration_specifiers init_declarator_list_opt','decl_body',2,'p_decl_body','c_parser.py',708),
  ('decl_body -> declaration_specifiers_no_type id_init_declarator_list_opt','decl_body',2,'p_decl_body','c_parser.py',709),
  ('declaration -> decl_body SEMI','declaration',2,'p_declaration','c_parser.py',769),
  ('declaration_list -> declaration','declaration_list',1,'p_declaration_list','c_parser.py',778),
  ('declaration_list -> declaration_list declaration','declaration_list',2,'p_declaration_list','c_parser.py',779),
  ('declaration_specifiers_no_type -> type_qualifier declaration_specifiers_no_type_opt','declaration_specifiers_no_type',2,'p_declaration_specifiers_no_type_1','c_parser.py',789),
  ('declaration_specifiers_no_type -> storage_class_specifier declaration_specifiers_no_type_opt','declaration_specifiers_no_type',2,'p_declaration_specifiers_no_type_2','c_parser.py',794),
  ('declaration_specifiers_no_type -> function_specifier declaration_specifiers_no_type_opt','declaration_specifiers_no_type',2,'p_declaration_specifiers_no_type_3','c_parser.py',799),
  ('declaration_specifiers_no_type -> atomic_specifier declaration_specifiers_no_type_opt','declaration_specifiers_no_type',2,'p_declaration_specifiers_no_type_4','c_parser.py',806),
  ('declaration_specifiers_no_type -> alignment_specifier declaration_specifiers_no_type_opt','declaration_specifiers_no_type',2,'p_declaration_specifiers_no_type_5','c_parser.py',811),
  ('declaration_specifiers -> declaration_specifiers type_qualifier','declaration_specifiers',2,'p_declaration_specifiers_1','c_parser.py',816),
  ('declaration_specifiers -> declaration_specifiers storage_class_specifier','declaration_specifiers',2,'p_declaration_specifiers_2','c_parser.py',821),
  ('declaration_specifiers -> declaration_specifiers function_specifier','declaration_specifiers',2,'p_declaration_specifiers_3','c_parser.py',826),
  ('declaration_specifiers -> declaration_specifiers type_specifier_no_typeid','declaration_specifiers',2,'p_declaration_specifiers_4','c_parser.py',831),
  ('declaration_specifiers -> type_specifier','declaration_specifiers',1,'p_declaration_specifiers_5','c_parser.py',836),
  ('declaration_specifiers -> declaration_specifiers_no_type type_specifier','declaration_specifiers',2,'p_declaration_specifiers_6','c_parser.py',841),
  ('declaration_specifiers -> declaration_specifiers alignment_specifier','declaration_specifiers',2,'p_declaration_specifiers_7','c_parser.py',846),
  ('storage_class_specifier -> AUTO','storage_class_specifier',1,'p_storage_class_specifier','c_parser.py',851),
  ('storage_class_specifier -> REGISTER','storage_class_specifier',1,'p_storage_class_specifier','c_parser.py',852),
  ('storage_class_specifier -> STATIC','storage_class_specifier',1,'p_storage_class_specifier','c_parser.py',853),
  ('storage_class_specifier -> EXTERN','storage_class_specifier',1,'p_storage_class_specifier','c_parser.py',854),
  ('storage_class_specifier -> TYPEDEF','storage_class_specifier',1,'p_storage_class_specifier','c_parser.py',855),
  ('storage_class_specifier -> _THREAD_LOCAL','storage_class_specifier',1,'p_storage_class_specifier','c_parser.py',856),
  ('function_specifier -> INLINE','function_specifier',1,'p_function_specifier','c_parser.py',861),
  ('function_specifier -> _NORETURN','function_specifier',1,'p_function_specifier','c_parser.py',862),
  ('type_specifier_no_typeid -> VOID','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',867),
  ('type_specifier_no_typeid -> _BOOL','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',868),
  ('type_specifier_no_typeid -> CHAR','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',869),
  ('type_specifier_no_typeid -> SHORT','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',870),
  ('type_specifier_no_typeid -> INT','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',871),
  ('type_specifier_no_typeid -> LONG','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',872),
  ('type_specifier_no_typeid -> FLOAT','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',873),
  ('type_specifier_no_typeid -> DOUBLE','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',874),
  ('type_specifier_no_typeid -> _COMPLEX','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',875),
  ('type_specifier_no_typeid -> SIGNED','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',876),
  ('type_specifier_no_typeid -> UNSIGNED','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',877),
  ('type_specifier_no_typeid -> __INT128','type_specifier_no_typeid',1,'p_type_specifier_no_typeid','c_parser.py',878),
  ('type_specifier -> typedef_name','type_specifier',1,'p_type_specifier','c_parser.py',883),
  ('type_specifier -> enum_specifier','type_specifier',1,'p_type_specifier','c_parser.py',884),
  ('type_specifier -> struct_or_union_specifier','type_specifier',1,'p_type_specifier','c_parser.py',885),
  ('type_specifier -> type_specifier_no_typeid','type_specifier',1,'p_type_specifier','c_parser.py',886),
  ('type_specifier -> atomic_specifier','type_specifier',1,'p_type_specifier','c_parser.py',887),
  ('atomic_specifier -> _ATOMIC LPAREN type_name RPAREN','atomic_specifier',4,'p_atomic_specifier','c_parser.py',893),
  ('type_qualifier -> CONST','type_qualifier',1,'p_type_qualifier','c_parser.py',900),
  ('type_qualifier -> RESTRICT','type_qualifier',1,'p_type_qualifier','c_parser.py',901),
  ('type_qualifier -> VOLATILE','type_qualifier',1,'p_type_qualifier','c_parser.py',902),
  ('type_qualifier -> _ATOMIC','type_qualifier',1,'p_type_qualifier','c_parser.py',903),
  ('init_declarator_list -> init_declarator','init_declarator_list',1,'p_init_declarator_list','c_parser.py',908),
  ('init_declarator_list -> init_declarator_list COMMA init_declarator','init_declarator_list',3,'p_init_declarator_list','c_parser.py',909),
  ('init_declarator -> declarator','init_declarator',1,'p_init_declarator','c_parser.py',917),
  ('init_declarator -> declarator EQUALS initializer','init_declarator',3,'p_init_declarator','c_parser.py',918),
  ('id_init_declarator_list -> id_init_declarator','id_init_declarator_list',1,'p_id_init_declarator_list','c_parser.py',923),
  ('id_init_declarator_list -> id_init_declarator_list COMMA init_declarator','id_init_declarator_list',3,'p_id_init_declarator_list','c_parser.py',924),
  ('id_init_declarator -> id_declarator','id_init_declarator',1,'p_id_init_declarator','c_parser.py',929),
  ('id_init_declarator -> id_declarator EQUALS initializer','id_init_declarator',3,'p_id_init_declarator','c_parser.py',930),
  ('specifier_qualifier_list -> specifier_qualifier_list type_specifier_no_typeid','specifier_qualifier_list',2,'p_specifier_qualifier_list_1','c_parser.py',937),
  ('specifier_qualifier_list -> specifier_qualifier_list type_qualifier','specifier_qualifier_list',2,'p_specifier_qualifier_list_2','c_parser.py',942),
  ('specifier_qualifier_list -> type_specifier','specifier_qualifier_list',1,'p_specifier_qualifier_list_3','c_parser.py',947),
  ('specifier_qualifier_list -> type_qualifier_list type_specifier','specifier_qualifier_list',2,'p_specifier_qualifier_list_4','c_parser.py',952),
  ('specifier_qualifier_list -> alignment_specifier','specifier_qualifier_list',1,'p_specifier_qualifier_list_5','c_parser.py',957),
  ('specifier_qualifier_list -> specifier_qualifier_list alignment_specifier','specifier_qualifier_list',2,'p_specifier_qualifier_list_6','c_parser.py',962),
  ('struct_or_union_specifier -> struct_or_union ID','struct_or_union_specifier',2,'p_struct_or_union_specifier_1','c_parser.py',970),
  ('struct_or_union_specifier -> struct_or_union TYPEID','struct_or_union_specifier',2,'p_struct_or_union_specifier_1','c_parser.py',971),
  ('struct_or_union_specifier -> struct_or_union brace_open struct_declaration_list brace_close','struct_or_union_specifier',4,'p_struct_or_union_specifier_2','c_parser.py',981),
  ('struct_or_union_specifier -> struct_or_union brace_open brace_close','struct_or_union_specifier',3,'p_struct_or_union_specifier_2','c_parser.py',982),
  ('struct_or_union_specifier -> struct_or_union ID brace_open struct_declaration_list brace_close','struct_or_union_specifier',5,'p_struct_or_union_specifier_3','c_parser.py',999),
  ('struct_or_union_specifier -> struct_or_union ID brace_open brace_close','struct_or_union_specifier',4,'p_struct_or_union_specifier_3','c_parser.py',1000),
  ('struct_or_union_specifier -> struct_or_union TYPEID brace_open struct_declaration_list brace_close','struct_or_union_specifier',5,'p_struct_or_union_specifier_3','c_parser.py',1001),
  ('struct_or_union_specifier -> struct_or_union TYPEID brace_open brace_close','struct_or_union_specifier',4,'p_struct_or_union_specifier_3','c_parser.py',1002),
  ('struct_or_union -> STRUCT','struct_or_union',1,'p_struct_or_union','c_parser.py',1018),
  ('struct_or_union -> UNION','struct_or_union',1,'p_struct_or_union','c_parser.py',1019),
  ('struct_declaration_list -> struct_declaration','struct_declaration_list',1,'p_struct_declaration_list','c_parser.py',1026),
  ('struct_declaration_list -> struct_declaration_list struct_declaration','struct_declaration_list',2,'p_struct_declaration_list','c_parser.py',1027),
  ('struct_declaration -> specifier_qualifier_list struct_declarator_list_opt SEMI','struct_declaration',3,'p_struct_declaration_1','c_parser.py',1035),
  ('struct_declaration -> SEMI','struct_declaration',1,'p_struct_declaration_2','c_parser.py',1073),
  ('struct_declaration -> pppragma_directive','struct_declaration',1,'p_struct_declaration_3','c_parser.py',1078),
  ('struct_declarator_list -> struct_declarator','struct_declarator_list',1,'p_struct_declarator_list','c_parser.py',1083),
  ('struct_declarator_list -> struct_declarator_list COMMA struct_declarator','struct_declarator_list',3,'p_struct_declarator_list','c_parser.py',1084),
  ('struct_declarator -> declarator','struct_declarator',1,'p_struct_declarator_1','c_parser.py',1092),
  ('struct_declarator -> declarator COLON constant_expression','struct_declarator',3,'p_struct_declarator_2','c_parser.py',1097),
  ('struct_declarator -> COLON constant_expression','struct_declarator',2,'p_struct_declarator_2','c_parser.py',1098),
  ('enum_specifier -> ENUM ID','enum_specifier',2,'p_enum_specifier_1','c_parser.py',1106),
  ('enum_specifier -> ENUM TYPEID','enum_specifier',2,'p_enum_specifier_1','c_parser.py',1107),
  ('enum_specifier -> ENUM brace_open enumerator_list brace_close','enum_specifier',4,'p_enum_specifier_2','c_parser.py',1112),
  ('enum_specifier -> ENUM ID brace_open enumerator_list brace_close','enum_specifier',5,'p_enum_specifier_3','c_parser.py',1117),
  ('enum_specifier -> ENUM TYPEID brace_open enumerator_list brace_close','enum_specifier',5,'p_enum_specifier_3','c_parser.py',1118),
  ('enumerator_list -> enumerator','enumerator_list',1,'p_enumerator_list','c_parser.py',1123),
  ('enumerator_list -> enumerator_list COMMA','enumerator_list',2,'p_enumerator_list','c_parser.py',1124),
  ('enumerator_list -> enumerator_list COMMA enumerator','enumerator_list',3,'p_enumerator_list','c_parser.py',1125),
  ('alignment_specifier -> _ALIGNAS LPAREN type_name RPAREN','alignment_specifier',4,'p_alignment_specifier','c_parser.py',1136),
  ('alignment_specifier -> _ALIGNAS LPAREN constant_expression RPAREN','alignment_specifier',4,'p_alignment_specifier','c_parser.py',1137),
  ('enumerator -> ID','enumerator',1,'p_enumerator','c_parser.py',1142),
  ('enumerator -> ID EQUALS constant_expression','enumerator',3,'p_enumerator','c_parser.py',1143),
  ('declarator -> id_declarator','declarator',1,'p_declarator','c_parser.py',1158),
  ('declarator -> typeid_declarator','declarator',1,'p_declarator','c_parser.py',1159),
  ('pointer -> TIMES type_qualifier_list_opt','pointer',2,'p_pointer','c_parser.py',1271),
  ('pointer -> TIMES type_qualifier_list_opt pointer','pointer',3,'p_pointer','c_parser.py',1272),
  ('type_qualifier_list -> type_qualifier','type_qualifier_list',1,'p_type_qualifier_list','c_parser.py',1301),
  ('type_qualifier_list -> type_qualifier_list type_qualifier','type_qualifier_list',2,'p_type_qualifier_list','c_parser.py',1302),
  ('parameter_type_list -> parameter_list','parameter_type_list',1,'p_parameter_type_list','c_parser.py',1307),
  ('parameter_type_list -> parameter_list COMMA ELLIPSIS','parameter_type_list',3,'p_parameter_type_list','c_parser.py',1308),
  ('parameter_list -> parameter_declaration','parameter_list',1,'p_parameter_list','c_parser.py',1316),
  ('parameter_list -> parameter_list COMMA parameter_declaration','parameter_list',3,'p_parameter_list','c_parser.py',1317),
  ('parameter_declaration -> declaration_specifiers id_declarator','parameter_declaration',2,'p_parameter_declaration_1','c_parser.py',1336),
  ('parameter_declaration -> declaration_specifiers typeid_noparen_declarator','parameter_declaration',2,'p_parameter_declaration_1','c_parser.py',1337),
  ('parameter_declaration -> declaration_specifiers abstract_declarator_opt','parameter_declaration',2,'p_parameter_declaration_2','c_parser.py',1348),
  ('identifier_list -> identifier','identifier_list',1,'p_identifier_list','c_parser.py',1380),
  ('identifier_list -> identifier_list COMMA identifier','identifier_list',3,'p_identifier_list','c_parser.py',1381),
  ('initializer -> assignment_expression','initializer',1,'p_initializer_1','c_parser.py',1390),
  ('initializer -> brace_open initializer_list_opt brace_close','initializer',3,'p_initializer_2','c_parser.py',1395),
  ('initializer -> brace_open initializer_list COMMA brace_close','initializer',4,'p_initializer_2','c_parser.py',1396),
  ('initializer_list -> designation_opt initializer','initializer_list',2,'p_initializer_list','c_parser.py',1404),
  ('initializer_list -> initializer_list COMMA designation_opt initializer','initializer_list',4,'p_initializer_list','c_parser.py',1405),
  ('designation -> designator_list EQUALS','designation',2,'p_designation','c_parser.py',1416),
  ('designator_list -> designator','designator_list',1,'p_designator_list','c_parser.py',1424),
  ('designator_list -> designator_list designator','designator_list',2,'p_designator_list','c_parser.py',1425),
  ('designator -> LBRACKET constant_expression RBRACKET','designator',3,'p_designator','c_parser.py',1430),
  ('designator -> PERIOD identifier','designator',2,'p_designator','c_parser.py',1431),
  ('type_name -> specifier_qualifier_list abstract_declarator_opt','type_name',2,'p_type_name','c_parser.py',1436),
  ('abstract_declarator -> pointer','abstract_declarator',1,'p_abstract_declarator_1','c_parser.py',1448),
  ('abstract_declarator -> pointer direct_abstract_declarator','abstract_declarator',2,'p_abstract_declarator_2','c_parser.py',1456),
  ('abstract_declarator -> direct_abstract_declarator','abstract_declarator',1,'p_abstract_declarator_3','c_parser.py',1461),
  ('direct_abstract_declarator -> LPAREN abstract_declarator RPAREN','direct_abstract_declarator',3,'p_direct_abstract_declarator_1','c_parser.py',1471),
  ('direct_abstract_declarator -> direct_abstract_declarator LBRACKET assignment_expression_opt RBRACKET','direct_abstract_declarator',4,'p_direct_abstract_declarator_2','c_parser.py',1475),
  ('direct_abstract_declarator -> LBRACKET type_qualifier_list_opt assignment_expression_opt RBRACKET','direct_abstract_declarator',4,'p_direct_abstract_declarator_3','c_parser.py',1486),
  ('direct_abstract_declarator -> direct_abstract_declarator LBRACKET TIMES RBRACKET','direct_abstract_declarator',4,'p_direct_abstract_declarator_4','c_parser.py',1496),
  ('direct_abstract_declarator -> LBRACKET TIMES RBRACKET','direct_abstract_declarator',3,'p_direct_abstract_declarator_5','c_parser.py',1507),
  ('direct_abstract_declarator -> direct_abstract_declarator LPAREN parameter_type_list_opt RPAREN','direct_abstract_declarator',4,'p_direct_abstract_declarator_6','c_parser.py',1516),
  ('direct_abstract_declarator -> LPAREN parameter_type_list_opt RPAREN','direct_abstract_declarator',3,'p_direct_abstract_declarator_7','c_parser.py',1526),
  ('block_item -> declaration','block_item',1,'p_block_item','c_parser.py',1537),
  ('block_item -> statement','block_item',1,'p_block_item','c_parser.py',1538),
  ('block_item_list -> block_item','block_item_list',1,'p_block_item_list','c_parser.py',1545),
  ('block_item_list -> block_item_list block_item','block_item_list',2,'p_block_item_list','c_parser.py',1546),
  ('compound_statement -> brace_open block_item_list_opt brace_close','compound_statement',3,'p_compound_statement_1','c_parser.py',1552),
  ('labeled_statement -> ID COLON pragmacomp_or_statement','labeled_statement',3,'p_labeled_statement_1','c_parser.py',1558),
  ('labeled_statement -> CASE constant_expression COLON pragmacomp_or_statement','labeled_statement',4,'p_labeled_statement_2','c_parser.py',1562),
  ('labeled_statement -> DEFAULT COLON pragmacomp_or_statement','labeled_statement',3,'p_labeled_statement_3','c_parser.py',1566),
  ('selection_statement -> IF LPAREN expression RPAREN pragmacomp_or_statement','selection_statement',5,'p_selection_statement_1','c_parser.py',1570),
  ('selection_statement -> IF LPAREN expression RPAREN statement ELSE pragmacomp_or_statement','selection_statement',7,'p_selection_statement_2','c_parser.py',1574),
  ('selection_statement -> SWITCH LPAREN expression RPAREN pragmacomp_or_statement','selection_statement',5,'p_selection_statement_3','c_parser.py',1578),
  ('iteration_statement -> WHILE LPAREN expression RPAREN pragmacomp_or_statement','iteration_statement',5,'p_iteration_statement_1','c_parser.py',1583),
  ('iteration_statement -> DO pragmacomp_or_statement WHILE LPAREN expression RPAREN SEMI','iteration_statement',7,'p_iteration_statement_2','c_parser.py',1587),
  ('iteration_statement -> FOR LPAREN expression_opt SEMI expression_opt SEMI expression_opt RPAREN pragmacomp_or_statement','iteration_statement',9,'p_iteration_statement_3','c_parser.py',1591),
  ('iteration_statement -> FOR LPAREN declaration expression_opt SEMI expression_opt RPAREN pragmacomp_or_statement','iteration_statement',8,'p_iteration_statement_4','c_parser.py',1595),
  ('jump_statement -> GOTO ID SEMI','jump_statement',3,'p_jump_statement_1','c_parser.py',1600),
  ('jump_statement -> BREAK SEMI','jump_statement',2,'p_jump_statement_2','c_parser.py',1604),
  ('jump_statement -> CONTINUE SEMI','jump_statement',2,'p_jump_statement_3','c_parser.py',1608),
  ('jump_statement -> RETURN expression SEMI','jump_statement',3,'p_jump_statement_4','c_parser.py',1612),
  ('jump_statement -> RETURN SEMI','jump_statement',2,'p_jump_statement_4','c_parser.py',1613),
  ('expression_statement -> expression_opt SEMI','expression_statement',2,'p_expression_statement','c_parser.py',1618),
  ('expression -> assignment_expression','expression',1,'p_expression','c_parser.py',1625),
  ('expression -> expression COMMA assignment_expression','expression',3,'p_expression','c_parser.py',1626),
  ('assignment_expression -> LPAREN compound_statement RPAREN','assignment_expression',3,'p_parenthesized_compound_expression','c_parser.py',1638),
  ('typedef_name -> TYPEID','typedef_name',1,'p_typedef_name','c_parser.py',1642),
  ('assignment_expression -> conditional_expression','assignment_expression',1,'p_assignment_expression','c_parser.py',1646),
  ('assignment_expression -> unary_expression assignment_operator assignment_expression','assignment_expression',3,'p_assignment_expression','c_parser.py',1647),
  ('assignment_operator -> EQUALS','assignment_operator',1,'p_assignment_operator','c_parser.py',1660),
  ('assignment_operator -> XOREQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1661),
  ('assignment_operator -> TIMESEQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1662),
  ('assignment_operator -> DIVEQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1663),
  ('assignment_operator -> MODEQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1664),
  ('assignment_operator -> PLUSEQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1665),
  ('assignment_operator -> MINUSEQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1666),
  ('assignment_operator -> LSHIFTEQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1667),
  ('assignment_operator -> RSHIFTEQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1668),
  ('assignment_operator -> ANDEQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1669),
  ('assignment_operator -> OREQUAL','assignment_operator',1,'p_assignment_operator','c_parser.py',1670),
  ('constant_expression -> conditional_expression','constant_expression',1,'p_constant_expression','c_parser.py',1675),
  ('conditional_expression -> binary_expression','conditional_expression',1,'p_conditional_expression','c_parser.py',1679),
  ('conditional_expression -> binary_expression CONDOP expression COLON conditional_expression','conditional_expression',5,'p_conditional_expression','c_parser.py',1680),
  ('binary_expression -> cast_expression','binary_expression',1,'p_binary_expression','c_parser.py',1688),
  ('binary_expression -> binary_expression TIMES binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1689),
  ('binary_expression -> binary_expression DIVIDE binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1690),
  ('binary_expression -> binary_expression MOD binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1691),
  ('binary_expression -> binary_expression PLUS binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1692),
  ('binary_expression -> binary_expression MINUS binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1693),
  ('binary_expression -> binary_expression RSHIFT binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1694),
  ('binary_expression -> binary_expression LSHIFT binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1695),
  ('binary_expression -> binary_expression LT binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1696),
  ('binary_expression -> binary_expression LE binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1697),
  ('binary_expression -> binary_expression GE binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1698),
  ('binary_expression -> binary_expression GT binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1699),
  ('binary_expression -> binary_expression EQ binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1700),
  ('binary_expression -> binary_expression NE binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1701),
  ('binary_expression -> binary_expression AND binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1702),
  ('binary_expression -> binary_expression OR binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1703),
  ('binary_expression -> binary_expression XOR binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1704),
  ('binary_expression -> binary_expression LAND binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1705),
  ('binary_expression -> binary_expression LOR binary_expression','binary_expression',3,'p_binary_expression','c_parser.py',1706),
  ('cast_expression -> unary_expression','cast_expression',1,'p_cast_expression_1','c_parser.py',1714),
  ('cast_expression -> LPAREN type_name RPAREN cast_expression','cast_expression',4,'p_cast_expression_2','c_parser.py',1718),
  ('unary_expression -> postfix_expression','unary_expression',1,'p_unary_expression_1','c_parser.py',1722),
  ('unary_expression -> PLUSPLUS unary_expression','unary_expression',2,'p_unary_expression_2','c_parser.py',1726),
  ('unary_expression -> MINUSMINUS unary_expression','unary_expression',2,'p_unary_expression_2','c_parser.py',1727),
  ('unary_expression -> unary_operator cast_expression','unary_expression',2,'p_unary_expression_2','c_parser.py',1728),
  ('unary_expression -> SIZEOF unary_expression','unary_expression',2,'p_unary_expression_3','c_parser.py',1733),
  ('unary_expression -> SIZEOF LPAREN type_name RPAREN','unary_expression',4,'p_unary_expression_3','c_parser.py',1734),
  ('unary_expression -> _ALIGNOF LPAREN type_name RPAREN','unary_expression',4,'p_unary_expression_3','c_parser.py',1735),
  ('unary_operator -> AND','unary_operator',1,'p_unary_operator','c_parser.py',1743),
  ('unary_operator -> TIMES','unary_operator',1,'p_unary_operator','c_parser.py',1744),
  ('unary_operator -> PLUS','unary_operator',1,'p_unary_operator','c_parser.py',1745),
  ('unary_operator -> MINUS','unary_operator',1,'p_unary_operator','c_parser.py',1746),
  ('unary_operator -> NOT','unary_operator',1,'p_unary_operator','c_parser.py',1747),
  ('unary_operator -> LNOT','unary_operator',1,'p_unary_operator','c_parser.py',1748),
  ('postfix_expression -> primary_expression','postfix_expression',1,'p_postfix_expression_1','c_parser.py',1753),
  ('postfix_expression -> postfix_expression LBRACKET expression RBRACKET','postfix_expression',4,'p_postfix_expression_2','c_parser.py',1757),
  ('postfix_expression -> postfix_expression LPAREN argument_expression_list RPAREN','postfix_expression',4,'p_postfix_expression_3','c_parser.py',1761),
  ('postfix_expression -> postfix_expression LPAREN RPAREN','postfix_expression',3,'p_postfix_expression_3','c_parser.py',1762),
  ('postfix_expression -> postfix_expression PERIOD ID','postfix_expression',3,'p_postfix_expression_4','c_parser.py',1767),
  ('postfix_expression -> postfix_expression PERIOD TYPEID','postfix_expression',3,'p_postfix_expression_4','c_parser.py',1768),
  ('postfix_expression -> postfix_expression ARROW ID','postfix_expression',3,'p_postfix_expression_4','c_parser.py',1769),
  ('postfix_expression -> postfix_expression ARROW TYPEID','postfix_expression',3,'p_postfix_expression_4','c_parser.py',1770),
  ('postfix_expression -> postfix_expression PLUSPLUS','postfix_expression',2,'p_postfix_expression_5','c_parser.py',1776),
  ('postfix_expression -> postfix_expression MINUSMINUS','postfix_expression',2,'p_postfix_expression_5','c_parser.py',1777),
  ('postfix_expression -> LPAREN type_name RPAREN brace_open initializer_list brace_close','postfix_expression',6,'p_postfix_expression_6','c_parser.py',1782),
  ('postfix_expression -> LPAREN type_name RPAREN brace_open initializer_list COMMA brace_close','postfix_expression',7,'p_postfix_expression_6','c_parser.py',1783),
  ('primary_expression -> identifier','primary_expression',1,'p_primary_expression_1','c_parser.py',1788),
  ('primary_expression -> constant','primary_expression',1,'p_primary_expression_2','c_parser.py',1792),
  ('primary_expression -> unified_string_literal','primary_expression',1,'p_primary_expression_3','c_parser.py',1796),
  ('primary_expression -> unified_wstring_literal','primary_expression',1,'p_primary_expression_3','c_parser.py',1797),
  ('primary_expression -> LPAREN expression RPAREN','primary_expression',3,'p_primary_expression_4','c_parser.py',1802),
  ('primary_expression -> OFFSETOF LPAREN type_name COMMA offsetof_member_designator RPAREN','primary_expression',6,'p_primary_expression_5','c_parser.py',1806),
  ('offsetof_member_designator -> identifier','offsetof_member_designator',1,'p_offsetof_member_designator','c_parser.py',1814),
  ('offsetof_member_designator -> offsetof_member_designator PERIOD identifier','offsetof_member_designator',3,'p_offsetof_member_designator','c_parser.py',1815),
  ('offsetof_member_designator -> offsetof_member_designator LBRACKET expression RBRACKET','offsetof_member_designator',4,'p_offsetof_member_designator','c_parser.py',1816),
  ('argument_expression_list -> assignment_expression','argument_expression_list',1,'p_argument_expression_list','c_parser.py',1828),
  ('argument_expression_list -> argument_expression_list COMMA assignment_expression','argument_expression_list',3,'p_argument_expression_list','c_parser.py',1829),
  ('identifier -> ID','identifier',1,'p_identifier','c_parser.py',1838),
  ('constant -> INT_CONST_DEC','constant',1,'p_constant_1','c_parser.py',1842),
  ('constant -> INT_CONST_OCT','constant',1,'p_constant_1','c_parser.py',1843),
  ('constant -> INT_CONST_HEX','constant',1,'p_constant_1','c_parser.py',1844),
  ('constant -> INT_CONST_BIN','constant',1,'p_constant_1','c_parser.py',1845),
  ('constant -> INT_CONST_CHAR','constant',1,'p_constant_1','c_parser.py',1846),
  ('constant -> FLOAT_CONST','constant',1,'p_constant_2','c_parser.py',1865),
  ('constant -> HEX_FLOAT_CONST','constant',1,'p_constant_2','c_parser.py',1866),
  ('constant -> CHAR_CONST','constant',1,'p_constant_3','c_parser.py',1882),
  ('constant -> WCHAR_CONST','constant',1,'p_constant_3','c_parser.py',1883),
  ('constant -> U8CHAR_CONST','constant',1,'p_constant_3','c_parser.py',1884),
  ('constant -> U16CHAR_CONST','constant',1,'p_constant_3','c_parser.py',1885),
  ('constant -> U32CHAR_CONST','constant',1,'p_constant_3','c_parser.py',1886),
  ('unified_string_literal -> STRING_LITERAL','unified_string_literal',1,'p_unified_string_literal','c_parser.py',1897),
  ('unified_string_literal -> unified_string_literal STRING_LITERAL','unified_string_literal',2,'p_unified_string_literal','c_parser.py',1898),
  ('unified_wstring_literal -> WSTRING_LITERAL','unified_wstring_literal',1,'p_unified_wstring_literal','c_parser.py',1908),
  ('unified_wstring_literal -> U8STRING_LITERAL','unified_wstring_literal',1,'p_unified_wstring_literal','c_parser.py',1909),
  ('unified_wstring_literal -> U16STRING_LITERAL','unified_wstring_literal',1,'p_unified_wstring_literal','c_parser.py',1910),
  ('unified_wstring_literal -> U32STRING_LITERAL','unified_wstring_literal',1,'p_unified_wstring_literal','c_parser.py',1911),
  ('unified_wstring_literal -> unified_wstring_literal WSTRING_LITERAL','unified_wstring_literal',2,'p_unified_wstring_literal','c_parser.py',1912),
  ('unified_wstring_literal -> unified_wstring_literal U8STRING_LITERAL','unified_wstring_literal',2,'p_unified_wstring_literal','c_parser.py',1913),
  ('unified_wstring_literal -> unified_wstring_literal U16STRING_LITERAL','unified_wstring_literal',2,'p_unified_wstring_literal','c_parser.py',1914),
  ('unified_wstring_literal -> unified_wstring_literal U32STRING_LITERAL','unified_wstring_literal',2,'p_unified_wstring_literal','c_parser.py',1915),
  ('brace_open -> LBRACE','brace_open',1,'p_brace_open','c_parser.py',1925),
  ('brace_close -> RBRACE','brace_close',1,'p_brace_close','c_parser.py',1931),
  ('empty -> <empty>','empty',0,'p_empty','c_parser.py',1937),
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\satask.py ===
"""
Module to evaluate the proposition with assumptions using SAT algorithm.
"""

from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.kind import NumberKind, UndefinedKind
from sympy.assumptions.ask_generated import get_all_known_matrix_facts, get_all_known_number_facts
from sympy.assumptions.assume import global_assumptions, AppliedPredicate
from sympy.assumptions.sathandlers import class_fact_registry
from sympy.core import oo
from sympy.logic.inference import satisfiable
from sympy.assumptions.cnf import CNF, EncodedCNF
from sympy.matrices.kind import MatrixKind


def satask(proposition, assumptions=True, context=global_assumptions,
        use_known_facts=True, iterations=oo):
    """
    Function to evaluate the proposition with assumptions using SAT algorithm.

    This function extracts every fact relevant to the expressions composing
    proposition and assumptions. For example, if a predicate containing
    ``Abs(x)`` is proposed, then ``Q.zero(Abs(x)) | Q.positive(Abs(x))``
    will be found and passed to SAT solver because ``Q.nonnegative`` is
    registered as a fact for ``Abs``.

    Proposition is evaluated to ``True`` or ``False`` if the truth value can be
    determined. If not, ``None`` is returned.

    Parameters
    ==========

    proposition : Any boolean expression.
        Proposition which will be evaluated to boolean value.

    assumptions : Any boolean expression, optional.
        Local assumptions to evaluate the *proposition*.

    context : AssumptionsContext, optional.
        Default assumptions to evaluate the *proposition*. By default,
        this is ``sympy.assumptions.global_assumptions`` variable.

    use_known_facts : bool, optional.
        If ``True``, facts from ``sympy.assumptions.ask_generated``
        module are passed to SAT solver as well.

    iterations : int, optional.
        Number of times that relevant facts are recursively extracted.
        Default is infinite times until no new fact is found.

    Returns
    =======

    ``True``, ``False``, or ``None``

    Examples
    ========

    >>> from sympy import Abs, Q
    >>> from sympy.assumptions.satask import satask
    >>> from sympy.abc import x
    >>> satask(Q.zero(Abs(x)), Q.zero(x))
    True

    """
    props = CNF.from_prop(proposition)
    _props = CNF.from_prop(~proposition)

    assumptions = CNF.from_prop(assumptions)

    context_cnf = CNF()
    if context:
        context_cnf = context_cnf.extend(context)

    sat = get_all_relevant_facts(props, assumptions, context_cnf,
        use_known_facts=use_known_facts, iterations=iterations)
    sat.add_from_cnf(assumptions)
    if context:
        sat.add_from_cnf(context_cnf)

    return check_satisfiability(props, _props, sat)


def check_satisfiability(prop, _prop, factbase):
    sat_true = factbase.copy()
    sat_false = factbase.copy()
    sat_true.add_from_cnf(prop)
    sat_false.add_from_cnf(_prop)
    can_be_true = satisfiable(sat_true)
    can_be_false = satisfiable(sat_false)

    if can_be_true and can_be_false:
        return None

    if can_be_true and not can_be_false:
        return True

    if not can_be_true and can_be_false:
        return False

    if not can_be_true and not can_be_false:
        # TODO: Run additional checks to see which combination of the
        # assumptions, global_assumptions, and relevant_facts are
        # inconsistent.
        raise ValueError("Inconsistent assumptions")


def extract_predargs(proposition, assumptions=None, context=None):
    """
    Extract every expression in the argument of predicates from *proposition*,
    *assumptions* and *context*.

    Parameters
    ==========

    proposition : sympy.assumptions.cnf.CNF

    assumptions : sympy.assumptions.cnf.CNF, optional.

    context : sympy.assumptions.cnf.CNF, optional.
        CNF generated from assumptions context.

    Examples
    ========

    >>> from sympy import Q, Abs
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.satask import extract_predargs
    >>> from sympy.abc import x, y
    >>> props = CNF.from_prop(Q.zero(Abs(x*y)))
    >>> assump = CNF.from_prop(Q.zero(x) & Q.zero(y))
    >>> extract_predargs(props, assump)
    {x, y, Abs(x*y)}

    """
    req_keys = find_symbols(proposition)
    keys = proposition.all_predicates()
    # XXX: We need this since True/False are not Basic
    lkeys = set()
    if assumptions:
        lkeys |= assumptions.all_predicates()
    if context:
        lkeys |= context.all_predicates()

    lkeys = lkeys - {S.true, S.false}
    tmp_keys = None
    while tmp_keys != set():
        tmp = set()
        for l in lkeys:
            syms = find_symbols(l)
            if (syms & req_keys) != set():
                tmp |= syms
        tmp_keys = tmp - req_keys
        req_keys |= tmp_keys
    keys |= {l for l in lkeys if find_symbols(l) & req_keys != set()}

    exprs = set()
    for key in keys:
        if isinstance(key, AppliedPredicate):
            exprs |= set(key.arguments)
        else:
            exprs.add(key)
    return exprs

def find_symbols(pred):
    """
    Find every :obj:`~.Symbol` in *pred*.

    Parameters
    ==========

    pred : sympy.assumptions.cnf.CNF, or any Expr.

    """
    if isinstance(pred, CNF):
        symbols = set()
        for a in pred.all_predicates():
            symbols |= find_symbols(a)
        return symbols
    return pred.atoms(Symbol)


def get_relevant_clsfacts(exprs, relevant_facts=None):
    """
    Extract relevant facts from the items in *exprs*. Facts are defined in
    ``assumptions.sathandlers`` module.

    This function is recursively called by ``get_all_relevant_facts()``.

    Parameters
    ==========

    exprs : set
        Expressions whose relevant facts are searched.

    relevant_facts : sympy.assumptions.cnf.CNF, optional.
        Pre-discovered relevant facts.

    Returns
    =======

    exprs : set
        Candidates for next relevant fact searching.

    relevant_facts : sympy.assumptions.cnf.CNF
        Updated relevant facts.

    Examples
    ========

    Here, we will see how facts relevant to ``Abs(x*y)`` are recursively
    extracted. On the first run, set containing the expression is passed
    without pre-discovered relevant facts. The result is a set containing
    candidates for next run, and ``CNF()`` instance containing facts
    which are relevant to ``Abs`` and its argument.

    >>> from sympy import Abs
    >>> from sympy.assumptions.satask import get_relevant_clsfacts
    >>> from sympy.abc import x, y
    >>> exprs = {Abs(x*y)}
    >>> exprs, facts = get_relevant_clsfacts(exprs)
    >>> exprs
    {x*y}
    >>> facts.clauses #doctest: +SKIP
    {frozenset({Literal(Q.odd(Abs(x*y)), False), Literal(Q.odd(x*y), True)}),
    frozenset({Literal(Q.zero(Abs(x*y)), False), Literal(Q.zero(x*y), True)}),
    frozenset({Literal(Q.even(Abs(x*y)), False), Literal(Q.even(x*y), True)}),
    frozenset({Literal(Q.zero(Abs(x*y)), True), Literal(Q.zero(x*y), False)}),
    frozenset({Literal(Q.even(Abs(x*y)), False),
                Literal(Q.odd(Abs(x*y)), False),
                Literal(Q.odd(x*y), True)}),
    frozenset({Literal(Q.even(Abs(x*y)), False),
                Literal(Q.even(x*y), True),
                Literal(Q.odd(Abs(x*y)), False)}),
    frozenset({Literal(Q.positive(Abs(x*y)), False),
                Literal(Q.zero(Abs(x*y)), False)})}

    We pass the first run's results to the second run, and get the expressions
    for next run and updated facts.

    >>> exprs, facts = get_relevant_clsfacts(exprs, relevant_facts=facts)
    >>> exprs
    {x, y}

    On final run, no more candidate is returned thus we know that all
    relevant facts are successfully retrieved.

    >>> exprs, facts = get_relevant_clsfacts(exprs, relevant_facts=facts)
    >>> exprs
    set()

    """
    if not relevant_facts:
        relevant_facts = CNF()

    newexprs = set()
    for expr in exprs:
        for fact in class_fact_registry(expr):
            newfact = CNF.to_CNF(fact)
            relevant_facts = relevant_facts._and(newfact)
            for key in newfact.all_predicates():
                if isinstance(key, AppliedPredicate):
                    newexprs |= set(key.arguments)

    return newexprs - exprs, relevant_facts


def get_all_relevant_facts(proposition, assumptions, context,
        use_known_facts=True, iterations=oo):
    """
    Extract all relevant facts from *proposition* and *assumptions*.

    This function extracts the facts by recursively calling
    ``get_relevant_clsfacts()``. Extracted facts are converted to
    ``EncodedCNF`` and returned.

    Parameters
    ==========

    proposition : sympy.assumptions.cnf.CNF
        CNF generated from proposition expression.

    assumptions : sympy.assumptions.cnf.CNF
        CNF generated from assumption expression.

    context : sympy.assumptions.cnf.CNF
        CNF generated from assumptions context.

    use_known_facts : bool, optional.
        If ``True``, facts from ``sympy.assumptions.ask_generated``
        module are encoded as well.

    iterations : int, optional.
        Number of times that relevant facts are recursively extracted.
        Default is infinite times until no new fact is found.

    Returns
    =======

    sympy.assumptions.cnf.EncodedCNF

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.assumptions.satask import get_all_relevant_facts
    >>> from sympy.abc import x, y
    >>> props = CNF.from_prop(Q.nonzero(x*y))
    >>> assump = CNF.from_prop(Q.nonzero(x))
    >>> context = CNF.from_prop(Q.nonzero(y))
    >>> get_all_relevant_facts(props, assump, context) #doctest: +SKIP
    <sympy.assumptions.cnf.EncodedCNF at 0x7f09faa6ccd0>

    """
    # The relevant facts might introduce new keys, e.g., Q.zero(x*y) will
    # introduce the keys Q.zero(x) and Q.zero(y), so we need to run it until
    # we stop getting new things. Hopefully this strategy won't lead to an
    # infinite loop in the future.
    i = 0
    relevant_facts = CNF()
    all_exprs = set()
    while True:
        if i == 0:
            exprs = extract_predargs(proposition, assumptions, context)
        all_exprs |= exprs
        exprs, relevant_facts = get_relevant_clsfacts(exprs, relevant_facts)
        i += 1
        if i >= iterations:
            break
        if not exprs:
            break

    if use_known_facts:
        known_facts_CNF = CNF()

        if any(expr.kind == MatrixKind(NumberKind) for expr in all_exprs):
            known_facts_CNF.add_clauses(get_all_known_matrix_facts())
        # check for undefinedKind since kind system isn't fully implemented
        if any(((expr.kind == NumberKind) or (expr.kind == UndefinedKind)) for expr in all_exprs):
            known_facts_CNF.add_clauses(get_all_known_number_facts())

        kf_encoded = EncodedCNF()
        kf_encoded.from_cnf(known_facts_CNF)

        def translate_literal(lit, delta):
            if lit > 0:
                return lit + delta
            else:
                return lit - delta

        def translate_data(data, delta):
            return [{translate_literal(i, delta) for i in clause} for clause in data]
        data = []
        symbols = []
        n_lit = len(kf_encoded.symbols)
        for i, expr in enumerate(all_exprs):
            symbols += [pred(expr) for pred in kf_encoded.symbols]
            data += translate_data(kf_encoded.data, i * n_lit)

        encoding = dict(list(zip(symbols, range(1, len(symbols)+1))))
        ctx = EncodedCNF(data, encoding)
    else:
        ctx = EncodedCNF()

    ctx.add_from_cnf(relevant_facts)

    return ctx

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\continued_fraction.py ===
from __future__ import annotations
import itertools
from sympy.core.exprtools import factor_terms
from sympy.core.numbers import Integer, Rational
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import _sympify
from sympy.utilities.misc import as_int


def continued_fraction(a) -> list:
    """Return the continued fraction representation of a Rational or
    quadratic irrational.

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction
    >>> from sympy import sqrt
    >>> continued_fraction((1 + 2*sqrt(3))/5)
    [0, 1, [8, 3, 34, 3]]

    See Also
    ========
    continued_fraction_periodic, continued_fraction_reduce, continued_fraction_convergents
    """
    e = _sympify(a)
    if all(i.is_Rational for i in e.atoms()):
        if e.is_Integer:
            return continued_fraction_periodic(e, 1, 0)
        elif e.is_Rational:
            return continued_fraction_periodic(e.p, e.q, 0)
        elif e.is_Pow and e.exp is S.Half and e.base.is_Integer:
            return continued_fraction_periodic(0, 1, e.base)
        elif e.is_Mul and len(e.args) == 2 and (
                e.args[0].is_Rational and
                e.args[1].is_Pow and
                e.args[1].base.is_Integer and
                e.args[1].exp is S.Half):
            a, b = e.args
            return continued_fraction_periodic(0, a.q, b.base, a.p)
        else:
            # this should not have to work very hard- no
            # simplification, cancel, etc... which should be
            # done by the user.  e.g. This is a fancy 1 but
            # the user should simplify it first:
            # sqrt(2)*(1 + sqrt(2))/(sqrt(2) + 2)
            p, d = e.expand().as_numer_denom()
            if d.is_Integer:
                if p.is_Rational:
                    return continued_fraction_periodic(p, d)
                # look for a + b*c
                # with c = sqrt(s)
                if p.is_Add and len(p.args) == 2:
                    a, bc = p.args
                else:
                    a = S.Zero
                    bc = p
                if a.is_Integer:
                    b = S.NaN
                    if bc.is_Mul and len(bc.args) == 2:
                        b, c = bc.args
                    elif bc.is_Pow:
                        b = Integer(1)
                        c = bc
                    if b.is_Integer and (
                            c.is_Pow and c.exp is S.Half and
                            c.base.is_Integer):
                        # (a + b*sqrt(c))/d
                        c = c.base
                        return continued_fraction_periodic(a, d, c, b)
    raise ValueError(
        'expecting a rational or quadratic irrational, not %s' % e)


def continued_fraction_periodic(p, q, d=0, s=1) -> list:
    r"""
    Find the periodic continued fraction expansion of a quadratic irrational.

    Compute the continued fraction expansion of a rational or a
    quadratic irrational number, i.e. `\frac{p + s\sqrt{d}}{q}`, where
    `p`, `q \ne 0` and `d \ge 0` are integers.

    Returns the continued fraction representation (canonical form) as
    a list of integers, optionally ending (for quadratic irrationals)
    with list of integers representing the repeating digits.

    Parameters
    ==========

    p : int
        the rational part of the number's numerator
    q : int
        the denominator of the number
    d : int, optional
        the irrational part (discriminator) of the number's numerator
    s : int, optional
        the coefficient of the irrational part

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction_periodic
    >>> continued_fraction_periodic(3, 2, 7)
    [2, [1, 4, 1, 1]]

    Golden ratio has the simplest continued fraction expansion:

    >>> continued_fraction_periodic(1, 2, 5)
    [[1]]

    If the discriminator is zero or a perfect square then the number will be a
    rational number:

    >>> continued_fraction_periodic(4, 3, 0)
    [1, 3]
    >>> continued_fraction_periodic(4, 3, 49)
    [3, 1, 2]

    See Also
    ========

    continued_fraction_iterator, continued_fraction_reduce

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Periodic_continued_fraction
    .. [2] K. Rosen. Elementary Number theory and its applications.
           Addison-Wesley, 3 Sub edition, pages 379-381, January 1992.

    """
    from sympy.functions import sqrt, floor

    p, q, d, s = list(map(as_int, [p, q, d, s]))

    if d < 0:
        raise ValueError("expected non-negative for `d` but got %s" % d)

    if q == 0:
        raise ValueError("The denominator cannot be 0.")

    if not s:
        d = 0

    # check for rational case
    sd = sqrt(d)
    if sd.is_Integer:
        return list(continued_fraction_iterator(Rational(p + s*sd, q)))

    # irrational case with sd != Integer
    if q < 0:
        p, q, s = -p, -q, -s

    n = (p + s*sd)/q
    if n < 0:
        w = floor(-n)
        f = -n - w
        one_f = continued_fraction(1 - f)  # 1-f < 1 so cf is [0 ... [...]]
        one_f[0] -= w + 1
        return one_f

    d *= s**2
    sd *= s

    if (d - p**2)%q:
        d *= q**2
        sd *= q
        p *= q
        q *= q

    terms: list[int] = []
    pq = {}

    while (p, q) not in pq:
        pq[(p, q)] = len(terms)
        terms.append((p + sd)//q)
        p = terms[-1]*q - p
        q = (d - p**2)//q

    i = pq[(p, q)]
    return terms[:i] + [terms[i:]]  # type: ignore


def continued_fraction_reduce(cf):
    """
    Reduce a continued fraction to a rational or quadratic irrational.

    Compute the rational or quadratic irrational number from its
    terminating or periodic continued fraction expansion.  The
    continued fraction expansion (cf) should be supplied as a
    terminating iterator supplying the terms of the expansion.  For
    terminating continued fractions, this is equivalent to
    ``list(continued_fraction_convergents(cf))[-1]``, only a little more
    efficient.  If the expansion has a repeating part, a list of the
    repeating terms should be returned as the last element from the
    iterator.  This is the format returned by
    continued_fraction_periodic.

    For quadratic irrationals, returns the largest solution found,
    which is generally the one sought, if the fraction is in canonical
    form (all terms positive except possibly the first).

    Examples
    ========

    >>> from sympy.ntheory.continued_fraction import continued_fraction_reduce
    >>> continued_fraction_reduce([1, 2, 3, 4, 5])
    225/157
    >>> continued_fraction_reduce([-2, 1, 9, 7, 1, 2])
    -256/233
    >>> continued_fraction_reduce([2, 1, 2, 1, 1, 4, 1, 1, 6, 1, 1, 8]).n(10)
    2.718281835
    >>> continued_fraction_reduce([1, 4, 2, [3, 1]])
    (sqrt(21) + 287)/238
    >>> continued_fraction_reduce([[1]])
    (1 + sqrt(5))/2
    >>> from sympy.ntheory.continued_fraction import continued_fraction_periodic
    >>> continued_fraction_reduce(continued_fraction_periodic(8, 5, 13))
    (sqrt(13) + 8)/5

    See Also
    ========

    continued_fraction_periodic

    """
    from sympy.solvers import solve

    period = []
    x = Dummy('x')

    def untillist(cf):
        for nxt in cf:
            if isinstance(nxt, list):
                period.extend(nxt)
                yield x
                break
            yield nxt

    a = S.Zero
    for a in continued_fraction_convergents(untillist(cf)):
        pass

    if period:
        y = Dummy('y')
        solns = solve(continued_fraction_reduce(period + [y]) - y, y)
        solns.sort()
        pure = solns[-1]
        rv = a.subs(x, pure).radsimp()
    else:
        rv = a
    if rv.is_Add:
        rv = factor_terms(rv)
        if rv.is_Mul and rv.args[0] == -1:
            rv = rv.func(*rv.args)
    return rv


def continued_fraction_iterator(x):
    """
    Return continued fraction expansion of x as iterator.

    Examples
    ========

    >>> from sympy import Rational, pi
    >>> from sympy.ntheory.continued_fraction import continued_fraction_iterator

    >>> list(continued_fraction_iterator(Rational(3, 8)))
    [0, 2, 1, 2]
    >>> list(continued_fraction_iterator(Rational(-3, 8)))
    [-1, 1, 1, 1, 2]

    >>> for i, v in enumerate(continued_fraction_iterator(pi)):
    ...     if i > 7:
    ...         break
    ...     print(v)
    3
    7
    15
    1
    292
    1
    1
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Continued_fraction

    """
    from sympy.functions import floor
    while True:
        i = floor(x)
        yield i
        x -= i
        if not x:
            break
        x = 1/x


def continued_fraction_convergents(cf):
    """
    Return an iterator over the convergents of a continued fraction (cf).

    The parameter should be in either of the following to forms:
    - A list of partial quotients, possibly with the last element being a list
    of repeating partial quotients, such as might be returned by
    continued_fraction and continued_fraction_periodic.
    - An iterable returning successive partial quotients of the continued
    fraction, such as might be returned by continued_fraction_iterator.

    In computing the convergents, the continued fraction need not be strictly
    in canonical form (all integers, all but the first positive).
    Rational and negative elements may be present in the expansion.

    Examples
    ========

    >>> from sympy.core import pi
    >>> from sympy import S
    >>> from sympy.ntheory.continued_fraction import \
            continued_fraction_convergents, continued_fraction_iterator

    >>> list(continued_fraction_convergents([0, 2, 1, 2]))
    [0, 1/2, 1/3, 3/8]

    >>> list(continued_fraction_convergents([1, S('1/2'), -7, S('1/4')]))
    [1, 3, 19/5, 7]

    >>> it = continued_fraction_convergents(continued_fraction_iterator(pi))
    >>> for n in range(7):
    ...     print(next(it))
    3
    22/7
    333/106
    355/113
    103993/33102
    104348/33215
    208341/66317

    >>> it = continued_fraction_convergents([1, [1, 2]])  # sqrt(3)
    >>> for n in range(7):
    ...     print(next(it))
    1
    2
    5/3
    7/4
    19/11
    26/15
    71/41

    See Also
    ========

    continued_fraction_iterator, continued_fraction, continued_fraction_periodic

    """
    if isinstance(cf, list) and isinstance(cf[-1], list):
        cf = itertools.chain(cf[:-1], itertools.cycle(cf[-1]))
    p_2, q_2 = S.Zero, S.One
    p_1, q_1 = S.One, S.Zero
    for a in cf:
        p, q = a*p_1 + p_2, a*q_1 + q_2
        p_2, q_2 = p_1, q_1
        p_1, q_1 = p, q
        yield p/q

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\support\relative_locator.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import warnings
from typing import Dict, List, NoReturn, Optional, Union, overload

from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By, ByType
from selenium.webdriver.remote.webelement import WebElement


def with_tag_name(tag_name: str) -> "RelativeBy":
    """Start searching for relative objects using a tag name.

    Parameters:
    -----------
    tag_name : str
        The DOM tag of element to start searching.

    Returns:
    --------
    RelativeBy
        Use this object to create filters within a `find_elements` call.

    Raises:
    -------
    WebDriverException
        If `tag_name` is None.

    Notes:
    ------
    - This method is deprecated and may be removed in future versions.
    - Please use `locate_with` instead.
    """
    warnings.warn("This method is deprecated and may be removed in future versions. Please use `locate_with` instead.")
    if not tag_name:
        raise WebDriverException("tag_name can not be null")
    return RelativeBy({By.CSS_SELECTOR: tag_name})


def locate_with(by: ByType, using: str) -> "RelativeBy":
    """Start searching for relative objects your search criteria with By.

    Parameters:
    -----------
    by : ByType
        The method to find the element.

    using : str
        The value from `By` passed in.

    Returns:
    --------
    RelativeBy
        Use this object to create filters within a `find_elements` call.

    Example:
    --------
    >>> lowest = driver.find_element(By.ID, "below")
    >>> elements = driver.find_elements(locate_with(By.CSS_SELECTOR, "p").above(lowest))
    """
    assert by is not None, "Please pass in a by argument"
    assert using is not None, "Please pass in a using argument"
    return RelativeBy({by: using})


class RelativeBy:
    """Gives the opportunity to find elements based on their relative location
    on the page from a root element. It is recommended that you use the helper
    function to create it.

    Example:
    --------
    >>> lowest = driver.find_element(By.ID, "below")
    >>> elements = driver.find_elements(locate_with(By.CSS_SELECTOR, "p").above(lowest))
    >>> ids = [el.get_attribute("id") for el in elements]
    >>> assert "above" in ids
    >>> assert "mid" in ids
    """

    LocatorType = Dict[ByType, str]

    def __init__(self, root: Optional[Dict[ByType, str]] = None, filters: Optional[List] = None):
        """Creates a new RelativeBy object. It is preferred if you use the
        `locate_with` method as this signature could change.

        Attributes:
        -----------
        root : Dict[By, str]
            - A dict with `By` enum as the key and the search query as the value

        filters : List
            - A list of the filters that will be searched. If none are passed
                in please use the fluent API on the object to create the filters
        """
        self.root = root
        self.filters = filters or []

    @overload
    def above(self, element_or_locator: Union[WebElement, LocatorType]) -> "RelativeBy": ...

    @overload
    def above(self, element_or_locator: None = None) -> "NoReturn": ...

    def above(self, element_or_locator: Union[WebElement, LocatorType, None] = None) -> "RelativeBy":
        """Add a filter to look for elements above.

        Parameters:
        -----------
        element_or_locator : Union[WebElement, Dict, None]
            Element to look above

        Returns:
        --------
        RelativeBy

        Raises:
        -------
        WebDriverException
            If `element_or_locator` is None.

        Example:
        --------
        >>> lowest = driver.find_element(By.ID, "below")
        >>> elements = driver.find_elements(locate_with(By.CSS_SELECTOR, "p").above(lowest))
        """
        if not element_or_locator:
            raise WebDriverException("Element or locator must be given when calling above method")

        self.filters.append({"kind": "above", "args": [element_or_locator]})
        return self

    @overload
    def below(self, element_or_locator: Union[WebElement, LocatorType]) -> "RelativeBy": ...

    @overload
    def below(self, element_or_locator: None = None) -> "NoReturn": ...

    def below(self, element_or_locator: Union[WebElement, Dict, None] = None) -> "RelativeBy":
        """Add a filter to look for elements below.

        Parameters:
        -----------
        element_or_locator : Union[WebElement, Dict, None]
            Element to look below

        Returns:
        --------
        RelativeBy

        Raises:
        -------
        WebDriverException
            If `element_or_locator` is None.

        Example:
        --------
        >>> highest = driver.find_element(By.ID, "high")
        >>> elements = driver.find_elements(locate_with(By.CSS_SELECTOR, "p").below(highest))
        """
        if not element_or_locator:
            raise WebDriverException("Element or locator must be given when calling below method")

        self.filters.append({"kind": "below", "args": [element_or_locator]})
        return self

    @overload
    def to_left_of(self, element_or_locator: Union[WebElement, LocatorType]) -> "RelativeBy": ...

    @overload
    def to_left_of(self, element_or_locator: None = None) -> "NoReturn": ...

    def to_left_of(self, element_or_locator: Union[WebElement, Dict, None] = None) -> "RelativeBy":
        """Add a filter to look for elements to the left of.

        Parameters:
        -----------
        element_or_locator : Union[WebElement, Dict, None]
            Element to look to the left of

        Returns:
        --------
        RelativeBy

        Raises:
        -------
        WebDriverException
            If `element_or_locator` is None.

        Example:
        --------
        >>> right = driver.find_element(By.ID, "right")
        >>> elements = driver.find_elements(locate_with(By.CSS_SELECTOR, "p").to_left_of(right))
        """
        if not element_or_locator:
            raise WebDriverException("Element or locator must be given when calling to_left_of method")

        self.filters.append({"kind": "left", "args": [element_or_locator]})
        return self

    @overload
    def to_right_of(self, element_or_locator: Union[WebElement, LocatorType]) -> "RelativeBy": ...

    @overload
    def to_right_of(self, element_or_locator: None = None) -> "NoReturn": ...

    def to_right_of(self, element_or_locator: Union[WebElement, Dict, None] = None) -> "RelativeBy":
        """Add a filter to look for elements right of.

        Parameters:
        -----------
        element_or_locator : Union[WebElement, Dict, None]
            Element to look right of

        Returns:
        --------
        RelativeBy

        Raises:
        -------
        WebDriverException
            If `element_or_locator` is None.

        Example:
        --------
        >>> left = driver.find_element(By.ID, "left")
        >>> elements = driver.find_elements(locate_with(By.CSS_SELECTOR, "p").to_right_of(left))
        """
        if not element_or_locator:
            raise WebDriverException("Element or locator must be given when calling to_right_of method")

        self.filters.append({"kind": "right", "args": [element_or_locator]})
        return self

    @overload
    def straight_above(self, element_or_locator: Union[WebElement, LocatorType]) -> "RelativeBy": ...

    @overload
    def straight_above(self, element_or_locator: None = None) -> "NoReturn": ...

    def straight_above(self, element_or_locator: Union[WebElement, LocatorType, None] = None) -> "RelativeBy":
        """Add a filter to look for elements above.

        :Args:
            - element_or_locator: Element to look above
        """
        if not element_or_locator:
            raise WebDriverException("Element or locator must be given when calling above method")

        self.filters.append({"kind": "straightAbove", "args": [element_or_locator]})
        return self

    @overload
    def straight_below(self, element_or_locator: Union[WebElement, LocatorType]) -> "RelativeBy": ...

    @overload
    def straight_below(self, element_or_locator: None = None) -> "NoReturn": ...

    def straight_below(self, element_or_locator: Union[WebElement, Dict, None] = None) -> "RelativeBy":
        """Add a filter to look for elements below.

        :Args:
            - element_or_locator: Element to look below
        """
        if not element_or_locator:
            raise WebDriverException("Element or locator must be given when calling below method")

        self.filters.append({"kind": "straightBelow", "args": [element_or_locator]})
        return self

    @overload
    def straight_left_of(self, element_or_locator: Union[WebElement, LocatorType]) -> "RelativeBy": ...

    @overload
    def straight_left_of(self, element_or_locator: None = None) -> "NoReturn": ...

    def straight_left_of(self, element_or_locator: Union[WebElement, Dict, None] = None) -> "RelativeBy":
        """Add a filter to look for elements to the left of.

        :Args:
            - element_or_locator: Element to look to the left of
        """
        if not element_or_locator:
            raise WebDriverException("Element or locator must be given when calling to_left_of method")

        self.filters.append({"kind": "straightLeft", "args": [element_or_locator]})
        return self

    @overload
    def straight_right_of(self, element_or_locator: Union[WebElement, LocatorType]) -> "RelativeBy": ...

    @overload
    def straight_right_of(self, element_or_locator: None = None) -> "NoReturn": ...

    def straight_right_of(self, element_or_locator: Union[WebElement, Dict, None] = None) -> "RelativeBy":
        """Add a filter to look for elements right of.

        :Args:
            - element_or_locator: Element to look right of
        """
        if not element_or_locator:
            raise WebDriverException("Element or locator must be given when calling to_right_of method")

        self.filters.append({"kind": "straightRight", "args": [element_or_locator]})
        return self

    @overload
    def near(self, element_or_locator: Union[WebElement, LocatorType], distance: int = 50) -> "RelativeBy": ...

    @overload
    def near(self, element_or_locator: None = None, distance: int = 50) -> "NoReturn": ...

    def near(self, element_or_locator: Union[WebElement, LocatorType, None] = None, distance: int = 50) -> "RelativeBy":
        """Add a filter to look for elements near.

        Parameters:
        -----------
        element_or_locator : Union[WebElement, Dict, None]
            Element to look near by the element or within a distance

        distance : int
            Distance in pixel

        Returns:
        --------
        RelativeBy

        Raises:
        -------
        WebDriverException
            - If `element_or_locator` is None
            - If `distance` is less than or equal to 0.

        Example:
        --------
        >>> near = driver.find_element(By.ID, "near")
        >>> elements = driver.find_elements(locate_with(By.CSS_SELECTOR, "p").near(near, 50))
        """
        if not element_or_locator:
            raise WebDriverException("Element or locator must be given when calling near method")
        if distance <= 0:
            raise WebDriverException("Distance must be positive")

        self.filters.append({"kind": "near", "args": [element_or_locator, distance]})
        return self

    def to_dict(self) -> Dict:
        """Create a dict that will be passed to the driver to start searching
        for the element."""
        return {
            "relative": {
                "root": self.root,
                "filters": self.filters,
            }
        }

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\finitefield.py ===
"""Implementation of :class:`FiniteField` class. """

import operator

from sympy.external.gmpy import GROUND_TYPES
from sympy.utilities.decorator import doctest_depends_on

from sympy.core.numbers import int_valued
from sympy.polys.domains.field import Field

from sympy.polys.domains.modularinteger import ModularIntegerFactory
from sympy.polys.domains.simpledomain import SimpleDomain
from sympy.polys.galoistools import gf_zassenhaus, gf_irred_p_rabin
from sympy.polys.polyerrors import CoercionFailed
from sympy.utilities import public
from sympy.polys.domains.groundtypes import SymPyInteger


if GROUND_TYPES == 'flint':
    __doctest_skip__ = ['FiniteField']


if GROUND_TYPES == 'flint':
    import flint
    # Don't use python-flint < 0.5.0 because nmod was missing some features in
    # previous versions of python-flint and fmpz_mod was not yet added.
    _major, _minor, *_ = flint.__version__.split('.')
    if (int(_major), int(_minor)) < (0, 5):
        flint = None
else:
    flint = None


def _modular_int_factory_nmod(mod):
    # nmod only recognises int
    index = operator.index
    mod = index(mod)
    nmod = flint.nmod
    nmod_poly = flint.nmod_poly

    # flint's nmod is only for moduli up to 2^64-1 (on a 64-bit machine)
    try:
        nmod(0, mod)
    except OverflowError:
        return None, None

    def ctx(x):
        try:
            return nmod(x, mod)
        except TypeError:
            return nmod(index(x), mod)

    def poly_ctx(cs):
        return nmod_poly(cs, mod)

    return ctx, poly_ctx


def _modular_int_factory_fmpz_mod(mod):
    index = operator.index
    fctx = flint.fmpz_mod_ctx(mod)
    fctx_poly = flint.fmpz_mod_poly_ctx(mod)
    fmpz_mod_poly = flint.fmpz_mod_poly

    def ctx(x):
        try:
            return fctx(x)
        except TypeError:
            # x might be Integer
            return fctx(index(x))

    def poly_ctx(cs):
        return fmpz_mod_poly(cs, fctx_poly)

    return ctx, poly_ctx


def _modular_int_factory(mod, dom, symmetric, self):
    # Convert the modulus to ZZ
    try:
        mod = dom.convert(mod)
    except CoercionFailed:
        raise ValueError('modulus must be an integer, got %s' % mod)

    ctx, poly_ctx, is_flint = None, None, False

    # Don't use flint if the modulus is not prime as it often crashes.
    if flint is not None and mod.is_prime():

        is_flint = True

        # Try to use flint's nmod first
        ctx, poly_ctx = _modular_int_factory_nmod(mod)

        if ctx is None:
            # Use fmpz_mod for larger moduli
            ctx, poly_ctx = _modular_int_factory_fmpz_mod(mod)

    if ctx is None:
        # Use the Python implementation if flint is not available or the
        # modulus is not prime.
        ctx = ModularIntegerFactory(mod, dom, symmetric, self)
        poly_ctx = None  # not used

    return ctx, poly_ctx, is_flint


@public
@doctest_depends_on(modules=['python', 'gmpy'])
class FiniteField(Field, SimpleDomain):
    r"""Finite field of prime order :ref:`GF(p)`

    A :ref:`GF(p)` domain represents a `finite field`_ `\mathbb{F}_p` of prime
    order as :py:class:`~.Domain` in the domain system (see
    :ref:`polys-domainsintro`).

    A :py:class:`~.Poly` created from an expression with integer
    coefficients will have the domain :ref:`ZZ`. However, if the ``modulus=p``
    option is given then the domain will be a finite field instead.

    >>> from sympy import Poly, Symbol
    >>> x = Symbol('x')
    >>> p = Poly(x**2 + 1)
    >>> p
    Poly(x**2 + 1, x, domain='ZZ')
    >>> p.domain
    ZZ
    >>> p2 = Poly(x**2 + 1, modulus=2)
    >>> p2
    Poly(x**2 + 1, x, modulus=2)
    >>> p2.domain
    GF(2)

    It is possible to factorise a polynomial over :ref:`GF(p)` using the
    modulus argument to :py:func:`~.factor` or by specifying the domain
    explicitly. The domain can also be given as a string.

    >>> from sympy import factor, GF
    >>> factor(x**2 + 1)
    x**2 + 1
    >>> factor(x**2 + 1, modulus=2)
    (x + 1)**2
    >>> factor(x**2 + 1, domain=GF(2))
    (x + 1)**2
    >>> factor(x**2 + 1, domain='GF(2)')
    (x + 1)**2

    It is also possible to use :ref:`GF(p)` with the :py:func:`~.cancel`
    and :py:func:`~.gcd` functions.

    >>> from sympy import cancel, gcd
    >>> cancel((x**2 + 1)/(x + 1))
    (x**2 + 1)/(x + 1)
    >>> cancel((x**2 + 1)/(x + 1), domain=GF(2))
    x + 1
    >>> gcd(x**2 + 1, x + 1)
    1
    >>> gcd(x**2 + 1, x + 1, domain=GF(2))
    x + 1

    When using the domain directly :ref:`GF(p)` can be used as a constructor
    to create instances which then support the operations ``+,-,*,**,/``

    >>> from sympy import GF
    >>> K = GF(5)
    >>> K
    GF(5)
    >>> x = K(3)
    >>> y = K(2)
    >>> x
    3 mod 5
    >>> y
    2 mod 5
    >>> x * y
    1 mod 5
    >>> x / y
    4 mod 5

    Notes
    =====

    It is also possible to create a :ref:`GF(p)` domain of **non-prime**
    order but the resulting ring is **not** a field: it is just the ring of
    the integers modulo ``n``.

    >>> K = GF(9)
    >>> z = K(3)
    >>> z
    3 mod 9
    >>> z**2
    0 mod 9

    It would be good to have a proper implementation of prime power fields
    (``GF(p**n)``) but these are not yet implemented in SymPY.

    .. _finite field: https://en.wikipedia.org/wiki/Finite_field
    """

    rep = 'FF'
    alias = 'FF'

    is_FiniteField = is_FF = True
    is_Numerical = True

    has_assoc_Ring = False
    has_assoc_Field = True

    dom = None
    mod = None

    def __init__(self, mod, symmetric=True):
        from sympy.polys.domains import ZZ
        dom = ZZ

        if mod <= 0:
            raise ValueError('modulus must be a positive integer, got %s' % mod)

        ctx, poly_ctx, is_flint = _modular_int_factory(mod, dom, symmetric, self)

        self.dtype = ctx
        self._poly_ctx = poly_ctx
        self._is_flint = is_flint

        self.zero = self.dtype(0)
        self.one = self.dtype(1)
        self.dom = dom
        self.mod = mod
        self.sym = symmetric
        self._tp = type(self.zero)

    @property
    def tp(self):
        return self._tp

    @property
    def is_Field(self):
        is_field = getattr(self, '_is_field', None)
        if is_field is None:
            from sympy.ntheory.primetest import isprime
            self._is_field = is_field = isprime(self.mod)
        return is_field

    def __str__(self):
        return 'GF(%s)' % self.mod

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.mod, self.dom))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, FiniteField) and \
            self.mod == other.mod and self.dom == other.dom

    def characteristic(self):
        """Return the characteristic of this domain. """
        return self.mod

    def get_field(self):
        """Returns a field associated with ``self``. """
        return self

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return SymPyInteger(self.to_int(a))

    def from_sympy(self, a):
        """Convert SymPy's Integer to SymPy's ``Integer``. """
        if a.is_Integer:
            return self.dtype(self.dom.dtype(int(a)))
        elif int_valued(a):
            return self.dtype(self.dom.dtype(int(a)))
        else:
            raise CoercionFailed("expected an integer, got %s" % a)

    def to_int(self, a):
        """Convert ``val`` to a Python ``int`` object. """
        aval = int(a)
        if self.sym and aval > self.mod // 2:
            aval -= self.mod
        return aval

    def is_positive(self, a):
        """Returns True if ``a`` is positive. """
        return bool(a)

    def is_nonnegative(self, a):
        """Returns True if ``a`` is non-negative. """
        return True

    def is_negative(self, a):
        """Returns True if ``a`` is negative. """
        return False

    def is_nonpositive(self, a):
        """Returns True if ``a`` is non-positive. """
        return not a

    def from_FF(K1, a, K0=None):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ(int(a), K0.dom))

    def from_FF_python(K1, a, K0=None):
        """Convert ``ModularInteger(int)`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_python(int(a), K0.dom))

    def from_ZZ(K1, a, K0=None):
        """Convert Python's ``int`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_python(a, K0))

    def from_ZZ_python(K1, a, K0=None):
        """Convert Python's ``int`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_python(a, K0))

    def from_QQ(K1, a, K0=None):
        """Convert Python's ``Fraction`` to ``dtype``. """
        if a.denominator == 1:
            return K1.from_ZZ_python(a.numerator)

    def from_QQ_python(K1, a, K0=None):
        """Convert Python's ``Fraction`` to ``dtype``. """
        if a.denominator == 1:
            return K1.from_ZZ_python(a.numerator)

    def from_FF_gmpy(K1, a, K0=None):
        """Convert ``ModularInteger(mpz)`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_gmpy(a.val, K0.dom))

    def from_ZZ_gmpy(K1, a, K0=None):
        """Convert GMPY's ``mpz`` to ``dtype``. """
        return K1.dtype(K1.dom.from_ZZ_gmpy(a, K0))

    def from_QQ_gmpy(K1, a, K0=None):
        """Convert GMPY's ``mpq`` to ``dtype``. """
        if a.denominator == 1:
            return K1.from_ZZ_gmpy(a.numerator)

    def from_RealField(K1, a, K0):
        """Convert mpmath's ``mpf`` to ``dtype``. """
        p, q = K0.to_rational(a)

        if q == 1:
            return K1.dtype(K1.dom.dtype(p))

    def is_square(self, a):
        """Returns True if ``a`` is a quadratic residue modulo p. """
        # a is not a square <=> x**2-a is irreducible
        poly = [int(x) for x in [self.one, self.zero, -a]]
        return not gf_irred_p_rabin(poly, self.mod, self.dom)

    def exsqrt(self, a):
        """Square root modulo p of ``a`` if it is a quadratic residue.

        Explanation
        ===========
        Always returns the square root that is no larger than ``p // 2``.
        """
        # x**2-a is not square-free if a=0 or the field is characteristic 2
        if self.mod == 2 or a == 0:
            return a
        # Otherwise, use square-free factorization routine to factorize x**2-a
        poly = [int(x) for x in [self.one, self.zero, -a]]
        for factor in gf_zassenhaus(poly, self.mod, self.dom):
            if len(factor) == 2 and factor[1] <= self.mod // 2:
                return self.dtype(factor[1])
        return None


FF = GF = FiniteField

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\config.py ===
from __future__ import annotations

import errno
import json
import os
import types
import typing as t

from werkzeug.utils import import_string

if t.TYPE_CHECKING:
    import typing_extensions as te

    from .sansio.app import App


T = t.TypeVar("T")


class ConfigAttribute(t.Generic[T]):
    """Makes an attribute forward to the config"""

    def __init__(
        self, name: str, get_converter: t.Callable[[t.Any], T] | None = None
    ) -> None:
        self.__name__ = name
        self.get_converter = get_converter

    @t.overload
    def __get__(self, obj: None, owner: None) -> te.Self: ...

    @t.overload
    def __get__(self, obj: App, owner: type[App]) -> T: ...

    def __get__(self, obj: App | None, owner: type[App] | None = None) -> T | te.Self:
        if obj is None:
            return self

        rv = obj.config[self.__name__]

        if self.get_converter is not None:
            rv = self.get_converter(rv)

        return rv  # type: ignore[no-any-return]

    def __set__(self, obj: App, value: t.Any) -> None:
        obj.config[self.__name__] = value


class Config(dict):  # type: ignore[type-arg]
    """Works exactly like a dict but provides ways to fill it from files
    or special dictionaries.  There are two common patterns to populate the
    config.

    Either you can fill the config from a config file::

        app.config.from_pyfile('yourconfig.cfg')

    Or alternatively you can define the configuration options in the
    module that calls :meth:`from_object` or provide an import path to
    a module that should be loaded.  It is also possible to tell it to
    use the same module and with that provide the configuration values
    just before the call::

        DEBUG = True
        SECRET_KEY = 'development key'
        app.config.from_object(__name__)

    In both cases (loading from any Python file or loading from modules),
    only uppercase keys are added to the config.  This makes it possible to use
    lowercase values in the config file for temporary values that are not added
    to the config or to define the config keys in the same file that implements
    the application.

    Probably the most interesting way to load configurations is from an
    environment variable pointing to a file::

        app.config.from_envvar('YOURAPPLICATION_SETTINGS')

    In this case before launching the application you have to set this
    environment variable to the file you want to use.  On Linux and OS X
    use the export statement::

        export YOURAPPLICATION_SETTINGS='/path/to/config/file'

    On windows use `set` instead.

    :param root_path: path to which files are read relative from.  When the
                      config object is created by the application, this is
                      the application's :attr:`~flask.Flask.root_path`.
    :param defaults: an optional dictionary of default values
    """

    def __init__(
        self,
        root_path: str | os.PathLike[str],
        defaults: dict[str, t.Any] | None = None,
    ) -> None:
        super().__init__(defaults or {})
        self.root_path = root_path

    def from_envvar(self, variable_name: str, silent: bool = False) -> bool:
        """Loads a configuration from an environment variable pointing to
        a configuration file.  This is basically just a shortcut with nicer
        error messages for this line of code::

            app.config.from_pyfile(os.environ['YOURAPPLICATION_SETTINGS'])

        :param variable_name: name of the environment variable
        :param silent: set to ``True`` if you want silent failure for missing
                       files.
        :return: ``True`` if the file was loaded successfully.
        """
        rv = os.environ.get(variable_name)
        if not rv:
            if silent:
                return False
            raise RuntimeError(
                f"The environment variable {variable_name!r} is not set"
                " and as such configuration could not be loaded. Set"
                " this variable and make it point to a configuration"
                " file"
            )
        return self.from_pyfile(rv, silent=silent)

    def from_prefixed_env(
        self, prefix: str = "FLASK", *, loads: t.Callable[[str], t.Any] = json.loads
    ) -> bool:
        """Load any environment variables that start with ``FLASK_``,
        dropping the prefix from the env key for the config key. Values
        are passed through a loading function to attempt to convert them
        to more specific types than strings.

        Keys are loaded in :func:`sorted` order.

        The default loading function attempts to parse values as any
        valid JSON type, including dicts and lists.

        Specific items in nested dicts can be set by separating the
        keys with double underscores (``__``). If an intermediate key
        doesn't exist, it will be initialized to an empty dict.

        :param prefix: Load env vars that start with this prefix,
            separated with an underscore (``_``).
        :param loads: Pass each string value to this function and use
            the returned value as the config value. If any error is
            raised it is ignored and the value remains a string. The
            default is :func:`json.loads`.

        .. versionadded:: 2.1
        """
        prefix = f"{prefix}_"

        for key in sorted(os.environ):
            if not key.startswith(prefix):
                continue

            value = os.environ[key]
            key = key.removeprefix(prefix)

            try:
                value = loads(value)
            except Exception:
                # Keep the value as a string if loading failed.
                pass

            if "__" not in key:
                # A non-nested key, set directly.
                self[key] = value
                continue

            # Traverse nested dictionaries with keys separated by "__".
            current = self
            *parts, tail = key.split("__")

            for part in parts:
                # If an intermediate dict does not exist, create it.
                if part not in current:
                    current[part] = {}

                current = current[part]

            current[tail] = value

        return True

    def from_pyfile(
        self, filename: str | os.PathLike[str], silent: bool = False
    ) -> bool:
        """Updates the values in the config from a Python file.  This function
        behaves as if the file was imported as module with the
        :meth:`from_object` function.

        :param filename: the filename of the config.  This can either be an
                         absolute filename or a filename relative to the
                         root path.
        :param silent: set to ``True`` if you want silent failure for missing
                       files.
        :return: ``True`` if the file was loaded successfully.

        .. versionadded:: 0.7
           `silent` parameter.
        """
        filename = os.path.join(self.root_path, filename)
        d = types.ModuleType("config")
        d.__file__ = filename
        try:
            with open(filename, mode="rb") as config_file:
                exec(compile(config_file.read(), filename, "exec"), d.__dict__)
        except OSError as e:
            if silent and e.errno in (errno.ENOENT, errno.EISDIR, errno.ENOTDIR):
                return False
            e.strerror = f"Unable to load configuration file ({e.strerror})"
            raise
        self.from_object(d)
        return True

    def from_object(self, obj: object | str) -> None:
        """Updates the values from the given object.  An object can be of one
        of the following two types:

        -   a string: in this case the object with that name will be imported
        -   an actual object reference: that object is used directly

        Objects are usually either modules or classes. :meth:`from_object`
        loads only the uppercase attributes of the module/class. A ``dict``
        object will not work with :meth:`from_object` because the keys of a
        ``dict`` are not attributes of the ``dict`` class.

        Example of module-based configuration::

            app.config.from_object('yourapplication.default_config')
            from yourapplication import default_config
            app.config.from_object(default_config)

        Nothing is done to the object before loading. If the object is a
        class and has ``@property`` attributes, it needs to be
        instantiated before being passed to this method.

        You should not use this function to load the actual configuration but
        rather configuration defaults.  The actual config should be loaded
        with :meth:`from_pyfile` and ideally from a location not within the
        package because the package might be installed system wide.

        See :ref:`config-dev-prod` for an example of class-based configuration
        using :meth:`from_object`.

        :param obj: an import name or object
        """
        if isinstance(obj, str):
            obj = import_string(obj)
        for key in dir(obj):
            if key.isupper():
                self[key] = getattr(obj, key)

    def from_file(
        self,
        filename: str | os.PathLike[str],
        load: t.Callable[[t.IO[t.Any]], t.Mapping[str, t.Any]],
        silent: bool = False,
        text: bool = True,
    ) -> bool:
        """Update the values in the config from a file that is loaded
        using the ``load`` parameter. The loaded data is passed to the
        :meth:`from_mapping` method.

        .. code-block:: python

            import json
            app.config.from_file("config.json", load=json.load)

            import tomllib
            app.config.from_file("config.toml", load=tomllib.load, text=False)

        :param filename: The path to the data file. This can be an
            absolute path or relative to the config root path.
        :param load: A callable that takes a file handle and returns a
            mapping of loaded data from the file.
        :type load: ``Callable[[Reader], Mapping]`` where ``Reader``
            implements a ``read`` method.
        :param silent: Ignore the file if it doesn't exist.
        :param text: Open the file in text or binary mode.
        :return: ``True`` if the file was loaded successfully.

        .. versionchanged:: 2.3
            The ``text`` parameter was added.

        .. versionadded:: 2.0
        """
        filename = os.path.join(self.root_path, filename)

        try:
            with open(filename, "r" if text else "rb") as f:
                obj = load(f)
        except OSError as e:
            if silent and e.errno in (errno.ENOENT, errno.EISDIR):
                return False

            e.strerror = f"Unable to load configuration file ({e.strerror})"
            raise

        return self.from_mapping(obj)

    def from_mapping(
        self, mapping: t.Mapping[str, t.Any] | None = None, **kwargs: t.Any
    ) -> bool:
        """Updates the config like :meth:`update` ignoring items with
        non-upper keys.

        :return: Always returns ``True``.

        .. versionadded:: 0.11
        """
        mappings: dict[str, t.Any] = {}
        if mapping is not None:
            mappings.update(mapping)
        mappings.update(kwargs)
        for key, value in mappings.items():
            if key.isupper():
                self[key] = value
        return True

    def get_namespace(
        self, namespace: str, lowercase: bool = True, trim_namespace: bool = True
    ) -> dict[str, t.Any]:
        """Returns a dictionary containing a subset of configuration options
        that match the specified namespace/prefix. Example usage::

            app.config['IMAGE_STORE_TYPE'] = 'fs'
            app.config['IMAGE_STORE_PATH'] = '/var/app/images'
            app.config['IMAGE_STORE_BASE_URL'] = 'http://img.website.com'
            image_store_config = app.config.get_namespace('IMAGE_STORE_')

        The resulting dictionary `image_store_config` would look like::

            {
                'type': 'fs',
                'path': '/var/app/images',
                'base_url': 'http://img.website.com'
            }

        This is often useful when configuration options map directly to
        keyword arguments in functions or class constructors.

        :param namespace: a configuration namespace
        :param lowercase: a flag indicating if the keys of the resulting
                          dictionary should be lowercase
        :param trim_namespace: a flag indicating if the keys of the resulting
                          dictionary should not include the namespace

        .. versionadded:: 0.11
        """
        rv = {}
        for k, v in self.items():
            if not k.startswith(namespace):
                continue
            if trim_namespace:
                key = k[len(namespace) :]
            else:
                key = k
            if lowercase:
                key = key.lower()
            rv[key] = v
        return rv

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {dict.__repr__(self)}>"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\json.py ===
# dialects/postgresql/json.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import Any
from typing import Callable
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union

from .array import ARRAY
from .array import array as _pg_array
from .operators import ASTEXT
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import DELETE_PATH
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from .operators import JSONPATH_ASTEXT
from .operators import PATH_EXISTS
from .operators import PATH_MATCH
from ... import types as sqltypes
from ...sql import cast
from ...sql._typing import _T

if TYPE_CHECKING:
    from ...engine.interfaces import Dialect
    from ...sql.elements import ColumnElement
    from ...sql.type_api import _BindProcessorType
    from ...sql.type_api import _LiteralProcessorType
    from ...sql.type_api import TypeEngine

__all__ = ("JSON", "JSONB")


class JSONPathType(sqltypes.JSON.JSONPathType):
    def _processor(
        self, dialect: Dialect, super_proc: Optional[Callable[[Any], Any]]
    ) -> Callable[[Any], Any]:
        def process(value: Any) -> Any:
            if isinstance(value, str):
                # If it's already a string assume that it's in json path
                # format. This allows using cast with json paths literals
                return value
            elif value:
                # If it's already a string assume that it's in json path
                # format. This allows using cast with json paths literals
                value = "{%s}" % (", ".join(map(str, value)))
            else:
                value = "{}"
            if super_proc:
                value = super_proc(value)
            return value

        return process

    def bind_processor(self, dialect: Dialect) -> _BindProcessorType[Any]:
        return self._processor(dialect, self.string_bind_processor(dialect))  # type: ignore[return-value]  # noqa: E501

    def literal_processor(
        self, dialect: Dialect
    ) -> _LiteralProcessorType[Any]:
        return self._processor(dialect, self.string_literal_processor(dialect))  # type: ignore[return-value]  # noqa: E501


class JSONPATH(JSONPathType):
    """JSON Path Type.

    This is usually required to cast literal values to json path when using
    json search like function, such as ``jsonb_path_query_array`` or
    ``jsonb_path_exists``::

        stmt = sa.select(
            sa.func.jsonb_path_query_array(
                table.c.jsonb_col, cast("$.address.id", JSONPATH)
            )
        )

    """

    __visit_name__ = "JSONPATH"


class JSON(sqltypes.JSON):
    """Represent the PostgreSQL JSON type.

    :class:`_postgresql.JSON` is used automatically whenever the base
    :class:`_types.JSON` datatype is used against a PostgreSQL backend,
    however base :class:`_types.JSON` datatype does not provide Python
    accessors for PostgreSQL-specific comparison methods such as
    :meth:`_postgresql.JSON.Comparator.astext`; additionally, to use
    PostgreSQL ``JSONB``, the :class:`_postgresql.JSONB` datatype should
    be used explicitly.

    .. seealso::

        :class:`_types.JSON` - main documentation for the generic
        cross-platform JSON datatype.

    The operators provided by the PostgreSQL version of :class:`_types.JSON`
    include:

    * Index operations (the ``->`` operator)::

        data_table.c.data["some key"]

        data_table.c.data[5]

    * Index operations returning text
      (the ``->>`` operator)::

        data_table.c.data["some key"].astext == "some value"

      Note that equivalent functionality is available via the
      :attr:`.JSON.Comparator.as_string` accessor.

    * Index operations with CAST
      (equivalent to ``CAST(col ->> ['some key'] AS <type>)``)::

        data_table.c.data["some key"].astext.cast(Integer) == 5

      Note that equivalent functionality is available via the
      :attr:`.JSON.Comparator.as_integer` and similar accessors.

    * Path index operations (the ``#>`` operator)::

        data_table.c.data[("key_1", "key_2", 5, ..., "key_n")]

    * Path index operations returning text (the ``#>>`` operator)::

        data_table.c.data[
            ("key_1", "key_2", 5, ..., "key_n")
        ].astext == "some value"

    Index operations return an expression object whose type defaults to
    :class:`_types.JSON` by default,
    so that further JSON-oriented instructions
    may be called upon the result type.

    Custom serializers and deserializers are specified at the dialect level,
    that is using :func:`_sa.create_engine`.  The reason for this is that when
    using psycopg2, the DBAPI only allows serializers at the per-cursor
    or per-connection level.   E.g.::

        engine = create_engine(
            "postgresql+psycopg2://scott:tiger@localhost/test",
            json_serializer=my_serialize_fn,
            json_deserializer=my_deserialize_fn,
        )

    When using the psycopg2 dialect, the json_deserializer is registered
    against the database using ``psycopg2.extras.register_default_json``.

    .. seealso::

        :class:`_types.JSON` - Core level JSON type

        :class:`_postgresql.JSONB`

    """  # noqa

    render_bind_cast = True
    astext_type: TypeEngine[str] = sqltypes.Text()

    def __init__(
        self,
        none_as_null: bool = False,
        astext_type: Optional[TypeEngine[str]] = None,
    ):
        """Construct a :class:`_types.JSON` type.

        :param none_as_null: if True, persist the value ``None`` as a
         SQL NULL value, not the JSON encoding of ``null``.   Note that
         when this flag is False, the :func:`.null` construct can still
         be used to persist a NULL value::

             from sqlalchemy import null

             conn.execute(table.insert(), {"data": null()})

         .. seealso::

              :attr:`_types.JSON.NULL`

        :param astext_type: the type to use for the
         :attr:`.JSON.Comparator.astext`
         accessor on indexed attributes.  Defaults to :class:`_types.Text`.

        """
        super().__init__(none_as_null=none_as_null)
        if astext_type is not None:
            self.astext_type = astext_type

    class Comparator(sqltypes.JSON.Comparator[_T]):
        """Define comparison operations for :class:`_types.JSON`."""

        type: JSON

        @property
        def astext(self) -> ColumnElement[str]:
            """On an indexed expression, use the "astext" (e.g. "->>")
            conversion when rendered in SQL.

            E.g.::

                select(data_table.c.data["some key"].astext)

            .. seealso::

                :meth:`_expression.ColumnElement.cast`

            """
            if isinstance(self.expr.right.type, sqltypes.JSON.JSONPathType):
                return self.expr.left.operate(  # type: ignore[no-any-return]
                    JSONPATH_ASTEXT,
                    self.expr.right,
                    result_type=self.type.astext_type,
                )
            else:
                return self.expr.left.operate(  # type: ignore[no-any-return]
                    ASTEXT, self.expr.right, result_type=self.type.astext_type
                )

    comparator_factory = Comparator


class JSONB(JSON):
    """Represent the PostgreSQL JSONB type.

    The :class:`_postgresql.JSONB` type stores arbitrary JSONB format data,
    e.g.::

        data_table = Table(
            "data_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", JSONB),
        )

        with engine.connect() as conn:
            conn.execute(
                data_table.insert(), data={"key1": "value1", "key2": "value2"}
            )

    The :class:`_postgresql.JSONB` type includes all operations provided by
    :class:`_types.JSON`, including the same behaviors for indexing
    operations.
    It also adds additional operators specific to JSONB, including
    :meth:`.JSONB.Comparator.has_key`, :meth:`.JSONB.Comparator.has_all`,
    :meth:`.JSONB.Comparator.has_any`, :meth:`.JSONB.Comparator.contains`,
    :meth:`.JSONB.Comparator.contained_by`,
    :meth:`.JSONB.Comparator.delete_path`,
    :meth:`.JSONB.Comparator.path_exists` and
    :meth:`.JSONB.Comparator.path_match`.

    Like the :class:`_types.JSON` type, the :class:`_postgresql.JSONB`
    type does not detect
    in-place changes when used with the ORM, unless the
    :mod:`sqlalchemy.ext.mutable` extension is used.

    Custom serializers and deserializers
    are shared with the :class:`_types.JSON` class,
    using the ``json_serializer``
    and ``json_deserializer`` keyword arguments.  These must be specified
    at the dialect level using :func:`_sa.create_engine`.  When using
    psycopg2, the serializers are associated with the jsonb type using
    ``psycopg2.extras.register_default_jsonb`` on a per-connection basis,
    in the same way that ``psycopg2.extras.register_default_json`` is used
    to register these handlers with the json type.

    .. seealso::

        :class:`_types.JSON`

    """

    __visit_name__ = "JSONB"

    class Comparator(JSON.Comparator[_T]):
        """Define comparison operations for :class:`_types.JSON`."""

        type: JSONB

        def has_key(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression.  Test for presence of a key (equivalent of
            the ``?`` operator).  Note that the key may be a SQLA expression.
            """
            return self.operate(HAS_KEY, other, result_type=sqltypes.Boolean)

        def has_all(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression.  Test for presence of all keys in jsonb
            (equivalent of the ``?&`` operator)
            """
            return self.operate(HAS_ALL, other, result_type=sqltypes.Boolean)

        def has_any(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression.  Test for presence of any key in jsonb
            (equivalent of the ``?|`` operator)
            """
            return self.operate(HAS_ANY, other, result_type=sqltypes.Boolean)

        def contains(self, other: Any, **kwargs: Any) -> ColumnElement[bool]:
            """Boolean expression.  Test if keys (or array) are a superset
            of/contained the keys of the argument jsonb expression
            (equivalent of the ``@>`` operator).

            kwargs may be ignored by this operator but are required for API
            conformance.
            """
            return self.operate(CONTAINS, other, result_type=sqltypes.Boolean)

        def contained_by(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression.  Test if keys are a proper subset of the
            keys of the argument jsonb expression
            (equivalent of the ``<@`` operator).
            """
            return self.operate(
                CONTAINED_BY, other, result_type=sqltypes.Boolean
            )

        def delete_path(
            self, array: Union[List[str], _pg_array[str]]
        ) -> ColumnElement[JSONB]:
            """JSONB expression. Deletes field or array element specified in
            the argument array (equivalent of the ``#-`` operator).

            The input may be a list of strings that will be coerced to an
            ``ARRAY`` or an instance of :meth:`_postgres.array`.

            .. versionadded:: 2.0
            """
            if not isinstance(array, _pg_array):
                array = _pg_array(array)
            right_side = cast(array, ARRAY(sqltypes.TEXT))
            return self.operate(DELETE_PATH, right_side, result_type=JSONB)

        def path_exists(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression. Test for presence of item given by the
            argument JSONPath expression (equivalent of the ``@?`` operator).

            .. versionadded:: 2.0
            """
            return self.operate(
                PATH_EXISTS, other, result_type=sqltypes.Boolean
            )

        def path_match(self, other: Any) -> ColumnElement[bool]:
            """Boolean expression. Test if JSONPath predicate given by the
            argument JSONPath expression matches
            (equivalent of the ``@@`` operator).

            Only the first item of the result is taken into account.

            .. versionadded:: 2.0
            """
            return self.operate(
                PATH_MATCH, other, result_type=sqltypes.Boolean
            )

    comparator_factory = Comparator

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\_dtype.py ===
"""
A place for code to be called from the implementation of np.dtype

String handling is much easier to do correctly in python.
"""
import numpy as np

_kind_to_stem = {
    'u': 'uint',
    'i': 'int',
    'c': 'complex',
    'f': 'float',
    'b': 'bool',
    'V': 'void',
    'O': 'object',
    'M': 'datetime',
    'm': 'timedelta',
    'S': 'bytes',
    'U': 'str',
}


def _kind_name(dtype):
    try:
        return _kind_to_stem[dtype.kind]
    except KeyError as e:
        raise RuntimeError(
            f"internal dtype error, unknown kind {dtype.kind!r}"
        ) from None


def __str__(dtype):
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=True)
    elif dtype.subdtype:
        return _subarray_str(dtype)
    elif issubclass(dtype.type, np.flexible) or not dtype.isnative:
        return dtype.str
    else:
        return dtype.name


def __repr__(dtype):
    arg_str = _construction_repr(dtype, include_align=False)
    if dtype.isalignedstruct:
        arg_str = arg_str + ", align=True"
    return f"dtype({arg_str})"


def _unpack_field(dtype, offset, title=None):
    """
    Helper function to normalize the items in dtype.fields.

    Call as:

    dtype, offset, title = _unpack_field(*dtype.fields[name])
    """
    return dtype, offset, title


def _isunsized(dtype):
    # PyDataType_ISUNSIZED
    return dtype.itemsize == 0


def _construction_repr(dtype, include_align=False, short=False):
    """
    Creates a string repr of the dtype, excluding the 'dtype()' part
    surrounding the object. This object may be a string, a list, or
    a dict depending on the nature of the dtype. This
    is the object passed as the first parameter to the dtype
    constructor, and if no additional constructor parameters are
    given, will reproduce the exact memory layout.

    Parameters
    ----------
    short : bool
        If true, this creates a shorter repr using 'kind' and 'itemsize',
        instead of the longer type name.

    include_align : bool
        If true, this includes the 'align=True' parameter
        inside the struct dtype construction dict when needed. Use this flag
        if you want a proper repr string without the 'dtype()' part around it.

        If false, this does not preserve the
        'align=True' parameter or sticky NPY_ALIGNED_STRUCT flag for
        struct arrays like the regular repr does, because the 'align'
        flag is not part of first dtype constructor parameter. This
        mode is intended for a full 'repr', where the 'align=True' is
        provided as the second parameter.
    """
    if dtype.fields is not None:
        return _struct_str(dtype, include_align=include_align)
    elif dtype.subdtype:
        return _subarray_str(dtype)
    else:
        return _scalar_str(dtype, short=short)


def _scalar_str(dtype, short):
    byteorder = _byte_order_str(dtype)

    if dtype.type == np.bool:
        if short:
            return "'?'"
        else:
            return "'bool'"

    elif dtype.type == np.object_:
        # The object reference may be different sizes on different
        # platforms, so it should never include the itemsize here.
        return "'O'"

    elif dtype.type == np.bytes_:
        if _isunsized(dtype):
            return "'S'"
        else:
            return "'S%d'" % dtype.itemsize

    elif dtype.type == np.str_:
        if _isunsized(dtype):
            return f"'{byteorder}U'"
        else:
            return "'%sU%d'" % (byteorder, dtype.itemsize / 4)

    elif dtype.type == str:
        return "'T'"

    elif not type(dtype)._legacy:
        return f"'{byteorder}{type(dtype).__name__}{dtype.itemsize * 8}'"

    # unlike the other types, subclasses of void are preserved - but
    # historically the repr does not actually reveal the subclass
    elif issubclass(dtype.type, np.void):
        if _isunsized(dtype):
            return "'V'"
        else:
            return "'V%d'" % dtype.itemsize

    elif dtype.type == np.datetime64:
        return f"'{byteorder}M8{_datetime_metadata_str(dtype)}'"

    elif dtype.type == np.timedelta64:
        return f"'{byteorder}m8{_datetime_metadata_str(dtype)}'"

    elif dtype.isbuiltin == 2:
        return dtype.type.__name__

    elif np.issubdtype(dtype, np.number):
        # Short repr with endianness, like '<f8'
        if short or dtype.byteorder not in ('=', '|'):
            return "'%s%c%d'" % (byteorder, dtype.kind, dtype.itemsize)

        # Longer repr, like 'float64'
        else:
            return "'%s%d'" % (_kind_name(dtype), 8 * dtype.itemsize)

    else:
        raise RuntimeError(
            "Internal error: NumPy dtype unrecognized type number")


def _byte_order_str(dtype):
    """ Normalize byteorder to '<' or '>' """
    # hack to obtain the native and swapped byte order characters
    swapped = np.dtype(int).newbyteorder('S')
    native = swapped.newbyteorder('S')

    byteorder = dtype.byteorder
    if byteorder == '=':
        return native.byteorder
    if byteorder == 'S':
        # TODO: this path can never be reached
        return swapped.byteorder
    elif byteorder == '|':
        return ''
    else:
        return byteorder


def _datetime_metadata_str(dtype):
    # TODO: this duplicates the C metastr_to_unicode functionality
    unit, count = np.datetime_data(dtype)
    if unit == 'generic':
        return ''
    elif count == 1:
        return f'[{unit}]'
    else:
        return f'[{count}{unit}]'


def _struct_dict_str(dtype, includealignedflag):
    # unpack the fields dictionary into ls
    names = dtype.names
    fld_dtypes = []
    offsets = []
    titles = []
    for name in names:
        fld_dtype, offset, title = _unpack_field(*dtype.fields[name])
        fld_dtypes.append(fld_dtype)
        offsets.append(offset)
        titles.append(title)

    # Build up a string to make the dictionary

    if np._core.arrayprint._get_legacy_print_mode() <= 121:
        colon = ":"
        fieldsep = ","
    else:
        colon = ": "
        fieldsep = ", "

    # First, the names
    ret = "{'names'%s[" % colon
    ret += fieldsep.join(repr(name) for name in names)

    # Second, the formats
    ret += f"], 'formats'{colon}["
    ret += fieldsep.join(
        _construction_repr(fld_dtype, short=True) for fld_dtype in fld_dtypes)

    # Third, the offsets
    ret += f"], 'offsets'{colon}["
    ret += fieldsep.join("%d" % offset for offset in offsets)

    # Fourth, the titles
    if any(title is not None for title in titles):
        ret += f"], 'titles'{colon}["
        ret += fieldsep.join(repr(title) for title in titles)

    # Fifth, the itemsize
    ret += "], 'itemsize'%s%d" % (colon, dtype.itemsize)

    if (includealignedflag and dtype.isalignedstruct):
        # Finally, the aligned flag
        ret += ", 'aligned'%sTrue}" % colon
    else:
        ret += "}"

    return ret


def _aligned_offset(offset, alignment):
    # round up offset:
    return - (-offset // alignment) * alignment


def _is_packed(dtype):
    """
    Checks whether the structured data type in 'dtype'
    has a simple layout, where all the fields are in order,
    and follow each other with no alignment padding.

    When this returns true, the dtype can be reconstructed
    from a list of the field names and dtypes with no additional
    dtype parameters.

    Duplicates the C `is_dtype_struct_simple_unaligned_layout` function.
    """
    align = dtype.isalignedstruct
    max_alignment = 1
    total_offset = 0
    for name in dtype.names:
        fld_dtype, fld_offset, title = _unpack_field(*dtype.fields[name])

        if align:
            total_offset = _aligned_offset(total_offset, fld_dtype.alignment)
            max_alignment = max(max_alignment, fld_dtype.alignment)

        if fld_offset != total_offset:
            return False
        total_offset += fld_dtype.itemsize

    if align:
        total_offset = _aligned_offset(total_offset, max_alignment)

    return total_offset == dtype.itemsize


def _struct_list_str(dtype):
    items = []
    for name in dtype.names:
        fld_dtype, fld_offset, title = _unpack_field(*dtype.fields[name])

        item = "("
        if title is not None:
            item += f"({title!r}, {name!r}), "
        else:
            item += f"{name!r}, "
        # Special case subarray handling here
        if fld_dtype.subdtype is not None:
            base, shape = fld_dtype.subdtype
            item += f"{_construction_repr(base, short=True)}, {shape}"
        else:
            item += _construction_repr(fld_dtype, short=True)

        item += ")"
        items.append(item)

    return "[" + ", ".join(items) + "]"


def _struct_str(dtype, include_align):
    # The list str representation can't include the 'align=' flag,
    # so if it is requested and the struct has the aligned flag set,
    # we must use the dict str instead.
    if not (include_align and dtype.isalignedstruct) and _is_packed(dtype):
        sub = _struct_list_str(dtype)

    else:
        sub = _struct_dict_str(dtype, include_align)

    # If the data type isn't the default, void, show it
    if dtype.type != np.void:
        return f"({dtype.type.__module__}.{dtype.type.__name__}, {sub})"
    else:
        return sub


def _subarray_str(dtype):
    base, shape = dtype.subdtype
    return f"({_construction_repr(base, short=True)}, {shape})"


def _name_includes_bit_suffix(dtype):
    if dtype.type == np.object_:
        # pointer size varies by system, best to omit it
        return False
    elif dtype.type == np.bool:
        # implied
        return False
    elif dtype.type is None:
        return True
    elif np.issubdtype(dtype, np.flexible) and _isunsized(dtype):
        # unspecified
        return False
    else:
        return True


def _name_get(dtype):
    # provides dtype.name.__get__, documented as returning a "bit name"

    if dtype.isbuiltin == 2:
        # user dtypes don't promise to do anything special
        return dtype.type.__name__

    if not type(dtype)._legacy:
        name = type(dtype).__name__

    elif issubclass(dtype.type, np.void):
        # historically, void subclasses preserve their name, eg `record64`
        name = dtype.type.__name__
    else:
        name = _kind_name(dtype)

    # append bit counts
    if _name_includes_bit_suffix(dtype):
        name += f"{dtype.itemsize * 8}"

    # append metadata to datetimes
    if dtype.type in (np.datetime64, np.timedelta64):
        name += _datetime_metadata_str(dtype)

    return name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\system_info.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: SystemInfo (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class GPUDevice:
    '''
    Describes a single graphics processor (GPU).
    '''
    #: PCI ID of the GPU vendor, if available; 0 otherwise.
    vendor_id: float

    #: PCI ID of the GPU device, if available; 0 otherwise.
    device_id: float

    #: String description of the GPU vendor, if the PCI ID is not available.
    vendor_string: str

    #: String description of the GPU device, if the PCI ID is not available.
    device_string: str

    #: String description of the GPU driver vendor.
    driver_vendor: str

    #: String description of the GPU driver version.
    driver_version: str

    #: Sub sys ID of the GPU, only available on Windows.
    sub_sys_id: typing.Optional[float] = None

    #: Revision of the GPU, only available on Windows.
    revision: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['vendorId'] = self.vendor_id
        json['deviceId'] = self.device_id
        json['vendorString'] = self.vendor_string
        json['deviceString'] = self.device_string
        json['driverVendor'] = self.driver_vendor
        json['driverVersion'] = self.driver_version
        if self.sub_sys_id is not None:
            json['subSysId'] = self.sub_sys_id
        if self.revision is not None:
            json['revision'] = self.revision
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            vendor_id=float(json['vendorId']),
            device_id=float(json['deviceId']),
            vendor_string=str(json['vendorString']),
            device_string=str(json['deviceString']),
            driver_vendor=str(json['driverVendor']),
            driver_version=str(json['driverVersion']),
            sub_sys_id=float(json['subSysId']) if 'subSysId' in json else None,
            revision=float(json['revision']) if 'revision' in json else None,
        )


@dataclass
class Size:
    '''
    Describes the width and height dimensions of an entity.
    '''
    #: Width in pixels.
    width: int

    #: Height in pixels.
    height: int

    def to_json(self):
        json = dict()
        json['width'] = self.width
        json['height'] = self.height
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            width=int(json['width']),
            height=int(json['height']),
        )


@dataclass
class VideoDecodeAcceleratorCapability:
    '''
    Describes a supported video decoding profile with its associated minimum and
    maximum resolutions.
    '''
    #: Video codec profile that is supported, e.g. VP9 Profile 2.
    profile: str

    #: Maximum video dimensions in pixels supported for this ``profile``.
    max_resolution: Size

    #: Minimum video dimensions in pixels supported for this ``profile``.
    min_resolution: Size

    def to_json(self):
        json = dict()
        json['profile'] = self.profile
        json['maxResolution'] = self.max_resolution.to_json()
        json['minResolution'] = self.min_resolution.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            profile=str(json['profile']),
            max_resolution=Size.from_json(json['maxResolution']),
            min_resolution=Size.from_json(json['minResolution']),
        )


@dataclass
class VideoEncodeAcceleratorCapability:
    '''
    Describes a supported video encoding profile with its associated maximum
    resolution and maximum framerate.
    '''
    #: Video codec profile that is supported, e.g H264 Main.
    profile: str

    #: Maximum video dimensions in pixels supported for this ``profile``.
    max_resolution: Size

    #: Maximum encoding framerate in frames per second supported for this
    #: ``profile``, as fraction's numerator and denominator, e.g. 24/1 fps,
    #: 24000/1001 fps, etc.
    max_framerate_numerator: int

    max_framerate_denominator: int

    def to_json(self):
        json = dict()
        json['profile'] = self.profile
        json['maxResolution'] = self.max_resolution.to_json()
        json['maxFramerateNumerator'] = self.max_framerate_numerator
        json['maxFramerateDenominator'] = self.max_framerate_denominator
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            profile=str(json['profile']),
            max_resolution=Size.from_json(json['maxResolution']),
            max_framerate_numerator=int(json['maxFramerateNumerator']),
            max_framerate_denominator=int(json['maxFramerateDenominator']),
        )


class SubsamplingFormat(enum.Enum):
    '''
    YUV subsampling type of the pixels of a given image.
    '''
    YUV420 = "yuv420"
    YUV422 = "yuv422"
    YUV444 = "yuv444"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ImageType(enum.Enum):
    '''
    Image format of a given image.
    '''
    JPEG = "jpeg"
    WEBP = "webp"
    UNKNOWN = "unknown"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ImageDecodeAcceleratorCapability:
    '''
    Describes a supported image decoding profile with its associated minimum and
    maximum resolutions and subsampling.
    '''
    #: Image coded, e.g. Jpeg.
    image_type: ImageType

    #: Maximum supported dimensions of the image in pixels.
    max_dimensions: Size

    #: Minimum supported dimensions of the image in pixels.
    min_dimensions: Size

    #: Optional array of supported subsampling formats, e.g. 4:2:0, if known.
    subsamplings: typing.List[SubsamplingFormat]

    def to_json(self):
        json = dict()
        json['imageType'] = self.image_type.to_json()
        json['maxDimensions'] = self.max_dimensions.to_json()
        json['minDimensions'] = self.min_dimensions.to_json()
        json['subsamplings'] = [i.to_json() for i in self.subsamplings]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            image_type=ImageType.from_json(json['imageType']),
            max_dimensions=Size.from_json(json['maxDimensions']),
            min_dimensions=Size.from_json(json['minDimensions']),
            subsamplings=[SubsamplingFormat.from_json(i) for i in json['subsamplings']],
        )


@dataclass
class GPUInfo:
    '''
    Provides information about the GPU(s) on the system.
    '''
    #: The graphics devices on the system. Element 0 is the primary GPU.
    devices: typing.List[GPUDevice]

    #: An optional array of GPU driver bug workarounds.
    driver_bug_workarounds: typing.List[str]

    #: Supported accelerated video decoding capabilities.
    video_decoding: typing.List[VideoDecodeAcceleratorCapability]

    #: Supported accelerated video encoding capabilities.
    video_encoding: typing.List[VideoEncodeAcceleratorCapability]

    #: Supported accelerated image decoding capabilities.
    image_decoding: typing.List[ImageDecodeAcceleratorCapability]

    #: An optional dictionary of additional GPU related attributes.
    aux_attributes: typing.Optional[dict] = None

    #: An optional dictionary of graphics features and their status.
    feature_status: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['devices'] = [i.to_json() for i in self.devices]
        json['driverBugWorkarounds'] = [i for i in self.driver_bug_workarounds]
        json['videoDecoding'] = [i.to_json() for i in self.video_decoding]
        json['videoEncoding'] = [i.to_json() for i in self.video_encoding]
        json['imageDecoding'] = [i.to_json() for i in self.image_decoding]
        if self.aux_attributes is not None:
            json['auxAttributes'] = self.aux_attributes
        if self.feature_status is not None:
            json['featureStatus'] = self.feature_status
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            devices=[GPUDevice.from_json(i) for i in json['devices']],
            driver_bug_workarounds=[str(i) for i in json['driverBugWorkarounds']],
            video_decoding=[VideoDecodeAcceleratorCapability.from_json(i) for i in json['videoDecoding']],
            video_encoding=[VideoEncodeAcceleratorCapability.from_json(i) for i in json['videoEncoding']],
            image_decoding=[ImageDecodeAcceleratorCapability.from_json(i) for i in json['imageDecoding']],
            aux_attributes=dict(json['auxAttributes']) if 'auxAttributes' in json else None,
            feature_status=dict(json['featureStatus']) if 'featureStatus' in json else None,
        )


@dataclass
class ProcessInfo:
    '''
    Represents process info.
    '''
    #: Specifies process type.
    type_: str

    #: Specifies process id.
    id_: int

    #: Specifies cumulative CPU usage in seconds across all threads of the
    #: process since the process start.
    cpu_time: float

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['id'] = self.id_
        json['cpuTime'] = self.cpu_time
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            id_=int(json['id']),
            cpu_time=float(json['cpuTime']),
        )


def get_info() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[GPUInfo, str, str, str]]:
    '''
    Returns information about the system.

    :returns: A tuple with the following items:

        0. **gpu** - Information about the GPUs on the system.
        1. **modelName** - A platform-dependent description of the model of the machine. On Mac OS, this is, for example, 'MacBookPro'. Will be the empty string if not supported.
        2. **modelVersion** - A platform-dependent description of the version of the machine. On Mac OS, this is, for example, '10.1'. Will be the empty string if not supported.
        3. **commandLine** - The command line string used to launch the browser. Will be the empty string if not supported.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'SystemInfo.getInfo',
    }
    json = yield cmd_dict
    return (
        GPUInfo.from_json(json['gpu']),
        str(json['modelName']),
        str(json['modelVersion']),
        str(json['commandLine'])
    )


def get_feature_state(
        feature_state: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Returns information about the feature state.

    :param feature_state:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['featureState'] = feature_state
    cmd_dict: T_JSON_DICT = {
        'method': 'SystemInfo.getFeatureState',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['featureEnabled'])


def get_process_info() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[ProcessInfo]]:
    '''
    Returns information about all running processes.

    :returns: An array of process info blocks.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'SystemInfo.getProcessInfo',
    }
    json = yield cmd_dict
    return [ProcessInfo.from_json(i) for i in json['processInfo']]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\system_info.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: SystemInfo (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class GPUDevice:
    '''
    Describes a single graphics processor (GPU).
    '''
    #: PCI ID of the GPU vendor, if available; 0 otherwise.
    vendor_id: float

    #: PCI ID of the GPU device, if available; 0 otherwise.
    device_id: float

    #: String description of the GPU vendor, if the PCI ID is not available.
    vendor_string: str

    #: String description of the GPU device, if the PCI ID is not available.
    device_string: str

    #: String description of the GPU driver vendor.
    driver_vendor: str

    #: String description of the GPU driver version.
    driver_version: str

    #: Sub sys ID of the GPU, only available on Windows.
    sub_sys_id: typing.Optional[float] = None

    #: Revision of the GPU, only available on Windows.
    revision: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['vendorId'] = self.vendor_id
        json['deviceId'] = self.device_id
        json['vendorString'] = self.vendor_string
        json['deviceString'] = self.device_string
        json['driverVendor'] = self.driver_vendor
        json['driverVersion'] = self.driver_version
        if self.sub_sys_id is not None:
            json['subSysId'] = self.sub_sys_id
        if self.revision is not None:
            json['revision'] = self.revision
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            vendor_id=float(json['vendorId']),
            device_id=float(json['deviceId']),
            vendor_string=str(json['vendorString']),
            device_string=str(json['deviceString']),
            driver_vendor=str(json['driverVendor']),
            driver_version=str(json['driverVersion']),
            sub_sys_id=float(json['subSysId']) if 'subSysId' in json else None,
            revision=float(json['revision']) if 'revision' in json else None,
        )


@dataclass
class Size:
    '''
    Describes the width and height dimensions of an entity.
    '''
    #: Width in pixels.
    width: int

    #: Height in pixels.
    height: int

    def to_json(self):
        json = dict()
        json['width'] = self.width
        json['height'] = self.height
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            width=int(json['width']),
            height=int(json['height']),
        )


@dataclass
class VideoDecodeAcceleratorCapability:
    '''
    Describes a supported video decoding profile with its associated minimum and
    maximum resolutions.
    '''
    #: Video codec profile that is supported, e.g. VP9 Profile 2.
    profile: str

    #: Maximum video dimensions in pixels supported for this ``profile``.
    max_resolution: Size

    #: Minimum video dimensions in pixels supported for this ``profile``.
    min_resolution: Size

    def to_json(self):
        json = dict()
        json['profile'] = self.profile
        json['maxResolution'] = self.max_resolution.to_json()
        json['minResolution'] = self.min_resolution.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            profile=str(json['profile']),
            max_resolution=Size.from_json(json['maxResolution']),
            min_resolution=Size.from_json(json['minResolution']),
        )


@dataclass
class VideoEncodeAcceleratorCapability:
    '''
    Describes a supported video encoding profile with its associated maximum
    resolution and maximum framerate.
    '''
    #: Video codec profile that is supported, e.g H264 Main.
    profile: str

    #: Maximum video dimensions in pixels supported for this ``profile``.
    max_resolution: Size

    #: Maximum encoding framerate in frames per second supported for this
    #: ``profile``, as fraction's numerator and denominator, e.g. 24/1 fps,
    #: 24000/1001 fps, etc.
    max_framerate_numerator: int

    max_framerate_denominator: int

    def to_json(self):
        json = dict()
        json['profile'] = self.profile
        json['maxResolution'] = self.max_resolution.to_json()
        json['maxFramerateNumerator'] = self.max_framerate_numerator
        json['maxFramerateDenominator'] = self.max_framerate_denominator
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            profile=str(json['profile']),
            max_resolution=Size.from_json(json['maxResolution']),
            max_framerate_numerator=int(json['maxFramerateNumerator']),
            max_framerate_denominator=int(json['maxFramerateDenominator']),
        )


class SubsamplingFormat(enum.Enum):
    '''
    YUV subsampling type of the pixels of a given image.
    '''
    YUV420 = "yuv420"
    YUV422 = "yuv422"
    YUV444 = "yuv444"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ImageType(enum.Enum):
    '''
    Image format of a given image.
    '''
    JPEG = "jpeg"
    WEBP = "webp"
    UNKNOWN = "unknown"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ImageDecodeAcceleratorCapability:
    '''
    Describes a supported image decoding profile with its associated minimum and
    maximum resolutions and subsampling.
    '''
    #: Image coded, e.g. Jpeg.
    image_type: ImageType

    #: Maximum supported dimensions of the image in pixels.
    max_dimensions: Size

    #: Minimum supported dimensions of the image in pixels.
    min_dimensions: Size

    #: Optional array of supported subsampling formats, e.g. 4:2:0, if known.
    subsamplings: typing.List[SubsamplingFormat]

    def to_json(self):
        json = dict()
        json['imageType'] = self.image_type.to_json()
        json['maxDimensions'] = self.max_dimensions.to_json()
        json['minDimensions'] = self.min_dimensions.to_json()
        json['subsamplings'] = [i.to_json() for i in self.subsamplings]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            image_type=ImageType.from_json(json['imageType']),
            max_dimensions=Size.from_json(json['maxDimensions']),
            min_dimensions=Size.from_json(json['minDimensions']),
            subsamplings=[SubsamplingFormat.from_json(i) for i in json['subsamplings']],
        )


@dataclass
class GPUInfo:
    '''
    Provides information about the GPU(s) on the system.
    '''
    #: The graphics devices on the system. Element 0 is the primary GPU.
    devices: typing.List[GPUDevice]

    #: An optional array of GPU driver bug workarounds.
    driver_bug_workarounds: typing.List[str]

    #: Supported accelerated video decoding capabilities.
    video_decoding: typing.List[VideoDecodeAcceleratorCapability]

    #: Supported accelerated video encoding capabilities.
    video_encoding: typing.List[VideoEncodeAcceleratorCapability]

    #: Supported accelerated image decoding capabilities.
    image_decoding: typing.List[ImageDecodeAcceleratorCapability]

    #: An optional dictionary of additional GPU related attributes.
    aux_attributes: typing.Optional[dict] = None

    #: An optional dictionary of graphics features and their status.
    feature_status: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['devices'] = [i.to_json() for i in self.devices]
        json['driverBugWorkarounds'] = [i for i in self.driver_bug_workarounds]
        json['videoDecoding'] = [i.to_json() for i in self.video_decoding]
        json['videoEncoding'] = [i.to_json() for i in self.video_encoding]
        json['imageDecoding'] = [i.to_json() for i in self.image_decoding]
        if self.aux_attributes is not None:
            json['auxAttributes'] = self.aux_attributes
        if self.feature_status is not None:
            json['featureStatus'] = self.feature_status
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            devices=[GPUDevice.from_json(i) for i in json['devices']],
            driver_bug_workarounds=[str(i) for i in json['driverBugWorkarounds']],
            video_decoding=[VideoDecodeAcceleratorCapability.from_json(i) for i in json['videoDecoding']],
            video_encoding=[VideoEncodeAcceleratorCapability.from_json(i) for i in json['videoEncoding']],
            image_decoding=[ImageDecodeAcceleratorCapability.from_json(i) for i in json['imageDecoding']],
            aux_attributes=dict(json['auxAttributes']) if 'auxAttributes' in json else None,
            feature_status=dict(json['featureStatus']) if 'featureStatus' in json else None,
        )


@dataclass
class ProcessInfo:
    '''
    Represents process info.
    '''
    #: Specifies process type.
    type_: str

    #: Specifies process id.
    id_: int

    #: Specifies cumulative CPU usage in seconds across all threads of the
    #: process since the process start.
    cpu_time: float

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['id'] = self.id_
        json['cpuTime'] = self.cpu_time
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            id_=int(json['id']),
            cpu_time=float(json['cpuTime']),
        )


def get_info() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[GPUInfo, str, str, str]]:
    '''
    Returns information about the system.

    :returns: A tuple with the following items:

        0. **gpu** - Information about the GPUs on the system.
        1. **modelName** - A platform-dependent description of the model of the machine. On Mac OS, this is, for example, 'MacBookPro'. Will be the empty string if not supported.
        2. **modelVersion** - A platform-dependent description of the version of the machine. On Mac OS, this is, for example, '10.1'. Will be the empty string if not supported.
        3. **commandLine** - The command line string used to launch the browser. Will be the empty string if not supported.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'SystemInfo.getInfo',
    }
    json = yield cmd_dict
    return (
        GPUInfo.from_json(json['gpu']),
        str(json['modelName']),
        str(json['modelVersion']),
        str(json['commandLine'])
    )


def get_feature_state(
        feature_state: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Returns information about the feature state.

    :param feature_state:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['featureState'] = feature_state
    cmd_dict: T_JSON_DICT = {
        'method': 'SystemInfo.getFeatureState',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['featureEnabled'])


def get_process_info() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[ProcessInfo]]:
    '''
    Returns information about all running processes.

    :returns: An array of process info blocks.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'SystemInfo.getProcessInfo',
    }
    json = yield cmd_dict
    return [ProcessInfo.from_json(i) for i in json['processInfo']]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\system_info.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: SystemInfo (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

@dataclass
class GPUDevice:
    '''
    Describes a single graphics processor (GPU).
    '''
    #: PCI ID of the GPU vendor, if available; 0 otherwise.
    vendor_id: float

    #: PCI ID of the GPU device, if available; 0 otherwise.
    device_id: float

    #: String description of the GPU vendor, if the PCI ID is not available.
    vendor_string: str

    #: String description of the GPU device, if the PCI ID is not available.
    device_string: str

    #: String description of the GPU driver vendor.
    driver_vendor: str

    #: String description of the GPU driver version.
    driver_version: str

    #: Sub sys ID of the GPU, only available on Windows.
    sub_sys_id: typing.Optional[float] = None

    #: Revision of the GPU, only available on Windows.
    revision: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['vendorId'] = self.vendor_id
        json['deviceId'] = self.device_id
        json['vendorString'] = self.vendor_string
        json['deviceString'] = self.device_string
        json['driverVendor'] = self.driver_vendor
        json['driverVersion'] = self.driver_version
        if self.sub_sys_id is not None:
            json['subSysId'] = self.sub_sys_id
        if self.revision is not None:
            json['revision'] = self.revision
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            vendor_id=float(json['vendorId']),
            device_id=float(json['deviceId']),
            vendor_string=str(json['vendorString']),
            device_string=str(json['deviceString']),
            driver_vendor=str(json['driverVendor']),
            driver_version=str(json['driverVersion']),
            sub_sys_id=float(json['subSysId']) if 'subSysId' in json else None,
            revision=float(json['revision']) if 'revision' in json else None,
        )


@dataclass
class Size:
    '''
    Describes the width and height dimensions of an entity.
    '''
    #: Width in pixels.
    width: int

    #: Height in pixels.
    height: int

    def to_json(self):
        json = dict()
        json['width'] = self.width
        json['height'] = self.height
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            width=int(json['width']),
            height=int(json['height']),
        )


@dataclass
class VideoDecodeAcceleratorCapability:
    '''
    Describes a supported video decoding profile with its associated minimum and
    maximum resolutions.
    '''
    #: Video codec profile that is supported, e.g. VP9 Profile 2.
    profile: str

    #: Maximum video dimensions in pixels supported for this ``profile``.
    max_resolution: Size

    #: Minimum video dimensions in pixels supported for this ``profile``.
    min_resolution: Size

    def to_json(self):
        json = dict()
        json['profile'] = self.profile
        json['maxResolution'] = self.max_resolution.to_json()
        json['minResolution'] = self.min_resolution.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            profile=str(json['profile']),
            max_resolution=Size.from_json(json['maxResolution']),
            min_resolution=Size.from_json(json['minResolution']),
        )


@dataclass
class VideoEncodeAcceleratorCapability:
    '''
    Describes a supported video encoding profile with its associated maximum
    resolution and maximum framerate.
    '''
    #: Video codec profile that is supported, e.g H264 Main.
    profile: str

    #: Maximum video dimensions in pixels supported for this ``profile``.
    max_resolution: Size

    #: Maximum encoding framerate in frames per second supported for this
    #: ``profile``, as fraction's numerator and denominator, e.g. 24/1 fps,
    #: 24000/1001 fps, etc.
    max_framerate_numerator: int

    max_framerate_denominator: int

    def to_json(self):
        json = dict()
        json['profile'] = self.profile
        json['maxResolution'] = self.max_resolution.to_json()
        json['maxFramerateNumerator'] = self.max_framerate_numerator
        json['maxFramerateDenominator'] = self.max_framerate_denominator
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            profile=str(json['profile']),
            max_resolution=Size.from_json(json['maxResolution']),
            max_framerate_numerator=int(json['maxFramerateNumerator']),
            max_framerate_denominator=int(json['maxFramerateDenominator']),
        )


class SubsamplingFormat(enum.Enum):
    '''
    YUV subsampling type of the pixels of a given image.
    '''
    YUV420 = "yuv420"
    YUV422 = "yuv422"
    YUV444 = "yuv444"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ImageType(enum.Enum):
    '''
    Image format of a given image.
    '''
    JPEG = "jpeg"
    WEBP = "webp"
    UNKNOWN = "unknown"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ImageDecodeAcceleratorCapability:
    '''
    Describes a supported image decoding profile with its associated minimum and
    maximum resolutions and subsampling.
    '''
    #: Image coded, e.g. Jpeg.
    image_type: ImageType

    #: Maximum supported dimensions of the image in pixels.
    max_dimensions: Size

    #: Minimum supported dimensions of the image in pixels.
    min_dimensions: Size

    #: Optional array of supported subsampling formats, e.g. 4:2:0, if known.
    subsamplings: typing.List[SubsamplingFormat]

    def to_json(self):
        json = dict()
        json['imageType'] = self.image_type.to_json()
        json['maxDimensions'] = self.max_dimensions.to_json()
        json['minDimensions'] = self.min_dimensions.to_json()
        json['subsamplings'] = [i.to_json() for i in self.subsamplings]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            image_type=ImageType.from_json(json['imageType']),
            max_dimensions=Size.from_json(json['maxDimensions']),
            min_dimensions=Size.from_json(json['minDimensions']),
            subsamplings=[SubsamplingFormat.from_json(i) for i in json['subsamplings']],
        )


@dataclass
class GPUInfo:
    '''
    Provides information about the GPU(s) on the system.
    '''
    #: The graphics devices on the system. Element 0 is the primary GPU.
    devices: typing.List[GPUDevice]

    #: An optional array of GPU driver bug workarounds.
    driver_bug_workarounds: typing.List[str]

    #: Supported accelerated video decoding capabilities.
    video_decoding: typing.List[VideoDecodeAcceleratorCapability]

    #: Supported accelerated video encoding capabilities.
    video_encoding: typing.List[VideoEncodeAcceleratorCapability]

    #: Supported accelerated image decoding capabilities.
    image_decoding: typing.List[ImageDecodeAcceleratorCapability]

    #: An optional dictionary of additional GPU related attributes.
    aux_attributes: typing.Optional[dict] = None

    #: An optional dictionary of graphics features and their status.
    feature_status: typing.Optional[dict] = None

    def to_json(self):
        json = dict()
        json['devices'] = [i.to_json() for i in self.devices]
        json['driverBugWorkarounds'] = [i for i in self.driver_bug_workarounds]
        json['videoDecoding'] = [i.to_json() for i in self.video_decoding]
        json['videoEncoding'] = [i.to_json() for i in self.video_encoding]
        json['imageDecoding'] = [i.to_json() for i in self.image_decoding]
        if self.aux_attributes is not None:
            json['auxAttributes'] = self.aux_attributes
        if self.feature_status is not None:
            json['featureStatus'] = self.feature_status
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            devices=[GPUDevice.from_json(i) for i in json['devices']],
            driver_bug_workarounds=[str(i) for i in json['driverBugWorkarounds']],
            video_decoding=[VideoDecodeAcceleratorCapability.from_json(i) for i in json['videoDecoding']],
            video_encoding=[VideoEncodeAcceleratorCapability.from_json(i) for i in json['videoEncoding']],
            image_decoding=[ImageDecodeAcceleratorCapability.from_json(i) for i in json['imageDecoding']],
            aux_attributes=dict(json['auxAttributes']) if 'auxAttributes' in json else None,
            feature_status=dict(json['featureStatus']) if 'featureStatus' in json else None,
        )


@dataclass
class ProcessInfo:
    '''
    Represents process info.
    '''
    #: Specifies process type.
    type_: str

    #: Specifies process id.
    id_: int

    #: Specifies cumulative CPU usage in seconds across all threads of the
    #: process since the process start.
    cpu_time: float

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        json['id'] = self.id_
        json['cpuTime'] = self.cpu_time
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            id_=int(json['id']),
            cpu_time=float(json['cpuTime']),
        )


def get_info() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[GPUInfo, str, str, str]]:
    '''
    Returns information about the system.

    :returns: A tuple with the following items:

        0. **gpu** - Information about the GPUs on the system.
        1. **modelName** - A platform-dependent description of the model of the machine. On Mac OS, this is, for example, 'MacBookPro'. Will be the empty string if not supported.
        2. **modelVersion** - A platform-dependent description of the version of the machine. On Mac OS, this is, for example, '10.1'. Will be the empty string if not supported.
        3. **commandLine** - The command line string used to launch the browser. Will be the empty string if not supported.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'SystemInfo.getInfo',
    }
    json = yield cmd_dict
    return (
        GPUInfo.from_json(json['gpu']),
        str(json['modelName']),
        str(json['modelVersion']),
        str(json['commandLine'])
    )


def get_feature_state(
        feature_state: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Returns information about the feature state.

    :param feature_state:
    :returns: 
    '''
    params: T_JSON_DICT = dict()
    params['featureState'] = feature_state
    cmd_dict: T_JSON_DICT = {
        'method': 'SystemInfo.getFeatureState',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['featureEnabled'])


def get_process_info() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[ProcessInfo]]:
    '''
    Returns information about all running processes.

    :returns: An array of process info blocks.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'SystemInfo.getProcessInfo',
    }
    json = yield cmd_dict
    return [ProcessInfo.from_json(i) for i in json['processInfo']]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\fixtures\base.py ===
# testing/fixtures/base.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


from __future__ import annotations

import sqlalchemy as sa
from .. import assertions
from .. import config
from ..assertions import eq_
from ..util import drop_all_tables_from_metadata
from ... import Column
from ... import func
from ... import Integer
from ... import select
from ... import Table
from ...orm import DeclarativeBase
from ...orm import MappedAsDataclass
from ...orm import registry


@config.mark_base_test_class()
class TestBase:
    # A sequence of requirement names matching testing.requires decorators
    __requires__ = ()

    # A sequence of dialect names to exclude from the test class.
    __unsupported_on__ = ()

    # If present, test class is only runnable for the *single* specified
    # dialect.  If you need multiple, use __unsupported_on__ and invert.
    __only_on__ = None

    # A sequence of no-arg callables. If any are True, the entire testcase is
    # skipped.
    __skip_if__ = None

    # if True, the testing reaper will not attempt to touch connection
    # state after a test is completed and before the outer teardown
    # starts
    __leave_connections_for_teardown__ = False

    def assert_(self, val, msg=None):
        assert val, msg

    @config.fixture()
    def nocache(self):
        _cache = config.db._compiled_cache
        config.db._compiled_cache = None
        yield
        config.db._compiled_cache = _cache

    @config.fixture()
    def connection_no_trans(self):
        eng = getattr(self, "bind", None) or config.db

        with eng.connect() as conn:
            yield conn

    @config.fixture()
    def connection(self):
        global _connection_fixture_connection

        eng = getattr(self, "bind", None) or config.db

        conn = eng.connect()
        trans = conn.begin()

        _connection_fixture_connection = conn
        yield conn

        _connection_fixture_connection = None

        if trans.is_active:
            trans.rollback()
        # trans would not be active here if the test is using
        # the legacy @provide_metadata decorator still, as it will
        # run a close all connections.
        conn.close()

    @config.fixture()
    def close_result_when_finished(self):
        to_close = []
        to_consume = []

        def go(result, consume=False):
            to_close.append(result)
            if consume:
                to_consume.append(result)

        yield go
        for r in to_consume:
            try:
                r.all()
            except:
                pass
        for r in to_close:
            try:
                r.close()
            except:
                pass

    @config.fixture()
    def registry(self, metadata):
        reg = registry(
            metadata=metadata,
            type_annotation_map={
                str: sa.String().with_variant(
                    sa.String(50), "mysql", "mariadb", "oracle"
                )
            },
        )
        yield reg
        reg.dispose()

    @config.fixture
    def decl_base(self, metadata):
        _md = metadata

        class Base(DeclarativeBase):
            metadata = _md
            type_annotation_map = {
                str: sa.String().with_variant(
                    sa.String(50), "mysql", "mariadb", "oracle"
                )
            }

        yield Base
        Base.registry.dispose()

    @config.fixture
    def dc_decl_base(self, metadata):
        _md = metadata

        class Base(MappedAsDataclass, DeclarativeBase):
            metadata = _md
            type_annotation_map = {
                str: sa.String().with_variant(
                    sa.String(50), "mysql", "mariadb"
                )
            }

        yield Base
        Base.registry.dispose()

    @config.fixture()
    def future_connection(self, future_engine, connection):
        # integrate the future_engine and connection fixtures so
        # that users of the "connection" fixture will get at the
        # "future" connection
        yield connection

    @config.fixture()
    def future_engine(self):
        yield

    @config.fixture()
    def testing_engine(self):
        from .. import engines

        def gen_testing_engine(
            url=None,
            options=None,
            future=None,
            asyncio=False,
            transfer_staticpool=False,
            share_pool=False,
        ):
            if options is None:
                options = {}
            options["scope"] = "fixture"
            return engines.testing_engine(
                url=url,
                options=options,
                asyncio=asyncio,
                transfer_staticpool=transfer_staticpool,
                share_pool=share_pool,
            )

        yield gen_testing_engine

        engines.testing_reaper._drop_testing_engines("fixture")

    @config.fixture()
    def async_testing_engine(self, testing_engine):
        def go(**kw):
            kw["asyncio"] = True
            return testing_engine(**kw)

        return go

    @config.fixture()
    def metadata(self, request):
        """Provide bound MetaData for a single test, dropping afterwards."""

        from ...sql import schema

        metadata = schema.MetaData()
        request.instance.metadata = metadata
        yield metadata
        del request.instance.metadata

        if (
            _connection_fixture_connection
            and _connection_fixture_connection.in_transaction()
        ):
            trans = _connection_fixture_connection.get_transaction()
            trans.rollback()
            with _connection_fixture_connection.begin():
                drop_all_tables_from_metadata(
                    metadata, _connection_fixture_connection
                )
        else:
            drop_all_tables_from_metadata(metadata, config.db)

    @config.fixture(
        params=[
            (rollback, second_operation, begin_nested)
            for rollback in (True, False)
            for second_operation in ("none", "execute", "begin")
            for begin_nested in (
                True,
                False,
            )
        ]
    )
    def trans_ctx_manager_fixture(self, request, metadata):
        rollback, second_operation, begin_nested = request.param

        t = Table("test", metadata, Column("data", Integer))
        eng = getattr(self, "bind", None) or config.db

        t.create(eng)

        def run_test(subject, trans_on_subject, execute_on_subject):
            with subject.begin() as trans:
                if begin_nested:
                    if not config.requirements.savepoints.enabled:
                        config.skip_test("savepoints not enabled")
                    if execute_on_subject:
                        nested_trans = subject.begin_nested()
                    else:
                        nested_trans = trans.begin_nested()

                    with nested_trans:
                        if execute_on_subject:
                            subject.execute(t.insert(), {"data": 10})
                        else:
                            trans.execute(t.insert(), {"data": 10})

                        # for nested trans, we always commit/rollback on the
                        # "nested trans" object itself.
                        # only Session(future=False) will affect savepoint
                        # transaction for session.commit/rollback

                        if rollback:
                            nested_trans.rollback()
                        else:
                            nested_trans.commit()

                        if second_operation != "none":
                            with assertions.expect_raises_message(
                                sa.exc.InvalidRequestError,
                                "Can't operate on closed transaction "
                                "inside context "
                                "manager.  Please complete the context "
                                "manager "
                                "before emitting further commands.",
                            ):
                                if second_operation == "execute":
                                    if execute_on_subject:
                                        subject.execute(
                                            t.insert(), {"data": 12}
                                        )
                                    else:
                                        trans.execute(t.insert(), {"data": 12})
                                elif second_operation == "begin":
                                    if execute_on_subject:
                                        subject.begin_nested()
                                    else:
                                        trans.begin_nested()

                    # outside the nested trans block, but still inside the
                    # transaction block, we can run SQL, and it will be
                    # committed
                    if execute_on_subject:
                        subject.execute(t.insert(), {"data": 14})
                    else:
                        trans.execute(t.insert(), {"data": 14})

                else:
                    if execute_on_subject:
                        subject.execute(t.insert(), {"data": 10})
                    else:
                        trans.execute(t.insert(), {"data": 10})

                    if trans_on_subject:
                        if rollback:
                            subject.rollback()
                        else:
                            subject.commit()
                    else:
                        if rollback:
                            trans.rollback()
                        else:
                            trans.commit()

                    if second_operation != "none":
                        with assertions.expect_raises_message(
                            sa.exc.InvalidRequestError,
                            "Can't operate on closed transaction inside "
                            "context "
                            "manager.  Please complete the context manager "
                            "before emitting further commands.",
                        ):
                            if second_operation == "execute":
                                if execute_on_subject:
                                    subject.execute(t.insert(), {"data": 12})
                                else:
                                    trans.execute(t.insert(), {"data": 12})
                            elif second_operation == "begin":
                                if hasattr(trans, "begin"):
                                    trans.begin()
                                else:
                                    subject.begin()
                            elif second_operation == "begin_nested":
                                if execute_on_subject:
                                    subject.begin_nested()
                                else:
                                    trans.begin_nested()

            expected_committed = 0
            if begin_nested:
                # begin_nested variant, we inserted a row after the nested
                # block
                expected_committed += 1
            if not rollback:
                # not rollback variant, our row inserted in the target
                # block itself would be committed
                expected_committed += 1

            if execute_on_subject:
                eq_(
                    subject.scalar(select(func.count()).select_from(t)),
                    expected_committed,
                )
            else:
                with subject.connect() as conn:
                    eq_(
                        conn.scalar(select(func.count()).select_from(t)),
                        expected_committed,
                    )

        return run_test


_connection_fixture_connection = None


class FutureEngineMixin:
    """alembic's suite still using this"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\tableform.py ===
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import SympifyError

from types import FunctionType


class TableForm:
    r"""
    Create a nice table representation of data.

    Examples
    ========

    >>> from sympy import TableForm
    >>> t = TableForm([[5, 7], [4, 2], [10, 3]])
    >>> print(t)
    5  7
    4  2
    10 3

    You can use the SymPy's printing system to produce tables in any
    format (ascii, latex, html, ...).

    >>> print(t.as_latex())
    \begin{tabular}{l l}
    $5$ & $7$ \\
    $4$ & $2$ \\
    $10$ & $3$ \\
    \end{tabular}

    """

    def __init__(self, data, **kwarg):
        """
        Creates a TableForm.

        Parameters:

            data ...
                            2D data to be put into the table; data can be
                            given as a Matrix

            headings ...
                            gives the labels for rows and columns:

                            Can be a single argument that applies to both
                            dimensions:

                                - None ... no labels
                                - "automatic" ... labels are 1, 2, 3, ...

                            Can be a list of labels for rows and columns:
                            The labels for each dimension can be given
                            as None, "automatic", or [l1, l2, ...] e.g.
                            ["automatic", None] will number the rows

                            [default: None]

            alignments ...
                            alignment of the columns with:

                                - "left" or "<"
                                - "center" or "^"
                                - "right" or ">"

                            When given as a single value, the value is used for
                            all columns. The row headings (if given) will be
                            right justified unless an explicit alignment is
                            given for it and all other columns.

                            [default: "left"]

            formats ...
                            a list of format strings or functions that accept
                            3 arguments (entry, row number, col number) and
                            return a string for the table entry. (If a function
                            returns None then the _print method will be used.)

            wipe_zeros ...
                            Do not show zeros in the table.

                            [default: True]

            pad ...
                            the string to use to indicate a missing value (e.g.
                            elements that are None or those that are missing
                            from the end of a row (i.e. any row that is shorter
                            than the rest is assumed to have missing values).
                            When None, nothing will be shown for values that
                            are missing from the end of a row; values that are
                            None, however, will be shown.

                            [default: None]

        Examples
        ========

        >>> from sympy import TableForm, Symbol
        >>> TableForm([[5, 7], [4, 2], [10, 3]])
        5  7
        4  2
        10 3
        >>> TableForm([list('.'*i) for i in range(1, 4)], headings='automatic')
          | 1 2 3
        ---------
        1 | .
        2 | . .
        3 | . . .
        >>> TableForm([[Symbol('.'*(j if not i%2 else 1)) for i in range(3)]
        ...            for j in range(4)], alignments='rcl')
            .
          . . .
         .. . ..
        ... . ...
        """
        from sympy.matrices.dense import Matrix

        # We only support 2D data. Check the consistency:
        if isinstance(data, Matrix):
            data = data.tolist()
        _h = len(data)

        # fill out any short lines
        pad = kwarg.get('pad', None)
        ok_None = False
        if pad is None:
            pad = " "
            ok_None = True
        pad = Symbol(pad)
        _w = max(len(line) for line in data)
        for i, line in enumerate(data):
            if len(line) != _w:
                line.extend([pad]*(_w - len(line)))
            for j, lj in enumerate(line):
                if lj is None:
                    if not ok_None:
                        lj = pad
                else:
                    try:
                        lj = S(lj)
                    except SympifyError:
                        lj = Symbol(str(lj))
                line[j] = lj
            data[i] = line
        _lines = Tuple(*[Tuple(*d) for d in data])

        headings = kwarg.get("headings", [None, None])
        if headings == "automatic":
            _headings = [range(1, _h + 1), range(1, _w + 1)]
        else:
            h1, h2 = headings
            if h1 == "automatic":
                h1 = range(1, _h + 1)
            if h2 == "automatic":
                h2 = range(1, _w + 1)
            _headings = [h1, h2]

        allow = ('l', 'r', 'c')
        alignments = kwarg.get("alignments", "l")

        def _std_align(a):
            a = a.strip().lower()
            if len(a) > 1:
                return {'left': 'l', 'right': 'r', 'center': 'c'}.get(a, a)
            else:
                return {'<': 'l', '>': 'r', '^': 'c'}.get(a, a)
        std_align = _std_align(alignments)
        if std_align in allow:
            _alignments = [std_align]*_w
        else:
            _alignments = []
            for a in alignments:
                std_align = _std_align(a)
                _alignments.append(std_align)
                if std_align not in ('l', 'r', 'c'):
                    raise ValueError('alignment "%s" unrecognized' %
                        alignments)
        if _headings[0] and len(_alignments) == _w + 1:
            _head_align = _alignments[0]
            _alignments = _alignments[1:]
        else:
            _head_align = 'r'
        if len(_alignments) != _w:
            raise ValueError(
                'wrong number of alignments: expected %s but got %s' %
                (_w, len(_alignments)))

        _column_formats = kwarg.get("formats", [None]*_w)

        _wipe_zeros = kwarg.get("wipe_zeros", True)

        self._w = _w
        self._h = _h
        self._lines = _lines
        self._headings = _headings
        self._head_align = _head_align
        self._alignments = _alignments
        self._column_formats = _column_formats
        self._wipe_zeros = _wipe_zeros

    def __repr__(self):
        from .str import sstr
        return sstr(self, order=None)

    def __str__(self):
        from .str import sstr
        return sstr(self, order=None)

    def as_matrix(self):
        """Returns the data of the table in Matrix form.

        Examples
        ========

        >>> from sympy import TableForm
        >>> t = TableForm([[5, 7], [4, 2], [10, 3]], headings='automatic')
        >>> t
          | 1  2
        --------
        1 | 5  7
        2 | 4  2
        3 | 10 3
        >>> t.as_matrix()
        Matrix([
        [ 5, 7],
        [ 4, 2],
        [10, 3]])
        """
        from sympy.matrices.dense import Matrix
        return Matrix(self._lines)

    def as_str(self):
        # XXX obsolete ?
        return str(self)

    def as_latex(self):
        from .latex import latex
        return latex(self)

    def _sympystr(self, p):
        """
        Returns the string representation of 'self'.

        Examples
        ========

        >>> from sympy import TableForm
        >>> t = TableForm([[5, 7], [4, 2], [10, 3]])
        >>> s = t.as_str()

        """
        column_widths = [0] * self._w
        lines = []
        for line in self._lines:
            new_line = []
            for i in range(self._w):
                # Format the item somehow if needed:
                s = str(line[i])
                if self._wipe_zeros and (s == "0"):
                    s = " "
                w = len(s)
                if w > column_widths[i]:
                    column_widths[i] = w
                new_line.append(s)
            lines.append(new_line)

        # Check heading:
        if self._headings[0]:
            self._headings[0] = [str(x) for x in self._headings[0]]
            _head_width = max(len(x) for x in self._headings[0])

        if self._headings[1]:
            new_line = []
            for i in range(self._w):
                # Format the item somehow if needed:
                s = str(self._headings[1][i])
                w = len(s)
                if w > column_widths[i]:
                    column_widths[i] = w
                new_line.append(s)
            self._headings[1] = new_line

        format_str = []

        def _align(align, w):
            return '%%%s%ss' % (
                ("-" if align == "l" else ""),
                str(w))
        format_str = [_align(align, w) for align, w in
                      zip(self._alignments, column_widths)]
        if self._headings[0]:
            format_str.insert(0, _align(self._head_align, _head_width))
            format_str.insert(1, '|')
        format_str = ' '.join(format_str) + '\n'

        s = []
        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                d = [""] + d
            first_line = format_str % tuple(d)
            s.append(first_line)
            s.append("-" * (len(first_line) - 1) + "\n")
        for i, line in enumerate(lines):
            d = [l if self._alignments[j] != 'c' else
                 l.center(column_widths[j]) for j, l in enumerate(line)]
            if self._headings[0]:
                l = self._headings[0][i]
                l = (l if self._head_align != 'c' else
                     l.center(_head_width))
                d = [l] + d
            s.append(format_str % tuple(d))
        return ''.join(s)[:-1]  # don't include trailing newline

    def _latex(self, printer):
        """
        Returns the string representation of 'self'.
        """
        # Check heading:
        if self._headings[1]:
            new_line = []
            for i in range(self._w):
                # Format the item somehow if needed:
                new_line.append(str(self._headings[1][i]))
            self._headings[1] = new_line

        alignments = []
        if self._headings[0]:
            self._headings[0] = [str(x) for x in self._headings[0]]
            alignments = [self._head_align]
        alignments.extend(self._alignments)

        s = r"\begin{tabular}{" + " ".join(alignments) + "}\n"

        if self._headings[1]:
            d = self._headings[1]
            if self._headings[0]:
                d = [""] + d
            first_line = " & ".join(d) + r" \\" + "\n"
            s += first_line
            s += r"\hline" + "\n"
        for i, line in enumerate(self._lines):
            d = []
            for j, x in enumerate(line):
                if self._wipe_zeros and (x in (0, "0")):
                    d.append(" ")
                    continue
                f = self._column_formats[j]
                if f:
                    if isinstance(f, FunctionType):
                        v = f(x, i, j)
                        if v is None:
                            v = printer._print(x)
                    else:
                        v = f % x
                    d.append(v)
                else:
                    v = printer._print(x)
                    d.append("$%s$" % v)
            if self._headings[0]:
                d = [self._headings[0][i]] + d
            s += " & ".join(d) + r" \\" + "\n"
        s += r"\end{tabular}"
        return s

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\h11\_state.py ===
################################################################
# The core state machine
################################################################
#
# Rule 1: everything that affects the state machine and state transitions must
# live here in this file. As much as possible goes into the table-based
# representation, but for the bits that don't quite fit, the actual code and
# state must nonetheless live here.
#
# Rule 2: this file does not know about what role we're playing; it only knows
# about HTTP request/response cycles in the abstract. This ensures that we
# don't cheat and apply different rules to local and remote parties.
#
#
# Theory of operation
# ===================
#
# Possibly the simplest way to think about this is that we actually have 5
# different state machines here. Yes, 5. These are:
#
# 1) The client state, with its complicated automaton (see the docs)
# 2) The server state, with its complicated automaton (see the docs)
# 3) The keep-alive state, with possible states {True, False}
# 4) The SWITCH_CONNECT state, with possible states {False, True}
# 5) The SWITCH_UPGRADE state, with possible states {False, True}
#
# For (3)-(5), the first state listed is the initial state.
#
# (1)-(3) are stored explicitly in member variables. The last
# two are stored implicitly in the pending_switch_proposals set as:
#   (state of 4) == (_SWITCH_CONNECT in pending_switch_proposals)
#   (state of 5) == (_SWITCH_UPGRADE in pending_switch_proposals)
#
# And each of these machines has two different kinds of transitions:
#
# a) Event-triggered
# b) State-triggered
#
# Event triggered is the obvious thing that you'd think it is: some event
# happens, and if it's the right event at the right time then a transition
# happens. But there are somewhat complicated rules for which machines can
# "see" which events. (As a rule of thumb, if a machine "sees" an event, this
# means two things: the event can affect the machine, and if the machine is
# not in a state where it expects that event then it's an error.) These rules
# are:
#
# 1) The client machine sees all h11.events objects emitted by the client.
#
# 2) The server machine sees all h11.events objects emitted by the server.
#
#    It also sees the client's Request event.
#
#    And sometimes, server events are annotated with a _SWITCH_* event. For
#    example, we can have a (Response, _SWITCH_CONNECT) event, which is
#    different from a regular Response event.
#
# 3) The keep-alive machine sees the process_keep_alive_disabled() event
#    (which is derived from Request/Response events), and this event
#    transitions it from True -> False, or from False -> False. There's no way
#    to transition back.
#
# 4&5) The _SWITCH_* machines transition from False->True when we get a
#    Request that proposes the relevant type of switch (via
#    process_client_switch_proposals), and they go from True->False when we
#    get a Response that has no _SWITCH_* annotation.
#
# So that's event-triggered transitions.
#
# State-triggered transitions are less standard. What they do here is couple
# the machines together. The way this works is, when certain *joint*
# configurations of states are achieved, then we automatically transition to a
# new *joint* state. So, for example, if we're ever in a joint state with
#
#   client: DONE
#   keep-alive: False
#
# then the client state immediately transitions to:
#
#   client: MUST_CLOSE
#
# This is fundamentally different from an event-based transition, because it
# doesn't matter how we arrived at the {client: DONE, keep-alive: False} state
# -- maybe the client transitioned SEND_BODY -> DONE, or keep-alive
# transitioned True -> False. Either way, once this precondition is satisfied,
# this transition is immediately triggered.
#
# What if two conflicting state-based transitions get enabled at the same
# time?  In practice there's only one case where this arises (client DONE ->
# MIGHT_SWITCH_PROTOCOL versus DONE -> MUST_CLOSE), and we resolve it by
# explicitly prioritizing the DONE -> MIGHT_SWITCH_PROTOCOL transition.
#
# Implementation
# --------------
#
# The event-triggered transitions for the server and client machines are all
# stored explicitly in a table. Ditto for the state-triggered transitions that
# involve just the server and client state.
#
# The transitions for the other machines, and the state-triggered transitions
# that involve the other machines, are written out as explicit Python code.
#
# It'd be nice if there were some cleaner way to do all this. This isn't
# *too* terrible, but I feel like it could probably be better.
#
# WARNING
# -------
#
# The script that generates the state machine diagrams for the docs knows how
# to read out the EVENT_TRIGGERED_TRANSITIONS and STATE_TRIGGERED_TRANSITIONS
# tables. But it can't automatically read the transitions that are written
# directly in Python code. So if you touch those, you need to also update the
# script to keep it in sync!
from typing import cast, Dict, Optional, Set, Tuple, Type, Union

from ._events import *
from ._util import LocalProtocolError, Sentinel

# Everything in __all__ gets re-exported as part of the h11 public API.
__all__ = [
    "CLIENT",
    "SERVER",
    "IDLE",
    "SEND_RESPONSE",
    "SEND_BODY",
    "DONE",
    "MUST_CLOSE",
    "CLOSED",
    "MIGHT_SWITCH_PROTOCOL",
    "SWITCHED_PROTOCOL",
    "ERROR",
]


class CLIENT(Sentinel, metaclass=Sentinel):
    pass


class SERVER(Sentinel, metaclass=Sentinel):
    pass


# States
class IDLE(Sentinel, metaclass=Sentinel):
    pass


class SEND_RESPONSE(Sentinel, metaclass=Sentinel):
    pass


class SEND_BODY(Sentinel, metaclass=Sentinel):
    pass


class DONE(Sentinel, metaclass=Sentinel):
    pass


class MUST_CLOSE(Sentinel, metaclass=Sentinel):
    pass


class CLOSED(Sentinel, metaclass=Sentinel):
    pass


class ERROR(Sentinel, metaclass=Sentinel):
    pass


# Switch types
class MIGHT_SWITCH_PROTOCOL(Sentinel, metaclass=Sentinel):
    pass


class SWITCHED_PROTOCOL(Sentinel, metaclass=Sentinel):
    pass


class _SWITCH_UPGRADE(Sentinel, metaclass=Sentinel):
    pass


class _SWITCH_CONNECT(Sentinel, metaclass=Sentinel):
    pass


EventTransitionType = Dict[
    Type[Sentinel],
    Dict[
        Type[Sentinel],
        Dict[Union[Type[Event], Tuple[Type[Event], Type[Sentinel]]], Type[Sentinel]],
    ],
]

EVENT_TRIGGERED_TRANSITIONS: EventTransitionType = {
    CLIENT: {
        IDLE: {Request: SEND_BODY, ConnectionClosed: CLOSED},
        SEND_BODY: {Data: SEND_BODY, EndOfMessage: DONE},
        DONE: {ConnectionClosed: CLOSED},
        MUST_CLOSE: {ConnectionClosed: CLOSED},
        CLOSED: {ConnectionClosed: CLOSED},
        MIGHT_SWITCH_PROTOCOL: {},
        SWITCHED_PROTOCOL: {},
        ERROR: {},
    },
    SERVER: {
        IDLE: {
            ConnectionClosed: CLOSED,
            Response: SEND_BODY,
            # Special case: server sees client Request events, in this form
            (Request, CLIENT): SEND_RESPONSE,
        },
        SEND_RESPONSE: {
            InformationalResponse: SEND_RESPONSE,
            Response: SEND_BODY,
            (InformationalResponse, _SWITCH_UPGRADE): SWITCHED_PROTOCOL,
            (Response, _SWITCH_CONNECT): SWITCHED_PROTOCOL,
        },
        SEND_BODY: {Data: SEND_BODY, EndOfMessage: DONE},
        DONE: {ConnectionClosed: CLOSED},
        MUST_CLOSE: {ConnectionClosed: CLOSED},
        CLOSED: {ConnectionClosed: CLOSED},
        SWITCHED_PROTOCOL: {},
        ERROR: {},
    },
}

StateTransitionType = Dict[
    Tuple[Type[Sentinel], Type[Sentinel]], Dict[Type[Sentinel], Type[Sentinel]]
]

# NB: there are also some special-case state-triggered transitions hard-coded
# into _fire_state_triggered_transitions below.
STATE_TRIGGERED_TRANSITIONS: StateTransitionType = {
    # (Client state, Server state) -> new states
    # Protocol negotiation
    (MIGHT_SWITCH_PROTOCOL, SWITCHED_PROTOCOL): {CLIENT: SWITCHED_PROTOCOL},
    # Socket shutdown
    (CLOSED, DONE): {SERVER: MUST_CLOSE},
    (CLOSED, IDLE): {SERVER: MUST_CLOSE},
    (ERROR, DONE): {SERVER: MUST_CLOSE},
    (DONE, CLOSED): {CLIENT: MUST_CLOSE},
    (IDLE, CLOSED): {CLIENT: MUST_CLOSE},
    (DONE, ERROR): {CLIENT: MUST_CLOSE},
}


class ConnectionState:
    def __init__(self) -> None:
        # Extra bits of state that don't quite fit into the state model.

        # If this is False then it enables the automatic DONE -> MUST_CLOSE
        # transition. Don't set this directly; call .keep_alive_disabled()
        self.keep_alive = True

        # This is a subset of {UPGRADE, CONNECT}, containing the proposals
        # made by the client for switching protocols.
        self.pending_switch_proposals: Set[Type[Sentinel]] = set()

        self.states: Dict[Type[Sentinel], Type[Sentinel]] = {CLIENT: IDLE, SERVER: IDLE}

    def process_error(self, role: Type[Sentinel]) -> None:
        self.states[role] = ERROR
        self._fire_state_triggered_transitions()

    def process_keep_alive_disabled(self) -> None:
        self.keep_alive = False
        self._fire_state_triggered_transitions()

    def process_client_switch_proposal(self, switch_event: Type[Sentinel]) -> None:
        self.pending_switch_proposals.add(switch_event)
        self._fire_state_triggered_transitions()

    def process_event(
        self,
        role: Type[Sentinel],
        event_type: Type[Event],
        server_switch_event: Optional[Type[Sentinel]] = None,
    ) -> None:
        _event_type: Union[Type[Event], Tuple[Type[Event], Type[Sentinel]]] = event_type
        if server_switch_event is not None:
            assert role is SERVER
            if server_switch_event not in self.pending_switch_proposals:
                raise LocalProtocolError(
                    "Received server _SWITCH_UPGRADE event without a pending proposal"
                )
            _event_type = (event_type, server_switch_event)
        if server_switch_event is None and _event_type is Response:
            self.pending_switch_proposals = set()
        self._fire_event_triggered_transitions(role, _event_type)
        # Special case: the server state does get to see Request
        # events.
        if _event_type is Request:
            assert role is CLIENT
            self._fire_event_triggered_transitions(SERVER, (Request, CLIENT))
        self._fire_state_triggered_transitions()

    def _fire_event_triggered_transitions(
        self,
        role: Type[Sentinel],
        event_type: Union[Type[Event], Tuple[Type[Event], Type[Sentinel]]],
    ) -> None:
        state = self.states[role]
        try:
            new_state = EVENT_TRIGGERED_TRANSITIONS[role][state][event_type]
        except KeyError:
            event_type = cast(Type[Event], event_type)
            raise LocalProtocolError(
                "can't handle event type {} when role={} and state={}".format(
                    event_type.__name__, role, self.states[role]
                )
            ) from None
        self.states[role] = new_state

    def _fire_state_triggered_transitions(self) -> None:
        # We apply these rules repeatedly until converging on a fixed point
        while True:
            start_states = dict(self.states)

            # It could happen that both these special-case transitions are
            # enabled at the same time:
            #
            #    DONE -> MIGHT_SWITCH_PROTOCOL
            #    DONE -> MUST_CLOSE
            #
            # For example, this will always be true of a HTTP/1.0 client
            # requesting CONNECT.  If this happens, the protocol switch takes
            # priority. From there the client will either go to
            # SWITCHED_PROTOCOL, in which case it's none of our business when
            # they close the connection, or else the server will deny the
            # request, in which case the client will go back to DONE and then
            # from there to MUST_CLOSE.
            if self.pending_switch_proposals:
                if self.states[CLIENT] is DONE:
                    self.states[CLIENT] = MIGHT_SWITCH_PROTOCOL

            if not self.pending_switch_proposals:
                if self.states[CLIENT] is MIGHT_SWITCH_PROTOCOL:
                    self.states[CLIENT] = DONE

            if not self.keep_alive:
                for role in (CLIENT, SERVER):
                    if self.states[role] is DONE:
                        self.states[role] = MUST_CLOSE

            # Tabular state-triggered transitions
            joint_state = (self.states[CLIENT], self.states[SERVER])
            changes = STATE_TRIGGERED_TRANSITIONS.get(joint_state, {})
            self.states.update(changes)

            if self.states == start_states:
                # Fixed point reached
                return

    def start_next_cycle(self) -> None:
        if self.states != {CLIENT: DONE, SERVER: DONE}:
            raise LocalProtocolError(
                f"not in a reusable state. self.states={self.states}"
            )
        # Can't reach DONE/DONE with any of these active, but still, let's be
        # sure.
        assert self.keep_alive
        assert not self.pending_switch_proposals
        self.states = {CLIENT: IDLE, SERVER: IDLE}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask_sqlalchemy\pagination.py ===
from __future__ import annotations

import typing as t
from math import ceil

import sqlalchemy as sa
import sqlalchemy.orm as sa_orm
from flask import abort
from flask import request


class Pagination:
    """Apply an offset and limit to the query based on the current page and number of
    items per page.

    Don't create pagination objects manually. They are created by
    :meth:`.SQLAlchemy.paginate` and :meth:`.Query.paginate`.

    This is a base class, a subclass must implement :meth:`_query_items` and
    :meth:`_query_count`. Those methods will use arguments passed as ``kwargs`` to
    perform the queries.

    :param page: The current page, used to calculate the offset. Defaults to the
        ``page`` query arg during a request, or 1 otherwise.
    :param per_page: The maximum number of items on a page, used to calculate the
        offset and limit. Defaults to the ``per_page`` query arg during a request,
        or 20 otherwise.
    :param max_per_page: The maximum allowed value for ``per_page``, to limit a
        user-provided value. Use ``None`` for no limit. Defaults to 100.
    :param error_out: Abort with a ``404 Not Found`` error if no items are returned
        and ``page`` is not 1, or if ``page`` or ``per_page`` is less than 1, or if
        either are not ints.
    :param count: Calculate the total number of values by issuing an extra count
        query. For very complex queries this may be inaccurate or slow, so it can be
        disabled and set manually if necessary.
    :param kwargs: Information about the query to paginate. Different subclasses will
        require different arguments.

    .. versionchanged:: 3.0
        Iterating over a pagination object iterates over its items.

    .. versionchanged:: 3.0
        Creating instances manually is not a public API.
    """

    def __init__(
        self,
        page: int | None = None,
        per_page: int | None = None,
        max_per_page: int | None = 100,
        error_out: bool = True,
        count: bool = True,
        **kwargs: t.Any,
    ) -> None:
        self._query_args = kwargs
        page, per_page = self._prepare_page_args(
            page=page,
            per_page=per_page,
            max_per_page=max_per_page,
            error_out=error_out,
        )

        self.page: int = page
        """The current page."""

        self.per_page: int = per_page
        """The maximum number of items on a page."""

        self.max_per_page: int | None = max_per_page
        """The maximum allowed value for ``per_page``."""

        items = self._query_items()

        if not items and page != 1 and error_out:
            abort(404)

        self.items: list[t.Any] = items
        """The items on the current page. Iterating over the pagination object is
        equivalent to iterating over the items.
        """

        if count:
            total = self._query_count()
        else:
            total = None

        self.total: int | None = total
        """The total number of items across all pages."""

    @staticmethod
    def _prepare_page_args(
        *,
        page: int | None = None,
        per_page: int | None = None,
        max_per_page: int | None = None,
        error_out: bool = True,
    ) -> tuple[int, int]:
        if request:
            if page is None:
                try:
                    page = int(request.args.get("page", 1))
                except (TypeError, ValueError):
                    if error_out:
                        abort(404)

                    page = 1

            if per_page is None:
                try:
                    per_page = int(request.args.get("per_page", 20))
                except (TypeError, ValueError):
                    if error_out:
                        abort(404)

                    per_page = 20
        else:
            if page is None:
                page = 1

            if per_page is None:
                per_page = 20

        if max_per_page is not None:
            per_page = min(per_page, max_per_page)

        if page < 1:
            if error_out:
                abort(404)
            else:
                page = 1

        if per_page < 1:
            if error_out:
                abort(404)
            else:
                per_page = 20

        return page, per_page

    @property
    def _query_offset(self) -> int:
        """The index of the first item to query, passed to ``offset()``.

        :meta private:

        .. versionadded:: 3.0
        """
        return (self.page - 1) * self.per_page

    def _query_items(self) -> list[t.Any]:
        """Execute the query to get the items on the current page.

        Uses init arguments stored in :attr:`_query_args`.

        :meta private:

        .. versionadded:: 3.0
        """
        raise NotImplementedError

    def _query_count(self) -> int:
        """Execute the query to get the total number of items.

        Uses init arguments stored in :attr:`_query_args`.

        :meta private:

        .. versionadded:: 3.0
        """
        raise NotImplementedError

    @property
    def first(self) -> int:
        """The number of the first item on the page, starting from 1, or 0 if there are
        no items.

        .. versionadded:: 3.0
        """
        if len(self.items) == 0:
            return 0

        return (self.page - 1) * self.per_page + 1

    @property
    def last(self) -> int:
        """The number of the last item on the page, starting from 1, inclusive, or 0 if
        there are no items.

        .. versionadded:: 3.0
        """
        first = self.first
        return max(first, first + len(self.items) - 1)

    @property
    def pages(self) -> int:
        """The total number of pages."""
        if self.total == 0 or self.total is None:
            return 0

        return ceil(self.total / self.per_page)

    @property
    def has_prev(self) -> bool:
        """``True`` if this is not the first page."""
        return self.page > 1

    @property
    def prev_num(self) -> int | None:
        """The previous page number, or ``None`` if this is the first page."""
        if not self.has_prev:
            return None

        return self.page - 1

    def prev(self, *, error_out: bool = False) -> Pagination:
        """Query the :class:`Pagination` object for the previous page.

        :param error_out: Abort with a ``404 Not Found`` error if no items are returned
            and ``page`` is not 1, or if ``page`` or ``per_page`` is less than 1, or if
            either are not ints.
        """
        p = type(self)(
            page=self.page - 1,
            per_page=self.per_page,
            error_out=error_out,
            count=False,
            **self._query_args,
        )
        p.total = self.total
        return p

    @property
    def has_next(self) -> bool:
        """``True`` if this is not the last page."""
        return self.page < self.pages

    @property
    def next_num(self) -> int | None:
        """The next page number, or ``None`` if this is the last page."""
        if not self.has_next:
            return None

        return self.page + 1

    def next(self, *, error_out: bool = False) -> Pagination:
        """Query the :class:`Pagination` object for the next page.

        :param error_out: Abort with a ``404 Not Found`` error if no items are returned
            and ``page`` is not 1, or if ``page`` or ``per_page`` is less than 1, or if
            either are not ints.
        """
        p = type(self)(
            page=self.page + 1,
            per_page=self.per_page,
            max_per_page=self.max_per_page,
            error_out=error_out,
            count=False,
            **self._query_args,
        )
        p.total = self.total
        return p

    def iter_pages(
        self,
        *,
        left_edge: int = 2,
        left_current: int = 2,
        right_current: int = 4,
        right_edge: int = 2,
    ) -> t.Iterator[int | None]:
        """Yield page numbers for a pagination widget. Skipped pages between the edges
        and middle are represented by a ``None``.

        For example, if there are 20 pages and the current page is 7, the following
        values are yielded.

        .. code-block:: python

            1, 2, None, 5, 6, 7, 8, 9, 10, 11, None, 19, 20

        :param left_edge: How many pages to show from the first page.
        :param left_current: How many pages to show left of the current page.
        :param right_current: How many pages to show right of the current page.
        :param right_edge: How many pages to show from the last page.

        .. versionchanged:: 3.0
            Improved efficiency of calculating what to yield.

        .. versionchanged:: 3.0
            ``right_current`` boundary is inclusive.

        .. versionchanged:: 3.0
            All parameters are keyword-only.
        """
        pages_end = self.pages + 1

        if pages_end == 1:
            return

        left_end = min(1 + left_edge, pages_end)
        yield from range(1, left_end)

        if left_end == pages_end:
            return

        mid_start = max(left_end, self.page - left_current)
        mid_end = min(self.page + right_current + 1, pages_end)

        if mid_start - left_end > 0:
            yield None

        yield from range(mid_start, mid_end)

        if mid_end == pages_end:
            return

        right_start = max(mid_end, pages_end - right_edge)

        if right_start - mid_end > 0:
            yield None

        yield from range(right_start, pages_end)

    def __iter__(self) -> t.Iterator[t.Any]:
        yield from self.items


class SelectPagination(Pagination):
    """Returned by :meth:`.SQLAlchemy.paginate`. Takes ``select`` and ``session``
    arguments in addition to the :class:`Pagination` arguments.

    .. versionadded:: 3.0
    """

    def _query_items(self) -> list[t.Any]:
        select = self._query_args["select"]
        select = select.limit(self.per_page).offset(self._query_offset)
        session = self._query_args["session"]
        return list(session.execute(select).unique().scalars())

    def _query_count(self) -> int:
        select = self._query_args["select"]
        sub = select.options(sa_orm.lazyload("*")).order_by(None).subquery()
        session = self._query_args["session"]
        out = session.execute(sa.select(sa.func.count()).select_from(sub)).scalar()
        return out  # type: ignore[no-any-return]


class QueryPagination(Pagination):
    """Returned by :meth:`.Query.paginate`. Takes a ``query`` argument in addition to
    the :class:`Pagination` arguments.

    .. versionadded:: 3.0
    """

    def _query_items(self) -> list[t.Any]:
        query = self._query_args["query"]
        out = query.limit(self.per_page).offset(self._query_offset).all()
        return out  # type: ignore[no-any-return]

    def _query_count(self) -> int:
        # Query.count automatically disables eager loads
        out = self._query_args["query"].order_by(None).count()
        return out  # type: ignore[no-any-return]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\memmap.py ===
import operator
from contextlib import nullcontext

import numpy as np
from numpy._utils import set_module

from .numeric import dtype, ndarray, uint8

__all__ = ['memmap']

dtypedescr = dtype
valid_filemodes = ["r", "c", "r+", "w+"]
writeable_filemodes = ["r+", "w+"]

mode_equivalents = {
    "readonly": "r",
    "copyonwrite": "c",
    "readwrite": "r+",
    "write": "w+"
    }


@set_module('numpy')
class memmap(ndarray):
    """Create a memory-map to an array stored in a *binary* file on disk.

    Memory-mapped files are used for accessing small segments of large files
    on disk, without reading the entire file into memory.  NumPy's
    memmap's are array-like objects.  This differs from Python's ``mmap``
    module, which uses file-like objects.

    This subclass of ndarray has some unpleasant interactions with
    some operations, because it doesn't quite fit properly as a subclass.
    An alternative to using this subclass is to create the ``mmap``
    object yourself, then create an ndarray with ndarray.__new__ directly,
    passing the object created in its 'buffer=' parameter.

    This class may at some point be turned into a factory function
    which returns a view into an mmap buffer.

    Flush the memmap instance to write the changes to the file. Currently there
    is no API to close the underlying ``mmap``. It is tricky to ensure the
    resource is actually closed, since it may be shared between different
    memmap instances.


    Parameters
    ----------
    filename : str, file-like object, or pathlib.Path instance
        The file name or file object to be used as the array data buffer.
    dtype : data-type, optional
        The data-type used to interpret the file contents.
        Default is `uint8`.
    mode : {'r+', 'r', 'w+', 'c'}, optional
        The file is opened in this mode:

        +------+-------------------------------------------------------------+
        | 'r'  | Open existing file for reading only.                        |
        +------+-------------------------------------------------------------+
        | 'r+' | Open existing file for reading and writing.                 |
        +------+-------------------------------------------------------------+
        | 'w+' | Create or overwrite existing file for reading and writing.  |
        |      | If ``mode == 'w+'`` then `shape` must also be specified.    |
        +------+-------------------------------------------------------------+
        | 'c'  | Copy-on-write: assignments affect data in memory, but       |
        |      | changes are not saved to disk.  The file on disk is         |
        |      | read-only.                                                  |
        +------+-------------------------------------------------------------+

        Default is 'r+'.
    offset : int, optional
        In the file, array data starts at this offset. Since `offset` is
        measured in bytes, it should normally be a multiple of the byte-size
        of `dtype`. When ``mode != 'r'``, even positive offsets beyond end of
        file are valid; The file will be extended to accommodate the
        additional data. By default, ``memmap`` will start at the beginning of
        the file, even if ``filename`` is a file pointer ``fp`` and
        ``fp.tell() != 0``.
    shape : int or sequence of ints, optional
        The desired shape of the array. If ``mode == 'r'`` and the number
        of remaining bytes after `offset` is not a multiple of the byte-size
        of `dtype`, you must specify `shape`. By default, the returned array
        will be 1-D with the number of elements determined by file size
        and data-type.

        .. versionchanged:: 2.0
         The shape parameter can now be any integer sequence type, previously
         types were limited to tuple and int.

    order : {'C', 'F'}, optional
        Specify the order of the ndarray memory layout:
        :term:`row-major`, C-style or :term:`column-major`,
        Fortran-style.  This only has an effect if the shape is
        greater than 1-D.  The default order is 'C'.

    Attributes
    ----------
    filename : str or pathlib.Path instance
        Path to the mapped file.
    offset : int
        Offset position in the file.
    mode : str
        File mode.

    Methods
    -------
    flush
        Flush any changes in memory to file on disk.
        When you delete a memmap object, flush is called first to write
        changes to disk.


    See also
    --------
    lib.format.open_memmap : Create or load a memory-mapped ``.npy`` file.

    Notes
    -----
    The memmap object can be used anywhere an ndarray is accepted.
    Given a memmap ``fp``, ``isinstance(fp, numpy.ndarray)`` returns
    ``True``.

    Memory-mapped files cannot be larger than 2GB on 32-bit systems.

    When a memmap causes a file to be created or extended beyond its
    current size in the filesystem, the contents of the new part are
    unspecified. On systems with POSIX filesystem semantics, the extended
    part will be filled with zero bytes.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.arange(12, dtype='float32')
    >>> data.resize((3,4))

    This example uses a temporary file so that doctest doesn't write
    files to your directory. You would use a 'normal' filename.

    >>> from tempfile import mkdtemp
    >>> import os.path as path
    >>> filename = path.join(mkdtemp(), 'newfile.dat')

    Create a memmap with dtype and shape that matches our data:

    >>> fp = np.memmap(filename, dtype='float32', mode='w+', shape=(3,4))
    >>> fp
    memmap([[0., 0., 0., 0.],
            [0., 0., 0., 0.],
            [0., 0., 0., 0.]], dtype=float32)

    Write data to memmap array:

    >>> fp[:] = data[:]
    >>> fp
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)

    >>> fp.filename == path.abspath(filename)
    True

    Flushes memory changes to disk in order to read them back

    >>> fp.flush()

    Load the memmap and verify data was stored:

    >>> newfp = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> newfp
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)

    Read-only memmap:

    >>> fpr = np.memmap(filename, dtype='float32', mode='r', shape=(3,4))
    >>> fpr.flags.writeable
    False

    Copy-on-write memmap:

    >>> fpc = np.memmap(filename, dtype='float32', mode='c', shape=(3,4))
    >>> fpc.flags.writeable
    True

    It's possible to assign to copy-on-write array, but values are only
    written into the memory copy of the array, and not written to disk:

    >>> fpc
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)
    >>> fpc[0,:] = 0
    >>> fpc
    memmap([[  0.,   0.,   0.,   0.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)

    File on disk is unchanged:

    >>> fpr
    memmap([[  0.,   1.,   2.,   3.],
            [  4.,   5.,   6.,   7.],
            [  8.,   9.,  10.,  11.]], dtype=float32)

    Offset into a memmap:

    >>> fpo = np.memmap(filename, dtype='float32', mode='r', offset=16)
    >>> fpo
    memmap([  4.,   5.,   6.,   7.,   8.,   9.,  10.,  11.], dtype=float32)

    """

    __array_priority__ = -100.0

    def __new__(subtype, filename, dtype=uint8, mode='r+', offset=0,
                shape=None, order='C'):
        # Import here to minimize 'import numpy' overhead
        import mmap
        import os.path
        try:
            mode = mode_equivalents[mode]
        except KeyError as e:
            if mode not in valid_filemodes:
                all_modes = valid_filemodes + list(mode_equivalents.keys())
                raise ValueError(
                    f"mode must be one of {all_modes!r} (got {mode!r})"
                ) from None

        if mode == 'w+' and shape is None:
            raise ValueError("shape must be given if mode == 'w+'")

        if hasattr(filename, 'read'):
            f_ctx = nullcontext(filename)
        else:
            f_ctx = open(
                os.fspath(filename),
                ('r' if mode == 'c' else mode) + 'b'
            )

        with f_ctx as fid:
            fid.seek(0, 2)
            flen = fid.tell()
            descr = dtypedescr(dtype)
            _dbytes = descr.itemsize

            if shape is None:
                bytes = flen - offset
                if bytes % _dbytes:
                    raise ValueError("Size of available data is not a "
                            "multiple of the data-type size.")
                size = bytes // _dbytes
                shape = (size,)
            else:
                if not isinstance(shape, (tuple, list)):
                    try:
                        shape = [operator.index(shape)]
                    except TypeError:
                        pass
                shape = tuple(shape)
                size = np.intp(1)  # avoid overflows
                for k in shape:
                    size *= k

            bytes = int(offset + size * _dbytes)

            if mode in ('w+', 'r+'):
                # gh-27723
                # if bytes == 0, we write out 1 byte to allow empty memmap.
                bytes = max(bytes, 1)
                if flen < bytes:
                    fid.seek(bytes - 1, 0)
                    fid.write(b'\0')
                    fid.flush()

            if mode == 'c':
                acc = mmap.ACCESS_COPY
            elif mode == 'r':
                acc = mmap.ACCESS_READ
            else:
                acc = mmap.ACCESS_WRITE

            start = offset - offset % mmap.ALLOCATIONGRANULARITY
            bytes -= start
            # bytes == 0 is problematic as in mmap length=0 maps the full file.
            # See PR gh-27723 for a more detailed explanation.
            if bytes == 0 and start > 0:
                bytes += mmap.ALLOCATIONGRANULARITY
                start -= mmap.ALLOCATIONGRANULARITY
            array_offset = offset - start
            mm = mmap.mmap(fid.fileno(), bytes, access=acc, offset=start)

            self = ndarray.__new__(subtype, shape, dtype=descr, buffer=mm,
                                   offset=array_offset, order=order)
            self._mmap = mm
            self.offset = offset
            self.mode = mode

            if isinstance(filename, os.PathLike):
                # special case - if we were constructed with a pathlib.path,
                # then filename is a path object, not a string
                self.filename = filename.resolve()
            elif hasattr(fid, "name") and isinstance(fid.name, str):
                # py3 returns int for TemporaryFile().name
                self.filename = os.path.abspath(fid.name)
            # same as memmap copies (e.g. memmap + 1)
            else:
                self.filename = None

        return self

    def __array_finalize__(self, obj):
        if hasattr(obj, '_mmap') and np.may_share_memory(self, obj):
            self._mmap = obj._mmap
            self.filename = obj.filename
            self.offset = obj.offset
            self.mode = obj.mode
        else:
            self._mmap = None
            self.filename = None
            self.offset = None
            self.mode = None

    def flush(self):
        """
        Write any changes in the array to the file on disk.

        For further information, see `memmap`.

        Parameters
        ----------
        None

        See Also
        --------
        memmap

        """
        if self.base is not None and hasattr(self.base, 'flush'):
            self.base.flush()

    def __array_wrap__(self, arr, context=None, return_scalar=False):
        arr = super().__array_wrap__(arr, context)

        # Return a memmap if a memmap was given as the output of the
        # ufunc. Leave the arr class unchanged if self is not a memmap
        # to keep original memmap subclasses behavior
        if self is arr or type(self) is not memmap:
            return arr

        # Return scalar instead of 0d memmap, e.g. for np.sum with
        # axis=None (note that subclasses will not reach here)
        if return_scalar:
            return arr[()]

        # Return ndarray otherwise
        return arr.view(np.ndarray)

    def __getitem__(self, index):
        res = super().__getitem__(index)
        if type(res) is memmap and res._mmap is None:
            return res.view(type=ndarray)
        return res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\tensorproduct.py ===
"""Abstract tensor product."""

from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.kind import KindDispatcher
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.sympify import sympify
from sympy.matrices.dense import DenseMatrix as Matrix
from sympy.matrices.immutable import ImmutableDenseMatrix as ImmutableMatrix
from sympy.printing.pretty.stringpict import prettyForm
from sympy.utilities.exceptions import sympy_deprecation_warning

from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum.kind import (
    KetKind, _KetKind,
    BraKind, _BraKind,
    OperatorKind, _OperatorKind
)
from sympy.physics.quantum.matrixutils import (
    numpy_ndarray,
    scipy_sparse_matrix,
    matrix_tensor_product
)
from sympy.physics.quantum.state import Ket, Bra
from sympy.physics.quantum.trace import Tr


__all__ = [
    'TensorProduct',
    'tensor_product_simp'
]

#-----------------------------------------------------------------------------
# Tensor product
#-----------------------------------------------------------------------------

_combined_printing = False


def combined_tensor_printing(combined):
    """Set flag controlling whether tensor products of states should be
    printed as a combined bra/ket or as an explicit tensor product of different
    bra/kets. This is a global setting for all TensorProduct class instances.

    Parameters
    ----------
    combine : bool
        When true, tensor product states are combined into one ket/bra, and
        when false explicit tensor product notation is used between each
        ket/bra.
    """
    global _combined_printing
    _combined_printing = combined


class TensorProduct(Expr):
    """The tensor product of two or more arguments.

    For matrices, this uses ``matrix_tensor_product`` to compute the Kronecker
    or tensor product matrix. For other objects a symbolic ``TensorProduct``
    instance is returned. The tensor product is a non-commutative
    multiplication that is used primarily with operators and states in quantum
    mechanics.

    Currently, the tensor product distinguishes between commutative and
    non-commutative arguments.  Commutative arguments are assumed to be scalars
    and are pulled out in front of the ``TensorProduct``. Non-commutative
    arguments remain in the resulting ``TensorProduct``.

    Parameters
    ==========

    args : tuple
        A sequence of the objects to take the tensor product of.

    Examples
    ========

    Start with a simple tensor product of SymPy matrices::

        >>> from sympy import Matrix
        >>> from sympy.physics.quantum import TensorProduct

        >>> m1 = Matrix([[1,2],[3,4]])
        >>> m2 = Matrix([[1,0],[0,1]])
        >>> TensorProduct(m1, m2)
        Matrix([
        [1, 0, 2, 0],
        [0, 1, 0, 2],
        [3, 0, 4, 0],
        [0, 3, 0, 4]])
        >>> TensorProduct(m2, m1)
        Matrix([
        [1, 2, 0, 0],
        [3, 4, 0, 0],
        [0, 0, 1, 2],
        [0, 0, 3, 4]])

    We can also construct tensor products of non-commutative symbols:

        >>> from sympy import Symbol
        >>> A = Symbol('A',commutative=False)
        >>> B = Symbol('B',commutative=False)
        >>> tp = TensorProduct(A, B)
        >>> tp
        AxB

    We can take the dagger of a tensor product (note the order does NOT reverse
    like the dagger of a normal product):

        >>> from sympy.physics.quantum import Dagger
        >>> Dagger(tp)
        Dagger(A)xDagger(B)

    Expand can be used to distribute a tensor product across addition:

        >>> C = Symbol('C',commutative=False)
        >>> tp = TensorProduct(A+B,C)
        >>> tp
        (A + B)xC
        >>> tp.expand(tensorproduct=True)
        AxC + BxC
    """
    is_commutative = False

    _kind_dispatcher = KindDispatcher("TensorProduct_kind_dispatcher", commutative=True)

    @property
    def kind(self):
        """Calculate the kind of a tensor product by looking at its children."""
        arg_kinds = (a.kind for a in self.args)
        return self._kind_dispatcher(*arg_kinds)

    def __new__(cls, *args):
        if isinstance(args[0], (Matrix, ImmutableMatrix, numpy_ndarray,
                                                    scipy_sparse_matrix)):
            return matrix_tensor_product(*args)
        c_part, new_args = cls.flatten(sympify(args))
        c_part = Mul(*c_part)
        if len(new_args) == 0:
            return c_part
        elif len(new_args) == 1:
            return c_part * new_args[0]
        else:
            tp = Expr.__new__(cls, *new_args)
            return c_part * tp

    @classmethod
    def flatten(cls, args):
        # TODO: disallow nested TensorProducts.
        c_part = []
        nc_parts = []
        for arg in args:
            cp, ncp = arg.args_cnc()
            c_part.extend(list(cp))
            nc_parts.append(Mul._from_args(ncp))
        return c_part, nc_parts

    def _eval_adjoint(self):
        return TensorProduct(*[Dagger(i) for i in self.args])

    def _eval_rewrite(self, rule, args, **hints):
        return TensorProduct(*args).expand(tensorproduct=True)

    def _sympystr(self, printer, *args):
        length = len(self.args)
        s = ''
        for i in range(length):
            if isinstance(self.args[i], (Add, Pow, Mul)):
                s = s + '('
            s = s + printer._print(self.args[i])
            if isinstance(self.args[i], (Add, Pow, Mul)):
                s = s + ')'
            if i != length - 1:
                s = s + 'x'
        return s

    def _pretty(self, printer, *args):

        if (_combined_printing and
                (all(isinstance(arg, Ket) for arg in self.args) or
                 all(isinstance(arg, Bra) for arg in self.args))):

            length = len(self.args)
            pform = printer._print('', *args)
            for i in range(length):
                next_pform = printer._print('', *args)
                length_i = len(self.args[i].args)
                for j in range(length_i):
                    part_pform = printer._print(self.args[i].args[j], *args)
                    next_pform = prettyForm(*next_pform.right(part_pform))
                    if j != length_i - 1:
                        next_pform = prettyForm(*next_pform.right(', '))

                if len(self.args[i].args) > 1:
                    next_pform = prettyForm(
                        *next_pform.parens(left='{', right='}'))
                pform = prettyForm(*pform.right(next_pform))
                if i != length - 1:
                    pform = prettyForm(*pform.right(',' + ' '))

            pform = prettyForm(*pform.left(self.args[0].lbracket))
            pform = prettyForm(*pform.right(self.args[0].rbracket))
            return pform

        length = len(self.args)
        pform = printer._print('', *args)
        for i in range(length):
            next_pform = printer._print(self.args[i], *args)
            if isinstance(self.args[i], (Add, Mul)):
                next_pform = prettyForm(
                    *next_pform.parens(left='(', right=')')
                )
            pform = prettyForm(*pform.right(next_pform))
            if i != length - 1:
                if printer._use_unicode:
                    pform = prettyForm(*pform.right('\N{N-ARY CIRCLED TIMES OPERATOR}' + ' '))
                else:
                    pform = prettyForm(*pform.right('x' + ' '))
        return pform

    def _latex(self, printer, *args):

        if (_combined_printing and
                (all(isinstance(arg, Ket) for arg in self.args) or
                 all(isinstance(arg, Bra) for arg in self.args))):

            def _label_wrap(label, nlabels):
                return label if nlabels == 1 else r"\left\{%s\right\}" % label

            s = r", ".join([_label_wrap(arg._print_label_latex(printer, *args),
                                        len(arg.args)) for arg in self.args])

            return r"{%s%s%s}" % (self.args[0].lbracket_latex, s,
                                  self.args[0].rbracket_latex)

        length = len(self.args)
        s = ''
        for i in range(length):
            if isinstance(self.args[i], (Add, Mul)):
                s = s + '\\left('
            # The extra {} brackets are needed to get matplotlib's latex
            # rendered to render this properly.
            s = s + '{' + printer._print(self.args[i], *args) + '}'
            if isinstance(self.args[i], (Add, Mul)):
                s = s + '\\right)'
            if i != length - 1:
                s = s + '\\otimes '
        return s

    def doit(self, **hints):
        return TensorProduct(*[item.doit(**hints) for item in self.args])

    def _eval_expand_tensorproduct(self, **hints):
        """Distribute TensorProducts across addition."""
        args = self.args
        add_args = []
        for i in range(len(args)):
            if isinstance(args[i], Add):
                for aa in args[i].args:
                    tp = TensorProduct(*args[:i] + (aa,) + args[i + 1:])
                    c_part, nc_part = tp.args_cnc()
                    # Check for TensorProduct object: is the one object in nc_part, if any:
                    # (Note: any other object type to be expanded must be added here)
                    if len(nc_part) == 1 and isinstance(nc_part[0], TensorProduct):
                        nc_part = (nc_part[0]._eval_expand_tensorproduct(), )
                    add_args.append(Mul(*c_part)*Mul(*nc_part))
                break

        if add_args:
            return Add(*add_args)
        else:
            return self

    def _eval_trace(self, **kwargs):
        indices = kwargs.get('indices', None)
        exp = self

        if indices is None or len(indices) == 0:
            return Mul(*[Tr(arg).doit() for arg in exp.args])
        else:
            return Mul(*[Tr(value).doit() if idx in indices else value
                         for idx, value in enumerate(exp.args)])


def tensor_product_simp_Mul(e):
    """Simplify a Mul with tensor products.

    .. deprecated:: 1.14.
        The transformations applied by this function are not done automatically
        when tensor products are combined.

    Originally, the main use of this function is to simplify a ``Mul`` of
    ``TensorProduct``s to a ``TensorProduct`` of ``Muls``.
    """
    sympy_deprecation_warning(
        """
        tensor_product_simp_Mul has been deprecated. The transformations
        performed by this function are now done automatically when
        tensor products are multiplied.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-tensorproduct-simp'
    )
    return e

def tensor_product_simp_Pow(e):
    """Evaluates ``Pow`` expressions whose base is ``TensorProduct``

    .. deprecated:: 1.14.
        The transformations applied by this function are not done automatically
        when tensor products are combined.
    """
    sympy_deprecation_warning(
        """
        tensor_product_simp_Pow has been deprecated. The transformations
        performed by this function are now done automatically when
        tensor products are exponentiated.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-tensorproduct-simp'
    )
    return e


def tensor_product_simp(e, **hints):
    """Try to simplify and combine tensor products.

    .. deprecated:: 1.14.
        The transformations applied by this function are not done automatically
        when tensor products are combined.

    Originally, this function tried to pull expressions inside of ``TensorProducts``.
    It only worked for relatively simple cases where the products have
    only scalars, raw ``TensorProducts``, not ``Add``, ``Pow``, ``Commutators``
    of ``TensorProducts``.
    """
    sympy_deprecation_warning(
        """
        tensor_product_simp has been deprecated. The transformations
        performed by this function are now done automatically when
        tensor products are combined.
        """,
        deprecated_since_version="1.14",
        active_deprecations_target='deprecated-tensorproduct-simp'
    )
    return e


@TensorProduct._kind_dispatcher.register(_OperatorKind, _OperatorKind)
def find_op_kind(e1, e2):
    return OperatorKind


@TensorProduct._kind_dispatcher.register(_KetKind, _KetKind)
def find_ket_kind(e1, e2):
    return KetKind


@TensorProduct._kind_dispatcher.register(_BraKind, _BraKind)
def find_bra_kind(e1, e2):
    return BraKind

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\markers.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import operator
import os
import platform
import sys
from typing import AbstractSet, Any, Callable, Literal, TypedDict, Union, cast

from ._parser import MarkerAtom, MarkerList, Op, Value, Variable
from ._parser import parse_marker as _parse_marker
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name

__all__ = [
    "EvaluateContext",
    "InvalidMarker",
    "Marker",
    "UndefinedComparison",
    "UndefinedEnvironmentName",
    "default_environment",
]

Operator = Callable[[str, Union[str, AbstractSet[str]]], bool]
EvaluateContext = Literal["metadata", "lock_file", "requirement"]
MARKERS_ALLOWING_SET = {"extras", "dependency_groups"}


class InvalidMarker(ValueError):
    """
    An invalid marker was found, users should refer to PEP 508.
    """


class UndefinedComparison(ValueError):
    """
    An invalid operation was attempted on a value that doesn't support it.
    """


class UndefinedEnvironmentName(ValueError):
    """
    A name was attempted to be used that does not exist inside of the
    environment.
    """


class Environment(TypedDict):
    implementation_name: str
    """The implementation's identifier, e.g. ``'cpython'``."""

    implementation_version: str
    """
    The implementation's version, e.g. ``'3.13.0a2'`` for CPython 3.13.0a2, or
    ``'7.3.13'`` for PyPy3.10 v7.3.13.
    """

    os_name: str
    """
    The value of :py:data:`os.name`. The name of the operating system dependent module
    imported, e.g. ``'posix'``.
    """

    platform_machine: str
    """
    Returns the machine type, e.g. ``'i386'``.

    An empty string if the value cannot be determined.
    """

    platform_release: str
    """
    The system's release, e.g. ``'2.2.0'`` or ``'NT'``.

    An empty string if the value cannot be determined.
    """

    platform_system: str
    """
    The system/OS name, e.g. ``'Linux'``, ``'Windows'`` or ``'Java'``.

    An empty string if the value cannot be determined.
    """

    platform_version: str
    """
    The system's release version, e.g. ``'#3 on degas'``.

    An empty string if the value cannot be determined.
    """

    python_full_version: str
    """
    The Python version as string ``'major.minor.patchlevel'``.

    Note that unlike the Python :py:data:`sys.version`, this value will always include
    the patchlevel (it defaults to 0).
    """

    platform_python_implementation: str
    """
    A string identifying the Python implementation, e.g. ``'CPython'``.
    """

    python_version: str
    """The Python version as string ``'major.minor'``."""

    sys_platform: str
    """
    This string contains a platform identifier that can be used to append
    platform-specific components to :py:data:`sys.path`, for instance.

    For Unix systems, except on Linux and AIX, this is the lowercased OS name as
    returned by ``uname -s`` with the first part of the version as returned by
    ``uname -r`` appended, e.g. ``'sunos5'`` or ``'freebsd8'``, at the time when Python
    was built.
    """


def _normalize_extra_values(results: Any) -> Any:
    """
    Normalize extra values.
    """
    if isinstance(results[0], tuple):
        lhs, op, rhs = results[0]
        if isinstance(lhs, Variable) and lhs.value == "extra":
            normalized_extra = canonicalize_name(rhs.value)
            rhs = Value(normalized_extra)
        elif isinstance(rhs, Variable) and rhs.value == "extra":
            normalized_extra = canonicalize_name(lhs.value)
            lhs = Value(normalized_extra)
        results[0] = lhs, op, rhs
    return results


def _format_marker(
    marker: list[str] | MarkerAtom | str, first: bool | None = True
) -> str:
    assert isinstance(marker, (list, tuple, str))

    # Sometimes we have a structure like [[...]] which is a single item list
    # where the single item is itself it's own list. In that case we want skip
    # the rest of this function so that we don't get extraneous () on the
    # outside.
    if (
        isinstance(marker, list)
        and len(marker) == 1
        and isinstance(marker[0], (list, tuple))
    ):
        return _format_marker(marker[0])

    if isinstance(marker, list):
        inner = (_format_marker(m, first=False) for m in marker)
        if first:
            return " ".join(inner)
        else:
            return "(" + " ".join(inner) + ")"
    elif isinstance(marker, tuple):
        return " ".join([m.serialize() for m in marker])
    else:
        return marker


_operators: dict[str, Operator] = {
    "in": lambda lhs, rhs: lhs in rhs,
    "not in": lambda lhs, rhs: lhs not in rhs,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def _eval_op(lhs: str, op: Op, rhs: str | AbstractSet[str]) -> bool:
    if isinstance(rhs, str):
        try:
            spec = Specifier("".join([op.serialize(), rhs]))
        except InvalidSpecifier:
            pass
        else:
            return spec.contains(lhs, prereleases=True)

    oper: Operator | None = _operators.get(op.serialize())
    if oper is None:
        raise UndefinedComparison(f"Undefined {op!r} on {lhs!r} and {rhs!r}.")

    return oper(lhs, rhs)


def _normalize(
    lhs: str, rhs: str | AbstractSet[str], key: str
) -> tuple[str, str | AbstractSet[str]]:
    # PEP 685 – Comparison of extra names for optional distribution dependencies
    # https://peps.python.org/pep-0685/
    # > When comparing extra names, tools MUST normalize the names being
    # > compared using the semantics outlined in PEP 503 for names
    if key == "extra":
        assert isinstance(rhs, str), "extra value must be a string"
        return (canonicalize_name(lhs), canonicalize_name(rhs))
    if key in MARKERS_ALLOWING_SET:
        if isinstance(rhs, str):  # pragma: no cover
            return (canonicalize_name(lhs), canonicalize_name(rhs))
        else:
            return (canonicalize_name(lhs), {canonicalize_name(v) for v in rhs})

    # other environment markers don't have such standards
    return lhs, rhs


def _evaluate_markers(
    markers: MarkerList, environment: dict[str, str | AbstractSet[str]]
) -> bool:
    groups: list[list[bool]] = [[]]

    for marker in markers:
        assert isinstance(marker, (list, tuple, str))

        if isinstance(marker, list):
            groups[-1].append(_evaluate_markers(marker, environment))
        elif isinstance(marker, tuple):
            lhs, op, rhs = marker

            if isinstance(lhs, Variable):
                environment_key = lhs.value
                lhs_value = environment[environment_key]
                rhs_value = rhs.value
            else:
                lhs_value = lhs.value
                environment_key = rhs.value
                rhs_value = environment[environment_key]
            assert isinstance(lhs_value, str), "lhs must be a string"
            lhs_value, rhs_value = _normalize(lhs_value, rhs_value, key=environment_key)
            groups[-1].append(_eval_op(lhs_value, op, rhs_value))
        else:
            assert marker in ["and", "or"]
            if marker == "or":
                groups.append([])

    return any(all(item) for item in groups)


def format_full_version(info: sys._version_info) -> str:
    version = f"{info.major}.{info.minor}.{info.micro}"
    kind = info.releaselevel
    if kind != "final":
        version += kind[0] + str(info.serial)
    return version


def default_environment() -> Environment:
    iver = format_full_version(sys.implementation.version)
    implementation_name = sys.implementation.name
    return {
        "implementation_name": implementation_name,
        "implementation_version": iver,
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_full_version": platform.python_version(),
        "platform_python_implementation": platform.python_implementation(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
    }


class Marker:
    def __init__(self, marker: str) -> None:
        # Note: We create a Marker object without calling this constructor in
        #       packaging.requirements.Requirement. If any additional logic is
        #       added here, make sure to mirror/adapt Requirement.
        try:
            self._markers = _normalize_extra_values(_parse_marker(marker))
            # The attribute `_markers` can be described in terms of a recursive type:
            # MarkerList = List[Union[Tuple[Node, ...], str, MarkerList]]
            #
            # For example, the following expression:
            # python_version > "3.6" or (python_version == "3.6" and os_name == "unix")
            #
            # is parsed into:
            # [
            #     (<Variable('python_version')>, <Op('>')>, <Value('3.6')>),
            #     'and',
            #     [
            #         (<Variable('python_version')>, <Op('==')>, <Value('3.6')>),
            #         'or',
            #         (<Variable('os_name')>, <Op('==')>, <Value('unix')>)
            #     ]
            # ]
        except ParserSyntaxError as e:
            raise InvalidMarker(str(e)) from e

    def __str__(self) -> str:
        return _format_marker(self._markers)

    def __repr__(self) -> str:
        return f"<Marker('{self}')>"

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, str(self)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Marker):
            return NotImplemented

        return str(self) == str(other)

    def evaluate(
        self,
        environment: dict[str, str] | None = None,
        context: EvaluateContext = "metadata",
    ) -> bool:
        """Evaluate a marker.

        Return the boolean from evaluating the given marker against the
        environment. environment is an optional argument to override all or
        part of the determined environment. The *context* parameter specifies what
        context the markers are being evaluated for, which influences what markers
        are considered valid. Acceptable values are "metadata" (for core metadata;
        default), "lock_file", and "requirement" (i.e. all other situations).

        The environment is determined from the current Python process.
        """
        current_environment = cast(
            "dict[str, str | AbstractSet[str]]", default_environment()
        )
        if context == "lock_file":
            current_environment.update(
                extras=frozenset(), dependency_groups=frozenset()
            )
        elif context == "metadata":
            current_environment["extra"] = ""
        if environment is not None:
            current_environment.update(environment)
            # The API used to allow setting extra to None. We need to handle this
            # case for backwards compatibility.
            if "extra" in current_environment and current_environment["extra"] is None:
                current_environment["extra"] = ""

        return _evaluate_markers(
            self._markers, _repair_python_full_version(current_environment)
        )


def _repair_python_full_version(
    env: dict[str, str | AbstractSet[str]],
) -> dict[str, str | AbstractSet[str]]:
    """
    Work around platform.python_version() returning something that is not PEP 440
    compliant for non-tagged Python builds.
    """
    python_full_version = cast(str, env["python_full_version"])
    if python_full_version.endswith("+"):
        env["python_full_version"] = f"{python_full_version}local"
    return env

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\markers.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import operator
import os
import platform
import sys
from typing import AbstractSet, Any, Callable, Literal, TypedDict, Union, cast

from ._parser import MarkerAtom, MarkerList, Op, Value, Variable
from ._parser import parse_marker as _parse_marker
from ._tokenizer import ParserSyntaxError
from .specifiers import InvalidSpecifier, Specifier
from .utils import canonicalize_name

__all__ = [
    "EvaluateContext",
    "InvalidMarker",
    "Marker",
    "UndefinedComparison",
    "UndefinedEnvironmentName",
    "default_environment",
]

Operator = Callable[[str, Union[str, AbstractSet[str]]], bool]
EvaluateContext = Literal["metadata", "lock_file", "requirement"]
MARKERS_ALLOWING_SET = {"extras", "dependency_groups"}


class InvalidMarker(ValueError):
    """
    An invalid marker was found, users should refer to PEP 508.
    """


class UndefinedComparison(ValueError):
    """
    An invalid operation was attempted on a value that doesn't support it.
    """


class UndefinedEnvironmentName(ValueError):
    """
    A name was attempted to be used that does not exist inside of the
    environment.
    """


class Environment(TypedDict):
    implementation_name: str
    """The implementation's identifier, e.g. ``'cpython'``."""

    implementation_version: str
    """
    The implementation's version, e.g. ``'3.13.0a2'`` for CPython 3.13.0a2, or
    ``'7.3.13'`` for PyPy3.10 v7.3.13.
    """

    os_name: str
    """
    The value of :py:data:`os.name`. The name of the operating system dependent module
    imported, e.g. ``'posix'``.
    """

    platform_machine: str
    """
    Returns the machine type, e.g. ``'i386'``.

    An empty string if the value cannot be determined.
    """

    platform_release: str
    """
    The system's release, e.g. ``'2.2.0'`` or ``'NT'``.

    An empty string if the value cannot be determined.
    """

    platform_system: str
    """
    The system/OS name, e.g. ``'Linux'``, ``'Windows'`` or ``'Java'``.

    An empty string if the value cannot be determined.
    """

    platform_version: str
    """
    The system's release version, e.g. ``'#3 on degas'``.

    An empty string if the value cannot be determined.
    """

    python_full_version: str
    """
    The Python version as string ``'major.minor.patchlevel'``.

    Note that unlike the Python :py:data:`sys.version`, this value will always include
    the patchlevel (it defaults to 0).
    """

    platform_python_implementation: str
    """
    A string identifying the Python implementation, e.g. ``'CPython'``.
    """

    python_version: str
    """The Python version as string ``'major.minor'``."""

    sys_platform: str
    """
    This string contains a platform identifier that can be used to append
    platform-specific components to :py:data:`sys.path`, for instance.

    For Unix systems, except on Linux and AIX, this is the lowercased OS name as
    returned by ``uname -s`` with the first part of the version as returned by
    ``uname -r`` appended, e.g. ``'sunos5'`` or ``'freebsd8'``, at the time when Python
    was built.
    """


def _normalize_extra_values(results: Any) -> Any:
    """
    Normalize extra values.
    """
    if isinstance(results[0], tuple):
        lhs, op, rhs = results[0]
        if isinstance(lhs, Variable) and lhs.value == "extra":
            normalized_extra = canonicalize_name(rhs.value)
            rhs = Value(normalized_extra)
        elif isinstance(rhs, Variable) and rhs.value == "extra":
            normalized_extra = canonicalize_name(lhs.value)
            lhs = Value(normalized_extra)
        results[0] = lhs, op, rhs
    return results


def _format_marker(
    marker: list[str] | MarkerAtom | str, first: bool | None = True
) -> str:
    assert isinstance(marker, (list, tuple, str))

    # Sometimes we have a structure like [[...]] which is a single item list
    # where the single item is itself it's own list. In that case we want skip
    # the rest of this function so that we don't get extraneous () on the
    # outside.
    if (
        isinstance(marker, list)
        and len(marker) == 1
        and isinstance(marker[0], (list, tuple))
    ):
        return _format_marker(marker[0])

    if isinstance(marker, list):
        inner = (_format_marker(m, first=False) for m in marker)
        if first:
            return " ".join(inner)
        else:
            return "(" + " ".join(inner) + ")"
    elif isinstance(marker, tuple):
        return " ".join([m.serialize() for m in marker])
    else:
        return marker


_operators: dict[str, Operator] = {
    "in": lambda lhs, rhs: lhs in rhs,
    "not in": lambda lhs, rhs: lhs not in rhs,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
    ">=": operator.ge,
    ">": operator.gt,
}


def _eval_op(lhs: str, op: Op, rhs: str | AbstractSet[str]) -> bool:
    if isinstance(rhs, str):
        try:
            spec = Specifier("".join([op.serialize(), rhs]))
        except InvalidSpecifier:
            pass
        else:
            return spec.contains(lhs, prereleases=True)

    oper: Operator | None = _operators.get(op.serialize())
    if oper is None:
        raise UndefinedComparison(f"Undefined {op!r} on {lhs!r} and {rhs!r}.")

    return oper(lhs, rhs)


def _normalize(
    lhs: str, rhs: str | AbstractSet[str], key: str
) -> tuple[str, str | AbstractSet[str]]:
    # PEP 685 – Comparison of extra names for optional distribution dependencies
    # https://peps.python.org/pep-0685/
    # > When comparing extra names, tools MUST normalize the names being
    # > compared using the semantics outlined in PEP 503 for names
    if key == "extra":
        assert isinstance(rhs, str), "extra value must be a string"
        return (canonicalize_name(lhs), canonicalize_name(rhs))
    if key in MARKERS_ALLOWING_SET:
        if isinstance(rhs, str):  # pragma: no cover
            return (canonicalize_name(lhs), canonicalize_name(rhs))
        else:
            return (canonicalize_name(lhs), {canonicalize_name(v) for v in rhs})

    # other environment markers don't have such standards
    return lhs, rhs


def _evaluate_markers(
    markers: MarkerList, environment: dict[str, str | AbstractSet[str]]
) -> bool:
    groups: list[list[bool]] = [[]]

    for marker in markers:
        assert isinstance(marker, (list, tuple, str))

        if isinstance(marker, list):
            groups[-1].append(_evaluate_markers(marker, environment))
        elif isinstance(marker, tuple):
            lhs, op, rhs = marker

            if isinstance(lhs, Variable):
                environment_key = lhs.value
                lhs_value = environment[environment_key]
                rhs_value = rhs.value
            else:
                lhs_value = lhs.value
                environment_key = rhs.value
                rhs_value = environment[environment_key]
            assert isinstance(lhs_value, str), "lhs must be a string"
            lhs_value, rhs_value = _normalize(lhs_value, rhs_value, key=environment_key)
            groups[-1].append(_eval_op(lhs_value, op, rhs_value))
        else:
            assert marker in ["and", "or"]
            if marker == "or":
                groups.append([])

    return any(all(item) for item in groups)


def format_full_version(info: sys._version_info) -> str:
    version = f"{info.major}.{info.minor}.{info.micro}"
    kind = info.releaselevel
    if kind != "final":
        version += kind[0] + str(info.serial)
    return version


def default_environment() -> Environment:
    iver = format_full_version(sys.implementation.version)
    implementation_name = sys.implementation.name
    return {
        "implementation_name": implementation_name,
        "implementation_version": iver,
        "os_name": os.name,
        "platform_machine": platform.machine(),
        "platform_release": platform.release(),
        "platform_system": platform.system(),
        "platform_version": platform.version(),
        "python_full_version": platform.python_version(),
        "platform_python_implementation": platform.python_implementation(),
        "python_version": ".".join(platform.python_version_tuple()[:2]),
        "sys_platform": sys.platform,
    }


class Marker:
    def __init__(self, marker: str) -> None:
        # Note: We create a Marker object without calling this constructor in
        #       packaging.requirements.Requirement. If any additional logic is
        #       added here, make sure to mirror/adapt Requirement.
        try:
            self._markers = _normalize_extra_values(_parse_marker(marker))
            # The attribute `_markers` can be described in terms of a recursive type:
            # MarkerList = List[Union[Tuple[Node, ...], str, MarkerList]]
            #
            # For example, the following expression:
            # python_version > "3.6" or (python_version == "3.6" and os_name == "unix")
            #
            # is parsed into:
            # [
            #     (<Variable('python_version')>, <Op('>')>, <Value('3.6')>),
            #     'and',
            #     [
            #         (<Variable('python_version')>, <Op('==')>, <Value('3.6')>),
            #         'or',
            #         (<Variable('os_name')>, <Op('==')>, <Value('unix')>)
            #     ]
            # ]
        except ParserSyntaxError as e:
            raise InvalidMarker(str(e)) from e

    def __str__(self) -> str:
        return _format_marker(self._markers)

    def __repr__(self) -> str:
        return f"<Marker('{self}')>"

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, str(self)))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Marker):
            return NotImplemented

        return str(self) == str(other)

    def evaluate(
        self,
        environment: dict[str, str] | None = None,
        context: EvaluateContext = "metadata",
    ) -> bool:
        """Evaluate a marker.

        Return the boolean from evaluating the given marker against the
        environment. environment is an optional argument to override all or
        part of the determined environment. The *context* parameter specifies what
        context the markers are being evaluated for, which influences what markers
        are considered valid. Acceptable values are "metadata" (for core metadata;
        default), "lock_file", and "requirement" (i.e. all other situations).

        The environment is determined from the current Python process.
        """
        current_environment = cast(
            "dict[str, str | AbstractSet[str]]", default_environment()
        )
        if context == "lock_file":
            current_environment.update(
                extras=frozenset(), dependency_groups=frozenset()
            )
        elif context == "metadata":
            current_environment["extra"] = ""
        if environment is not None:
            current_environment.update(environment)
            # The API used to allow setting extra to None. We need to handle this
            # case for backwards compatibility.
            if "extra" in current_environment and current_environment["extra"] is None:
                current_environment["extra"] = ""

        return _evaluate_markers(
            self._markers, _repair_python_full_version(current_environment)
        )


def _repair_python_full_version(
    env: dict[str, str | AbstractSet[str]],
) -> dict[str, str | AbstractSet[str]]:
    """
    Work around platform.python_version() returning something that is not PEP 440
    compliant for non-tagged Python builds.
    """
    python_full_version = cast(str, env["python_full_version"])
    if python_full_version.endswith("+"):
        env["python_full_version"] = f"{python_full_version}local"
    return env

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\lexers\__init__.py ===
"""
    pygments.lexers
    ~~~~~~~~~~~~~~~

    Pygments lexers.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re
import sys
import types
import fnmatch
from os.path import basename

from pip._vendor.pygments.lexers._mapping import LEXERS
from pip._vendor.pygments.modeline import get_filetype_from_buffer
from pip._vendor.pygments.plugin import find_plugin_lexers
from pip._vendor.pygments.util import ClassNotFound, guess_decode

COMPAT = {
    'Python3Lexer': 'PythonLexer',
    'Python3TracebackLexer': 'PythonTracebackLexer',
    'LeanLexer': 'Lean3Lexer',
}

__all__ = ['get_lexer_by_name', 'get_lexer_for_filename', 'find_lexer_class',
           'guess_lexer', 'load_lexer_from_file'] + list(LEXERS) + list(COMPAT)

_lexer_cache = {}
_pattern_cache = {}


def _fn_matches(fn, glob):
    """Return whether the supplied file name fn matches pattern filename."""
    if glob not in _pattern_cache:
        pattern = _pattern_cache[glob] = re.compile(fnmatch.translate(glob))
        return pattern.match(fn)
    return _pattern_cache[glob].match(fn)


def _load_lexers(module_name):
    """Load a lexer (and all others in the module too)."""
    mod = __import__(module_name, None, None, ['__all__'])
    for lexer_name in mod.__all__:
        cls = getattr(mod, lexer_name)
        _lexer_cache[cls.name] = cls


def get_all_lexers(plugins=True):
    """Return a generator of tuples in the form ``(name, aliases,
    filenames, mimetypes)`` of all know lexers.

    If *plugins* is true (the default), plugin lexers supplied by entrypoints
    are also returned.  Otherwise, only builtin ones are considered.
    """
    for item in LEXERS.values():
        yield item[1:]
    if plugins:
        for lexer in find_plugin_lexers():
            yield lexer.name, lexer.aliases, lexer.filenames, lexer.mimetypes


def find_lexer_class(name):
    """
    Return the `Lexer` subclass that with the *name* attribute as given by
    the *name* argument.
    """
    if name in _lexer_cache:
        return _lexer_cache[name]
    # lookup builtin lexers
    for module_name, lname, aliases, _, _ in LEXERS.values():
        if name == lname:
            _load_lexers(module_name)
            return _lexer_cache[name]
    # continue with lexers from setuptools entrypoints
    for cls in find_plugin_lexers():
        if cls.name == name:
            return cls


def find_lexer_class_by_name(_alias):
    """
    Return the `Lexer` subclass that has `alias` in its aliases list, without
    instantiating it.

    Like `get_lexer_by_name`, but does not instantiate the class.

    Will raise :exc:`pygments.util.ClassNotFound` if no lexer with that alias is
    found.

    .. versionadded:: 2.2
    """
    if not _alias:
        raise ClassNotFound(f'no lexer for alias {_alias!r} found')
    # lookup builtin lexers
    for module_name, name, aliases, _, _ in LEXERS.values():
        if _alias.lower() in aliases:
            if name not in _lexer_cache:
                _load_lexers(module_name)
            return _lexer_cache[name]
    # continue with lexers from setuptools entrypoints
    for cls in find_plugin_lexers():
        if _alias.lower() in cls.aliases:
            return cls
    raise ClassNotFound(f'no lexer for alias {_alias!r} found')


def get_lexer_by_name(_alias, **options):
    """
    Return an instance of a `Lexer` subclass that has `alias` in its
    aliases list. The lexer is given the `options` at its
    instantiation.

    Will raise :exc:`pygments.util.ClassNotFound` if no lexer with that alias is
    found.
    """
    if not _alias:
        raise ClassNotFound(f'no lexer for alias {_alias!r} found')

    # lookup builtin lexers
    for module_name, name, aliases, _, _ in LEXERS.values():
        if _alias.lower() in aliases:
            if name not in _lexer_cache:
                _load_lexers(module_name)
            return _lexer_cache[name](**options)
    # continue with lexers from setuptools entrypoints
    for cls in find_plugin_lexers():
        if _alias.lower() in cls.aliases:
            return cls(**options)
    raise ClassNotFound(f'no lexer for alias {_alias!r} found')


def load_lexer_from_file(filename, lexername="CustomLexer", **options):
    """Load a lexer from a file.

    This method expects a file located relative to the current working
    directory, which contains a Lexer class. By default, it expects the
    Lexer to be name CustomLexer; you can specify your own class name
    as the second argument to this function.

    Users should be very careful with the input, because this method
    is equivalent to running eval on the input file.

    Raises ClassNotFound if there are any problems importing the Lexer.

    .. versionadded:: 2.2
    """
    try:
        # This empty dict will contain the namespace for the exec'd file
        custom_namespace = {}
        with open(filename, 'rb') as f:
            exec(f.read(), custom_namespace)
        # Retrieve the class `lexername` from that namespace
        if lexername not in custom_namespace:
            raise ClassNotFound(f'no valid {lexername} class found in {filename}')
        lexer_class = custom_namespace[lexername]
        # And finally instantiate it with the options
        return lexer_class(**options)
    except OSError as err:
        raise ClassNotFound(f'cannot read {filename}: {err}')
    except ClassNotFound:
        raise
    except Exception as err:
        raise ClassNotFound(f'error when loading custom lexer: {err}')


def find_lexer_class_for_filename(_fn, code=None):
    """Get a lexer for a filename.

    If multiple lexers match the filename pattern, use ``analyse_text()`` to
    figure out which one is more appropriate.

    Returns None if not found.
    """
    matches = []
    fn = basename(_fn)
    for modname, name, _, filenames, _ in LEXERS.values():
        for filename in filenames:
            if _fn_matches(fn, filename):
                if name not in _lexer_cache:
                    _load_lexers(modname)
                matches.append((_lexer_cache[name], filename))
    for cls in find_plugin_lexers():
        for filename in cls.filenames:
            if _fn_matches(fn, filename):
                matches.append((cls, filename))

    if isinstance(code, bytes):
        # decode it, since all analyse_text functions expect unicode
        code = guess_decode(code)

    def get_rating(info):
        cls, filename = info
        # explicit patterns get a bonus
        bonus = '*' not in filename and 0.5 or 0
        # The class _always_ defines analyse_text because it's included in
        # the Lexer class.  The default implementation returns None which
        # gets turned into 0.0.  Run scripts/detect_missing_analyse_text.py
        # to find lexers which need it overridden.
        if code:
            return cls.analyse_text(code) + bonus, cls.__name__
        return cls.priority + bonus, cls.__name__

    if matches:
        matches.sort(key=get_rating)
        # print "Possible lexers, after sort:", matches
        return matches[-1][0]


def get_lexer_for_filename(_fn, code=None, **options):
    """Get a lexer for a filename.

    Return a `Lexer` subclass instance that has a filename pattern
    matching `fn`. The lexer is given the `options` at its
    instantiation.

    Raise :exc:`pygments.util.ClassNotFound` if no lexer for that filename
    is found.

    If multiple lexers match the filename pattern, use their ``analyse_text()``
    methods to figure out which one is more appropriate.
    """
    res = find_lexer_class_for_filename(_fn, code)
    if not res:
        raise ClassNotFound(f'no lexer for filename {_fn!r} found')
    return res(**options)


def get_lexer_for_mimetype(_mime, **options):
    """
    Return a `Lexer` subclass instance that has `mime` in its mimetype
    list. The lexer is given the `options` at its instantiation.

    Will raise :exc:`pygments.util.ClassNotFound` if not lexer for that mimetype
    is found.
    """
    for modname, name, _, _, mimetypes in LEXERS.values():
        if _mime in mimetypes:
            if name not in _lexer_cache:
                _load_lexers(modname)
            return _lexer_cache[name](**options)
    for cls in find_plugin_lexers():
        if _mime in cls.mimetypes:
            return cls(**options)
    raise ClassNotFound(f'no lexer for mimetype {_mime!r} found')


def _iter_lexerclasses(plugins=True):
    """Return an iterator over all lexer classes."""
    for key in sorted(LEXERS):
        module_name, name = LEXERS[key][:2]
        if name not in _lexer_cache:
            _load_lexers(module_name)
        yield _lexer_cache[name]
    if plugins:
        yield from find_plugin_lexers()


def guess_lexer_for_filename(_fn, _text, **options):
    """
    As :func:`guess_lexer()`, but only lexers which have a pattern in `filenames`
    or `alias_filenames` that matches `filename` are taken into consideration.

    :exc:`pygments.util.ClassNotFound` is raised if no lexer thinks it can
    handle the content.
    """
    fn = basename(_fn)
    primary = {}
    matching_lexers = set()
    for lexer in _iter_lexerclasses():
        for filename in lexer.filenames:
            if _fn_matches(fn, filename):
                matching_lexers.add(lexer)
                primary[lexer] = True
        for filename in lexer.alias_filenames:
            if _fn_matches(fn, filename):
                matching_lexers.add(lexer)
                primary[lexer] = False
    if not matching_lexers:
        raise ClassNotFound(f'no lexer for filename {fn!r} found')
    if len(matching_lexers) == 1:
        return matching_lexers.pop()(**options)
    result = []
    for lexer in matching_lexers:
        rv = lexer.analyse_text(_text)
        if rv == 1.0:
            return lexer(**options)
        result.append((rv, lexer))

    def type_sort(t):
        # sort by:
        # - analyse score
        # - is primary filename pattern?
        # - priority
        # - last resort: class name
        return (t[0], primary[t[1]], t[1].priority, t[1].__name__)
    result.sort(key=type_sort)

    return result[-1][1](**options)


def guess_lexer(_text, **options):
    """
    Return a `Lexer` subclass instance that's guessed from the text in
    `text`. For that, the :meth:`.analyse_text()` method of every known lexer
    class is called with the text as argument, and the lexer which returned the
    highest value will be instantiated and returned.

    :exc:`pygments.util.ClassNotFound` is raised if no lexer thinks it can
    handle the content.
    """

    if not isinstance(_text, str):
        inencoding = options.get('inencoding', options.get('encoding'))
        if inencoding:
            _text = _text.decode(inencoding or 'utf8')
        else:
            _text, _ = guess_decode(_text)

    # try to get a vim modeline first
    ft = get_filetype_from_buffer(_text)

    if ft is not None:
        try:
            return get_lexer_by_name(ft, **options)
        except ClassNotFound:
            pass

    best_lexer = [0.0, None]
    for lexer in _iter_lexerclasses():
        rv = lexer.analyse_text(_text)
        if rv == 1.0:
            return lexer(**options)
        if rv > best_lexer[0]:
            best_lexer[:] = (rv, lexer)
    if not best_lexer[0] or best_lexer[1] is None:
        raise ClassNotFound('no lexer matching the text found')
    return best_lexer[1](**options)


class _automodule(types.ModuleType):
    """Automatically import lexers."""

    def __getattr__(self, name):
        info = LEXERS.get(name)
        if info:
            _load_lexers(info[0])
            cls = _lexer_cache[info[1]]
            setattr(self, name, cls)
            return cls
        if name in COMPAT:
            return getattr(self, COMPAT[name])
        raise AttributeError(name)


oldmod = sys.modules[__name__]
newmod = _automodule(__name__)
newmod.__dict__.update(oldmod.__dict__)
sys.modules[__name__] = newmod
del newmod.newmod, newmod.oldmod, newmod.sys, newmod.types

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\t5\t5_encoder_decoder_init.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import logging
import os
import tempfile
from pathlib import Path

import numpy
import onnx
import torch
from onnx_model import OnnxModel
from past_helper import PastKeyValuesHelper
from t5_decoder import T5DecoderInit
from t5_encoder import T5Encoder, T5EncoderInputs
from torch_onnx_export_helper import torch_onnx_export
from transformers import MT5Config, T5Config

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


class T5EncoderDecoderInit(torch.nn.Module):
    """A combination of T5Encoder and T5DecoderInit."""

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        lm_head: torch.nn.Linear,
        config: T5Config | MT5Config,
        decoder_start_token_id: int | None = None,
        output_cross_only: bool = False,
    ):
        super().__init__()
        self.config: T5Config | MT5Config = config
        self.t5_encoder = T5Encoder(encoder, config)
        self.t5_decoder_init = T5DecoderInit(decoder, lm_head, config, decoder_start_token_id)
        self.output_cross_only = output_cross_only

    def forward(
        self,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor | None = None,
    ):
        encoder_hidden_states: torch.FloatTensor = self.t5_encoder(encoder_input_ids, encoder_attention_mask)

        lm_logits, past_self, past_cross = self.t5_decoder_init(
            decoder_input_ids, encoder_attention_mask, encoder_hidden_states
        )

        if self.output_cross_only:
            return past_cross
        else:
            return lm_logits, encoder_hidden_states, past_self, past_cross


class T5EncoderDecoderInitInputs:
    def __init__(self, encoder_input_ids, encoder_attention_mask, decoder_input_ids=None):
        self.encoder_input_ids: torch.LongTensor = encoder_input_ids
        self.encoder_attention_mask: torch.LongTensor = encoder_attention_mask
        self.decoder_input_ids: torch.LongTensor | None = decoder_input_ids

    @staticmethod
    def create_dummy(
        config: T5Config | MT5Config,
        batch_size: int,
        encode_sequence_length: int,
        use_decoder_input_ids: int,
        device: torch.device,
        use_int32_inputs: bool = False,
    ):  # -> T5EncoderDecoderInitInputs:
        encoder_inputs: T5EncoderInputs = T5EncoderInputs.create_dummy(
            batch_size,
            encode_sequence_length,
            config.vocab_size,
            device,
            use_int32_inputs=use_int32_inputs,
        )
        decoder_input_ids = None
        if use_decoder_input_ids:
            dtype = torch.int32 if use_int32_inputs else torch.int64
            decoder_input_ids = torch.ones((batch_size, 1), dtype=dtype, device=device) * config.decoder_start_token_id

        return T5EncoderDecoderInitInputs(encoder_inputs.input_ids, encoder_inputs.attention_mask, decoder_input_ids)

    def to_list(self) -> list:
        input_list = [self.encoder_input_ids, self.encoder_attention_mask]
        if self.decoder_input_ids is not None:
            input_list.append(self.decoder_input_ids)
        return input_list


class T5EncoderDecoderInitHelper:
    @staticmethod
    def export_onnx(
        model: T5EncoderDecoderInit,
        device: torch.device,
        onnx_model_path: str,
        use_decoder_input_ids: bool = True,
        verbose: bool = True,
        use_external_data_format: bool = False,
        use_int32_inputs: bool = False,
    ):
        """Export decoder to ONNX

        Args:
            model (T5EncoderDecoderInit): the model to export
            device (torch.device): device of decoder object
            onnx_model_path (str): onnx path
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
            use_int32_inputs (bool, optional): use int32 instead of int64 for integer inputs. Defaults to False.
        """
        assert isinstance(model, T5EncoderDecoderInit)

        # Do not exclude decoder in torch onnx export so that cross can show up.
        output_cross_only = model.output_cross_only
        model.output_cross_only = False

        inputs = T5EncoderDecoderInitInputs.create_dummy(
            model.config,
            batch_size=2,
            encode_sequence_length=3,
            use_decoder_input_ids=use_decoder_input_ids,
            device=device,
            use_int32_inputs=use_int32_inputs,
        )
        input_list = inputs.to_list()

        present_names = PastKeyValuesHelper.get_past_names(model.config.num_decoder_layers, present=True)

        output_names = ["logits", "encoder_hidden_states", *present_names]

        # Shape of input tensors (sequence_length==1):
        #    input_ids: (batch_size, sequence_length)
        #    encoder_attention_mask: (batch_size, encode_sequence_length)
        #    encoder_hidden_states: (batch_size, encode_sequence_length, hidden_size)
        #    past_self_*: (batch_size, num_heads, past_decode_sequence_length, head_size)
        #    past_cross_*: (batch_size, num_heads, encode_sequence_length, head_size)

        # Shape of output tensors:
        #    logits: (batch_size, sequence_length, vocab_size)
        #    past_self_*: (batch_size, num_heads, past_decode_sequence_length + sequence_length, head_size)
        #    past_cross_*: (batch_size, num_heads, encode_sequence_length, head_size)

        input_names = ["encoder_input_ids", "encoder_attention_mask"]

        # ONNX exporter might mark dimension like 'present_value_self_1_dim_2' in shape inference.
        # We use a workaround here: first use dim_param "1" for sequence_length, and later change to dim_value.
        sequence_length = "1"
        num_heads = str(model.config.num_heads)
        hidden_size = str(model.config.d_model)
        head_size = str(model.config.d_kv)

        dynamic_axes = {
            "encoder_input_ids": {0: "batch_size", 1: "encode_sequence_length"},
            "encoder_attention_mask": {0: "batch_size", 1: "encode_sequence_length"},
            "encoder_hidden_states": {
                0: "batch_size",
                1: "encode_sequence_length",
                2: hidden_size,
            },
            "logits": {
                0: "batch_size",
                1: sequence_length,
            },
        }

        if use_decoder_input_ids:
            input_names.append("decoder_input_ids")
            dynamic_axes["decoder_input_ids"] = {
                0: "batch_size",
                1: sequence_length,
            }

        for name in present_names:
            if "cross" in name:
                dynamic_axes[name] = {
                    0: "batch_size",
                    1: num_heads,
                    2: "encode_sequence_length",
                    3: head_size,
                }

            else:  # self attention past state
                dynamic_axes[name] = {
                    0: "batch_size",
                    1: num_heads,
                    2: sequence_length,
                    3: head_size,
                }

        with tempfile.TemporaryDirectory() as tmp_dir_name:
            temp_onnx_model_path = os.path.join(tmp_dir_name, "encoder_decoder_init.onnx")
            Path(temp_onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
            torch_onnx_export(
                model,
                args=tuple(input_list),
                f=temp_onnx_model_path,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=12,
                do_constant_folding=True,
                use_external_data_format=use_external_data_format,
                verbose=verbose,
            )

            # Restore output_cross_only setting.
            model.output_cross_only = output_cross_only

            # Workaround as mentioned earlier: change numeric dim_param to dim_value
            exported_model: onnx.ModelProto = onnx.load(temp_onnx_model_path)
            for tensor in exported_model.graph.output:
                for dim_proto in tensor.type.tensor_type.shape.dim:
                    if dim_proto.HasField("dim_param") and dim_proto.dim_param in [
                        sequence_length,
                        num_heads,
                        hidden_size,
                        head_size,
                    ]:
                        dim_value = int(dim_proto.dim_param)
                        dim_proto.Clear()
                        dim_proto.dim_value = dim_value

            if output_cross_only:
                # Rewrite onnx graph to only keep present_[key|value]_cross_* outputs.
                onnx_model = OnnxModel(exported_model)
                output_name_to_node = onnx_model.output_name_to_node()

                for output in exported_model.graph.output:
                    if "cross" in output.name:
                        assert output.name in output_name_to_node

                        transpose_node = output_name_to_node[output.name]
                        assert transpose_node and transpose_node.op_type == "Transpose"

                        permutation = OnnxModel.get_node_attribute(transpose_node, "perm")
                        assert isinstance(permutation, list)
                        assert permutation == [0, 2, 1, 3]

                        matched_nodes = onnx_model.match_parent_path(
                            transpose_node,
                            ["Reshape", "MatMul"],
                            [0, 0],
                            output_name_to_node,
                        )
                        assert matched_nodes is not None

                        reshape_node, matmul_node = matched_nodes
                        assert "encoder_hidden_states" in matmul_node.input

                        if not onnx_model.get_initializer("cross_reshape_shape"):
                            shape_tensor = onnx.helper.make_tensor(
                                name="cross_reshape_shape",
                                data_type=onnx.TensorProto.INT64,
                                dims=[4],
                                vals=[0, 0, int(num_heads), int(head_size)],
                                raw=False,
                            )
                            onnx_model.add_initializer(shape_tensor)

                        reshape_node.input[1] = "cross_reshape_shape"

                cross_outputs = [output.name for output in exported_model.graph.output if "cross" in output.name]
                onnx_model.prune_graph(cross_outputs, allow_remove_graph_inputs=True)

            OnnxModel.save(
                exported_model,
                onnx_model_path,
                save_as_external_data=use_external_data_format,
                all_tensors_to_one_file=True,
            )

    @staticmethod
    def onnxruntime_inference(ort_session, inputs: T5EncoderDecoderInitInputs):
        """Run inference of ONNX model."""
        logger.debug("start onnxruntime_inference")

        ort_inputs = {
            "encoder_input_ids": numpy.ascontiguousarray(inputs.encoder_input_ids.cpu().numpy()),
            "encoder_attention_mask": numpy.ascontiguousarray(inputs.encoder_attention_mask.cpu().numpy()),
        }
        if inputs.decoder_input_ids is not None:
            ort_inputs["decoder_input_ids"] = numpy.ascontiguousarray(inputs.decoder_input_ids.cpu().numpy())

        ort_outputs = ort_session.run(None, ort_inputs)
        return ort_outputs

    @staticmethod
    def verify_onnx(
        model: T5EncoderDecoderInit,
        ort_session: InferenceSession,
        device: torch.device,
        use_int32_inputs: bool,
        max_cases: int = 4,
    ):
        """Compare the result from PyTorch and OnnxRuntime to verify the ONNX model is good."""
        ort_inputs = ort_session.get_inputs()
        use_decoder_input_ids = len(ort_inputs) == 3

        test_cases = [(4, 11), (1, 2), (3, 1), (8, 5)]
        test_cases_max_diff = []
        for batch_size, encode_sequence_length in test_cases[:max_cases]:
            inputs = T5EncoderDecoderInitInputs.create_dummy(
                model.config,
                batch_size,
                encode_sequence_length,
                use_decoder_input_ids=use_decoder_input_ids,
                device=device,
                use_int32_inputs=use_int32_inputs,
            )

            ort_outputs = T5EncoderDecoderInitHelper.onnxruntime_inference(ort_session, inputs)

            # Run inference of PyTorch model
            input_list = inputs.to_list()
            torch_outputs = model(*input_list)

            num_decoder_layers = model.config.num_decoder_layers

            if not model.output_cross_only:
                assert torch_outputs[0].cpu().numpy().shape == ort_outputs[0].shape
                max_diff = numpy.amax(numpy.abs(torch_outputs[0].cpu().numpy() - ort_outputs[0]))
                logger.debug(f"logits max_diff={max_diff}")
                max_diff_all = max_diff

                assert torch_outputs[1].cpu().numpy().shape == ort_outputs[1].shape
                max_diff = numpy.amax(numpy.abs(torch_outputs[1].cpu().numpy() - ort_outputs[1]))
                logger.debug(f"encoder_hidden_states max_diff={max_diff}")
                max_diff_all = max(max_diff_all, max_diff)

                for i in range(2 * num_decoder_layers):
                    max_diff = numpy.amax(numpy.abs(torch_outputs[2][i].cpu().numpy() - ort_outputs[2 + i]))
                    logger.debug(f"self attention past state {i} max_diff={max_diff}")

                for i in range(2 * num_decoder_layers):
                    max_diff = numpy.amax(
                        numpy.abs(torch_outputs[3][i].cpu().numpy() - ort_outputs[2 + 2 * num_decoder_layers + i])
                    )
                    logger.debug(f"cross attention past state {i} max_diff={max_diff}")
                    max_diff_all = max(max_diff_all, max_diff)
            else:
                max_diff_all = -float("inf")
                for i in range(2 * num_decoder_layers):
                    max_diff = numpy.amax(numpy.abs(torch_outputs[i].cpu().numpy() - ort_outputs[i]))
                    logger.debug(f"cross attention past state {i} max_diff={max_diff}")
                    max_diff_all = max(max_diff_all, max_diff)

            test_cases_max_diff.append(max_diff_all)
            logger.info(
                f"batch_size={batch_size} encode_sequence_length={encode_sequence_length}, max_diff={max_diff_all}"
            )

        return max(test_cases_max_diff)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\utils\logging.py ===
import contextlib
import errno
import logging
import logging.handlers
import os
import sys
import threading
from dataclasses import dataclass
from io import TextIOWrapper
from logging import Filter
from typing import Any, ClassVar, Generator, List, Optional, Type

from pip._vendor.rich.console import (
    Console,
    ConsoleOptions,
    ConsoleRenderable,
    RenderableType,
    RenderResult,
    RichCast,
)
from pip._vendor.rich.highlighter import NullHighlighter
from pip._vendor.rich.logging import RichHandler
from pip._vendor.rich.segment import Segment
from pip._vendor.rich.style import Style

from pip._internal.utils._log import VERBOSE, getLogger
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.deprecation import DEPRECATION_MSG_PREFIX
from pip._internal.utils.misc import ensure_dir

_log_state = threading.local()
_stdout_console = None
_stderr_console = None
subprocess_logger = getLogger("pip.subprocessor")


class BrokenStdoutLoggingError(Exception):
    """
    Raised if BrokenPipeError occurs for the stdout stream while logging.
    """


def _is_broken_pipe_error(exc_class: Type[BaseException], exc: BaseException) -> bool:
    if exc_class is BrokenPipeError:
        return True

    # On Windows, a broken pipe can show up as EINVAL rather than EPIPE:
    # https://bugs.python.org/issue19612
    # https://bugs.python.org/issue30418
    if not WINDOWS:
        return False

    return isinstance(exc, OSError) and exc.errno in (errno.EINVAL, errno.EPIPE)


@contextlib.contextmanager
def indent_log(num: int = 2) -> Generator[None, None, None]:
    """
    A context manager which will cause the log output to be indented for any
    log messages emitted inside it.
    """
    # For thread-safety
    _log_state.indentation = get_indentation()
    _log_state.indentation += num
    try:
        yield
    finally:
        _log_state.indentation -= num


def get_indentation() -> int:
    return getattr(_log_state, "indentation", 0)


class IndentingFormatter(logging.Formatter):
    default_time_format = "%Y-%m-%dT%H:%M:%S"

    def __init__(
        self,
        *args: Any,
        add_timestamp: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        A logging.Formatter that obeys the indent_log() context manager.

        :param add_timestamp: A bool indicating output lines should be prefixed
            with their record's timestamp.
        """
        self.add_timestamp = add_timestamp
        super().__init__(*args, **kwargs)

    def get_message_start(self, formatted: str, levelno: int) -> str:
        """
        Return the start of the formatted log message (not counting the
        prefix to add to each line).
        """
        if levelno < logging.WARNING:
            return ""
        if formatted.startswith(DEPRECATION_MSG_PREFIX):
            # Then the message already has a prefix.  We don't want it to
            # look like "WARNING: DEPRECATION: ...."
            return ""
        if levelno < logging.ERROR:
            return "WARNING: "

        return "ERROR: "

    def format(self, record: logging.LogRecord) -> str:
        """
        Calls the standard formatter, but will indent all of the log message
        lines by our current indentation level.
        """
        formatted = super().format(record)
        message_start = self.get_message_start(formatted, record.levelno)
        formatted = message_start + formatted

        prefix = ""
        if self.add_timestamp:
            prefix = f"{self.formatTime(record)} "
        prefix += " " * get_indentation()
        formatted = "".join([prefix + line for line in formatted.splitlines(True)])
        return formatted


@dataclass
class IndentedRenderable:
    renderable: RenderableType
    indent: int

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        segments = console.render(self.renderable, options)
        lines = Segment.split_lines(segments)
        for line in lines:
            yield Segment(" " * self.indent)
            yield from line
            yield Segment("\n")


class PipConsole(Console):
    def on_broken_pipe(self) -> None:
        # Reraise the original exception, rich 13.8.0+ exits by default
        # instead, preventing our handler from firing.
        raise BrokenPipeError() from None


def get_console(*, stderr: bool = False) -> Console:
    if stderr:
        assert _stderr_console is not None, "stderr rich console is missing!"
        return _stderr_console
    else:
        assert _stdout_console is not None, "stdout rich console is missing!"
        return _stdout_console


class RichPipStreamHandler(RichHandler):
    KEYWORDS: ClassVar[Optional[List[str]]] = []

    def __init__(self, console: Console) -> None:
        super().__init__(
            console=console,
            show_time=False,
            show_level=False,
            show_path=False,
            highlighter=NullHighlighter(),
        )

    # Our custom override on Rich's logger, to make things work as we need them to.
    def emit(self, record: logging.LogRecord) -> None:
        style: Optional[Style] = None

        # If we are given a diagnostic error to present, present it with indentation.
        if getattr(record, "rich", False):
            assert isinstance(record.args, tuple)
            (rich_renderable,) = record.args
            assert isinstance(
                rich_renderable, (ConsoleRenderable, RichCast, str)
            ), f"{rich_renderable} is not rich-console-renderable"

            renderable: RenderableType = IndentedRenderable(
                rich_renderable, indent=get_indentation()
            )
        else:
            message = self.format(record)
            renderable = self.render_message(record, message)
            if record.levelno is not None:
                if record.levelno >= logging.ERROR:
                    style = Style(color="red")
                elif record.levelno >= logging.WARNING:
                    style = Style(color="yellow")

        try:
            self.console.print(renderable, overflow="ignore", crop=False, style=style)
        except Exception:
            self.handleError(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Called when logging is unable to log some output."""

        exc_class, exc = sys.exc_info()[:2]
        # If a broken pipe occurred while calling write() or flush() on the
        # stdout stream in logging's Handler.emit(), then raise our special
        # exception so we can handle it in main() instead of logging the
        # broken pipe error and continuing.
        if (
            exc_class
            and exc
            and self.console.file is sys.stdout
            and _is_broken_pipe_error(exc_class, exc)
        ):
            raise BrokenStdoutLoggingError()

        return super().handleError(record)


class BetterRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def _open(self) -> TextIOWrapper:
        ensure_dir(os.path.dirname(self.baseFilename))
        return super()._open()


class MaxLevelFilter(Filter):
    def __init__(self, level: int) -> None:
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno < self.level


class ExcludeLoggerFilter(Filter):
    """
    A logging Filter that excludes records from a logger (or its children).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        # The base Filter class allows only records from a logger (or its
        # children).
        return not super().filter(record)


def setup_logging(verbosity: int, no_color: bool, user_log_file: Optional[str]) -> int:
    """Configures and sets up all of the logging

    Returns the requested logging level, as its integer value.
    """

    # Determine the level to be logging at.
    if verbosity >= 2:
        level_number = logging.DEBUG
    elif verbosity == 1:
        level_number = VERBOSE
    elif verbosity == -1:
        level_number = logging.WARNING
    elif verbosity == -2:
        level_number = logging.ERROR
    elif verbosity <= -3:
        level_number = logging.CRITICAL
    else:
        level_number = logging.INFO

    level = logging.getLevelName(level_number)

    # The "root" logger should match the "console" level *unless* we also need
    # to log to a user log file.
    include_user_log = user_log_file is not None
    if include_user_log:
        additional_log_file = user_log_file
        root_level = "DEBUG"
    else:
        additional_log_file = "/dev/null"
        root_level = level

    # Disable any logging besides WARNING unless we have DEBUG level logging
    # enabled for vendored libraries.
    vendored_log_level = "WARNING" if level in ["INFO", "ERROR"] else "DEBUG"

    # Shorthands for clarity
    handler_classes = {
        "stream": "pip._internal.utils.logging.RichPipStreamHandler",
        "file": "pip._internal.utils.logging.BetterRotatingFileHandler",
    }
    handlers = ["console", "console_errors", "console_subprocess"] + (
        ["user_log"] if include_user_log else []
    )
    global _stdout_console, stderr_console
    _stdout_console = PipConsole(file=sys.stdout, no_color=no_color, soft_wrap=True)
    _stderr_console = PipConsole(file=sys.stderr, no_color=no_color, soft_wrap=True)

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "filters": {
                "exclude_warnings": {
                    "()": "pip._internal.utils.logging.MaxLevelFilter",
                    "level": logging.WARNING,
                },
                "restrict_to_subprocess": {
                    "()": "logging.Filter",
                    "name": subprocess_logger.name,
                },
                "exclude_subprocess": {
                    "()": "pip._internal.utils.logging.ExcludeLoggerFilter",
                    "name": subprocess_logger.name,
                },
            },
            "formatters": {
                "indent": {
                    "()": IndentingFormatter,
                    "format": "%(message)s",
                },
                "indent_with_timestamp": {
                    "()": IndentingFormatter,
                    "format": "%(message)s",
                    "add_timestamp": True,
                },
            },
            "handlers": {
                "console": {
                    "level": level,
                    "class": handler_classes["stream"],
                    "console": _stdout_console,
                    "filters": ["exclude_subprocess", "exclude_warnings"],
                    "formatter": "indent",
                },
                "console_errors": {
                    "level": "WARNING",
                    "class": handler_classes["stream"],
                    "console": _stderr_console,
                    "filters": ["exclude_subprocess"],
                    "formatter": "indent",
                },
                # A handler responsible for logging to the console messages
                # from the "subprocessor" logger.
                "console_subprocess": {
                    "level": level,
                    "class": handler_classes["stream"],
                    "console": _stderr_console,
                    "filters": ["restrict_to_subprocess"],
                    "formatter": "indent",
                },
                "user_log": {
                    "level": "DEBUG",
                    "class": handler_classes["file"],
                    "filename": additional_log_file,
                    "encoding": "utf-8",
                    "delay": True,
                    "formatter": "indent_with_timestamp",
                },
            },
            "root": {
                "level": root_level,
                "handlers": handlers,
            },
            "loggers": {"pip._vendor": {"level": vendored_log_level}},
        }
    )

    return level_number

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\charset_normalizer\models.py ===
from __future__ import annotations

from encodings.aliases import aliases
from hashlib import sha256
from json import dumps
from re import sub
from typing import Any, Iterator, List, Tuple

from .constant import RE_POSSIBLE_ENCODING_INDICATION, TOO_BIG_SEQUENCE
from .utils import iana_name, is_multi_byte_encoding, unicode_range


class CharsetMatch:
    def __init__(
        self,
        payload: bytes,
        guessed_encoding: str,
        mean_mess_ratio: float,
        has_sig_or_bom: bool,
        languages: CoherenceMatches,
        decoded_payload: str | None = None,
        preemptive_declaration: str | None = None,
    ):
        self._payload: bytes = payload

        self._encoding: str = guessed_encoding
        self._mean_mess_ratio: float = mean_mess_ratio
        self._languages: CoherenceMatches = languages
        self._has_sig_or_bom: bool = has_sig_or_bom
        self._unicode_ranges: list[str] | None = None

        self._leaves: list[CharsetMatch] = []
        self._mean_coherence_ratio: float = 0.0

        self._output_payload: bytes | None = None
        self._output_encoding: str | None = None

        self._string: str | None = decoded_payload

        self._preemptive_declaration: str | None = preemptive_declaration

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CharsetMatch):
            if isinstance(other, str):
                return iana_name(other) == self.encoding
            return False
        return self.encoding == other.encoding and self.fingerprint == other.fingerprint

    def __lt__(self, other: object) -> bool:
        """
        Implemented to make sorted available upon CharsetMatches items.
        """
        if not isinstance(other, CharsetMatch):
            raise ValueError

        chaos_difference: float = abs(self.chaos - other.chaos)
        coherence_difference: float = abs(self.coherence - other.coherence)

        # Below 1% difference --> Use Coherence
        if chaos_difference < 0.01 and coherence_difference > 0.02:
            return self.coherence > other.coherence
        elif chaos_difference < 0.01 and coherence_difference <= 0.02:
            # When having a difficult decision, use the result that decoded as many multi-byte as possible.
            # preserve RAM usage!
            if len(self._payload) >= TOO_BIG_SEQUENCE:
                return self.chaos < other.chaos
            return self.multi_byte_usage > other.multi_byte_usage

        return self.chaos < other.chaos

    @property
    def multi_byte_usage(self) -> float:
        return 1.0 - (len(str(self)) / len(self.raw))

    def __str__(self) -> str:
        # Lazy Str Loading
        if self._string is None:
            self._string = str(self._payload, self._encoding, "strict")
        return self._string

    def __repr__(self) -> str:
        return f"<CharsetMatch '{self.encoding}' bytes({self.fingerprint})>"

    def add_submatch(self, other: CharsetMatch) -> None:
        if not isinstance(other, CharsetMatch) or other == self:
            raise ValueError(
                "Unable to add instance <{}> as a submatch of a CharsetMatch".format(
                    other.__class__
                )
            )

        other._string = None  # Unload RAM usage; dirty trick.
        self._leaves.append(other)

    @property
    def encoding(self) -> str:
        return self._encoding

    @property
    def encoding_aliases(self) -> list[str]:
        """
        Encoding name are known by many name, using this could help when searching for IBM855 when it's listed as CP855.
        """
        also_known_as: list[str] = []
        for u, p in aliases.items():
            if self.encoding == u:
                also_known_as.append(p)
            elif self.encoding == p:
                also_known_as.append(u)
        return also_known_as

    @property
    def bom(self) -> bool:
        return self._has_sig_or_bom

    @property
    def byte_order_mark(self) -> bool:
        return self._has_sig_or_bom

    @property
    def languages(self) -> list[str]:
        """
        Return the complete list of possible languages found in decoded sequence.
        Usually not really useful. Returned list may be empty even if 'language' property return something != 'Unknown'.
        """
        return [e[0] for e in self._languages]

    @property
    def language(self) -> str:
        """
        Most probable language found in decoded sequence. If none were detected or inferred, the property will return
        "Unknown".
        """
        if not self._languages:
            # Trying to infer the language based on the given encoding
            # Its either English or we should not pronounce ourselves in certain cases.
            if "ascii" in self.could_be_from_charset:
                return "English"

            # doing it there to avoid circular import
            from charset_normalizer.cd import encoding_languages, mb_encoding_languages

            languages = (
                mb_encoding_languages(self.encoding)
                if is_multi_byte_encoding(self.encoding)
                else encoding_languages(self.encoding)
            )

            if len(languages) == 0 or "Latin Based" in languages:
                return "Unknown"

            return languages[0]

        return self._languages[0][0]

    @property
    def chaos(self) -> float:
        return self._mean_mess_ratio

    @property
    def coherence(self) -> float:
        if not self._languages:
            return 0.0
        return self._languages[0][1]

    @property
    def percent_chaos(self) -> float:
        return round(self.chaos * 100, ndigits=3)

    @property
    def percent_coherence(self) -> float:
        return round(self.coherence * 100, ndigits=3)

    @property
    def raw(self) -> bytes:
        """
        Original untouched bytes.
        """
        return self._payload

    @property
    def submatch(self) -> list[CharsetMatch]:
        return self._leaves

    @property
    def has_submatch(self) -> bool:
        return len(self._leaves) > 0

    @property
    def alphabets(self) -> list[str]:
        if self._unicode_ranges is not None:
            return self._unicode_ranges
        # list detected ranges
        detected_ranges: list[str | None] = [unicode_range(char) for char in str(self)]
        # filter and sort
        self._unicode_ranges = sorted(list({r for r in detected_ranges if r}))
        return self._unicode_ranges

    @property
    def could_be_from_charset(self) -> list[str]:
        """
        The complete list of encoding that output the exact SAME str result and therefore could be the originating
        encoding.
        This list does include the encoding available in property 'encoding'.
        """
        return [self._encoding] + [m.encoding for m in self._leaves]

    def output(self, encoding: str = "utf_8") -> bytes:
        """
        Method to get re-encoded bytes payload using given target encoding. Default to UTF-8.
        Any errors will be simply ignored by the encoder NOT replaced.
        """
        if self._output_encoding is None or self._output_encoding != encoding:
            self._output_encoding = encoding
            decoded_string = str(self)
            if (
                self._preemptive_declaration is not None
                and self._preemptive_declaration.lower()
                not in ["utf-8", "utf8", "utf_8"]
            ):
                patched_header = sub(
                    RE_POSSIBLE_ENCODING_INDICATION,
                    lambda m: m.string[m.span()[0] : m.span()[1]].replace(
                        m.groups()[0],
                        iana_name(self._output_encoding).replace("_", "-"),  # type: ignore[arg-type]
                    ),
                    decoded_string[:8192],
                    count=1,
                )

                decoded_string = patched_header + decoded_string[8192:]

            self._output_payload = decoded_string.encode(encoding, "replace")

        return self._output_payload  # type: ignore

    @property
    def fingerprint(self) -> str:
        """
        Retrieve the unique SHA256 computed using the transformed (re-encoded) payload. Not the original one.
        """
        return sha256(self.output()).hexdigest()


class CharsetMatches:
    """
    Container with every CharsetMatch items ordered by default from most probable to the less one.
    Act like a list(iterable) but does not implements all related methods.
    """

    def __init__(self, results: list[CharsetMatch] | None = None):
        self._results: list[CharsetMatch] = sorted(results) if results else []

    def __iter__(self) -> Iterator[CharsetMatch]:
        yield from self._results

    def __getitem__(self, item: int | str) -> CharsetMatch:
        """
        Retrieve a single item either by its position or encoding name (alias may be used here).
        Raise KeyError upon invalid index or encoding not present in results.
        """
        if isinstance(item, int):
            return self._results[item]
        if isinstance(item, str):
            item = iana_name(item, False)
            for result in self._results:
                if item in result.could_be_from_charset:
                    return result
        raise KeyError

    def __len__(self) -> int:
        return len(self._results)

    def __bool__(self) -> bool:
        return len(self._results) > 0

    def append(self, item: CharsetMatch) -> None:
        """
        Insert a single match. Will be inserted accordingly to preserve sort.
        Can be inserted as a submatch.
        """
        if not isinstance(item, CharsetMatch):
            raise ValueError(
                "Cannot append instance '{}' to CharsetMatches".format(
                    str(item.__class__)
                )
            )
        # We should disable the submatch factoring when the input file is too heavy (conserve RAM usage)
        if len(item.raw) < TOO_BIG_SEQUENCE:
            for match in self._results:
                if match.fingerprint == item.fingerprint and match.chaos == item.chaos:
                    match.add_submatch(item)
                    return
        self._results.append(item)
        self._results = sorted(self._results)

    def best(self) -> CharsetMatch | None:
        """
        Simply return the first match. Strict equivalent to matches[0].
        """
        if not self._results:
            return None
        return self._results[0]

    def first(self) -> CharsetMatch | None:
        """
        Redundant method, call the method best(). Kept for BC reasons.
        """
        return self.best()


CoherenceMatch = Tuple[str, float]
CoherenceMatches = List[CoherenceMatch]


class CliDetectionResult:
    def __init__(
        self,
        path: str,
        encoding: str | None,
        encoding_aliases: list[str],
        alternative_encodings: list[str],
        language: str,
        alphabets: list[str],
        has_sig_or_bom: bool,
        chaos: float,
        coherence: float,
        unicode_path: str | None,
        is_preferred: bool,
    ):
        self.path: str = path
        self.unicode_path: str | None = unicode_path
        self.encoding: str | None = encoding
        self.encoding_aliases: list[str] = encoding_aliases
        self.alternative_encodings: list[str] = alternative_encodings
        self.language: str = language
        self.alphabets: list[str] = alphabets
        self.has_sig_or_bom: bool = has_sig_or_bom
        self.chaos: float = chaos
        self.coherence: float = coherence
        self.is_preferred: bool = is_preferred

    @property
    def __dict__(self) -> dict[str, Any]:  # type: ignore
        return {
            "path": self.path,
            "encoding": self.encoding,
            "encoding_aliases": self.encoding_aliases,
            "alternative_encodings": self.alternative_encodings,
            "language": self.language,
            "alphabets": self.alphabets,
            "has_sig_or_bom": self.has_sig_or_bom,
            "chaos": self.chaos,
            "coherence": self.coherence,
            "unicode_path": self.unicode_path,
            "is_preferred": self.is_preferred,
        }

    def to_json(self) -> str:
        return dumps(self.__dict__, ensure_ascii=True, indent=4)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\tracing.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Tracing
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io


class MemoryDumpConfig(dict):
    '''
    Configuration for memory dump. Used only when "memory-infra" category is enabled.
    '''
    def to_json(self) -> dict:
        return self

    @classmethod
    def from_json(cls, json: dict) -> MemoryDumpConfig:
        return cls(json)

    def __repr__(self):
        return 'MemoryDumpConfig({})'.format(super().__repr__())


@dataclass
class TraceConfig:
    #: Controls how the trace buffer stores data.
    record_mode: typing.Optional[str] = None

    #: Size of the trace buffer in kilobytes. If not specified or zero is passed, a default value
    #: of 200 MB would be used.
    trace_buffer_size_in_kb: typing.Optional[float] = None

    #: Turns on JavaScript stack sampling.
    enable_sampling: typing.Optional[bool] = None

    #: Turns on system tracing.
    enable_systrace: typing.Optional[bool] = None

    #: Turns on argument filter.
    enable_argument_filter: typing.Optional[bool] = None

    #: Included category filters.
    included_categories: typing.Optional[typing.List[str]] = None

    #: Excluded category filters.
    excluded_categories: typing.Optional[typing.List[str]] = None

    #: Configuration to synthesize the delays in tracing.
    synthetic_delays: typing.Optional[typing.List[str]] = None

    #: Configuration for memory dump triggers. Used only when "memory-infra" category is enabled.
    memory_dump_config: typing.Optional[MemoryDumpConfig] = None

    def to_json(self):
        json = dict()
        if self.record_mode is not None:
            json['recordMode'] = self.record_mode
        if self.trace_buffer_size_in_kb is not None:
            json['traceBufferSizeInKb'] = self.trace_buffer_size_in_kb
        if self.enable_sampling is not None:
            json['enableSampling'] = self.enable_sampling
        if self.enable_systrace is not None:
            json['enableSystrace'] = self.enable_systrace
        if self.enable_argument_filter is not None:
            json['enableArgumentFilter'] = self.enable_argument_filter
        if self.included_categories is not None:
            json['includedCategories'] = [i for i in self.included_categories]
        if self.excluded_categories is not None:
            json['excludedCategories'] = [i for i in self.excluded_categories]
        if self.synthetic_delays is not None:
            json['syntheticDelays'] = [i for i in self.synthetic_delays]
        if self.memory_dump_config is not None:
            json['memoryDumpConfig'] = self.memory_dump_config.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            record_mode=str(json['recordMode']) if 'recordMode' in json else None,
            trace_buffer_size_in_kb=float(json['traceBufferSizeInKb']) if 'traceBufferSizeInKb' in json else None,
            enable_sampling=bool(json['enableSampling']) if 'enableSampling' in json else None,
            enable_systrace=bool(json['enableSystrace']) if 'enableSystrace' in json else None,
            enable_argument_filter=bool(json['enableArgumentFilter']) if 'enableArgumentFilter' in json else None,
            included_categories=[str(i) for i in json['includedCategories']] if 'includedCategories' in json else None,
            excluded_categories=[str(i) for i in json['excludedCategories']] if 'excludedCategories' in json else None,
            synthetic_delays=[str(i) for i in json['syntheticDelays']] if 'syntheticDelays' in json else None,
            memory_dump_config=MemoryDumpConfig.from_json(json['memoryDumpConfig']) if 'memoryDumpConfig' in json else None,
        )


class StreamFormat(enum.Enum):
    '''
    Data format of a trace. Can be either the legacy JSON format or the
    protocol buffer format. Note that the JSON format will be deprecated soon.
    '''
    JSON = "json"
    PROTO = "proto"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class StreamCompression(enum.Enum):
    '''
    Compression type to use for traces returned via streams.
    '''
    NONE = "none"
    GZIP = "gzip"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class MemoryDumpLevelOfDetail(enum.Enum):
    '''
    Details exposed when memory request explicitly declared.
    Keep consistent with memory_dump_request_args.h and
    memory_instrumentation.mojom
    '''
    BACKGROUND = "background"
    LIGHT = "light"
    DETAILED = "detailed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class TracingBackend(enum.Enum):
    '''
    Backend type to use for tracing. ``chrome`` uses the Chrome-integrated
    tracing service and is supported on all platforms. ``system`` is only
    supported on Chrome OS and uses the Perfetto system tracing service.
    ``auto`` chooses ``system`` when the perfettoConfig provided to Tracing.start
    specifies at least one non-Chrome data source; otherwise uses ``chrome``.
    '''
    AUTO = "auto"
    CHROME = "chrome"
    SYSTEM = "system"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def end() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Stop trace events collection.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.end',
    }
    json = yield cmd_dict


def get_categories() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Gets supported tracing categories.

    **EXPERIMENTAL**

    :returns: A list of supported tracing categories.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.getCategories',
    }
    json = yield cmd_dict
    return [str(i) for i in json['categories']]


def record_clock_sync_marker(
        sync_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Record a clock sync marker in the trace.

    **EXPERIMENTAL**

    :param sync_id: The ID of this clock sync marker
    '''
    params: T_JSON_DICT = dict()
    params['syncId'] = sync_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.recordClockSyncMarker',
        'params': params,
    }
    json = yield cmd_dict


def request_memory_dump(
        deterministic: typing.Optional[bool] = None,
        level_of_detail: typing.Optional[MemoryDumpLevelOfDetail] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Request a global memory dump.

    **EXPERIMENTAL**

    :param deterministic: *(Optional)* Enables more deterministic results by forcing garbage collection
    :param level_of_detail: *(Optional)* Specifies level of details in memory dump. Defaults to "detailed".
    :returns: A tuple with the following items:

        0. **dumpGuid** - GUID of the resulting global memory dump.
        1. **success** - True iff the global memory dump succeeded.
    '''
    params: T_JSON_DICT = dict()
    if deterministic is not None:
        params['deterministic'] = deterministic
    if level_of_detail is not None:
        params['levelOfDetail'] = level_of_detail.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.requestMemoryDump',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['dumpGuid']),
        bool(json['success'])
    )


def start(
        categories: typing.Optional[str] = None,
        options: typing.Optional[str] = None,
        buffer_usage_reporting_interval: typing.Optional[float] = None,
        transfer_mode: typing.Optional[str] = None,
        stream_format: typing.Optional[StreamFormat] = None,
        stream_compression: typing.Optional[StreamCompression] = None,
        trace_config: typing.Optional[TraceConfig] = None,
        perfetto_config: typing.Optional[str] = None,
        tracing_backend: typing.Optional[TracingBackend] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Start trace events collection.

    :param categories: **(EXPERIMENTAL)** *(Optional)* Category/tag filter
    :param options: **(EXPERIMENTAL)** *(Optional)* Tracing options
    :param buffer_usage_reporting_interval: **(EXPERIMENTAL)** *(Optional)* If set, the agent will issue bufferUsage events at this interval, specified in milliseconds
    :param transfer_mode: *(Optional)* Whether to report trace events as series of dataCollected events or to save trace to a stream (defaults to ```ReportEvents````).
    :param stream_format: *(Optional)* Trace data format to use. This only applies when using ````ReturnAsStream```` transfer mode (defaults to ````json````).
    :param stream_compression: **(EXPERIMENTAL)** *(Optional)* Compression format to use. This only applies when using ````ReturnAsStream```` transfer mode (defaults to ````none````)
    :param trace_config: *(Optional)*
    :param perfetto_config: **(EXPERIMENTAL)** *(Optional)* Base64-encoded serialized perfetto.protos.TraceConfig protobuf message When specified, the parameters ````categories````, ````options````, ````traceConfig```` are ignored.
    :param tracing_backend: **(EXPERIMENTAL)** *(Optional)* Backend type (defaults to ````auto```)
    '''
    params: T_JSON_DICT = dict()
    if categories is not None:
        params['categories'] = categories
    if options is not None:
        params['options'] = options
    if buffer_usage_reporting_interval is not None:
        params['bufferUsageReportingInterval'] = buffer_usage_reporting_interval
    if transfer_mode is not None:
        params['transferMode'] = transfer_mode
    if stream_format is not None:
        params['streamFormat'] = stream_format.to_json()
    if stream_compression is not None:
        params['streamCompression'] = stream_compression.to_json()
    if trace_config is not None:
        params['traceConfig'] = trace_config.to_json()
    if perfetto_config is not None:
        params['perfettoConfig'] = perfetto_config
    if tracing_backend is not None:
        params['tracingBackend'] = tracing_backend.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.start',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Tracing.bufferUsage')
@dataclass
class BufferUsage:
    '''
    **EXPERIMENTAL**


    '''
    #: A number in range [0..1] that indicates the used size of event buffer as a fraction of its
    #: total size.
    percent_full: typing.Optional[float]
    #: An approximate number of events in the trace log.
    event_count: typing.Optional[float]
    #: A number in range [0..1] that indicates the used size of event buffer as a fraction of its
    #: total size.
    value: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BufferUsage:
        return cls(
            percent_full=float(json['percentFull']) if 'percentFull' in json else None,
            event_count=float(json['eventCount']) if 'eventCount' in json else None,
            value=float(json['value']) if 'value' in json else None
        )


@event_class('Tracing.dataCollected')
@dataclass
class DataCollected:
    '''
    **EXPERIMENTAL**

    Contains a bucket of collected trace events. When tracing is stopped collected events will be
    sent as a sequence of dataCollected events followed by tracingComplete event.
    '''
    value: typing.List[dict]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DataCollected:
        return cls(
            value=[dict(i) for i in json['value']]
        )


@event_class('Tracing.tracingComplete')
@dataclass
class TracingComplete:
    '''
    Signals that tracing is stopped and there is no trace buffers pending flush, all data were
    delivered via dataCollected events.
    '''
    #: Indicates whether some trace data is known to have been lost, e.g. because the trace ring
    #: buffer wrapped around.
    data_loss_occurred: bool
    #: A handle of the stream that holds resulting trace data.
    stream: typing.Optional[io.StreamHandle]
    #: Trace data format of returned stream.
    trace_format: typing.Optional[StreamFormat]
    #: Compression format of returned stream.
    stream_compression: typing.Optional[StreamCompression]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TracingComplete:
        return cls(
            data_loss_occurred=bool(json['dataLossOccurred']),
            stream=io.StreamHandle.from_json(json['stream']) if 'stream' in json else None,
            trace_format=StreamFormat.from_json(json['traceFormat']) if 'traceFormat' in json else None,
            stream_compression=StreamCompression.from_json(json['streamCompression']) if 'streamCompression' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\tracing.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Tracing
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io


class MemoryDumpConfig(dict):
    '''
    Configuration for memory dump. Used only when "memory-infra" category is enabled.
    '''
    def to_json(self) -> dict:
        return self

    @classmethod
    def from_json(cls, json: dict) -> MemoryDumpConfig:
        return cls(json)

    def __repr__(self):
        return 'MemoryDumpConfig({})'.format(super().__repr__())


@dataclass
class TraceConfig:
    #: Controls how the trace buffer stores data.
    record_mode: typing.Optional[str] = None

    #: Size of the trace buffer in kilobytes. If not specified or zero is passed, a default value
    #: of 200 MB would be used.
    trace_buffer_size_in_kb: typing.Optional[float] = None

    #: Turns on JavaScript stack sampling.
    enable_sampling: typing.Optional[bool] = None

    #: Turns on system tracing.
    enable_systrace: typing.Optional[bool] = None

    #: Turns on argument filter.
    enable_argument_filter: typing.Optional[bool] = None

    #: Included category filters.
    included_categories: typing.Optional[typing.List[str]] = None

    #: Excluded category filters.
    excluded_categories: typing.Optional[typing.List[str]] = None

    #: Configuration to synthesize the delays in tracing.
    synthetic_delays: typing.Optional[typing.List[str]] = None

    #: Configuration for memory dump triggers. Used only when "memory-infra" category is enabled.
    memory_dump_config: typing.Optional[MemoryDumpConfig] = None

    def to_json(self):
        json = dict()
        if self.record_mode is not None:
            json['recordMode'] = self.record_mode
        if self.trace_buffer_size_in_kb is not None:
            json['traceBufferSizeInKb'] = self.trace_buffer_size_in_kb
        if self.enable_sampling is not None:
            json['enableSampling'] = self.enable_sampling
        if self.enable_systrace is not None:
            json['enableSystrace'] = self.enable_systrace
        if self.enable_argument_filter is not None:
            json['enableArgumentFilter'] = self.enable_argument_filter
        if self.included_categories is not None:
            json['includedCategories'] = [i for i in self.included_categories]
        if self.excluded_categories is not None:
            json['excludedCategories'] = [i for i in self.excluded_categories]
        if self.synthetic_delays is not None:
            json['syntheticDelays'] = [i for i in self.synthetic_delays]
        if self.memory_dump_config is not None:
            json['memoryDumpConfig'] = self.memory_dump_config.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            record_mode=str(json['recordMode']) if 'recordMode' in json else None,
            trace_buffer_size_in_kb=float(json['traceBufferSizeInKb']) if 'traceBufferSizeInKb' in json else None,
            enable_sampling=bool(json['enableSampling']) if 'enableSampling' in json else None,
            enable_systrace=bool(json['enableSystrace']) if 'enableSystrace' in json else None,
            enable_argument_filter=bool(json['enableArgumentFilter']) if 'enableArgumentFilter' in json else None,
            included_categories=[str(i) for i in json['includedCategories']] if 'includedCategories' in json else None,
            excluded_categories=[str(i) for i in json['excludedCategories']] if 'excludedCategories' in json else None,
            synthetic_delays=[str(i) for i in json['syntheticDelays']] if 'syntheticDelays' in json else None,
            memory_dump_config=MemoryDumpConfig.from_json(json['memoryDumpConfig']) if 'memoryDumpConfig' in json else None,
        )


class StreamFormat(enum.Enum):
    '''
    Data format of a trace. Can be either the legacy JSON format or the
    protocol buffer format. Note that the JSON format will be deprecated soon.
    '''
    JSON = "json"
    PROTO = "proto"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class StreamCompression(enum.Enum):
    '''
    Compression type to use for traces returned via streams.
    '''
    NONE = "none"
    GZIP = "gzip"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class MemoryDumpLevelOfDetail(enum.Enum):
    '''
    Details exposed when memory request explicitly declared.
    Keep consistent with memory_dump_request_args.h and
    memory_instrumentation.mojom
    '''
    BACKGROUND = "background"
    LIGHT = "light"
    DETAILED = "detailed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class TracingBackend(enum.Enum):
    '''
    Backend type to use for tracing. ``chrome`` uses the Chrome-integrated
    tracing service and is supported on all platforms. ``system`` is only
    supported on Chrome OS and uses the Perfetto system tracing service.
    ``auto`` chooses ``system`` when the perfettoConfig provided to Tracing.start
    specifies at least one non-Chrome data source; otherwise uses ``chrome``.
    '''
    AUTO = "auto"
    CHROME = "chrome"
    SYSTEM = "system"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def end() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Stop trace events collection.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.end',
    }
    json = yield cmd_dict


def get_categories() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Gets supported tracing categories.

    **EXPERIMENTAL**

    :returns: A list of supported tracing categories.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.getCategories',
    }
    json = yield cmd_dict
    return [str(i) for i in json['categories']]


def record_clock_sync_marker(
        sync_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Record a clock sync marker in the trace.

    **EXPERIMENTAL**

    :param sync_id: The ID of this clock sync marker
    '''
    params: T_JSON_DICT = dict()
    params['syncId'] = sync_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.recordClockSyncMarker',
        'params': params,
    }
    json = yield cmd_dict


def request_memory_dump(
        deterministic: typing.Optional[bool] = None,
        level_of_detail: typing.Optional[MemoryDumpLevelOfDetail] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Request a global memory dump.

    **EXPERIMENTAL**

    :param deterministic: *(Optional)* Enables more deterministic results by forcing garbage collection
    :param level_of_detail: *(Optional)* Specifies level of details in memory dump. Defaults to "detailed".
    :returns: A tuple with the following items:

        0. **dumpGuid** - GUID of the resulting global memory dump.
        1. **success** - True iff the global memory dump succeeded.
    '''
    params: T_JSON_DICT = dict()
    if deterministic is not None:
        params['deterministic'] = deterministic
    if level_of_detail is not None:
        params['levelOfDetail'] = level_of_detail.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.requestMemoryDump',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['dumpGuid']),
        bool(json['success'])
    )


def start(
        categories: typing.Optional[str] = None,
        options: typing.Optional[str] = None,
        buffer_usage_reporting_interval: typing.Optional[float] = None,
        transfer_mode: typing.Optional[str] = None,
        stream_format: typing.Optional[StreamFormat] = None,
        stream_compression: typing.Optional[StreamCompression] = None,
        trace_config: typing.Optional[TraceConfig] = None,
        perfetto_config: typing.Optional[str] = None,
        tracing_backend: typing.Optional[TracingBackend] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Start trace events collection.

    :param categories: **(EXPERIMENTAL)** *(Optional)* Category/tag filter
    :param options: **(EXPERIMENTAL)** *(Optional)* Tracing options
    :param buffer_usage_reporting_interval: **(EXPERIMENTAL)** *(Optional)* If set, the agent will issue bufferUsage events at this interval, specified in milliseconds
    :param transfer_mode: *(Optional)* Whether to report trace events as series of dataCollected events or to save trace to a stream (defaults to ```ReportEvents````).
    :param stream_format: *(Optional)* Trace data format to use. This only applies when using ````ReturnAsStream```` transfer mode (defaults to ````json````).
    :param stream_compression: **(EXPERIMENTAL)** *(Optional)* Compression format to use. This only applies when using ````ReturnAsStream```` transfer mode (defaults to ````none````)
    :param trace_config: *(Optional)*
    :param perfetto_config: **(EXPERIMENTAL)** *(Optional)* Base64-encoded serialized perfetto.protos.TraceConfig protobuf message When specified, the parameters ````categories````, ````options````, ````traceConfig```` are ignored.
    :param tracing_backend: **(EXPERIMENTAL)** *(Optional)* Backend type (defaults to ````auto```)
    '''
    params: T_JSON_DICT = dict()
    if categories is not None:
        params['categories'] = categories
    if options is not None:
        params['options'] = options
    if buffer_usage_reporting_interval is not None:
        params['bufferUsageReportingInterval'] = buffer_usage_reporting_interval
    if transfer_mode is not None:
        params['transferMode'] = transfer_mode
    if stream_format is not None:
        params['streamFormat'] = stream_format.to_json()
    if stream_compression is not None:
        params['streamCompression'] = stream_compression.to_json()
    if trace_config is not None:
        params['traceConfig'] = trace_config.to_json()
    if perfetto_config is not None:
        params['perfettoConfig'] = perfetto_config
    if tracing_backend is not None:
        params['tracingBackend'] = tracing_backend.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.start',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Tracing.bufferUsage')
@dataclass
class BufferUsage:
    '''
    **EXPERIMENTAL**


    '''
    #: A number in range [0..1] that indicates the used size of event buffer as a fraction of its
    #: total size.
    percent_full: typing.Optional[float]
    #: An approximate number of events in the trace log.
    event_count: typing.Optional[float]
    #: A number in range [0..1] that indicates the used size of event buffer as a fraction of its
    #: total size.
    value: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BufferUsage:
        return cls(
            percent_full=float(json['percentFull']) if 'percentFull' in json else None,
            event_count=float(json['eventCount']) if 'eventCount' in json else None,
            value=float(json['value']) if 'value' in json else None
        )


@event_class('Tracing.dataCollected')
@dataclass
class DataCollected:
    '''
    **EXPERIMENTAL**

    Contains a bucket of collected trace events. When tracing is stopped collected events will be
    sent as a sequence of dataCollected events followed by tracingComplete event.
    '''
    value: typing.List[dict]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DataCollected:
        return cls(
            value=[dict(i) for i in json['value']]
        )


@event_class('Tracing.tracingComplete')
@dataclass
class TracingComplete:
    '''
    Signals that tracing is stopped and there is no trace buffers pending flush, all data were
    delivered via dataCollected events.
    '''
    #: Indicates whether some trace data is known to have been lost, e.g. because the trace ring
    #: buffer wrapped around.
    data_loss_occurred: bool
    #: A handle of the stream that holds resulting trace data.
    stream: typing.Optional[io.StreamHandle]
    #: Trace data format of returned stream.
    trace_format: typing.Optional[StreamFormat]
    #: Compression format of returned stream.
    stream_compression: typing.Optional[StreamCompression]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TracingComplete:
        return cls(
            data_loss_occurred=bool(json['dataLossOccurred']),
            stream=io.StreamHandle.from_json(json['stream']) if 'stream' in json else None,
            trace_format=StreamFormat.from_json(json['traceFormat']) if 'traceFormat' in json else None,
            stream_compression=StreamCompression.from_json(json['streamCompression']) if 'streamCompression' in json else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\tracing.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Tracing
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import io


class MemoryDumpConfig(dict):
    '''
    Configuration for memory dump. Used only when "memory-infra" category is enabled.
    '''
    def to_json(self) -> dict:
        return self

    @classmethod
    def from_json(cls, json: dict) -> MemoryDumpConfig:
        return cls(json)

    def __repr__(self):
        return 'MemoryDumpConfig({})'.format(super().__repr__())


@dataclass
class TraceConfig:
    #: Controls how the trace buffer stores data.
    record_mode: typing.Optional[str] = None

    #: Size of the trace buffer in kilobytes. If not specified or zero is passed, a default value
    #: of 200 MB would be used.
    trace_buffer_size_in_kb: typing.Optional[float] = None

    #: Turns on JavaScript stack sampling.
    enable_sampling: typing.Optional[bool] = None

    #: Turns on system tracing.
    enable_systrace: typing.Optional[bool] = None

    #: Turns on argument filter.
    enable_argument_filter: typing.Optional[bool] = None

    #: Included category filters.
    included_categories: typing.Optional[typing.List[str]] = None

    #: Excluded category filters.
    excluded_categories: typing.Optional[typing.List[str]] = None

    #: Configuration to synthesize the delays in tracing.
    synthetic_delays: typing.Optional[typing.List[str]] = None

    #: Configuration for memory dump triggers. Used only when "memory-infra" category is enabled.
    memory_dump_config: typing.Optional[MemoryDumpConfig] = None

    def to_json(self):
        json = dict()
        if self.record_mode is not None:
            json['recordMode'] = self.record_mode
        if self.trace_buffer_size_in_kb is not None:
            json['traceBufferSizeInKb'] = self.trace_buffer_size_in_kb
        if self.enable_sampling is not None:
            json['enableSampling'] = self.enable_sampling
        if self.enable_systrace is not None:
            json['enableSystrace'] = self.enable_systrace
        if self.enable_argument_filter is not None:
            json['enableArgumentFilter'] = self.enable_argument_filter
        if self.included_categories is not None:
            json['includedCategories'] = [i for i in self.included_categories]
        if self.excluded_categories is not None:
            json['excludedCategories'] = [i for i in self.excluded_categories]
        if self.synthetic_delays is not None:
            json['syntheticDelays'] = [i for i in self.synthetic_delays]
        if self.memory_dump_config is not None:
            json['memoryDumpConfig'] = self.memory_dump_config.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            record_mode=str(json['recordMode']) if 'recordMode' in json else None,
            trace_buffer_size_in_kb=float(json['traceBufferSizeInKb']) if 'traceBufferSizeInKb' in json else None,
            enable_sampling=bool(json['enableSampling']) if 'enableSampling' in json else None,
            enable_systrace=bool(json['enableSystrace']) if 'enableSystrace' in json else None,
            enable_argument_filter=bool(json['enableArgumentFilter']) if 'enableArgumentFilter' in json else None,
            included_categories=[str(i) for i in json['includedCategories']] if 'includedCategories' in json else None,
            excluded_categories=[str(i) for i in json['excludedCategories']] if 'excludedCategories' in json else None,
            synthetic_delays=[str(i) for i in json['syntheticDelays']] if 'syntheticDelays' in json else None,
            memory_dump_config=MemoryDumpConfig.from_json(json['memoryDumpConfig']) if 'memoryDumpConfig' in json else None,
        )


class StreamFormat(enum.Enum):
    '''
    Data format of a trace. Can be either the legacy JSON format or the
    protocol buffer format. Note that the JSON format will be deprecated soon.
    '''
    JSON = "json"
    PROTO = "proto"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class StreamCompression(enum.Enum):
    '''
    Compression type to use for traces returned via streams.
    '''
    NONE = "none"
    GZIP = "gzip"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class MemoryDumpLevelOfDetail(enum.Enum):
    '''
    Details exposed when memory request explicitly declared.
    Keep consistent with memory_dump_request_args.h and
    memory_instrumentation.mojom
    '''
    BACKGROUND = "background"
    LIGHT = "light"
    DETAILED = "detailed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class TracingBackend(enum.Enum):
    '''
    Backend type to use for tracing. ``chrome`` uses the Chrome-integrated
    tracing service and is supported on all platforms. ``system`` is only
    supported on Chrome OS and uses the Perfetto system tracing service.
    ``auto`` chooses ``system`` when the perfettoConfig provided to Tracing.start
    specifies at least one non-Chrome data source; otherwise uses ``chrome``.
    '''
    AUTO = "auto"
    CHROME = "chrome"
    SYSTEM = "system"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def end() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Stop trace events collection.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.end',
    }
    json = yield cmd_dict


def get_categories() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Gets supported tracing categories.

    **EXPERIMENTAL**

    :returns: A list of supported tracing categories.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.getCategories',
    }
    json = yield cmd_dict
    return [str(i) for i in json['categories']]


def record_clock_sync_marker(
        sync_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Record a clock sync marker in the trace.

    **EXPERIMENTAL**

    :param sync_id: The ID of this clock sync marker
    '''
    params: T_JSON_DICT = dict()
    params['syncId'] = sync_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.recordClockSyncMarker',
        'params': params,
    }
    json = yield cmd_dict


def request_memory_dump(
        deterministic: typing.Optional[bool] = None,
        level_of_detail: typing.Optional[MemoryDumpLevelOfDetail] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Request a global memory dump.

    **EXPERIMENTAL**

    :param deterministic: *(Optional)* Enables more deterministic results by forcing garbage collection
    :param level_of_detail: *(Optional)* Specifies level of details in memory dump. Defaults to "detailed".
    :returns: A tuple with the following items:

        0. **dumpGuid** - GUID of the resulting global memory dump.
        1. **success** - True iff the global memory dump succeeded.
    '''
    params: T_JSON_DICT = dict()
    if deterministic is not None:
        params['deterministic'] = deterministic
    if level_of_detail is not None:
        params['levelOfDetail'] = level_of_detail.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.requestMemoryDump',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['dumpGuid']),
        bool(json['success'])
    )


def start(
        categories: typing.Optional[str] = None,
        options: typing.Optional[str] = None,
        buffer_usage_reporting_interval: typing.Optional[float] = None,
        transfer_mode: typing.Optional[str] = None,
        stream_format: typing.Optional[StreamFormat] = None,
        stream_compression: typing.Optional[StreamCompression] = None,
        trace_config: typing.Optional[TraceConfig] = None,
        perfetto_config: typing.Optional[str] = None,
        tracing_backend: typing.Optional[TracingBackend] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Start trace events collection.

    :param categories: **(EXPERIMENTAL)** *(Optional)* Category/tag filter
    :param options: **(EXPERIMENTAL)** *(Optional)* Tracing options
    :param buffer_usage_reporting_interval: **(EXPERIMENTAL)** *(Optional)* If set, the agent will issue bufferUsage events at this interval, specified in milliseconds
    :param transfer_mode: *(Optional)* Whether to report trace events as series of dataCollected events or to save trace to a stream (defaults to ```ReportEvents````).
    :param stream_format: *(Optional)* Trace data format to use. This only applies when using ````ReturnAsStream```` transfer mode (defaults to ````json````).
    :param stream_compression: **(EXPERIMENTAL)** *(Optional)* Compression format to use. This only applies when using ````ReturnAsStream```` transfer mode (defaults to ````none````)
    :param trace_config: *(Optional)*
    :param perfetto_config: **(EXPERIMENTAL)** *(Optional)* Base64-encoded serialized perfetto.protos.TraceConfig protobuf message When specified, the parameters ````categories````, ````options````, ````traceConfig```` are ignored.
    :param tracing_backend: **(EXPERIMENTAL)** *(Optional)* Backend type (defaults to ````auto```)
    '''
    params: T_JSON_DICT = dict()
    if categories is not None:
        params['categories'] = categories
    if options is not None:
        params['options'] = options
    if buffer_usage_reporting_interval is not None:
        params['bufferUsageReportingInterval'] = buffer_usage_reporting_interval
    if transfer_mode is not None:
        params['transferMode'] = transfer_mode
    if stream_format is not None:
        params['streamFormat'] = stream_format.to_json()
    if stream_compression is not None:
        params['streamCompression'] = stream_compression.to_json()
    if trace_config is not None:
        params['traceConfig'] = trace_config.to_json()
    if perfetto_config is not None:
        params['perfettoConfig'] = perfetto_config
    if tracing_backend is not None:
        params['tracingBackend'] = tracing_backend.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Tracing.start',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Tracing.bufferUsage')
@dataclass
class BufferUsage:
    '''
    **EXPERIMENTAL**


    '''
    #: A number in range [0..1] that indicates the used size of event buffer as a fraction of its
    #: total size.
    percent_full: typing.Optional[float]
    #: An approximate number of events in the trace log.
    event_count: typing.Optional[float]
    #: A number in range [0..1] that indicates the used size of event buffer as a fraction of its
    #: total size.
    value: typing.Optional[float]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> BufferUsage:
        return cls(
            percent_full=float(json['percentFull']) if 'percentFull' in json else None,
            event_count=float(json['eventCount']) if 'eventCount' in json else None,
            value=float(json['value']) if 'value' in json else None
        )


@event_class('Tracing.dataCollected')
@dataclass
class DataCollected:
    '''
    **EXPERIMENTAL**

    Contains a bucket of collected trace events. When tracing is stopped collected events will be
    sent as a sequence of dataCollected events followed by tracingComplete event.
    '''
    value: typing.List[dict]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DataCollected:
        return cls(
            value=[dict(i) for i in json['value']]
        )


@event_class('Tracing.tracingComplete')
@dataclass
class TracingComplete:
    '''
    Signals that tracing is stopped and there is no trace buffers pending flush, all data were
    delivered via dataCollected events.
    '''
    #: Indicates whether some trace data is known to have been lost, e.g. because the trace ring
    #: buffer wrapped around.
    data_loss_occurred: bool
    #: A handle of the stream that holds resulting trace data.
    stream: typing.Optional[io.StreamHandle]
    #: Trace data format of returned stream.
    trace_format: typing.Optional[StreamFormat]
    #: Compression format of returned stream.
    stream_compression: typing.Optional[StreamCompression]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TracingComplete:
        return cls(
            data_loss_occurred=bool(json['dataLossOccurred']),
            stream=io.StreamHandle.from_json(json['stream']) if 'stream' in json else None,
            trace_format=StreamFormat.from_json(json['traceFormat']) if 'traceFormat' in json else None,
            stream_compression=StreamCompression.from_json(json['streamCompression']) if 'streamCompression' in json else None
        )