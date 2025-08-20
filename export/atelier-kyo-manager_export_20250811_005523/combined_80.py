
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\gate.py ===
"""An implementation of gates that act on qubits.

Gates are unitary operators that act on the space of qubits.

Medium Term Todo:

* Optimize Gate._apply_operators_Qubit to remove the creation of many
  intermediate Qubit objects.
* Add commutation relationships to all operators and use this in gate_sort.
* Fix gate_sort and gate_simp.
* Get multi-target UGates plotting properly.
* Get UGate to work with either sympy/numpy matrices and output either
  format. This should also use the matrix slots.
"""

from itertools import chain
import random

from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.mul import Mul
from sympy.core.numbers import (I, Integer)
from sympy.core.power import Pow
from sympy.core.numbers import Number
from sympy.core.singleton import S as _S
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.pretty.stringpict import prettyForm, stringPict

from sympy.physics.quantum.anticommutator import AntiCommutator
from sympy.physics.quantum.commutator import Commutator
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import (UnitaryOperator, Operator,
                                            HermitianOperator)
from sympy.physics.quantum.matrixutils import matrix_tensor_product, matrix_eye
from sympy.physics.quantum.matrixcache import matrix_cache

from sympy.matrices.matrixbase import MatrixBase

from sympy.utilities.iterables import is_sequence

__all__ = [
    'Gate',
    'CGate',
    'UGate',
    'OneQubitGate',
    'TwoQubitGate',
    'IdentityGate',
    'HadamardGate',
    'XGate',
    'YGate',
    'ZGate',
    'TGate',
    'PhaseGate',
    'SwapGate',
    'CNotGate',
    # Aliased gate names
    'CNOT',
    'SWAP',
    'H',
    'X',
    'Y',
    'Z',
    'T',
    'S',
    'Phase',
    'normalized',
    'gate_sort',
    'gate_simp',
    'random_circuit',
    'CPHASE',
    'CGateS',
]

#-----------------------------------------------------------------------------
# Gate Super-Classes
#-----------------------------------------------------------------------------

_normalized = True


def _max(*args, **kwargs):
    if "key" not in kwargs:
        kwargs["key"] = default_sort_key
    return max(*args, **kwargs)


def _min(*args, **kwargs):
    if "key" not in kwargs:
        kwargs["key"] = default_sort_key
    return min(*args, **kwargs)


def normalized(normalize):
    r"""Set flag controlling normalization of Hadamard gates by `1/\sqrt{2}`.

    This is a global setting that can be used to simplify the look of various
    expressions, by leaving off the leading `1/\sqrt{2}` of the Hadamard gate.

    Parameters
    ----------
    normalize : bool
        Should the Hadamard gate include the `1/\sqrt{2}` normalization factor?
        When True, the Hadamard gate will have the `1/\sqrt{2}`. When False, the
        Hadamard gate will not have this factor.
    """
    global _normalized
    _normalized = normalize


def _validate_targets_controls(tandc):
    tandc = list(tandc)
    # Check for integers
    for bit in tandc:
        if not bit.is_Integer and not bit.is_Symbol:
            raise TypeError('Integer expected, got: %r' % tandc[bit])
    # Detect duplicates
    if len(set(tandc)) != len(tandc):
        raise QuantumError(
            'Target/control qubits in a gate cannot be duplicated'
        )


class Gate(UnitaryOperator):
    """Non-controlled unitary gate operator that acts on qubits.

    This is a general abstract gate that needs to be subclassed to do anything
    useful.

    Parameters
    ----------
    label : tuple, int
        A list of the target qubits (as ints) that the gate will apply to.

    Examples
    ========


    """

    _label_separator = ','

    gate_name = 'G'
    gate_name_latex = 'G'

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        args = Tuple(*UnitaryOperator._eval_args(args))
        _validate_targets_controls(args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
        return len(self.targets)

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(self.targets) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        return self.label

    @property
    def gate_name_plot(self):
        return r'$%s$' % self.gate_name_latex

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        """The matrix representation of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
        raise NotImplementedError(
            'get_target_matrix is not implemented in Gate.')

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_IntQubit(self, qubits, **options):
        """Redirect an apply from IntQubit to Qubit"""
        return self._apply_operator_Qubit(qubits, **options)

    def _apply_operator_Qubit(self, qubits, **options):
        """Apply this gate to a Qubit."""

        # Check number of qubits this gate acts on.
        if qubits.nqubits < self.min_qubits:
            raise QuantumError(
                'Gate needs a minimum of %r qubits to act on, got: %r' %
                (self.min_qubits, qubits.nqubits)
            )

        # If the controls are not met, just return
        if isinstance(self, CGate):
            if not self.eval_controls(qubits):
                return qubits

        targets = self.targets
        target_matrix = self.get_target_matrix(format='sympy')

        # Find which column of the target matrix this applies to.
        column_index = 0
        n = 1
        for target in targets:
            column_index += n*qubits[target]
            n = n << 1
        column = target_matrix[:, int(column_index)]

        # Now apply each column element to the qubit.
        result = 0
        for index in range(column.rows):
            # TODO: This can be optimized to reduce the number of Qubit
            # creations. We should simply manipulate the raw list of qubit
            # values and then build the new Qubit object once.
            # Make a copy of the incoming qubits.
            new_qubit = qubits.__class__(*qubits.args)
            # Flip the bits that need to be flipped.
            for bit, target in enumerate(targets):
                if new_qubit[target] != (index >> bit) & 1:
                    new_qubit = new_qubit.flip(target)
            # The value in that row and column times the flipped-bit qubit
            # is the result for that part.
            result += column[index]*new_qubit
        return result

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    def _represent_default_basis(self, **options):
        return self._represent_ZGate(None, **options)

    def _represent_ZGate(self, basis, **options):
        format = options.get('format', 'sympy')
        nqubits = options.get('nqubits', 0)
        if nqubits == 0:
            raise QuantumError(
                'The number of qubits must be given as nqubits.')

        # Make sure we have enough qubits for the gate.
        if nqubits < self.min_qubits:
            raise QuantumError(
                'The number of qubits %r is too small for the gate.' % nqubits
            )

        target_matrix = self.get_target_matrix(format)
        targets = self.targets
        if isinstance(self, CGate):
            controls = self.controls
        else:
            controls = []
        m = represent_zbasis(
            controls, targets, target_matrix, nqubits, format
        )
        return m

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------

    def _sympystr(self, printer, *args):
        label = self._print_label(printer, *args)
        return '%s(%s)' % (self.gate_name, label)

    def _pretty(self, printer, *args):
        a = stringPict(self.gate_name)
        b = self._print_label_pretty(printer, *args)
        return self._print_subscript_pretty(a, b)

    def _latex(self, printer, *args):
        label = self._print_label(printer, *args)
        return '%s_{%s}' % (self.gate_name_latex, label)

    def plot_gate(self, axes, gate_idx, gate_grid, wire_grid):
        raise NotImplementedError('plot_gate is not implemented.')


class CGate(Gate):
    """A general unitary gate with control qubits.

    A general control gate applies a target gate to a set of targets if all
    of the control qubits have a particular values (set by
    ``CGate.control_value``).

    Parameters
    ----------
    label : tuple
        The label in this case has the form (controls, gate), where controls
        is a tuple/list of control qubits (as ints) and gate is a ``Gate``
        instance that is the target operator.

    Examples
    ========

    """

    gate_name = 'C'
    gate_name_latex = 'C'

    # The values this class controls for.
    control_value = _S.One

    simplify_cgate = False

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        # _eval_args has the right logic for the controls argument.
        controls = args[0]
        gate = args[1]
        if not is_sequence(controls):
            controls = (controls,)
        controls = UnitaryOperator._eval_args(controls)
        _validate_targets_controls(chain(controls, gate.targets))
        return (Tuple(*controls), gate)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**_max(_max(args[0]) + 1, args[1].min_qubits)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def nqubits(self):
        """The total number of qubits this gate acts on.

        For controlled gate subclasses this includes both target and control
        qubits, so that, for examples the CNOT gate acts on 2 qubits.
        """
        return len(self.targets) + len(self.controls)

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(_max(self.controls), _max(self.targets)) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        return self.gate.targets

    @property
    def controls(self):
        """A tuple of control qubits."""
        return tuple(self.label[0])

    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
        return self.label[1]

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        return self.gate.get_target_matrix(format)

    def eval_controls(self, qubit):
        """Return True/False to indicate if the controls are satisfied."""
        return all(qubit[bit] == self.control_value for bit in self.controls)

    def decompose(self, **options):
        """Decompose the controlled gate into CNOT and single qubits gates."""
        if len(self.controls) == 1:
            c = self.controls[0]
            t = self.gate.targets[0]
            if isinstance(self.gate, YGate):
                g1 = PhaseGate(t)
                g2 = CNotGate(c, t)
                g3 = PhaseGate(t)
                g4 = ZGate(t)
                return g1*g2*g3*g4
            if isinstance(self.gate, ZGate):
                g1 = HadamardGate(t)
                g2 = CNotGate(c, t)
                g3 = HadamardGate(t)
                return g1*g2*g3
        else:
            return self

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------

    def _print_label(self, printer, *args):
        controls = self._print_sequence(self.controls, ',', printer, *args)
        gate = printer._print(self.gate, *args)
        return '(%s),%s' % (controls, gate)

    def _pretty(self, printer, *args):
        controls = self._print_sequence_pretty(
            self.controls, ',', printer, *args)
        gate = printer._print(self.gate)
        gate_name = stringPict(self.gate_name)
        first = self._print_subscript_pretty(gate_name, controls)
        gate = self._print_parens_pretty(gate)
        final = prettyForm(*first.right(gate))
        return final

    def _latex(self, printer, *args):
        controls = self._print_sequence(self.controls, ',', printer, *args)
        gate = printer._print(self.gate, *args)
        return r'%s_{%s}{\left(%s\right)}' % \
            (self.gate_name_latex, controls, gate)

    def plot_gate(self, circ_plot, gate_idx):
        """
        Plot the controlled gate. If *simplify_cgate* is true, simplify
        C-X and C-Z gates into their more familiar forms.
        """
        min_wire = int(_min(chain(self.controls, self.targets)))
        max_wire = int(_max(chain(self.controls, self.targets)))
        circ_plot.control_line(gate_idx, min_wire, max_wire)
        for c in self.controls:
            circ_plot.control_point(gate_idx, int(c))
        if self.simplify_cgate:
            if self.gate.gate_name == 'X':
                self.gate.plot_gate_plus(circ_plot, gate_idx)
            elif self.gate.gate_name == 'Z':
                circ_plot.control_point(gate_idx, self.targets[0])
            else:
                self.gate.plot_gate(circ_plot, gate_idx)
        else:
            self.gate.plot_gate(circ_plot, gate_idx)

    #-------------------------------------------------------------------------
    # Miscellaneous
    #-------------------------------------------------------------------------

    def _eval_dagger(self):
        if isinstance(self.gate, HermitianOperator):
            return self
        else:
            return Gate._eval_dagger(self)

    def _eval_inverse(self):
        if isinstance(self.gate, HermitianOperator):
            return self
        else:
            return Gate._eval_inverse(self)

    def _eval_power(self, exp):
        if isinstance(self.gate, HermitianOperator):
            if exp == -1:
                return Gate._eval_power(self, exp)
            elif abs(exp) % 2 == 0:
                return self*(Gate._eval_inverse(self))
            else:
                return self
        else:
            return Gate._eval_power(self, exp)

class CGateS(CGate):
    """Version of CGate that allows gate simplifications.
    I.e. cnot looks like an oplus, cphase has dots, etc.
    """
    simplify_cgate=True


class UGate(Gate):
    """General gate specified by a set of targets and a target matrix.

    Parameters
    ----------
    label : tuple
        A tuple of the form (targets, U), where targets is a tuple of the
        target qubits and U is a unitary matrix with dimension of
        len(targets).
    """
    gate_name = 'U'
    gate_name_latex = 'U'

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        targets = args[0]
        if not is_sequence(targets):
            targets = (targets,)
        targets = Gate._eval_args(targets)
        _validate_targets_controls(targets)
        mat = args[1]
        if not isinstance(mat, MatrixBase):
            raise TypeError('Matrix expected, got: %r' % mat)
        #make sure this matrix is of a Basic type
        mat = _sympify(mat)
        dim = 2**len(targets)
        if not all(dim == shape for shape in mat.shape):
            raise IndexError(
                'Number of targets must match the matrix size: %r %r' %
                (targets, mat)
            )
        return (targets, mat)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args[0]) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def targets(self):
        """A tuple of target qubits."""
        return tuple(self.label[0])

    #-------------------------------------------------------------------------
    # Gate methods
    #-------------------------------------------------------------------------

    def get_target_matrix(self, format='sympy'):
        """The matrix rep. of the target part of the gate.

        Parameters
        ----------
        format : str
            The format string ('sympy','numpy', etc.)
        """
        return self.label[1]

    #-------------------------------------------------------------------------
    # Print methods
    #-------------------------------------------------------------------------
    def _pretty(self, printer, *args):
        targets = self._print_sequence_pretty(
            self.targets, ',', printer, *args)
        gate_name = stringPict(self.gate_name)
        return self._print_subscript_pretty(gate_name, targets)

    def _latex(self, printer, *args):
        targets = self._print_sequence(self.targets, ',', printer, *args)
        return r'%s_{%s}' % (self.gate_name_latex, targets)

    def plot_gate(self, circ_plot, gate_idx):
        circ_plot.one_qubit_box(
            self.gate_name_plot,
            gate_idx, int(self.targets[0])
        )


class OneQubitGate(Gate):
    """A single qubit unitary gate base class."""

    nqubits = _S.One

    def plot_gate(self, circ_plot, gate_idx):
        circ_plot.one_qubit_box(
            self.gate_name_plot,
            gate_idx, int(self.targets[0])
        )

    def _eval_commutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return _S.Zero
        return Operator._eval_commutator(self, other, **hints)

    def _eval_anticommutator(self, other, **hints):
        if isinstance(other, OneQubitGate):
            if self.targets != other.targets or self.__class__ == other.__class__:
                return Integer(2)*self*other
        return Operator._eval_anticommutator(self, other, **hints)


class TwoQubitGate(Gate):
    """A two qubit unitary gate base class."""

    nqubits = Integer(2)

#-----------------------------------------------------------------------------
# Single Qubit Gates
#-----------------------------------------------------------------------------


class IdentityGate(OneQubitGate):
    """The single qubit identity gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian = True
    gate_name = '1'
    gate_name_latex = '1'

    # Short cut version of gate._apply_operator_Qubit
    def _apply_operator_Qubit(self, qubits, **options):
        # Check number of qubits this gate acts on (see gate._apply_operator_Qubit)
        if qubits.nqubits < self.min_qubits:
            raise QuantumError(
                'Gate needs a minimum of %r qubits to act on, got: %r' %
                (self.min_qubits, qubits.nqubits)
            )
        return qubits # no computation required for IdentityGate

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('eye2', format)

    def _eval_commutator(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator(self, other, **hints):
        return Integer(2)*other


class HadamardGate(HermitianOperator, OneQubitGate):
    """The single qubit Hadamard gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    >>> from sympy import sqrt
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> from sympy.physics.quantum.gate import HadamardGate
    >>> from sympy.physics.quantum.qapply import qapply
    >>> qapply(HadamardGate(0)*Qubit('1'))
    sqrt(2)*|0>/2 - sqrt(2)*|1>/2
    >>> # Hadamard on bell state, applied on 2 qubits.
    >>> psi = 1/sqrt(2)*(Qubit('00')+Qubit('11'))
    >>> qapply(HadamardGate(0)*HadamardGate(1)*psi)
    sqrt(2)*|00>/2 + sqrt(2)*|11>/2

    """
    gate_name = 'H'
    gate_name_latex = 'H'

    def get_target_matrix(self, format='sympy'):
        if _normalized:
            return matrix_cache.get_matrix('H', format)
        else:
            return matrix_cache.get_matrix('Hsqrt2', format)

    def _eval_commutator_XGate(self, other, **hints):
        return I*sqrt(2)*YGate(self.targets[0])

    def _eval_commutator_YGate(self, other, **hints):
        return I*sqrt(2)*(ZGate(self.targets[0]) - XGate(self.targets[0]))

    def _eval_commutator_ZGate(self, other, **hints):
        return -I*sqrt(2)*YGate(self.targets[0])

    def _eval_anticommutator_XGate(self, other, **hints):
        return sqrt(2)*IdentityGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator_ZGate(self, other, **hints):
        return sqrt(2)*IdentityGate(self.targets[0])


class XGate(HermitianOperator, OneQubitGate):
    """The single qubit X, or NOT, gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'X'
    gate_name_latex = 'X'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('X', format)

    def plot_gate(self, circ_plot, gate_idx):
        OneQubitGate.plot_gate(self,circ_plot,gate_idx)

    def plot_gate_plus(self, circ_plot, gate_idx):
        circ_plot.not_point(
            gate_idx, int(self.label[0])
        )

    def _eval_commutator_YGate(self, other, **hints):
        return Integer(2)*I*ZGate(self.targets[0])

    def _eval_anticommutator_XGate(self, other, **hints):
        return Integer(2)*IdentityGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero

    def _eval_anticommutator_ZGate(self, other, **hints):
        return _S.Zero


class YGate(HermitianOperator, OneQubitGate):
    """The single qubit Y gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'Y'
    gate_name_latex = 'Y'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('Y', format)

    def _eval_commutator_ZGate(self, other, **hints):
        return Integer(2)*I*XGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return Integer(2)*IdentityGate(self.targets[0])

    def _eval_anticommutator_ZGate(self, other, **hints):
        return _S.Zero


class ZGate(HermitianOperator, OneQubitGate):
    """The single qubit Z gate.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    gate_name = 'Z'
    gate_name_latex = 'Z'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('Z', format)

    def _eval_commutator_XGate(self, other, **hints):
        return Integer(2)*I*YGate(self.targets[0])

    def _eval_anticommutator_YGate(self, other, **hints):
        return _S.Zero


class PhaseGate(OneQubitGate):
    """The single qubit phase, or S, gate.

    This gate rotates the phase of the state by pi/2 if the state is ``|1>`` and
    does nothing if the state is ``|0>``.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian =  False
    gate_name = 'S'
    gate_name_latex = 'S'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('S', format)

    def _eval_commutator_ZGate(self, other, **hints):
        return _S.Zero

    def _eval_commutator_TGate(self, other, **hints):
        return _S.Zero


class TGate(OneQubitGate):
    """The single qubit pi/8 gate.

    This gate rotates the phase of the state by pi/4 if the state is ``|1>`` and
    does nothing if the state is ``|0>``.

    Parameters
    ----------
    target : int
        The target qubit this gate will apply to.

    Examples
    ========

    """
    is_hermitian = False
    gate_name = 'T'
    gate_name_latex = 'T'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('T', format)

    def _eval_commutator_ZGate(self, other, **hints):
        return _S.Zero

    def _eval_commutator_PhaseGate(self, other, **hints):
        return _S.Zero


# Aliases for gate names.
H = HadamardGate
X = XGate
Y = YGate
Z = ZGate
T = TGate
Phase = S = PhaseGate


#-----------------------------------------------------------------------------
# 2 Qubit Gates
#-----------------------------------------------------------------------------


class CNotGate(HermitianOperator, CGate, TwoQubitGate):
    """Two qubit controlled-NOT.

    This gate performs the NOT or X gate on the target qubit if the control
    qubits all have the value 1.

    Parameters
    ----------
    label : tuple
        A tuple of the form (control, target).

    Examples
    ========

    >>> from sympy.physics.quantum.gate import CNOT
    >>> from sympy.physics.quantum.qapply import qapply
    >>> from sympy.physics.quantum.qubit import Qubit
    >>> c = CNOT(1,0)
    >>> qapply(c*Qubit('10')) # note that qubits are indexed from right to left
    |11>

    """
    gate_name = 'CNOT'
    gate_name_latex = r'\text{CNOT}'
    simplify_cgate = True

    #-------------------------------------------------------------------------
    # Initialization
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        args = Gate._eval_args(args)
        return args

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**(_max(args) + 1)

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def min_qubits(self):
        """The minimum number of qubits this gate needs to act on."""
        return _max(self.label) + 1

    @property
    def targets(self):
        """A tuple of target qubits."""
        return (self.label[1],)

    @property
    def controls(self):
        """A tuple of control qubits."""
        return (self.label[0],)

    @property
    def gate(self):
        """The non-controlled gate that will be applied to the targets."""
        return XGate(self.label[1])

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    # The default printing of Gate works better than those of CGate, so we
    # go around the overridden methods in CGate.

    def _print_label(self, printer, *args):
        return Gate._print_label(self, printer, *args)

    def _pretty(self, printer, *args):
        return Gate._pretty(self, printer, *args)

    def _latex(self, printer, *args):
        return Gate._latex(self, printer, *args)

    #-------------------------------------------------------------------------
    # Commutator/AntiCommutator
    #-------------------------------------------------------------------------

    def _eval_commutator_ZGate(self, other, **hints):
        """[CNOT(i, j), Z(i)] == 0."""
        if self.controls[0] == other.targets[0]:
            return _S.Zero
        else:
            raise NotImplementedError('Commutator not implemented: %r' % other)

    def _eval_commutator_TGate(self, other, **hints):
        """[CNOT(i, j), T(i)] == 0."""
        return self._eval_commutator_ZGate(other, **hints)

    def _eval_commutator_PhaseGate(self, other, **hints):
        """[CNOT(i, j), S(i)] == 0."""
        return self._eval_commutator_ZGate(other, **hints)

    def _eval_commutator_XGate(self, other, **hints):
        """[CNOT(i, j), X(j)] == 0."""
        if self.targets[0] == other.targets[0]:
            return _S.Zero
        else:
            raise NotImplementedError('Commutator not implemented: %r' % other)

    def _eval_commutator_CNotGate(self, other, **hints):
        """[CNOT(i, j), CNOT(i,k)] == 0."""
        if self.controls[0] == other.controls[0]:
            return _S.Zero
        else:
            raise NotImplementedError('Commutator not implemented: %r' % other)


class SwapGate(TwoQubitGate):
    """Two qubit SWAP gate.

    This gate swap the values of the two qubits.

    Parameters
    ----------
    label : tuple
        A tuple of the form (target1, target2).

    Examples
    ========

    """
    is_hermitian = True
    gate_name = 'SWAP'
    gate_name_latex = r'\text{SWAP}'

    def get_target_matrix(self, format='sympy'):
        return matrix_cache.get_matrix('SWAP', format)

    def decompose(self, **options):
        """Decompose the SWAP gate into CNOT gates."""
        i, j = self.targets[0], self.targets[1]
        g1 = CNotGate(i, j)
        g2 = CNotGate(j, i)
        return g1*g2*g1

    def plot_gate(self, circ_plot, gate_idx):
        min_wire = int(_min(self.targets))
        max_wire = int(_max(self.targets))
        circ_plot.control_line(gate_idx, min_wire, max_wire)
        circ_plot.swap_point(gate_idx, min_wire)
        circ_plot.swap_point(gate_idx, max_wire)

    def _represent_ZGate(self, basis, **options):
        """Represent the SWAP gate in the computational basis.

        The following representation is used to compute this:

        SWAP = |1><1|x|1><1| + |0><0|x|0><0| + |1><0|x|0><1| + |0><1|x|1><0|
        """
        format = options.get('format', 'sympy')
        targets = [int(t) for t in self.targets]
        min_target = _min(targets)
        max_target = _max(targets)
        nqubits = options.get('nqubits', self.min_qubits)

        op01 = matrix_cache.get_matrix('op01', format)
        op10 = matrix_cache.get_matrix('op10', format)
        op11 = matrix_cache.get_matrix('op11', format)
        op00 = matrix_cache.get_matrix('op00', format)
        eye2 = matrix_cache.get_matrix('eye2', format)

        result = None
        for i, j in ((op01, op10), (op10, op01), (op00, op00), (op11, op11)):
            product = nqubits*[eye2]
            product[nqubits - min_target - 1] = i
            product[nqubits - max_target - 1] = j
            new_result = matrix_tensor_product(*product)
            if result is None:
                result = new_result
            else:
                result = result + new_result

        return result


# Aliases for gate names.
CNOT = CNotGate
SWAP = SwapGate
def CPHASE(a,b): return CGateS((a,),Z(b))


#-----------------------------------------------------------------------------
# Represent
#-----------------------------------------------------------------------------


def represent_zbasis(controls, targets, target_matrix, nqubits, format='sympy'):
    """Represent a gate with controls, targets and target_matrix.

    This function does the low-level work of representing gates as matrices
    in the standard computational basis (ZGate). Currently, we support two
    main cases:

    1. One target qubit and no control qubits.
    2. One target qubits and multiple control qubits.

    For the base of multiple controls, we use the following expression [1]:

    1_{2**n} + (|1><1|)^{(n-1)} x (target-matrix - 1_{2})

    Parameters
    ----------
    controls : list, tuple
        A sequence of control qubits.
    targets : list, tuple
        A sequence of target qubits.
    target_matrix : sympy.Matrix, numpy.matrix, scipy.sparse
        The matrix form of the transformation to be performed on the target
        qubits.  The format of this matrix must match that passed into
        the `format` argument.
    nqubits : int
        The total number of qubits used for the representation.
    format : str
        The format of the final matrix ('sympy', 'numpy', 'scipy.sparse').

    Examples
    ========

    References
    ----------
    [1] http://www.johnlapeyre.com/qinf/qinf_html/node6.html.
    """
    controls = [int(x) for x in controls]
    targets = [int(x) for x in targets]
    nqubits = int(nqubits)

    # This checks for the format as well.
    op11 = matrix_cache.get_matrix('op11', format)
    eye2 = matrix_cache.get_matrix('eye2', format)

    # Plain single qubit case
    if len(controls) == 0 and len(targets) == 1:
        product = []
        bit = targets[0]
        # Fill product with [I1,Gate,I2] such that the unitaries,
        # I, cause the gate to be applied to the correct Qubit
        if bit != nqubits - 1:
            product.append(matrix_eye(2**(nqubits - bit - 1), format=format))
        product.append(target_matrix)
        if bit != 0:
            product.append(matrix_eye(2**bit, format=format))
        return matrix_tensor_product(*product)

    # Single target, multiple controls.
    elif len(targets) == 1 and len(controls) >= 1:
        target = targets[0]

        # Build the non-trivial part.
        product2 = []
        for i in range(nqubits):
            product2.append(matrix_eye(2, format=format))
        for control in controls:
            product2[nqubits - 1 - control] = op11
        product2[nqubits - 1 - target] = target_matrix - eye2

        return matrix_eye(2**nqubits, format=format) + \
            matrix_tensor_product(*product2)

    # Multi-target, multi-control is not yet implemented.
    else:
        raise NotImplementedError(
            'The representation of multi-target, multi-control gates '
            'is not implemented.'
        )


#-----------------------------------------------------------------------------
# Gate manipulation functions.
#-----------------------------------------------------------------------------


def gate_simp(circuit):
    """Simplifies gates symbolically

    It first sorts gates using gate_sort. It then applies basic
    simplification rules to the circuit, e.g., XGate**2 = Identity
    """

    # Bubble sort out gates that commute.
    circuit = gate_sort(circuit)

    # Do simplifications by subing a simplification into the first element
    # which can be simplified. We recursively call gate_simp with new circuit
    # as input more simplifications exist.
    if isinstance(circuit, Add):
        return sum(gate_simp(t) for t in circuit.args)
    elif isinstance(circuit, Mul):
        circuit_args = circuit.args
    elif isinstance(circuit, Pow):
        b, e = circuit.as_base_exp()
        circuit_args = (gate_simp(b)**e,)
    else:
        return circuit

    # Iterate through each element in circuit, simplify if possible.
    for i in range(len(circuit_args)):
        # H,X,Y or Z squared is 1.
        # T**2 = S, S**2 = Z
        if isinstance(circuit_args[i], Pow):
            if isinstance(circuit_args[i].base,
                (HadamardGate, XGate, YGate, ZGate)) \
                    and isinstance(circuit_args[i].exp, Number):
                # Build a new circuit taking replacing the
                # H,X,Y,Z squared with one.
                newargs = (circuit_args[:i] +
                          (circuit_args[i].base**(circuit_args[i].exp % 2),) +
                           circuit_args[i + 1:])
                # Recursively simplify the new circuit.
                circuit = gate_simp(Mul(*newargs))
                break
            elif isinstance(circuit_args[i].base, PhaseGate):
                # Build a new circuit taking old circuit but splicing
                # in simplification.
                newargs = circuit_args[:i]
                # Replace PhaseGate**2 with ZGate.
                newargs = newargs + (ZGate(circuit_args[i].base.args[0])**
                (Integer(circuit_args[i].exp/2)), circuit_args[i].base**
                (circuit_args[i].exp % 2))
                # Append the last elements.
                newargs = newargs + circuit_args[i + 1:]
                # Recursively simplify the new circuit.
                circuit = gate_simp(Mul(*newargs))
                break
            elif isinstance(circuit_args[i].base, TGate):
                # Build a new circuit taking all the old elements.
                newargs = circuit_args[:i]

                # Put an Phasegate in place of any TGate**2.
                newargs = newargs + (PhaseGate(circuit_args[i].base.args[0])**
                Integer(circuit_args[i].exp/2), circuit_args[i].base**
                    (circuit_args[i].exp % 2))

                # Append the last elements.
                newargs = newargs + circuit_args[i + 1:]
                # Recursively simplify the new circuit.
                circuit = gate_simp(Mul(*newargs))
                break
    return circuit


def gate_sort(circuit):
    """Sorts the gates while keeping track of commutation relations

    This function uses a bubble sort to rearrange the order of gate
    application. Keeps track of Quantum computations special commutation
    relations (e.g. things that apply to the same Qubit do not commute with
    each other)

    circuit is the Mul of gates that are to be sorted.
    """
    # Make sure we have an Add or Mul.
    if isinstance(circuit, Add):
        return sum(gate_sort(t) for t in circuit.args)
    if isinstance(circuit, Pow):
        return gate_sort(circuit.base)**circuit.exp
    elif isinstance(circuit, Gate):
        return circuit
    if not isinstance(circuit, Mul):
        return circuit

    changes = True
    while changes:
        changes = False
        circ_array = circuit.args
        for i in range(len(circ_array) - 1):
            # Go through each element and switch ones that are in wrong order
            if isinstance(circ_array[i], (Gate, Pow)) and \
                    isinstance(circ_array[i + 1], (Gate, Pow)):
                # If we have a Pow object, look at only the base
                first_base, first_exp = circ_array[i].as_base_exp()
                second_base, second_exp = circ_array[i + 1].as_base_exp()

                # Use SymPy's hash based sorting. This is not mathematical
                # sorting, but is rather based on comparing hashes of objects.
                # See Basic.compare for details.
                if first_base.compare(second_base) > 0:
                    if Commutator(first_base, second_base).doit() == 0:
                        new_args = (circuit.args[:i] + (circuit.args[i + 1],) +
                                   (circuit.args[i],) + circuit.args[i + 2:])
                        circuit = Mul(*new_args)
                        changes = True
                        break
                    if AntiCommutator(first_base, second_base).doit() == 0:
                        new_args = (circuit.args[:i] + (circuit.args[i + 1],) +
                                   (circuit.args[i],) + circuit.args[i + 2:])
                        sign = _S.NegativeOne**(first_exp*second_exp)
                        circuit = sign*Mul(*new_args)
                        changes = True
                        break
    return circuit


#-----------------------------------------------------------------------------
# Utility functions
#-----------------------------------------------------------------------------


def random_circuit(ngates, nqubits, gate_space=(X, Y, Z, S, T, H, CNOT, SWAP)):
    """Return a random circuit of ngates and nqubits.

    This uses an equally weighted sample of (X, Y, Z, S, T, H, CNOT, SWAP)
    gates.

    Parameters
    ----------
    ngates : int
        The number of gates in the circuit.
    nqubits : int
        The number of qubits in the circuit.
    gate_space : tuple
        A tuple of the gate classes that will be used in the circuit.
        Repeating gate classes multiple times in this tuple will increase
        the frequency they appear in the random circuit.
    """
    qubit_space = range(nqubits)
    result = []
    for i in range(ngates):
        g = random.choice(gate_space)
        if g == CNotGate or g == SwapGate:
            qubits = random.sample(qubit_space, 2)
            g = g(*qubits)
        else:
            qubit = random.choice(qubit_space)
            g = g(qubit)
        result.append(g)
    return Mul(*result)


def zx_basis_transform(self, format='sympy'):
    """Transformation matrix from Z to X basis."""
    return matrix_cache.get_matrix('ZX', format)


def zy_basis_transform(self, format='sympy'):
    """Transformation matrix from Z to Y basis."""
    return matrix_cache.get_matrix('ZY', format)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_attention_unet.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

import numpy as np
from fusion_base import Fusion
from fusion_utils import NumpyHelper
from onnx import NodeProto, TensorProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionAttentionUnet(Fusion):
    """
    Fuse Attention subgraph of UNet into one Attention node.
    """

    def __init__(
        self,
        model: OnnxModel,
        hidden_size: int,
        num_heads: int,
        is_cross_attention: bool,
        enable_packed_qkv: bool,
        enable_packed_kv: bool,
    ):
        super().__init__(
            model,
            "Attention" if is_cross_attention and enable_packed_qkv else "MultiHeadAttention",
            ["LayerNormalization"],
        )
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.is_cross_attention = is_cross_attention

        # Note: pack Q/K/V or K/V weights into one tensor make it harder for updating initializers for LoRA.
        # To support LoRA, it is better to use separated Q, K and V inputs in offline optimization,
        # and CUDA operator pre-packs those tensors to preferred format based on available kernels.
        # In this way, we can support LoRA and get optimal performance at same time.
        self.enable_packed_qkv = enable_packed_qkv
        self.enable_packed_kv = enable_packed_kv

        # Flags to show warning only once
        self.num_heads_warning = True
        self.hidden_size_warning = True

    def get_num_heads(self, reshape_q: NodeProto, is_torch2: bool = False) -> int:
        """Detect num_heads from a reshape node.

        Args:
            reshape_q (NodeProto): reshape node for Q
            is_torch2 (bool): graph pattern is from PyTorch 2.*
        Returns:
            int: num_heads, or 0 if not found
        """
        num_heads = 0
        if is_torch2:
            # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
            reshape_parent = self.model.get_parent(reshape_q, 1)
            if reshape_parent and reshape_parent.op_type == "Concat" and len(reshape_parent.input) == 4:
                num_heads = self.model.get_constant_value(reshape_parent.input[2])
                if isinstance(num_heads, np.ndarray) and list(num_heads.shape) == [1]:
                    num_heads = int(num_heads)
        else:
            # we assume that reshape fusion has done, so the shape is a tensor like [0, 0, num_heads, head_size]
            q_shape_value = self.model.get_constant_value(reshape_q.input[1])
            if isinstance(q_shape_value, np.ndarray) and list(q_shape_value.shape) == [4]:
                num_heads = int(q_shape_value[2])

        if isinstance(num_heads, int) and num_heads > 0:
            return num_heads

        return 0

    def get_hidden_size(self, layernorm_node):
        """Detect hidden_size from LayerNormalization node.
        Args:
            layernorm_node (NodeProto): LayerNormalization node before Q, K and V
        Returns:
            int: hidden_size, or 0 if not found
        """
        layernorm_bias = self.model.get_initializer(layernorm_node.input[2])
        if layernorm_bias:
            return NumpyHelper.to_array(layernorm_bias).shape[0]

        return 0

    def get_num_heads_and_hidden_size(
        self, reshape_q: NodeProto, layernorm_node: NodeProto, is_torch2: bool = False
    ) -> tuple[int, int]:
        """Detect num_heads and hidden_size.

        Args:
            reshape_q (NodeProto): reshape node for Q
            is_torch2 (bool): graph pattern is from PyTorch 2.*
            layernorm_node (NodeProto): LayerNormalization node before Q, K, V
        Returns:
            Tuple[int, int]: num_heads and hidden_size
        """
        num_heads = self.get_num_heads(reshape_q, is_torch2)
        if num_heads <= 0:
            num_heads = self.num_heads  # Fall back to user specified value

        if self.num_heads > 0 and num_heads != self.num_heads:
            if self.num_heads_warning:
                logger.warning(f"--num_heads is {self.num_heads}. Detected value is {num_heads}. Using detected value.")
                self.num_heads_warning = False  # Do not show the warning more than once

        hidden_size = self.get_hidden_size(layernorm_node)
        if hidden_size <= 0:
            hidden_size = self.hidden_size  # Fall back to user specified value

        if self.hidden_size > 0 and hidden_size != self.hidden_size:
            if self.hidden_size_warning:
                logger.warning(
                    f"--hidden_size is {self.hidden_size}. Detected value is {hidden_size}. Using detected value."
                )
                self.hidden_size_warning = False  # Do not show the warning more than once

        return num_heads, hidden_size

    def create_attention_node(
        self,
        q_matmul: NodeProto,
        k_matmul: NodeProto,
        v_matmul: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
    ) -> NodeProto | None:
        """Create an Attention node.

        Args:
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        is_self_attention = not self.is_cross_attention

        if is_self_attention:
            if q_matmul.input[0] != input or k_matmul.input[0] != input or v_matmul.input[0] != input:
                logger.debug(
                    "For self attention, input hidden state for q and k/v shall be same. Got %s, %s, %s",
                    q_matmul.input[0],
                    k_matmul.input[0],
                    v_matmul.input[0],
                )
                return None
        else:
            if q_matmul.input[0] != input or (k_matmul.input[0] != v_matmul.input[0]) or (k_matmul.input[0] == input):
                logger.debug(
                    "For cross attention, input hidden state for q and k/v shall be different. Got %s, %s, %s",
                    q_matmul.input[0],
                    k_matmul.input[0],
                    v_matmul.input[0],
                )
                return None

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        if not (q_weight and k_weight and v_weight):
            return None

        # Sometimes weights are stored in fp16
        float_type = q_weight.data_type

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)
        logger.debug(f"qw={qw.shape} kw={kw.shape} vw={vw.shape} hidden_size={hidden_size}")

        # assert q and k have same shape as expected
        if is_self_attention:
            if qw.shape != kw.shape or qw.shape != vw.shape:
                return None

            qw_in_size = qw.shape[0]

            if hidden_size > 0 and hidden_size != qw_in_size:
                raise ValueError(
                    f"Input hidden size ({hidden_size}) is not same as weight dimension of q,k,v ({qw_in_size}). "
                    "Please provide a correct input hidden size or pass in 0"
                )

            # All the matrices can have the same shape or q, k matrics can have the same shape with v being different
            # For 2d weights, the shapes would be [in_size, out_size].
            # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
            qw_out_size = int(np.prod(qw.shape[1:]))

            if self.enable_packed_qkv:
                attention_node_name = self.model.create_node_name("MultiHeadAttention")

                c = qw_in_size
                n = num_heads
                h = qw_out_size // num_heads

                # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 3, H] shape
                qkv_weight = np.dstack([qw.reshape(c, n, h), kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(
                    c, n * 3 * h
                )

                matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_QKV")
                self.add_initializer(
                    name=matmul_node_name + "_weight",
                    data_type=float_type,
                    dims=[qkv_weight.shape[0], qkv_weight.shape[1]],
                    vals=qkv_weight,
                )

                matmul_node = helper.make_node(
                    "MatMul",
                    inputs=[k_matmul.input[0], matmul_node_name + "_weight"],
                    outputs=[matmul_node_name + "_out"],
                    name=matmul_node_name,
                )
                self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

                self.add_initializer(
                    name=matmul_node_name + "_reshape_shape",
                    data_type=TensorProto.INT64,
                    dims=[5],
                    vals=[0, 0, n, 3, h],
                    raw=False,
                )

                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[
                        matmul_node_name + "_out",
                        matmul_node_name + "_reshape_shape",
                    ],
                    outputs=[attention_node_name + "_qkv_input"],
                    name=matmul_node_name + "_reshape",
                )
                self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name
                self.nodes_to_add.extend([matmul_node, reshape_node])
                self.nodes_to_remove.extend([q_matmul, k_matmul, v_matmul])

            else:
                qkv_weight = np.stack((qw, kw, vw), axis=1)
                qkv_weight_dim = 3 * qw_out_size

                attention_node_name = self.model.create_node_name("Attention")

                self.add_initializer(
                    name=attention_node_name + "_qkv_weight",
                    data_type=float_type,
                    dims=[qw_in_size, qkv_weight_dim],
                    vals=qkv_weight,
                )
        else:  # cross attention
            attention_node_name = self.model.create_node_name("MultiHeadAttention")
            if self.enable_packed_kv:
                if kw.shape != vw.shape:
                    return None

                kw_in_size = kw.shape[0]
                vw_in_size = vw.shape[0]
                assert kw_in_size == vw_in_size

                qw_out_size = qw.shape[1]
                kw_out_size = kw.shape[1]
                vw_out_size = vw.shape[1]
                assert qw_out_size == vw_out_size and kw_out_size == vw_out_size

                c = kw_in_size
                n = num_heads
                h = kw_out_size // num_heads

                # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 2, H] shape
                kv_weight = np.dstack([kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(c, n * 2 * h)

                matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_KV")
                self.add_initializer(
                    name=matmul_node_name + "_weight",
                    data_type=float_type,
                    dims=[kv_weight.shape[0], kv_weight.shape[1]],
                    vals=kv_weight,
                )

                matmul_node = helper.make_node(
                    "MatMul",
                    inputs=[k_matmul.input[0], matmul_node_name + "_weight"],
                    outputs=[matmul_node_name + "_out"],
                    name=matmul_node_name,
                )
                self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

                self.add_initializer(
                    name=matmul_node_name + "_reshape_shape",
                    data_type=TensorProto.INT64,
                    dims=[5],
                    vals=[0, 0, n, 2, h],
                    raw=False,
                )

                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[
                        matmul_node_name + "_out",
                        matmul_node_name + "_reshape_shape",
                    ],
                    outputs=[attention_node_name + "_kv_input"],
                    name=matmul_node_name + "_reshape",
                )
                self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name
                self.nodes_to_add.extend([matmul_node, reshape_node])
                self.nodes_to_remove.extend([k_matmul, v_matmul])

        # No bias, use zeros
        qkv_bias = np.zeros([3, hidden_size], dtype=np.float32)
        qkv_bias_dim = 3 * hidden_size

        self.add_initializer(
            name=attention_node_name + "_qkv_bias",
            data_type=float_type,
            dims=[qkv_bias_dim],
            vals=qkv_bias,
        )

        if is_self_attention:
            if not self.enable_packed_qkv:
                attention_inputs = [
                    input,
                    attention_node_name + "_qkv_weight",
                    attention_node_name + "_qkv_bias",
                ]
            else:
                attention_inputs = [attention_node_name + "_qkv_input"]
        else:
            if not self.enable_packed_kv:
                attention_inputs = [
                    q_matmul.output[0],
                    k_matmul.output[0],
                    v_matmul.output[0],
                    attention_node_name + "_qkv_bias",
                ]
            else:
                attention_inputs = [
                    q_matmul.output[0],
                    attention_node_name + "_kv_input",
                ]

        attention_node = helper.make_node(
            "Attention" if (is_self_attention and not self.enable_packed_qkv) else "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        counter_name = (
            "Attention (self attention)"
            if is_self_attention and not self.enable_packed_qkv
            else "MultiHeadAttention ({})".format(
                "self attention with packed qkv"
                if self.enable_packed_qkv
                else "cross attention with packed kv"
                if self.enable_packed_kv
                else "cross attention"
            )
        )
        self.increase_counter(counter_name)
        return attention_node

    def create_attention_node_lora(
        self,
        q_matmul_add: NodeProto,
        k_matmul_add: NodeProto,
        v_matmul_add: NodeProto,
        num_heads: int,
        hidden_size: int,
        input: str,
        output: str,
    ) -> NodeProto | None:
        """Create an Attention node.

        Args:
            q_matmul (NodeProto): MatMul node in fully connection for Q
            k_matmul (NodeProto): MatMul node in fully connection for K
            v_matmul (NodeProto): MatMul node in fully connection for V
            num_heads (int): number of attention heads. If a model is pruned, it is the number of heads after pruning.
            hidden_size (int): hidden dimension. If a model is pruned, it is the hidden dimension after pruning.
            input (str): input name
            output (str): output name

        Returns:
            Union[NodeProto, None]: the node created or None if failed.
        """
        is_self_attention = not self.is_cross_attention

        q_matmul = self.model.match_parent(q_matmul_add, "MatMul", 0)
        k_matmul = self.model.match_parent(k_matmul_add, "MatMul", 0)
        v_matmul = self.model.match_parent(v_matmul_add, "MatMul", 0)

        q_lora_nodes = self.match_lora_path(q_matmul_add)
        if q_lora_nodes is None:
            return None
        (q_lora_last_node, q_lora_matmul_1) = q_lora_nodes

        k_lora_nodes = self.match_lora_path(k_matmul_add)
        if k_lora_nodes is None:
            return None
        (k_lora_last_node, k_lora_matmul_1) = k_lora_nodes

        v_lora_nodes = self.match_lora_path(v_matmul_add)
        if v_lora_nodes is None:
            return None
        (v_lora_last_node, v_lora_matmul_1) = v_lora_nodes

        if is_self_attention:
            if q_matmul.input[0] != input or k_matmul.input[0] != input or v_matmul.input[0] != input:
                logger.debug(
                    "For self attention, input hidden state for q and k/v shall be same. Got %s, %s, %s",
                    q_matmul.input[0],
                    k_matmul.input[0],
                    v_matmul.input[0],
                )
                return None

            if (
                q_lora_matmul_1.input[0] != input
                or k_lora_matmul_1.input[0] != input
                or v_lora_matmul_1.input[0] != input
            ):
                logger.debug(
                    "For self attention, input hidden state for LoRA q and k/v weights shall be same. Got %s, %s, %s",
                    q_lora_matmul_1.input[0],
                    k_lora_matmul_1.input[0],
                    v_lora_matmul_1.input[0],
                )
                return None
        else:
            if q_matmul.input[0] != input or (k_matmul.input[0] != v_matmul.input[0]) or (k_matmul.input[0] == input):
                logger.debug(
                    "For cross attention, input hidden state for q and k/v shall be different. Got %s, %s, %s",
                    q_matmul.input[0],
                    k_matmul.input[0],
                    v_matmul.input[0],
                )
                return None

            if (
                q_lora_matmul_1.input[0] != input
                or (k_lora_matmul_1.input[0] != v_lora_matmul_1.input[0])
                or (k_matmul.input[0] == input)
            ):
                logger.debug(
                    (
                        "For cross attention, input hidden state for LoRA q and k/v weights shall be different. "
                        "Got %s, %s, %s"
                    ),
                    q_lora_matmul_1.input[0],
                    k_lora_matmul_1.input[0],
                    v_lora_matmul_1.input[0],
                )
                return None

        if hidden_size > 0 and (hidden_size % num_heads) != 0:
            logger.debug(f"input hidden size {hidden_size} is not a multiple of num of heads {num_heads}")
            return None

        q_weight = self.model.get_initializer(q_matmul.input[1])
        k_weight = self.model.get_initializer(k_matmul.input[1])
        v_weight = self.model.get_initializer(v_matmul.input[1])
        if not (q_weight and k_weight and v_weight):
            return None

        # Sometimes weights are stored in fp16
        if q_weight.data_type == 10:
            logger.debug("weights are in fp16. Please run fp16 conversion after optimization")
            return None

        qw = NumpyHelper.to_array(q_weight)
        kw = NumpyHelper.to_array(k_weight)
        vw = NumpyHelper.to_array(v_weight)
        logger.debug(f"qw={qw.shape} kw={kw.shape} vw={vw.shape} hidden_size={hidden_size}")

        # assert q and k have same shape as expected
        if is_self_attention:
            if qw.shape != kw.shape or qw.shape != vw.shape:
                return None

            qw_in_size = qw.shape[0]

            if hidden_size > 0 and hidden_size != qw_in_size:
                raise ValueError(
                    f"Input hidden size ({hidden_size}) is not same as weight dimension of q,k,v ({qw_in_size}). "
                    "Please provide a correct input hidden size or pass in 0"
                )

            # All the matrices can have the same shape or q, k matrics can have the same shape with v being different
            # For 2d weights, the shapes would be [in_size, out_size].
            # For 3d weights, shape would be [in_size, a, b] where a*b = out_size
            qw_out_size = int(np.prod(qw.shape[1:]))

            if self.enable_packed_qkv:
                attention_node_name = self.model.create_node_name("MultiHeadAttention")

                c = qw_in_size
                n = num_heads
                h = qw_out_size // num_heads

                # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 3, H] shape
                qkv_weight = np.dstack([qw.reshape(c, n, h), kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(
                    c, n * 3 * h
                )

                matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_QKV")
                self.add_initializer(
                    name=matmul_node_name + "_weight",
                    data_type=TensorProto.FLOAT,
                    dims=[qkv_weight.shape[0], qkv_weight.shape[1]],
                    vals=qkv_weight,
                )

                matmul_node = helper.make_node(
                    "MatMul",
                    inputs=[k_matmul.input[0], matmul_node_name + "_weight"],
                    outputs=[matmul_node_name + "_out"],
                    name=matmul_node_name,
                )
                self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

                # Do the same thing with the LoRA weights, but don't constant fold the result. The goal is to allow
                # the Q/K/V weights to be changed without having to re-run the optimizer.
                lora_weight_shape_tensor_name = q_lora_last_node.name + "_reshape_shape"

                self.add_initializer(
                    name=lora_weight_shape_tensor_name,
                    data_type=TensorProto.INT64,
                    dims=[4],
                    vals=[0, 0, n, h],
                    raw=False,
                )

                # Reshape the LoRA Q weights
                q_lora_reshape_node_name = self.model.create_node_name("Reshape", name_prefix="Reshape_LoRA_Q")
                q_lora_reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[q_lora_last_node.output[0], lora_weight_shape_tensor_name],
                    outputs=[q_lora_reshape_node_name + "_out"],
                    name=q_lora_reshape_node_name,
                )
                self.node_name_to_graph_name[q_lora_reshape_node.name] = self.this_graph_name

                # Reshape the LoRA K weights
                k_lora_reshape_node_name = self.model.create_node_name("Reshape", name_prefix="Reshape_LoRA_K")
                k_lora_reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[k_lora_last_node.output[0], lora_weight_shape_tensor_name],
                    outputs=[k_lora_reshape_node_name + "_out"],
                    name=k_lora_reshape_node_name,
                )
                self.node_name_to_graph_name[k_lora_reshape_node.name] = self.this_graph_name

                # Reshape the LoRA V weights
                v_lora_reshape_node_name = self.model.create_node_name("Reshape", name_prefix="Reshape_LoRA_V")
                v_lora_reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[v_lora_last_node.output[0], lora_weight_shape_tensor_name],
                    outputs=[v_lora_reshape_node_name + "_out"],
                    name=v_lora_reshape_node_name,
                )
                self.node_name_to_graph_name[v_lora_reshape_node.name] = self.this_graph_name

                # Concat the reshaped LoRA Q/K/V weights together on the third axis
                qkv_lora_concat_node_name = self.model.create_node_name("Concat", name_prefix="Concat_LoRA_QKV")
                qkv_lora_concat_node = helper.make_node(
                    "Concat",
                    inputs=[
                        q_lora_reshape_node.output[0],
                        k_lora_reshape_node.output[0],
                        v_lora_reshape_node.output[0],
                    ],
                    outputs=[qkv_lora_concat_node_name + "_out"],
                    name=qkv_lora_concat_node_name,
                )
                qkv_lora_concat_node.attribute.extend([helper.make_attribute("axis", 3)])
                self.node_name_to_graph_name[qkv_lora_concat_node.name] = self.this_graph_name

                # Reshape the LoRA concatenated weights to [..., n * 3 * h]
                reshaped_lora_weights_shape_tensor_name = qkv_lora_concat_node.name + "_reshape_shape"
                self.add_initializer(
                    name=reshaped_lora_weights_shape_tensor_name,
                    data_type=TensorProto.INT64,
                    dims=[3],
                    vals=[0, 0, n * 3 * h],
                    raw=False,
                )

                qkv_lora_reshaped_node_name = self.model.create_node_name("Reshape", name_prefix="Reshape_LoRA_QKV")
                qkv_lora_reshaped_node = helper.make_node(
                    "Reshape",
                    inputs=[qkv_lora_concat_node.output[0], reshaped_lora_weights_shape_tensor_name],
                    outputs=[qkv_lora_reshaped_node_name + "_out"],
                    name=qkv_lora_reshaped_node_name,
                )
                self.node_name_to_graph_name[qkv_lora_reshaped_node.name] = self.this_graph_name

                # Add the LoRA Q/K/V weights to the base Q/K/V weights
                add_weights_node_name = self.model.create_node_name("Add", name_prefix="Add_Weights_QKV")
                add_weights_node = helper.make_node(
                    "Add",
                    inputs=[qkv_lora_reshaped_node.output[0], matmul_node.output[0]],
                    outputs=[add_weights_node_name + "_out"],
                    name=add_weights_node_name,
                )
                self.node_name_to_graph_name[add_weights_node.name] = self.this_graph_name

                # Finally, reshape the concatenated Q/K/V result to 5D
                shape_tensor_name = add_weights_node_name + "_reshape_shape"
                self.add_initializer(
                    name=shape_tensor_name,
                    data_type=TensorProto.INT64,
                    dims=[5],
                    vals=[0, 0, n, 3, h],
                    raw=False,
                )

                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[add_weights_node.output[0], shape_tensor_name],
                    outputs=[attention_node_name + "_qkv_input"],
                    name=add_weights_node_name + "_reshape",
                )
                self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name

                self.nodes_to_add.extend(
                    [
                        matmul_node,
                        q_lora_reshape_node,
                        k_lora_reshape_node,
                        v_lora_reshape_node,
                        qkv_lora_concat_node,
                        qkv_lora_reshaped_node,
                        add_weights_node,
                        reshape_node,
                    ]
                )
                self.nodes_to_remove.extend([q_matmul, k_matmul, v_matmul, q_matmul_add, k_matmul_add, v_matmul_add])
            else:
                # TODO: Support non-packed QKV
                return None
        else:  # cross attention
            attention_node_name = self.model.create_node_name("MultiHeadAttention")
            if self.enable_packed_kv:
                if kw.shape != vw.shape:
                    return None

                kw_in_size = kw.shape[0]
                vw_in_size = vw.shape[0]
                assert kw_in_size == vw_in_size

                qw_out_size = qw.shape[1]
                kw_out_size = kw.shape[1]
                vw_out_size = vw.shape[1]
                assert qw_out_size == vw_out_size and kw_out_size == vw_out_size

                c = kw_in_size
                n = num_heads
                h = kw_out_size // num_heads

                # Concat and interleave weights so that the output of fused KV GEMM has [B, S_kv, N, 2, H] shape
                kv_weight = np.dstack([kw.reshape(c, n, h), vw.reshape(c, n, h)]).reshape(c, n * 2 * h)

                matmul_node_name = self.model.create_node_name("MatMul", name_prefix="MatMul_KV")
                self.add_initializer(
                    name=matmul_node_name + "_weight",
                    data_type=TensorProto.FLOAT,
                    dims=[kv_weight.shape[0], kv_weight.shape[1]],
                    vals=kv_weight,
                )

                matmul_node = helper.make_node(
                    "MatMul",
                    inputs=[k_matmul.input[0], matmul_node_name + "_weight"],
                    outputs=[matmul_node_name + "_out"],
                    name=matmul_node_name,
                )
                self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name

                # Do the same thing with the LoRA weights, but don't constant fold the result. The goal is to allow
                # the Q/K/V weights to be changed without having to re-run the optimizer.
                kv_lora_weight_shape_tensor_name = q_lora_last_node.name + "_reshape_shape"
                self.add_initializer(
                    name=kv_lora_weight_shape_tensor_name,
                    data_type=TensorProto.INT64,
                    dims=[4],
                    vals=[0, 0, n, h],
                    raw=False,
                )

                # Reshape the LoRA K weights
                k_lora_reshape_node_name = self.model.create_node_name("Reshape", name_prefix="Reshape_LoRA_K")
                k_lora_reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[k_lora_last_node.output[0], kv_lora_weight_shape_tensor_name],
                    outputs=[k_lora_reshape_node_name + "_out"],
                    name=k_lora_reshape_node_name,
                )
                self.node_name_to_graph_name[k_lora_reshape_node.name] = self.this_graph_name

                # Reshape the LoRA V weights
                v_lora_reshape_node_name = self.model.create_node_name("Reshape", name_prefix="Reshape_LoRA_V")
                v_lora_reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[v_lora_last_node.output[0], kv_lora_weight_shape_tensor_name],
                    outputs=[v_lora_reshape_node_name + "_out"],
                    name=v_lora_reshape_node_name,
                )
                self.node_name_to_graph_name[v_lora_reshape_node.name] = self.this_graph_name

                # Concat the reshaped LoRA K/V weights together on the third axis
                kv_lora_concat_node_name = self.model.create_node_name("Concat", name_prefix="Concat_LoRA_KV")
                kv_lora_concat_node = helper.make_node(
                    "Concat",
                    inputs=[k_lora_reshape_node.output[0], v_lora_reshape_node.output[0]],
                    outputs=[kv_lora_concat_node_name + "_out"],
                    name=kv_lora_concat_node_name,
                )
                kv_lora_concat_node.attribute.extend([helper.make_attribute("axis", 3)])
                self.node_name_to_graph_name[kv_lora_concat_node.name] = self.this_graph_name

                # Reshape the LoRA concatenated weights to [..., n * 2 * h]
                reshaped_kv_lora_weights_shape_tensor_name = kv_lora_concat_node.name + "_reshape_shape"
                self.add_initializer(
                    name=reshaped_kv_lora_weights_shape_tensor_name,
                    data_type=TensorProto.INT64,
                    dims=[3],
                    vals=[0, 0, n * 2 * h],
                    raw=False,
                )

                kv_lora_reshaped_node_name = self.model.create_node_name("Reshape", name_prefix="Reshape_LoRA_KV")
                kv_lora_reshaped_node = helper.make_node(
                    "Reshape",
                    inputs=[kv_lora_concat_node.output[0], reshaped_kv_lora_weights_shape_tensor_name],
                    outputs=[kv_lora_reshaped_node_name + "_out"],
                    name=kv_lora_reshaped_node_name,
                )
                self.node_name_to_graph_name[kv_lora_reshaped_node.name] = self.this_graph_name

                # Add the LoRA K/V weights to the base K/V weights
                add_kv_weights_node_name = self.model.create_node_name("Add", name_prefix="Add_Weights_KV")
                add_kv_weights_node = helper.make_node(
                    "Add",
                    inputs=[kv_lora_reshaped_node.output[0], matmul_node.output[0]],
                    outputs=[add_kv_weights_node_name + "_out"],
                    name=add_kv_weights_node_name,
                )
                self.node_name_to_graph_name[add_kv_weights_node.name] = self.this_graph_name

                # Finally, reshape the concatenated K/V result to 5D
                shape_tensor_name = add_kv_weights_node_name + "_reshape_shape"
                self.add_initializer(
                    name=shape_tensor_name,
                    data_type=TensorProto.INT64,
                    dims=[5],
                    vals=[0, 0, n, 2, h],
                    raw=False,
                )

                reshape_node = helper.make_node(
                    "Reshape",
                    inputs=[add_kv_weights_node.output[0], shape_tensor_name],
                    outputs=[attention_node_name + "_kv_input"],
                    name=add_kv_weights_node_name + "_reshape",
                )
                self.node_name_to_graph_name[reshape_node.name] = self.this_graph_name
                self.nodes_to_add.extend(
                    [
                        matmul_node,
                        k_lora_reshape_node,
                        v_lora_reshape_node,
                        kv_lora_concat_node,
                        kv_lora_reshaped_node,
                        add_kv_weights_node,
                        reshape_node,
                    ]
                )
                self.nodes_to_remove.extend([k_matmul, v_matmul, k_matmul_add, v_matmul_add])
            else:
                # TODO: Support non-packed KV
                return None

        # No bias, use zeros
        qkv_bias = np.zeros([3, hidden_size], dtype=np.float32)
        qkv_bias_dim = 3 * hidden_size
        self.add_initializer(
            name=attention_node_name + "_qkv_bias",
            data_type=TensorProto.FLOAT,
            dims=[qkv_bias_dim],
            vals=qkv_bias,
        )

        if is_self_attention:
            if not self.enable_packed_qkv:
                # TODO: Support non-packed QKV
                return None
            else:
                attention_inputs = [attention_node_name + "_qkv_input"]
        else:
            if not self.enable_packed_kv:
                # TODO: Support non-packed QKV
                return None
            else:
                attention_inputs = [
                    q_matmul_add.output[0],
                    attention_node_name + "_kv_input",
                ]

        attention_node = helper.make_node(
            "Attention" if (is_self_attention and not self.enable_packed_qkv) else "MultiHeadAttention",
            inputs=attention_inputs,
            outputs=[output],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend([helper.make_attribute("num_heads", num_heads)])

        counter_name = (
            "Attention (self attention)"
            if is_self_attention and not self.enable_packed_qkv
            else "MultiHeadAttention ({})".format(
                "self attention with packed qkv"
                if self.enable_packed_qkv
                else "cross attention with packed kv"
                if self.enable_packed_kv
                else "cross attention"
            )
        )
        self.increase_counter(counter_name)
        return attention_node

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        if self.fuse_a1111_fp16(normalize_node, input_name_to_nodes, output_name_to_node):
            return

        node_before_layernorm = self.model.match_parent(normalize_node, "Add", 0)

        # In SD 1.5, for self attention, LayerNorm has parent Reshape
        if node_before_layernorm is None and not self.is_cross_attention:
            node_before_layernorm = self.model.match_parent(normalize_node, "Reshape", 0)

        if node_before_layernorm is None:
            return

        root_input = node_before_layernorm.output[0]

        children_nodes = input_name_to_nodes[root_input]
        skip_add = None
        for node in children_nodes:
            if node.op_type == "Add":  # SkipLayerNormalization fusion is not applied yet
                skip_add = node
                break
        if skip_add is None:
            return

        match_qkv = self.match_qkv_torch1(root_input, skip_add) or self.match_qkv_torch2(root_input, skip_add)
        if match_qkv is not None:
            is_torch2, reshape_qkv, transpose_qkv, reshape_q, matmul_q, matmul_k, matmul_v = match_qkv

            attention_last_node = reshape_qkv

            q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q, normalize_node, is_torch2)
            if q_num_heads <= 0:
                logger.debug("fuse_attention: failed to detect num_heads")
                return

            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            new_node = self.create_attention_node(
                matmul_q,
                matmul_k,
                matmul_v,
                q_num_heads,
                q_hidden_size,
                input=normalize_node.output[0],
                output=attention_last_node.output[0],
            )
            if new_node is None:
                return
        else:
            # Check if we have a LoRA pattern
            match_qkv = self.match_qkv_torch1_lora(root_input, skip_add) or self.match_qkv_torch2_lora(
                root_input, skip_add
            )
            if match_qkv is None:
                return

            is_torch2, reshape_qkv, transpose_qkv, reshape_q, matmul_add_q, matmul_add_k, matmul_add_v = match_qkv

            attention_last_node = reshape_qkv

            q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q, normalize_node, is_torch2)
            if q_num_heads <= 0:
                logger.debug("fuse_attention: failed to detect num_heads")
                return

            # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
            new_node = self.create_attention_node_lora(
                matmul_add_q,
                matmul_add_k,
                matmul_add_v,
                q_num_heads,
                q_hidden_size,
                input=normalize_node.output[0],
                output=attention_last_node.output[0],
            )
            if new_node is None:
                return

            q_num_heads, q_hidden_size = self.get_num_heads_and_hidden_size(reshape_q, normalize_node, is_torch2)
            if q_num_heads <= 0:
                logger.debug("fuse_attention: failed to detect num_heads")
                return

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([attention_last_node, transpose_qkv])

        # Use prune graph to remove nodes since they are shared by all attention nodes.
        self.prune_graph = True

    def match_qkv_torch1(self, root_input, skip_add):
        """Match Q, K and V paths exported by PyTorch 1.*"""
        another_input = 1 if skip_add.input[0] == root_input else 0
        qkv_nodes = self.model.match_parent_path(
            skip_add,
            ["Add", "MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            [another_input, None, None, 0, 0, 0],
        )

        if qkv_nodes is None:
            return None

        (_, _, reshape_qkv, transpose_qkv, _, matmul_qkv) = qkv_nodes

        # No bias. For cross-attention, the input of the MatMul is encoder_hidden_states graph input.
        v_nodes = self.model.match_parent_path(matmul_qkv, ["Reshape", "Transpose", "Reshape", "MatMul"], [1, 0, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return None
        (_, _, _, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Mul", "MatMul"], [0, 0, 0])
        if qk_nodes is not None:
            (_softmax_qk, _mul_qk, matmul_qk) = qk_nodes
        else:
            qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Add", "Mul", "MatMul"], [0, 0, 0, 0])
            if qk_nodes is not None:
                (_softmax_qk, _add_zero, _mul_qk, matmul_qk) = qk_nodes
            else:
                logger.debug("fuse_attention: failed to match qk path")
                return None

        q_nodes = self.model.match_parent_path(matmul_qk, ["Reshape", "Transpose", "Reshape", "MatMul"], [0, 0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return None
        (_, _transpose_q, reshape_q, matmul_q) = q_nodes

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Transpose", "Reshape", "Transpose", "Reshape", "MatMul"], [1, 0, 0, 0, 0]
        )
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return None

        (_, _, _, _, matmul_k) = k_nodes

        return False, reshape_qkv, transpose_qkv, reshape_q, matmul_q, matmul_k, matmul_v

    def match_qkv_torch2(self, root_input, skip_add):
        """Match Q, K and V paths exported by PyTorch 2.*"""
        another_input = 1 if skip_add.input[0] == root_input else 0
        qkv_nodes = self.model.match_parent_path(
            skip_add,
            ["Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [another_input, None, None, 0, 0],
        )

        if qkv_nodes is None:
            return None

        (_, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "MatMul"], [1, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return None
        (_, _, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "MatMul"], [0, 0])
        if qk_nodes is not None:
            (_softmax_qk, matmul_qk) = qk_nodes
        else:
            logger.debug("fuse_attention: failed to match qk path")
            return None

        q_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Transpose", "Reshape", "MatMul"], [0, None, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return None
        (mul_q, _transpose_q, reshape_q, matmul_q) = q_nodes

        k_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Transpose", "Reshape", "MatMul"], [1, None, 0, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return None

        (_mul_k, _, _, matmul_k) = k_nodes

        # The scalar for Q and K is sqrt(1.0/sqrt(head_size)).
        mul_q_nodes = self.model.match_parent_path(
            mul_q,
            ["Sqrt", "Div", "Sqrt", "Cast", "Slice", "Shape", "Transpose", "Reshape"],
            [None, 0, 1, 0, 0, 0, 0, 0],
        )
        if mul_q_nodes is None or mul_q_nodes[-1] != reshape_q:
            logger.debug("fuse_attention: failed to match mul_q path")
            return None

        return True, reshape_qkv, transpose_qkv, reshape_q, matmul_q, matmul_k, matmul_v

    def match_qkv_torch1_lora(self, root_input, skip_add):
        """Match Q, K and V paths exported by PyTorch 1 that contains LoRA patterns.*"""
        another_input = 1 if skip_add.input[0] == root_input else 0
        qkv_nodes = self.model.match_parent_path(
            skip_add,
            ["Add", "Add", "MatMul", "Reshape", "Transpose", "Reshape", "MatMul"],
            [another_input, 0, None, None, 0, 0, 0],
        )
        if qkv_nodes is None:
            return None

        (_, _, _, reshape_qkv, transpose_qkv, _, matmul_qkv) = qkv_nodes

        # No bias. For cross-attention, the input of the MatMul is encoder_hidden_states graph input.
        v_nodes = self.model.match_parent_path(matmul_qkv, ["Reshape", "Transpose", "Reshape", "Add"], [1, 0, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match LoRA v path")
            return None
        (_, _, _, matmul_add_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Mul", "MatMul"], [0, 0, 0])
        if qk_nodes is not None:
            (_softmax_qk, _mul_qk, matmul_qk) = qk_nodes
        else:
            qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Add", "Mul", "MatMul"], [0, 0, 0, 0])
            if qk_nodes is not None:
                (_softmax_qk, _add_zero, _mul_qk, matmul_qk) = qk_nodes
            else:
                logger.debug("fuse_attention: failed to match LoRA qk path")
                return None

        q_nodes = self.model.match_parent_path(matmul_qk, ["Reshape", "Transpose", "Reshape", "Add"], [0, 0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match LoRA q path")
            return None
        (_, _transpose_q, reshape_q, matmul_add_q) = q_nodes

        k_nodes = self.model.match_parent_path(
            matmul_qk, ["Transpose", "Reshape", "Transpose", "Reshape", "Add"], [1, 0, 0, 0, 0]
        )
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match LoRA k path")
            return None

        (_, _, _, _, matmul_add_k) = k_nodes

        return False, reshape_qkv, transpose_qkv, reshape_q, matmul_add_q, matmul_add_k, matmul_add_v

    def match_qkv_torch2_lora(self, root_input, skip_add):
        """Match Q, K and V paths exported by PyTorch 2 that contains LoRA patterns.*"""
        another_input = 1 if skip_add.input[0] == root_input else 0
        qkv_nodes = self.model.match_parent_path(
            skip_add,
            ["Add", "Add", "MatMul", "Reshape", "Transpose", "MatMul"],
            [another_input, 0, None, None, 0, 0],
        )
        if qkv_nodes is None:
            return None

        (_, _, _, reshape_qkv, transpose_qkv, matmul_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Transpose", "Reshape", "Add"], [1, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match LoRA v path")
            return None
        (_, _, matmul_add_v) = v_nodes

        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "MatMul"], [0, 0])
        if qk_nodes is not None:
            (_softmax_qk, matmul_qk) = qk_nodes
        else:
            logger.debug("fuse_attention: failed to match LoRA qk path")
            return None

        q_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Transpose", "Reshape", "Add"], [0, None, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match LoRA q path")
            return None
        (mul_q, _transpose_q, reshape_q, matmul_add_q) = q_nodes

        k_nodes = self.model.match_parent_path(matmul_qk, ["Mul", "Transpose", "Reshape", "Add"], [1, None, 0, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match LoRA k path")
            return None

        (_mul_k, _, _, matmul_add_k) = k_nodes

        # The scalar for Q and K is sqrt(1.0/sqrt(head_size)).
        mul_q_nodes = self.model.match_parent_path(
            mul_q,
            ["Sqrt", "Div", "Sqrt", "Cast", "Slice", "Shape", "Transpose", "Reshape"],
            [None, 0, 1, 0, 0, 0, 0, 0],
        )
        if mul_q_nodes is None or mul_q_nodes[-1] != reshape_q:
            logger.debug("fuse_attention: failed to match LoRA mul_q path")
            return None

        return True, reshape_qkv, transpose_qkv, reshape_q, matmul_add_q, matmul_add_k, matmul_add_v

    def match_lora_path(
        self,
        add_node: NodeProto,
    ):
        # Lora paths can look like one of the following options:
        # MatMul -> MatMul -> Add
        # MatMul -> MatMul -> Mul -> Add
        # MatMul -> MatMul -> Mul -> Mul -> Add

        # Try matching MatMul -> MatMul -> Add
        lora_nodes = self.model.match_parent_path(
            add_node,
            ["MatMul", "MatMul"],
            [1, 0],
        )

        if lora_nodes is not None:
            (lora_matmul_2_node, lora_matmul_1_node) = lora_nodes
            return (lora_matmul_2_node, lora_matmul_1_node)

        # Try matching MatMul -> MatMul -> Mul -> Add
        lora_nodes = self.model.match_parent_path(
            add_node,
            ["Mul", "MatMul", "MatMul"],
            [1, 0, 0],
        )

        if lora_nodes is not None:
            (lora_mul_node, _, lora_matmul_1_node) = lora_nodes
            return (lora_mul_node, lora_matmul_1_node)

        # Try matching MatMul -> MatMul -> Mul -> Mul -> Add
        lora_nodes = self.model.match_parent_path(
            add_node,
            ["Mul", "Mul", "MatMul", "MatMul"],
            [1, 0, 0, 0],
        )

        if lora_nodes is not None:
            (lora_mul_node, _, _, lora_matmul_1_node) = lora_nodes
            return (lora_mul_node, lora_matmul_1_node)

        return None

    def fuse_a1111_fp16(self, normalize_node, input_name_to_nodes, output_name_to_node):
        """Fuse attention of fp16 UNet exported in A1111 (stable diffusion webui) extension"""
        entry_path = self.model.match_parent_path(normalize_node, ["Cast", "Add"], [0, 0])
        if entry_path is None:
            entry_path = self.model.match_parent_path(normalize_node, ["Cast", "Reshape"], [0, 0])
            if entry_path is None:
                return False
        _cast, node_before_layernorm = entry_path

        root_input = node_before_layernorm.output[0]

        children_nodes = input_name_to_nodes[root_input]
        skip_add = None
        for node in children_nodes:
            if node.op_type == "Add":  # SkipLayerNormalization fusion is not applied yet
                skip_add = node
                break
        if skip_add is None:
            return False

        match_qkv = self.match_qkv_a1111(root_input, skip_add)
        if match_qkv is None:
            return False

        (
            reshape_qkv,
            transpose_qkv,
            reshape_q,
            matmul_q,
            matmul_k,
            matmul_v,
        ) = match_qkv

        cast_q = self.model.match_parent(matmul_q, "Cast", 0)
        cast_k = self.model.match_parent(matmul_k, "Cast", 0)
        cast_v = self.model.match_parent(matmul_v, "Cast", 0)
        if not (
            cast_q is not None
            and cast_k is not None
            and (cast_q == cast_k if not self.is_cross_attention else cast_q != cast_k)
            and cast_k == cast_v
        ):
            return False

        if cast_q.input[0] != normalize_node.output[0]:
            return False

        attention_last_node = reshape_qkv

        q_num_heads = self.get_num_heads(reshape_q, True) or self.get_num_heads(reshape_q, False)
        if q_num_heads <= 0:
            logger.debug("fuse_attention: failed to detect num_heads")
            return False

        q_hidden_size = self.get_hidden_size(normalize_node)

        # number of heads are same for all the paths, hence to create attention node, we pass the q_num_heads
        new_node = self.create_attention_node(
            matmul_q,
            matmul_k,
            matmul_v,
            q_num_heads,
            q_hidden_size,
            input=matmul_q.input[0],
            output=attention_last_node.output[0],
        )
        if new_node is None:
            return False

        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

        self.nodes_to_remove.extend([attention_last_node, transpose_qkv])

        # Use prune graph to remove nodes since they are shared by all attention nodes.
        self.prune_graph = True
        return True

    def match_qkv_a1111(self, root_input, skip_add):
        """Match Q, K and V paths exported by A1111 (stable diffusion webui) extension"""
        another_input = 1 if skip_add.input[0] == root_input else 0
        qkv_nodes = self.model.match_parent_path(
            skip_add,
            ["Add", "MatMul", "Reshape", "Transpose", "Reshape", "Einsum"],
            [another_input, None, None, 0, 0, 0],
        )

        if qkv_nodes is None:
            return None

        (_, _, reshape_qkv, transpose_qkv, reshape_einsum, einsum_qkv) = qkv_nodes

        v_nodes = self.model.match_parent_path(einsum_qkv, ["Reshape", "Transpose", "Reshape", "MatMul"], [1, 0, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return None
        (_, _, _, matmul_v) = v_nodes

        qk_nodes = self.model.match_parent_path(
            einsum_qkv, ["Cast", "Cast", "Softmax", "Mul", "Einsum"], [0, 0, 0, 0, None]
        )
        if qk_nodes is not None:
            (_, _, _softmax_qk, _, einsum_qk) = qk_nodes
        else:
            logger.debug("fuse_attention: failed to match qk path")
            return None

        q_nodes = self.model.match_parent_path(einsum_qk, ["Reshape", "Transpose", "Reshape", "MatMul"], [0, 0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return None
        (_, _transpose_q, reshape_q, matmul_q) = q_nodes

        k_nodes = self.model.match_parent_path(einsum_qk, ["Reshape", "Transpose", "Reshape", "MatMul"], [1, 0, 0, 0])
        if k_nodes is None:
            logger.debug("fuse_attention: failed to match k path")
            return None

        (_, _, _, matmul_k) = k_nodes

        return reshape_qkv, transpose_qkv, reshape_q, matmul_q, matmul_k, matmul_v

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\dependency.py ===
# orm/dependency.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""Relationship dependencies.

"""

from __future__ import annotations

from . import attributes
from . import exc
from . import sync
from . import unitofwork
from . import util as mapperutil
from .interfaces import MANYTOMANY
from .interfaces import MANYTOONE
from .interfaces import ONETOMANY
from .. import exc as sa_exc
from .. import sql
from .. import util


class DependencyProcessor:
    def __init__(self, prop):
        self.prop = prop
        self.cascade = prop.cascade
        self.mapper = prop.mapper
        self.parent = prop.parent
        self.secondary = prop.secondary
        self.direction = prop.direction
        self.post_update = prop.post_update
        self.passive_deletes = prop.passive_deletes
        self.passive_updates = prop.passive_updates
        self.enable_typechecks = prop.enable_typechecks
        if self.passive_deletes:
            self._passive_delete_flag = attributes.PASSIVE_NO_INITIALIZE
        else:
            self._passive_delete_flag = attributes.PASSIVE_OFF
        if self.passive_updates:
            self._passive_update_flag = attributes.PASSIVE_NO_INITIALIZE
        else:
            self._passive_update_flag = attributes.PASSIVE_OFF

        self.sort_key = "%s_%s" % (self.parent._sort_key, prop.key)
        self.key = prop.key
        if not self.prop.synchronize_pairs:
            raise sa_exc.ArgumentError(
                "Can't build a DependencyProcessor for relationship %s. "
                "No target attributes to populate between parent and "
                "child are present" % self.prop
            )

    @classmethod
    def from_relationship(cls, prop):
        return _direction_to_processor[prop.direction](prop)

    def hasparent(self, state):
        """return True if the given object instance has a parent,
        according to the ``InstrumentedAttribute`` handled by this
        ``DependencyProcessor``.

        """
        return self.parent.class_manager.get_impl(self.key).hasparent(state)

    def per_property_preprocessors(self, uow):
        """establish actions and dependencies related to a flush.

        These actions will operate on all relevant states in
        the aggregate.

        """
        uow.register_preprocessor(self, True)

    def per_property_flush_actions(self, uow):
        after_save = unitofwork.ProcessAll(uow, self, False, True)
        before_delete = unitofwork.ProcessAll(uow, self, True, True)

        parent_saves = unitofwork.SaveUpdateAll(
            uow, self.parent.primary_base_mapper
        )
        child_saves = unitofwork.SaveUpdateAll(
            uow, self.mapper.primary_base_mapper
        )

        parent_deletes = unitofwork.DeleteAll(
            uow, self.parent.primary_base_mapper
        )
        child_deletes = unitofwork.DeleteAll(
            uow, self.mapper.primary_base_mapper
        )

        self.per_property_dependencies(
            uow,
            parent_saves,
            child_saves,
            parent_deletes,
            child_deletes,
            after_save,
            before_delete,
        )

    def per_state_flush_actions(self, uow, states, isdelete):
        """establish actions and dependencies related to a flush.

        These actions will operate on all relevant states
        individually.    This occurs only if there are cycles
        in the 'aggregated' version of events.

        """

        child_base_mapper = self.mapper.primary_base_mapper
        child_saves = unitofwork.SaveUpdateAll(uow, child_base_mapper)
        child_deletes = unitofwork.DeleteAll(uow, child_base_mapper)

        # locate and disable the aggregate processors
        # for this dependency

        if isdelete:
            before_delete = unitofwork.ProcessAll(uow, self, True, True)
            before_delete.disabled = True
        else:
            after_save = unitofwork.ProcessAll(uow, self, False, True)
            after_save.disabled = True

        # check if the "child" side is part of the cycle

        if child_saves not in uow.cycles:
            # based on the current dependencies we use, the saves/
            # deletes should always be in the 'cycles' collection
            # together.   if this changes, we will have to break up
            # this method a bit more.
            assert child_deletes not in uow.cycles

            # child side is not part of the cycle, so we will link per-state
            # actions to the aggregate "saves", "deletes" actions
            child_actions = [(child_saves, False), (child_deletes, True)]
            child_in_cycles = False
        else:
            child_in_cycles = True

        # check if the "parent" side is part of the cycle
        if not isdelete:
            parent_saves = unitofwork.SaveUpdateAll(
                uow, self.parent.base_mapper
            )
            parent_deletes = before_delete = None
            if parent_saves in uow.cycles:
                parent_in_cycles = True
        else:
            parent_deletes = unitofwork.DeleteAll(uow, self.parent.base_mapper)
            parent_saves = after_save = None
            if parent_deletes in uow.cycles:
                parent_in_cycles = True

        # now create actions /dependencies for each state.

        for state in states:
            # detect if there's anything changed or loaded
            # by a preprocessor on this state/attribute.   In the
            # case of deletes we may try to load missing items here as well.
            sum_ = state.manager[self.key].impl.get_all_pending(
                state,
                state.dict,
                (
                    self._passive_delete_flag
                    if isdelete
                    else attributes.PASSIVE_NO_INITIALIZE
                ),
            )

            if not sum_:
                continue

            if isdelete:
                before_delete = unitofwork.ProcessState(uow, self, True, state)
                if parent_in_cycles:
                    parent_deletes = unitofwork.DeleteState(uow, state)
            else:
                after_save = unitofwork.ProcessState(uow, self, False, state)
                if parent_in_cycles:
                    parent_saves = unitofwork.SaveUpdateState(uow, state)

            if child_in_cycles:
                child_actions = []
                for child_state, child in sum_:
                    if child_state not in uow.states:
                        child_action = (None, None)
                    else:
                        (deleted, listonly) = uow.states[child_state]
                        if deleted:
                            child_action = (
                                unitofwork.DeleteState(uow, child_state),
                                True,
                            )
                        else:
                            child_action = (
                                unitofwork.SaveUpdateState(uow, child_state),
                                False,
                            )
                    child_actions.append(child_action)

            # establish dependencies between our possibly per-state
            # parent action and our possibly per-state child action.
            for child_action, childisdelete in child_actions:
                self.per_state_dependencies(
                    uow,
                    parent_saves,
                    parent_deletes,
                    child_action,
                    after_save,
                    before_delete,
                    isdelete,
                    childisdelete,
                )

    def presort_deletes(self, uowcommit, states):
        return False

    def presort_saves(self, uowcommit, states):
        return False

    def process_deletes(self, uowcommit, states):
        pass

    def process_saves(self, uowcommit, states):
        pass

    def prop_has_changes(self, uowcommit, states, isdelete):
        if not isdelete or self.passive_deletes:
            passive = (
                attributes.PASSIVE_NO_INITIALIZE
                | attributes.INCLUDE_PENDING_MUTATIONS
            )
        elif self.direction is MANYTOONE:
            # here, we were hoping to optimize having to fetch many-to-one
            # for history and ignore it, if there's no further cascades
            # to take place.  however there are too many less common conditions
            # that still take place and tests in test_relationships /
            # test_cascade etc. will still fail.
            passive = attributes.PASSIVE_NO_FETCH_RELATED
        else:
            passive = (
                attributes.PASSIVE_OFF | attributes.INCLUDE_PENDING_MUTATIONS
            )

        for s in states:
            # TODO: add a high speed method
            # to InstanceState which returns:  attribute
            # has a non-None value, or had one
            history = uowcommit.get_attribute_history(s, self.key, passive)
            if history and not history.empty():
                return True
        else:
            return (
                states
                and not self.prop._is_self_referential
                and self.mapper in uowcommit.mappers
            )

    def _verify_canload(self, state):
        if self.prop.uselist and state is None:
            raise exc.FlushError(
                "Can't flush None value found in "
                "collection %s" % (self.prop,)
            )
        elif state is not None and not self.mapper._canload(
            state, allow_subtypes=not self.enable_typechecks
        ):
            if self.mapper._canload(state, allow_subtypes=True):
                raise exc.FlushError(
                    "Attempting to flush an item of type "
                    "%(x)s as a member of collection "
                    '"%(y)s". Expected an object of type '
                    "%(z)s or a polymorphic subclass of "
                    "this type. If %(x)s is a subclass of "
                    '%(z)s, configure mapper "%(zm)s" to '
                    "load this subtype polymorphically, or "
                    "set enable_typechecks=False to allow "
                    "any subtype to be accepted for flush. "
                    % {
                        "x": state.class_,
                        "y": self.prop,
                        "z": self.mapper.class_,
                        "zm": self.mapper,
                    }
                )
            else:
                raise exc.FlushError(
                    "Attempting to flush an item of type "
                    "%(x)s as a member of collection "
                    '"%(y)s". Expected an object of type '
                    "%(z)s or a polymorphic subclass of "
                    "this type."
                    % {
                        "x": state.class_,
                        "y": self.prop,
                        "z": self.mapper.class_,
                    }
                )

    def _synchronize(self, state, child, associationrow, clearkeys, uowcommit):
        raise NotImplementedError()

    def _get_reversed_processed_set(self, uow):
        if not self.prop._reverse_property:
            return None

        process_key = tuple(
            sorted([self.key] + [p.key for p in self.prop._reverse_property])
        )
        return uow.memo(("reverse_key", process_key), set)

    def _post_update(self, state, uowcommit, related, is_m2o_delete=False):
        for x in related:
            if not is_m2o_delete or x is not None:
                uowcommit.register_post_update(
                    state, [r for l, r in self.prop.synchronize_pairs]
                )
                break

    def _pks_changed(self, uowcommit, state):
        raise NotImplementedError()

    def __repr__(self):
        return "%s(%s)" % (self.__class__.__name__, self.prop)


class OneToManyDP(DependencyProcessor):
    def per_property_dependencies(
        self,
        uow,
        parent_saves,
        child_saves,
        parent_deletes,
        child_deletes,
        after_save,
        before_delete,
    ):
        if self.post_update:
            child_post_updates = unitofwork.PostUpdateAll(
                uow, self.mapper.primary_base_mapper, False
            )
            child_pre_updates = unitofwork.PostUpdateAll(
                uow, self.mapper.primary_base_mapper, True
            )

            uow.dependencies.update(
                [
                    (child_saves, after_save),
                    (parent_saves, after_save),
                    (after_save, child_post_updates),
                    (before_delete, child_pre_updates),
                    (child_pre_updates, parent_deletes),
                    (child_pre_updates, child_deletes),
                ]
            )
        else:
            uow.dependencies.update(
                [
                    (parent_saves, after_save),
                    (after_save, child_saves),
                    (after_save, child_deletes),
                    (child_saves, parent_deletes),
                    (child_deletes, parent_deletes),
                    (before_delete, child_saves),
                    (before_delete, child_deletes),
                ]
            )

    def per_state_dependencies(
        self,
        uow,
        save_parent,
        delete_parent,
        child_action,
        after_save,
        before_delete,
        isdelete,
        childisdelete,
    ):
        if self.post_update:
            child_post_updates = unitofwork.PostUpdateAll(
                uow, self.mapper.primary_base_mapper, False
            )
            child_pre_updates = unitofwork.PostUpdateAll(
                uow, self.mapper.primary_base_mapper, True
            )

            # TODO: this whole block is not covered
            # by any tests
            if not isdelete:
                if childisdelete:
                    uow.dependencies.update(
                        [
                            (child_action, after_save),
                            (after_save, child_post_updates),
                        ]
                    )
                else:
                    uow.dependencies.update(
                        [
                            (save_parent, after_save),
                            (child_action, after_save),
                            (after_save, child_post_updates),
                        ]
                    )
            else:
                if childisdelete:
                    uow.dependencies.update(
                        [
                            (before_delete, child_pre_updates),
                            (child_pre_updates, delete_parent),
                        ]
                    )
                else:
                    uow.dependencies.update(
                        [
                            (before_delete, child_pre_updates),
                            (child_pre_updates, delete_parent),
                        ]
                    )
        elif not isdelete:
            uow.dependencies.update(
                [
                    (save_parent, after_save),
                    (after_save, child_action),
                    (save_parent, child_action),
                ]
            )
        else:
            uow.dependencies.update(
                [(before_delete, child_action), (child_action, delete_parent)]
            )

    def presort_deletes(self, uowcommit, states):
        # head object is being deleted, and we manage its list of
        # child objects the child objects have to have their
        # foreign key to the parent set to NULL
        should_null_fks = (
            not self.cascade.delete and not self.passive_deletes == "all"
        )

        for state in states:
            history = uowcommit.get_attribute_history(
                state, self.key, self._passive_delete_flag
            )
            if history:
                for child in history.deleted:
                    if child is not None and self.hasparent(child) is False:
                        if self.cascade.delete_orphan:
                            uowcommit.register_object(child, isdelete=True)
                        else:
                            uowcommit.register_object(child)

                if should_null_fks:
                    for child in history.unchanged:
                        if child is not None:
                            uowcommit.register_object(
                                child, operation="delete", prop=self.prop
                            )

    def presort_saves(self, uowcommit, states):
        children_added = uowcommit.memo(("children_added", self), set)

        should_null_fks = (
            not self.cascade.delete_orphan
            and not self.passive_deletes == "all"
        )

        for state in states:
            pks_changed = self._pks_changed(uowcommit, state)

            if not pks_changed or self.passive_updates:
                passive = (
                    attributes.PASSIVE_NO_INITIALIZE
                    | attributes.INCLUDE_PENDING_MUTATIONS
                )
            else:
                passive = (
                    attributes.PASSIVE_OFF
                    | attributes.INCLUDE_PENDING_MUTATIONS
                )

            history = uowcommit.get_attribute_history(state, self.key, passive)
            if history:
                for child in history.added:
                    if child is not None:
                        uowcommit.register_object(
                            child,
                            cancel_delete=True,
                            operation="add",
                            prop=self.prop,
                        )

                children_added.update(history.added)

                for child in history.deleted:
                    if not self.cascade.delete_orphan:
                        if should_null_fks:
                            uowcommit.register_object(
                                child,
                                isdelete=False,
                                operation="delete",
                                prop=self.prop,
                            )
                    elif self.hasparent(child) is False:
                        uowcommit.register_object(
                            child,
                            isdelete=True,
                            operation="delete",
                            prop=self.prop,
                        )
                        for c, m, st_, dct_ in self.mapper.cascade_iterator(
                            "delete", child
                        ):
                            uowcommit.register_object(st_, isdelete=True)

            if pks_changed:
                if history:
                    for child in history.unchanged:
                        if child is not None:
                            uowcommit.register_object(
                                child,
                                False,
                                self.passive_updates,
                                operation="pk change",
                                prop=self.prop,
                            )

    def process_deletes(self, uowcommit, states):
        # head object is being deleted, and we manage its list of
        # child objects the child objects have to have their foreign
        # key to the parent set to NULL this phase can be called
        # safely for any cascade but is unnecessary if delete cascade
        # is on.

        if self.post_update or not self.passive_deletes == "all":
            children_added = uowcommit.memo(("children_added", self), set)

            for state in states:
                history = uowcommit.get_attribute_history(
                    state, self.key, self._passive_delete_flag
                )
                if history:
                    for child in history.deleted:
                        if (
                            child is not None
                            and self.hasparent(child) is False
                        ):
                            self._synchronize(
                                state, child, None, True, uowcommit, False
                            )
                            if self.post_update and child:
                                self._post_update(child, uowcommit, [state])

                    if self.post_update or not self.cascade.delete:
                        for child in set(history.unchanged).difference(
                            children_added
                        ):
                            if child is not None:
                                self._synchronize(
                                    state, child, None, True, uowcommit, False
                                )
                                if self.post_update and child:
                                    self._post_update(
                                        child, uowcommit, [state]
                                    )

                    # technically, we can even remove each child from the
                    # collection here too.  but this would be a somewhat
                    # inconsistent behavior since it wouldn't happen
                    # if the old parent wasn't deleted but child was moved.

    def process_saves(self, uowcommit, states):
        should_null_fks = (
            not self.cascade.delete_orphan
            and not self.passive_deletes == "all"
        )

        for state in states:
            history = uowcommit.get_attribute_history(
                state, self.key, attributes.PASSIVE_NO_INITIALIZE
            )
            if history:
                for child in history.added:
                    self._synchronize(
                        state, child, None, False, uowcommit, False
                    )
                    if child is not None and self.post_update:
                        self._post_update(child, uowcommit, [state])

                for child in history.deleted:
                    if (
                        should_null_fks
                        and not self.cascade.delete_orphan
                        and not self.hasparent(child)
                    ):
                        self._synchronize(
                            state, child, None, True, uowcommit, False
                        )

                if self._pks_changed(uowcommit, state):
                    for child in history.unchanged:
                        self._synchronize(
                            state, child, None, False, uowcommit, True
                        )

    def _synchronize(
        self, state, child, associationrow, clearkeys, uowcommit, pks_changed
    ):
        source = state
        dest = child
        self._verify_canload(child)
        if dest is None or (
            not self.post_update and uowcommit.is_deleted(dest)
        ):
            return
        if clearkeys:
            sync.clear(dest, self.mapper, self.prop.synchronize_pairs)
        else:
            sync.populate(
                source,
                self.parent,
                dest,
                self.mapper,
                self.prop.synchronize_pairs,
                uowcommit,
                self.passive_updates and pks_changed,
            )

    def _pks_changed(self, uowcommit, state):
        return sync.source_modified(
            uowcommit, state, self.parent, self.prop.synchronize_pairs
        )


class ManyToOneDP(DependencyProcessor):
    def __init__(self, prop):
        DependencyProcessor.__init__(self, prop)
        for mapper in self.mapper.self_and_descendants:
            mapper._dependency_processors.append(DetectKeySwitch(prop))

    def per_property_dependencies(
        self,
        uow,
        parent_saves,
        child_saves,
        parent_deletes,
        child_deletes,
        after_save,
        before_delete,
    ):
        if self.post_update:
            parent_post_updates = unitofwork.PostUpdateAll(
                uow, self.parent.primary_base_mapper, False
            )
            parent_pre_updates = unitofwork.PostUpdateAll(
                uow, self.parent.primary_base_mapper, True
            )

            uow.dependencies.update(
                [
                    (child_saves, after_save),
                    (parent_saves, after_save),
                    (after_save, parent_post_updates),
                    (after_save, parent_pre_updates),
                    (before_delete, parent_pre_updates),
                    (parent_pre_updates, child_deletes),
                    (parent_pre_updates, parent_deletes),
                ]
            )
        else:
            uow.dependencies.update(
                [
                    (child_saves, after_save),
                    (after_save, parent_saves),
                    (parent_saves, child_deletes),
                    (parent_deletes, child_deletes),
                ]
            )

    def per_state_dependencies(
        self,
        uow,
        save_parent,
        delete_parent,
        child_action,
        after_save,
        before_delete,
        isdelete,
        childisdelete,
    ):
        if self.post_update:
            if not isdelete:
                parent_post_updates = unitofwork.PostUpdateAll(
                    uow, self.parent.primary_base_mapper, False
                )
                if childisdelete:
                    uow.dependencies.update(
                        [
                            (after_save, parent_post_updates),
                            (parent_post_updates, child_action),
                        ]
                    )
                else:
                    uow.dependencies.update(
                        [
                            (save_parent, after_save),
                            (child_action, after_save),
                            (after_save, parent_post_updates),
                        ]
                    )
            else:
                parent_pre_updates = unitofwork.PostUpdateAll(
                    uow, self.parent.primary_base_mapper, True
                )

                uow.dependencies.update(
                    [
                        (before_delete, parent_pre_updates),
                        (parent_pre_updates, delete_parent),
                        (parent_pre_updates, child_action),
                    ]
                )

        elif not isdelete:
            if not childisdelete:
                uow.dependencies.update(
                    [(child_action, after_save), (after_save, save_parent)]
                )
            else:
                uow.dependencies.update([(after_save, save_parent)])

        else:
            if childisdelete:
                uow.dependencies.update([(delete_parent, child_action)])

    def presort_deletes(self, uowcommit, states):
        if self.cascade.delete or self.cascade.delete_orphan:
            for state in states:
                history = uowcommit.get_attribute_history(
                    state, self.key, self._passive_delete_flag
                )
                if history:
                    if self.cascade.delete_orphan:
                        todelete = history.sum()
                    else:
                        todelete = history.non_deleted()
                    for child in todelete:
                        if child is None:
                            continue
                        uowcommit.register_object(
                            child,
                            isdelete=True,
                            operation="delete",
                            prop=self.prop,
                        )
                        t = self.mapper.cascade_iterator("delete", child)
                        for c, m, st_, dct_ in t:
                            uowcommit.register_object(st_, isdelete=True)

    def presort_saves(self, uowcommit, states):
        for state in states:
            uowcommit.register_object(state, operation="add", prop=self.prop)
            if self.cascade.delete_orphan:
                history = uowcommit.get_attribute_history(
                    state, self.key, self._passive_delete_flag
                )
                if history:
                    for child in history.deleted:
                        if self.hasparent(child) is False:
                            uowcommit.register_object(
                                child,
                                isdelete=True,
                                operation="delete",
                                prop=self.prop,
                            )

                            t = self.mapper.cascade_iterator("delete", child)
                            for c, m, st_, dct_ in t:
                                uowcommit.register_object(st_, isdelete=True)

    def process_deletes(self, uowcommit, states):
        if (
            self.post_update
            and not self.cascade.delete_orphan
            and not self.passive_deletes == "all"
        ):
            # post_update means we have to update our
            # row to not reference the child object
            # before we can DELETE the row
            for state in states:
                self._synchronize(state, None, None, True, uowcommit)
                if state and self.post_update:
                    history = uowcommit.get_attribute_history(
                        state, self.key, self._passive_delete_flag
                    )
                    if history:
                        self._post_update(
                            state, uowcommit, history.sum(), is_m2o_delete=True
                        )

    def process_saves(self, uowcommit, states):
        for state in states:
            history = uowcommit.get_attribute_history(
                state, self.key, attributes.PASSIVE_NO_INITIALIZE
            )
            if history:
                if history.added:
                    for child in history.added:
                        self._synchronize(
                            state, child, None, False, uowcommit, "add"
                        )
                elif history.deleted:
                    self._synchronize(
                        state, None, None, True, uowcommit, "delete"
                    )
                if self.post_update:
                    self._post_update(state, uowcommit, history.sum())

    def _synchronize(
        self,
        state,
        child,
        associationrow,
        clearkeys,
        uowcommit,
        operation=None,
    ):
        if state is None or (
            not self.post_update and uowcommit.is_deleted(state)
        ):
            return

        if (
            operation is not None
            and child is not None
            and not uowcommit.session._contains_state(child)
        ):
            util.warn(
                "Object of type %s not in session, %s "
                "operation along '%s' won't proceed"
                % (mapperutil.state_class_str(child), operation, self.prop)
            )
            return

        if clearkeys or child is None:
            sync.clear(state, self.parent, self.prop.synchronize_pairs)
        else:
            self._verify_canload(child)
            sync.populate(
                child,
                self.mapper,
                state,
                self.parent,
                self.prop.synchronize_pairs,
                uowcommit,
                False,
            )


class DetectKeySwitch(DependencyProcessor):
    """For many-to-one relationships with no one-to-many backref,
    searches for parents through the unit of work when a primary
    key has changed and updates them.

    Theoretically, this approach could be expanded to support transparent
    deletion of objects referenced via many-to-one as well, although
    the current attribute system doesn't do enough bookkeeping for this
    to be efficient.

    """

    def per_property_preprocessors(self, uow):
        if self.prop._reverse_property:
            if self.passive_updates:
                return
            else:
                if False in (
                    prop.passive_updates
                    for prop in self.prop._reverse_property
                ):
                    return

        uow.register_preprocessor(self, False)

    def per_property_flush_actions(self, uow):
        parent_saves = unitofwork.SaveUpdateAll(uow, self.parent.base_mapper)
        after_save = unitofwork.ProcessAll(uow, self, False, False)
        uow.dependencies.update([(parent_saves, after_save)])

    def per_state_flush_actions(self, uow, states, isdelete):
        pass

    def presort_deletes(self, uowcommit, states):
        pass

    def presort_saves(self, uow, states):
        if not self.passive_updates:
            # for non-passive updates, register in the preprocess stage
            # so that mapper save_obj() gets a hold of changes
            self._process_key_switches(states, uow)

    def prop_has_changes(self, uow, states, isdelete):
        if not isdelete and self.passive_updates:
            d = self._key_switchers(uow, states)
            return bool(d)

        return False

    def process_deletes(self, uowcommit, states):
        assert False

    def process_saves(self, uowcommit, states):
        # for passive updates, register objects in the process stage
        # so that we avoid ManyToOneDP's registering the object without
        # the listonly flag in its own preprocess stage (results in UPDATE)
        # statements being emitted
        assert self.passive_updates
        self._process_key_switches(states, uowcommit)

    def _key_switchers(self, uow, states):
        switched, notswitched = uow.memo(
            ("pk_switchers", self), lambda: (set(), set())
        )

        allstates = switched.union(notswitched)
        for s in states:
            if s not in allstates:
                if self._pks_changed(uow, s):
                    switched.add(s)
                else:
                    notswitched.add(s)
        return switched

    def _process_key_switches(self, deplist, uowcommit):
        switchers = self._key_switchers(uowcommit, deplist)
        if switchers:
            # if primary key values have actually changed somewhere, perform
            # a linear search through the UOW in search of a parent.
            for state in uowcommit.session.identity_map.all_states():
                if not issubclass(state.class_, self.parent.class_):
                    continue
                dict_ = state.dict
                related = state.get_impl(self.key).get(
                    state, dict_, passive=self._passive_update_flag
                )
                if (
                    related is not attributes.PASSIVE_NO_RESULT
                    and related is not None
                ):
                    if self.prop.uselist:
                        if not related:
                            continue
                        related_obj = related[0]
                    else:
                        related_obj = related
                    related_state = attributes.instance_state(related_obj)
                    if related_state in switchers:
                        uowcommit.register_object(
                            state, False, self.passive_updates
                        )
                        sync.populate(
                            related_state,
                            self.mapper,
                            state,
                            self.parent,
                            self.prop.synchronize_pairs,
                            uowcommit,
                            self.passive_updates,
                        )

    def _pks_changed(self, uowcommit, state):
        return bool(state.key) and sync.source_modified(
            uowcommit, state, self.mapper, self.prop.synchronize_pairs
        )


class ManyToManyDP(DependencyProcessor):
    def per_property_dependencies(
        self,
        uow,
        parent_saves,
        child_saves,
        parent_deletes,
        child_deletes,
        after_save,
        before_delete,
    ):
        uow.dependencies.update(
            [
                (parent_saves, after_save),
                (child_saves, after_save),
                (after_save, child_deletes),
                # a rowswitch on the parent from  deleted to saved
                # can make this one occur, as the "save" may remove
                # an element from the
                # "deleted" list before we have a chance to
                # process its child rows
                (before_delete, parent_saves),
                (before_delete, parent_deletes),
                (before_delete, child_deletes),
                (before_delete, child_saves),
            ]
        )

    def per_state_dependencies(
        self,
        uow,
        save_parent,
        delete_parent,
        child_action,
        after_save,
        before_delete,
        isdelete,
        childisdelete,
    ):
        if not isdelete:
            if childisdelete:
                uow.dependencies.update(
                    [(save_parent, after_save), (after_save, child_action)]
                )
            else:
                uow.dependencies.update(
                    [(save_parent, after_save), (child_action, after_save)]
                )
        else:
            uow.dependencies.update(
                [(before_delete, child_action), (before_delete, delete_parent)]
            )

    def presort_deletes(self, uowcommit, states):
        # TODO: no tests fail if this whole
        # thing is removed !!!!
        if not self.passive_deletes:
            # if no passive deletes, load history on
            # the collection, so that prop_has_changes()
            # returns True
            for state in states:
                uowcommit.get_attribute_history(
                    state, self.key, self._passive_delete_flag
                )

    def presort_saves(self, uowcommit, states):
        if not self.passive_updates:
            # if no passive updates, load history on
            # each collection where parent has changed PK,
            # so that prop_has_changes() returns True
            for state in states:
                if self._pks_changed(uowcommit, state):
                    uowcommit.get_attribute_history(
                        state, self.key, attributes.PASSIVE_OFF
                    )

        if not self.cascade.delete_orphan:
            return

        # check for child items removed from the collection
        # if delete_orphan check is turned on.
        for state in states:
            history = uowcommit.get_attribute_history(
                state, self.key, attributes.PASSIVE_NO_INITIALIZE
            )
            if history:
                for child in history.deleted:
                    if self.hasparent(child) is False:
                        uowcommit.register_object(
                            child,
                            isdelete=True,
                            operation="delete",
                            prop=self.prop,
                        )
                        for c, m, st_, dct_ in self.mapper.cascade_iterator(
                            "delete", child
                        ):
                            uowcommit.register_object(st_, isdelete=True)

    def process_deletes(self, uowcommit, states):
        secondary_delete = []
        secondary_insert = []
        secondary_update = []

        processed = self._get_reversed_processed_set(uowcommit)
        tmp = set()
        for state in states:
            # this history should be cached already, as
            # we loaded it in preprocess_deletes
            history = uowcommit.get_attribute_history(
                state, self.key, self._passive_delete_flag
            )
            if history:
                for child in history.non_added():
                    if child is None or (
                        processed is not None and (state, child) in processed
                    ):
                        continue
                    associationrow = {}
                    if not self._synchronize(
                        state,
                        child,
                        associationrow,
                        False,
                        uowcommit,
                        "delete",
                    ):
                        continue
                    secondary_delete.append(associationrow)

                tmp.update((c, state) for c in history.non_added())

        if processed is not None:
            processed.update(tmp)

        self._run_crud(
            uowcommit, secondary_insert, secondary_update, secondary_delete
        )

    def process_saves(self, uowcommit, states):
        secondary_delete = []
        secondary_insert = []
        secondary_update = []

        processed = self._get_reversed_processed_set(uowcommit)
        tmp = set()

        for state in states:
            need_cascade_pks = not self.passive_updates and self._pks_changed(
                uowcommit, state
            )
            if need_cascade_pks:
                passive = (
                    attributes.PASSIVE_OFF
                    | attributes.INCLUDE_PENDING_MUTATIONS
                )
            else:
                passive = (
                    attributes.PASSIVE_NO_INITIALIZE
                    | attributes.INCLUDE_PENDING_MUTATIONS
                )
            history = uowcommit.get_attribute_history(state, self.key, passive)
            if history:
                for child in history.added:
                    if processed is not None and (state, child) in processed:
                        continue
                    associationrow = {}
                    if not self._synchronize(
                        state, child, associationrow, False, uowcommit, "add"
                    ):
                        continue
                    secondary_insert.append(associationrow)
                for child in history.deleted:
                    if processed is not None and (state, child) in processed:
                        continue
                    associationrow = {}
                    if not self._synchronize(
                        state,
                        child,
                        associationrow,
                        False,
                        uowcommit,
                        "delete",
                    ):
                        continue
                    secondary_delete.append(associationrow)

                tmp.update((c, state) for c in history.added + history.deleted)

                if need_cascade_pks:
                    for child in history.unchanged:
                        associationrow = {}
                        sync.update(
                            state,
                            self.parent,
                            associationrow,
                            "old_",
                            self.prop.synchronize_pairs,
                        )
                        sync.update(
                            child,
                            self.mapper,
                            associationrow,
                            "old_",
                            self.prop.secondary_synchronize_pairs,
                        )

                        secondary_update.append(associationrow)

        if processed is not None:
            processed.update(tmp)

        self._run_crud(
            uowcommit, secondary_insert, secondary_update, secondary_delete
        )

    def _run_crud(
        self, uowcommit, secondary_insert, secondary_update, secondary_delete
    ):
        connection = uowcommit.transaction.connection(self.mapper)

        if secondary_delete:
            associationrow = secondary_delete[0]
            statement = self.secondary.delete().where(
                sql.and_(
                    *[
                        c == sql.bindparam(c.key, type_=c.type)
                        for c in self.secondary.c
                        if c.key in associationrow
                    ]
                )
            )
            result = connection.execute(statement, secondary_delete)

            if (
                result.supports_sane_multi_rowcount()
            ) and result.rowcount != len(secondary_delete):
                raise exc.StaleDataError(
                    "DELETE statement on table '%s' expected to delete "
                    "%d row(s); Only %d were matched."
                    % (
                        self.secondary.description,
                        len(secondary_delete),
                        result.rowcount,
                    )
                )

        if secondary_update:
            associationrow = secondary_update[0]
            statement = self.secondary.update().where(
                sql.and_(
                    *[
                        c == sql.bindparam("old_" + c.key, type_=c.type)
                        for c in self.secondary.c
                        if c.key in associationrow
                    ]
                )
            )
            result = connection.execute(statement, secondary_update)

            if (
                result.supports_sane_multi_rowcount()
            ) and result.rowcount != len(secondary_update):
                raise exc.StaleDataError(
                    "UPDATE statement on table '%s' expected to update "
                    "%d row(s); Only %d were matched."
                    % (
                        self.secondary.description,
                        len(secondary_update),
                        result.rowcount,
                    )
                )

        if secondary_insert:
            statement = self.secondary.insert()
            connection.execute(statement, secondary_insert)

    def _synchronize(
        self, state, child, associationrow, clearkeys, uowcommit, operation
    ):
        # this checks for None if uselist=True
        self._verify_canload(child)

        # but if uselist=False we get here.   If child is None,
        # no association row can be generated, so return.
        if child is None:
            return False

        if child is not None and not uowcommit.session._contains_state(child):
            if not child.deleted:
                util.warn(
                    "Object of type %s not in session, %s "
                    "operation along '%s' won't proceed"
                    % (mapperutil.state_class_str(child), operation, self.prop)
                )
            return False

        sync.populate_dict(
            state, self.parent, associationrow, self.prop.synchronize_pairs
        )
        sync.populate_dict(
            child,
            self.mapper,
            associationrow,
            self.prop.secondary_synchronize_pairs,
        )

        return True

    def _pks_changed(self, uowcommit, state):
        return sync.source_modified(
            uowcommit, state, self.parent, self.prop.synchronize_pairs
        )


_direction_to_processor = {
    ONETOMANY: OneToManyDP,
    MANYTOONE: ManyToOneDP,
    MANYTOMANY: ManyToManyDP,
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\intpoly.py ===
"""
Module to implement integration of uni/bivariate polynomials over
2D Polytopes and uni/bi/trivariate polynomials over 3D Polytopes.

Uses evaluation techniques as described in Chin et al. (2015) [1].


References
===========

.. [1] Chin, Eric B., Jean B. Lasserre, and N. Sukumar. "Numerical integration
of homogeneous functions on convex and nonconvex polygons and polyhedra."
Computational Mechanics 56.6 (2015): 967-981

PDF link : http://dilbert.engr.ucdavis.edu/~suku/quadrature/cls-integration.pdf
"""

from functools import cmp_to_key

from sympy.abc import x, y, z
from sympy.core import S, diff, Expr, Symbol
from sympy.core.sympify import _sympify
from sympy.geometry import Segment2D, Polygon, Point, Point2D
from sympy.polys.polytools import LC, gcd_list, degree_list, Poly
from sympy.simplify.simplify import nsimplify


def polytope_integrate(poly, expr=None, *, clockwise=False, max_degree=None):
    """Integrates polynomials over 2/3-Polytopes.

    Explanation
    ===========

    This function accepts the polytope in ``poly`` and the function in ``expr``
    (uni/bi/trivariate polynomials are implemented) and returns
    the exact integral of ``expr`` over ``poly``.

    Parameters
    ==========

    poly : The input Polygon.

    expr : The input polynomial.

    clockwise : Binary value to sort input points of 2-Polytope clockwise.(Optional)

    max_degree : The maximum degree of any monomial of the input polynomial.(Optional)

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy import Point, Polygon
    >>> from sympy.integrals.intpoly import polytope_integrate
    >>> polygon = Polygon(Point(0, 0), Point(0, 1), Point(1, 1), Point(1, 0))
    >>> polys = [1, x, y, x*y, x**2*y, x*y**2]
    >>> expr = x*y
    >>> polytope_integrate(polygon, expr)
    1/4
    >>> polytope_integrate(polygon, polys, max_degree=3)
    {1: 1, x: 1/2, y: 1/2, x*y: 1/4, x*y**2: 1/6, x**2*y: 1/6}
    """
    if clockwise:
        if isinstance(poly, Polygon):
            poly = Polygon(*point_sort(poly.vertices), evaluate=False)
        else:
            raise TypeError("clockwise=True works for only 2-Polytope"
                            "V-representation input")

    if isinstance(poly, Polygon):
        # For Vertex Representation(2D case)
        hp_params = hyperplane_parameters(poly)
        facets = poly.sides
    elif len(poly[0]) == 2:
        # For Hyperplane Representation(2D case)
        plen = len(poly)
        if len(poly[0][0]) == 2:
            intersections = [intersection(poly[(i - 1) % plen], poly[i],
                                          "plane2D")
                             for i in range(0, plen)]
            hp_params = poly
            lints = len(intersections)
            facets = [Segment2D(intersections[i],
                                intersections[(i + 1) % lints])
                      for i in range(lints)]
        else:
            raise NotImplementedError("Integration for H-representation 3D"
                                      "case not implemented yet.")
    else:
        # For Vertex Representation(3D case)
        vertices = poly[0]
        facets = poly[1:]
        hp_params = hyperplane_parameters(facets, vertices)

        if max_degree is None:
            if expr is None:
                raise TypeError('Input expression must be a valid SymPy expression')
            return main_integrate3d(expr, facets, vertices, hp_params)

    if max_degree is not None:
        result = {}
        if expr is not None:
            f_expr = []
            for e in expr:
                _ = decompose(e)
                if len(_) == 1 and not _.popitem()[0]:
                    f_expr.append(e)
                elif Poly(e).total_degree() <= max_degree:
                    f_expr.append(e)
            expr = f_expr

        if not isinstance(expr, list) and expr is not None:
            raise TypeError('Input polynomials must be list of expressions')

        if len(hp_params[0][0]) == 3:
            result_dict = main_integrate3d(0, facets, vertices, hp_params,
                                           max_degree)
        else:
            result_dict = main_integrate(0, facets, hp_params, max_degree)

        if expr is None:
            return result_dict

        for poly in expr:
            poly = _sympify(poly)
            if poly not in result:
                if poly.is_zero:
                    result[S.Zero] = S.Zero
                    continue
                integral_value = S.Zero
                monoms = decompose(poly, separate=True)
                for monom in monoms:
                    monom = nsimplify(monom)
                    coeff, m = strip(monom)
                    integral_value += result_dict[m] * coeff
                result[poly] = integral_value
        return result

    if expr is None:
        raise TypeError('Input expression must be a valid SymPy expression')

    return main_integrate(expr, facets, hp_params)


def strip(monom):
    if monom.is_zero:
        return S.Zero, S.Zero
    elif monom.is_number:
        return monom, S.One
    else:
        coeff = LC(monom)
        return coeff, monom / coeff

def _polynomial_integrate(polynomials, facets, hp_params):
    dims = (x, y)
    dim_length = len(dims)
    integral_value = S.Zero
    for deg in polynomials:
        poly_contribute = S.Zero
        facet_count = 0
        for hp in hp_params:
            value_over_boundary = integration_reduction(facets,
                                                        facet_count,
                                                        hp[0], hp[1],
                                                        polynomials[deg],
                                                        dims, deg)
            poly_contribute += value_over_boundary * (hp[1] / norm(hp[0]))
            facet_count += 1
        poly_contribute /= (dim_length + deg)
        integral_value += poly_contribute

    return integral_value


def main_integrate3d(expr, facets, vertices, hp_params, max_degree=None):
    """Function to translate the problem of integrating uni/bi/tri-variate
    polynomials over a 3-Polytope to integrating over its faces.
    This is done using Generalized Stokes' Theorem and Euler's Theorem.

    Parameters
    ==========

    expr :
        The input polynomial.
    facets :
        Faces of the 3-Polytope(expressed as indices of `vertices`).
    vertices :
        Vertices that constitute the Polytope.
    hp_params :
        Hyperplane Parameters of the facets.
    max_degree : optional
        Max degree of constituent monomial in given list of polynomial.

    Examples
    ========

    >>> from sympy.integrals.intpoly import main_integrate3d, \
    hyperplane_parameters
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> vertices = cube[0]
    >>> faces = cube[1:]
    >>> hp_params = hyperplane_parameters(faces, vertices)
    >>> main_integrate3d(1, faces, vertices, hp_params)
    -125
    """
    result = {}
    dims = (x, y, z)
    dim_length = len(dims)
    if max_degree:
        grad_terms = gradient_terms(max_degree, 3)
        flat_list = [term for z_terms in grad_terms
                     for x_term in z_terms
                     for term in x_term]

        for term in flat_list:
            result[term[0]] = 0

        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            x0 = vertices[facets[facet_count][0]]

            for i, monom in enumerate(flat_list):
                #  Every monomial is a tuple :
                #  (term, x_degree, y_degree, z_degree, value over boundary)
                expr, x_d, y_d, z_d, z_index, y_index, x_index, _ = monom
                degree = x_d + y_d + z_d
                if b.is_zero:
                    value_over_face = S.Zero
                else:
                    value_over_face = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, expr, degree, dims,
                                                      x_index, y_index,
                                                      z_index, x0, grad_terms,
                                                      i, vertices, hp)
                monom[7] = value_over_face
                result[expr] += value_over_face * \
                    (b / norm(a)) / (dim_length + x_d + y_d + z_d)
        return result
    else:
        integral_value = S.Zero
        polynomials = decompose(expr)
        for deg in polynomials:
            poly_contribute = S.Zero
            facet_count = 0
            for i, facet in enumerate(facets):
                hp = hp_params[i]
                if hp[1].is_zero:
                    continue
                pi = polygon_integrate(facet, hp, i, facets, vertices, expr, deg)
                poly_contribute += pi *\
                    (hp[1] / norm(tuple(hp[0])))
                facet_count += 1
            poly_contribute /= (dim_length + deg)
            integral_value += poly_contribute
    return integral_value


def main_integrate(expr, facets, hp_params, max_degree=None):
    """Function to translate the problem of integrating univariate/bivariate
    polynomials over a 2-Polytope to integrating over its boundary facets.
    This is done using Generalized Stokes's Theorem and Euler's Theorem.

    Parameters
    ==========

    expr :
        The input polynomial.
    facets :
        Facets(Line Segments) of the 2-Polytope.
    hp_params :
        Hyperplane Parameters of the facets.
    max_degree : optional
        The maximum degree of any monomial of the input polynomial.

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import main_integrate,\
    hyperplane_parameters
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> hp_params = hyperplane_parameters(triangle)
    >>> main_integrate(x**2 + y**2, facets, hp_params)
    325/6
    """
    dims = (x, y)
    dim_length = len(dims)
    result = {}

    if max_degree:
        grad_terms = [[0, 0, 0, 0]] + gradient_terms(max_degree)

        for facet_count, hp in enumerate(hp_params):
            a, b = hp[0], hp[1]
            x0 = facets[facet_count].points[0]

            for i, monom in enumerate(grad_terms):
                #  Every monomial is a tuple :
                #  (term, x_degree, y_degree, value over boundary)
                m, x_d, y_d, _ = monom
                value = result.get(m, None)
                degree = S.Zero
                if b.is_zero:
                    value_over_boundary = S.Zero
                else:
                    degree = x_d + y_d
                    value_over_boundary = \
                        integration_reduction_dynamic(facets, facet_count, a,
                                                      b, m, degree, dims, x_d,
                                                      y_d, max_degree, x0,
                                                      grad_terms, i)
                monom[3] = value_over_boundary
                if value is not None:
                    result[m] += value_over_boundary * \
                                        (b / norm(a)) / (dim_length + degree)
                else:
                    result[m] = value_over_boundary * \
                                (b / norm(a)) / (dim_length + degree)
        return result
    else:
        if not isinstance(expr, list):
            polynomials = decompose(expr)
            return _polynomial_integrate(polynomials, facets, hp_params)
        else:
            return {e: _polynomial_integrate(decompose(e), facets, hp_params) for e in expr}


def polygon_integrate(facet, hp_param, index, facets, vertices, expr, degree):
    """Helper function to integrate the input uni/bi/trivariate polynomial
    over a certain face of the 3-Polytope.

    Parameters
    ==========

    facet :
        Particular face of the 3-Polytope over which ``expr`` is integrated.
    index :
        The index of ``facet`` in ``facets``.
    facets :
        Faces of the 3-Polytope(expressed as indices of `vertices`).
    vertices :
        Vertices that constitute the facet.
    expr :
        The input polynomial.
    degree :
        Degree of ``expr``.

    Examples
    ========

    >>> from sympy.integrals.intpoly import polygon_integrate
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                 (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                 [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                 [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> facet = cube[1]
    >>> facets = cube[1:]
    >>> vertices = cube[0]
    >>> polygon_integrate(facet, [(0, 1, 0), 5], 0, facets, vertices, 1, 0)
    -25
    """
    expr = S(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = vertices[facet[0]]
    facet_len = len(facet)
    for i, fac in enumerate(facet):
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])
        result += distance_to_side(x0, side, hp_param[0]) *\
            lineseg_integrate(facet, i, side, expr, degree)
    if not expr.is_number:
        expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] +\
            diff(expr, z) * x0[2]
        result += polygon_integrate(facet, hp_param, index, facets, vertices,
                                    expr, degree - 1)
    result /= (degree + 2)
    return result


def distance_to_side(point, line_seg, A):
    """Helper function to compute the signed distance between given 3D point
    and a line segment.

    Parameters
    ==========

    point : 3D Point
    line_seg : Line Segment

    Examples
    ========

    >>> from sympy.integrals.intpoly import distance_to_side
    >>> point = (0, 0, 0)
    >>> distance_to_side(point, [(0, 0, 1), (0, 1, 0)], (1, 0, 0))
    -sqrt(2)/2
    """
    x1, x2 = line_seg
    rev_normal = [-1 * S(i)/norm(A) for i in A]
    vector = [x2[i] - x1[i] for i in range(0, 3)]
    vector = [vector[i]/norm(vector) for i in range(0, 3)]

    n_side = cross_product((0, 0, 0), rev_normal, vector)
    vectorx0 = [line_seg[0][i] - point[i] for i in range(0, 3)]
    dot_product = sum(vectorx0[i] * n_side[i] for i in range(0, 3))

    return dot_product


def lineseg_integrate(polygon, index, line_seg, expr, degree):
    """Helper function to compute the line integral of ``expr`` over ``line_seg``.

    Parameters
    ===========

    polygon :
        Face of a 3-Polytope.
    index :
        Index of line_seg in polygon.
    line_seg :
        Line Segment.

    Examples
    ========

    >>> from sympy.integrals.intpoly import lineseg_integrate
    >>> polygon = [(0, 5, 0), (5, 5, 0), (5, 5, 5), (0, 5, 5)]
    >>> line_seg = [(0, 5, 0), (5, 5, 0)]
    >>> lineseg_integrate(polygon, 0, line_seg, 1, 0)
    5
    """
    expr = _sympify(expr)
    if expr.is_zero:
        return S.Zero
    result = S.Zero
    x0 = line_seg[0]
    distance = norm(tuple([line_seg[1][i] - line_seg[0][i] for i in
                           range(3)]))
    if isinstance(expr, Expr):
        expr_dict = {x: line_seg[1][0],
                     y: line_seg[1][1],
                     z: line_seg[1][2]}
        result += distance * expr.subs(expr_dict)
    else:
        result += distance * expr

    expr = diff(expr, x) * x0[0] + diff(expr, y) * x0[1] +\
        diff(expr, z) * x0[2]

    result += lineseg_integrate(polygon, index, line_seg, expr, degree - 1)
    result /= (degree + 1)
    return result


def integration_reduction(facets, index, a, b, expr, dims, degree):
    """Helper method for main_integrate. Returns the value of the input
    expression evaluated over the polytope facet referenced by a given index.

    Parameters
    ===========

    facets :
        List of facets of the polytope.
    index :
        Index referencing the facet to integrate the expression over.
    a :
        Hyperplane parameter denoting direction.
    b :
        Hyperplane parameter denoting distance.
    expr :
        The expression to integrate over the facet.
    dims :
        List of symbols denoting axes.
    degree :
        Degree of the homogeneous polynomial.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import integration_reduction,\
    hyperplane_parameters
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> a, b = hyperplane_parameters(triangle)[0]
    >>> integration_reduction(facets, 0, a, b, 1, (x, y), 0)
    5
    """
    expr = _sympify(expr)
    if expr.is_zero:
        return expr

    value = S.Zero
    x0 = facets[index].points[0]
    m = len(facets)
    gens = (x, y)

    inner_product = diff(expr, gens[0]) * x0[0] + diff(expr, gens[1]) * x0[1]

    if inner_product != 0:
        value += integration_reduction(facets, index, a, b,
                                       inner_product, dims, degree - 1)

    value += left_integral2D(m, index, facets, x0, expr, gens)

    return value/(len(dims) + degree - 1)


def left_integral2D(m, index, facets, x0, expr, gens):
    """Computes the left integral of Eq 10 in Chin et al.
    For the 2D case, the integral is just an evaluation of the polynomial
    at the intersection of two facets which is multiplied by the distance
    between the first point of facet and that intersection.

    Parameters
    ==========

    m :
        No. of hyperplanes.
    index :
        Index of facet to find intersections with.
    facets :
        List of facets(Line Segments in 2D case).
    x0 :
        First point on facet referenced by index.
    expr :
        Input polynomial
    gens :
        Generators which generate the polynomial

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import left_integral2D
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> left_integral2D(3, 0, facets, facets[0].points[0], 1, (x, y))
    5
    """
    value = S.Zero
    for j in range(m):
        intersect = ()
        if j in ((index - 1) % m, (index + 1) % m):
            intersect = intersection(facets[index], facets[j], "segment2D")
        if intersect:
            distance_origin = norm(tuple(map(lambda x, y: x - y,
                                             intersect, x0)))
            if is_vertex(intersect):
                if isinstance(expr, Expr):
                    if len(gens) == 3:
                        expr_dict = {gens[0]: intersect[0],
                                     gens[1]: intersect[1],
                                     gens[2]: intersect[2]}
                    else:
                        expr_dict = {gens[0]: intersect[0],
                                     gens[1]: intersect[1]}
                    value += distance_origin * expr.subs(expr_dict)
                else:
                    value += distance_origin * expr
    return value


def integration_reduction_dynamic(facets, index, a, b, expr, degree, dims,
                                  x_index, y_index, max_index, x0,
                                  monomial_values, monom_index, vertices=None,
                                  hp_param=None):
    """The same integration_reduction function which uses a dynamic
    programming approach to compute terms by using the values of the integral
    of previously computed terms.

    Parameters
    ==========

    facets :
        Facets of the Polytope.
    index :
        Index of facet to find intersections with.(Used in left_integral()).
    a, b :
        Hyperplane parameters.
    expr :
        Input monomial.
    degree :
        Total degree of ``expr``.
    dims :
        Tuple denoting axes variables.
    x_index :
        Exponent of 'x' in ``expr``.
    y_index :
        Exponent of 'y' in ``expr``.
    max_index :
        Maximum exponent of any monomial in ``monomial_values``.
    x0 :
        First point on ``facets[index]``.
    monomial_values :
        List of monomial values constituting the polynomial.
    monom_index :
        Index of monomial whose integration is being found.
    vertices : optional
        Coordinates of vertices constituting the 3-Polytope.
    hp_param : optional
        Hyperplane Parameter of the face of the facets[index].

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import (integration_reduction_dynamic, \
            hyperplane_parameters)
    >>> from sympy import Point, Polygon
    >>> triangle = Polygon(Point(0, 3), Point(5, 3), Point(1, 1))
    >>> facets = triangle.sides
    >>> a, b = hyperplane_parameters(triangle)[0]
    >>> x0 = facets[0].points[0]
    >>> monomial_values = [[0, 0, 0, 0], [1, 0, 0, 5],\
                           [y, 0, 1, 15], [x, 1, 0, None]]
    >>> integration_reduction_dynamic(facets, 0, a, b, x, 1, (x, y), 1, 0, 1,\
                                      x0, monomial_values, 3)
    25/2
    """
    value = S.Zero
    m = len(facets)

    if expr == S.Zero:
        return expr

    if len(dims) == 2:
        if not expr.is_number:
            _, x_degree, y_degree, _ = monomial_values[monom_index]
            x_index = monom_index - max_index + \
                x_index - 2 if x_degree > 0 else 0
            y_index = monom_index - 1 if y_degree > 0 else 0
            x_value, y_value =\
                monomial_values[x_index][3], monomial_values[y_index][3]

            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1]

        value += left_integral2D(m, index, facets, x0, expr, dims)
    else:
        # For 3D use case the max_index contains the z_degree of the term
        z_index = max_index
        if not expr.is_number:
            x_degree, y_degree, z_degree = y_index,\
                                           z_index - x_index - y_index, x_index
            x_value = monomial_values[z_index - 1][y_index - 1][x_index][7]\
                if x_degree > 0 else 0
            y_value = monomial_values[z_index - 1][y_index][x_index][7]\
                if y_degree > 0 else 0
            z_value = monomial_values[z_index - 1][y_index][x_index - 1][7]\
                if z_degree > 0 else 0

            value += x_degree * x_value * x0[0] + y_degree * y_value * x0[1] \
                + z_degree * z_value * x0[2]

        value += left_integral3D(facets, index, expr,
                                 vertices, hp_param, degree)
    return value / (len(dims) + degree - 1)


def left_integral3D(facets, index, expr, vertices, hp_param, degree):
    """Computes the left integral of Eq 10 in Chin et al.

    Explanation
    ===========

    For the 3D case, this is the sum of the integral values over constituting
    line segments of the face (which is accessed by facets[index]) multiplied
    by the distance between the first point of facet and that line segment.

    Parameters
    ==========

    facets :
        List of faces of the 3-Polytope.
    index :
        Index of face over which integral is to be calculated.
    expr :
        Input polynomial.
    vertices :
        List of vertices that constitute the 3-Polytope.
    hp_param :
        The hyperplane parameters of the face.
    degree :
        Degree of the ``expr``.

    Examples
    ========

    >>> from sympy.integrals.intpoly import left_integral3D
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                 (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                 [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                 [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> facets = cube[1:]
    >>> vertices = cube[0]
    >>> left_integral3D(facets, 3, 1, vertices, ([0, -1, 0], -5), 0)
    -50
    """
    value = S.Zero
    facet = facets[index]
    x0 = vertices[facet[0]]
    facet_len = len(facet)
    for i, fac in enumerate(facet):
        side = (vertices[fac], vertices[facet[(i + 1) % facet_len]])
        value += distance_to_side(x0, side, hp_param[0]) * \
            lineseg_integrate(facet, i, side, expr, degree)
    return value


def gradient_terms(binomial_power=0, no_of_gens=2):
    """Returns a list of all the possible monomials between
    0 and y**binomial_power for 2D case and z**binomial_power
    for 3D case.

    Parameters
    ==========

    binomial_power :
        Power upto which terms are generated.
    no_of_gens :
        Denotes whether terms are being generated for 2D or 3D case.

    Examples
    ========

    >>> from sympy.integrals.intpoly import gradient_terms
    >>> gradient_terms(2)
    [[1, 0, 0, 0], [y, 0, 1, 0], [y**2, 0, 2, 0], [x, 1, 0, 0],
    [x*y, 1, 1, 0], [x**2, 2, 0, 0]]
    >>> gradient_terms(2, 3)
    [[[[1, 0, 0, 0, 0, 0, 0, 0]]], [[[y, 0, 1, 0, 1, 0, 0, 0],
    [z, 0, 0, 1, 1, 0, 1, 0]], [[x, 1, 0, 0, 1, 1, 0, 0]]],
    [[[y**2, 0, 2, 0, 2, 0, 0, 0], [y*z, 0, 1, 1, 2, 0, 1, 0],
    [z**2, 0, 0, 2, 2, 0, 2, 0]], [[x*y, 1, 1, 0, 2, 1, 0, 0],
    [x*z, 1, 0, 1, 2, 1, 1, 0]], [[x**2, 2, 0, 0, 2, 2, 0, 0]]]]
    """
    if no_of_gens == 2:
        count = 0
        terms = [None] * int((binomial_power ** 2 + 3 * binomial_power + 2) / 2)
        for x_count in range(0, binomial_power + 1):
            for y_count in range(0, binomial_power - x_count + 1):
                terms[count] = [x**x_count*y**y_count,
                                x_count, y_count, 0]
                count += 1
    else:
        terms = [[[[x ** x_count * y ** y_count *
                    z ** (z_count - y_count - x_count),
                    x_count, y_count, z_count - y_count - x_count,
                    z_count, x_count, z_count - y_count - x_count, 0]
                 for y_count in range(z_count - x_count, -1, -1)]
                 for x_count in range(0, z_count + 1)]
                 for z_count in range(0, binomial_power + 1)]
    return terms


def hyperplane_parameters(poly, vertices=None):
    """A helper function to return the hyperplane parameters
    of which the facets of the polytope are a part of.

    Parameters
    ==========

    poly :
        The input 2/3-Polytope.
    vertices :
        Vertex indices of 3-Polytope.

    Examples
    ========

    >>> from sympy import Point, Polygon
    >>> from sympy.integrals.intpoly import hyperplane_parameters
    >>> hyperplane_parameters(Polygon(Point(0, 3), Point(5, 3), Point(1, 1)))
    [((0, 1), 3), ((1, -2), -1), ((-2, -1), -3)]
    >>> cube = [[(0, 0, 0), (0, 0, 5), (0, 5, 0), (0, 5, 5), (5, 0, 0),\
                (5, 0, 5), (5, 5, 0), (5, 5, 5)],\
                [2, 6, 7, 3], [3, 7, 5, 1], [7, 6, 4, 5], [1, 5, 4, 0],\
                [3, 1, 0, 2], [0, 4, 6, 2]]
    >>> hyperplane_parameters(cube[1:], cube[0])
    [([0, -1, 0], -5), ([0, 0, -1], -5), ([-1, 0, 0], -5),
    ([0, 1, 0], 0), ([1, 0, 0], 0), ([0, 0, 1], 0)]
    """
    if isinstance(poly, Polygon):
        vertices = list(poly.vertices) + [poly.vertices[0]]  # Close the polygon
        params = [None] * (len(vertices) - 1)

        for i in range(len(vertices) - 1):
            v1 = vertices[i]
            v2 = vertices[i + 1]

            a1 = v1[1] - v2[1]
            a2 = v2[0] - v1[0]
            b = v2[0] * v1[1] - v2[1] * v1[0]

            factor = gcd_list([a1, a2, b])

            b = S(b) / factor
            a = (S(a1) / factor, S(a2) / factor)
            params[i] = (a, b)
    else:
        params = [None] * len(poly)
        for i, polygon in enumerate(poly):
            v1, v2, v3 = [vertices[vertex] for vertex in polygon[:3]]
            normal = cross_product(v1, v2, v3)
            b = sum(normal[j] * v1[j] for j in range(0, 3))
            fac = gcd_list(normal)
            if fac.is_zero:
                fac = 1
            normal = [j / fac for j in normal]
            b = b / fac
            params[i] = (normal, b)
    return params


def cross_product(v1, v2, v3):
    """Returns the cross-product of vectors (v2 - v1) and (v3 - v1)
    That is : (v2 - v1) X (v3 - v1)
    """
    v2 = [v2[j] - v1[j] for j in range(0, 3)]
    v3 = [v3[j] - v1[j] for j in range(0, 3)]
    return [v3[2] * v2[1] - v3[1] * v2[2],
            v3[0] * v2[2] - v3[2] * v2[0],
            v3[1] * v2[0] - v3[0] * v2[1]]


def best_origin(a, b, lineseg, expr):
    """Helper method for polytope_integrate. Currently not used in the main
    algorithm.

    Explanation
    ===========

    Returns a point on the lineseg whose vector inner product with the
    divergence of `expr` yields an expression with the least maximum
    total power.

    Parameters
    ==========

    a :
        Hyperplane parameter denoting direction.
    b :
        Hyperplane parameter denoting distance.
    lineseg :
        Line segment on which to find the origin.
    expr :
        The expression which determines the best point.

    Algorithm(currently works only for 2D use case)
    ===============================================

    1 > Firstly, check for edge cases. Here that would refer to vertical
        or horizontal lines.

    2 > If input expression is a polynomial containing more than one generator
        then find out the total power of each of the generators.

        x**2 + 3 + x*y + x**4*y**5 ---> {x: 7, y: 6}

        If expression is a constant value then pick the first boundary point
        of the line segment.

    3 > First check if a point exists on the line segment where the value of
        the highest power generator becomes 0. If not check if the value of
        the next highest becomes 0. If none becomes 0 within line segment
        constraints then pick the first boundary point of the line segment.
        Actually, any point lying on the segment can be picked as best origin
        in the last case.

    Examples
    ========

    >>> from sympy.integrals.intpoly import best_origin
    >>> from sympy.abc import x, y
    >>> from sympy import Point, Segment2D
    >>> l = Segment2D(Point(0, 3), Point(1, 1))
    >>> expr = x**3*y**7
    >>> best_origin((2, 1), 3, l, expr)
    (0, 3.0)
    """
    a1, b1 = lineseg.points[0]

    def x_axis_cut(ls):
        """Returns the point where the input line segment
        intersects the x-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        p, q = ls.points
        if p.y.is_zero:
            return tuple(p)
        elif q.y.is_zero:
            return tuple(q)
        elif p.y/q.y < S.Zero:
            return p.y * (p.x - q.x)/(q.y - p.y) + p.x, S.Zero
        else:
            return ()

    def y_axis_cut(ls):
        """Returns the point where the input line segment
        intersects the y-axis.

        Parameters
        ==========

        ls :
            Line segment
        """
        p, q = ls.points
        if p.x.is_zero:
            return tuple(p)
        elif q.x.is_zero:
            return tuple(q)
        elif p.x/q.x < S.Zero:
            return S.Zero, p.x * (p.y - q.y)/(q.x - p.x) + p.y
        else:
            return ()

    gens = (x, y)
    power_gens = {}

    for i in gens:
        power_gens[i] = S.Zero

    if len(gens) > 1:
        # Special case for vertical and horizontal lines
        if len(gens) == 2:
            if a[0] == 0:
                if y_axis_cut(lineseg):
                    return S.Zero, b/a[1]
                else:
                    return a1, b1
            elif a[1] == 0:
                if x_axis_cut(lineseg):
                    return b/a[0], S.Zero
                else:
                    return a1, b1

        if isinstance(expr, Expr):  # Find the sum total of power of each
            if expr.is_Add:         # generator and store in a dictionary.
                for monomial in expr.args:
                    if monomial.is_Pow:
                        if monomial.args[0] in gens:
                            power_gens[monomial.args[0]] += monomial.args[1]
                    else:
                        for univariate in monomial.args:
                            term_type = len(univariate.args)
                            if term_type == 0 and univariate in gens:
                                power_gens[univariate] += 1
                            elif term_type == 2 and univariate.args[0] in gens:
                                power_gens[univariate.args[0]] +=\
                                           univariate.args[1]
            elif expr.is_Mul:
                for term in expr.args:
                    term_type = len(term.args)
                    if term_type == 0 and term in gens:
                        power_gens[term] += 1
                    elif term_type == 2 and term.args[0] in gens:
                        power_gens[term.args[0]] += term.args[1]
            elif expr.is_Pow:
                power_gens[expr.args[0]] = expr.args[1]
            elif expr.is_Symbol:
                power_gens[expr] += 1
        else:  # If `expr` is a constant take first vertex of the line segment.
            return a1, b1

        #  TODO : This part is quite hacky. Should be made more robust with
        #  TODO : respect to symbol names and scalable w.r.t higher dimensions.
        power_gens = sorted(power_gens.items(), key=lambda k: str(k[0]))
        if power_gens[0][1] >= power_gens[1][1]:
            if y_axis_cut(lineseg):
                x0 = (S.Zero, b / a[1])
            elif x_axis_cut(lineseg):
                x0 = (b / a[0], S.Zero)
            else:
                x0 = (a1, b1)
        else:
            if x_axis_cut(lineseg):
                x0 = (b/a[0], S.Zero)
            elif y_axis_cut(lineseg):
                x0 = (S.Zero, b/a[1])
            else:
                x0 = (a1, b1)
    else:
        x0 = (b/a[0])
    return x0


def decompose(expr, separate=False):
    """Decomposes an input polynomial into homogeneous ones of
    smaller or equal degree.

    Explanation
    ===========

    Returns a dictionary with keys as the degree of the smaller
    constituting polynomials. Values are the constituting polynomials.

    Parameters
    ==========

    expr : Expr
        Polynomial(SymPy expression).
    separate : bool
        If True then simply return a list of the constituent monomials
        If not then break up the polynomial into constituent homogeneous
        polynomials.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.integrals.intpoly import decompose
    >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5)
    {1: x + y, 2: x**2 + x*y, 5: x**3*y**2 + y**5}
    >>> decompose(x**2 + x*y + x + y + x**3*y**2 + y**5, True)
    {x, x**2, y, y**5, x*y, x**3*y**2}
    """
    poly_dict = {}

    if isinstance(expr, Expr) and not expr.is_number:
        if expr.is_Symbol:
            poly_dict[1] = expr
        elif expr.is_Add:
            symbols = expr.atoms(Symbol)
            degrees = [(sum(degree_list(monom, *symbols)), monom)
                       for monom in expr.args]
            if separate:
                return {monom[1] for monom in degrees}
            else:
                for monom in degrees:
                    degree, term = monom
                    if poly_dict.get(degree):
                        poly_dict[degree] += term
                    else:
                        poly_dict[degree] = term
        elif expr.is_Pow:
            _, degree = expr.args
            poly_dict[degree] = expr
        else:  # Now expr can only be of `Mul` type
            degree = 0
            for term in expr.args:
                term_type = len(term.args)
                if term_type == 0 and term.is_Symbol:
                    degree += 1
                elif term_type == 2:
                    degree += term.args[1]
            poly_dict[degree] = expr
    else:
        poly_dict[0] = expr

    if separate:
        return set(poly_dict.values())
    return poly_dict


def point_sort(poly, normal=None, clockwise=True):
    """Returns the same polygon with points sorted in clockwise or
    anti-clockwise order.

    Note that it's necessary for input points to be sorted in some order
    (clockwise or anti-clockwise) for the integration algorithm to work.
    As a convention algorithm has been implemented keeping clockwise
    orientation in mind.

    Parameters
    ==========

    poly:
        2D or 3D Polygon.
    normal : optional
        The normal of the plane which the 3-Polytope is a part of.
    clockwise : bool, optional
        Returns points sorted in clockwise order if True and
        anti-clockwise if False.

    Examples
    ========

    >>> from sympy.integrals.intpoly import point_sort
    >>> from sympy import Point
    >>> point_sort([Point(0, 0), Point(1, 0), Point(1, 1)])
    [Point2D(1, 1), Point2D(1, 0), Point2D(0, 0)]
    """
    pts = poly.vertices if isinstance(poly, Polygon) else poly
    n = len(pts)
    if n < 2:
        return list(pts)

    order = S.One if clockwise else S.NegativeOne
    dim = len(pts[0])
    if dim == 2:
        center = Point(sum((vertex.x for vertex in pts)) / n,
                        sum((vertex.y for vertex in pts)) / n)
    else:
        center = Point(sum((vertex.x for vertex in pts)) / n,
                        sum((vertex.y for vertex in pts)) / n,
                        sum((vertex.z for vertex in pts)) / n)

    def compare(a, b):
        if a.x - center.x >= S.Zero and b.x - center.x < S.Zero:
            return -order
        elif a.x - center.x < 0 and b.x - center.x >= 0:
            return order
        elif a.x - center.x == 0 and b.x - center.x == 0:
            if a.y - center.y >= 0 or b.y - center.y >= 0:
                return -order if a.y > b.y else order
            return -order if b.y > a.y else order

        det = (a.x - center.x) * (b.y - center.y) -\
              (b.x - center.x) * (a.y - center.y)
        if det < 0:
            return -order
        elif det > 0:
            return order

        first = (a.x - center.x) * (a.x - center.x) +\
                (a.y - center.y) * (a.y - center.y)
        second = (b.x - center.x) * (b.x - center.x) +\
                 (b.y - center.y) * (b.y - center.y)
        return -order if first > second else order

    def compare3d(a, b):
        det = cross_product(center, a, b)
        dot_product = sum(det[i] * normal[i] for i in range(0, 3))
        if dot_product < 0:
            return -order
        elif dot_product > 0:
            return order

    return sorted(pts, key=cmp_to_key(compare if dim==2 else compare3d))


def norm(point):
    """Returns the Euclidean norm of a point from origin.

    Parameters
    ==========

    point:
        This denotes a point in the dimension_al spac_e.

    Examples
    ========

    >>> from sympy.integrals.intpoly import norm
    >>> from sympy import Point
    >>> norm(Point(2, 7))
    sqrt(53)
    """
    half = S.Half
    if isinstance(point, (list, tuple)):
        return sum(coord ** 2 for coord in point) ** half
    elif isinstance(point, Point):
        if isinstance(point, Point2D):
            return (point.x ** 2 + point.y ** 2) ** half
        else:
            return (point.x ** 2 + point.y ** 2 + point.z) ** half
    elif isinstance(point, dict):
        return sum(i**2 for i in point.values()) ** half


def intersection(geom_1, geom_2, intersection_type):
    """Returns intersection between geometric objects.

    Explanation
    ===========

    Note that this function is meant for use in integration_reduction and
    at that point in the calling function the lines denoted by the segments
    surely intersect within segment boundaries. Coincident lines are taken
    to be non-intersecting. Also, the hyperplane intersection for 2D case is
    also implemented.

    Parameters
    ==========

    geom_1, geom_2:
        The input line segments.

    Examples
    ========

    >>> from sympy.integrals.intpoly import intersection
    >>> from sympy import Point, Segment2D
    >>> l1 = Segment2D(Point(1, 1), Point(3, 5))
    >>> l2 = Segment2D(Point(2, 0), Point(2, 5))
    >>> intersection(l1, l2, "segment2D")
    (2, 3)
    >>> p1 = ((-1, 0), 0)
    >>> p2 = ((0, 1), 1)
    >>> intersection(p1, p2, "plane2D")
    (0, 1)
    """
    if intersection_type[:-2] == "segment":
        if intersection_type == "segment2D":
            x1, y1 = geom_1.points[0]
            x2, y2 = geom_1.points[1]
            x3, y3 = geom_2.points[0]
            x4, y4 = geom_2.points[1]
        elif intersection_type == "segment3D":
            x1, y1, z1 = geom_1.points[0]
            x2, y2, z2 = geom_1.points[1]
            x3, y3, z3 = geom_2.points[0]
            x4, y4, z4 = geom_2.points[1]

        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom:
            t1 = x1 * y2 - y1 * x2
            t2 = x3 * y4 - x4 * y3
            return (S(t1 * (x3 - x4) - t2 * (x1 - x2)) / denom,
                    S(t1 * (y3 - y4) - t2 * (y1 - y2)) / denom)
    if intersection_type[:-2] == "plane":
        if intersection_type == "plane2D":  # Intersection of hyperplanes
            a1x, a1y = geom_1[0]
            a2x, a2y = geom_2[0]
            b1, b2 = geom_1[1], geom_2[1]

            denom = a1x * a2y - a2x * a1y
            if denom:
                return (S(b1 * a2y - b2 * a1y) / denom,
                        S(b2 * a1x - b1 * a2x) / denom)


def is_vertex(ent):
    """If the input entity is a vertex return True.

    Parameter
    =========

    ent :
        Denotes a geometric entity representing a point.

    Examples
    ========

    >>> from sympy import Point
    >>> from sympy.integrals.intpoly import is_vertex
    >>> is_vertex((2, 3))
    True
    >>> is_vertex((2, 3, 6))
    True
    >>> is_vertex(Point(2, 3))
    True
    """
    if isinstance(ent, tuple):
        if len(ent) in [2, 3]:
            return True
    elif isinstance(ent, Point):
        return True
    return False


def plot_polytope(poly):
    """Plots the 2D polytope using the functions written in plotting
    module which in turn uses matplotlib backend.

    Parameter
    =========

    poly:
        Denotes a 2-Polytope.
    """
    from sympy.plotting.plot import Plot, List2DSeries

    xl = [vertex.x for vertex in poly.vertices]
    yl = [vertex.y for vertex in poly.vertices]

    xl.append(poly.vertices[0].x)  # Closing the polygon
    yl.append(poly.vertices[0].y)

    l2ds = List2DSeries(xl, yl)
    p = Plot(l2ds, axes='label_axes=True')
    p.show()


def plot_polynomial(expr):
    """Plots the polynomial using the functions written in
    plotting module which in turn uses matplotlib backend.

    Parameter
    =========

    expr:
        Denotes a polynomial(SymPy expression).
    """
    from sympy.plotting.plot import plot3d, plot
    gens = expr.free_symbols
    if len(gens) == 2:
        plot3d(expr)
    else:
        plot(expr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\rootoftools.py ===
"""Implementation of RootOf class and related tools. """



from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
    symbols, sympify, Rational, Dummy)
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
    MultivariatePolynomialError,
    GeneratorsNeeded,
    PolynomialError,
    DomainError)
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
    roots_linear, roots_quadratic, roots_binomial,
    preprocess_roots, roots)
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
    dup_isolate_complex_roots_sqf,
    dup_isolate_real_roots_sqf)
from sympy.utilities import lambdify, public, sift, numbered_symbols

from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain


__all__ = ['CRootOf']



class _pure_key_dict:
    """A minimal dictionary that makes sure that the key is a
    univariate PurePoly instance.

    Examples
    ========

    Only the following actions are guaranteed:

    >>> from sympy.polys.rootoftools import _pure_key_dict
    >>> from sympy import PurePoly
    >>> from sympy.abc import x, y

    1) creation

    >>> P = _pure_key_dict()

    2) assignment for a PurePoly or univariate polynomial

    >>> P[x] = 1
    >>> P[PurePoly(x - y, x)] = 2

    3) retrieval based on PurePoly key comparison (use this
       instead of the get method)

    >>> P[y]
    1

    4) KeyError when trying to retrieve a nonexisting key

    >>> P[y + 1]
    Traceback (most recent call last):
    ...
    KeyError: PurePoly(y + 1, y, domain='ZZ')

    5) ability to query with ``in``

    >>> x + 1 in P
    False

    NOTE: this is a *not* a dictionary. It is a very basic object
    for internal use that makes sure to always address its cache
    via PurePoly instances. It does not, for example, implement
    ``get`` or ``setdefault``.
    """
    def __init__(self):
        self._dict = {}

    def __getitem__(self, k):
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise KeyError
            k = PurePoly(k, expand=False)
        return self._dict[k]

    def __setitem__(self, k, v):
        if not isinstance(k, PurePoly):
            if not (isinstance(k, Expr) and len(k.free_symbols) == 1):
                raise ValueError('expecting univariate expression')
            k = PurePoly(k, expand=False)
        self._dict[k] = v

    def __contains__(self, k):
        try:
            self[k]
            return True
        except KeyError:
            return False

_reals_cache = _pure_key_dict()
_complexes_cache = _pure_key_dict()


def _pure_factors(poly):
    _, factors = poly.factor_list()
    return [(PurePoly(f, expand=False), m) for f, m in factors]


def _imag_count_of_factor(f):
    """Return the number of imaginary roots for irreducible
    univariate polynomial ``f``.
    """
    terms = [(i, j) for (i,), j in f.terms()]
    if any(i % 2 for i, j in terms):
        return 0
    # update signs
    even = [(i, I**i*j) for i, j in terms]
    even = Poly.from_dict(dict(even), Dummy('x'))
    return int(even.count_roots(-oo, oo))


@public
def rootof(f, x, index=None, radicals=True, expand=True):
    """An indexed root of a univariate polynomial.

    Returns either a :obj:`ComplexRootOf` object or an explicit
    expression involving radicals.

    Parameters
    ==========

    f : Expr
        Univariate polynomial.
    x : Symbol, optional
        Generator for ``f``.
    index : int or Integer
    radicals : bool
               Return a radical expression if possible.
    expand : bool
             Expand ``f``.
    """
    return CRootOf(f, x, index=index, radicals=radicals, expand=expand)


@public
class RootOf(Expr):
    """Represents a root of a univariate polynomial.

    Base class for roots of different kinds of polynomials.
    Only complex roots are currently supported.
    """

    __slots__ = ('poly',)

    def __new__(cls, f, x, index=None, radicals=True, expand=True):
        """Construct a new ``CRootOf`` object for ``k``-th root of ``f``."""
        return rootof(f, x, index=index, radicals=radicals, expand=expand)

@public
class ComplexRootOf(RootOf):
    """Represents an indexed complex root of a polynomial.

    Roots of a univariate polynomial separated into disjoint
    real or complex intervals and indexed in a fixed order:

    * real roots come first and are sorted in increasing order;
    * complex roots come next and are sorted primarily by increasing
      real part, secondarily by increasing imaginary part.

    Currently only rational coefficients are allowed.
    Can be imported as ``CRootOf``. To avoid confusion, the
    generator must be a Symbol.


    Examples
    ========

    >>> from sympy import CRootOf, rootof
    >>> from sympy.abc import x

    CRootOf is a way to reference a particular root of a
    polynomial. If there is a rational root, it will be returned:

    >>> CRootOf.clear_cache()  # for doctest reproducibility
    >>> CRootOf(x**2 - 4, 0)
    -2

    Whether roots involving radicals are returned or not
    depends on whether the ``radicals`` flag is true (which is
    set to True with rootof):

    >>> CRootOf(x**2 - 3, 0)
    CRootOf(x**2 - 3, 0)
    >>> CRootOf(x**2 - 3, 0, radicals=True)
    -sqrt(3)
    >>> rootof(x**2 - 3, 0)
    -sqrt(3)

    The following cannot be expressed in terms of radicals:

    >>> r = rootof(4*x**5 + 16*x**3 + 12*x**2 + 7, 0); r
    CRootOf(4*x**5 + 16*x**3 + 12*x**2 + 7, 0)

    The root bounds can be seen, however, and they are used by the
    evaluation methods to get numerical approximations for the root.

    >>> interval = r._get_interval(); interval
    (-1, 0)
    >>> r.evalf(2)
    -0.98

    The evalf method refines the width of the root bounds until it
    guarantees that any decimal approximation within those bounds
    will satisfy the desired precision. It then stores the refined
    interval so subsequent requests at or below the requested
    precision will not have to recompute the root bounds and will
    return very quickly.

    Before evaluation above, the interval was

    >>> interval
    (-1, 0)

    After evaluation it is now

    >>> r._get_interval() # doctest: +SKIP
    (-165/169, -206/211)

    To reset all intervals for a given polynomial, the :meth:`_reset` method
    can be called from any CRootOf instance of the polynomial:

    >>> r._reset()
    >>> r._get_interval()
    (-1, 0)

    The :meth:`eval_approx` method will also find the root to a given
    precision but the interval is not modified unless the search
    for the root fails to converge within the root bounds. And
    the secant method is used to find the root. (The ``evalf``
    method uses bisection and will always update the interval.)

    >>> r.eval_approx(2)
    -0.98

    The interval needed to be slightly updated to find that root:

    >>> r._get_interval()
    (-1, -1/2)

    The ``evalf_rational`` will compute a rational approximation
    of the root to the desired accuracy or precision.

    >>> r.eval_rational(n=2)
    -69629/71318

    >>> t = CRootOf(x**3 + 10*x + 1, 1)
    >>> t.eval_rational(1e-1)
    15/256 - 805*I/256
    >>> t.eval_rational(1e-1, 1e-4)
    3275/65536 - 414645*I/131072
    >>> t.eval_rational(1e-4, 1e-4)
    6545/131072 - 414645*I/131072
    >>> t.eval_rational(n=2)
    104755/2097152 - 6634255*I/2097152

    Notes
    =====

    Although a PurePoly can be constructed from a non-symbol generator
    RootOf instances of non-symbols are disallowed to avoid confusion
    over what root is being represented.

    >>> from sympy import exp, PurePoly
    >>> PurePoly(x) == PurePoly(exp(x))
    True
    >>> CRootOf(x - 1, 0)
    1
    >>> CRootOf(exp(x) - 1, 0)  # would correspond to x == 0
    Traceback (most recent call last):
    ...
    sympy.polys.polyerrors.PolynomialError: generator must be a Symbol

    See Also
    ========

    eval_approx
    eval_rational

    """

    __slots__ = ('index',)
    is_complex = True
    is_number = True
    is_finite = True
    is_algebraic = True

    def __new__(cls, f, x, index=None, radicals=False, expand=True):
        """ Construct an indexed complex root of a polynomial.

        See ``rootof`` for the parameters.

        The default value of ``radicals`` is ``False`` to satisfy
        ``eval(srepr(expr) == expr``.
        """
        x = sympify(x)

        if index is None and x.is_Integer:
            x, index = None, x
        else:
            index = sympify(index)

        if index is not None and index.is_Integer:
            index = int(index)
        else:
            raise ValueError("expected an integer root index, got %s" % index)

        poly = PurePoly(f, x, greedy=False, expand=expand)

        if not poly.is_univariate:
            raise PolynomialError("only univariate polynomials are allowed")

        if not poly.gen.is_Symbol:
            # PurePoly(sin(x) + 1) == PurePoly(x + 1) but the roots of
            # x for each are not the same: issue 8617
            raise PolynomialError("generator must be a Symbol")

        degree = poly.degree()

        if degree <= 0:
            raise PolynomialError("Cannot construct CRootOf object for %s" % f)

        if index < -degree or index >= degree:
            raise IndexError("root index out of [%d, %d] range, got %d" %
                             (-degree, degree - 1, index))
        elif index < 0:
            index += degree

        dom = poly.get_domain()

        if not dom.is_Exact:
            poly = poly.to_exact()

        roots = cls._roots_trivial(poly, radicals)

        if roots is not None:
            return roots[index]

        coeff, poly = preprocess_roots(poly)
        dom = poly.get_domain()

        if not dom.is_ZZ:
            raise NotImplementedError("CRootOf is not supported over %s" % dom)

        root = cls._indexed_root(poly, index, lazy=True)
        return coeff * cls._postprocess_root(root, radicals)

    @classmethod
    def _new(cls, poly, index):
        """Construct new ``CRootOf`` object from raw data. """
        obj = Expr.__new__(cls)

        obj.poly = PurePoly(poly)
        obj.index = index

        try:
            _reals_cache[obj.poly] = _reals_cache[poly]
            _complexes_cache[obj.poly] = _complexes_cache[poly]
        except KeyError:
            pass

        return obj

    def _hashable_content(self):
        return (self.poly, self.index)

    @property
    def expr(self):
        return self.poly.as_expr()

    @property
    def args(self):
        return (self.expr, Integer(self.index))

    @property
    def free_symbols(self):
        # CRootOf currently only works with univariate expressions
        # whose poly attribute should be a PurePoly with no free
        # symbols
        return set()

    def _eval_is_real(self):
        """Return ``True`` if the root is real. """
        self._ensure_reals_init()
        return self.index < len(_reals_cache[self.poly])

    def _eval_is_imaginary(self):
        """Return ``True`` if the root is imaginary. """
        self._ensure_reals_init()
        if self.index >= len(_reals_cache[self.poly]):
            ivl = self._get_interval()
            return ivl.ax*ivl.bx <= 0  # all others are on one side or the other
        return False  # XXX is this necessary?

    @classmethod
    def real_roots(cls, poly, radicals=True):
        """Get real roots of a polynomial. """
        return cls._get_roots("_real_roots", poly, radicals)

    @classmethod
    def all_roots(cls, poly, radicals=True):
        """Get real and complex roots of a polynomial. """
        return cls._get_roots("_all_roots", poly, radicals)

    @classmethod
    def _get_reals_sqf(cls, currentfactor, use_cache=True):
        """Get real root isolating intervals for a square-free factor."""
        if use_cache and currentfactor in _reals_cache:
            real_part = _reals_cache[currentfactor]
        else:
            _reals_cache[currentfactor] = real_part = \
                dup_isolate_real_roots_sqf(
                    currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)

        return real_part

    @classmethod
    def _get_complexes_sqf(cls, currentfactor, use_cache=True):
        """Get complex root isolating intervals for a square-free factor."""
        if use_cache and currentfactor in _complexes_cache:
            complex_part = _complexes_cache[currentfactor]
        else:
            _complexes_cache[currentfactor] = complex_part = \
                dup_isolate_complex_roots_sqf(
                currentfactor.rep.to_list(), currentfactor.rep.dom, blackbox=True)
        return complex_part

    @classmethod
    def _get_reals(cls, factors, use_cache=True):
        """Compute real root isolating intervals for a list of factors. """
        reals = []

        for currentfactor, k in factors:
            try:
                if not use_cache:
                    raise KeyError
                r = _reals_cache[currentfactor]
                reals.extend([(i, currentfactor, k) for i in r])
            except KeyError:
                real_part = cls._get_reals_sqf(currentfactor, use_cache)
                new = [(root, currentfactor, k) for root in real_part]
                reals.extend(new)

        reals = cls._reals_sorted(reals)
        return reals

    @classmethod
    def _get_complexes(cls, factors, use_cache=True):
        """Compute complex root isolating intervals for a list of factors. """
        complexes = []

        for currentfactor, k in ordered(factors):
            try:
                if not use_cache:
                    raise KeyError
                c = _complexes_cache[currentfactor]
                complexes.extend([(i, currentfactor, k) for i in c])
            except KeyError:
                complex_part = cls._get_complexes_sqf(currentfactor, use_cache)
                new = [(root, currentfactor, k) for root in complex_part]
                complexes.extend(new)

        complexes = cls._complexes_sorted(complexes)
        return complexes

    @classmethod
    def _reals_sorted(cls, reals):
        """Make real isolating intervals disjoint and sort roots. """
        cache = {}

        for i, (u, f, k) in enumerate(reals):
            for j, (v, g, m) in enumerate(reals[i + 1:]):
                u, v = u.refine_disjoint(v)
                reals[i + j + 1] = (v, g, m)

            reals[i] = (u, f, k)

        reals = sorted(reals, key=lambda r: r[0].a)

        for root, currentfactor, _ in reals:
            if currentfactor in cache:
                cache[currentfactor].append(root)
            else:
                cache[currentfactor] = [root]

        for currentfactor, root in cache.items():
            _reals_cache[currentfactor] = root

        return reals

    @classmethod
    def _refine_imaginary(cls, complexes):
        sifted = sift(complexes, lambda c: c[1])
        complexes = []
        for f in ordered(sifted):
            nimag = _imag_count_of_factor(f)
            if nimag == 0:
                # refine until xbounds are neg or pos
                for u, f, k in sifted[f]:
                    while u.ax*u.bx <= 0:
                        u = u._inner_refine()
                    complexes.append((u, f, k))
            else:
                # refine until all but nimag xbounds are neg or pos
                potential_imag = list(range(len(sifted[f])))
                while True:
                    assert len(potential_imag) > 1
                    for i in list(potential_imag):
                        u, f, k = sifted[f][i]
                        if u.ax*u.bx > 0:
                            potential_imag.remove(i)
                        elif u.ax != u.bx:
                            u = u._inner_refine()
                            sifted[f][i] = u, f, k
                    if len(potential_imag) == nimag:
                        break
                complexes.extend(sifted[f])
        return complexes

    @classmethod
    def _refine_complexes(cls, complexes):
        """return complexes such that no bounding rectangles of non-conjugate
        roots would intersect. In addition, assure that neither ay nor by is
        0 to guarantee that non-real roots are distinct from real roots in
        terms of the y-bounds.
        """
        # get the intervals pairwise-disjoint.
        # If rectangles were drawn around the coordinates of the bounding
        # rectangles, no rectangles would intersect after this procedure.
        for i, (u, f, k) in enumerate(complexes):
            for j, (v, g, m) in enumerate(complexes[i + 1:]):
                u, v = u.refine_disjoint(v)
                complexes[i + j + 1] = (v, g, m)

            complexes[i] = (u, f, k)

        # refine until the x-bounds are unambiguously positive or negative
        # for non-imaginary roots
        complexes = cls._refine_imaginary(complexes)

        # make sure that all y bounds are off the real axis
        # and on the same side of the axis
        for i, (u, f, k) in enumerate(complexes):
            while u.ay*u.by <= 0:
                u = u.refine()
            complexes[i] = u, f, k
        return complexes

    @classmethod
    def _complexes_sorted(cls, complexes):
        """Make complex isolating intervals disjoint and sort roots. """
        complexes = cls._refine_complexes(complexes)
        # XXX don't sort until you are sure that it is compatible
        # with the indexing method but assert that the desired state
        # is not broken
        C, F = 0, 1  # location of ComplexInterval and factor
        fs = {i[F] for i in complexes}
        for i in range(1, len(complexes)):
            if complexes[i][F] != complexes[i - 1][F]:
                # if this fails the factors of a root were not
                # contiguous because a discontinuity should only
                # happen once
                fs.remove(complexes[i - 1][F])
        for i, cmplx in enumerate(complexes):
            # negative im part (conj=True) comes before
            # positive im part (conj=False)
            assert cmplx[C].conj is (i % 2 == 0)

        # update cache
        cache = {}
        # -- collate
        for root, currentfactor, _ in complexes:
            cache.setdefault(currentfactor, []).append(root)
        # -- store
        for currentfactor, root in cache.items():
            _complexes_cache[currentfactor] = root

        return complexes

    @classmethod
    def _reals_index(cls, reals, index):
        """
        Map initial real root index to an index in a factor where
        the root belongs.
        """
        i = 0

        for j, (_, currentfactor, k) in enumerate(reals):
            if index < i + k:
                poly, index = currentfactor, 0

                for _, currentfactor, _ in reals[:j]:
                    if currentfactor == poly:
                        index += 1

                return poly, index
            else:
                i += k

    @classmethod
    def _complexes_index(cls, complexes, index):
        """
        Map initial complex root index to an index in a factor where
        the root belongs.
        """
        i = 0
        for j, (_, currentfactor, k) in enumerate(complexes):
            if index < i + k:
                poly, index = currentfactor, 0

                for _, currentfactor, _ in complexes[:j]:
                    if currentfactor == poly:
                        index += 1

                index += len(_reals_cache[poly])

                return poly, index
            else:
                i += k

    @classmethod
    def _count_roots(cls, roots):
        """Count the number of real or complex roots with multiplicities."""
        return sum(k for _, _, k in roots)

    @classmethod
    def _indexed_root(cls, poly, index, lazy=False):
        """Get a root of a composite polynomial by index. """
        factors = _pure_factors(poly)

        # If the given poly is already irreducible, then the index does not
        # need to be adjusted, and we can postpone the heavy lifting of
        # computing and refining isolating intervals until that is needed.
        # Note, however, that `_pure_factors()` extracts a negative leading
        # coeff if present, so `factors[0][0]` may differ from `poly`, and
        # is the "normalized" version of `poly` that we must return.
        if lazy and len(factors) == 1 and factors[0][1] == 1:
            return factors[0][0], index

        reals = cls._get_reals(factors)
        reals_count = cls._count_roots(reals)

        if index < reals_count:
            return cls._reals_index(reals, index)
        else:
            complexes = cls._get_complexes(factors)
            return cls._complexes_index(complexes, index - reals_count)

    def _ensure_reals_init(self):
        """Ensure that our poly has entries in the reals cache. """
        if self.poly not in _reals_cache:
            self._indexed_root(self.poly, self.index)

    def _ensure_complexes_init(self):
        """Ensure that our poly has entries in the complexes cache. """
        if self.poly not in _complexes_cache:
            self._indexed_root(self.poly, self.index)

    @classmethod
    def _real_roots(cls, poly):
        """Get real roots of a composite polynomial. """
        factors = _pure_factors(poly)

        reals = cls._get_reals(factors)
        reals_count = cls._count_roots(reals)

        roots = []

        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))

        return roots

    def _reset(self):
        """
        Reset all intervals
        """
        self._all_roots(self.poly, use_cache=False)

    @classmethod
    def _all_roots(cls, poly, use_cache=True):
        """Get real and complex roots of a composite polynomial. """
        factors = _pure_factors(poly)

        reals = cls._get_reals(factors, use_cache=use_cache)
        reals_count = cls._count_roots(reals)

        roots = []

        for index in range(0, reals_count):
            roots.append(cls._reals_index(reals, index))

        complexes = cls._get_complexes(factors, use_cache=use_cache)
        complexes_count = cls._count_roots(complexes)

        for index in range(0, complexes_count):
            roots.append(cls._complexes_index(complexes, index))

        return roots

    @classmethod
    @cacheit
    def _roots_trivial(cls, poly, radicals):
        """Compute roots in linear, quadratic and binomial cases. """
        if poly.degree() == 1:
            return roots_linear(poly)

        if not radicals:
            return None

        if poly.degree() == 2:
            return roots_quadratic(poly)
        elif poly.length() == 2 and poly.TC():
            return roots_binomial(poly)
        else:
            return None

    @classmethod
    def _preprocess_roots(cls, poly):
        """Take heroic measures to make ``poly`` compatible with ``CRootOf``."""
        dom = poly.get_domain()

        if not dom.is_Exact:
            poly = poly.to_exact()

        coeff, poly = preprocess_roots(poly)
        dom = poly.get_domain()

        if not dom.is_ZZ:
            raise NotImplementedError(
                "sorted roots not supported over %s" % dom)

        return coeff, poly

    @classmethod
    def _postprocess_root(cls, root, radicals):
        """Return the root if it is trivial or a ``CRootOf`` object. """
        poly, index = root
        roots = cls._roots_trivial(poly, radicals)

        if roots is not None:
            return roots[index]
        else:
            return cls._new(poly, index)

    @classmethod
    def _get_roots(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind. """
        if not poly.is_univariate:
            raise PolynomialError("only univariate polynomials are allowed")

        dom = poly.get_domain()

        # get rid of gen and it's free symbol
        d = Dummy()
        poly = poly.subs(poly.gen, d)
        x = symbols('x')
        # see what others are left and select x or a numbered x
        # that doesn't clash
        free_names = {str(i) for i in poly.free_symbols}
        for x in chain((symbols('x'),), numbered_symbols('x')):
            if x.name not in free_names:
                poly = poly.replace(d, x)
                break

        if dom.is_QQ or dom.is_ZZ:
            return cls._get_roots_qq(method, poly, radicals)
        elif dom.is_AlgebraicField or dom.is_ZZ_I or dom.is_QQ_I:
            return cls._get_roots_alg(method, poly, radicals)
        else:
            # XXX: not sure how to handle ZZ[x] which appears in some tests?
            # this makes the tests pass alright but has to be a better way?
            return cls._get_roots_qq(method, poly, radicals)


    @classmethod
    def _get_roots_qq(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind
         for polynomials with rational coefficients. """
        coeff, poly = cls._preprocess_roots(poly)
        roots = []

        for root in getattr(cls, method)(poly):
            roots.append(coeff*cls._postprocess_root(root, radicals))

        return roots

    @classmethod
    def _get_roots_alg(cls, method, poly, radicals):
        """Return postprocessed roots of specified kind
         for polynomials with algebraic coefficients. It assumes
         the domain is already an algebraic field. First it
         finds the roots using _get_roots_qq, then uses the
         square-free factors to filter roots and get the correct
         multiplicity.
         """

        # Existing QQ code can find and sort the roots
        roots = cls._get_roots_qq(method, poly.lift(), radicals)

        subroots = {}
        for f, m in poly.sqf_list()[1]:
            if method == "_real_roots":
                roots_filt = f.which_real_roots(roots)
            elif method == "_all_roots":
                roots_filt = f.which_all_roots(roots)
            for r in roots_filt:
                subroots[r] = m

        roots_seen = set()
        roots_flat = []
        for r in roots:
            if r in subroots and r not in roots_seen:
                m = subroots[r]
                roots_flat.extend([r] * m)
                roots_seen.add(r)

        return roots_flat

    @classmethod
    def clear_cache(cls):
        """Reset cache for reals and complexes.

        The intervals used to approximate a root instance are updated
        as needed. When a request is made to see the intervals, the
        most current values are shown. `clear_cache` will reset all
        CRootOf instances back to their original state.

        See Also
        ========

        _reset
        """
        global _reals_cache, _complexes_cache
        _reals_cache = _pure_key_dict()
        _complexes_cache = _pure_key_dict()

    def _get_interval(self):
        """Internal function for retrieving isolation interval from cache. """
        self._ensure_reals_init()
        if self.is_real:
            return _reals_cache[self.poly][self.index]
        else:
            reals_count = len(_reals_cache[self.poly])
            self._ensure_complexes_init()
            return _complexes_cache[self.poly][self.index - reals_count]

    def _set_interval(self, interval):
        """Internal function for updating isolation interval in cache. """
        self._ensure_reals_init()
        if self.is_real:
            _reals_cache[self.poly][self.index] = interval
        else:
            reals_count = len(_reals_cache[self.poly])
            self._ensure_complexes_init()
            _complexes_cache[self.poly][self.index - reals_count] = interval

    def _eval_subs(self, old, new):
        # don't allow subs to change anything
        return self

    def _eval_conjugate(self):
        if self.is_real:
            return self
        expr, i = self.args
        return self.func(expr, i + (1 if self._get_interval().conj else -1))

    def eval_approx(self, n, return_mpmath=False):
        """Evaluate this complex root to the given precision.

        This uses secant method and root bounds are used to both
        generate an initial guess and to check that the root
        returned is valid. If ever the method converges outside the
        root bounds, the bounds will be made smaller and updated.
        """
        prec = dps_to_prec(n)
        with workprec(prec):
            g = self.poly.gen
            if not g.is_Symbol:
                d = Dummy('x')
                if self.is_imaginary:
                    d *= I
                func = lambdify(d, self.expr.subs(g, d))
            else:
                expr = self.expr
                if self.is_imaginary:
                    expr = self.expr.subs(g, I*g)
                func = lambdify(g, expr)

            interval = self._get_interval()
            while True:
                if self.is_real:
                    a = mpf(str(interval.a))
                    b = mpf(str(interval.b))
                    if a == b:
                        root = a
                        break
                    x0 = mpf(str(interval.center))
                    x1 = x0 + mpf(str(interval.dx))/4
                elif self.is_imaginary:
                    a = mpf(str(interval.ay))
                    b = mpf(str(interval.by))
                    if a == b:
                        root = mpc(mpf('0'), a)
                        break
                    x0 = mpf(str(interval.center[1]))
                    x1 = x0 + mpf(str(interval.dy))/4
                else:
                    ax = mpf(str(interval.ax))
                    bx = mpf(str(interval.bx))
                    ay = mpf(str(interval.ay))
                    by = mpf(str(interval.by))
                    if ax == bx and ay == by:
                        root = mpc(ax, ay)
                        break
                    x0 = mpc(*map(str, interval.center))
                    x1 = x0 + mpc(*map(str, (interval.dx, interval.dy)))/4
                try:
                    # without a tolerance, this will return when (to within
                    # the given precision) x_i == x_{i-1}
                    root = findroot(func, (x0, x1))
                    # If the (real or complex) root is not in the 'interval',
                    # then keep refining the interval. This happens if findroot
                    # accidentally finds a different root outside of this
                    # interval because our initial estimate 'x0' was not close
                    # enough. It is also possible that the secant method will
                    # get trapped by a max/min in the interval; the root
                    # verification by findroot will raise a ValueError in this
                    # case and the interval will then be tightened -- and
                    # eventually the root will be found.
                    #
                    # It is also possible that findroot will not have any
                    # successful iterations to process (in which case it
                    # will fail to initialize a variable that is tested
                    # after the iterations and raise an UnboundLocalError).
                    if self.is_real or self.is_imaginary:
                        if not bool(root.imag) == self.is_real and (
                                a <= root <= b):
                            if self.is_imaginary:
                                root = mpc(mpf('0'), root.real)
                            break
                    elif (ax <= root.real <= bx and ay <= root.imag <= by):
                        break
                except (UnboundLocalError, ValueError):
                    pass
                interval = interval.refine()

        # update the interval so we at least (for this precision or
        # less) don't have much work to do to recompute the root
        self._set_interval(interval)
        if return_mpmath:
            return root
        return (Float._new(root.real._mpf_, prec) +
            I*Float._new(root.imag._mpf_, prec))

    def _eval_evalf(self, prec, **kwargs):
        """Evaluate this complex root to the given precision."""
        # all kwargs are ignored
        return self.eval_rational(n=prec_to_dps(prec))._evalf(prec)

    def eval_rational(self, dx=None, dy=None, n=15):
        """
        Return a Rational approximation of ``self`` that has real
        and imaginary component approximations that are within ``dx``
        and ``dy`` of the true values, respectively. Alternatively,
        ``n`` digits of precision can be specified.

        The interval is refined with bisection and is sure to
        converge. The root bounds are updated when the refinement
        is complete so recalculation at the same or lesser precision
        will not have to repeat the refinement and should be much
        faster.

        The following example first obtains Rational approximation to
        1e-8 accuracy for all roots of the 4-th order Legendre
        polynomial. Since the roots are all less than 1, this will
        ensure the decimal representation of the approximation will be
        correct (including rounding) to 6 digits:

        >>> from sympy import legendre_poly, Symbol
        >>> x = Symbol("x")
        >>> p = legendre_poly(4, x, polys=True)
        >>> r = p.real_roots()[-1]
        >>> r.eval_rational(10**-8).n(6)
        0.861136

        It is not necessary to a two-step calculation, however: the
        decimal representation can be computed directly:

        >>> r.evalf(17)
        0.86113631159405258

        """
        dy = dy or dx
        if dx:
            rtol = None
            dx = dx if isinstance(dx, Rational) else Rational(str(dx))
            dy = dy if isinstance(dy, Rational) else Rational(str(dy))
        else:
            # 5 binary (or 2 decimal) digits are needed to ensure that
            # a given digit is correctly rounded
            # prec_to_dps(dps_to_prec(n) + 5) - n <= 2 (tested for
            # n in range(1000000)
            rtol = S(10)**-(n + 2)  # +2 for guard digits
        interval = self._get_interval()
        while True:
            if self.is_real:
                if rtol:
                    dx = abs(interval.center*rtol)
                interval = interval.refine_size(dx=dx)
                c = interval.center
                real = Rational(c)
                imag = S.Zero
                if not rtol or interval.dx < abs(c*rtol):
                    break
            elif self.is_imaginary:
                if rtol:
                    dy = abs(interval.center[1]*rtol)
                    dx = 1
                interval = interval.refine_size(dx=dx, dy=dy)
                c = interval.center[1]
                imag = Rational(c)
                real = S.Zero
                if not rtol or interval.dy < abs(c*rtol):
                    break
            else:
                if rtol:
                    dx = abs(interval.center[0]*rtol)
                    dy = abs(interval.center[1]*rtol)
                interval = interval.refine_size(dx, dy)
                c = interval.center
                real, imag = map(Rational, c)
                if not rtol or (
                        interval.dx < abs(c[0]*rtol) and
                        interval.dy < abs(c[1]*rtol)):
                    break

        # update the interval so we at least (for this precision or
        # less) don't have much work to do to recompute the root
        self._set_interval(interval)
        return real + I*imag


CRootOf = ComplexRootOf


@dispatch(ComplexRootOf, ComplexRootOf)
def _eval_is_eq(lhs, rhs): # noqa:F811
    # if we use is_eq to check here, we get infinite recursion
    return lhs == rhs


@dispatch(ComplexRootOf, Basic)  # type:ignore
def _eval_is_eq(lhs, rhs): # noqa:F811
    # CRootOf represents a Root, so if rhs is that root, it should set
    # the expression to zero *and* it should be in the interval of the
    # CRootOf instance. It must also be a number that agrees with the
    # is_real value of the CRootOf instance.
    if not rhs.is_number:
        return None
    if not rhs.is_finite:
        return False
    z = lhs.expr.subs(lhs.expr.free_symbols.pop(), rhs).is_zero
    if z is False:  # all roots will make z True but we don't know
        # whether this is the right root if z is True
        return False
    o = rhs.is_real, rhs.is_imaginary
    s = lhs.is_real, lhs.is_imaginary
    assert None not in s  # this is part of initial refinement
    if o != s and None not in o:
        return False
    re, im = rhs.as_real_imag()
    if lhs.is_real:
        if im:
            return False
        i = lhs._get_interval()
        a, b = [Rational(str(_)) for _ in (i.a, i.b)]
        return sympify(a <= rhs and rhs <= b)
    i = lhs._get_interval()
    r1, r2, i1, i2 = [Rational(str(j)) for j in (
        i.ax, i.bx, i.ay, i.by)]
    return is_le(r1, re) and is_le(re,r2) and is_le(i1,im) and is_le(im,i2)


@public
class RootSum(Expr):
    """Represents a sum of all roots of a univariate polynomial. """

    __slots__ = ('poly', 'fun', 'auto')

    def __new__(cls, expr, func=None, x=None, auto=True, quadratic=False):
        """Construct a new ``RootSum`` instance of roots of a polynomial."""
        coeff, poly = cls._transform(expr, x)

        if not poly.is_univariate:
            raise MultivariatePolynomialError(
                "only univariate polynomials are allowed")

        if func is None:
            func = Lambda(poly.gen, poly.gen)
        else:
            is_func = getattr(func, 'is_Function', False)

            if is_func and 1 in func.nargs:
                if not isinstance(func, Lambda):
                    func = Lambda(poly.gen, func(poly.gen))
            else:
                raise ValueError(
                    "expected a univariate function, got %s" % func)

        var, expr = func.variables[0], func.expr

        if coeff is not S.One:
            expr = expr.subs(var, coeff*var)

        deg = poly.degree()

        if not expr.has(var):
            return deg*expr

        if expr.is_Add:
            add_const, expr = expr.as_independent(var)
        else:
            add_const = S.Zero

        if expr.is_Mul:
            mul_const, expr = expr.as_independent(var)
        else:
            mul_const = S.One

        func = Lambda(var, expr)

        rational = cls._is_func_rational(poly, func)
        factors, terms = _pure_factors(poly), []

        for poly, k in factors:
            if poly.is_linear:
                term = func(roots_linear(poly)[0])
            elif quadratic and poly.is_quadratic:
                term = sum(map(func, roots_quadratic(poly)))
            else:
                if not rational or not auto:
                    term = cls._new(poly, func, auto)
                else:
                    term = cls._rational_case(poly, func)

            terms.append(k*term)

        return mul_const*Add(*terms) + deg*add_const

    @classmethod
    def _new(cls, poly, func, auto=True):
        """Construct new raw ``RootSum`` instance. """
        obj = Expr.__new__(cls)

        obj.poly = poly
        obj.fun = func
        obj.auto = auto

        return obj

    @classmethod
    def new(cls, poly, func, auto=True):
        """Construct new ``RootSum`` instance. """
        if not func.expr.has(*func.variables):
            return func.expr

        rational = cls._is_func_rational(poly, func)

        if not rational or not auto:
            return cls._new(poly, func, auto)
        else:
            return cls._rational_case(poly, func)

    @classmethod
    def _transform(cls, expr, x):
        """Transform an expression to a polynomial. """
        poly = PurePoly(expr, x, greedy=False)
        return preprocess_roots(poly)

    @classmethod
    def _is_func_rational(cls, poly, func):
        """Check if a lambda is a rational function. """
        var, expr = func.variables[0], func.expr
        return expr.is_rational_function(var)

    @classmethod
    def _rational_case(cls, poly, func):
        """Handle the rational function case. """
        roots = symbols('r:%d' % poly.degree())
        var, expr = func.variables[0], func.expr

        f = sum(expr.subs(var, r) for r in roots)
        p, q = together(f).as_numer_denom()

        domain = QQ[roots]

        p = p.expand()
        q = q.expand()

        try:
            p = Poly(p, domain=domain, expand=False)
        except GeneratorsNeeded:
            p, p_coeff = None, (p,)
        else:
            p_monom, p_coeff = zip(*p.terms())

        try:
            q = Poly(q, domain=domain, expand=False)
        except GeneratorsNeeded:
            q, q_coeff = None, (q,)
        else:
            q_monom, q_coeff = zip(*q.terms())

        coeffs, mapping = symmetrize(p_coeff + q_coeff, formal=True)
        formulas, values = viete(poly, roots), []

        for (sym, _), (_, val) in zip(mapping, formulas):
            values.append((sym, val))

        for i, (coeff, _) in enumerate(coeffs):
            coeffs[i] = coeff.subs(values)

        n = len(p_coeff)

        p_coeff = coeffs[:n]
        q_coeff = coeffs[n:]

        if p is not None:
            p = Poly(dict(zip(p_monom, p_coeff)), *p.gens).as_expr()
        else:
            (p,) = p_coeff

        if q is not None:
            q = Poly(dict(zip(q_monom, q_coeff)), *q.gens).as_expr()
        else:
            (q,) = q_coeff

        return factor(p/q)

    def _hashable_content(self):
        return (self.poly, self.fun)

    @property
    def expr(self):
        return self.poly.as_expr()

    @property
    def args(self):
        return (self.expr, self.fun, self.poly.gen)

    @property
    def free_symbols(self):
        return self.poly.free_symbols | self.fun.free_symbols

    @property
    def is_commutative(self):
        return True

    def doit(self, **hints):
        if not hints.get('roots', True):
            return self

        _roots = roots(self.poly, multiple=True)

        if len(_roots) < self.poly.degree():
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_evalf(self, prec):
        try:
            _roots = self.poly.nroots(n=prec_to_dps(prec))
        except (DomainError, PolynomialError):
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_derivative(self, x):
        var, expr = self.fun.args
        func = Lambda(var, expr.diff(x))
        return self.new(self.poly, func, self.auto)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\locators.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012-2023 Vinay Sajip.
# Licensed to the Python Software Foundation under a contributor agreement.
# See LICENSE.txt and CONTRIBUTORS.txt.
#

import gzip
from io import BytesIO
import json
import logging
import os
import posixpath
import re
try:
    import threading
except ImportError:  # pragma: no cover
    import dummy_threading as threading
import zlib

from . import DistlibException
from .compat import (urljoin, urlparse, urlunparse, url2pathname, pathname2url, queue, quote, unescape, build_opener,
                     HTTPRedirectHandler as BaseRedirectHandler, text_type, Request, HTTPError, URLError)
from .database import Distribution, DistributionPath, make_dist
from .metadata import Metadata, MetadataInvalidError
from .util import (cached_property, ensure_slash, split_filename, get_project_data, parse_requirement,
                   parse_name_and_version, ServerProxy, normalize_name)
from .version import get_scheme, UnsupportedVersionError
from .wheel import Wheel, is_compatible

logger = logging.getLogger(__name__)

HASHER_HASH = re.compile(r'^(\w+)=([a-f0-9]+)')
CHARSET = re.compile(r';\s*charset\s*=\s*(.*)\s*$', re.I)
HTML_CONTENT_TYPE = re.compile('text/html|application/x(ht)?ml')
DEFAULT_INDEX = 'https://pypi.org/pypi'


def get_all_distribution_names(url=None):
    """
    Return all distribution names known by an index.
    :param url: The URL of the index.
    :return: A list of all known distribution names.
    """
    if url is None:
        url = DEFAULT_INDEX
    client = ServerProxy(url, timeout=3.0)
    try:
        return client.list_packages()
    finally:
        client('close')()


class RedirectHandler(BaseRedirectHandler):
    """
    A class to work around a bug in some Python 3.2.x releases.
    """

    # There's a bug in the base version for some 3.2.x
    # (e.g. 3.2.2 on Ubuntu Oneiric). If a Location header
    # returns e.g. /abc, it bails because it says the scheme ''
    # is bogus, when actually it should use the request's
    # URL for the scheme. See Python issue #13696.
    def http_error_302(self, req, fp, code, msg, headers):
        # Some servers (incorrectly) return multiple Location headers
        # (so probably same goes for URI).  Use first header.
        newurl = None
        for key in ('location', 'uri'):
            if key in headers:
                newurl = headers[key]
                break
        if newurl is None:  # pragma: no cover
            return
        urlparts = urlparse(newurl)
        if urlparts.scheme == '':
            newurl = urljoin(req.get_full_url(), newurl)
            if hasattr(headers, 'replace_header'):
                headers.replace_header(key, newurl)
            else:
                headers[key] = newurl
        return BaseRedirectHandler.http_error_302(self, req, fp, code, msg, headers)

    http_error_301 = http_error_303 = http_error_307 = http_error_302


class Locator(object):
    """
    A base class for locators - things that locate distributions.
    """
    source_extensions = ('.tar.gz', '.tar.bz2', '.tar', '.zip', '.tgz', '.tbz')
    binary_extensions = ('.egg', '.exe', '.whl')
    excluded_extensions = ('.pdf', )

    # A list of tags indicating which wheels you want to match. The default
    # value of None matches against the tags compatible with the running
    # Python. If you want to match other values, set wheel_tags on a locator
    # instance to a list of tuples (pyver, abi, arch) which you want to match.
    wheel_tags = None

    downloadable_extensions = source_extensions + ('.whl', )

    def __init__(self, scheme='default'):
        """
        Initialise an instance.
        :param scheme: Because locators look for most recent versions, they
                       need to know the version scheme to use. This specifies
                       the current PEP-recommended scheme - use ``'legacy'``
                       if you need to support existing distributions on PyPI.
        """
        self._cache = {}
        self.scheme = scheme
        # Because of bugs in some of the handlers on some of the platforms,
        # we use our own opener rather than just using urlopen.
        self.opener = build_opener(RedirectHandler())
        # If get_project() is called from locate(), the matcher instance
        # is set from the requirement passed to locate(). See issue #18 for
        # why this can be useful to know.
        self.matcher = None
        self.errors = queue.Queue()

    def get_errors(self):
        """
        Return any errors which have occurred.
        """
        result = []
        while not self.errors.empty():  # pragma: no cover
            try:
                e = self.errors.get(False)
                result.append(e)
            except self.errors.Empty:
                continue
            self.errors.task_done()
        return result

    def clear_errors(self):
        """
        Clear any errors which may have been logged.
        """
        # Just get the errors and throw them away
        self.get_errors()

    def clear_cache(self):
        self._cache.clear()

    def _get_scheme(self):
        return self._scheme

    def _set_scheme(self, value):
        self._scheme = value

    scheme = property(_get_scheme, _set_scheme)

    def _get_project(self, name):
        """
        For a given project, get a dictionary mapping available versions to Distribution
        instances.

        This should be implemented in subclasses.

        If called from a locate() request, self.matcher will be set to a
        matcher for the requirement to satisfy, otherwise it will be None.
        """
        raise NotImplementedError('Please implement in the subclass')

    def get_distribution_names(self):
        """
        Return all the distribution names known to this locator.
        """
        raise NotImplementedError('Please implement in the subclass')

    def get_project(self, name):
        """
        For a given project, get a dictionary mapping available versions to Distribution
        instances.

        This calls _get_project to do all the work, and just implements a caching layer on top.
        """
        if self._cache is None:  # pragma: no cover
            result = self._get_project(name)
        elif name in self._cache:
            result = self._cache[name]
        else:
            self.clear_errors()
            result = self._get_project(name)
            self._cache[name] = result
        return result

    def score_url(self, url):
        """
        Give an url a score which can be used to choose preferred URLs
        for a given project release.
        """
        t = urlparse(url)
        basename = posixpath.basename(t.path)
        compatible = True
        is_wheel = basename.endswith('.whl')
        is_downloadable = basename.endswith(self.downloadable_extensions)
        if is_wheel:
            compatible = is_compatible(Wheel(basename), self.wheel_tags)
        return (t.scheme == 'https', 'pypi.org' in t.netloc, is_downloadable, is_wheel, compatible, basename)

    def prefer_url(self, url1, url2):
        """
        Choose one of two URLs where both are candidates for distribution
        archives for the same version of a distribution (for example,
        .tar.gz vs. zip).

        The current implementation favours https:// URLs over http://, archives
        from PyPI over those from other locations, wheel compatibility (if a
        wheel) and then the archive name.
        """
        result = url2
        if url1:
            s1 = self.score_url(url1)
            s2 = self.score_url(url2)
            if s1 > s2:
                result = url1
            if result != url2:
                logger.debug('Not replacing %r with %r', url1, url2)
            else:
                logger.debug('Replacing %r with %r', url1, url2)
        return result

    def split_filename(self, filename, project_name):
        """
        Attempt to split a filename in project name, version and Python version.
        """
        return split_filename(filename, project_name)

    def convert_url_to_download_info(self, url, project_name):
        """
        See if a URL is a candidate for a download URL for a project (the URL
        has typically been scraped from an HTML page).

        If it is, a dictionary is returned with keys "name", "version",
        "filename" and "url"; otherwise, None is returned.
        """

        def same_project(name1, name2):
            return normalize_name(name1) == normalize_name(name2)

        result = None
        scheme, netloc, path, params, query, frag = urlparse(url)
        if frag.lower().startswith('egg='):  # pragma: no cover
            logger.debug('%s: version hint in fragment: %r', project_name, frag)
        m = HASHER_HASH.match(frag)
        if m:
            algo, digest = m.groups()
        else:
            algo, digest = None, None
        origpath = path
        if path and path[-1] == '/':  # pragma: no cover
            path = path[:-1]
        if path.endswith('.whl'):
            try:
                wheel = Wheel(path)
                if not is_compatible(wheel, self.wheel_tags):
                    logger.debug('Wheel not compatible: %s', path)
                else:
                    if project_name is None:
                        include = True
                    else:
                        include = same_project(wheel.name, project_name)
                    if include:
                        result = {
                            'name': wheel.name,
                            'version': wheel.version,
                            'filename': wheel.filename,
                            'url': urlunparse((scheme, netloc, origpath, params, query, '')),
                            'python-version': ', '.join(['.'.join(list(v[2:])) for v in wheel.pyver]),
                        }
            except Exception:  # pragma: no cover
                logger.warning('invalid path for wheel: %s', path)
        elif not path.endswith(self.downloadable_extensions):  # pragma: no cover
            logger.debug('Not downloadable: %s', path)
        else:  # downloadable extension
            path = filename = posixpath.basename(path)
            for ext in self.downloadable_extensions:
                if path.endswith(ext):
                    path = path[:-len(ext)]
                    t = self.split_filename(path, project_name)
                    if not t:  # pragma: no cover
                        logger.debug('No match for project/version: %s', path)
                    else:
                        name, version, pyver = t
                        if not project_name or same_project(project_name, name):
                            result = {
                                'name': name,
                                'version': version,
                                'filename': filename,
                                'url': urlunparse((scheme, netloc, origpath, params, query, '')),
                            }
                            if pyver:  # pragma: no cover
                                result['python-version'] = pyver
                    break
        if result and algo:
            result['%s_digest' % algo] = digest
        return result

    def _get_digest(self, info):
        """
        Get a digest from a dictionary by looking at a "digests" dictionary
        or keys of the form 'algo_digest'.

        Returns a 2-tuple (algo, digest) if found, else None. Currently
        looks only for SHA256, then MD5.
        """
        result = None
        if 'digests' in info:
            digests = info['digests']
            for algo in ('sha256', 'md5'):
                if algo in digests:
                    result = (algo, digests[algo])
                    break
        if not result:
            for algo in ('sha256', 'md5'):
                key = '%s_digest' % algo
                if key in info:
                    result = (algo, info[key])
                    break
        return result

    def _update_version_data(self, result, info):
        """
        Update a result dictionary (the final result from _get_project) with a
        dictionary for a specific version, which typically holds information
        gleaned from a filename or URL for an archive for the distribution.
        """
        name = info.pop('name')
        version = info.pop('version')
        if version in result:
            dist = result[version]
            md = dist.metadata
        else:
            dist = make_dist(name, version, scheme=self.scheme)
            md = dist.metadata
        dist.digest = digest = self._get_digest(info)
        url = info['url']
        result['digests'][url] = digest
        if md.source_url != info['url']:
            md.source_url = self.prefer_url(md.source_url, url)
            result['urls'].setdefault(version, set()).add(url)
        dist.locator = self
        result[version] = dist

    def locate(self, requirement, prereleases=False):
        """
        Find the most recent distribution which matches the given
        requirement.

        :param requirement: A requirement of the form 'foo (1.0)' or perhaps
                            'foo (>= 1.0, < 2.0, != 1.3)'
        :param prereleases: If ``True``, allow pre-release versions
                            to be located. Otherwise, pre-release versions
                            are not returned.
        :return: A :class:`Distribution` instance, or ``None`` if no such
                 distribution could be located.
        """
        result = None
        r = parse_requirement(requirement)
        if r is None:  # pragma: no cover
            raise DistlibException('Not a valid requirement: %r' % requirement)
        scheme = get_scheme(self.scheme)
        self.matcher = matcher = scheme.matcher(r.requirement)
        logger.debug('matcher: %s (%s)', matcher, type(matcher).__name__)
        versions = self.get_project(r.name)
        if len(versions) > 2:  # urls and digests keys are present
            # sometimes, versions are invalid
            slist = []
            vcls = matcher.version_class
            for k in versions:
                if k in ('urls', 'digests'):
                    continue
                try:
                    if not matcher.match(k):
                        pass  # logger.debug('%s did not match %r', matcher, k)
                    else:
                        if prereleases or not vcls(k).is_prerelease:
                            slist.append(k)
                except Exception:  # pragma: no cover
                    logger.warning('error matching %s with %r', matcher, k)
                    pass  # slist.append(k)
            if len(slist) > 1:
                slist = sorted(slist, key=scheme.key)
            if slist:
                logger.debug('sorted list: %s', slist)
                version = slist[-1]
                result = versions[version]
        if result:
            if r.extras:
                result.extras = r.extras
            result.download_urls = versions.get('urls', {}).get(version, set())
            d = {}
            sd = versions.get('digests', {})
            for url in result.download_urls:
                if url in sd:  # pragma: no cover
                    d[url] = sd[url]
            result.digests = d
        self.matcher = None
        return result


class PyPIRPCLocator(Locator):
    """
    This locator uses XML-RPC to locate distributions. It therefore
    cannot be used with simple mirrors (that only mirror file content).
    """

    def __init__(self, url, **kwargs):
        """
        Initialise an instance.

        :param url: The URL to use for XML-RPC.
        :param kwargs: Passed to the superclass constructor.
        """
        super(PyPIRPCLocator, self).__init__(**kwargs)
        self.base_url = url
        self.client = ServerProxy(url, timeout=3.0)

    def get_distribution_names(self):
        """
        Return all the distribution names known to this locator.
        """
        return set(self.client.list_packages())

    def _get_project(self, name):
        result = {'urls': {}, 'digests': {}}
        versions = self.client.package_releases(name, True)
        for v in versions:
            urls = self.client.release_urls(name, v)
            data = self.client.release_data(name, v)
            metadata = Metadata(scheme=self.scheme)
            metadata.name = data['name']
            metadata.version = data['version']
            metadata.license = data.get('license')
            metadata.keywords = data.get('keywords', [])
            metadata.summary = data.get('summary')
            dist = Distribution(metadata)
            if urls:
                info = urls[0]
                metadata.source_url = info['url']
                dist.digest = self._get_digest(info)
                dist.locator = self
                result[v] = dist
                for info in urls:
                    url = info['url']
                    digest = self._get_digest(info)
                    result['urls'].setdefault(v, set()).add(url)
                    result['digests'][url] = digest
        return result


class PyPIJSONLocator(Locator):
    """
    This locator uses PyPI's JSON interface. It's very limited in functionality
    and probably not worth using.
    """

    def __init__(self, url, **kwargs):
        super(PyPIJSONLocator, self).__init__(**kwargs)
        self.base_url = ensure_slash(url)

    def get_distribution_names(self):
        """
        Return all the distribution names known to this locator.
        """
        raise NotImplementedError('Not available from this locator')

    def _get_project(self, name):
        result = {'urls': {}, 'digests': {}}
        url = urljoin(self.base_url, '%s/json' % quote(name))
        try:
            resp = self.opener.open(url)
            data = resp.read().decode()  # for now
            d = json.loads(data)
            md = Metadata(scheme=self.scheme)
            data = d['info']
            md.name = data['name']
            md.version = data['version']
            md.license = data.get('license')
            md.keywords = data.get('keywords', [])
            md.summary = data.get('summary')
            dist = Distribution(md)
            dist.locator = self
            # urls = d['urls']
            result[md.version] = dist
            for info in d['urls']:
                url = info['url']
                dist.download_urls.add(url)
                dist.digests[url] = self._get_digest(info)
                result['urls'].setdefault(md.version, set()).add(url)
                result['digests'][url] = self._get_digest(info)
            # Now get other releases
            for version, infos in d['releases'].items():
                if version == md.version:
                    continue  # already done
                omd = Metadata(scheme=self.scheme)
                omd.name = md.name
                omd.version = version
                odist = Distribution(omd)
                odist.locator = self
                result[version] = odist
                for info in infos:
                    url = info['url']
                    odist.download_urls.add(url)
                    odist.digests[url] = self._get_digest(info)
                    result['urls'].setdefault(version, set()).add(url)
                    result['digests'][url] = self._get_digest(info)


#            for info in urls:
#                md.source_url = info['url']
#                dist.digest = self._get_digest(info)
#                dist.locator = self
#                for info in urls:
#                    url = info['url']
#                    result['urls'].setdefault(md.version, set()).add(url)
#                    result['digests'][url] = self._get_digest(info)
        except Exception as e:
            self.errors.put(text_type(e))
            logger.exception('JSON fetch failed: %s', e)
        return result


class Page(object):
    """
    This class represents a scraped HTML page.
    """
    # The following slightly hairy-looking regex just looks for the contents of
    # an anchor link, which has an attribute "href" either immediately preceded
    # or immediately followed by a "rel" attribute. The attribute values can be
    # declared with double quotes, single quotes or no quotes - which leads to
    # the length of the expression.
    _href = re.compile(
        """
(rel\\s*=\\s*(?:"(?P<rel1>[^"]*)"|'(?P<rel2>[^']*)'|(?P<rel3>[^>\\s\n]*))\\s+)?
href\\s*=\\s*(?:"(?P<url1>[^"]*)"|'(?P<url2>[^']*)'|(?P<url3>[^>\\s\n]*))
(\\s+rel\\s*=\\s*(?:"(?P<rel4>[^"]*)"|'(?P<rel5>[^']*)'|(?P<rel6>[^>\\s\n]*)))?
""", re.I | re.S | re.X)
    _base = re.compile(r"""<base\s+href\s*=\s*['"]?([^'">]+)""", re.I | re.S)

    def __init__(self, data, url):
        """
        Initialise an instance with the Unicode page contents and the URL they
        came from.
        """
        self.data = data
        self.base_url = self.url = url
        m = self._base.search(self.data)
        if m:
            self.base_url = m.group(1)

    _clean_re = re.compile(r'[^a-z0-9$&+,/:;=?@.#%_\\|-]', re.I)

    @cached_property
    def links(self):
        """
        Return the URLs of all the links on a page together with information
        about their "rel" attribute, for determining which ones to treat as
        downloads and which ones to queue for further scraping.
        """

        def clean(url):
            "Tidy up an URL."
            scheme, netloc, path, params, query, frag = urlparse(url)
            return urlunparse((scheme, netloc, quote(path), params, query, frag))

        result = set()
        for match in self._href.finditer(self.data):
            d = match.groupdict('')
            rel = (d['rel1'] or d['rel2'] or d['rel3'] or d['rel4'] or d['rel5'] or d['rel6'])
            url = d['url1'] or d['url2'] or d['url3']
            url = urljoin(self.base_url, url)
            url = unescape(url)
            url = self._clean_re.sub(lambda m: '%%%2x' % ord(m.group(0)), url)
            result.add((url, rel))
        # We sort the result, hoping to bring the most recent versions
        # to the front
        result = sorted(result, key=lambda t: t[0], reverse=True)
        return result


class SimpleScrapingLocator(Locator):
    """
    A locator which scrapes HTML pages to locate downloads for a distribution.
    This runs multiple threads to do the I/O; performance is at least as good
    as pip's PackageFinder, which works in an analogous fashion.
    """

    # These are used to deal with various Content-Encoding schemes.
    decoders = {
        'deflate': zlib.decompress,
        'gzip': lambda b: gzip.GzipFile(fileobj=BytesIO(b)).read(),
        'none': lambda b: b,
    }

    def __init__(self, url, timeout=None, num_workers=10, **kwargs):
        """
        Initialise an instance.
        :param url: The root URL to use for scraping.
        :param timeout: The timeout, in seconds, to be applied to requests.
                        This defaults to ``None`` (no timeout specified).
        :param num_workers: The number of worker threads you want to do I/O,
                            This defaults to 10.
        :param kwargs: Passed to the superclass.
        """
        super(SimpleScrapingLocator, self).__init__(**kwargs)
        self.base_url = ensure_slash(url)
        self.timeout = timeout
        self._page_cache = {}
        self._seen = set()
        self._to_fetch = queue.Queue()
        self._bad_hosts = set()
        self.skip_externals = False
        self.num_workers = num_workers
        self._lock = threading.RLock()
        # See issue #45: we need to be resilient when the locator is used
        # in a thread, e.g. with concurrent.futures. We can't use self._lock
        # as it is for coordinating our internal threads - the ones created
        # in _prepare_threads.
        self._gplock = threading.RLock()
        self.platform_check = False  # See issue #112

    def _prepare_threads(self):
        """
        Threads are created only when get_project is called, and terminate
        before it returns. They are there primarily to parallelise I/O (i.e.
        fetching web pages).
        """
        self._threads = []
        for i in range(self.num_workers):
            t = threading.Thread(target=self._fetch)
            t.daemon = True
            t.start()
            self._threads.append(t)

    def _wait_threads(self):
        """
        Tell all the threads to terminate (by sending a sentinel value) and
        wait for them to do so.
        """
        # Note that you need two loops, since you can't say which
        # thread will get each sentinel
        for t in self._threads:
            self._to_fetch.put(None)  # sentinel
        for t in self._threads:
            t.join()
        self._threads = []

    def _get_project(self, name):
        result = {'urls': {}, 'digests': {}}
        with self._gplock:
            self.result = result
            self.project_name = name
            url = urljoin(self.base_url, '%s/' % quote(name))
            self._seen.clear()
            self._page_cache.clear()
            self._prepare_threads()
            try:
                logger.debug('Queueing %s', url)
                self._to_fetch.put(url)
                self._to_fetch.join()
            finally:
                self._wait_threads()
            del self.result
        return result

    platform_dependent = re.compile(r'\b(linux_(i\d86|x86_64|arm\w+)|'
                                    r'win(32|_amd64)|macosx_?\d+)\b', re.I)

    def _is_platform_dependent(self, url):
        """
        Does an URL refer to a platform-specific download?
        """
        return self.platform_dependent.search(url)

    def _process_download(self, url):
        """
        See if an URL is a suitable download for a project.

        If it is, register information in the result dictionary (for
        _get_project) about the specific version it's for.

        Note that the return value isn't actually used other than as a boolean
        value.
        """
        if self.platform_check and self._is_platform_dependent(url):
            info = None
        else:
            info = self.convert_url_to_download_info(url, self.project_name)
        logger.debug('process_download: %s -> %s', url, info)
        if info:
            with self._lock:  # needed because self.result is shared
                self._update_version_data(self.result, info)
        return info

    def _should_queue(self, link, referrer, rel):
        """
        Determine whether a link URL from a referring page and with a
        particular "rel" attribute should be queued for scraping.
        """
        scheme, netloc, path, _, _, _ = urlparse(link)
        if path.endswith(self.source_extensions + self.binary_extensions + self.excluded_extensions):
            result = False
        elif self.skip_externals and not link.startswith(self.base_url):
            result = False
        elif not referrer.startswith(self.base_url):
            result = False
        elif rel not in ('homepage', 'download'):
            result = False
        elif scheme not in ('http', 'https', 'ftp'):
            result = False
        elif self._is_platform_dependent(link):
            result = False
        else:
            host = netloc.split(':', 1)[0]
            if host.lower() == 'localhost':
                result = False
            else:
                result = True
        logger.debug('should_queue: %s (%s) from %s -> %s', link, rel, referrer, result)
        return result

    def _fetch(self):
        """
        Get a URL to fetch from the work queue, get the HTML page, examine its
        links for download candidates and candidates for further scraping.

        This is a handy method to run in a thread.
        """
        while True:
            url = self._to_fetch.get()
            try:
                if url:
                    page = self.get_page(url)
                    if page is None:  # e.g. after an error
                        continue
                    for link, rel in page.links:
                        if link not in self._seen:
                            try:
                                self._seen.add(link)
                                if (not self._process_download(link) and self._should_queue(link, url, rel)):
                                    logger.debug('Queueing %s from %s', link, url)
                                    self._to_fetch.put(link)
                            except MetadataInvalidError:  # e.g. invalid versions
                                pass
            except Exception as e:  # pragma: no cover
                self.errors.put(text_type(e))
            finally:
                # always do this, to avoid hangs :-)
                self._to_fetch.task_done()
            if not url:
                # logger.debug('Sentinel seen, quitting.')
                break

    def get_page(self, url):
        """
        Get the HTML for an URL, possibly from an in-memory cache.

        XXX TODO Note: this cache is never actually cleared. It's assumed that
        the data won't get stale over the lifetime of a locator instance (not
        necessarily true for the default_locator).
        """
        # http://peak.telecommunity.com/DevCenter/EasyInstall#package-index-api
        scheme, netloc, path, _, _, _ = urlparse(url)
        if scheme == 'file' and os.path.isdir(url2pathname(path)):
            url = urljoin(ensure_slash(url), 'index.html')

        if url in self._page_cache:
            result = self._page_cache[url]
            logger.debug('Returning %s from cache: %s', url, result)
        else:
            host = netloc.split(':', 1)[0]
            result = None
            if host in self._bad_hosts:
                logger.debug('Skipping %s due to bad host %s', url, host)
            else:
                req = Request(url, headers={'Accept-encoding': 'identity'})
                try:
                    logger.debug('Fetching %s', url)
                    resp = self.opener.open(req, timeout=self.timeout)
                    logger.debug('Fetched %s', url)
                    headers = resp.info()
                    content_type = headers.get('Content-Type', '')
                    if HTML_CONTENT_TYPE.match(content_type):
                        final_url = resp.geturl()
                        data = resp.read()
                        encoding = headers.get('Content-Encoding')
                        if encoding:
                            decoder = self.decoders[encoding]  # fail if not found
                            data = decoder(data)
                        encoding = 'utf-8'
                        m = CHARSET.search(content_type)
                        if m:
                            encoding = m.group(1)
                        try:
                            data = data.decode(encoding)
                        except UnicodeError:  # pragma: no cover
                            data = data.decode('latin-1')  # fallback
                        result = Page(data, final_url)
                        self._page_cache[final_url] = result
                except HTTPError as e:
                    if e.code != 404:
                        logger.exception('Fetch failed: %s: %s', url, e)
                except URLError as e:  # pragma: no cover
                    logger.exception('Fetch failed: %s: %s', url, e)
                    with self._lock:
                        self._bad_hosts.add(host)
                except Exception as e:  # pragma: no cover
                    logger.exception('Fetch failed: %s: %s', url, e)
                finally:
                    self._page_cache[url] = result  # even if None (failure)
        return result

    _distname_re = re.compile('<a href=[^>]*>([^<]+)<')

    def get_distribution_names(self):
        """
        Return all the distribution names known to this locator.
        """
        result = set()
        page = self.get_page(self.base_url)
        if not page:
            raise DistlibException('Unable to get %s' % self.base_url)
        for match in self._distname_re.finditer(page.data):
            result.add(match.group(1))
        return result


class DirectoryLocator(Locator):
    """
    This class locates distributions in a directory tree.
    """

    def __init__(self, path, **kwargs):
        """
        Initialise an instance.
        :param path: The root of the directory tree to search.
        :param kwargs: Passed to the superclass constructor,
                       except for:
                       * recursive - if True (the default), subdirectories are
                         recursed into. If False, only the top-level directory
                         is searched,
        """
        self.recursive = kwargs.pop('recursive', True)
        super(DirectoryLocator, self).__init__(**kwargs)
        path = os.path.abspath(path)
        if not os.path.isdir(path):  # pragma: no cover
            raise DistlibException('Not a directory: %r' % path)
        self.base_dir = path

    def should_include(self, filename, parent):
        """
        Should a filename be considered as a candidate for a distribution
        archive? As well as the filename, the directory which contains it
        is provided, though not used by the current implementation.
        """
        return filename.endswith(self.downloadable_extensions)

    def _get_project(self, name):
        result = {'urls': {}, 'digests': {}}
        for root, dirs, files in os.walk(self.base_dir):
            for fn in files:
                if self.should_include(fn, root):
                    fn = os.path.join(root, fn)
                    url = urlunparse(('file', '', pathname2url(os.path.abspath(fn)), '', '', ''))
                    info = self.convert_url_to_download_info(url, name)
                    if info:
                        self._update_version_data(result, info)
            if not self.recursive:
                break
        return result

    def get_distribution_names(self):
        """
        Return all the distribution names known to this locator.
        """
        result = set()
        for root, dirs, files in os.walk(self.base_dir):
            for fn in files:
                if self.should_include(fn, root):
                    fn = os.path.join(root, fn)
                    url = urlunparse(('file', '', pathname2url(os.path.abspath(fn)), '', '', ''))
                    info = self.convert_url_to_download_info(url, None)
                    if info:
                        result.add(info['name'])
            if not self.recursive:
                break
        return result


class JSONLocator(Locator):
    """
    This locator uses special extended metadata (not available on PyPI) and is
    the basis of performant dependency resolution in distlib. Other locators
    require archive downloads before dependencies can be determined! As you
    might imagine, that can be slow.
    """

    def get_distribution_names(self):
        """
        Return all the distribution names known to this locator.
        """
        raise NotImplementedError('Not available from this locator')

    def _get_project(self, name):
        result = {'urls': {}, 'digests': {}}
        data = get_project_data(name)
        if data:
            for info in data.get('files', []):
                if info['ptype'] != 'sdist' or info['pyversion'] != 'source':
                    continue
                # We don't store summary in project metadata as it makes
                # the data bigger for no benefit during dependency
                # resolution
                dist = make_dist(data['name'],
                                 info['version'],
                                 summary=data.get('summary', 'Placeholder for summary'),
                                 scheme=self.scheme)
                md = dist.metadata
                md.source_url = info['url']
                # TODO SHA256 digest
                if 'digest' in info and info['digest']:
                    dist.digest = ('md5', info['digest'])
                md.dependencies = info.get('requirements', {})
                dist.exports = info.get('exports', {})
                result[dist.version] = dist
                result['urls'].setdefault(dist.version, set()).add(info['url'])
        return result


class DistPathLocator(Locator):
    """
    This locator finds installed distributions in a path. It can be useful for
    adding to an :class:`AggregatingLocator`.
    """

    def __init__(self, distpath, **kwargs):
        """
        Initialise an instance.

        :param distpath: A :class:`DistributionPath` instance to search.
        """
        super(DistPathLocator, self).__init__(**kwargs)
        assert isinstance(distpath, DistributionPath)
        self.distpath = distpath

    def _get_project(self, name):
        dist = self.distpath.get_distribution(name)
        if dist is None:
            result = {'urls': {}, 'digests': {}}
        else:
            result = {
                dist.version: dist,
                'urls': {
                    dist.version: set([dist.source_url])
                },
                'digests': {
                    dist.version: set([None])
                }
            }
        return result


class AggregatingLocator(Locator):
    """
    This class allows you to chain and/or merge a list of locators.
    """

    def __init__(self, *locators, **kwargs):
        """
        Initialise an instance.

        :param locators: The list of locators to search.
        :param kwargs: Passed to the superclass constructor,
                       except for:
                       * merge - if False (the default), the first successful
                         search from any of the locators is returned. If True,
                         the results from all locators are merged (this can be
                         slow).
        """
        self.merge = kwargs.pop('merge', False)
        self.locators = locators
        super(AggregatingLocator, self).__init__(**kwargs)

    def clear_cache(self):
        super(AggregatingLocator, self).clear_cache()
        for locator in self.locators:
            locator.clear_cache()

    def _set_scheme(self, value):
        self._scheme = value
        for locator in self.locators:
            locator.scheme = value

    scheme = property(Locator.scheme.fget, _set_scheme)

    def _get_project(self, name):
        result = {}
        for locator in self.locators:
            d = locator.get_project(name)
            if d:
                if self.merge:
                    files = result.get('urls', {})
                    digests = result.get('digests', {})
                    # next line could overwrite result['urls'], result['digests']
                    result.update(d)
                    df = result.get('urls')
                    if files and df:
                        for k, v in files.items():
                            if k in df:
                                df[k] |= v
                            else:
                                df[k] = v
                    dd = result.get('digests')
                    if digests and dd:
                        dd.update(digests)
                else:
                    # See issue #18. If any dists are found and we're looking
                    # for specific constraints, we only return something if
                    # a match is found. For example, if a DirectoryLocator
                    # returns just foo (1.0) while we're looking for
                    # foo (>= 2.0), we'll pretend there was nothing there so
                    # that subsequent locators can be queried. Otherwise we
                    # would just return foo (1.0) which would then lead to a
                    # failure to find foo (>= 2.0), because other locators
                    # weren't searched. Note that this only matters when
                    # merge=False.
                    if self.matcher is None:
                        found = True
                    else:
                        found = False
                        for k in d:
                            if self.matcher.match(k):
                                found = True
                                break
                    if found:
                        result = d
                        break
        return result

    def get_distribution_names(self):
        """
        Return all the distribution names known to this locator.
        """
        result = set()
        for locator in self.locators:
            try:
                result |= locator.get_distribution_names()
            except NotImplementedError:
                pass
        return result


# We use a legacy scheme simply because most of the dists on PyPI use legacy
# versions which don't conform to PEP 440.
default_locator = AggregatingLocator(
    # JSONLocator(), # don't use as PEP 426 is withdrawn
    SimpleScrapingLocator('https://pypi.org/simple/', timeout=3.0),
    scheme='legacy')

locate = default_locator.locate


class DependencyFinder(object):
    """
    Locate dependencies for distributions.
    """

    def __init__(self, locator=None):
        """
        Initialise an instance, using the specified locator
        to locate distributions.
        """
        self.locator = locator or default_locator
        self.scheme = get_scheme(self.locator.scheme)

    def add_distribution(self, dist):
        """
        Add a distribution to the finder. This will update internal information
        about who provides what.
        :param dist: The distribution to add.
        """
        logger.debug('adding distribution %s', dist)
        name = dist.key
        self.dists_by_name[name] = dist
        self.dists[(name, dist.version)] = dist
        for p in dist.provides:
            name, version = parse_name_and_version(p)
            logger.debug('Add to provided: %s, %s, %s', name, version, dist)
            self.provided.setdefault(name, set()).add((version, dist))

    def remove_distribution(self, dist):
        """
        Remove a distribution from the finder. This will update internal
        information about who provides what.
        :param dist: The distribution to remove.
        """
        logger.debug('removing distribution %s', dist)
        name = dist.key
        del self.dists_by_name[name]
        del self.dists[(name, dist.version)]
        for p in dist.provides:
            name, version = parse_name_and_version(p)
            logger.debug('Remove from provided: %s, %s, %s', name, version, dist)
            s = self.provided[name]
            s.remove((version, dist))
            if not s:
                del self.provided[name]

    def get_matcher(self, reqt):
        """
        Get a version matcher for a requirement.
        :param reqt: The requirement
        :type reqt: str
        :return: A version matcher (an instance of
                 :class:`distlib.version.Matcher`).
        """
        try:
            matcher = self.scheme.matcher(reqt)
        except UnsupportedVersionError:  # pragma: no cover
            # XXX compat-mode if cannot read the version
            name = reqt.split()[0]
            matcher = self.scheme.matcher(name)
        return matcher

    def find_providers(self, reqt):
        """
        Find the distributions which can fulfill a requirement.

        :param reqt: The requirement.
         :type reqt: str
        :return: A set of distribution which can fulfill the requirement.
        """
        matcher = self.get_matcher(reqt)
        name = matcher.key  # case-insensitive
        result = set()
        provided = self.provided
        if name in provided:
            for version, provider in provided[name]:
                try:
                    match = matcher.match(version)
                except UnsupportedVersionError:
                    match = False

                if match:
                    result.add(provider)
                    break
        return result

    def try_to_replace(self, provider, other, problems):
        """
        Attempt to replace one provider with another. This is typically used
        when resolving dependencies from multiple sources, e.g. A requires
        (B >= 1.0) while C requires (B >= 1.1).

        For successful replacement, ``provider`` must meet all the requirements
        which ``other`` fulfills.

        :param provider: The provider we are trying to replace with.
        :param other: The provider we're trying to replace.
        :param problems: If False is returned, this will contain what
                         problems prevented replacement. This is currently
                         a tuple of the literal string 'cantreplace',
                         ``provider``, ``other``  and the set of requirements
                         that ``provider`` couldn't fulfill.
        :return: True if we can replace ``other`` with ``provider``, else
                 False.
        """
        rlist = self.reqts[other]
        unmatched = set()
        for s in rlist:
            matcher = self.get_matcher(s)
            if not matcher.match(provider.version):
                unmatched.add(s)
        if unmatched:
            # can't replace other with provider
            problems.add(('cantreplace', provider, other, frozenset(unmatched)))
            result = False
        else:
            # can replace other with provider
            self.remove_distribution(other)
            del self.reqts[other]
            for s in rlist:
                self.reqts.setdefault(provider, set()).add(s)
            self.add_distribution(provider)
            result = True
        return result

    def find(self, requirement, meta_extras=None, prereleases=False):
        """
        Find a distribution and all distributions it depends on.

        :param requirement: The requirement specifying the distribution to
                            find, or a Distribution instance.
        :param meta_extras: A list of meta extras such as :test:, :build: and
                            so on.
        :param prereleases: If ``True``, allow pre-release versions to be
                            returned - otherwise, don't return prereleases
                            unless they're all that's available.

        Return a set of :class:`Distribution` instances and a set of
        problems.

        The distributions returned should be such that they have the
        :attr:`required` attribute set to ``True`` if they were
        from the ``requirement`` passed to ``find()``, and they have the
        :attr:`build_time_dependency` attribute set to ``True`` unless they
        are post-installation dependencies of the ``requirement``.

        The problems should be a tuple consisting of the string
        ``'unsatisfied'`` and the requirement which couldn't be satisfied
        by any distribution known to the locator.
        """

        self.provided = {}
        self.dists = {}
        self.dists_by_name = {}
        self.reqts = {}

        meta_extras = set(meta_extras or [])
        if ':*:' in meta_extras:
            meta_extras.remove(':*:')
            # :meta: and :run: are implicitly included
            meta_extras |= set([':test:', ':build:', ':dev:'])

        if isinstance(requirement, Distribution):
            dist = odist = requirement
            logger.debug('passed %s as requirement', odist)
        else:
            dist = odist = self.locator.locate(requirement, prereleases=prereleases)
            if dist is None:
                raise DistlibException('Unable to locate %r' % requirement)
            logger.debug('located %s', odist)
        dist.requested = True
        problems = set()
        todo = set([dist])
        install_dists = set([odist])
        while todo:
            dist = todo.pop()
            name = dist.key  # case-insensitive
            if name not in self.dists_by_name:
                self.add_distribution(dist)
            else:
                # import pdb; pdb.set_trace()
                other = self.dists_by_name[name]
                if other != dist:
                    self.try_to_replace(dist, other, problems)

            ireqts = dist.run_requires | dist.meta_requires
            sreqts = dist.build_requires
            ereqts = set()
            if meta_extras and dist in install_dists:
                for key in ('test', 'build', 'dev'):
                    e = ':%s:' % key
                    if e in meta_extras:
                        ereqts |= getattr(dist, '%s_requires' % key)
            all_reqts = ireqts | sreqts | ereqts
            for r in all_reqts:
                providers = self.find_providers(r)
                if not providers:
                    logger.debug('No providers found for %r', r)
                    provider = self.locator.locate(r, prereleases=prereleases)
                    # If no provider is found and we didn't consider
                    # prereleases, consider them now.
                    if provider is None and not prereleases:
                        provider = self.locator.locate(r, prereleases=True)
                    if provider is None:
                        logger.debug('Cannot satisfy %r', r)
                        problems.add(('unsatisfied', r))
                    else:
                        n, v = provider.key, provider.version
                        if (n, v) not in self.dists:
                            todo.add(provider)
                        providers.add(provider)
                        if r in ireqts and dist in install_dists:
                            install_dists.add(provider)
                            logger.debug('Adding %s to install_dists', provider.name_and_version)
                for p in providers:
                    name = p.key
                    if name not in self.dists_by_name:
                        self.reqts.setdefault(p, set()).add(r)
                    else:
                        other = self.dists_by_name[name]
                        if other != p:
                            # see if other can be replaced by p
                            self.try_to_replace(p, other, problems)

        dists = set(self.dists.values())
        for dist in dists:
            dist.build_time_dependency = dist not in install_dists
            if dist.build_time_dependency:
                logger.debug('%s is a build-time dependency only.', dist.name_and_version)
        logger.debug('find done for %s', odist)
        return dists, problems

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\asyncpg.py ===
# dialects/postgresql/asyncpg.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors <see AUTHORS
# file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

r"""
.. dialect:: postgresql+asyncpg
    :name: asyncpg
    :dbapi: asyncpg
    :connectstring: postgresql+asyncpg://user:password@host:port/dbname[?key=value&key=value...]
    :url: https://magicstack.github.io/asyncpg/

The asyncpg dialect is SQLAlchemy's first Python asyncio dialect.

Using a special asyncio mediation layer, the asyncpg dialect is usable
as the backend for the :ref:`SQLAlchemy asyncio <asyncio_toplevel>`
extension package.

This dialect should normally be used only with the
:func:`_asyncio.create_async_engine` engine creation function::

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(
        "postgresql+asyncpg://user:pass@hostname/dbname"
    )

.. versionadded:: 1.4

.. note::

    By default asyncpg does not decode the ``json`` and ``jsonb`` types and
    returns them as strings. SQLAlchemy sets default type decoder for ``json``
    and ``jsonb`` types using the python builtin ``json.loads`` function.
    The json implementation used can be changed by setting the attribute
    ``json_deserializer`` when creating the engine with
    :func:`create_engine` or :func:`create_async_engine`.

.. _asyncpg_multihost:

Multihost Connections
--------------------------

The asyncpg dialect features support for multiple fallback hosts in the
same way as that of the psycopg2 and psycopg dialects.  The
syntax is the same,
using ``host=<host>:<port>`` combinations as additional query string arguments;
however, there is no default port, so all hosts must have a complete port number
present, otherwise an exception is raised::

    engine = create_async_engine(
        "postgresql+asyncpg://user:password@/dbname?host=HostA:5432&host=HostB:5432&host=HostC:5432"
    )

For complete background on this syntax, see :ref:`psycopg2_multi_host`.

.. versionadded:: 2.0.18

.. seealso::

    :ref:`psycopg2_multi_host`

.. _asyncpg_prepared_statement_cache:

Prepared Statement Cache
--------------------------

The asyncpg SQLAlchemy dialect makes use of ``asyncpg.connection.prepare()``
for all statements.   The prepared statement objects are cached after
construction which appears to grant a 10% or more performance improvement for
statement invocation.   The cache is on a per-DBAPI connection basis, which
means that the primary storage for prepared statements is within DBAPI
connections pooled within the connection pool.   The size of this cache
defaults to 100 statements per DBAPI connection and may be adjusted using the
``prepared_statement_cache_size`` DBAPI argument (note that while this argument
is implemented by SQLAlchemy, it is part of the DBAPI emulation portion of the
asyncpg dialect, therefore is handled as a DBAPI argument, not a dialect
argument)::


    engine = create_async_engine(
        "postgresql+asyncpg://user:pass@hostname/dbname?prepared_statement_cache_size=500"
    )

To disable the prepared statement cache, use a value of zero::

    engine = create_async_engine(
        "postgresql+asyncpg://user:pass@hostname/dbname?prepared_statement_cache_size=0"
    )

.. versionadded:: 1.4.0b2 Added ``prepared_statement_cache_size`` for asyncpg.


.. warning::  The ``asyncpg`` database driver necessarily uses caches for
   PostgreSQL type OIDs, which become stale when custom PostgreSQL datatypes
   such as ``ENUM`` objects are changed via DDL operations.   Additionally,
   prepared statements themselves which are optionally cached by SQLAlchemy's
   driver as described above may also become "stale" when DDL has been emitted
   to the PostgreSQL database which modifies the tables or other objects
   involved in a particular prepared statement.

   The SQLAlchemy asyncpg dialect will invalidate these caches within its local
   process when statements that represent DDL are emitted on a local
   connection, but this is only controllable within a single Python process /
   database engine.     If DDL changes are made from other database engines
   and/or processes, a running application may encounter asyncpg exceptions
   ``InvalidCachedStatementError`` and/or ``InternalServerError("cache lookup
   failed for type <oid>")`` if it refers to pooled database connections which
   operated upon the previous structures. The SQLAlchemy asyncpg dialect will
   recover from these error cases when the driver raises these exceptions by
   clearing its internal caches as well as those of the asyncpg driver in
   response to them, but cannot prevent them from being raised in the first
   place if the cached prepared statement or asyncpg type caches have gone
   stale, nor can it retry the statement as the PostgreSQL transaction is
   invalidated when these errors occur.

.. _asyncpg_prepared_statement_name:

Prepared Statement Name with PGBouncer
--------------------------------------

By default, asyncpg enumerates prepared statements in numeric order, which
can lead to errors if a name has already been taken for another prepared
statement. This issue can arise if your application uses database proxies
such as PgBouncer to handle connections. One possible workaround is to
use dynamic prepared statement names, which asyncpg now supports through
an optional ``name`` value for the statement name. This allows you to
generate your own unique names that won't conflict with existing ones.
To achieve this, you can provide a function that will be called every time
a prepared statement is prepared::

    from uuid import uuid4

    engine = create_async_engine(
        "postgresql+asyncpg://user:pass@somepgbouncer/dbname",
        poolclass=NullPool,
        connect_args={
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid4()}__",
        },
    )

.. seealso::

   https://github.com/MagicStack/asyncpg/issues/837

   https://github.com/sqlalchemy/sqlalchemy/issues/6467

.. warning:: When using PGBouncer, to prevent a buildup of useless prepared statements in
   your application, it's important to use the :class:`.NullPool` pool
   class, and to configure PgBouncer to use `DISCARD <https://www.postgresql.org/docs/current/sql-discard.html>`_
   when returning connections.  The DISCARD command is used to release resources held by the db connection,
   including prepared statements. Without proper setup, prepared statements can
   accumulate quickly and cause performance issues.

Disabling the PostgreSQL JIT to improve ENUM datatype handling
---------------------------------------------------------------

Asyncpg has an `issue <https://github.com/MagicStack/asyncpg/issues/727>`_ when
using PostgreSQL ENUM datatypes, where upon the creation of new database
connections, an expensive query may be emitted in order to retrieve metadata
regarding custom types which has been shown to negatively affect performance.
To mitigate this issue, the PostgreSQL "jit" setting may be disabled from the
client using this setting passed to :func:`_asyncio.create_async_engine`::

    engine = create_async_engine(
        "postgresql+asyncpg://user:password@localhost/tmp",
        connect_args={"server_settings": {"jit": "off"}},
    )

.. seealso::

    https://github.com/MagicStack/asyncpg/issues/727

"""  # noqa

from __future__ import annotations

from collections import deque
import decimal
import json as _py_json
import re
import time

from . import json
from . import ranges
from .array import ARRAY as PGARRAY
from .base import _DECIMAL_TYPES
from .base import _FLOAT_TYPES
from .base import _INT_TYPES
from .base import ENUM
from .base import INTERVAL
from .base import OID
from .base import PGCompiler
from .base import PGDialect
from .base import PGExecutionContext
from .base import PGIdentifierPreparer
from .base import REGCLASS
from .base import REGCONFIG
from .types import BIT
from .types import BYTEA
from .types import CITEXT
from ... import exc
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...engine import processors
from ...sql import sqltypes
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only


class AsyncpgARRAY(PGARRAY):
    render_bind_cast = True


class AsyncpgString(sqltypes.String):
    render_bind_cast = True


class AsyncpgREGCONFIG(REGCONFIG):
    render_bind_cast = True


class AsyncpgTime(sqltypes.Time):
    render_bind_cast = True


class AsyncpgBit(BIT):
    render_bind_cast = True


class AsyncpgByteA(BYTEA):
    render_bind_cast = True


class AsyncpgDate(sqltypes.Date):
    render_bind_cast = True


class AsyncpgDateTime(sqltypes.DateTime):
    render_bind_cast = True


class AsyncpgBoolean(sqltypes.Boolean):
    render_bind_cast = True


class AsyncPgInterval(INTERVAL):
    render_bind_cast = True

    @classmethod
    def adapt_emulated_to_native(cls, interval, **kw):
        return AsyncPgInterval(precision=interval.second_precision)


class AsyncPgEnum(ENUM):
    render_bind_cast = True


class AsyncpgInteger(sqltypes.Integer):
    render_bind_cast = True


class AsyncpgSmallInteger(sqltypes.SmallInteger):
    render_bind_cast = True


class AsyncpgBigInteger(sqltypes.BigInteger):
    render_bind_cast = True


class AsyncpgJSON(json.JSON):
    def result_processor(self, dialect, coltype):
        return None


class AsyncpgJSONB(json.JSONB):
    def result_processor(self, dialect, coltype):
        return None


class AsyncpgJSONIndexType(sqltypes.JSON.JSONIndexType):
    pass


class AsyncpgJSONIntIndexType(sqltypes.JSON.JSONIntIndexType):
    __visit_name__ = "json_int_index"

    render_bind_cast = True


class AsyncpgJSONStrIndexType(sqltypes.JSON.JSONStrIndexType):
    __visit_name__ = "json_str_index"

    render_bind_cast = True


class AsyncpgJSONPathType(json.JSONPathType):
    def bind_processor(self, dialect):
        def process(value):
            if isinstance(value, str):
                # If it's already a string assume that it's in json path
                # format. This allows using cast with json paths literals
                return value
            elif value:
                tokens = [str(elem) for elem in value]
                return tokens
            else:
                return []

        return process


class AsyncpgNumeric(sqltypes.Numeric):
    render_bind_cast = True

    def bind_processor(self, dialect):
        return None

    def result_processor(self, dialect, coltype):
        if self.asdecimal:
            if coltype in _FLOAT_TYPES:
                return processors.to_decimal_processor_factory(
                    decimal.Decimal, self._effective_decimal_return_scale
                )
            elif coltype in _DECIMAL_TYPES or coltype in _INT_TYPES:
                # pg8000 returns Decimal natively for 1700
                return None
            else:
                raise exc.InvalidRequestError(
                    "Unknown PG numeric type: %d" % coltype
                )
        else:
            if coltype in _FLOAT_TYPES:
                # pg8000 returns float natively for 701
                return None
            elif coltype in _DECIMAL_TYPES or coltype in _INT_TYPES:
                return processors.to_float
            else:
                raise exc.InvalidRequestError(
                    "Unknown PG numeric type: %d" % coltype
                )


class AsyncpgFloat(AsyncpgNumeric, sqltypes.Float):
    __visit_name__ = "float"
    render_bind_cast = True


class AsyncpgREGCLASS(REGCLASS):
    render_bind_cast = True


class AsyncpgOID(OID):
    render_bind_cast = True


class AsyncpgCHAR(sqltypes.CHAR):
    render_bind_cast = True


class _AsyncpgRange(ranges.AbstractSingleRangeImpl):
    def bind_processor(self, dialect):
        asyncpg_Range = dialect.dbapi.asyncpg.Range

        def to_range(value):
            if isinstance(value, ranges.Range):
                value = asyncpg_Range(
                    value.lower,
                    value.upper,
                    lower_inc=value.bounds[0] == "[",
                    upper_inc=value.bounds[1] == "]",
                    empty=value.empty,
                )
            return value

        return to_range

    def result_processor(self, dialect, coltype):
        def to_range(value):
            if value is not None:
                empty = value.isempty
                value = ranges.Range(
                    value.lower,
                    value.upper,
                    bounds=f"{'[' if empty or value.lower_inc else '('}"  # type: ignore  # noqa: E501
                    f"{']' if not empty and value.upper_inc else ')'}",
                    empty=empty,
                )
            return value

        return to_range


class _AsyncpgMultiRange(ranges.AbstractMultiRangeImpl):
    def bind_processor(self, dialect):
        asyncpg_Range = dialect.dbapi.asyncpg.Range

        NoneType = type(None)

        def to_range(value):
            if isinstance(value, (str, NoneType)):
                return value

            def to_range(value):
                if isinstance(value, ranges.Range):
                    value = asyncpg_Range(
                        value.lower,
                        value.upper,
                        lower_inc=value.bounds[0] == "[",
                        upper_inc=value.bounds[1] == "]",
                        empty=value.empty,
                    )
                return value

            return [to_range(element) for element in value]

        return to_range

    def result_processor(self, dialect, coltype):
        def to_range_array(value):
            def to_range(rvalue):
                if rvalue is not None:
                    empty = rvalue.isempty
                    rvalue = ranges.Range(
                        rvalue.lower,
                        rvalue.upper,
                        bounds=f"{'[' if empty or rvalue.lower_inc else '('}"  # type: ignore  # noqa: E501
                        f"{']' if not empty and rvalue.upper_inc else ')'}",
                        empty=empty,
                    )
                return rvalue

            if value is not None:
                value = ranges.MultiRange(to_range(elem) for elem in value)

            return value

        return to_range_array


class PGExecutionContext_asyncpg(PGExecutionContext):
    def handle_dbapi_exception(self, e):
        if isinstance(
            e,
            (
                self.dialect.dbapi.InvalidCachedStatementError,
                self.dialect.dbapi.InternalServerError,
            ),
        ):
            self.dialect._invalidate_schema_cache()

    def pre_exec(self):
        if self.isddl:
            self.dialect._invalidate_schema_cache()

        self.cursor._invalidate_schema_cache_asof = (
            self.dialect._invalidate_schema_cache_asof
        )

        if not self.compiled:
            return

    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor(server_side=True)


class PGCompiler_asyncpg(PGCompiler):
    pass


class PGIdentifierPreparer_asyncpg(PGIdentifierPreparer):
    pass


class AsyncAdapt_asyncpg_cursor:
    __slots__ = (
        "_adapt_connection",
        "_connection",
        "_rows",
        "description",
        "arraysize",
        "rowcount",
        "_cursor",
        "_invalidate_schema_cache_asof",
    )

    server_side = False

    def __init__(self, adapt_connection):
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self._rows = deque()
        self._cursor = None
        self.description = None
        self.arraysize = 1
        self.rowcount = -1
        self._invalidate_schema_cache_asof = 0

    def close(self):
        self._rows.clear()

    def _handle_exception(self, error):
        self._adapt_connection._handle_exception(error)

    async def _prepare_and_execute(self, operation, parameters):
        adapt_connection = self._adapt_connection

        async with adapt_connection._execute_mutex:
            if not adapt_connection._started:
                await adapt_connection._start_transaction()

            if parameters is None:
                parameters = ()

            try:
                prepared_stmt, attributes = await adapt_connection._prepare(
                    operation, self._invalidate_schema_cache_asof
                )

                if attributes:
                    self.description = [
                        (
                            attr.name,
                            attr.type.oid,
                            None,
                            None,
                            None,
                            None,
                            None,
                        )
                        for attr in attributes
                    ]
                else:
                    self.description = None

                if self.server_side:
                    self._cursor = await prepared_stmt.cursor(*parameters)
                    self.rowcount = -1
                else:
                    self._rows = deque(await prepared_stmt.fetch(*parameters))
                    status = prepared_stmt.get_statusmsg()

                    reg = re.match(
                        r"(?:SELECT|UPDATE|DELETE|INSERT \d+) (\d+)",
                        status or "",
                    )
                    if reg:
                        self.rowcount = int(reg.group(1))
                    else:
                        self.rowcount = -1

            except Exception as error:
                self._handle_exception(error)

    async def _executemany(self, operation, seq_of_parameters):
        adapt_connection = self._adapt_connection

        self.description = None
        async with adapt_connection._execute_mutex:
            await adapt_connection._check_type_cache_invalidation(
                self._invalidate_schema_cache_asof
            )

            if not adapt_connection._started:
                await adapt_connection._start_transaction()

            try:
                return await self._connection.executemany(
                    operation, seq_of_parameters
                )
            except Exception as error:
                self._handle_exception(error)

    def execute(self, operation, parameters=None):
        self._adapt_connection.await_(
            self._prepare_and_execute(operation, parameters)
        )

    def executemany(self, operation, seq_of_parameters):
        return self._adapt_connection.await_(
            self._executemany(operation, seq_of_parameters)
        )

    def setinputsizes(self, *inputsizes):
        raise NotImplementedError()

    def __iter__(self):
        while self._rows:
            yield self._rows.popleft()

    def fetchone(self):
        if self._rows:
            return self._rows.popleft()
        else:
            return None

    def fetchmany(self, size=None):
        if size is None:
            size = self.arraysize

        rr = self._rows
        return [rr.popleft() for _ in range(min(size, len(rr)))]

    def fetchall(self):
        retval = list(self._rows)
        self._rows.clear()
        return retval


class AsyncAdapt_asyncpg_ss_cursor(AsyncAdapt_asyncpg_cursor):
    server_side = True
    __slots__ = ("_rowbuffer",)

    def __init__(self, adapt_connection):
        super().__init__(adapt_connection)
        self._rowbuffer = deque()

    def close(self):
        self._cursor = None
        self._rowbuffer.clear()

    def _buffer_rows(self):
        assert self._cursor is not None
        new_rows = self._adapt_connection.await_(self._cursor.fetch(50))
        self._rowbuffer.extend(new_rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        while True:
            while self._rowbuffer:
                yield self._rowbuffer.popleft()

            self._buffer_rows()
            if not self._rowbuffer:
                break

    def fetchone(self):
        if not self._rowbuffer:
            self._buffer_rows()
            if not self._rowbuffer:
                return None
        return self._rowbuffer.popleft()

    def fetchmany(self, size=None):
        if size is None:
            return self.fetchall()

        if not self._rowbuffer:
            self._buffer_rows()

        assert self._cursor is not None
        rb = self._rowbuffer
        lb = len(rb)
        if size > lb:
            rb.extend(
                self._adapt_connection.await_(self._cursor.fetch(size - lb))
            )

        return [rb.popleft() for _ in range(min(size, len(rb)))]

    def fetchall(self):
        ret = list(self._rowbuffer)
        ret.extend(self._adapt_connection.await_(self._all()))
        self._rowbuffer.clear()
        return ret

    async def _all(self):
        rows = []

        # TODO: looks like we have to hand-roll some kind of batching here.
        # hardcoding for the moment but this should be improved.
        while True:
            batch = await self._cursor.fetch(1000)
            if batch:
                rows.extend(batch)
                continue
            else:
                break
        return rows

    def executemany(self, operation, seq_of_parameters):
        raise NotImplementedError(
            "server side cursor doesn't support executemany yet"
        )


class AsyncAdapt_asyncpg_connection(AdaptedConnection):
    __slots__ = (
        "dbapi",
        "isolation_level",
        "_isolation_setting",
        "readonly",
        "deferrable",
        "_transaction",
        "_started",
        "_prepared_statement_cache",
        "_prepared_statement_name_func",
        "_invalidate_schema_cache_asof",
        "_execute_mutex",
    )

    await_ = staticmethod(await_only)

    def __init__(
        self,
        dbapi,
        connection,
        prepared_statement_cache_size=100,
        prepared_statement_name_func=None,
    ):
        self.dbapi = dbapi
        self._connection = connection
        self.isolation_level = self._isolation_setting = None
        self.readonly = False
        self.deferrable = False
        self._transaction = None
        self._started = False
        self._invalidate_schema_cache_asof = time.time()
        self._execute_mutex = asyncio.Lock()

        if prepared_statement_cache_size:
            self._prepared_statement_cache = util.LRUCache(
                prepared_statement_cache_size
            )
        else:
            self._prepared_statement_cache = None

        if prepared_statement_name_func:
            self._prepared_statement_name_func = prepared_statement_name_func
        else:
            self._prepared_statement_name_func = self._default_name_func

    async def _check_type_cache_invalidation(self, invalidate_timestamp):
        if invalidate_timestamp > self._invalidate_schema_cache_asof:
            await self._connection.reload_schema_state()
            self._invalidate_schema_cache_asof = invalidate_timestamp

    async def _prepare(self, operation, invalidate_timestamp):
        await self._check_type_cache_invalidation(invalidate_timestamp)

        cache = self._prepared_statement_cache
        if cache is None:
            prepared_stmt = await self._connection.prepare(
                operation, name=self._prepared_statement_name_func()
            )
            attributes = prepared_stmt.get_attributes()
            return prepared_stmt, attributes

        # asyncpg uses a type cache for the "attributes" which seems to go
        # stale independently of the PreparedStatement itself, so place that
        # collection in the cache as well.
        if operation in cache:
            prepared_stmt, attributes, cached_timestamp = cache[operation]

            # preparedstatements themselves also go stale for certain DDL
            # changes such as size of a VARCHAR changing, so there is also
            # a cross-connection invalidation timestamp
            if cached_timestamp > invalidate_timestamp:
                return prepared_stmt, attributes

        prepared_stmt = await self._connection.prepare(
            operation, name=self._prepared_statement_name_func()
        )
        attributes = prepared_stmt.get_attributes()
        cache[operation] = (prepared_stmt, attributes, time.time())

        return prepared_stmt, attributes

    def _handle_exception(self, error):
        if self._connection.is_closed():
            self._transaction = None
            self._started = False

        if not isinstance(error, AsyncAdapt_asyncpg_dbapi.Error):
            exception_mapping = self.dbapi._asyncpg_error_translate

            for super_ in type(error).__mro__:
                if super_ in exception_mapping:
                    translated_error = exception_mapping[super_](
                        "%s: %s" % (type(error), error)
                    )
                    translated_error.pgcode = translated_error.sqlstate = (
                        getattr(error, "sqlstate", None)
                    )
                    raise translated_error from error
            else:
                raise error
        else:
            raise error

    @property
    def autocommit(self):
        return self.isolation_level == "autocommit"

    @autocommit.setter
    def autocommit(self, value):
        if value:
            self.isolation_level = "autocommit"
        else:
            self.isolation_level = self._isolation_setting

    def ping(self):
        try:
            _ = self.await_(self._async_ping())
        except Exception as error:
            self._handle_exception(error)

    async def _async_ping(self):
        if self._transaction is None and self.isolation_level != "autocommit":
            # create a tranasction explicitly to support pgbouncer
            # transaction mode.   See #10226
            tr = self._connection.transaction()
            await tr.start()
            try:
                await self._connection.fetchrow(";")
            finally:
                await tr.rollback()
        else:
            await self._connection.fetchrow(";")

    def set_isolation_level(self, level):
        if self._started:
            self.rollback()
        self.isolation_level = self._isolation_setting = level

    async def _start_transaction(self):
        if self.isolation_level == "autocommit":
            return

        try:
            self._transaction = self._connection.transaction(
                isolation=self.isolation_level,
                readonly=self.readonly,
                deferrable=self.deferrable,
            )
            await self._transaction.start()
        except Exception as error:
            self._handle_exception(error)
        else:
            self._started = True

    def cursor(self, server_side=False):
        if server_side:
            return AsyncAdapt_asyncpg_ss_cursor(self)
        else:
            return AsyncAdapt_asyncpg_cursor(self)

    async def _rollback_and_discard(self):
        try:
            await self._transaction.rollback()
        finally:
            # if asyncpg .rollback() was actually called, then whether or
            # not it raised or succeeded, the transation is done, discard it
            self._transaction = None
            self._started = False

    async def _commit_and_discard(self):
        try:
            await self._transaction.commit()
        finally:
            # if asyncpg .commit() was actually called, then whether or
            # not it raised or succeeded, the transation is done, discard it
            self._transaction = None
            self._started = False

    def rollback(self):
        if self._started:
            try:
                self.await_(self._rollback_and_discard())
                self._transaction = None
                self._started = False
            except Exception as error:
                # don't dereference asyncpg transaction if we didn't
                # actually try to call rollback() on it
                self._handle_exception(error)

    def commit(self):
        if self._started:
            try:
                self.await_(self._commit_and_discard())
                self._transaction = None
                self._started = False
            except Exception as error:
                # don't dereference asyncpg transaction if we didn't
                # actually try to call commit() on it
                self._handle_exception(error)

    def close(self):
        self.rollback()

        self.await_(self._connection.close())

    def terminate(self):
        if util.concurrency.in_greenlet():
            # in a greenlet; this is the connection was invalidated
            # case.
            try:
                # try to gracefully close; see #10717
                # timeout added in asyncpg 0.14.0 December 2017
                self.await_(asyncio.shield(self._connection.close(timeout=2)))
            except (
                asyncio.TimeoutError,
                asyncio.CancelledError,
                OSError,
                self.dbapi.asyncpg.PostgresError,
            ):
                # in the case where we are recycling an old connection
                # that may have already been disconnected, close() will
                # fail with the above timeout.  in this case, terminate
                # the connection without any further waiting.
                # see issue #8419
                self._connection.terminate()
        else:
            # not in a greenlet; this is the gc cleanup case
            self._connection.terminate()
        self._started = False

    @staticmethod
    def _default_name_func():
        return None


class AsyncAdaptFallback_asyncpg_connection(AsyncAdapt_asyncpg_connection):
    __slots__ = ()

    await_ = staticmethod(await_fallback)


class AsyncAdapt_asyncpg_dbapi:
    def __init__(self, asyncpg):
        self.asyncpg = asyncpg
        self.paramstyle = "numeric_dollar"

    def connect(self, *arg, **kw):
        async_fallback = kw.pop("async_fallback", False)
        creator_fn = kw.pop("async_creator_fn", self.asyncpg.connect)
        prepared_statement_cache_size = kw.pop(
            "prepared_statement_cache_size", 100
        )
        prepared_statement_name_func = kw.pop(
            "prepared_statement_name_func", None
        )

        if util.asbool(async_fallback):
            return AsyncAdaptFallback_asyncpg_connection(
                self,
                await_fallback(creator_fn(*arg, **kw)),
                prepared_statement_cache_size=prepared_statement_cache_size,
                prepared_statement_name_func=prepared_statement_name_func,
            )
        else:
            return AsyncAdapt_asyncpg_connection(
                self,
                await_only(creator_fn(*arg, **kw)),
                prepared_statement_cache_size=prepared_statement_cache_size,
                prepared_statement_name_func=prepared_statement_name_func,
            )

    class Error(Exception):
        pass

    class Warning(Exception):  # noqa
        pass

    class InterfaceError(Error):
        pass

    class DatabaseError(Error):
        pass

    class InternalError(DatabaseError):
        pass

    class OperationalError(DatabaseError):
        pass

    class ProgrammingError(DatabaseError):
        pass

    class IntegrityError(DatabaseError):
        pass

    class DataError(DatabaseError):
        pass

    class NotSupportedError(DatabaseError):
        pass

    class InternalServerError(InternalError):
        pass

    class InvalidCachedStatementError(NotSupportedError):
        def __init__(self, message):
            super().__init__(
                message + " (SQLAlchemy asyncpg dialect will now invalidate "
                "all prepared caches in response to this exception)",
            )

    # pep-249 datatype placeholders.  As of SQLAlchemy 2.0 these aren't
    # used, however the test suite looks for these in a few cases.
    STRING = util.symbol("STRING")
    NUMBER = util.symbol("NUMBER")
    DATETIME = util.symbol("DATETIME")

    @util.memoized_property
    def _asyncpg_error_translate(self):
        import asyncpg

        return {
            asyncpg.exceptions.IntegrityConstraintViolationError: self.IntegrityError,  # noqa: E501
            asyncpg.exceptions.PostgresError: self.Error,
            asyncpg.exceptions.SyntaxOrAccessError: self.ProgrammingError,
            asyncpg.exceptions.InterfaceError: self.InterfaceError,
            asyncpg.exceptions.InvalidCachedStatementError: self.InvalidCachedStatementError,  # noqa: E501
            asyncpg.exceptions.InternalServerError: self.InternalServerError,
        }

    def Binary(self, value):
        return value


class PGDialect_asyncpg(PGDialect):
    driver = "asyncpg"
    supports_statement_cache = True

    supports_server_side_cursors = True

    render_bind_cast = True
    has_terminate = True

    default_paramstyle = "numeric_dollar"
    supports_sane_multi_rowcount = False
    execution_ctx_cls = PGExecutionContext_asyncpg
    statement_compiler = PGCompiler_asyncpg
    preparer = PGIdentifierPreparer_asyncpg

    colspecs = util.update_copy(
        PGDialect.colspecs,
        {
            sqltypes.String: AsyncpgString,
            sqltypes.ARRAY: AsyncpgARRAY,
            BIT: AsyncpgBit,
            CITEXT: CITEXT,
            REGCONFIG: AsyncpgREGCONFIG,
            sqltypes.Time: AsyncpgTime,
            sqltypes.Date: AsyncpgDate,
            sqltypes.DateTime: AsyncpgDateTime,
            sqltypes.Interval: AsyncPgInterval,
            INTERVAL: AsyncPgInterval,
            sqltypes.Boolean: AsyncpgBoolean,
            sqltypes.Integer: AsyncpgInteger,
            sqltypes.SmallInteger: AsyncpgSmallInteger,
            sqltypes.BigInteger: AsyncpgBigInteger,
            sqltypes.Numeric: AsyncpgNumeric,
            sqltypes.Float: AsyncpgFloat,
            sqltypes.JSON: AsyncpgJSON,
            sqltypes.LargeBinary: AsyncpgByteA,
            json.JSONB: AsyncpgJSONB,
            sqltypes.JSON.JSONPathType: AsyncpgJSONPathType,
            sqltypes.JSON.JSONIndexType: AsyncpgJSONIndexType,
            sqltypes.JSON.JSONIntIndexType: AsyncpgJSONIntIndexType,
            sqltypes.JSON.JSONStrIndexType: AsyncpgJSONStrIndexType,
            sqltypes.Enum: AsyncPgEnum,
            OID: AsyncpgOID,
            REGCLASS: AsyncpgREGCLASS,
            sqltypes.CHAR: AsyncpgCHAR,
            ranges.AbstractSingleRange: _AsyncpgRange,
            ranges.AbstractMultiRange: _AsyncpgMultiRange,
        },
    )
    is_async = True
    _invalidate_schema_cache_asof = 0

    def _invalidate_schema_cache(self):
        self._invalidate_schema_cache_asof = time.time()

    @util.memoized_property
    def _dbapi_version(self):
        if self.dbapi and hasattr(self.dbapi, "__version__"):
            return tuple(
                [
                    int(x)
                    for x in re.findall(
                        r"(\d+)(?:[-\.]?|$)", self.dbapi.__version__
                    )
                ]
            )
        else:
            return (99, 99, 99)

    @classmethod
    def import_dbapi(cls):
        return AsyncAdapt_asyncpg_dbapi(__import__("asyncpg"))

    @util.memoized_property
    def _isolation_lookup(self):
        return {
            "AUTOCOMMIT": "autocommit",
            "READ COMMITTED": "read_committed",
            "REPEATABLE READ": "repeatable_read",
            "SERIALIZABLE": "serializable",
        }

    def get_isolation_level_values(self, dbapi_connection):
        return list(self._isolation_lookup)

    def set_isolation_level(self, dbapi_connection, level):
        dbapi_connection.set_isolation_level(self._isolation_lookup[level])

    def set_readonly(self, connection, value):
        connection.readonly = value

    def get_readonly(self, connection):
        return connection.readonly

    def set_deferrable(self, connection, value):
        connection.deferrable = value

    def get_deferrable(self, connection):
        return connection.deferrable

    def do_terminate(self, dbapi_connection) -> None:
        dbapi_connection.terminate()

    def create_connect_args(self, url):
        opts = url.translate_connect_args(username="user")
        multihosts, multiports = self._split_multihost_from_url(url)

        opts.update(url.query)

        if multihosts:
            assert multiports
            if len(multihosts) == 1:
                opts["host"] = multihosts[0]
                if multiports[0] is not None:
                    opts["port"] = multiports[0]
            elif not all(multihosts):
                raise exc.ArgumentError(
                    "All hosts are required to be present"
                    " for asyncpg multiple host URL"
                )
            elif not all(multiports):
                raise exc.ArgumentError(
                    "All ports are required to be present"
                    " for asyncpg multiple host URL"
                )
            else:
                opts["host"] = list(multihosts)
                opts["port"] = list(multiports)
        else:
            util.coerce_kw_type(opts, "port", int)
        util.coerce_kw_type(opts, "prepared_statement_cache_size", int)
        return ([], opts)

    def do_ping(self, dbapi_connection):
        dbapi_connection.ping()
        return True

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get("async_fallback", False)

        if util.asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def is_disconnect(self, e, connection, cursor):
        if connection:
            return connection._connection.is_closed()
        else:
            return isinstance(
                e, self.dbapi.InterfaceError
            ) and "connection is closed" in str(e)

    async def setup_asyncpg_json_codec(self, conn):
        """set up JSON codec for asyncpg.

        This occurs for all new connections and
        can be overridden by third party dialects.

        .. versionadded:: 1.4.27

        """

        asyncpg_connection = conn._connection
        deserializer = self._json_deserializer or _py_json.loads

        def _json_decoder(bin_value):
            return deserializer(bin_value.decode())

        await asyncpg_connection.set_type_codec(
            "json",
            encoder=str.encode,
            decoder=_json_decoder,
            schema="pg_catalog",
            format="binary",
        )

    async def setup_asyncpg_jsonb_codec(self, conn):
        """set up JSONB codec for asyncpg.

        This occurs for all new connections and
        can be overridden by third party dialects.

        .. versionadded:: 1.4.27

        """

        asyncpg_connection = conn._connection
        deserializer = self._json_deserializer or _py_json.loads

        def _jsonb_encoder(str_value):
            # \x01 is the prefix for jsonb used by PostgreSQL.
            # asyncpg requires it when format='binary'
            return b"\x01" + str_value.encode()

        deserializer = self._json_deserializer or _py_json.loads

        def _jsonb_decoder(bin_value):
            # the byte is the \x01 prefix for jsonb used by PostgreSQL.
            # asyncpg returns it when format='binary'
            return deserializer(bin_value[1:].decode())

        await asyncpg_connection.set_type_codec(
            "jsonb",
            encoder=_jsonb_encoder,
            decoder=_jsonb_decoder,
            schema="pg_catalog",
            format="binary",
        )

    async def _disable_asyncpg_inet_codecs(self, conn):
        asyncpg_connection = conn._connection

        await asyncpg_connection.set_type_codec(
            "inet",
            encoder=lambda s: s,
            decoder=lambda s: s,
            schema="pg_catalog",
            format="text",
        )

        await asyncpg_connection.set_type_codec(
            "cidr",
            encoder=lambda s: s,
            decoder=lambda s: s,
            schema="pg_catalog",
            format="text",
        )

    def on_connect(self):
        """on_connect for asyncpg

        A major component of this for asyncpg is to set up type decoders at the
        asyncpg level.

        See https://github.com/MagicStack/asyncpg/issues/623 for
        notes on JSON/JSONB implementation.

        """

        super_connect = super().on_connect()

        def connect(conn):
            conn.await_(self.setup_asyncpg_json_codec(conn))
            conn.await_(self.setup_asyncpg_jsonb_codec(conn))

            if self._native_inet_types is False:
                conn.await_(self._disable_asyncpg_inet_codecs(conn))
            if super_connect is not None:
                super_connect(conn)

        return connect

    def get_driver_connection(self, connection):
        return connection._connection


dialect = PGDialect_asyncpg

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\elementary\exponential.py ===
from __future__ import annotations
from itertools import product

from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (DefinedFunction, ArgumentIndexError, expand_log,
    expand_mul, FunctionClass, PoleError, expand_multinomial, expand_complex)
from sympy.core.logic import fuzzy_and, fuzzy_not, fuzzy_or
from sympy.core.mul import Mul
from sympy.core.numbers import Integer, Rational, pi, I
from sympy.core.parameters import global_parameters
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.complexes import arg, unpolarify, im, re, Abs
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.ntheory import multiplicity, perfect_power
from sympy.ntheory.factor_ import factorint

# NOTE IMPORTANT
# The series expansion code in this file is an important part of the gruntz
# algorithm for determining limits. _eval_nseries has to return a generalized
# power series with coefficients in C(log(x), log).
# In more detail, the result of _eval_nseries(self, x, n) must be
#   c_0*x**e_0 + ... (finitely many terms)
# where e_i are numbers (not necessarily integers) and c_i involve only
# numbers, the function log, and log(x). [This also means it must not contain
# log(x(1+p)), this *has* to be expanded to log(x)+log(1+p) if x.is_positive and
# p.is_positive.]


class ExpBase(DefinedFunction):

    unbranched = True
    _singularities = (S.ComplexInfinity,)

    @property
    def kind(self):
        return self.exp.kind

    def inverse(self, argindex=1):
        """
        Returns the inverse function of ``exp(x)``.
        """
        return log

    def as_numer_denom(self):
        """
        Returns this with a positive exponent as a 2-tuple (a fraction).

        Examples
        ========

        >>> from sympy import exp
        >>> from sympy.abc import x
        >>> exp(-x).as_numer_denom()
        (1, exp(x))
        >>> exp(x).as_numer_denom()
        (exp(x), 1)
        """
        # this should be the same as Pow.as_numer_denom wrt
        # exponent handling
        if not self.is_commutative:
            return self, S.One
        exp = self.exp
        neg_exp = exp.is_negative
        if not neg_exp and not (-exp).is_negative:
            neg_exp = exp.could_extract_minus_sign()
        if neg_exp:
            return S.One, self.func(-exp)
        return self, S.One

    @property
    def exp(self):
        """
        Returns the exponent of the function.
        """
        return self.args[0]

    def as_base_exp(self):
        """
        Returns the 2-tuple (base, exponent).
        """
        return self.func(1), Mul(*self.args)

    def _eval_adjoint(self):
        return self.func(self.exp.adjoint())

    def _eval_conjugate(self):
        return self.func(self.exp.conjugate())

    def _eval_transpose(self):
        return self.func(self.exp.transpose())

    def _eval_is_finite(self):
        arg = self.exp
        if arg.is_infinite:
            if arg.is_extended_negative:
                return True
            if arg.is_extended_positive:
                return False
        if arg.is_finite:
            return True

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            z = s.exp.is_zero
            if z:
                return True
            elif s.exp.is_rational and fuzzy_not(z):
                return False
        else:
            return s.is_rational

    def _eval_is_zero(self):
        return self.exp is S.NegativeInfinity

    def _eval_power(self, other):
        """exp(arg)**e -> exp(arg*e) if assumptions allow it.
        """
        b, e = self.as_base_exp()
        return Pow._eval_power(Pow(b, e, evaluate=False), other)

    def _eval_expand_power_exp(self, **hints):
        from sympy.concrete.products import Product
        from sympy.concrete.summations import Sum
        arg = self.args[0]
        if arg.is_Add and arg.is_commutative:
            return Mul.fromiter(self.func(x) for x in arg.args)
        elif isinstance(arg, Sum) and arg.is_commutative:
            return Product(self.func(arg.function), *arg.limits)
        return self.func(arg)


class exp_polar(ExpBase):
    r"""
    Represent a *polar number* (see g-function Sphinx documentation).

    Explanation
    ===========

    ``exp_polar`` represents the function
    `Exp: \mathbb{C} \rightarrow \mathcal{S}`, sending the complex number
    `z = a + bi` to the polar number `r = exp(a), \theta = b`. It is one of
    the main functions to construct polar numbers.

    Examples
    ========

    >>> from sympy import exp_polar, pi, I, exp

    The main difference is that polar numbers do not "wrap around" at `2 \pi`:

    >>> exp(2*pi*I)
    1
    >>> exp_polar(2*pi*I)
    exp_polar(2*I*pi)

    apart from that they behave mostly like classical complex numbers:

    >>> exp_polar(2)*exp_polar(3)
    exp_polar(5)

    See Also
    ========

    sympy.simplify.powsimp.powsimp
    polar_lift
    periodic_argument
    principal_branch
    """

    is_polar = True
    is_comparable = False  # cannot be evalf'd

    def _eval_Abs(self):   # Abs is never a polar number
        return exp(re(self.args[0]))

    def _eval_evalf(self, prec):
        """ Careful! any evalf of polar numbers is flaky """
        i = im(self.args[0])
        try:
            bad = (i <= -pi or i > pi)
        except TypeError:
            bad = True
        if bad:
            return self  # cannot evalf for this argument
        res = exp(self.args[0])._eval_evalf(prec)
        if i > 0 and im(res) < 0:
            # i ~ pi, but exp(I*i) evaluated to argument slightly bigger than pi
            return re(res)
        return res

    def _eval_power(self, other):
        return self.func(self.args[0]*other)

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True

    def as_base_exp(self):
        # XXX exp_polar(0) is special!
        if self.args[0] == 0:
            return self, S.One
        return ExpBase.as_base_exp(self)


class ExpMeta(FunctionClass):
    def __instancecheck__(cls, instance):
        if exp in instance.__class__.__mro__:
            return True
        return isinstance(instance, Pow) and instance.base is S.Exp1


class exp(ExpBase, metaclass=ExpMeta):
    """
    The exponential function, :math:`e^x`.

    Examples
    ========

    >>> from sympy import exp, I, pi
    >>> from sympy.abc import x
    >>> exp(x)
    exp(x)
    >>> exp(x).diff(x)
    exp(x)
    >>> exp(I*pi)
    -1

    Parameters
    ==========

    arg : Expr

    See Also
    ========

    log
    """

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return self
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_refine(self, assumptions):
        from sympy.assumptions import ask, Q
        arg = self.args[0]
        if arg.is_Mul:
            Ioo = I*S.Infinity
            if arg in [Ioo, -Ioo]:
                return S.NaN

            coeff = arg.as_coefficient(pi*I)
            if coeff:
                if ask(Q.integer(2*coeff)):
                    if ask(Q.even(coeff)):
                        return S.One
                    elif ask(Q.odd(coeff)):
                        return S.NegativeOne
                    elif ask(Q.even(coeff + S.Half)):
                        return -I
                    elif ask(Q.odd(coeff + S.Half)):
                        return I

    @classmethod
    def eval(cls, arg):
        from sympy.calculus import AccumBounds
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.sets.setexpr import SetExpr
        from sympy.simplify.simplify import logcombine
        if isinstance(arg, MatrixBase):
            return arg.exp()
        elif global_parameters.exp_is_pow:
            return Pow(S.Exp1, arg)
        elif arg.is_Number:
            if arg is S.NaN:
                return S.NaN
            elif arg.is_zero:
                return S.One
            elif arg is S.One:
                return S.Exp1
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Zero
        elif arg is S.ComplexInfinity:
            return S.NaN
        elif isinstance(arg, log):
            return arg.args[0]
        elif isinstance(arg, AccumBounds):
            return AccumBounds(exp(arg.min), exp(arg.max))
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)
        elif arg.is_Mul:
            coeff = arg.as_coefficient(pi*I)
            if coeff:
                if (2*coeff).is_integer:
                    if coeff.is_even:
                        return S.One
                    elif coeff.is_odd:
                        return S.NegativeOne
                    elif (coeff + S.Half).is_even:
                        return -I
                    elif (coeff + S.Half).is_odd:
                        return I
                elif coeff.is_Rational:
                    ncoeff = coeff % 2 # restrict to [0, 2pi)
                    if ncoeff > 1: # restrict to (-pi, pi]
                        ncoeff -= 2
                    if ncoeff != coeff:
                        return cls(ncoeff*pi*I)

            # Warning: code in risch.py will be very sensitive to changes
            # in this (see DifferentialExtension).

            # look for a single log factor

            coeff, terms = arg.as_coeff_Mul()

            # but it can't be multiplied by oo
            if coeff in [S.NegativeInfinity, S.Infinity]:
                if terms.is_number:
                    if coeff is S.NegativeInfinity:
                        terms = -terms
                    if re(terms).is_zero and terms is not S.Zero:
                        return S.NaN
                    if re(terms).is_positive and im(terms) is not S.Zero:
                        return S.ComplexInfinity
                    if re(terms).is_negative:
                        return S.Zero
                return None

            coeffs, log_term = [coeff], None
            for term in Mul.make_args(terms):
                term_ = logcombine(term)
                if isinstance(term_, log):
                    if log_term is None:
                        log_term = term_.args[0]
                    else:
                        return None
                elif term.is_comparable:
                    coeffs.append(term)
                else:
                    return None

            return log_term**Mul(*coeffs) if log_term else None

        elif arg.is_Add:
            out = []
            add = []
            argchanged = False
            for a in arg.args:
                if a is S.One:
                    add.append(a)
                    continue
                newa = cls(a)
                if isinstance(newa, cls):
                    if newa.args[0] != a:
                        add.append(newa.args[0])
                        argchanged = True
                    else:
                        add.append(a)
                else:
                    out.append(newa)
            if out or argchanged:
                return Mul(*out)*cls(Add(*add), evaluate=False)

        if arg.is_zero:
            return S.One

    @property
    def base(self):
        """
        Returns the base of the exponential function.
        """
        return S.Exp1

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):
        """
        Calculates the next term in the Taylor series expansion.
        """
        if n < 0:
            return S.Zero
        if n == 0:
            return S.One
        x = sympify(x)
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return p * x / n
        return x**n/factorial(n)

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a 2-tuple representing a complex number.

        Examples
        ========

        >>> from sympy import exp, I
        >>> from sympy.abc import x
        >>> exp(x).as_real_imag()
        (exp(re(x))*cos(im(x)), exp(re(x))*sin(im(x)))
        >>> exp(1).as_real_imag()
        (E, 0)
        >>> exp(I).as_real_imag()
        (cos(1), sin(1))
        >>> exp(1+I).as_real_imag()
        (E*cos(1), E*sin(1))

        See Also
        ========

        sympy.functions.elementary.complexes.re
        sympy.functions.elementary.complexes.im
        """
        from sympy.functions.elementary.trigonometric import cos, sin
        re, im = self.args[0].as_real_imag()
        if deep:
            re = re.expand(deep, **hints)
            im = im.expand(deep, **hints)
        cos, sin = cos(im), sin(im)
        return (exp(re)*cos, exp(re)*sin)

    def _eval_subs(self, old, new):
        # keep processing of power-like args centralized in Pow
        if old.is_Pow:  # handle (exp(3*log(x))).subs(x**2, z) -> z**(3/2)
            old = exp(old.exp*log(old.base))
        elif old is S.Exp1 and new.is_Function:
            old = exp
        if isinstance(old, exp) or old is S.Exp1:
            f = lambda a: Pow(*a.as_base_exp(), evaluate=False) if (
                a.is_Pow or isinstance(a, exp)) else a
            return Pow._eval_subs(f(self), f(old), new)

        if old is exp and not new.is_Function:
            return new**self.exp._subs(old, new)
        return super()._eval_subs(old, new)

    def _eval_is_extended_real(self):
        if self.args[0].is_extended_real:
            return True
        elif self.args[0].is_imaginary:
            arg2 = -S(2) * I * self.args[0] / pi
            return arg2.is_even

    def _eval_is_complex(self):
        def complex_extended_negative(arg):
            yield arg.is_complex
            yield arg.is_extended_negative
        return fuzzy_or(complex_extended_negative(self.args[0]))

    def _eval_is_algebraic(self):
        if (self.exp / pi / I).is_rational:
            return True
        if fuzzy_not(self.exp.is_zero):
            if self.exp.is_algebraic:
                return False
            elif (self.exp / pi).is_rational:
                return False

    def _eval_is_extended_positive(self):
        if self.exp.is_extended_real:
            return self.args[0] is not S.NegativeInfinity
        elif self.exp.is_imaginary:
            arg2 = -I * self.args[0] / pi
            return arg2.is_even

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE Please see the comment at the beginning of this file, labelled
        #      IMPORTANT.
        from sympy.functions.elementary.complexes import sign
        from sympy.functions.elementary.integers import ceiling
        from sympy.series.limits import limit
        from sympy.series.order import Order
        from sympy.simplify.powsimp import powsimp
        arg = self.exp
        arg_series = arg._eval_nseries(x, n=n, logx=logx)
        if arg_series.is_Order:
            return 1 + arg_series
        arg0 = limit(arg_series.removeO(), x, 0)
        if arg0 is S.NegativeInfinity:
            return Order(x**n, x)
        if arg0 is S.Infinity:
            return self
        if arg0.is_infinite:
            raise PoleError("Cannot expand %s around 0" % (self))
        # checking for indecisiveness/ sign terms in arg0
        if any(isinstance(arg, sign) for arg in arg0.args):
            return self
        t = Dummy("t")
        nterms = n
        try:
            cf = Order(arg.as_leading_term(x, logx=logx), x).getn()
        except (NotImplementedError, PoleError):
            cf = 0
        if cf and cf > 0:
            nterms = ceiling(n/cf)
        exp_series = exp(t)._taylor(t, nterms)
        r = exp(arg0)*exp_series.subs(t, arg_series - arg0)
        rep = {logx: log(x)} if logx is not None else {}
        if r.subs(rep) == self:
            return r
        if cf and cf > 1:
            r += Order((arg_series - arg0)**n, x)/x**((cf-1)*n)
        else:
            r += Order((arg_series - arg0)**n, x)
        r = r.expand()
        r = powsimp(r, deep=True, combine='exp')
        # powsimp may introduce unexpanded (-1)**Rational; see PR #17201
        simplerat = lambda x: x.is_Rational and x.q in [3, 4, 6]
        w = Wild('w', properties=[simplerat])
        r = r.replace(S.NegativeOne**w, expand_complex(S.NegativeOne**w))
        return r

    def _taylor(self, x, n):
        l = []
        g = None
        for i in range(n):
            g = self.taylor_term(i, self.args[0], g)
            g = g.nseries(x, n=n)
            l.append(g.removeO())
        return Add(*l)

    def _eval_as_leading_term(self, x, logx, cdir):
        from sympy.calculus.util import AccumBounds
        arg = self.args[0].cancel().as_leading_term(x, logx=logx)
        arg0 = arg.subs(x, 0)
        if arg is S.NaN:
            return S.NaN
        if isinstance(arg0, AccumBounds):
            # This check addresses a corner case involving AccumBounds.
            # if isinstance(arg, AccumBounds) is True, then arg0 can either be 0,
            # AccumBounds(-oo, 0) or AccumBounds(-oo, oo).
            # Check out function: test_issue_18473() in test_exponential.py and
            # test_limits.py for more information.
            if re(cdir) < S.Zero:
                return exp(-arg0)
            return exp(arg0)
        if arg0 is S.NaN:
            arg0 = arg.limit(x, 0)
        if arg0.is_infinite is False:
            return exp(arg0)
        raise PoleError("Cannot expand %s around 0" % (self))

    def _eval_rewrite_as_sin(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import sin
        return sin(I*arg + pi/2) - I*sin(I*arg)

    def _eval_rewrite_as_cos(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import cos
        return cos(I*arg) + I*cos(I*arg + pi/2)

    def _eval_rewrite_as_tanh(self, arg, **kwargs):
        from sympy.functions.elementary.hyperbolic import tanh
        return (1 + tanh(arg/2))/(1 - tanh(arg/2))

    def _eval_rewrite_as_sqrt(self, arg, **kwargs):
        from sympy.functions.elementary.trigonometric import sin, cos
        if arg.is_Mul:
            coeff = arg.coeff(pi*I)
            if coeff and coeff.is_number:
                cosine, sine = cos(pi*coeff), sin(pi*coeff)
                if not isinstance(cosine, cos) and not isinstance (sine, sin):
                    return cosine + I*sine

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        if arg.is_Mul:
            logs = [a for a in arg.args if isinstance(a, log) and len(a.args) == 1]
            if logs:
                return Pow(logs[0].args[0], arg.coeff(logs[0]))


def match_real_imag(expr):
    r"""
    Try to match expr with $a + Ib$ for real $a$ and $b$.

    ``match_real_imag`` returns a tuple containing the real and imaginary
    parts of expr or ``(None, None)`` if direct matching is not possible. Contrary
    to :func:`~.re`, :func:`~.im``, and ``as_real_imag()``, this helper will not force things
    by returning expressions themselves containing ``re()`` or ``im()`` and it
    does not expand its argument either.

    """
    r_, i_ = expr.as_independent(I, as_Add=True)
    if i_ == 0 and r_.is_real:
        return (r_, i_)
    i_ = i_.as_coefficient(I)
    if i_ and i_.is_real and r_.is_real:
        return (r_, i_)
    else:
        return (None, None) # simpler to check for than None


class log(DefinedFunction):
    r"""
    The natural logarithm function `\ln(x)` or `\log(x)`.

    Explanation
    ===========

    Logarithms are taken with the natural base, `e`. To get
    a logarithm of a different base ``b``, use ``log(x, b)``,
    which is essentially short-hand for ``log(x)/log(b)``.

    ``log`` represents the principal branch of the natural
    logarithm. As such it has a branch cut along the negative
    real axis and returns values having a complex argument in
    `(-\pi, \pi]`.

    Examples
    ========

    >>> from sympy import log, sqrt, S, I
    >>> log(8, 2)
    3
    >>> log(S(8)/3, 2)
    -log(3)/log(2) + 3
    >>> log(-1 + I*sqrt(3))
    log(2) + 2*I*pi/3

    See Also
    ========

    exp

    """

    args: tuple[Expr]

    _singularities = (S.Zero, S.ComplexInfinity)

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of the function.
        """
        if argindex == 1:
            return 1/self.args[0]
        else:
            raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        r"""
        Returns `e^x`, the inverse function of `\log(x)`.
        """
        return exp

    @classmethod
    def eval(cls, arg, base=None):
        from sympy.calculus import AccumBounds
        from sympy.sets.setexpr import SetExpr

        arg = sympify(arg)

        if base is not None:
            base = sympify(base)
            if base == 1:
                if arg == 1:
                    return S.NaN
                else:
                    return S.ComplexInfinity
            try:
                # handle extraction of powers of the base now
                # or else expand_log in Mul would have to handle this
                n = multiplicity(base, arg)
                if n:
                    return n + log(arg / base**n) / log(base)
                else:
                    return log(arg)/log(base)
            except ValueError:
                pass
            if base is not S.Exp1:
                return cls(arg)/cls(base)
            else:
                return cls(arg)

        if arg.is_Number:
            if arg.is_zero:
                return S.ComplexInfinity
            elif arg is S.One:
                return S.Zero
            elif arg is S.Infinity:
                return S.Infinity
            elif arg is S.NegativeInfinity:
                return S.Infinity
            elif arg is S.NaN:
                return S.NaN
            elif arg.is_Rational and arg.p == 1:
                return -cls(arg.q)

        if arg.is_Pow and arg.base is S.Exp1 and arg.exp.is_extended_real:
            return arg.exp
        if isinstance(arg, exp) and arg.exp.is_extended_real:
            return arg.exp
        elif isinstance(arg, exp) and arg.exp.is_number:
            r_, i_ = match_real_imag(arg.exp)
            if i_ and i_.is_comparable:
                i_ %= 2*pi
                if i_ > pi:
                    i_ -= 2*pi
                return r_ + expand_mul(i_ * I, deep=False)
        elif isinstance(arg, exp_polar):
            return unpolarify(arg.exp)
        elif isinstance(arg, AccumBounds):
            if arg.min.is_positive:
                return AccumBounds(log(arg.min), log(arg.max))
            elif arg.min.is_zero:
                return AccumBounds(S.NegativeInfinity, log(arg.max))
            else:
                return S.NaN
        elif isinstance(arg, SetExpr):
            return arg._eval_func(cls)

        if arg.is_number:
            if arg.is_negative:
                return pi * I + cls(-arg)
            elif arg is S.ComplexInfinity:
                return S.ComplexInfinity
            elif arg is S.Exp1:
                return S.One

        if arg.is_zero:
            return S.ComplexInfinity

        # don't autoexpand Pow or Mul (see the issue 3351):
        if not arg.is_Add:
            coeff = arg.as_coefficient(I)

            if coeff is not None:
                if coeff is S.Infinity:
                    return S.Infinity
                elif coeff is S.NegativeInfinity:
                    return S.Infinity
                elif coeff.is_Rational:
                    if coeff.is_nonnegative:
                        return pi * I * S.Half + cls(coeff)
                    else:
                        return -pi * I * S.Half + cls(-coeff)

        if arg.is_number and arg.is_algebraic:
            # Match arg = coeff*(r_ + i_*I) with coeff>0, r_ and i_ real.
            coeff, arg_ = arg.as_independent(I, as_Add=False)
            if coeff.is_negative:
                coeff *= -1
                arg_ *= -1
            arg_ = expand_mul(arg_, deep=False)
            r_, i_ = arg_.as_independent(I, as_Add=True)
            i_ = i_.as_coefficient(I)
            if coeff.is_real and i_ and i_.is_real and r_.is_real:
                if r_.is_zero:
                    if i_.is_positive:
                        return pi * I * S.Half + cls(coeff * i_)
                    elif i_.is_negative:
                        return -pi * I * S.Half + cls(coeff * -i_)
                else:
                    from sympy.simplify import ratsimp
                    # Check for arguments involving rational multiples of pi
                    t = (i_/r_).cancel()
                    t1 = (-t).cancel()
                    atan_table = _log_atan_table()
                    if t in atan_table:
                        modulus = ratsimp(coeff * Abs(arg_))
                        if r_.is_positive:
                            return cls(modulus) + I * atan_table[t]
                        else:
                            return cls(modulus) + I * (atan_table[t] - pi)
                    elif t1 in atan_table:
                        modulus = ratsimp(coeff * Abs(arg_))
                        if r_.is_positive:
                            return cls(modulus) + I * (-atan_table[t1])
                        else:
                            return cls(modulus) + I * (pi - atan_table[t1])

    @staticmethod
    @cacheit
    def taylor_term(n, x, *previous_terms):  # of log(1+x)
        r"""
        Returns the next term in the Taylor series expansion of `\log(1+x)`.
        """
        from sympy.simplify.powsimp import powsimp
        if n < 0:
            return S.Zero
        x = sympify(x)
        if n == 0:
            return x
        if previous_terms:
            p = previous_terms[-1]
            if p is not None:
                return powsimp((-n) * p * x / (n + 1), deep=True, combine='exp')
        return (1 - 2*(n % 2)) * x**(n + 1)/(n + 1)

    def _eval_expand_log(self, deep=True, **hints):
        from sympy.concrete import Sum, Product
        force = hints.get('force', False)
        factor = hints.get('factor', False)
        if (len(self.args) == 2):
            return expand_log(self.func(*self.args), deep=deep, force=force)
        arg = self.args[0]
        if arg.is_Integer:
            # remove perfect powers
            p = perfect_power(arg)
            logarg = None
            coeff = 1
            if p is not False:
                arg, coeff = p
                logarg = self.func(arg)
            # expand as product of its prime factors if factor=True
            if factor:
                p = factorint(arg)
                if arg not in p.keys():
                    logarg = sum(n*log(val) for val, n in p.items())
            if logarg is not None:
                return coeff*logarg
        elif arg.is_Rational:
            return log(arg.p) - log(arg.q)
        elif arg.is_Mul:
            expr = []
            nonpos = []
            for x in arg.args:
                if force or x.is_positive or x.is_polar:
                    a = self.func(x)
                    if isinstance(a, log):
                        expr.append(self.func(x)._eval_expand_log(**hints))
                    else:
                        expr.append(a)
                elif x.is_negative:
                    a = self.func(-x)
                    expr.append(a)
                    nonpos.append(S.NegativeOne)
                else:
                    nonpos.append(x)
            return Add(*expr) + log(Mul(*nonpos))
        elif arg.is_Pow or isinstance(arg, exp):
            if force or (arg.exp.is_extended_real and (arg.base.is_positive or ((arg.exp+1)
                .is_positive and (arg.exp-1).is_nonpositive))) or arg.base.is_polar:
                b = arg.base
                e = arg.exp
                a = self.func(b)
                if isinstance(a, log):
                    return unpolarify(e) * a._eval_expand_log(**hints)
                else:
                    return unpolarify(e) * a
        elif isinstance(arg, Product):
            if force or arg.function.is_positive:
                return Sum(log(arg.function), *arg.limits)

        return self.func(arg)

    def _eval_simplify(self, **kwargs):
        from sympy.simplify.simplify import expand_log, simplify, inversecombine
        if len(self.args) == 2:  # it's unevaluated
            return simplify(self.func(*self.args), **kwargs)

        expr = self.func(simplify(self.args[0], **kwargs))
        if kwargs['inverse']:
            expr = inversecombine(expr)
        expr = expand_log(expr, deep=True)
        return min([expr, self], key=kwargs['measure'])

    def as_real_imag(self, deep=True, **hints):
        """
        Returns this function as a complex coordinate.

        Examples
        ========

        >>> from sympy import I, log
        >>> from sympy.abc import x
        >>> log(x).as_real_imag()
        (log(Abs(x)), arg(x))
        >>> log(I).as_real_imag()
        (0, pi/2)
        >>> log(1 + I).as_real_imag()
        (log(sqrt(2)), pi/4)
        >>> log(I*x).as_real_imag()
        (log(Abs(x)), arg(I*x))

        """
        sarg = self.args[0]
        if deep:
            sarg = self.args[0].expand(deep, **hints)
        sarg_abs = Abs(sarg)
        if sarg_abs == sarg:
            return self, S.Zero
        sarg_arg = arg(sarg)
        if hints.get('log', False):  # Expand the log
            hints['complex'] = False
            return (log(sarg_abs).expand(deep, **hints), sarg_arg)
        else:
            return log(sarg_abs), sarg_arg

    def _eval_is_rational(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if (self.args[0] - 1).is_zero:
                return True
            if s.args[0].is_rational and fuzzy_not((self.args[0] - 1).is_zero):
                return False
        else:
            return s.is_rational

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if (self.args[0] - 1).is_zero:
                return True
            elif fuzzy_not((self.args[0] - 1).is_zero):
                if self.args[0].is_algebraic:
                    return False
        else:
            return s.is_algebraic

    def _eval_is_extended_real(self):
        return self.args[0].is_extended_positive

    def _eval_is_complex(self):
        z = self.args[0]
        return fuzzy_and([z.is_complex, fuzzy_not(z.is_zero)])

    def _eval_is_finite(self):
        arg = self.args[0]
        if arg.is_zero:
            return False
        return arg.is_finite

    def _eval_is_extended_positive(self):
        return (self.args[0] - 1).is_extended_positive

    def _eval_is_zero(self):
        return (self.args[0] - 1).is_zero

    def _eval_is_extended_nonnegative(self):
        return (self.args[0] - 1).is_extended_nonnegative

    def _eval_nseries(self, x, n, logx, cdir=0):
        # NOTE Please see the comment at the beginning of this file, labelled
        #      IMPORTANT.
        from sympy.series.order import Order
        from sympy.simplify.simplify import logcombine
        from sympy.core.symbol import Dummy

        if self.args[0] == x:
            return log(x) if logx is None else logx
        arg = self.args[0]
        t = Dummy('t', positive=True)
        if cdir == 0:
            cdir = 1
        z = arg.subs(x, cdir*t)

        k, l = Wild("k"), Wild("l")
        r = z.match(k*t**l)
        if r is not None:
            k, l = r[k], r[l]
            if l != 0 and not l.has(t) and not k.has(t):
                r = l*log(x) if logx is None else l*logx
                r += log(k) - l*log(cdir) # XXX true regardless of assumptions?
                return r

        def coeff_exp(term, x):
            coeff, exp = S.One, S.Zero
            for factor in Mul.make_args(term):
                if factor.has(x):
                    base, exp = factor.as_base_exp()
                    if base != x:
                        try:
                            return term.leadterm(x)
                        except ValueError:
                            return term, S.Zero
                else:
                    coeff *= factor
            return coeff, exp

        # TODO new and probably slow
        try:
            a, b = z.leadterm(t, logx=logx, cdir=1)
        except (ValueError, NotImplementedError, PoleError):
            s = z._eval_nseries(t, n=n, logx=logx, cdir=1)
            while s.is_Order:
                n += 1
                s = z._eval_nseries(t, n=n, logx=logx, cdir=1)
            try:
                a, b = s.removeO().leadterm(t, cdir=1)
            except ValueError:
                a, b = s.removeO().as_leading_term(t, cdir=1), S.Zero

        p = (z/(a*t**b) - 1).cancel()._eval_nseries(t, n=n, logx=logx, cdir=1)
        if p.has(exp):
            p = logcombine(p)
        if isinstance(p, Order):
            n = p.getn()
        _, d = coeff_exp(p, t)
        logx = log(x) if logx is None else logx

        if not d.is_positive:
            res = log(a) - b*log(cdir) + b*logx
            _res = res
            logflags = {"deep": True, "log": True, "mul": False, "power_exp": False,
                "power_base": False, "multinomial": False, "basic": False, "force": True,
                "factor": False}
            expr = self.expand(**logflags)
            if (not a.could_extract_minus_sign() and
                logx.could_extract_minus_sign()):
                _res = _res.subs(-logx, -log(x)).expand(**logflags)
            else:
                _res = _res.subs(logx, log(x)).expand(**logflags)
            if _res == expr:
                return res
            return res + Order(x**n, x)

        def mul(d1, d2):
            res = {}
            for e1, e2 in product(d1, d2):
                ex = e1 + e2
                if ex < n:
                    res[ex] = res.get(ex, S.Zero) + d1[e1]*d2[e2]
            return res

        pterms = {}

        for term in Add.make_args(p.removeO()):
            co1, e1 = coeff_exp(term, t)
            pterms[e1] = pterms.get(e1, S.Zero) + co1

        k = S.One
        terms = {}
        pk = pterms

        while k*d < n:
            coeff = -S.NegativeOne**k/k
            for ex in pk:
                terms[ex] = terms.get(ex, S.Zero) + coeff*pk[ex]
            pk = mul(pk, pterms)
            k += S.One

        res = log(a) - b*log(cdir) + b*logx
        for ex in terms:
            res += terms[ex].cancel()*t**(ex)

        if a.is_negative and im(z) != 0:
            from sympy.functions.special.delta_functions import Heaviside
            for i, term in enumerate(z.lseries(t)):
                if not term.is_real or i == 5:
                    break
            if i < 5:
                coeff, _ = term.as_coeff_exponent(t)
                res += -2*I*pi*Heaviside(-im(coeff), 0)

        res = res.subs(t, x/cdir)
        return res + Order(x**n, x)

    def _eval_as_leading_term(self, x, logx, cdir):
        # NOTE
        # Refer https://github.com/sympy/sympy/pull/23592 for more information
        # on each of the following steps involved in this method.
        arg0 = self.args[0].together()

        # STEP 1
        t = Dummy('t', positive=True)
        if cdir == 0:
            cdir = 1
        z = arg0.subs(x, cdir*t)

        # STEP 2
        try:
            c, e = z.leadterm(t, logx=logx, cdir=1)
        except ValueError:
            arg = arg0.as_leading_term(x, logx=logx, cdir=cdir)
            return log(arg)
        if c.has(t):
            c = c.subs(t, x/cdir)
            if e != 0:
                raise PoleError("Cannot expand %s around 0" % (self))
            return log(c)

        # STEP 3
        if c == S.One and e == S.Zero:
            return (arg0 - S.One).as_leading_term(x, logx=logx)

        # STEP 4
        res = log(c) - e*log(cdir)
        logx = log(x) if logx is None else logx
        res += e*logx

        # STEP 5
        if c.is_negative and im(z) != 0:
            from sympy.functions.special.delta_functions import Heaviside
            for i, term in enumerate(z.lseries(t)):
                if not term.is_real or i == 5:
                    break
            if i < 5:
                coeff, _ = term.as_coeff_exponent(t)
                res += -2*I*pi*Heaviside(-im(coeff), 0)
        return res


class LambertW(DefinedFunction):
    r"""
    The Lambert W function $W(z)$ is defined as the inverse
    function of $w \exp(w)$ [1]_.

    Explanation
    ===========

    In other words, the value of $W(z)$ is such that $z = W(z) \exp(W(z))$
    for any complex number $z$.  The Lambert W function is a multivalued
    function with infinitely many branches $W_k(z)$, indexed by
    $k \in \mathbb{Z}$.  Each branch gives a different solution $w$
    of the equation $z = w \exp(w)$.

    The Lambert W function has two partially real branches: the
    principal branch ($k = 0$) is real for real $z > -1/e$, and the
    $k = -1$ branch is real for $-1/e < z < 0$. All branches except
    $k = 0$ have a logarithmic singularity at $z = 0$.

    Examples
    ========

    >>> from sympy import LambertW
    >>> LambertW(1.2)
    0.635564016364870
    >>> LambertW(1.2, -1).n()
    -1.34747534407696 - 4.41624341514535*I
    >>> LambertW(-1).is_real
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Lambert_W_function
    """
    _singularities = (-Pow(S.Exp1, -1, evaluate=False), S.ComplexInfinity)

    @classmethod
    def eval(cls, x, k=None):
        if k == S.Zero:
            return cls(x)
        elif k is None:
            k = S.Zero

        if k.is_zero:
            if x.is_zero:
                return S.Zero
            if x is S.Exp1:
                return S.One
            if x == -1/S.Exp1:
                return S.NegativeOne
            if x == -log(2)/2:
                return -log(2)
            if x == 2*log(2):
                return log(2)
            if x == -pi/2:
                return I*pi/2
            if x == exp(1 + S.Exp1):
                return S.Exp1
            if x is S.Infinity:
                return S.Infinity

        if fuzzy_not(k.is_zero):
            if x.is_zero:
                return S.NegativeInfinity
        if k is S.NegativeOne:
            if x == -pi/2:
                return -I*pi/2
            elif x == -1/S.Exp1:
                return S.NegativeOne
            elif x == -2*exp(-2):
                return -Integer(2)

    def fdiff(self, argindex=1):
        """
        Return the first derivative of this function.
        """
        x = self.args[0]

        if len(self.args) == 1:
            if argindex == 1:
                return LambertW(x)/(x*(1 + LambertW(x)))
        else:
            k = self.args[1]
            if argindex == 1:
                return LambertW(x, k)/(x*(1 + LambertW(x, k)))

        raise ArgumentIndexError(self, argindex)

    def _eval_is_extended_real(self):
        x = self.args[0]
        if len(self.args) == 1:
            k = S.Zero
        else:
            k = self.args[1]
        if k.is_zero:
            if (x + 1/S.Exp1).is_positive:
                return True
            elif (x + 1/S.Exp1).is_nonpositive:
                return False
        elif (k + 1).is_zero:
            if x.is_negative and (x + 1/S.Exp1).is_positive:
                return True
            elif x.is_nonpositive or (x + 1/S.Exp1).is_nonnegative:
                return False
        elif fuzzy_not(k.is_zero) and fuzzy_not((k + 1).is_zero):
            if x.is_extended_real:
                return False

    def _eval_is_finite(self):
        return self.args[0].is_finite

    def _eval_is_algebraic(self):
        s = self.func(*self.args)
        if s.func == self.func:
            if fuzzy_not(self.args[0].is_zero) and self.args[0].is_algebraic:
                return False
        else:
            return s.is_algebraic

    def _eval_as_leading_term(self, x, logx, cdir):
        if len(self.args) == 1:
            arg = self.args[0]
            arg0 = arg.subs(x, 0).cancel()
            if not arg0.is_zero:
                return self.func(arg0)
            return arg.as_leading_term(x)

    def _eval_nseries(self, x, n, logx, cdir=0):
        if len(self.args) == 1:
            from sympy.functions.elementary.integers import ceiling
            from sympy.series.order import Order
            arg = self.args[0].nseries(x, n=n, logx=logx)
            lt = arg.as_leading_term(x, logx=logx)
            lte = 1
            if lt.is_Pow:
                lte = lt.exp
            if ceiling(n/lte) >= 1:
                s = Add(*[(-S.One)**(k - 1)*Integer(k)**(k - 2)/
                          factorial(k - 1)*arg**k for k in range(1, ceiling(n/lte))])
                s = expand_multinomial(s)
            else:
                s = S.Zero

            return s + Order(x**n, x)
        return super()._eval_nseries(x, n, logx)

    def _eval_is_zero(self):
        x = self.args[0]
        if len(self.args) == 1:
            return x.is_zero
        else:
            return fuzzy_and([x.is_zero, self.args[1].is_zero])


@cacheit
def _log_atan_table():
    return {
        # first quadrant only
        sqrt(3): pi / 3,
        1: pi / 4,
        sqrt(5 - 2 * sqrt(5)): pi / 5,
        sqrt(2) * sqrt(5 - sqrt(5)) / (1 + sqrt(5)): pi / 5,
        sqrt(5 + 2 * sqrt(5)): pi * Rational(2, 5),
        sqrt(2) * sqrt(sqrt(5) + 5) / (-1 + sqrt(5)): pi * Rational(2, 5),
        sqrt(3) / 3: pi / 6,
        sqrt(2) - 1: pi / 8,
        sqrt(2 - sqrt(2)) / sqrt(sqrt(2) + 2): pi / 8,
        sqrt(2) + 1: pi * Rational(3, 8),
        sqrt(sqrt(2) + 2) / sqrt(2 - sqrt(2)): pi * Rational(3, 8),
        sqrt(1 - 2 * sqrt(5) / 5): pi / 10,
        (-sqrt(2) + sqrt(10)) / (2 * sqrt(sqrt(5) + 5)): pi / 10,
        sqrt(1 + 2 * sqrt(5) / 5): pi * Rational(3, 10),
        (sqrt(2) + sqrt(10)) / (2 * sqrt(5 - sqrt(5))): pi * Rational(3, 10),
        2 - sqrt(3): pi / 12,
        (-1 + sqrt(3)) / (1 + sqrt(3)): pi / 12,
        2 + sqrt(3): pi * Rational(5, 12),
        (1 + sqrt(3)) / (-1 + sqrt(3)): pi * Rational(5, 12)
    }