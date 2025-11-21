
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\stochastic_process_types.py ===
from __future__ import annotations
import random
import itertools
from typing import Sequence as tSequence
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.intfunc import igcd
from sympy.core.numbers import (Integer, Rational, oo, pi)
from sympy.core.relational import (Eq, Ge, Gt, Le, Lt, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.functions.combinatorial.factorials import factorial
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.integers import ceiling
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.gamma_functions import gamma
from sympy.logic.boolalg import (And, Not, Or)
from sympy.matrices.exceptions import NonSquareMatrixError
from sympy.matrices.dense import (Matrix, eye, ones, zeros)
from sympy.matrices.expressions.blockmatrix import BlockMatrix
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.special import Identity
from sympy.matrices.immutable import ImmutableMatrix
from sympy.sets.conditionset import ConditionSet
from sympy.sets.contains import Contains
from sympy.sets.fancysets import Range
from sympy.sets.sets import (FiniteSet, Intersection, Interval, Set, Union)
from sympy.solvers.solveset import linsolve
from sympy.tensor.indexed import (Indexed, IndexedBase)
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import strongly_connected_components
from sympy.stats.joint_rv import JointDistribution
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.rv import (RandomIndexedSymbol, random_symbols, RandomSymbol,
                            _symbol_converter, _value_check, pspace, given,
                           dependent, is_random, sample_iter, Distribution,
                           Density)
from sympy.stats.stochastic_process import StochasticPSpace
from sympy.stats.symbolic_probability import Probability, Expectation
from sympy.stats.frv_types import Bernoulli, BernoulliDistribution, FiniteRV
from sympy.stats.drv_types import Poisson, PoissonDistribution
from sympy.stats.crv_types import Normal, NormalDistribution, Gamma, GammaDistribution
from sympy.core.sympify import _sympify, sympify

EmptySet = S.EmptySet

__all__ = [
    'StochasticProcess',
    'DiscreteTimeStochasticProcess',
    'DiscreteMarkovChain',
    'TransitionMatrixOf',
    'StochasticStateSpaceOf',
    'GeneratorMatrixOf',
    'ContinuousMarkovChain',
    'BernoulliProcess',
    'PoissonProcess',
    'WienerProcess',
    'GammaProcess'
]


@is_random.register(Indexed)
def _(x):
    return is_random(x.base)

@is_random.register(RandomIndexedSymbol)  # type: ignore
def _(x):
    return True

def _set_converter(itr):
    """
    Helper function for converting list/tuple/set to Set.
    If parameter is not an instance of list/tuple/set then
    no operation is performed.

    Returns
    =======

    Set
        The argument converted to Set.


    Raises
    ======

    TypeError
        If the argument is not an instance of list/tuple/set.
    """
    if isinstance(itr, (list, tuple, set)):
        itr = FiniteSet(*itr)
    if not isinstance(itr, Set):
        raise TypeError("%s is not an instance of list/tuple/set."%(itr))
    return itr

def _state_converter(itr: tSequence) -> Tuple | Range:
    """
    Helper function for converting list/tuple/set/Range/Tuple/FiniteSet
    to tuple/Range.
    """
    itr_ret: Tuple | Range

    if isinstance(itr, (Tuple, set, FiniteSet)):
        itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))

    elif isinstance(itr, (list, tuple)):
        # check if states are unique
        if len(set(itr)) != len(itr):
            raise ValueError('The state space must have unique elements.')
        itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))

    elif isinstance(itr, Range):
        # the only ordered set in SymPy I know of
        # try to convert to tuple
        try:
            itr_ret = Tuple(*(sympify(i) if isinstance(i, str) else i for i in itr))
        except (TypeError, ValueError):
            itr_ret = itr

    else:
        raise TypeError("%s is not an instance of list/tuple/set/Range/Tuple/FiniteSet." % (itr))
    return itr_ret

def _sym_sympify(arg):
    """
    Converts an arbitrary expression to a type that can be used inside SymPy.
    As generally strings are unwise to use in the expressions,
    it returns the Symbol of argument if the string type argument is passed.

    Parameters
    =========

    arg: The parameter to be converted to be used in SymPy.

    Returns
    =======

    The converted parameter.

    """
    if isinstance(arg, str):
        return Symbol(arg)
    else:
        return _sympify(arg)

def _matrix_checks(matrix):
    if not isinstance(matrix, (Matrix, MatrixSymbol, ImmutableMatrix)):
        raise TypeError("Transition probabilities either should "
                            "be a Matrix or a MatrixSymbol.")
    if matrix.shape[0] != matrix.shape[1]:
        raise NonSquareMatrixError("%s is not a square matrix"%(matrix))
    if isinstance(matrix, Matrix):
        matrix = ImmutableMatrix(matrix.tolist())
    return matrix

class StochasticProcess(Basic):
    """
    Base class for all the stochastic processes whether
    discrete or continuous.

    Parameters
    ==========

    sym: Symbol or str
    state_space: Set
        The state space of the stochastic process, by default S.Reals.
        For discrete sets it is zero indexed.

    See Also
    ========

    DiscreteTimeStochasticProcess
    """

    index_set = S.Reals

    def __new__(cls, sym, state_space=S.Reals, **kwargs):
        sym = _symbol_converter(sym)
        state_space = _set_converter(state_space)
        return Basic.__new__(cls, sym, state_space)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def state_space(self) -> FiniteSet | Range:
        if not isinstance(self.args[1], (FiniteSet, Range)):
            assert isinstance(self.args[1], Tuple)
            return FiniteSet(*self.args[1])
        return self.args[1]

    def _deprecation_warn_distribution(self):
        sympy_deprecation_warning(
            """
            Calling the distribution method with a RandomIndexedSymbol
            argument, like X.distribution(X(t)) is deprecated. Instead, call
            distribution() with the given timestamp, like

            X.distribution(t)
            """,
            deprecated_since_version="1.7.1",
            active_deprecations_target="deprecated-distribution-randomindexedsymbol",
            stacklevel=4,
        )

    def distribution(self, key=None):
        if key is None:
            self._deprecation_warn_distribution()
        return Distribution()

    def density(self, x):
        return Density()

    def __call__(self, time):
        """
        Overridden in ContinuousTimeStochasticProcess.
        """
        raise NotImplementedError("Use [] for indexing discrete time stochastic process.")

    def __getitem__(self, time):
        """
        Overridden in DiscreteTimeStochasticProcess.
        """
        raise NotImplementedError("Use () for indexing continuous time stochastic process.")

    def probability(self, condition):
        raise NotImplementedError()

    def joint_distribution(self, *args):
        """
        Computes the joint distribution of the random indexed variables.

        Parameters
        ==========

        args: iterable
            The finite list of random indexed variables/the key of a stochastic
            process whose joint distribution has to be computed.

        Returns
        =======

        JointDistribution
            The joint distribution of the list of random indexed variables.
            An unevaluated object is returned if it is not possible to
            compute the joint distribution.

        Raises
        ======

        ValueError: When the arguments passed are not of type RandomIndexSymbol
        or Number.
        """
        args = list(args)
        for i, arg in enumerate(args):
            if S(arg).is_Number:
                if self.index_set.is_subset(S.Integers):
                    args[i] = self.__getitem__(arg)
                else:
                    args[i] = self.__call__(arg)
            elif not isinstance(arg, RandomIndexedSymbol):
                raise ValueError("Expected a RandomIndexedSymbol or "
                                "key not  %s"%(type(arg)))

        if args[0].pspace.distribution == Distribution():
            return JointDistribution(*args)
        density = Lambda(tuple(args),
                         expr=Mul.fromiter(arg.pspace.process.density(arg) for arg in args))
        return JointDistributionHandmade(density)

    def expectation(self, condition, given_condition):
        raise NotImplementedError("Abstract method for expectation queries.")

    def sample(self):
        raise NotImplementedError("Abstract method for sampling queries.")

class DiscreteTimeStochasticProcess(StochasticProcess):
    """
    Base class for all discrete stochastic processes.
    """
    def __getitem__(self, time):
        """
        For indexing discrete time stochastic processes.

        Returns
        =======

        RandomIndexedSymbol
        """
        time = sympify(time)
        if not time.is_symbol and time not in self.index_set:
            raise IndexError("%s is not in the index set of %s"%(time, self.symbol))
        idx_obj = Indexed(self.symbol, time)
        pspace_obj = StochasticPSpace(self.symbol, self, self.distribution(time))
        return RandomIndexedSymbol(idx_obj, pspace_obj)

class ContinuousTimeStochasticProcess(StochasticProcess):
    """
    Base class for all continuous time stochastic process.
    """
    def __call__(self, time):
        """
        For indexing continuous time stochastic processes.

        Returns
        =======

        RandomIndexedSymbol
        """
        time = sympify(time)
        if not time.is_symbol and time not in self.index_set:
            raise IndexError("%s is not in the index set of %s"%(time, self.symbol))
        func_obj = Function(self.symbol)(time)
        pspace_obj = StochasticPSpace(self.symbol, self, self.distribution(time))
        return RandomIndexedSymbol(func_obj, pspace_obj)

class TransitionMatrixOf(Boolean):
    """
    Assumes that the matrix is the transition matrix
    of the process.
    """

    def __new__(cls, process, matrix):
        if not isinstance(process, DiscreteMarkovChain):
            raise ValueError("Currently only DiscreteMarkovChain "
                                "support TransitionMatrixOf.")
        matrix = _matrix_checks(matrix)
        return Basic.__new__(cls, process, matrix)

    process = property(lambda self: self.args[0])
    matrix = property(lambda self: self.args[1])

class GeneratorMatrixOf(TransitionMatrixOf):
    """
    Assumes that the matrix is the generator matrix
    of the process.
    """

    def __new__(cls, process, matrix):
        if not isinstance(process, ContinuousMarkovChain):
            raise ValueError("Currently only ContinuousMarkovChain "
                                "support GeneratorMatrixOf.")
        matrix = _matrix_checks(matrix)
        return Basic.__new__(cls, process, matrix)

class StochasticStateSpaceOf(Boolean):

    def __new__(cls, process, state_space):
        if not isinstance(process, (DiscreteMarkovChain, ContinuousMarkovChain)):
            raise ValueError("Currently only DiscreteMarkovChain and ContinuousMarkovChain "
                                "support StochasticStateSpaceOf.")
        state_space = _state_converter(state_space)
        if isinstance(state_space, Range):
            ss_size = ceiling((state_space.stop - state_space.start) / state_space.step)
        else:
            ss_size = len(state_space)
        state_index = Range(ss_size)
        return Basic.__new__(cls, process, state_index)

    process = property(lambda self: self.args[0])
    state_index = property(lambda self: self.args[1])

class MarkovProcess(StochasticProcess):
    """
    Contains methods that handle queries
    common to Markov processes.
    """

    @property
    def number_of_states(self) -> Integer | Symbol:
        """
        The number of states in the Markov Chain.
        """
        return _sympify(self.args[2].shape[0]) # type: ignore

    @property
    def _state_index(self):
        """
        Returns state index as Range.
        """
        return self.args[1]

    @classmethod
    def _sanity_checks(cls, state_space, trans_probs):
        # Try to never have None as state_space or trans_probs.
        # This helps a lot if we get it done at the start.
        if (state_space is None) and (trans_probs is None):
            _n = Dummy('n', integer=True, nonnegative=True)
            state_space = _state_converter(Range(_n))
            trans_probs = _matrix_checks(MatrixSymbol('_T', _n, _n))

        elif state_space is None:
            trans_probs = _matrix_checks(trans_probs)
            state_space = _state_converter(Range(trans_probs.shape[0]))

        elif trans_probs is None:
            state_space = _state_converter(state_space)
            if isinstance(state_space, Range):
                _n = ceiling((state_space.stop - state_space.start) / state_space.step)
            else:
                _n = len(state_space)
            trans_probs = MatrixSymbol('_T', _n, _n)

        else:
            state_space = _state_converter(state_space)
            trans_probs = _matrix_checks(trans_probs)
            # Range object doesn't want to give a symbolic size
            # so we do it ourselves.
            if isinstance(state_space, Range):
                ss_size = ceiling((state_space.stop - state_space.start) / state_space.step)
            else:
                ss_size = len(state_space)
            if ss_size != trans_probs.shape[0]:
                raise ValueError('The size of the state space and the number of '
                                 'rows of the transition matrix must be the same.')

        return state_space, trans_probs

    def _extract_information(self, given_condition):
        """
        Helper function to extract information, like,
        transition matrix/generator matrix, state space, etc.
        """
        if isinstance(self, DiscreteMarkovChain):
            trans_probs = self.transition_probabilities
            state_index = self._state_index
        elif isinstance(self, ContinuousMarkovChain):
            trans_probs = self.generator_matrix
            state_index = self._state_index
        if isinstance(given_condition, And):
            gcs = given_condition.args
            given_condition = S.true
            for gc in gcs:
                if isinstance(gc, TransitionMatrixOf):
                    trans_probs = gc.matrix
                if isinstance(gc, StochasticStateSpaceOf):
                    state_index = gc.state_index
                if isinstance(gc, Relational):
                    given_condition = given_condition & gc
        if isinstance(given_condition, TransitionMatrixOf):
            trans_probs = given_condition.matrix
            given_condition = S.true
        if isinstance(given_condition, StochasticStateSpaceOf):
            state_index = given_condition.state_index
            given_condition = S.true
        return trans_probs, state_index, given_condition

    def _check_trans_probs(self, trans_probs, row_sum=1):
        """
        Helper function for checking the validity of transition
        probabilities.
        """
        if not isinstance(trans_probs, MatrixSymbol):
            rows = trans_probs.tolist()
            for row in rows:
                if (sum(row) - row_sum) != 0:
                    raise ValueError("Values in a row must sum to %s. "
                    "If you are using Float or floats then please use Rational."%(row_sum))

    def _work_out_state_index(self, state_index, given_condition, trans_probs):
        """
        Helper function to extract state space if there
        is a random symbol in the given condition.
        """
        # if given condition is None, then there is no need to work out
        # state_space from random variables
        if given_condition != None:
            rand_var = list(given_condition.atoms(RandomSymbol) -
                        given_condition.atoms(RandomIndexedSymbol))
            if len(rand_var) == 1:
                state_index = rand_var[0].pspace.set

        # `not None` is `True`. So the old test fails for symbolic sizes.
        # Need to build the statement differently.
        sym_cond = not self.number_of_states.is_Integer
        cond1 = not sym_cond and len(state_index) != trans_probs.shape[0]
        if cond1:
            raise ValueError("state space is not compatible with the transition probabilities.")
        if not isinstance(trans_probs.shape[0], Symbol):
            state_index = FiniteSet(*range(trans_probs.shape[0]))
        return state_index

    @cacheit
    def _preprocess(self, given_condition, evaluate):
        """
        Helper function for pre-processing the information.
        """
        is_insufficient = False

        if not evaluate: # avoid pre-processing if the result is not to be evaluated
            return (True, None, None, None)

        # extracting transition matrix and state space
        trans_probs, state_index, given_condition = self._extract_information(given_condition)

        # given_condition does not have sufficient information
        # for computations
        if trans_probs is None or \
            given_condition is None:
            is_insufficient = True
        else:
            # checking transition probabilities
            if isinstance(self, DiscreteMarkovChain):
                self._check_trans_probs(trans_probs, row_sum=1)
            elif isinstance(self, ContinuousMarkovChain):
                self._check_trans_probs(trans_probs, row_sum=0)

            # working out state space
            state_index = self._work_out_state_index(state_index, given_condition, trans_probs)

        return is_insufficient, trans_probs, state_index, given_condition

    def replace_with_index(self, condition):
        if isinstance(condition, Relational):
            lhs, rhs = condition.lhs, condition.rhs
            if not isinstance(lhs, RandomIndexedSymbol):
                lhs, rhs = rhs, lhs
            condition = type(condition)(self.index_of.get(lhs, lhs),
                                        self.index_of.get(rhs, rhs))
        return condition

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Handles probability queries for Markov process.

        Parameters
        ==========

        condition: Relational
        given_condition: Relational/And

        Returns
        =======
        Probability
            If the information is not sufficient.
        Expr
            In all other cases.

        Note
        ====
        Any information passed at the time of query overrides
        any information passed at the time of object creation like
        transition probabilities, state space.
        Pass the transition matrix using TransitionMatrixOf,
        generator matrix using GeneratorMatrixOf and state space
        using StochasticStateSpaceOf in given_condition using & or And.
        """
        check, mat, state_index, new_given_condition = \
            self._preprocess(given_condition, evaluate)

        rv = list(condition.atoms(RandomIndexedSymbol))
        symbolic = False
        for sym in rv:
            if sym.key.is_symbol:
                symbolic = True
                break

        if check:
            return Probability(condition, new_given_condition)

        if isinstance(self, ContinuousMarkovChain):
            trans_probs = self.transition_probabilities(mat)
        elif isinstance(self, DiscreteMarkovChain):
            trans_probs = mat
        condition = self.replace_with_index(condition)
        given_condition = self.replace_with_index(given_condition)
        new_given_condition = self.replace_with_index(new_given_condition)

        if isinstance(condition, Relational):
            if isinstance(new_given_condition, And):
                gcs = new_given_condition.args
            else:
                gcs = (new_given_condition, )
            min_key_rv = list(new_given_condition.atoms(RandomIndexedSymbol))

            if len(min_key_rv):
                min_key_rv = min_key_rv[0]
                for r in rv:
                    if min_key_rv.key.is_symbol or r.key.is_symbol:
                        continue
                    if min_key_rv.key > r.key:
                        return Probability(condition)
            else:
                min_key_rv = None
                return Probability(condition)

            if symbolic:
                return self._symbolic_probability(condition, new_given_condition, rv, min_key_rv)

            if len(rv) > 1:
                rv[0] = condition.lhs
                rv[1] = condition.rhs
                if rv[0].key < rv[1].key:
                        rv[0], rv[1] = rv[1], rv[0]
                        if isinstance(condition, Gt):
                            condition = Lt(condition.lhs, condition.rhs)
                        elif isinstance(condition, Lt):
                            condition = Gt(condition.lhs, condition.rhs)
                        elif isinstance(condition, Ge):
                            condition = Le(condition.lhs, condition.rhs)
                        elif isinstance(condition, Le):
                            condition = Ge(condition.lhs, condition.rhs)
                s = Rational(0, 1)
                n = len(self.state_space)

                if isinstance(condition, (Eq, Ne)):
                    for i in range(0, n):
                        s += self.probability(Eq(rv[0], i), Eq(rv[1], i)) * self.probability(Eq(rv[1], i), new_given_condition)
                    return s if isinstance(condition, Eq) else 1 - s
                else:
                    upper = 0
                    greater = False
                    if isinstance(condition, (Ge, Lt)):
                        upper = 1
                    if isinstance(condition, (Ge, Gt)):
                        greater = True

                    for i in range(0, n):
                        if i <= n//2:
                            for j in range(0, i + upper):
                                s += self.probability(Eq(rv[0], i), Eq(rv[1], j)) * self.probability(Eq(rv[1], j), new_given_condition)
                        else:
                            s += self.probability(Eq(rv[0], i), new_given_condition)
                            for j in range(i + upper, n):
                                s -= self.probability(Eq(rv[0], i), Eq(rv[1], j)) * self.probability(Eq(rv[1], j), new_given_condition)
                    return s if greater else 1 - s

            rv = rv[0]
            states = condition.as_set()
            prob, gstate = {}, None
            for gc in gcs:
                if gc.has(min_key_rv):
                    if gc.has(Probability):
                        p, gp = (gc.rhs, gc.lhs) if isinstance(gc.lhs, Probability) \
                                    else (gc.lhs, gc.rhs)
                        gr = gp.args[0]
                        gset = Intersection(gr.as_set(), state_index)
                        gstate = list(gset)[0]
                        prob[gset] = p
                    else:
                        _, gstate = (gc.lhs.key, gc.rhs) if isinstance(gc.lhs, RandomIndexedSymbol) \
                                    else (gc.rhs.key, gc.lhs)

            if not all(k in self.index_set for k in (rv.key, min_key_rv.key)):
                raise IndexError("The timestamps of the process are not in it's index set.")
            states = Intersection(states, state_index) if not isinstance(self.number_of_states, Symbol) else states
            for state in Union(states, FiniteSet(gstate)):
                if not state.is_Integer or Ge(state, mat.shape[0]) is True:
                    raise IndexError("No information is available for (%s, %s) in "
                        "transition probabilities of shape, (%s, %s). "
                        "State space is zero indexed."
                        %(gstate, state, mat.shape[0], mat.shape[1]))
            if prob:
                gstates = Union(*prob.keys())
                if len(gstates) == 1:
                    gstate = list(gstates)[0]
                    gprob = list(prob.values())[0]
                    prob[gstates] = gprob
                elif len(gstates) == len(state_index) - 1:
                    gstate = list(state_index - gstates)[0]
                    gprob = S.One - sum(prob.values())
                    prob[state_index - gstates] = gprob
                else:
                    raise ValueError("Conflicting information.")
            else:
                gprob = S.One

            if min_key_rv == rv:
                return sum(prob[FiniteSet(state)] for state in states)
            if isinstance(self, ContinuousMarkovChain):
                return gprob * sum(trans_probs(rv.key - min_key_rv.key).__getitem__((gstate, state))
                                    for state in states)
            if isinstance(self, DiscreteMarkovChain):
                return gprob * sum((trans_probs**(rv.key - min_key_rv.key)).__getitem__((gstate, state))
                                    for state in states)

        if isinstance(condition, Not):
            expr = condition.args[0]
            return S.One - self.probability(expr, given_condition, evaluate, **kwargs)

        if isinstance(condition, And):
            compute_later, state2cond, conds = [], {}, condition.args
            for expr in conds:
                if isinstance(expr, Relational):
                    ris = list(expr.atoms(RandomIndexedSymbol))[0]
                    if state2cond.get(ris, None) is None:
                        state2cond[ris] = S.true
                    state2cond[ris] &= expr
                else:
                    compute_later.append(expr)
            ris = []
            for ri in state2cond:
                ris.append(ri)
                cset = Intersection(state2cond[ri].as_set(), state_index)
                if len(cset) == 0:
                    return S.Zero
                state2cond[ri] = cset.as_relational(ri)
            sorted_ris = sorted(ris, key=lambda ri: ri.key)
            prod = self.probability(state2cond[sorted_ris[0]], given_condition, evaluate, **kwargs)
            for i in range(1, len(sorted_ris)):
                ri, prev_ri = sorted_ris[i], sorted_ris[i-1]
                if not isinstance(state2cond[ri], Eq):
                    raise ValueError("The process is in multiple states at %s, unable to determine the probability."%(ri))
                mat_of = TransitionMatrixOf(self, mat) if isinstance(self, DiscreteMarkovChain) else GeneratorMatrixOf(self, mat)
                prod *= self.probability(state2cond[ri], state2cond[prev_ri]
                                 & mat_of
                                 & StochasticStateSpaceOf(self, state_index),
                                 evaluate, **kwargs)
            for expr in compute_later:
                prod *= self.probability(expr, given_condition, evaluate, **kwargs)
            return prod

        if isinstance(condition, Or):
            return sum(self.probability(expr, given_condition, evaluate, **kwargs)
                        for expr in condition.args)

        raise NotImplementedError("Mechanism for handling (%s, %s) queries hasn't been "
                                "implemented yet."%(condition, given_condition))

    def _symbolic_probability(self, condition, new_given_condition, rv, min_key_rv):
        #Function to calculate probability for queries with symbols
        if isinstance(condition, Relational):
            curr_state = new_given_condition.rhs if isinstance(new_given_condition.lhs, RandomIndexedSymbol) \
                    else new_given_condition.lhs
            next_state = condition.rhs if isinstance(condition.lhs, RandomIndexedSymbol) \
                else condition.lhs

            if isinstance(condition, (Eq, Ne)):
                if isinstance(self, DiscreteMarkovChain):
                    P = self.transition_probabilities**(rv[0].key - min_key_rv.key)
                else:
                    P = exp(self.generator_matrix*(rv[0].key - min_key_rv.key))
                prob = P[curr_state, next_state] if isinstance(condition, Eq) else 1 - P[curr_state, next_state]
                return Piecewise((prob, rv[0].key > min_key_rv.key), (Probability(condition), True))
            else:
                upper = 1
                greater = False
                if isinstance(condition, (Ge, Lt)):
                    upper = 0
                if isinstance(condition, (Ge, Gt)):
                    greater = True
                k = Dummy('k')
                condition = Eq(condition.lhs, k) if isinstance(condition.lhs, RandomIndexedSymbol)\
                    else Eq(condition.rhs, k)
                total = Sum(self.probability(condition, new_given_condition), (k, next_state + upper, self.state_space._sup))
                return Piecewise((total, rv[0].key > min_key_rv.key), (Probability(condition), True)) if greater\
                    else Piecewise((1 - total, rv[0].key > min_key_rv.key), (Probability(condition), True))
        else:
            return Probability(condition, new_given_condition)

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Handles expectation queries for markov process.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation
            Unevaluated object if computations cannot be done due to
            insufficient information.
        Expr
            In all other cases when the computations are successful.

        Note
        ====

        Any information passed at the time of query overrides
        any information passed at the time of object creation like
        transition probabilities, state space.

        Pass the transition matrix using TransitionMatrixOf,
        generator matrix using GeneratorMatrixOf and state space
        using StochasticStateSpaceOf in given_condition using & or And.
        """

        check, mat, state_index, condition = \
            self._preprocess(condition, evaluate)

        if check:
            return Expectation(expr, condition)

        rvs = random_symbols(expr)
        if isinstance(expr, Expr) and isinstance(condition, Eq) \
            and len(rvs) == 1:
            # handle queries similar to E(f(X[i]), Eq(X[i-m], <some-state>))
            condition=self.replace_with_index(condition)
            state_index=self.replace_with_index(state_index)
            rv = list(rvs)[0]
            lhsg, rhsg = condition.lhs, condition.rhs
            if not isinstance(lhsg, RandomIndexedSymbol):
                lhsg, rhsg = (rhsg, lhsg)
            if rhsg not in state_index:
                raise ValueError("%s state is not in the state space."%(rhsg))
            if rv.key < lhsg.key:
                raise ValueError("Incorrect given condition is given, expectation "
                    "time %s < time %s"%(rv.key, rv.key))
            mat_of = TransitionMatrixOf(self, mat) if isinstance(self, DiscreteMarkovChain) else GeneratorMatrixOf(self, mat)
            cond = condition & mat_of & \
                    StochasticStateSpaceOf(self, state_index)
            func = lambda s: self.probability(Eq(rv, s), cond) * expr.subs(rv, self._state_index[s])
            return sum(func(s) for s in state_index)

        raise NotImplementedError("Mechanism for handling (%s, %s) queries hasn't been "
                                "implemented yet."%(expr, condition))

class DiscreteMarkovChain(DiscreteTimeStochasticProcess, MarkovProcess):
    """
    Represents a finite discrete time-homogeneous Markov chain.

    This type of Markov Chain can be uniquely characterised by
    its (ordered) state space and its one-step transition probability
    matrix.

    Parameters
    ==========

    sym:
        The name given to the Markov Chain
    state_space:
        Optional, by default, Range(n)
    trans_probs:
        Optional, by default, MatrixSymbol('_T', n, n)

    Examples
    ========

    >>> from sympy.stats import DiscreteMarkovChain, TransitionMatrixOf, P, E
    >>> from sympy import Matrix, MatrixSymbol, Eq, symbols
    >>> T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> YS = DiscreteMarkovChain("Y")

    >>> Y.state_space
    {0, 1, 2}
    >>> Y.transition_probabilities
    Matrix([
    [0.5, 0.2, 0.3],
    [0.2, 0.5, 0.3],
    [0.2, 0.3, 0.5]])
    >>> TS = MatrixSymbol('T', 3, 3)
    >>> P(Eq(YS[3], 2), Eq(YS[1], 1) & TransitionMatrixOf(YS, TS))
    T[0, 2]*T[1, 0] + T[1, 1]*T[1, 2] + T[1, 2]*T[2, 2]
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    Probabilities will be calculated based on indexes rather
    than state names. For example, with the Sunny-Cloudy-Rainy
    model with string state names:

    >>> from sympy.core.symbol import Str
    >>> Y = DiscreteMarkovChain("Y", [Str('Sunny'), Str('Cloudy'), Str('Rainy')], T)
    >>> P(Eq(Y[3], 2), Eq(Y[1], 1)).round(2)
    0.36

    This gives the same answer as the ``[0, 1, 2]`` state space.
    Currently, there is no support for state names within probability
    and expectation statements. Here is a work-around using ``Str``:

    >>> P(Eq(Str('Rainy'), Y[3]), Eq(Y[1], Str('Cloudy'))).round(2)
    0.36

    Symbol state names can also be used:

    >>> sunny, cloudy, rainy = symbols('Sunny, Cloudy, Rainy')
    >>> Y = DiscreteMarkovChain("Y", [sunny, cloudy, rainy], T)
    >>> P(Eq(Y[3], rainy), Eq(Y[1], cloudy)).round(2)
    0.36

    Expectations will be calculated as follows:

    >>> E(Y[3], Eq(Y[1], cloudy))
    0.38*Cloudy + 0.36*Rainy + 0.26*Sunny

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    transition matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> T = Matrix([[Rational(5, 10), Rational(3, 10), Rational(2, 10)], [Rational(2, 10), Rational(7, 10), Rational(1, 10)], [Rational(3, 10), Rational(3, 10), Rational(4, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> P(Eq(Y[3], Y[1]), Eq(Y[0], 0)).round(3)
    0.409
    >>> P(Gt(Y[3], Y[1]), Eq(Y[0], 0)).round(2)
    0.36
    >>> P(Le(Y[15], Y[10]), Eq(Y[8], 2)).round(7)
    0.6963328

    Symbolic probability queries are also supported

    >>> a, b, c, d = symbols('a b c d')
    >>> T = Matrix([[Rational(1, 10), Rational(4, 10), Rational(5, 10)], [Rational(3, 10), Rational(4, 10), Rational(3, 10)], [Rational(7, 10), Rational(2, 10), Rational(1, 10)]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1}).round(4)
    0.3096
    >>> P(Eq(Y[10], 2), Eq(Y[5], 1)).evalf().round(4)
    0.3096
    >>> query_gt = P(Gt(Y[a], b), Eq(Y[c], d))
    >>> query_gt.subs({a:21, b:0, c:5, d:0}).evalf().round(5)
    0.64705
    >>> P(Gt(Y[21], 0), Eq(Y[5], 0)).round(5)
    0.64705

    There is limited support for arbitrarily sized states:

    >>> n = symbols('n', nonnegative=True, integer=True)
    >>> T = MatrixSymbol('T', n, n)
    >>> Y = DiscreteMarkovChain("Y", trans_probs=T)
    >>> Y.state_space
    Range(0, n, 1)
    >>> query = P(Eq(Y[a], b), Eq(Y[c], d))
    >>> query.subs({a:10, b:2, c:5, d:1})
    (T**5)[1, 2]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Discrete-time_Markov_chain
    .. [2] https://web.archive.org/web/20201230182007/https://www.dartmouth.edu/~chance/teaching_aids/books_articles/probability_book/Chapter11.pdf
    """
    index_set = S.Naturals0

    def __new__(cls, sym, state_space=None, trans_probs=None):
        sym = _symbol_converter(sym)

        state_space, trans_probs = MarkovProcess._sanity_checks(state_space, trans_probs)

        obj = Basic.__new__(cls, sym, state_space, trans_probs) # type: ignore
        indices = {}
        if isinstance(obj.number_of_states, Integer):
            for index, state in enumerate(obj._state_index):
                indices[state] = index
        obj.index_of = indices
        return obj

    @property
    def transition_probabilities(self):
        """
        Transition probabilities of discrete Markov chain,
        either an instance of Matrix or MatrixSymbol.
        """
        return self.args[2]

    def communication_classes(self) -> list[tuple[list[Basic], Boolean, Integer]]:
        """
        Returns the list of communication classes that partition
        the states of the markov chain.

        A communication class is defined to be a set of states
        such that every state in that set is reachable from
        every other state in that set. Due to its properties
        this forms a class in the mathematical sense.
        Communication classes are also known as recurrence
        classes.

        Returns
        =======

        classes
            The ``classes`` are a list of tuples. Each
            tuple represents a single communication class
            with its properties. The first element in the
            tuple is the list of states in the class, the
            second element is whether the class is recurrent
            and the third element is the period of the
            communication class.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix
        >>> T = Matrix([[0, 1, 0],
        ...             [1, 0, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', [1, 2, 3], T)
        >>> classes = X.communication_classes()
        >>> for states, is_recurrent, period in classes:
        ...     states, is_recurrent, period
        ([1, 2], True, 2)
        ([3], False, 1)

        From this we can see that states ``1`` and ``2``
        communicate, are recurrent and have a period
        of 2. We can also see state ``3`` is transient
        with a period of 1.

        Notes
        =====

        The algorithm used is of order ``O(n**2)`` where
        ``n`` is the number of states in the markov chain.
        It uses Tarjan's algorithm to find the classes
        themselves and then it uses a breadth-first search
        algorithm to find each class's periodicity.
        Most of the algorithm's components approach ``O(n)``
        as the matrix becomes more and more sparse.

        References
        ==========

        .. [1] https://web.archive.org/web/20220207032113/https://www.columbia.edu/~ww2040/4701Sum07/4701-06-Notes-MCII.pdf
        .. [2] https://cecas.clemson.edu/~shierd/Shier/markov.pdf
        .. [3] https://www.proquest.com/openview/4adc6a51d8371be5b0e4c7dff287fc70/1?pq-origsite=gscholar&cbl=2026366&diss=y
        .. [4] https://www.mathworks.com/help/econ/dtmc.classify.html
        """
        n = self.number_of_states
        T = self.transition_probabilities

        if isinstance(T, MatrixSymbol):
            raise NotImplementedError("Cannot perform the operation with a symbolic matrix.")

        # begin Tarjan's algorithm
        V = Range(n)
        # don't use state names. Rather use state
        # indexes since we use them for matrix
        # indexing here and later onward
        E = [(i, j) for i in V for j in V if T[i, j] != 0]
        classes = strongly_connected_components((V, E))
        # end Tarjan's algorithm

        recurrence = []
        periods = []
        for class_ in classes:
            # begin recurrent check (similar to self._check_trans_probs())
            submatrix = T[class_, class_]  # get the submatrix with those states
            is_recurrent = S.true
            rows = submatrix.tolist()
            for row in rows:
                if (sum(row) - 1) != 0:
                    is_recurrent = S.false
                    break
            recurrence.append(is_recurrent)
            # end recurrent check

            # begin breadth-first search
            non_tree_edge_values: set[int] = set()
            visited = {class_[0]}
            newly_visited = {class_[0]}
            level = {class_[0]: 0}
            current_level = 0
            done = False  # imitate a do-while loop
            while not done:  # runs at most len(class_) times
                done = len(visited) == len(class_)
                current_level += 1

                # this loop and the while loop above run a combined len(class_) number of times.
                # so this triple nested loop runs through each of the n states once.
                for i in newly_visited:

                    # the loop below runs len(class_) number of times
                    # complexity is around about O(n * avg(len(class_)))
                    newly_visited = {j for j in class_ if T[i, j] != 0}

                    new_tree_edges = newly_visited.difference(visited)
                    for j in new_tree_edges:
                        level[j] = current_level

                    new_non_tree_edges = newly_visited.intersection(visited)
                    new_non_tree_edge_values = {level[i]-level[j]+1 for j in new_non_tree_edges}

                    non_tree_edge_values = non_tree_edge_values.union(new_non_tree_edge_values)
                    visited = visited.union(new_tree_edges)

            # igcd needs at least 2 arguments
            positive_ntev = {val_e for val_e in non_tree_edge_values if val_e > 0}
            if len(positive_ntev) == 0:
                periods.append(len(class_))
            elif len(positive_ntev) == 1:
                periods.append(positive_ntev.pop())
            else:
                periods.append(igcd(*positive_ntev))
            # end breadth-first search

        # convert back to the user's state names
        classes = [[_sympify(self._state_index[i]) for i in class_] for class_ in classes]
        return list(zip(classes, recurrence, map(Integer,periods)))

    def fundamental_matrix(self):
        """
        Each entry fundamental matrix can be interpreted as
        the expected number of times the chains is in state j
        if it started in state i.

        References
        ==========

        .. [1] https://lips.cs.princeton.edu/the-fundamental-matrix-of-a-finite-markov-chain/

        """
        _, _, _, Q = self.decompose()

        if Q.shape[0] > 0:  # if non-ergodic
            I = eye(Q.shape[0])
            if (I - Q).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - Q).inv().as_immutable()
        else:  # if ergodic
            P = self.transition_probabilities
            I = eye(P.shape[0])
            w = self.fixed_row_vector()
            W = Matrix([list(w) for i in range(0, P.shape[0])])
            if (I - P + W).det() == 0:
                raise ValueError("The fundamental matrix doesn't exist.")
            return (I - P + W).inv().as_immutable()

    def absorbing_probabilities(self):
        """
        Computes the absorbing probabilities, i.e.
        the ij-th entry of the matrix denotes the
        probability of Markov chain being absorbed
        in state j starting from state i.
        """
        _, _, R, _ = self.decompose()
        N = self.fundamental_matrix()
        if R is None or N is None:
            return None
        return N*R

    def absorbing_probabilites(self):
        sympy_deprecation_warning(
            """
            DiscreteMarkovChain.absorbing_probabilites() is deprecated. Use
            absorbing_probabilities() instead (note the spelling difference).
            """,
            deprecated_since_version="1.7",
            active_deprecations_target="deprecated-absorbing_probabilites",
        )
        return self.absorbing_probabilities()

    def is_regular(self):
        tuples = self.communication_classes()
        if len(tuples) == 0:
            return S.false  # not defined for a 0x0 matrix
        classes, _, periods = list(zip(*tuples))
        return And(len(classes) == 1, periods[0] == 1)

    def is_ergodic(self):
        tuples = self.communication_classes()
        if len(tuples) == 0:
            return S.false  # not defined for a 0x0 matrix
        classes, _, _ = list(zip(*tuples))
        return S(len(classes) == 1)

    def is_absorbing_state(self, state):
        trans_probs = self.transition_probabilities
        if isinstance(trans_probs, ImmutableMatrix) and \
            state < trans_probs.shape[0]:
            return S(trans_probs[state, state]) is S.One

    def is_absorbing_chain(self):
        states, A, B, C = self.decompose()
        r = A.shape[0]
        return And(r > 0, A == Identity(r).as_explicit())

    def stationary_distribution(self, condition_set=False) -> ImmutableMatrix | ConditionSet | Lambda:
        r"""
        The stationary distribution is any row vector, p, that solves p = pP,
        is row stochastic and each element in p must be nonnegative.
        That means in matrix form: :math:`(P-I)^T p^T = 0` and
        :math:`(1, \dots, 1) p = 1`
        where ``P`` is the one-step transition matrix.

        All time-homogeneous Markov Chains with a finite state space
        have at least one stationary distribution. In addition, if
        a finite time-homogeneous Markov Chain is irreducible, the
        stationary distribution is unique.

        Parameters
        ==========

        condition_set : bool
            If the chain has a symbolic size or transition matrix,
            it will return a ``Lambda`` if ``False`` and return a
            ``ConditionSet`` if ``True``.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        An irreducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [1, 0, 0]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13, 5/13, 0]])

        A reducible Markov Chain

        >>> T = Matrix([[S(1)/2, S(1)/2, 0],
        ...             [S(4)/5, S(1)/5, 0],
        ...             [0, 0, 1]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> X.stationary_distribution()
        Matrix([[8/13 - 8*tau0/13, 5/13 - 5*tau0/13, tau0]])

        >>> Y = DiscreteMarkovChain('Y')
        >>> Y.stationary_distribution()
        Lambda((wm, _T), Eq(wm*_T, wm))

        >>> Y.stationary_distribution(condition_set=True)
        ConditionSet(wm, Eq(wm*_T, wm))

        References
        ==========

        .. [1] https://www.probabilitycourse.com/chapter11/11_2_6_stationary_and_limiting_distributions.php
        .. [2] https://web.archive.org/web/20210508104430/https://galton.uchicago.edu/~yibi/teaching/stat317/2014/Lectures/Lecture4_6up.pdf

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.limiting_distribution
        """
        trans_probs = self.transition_probabilities
        n = self.number_of_states

        if n == 0:
            return ImmutableMatrix(Matrix([[]]))

        # symbolic matrix version
        if isinstance(trans_probs, MatrixSymbol):
            wm = MatrixSymbol('wm', 1, n)
            if condition_set:
                return ConditionSet(wm, Eq(wm * trans_probs, wm))
            else:
                return Lambda((wm, trans_probs), Eq(wm * trans_probs, wm))

        # numeric matrix version
        a = Matrix(trans_probs - Identity(n)).T
        a[0, 0:n] = ones(1, n) # type: ignore
        b = zeros(n, 1)
        b[0, 0] = 1

        soln = list(linsolve((a, b)))[0]
        return ImmutableMatrix([soln])

    def fixed_row_vector(self):
        """
        A wrapper for ``stationary_distribution()``.
        """
        return self.stationary_distribution()

    @property
    def limiting_distribution(self):
        """
        The fixed row vector is the limiting
        distribution of a discrete Markov chain.
        """
        return self.fixed_row_vector()

    def decompose(self) -> tuple[list[Basic], ImmutableMatrix, ImmutableMatrix, ImmutableMatrix]:
        """
        Decomposes the transition matrix into submatrices with
        special properties.

        The transition matrix can be decomposed into 4 submatrices:
        - A - the submatrix from recurrent states to recurrent states.
        - B - the submatrix from transient to recurrent states.
        - C - the submatrix from transient to transient states.
        - O - the submatrix of zeros for recurrent to transient states.

        Returns
        =======

        states, A, B, C
            ``states`` - a list of state names with the first being
            the recurrent states and the last being
            the transient states in the order
            of the row names of A and then the row names of C.
            ``A`` - the submatrix from recurrent states to recurrent states.
            ``B`` - the submatrix from transient to recurrent states.
            ``C`` - the submatrix from transient to transient states.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        One can decompose this chain for example:

        >>> T = Matrix([[S(1)/2, S(1)/2, 0,      0,      0],
        ...             [S(2)/5, S(1)/5, S(2)/5, 0,      0],
        ...             [0,      0,      1,      0,      0],
        ...             [0,      0,      S(1)/2, S(1)/2, 0],
        ...             [S(1)/2, 0,      0,      0, S(1)/2]])
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> states, A, B, C = X.decompose()
        >>> states
        [2, 0, 1, 3, 4]

        >>> A   # recurrent to recurrent
        Matrix([[1]])

        >>> B  # transient to recurrent
        Matrix([
        [  0],
        [2/5],
        [1/2],
        [  0]])

        >>> C  # transient to transient
        Matrix([
        [1/2, 1/2,   0,   0],
        [2/5, 1/5,   0,   0],
        [  0,   0, 1/2,   0],
        [1/2,   0,   0, 1/2]])

        This means that state 2 is the only absorbing state
        (since A is a 1x1 matrix). B is a 4x1 matrix since
        the 4 remaining transient states all merge into recurrent
        state 2. And C is the 4x4 matrix that shows how the
        transient states 0, 1, 3, 4 all interact.

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.communication_classes
        sympy.stats.DiscreteMarkovChain.canonical_form

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Absorbing_Markov_chain
        .. [2] https://people.brandeis.edu/~igusa/Math56aS08/Math56a_S08_notes015.pdf
        """
        trans_probs = self.transition_probabilities

        classes = self.communication_classes()
        r_states = []
        t_states = []

        for states, recurrent, period in classes:
            if recurrent:
                r_states += states
            else:
                t_states += states

        states = r_states + t_states
        indexes = [self.index_of[state] for state in states] # type: ignore

        A = Matrix(len(r_states), len(r_states),
                   lambda i, j: trans_probs[indexes[i], indexes[j]])

        B = Matrix(len(t_states), len(r_states),
                   lambda i, j: trans_probs[indexes[len(r_states) + i], indexes[j]])

        C = Matrix(len(t_states), len(t_states),
                   lambda i, j: trans_probs[indexes[len(r_states) + i], indexes[len(r_states) + j]])

        return states, A.as_immutable(), B.as_immutable(), C.as_immutable()

    def canonical_form(self) -> tuple[list[Basic], ImmutableMatrix]:
        """
        Reorders the one-step transition matrix
        so that recurrent states appear first and transient
        states appear last. Other representations include inserting
        transient states first and recurrent states last.

        Returns
        =======

        states, P_new
            ``states`` is the list that describes the order of the
            new states in the matrix
            so that the ith element in ``states`` is the state of the
            ith row of A.
            ``P_new`` is the new transition matrix in canonical form.

        Examples
        ========

        >>> from sympy.stats import DiscreteMarkovChain
        >>> from sympy import Matrix, S

        You can convert your chain into canonical form:

        >>> T = Matrix([[S(1)/2, S(1)/2, 0,      0,      0],
        ...             [S(2)/5, S(1)/5, S(2)/5, 0,      0],
        ...             [0,      0,      1,      0,      0],
        ...             [0,      0,      S(1)/2, S(1)/2, 0],
        ...             [S(1)/2, 0,      0,      0, S(1)/2]])
        >>> X = DiscreteMarkovChain('X', list(range(1, 6)), trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        The new states are [3, 1, 2, 4, 5] and you can
        create a new chain with this and its canonical
        form will remain the same (since it is already
        in canonical form).

        >>> X = DiscreteMarkovChain('X', states, new_matrix)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [3, 1, 2, 4, 5]

        >>> new_matrix
        Matrix([
        [  1,   0,   0,   0,   0],
        [  0, 1/2, 1/2,   0,   0],
        [2/5, 2/5, 1/5,   0,   0],
        [1/2,   0,   0, 1/2,   0],
        [  0, 1/2,   0,   0, 1/2]])

        This is not limited to absorbing chains:

        >>> T = Matrix([[0, 5,  5, 0,  0],
        ...             [0, 0,  0, 10, 0],
        ...             [5, 0,  5, 0,  0],
        ...             [0, 10, 0, 0,  0],
        ...             [0, 3,  0, 3,  4]])/10
        >>> X = DiscreteMarkovChain('X', trans_probs=T)
        >>> states, new_matrix = X.canonical_form()
        >>> states
        [1, 3, 0, 2, 4]

        >>> new_matrix
        Matrix([
        [   0,    1,   0,   0,   0],
        [   1,    0,   0,   0,   0],
        [ 1/2,    0,   0, 1/2,   0],
        [   0,    0, 1/2, 1/2,   0],
        [3/10, 3/10,   0,   0, 2/5]])

        See Also
        ========

        sympy.stats.DiscreteMarkovChain.communication_classes
        sympy.stats.DiscreteMarkovChain.decompose

        References
        ==========

        .. [1] https://onlinelibrary.wiley.com/doi/pdf/10.1002/9780470316887.app1
        .. [2] http://www.columbia.edu/~ww2040/6711F12/lect1023big.pdf
        """
        states, A, B, C = self.decompose()
        O = zeros(A.shape[0], C.shape[1])
        return states, BlockMatrix([[A, O], [B, C]]).as_explicit()

    def sample(self):
        """
        Returns
        =======

        sample: iterator object
            iterator object containing the sample

        """
        if not isinstance(self.transition_probabilities, (Matrix, ImmutableMatrix)):
            raise ValueError("Transition Matrix must be provided for sampling")
        Tlist = self.transition_probabilities.tolist()
        samps = [random.choice(list(self.state_space))]
        yield samps[0]
        time = 1
        densities = {}
        for state in self.state_space:
            states = list(self.state_space)
            densities[state] = {states[i]: Tlist[state][i]
                        for i in range(len(states))}
        while time < S.Infinity:
            samps.append((next(sample_iter(FiniteRV("_", densities[samps[time - 1]])))))
            yield samps[time]
            time += 1

class ContinuousMarkovChain(ContinuousTimeStochasticProcess, MarkovProcess):
    """
    Represents continuous time Markov chain.

    Parameters
    ==========

    sym : Symbol/str
    state_space : Set
        Optional, by default, S.Reals
    gen_mat : Matrix/ImmutableMatrix/MatrixSymbol
        Optional, by default, None

    Examples
    ========

    >>> from sympy.stats import ContinuousMarkovChain, P
    >>> from sympy import Matrix, S, Eq, Gt
    >>> G = Matrix([[-S(1), S(1)], [S(1), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1], gen_mat=G)
    >>> C.limiting_distribution()
    Matrix([[1/2, 1/2]])
    >>> C.state_space
    {0, 1}
    >>> C.generator_matrix
    Matrix([
    [-1,  1],
    [ 1, -1]])

    Probability queries are supported

    >>> P(Eq(C(1.96), 0), Eq(C(0.78), 1)).round(5)
    0.45279
    >>> P(Gt(C(1.7), 0), Eq(C(0.82), 1)).round(5)
    0.58602

    Probability of expressions with multiple RandomIndexedSymbols
    can also be calculated provided there is only 1 RandomIndexedSymbol
    in the given condition. It is always better to use Rational instead
    of floating point numbers for the probabilities in the
    generator matrix to avoid errors.

    >>> from sympy import Gt, Le, Rational
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> P(Eq(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.37933
    >>> P(Gt(C(3.92), C(1.75)), Eq(C(0.46), 0)).round(5)
    0.34211
    >>> P(Le(C(1.57), C(3.14)), Eq(C(1.22), 1)).round(4)
    0.7143

    Symbolic probability queries are also supported

    >>> from sympy import symbols
    >>> a,b,c,d = symbols('a b c d')
    >>> G = Matrix([[-S(1), Rational(1, 10), Rational(9, 10)], [Rational(2, 5), -S(1), Rational(3, 5)], [Rational(1, 2), Rational(1, 2), -S(1)]])
    >>> C = ContinuousMarkovChain('C', state_space=[0, 1, 2], gen_mat=G)
    >>> query = P(Eq(C(a), b), Eq(C(c), d))
    >>> query.subs({a:3.65, b:2, c:1.78, d:1}).evalf().round(10)
    0.4002723175
    >>> P(Eq(C(3.65), 2), Eq(C(1.78), 1)).round(10)
    0.4002723175
    >>> query_gt = P(Gt(C(a), b), Eq(C(c), d))
    >>> query_gt.subs({a:43.2, b:0, c:3.29, d:2}).evalf().round(10)
    0.6832579186
    >>> P(Gt(C(43.2), 0), Eq(C(3.29), 2)).round(10)
    0.6832579186

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Markov_chain#Continuous-time_Markov_chain
    .. [2] https://u.math.biu.ac.il/~amirgi/CTMCnotes.pdf
    """
    index_set = S.Reals

    def __new__(cls, sym, state_space=None, gen_mat=None):
        sym = _symbol_converter(sym)
        state_space, gen_mat = MarkovProcess._sanity_checks(state_space, gen_mat)
        obj = Basic.__new__(cls, sym, state_space, gen_mat)
        indices = {}
        if isinstance(obj.number_of_states, Integer):
            for index, state in enumerate(obj.state_space):
                indices[state] = index
        obj.index_of = indices
        return obj

    @property
    def generator_matrix(self):
        return self.args[2]

    @cacheit
    def transition_probabilities(self, gen_mat=None):
        t = Dummy('t')
        if isinstance(gen_mat, (Matrix, ImmutableMatrix)) and \
                gen_mat.is_diagonalizable():
            # for faster computation use diagonalized generator matrix
            Q, D = gen_mat.diagonalize()
            return Lambda(t, Q*exp(t*D)*Q.inv())
        if gen_mat != None:
            return Lambda(t, exp(t*gen_mat))

    def limiting_distribution(self):
        gen_mat = self.generator_matrix
        if gen_mat is None:
            return None
        if isinstance(gen_mat, MatrixSymbol):
            wm = MatrixSymbol('wm', 1, gen_mat.shape[0])
            return Lambda((wm, gen_mat), Eq(wm*gen_mat, wm))
        w = IndexedBase('w')
        wi = [w[i] for i in range(gen_mat.shape[0])]
        wm = Matrix([wi])
        eqs = (wm*gen_mat).tolist()[0]
        eqs.append(sum(wi) - 1)
        soln = list(linsolve(eqs, wi))[0]
        return ImmutableMatrix([soln])


class BernoulliProcess(DiscreteTimeStochasticProcess):
    """
    The Bernoulli process consists of repeated
    independent Bernoulli process trials with the same parameter `p`.
    It's assumed that the probability `p` applies to every
    trial and that the outcomes of each trial
    are independent of all the rest. Therefore Bernoulli Process
    is Discrete State and Discrete Time Stochastic Process.

    Parameters
    ==========

    sym : Symbol/str
    success : Integer/str
            The event which is considered to be success. Default: 1.
    failure: Integer/str
            The event which is considered to be failure. Default: 0.
    p : Real Number between 0 and 1
            Represents the probability of getting success.

    Examples
    ========

    >>> from sympy.stats import BernoulliProcess, P, E
    >>> from sympy import Eq, Gt
    >>> B = BernoulliProcess("B", p=0.7, success=1, failure=0)
    >>> B.state_space
    {0, 1}
    >>> B.p.round(2)
    0.70
    >>> B.success
    1
    >>> B.failure
    0
    >>> X = B[1] + B[2] + B[3]
    >>> P(Eq(X, 0)).round(2)
    0.03
    >>> P(Eq(X, 2)).round(2)
    0.44
    >>> P(Eq(X, 4)).round(2)
    0
    >>> P(Gt(X, 1)).round(2)
    0.78
    >>> P(Eq(B[1], 0) & Eq(B[2], 1) & Eq(B[3], 0) & Eq(B[4], 1)).round(2)
    0.04
    >>> B.joint_distribution(B[1], B[2])
    JointDistributionHandmade(Lambda((B[1], B[2]), Piecewise((0.7, Eq(B[1], 1)),
    (0.3, Eq(B[1], 0)), (0, True))*Piecewise((0.7, Eq(B[2], 1)), (0.3, Eq(B[2], 0)),
    (0, True))))
    >>> E(2*B[1] + B[2]).round(2)
    2.10
    >>> P(B[1] < 1).round(2)
    0.30

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Bernoulli_process
    .. [2] https://mathcs.clarku.edu/~djoyce/ma217/bernoulli.pdf

    """

    index_set = S.Naturals0

    def __new__(cls, sym, p, success=1, failure=0):
        _value_check(p >= 0 and p <= 1, 'Value of p must be between 0 and 1.')
        sym = _symbol_converter(sym)
        p = _sympify(p)
        success = _sym_sympify(success)
        failure = _sym_sympify(failure)
        return Basic.__new__(cls, sym, p, success, failure)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def p(self):
        return self.args[1]

    @property
    def success(self):
        return self.args[2]

    @property
    def failure(self):
        return self.args[3]

    @property
    def state_space(self):
        return _set_converter([self.success, self.failure])

    def distribution(self, key=None):
        if key is None:
            self._deprecation_warn_distribution()
            return BernoulliDistribution(self.p)
        return BernoulliDistribution(self.p, self.success, self.failure)

    def simple_rv(self, rv):
        return Bernoulli(rv.name, p=self.p,
                succ=self.success, fail=self.failure)

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation.

        Parameters
        ==========

        expr : RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition : Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation of the RandomIndexedSymbol.

        """

        return _SubstituteRV._expectation(expr, condition, evaluate, **kwargs)

    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Computes probability.

        Parameters
        ==========

        condition : Relational
                Condition for which probability has to be computed. Must
                contain a RandomIndexedSymbol of the process.
        given_condition : Relational, Logic
                The given conditions under which computations should be done.

        Returns
        =======

        Probability of the condition.

        """

        return _SubstituteRV._probability(condition, given_condition, evaluate, **kwargs)

    def density(self, x):
        return Piecewise((self.p, Eq(x, self.success)),
                         (1 - self.p, Eq(x, self.failure)),
                         (S.Zero, True))

class _SubstituteRV:
    """
    Internal class to handle the queries of expectation and probability
    by substitution.
    """

    @staticmethod
    def _rvindexed_subs(expr, condition=None):
        """
        Substitutes the RandomIndexedSymbol with the RandomSymbol with
        same name, distribution and probability as RandomIndexedSymbol.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        """

        rvs_expr = random_symbols(expr)
        if len(rvs_expr) != 0:
            swapdict_expr = {}
            for rv in rvs_expr:
                if isinstance(rv, RandomIndexedSymbol):
                    newrv = rv.pspace.process.simple_rv(rv) # substitute with equivalent simple rv
                    swapdict_expr[rv] = newrv
            expr = expr.subs(swapdict_expr)
        rvs_cond = random_symbols(condition)
        if len(rvs_cond)!=0:
            swapdict_cond = {}
            for rv in rvs_cond:
                if isinstance(rv, RandomIndexedSymbol):
                    newrv = rv.pspace.process.simple_rv(rv)
                    swapdict_cond[rv] = newrv
            condition = condition.subs(swapdict_cond)
        return expr, condition

    @classmethod
    def _expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Internal method for computing expectation of indexed RV.

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Logic
            The given conditions under which computations should be done.

        Returns
        =======

        Expectation of the RandomIndexedSymbol.

        """
        new_expr, new_condition = self._rvindexed_subs(expr, condition)

        if not is_random(new_expr):
            return new_expr
        new_pspace = pspace(new_expr)
        if new_condition is not None:
            new_expr = given(new_expr, new_condition)
        if new_expr.is_Add:  # As E is Linear
            return Add(*[new_pspace.compute_expectation(
                        expr=arg, evaluate=evaluate, **kwargs)
                        for arg in new_expr.args])
        return new_pspace.compute_expectation(
                new_expr, evaluate=evaluate, **kwargs)

    @classmethod
    def _probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Internal method for computing probability of indexed RV

        Parameters
        ==========

        condition: Relational
                Condition for which probability has to be computed. Must
                contain a RandomIndexedSymbol of the process.
        given_condition: Relational/And
                The given conditions under which computations should be done.

        Returns
        =======

        Probability of the condition.

        """
        new_condition, new_givencondition = self._rvindexed_subs(condition, given_condition)

        if isinstance(new_givencondition, RandomSymbol):
            condrv = random_symbols(new_condition)
            if len(condrv) == 1 and condrv[0] == new_givencondition:
                return BernoulliDistribution(self._probability(new_condition), 0, 1)

            if any(dependent(rv, new_givencondition) for rv in condrv):
                return Probability(new_condition, new_givencondition)
            else:
                return self._probability(new_condition)

        if new_givencondition is not None and \
                not isinstance(new_givencondition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (new_givencondition))
        if new_givencondition == False or new_condition == False:
            return S.Zero
        if new_condition == True:
            return S.One
        if not isinstance(new_condition, (Relational, Boolean)):
            raise ValueError("%s is not a relational or combination of relationals"
                    % (new_condition))

        if new_givencondition is not None:  # If there is a condition
        # Recompute on new conditional expr
            return self._probability(given(new_condition, new_givencondition, **kwargs), **kwargs)
        result = pspace(new_condition).probability(new_condition, **kwargs)
        if evaluate and hasattr(result, 'doit'):
            return result.doit()
        else:
            return result

def get_timerv_swaps(expr, condition):
    """
    Finds the appropriate interval for each time stamp in expr by parsing
    the given condition and returns intervals for each timestamp and
    dictionary that maps variable time-stamped Random Indexed Symbol to its
    corresponding Random Indexed variable with fixed time stamp.

    Parameters
    ==========

    expr: SymPy Expression
        Expression containing Random Indexed Symbols with variable time stamps
    condition: Relational/Boolean Expression
        Expression containing time bounds of variable time stamps in expr

    Examples
    ========

    >>> from sympy.stats.stochastic_process_types import get_timerv_swaps, PoissonProcess
    >>> from sympy import symbols, Contains, Interval
    >>> x, t, d = symbols('x t d', positive=True)
    >>> X = PoissonProcess("X", 3)
    >>> get_timerv_swaps(x*X(t), Contains(t, Interval.Lopen(0, 1)))
    ([Interval.Lopen(0, 1)], {X(t): X(1)})
    >>> get_timerv_swaps((X(t)**2 + X(d)**2), Contains(t, Interval.Lopen(0, 1))
    ... & Contains(d, Interval.Ropen(1, 4))) # doctest: +SKIP
    ([Interval.Ropen(1, 4), Interval.Lopen(0, 1)], {X(d): X(3), X(t): X(1)})

    Returns
    =======

    intervals: list
        List of Intervals/FiniteSet on which each time stamp is defined
    rv_swap: dict
        Dictionary mapping variable time Random Indexed Symbol to constant time
        Random Indexed Variable

    """

    if not isinstance(condition, (Relational, Boolean)):
        raise ValueError("%s is not a relational or combination of relationals"
            % (condition))
    expr_syms = list(expr.atoms(RandomIndexedSymbol))
    if isinstance(condition, (And, Or)):
        given_cond_args = condition.args
    else: # single condition
        given_cond_args = (condition, )
    rv_swap = {}
    intervals = []
    for expr_sym in expr_syms:
        for arg in given_cond_args:
            if arg.has(expr_sym.key) and isinstance(expr_sym.key, Symbol):
                intv = _set_converter(arg.args[1])
                diff_key = intv._sup - intv._inf
                if diff_key == oo:
                    raise ValueError("%s should have finite bounds" % str(expr_sym.name))
                elif diff_key == S.Zero: # has singleton set
                    diff_key = intv._sup
                rv_swap[expr_sym] = expr_sym.subs({expr_sym.key: diff_key})
                intervals.append(intv)
    return intervals, rv_swap


class CountingProcess(ContinuousTimeStochasticProcess):
    """
    This class handles the common methods of the Counting Processes
    such as Poisson, Wiener and Gamma Processes
    """
    index_set = _set_converter(Interval(0, oo))

    @property
    def symbol(self):
        return self.args[0]

    def expectation(self, expr, condition=None, evaluate=True, **kwargs):
        """
        Computes expectation

        Parameters
        ==========

        expr: RandomIndexedSymbol, Relational, Logic
            Condition for which expectation has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        condition: Relational, Boolean
            The given conditions under which computations should be done, i.e,
            the intervals on which each variable time stamp in expr is defined

        Returns
        =======

        Expectation of the given expr

        """
        if condition is not None:
            intervals, rv_swap = get_timerv_swaps(expr, condition)
             # they are independent when they have non-overlapping intervals
            if len(intervals) == 1 or all(Intersection(*intv_comb) == EmptySet
                for intv_comb in itertools.combinations(intervals, 2)):
                if expr.is_Add:
                    return Add.fromiter(self.expectation(arg, condition)
                            for arg in expr.args)
                expr = expr.subs(rv_swap)
            else:
                return Expectation(expr, condition)

        return _SubstituteRV._expectation(expr, evaluate=evaluate, **kwargs)

    def _solve_argwith_tworvs(self, arg):
        if arg.args[0].key >= arg.args[1].key or isinstance(arg, Eq):
            diff_key = abs(arg.args[0].key - arg.args[1].key)
            rv = arg.args[0]
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        else:
            diff_key = arg.args[1].key - arg.args[0].key
            rv = arg.args[1]
            arg = arg.__class__(rv.pspace.process(diff_key), 0)
        return arg

    def _solve_numerical(self, condition, given_condition=None):
        if isinstance(condition, And):
            args_list = list(condition.args)
        else:
            args_list = [condition]
        if given_condition is not None:
            if isinstance(given_condition, And):
                args_list.extend(list(given_condition.args))
            else:
                args_list.extend([given_condition])
        # sort the args based on timestamp to get the independent increments in
        # each segment using all the condition args as well as given_condition args
        args_list = sorted(args_list, key=lambda x: x.args[0].key)
        result = []
        cond_args = list(condition.args) if isinstance(condition, And) else [condition]
        if args_list[0] in cond_args and not (is_random(args_list[0].args[0])
                        and is_random(args_list[0].args[1])):
            result.append(_SubstituteRV._probability(args_list[0]))

        if is_random(args_list[0].args[0]) and is_random(args_list[0].args[1]):
            arg = self._solve_argwith_tworvs(args_list[0])
            result.append(_SubstituteRV._probability(arg))

        for i in range(len(args_list) - 1):
            curr, nex = args_list[i], args_list[i + 1]
            diff_key = nex.args[0].key - curr.args[0].key
            working_set = curr.args[0].pspace.process.state_space
            if curr.args[1] > nex.args[1]: #impossible condition so return 0
                result.append(0)
                break
            if isinstance(curr, Eq):
                working_set = Intersection(working_set, Interval.Lopen(curr.args[1], oo))
            else:
                working_set = Intersection(working_set, curr.as_set())
            if isinstance(nex, Eq):
                working_set = Intersection(working_set, Interval(-oo, nex.args[1]))
            else:
                working_set = Intersection(working_set, nex.as_set())
            if working_set == EmptySet:
                rv = Eq(curr.args[0].pspace.process(diff_key), 0)
                result.append(_SubstituteRV._probability(rv))
            else:
                if working_set.is_finite_set:
                    if isinstance(curr, Eq) and isinstance(nex, Eq):
                        rv = Eq(curr.args[0].pspace.process(diff_key), len(working_set))
                        result.append(_SubstituteRV._probability(rv))
                    elif isinstance(curr, Eq) ^ isinstance(nex, Eq):
                        result.append(Add.fromiter(_SubstituteRV._probability(Eq(
                        curr.args[0].pspace.process(diff_key), x))
                                for x in range(len(working_set))))
                    else:
                        n = len(working_set)
                        result.append(Add.fromiter((n - x)*_SubstituteRV._probability(Eq(
                        curr.args[0].pspace.process(diff_key), x)) for x in range(n)))
                else:
                    result.append(_SubstituteRV._probability(
                    curr.args[0].pspace.process(diff_key) <= working_set._sup - working_set._inf))
        return Mul.fromiter(result)


    def probability(self, condition, given_condition=None, evaluate=True, **kwargs):
        """
        Computes probability.

        Parameters
        ==========

        condition: Relational
            Condition for which probability has to be computed. Must
            contain a RandomIndexedSymbol of the process.
        given_condition: Relational, Boolean
            The given conditions under which computations should be done, i.e,
            the intervals on which each variable time stamp in expr is defined

        Returns
        =======

        Probability of the condition

        """
        check_numeric = True
        if isinstance(condition, (And, Or)):
            cond_args = condition.args
        else:
            cond_args = (condition, )
        # check that condition args are numeric or not
        if not all(arg.args[0].key.is_number for arg in cond_args):
            check_numeric = False
        if given_condition is not None:
            check_given_numeric = True
            if isinstance(given_condition, (And, Or)):
                given_cond_args = given_condition.args
            else:
                given_cond_args = (given_condition, )
            # check that given condition args are numeric or not
            if given_condition.has(Contains):
                check_given_numeric = False
            # Handle numerical queries
            if check_numeric and check_given_numeric:
                res = []
                if isinstance(condition, Or):
                    res.append(Add.fromiter(self._solve_numerical(arg, given_condition)
                            for arg in condition.args))
                if isinstance(given_condition, Or):
                    res.append(Add.fromiter(self._solve_numerical(condition, arg)
                            for arg in given_condition.args))
                if res:
                    return Add.fromiter(res)
                return self._solve_numerical(condition, given_condition)

            # No numeric queries, go by Contains?... then check that all the
            # given condition are in form of `Contains`
            if not all(arg.has(Contains) for arg in given_cond_args):
                raise ValueError("If given condition is passed with `Contains`, then "
                "please pass the evaluated condition with its corresponding information "
                "in terms of intervals of each time stamp to be passed in given condition.")

            intervals, rv_swap = get_timerv_swaps(condition, given_condition)
            # they are independent when they have non-overlapping intervals
            if len(intervals) == 1 or all(Intersection(*intv_comb) == EmptySet
                for intv_comb in itertools.combinations(intervals, 2)):
                if isinstance(condition, And):
                    return Mul.fromiter(self.probability(arg, given_condition)
                            for arg in condition.args)
                elif isinstance(condition, Or):
                    return Add.fromiter(self.probability(arg, given_condition)
                            for arg in condition.args)
                condition = condition.subs(rv_swap)
            else:
                return Probability(condition, given_condition)
        if check_numeric:
            return self._solve_numerical(condition)
        return _SubstituteRV._probability(condition, evaluate=evaluate, **kwargs)

class PoissonProcess(CountingProcess):
    """
    The Poisson process is a counting process. It is usually used in scenarios
    where we are counting the occurrences of certain events that appear
    to happen at a certain rate, but completely at random.

    Parameters
    ==========

    sym : Symbol/str
    lamda : Positive number
        Rate of the process, ``lambda > 0``

    Examples
    ========

    >>> from sympy.stats import PoissonProcess, P, E
    >>> from sympy import symbols, Eq, Ne, Contains, Interval
    >>> X = PoissonProcess("X", lamda=3)
    >>> X.state_space
    Naturals0
    >>> X.lamda
    3
    >>> t1, t2 = symbols('t1 t2', positive=True)
    >>> P(X(t1) < 4)
    (9*t1**3/2 + 9*t1**2/2 + 3*t1 + 1)*exp(-3*t1)
    >>> P(Eq(X(t1), 2) | Ne(X(t1), 4), Contains(t1, Interval.Ropen(2, 4)))
    1 - 36*exp(-6)
    >>> P(Eq(X(t1), 2) & Eq(X(t2), 3), Contains(t1, Interval.Lopen(0, 2))
    ... & Contains(t2, Interval.Lopen(2, 4)))
    648*exp(-12)
    >>> E(X(t1))
    3*t1
    >>> E(X(t1)**2 + 2*X(t2),  Contains(t1, Interval.Lopen(0, 1))
    ... & Contains(t2, Interval.Lopen(1, 2)))
    18
    >>> P(X(3) < 1, Eq(X(1), 0))
    exp(-6)
    >>> P(Eq(X(4), 3), Eq(X(2), 3))
    exp(-6)
    >>> P(X(2) <= 3, X(1) > 1)
    5*exp(-3)

    Merging two Poisson Processes

    >>> Y = PoissonProcess("Y", lamda=4)
    >>> Z = X + Y
    >>> Z.lamda
    7

    Splitting a Poisson Process into two independent Poisson Processes

    >>> N, M = Z.split(l1=2, l2=5)
    >>> N.lamda, M.lamda
    (2, 5)

    References
    ==========

    .. [1] https://www.probabilitycourse.com/chapter11/11_0_0_intro.php
    .. [2] https://en.wikipedia.org/wiki/Poisson_point_process

    """

    def __new__(cls, sym, lamda):
        _value_check(lamda > 0, 'lamda should be a positive number.')
        sym = _symbol_converter(sym)
        lamda = _sympify(lamda)
        return Basic.__new__(cls, sym, lamda)

    @property
    def lamda(self):
        return self.args[1]

    @property
    def state_space(self):
        return S.Naturals0

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return PoissonDistribution(self.lamda*key.key)
        return PoissonDistribution(self.lamda*key)

    def density(self, x):
        return (self.lamda*x.key)**x / factorial(x) * exp(-(self.lamda*x.key))

    def simple_rv(self, rv):
        return Poisson(rv.name, lamda=self.lamda*rv.key)

    def __add__(self, other):
        if not isinstance(other, PoissonProcess):
            raise ValueError("Only instances of Poisson Process can be merged")
        return PoissonProcess(Dummy(self.symbol.name + other.symbol.name),
                self.lamda + other.lamda)

    def split(self, l1, l2):
        if _sympify(l1 + l2) != self.lamda:
            raise ValueError("Sum of l1 and l2 should be %s" % str(self.lamda))
        return PoissonProcess(Dummy("l1"), l1), PoissonProcess(Dummy("l2"), l2)

class WienerProcess(CountingProcess):
    """
    The Wiener process is a real valued continuous-time stochastic process.
    In physics it is used to study Brownian motion and it is often also called
    Brownian motion due to its historical connection with physical process of the
    same name originally observed by Scottish botanist Robert Brown.

    Parameters
    ==========

    sym : Symbol/str

    Examples
    ========

    >>> from sympy.stats import WienerProcess, P, E
    >>> from sympy import symbols, Contains, Interval
    >>> X = WienerProcess("X")
    >>> X.state_space
    Reals
    >>> t1, t2 = symbols('t1 t2', positive=True)
    >>> P(X(t1) < 7).simplify()
    erf(7*sqrt(2)/(2*sqrt(t1)))/2 + 1/2
    >>> P((X(t1) > 2) | (X(t1) < 4), Contains(t1, Interval.Ropen(2, 4))).simplify()
    -erf(1)/2 + erf(2)/2 + 1
    >>> E(X(t1))
    0
    >>> E(X(t1) + 2*X(t2),  Contains(t1, Interval.Lopen(0, 1))
    ... & Contains(t2, Interval.Lopen(1, 2)))
    0

    References
    ==========

    .. [1] https://www.probabilitycourse.com/chapter11/11_4_0_brownian_motion_wiener_process.php
    .. [2] https://en.wikipedia.org/wiki/Wiener_process

    """
    def __new__(cls, sym):
        sym = _symbol_converter(sym)
        return Basic.__new__(cls, sym)

    @property
    def state_space(self):
        return S.Reals

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return NormalDistribution(0, sqrt(key.key))
        return NormalDistribution(0, sqrt(key))

    def density(self, x):
        return exp(-x**2/(2*x.key)) / (sqrt(2*pi)*sqrt(x.key))

    def simple_rv(self, rv):
        return Normal(rv.name, 0, sqrt(rv.key))


class GammaProcess(CountingProcess):
    r"""
    A Gamma process is a random process with independent gamma distributed
    increments. It is a pure-jump increasing Levy process.

    Parameters
    ==========

    sym : Symbol/str
    lamda : Positive number
        Jump size of the process, ``lamda > 0``
    gamma : Positive number
        Rate of jump arrivals, `\gamma > 0`

    Examples
    ========

    >>> from sympy.stats import GammaProcess, E, P, variance
    >>> from sympy import symbols, Contains, Interval, Not
    >>> t, d, x, l, g = symbols('t d x l g', positive=True)
    >>> X = GammaProcess("X", l, g)
    >>> E(X(t))
    g*t/l
    >>> variance(X(t)).simplify()
    g*t/l**2
    >>> X = GammaProcess('X', 1, 2)
    >>> P(X(t) < 1).simplify()
    lowergamma(2*t, 1)/gamma(2*t)
    >>> P(Not((X(t) < 5) & (X(d) > 3)), Contains(t, Interval.Ropen(2, 4)) &
    ... Contains(d, Interval.Lopen(7, 8))).simplify()
    -4*exp(-3) + 472*exp(-8)/3 + 1
    >>> E(X(2) + x*E(X(5)))
    10*x + 4

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gamma_process

    """
    def __new__(cls, sym, lamda, gamma):
        _value_check(lamda > 0, 'lamda should be a positive number')
        _value_check(gamma > 0, 'gamma should be a positive number')
        sym = _symbol_converter(sym)
        gamma = _sympify(gamma)
        lamda = _sympify(lamda)
        return Basic.__new__(cls, sym, lamda, gamma)

    @property
    def lamda(self):
        return self.args[1]

    @property
    def gamma(self):
        return self.args[2]

    @property
    def state_space(self):
        return _set_converter(Interval(0, oo))

    def distribution(self, key):
        if isinstance(key, RandomIndexedSymbol):
            self._deprecation_warn_distribution()
            return GammaDistribution(self.gamma*key.key, 1/self.lamda)
        return GammaDistribution(self.gamma*key, 1/self.lamda)

    def density(self, x):
        k = self.gamma*x.key
        theta = 1/self.lamda
        return x**(k - 1) * exp(-x/theta) / (gamma(k)*theta**k)

    def simple_rv(self, rv):
        return Gamma(rv.name, self.gamma*rv.key, 1/self.lamda)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\default.py ===
# engine/default.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Default implementations of per-dialect sqlalchemy.engine classes.

These are semi-private implementation classes which are only of importance
to database dialect authors; dialects will usually use the classes here
as the base class for their own corresponding classes.

"""

from __future__ import annotations

import functools
import operator
import random
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import MutableSequence
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import weakref

from . import characteristics
from . import cursor as _cursor
from . import interfaces
from .base import Connection
from .interfaces import CacheStats
from .interfaces import DBAPICursor
from .interfaces import Dialect
from .interfaces import ExecuteStyle
from .interfaces import ExecutionContext
from .reflection import ObjectKind
from .reflection import ObjectScope
from .. import event
from .. import exc
from .. import pool
from .. import util
from ..sql import compiler
from ..sql import dml
from ..sql import expression
from ..sql import type_api
from ..sql import util as sql_util
from ..sql._typing import is_tuple_type
from ..sql.base import _NoArg
from ..sql.compiler import DDLCompiler
from ..sql.compiler import InsertmanyvaluesSentinelOpts
from ..sql.compiler import SQLCompiler
from ..sql.elements import quoted_name
from ..util.typing import Final
from ..util.typing import Literal

if typing.TYPE_CHECKING:
    from types import ModuleType

    from .base import Engine
    from .cursor import ResultFetchStrategy
    from .interfaces import _CoreMultiExecuteParams
    from .interfaces import _CoreSingleExecuteParams
    from .interfaces import _DBAPICursorDescription
    from .interfaces import _DBAPIMultiExecuteParams
    from .interfaces import _DBAPISingleExecuteParams
    from .interfaces import _ExecuteOptions
    from .interfaces import _MutableCoreSingleExecuteParams
    from .interfaces import _ParamStyle
    from .interfaces import ConnectArgsType
    from .interfaces import DBAPIConnection
    from .interfaces import IsolationLevel
    from .row import Row
    from .url import URL
    from ..event import _ListenerFnType
    from ..pool import Pool
    from ..pool import PoolProxiedConnection
    from ..sql import Executable
    from ..sql.compiler import Compiled
    from ..sql.compiler import Linting
    from ..sql.compiler import ResultColumnsEntry
    from ..sql.dml import DMLState
    from ..sql.dml import UpdateBase
    from ..sql.elements import BindParameter
    from ..sql.schema import Column
    from ..sql.type_api import _BindProcessorType
    from ..sql.type_api import _ResultProcessorType
    from ..sql.type_api import TypeEngine


# When we're handed literal SQL, ensure it's a SELECT query
SERVER_SIDE_CURSOR_RE = re.compile(r"\s*SELECT", re.I | re.UNICODE)


(
    CACHE_HIT,
    CACHE_MISS,
    CACHING_DISABLED,
    NO_CACHE_KEY,
    NO_DIALECT_SUPPORT,
) = list(CacheStats)


class DefaultDialect(Dialect):
    """Default implementation of Dialect"""

    statement_compiler = compiler.SQLCompiler
    ddl_compiler = compiler.DDLCompiler
    type_compiler_cls = compiler.GenericTypeCompiler

    preparer = compiler.IdentifierPreparer
    supports_alter = True
    supports_comments = False
    supports_constraint_comments = False
    inline_comments = False
    supports_statement_cache = True

    div_is_floordiv = True

    bind_typing = interfaces.BindTyping.NONE

    include_set_input_sizes: Optional[Set[Any]] = None
    exclude_set_input_sizes: Optional[Set[Any]] = None

    # the first value we'd get for an autoincrement column.
    default_sequence_base = 1

    # most DBAPIs happy with this for execute().
    # not cx_oracle.
    execute_sequence_format = tuple

    supports_schemas = True
    supports_views = True
    supports_sequences = False
    sequences_optional = False
    preexecute_autoincrement_sequences = False
    supports_identity_columns = False
    postfetch_lastrowid = True
    favor_returning_over_lastrowid = False
    insert_null_pk_still_autoincrements = False
    update_returning = False
    delete_returning = False
    update_returning_multifrom = False
    delete_returning_multifrom = False
    insert_returning = False

    cte_follows_insert = False

    supports_native_enum = False
    supports_native_boolean = False
    supports_native_uuid = False
    returns_native_bytes = False

    non_native_boolean_check_constraint = True

    supports_simple_order_by_label = True

    tuple_in_values = False

    connection_characteristics = util.immutabledict(
        {
            "isolation_level": characteristics.IsolationLevelCharacteristic(),
            "logging_token": characteristics.LoggingTokenCharacteristic(),
        }
    )

    engine_config_types: Mapping[str, Any] = util.immutabledict(
        {
            "pool_timeout": util.asint,
            "echo": util.bool_or_str("debug"),
            "echo_pool": util.bool_or_str("debug"),
            "pool_recycle": util.asint,
            "pool_size": util.asint,
            "max_overflow": util.asint,
            "future": util.asbool,
        }
    )

    # if the NUMERIC type
    # returns decimal.Decimal.
    # *not* the FLOAT type however.
    supports_native_decimal = False

    name = "default"

    # length at which to truncate
    # any identifier.
    max_identifier_length = 9999
    _user_defined_max_identifier_length: Optional[int] = None

    isolation_level: Optional[str] = None

    # sub-categories of max_identifier_length.
    # currently these accommodate for MySQL which allows alias names
    # of 255 but DDL names only of 64.
    max_index_name_length: Optional[int] = None
    max_constraint_name_length: Optional[int] = None

    supports_sane_rowcount = True
    supports_sane_multi_rowcount = True
    colspecs: MutableMapping[Type[TypeEngine[Any]], Type[TypeEngine[Any]]] = {}
    default_paramstyle = "named"

    supports_default_values = False
    """dialect supports INSERT... DEFAULT VALUES syntax"""

    supports_default_metavalue = False
    """dialect supports INSERT... VALUES (DEFAULT) syntax"""

    default_metavalue_token = "DEFAULT"
    """for INSERT... VALUES (DEFAULT) syntax, the token to put in the
    parenthesis."""

    # not sure if this is a real thing but the compiler will deliver it
    # if this is the only flag enabled.
    supports_empty_insert = True
    """dialect supports INSERT () VALUES ()"""

    supports_multivalues_insert = False

    use_insertmanyvalues: bool = False

    use_insertmanyvalues_wo_returning: bool = False

    insertmanyvalues_implicit_sentinel: InsertmanyvaluesSentinelOpts = (
        InsertmanyvaluesSentinelOpts.NOT_SUPPORTED
    )

    insertmanyvalues_page_size: int = 1000
    insertmanyvalues_max_parameters = 32700

    supports_is_distinct_from = True

    supports_server_side_cursors = False

    server_side_cursors = False

    # extra record-level locking features (#4860)
    supports_for_update_of = False

    server_version_info = None

    default_schema_name: Optional[str] = None

    # indicates symbol names are
    # UPPERCASED if they are case insensitive
    # within the database.
    # if this is True, the methods normalize_name()
    # and denormalize_name() must be provided.
    requires_name_normalize = False

    is_async = False

    has_terminate = False

    # TODO: this is not to be part of 2.0.  implement rudimentary binary
    # literals for SQLite, PostgreSQL, MySQL only within
    # _Binary.literal_processor
    _legacy_binary_type_literal_encoding = "utf-8"

    @util.deprecated_params(
        empty_in_strategy=(
            "1.4",
            "The :paramref:`_sa.create_engine.empty_in_strategy` keyword is "
            "deprecated, and no longer has any effect.  All IN expressions "
            "are now rendered using "
            'the "expanding parameter" strategy which renders a set of bound'
            'expressions, or an "empty set" SELECT, at statement execution'
            "time.",
        ),
        server_side_cursors=(
            "1.4",
            "The :paramref:`_sa.create_engine.server_side_cursors` parameter "
            "is deprecated and will be removed in a future release.  Please "
            "use the "
            ":paramref:`_engine.Connection.execution_options.stream_results` "
            "parameter.",
        ),
    )
    def __init__(
        self,
        paramstyle: Optional[_ParamStyle] = None,
        isolation_level: Optional[IsolationLevel] = None,
        dbapi: Optional[ModuleType] = None,
        implicit_returning: Literal[True] = True,
        supports_native_boolean: Optional[bool] = None,
        max_identifier_length: Optional[int] = None,
        label_length: Optional[int] = None,
        insertmanyvalues_page_size: Union[_NoArg, int] = _NoArg.NO_ARG,
        use_insertmanyvalues: Optional[bool] = None,
        # util.deprecated_params decorator cannot render the
        # Linting.NO_LINTING constant
        compiler_linting: Linting = int(compiler.NO_LINTING),  # type: ignore
        server_side_cursors: bool = False,
        **kwargs: Any,
    ):
        if server_side_cursors:
            if not self.supports_server_side_cursors:
                raise exc.ArgumentError(
                    "Dialect %s does not support server side cursors" % self
                )
            else:
                self.server_side_cursors = True

        if getattr(self, "use_setinputsizes", False):
            util.warn_deprecated(
                "The dialect-level use_setinputsizes attribute is "
                "deprecated.  Please use "
                "bind_typing = BindTyping.SETINPUTSIZES",
                "2.0",
            )
            self.bind_typing = interfaces.BindTyping.SETINPUTSIZES

        self.positional = False
        self._ischema = None

        self.dbapi = dbapi

        if paramstyle is not None:
            self.paramstyle = paramstyle
        elif self.dbapi is not None:
            self.paramstyle = self.dbapi.paramstyle
        else:
            self.paramstyle = self.default_paramstyle
        self.positional = self.paramstyle in (
            "qmark",
            "format",
            "numeric",
            "numeric_dollar",
        )
        self.identifier_preparer = self.preparer(self)
        self._on_connect_isolation_level = isolation_level

        legacy_tt_callable = getattr(self, "type_compiler", None)
        if legacy_tt_callable is not None:
            tt_callable = cast(
                Type[compiler.GenericTypeCompiler],
                self.type_compiler,
            )
        else:
            tt_callable = self.type_compiler_cls

        self.type_compiler_instance = self.type_compiler = tt_callable(self)

        if supports_native_boolean is not None:
            self.supports_native_boolean = supports_native_boolean

        self._user_defined_max_identifier_length = max_identifier_length
        if self._user_defined_max_identifier_length:
            self.max_identifier_length = (
                self._user_defined_max_identifier_length
            )
        self.label_length = label_length
        self.compiler_linting = compiler_linting

        if use_insertmanyvalues is not None:
            self.use_insertmanyvalues = use_insertmanyvalues

        if insertmanyvalues_page_size is not _NoArg.NO_ARG:
            self.insertmanyvalues_page_size = insertmanyvalues_page_size

    @property
    @util.deprecated(
        "2.0",
        "full_returning is deprecated, please use insert_returning, "
        "update_returning, delete_returning",
    )
    def full_returning(self):
        return (
            self.insert_returning
            and self.update_returning
            and self.delete_returning
        )

    @util.memoized_property
    def insert_executemany_returning(self):
        """Default implementation for insert_executemany_returning, if not
        otherwise overridden by the specific dialect.

        The default dialect determines "insert_executemany_returning" is
        available if the dialect in use has opted into using the
        "use_insertmanyvalues" feature. If they haven't opted into that, then
        this attribute is False, unless the dialect in question overrides this
        and provides some other implementation (such as the Oracle Database
        dialects).

        """
        return self.insert_returning and self.use_insertmanyvalues

    @util.memoized_property
    def insert_executemany_returning_sort_by_parameter_order(self):
        """Default implementation for
        insert_executemany_returning_deterministic_order, if not otherwise
        overridden by the specific dialect.

        The default dialect determines "insert_executemany_returning" can have
        deterministic order only if the dialect in use has opted into using the
        "use_insertmanyvalues" feature, which implements deterministic ordering
        using client side sentinel columns only by default.  The
        "insertmanyvalues" feature also features alternate forms that can
        use server-generated PK values as "sentinels", but those are only
        used if the :attr:`.Dialect.insertmanyvalues_implicit_sentinel`
        bitflag enables those alternate SQL forms, which are disabled
        by default.

        If the dialect in use hasn't opted into that, then this attribute is
        False, unless the dialect in question overrides this and provides some
        other implementation (such as the Oracle Database dialects).

        """
        return self.insert_returning and self.use_insertmanyvalues

    update_executemany_returning = False
    delete_executemany_returning = False

    @util.memoized_property
    def loaded_dbapi(self) -> ModuleType:
        if self.dbapi is None:
            raise exc.InvalidRequestError(
                f"Dialect {self} does not have a Python DBAPI established "
                "and cannot be used for actual database interaction"
            )
        return self.dbapi

    @util.memoized_property
    def _bind_typing_render_casts(self):
        return self.bind_typing is interfaces.BindTyping.RENDER_CASTS

    def _ensure_has_table_connection(self, arg: Connection) -> None:
        if not isinstance(arg, Connection):
            raise exc.ArgumentError(
                "The argument passed to Dialect.has_table() should be a "
                "%s, got %s. "
                "Additionally, the Dialect.has_table() method is for "
                "internal dialect "
                "use only; please use "
                "``inspect(some_engine).has_table(<tablename>>)`` "
                "for public API use." % (Connection, type(arg))
            )

    @util.memoized_property
    def _supports_statement_cache(self):
        ssc = self.__class__.__dict__.get("supports_statement_cache", None)
        if ssc is None:
            util.warn(
                "Dialect %s:%s will not make use of SQL compilation caching "
                "as it does not set the 'supports_statement_cache' attribute "
                "to ``True``.  This can have "
                "significant performance implications including some "
                "performance degradations in comparison to prior SQLAlchemy "
                "versions.  Dialect maintainers should seek to set this "
                "attribute to True after appropriate development and testing "
                "for SQLAlchemy 1.4 caching support.   Alternatively, this "
                "attribute may be set to False which will disable this "
                "warning." % (self.name, self.driver),
                code="cprf",
            )

        return bool(ssc)

    @util.memoized_property
    def _type_memos(self):
        return weakref.WeakKeyDictionary()

    @property
    def dialect_description(self):
        return self.name + "+" + self.driver

    @property
    def supports_sane_rowcount_returning(self):
        """True if this dialect supports sane rowcount even if RETURNING is
        in use.

        For dialects that don't support RETURNING, this is synonymous with
        ``supports_sane_rowcount``.

        """
        return self.supports_sane_rowcount

    @classmethod
    def get_pool_class(cls, url: URL) -> Type[Pool]:
        return getattr(cls, "poolclass", pool.QueuePool)

    def get_dialect_pool_class(self, url: URL) -> Type[Pool]:
        return self.get_pool_class(url)

    @classmethod
    def load_provisioning(cls):
        package = ".".join(cls.__module__.split(".")[0:-1])
        try:
            __import__(package + ".provision")
        except ImportError:
            pass

    def _builtin_onconnect(self) -> Optional[_ListenerFnType]:
        if self._on_connect_isolation_level is not None:

            def builtin_connect(dbapi_conn, conn_rec):
                self._assert_and_set_isolation_level(
                    dbapi_conn, self._on_connect_isolation_level
                )

            return builtin_connect
        else:
            return None

    def initialize(self, connection: Connection) -> None:
        try:
            self.server_version_info = self._get_server_version_info(
                connection
            )
        except NotImplementedError:
            self.server_version_info = None
        try:
            self.default_schema_name = self._get_default_schema_name(
                connection
            )
        except NotImplementedError:
            self.default_schema_name = None

        try:
            self.default_isolation_level = self.get_default_isolation_level(
                connection.connection.dbapi_connection
            )
        except NotImplementedError:
            self.default_isolation_level = None

        if not self._user_defined_max_identifier_length:
            max_ident_length = self._check_max_identifier_length(connection)
            if max_ident_length:
                self.max_identifier_length = max_ident_length

        if (
            self.label_length
            and self.label_length > self.max_identifier_length
        ):
            raise exc.ArgumentError(
                "Label length of %d is greater than this dialect's"
                " maximum identifier length of %d"
                % (self.label_length, self.max_identifier_length)
            )

    def on_connect(self) -> Optional[Callable[[Any], Any]]:
        # inherits the docstring from interfaces.Dialect.on_connect
        return None

    def _check_max_identifier_length(self, connection):
        """Perform a connection / server version specific check to determine
        the max_identifier_length.

        If the dialect's class level max_identifier_length should be used,
        can return None.

        .. versionadded:: 1.3.9

        """
        return None

    def get_default_isolation_level(self, dbapi_conn):
        """Given a DBAPI connection, return its isolation level, or
        a default isolation level if one cannot be retrieved.

        May be overridden by subclasses in order to provide a
        "fallback" isolation level for databases that cannot reliably
        retrieve the actual isolation level.

        By default, calls the :meth:`_engine.Interfaces.get_isolation_level`
        method, propagating any exceptions raised.

        .. versionadded:: 1.3.22

        """
        return self.get_isolation_level(dbapi_conn)

    def type_descriptor(self, typeobj):
        """Provide a database-specific :class:`.TypeEngine` object, given
        the generic object which comes from the types module.

        This method looks for a dictionary called
        ``colspecs`` as a class or instance-level variable,
        and passes on to :func:`_types.adapt_type`.

        """
        return type_api.adapt_type(typeobj, self.colspecs)

    def has_index(self, connection, table_name, index_name, schema=None, **kw):
        if not self.has_table(connection, table_name, schema=schema, **kw):
            return False
        for idx in self.get_indexes(
            connection, table_name, schema=schema, **kw
        ):
            if idx["name"] == index_name:
                return True
        else:
            return False

    def has_schema(
        self, connection: Connection, schema_name: str, **kw: Any
    ) -> bool:
        return schema_name in self.get_schema_names(connection, **kw)

    def validate_identifier(self, ident: str) -> None:
        if len(ident) > self.max_identifier_length:
            raise exc.IdentifierError(
                "Identifier '%s' exceeds maximum length of %d characters"
                % (ident, self.max_identifier_length)
            )

    def connect(self, *cargs: Any, **cparams: Any) -> DBAPIConnection:
        # inherits the docstring from interfaces.Dialect.connect
        return self.loaded_dbapi.connect(*cargs, **cparams)  # type: ignore[no-any-return]  # NOQA: E501

    def create_connect_args(self, url: URL) -> ConnectArgsType:
        # inherits the docstring from interfaces.Dialect.create_connect_args
        opts = url.translate_connect_args()
        opts.update(url.query)
        return ([], opts)

    def set_engine_execution_options(
        self, engine: Engine, opts: Mapping[str, Any]
    ) -> None:
        supported_names = set(self.connection_characteristics).intersection(
            opts
        )
        if supported_names:
            characteristics: Mapping[str, Any] = util.immutabledict(
                (name, opts[name]) for name in supported_names
            )

            @event.listens_for(engine, "engine_connect")
            def set_connection_characteristics(connection):
                self._set_connection_characteristics(
                    connection, characteristics
                )

    def set_connection_execution_options(
        self, connection: Connection, opts: Mapping[str, Any]
    ) -> None:
        supported_names = set(self.connection_characteristics).intersection(
            opts
        )
        if supported_names:
            characteristics: Mapping[str, Any] = util.immutabledict(
                (name, opts[name]) for name in supported_names
            )
            self._set_connection_characteristics(connection, characteristics)

    def _set_connection_characteristics(self, connection, characteristics):
        characteristic_values = [
            (name, self.connection_characteristics[name], value)
            for name, value in characteristics.items()
        ]

        if connection.in_transaction():
            trans_objs = [
                (name, obj)
                for name, obj, _ in characteristic_values
                if obj.transactional
            ]
            if trans_objs:
                raise exc.InvalidRequestError(
                    "This connection has already initialized a SQLAlchemy "
                    "Transaction() object via begin() or autobegin; "
                    "%s may not be altered unless rollback() or commit() "
                    "is called first."
                    % (", ".join(name for name, obj in trans_objs))
                )

        dbapi_connection = connection.connection.dbapi_connection
        for _, characteristic, value in characteristic_values:
            characteristic.set_connection_characteristic(
                self, connection, dbapi_connection, value
            )
        connection.connection._connection_record.finalize_callback.append(
            functools.partial(self._reset_characteristics, characteristics)
        )

    def _reset_characteristics(self, characteristics, dbapi_connection):
        for characteristic_name in characteristics:
            characteristic = self.connection_characteristics[
                characteristic_name
            ]
            characteristic.reset_characteristic(self, dbapi_connection)

    def do_begin(self, dbapi_connection):
        pass

    def do_rollback(self, dbapi_connection):
        dbapi_connection.rollback()

    def do_commit(self, dbapi_connection):
        dbapi_connection.commit()

    def do_terminate(self, dbapi_connection):
        self.do_close(dbapi_connection)

    def do_close(self, dbapi_connection):
        dbapi_connection.close()

    @util.memoized_property
    def _dialect_specific_select_one(self):
        return str(expression.select(1).compile(dialect=self))

    def _do_ping_w_event(self, dbapi_connection: DBAPIConnection) -> bool:
        try:
            return self.do_ping(dbapi_connection)
        except self.loaded_dbapi.Error as err:
            is_disconnect = self.is_disconnect(err, dbapi_connection, None)

            if self._has_events:
                try:
                    Connection._handle_dbapi_exception_noconnection(
                        err,
                        self,
                        is_disconnect=is_disconnect,
                        invalidate_pool_on_disconnect=False,
                        is_pre_ping=True,
                    )
                except exc.StatementError as new_err:
                    is_disconnect = new_err.connection_invalidated

            if is_disconnect:
                return False
            else:
                raise

    def do_ping(self, dbapi_connection: DBAPIConnection) -> bool:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute(self._dialect_specific_select_one)
        finally:
            cursor.close()
        return True

    def create_xid(self):
        """Create a random two-phase transaction ID.

        This id will be passed to do_begin_twophase(), do_rollback_twophase(),
        do_commit_twophase().  Its format is unspecified.
        """

        return "_sa_%032x" % random.randint(0, 2**128)

    def do_savepoint(self, connection, name):
        connection.execute(expression.SavepointClause(name))

    def do_rollback_to_savepoint(self, connection, name):
        connection.execute(expression.RollbackToSavepointClause(name))

    def do_release_savepoint(self, connection, name):
        connection.execute(expression.ReleaseSavepointClause(name))

    def _deliver_insertmanyvalues_batches(
        self,
        connection,
        cursor,
        statement,
        parameters,
        generic_setinputsizes,
        context,
    ):
        context = cast(DefaultExecutionContext, context)
        compiled = cast(SQLCompiler, context.compiled)

        _composite_sentinel_proc: Sequence[
            Optional[_ResultProcessorType[Any]]
        ] = ()
        _scalar_sentinel_proc: Optional[_ResultProcessorType[Any]] = None
        _sentinel_proc_initialized: bool = False

        compiled_parameters = context.compiled_parameters

        imv = compiled._insertmanyvalues
        assert imv is not None

        is_returning: Final[bool] = bool(compiled.effective_returning)
        batch_size = context.execution_options.get(
            "insertmanyvalues_page_size", self.insertmanyvalues_page_size
        )

        if compiled.schema_translate_map:
            schema_translate_map = context.execution_options.get(
                "schema_translate_map", {}
            )
        else:
            schema_translate_map = None

        if is_returning:
            result: Optional[List[Any]] = []
            context._insertmanyvalues_rows = result

            sort_by_parameter_order = imv.sort_by_parameter_order

        else:
            sort_by_parameter_order = False
            result = None

        for imv_batch in compiled._deliver_insertmanyvalues_batches(
            statement,
            parameters,
            compiled_parameters,
            generic_setinputsizes,
            batch_size,
            sort_by_parameter_order,
            schema_translate_map,
        ):
            yield imv_batch

            if is_returning:

                try:
                    rows = context.fetchall_for_returning(cursor)
                except BaseException as be:
                    connection._handle_dbapi_exception(
                        be,
                        sql_util._long_statement(imv_batch.replaced_statement),
                        imv_batch.replaced_parameters,
                        None,
                        context,
                        is_sub_exec=True,
                    )

                # I would have thought "is_returning: Final[bool]"
                # would have assured this but pylance thinks not
                assert result is not None

                if imv.num_sentinel_columns and not imv_batch.is_downgraded:
                    composite_sentinel = imv.num_sentinel_columns > 1
                    if imv.implicit_sentinel:
                        # for implicit sentinel, which is currently single-col
                        # integer autoincrement, do a simple sort.
                        assert not composite_sentinel
                        result.extend(
                            sorted(rows, key=operator.itemgetter(-1))
                        )
                        continue

                    # otherwise, create dictionaries to match up batches
                    # with parameters
                    assert imv.sentinel_param_keys
                    assert imv.sentinel_columns

                    _nsc = imv.num_sentinel_columns

                    if not _sentinel_proc_initialized:
                        if composite_sentinel:
                            _composite_sentinel_proc = [
                                col.type._cached_result_processor(
                                    self, cursor_desc[1]
                                )
                                for col, cursor_desc in zip(
                                    imv.sentinel_columns,
                                    cursor.description[-_nsc:],
                                )
                            ]
                        else:
                            _scalar_sentinel_proc = (
                                imv.sentinel_columns[0]
                            ).type._cached_result_processor(
                                self, cursor.description[-1][1]
                            )
                        _sentinel_proc_initialized = True

                    rows_by_sentinel: Union[
                        Dict[Tuple[Any, ...], Any],
                        Dict[Any, Any],
                    ]
                    if composite_sentinel:
                        rows_by_sentinel = {
                            tuple(
                                (proc(val) if proc else val)
                                for val, proc in zip(
                                    row[-_nsc:], _composite_sentinel_proc
                                )
                            ): row
                            for row in rows
                        }
                    elif _scalar_sentinel_proc:
                        rows_by_sentinel = {
                            _scalar_sentinel_proc(row[-1]): row for row in rows
                        }
                    else:
                        rows_by_sentinel = {row[-1]: row for row in rows}

                    if len(rows_by_sentinel) != len(imv_batch.batch):
                        # see test_insert_exec.py::
                        # IMVSentinelTest::test_sentinel_incorrect_rowcount
                        # for coverage / demonstration
                        raise exc.InvalidRequestError(
                            f"Sentinel-keyed result set did not produce "
                            f"correct number of rows {len(imv_batch.batch)}; "
                            "produced "
                            f"{len(rows_by_sentinel)}.  Please ensure the "
                            "sentinel column is fully unique and populated in "
                            "all cases."
                        )

                    try:
                        ordered_rows = [
                            rows_by_sentinel[sentinel_keys]
                            for sentinel_keys in imv_batch.sentinel_values
                        ]
                    except KeyError as ke:
                        # see test_insert_exec.py::
                        # IMVSentinelTest::test_sentinel_cant_match_keys
                        # for coverage / demonstration
                        raise exc.InvalidRequestError(
                            f"Can't match sentinel values in result set to "
                            f"parameter sets; key {ke.args[0]!r} was not "
                            "found. "
                            "There may be a mismatch between the datatype "
                            "passed to the DBAPI driver vs. that which it "
                            "returns in a result row.  Ensure the given "
                            "Python value matches the expected result type "
                            "*exactly*, taking care to not rely upon implicit "
                            "conversions which may occur such as when using "
                            "strings in place of UUID or integer values, etc. "
                        ) from ke

                    result.extend(ordered_rows)

                else:
                    result.extend(rows)

    def do_executemany(self, cursor, statement, parameters, context=None):
        cursor.executemany(statement, parameters)

    def do_execute(self, cursor, statement, parameters, context=None):
        cursor.execute(statement, parameters)

    def do_execute_no_params(self, cursor, statement, context=None):
        cursor.execute(statement)

    def is_disconnect(
        self,
        e: Exception,
        connection: Union[
            pool.PoolProxiedConnection, interfaces.DBAPIConnection, None
        ],
        cursor: Optional[interfaces.DBAPICursor],
    ) -> bool:
        return False

    @util.memoized_instancemethod
    def _gen_allowed_isolation_levels(self, dbapi_conn):
        try:
            raw_levels = list(self.get_isolation_level_values(dbapi_conn))
        except NotImplementedError:
            return None
        else:
            normalized_levels = [
                level.replace("_", " ").upper() for level in raw_levels
            ]
            if raw_levels != normalized_levels:
                raise ValueError(
                    f"Dialect {self.name!r} get_isolation_level_values() "
                    f"method should return names as UPPERCASE using spaces, "
                    f"not underscores; got "
                    f"{sorted(set(raw_levels).difference(normalized_levels))}"
                )
            return tuple(normalized_levels)

    def _assert_and_set_isolation_level(self, dbapi_conn, level):
        level = level.replace("_", " ").upper()

        _allowed_isolation_levels = self._gen_allowed_isolation_levels(
            dbapi_conn
        )
        if (
            _allowed_isolation_levels
            and level not in _allowed_isolation_levels
        ):
            raise exc.ArgumentError(
                f"Invalid value {level!r} for isolation_level. "
                f"Valid isolation levels for {self.name!r} are "
                f"{', '.join(_allowed_isolation_levels)}"
            )

        self.set_isolation_level(dbapi_conn, level)

    def reset_isolation_level(self, dbapi_conn):
        if self._on_connect_isolation_level is not None:
            assert (
                self._on_connect_isolation_level == "AUTOCOMMIT"
                or self._on_connect_isolation_level
                == self.default_isolation_level
            )
            self._assert_and_set_isolation_level(
                dbapi_conn, self._on_connect_isolation_level
            )
        else:
            assert self.default_isolation_level is not None
            self._assert_and_set_isolation_level(
                dbapi_conn,
                self.default_isolation_level,
            )

    def normalize_name(self, name):
        if name is None:
            return None

        name_lower = name.lower()
        name_upper = name.upper()

        if name_upper == name_lower:
            # name has no upper/lower conversion, e.g. non-european characters.
            # return unchanged
            return name
        elif name_upper == name and not (
            self.identifier_preparer._requires_quotes
        )(name_lower):
            # name is all uppercase and doesn't require quoting; normalize
            # to all lower case
            return name_lower
        elif name_lower == name:
            # name is all lower case, which if denormalized means we need to
            # force quoting on it
            return quoted_name(name, quote=True)
        else:
            # name is mixed case, means it will be quoted in SQL when used
            # later, no normalizes
            return name

    def denormalize_name(self, name):
        if name is None:
            return None

        name_lower = name.lower()
        name_upper = name.upper()

        if name_upper == name_lower:
            # name has no upper/lower conversion, e.g. non-european characters.
            # return unchanged
            return name
        elif name_lower == name and not (
            self.identifier_preparer._requires_quotes
        )(name_lower):
            name = name_upper
        return name

    def get_driver_connection(self, connection):
        return connection

    def _overrides_default(self, method):
        return (
            getattr(type(self), method).__code__
            is not getattr(DefaultDialect, method).__code__
        )

    def _default_multi_reflect(
        self,
        single_tbl_method,
        connection,
        kind,
        schema,
        filter_names,
        scope,
        **kw,
    ):
        names_fns = []
        temp_names_fns = []
        if ObjectKind.TABLE in kind:
            names_fns.append(self.get_table_names)
            temp_names_fns.append(self.get_temp_table_names)
        if ObjectKind.VIEW in kind:
            names_fns.append(self.get_view_names)
            temp_names_fns.append(self.get_temp_view_names)
        if ObjectKind.MATERIALIZED_VIEW in kind:
            names_fns.append(self.get_materialized_view_names)
            # no temp materialized view at the moment
            # temp_names_fns.append(self.get_temp_materialized_view_names)

        unreflectable = kw.pop("unreflectable", {})

        if (
            filter_names
            and scope is ObjectScope.ANY
            and kind is ObjectKind.ANY
        ):
            # if names are given and no qualification on type of table
            # (i.e. the Table(..., autoload) case), take the names as given,
            # don't run names queries. If a table does not exit
            # NoSuchTableError is raised and it's skipped

            # this also suits the case for mssql where we can reflect
            # individual temp tables but there's no temp_names_fn
            names = filter_names
        else:
            names = []
            name_kw = {"schema": schema, **kw}
            fns = []
            if ObjectScope.DEFAULT in scope:
                fns.extend(names_fns)
            if ObjectScope.TEMPORARY in scope:
                fns.extend(temp_names_fns)

            for fn in fns:
                try:
                    names.extend(fn(connection, **name_kw))
                except NotImplementedError:
                    pass

        if filter_names:
            filter_names = set(filter_names)

        # iterate over all the tables/views and call the single table method
        for table in names:
            if not filter_names or table in filter_names:
                key = (schema, table)
                try:
                    yield (
                        key,
                        single_tbl_method(
                            connection, table, schema=schema, **kw
                        ),
                    )
                except exc.UnreflectableTableError as err:
                    if key not in unreflectable:
                        unreflectable[key] = err
                except exc.NoSuchTableError:
                    pass

    def get_multi_table_options(self, connection, **kw):
        return self._default_multi_reflect(
            self.get_table_options, connection, **kw
        )

    def get_multi_columns(self, connection, **kw):
        return self._default_multi_reflect(self.get_columns, connection, **kw)

    def get_multi_pk_constraint(self, connection, **kw):
        return self._default_multi_reflect(
            self.get_pk_constraint, connection, **kw
        )

    def get_multi_foreign_keys(self, connection, **kw):
        return self._default_multi_reflect(
            self.get_foreign_keys, connection, **kw
        )

    def get_multi_indexes(self, connection, **kw):
        return self._default_multi_reflect(self.get_indexes, connection, **kw)

    def get_multi_unique_constraints(self, connection, **kw):
        return self._default_multi_reflect(
            self.get_unique_constraints, connection, **kw
        )

    def get_multi_check_constraints(self, connection, **kw):
        return self._default_multi_reflect(
            self.get_check_constraints, connection, **kw
        )

    def get_multi_table_comment(self, connection, **kw):
        return self._default_multi_reflect(
            self.get_table_comment, connection, **kw
        )


class StrCompileDialect(DefaultDialect):
    statement_compiler = compiler.StrSQLCompiler
    ddl_compiler = compiler.DDLCompiler
    type_compiler_cls = compiler.StrSQLTypeCompiler
    preparer = compiler.IdentifierPreparer

    insert_returning = True
    update_returning = True
    delete_returning = True

    supports_statement_cache = True

    supports_identity_columns = True

    supports_sequences = True
    sequences_optional = True
    preexecute_autoincrement_sequences = False

    supports_native_boolean = True

    supports_multivalues_insert = True
    supports_simple_order_by_label = True


class DefaultExecutionContext(ExecutionContext):
    isinsert = False
    isupdate = False
    isdelete = False
    is_crud = False
    is_text = False
    isddl = False

    execute_style: ExecuteStyle = ExecuteStyle.EXECUTE

    compiled: Optional[Compiled] = None
    result_column_struct: Optional[
        Tuple[List[ResultColumnsEntry], bool, bool, bool, bool]
    ] = None
    returned_default_rows: Optional[Sequence[Row[Any]]] = None

    execution_options: _ExecuteOptions = util.EMPTY_DICT

    cursor_fetch_strategy = _cursor._DEFAULT_FETCH

    invoked_statement: Optional[Executable] = None

    _is_implicit_returning = False
    _is_explicit_returning = False
    _is_supplemental_returning = False
    _is_server_side = False

    _soft_closed = False

    _rowcount: Optional[int] = None

    # a hook for SQLite's translation of
    # result column names
    # NOTE: pyhive is using this hook, can't remove it :(
    _translate_colname: Optional[Callable[[str], str]] = None

    _expanded_parameters: Mapping[str, List[str]] = util.immutabledict()
    """used by set_input_sizes().

    This collection comes from ``ExpandedState.parameter_expansion``.

    """

    cache_hit = NO_CACHE_KEY

    root_connection: Connection
    _dbapi_connection: PoolProxiedConnection
    dialect: Dialect
    unicode_statement: str
    cursor: DBAPICursor
    compiled_parameters: List[_MutableCoreSingleExecuteParams]
    parameters: _DBAPIMultiExecuteParams
    extracted_parameters: Optional[Sequence[BindParameter[Any]]]

    _empty_dict_params = cast("Mapping[str, Any]", util.EMPTY_DICT)

    _insertmanyvalues_rows: Optional[List[Tuple[Any, ...]]] = None
    _num_sentinel_cols: int = 0

    @classmethod
    def _init_ddl(
        cls,
        dialect: Dialect,
        connection: Connection,
        dbapi_connection: PoolProxiedConnection,
        execution_options: _ExecuteOptions,
        compiled_ddl: DDLCompiler,
    ) -> ExecutionContext:
        """Initialize execution context for an ExecutableDDLElement
        construct."""

        self = cls.__new__(cls)
        self.root_connection = connection
        self._dbapi_connection = dbapi_connection
        self.dialect = connection.dialect

        self.compiled = compiled = compiled_ddl
        self.isddl = True

        self.execution_options = execution_options

        self.unicode_statement = str(compiled)
        if compiled.schema_translate_map:
            schema_translate_map = self.execution_options.get(
                "schema_translate_map", {}
            )

            rst = compiled.preparer._render_schema_translates
            self.unicode_statement = rst(
                self.unicode_statement, schema_translate_map
            )

        self.statement = self.unicode_statement

        self.cursor = self.create_cursor()
        self.compiled_parameters = []

        if dialect.positional:
            self.parameters = [dialect.execute_sequence_format()]
        else:
            self.parameters = [self._empty_dict_params]

        return self

    @classmethod
    def _init_compiled(
        cls,
        dialect: Dialect,
        connection: Connection,
        dbapi_connection: PoolProxiedConnection,
        execution_options: _ExecuteOptions,
        compiled: SQLCompiler,
        parameters: _CoreMultiExecuteParams,
        invoked_statement: Executable,
        extracted_parameters: Optional[Sequence[BindParameter[Any]]],
        cache_hit: CacheStats = CacheStats.CACHING_DISABLED,
    ) -> ExecutionContext:
        """Initialize execution context for a Compiled construct."""

        self = cls.__new__(cls)
        self.root_connection = connection
        self._dbapi_connection = dbapi_connection
        self.dialect = connection.dialect
        self.extracted_parameters = extracted_parameters
        self.invoked_statement = invoked_statement
        self.compiled = compiled
        self.cache_hit = cache_hit

        self.execution_options = execution_options

        self.result_column_struct = (
            compiled._result_columns,
            compiled._ordered_columns,
            compiled._textual_ordered_columns,
            compiled._ad_hoc_textual,
            compiled._loose_column_name_matching,
        )

        self.isinsert = ii = compiled.isinsert
        self.isupdate = iu = compiled.isupdate
        self.isdelete = id_ = compiled.isdelete
        self.is_text = compiled.isplaintext

        if ii or iu or id_:
            dml_statement = compiled.compile_state.statement  # type: ignore
            if TYPE_CHECKING:
                assert isinstance(dml_statement, UpdateBase)
            self.is_crud = True
            self._is_explicit_returning = ier = bool(dml_statement._returning)
            self._is_implicit_returning = iir = bool(
                compiled.implicit_returning
            )
            if iir and dml_statement._supplemental_returning:
                self._is_supplemental_returning = True

            # dont mix implicit and explicit returning
            assert not (iir and ier)

            if (ier or iir) and compiled.for_executemany:
                if ii and not self.dialect.insert_executemany_returning:
                    raise exc.InvalidRequestError(
                        f"Dialect {self.dialect.dialect_description} with "
                        f"current server capabilities does not support "
                        "INSERT..RETURNING when executemany is used"
                    )
                elif (
                    ii
                    and dml_statement._sort_by_parameter_order
                    and not self.dialect.insert_executemany_returning_sort_by_parameter_order  # noqa: E501
                ):
                    raise exc.InvalidRequestError(
                        f"Dialect {self.dialect.dialect_description} with "
                        f"current server capabilities does not support "
                        "INSERT..RETURNING with deterministic row ordering "
                        "when executemany is used"
                    )
                elif (
                    ii
                    and self.dialect.use_insertmanyvalues
                    and not compiled._insertmanyvalues
                ):
                    raise exc.InvalidRequestError(
                        'Statement does not have "insertmanyvalues" '
                        "enabled, can't use INSERT..RETURNING with "
                        "executemany in this case."
                    )
                elif iu and not self.dialect.update_executemany_returning:
                    raise exc.InvalidRequestError(
                        f"Dialect {self.dialect.dialect_description} with "
                        f"current server capabilities does not support "
                        "UPDATE..RETURNING when executemany is used"
                    )
                elif id_ and not self.dialect.delete_executemany_returning:
                    raise exc.InvalidRequestError(
                        f"Dialect {self.dialect.dialect_description} with "
                        f"current server capabilities does not support "
                        "DELETE..RETURNING when executemany is used"
                    )

        if not parameters:
            self.compiled_parameters = [
                compiled.construct_params(
                    extracted_parameters=extracted_parameters,
                    escape_names=False,
                )
            ]
        else:
            self.compiled_parameters = [
                compiled.construct_params(
                    m,
                    escape_names=False,
                    _group_number=grp,
                    extracted_parameters=extracted_parameters,
                )
                for grp, m in enumerate(parameters)
            ]

            if len(parameters) > 1:
                if self.isinsert and compiled._insertmanyvalues:
                    self.execute_style = ExecuteStyle.INSERTMANYVALUES

                    imv = compiled._insertmanyvalues
                    if imv.sentinel_columns is not None:
                        self._num_sentinel_cols = imv.num_sentinel_columns
                else:
                    self.execute_style = ExecuteStyle.EXECUTEMANY

        self.unicode_statement = compiled.string

        self.cursor = self.create_cursor()

        if self.compiled.insert_prefetch or self.compiled.update_prefetch:
            self._process_execute_defaults()

        processors = compiled._bind_processors

        flattened_processors: Mapping[
            str, _BindProcessorType[Any]
        ] = processors  # type: ignore[assignment]

        if compiled.literal_execute_params or compiled.post_compile_params:
            if self.executemany:
                raise exc.InvalidRequestError(
                    "'literal_execute' or 'expanding' parameters can't be "
                    "used with executemany()"
                )

            expanded_state = compiled._process_parameters_for_postcompile(
                self.compiled_parameters[0]
            )

            # re-assign self.unicode_statement
            self.unicode_statement = expanded_state.statement

            self._expanded_parameters = expanded_state.parameter_expansion

            flattened_processors = dict(processors)  # type: ignore
            flattened_processors.update(expanded_state.processors)
            positiontup = expanded_state.positiontup
        elif compiled.positional:
            positiontup = self.compiled.positiontup
        else:
            positiontup = None

        if compiled.schema_translate_map:
            schema_translate_map = self.execution_options.get(
                "schema_translate_map", {}
            )
            rst = compiled.preparer._render_schema_translates
            self.unicode_statement = rst(
                self.unicode_statement, schema_translate_map
            )

        # final self.unicode_statement is now assigned, encode if needed
        # by dialect
        self.statement = self.unicode_statement

        # Convert the dictionary of bind parameter values
        # into a dict or list to be sent to the DBAPI's
        # execute() or executemany() method.

        if compiled.positional:
            core_positional_parameters: MutableSequence[Sequence[Any]] = []
            assert positiontup is not None
            for compiled_params in self.compiled_parameters:
                l_param: List[Any] = [
                    (
                        flattened_processors[key](compiled_params[key])
                        if key in flattened_processors
                        else compiled_params[key]
                    )
                    for key in positiontup
                ]
                core_positional_parameters.append(
                    dialect.execute_sequence_format(l_param)
                )

            self.parameters = core_positional_parameters
        else:
            core_dict_parameters: MutableSequence[Dict[str, Any]] = []
            escaped_names = compiled.escaped_bind_names

            # note that currently, "expanded" parameters will be present
            # in self.compiled_parameters in their quoted form.   This is
            # slightly inconsistent with the approach taken as of
            # #8056 where self.compiled_parameters is meant to contain unquoted
            # param names.
            d_param: Dict[str, Any]
            for compiled_params in self.compiled_parameters:
                if escaped_names:
                    d_param = {
                        escaped_names.get(key, key): (
                            flattened_processors[key](compiled_params[key])
                            if key in flattened_processors
                            else compiled_params[key]
                        )
                        for key in compiled_params
                    }
                else:
                    d_param = {
                        key: (
                            flattened_processors[key](compiled_params[key])
                            if key in flattened_processors
                            else compiled_params[key]
                        )
                        for key in compiled_params
                    }

                core_dict_parameters.append(d_param)

            self.parameters = core_dict_parameters

        return self

    @classmethod
    def _init_statement(
        cls,
        dialect: Dialect,
        connection: Connection,
        dbapi_connection: PoolProxiedConnection,
        execution_options: _ExecuteOptions,
        statement: str,
        parameters: _DBAPIMultiExecuteParams,
    ) -> ExecutionContext:
        """Initialize execution context for a string SQL statement."""

        self = cls.__new__(cls)
        self.root_connection = connection
        self._dbapi_connection = dbapi_connection
        self.dialect = connection.dialect
        self.is_text = True

        self.execution_options = execution_options

        if not parameters:
            if self.dialect.positional:
                self.parameters = [dialect.execute_sequence_format()]
            else:
                self.parameters = [self._empty_dict_params]
        elif isinstance(parameters[0], dialect.execute_sequence_format):
            self.parameters = parameters
        elif isinstance(parameters[0], dict):
            self.parameters = parameters
        else:
            self.parameters = [
                dialect.execute_sequence_format(p) for p in parameters
            ]

        if len(parameters) > 1:
            self.execute_style = ExecuteStyle.EXECUTEMANY

        self.statement = self.unicode_statement = statement

        self.cursor = self.create_cursor()
        return self

    @classmethod
    def _init_default(
        cls,
        dialect: Dialect,
        connection: Connection,
        dbapi_connection: PoolProxiedConnection,
        execution_options: _ExecuteOptions,
    ) -> ExecutionContext:
        """Initialize execution context for a ColumnDefault construct."""

        self = cls.__new__(cls)
        self.root_connection = connection
        self._dbapi_connection = dbapi_connection
        self.dialect = connection.dialect

        self.execution_options = execution_options

        self.cursor = self.create_cursor()
        return self

    def _get_cache_stats(self) -> str:
        if self.compiled is None:
            return "raw sql"

        now = perf_counter()

        ch = self.cache_hit

        gen_time = self.compiled._gen_time
        assert gen_time is not None

        if ch is NO_CACHE_KEY:
            return "no key %.5fs" % (now - gen_time,)
        elif ch is CACHE_HIT:
            return "cached since %.4gs ago" % (now - gen_time,)
        elif ch is CACHE_MISS:
            return "generated in %.5fs" % (now - gen_time,)
        elif ch is CACHING_DISABLED:
            if "_cache_disable_reason" in self.execution_options:
                return "caching disabled (%s) %.5fs " % (
                    self.execution_options["_cache_disable_reason"],
                    now - gen_time,
                )
            else:
                return "caching disabled %.5fs" % (now - gen_time,)
        elif ch is NO_DIALECT_SUPPORT:
            return "dialect %s+%s does not support caching %.5fs" % (
                self.dialect.name,
                self.dialect.driver,
                now - gen_time,
            )
        else:
            return "unknown"

    @property
    def executemany(self):
        return self.execute_style in (
            ExecuteStyle.EXECUTEMANY,
            ExecuteStyle.INSERTMANYVALUES,
        )

    @util.memoized_property
    def identifier_preparer(self):
        if self.compiled:
            return self.compiled.preparer
        elif "schema_translate_map" in self.execution_options:
            return self.dialect.identifier_preparer._with_schema_translate(
                self.execution_options["schema_translate_map"]
            )
        else:
            return self.dialect.identifier_preparer

    @util.memoized_property
    def engine(self):
        return self.root_connection.engine

    @util.memoized_property
    def postfetch_cols(self) -> Optional[Sequence[Column[Any]]]:
        if TYPE_CHECKING:
            assert isinstance(self.compiled, SQLCompiler)
        return self.compiled.postfetch

    @util.memoized_property
    def prefetch_cols(self) -> Optional[Sequence[Column[Any]]]:
        if TYPE_CHECKING:
            assert isinstance(self.compiled, SQLCompiler)
        if self.isinsert:
            return self.compiled.insert_prefetch
        elif self.isupdate:
            return self.compiled.update_prefetch
        else:
            return ()

    @util.memoized_property
    def no_parameters(self):
        return self.execution_options.get("no_parameters", False)

    def _execute_scalar(
        self,
        stmt: str,
        type_: Optional[TypeEngine[Any]],
        parameters: Optional[_DBAPISingleExecuteParams] = None,
    ) -> Any:
        """Execute a string statement on the current cursor, returning a
        scalar result.

        Used to fire off sequences, default phrases, and "select lastrowid"
        types of statements individually or in the context of a parent INSERT
        or UPDATE statement.

        """

        conn = self.root_connection

        if "schema_translate_map" in self.execution_options:
            schema_translate_map = self.execution_options.get(
                "schema_translate_map", {}
            )

            rst = self.identifier_preparer._render_schema_translates
            stmt = rst(stmt, schema_translate_map)

        if not parameters:
            if self.dialect.positional:
                parameters = self.dialect.execute_sequence_format()
            else:
                parameters = {}

        conn._cursor_execute(self.cursor, stmt, parameters, context=self)
        row = self.cursor.fetchone()
        if row is not None:
            r = row[0]
        else:
            r = None
        if type_ is not None:
            # apply type post processors to the result
            proc = type_._cached_result_processor(
                self.dialect, self.cursor.description[0][1]
            )
            if proc:
                return proc(r)
        return r

    @util.memoized_property
    def connection(self):
        return self.root_connection

    def _use_server_side_cursor(self):
        if not self.dialect.supports_server_side_cursors:
            return False

        if self.dialect.server_side_cursors:
            # this is deprecated
            use_server_side = self.execution_options.get(
                "stream_results", True
            ) and (
                self.compiled
                and isinstance(self.compiled.statement, expression.Selectable)
                or (
                    (
                        not self.compiled
                        or isinstance(
                            self.compiled.statement, expression.TextClause
                        )
                    )
                    and self.unicode_statement
                    and SERVER_SIDE_CURSOR_RE.match(self.unicode_statement)
                )
            )
        else:
            use_server_side = self.execution_options.get(
                "stream_results", False
            )

        return use_server_side

    def create_cursor(self) -> DBAPICursor:
        if (
            # inlining initial preference checks for SS cursors
            self.dialect.supports_server_side_cursors
            and (
                self.execution_options.get("stream_results", False)
                or (
                    self.dialect.server_side_cursors
                    and self._use_server_side_cursor()
                )
            )
        ):
            self._is_server_side = True
            return self.create_server_side_cursor()
        else:
            self._is_server_side = False
            return self.create_default_cursor()

    def fetchall_for_returning(self, cursor):
        return cursor.fetchall()

    def create_default_cursor(self) -> DBAPICursor:
        return self._dbapi_connection.cursor()

    def create_server_side_cursor(self) -> DBAPICursor:
        raise NotImplementedError()

    def pre_exec(self):
        pass

    def get_out_parameter_values(self, names):
        raise NotImplementedError(
            "This dialect does not support OUT parameters"
        )

    def post_exec(self):
        pass

    def get_result_processor(self, type_, colname, coltype):
        """Return a 'result processor' for a given type as present in
        cursor.description.

        This has a default implementation that dialects can override
        for context-sensitive result type handling.

        """
        return type_._cached_result_processor(self.dialect, coltype)

    def get_lastrowid(self):
        """return self.cursor.lastrowid, or equivalent, after an INSERT.

        This may involve calling special cursor functions, issuing a new SELECT
        on the cursor (or a new one), or returning a stored value that was
        calculated within post_exec().

        This function will only be called for dialects which support "implicit"
        primary key generation, keep preexecute_autoincrement_sequences set to
        False, and when no explicit id value was bound to the statement.

        The function is called once for an INSERT statement that would need to
        return the last inserted primary key for those dialects that make use
        of the lastrowid concept.  In these cases, it is called directly after
        :meth:`.ExecutionContext.post_exec`.

        """
        return self.cursor.lastrowid

    def handle_dbapi_exception(self, e):
        pass

    @util.non_memoized_property
    def rowcount(self) -> int:
        if self._rowcount is not None:
            return self._rowcount
        else:
            return self.cursor.rowcount

    @property
    def _has_rowcount(self):
        return self._rowcount is not None

    def supports_sane_rowcount(self):
        return self.dialect.supports_sane_rowcount

    def supports_sane_multi_rowcount(self):
        return self.dialect.supports_sane_multi_rowcount

    def _setup_result_proxy(self):
        exec_opt = self.execution_options

        if self._rowcount is None and exec_opt.get("preserve_rowcount", False):
            self._rowcount = self.cursor.rowcount

        if self.is_crud or self.is_text:
            result = self._setup_dml_or_text_result()
            yp = False
        else:
            yp = exec_opt.get("yield_per", None)
            sr = self._is_server_side or exec_opt.get("stream_results", False)
            strategy = self.cursor_fetch_strategy
            if sr and strategy is _cursor._DEFAULT_FETCH:
                strategy = _cursor.BufferedRowCursorFetchStrategy(
                    self.cursor, self.execution_options
                )
            cursor_description: _DBAPICursorDescription = (
                strategy.alternate_cursor_description
                or self.cursor.description
            )
            if cursor_description is None:
                strategy = _cursor._NO_CURSOR_DQL

            result = _cursor.CursorResult(self, strategy, cursor_description)

        compiled = self.compiled

        if (
            compiled
            and not self.isddl
            and cast(SQLCompiler, compiled).has_out_parameters
        ):
            self._setup_out_parameters(result)

        self._soft_closed = result._soft_closed

        if yp:
            result = result.yield_per(yp)

        return result

    def _setup_out_parameters(self, result):
        compiled = cast(SQLCompiler, self.compiled)

        out_bindparams = [
            (param, name)
            for param, name in compiled.bind_names.items()
            if param.isoutparam
        ]
        out_parameters = {}

        for bindparam, raw_value in zip(
            [param for param, name in out_bindparams],
            self.get_out_parameter_values(
                [name for param, name in out_bindparams]
            ),
        ):
            type_ = bindparam.type
            impl_type = type_.dialect_impl(self.dialect)
            dbapi_type = impl_type.get_dbapi_type(self.dialect.loaded_dbapi)
            result_processor = impl_type.result_processor(
                self.dialect, dbapi_type
            )
            if result_processor is not None:
                raw_value = result_processor(raw_value)
            out_parameters[bindparam.key] = raw_value

        result.out_parameters = out_parameters

    def _setup_dml_or_text_result(self):
        compiled = cast(SQLCompiler, self.compiled)

        strategy: ResultFetchStrategy = self.cursor_fetch_strategy

        if self.isinsert:
            if (
                self.execute_style is ExecuteStyle.INSERTMANYVALUES
                and compiled.effective_returning
            ):
                strategy = _cursor.FullyBufferedCursorFetchStrategy(
                    self.cursor,
                    initial_buffer=self._insertmanyvalues_rows,
                    # maintain alt cursor description if set by the
                    # dialect, e.g. mssql preserves it
                    alternate_description=(
                        strategy.alternate_cursor_description
                    ),
                )

            if compiled.postfetch_lastrowid:
                self.inserted_primary_key_rows = (
                    self._setup_ins_pk_from_lastrowid()
                )
            # else if not self._is_implicit_returning,
            # the default inserted_primary_key_rows accessor will
            # return an "empty" primary key collection when accessed.

        if self._is_server_side and strategy is _cursor._DEFAULT_FETCH:
            strategy = _cursor.BufferedRowCursorFetchStrategy(
                self.cursor, self.execution_options
            )

        if strategy is _cursor._NO_CURSOR_DML:
            cursor_description = None
        else:
            cursor_description = (
                strategy.alternate_cursor_description
                or self.cursor.description
            )

        if cursor_description is None:
            strategy = _cursor._NO_CURSOR_DML
        elif self._num_sentinel_cols:
            assert self.execute_style is ExecuteStyle.INSERTMANYVALUES
            # strip out the sentinel columns from cursor description
            # a similar logic is done to the rows only in CursorResult
            cursor_description = cursor_description[
                0 : -self._num_sentinel_cols
            ]

        result: _cursor.CursorResult[Any] = _cursor.CursorResult(
            self, strategy, cursor_description
        )

        if self.isinsert:
            if self._is_implicit_returning:
                rows = result.all()

                self.returned_default_rows = rows

                self.inserted_primary_key_rows = (
                    self._setup_ins_pk_from_implicit_returning(result, rows)
                )

                # test that it has a cursor metadata that is accurate. the
                # first row will have been fetched and current assumptions
                # are that the result has only one row, until executemany()
                # support is added here.
                assert result._metadata.returns_rows

                # Insert statement has both return_defaults() and
                # returning().  rewind the result on the list of rows
                # we just used.
                if self._is_supplemental_returning:
                    result._rewind(rows)
                else:
                    result._soft_close()
            elif not self._is_explicit_returning:
                result._soft_close()

                # we assume here the result does not return any rows.
                # *usually*, this will be true.  However, some dialects
                # such as that of MSSQL/pyodbc need to SELECT a post fetch
                # function so this is not necessarily true.
                # assert not result.returns_rows

        elif self._is_implicit_returning:
            rows = result.all()

            if rows:
                self.returned_default_rows = rows
            self._rowcount = len(rows)

            if self._is_supplemental_returning:
                result._rewind(rows)
            else:
                result._soft_close()

            # test that it has a cursor metadata that is accurate.
            # the rows have all been fetched however.
            assert result._metadata.returns_rows

        elif not result._metadata.returns_rows:
            # no results, get rowcount
            # (which requires open cursor on some drivers)
            if self._rowcount is None:
                self._rowcount = self.cursor.rowcount
            result._soft_close()
        elif self.isupdate or self.isdelete:
            if self._rowcount is None:
                self._rowcount = self.cursor.rowcount
        return result

    @util.memoized_property
    def inserted_primary_key_rows(self):
        # if no specific "get primary key" strategy was set up
        # during execution, return a "default" primary key based
        # on what's in the compiled_parameters and nothing else.
        return self._setup_ins_pk_from_empty()

    def _setup_ins_pk_from_lastrowid(self):
        getter = cast(
            SQLCompiler, self.compiled
        )._inserted_primary_key_from_lastrowid_getter
        lastrowid = self.get_lastrowid()
        return [getter(lastrowid, self.compiled_parameters[0])]

    def _setup_ins_pk_from_empty(self):
        getter = cast(
            SQLCompiler, self.compiled
        )._inserted_primary_key_from_lastrowid_getter
        return [getter(None, param) for param in self.compiled_parameters]

    def _setup_ins_pk_from_implicit_returning(self, result, rows):
        if not rows:
            return []

        getter = cast(
            SQLCompiler, self.compiled
        )._inserted_primary_key_from_returning_getter
        compiled_params = self.compiled_parameters

        return [
            getter(row, param) for row, param in zip(rows, compiled_params)
        ]

    def lastrow_has_defaults(self):
        return (self.isinsert or self.isupdate) and bool(
            cast(SQLCompiler, self.compiled).postfetch
        )

    def _prepare_set_input_sizes(
        self,
    ) -> Optional[List[Tuple[str, Any, TypeEngine[Any]]]]:
        """Given a cursor and ClauseParameters, prepare arguments
        in order to call the appropriate
        style of ``setinputsizes()`` on the cursor, using DB-API types
        from the bind parameter's ``TypeEngine`` objects.

        This method only called by those dialects which set the
        :attr:`.Dialect.bind_typing` attribute to
        :attr:`.BindTyping.SETINPUTSIZES`.  Python-oracledb and cx_Oracle are
        the only DBAPIs that requires setinputsizes(); pyodbc offers it as an
        option.

        Prior to SQLAlchemy 2.0, the setinputsizes() approach was also used
        for pg8000 and asyncpg, which has been changed to inline rendering
        of casts.

        """
        if self.isddl or self.is_text:
            return None

        compiled = cast(SQLCompiler, self.compiled)

        inputsizes = compiled._get_set_input_sizes_lookup()

        if inputsizes is None:
            return None

        dialect = self.dialect

        # all of the rest of this... cython?

        if dialect._has_events:
            inputsizes = dict(inputsizes)
            dialect.dispatch.do_setinputsizes(
                inputsizes, self.cursor, self.statement, self.parameters, self
            )

        if compiled.escaped_bind_names:
            escaped_bind_names = compiled.escaped_bind_names
        else:
            escaped_bind_names = None

        if dialect.positional:
            items = [
                (key, compiled.binds[key])
                for key in compiled.positiontup or ()
            ]
        else:
            items = [
                (key, bindparam)
                for bindparam, key in compiled.bind_names.items()
            ]

        generic_inputsizes: List[Tuple[str, Any, TypeEngine[Any]]] = []
        for key, bindparam in items:
            if bindparam in compiled.literal_execute_params:
                continue

            if key in self._expanded_parameters:
                if is_tuple_type(bindparam.type):
                    num = len(bindparam.type.types)
                    dbtypes = inputsizes[bindparam]
                    generic_inputsizes.extend(
                        (
                            (
                                escaped_bind_names.get(paramname, paramname)
                                if escaped_bind_names is not None
                                else paramname
                            ),
                            dbtypes[idx % num],
                            bindparam.type.types[idx % num],
                        )
                        for idx, paramname in enumerate(
                            self._expanded_parameters[key]
                        )
                    )
                else:
                    dbtype = inputsizes.get(bindparam, None)
                    generic_inputsizes.extend(
                        (
                            (
                                escaped_bind_names.get(paramname, paramname)
                                if escaped_bind_names is not None
                                else paramname
                            ),
                            dbtype,
                            bindparam.type,
                        )
                        for paramname in self._expanded_parameters[key]
                    )
            else:
                dbtype = inputsizes.get(bindparam, None)

                escaped_name = (
                    escaped_bind_names.get(key, key)
                    if escaped_bind_names is not None
                    else key
                )

                generic_inputsizes.append(
                    (escaped_name, dbtype, bindparam.type)
                )

        return generic_inputsizes

    def _exec_default(self, column, default, type_):
        if default.is_sequence:
            return self.fire_sequence(default, type_)
        elif default.is_callable:
            # this codepath is not normally used as it's inlined
            # into _process_execute_defaults
            self.current_column = column
            return default.arg(self)
        elif default.is_clause_element:
            return self._exec_default_clause_element(column, default, type_)
        else:
            # this codepath is not normally used as it's inlined
            # into _process_execute_defaults
            return default.arg

    def _exec_default_clause_element(self, column, default, type_):
        # execute a default that's a complete clause element.  Here, we have
        # to re-implement a miniature version of the compile->parameters->
        # cursor.execute() sequence, since we don't want to modify the state
        # of the connection  / result in progress or create new connection/
        # result objects etc.
        # .. versionchanged:: 1.4

        if not default._arg_is_typed:
            default_arg = expression.type_coerce(default.arg, type_)
        else:
            default_arg = default.arg
        compiled = expression.select(default_arg).compile(dialect=self.dialect)
        compiled_params = compiled.construct_params()
        processors = compiled._bind_processors
        if compiled.positional:
            parameters = self.dialect.execute_sequence_format(
                [
                    (
                        processors[key](compiled_params[key])  # type: ignore
                        if key in processors
                        else compiled_params[key]
                    )
                    for key in compiled.positiontup or ()
                ]
            )
        else:
            parameters = {
                key: (
                    processors[key](compiled_params[key])  # type: ignore
                    if key in processors
                    else compiled_params[key]
                )
                for key in compiled_params
            }
        return self._execute_scalar(
            str(compiled), type_, parameters=parameters
        )

    current_parameters: Optional[_CoreSingleExecuteParams] = None
    """A dictionary of parameters applied to the current row.

    This attribute is only available in the context of a user-defined default
    generation function, e.g. as described at :ref:`context_default_functions`.
    It consists of a dictionary which includes entries for each column/value
    pair that is to be part of the INSERT or UPDATE statement. The keys of the
    dictionary will be the key value of each :class:`_schema.Column`,
    which is usually
    synonymous with the name.

    Note that the :attr:`.DefaultExecutionContext.current_parameters` attribute
    does not accommodate for the "multi-values" feature of the
    :meth:`_expression.Insert.values` method.  The
    :meth:`.DefaultExecutionContext.get_current_parameters` method should be
    preferred.

    .. seealso::

        :meth:`.DefaultExecutionContext.get_current_parameters`

        :ref:`context_default_functions`

    """

    def get_current_parameters(self, isolate_multiinsert_groups=True):
        """Return a dictionary of parameters applied to the current row.

        This method can only be used in the context of a user-defined default
        generation function, e.g. as described at
        :ref:`context_default_functions`. When invoked, a dictionary is
        returned which includes entries for each column/value pair that is part
        of the INSERT or UPDATE statement. The keys of the dictionary will be
        the key value of each :class:`_schema.Column`,
        which is usually synonymous
        with the name.

        :param isolate_multiinsert_groups=True: indicates that multi-valued
         INSERT constructs created using :meth:`_expression.Insert.values`
         should be
         handled by returning only the subset of parameters that are local
         to the current column default invocation.   When ``False``, the
         raw parameters of the statement are returned including the
         naming convention used in the case of multi-valued INSERT.

        .. versionadded:: 1.2  added
           :meth:`.DefaultExecutionContext.get_current_parameters`
           which provides more functionality over the existing
           :attr:`.DefaultExecutionContext.current_parameters`
           attribute.

        .. seealso::

            :attr:`.DefaultExecutionContext.current_parameters`

            :ref:`context_default_functions`

        """
        try:
            parameters = self.current_parameters
            column = self.current_column
        except AttributeError:
            raise exc.InvalidRequestError(
                "get_current_parameters() can only be invoked in the "
                "context of a Python side column default function"
            )
        else:
            assert column is not None
            assert parameters is not None
        compile_state = cast(
            "DMLState", cast(SQLCompiler, self.compiled).compile_state
        )
        assert compile_state is not None
        if (
            isolate_multiinsert_groups
            and dml.isinsert(compile_state)
            and compile_state._has_multi_parameters
        ):
            if column._is_multiparam_column:
                index = column.index + 1
                d = {column.original.key: parameters[column.key]}
            else:
                d = {column.key: parameters[column.key]}
                index = 0
            assert compile_state._dict_parameters is not None
            keys = compile_state._dict_parameters.keys()
            d.update(
                (key, parameters["%s_m%d" % (key, index)]) for key in keys
            )
            return d
        else:
            return parameters

    def get_insert_default(self, column):
        if column.default is None:
            return None
        else:
            return self._exec_default(column, column.default, column.type)

    def get_update_default(self, column):
        if column.onupdate is None:
            return None
        else:
            return self._exec_default(column, column.onupdate, column.type)

    def _process_execute_defaults(self):
        compiled = cast(SQLCompiler, self.compiled)

        key_getter = compiled._within_exec_param_key_getter

        sentinel_counter = 0

        if compiled.insert_prefetch:
            prefetch_recs = [
                (
                    c,
                    key_getter(c),
                    c._default_description_tuple,
                    self.get_insert_default,
                )
                for c in compiled.insert_prefetch
            ]
        elif compiled.update_prefetch:
            prefetch_recs = [
                (
                    c,
                    key_getter(c),
                    c._onupdate_description_tuple,
                    self.get_update_default,
                )
                for c in compiled.update_prefetch
            ]
        else:
            prefetch_recs = []

        for param in self.compiled_parameters:
            self.current_parameters = param

            for (
                c,
                param_key,
                (arg, is_scalar, is_callable, is_sentinel),
                fallback,
            ) in prefetch_recs:
                if is_sentinel:
                    param[param_key] = sentinel_counter
                    sentinel_counter += 1
                elif is_scalar:
                    param[param_key] = arg
                elif is_callable:
                    self.current_column = c
                    param[param_key] = arg(self)
                else:
                    val = fallback(c)
                    if val is not None:
                        param[param_key] = val

        del self.current_parameters


DefaultDialect.execution_ctx_cls = DefaultExecutionContext

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\integrals\laplace.py ===
"""Laplace Transforms"""
import sys
import sympy
from sympy.core import S, pi, I
from sympy.core.add import Add
from sympy.core.cache import cacheit
from sympy.core.expr import Expr
from sympy.core.function import (
    AppliedUndef, Derivative, expand, expand_complex, expand_mul, expand_trig,
    Lambda, WildFunction, diff, Subs)
from sympy.core.mul import Mul, prod
from sympy.core.relational import (
    _canonical, Ge, Gt, Lt, Unequality, Eq, Ne, Relational)
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.functions.elementary.complexes import (
    re, im, arg, Abs, polar_lift, periodic_argument)
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, coth, sinh, asinh
from sympy.functions.elementary.miscellaneous import Max, Min, sqrt
from sympy.functions.elementary.piecewise import (
    Piecewise, piecewise_exclusive)
from sympy.functions.elementary.trigonometric import cos, sin, atan, sinc
from sympy.functions.special.bessel import besseli, besselj, besselk, bessely
from sympy.functions.special.delta_functions import DiracDelta, Heaviside
from sympy.functions.special.error_functions import erf, erfc, Ei
from sympy.functions.special.gamma_functions import (
    digamma, gamma, lowergamma, uppergamma)
from sympy.functions.special.singularity_functions import SingularityFunction
from sympy.integrals import integrate, Integral
from sympy.integrals.transforms import (
    _simplify, IntegralTransform, IntegralTransformError)
from sympy.logic.boolalg import to_cnf, conjuncts, disjuncts, Or, And
from sympy.matrices.matrixbase import MatrixBase
from sympy.polys.matrices.linsolve import _lin_eq2dict
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyroots import roots
from sympy.polys.polytools import Poly
from sympy.polys.rationaltools import together
from sympy.polys.rootoftools import RootSum
from sympy.utilities.exceptions import (
    sympy_deprecation_warning, SymPyDeprecationWarning, ignore_warnings)
from sympy.utilities.misc import debugf

_LT_level = 0


def DEBUG_WRAP(func):
    def wrap(*args, **kwargs):
        from sympy import SYMPY_DEBUG
        global _LT_level

        if not SYMPY_DEBUG:
            return func(*args, **kwargs)

        if _LT_level == 0:
            print('\n' + '-'*78, file=sys.stderr)
        print('-LT- %s%s%s' % ('  '*_LT_level, func.__name__, args),
              file=sys.stderr)
        _LT_level += 1
        if (
                func.__name__ == '_laplace_transform_integration' or
                func.__name__ == '_inverse_laplace_transform_integration'):
            sympy.SYMPY_DEBUG = False
            print('**** %sIntegrating ...' % ('  '*_LT_level), file=sys.stderr)
            result = func(*args, **kwargs)
            sympy.SYMPY_DEBUG = True
        else:
            result = func(*args, **kwargs)
        _LT_level -= 1
        print('-LT- %s---> %s' % ('  '*_LT_level, result), file=sys.stderr)
        if _LT_level == 0:
            print('-'*78 + '\n', file=sys.stderr)
        return result
    return wrap


def _debug(text):
    from sympy import SYMPY_DEBUG

    if SYMPY_DEBUG:
        print('-LT- %s%s' % ('  '*_LT_level, text), file=sys.stderr)


def _simplifyconds(expr, s, a):
    r"""
    Naively simplify some conditions occurring in ``expr``,
    given that `\operatorname{Re}(s) > a`.

    Examples
    ========

    >>> from sympy.integrals.laplace import _simplifyconds
    >>> from sympy.abc import x
    >>> from sympy import sympify as S
    >>> _simplifyconds(abs(x**2) < 1, x, 1)
    False
    >>> _simplifyconds(abs(x**2) < 1, x, 2)
    False
    >>> _simplifyconds(abs(x**2) < 1, x, 0)
    Abs(x**2) < 1
    >>> _simplifyconds(abs(1/x**2) < 1, x, 1)
    True
    >>> _simplifyconds(S(1) < abs(x), x, 1)
    True
    >>> _simplifyconds(S(1) < abs(1/x), x, 1)
    False

    >>> from sympy import Ne
    >>> _simplifyconds(Ne(1, x**3), x, 1)
    True
    >>> _simplifyconds(Ne(1, x**3), x, 2)
    True
    >>> _simplifyconds(Ne(1, x**3), x, 0)
    Ne(1, x**3)
    """

    def power(ex):
        if ex == s:
            return 1
        if ex.is_Pow and ex.base == s:
            return ex.exp
        return None

    def bigger(ex1, ex2):
        """ Return True only if |ex1| > |ex2|, False only if |ex1| < |ex2|.
            Else return None. """
        if ex1.has(s) and ex2.has(s):
            return None
        if isinstance(ex1, Abs):
            ex1 = ex1.args[0]
        if isinstance(ex2, Abs):
            ex2 = ex2.args[0]
        if ex1.has(s):
            return bigger(1/ex2, 1/ex1)
        n = power(ex2)
        if n is None:
            return None
        try:
            if n > 0 and (Abs(ex1) <= Abs(a)**n) == S.true:
                return False
            if n < 0 and (Abs(ex1) >= Abs(a)**n) == S.true:
                return True
        except TypeError:
            return None

    def replie(x, y):
        """ simplify x < y """
        if (not (x.is_positive or isinstance(x, Abs))
                or not (y.is_positive or isinstance(y, Abs))):
            return (x < y)
        r = bigger(x, y)
        if r is not None:
            return not r
        return (x < y)

    def replue(x, y):
        b = bigger(x, y)
        if b in (True, False):
            return True
        return Unequality(x, y)

    def repl(ex, *args):
        if ex in (True, False):
            return bool(ex)
        return ex.replace(*args)

    from sympy.simplify.radsimp import collect_abs
    expr = collect_abs(expr)
    expr = repl(expr, Lt, replie)
    expr = repl(expr, Gt, lambda x, y: replie(y, x))
    expr = repl(expr, Unequality, replue)
    return S(expr)


@DEBUG_WRAP
def expand_dirac_delta(expr):
    """
    Expand an expression involving DiractDelta to get it as a linear
    combination of DiracDelta functions.
    """
    return _lin_eq2dict(expr, expr.atoms(DiracDelta))


@DEBUG_WRAP
def _laplace_transform_integration(f, t, s_, *, simplify):
    """ The backend function for doing Laplace transforms by integration.

    This backend assumes that the frontend has already split sums
    such that `f` is to an addition anymore.
    """
    s = Dummy('s')

    if f.has(DiracDelta):
        return None

    F = integrate(f*exp(-s*t), (t, S.Zero, S.Infinity))

    if not F.has(Integral):
        return _simplify(F.subs(s, s_), simplify), S.NegativeInfinity, S.true

    if not F.is_Piecewise:
        return None

    F, cond = F.args[0]
    if F.has(Integral):
        return None

    def process_conds(conds):
        """ Turn ``conds`` into a strip and auxiliary conditions. """
        from sympy.solvers.inequalities import _solve_inequality
        a = S.NegativeInfinity
        aux = S.true
        conds = conjuncts(to_cnf(conds))
        p, q, w1, w2, w3, w4, w5 = symbols(
            'p q w1 w2 w3 w4 w5', cls=Wild, exclude=[s])
        patterns = (
            p*Abs(arg((s + w3)*q)) < w2,
            p*Abs(arg((s + w3)*q)) <= w2,
            Abs(periodic_argument((s + w3)**p*q, w1)) < w2,
            Abs(periodic_argument((s + w3)**p*q, w1)) <= w2,
            Abs(periodic_argument((polar_lift(s + w3))**p*q, w1)) < w2,
            Abs(periodic_argument((polar_lift(s + w3))**p*q, w1)) <= w2)
        for c in conds:
            a_ = S.Infinity
            aux_ = []
            for d in disjuncts(c):
                if d.is_Relational and s in d.rhs.free_symbols:
                    d = d.reversed
                if d.is_Relational and isinstance(d, (Ge, Gt)):
                    d = d.reversedsign
                for pat in patterns:
                    m = d.match(pat)
                    if m:
                        break
                if m and m[q].is_positive and m[w2]/m[p] == pi/2:
                    d = -re(s + m[w3]) < 0
                m = d.match(p - cos(w1*Abs(arg(s*w5))*w2)*Abs(s**w3)**w4 < 0)
                if not m:
                    m = d.match(
                        cos(p - Abs(periodic_argument(s**w1*w5, q))*w2) *
                        Abs(s**w3)**w4 < 0)
                if not m:
                    m = d.match(
                        p - cos(
                            Abs(periodic_argument(polar_lift(s)**w1*w5, q))*w2
                            )*Abs(s**w3)**w4 < 0)
                if m and all(m[wild].is_positive for wild in [
                        w1, w2, w3, w4, w5]):
                    d = re(s) > m[p]
                d_ = d.replace(
                    re, lambda x: x.expand().as_real_imag()[0]).subs(re(s), t)
                if (
                        not d.is_Relational or d.rel_op in ('==', '!=')
                        or d_.has(s) or not d_.has(t)):
                    aux_ += [d]
                    continue
                soln = _solve_inequality(d_, t)
                if not soln.is_Relational or soln.rel_op in ('==', '!='):
                    aux_ += [d]
                    continue
                if soln.lts == t:
                    return None
                else:
                    a_ = Min(soln.lts, a_)
            if a_ is not S.Infinity:
                a = Max(a_, a)
            else:
                aux = And(aux, Or(*aux_))
        return a, aux.canonical if aux.is_Relational else aux

    conds = [process_conds(c) for c in disjuncts(cond)]
    conds2 = [x for x in conds if x[1] !=
              S.false and x[0] is not S.NegativeInfinity]
    if not conds2:
        conds2 = [x for x in conds if x[1] != S.false]
    conds = list(ordered(conds2))

    def cnt(expr):
        if expr in (True, False):
            return 0
        return expr.count_ops()
    conds.sort(key=lambda x: (-x[0], cnt(x[1])))

    if not conds:
        return None
    a, aux = conds[0]  # XXX is [0] always the right one?

    def sbs(expr):
        return expr.subs(s, s_)
    if simplify:
        F = _simplifyconds(F, s, a)
        aux = _simplifyconds(aux, s, a)
    return _simplify(F.subs(s, s_), simplify), sbs(a), _canonical(sbs(aux))


@DEBUG_WRAP
def _laplace_deep_collect(f, t):
    """
    This is an internal helper function that traverses through the expression
    tree of `f(t)` and collects arguments. The purpose of it is that
    anything like `f(w*t-1*t-c)` will be written as `f((w-1)*t-c)` such that
    it can match `f(a*t+b)`.
    """
    if not isinstance(f, Expr):
        return f
    if (p := f.as_poly(t)) is not None:
        return p.as_expr()
    func = f.func
    args = [_laplace_deep_collect(arg, t) for arg in f.args]
    return func(*args)


@cacheit
def _laplace_build_rules():
    """
    This is an internal helper function that returns the table of Laplace
    transform rules in terms of the time variable `t` and the frequency
    variable `s`.  It is used by ``_laplace_apply_rules``.  Each entry is a
    tuple containing:

        (time domain pattern,
         frequency-domain replacement,
         condition for the rule to be applied,
         convergence plane,
         preparation function)

    The preparation function is a function with one argument that is applied
    to the expression before matching. For most rules it should be
    ``_laplace_deep_collect``.
    """
    t = Dummy('t')
    s = Dummy('s')
    a = Wild('a', exclude=[t])
    b = Wild('b', exclude=[t])
    n = Wild('n', exclude=[t])
    tau = Wild('tau', exclude=[t])
    omega = Wild('omega', exclude=[t])
    def dco(f): return _laplace_deep_collect(f, t)
    _debug('_laplace_build_rules is building rules')

    laplace_transform_rules = [
        (a, a/s,
         S.true, S.Zero, dco),  # 4.2.1
        (DiracDelta(a*t-b), exp(-s*b/a)/Abs(a),
         Or(And(a > 0, b >= 0), And(a < 0, b <= 0)),
         S.NegativeInfinity, dco),  # Not in Bateman54
        (DiracDelta(a*t-b), S(0),
         Or(And(a < 0, b >= 0), And(a > 0, b <= 0)),
         S.NegativeInfinity, dco),  # Not in Bateman54
        (Heaviside(a*t-b), exp(-s*b/a)/s,
         And(a > 0, b > 0), S.Zero, dco),  # 4.4.1
        (Heaviside(a*t-b), (1-exp(-s*b/a))/s,
         And(a < 0, b < 0), S.Zero, dco),  # 4.4.1
        (Heaviside(a*t-b), 1/s,
         And(a > 0, b <= 0), S.Zero, dco),  # 4.4.1
        (Heaviside(a*t-b), 0,
         And(a < 0, b > 0), S.Zero, dco),  # 4.4.1
        (t, 1/s**2,
         S.true, S.Zero, dco),  # 4.2.3
        (1/(a*t+b), -exp(-b/a*s)*Ei(-b/a*s)/a,
         Abs(arg(b/a)) < pi, S.Zero, dco),  # 4.2.6
        (1/sqrt(a*t+b), sqrt(a*pi/s)*exp(b/a*s)*erfc(sqrt(b/a*s))/a,
         Abs(arg(b/a)) < pi, S.Zero, dco),  # 4.2.18
        ((a*t+b)**(-S(3)/2),
         2*b**(-S(1)/2)-2*(pi*s/a)**(S(1)/2)*exp(b/a*s) * erfc(sqrt(b/a*s))/a,
         Abs(arg(b/a)) < pi, S.Zero, dco),  # 4.2.20
        (sqrt(t)/(t+b), sqrt(pi/s)-pi*sqrt(b)*exp(b*s)*erfc(sqrt(b*s)),
         Abs(arg(b)) < pi, S.Zero, dco),  # 4.2.22
        (1/(a*sqrt(t) + t**(3/2)), pi*a**(S(1)/2)*exp(a*s)*erfc(sqrt(a*s)),
         S.true, S.Zero, dco),  # Not in Bateman54
        (t**n, gamma(n+1)/s**(n+1),
         n > -1, S.Zero, dco),  # 4.3.1
        ((a*t+b)**n, uppergamma(n+1, b/a*s)*exp(-b/a*s)/s**(n+1)/a,
         And(n > -1, Abs(arg(b/a)) < pi), S.Zero, dco),  # 4.3.4
        (t**n/(t+a), a**n*gamma(n+1)*uppergamma(-n, a*s),
         And(n > -1, Abs(arg(a)) < pi), S.Zero, dco),  # 4.3.7
        (exp(a*t-tau), exp(-tau)/(s-a),
         S.true, re(a), dco),  # 4.5.1
        (t*exp(a*t-tau), exp(-tau)/(s-a)**2,
         S.true, re(a), dco),  # 4.5.2
        (t**n*exp(a*t), gamma(n+1)/(s-a)**(n+1),
         re(n) > -1, re(a), dco),  # 4.5.3
        (exp(-a*t**2), sqrt(pi/4/a)*exp(s**2/4/a)*erfc(s/sqrt(4*a)),
         re(a) > 0, S.Zero, dco),  # 4.5.21
        (t*exp(-a*t**2),
         1/(2*a)-2/sqrt(pi)/(4*a)**(S(3)/2)*s*erfc(s/sqrt(4*a)),
         re(a) > 0, S.Zero, dco),  # 4.5.22
        (exp(-a/t), 2*sqrt(a/s)*besselk(1, 2*sqrt(a*s)),
         re(a) >= 0, S.Zero, dco),  # 4.5.25
        (sqrt(t)*exp(-a/t),
         S(1)/2*sqrt(pi/s**3)*(1+2*sqrt(a*s))*exp(-2*sqrt(a*s)),
         re(a) >= 0, S.Zero, dco),  # 4.5.26
        (exp(-a/t)/sqrt(t), sqrt(pi/s)*exp(-2*sqrt(a*s)),
         re(a) >= 0, S.Zero, dco),  # 4.5.27
        (exp(-a/t)/(t*sqrt(t)), sqrt(pi/a)*exp(-2*sqrt(a*s)),
         re(a) > 0, S.Zero, dco),  # 4.5.28
        (t**n*exp(-a/t), 2*(a/s)**((n+1)/2)*besselk(n+1, 2*sqrt(a*s)),
         re(a) > 0, S.Zero, dco),  # 4.5.29
        # TODO: rules with sqrt(a*t) and sqrt(a/t) have stopped working after
        #       changes to as_base_exp
        # (exp(-2*sqrt(a*t)),
        #  s**(-1)-sqrt(pi*a)*s**(-S(3)/2)*exp(a/s) * erfc(sqrt(a/s)),
        #  Abs(arg(a)) < pi, S.Zero, dco),  # 4.5.31
        # (exp(-2*sqrt(a*t))/sqrt(t), (pi/s)**(S(1)/2)*exp(a/s)*erfc(sqrt(a/s)),
        #  Abs(arg(a)) < pi, S.Zero, dco),  # 4.5.33
        (exp(-a*exp(-t)), a**(-s)*lowergamma(s, a),
         S.true, S.Zero, dco),  # 4.5.36
        (exp(-a*exp(t)), a**s*uppergamma(-s, a),
         re(a) > 0, S.Zero, dco),  # 4.5.37
        (log(a*t), -log(exp(S.EulerGamma)*s/a)/s,
         a > 0, S.Zero, dco),  # 4.6.1
        (log(1+a*t), -exp(s/a)/s*Ei(-s/a),
         Abs(arg(a)) < pi, S.Zero, dco),  # 4.6.4
        (log(a*t+b), (log(b)-exp(s/b/a)/s*a*Ei(-s/b))/s*a,
         And(a > 0, Abs(arg(b)) < pi), S.Zero, dco),  # 4.6.5
        (log(t)/sqrt(t), -sqrt(pi/s)*log(4*s*exp(S.EulerGamma)),
         S.true, S.Zero, dco),  # 4.6.9
        (t**n*log(t), gamma(n+1)*s**(-n-1)*(digamma(n+1)-log(s)),
         re(n) > -1, S.Zero, dco),  # 4.6.11
        (log(a*t)**2, (log(exp(S.EulerGamma)*s/a)**2+pi**2/6)/s,
         a > 0, S.Zero, dco),  # 4.6.13
        (sin(omega*t), omega/(s**2+omega**2),
         S.true, Abs(im(omega)), dco),  # 4,7,1
        (Abs(sin(omega*t)), omega/(s**2+omega**2)*coth(pi*s/2/omega),
         omega > 0, S.Zero, dco),  # 4.7.2
        (sin(omega*t)/t, atan(omega/s),
         S.true, Abs(im(omega)), dco),  # 4.7.16
        (sin(omega*t)**2/t, log(1+4*omega**2/s**2)/4,
         S.true, 2*Abs(im(omega)), dco),  # 4.7.17
        (sin(omega*t)**2/t**2,
         omega*atan(2*omega/s)-s*log(1+4*omega**2/s**2)/4,
         S.true, 2*Abs(im(omega)), dco),  # 4.7.20
        # (sin(2*sqrt(a*t)), sqrt(pi*a)/s/sqrt(s)*exp(-a/s),
        #  S.true, S.Zero, dco),  # 4.7.32
        # (sin(2*sqrt(a*t))/t, pi*erf(sqrt(a/s)),
        #  S.true, S.Zero, dco),  # 4.7.34
        (cos(omega*t), s/(s**2+omega**2),
         S.true, Abs(im(omega)), dco),  # 4.7.43
        (cos(omega*t)**2, (s**2+2*omega**2)/(s**2+4*omega**2)/s,
         S.true, 2*Abs(im(omega)), dco),  # 4.7.45
        # (sqrt(t)*cos(2*sqrt(a*t)), sqrt(pi)/2*s**(-S(5)/2)*(s-2*a)*exp(-a/s),
        #  S.true, S.Zero, dco),  # 4.7.66
        # (cos(2*sqrt(a*t))/sqrt(t), sqrt(pi/s)*exp(-a/s),
        #  S.true, S.Zero, dco),  # 4.7.67
        (sin(a*t)*sin(b*t), 2*a*b*s/(s**2+(a+b)**2)/(s**2+(a-b)**2),
         S.true, Abs(im(a))+Abs(im(b)), dco),  # 4.7.78
        (cos(a*t)*sin(b*t), b*(s**2-a**2+b**2)/(s**2+(a+b)**2)/(s**2+(a-b)**2),
         S.true, Abs(im(a))+Abs(im(b)), dco),  # 4.7.79
        (cos(a*t)*cos(b*t), s*(s**2+a**2+b**2)/(s**2+(a+b)**2)/(s**2+(a-b)**2),
         S.true, Abs(im(a))+Abs(im(b)), dco),  # 4.7.80
        (sinh(a*t), a/(s**2-a**2),
         S.true, Abs(re(a)), dco),  # 4.9.1
        (cosh(a*t), s/(s**2-a**2),
         S.true, Abs(re(a)), dco),  # 4.9.2
        (sinh(a*t)**2, 2*a**2/(s**3-4*a**2*s),
         S.true, 2*Abs(re(a)), dco),  # 4.9.3
        (cosh(a*t)**2, (s**2-2*a**2)/(s**3-4*a**2*s),
         S.true, 2*Abs(re(a)), dco),  # 4.9.4
        (sinh(a*t)/t, log((s+a)/(s-a))/2,
         S.true, Abs(re(a)), dco),  # 4.9.12
        (t**n*sinh(a*t), gamma(n+1)/2*((s-a)**(-n-1)-(s+a)**(-n-1)),
         n > -2, Abs(a), dco),  # 4.9.18
        (t**n*cosh(a*t), gamma(n+1)/2*((s-a)**(-n-1)+(s+a)**(-n-1)),
         n > -1, Abs(a), dco),  # 4.9.19
        # TODO
        # (sinh(2*sqrt(a*t)), sqrt(pi*a)/s/sqrt(s)*exp(a/s),
        #  S.true, S.Zero, dco),  # 4.9.34
        # (cosh(2*sqrt(a*t)), 1/s+sqrt(pi*a)/s/sqrt(s)*exp(a/s)*erf(sqrt(a/s)),
        #  S.true, S.Zero, dco),  # 4.9.35
        # (
        #     sqrt(t)*sinh(2*sqrt(a*t)),
        #     pi**(S(1)/2)*s**(-S(5)/2)*(s/2+a) *
        #     exp(a/s)*erf(sqrt(a/s))-a**(S(1)/2)*s**(-2),
        #     S.true, S.Zero, dco),  # 4.9.36
        # (sqrt(t)*cosh(2*sqrt(a*t)), pi**(S(1)/2)*s**(-S(5)/2)*(s/2+a)*exp(a/s),
        #   S.true, S.Zero, dco),  # 4.9.37
        # (sinh(2*sqrt(a*t))/sqrt(t),
        #  pi**(S(1)/2)*s**(-S(1)/2)*exp(a/s) * erf(sqrt(a/s)),
        #     S.true, S.Zero, dco),  # 4.9.38
        # (cosh(2*sqrt(a*t))/sqrt(t), pi**(S(1)/2)*s**(-S(1)/2)*exp(a/s),
        #  S.true, S.Zero, dco),  # 4.9.39
        # (sinh(sqrt(a*t))**2/sqrt(t), pi**(S(1)/2)/2*s**(-S(1)/2)*(exp(a/s)-1),
        #  S.true, S.Zero, dco),  # 4.9.40
        # (cosh(sqrt(a*t))**2/sqrt(t), pi**(S(1)/2)/2*s**(-S(1)/2)*(exp(a/s)+1),
        #  S.true, S.Zero, dco),  # 4.9.41
        (erf(a*t), exp(s**2/(2*a)**2)*erfc(s/(2*a))/s,
         4*Abs(arg(a)) < pi, S.Zero, dco),  # 4.12.2
        # (erf(sqrt(a*t)), sqrt(a)/sqrt(s+a)/s,
        #  S.true, Max(S.Zero, -re(a)), dco),  # 4.12.4
        # (exp(a*t)*erf(sqrt(a*t)), sqrt(a)/sqrt(s)/(s-a),
        #  S.true, Max(S.Zero, re(a)), dco),  # 4.12.5
        # (erf(sqrt(a/t)/2), (1-exp(-sqrt(a*s)))/s,
        #  re(a) > 0, S.Zero, dco),  # 4.12.6
        # (erfc(sqrt(a*t)), (sqrt(s+a)-sqrt(a))/sqrt(s+a)/s,
        #  S.true, -re(a), dco),  # 4.12.9
        # (exp(a*t)*erfc(sqrt(a*t)), 1/(s+sqrt(a*s)),
        #  S.true, S.Zero, dco),  # 4.12.10
        # (erfc(sqrt(a/t)/2), exp(-sqrt(a*s))/s,
        #  re(a) > 0, S.Zero, dco),  # 4.2.11
        (besselj(n, a*t), a**n/(sqrt(s**2+a**2)*(s+sqrt(s**2+a**2))**n),
         re(n) > -1, Abs(im(a)), dco),  # 4.14.1
        (t**b*besselj(n, a*t),
         2**n/sqrt(pi)*gamma(n+S.Half)*a**n*(s**2+a**2)**(-n-S.Half),
         And(re(n) > -S.Half, Eq(b, n)), Abs(im(a)), dco),  # 4.14.7
        (t**b*besselj(n, a*t),
         2**(n+1)/sqrt(pi)*gamma(n+S(3)/2)*a**n*s*(s**2+a**2)**(-n-S(3)/2),
         And(re(n) > -1, Eq(b, n+1)), Abs(im(a)), dco),  # 4.14.8
        # (besselj(0, 2*sqrt(a*t)), exp(-a/s)/s,
        #  S.true, S.Zero, dco),  # 4.14.25
        # (t**(b)*besselj(n, 2*sqrt(a*t)), a**(n/2)*s**(-n-1)*exp(-a/s),
        #  And(re(n) > -1, Eq(b, n*S.Half)), S.Zero, dco),  # 4.14.30
        (besselj(0, a*sqrt(t**2+b*t)),
         exp(b*s-b*sqrt(s**2+a**2))/sqrt(s**2+a**2),
         Abs(arg(b)) < pi, Abs(im(a)), dco),  # 4.15.19
        (besseli(n, a*t), a**n/(sqrt(s**2-a**2)*(s+sqrt(s**2-a**2))**n),
         re(n) > -1, Abs(re(a)), dco),  # 4.16.1
        (t**b*besseli(n, a*t),
         2**n/sqrt(pi)*gamma(n+S.Half)*a**n*(s**2-a**2)**(-n-S.Half),
         And(re(n) > -S.Half, Eq(b, n)), Abs(re(a)), dco),  # 4.16.6
        (t**b*besseli(n, a*t),
         2**(n+1)/sqrt(pi)*gamma(n+S(3)/2)*a**n*s*(s**2-a**2)**(-n-S(3)/2),
         And(re(n) > -1, Eq(b, n+1)), Abs(re(a)), dco),  # 4.16.7
        # (t**(b)*besseli(n, 2*sqrt(a*t)), a**(n/2)*s**(-n-1)*exp(a/s),
        #  And(re(n) > -1, Eq(b, n*S.Half)), S.Zero, dco),  # 4.16.18
        (bessely(0, a*t), -2/pi*asinh(s/a)/sqrt(s**2+a**2),
         S.true, Abs(im(a)), dco),  # 4.15.44
        (besselk(0, a*t), log((s + sqrt(s**2-a**2))/a)/(sqrt(s**2-a**2)),
         S.true, -re(a), dco)  # 4.16.23
    ]
    return laplace_transform_rules, t, s


@DEBUG_WRAP
def _laplace_rule_timescale(f, t, s):
    """
    This function applies the time-scaling rule of the Laplace transform in
    a straight-forward way. For example, if it gets ``(f(a*t), t, s)``, it will
    compute ``LaplaceTransform(f(t)/a, t, s/a)`` if ``a>0``.
    """

    a = Wild('a', exclude=[t])
    g = WildFunction('g', nargs=1)
    ma1 = f.match(g)
    if ma1:
        arg = ma1[g].args[0].collect(t)
        ma2 = arg.match(a*t)
        if ma2 and ma2[a].is_positive and ma2[a] != 1:
            _debug('     rule: time scaling (4.1.4)')
            r, pr, cr = _laplace_transform(
                1/ma2[a]*ma1[g].func(t), t, s/ma2[a], simplify=False)
            return (r, pr, cr)
    return None


@DEBUG_WRAP
def _laplace_rule_heaviside(f, t, s):
    """
    This function deals with time-shifted Heaviside step functions. If the time
    shift is positive, it applies the time-shift rule of the Laplace transform.
    For example, if it gets ``(Heaviside(t-a)*f(t), t, s)``, it will compute
    ``exp(-a*s)*LaplaceTransform(f(t+a), t, s)``.

    If the time shift is negative, the Heaviside function is simply removed
    as it means nothing to the Laplace transform.

    The function does not remove a factor ``Heaviside(t)``; this is done by
    the simple rules.
    """

    a = Wild('a', exclude=[t])
    y = Wild('y')
    g = Wild('g')
    if ma1 := f.match(Heaviside(y) * g):
        if ma2 := ma1[y].match(t - a):
            if ma2[a].is_positive:
                _debug('     rule: time shift (4.1.4)')
                r, pr, cr = _laplace_transform(
                    ma1[g].subs(t, t + ma2[a]), t, s, simplify=False)
                return (exp(-ma2[a] * s) * r, pr, cr)
            if ma2[a].is_negative:
                _debug(
                    '     rule: Heaviside factor; negative time shift (4.1.4)')
                r, pr, cr = _laplace_transform(ma1[g], t, s, simplify=False)
                return (r, pr, cr)
        if ma2 := ma1[y].match(a - t):
            if ma2[a].is_positive:
                _debug('     rule: Heaviside window open')
                r, pr, cr = _laplace_transform(
                    (1 - Heaviside(t - ma2[a])) * ma1[g], t, s, simplify=False)
                return (r, pr, cr)
            if ma2[a].is_negative:
                _debug('     rule: Heaviside window closed')
                return (0, 0, S.true)
    return None


@DEBUG_WRAP
def _laplace_rule_exp(f, t, s):
    """
    If this function finds a factor ``exp(a*t)``, it applies the
    frequency-shift rule of the Laplace transform and adjusts the convergence
    plane accordingly.  For example, if it gets ``(exp(-a*t)*f(t), t, s)``, it
    will compute ``LaplaceTransform(f(t), t, s+a)``.
    """

    a = Wild('a', exclude=[t])
    y = Wild('y')
    z = Wild('z')
    ma1 = f.match(exp(y)*z)
    if ma1:
        ma2 = ma1[y].collect(t).match(a*t)
        if ma2:
            _debug('     rule: multiply with exp (4.1.5)')
            r, pr, cr = _laplace_transform(ma1[z], t, s-ma2[a],
                                           simplify=False)
            return (r, pr+re(ma2[a]), cr)
    return None


@DEBUG_WRAP
def _laplace_rule_delta(f, t, s):
    """
    If this function finds a factor ``DiracDelta(b*t-a)``, it applies the
    masking property of the delta distribution. For example, if it gets
    ``(DiracDelta(t-a)*f(t), t, s)``, it will return
    ``(f(a)*exp(-a*s), -a, True)``.
    """
    # This rule is not in Bateman54

    a = Wild('a', exclude=[t])
    b = Wild('b', exclude=[t])

    y = Wild('y')
    z = Wild('z')
    ma1 = f.match(DiracDelta(y)*z)
    if ma1 and not ma1[z].has(DiracDelta):
        ma2 = ma1[y].collect(t).match(b*t-a)
        if ma2:
            _debug('     rule: multiply with DiracDelta')
            loc = ma2[a]/ma2[b]
            if re(loc) >= 0 and im(loc) == 0:
                fn = exp(-ma2[a]/ma2[b]*s)*ma1[z]
                if fn.has(sin, cos):
                    # Then it may be possible that a sinc() is present in the
                    # term; let's try this:
                    fn = fn.rewrite(sinc).ratsimp()
                n, d = [x.subs(t, ma2[a]/ma2[b]) for x in fn.as_numer_denom()]
                if d != 0:
                    return (n/d/ma2[b], S.NegativeInfinity, S.true)
                else:
                    return None
            else:
                return (0, S.NegativeInfinity, S.true)
        if ma1[y].is_polynomial(t):
            ro = roots(ma1[y], t)
            if ro != {} and set(ro.values()) == {1}:
                slope = diff(ma1[y], t)
                r = Add(
                    *[exp(-x*s)*ma1[z].subs(t, s)/slope.subs(t, x)
                      for x in list(ro.keys()) if im(x) == 0 and re(x) >= 0])
                return (r, S.NegativeInfinity, S.true)
    return None


@DEBUG_WRAP
def _laplace_trig_split(fn):
    """
    Helper function for `_laplace_rule_trig`.  This function returns two terms
    `f` and `g`.  `f` contains all product terms with sin, cos, sinh, cosh in
    them; `g` contains everything else.
    """
    trigs = [S.One]
    other = [S.One]
    for term in Mul.make_args(fn):
        if term.has(sin, cos, sinh, cosh, exp):
            trigs.append(term)
        else:
            other.append(term)
    f = Mul(*trigs)
    g = Mul(*other)
    return f, g


@DEBUG_WRAP
def _laplace_trig_expsum(f, t):
    """
    Helper function for `_laplace_rule_trig`.  This function expects the `f`
    from `_laplace_trig_split`.  It returns two lists `xm` and `xn`.  `xm` is
    a list of dictionaries with keys `k` and `a` representing a function
    `k*exp(a*t)`.  `xn` is a list of all terms that cannot be brought into
    that form, which may happen, e.g., when a trigonometric function has
    another function in its argument.
    """
    c1 = Wild('c1', exclude=[t])
    c0 = Wild('c0', exclude=[t])
    p = Wild('p', exclude=[t])
    xm = []
    xn = []

    x1 = f.rewrite(exp).expand()

    for term in Add.make_args(x1):
        if not term.has(t):
            xm.append({'k': term, 'a': 0, re: 0, im: 0})
            continue
        term = _laplace_deep_collect(term.powsimp(combine='exp'), t)

        if (r := term.match(p*exp(c1*t+c0))) is not None:
            xm.append({
                'k': r[p]*exp(r[c0]), 'a': r[c1],
                re: re(r[c1]), im: im(r[c1])})
        else:
            xn.append(term)
    return xm, xn


@DEBUG_WRAP
def _laplace_trig_ltex(xm, t, s):
    """
    Helper function for `_laplace_rule_trig`.  This function takes the list of
    exponentials `xm` from `_laplace_trig_expsum` and simplifies complex
    conjugate and real symmetric poles.  It returns the result as a sum and
    the convergence plane.
    """
    results = []
    planes = []

    def _simpc(coeffs):
        nc = coeffs.copy()
        for k in range(len(nc)):
            ri = nc[k].as_real_imag()
            if ri[0].has(im):
                nc[k] = nc[k].rewrite(cos)
            else:
                nc[k] = (ri[0] + I*ri[1]).rewrite(cos)
        return nc

    def _quadpole(t1, k1, k2, k3, s):
        a, k0, a_r, a_i = t1['a'], t1['k'], t1[re], t1[im]
        nc = [
            k0 + k1 + k2 + k3,
            a*(k0 + k1 - k2 - k3) - 2*I*a_i*k1 + 2*I*a_i*k2,
            (
                a**2*(-k0 - k1 - k2 - k3) +
                a*(4*I*a_i*k0 + 4*I*a_i*k3) +
                4*a_i**2*k0 + 4*a_i**2*k3),
            (
                a**3*(-k0 - k1 + k2 + k3) +
                a**2*(4*I*a_i*k0 + 2*I*a_i*k1 - 2*I*a_i*k2 - 4*I*a_i*k3) +
                a*(4*a_i**2*k0 - 4*a_i**2*k3))
        ]
        dc = [
            S.One, S.Zero, 2*a_i**2 - 2*a_r**2,
            S.Zero, a_i**4 + 2*a_i**2*a_r**2 + a_r**4]
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    def _ccpole(t1, k1, s):
        a, k0, a_r, a_i = t1['a'], t1['k'], t1[re], t1[im]
        nc = [k0 + k1, -a*k0 - a*k1 + 2*I*a_i*k0]
        dc = [S.One, -2*a_r, a_i**2 + a_r**2]
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    def _rspole(t1, k2, s):
        a, k0, a_r, a_i = t1['a'], t1['k'], t1[re], t1[im]
        nc = [k0 + k2, a*k0 - a*k2 - 2*I*a_i*k0]
        dc = [S.One, -2*I*a_i, -a_i**2 - a_r**2]
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    def _sypole(t1, k3, s):
        a, k0 = t1['a'], t1['k']
        nc = [k0 + k3, a*(k0 - k3)]
        dc = [S.One, S.Zero, -a**2]
        n = Add(
            *[x*s**y for x, y in zip(_simpc(nc), range(len(nc))[::-1])])
        d = Add(
            *[x*s**y for x, y in zip(dc, range(len(dc))[::-1])])
        return n/d

    def _simplepole(t1, s):
        a, k0 = t1['a'], t1['k']
        n = k0
        d = s - a
        return n/d

    while len(xm) > 0:
        t1 = xm.pop()
        i_imagsym = None
        i_realsym = None
        i_pointsym = None
        # The following code checks all remaining poles. If t1 is a pole at
        # a+b*I, then we check for a-b*I, -a+b*I, and -a-b*I, and
        # assign the respective indices to i_imagsym, i_realsym, i_pointsym.
        # -a-b*I / i_pointsym only applies if both a and b are != 0.
        for i in range(len(xm)):
            real_eq = t1[re] == xm[i][re]
            realsym = t1[re] == -xm[i][re]
            imag_eq = t1[im] == xm[i][im]
            imagsym = t1[im] == -xm[i][im]
            if realsym and imagsym and t1[re] != 0 and t1[im] != 0:
                i_pointsym = i
            elif realsym and imag_eq and t1[re] != 0:
                i_realsym = i
            elif real_eq and imagsym and t1[im] != 0:
                i_imagsym = i

        # The next part looks for four possible pole constellations:
        # quad:   a+b*I, a-b*I, -a+b*I, -a-b*I
        # cc:     a+b*I, a-b*I (a may be zero)
        # quad:   a+b*I, -a+b*I (b may be zero)
        # point:  a+b*I, -a-b*I (a!=0 and b!=0 is needed, but that has been
        #                        asserted when finding i_pointsym above.)
        # If none apply, then t1 is a simple pole.
        if (
                i_imagsym is not None and i_realsym is not None
                and i_pointsym is not None):
            results.append(
                _quadpole(t1,
                          xm[i_imagsym]['k'], xm[i_realsym]['k'],
                          xm[i_pointsym]['k'], s))
            planes.append(Abs(re(t1['a'])))
            # The three additional poles have now been used; to pop them
            # easily we have to do it from the back.
            indices_to_pop = [i_imagsym, i_realsym, i_pointsym]
            indices_to_pop.sort(reverse=True)
            for i in indices_to_pop:
                xm.pop(i)
        elif i_imagsym is not None:
            results.append(_ccpole(t1, xm[i_imagsym]['k'], s))
            planes.append(t1[re])
            xm.pop(i_imagsym)
        elif i_realsym is not None:
            results.append(_rspole(t1, xm[i_realsym]['k'], s))
            planes.append(Abs(t1[re]))
            xm.pop(i_realsym)
        elif i_pointsym is not None:
            results.append(_sypole(t1, xm[i_pointsym]['k'], s))
            planes.append(Abs(t1[re]))
            xm.pop(i_pointsym)
        else:
            results.append(_simplepole(t1, s))
            planes.append(t1[re])

    return Add(*results), Max(*planes)


@DEBUG_WRAP
def _laplace_rule_trig(fn, t_, s):
    """
    This rule covers trigonometric factors by splitting everything into a
    sum of exponential functions and collecting complex conjugate poles and
    real symmetric poles.
    """
    t = Dummy('t', real=True)

    if not fn.has(sin, cos, sinh, cosh):
        return None

    f, g = _laplace_trig_split(fn.subs(t_, t))
    xm, xn = _laplace_trig_expsum(f, t)

    if len(xn) > 0:
        # TODO not implemented yet, but also not important
        return None

    if not g.has(t):
        r, p = _laplace_trig_ltex(xm, t, s)
        return g*r, p, S.true
    else:
        # Just transform `g` and make frequency-shifted copies
        planes = []
        results = []
        G, G_plane, G_cond = _laplace_transform(g, t, s, simplify=False)
        for x1 in xm:
            results.append(x1['k']*G.subs(s, s-x1['a']))
            planes.append(G_plane+re(x1['a']))
    return Add(*results).subs(t, t_), Max(*planes), G_cond


@DEBUG_WRAP
def _laplace_rule_diff(f, t, s):
    """
    This function looks for derivatives in the time domain and replaces it
    by factors of `s` and initial conditions in the frequency domain. For
    example, if it gets ``(diff(f(t), t), t, s)``, it will compute
    ``s*LaplaceTransform(f(t), t, s) - f(0)``.
    """

    a = Wild('a', exclude=[t])
    n = Wild('n', exclude=[t])
    g = WildFunction('g')
    ma1 = f.match(a*Derivative(g, (t, n)))
    if ma1 and ma1[n].is_integer:
        m = [z.has(t) for z in ma1[g].args]
        if sum(m) == 1:
            _debug('     rule: time derivative (4.1.8)')
            d = []
            for k in range(ma1[n]):
                if k == 0:
                    y = ma1[g].subs(t, 0)
                else:
                    y = Derivative(ma1[g], (t, k)).subs(t, 0)
                d.append(s**(ma1[n]-k-1)*y)
            r, pr, cr = _laplace_transform(ma1[g], t, s, simplify=False)
            return (ma1[a]*(s**ma1[n]*r - Add(*d)),  pr, cr)
    return None


@DEBUG_WRAP
def _laplace_rule_sdiff(f, t, s):
    """
    This function looks for multiplications with polynoimials in `t` as they
    correspond to differentiation in the frequency domain. For example, if it
    gets ``(t*f(t), t, s)``, it will compute
    ``-Derivative(LaplaceTransform(f(t), t, s), s)``.
    """

    if f.is_Mul:
        pfac = [1]
        ofac = [1]
        for fac in Mul.make_args(f):
            if fac.is_polynomial(t):
                pfac.append(fac)
            else:
                ofac.append(fac)
        if len(pfac) > 1:
            pex = prod(pfac)
            pc = Poly(pex, t).all_coeffs()
            N = len(pc)
            if N > 1:
                oex = prod(ofac)
                r_, p_, c_ = _laplace_transform(oex, t, s, simplify=False)
                deri = [r_]
                d1 = False
                try:
                    d1 = -diff(deri[-1], s)
                except ValueError:
                    d1 = False
                if r_.has(LaplaceTransform):
                    for k in range(N-1):
                        deri.append((-1)**(k+1)*Derivative(r_, s, k+1))
                elif d1:
                    deri.append(d1)
                    for k in range(N-2):
                        deri.append(-diff(deri[-1], s))
                if d1:
                    r = Add(*[pc[N-n-1]*deri[n] for n in range(N)])
                    return (r, p_, c_)
    # We still have to cover the possibility that there is a symbolic positive
    # integer exponent.
    n = Wild('n', exclude=[t])
    g = Wild('g')
    if ma1 := f.match(t**n*g):
        if ma1[n].is_integer and ma1[n].is_positive:
            r_, p_, c_ = _laplace_transform(ma1[g], t, s, simplify=False)
            return (-1)**ma1[n]*diff(r_, (s, ma1[n])), p_, c_
    return None


@DEBUG_WRAP
def _laplace_expand(f, t, s):
    """
    This function tries to expand its argument with successively stronger
    methods: first it will expand on the top level, then it will expand any
    multiplications in depth, then it will try all available expansion methods,
    and finally it will try to expand trigonometric functions.

    If it can expand, it will then compute the Laplace transform of the
    expanded term.
    """

    r = expand(f, deep=False)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    r = expand_mul(f)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    r = expand(f)
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    if r != f:
        return _laplace_transform(r, t, s, simplify=False)
    r = expand(expand_trig(f))
    if r.is_Add:
        return _laplace_transform(r, t, s, simplify=False)
    return None


@DEBUG_WRAP
def _laplace_apply_prog_rules(f, t, s):
    """
    This function applies all program rules and returns the result if one
    of them gives a result.
    """

    prog_rules = [_laplace_rule_heaviside, _laplace_rule_delta,
                  _laplace_rule_timescale, _laplace_rule_exp,
                  _laplace_rule_trig,
                  _laplace_rule_diff, _laplace_rule_sdiff]

    for p_rule in prog_rules:
        if (L := p_rule(f, t, s)) is not None:
            return L
    return None


@DEBUG_WRAP
def _laplace_apply_simple_rules(f, t, s):
    """
    This function applies all simple rules and returns the result if one
    of them gives a result.
    """
    simple_rules, t_, s_ = _laplace_build_rules()
    prep_old = ''
    prep_f = ''
    for t_dom, s_dom, check, plane, prep in simple_rules:
        if prep_old != prep:
            prep_f = prep(f.subs({t: t_}))
            prep_old = prep
        ma = prep_f.match(t_dom)
        if ma:
            try:
                c = check.xreplace(ma)
            except TypeError:
                # This may happen if the time function has imaginary
                # numbers in it. Then we give up.
                continue
            if c == S.true:
                return (s_dom.xreplace(ma).subs({s_: s}),
                        plane.xreplace(ma), S.true)
    return None


@DEBUG_WRAP
def _piecewise_to_heaviside(f, t):
    """
    This function converts a Piecewise expression to an expression written
    with Heaviside. It is not exact, but valid in the context of the Laplace
    transform.
    """
    if not t.is_real:
        r = Dummy('r', real=True)
        return _piecewise_to_heaviside(f.xreplace({t: r}), r).xreplace({r: t})
    x = piecewise_exclusive(f)
    r = []
    for fn, cond in x.args:
        # Here we do not need to do many checks because piecewise_exclusive
        # has a clearly predictable output. However, if any of the conditions
        # is not relative to t, this function just returns the input argument.
        if isinstance(cond, Relational) and t in cond.args:
            if isinstance(cond, (Eq, Ne)):
                # We do not cover this case; these would be single-point
                # exceptions that do not play a role in Laplace practice,
                # except if they contain Dirac impulses, and then we can
                # expect users to not try to use Piecewise for writing it.
                return f
            else:
                r.append(Heaviside(cond.gts - cond.lts)*fn)
        elif isinstance(cond, Or) and len(cond.args) == 2:
            # Or(t<2, t>4), Or(t>4, t<=2), ... in any order with any <= >=
            for c2 in cond.args:
                if c2.lhs == t:
                    r.append(Heaviside(c2.gts - c2.lts)*fn)
                else:
                    return f
        elif isinstance(cond, And) and len(cond.args) == 2:
            # And(t>2, t<4), And(t>4, t<=2), ...  in any order with any <= >=
            c0, c1 = cond.args
            if c0.lhs == t and c1.lhs == t:
                if '>' in c0.rel_op:
                    c0, c1 = c1, c0
                r.append(
                    (Heaviside(c1.gts - c1.lts) -
                     Heaviside(c0.lts - c0.gts))*fn)
            else:
                return f
        else:
            return f
    return Add(*r)


def laplace_correspondence(f, fdict, /):
    """
    This helper function takes a function `f` that is the result of a
    ``laplace_transform`` or an ``inverse_laplace_transform``.  It replaces all
    unevaluated ``LaplaceTransform(y(t), t, s)`` by `Y(s)` for any `s` and
    all ``InverseLaplaceTransform(Y(s), s, t)`` by `y(t)` for any `t` if
    ``fdict`` contains a correspondence ``{y: Y}``.

    Parameters
    ==========

    f : sympy expression
        Expression containing unevaluated ``LaplaceTransform`` or
        ``LaplaceTransform`` objects.
    fdict : dictionary
        Dictionary containing one or more function correspondences,
        e.g., ``{x: X, y: Y}`` meaning that ``X`` and ``Y`` are the
        Laplace transforms of ``x`` and ``y``, respectively.

    Examples
    ========

    >>> from sympy import laplace_transform, diff, Function
    >>> from sympy import laplace_correspondence, inverse_laplace_transform
    >>> from sympy.abc import t, s
    >>> y = Function("y")
    >>> Y = Function("Y")
    >>> z = Function("z")
    >>> Z = Function("Z")
    >>> f = laplace_transform(diff(y(t), t, 1) + z(t), t, s, noconds=True)
    >>> laplace_correspondence(f, {y: Y, z: Z})
    s*Y(s) + Z(s) - y(0)
    >>> f = inverse_laplace_transform(Y(s), s, t)
    >>> laplace_correspondence(f, {y: Y})
    y(t)
    """
    p = Wild('p')
    s = Wild('s')
    t = Wild('t')
    a = Wild('a')
    if (
            not isinstance(f, Expr)
            or (not f.has(LaplaceTransform)
                and not f.has(InverseLaplaceTransform))):
        return f
    for y, Y in fdict.items():
        if (
                (m := f.match(LaplaceTransform(y(a), t, s))) is not None
                and m[a] == m[t]):
            return Y(m[s])
        if (
                (m := f.match(InverseLaplaceTransform(Y(a), s, t, p)))
                is not None
                and m[a] == m[s]):
            return y(m[t])
    func = f.func
    args = [laplace_correspondence(arg, fdict) for arg in f.args]
    return func(*args)


def laplace_initial_conds(f, t, fdict, /):
    """
    This helper function takes a function `f` that is the result of a
    ``laplace_transform``.  It takes an fdict of the form ``{y: [1, 4, 2]}``,
    where the values in the list are the initial value, the initial slope, the
    initial second derivative, etc., of the function `y(t)`, and replaces all
    unevaluated initial conditions.

    Parameters
    ==========

    f : sympy expression
        Expression containing initial conditions of unevaluated functions.
    t : sympy expression
        Variable for which the initial conditions are to be applied.
    fdict : dictionary
        Dictionary containing a list of initial conditions for every
        function, e.g., ``{y: [0, 1, 2], x: [3, 4, 5]}``. The order
        of derivatives is ascending, so `0`, `1`, `2` are `y(0)`, `y'(0)`,
        and `y''(0)`, respectively.

    Examples
    ========

    >>> from sympy import laplace_transform, diff, Function
    >>> from sympy import laplace_correspondence, laplace_initial_conds
    >>> from sympy.abc import t, s
    >>> y = Function("y")
    >>> Y = Function("Y")
    >>> f = laplace_transform(diff(y(t), t, 3), t, s, noconds=True)
    >>> g = laplace_correspondence(f, {y: Y})
    >>> laplace_initial_conds(g, t, {y: [2, 4, 8, 16, 32]})
    s**3*Y(s) - 2*s**2 - 4*s - 8
    """
    for y, ic in fdict.items():
        for k in range(len(ic)):
            if k == 0:
                f = f.replace(y(0), ic[0])
            elif k == 1:
                f = f.replace(Subs(Derivative(y(t), t), t, 0), ic[1])
            else:
                f = f.replace(Subs(Derivative(y(t), (t, k)), t, 0), ic[k])
    return f


@DEBUG_WRAP
def _laplace_transform(fn, t_, s_, *, simplify):
    """
    Front-end function of the Laplace transform. It tries to apply all known
    rules recursively, and if everything else fails, it tries to integrate.
    """

    terms_t = Add.make_args(fn)
    terms_s = []
    terms = []
    planes = []
    conditions = []

    for ff in terms_t:
        k, ft = ff.as_independent(t_, as_Add=False)
        if ft.has(SingularityFunction):
            _terms = Add.make_args(ft.rewrite(Heaviside))
            for _term in _terms:
                k1, f1 = _term.as_independent(t_, as_Add=False)
                terms.append((k*k1, f1))
        elif ft.func == Piecewise and not ft.has(DiracDelta(t_)):
            _terms = Add.make_args(_piecewise_to_heaviside(ft, t_))
            for _term in _terms:
                k1, f1 = _term.as_independent(t_, as_Add=False)
                terms.append((k*k1, f1))
        else:
            terms.append((k, ft))

    for k, ft in terms:
        if ft.has(SingularityFunction):
            r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
        else:
            if ft.has(Heaviside(t_)) and not ft.has(DiracDelta(t_)):
                # For t>=0, Heaviside(t)=1 can be used, except if there is also
                # a DiracDelta(t) present, in which case removing Heaviside(t)
                # is unnecessary because _laplace_rule_delta can deal with it.
                ft = ft.subs(Heaviside(t_), 1)
            if (
                    (r := _laplace_apply_simple_rules(ft, t_, s_))
                    is not None or
                    (r := _laplace_apply_prog_rules(ft, t_, s_))
                    is not None or
                    (r := _laplace_expand(ft, t_, s_)) is not None):
                pass
            elif any(undef.has(t_) for undef in ft.atoms(AppliedUndef)):
                # If there are undefined functions f(t) then integration is
                # unlikely to do anything useful so we skip it and given an
                # unevaluated LaplaceTransform.
                r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
            elif (r := _laplace_transform_integration(
                    ft, t_, s_, simplify=simplify)) is not None:
                pass
            else:
                r = (LaplaceTransform(ft, t_, s_), S.NegativeInfinity, True)
        (ri_, pi_, ci_) = r
        terms_s.append(k*ri_)
        planes.append(pi_)
        conditions.append(ci_)

    result = Add(*terms_s)
    if simplify:
        result = result.simplify(doit=False)
    plane = Max(*planes)
    condition = And(*conditions)

    return result, plane, condition


class LaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute Laplace transforms, see the :func:`laplace_transform`
    docstring.

    If this is called with ``.doit()``, it returns the Laplace transform as an
    expression. If it is called with ``.doit(noconds=False)``, it returns a
    tuple containing the same expression, a convergence plane, and conditions.
    """

    _name = 'Laplace'

    def _compute_transform(self, f, t, s, **hints):
        _simplify = hints.get('simplify', False)
        LT = _laplace_transform_integration(f, t, s, simplify=_simplify)
        return LT

    def _as_integral(self, f, t, s):
        return Integral(f*exp(-s*t), (t, S.Zero, S.Infinity))

    def doit(self, **hints):
        """
        Try to evaluate the transform in closed form.

        Explanation
        ===========

        Standard hints are the following:
        - ``noconds``:  if True, do not return convergence conditions. The
        default setting is `True`.
        - ``simplify``: if True, it simplifies the final result. The
        default setting is `False`.
        """
        _noconds = hints.get('noconds', True)
        _simplify = hints.get('simplify', False)

        debugf('[LT doit] (%s, %s, %s)', (self.function,
                                          self.function_variable,
                                          self.transform_variable))

        t_ = self.function_variable
        s_ = self.transform_variable
        fn = self.function

        r = _laplace_transform(fn, t_, s_, simplify=_simplify)

        if _noconds:
            return r[0]
        else:
            return r


def laplace_transform(f, t, s, legacy_matrix=True, **hints):
    r"""
    Compute the Laplace Transform `F(s)` of `f(t)`,

    .. math :: F(s) = \int_{0^{-}}^\infty e^{-st} f(t) \mathrm{d}t.

    Explanation
    ===========

    For all sensible functions, this converges absolutely in a
    half-plane

    .. math :: a < \operatorname{Re}(s)

    This function returns ``(F, a, cond)`` where ``F`` is the Laplace
    transform of ``f``, `a` is the half-plane of convergence, and `cond` are
    auxiliary convergence conditions.

    The implementation is rule-based, and if you are interested in which
    rules are applied, and whether integration is attempted, you can switch
    debug information on by setting ``sympy.SYMPY_DEBUG=True``. The numbers
    of the rules in the debug information (and the code) refer to Bateman's
    Tables of Integral Transforms [1].

    The lower bound is `0-`, meaning that this bound should be approached
    from the lower side. This is only necessary if distributions are involved.
    At present, it is only done if `f(t)` contains ``DiracDelta``, in which
    case the Laplace transform is computed implicitly as

    .. math ::
        F(s) = \lim_{\tau\to 0^{-}} \int_{\tau}^\infty e^{-st}
        f(t) \mathrm{d}t

    by applying rules.

    If the Laplace transform cannot be fully computed in closed form, this
    function returns expressions containing unevaluated
    :class:`LaplaceTransform` objects.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`. If
    ``noconds=True``, only `F` will be returned (i.e. not ``cond``, and also
    not the plane ``a``).

    .. deprecated:: 1.9
        Legacy behavior for matrices where ``laplace_transform`` with
        ``noconds=False`` (the default) returns a Matrix whose elements are
        tuples. The behavior of ``laplace_transform`` for matrices will change
        in a future release of SymPy to return a tuple of the transformed
        Matrix and the convergence conditions for the matrix as a whole. Use
        ``legacy_matrix=False`` to enable the new behavior.

    Examples
    ========

    >>> from sympy import DiracDelta, exp, laplace_transform
    >>> from sympy.abc import t, s, a
    >>> laplace_transform(t**4, t, s)
    (24/s**5, 0, True)
    >>> laplace_transform(t**a, t, s)
    (gamma(a + 1)/(s*s**a), 0, re(a) > -1)
    >>> laplace_transform(DiracDelta(t)-a*exp(-a*t), t, s, simplify=True)
    (s/(a + s), -re(a), True)

    There are also helper functions that make it easy to solve differential
    equations by Laplace transform. For example, to solve

    .. math :: m x''(t) + d x'(t) + k x(t) = 0

    with initial value `0` and initial derivative `v`:

    >>> from sympy import Function, laplace_correspondence, diff, solve
    >>> from sympy import laplace_initial_conds, inverse_laplace_transform
    >>> from sympy.abc import d, k, m, v
    >>> x = Function('x')
    >>> X = Function('X')
    >>> f = m*diff(x(t), t, 2) + d*diff(x(t), t) + k*x(t)
    >>> F = laplace_transform(f, t, s, noconds=True)
    >>> F = laplace_correspondence(F, {x: X})
    >>> F = laplace_initial_conds(F, t, {x: [0, v]})
    >>> F
    d*s*X(s) + k*X(s) + m*(s**2*X(s) - v)
    >>> Xs = solve(F, X(s))[0]
    >>> Xs
    m*v/(d*s + k + m*s**2)
    >>> inverse_laplace_transform(Xs, s, t)
    2*v*exp(-d*t/(2*m))*sin(t*sqrt((-d**2 + 4*k*m)/m**2)/2)*Heaviside(t)/sqrt((-d**2 + 4*k*m)/m**2)

    References
    ==========

    .. [1] Erdelyi, A. (ed.), Tables of Integral Transforms, Volume 1,
           Bateman Manuscript Prooject, McGraw-Hill (1954), available:
           https://resolver.caltech.edu/CaltechAUTHORS:20140123-101456353

    See Also
    ========

    inverse_laplace_transform, mellin_transform, fourier_transform
    hankel_transform, inverse_hankel_transform

    """

    _noconds = hints.get('noconds', False)
    _simplify = hints.get('simplify', False)

    if isinstance(f, MatrixBase) and hasattr(f, 'applyfunc'):

        conds = not hints.get('noconds', False)

        if conds and legacy_matrix:
            adt = 'deprecated-laplace-transform-matrix'
            sympy_deprecation_warning(
                """
Calling laplace_transform() on a Matrix with noconds=False (the default) is
deprecated. Either noconds=True or use legacy_matrix=False to get the new
behavior.
                """,
                deprecated_since_version='1.9',
                active_deprecations_target=adt,
            )
            # Temporarily disable the deprecation warning for non-Expr objects
            # in Matrix
            with ignore_warnings(SymPyDeprecationWarning):
                return f.applyfunc(
                    lambda fij: laplace_transform(fij, t, s, **hints))
        else:
            elements_trans = [laplace_transform(
                fij, t, s, **hints) for fij in f]
            if conds:
                elements, avals, conditions = zip(*elements_trans)
                f_laplace = type(f)(*f.shape, elements)
                return f_laplace, Max(*avals), And(*conditions)
            else:
                return type(f)(*f.shape, elements_trans)

    LT, p, c = LaplaceTransform(f, t, s).doit(noconds=False,
                                              simplify=_simplify)

    if not _noconds:
        return LT, p, c
    else:
        return LT


@DEBUG_WRAP
def _inverse_laplace_transform_integration(F, s, t_, plane, *, simplify):
    """ The backend function for inverse Laplace transforms. """
    from sympy.integrals.meijerint import meijerint_inversion, _get_coeff_exp
    from sympy.integrals.transforms import inverse_mellin_transform

    # There are two strategies we can try:
    # 1) Use inverse mellin transform, related by a simple change of variables.
    # 2) Use the inversion integral.

    t = Dummy('t', real=True)

    def pw_simp(*args):
        """ Simplify a piecewise expression from hyperexpand. """
        if len(args) != 3:
            return Piecewise(*args)
        arg = args[2].args[0].argument
        coeff, exponent = _get_coeff_exp(arg, t)
        e1 = args[0].args[0]
        e2 = args[1].args[0]
        return (
            Heaviside(1/Abs(coeff) - t**exponent)*e1 +
            Heaviside(t**exponent - 1/Abs(coeff))*e2)

    if F.is_rational_function(s):
        F = F.apart(s)

    if F.is_Add:
        f = Add(
            *[_inverse_laplace_transform_integration(X, s, t, plane, simplify)
              for X in F.args])
        return _simplify(f.subs(t, t_), simplify), True

    try:
        f, cond = inverse_mellin_transform(F, s, exp(-t), (None, S.Infinity),
                                           needeval=True, noconds=False)
    except IntegralTransformError:
        f = None

    if f is None:
        f = meijerint_inversion(F, s, t)
        if f is None:
            return None
        if f.is_Piecewise:
            f, cond = f.args[0]
            if f.has(Integral):
                return None
        else:
            cond = S.true
        f = f.replace(Piecewise, pw_simp)

    if f.is_Piecewise:
        # many of the functions called below can't work with piecewise
        # (b/c it has a bool in args)
        return f.subs(t, t_), cond

    u = Dummy('u')

    def simp_heaviside(arg, H0=S.Half):
        a = arg.subs(exp(-t), u)
        if a.has(t):
            return Heaviside(arg, H0)
        from sympy.solvers.inequalities import _solve_inequality
        rel = _solve_inequality(a > 0, u)
        if rel.lts == u:
            k = log(rel.gts)
            return Heaviside(t + k, H0)
        else:
            k = log(rel.lts)
            return Heaviside(-(t + k), H0)

    f = f.replace(Heaviside, simp_heaviside)

    def simp_exp(arg):
        return expand_complex(exp(arg))

    f = f.replace(exp, simp_exp)

    return _simplify(f.subs(t, t_), simplify), cond


@DEBUG_WRAP
def _complete_the_square_in_denom(f, s):
    from sympy.simplify.radsimp import fraction
    [n, d] = fraction(f)
    if d.is_polynomial(s):
        cf = d.as_poly(s).all_coeffs()
        if len(cf) == 3:
            a, b, c = cf
            d = a*((s+b/(2*a))**2+c/a-(b/(2*a))**2)
    return n/d


@cacheit
def _inverse_laplace_build_rules():
    """
    This is an internal helper function that returns the table of inverse
    Laplace transform rules in terms of the time variable `t` and the
    frequency variable `s`.  It is used by `_inverse_laplace_apply_rules`.
    """
    s = Dummy('s')
    t = Dummy('t')
    a = Wild('a', exclude=[s])
    b = Wild('b', exclude=[s])
    c = Wild('c', exclude=[s])

    _debug('_inverse_laplace_build_rules is building rules')

    def _frac(f, s):
        try:
            return f.factor(s)
        except PolynomialError:
            return f

    def same(f): return f
    # This list is sorted according to the prep function needed.
    _ILT_rules = [
        (a/s, a, S.true, same, 1),
        (
            b*(s+a)**(-c), t**(c-1)*exp(-a*t)/gamma(c),
            S.true, same, 1),
        (1/(s**2+a**2)**2, (sin(a*t) - a*t*cos(a*t))/(2*a**3),
         S.true, same, 1),
        # The next two rules must be there in that order. For the second
        # one, the condition would be a != 0 or, respectively, to take the
        # limit a -> 0 after the transform if a == 0. It is much simpler if
        # the case a == 0 has its own rule.
        (1/(s**b), t**(b - 1)/gamma(b), S.true, same, 1),
        (1/(s*(s+a)**b), lowergamma(b, a*t)/(a**b*gamma(b)),
         S.true, same, 1)
    ]
    return _ILT_rules, s, t


@DEBUG_WRAP
def _inverse_laplace_apply_simple_rules(f, s, t):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    if f == 1:
        _debug('     rule: 1 o---o DiracDelta()')
        return DiracDelta(t), S.true

    _ILT_rules, s_, t_ = _inverse_laplace_build_rules()
    _prep = ''
    fsubs = f.subs({s: s_})

    for s_dom, t_dom, check, prep, fac in _ILT_rules:
        if _prep != (prep, fac):
            _F = prep(fsubs*fac)
            _prep = (prep, fac)
        ma = _F.match(s_dom)
        if ma:
            c = check
            if c is not S.true:
                args = [x.xreplace(ma) for x in c[0]]
                c = c[1](*args)
            if c == S.true:
                return Heaviside(t)*t_dom.xreplace(ma).subs({t_: t}), S.true

    return None


@DEBUG_WRAP
def _inverse_laplace_diff(f, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    a = Wild('a', exclude=[s])
    n = Wild('n', exclude=[s])
    g = Wild('g')
    ma = f.match(a*Derivative(g, (s, n)))
    if ma and ma[n].is_integer:
        _debug('     rule: t**n*f(t) o---o (-1)**n*diff(F(s), s, n)')
        r, c = _inverse_laplace_transform(
            ma[g], s, t, plane, simplify=False, dorational=False)
        return (-t)**ma[n]*r, c
    return None


@DEBUG_WRAP
def _inverse_laplace_time_shift(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    a = Wild('a', exclude=[s])
    g = Wild('g')

    if not F.has(s):
        return F*DiracDelta(t), S.true
    if not F.has(exp):
        return None

    ma1 = F.match(exp(a*s))
    if ma1:
        if ma1[a].is_negative:
            _debug('     rule: exp(-a*s) o---o DiracDelta(t-a)')
            return DiracDelta(t+ma1[a]), S.true
        else:
            return InverseLaplaceTransform(F, s, t, plane), S.true

    ma1 = F.match(exp(a*s)*g)
    if ma1:
        if ma1[a].is_negative:
            _debug('     rule: exp(-a*s)*F(s) o---o Heaviside(t-a)*f(t-a)')
            return _inverse_laplace_transform(
                ma1[g], s, t+ma1[a], plane, simplify=False, dorational=True)
        else:
            return InverseLaplaceTransform(F, s, t, plane), S.true
    return None


@DEBUG_WRAP
def _inverse_laplace_freq_shift(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    if not F.has(s):
        return F*DiracDelta(t), S.true
    if len(args := F.args) == 1:
        a = Wild('a', exclude=[s])
        if (ma := args[0].match(s-a)) and re(ma[a]).is_positive:
            _debug('     rule: F(s-a) o---o exp(-a*t)*f(t)')
            return (
                exp(-ma[a]*t) *
                InverseLaplaceTransform(F.func(s), s, t, plane), S.true)
    return None


@DEBUG_WRAP
def _inverse_laplace_time_diff(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    n = Wild('n', exclude=[s])
    g = Wild('g')

    ma1 = F.match(s**n*g)
    if ma1 and ma1[n].is_integer and ma1[n].is_positive:
        _debug('     rule: s**n*F(s) o---o diff(f(t), t, n)')
        r, c = _inverse_laplace_transform(
            ma1[g], s, t, plane, simplify=False, dorational=True)
        r = r.replace(Heaviside(t), 1)
        if r.has(InverseLaplaceTransform):
            return diff(r, t, ma1[n]), c
        else:
            return Heaviside(t)*diff(r, t, ma1[n]), c
    return None


@DEBUG_WRAP
def _inverse_laplace_irrational(fn, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """

    a = Wild('a', exclude=[s])
    b = Wild('b', exclude=[s])
    m = Wild('m', exclude=[s])
    n = Wild('n', exclude=[s])

    result = None
    condition = S.true

    fa = fn.as_ordered_factors()

    ma = [x.match((a*s**m+b)**n) for x in fa]

    if None in ma:
        return None

    constants = S.One
    zeros = []
    poles = []
    rest = []

    for term in ma:
        if term[a] == 0:
            constants = constants*term
        elif term[n].is_positive:
            zeros.append(term)
        elif term[n].is_negative:
            poles.append(term)
        else:
            rest.append(term)

    # The code below assumes that the poles are sorted in a specific way:
    poles = sorted(poles, key=lambda x: (x[n], x[b] != 0, x[b]))
    zeros = sorted(zeros, key=lambda x: (x[n], x[b] != 0, x[b]))

    if len(rest) != 0:
        return None

    if len(poles) == 1 and len(zeros) == 0:
        if poles[0][n] == -1 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0) == 1/a0 * 1/(sqrt(s)+b0/a0)
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]*constants
            if a_.is_positive:
                result = (
                    k_/sqrt(pi)/sqrt(t) -
                    k_*a_*exp(a_**2*t)*erfc(a_*sqrt(t)))
                _debug('     rule 5.3.4')
        elif poles[0][n] == -2 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0)**2 == 1/a0**2 * 1/(sqrt(s)+b0/a0)**2
            a_sq = poles[0][b]/poles[0][a]
            a_ = a_sq**2
            k_ = 1/poles[0][a]**2*constants
            if a_sq.is_positive:
                result = (
                    k_*(1 - 2/sqrt(pi)*sqrt(a_)*sqrt(t) +
                        (1-2*a_*t)*exp(a_*t)*(erf(sqrt(a_)*sqrt(t))-1)))
                _debug('     rule 5.3.10')
        elif poles[0][n] == -3 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0)**3 == 1/a0**3 * 1/(sqrt(s)+b0/a0)**3
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]**3*constants
            if a_.is_positive:
                result = (
                    k_*(2/sqrt(pi)*(a_**2*t+1)*sqrt(t) -
                        a_*t*exp(a_**2*t)*(2*a_**2*t+3)*erfc(a_*sqrt(t))))
                _debug('     rule 5.3.13')
        elif poles[0][n] == -4 and poles[0][m] == S.Half:
            # 1/(a0*sqrt(s)+b0)**4 == 1/a0**4 * 1/(sqrt(s)+b0/a0)**4
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]**4*constants/3
            if a_.is_positive:
                result = (
                    k_*(t*(4*a_**4*t**2+12*a_**2*t+3)*exp(a_**2*t) *
                        erfc(a_*sqrt(t)) -
                        2/sqrt(pi)*a_**3*t**(S(5)/2)*(2*a_**2*t+5)))
                _debug('     rule 5.3.16')
        elif poles[0][n] == -S.Half and poles[0][m] == 2:
            # 1/sqrt(a0*s**2+b0) == 1/sqrt(a0) * 1/sqrt(s**2+b0/a0)
            a_ = sqrt(poles[0][b]/poles[0][a])
            k_ = 1/sqrt(poles[0][a])*constants
            result = (k_*(besselj(0, a_*t)))
            _debug('     rule 5.3.35/44')

    elif len(poles) == 1 and len(zeros) == 1:
        if (
                poles[0][n] == -3 and poles[0][m] == S.Half and
                zeros[0][n] == S.Half and zeros[0][b] == 0):
            # sqrt(az*s)/(ap*sqrt(s+bp)**3)
            # == sqrt(az)/ap * sqrt(s)/(sqrt(s+bp)**3)
            a_ = poles[0][b]
            k_ = sqrt(zeros[0][a])/poles[0][a]*constants
            result = (
                k_*(2*a_**4*t**2+5*a_**2*t+1)*exp(a_**2*t) *
                erfc(a_*sqrt(t)) - 2/sqrt(pi)*a_*(a_**2*t+2)*sqrt(t))
            _debug('     rule 5.3.14')
        if (
                poles[0][n] == -1 and poles[0][m] == 1 and
                zeros[0][n] == S.Half and zeros[0][m] == 1):
            # sqrt(az*s+bz)/(ap*s+bp)
            # == sqrt(az)/ap * (sqrt(s+bz/az)/(s+bp/ap))
            a_ = zeros[0][b]/zeros[0][a]
            b_ = poles[0][b]/poles[0][a]
            k_ = sqrt(zeros[0][a])/poles[0][a]*constants
            result = (
                k_*(exp(-a_*t)/sqrt(t)/sqrt(pi)+sqrt(a_-b_) *
                    exp(-b_*t)*erf(sqrt(a_-b_)*sqrt(t))))
            _debug('     rule 5.3.22')

    elif len(poles) == 2 and len(zeros) == 0:
        if (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[1][n] == -S.Half and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1/((a0*s+b0)*sqrt(a1*s))
            # == 1/(a0*sqrt(a1)) * 1/((s+b0/a0)*sqrt(s))
            a_ = -poles[0][b]/poles[0][a]
            k_ = 1/sqrt(poles[1][a])/poles[0][a]*constants
            if a_.is_positive:
                result = (k_/sqrt(a_)*exp(a_*t)*erf(sqrt(a_)*sqrt(t)))
                _debug('     rule 5.3.1')
        elif (
                poles[0][n] == -1 and poles[0][m] == 1 and poles[0][b] == 0 and
                poles[1][n] == -1 and poles[1][m] == S.Half):
            # 1/(a0*s*(a1*sqrt(s)+b1))
            # == 1/(a0*a1) * 1/(s*(sqrt(s)+b1/a1))
            a_ = poles[1][b]/poles[1][a]
            k_ = 1/poles[0][a]/poles[1][a]/a_*constants
            if a_.is_positive:
                result = k_*(1-exp(a_**2*t)*erfc(a_*sqrt(t)))
                _debug('     rule 5.3.5')
        elif (
                poles[0][n] == -1 and poles[0][m] == S.Half and
                poles[1][n] == -S.Half and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1/((a0*sqrt(s)+b0)*(sqrt(a1*s))
            # == 1/(a0*sqrt(a1)) * 1/((sqrt(s)+b0/a0)"sqrt(s))
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/(poles[0][a]*sqrt(poles[1][a]))*constants
            if a_.is_positive:
                result = k_*exp(a_**2*t)*erfc(a_*sqrt(t))
                _debug('     rule 5.3.7')
        elif (
                poles[0][n] == -S(3)/2 and poles[0][m] == 1 and
                poles[0][b] == 0 and poles[1][n] == -1 and
                poles[1][m] == S.Half):
            # 1/((a0**(3/2)*s**(3/2))*(a1*sqrt(s)+b1))
            # == 1/(a0**(3/2)*a1)  1/((s**(3/2))*(sqrt(s)+b1/a1))
            # Note that Bateman54 5.3 (8) is incorrect; there (sqrt(p)+a)
            # should be (sqrt(p)+a)**(-1).
            a_ = poles[1][b]/poles[1][a]
            k_ = 1/(poles[0][a]**(S(3)/2)*poles[1][a])/a_**2*constants
            if a_.is_positive:
                result = (
                    k_*(2/sqrt(pi)*a_*sqrt(t)+exp(a_**2*t)*erfc(a_*sqrt(t))-1))
                _debug('     rule 5.3.8')
        elif (
                poles[0][n] == -2 and poles[0][m] == S.Half and
                poles[1][n] == -1 and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1/((a0*sqrt(s)+b0)**2*a1*s)
            # == 1/a0**2/a1 * 1/(sqrt(s)+b0/a0)**2/s
            a_sq = poles[0][b]/poles[0][a]
            a_ = a_sq**2
            k_ = 1/poles[0][a]**2/poles[1][a]*constants
            if a_sq.is_positive:
                result = (
                    k_*(1/a_ + (2*t-1/a_)*exp(a_*t)*erfc(sqrt(a_)*sqrt(t)) -
                        2/sqrt(pi)/sqrt(a_)*sqrt(t)))
                _debug('     rule 5.3.11')
        elif (
                poles[0][n] == -2 and poles[0][m] == S.Half and
                poles[1][n] == -S.Half and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1/((a0*sqrt(s)+b0)**2*sqrt(a1*s))
            # == 1/a0**2/sqrt(a1) * 1/(sqrt(s)+b0/a0)**2/sqrt(s)
            a_ = poles[0][b]/poles[0][a]
            k_ = 1/poles[0][a]**2/sqrt(poles[1][a])*constants
            if a_.is_positive:
                result = (
                    k_*(2/sqrt(pi)*sqrt(t) -
                        2*a_*t*exp(a_**2*t)*erfc(a_*sqrt(t))))
                _debug('     rule 5.3.12')
        elif (
                poles[0][n] == -3 and poles[0][m] == S.Half and
                poles[1][n] == -S.Half and poles[1][m] == 1 and
                poles[1][b] == 0):
            # 1 / (sqrt(a1*s)*(a0*sqrt(s+b0)**3))
            # == 1/(sqrt(a1)*a0) * 1/(sqrt(s)*(sqrt(s+b0)**3))
            a_ = poles[0][b]
            k_ = constants/sqrt(poles[1][a])/poles[0][a]
            result = k_*(
                (2*a_**2*t+1)*t*exp(a_**2*t)*erfc(a_*sqrt(t)) -
                2/sqrt(pi)*a_*t**(S(3)/2))
            _debug('     rule 5.3.15')
        elif (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[1][n] == -S.Half and poles[1][m] == 1):
            # 1 / ( (a0*s+b0)* sqrt(a1*s+b1) )
            # == 1/(sqrt(a1)*a0) * 1 / ( (s+b0/a0)* sqrt(s+b1/a1) )
            a_ = poles[0][b]/poles[0][a]
            b_ = poles[1][b]/poles[1][a]
            k_ = constants/sqrt(poles[1][a])/poles[0][a]
            result = k_*(
                1/sqrt(b_-a_)*exp(-a_*t)*erf(sqrt(b_-a_)*sqrt(t)))
            _debug('     rule 5.3.23')

    elif len(poles) == 2 and len(zeros) == 1:
        if (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[1][n] == -1 and poles[1][m] == S.Half and
                zeros[0][n] == S.Half and zeros[0][m] == 1 and
                zeros[0][b] == 0):
            # sqrt(za0*s)/((a0*s+b0)*(a1*sqrt(s)+b1))
            # == sqrt(za0)/(a0*a1) * s/((s+b0/a0)*(sqrt(s)+b1/a1))
            a_sq = poles[1][b]/poles[1][a]
            a_ = a_sq**2
            b_ = -poles[0][b]/poles[0][a]
            k_ = sqrt(zeros[0][a])/poles[0][a]/poles[1][a]/(a_-b_)*constants
            if a_sq.is_positive and b_.is_positive:
                result = k_*(
                    a_*exp(a_*t)*erfc(sqrt(a_)*sqrt(t)) +
                    sqrt(a_)*sqrt(b_)*exp(b_*t)*erfc(sqrt(b_)*sqrt(t)) -
                    b_*exp(b_*t))
                _debug('     rule 5.3.6')
        elif (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[0][b] == 0 and poles[1][n] == -1 and
                poles[1][m] == S.Half and zeros[0][n] == 1 and
                zeros[0][m] == S.Half):
            # (az*sqrt(s)+bz)/(a0*s*(a1*sqrt(s)+b1))
            # == az/a0/a1 * (sqrt(z)+bz/az)/(s*(sqrt(s)+b1/a1))
            a_num = zeros[0][b]/zeros[0][a]
            a_ = poles[1][b]/poles[1][a]
            if a_+a_num == 0:
                k_ = zeros[0][a]/poles[0][a]/poles[1][a]*constants
                result = k_*(
                    2*exp(a_**2*t)*erfc(a_*sqrt(t))-1)
                _debug('     rule 5.3.17')
        elif (
                poles[1][n] == -1 and poles[1][m] == 1 and
                poles[1][b] == 0 and poles[0][n] == -2 and
                poles[0][m] == S.Half and zeros[0][n] == 2 and
                zeros[0][m] == S.Half):
            # (az*sqrt(s)+bz)**2/(a1*s*(a0*sqrt(s)+b0)**2)
            # == az**2/a1/a0**2 * (sqrt(z)+bz/az)**2/(s*(sqrt(s)+b0/a0)**2)
            a_num = zeros[0][b]/zeros[0][a]
            a_ = poles[0][b]/poles[0][a]
            if a_+a_num == 0:
                k_ = zeros[0][a]**2/poles[1][a]/poles[0][a]**2*constants
                result = k_*(
                    1 + 8*a_**2*t*exp(a_**2*t)*erfc(a_*sqrt(t)) -
                    8/sqrt(pi)*a_*sqrt(t))
                _debug('     rule 5.3.18')
        elif (
                poles[1][n] == -1 and poles[1][m] == 1 and
                poles[1][b] == 0 and poles[0][n] == -3 and
                poles[0][m] == S.Half and zeros[0][n] == 3 and
                zeros[0][m] == S.Half):
            # (az*sqrt(s)+bz)**3/(a1*s*(a0*sqrt(s)+b0)**3)
            # == az**3/a1/a0**3 * (sqrt(z)+bz/az)**3/(s*(sqrt(s)+b0/a0)**3)
            a_num = zeros[0][b]/zeros[0][a]
            a_ = poles[0][b]/poles[0][a]
            if a_+a_num == 0:
                k_ = zeros[0][a]**3/poles[1][a]/poles[0][a]**3*constants
                result = k_*(
                    2*(8*a_**4*t**2+8*a_**2*t+1)*exp(a_**2*t) *
                    erfc(a_*sqrt(t))-8/sqrt(pi)*a_*sqrt(t)*(2*a_**2*t+1)-1)
                _debug('     rule 5.3.19')

    elif len(poles) == 3 and len(zeros) == 0:
        if (
                poles[0][n] == -1 and poles[0][b] == 0 and poles[0][m] == 1 and
                poles[1][n] == -1 and poles[1][m] == 1 and
                poles[2][n] == -S.Half and poles[2][m] == 1):
            # 1/((a0*s)*(a1*s+b1)*sqrt(a2*s))
            # == 1/(a0*a1*sqrt(a2)) * 1/((s)*(s+b1/a1)*sqrt(s))
            a_ = -poles[1][b]/poles[1][a]
            k_ = 1/poles[0][a]/poles[1][a]/sqrt(poles[2][a])*constants
            if a_.is_positive:
                result = k_ * (
                    a_**(-S(3)/2) * exp(a_*t) * erf(sqrt(a_)*sqrt(t)) -
                    2/a_/sqrt(pi)*sqrt(t))
                _debug('     rule 5.3.2')
        elif (
                poles[0][n] == -1 and poles[0][m] == 1 and
                poles[1][n] == -1 and poles[1][m] == S.Half and
                poles[2][n] == -S.Half and poles[2][m] == 1 and
                poles[2][b] == 0):
            # 1/((a0*s+b0)*(a1*sqrt(s)+b1)*(sqrt(a2)*sqrt(s)))
            # == 1/(a0*a1*sqrt(a2)) * 1/((s+b0/a0)*(sqrt(s)+b1/a1)*sqrt(s))
            a_sq = poles[1][b]/poles[1][a]
            a_ = a_sq**2
            b_ = -poles[0][b]/poles[0][a]
            k_ = (
                1/poles[0][a]/poles[1][a]/sqrt(poles[2][a]) /
                (sqrt(b_)*(a_-b_)))
            if a_sq.is_positive and b_.is_positive:
                result = k_ * (
                    sqrt(b_)*exp(a_*t)*erfc(sqrt(a_)*sqrt(t)) +
                    sqrt(a_)*exp(b_*t)*erf(sqrt(b_)*sqrt(t)) -
                    sqrt(b_)*exp(b_*t))
                _debug('     rule 5.3.9')

    if result is None:
        return None
    else:
        return Heaviside(t)*result, condition


@DEBUG_WRAP
def _inverse_laplace_early_prog_rules(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    prog_rules = [_inverse_laplace_irrational]

    for p_rule in prog_rules:
        if (r := p_rule(F, s, t, plane)) is not None:
            return r
    return None


@DEBUG_WRAP
def _inverse_laplace_apply_prog_rules(F, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    prog_rules = [_inverse_laplace_time_shift, _inverse_laplace_freq_shift,
                  _inverse_laplace_time_diff, _inverse_laplace_diff,
                  _inverse_laplace_irrational]

    for p_rule in prog_rules:
        if (r := p_rule(F, s, t, plane)) is not None:
            return r
    return None


@DEBUG_WRAP
def _inverse_laplace_expand(fn, s, t, plane):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    if fn.is_Add:
        return None
    r = expand(fn, deep=False)
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    r = expand_mul(fn)
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    r = expand(fn)
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    if fn.is_rational_function(s):
        r = fn.apart(s).doit()
    if r.is_Add:
        return _inverse_laplace_transform(
            r, s, t, plane, simplify=False, dorational=True)
    return None


@DEBUG_WRAP
def _inverse_laplace_rational(fn, s, t, plane, *, simplify):
    """
    Helper function for the class InverseLaplaceTransform.
    """
    x_ = symbols('x_')
    f = fn.apart(s)
    terms = Add.make_args(f)
    terms_t = []
    conditions = [S.true]
    for term in terms:
        [n, d] = term.as_numer_denom()
        dc = d.as_poly(s).all_coeffs()
        dc_lead = dc[0]
        dc = [x/dc_lead for x in dc]
        nc = [x/dc_lead for x in n.as_poly(s).all_coeffs()]
        if len(dc) == 1:
            N = len(nc)-1
            for c in enumerate(nc):
                r = c[1]*DiracDelta(t, N-c[0])
                terms_t.append(r)
        elif len(dc) == 2:
            r = nc[0]*exp(-dc[1]*t)
            terms_t.append(Heaviside(t)*r)
        elif len(dc) == 3:
            a = dc[1]/2
            b = (dc[2]-a**2).factor()
            if len(nc) == 1:
                nc = [S.Zero] + nc
            l, m = tuple(nc)
            if b == 0:
                r = (m*t+l*(1-a*t))*exp(-a*t)
            else:
                hyp = False
                if b.is_negative:
                    b = -b
                    hyp = True
                b2 = list(roots(x_**2-b, x_).keys())[0]
                bs = sqrt(b).simplify()
                if hyp:
                    r = (
                        l*exp(-a*t)*cosh(b2*t) + (m-a*l) /
                        bs*exp(-a*t)*sinh(bs*t))
                else:
                    r = l*exp(-a*t)*cos(b2*t) + (m-a*l)/bs*exp(-a*t)*sin(bs*t)
            terms_t.append(Heaviside(t)*r)
        else:
            ft, cond = _inverse_laplace_transform(
                term, s, t, plane, simplify=simplify, dorational=False)
            terms_t.append(ft)
            conditions.append(cond)

    result = Add(*terms_t)
    if simplify:
        result = result.simplify(doit=False)
    return result, And(*conditions)


@DEBUG_WRAP
def _inverse_laplace_transform(fn, s_, t_, plane, *, simplify, dorational):
    """
    Front-end function of the inverse Laplace transform. It tries to apply all
    known rules recursively.  If everything else fails, it tries to integrate.
    """
    terms = Add.make_args(fn)
    terms_t = []
    conditions = []

    for term in terms:
        if term.has(exp):
            # Simplify expressions with exp() such that time-shifted
            # expressions have negative exponents in the numerator instead of
            # positive exponents in the numerator and denominator; this is a
            # (necessary) trick. It will, for example, convert
            # (s**2*exp(2*s) + 4*exp(s) - 4)*exp(-2*s)/(s*(s**2 + 1)) into
            # (s**2 + 4*exp(-s) - 4*exp(-2*s))/(s*(s**2 + 1))
            term = term.subs(s_, -s_).together().subs(s_, -s_)
        k, f = term.as_independent(s_, as_Add=False)
        if (
                dorational and term.is_rational_function(s_) and
                (r := _inverse_laplace_rational(
                    f, s_, t_, plane, simplify=simplify))
                is not None or
                (r := _inverse_laplace_apply_simple_rules(f, s_, t_))
                is not None or
                (r := _inverse_laplace_early_prog_rules(f, s_, t_, plane))
                is not None or
                (r := _inverse_laplace_expand(f, s_, t_, plane))
                is not None or
                (r := _inverse_laplace_apply_prog_rules(f, s_, t_, plane))
                is not None):
            pass
        elif any(undef.has(s_) for undef in f.atoms(AppliedUndef)):
            # If there are undefined functions f(t) then integration is
            # unlikely to do anything useful so we skip it and given an
            # unevaluated LaplaceTransform.
            r = (InverseLaplaceTransform(f, s_, t_, plane), S.true)
        elif (
                r := _inverse_laplace_transform_integration(
                    f, s_, t_, plane, simplify=simplify)) is not None:
            pass
        else:
            r = (InverseLaplaceTransform(f, s_, t_, plane), S.true)
        (ri_, ci_) = r
        terms_t.append(k*ri_)
        conditions.append(ci_)

    result = Add(*terms_t)
    if simplify:
        result = result.simplify(doit=False)
    condition = And(*conditions)

    return result, condition


class InverseLaplaceTransform(IntegralTransform):
    """
    Class representing unevaluated inverse Laplace transforms.

    For usage of this class, see the :class:`IntegralTransform` docstring.

    For how to compute inverse Laplace transforms, see the
    :func:`inverse_laplace_transform` docstring.
    """

    _name = 'Inverse Laplace'
    _none_sentinel = Dummy('None')
    _c = Dummy('c')

    def __new__(cls, F, s, x, plane, **opts):
        if plane is None:
            plane = InverseLaplaceTransform._none_sentinel
        return IntegralTransform.__new__(cls, F, s, x, plane, **opts)

    @property
    def fundamental_plane(self):
        plane = self.args[3]
        if plane is InverseLaplaceTransform._none_sentinel:
            plane = None
        return plane

    def _compute_transform(self, F, s, t, **hints):
        return _inverse_laplace_transform_integration(
            F, s, t, self.fundamental_plane, **hints)

    def _as_integral(self, F, s, t):
        c = self.__class__._c
        return (
            Integral(exp(s*t)*F, (s, c - S.ImaginaryUnit*S.Infinity,
                                  c + S.ImaginaryUnit*S.Infinity)) /
            (2*S.Pi*S.ImaginaryUnit))

    def doit(self, **hints):
        """
        Try to evaluate the transform in closed form.

        Explanation
        ===========

        Standard hints are the following:
        - ``noconds``:  if True, do not return convergence conditions. The
        default setting is `True`.
        - ``simplify``: if True, it simplifies the final result. The
        default setting is `False`.
        """
        _noconds = hints.get('noconds', True)
        _simplify = hints.get('simplify', False)

        debugf('[ILT doit] (%s, %s, %s)', (self.function,
                                           self.function_variable,
                                           self.transform_variable))

        s_ = self.function_variable
        t_ = self.transform_variable
        fn = self.function
        plane = self.fundamental_plane

        r = _inverse_laplace_transform(
            fn, s_, t_, plane, simplify=_simplify, dorational=True)

        if _noconds:
            return r[0]
        else:
            return r


def inverse_laplace_transform(F, s, t, plane=None, **hints):
    r"""
    Compute the inverse Laplace transform of `F(s)`, defined as

    .. math ::
        f(t) = \frac{1}{2\pi i} \int_{c-i\infty}^{c+i\infty} e^{st}
        F(s) \mathrm{d}s,

    for `c` so large that `F(s)` has no singularites in the
    half-plane `\operatorname{Re}(s) > c-\epsilon`.

    Explanation
    ===========

    The plane can be specified by
    argument ``plane``, but will be inferred if passed as None.

    Under certain regularity conditions, this recovers `f(t)` from its
    Laplace Transform `F(s)`, for non-negative `t`, and vice
    versa.

    If the integral cannot be computed in closed form, this function returns
    an unevaluated :class:`InverseLaplaceTransform` object.

    Note that this function will always assume `t` to be real,
    regardless of the SymPy assumption on `t`.

    For a description of possible hints, refer to the docstring of
    :func:`sympy.integrals.transforms.IntegralTransform.doit`.

    Examples
    ========

    >>> from sympy import inverse_laplace_transform, exp, Symbol
    >>> from sympy.abc import s, t
    >>> a = Symbol('a', positive=True)
    >>> inverse_laplace_transform(exp(-a*s)/s, s, t)
    Heaviside(-a + t)

    See Also
    ========

    laplace_transform
    hankel_transform, inverse_hankel_transform
    """
    _noconds = hints.get('noconds', True)
    _simplify = hints.get('simplify', False)

    if isinstance(F, MatrixBase) and hasattr(F, 'applyfunc'):
        return F.applyfunc(
            lambda Fij: inverse_laplace_transform(Fij, s, t, plane, **hints))

    r, c = InverseLaplaceTransform(F, s, t, plane).doit(
        noconds=False, simplify=_simplify)

    if _noconds:
        return r
    else:
        return r, c


def _fast_inverse_laplace(e, s, t):
    """Fast inverse Laplace transform of rational function including RootSum"""
    a, b, n = symbols('a, b, n', cls=Wild, exclude=[s])

    def _ilt(e):
        if not e.has(s):
            return e
        elif e.is_Add:
            return _ilt_add(e)
        elif e.is_Mul:
            return _ilt_mul(e)
        elif e.is_Pow:
            return _ilt_pow(e)
        elif isinstance(e, RootSum):
            return _ilt_rootsum(e)
        else:
            raise NotImplementedError

    def _ilt_add(e):
        return e.func(*map(_ilt, e.args))

    def _ilt_mul(e):
        coeff, expr = e.as_independent(s)
        if expr.is_Mul:
            raise NotImplementedError
        return coeff * _ilt(expr)

    def _ilt_pow(e):
        match = e.match((a*s + b)**n)
        if match is not None:
            nm, am, bm = match[n], match[a], match[b]
            if nm.is_Integer and nm < 0:
                return t**(-nm-1)*exp(-(bm/am)*t)/(am**-nm*gamma(-nm))
            if nm == 1:
                return exp(-(bm/am)*t) / am
        raise NotImplementedError

    def _ilt_rootsum(e):
        expr = e.fun.expr
        [variable] = e.fun.variables
        return RootSum(e.poly, Lambda(variable, together(_ilt(expr))))

    return _ilt(e)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\internals\managers.py ===
from __future__ import annotations

from collections.abc import (
    Hashable,
    Sequence,
)
import itertools
from typing import (
    TYPE_CHECKING,
    Callable,
    Literal,
    cast,
)
import warnings

import numpy as np

from pandas._config import (
    using_copy_on_write,
    warn_copy_on_write,
)

from pandas._libs import (
    internals as libinternals,
    lib,
)
from pandas._libs.internals import (
    BlockPlacement,
    BlockValuesRefs,
)
from pandas._libs.tslibs import Timestamp
from pandas.errors import PerformanceWarning
from pandas.util._decorators import cache_readonly
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.cast import infer_dtype_from_scalar
from pandas.core.dtypes.common import (
    ensure_platform_int,
    is_1d_only_ea_dtype,
    is_list_like,
)
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    ExtensionDtype,
)
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)
from pandas.core.dtypes.missing import (
    array_equals,
    isna,
)

import pandas.core.algorithms as algos
from pandas.core.arrays import (
    ArrowExtensionArray,
    ArrowStringArray,
    DatetimeArray,
)
from pandas.core.arrays._mixins import NDArrayBackedExtensionArray
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
)
from pandas.core.indexers import maybe_convert_indices
from pandas.core.indexes.api import (
    Index,
    ensure_index,
)
from pandas.core.internals.base import (
    DataManager,
    SingleDataManager,
    ensure_np_dtype,
    interleaved_dtype,
)
from pandas.core.internals.blocks import (
    COW_WARNING_GENERAL_MSG,
    COW_WARNING_SETITEM_MSG,
    Block,
    NumpyBlock,
    ensure_block_shape,
    extend_blocks,
    get_block_type,
    maybe_coerce_values,
    new_block,
    new_block_2d,
)
from pandas.core.internals.ops import (
    blockwise_all,
    operate_blockwise,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        AxisInt,
        DtypeObj,
        QuantileInterpolation,
        Self,
        Shape,
        npt,
    )

    from pandas.api.extensions import ExtensionArray


class BaseBlockManager(DataManager):
    """
    Core internal data structure to implement DataFrame, Series, etc.

    Manage a bunch of labeled 2D mixed-type ndarrays. Essentially it's a
    lightweight blocked set of labeled data to be manipulated by the DataFrame
    public API class

    Attributes
    ----------
    shape
    ndim
    axes
    values
    items

    Methods
    -------
    set_axis(axis, new_labels)
    copy(deep=True)

    get_dtypes

    apply(func, axes, block_filter_fn)

    get_bool_data
    get_numeric_data

    get_slice(slice_like, axis)
    get(label)
    iget(loc)

    take(indexer, axis)
    reindex_axis(new_labels, axis)
    reindex_indexer(new_labels, indexer, axis)

    delete(label)
    insert(loc, label, value)
    set(label, value)

    Parameters
    ----------
    blocks: Sequence of Block
    axes: Sequence of Index
    verify_integrity: bool, default True

    Notes
    -----
    This is *not* a public API class
    """

    __slots__ = ()

    _blknos: npt.NDArray[np.intp]
    _blklocs: npt.NDArray[np.intp]
    blocks: tuple[Block, ...]
    axes: list[Index]

    @property
    def ndim(self) -> int:
        raise NotImplementedError

    _known_consolidated: bool
    _is_consolidated: bool

    def __init__(self, blocks, axes, verify_integrity: bool = True) -> None:
        raise NotImplementedError

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        raise NotImplementedError

    @property
    def blknos(self) -> npt.NDArray[np.intp]:
        """
        Suppose we want to find the array corresponding to our i'th column.

        blknos[i] identifies the block from self.blocks that contains this column.

        blklocs[i] identifies the column of interest within
        self.blocks[self.blknos[i]]
        """
        if self._blknos is None:
            # Note: these can be altered by other BlockManager methods.
            self._rebuild_blknos_and_blklocs()

        return self._blknos

    @property
    def blklocs(self) -> npt.NDArray[np.intp]:
        """
        See blknos.__doc__
        """
        if self._blklocs is None:
            # Note: these can be altered by other BlockManager methods.
            self._rebuild_blknos_and_blklocs()

        return self._blklocs

    def make_empty(self, axes=None) -> Self:
        """return an empty BlockManager with the items axis of len 0"""
        if axes is None:
            axes = [Index([])] + self.axes[1:]

        # preserve dtype if possible
        if self.ndim == 1:
            assert isinstance(self, SingleBlockManager)  # for mypy
            blk = self.blocks[0]
            arr = blk.values[:0]
            bp = BlockPlacement(slice(0, 0))
            nb = blk.make_block_same_class(arr, placement=bp)
            blocks = [nb]
        else:
            blocks = []
        return type(self).from_blocks(blocks, axes)

    def __nonzero__(self) -> bool:
        return True

    # Python3 compat
    __bool__ = __nonzero__

    def _normalize_axis(self, axis: AxisInt) -> int:
        # switch axis to follow BlockManager logic
        if self.ndim == 2:
            axis = 1 if axis == 0 else 0
        return axis

    def set_axis(self, axis: AxisInt, new_labels: Index) -> None:
        # Caller is responsible for ensuring we have an Index object.
        self._validate_set_axis(axis, new_labels)
        self.axes[axis] = new_labels

    @property
    def is_single_block(self) -> bool:
        # Assumes we are 2D; overridden by SingleBlockManager
        return len(self.blocks) == 1

    @property
    def items(self) -> Index:
        return self.axes[0]

    def _has_no_reference(self, i: int) -> bool:
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
        blkno = self.blknos[i]
        return self._has_no_reference_block(blkno)

    def _has_no_reference_block(self, blkno: int) -> bool:
        """
        Check for block `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the block has no references.
        """
        return not self.blocks[blkno].refs.has_reference()

    def add_references(self, mgr: BaseBlockManager) -> None:
        """
        Adds the references from one manager to another. We assume that both
        managers have the same block structure.
        """
        if len(self.blocks) != len(mgr.blocks):
            # If block structure changes, then we made a copy
            return
        for i, blk in enumerate(self.blocks):
            blk.refs = mgr.blocks[i].refs
            blk.refs.add_reference(blk)

    def references_same_values(self, mgr: BaseBlockManager, blkno: int) -> bool:
        """
        Checks if two blocks from two different block managers reference the
        same underlying values.
        """
        blk = self.blocks[blkno]
        return any(blk is ref() for ref in mgr.blocks[blkno].refs.referenced_blocks)

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        dtypes = np.array([blk.dtype for blk in self.blocks], dtype=object)
        return dtypes.take(self.blknos)

    @property
    def arrays(self) -> list[ArrayLike]:
        """
        Quick access to the backing arrays of the Blocks.

        Only for compatibility with ArrayManager for testing convenience.
        Not to be used in actual code, and return value is not the same as the
        ArrayManager method (list of 1D arrays vs iterator of 2D ndarrays / 1D EAs).

        Warning! The returned arrays don't handle Copy-on-Write, so this should
        be used with caution (only in read-mode).
        """
        return [blk.values for blk in self.blocks]

    def __repr__(self) -> str:
        output = type(self).__name__
        for i, ax in enumerate(self.axes):
            if i == 0:
                output += f"\nItems: {ax}"
            else:
                output += f"\nAxis {i}: {ax}"

        for block in self.blocks:
            output += f"\n{block}"
        return output

    def apply(
        self,
        f,
        align_keys: list[str] | None = None,
        **kwargs,
    ) -> Self:
        """
        Iterate over the blocks, collect and create a new BlockManager.

        Parameters
        ----------
        f : str or callable
            Name of the Block method to apply.
        align_keys: List[str] or None, default None
        **kwargs
            Keywords to pass to `f`

        Returns
        -------
        BlockManager
        """
        assert "filter" not in kwargs

        align_keys = align_keys or []
        result_blocks: list[Block] = []
        # fillna: Series/DataFrame is responsible for making sure value is aligned

        aligned_args = {k: kwargs[k] for k in align_keys}

        for b in self.blocks:
            if aligned_args:
                for k, obj in aligned_args.items():
                    if isinstance(obj, (ABCSeries, ABCDataFrame)):
                        # The caller is responsible for ensuring that
                        #  obj.axes[-1].equals(self.items)
                        if obj.ndim == 1:
                            kwargs[k] = obj.iloc[b.mgr_locs.indexer]._values
                        else:
                            kwargs[k] = obj.iloc[:, b.mgr_locs.indexer]._values
                    else:
                        # otherwise we have an ndarray
                        kwargs[k] = obj[b.mgr_locs.indexer]

            if callable(f):
                applied = b.apply(f, **kwargs)
            else:
                applied = getattr(b, f)(**kwargs)
            result_blocks = extend_blocks(applied, result_blocks)

        out = type(self).from_blocks(result_blocks, self.axes)
        return out

    # Alias so we can share code with ArrayManager
    apply_with_block = apply

    def setitem(self, indexer, value, warn: bool = True) -> Self:
        """
        Set values with indexer.

        For SingleBlockManager, this backs s[indexer] = value
        """
        if isinstance(indexer, np.ndarray) and indexer.ndim > self.ndim:
            raise ValueError(f"Cannot set values with ndim > {self.ndim}")

        if warn and warn_copy_on_write() and not self._has_no_reference(0):
            warnings.warn(
                COW_WARNING_GENERAL_MSG,
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        elif using_copy_on_write() and not self._has_no_reference(0):
            # this method is only called if there is a single block -> hardcoded 0
            # Split blocks to only copy the columns we want to modify
            if self.ndim == 2 and isinstance(indexer, tuple):
                blk_loc = self.blklocs[indexer[1]]
                if is_list_like(blk_loc) and blk_loc.ndim == 2:
                    blk_loc = np.squeeze(blk_loc, axis=0)
                elif not is_list_like(blk_loc):
                    # Keep dimension and copy data later
                    blk_loc = [blk_loc]  # type: ignore[assignment]
                if len(blk_loc) == 0:
                    return self.copy(deep=False)

                values = self.blocks[0].values
                if values.ndim == 2:
                    values = values[blk_loc]
                    # "T" has no attribute "_iset_split_block"
                    self._iset_split_block(  # type: ignore[attr-defined]
                        0, blk_loc, values
                    )
                    # first block equals values
                    self.blocks[0].setitem((indexer[0], np.arange(len(blk_loc))), value)
                    return self
            # No need to split if we either set all columns or on a single block
            # manager
            self = self.copy()

        return self.apply("setitem", indexer=indexer, value=value)

    def diff(self, n: int) -> Self:
        # only reached with self.ndim == 2
        return self.apply("diff", n=n)

    def astype(self, dtype, copy: bool | None = False, errors: str = "raise") -> Self:
        if copy is None:
            if using_copy_on_write():
                copy = False
            else:
                copy = True
        elif using_copy_on_write():
            copy = False

        return self.apply(
            "astype",
            dtype=dtype,
            copy=copy,
            errors=errors,
            using_cow=using_copy_on_write(),
        )

    def convert(self, copy: bool | None) -> Self:
        if copy is None:
            if using_copy_on_write():
                copy = False
            else:
                copy = True
        elif using_copy_on_write():
            copy = False

        return self.apply("convert", copy=copy, using_cow=using_copy_on_write())

    def convert_dtypes(self, **kwargs):
        if using_copy_on_write():
            copy = False
        else:
            copy = True

        return self.apply(
            "convert_dtypes", copy=copy, using_cow=using_copy_on_write(), **kwargs
        )

    def get_values_for_csv(
        self, *, float_format, date_format, decimal, na_rep: str = "nan", quoting=None
    ) -> Self:
        """
        Convert values to native types (strings / python objects) that are used
        in formatting (repr / csv).
        """
        return self.apply(
            "get_values_for_csv",
            na_rep=na_rep,
            quoting=quoting,
            float_format=float_format,
            date_format=date_format,
            decimal=decimal,
        )

    @property
    def any_extension_types(self) -> bool:
        """Whether any of the blocks in this manager are extension blocks"""
        return any(block.is_extension for block in self.blocks)

    @property
    def is_view(self) -> bool:
        """return a boolean if we are a single block and are a view"""
        if len(self.blocks) == 1:
            return self.blocks[0].is_view

        # It is technically possible to figure out which blocks are views
        # e.g. [ b.values.base is not None for b in self.blocks ]
        # but then we have the case of possibly some blocks being a view
        # and some blocks not. setting in theory is possible on the non-view
        # blocks w/o causing a SettingWithCopy raise/warn. But this is a bit
        # complicated

        return False

    def _get_data_subset(self, predicate: Callable) -> Self:
        blocks = [blk for blk in self.blocks if predicate(blk.values)]
        return self._combine(blocks)

    def get_bool_data(self) -> Self:
        """
        Select blocks that are bool-dtype and columns from object-dtype blocks
        that are all-bool.
        """

        new_blocks = []

        for blk in self.blocks:
            if blk.dtype == bool:
                new_blocks.append(blk)

            elif blk.is_object:
                nbs = blk._split()
                new_blocks.extend(nb for nb in nbs if nb.is_bool)

        return self._combine(new_blocks)

    def get_numeric_data(self) -> Self:
        numeric_blocks = [blk for blk in self.blocks if blk.is_numeric]
        if len(numeric_blocks) == len(self.blocks):
            # Avoid somewhat expensive _combine
            return self
        return self._combine(numeric_blocks)

    def _combine(self, blocks: list[Block], index: Index | None = None) -> Self:
        """return a new manager with the blocks"""
        if len(blocks) == 0:
            if self.ndim == 2:
                # retain our own Index dtype
                if index is not None:
                    axes = [self.items[:0], index]
                else:
                    axes = [self.items[:0]] + self.axes[1:]
                return self.make_empty(axes)
            return self.make_empty()

        # FIXME: optimization potential
        indexer = np.sort(np.concatenate([b.mgr_locs.as_array for b in blocks]))
        inv_indexer = lib.get_reverse_indexer(indexer, self.shape[0])

        new_blocks: list[Block] = []
        for b in blocks:
            nb = b.copy(deep=False)
            nb.mgr_locs = BlockPlacement(inv_indexer[nb.mgr_locs.indexer])
            new_blocks.append(nb)

        axes = list(self.axes)
        if index is not None:
            axes[-1] = index
        axes[0] = self.items.take(indexer)

        return type(self).from_blocks(new_blocks, axes)

    @property
    def nblocks(self) -> int:
        return len(self.blocks)

    def copy(self, deep: bool | None | Literal["all"] = True) -> Self:
        """
        Make deep or shallow copy of BlockManager

        Parameters
        ----------
        deep : bool, string or None, default True
            If False or None, return a shallow copy (do not copy data)
            If 'all', copy data and a deep copy of the index

        Returns
        -------
        BlockManager
        """
        if deep is None:
            if using_copy_on_write():
                # use shallow copy
                deep = False
            else:
                # preserve deep copy for BlockManager with copy=None
                deep = True

        # this preserves the notion of view copying of axes
        if deep:
            # hit in e.g. tests.io.json.test_pandas

            def copy_func(ax):
                return ax.copy(deep=True) if deep == "all" else ax.view()

            new_axes = [copy_func(ax) for ax in self.axes]
        else:
            if using_copy_on_write():
                new_axes = [ax.view() for ax in self.axes]
            else:
                new_axes = list(self.axes)

        res = self.apply("copy", deep=deep)
        res.axes = new_axes

        if self.ndim > 1:
            # Avoid needing to re-compute these
            blknos = self._blknos
            if blknos is not None:
                res._blknos = blknos.copy()
                res._blklocs = self._blklocs.copy()

        if deep:
            res._consolidate_inplace()
        return res

    def consolidate(self) -> Self:
        """
        Join together blocks having same dtype

        Returns
        -------
        y : BlockManager
        """
        if self.is_consolidated():
            return self

        bm = type(self)(self.blocks, self.axes, verify_integrity=False)
        bm._is_consolidated = False
        bm._consolidate_inplace()
        return bm

    def reindex_indexer(
        self,
        new_axis: Index,
        indexer: npt.NDArray[np.intp] | None,
        axis: AxisInt,
        fill_value=None,
        allow_dups: bool = False,
        copy: bool | None = True,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
    ) -> Self:
        """
        Parameters
        ----------
        new_axis : Index
        indexer : ndarray[intp] or None
        axis : int
        fill_value : object, default None
        allow_dups : bool, default False
        copy : bool or None, default True
            If None, regard as False to get shallow copy.
        only_slice : bool, default False
            Whether to take views, not copies, along columns.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.

        pandas-indexer with -1's only.
        """
        if copy is None:
            if using_copy_on_write():
                # use shallow copy
                copy = False
            else:
                # preserve deep copy for BlockManager with copy=None
                copy = True

        if indexer is None:
            if new_axis is self.axes[axis] and not copy:
                return self

            result = self.copy(deep=copy)
            result.axes = list(self.axes)
            result.axes[axis] = new_axis
            return result

        # Should be intp, but in some cases we get int64 on 32bit builds
        assert isinstance(indexer, np.ndarray)

        # some axes don't allow reindexing with dups
        if not allow_dups:
            self.axes[axis]._validate_can_reindex(indexer)

        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        if axis == 0:
            new_blocks = self._slice_take_blocks_ax0(
                indexer,
                fill_value=fill_value,
                only_slice=only_slice,
                use_na_proxy=use_na_proxy,
            )
        else:
            new_blocks = [
                blk.take_nd(
                    indexer,
                    axis=1,
                    fill_value=(
                        fill_value if fill_value is not None else blk.fill_value
                    ),
                )
                for blk in self.blocks
            ]

        new_axes = list(self.axes)
        new_axes[axis] = new_axis

        new_mgr = type(self).from_blocks(new_blocks, new_axes)
        if axis == 1:
            # We can avoid the need to rebuild these
            new_mgr._blknos = self.blknos.copy()
            new_mgr._blklocs = self.blklocs.copy()
        return new_mgr

    def _slice_take_blocks_ax0(
        self,
        slice_or_indexer: slice | np.ndarray,
        fill_value=lib.no_default,
        only_slice: bool = False,
        *,
        use_na_proxy: bool = False,
        ref_inplace_op: bool = False,
    ) -> list[Block]:
        """
        Slice/take blocks along axis=0.

        Overloaded for SingleBlock

        Parameters
        ----------
        slice_or_indexer : slice or np.ndarray[int64]
        fill_value : scalar, default lib.no_default
        only_slice : bool, default False
            If True, we always return views on existing arrays, never copies.
            This is used when called from ops.blockwise.operate_blockwise.
        use_na_proxy : bool, default False
            Whether to use a np.void ndarray for newly introduced columns.
        ref_inplace_op: bool, default False
            Don't track refs if True because we operate inplace

        Returns
        -------
        new_blocks : list of Block
        """
        allow_fill = fill_value is not lib.no_default

        sl_type, slobj, sllen = _preprocess_slice_or_indexer(
            slice_or_indexer, self.shape[0], allow_fill=allow_fill
        )

        if self.is_single_block:
            blk = self.blocks[0]

            if sl_type == "slice":
                # GH#32959 EABlock would fail since we can't make 0-width
                # TODO(EA2D): special casing unnecessary with 2D EAs
                if sllen == 0:
                    return []
                bp = BlockPlacement(slice(0, sllen))
                return [blk.getitem_block_columns(slobj, new_mgr_locs=bp)]
            elif not allow_fill or self.ndim == 1:
                if allow_fill and fill_value is None:
                    fill_value = blk.fill_value

                if not allow_fill and only_slice:
                    # GH#33597 slice instead of take, so we get
                    #  views instead of copies
                    blocks = [
                        blk.getitem_block_columns(
                            slice(ml, ml + 1),
                            new_mgr_locs=BlockPlacement(i),
                            ref_inplace_op=ref_inplace_op,
                        )
                        for i, ml in enumerate(slobj)
                    ]
                    return blocks
                else:
                    bp = BlockPlacement(slice(0, sllen))
                    return [
                        blk.take_nd(
                            slobj,
                            axis=0,
                            new_mgr_locs=bp,
                            fill_value=fill_value,
                        )
                    ]

        if sl_type == "slice":
            blknos = self.blknos[slobj]
            blklocs = self.blklocs[slobj]
        else:
            blknos = algos.take_nd(
                self.blknos, slobj, fill_value=-1, allow_fill=allow_fill
            )
            blklocs = algos.take_nd(
                self.blklocs, slobj, fill_value=-1, allow_fill=allow_fill
            )

        # When filling blknos, make sure blknos is updated before appending to
        # blocks list, that way new blkno is exactly len(blocks).
        blocks = []
        group = not only_slice
        for blkno, mgr_locs in libinternals.get_blkno_placements(blknos, group=group):
            if blkno == -1:
                # If we've got here, fill_value was not lib.no_default

                blocks.append(
                    self._make_na_block(
                        placement=mgr_locs,
                        fill_value=fill_value,
                        use_na_proxy=use_na_proxy,
                    )
                )
            else:
                blk = self.blocks[blkno]

                # Otherwise, slicing along items axis is necessary.
                if not blk._can_consolidate and not blk._validate_ndim:
                    # i.e. we dont go through here for DatetimeTZBlock
                    # A non-consolidatable block, it's easy, because there's
                    # only one item and each mgr loc is a copy of that single
                    # item.
                    deep = not (only_slice or using_copy_on_write())
                    for mgr_loc in mgr_locs:
                        newblk = blk.copy(deep=deep)
                        newblk.mgr_locs = BlockPlacement(slice(mgr_loc, mgr_loc + 1))
                        blocks.append(newblk)

                else:
                    # GH#32779 to avoid the performance penalty of copying,
                    #  we may try to only slice
                    taker = blklocs[mgr_locs.indexer]
                    max_len = max(len(mgr_locs), taker.max() + 1)
                    if only_slice or using_copy_on_write():
                        taker = lib.maybe_indices_to_slice(taker, max_len)

                    if isinstance(taker, slice):
                        nb = blk.getitem_block_columns(taker, new_mgr_locs=mgr_locs)
                        blocks.append(nb)
                    elif only_slice:
                        # GH#33597 slice instead of take, so we get
                        #  views instead of copies
                        for i, ml in zip(taker, mgr_locs):
                            slc = slice(i, i + 1)
                            bp = BlockPlacement(ml)
                            nb = blk.getitem_block_columns(slc, new_mgr_locs=bp)
                            # We have np.shares_memory(nb.values, blk.values)
                            blocks.append(nb)
                    else:
                        nb = blk.take_nd(taker, axis=0, new_mgr_locs=mgr_locs)
                        blocks.append(nb)

        return blocks

    def _make_na_block(
        self, placement: BlockPlacement, fill_value=None, use_na_proxy: bool = False
    ) -> Block:
        # Note: we only get here with self.ndim == 2

        if use_na_proxy:
            assert fill_value is None
            shape = (len(placement), self.shape[1])
            vals = np.empty(shape, dtype=np.void)
            nb = NumpyBlock(vals, placement, ndim=2)
            return nb

        if fill_value is None:
            fill_value = np.nan

        shape = (len(placement), self.shape[1])

        dtype, fill_value = infer_dtype_from_scalar(fill_value)
        block_values = make_na_array(dtype, shape, fill_value)
        return new_block_2d(block_values, placement=placement)

    def take(
        self,
        indexer: npt.NDArray[np.intp],
        axis: AxisInt = 1,
        verify: bool = True,
    ) -> Self:
        """
        Take items along any axis.

        indexer : np.ndarray[np.intp]
        axis : int, default 1
        verify : bool, default True
            Check that all entries are between 0 and len(self) - 1, inclusive.
            Pass verify=False if this check has been done by the caller.

        Returns
        -------
        BlockManager
        """
        # Caller is responsible for ensuring indexer annotation is accurate

        n = self.shape[axis]
        indexer = maybe_convert_indices(indexer, n, verify=verify)

        new_labels = self.axes[axis].take(indexer)
        return self.reindex_indexer(
            new_axis=new_labels,
            indexer=indexer,
            axis=axis,
            allow_dups=True,
            copy=None,
        )


class BlockManager(libinternals.BlockManager, BaseBlockManager):
    """
    BaseBlockManager that holds 2D blocks.
    """

    ndim = 2

    # ----------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        blocks: Sequence[Block],
        axes: Sequence[Index],
        verify_integrity: bool = True,
    ) -> None:
        if verify_integrity:
            # Assertion disabled for performance
            # assert all(isinstance(x, Index) for x in axes)

            for block in blocks:
                if self.ndim != block.ndim:
                    raise AssertionError(
                        f"Number of Block dimensions ({block.ndim}) must equal "
                        f"number of axes ({self.ndim})"
                    )
                # As of 2.0, the caller is responsible for ensuring that
                #  DatetimeTZBlock with block.ndim == 2 has block.values.ndim ==2;
                #  previously there was a special check for fastparquet compat.

            self._verify_integrity()

    def _verify_integrity(self) -> None:
        mgr_shape = self.shape
        tot_items = sum(len(x.mgr_locs) for x in self.blocks)
        for block in self.blocks:
            if block.shape[1:] != mgr_shape[1:]:
                raise_construction_error(tot_items, block.shape[1:], self.axes)
        if len(self.items) != tot_items:
            raise AssertionError(
                "Number of manager items must equal union of "
                f"block items\n# manager items: {len(self.items)}, # "
                f"tot_items: {tot_items}"
            )

    @classmethod
    def from_blocks(cls, blocks: list[Block], axes: list[Index]) -> Self:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        return cls(blocks, axes, verify_integrity=False)

    # ----------------------------------------------------------------
    # Indexing

    def fast_xs(self, loc: int) -> SingleBlockManager:
        """
        Return the array corresponding to `frame.iloc[loc]`.

        Parameters
        ----------
        loc : int

        Returns
        -------
        np.ndarray or ExtensionArray
        """
        if len(self.blocks) == 1:
            # TODO: this could be wrong if blk.mgr_locs is not slice(None)-like;
            #  is this ruled out in the general case?
            result = self.blocks[0].iget((slice(None), loc))
            # in the case of a single block, the new block is a view
            bp = BlockPlacement(slice(0, len(result)))
            block = new_block(
                result,
                placement=bp,
                ndim=1,
                refs=self.blocks[0].refs,
            )
            return SingleBlockManager(block, self.axes[0])

        dtype = interleaved_dtype([blk.dtype for blk in self.blocks])

        n = len(self)

        if isinstance(dtype, ExtensionDtype):
            # TODO: use object dtype as workaround for non-performant
            #  EA.__setitem__ methods. (primarily ArrowExtensionArray.__setitem__
            #  when iteratively setting individual values)
            #  https://github.com/pandas-dev/pandas/pull/54508#issuecomment-1675827918
            result = np.empty(n, dtype=object)
        else:
            result = np.empty(n, dtype=dtype)
            result = ensure_wrapped_if_datetimelike(result)

        for blk in self.blocks:
            # Such assignment may incorrectly coerce NaT to None
            # result[blk.mgr_locs] = blk._slice((slice(None), loc))
            for i, rl in enumerate(blk.mgr_locs):
                result[rl] = blk.iget((i, loc))

        if isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            result = cls._from_sequence(result, dtype=dtype)

        bp = BlockPlacement(slice(0, len(result)))
        block = new_block(result, placement=bp, ndim=1)
        return SingleBlockManager(block, self.axes[0])

    def iget(self, i: int, track_ref: bool = True) -> SingleBlockManager:
        """
        Return the data as a SingleBlockManager.
        """
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])

        # shortcut for select a single-dim from a 2-dim BM
        bp = BlockPlacement(slice(0, len(values)))
        nb = type(block)(
            values, placement=bp, ndim=1, refs=block.refs if track_ref else None
        )
        return SingleBlockManager(nb, self.axes[1])

    def iget_values(self, i: int) -> ArrayLike:
        """
        Return the data for column i as the values (ndarray or ExtensionArray).

        Warning! The returned array is a view but doesn't handle Copy-on-Write,
        so this should be used with caution.
        """
        # TODO(CoW) making the arrays read-only might make this safer to use?
        block = self.blocks[self.blknos[i]]
        values = block.iget(self.blklocs[i])
        return values

    @property
    def column_arrays(self) -> list[np.ndarray]:
        """
        Used in the JSON C code to access column arrays.
        This optimizes compared to using `iget_values` by converting each

        Warning! This doesn't handle Copy-on-Write, so should be used with
        caution (current use case of consuming this in the JSON code is fine).
        """
        # This is an optimized equivalent to
        #  result = [self.iget_values(i) for i in range(len(self.items))]
        result: list[np.ndarray | None] = [None] * len(self.items)

        for blk in self.blocks:
            mgr_locs = blk._mgr_locs
            values = blk.array_values._values_for_json()
            if values.ndim == 1:
                # TODO(EA2D): special casing not needed with 2D EAs
                result[mgr_locs[0]] = values

            else:
                for i, loc in enumerate(mgr_locs):
                    result[loc] = values[i]

        # error: Incompatible return value type (got "List[None]",
        # expected "List[ndarray[Any, Any]]")
        return result  # type: ignore[return-value]

    def iset(
        self,
        loc: int | slice | np.ndarray,
        value: ArrayLike,
        inplace: bool = False,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        """
        Set new item in-place. Does not consolidate. Adds new Block if not
        contained in the current set of items
        """

        # FIXME: refactor, clearly separate broadcasting & zip-like assignment
        #        can prob also fix the various if tests for sparse/categorical
        if self._blklocs is None and self.ndim > 1:
            self._rebuild_blknos_and_blklocs()

        # Note: we exclude DTA/TDA here
        value_is_extension_type = is_1d_only_ea_dtype(value.dtype)
        if not value_is_extension_type:
            if value.ndim == 2:
                value = value.T
            else:
                value = ensure_block_shape(value, ndim=2)

            if value.shape[1:] != self.shape[1:]:
                raise AssertionError(
                    "Shape of new values must be compatible with manager shape"
                )

        if lib.is_integer(loc):
            # We have 6 tests where loc is _not_ an int.
            # In this case, get_blkno_placements will yield only one tuple,
            #  containing (self._blknos[loc], BlockPlacement(slice(0, 1, 1)))

            # Check if we can use _iset_single fastpath
            loc = cast(int, loc)
            blkno = self.blknos[loc]
            blk = self.blocks[blkno]
            if len(blk._mgr_locs) == 1:  # TODO: fastest way to check this?
                return self._iset_single(
                    loc,
                    value,
                    inplace=inplace,
                    blkno=blkno,
                    blk=blk,
                    refs=refs,
                )

            # error: Incompatible types in assignment (expression has type
            # "List[Union[int, slice, ndarray]]", variable has type "Union[int,
            # slice, ndarray]")
            loc = [loc]  # type: ignore[assignment]

        # categorical/sparse/datetimetz
        if value_is_extension_type:

            def value_getitem(placement):
                return value

        else:

            def value_getitem(placement):
                return value[placement.indexer]

        # Accessing public blknos ensures the public versions are initialized
        blknos = self.blknos[loc]
        blklocs = self.blklocs[loc].copy()

        unfit_mgr_locs = []
        unfit_val_locs = []
        removed_blknos = []
        for blkno_l, val_locs in libinternals.get_blkno_placements(blknos, group=True):
            blk = self.blocks[blkno_l]
            blk_locs = blklocs[val_locs.indexer]
            if inplace and blk.should_store(value):
                # Updating inplace -> check if we need to do Copy-on-Write
                if using_copy_on_write() and not self._has_no_reference_block(blkno_l):
                    self._iset_split_block(
                        blkno_l, blk_locs, value_getitem(val_locs), refs=refs
                    )
                else:
                    blk.set_inplace(blk_locs, value_getitem(val_locs))
                    continue
            else:
                unfit_mgr_locs.append(blk.mgr_locs.as_array[blk_locs])
                unfit_val_locs.append(val_locs)

                # If all block items are unfit, schedule the block for removal.
                if len(val_locs) == len(blk.mgr_locs):
                    removed_blknos.append(blkno_l)
                    continue
                else:
                    # Defer setting the new values to enable consolidation
                    self._iset_split_block(blkno_l, blk_locs, refs=refs)

        if len(removed_blknos):
            # Remove blocks & update blknos accordingly
            is_deleted = np.zeros(self.nblocks, dtype=np.bool_)
            is_deleted[removed_blknos] = True

            new_blknos = np.empty(self.nblocks, dtype=np.intp)
            new_blknos.fill(-1)
            new_blknos[~is_deleted] = np.arange(self.nblocks - len(removed_blknos))
            self._blknos = new_blknos[self._blknos]
            self.blocks = tuple(
                blk for i, blk in enumerate(self.blocks) if i not in set(removed_blknos)
            )

        if unfit_val_locs:
            unfit_idxr = np.concatenate(unfit_mgr_locs)
            unfit_count = len(unfit_idxr)

            new_blocks: list[Block] = []
            if value_is_extension_type:
                # This code (ab-)uses the fact that EA blocks contain only
                # one item.
                # TODO(EA2D): special casing unnecessary with 2D EAs
                new_blocks.extend(
                    new_block_2d(
                        values=value,
                        placement=BlockPlacement(slice(mgr_loc, mgr_loc + 1)),
                        refs=refs,
                    )
                    for mgr_loc in unfit_idxr
                )

                self._blknos[unfit_idxr] = np.arange(unfit_count) + len(self.blocks)
                self._blklocs[unfit_idxr] = 0

            else:
                # unfit_val_locs contains BlockPlacement objects
                unfit_val_items = unfit_val_locs[0].append(unfit_val_locs[1:])

                new_blocks.append(
                    new_block_2d(
                        values=value_getitem(unfit_val_items),
                        placement=BlockPlacement(unfit_idxr),
                        refs=refs,
                    )
                )

                self._blknos[unfit_idxr] = len(self.blocks)
                self._blklocs[unfit_idxr] = np.arange(unfit_count)

            self.blocks += tuple(new_blocks)

            # Newly created block's dtype may already be present.
            self._known_consolidated = False

    def _iset_split_block(
        self,
        blkno_l: int,
        blk_locs: np.ndarray | list[int],
        value: ArrayLike | None = None,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        """Removes columns from a block by splitting the block.

        Avoids copying the whole block through slicing and updates the manager
        after determinint the new block structure. Optionally adds a new block,
        otherwise has to be done by the caller.

        Parameters
        ----------
        blkno_l: The block number to operate on, relevant for updating the manager
        blk_locs: The locations of our block that should be deleted.
        value: The value to set as a replacement.
        refs: The reference tracking object of the value to set.
        """
        blk = self.blocks[blkno_l]

        if self._blklocs is None:
            self._rebuild_blknos_and_blklocs()

        nbs_tup = tuple(blk.delete(blk_locs))
        if value is not None:
            locs = blk.mgr_locs.as_array[blk_locs]
            first_nb = new_block_2d(value, BlockPlacement(locs), refs=refs)
        else:
            first_nb = nbs_tup[0]
            nbs_tup = tuple(nbs_tup[1:])

        nr_blocks = len(self.blocks)
        blocks_tup = (
            self.blocks[:blkno_l] + (first_nb,) + self.blocks[blkno_l + 1 :] + nbs_tup
        )
        self.blocks = blocks_tup

        if not nbs_tup and value is not None:
            # No need to update anything if split did not happen
            return

        self._blklocs[first_nb.mgr_locs.indexer] = np.arange(len(first_nb))

        for i, nb in enumerate(nbs_tup):
            self._blklocs[nb.mgr_locs.indexer] = np.arange(len(nb))
            self._blknos[nb.mgr_locs.indexer] = i + nr_blocks

    def _iset_single(
        self,
        loc: int,
        value: ArrayLike,
        inplace: bool,
        blkno: int,
        blk: Block,
        refs: BlockValuesRefs | None = None,
    ) -> None:
        """
        Fastpath for iset when we are only setting a single position and
        the Block currently in that position is itself single-column.

        In this case we can swap out the entire Block and blklocs and blknos
        are unaffected.
        """
        # Caller is responsible for verifying value.shape

        if inplace and blk.should_store(value):
            copy = False
            if using_copy_on_write() and not self._has_no_reference_block(blkno):
                # perform Copy-on-Write and clear the reference
                copy = True
            iloc = self.blklocs[loc]
            blk.set_inplace(slice(iloc, iloc + 1), value, copy=copy)
            return

        nb = new_block_2d(value, placement=blk._mgr_locs, refs=refs)
        old_blocks = self.blocks
        new_blocks = old_blocks[:blkno] + (nb,) + old_blocks[blkno + 1 :]
        self.blocks = new_blocks
        return

    def column_setitem(
        self, loc: int, idx: int | slice | np.ndarray, value, inplace_only: bool = False
    ) -> None:
        """
        Set values ("setitem") into a single column (not setting the full column).

        This is a method on the BlockManager level, to avoid creating an
        intermediate Series at the DataFrame level (`s = df[loc]; s[idx] = value`)
        """
        needs_to_warn = False
        if warn_copy_on_write() and not self._has_no_reference(loc):
            if not isinstance(
                self.blocks[self.blknos[loc]].values,
                (ArrowExtensionArray, ArrowStringArray),
            ):
                # We might raise if we are in an expansion case, so defer
                # warning till we actually updated
                needs_to_warn = True

        elif using_copy_on_write() and not self._has_no_reference(loc):
            blkno = self.blknos[loc]
            # Split blocks to only copy the column we want to modify
            blk_loc = self.blklocs[loc]
            # Copy our values
            values = self.blocks[blkno].values
            if values.ndim == 1:
                values = values.copy()
            else:
                # Use [blk_loc] as indexer to keep ndim=2, this already results in a
                # copy
                values = values[[blk_loc]]
            self._iset_split_block(blkno, [blk_loc], values)

        # this manager is only created temporarily to mutate the values in place
        # so don't track references, otherwise the `setitem` would perform CoW again
        col_mgr = self.iget(loc, track_ref=False)
        if inplace_only:
            col_mgr.setitem_inplace(idx, value)
        else:
            new_mgr = col_mgr.setitem((idx,), value)
            self.iset(loc, new_mgr._block.values, inplace=True)

        if needs_to_warn:
            warnings.warn(
                COW_WARNING_GENERAL_MSG,
                FutureWarning,
                stacklevel=find_stack_level(),
            )

    def insert(self, loc: int, item: Hashable, value: ArrayLike, refs=None) -> None:
        """
        Insert item at selected position.

        Parameters
        ----------
        loc : int
        item : hashable
        value : np.ndarray or ExtensionArray
        refs : The reference tracking object of the value to set.
        """
        with warnings.catch_warnings():
            # TODO: re-issue this with setitem-specific message?
            warnings.filterwarnings(
                "ignore",
                "The behavior of Index.insert with object-dtype is deprecated",
                category=FutureWarning,
            )
            new_axis = self.items.insert(loc, item)

        if value.ndim == 2:
            value = value.T
            if len(value) > 1:
                raise ValueError(
                    f"Expected a 1D array, got an array with shape {value.T.shape}"
                )
        else:
            value = ensure_block_shape(value, ndim=self.ndim)

        bp = BlockPlacement(slice(loc, loc + 1))
        block = new_block_2d(values=value, placement=bp, refs=refs)

        if not len(self.blocks):
            # Fastpath
            self._blklocs = np.array([0], dtype=np.intp)
            self._blknos = np.array([0], dtype=np.intp)
        else:
            self._insert_update_mgr_locs(loc)
            self._insert_update_blklocs_and_blknos(loc)

        self.axes[0] = new_axis
        self.blocks += (block,)

        self._known_consolidated = False

        if sum(not block.is_extension for block in self.blocks) > 100:
            warnings.warn(
                "DataFrame is highly fragmented.  This is usually the result "
                "of calling `frame.insert` many times, which has poor performance.  "
                "Consider joining all columns at once using pd.concat(axis=1) "
                "instead. To get a de-fragmented frame, use `newframe = frame.copy()`",
                PerformanceWarning,
                stacklevel=find_stack_level(),
            )

    def _insert_update_mgr_locs(self, loc) -> None:
        """
        When inserting a new Block at location 'loc', we increment
        all of the mgr_locs of blocks above that by one.
        """
        for blkno, count in _fast_count_smallints(self.blknos[loc:]):
            # .620 this way, .326 of which is in increment_above
            blk = self.blocks[blkno]
            blk._mgr_locs = blk._mgr_locs.increment_above(loc)

    def _insert_update_blklocs_and_blknos(self, loc) -> None:
        """
        When inserting a new Block at location 'loc', we update our
        _blklocs and _blknos.
        """

        # Accessing public blklocs ensures the public versions are initialized
        if loc == self.blklocs.shape[0]:
            # np.append is a lot faster, let's use it if we can.
            self._blklocs = np.append(self._blklocs, 0)
            self._blknos = np.append(self._blknos, len(self.blocks))
        elif loc == 0:
            # np.append is a lot faster, let's use it if we can.
            self._blklocs = np.append(self._blklocs[::-1], 0)[::-1]
            self._blknos = np.append(self._blknos[::-1], len(self.blocks))[::-1]
        else:
            new_blklocs, new_blknos = libinternals.update_blklocs_and_blknos(
                self.blklocs, self.blknos, loc, len(self.blocks)
            )
            self._blklocs = new_blklocs
            self._blknos = new_blknos

    def idelete(self, indexer) -> BlockManager:
        """
        Delete selected locations, returning a new BlockManager.
        """
        is_deleted = np.zeros(self.shape[0], dtype=np.bool_)
        is_deleted[indexer] = True
        taker = (~is_deleted).nonzero()[0]

        nbs = self._slice_take_blocks_ax0(taker, only_slice=True, ref_inplace_op=True)
        new_columns = self.items[~is_deleted]
        axes = [new_columns, self.axes[1]]
        return type(self)(tuple(nbs), axes, verify_integrity=False)

    # ----------------------------------------------------------------
    # Block-wise Operation

    def grouped_reduce(self, func: Callable) -> Self:
        """
        Apply grouped reduction function blockwise, returning a new BlockManager.

        Parameters
        ----------
        func : grouped reduction function

        Returns
        -------
        BlockManager
        """
        result_blocks: list[Block] = []

        for blk in self.blocks:
            if blk.is_object:
                # split on object-dtype blocks bc some columns may raise
                #  while others do not.
                for sb in blk._split():
                    applied = sb.apply(func)
                    result_blocks = extend_blocks(applied, result_blocks)
            else:
                applied = blk.apply(func)
                result_blocks = extend_blocks(applied, result_blocks)

        if len(result_blocks) == 0:
            nrows = 0
        else:
            nrows = result_blocks[0].values.shape[-1]
        index = Index(range(nrows))

        return type(self).from_blocks(result_blocks, [self.axes[0], index])

    def reduce(self, func: Callable) -> Self:
        """
        Apply reduction function blockwise, returning a single-row BlockManager.

        Parameters
        ----------
        func : reduction function

        Returns
        -------
        BlockManager
        """
        # If 2D, we assume that we're operating column-wise
        assert self.ndim == 2

        res_blocks: list[Block] = []
        for blk in self.blocks:
            nbs = blk.reduce(func)
            res_blocks.extend(nbs)

        index = Index([None])  # placeholder
        new_mgr = type(self).from_blocks(res_blocks, [self.items, index])
        return new_mgr

    def operate_blockwise(self, other: BlockManager, array_op) -> BlockManager:
        """
        Apply array_op blockwise with another (aligned) BlockManager.
        """
        return operate_blockwise(self, other, array_op)

    def _equal_values(self: BlockManager, other: BlockManager) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        return blockwise_all(self, other, array_equals)

    def quantile(
        self,
        *,
        qs: Index,  # with dtype float 64
        interpolation: QuantileInterpolation = "linear",
    ) -> Self:
        """
        Iterate over blocks applying quantile reduction.
        This routine is intended for reduction type operations and
        will do inference on the generated blocks.

        Parameters
        ----------
        interpolation : type of interpolation, default 'linear'
        qs : list of the quantiles to be computed

        Returns
        -------
        BlockManager
        """
        # Series dispatches to DataFrame for quantile, which allows us to
        #  simplify some of the code here and in the blocks
        assert self.ndim >= 2
        assert is_list_like(qs)  # caller is responsible for this

        new_axes = list(self.axes)
        new_axes[1] = Index(qs, dtype=np.float64)

        blocks = [
            blk.quantile(qs=qs, interpolation=interpolation) for blk in self.blocks
        ]

        return type(self)(blocks, new_axes)

    # ----------------------------------------------------------------

    def unstack(self, unstacker, fill_value) -> BlockManager:
        """
        Return a BlockManager with all blocks unstacked.

        Parameters
        ----------
        unstacker : reshape._Unstacker
        fill_value : Any
            fill_value for newly introduced missing values.

        Returns
        -------
        unstacked : BlockManager
        """
        new_columns = unstacker.get_new_columns(self.items)
        new_index = unstacker.new_index

        allow_fill = not unstacker.mask_all
        if allow_fill:
            # calculating the full mask once and passing it to Block._unstack is
            #  faster than letting calculating it in each repeated call
            new_mask2D = (~unstacker.mask).reshape(*unstacker.full_shape)
            needs_masking = new_mask2D.any(axis=0)
        else:
            needs_masking = np.zeros(unstacker.full_shape[1], dtype=bool)

        new_blocks: list[Block] = []
        columns_mask: list[np.ndarray] = []

        if len(self.items) == 0:
            factor = 1
        else:
            fac = len(new_columns) / len(self.items)
            assert fac == int(fac)
            factor = int(fac)

        for blk in self.blocks:
            mgr_locs = blk.mgr_locs
            new_placement = mgr_locs.tile_for_unstack(factor)

            blocks, mask = blk._unstack(
                unstacker,
                fill_value,
                new_placement=new_placement,
                needs_masking=needs_masking,
            )

            new_blocks.extend(blocks)
            columns_mask.extend(mask)

            # Block._unstack should ensure this holds,
            assert mask.sum() == sum(len(nb._mgr_locs) for nb in blocks)
            # In turn this ensures that in the BlockManager call below
            #  we have len(new_columns) == sum(x.shape[0] for x in new_blocks)
            #  which suffices to allow us to pass verify_inegrity=False

        new_columns = new_columns[columns_mask]

        bm = BlockManager(new_blocks, [new_columns, new_index], verify_integrity=False)
        return bm

    def to_dict(self) -> dict[str, Self]:
        """
        Return a dict of str(dtype) -> BlockManager

        Returns
        -------
        values : a dict of dtype -> BlockManager
        """

        bd: dict[str, list[Block]] = {}
        for b in self.blocks:
            bd.setdefault(str(b.dtype), []).append(b)

        # TODO(EA2D): the combine will be unnecessary with 2D EAs
        return {dtype: self._combine(blocks) for dtype, blocks in bd.items()}

    def as_array(
        self,
        dtype: np.dtype | None = None,
        copy: bool = False,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Convert the blockmanager data into an numpy array.

        Parameters
        ----------
        dtype : np.dtype or None, default None
            Data type of the return array.
        copy : bool, default False
            If True then guarantee that a copy is returned. A value of
            False does not guarantee that the underlying data is not
            copied.
        na_value : object, default lib.no_default
            Value to be used as the missing value sentinel.

        Returns
        -------
        arr : ndarray
        """
        passed_nan = lib.is_float(na_value) and isna(na_value)

        if len(self.blocks) == 0:
            arr = np.empty(self.shape, dtype=float)
            return arr.transpose()

        if self.is_single_block:
            blk = self.blocks[0]

            if na_value is not lib.no_default:
                # We want to copy when na_value is provided to avoid
                # mutating the original object
                if lib.is_np_dtype(blk.dtype, "f") and passed_nan:
                    # We are already numpy-float and na_value=np.nan
                    pass
                else:
                    copy = True

            if blk.is_extension:
                # Avoid implicit conversion of extension blocks to object

                # error: Item "ndarray" of "Union[ndarray, ExtensionArray]" has no
                # attribute "to_numpy"
                arr = blk.values.to_numpy(  # type: ignore[union-attr]
                    dtype=dtype,
                    na_value=na_value,
                    copy=copy,
                ).reshape(blk.shape)
            elif not copy:
                arr = np.asarray(blk.values, dtype=dtype)
            else:
                arr = np.array(blk.values, dtype=dtype, copy=copy)

            if using_copy_on_write() and not copy:
                arr = arr.view()
                arr.flags.writeable = False
        else:
            arr = self._interleave(dtype=dtype, na_value=na_value)
            # The underlying data was copied within _interleave, so no need
            # to further copy if copy=True or setting na_value

        if na_value is lib.no_default:
            pass
        elif arr.dtype.kind == "f" and passed_nan:
            pass
        else:
            arr[isna(arr)] = na_value

        return arr.transpose()

    def _interleave(
        self,
        dtype: np.dtype | None = None,
        na_value: object = lib.no_default,
    ) -> np.ndarray:
        """
        Return ndarray from blocks with specified item order
        Items must be contained in the blocks
        """
        if not dtype:
            # Incompatible types in assignment (expression has type
            # "Optional[Union[dtype[Any], ExtensionDtype]]", variable has
            # type "Optional[dtype[Any]]")
            dtype = interleaved_dtype(  # type: ignore[assignment]
                [blk.dtype for blk in self.blocks]
            )

        # error: Argument 1 to "ensure_np_dtype" has incompatible type
        # "Optional[dtype[Any]]"; expected "Union[dtype[Any], ExtensionDtype]"
        dtype = ensure_np_dtype(dtype)  # type: ignore[arg-type]
        result = np.empty(self.shape, dtype=dtype)

        itemmask = np.zeros(self.shape[0])

        if dtype == np.dtype("object") and na_value is lib.no_default:
            # much more performant than using to_numpy below
            for blk in self.blocks:
                rl = blk.mgr_locs
                arr = blk.get_values(dtype)
                result[rl.indexer] = arr
                itemmask[rl.indexer] = 1
            return result

        for blk in self.blocks:
            rl = blk.mgr_locs
            if blk.is_extension:
                # Avoid implicit conversion of extension blocks to object

                # error: Item "ndarray" of "Union[ndarray, ExtensionArray]" has no
                # attribute "to_numpy"
                arr = blk.values.to_numpy(  # type: ignore[union-attr]
                    dtype=dtype,
                    na_value=na_value,
                )
            else:
                arr = blk.get_values(dtype)
            result[rl.indexer] = arr
            itemmask[rl.indexer] = 1

        if not itemmask.all():
            raise AssertionError("Some items were not contained in blocks")

        return result

    # ----------------------------------------------------------------
    # Consolidation

    def is_consolidated(self) -> bool:
        """
        Return True if more than one block with the same dtype
        """
        if not self._known_consolidated:
            self._consolidate_check()
        return self._is_consolidated

    def _consolidate_check(self) -> None:
        if len(self.blocks) == 1:
            # fastpath
            self._is_consolidated = True
            self._known_consolidated = True
            return
        dtypes = [blk.dtype for blk in self.blocks if blk._can_consolidate]
        self._is_consolidated = len(dtypes) == len(set(dtypes))
        self._known_consolidated = True

    def _consolidate_inplace(self) -> None:
        # In general, _consolidate_inplace should only be called via
        #  DataFrame._consolidate_inplace, otherwise we will fail to invalidate
        #  the DataFrame's _item_cache. The exception is for newly-created
        #  BlockManager objects not yet attached to a DataFrame.
        if not self.is_consolidated():
            self.blocks = _consolidate(self.blocks)
            self._is_consolidated = True
            self._known_consolidated = True
            self._rebuild_blknos_and_blklocs()

    # ----------------------------------------------------------------
    # Concatenation

    @classmethod
    def concat_horizontal(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers horizontally.
        """
        offset = 0
        blocks: list[Block] = []
        for mgr in mgrs:
            for blk in mgr.blocks:
                # We need to do getitem_block here otherwise we would be altering
                #  blk.mgr_locs in place, which would render it invalid. This is only
                #  relevant in the copy=False case.
                nb = blk.slice_block_columns(slice(None))
                nb._mgr_locs = nb._mgr_locs.add(offset)
                blocks.append(nb)

            offset += len(mgr.items)

        new_mgr = cls(tuple(blocks), axes)
        return new_mgr

    @classmethod
    def concat_vertical(cls, mgrs: list[Self], axes: list[Index]) -> Self:
        """
        Concatenate uniformly-indexed BlockManagers vertically.
        """
        raise NotImplementedError("This logic lives (for now) in internals.concat")


class SingleBlockManager(BaseBlockManager, SingleDataManager):
    """manage a single block with"""

    @property
    def ndim(self) -> Literal[1]:
        return 1

    _is_consolidated = True
    _known_consolidated = True
    __slots__ = ()
    is_single_block = True

    def __init__(
        self,
        block: Block,
        axis: Index,
        verify_integrity: bool = False,
    ) -> None:
        # Assertions disabled for performance
        # assert isinstance(block, Block), type(block)
        # assert isinstance(axis, Index), type(axis)

        self.axes = [axis]
        self.blocks = (block,)

    @classmethod
    def from_blocks(
        cls,
        blocks: list[Block],
        axes: list[Index],
    ) -> Self:
        """
        Constructor for BlockManager and SingleBlockManager with same signature.
        """
        assert len(blocks) == 1
        assert len(axes) == 1
        return cls(blocks[0], axes[0], verify_integrity=False)

    @classmethod
    def from_array(
        cls, array: ArrayLike, index: Index, refs: BlockValuesRefs | None = None
    ) -> SingleBlockManager:
        """
        Constructor for if we have an array that is not yet a Block.
        """
        array = maybe_coerce_values(array)
        bp = BlockPlacement(slice(0, len(index)))
        block = new_block(array, placement=bp, ndim=1, refs=refs)
        return cls(block, index)

    def to_2d_mgr(self, columns: Index) -> BlockManager:
        """
        Manager analogue of Series.to_frame
        """
        blk = self.blocks[0]
        arr = ensure_block_shape(blk.values, ndim=2)
        bp = BlockPlacement(0)
        new_blk = type(blk)(arr, placement=bp, ndim=2, refs=blk.refs)
        axes = [columns, self.axes[0]]
        return BlockManager([new_blk], axes=axes, verify_integrity=False)

    def _has_no_reference(self, i: int = 0) -> bool:
        """
        Check for column `i` if it has references.
        (whether it references another array or is itself being referenced)
        Returns True if the column has no references.
        """
        return not self.blocks[0].refs.has_reference()

    def __getstate__(self):
        block_values = [b.values for b in self.blocks]
        block_items = [self.items[b.mgr_locs.indexer] for b in self.blocks]
        axes_array = list(self.axes)

        extra_state = {
            "0.14.1": {
                "axes": axes_array,
                "blocks": [
                    {"values": b.values, "mgr_locs": b.mgr_locs.indexer}
                    for b in self.blocks
                ],
            }
        }

        # First three elements of the state are to maintain forward
        # compatibility with 0.13.1.
        return axes_array, block_values, block_items, extra_state

    def __setstate__(self, state) -> None:
        def unpickle_block(values, mgr_locs, ndim: int) -> Block:
            # TODO(EA2D): ndim would be unnecessary with 2D EAs
            # older pickles may store e.g. DatetimeIndex instead of DatetimeArray
            values = extract_array(values, extract_numpy=True)
            if not isinstance(mgr_locs, BlockPlacement):
                mgr_locs = BlockPlacement(mgr_locs)

            values = maybe_coerce_values(values)
            return new_block(values, placement=mgr_locs, ndim=ndim)

        if isinstance(state, tuple) and len(state) >= 4 and "0.14.1" in state[3]:
            state = state[3]["0.14.1"]
            self.axes = [ensure_index(ax) for ax in state["axes"]]
            ndim = len(self.axes)
            self.blocks = tuple(
                unpickle_block(b["values"], b["mgr_locs"], ndim=ndim)
                for b in state["blocks"]
            )
        else:
            raise NotImplementedError("pre-0.14.1 pickles are no longer supported")

        self._post_setstate()

    def _post_setstate(self) -> None:
        pass

    @cache_readonly
    def _block(self) -> Block:
        return self.blocks[0]

    @property
    def _blknos(self):
        """compat with BlockManager"""
        return None

    @property
    def _blklocs(self):
        """compat with BlockManager"""
        return None

    def get_rows_with_mask(self, indexer: npt.NDArray[np.bool_]) -> Self:
        # similar to get_slice, but not restricted to slice indexer
        blk = self._block
        if using_copy_on_write() and len(indexer) > 0 and indexer.all():
            return type(self)(blk.copy(deep=False), self.index)
        array = blk.values[indexer]

        if isinstance(indexer, np.ndarray) and indexer.dtype.kind == "b":
            # boolean indexing always gives a copy with numpy
            refs = None
        else:
            # TODO(CoW) in theory only need to track reference if new_array is a view
            refs = blk.refs

        bp = BlockPlacement(slice(0, len(array)))
        block = type(blk)(array, placement=bp, ndim=1, refs=refs)

        new_idx = self.index[indexer]
        return type(self)(block, new_idx)

    def get_slice(self, slobj: slice, axis: AxisInt = 0) -> SingleBlockManager:
        # Assertion disabled for performance
        # assert isinstance(slobj, slice), type(slobj)
        if axis >= self.ndim:
            raise IndexError("Requested axis not found in manager")

        blk = self._block
        array = blk.values[slobj]
        bp = BlockPlacement(slice(0, len(array)))
        # TODO this method is only used in groupby SeriesSplitter at the moment,
        # so passing refs is not yet covered by the tests
        block = type(blk)(array, placement=bp, ndim=1, refs=blk.refs)
        new_index = self.index._getitem_slice(slobj)
        return type(self)(block, new_index)

    @property
    def index(self) -> Index:
        return self.axes[0]

    @property
    def dtype(self) -> DtypeObj:
        return self._block.dtype

    def get_dtypes(self) -> npt.NDArray[np.object_]:
        return np.array([self._block.dtype], dtype=object)

    def external_values(self):
        """The array that Series.values returns"""
        return self._block.external_values()

    def internal_values(self):
        """The array that Series._values returns"""
        return self._block.values

    def array_values(self) -> ExtensionArray:
        """The array that Series.array returns"""
        return self._block.array_values

    def get_numeric_data(self) -> Self:
        if self._block.is_numeric:
            return self.copy(deep=False)
        return self.make_empty()

    @property
    def _can_hold_na(self) -> bool:
        return self._block._can_hold_na

    def setitem_inplace(self, indexer, value, warn: bool = True) -> None:
        """
        Set values with indexer.

        For Single[Block/Array]Manager, this backs s[indexer] = value

        This is an inplace version of `setitem()`, mutating the manager/values
        in place, not returning a new Manager (and Block), and thus never changing
        the dtype.
        """
        using_cow = using_copy_on_write()
        warn_cow = warn_copy_on_write()
        if (using_cow or warn_cow) and not self._has_no_reference(0):
            if using_cow:
                self.blocks = (self._block.copy(),)
                self._cache.clear()
            elif warn_cow and warn:
                warnings.warn(
                    COW_WARNING_SETITEM_MSG,
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

        super().setitem_inplace(indexer, value)

    def idelete(self, indexer) -> SingleBlockManager:
        """
        Delete single location from SingleBlockManager.

        Ensures that self.blocks doesn't become empty.
        """
        nb = self._block.delete(indexer)[0]
        self.blocks = (nb,)
        self.axes[0] = self.axes[0].delete(indexer)
        self._cache.clear()
        return self

    def fast_xs(self, loc):
        """
        fast path for getting a cross-section
        return a view of the data
        """
        raise NotImplementedError("Use series._values[loc] instead")

    def set_values(self, values: ArrayLike) -> None:
        """
        Set the values of the single block in place.

        Use at your own risk! This does not check if the passed values are
        valid for the current Block/SingleBlockManager (length, dtype, etc),
        and this does not properly keep track of references.
        """
        # NOTE(CoW) Currently this is only used for FrameColumnApply.series_generator
        # which handles CoW by setting the refs manually if necessary
        self.blocks[0].values = values
        self.blocks[0]._mgr_locs = BlockPlacement(slice(len(values)))

    def _equal_values(self, other: Self) -> bool:
        """
        Used in .equals defined in base class. Only check the column values
        assuming shape and indexes have already been checked.
        """
        # For SingleBlockManager (i.e.Series)
        if other.ndim != 1:
            return False
        left = self.blocks[0].values
        right = other.blocks[0].values
        return array_equals(left, right)


# --------------------------------------------------------------------
# Constructor Helpers


def create_block_manager_from_blocks(
    blocks: list[Block],
    axes: list[Index],
    consolidate: bool = True,
    verify_integrity: bool = True,
) -> BlockManager:
    # If verify_integrity=False, then caller is responsible for checking
    #  all(x.shape[-1] == len(axes[1]) for x in blocks)
    #  sum(x.shape[0] for x in blocks) == len(axes[0])
    #  set(x for blk in blocks for x in blk.mgr_locs) == set(range(len(axes[0])))
    #  all(blk.ndim == 2 for blk in blocks)
    # This allows us to safely pass verify_integrity=False

    try:
        mgr = BlockManager(blocks, axes, verify_integrity=verify_integrity)

    except ValueError as err:
        arrays = [blk.values for blk in blocks]
        tot_items = sum(arr.shape[0] for arr in arrays)
        raise_construction_error(tot_items, arrays[0].shape[1:], axes, err)

    if consolidate:
        mgr._consolidate_inplace()
    return mgr


def create_block_manager_from_column_arrays(
    arrays: list[ArrayLike],
    axes: list[Index],
    consolidate: bool,
    refs: list,
) -> BlockManager:
    # Assertions disabled for performance (caller is responsible for verifying)
    # assert isinstance(axes, list)
    # assert all(isinstance(x, Index) for x in axes)
    # assert all(isinstance(x, (np.ndarray, ExtensionArray)) for x in arrays)
    # assert all(type(x) is not NumpyExtensionArray for x in arrays)
    # assert all(x.ndim == 1 for x in arrays)
    # assert all(len(x) == len(axes[1]) for x in arrays)
    # assert len(arrays) == len(axes[0])
    # These last three are sufficient to allow us to safely pass
    #  verify_integrity=False below.

    try:
        blocks = _form_blocks(arrays, consolidate, refs)
        mgr = BlockManager(blocks, axes, verify_integrity=False)
    except ValueError as e:
        raise_construction_error(len(arrays), arrays[0].shape, axes, e)
    if consolidate:
        mgr._consolidate_inplace()
    return mgr


def raise_construction_error(
    tot_items: int,
    block_shape: Shape,
    axes: list[Index],
    e: ValueError | None = None,
):
    """raise a helpful message about our construction"""
    passed = tuple(map(int, [tot_items] + list(block_shape)))
    # Correcting the user facing error message during dataframe construction
    if len(passed) <= 2:
        passed = passed[::-1]

    implied = tuple(len(ax) for ax in axes)
    # Correcting the user facing error message during dataframe construction
    if len(implied) <= 2:
        implied = implied[::-1]

    # We return the exception object instead of raising it so that we
    #  can raise it in the caller; mypy plays better with that
    if passed == implied and e is not None:
        raise e
    if block_shape[0] == 0:
        raise ValueError("Empty data passed with indices specified.")
    raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")


# -----------------------------------------------------------------------


def _grouping_func(tup: tuple[int, ArrayLike]) -> tuple[int, DtypeObj]:
    dtype = tup[1].dtype

    if is_1d_only_ea_dtype(dtype):
        # We know these won't be consolidated, so don't need to group these.
        # This avoids expensive comparisons of CategoricalDtype objects
        sep = id(dtype)
    else:
        sep = 0

    return sep, dtype


def _form_blocks(arrays: list[ArrayLike], consolidate: bool, refs: list) -> list[Block]:
    tuples = list(enumerate(arrays))

    if not consolidate:
        return _tuples_to_blocks_no_consolidate(tuples, refs)

    # when consolidating, we can ignore refs (either stacking always copies,
    # or the EA is already copied in the calling dict_to_mgr)

    # group by dtype
    grouper = itertools.groupby(tuples, _grouping_func)

    nbs: list[Block] = []
    for (_, dtype), tup_block in grouper:
        block_type = get_block_type(dtype)

        if isinstance(dtype, np.dtype):
            is_dtlike = dtype.kind in "mM"

            if issubclass(dtype.type, (str, bytes)):
                dtype = np.dtype(object)

            values, placement = _stack_arrays(list(tup_block), dtype)
            if is_dtlike:
                values = ensure_wrapped_if_datetimelike(values)
            blk = block_type(values, placement=BlockPlacement(placement), ndim=2)
            nbs.append(blk)

        elif is_1d_only_ea_dtype(dtype):
            dtype_blocks = [
                block_type(x[1], placement=BlockPlacement(x[0]), ndim=2)
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)

        else:
            dtype_blocks = [
                block_type(
                    ensure_block_shape(x[1], 2), placement=BlockPlacement(x[0]), ndim=2
                )
                for x in tup_block
            ]
            nbs.extend(dtype_blocks)
    return nbs


def _tuples_to_blocks_no_consolidate(tuples, refs) -> list[Block]:
    # tuples produced within _form_blocks are of the form (placement, array)
    return [
        new_block_2d(
            ensure_block_shape(arr, ndim=2), placement=BlockPlacement(i), refs=ref
        )
        for ((i, arr), ref) in zip(tuples, refs)
    ]


def _stack_arrays(tuples, dtype: np.dtype):
    placement, arrays = zip(*tuples)

    first = arrays[0]
    shape = (len(arrays),) + first.shape

    stacked = np.empty(shape, dtype=dtype)
    for i, arr in enumerate(arrays):
        stacked[i] = arr

    return stacked, placement


def _consolidate(blocks: tuple[Block, ...]) -> tuple[Block, ...]:
    """
    Merge blocks having same dtype, exclude non-consolidating blocks
    """
    # sort by _can_consolidate, dtype
    gkey = lambda x: x._consolidate_key
    grouper = itertools.groupby(sorted(blocks, key=gkey), gkey)

    new_blocks: list[Block] = []
    for (_can_consolidate, dtype), group_blocks in grouper:
        merged_blocks, _ = _merge_blocks(
            list(group_blocks), dtype=dtype, can_consolidate=_can_consolidate
        )
        new_blocks = extend_blocks(merged_blocks, new_blocks)
    return tuple(new_blocks)


def _merge_blocks(
    blocks: list[Block], dtype: DtypeObj, can_consolidate: bool
) -> tuple[list[Block], bool]:
    if len(blocks) == 1:
        return blocks, False

    if can_consolidate:
        # TODO: optimization potential in case all mgrs contain slices and
        # combination of those slices is a slice, too.
        new_mgr_locs = np.concatenate([b.mgr_locs.as_array for b in blocks])

        new_values: ArrayLike

        if isinstance(blocks[0].dtype, np.dtype):
            # error: List comprehension has incompatible type List[Union[ndarray,
            # ExtensionArray]]; expected List[Union[complex, generic,
            # Sequence[Union[int, float, complex, str, bytes, generic]],
            # Sequence[Sequence[Any]], SupportsArray]]
            new_values = np.vstack([b.values for b in blocks])  # type: ignore[misc]
        else:
            bvals = [blk.values for blk in blocks]
            bvals2 = cast(Sequence[NDArrayBackedExtensionArray], bvals)
            new_values = bvals2[0]._concat_same_type(bvals2, axis=0)

        argsort = np.argsort(new_mgr_locs)
        new_values = new_values[argsort]
        new_mgr_locs = new_mgr_locs[argsort]

        bp = BlockPlacement(new_mgr_locs)
        return [new_block_2d(new_values, placement=bp)], True

    # can't consolidate --> no merge
    return blocks, False


def _fast_count_smallints(arr: npt.NDArray[np.intp]):
    """Faster version of set(arr) for sequences of small numbers."""
    counts = np.bincount(arr)
    nz = counts.nonzero()[0]
    # Note: list(zip(...) outperforms list(np.c_[nz, counts[nz]]) here,
    #  in one benchmark by a factor of 11
    return zip(nz, counts[nz])


def _preprocess_slice_or_indexer(
    slice_or_indexer: slice | np.ndarray, length: int, allow_fill: bool
):
    if isinstance(slice_or_indexer, slice):
        return (
            "slice",
            slice_or_indexer,
            libinternals.slice_len(slice_or_indexer, length),
        )
    else:
        if (
            not isinstance(slice_or_indexer, np.ndarray)
            or slice_or_indexer.dtype.kind != "i"
        ):
            dtype = getattr(slice_or_indexer, "dtype", None)
            raise TypeError(type(slice_or_indexer), dtype)

        indexer = ensure_platform_int(slice_or_indexer)
        if not allow_fill:
            indexer = maybe_convert_indices(indexer, length)
        return "fancy", indexer, len(indexer)


def make_na_array(dtype: DtypeObj, shape: Shape, fill_value) -> ArrayLike:
    if isinstance(dtype, DatetimeTZDtype):
        # NB: exclude e.g. pyarrow[dt64tz] dtypes
        ts = Timestamp(fill_value).as_unit(dtype.unit)
        i8values = np.full(shape, ts._value)
        dt64values = i8values.view(f"M8[{dtype.unit}]")
        return DatetimeArray._simple_new(dt64values, dtype=dtype)

    elif is_1d_only_ea_dtype(dtype):
        dtype = cast(ExtensionDtype, dtype)
        cls = dtype.construct_array_type()

        missing_arr = cls._from_sequence([], dtype=dtype)
        ncols, nrows = shape
        assert ncols == 1, ncols
        empty_arr = -1 * np.ones((nrows,), dtype=np.intp)
        return missing_arr.take(empty_arr, allow_fill=True, fill_value=fill_value)
    elif isinstance(dtype, ExtensionDtype):
        # TODO: no tests get here, a handful would if we disabled
        #  the dt64tz special-case above (which is faster)
        cls = dtype.construct_array_type()
        missing_arr = cls._empty(shape=shape, dtype=dtype)
        missing_arr[:] = fill_value
        return missing_arr
    else:
        # NB: we should never get here with dtype integer or bool;
        #  if we did, the missing_arr.fill would cast to gibberish
        missing_arr = np.empty(shape, dtype=dtype)
        missing_arr.fill(fill_value)

        if dtype.kind in "mM":
            missing_arr = ensure_wrapped_if_datetimelike(missing_arr)
        return missing_arr

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\type_api.py ===
# sql/type_api.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Base types API.

"""

from __future__ import annotations

from enum import Enum
from types import ModuleType
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Generic
from typing import Mapping
from typing import NewType
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from .base import SchemaEventTarget
from .cache_key import CacheConst
from .cache_key import NO_CACHE
from .operators import ColumnOperators
from .visitors import Visitable
from .. import exc
from .. import util
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypeAliasType
from ..util.typing import TypedDict
from ..util.typing import TypeGuard

# these are back-assigned by sqltypes.
if typing.TYPE_CHECKING:
    from ._typing import _TypeEngineArgument
    from .elements import BindParameter
    from .elements import ColumnElement
    from .operators import OperatorType
    from .sqltypes import _resolve_value_to_type as _resolve_value_to_type
    from .sqltypes import BOOLEANTYPE as BOOLEANTYPE  # noqa: F401
    from .sqltypes import INDEXABLE as INDEXABLE  # noqa: F401
    from .sqltypes import INTEGERTYPE as INTEGERTYPE  # noqa: F401
    from .sqltypes import MATCHTYPE as MATCHTYPE  # noqa: F401
    from .sqltypes import NULLTYPE as NULLTYPE
    from .sqltypes import NUMERICTYPE as NUMERICTYPE  # noqa: F401
    from .sqltypes import STRINGTYPE as STRINGTYPE  # noqa: F401
    from .sqltypes import TABLEVALUE as TABLEVALUE  # noqa: F401
    from ..engine.interfaces import Dialect
    from ..util.typing import GenericProtocol

_T = TypeVar("_T", bound=Any)
_T_co = TypeVar("_T_co", bound=Any, covariant=True)
_T_con = TypeVar("_T_con", bound=Any, contravariant=True)
_O = TypeVar("_O", bound=object)
_TE = TypeVar("_TE", bound="TypeEngine[Any]")
_CT = TypeVar("_CT", bound=Any)
_RT = TypeVar("_RT", bound=Any)

_MatchedOnType = Union[
    "GenericProtocol[Any]", TypeAliasType, NewType, Type[Any]
]


class _NoValueInList(Enum):
    NO_VALUE_IN_LIST = 0
    """indicates we are trying to determine the type of an expression
    against an empty list."""


_NO_VALUE_IN_LIST = _NoValueInList.NO_VALUE_IN_LIST


class _LiteralProcessorType(Protocol[_T_co]):
    def __call__(self, value: Any) -> str: ...


class _BindProcessorType(Protocol[_T_con]):
    def __call__(self, value: Optional[_T_con]) -> Any: ...


class _ResultProcessorType(Protocol[_T_co]):
    def __call__(self, value: Any) -> Optional[_T_co]: ...


class _SentinelProcessorType(Protocol[_T_co]):
    def __call__(self, value: Any) -> Optional[_T_co]: ...


class _BaseTypeMemoDict(TypedDict):
    impl: TypeEngine[Any]
    result: Dict[Any, Optional[_ResultProcessorType[Any]]]


class _TypeMemoDict(_BaseTypeMemoDict, total=False):
    literal: Optional[_LiteralProcessorType[Any]]
    bind: Optional[_BindProcessorType[Any]]
    sentinel: Optional[_SentinelProcessorType[Any]]
    custom: Dict[Any, object]


class _ComparatorFactory(Protocol[_T]):
    def __call__(
        self, expr: ColumnElement[_T]
    ) -> TypeEngine.Comparator[_T]: ...


class TypeEngine(Visitable, Generic[_T]):
    """The ultimate base class for all SQL datatypes.

    Common subclasses of :class:`.TypeEngine` include
    :class:`.String`, :class:`.Integer`, and :class:`.Boolean`.

    For an overview of the SQLAlchemy typing system, see
    :ref:`types_toplevel`.

    .. seealso::

        :ref:`types_toplevel`

    """

    _sqla_type = True
    _isnull = False
    _is_tuple_type = False
    _is_table_value = False
    _is_array = False
    _is_type_decorator = False

    render_bind_cast = False
    """Render bind casts for :attr:`.BindTyping.RENDER_CASTS` mode.

    If True, this type (usually a dialect level impl type) signals
    to the compiler that a cast should be rendered around a bound parameter
    for this type.

    .. versionadded:: 2.0

    .. seealso::

        :class:`.BindTyping`

    """

    render_literal_cast = False
    """render casts when rendering a value as an inline literal,
    e.g. with :meth:`.TypeEngine.literal_processor`.

    .. versionadded:: 2.0

    """

    class Comparator(
        ColumnOperators,
        Generic[_CT],
    ):
        """Base class for custom comparison operations defined at the
        type level.  See :attr:`.TypeEngine.comparator_factory`.


        """

        __slots__ = "expr", "type"

        expr: ColumnElement[_CT]
        type: TypeEngine[_CT]

        def __clause_element__(self) -> ColumnElement[_CT]:
            return self.expr

        def __init__(self, expr: ColumnElement[_CT]):
            self.expr = expr
            self.type = expr.type

        def __reduce__(self) -> Any:
            return self.__class__, (self.expr,)

        @overload
        def operate(
            self,
            op: OperatorType,
            *other: Any,
            result_type: Type[TypeEngine[_RT]],
            **kwargs: Any,
        ) -> ColumnElement[_RT]: ...

        @overload
        def operate(
            self, op: OperatorType, *other: Any, **kwargs: Any
        ) -> ColumnElement[_CT]: ...

        @util.preload_module("sqlalchemy.sql.default_comparator")
        def operate(
            self, op: OperatorType, *other: Any, **kwargs: Any
        ) -> ColumnElement[Any]:
            default_comparator = util.preloaded.sql_default_comparator
            op_fn, addtl_kw = default_comparator.operator_lookup[op.__name__]
            if kwargs:
                addtl_kw = addtl_kw.union(kwargs)
            return op_fn(self.expr, op, *other, **addtl_kw)

        @util.preload_module("sqlalchemy.sql.default_comparator")
        def reverse_operate(
            self, op: OperatorType, other: Any, **kwargs: Any
        ) -> ColumnElement[_CT]:
            default_comparator = util.preloaded.sql_default_comparator
            op_fn, addtl_kw = default_comparator.operator_lookup[op.__name__]
            if kwargs:
                addtl_kw = addtl_kw.union(kwargs)
            return op_fn(self.expr, op, other, reverse=True, **addtl_kw)

        def _adapt_expression(
            self,
            op: OperatorType,
            other_comparator: TypeEngine.Comparator[Any],
        ) -> Tuple[OperatorType, TypeEngine[Any]]:
            """evaluate the return type of <self> <op> <othertype>,
            and apply any adaptations to the given operator.

            This method determines the type of a resulting binary expression
            given two source types and an operator.   For example, two
            :class:`_schema.Column` objects, both of the type
            :class:`.Integer`, will
            produce a :class:`.BinaryExpression` that also has the type
            :class:`.Integer` when compared via the addition (``+``) operator.
            However, using the addition operator with an :class:`.Integer`
            and a :class:`.Date` object will produce a :class:`.Date`, assuming
            "days delta" behavior by the database (in reality, most databases
            other than PostgreSQL don't accept this particular operation).

            The method returns a tuple of the form <operator>, <type>.
            The resulting operator and type will be those applied to the
            resulting :class:`.BinaryExpression` as the final operator and the
            right-hand side of the expression.

            Note that only a subset of operators make usage of
            :meth:`._adapt_expression`,
            including math operators and user-defined operators, but not
            boolean comparison or special SQL keywords like MATCH or BETWEEN.

            """

            return op, self.type

    hashable = True
    """Flag, if False, means values from this type aren't hashable.

    Used by the ORM when uniquing result lists.

    """

    comparator_factory: _ComparatorFactory[Any] = Comparator
    """A :class:`.TypeEngine.Comparator` class which will apply
    to operations performed by owning :class:`_expression.ColumnElement`
    objects.

    The :attr:`.comparator_factory` attribute is a hook consulted by
    the core expression system when column and SQL expression operations
    are performed.   When a :class:`.TypeEngine.Comparator` class is
    associated with this attribute, it allows custom re-definition of
    all existing operators, as well as definition of new operators.
    Existing operators include those provided by Python operator overloading
    such as :meth:`.operators.ColumnOperators.__add__` and
    :meth:`.operators.ColumnOperators.__eq__`,
    those provided as standard
    attributes of :class:`.operators.ColumnOperators` such as
    :meth:`.operators.ColumnOperators.like`
    and :meth:`.operators.ColumnOperators.in_`.

    Rudimentary usage of this hook is allowed through simple subclassing
    of existing types, or alternatively by using :class:`.TypeDecorator`.
    See the documentation section :ref:`types_operators` for examples.

    """

    sort_key_function: Optional[Callable[[Any], Any]] = None
    """A sorting function that can be passed as the key to sorted.

    The default value of ``None`` indicates that the values stored by
    this type are self-sorting.

    .. versionadded:: 1.3.8

    """

    should_evaluate_none: bool = False
    """If True, the Python constant ``None`` is considered to be handled
    explicitly by this type.

    The ORM uses this flag to indicate that a positive value of ``None``
    is passed to the column in an INSERT statement, rather than omitting
    the column from the INSERT statement which has the effect of firing
    off column-level defaults.   It also allows types which have special
    behavior for Python None, such as a JSON type, to indicate that
    they'd like to handle the None value explicitly.

    To set this flag on an existing type, use the
    :meth:`.TypeEngine.evaluates_none` method.

    .. seealso::

        :meth:`.TypeEngine.evaluates_none`

    """

    _variant_mapping: util.immutabledict[str, TypeEngine[Any]] = (
        util.EMPTY_DICT
    )

    def evaluates_none(self) -> Self:
        """Return a copy of this type which has the
        :attr:`.should_evaluate_none` flag set to True.

        E.g.::

                Table(
                    "some_table",
                    metadata,
                    Column(
                        String(50).evaluates_none(),
                        nullable=True,
                        server_default="no value",
                    ),
                )

        The ORM uses this flag to indicate that a positive value of ``None``
        is passed to the column in an INSERT statement, rather than omitting
        the column from the INSERT statement which has the effect of firing
        off column-level defaults.   It also allows for types which have
        special behavior associated with the Python None value to indicate
        that the value doesn't necessarily translate into SQL NULL; a
        prime example of this is a JSON type which may wish to persist the
        JSON value ``'null'``.

        In all cases, the actual NULL SQL value can be always be
        persisted in any column by using
        the :obj:`_expression.null` SQL construct in an INSERT statement
        or associated with an ORM-mapped attribute.

        .. note::

            The "evaluates none" flag does **not** apply to a value
            of ``None`` passed to :paramref:`_schema.Column.default` or
            :paramref:`_schema.Column.server_default`; in these cases,
            ``None``
            still means "no default".

        .. seealso::

            :ref:`session_forcing_null` - in the ORM documentation

            :paramref:`.postgresql.JSON.none_as_null` - PostgreSQL JSON
            interaction with this flag.

            :attr:`.TypeEngine.should_evaluate_none` - class-level flag

        """
        typ = self.copy()
        typ.should_evaluate_none = True
        return typ

    def copy(self, **kw: Any) -> Self:
        return self.adapt(self.__class__)

    def copy_value(self, value: Any) -> Any:
        return value

    def literal_processor(
        self, dialect: Dialect
    ) -> Optional[_LiteralProcessorType[_T]]:
        """Return a conversion function for processing literal values that are
        to be rendered directly without using binds.

        This function is used when the compiler makes use of the
        "literal_binds" flag, typically used in DDL generation as well
        as in certain scenarios where backends don't accept bound parameters.

        Returns a callable which will receive a literal Python value
        as the sole positional argument and will return a string representation
        to be rendered in a SQL statement.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.literal_processor`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.literal_processor`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.process_literal_param`.

            .. seealso::

                :ref:`types_typedecorator`


        """
        return None

    def bind_processor(
        self, dialect: Dialect
    ) -> Optional[_BindProcessorType[_T]]:
        """Return a conversion function for processing bind values.

        Returns a callable which will receive a bind parameter value
        as the sole positional argument and will return a value to
        send to the DB-API.

        If processing is not necessary, the method should return ``None``.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.bind_processor`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.bind_processor`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.process_bind_param`.

            .. seealso::

                :ref:`types_typedecorator`


        :param dialect: Dialect instance in use.

        """
        return None

    def result_processor(
        self, dialect: Dialect, coltype: object
    ) -> Optional[_ResultProcessorType[_T]]:
        """Return a conversion function for processing result row values.

        Returns a callable which will receive a result row column
        value as the sole positional argument and will return a value
        to return to the user.

        If processing is not necessary, the method should return ``None``.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.result_processor`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.result_processor`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.process_result_value`.

            .. seealso::

                :ref:`types_typedecorator`

        :param dialect: Dialect instance in use.

        :param coltype: DBAPI coltype argument received in cursor.description.

        """
        return None

    def column_expression(
        self, colexpr: ColumnElement[_T]
    ) -> Optional[ColumnElement[_T]]:
        """Given a SELECT column expression, return a wrapping SQL expression.

        This is typically a SQL function that wraps a column expression
        as rendered in the columns clause of a SELECT statement.
        It is used for special data types that require
        columns to be wrapped in some special database function in order
        to coerce the value before being sent back to the application.
        It is the SQL analogue of the :meth:`.TypeEngine.result_processor`
        method.

        This method is called during the **SQL compilation** phase of a
        statement, when rendering a SQL string. It is **not** called
        against specific values.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.column_expression`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.column_expression`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.column_expression`.

            .. seealso::

                :ref:`types_typedecorator`


        .. seealso::

            :ref:`types_sql_value_processing`

        """

        return None

    @util.memoized_property
    def _has_column_expression(self) -> bool:
        """memoized boolean, check if column_expression is implemented.

        Allows the method to be skipped for the vast majority of expression
        types that don't use this feature.

        """

        return (
            self.__class__.column_expression.__code__
            is not TypeEngine.column_expression.__code__
        )

    def bind_expression(
        self, bindvalue: BindParameter[_T]
    ) -> Optional[ColumnElement[_T]]:
        """Given a bind value (i.e. a :class:`.BindParameter` instance),
        return a SQL expression in its place.

        This is typically a SQL function that wraps the existing bound
        parameter within the statement.  It is used for special data types
        that require literals being wrapped in some special database function
        in order to coerce an application-level value into a database-specific
        format.  It is the SQL analogue of the
        :meth:`.TypeEngine.bind_processor` method.

        This method is called during the **SQL compilation** phase of a
        statement, when rendering a SQL string. It is **not** called
        against specific values.

        Note that this method, when implemented, should always return
        the exact same structure, without any conditional logic, as it
        may be used in an executemany() call against an arbitrary number
        of bound parameter sets.

        .. note::

            This method is only called relative to a **dialect specific type
            object**, which is often **private to a dialect in use** and is not
            the same type object as the public facing one, which means it's not
            feasible to subclass a :class:`.types.TypeEngine` class in order to
            provide an alternate :meth:`_types.TypeEngine.bind_expression`
            method, unless subclassing the :class:`_types.UserDefinedType`
            class explicitly.

            To provide alternate behavior for
            :meth:`_types.TypeEngine.bind_expression`, implement a
            :class:`_types.TypeDecorator` class and provide an implementation
            of :meth:`_types.TypeDecorator.bind_expression`.

            .. seealso::

                :ref:`types_typedecorator`

        .. seealso::

            :ref:`types_sql_value_processing`

        """
        return None

    @util.memoized_property
    def _has_bind_expression(self) -> bool:
        """memoized boolean, check if bind_expression is implemented.

        Allows the method to be skipped for the vast majority of expression
        types that don't use this feature.

        """

        return util.method_is_overridden(self, TypeEngine.bind_expression)

    @staticmethod
    def _to_instance(cls_or_self: Union[Type[_TE], _TE]) -> _TE:
        return to_instance(cls_or_self)

    def compare_values(self, x: Any, y: Any) -> bool:
        """Compare two values for equality."""

        return x == y  # type: ignore[no-any-return]

    def get_dbapi_type(self, dbapi: ModuleType) -> Optional[Any]:
        """Return the corresponding type object from the underlying DB-API, if
        any.

        This can be useful for calling ``setinputsizes()``, for example.

        """
        return None

    @property
    def python_type(self) -> Type[Any]:
        """Return the Python type object expected to be returned
        by instances of this type, if known.

        Basically, for those types which enforce a return type,
        or are known across the board to do such for all common
        DBAPIs (like ``int`` for example), will return that type.

        If a return type is not defined, raises
        ``NotImplementedError``.

        Note that any type also accommodates NULL in SQL which
        means you can also get back ``None`` from any type
        in practice.

        """
        raise NotImplementedError()

    def with_variant(
        self,
        type_: _TypeEngineArgument[Any],
        *dialect_names: str,
    ) -> Self:
        r"""Produce a copy of this type object that will utilize the given
        type when applied to the dialect of the given name.

        e.g.::

            from sqlalchemy.types import String
            from sqlalchemy.dialects import mysql

            string_type = String()

            string_type = string_type.with_variant(
                mysql.VARCHAR(collation="foo"), "mysql", "mariadb"
            )

        The variant mapping indicates that when this type is
        interpreted by a specific dialect, it will instead be
        transmuted into the given type, rather than using the
        primary type.

        .. versionchanged:: 2.0 the :meth:`_types.TypeEngine.with_variant`
           method now works with a :class:`_types.TypeEngine` object "in
           place", returning a copy of the original type rather than returning
           a wrapping object; the ``Variant`` class is no longer used.

        :param type\_: a :class:`.TypeEngine` that will be selected
         as a variant from the originating type, when a dialect
         of the given name is in use.
        :param \*dialect_names: one or more base names of the dialect which
         uses this type. (i.e. ``'postgresql'``, ``'mysql'``, etc.)

         .. versionchanged:: 2.0 multiple dialect names can be specified
            for one variant.

        .. seealso::

            :ref:`types_with_variant` - illustrates the use of
            :meth:`_types.TypeEngine.with_variant`.

        """

        if not dialect_names:
            raise exc.ArgumentError("At least one dialect name is required")
        for dialect_name in dialect_names:
            if dialect_name in self._variant_mapping:
                raise exc.ArgumentError(
                    f"Dialect {dialect_name!r} is already present in "
                    f"the mapping for this {self!r}"
                )
        new_type = self.copy()
        type_ = to_instance(type_)
        if type_._variant_mapping:
            raise exc.ArgumentError(
                "can't pass a type that already has variants as a "
                "dialect-level type to with_variant()"
            )

        new_type._variant_mapping = self._variant_mapping.union(
            {dialect_name: type_ for dialect_name in dialect_names}
        )
        return new_type

    def _resolve_for_literal(self, value: Any) -> Self:
        """adjust this type given a literal Python value that will be
        stored in a bound parameter.

        Used exclusively by _resolve_value_to_type().

        .. versionadded:: 1.4.30 or 2.0

        TODO: this should be part of public API

        .. seealso::

            :meth:`.TypeEngine._resolve_for_python_type`

        """
        return self

    def _resolve_for_python_type(
        self,
        python_type: Type[Any],
        matched_on: _MatchedOnType,
        matched_on_flattened: Type[Any],
    ) -> Optional[Self]:
        """given a Python type (e.g. ``int``, ``str``, etc. ) return an
        instance of this :class:`.TypeEngine` that's appropriate for this type.

        An additional argument ``matched_on`` is passed, which indicates an
        entry from the ``__mro__`` of the given ``python_type`` that more
        specifically matches how the caller located this :class:`.TypeEngine`
        object.   Such as, if a lookup of some kind links the ``int`` Python
        type to the :class:`.Integer` SQL type, and the original object
        was some custom subclass of ``int`` such as ``MyInt(int)``, the
        arguments passed would be ``(MyInt, int)``.

        If the given Python type does not correspond to this
        :class:`.TypeEngine`, or the Python type is otherwise ambiguous, the
        method should return None.

        For simple cases, the method checks that the ``python_type``
        and ``matched_on`` types are the same (i.e. not a subclass), and
        returns self; for all other cases, it returns ``None``.

        The initial use case here is for the ORM to link user-defined
        Python standard library ``enum.Enum`` classes to the SQLAlchemy
        :class:`.Enum` SQL type when constructing ORM Declarative mappings.

        :param python_type: the Python type we want to use
        :param matched_on: the Python type that led us to choose this
         particular :class:`.TypeEngine` class, which would be a supertype
         of ``python_type``.   By default, the request is rejected if
         ``python_type`` doesn't match ``matched_on`` (None is returned).

        .. versionadded:: 2.0.0b4

        TODO: this should be part of public API

        .. seealso::

            :meth:`.TypeEngine._resolve_for_literal`

        """

        if python_type is not matched_on_flattened:
            return None

        return self

    def _with_collation(self, collation: str) -> Self:
        """set up error handling for the collate expression"""
        raise NotImplementedError("this datatype does not support collation")

    @util.ro_memoized_property
    def _type_affinity(self) -> Optional[Type[TypeEngine[_T]]]:
        """Return a rudimental 'affinity' value expressing the general class
        of type."""

        typ = None
        for t in self.__class__.__mro__:
            if t is TypeEngine or TypeEngineMixin in t.__bases__:
                return typ
            elif issubclass(t, TypeEngine):
                typ = t
        else:
            return self.__class__

    @util.ro_memoized_property
    def _generic_type_affinity(
        self,
    ) -> Type[TypeEngine[_T]]:
        best_camelcase = None
        best_uppercase = None

        if not isinstance(self, TypeEngine):
            return self.__class__

        for t in self.__class__.__mro__:
            if (
                t.__module__
                in (
                    "sqlalchemy.sql.sqltypes",
                    "sqlalchemy.sql.type_api",
                )
                and issubclass(t, TypeEngine)
                and TypeEngineMixin not in t.__bases__
                and t not in (TypeEngine, TypeEngineMixin)
                and t.__name__[0] != "_"
            ):
                if t.__name__.isupper() and not best_uppercase:
                    best_uppercase = t
                elif not t.__name__.isupper() and not best_camelcase:
                    best_camelcase = t

        return (
            best_camelcase
            or best_uppercase
            or cast("Type[TypeEngine[_T]]", NULLTYPE.__class__)
        )

    def as_generic(self, allow_nulltype: bool = False) -> TypeEngine[_T]:
        """
        Return an instance of the generic type corresponding to this type
        using heuristic rule. The method may be overridden if this
        heuristic rule is not sufficient.

        >>> from sqlalchemy.dialects.mysql import INTEGER
        >>> INTEGER(display_width=4).as_generic()
        Integer()

        >>> from sqlalchemy.dialects.mysql import NVARCHAR
        >>> NVARCHAR(length=100).as_generic()
        Unicode(length=100)

        .. versionadded:: 1.4.0b2


        .. seealso::

            :ref:`metadata_reflection_dbagnostic_types` - describes the
            use of :meth:`_types.TypeEngine.as_generic` in conjunction with
            the :meth:`_sql.DDLEvents.column_reflect` event, which is its
            intended use.

        """
        if (
            not allow_nulltype
            and self._generic_type_affinity == NULLTYPE.__class__
        ):
            raise NotImplementedError(
                "Default TypeEngine.as_generic() "
                "heuristic method was unsuccessful for {}. A custom "
                "as_generic() method must be implemented for this "
                "type class.".format(
                    self.__class__.__module__ + "." + self.__class__.__name__
                )
            )

        return util.constructor_copy(self, self._generic_type_affinity)

    def dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
        """Return a dialect-specific implementation for this
        :class:`.TypeEngine`.

        """
        try:
            tm = dialect._type_memos[self]
        except KeyError:
            pass
        else:
            return tm["impl"]
        return self._dialect_info(dialect)["impl"]

    def _unwrapped_dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
        """Return the 'unwrapped' dialect impl for this type.

        For a type that applies wrapping logic (e.g. TypeDecorator), give
        us the real, actual dialect-level type that is used.

        This is used by TypeDecorator itself as well at least one case where
        dialects need to check that a particular specific dialect-level
        type is in use, within the :meth:`.DefaultDialect.set_input_sizes`
        method.

        """
        return self.dialect_impl(dialect)

    def _cached_literal_processor(
        self, dialect: Dialect
    ) -> Optional[_LiteralProcessorType[_T]]:
        """Return a dialect-specific literal processor for this type."""

        try:
            return dialect._type_memos[self]["literal"]
        except KeyError:
            pass

        # avoid KeyError context coming into literal_processor() function
        # raises
        d = self._dialect_info(dialect)
        d["literal"] = lp = d["impl"].literal_processor(dialect)
        return lp

    def _cached_bind_processor(
        self, dialect: Dialect
    ) -> Optional[_BindProcessorType[_T]]:
        """Return a dialect-specific bind processor for this type."""

        try:
            return dialect._type_memos[self]["bind"]
        except KeyError:
            pass

        # avoid KeyError context coming into bind_processor() function
        # raises
        d = self._dialect_info(dialect)
        d["bind"] = bp = d["impl"].bind_processor(dialect)
        return bp

    def _cached_result_processor(
        self, dialect: Dialect, coltype: Any
    ) -> Optional[_ResultProcessorType[_T]]:
        """Return a dialect-specific result processor for this type."""

        try:
            return dialect._type_memos[self]["result"][coltype]
        except KeyError:
            pass

        # avoid KeyError context coming into result_processor() function
        # raises
        d = self._dialect_info(dialect)
        # key assumption: DBAPI type codes are
        # constants.  Else this dictionary would
        # grow unbounded.
        rp = d["impl"].result_processor(dialect, coltype)
        d["result"][coltype] = rp
        return rp

    def _cached_custom_processor(
        self, dialect: Dialect, key: str, fn: Callable[[TypeEngine[_T]], _O]
    ) -> _O:
        """return a dialect-specific processing object for
        custom purposes.

        The cx_Oracle dialect uses this at the moment.

        """
        try:
            return cast(_O, dialect._type_memos[self]["custom"][key])
        except KeyError:
            pass
        # avoid KeyError context coming into fn() function
        # raises
        d = self._dialect_info(dialect)
        impl = d["impl"]
        custom_dict = d.setdefault("custom", {})
        custom_dict[key] = result = fn(impl)
        return result

    def _dialect_info(self, dialect: Dialect) -> _TypeMemoDict:
        """Return a dialect-specific registry which
        caches a dialect-specific implementation, bind processing
        function, and one or more result processing functions."""

        if self in dialect._type_memos:
            return dialect._type_memos[self]
        else:
            impl = self._gen_dialect_impl(dialect)
            if impl is self:
                impl = self.adapt(type(self))
            # this can't be self, else we create a cycle
            assert impl is not self
            d: _TypeMemoDict = {"impl": impl, "result": {}}
            dialect._type_memos[self] = d
            return d

    def _gen_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        if dialect.name in self._variant_mapping:
            return self._variant_mapping[dialect.name]._gen_dialect_impl(
                dialect
            )
        else:
            return dialect.type_descriptor(self)

    @util.memoized_property
    def _static_cache_key(
        self,
    ) -> Union[CacheConst, Tuple[Any, ...]]:
        names = util.get_cls_kwargs(self.__class__)
        return (self.__class__,) + tuple(
            (
                k,
                (
                    self.__dict__[k]._static_cache_key
                    if isinstance(self.__dict__[k], TypeEngine)
                    else self.__dict__[k]
                ),
            )
            for k in names
            if k in self.__dict__
            and not k.startswith("_")
            and self.__dict__[k] is not None
        )

    @overload
    def adapt(self, cls: Type[_TE], **kw: Any) -> _TE: ...

    @overload
    def adapt(
        self, cls: Type[TypeEngineMixin], **kw: Any
    ) -> TypeEngine[Any]: ...

    def adapt(
        self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any
    ) -> TypeEngine[Any]:
        """Produce an "adapted" form of this type, given an "impl" class
        to work with.

        This method is used internally to associate generic
        types with "implementation" types that are specific to a particular
        dialect.
        """
        typ = util.constructor_copy(
            self, cast(Type[TypeEngine[Any]], cls), **kw
        )
        typ._variant_mapping = self._variant_mapping
        return typ

    def coerce_compared_value(
        self, op: Optional[OperatorType], value: Any
    ) -> TypeEngine[Any]:
        """Suggest a type for a 'coerced' Python value in an expression.

        Given an operator and value, gives the type a chance
        to return a type which the value should be coerced into.

        The default behavior here is conservative; if the right-hand
        side is already coerced into a SQL type based on its
        Python type, it is usually left alone.

        End-user functionality extension here should generally be via
        :class:`.TypeDecorator`, which provides more liberal behavior in that
        it defaults to coercing the other side of the expression into this
        type, thus applying special Python conversions above and beyond those
        needed by the DBAPI to both ides. It also provides the public method
        :meth:`.TypeDecorator.coerce_compared_value` which is intended for
        end-user customization of this behavior.

        """
        _coerced_type = _resolve_value_to_type(value)
        if (
            _coerced_type is NULLTYPE
            or _coerced_type._type_affinity is self._type_affinity
        ):
            return self
        else:
            return _coerced_type

    def _compare_type_affinity(self, other: TypeEngine[Any]) -> bool:
        return self._type_affinity is other._type_affinity

    def compile(self, dialect: Optional[Dialect] = None) -> str:
        """Produce a string-compiled form of this :class:`.TypeEngine`.

        When called with no arguments, uses a "default" dialect
        to produce a string result.

        :param dialect: a :class:`.Dialect` instance.

        """
        # arg, return value is inconsistent with
        # ClauseElement.compile()....this is a mistake.

        if dialect is None:
            dialect = self._default_dialect()

        return dialect.type_compiler_instance.process(self)

    @util.preload_module("sqlalchemy.engine.default")
    def _default_dialect(self) -> Dialect:
        default = util.preloaded.engine_default

        # dmypy / mypy seems to sporadically keep thinking this line is
        # returning Any, which seems to be caused by the @deprecated_params
        # decorator on the DefaultDialect constructor
        return default.StrCompileDialect()  # type: ignore

    def __str__(self) -> str:
        return str(self.compile())

    def __repr__(self) -> str:
        return util.generic_repr(self)


class TypeEngineMixin:
    """classes which subclass this can act as "mixin" classes for
    TypeEngine."""

    __slots__ = ()

    if TYPE_CHECKING:

        @util.memoized_property
        def _static_cache_key(
            self,
        ) -> Union[CacheConst, Tuple[Any, ...]]: ...

        @overload
        def adapt(self, cls: Type[_TE], **kw: Any) -> _TE: ...

        @overload
        def adapt(
            self, cls: Type[TypeEngineMixin], **kw: Any
        ) -> TypeEngine[Any]: ...

        def adapt(
            self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any
        ) -> TypeEngine[Any]: ...

        def dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]: ...


class ExternalType(TypeEngineMixin):
    """mixin that defines attributes and behaviors specific to third-party
    datatypes.

    "Third party" refers to datatypes that are defined outside the scope
    of SQLAlchemy within either end-user application code or within
    external extensions to SQLAlchemy.

    Subclasses currently include :class:`.TypeDecorator` and
    :class:`.UserDefinedType`.

    .. versionadded:: 1.4.28

    """

    cache_ok: Optional[bool] = None
    '''Indicate if statements using this :class:`.ExternalType` are "safe to
    cache".

    The default value ``None`` will emit a warning and then not allow caching
    of a statement which includes this type.   Set to ``False`` to disable
    statements using this type from being cached at all without a warning.
    When set to ``True``, the object's class and selected elements from its
    state will be used as part of the cache key.  For example, using a
    :class:`.TypeDecorator`::

        class MyType(TypeDecorator):
            impl = String

            cache_ok = True

            def __init__(self, choices):
                self.choices = tuple(choices)
                self.internal_only = True

    The cache key for the above type would be equivalent to::

        >>> MyType(["a", "b", "c"])._static_cache_key
        (<class '__main__.MyType'>, ('choices', ('a', 'b', 'c')))

    The caching scheme will extract attributes from the type that correspond
    to the names of parameters in the ``__init__()`` method.  Above, the
    "choices" attribute becomes part of the cache key but "internal_only"
    does not, because there is no parameter named "internal_only".

    The requirements for cacheable elements is that they are hashable
    and also that they indicate the same SQL rendered for expressions using
    this type every time for a given cache value.

    To accommodate for datatypes that refer to unhashable structures such
    as dictionaries, sets and lists, these objects can be made "cacheable"
    by assigning hashable structures to the attributes whose names
    correspond with the names of the arguments.  For example, a datatype
    which accepts a dictionary of lookup values may publish this as a sorted
    series of tuples.   Given a previously un-cacheable type as::

        class LookupType(UserDefinedType):
            """a custom type that accepts a dictionary as a parameter.

            this is the non-cacheable version, as "self.lookup" is not
            hashable.

            """

            def __init__(self, lookup):
                self.lookup = lookup

            def get_col_spec(self, **kw):
                return "VARCHAR(255)"

            def bind_processor(self, dialect): ...  # works with "self.lookup" ...

    Where "lookup" is a dictionary.  The type will not be able to generate
    a cache key::

        >>> type_ = LookupType({"a": 10, "b": 20})
        >>> type_._static_cache_key
        <stdin>:1: SAWarning: UserDefinedType LookupType({'a': 10, 'b': 20}) will not
        produce a cache key because the ``cache_ok`` flag is not set to True.
        Set this flag to True if this type object's state is safe to use
        in a cache key, or False to disable this warning.
        symbol('no_cache')

    If we **did** set up such a cache key, it wouldn't be usable. We would
    get a tuple structure that contains a dictionary inside of it, which
    cannot itself be used as a key in a "cache dictionary" such as SQLAlchemy's
    statement cache, since Python dictionaries aren't hashable::

        >>> # set cache_ok = True
        >>> type_.cache_ok = True

        >>> # this is the cache key it would generate
        >>> key = type_._static_cache_key
        >>> key
        (<class '__main__.LookupType'>, ('lookup', {'a': 10, 'b': 20}))

        >>> # however this key is not hashable, will fail when used with
        >>> # SQLAlchemy statement cache
        >>> some_cache = {key: "some sql value"}
        Traceback (most recent call last): File "<stdin>", line 1,
        in <module> TypeError: unhashable type: 'dict'

    The type may be made cacheable by assigning a sorted tuple of tuples
    to the ".lookup" attribute::

        class LookupType(UserDefinedType):
            """a custom type that accepts a dictionary as a parameter.

            The dictionary is stored both as itself in a private variable,
            and published in a public variable as a sorted tuple of tuples,
            which is hashable and will also return the same value for any
            two equivalent dictionaries.  Note it assumes the keys and
            values of the dictionary are themselves hashable.

            """

            cache_ok = True

            def __init__(self, lookup):
                self._lookup = lookup

                # assume keys/values of "lookup" are hashable; otherwise
                # they would also need to be converted in some way here
                self.lookup = tuple((key, lookup[key]) for key in sorted(lookup))

            def get_col_spec(self, **kw):
                return "VARCHAR(255)"

            def bind_processor(self, dialect): ...  # works with "self._lookup" ...

    Where above, the cache key for ``LookupType({"a": 10, "b": 20})`` will be::

        >>> LookupType({"a": 10, "b": 20})._static_cache_key
        (<class '__main__.LookupType'>, ('lookup', (('a', 10), ('b', 20))))

    .. versionadded:: 1.4.14 - added the ``cache_ok`` flag to allow
       some configurability of caching for :class:`.TypeDecorator` classes.

    .. versionadded:: 1.4.28 - added the :class:`.ExternalType` mixin which
       generalizes the ``cache_ok`` flag to both the :class:`.TypeDecorator`
       and :class:`.UserDefinedType` classes.

    .. seealso::

        :ref:`sql_caching`

    '''  # noqa: E501

    @util.non_memoized_property
    def _static_cache_key(
        self,
    ) -> Union[CacheConst, Tuple[Any, ...]]:
        cache_ok = self.__class__.__dict__.get("cache_ok", None)

        if cache_ok is None:
            for subtype in self.__class__.__mro__:
                if ExternalType in subtype.__bases__:
                    break
            else:
                subtype = self.__class__.__mro__[1]

            util.warn(
                "%s %r will not produce a cache key because "
                "the ``cache_ok`` attribute is not set to True.  This can "
                "have significant performance implications including some "
                "performance degradations in comparison to prior SQLAlchemy "
                "versions.  Set this attribute to True if this type object's "
                "state is safe to use in a cache key, or False to "
                "disable this warning." % (subtype.__name__, self),
                code="cprf",
            )
        elif cache_ok is True:
            return super()._static_cache_key

        return NO_CACHE


class UserDefinedType(
    ExternalType, TypeEngineMixin, TypeEngine[_T], util.EnsureKWArg
):
    """Base for user defined types.

    This should be the base of new types.  Note that
    for most cases, :class:`.TypeDecorator` is probably
    more appropriate::

      import sqlalchemy.types as types


      class MyType(types.UserDefinedType):
          cache_ok = True

          def __init__(self, precision=8):
              self.precision = precision

          def get_col_spec(self, **kw):
              return "MYTYPE(%s)" % self.precision

          def bind_processor(self, dialect):
              def process(value):
                  return value

              return process

          def result_processor(self, dialect, coltype):
              def process(value):
                  return value

              return process

    Once the type is made, it's immediately usable::

      table = Table(
          "foo",
          metadata_obj,
          Column("id", Integer, primary_key=True),
          Column("data", MyType(16)),
      )

    The ``get_col_spec()`` method will in most cases receive a keyword
    argument ``type_expression`` which refers to the owning expression
    of the type as being compiled, such as a :class:`_schema.Column` or
    :func:`.cast` construct.  This keyword is only sent if the method
    accepts keyword arguments (e.g. ``**kw``) in its argument signature;
    introspection is used to check for this in order to support legacy
    forms of this function.

    The :attr:`.UserDefinedType.cache_ok` class-level flag indicates if this
    custom :class:`.UserDefinedType` is safe to be used as part of a cache key.
    This flag defaults to ``None`` which will initially generate a warning
    when the SQL compiler attempts to generate a cache key for a statement
    that uses this type.  If the :class:`.UserDefinedType` is not guaranteed
    to produce the same bind/result behavior and SQL generation
    every time, this flag should be set to ``False``; otherwise if the
    class produces the same behavior each time, it may be set to ``True``.
    See :attr:`.UserDefinedType.cache_ok` for further notes on how this works.

    .. versionadded:: 1.4.28 Generalized the :attr:`.ExternalType.cache_ok`
       flag so that it is available for both :class:`.TypeDecorator` as well
       as :class:`.UserDefinedType`.

    """

    __visit_name__ = "user_defined"

    ensure_kwarg = "get_col_spec"

    def coerce_compared_value(
        self, op: Optional[OperatorType], value: Any
    ) -> TypeEngine[Any]:
        """Suggest a type for a 'coerced' Python value in an expression.

        Default behavior for :class:`.UserDefinedType` is the
        same as that of :class:`.TypeDecorator`; by default it returns
        ``self``, assuming the compared value should be coerced into
        the same type as this one.  See
        :meth:`.TypeDecorator.coerce_compared_value` for more detail.

        """

        return self

    if TYPE_CHECKING:

        def get_col_spec(self, **kw: Any) -> str: ...


class Emulated(TypeEngineMixin):
    """Mixin for base types that emulate the behavior of a DB-native type.

    An :class:`.Emulated` type will use an available database type
    in conjunction with Python-side routines and/or database constraints
    in order to approximate the behavior of a database type that is provided
    natively by some backends.  When a native-providing backend is in
    use, the native version of the type is used.  This native version
    should include the :class:`.NativeForEmulated` mixin to allow it to be
    distinguished from :class:`.Emulated`.

    Current examples of :class:`.Emulated` are:  :class:`.Interval`,
    :class:`.Enum`, :class:`.Boolean`.

    .. versionadded:: 1.2.0b3

    """

    native: bool

    def adapt_to_emulated(
        self,
        impltype: Type[Union[TypeEngine[Any], TypeEngineMixin]],
        **kw: Any,
    ) -> TypeEngine[Any]:
        """Given an impl class, adapt this type to the impl assuming
        "emulated".

        The impl should also be an "emulated" version of this type,
        most likely the same class as this type itself.

        e.g.: sqltypes.Enum adapts to the Enum class.

        """
        return super().adapt(impltype, **kw)

    @overload
    def adapt(self, cls: Type[_TE], **kw: Any) -> _TE: ...

    @overload
    def adapt(
        self, cls: Type[TypeEngineMixin], **kw: Any
    ) -> TypeEngine[Any]: ...

    def adapt(
        self, cls: Type[Union[TypeEngine[Any], TypeEngineMixin]], **kw: Any
    ) -> TypeEngine[Any]:
        if _is_native_for_emulated(cls):
            if self.native:
                # native support requested, dialect gave us a native
                # implementor, pass control over to it
                return cls.adapt_emulated_to_native(self, **kw)
            else:
                # non-native support, let the native implementor
                # decide also, at the moment this is just to help debugging
                # as only the default logic is implemented.
                return cls.adapt_native_to_emulated(self, **kw)
        else:
            # this would be, both classes are Enum, or both classes
            # are postgresql.ENUM
            if issubclass(cls, self.__class__):
                return self.adapt_to_emulated(cls, **kw)
            else:
                return super().adapt(cls, **kw)


def _is_native_for_emulated(
    typ: Type[Union[TypeEngine[Any], TypeEngineMixin]],
) -> TypeGuard[Type[NativeForEmulated]]:
    return hasattr(typ, "adapt_emulated_to_native")


class NativeForEmulated(TypeEngineMixin):
    """Indicates DB-native types supported by an :class:`.Emulated` type.

    .. versionadded:: 1.2.0b3

    """

    @classmethod
    def adapt_native_to_emulated(
        cls,
        impl: Union[TypeEngine[Any], TypeEngineMixin],
        **kw: Any,
    ) -> TypeEngine[Any]:
        """Given an impl, adapt this type's class to the impl assuming
        "emulated".


        """
        impltype = impl.__class__
        return impl.adapt(impltype, **kw)

    @classmethod
    def adapt_emulated_to_native(
        cls,
        impl: Union[TypeEngine[Any], TypeEngineMixin],
        **kw: Any,
    ) -> TypeEngine[Any]:
        """Given an impl, adapt this type's class to the impl assuming
        "native".

        The impl will be an :class:`.Emulated` class but not a
        :class:`.NativeForEmulated`.

        e.g.: postgresql.ENUM produces a type given an Enum instance.

        """

        # dmypy seems to crash on this
        return cls(**kw)  # type: ignore

    # dmypy seems to crash with this, on repeated runs with changes
    # if TYPE_CHECKING:
    #    def __init__(self, **kw: Any):
    #        ...


class TypeDecorator(SchemaEventTarget, ExternalType, TypeEngine[_T]):
    '''Allows the creation of types which add additional functionality
    to an existing type.

    This method is preferred to direct subclassing of SQLAlchemy's
    built-in types as it ensures that all required functionality of
    the underlying type is kept in place.

    Typical usage::

      import sqlalchemy.types as types


      class MyType(types.TypeDecorator):
          """Prefixes Unicode values with "PREFIX:" on the way in and
          strips it off on the way out.
          """

          impl = types.Unicode

          cache_ok = True

          def process_bind_param(self, value, dialect):
              return "PREFIX:" + value

          def process_result_value(self, value, dialect):
              return value[7:]

          def copy(self, **kw):
              return MyType(self.impl.length)

    The class-level ``impl`` attribute is required, and can reference any
    :class:`.TypeEngine` class.  Alternatively, the :meth:`load_dialect_impl`
    method can be used to provide different type classes based on the dialect
    given; in this case, the ``impl`` variable can reference
    ``TypeEngine`` as a placeholder.

    The :attr:`.TypeDecorator.cache_ok` class-level flag indicates if this
    custom :class:`.TypeDecorator` is safe to be used as part of a cache key.
    This flag defaults to ``None`` which will initially generate a warning
    when the SQL compiler attempts to generate a cache key for a statement
    that uses this type.  If the :class:`.TypeDecorator` is not guaranteed
    to produce the same bind/result behavior and SQL generation
    every time, this flag should be set to ``False``; otherwise if the
    class produces the same behavior each time, it may be set to ``True``.
    See :attr:`.TypeDecorator.cache_ok` for further notes on how this works.

    Types that receive a Python type that isn't similar to the ultimate type
    used may want to define the :meth:`TypeDecorator.coerce_compared_value`
    method. This is used to give the expression system a hint when coercing
    Python objects into bind parameters within expressions. Consider this
    expression::

        mytable.c.somecol + datetime.date(2009, 5, 15)

    Above, if "somecol" is an ``Integer`` variant, it makes sense that
    we're doing date arithmetic, where above is usually interpreted
    by databases as adding a number of days to the given date.
    The expression system does the right thing by not attempting to
    coerce the "date()" value into an integer-oriented bind parameter.

    However, in the case of ``TypeDecorator``, we are usually changing an
    incoming Python type to something new - ``TypeDecorator`` by default will
    "coerce" the non-typed side to be the same type as itself. Such as below,
    we define an "epoch" type that stores a date value as an integer::

        class MyEpochType(types.TypeDecorator):
            impl = types.Integer

            cache_ok = True

            epoch = datetime.date(1970, 1, 1)

            def process_bind_param(self, value, dialect):
                return (value - self.epoch).days

            def process_result_value(self, value, dialect):
                return self.epoch + timedelta(days=value)

    Our expression of ``somecol + date`` with the above type will coerce the
    "date" on the right side to also be treated as ``MyEpochType``.

    This behavior can be overridden via the
    :meth:`~TypeDecorator.coerce_compared_value` method, which returns a type
    that should be used for the value of the expression. Below we set it such
    that an integer value will be treated as an ``Integer``, and any other
    value is assumed to be a date and will be treated as a ``MyEpochType``::

        def coerce_compared_value(self, op, value):
            if isinstance(value, int):
                return Integer()
            else:
                return self

    .. warning::

       Note that the **behavior of coerce_compared_value is not inherited
       by default from that of the base type**.
       If the :class:`.TypeDecorator` is augmenting a
       type that requires special logic for certain types of operators,
       this method **must** be overridden.  A key example is when decorating
       the :class:`_postgresql.JSON` and :class:`_postgresql.JSONB` types;
       the default rules of :meth:`.TypeEngine.coerce_compared_value` should
       be used in order to deal with operators like index operations::

            from sqlalchemy import JSON
            from sqlalchemy import TypeDecorator


            class MyJsonType(TypeDecorator):
                impl = JSON

                cache_ok = True

                def coerce_compared_value(self, op, value):
                    return self.impl.coerce_compared_value(op, value)

       Without the above step, index operations such as ``mycol['foo']``
       will cause the index value ``'foo'`` to be JSON encoded.

       Similarly, when working with the :class:`.ARRAY` datatype, the
       type coercion for index operations (e.g. ``mycol[5]``) is also
       handled by :meth:`.TypeDecorator.coerce_compared_value`, where
       again a simple override is sufficient unless special rules are needed
       for particular operators::

            from sqlalchemy import ARRAY
            from sqlalchemy import TypeDecorator


            class MyArrayType(TypeDecorator):
                impl = ARRAY

                cache_ok = True

                def coerce_compared_value(self, op, value):
                    return self.impl.coerce_compared_value(op, value)

    '''

    __visit_name__ = "type_decorator"

    _is_type_decorator = True

    # this is that pattern I've used in a few places (Dialect.dbapi,
    # Dialect.type_compiler) where the "cls.attr" is a class to make something,
    # and "instance.attr" is an instance of that thing.  It's such a nifty,
    # great pattern, and there is zero chance Python typing tools will ever be
    # OK with it.  For TypeDecorator.impl, this is a highly public attribute so
    # we really can't change its behavior without a major deprecation routine.
    impl: Union[TypeEngine[Any], Type[TypeEngine[Any]]]

    # we are changing its behavior *slightly*, which is that we now consume
    # the instance level version from this memoized property instead, so you
    # can't reassign "impl" on an existing TypeDecorator that's already been
    # used (something one shouldn't do anyway) without also updating
    # impl_instance.
    @util.memoized_property
    def impl_instance(self) -> TypeEngine[Any]:
        return self.impl  # type: ignore

    def __init__(self, *args: Any, **kwargs: Any):
        """Construct a :class:`.TypeDecorator`.

        Arguments sent here are passed to the constructor
        of the class assigned to the ``impl`` class level attribute,
        assuming the ``impl`` is a callable, and the resulting
        object is assigned to the ``self.impl`` instance attribute
        (thus overriding the class attribute of the same name).

        If the class level ``impl`` is not a callable (the unusual case),
        it will be assigned to the same instance attribute 'as-is',
        ignoring those arguments passed to the constructor.

        Subclasses can override this to customize the generation
        of ``self.impl`` entirely.

        """

        if not hasattr(self.__class__, "impl"):
            raise AssertionError(
                "TypeDecorator implementations "
                "require a class-level variable "
                "'impl' which refers to the class of "
                "type being decorated"
            )

        self.impl = to_instance(self.__class__.impl, *args, **kwargs)

    coerce_to_is_types: Sequence[Type[Any]] = (type(None),)
    """Specify those Python types which should be coerced at the expression
    level to "IS <constant>" when compared using ``==`` (and same for
    ``IS NOT`` in conjunction with ``!=``).

    For most SQLAlchemy types, this includes ``NoneType``, as well as
    ``bool``.

    :class:`.TypeDecorator` modifies this list to only include ``NoneType``,
    as typedecorator implementations that deal with boolean types are common.

    Custom :class:`.TypeDecorator` classes can override this attribute to
    return an empty tuple, in which case no values will be coerced to
    constants.

    """

    class Comparator(TypeEngine.Comparator[_CT]):
        """A :class:`.TypeEngine.Comparator` that is specific to
        :class:`.TypeDecorator`.

        User-defined :class:`.TypeDecorator` classes should not typically
        need to modify this.


        """

        __slots__ = ()

        def operate(
            self, op: OperatorType, *other: Any, **kwargs: Any
        ) -> ColumnElement[_CT]:
            if TYPE_CHECKING:
                assert isinstance(self.expr.type, TypeDecorator)
            kwargs["_python_is_types"] = self.expr.type.coerce_to_is_types
            return super().operate(op, *other, **kwargs)

        def reverse_operate(
            self, op: OperatorType, other: Any, **kwargs: Any
        ) -> ColumnElement[_CT]:
            if TYPE_CHECKING:
                assert isinstance(self.expr.type, TypeDecorator)
            kwargs["_python_is_types"] = self.expr.type.coerce_to_is_types
            return super().reverse_operate(op, other, **kwargs)

    @staticmethod
    def _reduce_td_comparator(
        impl: TypeEngine[Any], expr: ColumnElement[_T]
    ) -> Any:
        return TypeDecorator._create_td_comparator_type(impl)(expr)

    @staticmethod
    def _create_td_comparator_type(
        impl: TypeEngine[Any],
    ) -> _ComparatorFactory[Any]:

        def __reduce__(self: TypeDecorator.Comparator[Any]) -> Any:
            return (TypeDecorator._reduce_td_comparator, (impl, self.expr))

        return type(
            "TDComparator",
            (TypeDecorator.Comparator, impl.comparator_factory),  # type: ignore # noqa: E501
            {"__reduce__": __reduce__},
        )

    @property
    def comparator_factory(  # type: ignore  # mypy properties bug
        self,
    ) -> _ComparatorFactory[Any]:
        if TypeDecorator.Comparator in self.impl.comparator_factory.__mro__:  # type: ignore # noqa: E501
            return self.impl_instance.comparator_factory
        else:
            # reconcile the Comparator class on the impl with that
            # of TypeDecorator.
            # the use of multiple staticmethods is to support repeated
            # pickling of the Comparator itself
            return TypeDecorator._create_td_comparator_type(self.impl_instance)

    def _copy_with_check(self) -> Self:
        tt = self.copy()
        if not isinstance(tt, self.__class__):
            raise AssertionError(
                "Type object %s does not properly "
                "implement the copy() method, it must "
                "return an object of type %s" % (self, self.__class__)
            )
        return tt

    def _gen_dialect_impl(self, dialect: Dialect) -> TypeEngine[_T]:
        if dialect.name in self._variant_mapping:
            adapted = dialect.type_descriptor(
                self._variant_mapping[dialect.name]
            )
        else:
            adapted = dialect.type_descriptor(self)
        if adapted is not self:
            return adapted

        # otherwise adapt the impl type, link
        # to a copy of this TypeDecorator and return
        # that.
        typedesc = self.load_dialect_impl(dialect).dialect_impl(dialect)
        tt = self._copy_with_check()
        tt.impl = tt.impl_instance = typedesc
        return tt

    def _with_collation(self, collation: str) -> Self:
        tt = self._copy_with_check()
        tt.impl = tt.impl_instance = self.impl_instance._with_collation(
            collation
        )
        return tt

    @util.ro_non_memoized_property
    def _type_affinity(self) -> Optional[Type[TypeEngine[Any]]]:
        return self.impl_instance._type_affinity

    def _set_parent(
        self, parent: SchemaEventTarget, outer: bool = False, **kw: Any
    ) -> None:
        """Support SchemaEventTarget"""

        super()._set_parent(parent)

        if not outer and isinstance(self.impl_instance, SchemaEventTarget):
            self.impl_instance._set_parent(parent, outer=False, **kw)

    def _set_parent_with_dispatch(
        self, parent: SchemaEventTarget, **kw: Any
    ) -> None:
        """Support SchemaEventTarget"""

        super()._set_parent_with_dispatch(parent, outer=True, **kw)

        if isinstance(self.impl_instance, SchemaEventTarget):
            self.impl_instance._set_parent_with_dispatch(parent)

    def type_engine(self, dialect: Dialect) -> TypeEngine[Any]:
        """Return a dialect-specific :class:`.TypeEngine` instance
        for this :class:`.TypeDecorator`.

        In most cases this returns a dialect-adapted form of
        the :class:`.TypeEngine` type represented by ``self.impl``.
        Makes usage of :meth:`dialect_impl`.
        Behavior can be customized here by overriding
        :meth:`load_dialect_impl`.

        """
        adapted = dialect.type_descriptor(self)
        if not isinstance(adapted, type(self)):
            return adapted
        else:
            return self.load_dialect_impl(dialect)

    def load_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        """Return a :class:`.TypeEngine` object corresponding to a dialect.

        This is an end-user override hook that can be used to provide
        differing types depending on the given dialect.  It is used
        by the :class:`.TypeDecorator` implementation of :meth:`type_engine`
        to help determine what type should ultimately be returned
        for a given :class:`.TypeDecorator`.

        By default returns ``self.impl``.

        """
        return self.impl_instance

    def _unwrapped_dialect_impl(self, dialect: Dialect) -> TypeEngine[Any]:
        """Return the 'unwrapped' dialect impl for this type.

        This is used by the :meth:`.DefaultDialect.set_input_sizes`
        method.

        """
        # some dialects have a lookup for a TypeDecorator subclass directly.
        # postgresql.INTERVAL being the main example
        typ = self.dialect_impl(dialect)

        # if we are still a type decorator, load the per-dialect switch
        # (such as what Variant uses), then get the dialect impl for that.
        if isinstance(typ, self.__class__):
            return typ.load_dialect_impl(dialect).dialect_impl(dialect)
        else:
            return typ

    def __getattr__(self, key: str) -> Any:
        """Proxy all other undefined accessors to the underlying
        implementation."""
        return getattr(self.impl_instance, key)

    def process_literal_param(
        self, value: Optional[_T], dialect: Dialect
    ) -> str:
        """Receive a literal parameter value to be rendered inline within
        a statement.

        .. note::

            This method is called during the **SQL compilation** phase of a
            statement, when rendering a SQL string. Unlike other SQL
            compilation methods, it is passed a specific Python value to be
            rendered as a string. However it should not be confused with the
            :meth:`_types.TypeDecorator.process_bind_param` method, which is
            the more typical method that processes the actual value passed to a
            particular parameter at statement execution time.

        Custom subclasses of :class:`_types.TypeDecorator` should override
        this method to provide custom behaviors for incoming data values
        that are in the special case of being rendered as literals.

        The returned string will be rendered into the output string.

        """
        raise NotImplementedError()

    def process_bind_param(self, value: Optional[_T], dialect: Dialect) -> Any:
        """Receive a bound parameter value to be converted.

        Custom subclasses of :class:`_types.TypeDecorator` should override
        this method to provide custom behaviors for incoming data values.
        This method is called at **statement execution time** and is passed
        the literal Python data value which is to be associated with a bound
        parameter in the statement.

        The operation could be anything desired to perform custom
        behavior, such as transforming or serializing data.
        This could also be used as a hook for validating logic.

        :param value: Data to operate upon, of any type expected by
         this method in the subclass.  Can be ``None``.
        :param dialect: the :class:`.Dialect` in use.

        .. seealso::

            :ref:`types_typedecorator`

            :meth:`_types.TypeDecorator.process_result_value`

        """

        raise NotImplementedError()

    def process_result_value(
        self, value: Optional[Any], dialect: Dialect
    ) -> Optional[_T]:
        """Receive a result-row column value to be converted.

        Custom subclasses of :class:`_types.TypeDecorator` should override
        this method to provide custom behaviors for data values
        being received in result rows coming from the database.
        This method is called at **result fetching time** and is passed
        the literal Python data value that's extracted from a database result
        row.

        The operation could be anything desired to perform custom
        behavior, such as transforming or deserializing data.

        :param value: Data to operate upon, of any type expected by
         this method in the subclass.  Can be ``None``.
        :param dialect: the :class:`.Dialect` in use.

        .. seealso::

            :ref:`types_typedecorator`

            :meth:`_types.TypeDecorator.process_bind_param`


        """

        raise NotImplementedError()

    @util.memoized_property
    def _has_bind_processor(self) -> bool:
        """memoized boolean, check if process_bind_param is implemented.

        Allows the base process_bind_param to raise
        NotImplementedError without needing to test an expensive
        exception throw.

        """

        return util.method_is_overridden(
            self, TypeDecorator.process_bind_param
        )

    @util.memoized_property
    def _has_literal_processor(self) -> bool:
        """memoized boolean, check if process_literal_param is implemented."""

        return util.method_is_overridden(
            self, TypeDecorator.process_literal_param
        )

    def literal_processor(
        self, dialect: Dialect
    ) -> Optional[_LiteralProcessorType[_T]]:
        """Provide a literal processing function for the given
        :class:`.Dialect`.

        This is the method that fulfills the :class:`.TypeEngine`
        contract for literal value conversion which normally occurs via
        the :meth:`_types.TypeEngine.literal_processor` method.

        .. note::

            User-defined subclasses of :class:`_types.TypeDecorator` should
            **not** implement this method, and should instead implement
            :meth:`_types.TypeDecorator.process_literal_param` so that the
            "inner" processing provided by the implementing type is maintained.

        """

        if self._has_literal_processor:
            process_literal_param = self.process_literal_param
            process_bind_param = None
        elif self._has_bind_processor:
            # use the bind processor if dont have a literal processor,
            # but we have an impl literal processor
            process_literal_param = None
            process_bind_param = self.process_bind_param
        else:
            process_literal_param = None
            process_bind_param = None

        if process_literal_param is not None:
            impl_processor = self.impl_instance.literal_processor(dialect)
            if impl_processor:
                fixed_impl_processor = impl_processor
                fixed_process_literal_param = process_literal_param

                def process(value: Any) -> str:
                    return fixed_impl_processor(
                        fixed_process_literal_param(value, dialect)
                    )

            else:
                fixed_process_literal_param = process_literal_param

                def process(value: Any) -> str:
                    return fixed_process_literal_param(value, dialect)

            return process

        elif process_bind_param is not None:
            impl_processor = self.impl_instance.literal_processor(dialect)
            if not impl_processor:
                return None
            else:
                fixed_impl_processor = impl_processor
                fixed_process_bind_param = process_bind_param

                def process(value: Any) -> str:
                    return fixed_impl_processor(
                        fixed_process_bind_param(value, dialect)
                    )

                return process
        else:
            return self.impl_instance.literal_processor(dialect)

    def bind_processor(
        self, dialect: Dialect
    ) -> Optional[_BindProcessorType[_T]]:
        """Provide a bound value processing function for the
        given :class:`.Dialect`.

        This is the method that fulfills the :class:`.TypeEngine`
        contract for bound value conversion which normally occurs via
        the :meth:`_types.TypeEngine.bind_processor` method.

        .. note::

            User-defined subclasses of :class:`_types.TypeDecorator` should
            **not** implement this method, and should instead implement
            :meth:`_types.TypeDecorator.process_bind_param` so that the "inner"
            processing provided by the implementing type is maintained.

        :param dialect: Dialect instance in use.

        """
        if self._has_bind_processor:
            process_param = self.process_bind_param
            impl_processor = self.impl_instance.bind_processor(dialect)
            if impl_processor:
                fixed_impl_processor = impl_processor
                fixed_process_param = process_param

                def process(value: Optional[_T]) -> Any:
                    return fixed_impl_processor(
                        fixed_process_param(value, dialect)
                    )

            else:
                fixed_process_param = process_param

                def process(value: Optional[_T]) -> Any:
                    return fixed_process_param(value, dialect)

            return process
        else:
            return self.impl_instance.bind_processor(dialect)

    @util.memoized_property
    def _has_result_processor(self) -> bool:
        """memoized boolean, check if process_result_value is implemented.

        Allows the base process_result_value to raise
        NotImplementedError without needing to test an expensive
        exception throw.

        """

        return util.method_is_overridden(
            self, TypeDecorator.process_result_value
        )

    def result_processor(
        self, dialect: Dialect, coltype: Any
    ) -> Optional[_ResultProcessorType[_T]]:
        """Provide a result value processing function for the given
        :class:`.Dialect`.

        This is the method that fulfills the :class:`.TypeEngine`
        contract for bound value conversion which normally occurs via
        the :meth:`_types.TypeEngine.result_processor` method.

        .. note::

            User-defined subclasses of :class:`_types.TypeDecorator` should
            **not** implement this method, and should instead implement
            :meth:`_types.TypeDecorator.process_result_value` so that the
            "inner" processing provided by the implementing type is maintained.

        :param dialect: Dialect instance in use.
        :param coltype: A SQLAlchemy data type

        """
        if self._has_result_processor:
            process_value = self.process_result_value
            impl_processor = self.impl_instance.result_processor(
                dialect, coltype
            )
            if impl_processor:
                fixed_process_value = process_value
                fixed_impl_processor = impl_processor

                def process(value: Any) -> Optional[_T]:
                    return fixed_process_value(
                        fixed_impl_processor(value), dialect
                    )

            else:
                fixed_process_value = process_value

                def process(value: Any) -> Optional[_T]:
                    return fixed_process_value(value, dialect)

            return process
        else:
            return self.impl_instance.result_processor(dialect, coltype)

    @util.memoized_property
    def _has_bind_expression(self) -> bool:
        return (
            util.method_is_overridden(self, TypeDecorator.bind_expression)
            or self.impl_instance._has_bind_expression
        )

    def bind_expression(
        self, bindparam: BindParameter[_T]
    ) -> Optional[ColumnElement[_T]]:
        """Given a bind value (i.e. a :class:`.BindParameter` instance),
        return a SQL expression which will typically wrap the given parameter.

        .. note::

            This method is called during the **SQL compilation** phase of a
            statement, when rendering a SQL string. It is **not** necessarily
            called against specific values, and should not be confused with the
            :meth:`_types.TypeDecorator.process_bind_param` method, which is
            the more typical method that processes the actual value passed to a
            particular parameter at statement execution time.

        Subclasses of :class:`_types.TypeDecorator` can override this method
        to provide custom bind expression behavior for the type.  This
        implementation will **replace** that of the underlying implementation
        type.

        """
        return self.impl_instance.bind_expression(bindparam)

    @util.memoized_property
    def _has_column_expression(self) -> bool:
        """memoized boolean, check if column_expression is implemented.

        Allows the method to be skipped for the vast majority of expression
        types that don't use this feature.

        """

        return (
            util.method_is_overridden(self, TypeDecorator.column_expression)
            or self.impl_instance._has_column_expression
        )

    def column_expression(
        self, column: ColumnElement[_T]
    ) -> Optional[ColumnElement[_T]]:
        """Given a SELECT column expression, return a wrapping SQL expression.

        .. note::

            This method is called during the **SQL compilation** phase of a
            statement, when rendering a SQL string. It is **not** called
            against specific values, and should not be confused with the
            :meth:`_types.TypeDecorator.process_result_value` method, which is
            the more typical method that processes the actual value returned
            in a result row subsequent to statement execution time.

        Subclasses of :class:`_types.TypeDecorator` can override this method
        to provide custom column expression behavior for the type.  This
        implementation will **replace** that of the underlying implementation
        type.

        See the description of :meth:`_types.TypeEngine.column_expression`
        for a complete description of the method's use.

        """

        return self.impl_instance.column_expression(column)

    def coerce_compared_value(
        self, op: Optional[OperatorType], value: Any
    ) -> Any:
        """Suggest a type for a 'coerced' Python value in an expression.

        By default, returns self.   This method is called by
        the expression system when an object using this type is
        on the left or right side of an expression against a plain Python
        object which does not yet have a SQLAlchemy type assigned::

            expr = table.c.somecolumn + 35

        Where above, if ``somecolumn`` uses this type, this method will
        be called with the value ``operator.add``
        and ``35``.  The return value is whatever SQLAlchemy type should
        be used for ``35`` for this particular operation.

        """
        return self

    def copy(self, **kw: Any) -> Self:
        """Produce a copy of this :class:`.TypeDecorator` instance.

        This is a shallow copy and is provided to fulfill part of
        the :class:`.TypeEngine` contract.  It usually does not
        need to be overridden unless the user-defined :class:`.TypeDecorator`
        has local state that should be deep-copied.

        """

        instance = self.__class__.__new__(self.__class__)
        instance.__dict__.update(self.__dict__)
        return instance

    def get_dbapi_type(self, dbapi: ModuleType) -> Optional[Any]:
        """Return the DBAPI type object represented by this
        :class:`.TypeDecorator`.

        By default this calls upon :meth:`.TypeEngine.get_dbapi_type` of the
        underlying "impl".
        """
        return self.impl_instance.get_dbapi_type(dbapi)

    def compare_values(self, x: Any, y: Any) -> bool:
        """Given two values, compare them for equality.

        By default this calls upon :meth:`.TypeEngine.compare_values`
        of the underlying "impl", which in turn usually
        uses the Python equals operator ``==``.

        This function is used by the ORM to compare
        an original-loaded value with an intercepted
        "changed" value, to determine if a net change
        has occurred.

        """
        return self.impl_instance.compare_values(x, y)

    # mypy property bug
    @property
    def sort_key_function(self) -> Optional[Callable[[Any], Any]]:  # type: ignore # noqa: E501
        return self.impl_instance.sort_key_function

    def __repr__(self) -> str:
        return util.generic_repr(self, to_inspect=self.impl_instance)


class Variant(TypeDecorator[_T]):
    """deprecated.  symbol is present for backwards-compatibility with
    workaround recipes, however this actual type should not be used.

    """

    def __init__(self, *arg: Any, **kw: Any):
        raise NotImplementedError(
            "Variant is no longer used in SQLAlchemy; this is a "
            "placeholder symbol for backwards compatibility."
        )


@overload
def to_instance(
    typeobj: Union[Type[_TE], _TE], *arg: Any, **kw: Any
) -> _TE: ...


@overload
def to_instance(typeobj: None, *arg: Any, **kw: Any) -> TypeEngine[None]: ...


def to_instance(
    typeobj: Union[Type[_TE], _TE, None], *arg: Any, **kw: Any
) -> Union[_TE, TypeEngine[None]]:
    if typeobj is None:
        return NULLTYPE

    if callable(typeobj):
        return typeobj(*arg, **kw)
    else:
        return typeobj


def adapt_type(
    typeobj: _TypeEngineArgument[Any],
    colspecs: Mapping[Type[Any], Type[TypeEngine[Any]]],
) -> TypeEngine[Any]:
    typeobj = to_instance(typeobj)
    for t in typeobj.__class__.__mro__[0:-1]:
        try:
            impltype = colspecs[t]
            break
        except KeyError:
            pass
    else:
        # couldn't adapt - so just return the type itself
        # (it may be a user-defined type)
        return typeobj
    # if we adapted the given generic type to a database-specific type,
    # but it turns out the originally given "generic" type
    # is actually a subclass of our resulting type, then we were already
    # given a more specific type than that required; so use that.
    if issubclass(typeobj.__class__, impltype):
        return typeobj
    return typeobj.adapt(impltype)