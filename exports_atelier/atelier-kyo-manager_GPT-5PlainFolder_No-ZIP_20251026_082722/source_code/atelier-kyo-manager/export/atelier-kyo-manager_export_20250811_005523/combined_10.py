
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\compiler.py ===
# sql/compiler.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Base SQL and DDL compiler implementations.

Classes provided include:

:class:`.compiler.SQLCompiler` - renders SQL
strings

:class:`.compiler.DDLCompiler` - renders DDL
(data definition language) strings

:class:`.compiler.GenericTypeCompiler` - renders
type specification strings.

To generate user-defined SQL strings, see
:doc:`/ext/compiler`.

"""
from __future__ import annotations

import collections
import collections.abc as collections_abc
import contextlib
from enum import IntEnum
import functools
import itertools
import operator
import re
from time import perf_counter
import typing
from typing import Any
from typing import Callable
from typing import cast
from typing import ClassVar
from typing import Dict
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import MutableMapping
from typing import NamedTuple
from typing import NoReturn
from typing import Optional
from typing import Pattern
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union

from . import base
from . import coercions
from . import crud
from . import elements
from . import functions
from . import operators
from . import roles
from . import schema
from . import selectable
from . import sqltypes
from . import util as sql_util
from ._typing import is_column_element
from ._typing import is_dml
from .base import _de_clone
from .base import _from_objects
from .base import _NONE_NAME
from .base import _SentinelDefaultCharacterization
from .base import NO_ARG
from .elements import quoted_name
from .sqltypes import TupleType
from .visitors import prefix_anon_map
from .. import exc
from .. import util
from ..util import FastIntFlag
from ..util.typing import Literal
from ..util.typing import Protocol
from ..util.typing import Self
from ..util.typing import TypedDict

if typing.TYPE_CHECKING:
    from .annotation import _AnnotationDict
    from .base import _AmbiguousTableNameMap
    from .base import CompileState
    from .base import Executable
    from .cache_key import CacheKey
    from .ddl import ExecutableDDLElement
    from .dml import Insert
    from .dml import Update
    from .dml import UpdateBase
    from .dml import UpdateDMLState
    from .dml import ValuesBase
    from .elements import _truncated_label
    from .elements import BinaryExpression
    from .elements import BindParameter
    from .elements import ClauseElement
    from .elements import ColumnClause
    from .elements import ColumnElement
    from .elements import False_
    from .elements import Label
    from .elements import Null
    from .elements import True_
    from .functions import Function
    from .schema import Column
    from .schema import Constraint
    from .schema import ForeignKeyConstraint
    from .schema import Index
    from .schema import PrimaryKeyConstraint
    from .schema import Table
    from .schema import UniqueConstraint
    from .selectable import _ColumnsClauseElement
    from .selectable import AliasedReturnsRows
    from .selectable import CompoundSelectState
    from .selectable import CTE
    from .selectable import FromClause
    from .selectable import NamedFromClause
    from .selectable import ReturnsRows
    from .selectable import Select
    from .selectable import SelectState
    from .type_api import _BindProcessorType
    from .type_api import TypeDecorator
    from .type_api import TypeEngine
    from .type_api import UserDefinedType
    from .visitors import Visitable
    from ..engine.cursor import CursorResultMetaData
    from ..engine.interfaces import _CoreSingleExecuteParams
    from ..engine.interfaces import _DBAPIAnyExecuteParams
    from ..engine.interfaces import _DBAPIMultiExecuteParams
    from ..engine.interfaces import _DBAPISingleExecuteParams
    from ..engine.interfaces import _ExecuteOptions
    from ..engine.interfaces import _GenericSetInputSizesType
    from ..engine.interfaces import _MutableCoreSingleExecuteParams
    from ..engine.interfaces import Dialect
    from ..engine.interfaces import SchemaTranslateMapType


_FromHintsType = Dict["FromClause", str]

RESERVED_WORDS = {
    "all",
    "analyse",
    "analyze",
    "and",
    "any",
    "array",
    "as",
    "asc",
    "asymmetric",
    "authorization",
    "between",
    "binary",
    "both",
    "case",
    "cast",
    "check",
    "collate",
    "column",
    "constraint",
    "create",
    "cross",
    "current_date",
    "current_role",
    "current_time",
    "current_timestamp",
    "current_user",
    "default",
    "deferrable",
    "desc",
    "distinct",
    "do",
    "else",
    "end",
    "except",
    "false",
    "for",
    "foreign",
    "freeze",
    "from",
    "full",
    "grant",
    "group",
    "having",
    "ilike",
    "in",
    "initially",
    "inner",
    "intersect",
    "into",
    "is",
    "isnull",
    "join",
    "leading",
    "left",
    "like",
    "limit",
    "localtime",
    "localtimestamp",
    "natural",
    "new",
    "not",
    "notnull",
    "null",
    "off",
    "offset",
    "old",
    "on",
    "only",
    "or",
    "order",
    "outer",
    "overlaps",
    "placing",
    "primary",
    "references",
    "right",
    "select",
    "session_user",
    "set",
    "similar",
    "some",
    "symmetric",
    "table",
    "then",
    "to",
    "trailing",
    "true",
    "union",
    "unique",
    "user",
    "using",
    "verbose",
    "when",
    "where",
}

LEGAL_CHARACTERS = re.compile(r"^[A-Z0-9_$]+$", re.I)
LEGAL_CHARACTERS_PLUS_SPACE = re.compile(r"^[A-Z0-9_ $]+$", re.I)
ILLEGAL_INITIAL_CHARACTERS = {str(x) for x in range(0, 10)}.union(["$"])

FK_ON_DELETE = re.compile(
    r"^(?:RESTRICT|CASCADE|SET NULL|NO ACTION|SET DEFAULT)$", re.I
)
FK_ON_UPDATE = re.compile(
    r"^(?:RESTRICT|CASCADE|SET NULL|NO ACTION|SET DEFAULT)$", re.I
)
FK_INITIALLY = re.compile(r"^(?:DEFERRED|IMMEDIATE)$", re.I)
BIND_PARAMS = re.compile(r"(?<![:\w\$\x5c]):([\w\$]+)(?![:\w\$])", re.UNICODE)
BIND_PARAMS_ESC = re.compile(r"\x5c(:[\w\$]*)(?![:\w\$])", re.UNICODE)

_pyformat_template = "%%(%(name)s)s"
BIND_TEMPLATES = {
    "pyformat": _pyformat_template,
    "qmark": "?",
    "format": "%%s",
    "numeric": ":[_POSITION]",
    "numeric_dollar": "$[_POSITION]",
    "named": ":%(name)s",
}


OPERATORS = {
    # binary
    operators.and_: " AND ",
    operators.or_: " OR ",
    operators.add: " + ",
    operators.mul: " * ",
    operators.sub: " - ",
    operators.mod: " % ",
    operators.neg: "-",
    operators.lt: " < ",
    operators.le: " <= ",
    operators.ne: " != ",
    operators.gt: " > ",
    operators.ge: " >= ",
    operators.eq: " = ",
    operators.is_distinct_from: " IS DISTINCT FROM ",
    operators.is_not_distinct_from: " IS NOT DISTINCT FROM ",
    operators.concat_op: " || ",
    operators.match_op: " MATCH ",
    operators.not_match_op: " NOT MATCH ",
    operators.in_op: " IN ",
    operators.not_in_op: " NOT IN ",
    operators.comma_op: ", ",
    operators.from_: " FROM ",
    operators.as_: " AS ",
    operators.is_: " IS ",
    operators.is_not: " IS NOT ",
    operators.collate: " COLLATE ",
    # unary
    operators.exists: "EXISTS ",
    operators.distinct_op: "DISTINCT ",
    operators.inv: "NOT ",
    operators.any_op: "ANY ",
    operators.all_op: "ALL ",
    # modifiers
    operators.desc_op: " DESC",
    operators.asc_op: " ASC",
    operators.nulls_first_op: " NULLS FIRST",
    operators.nulls_last_op: " NULLS LAST",
    # bitwise
    operators.bitwise_xor_op: " ^ ",
    operators.bitwise_or_op: " | ",
    operators.bitwise_and_op: " & ",
    operators.bitwise_not_op: "~",
    operators.bitwise_lshift_op: " << ",
    operators.bitwise_rshift_op: " >> ",
}

FUNCTIONS: Dict[Type[Function[Any]], str] = {
    functions.coalesce: "coalesce",
    functions.current_date: "CURRENT_DATE",
    functions.current_time: "CURRENT_TIME",
    functions.current_timestamp: "CURRENT_TIMESTAMP",
    functions.current_user: "CURRENT_USER",
    functions.localtime: "LOCALTIME",
    functions.localtimestamp: "LOCALTIMESTAMP",
    functions.random: "random",
    functions.sysdate: "sysdate",
    functions.session_user: "SESSION_USER",
    functions.user: "USER",
    functions.cube: "CUBE",
    functions.rollup: "ROLLUP",
    functions.grouping_sets: "GROUPING SETS",
}


EXTRACT_MAP = {
    "month": "month",
    "day": "day",
    "year": "year",
    "second": "second",
    "hour": "hour",
    "doy": "doy",
    "minute": "minute",
    "quarter": "quarter",
    "dow": "dow",
    "week": "week",
    "epoch": "epoch",
    "milliseconds": "milliseconds",
    "microseconds": "microseconds",
    "timezone_hour": "timezone_hour",
    "timezone_minute": "timezone_minute",
}

COMPOUND_KEYWORDS = {
    selectable._CompoundSelectKeyword.UNION: "UNION",
    selectable._CompoundSelectKeyword.UNION_ALL: "UNION ALL",
    selectable._CompoundSelectKeyword.EXCEPT: "EXCEPT",
    selectable._CompoundSelectKeyword.EXCEPT_ALL: "EXCEPT ALL",
    selectable._CompoundSelectKeyword.INTERSECT: "INTERSECT",
    selectable._CompoundSelectKeyword.INTERSECT_ALL: "INTERSECT ALL",
}


class ResultColumnsEntry(NamedTuple):
    """Tracks a column expression that is expected to be represented
    in the result rows for this statement.

    This normally refers to the columns clause of a SELECT statement
    but may also refer to a RETURNING clause, as well as for dialect-specific
    emulations.

    """

    keyname: str
    """string name that's expected in cursor.description"""

    name: str
    """column name, may be labeled"""

    objects: Tuple[Any, ...]
    """sequence of objects that should be able to locate this column
    in a RowMapping.  This is typically string names and aliases
    as well as Column objects.

    """

    type: TypeEngine[Any]
    """Datatype to be associated with this column.   This is where
    the "result processing" logic directly links the compiled statement
    to the rows that come back from the cursor.

    """


class _ResultMapAppender(Protocol):
    def __call__(
        self,
        keyname: str,
        name: str,
        objects: Sequence[Any],
        type_: TypeEngine[Any],
    ) -> None: ...


# integer indexes into ResultColumnsEntry used by cursor.py.
# some profiling showed integer access faster than named tuple
RM_RENDERED_NAME: Literal[0] = 0
RM_NAME: Literal[1] = 1
RM_OBJECTS: Literal[2] = 2
RM_TYPE: Literal[3] = 3


class _BaseCompilerStackEntry(TypedDict):
    asfrom_froms: Set[FromClause]
    correlate_froms: Set[FromClause]
    selectable: ReturnsRows


class _CompilerStackEntry(_BaseCompilerStackEntry, total=False):
    compile_state: CompileState
    need_result_map_for_nested: bool
    need_result_map_for_compound: bool
    select_0: ReturnsRows
    insert_from_select: Select[Any]


class ExpandedState(NamedTuple):
    """represents state to use when producing "expanded" and
    "post compile" bound parameters for a statement.

    "expanded" parameters are parameters that are generated at
    statement execution time to suit a number of parameters passed, the most
    prominent example being the individual elements inside of an IN expression.

    "post compile" parameters are parameters where the SQL literal value
    will be rendered into the SQL statement at execution time, rather than
    being passed as separate parameters to the driver.

    To create an :class:`.ExpandedState` instance, use the
    :meth:`.SQLCompiler.construct_expanded_state` method on any
    :class:`.SQLCompiler` instance.

    """

    statement: str
    """String SQL statement with parameters fully expanded"""

    parameters: _CoreSingleExecuteParams
    """Parameter dictionary with parameters fully expanded.

    For a statement that uses named parameters, this dictionary will map
    exactly to the names in the statement.  For a statement that uses
    positional parameters, the :attr:`.ExpandedState.positional_parameters`
    will yield a tuple with the positional parameter set.

    """

    processors: Mapping[str, _BindProcessorType[Any]]
    """mapping of bound value processors"""

    positiontup: Optional[Sequence[str]]
    """Sequence of string names indicating the order of positional
    parameters"""

    parameter_expansion: Mapping[str, List[str]]
    """Mapping representing the intermediary link from original parameter
    name to list of "expanded" parameter names, for those parameters that
    were expanded."""

    @property
    def positional_parameters(self) -> Tuple[Any, ...]:
        """Tuple of positional parameters, for statements that were compiled
        using a positional paramstyle.

        """
        if self.positiontup is None:
            raise exc.InvalidRequestError(
                "statement does not use a positional paramstyle"
            )
        return tuple(self.parameters[key] for key in self.positiontup)

    @property
    def additional_parameters(self) -> _CoreSingleExecuteParams:
        """synonym for :attr:`.ExpandedState.parameters`."""
        return self.parameters


class _InsertManyValues(NamedTuple):
    """represents state to use for executing an "insertmanyvalues" statement.

    The primary consumers of this object are the
    :meth:`.SQLCompiler._deliver_insertmanyvalues_batches` and
    :meth:`.DefaultDialect._deliver_insertmanyvalues_batches` methods.

    .. versionadded:: 2.0

    """

    is_default_expr: bool
    """if True, the statement is of the form
    ``INSERT INTO TABLE DEFAULT VALUES``, and can't be rewritten as a "batch"

    """

    single_values_expr: str
    """The rendered "values" clause of the INSERT statement.

    This is typically the parenthesized section e.g. "(?, ?, ?)" or similar.
    The insertmanyvalues logic uses this string as a search and replace
    target.

    """

    insert_crud_params: List[crud._CrudParamElementStr]
    """List of Column / bind names etc. used while rewriting the statement"""

    num_positional_params_counted: int
    """the number of bound parameters in a single-row statement.

    This count may be larger or smaller than the actual number of columns
    targeted in the INSERT, as it accommodates for SQL expressions
    in the values list that may have zero or more parameters embedded
    within them.

    This count is part of what's used to organize rewritten parameter lists
    when batching.

    """

    sort_by_parameter_order: bool = False
    """if the deterministic_returnined_order parameter were used on the
    insert.

    All of the attributes following this will only be used if this is True.

    """

    includes_upsert_behaviors: bool = False
    """if True, we have to accommodate for upsert behaviors.

    This will in some cases downgrade "insertmanyvalues" that requests
    deterministic ordering.

    """

    sentinel_columns: Optional[Sequence[Column[Any]]] = None
    """List of sentinel columns that were located.

    This list is only here if the INSERT asked for
    sort_by_parameter_order=True,
    and dialect-appropriate sentinel columns were located.

    .. versionadded:: 2.0.10

    """

    num_sentinel_columns: int = 0
    """how many sentinel columns are in the above list, if any.

    This is the same as
    ``len(sentinel_columns) if sentinel_columns is not None else 0``

    """

    sentinel_param_keys: Optional[Sequence[str]] = None
    """parameter str keys in each param dictionary / tuple
    that would link to the client side "sentinel" values for that row, which
    we can use to match up parameter sets to result rows.

    This is only present if sentinel_columns is present and the INSERT
    statement actually refers to client side values for these sentinel
    columns.

    .. versionadded:: 2.0.10

    .. versionchanged:: 2.0.29 - the sequence is now string dictionary keys
       only, used against the "compiled parameteters" collection before
       the parameters were converted by bound parameter processors

    """

    implicit_sentinel: bool = False
    """if True, we have exactly one sentinel column and it uses a server side
    value, currently has to generate an incrementing integer value.

    The dialect in question would have asserted that it supports receiving
    these values back and sorting on that value as a means of guaranteeing
    correlation with the incoming parameter list.

    .. versionadded:: 2.0.10

    """

    embed_values_counter: bool = False
    """Whether to embed an incrementing integer counter in each parameter
    set within the VALUES clause as parameters are batched over.

    This is only used for a specific INSERT..SELECT..VALUES..RETURNING syntax
    where a subquery is used to produce value tuples.  Current support
    includes PostgreSQL, Microsoft SQL Server.

    .. versionadded:: 2.0.10

    """


class _InsertManyValuesBatch(NamedTuple):
    """represents an individual batch SQL statement for insertmanyvalues.

    This is passed through the
    :meth:`.SQLCompiler._deliver_insertmanyvalues_batches` and
    :meth:`.DefaultDialect._deliver_insertmanyvalues_batches` methods out
    to the :class:`.Connection` within the
    :meth:`.Connection._exec_insertmany_context` method.

    .. versionadded:: 2.0.10

    """

    replaced_statement: str
    replaced_parameters: _DBAPIAnyExecuteParams
    processed_setinputsizes: Optional[_GenericSetInputSizesType]
    batch: Sequence[_DBAPISingleExecuteParams]
    sentinel_values: Sequence[Tuple[Any, ...]]
    current_batch_size: int
    batchnum: int
    total_batches: int
    rows_sorted: bool
    is_downgraded: bool


class InsertmanyvaluesSentinelOpts(FastIntFlag):
    """bitflag enum indicating styles of PK defaults
    which can work as implicit sentinel columns

    """

    NOT_SUPPORTED = 1
    AUTOINCREMENT = 2
    IDENTITY = 4
    SEQUENCE = 8

    ANY_AUTOINCREMENT = AUTOINCREMENT | IDENTITY | SEQUENCE
    _SUPPORTED_OR_NOT = NOT_SUPPORTED | ANY_AUTOINCREMENT

    USE_INSERT_FROM_SELECT = 16
    RENDER_SELECT_COL_CASTS = 64


class CompilerState(IntEnum):
    COMPILING = 0
    """statement is present, compilation phase in progress"""

    STRING_APPLIED = 1
    """statement is present, string form of the statement has been applied.

    Additional processors by subclasses may still be pending.

    """

    NO_STATEMENT = 2
    """compiler does not have a statement to compile, is used
    for method access"""


class Linting(IntEnum):
    """represent preferences for the 'SQL linting' feature.

    this feature currently includes support for flagging cartesian products
    in SQL statements.

    """

    NO_LINTING = 0
    "Disable all linting."

    COLLECT_CARTESIAN_PRODUCTS = 1
    """Collect data on FROMs and cartesian products and gather into
    'self.from_linter'"""

    WARN_LINTING = 2
    "Emit warnings for linters that find problems"

    FROM_LINTING = COLLECT_CARTESIAN_PRODUCTS | WARN_LINTING
    """Warn for cartesian products; combines COLLECT_CARTESIAN_PRODUCTS
    and WARN_LINTING"""


NO_LINTING, COLLECT_CARTESIAN_PRODUCTS, WARN_LINTING, FROM_LINTING = tuple(
    Linting
)


class FromLinter(collections.namedtuple("FromLinter", ["froms", "edges"])):
    """represents current state for the "cartesian product" detection
    feature."""

    def lint(self, start=None):
        froms = self.froms
        if not froms:
            return None, None

        edges = set(self.edges)
        the_rest = set(froms)

        if start is not None:
            start_with = start
            the_rest.remove(start_with)
        else:
            start_with = the_rest.pop()

        stack = collections.deque([start_with])

        while stack and the_rest:
            node = stack.popleft()
            the_rest.discard(node)

            # comparison of nodes in edges here is based on hash equality, as
            # there are "annotated" elements that match the non-annotated ones.
            #   to remove the need for in-python hash() calls, use native
            # containment routines (e.g. "node in edge", "edge.index(node)")
            to_remove = {edge for edge in edges if node in edge}

            # appendleft the node in each edge that is not
            # the one that matched.
            stack.extendleft(edge[not edge.index(node)] for edge in to_remove)
            edges.difference_update(to_remove)

        # FROMS left over?  boom
        if the_rest:
            return the_rest, start_with
        else:
            return None, None

    def warn(self, stmt_type="SELECT"):
        the_rest, start_with = self.lint()

        # FROMS left over?  boom
        if the_rest:
            froms = the_rest
            if froms:
                template = (
                    "{stmt_type} statement has a cartesian product between "
                    "FROM element(s) {froms} and "
                    'FROM element "{start}".  Apply join condition(s) '
                    "between each element to resolve."
                )
                froms_str = ", ".join(
                    f'"{self.froms[from_]}"' for from_ in froms
                )
                message = template.format(
                    stmt_type=stmt_type,
                    froms=froms_str,
                    start=self.froms[start_with],
                )

                util.warn(message)


class Compiled:
    """Represent a compiled SQL or DDL expression.

    The ``__str__`` method of the ``Compiled`` object should produce
    the actual text of the statement.  ``Compiled`` objects are
    specific to their underlying database dialect, and also may
    or may not be specific to the columns referenced within a
    particular set of bind parameters.  In no case should the
    ``Compiled`` object be dependent on the actual values of those
    bind parameters, even though it may reference those values as
    defaults.
    """

    statement: Optional[ClauseElement] = None
    "The statement to compile."
    string: str = ""
    "The string representation of the ``statement``"

    state: CompilerState
    """description of the compiler's state"""

    is_sql = False
    is_ddl = False

    _cached_metadata: Optional[CursorResultMetaData] = None

    _result_columns: Optional[List[ResultColumnsEntry]] = None

    schema_translate_map: Optional[SchemaTranslateMapType] = None

    execution_options: _ExecuteOptions = util.EMPTY_DICT
    """
    Execution options propagated from the statement.   In some cases,
    sub-elements of the statement can modify these.
    """

    preparer: IdentifierPreparer

    _annotations: _AnnotationDict = util.EMPTY_DICT

    compile_state: Optional[CompileState] = None
    """Optional :class:`.CompileState` object that maintains additional
    state used by the compiler.

    Major executable objects such as :class:`_expression.Insert`,
    :class:`_expression.Update`, :class:`_expression.Delete`,
    :class:`_expression.Select` will generate this
    state when compiled in order to calculate additional information about the
    object.   For the top level object that is to be executed, the state can be
    stored here where it can also have applicability towards result set
    processing.

    .. versionadded:: 1.4

    """

    dml_compile_state: Optional[CompileState] = None
    """Optional :class:`.CompileState` assigned at the same point that
    .isinsert, .isupdate, or .isdelete is assigned.

    This will normally be the same object as .compile_state, with the
    exception of cases like the :class:`.ORMFromStatementCompileState`
    object.

    .. versionadded:: 1.4.40

    """

    cache_key: Optional[CacheKey] = None
    """The :class:`.CacheKey` that was generated ahead of creating this
    :class:`.Compiled` object.

    This is used for routines that need access to the original
    :class:`.CacheKey` instance generated when the :class:`.Compiled`
    instance was first cached, typically in order to reconcile
    the original list of :class:`.BindParameter` objects with a
    per-statement list that's generated on each call.

    """

    _gen_time: float
    """Generation time of this :class:`.Compiled`, used for reporting
    cache stats."""

    def __init__(
        self,
        dialect: Dialect,
        statement: Optional[ClauseElement],
        schema_translate_map: Optional[SchemaTranslateMapType] = None,
        render_schema_translate: bool = False,
        compile_kwargs: Mapping[str, Any] = util.immutabledict(),
    ):
        """Construct a new :class:`.Compiled` object.

        :param dialect: :class:`.Dialect` to compile against.

        :param statement: :class:`_expression.ClauseElement` to be compiled.

        :param schema_translate_map: dictionary of schema names to be
         translated when forming the resultant SQL

         .. seealso::

            :ref:`schema_translating`

        :param compile_kwargs: additional kwargs that will be
         passed to the initial call to :meth:`.Compiled.process`.


        """
        self.dialect = dialect
        self.preparer = self.dialect.identifier_preparer
        if schema_translate_map:
            self.schema_translate_map = schema_translate_map
            self.preparer = self.preparer._with_schema_translate(
                schema_translate_map
            )

        if statement is not None:
            self.state = CompilerState.COMPILING
            self.statement = statement
            self.can_execute = statement.supports_execution
            self._annotations = statement._annotations
            if self.can_execute:
                if TYPE_CHECKING:
                    assert isinstance(statement, Executable)
                self.execution_options = statement._execution_options
            self.string = self.process(self.statement, **compile_kwargs)

            if render_schema_translate:
                assert schema_translate_map is not None
                self.string = self.preparer._render_schema_translates(
                    self.string, schema_translate_map
                )

            self.state = CompilerState.STRING_APPLIED
        else:
            self.state = CompilerState.NO_STATEMENT

        self._gen_time = perf_counter()

    def __init_subclass__(cls) -> None:
        cls._init_compiler_cls()
        return super().__init_subclass__()

    @classmethod
    def _init_compiler_cls(cls):
        pass

    def _execute_on_connection(
        self, connection, distilled_params, execution_options
    ):
        if self.can_execute:
            return connection._execute_compiled(
                self, distilled_params, execution_options
            )
        else:
            raise exc.ObjectNotExecutableError(self.statement)

    def visit_unsupported_compilation(self, element, err, **kw):
        raise exc.UnsupportedCompilationError(self, type(element)) from err

    @property
    def sql_compiler(self) -> SQLCompiler:
        """Return a Compiled that is capable of processing SQL expressions.

        If this compiler is one, it would likely just return 'self'.

        """

        raise NotImplementedError()

    def process(self, obj: Visitable, **kwargs: Any) -> str:
        return obj._compiler_dispatch(self, **kwargs)

    def __str__(self) -> str:
        """Return the string text of the generated SQL or DDL."""

        if self.state is CompilerState.STRING_APPLIED:
            return self.string
        else:
            return ""

    def construct_params(
        self,
        params: Optional[_CoreSingleExecuteParams] = None,
        extracted_parameters: Optional[Sequence[BindParameter[Any]]] = None,
        escape_names: bool = True,
    ) -> Optional[_MutableCoreSingleExecuteParams]:
        """Return the bind params for this compiled object.

        :param params: a dict of string/object pairs whose values will
                       override bind values compiled in to the
                       statement.
        """

        raise NotImplementedError()

    @property
    def params(self):
        """Return the bind params for this compiled object."""
        return self.construct_params()


class TypeCompiler(util.EnsureKWArg):
    """Produces DDL specification for TypeEngine objects."""

    ensure_kwarg = r"visit_\w+"

    def __init__(self, dialect: Dialect):
        self.dialect = dialect

    def process(self, type_: TypeEngine[Any], **kw: Any) -> str:
        if (
            type_._variant_mapping
            and self.dialect.name in type_._variant_mapping
        ):
            type_ = type_._variant_mapping[self.dialect.name]
        return type_._compiler_dispatch(self, **kw)

    def visit_unsupported_compilation(
        self, element: Any, err: Exception, **kw: Any
    ) -> NoReturn:
        raise exc.UnsupportedCompilationError(self, element) from err


# this was a Visitable, but to allow accurate detection of
# column elements this is actually a column element
class _CompileLabel(
    roles.BinaryElementRole[Any], elements.CompilerColumnElement
):
    """lightweight label object which acts as an expression.Label."""

    __visit_name__ = "label"
    __slots__ = "element", "name", "_alt_names"

    def __init__(self, col, name, alt_names=()):
        self.element = col
        self.name = name
        self._alt_names = (col,) + alt_names

    @property
    def proxy_set(self):
        return self.element.proxy_set

    @property
    def type(self):
        return self.element.type

    def self_group(self, **kw):
        return self


class ilike_case_insensitive(
    roles.BinaryElementRole[Any], elements.CompilerColumnElement
):
    """produce a wrapping element for a case-insensitive portion of
    an ILIKE construct.

    The construct usually renders the ``lower()`` function, but on
    PostgreSQL will pass silently with the assumption that "ILIKE"
    is being used.

    .. versionadded:: 2.0

    """

    __visit_name__ = "ilike_case_insensitive_operand"
    __slots__ = "element", "comparator"

    def __init__(self, element):
        self.element = element
        self.comparator = element.comparator

    @property
    def proxy_set(self):
        return self.element.proxy_set

    @property
    def type(self):
        return self.element.type

    def self_group(self, **kw):
        return self

    def _with_binary_element_type(self, type_):
        return ilike_case_insensitive(
            self.element._with_binary_element_type(type_)
        )


class SQLCompiler(Compiled):
    """Default implementation of :class:`.Compiled`.

    Compiles :class:`_expression.ClauseElement` objects into SQL strings.

    """

    extract_map = EXTRACT_MAP

    bindname_escape_characters: ClassVar[Mapping[str, str]] = (
        util.immutabledict(
            {
                "%": "P",
                "(": "A",
                ")": "Z",
                ":": "C",
                ".": "_",
                "[": "_",
                "]": "_",
                " ": "_",
            }
        )
    )
    """A mapping (e.g. dict or similar) containing a lookup of
    characters keyed to replacement characters which will be applied to all
    'bind names' used in SQL statements as a form of 'escaping'; the given
    characters are replaced entirely with the 'replacement' character when
    rendered in the SQL statement, and a similar translation is performed
    on the incoming names used in parameter dictionaries passed to methods
    like :meth:`_engine.Connection.execute`.

    This allows bound parameter names used in :func:`_sql.bindparam` and
    other constructs to have any arbitrary characters present without any
    concern for characters that aren't allowed at all on the target database.

    Third party dialects can establish their own dictionary here to replace the
    default mapping, which will ensure that the particular characters in the
    mapping will never appear in a bound parameter name.

    The dictionary is evaluated at **class creation time**, so cannot be
    modified at runtime; it must be present on the class when the class
    is first declared.

    Note that for dialects that have additional bound parameter rules such
    as additional restrictions on leading characters, the
    :meth:`_sql.SQLCompiler.bindparam_string` method may need to be augmented.
    See the cx_Oracle compiler for an example of this.

    .. versionadded:: 2.0.0rc1

    """

    _bind_translate_re: ClassVar[Pattern[str]]
    _bind_translate_chars: ClassVar[Mapping[str, str]]

    is_sql = True

    compound_keywords = COMPOUND_KEYWORDS

    isdelete: bool = False
    isinsert: bool = False
    isupdate: bool = False
    """class-level defaults which can be set at the instance
    level to define if this Compiled instance represents
    INSERT/UPDATE/DELETE
    """

    postfetch: Optional[List[Column[Any]]]
    """list of columns that can be post-fetched after INSERT or UPDATE to
    receive server-updated values"""

    insert_prefetch: Sequence[Column[Any]] = ()
    """list of columns for which default values should be evaluated before
    an INSERT takes place"""

    update_prefetch: Sequence[Column[Any]] = ()
    """list of columns for which onupdate default values should be evaluated
    before an UPDATE takes place"""

    implicit_returning: Optional[Sequence[ColumnElement[Any]]] = None
    """list of "implicit" returning columns for a toplevel INSERT or UPDATE
    statement, used to receive newly generated values of columns.

    .. versionadded:: 2.0  ``implicit_returning`` replaces the previous
       ``returning`` collection, which was not a generalized RETURNING
       collection and instead was in fact specific to the "implicit returning"
       feature.

    """

    isplaintext: bool = False

    binds: Dict[str, BindParameter[Any]]
    """a dictionary of bind parameter keys to BindParameter instances."""

    bind_names: Dict[BindParameter[Any], str]
    """a dictionary of BindParameter instances to "compiled" names
    that are actually present in the generated SQL"""

    stack: List[_CompilerStackEntry]
    """major statements such as SELECT, INSERT, UPDATE, DELETE are
    tracked in this stack using an entry format."""

    returning_precedes_values: bool = False
    """set to True classwide to generate RETURNING
    clauses before the VALUES or WHERE clause (i.e. MSSQL)
    """

    render_table_with_column_in_update_from: bool = False
    """set to True classwide to indicate the SET clause
    in a multi-table UPDATE statement should qualify
    columns with the table name (i.e. MySQL only)
    """

    ansi_bind_rules: bool = False
    """SQL 92 doesn't allow bind parameters to be used
    in the columns clause of a SELECT, nor does it allow
    ambiguous expressions like "? = ?".  A compiler
    subclass can set this flag to False if the target
    driver/DB enforces this
    """

    bindtemplate: str
    """template to render bound parameters based on paramstyle."""

    compilation_bindtemplate: str
    """template used by compiler to render parameters before positional
    paramstyle application"""

    _numeric_binds_identifier_char: str
    """Character that's used to as the identifier of a numerical bind param.
    For example if this char is set to ``$``, numerical binds will be rendered
    in the form ``$1, $2, $3``.
    """

    _result_columns: List[ResultColumnsEntry]
    """relates label names in the final SQL to a tuple of local
    column/label name, ColumnElement object (if any) and
    TypeEngine. CursorResult uses this for type processing and
    column targeting"""

    _textual_ordered_columns: bool = False
    """tell the result object that the column names as rendered are important,
    but they are also "ordered" vs. what is in the compiled object here.

    As of 1.4.42 this condition is only present when the statement is a
    TextualSelect, e.g. text("....").columns(...), where it is required
    that the columns are considered positionally and not by name.

    """

    _ad_hoc_textual: bool = False
    """tell the result that we encountered text() or '*' constructs in the
    middle of the result columns, but we also have compiled columns, so
    if the number of columns in cursor.description does not match how many
    expressions we have, that means we can't rely on positional at all and
    should match on name.

    """

    _ordered_columns: bool = True
    """
    if False, means we can't be sure the list of entries
    in _result_columns is actually the rendered order.  Usually
    True unless using an unordered TextualSelect.
    """

    _loose_column_name_matching: bool = False
    """tell the result object that the SQL statement is textual, wants to match
    up to Column objects, and may be using the ._tq_label in the SELECT rather
    than the base name.

    """

    _numeric_binds: bool = False
    """
    True if paramstyle is "numeric".  This paramstyle is trickier than
    all the others.

    """

    _render_postcompile: bool = False
    """
    whether to render out POSTCOMPILE params during the compile phase.

    This attribute is used only for end-user invocation of stmt.compile();
    it's never used for actual statement execution, where instead the
    dialect internals access and render the internal postcompile structure
    directly.

    """

    _post_compile_expanded_state: Optional[ExpandedState] = None
    """When render_postcompile is used, the ``ExpandedState`` used to create
    the "expanded" SQL is assigned here, and then used by the ``.params``
    accessor and ``.construct_params()`` methods for their return values.

    .. versionadded:: 2.0.0rc1

    """

    _pre_expanded_string: Optional[str] = None
    """Stores the original string SQL before 'post_compile' is applied,
    for cases where 'post_compile' were used.

    """

    _pre_expanded_positiontup: Optional[List[str]] = None

    _insertmanyvalues: Optional[_InsertManyValues] = None

    _insert_crud_params: Optional[crud._CrudParamSequence] = None

    literal_execute_params: FrozenSet[BindParameter[Any]] = frozenset()
    """bindparameter objects that are rendered as literal values at statement
    execution time.

    """

    post_compile_params: FrozenSet[BindParameter[Any]] = frozenset()
    """bindparameter objects that are rendered as bound parameter placeholders
    at statement execution time.

    """

    escaped_bind_names: util.immutabledict[str, str] = util.EMPTY_DICT
    """Late escaping of bound parameter names that has to be converted
    to the original name when looking in the parameter dictionary.

    """

    has_out_parameters = False
    """if True, there are bindparam() objects that have the isoutparam
    flag set."""

    postfetch_lastrowid = False
    """if True, and this in insert, use cursor.lastrowid to populate
    result.inserted_primary_key. """

    _cache_key_bind_match: Optional[
        Tuple[
            Dict[
                BindParameter[Any],
                List[BindParameter[Any]],
            ],
            Dict[
                str,
                BindParameter[Any],
            ],
        ]
    ] = None
    """a mapping that will relate the BindParameter object we compile
    to those that are part of the extracted collection of parameters
    in the cache key, if we were given a cache key.

    """

    positiontup: Optional[List[str]] = None
    """for a compiled construct that uses a positional paramstyle, will be
    a sequence of strings, indicating the names of bound parameters in order.

    This is used in order to render bound parameters in their correct order,
    and is combined with the :attr:`_sql.Compiled.params` dictionary to
    render parameters.

    This sequence always contains the unescaped name of the parameters.

    .. seealso::

        :ref:`faq_sql_expression_string` - includes a usage example for
        debugging use cases.

    """
    _values_bindparam: Optional[List[str]] = None

    _visited_bindparam: Optional[List[str]] = None

    inline: bool = False

    ctes: Optional[MutableMapping[CTE, str]]

    # Detect same CTE references - Dict[(level, name), cte]
    # Level is required for supporting nesting
    ctes_by_level_name: Dict[Tuple[int, str], CTE]

    # To retrieve key/level in ctes_by_level_name -
    # Dict[cte_reference, (level, cte_name, cte_opts)]
    level_name_by_cte: Dict[CTE, Tuple[int, str, selectable._CTEOpts]]

    ctes_recursive: bool

    _post_compile_pattern = re.compile(r"__\[POSTCOMPILE_(\S+?)(~~.+?~~)?\]")
    _pyformat_pattern = re.compile(r"%\(([^)]+?)\)s")
    _positional_pattern = re.compile(
        f"{_pyformat_pattern.pattern}|{_post_compile_pattern.pattern}"
    )

    @classmethod
    def _init_compiler_cls(cls):
        cls._init_bind_translate()

    @classmethod
    def _init_bind_translate(cls):
        reg = re.escape("".join(cls.bindname_escape_characters))
        cls._bind_translate_re = re.compile(f"[{reg}]")
        cls._bind_translate_chars = cls.bindname_escape_characters

    def __init__(
        self,
        dialect: Dialect,
        statement: Optional[ClauseElement],
        cache_key: Optional[CacheKey] = None,
        column_keys: Optional[Sequence[str]] = None,
        for_executemany: bool = False,
        linting: Linting = NO_LINTING,
        _supporting_against: Optional[SQLCompiler] = None,
        **kwargs: Any,
    ):
        """Construct a new :class:`.SQLCompiler` object.

        :param dialect: :class:`.Dialect` to be used

        :param statement: :class:`_expression.ClauseElement` to be compiled

        :param column_keys:  a list of column names to be compiled into an
         INSERT or UPDATE statement.

        :param for_executemany: whether INSERT / UPDATE statements should
         expect that they are to be invoked in an "executemany" style,
         which may impact how the statement will be expected to return the
         values of defaults and autoincrement / sequences and similar.
         Depending on the backend and driver in use, support for retrieving
         these values may be disabled which means SQL expressions may
         be rendered inline, RETURNING may not be rendered, etc.

        :param kwargs: additional keyword arguments to be consumed by the
         superclass.

        """
        self.column_keys = column_keys

        self.cache_key = cache_key

        if cache_key:
            cksm = {b.key: b for b in cache_key[1]}
            ckbm = {b: [b] for b in cache_key[1]}
            self._cache_key_bind_match = (ckbm, cksm)

        # compile INSERT/UPDATE defaults/sequences to expect executemany
        # style execution, which may mean no pre-execute of defaults,
        # or no RETURNING
        self.for_executemany = for_executemany

        self.linting = linting

        # a dictionary of bind parameter keys to BindParameter
        # instances.
        self.binds = {}

        # a dictionary of BindParameter instances to "compiled" names
        # that are actually present in the generated SQL
        self.bind_names = util.column_dict()

        # stack which keeps track of nested SELECT statements
        self.stack = []

        self._result_columns = []

        # true if the paramstyle is positional
        self.positional = dialect.positional
        if self.positional:
            self._numeric_binds = nb = dialect.paramstyle.startswith("numeric")
            if nb:
                self._numeric_binds_identifier_char = (
                    "$" if dialect.paramstyle == "numeric_dollar" else ":"
                )

            self.compilation_bindtemplate = _pyformat_template
        else:
            self.compilation_bindtemplate = BIND_TEMPLATES[dialect.paramstyle]

        self.ctes = None

        self.label_length = (
            dialect.label_length or dialect.max_identifier_length
        )

        # a map which tracks "anonymous" identifiers that are created on
        # the fly here
        self.anon_map = prefix_anon_map()

        # a map which tracks "truncated" names based on
        # dialect.label_length or dialect.max_identifier_length
        self.truncated_names: Dict[Tuple[str, str], str] = {}
        self._truncated_counters: Dict[str, int] = {}

        Compiled.__init__(self, dialect, statement, **kwargs)

        if self.isinsert or self.isupdate or self.isdelete:
            if TYPE_CHECKING:
                assert isinstance(statement, UpdateBase)

            if self.isinsert or self.isupdate:
                if TYPE_CHECKING:
                    assert isinstance(statement, ValuesBase)
                if statement._inline:
                    self.inline = True
                elif self.for_executemany and (
                    not self.isinsert
                    or (
                        self.dialect.insert_executemany_returning
                        and statement._return_defaults
                    )
                ):
                    self.inline = True

        self.bindtemplate = BIND_TEMPLATES[dialect.paramstyle]

        if _supporting_against:
            self.__dict__.update(
                {
                    k: v
                    for k, v in _supporting_against.__dict__.items()
                    if k
                    not in {
                        "state",
                        "dialect",
                        "preparer",
                        "positional",
                        "_numeric_binds",
                        "compilation_bindtemplate",
                        "bindtemplate",
                    }
                }
            )

        if self.state is CompilerState.STRING_APPLIED:
            if self.positional:
                if self._numeric_binds:
                    self._process_numeric()
                else:
                    self._process_positional()

            if self._render_postcompile:
                parameters = self.construct_params(
                    escape_names=False,
                    _no_postcompile=True,
                )

                self._process_parameters_for_postcompile(
                    parameters, _populate_self=True
                )

    @property
    def insert_single_values_expr(self) -> Optional[str]:
        """When an INSERT is compiled with a single set of parameters inside
        a VALUES expression, the string is assigned here, where it can be
        used for insert batching schemes to rewrite the VALUES expression.

        .. versionadded:: 1.3.8

        .. versionchanged:: 2.0 This collection is no longer used by
           SQLAlchemy's built-in dialects, in favor of the currently
           internal ``_insertmanyvalues`` collection that is used only by
           :class:`.SQLCompiler`.

        """
        if self._insertmanyvalues is None:
            return None
        else:
            return self._insertmanyvalues.single_values_expr

    @util.ro_memoized_property
    def effective_returning(self) -> Optional[Sequence[ColumnElement[Any]]]:
        """The effective "returning" columns for INSERT, UPDATE or DELETE.

        This is either the so-called "implicit returning" columns which are
        calculated by the compiler on the fly, or those present based on what's
        present in ``self.statement._returning`` (expanded into individual
        columns using the ``._all_selected_columns`` attribute) i.e. those set
        explicitly using the :meth:`.UpdateBase.returning` method.

        .. versionadded:: 2.0

        """
        if self.implicit_returning:
            return self.implicit_returning
        elif self.statement is not None and is_dml(self.statement):
            return [
                c
                for c in self.statement._all_selected_columns
                if is_column_element(c)
            ]

        else:
            return None

    @property
    def returning(self):
        """backwards compatibility; returns the
        effective_returning collection.

        """
        return self.effective_returning

    @property
    def current_executable(self):
        """Return the current 'executable' that is being compiled.

        This is currently the :class:`_sql.Select`, :class:`_sql.Insert`,
        :class:`_sql.Update`, :class:`_sql.Delete`,
        :class:`_sql.CompoundSelect` object that is being compiled.
        Specifically it's assigned to the ``self.stack`` list of elements.

        When a statement like the above is being compiled, it normally
        is also assigned to the ``.statement`` attribute of the
        :class:`_sql.Compiler` object.   However, all SQL constructs are
        ultimately nestable, and this attribute should never be consulted
        by a ``visit_`` method, as it is not guaranteed to be assigned
        nor guaranteed to correspond to the current statement being compiled.

        .. versionadded:: 1.3.21

            For compatibility with previous versions, use the following
            recipe::

                statement = getattr(self, "current_executable", False)
                if statement is False:
                    statement = self.stack[-1]["selectable"]

            For versions 1.4 and above, ensure only .current_executable
            is used; the format of "self.stack" may change.


        """
        try:
            return self.stack[-1]["selectable"]
        except IndexError as ie:
            raise IndexError("Compiler does not have a stack entry") from ie

    @property
    def prefetch(self):
        return list(self.insert_prefetch) + list(self.update_prefetch)

    @util.memoized_property
    def _global_attributes(self) -> Dict[Any, Any]:
        return {}

    @util.memoized_instancemethod
    def _init_cte_state(self) -> MutableMapping[CTE, str]:
        """Initialize collections related to CTEs only if
        a CTE is located, to save on the overhead of
        these collections otherwise.

        """
        # collect CTEs to tack on top of a SELECT
        # To store the query to print - Dict[cte, text_query]
        ctes: MutableMapping[CTE, str] = util.OrderedDict()
        self.ctes = ctes

        # Detect same CTE references - Dict[(level, name), cte]
        # Level is required for supporting nesting
        self.ctes_by_level_name = {}

        # To retrieve key/level in ctes_by_level_name -
        # Dict[cte_reference, (level, cte_name, cte_opts)]
        self.level_name_by_cte = {}

        self.ctes_recursive = False

        return ctes

    @contextlib.contextmanager
    def _nested_result(self):
        """special API to support the use case of 'nested result sets'"""
        result_columns, ordered_columns = (
            self._result_columns,
            self._ordered_columns,
        )
        self._result_columns, self._ordered_columns = [], False

        try:
            if self.stack:
                entry = self.stack[-1]
                entry["need_result_map_for_nested"] = True
            else:
                entry = None
            yield self._result_columns, self._ordered_columns
        finally:
            if entry:
                entry.pop("need_result_map_for_nested")
            self._result_columns, self._ordered_columns = (
                result_columns,
                ordered_columns,
            )

    def _process_positional(self):
        assert not self.positiontup
        assert self.state is CompilerState.STRING_APPLIED
        assert not self._numeric_binds

        if self.dialect.paramstyle == "format":
            placeholder = "%s"
        else:
            assert self.dialect.paramstyle == "qmark"
            placeholder = "?"

        positions = []

        def find_position(m: re.Match[str]) -> str:
            normal_bind = m.group(1)
            if normal_bind:
                positions.append(normal_bind)
                return placeholder
            else:
                # this a post-compile bind
                positions.append(m.group(2))
                return m.group(0)

        self.string = re.sub(
            self._positional_pattern, find_position, self.string
        )

        if self.escaped_bind_names:
            reverse_escape = {v: k for k, v in self.escaped_bind_names.items()}
            assert len(self.escaped_bind_names) == len(reverse_escape)
            self.positiontup = [
                reverse_escape.get(name, name) for name in positions
            ]
        else:
            self.positiontup = positions

        if self._insertmanyvalues:
            positions = []

            single_values_expr = re.sub(
                self._positional_pattern,
                find_position,
                self._insertmanyvalues.single_values_expr,
            )
            insert_crud_params = [
                (
                    v[0],
                    v[1],
                    re.sub(self._positional_pattern, find_position, v[2]),
                    v[3],
                )
                for v in self._insertmanyvalues.insert_crud_params
            ]

            self._insertmanyvalues = self._insertmanyvalues._replace(
                single_values_expr=single_values_expr,
                insert_crud_params=insert_crud_params,
            )

    def _process_numeric(self):
        assert self._numeric_binds
        assert self.state is CompilerState.STRING_APPLIED

        num = 1
        param_pos: Dict[str, str] = {}
        order: Iterable[str]
        if self._insertmanyvalues and self._values_bindparam is not None:
            # bindparams that are not in values are always placed first.
            # this avoids the need of changing them when using executemany
            # values () ()
            order = itertools.chain(
                (
                    name
                    for name in self.bind_names.values()
                    if name not in self._values_bindparam
                ),
                self.bind_names.values(),
            )
        else:
            order = self.bind_names.values()

        for bind_name in order:
            if bind_name in param_pos:
                continue
            bind = self.binds[bind_name]
            if (
                bind in self.post_compile_params
                or bind in self.literal_execute_params
            ):
                # set to None to just mark the in positiontup, it will not
                # be replaced below.
                param_pos[bind_name] = None  # type: ignore
            else:
                ph = f"{self._numeric_binds_identifier_char}{num}"
                num += 1
                param_pos[bind_name] = ph

        self.next_numeric_pos = num

        self.positiontup = list(param_pos)
        if self.escaped_bind_names:
            len_before = len(param_pos)
            param_pos = {
                self.escaped_bind_names.get(name, name): pos
                for name, pos in param_pos.items()
            }
            assert len(param_pos) == len_before

        # Can't use format here since % chars are not escaped.
        self.string = self._pyformat_pattern.sub(
            lambda m: param_pos[m.group(1)], self.string
        )

        if self._insertmanyvalues:
            single_values_expr = (
                # format is ok here since single_values_expr includes only
                # place-holders
                self._insertmanyvalues.single_values_expr
                % param_pos
            )
            insert_crud_params = [
                (v[0], v[1], "%s", v[3])
                for v in self._insertmanyvalues.insert_crud_params
            ]

            self._insertmanyvalues = self._insertmanyvalues._replace(
                # This has the numbers (:1, :2)
                single_values_expr=single_values_expr,
                # The single binds are instead %s so they can be formatted
                insert_crud_params=insert_crud_params,
            )

    @util.memoized_property
    def _bind_processors(
        self,
    ) -> MutableMapping[
        str, Union[_BindProcessorType[Any], Sequence[_BindProcessorType[Any]]]
    ]:
        # mypy is not able to see the two value types as the above Union,
        # it just sees "object".  don't know how to resolve
        return {
            key: value  # type: ignore
            for key, value in (
                (
                    self.bind_names[bindparam],
                    (
                        bindparam.type._cached_bind_processor(self.dialect)
                        if not bindparam.type._is_tuple_type
                        else tuple(
                            elem_type._cached_bind_processor(self.dialect)
                            for elem_type in cast(
                                TupleType, bindparam.type
                            ).types
                        )
                    ),
                )
                for bindparam in self.bind_names
            )
            if value is not None
        }

    def is_subquery(self):
        return len(self.stack) > 1

    @property
    def sql_compiler(self) -> Self:
        return self

    def construct_expanded_state(
        self,
        params: Optional[_CoreSingleExecuteParams] = None,
        escape_names: bool = True,
    ) -> ExpandedState:
        """Return a new :class:`.ExpandedState` for a given parameter set.

        For queries that use "expanding" or other late-rendered parameters,
        this method will provide for both the finalized SQL string as well
        as the parameters that would be used for a particular parameter set.

        .. versionadded:: 2.0.0rc1

        """
        parameters = self.construct_params(
            params,
            escape_names=escape_names,
            _no_postcompile=True,
        )
        return self._process_parameters_for_postcompile(
            parameters,
        )

    def construct_params(
        self,
        params: Optional[_CoreSingleExecuteParams] = None,
        extracted_parameters: Optional[Sequence[BindParameter[Any]]] = None,
        escape_names: bool = True,
        _group_number: Optional[int] = None,
        _check: bool = True,
        _no_postcompile: bool = False,
    ) -> _MutableCoreSingleExecuteParams:
        """return a dictionary of bind parameter keys and values"""

        if self._render_postcompile and not _no_postcompile:
            assert self._post_compile_expanded_state is not None
            if not params:
                return dict(self._post_compile_expanded_state.parameters)
            else:
                raise exc.InvalidRequestError(
                    "can't construct new parameters when render_postcompile "
                    "is used; the statement is hard-linked to the original "
                    "parameters.  Use construct_expanded_state to generate a "
                    "new statement and parameters."
                )

        has_escaped_names = escape_names and bool(self.escaped_bind_names)

        if extracted_parameters:
            # related the bound parameters collected in the original cache key
            # to those collected in the incoming cache key.  They will not have
            # matching names but they will line up positionally in the same
            # way.   The parameters present in self.bind_names may be clones of
            # these original cache key params in the case of DML but the .key
            # will be guaranteed to match.
            if self.cache_key is None:
                raise exc.CompileError(
                    "This compiled object has no original cache key; "
                    "can't pass extracted_parameters to construct_params"
                )
            else:
                orig_extracted = self.cache_key[1]

            ckbm_tuple = self._cache_key_bind_match
            assert ckbm_tuple is not None
            ckbm, _ = ckbm_tuple
            resolved_extracted = {
                bind: extracted
                for b, extracted in zip(orig_extracted, extracted_parameters)
                for bind in ckbm[b]
            }
        else:
            resolved_extracted = None

        if params:
            pd = {}
            for bindparam, name in self.bind_names.items():
                escaped_name = (
                    self.escaped_bind_names.get(name, name)
                    if has_escaped_names
                    else name
                )

                if bindparam.key in params:
                    pd[escaped_name] = params[bindparam.key]
                elif name in params:
                    pd[escaped_name] = params[name]

                elif _check and bindparam.required:
                    if _group_number:
                        raise exc.InvalidRequestError(
                            "A value is required for bind parameter %r, "
                            "in parameter group %d"
                            % (bindparam.key, _group_number),
                            code="cd3x",
                        )
                    else:
                        raise exc.InvalidRequestError(
                            "A value is required for bind parameter %r"
                            % bindparam.key,
                            code="cd3x",
                        )
                else:
                    if resolved_extracted:
                        value_param = resolved_extracted.get(
                            bindparam, bindparam
                        )
                    else:
                        value_param = bindparam

                    if bindparam.callable:
                        pd[escaped_name] = value_param.effective_value
                    else:
                        pd[escaped_name] = value_param.value
            return pd
        else:
            pd = {}
            for bindparam, name in self.bind_names.items():
                escaped_name = (
                    self.escaped_bind_names.get(name, name)
                    if has_escaped_names
                    else name
                )

                if _check and bindparam.required:
                    if _group_number:
                        raise exc.InvalidRequestError(
                            "A value is required for bind parameter %r, "
                            "in parameter group %d"
                            % (bindparam.key, _group_number),
                            code="cd3x",
                        )
                    else:
                        raise exc.InvalidRequestError(
                            "A value is required for bind parameter %r"
                            % bindparam.key,
                            code="cd3x",
                        )

                if resolved_extracted:
                    value_param = resolved_extracted.get(bindparam, bindparam)
                else:
                    value_param = bindparam

                if bindparam.callable:
                    pd[escaped_name] = value_param.effective_value
                else:
                    pd[escaped_name] = value_param.value

            return pd

    @util.memoized_instancemethod
    def _get_set_input_sizes_lookup(self):
        dialect = self.dialect

        include_types = dialect.include_set_input_sizes
        exclude_types = dialect.exclude_set_input_sizes

        dbapi = dialect.dbapi

        def lookup_type(typ):
            dbtype = typ._unwrapped_dialect_impl(dialect).get_dbapi_type(dbapi)

            if (
                dbtype is not None
                and (exclude_types is None or dbtype not in exclude_types)
                and (include_types is None or dbtype in include_types)
            ):
                return dbtype
            else:
                return None

        inputsizes = {}

        literal_execute_params = self.literal_execute_params

        for bindparam in self.bind_names:
            if bindparam in literal_execute_params:
                continue

            if bindparam.type._is_tuple_type:
                inputsizes[bindparam] = [
                    lookup_type(typ)
                    for typ in cast(TupleType, bindparam.type).types
                ]
            else:
                inputsizes[bindparam] = lookup_type(bindparam.type)

        return inputsizes

    @property
    def params(self):
        """Return the bind param dictionary embedded into this
        compiled object, for those values that are present.

        .. seealso::

            :ref:`faq_sql_expression_string` - includes a usage example for
            debugging use cases.

        """
        return self.construct_params(_check=False)

    def _process_parameters_for_postcompile(
        self,
        parameters: _MutableCoreSingleExecuteParams,
        _populate_self: bool = False,
    ) -> ExpandedState:
        """handle special post compile parameters.

        These include:

        * "expanding" parameters -typically IN tuples that are rendered
          on a per-parameter basis for an otherwise fixed SQL statement string.

        * literal_binds compiled with the literal_execute flag.  Used for
          things like SQL Server "TOP N" where the driver does not accommodate
          N as a bound parameter.

        """

        expanded_parameters = {}
        new_positiontup: Optional[List[str]]

        pre_expanded_string = self._pre_expanded_string
        if pre_expanded_string is None:
            pre_expanded_string = self.string

        if self.positional:
            new_positiontup = []

            pre_expanded_positiontup = self._pre_expanded_positiontup
            if pre_expanded_positiontup is None:
                pre_expanded_positiontup = self.positiontup

        else:
            new_positiontup = pre_expanded_positiontup = None

        processors = self._bind_processors
        single_processors = cast(
            "Mapping[str, _BindProcessorType[Any]]", processors
        )
        tuple_processors = cast(
            "Mapping[str, Sequence[_BindProcessorType[Any]]]", processors
        )

        new_processors: Dict[str, _BindProcessorType[Any]] = {}

        replacement_expressions: Dict[str, Any] = {}
        to_update_sets: Dict[str, Any] = {}

        # notes:
        # *unescaped* parameter names in:
        # self.bind_names, self.binds, self._bind_processors, self.positiontup
        #
        # *escaped* parameter names in:
        # construct_params(), replacement_expressions

        numeric_positiontup: Optional[List[str]] = None

        if self.positional and pre_expanded_positiontup is not None:
            names: Iterable[str] = pre_expanded_positiontup
            if self._numeric_binds:
                numeric_positiontup = []
        else:
            names = self.bind_names.values()

        ebn = self.escaped_bind_names
        for name in names:
            escaped_name = ebn.get(name, name) if ebn else name
            parameter = self.binds[name]

            if parameter in self.literal_execute_params:
                if escaped_name not in replacement_expressions:
                    replacement_expressions[escaped_name] = (
                        self.render_literal_bindparam(
                            parameter,
                            render_literal_value=parameters.pop(escaped_name),
                        )
                    )
                continue

            if parameter in self.post_compile_params:
                if escaped_name in replacement_expressions:
                    to_update = to_update_sets[escaped_name]
                    values = None
                else:
                    # we are removing the parameter from parameters
                    # because it is a list value, which is not expected by
                    # TypeEngine objects that would otherwise be asked to
                    # process it. the single name is being replaced with
                    # individual numbered parameters for each value in the
                    # param.
                    #
                    # note we are also inserting *escaped* parameter names
                    # into the given dictionary.   default dialect will
                    # use these param names directly as they will not be
                    # in the escaped_bind_names dictionary.
                    values = parameters.pop(name)

                    leep_res = self._literal_execute_expanding_parameter(
                        escaped_name, parameter, values
                    )
                    (to_update, replacement_expr) = leep_res

                    to_update_sets[escaped_name] = to_update
                    replacement_expressions[escaped_name] = replacement_expr

                if not parameter.literal_execute:
                    parameters.update(to_update)
                    if parameter.type._is_tuple_type:
                        assert values is not None
                        new_processors.update(
                            (
                                "%s_%s_%s" % (name, i, j),
                                tuple_processors[name][j - 1],
                            )
                            for i, tuple_element in enumerate(values, 1)
                            for j, _ in enumerate(tuple_element, 1)
                            if name in tuple_processors
                            and tuple_processors[name][j - 1] is not None
                        )
                    else:
                        new_processors.update(
                            (key, single_processors[name])
                            for key, _ in to_update
                            if name in single_processors
                        )
                    if numeric_positiontup is not None:
                        numeric_positiontup.extend(
                            name for name, _ in to_update
                        )
                    elif new_positiontup is not None:
                        # to_update has escaped names, but that's ok since
                        # these are new names, that aren't in the
                        # escaped_bind_names dict.
                        new_positiontup.extend(name for name, _ in to_update)
                    expanded_parameters[name] = [
                        expand_key for expand_key, _ in to_update
                    ]
            elif new_positiontup is not None:
                new_positiontup.append(name)

        def process_expanding(m):
            key = m.group(1)
            expr = replacement_expressions[key]

            # if POSTCOMPILE included a bind_expression, render that
            # around each element
            if m.group(2):
                tok = m.group(2).split("~~")
                be_left, be_right = tok[1], tok[3]
                expr = ", ".join(
                    "%s%s%s" % (be_left, exp, be_right)
                    for exp in expr.split(", ")
                )
            return expr

        statement = re.sub(
            self._post_compile_pattern, process_expanding, pre_expanded_string
        )

        if numeric_positiontup is not None:
            assert new_positiontup is not None
            param_pos = {
                key: f"{self._numeric_binds_identifier_char}{num}"
                for num, key in enumerate(
                    numeric_positiontup, self.next_numeric_pos
                )
            }
            # Can't use format here since % chars are not escaped.
            statement = self._pyformat_pattern.sub(
                lambda m: param_pos[m.group(1)], statement
            )
            new_positiontup.extend(numeric_positiontup)

        expanded_state = ExpandedState(
            statement,
            parameters,
            new_processors,
            new_positiontup,
            expanded_parameters,
        )

        if _populate_self:
            # this is for the "render_postcompile" flag, which is not
            # otherwise used internally and is for end-user debugging and
            # special use cases.
            self._pre_expanded_string = pre_expanded_string
            self._pre_expanded_positiontup = pre_expanded_positiontup
            self.string = expanded_state.statement
            self.positiontup = (
                list(expanded_state.positiontup or ())
                if self.positional
                else None
            )
            self._post_compile_expanded_state = expanded_state

        return expanded_state

    @util.preload_module("sqlalchemy.engine.cursor")
    def _create_result_map(self):
        """utility method used for unit tests only."""
        cursor = util.preloaded.engine_cursor
        return cursor.CursorResultMetaData._create_description_match_map(
            self._result_columns
        )

    # assigned by crud.py for insert/update statements
    _get_bind_name_for_col: _BindNameForColProtocol

    @util.memoized_property
    def _within_exec_param_key_getter(self) -> Callable[[Any], str]:
        getter = self._get_bind_name_for_col
        return getter

    @util.memoized_property
    @util.preload_module("sqlalchemy.engine.result")
    def _inserted_primary_key_from_lastrowid_getter(self):
        result = util.preloaded.engine_result

        param_key_getter = self._within_exec_param_key_getter

        assert self.compile_state is not None
        statement = self.compile_state.statement

        if TYPE_CHECKING:
            assert isinstance(statement, Insert)

        table = statement.table

        getters = [
            (operator.methodcaller("get", param_key_getter(col), None), col)
            for col in table.primary_key
        ]

        autoinc_getter = None
        autoinc_col = table._autoincrement_column
        if autoinc_col is not None:
            # apply type post processors to the lastrowid
            lastrowid_processor = autoinc_col.type._cached_result_processor(
                self.dialect, None
            )
            autoinc_key = param_key_getter(autoinc_col)

            # if a bind value is present for the autoincrement column
            # in the parameters, we need to do the logic dictated by
            # #7998; honor a non-None user-passed parameter over lastrowid.
            # previously in the 1.4 series we weren't fetching lastrowid
            # at all if the key were present in the parameters
            if autoinc_key in self.binds:

                def _autoinc_getter(lastrowid, parameters):
                    param_value = parameters.get(autoinc_key, lastrowid)
                    if param_value is not None:
                        # they supplied non-None parameter, use that.
                        # SQLite at least is observed to return the wrong
                        # cursor.lastrowid for INSERT..ON CONFLICT so it
                        # can't be used in all cases
                        return param_value
                    else:
                        # use lastrowid
                        return lastrowid

                # work around mypy https://github.com/python/mypy/issues/14027
                autoinc_getter = _autoinc_getter

        else:
            lastrowid_processor = None

        row_fn = result.result_tuple([col.key for col in table.primary_key])

        def get(lastrowid, parameters):
            """given cursor.lastrowid value and the parameters used for INSERT,
            return a "row" that represents the primary key, either by
            using the "lastrowid" or by extracting values from the parameters
            that were sent along with the INSERT.

            """
            if lastrowid_processor is not None:
                lastrowid = lastrowid_processor(lastrowid)

            if lastrowid is None:
                return row_fn(getter(parameters) for getter, col in getters)
            else:
                return row_fn(
                    (
                        (
                            autoinc_getter(lastrowid, parameters)
                            if autoinc_getter is not None
                            else lastrowid
                        )
                        if col is autoinc_col
                        else getter(parameters)
                    )
                    for getter, col in getters
                )

        return get

    @util.memoized_property
    @util.preload_module("sqlalchemy.engine.result")
    def _inserted_primary_key_from_returning_getter(self):
        if typing.TYPE_CHECKING:
            from ..engine import result
        else:
            result = util.preloaded.engine_result

        assert self.compile_state is not None
        statement = self.compile_state.statement

        if TYPE_CHECKING:
            assert isinstance(statement, Insert)

        param_key_getter = self._within_exec_param_key_getter
        table = statement.table

        returning = self.implicit_returning
        assert returning is not None
        ret = {col: idx for idx, col in enumerate(returning)}

        getters = cast(
            "List[Tuple[Callable[[Any], Any], bool]]",
            [
                (
                    (operator.itemgetter(ret[col]), True)
                    if col in ret
                    else (
                        operator.methodcaller(
                            "get", param_key_getter(col), None
                        ),
                        False,
                    )
                )
                for col in table.primary_key
            ],
        )

        row_fn = result.result_tuple([col.key for col in table.primary_key])

        def get(row, parameters):
            return row_fn(
                getter(row) if use_row else getter(parameters)
                for getter, use_row in getters
            )

        return get

    def default_from(self) -> str:
        """Called when a SELECT statement has no froms, and no FROM clause is
        to be appended.

        Gives Oracle Database a chance to tack on a ``FROM DUAL`` to the string
        output.

        """
        return ""

    def visit_override_binds(self, override_binds, **kw):
        """SQL compile the nested element of an _OverrideBinds with
        bindparams swapped out.

        The _OverrideBinds is not normally expected to be compiled; it
        is meant to be used when an already cached statement is to be used,
        the compilation was already performed, and only the bound params should
        be swapped in at execution time.

        However, there are test cases that exericise this object, and
        additionally the ORM subquery loader is known to feed in expressions
        which include this construct into new queries (discovered in #11173),
        so it has to do the right thing at compile time as well.

        """

        # get SQL text first
        sqltext = override_binds.element._compiler_dispatch(self, **kw)

        # for a test compile that is not for caching, change binds after the
        # fact.  note that we don't try to
        # swap the bindparam as we compile, because our element may be
        # elsewhere in the statement already (e.g. a subquery or perhaps a
        # CTE) and was already visited / compiled. See
        # test_relationship_criteria.py ->
        #    test_selectinload_local_criteria_subquery
        for k in override_binds.translate:
            if k not in self.binds:
                continue
            bp = self.binds[k]

            # so this would work, just change the value of bp in place.
            # but we dont want to mutate things outside.
            # bp.value = override_binds.translate[bp.key]
            # continue

            # instead, need to replace bp with new_bp or otherwise accommodate
            # in all internal collections
            new_bp = bp._with_value(
                override_binds.translate[bp.key],
                maintain_key=True,
                required=False,
            )

            name = self.bind_names[bp]
            self.binds[k] = self.binds[name] = new_bp
            self.bind_names[new_bp] = name
            self.bind_names.pop(bp, None)

            if bp in self.post_compile_params:
                self.post_compile_params |= {new_bp}
            if bp in self.literal_execute_params:
                self.literal_execute_params |= {new_bp}

            ckbm_tuple = self._cache_key_bind_match
            if ckbm_tuple:
                ckbm, cksm = ckbm_tuple
                for bp in bp._cloned_set:
                    if bp.key in cksm:
                        cb = cksm[bp.key]
                        ckbm[cb].append(new_bp)

        return sqltext

    def visit_grouping(self, grouping, asfrom=False, **kwargs):
        return "(" + grouping.element._compiler_dispatch(self, **kwargs) + ")"

    def visit_select_statement_grouping(self, grouping, **kwargs):
        return "(" + grouping.element._compiler_dispatch(self, **kwargs) + ")"

    def visit_label_reference(
        self, element, within_columns_clause=False, **kwargs
    ):
        if self.stack and self.dialect.supports_simple_order_by_label:
            try:
                compile_state = cast(
                    "Union[SelectState, CompoundSelectState]",
                    self.stack[-1]["compile_state"],
                )
            except KeyError as ke:
                raise exc.CompileError(
                    "Can't resolve label reference for ORDER BY / "
                    "GROUP BY / DISTINCT etc."
                ) from ke

            (
                with_cols,
                only_froms,
                only_cols,
            ) = compile_state._label_resolve_dict
            if within_columns_clause:
                resolve_dict = only_froms
            else:
                resolve_dict = only_cols

            # this can be None in the case that a _label_reference()
            # were subject to a replacement operation, in which case
            # the replacement of the Label element may have changed
            # to something else like a ColumnClause expression.
            order_by_elem = element.element._order_by_label_element

            if (
                order_by_elem is not None
                and order_by_elem.name in resolve_dict
                and order_by_elem.shares_lineage(
                    resolve_dict[order_by_elem.name]
                )
            ):
                kwargs["render_label_as_label"] = (
                    element.element._order_by_label_element
                )
        return self.process(
            element.element,
            within_columns_clause=within_columns_clause,
            **kwargs,
        )

    def visit_textual_label_reference(
        self, element, within_columns_clause=False, **kwargs
    ):
        if not self.stack:
            # compiling the element outside of the context of a SELECT
            return self.process(element._text_clause)

        try:
            compile_state = cast(
                "Union[SelectState, CompoundSelectState]",
                self.stack[-1]["compile_state"],
            )
        except KeyError as ke:
            coercions._no_text_coercion(
                element.element,
                extra=(
                    "Can't resolve label reference for ORDER BY / "
                    "GROUP BY / DISTINCT etc."
                ),
                exc_cls=exc.CompileError,
                err=ke,
            )

        with_cols, only_froms, only_cols = compile_state._label_resolve_dict
        try:
            if within_columns_clause:
                col = only_froms[element.element]
            else:
                col = with_cols[element.element]
        except KeyError as err:
            coercions._no_text_coercion(
                element.element,
                extra=(
                    "Can't resolve label reference for ORDER BY / "
                    "GROUP BY / DISTINCT etc."
                ),
                exc_cls=exc.CompileError,
                err=err,
            )
        else:
            kwargs["render_label_as_label"] = col
            return self.process(
                col, within_columns_clause=within_columns_clause, **kwargs
            )

    def visit_label(
        self,
        label,
        add_to_result_map=None,
        within_label_clause=False,
        within_columns_clause=False,
        render_label_as_label=None,
        result_map_targets=(),
        **kw,
    ):
        # only render labels within the columns clause
        # or ORDER BY clause of a select.  dialect-specific compilers
        # can modify this behavior.
        render_label_with_as = (
            within_columns_clause and not within_label_clause
        )
        render_label_only = render_label_as_label is label

        if render_label_only or render_label_with_as:
            if isinstance(label.name, elements._truncated_label):
                labelname = self._truncated_identifier("colident", label.name)
            else:
                labelname = label.name

        if render_label_with_as:
            if add_to_result_map is not None:
                add_to_result_map(
                    labelname,
                    label.name,
                    (label, labelname) + label._alt_names + result_map_targets,
                    label.type,
                )
            return (
                label.element._compiler_dispatch(
                    self,
                    within_columns_clause=True,
                    within_label_clause=True,
                    **kw,
                )
                + OPERATORS[operators.as_]
                + self.preparer.format_label(label, labelname)
            )
        elif render_label_only:
            return self.preparer.format_label(label, labelname)
        else:
            return label.element._compiler_dispatch(
                self, within_columns_clause=False, **kw
            )

    def _fallback_column_name(self, column):
        raise exc.CompileError(
            "Cannot compile Column object until its 'name' is assigned."
        )

    def visit_lambda_element(self, element, **kw):
        sql_element = element._resolved
        return self.process(sql_element, **kw)

    def visit_column(
        self,
        column: ColumnClause[Any],
        add_to_result_map: Optional[_ResultMapAppender] = None,
        include_table: bool = True,
        result_map_targets: Tuple[Any, ...] = (),
        ambiguous_table_name_map: Optional[_AmbiguousTableNameMap] = None,
        **kwargs: Any,
    ) -> str:
        name = orig_name = column.name
        if name is None:
            name = self._fallback_column_name(column)

        is_literal = column.is_literal
        if not is_literal and isinstance(name, elements._truncated_label):
            name = self._truncated_identifier("colident", name)

        if add_to_result_map is not None:
            targets = (column, name, column.key) + result_map_targets
            if column._tq_label:
                targets += (column._tq_label,)

            add_to_result_map(name, orig_name, targets, column.type)

        if is_literal:
            # note we are not currently accommodating for
            # literal_column(quoted_name('ident', True)) here
            name = self.escape_literal_column(name)
        else:
            name = self.preparer.quote(name)
        table = column.table
        if table is None or not include_table or not table.named_with_column:
            return name
        else:
            effective_schema = self.preparer.schema_for_object(table)

            if effective_schema:
                schema_prefix = (
                    self.preparer.quote_schema(effective_schema) + "."
                )
            else:
                schema_prefix = ""

            if TYPE_CHECKING:
                assert isinstance(table, NamedFromClause)
            tablename = table.name

            if (
                not effective_schema
                and ambiguous_table_name_map
                and tablename in ambiguous_table_name_map
            ):
                tablename = ambiguous_table_name_map[tablename]

            if isinstance(tablename, elements._truncated_label):
                tablename = self._truncated_identifier("alias", tablename)

            return schema_prefix + self.preparer.quote(tablename) + "." + name

    def visit_collation(self, element, **kw):
        return self.preparer.format_collation(element.collation)

    def visit_fromclause(self, fromclause, **kwargs):
        return fromclause.name

    def visit_index(self, index, **kwargs):
        return index.name

    def visit_typeclause(self, typeclause, **kw):
        kw["type_expression"] = typeclause
        kw["identifier_preparer"] = self.preparer
        return self.dialect.type_compiler_instance.process(
            typeclause.type, **kw
        )

    def post_process_text(self, text):
        if self.preparer._double_percents:
            text = text.replace("%", "%%")
        return text

    def escape_literal_column(self, text):
        if self.preparer._double_percents:
            text = text.replace("%", "%%")
        return text

    def visit_textclause(self, textclause, add_to_result_map=None, **kw):
        def do_bindparam(m):
            name = m.group(1)
            if name in textclause._bindparams:
                return self.process(textclause._bindparams[name], **kw)
            else:
                return self.bindparam_string(name, **kw)

        if not self.stack:
            self.isplaintext = True

        if add_to_result_map:
            # text() object is present in the columns clause of a
            # select().   Add a no-name entry to the result map so that
            # row[text()] produces a result
            add_to_result_map(None, None, (textclause,), sqltypes.NULLTYPE)

        # un-escape any \:params
        return BIND_PARAMS_ESC.sub(
            lambda m: m.group(1),
            BIND_PARAMS.sub(
                do_bindparam, self.post_process_text(textclause.text)
            ),
        )

    def visit_textual_select(
        self, taf, compound_index=None, asfrom=False, **kw
    ):
        toplevel = not self.stack
        entry = self._default_stack_entry if toplevel else self.stack[-1]

        new_entry: _CompilerStackEntry = {
            "correlate_froms": set(),
            "asfrom_froms": set(),
            "selectable": taf,
        }
        self.stack.append(new_entry)

        if taf._independent_ctes:
            self._dispatch_independent_ctes(taf, kw)

        populate_result_map = (
            toplevel
            or (
                compound_index == 0
                and entry.get("need_result_map_for_compound", False)
            )
            or entry.get("need_result_map_for_nested", False)
        )

        if populate_result_map:
            self._ordered_columns = self._textual_ordered_columns = (
                taf.positional
            )

            # enable looser result column matching when the SQL text links to
            # Column objects by name only
            self._loose_column_name_matching = not taf.positional and bool(
                taf.column_args
            )

            for c in taf.column_args:
                self.process(
                    c,
                    within_columns_clause=True,
                    add_to_result_map=self._add_to_result_map,
                )

        text = self.process(taf.element, **kw)
        if self.ctes:
            nesting_level = len(self.stack) if not toplevel else None
            text = self._render_cte_clause(nesting_level=nesting_level) + text

        self.stack.pop(-1)

        return text

    def visit_null(self, expr: Null, **kw: Any) -> str:
        return "NULL"

    def visit_true(self, expr: True_, **kw: Any) -> str:
        if self.dialect.supports_native_boolean:
            return "true"
        else:
            return "1"

    def visit_false(self, expr: False_, **kw: Any) -> str:
        if self.dialect.supports_native_boolean:
            return "false"
        else:
            return "0"

    def _generate_delimited_list(self, elements, separator, **kw):
        return separator.join(
            s
            for s in (c._compiler_dispatch(self, **kw) for c in elements)
            if s
        )

    def _generate_delimited_and_list(self, clauses, **kw):
        lcc, clauses = elements.BooleanClauseList._process_clauses_for_boolean(
            operators.and_,
            elements.True_._singleton,
            elements.False_._singleton,
            clauses,
        )
        if lcc == 1:
            return clauses[0]._compiler_dispatch(self, **kw)
        else:
            separator = OPERATORS[operators.and_]
            return separator.join(
                s
                for s in (c._compiler_dispatch(self, **kw) for c in clauses)
                if s
            )

    def visit_tuple(self, clauselist, **kw):
        return "(%s)" % self.visit_clauselist(clauselist, **kw)

    def visit_clauselist(self, clauselist, **kw):
        sep = clauselist.operator
        if sep is None:
            sep = " "
        else:
            sep = OPERATORS[clauselist.operator]

        return self._generate_delimited_list(clauselist.clauses, sep, **kw)

    def visit_expression_clauselist(self, clauselist, **kw):
        operator_ = clauselist.operator

        disp = self._get_operator_dispatch(
            operator_, "expression_clauselist", None
        )
        if disp:
            return disp(clauselist, operator_, **kw)

        try:
            opstring = OPERATORS[operator_]
        except KeyError as err:
            raise exc.UnsupportedCompilationError(self, operator_) from err
        else:
            kw["_in_operator_expression"] = True
            return self._generate_delimited_list(
                clauselist.clauses, opstring, **kw
            )

    def visit_case(self, clause, **kwargs):
        x = "CASE "
        if clause.value is not None:
            x += clause.value._compiler_dispatch(self, **kwargs) + " "
        for cond, result in clause.whens:
            x += (
                "WHEN "
                + cond._compiler_dispatch(self, **kwargs)
                + " THEN "
                + result._compiler_dispatch(self, **kwargs)
                + " "
            )
        if clause.else_ is not None:
            x += (
                "ELSE " + clause.else_._compiler_dispatch(self, **kwargs) + " "
            )
        x += "END"
        return x

    def visit_type_coerce(self, type_coerce, **kw):
        return type_coerce.typed_expression._compiler_dispatch(self, **kw)

    def visit_cast(self, cast, **kwargs):
        type_clause = cast.typeclause._compiler_dispatch(self, **kwargs)
        match = re.match("(.*)( COLLATE .*)", type_clause)
        return "CAST(%s AS %s)%s" % (
            cast.clause._compiler_dispatch(self, **kwargs),
            match.group(1) if match else type_clause,
            match.group(2) if match else "",
        )

    def _format_frame_clause(self, range_, **kw):
        return "%s AND %s" % (
            (
                "UNBOUNDED PRECEDING"
                if range_[0] is elements.RANGE_UNBOUNDED
                else (
                    "CURRENT ROW"
                    if range_[0] is elements.RANGE_CURRENT
                    else (
                        "%s PRECEDING"
                        % (
                            self.process(
                                elements.literal(abs(range_[0])), **kw
                            ),
                        )
                        if range_[0] < 0
                        else "%s FOLLOWING"
                        % (self.process(elements.literal(range_[0]), **kw),)
                    )
                )
            ),
            (
                "UNBOUNDED FOLLOWING"
                if range_[1] is elements.RANGE_UNBOUNDED
                else (
                    "CURRENT ROW"
                    if range_[1] is elements.RANGE_CURRENT
                    else (
                        "%s PRECEDING"
                        % (
                            self.process(
                                elements.literal(abs(range_[1])), **kw
                            ),
                        )
                        if range_[1] < 0
                        else "%s FOLLOWING"
                        % (self.process(elements.literal(range_[1]), **kw),)
                    )
                )
            ),
        )

    def visit_over(self, over, **kwargs):
        text = over.element._compiler_dispatch(self, **kwargs)
        if over.range_ is not None:
            range_ = "RANGE BETWEEN %s" % self._format_frame_clause(
                over.range_, **kwargs
            )
        elif over.rows is not None:
            range_ = "ROWS BETWEEN %s" % self._format_frame_clause(
                over.rows, **kwargs
            )
        elif over.groups is not None:
            range_ = "GROUPS BETWEEN %s" % self._format_frame_clause(
                over.groups, **kwargs
            )
        else:
            range_ = None

        return "%s OVER (%s)" % (
            text,
            " ".join(
                [
                    "%s BY %s"
                    % (word, clause._compiler_dispatch(self, **kwargs))
                    for word, clause in (
                        ("PARTITION", over.partition_by),
                        ("ORDER", over.order_by),
                    )
                    if clause is not None and len(clause)
                ]
                + ([range_] if range_ else [])
            ),
        )

    def visit_withingroup(self, withingroup, **kwargs):
        return "%s WITHIN GROUP (ORDER BY %s)" % (
            withingroup.element._compiler_dispatch(self, **kwargs),
            withingroup.order_by._compiler_dispatch(self, **kwargs),
        )

    def visit_funcfilter(self, funcfilter, **kwargs):
        return "%s FILTER (WHERE %s)" % (
            funcfilter.func._compiler_dispatch(self, **kwargs),
            funcfilter.criterion._compiler_dispatch(self, **kwargs),
        )

    def visit_extract(self, extract, **kwargs):
        field = self.extract_map.get(extract.field, extract.field)
        return "EXTRACT(%s FROM %s)" % (
            field,
            extract.expr._compiler_dispatch(self, **kwargs),
        )

    def visit_scalar_function_column(self, element, **kw):
        compiled_fn = self.visit_function(element.fn, **kw)
        compiled_col = self.visit_column(element, **kw)
        return "(%s).%s" % (compiled_fn, compiled_col)

    def visit_function(
        self,
        func: Function[Any],
        add_to_result_map: Optional[_ResultMapAppender] = None,
        **kwargs: Any,
    ) -> str:
        if add_to_result_map is not None:
            add_to_result_map(func.name, func.name, (func.name,), func.type)

        disp = getattr(self, "visit_%s_func" % func.name.lower(), None)

        text: str

        if disp:
            text = disp(func, **kwargs)
        else:
            name = FUNCTIONS.get(func._deannotate().__class__, None)
            if name:
                if func._has_args:
                    name += "%(expr)s"
            else:
                name = func.name
                name = (
                    self.preparer.quote(name)
                    if self.preparer._requires_quotes_illegal_chars(name)
                    or isinstance(name, elements.quoted_name)
                    else name
                )
                name = name + "%(expr)s"
            text = ".".join(
                [
                    (
                        self.preparer.quote(tok)
                        if self.preparer._requires_quotes_illegal_chars(tok)
                        or isinstance(name, elements.quoted_name)
                        else tok
                    )
                    for tok in func.packagenames
                ]
                + [name]
            ) % {"expr": self.function_argspec(func, **kwargs)}

        if func._with_ordinality:
            text += " WITH ORDINALITY"
        return text

    def visit_next_value_func(self, next_value, **kw):
        return self.visit_sequence(next_value.sequence)

    def visit_sequence(self, sequence, **kw):
        raise NotImplementedError(
            "Dialect '%s' does not support sequence increments."
            % self.dialect.name
        )

    def function_argspec(self, func: Function[Any], **kwargs: Any) -> str:
        return func.clause_expr._compiler_dispatch(self, **kwargs)

    def visit_compound_select(
        self, cs, asfrom=False, compound_index=None, **kwargs
    ):
        toplevel = not self.stack

        compile_state = cs._compile_state_factory(cs, self, **kwargs)

        if toplevel and not self.compile_state:
            self.compile_state = compile_state

        compound_stmt = compile_state.statement

        entry = self._default_stack_entry if toplevel else self.stack[-1]
        need_result_map = toplevel or (
            not compound_index
            and entry.get("need_result_map_for_compound", False)
        )

        # indicates there is already a CompoundSelect in play
        if compound_index == 0:
            entry["select_0"] = cs

        self.stack.append(
            {
                "correlate_froms": entry["correlate_froms"],
                "asfrom_froms": entry["asfrom_froms"],
                "selectable": cs,
                "compile_state": compile_state,
                "need_result_map_for_compound": need_result_map,
            }
        )

        if compound_stmt._independent_ctes:
            self._dispatch_independent_ctes(compound_stmt, kwargs)

        keyword = self.compound_keywords[cs.keyword]

        text = (" " + keyword + " ").join(
            (
                c._compiler_dispatch(
                    self, asfrom=asfrom, compound_index=i, **kwargs
                )
                for i, c in enumerate(cs.selects)
            )
        )

        kwargs["include_table"] = False
        text += self.group_by_clause(cs, **dict(asfrom=asfrom, **kwargs))
        text += self.order_by_clause(cs, **kwargs)
        if cs._has_row_limiting_clause:
            text += self._row_limit_clause(cs, **kwargs)

        if self.ctes:
            nesting_level = len(self.stack) if not toplevel else None
            text = (
                self._render_cte_clause(
                    nesting_level=nesting_level,
                    include_following_stack=True,
                )
                + text
            )

        self.stack.pop(-1)
        return text

    def _row_limit_clause(self, cs, **kwargs):
        if cs._fetch_clause is not None:
            return self.fetch_clause(cs, **kwargs)
        else:
            return self.limit_clause(cs, **kwargs)

    def _get_operator_dispatch(self, operator_, qualifier1, qualifier2):
        attrname = "visit_%s_%s%s" % (
            operator_.__name__,
            qualifier1,
            "_" + qualifier2 if qualifier2 else "",
        )
        return getattr(self, attrname, None)

    def visit_unary(
        self, unary, add_to_result_map=None, result_map_targets=(), **kw
    ):
        if add_to_result_map is not None:
            result_map_targets += (unary,)
            kw["add_to_result_map"] = add_to_result_map
            kw["result_map_targets"] = result_map_targets

        if unary.operator:
            if unary.modifier:
                raise exc.CompileError(
                    "Unary expression does not support operator "
                    "and modifier simultaneously"
                )
            disp = self._get_operator_dispatch(
                unary.operator, "unary", "operator"
            )
            if disp:
                return disp(unary, unary.operator, **kw)
            else:
                return self._generate_generic_unary_operator(
                    unary, OPERATORS[unary.operator], **kw
                )
        elif unary.modifier:
            disp = self._get_operator_dispatch(
                unary.modifier, "unary", "modifier"
            )
            if disp:
                return disp(unary, unary.modifier, **kw)
            else:
                return self._generate_generic_unary_modifier(
                    unary, OPERATORS[unary.modifier], **kw
                )
        else:
            raise exc.CompileError(
                "Unary expression has no operator or modifier"
            )

    def visit_truediv_binary(self, binary, operator, **kw):
        if self.dialect.div_is_floordiv:
            return (
                self.process(binary.left, **kw)
                + " / "
                # TODO: would need a fast cast again here,
                # unless we want to use an implicit cast like "+ 0.0"
                + self.process(
                    elements.Cast(
                        binary.right,
                        (
                            binary.right.type
                            if binary.right.type._type_affinity
                            is sqltypes.Numeric
                            else sqltypes.Numeric()
                        ),
                    ),
                    **kw,
                )
            )
        else:
            return (
                self.process(binary.left, **kw)
                + " / "
                + self.process(binary.right, **kw)
            )

    def visit_floordiv_binary(self, binary, operator, **kw):
        if (
            self.dialect.div_is_floordiv
            and binary.right.type._type_affinity is sqltypes.Integer
        ):
            return (
                self.process(binary.left, **kw)
                + " / "
                + self.process(binary.right, **kw)
            )
        else:
            return "FLOOR(%s)" % (
                self.process(binary.left, **kw)
                + " / "
                + self.process(binary.right, **kw)
            )

    def visit_is_true_unary_operator(self, element, operator, **kw):
        if (
            element._is_implicitly_boolean
            or self.dialect.supports_native_boolean
        ):
            return self.process(element.element, **kw)
        else:
            return "%s = 1" % self.process(element.element, **kw)

    def visit_is_false_unary_operator(self, element, operator, **kw):
        if (
            element._is_implicitly_boolean
            or self.dialect.supports_native_boolean
        ):
            return "NOT %s" % self.process(element.element, **kw)
        else:
            return "%s = 0" % self.process(element.element, **kw)

    def visit_not_match_op_binary(self, binary, operator, **kw):
        return "NOT %s" % self.visit_binary(
            binary, override_operator=operators.match_op
        )

    def visit_not_in_op_binary(self, binary, operator, **kw):
        # The brackets are required in the NOT IN operation because the empty
        # case is handled using the form "(col NOT IN (null) OR 1 = 1)".
        # The presence of the OR makes the brackets required.
        return "(%s)" % self._generate_generic_binary(
            binary, OPERATORS[operator], **kw
        )

    def visit_empty_set_op_expr(self, type_, expand_op, **kw):
        if expand_op is operators.not_in_op:
            if len(type_) > 1:
                return "(%s)) OR (1 = 1" % (
                    ", ".join("NULL" for element in type_)
                )
            else:
                return "NULL) OR (1 = 1"
        elif expand_op is operators.in_op:
            if len(type_) > 1:
                return "(%s)) AND (1 != 1" % (
                    ", ".join("NULL" for element in type_)
                )
            else:
                return "NULL) AND (1 != 1"
        else:
            return self.visit_empty_set_expr(type_)

    def visit_empty_set_expr(self, element_types, **kw):
        raise NotImplementedError(
            "Dialect '%s' does not support empty set expression."
            % self.dialect.name
        )

    def _literal_execute_expanding_parameter_literal_binds(
        self, parameter, values, bind_expression_template=None
    ):
        typ_dialect_impl = parameter.type._unwrapped_dialect_impl(self.dialect)

        if not values:
            # empty IN expression.  note we don't need to use
            # bind_expression_template here because there are no
            # expressions to render.

            if typ_dialect_impl._is_tuple_type:
                replacement_expression = (
                    "VALUES " if self.dialect.tuple_in_values else ""
                ) + self.visit_empty_set_op_expr(
                    parameter.type.types, parameter.expand_op
                )

            else:
                replacement_expression = self.visit_empty_set_op_expr(
                    [parameter.type], parameter.expand_op
                )

        elif typ_dialect_impl._is_tuple_type or (
            typ_dialect_impl._isnull
            and isinstance(values[0], collections_abc.Sequence)
            and not isinstance(values[0], (str, bytes))
        ):
            if typ_dialect_impl._has_bind_expression:
                raise NotImplementedError(
                    "bind_expression() on TupleType not supported with "
                    "literal_binds"
                )

            replacement_expression = (
                "VALUES " if self.dialect.tuple_in_values else ""
            ) + ", ".join(
                "(%s)"
                % (
                    ", ".join(
                        self.render_literal_value(value, param_type)
                        for value, param_type in zip(
                            tuple_element, parameter.type.types
                        )
                    )
                )
                for i, tuple_element in enumerate(values)
            )
        else:
            if bind_expression_template:
                post_compile_pattern = self._post_compile_pattern
                m = post_compile_pattern.search(bind_expression_template)
                assert m and m.group(
                    2
                ), "unexpected format for expanding parameter"

                tok = m.group(2).split("~~")
                be_left, be_right = tok[1], tok[3]
                replacement_expression = ", ".join(
                    "%s%s%s"
                    % (
                        be_left,
                        self.render_literal_value(value, parameter.type),
                        be_right,
                    )
                    for value in values
                )
            else:
                replacement_expression = ", ".join(
                    self.render_literal_value(value, parameter.type)
                    for value in values
                )

        return (), replacement_expression

    def _literal_execute_expanding_parameter(self, name, parameter, values):
        if parameter.literal_execute:
            return self._literal_execute_expanding_parameter_literal_binds(
                parameter, values
            )

        dialect = self.dialect
        typ_dialect_impl = parameter.type._unwrapped_dialect_impl(dialect)

        if self._numeric_binds:
            bind_template = self.compilation_bindtemplate
        else:
            bind_template = self.bindtemplate

        if (
            self.dialect._bind_typing_render_casts
            and typ_dialect_impl.render_bind_cast
        ):

            def _render_bindtemplate(name):
                return self.render_bind_cast(
                    parameter.type,
                    typ_dialect_impl,
                    bind_template % {"name": name},
                )

        else:

            def _render_bindtemplate(name):
                return bind_template % {"name": name}

        if not values:
            to_update = []
            if typ_dialect_impl._is_tuple_type:
                replacement_expression = self.visit_empty_set_op_expr(
                    parameter.type.types, parameter.expand_op
                )
            else:
                replacement_expression = self.visit_empty_set_op_expr(
                    [parameter.type], parameter.expand_op
                )

        elif typ_dialect_impl._is_tuple_type or (
            typ_dialect_impl._isnull
            and isinstance(values[0], collections_abc.Sequence)
            and not isinstance(values[0], (str, bytes))
        ):
            assert not typ_dialect_impl._is_array
            to_update = [
                ("%s_%s_%s" % (name, i, j), value)
                for i, tuple_element in enumerate(values, 1)
                for j, value in enumerate(tuple_element, 1)
            ]

            replacement_expression = (
                "VALUES " if dialect.tuple_in_values else ""
            ) + ", ".join(
                "(%s)"
                % (
                    ", ".join(
                        _render_bindtemplate(
                            to_update[i * len(tuple_element) + j][0]
                        )
                        for j, value in enumerate(tuple_element)
                    )
                )
                for i, tuple_element in enumerate(values)
            )
        else:
            to_update = [
                ("%s_%s" % (name, i), value)
                for i, value in enumerate(values, 1)
            ]
            replacement_expression = ", ".join(
                _render_bindtemplate(key) for key, value in to_update
            )

        return to_update, replacement_expression

    def visit_binary(
        self,
        binary,
        override_operator=None,
        eager_grouping=False,
        from_linter=None,
        lateral_from_linter=None,
        **kw,
    ):
        if from_linter and operators.is_comparison(binary.operator):
            if lateral_from_linter is not None:
                enclosing_lateral = kw["enclosing_lateral"]
                lateral_from_linter.edges.update(
                    itertools.product(
                        _de_clone(
                            binary.left._from_objects + [enclosing_lateral]
                        ),
                        _de_clone(
                            binary.right._from_objects + [enclosing_lateral]
                        ),
                    )
                )
            else:
                from_linter.edges.update(
                    itertools.product(
                        _de_clone(binary.left._from_objects),
                        _de_clone(binary.right._from_objects),
                    )
                )

        # don't allow "? = ?" to render
        if (
            self.ansi_bind_rules
            and isinstance(binary.left, elements.BindParameter)
            and isinstance(binary.right, elements.BindParameter)
        ):
            kw["literal_execute"] = True

        operator_ = override_operator or binary.operator
        disp = self._get_operator_dispatch(operator_, "binary", None)
        if disp:
            return disp(binary, operator_, **kw)
        else:
            try:
                opstring = OPERATORS[operator_]
            except KeyError as err:
                raise exc.UnsupportedCompilationError(self, operator_) from err
            else:
                return self._generate_generic_binary(
                    binary,
                    opstring,
                    from_linter=from_linter,
                    lateral_from_linter=lateral_from_linter,
                    **kw,
                )

    def visit_function_as_comparison_op_binary(self, element, operator, **kw):
        return self.process(element.sql_function, **kw)

    def visit_mod_binary(self, binary, operator, **kw):
        if self.preparer._double_percents:
            return (
                self.process(binary.left, **kw)
                + " %% "
                + self.process(binary.right, **kw)
            )
        else:
            return (
                self.process(binary.left, **kw)
                + " % "
                + self.process(binary.right, **kw)
            )

    def visit_custom_op_binary(self, element, operator, **kw):
        kw["eager_grouping"] = operator.eager_grouping
        return self._generate_generic_binary(
            element,
            " " + self.escape_literal_column(operator.opstring) + " ",
            **kw,
        )

    def visit_custom_op_unary_operator(self, element, operator, **kw):
        return self._generate_generic_unary_operator(
            element, self.escape_literal_column(operator.opstring) + " ", **kw
        )

    def visit_custom_op_unary_modifier(self, element, operator, **kw):
        return self._generate_generic_unary_modifier(
            element, " " + self.escape_literal_column(operator.opstring), **kw
        )

    def _generate_generic_binary(
        self,
        binary: BinaryExpression[Any],
        opstring: str,
        eager_grouping: bool = False,
        **kw: Any,
    ) -> str:
        _in_operator_expression = kw.get("_in_operator_expression", False)

        kw["_in_operator_expression"] = True
        kw["_binary_op"] = binary.operator
        text = (
            binary.left._compiler_dispatch(
                self, eager_grouping=eager_grouping, **kw
            )
            + opstring
            + binary.right._compiler_dispatch(
                self, eager_grouping=eager_grouping, **kw
            )
        )

        if _in_operator_expression and eager_grouping:
            text = "(%s)" % text
        return text

    def _generate_generic_unary_operator(self, unary, opstring, **kw):
        return opstring + unary.element._compiler_dispatch(self, **kw)

    def _generate_generic_unary_modifier(self, unary, opstring, **kw):
        return unary.element._compiler_dispatch(self, **kw) + opstring

    @util.memoized_property
    def _like_percent_literal(self):
        return elements.literal_column("'%'", type_=sqltypes.STRINGTYPE)

    def visit_ilike_case_insensitive_operand(self, element, **kw):
        return f"lower({element.element._compiler_dispatch(self, **kw)})"

    def visit_contains_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.right = percent.concat(binary.right).concat(percent)
        return self.visit_like_op_binary(binary, operator, **kw)

    def visit_not_contains_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.right = percent.concat(binary.right).concat(percent)
        return self.visit_not_like_op_binary(binary, operator, **kw)

    def visit_icontains_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.left = ilike_case_insensitive(binary.left)
        binary.right = percent.concat(
            ilike_case_insensitive(binary.right)
        ).concat(percent)
        return self.visit_ilike_op_binary(binary, operator, **kw)

    def visit_not_icontains_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.left = ilike_case_insensitive(binary.left)
        binary.right = percent.concat(
            ilike_case_insensitive(binary.right)
        ).concat(percent)
        return self.visit_not_ilike_op_binary(binary, operator, **kw)

    def visit_startswith_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.right = percent._rconcat(binary.right)
        return self.visit_like_op_binary(binary, operator, **kw)

    def visit_not_startswith_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.right = percent._rconcat(binary.right)
        return self.visit_not_like_op_binary(binary, operator, **kw)

    def visit_istartswith_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.left = ilike_case_insensitive(binary.left)
        binary.right = percent._rconcat(ilike_case_insensitive(binary.right))
        return self.visit_ilike_op_binary(binary, operator, **kw)

    def visit_not_istartswith_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.left = ilike_case_insensitive(binary.left)
        binary.right = percent._rconcat(ilike_case_insensitive(binary.right))
        return self.visit_not_ilike_op_binary(binary, operator, **kw)

    def visit_endswith_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.right = percent.concat(binary.right)
        return self.visit_like_op_binary(binary, operator, **kw)

    def visit_not_endswith_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.right = percent.concat(binary.right)
        return self.visit_not_like_op_binary(binary, operator, **kw)

    def visit_iendswith_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.left = ilike_case_insensitive(binary.left)
        binary.right = percent.concat(ilike_case_insensitive(binary.right))
        return self.visit_ilike_op_binary(binary, operator, **kw)

    def visit_not_iendswith_op_binary(self, binary, operator, **kw):
        binary = binary._clone()
        percent = self._like_percent_literal
        binary.left = ilike_case_insensitive(binary.left)
        binary.right = percent.concat(ilike_case_insensitive(binary.right))
        return self.visit_not_ilike_op_binary(binary, operator, **kw)

    def visit_like_op_binary(self, binary, operator, **kw):
        escape = binary.modifiers.get("escape", None)

        return "%s LIKE %s" % (
            binary.left._compiler_dispatch(self, **kw),
            binary.right._compiler_dispatch(self, **kw),
        ) + (
            " ESCAPE " + self.render_literal_value(escape, sqltypes.STRINGTYPE)
            if escape is not None
            else ""
        )

    def visit_not_like_op_binary(self, binary, operator, **kw):
        escape = binary.modifiers.get("escape", None)
        return "%s NOT LIKE %s" % (
            binary.left._compiler_dispatch(self, **kw),
            binary.right._compiler_dispatch(self, **kw),
        ) + (
            " ESCAPE " + self.render_literal_value(escape, sqltypes.STRINGTYPE)
            if escape is not None
            else ""
        )

    def visit_ilike_op_binary(self, binary, operator, **kw):
        if operator is operators.ilike_op:
            binary = binary._clone()
            binary.left = ilike_case_insensitive(binary.left)
            binary.right = ilike_case_insensitive(binary.right)
        # else we assume ilower() has been applied

        return self.visit_like_op_binary(binary, operator, **kw)

    def visit_not_ilike_op_binary(self, binary, operator, **kw):
        if operator is operators.not_ilike_op:
            binary = binary._clone()
            binary.left = ilike_case_insensitive(binary.left)
            binary.right = ilike_case_insensitive(binary.right)
        # else we assume ilower() has been applied

        return self.visit_not_like_op_binary(binary, operator, **kw)

    def visit_between_op_binary(self, binary, operator, **kw):
        symmetric = binary.modifiers.get("symmetric", False)
        return self._generate_generic_binary(
            binary, " BETWEEN SYMMETRIC " if symmetric else " BETWEEN ", **kw
        )

    def visit_not_between_op_binary(self, binary, operator, **kw):
        symmetric = binary.modifiers.get("symmetric", False)
        return self._generate_generic_binary(
            binary,
            " NOT BETWEEN SYMMETRIC " if symmetric else " NOT BETWEEN ",
            **kw,
        )

    def visit_regexp_match_op_binary(
        self, binary: BinaryExpression[Any], operator: Any, **kw: Any
    ) -> str:
        raise exc.CompileError(
            "%s dialect does not support regular expressions"
            % self.dialect.name
        )

    def visit_not_regexp_match_op_binary(
        self, binary: BinaryExpression[Any], operator: Any, **kw: Any
    ) -> str:
        raise exc.CompileError(
            "%s dialect does not support regular expressions"
            % self.dialect.name
        )

    def visit_regexp_replace_op_binary(
        self, binary: BinaryExpression[Any], operator: Any, **kw: Any
    ) -> str:
        raise exc.CompileError(
            "%s dialect does not support regular expression replacements"
            % self.dialect.name
        )

    def visit_bindparam(
        self,
        bindparam,
        within_columns_clause=False,
        literal_binds=False,
        skip_bind_expression=False,
        literal_execute=False,
        render_postcompile=False,
        **kwargs,
    ):

        if not skip_bind_expression:
            impl = bindparam.type.dialect_impl(self.dialect)
            if impl._has_bind_expression:
                bind_expression = impl.bind_expression(bindparam)
                wrapped = self.process(
                    bind_expression,
                    skip_bind_expression=True,
                    within_columns_clause=within_columns_clause,
                    literal_binds=literal_binds and not bindparam.expanding,
                    literal_execute=literal_execute,
                    render_postcompile=render_postcompile,
                    **kwargs,
                )
                if bindparam.expanding:
                    # for postcompile w/ expanding, move the "wrapped" part
                    # of this into the inside

                    m = re.match(
                        r"^(.*)\(__\[POSTCOMPILE_(\S+?)\]\)(.*)$", wrapped
                    )
                    assert m, "unexpected format for expanding parameter"
                    wrapped = "(__[POSTCOMPILE_%s~~%s~~REPL~~%s~~])" % (
                        m.group(2),
                        m.group(1),
                        m.group(3),
                    )

                    if literal_binds:
                        ret = self.render_literal_bindparam(
                            bindparam,
                            within_columns_clause=True,
                            bind_expression_template=wrapped,
                            **kwargs,
                        )
                        return "(%s)" % ret

                return wrapped

        if not literal_binds:
            literal_execute = (
                literal_execute
                or bindparam.literal_execute
                or (within_columns_clause and self.ansi_bind_rules)
            )
            post_compile = literal_execute or bindparam.expanding
        else:
            post_compile = False

        if literal_binds:
            ret = self.render_literal_bindparam(
                bindparam, within_columns_clause=True, **kwargs
            )
            if bindparam.expanding:
                ret = "(%s)" % ret
            return ret

        name = self._truncate_bindparam(bindparam)

        if name in self.binds:
            existing = self.binds[name]
            if existing is not bindparam:
                if (
                    (existing.unique or bindparam.unique)
                    and not existing.proxy_set.intersection(
                        bindparam.proxy_set
                    )
                    and not existing._cloned_set.intersection(
                        bindparam._cloned_set
                    )
                ):
                    raise exc.CompileError(
                        "Bind parameter '%s' conflicts with "
                        "unique bind parameter of the same name" % name
                    )
                elif existing.expanding != bindparam.expanding:
                    raise exc.CompileError(
                        "Can't reuse bound parameter name '%s' in both "
                        "'expanding' (e.g. within an IN expression) and "
                        "non-expanding contexts.  If this parameter is to "
                        "receive a list/array value, set 'expanding=True' on "
                        "it for expressions that aren't IN, otherwise use "
                        "a different parameter name." % (name,)
                    )
                elif existing._is_crud or bindparam._is_crud:
                    if existing._is_crud and bindparam._is_crud:
                        # TODO: this condition is not well understood.
                        # see tests in test/sql/test_update.py
                        raise exc.CompileError(
                            "Encountered unsupported case when compiling an "
                            "INSERT or UPDATE statement.  If this is a "
                            "multi-table "
                            "UPDATE statement, please provide string-named "
                            "arguments to the "
                            "values() method with distinct names; support for "
                            "multi-table UPDATE statements that "
                            "target multiple tables for UPDATE is very "
                            "limited",
                        )
                    else:
                        raise exc.CompileError(
                            f"bindparam() name '{bindparam.key}' is reserved "
                            "for automatic usage in the VALUES or SET "
                            "clause of this "
                            "insert/update statement.   Please use a "
                            "name other than column name when using "
                            "bindparam() "
                            "with insert() or update() (for example, "
                            f"'b_{bindparam.key}')."
                        )

        self.binds[bindparam.key] = self.binds[name] = bindparam

        # if we are given a cache key that we're going to match against,
        # relate the bindparam here to one that is most likely present
        # in the "extracted params" portion of the cache key.  this is used
        # to set up a positional mapping that is used to determine the
        # correct parameters for a subsequent use of this compiled with
        # a different set of parameter values.   here, we accommodate for
        # parameters that may have been cloned both before and after the cache
        # key was been generated.
        ckbm_tuple = self._cache_key_bind_match

        if ckbm_tuple:
            ckbm, cksm = ckbm_tuple
            for bp in bindparam._cloned_set:
                if bp.key in cksm:
                    cb = cksm[bp.key]
                    ckbm[cb].append(bindparam)

        if bindparam.isoutparam:
            self.has_out_parameters = True

        if post_compile:
            if render_postcompile:
                self._render_postcompile = True

            if literal_execute:
                self.literal_execute_params |= {bindparam}
            else:
                self.post_compile_params |= {bindparam}

        ret = self.bindparam_string(
            name,
            post_compile=post_compile,
            expanding=bindparam.expanding,
            bindparam_type=bindparam.type,
            **kwargs,
        )

        if bindparam.expanding:
            ret = "(%s)" % ret

        return ret

    def render_bind_cast(self, type_, dbapi_type, sqltext):
        raise NotImplementedError()

    def render_literal_bindparam(
        self,
        bindparam,
        render_literal_value=NO_ARG,
        bind_expression_template=None,
        **kw,
    ):
        if render_literal_value is not NO_ARG:
            value = render_literal_value
        else:
            if bindparam.value is None and bindparam.callable is None:
                op = kw.get("_binary_op", None)
                if op and op not in (operators.is_, operators.is_not):
                    util.warn_limited(
                        "Bound parameter '%s' rendering literal NULL in a SQL "
                        "expression; comparisons to NULL should not use "
                        "operators outside of 'is' or 'is not'",
                        (bindparam.key,),
                    )
                return self.process(sqltypes.NULLTYPE, **kw)
            value = bindparam.effective_value

        if bindparam.expanding:
            leep = self._literal_execute_expanding_parameter_literal_binds
            to_update, replacement_expr = leep(
                bindparam,
                value,
                bind_expression_template=bind_expression_template,
            )
            return replacement_expr
        else:
            return self.render_literal_value(value, bindparam.type)

    def render_literal_value(
        self, value: Any, type_: sqltypes.TypeEngine[Any]
    ) -> str:
        """Render the value of a bind parameter as a quoted literal.

        This is used for statement sections that do not accept bind parameters
        on the target driver/database.

        This should be implemented by subclasses using the quoting services
        of the DBAPI.

        """

        if value is None and not type_.should_evaluate_none:
            # issue #10535 - handle NULL in the compiler without placing
            # this onto each type, except for "evaluate None" types
            # (e.g. JSON)
            return self.process(elements.Null._instance())

        processor = type_._cached_literal_processor(self.dialect)
        if processor:
            try:
                return processor(value)
            except Exception as e:
                raise exc.CompileError(
                    f"Could not render literal value "
                    f'"{sql_util._repr_single_value(value)}" '
                    f"with datatype "
                    f"{type_}; see parent stack trace for "
                    "more detail."
                ) from e

        else:
            raise exc.CompileError(
                f"No literal value renderer is available for literal value "
                f'"{sql_util._repr_single_value(value)}" '
                f"with datatype {type_}"
            )

    def _truncate_bindparam(self, bindparam):
        if bindparam in self.bind_names:
            return self.bind_names[bindparam]

        bind_name = bindparam.key
        if isinstance(bind_name, elements._truncated_label):
            bind_name = self._truncated_identifier("bindparam", bind_name)

        # add to bind_names for translation
        self.bind_names[bindparam] = bind_name

        return bind_name

    def _truncated_identifier(
        self, ident_class: str, name: _truncated_label
    ) -> str:
        if (ident_class, name) in self.truncated_names:
            return self.truncated_names[(ident_class, name)]

        anonname = name.apply_map(self.anon_map)

        if len(anonname) > self.label_length - 6:
            counter = self._truncated_counters.get(ident_class, 1)
            truncname = (
                anonname[0 : max(self.label_length - 6, 0)]
                + "_"
                + hex(counter)[2:]
            )
            self._truncated_counters[ident_class] = counter + 1
        else:
            truncname = anonname
        self.truncated_names[(ident_class, name)] = truncname
        return truncname

    def _anonymize(self, name: str) -> str:
        return name % self.anon_map

    def bindparam_string(
        self,
        name: str,
        post_compile: bool = False,
        expanding: bool = False,
        escaped_from: Optional[str] = None,
        bindparam_type: Optional[TypeEngine[Any]] = None,
        accumulate_bind_names: Optional[Set[str]] = None,
        visited_bindparam: Optional[List[str]] = None,
        **kw: Any,
    ) -> str:
        # TODO: accumulate_bind_names is passed by crud.py to gather
        # names on a per-value basis, visited_bindparam is passed by
        # visit_insert() to collect all parameters in the statement.
        # see if this gathering can be simplified somehow
        if accumulate_bind_names is not None:
            accumulate_bind_names.add(name)
        if visited_bindparam is not None:
            visited_bindparam.append(name)

        if not escaped_from:
            if self._bind_translate_re.search(name):
                # not quite the translate use case as we want to
                # also get a quick boolean if we even found
                # unusual characters in the name
                new_name = self._bind_translate_re.sub(
                    lambda m: self._bind_translate_chars[m.group(0)],
                    name,
                )
                escaped_from = name
                name = new_name

        if escaped_from:
            self.escaped_bind_names = self.escaped_bind_names.union(
                {escaped_from: name}
            )
        if post_compile:
            ret = "__[POSTCOMPILE_%s]" % name
            if expanding:
                # for expanding, bound parameters or literal values will be
                # rendered per item
                return ret

            # otherwise, for non-expanding "literal execute", apply
            # bind casts as determined by the datatype
            if bindparam_type is not None:
                type_impl = bindparam_type._unwrapped_dialect_impl(
                    self.dialect
                )
                if type_impl.render_literal_cast:
                    ret = self.render_bind_cast(bindparam_type, type_impl, ret)
            return ret
        elif self.state is CompilerState.COMPILING:
            ret = self.compilation_bindtemplate % {"name": name}
        else:
            ret = self.bindtemplate % {"name": name}

        if (
            bindparam_type is not None
            and self.dialect._bind_typing_render_casts
        ):
            type_impl = bindparam_type._unwrapped_dialect_impl(self.dialect)
            if type_impl.render_bind_cast:
                ret = self.render_bind_cast(bindparam_type, type_impl, ret)

        return ret

    def _dispatch_independent_ctes(self, stmt, kw):
        local_kw = kw.copy()
        local_kw.pop("cte_opts", None)
        for cte, opt in zip(
            stmt._independent_ctes, stmt._independent_ctes_opts
        ):
            cte._compiler_dispatch(self, cte_opts=opt, **local_kw)

    def visit_cte(
        self,
        cte: CTE,
        asfrom: bool = False,
        ashint: bool = False,
        fromhints: Optional[_FromHintsType] = None,
        visiting_cte: Optional[CTE] = None,
        from_linter: Optional[FromLinter] = None,
        cte_opts: selectable._CTEOpts = selectable._CTEOpts(False),
        **kwargs: Any,
    ) -> Optional[str]:
        self_ctes = self._init_cte_state()
        assert self_ctes is self.ctes

        kwargs["visiting_cte"] = cte

        cte_name = cte.name

        if isinstance(cte_name, elements._truncated_label):
            cte_name = self._truncated_identifier("alias", cte_name)

        is_new_cte = True
        embedded_in_current_named_cte = False

        _reference_cte = cte._get_reference_cte()

        nesting = cte.nesting or cte_opts.nesting

        # check for CTE already encountered
        if _reference_cte in self.level_name_by_cte:
            cte_level, _, existing_cte_opts = self.level_name_by_cte[
                _reference_cte
            ]
            assert _ == cte_name

            cte_level_name = (cte_level, cte_name)
            existing_cte = self.ctes_by_level_name[cte_level_name]

            # check if we are receiving it here with a specific
            # "nest_here" location; if so, move it to this location

            if cte_opts.nesting:
                if existing_cte_opts.nesting:
                    raise exc.CompileError(
                        "CTE is stated as 'nest_here' in "
                        "more than one location"
                    )

                old_level_name = (cte_level, cte_name)
                cte_level = len(self.stack) if nesting else 1
                cte_level_name = new_level_name = (cte_level, cte_name)

                del self.ctes_by_level_name[old_level_name]
                self.ctes_by_level_name[new_level_name] = existing_cte
                self.level_name_by_cte[_reference_cte] = new_level_name + (
                    cte_opts,
                )

        else:
            cte_level = len(self.stack) if nesting else 1
            cte_level_name = (cte_level, cte_name)

            if cte_level_name in self.ctes_by_level_name:
                existing_cte = self.ctes_by_level_name[cte_level_name]
            else:
                existing_cte = None

        if existing_cte is not None:
            embedded_in_current_named_cte = visiting_cte is existing_cte

            # we've generated a same-named CTE that we are enclosed in,
            # or this is the same CTE.  just return the name.
            if cte is existing_cte._restates or cte is existing_cte:
                is_new_cte = False
            elif existing_cte is cte._restates:
                # we've generated a same-named CTE that is
                # enclosed in us - we take precedence, so
                # discard the text for the "inner".
                del self_ctes[existing_cte]

                existing_cte_reference_cte = existing_cte._get_reference_cte()

                assert existing_cte_reference_cte is _reference_cte
                assert existing_cte_reference_cte is existing_cte

                del self.level_name_by_cte[existing_cte_reference_cte]
            else:
                if (
                    # if the two CTEs have the same hash, which we expect
                    # here means that one/both is an annotated of the other
                    (hash(cte) == hash(existing_cte))
                    # or...
                    or (
                        (
                            # if they are clones, i.e. they came from the ORM
                            # or some other visit method
                            cte._is_clone_of is not None
                            or existing_cte._is_clone_of is not None
                        )
                        # and are deep-copy identical
                        and cte.compare(existing_cte)
                    )
                ):
                    # then consider these two CTEs the same
                    is_new_cte = False
                else:
                    # otherwise these are two CTEs that either will render
                    # differently, or were indicated separately by the user,
                    # with the same name
                    raise exc.CompileError(
                        "Multiple, unrelated CTEs found with "
                        "the same name: %r" % cte_name
                    )

        if not asfrom and not is_new_cte:
            return None

        if cte._cte_alias is not None:
            pre_alias_cte = cte._cte_alias
            cte_pre_alias_name = cte._cte_alias.name
            if isinstance(cte_pre_alias_name, elements._truncated_label):
                cte_pre_alias_name = self._truncated_identifier(
                    "alias", cte_pre_alias_name
                )
        else:
            pre_alias_cte = cte
            cte_pre_alias_name = None

        if is_new_cte:
            self.ctes_by_level_name[cte_level_name] = cte
            self.level_name_by_cte[_reference_cte] = cte_level_name + (
                cte_opts,
            )

            if pre_alias_cte not in self.ctes:
                self.visit_cte(pre_alias_cte, **kwargs)

            if not cte_pre_alias_name and cte not in self_ctes:
                if cte.recursive:
                    self.ctes_recursive = True
                text = self.preparer.format_alias(cte, cte_name)
                if cte.recursive:
                    col_source = cte.element

                    # TODO: can we get at the .columns_plus_names collection
                    # that is already (or will be?) generated for the SELECT
                    # rather than calling twice?
                    recur_cols = [
                        # TODO: proxy_name is not technically safe,
                        # see test_cte->
                        # test_with_recursive_no_name_currently_buggy.  not
                        # clear what should be done with such a case
                        fallback_label_name or proxy_name
                        for (
                            _,
                            proxy_name,
                            fallback_label_name,
                            c,
                            repeated,
                        ) in (col_source._generate_columns_plus_names(True))
                        if not repeated
                    ]

                    text += "(%s)" % (
                        ", ".join(
                            self.preparer.format_label_name(
                                ident, anon_map=self.anon_map
                            )
                            for ident in recur_cols
                        )
                    )

                assert kwargs.get("subquery", False) is False

                if not self.stack:
                    # toplevel, this is a stringify of the
                    # cte directly.  just compile the inner
                    # the way alias() does.
                    return cte.element._compiler_dispatch(
                        self, asfrom=asfrom, **kwargs
                    )
                else:
                    prefixes = self._generate_prefixes(
                        cte, cte._prefixes, **kwargs
                    )
                    inner = cte.element._compiler_dispatch(
                        self, asfrom=True, **kwargs
                    )

                    text += " AS %s\n(%s)" % (prefixes, inner)

                if cte._suffixes:
                    text += " " + self._generate_prefixes(
                        cte, cte._suffixes, **kwargs
                    )

                self_ctes[cte] = text

        if asfrom:
            if from_linter:
                from_linter.froms[cte._de_clone()] = cte_name

            if not is_new_cte and embedded_in_current_named_cte:
                return self.preparer.format_alias(cte, cte_name)

            if cte_pre_alias_name:
                text = self.preparer.format_alias(cte, cte_pre_alias_name)
                if self.preparer._requires_quotes(cte_name):
                    cte_name = self.preparer.quote(cte_name)
                text += self.get_render_as_alias_suffix(cte_name)
                return text
            else:
                return self.preparer.format_alias(cte, cte_name)

        return None

    def visit_table_valued_alias(self, element, **kw):
        if element.joins_implicitly:
            kw["from_linter"] = None
        if element._is_lateral:
            return self.visit_lateral(element, **kw)
        else:
            return self.visit_alias(element, **kw)

    def visit_table_valued_column(self, element, **kw):
        return self.visit_column(element, **kw)

    def visit_alias(
        self,
        alias,
        asfrom=False,
        ashint=False,
        iscrud=False,
        fromhints=None,
        subquery=False,
        lateral=False,
        enclosing_alias=None,
        from_linter=None,
        **kwargs,
    ):
        if lateral:
            if "enclosing_lateral" not in kwargs:
                # if lateral is set and enclosing_lateral is not
                # present, we assume we are being called directly
                # from visit_lateral() and we need to set enclosing_lateral.
                assert alias._is_lateral
                kwargs["enclosing_lateral"] = alias

            # for lateral objects, we track a second from_linter that is...
            # lateral!  to the level above us.
            if (
                from_linter
                and "lateral_from_linter" not in kwargs
                and "enclosing_lateral" in kwargs
            ):
                kwargs["lateral_from_linter"] = from_linter

        if enclosing_alias is not None and enclosing_alias.element is alias:
            inner = alias.element._compiler_dispatch(
                self,
                asfrom=asfrom,
                ashint=ashint,
                iscrud=iscrud,
                fromhints=fromhints,
                lateral=lateral,
                enclosing_alias=alias,
                **kwargs,
            )
            if subquery and (asfrom or lateral):
                inner = "(%s)" % (inner,)
            return inner
        else:
            kwargs["enclosing_alias"] = alias

        if asfrom or ashint:
            if isinstance(alias.name, elements._truncated_label):
                alias_name = self._truncated_identifier("alias", alias.name)
            else:
                alias_name = alias.name

        if ashint:
            return self.preparer.format_alias(alias, alias_name)
        elif asfrom:
            if from_linter:
                from_linter.froms[alias._de_clone()] = alias_name

            inner = alias.element._compiler_dispatch(
                self, asfrom=True, lateral=lateral, **kwargs
            )
            if subquery:
                inner = "(%s)" % (inner,)

            ret = inner + self.get_render_as_alias_suffix(
                self.preparer.format_alias(alias, alias_name)
            )

            if alias._supports_derived_columns and alias._render_derived:
                ret += "(%s)" % (
                    ", ".join(
                        "%s%s"
                        % (
                            self.preparer.quote(col.name),
                            (
                                " %s"
                                % self.dialect.type_compiler_instance.process(
                                    col.type, **kwargs
                                )
                                if alias._render_derived_w_types
                                else ""
                            ),
                        )
                        for col in alias.c
                    )
                )

            if fromhints and alias in fromhints:
                ret = self.format_from_hint_text(
                    ret, alias, fromhints[alias], iscrud
                )

            return ret
        else:
            # note we cancel the "subquery" flag here as well
            return alias.element._compiler_dispatch(
                self, lateral=lateral, **kwargs
            )

    def visit_subquery(self, subquery, **kw):
        kw["subquery"] = True
        return self.visit_alias(subquery, **kw)

    def visit_lateral(self, lateral_, **kw):
        kw["lateral"] = True
        return "LATERAL %s" % self.visit_alias(lateral_, **kw)

    def visit_tablesample(self, tablesample, asfrom=False, **kw):
        text = "%s TABLESAMPLE %s" % (
            self.visit_alias(tablesample, asfrom=True, **kw),
            tablesample._get_method()._compiler_dispatch(self, **kw),
        )

        if tablesample.seed is not None:
            text += " REPEATABLE (%s)" % (
                tablesample.seed._compiler_dispatch(self, **kw)
            )

        return text

    def _render_values(self, element, **kw):
        kw.setdefault("literal_binds", element.literal_binds)
        tuples = ", ".join(
            self.process(
                elements.Tuple(
                    types=element._column_types, *elem
                ).self_group(),
                **kw,
            )
            for chunk in element._data
            for elem in chunk
        )
        return f"VALUES {tuples}"

    def visit_values(self, element, asfrom=False, from_linter=None, **kw):
        v = self._render_values(element, **kw)

        if element._unnamed:
            name = None
        elif isinstance(element.name, elements._truncated_label):
            name = self._truncated_identifier("values", element.name)
        else:
            name = element.name

        if element._is_lateral:
            lateral = "LATERAL "
        else:
            lateral = ""

        if asfrom:
            if from_linter:
                from_linter.froms[element._de_clone()] = (
                    name if name is not None else "(unnamed VALUES element)"
                )

            if name:
                kw["include_table"] = False
                v = "%s(%s)%s (%s)" % (
                    lateral,
                    v,
                    self.get_render_as_alias_suffix(self.preparer.quote(name)),
                    (
                        ", ".join(
                            c._compiler_dispatch(self, **kw)
                            for c in element.columns
                        )
                    ),
                )
            else:
                v = "%s(%s)" % (lateral, v)
        return v

    def visit_scalar_values(self, element, **kw):
        return f"({self._render_values(element, **kw)})"

    def get_render_as_alias_suffix(self, alias_name_text):
        return " AS " + alias_name_text

    def _add_to_result_map(
        self,
        keyname: str,
        name: str,
        objects: Tuple[Any, ...],
        type_: TypeEngine[Any],
    ) -> None:

        # note objects must be non-empty for cursor.py to handle the
        # collection properly
        assert objects

        if keyname is None or keyname == "*":
            self._ordered_columns = False
            self._ad_hoc_textual = True
        if type_._is_tuple_type:
            raise exc.CompileError(
                "Most backends don't support SELECTing "
                "from a tuple() object.  If this is an ORM query, "
                "consider using the Bundle object."
            )
        self._result_columns.append(
            ResultColumnsEntry(keyname, name, objects, type_)
        )

    def _label_returning_column(
        self, stmt, column, populate_result_map, column_clause_args=None, **kw
    ):
        """Render a column with necessary labels inside of a RETURNING clause.

        This method is provided for individual dialects in place of calling
        the _label_select_column method directly, so that the two use cases
        of RETURNING vs. SELECT can be disambiguated going forward.

        .. versionadded:: 1.4.21

        """
        return self._label_select_column(
            None,
            column,
            populate_result_map,
            False,
            {} if column_clause_args is None else column_clause_args,
            **kw,
        )

    def _label_select_column(
        self,
        select,
        column,
        populate_result_map,
        asfrom,
        column_clause_args,
        name=None,
        proxy_name=None,
        fallback_label_name=None,
        within_columns_clause=True,
        column_is_repeated=False,
        need_column_expressions=False,
        include_table=True,
    ):
        """produce labeled columns present in a select()."""
        impl = column.type.dialect_impl(self.dialect)

        if impl._has_column_expression and (
            need_column_expressions or populate_result_map
        ):
            col_expr = impl.column_expression(column)
        else:
            col_expr = column

        if populate_result_map:
            # pass an "add_to_result_map" callable into the compilation
            # of embedded columns.  this collects information about the
            # column as it will be fetched in the result and is coordinated
            # with cursor.description when the query is executed.
            add_to_result_map = self._add_to_result_map

            # if the SELECT statement told us this column is a repeat,
            # wrap the callable with one that prevents the addition of the
            # targets
            if column_is_repeated:
                _add_to_result_map = add_to_result_map

                def add_to_result_map(keyname, name, objects, type_):
                    _add_to_result_map(keyname, name, (keyname,), type_)

            # if we redefined col_expr for type expressions, wrap the
            # callable with one that adds the original column to the targets
            elif col_expr is not column:
                _add_to_result_map = add_to_result_map

                def add_to_result_map(keyname, name, objects, type_):
                    _add_to_result_map(
                        keyname, name, (column,) + objects, type_
                    )

        else:
            add_to_result_map = None

        # this method is used by some of the dialects for RETURNING,
        # which has different inputs.  _label_returning_column was added
        # as the better target for this now however for 1.4 we will keep
        # _label_select_column directly compatible with this use case.
        # these assertions right now set up the current expected inputs
        assert within_columns_clause, (
            "_label_select_column is only relevant within "
            "the columns clause of a SELECT or RETURNING"
        )
        if isinstance(column, elements.Label):
            if col_expr is not column:
                result_expr = _CompileLabel(
                    col_expr, column.name, alt_names=(column.element,)
                )
            else:
                result_expr = col_expr

        elif name:
            # here, _columns_plus_names has determined there's an explicit
            # label name we need to use.  this is the default for
            # tablenames_plus_columnnames as well as when columns are being
            # deduplicated on name

            assert (
                proxy_name is not None
            ), "proxy_name is required if 'name' is passed"

            result_expr = _CompileLabel(
                col_expr,
                name,
                alt_names=(
                    proxy_name,
                    # this is a hack to allow legacy result column lookups
                    # to work as they did before; this goes away in 2.0.
                    # TODO: this only seems to be tested indirectly
                    # via test/orm/test_deprecations.py.   should be a
                    # resultset test for this
                    column._tq_label,
                ),
            )
        else:
            # determine here whether this column should be rendered in
            # a labelled context or not, as we were given no required label
            # name from the caller. Here we apply heuristics based on the kind
            # of SQL expression involved.

            if col_expr is not column:
                # type-specific expression wrapping the given column,
                # so we render a label
                render_with_label = True
            elif isinstance(column, elements.ColumnClause):
                # table-bound column, we render its name as a label if we are
                # inside of a subquery only
                render_with_label = (
                    asfrom
                    and not column.is_literal
                    and column.table is not None
                )
            elif isinstance(column, elements.TextClause):
                render_with_label = False
            elif isinstance(column, elements.UnaryExpression):
                render_with_label = column.wraps_column_expression or asfrom
            elif (
                # general class of expressions that don't have a SQL-column
                # addressible name.  includes scalar selects, bind parameters,
                # SQL functions, others
                not isinstance(column, elements.NamedColumn)
                # deeper check that indicates there's no natural "name" to
                # this element, which accommodates for custom SQL constructs
                # that might have a ".name" attribute (but aren't SQL
                # functions) but are not implementing this more recently added
                # base class.  in theory the "NamedColumn" check should be
                # enough, however here we seek to maintain legacy behaviors
                # as well.
                and column._non_anon_label is None
            ):
                render_with_label = True
            else:
                render_with_label = False

            if render_with_label:
                if not fallback_label_name:
                    # used by the RETURNING case right now.  we generate it
                    # here as 3rd party dialects may be referring to
                    # _label_select_column method directly instead of the
                    # just-added _label_returning_column method
                    assert not column_is_repeated
                    fallback_label_name = column._anon_name_label

                fallback_label_name = (
                    elements._truncated_label(fallback_label_name)
                    if not isinstance(
                        fallback_label_name, elements._truncated_label
                    )
                    else fallback_label_name
                )

                result_expr = _CompileLabel(
                    col_expr, fallback_label_name, alt_names=(proxy_name,)
                )
            else:
                result_expr = col_expr

        column_clause_args.update(
            within_columns_clause=within_columns_clause,
            add_to_result_map=add_to_result_map,
            include_table=include_table,
        )
        return result_expr._compiler_dispatch(self, **column_clause_args)

    def format_from_hint_text(self, sqltext, table, hint, iscrud):
        hinttext = self.get_from_hint_text(table, hint)
        if hinttext:
            sqltext += " " + hinttext
        return sqltext

    def get_select_hint_text(self, byfroms):
        return None

    def get_from_hint_text(
        self, table: FromClause, text: Optional[str]
    ) -> Optional[str]:
        return None

    def get_crud_hint_text(self, table, text):
        return None

    def get_statement_hint_text(self, hint_texts):
        return " ".join(hint_texts)

    _default_stack_entry: _CompilerStackEntry

    if not typing.TYPE_CHECKING:
        _default_stack_entry = util.immutabledict(
            [("correlate_froms", frozenset()), ("asfrom_froms", frozenset())]
        )

    def _display_froms_for_select(
        self, select_stmt, asfrom, lateral=False, **kw
    ):
        # utility method to help external dialects
        # get the correct from list for a select.
        # specifically the oracle dialect needs this feature
        # right now.
        toplevel = not self.stack
        entry = self._default_stack_entry if toplevel else self.stack[-1]

        compile_state = select_stmt._compile_state_factory(select_stmt, self)

        correlate_froms = entry["correlate_froms"]
        asfrom_froms = entry["asfrom_froms"]

        if asfrom and not lateral:
            froms = compile_state._get_display_froms(
                explicit_correlate_froms=correlate_froms.difference(
                    asfrom_froms
                ),
                implicit_correlate_froms=(),
            )
        else:
            froms = compile_state._get_display_froms(
                explicit_correlate_froms=correlate_froms,
                implicit_correlate_froms=asfrom_froms,
            )
        return froms

    translate_select_structure: Any = None
    """if not ``None``, should be a callable which accepts ``(select_stmt,
    **kw)`` and returns a select object.   this is used for structural changes
    mostly to accommodate for LIMIT/OFFSET schemes

    """

    def visit_select(
        self,
        select_stmt,
        asfrom=False,
        insert_into=False,
        fromhints=None,
        compound_index=None,
        select_wraps_for=None,
        lateral=False,
        from_linter=None,
        **kwargs,
    ):
        assert select_wraps_for is None, (
            "SQLAlchemy 1.4 requires use of "
            "the translate_select_structure hook for structural "
            "translations of SELECT objects"
        )

        # initial setup of SELECT.  the compile_state_factory may now
        # be creating a totally different SELECT from the one that was
        # passed in.  for ORM use this will convert from an ORM-state
        # SELECT to a regular "Core" SELECT.  other composed operations
        # such as computation of joins will be performed.

        kwargs["within_columns_clause"] = False

        compile_state = select_stmt._compile_state_factory(
            select_stmt, self, **kwargs
        )
        kwargs["ambiguous_table_name_map"] = (
            compile_state._ambiguous_table_name_map
        )

        select_stmt = compile_state.statement

        toplevel = not self.stack

        if toplevel and not self.compile_state:
            self.compile_state = compile_state

        is_embedded_select = compound_index is not None or insert_into

        # translate step for Oracle, SQL Server which often need to
        # restructure the SELECT to allow for LIMIT/OFFSET and possibly
        # other conditions
        if self.translate_select_structure:
            new_select_stmt = self.translate_select_structure(
                select_stmt, asfrom=asfrom, **kwargs
            )

            # if SELECT was restructured, maintain a link to the originals
            # and assemble a new compile state
            if new_select_stmt is not select_stmt:
                compile_state_wraps_for = compile_state
                select_wraps_for = select_stmt
                select_stmt = new_select_stmt

                compile_state = select_stmt._compile_state_factory(
                    select_stmt, self, **kwargs
                )
                select_stmt = compile_state.statement

        entry = self._default_stack_entry if toplevel else self.stack[-1]

        populate_result_map = need_column_expressions = (
            toplevel
            or entry.get("need_result_map_for_compound", False)
            or entry.get("need_result_map_for_nested", False)
        )

        # indicates there is a CompoundSelect in play and we are not the
        # first select
        if compound_index:
            populate_result_map = False

        # this was first proposed as part of #3372; however, it is not
        # reached in current tests and could possibly be an assertion
        # instead.
        if not populate_result_map and "add_to_result_map" in kwargs:
            del kwargs["add_to_result_map"]

        froms = self._setup_select_stack(
            select_stmt, compile_state, entry, asfrom, lateral, compound_index
        )

        column_clause_args = kwargs.copy()
        column_clause_args.update(
            {"within_label_clause": False, "within_columns_clause": False}
        )

        text = "SELECT "  # we're off to a good start !

        if select_stmt._hints:
            hint_text, byfrom = self._setup_select_hints(select_stmt)
            if hint_text:
                text += hint_text + " "
        else:
            byfrom = None

        if select_stmt._independent_ctes:
            self._dispatch_independent_ctes(select_stmt, kwargs)

        if select_stmt._prefixes:
            text += self._generate_prefixes(
                select_stmt, select_stmt._prefixes, **kwargs
            )

        text += self.get_select_precolumns(select_stmt, **kwargs)
        # the actual list of columns to print in the SELECT column list.
        inner_columns = [
            c
            for c in [
                self._label_select_column(
                    select_stmt,
                    column,
                    populate_result_map,
                    asfrom,
                    column_clause_args,
                    name=name,
                    proxy_name=proxy_name,
                    fallback_label_name=fallback_label_name,
                    column_is_repeated=repeated,
                    need_column_expressions=need_column_expressions,
                )
                for (
                    name,
                    proxy_name,
                    fallback_label_name,
                    column,
                    repeated,
                ) in compile_state.columns_plus_names
            ]
            if c is not None
        ]

        if populate_result_map and select_wraps_for is not None:
            # if this select was generated from translate_select,
            # rewrite the targeted columns in the result map

            translate = dict(
                zip(
                    [
                        name
                        for (
                            key,
                            proxy_name,
                            fallback_label_name,
                            name,
                            repeated,
                        ) in compile_state.columns_plus_names
                    ],
                    [
                        name
                        for (
                            key,
                            proxy_name,
                            fallback_label_name,
                            name,
                            repeated,
                        ) in compile_state_wraps_for.columns_plus_names
                    ],
                )
            )

            self._result_columns = [
                ResultColumnsEntry(
                    key, name, tuple(translate.get(o, o) for o in obj), type_
                )
                for key, name, obj, type_ in self._result_columns
            ]

        text = self._compose_select_body(
            text,
            select_stmt,
            compile_state,
            inner_columns,
            froms,
            byfrom,
            toplevel,
            kwargs,
        )

        if select_stmt._statement_hints:
            per_dialect = [
                ht
                for (dialect_name, ht) in select_stmt._statement_hints
                if dialect_name in ("*", self.dialect.name)
            ]
            if per_dialect:
                text += " " + self.get_statement_hint_text(per_dialect)

        # In compound query, CTEs are shared at the compound level
        if self.ctes and (not is_embedded_select or toplevel):
            nesting_level = len(self.stack) if not toplevel else None
            text = self._render_cte_clause(nesting_level=nesting_level) + text

        if select_stmt._suffixes:
            text += " " + self._generate_prefixes(
                select_stmt, select_stmt._suffixes, **kwargs
            )

        self.stack.pop(-1)

        return text

    def _setup_select_hints(
        self, select: Select[Any]
    ) -> Tuple[str, _FromHintsType]:
        byfrom = {
            from_: hinttext
            % {"name": from_._compiler_dispatch(self, ashint=True)}
            for (from_, dialect), hinttext in select._hints.items()
            if dialect in ("*", self.dialect.name)
        }
        hint_text = self.get_select_hint_text(byfrom)
        return hint_text, byfrom

    def _setup_select_stack(
        self, select, compile_state, entry, asfrom, lateral, compound_index
    ):
        correlate_froms = entry["correlate_froms"]
        asfrom_froms = entry["asfrom_froms"]

        if compound_index == 0:
            entry["select_0"] = select
        elif compound_index:
            select_0 = entry["select_0"]
            numcols = len(select_0._all_selected_columns)

            if len(compile_state.columns_plus_names) != numcols:
                raise exc.CompileError(
                    "All selectables passed to "
                    "CompoundSelect must have identical numbers of "
                    "columns; select #%d has %d columns, select "
                    "#%d has %d"
                    % (
                        1,
                        numcols,
                        compound_index + 1,
                        len(select._all_selected_columns),
                    )
                )

        if asfrom and not lateral:
            froms = compile_state._get_display_froms(
                explicit_correlate_froms=correlate_froms.difference(
                    asfrom_froms
                ),
                implicit_correlate_froms=(),
            )
        else:
            froms = compile_state._get_display_froms(
                explicit_correlate_froms=correlate_froms,
                implicit_correlate_froms=asfrom_froms,
            )

        new_correlate_froms = set(_from_objects(*froms))
        all_correlate_froms = new_correlate_froms.union(correlate_froms)

        new_entry: _CompilerStackEntry = {
            "asfrom_froms": new_correlate_froms,
            "correlate_froms": all_correlate_froms,
            "selectable": select,
            "compile_state": compile_state,
        }
        self.stack.append(new_entry)

        return froms

    def _compose_select_body(
        self,
        text,
        select,
        compile_state,
        inner_columns,
        froms,
        byfrom,
        toplevel,
        kwargs,
    ):
        text += ", ".join(inner_columns)

        if self.linting & COLLECT_CARTESIAN_PRODUCTS:
            from_linter = FromLinter({}, set())
            warn_linting = self.linting & WARN_LINTING
            if toplevel:
                self.from_linter = from_linter
        else:
            from_linter = None
            warn_linting = False

        # adjust the whitespace for no inner columns, part of #9440,
        # so that a no-col SELECT comes out as "SELECT WHERE..." or
        # "SELECT FROM ...".
        # while it would be better to have built the SELECT starting string
        # without trailing whitespace first, then add whitespace only if inner
        # cols were present, this breaks compatibility with various custom
        # compilation schemes that are currently being tested.
        if not inner_columns:
            text = text.rstrip()

        if froms:
            text += " \nFROM "

            if select._hints:
                text += ", ".join(
                    [
                        f._compiler_dispatch(
                            self,
                            asfrom=True,
                            fromhints=byfrom,
                            from_linter=from_linter,
                            **kwargs,
                        )
                        for f in froms
                    ]
                )
            else:
                text += ", ".join(
                    [
                        f._compiler_dispatch(
                            self,
                            asfrom=True,
                            from_linter=from_linter,
                            **kwargs,
                        )
                        for f in froms
                    ]
                )
        else:
            text += self.default_from()

        if select._where_criteria:
            t = self._generate_delimited_and_list(
                select._where_criteria, from_linter=from_linter, **kwargs
            )
            if t:
                text += " \nWHERE " + t

        if warn_linting:
            assert from_linter is not None
            from_linter.warn()

        if select._group_by_clauses:
            text += self.group_by_clause(select, **kwargs)

        if select._having_criteria:
            t = self._generate_delimited_and_list(
                select._having_criteria, **kwargs
            )
            if t:
                text += " \nHAVING " + t

        if select._order_by_clauses:
            text += self.order_by_clause(select, **kwargs)

        if select._has_row_limiting_clause:
            text += self._row_limit_clause(select, **kwargs)

        if select._for_update_arg is not None:
            text += self.for_update_clause(select, **kwargs)

        return text

    def _generate_prefixes(self, stmt, prefixes, **kw):
        clause = " ".join(
            prefix._compiler_dispatch(self, **kw)
            for prefix, dialect_name in prefixes
            if dialect_name in (None, "*") or dialect_name == self.dialect.name
        )
        if clause:
            clause += " "
        return clause

    def _render_cte_clause(
        self,
        nesting_level=None,
        include_following_stack=False,
    ):
        """
        include_following_stack
            Also render the nesting CTEs on the next stack. Useful for
            SQL structures like UNION or INSERT that can wrap SELECT
            statements containing nesting CTEs.
        """
        if not self.ctes:
            return ""

        ctes: MutableMapping[CTE, str]

        if nesting_level and nesting_level > 1:
            ctes = util.OrderedDict()
            for cte in list(self.ctes.keys()):
                cte_level, cte_name, cte_opts = self.level_name_by_cte[
                    cte._get_reference_cte()
                ]
                nesting = cte.nesting or cte_opts.nesting
                is_rendered_level = cte_level == nesting_level or (
                    include_following_stack and cte_level == nesting_level + 1
                )
                if not (nesting and is_rendered_level):
                    continue

                ctes[cte] = self.ctes[cte]

        else:
            ctes = self.ctes

        if not ctes:
            return ""
        ctes_recursive = any([cte.recursive for cte in ctes])

        cte_text = self.get_cte_preamble(ctes_recursive) + " "
        cte_text += ", \n".join([txt for txt in ctes.values()])
        cte_text += "\n "

        if nesting_level and nesting_level > 1:
            for cte in list(ctes.keys()):
                cte_level, cte_name, cte_opts = self.level_name_by_cte[
                    cte._get_reference_cte()
                ]
                del self.ctes[cte]
                del self.ctes_by_level_name[(cte_level, cte_name)]
                del self.level_name_by_cte[cte._get_reference_cte()]

        return cte_text

    def get_cte_preamble(self, recursive):
        if recursive:
            return "WITH RECURSIVE"
        else:
            return "WITH"

    def get_select_precolumns(self, select: Select[Any], **kw: Any) -> str:
        """Called when building a ``SELECT`` statement, position is just
        before column list.

        """
        if select._distinct_on:
            util.warn_deprecated(
                "DISTINCT ON is currently supported only by the PostgreSQL "
                "dialect.  Use of DISTINCT ON for other backends is currently "
                "silently ignored, however this usage is deprecated, and will "
                "raise CompileError in a future release for all backends "
                "that do not support this syntax.",
                version="1.4",
            )
        return "DISTINCT " if select._distinct else ""

    def group_by_clause(self, select, **kw):
        """allow dialects to customize how GROUP BY is rendered."""

        group_by = self._generate_delimited_list(
            select._group_by_clauses, OPERATORS[operators.comma_op], **kw
        )
        if group_by:
            return " GROUP BY " + group_by
        else:
            return ""

    def order_by_clause(self, select, **kw):
        """allow dialects to customize how ORDER BY is rendered."""

        order_by = self._generate_delimited_list(
            select._order_by_clauses, OPERATORS[operators.comma_op], **kw
        )

        if order_by:
            return " ORDER BY " + order_by
        else:
            return ""

    def for_update_clause(self, select, **kw):
        return " FOR UPDATE"

    def returning_clause(
        self,
        stmt: UpdateBase,
        returning_cols: Sequence[_ColumnsClauseElement],
        *,
        populate_result_map: bool,
        **kw: Any,
    ) -> str:
        columns = [
            self._label_returning_column(
                stmt,
                column,
                populate_result_map,
                fallback_label_name=fallback_label_name,
                column_is_repeated=repeated,
                name=name,
                proxy_name=proxy_name,
                **kw,
            )
            for (
                name,
                proxy_name,
                fallback_label_name,
                column,
                repeated,
            ) in stmt._generate_columns_plus_names(
                True, cols=base._select_iterables(returning_cols)
            )
        ]

        return "RETURNING " + ", ".join(columns)

    def limit_clause(self, select, **kw):
        text = ""
        if select._limit_clause is not None:
            text += "\n LIMIT " + self.process(select._limit_clause, **kw)
        if select._offset_clause is not None:
            if select._limit_clause is None:
                text += "\n LIMIT -1"
            text += " OFFSET " + self.process(select._offset_clause, **kw)
        return text

    def fetch_clause(
        self,
        select,
        fetch_clause=None,
        require_offset=False,
        use_literal_execute_for_simple_int=False,
        **kw,
    ):
        if fetch_clause is None:
            fetch_clause = select._fetch_clause
            fetch_clause_options = select._fetch_clause_options
        else:
            fetch_clause_options = {"percent": False, "with_ties": False}

        text = ""

        if select._offset_clause is not None:
            offset_clause = select._offset_clause
            if (
                use_literal_execute_for_simple_int
                and select._simple_int_clause(offset_clause)
            ):
                offset_clause = offset_clause.render_literal_execute()
            offset_str = self.process(offset_clause, **kw)
            text += "\n OFFSET %s ROWS" % offset_str
        elif require_offset:
            text += "\n OFFSET 0 ROWS"

        if fetch_clause is not None:
            if (
                use_literal_execute_for_simple_int
                and select._simple_int_clause(fetch_clause)
            ):
                fetch_clause = fetch_clause.render_literal_execute()
            text += "\n FETCH FIRST %s%s ROWS %s" % (
                self.process(fetch_clause, **kw),
                " PERCENT" if fetch_clause_options["percent"] else "",
                "WITH TIES" if fetch_clause_options["with_ties"] else "ONLY",
            )
        return text

    def visit_table(
        self,
        table,
        asfrom=False,
        iscrud=False,
        ashint=False,
        fromhints=None,
        use_schema=True,
        from_linter=None,
        ambiguous_table_name_map=None,
        enclosing_alias=None,
        **kwargs,
    ):
        if from_linter:
            from_linter.froms[table] = table.fullname

        if asfrom or ashint:
            effective_schema = self.preparer.schema_for_object(table)

            if use_schema and effective_schema:
                ret = (
                    self.preparer.quote_schema(effective_schema)
                    + "."
                    + self.preparer.quote(table.name)
                )
            else:
                ret = self.preparer.quote(table.name)

                if (
                    (
                        enclosing_alias is None
                        or enclosing_alias.element is not table
                    )
                    and not effective_schema
                    and ambiguous_table_name_map
                    and table.name in ambiguous_table_name_map
                ):
                    anon_name = self._truncated_identifier(
                        "alias", ambiguous_table_name_map[table.name]
                    )

                    ret = ret + self.get_render_as_alias_suffix(
                        self.preparer.format_alias(None, anon_name)
                    )

            if fromhints and table in fromhints:
                ret = self.format_from_hint_text(
                    ret, table, fromhints[table], iscrud
                )
            return ret
        else:
            return ""

    def visit_join(self, join, asfrom=False, from_linter=None, **kwargs):
        if from_linter:
            from_linter.edges.update(
                itertools.product(
                    _de_clone(join.left._from_objects),
                    _de_clone(join.right._from_objects),
                )
            )

        if join.full:
            join_type = " FULL OUTER JOIN "
        elif join.isouter:
            join_type = " LEFT OUTER JOIN "
        else:
            join_type = " JOIN "
        return (
            join.left._compiler_dispatch(
                self, asfrom=True, from_linter=from_linter, **kwargs
            )
            + join_type
            + join.right._compiler_dispatch(
                self, asfrom=True, from_linter=from_linter, **kwargs
            )
            + " ON "
            # TODO: likely need asfrom=True here?
            + join.onclause._compiler_dispatch(
                self, from_linter=from_linter, **kwargs
            )
        )

    def _setup_crud_hints(self, stmt, table_text):
        dialect_hints = {
            table: hint_text
            for (table, dialect), hint_text in stmt._hints.items()
            if dialect in ("*", self.dialect.name)
        }
        if stmt.table in dialect_hints:
            table_text = self.format_from_hint_text(
                table_text, stmt.table, dialect_hints[stmt.table], True
            )
        return dialect_hints, table_text

    # within the realm of "insertmanyvalues sentinel columns",
    # these lookups match different kinds of Column() configurations
    # to specific backend capabilities.  they are broken into two
    # lookups, one for autoincrement columns and the other for non
    # autoincrement columns
    _sentinel_col_non_autoinc_lookup = util.immutabledict(
        {
            _SentinelDefaultCharacterization.CLIENTSIDE: (
                InsertmanyvaluesSentinelOpts._SUPPORTED_OR_NOT
            ),
            _SentinelDefaultCharacterization.SENTINEL_DEFAULT: (
                InsertmanyvaluesSentinelOpts._SUPPORTED_OR_NOT
            ),
            _SentinelDefaultCharacterization.NONE: (
                InsertmanyvaluesSentinelOpts._SUPPORTED_OR_NOT
            ),
            _SentinelDefaultCharacterization.IDENTITY: (
                InsertmanyvaluesSentinelOpts.IDENTITY
            ),
            _SentinelDefaultCharacterization.SEQUENCE: (
                InsertmanyvaluesSentinelOpts.SEQUENCE
            ),
        }
    )
    _sentinel_col_autoinc_lookup = _sentinel_col_non_autoinc_lookup.union(
        {
            _SentinelDefaultCharacterization.NONE: (
                InsertmanyvaluesSentinelOpts.AUTOINCREMENT
            ),
        }
    )

    def _get_sentinel_column_for_table(
        self, table: Table
    ) -> Optional[Sequence[Column[Any]]]:
        """given a :class:`.Table`, return a usable sentinel column or
        columns for this dialect if any.

        Return None if no sentinel columns could be identified, or raise an
        error if a column was marked as a sentinel explicitly but isn't
        compatible with this dialect.

        """

        sentinel_opts = self.dialect.insertmanyvalues_implicit_sentinel
        sentinel_characteristics = table._sentinel_column_characteristics

        sent_cols = sentinel_characteristics.columns

        if sent_cols is None:
            return None

        if sentinel_characteristics.is_autoinc:
            bitmask = self._sentinel_col_autoinc_lookup.get(
                sentinel_characteristics.default_characterization, 0
            )
        else:
            bitmask = self._sentinel_col_non_autoinc_lookup.get(
                sentinel_characteristics.default_characterization, 0
            )

        if sentinel_opts & bitmask:
            return sent_cols

        if sentinel_characteristics.is_explicit:
            # a column was explicitly marked as insert_sentinel=True,
            # however it is not compatible with this dialect.   they should
            # not indicate this column as a sentinel if they need to include
            # this dialect.

            # TODO: do we want non-primary key explicit sentinel cols
            # that can gracefully degrade for some backends?
            # insert_sentinel="degrade" perhaps.  not for the initial release.
            # I am hoping people are generally not dealing with this sentinel
            # business at all.

            # if is_explicit is True, there will be only one sentinel column.

            raise exc.InvalidRequestError(
                f"Column {sent_cols[0]} can't be explicitly "
                "marked as a sentinel column when using the "
                f"{self.dialect.name} dialect, as the "
                "particular type of default generation on this column is "
                "not currently compatible with this dialect's specific "
                f"INSERT..RETURNING syntax which can receive the "
                "server-generated value in "
                "a deterministic way.  To remove this error, remove "
                "insert_sentinel=True from primary key autoincrement "
                "columns; these columns are automatically used as "
                "sentinels for supported dialects in any case."
            )

        return None

    def _deliver_insertmanyvalues_batches(
        self,
        statement: str,
        parameters: _DBAPIMultiExecuteParams,
        compiled_parameters: List[_MutableCoreSingleExecuteParams],
        generic_setinputsizes: Optional[_GenericSetInputSizesType],
        batch_size: int,
        sort_by_parameter_order: bool,
        schema_translate_map: Optional[SchemaTranslateMapType],
    ) -> Iterator[_InsertManyValuesBatch]:
        imv = self._insertmanyvalues
        assert imv is not None

        if not imv.sentinel_param_keys:
            _sentinel_from_params = None
        else:
            _sentinel_from_params = operator.itemgetter(
                *imv.sentinel_param_keys
            )

        lenparams = len(parameters)
        if imv.is_default_expr and not self.dialect.supports_default_metavalue:
            # backend doesn't support
            # INSERT INTO table (pk_col) VALUES (DEFAULT), (DEFAULT), ...
            # at the moment this is basically SQL Server due to
            # not being able to use DEFAULT for identity column
            # just yield out that many single statements!  still
            # faster than a whole connection.execute() call ;)
            #
            # note we still are taking advantage of the fact that we know
            # we are using RETURNING.   The generalized approach of fetching
            # cursor.lastrowid etc. still goes through the more heavyweight
            # "ExecutionContext per statement" system as it isn't usable
            # as a generic "RETURNING" approach
            use_row_at_a_time = True
            downgraded = False
        elif not self.dialect.supports_multivalues_insert or (
            sort_by_parameter_order
            and self._result_columns
            and (imv.sentinel_columns is None or imv.includes_upsert_behaviors)
        ):
            # deterministic order was requested and the compiler could
            # not organize sentinel columns for this dialect/statement.
            # use row at a time
            use_row_at_a_time = True
            downgraded = True
        else:
            use_row_at_a_time = False
            downgraded = False

        if use_row_at_a_time:
            for batchnum, (param, compiled_param) in enumerate(
                cast(
                    "Sequence[Tuple[_DBAPISingleExecuteParams, _MutableCoreSingleExecuteParams]]",  # noqa: E501
                    zip(parameters, compiled_parameters),
                ),
                1,
            ):
                yield _InsertManyValuesBatch(
                    statement,
                    param,
                    generic_setinputsizes,
                    [param],
                    (
                        [_sentinel_from_params(compiled_param)]
                        if _sentinel_from_params
                        else []
                    ),
                    1,
                    batchnum,
                    lenparams,
                    sort_by_parameter_order,
                    downgraded,
                )
            return

        if schema_translate_map:
            rst = functools.partial(
                self.preparer._render_schema_translates,
                schema_translate_map=schema_translate_map,
            )
        else:
            rst = None

        imv_single_values_expr = imv.single_values_expr
        if rst:
            imv_single_values_expr = rst(imv_single_values_expr)

        executemany_values = f"({imv_single_values_expr})"
        statement = statement.replace(executemany_values, "__EXECMANY_TOKEN__")

        # Use optional insertmanyvalues_max_parameters
        # to further shrink the batch size so that there are no more than
        # insertmanyvalues_max_parameters params.
        # Currently used by SQL Server, which limits statements to 2100 bound
        # parameters (actually 2099).
        max_params = self.dialect.insertmanyvalues_max_parameters
        if max_params:
            total_num_of_params = len(self.bind_names)
            num_params_per_batch = len(imv.insert_crud_params)
            num_params_outside_of_batch = (
                total_num_of_params - num_params_per_batch
            )
            batch_size = min(
                batch_size,
                (
                    (max_params - num_params_outside_of_batch)
                    // num_params_per_batch
                ),
            )

        batches = cast("List[Sequence[Any]]", list(parameters))
        compiled_batches = cast(
            "List[Sequence[Any]]", list(compiled_parameters)
        )

        processed_setinputsizes: Optional[_GenericSetInputSizesType] = None
        batchnum = 1
        total_batches = lenparams // batch_size + (
            1 if lenparams % batch_size else 0
        )

        insert_crud_params = imv.insert_crud_params
        assert insert_crud_params is not None

        if rst:
            insert_crud_params = [
                (col, key, rst(expr), st)
                for col, key, expr, st in insert_crud_params
            ]

        escaped_bind_names: Mapping[str, str]
        expand_pos_lower_index = expand_pos_upper_index = 0

        if not self.positional:
            if self.escaped_bind_names:
                escaped_bind_names = self.escaped_bind_names
            else:
                escaped_bind_names = {}

            all_keys = set(parameters[0])

            def apply_placeholders(keys, formatted):
                for key in keys:
                    key = escaped_bind_names.get(key, key)
                    formatted = formatted.replace(
                        self.bindtemplate % {"name": key},
                        self.bindtemplate
                        % {"name": f"{key}__EXECMANY_INDEX__"},
                    )
                return formatted

            if imv.embed_values_counter:
                imv_values_counter = ", _IMV_VALUES_COUNTER"
            else:
                imv_values_counter = ""
            formatted_values_clause = f"""({', '.join(
                apply_placeholders(bind_keys, formatted)
                for _, _, formatted, bind_keys in insert_crud_params
            )}{imv_values_counter})"""

            keys_to_replace = all_keys.intersection(
                escaped_bind_names.get(key, key)
                for _, _, _, bind_keys in insert_crud_params
                for key in bind_keys
            )
            base_parameters = {
                key: parameters[0][key]
                for key in all_keys.difference(keys_to_replace)
            }
            executemany_values_w_comma = ""
        else:
            formatted_values_clause = ""
            keys_to_replace = set()
            base_parameters = {}

            if imv.embed_values_counter:
                executemany_values_w_comma = (
                    f"({imv_single_values_expr}, _IMV_VALUES_COUNTER), "
                )
            else:
                executemany_values_w_comma = f"({imv_single_values_expr}), "

            all_names_we_will_expand: Set[str] = set()
            for elem in imv.insert_crud_params:
                all_names_we_will_expand.update(elem[3])

            # get the start and end position in a particular list
            # of parameters where we will be doing the "expanding".
            # statements can have params on either side or both sides,
            # given RETURNING and CTEs
            if all_names_we_will_expand:
                positiontup = self.positiontup
                assert positiontup is not None

                all_expand_positions = {
                    idx
                    for idx, name in enumerate(positiontup)
                    if name in all_names_we_will_expand
                }
                expand_pos_lower_index = min(all_expand_positions)
                expand_pos_upper_index = max(all_expand_positions) + 1
                assert (
                    len(all_expand_positions)
                    == expand_pos_upper_index - expand_pos_lower_index
                )

            if self._numeric_binds:
                escaped = re.escape(self._numeric_binds_identifier_char)
                executemany_values_w_comma = re.sub(
                    rf"{escaped}\d+", "%s", executemany_values_w_comma
                )

        while batches:
            batch = batches[0:batch_size]
            compiled_batch = compiled_batches[0:batch_size]

            batches[0:batch_size] = []
            compiled_batches[0:batch_size] = []

            if batches:
                current_batch_size = batch_size
            else:
                current_batch_size = len(batch)

            if generic_setinputsizes:
                # if setinputsizes is present, expand this collection to
                # suit the batch length as well
                # currently this will be mssql+pyodbc for internal dialects
                processed_setinputsizes = [
                    (new_key, len_, typ)
                    for new_key, len_, typ in (
                        (f"{key}_{index}", len_, typ)
                        for index in range(current_batch_size)
                        for key, len_, typ in generic_setinputsizes
                    )
                ]

            replaced_parameters: Any
            if self.positional:
                num_ins_params = imv.num_positional_params_counted

                batch_iterator: Iterable[Sequence[Any]]
                extra_params_left: Sequence[Any]
                extra_params_right: Sequence[Any]

                if num_ins_params == len(batch[0]):
                    extra_params_left = extra_params_right = ()
                    batch_iterator = batch
                else:
                    extra_params_left = batch[0][:expand_pos_lower_index]
                    extra_params_right = batch[0][expand_pos_upper_index:]
                    batch_iterator = (
                        b[expand_pos_lower_index:expand_pos_upper_index]
                        for b in batch
                    )

                if imv.embed_values_counter:
                    expanded_values_string = (
                        "".join(
                            executemany_values_w_comma.replace(
                                "_IMV_VALUES_COUNTER", str(i)
                            )
                            for i, _ in enumerate(batch)
                        )
                    )[:-2]
                else:
                    expanded_values_string = (
                        (executemany_values_w_comma * current_batch_size)
                    )[:-2]

                if self._numeric_binds and num_ins_params > 0:
                    # numeric will always number the parameters inside of
                    # VALUES (and thus order self.positiontup) to be higher
                    # than non-VALUES parameters, no matter where in the
                    # statement those non-VALUES parameters appear (this is
                    # ensured in _process_numeric by numbering first all
                    # params that are not in _values_bindparam)
                    # therefore all extra params are always
                    # on the left side and numbered lower than the VALUES
                    # parameters
                    assert not extra_params_right

                    start = expand_pos_lower_index + 1
                    end = num_ins_params * (current_batch_size) + start

                    # need to format here, since statement may contain
                    # unescaped %, while values_string contains just (%s, %s)
                    positions = tuple(
                        f"{self._numeric_binds_identifier_char}{i}"
                        for i in range(start, end)
                    )
                    expanded_values_string = expanded_values_string % positions

                replaced_statement = statement.replace(
                    "__EXECMANY_TOKEN__", expanded_values_string
                )

                replaced_parameters = tuple(
                    itertools.chain.from_iterable(batch_iterator)
                )

                replaced_parameters = (
                    extra_params_left
                    + replaced_parameters
                    + extra_params_right
                )

            else:
                replaced_values_clauses = []
                replaced_parameters = base_parameters.copy()

                for i, param in enumerate(batch):
                    fmv = formatted_values_clause.replace(
                        "EXECMANY_INDEX__", str(i)
                    )
                    if imv.embed_values_counter:
                        fmv = fmv.replace("_IMV_VALUES_COUNTER", str(i))

                    replaced_values_clauses.append(fmv)
                    replaced_parameters.update(
                        {f"{key}__{i}": param[key] for key in keys_to_replace}
                    )

                replaced_statement = statement.replace(
                    "__EXECMANY_TOKEN__",
                    ", ".join(replaced_values_clauses),
                )

            yield _InsertManyValuesBatch(
                replaced_statement,
                replaced_parameters,
                processed_setinputsizes,
                batch,
                (
                    [_sentinel_from_params(cb) for cb in compiled_batch]
                    if _sentinel_from_params
                    else []
                ),
                current_batch_size,
                batchnum,
                total_batches,
                sort_by_parameter_order,
                False,
            )
            batchnum += 1

    def visit_insert(
        self, insert_stmt, visited_bindparam=None, visiting_cte=None, **kw
    ):
        compile_state = insert_stmt._compile_state_factory(
            insert_stmt, self, **kw
        )
        insert_stmt = compile_state.statement

        if visiting_cte is not None:
            kw["visiting_cte"] = visiting_cte
            toplevel = False
        else:
            toplevel = not self.stack

        if toplevel:
            self.isinsert = True
            if not self.dml_compile_state:
                self.dml_compile_state = compile_state
            if not self.compile_state:
                self.compile_state = compile_state

        self.stack.append(
            {
                "correlate_froms": set(),
                "asfrom_froms": set(),
                "selectable": insert_stmt,
            }
        )

        counted_bindparam = 0

        # reset any incoming "visited_bindparam" collection
        visited_bindparam = None

        # for positional, insertmanyvalues needs to know how many
        # bound parameters are in the VALUES sequence; there's no simple
        # rule because default expressions etc. can have zero or more
        # params inside them.   After multiple attempts to figure this out,
        # this very simplistic "count after" works and is
        # likely the least amount of callcounts, though looks clumsy
        if self.positional and visiting_cte is None:
            # if we are inside a CTE, don't count parameters
            # here since they wont be for insertmanyvalues. keep
            # visited_bindparam at None so no counting happens.
            # see #9173
            visited_bindparam = []

        crud_params_struct = crud._get_crud_params(
            self,
            insert_stmt,
            compile_state,
            toplevel,
            visited_bindparam=visited_bindparam,
            **kw,
        )

        if self.positional and visited_bindparam is not None:
            counted_bindparam = len(visited_bindparam)
            if self._numeric_binds:
                if self._values_bindparam is not None:
                    self._values_bindparam += visited_bindparam
                else:
                    self._values_bindparam = visited_bindparam

        crud_params_single = crud_params_struct.single_params

        if (
            not crud_params_single
            and not self.dialect.supports_default_values
            and not self.dialect.supports_default_metavalue
            and not self.dialect.supports_empty_insert
        ):
            raise exc.CompileError(
                "The '%s' dialect with current database "
                "version settings does not support empty "
                "inserts." % self.dialect.name
            )

        if compile_state._has_multi_parameters:
            if not self.dialect.supports_multivalues_insert:
                raise exc.CompileError(
                    "The '%s' dialect with current database "
                    "version settings does not support "
                    "in-place multirow inserts." % self.dialect.name
                )
            elif (
                self.implicit_returning or insert_stmt._returning
            ) and insert_stmt._sort_by_parameter_order:
                raise exc.CompileError(
                    "RETURNING cannot be determinstically sorted when "
                    "using an INSERT which includes multi-row values()."
                )
            crud_params_single = crud_params_struct.single_params
        else:
            crud_params_single = crud_params_struct.single_params

        preparer = self.preparer
        supports_default_values = self.dialect.supports_default_values

        text = "INSERT "

        if insert_stmt._prefixes:
            text += self._generate_prefixes(
                insert_stmt, insert_stmt._prefixes, **kw
            )

        text += "INTO "
        table_text = preparer.format_table(insert_stmt.table)

        if insert_stmt._hints:
            _, table_text = self._setup_crud_hints(insert_stmt, table_text)

        if insert_stmt._independent_ctes:
            self._dispatch_independent_ctes(insert_stmt, kw)

        text += table_text

        if crud_params_single or not supports_default_values:
            text += " (%s)" % ", ".join(
                [expr for _, expr, _, _ in crud_params_single]
            )

        # look for insertmanyvalues attributes that would have been configured
        # by crud.py as it scanned through the columns to be part of the
        # INSERT
        use_insertmanyvalues = crud_params_struct.use_insertmanyvalues
        named_sentinel_params: Optional[Sequence[str]] = None
        add_sentinel_cols = None
        implicit_sentinel = False

        returning_cols = self.implicit_returning or insert_stmt._returning
        if returning_cols:
            add_sentinel_cols = crud_params_struct.use_sentinel_columns
            if add_sentinel_cols is not None:
                assert use_insertmanyvalues

                # search for the sentinel column explicitly present
                # in the INSERT columns list, and additionally check that
                # this column has a bound parameter name set up that's in the
                # parameter list.  If both of these cases are present, it means
                # we will have a client side value for the sentinel in each
                # parameter set.

                _params_by_col = {
                    col: param_names
                    for col, _, _, param_names in crud_params_single
                }
                named_sentinel_params = []
                for _add_sentinel_col in add_sentinel_cols:
                    if _add_sentinel_col not in _params_by_col:
                        named_sentinel_params = None
                        break
                    param_name = self._within_exec_param_key_getter(
                        _add_sentinel_col
                    )
                    if param_name not in _params_by_col[_add_sentinel_col]:
                        named_sentinel_params = None
                        break
                    named_sentinel_params.append(param_name)

                if named_sentinel_params is None:
                    # if we are not going to have a client side value for
                    # the sentinel in the parameter set, that means it's
                    # an autoincrement, an IDENTITY, or a server-side SQL
                    # expression like nextval('seqname').  So this is
                    # an "implicit" sentinel; we will look for it in
                    # RETURNING
                    # only, and then sort on it.  For this case on PG,
                    # SQL Server we have to use a special INSERT form
                    # that guarantees the server side function lines up with
                    # the entries in the VALUES.
                    if (
                        self.dialect.insertmanyvalues_implicit_sentinel
                        & InsertmanyvaluesSentinelOpts.ANY_AUTOINCREMENT
                    ):
                        implicit_sentinel = True
                    else:
                        # here, we are not using a sentinel at all
                        # and we are likely the SQLite dialect.
                        # The first add_sentinel_col that we have should not
                        # be marked as "insert_sentinel=True".  if it was,
                        # an error should have been raised in
                        # _get_sentinel_column_for_table.
                        assert not add_sentinel_cols[0]._insert_sentinel, (
                            "sentinel selection rules should have prevented "
                            "us from getting here for this dialect"
                        )

                # always put the sentinel columns last.  even if they are
                # in the returning list already, they will be there twice
                # then.
                returning_cols = list(returning_cols) + list(add_sentinel_cols)

            returning_clause = self.returning_clause(
                insert_stmt,
                returning_cols,
                populate_result_map=toplevel,
            )

            if self.returning_precedes_values:
                text += " " + returning_clause

        else:
            returning_clause = None

        if insert_stmt.select is not None:
            # placed here by crud.py
            select_text = self.process(
                self.stack[-1]["insert_from_select"], insert_into=True, **kw
            )

            if self.ctes and self.dialect.cte_follows_insert:
                nesting_level = len(self.stack) if not toplevel else None
                text += " %s%s" % (
                    self._render_cte_clause(
                        nesting_level=nesting_level,
                        include_following_stack=True,
                    ),
                    select_text,
                )
            else:
                text += " %s" % select_text
        elif not crud_params_single and supports_default_values:
            text += " DEFAULT VALUES"
            if use_insertmanyvalues:
                self._insertmanyvalues = _InsertManyValues(
                    True,
                    self.dialect.default_metavalue_token,
                    cast(
                        "List[crud._CrudParamElementStr]", crud_params_single
                    ),
                    counted_bindparam,
                    sort_by_parameter_order=(
                        insert_stmt._sort_by_parameter_order
                    ),
                    includes_upsert_behaviors=(
                        insert_stmt._post_values_clause is not None
                    ),
                    sentinel_columns=add_sentinel_cols,
                    num_sentinel_columns=(
                        len(add_sentinel_cols) if add_sentinel_cols else 0
                    ),
                    implicit_sentinel=implicit_sentinel,
                )
        elif compile_state._has_multi_parameters:
            text += " VALUES %s" % (
                ", ".join(
                    "(%s)"
                    % (", ".join(value for _, _, value, _ in crud_param_set))
                    for crud_param_set in crud_params_struct.all_multi_params
                ),
            )
        else:
            insert_single_values_expr = ", ".join(
                [
                    value
                    for _, _, value, _ in cast(
                        "List[crud._CrudParamElementStr]",
                        crud_params_single,
                    )
                ]
            )

            if use_insertmanyvalues:
                if (
                    implicit_sentinel
                    and (
                        self.dialect.insertmanyvalues_implicit_sentinel
                        & InsertmanyvaluesSentinelOpts.USE_INSERT_FROM_SELECT
                    )
                    # this is checking if we have
                    # INSERT INTO table (id) VALUES (DEFAULT).
                    and not (crud_params_struct.is_default_metavalue_only)
                ):
                    # if we have a sentinel column that is server generated,
                    # then for selected backends render the VALUES list as a
                    # subquery.  This is the orderable form supported by
                    # PostgreSQL and SQL Server.
                    embed_sentinel_value = True

                    render_bind_casts = (
                        self.dialect.insertmanyvalues_implicit_sentinel
                        & InsertmanyvaluesSentinelOpts.RENDER_SELECT_COL_CASTS
                    )

                    colnames = ", ".join(
                        f"p{i}" for i, _ in enumerate(crud_params_single)
                    )

                    if render_bind_casts:
                        # render casts for the SELECT list.  For PG, we are
                        # already rendering bind casts in the parameter list,
                        # selectively for the more "tricky" types like ARRAY.
                        # however, even for the "easy" types, if the parameter
                        # is NULL for every entry, PG gives up and says
                        # "it must be TEXT", which fails for other easy types
                        # like ints.  So we cast on this side too.
                        colnames_w_cast = ", ".join(
                            self.render_bind_cast(
                                col.type,
                                col.type._unwrapped_dialect_impl(self.dialect),
                                f"p{i}",
                            )
                            for i, (col, *_) in enumerate(crud_params_single)
                        )
                    else:
                        colnames_w_cast = colnames

                    text += (
                        f" SELECT {colnames_w_cast} FROM "
                        f"(VALUES ({insert_single_values_expr})) "
                        f"AS imp_sen({colnames}, sen_counter) "
                        "ORDER BY sen_counter"
                    )
                else:
                    # otherwise, if no sentinel or backend doesn't support
                    # orderable subquery form, use a plain VALUES list
                    embed_sentinel_value = False
                    text += f" VALUES ({insert_single_values_expr})"

                self._insertmanyvalues = _InsertManyValues(
                    is_default_expr=False,
                    single_values_expr=insert_single_values_expr,
                    insert_crud_params=cast(
                        "List[crud._CrudParamElementStr]",
                        crud_params_single,
                    ),
                    num_positional_params_counted=counted_bindparam,
                    sort_by_parameter_order=(
                        insert_stmt._sort_by_parameter_order
                    ),
                    includes_upsert_behaviors=(
                        insert_stmt._post_values_clause is not None
                    ),
                    sentinel_columns=add_sentinel_cols,
                    num_sentinel_columns=(
                        len(add_sentinel_cols) if add_sentinel_cols else 0
                    ),
                    sentinel_param_keys=named_sentinel_params,
                    implicit_sentinel=implicit_sentinel,
                    embed_values_counter=embed_sentinel_value,
                )

            else:
                text += f" VALUES ({insert_single_values_expr})"

        if insert_stmt._post_values_clause is not None:
            post_values_clause = self.process(
                insert_stmt._post_values_clause, **kw
            )
            if post_values_clause:
                text += " " + post_values_clause

        if returning_clause and not self.returning_precedes_values:
            text += " " + returning_clause

        if self.ctes and not self.dialect.cte_follows_insert:
            nesting_level = len(self.stack) if not toplevel else None
            text = (
                self._render_cte_clause(
                    nesting_level=nesting_level,
                    include_following_stack=True,
                )
                + text
            )

        self.stack.pop(-1)

        return text

    def update_limit_clause(self, update_stmt):
        """Provide a hook for MySQL to add LIMIT to the UPDATE"""
        return None

    def delete_limit_clause(self, delete_stmt):
        """Provide a hook for MySQL to add LIMIT to the DELETE"""
        return None

    def update_tables_clause(self, update_stmt, from_table, extra_froms, **kw):
        """Provide a hook to override the initial table clause
        in an UPDATE statement.

        MySQL overrides this.

        """
        kw["asfrom"] = True
        return from_table._compiler_dispatch(self, iscrud=True, **kw)

    def update_from_clause(
        self, update_stmt, from_table, extra_froms, from_hints, **kw
    ):
        """Provide a hook to override the generation of an
        UPDATE..FROM clause.

        MySQL and MSSQL override this.

        """
        raise NotImplementedError(
            "This backend does not support multiple-table "
            "criteria within UPDATE"
        )

    def visit_update(
        self,
        update_stmt: Update,
        visiting_cte: Optional[CTE] = None,
        **kw: Any,
    ) -> str:
        compile_state = update_stmt._compile_state_factory(  # type: ignore[call-arg] # noqa: E501
            update_stmt, self, **kw  # type: ignore[arg-type]
        )
        if TYPE_CHECKING:
            assert isinstance(compile_state, UpdateDMLState)
        update_stmt = compile_state.statement  # type: ignore[assignment]

        if visiting_cte is not None:
            kw["visiting_cte"] = visiting_cte
            toplevel = False
        else:
            toplevel = not self.stack

        if toplevel:
            self.isupdate = True
            if not self.dml_compile_state:
                self.dml_compile_state = compile_state
            if not self.compile_state:
                self.compile_state = compile_state

        if self.linting & COLLECT_CARTESIAN_PRODUCTS:
            from_linter = FromLinter({}, set())
            warn_linting = self.linting & WARN_LINTING
            if toplevel:
                self.from_linter = from_linter
        else:
            from_linter = None
            warn_linting = False

        extra_froms = compile_state._extra_froms
        is_multitable = bool(extra_froms)

        if is_multitable:
            # main table might be a JOIN
            main_froms = set(_from_objects(update_stmt.table))
            render_extra_froms = [
                f for f in extra_froms if f not in main_froms
            ]
            correlate_froms = main_froms.union(extra_froms)
        else:
            render_extra_froms = []
            correlate_froms = {update_stmt.table}

        self.stack.append(
            {
                "correlate_froms": correlate_froms,
                "asfrom_froms": correlate_froms,
                "selectable": update_stmt,
            }
        )

        text = "UPDATE "

        if update_stmt._prefixes:
            text += self._generate_prefixes(
                update_stmt, update_stmt._prefixes, **kw
            )

        table_text = self.update_tables_clause(
            update_stmt,
            update_stmt.table,
            render_extra_froms,
            from_linter=from_linter,
            **kw,
        )
        crud_params_struct = crud._get_crud_params(
            self, update_stmt, compile_state, toplevel, **kw
        )
        crud_params = crud_params_struct.single_params

        if update_stmt._hints:
            dialect_hints, table_text = self._setup_crud_hints(
                update_stmt, table_text
            )
        else:
            dialect_hints = None

        if update_stmt._independent_ctes:
            self._dispatch_independent_ctes(update_stmt, kw)

        text += table_text

        text += " SET "
        text += ", ".join(
            expr + "=" + value
            for _, expr, value, _ in cast(
                "List[Tuple[Any, str, str, Any]]", crud_params
            )
        )

        if self.implicit_returning or update_stmt._returning:
            if self.returning_precedes_values:
                text += " " + self.returning_clause(
                    update_stmt,
                    self.implicit_returning or update_stmt._returning,
                    populate_result_map=toplevel,
                )

        if extra_froms:
            extra_from_text = self.update_from_clause(
                update_stmt,
                update_stmt.table,
                render_extra_froms,
                dialect_hints,
                from_linter=from_linter,
                **kw,
            )
            if extra_from_text:
                text += " " + extra_from_text

        if update_stmt._where_criteria:
            t = self._generate_delimited_and_list(
                update_stmt._where_criteria, from_linter=from_linter, **kw
            )
            if t:
                text += " WHERE " + t

        limit_clause = self.update_limit_clause(update_stmt)
        if limit_clause:
            text += " " + limit_clause

        if (
            self.implicit_returning or update_stmt._returning
        ) and not self.returning_precedes_values:
            text += " " + self.returning_clause(
                update_stmt,
                self.implicit_returning or update_stmt._returning,
                populate_result_map=toplevel,
            )

        if self.ctes:
            nesting_level = len(self.stack) if not toplevel else None
            text = self._render_cte_clause(nesting_level=nesting_level) + text

        if warn_linting:
            assert from_linter is not None
            from_linter.warn(stmt_type="UPDATE")

        self.stack.pop(-1)

        return text

    def delete_extra_from_clause(
        self, delete_stmt, from_table, extra_froms, from_hints, **kw
    ):
        """Provide a hook to override the generation of an
        DELETE..FROM clause.

        This can be used to implement DELETE..USING for example.

        MySQL and MSSQL override this.

        """
        raise NotImplementedError(
            "This backend does not support multiple-table "
            "criteria within DELETE"
        )

    def delete_table_clause(self, delete_stmt, from_table, extra_froms, **kw):
        return from_table._compiler_dispatch(
            self, asfrom=True, iscrud=True, **kw
        )

    def visit_delete(self, delete_stmt, visiting_cte=None, **kw):
        compile_state = delete_stmt._compile_state_factory(
            delete_stmt, self, **kw
        )
        delete_stmt = compile_state.statement

        if visiting_cte is not None:
            kw["visiting_cte"] = visiting_cte
            toplevel = False
        else:
            toplevel = not self.stack

        if toplevel:
            self.isdelete = True
            if not self.dml_compile_state:
                self.dml_compile_state = compile_state
            if not self.compile_state:
                self.compile_state = compile_state

        if self.linting & COLLECT_CARTESIAN_PRODUCTS:
            from_linter = FromLinter({}, set())
            warn_linting = self.linting & WARN_LINTING
            if toplevel:
                self.from_linter = from_linter
        else:
            from_linter = None
            warn_linting = False

        extra_froms = compile_state._extra_froms

        correlate_froms = {delete_stmt.table}.union(extra_froms)
        self.stack.append(
            {
                "correlate_froms": correlate_froms,
                "asfrom_froms": correlate_froms,
                "selectable": delete_stmt,
            }
        )

        text = "DELETE "

        if delete_stmt._prefixes:
            text += self._generate_prefixes(
                delete_stmt, delete_stmt._prefixes, **kw
            )

        text += "FROM "

        try:
            table_text = self.delete_table_clause(
                delete_stmt,
                delete_stmt.table,
                extra_froms,
                from_linter=from_linter,
            )
        except TypeError:
            # anticipate 3rd party dialects that don't include **kw
            # TODO: remove in 2.1
            table_text = self.delete_table_clause(
                delete_stmt, delete_stmt.table, extra_froms
            )
            if from_linter:
                _ = self.process(delete_stmt.table, from_linter=from_linter)

        crud._get_crud_params(self, delete_stmt, compile_state, toplevel, **kw)

        if delete_stmt._hints:
            dialect_hints, table_text = self._setup_crud_hints(
                delete_stmt, table_text
            )
        else:
            dialect_hints = None

        if delete_stmt._independent_ctes:
            self._dispatch_independent_ctes(delete_stmt, kw)

        text += table_text

        if (
            self.implicit_returning or delete_stmt._returning
        ) and self.returning_precedes_values:
            text += " " + self.returning_clause(
                delete_stmt,
                self.implicit_returning or delete_stmt._returning,
                populate_result_map=toplevel,
            )

        if extra_froms:
            extra_from_text = self.delete_extra_from_clause(
                delete_stmt,
                delete_stmt.table,
                extra_froms,
                dialect_hints,
                from_linter=from_linter,
                **kw,
            )
            if extra_from_text:
                text += " " + extra_from_text

        if delete_stmt._where_criteria:
            t = self._generate_delimited_and_list(
                delete_stmt._where_criteria, from_linter=from_linter, **kw
            )
            if t:
                text += " WHERE " + t

        limit_clause = self.delete_limit_clause(delete_stmt)
        if limit_clause:
            text += " " + limit_clause

        if (
            self.implicit_returning or delete_stmt._returning
        ) and not self.returning_precedes_values:
            text += " " + self.returning_clause(
                delete_stmt,
                self.implicit_returning or delete_stmt._returning,
                populate_result_map=toplevel,
            )

        if self.ctes:
            nesting_level = len(self.stack) if not toplevel else None
            text = self._render_cte_clause(nesting_level=nesting_level) + text

        if warn_linting:
            assert from_linter is not None
            from_linter.warn(stmt_type="DELETE")

        self.stack.pop(-1)

        return text

    def visit_savepoint(self, savepoint_stmt, **kw):
        return "SAVEPOINT %s" % self.preparer.format_savepoint(savepoint_stmt)

    def visit_rollback_to_savepoint(self, savepoint_stmt, **kw):
        return "ROLLBACK TO SAVEPOINT %s" % self.preparer.format_savepoint(
            savepoint_stmt
        )

    def visit_release_savepoint(self, savepoint_stmt, **kw):
        return "RELEASE SAVEPOINT %s" % self.preparer.format_savepoint(
            savepoint_stmt
        )


class StrSQLCompiler(SQLCompiler):
    """A :class:`.SQLCompiler` subclass which allows a small selection
    of non-standard SQL features to render into a string value.

    The :class:`.StrSQLCompiler` is invoked whenever a Core expression
    element is directly stringified without calling upon the
    :meth:`_expression.ClauseElement.compile` method.
    It can render a limited set
    of non-standard SQL constructs to assist in basic stringification,
    however for more substantial custom or dialect-specific SQL constructs,
    it will be necessary to make use of
    :meth:`_expression.ClauseElement.compile`
    directly.

    .. seealso::

        :ref:`faq_sql_expression_string`

    """

    def _fallback_column_name(self, column):
        return "<name unknown>"

    @util.preload_module("sqlalchemy.engine.url")
    def visit_unsupported_compilation(self, element, err, **kw):
        if element.stringify_dialect != "default":
            url = util.preloaded.engine_url
            dialect = url.URL.create(element.stringify_dialect).get_dialect()()

            compiler = dialect.statement_compiler(
                dialect, None, _supporting_against=self
            )
            if not isinstance(compiler, StrSQLCompiler):
                return compiler.process(element, **kw)

        return super().visit_unsupported_compilation(element, err)

    def visit_getitem_binary(self, binary, operator, **kw):
        return "%s[%s]" % (
            self.process(binary.left, **kw),
            self.process(binary.right, **kw),
        )

    def visit_json_getitem_op_binary(self, binary, operator, **kw):
        return self.visit_getitem_binary(binary, operator, **kw)

    def visit_json_path_getitem_op_binary(self, binary, operator, **kw):
        return self.visit_getitem_binary(binary, operator, **kw)

    def visit_sequence(self, sequence, **kw):
        return (
            f"<next sequence value: {self.preparer.format_sequence(sequence)}>"
        )

    def returning_clause(
        self,
        stmt: UpdateBase,
        returning_cols: Sequence[_ColumnsClauseElement],
        *,
        populate_result_map: bool,
        **kw: Any,
    ) -> str:
        columns = [
            self._label_select_column(None, c, True, False, {})
            for c in base._select_iterables(returning_cols)
        ]
        return "RETURNING " + ", ".join(columns)

    def update_from_clause(
        self, update_stmt, from_table, extra_froms, from_hints, **kw
    ):
        kw["asfrom"] = True
        return "FROM " + ", ".join(
            t._compiler_dispatch(self, fromhints=from_hints, **kw)
            for t in extra_froms
        )

    def delete_extra_from_clause(
        self, delete_stmt, from_table, extra_froms, from_hints, **kw
    ):
        kw["asfrom"] = True
        return ", " + ", ".join(
            t._compiler_dispatch(self, fromhints=from_hints, **kw)
            for t in extra_froms
        )

    def visit_empty_set_expr(self, element_types, **kw):
        return "SELECT 1 WHERE 1!=1"

    def get_from_hint_text(self, table, text):
        return "[%s]" % text

    def visit_regexp_match_op_binary(self, binary, operator, **kw):
        return self._generate_generic_binary(binary, " <regexp> ", **kw)

    def visit_not_regexp_match_op_binary(self, binary, operator, **kw):
        return self._generate_generic_binary(binary, " <not regexp> ", **kw)

    def visit_regexp_replace_op_binary(self, binary, operator, **kw):
        return "<regexp replace>(%s, %s)" % (
            binary.left._compiler_dispatch(self, **kw),
            binary.right._compiler_dispatch(self, **kw),
        )

    def visit_try_cast(self, cast, **kwargs):
        return "TRY_CAST(%s AS %s)" % (
            cast.clause._compiler_dispatch(self, **kwargs),
            cast.typeclause._compiler_dispatch(self, **kwargs),
        )


class DDLCompiler(Compiled):
    is_ddl = True

    if TYPE_CHECKING:

        def __init__(
            self,
            dialect: Dialect,
            statement: ExecutableDDLElement,
            schema_translate_map: Optional[SchemaTranslateMapType] = ...,
            render_schema_translate: bool = ...,
            compile_kwargs: Mapping[str, Any] = ...,
        ): ...

    @util.ro_memoized_property
    def sql_compiler(self) -> SQLCompiler:
        return self.dialect.statement_compiler(
            self.dialect, None, schema_translate_map=self.schema_translate_map
        )

    @util.memoized_property
    def type_compiler(self):
        return self.dialect.type_compiler_instance

    def construct_params(
        self,
        params: Optional[_CoreSingleExecuteParams] = None,
        extracted_parameters: Optional[Sequence[BindParameter[Any]]] = None,
        escape_names: bool = True,
    ) -> Optional[_MutableCoreSingleExecuteParams]:
        return None

    def visit_ddl(self, ddl, **kwargs):
        # table events can substitute table and schema name
        context = ddl.context
        if isinstance(ddl.target, schema.Table):
            context = context.copy()

            preparer = self.preparer
            path = preparer.format_table_seq(ddl.target)
            if len(path) == 1:
                table, sch = path[0], ""
            else:
                table, sch = path[-1], path[0]

            context.setdefault("table", table)
            context.setdefault("schema", sch)
            context.setdefault("fullname", preparer.format_table(ddl.target))

        return self.sql_compiler.post_process_text(ddl.statement % context)

    def visit_create_schema(self, create, **kw):
        text = "CREATE SCHEMA "
        if create.if_not_exists:
            text += "IF NOT EXISTS "
        return text + self.preparer.format_schema(create.element)

    def visit_drop_schema(self, drop, **kw):
        text = "DROP SCHEMA "
        if drop.if_exists:
            text += "IF EXISTS "
        text += self.preparer.format_schema(drop.element)
        if drop.cascade:
            text += " CASCADE"
        return text

    def visit_create_table(self, create, **kw):
        table = create.element
        preparer = self.preparer

        text = "\nCREATE "
        if table._prefixes:
            text += " ".join(table._prefixes) + " "

        text += "TABLE "
        if create.if_not_exists:
            text += "IF NOT EXISTS "

        text += preparer.format_table(table) + " "

        create_table_suffix = self.create_table_suffix(table)
        if create_table_suffix:
            text += create_table_suffix + " "

        text += "("

        separator = "\n"

        # if only one primary key, specify it along with the column
        first_pk = False
        for create_column in create.columns:
            column = create_column.element
            try:
                processed = self.process(
                    create_column, first_pk=column.primary_key and not first_pk
                )
                if processed is not None:
                    text += separator
                    separator = ", \n"
                    text += "\t" + processed
                if column.primary_key:
                    first_pk = True
            except exc.CompileError as ce:
                raise exc.CompileError(
                    "(in table '%s', column '%s'): %s"
                    % (table.description, column.name, ce.args[0])
                ) from ce

        const = self.create_table_constraints(
            table,
            _include_foreign_key_constraints=create.include_foreign_key_constraints,  # noqa
        )
        if const:
            text += separator + "\t" + const

        text += "\n)%s\n\n" % self.post_create_table(table)
        return text

    def visit_create_column(self, create, first_pk=False, **kw):
        column = create.element

        if column.system:
            return None

        text = self.get_column_specification(column, first_pk=first_pk)
        const = " ".join(
            self.process(constraint) for constraint in column.constraints
        )
        if const:
            text += " " + const

        return text

    def create_table_constraints(
        self, table, _include_foreign_key_constraints=None, **kw
    ):
        # On some DB order is significant: visit PK first, then the
        # other constraints (engine.ReflectionTest.testbasic failed on FB2)
        constraints = []
        if table.primary_key:
            constraints.append(table.primary_key)

        all_fkcs = table.foreign_key_constraints
        if _include_foreign_key_constraints is not None:
            omit_fkcs = all_fkcs.difference(_include_foreign_key_constraints)
        else:
            omit_fkcs = set()

        constraints.extend(
            [
                c
                for c in table._sorted_constraints
                if c is not table.primary_key and c not in omit_fkcs
            ]
        )

        return ", \n\t".join(
            p
            for p in (
                self.process(constraint)
                for constraint in constraints
                if (constraint._should_create_for_compiler(self))
                and (
                    not self.dialect.supports_alter
                    or not getattr(constraint, "use_alter", False)
                )
            )
            if p is not None
        )

    def visit_drop_table(self, drop, **kw):
        text = "\nDROP TABLE "
        if drop.if_exists:
            text += "IF EXISTS "
        return text + self.preparer.format_table(drop.element)

    def visit_drop_view(self, drop, **kw):
        return "\nDROP VIEW " + self.preparer.format_table(drop.element)

    def _verify_index_table(self, index: Index) -> None:
        if index.table is None:
            raise exc.CompileError(
                "Index '%s' is not associated with any table." % index.name
            )

    def visit_create_index(
        self, create, include_schema=False, include_table_schema=True, **kw
    ):
        index = create.element
        self._verify_index_table(index)
        preparer = self.preparer
        text = "CREATE "
        if index.unique:
            text += "UNIQUE "
        if index.name is None:
            raise exc.CompileError(
                "CREATE INDEX requires that the index have a name"
            )

        text += "INDEX "
        if create.if_not_exists:
            text += "IF NOT EXISTS "

        text += "%s ON %s (%s)" % (
            self._prepared_index_name(index, include_schema=include_schema),
            preparer.format_table(
                index.table, use_schema=include_table_schema
            ),
            ", ".join(
                self.sql_compiler.process(
                    expr, include_table=False, literal_binds=True
                )
                for expr in index.expressions
            ),
        )
        return text

    def visit_drop_index(self, drop, **kw):
        index = drop.element

        if index.name is None:
            raise exc.CompileError(
                "DROP INDEX requires that the index have a name"
            )
        text = "\nDROP INDEX "
        if drop.if_exists:
            text += "IF EXISTS "

        return text + self._prepared_index_name(index, include_schema=True)

    def _prepared_index_name(
        self, index: Index, include_schema: bool = False
    ) -> str:
        if index.table is not None:
            effective_schema = self.preparer.schema_for_object(index.table)
        else:
            effective_schema = None
        if include_schema and effective_schema:
            schema_name = self.preparer.quote_schema(effective_schema)
        else:
            schema_name = None

        index_name = self.preparer.format_index(index)

        if schema_name:
            index_name = schema_name + "." + index_name
        return index_name

    def visit_add_constraint(self, create, **kw):
        return "ALTER TABLE %s ADD %s" % (
            self.preparer.format_table(create.element.table),
            self.process(create.element),
        )

    def visit_set_table_comment(self, create, **kw):
        return "COMMENT ON TABLE %s IS %s" % (
            self.preparer.format_table(create.element),
            self.sql_compiler.render_literal_value(
                create.element.comment, sqltypes.String()
            ),
        )

    def visit_drop_table_comment(self, drop, **kw):
        return "COMMENT ON TABLE %s IS NULL" % self.preparer.format_table(
            drop.element
        )

    def visit_set_column_comment(self, create, **kw):
        return "COMMENT ON COLUMN %s IS %s" % (
            self.preparer.format_column(
                create.element, use_table=True, use_schema=True
            ),
            self.sql_compiler.render_literal_value(
                create.element.comment, sqltypes.String()
            ),
        )

    def visit_drop_column_comment(self, drop, **kw):
        return "COMMENT ON COLUMN %s IS NULL" % self.preparer.format_column(
            drop.element, use_table=True
        )

    def visit_set_constraint_comment(self, create, **kw):
        raise exc.UnsupportedCompilationError(self, type(create))

    def visit_drop_constraint_comment(self, drop, **kw):
        raise exc.UnsupportedCompilationError(self, type(drop))

    def get_identity_options(self, identity_options):
        text = []
        if identity_options.increment is not None:
            text.append("INCREMENT BY %d" % identity_options.increment)
        if identity_options.start is not None:
            text.append("START WITH %d" % identity_options.start)
        if identity_options.minvalue is not None:
            text.append("MINVALUE %d" % identity_options.minvalue)
        if identity_options.maxvalue is not None:
            text.append("MAXVALUE %d" % identity_options.maxvalue)
        if identity_options.nominvalue is not None:
            text.append("NO MINVALUE")
        if identity_options.nomaxvalue is not None:
            text.append("NO MAXVALUE")
        if identity_options.cache is not None:
            text.append("CACHE %d" % identity_options.cache)
        if identity_options.cycle is not None:
            text.append("CYCLE" if identity_options.cycle else "NO CYCLE")
        return " ".join(text)

    def visit_create_sequence(self, create, prefix=None, **kw):
        text = "CREATE SEQUENCE "
        if create.if_not_exists:
            text += "IF NOT EXISTS "
        text += self.preparer.format_sequence(create.element)

        if prefix:
            text += prefix
        options = self.get_identity_options(create.element)
        if options:
            text += " " + options
        return text

    def visit_drop_sequence(self, drop, **kw):
        text = "DROP SEQUENCE "
        if drop.if_exists:
            text += "IF EXISTS "
        return text + self.preparer.format_sequence(drop.element)

    def visit_drop_constraint(self, drop, **kw):
        constraint = drop.element
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
        else:
            formatted_name = None

        if formatted_name is None:
            raise exc.CompileError(
                "Can't emit DROP CONSTRAINT for constraint %r; "
                "it has no name" % drop.element
            )
        return "ALTER TABLE %s DROP CONSTRAINT %s%s%s" % (
            self.preparer.format_table(drop.element.table),
            "IF EXISTS " if drop.if_exists else "",
            formatted_name,
            " CASCADE" if drop.cascade else "",
        )

    def get_column_specification(self, column, **kwargs):
        colspec = (
            self.preparer.format_column(column)
            + " "
            + self.dialect.type_compiler_instance.process(
                column.type, type_expression=column
            )
        )
        default = self.get_column_default_string(column)
        if default is not None:
            colspec += " DEFAULT " + default

        if column.computed is not None:
            colspec += " " + self.process(column.computed)

        if (
            column.identity is not None
            and self.dialect.supports_identity_columns
        ):
            colspec += " " + self.process(column.identity)

        if not column.nullable and (
            not column.identity or not self.dialect.supports_identity_columns
        ):
            colspec += " NOT NULL"
        return colspec

    def create_table_suffix(self, table):
        return ""

    def post_create_table(self, table):
        return ""

    def get_column_default_string(self, column: Column[Any]) -> Optional[str]:
        if isinstance(column.server_default, schema.DefaultClause):
            return self.render_default_string(column.server_default.arg)
        else:
            return None

    def render_default_string(self, default: Union[Visitable, str]) -> str:
        if isinstance(default, str):
            return self.sql_compiler.render_literal_value(
                default, sqltypes.STRINGTYPE
            )
        else:
            return self.sql_compiler.process(default, literal_binds=True)

    def visit_table_or_column_check_constraint(self, constraint, **kw):
        if constraint.is_column_level:
            return self.visit_column_check_constraint(constraint)
        else:
            return self.visit_check_constraint(constraint)

    def visit_check_constraint(self, constraint, **kw):
        text = ""
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += "CONSTRAINT %s " % formatted_name
        text += "CHECK (%s)" % self.sql_compiler.process(
            constraint.sqltext, include_table=False, literal_binds=True
        )
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_column_check_constraint(self, constraint, **kw):
        text = ""
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += "CONSTRAINT %s " % formatted_name
        text += "CHECK (%s)" % self.sql_compiler.process(
            constraint.sqltext, include_table=False, literal_binds=True
        )
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_primary_key_constraint(
        self, constraint: PrimaryKeyConstraint, **kw: Any
    ) -> str:
        if len(constraint) == 0:
            return ""
        text = ""
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += "CONSTRAINT %s " % formatted_name
        text += "PRIMARY KEY "
        text += "(%s)" % ", ".join(
            self.preparer.quote(c.name)
            for c in (
                constraint.columns_autoinc_first
                if constraint._implicit_generated
                else constraint.columns
            )
        )
        text += self.define_constraint_deferrability(constraint)
        return text

    def visit_foreign_key_constraint(self, constraint, **kw):
        preparer = self.preparer
        text = ""
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += "CONSTRAINT %s " % formatted_name
        remote_table = list(constraint.elements)[0].column.table
        text += "FOREIGN KEY(%s) REFERENCES %s (%s)" % (
            ", ".join(
                preparer.quote(f.parent.name) for f in constraint.elements
            ),
            self.define_constraint_remote_table(
                constraint, remote_table, preparer
            ),
            ", ".join(
                preparer.quote(f.column.name) for f in constraint.elements
            ),
        )
        text += self.define_constraint_match(constraint)
        text += self.define_constraint_cascades(constraint)
        text += self.define_constraint_deferrability(constraint)
        return text

    def define_constraint_remote_table(self, constraint, table, preparer):
        """Format the remote table clause of a CREATE CONSTRAINT clause."""

        return preparer.format_table(table)

    def visit_unique_constraint(
        self, constraint: UniqueConstraint, **kw: Any
    ) -> str:
        if len(constraint) == 0:
            return ""
        text = ""
        if constraint.name is not None:
            formatted_name = self.preparer.format_constraint(constraint)
            if formatted_name is not None:
                text += "CONSTRAINT %s " % formatted_name
        text += "UNIQUE %s(%s)" % (
            self.define_unique_constraint_distinct(constraint, **kw),
            ", ".join(self.preparer.quote(c.name) for c in constraint),
        )
        text += self.define_constraint_deferrability(constraint)
        return text

    def define_unique_constraint_distinct(
        self, constraint: UniqueConstraint, **kw: Any
    ) -> str:
        return ""

    def define_constraint_cascades(
        self, constraint: ForeignKeyConstraint
    ) -> str:
        text = ""
        if constraint.ondelete is not None:
            text += self.define_constraint_ondelete_cascade(constraint)

        if constraint.onupdate is not None:
            text += self.define_constraint_onupdate_cascade(constraint)
        return text

    def define_constraint_ondelete_cascade(
        self, constraint: ForeignKeyConstraint
    ) -> str:
        return " ON DELETE %s" % self.preparer.validate_sql_phrase(
            constraint.ondelete, FK_ON_DELETE
        )

    def define_constraint_onupdate_cascade(
        self, constraint: ForeignKeyConstraint
    ) -> str:
        return " ON UPDATE %s" % self.preparer.validate_sql_phrase(
            constraint.onupdate, FK_ON_UPDATE
        )

    def define_constraint_deferrability(self, constraint: Constraint) -> str:
        text = ""
        if constraint.deferrable is not None:
            if constraint.deferrable:
                text += " DEFERRABLE"
            else:
                text += " NOT DEFERRABLE"
        if constraint.initially is not None:
            text += " INITIALLY %s" % self.preparer.validate_sql_phrase(
                constraint.initially, FK_INITIALLY
            )
        return text

    def define_constraint_match(self, constraint):
        text = ""
        if constraint.match is not None:
            text += " MATCH %s" % constraint.match
        return text

    def visit_computed_column(self, generated, **kw):
        text = "GENERATED ALWAYS AS (%s)" % self.sql_compiler.process(
            generated.sqltext, include_table=False, literal_binds=True
        )
        if generated.persisted is True:
            text += " STORED"
        elif generated.persisted is False:
            text += " VIRTUAL"
        return text

    def visit_identity_column(self, identity, **kw):
        text = "GENERATED %s AS IDENTITY" % (
            "ALWAYS" if identity.always else "BY DEFAULT",
        )
        options = self.get_identity_options(identity)
        if options:
            text += " (%s)" % options
        return text


class GenericTypeCompiler(TypeCompiler):
    def visit_FLOAT(self, type_: sqltypes.Float[Any], **kw: Any) -> str:
        return "FLOAT"

    def visit_DOUBLE(self, type_: sqltypes.Double[Any], **kw: Any) -> str:
        return "DOUBLE"

    def visit_DOUBLE_PRECISION(
        self, type_: sqltypes.DOUBLE_PRECISION[Any], **kw: Any
    ) -> str:
        return "DOUBLE PRECISION"

    def visit_REAL(self, type_: sqltypes.REAL[Any], **kw: Any) -> str:
        return "REAL"

    def visit_NUMERIC(self, type_: sqltypes.Numeric[Any], **kw: Any) -> str:
        if type_.precision is None:
            return "NUMERIC"
        elif type_.scale is None:
            return "NUMERIC(%(precision)s)" % {"precision": type_.precision}
        else:
            return "NUMERIC(%(precision)s, %(scale)s)" % {
                "precision": type_.precision,
                "scale": type_.scale,
            }

    def visit_DECIMAL(self, type_: sqltypes.DECIMAL[Any], **kw: Any) -> str:
        if type_.precision is None:
            return "DECIMAL"
        elif type_.scale is None:
            return "DECIMAL(%(precision)s)" % {"precision": type_.precision}
        else:
            return "DECIMAL(%(precision)s, %(scale)s)" % {
                "precision": type_.precision,
                "scale": type_.scale,
            }

    def visit_INTEGER(self, type_: sqltypes.Integer, **kw: Any) -> str:
        return "INTEGER"

    def visit_SMALLINT(self, type_: sqltypes.SmallInteger, **kw: Any) -> str:
        return "SMALLINT"

    def visit_BIGINT(self, type_: sqltypes.BigInteger, **kw: Any) -> str:
        return "BIGINT"

    def visit_TIMESTAMP(self, type_: sqltypes.TIMESTAMP, **kw: Any) -> str:
        return "TIMESTAMP"

    def visit_DATETIME(self, type_: sqltypes.DateTime, **kw: Any) -> str:
        return "DATETIME"

    def visit_DATE(self, type_: sqltypes.Date, **kw: Any) -> str:
        return "DATE"

    def visit_TIME(self, type_: sqltypes.Time, **kw: Any) -> str:
        return "TIME"

    def visit_CLOB(self, type_: sqltypes.CLOB, **kw: Any) -> str:
        return "CLOB"

    def visit_NCLOB(self, type_: sqltypes.Text, **kw: Any) -> str:
        return "NCLOB"

    def _render_string_type(
        self, name: str, length: Optional[int], collation: Optional[str]
    ) -> str:
        text = name
        if length:
            text += f"({length})"
        if collation:
            text += f' COLLATE "{collation}"'
        return text

    def visit_CHAR(self, type_: sqltypes.CHAR, **kw: Any) -> str:
        return self._render_string_type("CHAR", type_.length, type_.collation)

    def visit_NCHAR(self, type_: sqltypes.NCHAR, **kw: Any) -> str:
        return self._render_string_type("NCHAR", type_.length, type_.collation)

    def visit_VARCHAR(self, type_: sqltypes.String, **kw: Any) -> str:
        return self._render_string_type(
            "VARCHAR", type_.length, type_.collation
        )

    def visit_NVARCHAR(self, type_: sqltypes.NVARCHAR, **kw: Any) -> str:
        return self._render_string_type(
            "NVARCHAR", type_.length, type_.collation
        )

    def visit_TEXT(self, type_: sqltypes.Text, **kw: Any) -> str:
        return self._render_string_type("TEXT", type_.length, type_.collation)

    def visit_UUID(self, type_: sqltypes.Uuid[Any], **kw: Any) -> str:
        return "UUID"

    def visit_BLOB(self, type_: sqltypes.LargeBinary, **kw: Any) -> str:
        return "BLOB"

    def visit_BINARY(self, type_: sqltypes.BINARY, **kw: Any) -> str:
        return "BINARY" + (type_.length and "(%d)" % type_.length or "")

    def visit_VARBINARY(self, type_: sqltypes.VARBINARY, **kw: Any) -> str:
        return "VARBINARY" + (type_.length and "(%d)" % type_.length or "")

    def visit_BOOLEAN(self, type_: sqltypes.Boolean, **kw: Any) -> str:
        return "BOOLEAN"

    def visit_uuid(self, type_: sqltypes.Uuid[Any], **kw: Any) -> str:
        if not type_.native_uuid or not self.dialect.supports_native_uuid:
            return self._render_string_type("CHAR", length=32, collation=None)
        else:
            return self.visit_UUID(type_, **kw)

    def visit_large_binary(
        self, type_: sqltypes.LargeBinary, **kw: Any
    ) -> str:
        return self.visit_BLOB(type_, **kw)

    def visit_boolean(self, type_: sqltypes.Boolean, **kw: Any) -> str:
        return self.visit_BOOLEAN(type_, **kw)

    def visit_time(self, type_: sqltypes.Time, **kw: Any) -> str:
        return self.visit_TIME(type_, **kw)

    def visit_datetime(self, type_: sqltypes.DateTime, **kw: Any) -> str:
        return self.visit_DATETIME(type_, **kw)

    def visit_date(self, type_: sqltypes.Date, **kw: Any) -> str:
        return self.visit_DATE(type_, **kw)

    def visit_big_integer(self, type_: sqltypes.BigInteger, **kw: Any) -> str:
        return self.visit_BIGINT(type_, **kw)

    def visit_small_integer(
        self, type_: sqltypes.SmallInteger, **kw: Any
    ) -> str:
        return self.visit_SMALLINT(type_, **kw)

    def visit_integer(self, type_: sqltypes.Integer, **kw: Any) -> str:
        return self.visit_INTEGER(type_, **kw)

    def visit_real(self, type_: sqltypes.REAL[Any], **kw: Any) -> str:
        return self.visit_REAL(type_, **kw)

    def visit_float(self, type_: sqltypes.Float[Any], **kw: Any) -> str:
        return self.visit_FLOAT(type_, **kw)

    def visit_double(self, type_: sqltypes.Double[Any], **kw: Any) -> str:
        return self.visit_DOUBLE(type_, **kw)

    def visit_numeric(self, type_: sqltypes.Numeric[Any], **kw: Any) -> str:
        return self.visit_NUMERIC(type_, **kw)

    def visit_string(self, type_: sqltypes.String, **kw: Any) -> str:
        return self.visit_VARCHAR(type_, **kw)

    def visit_unicode(self, type_: sqltypes.Unicode, **kw: Any) -> str:
        return self.visit_VARCHAR(type_, **kw)

    def visit_text(self, type_: sqltypes.Text, **kw: Any) -> str:
        return self.visit_TEXT(type_, **kw)

    def visit_unicode_text(
        self, type_: sqltypes.UnicodeText, **kw: Any
    ) -> str:
        return self.visit_TEXT(type_, **kw)

    def visit_enum(self, type_: sqltypes.Enum, **kw: Any) -> str:
        return self.visit_VARCHAR(type_, **kw)

    def visit_null(self, type_, **kw):
        raise exc.CompileError(
            "Can't generate DDL for %r; "
            "did you forget to specify a "
            "type on this Column?" % type_
        )

    def visit_type_decorator(
        self, type_: TypeDecorator[Any], **kw: Any
    ) -> str:
        return self.process(type_.type_engine(self.dialect), **kw)

    def visit_user_defined(
        self, type_: UserDefinedType[Any], **kw: Any
    ) -> str:
        return type_.get_col_spec(**kw)


class StrSQLTypeCompiler(GenericTypeCompiler):
    def process(self, type_, **kw):
        try:
            _compiler_dispatch = type_._compiler_dispatch
        except AttributeError:
            return self._visit_unknown(type_, **kw)
        else:
            return _compiler_dispatch(self, **kw)

    def __getattr__(self, key):
        if key.startswith("visit_"):
            return self._visit_unknown
        else:
            raise AttributeError(key)

    def _visit_unknown(self, type_, **kw):
        if type_.__class__.__name__ == type_.__class__.__name__.upper():
            return type_.__class__.__name__
        else:
            return repr(type_)

    def visit_null(self, type_, **kw):
        return "NULL"

    def visit_user_defined(self, type_, **kw):
        try:
            get_col_spec = type_.get_col_spec
        except AttributeError:
            return repr(type_)
        else:
            return get_col_spec(**kw)


class _SchemaForObjectCallable(Protocol):
    def __call__(self, __obj: Any) -> str: ...


class _BindNameForColProtocol(Protocol):
    def __call__(self, col: ColumnClause[Any]) -> str: ...


class IdentifierPreparer:
    """Handle quoting and case-folding of identifiers based on options."""

    reserved_words = RESERVED_WORDS

    legal_characters = LEGAL_CHARACTERS

    illegal_initial_characters = ILLEGAL_INITIAL_CHARACTERS

    initial_quote: str

    final_quote: str

    _strings: MutableMapping[str, str]

    schema_for_object: _SchemaForObjectCallable = operator.attrgetter("schema")
    """Return the .schema attribute for an object.

    For the default IdentifierPreparer, the schema for an object is always
    the value of the ".schema" attribute.   if the preparer is replaced
    with one that has a non-empty schema_translate_map, the value of the
    ".schema" attribute is rendered a symbol that will be converted to a
    real schema name from the mapping post-compile.

    """

    _includes_none_schema_translate: bool = False

    def __init__(
        self,
        dialect: Dialect,
        initial_quote: str = '"',
        final_quote: Optional[str] = None,
        escape_quote: str = '"',
        quote_case_sensitive_collations: bool = True,
        omit_schema: bool = False,
    ):
        """Construct a new ``IdentifierPreparer`` object.

        initial_quote
          Character that begins a delimited identifier.

        final_quote
          Character that ends a delimited identifier. Defaults to
          `initial_quote`.

        omit_schema
          Prevent prepending schema name. Useful for databases that do
          not support schemae.
        """

        self.dialect = dialect
        self.initial_quote = initial_quote
        self.final_quote = final_quote or self.initial_quote
        self.escape_quote = escape_quote
        self.escape_to_quote = self.escape_quote * 2
        self.omit_schema = omit_schema
        self.quote_case_sensitive_collations = quote_case_sensitive_collations
        self._strings = {}
        self._double_percents = self.dialect.paramstyle in (
            "format",
            "pyformat",
        )

    def _with_schema_translate(self, schema_translate_map):
        prep = self.__class__.__new__(self.__class__)
        prep.__dict__.update(self.__dict__)

        includes_none = None in schema_translate_map

        def symbol_getter(obj):
            name = obj.schema
            if obj._use_schema_map and (name is not None or includes_none):
                if name is not None and ("[" in name or "]" in name):
                    raise exc.CompileError(
                        "Square bracket characters ([]) not supported "
                        "in schema translate name '%s'" % name
                    )
                return quoted_name(
                    "__[SCHEMA_%s]" % (name or "_none"), quote=False
                )
            else:
                return obj.schema

        prep.schema_for_object = symbol_getter
        prep._includes_none_schema_translate = includes_none
        return prep

    def _render_schema_translates(
        self, statement: str, schema_translate_map: SchemaTranslateMapType
    ) -> str:
        d = schema_translate_map
        if None in d:
            if not self._includes_none_schema_translate:
                raise exc.InvalidRequestError(
                    "schema translate map which previously did not have "
                    "`None` present as a key now has `None` present; compiled "
                    "statement may lack adequate placeholders.  Please use "
                    "consistent keys in successive "
                    "schema_translate_map dictionaries."
                )

            d["_none"] = d[None]  # type: ignore[index]

        def replace(m):
            name = m.group(2)
            if name in d:
                effective_schema = d[name]
            else:
                if name in (None, "_none"):
                    raise exc.InvalidRequestError(
                        "schema translate map which previously had `None` "
                        "present as a key now no longer has it present; don't "
                        "know how to apply schema for compiled statement. "
                        "Please use consistent keys in successive "
                        "schema_translate_map dictionaries."
                    )
                effective_schema = name

            if not effective_schema:
                effective_schema = self.dialect.default_schema_name
                if not effective_schema:
                    # TODO: no coverage here
                    raise exc.CompileError(
                        "Dialect has no default schema name; can't "
                        "use None as dynamic schema target."
                    )
            return self.quote_schema(effective_schema)

        return re.sub(r"(__\[SCHEMA_([^\]]+)\])", replace, statement)

    def _escape_identifier(self, value: str) -> str:
        """Escape an identifier.

        Subclasses should override this to provide database-dependent
        escaping behavior.
        """

        value = value.replace(self.escape_quote, self.escape_to_quote)
        if self._double_percents:
            value = value.replace("%", "%%")
        return value

    def _unescape_identifier(self, value: str) -> str:
        """Canonicalize an escaped identifier.

        Subclasses should override this to provide database-dependent
        unescaping behavior that reverses _escape_identifier.
        """

        return value.replace(self.escape_to_quote, self.escape_quote)

    def validate_sql_phrase(self, element, reg):
        """keyword sequence filter.

        a filter for elements that are intended to represent keyword sequences,
        such as "INITIALLY", "INITIALLY DEFERRED", etc.   no special characters
        should be present.

        .. versionadded:: 1.3

        """

        if element is not None and not reg.match(element):
            raise exc.CompileError(
                "Unexpected SQL phrase: %r (matching against %r)"
                % (element, reg.pattern)
            )
        return element

    def quote_identifier(self, value: str) -> str:
        """Quote an identifier.

        Subclasses should override this to provide database-dependent
        quoting behavior.
        """

        return (
            self.initial_quote
            + self._escape_identifier(value)
            + self.final_quote
        )

    def _requires_quotes(self, value: str) -> bool:
        """Return True if the given identifier requires quoting."""
        lc_value = value.lower()
        return (
            lc_value in self.reserved_words
            or value[0] in self.illegal_initial_characters
            or not self.legal_characters.match(str(value))
            or (lc_value != value)
        )

    def _requires_quotes_illegal_chars(self, value):
        """Return True if the given identifier requires quoting, but
        not taking case convention into account."""
        return not self.legal_characters.match(str(value))

    def quote_schema(self, schema: str, force: Any = None) -> str:
        """Conditionally quote a schema name.


        The name is quoted if it is a reserved word, contains quote-necessary
        characters, or is an instance of :class:`.quoted_name` which includes
        ``quote`` set to ``True``.

        Subclasses can override this to provide database-dependent
        quoting behavior for schema names.

        :param schema: string schema name
        :param force: unused

            .. deprecated:: 0.9

                The :paramref:`.IdentifierPreparer.quote_schema.force`
                parameter is deprecated and will be removed in a future
                release.  This flag has no effect on the behavior of the
                :meth:`.IdentifierPreparer.quote` method; please refer to
                :class:`.quoted_name`.

        """
        if force is not None:
            # not using the util.deprecated_params() decorator in this
            # case because of the additional function call overhead on this
            # very performance-critical spot.
            util.warn_deprecated(
                "The IdentifierPreparer.quote_schema.force parameter is "
                "deprecated and will be removed in a future release.  This "
                "flag has no effect on the behavior of the "
                "IdentifierPreparer.quote method; please refer to "
                "quoted_name().",
                # deprecated 0.9. warning from 1.3
                version="0.9",
            )

        return self.quote(schema)

    def quote(self, ident: str, force: Any = None) -> str:
        """Conditionally quote an identifier.

        The identifier is quoted if it is a reserved word, contains
        quote-necessary characters, or is an instance of
        :class:`.quoted_name` which includes ``quote`` set to ``True``.

        Subclasses can override this to provide database-dependent
        quoting behavior for identifier names.

        :param ident: string identifier
        :param force: unused

            .. deprecated:: 0.9

                The :paramref:`.IdentifierPreparer.quote.force`
                parameter is deprecated and will be removed in a future
                release.  This flag has no effect on the behavior of the
                :meth:`.IdentifierPreparer.quote` method; please refer to
                :class:`.quoted_name`.

        """
        if force is not None:
            # not using the util.deprecated_params() decorator in this
            # case because of the additional function call overhead on this
            # very performance-critical spot.
            util.warn_deprecated(
                "The IdentifierPreparer.quote.force parameter is "
                "deprecated and will be removed in a future release.  This "
                "flag has no effect on the behavior of the "
                "IdentifierPreparer.quote method; please refer to "
                "quoted_name().",
                # deprecated 0.9. warning from 1.3
                version="0.9",
            )

        force = getattr(ident, "quote", None)

        if force is None:
            if ident in self._strings:
                return self._strings[ident]
            else:
                if self._requires_quotes(ident):
                    self._strings[ident] = self.quote_identifier(ident)
                else:
                    self._strings[ident] = ident
                return self._strings[ident]
        elif force:
            return self.quote_identifier(ident)
        else:
            return ident

    def format_collation(self, collation_name):
        if self.quote_case_sensitive_collations:
            return self.quote(collation_name)
        else:
            return collation_name

    def format_sequence(
        self, sequence: schema.Sequence, use_schema: bool = True
    ) -> str:
        name = self.quote(sequence.name)

        effective_schema = self.schema_for_object(sequence)

        if (
            not self.omit_schema
            and use_schema
            and effective_schema is not None
        ):
            name = self.quote_schema(effective_schema) + "." + name
        return name

    def format_label(
        self, label: Label[Any], name: Optional[str] = None
    ) -> str:
        return self.quote(name or label.name)

    def format_alias(
        self, alias: Optional[AliasedReturnsRows], name: Optional[str] = None
    ) -> str:
        if name is None:
            assert alias is not None
            return self.quote(alias.name)
        else:
            return self.quote(name)

    def format_savepoint(self, savepoint, name=None):
        # Running the savepoint name through quoting is unnecessary
        # for all known dialects.  This is here to support potential
        # third party use cases
        ident = name or savepoint.ident
        if self._requires_quotes(ident):
            ident = self.quote_identifier(ident)
        return ident

    @util.preload_module("sqlalchemy.sql.naming")
    def format_constraint(
        self, constraint: Union[Constraint, Index], _alembic_quote: bool = True
    ) -> Optional[str]:
        naming = util.preloaded.sql_naming

        if constraint.name is _NONE_NAME:
            name = naming._constraint_name_for_table(
                constraint, constraint.table
            )

            if name is None:
                return None
        else:
            name = constraint.name

        assert name is not None
        if constraint.__visit_name__ == "index":
            return self.truncate_and_render_index_name(
                name, _alembic_quote=_alembic_quote
            )
        else:
            return self.truncate_and_render_constraint_name(
                name, _alembic_quote=_alembic_quote
            )

    def truncate_and_render_index_name(
        self, name: str, _alembic_quote: bool = True
    ) -> str:
        # calculate these at format time so that ad-hoc changes
        # to dialect.max_identifier_length etc. can be reflected
        # as IdentifierPreparer is long lived
        max_ = (
            self.dialect.max_index_name_length
            or self.dialect.max_identifier_length
        )
        return self._truncate_and_render_maxlen_name(
            name, max_, _alembic_quote
        )

    def truncate_and_render_constraint_name(
        self, name: str, _alembic_quote: bool = True
    ) -> str:
        # calculate these at format time so that ad-hoc changes
        # to dialect.max_identifier_length etc. can be reflected
        # as IdentifierPreparer is long lived
        max_ = (
            self.dialect.max_constraint_name_length
            or self.dialect.max_identifier_length
        )
        return self._truncate_and_render_maxlen_name(
            name, max_, _alembic_quote
        )

    def _truncate_and_render_maxlen_name(
        self, name: str, max_: int, _alembic_quote: bool
    ) -> str:
        if isinstance(name, elements._truncated_label):
            if len(name) > max_:
                name = name[0 : max_ - 8] + "_" + util.md5_hex(name)[-4:]
        else:
            self.dialect.validate_identifier(name)

        if not _alembic_quote:
            return name
        else:
            return self.quote(name)

    def format_index(self, index: Index) -> str:
        name = self.format_constraint(index)
        assert name is not None
        return name

    def format_table(
        self,
        table: FromClause,
        use_schema: bool = True,
        name: Optional[str] = None,
    ) -> str:
        """Prepare a quoted table and schema name."""
        if name is None:
            if TYPE_CHECKING:
                assert isinstance(table, NamedFromClause)
            name = table.name

        result = self.quote(name)

        effective_schema = self.schema_for_object(table)

        if not self.omit_schema and use_schema and effective_schema:
            result = self.quote_schema(effective_schema) + "." + result
        return result

    def format_schema(self, name):
        """Prepare a quoted schema name."""

        return self.quote(name)

    def format_label_name(
        self,
        name,
        anon_map=None,
    ):
        """Prepare a quoted column name."""

        if anon_map is not None and isinstance(
            name, elements._truncated_label
        ):
            name = name.apply_map(anon_map)

        return self.quote(name)

    def format_column(
        self,
        column: ColumnElement[Any],
        use_table: bool = False,
        name: Optional[str] = None,
        table_name: Optional[str] = None,
        use_schema: bool = False,
        anon_map: Optional[Mapping[str, Any]] = None,
    ) -> str:
        """Prepare a quoted column name."""

        if name is None:
            name = column.name
            assert name is not None

        if anon_map is not None and isinstance(
            name, elements._truncated_label
        ):
            name = name.apply_map(anon_map)

        if not getattr(column, "is_literal", False):
            if use_table:
                return (
                    self.format_table(
                        column.table, use_schema=use_schema, name=table_name
                    )
                    + "."
                    + self.quote(name)
                )
            else:
                return self.quote(name)
        else:
            # literal textual elements get stuck into ColumnClause a lot,
            # which shouldn't get quoted

            if use_table:
                return (
                    self.format_table(
                        column.table, use_schema=use_schema, name=table_name
                    )
                    + "."
                    + name
                )
            else:
                return name

    def format_table_seq(self, table, use_schema=True):
        """Format table name and schema as a tuple."""

        # Dialects with more levels in their fully qualified references
        # ('database', 'owner', etc.) could override this and return
        # a longer sequence.

        effective_schema = self.schema_for_object(table)

        if not self.omit_schema and use_schema and effective_schema:
            return (
                self.quote_schema(effective_schema),
                self.format_table(table, use_schema=False),
            )
        else:
            return (self.format_table(table, use_schema=False),)

    @util.memoized_property
    def _r_identifiers(self):
        initial, final, escaped_final = (
            re.escape(s)
            for s in (
                self.initial_quote,
                self.final_quote,
                self._escape_identifier(self.final_quote),
            )
        )
        r = re.compile(
            r"(?:"
            r"(?:%(initial)s((?:%(escaped)s|[^%(final)s])+)%(final)s"
            r"|([^\.]+))(?=\.|$))+"
            % {"initial": initial, "final": final, "escaped": escaped_final}
        )
        return r

    def unformat_identifiers(self, identifiers: str) -> Sequence[str]:
        """Unpack 'schema.table.column'-like strings into components."""

        r = self._r_identifiers
        return [
            self._unescape_identifier(i)
            for i in [a or b for a, b in r.findall(identifiers)]
        ]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\base.py ===
from __future__ import annotations

from collections import abc
from datetime import datetime
import functools
from itertools import zip_longest
import operator
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    NoReturn,
    cast,
    final,
    overload,
)
import warnings

import numpy as np

from pandas._config import (
    get_option,
    using_copy_on_write,
    using_string_dtype,
)

from pandas._libs import (
    NaT,
    algos as libalgos,
    index as libindex,
    lib,
    writers,
)
from pandas._libs.internals import BlockValuesRefs
import pandas._libs.join as libjoin
from pandas._libs.lib import (
    is_datetime_array,
    no_default,
)
from pandas._libs.tslibs import (
    IncompatibleFrequency,
    OutOfBoundsDatetime,
    Timestamp,
    tz_compare,
)
from pandas._typing import (
    AnyAll,
    ArrayLike,
    Axes,
    Axis,
    DropKeep,
    DtypeObj,
    F,
    IgnoreRaise,
    IndexLabel,
    JoinHow,
    Level,
    NaPosition,
    ReindexMethod,
    Self,
    Shape,
    npt,
)
from pandas.compat.numpy import function as nv
from pandas.errors import (
    DuplicateLabelError,
    InvalidIndexError,
)
from pandas.util._decorators import (
    Appender,
    cache_readonly,
    deprecate_nonkeyword_arguments,
    doc,
)
from pandas.util._exceptions import (
    find_stack_level,
    rewrite_exception,
)

from pandas.core.dtypes.astype import (
    astype_array,
    astype_is_view,
)
from pandas.core.dtypes.cast import (
    LossySetitemError,
    can_hold_element,
    common_dtype_categorical_compat,
    find_result_type,
    infer_dtype_from,
    maybe_cast_pointwise_result,
    np_can_hold_element,
)
from pandas.core.dtypes.common import (
    ensure_int64,
    ensure_object,
    ensure_platform_int,
    is_any_real_numeric_dtype,
    is_bool_dtype,
    is_ea_or_datetimelike_dtype,
    is_float,
    is_hashable,
    is_integer,
    is_iterator,
    is_list_like,
    is_numeric_dtype,
    is_object_dtype,
    is_scalar,
    is_signed_integer_dtype,
    is_string_dtype,
    needs_i8_conversion,
    pandas_dtype,
    validate_all_hashable,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import (
    ArrowDtype,
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import (
    ABCCategoricalIndex,
    ABCDataFrame,
    ABCDatetimeIndex,
    ABCIntervalIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
    ABCRangeIndex,
    ABCSeries,
    ABCTimedeltaIndex,
)
from pandas.core.dtypes.inference import is_dict_like
from pandas.core.dtypes.missing import (
    array_equivalent,
    is_valid_na_for_dtype,
    isna,
)

from pandas.core import (
    arraylike,
    nanops,
    ops,
)
from pandas.core.accessor import CachedAccessor
import pandas.core.algorithms as algos
from pandas.core.array_algos.putmask import (
    setitem_datetimelike_compat,
    validate_putmask,
)
from pandas.core.arrays import (
    ArrowExtensionArray,
    BaseMaskedArray,
    Categorical,
    DatetimeArray,
    ExtensionArray,
    TimedeltaArray,
)
from pandas.core.arrays.string_ import (
    StringArray,
    StringDtype,
)
from pandas.core.base import (
    IndexOpsMixin,
    PandasObject,
)
import pandas.core.common as com
from pandas.core.construction import (
    ensure_wrapped_if_datetimelike,
    extract_array,
    sanitize_array,
)
from pandas.core.indexers import (
    disallow_ndim_indexing,
    is_valid_positional_slice,
)
from pandas.core.indexes.frozen import FrozenList
from pandas.core.missing import clean_reindex_fill_method
from pandas.core.ops import get_op_result_name
from pandas.core.ops.invalid import make_invalid_op
from pandas.core.sorting import (
    ensure_key_mapped,
    get_group_index_sorter,
    nargsort,
)
from pandas.core.strings.accessor import StringMethods

from pandas.io.formats.printing import (
    PrettyDict,
    default_pprint,
    format_object_summary,
    pprint_thing,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Sequence,
    )

    from pandas import (
        CategoricalIndex,
        DataFrame,
        MultiIndex,
        Series,
    )
    from pandas.core.arrays import (
        IntervalArray,
        PeriodArray,
    )

__all__ = ["Index"]

_unsortable_types = frozenset(("mixed", "mixed-integer"))

_index_doc_kwargs: dict[str, str] = {
    "klass": "Index",
    "inplace": "",
    "target_klass": "Index",
    "raises_section": "",
    "unique": "Index",
    "duplicated": "np.ndarray",
}
_index_shared_docs: dict[str, str] = {}
str_t = str

_dtype_obj = np.dtype("object")

_masked_engines = {
    "Complex128": libindex.MaskedComplex128Engine,
    "Complex64": libindex.MaskedComplex64Engine,
    "Float64": libindex.MaskedFloat64Engine,
    "Float32": libindex.MaskedFloat32Engine,
    "UInt64": libindex.MaskedUInt64Engine,
    "UInt32": libindex.MaskedUInt32Engine,
    "UInt16": libindex.MaskedUInt16Engine,
    "UInt8": libindex.MaskedUInt8Engine,
    "Int64": libindex.MaskedInt64Engine,
    "Int32": libindex.MaskedInt32Engine,
    "Int16": libindex.MaskedInt16Engine,
    "Int8": libindex.MaskedInt8Engine,
    "boolean": libindex.MaskedBoolEngine,
    "double[pyarrow]": libindex.MaskedFloat64Engine,
    "float64[pyarrow]": libindex.MaskedFloat64Engine,
    "float32[pyarrow]": libindex.MaskedFloat32Engine,
    "float[pyarrow]": libindex.MaskedFloat32Engine,
    "uint64[pyarrow]": libindex.MaskedUInt64Engine,
    "uint32[pyarrow]": libindex.MaskedUInt32Engine,
    "uint16[pyarrow]": libindex.MaskedUInt16Engine,
    "uint8[pyarrow]": libindex.MaskedUInt8Engine,
    "int64[pyarrow]": libindex.MaskedInt64Engine,
    "int32[pyarrow]": libindex.MaskedInt32Engine,
    "int16[pyarrow]": libindex.MaskedInt16Engine,
    "int8[pyarrow]": libindex.MaskedInt8Engine,
    "bool[pyarrow]": libindex.MaskedBoolEngine,
}


def _maybe_return_indexers(meth: F) -> F:
    """
    Decorator to simplify 'return_indexers' checks in Index.join.
    """

    @functools.wraps(meth)
    def join(
        self,
        other: Index,
        *,
        how: JoinHow = "left",
        level=None,
        return_indexers: bool = False,
        sort: bool = False,
    ):
        join_index, lidx, ridx = meth(self, other, how=how, level=level, sort=sort)
        if not return_indexers:
            return join_index

        if lidx is not None:
            lidx = ensure_platform_int(lidx)
        if ridx is not None:
            ridx = ensure_platform_int(ridx)
        return join_index, lidx, ridx

    return cast(F, join)


def _new_Index(cls, d):
    """
    This is called upon unpickling, rather than the default which doesn't
    have arguments and breaks __new__.
    """
    # required for backward compat, because PI can't be instantiated with
    # ordinals through __new__ GH #13277
    if issubclass(cls, ABCPeriodIndex):
        from pandas.core.indexes.period import _new_PeriodIndex

        return _new_PeriodIndex(cls, **d)

    if issubclass(cls, ABCMultiIndex):
        if "labels" in d and "codes" not in d:
            # GH#23752 "labels" kwarg has been replaced with "codes"
            d["codes"] = d.pop("labels")

        # Since this was a valid MultiIndex at pickle-time, we don't need to
        #  check validty at un-pickle time.
        d["verify_integrity"] = False

    elif "dtype" not in d and "data" in d:
        # Prevent Index.__new__ from conducting inference;
        #  "data" key not in RangeIndex
        d["dtype"] = d["data"].dtype
    return cls.__new__(cls, **d)


class Index(IndexOpsMixin, PandasObject):
    """
    Immutable sequence used for indexing and alignment.

    The basic object storing axis labels for all pandas objects.

    .. versionchanged:: 2.0.0

       Index can hold all numpy numeric dtypes (except float16). Previously only
       int64/uint64/float64 dtypes were accepted.

    Parameters
    ----------
    data : array-like (1-dimensional)
    dtype : str, numpy.dtype, or ExtensionDtype, optional
        Data type for the output Index. If not specified, this will be
        inferred from `data`.
        See the :ref:`user guide <basics.dtypes>` for more usages.
    copy : bool, default False
        Copy input data.
    name : object
        Name to be stored in the index.
    tupleize_cols : bool (default: True)
        When True, attempt to create a MultiIndex if possible.

    See Also
    --------
    RangeIndex : Index implementing a monotonic integer range.
    CategoricalIndex : Index of :class:`Categorical` s.
    MultiIndex : A multi-level, or hierarchical Index.
    IntervalIndex : An Index of :class:`Interval` s.
    DatetimeIndex : Index of datetime64 data.
    TimedeltaIndex : Index of timedelta64 data.
    PeriodIndex : Index of Period data.

    Notes
    -----
    An Index instance can **only** contain hashable objects.
    An Index instance *can not* hold numpy float16 dtype.

    Examples
    --------
    >>> pd.Index([1, 2, 3])
    Index([1, 2, 3], dtype='int64')

    >>> pd.Index(list('abc'))
    Index(['a', 'b', 'c'], dtype='object')

    >>> pd.Index([1, 2, 3], dtype="uint8")
    Index([1, 2, 3], dtype='uint8')
    """

    # similar to __array_priority__, positions Index after Series and DataFrame
    #  but before ExtensionArray.  Should NOT be overridden by subclasses.
    __pandas_priority__ = 2000

    # Cython methods; see github.com/cython/cython/issues/2647
    #  for why we need to wrap these instead of making them class attributes
    # Moreover, cython will choose the appropriate-dtyped sub-function
    #  given the dtypes of the passed arguments

    @final
    def _left_indexer_unique(self, other: Self) -> npt.NDArray[np.intp]:
        # Caller is responsible for ensuring other.dtype == self.dtype
        sv = self._get_join_target()
        ov = other._get_join_target()
        # similar but not identical to ov.searchsorted(sv)
        return libjoin.left_join_indexer_unique(sv, ov)

    @final
    def _left_indexer(
        self, other: Self
    ) -> tuple[ArrayLike, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        # Caller is responsible for ensuring other.dtype == self.dtype
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.left_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return joined, lidx, ridx

    @final
    def _inner_indexer(
        self, other: Self
    ) -> tuple[ArrayLike, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        # Caller is responsible for ensuring other.dtype == self.dtype
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.inner_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return joined, lidx, ridx

    @final
    def _outer_indexer(
        self, other: Self
    ) -> tuple[ArrayLike, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        # Caller is responsible for ensuring other.dtype == self.dtype
        sv = self._get_join_target()
        ov = other._get_join_target()
        joined_ndarray, lidx, ridx = libjoin.outer_join_indexer(sv, ov)
        joined = self._from_join_target(joined_ndarray)
        return joined, lidx, ridx

    _typ: str = "index"
    _data: ExtensionArray | np.ndarray
    _data_cls: type[ExtensionArray] | tuple[type[np.ndarray], type[ExtensionArray]] = (
        np.ndarray,
        ExtensionArray,
    )
    _id: object | None = None
    _name: Hashable = None
    # MultiIndex.levels previously allowed setting the index name. We
    # don't allow this anymore, and raise if it happens rather than
    # failing silently.
    _no_setting_name: bool = False
    _comparables: list[str] = ["name"]
    _attributes: list[str] = ["name"]

    @cache_readonly
    def _can_hold_strings(self) -> bool:
        return not is_numeric_dtype(self.dtype)

    _engine_types: dict[np.dtype | ExtensionDtype, type[libindex.IndexEngine]] = {
        np.dtype(np.int8): libindex.Int8Engine,
        np.dtype(np.int16): libindex.Int16Engine,
        np.dtype(np.int32): libindex.Int32Engine,
        np.dtype(np.int64): libindex.Int64Engine,
        np.dtype(np.uint8): libindex.UInt8Engine,
        np.dtype(np.uint16): libindex.UInt16Engine,
        np.dtype(np.uint32): libindex.UInt32Engine,
        np.dtype(np.uint64): libindex.UInt64Engine,
        np.dtype(np.float32): libindex.Float32Engine,
        np.dtype(np.float64): libindex.Float64Engine,
        np.dtype(np.complex64): libindex.Complex64Engine,
        np.dtype(np.complex128): libindex.Complex128Engine,
    }

    @property
    def _engine_type(
        self,
    ) -> type[libindex.IndexEngine | libindex.ExtensionEngine]:
        return self._engine_types.get(self.dtype, libindex.ObjectEngine)

    # whether we support partial string indexing. Overridden
    # in DatetimeIndex and PeriodIndex
    _supports_partial_string_indexing = False

    _accessors = {"str"}

    str = CachedAccessor("str", StringMethods)

    _references = None

    # --------------------------------------------------------------------
    # Constructors

    def __new__(
        cls,
        data=None,
        dtype=None,
        copy: bool = False,
        name=None,
        tupleize_cols: bool = True,
    ) -> Self:
        from pandas.core.indexes.range import RangeIndex

        name = maybe_extract_name(name, data, cls)

        if dtype is not None:
            dtype = pandas_dtype(dtype)

        data_dtype = getattr(data, "dtype", None)

        refs = None
        if not copy and isinstance(data, (ABCSeries, Index)):
            refs = data._references

        is_pandas_object = isinstance(data, (ABCSeries, Index, ExtensionArray))

        # range
        if isinstance(data, (range, RangeIndex)):
            result = RangeIndex(start=data, copy=copy, name=name)
            if dtype is not None:
                return result.astype(dtype, copy=False)
            # error: Incompatible return value type (got "MultiIndex",
            # expected "Self")
            return result  # type: ignore[return-value]

        elif is_ea_or_datetimelike_dtype(dtype):
            # non-EA dtype indexes have special casting logic, so we punt here
            if isinstance(data, (set, frozenset)):
                data = list(data)

        elif is_ea_or_datetimelike_dtype(data_dtype):
            pass

        elif isinstance(data, (np.ndarray, Index, ABCSeries)):
            if isinstance(data, ABCMultiIndex):
                data = data._values

            if data.dtype.kind not in "iufcbmM":
                # GH#11836 we need to avoid having numpy coerce
                # things that look like ints/floats to ints unless
                # they are actually ints, e.g. '0' and 0.0
                # should not be coerced
                data = com.asarray_tuplesafe(data, dtype=_dtype_obj)

        elif is_scalar(data):
            raise cls._raise_scalar_data_error(data)
        elif hasattr(data, "__array__"):
            return cls(np.asarray(data), dtype=dtype, copy=copy, name=name)
        elif not is_list_like(data) and not isinstance(data, memoryview):
            # 2022-11-16 the memoryview check is only necessary on some CI
            #  builds, not clear why
            raise cls._raise_scalar_data_error(data)

        else:
            if tupleize_cols:
                # GH21470: convert iterable to list before determining if empty
                if is_iterator(data):
                    data = list(data)

                if data and all(isinstance(e, tuple) for e in data):
                    # we must be all tuples, otherwise don't construct
                    # 10697
                    from pandas.core.indexes.multi import MultiIndex

                    # error: Incompatible return value type (got "MultiIndex",
                    # expected "Self")
                    return MultiIndex.from_tuples(  # type: ignore[return-value]
                        data, names=name
                    )
            # other iterable of some kind

            if not isinstance(data, (list, tuple)):
                # we allow set/frozenset, which Series/sanitize_array does not, so
                #  cast to list here
                data = list(data)
            if len(data) == 0:
                # unlike Series, we default to object dtype:
                data = np.array(data, dtype=object)

            if len(data) and isinstance(data[0], tuple):
                # Ensure we get 1-D array of tuples instead of 2D array.
                data = com.asarray_tuplesafe(data, dtype=_dtype_obj)

        try:
            arr = sanitize_array(data, None, dtype=dtype, copy=copy)
        except ValueError as err:
            if "index must be specified when data is not list-like" in str(err):
                raise cls._raise_scalar_data_error(data) from err
            if "Data must be 1-dimensional" in str(err):
                raise ValueError("Index data must be 1-dimensional") from err
            raise
        arr = ensure_wrapped_if_datetimelike(arr)

        klass = cls._dtype_to_subclass(arr.dtype)

        arr = klass._ensure_array(arr, arr.dtype, copy=False)
        result = klass._simple_new(arr, name, refs=refs)
        if dtype is None and is_pandas_object and data_dtype == np.object_:
            if result.dtype != data_dtype:
                warnings.warn(
                    "Dtype inference on a pandas object "
                    "(Series, Index, ExtensionArray) is deprecated. The Index "
                    "constructor will keep the original dtype in the future. "
                    "Call `infer_objects` on the result to get the old "
                    "behavior.",
                    FutureWarning,
                    stacklevel=2,
                )
        return result  # type: ignore[return-value]

    @classmethod
    def _ensure_array(cls, data, dtype, copy: bool):
        """
        Ensure we have a valid array to pass to _simple_new.
        """
        if data.ndim > 1:
            # GH#13601, GH#20285, GH#27125
            raise ValueError("Index data must be 1-dimensional")
        elif dtype == np.float16:
            # float16 not supported (no indexing engine)
            raise NotImplementedError("float16 indexes are not supported")

        if copy:
            # asarray_tuplesafe does not always copy underlying data,
            #  so need to make sure that this happens
            data = data.copy()
        return data

    @final
    @classmethod
    def _dtype_to_subclass(cls, dtype: DtypeObj):
        # Delay import for perf. https://github.com/pandas-dev/pandas/pull/31423

        if isinstance(dtype, ExtensionDtype):
            return dtype.index_class

        if dtype.kind == "M":
            from pandas import DatetimeIndex

            return DatetimeIndex

        elif dtype.kind == "m":
            from pandas import TimedeltaIndex

            return TimedeltaIndex

        elif dtype.kind == "O":
            # NB: assuming away MultiIndex
            return Index

        elif issubclass(dtype.type, str) or is_numeric_dtype(dtype):
            return Index

        raise NotImplementedError(dtype)

    # NOTE for new Index creation:

    # - _simple_new: It returns new Index with the same type as the caller.
    #   All metadata (such as name) must be provided by caller's responsibility.
    #   Using _shallow_copy is recommended because it fills these metadata
    #   otherwise specified.

    # - _shallow_copy: It returns new Index with the same type (using
    #   _simple_new), but fills caller's metadata otherwise specified. Passed
    #   kwargs will overwrite corresponding metadata.

    # See each method's docstring.

    @classmethod
    def _simple_new(
        cls, values: ArrayLike, name: Hashable | None = None, refs=None
    ) -> Self:
        """
        We require that we have a dtype compat for the values. If we are passed
        a non-dtype compat, then coerce using the constructor.

        Must be careful not to recurse.
        """
        assert isinstance(values, cls._data_cls), type(values)

        result = object.__new__(cls)
        result._data = values
        result._name = name
        result._cache = {}
        result._reset_identity()
        if refs is not None:
            result._references = refs
        else:
            result._references = BlockValuesRefs()
        result._references.add_index_reference(result)

        return result

    @classmethod
    def _with_infer(cls, *args, **kwargs):
        """
        Constructor that uses the 1.0.x behavior inferring numeric dtypes
        for ndarray[object] inputs.
        """
        result = cls(*args, **kwargs)

        if result.dtype == _dtype_obj and not result._is_multi:
            # error: Argument 1 to "maybe_convert_objects" has incompatible type
            # "Union[ExtensionArray, ndarray[Any, Any]]"; expected
            # "ndarray[Any, Any]"
            values = lib.maybe_convert_objects(result._values)  # type: ignore[arg-type]
            if values.dtype.kind in "iufb":
                return Index(values, name=result.name)

        return result

    @cache_readonly
    def _constructor(self) -> type[Self]:
        return type(self)

    @final
    def _maybe_check_unique(self) -> None:
        """
        Check that an Index has no duplicates.

        This is typically only called via
        `NDFrame.flags.allows_duplicate_labels.setter` when it's set to
        True (duplicates aren't allowed).

        Raises
        ------
        DuplicateLabelError
            When the index is not unique.
        """
        if not self.is_unique:
            msg = """Index has duplicates."""
            duplicates = self._format_duplicate_message()
            msg += f"\n{duplicates}"

            raise DuplicateLabelError(msg)

    @final
    def _format_duplicate_message(self) -> DataFrame:
        """
        Construct the DataFrame for a DuplicateLabelError.

        This returns a DataFrame indicating the labels and positions
        of duplicates in an index. This should only be called when it's
        already known that duplicates are present.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'a'])
        >>> idx._format_duplicate_message()
            positions
        label
        a        [0, 2]
        """
        from pandas import Series

        duplicates = self[self.duplicated(keep="first")].unique()
        assert len(duplicates)

        out = (
            Series(np.arange(len(self)), copy=False)
            .groupby(self, observed=False)
            .agg(list)[duplicates]
        )
        if self._is_multi:
            # test_format_duplicate_labels_message_multi
            # error: "Type[Index]" has no attribute "from_tuples"  [attr-defined]
            out.index = type(self).from_tuples(out.index)  # type: ignore[attr-defined]

        if self.nlevels == 1:
            out = out.rename_axis("label")
        return out.to_frame(name="positions")

    # --------------------------------------------------------------------
    # Index Internals Methods

    def _shallow_copy(self, values, name: Hashable = no_default) -> Self:
        """
        Create a new Index with the same class as the caller, don't copy the
        data, use the same object attributes with passed in attributes taking
        precedence.

        *this is an internal non-public method*

        Parameters
        ----------
        values : the values to create the new Index, optional
        name : Label, defaults to self.name
        """
        name = self._name if name is no_default else name

        return self._simple_new(values, name=name, refs=self._references)

    def _view(self) -> Self:
        """
        fastpath to make a shallow copy, i.e. new object with same data.
        """
        result = self._simple_new(self._values, name=self._name, refs=self._references)

        result._cache = self._cache
        return result

    @final
    def _rename(self, name: Hashable) -> Self:
        """
        fastpath for rename if new name is already validated.
        """
        result = self._view()
        result._name = name
        return result

    @final
    def is_(self, other) -> bool:
        """
        More flexible, faster check like ``is`` but that works through views.

        Note: this is *not* the same as ``Index.identical()``, which checks
        that metadata is also the same.

        Parameters
        ----------
        other : object
            Other object to compare against.

        Returns
        -------
        bool
            True if both have same underlying data, False otherwise.

        See Also
        --------
        Index.identical : Works like ``Index.is_`` but also checks metadata.

        Examples
        --------
        >>> idx1 = pd.Index(['1', '2', '3'])
        >>> idx1.is_(idx1.view())
        True

        >>> idx1.is_(idx1.copy())
        False
        """
        if self is other:
            return True
        elif not hasattr(other, "_id"):
            return False
        elif self._id is None or other._id is None:
            return False
        else:
            return self._id is other._id

    @final
    def _reset_identity(self) -> None:
        """
        Initializes or resets ``_id`` attribute with new object.
        """
        self._id = object()

    @final
    def _cleanup(self) -> None:
        self._engine.clear_mapping()

    @cache_readonly
    def _engine(
        self,
    ) -> libindex.IndexEngine | libindex.ExtensionEngine | libindex.MaskedIndexEngine:
        # For base class (object dtype) we get ObjectEngine
        target_values = self._get_engine_target()

        if isinstance(self._values, ArrowExtensionArray) and self.dtype.kind in "Mm":
            import pyarrow as pa

            pa_type = self._values._pa_array.type
            if pa.types.is_timestamp(pa_type):
                target_values = self._values._to_datetimearray()
                return libindex.DatetimeEngine(target_values._ndarray)
            elif pa.types.is_duration(pa_type):
                target_values = self._values._to_timedeltaarray()
                return libindex.TimedeltaEngine(target_values._ndarray)

        if isinstance(target_values, ExtensionArray):
            if isinstance(target_values, (BaseMaskedArray, ArrowExtensionArray)):
                try:
                    return _masked_engines[target_values.dtype.name](target_values)
                except KeyError:
                    # Not supported yet e.g. decimal
                    pass
            elif self._engine_type is libindex.ObjectEngine:
                return libindex.ExtensionEngine(target_values)

        target_values = cast(np.ndarray, target_values)
        # to avoid a reference cycle, bind `target_values` to a local variable, so
        # `self` is not passed into the lambda.
        if target_values.dtype == bool:
            return libindex.BoolEngine(target_values)
        elif target_values.dtype == np.complex64:
            return libindex.Complex64Engine(target_values)
        elif target_values.dtype == np.complex128:
            return libindex.Complex128Engine(target_values)
        elif needs_i8_conversion(self.dtype):
            # We need to keep M8/m8 dtype when initializing the Engine,
            #  but don't want to change _get_engine_target bc it is used
            #  elsewhere
            # error: Item "ExtensionArray" of "Union[ExtensionArray,
            # ndarray[Any, Any]]" has no attribute "_ndarray"  [union-attr]
            target_values = self._data._ndarray  # type: ignore[union-attr]
        elif is_string_dtype(self.dtype) and not is_object_dtype(self.dtype):
            return libindex.StringObjectEngine(target_values, self.dtype.na_value)  # type: ignore[union-attr]

        # error: Argument 1 to "ExtensionEngine" has incompatible type
        # "ndarray[Any, Any]"; expected "ExtensionArray"
        return self._engine_type(target_values)  # type: ignore[arg-type]

    @final
    @cache_readonly
    def _dir_additions_for_owner(self) -> set[str_t]:
        """
        Add the string-like labels to the owner dataframe/series dir output.

        If this is a MultiIndex, it's first level values are used.
        """
        return {
            c
            for c in self.unique(level=0)[: get_option("display.max_dir_items")]
            if isinstance(c, str) and c.isidentifier()
        }

    # --------------------------------------------------------------------
    # Array-Like Methods

    # ndarray compat
    def __len__(self) -> int:
        """
        Return the length of the Index.
        """
        return len(self._data)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """
        The array interface, return my values.
        """
        if copy is None:
            # Note, that the if branch exists for NumPy 1.x support
            return np.asarray(self._data, dtype=dtype)

        return np.array(self._data, dtype=dtype, copy=copy)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str_t, *inputs, **kwargs):
        if any(isinstance(other, (ABCSeries, ABCDataFrame)) for other in inputs):
            return NotImplemented

        result = arraylike.maybe_dispatch_ufunc_to_dunder_op(
            self, ufunc, method, *inputs, **kwargs
        )
        if result is not NotImplemented:
            return result

        if "out" in kwargs:
            # e.g. test_dti_isub_tdi
            return arraylike.dispatch_ufunc_with_out(
                self, ufunc, method, *inputs, **kwargs
            )

        if method == "reduce":
            result = arraylike.dispatch_reduction_ufunc(
                self, ufunc, method, *inputs, **kwargs
            )
            if result is not NotImplemented:
                return result

        new_inputs = [x if x is not self else x._values for x in inputs]
        result = getattr(ufunc, method)(*new_inputs, **kwargs)
        if ufunc.nout == 2:
            # i.e. np.divmod, np.modf, np.frexp
            return tuple(self.__array_wrap__(x) for x in result)
        elif method == "reduce":
            result = lib.item_from_zerodim(result)
            return result

        if result.dtype == np.float16:
            result = result.astype(np.float32)

        return self.__array_wrap__(result)

    @final
    def __array_wrap__(self, result, context=None, return_scalar=False):
        """
        Gets called after a ufunc and other functions e.g. np.split.
        """
        result = lib.item_from_zerodim(result)
        if (not isinstance(result, Index) and is_bool_dtype(result.dtype)) or np.ndim(
            result
        ) > 1:
            # exclude Index to avoid warning from is_bool_dtype deprecation;
            #  in the Index case it doesn't matter which path we go down.
            # reached in plotting tests with e.g. np.nonzero(index)
            return result

        return Index(result, name=self.name)

    @cache_readonly
    def dtype(self) -> DtypeObj:
        """
        Return the dtype object of the underlying data.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.dtype
        dtype('int64')
        """
        return self._data.dtype

    @final
    def ravel(self, order: str_t = "C") -> Self:
        """
        Return a view on self.

        Returns
        -------
        Index

        See Also
        --------
        numpy.ndarray.ravel : Return a flattened array.

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
        >>> s.index.ravel()
        Index(['a', 'b', 'c'], dtype='object')
        """
        return self[:]

    def view(self, cls=None):
        # we need to see if we are subclassing an
        # index type here
        if cls is not None and not hasattr(cls, "_typ"):
            dtype = cls
            if isinstance(cls, str):
                dtype = pandas_dtype(cls)

            if needs_i8_conversion(dtype):
                idx_cls = self._dtype_to_subclass(dtype)
                arr = self.array.view(dtype)
                if isinstance(arr, ExtensionArray):
                    # here we exclude non-supported dt64/td64 dtypes
                    return idx_cls._simple_new(
                        arr, name=self.name, refs=self._references
                    )
                return arr

            result = self._data.view(cls)
        else:
            if cls is not None:
                warnings.warn(
                    # GH#55709
                    f"Passing a type in {type(self).__name__}.view is deprecated "
                    "and will raise in a future version. "
                    "Call view without any argument to retain the old behavior.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )

            result = self._view()
        if isinstance(result, Index):
            result._id = self._id
        return result

    def astype(self, dtype, copy: bool = True):
        """
        Create an Index with values cast to dtypes.

        The class of a new Index is determined by dtype. When conversion is
        impossible, a TypeError exception is raised.

        Parameters
        ----------
        dtype : numpy dtype or pandas type
            Note that any signed integer `dtype` is treated as ``'int64'``,
            and any unsigned integer `dtype` is treated as ``'uint64'``,
            regardless of the size.
        copy : bool, default True
            By default, astype always returns a newly allocated object.
            If copy is set to False and internal requirements on dtype are
            satisfied, the original data is used to create a new Index
            or the original Index is returned.

        Returns
        -------
        Index
            Index with values cast to specified dtype.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.astype('float')
        Index([1.0, 2.0, 3.0], dtype='float64')
        """
        if dtype is not None:
            dtype = pandas_dtype(dtype)

        if self.dtype == dtype:
            # Ensure that self.astype(self.dtype) is self
            return self.copy() if copy else self

        values = self._data
        if isinstance(values, ExtensionArray):
            with rewrite_exception(type(values).__name__, type(self).__name__):
                new_values = values.astype(dtype, copy=copy)

        elif isinstance(dtype, ExtensionDtype):
            cls = dtype.construct_array_type()
            # Note: for RangeIndex and CategoricalDtype self vs self._values
            #  behaves differently here.
            new_values = cls._from_sequence(self, dtype=dtype, copy=copy)

        else:
            # GH#13149 specifically use astype_array instead of astype
            new_values = astype_array(values, dtype=dtype, copy=copy)

        # pass copy=False because any copying will be done in the astype above
        result = Index(new_values, name=self.name, dtype=new_values.dtype, copy=False)
        if (
            not copy
            and self._references is not None
            and astype_is_view(self.dtype, dtype)
        ):
            result._references = self._references
            result._references.add_index_reference(result)
        return result

    _index_shared_docs[
        "take"
    ] = """
        Return a new %(klass)s of the values selected by the indices.

        For internal compatibility with numpy arrays.

        Parameters
        ----------
        indices : array-like
            Indices to be taken.
        axis : int, optional
            The axis over which to select values, always 0.
        allow_fill : bool, default True
        fill_value : scalar, default None
            If allow_fill=True and fill_value is not None, indices specified by
            -1 are regarded as NA. If Index doesn't hold NA, raise ValueError.

        Returns
        -------
        Index
            An index formed of elements at the given indices. Will be the same
            type as self, except for RangeIndex.

        See Also
        --------
        numpy.ndarray.take: Return an array formed from the
            elements of a at the given indices.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.take([2, 2, 1, 2])
        Index(['c', 'c', 'b', 'c'], dtype='object')
        """

    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Self:
        if kwargs:
            nv.validate_take((), kwargs)
        if is_scalar(indices):
            raise TypeError("Expected indices to be array-like")
        indices = ensure_platform_int(indices)
        allow_fill = self._maybe_disallow_fill(allow_fill, fill_value, indices)

        # Note: we discard fill_value and use self._na_value, only relevant
        #  in the case where allow_fill is True and fill_value is not None
        values = self._values
        if isinstance(values, np.ndarray):
            taken = algos.take(
                values, indices, allow_fill=allow_fill, fill_value=self._na_value
            )
        else:
            # algos.take passes 'axis' keyword which not all EAs accept
            taken = values.take(
                indices, allow_fill=allow_fill, fill_value=self._na_value
            )
        return self._constructor._simple_new(taken, name=self.name)

    @final
    def _maybe_disallow_fill(self, allow_fill: bool, fill_value, indices) -> bool:
        """
        We only use pandas-style take when allow_fill is True _and_
        fill_value is not None.
        """
        if allow_fill and fill_value is not None:
            # only fill if we are passing a non-None fill_value
            if self._can_hold_na:
                if (indices < -1).any():
                    raise ValueError(
                        "When allow_fill=True and fill_value is not None, "
                        "all indices must be >= -1"
                    )
            else:
                cls_name = type(self).__name__
                raise ValueError(
                    f"Unable to fill values because {cls_name} cannot contain NA"
                )
        else:
            allow_fill = False
        return allow_fill

    _index_shared_docs[
        "repeat"
    ] = """
        Repeat elements of a %(klass)s.

        Returns a new %(klass)s where each element of the current %(klass)s
        is repeated consecutively a given number of times.

        Parameters
        ----------
        repeats : int or array of ints
            The number of repetitions for each element. This should be a
            non-negative integer. Repeating 0 times will return an empty
            %(klass)s.
        axis : None
            Must be ``None``. Has no effect but is accepted for compatibility
            with numpy.

        Returns
        -------
        %(klass)s
            Newly created %(klass)s with repeated elements.

        See Also
        --------
        Series.repeat : Equivalent function for Series.
        numpy.repeat : Similar method for :class:`numpy.ndarray`.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')
        >>> idx.repeat(2)
        Index(['a', 'a', 'b', 'b', 'c', 'c'], dtype='object')
        >>> idx.repeat([1, 2, 3])
        Index(['a', 'b', 'b', 'c', 'c', 'c'], dtype='object')
        """

    @Appender(_index_shared_docs["repeat"] % _index_doc_kwargs)
    def repeat(self, repeats, axis: None = None) -> Self:
        repeats = ensure_platform_int(repeats)
        nv.validate_repeat((), {"axis": axis})
        res_values = self._values.repeat(repeats)

        # _constructor so RangeIndex-> Index with an int64 dtype
        return self._constructor._simple_new(res_values, name=self.name)

    # --------------------------------------------------------------------
    # Copying Methods

    def copy(
        self,
        name: Hashable | None = None,
        deep: bool = False,
    ) -> Self:
        """
        Make a copy of this object.

        Name is set on the new object.

        Parameters
        ----------
        name : Label, optional
            Set name for new object.
        deep : bool, default False

        Returns
        -------
        Index
            Index refer to new object which is a copy of this object.

        Notes
        -----
        In most cases, there should be no functional difference from using
        ``deep``, but if ``deep`` is passed it will attempt to deepcopy.

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> new_idx = idx.copy()
        >>> idx is new_idx
        False
        """

        name = self._validate_names(name=name, deep=deep)[0]
        if deep:
            new_data = self._data.copy()
            new_index = type(self)._simple_new(new_data, name=name)
        else:
            new_index = self._rename(name=name)
        return new_index

    @final
    def __copy__(self, **kwargs) -> Self:
        return self.copy(**kwargs)

    @final
    def __deepcopy__(self, memo=None) -> Self:
        """
        Parameters
        ----------
        memo, default None
            Standard signature. Unused
        """
        return self.copy(deep=True)

    # --------------------------------------------------------------------
    # Rendering Methods

    @final
    def __repr__(self) -> str_t:
        """
        Return a string representation for this object.
        """
        klass_name = type(self).__name__
        data = self._format_data()
        attrs = self._format_attrs()
        attrs_str = [f"{k}={v}" for k, v in attrs]
        prepr = ", ".join(attrs_str)

        return f"{klass_name}({data}{prepr})"

    @property
    def _formatter_func(self):
        """
        Return the formatter function.
        """
        return default_pprint

    @final
    def _format_data(self, name=None) -> str_t:
        """
        Return the formatted data as a unicode string.
        """
        # do we want to justify (only do so for non-objects)
        is_justify = True

        if self.inferred_type == "string":
            is_justify = False
        elif isinstance(self.dtype, CategoricalDtype):
            self = cast("CategoricalIndex", self)
            if is_object_dtype(self.categories.dtype):
                is_justify = False
        elif isinstance(self, ABCRangeIndex):
            # We will do the relevant formatting via attrs
            return ""

        return format_object_summary(
            self,
            self._formatter_func,
            is_justify=is_justify,
            name=name,
            line_break_each_value=self._is_multi,
        )

    def _format_attrs(self) -> list[tuple[str_t, str_t | int | bool | None]]:
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs: list[tuple[str_t, str_t | int | bool | None]] = []

        if not self._is_multi:
            attrs.append(("dtype", f"'{self.dtype}'"))

        if self.name is not None:
            attrs.append(("name", default_pprint(self.name)))
        elif self._is_multi and any(x is not None for x in self.names):
            attrs.append(("names", default_pprint(self.names)))

        max_seq_items = get_option("display.max_seq_items") or len(self)
        if len(self) > max_seq_items:
            attrs.append(("length", len(self)))
        return attrs

    @final
    def _get_level_names(self) -> Hashable | Sequence[Hashable]:
        """
        Return a name or list of names with None replaced by the level number.
        """
        if self._is_multi:
            return [
                level if name is None else name for level, name in enumerate(self.names)
            ]
        else:
            return 0 if self.name is None else self.name

    @final
    def _mpl_repr(self) -> np.ndarray:
        # how to represent ourselves to matplotlib
        if isinstance(self.dtype, np.dtype) and self.dtype.kind != "M":
            return cast(np.ndarray, self.values)
        return self.astype(object, copy=False)._values

    def format(
        self,
        name: bool = False,
        formatter: Callable | None = None,
        na_rep: str_t = "NaN",
    ) -> list[str_t]:
        """
        Render a string representation of the Index.
        """
        warnings.warn(
            # GH#55413
            f"{type(self).__name__}.format is deprecated and will be removed "
            "in a future version. Convert using index.astype(str) or "
            "index.map(formatter) instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        header = []
        if name:
            header.append(
                pprint_thing(self.name, escape_chars=("\t", "\r", "\n"))
                if self.name is not None
                else ""
            )

        if formatter is not None:
            return header + list(self.map(formatter))

        return self._format_with_header(header=header, na_rep=na_rep)

    _default_na_rep = "NaN"

    @final
    def _format_flat(
        self,
        *,
        include_name: bool,
        formatter: Callable | None = None,
    ) -> list[str_t]:
        """
        Render a string representation of the Index.
        """
        header = []
        if include_name:
            header.append(
                pprint_thing(self.name, escape_chars=("\t", "\r", "\n"))
                if self.name is not None
                else ""
            )

        if formatter is not None:
            return header + list(self.map(formatter))

        return self._format_with_header(header=header, na_rep=self._default_na_rep)

    def _format_with_header(self, *, header: list[str_t], na_rep: str_t) -> list[str_t]:
        from pandas.io.formats.format import format_array

        values = self._values

        if (
            is_object_dtype(values.dtype)
            or is_string_dtype(values.dtype)
            or isinstance(self.dtype, (IntervalDtype, CategoricalDtype))
        ):
            # TODO: why do we need different justify for these cases?
            justify = "all"
        else:
            justify = "left"
        # passing leading_space=False breaks test_format_missing,
        #  test_index_repr_in_frame_with_nan, but would otherwise make
        #  trim_front unnecessary
        formatted = format_array(values, None, justify=justify)
        result = trim_front(formatted)
        return header + result

    def _get_values_for_csv(
        self,
        *,
        na_rep: str_t = "",
        decimal: str_t = ".",
        float_format=None,
        date_format=None,
        quoting=None,
    ) -> npt.NDArray[np.object_]:
        return get_values_for_csv(
            self._values,
            na_rep=na_rep,
            decimal=decimal,
            float_format=float_format,
            date_format=date_format,
            quoting=quoting,
        )

    def _summary(self, name=None) -> str_t:
        """
        Return a summarized representation.

        Parameters
        ----------
        name : str
            name to use in the summary representation

        Returns
        -------
        String with a summarized representation of the index
        """
        if len(self) > 0:
            head = self[0]
            if hasattr(head, "format") and not isinstance(head, str):
                head = head.format()
            elif needs_i8_conversion(self.dtype):
                # e.g. Timedelta, display as values, not quoted
                head = self._formatter_func(head).replace("'", "")
            tail = self[-1]
            if hasattr(tail, "format") and not isinstance(tail, str):
                tail = tail.format()
            elif needs_i8_conversion(self.dtype):
                # e.g. Timedelta, display as values, not quoted
                tail = self._formatter_func(tail).replace("'", "")

            index_summary = f", {head} to {tail}"
        else:
            index_summary = ""

        if name is None:
            name = type(self).__name__
        return f"{name}: {len(self)} entries{index_summary}"

    # --------------------------------------------------------------------
    # Conversion Methods

    def to_flat_index(self) -> Self:
        """
        Identity method.

        This is implemented for compatibility with subclass implementations
        when chaining.

        Returns
        -------
        pd.Index
            Caller.

        See Also
        --------
        MultiIndex.to_flat_index : Subclass implementation.
        """
        return self

    @final
    def to_series(self, index=None, name: Hashable | None = None) -> Series:
        """
        Create a Series with both index and values equal to the index keys.

        Useful with map for returning an indexer based on an index.

        Parameters
        ----------
        index : Index, optional
            Index of resulting Series. If None, defaults to original index.
        name : str, optional
            Name of resulting Series. If None, defaults to name of original
            index.

        Returns
        -------
        Series
            The dtype will be based on the type of the Index values.

        See Also
        --------
        Index.to_frame : Convert an Index to a DataFrame.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')

        By default, the original index and original name is reused.

        >>> idx.to_series()
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: animal, dtype: object

        To enforce a new index, specify new labels to ``index``:

        >>> idx.to_series(index=[0, 1, 2])
        0     Ant
        1    Bear
        2     Cow
        Name: animal, dtype: object

        To override the name of the resulting column, specify ``name``:

        >>> idx.to_series(name='zoo')
        animal
        Ant      Ant
        Bear    Bear
        Cow      Cow
        Name: zoo, dtype: object
        """
        from pandas import Series

        if index is None:
            index = self._view()
        if name is None:
            name = self.name

        return Series(self._values.copy(), index=index, name=name)

    def to_frame(
        self, index: bool = True, name: Hashable = lib.no_default
    ) -> DataFrame:
        """
        Create a DataFrame with a column containing the Index.

        Parameters
        ----------
        index : bool, default True
            Set the index of the returned DataFrame as the original Index.

        name : object, defaults to index.name
            The passed name should substitute for the index name (if it has
            one).

        Returns
        -------
        DataFrame
            DataFrame containing the original Index data.

        See Also
        --------
        Index.to_series : Convert an Index to a Series.
        Series.to_frame : Convert Series to DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['Ant', 'Bear', 'Cow'], name='animal')
        >>> idx.to_frame()
               animal
        animal
        Ant       Ant
        Bear     Bear
        Cow       Cow

        By default, the original Index is reused. To enforce a new Index:

        >>> idx.to_frame(index=False)
            animal
        0   Ant
        1  Bear
        2   Cow

        To override the name of the resulting column, specify `name`:

        >>> idx.to_frame(index=False, name='zoo')
            zoo
        0   Ant
        1  Bear
        2   Cow
        """
        from pandas import DataFrame

        if name is lib.no_default:
            name = self._get_level_names()
        result = DataFrame({name: self}, copy=not using_copy_on_write())

        if index:
            result.index = self
        return result

    # --------------------------------------------------------------------
    # Name-Centric Methods

    @property
    def name(self) -> Hashable:
        """
        Return Index or MultiIndex name.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3], name='x')
        >>> idx
        Index([1, 2, 3], dtype='int64',  name='x')
        >>> idx.name
        'x'
        """
        return self._name

    @name.setter
    def name(self, value: Hashable) -> None:
        if self._no_setting_name:
            # Used in MultiIndex.levels to avoid silently ignoring name updates.
            raise RuntimeError(
                "Cannot set name on a level of a MultiIndex. Use "
                "'MultiIndex.set_names' instead."
            )
        maybe_extract_name(value, None, type(self))
        self._name = value

    @final
    def _validate_names(
        self, name=None, names=None, deep: bool = False
    ) -> list[Hashable]:
        """
        Handles the quirks of having a singular 'name' parameter for general
        Index and plural 'names' parameter for MultiIndex.
        """
        from copy import deepcopy

        if names is not None and name is not None:
            raise TypeError("Can only provide one of `names` and `name`")
        if names is None and name is None:
            new_names = deepcopy(self.names) if deep else self.names
        elif names is not None:
            if not is_list_like(names):
                raise TypeError("Must pass list-like as `names`.")
            new_names = names
        elif not is_list_like(name):
            new_names = [name]
        else:
            new_names = name

        if len(new_names) != len(self.names):
            raise ValueError(
                f"Length of new names must be {len(self.names)}, got {len(new_names)}"
            )

        # All items in 'new_names' need to be hashable
        validate_all_hashable(*new_names, error_name=f"{type(self).__name__}.name")

        return new_names

    def _get_default_index_names(
        self, names: Hashable | Sequence[Hashable] | None = None, default=None
    ) -> list[Hashable]:
        """
        Get names of index.

        Parameters
        ----------
        names : int, str or 1-dimensional list, default None
            Index names to set.
        default : str
            Default name of index.

        Raises
        ------
        TypeError
            if names not str or list-like
        """
        from pandas.core.indexes.multi import MultiIndex

        if names is not None:
            if isinstance(names, (int, str)):
                names = [names]

        if not isinstance(names, list) and names is not None:
            raise ValueError("Index names must be str or 1-dimensional list")

        if not names:
            if isinstance(self, MultiIndex):
                names = com.fill_missing_names(self.names)
            else:
                names = [default] if self.name is None else [self.name]

        return names

    def _get_names(self) -> FrozenList:
        return FrozenList((self.name,))

    def _set_names(self, values, *, level=None) -> None:
        """
        Set new names on index. Each name has to be a hashable type.

        Parameters
        ----------
        values : str or sequence
            name(s) to set
        level : int, level name, or sequence of int/level names (default None)
            If the index is a MultiIndex (hierarchical), level(s) to set (None
            for all levels).  Otherwise level must be None

        Raises
        ------
        TypeError if each name is not hashable.
        """
        if not is_list_like(values):
            raise ValueError("Names must be a list-like")
        if len(values) != 1:
            raise ValueError(f"Length of new names must be 1, got {len(values)}")

        # GH 20527
        # All items in 'name' need to be hashable:
        validate_all_hashable(*values, error_name=f"{type(self).__name__}.name")

        self._name = values[0]

    names = property(fset=_set_names, fget=_get_names)

    @overload
    def set_names(self, names, *, level=..., inplace: Literal[False] = ...) -> Self:
        ...

    @overload
    def set_names(self, names, *, level=..., inplace: Literal[True]) -> None:
        ...

    @overload
    def set_names(self, names, *, level=..., inplace: bool = ...) -> Self | None:
        ...

    def set_names(self, names, *, level=None, inplace: bool = False) -> Self | None:
        """
        Set Index or MultiIndex name.

        Able to set new names partially and by level.

        Parameters
        ----------

        names : label or list of label or dict-like for MultiIndex
            Name(s) to set.

            .. versionchanged:: 1.3.0

        level : int, label or list of int or label, optional
            If the index is a MultiIndex and names is not dict-like, level(s) to set
            (None for all levels). Otherwise level must be None.

            .. versionchanged:: 1.3.0

        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Index.rename : Able to set new names without level.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')
        >>> idx.set_names('quarter')
        Index([1, 2, 3, 4], dtype='int64', name='quarter')

        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],
        ...                                   [2018, 2019]])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   )
        >>> idx = idx.set_names(['kind', 'year'])
        >>> idx.set_names('species', level=0)
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['species', 'year'])

        When renaming levels with a dict, levels can not be passed.

        >>> idx.set_names({'kind': 'snake'})
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['snake', 'year'])
        """
        if level is not None and not isinstance(self, ABCMultiIndex):
            raise ValueError("Level must be None for non-MultiIndex")

        if level is not None and not is_list_like(level) and is_list_like(names):
            raise TypeError("Names must be a string when a single level is provided.")

        if not is_list_like(names) and level is None and self.nlevels > 1:
            raise TypeError("Must pass list-like as `names`.")

        if is_dict_like(names) and not isinstance(self, ABCMultiIndex):
            raise TypeError("Can only pass dict-like as `names` for MultiIndex.")

        if is_dict_like(names) and level is not None:
            raise TypeError("Can not pass level for dictlike `names`.")

        if isinstance(self, ABCMultiIndex) and is_dict_like(names) and level is None:
            # Transform dict to list of new names and corresponding levels
            level, names_adjusted = [], []
            for i, name in enumerate(self.names):
                if name in names.keys():
                    level.append(i)
                    names_adjusted.append(names[name])
            names = names_adjusted

        if not is_list_like(names):
            names = [names]
        if level is not None and not is_list_like(level):
            level = [level]

        if inplace:
            idx = self
        else:
            idx = self._view()

        idx._set_names(names, level=level)
        if not inplace:
            return idx
        return None

    @overload
    def rename(self, name, *, inplace: Literal[False] = ...) -> Self:
        ...

    @overload
    def rename(self, name, *, inplace: Literal[True]) -> None:
        ...

    @deprecate_nonkeyword_arguments(
        version="3.0", allowed_args=["self", "name"], name="rename"
    )
    def rename(self, name, inplace: bool = False) -> Self | None:
        """
        Alter Index or MultiIndex name.

        Able to set new names without level. Defaults to returning new index.
        Length of names must match number of levels in MultiIndex.

        Parameters
        ----------
        name : label or list of labels
            Name(s) to set.
        inplace : bool, default False
            Modifies the object directly, instead of creating a new Index or
            MultiIndex.

        Returns
        -------
        Index or None
            The same type as the caller or None if ``inplace=True``.

        See Also
        --------
        Index.set_names : Able to set new names partially and by level.

        Examples
        --------
        >>> idx = pd.Index(['A', 'C', 'A', 'B'], name='score')
        >>> idx.rename('grade')
        Index(['A', 'C', 'A', 'B'], dtype='object', name='grade')

        >>> idx = pd.MultiIndex.from_product([['python', 'cobra'],
        ...                                   [2018, 2019]],
        ...                                   names=['kind', 'year'])
        >>> idx
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['kind', 'year'])
        >>> idx.rename(['species', 'year'])
        MultiIndex([('python', 2018),
                    ('python', 2019),
                    ( 'cobra', 2018),
                    ( 'cobra', 2019)],
                   names=['species', 'year'])
        >>> idx.rename('species')
        Traceback (most recent call last):
        TypeError: Must pass list-like as `names`.
        """
        return self.set_names([name], inplace=inplace)

    # --------------------------------------------------------------------
    # Level-Centric Methods

    @property
    def nlevels(self) -> int:
        """
        Number of levels.
        """
        return 1

    def _sort_levels_monotonic(self) -> Self:
        """
        Compat with MultiIndex.
        """
        return self

    @final
    def _validate_index_level(self, level) -> None:
        """
        Validate index level.

        For single-level Index getting level number is a no-op, but some
        verification must be done like in MultiIndex.

        """
        if isinstance(level, int):
            if level < 0 and level != -1:
                raise IndexError(
                    "Too many levels: Index has only 1 level, "
                    f"{level} is not a valid level number"
                )
            if level > 0:
                raise IndexError(
                    f"Too many levels: Index has only 1 level, not {level + 1}"
                )
        elif level != self.name:
            raise KeyError(
                f"Requested level ({level}) does not match index name ({self.name})"
            )

    def _get_level_number(self, level) -> int:
        self._validate_index_level(level)
        return 0

    def sortlevel(
        self,
        level=None,
        ascending: bool | list[bool] = True,
        sort_remaining=None,
        na_position: NaPosition = "first",
    ):
        """
        For internal compatibility with the Index API.

        Sort the Index. This is for compat with MultiIndex

        Parameters
        ----------
        ascending : bool, default True
            False to sort in descending order
        na_position : {'first' or 'last'}, default 'first'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.

            .. versionadded:: 2.1.0

        level, sort_remaining are compat parameters

        Returns
        -------
        Index
        """
        if not isinstance(ascending, (list, bool)):
            raise TypeError(
                "ascending must be a single bool value or"
                "a list of bool values of length 1"
            )

        if isinstance(ascending, list):
            if len(ascending) != 1:
                raise TypeError("ascending must be a list of bool values of length 1")
            ascending = ascending[0]

        if not isinstance(ascending, bool):
            raise TypeError("ascending must be a bool value")

        return self.sort_values(
            return_indexer=True, ascending=ascending, na_position=na_position
        )

    def _get_level_values(self, level) -> Index:
        """
        Return an Index of values for requested level.

        This is primarily useful to get an individual level of values from a
        MultiIndex, but is provided on Index as well for compatibility.

        Parameters
        ----------
        level : int or str
            It is either the integer position or the name of the level.

        Returns
        -------
        Index
            Calling object, as there is only one level in the Index.

        See Also
        --------
        MultiIndex.get_level_values : Get values for a level of a MultiIndex.

        Notes
        -----
        For Index, level should be 0, since there are no multiple levels.

        Examples
        --------
        >>> idx = pd.Index(list('abc'))
        >>> idx
        Index(['a', 'b', 'c'], dtype='object')

        Get level values by supplying `level` as integer:

        >>> idx.get_level_values(0)
        Index(['a', 'b', 'c'], dtype='object')
        """
        self._validate_index_level(level)
        return self

    get_level_values = _get_level_values

    @final
    def droplevel(self, level: IndexLabel = 0):
        """
        Return index with requested level(s) removed.

        If resulting index has only 1 level left, the result will be
        of Index type, not MultiIndex. The original index is not modified inplace.

        Parameters
        ----------
        level : int, str, or list-like, default 0
            If a string is given, must be the name of a level
            If list-like, elements must be names or indexes of levels.

        Returns
        -------
        Index or MultiIndex

        Examples
        --------
        >>> mi = pd.MultiIndex.from_arrays(
        ... [[1, 2], [3, 4], [5, 6]], names=['x', 'y', 'z'])
        >>> mi
        MultiIndex([(1, 3, 5),
                    (2, 4, 6)],
                   names=['x', 'y', 'z'])

        >>> mi.droplevel()
        MultiIndex([(3, 5),
                    (4, 6)],
                   names=['y', 'z'])

        >>> mi.droplevel(2)
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel('z')
        MultiIndex([(1, 3),
                    (2, 4)],
                   names=['x', 'y'])

        >>> mi.droplevel(['x', 'y'])
        Index([5, 6], dtype='int64', name='z')
        """
        if not isinstance(level, (tuple, list)):
            level = [level]

        levnums = sorted(self._get_level_number(lev) for lev in level)[::-1]

        return self._drop_level_numbers(levnums)

    @final
    def _drop_level_numbers(self, levnums: list[int]):
        """
        Drop MultiIndex levels by level _number_, not name.
        """

        if not levnums and not isinstance(self, ABCMultiIndex):
            return self
        if len(levnums) >= self.nlevels:
            raise ValueError(
                f"Cannot remove {len(levnums)} levels from an index with "
                f"{self.nlevels} levels: at least one level must be left."
            )
        # The two checks above guarantee that here self is a MultiIndex
        self = cast("MultiIndex", self)

        new_levels = list(self.levels)
        new_codes = list(self.codes)
        new_names = list(self.names)

        for i in levnums:
            new_levels.pop(i)
            new_codes.pop(i)
            new_names.pop(i)

        if len(new_levels) == 1:
            lev = new_levels[0]

            if len(lev) == 0:
                # If lev is empty, lev.take will fail GH#42055
                if len(new_codes[0]) == 0:
                    # GH#45230 preserve RangeIndex here
                    #  see test_reset_index_empty_rangeindex
                    result = lev[:0]
                else:
                    res_values = algos.take(lev._values, new_codes[0], allow_fill=True)
                    # _constructor instead of type(lev) for RangeIndex compat GH#35230
                    result = lev._constructor._simple_new(res_values, name=new_names[0])
            else:
                # set nan if needed
                mask = new_codes[0] == -1
                result = new_levels[0].take(new_codes[0])
                if mask.any():
                    result = result.putmask(mask, np.nan)

                result._name = new_names[0]

            return result
        else:
            from pandas.core.indexes.multi import MultiIndex

            return MultiIndex(
                levels=new_levels,
                codes=new_codes,
                names=new_names,
                verify_integrity=False,
            )

    # --------------------------------------------------------------------
    # Introspection Methods

    @cache_readonly
    @final
    def _can_hold_na(self) -> bool:
        if isinstance(self.dtype, ExtensionDtype):
            return self.dtype._can_hold_na
        if self.dtype.kind in "iub":
            return False
        return True

    @property
    def is_monotonic_increasing(self) -> bool:
        """
        Return a boolean if the values are equal or increasing.

        Returns
        -------
        bool

        See Also
        --------
        Index.is_monotonic_decreasing : Check if the values are equal or decreasing.

        Examples
        --------
        >>> pd.Index([1, 2, 3]).is_monotonic_increasing
        True
        >>> pd.Index([1, 2, 2]).is_monotonic_increasing
        True
        >>> pd.Index([1, 3, 2]).is_monotonic_increasing
        False
        """
        return self._engine.is_monotonic_increasing

    @property
    def is_monotonic_decreasing(self) -> bool:
        """
        Return a boolean if the values are equal or decreasing.

        Returns
        -------
        bool

        See Also
        --------
        Index.is_monotonic_increasing : Check if the values are equal or increasing.

        Examples
        --------
        >>> pd.Index([3, 2, 1]).is_monotonic_decreasing
        True
        >>> pd.Index([3, 2, 2]).is_monotonic_decreasing
        True
        >>> pd.Index([3, 1, 2]).is_monotonic_decreasing
        False
        """
        return self._engine.is_monotonic_decreasing

    @final
    @property
    def _is_strictly_monotonic_increasing(self) -> bool:
        """
        Return if the index is strictly monotonic increasing
        (only increasing) values.

        Examples
        --------
        >>> Index([1, 2, 3])._is_strictly_monotonic_increasing
        True
        >>> Index([1, 2, 2])._is_strictly_monotonic_increasing
        False
        >>> Index([1, 3, 2])._is_strictly_monotonic_increasing
        False
        """
        return self.is_unique and self.is_monotonic_increasing

    @final
    @property
    def _is_strictly_monotonic_decreasing(self) -> bool:
        """
        Return if the index is strictly monotonic decreasing
        (only decreasing) values.

        Examples
        --------
        >>> Index([3, 2, 1])._is_strictly_monotonic_decreasing
        True
        >>> Index([3, 2, 2])._is_strictly_monotonic_decreasing
        False
        >>> Index([3, 1, 2])._is_strictly_monotonic_decreasing
        False
        """
        return self.is_unique and self.is_monotonic_decreasing

    @cache_readonly
    def is_unique(self) -> bool:
        """
        Return if the index has unique values.

        Returns
        -------
        bool

        See Also
        --------
        Index.has_duplicates : Inverse method that checks if it has duplicate values.

        Examples
        --------
        >>> idx = pd.Index([1, 5, 7, 7])
        >>> idx.is_unique
        False

        >>> idx = pd.Index([1, 5, 7])
        >>> idx.is_unique
        True

        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_unique
        False

        >>> idx = pd.Index(["Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_unique
        True
        """
        return self._engine.is_unique

    @final
    @property
    def has_duplicates(self) -> bool:
        """
        Check if the Index has duplicate values.

        Returns
        -------
        bool
            Whether or not the Index has duplicate values.

        See Also
        --------
        Index.is_unique : Inverse method that checks if it has unique values.

        Examples
        --------
        >>> idx = pd.Index([1, 5, 7, 7])
        >>> idx.has_duplicates
        True

        >>> idx = pd.Index([1, 5, 7])
        >>> idx.has_duplicates
        False

        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.has_duplicates
        True

        >>> idx = pd.Index(["Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.has_duplicates
        False
        """
        return not self.is_unique

    @final
    def is_boolean(self) -> bool:
        """
        Check if the Index only consists of booleans.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.is_bool_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of booleans.

        See Also
        --------
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype (deprecated).
        is_categorical : Check if the Index holds categorical data.
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index([True, False, True])
        >>> idx.is_boolean()  # doctest: +SKIP
        True

        >>> idx = pd.Index(["True", "False", "True"])
        >>> idx.is_boolean()  # doctest: +SKIP
        False

        >>> idx = pd.Index([True, False, "True"])
        >>> idx.is_boolean()  # doctest: +SKIP
        False
        """
        warnings.warn(
            f"{type(self).__name__}.is_boolean is deprecated. "
            "Use pandas.api.types.is_bool_type instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.inferred_type in ["boolean"]

    @final
    def is_integer(self) -> bool:
        """
        Check if the Index only consists of integers.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.is_integer_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of integers.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx.is_integer()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_integer()  # doctest: +SKIP
        False

        >>> idx = pd.Index(["Apple", "Mango", "Watermelon"])
        >>> idx.is_integer()  # doctest: +SKIP
        False
        """
        warnings.warn(
            f"{type(self).__name__}.is_integer is deprecated. "
            "Use pandas.api.types.is_integer_dtype instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.inferred_type in ["integer"]

    @final
    def is_floating(self) -> bool:
        """
        Check if the Index is a floating type.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.is_float_dtype` instead

        The Index may consist of only floats, NaNs, or a mix of floats,
        integers, or NaNs.

        Returns
        -------
        bool
            Whether or not the Index only consists of only consists of floats, NaNs, or
            a mix of floats, integers, or NaNs.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_floating()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1.0, 2.0, np.nan, 4.0])
        >>> idx.is_floating()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4, np.nan])
        >>> idx.is_floating()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx.is_floating()  # doctest: +SKIP
        False
        """
        warnings.warn(
            f"{type(self).__name__}.is_floating is deprecated. "
            "Use pandas.api.types.is_float_dtype instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.inferred_type in ["floating", "mixed-integer-float", "integer-na"]

    @final
    def is_numeric(self) -> bool:
        """
        Check if the Index only consists of numeric data.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.is_numeric_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index only consists of numeric data.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_numeric()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4.0])
        >>> idx.is_numeric()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx.is_numeric()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4.0, np.nan])
        >>> idx.is_numeric()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 2, 3, 4.0, np.nan, "Apple"])
        >>> idx.is_numeric()  # doctest: +SKIP
        False
        """
        warnings.warn(
            f"{type(self).__name__}.is_numeric is deprecated. "
            "Use pandas.api.types.is_any_real_numeric_dtype instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.inferred_type in ["integer", "floating"]

    @final
    def is_object(self) -> bool:
        """
        Check if the Index is of the object dtype.

        .. deprecated:: 2.0.0
           Use `pandas.api.types.is_object_dtype` instead.

        Returns
        -------
        bool
            Whether or not the Index is of the object dtype.

        See Also
        --------
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index(["Apple", "Mango", "Watermelon"])
        >>> idx.is_object()  # doctest: +SKIP
        True

        >>> idx = pd.Index(["Apple", "Mango", 2.0])
        >>> idx.is_object()  # doctest: +SKIP
        True

        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_object()  # doctest: +SKIP
        False

        >>> idx = pd.Index([1.0, 2.0, 3.0, 4.0])
        >>> idx.is_object()  # doctest: +SKIP
        False
        """
        warnings.warn(
            f"{type(self).__name__}.is_object is deprecated."
            "Use pandas.api.types.is_object_dtype instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return is_object_dtype(self.dtype)

    @final
    def is_categorical(self) -> bool:
        """
        Check if the Index holds categorical data.

        .. deprecated:: 2.0.0
              Use `isinstance(index.dtype, pd.CategoricalDtype)` instead.

        Returns
        -------
        bool
            True if the Index is categorical.

        See Also
        --------
        CategoricalIndex : Index for categorical data.
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_interval : Check if the Index holds Interval objects (deprecated).

        Examples
        --------
        >>> idx = pd.Index(["Watermelon", "Orange", "Apple",
        ...                 "Watermelon"]).astype("category")
        >>> idx.is_categorical()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 3, 5, 7])
        >>> idx.is_categorical()  # doctest: +SKIP
        False

        >>> s = pd.Series(["Peter", "Victor", "Elisabeth", "Mar"])
        >>> s
        0        Peter
        1       Victor
        2    Elisabeth
        3          Mar
        dtype: object
        >>> s.index.is_categorical()  # doctest: +SKIP
        False
        """
        warnings.warn(
            f"{type(self).__name__}.is_categorical is deprecated."
            "Use pandas.api.types.is_categorical_dtype instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

        return self.inferred_type in ["categorical"]

    @final
    def is_interval(self) -> bool:
        """
        Check if the Index holds Interval objects.

        .. deprecated:: 2.0.0
            Use `isinstance(index.dtype, pd.IntervalDtype)` instead.

        Returns
        -------
        bool
            Whether or not the Index holds Interval objects.

        See Also
        --------
        IntervalIndex : Index for Interval objects.
        is_boolean : Check if the Index only consists of booleans (deprecated).
        is_integer : Check if the Index only consists of integers (deprecated).
        is_floating : Check if the Index is a floating type (deprecated).
        is_numeric : Check if the Index only consists of numeric data (deprecated).
        is_object : Check if the Index is of the object dtype. (deprecated).
        is_categorical : Check if the Index holds categorical data (deprecated).

        Examples
        --------
        >>> idx = pd.Index([pd.Interval(left=0, right=5),
        ...                 pd.Interval(left=5, right=10)])
        >>> idx.is_interval()  # doctest: +SKIP
        True

        >>> idx = pd.Index([1, 3, 5, 7])
        >>> idx.is_interval()  # doctest: +SKIP
        False
        """
        warnings.warn(
            f"{type(self).__name__}.is_interval is deprecated."
            "Use pandas.api.types.is_interval_dtype instead",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self.inferred_type in ["interval"]

    @final
    def _holds_integer(self) -> bool:
        """
        Whether the type is an integer type.
        """
        return self.inferred_type in ["integer", "mixed-integer"]

    @final
    def holds_integer(self) -> bool:
        """
        Whether the type is an integer type.

        .. deprecated:: 2.0.0
            Use `pandas.api.types.infer_dtype` instead
        """
        warnings.warn(
            f"{type(self).__name__}.holds_integer is deprecated. "
            "Use pandas.api.types.infer_dtype instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        return self._holds_integer()

    @cache_readonly
    def inferred_type(self) -> str_t:
        """
        Return a string of the type inferred from the values.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.inferred_type
        'integer'
        """
        return lib.infer_dtype(self._values, skipna=False)

    @cache_readonly
    @final
    def _is_all_dates(self) -> bool:
        """
        Whether or not the index values only consist of dates.
        """
        if needs_i8_conversion(self.dtype):
            return True
        elif self.dtype != _dtype_obj:
            # TODO(ExtensionIndex): 3rd party EA might override?
            # Note: this includes IntervalIndex, even when the left/right
            #  contain datetime-like objects.
            return False
        elif self._is_multi:
            return False
        return is_datetime_array(ensure_object(self._values))

    @final
    @cache_readonly
    def _is_multi(self) -> bool:
        """
        Cached check equivalent to isinstance(self, MultiIndex)
        """
        return isinstance(self, ABCMultiIndex)

    # --------------------------------------------------------------------
    # Pickle Methods

    def __reduce__(self):
        d = {"data": self._data, "name": self.name}
        return _new_Index, (type(self), d), None

    # --------------------------------------------------------------------
    # Null Handling Methods

    @cache_readonly
    def _na_value(self):
        """The expected NA value to use with this index."""
        dtype = self.dtype
        if isinstance(dtype, np.dtype):
            if dtype.kind in "mM":
                return NaT
            return np.nan
        return dtype.na_value

    @cache_readonly
    def _isnan(self) -> npt.NDArray[np.bool_]:
        """
        Return if each value is NaN.
        """
        if self._can_hold_na:
            return isna(self)
        else:
            # shouldn't reach to this condition by checking hasnans beforehand
            values = np.empty(len(self), dtype=np.bool_)
            values.fill(False)
            return values

    @cache_readonly
    def hasnans(self) -> bool:
        """
        Return True if there are any NaNs.

        Enables various performance speedups.

        Returns
        -------
        bool

        Examples
        --------
        >>> s = pd.Series([1, 2, 3], index=['a', 'b', None])
        >>> s
        a    1
        b    2
        None 3
        dtype: int64
        >>> s.index.hasnans
        True
        """
        if self._can_hold_na:
            return bool(self._isnan.any())
        else:
            return False

    @final
    def isna(self) -> npt.NDArray[np.bool_]:
        """
        Detect missing values.

        Return a boolean same-sized object indicating if the values are NA.
        NA values, such as ``None``, :attr:`numpy.NaN` or :attr:`pd.NaT`, get
        mapped to ``True`` values.
        Everything else get mapped to ``False`` values. Characters such as
        empty strings `''` or :attr:`numpy.inf` are not considered NA values.

        Returns
        -------
        numpy.ndarray[bool]
            A boolean array of whether my values are NA.

        See Also
        --------
        Index.notna : Boolean inverse of isna.
        Index.dropna : Omit entries with missing values.
        isna : Top-level isna.
        Series.isna : Detect missing values in Series object.

        Examples
        --------
        Show which entries in a pandas.Index are NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.nan])
        >>> idx
        Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.isna()
        array([False, False,  True])

        Empty strings are not considered NA values. None is considered an NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.isna()
        array([False, False, False,  True])

        For datetimes, `NaT` (Not a Time) is considered as an NA value.

        >>> idx = pd.DatetimeIndex([pd.Timestamp('1940-04-25'),
        ...                         pd.Timestamp(''), None, pd.NaT])
        >>> idx
        DatetimeIndex(['1940-04-25', 'NaT', 'NaT', 'NaT'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.isna()
        array([False,  True,  True,  True])
        """
        return self._isnan

    isnull = isna

    @final
    def notna(self) -> npt.NDArray[np.bool_]:
        """
        Detect existing (non-missing) values.

        Return a boolean same-sized object indicating if the values are not NA.
        Non-missing values get mapped to ``True``. Characters such as empty
        strings ``''`` or :attr:`numpy.inf` are not considered NA values.
        NA values, such as None or :attr:`numpy.NaN`, get mapped to ``False``
        values.

        Returns
        -------
        numpy.ndarray[bool]
            Boolean array to indicate which entries are not NA.

        See Also
        --------
        Index.notnull : Alias of notna.
        Index.isna: Inverse of notna.
        notna : Top-level notna.

        Examples
        --------
        Show which entries in an Index are not NA. The result is an
        array.

        >>> idx = pd.Index([5.2, 6.0, np.nan])
        >>> idx
        Index([5.2, 6.0, nan], dtype='float64')
        >>> idx.notna()
        array([ True,  True, False])

        Empty strings are not considered NA values. None is considered a NA
        value.

        >>> idx = pd.Index(['black', '', 'red', None])
        >>> idx
        Index(['black', '', 'red', None], dtype='object')
        >>> idx.notna()
        array([ True,  True,  True, False])
        """
        return ~self.isna()

    notnull = notna

    def fillna(self, value=None, downcast=lib.no_default):
        """
        Fill NA/NaN values with the specified value.

        Parameters
        ----------
        value : scalar
            Scalar value to use to fill holes (e.g. 0).
            This value cannot be a list-likes.
        downcast : dict, default is None
            A dict of item->dtype of what to downcast if possible,
            or the string 'infer' which will try to downcast to an appropriate
            equal type (e.g. float64 to int64 if possible).

            .. deprecated:: 2.1.0

        Returns
        -------
        Index

        See Also
        --------
        DataFrame.fillna : Fill NaN values of a DataFrame.
        Series.fillna : Fill NaN Values of a Series.

        Examples
        --------
        >>> idx = pd.Index([np.nan, np.nan, 3])
        >>> idx.fillna(0)
        Index([0.0, 0.0, 3.0], dtype='float64')
        """
        if not is_scalar(value):
            raise TypeError(f"'value' must be a scalar, passed: {type(value).__name__}")
        if downcast is not lib.no_default:
            warnings.warn(
                f"The 'downcast' keyword in {type(self).__name__}.fillna is "
                "deprecated and will be removed in a future version. "
                "It was previously silently ignored.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        else:
            downcast = None

        if self.hasnans:
            result = self.putmask(self._isnan, value)
            if downcast is None:
                # no need to care metadata other than name
                # because it can't have freq if it has NaTs
                # _with_infer needed for test_fillna_categorical
                return Index._with_infer(result, name=self.name)
            raise NotImplementedError(
                f"{type(self).__name__}.fillna does not support 'downcast' "
                "argument values other than 'None'."
            )
        return self._view()

    def dropna(self, how: AnyAll = "any") -> Self:
        """
        Return Index without NA/NaN values.

        Parameters
        ----------
        how : {'any', 'all'}, default 'any'
            If the Index is a MultiIndex, drop the value when any or all levels
            are NaN.

        Returns
        -------
        Index

        Examples
        --------
        >>> idx = pd.Index([1, np.nan, 3])
        >>> idx.dropna()
        Index([1.0, 3.0], dtype='float64')
        """
        if how not in ("any", "all"):
            raise ValueError(f"invalid how option: {how}")

        if self.hasnans:
            res_values = self._values[~self._isnan]
            return type(self)._simple_new(res_values, name=self.name)
        return self._view()

    # --------------------------------------------------------------------
    # Uniqueness Methods

    def unique(self, level: Hashable | None = None) -> Self:
        """
        Return unique values in the index.

        Unique values are returned in order of appearance, this does NOT sort.

        Parameters
        ----------
        level : int or hashable, optional
            Only return values from specified level (for MultiIndex).
            If int, gets the level by integer position, else by level name.

        Returns
        -------
        Index

        See Also
        --------
        unique : Numpy array of unique values in that column.
        Series.unique : Return unique values of Series object.

        Examples
        --------
        >>> idx = pd.Index([1, 1, 2, 3, 3])
        >>> idx.unique()
        Index([1, 2, 3], dtype='int64')
        """
        if level is not None:
            self._validate_index_level(level)

        if self.is_unique:
            return self._view()

        result = super().unique()
        return self._shallow_copy(result)

    def drop_duplicates(self, *, keep: DropKeep = "first") -> Self:
        """
        Return Index with duplicate values removed.

        Parameters
        ----------
        keep : {'first', 'last', ``False``}, default 'first'
            - 'first' : Drop duplicates except for the first occurrence.
            - 'last' : Drop duplicates except for the last occurrence.
            - ``False`` : Drop all duplicates.

        Returns
        -------
        Index

        See Also
        --------
        Series.drop_duplicates : Equivalent method on Series.
        DataFrame.drop_duplicates : Equivalent method on DataFrame.
        Index.duplicated : Related method on Index, indicating duplicate
            Index values.

        Examples
        --------
        Generate an pandas.Index with duplicate values.

        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama', 'hippo'])

        The `keep` parameter controls  which duplicate values are removed.
        The value 'first' keeps the first occurrence for each
        set of duplicated entries. The default value of keep is 'first'.

        >>> idx.drop_duplicates(keep='first')
        Index(['lama', 'cow', 'beetle', 'hippo'], dtype='object')

        The value 'last' keeps the last occurrence for each set of duplicated
        entries.

        >>> idx.drop_duplicates(keep='last')
        Index(['cow', 'beetle', 'lama', 'hippo'], dtype='object')

        The value ``False`` discards all sets of duplicated entries.

        >>> idx.drop_duplicates(keep=False)
        Index(['cow', 'beetle', 'hippo'], dtype='object')
        """
        if self.is_unique:
            return self._view()

        return super().drop_duplicates(keep=keep)

    def duplicated(self, keep: DropKeep = "first") -> npt.NDArray[np.bool_]:
        """
        Indicate duplicate index values.

        Duplicated values are indicated as ``True`` values in the resulting
        array. Either all duplicates, all except the first, or all except the
        last occurrence of duplicates can be indicated.

        Parameters
        ----------
        keep : {'first', 'last', False}, default 'first'
            The value or values in a set of duplicates to mark as missing.

            - 'first' : Mark duplicates as ``True`` except for the first
              occurrence.
            - 'last' : Mark duplicates as ``True`` except for the last
              occurrence.
            - ``False`` : Mark all duplicates as ``True``.

        Returns
        -------
        np.ndarray[bool]

        See Also
        --------
        Series.duplicated : Equivalent method on pandas.Series.
        DataFrame.duplicated : Equivalent method on pandas.DataFrame.
        Index.drop_duplicates : Remove duplicate values from Index.

        Examples
        --------
        By default, for each set of duplicated values, the first occurrence is
        set to False and all others to True:

        >>> idx = pd.Index(['lama', 'cow', 'lama', 'beetle', 'lama'])
        >>> idx.duplicated()
        array([False, False,  True, False,  True])

        which is equivalent to

        >>> idx.duplicated(keep='first')
        array([False, False,  True, False,  True])

        By using 'last', the last occurrence of each set of duplicated values
        is set on False and all others on True:

        >>> idx.duplicated(keep='last')
        array([ True, False,  True, False, False])

        By setting keep on ``False``, all duplicates are True:

        >>> idx.duplicated(keep=False)
        array([ True, False,  True, False,  True])
        """
        if self.is_unique:
            # fastpath available bc we are immutable
            return np.zeros(len(self), dtype=bool)
        return self._duplicated(keep=keep)

    # --------------------------------------------------------------------
    # Arithmetic & Logical Methods

    def __iadd__(self, other):
        # alias for __add__
        return self + other

    @final
    def __nonzero__(self) -> NoReturn:
        raise ValueError(
            f"The truth value of a {type(self).__name__} is ambiguous. "
            "Use a.empty, a.bool(), a.item(), a.any() or a.all()."
        )

    __bool__ = __nonzero__

    # --------------------------------------------------------------------
    # Set Operation Methods

    def _get_reconciled_name_object(self, other):
        """
        If the result of a set operation will be self,
        return self, unless the name changes, in which
        case make a shallow copy of self.
        """
        name = get_op_result_name(self, other)
        if self.name is not name:
            return self.rename(name)
        return self

    @final
    def _validate_sort_keyword(self, sort):
        if sort not in [None, False, True]:
            raise ValueError(
                "The 'sort' keyword only takes the values of "
                f"None, True, or False; {sort} was passed."
            )

    @final
    def _dti_setop_align_tzs(self, other: Index, setop: str_t) -> tuple[Index, Index]:
        """
        With mismatched timezones, cast both to UTC.
        """
        # Caller is responsibelf or checking
        #  `self.dtype != other.dtype`
        if (
            isinstance(self, ABCDatetimeIndex)
            and isinstance(other, ABCDatetimeIndex)
            and self.tz is not None
            and other.tz is not None
        ):
            # GH#39328, GH#45357
            left = self.tz_convert("UTC")
            right = other.tz_convert("UTC")
            return left, right
        return self, other

    @final
    def union(self, other, sort=None):
        """
        Form the union of two Index objects.

        If the Index objects are incompatible, both Index objects will be
        cast to dtype('object') first.

        Parameters
        ----------
        other : Index or array-like
        sort : bool or None, default None
            Whether to sort the resulting Index.

            * None : Sort the result, except when

              1. `self` and `other` are equal.
              2. `self` or `other` has length 0.
              3. Some values in `self` or `other` cannot be compared.
                 A RuntimeWarning is issued in this case.

            * False : do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index

        Examples
        --------
        Union matching dtypes

        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.union(idx2)
        Index([1, 2, 3, 4, 5, 6], dtype='int64')

        Union mismatched dtypes

        >>> idx1 = pd.Index(['a', 'b', 'c', 'd'])
        >>> idx2 = pd.Index([1, 2, 3, 4])
        >>> idx1.union(idx2)
        Index(['a', 'b', 'c', 'd', 1, 2, 3, 4], dtype='object')

        MultiIndex case

        >>> idx1 = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], ["Red", "Blue", "Red", "Blue"]]
        ... )
        >>> idx1
        MultiIndex([(1,  'Red'),
            (1, 'Blue'),
            (2,  'Red'),
            (2, 'Blue')],
           )
        >>> idx2 = pd.MultiIndex.from_arrays(
        ...     [[3, 3, 2, 2], ["Red", "Green", "Red", "Green"]]
        ... )
        >>> idx2
        MultiIndex([(3,   'Red'),
            (3, 'Green'),
            (2,   'Red'),
            (2, 'Green')],
           )
        >>> idx1.union(idx2)
        MultiIndex([(1,  'Blue'),
            (1,   'Red'),
            (2,  'Blue'),
            (2, 'Green'),
            (2,   'Red'),
            (3, 'Green'),
            (3,   'Red')],
           )
        >>> idx1.union(idx2, sort=False)
        MultiIndex([(1,   'Red'),
            (1,  'Blue'),
            (2,   'Red'),
            (2,  'Blue'),
            (3,   'Red'),
            (3, 'Green'),
            (2, 'Green')],
           )
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)

        if self.dtype != other.dtype:
            if (
                isinstance(self, ABCMultiIndex)
                and not is_object_dtype(_unpack_nested_dtype(other))
                and len(other) > 0
            ):
                raise NotImplementedError(
                    "Can only union MultiIndex with MultiIndex or Index of tuples, "
                    "try mi.to_flat_index().union(other) instead."
                )
            self, other = self._dti_setop_align_tzs(other, "union")

            dtype = self._find_common_type_compat(other)
            left = self.astype(dtype, copy=False)
            right = other.astype(dtype, copy=False)
            return left.union(right, sort=sort)

        elif not len(other) or self.equals(other):
            # NB: whether this (and the `if not len(self)` check below) come before
            #  or after the dtype equality check above affects the returned dtype
            result = self._get_reconciled_name_object(other)
            if sort is True:
                return result.sort_values()
            return result

        elif not len(self):
            result = other._get_reconciled_name_object(self)
            if sort is True:
                return result.sort_values()
            return result

        result = self._union(other, sort=sort)

        return self._wrap_setop_result(other, result)

    def _union(self, other: Index, sort: bool | None):
        """
        Specific union logic should go here. In subclasses, union behavior
        should be overwritten here rather than in `self.union`.

        Parameters
        ----------
        other : Index or array-like
        sort : False or None, default False
            Whether to sort the resulting index.

            * True : sort the result
            * False : do not sort the result.
            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.

        Returns
        -------
        Index
        """
        lvals = self._values
        rvals = other._values

        if (
            sort in (None, True)
            and self.is_monotonic_increasing
            and other.is_monotonic_increasing
            and not (self.has_duplicates and other.has_duplicates)
            and self._can_use_libjoin
            and other._can_use_libjoin
        ):
            # Both are monotonic and at least one is unique, so can use outer join
            #  (actually don't need either unique, but without this restriction
            #  test_union_same_value_duplicated_in_both fails)
            try:
                return self._outer_indexer(other)[0]
            except (TypeError, IncompatibleFrequency):
                # incomparable objects; should only be for object dtype
                value_list = list(lvals)

                # worth making this faster? a very unusual case
                value_set = set(lvals)
                value_list.extend([x for x in rvals if x not in value_set])
                # If objects are unorderable, we must have object dtype.
                return np.array(value_list, dtype=object)

        elif not other.is_unique:
            # other has duplicates
            result_dups = algos.union_with_duplicates(self, other)
            return _maybe_try_sort(result_dups, sort)

        # The rest of this method is analogous to Index._intersection_via_get_indexer

        # Self may have duplicates; other already checked as unique
        # find indexes of things in "other" that are not in "self"
        if self._index_as_unique:
            indexer = self.get_indexer(other)
            missing = (indexer == -1).nonzero()[0]
        else:
            missing = algos.unique1d(self.get_indexer_non_unique(other)[1])

        result: Index | MultiIndex | ArrayLike
        if self._is_multi:
            # Preserve MultiIndex to avoid losing dtypes
            result = self.append(other.take(missing))

        else:
            if len(missing) > 0:
                other_diff = rvals.take(missing)
                result = concat_compat((lvals, other_diff))
            else:
                result = lvals

        if not self.is_monotonic_increasing or not other.is_monotonic_increasing:
            # if both are monotonic then result should already be sorted
            result = _maybe_try_sort(result, sort)

        return result

    @final
    def _wrap_setop_result(self, other: Index, result) -> Index:
        name = get_op_result_name(self, other)
        if isinstance(result, Index):
            if result.name != name:
                result = result.rename(name)
        else:
            result = self._shallow_copy(result, name=name)
        return result

    @final
    def intersection(self, other, sort: bool = False):
        # default sort keyword is different here from other setops intentionally
        #  done in GH#25063
        """
        Form the intersection of two Index objects.

        This returns a new Index with elements common to the index and `other`.

        Parameters
        ----------
        other : Index or array-like
        sort : True, False or None, default False
            Whether to sort the resulting index.

            * None : sort the result, except when `self` and `other` are equal
              or when the values cannot be compared.
            * False : do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.intersection(idx2)
        Index([3, 4], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)

        if self.dtype != other.dtype:
            self, other = self._dti_setop_align_tzs(other, "intersection")

        if self.equals(other):
            if not self.is_unique:
                result = self.unique()._get_reconciled_name_object(other)
            else:
                result = self._get_reconciled_name_object(other)
            if sort is True:
                result = result.sort_values()
            return result

        if len(self) == 0 or len(other) == 0:
            # fastpath; we need to be careful about having commutativity

            if self._is_multi or other._is_multi:
                # _convert_can_do_setop ensures that we have both or neither
                # We retain self.levels
                return self[:0].rename(result_name)

            dtype = self._find_common_type_compat(other)
            if self.dtype == dtype:
                # Slicing allows us to retain DTI/TDI.freq, RangeIndex

                # Note: self[:0] vs other[:0] affects
                #  1) which index's `freq` we get in DTI/TDI cases
                #     This may be a historical artifact, i.e. no documented
                #     reason for this choice.
                #  2) The `step` we get in RangeIndex cases
                if len(self) == 0:
                    return self[:0].rename(result_name)
                else:
                    return other[:0].rename(result_name)

            return Index([], dtype=dtype, name=result_name)

        elif not self._should_compare(other):
            # We can infer that the intersection is empty.
            if isinstance(self, ABCMultiIndex):
                return self[:0].rename(result_name)
            return Index([], name=result_name)

        elif self.dtype != other.dtype:
            dtype = self._find_common_type_compat(other)
            this = self.astype(dtype, copy=False)
            other = other.astype(dtype, copy=False)
            return this.intersection(other, sort=sort)

        result = self._intersection(other, sort=sort)
        return self._wrap_intersection_result(other, result)

    def _intersection(self, other: Index, sort: bool = False):
        """
        intersection specialized to the case with matching dtypes.
        """
        if (
            self.is_monotonic_increasing
            and other.is_monotonic_increasing
            and self._can_use_libjoin
            and other._can_use_libjoin
        ):
            try:
                res_indexer, indexer, _ = self._inner_indexer(other)
            except TypeError:
                # non-comparable; should only be for object dtype
                pass
            else:
                # TODO: algos.unique1d should preserve DTA/TDA
                if is_numeric_dtype(self.dtype):
                    # This is faster, because Index.unique() checks for uniqueness
                    # before calculating the unique values.
                    res = algos.unique1d(res_indexer)
                else:
                    result = self.take(indexer)
                    res = result.drop_duplicates()
                return ensure_wrapped_if_datetimelike(res)

        res_values = self._intersection_via_get_indexer(other, sort=sort)
        res_values = _maybe_try_sort(res_values, sort)
        return res_values

    def _wrap_intersection_result(self, other, result):
        # We will override for MultiIndex to handle empty results
        return self._wrap_setop_result(other, result)

    @final
    def _intersection_via_get_indexer(
        self, other: Index | MultiIndex, sort
    ) -> ArrayLike | MultiIndex:
        """
        Find the intersection of two Indexes using get_indexer.

        Returns
        -------
        np.ndarray or ExtensionArray or MultiIndex
            The returned array will be unique.
        """
        left_unique = self.unique()
        right_unique = other.unique()

        # even though we are unique, we need get_indexer_for for IntervalIndex
        indexer = left_unique.get_indexer_for(right_unique)

        mask = indexer != -1

        taker = indexer.take(mask.nonzero()[0])
        if sort is False:
            # sort bc we want the elements in the same order they are in self
            # unnecessary in the case with sort=None bc we will sort later
            taker = np.sort(taker)

        result: MultiIndex | ExtensionArray | np.ndarray
        if isinstance(left_unique, ABCMultiIndex):
            result = left_unique.take(taker)
        else:
            result = left_unique.take(taker)._values
        return result

    @final
    def difference(self, other, sort=None):
        """
        Return a new Index with elements of index not in `other`.

        This is the set difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        sort : bool or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index

        Examples
        --------
        >>> idx1 = pd.Index([2, 1, 3, 4])
        >>> idx2 = pd.Index([3, 4, 5, 6])
        >>> idx1.difference(idx2)
        Index([1, 2], dtype='int64')
        >>> idx1.difference(idx2, sort=False)
        Index([2, 1], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name = self._convert_can_do_setop(other)

        # Note: we do NOT call _dti_setop_align_tzs here, as there
        #  is no requirement that .difference be commutative, so it does
        #  not cast to object.

        if self.equals(other):
            # Note: we do not (yet) sort even if sort=None GH#24959
            return self[:0].rename(result_name)

        if len(other) == 0:
            # Note: we do not (yet) sort even if sort=None GH#24959
            result = self.unique().rename(result_name)
            if sort is True:
                return result.sort_values()
            return result

        if not self._should_compare(other):
            # Nothing matches -> difference is everything
            result = self.unique().rename(result_name)
            if sort is True:
                return result.sort_values()
            return result

        result = self._difference(other, sort=sort)
        return self._wrap_difference_result(other, result)

    def _difference(self, other, sort):
        # overridden by RangeIndex
        this = self
        if isinstance(self, ABCCategoricalIndex) and self.hasnans and other.hasnans:
            this = this.dropna()
        other = other.unique()
        the_diff = this[other.get_indexer_for(this) == -1]
        the_diff = the_diff if this.is_unique else the_diff.unique()
        the_diff = _maybe_try_sort(the_diff, sort)
        return the_diff

    def _wrap_difference_result(self, other, result):
        # We will override for MultiIndex to handle empty results
        return self._wrap_setop_result(other, result)

    def symmetric_difference(self, other, result_name=None, sort=None):
        """
        Compute the symmetric difference of two Index objects.

        Parameters
        ----------
        other : Index or array-like
        result_name : str
        sort : bool or None, default None
            Whether to sort the resulting index. By default, the
            values are attempted to be sorted, but any TypeError from
            incomparable elements is caught by pandas.

            * None : Attempt to sort the result, but catch any TypeErrors
              from comparing incomparable elements.
            * False : Do not sort the result.
            * True : Sort the result (which may raise TypeError).

        Returns
        -------
        Index

        Notes
        -----
        ``symmetric_difference`` contains elements that appear in either
        ``idx1`` or ``idx2`` but not both. Equivalent to the Index created by
        ``idx1.difference(idx2) | idx2.difference(idx1)`` with duplicates
        dropped.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3, 4])
        >>> idx2 = pd.Index([2, 3, 4, 5])
        >>> idx1.symmetric_difference(idx2)
        Index([1, 5], dtype='int64')
        """
        self._validate_sort_keyword(sort)
        self._assert_can_do_setop(other)
        other, result_name_update = self._convert_can_do_setop(other)
        if result_name is None:
            result_name = result_name_update

        if self.dtype != other.dtype:
            self, other = self._dti_setop_align_tzs(other, "symmetric_difference")

        if not self._should_compare(other):
            return self.union(other, sort=sort).rename(result_name)

        elif self.dtype != other.dtype:
            dtype = self._find_common_type_compat(other)
            this = self.astype(dtype, copy=False)
            that = other.astype(dtype, copy=False)
            return this.symmetric_difference(that, sort=sort).rename(result_name)

        this = self.unique()
        other = other.unique()
        indexer = this.get_indexer_for(other)

        # {this} minus {other}
        common_indexer = indexer.take((indexer != -1).nonzero()[0])
        left_indexer = np.setdiff1d(
            np.arange(this.size), common_indexer, assume_unique=True
        )
        left_diff = this.take(left_indexer)

        # {other} minus {this}
        right_indexer = (indexer == -1).nonzero()[0]
        right_diff = other.take(right_indexer)

        res_values = left_diff.append(right_diff)
        result = _maybe_try_sort(res_values, sort)

        if not self._is_multi:
            return Index(result, name=result_name, dtype=res_values.dtype)
        else:
            left_diff = cast("MultiIndex", left_diff)
            if len(result) == 0:
                # result might be an Index, if other was an Index
                return left_diff.remove_unused_levels().set_names(result_name)
            return result.set_names(result_name)

    @final
    def _assert_can_do_setop(self, other) -> bool:
        if not is_list_like(other):
            raise TypeError("Input must be Index or array-like")
        return True

    def _convert_can_do_setop(self, other) -> tuple[Index, Hashable]:
        if not isinstance(other, Index):
            other = Index(other, name=self.name)
            result_name = self.name
        else:
            result_name = get_op_result_name(self, other)
        return other, result_name

    # --------------------------------------------------------------------
    # Indexing Methods

    def get_loc(self, key):
        """
        Get integer location, slice or boolean mask for requested label.

        Parameters
        ----------
        key : label

        Returns
        -------
        int if unique index, slice if monotonic index, else mask

        Examples
        --------
        >>> unique_index = pd.Index(list('abc'))
        >>> unique_index.get_loc('b')
        1

        >>> monotonic_index = pd.Index(list('abbc'))
        >>> monotonic_index.get_loc('b')
        slice(1, 3, None)

        >>> non_monotonic_index = pd.Index(list('abcb'))
        >>> non_monotonic_index.get_loc('b')
        array([False,  True, False,  True])
        """
        casted_key = self._maybe_cast_indexer(key)
        try:
            return self._engine.get_loc(casted_key)
        except KeyError as err:
            if isinstance(casted_key, slice) or (
                isinstance(casted_key, abc.Iterable)
                and any(isinstance(x, slice) for x in casted_key)
            ):
                raise InvalidIndexError(key)
            raise KeyError(key) from err
        except TypeError:
            # If we have a listlike key, _check_indexing_error will raise
            #  InvalidIndexError. Otherwise we fall through and re-raise
            #  the TypeError.
            self._check_indexing_error(key)
            raise

    @final
    def get_indexer(
        self,
        target,
        method: ReindexMethod | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> npt.NDArray[np.intp]:
        """
        Compute indexer and mask for new index given the current index.

        The indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : Index
        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional
            * default: exact matches only.
            * pad / ffill: find the PREVIOUS index value if no exact match.
            * backfill / bfill: use NEXT index value if no exact match
            * nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index value.
        limit : int, optional
            Maximum number of consecutive labels in ``target`` to match for
            inexact matches.
        tolerance : optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index's type.

        Returns
        -------
        np.ndarray[np.intp]
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.

        Notes
        -----
        Returns -1 for unmatched values, for further explanation see the
        example below.

        Examples
        --------
        >>> index = pd.Index(['c', 'a', 'b'])
        >>> index.get_indexer(['a', 'b', 'x'])
        array([ 1,  2, -1])

        Notice that the return value is an array of locations in ``index``
        and ``x`` is marked by -1, as it is not in ``index``.
        """
        method = clean_reindex_fill_method(method)
        orig_target = target
        target = self._maybe_cast_listlike_indexer(target)

        self._check_indexing_method(method, limit, tolerance)

        if not self._index_as_unique:
            raise InvalidIndexError(self._requires_unique_msg)

        if len(target) == 0:
            return np.array([], dtype=np.intp)

        if not self._should_compare(target) and not self._should_partial_index(target):
            # IntervalIndex get special treatment bc numeric scalars can be
            #  matched to Interval scalars
            return self._get_indexer_non_comparable(target, method=method, unique=True)

        if isinstance(self.dtype, CategoricalDtype):
            # _maybe_cast_listlike_indexer ensures target has our dtype
            #  (could improve perf by doing _should_compare check earlier?)
            assert self.dtype == target.dtype

            indexer = self._engine.get_indexer(target.codes)
            if self.hasnans and target.hasnans:
                # After _maybe_cast_listlike_indexer, target elements which do not
                # belong to some category are changed to NaNs
                # Mask to track actual NaN values compared to inserted NaN values
                # GH#45361
                target_nans = isna(orig_target)
                loc = self.get_loc(np.nan)
                mask = target.isna()
                indexer[target_nans] = loc
                indexer[mask & ~target_nans] = -1
            return indexer

        if isinstance(target.dtype, CategoricalDtype):
            # potential fastpath
            # get an indexer for unique categories then propagate to codes via take_nd
            # get_indexer instead of _get_indexer needed for MultiIndex cases
            #  e.g. test_append_different_columns_types
            categories_indexer = self.get_indexer(target.categories)

            indexer = algos.take_nd(categories_indexer, target.codes, fill_value=-1)

            if (not self._is_multi and self.hasnans) and target.hasnans:
                # Exclude MultiIndex because hasnans raises NotImplementedError
                # we should only get here if we are unique, so loc is an integer
                # GH#41934
                loc = self.get_loc(np.nan)
                mask = target.isna()
                indexer[mask] = loc

            return ensure_platform_int(indexer)

        pself, ptarget = self._maybe_downcast_for_indexing(target)
        if pself is not self or ptarget is not target:
            return pself.get_indexer(
                ptarget, method=method, limit=limit, tolerance=tolerance
            )

        if self.dtype == target.dtype and self.equals(target):
            # Only call equals if we have same dtype to avoid inference/casting
            return np.arange(len(target), dtype=np.intp)

        if self.dtype != target.dtype and not self._should_partial_index(target):
            # _should_partial_index e.g. IntervalIndex with numeric scalars
            #  that can be matched to Interval scalars.
            dtype = self._find_common_type_compat(target)

            this = self.astype(dtype, copy=False)
            target = target.astype(dtype, copy=False)
            return this._get_indexer(
                target, method=method, limit=limit, tolerance=tolerance
            )

        return self._get_indexer(target, method, limit, tolerance)

    def _get_indexer(
        self,
        target: Index,
        method: str_t | None = None,
        limit: int | None = None,
        tolerance=None,
    ) -> npt.NDArray[np.intp]:
        if tolerance is not None:
            tolerance = self._convert_tolerance(tolerance, target)

        if method in ["pad", "backfill"]:
            indexer = self._get_fill_indexer(target, method, limit, tolerance)
        elif method == "nearest":
            indexer = self._get_nearest_indexer(target, limit, tolerance)
        else:
            if target._is_multi and self._is_multi:
                engine = self._engine
                # error: Item "IndexEngine" of "Union[IndexEngine, ExtensionEngine]"
                # has no attribute "_extract_level_codes"
                tgt_values = engine._extract_level_codes(  # type: ignore[union-attr]
                    target
                )
            else:
                tgt_values = target._get_engine_target()

            indexer = self._engine.get_indexer(tgt_values)

        return ensure_platform_int(indexer)

    @final
    def _should_partial_index(self, target: Index) -> bool:
        """
        Should we attempt partial-matching indexing?
        """
        if isinstance(self.dtype, IntervalDtype):
            if isinstance(target.dtype, IntervalDtype):
                return False
            # "Index" has no attribute "left"
            return self.left._should_compare(target)  # type: ignore[attr-defined]
        return False

    @final
    def _check_indexing_method(
        self,
        method: str_t | None,
        limit: int | None = None,
        tolerance=None,
    ) -> None:
        """
        Raise if we have a get_indexer `method` that is not supported or valid.
        """
        if method not in [None, "bfill", "backfill", "pad", "ffill", "nearest"]:
            # in practice the clean_reindex_fill_method call would raise
            #  before we get here
            raise ValueError("Invalid fill method")  # pragma: no cover

        if self._is_multi:
            if method == "nearest":
                raise NotImplementedError(
                    "method='nearest' not implemented yet "
                    "for MultiIndex; see GitHub issue 9365"
                )
            if method in ("pad", "backfill"):
                if tolerance is not None:
                    raise NotImplementedError(
                        "tolerance not implemented yet for MultiIndex"
                    )

        if isinstance(self.dtype, (IntervalDtype, CategoricalDtype)):
            # GH#37871 for now this is only for IntervalIndex and CategoricalIndex
            if method is not None:
                raise NotImplementedError(
                    f"method {method} not yet implemented for {type(self).__name__}"
                )

        if method is None:
            if tolerance is not None:
                raise ValueError(
                    "tolerance argument only valid if doing pad, "
                    "backfill or nearest reindexing"
                )
            if limit is not None:
                raise ValueError(
                    "limit argument only valid if doing pad, "
                    "backfill or nearest reindexing"
                )

    def _convert_tolerance(self, tolerance, target: np.ndarray | Index) -> np.ndarray:
        # override this method on subclasses
        tolerance = np.asarray(tolerance)
        if target.size != tolerance.size and tolerance.size > 1:
            raise ValueError("list-like tolerance size must match target index size")
        elif is_numeric_dtype(self) and not np.issubdtype(tolerance.dtype, np.number):
            if tolerance.ndim > 0:
                raise ValueError(
                    f"tolerance argument for {type(self).__name__} with dtype "
                    f"{self.dtype} must contain numeric elements if it is list type"
                )

            raise ValueError(
                f"tolerance argument for {type(self).__name__} with dtype {self.dtype} "
                f"must be numeric if it is a scalar: {repr(tolerance)}"
            )
        return tolerance

    @final
    def _get_fill_indexer(
        self, target: Index, method: str_t, limit: int | None = None, tolerance=None
    ) -> npt.NDArray[np.intp]:
        if self._is_multi:
            if not (self.is_monotonic_increasing or self.is_monotonic_decreasing):
                raise ValueError("index must be monotonic increasing or decreasing")
            encoded = self.append(target)._engine.values  # type: ignore[union-attr]
            self_encoded = Index(encoded[: len(self)])
            target_encoded = Index(encoded[len(self) :])
            return self_encoded._get_fill_indexer(
                target_encoded, method, limit, tolerance
            )

        if self.is_monotonic_increasing and target.is_monotonic_increasing:
            target_values = target._get_engine_target()
            own_values = self._get_engine_target()
            if not isinstance(target_values, np.ndarray) or not isinstance(
                own_values, np.ndarray
            ):
                raise NotImplementedError

            if method == "pad":
                indexer = libalgos.pad(own_values, target_values, limit=limit)
            else:
                # i.e. "backfill"
                indexer = libalgos.backfill(own_values, target_values, limit=limit)
        else:
            indexer = self._get_fill_indexer_searchsorted(target, method, limit)
        if tolerance is not None and len(self):
            indexer = self._filter_indexer_tolerance(target, indexer, tolerance)
        return indexer

    @final
    def _get_fill_indexer_searchsorted(
        self, target: Index, method: str_t, limit: int | None = None
    ) -> npt.NDArray[np.intp]:
        """
        Fallback pad/backfill get_indexer that works for monotonic decreasing
        indexes and non-monotonic targets.
        """
        if limit is not None:
            raise ValueError(
                f"limit argument for {repr(method)} method only well-defined "
                "if index and target are monotonic"
            )

        side: Literal["left", "right"] = "left" if method == "pad" else "right"

        # find exact matches first (this simplifies the algorithm)
        indexer = self.get_indexer(target)
        nonexact = indexer == -1
        indexer[nonexact] = self._searchsorted_monotonic(target[nonexact], side)
        if side == "left":
            # searchsorted returns "indices into a sorted array such that,
            # if the corresponding elements in v were inserted before the
            # indices, the order of a would be preserved".
            # Thus, we need to subtract 1 to find values to the left.
            indexer[nonexact] -= 1
            # This also mapped not found values (values of 0 from
            # np.searchsorted) to -1, which conveniently is also our
            # sentinel for missing values
        else:
            # Mark indices to the right of the largest value as not found
            indexer[indexer == len(self)] = -1
        return indexer

    @final
    def _get_nearest_indexer(
        self, target: Index, limit: int | None, tolerance
    ) -> npt.NDArray[np.intp]:
        """
        Get the indexer for the nearest index labels; requires an index with
        values that can be subtracted from each other (e.g., not strings or
        tuples).
        """
        if not len(self):
            return self._get_fill_indexer(target, "pad")

        left_indexer = self.get_indexer(target, "pad", limit=limit)
        right_indexer = self.get_indexer(target, "backfill", limit=limit)

        left_distances = self._difference_compat(target, left_indexer)
        right_distances = self._difference_compat(target, right_indexer)

        op = operator.lt if self.is_monotonic_increasing else operator.le
        indexer = np.where(
            # error: Argument 1&2 has incompatible type "Union[ExtensionArray,
            # ndarray[Any, Any]]"; expected "Union[SupportsDunderLE,
            # SupportsDunderGE, SupportsDunderGT, SupportsDunderLT]"
            op(left_distances, right_distances)  # type: ignore[arg-type]
            | (right_indexer == -1),
            left_indexer,
            right_indexer,
        )
        if tolerance is not None:
            indexer = self._filter_indexer_tolerance(target, indexer, tolerance)
        return indexer

    @final
    def _filter_indexer_tolerance(
        self,
        target: Index,
        indexer: npt.NDArray[np.intp],
        tolerance,
    ) -> npt.NDArray[np.intp]:
        distance = self._difference_compat(target, indexer)

        return np.where(distance <= tolerance, indexer, -1)

    @final
    def _difference_compat(
        self, target: Index, indexer: npt.NDArray[np.intp]
    ) -> ArrayLike:
        # Compatibility for PeriodArray, for which __sub__ returns an ndarray[object]
        #  of DateOffset objects, which do not support __abs__ (and would be slow
        #  if they did)

        if isinstance(self.dtype, PeriodDtype):
            # Note: we only get here with matching dtypes
            own_values = cast("PeriodArray", self._data)._ndarray
            target_values = cast("PeriodArray", target._data)._ndarray
            diff = own_values[indexer] - target_values
        else:
            # error: Unsupported left operand type for - ("ExtensionArray")
            diff = self._values[indexer] - target._values  # type: ignore[operator]
        return abs(diff)

    # --------------------------------------------------------------------
    # Indexer Conversion Methods

    @final
    def _validate_positional_slice(self, key: slice) -> None:
        """
        For positional indexing, a slice must have either int or None
        for each of start, stop, and step.
        """
        self._validate_indexer("positional", key.start, "iloc")
        self._validate_indexer("positional", key.stop, "iloc")
        self._validate_indexer("positional", key.step, "iloc")

    def _convert_slice_indexer(self, key: slice, kind: Literal["loc", "getitem"]):
        """
        Convert a slice indexer.

        By definition, these are labels unless 'iloc' is passed in.
        Floats are not allowed as the start, step, or stop of the slice.

        Parameters
        ----------
        key : label of the slice bound
        kind : {'loc', 'getitem'}
        """

        # potentially cast the bounds to integers
        start, stop, step = key.start, key.stop, key.step

        # figure out if this is a positional indexer
        is_index_slice = is_valid_positional_slice(key)

        # TODO(GH#50617): once Series.__[gs]etitem__ is removed we should be able
        #  to simplify this.
        if lib.is_np_dtype(self.dtype, "f"):
            # We always treat __getitem__ slicing as label-based
            # translate to locations
            if kind == "getitem" and is_index_slice and not start == stop and step != 0:
                # exclude step=0 from the warning because it will raise anyway
                # start/stop both None e.g. [:] or [::-1] won't change.
                # exclude start==stop since it will be empty either way, or
                # will be [:] or [::-1] which won't change
                warnings.warn(
                    # GH#49612
                    "The behavior of obj[i:j] with a float-dtype index is "
                    "deprecated. In a future version, this will be treated as "
                    "positional instead of label-based. For label-based slicing, "
                    "use obj.loc[i:j] instead",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
            return self.slice_indexer(start, stop, step)

        if kind == "getitem":
            # called from the getitem slicers, validate that we are in fact integers
            if is_index_slice:
                # In this case the _validate_indexer checks below are redundant
                return key
            elif self.dtype.kind in "iu":
                # Note: these checks are redundant if we know is_index_slice
                self._validate_indexer("slice", key.start, "getitem")
                self._validate_indexer("slice", key.stop, "getitem")
                self._validate_indexer("slice", key.step, "getitem")
                return key

        # convert the slice to an indexer here; checking that the user didn't
        #  pass a positional slice to loc
        is_positional = is_index_slice and self._should_fallback_to_positional

        # if we are mixed and have integers
        if is_positional:
            try:
                # Validate start & stop
                if start is not None:
                    self.get_loc(start)
                if stop is not None:
                    self.get_loc(stop)
                is_positional = False
            except KeyError:
                pass

        if com.is_null_slice(key):
            # It doesn't matter if we are positional or label based
            indexer = key
        elif is_positional:
            if kind == "loc":
                # GH#16121, GH#24612, GH#31810
                raise TypeError(
                    "Slicing a positional slice with .loc is not allowed, "
                    "Use .loc with labels or .iloc with positions instead.",
                )
            indexer = key
        else:
            indexer = self.slice_indexer(start, stop, step)

        return indexer

    @final
    def _raise_invalid_indexer(
        self,
        form: Literal["slice", "positional"],
        key,
        reraise: lib.NoDefault | None | Exception = lib.no_default,
    ) -> None:
        """
        Raise consistent invalid indexer message.
        """
        msg = (
            f"cannot do {form} indexing on {type(self).__name__} with these "
            f"indexers [{key}] of type {type(key).__name__}"
        )
        if reraise is not lib.no_default:
            raise TypeError(msg) from reraise
        raise TypeError(msg)

    # --------------------------------------------------------------------
    # Reindex Methods

    @final
    def _validate_can_reindex(self, indexer: np.ndarray) -> None:
        """
        Check if we are allowing reindexing with this particular indexer.

        Parameters
        ----------
        indexer : an integer ndarray

        Raises
        ------
        ValueError if its a duplicate axis
        """
        # trying to reindex on an axis with duplicates
        if not self._index_as_unique and len(indexer):
            raise ValueError("cannot reindex on an axis with duplicate labels")

    def reindex(
        self,
        target,
        method: ReindexMethod | None = None,
        level=None,
        limit: int | None = None,
        tolerance: float | None = None,
    ) -> tuple[Index, npt.NDArray[np.intp] | None]:
        """
        Create index with target's values.

        Parameters
        ----------
        target : an iterable
        method : {None, 'pad'/'ffill', 'backfill'/'bfill', 'nearest'}, optional
            * default: exact matches only.
            * pad / ffill: find the PREVIOUS index value if no exact match.
            * backfill / bfill: use NEXT index value if no exact match
            * nearest: use the NEAREST index value if no exact match. Tied
              distances are broken by preferring the larger index value.
        level : int, optional
            Level of multiindex.
        limit : int, optional
            Maximum number of consecutive labels in ``target`` to match for
            inexact matches.
        tolerance : int or float, optional
            Maximum distance between original and new labels for inexact
            matches. The values of the index at the matching locations must
            satisfy the equation ``abs(index[indexer] - target) <= tolerance``.

            Tolerance may be a scalar value, which applies the same tolerance
            to all values, or list-like, which applies variable tolerance per
            element. List-like includes list, tuple, array, Series, and must be
            the same size as the index and its dtype must exactly match the
            index's type.

        Returns
        -------
        new_index : pd.Index
            Resulting index.
        indexer : np.ndarray[np.intp] or None
            Indices of output values in original index.

        Raises
        ------
        TypeError
            If ``method`` passed along with ``level``.
        ValueError
            If non-unique multi-index
        ValueError
            If non-unique index and ``method`` or ``limit`` passed.

        See Also
        --------
        Series.reindex : Conform Series to new index with optional filling logic.
        DataFrame.reindex : Conform DataFrame to new index with optional filling logic.

        Examples
        --------
        >>> idx = pd.Index(['car', 'bike', 'train', 'tractor'])
        >>> idx
        Index(['car', 'bike', 'train', 'tractor'], dtype='object')
        >>> idx.reindex(['car', 'bike'])
        (Index(['car', 'bike'], dtype='object'), array([0, 1]))
        """
        # GH6552: preserve names when reindexing to non-named target
        # (i.e. neither Index nor Series).
        preserve_names = not hasattr(target, "name")

        # GH7774: preserve dtype/tz if target is empty and not an Index.
        target = ensure_has_len(target)  # target may be an iterator

        if not isinstance(target, Index) and len(target) == 0:
            if level is not None and self._is_multi:
                # "Index" has no attribute "levels"; maybe "nlevels"?
                idx = self.levels[level]  # type: ignore[attr-defined]
            else:
                idx = self
            target = idx[:0]
        else:
            target = ensure_index(target)

        if level is not None and (
            isinstance(self, ABCMultiIndex) or isinstance(target, ABCMultiIndex)
        ):
            if method is not None:
                raise TypeError("Fill method not supported if level passed")

            # TODO: tests where passing `keep_order=not self._is_multi`
            #  makes a difference for non-MultiIndex case
            target, indexer, _ = self._join_level(
                target, level, how="right", keep_order=not self._is_multi
            )

        else:
            if self.equals(target):
                indexer = None
            else:
                if self._index_as_unique:
                    indexer = self.get_indexer(
                        target, method=method, limit=limit, tolerance=tolerance
                    )
                elif self._is_multi:
                    raise ValueError("cannot handle a non-unique multi-index!")
                elif not self.is_unique:
                    # GH#42568
                    raise ValueError("cannot reindex on an axis with duplicate labels")
                else:
                    indexer, _ = self.get_indexer_non_unique(target)

        target = self._wrap_reindex_result(target, indexer, preserve_names)
        return target, indexer

    def _wrap_reindex_result(self, target, indexer, preserve_names: bool):
        target = self._maybe_preserve_names(target, preserve_names)
        return target

    def _maybe_preserve_names(self, target: Index, preserve_names: bool):
        if preserve_names and target.nlevels == 1 and target.name != self.name:
            target = target.copy(deep=False)
            target.name = self.name
        return target

    @final
    def _reindex_non_unique(
        self, target: Index
    ) -> tuple[Index, npt.NDArray[np.intp], npt.NDArray[np.intp] | None]:
        """
        Create a new index with target's values (move/add/delete values as
        necessary) use with non-unique Index and a possibly non-unique target.

        Parameters
        ----------
        target : an iterable

        Returns
        -------
        new_index : pd.Index
            Resulting index.
        indexer : np.ndarray[np.intp]
            Indices of output values in original index.
        new_indexer : np.ndarray[np.intp] or None

        """
        target = ensure_index(target)
        if len(target) == 0:
            # GH#13691
            return self[:0], np.array([], dtype=np.intp), None

        indexer, missing = self.get_indexer_non_unique(target)
        check = indexer != -1
        new_labels: Index | np.ndarray = self.take(indexer[check])
        new_indexer = None

        if len(missing):
            length = np.arange(len(indexer), dtype=np.intp)

            missing = ensure_platform_int(missing)
            missing_labels = target.take(missing)
            missing_indexer = length[~check]
            cur_labels = self.take(indexer[check]).values
            cur_indexer = length[check]

            # Index constructor below will do inference
            new_labels = np.empty((len(indexer),), dtype=object)
            new_labels[cur_indexer] = cur_labels
            new_labels[missing_indexer] = missing_labels

            # GH#38906
            if not len(self):
                new_indexer = np.arange(0, dtype=np.intp)

            # a unique indexer
            elif target.is_unique:
                # see GH5553, make sure we use the right indexer
                new_indexer = np.arange(len(indexer), dtype=np.intp)
                new_indexer[cur_indexer] = np.arange(len(cur_labels))
                new_indexer[missing_indexer] = -1

            # we have a non_unique selector, need to use the original
            # indexer here
            else:
                # need to retake to have the same size as the indexer
                indexer[~check] = -1

                # reset the new indexer to account for the new size
                new_indexer = np.arange(len(self.take(indexer)), dtype=np.intp)
                new_indexer[~check] = -1

        if not isinstance(self, ABCMultiIndex):
            new_index = Index(new_labels, name=self.name)
        else:
            new_index = type(self).from_tuples(new_labels, names=self.names)
        return new_index, indexer, new_indexer

    # --------------------------------------------------------------------
    # Join Methods

    @overload
    def join(
        self,
        other: Index,
        *,
        how: JoinHow = ...,
        level: Level = ...,
        return_indexers: Literal[True],
        sort: bool = ...,
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        ...

    @overload
    def join(
        self,
        other: Index,
        *,
        how: JoinHow = ...,
        level: Level = ...,
        return_indexers: Literal[False] = ...,
        sort: bool = ...,
    ) -> Index:
        ...

    @overload
    def join(
        self,
        other: Index,
        *,
        how: JoinHow = ...,
        level: Level = ...,
        return_indexers: bool = ...,
        sort: bool = ...,
    ) -> Index | tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        ...

    @final
    @_maybe_return_indexers
    def join(
        self,
        other: Index,
        *,
        how: JoinHow = "left",
        level: Level | None = None,
        return_indexers: bool = False,
        sort: bool = False,
    ) -> Index | tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        """
        Compute join_index and indexers to conform data structures to the new index.

        Parameters
        ----------
        other : Index
        how : {'left', 'right', 'inner', 'outer'}
        level : int or level name, default None
        return_indexers : bool, default False
        sort : bool, default False
            Sort the join keys lexicographically in the result Index. If False,
            the order of the join keys depends on the join type (how keyword).

        Returns
        -------
        join_index, (left_indexer, right_indexer)

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx2 = pd.Index([4, 5, 6])
        >>> idx1.join(idx2, how='outer')
        Index([1, 2, 3, 4, 5, 6], dtype='int64')
        """
        other = ensure_index(other)
        sort = sort or how == "outer"

        if isinstance(self, ABCDatetimeIndex) and isinstance(other, ABCDatetimeIndex):
            if (self.tz is None) ^ (other.tz is None):
                # Raise instead of casting to object below.
                raise TypeError("Cannot join tz-naive with tz-aware DatetimeIndex")

        if not self._is_multi and not other._is_multi:
            # We have specific handling for MultiIndex below
            pself, pother = self._maybe_downcast_for_indexing(other)
            if pself is not self or pother is not other:
                return pself.join(
                    pother, how=how, level=level, return_indexers=True, sort=sort
                )

        # try to figure out the join level
        # GH3662
        if level is None and (self._is_multi or other._is_multi):
            # have the same levels/names so a simple join
            if self.names == other.names:
                pass
            else:
                return self._join_multi(other, how=how)

        # join on the level
        if level is not None and (self._is_multi or other._is_multi):
            return self._join_level(other, level, how=how)

        if len(self) == 0 or len(other) == 0:
            try:
                return self._join_empty(other, how, sort)
            except TypeError:
                # object dtype; non-comparable objects
                pass

        if self.dtype != other.dtype:
            dtype = self._find_common_type_compat(other)
            this = self.astype(dtype, copy=False)
            other = other.astype(dtype, copy=False)
            return this.join(other, how=how, return_indexers=True)
        elif (
            isinstance(self, ABCCategoricalIndex)
            and isinstance(other, ABCCategoricalIndex)
            and not self.ordered
            and not self.categories.equals(other.categories)
        ):
            # dtypes are "equal" but categories are in different order
            other = Index(other._values.reorder_categories(self.categories))

        _validate_join_method(how)

        if (
            self.is_monotonic_increasing
            and other.is_monotonic_increasing
            and self._can_use_libjoin
            and other._can_use_libjoin
            and (self.is_unique or other.is_unique)
        ):
            try:
                return self._join_monotonic(other, how=how)
            except TypeError:
                # object dtype; non-comparable objects
                pass
        elif not self.is_unique or not other.is_unique:
            return self._join_non_unique(other, how=how, sort=sort)

        return self._join_via_get_indexer(other, how, sort)

    @final
    def _join_empty(
        self, other: Index, how: JoinHow, sort: bool
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        assert len(self) == 0 or len(other) == 0
        _validate_join_method(how)

        lidx: np.ndarray | None
        ridx: np.ndarray | None

        if len(other):
            how = cast(JoinHow, {"left": "right", "right": "left"}.get(how, how))
            join_index, ridx, lidx = other._join_empty(self, how, sort)
        elif how in ["left", "outer"]:
            if sort and not self.is_monotonic_increasing:
                lidx = self.argsort()
                join_index = self.take(lidx)
            else:
                lidx = None
                join_index = self._view()
            ridx = np.broadcast_to(np.intp(-1), len(join_index))
        else:
            join_index = other._view()
            lidx = np.array([], dtype=np.intp)
            ridx = None
        return join_index, lidx, ridx

    @final
    def _join_via_get_indexer(
        self, other: Index, how: JoinHow, sort: bool
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        # Fallback if we do not have any fastpaths available based on
        #  uniqueness/monotonicity

        # Note: at this point we have checked matching dtypes

        if how == "left":
            join_index = self.sort_values() if sort else self
        elif how == "right":
            join_index = other.sort_values() if sort else other
        elif how == "inner":
            join_index = self.intersection(other, sort=sort)
        elif how == "outer":
            try:
                join_index = self.union(other, sort=sort)
            except TypeError:
                join_index = self.union(other)
                try:
                    join_index = _maybe_try_sort(join_index, sort)
                except TypeError:
                    pass

        if join_index is self:
            lindexer = None
        else:
            lindexer = self.get_indexer_for(join_index)
        if join_index is other:
            rindexer = None
        else:
            rindexer = other.get_indexer_for(join_index)
        return join_index, lindexer, rindexer

    @final
    def _join_multi(self, other: Index, how: JoinHow):
        from pandas.core.indexes.multi import MultiIndex
        from pandas.core.reshape.merge import restore_dropped_levels_multijoin

        # figure out join names
        self_names_list = list(com.not_none(*self.names))
        other_names_list = list(com.not_none(*other.names))
        self_names_order = self_names_list.index
        other_names_order = other_names_list.index
        self_names = set(self_names_list)
        other_names = set(other_names_list)
        overlap = self_names & other_names

        # need at least 1 in common
        if not overlap:
            raise ValueError("cannot join with no overlapping index names")

        if isinstance(self, MultiIndex) and isinstance(other, MultiIndex):
            # Drop the non-matching levels from left and right respectively
            ldrop_names = sorted(self_names - overlap, key=self_names_order)
            rdrop_names = sorted(other_names - overlap, key=other_names_order)

            # if only the order differs
            if not len(ldrop_names + rdrop_names):
                self_jnlevels = self
                other_jnlevels = other.reorder_levels(self.names)
            else:
                self_jnlevels = self.droplevel(ldrop_names)
                other_jnlevels = other.droplevel(rdrop_names)

            # Join left and right
            # Join on same leveled multi-index frames is supported
            join_idx, lidx, ridx = self_jnlevels.join(
                other_jnlevels, how=how, return_indexers=True
            )

            # Restore the dropped levels
            # Returned index level order is
            # common levels, ldrop_names, rdrop_names
            dropped_names = ldrop_names + rdrop_names

            # error: Argument 5/6 to "restore_dropped_levels_multijoin" has
            # incompatible type "Optional[ndarray[Any, dtype[signedinteger[Any
            # ]]]]"; expected "ndarray[Any, dtype[signedinteger[Any]]]"
            levels, codes, names = restore_dropped_levels_multijoin(
                self,
                other,
                dropped_names,
                join_idx,
                lidx,  # type: ignore[arg-type]
                ridx,  # type: ignore[arg-type]
            )

            # Re-create the multi-index
            multi_join_idx = MultiIndex(
                levels=levels, codes=codes, names=names, verify_integrity=False
            )

            multi_join_idx = multi_join_idx.remove_unused_levels()

            # maintain the order of the index levels
            if how == "right":
                level_order = other_names_list + ldrop_names
            else:
                level_order = self_names_list + rdrop_names
            multi_join_idx = multi_join_idx.reorder_levels(level_order)

            return multi_join_idx, lidx, ridx

        jl = next(iter(overlap))

        # Case where only one index is multi
        # make the indices into mi's that match
        flip_order = False
        if isinstance(self, MultiIndex):
            self, other = other, self
            flip_order = True
            # flip if join method is right or left
            flip: dict[JoinHow, JoinHow] = {"right": "left", "left": "right"}
            how = flip.get(how, how)

        level = other.names.index(jl)
        result = self._join_level(other, level, how=how)

        if flip_order:
            return result[0], result[2], result[1]
        return result

    @final
    def _join_non_unique(
        self, other: Index, how: JoinHow = "left", sort: bool = False
    ) -> tuple[Index, npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        from pandas.core.reshape.merge import get_join_indexers_non_unique

        # We only get here if dtypes match
        assert self.dtype == other.dtype

        left_idx, right_idx = get_join_indexers_non_unique(
            self._values, other._values, how=how, sort=sort
        )
        mask = left_idx == -1

        join_idx = self.take(left_idx)
        right = other.take(right_idx)
        join_index = join_idx.putmask(mask, right)
        if isinstance(join_index, ABCMultiIndex) and how == "outer":
            # test_join_index_levels
            join_index = join_index._sort_levels_monotonic()
        return join_index, left_idx, right_idx

    @final
    def _join_level(
        self, other: Index, level, how: JoinHow = "left", keep_order: bool = True
    ) -> tuple[MultiIndex, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        """
        The join method *only* affects the level of the resulting
        MultiIndex. Otherwise it just exactly aligns the Index data to the
        labels of the level in the MultiIndex.

        If ```keep_order == True```, the order of the data indexed by the
        MultiIndex will not be changed; otherwise, it will tie out
        with `other`.
        """
        from pandas.core.indexes.multi import MultiIndex

        def _get_leaf_sorter(labels: list[np.ndarray]) -> npt.NDArray[np.intp]:
            """
            Returns sorter for the inner most level while preserving the
            order of higher levels.

            Parameters
            ----------
            labels : list[np.ndarray]
                Each ndarray has signed integer dtype, not necessarily identical.

            Returns
            -------
            np.ndarray[np.intp]
            """
            if labels[0].size == 0:
                return np.empty(0, dtype=np.intp)

            if len(labels) == 1:
                return get_group_index_sorter(ensure_platform_int(labels[0]))

            # find indexers of beginning of each set of
            # same-key labels w.r.t all but last level
            tic = labels[0][:-1] != labels[0][1:]
            for lab in labels[1:-1]:
                tic |= lab[:-1] != lab[1:]

            starts = np.hstack(([True], tic, [True])).nonzero()[0]
            lab = ensure_int64(labels[-1])
            return lib.get_level_sorter(lab, ensure_platform_int(starts))

        if isinstance(self, MultiIndex) and isinstance(other, MultiIndex):
            raise TypeError("Join on level between two MultiIndex objects is ambiguous")

        left, right = self, other

        flip_order = not isinstance(self, MultiIndex)
        if flip_order:
            left, right = right, left
            flip: dict[JoinHow, JoinHow] = {"right": "left", "left": "right"}
            how = flip.get(how, how)

        assert isinstance(left, MultiIndex)

        level = left._get_level_number(level)
        old_level = left.levels[level]

        if not right.is_unique:
            raise NotImplementedError(
                "Index._join_level on non-unique index is not implemented"
            )

        new_level, left_lev_indexer, right_lev_indexer = old_level.join(
            right, how=how, return_indexers=True
        )

        if left_lev_indexer is None:
            if keep_order or len(left) == 0:
                left_indexer = None
                join_index = left
            else:  # sort the leaves
                left_indexer = _get_leaf_sorter(left.codes[: level + 1])
                join_index = left[left_indexer]

        else:
            left_lev_indexer = ensure_platform_int(left_lev_indexer)
            rev_indexer = lib.get_reverse_indexer(left_lev_indexer, len(old_level))
            old_codes = left.codes[level]

            taker = old_codes[old_codes != -1]
            new_lev_codes = rev_indexer.take(taker)

            new_codes = list(left.codes)
            new_codes[level] = new_lev_codes

            new_levels = list(left.levels)
            new_levels[level] = new_level

            if keep_order:  # just drop missing values. o.w. keep order
                left_indexer = np.arange(len(left), dtype=np.intp)
                left_indexer = cast(np.ndarray, left_indexer)
                mask = new_lev_codes != -1
                if not mask.all():
                    new_codes = [lab[mask] for lab in new_codes]
                    left_indexer = left_indexer[mask]

            else:  # tie out the order with other
                if level == 0:  # outer most level, take the fast route
                    max_new_lev = 0 if len(new_lev_codes) == 0 else new_lev_codes.max()
                    ngroups = 1 + max_new_lev
                    left_indexer, counts = libalgos.groupsort_indexer(
                        new_lev_codes, ngroups
                    )

                    # missing values are placed first; drop them!
                    left_indexer = left_indexer[counts[0] :]
                    new_codes = [lab[left_indexer] for lab in new_codes]

                else:  # sort the leaves
                    mask = new_lev_codes != -1
                    mask_all = mask.all()
                    if not mask_all:
                        new_codes = [lab[mask] for lab in new_codes]

                    left_indexer = _get_leaf_sorter(new_codes[: level + 1])
                    new_codes = [lab[left_indexer] for lab in new_codes]

                    # left_indexers are w.r.t masked frame.
                    # reverse to original frame!
                    if not mask_all:
                        left_indexer = mask.nonzero()[0][left_indexer]

            join_index = MultiIndex(
                levels=new_levels,
                codes=new_codes,
                names=left.names,
                verify_integrity=False,
            )

        if right_lev_indexer is not None:
            right_indexer = right_lev_indexer.take(join_index.codes[level])
        else:
            right_indexer = join_index.codes[level]

        if flip_order:
            left_indexer, right_indexer = right_indexer, left_indexer

        left_indexer = (
            None if left_indexer is None else ensure_platform_int(left_indexer)
        )
        right_indexer = (
            None if right_indexer is None else ensure_platform_int(right_indexer)
        )
        return join_index, left_indexer, right_indexer

    @final
    def _join_monotonic(
        self, other: Index, how: JoinHow = "left"
    ) -> tuple[Index, npt.NDArray[np.intp] | None, npt.NDArray[np.intp] | None]:
        # We only get here with matching dtypes and both monotonic increasing
        assert other.dtype == self.dtype
        assert self._can_use_libjoin and other._can_use_libjoin

        if self.equals(other):
            # This is a convenient place for this check, but its correctness
            #  does not depend on monotonicity, so it could go earlier
            #  in the calling method.
            ret_index = other if how == "right" else self
            return ret_index, None, None

        ridx: npt.NDArray[np.intp] | None
        lidx: npt.NDArray[np.intp] | None

        if self.is_unique and other.is_unique:
            # We can perform much better than the general case
            if how == "left":
                join_index = self
                lidx = None
                ridx = self._left_indexer_unique(other)
            elif how == "right":
                join_index = other
                lidx = other._left_indexer_unique(self)
                ridx = None
            elif how == "inner":
                join_array, lidx, ridx = self._inner_indexer(other)
                join_index = self._wrap_joined_index(join_array, other, lidx, ridx)
            elif how == "outer":
                join_array, lidx, ridx = self._outer_indexer(other)
                join_index = self._wrap_joined_index(join_array, other, lidx, ridx)
        else:
            if how == "left":
                join_array, lidx, ridx = self._left_indexer(other)
            elif how == "right":
                join_array, ridx, lidx = other._left_indexer(self)
            elif how == "inner":
                join_array, lidx, ridx = self._inner_indexer(other)
            elif how == "outer":
                join_array, lidx, ridx = self._outer_indexer(other)

            assert lidx is not None
            assert ridx is not None

            join_index = self._wrap_joined_index(join_array, other, lidx, ridx)

        lidx = None if lidx is None else ensure_platform_int(lidx)
        ridx = None if ridx is None else ensure_platform_int(ridx)
        return join_index, lidx, ridx

    def _wrap_joined_index(
        self,
        joined: ArrayLike,
        other: Self,
        lidx: npt.NDArray[np.intp],
        ridx: npt.NDArray[np.intp],
    ) -> Self:
        assert other.dtype == self.dtype

        if isinstance(self, ABCMultiIndex):
            name = self.names if self.names == other.names else None
            # error: Incompatible return value type (got "MultiIndex",
            # expected "Self")
            mask = lidx == -1
            join_idx = self.take(lidx)
            right = cast("MultiIndex", other.take(ridx))
            join_index = join_idx.putmask(mask, right)._sort_levels_monotonic()
            return join_index.set_names(name)  # type: ignore[return-value]
        else:
            name = get_op_result_name(self, other)
            return self._constructor._with_infer(joined, name=name, dtype=self.dtype)

    @final
    @cache_readonly
    def _can_use_libjoin(self) -> bool:
        """
        Whether we can use the fastpaths implemented in _libs.join.

        This is driven by whether (in monotonic increasing cases that are
        guaranteed not to have NAs) we can convert to a np.ndarray without
        making a copy. If we cannot, this negates the performance benefit
        of using libjoin.
        """
        if type(self) is Index:
            # excludes EAs, but include masks, we get here with monotonic
            # values only, meaning no NA
            return (
                isinstance(self.dtype, np.dtype)
                or isinstance(self._values, (ArrowExtensionArray, BaseMaskedArray))
                or (
                    isinstance(self.dtype, StringDtype)
                    and self.dtype.storage == "python"
                )
            )
        # Exclude index types where the conversion to numpy converts to object dtype,
        #  which negates the performance benefit of libjoin
        # Subclasses should override to return False if _get_join_target is
        #  not zero-copy.
        # TODO: exclude RangeIndex (which allocates memory)?
        #  Doing so seems to break test_concat_datetime_timezone
        return not isinstance(self, (ABCIntervalIndex, ABCMultiIndex))

    # --------------------------------------------------------------------
    # Uncategorized Methods

    @property
    def values(self) -> ArrayLike:
        """
        Return an array representing the data in the Index.

        .. warning::

           We recommend using :attr:`Index.array` or
           :meth:`Index.to_numpy`, depending on whether you need
           a reference to the underlying data or a NumPy array.

        Returns
        -------
        array: numpy.ndarray or ExtensionArray

        See Also
        --------
        Index.array : Reference to the underlying data.
        Index.to_numpy : A NumPy array representing the underlying data.

        Examples
        --------
        For :class:`pandas.Index`:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.values
        array([1, 2, 3])

        For :class:`pandas.IntervalIndex`:

        >>> idx = pd.interval_range(start=0, end=5)
        >>> idx.values
        <IntervalArray>
        [(0, 1], (1, 2], (2, 3], (3, 4], (4, 5]]
        Length: 5, dtype: interval[int64, right]
        """
        if using_copy_on_write():
            data = self._data
            if isinstance(data, np.ndarray):
                data = data.view()
                data.flags.writeable = False
            return data
        return self._data

    @cache_readonly
    @doc(IndexOpsMixin.array)
    def array(self) -> ExtensionArray:
        array = self._data
        if isinstance(array, np.ndarray):
            from pandas.core.arrays.numpy_ import NumpyExtensionArray

            array = NumpyExtensionArray(array)
        return array

    @property
    def _values(self) -> ExtensionArray | np.ndarray:
        """
        The best array representation.

        This is an ndarray or ExtensionArray.

        ``_values`` are consistent between ``Series`` and ``Index``.

        It may differ from the public '.values' method.

        index             | values          | _values       |
        ----------------- | --------------- | ------------- |
        Index             | ndarray         | ndarray       |
        CategoricalIndex  | Categorical     | Categorical   |
        DatetimeIndex     | ndarray[M8ns]   | DatetimeArray |
        DatetimeIndex[tz] | ndarray[M8ns]   | DatetimeArray |
        PeriodIndex       | ndarray[object] | PeriodArray   |
        IntervalIndex     | IntervalArray   | IntervalArray |

        See Also
        --------
        values : Values
        """
        return self._data

    def _get_engine_target(self) -> ArrayLike:
        """
        Get the ndarray or ExtensionArray that we can pass to the IndexEngine
        constructor.
        """
        vals = self._values
        if isinstance(vals, StringArray):
            # GH#45652 much more performant than ExtensionEngine
            return vals._ndarray
        if isinstance(vals, ArrowExtensionArray) and self.dtype.kind in "Mm":
            import pyarrow as pa

            pa_type = vals._pa_array.type
            if pa.types.is_timestamp(pa_type):
                vals = vals._to_datetimearray()
                return vals._ndarray.view("i8")
            elif pa.types.is_duration(pa_type):
                vals = vals._to_timedeltaarray()
                return vals._ndarray.view("i8")
        if (
            type(self) is Index
            and isinstance(self._values, ExtensionArray)
            and not isinstance(self._values, BaseMaskedArray)
            and not (
                isinstance(self._values, ArrowExtensionArray)
                and is_numeric_dtype(self.dtype)
                # Exclude decimal
                and self.dtype.kind != "O"
            )
        ):
            # TODO(ExtensionIndex): remove special-case, just use self._values
            return self._values.astype(object)
        return vals

    @final
    def _get_join_target(self) -> np.ndarray:
        """
        Get the ndarray or ExtensionArray that we can pass to the join
        functions.
        """
        if isinstance(self._values, BaseMaskedArray):
            # This is only used if our array is monotonic, so no NAs present
            return self._values._data
        elif isinstance(self._values, ArrowExtensionArray):
            # This is only used if our array is monotonic, so no missing values
            # present
            return self._values.to_numpy()

        # TODO: exclude ABCRangeIndex case here as it copies
        target = self._get_engine_target()
        if not isinstance(target, np.ndarray):
            raise ValueError("_can_use_libjoin should return False.")
        return target

    def _from_join_target(self, result: np.ndarray) -> ArrayLike:
        """
        Cast the ndarray returned from one of the libjoin.foo_indexer functions
        back to type(self._data).
        """
        if isinstance(self.values, BaseMaskedArray):
            return type(self.values)(result, np.zeros(result.shape, dtype=np.bool_))
        elif isinstance(self.values, (ArrowExtensionArray, StringArray)):
            return type(self.values)._from_sequence(result, dtype=self.dtype)
        return result

    @doc(IndexOpsMixin._memory_usage)
    def memory_usage(self, deep: bool = False) -> int:
        result = self._memory_usage(deep=deep)

        # include our engine hashtable
        result += self._engine.sizeof(deep=deep)
        return result

    @final
    def where(self, cond, other=None) -> Index:
        """
        Replace values where the condition is False.

        The replacement is taken from other.

        Parameters
        ----------
        cond : bool array-like with the same length as self
            Condition to select the values on.
        other : scalar, or array-like, default None
            Replacement if the condition is False.

        Returns
        -------
        pandas.Index
            A copy of self with values replaced from other
            where the condition is False.

        See Also
        --------
        Series.where : Same method for Series.
        DataFrame.where : Same method for DataFrame.

        Examples
        --------
        >>> idx = pd.Index(['car', 'bike', 'train', 'tractor'])
        >>> idx
        Index(['car', 'bike', 'train', 'tractor'], dtype='object')
        >>> idx.where(idx.isin(['car', 'train']), 'other')
        Index(['car', 'other', 'train', 'other'], dtype='object')
        """
        if isinstance(self, ABCMultiIndex):
            raise NotImplementedError(
                ".where is not supported for MultiIndex operations"
            )
        cond = np.asarray(cond, dtype=bool)
        return self.putmask(~cond, other)

    # construction helpers
    @final
    @classmethod
    def _raise_scalar_data_error(cls, data):
        # We return the TypeError so that we can raise it from the constructor
        #  in order to keep mypy happy
        raise TypeError(
            f"{cls.__name__}(...) must be called with a collection of some "
            f"kind, {repr(data) if not isinstance(data, np.generic) else str(data)} "
            "was passed"
        )

    def _validate_fill_value(self, value):
        """
        Check if the value can be inserted into our array without casting,
        and convert it to an appropriate native type if necessary.

        Raises
        ------
        TypeError
            If the value cannot be inserted into an array of this dtype.
        """
        dtype = self.dtype
        if isinstance(dtype, np.dtype) and dtype.kind not in "mM":
            # return np_can_hold_element(dtype, value)
            try:
                return np_can_hold_element(dtype, value)
            except LossySetitemError as err:
                # re-raise as TypeError for consistency
                raise TypeError from err
        elif not can_hold_element(self._values, value):
            raise TypeError
        return value

    def _is_memory_usage_qualified(self) -> bool:
        """
        Return a boolean if we need a qualified .info display.
        """
        return is_object_dtype(self.dtype) or (
            is_string_dtype(self.dtype) and self.dtype.storage == "python"  # type: ignore[union-attr]
        )

    def __contains__(self, key: Any) -> bool:
        """
        Return a boolean indicating whether the provided key is in the index.

        Parameters
        ----------
        key : label
            The key to check if it is present in the index.

        Returns
        -------
        bool
            Whether the key search is in the index.

        Raises
        ------
        TypeError
            If the key is not hashable.

        See Also
        --------
        Index.isin : Returns an ndarray of boolean dtype indicating whether the
            list-like key is in the index.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3, 4])
        >>> idx
        Index([1, 2, 3, 4], dtype='int64')

        >>> 2 in idx
        True
        >>> 6 in idx
        False
        """
        hash(key)
        try:
            return key in self._engine
        except (OverflowError, TypeError, ValueError):
            return False

    # https://github.com/python/typeshed/issues/2148#issuecomment-520783318
    # Incompatible types in assignment (expression has type "None", base class
    # "object" defined the type as "Callable[[object], int]")
    __hash__: ClassVar[None]  # type: ignore[assignment]

    @final
    def __setitem__(self, key, value) -> None:
        raise TypeError("Index does not support mutable operations")

    def __getitem__(self, key):
        """
        Override numpy.ndarray's __getitem__ method to work as desired.

        This function adds lists and Series as valid boolean indexers
        (ndarrays only supports ndarray with dtype=bool).

        If resulting ndim != 1, plain ndarray is returned instead of
        corresponding `Index` subclass.

        """
        getitem = self._data.__getitem__

        if is_integer(key) or is_float(key):
            # GH#44051 exclude bool, which would return a 2d ndarray
            key = com.cast_scalar_indexer(key)
            return getitem(key)

        if isinstance(key, slice):
            # This case is separated from the conditional above to avoid
            # pessimization com.is_bool_indexer and ndim checks.
            return self._getitem_slice(key)

        if com.is_bool_indexer(key):
            # if we have list[bools, length=1e5] then doing this check+convert
            #  takes 166 µs + 2.1 ms and cuts the ndarray.__getitem__
            #  time below from 3.8 ms to 496 µs
            # if we already have ndarray[bool], the overhead is 1.4 µs or .25%
            if isinstance(getattr(key, "dtype", None), ExtensionDtype):
                key = key.to_numpy(dtype=bool, na_value=False)
            else:
                key = np.asarray(key, dtype=bool)

            if not isinstance(self.dtype, ExtensionDtype):
                if len(key) == 0 and len(key) != len(self):
                    warnings.warn(
                        "Using a boolean indexer with length 0 on an Index with "
                        "length greater than 0 is deprecated and will raise in a "
                        "future version.",
                        FutureWarning,
                        stacklevel=find_stack_level(),
                    )

        result = getitem(key)
        # Because we ruled out integer above, we always get an arraylike here
        if result.ndim > 1:
            disallow_ndim_indexing(result)

        # NB: Using _constructor._simple_new would break if MultiIndex
        #  didn't override __getitem__
        return self._constructor._simple_new(result, name=self._name)

    def _getitem_slice(self, slobj: slice) -> Self:
        """
        Fastpath for __getitem__ when we know we have a slice.
        """
        res = self._data[slobj]
        result = type(self)._simple_new(res, name=self._name, refs=self._references)
        if "_engine" in self._cache:
            reverse = slobj.step is not None and slobj.step < 0
            result._engine._update_from_sliced(self._engine, reverse=reverse)  # type: ignore[union-attr]

        return result

    @final
    def _can_hold_identifiers_and_holds_name(self, name) -> bool:
        """
        Faster check for ``name in self`` when we know `name` is a Python
        identifier (e.g. in NDFrame.__getattr__, which hits this to support
        . key lookup). For indexes that can't hold identifiers (everything
        but object & categorical) we just return False.

        https://github.com/pandas-dev/pandas/issues/19764
        """
        if (
            is_object_dtype(self.dtype)
            or is_string_dtype(self.dtype)
            or isinstance(self.dtype, CategoricalDtype)
        ):
            return name in self
        return False

    def append(self, other: Index | Sequence[Index]) -> Index:
        """
        Append a collection of Index options together.

        Parameters
        ----------
        other : Index or list/tuple of indices

        Returns
        -------
        Index

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.append(pd.Index([4]))
        Index([1, 2, 3, 4], dtype='int64')
        """
        to_concat = [self]

        if isinstance(other, (list, tuple)):
            to_concat += list(other)
        else:
            # error: Argument 1 to "append" of "list" has incompatible type
            # "Union[Index, Sequence[Index]]"; expected "Index"
            to_concat.append(other)  # type: ignore[arg-type]

        for obj in to_concat:
            if not isinstance(obj, Index):
                raise TypeError("all inputs must be Index")

        names = {obj.name for obj in to_concat}
        name = None if len(names) > 1 else self.name

        return self._concat(to_concat, name)

    def _concat(self, to_concat: list[Index], name: Hashable) -> Index:
        """
        Concatenate multiple Index objects.
        """
        to_concat_vals = [x._values for x in to_concat]

        result = concat_compat(to_concat_vals)

        return Index._with_infer(result, name=name)

    def putmask(self, mask, value) -> Index:
        """
        Return a new Index of the values set with the mask.

        Returns
        -------
        Index

        See Also
        --------
        numpy.ndarray.putmask : Changes elements of an array
            based on conditional and input values.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx2 = pd.Index([5, 6, 7])
        >>> idx1.putmask([True, False, False], idx2)
        Index([5, 2, 3], dtype='int64')
        """
        mask, noop = validate_putmask(self._values, mask)
        if noop:
            return self.copy()

        if self.dtype != object and is_valid_na_for_dtype(value, self.dtype):
            # e.g. None -> np.nan, see also Block._standardize_fill_value
            value = self._na_value

        try:
            converted = self._validate_fill_value(value)
        except (LossySetitemError, ValueError, TypeError) as err:
            if is_object_dtype(self.dtype):  # pragma: no cover
                raise err

            # See also: Block.coerce_to_target_dtype
            dtype = self._find_common_type_compat(value)
            return self.astype(dtype).putmask(mask, value)

        values = self._values.copy()

        if isinstance(values, np.ndarray):
            converted = setitem_datetimelike_compat(values, mask.sum(), converted)
            np.putmask(values, mask, converted)

        else:
            # Note: we use the original value here, not converted, as
            #  _validate_fill_value is not idempotent
            values._putmask(mask, value)

        return self._shallow_copy(values)

    def equals(self, other: Any) -> bool:
        """
        Determine if two Index object are equal.

        The things that are being compared are:

        * The elements inside the Index object.
        * The order of the elements inside the Index object.

        Parameters
        ----------
        other : Any
            The other object to compare against.

        Returns
        -------
        bool
            True if "other" is an Index and it has the same elements and order
            as the calling index; False otherwise.

        Examples
        --------
        >>> idx1 = pd.Index([1, 2, 3])
        >>> idx1
        Index([1, 2, 3], dtype='int64')
        >>> idx1.equals(pd.Index([1, 2, 3]))
        True

        The elements inside are compared

        >>> idx2 = pd.Index(["1", "2", "3"])
        >>> idx2
        Index(['1', '2', '3'], dtype='object')

        >>> idx1.equals(idx2)
        False

        The order is compared

        >>> ascending_idx = pd.Index([1, 2, 3])
        >>> ascending_idx
        Index([1, 2, 3], dtype='int64')
        >>> descending_idx = pd.Index([3, 2, 1])
        >>> descending_idx
        Index([3, 2, 1], dtype='int64')
        >>> ascending_idx.equals(descending_idx)
        False

        The dtype is *not* compared

        >>> int64_idx = pd.Index([1, 2, 3], dtype='int64')
        >>> int64_idx
        Index([1, 2, 3], dtype='int64')
        >>> uint64_idx = pd.Index([1, 2, 3], dtype='uint64')
        >>> uint64_idx
        Index([1, 2, 3], dtype='uint64')
        >>> int64_idx.equals(uint64_idx)
        True
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False

        if len(self) != len(other):
            # quickly return if the lengths are different
            return False

        if (
            isinstance(self.dtype, StringDtype)
            and self.dtype.na_value is np.nan
            and other.dtype != self.dtype
        ):
            # TODO(infer_string) can we avoid this special case?
            # special case for object behavior
            return other.equals(self.astype(object))

        if is_object_dtype(self.dtype) and not is_object_dtype(other.dtype):
            # if other is not object, use other's logic for coercion
            return other.equals(self)

        if isinstance(other, ABCMultiIndex):
            # d-level MultiIndex can equal d-tuple Index
            return other.equals(self)

        if isinstance(self._values, ExtensionArray):
            # Dispatch to the ExtensionArray's .equals method.
            if not isinstance(other, type(self)):
                return False

            earr = cast(ExtensionArray, self._data)
            return earr.equals(other._data)

        if isinstance(other.dtype, ExtensionDtype):
            # All EA-backed Index subclasses override equals
            return other.equals(self)

        return array_equivalent(self._values, other._values)

    @final
    def identical(self, other) -> bool:
        """
        Similar to equals, but checks that object attributes and types are also equal.

        Returns
        -------
        bool
            If two Index objects have equal elements and same type True,
            otherwise False.

        Examples
        --------
        >>> idx1 = pd.Index(['1', '2', '3'])
        >>> idx2 = pd.Index(['1', '2', '3'])
        >>> idx2.identical(idx1)
        True

        >>> idx1 = pd.Index(['1', '2', '3'], name="A")
        >>> idx2 = pd.Index(['1', '2', '3'], name="B")
        >>> idx2.identical(idx1)
        False
        """
        return (
            self.equals(other)
            and all(
                getattr(self, c, None) == getattr(other, c, None)
                for c in self._comparables
            )
            and type(self) == type(other)
            and self.dtype == other.dtype
        )

    @final
    def asof(self, label):
        """
        Return the label from the index, or, if not present, the previous one.

        Assuming that the index is sorted, return the passed index label if it
        is in the index, or return the previous index label if the passed one
        is not in the index.

        Parameters
        ----------
        label : object
            The label up to which the method returns the latest index label.

        Returns
        -------
        object
            The passed label if it is in the index. The previous label if the
            passed label is not in the sorted index or `NaN` if there is no
            such label.

        See Also
        --------
        Series.asof : Return the latest value in a Series up to the
            passed index.
        merge_asof : Perform an asof merge (similar to left join but it
            matches on nearest key rather than equal key).
        Index.get_loc : An `asof` is a thin wrapper around `get_loc`
            with method='pad'.

        Examples
        --------
        `Index.asof` returns the latest index label up to the passed label.

        >>> idx = pd.Index(['2013-12-31', '2014-01-02', '2014-01-03'])
        >>> idx.asof('2014-01-01')
        '2013-12-31'

        If the label is in the index, the method returns the passed label.

        >>> idx.asof('2014-01-02')
        '2014-01-02'

        If all of the labels in the index are later than the passed label,
        NaN is returned.

        >>> idx.asof('1999-01-02')
        nan

        If the index is not sorted, an error is raised.

        >>> idx_not_sorted = pd.Index(['2013-12-31', '2015-01-02',
        ...                            '2014-01-03'])
        >>> idx_not_sorted.asof('2013-12-31')
        Traceback (most recent call last):
        ValueError: index must be monotonic increasing or decreasing
        """
        self._searchsorted_monotonic(label)  # validate sortedness
        try:
            loc = self.get_loc(label)
        except (KeyError, TypeError):
            # KeyError -> No exact match, try for padded
            # TypeError -> passed e.g. non-hashable, fall through to get
            #  the tested exception message
            indexer = self.get_indexer([label], method="pad")
            if indexer.ndim > 1 or indexer.size > 1:
                raise TypeError("asof requires scalar valued input")
            loc = indexer.item()
            if loc == -1:
                return self._na_value
        else:
            if isinstance(loc, slice):
                loc = loc.indices(len(self))[-1]

        return self[loc]

    def asof_locs(
        self, where: Index, mask: npt.NDArray[np.bool_]
    ) -> npt.NDArray[np.intp]:
        """
        Return the locations (indices) of labels in the index.

        As in the :meth:`pandas.Index.asof`, if the label (a particular entry in
        ``where``) is not in the index, the latest index label up to the
        passed label is chosen and its index returned.

        If all of the labels in the index are later than a label in ``where``,
        -1 is returned.

        ``mask`` is used to ignore ``NA`` values in the index during calculation.

        Parameters
        ----------
        where : Index
            An Index consisting of an array of timestamps.
        mask : np.ndarray[bool]
            Array of booleans denoting where values in the original
            data are not ``NA``.

        Returns
        -------
        np.ndarray[np.intp]
            An array of locations (indices) of the labels from the index
            which correspond to the return values of :meth:`pandas.Index.asof`
            for every element in ``where``.

        See Also
        --------
        Index.asof : Return the label from the index, or, if not present, the
            previous one.

        Examples
        --------
        >>> idx = pd.date_range('2023-06-01', periods=3, freq='D')
        >>> where = pd.DatetimeIndex(['2023-05-30 00:12:00', '2023-06-01 00:00:00',
        ...                           '2023-06-02 23:59:59'])
        >>> mask = np.ones(3, dtype=bool)
        >>> idx.asof_locs(where, mask)
        array([-1,  0,  1])

        We can use ``mask`` to ignore certain values in the index during calculation.

        >>> mask[1] = False
        >>> idx.asof_locs(where, mask)
        array([-1,  0,  0])
        """
        # error: No overload variant of "searchsorted" of "ndarray" matches argument
        # types "Union[ExtensionArray, ndarray[Any, Any]]", "str"
        # TODO: will be fixed when ExtensionArray.searchsorted() is fixed
        locs = self._values[mask].searchsorted(
            where._values, side="right"  # type: ignore[call-overload]
        )
        locs = np.where(locs > 0, locs - 1, 0)

        result = np.arange(len(self), dtype=np.intp)[mask].take(locs)

        first_value = self._values[mask.argmax()]
        result[(locs == 0) & (where._values < first_value)] = -1

        return result

    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[False] = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self:
        ...

    @overload
    def sort_values(
        self,
        *,
        return_indexer: Literal[True],
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> tuple[Self, np.ndarray]:
        ...

    @overload
    def sort_values(
        self,
        *,
        return_indexer: bool = ...,
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> Self | tuple[Self, np.ndarray]:
        ...

    @deprecate_nonkeyword_arguments(
        version="3.0", allowed_args=["self"], name="sort_values"
    )
    def sort_values(
        self,
        return_indexer: bool = False,
        ascending: bool = True,
        na_position: NaPosition = "last",
        key: Callable | None = None,
    ) -> Self | tuple[Self, np.ndarray]:
        """
        Return a sorted copy of the index.

        Return a sorted copy of the index, and optionally return the indices
        that sorted the index itself.

        Parameters
        ----------
        return_indexer : bool, default False
            Should the indices that would sort the index be returned.
        ascending : bool, default True
            Should the index values be sorted in an ascending order.
        na_position : {'first' or 'last'}, default 'last'
            Argument 'first' puts NaNs at the beginning, 'last' puts NaNs at
            the end.
        key : callable, optional
            If not None, apply the key function to the index values
            before sorting. This is similar to the `key` argument in the
            builtin :meth:`sorted` function, with the notable difference that
            this `key` function should be *vectorized*. It should expect an
            ``Index`` and return an ``Index`` of the same shape.

        Returns
        -------
        sorted_index : pandas.Index
            Sorted copy of the index.
        indexer : numpy.ndarray, optional
            The indices that the index itself was sorted by.

        See Also
        --------
        Series.sort_values : Sort values of a Series.
        DataFrame.sort_values : Sort values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([10, 100, 1, 1000])
        >>> idx
        Index([10, 100, 1, 1000], dtype='int64')

        Sort values in ascending order (default behavior).

        >>> idx.sort_values()
        Index([1, 10, 100, 1000], dtype='int64')

        Sort values in descending order, and also get the indices `idx` was
        sorted by.

        >>> idx.sort_values(ascending=False, return_indexer=True)
        (Index([1000, 100, 10, 1], dtype='int64'), array([3, 1, 0, 2]))
        """
        if key is None and (
            (ascending and self.is_monotonic_increasing)
            or (not ascending and self.is_monotonic_decreasing)
        ):
            if return_indexer:
                indexer = np.arange(len(self), dtype=np.intp)
                return self.copy(), indexer
            else:
                return self.copy()

        # GH 35584. Sort missing values according to na_position kwarg
        # ignore na_position for MultiIndex
        if not isinstance(self, ABCMultiIndex):
            _as = nargsort(
                items=self, ascending=ascending, na_position=na_position, key=key
            )
        else:
            idx = cast(Index, ensure_key_mapped(self, key))
            _as = idx.argsort(na_position=na_position)
            if not ascending:
                _as = _as[::-1]

        sorted_index = self.take(_as)

        if return_indexer:
            return sorted_index, _as
        else:
            return sorted_index

    @final
    def sort(self, *args, **kwargs):
        """
        Use sort_values instead.
        """
        raise TypeError("cannot sort an Index object in-place, use sort_values instead")

    def shift(self, periods: int = 1, freq=None):
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or str, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.Index
            Shifted index.

        See Also
        --------
        Series.shift : Shift values of Series.

        Notes
        -----
        This method is only implemented for datetime-like index classes,
        i.e., DatetimeIndex, PeriodIndex and TimedeltaIndex.

        Examples
        --------
        Put the first 5 month starts of 2011 into an index.

        >>> month_starts = pd.date_range('1/1/2011', periods=5, freq='MS')
        >>> month_starts
        DatetimeIndex(['2011-01-01', '2011-02-01', '2011-03-01', '2011-04-01',
                       '2011-05-01'],
                      dtype='datetime64[ns]', freq='MS')

        Shift the index by 10 days.

        >>> month_starts.shift(10, freq='D')
        DatetimeIndex(['2011-01-11', '2011-02-11', '2011-03-11', '2011-04-11',
                       '2011-05-11'],
                      dtype='datetime64[ns]', freq=None)

        The default value of `freq` is the `freq` attribute of the index,
        which is 'MS' (month start) in this example.

        >>> month_starts.shift(10)
        DatetimeIndex(['2011-11-01', '2011-12-01', '2012-01-01', '2012-02-01',
                       '2012-03-01'],
                      dtype='datetime64[ns]', freq='MS')
        """
        raise NotImplementedError(
            f"This method is only implemented for DatetimeIndex, PeriodIndex and "
            f"TimedeltaIndex; Got type {type(self).__name__}"
        )

    def argsort(self, *args, **kwargs) -> npt.NDArray[np.intp]:
        """
        Return the integer indices that would sort the index.

        Parameters
        ----------
        *args
            Passed to `numpy.ndarray.argsort`.
        **kwargs
            Passed to `numpy.ndarray.argsort`.

        Returns
        -------
        np.ndarray[np.intp]
            Integer indices that would sort the index if used as
            an indexer.

        See Also
        --------
        numpy.argsort : Similar method for NumPy arrays.
        Index.sort_values : Return sorted copy of Index.

        Examples
        --------
        >>> idx = pd.Index(['b', 'a', 'd', 'c'])
        >>> idx
        Index(['b', 'a', 'd', 'c'], dtype='object')

        >>> order = idx.argsort()
        >>> order
        array([1, 0, 3, 2])

        >>> idx[order]
        Index(['a', 'b', 'c', 'd'], dtype='object')
        """
        # This works for either ndarray or EA, is overridden
        #  by RangeIndex, MultIIndex
        return self._data.argsort(*args, **kwargs)

    def _check_indexing_error(self, key):
        if not is_scalar(key):
            # if key is not a scalar, directly raise an error (the code below
            # would convert to numpy arrays and raise later any way) - GH29926
            raise InvalidIndexError(key)

    @cache_readonly
    def _should_fallback_to_positional(self) -> bool:
        """
        Should an integer key be treated as positional?
        """
        return self.inferred_type not in {
            "integer",
            "mixed-integer",
            "floating",
            "complex",
        }

    _index_shared_docs[
        "get_indexer_non_unique"
    ] = """
        Compute indexer and mask for new index given the current index.

        The indexer should be then used as an input to ndarray.take to align the
        current data to the new index.

        Parameters
        ----------
        target : %(target_klass)s

        Returns
        -------
        indexer : np.ndarray[np.intp]
            Integers from 0 to n - 1 indicating that the index at these
            positions matches the corresponding target values. Missing values
            in the target are marked by -1.
        missing : np.ndarray[np.intp]
            An indexer into the target of the values not found.
            These correspond to the -1 in the indexer array.

        Examples
        --------
        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])
        >>> index.get_indexer_non_unique(['b', 'b'])
        (array([1, 3, 4, 1, 3, 4]), array([], dtype=int64))

        In the example below there are no matched values.

        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])
        >>> index.get_indexer_non_unique(['q', 'r', 't'])
        (array([-1, -1, -1]), array([0, 1, 2]))

        For this reason, the returned ``indexer`` contains only integers equal to -1.
        It demonstrates that there's no match between the index and the ``target``
        values at these positions. The mask [0, 1, 2] in the return value shows that
        the first, second, and third elements are missing.

        Notice that the return value is a tuple contains two items. In the example
        below the first item is an array of locations in ``index``. The second
        item is a mask shows that the first and third elements are missing.

        >>> index = pd.Index(['c', 'b', 'a', 'b', 'b'])
        >>> index.get_indexer_non_unique(['f', 'b', 's'])
        (array([-1,  1,  3,  4, -1]), array([0, 2]))
        """

    @Appender(_index_shared_docs["get_indexer_non_unique"] % _index_doc_kwargs)
    def get_indexer_non_unique(
        self, target
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        target = self._maybe_cast_listlike_indexer(target)

        if not self._should_compare(target) and not self._should_partial_index(target):
            # _should_partial_index e.g. IntervalIndex with numeric scalars
            #  that can be matched to Interval scalars.
            return self._get_indexer_non_comparable(target, method=None, unique=False)

        pself, ptarget = self._maybe_downcast_for_indexing(target)
        if pself is not self or ptarget is not target:
            return pself.get_indexer_non_unique(ptarget)

        if self.dtype != target.dtype:
            # TODO: if object, could use infer_dtype to preempt costly
            #  conversion if still non-comparable?
            dtype = self._find_common_type_compat(target)

            this = self.astype(dtype, copy=False)
            that = target.astype(dtype, copy=False)
            return this.get_indexer_non_unique(that)

        # TODO: get_indexer has fastpaths for both Categorical-self and
        #  Categorical-target. Can we do something similar here?

        # Note: _maybe_downcast_for_indexing ensures we never get here
        #  with MultiIndex self and non-Multi target
        if self._is_multi and target._is_multi:
            engine = self._engine
            # Item "IndexEngine" of "Union[IndexEngine, ExtensionEngine]" has
            # no attribute "_extract_level_codes"
            tgt_values = engine._extract_level_codes(target)  # type: ignore[union-attr]
        else:
            tgt_values = target._get_engine_target()

        indexer, missing = self._engine.get_indexer_non_unique(tgt_values)
        return ensure_platform_int(indexer), ensure_platform_int(missing)

    @final
    def get_indexer_for(self, target) -> npt.NDArray[np.intp]:
        """
        Guaranteed return of an indexer even when non-unique.

        This dispatches to get_indexer or get_indexer_non_unique
        as appropriate.

        Returns
        -------
        np.ndarray[np.intp]
            List of indices.

        Examples
        --------
        >>> idx = pd.Index([np.nan, 'var1', np.nan])
        >>> idx.get_indexer_for([np.nan])
        array([0, 2])
        """
        if self._index_as_unique:
            return self.get_indexer(target)
        indexer, _ = self.get_indexer_non_unique(target)
        return indexer

    def _get_indexer_strict(self, key, axis_name: str_t) -> tuple[Index, np.ndarray]:
        """
        Analogue to get_indexer that raises if any elements are missing.
        """
        keyarr = key
        if not isinstance(keyarr, Index):
            keyarr = com.asarray_tuplesafe(keyarr)

        if self._index_as_unique:
            indexer = self.get_indexer_for(keyarr)
            keyarr = self.reindex(keyarr)[0]
        else:
            keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)

        self._raise_if_missing(keyarr, indexer, axis_name)

        keyarr = self.take(indexer)
        if isinstance(key, Index):
            # GH 42790 - Preserve name from an Index
            keyarr.name = key.name
        if lib.is_np_dtype(keyarr.dtype, "mM") or isinstance(
            keyarr.dtype, DatetimeTZDtype
        ):
            # DTI/TDI.take can infer a freq in some cases when we dont want one
            if isinstance(key, list) or (
                isinstance(key, type(self))
                # "Index" has no attribute "freq"
                and key.freq is None  # type: ignore[attr-defined]
            ):
                keyarr = keyarr._with_freq(None)

        return keyarr, indexer

    def _raise_if_missing(self, key, indexer, axis_name: str_t) -> None:
        """
        Check that indexer can be used to return a result.

        e.g. at least one element was found,
        unless the list of keys was actually empty.

        Parameters
        ----------
        key : list-like
            Targeted labels (only used to show correct error message).
        indexer: array-like of booleans
            Indices corresponding to the key,
            (with -1 indicating not found).
        axis_name : str

        Raises
        ------
        KeyError
            If at least one key was requested but none was found.
        """
        if len(key) == 0:
            return

        # Count missing values
        missing_mask = indexer < 0
        nmissing = missing_mask.sum()

        if nmissing:
            if nmissing == len(indexer):
                raise KeyError(f"None of [{key}] are in the [{axis_name}]")

            not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
            raise KeyError(f"{not_found} not in index")

    @overload
    def _get_indexer_non_comparable(
        self, target: Index, method, unique: Literal[True] = ...
    ) -> npt.NDArray[np.intp]:
        ...

    @overload
    def _get_indexer_non_comparable(
        self, target: Index, method, unique: Literal[False]
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        ...

    @overload
    def _get_indexer_non_comparable(
        self, target: Index, method, unique: bool = True
    ) -> npt.NDArray[np.intp] | tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        ...

    @final
    def _get_indexer_non_comparable(
        self, target: Index, method, unique: bool = True
    ) -> npt.NDArray[np.intp] | tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        Called from get_indexer or get_indexer_non_unique when the target
        is of a non-comparable dtype.

        For get_indexer lookups with method=None, get_indexer is an _equality_
        check, so non-comparable dtypes mean we will always have no matches.

        For get_indexer lookups with a method, get_indexer is an _inequality_
        check, so non-comparable dtypes mean we will always raise TypeError.

        Parameters
        ----------
        target : Index
        method : str or None
        unique : bool, default True
            * True if called from get_indexer.
            * False if called from get_indexer_non_unique.

        Raises
        ------
        TypeError
            If doing an inequality check, i.e. method is not None.
        """
        if method is not None:
            other_dtype = _unpack_nested_dtype(target)
            raise TypeError(f"Cannot compare dtypes {self.dtype} and {other_dtype}")

        no_matches = -1 * np.ones(target.shape, dtype=np.intp)
        if unique:
            # This is for get_indexer
            return no_matches
        else:
            # This is for get_indexer_non_unique
            missing = np.arange(len(target), dtype=np.intp)
            return no_matches, missing

    @property
    def _index_as_unique(self) -> bool:
        """
        Whether we should treat this as unique for the sake of
        get_indexer vs get_indexer_non_unique.

        For IntervalIndex compat.
        """
        return self.is_unique

    _requires_unique_msg = "Reindexing only valid with uniquely valued Index objects"

    @final
    def _maybe_downcast_for_indexing(self, other: Index) -> tuple[Index, Index]:
        """
        When dealing with an object-dtype Index and a non-object Index, see
        if we can upcast the object-dtype one to improve performance.
        """

        if isinstance(self, ABCDatetimeIndex) and isinstance(other, ABCDatetimeIndex):
            if (
                self.tz is not None
                and other.tz is not None
                and not tz_compare(self.tz, other.tz)
            ):
                # standardize on UTC
                return self.tz_convert("UTC"), other.tz_convert("UTC")

        elif self.inferred_type == "date" and isinstance(other, ABCDatetimeIndex):
            try:
                return type(other)(self), other
            except OutOfBoundsDatetime:
                return self, other
        elif self.inferred_type == "timedelta" and isinstance(other, ABCTimedeltaIndex):
            # TODO: we dont have tests that get here
            return type(other)(self), other

        elif self.dtype.kind == "u" and other.dtype.kind == "i":
            # GH#41873
            if other.min() >= 0:
                # lookup min as it may be cached
                # TODO: may need itemsize check if we have non-64-bit Indexes
                return self, other.astype(self.dtype)

        elif self._is_multi and not other._is_multi:
            try:
                # "Type[Index]" has no attribute "from_tuples"
                other = type(self).from_tuples(other)  # type: ignore[attr-defined]
            except (TypeError, ValueError):
                # let's instead try with a straight Index
                self = Index(self._values)

        if not is_object_dtype(self.dtype) and is_object_dtype(other.dtype):
            # Reverse op so we dont need to re-implement on the subclasses
            other, self = other._maybe_downcast_for_indexing(self)

        return self, other

    @final
    def _find_common_type_compat(self, target) -> DtypeObj:
        """
        Implementation of find_common_type that adjusts for Index-specific
        special cases.
        """
        target_dtype, _ = infer_dtype_from(target)

        # special case: if one dtype is uint64 and the other a signed int, return object
        # See https://github.com/pandas-dev/pandas/issues/26778 for discussion
        # Now it's:
        # * float | [u]int -> float
        # * uint64 | signed int  -> object
        # We may change union(float | [u]int) to go to object.
        if self.dtype == "uint64" or target_dtype == "uint64":
            if is_signed_integer_dtype(self.dtype) or is_signed_integer_dtype(
                target_dtype
            ):
                return _dtype_obj

        dtype = find_result_type(self.dtype, target)
        dtype = common_dtype_categorical_compat([self, target], dtype)
        return dtype

    @final
    def _should_compare(self, other: Index) -> bool:
        """
        Check if `self == other` can ever have non-False entries.
        """

        # NB: we use inferred_type rather than is_bool_dtype to catch
        #  object_dtype_of_bool and categorical[object_dtype_of_bool] cases
        if (
            other.inferred_type == "boolean" and is_any_real_numeric_dtype(self.dtype)
        ) or (
            self.inferred_type == "boolean" and is_any_real_numeric_dtype(other.dtype)
        ):
            # GH#16877 Treat boolean labels passed to a numeric index as not
            #  found. Without this fix False and True would be treated as 0 and 1
            #  respectively.
            return False

        dtype = _unpack_nested_dtype(other)
        return (
            self._is_comparable_dtype(dtype)
            or is_object_dtype(dtype)
            or is_string_dtype(dtype)
        )

    def _is_comparable_dtype(self, dtype: DtypeObj) -> bool:
        """
        Can we compare values of the given dtype to our own?
        """
        if self.dtype.kind == "b":
            return dtype.kind == "b"
        elif is_numeric_dtype(self.dtype):
            return is_numeric_dtype(dtype)
        # TODO: this was written assuming we only get here with object-dtype,
        #  which is no longer correct. Can we specialize for EA?
        return True

    @final
    def groupby(self, values) -> PrettyDict[Hashable, np.ndarray]:
        """
        Group the index labels by a given array of values.

        Parameters
        ----------
        values : array
            Values used to determine the groups.

        Returns
        -------
        dict
            {group name -> group labels}
        """
        # TODO: if we are a MultiIndex, we can do better
        # that converting to tuples
        if isinstance(values, ABCMultiIndex):
            values = values._values
        values = Categorical(values)
        result = values._reverse_indexer()

        # map to the label
        result = {k: self.take(v) for k, v in result.items()}

        return PrettyDict(result)

    def map(self, mapper, na_action: Literal["ignore"] | None = None):
        """
        Map values using an input mapping or function.

        Parameters
        ----------
        mapper : function, dict, or Series
            Mapping correspondence.
        na_action : {None, 'ignore'}
            If 'ignore', propagate NA values, without passing them to the
            mapping correspondence.

        Returns
        -------
        Union[Index, MultiIndex]
            The output of the mapping function applied to the index.
            If the function returns a tuple with more than one element
            a MultiIndex will be returned.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx.map({1: 'a', 2: 'b', 3: 'c'})
        Index(['a', 'b', 'c'], dtype='object')

        Using `map` with a function:

        >>> idx = pd.Index([1, 2, 3])
        >>> idx.map('I am a {}'.format)
        Index(['I am a 1', 'I am a 2', 'I am a 3'], dtype='object')

        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.map(lambda x: x.upper())
        Index(['A', 'B', 'C'], dtype='object')
        """
        from pandas.core.indexes.multi import MultiIndex

        new_values = self._map_values(mapper, na_action=na_action)

        # we can return a MultiIndex
        if new_values.size and isinstance(new_values[0], tuple):
            if isinstance(self, MultiIndex):
                names = self.names
            elif self.name:
                names = [self.name] * len(new_values[0])
            else:
                names = None
            return MultiIndex.from_tuples(new_values, names=names)

        dtype = None
        if not new_values.size:
            # empty
            dtype = self.dtype

        # e.g. if we are floating and new_values is all ints, then we
        #  don't want to cast back to floating.  But if we are UInt64
        #  and new_values is all ints, we want to try.
        same_dtype = lib.infer_dtype(new_values, skipna=False) == self.inferred_type
        if same_dtype:
            new_values = maybe_cast_pointwise_result(
                new_values, self.dtype, same_dtype=same_dtype
            )

        return Index._with_infer(new_values, dtype=dtype, copy=False, name=self.name)

    # TODO: De-duplicate with map, xref GH#32349
    @final
    def _transform_index(self, func, *, level=None) -> Index:
        """
        Apply function to all values found in index.

        This includes transforming multiindex entries separately.
        Only apply function to one level of the MultiIndex if level is specified.
        """
        if isinstance(self, ABCMultiIndex):
            values = [
                self.get_level_values(i).map(func)
                if i == level or level is None
                else self.get_level_values(i)
                for i in range(self.nlevels)
            ]
            return type(self).from_arrays(values)
        else:
            items = [func(x) for x in self]
            return Index(items, name=self.name, tupleize_cols=False)

    def isin(self, values, level=None) -> npt.NDArray[np.bool_]:
        """
        Return a boolean array where the index values are in `values`.

        Compute boolean array of whether each index value is found in the
        passed set of values. The length of the returned boolean array matches
        the length of the index.

        Parameters
        ----------
        values : set or list-like
            Sought values.
        level : str or int, optional
            Name or position of the index level to use (if the index is a
            `MultiIndex`).

        Returns
        -------
        np.ndarray[bool]
            NumPy array of boolean values.

        See Also
        --------
        Series.isin : Same for Series.
        DataFrame.isin : Same method for DataFrames.

        Notes
        -----
        In the case of `MultiIndex` you must either specify `values` as a
        list-like object containing tuples that are the same length as the
        number of levels, or specify `level`. Otherwise it will raise a
        ``ValueError``.

        If `level` is specified:

        - if it is the name of one *and only one* index level, use that level;
        - otherwise it should be a number indicating level position.

        Examples
        --------
        >>> idx = pd.Index([1,2,3])
        >>> idx
        Index([1, 2, 3], dtype='int64')

        Check whether each index value in a list of values.

        >>> idx.isin([1, 4])
        array([ True, False, False])

        >>> midx = pd.MultiIndex.from_arrays([[1,2,3],
        ...                                  ['red', 'blue', 'green']],
        ...                                  names=('number', 'color'))
        >>> midx
        MultiIndex([(1,   'red'),
                    (2,  'blue'),
                    (3, 'green')],
                   names=['number', 'color'])

        Check whether the strings in the 'color' level of the MultiIndex
        are in a list of colors.

        >>> midx.isin(['red', 'orange', 'yellow'], level='color')
        array([ True, False, False])

        To check across the levels of a MultiIndex, pass a list of tuples:

        >>> midx.isin([(1, 'red'), (3, 'red')])
        array([ True, False, False])
        """
        if level is not None:
            self._validate_index_level(level)
        return algos.isin(self._values, values)

    def _get_string_slice(self, key: str_t):
        # this is for partial string indexing,
        # overridden in DatetimeIndex, TimedeltaIndex and PeriodIndex
        raise NotImplementedError

    def slice_indexer(
        self,
        start: Hashable | None = None,
        end: Hashable | None = None,
        step: int | None = None,
    ) -> slice:
        """
        Compute the slice indexer for input labels and step.

        Index needs to be ordered and unique.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, default None

        Returns
        -------
        slice

        Raises
        ------
        KeyError : If key does not exist, or key is not unique and index is
            not ordered.

        Notes
        -----
        This function assumes that the data is sorted, so use at your own peril

        Examples
        --------
        This is a method on all index types. For example you can do:

        >>> idx = pd.Index(list('abcd'))
        >>> idx.slice_indexer(start='b', end='c')
        slice(1, 3, None)

        >>> idx = pd.MultiIndex.from_arrays([list('abcd'), list('efgh')])
        >>> idx.slice_indexer(start='b', end=('c', 'g'))
        slice(1, 3, None)
        """
        start_slice, end_slice = self.slice_locs(start, end, step=step)

        # return a slice
        if not is_scalar(start_slice):
            raise AssertionError("Start slice bound is non-scalar")
        if not is_scalar(end_slice):
            raise AssertionError("End slice bound is non-scalar")

        return slice(start_slice, end_slice, step)

    def _maybe_cast_indexer(self, key):
        """
        If we have a float key and are not a floating index, then try to cast
        to an int if equivalent.
        """
        return key

    def _maybe_cast_listlike_indexer(self, target) -> Index:
        """
        Analogue to maybe_cast_indexer for get_indexer instead of get_loc.
        """
        target_index = ensure_index(target)
        if (
            not hasattr(target, "dtype")
            and self.dtype == object
            and target_index.dtype == "string"
        ):
            # If we started with a list-like, avoid inference to string dtype if self
            # is object dtype (coercing to string dtype will alter the missing values)
            target_index = Index(target, dtype=self.dtype)
        return target_index

    @final
    def _validate_indexer(
        self,
        form: Literal["positional", "slice"],
        key,
        kind: Literal["getitem", "iloc"],
    ) -> None:
        """
        If we are positional indexer, validate that we have appropriate
        typed bounds must be an integer.
        """
        if not lib.is_int_or_none(key):
            self._raise_invalid_indexer(form, key)

    def _maybe_cast_slice_bound(self, label, side: str_t):
        """
        This function should be overloaded in subclasses that allow non-trivial
        casting on label-slice bounds, e.g. datetime-like indices allowing
        strings containing formatted datetimes.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """

        # We are a plain index here (sub-class override this method if they
        # wish to have special treatment for floats/ints, e.g. datetimelike Indexes

        if is_numeric_dtype(self.dtype):
            return self._maybe_cast_indexer(label)

        # reject them, if index does not contain label
        if (is_float(label) or is_integer(label)) and label not in self:
            self._raise_invalid_indexer("slice", label)

        return label

    def _searchsorted_monotonic(self, label, side: Literal["left", "right"] = "left"):
        if self.is_monotonic_increasing:
            return self.searchsorted(label, side=side)
        elif self.is_monotonic_decreasing:
            # np.searchsorted expects ascending sort order, have to reverse
            # everything for it to work (element ordering, search side and
            # resulting value).
            pos = self[::-1].searchsorted(
                label, side="right" if side == "left" else "left"
            )
            return len(self) - pos

        raise ValueError("index must be monotonic increasing or decreasing")

    def get_slice_bound(self, label, side: Literal["left", "right"]) -> int:
        """
        Calculate slice bound that corresponds to given label.

        Returns leftmost (one-past-the-rightmost if ``side=='right'``) position
        of given label.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        int
            Index of label.

        See Also
        --------
        Index.get_loc : Get integer location, slice or boolean mask for requested
            label.

        Examples
        --------
        >>> idx = pd.RangeIndex(5)
        >>> idx.get_slice_bound(3, 'left')
        3

        >>> idx.get_slice_bound(3, 'right')
        4

        If ``label`` is non-unique in the index, an error will be raised.

        >>> idx_duplicate = pd.Index(['a', 'b', 'a', 'c', 'd'])
        >>> idx_duplicate.get_slice_bound('a', 'left')
        Traceback (most recent call last):
        KeyError: Cannot get left slice bound for non-unique label: 'a'
        """

        if side not in ("left", "right"):
            raise ValueError(
                "Invalid value for side kwarg, must be either "
                f"'left' or 'right': {side}"
            )

        original_label = label

        # For datetime indices label may be a string that has to be converted
        # to datetime boundary according to its resolution.
        label = self._maybe_cast_slice_bound(label, side)

        # we need to look up the label
        try:
            slc = self.get_loc(label)
        except KeyError as err:
            try:
                return self._searchsorted_monotonic(label, side)
            except ValueError:
                # raise the original KeyError
                raise err

        if isinstance(slc, np.ndarray):
            # get_loc may return a boolean array, which
            # is OK as long as they are representable by a slice.
            assert is_bool_dtype(slc.dtype)
            slc = lib.maybe_booleans_to_slice(slc.view("u1"))
            if isinstance(slc, np.ndarray):
                raise KeyError(
                    f"Cannot get {side} slice bound for non-unique "
                    f"label: {repr(original_label)}"
                )

        if isinstance(slc, slice):
            if side == "left":
                return slc.start
            else:
                return slc.stop
        else:
            if side == "right":
                return slc + 1
            else:
                return slc

    def slice_locs(self, start=None, end=None, step=None) -> tuple[int, int]:
        """
        Compute slice locations for input labels.

        Parameters
        ----------
        start : label, default None
            If None, defaults to the beginning.
        end : label, default None
            If None, defaults to the end.
        step : int, defaults None
            If None, defaults to 1.

        Returns
        -------
        tuple[int, int]

        See Also
        --------
        Index.get_loc : Get location for a single label.

        Notes
        -----
        This method only works if the index is monotonic or unique.

        Examples
        --------
        >>> idx = pd.Index(list('abcd'))
        >>> idx.slice_locs(start='b', end='c')
        (1, 3)
        """
        inc = step is None or step >= 0

        if not inc:
            # If it's a reverse slice, temporarily swap bounds.
            start, end = end, start

        # GH 16785: If start and end happen to be date strings with UTC offsets
        # attempt to parse and check that the offsets are the same
        if isinstance(start, (str, datetime)) and isinstance(end, (str, datetime)):
            try:
                ts_start = Timestamp(start)
                ts_end = Timestamp(end)
            except (ValueError, TypeError):
                pass
            else:
                if not tz_compare(ts_start.tzinfo, ts_end.tzinfo):
                    raise ValueError("Both dates must have the same UTC offset")

        start_slice = None
        if start is not None:
            start_slice = self.get_slice_bound(start, "left")
        if start_slice is None:
            start_slice = 0

        end_slice = None
        if end is not None:
            end_slice = self.get_slice_bound(end, "right")
        if end_slice is None:
            end_slice = len(self)

        if not inc:
            # Bounds at this moment are swapped, swap them back and shift by 1.
            #
            # slice_locs('B', 'A', step=-1): s='B', e='A'
            #
            #              s='A'                 e='B'
            # AFTER SWAP:    |                     |
            #                v ------------------> V
            #           -----------------------------------
            #           | | |A|A|A|A| | | | | |B|B| | | | |
            #           -----------------------------------
            #              ^ <------------------ ^
            # SHOULD BE:   |                     |
            #           end=s-1              start=e-1
            #
            end_slice, start_slice = start_slice - 1, end_slice - 1

            # i == -1 triggers ``len(self) + i`` selection that points to the
            # last element, not before-the-first one, subtracting len(self)
            # compensates that.
            if end_slice == -1:
                end_slice -= len(self)
            if start_slice == -1:
                start_slice -= len(self)

        return start_slice, end_slice

    def delete(self, loc) -> Self:
        """
        Make new Index with passed location(-s) deleted.

        Parameters
        ----------
        loc : int or list of int
            Location of item(-s) which will be deleted.
            Use a list of locations to delete more than one value at the same time.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        See Also
        --------
        numpy.delete : Delete any rows and column from NumPy array (ndarray).

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.delete(1)
        Index(['a', 'c'], dtype='object')

        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.delete([0, 2])
        Index(['b'], dtype='object')
        """
        values = self._values
        res_values: ArrayLike
        if isinstance(values, np.ndarray):
            # TODO(__array_function__): special casing will be unnecessary
            res_values = np.delete(values, loc)
        else:
            res_values = values.delete(loc)

        # _constructor so RangeIndex-> Index with an int64 dtype
        return self._constructor._simple_new(res_values, name=self.name)

    def insert(self, loc: int, item) -> Index:
        """
        Make new Index inserting new item at location.

        Follows Python numpy.insert semantics for negative values.

        Parameters
        ----------
        loc : int
        item : object

        Returns
        -------
        Index

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.insert(1, 'x')
        Index(['a', 'x', 'b', 'c'], dtype='object')
        """
        item = lib.item_from_zerodim(item)
        if is_valid_na_for_dtype(item, self.dtype) and self.dtype != object:
            item = self._na_value

        arr = self._values

        try:
            if isinstance(arr, ExtensionArray):
                res_values = arr.insert(loc, item)
                return type(self)._simple_new(res_values, name=self.name)
            else:
                item = self._validate_fill_value(item)
        except (TypeError, ValueError, LossySetitemError):
            # e.g. trying to insert an integer into a DatetimeIndex
            #  We cannot keep the same dtype, so cast to the (often object)
            #  minimal shared dtype before doing the insert.
            dtype = self._find_common_type_compat(item)
            if dtype == self.dtype:
                # EA's might run into recursion errors if loc is invalid
                raise
            return self.astype(dtype).insert(loc, item)

        if arr.dtype != object or not isinstance(
            item, (tuple, np.datetime64, np.timedelta64)
        ):
            # with object-dtype we need to worry about numpy incorrectly casting
            # dt64/td64 to integer, also about treating tuples as sequences
            # special-casing dt64/td64 https://github.com/numpy/numpy/issues/12550
            casted = arr.dtype.type(item)
            new_values = np.insert(arr, loc, casted)

        else:
            # error: No overload variant of "insert" matches argument types
            # "ndarray[Any, Any]", "int", "None"
            new_values = np.insert(arr, loc, None)  # type: ignore[call-overload]
            loc = loc if loc >= 0 else loc - 1
            new_values[loc] = item

        out = Index._with_infer(new_values, name=self.name)
        if (
            using_string_dtype()
            and is_string_dtype(out.dtype)
            and new_values.dtype == object
        ):
            out = out.astype(new_values.dtype)
        if self.dtype == object and out.dtype != object:
            # GH#51363
            warnings.warn(
                "The behavior of Index.insert with object-dtype is deprecated, "
                "in a future version this will return an object-dtype Index "
                "instead of inferring a non-object dtype. To retain the old "
                "behavior, do `idx.insert(loc, item).infer_objects(copy=False)`",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return out

    def drop(
        self,
        labels: Index | np.ndarray | Iterable[Hashable],
        errors: IgnoreRaise = "raise",
    ) -> Index:
        """
        Make new Index with passed list of labels deleted.

        Parameters
        ----------
        labels : array-like or scalar
        errors : {'ignore', 'raise'}, default 'raise'
            If 'ignore', suppress error and existing labels are dropped.

        Returns
        -------
        Index
            Will be same type as self, except for RangeIndex.

        Raises
        ------
        KeyError
            If not all of the labels are found in the selected axis

        Examples
        --------
        >>> idx = pd.Index(['a', 'b', 'c'])
        >>> idx.drop(['a'])
        Index(['b', 'c'], dtype='object')
        """
        if not isinstance(labels, Index):
            # avoid materializing e.g. RangeIndex
            arr_dtype = "object" if self.dtype == "object" else None
            labels = com.index_labels_to_array(labels, dtype=arr_dtype)

        indexer = self.get_indexer_for(labels)
        mask = indexer == -1
        if mask.any():
            if errors != "ignore":
                raise KeyError(f"{labels[mask].tolist()} not found in axis")
            indexer = indexer[~mask]
        return self.delete(indexer)

    @final
    def infer_objects(self, copy: bool = True) -> Index:
        """
        If we have an object dtype, try to infer a non-object dtype.

        Parameters
        ----------
        copy : bool, default True
            Whether to make a copy in cases where no inference occurs.
        """
        if self._is_multi:
            raise NotImplementedError(
                "infer_objects is not implemented for MultiIndex. "
                "Use index.to_frame().infer_objects() instead."
            )
        if self.dtype != object:
            return self.copy() if copy else self

        values = self._values
        values = cast("npt.NDArray[np.object_]", values)
        res_values = lib.maybe_convert_objects(
            values,
            convert_non_numeric=True,
        )
        if copy and res_values is values:
            return self.copy()
        result = Index(res_values, name=self.name)
        if not copy and res_values is values and self._references is not None:
            result._references = self._references
            result._references.add_index_reference(result)
        return result

    @final
    def diff(self, periods: int = 1) -> Index:
        """
        Computes the difference between consecutive values in the Index object.

        If periods is greater than 1, computes the difference between values that
        are `periods` number of positions apart.

        Parameters
        ----------
        periods : int, optional
            The number of positions between the current and previous
            value to compute the difference with. Default is 1.

        Returns
        -------
        Index
            A new Index object with the computed differences.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.Index([10, 20, 30, 40, 50])
        >>> idx.diff()
        Index([nan, 10.0, 10.0, 10.0, 10.0], dtype='float64')

        """
        return Index(self.to_series().diff(periods))

    @final
    def round(self, decimals: int = 0) -> Self:
        """
        Round each value in the Index to the given number of decimals.

        Parameters
        ----------
        decimals : int, optional
            Number of decimal places to round to. If decimals is negative,
            it specifies the number of positions to the left of the decimal point.

        Returns
        -------
        Index
            A new Index with the rounded values.

        Examples
        --------
        >>> import pandas as pd
        >>> idx = pd.Index([10.1234, 20.5678, 30.9123, 40.4567, 50.7890])
        >>> idx.round(decimals=2)
        Index([10.12, 20.57, 30.91, 40.46, 50.79], dtype='float64')

        """
        return self._constructor(self.to_series().round(decimals))

    # --------------------------------------------------------------------
    # Generated Arithmetic, Comparison, and Unary Methods

    def _cmp_method(self, other, op):
        """
        Wrapper used to dispatch comparison operations.
        """
        if self.is_(other):
            # fastpath
            if op in {operator.eq, operator.le, operator.ge}:
                arr = np.ones(len(self), dtype=bool)
                if self._can_hold_na and not isinstance(self, ABCMultiIndex):
                    # TODO: should set MultiIndex._can_hold_na = False?
                    arr[self.isna()] = False
                return arr
            elif op is operator.ne:
                arr = np.zeros(len(self), dtype=bool)
                if self._can_hold_na and not isinstance(self, ABCMultiIndex):
                    arr[self.isna()] = True
                return arr

        if isinstance(other, (np.ndarray, Index, ABCSeries, ExtensionArray)) and len(
            self
        ) != len(other):
            raise ValueError("Lengths must match to compare")

        if not isinstance(other, ABCMultiIndex):
            other = extract_array(other, extract_numpy=True)
        else:
            other = np.asarray(other)

        if is_object_dtype(self.dtype) and isinstance(other, ExtensionArray):
            # e.g. PeriodArray, Categorical
            result = op(self._values, other)

        elif isinstance(self._values, ExtensionArray):
            result = op(self._values, other)

        elif is_object_dtype(self.dtype) and not isinstance(self, ABCMultiIndex):
            # don't pass MultiIndex
            result = ops.comp_method_OBJECT_ARRAY(op, self._values, other)

        else:
            result = ops.comparison_op(self._values, other, op)

        return result

    @final
    def _logical_method(self, other, op):
        res_name = ops.get_op_result_name(self, other)

        lvalues = self._values
        rvalues = extract_array(other, extract_numpy=True, extract_range=True)

        res_values = ops.logical_op(lvalues, rvalues, op)
        return self._construct_result(res_values, name=res_name)

    @final
    def _construct_result(self, result, name):
        if isinstance(result, tuple):
            return (
                Index(result[0], name=name, dtype=result[0].dtype),
                Index(result[1], name=name, dtype=result[1].dtype),
            )
        return Index(result, name=name, dtype=result.dtype)

    def _arith_method(self, other, op):
        if (
            isinstance(other, Index)
            and is_object_dtype(other.dtype)
            and type(other) is not Index
        ):
            # We return NotImplemented for object-dtype index *subclasses* so they have
            # a chance to implement ops before we unwrap them.
            # See https://github.com/pandas-dev/pandas/issues/31109
            return NotImplemented

        return super()._arith_method(other, op)

    @final
    def _unary_method(self, op):
        result = op(self._values)
        return Index(result, name=self.name)

    def __abs__(self) -> Index:
        return self._unary_method(operator.abs)

    def __neg__(self) -> Index:
        return self._unary_method(operator.neg)

    def __pos__(self) -> Index:
        return self._unary_method(operator.pos)

    def __invert__(self) -> Index:
        # GH#8875
        return self._unary_method(operator.inv)

    # --------------------------------------------------------------------
    # Reductions

    def any(self, *args, **kwargs):
        """
        Return whether any element is Truthy.

        Parameters
        ----------
        *args
            Required for compatibility with numpy.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        bool or array-like (if axis is specified)
            A single element array-like may be converted to bool.

        See Also
        --------
        Index.all : Return whether all elements are True.
        Series.all : Return whether all elements are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        >>> index = pd.Index([0, 1, 2])
        >>> index.any()
        True

        >>> index = pd.Index([0, 0, 0])
        >>> index.any()
        False
        """
        nv.validate_any(args, kwargs)
        self._maybe_disable_logical_methods("any")
        vals = self._values
        if not isinstance(vals, np.ndarray):
            # i.e. EA, call _reduce instead of "any" to get TypeError instead
            #  of AttributeError
            return vals._reduce("any")
        return np.any(vals)

    def all(self, *args, **kwargs):
        """
        Return whether all elements are Truthy.

        Parameters
        ----------
        *args
            Required for compatibility with numpy.
        **kwargs
            Required for compatibility with numpy.

        Returns
        -------
        bool or array-like (if axis is specified)
            A single element array-like may be converted to bool.

        See Also
        --------
        Index.any : Return whether any element in an Index is True.
        Series.any : Return whether any element in a Series is True.
        Series.all : Return whether all elements in a Series are True.

        Notes
        -----
        Not a Number (NaN), positive infinity and negative infinity
        evaluate to True because these are not equal to zero.

        Examples
        --------
        True, because nonzero integers are considered True.

        >>> pd.Index([1, 2, 3]).all()
        True

        False, because ``0`` is considered False.

        >>> pd.Index([0, 1, 2]).all()
        False
        """
        nv.validate_all(args, kwargs)
        self._maybe_disable_logical_methods("all")
        vals = self._values
        if not isinstance(vals, np.ndarray):
            # i.e. EA, call _reduce instead of "all" to get TypeError instead
            #  of AttributeError
            return vals._reduce("all")
        return np.all(vals)

    @final
    def _maybe_disable_logical_methods(self, opname: str_t) -> None:
        """
        raise if this Index subclass does not support any or all.
        """
        if (
            isinstance(self, ABCMultiIndex)
            # TODO(3.0): PeriodArray and DatetimeArray any/all will raise,
            #  so checking needs_i8_conversion will be unnecessary
            or (needs_i8_conversion(self.dtype) and self.dtype.kind != "m")
        ):
            # This call will raise
            make_invalid_op(opname)(self)

    @Appender(IndexOpsMixin.argmin.__doc__)
    def argmin(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        nv.validate_argmin(args, kwargs)
        nv.validate_minmax_axis(axis)

        if not self._is_multi and self.hasnans:
            # Take advantage of cache
            mask = self._isnan
            if not skipna or mask.all():
                warnings.warn(
                    f"The behavior of {type(self).__name__}.argmax/argmin "
                    "with skipna=False and NAs, or with all-NAs is deprecated. "
                    "In a future version this will raise ValueError.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                return -1
        return super().argmin(skipna=skipna)

    @Appender(IndexOpsMixin.argmax.__doc__)
    def argmax(self, axis=None, skipna: bool = True, *args, **kwargs) -> int:
        nv.validate_argmax(args, kwargs)
        nv.validate_minmax_axis(axis)

        if not self._is_multi and self.hasnans:
            # Take advantage of cache
            mask = self._isnan
            if not skipna or mask.all():
                warnings.warn(
                    f"The behavior of {type(self).__name__}.argmax/argmin "
                    "with skipna=False and NAs, or with all-NAs is deprecated. "
                    "In a future version this will raise ValueError.",
                    FutureWarning,
                    stacklevel=find_stack_level(),
                )
                return -1
        return super().argmax(skipna=skipna)

    def min(self, axis=None, skipna: bool = True, *args, **kwargs):
        """
        Return the minimum value of the Index.

        Parameters
        ----------
        axis : {None}
            Dummy argument for consistency with Series.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        scalar
            Minimum value.

        See Also
        --------
        Index.max : Return the maximum value of the object.
        Series.min : Return the minimum value in a Series.
        DataFrame.min : Return the minimum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.min()
        1

        >>> idx = pd.Index(['c', 'b', 'a'])
        >>> idx.min()
        'a'

        For a MultiIndex, the minimum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])
        >>> idx.min()
        ('a', 1)
        """
        nv.validate_min(args, kwargs)
        nv.validate_minmax_axis(axis)

        if not len(self):
            return self._na_value

        if len(self) and self.is_monotonic_increasing:
            # quick check
            first = self[0]
            if not isna(first):
                return first

        if not self._is_multi and self.hasnans:
            # Take advantage of cache
            mask = self._isnan
            if not skipna or mask.all():
                return self._na_value

        if not self._is_multi and not isinstance(self._values, np.ndarray):
            return self._values._reduce(name="min", skipna=skipna)

        return nanops.nanmin(self._values, skipna=skipna)

    def max(self, axis=None, skipna: bool = True, *args, **kwargs):
        """
        Return the maximum value of the Index.

        Parameters
        ----------
        axis : int, optional
            For compatibility with NumPy. Only 0 or None are allowed.
        skipna : bool, default True
            Exclude NA/null values when showing the result.
        *args, **kwargs
            Additional arguments and keywords for compatibility with NumPy.

        Returns
        -------
        scalar
            Maximum value.

        See Also
        --------
        Index.min : Return the minimum value in an Index.
        Series.max : Return the maximum value in a Series.
        DataFrame.max : Return the maximum values in a DataFrame.

        Examples
        --------
        >>> idx = pd.Index([3, 2, 1])
        >>> idx.max()
        3

        >>> idx = pd.Index(['c', 'b', 'a'])
        >>> idx.max()
        'c'

        For a MultiIndex, the maximum is determined lexicographically.

        >>> idx = pd.MultiIndex.from_product([('a', 'b'), (2, 1)])
        >>> idx.max()
        ('b', 2)
        """

        nv.validate_max(args, kwargs)
        nv.validate_minmax_axis(axis)

        if not len(self):
            return self._na_value

        if len(self) and self.is_monotonic_increasing:
            # quick check
            last = self[-1]
            if not isna(last):
                return last

        if not self._is_multi and self.hasnans:
            # Take advantage of cache
            mask = self._isnan
            if not skipna or mask.all():
                return self._na_value

        if not self._is_multi and not isinstance(self._values, np.ndarray):
            return self._values._reduce(name="max", skipna=skipna)

        return nanops.nanmax(self._values, skipna=skipna)

    # --------------------------------------------------------------------

    @final
    @property
    def shape(self) -> Shape:
        """
        Return a tuple of the shape of the underlying data.

        Examples
        --------
        >>> idx = pd.Index([1, 2, 3])
        >>> idx
        Index([1, 2, 3], dtype='int64')
        >>> idx.shape
        (3,)
        """
        # See GH#27775, GH#27384 for history/reasoning in how this is defined.
        return (len(self),)


def ensure_index_from_sequences(sequences, names=None) -> Index:
    """
    Construct an index from sequences of data.

    A single sequence returns an Index. Many sequences returns a
    MultiIndex.

    Parameters
    ----------
    sequences : sequence of sequences
    names : sequence of str

    Returns
    -------
    index : Index or MultiIndex

    Examples
    --------
    >>> ensure_index_from_sequences([[1, 2, 3]], names=["name"])
    Index([1, 2, 3], dtype='int64', name='name')

    >>> ensure_index_from_sequences([["a", "a"], ["a", "b"]], names=["L1", "L2"])
    MultiIndex([('a', 'a'),
                ('a', 'b')],
               names=['L1', 'L2'])

    See Also
    --------
    ensure_index
    """
    from pandas.core.indexes.multi import MultiIndex

    if len(sequences) == 1:
        if names is not None:
            names = names[0]
        return Index(sequences[0], name=names)
    else:
        return MultiIndex.from_arrays(sequences, names=names)


def ensure_index(index_like: Axes, copy: bool = False) -> Index:
    """
    Ensure that we have an index from some index-like object.

    Parameters
    ----------
    index_like : sequence
        An Index or other sequence
    copy : bool, default False

    Returns
    -------
    index : Index or MultiIndex

    See Also
    --------
    ensure_index_from_sequences

    Examples
    --------
    >>> ensure_index(['a', 'b'])
    Index(['a', 'b'], dtype='object')

    >>> ensure_index([('a', 'a'),  ('b', 'c')])
    Index([('a', 'a'), ('b', 'c')], dtype='object')

    >>> ensure_index([['a', 'a'], ['b', 'c']])
    MultiIndex([('a', 'b'),
            ('a', 'c')],
           )
    """
    if isinstance(index_like, Index):
        if copy:
            index_like = index_like.copy()
        return index_like

    if isinstance(index_like, ABCSeries):
        name = index_like.name
        return Index(index_like, name=name, copy=copy)

    if is_iterator(index_like):
        index_like = list(index_like)

    if isinstance(index_like, list):
        if type(index_like) is not list:  # noqa: E721
            # must check for exactly list here because of strict type
            # check in clean_index_list
            index_like = list(index_like)

        if len(index_like) and lib.is_all_arraylike(index_like):
            from pandas.core.indexes.multi import MultiIndex

            return MultiIndex.from_arrays(index_like)
        else:
            return Index(index_like, copy=copy, tupleize_cols=False)
    else:
        return Index(index_like, copy=copy)


def ensure_has_len(seq):
    """
    If seq is an iterator, put its values into a list.
    """
    try:
        len(seq)
    except TypeError:
        return list(seq)
    else:
        return seq


def trim_front(strings: list[str]) -> list[str]:
    """
    Trims zeros and decimal points.

    Examples
    --------
    >>> trim_front([" a", " b"])
    ['a', 'b']

    >>> trim_front([" a", " "])
    ['a', '']
    """
    if not strings:
        return strings
    while all(strings) and all(x[0] == " " for x in strings):
        strings = [x[1:] for x in strings]
    return strings


def _validate_join_method(method: str) -> None:
    if method not in ["left", "right", "inner", "outer"]:
        raise ValueError(f"do not recognize join method {method}")


def maybe_extract_name(name, obj, cls) -> Hashable:
    """
    If no name is passed, then extract it from data, validating hashability.
    """
    if name is None and isinstance(obj, (Index, ABCSeries)):
        # Note we don't just check for "name" attribute since that would
        #  pick up e.g. dtype.name
        name = obj.name

    # GH#29069
    if not is_hashable(name):
        raise TypeError(f"{cls.__name__}.name must be a hashable type")

    return name


def get_unanimous_names(*indexes: Index) -> tuple[Hashable, ...]:
    """
    Return common name if all indices agree, otherwise None (level-by-level).

    Parameters
    ----------
    indexes : list of Index objects

    Returns
    -------
    list
        A list representing the unanimous 'names' found.
    """
    name_tups = [tuple(i.names) for i in indexes]
    name_sets = [{*ns} for ns in zip_longest(*name_tups)]
    names = tuple(ns.pop() if len(ns) == 1 else None for ns in name_sets)
    return names


def _unpack_nested_dtype(other: Index) -> DtypeObj:
    """
    When checking if our dtype is comparable with another, we need
    to unpack CategoricalDtype to look at its categories.dtype.

    Parameters
    ----------
    other : Index

    Returns
    -------
    np.dtype or ExtensionDtype
    """
    dtype = other.dtype
    if isinstance(dtype, CategoricalDtype):
        # If there is ever a SparseIndex, this could get dispatched
        #  here too.
        return dtype.categories.dtype
    elif isinstance(dtype, ArrowDtype):
        # GH 53617
        import pyarrow as pa

        if pa.types.is_dictionary(dtype.pyarrow_dtype):
            other = other[:0].astype(ArrowDtype(dtype.pyarrow_dtype.value_type))
    return other.dtype


def _maybe_try_sort(result: Index | ArrayLike, sort: bool | None):
    if sort is not False:
        try:
            # error: Incompatible types in assignment (expression has type
            # "Union[ExtensionArray, ndarray[Any, Any], Index, Series,
            # Tuple[Union[Union[ExtensionArray, ndarray[Any, Any]], Index, Series],
            # ndarray[Any, Any]]]", variable has type "Union[Index,
            # Union[ExtensionArray, ndarray[Any, Any]]]")
            result = algos.safe_sort(result)  # type: ignore[assignment]
        except TypeError as err:
            if sort is True:
                raise
            warnings.warn(
                f"{err}, sort order is undefined for incomparable objects.",
                RuntimeWarning,
                stacklevel=find_stack_level(),
            )
    return result


def get_values_for_csv(
    values: ArrayLike,
    *,
    date_format,
    na_rep: str = "nan",
    quoting=None,
    float_format=None,
    decimal: str = ".",
) -> npt.NDArray[np.object_]:
    """
    Convert to types which can be consumed by the standard library's
    csv.writer.writerows.
    """
    if isinstance(values, Categorical) and values.categories.dtype.kind in "Mm":
        # GH#40754 Convert categorical datetimes to datetime array
        values = algos.take_nd(
            values.categories._values,
            ensure_platform_int(values._codes),
            fill_value=na_rep,
        )

    values = ensure_wrapped_if_datetimelike(values)

    if isinstance(values, (DatetimeArray, TimedeltaArray)):
        if values.ndim == 1:
            result = values._format_native_types(na_rep=na_rep, date_format=date_format)
            result = result.astype(object, copy=False)
            return result

        # GH#21734 Process every column separately, they might have different formats
        results_converted = []
        for i in range(len(values)):
            result = values[i, :]._format_native_types(
                na_rep=na_rep, date_format=date_format
            )
            results_converted.append(result.astype(object, copy=False))
        return np.vstack(results_converted)

    elif isinstance(values.dtype, PeriodDtype):
        # TODO: tests that get here in column path
        values = cast("PeriodArray", values)
        res = values._format_native_types(na_rep=na_rep, date_format=date_format)
        return res

    elif isinstance(values.dtype, IntervalDtype):
        # TODO: tests that get here in column path
        values = cast("IntervalArray", values)
        mask = values.isna()
        if not quoting:
            result = np.asarray(values).astype(str)
        else:
            result = np.array(values, dtype=object, copy=True)

        result[mask] = na_rep
        return result

    elif values.dtype.kind == "f" and not isinstance(values.dtype, SparseDtype):
        # see GH#13418: no special formatting is desired at the
        # output (important for appropriate 'quoting' behaviour),
        # so do not pass it through the FloatArrayFormatter
        if float_format is None and decimal == ".":
            mask = isna(values)

            if not quoting:
                values = values.astype(str)
            else:
                values = np.array(values, dtype="object")

            values[mask] = na_rep
            values = values.astype(object, copy=False)
            return values

        from pandas.io.formats.format import FloatArrayFormatter

        formatter = FloatArrayFormatter(
            values,
            na_rep=na_rep,
            float_format=float_format,
            decimal=decimal,
            quoting=quoting,
            fixed_width=False,
        )
        res = formatter.get_result_as_array()
        res = res.astype(object, copy=False)
        return res

    elif isinstance(values, ExtensionArray):
        mask = isna(values)

        new_values = np.asarray(values.astype(object))
        new_values[mask] = na_rep
        return new_values

    else:
        mask = isna(values)
        itemsize = writers.word_len(na_rep)

        if values.dtype != _dtype_obj and not quoting and itemsize:
            values = values.astype(str)
            if values.dtype.itemsize / np.dtype("U1").itemsize < itemsize:
                # enlarge for the na_rep
                values = values.astype(f"<U{itemsize}")
        else:
            values = np.array(values, dtype="object")

        values[mask] = na_rep
        values = values.astype(object, copy=False)
        return values