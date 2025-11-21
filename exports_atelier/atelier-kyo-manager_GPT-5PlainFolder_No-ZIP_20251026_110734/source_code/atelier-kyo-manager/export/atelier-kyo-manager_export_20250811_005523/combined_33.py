
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\context.py ===
# orm/context.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from __future__ import annotations

import itertools
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import attributes
from . import interfaces
from . import loading
from .base import _is_aliased_class
from .interfaces import ORMColumnDescription
from .interfaces import ORMColumnsClauseRole
from .path_registry import PathRegistry
from .util import _entity_corresponds_to
from .util import _ORMJoin
from .util import _TraceAdaptRole
from .util import AliasedClass
from .util import Bundle
from .util import ORMAdapter
from .util import ORMStatementAdapter
from .. import exc as sa_exc
from .. import future
from .. import inspect
from .. import sql
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..sql import util as sql_util
from ..sql import visitors
from ..sql._typing import _TP
from ..sql._typing import is_dml
from ..sql._typing import is_insert_update
from ..sql._typing import is_select_base
from ..sql.base import _select_iterables
from ..sql.base import CacheableOptions
from ..sql.base import CompileState
from ..sql.base import Executable
from ..sql.base import Generative
from ..sql.base import Options
from ..sql.dml import UpdateBase
from ..sql.elements import GroupedElement
from ..sql.elements import TextClause
from ..sql.selectable import CompoundSelectState
from ..sql.selectable import LABEL_STYLE_DISAMBIGUATE_ONLY
from ..sql.selectable import LABEL_STYLE_NONE
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..sql.selectable import Select
from ..sql.selectable import SelectLabelStyle
from ..sql.selectable import SelectState
from ..sql.selectable import TypedReturnsRows
from ..sql.visitors import InternalTraversal

if TYPE_CHECKING:
    from ._typing import _InternalEntityType
    from ._typing import OrmExecuteOptionsParameter
    from .loading import PostLoad
    from .mapper import Mapper
    from .query import Query
    from .session import _BindArguments
    from .session import Session
    from ..engine import Result
    from ..engine.interfaces import _CoreSingleExecuteParams
    from ..sql._typing import _ColumnsClauseArgument
    from ..sql.compiler import SQLCompiler
    from ..sql.dml import _DMLTableElement
    from ..sql.elements import ColumnElement
    from ..sql.selectable import _JoinTargetElement
    from ..sql.selectable import _LabelConventionCallable
    from ..sql.selectable import _SetupJoinsElement
    from ..sql.selectable import ExecutableReturnsRows
    from ..sql.selectable import SelectBase
    from ..sql.type_api import TypeEngine

_T = TypeVar("_T", bound=Any)
_path_registry = PathRegistry.root

_EMPTY_DICT = util.immutabledict()


LABEL_STYLE_LEGACY_ORM = SelectLabelStyle.LABEL_STYLE_LEGACY_ORM


class QueryContext:
    __slots__ = (
        "top_level_context",
        "compile_state",
        "query",
        "user_passed_query",
        "params",
        "load_options",
        "bind_arguments",
        "execution_options",
        "session",
        "autoflush",
        "populate_existing",
        "invoke_all_eagers",
        "version_check",
        "refresh_state",
        "create_eager_joins",
        "propagated_loader_options",
        "attributes",
        "runid",
        "partials",
        "post_load_paths",
        "identity_token",
        "yield_per",
        "loaders_require_buffering",
        "loaders_require_uniquing",
    )

    runid: int
    post_load_paths: Dict[PathRegistry, PostLoad]
    compile_state: ORMCompileState

    class default_load_options(Options):
        _only_return_tuples = False
        _populate_existing = False
        _version_check = False
        _invoke_all_eagers = True
        _autoflush = True
        _identity_token = None
        _yield_per = None
        _refresh_state = None
        _lazy_loaded_from = None
        _legacy_uniquing = False
        _sa_top_level_orm_context = None
        _is_user_refresh = False

    def __init__(
        self,
        compile_state: CompileState,
        statement: Union[Select[Any], FromStatement[Any], UpdateBase],
        user_passed_query: Union[
            Select[Any],
            FromStatement[Any],
            UpdateBase,
        ],
        params: _CoreSingleExecuteParams,
        session: Session,
        load_options: Union[
            Type[QueryContext.default_load_options],
            QueryContext.default_load_options,
        ],
        execution_options: Optional[OrmExecuteOptionsParameter] = None,
        bind_arguments: Optional[_BindArguments] = None,
    ):
        self.load_options = load_options
        self.execution_options = execution_options or _EMPTY_DICT
        self.bind_arguments = bind_arguments or _EMPTY_DICT
        self.compile_state = compile_state
        self.query = statement

        # the query that the end user passed to Session.execute() or similar.
        # this is usually the same as .query, except in the bulk_persistence
        # routines where a separate FromStatement is manufactured in the
        # compile stage; this allows differentiation in that case.
        self.user_passed_query = user_passed_query

        self.session = session
        self.loaders_require_buffering = False
        self.loaders_require_uniquing = False
        self.params = params
        self.top_level_context = load_options._sa_top_level_orm_context

        cached_options = compile_state.select_statement._with_options
        uncached_options = user_passed_query._with_options

        # see issue #7447 , #8399 for some background
        # propagated loader options will be present on loaded InstanceState
        # objects under state.load_options and are typically used by
        # LazyLoader to apply options to the SELECT statement it emits.
        # For compile state options (i.e. loader strategy options), these
        # need to line up with the ".load_path" attribute which in
        # loader.py is pulled from context.compile_state.current_path.
        # so, this means these options have to be the ones from the
        # *cached* statement that's travelling with compile_state, not the
        # *current* statement which won't match up for an ad-hoc
        # AliasedClass
        self.propagated_loader_options = tuple(
            opt._adapt_cached_option_to_uncached_option(self, uncached_opt)
            for opt, uncached_opt in zip(cached_options, uncached_options)
            if opt.propagate_to_loaders
        )

        self.attributes = dict(compile_state.attributes)

        self.autoflush = load_options._autoflush
        self.populate_existing = load_options._populate_existing
        self.invoke_all_eagers = load_options._invoke_all_eagers
        self.version_check = load_options._version_check
        self.refresh_state = load_options._refresh_state
        self.yield_per = load_options._yield_per
        self.identity_token = load_options._identity_token

    def _get_top_level_context(self) -> QueryContext:
        return self.top_level_context or self


_orm_load_exec_options = util.immutabledict(
    {"_result_disable_adapt_to_context": True}
)


class AbstractORMCompileState(CompileState):
    is_dml_returning = False

    def _init_global_attributes(
        self, statement, compiler, *, toplevel, process_criteria_for_toplevel
    ):
        self.attributes = {}

        if compiler is None:
            # this is the legacy / testing only ORM _compile_state() use case.
            # there is no need to apply criteria options for this.
            self.global_attributes = {}
            assert toplevel
            return
        else:
            self.global_attributes = ga = compiler._global_attributes

        if toplevel:
            ga["toplevel_orm"] = True

            if process_criteria_for_toplevel:
                for opt in statement._with_options:
                    if opt._is_criteria_option:
                        opt.process_compile_state(self)

            return
        elif ga.get("toplevel_orm", False):
            return

        stack_0 = compiler.stack[0]

        try:
            toplevel_stmt = stack_0["selectable"]
        except KeyError:
            pass
        else:
            for opt in toplevel_stmt._with_options:
                if opt._is_compile_state and opt._is_criteria_option:
                    opt.process_compile_state(self)

        ga["toplevel_orm"] = True

    @classmethod
    def create_for_statement(
        cls,
        statement: Executable,
        compiler: SQLCompiler,
        **kw: Any,
    ) -> CompileState:
        """Create a context for a statement given a :class:`.Compiler`.

        This method is always invoked in the context of SQLCompiler.process().

        For a Select object, this would be invoked from
        SQLCompiler.visit_select(). For the special FromStatement object used
        by Query to indicate "Query.from_statement()", this is called by
        FromStatement._compiler_dispatch() that would be called by
        SQLCompiler.process().
        """
        return super().create_for_statement(statement, compiler, **kw)

    @classmethod
    def orm_pre_session_exec(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        is_pre_event,
    ):
        raise NotImplementedError()

    @classmethod
    def orm_execute_statement(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        conn,
    ) -> Result:
        result = conn.execute(
            statement, params or {}, execution_options=execution_options
        )
        return cls.orm_setup_cursor_result(
            session,
            statement,
            params,
            execution_options,
            bind_arguments,
            result,
        )

    @classmethod
    def orm_setup_cursor_result(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        result,
    ):
        raise NotImplementedError()


class AutoflushOnlyORMCompileState(AbstractORMCompileState):
    """ORM compile state that is a passthrough, except for autoflush."""

    @classmethod
    def orm_pre_session_exec(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        is_pre_event,
    ):
        # consume result-level load_options.  These may have been set up
        # in an ORMExecuteState hook
        (
            load_options,
            execution_options,
        ) = QueryContext.default_load_options.from_execution_options(
            "_sa_orm_load_options",
            {
                "autoflush",
            },
            execution_options,
            statement._execution_options,
        )

        if not is_pre_event and load_options._autoflush:
            session._autoflush()

        return statement, execution_options

    @classmethod
    def orm_setup_cursor_result(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        result,
    ):
        return result


class ORMCompileState(AbstractORMCompileState):
    class default_compile_options(CacheableOptions):
        _cache_key_traversal = [
            ("_use_legacy_query_style", InternalTraversal.dp_boolean),
            ("_for_statement", InternalTraversal.dp_boolean),
            ("_bake_ok", InternalTraversal.dp_boolean),
            ("_current_path", InternalTraversal.dp_has_cache_key),
            ("_enable_single_crit", InternalTraversal.dp_boolean),
            ("_enable_eagerloads", InternalTraversal.dp_boolean),
            ("_only_load_props", InternalTraversal.dp_plain_obj),
            ("_set_base_alias", InternalTraversal.dp_boolean),
            ("_for_refresh_state", InternalTraversal.dp_boolean),
            ("_render_for_subquery", InternalTraversal.dp_boolean),
            ("_is_star", InternalTraversal.dp_boolean),
        ]

        # set to True by default from Query._statement_20(), to indicate
        # the rendered query should look like a legacy ORM query.  right
        # now this basically indicates we should use tablename_columnname
        # style labels.    Generally indicates the statement originated
        # from a Query object.
        _use_legacy_query_style = False

        # set *only* when we are coming from the Query.statement
        # accessor, or a Query-level equivalent such as
        # query.subquery().  this supersedes "toplevel".
        _for_statement = False

        _bake_ok = True
        _current_path = _path_registry
        _enable_single_crit = True
        _enable_eagerloads = True
        _only_load_props = None
        _set_base_alias = False
        _for_refresh_state = False
        _render_for_subquery = False
        _is_star = False

    attributes: Dict[Any, Any]
    global_attributes: Dict[Any, Any]

    statement: Union[Select[Any], FromStatement[Any], UpdateBase]
    select_statement: Union[Select[Any], FromStatement[Any], UpdateBase]
    _entities: List[_QueryEntity]
    _polymorphic_adapters: Dict[_InternalEntityType, ORMAdapter]
    compile_options: Union[
        Type[default_compile_options], default_compile_options
    ]
    _primary_entity: Optional[_QueryEntity]
    use_legacy_query_style: bool
    _label_convention: _LabelConventionCallable
    primary_columns: List[ColumnElement[Any]]
    secondary_columns: List[ColumnElement[Any]]
    dedupe_columns: Set[ColumnElement[Any]]
    create_eager_joins: List[
        # TODO: this structure is set up by JoinedLoader
        Tuple[Any, ...]
    ]
    current_path: PathRegistry = _path_registry
    _has_mapper_entities = False

    def __init__(self, *arg, **kw):
        raise NotImplementedError()

    @classmethod
    def create_for_statement(
        cls,
        statement: Executable,
        compiler: SQLCompiler,
        **kw: Any,
    ) -> ORMCompileState:
        return cls._create_orm_context(
            cast("Union[Select, FromStatement]", statement),
            toplevel=not compiler.stack,
            compiler=compiler,
            **kw,
        )

    @classmethod
    def _create_orm_context(
        cls,
        statement: Union[Select, FromStatement],
        *,
        toplevel: bool,
        compiler: Optional[SQLCompiler],
        **kw: Any,
    ) -> ORMCompileState:
        raise NotImplementedError()

    def _append_dedupe_col_collection(self, obj, col_collection):
        dedupe = self.dedupe_columns
        if obj not in dedupe:
            dedupe.add(obj)
            col_collection.append(obj)

    @classmethod
    def _column_naming_convention(
        cls, label_style: SelectLabelStyle, legacy: bool
    ) -> _LabelConventionCallable:
        if legacy:

            def name(col, col_name=None):
                if col_name:
                    return col_name
                else:
                    return getattr(col, "key")

            return name
        else:
            return SelectState._column_naming_convention(label_style)

    @classmethod
    def get_column_descriptions(cls, statement):
        return _column_descriptions(statement)

    @classmethod
    def orm_pre_session_exec(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        is_pre_event,
    ):
        # consume result-level load_options.  These may have been set up
        # in an ORMExecuteState hook
        (
            load_options,
            execution_options,
        ) = QueryContext.default_load_options.from_execution_options(
            "_sa_orm_load_options",
            {
                "populate_existing",
                "autoflush",
                "yield_per",
                "identity_token",
                "sa_top_level_orm_context",
            },
            execution_options,
            statement._execution_options,
        )

        # default execution options for ORM results:
        # 1. _result_disable_adapt_to_context=True
        #    this will disable the ResultSetMetadata._adapt_to_context()
        #    step which we don't need, as we have result processors cached
        #    against the original SELECT statement before caching.

        if "sa_top_level_orm_context" in execution_options:
            ctx = execution_options["sa_top_level_orm_context"]
            execution_options = ctx.query._execution_options.merge_with(
                ctx.execution_options, execution_options
            )

        if not execution_options:
            execution_options = _orm_load_exec_options
        else:
            execution_options = execution_options.union(_orm_load_exec_options)

        # would have been placed here by legacy Query only
        if load_options._yield_per:
            execution_options = execution_options.union(
                {"yield_per": load_options._yield_per}
            )

        if (
            getattr(statement._compile_options, "_current_path", None)
            and len(statement._compile_options._current_path) > 10
            and execution_options.get("compiled_cache", True) is not None
        ):
            execution_options: util.immutabledict[str, Any] = (
                execution_options.union(
                    {
                        "compiled_cache": None,
                        "_cache_disable_reason": "excess depth for "
                        "ORM loader options",
                    }
                )
            )

        bind_arguments["clause"] = statement

        # new in 1.4 - the coercions system is leveraged to allow the
        # "subject" mapper of a statement be propagated to the top
        # as the statement is built.   "subject" mapper is the generally
        # standard object used as an identifier for multi-database schemes.

        # we are here based on the fact that _propagate_attrs contains
        # "compile_state_plugin": "orm".   The "plugin_subject"
        # needs to be present as well.

        try:
            plugin_subject = statement._propagate_attrs["plugin_subject"]
        except KeyError:
            assert False, "statement had 'orm' plugin but no plugin_subject"
        else:
            if plugin_subject:
                bind_arguments["mapper"] = plugin_subject.mapper

        if not is_pre_event and load_options._autoflush:
            session._autoflush()

        return statement, execution_options

    @classmethod
    def orm_setup_cursor_result(
        cls,
        session,
        statement,
        params,
        execution_options,
        bind_arguments,
        result,
    ):
        execution_context = result.context
        compile_state = execution_context.compiled.compile_state

        # cover edge case where ORM entities used in legacy select
        # were passed to session.execute:
        # session.execute(legacy_select([User.id, User.name]))
        # see test_query->test_legacy_tuple_old_select

        load_options = execution_options.get(
            "_sa_orm_load_options", QueryContext.default_load_options
        )

        if compile_state.compile_options._is_star:
            return result

        querycontext = QueryContext(
            compile_state,
            statement,
            statement,
            params,
            session,
            load_options,
            execution_options,
            bind_arguments,
        )
        return loading.instances(result, querycontext)

    @property
    def _lead_mapper_entities(self):
        """return all _MapperEntity objects in the lead entities collection.

        Does **not** include entities that have been replaced by
        with_entities(), with_only_columns()

        """
        return [
            ent for ent in self._entities if isinstance(ent, _MapperEntity)
        ]

    def _create_with_polymorphic_adapter(self, ext_info, selectable):
        """given MapperEntity or ORMColumnEntity, setup polymorphic loading
        if called for by the Mapper.

        As of #8168 in 2.0.0rc1, polymorphic adapters, which greatly increase
        the complexity of the query creation process, are not used at all
        except in the quasi-legacy cases of with_polymorphic referring to an
        alias and/or subquery. This would apply to concrete polymorphic
        loading, and joined inheritance where a subquery is
        passed to with_polymorphic (which is completely unnecessary in modern
        use).

        """
        if (
            not ext_info.is_aliased_class
            and ext_info.mapper.persist_selectable
            not in self._polymorphic_adapters
        ):
            for mp in ext_info.mapper.iterate_to_root():
                self._mapper_loads_polymorphically_with(
                    mp,
                    ORMAdapter(
                        _TraceAdaptRole.WITH_POLYMORPHIC_ADAPTER,
                        mp,
                        equivalents=mp._equivalent_columns,
                        selectable=selectable,
                    ),
                )

    def _mapper_loads_polymorphically_with(self, mapper, adapter):
        for m2 in mapper._with_polymorphic_mappers or [mapper]:
            self._polymorphic_adapters[m2] = adapter

            for m in m2.iterate_to_root():
                self._polymorphic_adapters[m.local_table] = adapter

    @classmethod
    def _create_entities_collection(cls, query, legacy):
        raise NotImplementedError(
            "this method only works for ORMSelectCompileState"
        )


class _DMLReturningColFilter:
    """a base for an adapter used for the DML RETURNING cases

    Has a subset of the interface used by
    :class:`.ORMAdapter` and is used for :class:`._QueryEntity`
    instances to set up their columns as used in RETURNING for a
    DML statement.

    """

    __slots__ = ("mapper", "columns", "__weakref__")

    def __init__(self, target_mapper, immediate_dml_mapper):
        if (
            immediate_dml_mapper is not None
            and target_mapper.local_table
            is not immediate_dml_mapper.local_table
        ):
            # joined inh, or in theory other kinds of multi-table mappings
            self.mapper = immediate_dml_mapper
        else:
            # single inh, normal mappings, etc.
            self.mapper = target_mapper
        self.columns = self.columns = util.WeakPopulateDict(
            self.adapt_check_present  # type: ignore
        )

    def __call__(self, col, as_filter):
        for cc in sql_util._find_columns(col):
            c2 = self.adapt_check_present(cc)
            if c2 is not None:
                return col
        else:
            return None

    def adapt_check_present(self, col):
        raise NotImplementedError()


class _DMLBulkInsertReturningColFilter(_DMLReturningColFilter):
    """an adapter used for the DML RETURNING case specifically
    for ORM bulk insert (or any hypothetical DML that is splitting out a class
    hierarchy among multiple DML statements....ORM bulk insert is the only
    example right now)

    its main job is to limit the columns in a RETURNING to only a specific
    mapped table in a hierarchy.

    """

    def adapt_check_present(self, col):
        mapper = self.mapper
        prop = mapper._columntoproperty.get(col, None)
        if prop is None:
            return None
        return mapper.local_table.c.corresponding_column(col)


class _DMLUpdateDeleteReturningColFilter(_DMLReturningColFilter):
    """an adapter used for the DML RETURNING case specifically
    for ORM enabled UPDATE/DELETE

    its main job is to limit the columns in a RETURNING to include
    only direct persisted columns from the immediate selectable, not
    expressions like column_property(), or to also allow columns from other
    mappers for the UPDATE..FROM use case.

    """

    def adapt_check_present(self, col):
        mapper = self.mapper
        prop = mapper._columntoproperty.get(col, None)
        if prop is not None:
            # if the col is from the immediate mapper, only return a persisted
            # column, not any kind of column_property expression
            return mapper.persist_selectable.c.corresponding_column(col)

        # if the col is from some other mapper, just return it, assume the
        # user knows what they are doing
        return col


@sql.base.CompileState.plugin_for("orm", "orm_from_statement")
class ORMFromStatementCompileState(ORMCompileState):
    _from_obj_alias = None
    _has_mapper_entities = False

    statement_container: FromStatement
    requested_statement: Union[SelectBase, TextClause, UpdateBase]
    dml_table: Optional[_DMLTableElement] = None

    _has_orm_entities = False
    multi_row_eager_loaders = False
    eager_adding_joins = False
    compound_eager_adapter = None

    extra_criteria_entities = _EMPTY_DICT
    eager_joins = _EMPTY_DICT

    @classmethod
    def _create_orm_context(
        cls,
        statement: Union[Select, FromStatement],
        *,
        toplevel: bool,
        compiler: Optional[SQLCompiler],
        **kw: Any,
    ) -> ORMFromStatementCompileState:
        statement_container = statement

        assert isinstance(statement_container, FromStatement)

        if compiler is not None and compiler.stack:
            raise sa_exc.CompileError(
                "The ORM FromStatement construct only supports being "
                "invoked as the topmost statement, as it is only intended to "
                "define how result rows should be returned."
            )

        self = cls.__new__(cls)
        self._primary_entity = None

        self.use_legacy_query_style = (
            statement_container._compile_options._use_legacy_query_style
        )
        self.statement_container = self.select_statement = statement_container
        self.requested_statement = statement = statement_container.element

        if statement.is_dml:
            self.dml_table = statement.table
            self.is_dml_returning = True

        self._entities = []
        self._polymorphic_adapters = {}

        self.compile_options = statement_container._compile_options

        if (
            self.use_legacy_query_style
            and isinstance(statement, expression.SelectBase)
            and not statement._is_textual
            and not statement.is_dml
            and statement._label_style is LABEL_STYLE_NONE
        ):
            self.statement = statement.set_label_style(
                LABEL_STYLE_TABLENAME_PLUS_COL
            )
        else:
            self.statement = statement

        self._label_convention = self._column_naming_convention(
            (
                statement._label_style
                if not statement._is_textual and not statement.is_dml
                else LABEL_STYLE_NONE
            ),
            self.use_legacy_query_style,
        )

        _QueryEntity.to_compile_state(
            self,
            statement_container._raw_columns,
            self._entities,
            is_current_entities=True,
        )

        self.current_path = statement_container._compile_options._current_path

        self._init_global_attributes(
            statement_container,
            compiler,
            process_criteria_for_toplevel=False,
            toplevel=True,
        )

        if statement_container._with_options:
            for opt in statement_container._with_options:
                if opt._is_compile_state:
                    opt.process_compile_state(self)

        if statement_container._with_context_options:
            for fn, key in statement_container._with_context_options:
                fn(self)

        self.primary_columns = []
        self.secondary_columns = []
        self.dedupe_columns = set()
        self.create_eager_joins = []
        self._fallback_from_clauses = []

        self.order_by = None

        if isinstance(self.statement, expression.TextClause):
            # TextClause has no "column" objects at all.  for this case,
            # we generate columns from our _QueryEntity objects, then
            # flip on all the "please match no matter what" parameters.
            self.extra_criteria_entities = {}

            for entity in self._entities:
                entity.setup_compile_state(self)

            compiler._ordered_columns = compiler._textual_ordered_columns = (
                False
            )

            # enable looser result column matching.  this is shown to be
            # needed by test_query.py::TextTest
            compiler._loose_column_name_matching = True

            for c in self.primary_columns:
                compiler.process(
                    c,
                    within_columns_clause=True,
                    add_to_result_map=compiler._add_to_result_map,
                )
        else:
            # for everyone else, Select, Insert, Update, TextualSelect, they
            # have column objects already.  After much
            # experimentation here, the best approach seems to be, use
            # those columns completely, don't interfere with the compiler
            # at all; just in ORM land, use an adapter to convert from
            # our ORM columns to whatever columns are in the statement,
            # before we look in the result row. Adapt on names
            # to accept cases such as issue #9217, however also allow
            # this to be overridden for cases such as #9273.
            self._from_obj_alias = ORMStatementAdapter(
                _TraceAdaptRole.ADAPT_FROM_STATEMENT,
                self.statement,
                adapt_on_names=statement_container._adapt_on_names,
            )

        return self

    def _adapt_col_list(self, cols, current_adapter):
        return cols

    def _get_current_adapter(self):
        return None

    def setup_dml_returning_compile_state(self, dml_mapper):
        """used by BulkORMInsert, Update, Delete to set up a handler
        for RETURNING to return ORM objects and expressions

        """
        target_mapper = self.statement._propagate_attrs.get(
            "plugin_subject", None
        )

        if self.statement.is_insert:
            adapter = _DMLBulkInsertReturningColFilter(
                target_mapper, dml_mapper
            )
        elif self.statement.is_update or self.statement.is_delete:
            adapter = _DMLUpdateDeleteReturningColFilter(
                target_mapper, dml_mapper
            )
        else:
            adapter = None

        if self.compile_options._is_star and (len(self._entities) != 1):
            raise sa_exc.CompileError(
                "Can't generate ORM query that includes multiple expressions "
                "at the same time as '*'; query for '*' alone if present"
            )

        for entity in self._entities:
            entity.setup_dml_returning_compile_state(self, adapter)


class FromStatement(GroupedElement, Generative, TypedReturnsRows[_TP]):
    """Core construct that represents a load of ORM objects from various
    :class:`.ReturnsRows` and other classes including:

    :class:`.Select`, :class:`.TextClause`, :class:`.TextualSelect`,
    :class:`.CompoundSelect`, :class`.Insert`, :class:`.Update`,
    and in theory, :class:`.Delete`.

    """

    __visit_name__ = "orm_from_statement"

    _compile_options = ORMFromStatementCompileState.default_compile_options

    _compile_state_factory = ORMFromStatementCompileState.create_for_statement

    _for_update_arg = None

    element: Union[ExecutableReturnsRows, TextClause]

    _adapt_on_names: bool

    _traverse_internals = [
        ("_raw_columns", InternalTraversal.dp_clauseelement_list),
        ("element", InternalTraversal.dp_clauseelement),
    ] + Executable._executable_traverse_internals

    _cache_key_traversal = _traverse_internals + [
        ("_compile_options", InternalTraversal.dp_has_cache_key)
    ]

    is_from_statement = True

    def __init__(
        self,
        entities: Iterable[_ColumnsClauseArgument[Any]],
        element: Union[ExecutableReturnsRows, TextClause],
        _adapt_on_names: bool = True,
    ):
        self._raw_columns = [
            coercions.expect(
                roles.ColumnsClauseRole,
                ent,
                apply_propagate_attrs=self,
                post_inspect=True,
            )
            for ent in util.to_list(entities)
        ]
        self.element = element
        self.is_dml = element.is_dml
        self.is_select = element.is_select
        self.is_delete = element.is_delete
        self.is_insert = element.is_insert
        self.is_update = element.is_update
        self._label_style = (
            element._label_style if is_select_base(element) else None
        )
        self._adapt_on_names = _adapt_on_names

    def _compiler_dispatch(self, compiler, **kw):
        """provide a fixed _compiler_dispatch method.

        This is roughly similar to using the sqlalchemy.ext.compiler
        ``@compiles`` extension.

        """

        compile_state = self._compile_state_factory(self, compiler, **kw)

        toplevel = not compiler.stack

        if toplevel:
            compiler.compile_state = compile_state

        return compiler.process(compile_state.statement, **kw)

    @property
    def column_descriptions(self):
        """Return a :term:`plugin-enabled` 'column descriptions' structure
        referring to the columns which are SELECTed by this statement.

        See the section :ref:`queryguide_inspection` for an overview
        of this feature.

        .. seealso::

            :ref:`queryguide_inspection` - ORM background

        """
        meth = cast(
            ORMSelectCompileState, SelectState.get_plugin_class(self)
        ).get_column_descriptions
        return meth(self)

    def _ensure_disambiguated_names(self):
        return self

    def get_children(self, **kw):
        yield from itertools.chain.from_iterable(
            element._from_objects for element in self._raw_columns
        )
        yield from super().get_children(**kw)

    @property
    def _all_selected_columns(self):
        return self.element._all_selected_columns

    @property
    def _return_defaults(self):
        return self.element._return_defaults if is_dml(self.element) else None

    @property
    def _returning(self):
        return self.element._returning if is_dml(self.element) else None

    @property
    def _inline(self):
        return self.element._inline if is_insert_update(self.element) else None


@sql.base.CompileState.plugin_for("orm", "compound_select")
class CompoundSelectCompileState(
    AutoflushOnlyORMCompileState, CompoundSelectState
):
    pass


@sql.base.CompileState.plugin_for("orm", "select")
class ORMSelectCompileState(ORMCompileState, SelectState):
    _already_joined_edges = ()

    _memoized_entities = _EMPTY_DICT

    _from_obj_alias = None
    _has_mapper_entities = False

    _has_orm_entities = False
    multi_row_eager_loaders = False
    eager_adding_joins = False
    compound_eager_adapter = None

    correlate = None
    correlate_except = None
    _where_criteria = ()
    _having_criteria = ()

    @classmethod
    def _create_orm_context(
        cls,
        statement: Union[Select, FromStatement],
        *,
        toplevel: bool,
        compiler: Optional[SQLCompiler],
        **kw: Any,
    ) -> ORMSelectCompileState:

        self = cls.__new__(cls)

        select_statement = statement

        # if we are a select() that was never a legacy Query, we won't
        # have ORM level compile options.
        statement._compile_options = cls.default_compile_options.safe_merge(
            statement._compile_options
        )

        if select_statement._execution_options:
            # execution options should not impact the compilation of a
            # query, and at the moment subqueryloader is putting some things
            # in here that we explicitly don't want stuck in a cache.
            self.select_statement = select_statement._clone()
            self.select_statement._execution_options = util.immutabledict()
        else:
            self.select_statement = select_statement

        # indicates this select() came from Query.statement
        self.for_statement = select_statement._compile_options._for_statement

        # generally if we are from Query or directly from a select()
        self.use_legacy_query_style = (
            select_statement._compile_options._use_legacy_query_style
        )

        self._entities = []
        self._primary_entity = None
        self._polymorphic_adapters = {}

        self.compile_options = select_statement._compile_options

        if not toplevel:
            # for subqueries, turn off eagerloads and set
            # "render_for_subquery".
            self.compile_options += {
                "_enable_eagerloads": False,
                "_render_for_subquery": True,
            }

        # determine label style.   we can make different decisions here.
        # at the moment, trying to see if we can always use DISAMBIGUATE_ONLY
        # rather than LABEL_STYLE_NONE, and if we can use disambiguate style
        # for new style ORM selects too.
        if (
            self.use_legacy_query_style
            and self.select_statement._label_style is LABEL_STYLE_LEGACY_ORM
        ):
            if not self.for_statement:
                self.label_style = LABEL_STYLE_TABLENAME_PLUS_COL
            else:
                self.label_style = LABEL_STYLE_DISAMBIGUATE_ONLY
        else:
            self.label_style = self.select_statement._label_style

        if select_statement._memoized_select_entities:
            self._memoized_entities = {
                memoized_entities: _QueryEntity.to_compile_state(
                    self,
                    memoized_entities._raw_columns,
                    [],
                    is_current_entities=False,
                )
                for memoized_entities in (
                    select_statement._memoized_select_entities
                )
            }

        # label_convention is stateful and will yield deduping keys if it
        # sees the same key twice.  therefore it's important that it is not
        # invoked for the above "memoized" entities that aren't actually
        # in the columns clause
        self._label_convention = self._column_naming_convention(
            statement._label_style, self.use_legacy_query_style
        )

        _QueryEntity.to_compile_state(
            self,
            select_statement._raw_columns,
            self._entities,
            is_current_entities=True,
        )

        self.current_path = select_statement._compile_options._current_path

        self.eager_order_by = ()

        self._init_global_attributes(
            select_statement,
            compiler,
            toplevel=toplevel,
            process_criteria_for_toplevel=False,
        )

        if toplevel and (
            select_statement._with_options
            or select_statement._memoized_select_entities
        ):
            for (
                memoized_entities
            ) in select_statement._memoized_select_entities:
                for opt in memoized_entities._with_options:
                    if opt._is_compile_state:
                        opt.process_compile_state_replaced_entities(
                            self,
                            [
                                ent
                                for ent in self._memoized_entities[
                                    memoized_entities
                                ]
                                if isinstance(ent, _MapperEntity)
                            ],
                        )

            for opt in self.select_statement._with_options:
                if opt._is_compile_state:
                    opt.process_compile_state(self)

        # uncomment to print out the context.attributes structure
        # after it's been set up above
        # self._dump_option_struct()

        if select_statement._with_context_options:
            for fn, key in select_statement._with_context_options:
                fn(self)

        self.primary_columns = []
        self.secondary_columns = []
        self.dedupe_columns = set()
        self.eager_joins = {}
        self.extra_criteria_entities = {}
        self.create_eager_joins = []
        self._fallback_from_clauses = []

        # normalize the FROM clauses early by themselves, as this makes
        # it an easier job when we need to assemble a JOIN onto these,
        # for select.join() as well as joinedload().   As of 1.4 there are now
        # potentially more complex sets of FROM objects here as the use
        # of lambda statements for lazyload, load_on_pk etc. uses more
        # cloning of the select() construct.  See #6495
        self.from_clauses = self._normalize_froms(
            info.selectable for info in select_statement._from_obj
        )

        # this is a fairly arbitrary break into a second method,
        # so it might be nicer to break up create_for_statement()
        # and _setup_for_generate into three or four logical sections
        self._setup_for_generate()

        SelectState.__init__(self, self.statement, compiler, **kw)
        return self

    def _dump_option_struct(self):
        print("\n---------------------------------------------------\n")
        print(f"current path: {self.current_path}")
        for key in self.attributes:
            if isinstance(key, tuple) and key[0] == "loader":
                print(f"\nLoader:           {PathRegistry.coerce(key[1])}")
                print(f"    {self.attributes[key]}")
                print(f"    {self.attributes[key].__dict__}")
            elif isinstance(key, tuple) and key[0] == "path_with_polymorphic":
                print(f"\nWith Polymorphic: {PathRegistry.coerce(key[1])}")
                print(f"    {self.attributes[key]}")

    def _setup_for_generate(self):
        query = self.select_statement

        self.statement = None
        self._join_entities = ()

        if self.compile_options._set_base_alias:
            # legacy Query only
            self._set_select_from_alias()

        for memoized_entities in query._memoized_select_entities:
            if memoized_entities._setup_joins:
                self._join(
                    memoized_entities._setup_joins,
                    self._memoized_entities[memoized_entities],
                )

        if query._setup_joins:
            self._join(query._setup_joins, self._entities)

        current_adapter = self._get_current_adapter()

        if query._where_criteria:
            self._where_criteria = query._where_criteria

            if current_adapter:
                self._where_criteria = tuple(
                    current_adapter(crit, True)
                    for crit in self._where_criteria
                )

        # TODO: some complexity with order_by here was due to mapper.order_by.
        # now that this is removed we can hopefully make order_by /
        # group_by act identically to how they are in Core select.
        self.order_by = (
            self._adapt_col_list(query._order_by_clauses, current_adapter)
            if current_adapter and query._order_by_clauses not in (None, False)
            else query._order_by_clauses
        )

        if query._having_criteria:
            self._having_criteria = tuple(
                current_adapter(crit, True) if current_adapter else crit
                for crit in query._having_criteria
            )

        self.group_by = (
            self._adapt_col_list(
                util.flatten_iterator(query._group_by_clauses), current_adapter
            )
            if current_adapter and query._group_by_clauses not in (None, False)
            else query._group_by_clauses or None
        )

        if self.eager_order_by:
            adapter = self.from_clauses[0]._target_adapter
            self.eager_order_by = adapter.copy_and_process(self.eager_order_by)

        if query._distinct_on:
            self.distinct_on = self._adapt_col_list(
                query._distinct_on, current_adapter
            )
        else:
            self.distinct_on = ()

        self.distinct = query._distinct

        if query._correlate:
            # ORM mapped entities that are mapped to joins can be passed
            # to .correlate, so here they are broken into their component
            # tables.
            self.correlate = tuple(
                util.flatten_iterator(
                    sql_util.surface_selectables(s) if s is not None else None
                    for s in query._correlate
                )
            )
        elif query._correlate_except is not None:
            self.correlate_except = tuple(
                util.flatten_iterator(
                    sql_util.surface_selectables(s) if s is not None else None
                    for s in query._correlate_except
                )
            )
        elif not query._auto_correlate:
            self.correlate = (None,)

        # PART II

        self._for_update_arg = query._for_update_arg

        if self.compile_options._is_star and (len(self._entities) != 1):
            raise sa_exc.CompileError(
                "Can't generate ORM query that includes multiple expressions "
                "at the same time as '*'; query for '*' alone if present"
            )
        for entity in self._entities:
            entity.setup_compile_state(self)

        for rec in self.create_eager_joins:
            strategy = rec[0]
            strategy(self, *rec[1:])

        # else "load from discrete FROMs" mode,
        # i.e. when each _MappedEntity has its own FROM

        if self.compile_options._enable_single_crit:
            self._adjust_for_extra_criteria()

        if not self.primary_columns:
            if self.compile_options._only_load_props:
                assert False, "no columns were included in _only_load_props"

            raise sa_exc.InvalidRequestError(
                "Query contains no columns with which to SELECT from."
            )

        if not self.from_clauses:
            self.from_clauses = list(self._fallback_from_clauses)

        if self.order_by is False:
            self.order_by = None

        if (
            self.multi_row_eager_loaders
            and self.eager_adding_joins
            and self._should_nest_selectable
        ):
            self.statement = self._compound_eager_statement()
        else:
            self.statement = self._simple_statement()

        if self.for_statement:
            ezero = self._mapper_zero()
            if ezero is not None:
                # TODO: this goes away once we get rid of the deep entity
                # thing
                self.statement = self.statement._annotate(
                    {"deepentity": ezero}
                )

    @classmethod
    def _create_entities_collection(cls, query, legacy):
        """Creates a partial ORMSelectCompileState that includes
        the full collection of _MapperEntity and other _QueryEntity objects.

        Supports a few remaining use cases that are pre-compilation
        but still need to gather some of the column  / adaption information.

        """
        self = cls.__new__(cls)

        self._entities = []
        self._primary_entity = None
        self._polymorphic_adapters = {}

        self._label_convention = self._column_naming_convention(
            query._label_style, legacy
        )

        # entities will also set up polymorphic adapters for mappers
        # that have with_polymorphic configured
        _QueryEntity.to_compile_state(
            self, query._raw_columns, self._entities, is_current_entities=True
        )
        return self

    @classmethod
    def determine_last_joined_entity(cls, statement):
        setup_joins = statement._setup_joins

        return _determine_last_joined_entity(setup_joins, None)

    @classmethod
    def all_selected_columns(cls, statement):
        for element in statement._raw_columns:
            if (
                element.is_selectable
                and "entity_namespace" in element._annotations
            ):
                ens = element._annotations["entity_namespace"]
                if not ens.is_mapper and not ens.is_aliased_class:
                    yield from _select_iterables([element])
                else:
                    yield from _select_iterables(ens._all_column_expressions)
            else:
                yield from _select_iterables([element])

    @classmethod
    def get_columns_clause_froms(cls, statement):
        return cls._normalize_froms(
            itertools.chain.from_iterable(
                (
                    element._from_objects
                    if "parententity" not in element._annotations
                    else [
                        element._annotations[
                            "parententity"
                        ].__clause_element__()
                    ]
                )
                for element in statement._raw_columns
            )
        )

    @classmethod
    def from_statement(cls, statement, from_statement):
        from_statement = coercions.expect(
            roles.ReturnsRowsRole,
            from_statement,
            apply_propagate_attrs=statement,
        )

        stmt = FromStatement(statement._raw_columns, from_statement)

        stmt.__dict__.update(
            _with_options=statement._with_options,
            _with_context_options=statement._with_context_options,
            _execution_options=statement._execution_options,
            _propagate_attrs=statement._propagate_attrs,
        )
        return stmt

    def _set_select_from_alias(self):
        """used only for legacy Query cases"""

        query = self.select_statement  # query

        assert self.compile_options._set_base_alias
        assert len(query._from_obj) == 1

        adapter = self._get_select_from_alias_from_obj(query._from_obj[0])
        if adapter:
            self.compile_options += {"_enable_single_crit": False}
            self._from_obj_alias = adapter

    def _get_select_from_alias_from_obj(self, from_obj):
        """used only for legacy Query cases"""

        info = from_obj

        if "parententity" in info._annotations:
            info = info._annotations["parententity"]

        if hasattr(info, "mapper"):
            if not info.is_aliased_class:
                raise sa_exc.ArgumentError(
                    "A selectable (FromClause) instance is "
                    "expected when the base alias is being set."
                )
            else:
                return info._adapter

        elif isinstance(info.selectable, sql.selectable.AliasedReturnsRows):
            equivs = self._all_equivs()
            assert info is info.selectable
            return ORMStatementAdapter(
                _TraceAdaptRole.LEGACY_SELECT_FROM_ALIAS,
                info.selectable,
                equivalents=equivs,
            )
        else:
            return None

    def _mapper_zero(self):
        """return the Mapper associated with the first QueryEntity."""
        return self._entities[0].mapper

    def _entity_zero(self):
        """Return the 'entity' (mapper or AliasedClass) associated
        with the first QueryEntity, or alternatively the 'select from'
        entity if specified."""

        for ent in self.from_clauses:
            if "parententity" in ent._annotations:
                return ent._annotations["parententity"]
        for qent in self._entities:
            if qent.entity_zero:
                return qent.entity_zero

        return None

    def _only_full_mapper_zero(self, methname):
        if self._entities != [self._primary_entity]:
            raise sa_exc.InvalidRequestError(
                "%s() can only be used against "
                "a single mapped class." % methname
            )
        return self._primary_entity.entity_zero

    def _only_entity_zero(self, rationale=None):
        if len(self._entities) > 1:
            raise sa_exc.InvalidRequestError(
                rationale
                or "This operation requires a Query "
                "against a single mapper."
            )
        return self._entity_zero()

    def _all_equivs(self):
        equivs = {}

        for memoized_entities in self._memoized_entities.values():
            for ent in [
                ent
                for ent in memoized_entities
                if isinstance(ent, _MapperEntity)
            ]:
                equivs.update(ent.mapper._equivalent_columns)

        for ent in [
            ent for ent in self._entities if isinstance(ent, _MapperEntity)
        ]:
            equivs.update(ent.mapper._equivalent_columns)
        return equivs

    def _compound_eager_statement(self):
        # for eager joins present and LIMIT/OFFSET/DISTINCT,
        # wrap the query inside a select,
        # then append eager joins onto that

        if self.order_by:
            # the default coercion for ORDER BY is now the OrderByRole,
            # which adds an additional post coercion to ByOfRole in that
            # elements are converted into label references.  For the
            # eager load / subquery wrapping case, we need to un-coerce
            # the original expressions outside of the label references
            # in order to have them render.
            unwrapped_order_by = [
                (
                    elem.element
                    if isinstance(elem, sql.elements._label_reference)
                    else elem
                )
                for elem in self.order_by
            ]

            order_by_col_expr = sql_util.expand_column_list_from_order_by(
                self.primary_columns, unwrapped_order_by
            )
        else:
            order_by_col_expr = []
            unwrapped_order_by = None

        # put FOR UPDATE on the inner query, where MySQL will honor it,
        # as well as if it has an OF so PostgreSQL can use it.
        inner = self._select_statement(
            self.primary_columns
            + [c for c in order_by_col_expr if c not in self.dedupe_columns],
            self.from_clauses,
            self._where_criteria,
            self._having_criteria,
            self.label_style,
            self.order_by,
            for_update=self._for_update_arg,
            hints=self.select_statement._hints,
            statement_hints=self.select_statement._statement_hints,
            correlate=self.correlate,
            correlate_except=self.correlate_except,
            **self._select_args,
        )

        inner = inner.alias()

        equivs = self._all_equivs()

        self.compound_eager_adapter = ORMStatementAdapter(
            _TraceAdaptRole.COMPOUND_EAGER_STATEMENT, inner, equivalents=equivs
        )

        statement = future.select(
            *([inner] + self.secondary_columns)  # use_labels=self.labels
        )
        statement._label_style = self.label_style

        # Oracle Database however does not allow FOR UPDATE on the subquery,
        # and the Oracle Database dialects ignore it, plus for PostgreSQL,
        # MySQL we expect that all elements of the row are locked, so also put
        # it on the outside (except in the case of PG when OF is used)
        if (
            self._for_update_arg is not None
            and self._for_update_arg.of is None
        ):
            statement._for_update_arg = self._for_update_arg

        from_clause = inner
        for eager_join in self.eager_joins.values():
            # EagerLoader places a 'stop_on' attribute on the join,
            # giving us a marker as to where the "splice point" of
            # the join should be
            from_clause = sql_util.splice_joins(
                from_clause, eager_join, eager_join.stop_on
            )

        statement.select_from.non_generative(statement, from_clause)

        if unwrapped_order_by:
            statement.order_by.non_generative(
                statement,
                *self.compound_eager_adapter.copy_and_process(
                    unwrapped_order_by
                ),
            )

        statement.order_by.non_generative(statement, *self.eager_order_by)
        return statement

    def _simple_statement(self):
        statement = self._select_statement(
            self.primary_columns + self.secondary_columns,
            tuple(self.from_clauses) + tuple(self.eager_joins.values()),
            self._where_criteria,
            self._having_criteria,
            self.label_style,
            self.order_by,
            for_update=self._for_update_arg,
            hints=self.select_statement._hints,
            statement_hints=self.select_statement._statement_hints,
            correlate=self.correlate,
            correlate_except=self.correlate_except,
            **self._select_args,
        )

        if self.eager_order_by:
            statement.order_by.non_generative(statement, *self.eager_order_by)
        return statement

    def _select_statement(
        self,
        raw_columns,
        from_obj,
        where_criteria,
        having_criteria,
        label_style,
        order_by,
        for_update,
        hints,
        statement_hints,
        correlate,
        correlate_except,
        limit_clause,
        offset_clause,
        fetch_clause,
        fetch_clause_options,
        distinct,
        distinct_on,
        prefixes,
        suffixes,
        group_by,
        independent_ctes,
        independent_ctes_opts,
    ):
        statement = Select._create_raw_select(
            _raw_columns=raw_columns,
            _from_obj=from_obj,
            _label_style=label_style,
        )

        if where_criteria:
            statement._where_criteria = where_criteria
        if having_criteria:
            statement._having_criteria = having_criteria

        if order_by:
            statement._order_by_clauses += tuple(order_by)

        if distinct_on:
            statement.distinct.non_generative(statement, *distinct_on)
        elif distinct:
            statement.distinct.non_generative(statement)

        if group_by:
            statement._group_by_clauses += tuple(group_by)

        statement._limit_clause = limit_clause
        statement._offset_clause = offset_clause
        statement._fetch_clause = fetch_clause
        statement._fetch_clause_options = fetch_clause_options
        statement._independent_ctes = independent_ctes
        statement._independent_ctes_opts = independent_ctes_opts

        if prefixes:
            statement._prefixes = prefixes

        if suffixes:
            statement._suffixes = suffixes

        statement._for_update_arg = for_update

        if hints:
            statement._hints = hints
        if statement_hints:
            statement._statement_hints = statement_hints

        if correlate:
            statement.correlate.non_generative(statement, *correlate)

        if correlate_except is not None:
            statement.correlate_except.non_generative(
                statement, *correlate_except
            )

        return statement

    def _adapt_polymorphic_element(self, element):
        if "parententity" in element._annotations:
            search = element._annotations["parententity"]
            alias = self._polymorphic_adapters.get(search, None)
            if alias:
                return alias.adapt_clause(element)

        if isinstance(element, expression.FromClause):
            search = element
        elif hasattr(element, "table"):
            search = element.table
        else:
            return None

        alias = self._polymorphic_adapters.get(search, None)
        if alias:
            return alias.adapt_clause(element)

    def _adapt_col_list(self, cols, current_adapter):
        if current_adapter:
            return [current_adapter(o, True) for o in cols]
        else:
            return cols

    def _get_current_adapter(self):
        adapters = []

        if self._from_obj_alias:
            # used for legacy going forward for query set_ops, e.g.
            # union(), union_all(), etc.
            # 1.4 and previously, also used for from_self(),
            # select_entity_from()
            #
            # for the "from obj" alias, apply extra rule to the
            # 'ORM only' check, if this query were generated from a
            # subquery of itself, i.e. _from_selectable(), apply adaption
            # to all SQL constructs.
            adapters.append(
                (
                    True,
                    self._from_obj_alias.replace,
                )
            )

        # this was *hopefully* the only adapter we were going to need
        # going forward...however, we unfortunately need _from_obj_alias
        # for query.union(), which we can't drop
        if self._polymorphic_adapters:
            adapters.append((False, self._adapt_polymorphic_element))

        if not adapters:
            return None

        def _adapt_clause(clause, as_filter):
            # do we adapt all expression elements or only those
            # tagged as 'ORM' constructs ?

            def replace(elem):
                is_orm_adapt = (
                    "_orm_adapt" in elem._annotations
                    or "parententity" in elem._annotations
                )
                for always_adapt, adapter in adapters:
                    if is_orm_adapt or always_adapt:
                        e = adapter(elem)
                        if e is not None:
                            return e

            return visitors.replacement_traverse(clause, {}, replace)

        return _adapt_clause

    def _join(self, args, entities_collection):
        for right, onclause, from_, flags in args:
            isouter = flags["isouter"]
            full = flags["full"]

            right = inspect(right)
            if onclause is not None:
                onclause = inspect(onclause)

            if isinstance(right, interfaces.PropComparator):
                if onclause is not None:
                    raise sa_exc.InvalidRequestError(
                        "No 'on clause' argument may be passed when joining "
                        "to a relationship path as a target"
                    )

                onclause = right
                right = None
            elif "parententity" in right._annotations:
                right = right._annotations["parententity"]

            if onclause is None:
                if not right.is_selectable and not hasattr(right, "mapper"):
                    raise sa_exc.ArgumentError(
                        "Expected mapped entity or "
                        "selectable/table as join target"
                    )

            if isinstance(onclause, interfaces.PropComparator):
                # descriptor/property given (or determined); this tells us
                # explicitly what the expected "left" side of the join is.

                of_type = getattr(onclause, "_of_type", None)

                if right is None:
                    if of_type:
                        right = of_type
                    else:
                        right = onclause.property

                        try:
                            right = right.entity
                        except AttributeError as err:
                            raise sa_exc.ArgumentError(
                                "Join target %s does not refer to a "
                                "mapped entity" % right
                            ) from err

                left = onclause._parententity

                prop = onclause.property
                if not isinstance(onclause, attributes.QueryableAttribute):
                    onclause = prop

                # check for this path already present.  don't render in that
                # case.
                if (left, right, prop.key) in self._already_joined_edges:
                    continue

                if from_ is not None:
                    if (
                        from_ is not left
                        and from_._annotations.get("parententity", None)
                        is not left
                    ):
                        raise sa_exc.InvalidRequestError(
                            "explicit from clause %s does not match left side "
                            "of relationship attribute %s"
                            % (
                                from_._annotations.get("parententity", from_),
                                onclause,
                            )
                        )
            elif from_ is not None:
                prop = None
                left = from_
            else:
                # no descriptor/property given; we will need to figure out
                # what the effective "left" side is
                prop = left = None

            # figure out the final "left" and "right" sides and create an
            # ORMJoin to add to our _from_obj tuple
            self._join_left_to_right(
                entities_collection,
                left,
                right,
                onclause,
                prop,
                isouter,
                full,
            )

    def _join_left_to_right(
        self,
        entities_collection,
        left,
        right,
        onclause,
        prop,
        outerjoin,
        full,
    ):
        """given raw "left", "right", "onclause" parameters consumed from
        a particular key within _join(), add a real ORMJoin object to
        our _from_obj list (or augment an existing one)

        """

        if left is None:
            # left not given (e.g. no relationship object/name specified)
            # figure out the best "left" side based on our existing froms /
            # entities
            assert prop is None
            (
                left,
                replace_from_obj_index,
                use_entity_index,
            ) = self._join_determine_implicit_left_side(
                entities_collection, left, right, onclause
            )
        else:
            # left is given via a relationship/name, or as explicit left side.
            # Determine where in our
            # "froms" list it should be spliced/appended as well as what
            # existing entity it corresponds to.
            (
                replace_from_obj_index,
                use_entity_index,
            ) = self._join_place_explicit_left_side(entities_collection, left)

        if left is right:
            raise sa_exc.InvalidRequestError(
                "Can't construct a join from %s to %s, they "
                "are the same entity" % (left, right)
            )

        # the right side as given often needs to be adapted.  additionally
        # a lot of things can be wrong with it.  handle all that and
        # get back the new effective "right" side
        r_info, right, onclause = self._join_check_and_adapt_right_side(
            left, right, onclause, prop
        )

        if not r_info.is_selectable:
            extra_criteria = self._get_extra_criteria(r_info)
        else:
            extra_criteria = ()

        if replace_from_obj_index is not None:
            # splice into an existing element in the
            # self._from_obj list
            left_clause = self.from_clauses[replace_from_obj_index]

            self.from_clauses = (
                self.from_clauses[:replace_from_obj_index]
                + [
                    _ORMJoin(
                        left_clause,
                        right,
                        onclause,
                        isouter=outerjoin,
                        full=full,
                        _extra_criteria=extra_criteria,
                    )
                ]
                + self.from_clauses[replace_from_obj_index + 1 :]
            )
        else:
            # add a new element to the self._from_obj list
            if use_entity_index is not None:
                # make use of _MapperEntity selectable, which is usually
                # entity_zero.selectable, but if with_polymorphic() were used
                # might be distinct
                assert isinstance(
                    entities_collection[use_entity_index], _MapperEntity
                )
                left_clause = entities_collection[use_entity_index].selectable
            else:
                left_clause = left

            self.from_clauses = self.from_clauses + [
                _ORMJoin(
                    left_clause,
                    r_info,
                    onclause,
                    isouter=outerjoin,
                    full=full,
                    _extra_criteria=extra_criteria,
                )
            ]

    def _join_determine_implicit_left_side(
        self, entities_collection, left, right, onclause
    ):
        """When join conditions don't express the left side explicitly,
        determine if an existing FROM or entity in this query
        can serve as the left hand side.

        """

        # when we are here, it means join() was called without an ORM-
        # specific way of telling us what the "left" side is, e.g.:
        #
        # join(RightEntity)
        #
        # or
        #
        # join(RightEntity, RightEntity.foo == LeftEntity.bar)
        #

        r_info = inspect(right)

        replace_from_obj_index = use_entity_index = None

        if self.from_clauses:
            # we have a list of FROMs already.  So by definition this
            # join has to connect to one of those FROMs.

            indexes = sql_util.find_left_clause_to_join_from(
                self.from_clauses, r_info.selectable, onclause
            )

            if len(indexes) == 1:
                replace_from_obj_index = indexes[0]
                left = self.from_clauses[replace_from_obj_index]
            elif len(indexes) > 1:
                raise sa_exc.InvalidRequestError(
                    "Can't determine which FROM clause to join "
                    "from, there are multiple FROMS which can "
                    "join to this entity. Please use the .select_from() "
                    "method to establish an explicit left side, as well as "
                    "providing an explicit ON clause if not present already "
                    "to help resolve the ambiguity."
                )
            else:
                raise sa_exc.InvalidRequestError(
                    "Don't know how to join to %r. "
                    "Please use the .select_from() "
                    "method to establish an explicit left side, as well as "
                    "providing an explicit ON clause if not present already "
                    "to help resolve the ambiguity." % (right,)
                )

        elif entities_collection:
            # we have no explicit FROMs, so the implicit left has to
            # come from our list of entities.

            potential = {}
            for entity_index, ent in enumerate(entities_collection):
                entity = ent.entity_zero_or_selectable
                if entity is None:
                    continue
                ent_info = inspect(entity)
                if ent_info is r_info:  # left and right are the same, skip
                    continue

                # by using a dictionary with the selectables as keys this
                # de-duplicates those selectables as occurs when the query is
                # against a series of columns from the same selectable
                if isinstance(ent, _MapperEntity):
                    potential[ent.selectable] = (entity_index, entity)
                else:
                    potential[ent_info.selectable] = (None, entity)

            all_clauses = list(potential.keys())
            indexes = sql_util.find_left_clause_to_join_from(
                all_clauses, r_info.selectable, onclause
            )

            if len(indexes) == 1:
                use_entity_index, left = potential[all_clauses[indexes[0]]]
            elif len(indexes) > 1:
                raise sa_exc.InvalidRequestError(
                    "Can't determine which FROM clause to join "
                    "from, there are multiple FROMS which can "
                    "join to this entity. Please use the .select_from() "
                    "method to establish an explicit left side, as well as "
                    "providing an explicit ON clause if not present already "
                    "to help resolve the ambiguity."
                )
            else:
                raise sa_exc.InvalidRequestError(
                    "Don't know how to join to %r. "
                    "Please use the .select_from() "
                    "method to establish an explicit left side, as well as "
                    "providing an explicit ON clause if not present already "
                    "to help resolve the ambiguity." % (right,)
                )
        else:
            raise sa_exc.InvalidRequestError(
                "No entities to join from; please use "
                "select_from() to establish the left "
                "entity/selectable of this join"
            )

        return left, replace_from_obj_index, use_entity_index

    def _join_place_explicit_left_side(self, entities_collection, left):
        """When join conditions express a left side explicitly, determine
        where in our existing list of FROM clauses we should join towards,
        or if we need to make a new join, and if so is it from one of our
        existing entities.

        """

        # when we are here, it means join() was called with an indicator
        # as to an exact left side, which means a path to a
        # Relationship was given, e.g.:
        #
        # join(RightEntity, LeftEntity.right)
        #
        # or
        #
        # join(LeftEntity.right)
        #
        # as well as string forms:
        #
        # join(RightEntity, "right")
        #
        # etc.
        #

        replace_from_obj_index = use_entity_index = None

        l_info = inspect(left)
        if self.from_clauses:
            indexes = sql_util.find_left_clause_that_matches_given(
                self.from_clauses, l_info.selectable
            )

            if len(indexes) > 1:
                raise sa_exc.InvalidRequestError(
                    "Can't identify which entity in which to assign the "
                    "left side of this join.   Please use a more specific "
                    "ON clause."
                )

            # have an index, means the left side is already present in
            # an existing FROM in the self._from_obj tuple
            if indexes:
                replace_from_obj_index = indexes[0]

            # no index, means we need to add a new element to the
            # self._from_obj tuple

        # no from element present, so we will have to add to the
        # self._from_obj tuple.  Determine if this left side matches up
        # with existing mapper entities, in which case we want to apply the
        # aliasing / adaptation rules present on that entity if any
        if (
            replace_from_obj_index is None
            and entities_collection
            and hasattr(l_info, "mapper")
        ):
            for idx, ent in enumerate(entities_collection):
                # TODO: should we be checking for multiple mapper entities
                # matching?
                if isinstance(ent, _MapperEntity) and ent.corresponds_to(left):
                    use_entity_index = idx
                    break

        return replace_from_obj_index, use_entity_index

    def _join_check_and_adapt_right_side(self, left, right, onclause, prop):
        """transform the "right" side of the join as well as the onclause
        according to polymorphic mapping translations, aliasing on the query
        or on the join, special cases where the right and left side have
        overlapping tables.

        """

        l_info = inspect(left)
        r_info = inspect(right)

        overlap = False

        right_mapper = getattr(r_info, "mapper", None)
        # if the target is a joined inheritance mapping,
        # be more liberal about auto-aliasing.
        if right_mapper and (
            right_mapper.with_polymorphic
            or isinstance(right_mapper.persist_selectable, expression.Join)
        ):
            for from_obj in self.from_clauses or [l_info.selectable]:
                if sql_util.selectables_overlap(
                    l_info.selectable, from_obj
                ) and sql_util.selectables_overlap(
                    from_obj, r_info.selectable
                ):
                    overlap = True
                    break

        if overlap and l_info.selectable is r_info.selectable:
            raise sa_exc.InvalidRequestError(
                "Can't join table/selectable '%s' to itself"
                % l_info.selectable
            )

        right_mapper, right_selectable, right_is_aliased = (
            getattr(r_info, "mapper", None),
            r_info.selectable,
            getattr(r_info, "is_aliased_class", False),
        )

        if (
            right_mapper
            and prop
            and not right_mapper.common_parent(prop.mapper)
        ):
            raise sa_exc.InvalidRequestError(
                "Join target %s does not correspond to "
                "the right side of join condition %s" % (right, onclause)
            )

        # _join_entities is used as a hint for single-table inheritance
        # purposes at the moment
        if hasattr(r_info, "mapper"):
            self._join_entities += (r_info,)

        need_adapter = False

        # test for joining to an unmapped selectable as the target
        if r_info.is_clause_element:
            if prop:
                right_mapper = prop.mapper

            if right_selectable._is_lateral:
                # orm_only is disabled to suit the case where we have to
                # adapt an explicit correlate(Entity) - the select() loses
                # the ORM-ness in this case right now, ideally it would not
                current_adapter = self._get_current_adapter()
                if current_adapter is not None:
                    # TODO: we had orm_only=False here before, removing
                    # it didn't break things.   if we identify the rationale,
                    # may need to apply "_orm_only" annotation here.
                    right = current_adapter(right, True)

            elif prop:
                # joining to selectable with a mapper property given
                # as the ON clause

                if not right_selectable.is_derived_from(
                    right_mapper.persist_selectable
                ):
                    raise sa_exc.InvalidRequestError(
                        "Selectable '%s' is not derived from '%s'"
                        % (
                            right_selectable.description,
                            right_mapper.persist_selectable.description,
                        )
                    )

                # if the destination selectable is a plain select(),
                # turn it into an alias().
                if isinstance(right_selectable, expression.SelectBase):
                    right_selectable = coercions.expect(
                        roles.FromClauseRole, right_selectable
                    )
                    need_adapter = True

                # make the right hand side target into an ORM entity
                right = AliasedClass(right_mapper, right_selectable)

                util.warn_deprecated(
                    "An alias is being generated automatically against "
                    "joined entity %s for raw clauseelement, which is "
                    "deprecated and will be removed in a later release. "
                    "Use the aliased() "
                    "construct explicitly, see the linked example."
                    % right_mapper,
                    "1.4",
                    code="xaj1",
                )

        # test for overlap:
        # orm/inheritance/relationships.py
        # SelfReferentialM2MTest
        aliased_entity = right_mapper and not right_is_aliased and overlap

        if not need_adapter and aliased_entity:
            # there are a few places in the ORM that automatic aliasing
            # is still desirable, and can't be automatic with a Core
            # only approach.  For illustrations of "overlaps" see
            # test/orm/inheritance/test_relationships.py.  There are also
            # general overlap cases with many-to-many tables where automatic
            # aliasing is desirable.
            right = AliasedClass(right, flat=True)
            need_adapter = True

            util.warn(
                "An alias is being generated automatically against "
                "joined entity %s due to overlapping tables.  This is a "
                "legacy pattern which may be "
                "deprecated in a later release.  Use the "
                "aliased(<entity>, flat=True) "
                "construct explicitly, see the linked example." % right_mapper,
                code="xaj2",
            )

        if need_adapter:
            # if need_adapter is True, we are in a deprecated case and
            # a warning has been emitted.
            assert right_mapper

            adapter = ORMAdapter(
                _TraceAdaptRole.DEPRECATED_JOIN_ADAPT_RIGHT_SIDE,
                inspect(right),
                equivalents=right_mapper._equivalent_columns,
            )

            # if an alias() on the right side was generated,
            # which is intended to wrap a the right side in a subquery,
            # ensure that columns retrieved from this target in the result
            # set are also adapted.
            self._mapper_loads_polymorphically_with(right_mapper, adapter)
        elif (
            not r_info.is_clause_element
            and not right_is_aliased
            and right_mapper._has_aliased_polymorphic_fromclause
        ):
            # for the case where the target mapper has a with_polymorphic
            # set up, ensure an adapter is set up for criteria that works
            # against this mapper.  Previously, this logic used to
            # use the "create_aliases or aliased_entity" case to generate
            # an aliased() object, but this creates an alias that isn't
            # strictly necessary.
            # see test/orm/test_core_compilation.py
            # ::RelNaturalAliasedJoinsTest::test_straight
            # and similar
            self._mapper_loads_polymorphically_with(
                right_mapper,
                ORMAdapter(
                    _TraceAdaptRole.WITH_POLYMORPHIC_ADAPTER_RIGHT_JOIN,
                    right_mapper,
                    selectable=right_mapper.selectable,
                    equivalents=right_mapper._equivalent_columns,
                ),
            )
        # if the onclause is a ClauseElement, adapt it with any
        # adapters that are in place right now
        if isinstance(onclause, expression.ClauseElement):
            current_adapter = self._get_current_adapter()
            if current_adapter:
                onclause = current_adapter(onclause, True)

        # if joining on a MapperProperty path,
        # track the path to prevent redundant joins
        if prop:
            self._already_joined_edges += ((left, right, prop.key),)

        return inspect(right), right, onclause

    @property
    def _select_args(self):
        return {
            "limit_clause": self.select_statement._limit_clause,
            "offset_clause": self.select_statement._offset_clause,
            "distinct": self.distinct,
            "distinct_on": self.distinct_on,
            "prefixes": self.select_statement._prefixes,
            "suffixes": self.select_statement._suffixes,
            "group_by": self.group_by or None,
            "fetch_clause": self.select_statement._fetch_clause,
            "fetch_clause_options": (
                self.select_statement._fetch_clause_options
            ),
            "independent_ctes": self.select_statement._independent_ctes,
            "independent_ctes_opts": (
                self.select_statement._independent_ctes_opts
            ),
        }

    @property
    def _should_nest_selectable(self):
        kwargs = self._select_args
        return (
            kwargs.get("limit_clause") is not None
            or kwargs.get("offset_clause") is not None
            or kwargs.get("distinct", False)
            or kwargs.get("distinct_on", ())
            or kwargs.get("group_by", False)
        )

    def _get_extra_criteria(self, ext_info):
        if (
            "additional_entity_criteria",
            ext_info.mapper,
        ) in self.global_attributes:
            return tuple(
                ae._resolve_where_criteria(ext_info)
                for ae in self.global_attributes[
                    ("additional_entity_criteria", ext_info.mapper)
                ]
                if (ae.include_aliases or ae.entity is ext_info)
                and ae._should_include(self)
            )
        else:
            return ()

    def _adjust_for_extra_criteria(self):
        """Apply extra criteria filtering.

        For all distinct single-table-inheritance mappers represented in
        the columns clause of this query, as well as the "select from entity",
        add criterion to the WHERE
        clause of the given QueryContext such that only the appropriate
        subtypes are selected from the total results.

        Additionally, add WHERE criteria originating from LoaderCriteriaOptions
        associated with the global context.

        """

        for fromclause in self.from_clauses:
            ext_info = fromclause._annotations.get("parententity", None)

            if (
                ext_info
                and (
                    ext_info.mapper._single_table_criterion is not None
                    or ("additional_entity_criteria", ext_info.mapper)
                    in self.global_attributes
                )
                and ext_info not in self.extra_criteria_entities
            ):
                self.extra_criteria_entities[ext_info] = (
                    ext_info,
                    ext_info._adapter if ext_info.is_aliased_class else None,
                )

        search = set(self.extra_criteria_entities.values())

        for ext_info, adapter in search:
            if ext_info in self._join_entities:
                continue

            single_crit = ext_info.mapper._single_table_criterion

            if self.compile_options._for_refresh_state:
                additional_entity_criteria = []
            else:
                additional_entity_criteria = self._get_extra_criteria(ext_info)

            if single_crit is not None:
                additional_entity_criteria += (single_crit,)

            current_adapter = self._get_current_adapter()
            for crit in additional_entity_criteria:
                if adapter:
                    crit = adapter.traverse(crit)

                if current_adapter:
                    crit = sql_util._deep_annotate(crit, {"_orm_adapt": True})
                    crit = current_adapter(crit, False)
                self._where_criteria += (crit,)


def _column_descriptions(
    query_or_select_stmt: Union[Query, Select, FromStatement],
    compile_state: Optional[ORMSelectCompileState] = None,
    legacy: bool = False,
) -> List[ORMColumnDescription]:
    if compile_state is None:
        compile_state = ORMSelectCompileState._create_entities_collection(
            query_or_select_stmt, legacy=legacy
        )
    ctx = compile_state
    d = [
        {
            "name": ent._label_name,
            "type": ent.type,
            "aliased": getattr(insp_ent, "is_aliased_class", False),
            "expr": ent.expr,
            "entity": (
                getattr(insp_ent, "entity", None)
                if ent.entity_zero is not None
                and not insp_ent.is_clause_element
                else None
            ),
        }
        for ent, insp_ent in [
            (_ent, _ent.entity_zero) for _ent in ctx._entities
        ]
    ]
    return d


def _legacy_filter_by_entity_zero(
    query_or_augmented_select: Union[Query[Any], Select[Any]]
) -> Optional[_InternalEntityType[Any]]:
    self = query_or_augmented_select
    if self._setup_joins:
        _last_joined_entity = self._last_joined_entity
        if _last_joined_entity is not None:
            return _last_joined_entity

    if self._from_obj and "parententity" in self._from_obj[0]._annotations:
        return self._from_obj[0]._annotations["parententity"]

    return _entity_from_pre_ent_zero(self)


def _entity_from_pre_ent_zero(
    query_or_augmented_select: Union[Query[Any], Select[Any]]
) -> Optional[_InternalEntityType[Any]]:
    self = query_or_augmented_select
    if not self._raw_columns:
        return None

    ent = self._raw_columns[0]

    if "parententity" in ent._annotations:
        return ent._annotations["parententity"]
    elif isinstance(ent, ORMColumnsClauseRole):
        return ent.entity
    elif "bundle" in ent._annotations:
        return ent._annotations["bundle"]
    else:
        return ent


def _determine_last_joined_entity(
    setup_joins: Tuple[_SetupJoinsElement, ...],
    entity_zero: Optional[_InternalEntityType[Any]] = None,
) -> Optional[Union[_InternalEntityType[Any], _JoinTargetElement]]:
    if not setup_joins:
        return None

    (target, onclause, from_, flags) = setup_joins[-1]

    if isinstance(
        target,
        attributes.QueryableAttribute,
    ):
        return target.entity
    else:
        return target


class _QueryEntity:
    """represent an entity column returned within a Query result."""

    __slots__ = ()

    supports_single_entity: bool

    _non_hashable_value = False
    _null_column_type = False
    use_id_for_hash = False

    _label_name: Optional[str]
    type: Union[Type[Any], TypeEngine[Any]]
    expr: Union[_InternalEntityType, ColumnElement[Any]]
    entity_zero: Optional[_InternalEntityType]

    def setup_compile_state(self, compile_state: ORMCompileState) -> None:
        raise NotImplementedError()

    def setup_dml_returning_compile_state(
        self,
        compile_state: ORMCompileState,
        adapter: Optional[_DMLReturningColFilter],
    ) -> None:
        raise NotImplementedError()

    def row_processor(self, context, result):
        raise NotImplementedError()

    @classmethod
    def to_compile_state(
        cls, compile_state, entities, entities_collection, is_current_entities
    ):
        for idx, entity in enumerate(entities):
            if entity._is_lambda_element:
                if entity._is_sequence:
                    cls.to_compile_state(
                        compile_state,
                        entity._resolved,
                        entities_collection,
                        is_current_entities,
                    )
                    continue
                else:
                    entity = entity._resolved

            if entity.is_clause_element:
                if entity.is_selectable:
                    if "parententity" in entity._annotations:
                        _MapperEntity(
                            compile_state,
                            entity,
                            entities_collection,
                            is_current_entities,
                        )
                    else:
                        _ColumnEntity._for_columns(
                            compile_state,
                            entity._select_iterable,
                            entities_collection,
                            idx,
                            is_current_entities,
                        )
                else:
                    if entity._annotations.get("bundle", False):
                        _BundleEntity(
                            compile_state,
                            entity,
                            entities_collection,
                            is_current_entities,
                        )
                    elif entity._is_clause_list:
                        # this is legacy only - test_composites.py
                        # test_query_cols_legacy
                        _ColumnEntity._for_columns(
                            compile_state,
                            entity._select_iterable,
                            entities_collection,
                            idx,
                            is_current_entities,
                        )
                    else:
                        _ColumnEntity._for_columns(
                            compile_state,
                            [entity],
                            entities_collection,
                            idx,
                            is_current_entities,
                        )
            elif entity.is_bundle:
                _BundleEntity(compile_state, entity, entities_collection)

        return entities_collection


class _MapperEntity(_QueryEntity):
    """mapper/class/AliasedClass entity"""

    __slots__ = (
        "expr",
        "mapper",
        "entity_zero",
        "is_aliased_class",
        "path",
        "_extra_entities",
        "_label_name",
        "_with_polymorphic_mappers",
        "selectable",
        "_polymorphic_discriminator",
    )

    expr: _InternalEntityType
    mapper: Mapper[Any]
    entity_zero: _InternalEntityType
    is_aliased_class: bool
    path: PathRegistry
    _label_name: str

    def __init__(
        self, compile_state, entity, entities_collection, is_current_entities
    ):
        entities_collection.append(self)
        if is_current_entities:
            if compile_state._primary_entity is None:
                compile_state._primary_entity = self
            compile_state._has_mapper_entities = True
            compile_state._has_orm_entities = True

        entity = entity._annotations["parententity"]
        entity._post_inspect
        ext_info = self.entity_zero = entity
        entity = ext_info.entity

        self.expr = entity
        self.mapper = mapper = ext_info.mapper

        self._extra_entities = (self.expr,)

        if ext_info.is_aliased_class:
            self._label_name = ext_info.name
        else:
            self._label_name = mapper.class_.__name__

        self.is_aliased_class = ext_info.is_aliased_class
        self.path = ext_info._path_registry

        self.selectable = ext_info.selectable
        self._with_polymorphic_mappers = ext_info.with_polymorphic_mappers
        self._polymorphic_discriminator = ext_info.polymorphic_on

        if mapper._should_select_with_poly_adapter:
            compile_state._create_with_polymorphic_adapter(
                ext_info, self.selectable
            )

    supports_single_entity = True

    _non_hashable_value = True
    use_id_for_hash = True

    @property
    def type(self):
        return self.mapper.class_

    @property
    def entity_zero_or_selectable(self):
        return self.entity_zero

    def corresponds_to(self, entity):
        return _entity_corresponds_to(self.entity_zero, entity)

    def _get_entity_clauses(self, compile_state):
        adapter = None

        if not self.is_aliased_class:
            if compile_state._polymorphic_adapters:
                adapter = compile_state._polymorphic_adapters.get(
                    self.mapper, None
                )
        else:
            adapter = self.entity_zero._adapter

        if adapter:
            if compile_state._from_obj_alias:
                ret = adapter.wrap(compile_state._from_obj_alias)
            else:
                ret = adapter
        else:
            ret = compile_state._from_obj_alias

        return ret

    def row_processor(self, context, result):
        compile_state = context.compile_state
        adapter = self._get_entity_clauses(compile_state)

        if compile_state.compound_eager_adapter and adapter:
            adapter = adapter.wrap(compile_state.compound_eager_adapter)
        elif not adapter:
            adapter = compile_state.compound_eager_adapter

        if compile_state._primary_entity is self:
            only_load_props = compile_state.compile_options._only_load_props
            refresh_state = context.refresh_state
        else:
            only_load_props = refresh_state = None

        _instance = loading._instance_processor(
            self,
            self.mapper,
            context,
            result,
            self.path,
            adapter,
            only_load_props=only_load_props,
            refresh_state=refresh_state,
            polymorphic_discriminator=self._polymorphic_discriminator,
        )

        return _instance, self._label_name, self._extra_entities

    def setup_dml_returning_compile_state(
        self,
        compile_state: ORMCompileState,
        adapter: Optional[_DMLReturningColFilter],
    ) -> None:
        loading._setup_entity_query(
            compile_state,
            self.mapper,
            self,
            self.path,
            adapter,
            compile_state.primary_columns,
            with_polymorphic=self._with_polymorphic_mappers,
            only_load_props=compile_state.compile_options._only_load_props,
            polymorphic_discriminator=self._polymorphic_discriminator,
        )

    def setup_compile_state(self, compile_state):
        adapter = self._get_entity_clauses(compile_state)

        single_table_crit = self.mapper._single_table_criterion
        if (
            single_table_crit is not None
            or ("additional_entity_criteria", self.mapper)
            in compile_state.global_attributes
        ):
            ext_info = self.entity_zero
            compile_state.extra_criteria_entities[ext_info] = (
                ext_info,
                ext_info._adapter if ext_info.is_aliased_class else None,
            )

        loading._setup_entity_query(
            compile_state,
            self.mapper,
            self,
            self.path,
            adapter,
            compile_state.primary_columns,
            with_polymorphic=self._with_polymorphic_mappers,
            only_load_props=compile_state.compile_options._only_load_props,
            polymorphic_discriminator=self._polymorphic_discriminator,
        )
        compile_state._fallback_from_clauses.append(self.selectable)


class _BundleEntity(_QueryEntity):
    _extra_entities = ()

    __slots__ = (
        "bundle",
        "expr",
        "type",
        "_label_name",
        "_entities",
        "supports_single_entity",
    )

    _entities: List[_QueryEntity]
    bundle: Bundle
    type: Type[Any]
    _label_name: str
    supports_single_entity: bool
    expr: Bundle

    def __init__(
        self,
        compile_state,
        expr,
        entities_collection,
        is_current_entities,
        setup_entities=True,
        parent_bundle=None,
    ):
        compile_state._has_orm_entities = True

        expr = expr._annotations["bundle"]
        if parent_bundle:
            parent_bundle._entities.append(self)
        else:
            entities_collection.append(self)

        if isinstance(
            expr, (attributes.QueryableAttribute, interfaces.PropComparator)
        ):
            bundle = expr.__clause_element__()
        else:
            bundle = expr

        self.bundle = self.expr = bundle
        self.type = type(bundle)
        self._label_name = bundle.name
        self._entities = []

        if setup_entities:
            for expr in bundle.exprs:
                if "bundle" in expr._annotations:
                    _BundleEntity(
                        compile_state,
                        expr,
                        entities_collection,
                        is_current_entities,
                        parent_bundle=self,
                    )
                elif isinstance(expr, Bundle):
                    _BundleEntity(
                        compile_state,
                        expr,
                        entities_collection,
                        is_current_entities,
                        parent_bundle=self,
                    )
                else:
                    _ORMColumnEntity._for_columns(
                        compile_state,
                        [expr],
                        entities_collection,
                        None,
                        is_current_entities,
                        parent_bundle=self,
                    )

        self.supports_single_entity = self.bundle.single_entity

    @property
    def mapper(self):
        ezero = self.entity_zero
        if ezero is not None:
            return ezero.mapper
        else:
            return None

    @property
    def entity_zero(self):
        for ent in self._entities:
            ezero = ent.entity_zero
            if ezero is not None:
                return ezero
        else:
            return None

    def corresponds_to(self, entity):
        # TODO: we might be able to implement this but for now
        # we are working around it
        return False

    @property
    def entity_zero_or_selectable(self):
        for ent in self._entities:
            ezero = ent.entity_zero_or_selectable
            if ezero is not None:
                return ezero
        else:
            return None

    def setup_compile_state(self, compile_state):
        for ent in self._entities:
            ent.setup_compile_state(compile_state)

    def setup_dml_returning_compile_state(
        self,
        compile_state: ORMCompileState,
        adapter: Optional[_DMLReturningColFilter],
    ) -> None:
        return self.setup_compile_state(compile_state)

    def row_processor(self, context, result):
        procs, labels, extra = zip(
            *[ent.row_processor(context, result) for ent in self._entities]
        )

        proc = self.bundle.create_row_processor(context.query, procs, labels)

        return proc, self._label_name, self._extra_entities


class _ColumnEntity(_QueryEntity):
    __slots__ = (
        "_fetch_column",
        "_row_processor",
        "raw_column_index",
        "translate_raw_column",
    )

    @classmethod
    def _for_columns(
        cls,
        compile_state,
        columns,
        entities_collection,
        raw_column_index,
        is_current_entities,
        parent_bundle=None,
    ):
        for column in columns:
            annotations = column._annotations
            if "parententity" in annotations:
                _entity = annotations["parententity"]
            else:
                _entity = sql_util.extract_first_column_annotation(
                    column, "parententity"
                )

            if _entity:
                if "identity_token" in column._annotations:
                    _IdentityTokenEntity(
                        compile_state,
                        column,
                        entities_collection,
                        _entity,
                        raw_column_index,
                        is_current_entities,
                        parent_bundle=parent_bundle,
                    )
                else:
                    _ORMColumnEntity(
                        compile_state,
                        column,
                        entities_collection,
                        _entity,
                        raw_column_index,
                        is_current_entities,
                        parent_bundle=parent_bundle,
                    )
            else:
                _RawColumnEntity(
                    compile_state,
                    column,
                    entities_collection,
                    raw_column_index,
                    is_current_entities,
                    parent_bundle=parent_bundle,
                )

    @property
    def type(self):
        return self.column.type

    @property
    def _non_hashable_value(self):
        return not self.column.type.hashable

    @property
    def _null_column_type(self):
        return self.column.type._isnull

    def row_processor(self, context, result):
        compile_state = context.compile_state

        # the resulting callable is entirely cacheable so just return
        # it if we already made one
        if self._row_processor is not None:
            getter, label_name, extra_entities = self._row_processor
            if self.translate_raw_column:
                extra_entities += (
                    context.query._raw_columns[self.raw_column_index],
                )

            return getter, label_name, extra_entities

        # retrieve the column that would have been set up in
        # setup_compile_state, to avoid doing redundant work
        if self._fetch_column is not None:
            column = self._fetch_column
        else:
            # fetch_column will be None when we are doing a from_statement
            # and setup_compile_state may not have been called.
            column = self.column

            # previously, the RawColumnEntity didn't look for from_obj_alias
            # however I can't think of a case where we would be here and
            # we'd want to ignore it if this is the from_statement use case.
            # it's not really a use case to have raw columns + from_statement
            if compile_state._from_obj_alias:
                column = compile_state._from_obj_alias.columns[column]

            if column._annotations:
                # annotated columns perform more slowly in compiler and
                # result due to the __eq__() method, so use deannotated
                column = column._deannotate()

        if compile_state.compound_eager_adapter:
            column = compile_state.compound_eager_adapter.columns[column]

        getter = result._getter(column)
        ret = getter, self._label_name, self._extra_entities
        self._row_processor = ret

        if self.translate_raw_column:
            extra_entities = self._extra_entities + (
                context.query._raw_columns[self.raw_column_index],
            )
            return getter, self._label_name, extra_entities
        else:
            return ret


class _RawColumnEntity(_ColumnEntity):
    entity_zero = None
    mapper = None
    supports_single_entity = False

    __slots__ = (
        "expr",
        "column",
        "_label_name",
        "entity_zero_or_selectable",
        "_extra_entities",
    )

    def __init__(
        self,
        compile_state,
        column,
        entities_collection,
        raw_column_index,
        is_current_entities,
        parent_bundle=None,
    ):
        self.expr = column
        self.raw_column_index = raw_column_index
        self.translate_raw_column = raw_column_index is not None

        if column._is_star:
            compile_state.compile_options += {"_is_star": True}

        if not is_current_entities or column._is_text_clause:
            self._label_name = None
        else:
            if parent_bundle:
                self._label_name = column._proxy_key
            else:
                self._label_name = compile_state._label_convention(column)

        if parent_bundle:
            parent_bundle._entities.append(self)
        else:
            entities_collection.append(self)

        self.column = column
        self.entity_zero_or_selectable = (
            self.column._from_objects[0] if self.column._from_objects else None
        )
        self._extra_entities = (self.expr, self.column)
        self._fetch_column = self._row_processor = None

    def corresponds_to(self, entity):
        return False

    def setup_dml_returning_compile_state(
        self,
        compile_state: ORMCompileState,
        adapter: Optional[_DMLReturningColFilter],
    ) -> None:
        return self.setup_compile_state(compile_state)

    def setup_compile_state(self, compile_state):
        current_adapter = compile_state._get_current_adapter()
        if current_adapter:
            column = current_adapter(self.column, False)
            if column is None:
                return
        else:
            column = self.column

        if column._annotations:
            # annotated columns perform more slowly in compiler and
            # result due to the __eq__() method, so use deannotated
            column = column._deannotate()

        compile_state.dedupe_columns.add(column)
        compile_state.primary_columns.append(column)
        self._fetch_column = column


class _ORMColumnEntity(_ColumnEntity):
    """Column/expression based entity."""

    supports_single_entity = False

    __slots__ = (
        "expr",
        "mapper",
        "column",
        "_label_name",
        "entity_zero_or_selectable",
        "entity_zero",
        "_extra_entities",
    )

    def __init__(
        self,
        compile_state,
        column,
        entities_collection,
        parententity,
        raw_column_index,
        is_current_entities,
        parent_bundle=None,
    ):
        annotations = column._annotations

        _entity = parententity

        # an AliasedClass won't have proxy_key in the annotations for
        # a column if it was acquired using the class' adapter directly,
        # such as using AliasedInsp._adapt_element().  this occurs
        # within internal loaders.

        orm_key = annotations.get("proxy_key", None)
        proxy_owner = annotations.get("proxy_owner", _entity)
        if orm_key:
            self.expr = getattr(proxy_owner.entity, orm_key)
            self.translate_raw_column = False
        else:
            # if orm_key is not present, that means this is an ad-hoc
            # SQL ColumnElement, like a CASE() or other expression.
            # include this column position from the invoked statement
            # in the ORM-level ResultSetMetaData on each execute, so that
            # it can be targeted by identity after caching
            self.expr = column
            self.translate_raw_column = raw_column_index is not None

        self.raw_column_index = raw_column_index

        if is_current_entities:
            if parent_bundle:
                self._label_name = orm_key if orm_key else column._proxy_key
            else:
                self._label_name = compile_state._label_convention(
                    column, col_name=orm_key
                )
        else:
            self._label_name = None

        _entity._post_inspect
        self.entity_zero = self.entity_zero_or_selectable = ezero = _entity
        self.mapper = mapper = _entity.mapper

        if parent_bundle:
            parent_bundle._entities.append(self)
        else:
            entities_collection.append(self)

        compile_state._has_orm_entities = True

        self.column = column

        self._fetch_column = self._row_processor = None

        self._extra_entities = (self.expr, self.column)

        if mapper._should_select_with_poly_adapter:
            compile_state._create_with_polymorphic_adapter(
                ezero, ezero.selectable
            )

    def corresponds_to(self, entity):
        if _is_aliased_class(entity):
            # TODO: polymorphic subclasses ?
            return entity is self.entity_zero
        else:
            return not _is_aliased_class(
                self.entity_zero
            ) and entity.common_parent(self.entity_zero)

    def setup_dml_returning_compile_state(
        self,
        compile_state: ORMCompileState,
        adapter: Optional[_DMLReturningColFilter],
    ) -> None:

        self._fetch_column = column = self.column
        if adapter:
            column = adapter(column, False)

        if column is not None:
            compile_state.dedupe_columns.add(column)
            compile_state.primary_columns.append(column)

    def setup_compile_state(self, compile_state):
        current_adapter = compile_state._get_current_adapter()
        if current_adapter:
            column = current_adapter(self.column, False)
            if column is None:
                assert compile_state.is_dml_returning
                self._fetch_column = self.column
                return
        else:
            column = self.column

        ezero = self.entity_zero

        single_table_crit = self.mapper._single_table_criterion
        if (
            single_table_crit is not None
            or ("additional_entity_criteria", self.mapper)
            in compile_state.global_attributes
        ):
            compile_state.extra_criteria_entities[ezero] = (
                ezero,
                ezero._adapter if ezero.is_aliased_class else None,
            )

        if column._annotations and not column._expression_label:
            # annotated columns perform more slowly in compiler and
            # result due to the __eq__() method, so use deannotated
            column = column._deannotate()

        # use entity_zero as the from if we have it. this is necessary
        # for polymorphic scenarios where our FROM is based on ORM entity,
        # not the FROM of the column.  but also, don't use it if our column
        # doesn't actually have any FROMs that line up, such as when its
        # a scalar subquery.
        if set(self.column._from_objects).intersection(
            ezero.selectable._from_objects
        ):
            compile_state._fallback_from_clauses.append(ezero.selectable)

        compile_state.dedupe_columns.add(column)
        compile_state.primary_columns.append(column)
        self._fetch_column = column


class _IdentityTokenEntity(_ORMColumnEntity):
    translate_raw_column = False

    def setup_compile_state(self, compile_state):
        pass

    def row_processor(self, context, result):
        def getter(row):
            return context.load_options._identity_token

        return getter, self._label_name, self._extra_entities

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\latex.py ===
"""
A Printer which converts an expression into its LaTeX equivalent.
"""
from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING

import itertools

from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse


# sympy.printing imports
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE

from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str

from sympy.utilities.iterables import has_variety, sift

import re

if TYPE_CHECKING:
    from sympy.tensor.array import NDimArray
    from sympy.vector.basisdependent import BasisDependent

# Hand-picked functions which can be used directly in both LaTeX and MathJax
# Complete list at
# https://docs.mathjax.org/en/latest/tex.html#supported-latex-commands
# This variable only contains those functions which SymPy uses.
accepted_latex_functions = ['arcsin', 'arccos', 'arctan', 'sin', 'cos', 'tan',
                            'sinh', 'cosh', 'tanh', 'sqrt', 'ln', 'log', 'sec',
                            'csc', 'cot', 'coth', 're', 'im', 'frac', 'root',
                            'arg',
                            ]

tex_greek_dictionary = {
    'Alpha': r'\mathrm{A}',
    'Beta': r'\mathrm{B}',
    'Gamma': r'\Gamma',
    'Delta': r'\Delta',
    'Epsilon': r'\mathrm{E}',
    'Zeta': r'\mathrm{Z}',
    'Eta': r'\mathrm{H}',
    'Theta': r'\Theta',
    'Iota': r'\mathrm{I}',
    'Kappa': r'\mathrm{K}',
    'Lambda': r'\Lambda',
    'Mu': r'\mathrm{M}',
    'Nu': r'\mathrm{N}',
    'Xi': r'\Xi',
    'omicron': 'o',
    'Omicron': r'\mathrm{O}',
    'Pi': r'\Pi',
    'Rho': r'\mathrm{P}',
    'Sigma': r'\Sigma',
    'Tau': r'\mathrm{T}',
    'Upsilon': r'\Upsilon',
    'Phi': r'\Phi',
    'Chi': r'\mathrm{X}',
    'Psi': r'\Psi',
    'Omega': r'\Omega',
    'lamda': r'\lambda',
    'Lamda': r'\Lambda',
    'khi': r'\chi',
    'Khi': r'\mathrm{X}',
    'varepsilon': r'\varepsilon',
    'varkappa': r'\varkappa',
    'varphi': r'\varphi',
    'varpi': r'\varpi',
    'varrho': r'\varrho',
    'varsigma': r'\varsigma',
    'vartheta': r'\vartheta',
}

other_symbols = {'aleph', 'beth', 'daleth', 'gimel', 'ell', 'eth', 'hbar',
                     'hslash', 'mho', 'wp'}

# Variable name modifiers
modifier_dict: dict[str, Callable[[str], str]] = {
    # Accents
    'mathring': lambda s: r'\mathring{'+s+r'}',
    'ddddot': lambda s: r'\ddddot{'+s+r'}',
    'dddot': lambda s: r'\dddot{'+s+r'}',
    'ddot': lambda s: r'\ddot{'+s+r'}',
    'dot': lambda s: r'\dot{'+s+r'}',
    'check': lambda s: r'\check{'+s+r'}',
    'breve': lambda s: r'\breve{'+s+r'}',
    'acute': lambda s: r'\acute{'+s+r'}',
    'grave': lambda s: r'\grave{'+s+r'}',
    'tilde': lambda s: r'\tilde{'+s+r'}',
    'hat': lambda s: r'\hat{'+s+r'}',
    'bar': lambda s: r'\bar{'+s+r'}',
    'vec': lambda s: r'\vec{'+s+r'}',
    'prime': lambda s: "{"+s+"}'",
    'prm': lambda s: "{"+s+"}'",
    # Faces
    'bold': lambda s: r'\boldsymbol{'+s+r'}',
    'bm': lambda s: r'\boldsymbol{'+s+r'}',
    'cal': lambda s: r'\mathcal{'+s+r'}',
    'scr': lambda s: r'\mathscr{'+s+r'}',
    'frak': lambda s: r'\mathfrak{'+s+r'}',
    # Brackets
    'norm': lambda s: r'\left\|{'+s+r'}\right\|',
    'avg': lambda s: r'\left\langle{'+s+r'}\right\rangle',
    'abs': lambda s: r'\left|{'+s+r'}\right|',
    'mag': lambda s: r'\left|{'+s+r'}\right|',
}

greek_letters_set = frozenset(greeks)

_between_two_numbers_p = (
    re.compile(r'[0-9][} ]*$'),  # search
    re.compile(r'(\d|\\frac{\d+}{\d+})'),  # match
)


def latex_escape(s: str) -> str:
    """
    Escape a string such that latex interprets it as plaintext.

    We cannot use verbatim easily with mathjax, so escaping is easier.
    Rules from https://tex.stackexchange.com/a/34586/41112.
    """
    s = s.replace('\\', r'\textbackslash')
    for c in '&%$#_{}':
        s = s.replace(c, '\\' + c)
    s = s.replace('~', r'\textasciitilde')
    s = s.replace('^', r'\textasciicircum')
    return s


class LatexPrinter(Printer):
    printmethod = "_latex"

    _default_settings: dict[str, Any] = {
        "full_prec": False,
        "fold_frac_powers": False,
        "fold_func_brackets": False,
        "fold_short_frac": None,
        "inv_trig_style": "abbreviated",
        "itex": False,
        "ln_notation": False,
        "long_frac_ratio": None,
        "mat_delim": "[",
        "mat_str": None,
        "mode": "plain",
        "mul_symbol": None,
        "order": None,
        "symbol_names": {},
        "root_notation": True,
        "mat_symbol_style": "plain",
        "imaginary_unit": "i",
        "gothic_re_im": False,
        "decimal_separator": "period",
        "perm_cyclic": True,
        "parenthesize_super": True,
        "min": None,
        "max": None,
        "diff_operator": "d",
        "adjoint_style": "dagger",
        "disable_split_super_sub": False,
    }

    def __init__(self, settings=None):
        Printer.__init__(self, settings)

        if 'mode' in self._settings:
            valid_modes = ['inline', 'plain', 'equation',
                           'equation*']
            if self._settings['mode'] not in valid_modes:
                raise ValueError("'mode' must be one of 'inline', 'plain', "
                                 "'equation' or 'equation*'")

        if self._settings['fold_short_frac'] is None and \
                self._settings['mode'] == 'inline':
            self._settings['fold_short_frac'] = True

        mul_symbol_table = {
            None: r" ",
            "ldot": r" \,.\, ",
            "dot": r" \cdot ",
            "times": r" \times "
        }
        try:
            self._settings['mul_symbol_latex'] = \
                mul_symbol_table[self._settings['mul_symbol']]
        except KeyError:
            self._settings['mul_symbol_latex'] = \
                self._settings['mul_symbol']
        try:
            self._settings['mul_symbol_latex_numbers'] = \
                mul_symbol_table[self._settings['mul_symbol'] or 'dot']
        except KeyError:
            if (self._settings['mul_symbol'].strip() in
                    ['', ' ', '\\', '\\,', '\\:', '\\;', '\\quad']):
                self._settings['mul_symbol_latex_numbers'] = \
                    mul_symbol_table['dot']
            else:
                self._settings['mul_symbol_latex_numbers'] = \
                    self._settings['mul_symbol']

        self._delim_dict = {'(': ')', '[': ']'}

        imaginary_unit_table = {
            None: r"i",
            "i": r"i",
            "ri": r"\mathrm{i}",
            "ti": r"\text{i}",
            "j": r"j",
            "rj": r"\mathrm{j}",
            "tj": r"\text{j}",
        }
        imag_unit = self._settings['imaginary_unit']
        self._settings['imaginary_unit_latex'] = imaginary_unit_table.get(imag_unit, imag_unit)

        diff_operator_table = {
            None: r"d",
            "d": r"d",
            "rd": r"\mathrm{d}",
            "td": r"\text{d}",
        }
        diff_operator = self._settings['diff_operator']
        self._settings["diff_operator_latex"] = diff_operator_table.get(diff_operator, diff_operator)

    def _add_parens(self, s) -> str:
        return r"\left({}\right)".format(s)

    # TODO: merge this with the above, which requires a lot of test changes
    def _add_parens_lspace(self, s) -> str:
        return r"\left( {}\right)".format(s)

    def parenthesize(self, item, level, is_neg=False, strict=False) -> str:
        prec_val = precedence_traditional(item)
        if is_neg and strict:
            return self._add_parens(self._print(item))

        if (prec_val < level) or ((not strict) and prec_val <= level):
            return self._add_parens(self._print(item))
        else:
            return self._print(item)

    def parenthesize_super(self, s):
        """
        Protect superscripts in s

        If the parenthesize_super option is set, protect with parentheses, else
        wrap in braces.
        """
        if "^" in s:
            if self._settings['parenthesize_super']:
                return self._add_parens(s)
            else:
                return "{{{}}}".format(s)
        return s

    def doprint(self, expr) -> str:
        tex = Printer.doprint(self, expr)

        if self._settings['mode'] == 'plain':
            return tex
        elif self._settings['mode'] == 'inline':
            return r"$%s$" % tex
        elif self._settings['itex']:
            return r"$$%s$$" % tex
        else:
            env_str = self._settings['mode']
            return r"\begin{%s}%s\end{%s}" % (env_str, tex, env_str)

    def _needs_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed, False otherwise. For example: a + b => True; a => False;
        10 => False; -10 => True.
        """
        return not ((expr.is_Integer and expr.is_nonnegative)
                    or (expr.is_Atom and (expr is not S.NegativeOne
                                          and expr.is_Rational is False)))

    def _needs_function_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        passed as an argument to a function, False otherwise. This is a more
        liberal version of _needs_brackets, in that many expressions which need
        to be wrapped in brackets when added/subtracted/raised to a power do
        not need them when passed to a function. Such an example is a*b.
        """
        if not self._needs_brackets(expr):
            return False
        else:
            # Muls of the form a*b*c... can be folded
            if expr.is_Mul and not self._mul_is_clean(expr):
                return True
            # Pows which don't need brackets can be folded
            elif expr.is_Pow and not self._pow_is_clean(expr):
                return True
            # Add and Function always need brackets
            elif expr.is_Add or expr.is_Function:
                return True
            else:
                return False

    def _needs_mul_brackets(self, expr, first=False, last=False) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of a Mul, False otherwise. This is True for Add,
        but also for some container objects that would not need brackets
        when appearing last in a Mul, e.g. an Integral. ``last=True``
        specifies that this expr is the last to appear in a Mul.
        ``first=True`` specifies that this expr is the first to appear in
        a Mul.
        """
        from sympy.concrete.products import Product
        from sympy.concrete.summations import Sum
        from sympy.integrals.integrals import Integral

        if expr.is_Mul:
            if not first and expr.could_extract_minus_sign():
                return True
        elif precedence_traditional(expr) < PRECEDENCE["Mul"]:
            return True
        elif expr.is_Relational:
            return True
        if expr.is_Piecewise:
            return True
        if any(expr.has(x) for x in (Mod,)):
            return True
        if (not last and
                any(expr.has(x) for x in (Integral, Product, Sum))):
            return True

        return False

    def _needs_add_brackets(self, expr) -> bool:
        """
        Returns True if the expression needs to be wrapped in brackets when
        printed as part of an Add, False otherwise.  This is False for most
        things.
        """
        if expr.is_Relational:
            return True
        if any(expr.has(x) for x in (Mod,)):
            return True
        if expr.is_Add:
            return True
        return False

    def _mul_is_clean(self, expr) -> bool:
        for arg in expr.args:
            if arg.is_Function:
                return False
        return True

    def _pow_is_clean(self, expr) -> bool:
        return not self._needs_brackets(expr.base)

    def _do_exponent(self, expr: str, exp):
        if exp is not None:
            return r"\left(%s\right)^{%s}" % (expr, exp)
        else:
            return expr

    def _print_Basic(self, expr):
        name = self._deal_with_super_sub(expr.__class__.__name__)
        if expr.args:
            ls = [self._print(o) for o in expr.args]
            s = r"\operatorname{{{}}}\left({}\right)"
            return s.format(name, ", ".join(ls))
        else:
            return r"\text{{{}}}".format(name)

    def _print_bool(self, e: bool | BooleanTrue | BooleanFalse):
        return r"\text{%s}" % e

    _print_BooleanTrue = _print_bool
    _print_BooleanFalse = _print_bool

    def _print_NoneType(self, e):
        return r"\text{%s}" % e

    def _print_Add(self, expr, order=None):
        terms = self._as_ordered_terms(expr, order=order)

        tex = ""
        for i, term in enumerate(terms):
            if i == 0:
                pass
            elif term.could_extract_minus_sign():
                tex += " - "
                term = -term
            else:
                tex += " + "
            term_tex = self._print(term)
            if self._needs_add_brackets(term):
                term_tex = r"\left(%s\right)" % term_tex
            tex += term_tex

        return tex

    def _print_Cycle(self, expr):
        from sympy.combinatorics.permutations import Permutation
        if expr.size == 0:
            return r"\left( \right)"
        expr = Permutation(expr)
        expr_perm = expr.cyclic_form
        siz = expr.size
        if expr.array_form[-1] == siz - 1:
            expr_perm = expr_perm + [[siz - 1]]
        term_tex = ''
        for i in expr_perm:
            term_tex += str(i).replace(',', r"\;")
        term_tex = term_tex.replace('[', r"\left( ")
        term_tex = term_tex.replace(']', r"\right)")
        return term_tex

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation
        from sympy.utilities.exceptions import sympy_deprecation_warning

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(
                f"""
                Setting Permutation.print_cyclic is deprecated. Instead use
                init_printing(perm_cyclic={perm_cyclic}).
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-permutation-print_cyclic",
                stacklevel=8,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            return self._print_Cycle(expr)

        if expr.size == 0:
            return r"\left( \right)"

        lower = [self._print(arg) for arg in expr.array_form]
        upper = [self._print(arg) for arg in range(len(lower))]

        row1 = " & ".join(upper)
        row2 = " & ".join(lower)
        mat = r" \\ ".join((row1, row2))
        return r"\begin{pmatrix} %s \end{pmatrix}" % mat


    def _print_AppliedPermutation(self, expr):
        perm, var = expr.args
        return r"\sigma_{%s}(%s)" % (self._print(perm), self._print(var))

    def _print_Float(self, expr):
        # Based off of that in StrPrinter
        dps = prec_to_dps(expr._prec)
        strip = False if self._settings['full_prec'] else True
        low = self._settings["min"] if "min" in self._settings else None
        high = self._settings["max"] if "max" in self._settings else None
        str_real = mlib_to_str(expr._mpf_, dps, strip_zeros=strip, min_fixed=low, max_fixed=high)

        # Must always have a mul symbol (as 2.5 10^{20} just looks odd)
        # thus we use the number separator
        separator = self._settings['mul_symbol_latex_numbers']

        if 'e' in str_real:
            (mant, exp) = str_real.split('e')

            if exp[0] == '+':
                exp = exp[1:]
            if self._settings['decimal_separator'] == 'comma':
                mant = mant.replace('.','{,}')

            return r"%s%s10^{%s}" % (mant, separator, exp)
        elif str_real == "+inf":
            return r"\infty"
        elif str_real == "-inf":
            return r"- \infty"
        else:
            if self._settings['decimal_separator'] == 'comma':
                str_real = str_real.replace('.','{,}')
            return str_real

    def _print_Cross(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \times %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                  self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Curl(self, expr):
        vec = expr._expr
        return r"\nabla\times %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Divergence(self, expr):
        vec = expr._expr
        return r"\nabla\cdot %s" % self.parenthesize(vec, PRECEDENCE['Mul'])

    def _print_Dot(self, expr):
        vec1 = expr._expr1
        vec2 = expr._expr2
        return r"%s \cdot %s" % (self.parenthesize(vec1, PRECEDENCE['Mul']),
                                 self.parenthesize(vec2, PRECEDENCE['Mul']))

    def _print_Gradient(self, expr):
        func = expr._expr
        return r"\nabla %s" % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_Laplacian(self, expr):
        func = expr._expr
        return r"\Delta %s" % self.parenthesize(func, PRECEDENCE['Mul'])

    def _print_Mul(self, expr: Expr):
        from sympy.simplify import fraction
        separator: str = self._settings['mul_symbol_latex']
        numbersep: str = self._settings['mul_symbol_latex_numbers']

        def convert(expr) -> str:
            if not expr.is_Mul:
                return str(self._print(expr))
            else:
                if self.order not in ('old', 'none'):
                    args = expr.as_ordered_factors()
                else:
                    args = list(expr.args)

                # If there are quantities or prefixes, append them at the back.
                units, nonunits = sift(args, lambda x: (hasattr(x, "_scale_factor") or hasattr(x, "is_physical_constant")) or
                              (isinstance(x, Pow) and
                               hasattr(x.base, "is_physical_constant")), binary=True)
                prefixes, units = sift(units, lambda x: hasattr(x, "_scale_factor"), binary=True)
                return convert_args(nonunits + prefixes + units)

        def convert_args(args) -> str:
            _tex = last_term_tex = ""

            for i, term in enumerate(args):
                term_tex = self._print(term)
                if not (hasattr(term, "_scale_factor") or hasattr(term, "is_physical_constant")):
                    if self._needs_mul_brackets(term, first=(i == 0),
                                                last=(i == len(args) - 1)):
                        term_tex = r"\left(%s\right)" % term_tex

                    if  _between_two_numbers_p[0].search(last_term_tex) and \
                        _between_two_numbers_p[1].match(term_tex):
                        # between two numbers
                        _tex += numbersep
                    elif _tex:
                        _tex += separator
                elif _tex:
                    _tex += separator

                _tex += term_tex
                last_term_tex = term_tex
            return _tex

        # Check for unevaluated Mul. In this case we need to make sure the
        # identities are visible, multiple Rational factors are not combined
        # etc so we display in a straight-forward form that fully preserves all
        # args and their order.
        # XXX: _print_Pow calls this routine with instances of Pow...
        if isinstance(expr, Mul):
            args = expr.args
            if args[0] is S.One or any(isinstance(arg, Number) for arg in args[1:]):
                return convert_args(args)

        include_parens = False
        if expr.could_extract_minus_sign():
            expr = -expr
            tex = "- "
            if expr.is_Add:
                tex += "("
                include_parens = True
        else:
            tex = ""

        numer, denom = fraction(expr, exact=True)

        if denom is S.One and Pow(1, -1, evaluate=False) not in expr.args:
            # use the original expression here, since fraction() may have
            # altered it when producing numer and denom
            tex += convert(expr)

        else:
            snumer = convert(numer)
            sdenom = convert(denom)
            ldenom = len(sdenom.split())
            ratio = self._settings['long_frac_ratio']
            if self._settings['fold_short_frac'] and ldenom <= 2 and \
                    "^" not in sdenom:
                # handle short fractions
                if self._needs_mul_brackets(numer, last=False):
                    tex += r"\left(%s\right) / %s" % (snumer, sdenom)
                else:
                    tex += r"%s / %s" % (snumer, sdenom)
            elif ratio is not None and \
                    len(snumer.split()) > ratio*ldenom:
                # handle long fractions
                if self._needs_mul_brackets(numer, last=True):
                    tex += r"\frac{1}{%s}%s\left(%s\right)" \
                        % (sdenom, separator, snumer)
                elif numer.is_Mul:
                    # split a long numerator
                    a = S.One
                    b = S.One
                    for x in numer.args:
                        if self._needs_mul_brackets(x, last=False) or \
                                len(convert(a*x).split()) > ratio*ldenom or \
                                (b.is_commutative is x.is_commutative is False):
                            b *= x
                        else:
                            a *= x
                    if self._needs_mul_brackets(b, last=True):
                        tex += r"\frac{%s}{%s}%s\left(%s\right)" \
                            % (convert(a), sdenom, separator, convert(b))
                    else:
                        tex += r"\frac{%s}{%s}%s%s" \
                            % (convert(a), sdenom, separator, convert(b))
                else:
                    tex += r"\frac{1}{%s}%s%s" % (sdenom, separator, snumer)
            else:
                tex += r"\frac{%s}{%s}" % (snumer, sdenom)

        if include_parens:
            tex += ")"
        return tex

    def _print_AlgebraicNumber(self, expr):
        if expr.is_aliased:
            return self._print(expr.as_poly().as_expr())
        else:
            return self._print(expr.as_expr())

    def _print_PrimeIdeal(self, expr):
        p = self._print(expr.p)
        if expr.is_inert:
            return rf'\left({p}\right)'
        alpha = self._print(expr.alpha.as_expr())
        return rf'\left({p}, {alpha}\right)'

    def _print_Pow(self, expr: Pow):
        # Treat x**Rational(1,n) as special case
        if expr.exp.is_Rational:
            p: int = expr.exp.p  # type: ignore
            q: int = expr.exp.q  # type: ignore
            if abs(p) == 1 and q != 1 and self._settings['root_notation']:
                base = self._print(expr.base)
                if q == 2:
                    tex = r"\sqrt{%s}" % base
                elif self._settings['itex']:
                    tex = r"\root{%d}{%s}" % (q, base)
                else:
                    tex = r"\sqrt[%d]{%s}" % (q, base)
                if expr.exp.is_negative:
                    return r"\frac{1}{%s}" % tex
                else:
                    return tex
            elif self._settings['fold_frac_powers'] and q != 1:
                base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
                # issue #12886: add parentheses for superscripts raised to powers
                if expr.base.is_Symbol:
                    base = self.parenthesize_super(base)
                if expr.base.is_Function:
                    return self._print(expr.base, exp="%s/%s" % (p, q))
                return r"%s^{%s/%s}" % (base, p, q)
            elif expr.exp.is_negative and expr.base.is_commutative:
                # special case for 1^(-x), issue 9216
                if expr.base == 1:
                    return r"%s^{%s}" % (expr.base, expr.exp)
                # special case for (1/x)^(-y) and (-1/-x)^(-y), issue 20252
                if expr.base.is_Rational:
                    base_p: int = expr.base.p  # type: ignore
                    base_q: int = expr.base.q  # type: ignore
                    if base_p * base_q == abs(base_q):
                        if expr.exp == -1:
                            return r"\frac{1}{\frac{%s}{%s}}" % (base_p, base_q)
                        else:
                            return r"\frac{1}{(\frac{%s}{%s})^{%s}}" % (base_p, base_q, abs(expr.exp))
                # things like 1/x
                return self._print_Mul(expr)
        if expr.base.is_Function:
            return self._print(expr.base, exp=self._print(expr.exp))
        tex = r"%s^{%s}"
        return self._helper_print_standard_power(expr, tex)

    def _helper_print_standard_power(self, expr, template: str) -> str:
        exp = self._print(expr.exp)
        # issue #12886: add parentheses around superscripts raised
        # to powers
        base = self.parenthesize(expr.base, PRECEDENCE['Pow'])
        if expr.base.is_Symbol:
            base = self.parenthesize_super(base)
        elif expr.base.is_Float:
            base = r"{%s}" % base
        elif (isinstance(expr.base, Derivative)
            and base.startswith(r'\left(')
            and re.match(r'\\left\(\\d?d?dot', base)
            and base.endswith(r'\right)')):
            # don't use parentheses around dotted derivative
            base = base[6: -7]  # remove outermost added parens
        return template % (base, exp)

    def _print_UnevaluatedExpr(self, expr):
        return self._print(expr.args[0])

    def _print_Sum(self, expr):
        if len(expr.limits) == 1:
            tex = r"\sum_{%s=%s}^{%s} " % \
                tuple([self._print(i) for i in expr.limits[0]])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\sum_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex

    def _print_Product(self, expr):
        if len(expr.limits) == 1:
            tex = r"\prod_{%s=%s}^{%s} " % \
                tuple([self._print(i) for i in expr.limits[0]])
        else:
            def _format_ineq(l):
                return r"%s \leq %s \leq %s" % \
                    tuple([self._print(s) for s in (l[1], l[0], l[2])])

            tex = r"\prod_{\substack{%s}} " % \
                str.join('\\\\', [_format_ineq(l) for l in expr.limits])

        if isinstance(expr.function, Add):
            tex += r"\left(%s\right)" % self._print(expr.function)
        else:
            tex += self._print(expr.function)

        return tex

    def _print_BasisDependent(self, expr: 'BasisDependent'):
        from sympy.vector import Vector

        o1: list[str] = []
        if expr == expr.zero:
            return expr.zero._latex_form
        if isinstance(expr, Vector):
            items = expr.separate().items()
        else:
            items = [(0, expr)]

        for system, vect in items:
            inneritems = list(vect.components.items())
            inneritems.sort(key=lambda x: x[0].__str__())
            for k, v in inneritems:
                if v == 1:
                    o1.append(' + ' + k._latex_form)
                elif v == -1:
                    o1.append(' - ' + k._latex_form)
                else:
                    arg_str = r'\left(' + self._print(v) + r'\right)'
                    o1.append(' + ' + arg_str + k._latex_form)

        outstr = (''.join(o1))
        if outstr[1] != '-':
            outstr = outstr[3:]
        else:
            outstr = outstr[1:]
        return outstr

    def _print_Indexed(self, expr):
        tex_base = self._print(expr.base)
        tex = '{'+tex_base+'}'+'_{%s}' % ','.join(
            map(self._print, expr.indices))
        return tex

    def _print_IndexedBase(self, expr):
        return self._print(expr.label)

    def _print_Idx(self, expr):
        label = self._print(expr.label)
        if expr.upper is not None:
            upper = self._print(expr.upper)
            if expr.lower is not None:
                lower = self._print(expr.lower)
            else:
                lower = self._print(S.Zero)
            interval = '{lower}\\mathrel{{..}}\\nobreak {upper}'.format(
                    lower = lower, upper = upper)
            return '{{{label}}}_{{{interval}}}'.format(
                label = label, interval = interval)
        #if no bounds are defined this just prints the label
        return label

    def _print_Derivative(self, expr):
        if requires_partial(expr.expr):
            diff_symbol = r'\partial'
        else:
            diff_symbol = self._settings["diff_operator_latex"]

        tex = ""
        dim = 0
        for x, num in reversed(expr.variable_count):
            dim += num
            if num == 1:
                tex += r"%s %s" % (diff_symbol, self._print(x))
            else:
                tex += r"%s %s^{%s}" % (diff_symbol,
                                        self.parenthesize_super(self._print(x)),
                                        self._print(num))

        if dim == 1:
            tex = r"\frac{%s}{%s}" % (diff_symbol, tex)
        else:
            tex = r"\frac{%s^{%s}}{%s}" % (diff_symbol, self._print(dim), tex)

        if any(i.could_extract_minus_sign() for i in expr.args):
            return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  is_neg=True,
                                                  strict=True))

        return r"%s %s" % (tex, self.parenthesize(expr.expr,
                                                  PRECEDENCE["Mul"],
                                                  is_neg=False,
                                                  strict=True))

    def _print_Subs(self, subs):
        expr, old, new = subs.args
        latex_expr = self._print(expr)
        latex_old = (self._print(e) for e in old)
        latex_new = (self._print(e) for e in new)
        latex_subs = r'\\ '.join(
            e[0] + '=' + e[1] for e in zip(latex_old, latex_new))
        return r'\left. %s \right|_{\substack{ %s }}' % (latex_expr,
                                                         latex_subs)

    def _print_Integral(self, expr):
        tex, symbols = "", []
        diff_symbol = self._settings["diff_operator_latex"]

        # Only up to \iiiint exists
        if len(expr.limits) <= 4 and all(len(lim) == 1 for lim in expr.limits):
            # Use len(expr.limits)-1 so that syntax highlighters don't think
            # \" is an escaped quote
            tex = r"\i" + "i"*(len(expr.limits) - 1) + "nt"
            symbols = [r"\, %s%s" % (diff_symbol, self._print(symbol[0]))
                       for symbol in expr.limits]

        else:
            for lim in reversed(expr.limits):
                symbol = lim[0]
                tex += r"\int"

                if len(lim) > 1:
                    if self._settings['mode'] != 'inline' \
                            and not self._settings['itex']:
                        tex += r"\limits"

                    if len(lim) == 3:
                        tex += "_{%s}^{%s}" % (self._print(lim[1]),
                                               self._print(lim[2]))
                    if len(lim) == 2:
                        tex += "^{%s}" % (self._print(lim[1]))

                symbols.insert(0, r"\, %s%s" % (diff_symbol, self._print(symbol)))

        return r"%s %s%s" % (tex, self.parenthesize(expr.function,
                                                    PRECEDENCE["Mul"],
                                                    is_neg=any(i.could_extract_minus_sign() for i in expr.args),
                                                    strict=True),
                             "".join(symbols))

    def _print_Limit(self, expr):
        e, z, z0, dir = expr.args

        tex = r"\lim_{%s \to " % self._print(z)
        if str(dir) == '+-' or z0 in (S.Infinity, S.NegativeInfinity):
            tex += r"%s}" % self._print(z0)
        else:
            tex += r"%s^%s}" % (self._print(z0), self._print(dir))

        if isinstance(e, AssocOp):
            return r"%s\left(%s\right)" % (tex, self._print(e))
        else:
            return r"%s %s" % (tex, self._print(e))

    def _hprint_Function(self, func: str) -> str:
        r'''
        Logic to decide how to render a function to latex
          - if it is a recognized latex name, use the appropriate latex command
          - if it is a single letter, excluding sub- and superscripts, just use that letter
          - if it is a longer name, then put \operatorname{} around it and be
            mindful of undercores in the name
        '''
        func = self._deal_with_super_sub(func)
        superscriptidx = func.find("^")
        subscriptidx = func.find("_")
        if func in accepted_latex_functions:
            name = r"\%s" % func
        elif len(func) == 1 or func.startswith('\\') or subscriptidx == 1 or superscriptidx == 1:
            name = func
        else:
            if superscriptidx > 0 and subscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:min(subscriptidx,superscriptidx)],
                    func[min(subscriptidx,superscriptidx):])
            elif superscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:superscriptidx],
                    func[superscriptidx:])
            elif subscriptidx > 0:
                name = r"\operatorname{%s}%s" %(
                    func[:subscriptidx],
                    func[subscriptidx:])
            else:
                name = r"\operatorname{%s}" % func
        return name

    def _print_Function(self, expr: Function, exp=None) -> str:
        r'''
        Render functions to LaTeX, handling functions that LaTeX knows about
        e.g., sin, cos, ... by using the proper LaTeX command (\sin, \cos, ...).
        For single-letter function names, render them as regular LaTeX math
        symbols. For multi-letter function names that LaTeX does not know
        about, (e.g., Li, sech) use \operatorname{} so that the function name
        is rendered in Roman font and LaTeX handles spacing properly.

        expr is the expression involving the function
        exp is an exponent
        '''
        func = expr.func.__name__
        if hasattr(self, '_print_' + func) and \
                not isinstance(expr, AppliedUndef):
            return getattr(self, '_print_' + func)(expr, exp)
        else:
            args = [str(self._print(arg)) for arg in expr.args]
            # How inverse trig functions should be displayed, formats are:
            # abbreviated: asin, full: arcsin, power: sin^-1
            inv_trig_style = self._settings['inv_trig_style']
            # If we are dealing with a power-style inverse trig function
            inv_trig_power_case = False
            # If it is applicable to fold the argument brackets
            can_fold_brackets = self._settings['fold_func_brackets'] and \
                len(args) == 1 and \
                not self._needs_function_brackets(expr.args[0])

            inv_trig_table = [
                "asin", "acos", "atan",
                "acsc", "asec", "acot",
                "asinh", "acosh", "atanh",
                "acsch", "asech", "acoth",
            ]

            # If the function is an inverse trig function, handle the style
            if func in inv_trig_table:
                if inv_trig_style == "abbreviated":
                    pass
                elif inv_trig_style == "full":
                    func = ("ar" if func[-1] == "h" else "arc") + func[1:]
                elif inv_trig_style == "power":
                    func = func[1:]
                    inv_trig_power_case = True

                    # Can never fold brackets if we're raised to a power
                    if exp is not None:
                        can_fold_brackets = False

            if inv_trig_power_case:
                if func in accepted_latex_functions:
                    name = r"\%s^{-1}" % func
                else:
                    name = r"\operatorname{%s}^{-1}" % func
            elif exp is not None:
                func_tex = self._hprint_Function(func)
                func_tex = self.parenthesize_super(func_tex)
                name = r'%s^{%s}' % (func_tex, exp)
            else:
                name = self._hprint_Function(func)

            if can_fold_brackets:
                if func in accepted_latex_functions:
                    # Wrap argument safely to avoid parse-time conflicts
                    # with the function name itself
                    name += r" {%s}"
                else:
                    name += r"%s"
            else:
                name += r"{\left(%s \right)}"

            if inv_trig_power_case and exp is not None:
                name += r"^{%s}" % exp

            return name % ",".join(args)

    def _print_UndefinedFunction(self, expr):
        return self._hprint_Function(str(expr))

    def _print_ElementwiseApplyFunction(self, expr):
        return r"{%s}_{\circ}\left({%s}\right)" % (
            self._print(expr.function),
            self._print(expr.expr),
        )

    @property
    def _special_function_classes(self):
        from sympy.functions.special.tensor_functions import KroneckerDelta
        from sympy.functions.special.gamma_functions import gamma, lowergamma
        from sympy.functions.special.beta_functions import beta
        from sympy.functions.special.delta_functions import DiracDelta
        from sympy.functions.special.error_functions import Chi
        return {KroneckerDelta: r'\delta',
                gamma:  r'\Gamma',
                lowergamma: r'\gamma',
                beta: r'\operatorname{B}',
                DiracDelta: r'\delta',
                Chi: r'\operatorname{Chi}'}

    def _print_FunctionClass(self, expr):
        for cls in self._special_function_classes:
            if issubclass(expr, cls) and expr.__name__ == cls.__name__:
                return self._special_function_classes[cls]
        return self._hprint_Function(str(expr))

    def _print_Lambda(self, expr):
        symbols, expr = expr.args

        if len(symbols) == 1:
            symbols = self._print(symbols[0])
        else:
            symbols = self._print(tuple(symbols))

        tex = r"\left( %s \mapsto %s \right)" % (symbols, self._print(expr))

        return tex

    def _print_IdentityFunction(self, expr):
        return r"\left( x \mapsto x \right)"

    def _hprint_variadic_function(self, expr, exp=None) -> str:
        args = sorted(expr.args, key=default_sort_key)
        texargs = [r"%s" % self._print(symbol) for symbol in args]
        tex = r"\%s\left(%s\right)" % (str(expr.func).lower(),
                                       ", ".join(texargs))
        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    _print_Min = _print_Max = _hprint_variadic_function

    def _print_floor(self, expr, exp=None):
        tex = r"\left\lfloor{%s}\right\rfloor" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_ceiling(self, expr, exp=None):
        tex = r"\left\lceil{%s}\right\rceil" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_log(self, expr, exp=None):
        if not self._settings["ln_notation"]:
            tex = r"\log{\left(%s \right)}" % self._print(expr.args[0])
        else:
            tex = r"\ln{\left(%s \right)}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_Abs(self, expr, exp=None):
        tex = r"\left|{%s}\right|" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_re(self, expr, exp=None):
        if self._settings['gothic_re_im']:
            tex = r"\Re{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{re}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        return self._do_exponent(tex, exp)

    def _print_im(self, expr, exp=None):
        if self._settings['gothic_re_im']:
            tex = r"\Im{%s}" % self.parenthesize(expr.args[0], PRECEDENCE['Atom'])
        else:
            tex = r"\operatorname{{im}}{{{}}}".format(self.parenthesize(expr.args[0], PRECEDENCE['Atom']))

        return self._do_exponent(tex, exp)

    def _print_Not(self, e):
        from sympy.logic.boolalg import (Equivalent, Implies)
        if isinstance(e.args[0], Equivalent):
            return self._print_Equivalent(e.args[0], r"\not\Leftrightarrow")
        if isinstance(e.args[0], Implies):
            return self._print_Implies(e.args[0], r"\not\Rightarrow")
        if (e.args[0].is_Boolean):
            return r"\neg \left(%s\right)" % self._print(e.args[0])
        else:
            return r"\neg %s" % self._print(e.args[0])

    def _print_LogOp(self, args, char):
        arg = args[0]
        if arg.is_Boolean and not arg.is_Not:
            tex = r"\left(%s\right)" % self._print(arg)
        else:
            tex = r"%s" % self._print(arg)

        for arg in args[1:]:
            if arg.is_Boolean and not arg.is_Not:
                tex += r" %s \left(%s\right)" % (char, self._print(arg))
            else:
                tex += r" %s %s" % (char, self._print(arg))

        return tex

    def _print_And(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\wedge")

    def _print_Or(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\vee")

    def _print_Xor(self, e):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, r"\veebar")

    def _print_Implies(self, e, altchar=None):
        return self._print_LogOp(e.args, altchar or r"\Rightarrow")

    def _print_Equivalent(self, e, altchar=None):
        args = sorted(e.args, key=default_sort_key)
        return self._print_LogOp(args, altchar or r"\Leftrightarrow")

    def _print_conjugate(self, expr, exp=None):
        tex = r"\overline{%s}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_polar_lift(self, expr, exp=None):
        func = r"\operatorname{polar\_lift}"
        arg = r"{\left(%s \right)}" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (func, exp, arg)
        else:
            return r"%s%s" % (func, arg)

    def _print_ExpBase(self, expr, exp=None):
        # TODO should exp_polar be printed differently?
        #      what about exp_polar(0), exp_polar(1)?
        tex = r"e^{%s}" % self._print(expr.args[0])
        return self._do_exponent(tex, exp)

    def _print_Exp1(self, expr, exp=None):
        return "e"

    def _print_elliptic_k(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"K^{%s}%s" % (exp, tex)
        else:
            return r"K%s" % tex

    def _print_elliptic_f(self, expr, exp=None):
        tex = r"\left(%s\middle| %s\right)" % \
            (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"F^{%s}%s" % (exp, tex)
        else:
            return r"F%s" % tex

    def _print_elliptic_e(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"E^{%s}%s" % (exp, tex)
        else:
            return r"E%s" % tex

    def _print_elliptic_pi(self, expr, exp=None):
        if len(expr.args) == 3:
            tex = r"\left(%s; %s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]),
                 self._print(expr.args[2]))
        else:
            tex = r"\left(%s\middle| %s\right)" % \
                (self._print(expr.args[0]), self._print(expr.args[1]))
        if exp is not None:
            return r"\Pi^{%s}%s" % (exp, tex)
        else:
            return r"\Pi%s" % tex

    def _print_beta(self, expr, exp=None):
        x = expr.args[0]
        # Deal with unevaluated single argument beta
        y = expr.args[0] if len(expr.args) == 1 else expr.args[1]
        tex = rf"\left({x}, {y}\right)"

        if exp is not None:
            return r"\operatorname{B}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{B}%s" % tex

    def _print_betainc(self, expr, exp=None, operator='B'):
        largs = [self._print(arg) for arg in expr.args]
        tex = r"\left(%s, %s\right)" % (largs[0], largs[1])

        if exp is not None:
            return r"\operatorname{%s}_{(%s, %s)}^{%s}%s" % (operator, largs[2], largs[3], exp, tex)
        else:
            return r"\operatorname{%s}_{(%s, %s)}%s" % (operator, largs[2], largs[3], tex)

    def _print_betainc_regularized(self, expr, exp=None):
        return self._print_betainc(expr, exp, operator='I')

    def _print_uppergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\Gamma^{%s}%s" % (exp, tex)
        else:
            return r"\Gamma%s" % tex

    def _print_lowergamma(self, expr, exp=None):
        tex = r"\left(%s, %s\right)" % (self._print(expr.args[0]),
                                        self._print(expr.args[1]))

        if exp is not None:
            return r"\gamma^{%s}%s" % (exp, tex)
        else:
            return r"\gamma%s" % tex

    def _hprint_one_arg_func(self, expr, exp=None) -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (self._print(expr.func), exp, tex)
        else:
            return r"%s%s" % (self._print(expr.func), tex)

    _print_gamma = _hprint_one_arg_func

    def _print_Chi(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{Chi}^{%s}%s" % (exp, tex)
        else:
            return r"\operatorname{Chi}%s" % tex

    def _print_expint(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[1])
        nu = self._print(expr.args[0])

        if exp is not None:
            return r"\operatorname{E}_{%s}^{%s}%s" % (nu, exp, tex)
        else:
            return r"\operatorname{E}_{%s}%s" % (nu, tex)

    def _print_fresnels(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"S^{%s}%s" % (exp, tex)
        else:
            return r"S%s" % tex

    def _print_fresnelc(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"C^{%s}%s" % (exp, tex)
        else:
            return r"C%s" % tex

    def _print_subfactorial(self, expr, exp=None):
        tex = r"!%s" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"\left(%s\right)^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial(self, expr, exp=None):
        tex = r"%s!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_factorial2(self, expr, exp=None):
        tex = r"%s!!" % self.parenthesize(expr.args[0], PRECEDENCE["Func"])

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_binomial(self, expr, exp=None):
        tex = r"{\binom{%s}{%s}}" % (self._print(expr.args[0]),
                                     self._print(expr.args[1]))

        if exp is not None:
            return r"%s^{%s}" % (tex, exp)
        else:
            return tex

    def _print_RisingFactorial(self, expr, exp=None):
        n, k = expr.args
        base = r"%s" % self.parenthesize(n, PRECEDENCE['Func'])

        tex = r"{%s}^{\left(%s\right)}" % (base, self._print(k))

        return self._do_exponent(tex, exp)

    def _print_FallingFactorial(self, expr, exp=None):
        n, k = expr.args
        sub = r"%s" % self.parenthesize(k, PRECEDENCE['Func'])

        tex = r"{\left(%s\right)}_{%s}" % (self._print(n), sub)

        return self._do_exponent(tex, exp)

    def _hprint_BesselBase(self, expr, exp, sym: str) -> str:
        tex = r"%s" % (sym)

        need_exp = False
        if exp is not None:
            if tex.find('^') == -1:
                tex = r"%s^{%s}" % (tex, exp)
            else:
                need_exp = True

        tex = r"%s_{%s}\left(%s\right)" % (tex, self._print(expr.order),
                                           self._print(expr.argument))

        if need_exp:
            tex = self._do_exponent(tex, exp)
        return tex

    def _hprint_vec(self, vec) -> str:
        if not vec:
            return ""
        s = ""
        for i in vec[:-1]:
            s += "%s, " % self._print(i)
        s += self._print(vec[-1])
        return s

    def _print_besselj(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'J')

    def _print_besseli(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'I')

    def _print_besselk(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'K')

    def _print_bessely(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'Y')

    def _print_yn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'y')

    def _print_jn(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'j')

    def _print_hankel1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(1)}')

    def _print_hankel2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'H^{(2)}')

    def _print_hn1(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(1)}')

    def _print_hn2(self, expr, exp=None):
        return self._hprint_BesselBase(expr, exp, 'h^{(2)}')

    def _hprint_airy(self, expr, exp=None, notation="") -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"%s^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s%s" % (notation, tex)

    def _hprint_airy_prime(self, expr, exp=None, notation="") -> str:
        tex = r"\left(%s\right)" % self._print(expr.args[0])

        if exp is not None:
            return r"{%s^\prime}^{%s}%s" % (notation, exp, tex)
        else:
            return r"%s^\prime%s" % (notation, tex)

    def _print_airyai(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Ai')

    def _print_airybi(self, expr, exp=None):
        return self._hprint_airy(expr, exp, 'Bi')

    def _print_airyaiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Ai')

    def _print_airybiprime(self, expr, exp=None):
        return self._hprint_airy_prime(expr, exp, 'Bi')

    def _print_hyper(self, expr, exp=None):
        tex = r"{{}_{%s}F_{%s}\left(\begin{matrix} %s \\ %s \end{matrix}" \
              r"\middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._hprint_vec(expr.ap), self._hprint_vec(expr.bq),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, exp)
        return tex

    def _print_meijerg(self, expr, exp=None):
        tex = r"{G_{%s, %s}^{%s, %s}\left(\begin{matrix} %s & %s \\" \
              r"%s & %s \end{matrix} \middle| {%s} \right)}" % \
            (self._print(len(expr.ap)), self._print(len(expr.bq)),
              self._print(len(expr.bm)), self._print(len(expr.an)),
              self._hprint_vec(expr.an), self._hprint_vec(expr.aother),
              self._hprint_vec(expr.bm), self._hprint_vec(expr.bother),
              self._print(expr.argument))

        if exp is not None:
            tex = r"{%s}^{%s}" % (tex, exp)
        return tex

    def _print_dirichlet_eta(self, expr, exp=None):
        tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\eta^{%s}%s" % (exp, tex)
        return r"\eta%s" % tex

    def _print_zeta(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"\left(%s, %s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\zeta^{%s}%s" % (exp, tex)
        return r"\zeta%s" % tex

    def _print_stieltjes(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_{%s}\left(%s\right)" % tuple(map(self._print, expr.args))
        else:
            tex = r"_{%s}" % self._print(expr.args[0])
        if exp is not None:
            return r"\gamma%s^{%s}" % (tex, exp)
        return r"\gamma%s" % tex

    def _print_lerchphi(self, expr, exp=None):
        tex = r"\left(%s, %s, %s\right)" % tuple(map(self._print, expr.args))
        if exp is None:
            return r"\Phi%s" % tex
        return r"\Phi^{%s}%s" % (exp, tex)

    def _print_polylog(self, expr, exp=None):
        s, z = map(self._print, expr.args)
        tex = r"\left(%s\right)" % z
        if exp is None:
            return r"\operatorname{Li}_{%s}%s" % (s, tex)
        return r"\operatorname{Li}_{%s}^{%s}%s" % (s, exp, tex)

    def _print_jacobi(self, expr, exp=None):
        n, a, b, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s,%s\right)}\left(%s\right)" % (n, a, b, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_gegenbauer(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"C_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_chebyshevt(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"T_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_chebyshevu(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"U_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_legendre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"P_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_assoc_legendre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"P_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_hermite(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"H_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_laguerre(self, expr, exp=None):
        n, x = map(self._print, expr.args)
        tex = r"L_{%s}\left(%s\right)" % (n, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_assoc_laguerre(self, expr, exp=None):
        n, a, x = map(self._print, expr.args)
        tex = r"L_{%s}^{\left(%s\right)}\left(%s\right)" % (n, a, x)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_Ynm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Y_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def _print_Znm(self, expr, exp=None):
        n, m, theta, phi = map(self._print, expr.args)
        tex = r"Z_{%s}^{%s}\left(%s,%s\right)" % (n, m, theta, phi)
        if exp is not None:
            tex = r"\left(" + tex + r"\right)^{%s}" % (exp)
        return tex

    def __print_mathieu_functions(self, character, args, prime=False, exp=None):
        a, q, z = map(self._print, args)
        sup = r"^{\prime}" if prime else ""
        exp = "" if not exp else "^{%s}" % exp
        return r"%s%s\left(%s, %s, %s\right)%s" % (character, sup, a, q, z, exp)

    def _print_mathieuc(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, exp=exp)

    def _print_mathieus(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, exp=exp)

    def _print_mathieucprime(self, expr, exp=None):
        return self.__print_mathieu_functions("C", expr.args, prime=True, exp=exp)

    def _print_mathieusprime(self, expr, exp=None):
        return self.__print_mathieu_functions("S", expr.args, prime=True, exp=exp)

    def _print_Rational(self, expr):
        if expr.q != 1:
            sign = ""
            p = expr.p
            if expr.p < 0:
                sign = "- "
                p = -p
            if self._settings['fold_short_frac']:
                return r"%s%d / %d" % (sign, p, expr.q)
            return r"%s\frac{%d}{%d}" % (sign, p, expr.q)
        else:
            return self._print(expr.p)

    def _print_Order(self, expr):
        s = self._print(expr.expr)
        if expr.point and any(p != S.Zero for p in expr.point) or \
           len(expr.variables) > 1:
            s += '; '
            if len(expr.variables) > 1:
                s += self._print(expr.variables)
            elif expr.variables:
                s += self._print(expr.variables[0])
            s += r'\rightarrow '
            if len(expr.point) > 1:
                s += self._print(expr.point)
            else:
                s += self._print(expr.point[0])
        return r"O\left(%s\right)" % s

    def _print_Symbol(self, expr: Symbol, style='plain'):
        name: str = self._settings['symbol_names'].get(expr)
        if name is not None:
            return name

        return self._deal_with_super_sub(expr.name, style=style)

    _print_RandomSymbol = _print_Symbol

    def _split_super_sub(self, name: str) -> tuple[str, list[str], list[str]]:
        if name is None or '{' in name:
            return (name, [], [])
        elif self._settings["disable_split_super_sub"]:
            name, supers, subs = (name.replace('_', '\\_').replace('^', '\\^'), [], [])
        else:
            name, supers, subs = split_super_sub(name)
        name = translate(name)
        supers = [translate(sup) for sup in supers]
        subs = [translate(sub) for sub in subs]
        return (name, supers, subs)

    def _deal_with_super_sub(self, string: str, style='plain') -> str:
        name, supers, subs = self._split_super_sub(string)

        # apply the style only to the name
        if style == 'bold':
            name = "\\mathbf{{{}}}".format(name)

        # glue all items together:
        if supers:
            name += "^{%s}" % " ".join(supers)
        if subs:
            name += "_{%s}" % " ".join(subs)

        return name

    def _print_Relational(self, expr):
        if self._settings['itex']:
            gt = r"\gt"
            lt = r"\lt"
        else:
            gt = ">"
            lt = "<"

        charmap = {
            "==": "=",
            ">": gt,
            "<": lt,
            ">=": r"\geq",
            "<=": r"\leq",
            "!=": r"\neq",
        }

        return "%s %s %s" % (self._print(expr.lhs),
                             charmap[expr.rel_op], self._print(expr.rhs))

    def _print_Piecewise(self, expr):
        ecpairs = [r"%s & \text{for}\: %s" % (self._print(e), self._print(c))
                   for e, c in expr.args[:-1]]
        if expr.args[-1].cond == true:
            ecpairs.append(r"%s & \text{otherwise}" %
                           self._print(expr.args[-1].expr))
        else:
            ecpairs.append(r"%s & \text{for}\: %s" %
                           (self._print(expr.args[-1].expr),
                            self._print(expr.args[-1].cond)))
        tex = r"\begin{cases} %s \end{cases}"
        return tex % r" \\".join(ecpairs)

    def _print_matrix_contents(self, expr):
        lines = []

        for line in range(expr.rows):  # horrible, should be 'rows'
            lines.append(" & ".join([self._print(i) for i in expr[line, :]]))

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.cols <= 10) is True:
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'

        out_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        out_str = out_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            out_str = out_str.replace('%s', '{' + 'c'*expr.cols + '}%s')
        return out_str % r"\\".join(lines)

    def _print_MatrixBase(self, expr):
        out_str = self._print_matrix_contents(expr)
        if self._settings['mat_delim']:
            left_delim = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            out_str = r'\left' + left_delim + out_str + \
                      r'\right' + right_delim
        return out_str

    def _print_MatrixElement(self, expr):
        matrix_part = self.parenthesize(expr.parent, PRECEDENCE['Atom'], strict=True)
        index_part = f"{self._print(expr.i)},{self._print(expr.j)}"
        return f"{{{matrix_part}}}_{{{index_part}}}"

    def _print_MatrixSlice(self, expr):
        def latexslice(x, dim):
            x = list(x)
            if x[2] == 1:
                del x[2]
            if x[0] == 0:
                x[0] = None
            if x[1] == dim:
                x[1] = None
            return ':'.join(self._print(xi) if xi is not None else '' for xi in x)
        return (self.parenthesize(expr.parent, PRECEDENCE["Atom"], strict=True) + r'\left[' +
                latexslice(expr.rowslice, expr.parent.rows) + ', ' +
                latexslice(expr.colslice, expr.parent.cols) + r'\right]')

    def _print_BlockMatrix(self, expr):
        return self._print(expr.blocks)

    def _print_Transpose(self, expr):
        mat = expr.arg
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            return r"\left(%s\right)^{T}" % self._print(mat)
        else:
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return r"\left(%s\right)^{T}" % s
            else:
                return "%s^{T}" % s

    def _print_Trace(self, expr):
        mat = expr.arg
        return r"\operatorname{tr}\left(%s \right)" % self._print(mat)

    def _print_Adjoint(self, expr):
        style_to_latex = {
            "dagger"   : r"\dagger",
            "star"     : r"\ast",
            "hermitian": r"\mathsf{H}"
        }
        adjoint_style = style_to_latex.get(self._settings["adjoint_style"], r"\dagger")
        mat = expr.arg
        from sympy.matrices import MatrixSymbol, BlockMatrix
        if (not isinstance(mat, MatrixSymbol) and
            not isinstance(mat, BlockMatrix) and mat.is_MatrixExpr):
            return r"\left(%s\right)^{%s}" % (self._print(mat), adjoint_style)
        else:
            s = self.parenthesize(mat, precedence_traditional(expr), True)
            if '^' in s:
                return r"\left(%s\right)^{%s}" % (s, adjoint_style)
            else:
                return r"%s^{%s}" % (s, adjoint_style)

    def _print_MatMul(self, expr):
        from sympy import MatMul

        # Parenthesize nested MatMul but not other types of Mul objects:
        parens = lambda x: self._print(x) if isinstance(x, Mul) and not isinstance(x, MatMul) else \
            self.parenthesize(x, precedence_traditional(expr), False)

        args = list(expr.args)
        if expr.could_extract_minus_sign():
            if args[0] == -1:
                args = args[1:]
            else:
                args[0] = -args[0]
            return '- ' + ' '.join(map(parens, args))
        else:
            return ' '.join(map(parens, args))

    def _print_DotProduct(self, expr):
        level = precedence_traditional(expr)
        left, right = expr.args
        return rf"{self.parenthesize(left, level)} \cdot {self.parenthesize(right, level)}"

    def _print_Determinant(self, expr):
        mat = expr.arg
        if mat.is_MatrixExpr:
            from sympy.matrices.expressions.blockmatrix import BlockMatrix
            if isinstance(mat, BlockMatrix):
                return r"\left|{%s}\right|" % self._print_matrix_contents(mat.blocks)
            return r"\left|{%s}\right|" % self._print(mat)
        return r"\left|{%s}\right|" % self._print_matrix_contents(mat)


    def _print_Mod(self, expr, exp=None):
        if exp is not None:
            return r'\left(%s \bmod %s\right)^{%s}' % \
                (self.parenthesize(expr.args[0], PRECEDENCE['Mul'],
                                   strict=True),
                 self.parenthesize(expr.args[1], PRECEDENCE['Mul'],
                                   strict=True),
                 exp)
        return r'%s \bmod %s' % (self.parenthesize(expr.args[0],
                                                   PRECEDENCE['Mul'],
                                                   strict=True),
                                 self.parenthesize(expr.args[1],
                                                   PRECEDENCE['Mul'],
                                                   strict=True))

    def _print_HadamardProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize

        return r' \circ '.join(
            (parens(arg, prec, strict=True) for arg in args))

    def _print_HadamardPower(self, expr):
        if precedence_traditional(expr.exp) < PRECEDENCE["Mul"]:
            template = r"%s^{\circ \left({%s}\right)}"
        else:
            template = r"%s^{\circ {%s}}"
        return self._helper_print_standard_power(expr, template)

    def _print_KroneckerProduct(self, expr):
        args = expr.args
        prec = PRECEDENCE['Pow']
        parens = self.parenthesize

        return r' \otimes '.join(
            (parens(arg, prec, strict=True) for arg in args))

    def _print_MatPow(self, expr):
        base, exp = expr.base, expr.exp
        from sympy.matrices import MatrixSymbol
        if not isinstance(base, MatrixSymbol) and base.is_MatrixExpr:
            return "\\left(%s\\right)^{%s}" % (self._print(base),
                                              self._print(exp))
        else:
            base_str = self._print(base)
            if '^' in base_str:
                return r"\left(%s\right)^{%s}" % (base_str, self._print(exp))
            else:
                return "%s^{%s}" % (base_str, self._print(exp))

    def _print_MatrixSymbol(self, expr):
        return self._print_Symbol(expr, style=self._settings[
            'mat_symbol_style'])

    def _print_ZeroMatrix(self, Z):
        return "0" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{0}"

    def _print_OneMatrix(self, O):
        return "1" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{1}"

    def _print_Identity(self, I):
        return r"\mathbb{I}" if self._settings[
            'mat_symbol_style'] == 'plain' else r"\mathbf{I}"

    def _print_PermutationMatrix(self, P):
        perm_str = self._print(P.args[0])
        return "P_{%s}" % perm_str

    def _print_NDimArray(self, expr: NDimArray):

        if expr.rank() == 0:
            return self._print(expr[()])

        mat_str = self._settings['mat_str']
        if mat_str is None:
            if self._settings['mode'] == 'inline':
                mat_str = 'smallmatrix'
            else:
                if (expr.rank() == 0) or (expr.shape[-1] <= 10):
                    mat_str = 'matrix'
                else:
                    mat_str = 'array'
        block_str = r'\begin{%MATSTR%}%s\end{%MATSTR%}'
        block_str = block_str.replace('%MATSTR%', mat_str)
        if mat_str == 'array':
            block_str = block_str.replace('%s', '{' + 'c'*expr.shape[0] + '}%s')

        if self._settings['mat_delim']:
            left_delim: str = self._settings['mat_delim']
            right_delim = self._delim_dict[left_delim]
            block_str = r'\left' + left_delim + block_str + \
                        r'\right' + right_delim

        if expr.rank() == 0:
            return block_str % ""

        level_str: list[list[str]] = [[] for i in range(expr.rank() + 1)]
        shape_ranges = [list(range(i)) for i in expr.shape]
        for outer_i in itertools.product(*shape_ranges):
            level_str[-1].append(self._print(expr[outer_i]))
            even = True
            for back_outer_i in range(expr.rank()-1, -1, -1):
                if len(level_str[back_outer_i+1]) < expr.shape[back_outer_i]:
                    break
                if even:
                    level_str[back_outer_i].append(
                        r" & ".join(level_str[back_outer_i+1]))
                else:
                    level_str[back_outer_i].append(
                        block_str % (r"\\".join(level_str[back_outer_i+1])))
                    if len(level_str[back_outer_i+1]) == 1:
                        level_str[back_outer_i][-1] = r"\left[" + \
                            level_str[back_outer_i][-1] + r"\right]"
                even = not even
                level_str[back_outer_i+1] = []

        out_str = level_str[0][0]

        if expr.rank() % 2 == 1:
            out_str = block_str % out_str

        return out_str

    def _printer_tensor_indices(self, name, indices, index_map: dict):
        out_str = self._print(name)
        last_valence = None
        prev_map = None
        for index in indices:
            new_valence = index.is_up
            if ((index in index_map) or prev_map) and \
                    last_valence == new_valence:
                out_str += ","
            if last_valence != new_valence:
                if last_valence is not None:
                    out_str += "}"
                if index.is_up:
                    out_str += "{}^{"
                else:
                    out_str += "{}_{"
            out_str += self._print(index.args[0])
            if index in index_map:
                out_str += "="
                out_str += self._print(index_map[index])
                prev_map = True
            else:
                prev_map = False
            last_valence = new_valence
        if last_valence is not None:
            out_str += "}"
        return out_str

    def _print_Tensor(self, expr):
        name = expr.args[0].args[0]
        indices = expr.get_indices()
        return self._printer_tensor_indices(name, indices, {})

    def _print_TensorElement(self, expr):
        name = expr.expr.args[0].args[0]
        indices = expr.expr.get_indices()
        index_map = expr.index_map
        return self._printer_tensor_indices(name, indices, index_map)

    def _print_TensMul(self, expr):
        # prints expressions like "A(a)", "3*A(a)", "(1+x)*A(a)"
        sign, args = expr._get_args_for_traditional_printer()
        return sign + "".join(
            [self.parenthesize(arg, precedence(expr)) for arg in args]
        )

    def _print_TensAdd(self, expr):
        a = []
        args = expr.args
        for x in args:
            a.append(self.parenthesize(x, precedence(expr)))
        a.sort()
        s = ' + '.join(a)
        s = s.replace('+ -', '- ')
        return s

    def _print_TensorIndex(self, expr):
        return "{}%s{%s}" % (
            "^" if expr.is_up else "_",
            self._print(expr.args[0])
        )

    def _print_PartialDerivative(self, expr):
        if len(expr.variables) == 1:
            return r"\frac{\partial}{\partial {%s}}{%s}" % (
                self._print(expr.variables[0]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )
        else:
            return r"\frac{\partial^{%s}}{%s}{%s}" % (
                len(expr.variables),
                " ".join([r"\partial {%s}" % self._print(i) for i in expr.variables]),
                self.parenthesize(expr.expr, PRECEDENCE["Mul"], False)
            )

    def _print_ArraySymbol(self, expr):
        return self._print(expr.name)

    def _print_ArrayElement(self, expr):
        return "{{%s}_{%s}}" % (
            self.parenthesize(expr.name, PRECEDENCE["Func"], True),
            ", ".join([f"{self._print(i)}" for i in expr.indices]))

    def _print_UniversalSet(self, expr):
        return r"\mathbb{U}"

    def _print_frac(self, expr, exp=None):
        if exp is None:
            return r"\operatorname{frac}{\left(%s\right)}" % self._print(expr.args[0])
        else:
            return r"\operatorname{frac}{\left(%s\right)}^{%s}" % (
                    self._print(expr.args[0]), exp)

    def _print_tuple(self, expr):
        if self._settings['decimal_separator'] == 'comma':
            sep = ";"
        elif self._settings['decimal_separator'] == 'period':
            sep = ","
        else:
            raise ValueError('Unknown Decimal Separator')

        if len(expr) == 1:
            # 1-tuple needs a trailing separator
            return self._add_parens_lspace(self._print(expr[0]) + sep)
        else:
            return self._add_parens_lspace(
                (sep + r" \  ").join([self._print(i) for i in expr]))

    def _print_TensorProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \otimes '.join(elements)

    def _print_WedgeProduct(self, expr):
        elements = [self._print(a) for a in expr.args]
        return r' \wedge '.join(elements)

    def _print_Tuple(self, expr):
        return self._print_tuple(expr)

    def _print_list(self, expr):
        if self._settings['decimal_separator'] == 'comma':
            return r"\left[ %s\right]" % \
                r"; \  ".join([self._print(i) for i in expr])
        elif self._settings['decimal_separator'] == 'period':
            return r"\left[ %s\right]" % \
                r", \  ".join([self._print(i) for i in expr])
        else:
            raise ValueError('Unknown Decimal Separator')


    def _print_dict(self, d):
        keys = sorted(d.keys(), key=default_sort_key)
        items = []

        for key in keys:
            val = d[key]
            items.append("%s : %s" % (self._print(key), self._print(val)))

        return r"\left\{ %s\right\}" % r", \  ".join(items)

    def _print_Dict(self, expr):
        return self._print_dict(expr)

    def _print_DiracDelta(self, expr, exp=None):
        if len(expr.args) == 1 or expr.args[1] == 0:
            tex = r"\delta\left(%s\right)" % self._print(expr.args[0])
        else:
            tex = r"\delta^{\left( %s \right)}\left( %s \right)" % (
                self._print(expr.args[1]), self._print(expr.args[0]))
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex

    def _print_SingularityFunction(self, expr, exp=None):
        shift = self._print(expr.args[0] - expr.args[1])
        power = self._print(expr.args[2])
        tex = r"{\left\langle %s \right\rangle}^{%s}" % (shift, power)
        if exp is not None:
            tex = r"{\left({\langle %s \rangle}^{%s}\right)}^{%s}" % (shift, power, exp)
        return tex

    def _print_Heaviside(self, expr, exp=None):
        pargs = ', '.join(self._print(arg) for arg in expr.pargs)
        tex = r"\theta\left(%s\right)" % pargs
        if exp:
            tex = r"\left(%s\right)^{%s}" % (tex, exp)
        return tex

    def _print_KroneckerDelta(self, expr, exp=None):
        i = self._print(expr.args[0])
        j = self._print(expr.args[1])
        if expr.args[0].is_Atom and expr.args[1].is_Atom:
            tex = r'\delta_{%s %s}' % (i, j)
        else:
            tex = r'\delta_{%s, %s}' % (i, j)
        if exp is not None:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_LeviCivita(self, expr, exp=None):
        indices = map(self._print, expr.args)
        if all(x.is_Atom for x in expr.args):
            tex = r'\varepsilon_{%s}' % " ".join(indices)
        else:
            tex = r'\varepsilon_{%s}' % ", ".join(indices)
        if exp:
            tex = r'\left(%s\right)^{%s}' % (tex, exp)
        return tex

    def _print_RandomDomain(self, d):
        if hasattr(d, 'as_boolean'):
            return '\\text{Domain: }' + self._print(d.as_boolean())
        elif hasattr(d, 'set'):
            return ('\\text{Domain: }' + self._print(d.symbols) + ' \\in ' +
                    self._print(d.set))
        elif hasattr(d, 'symbols'):
            return '\\text{Domain on }' + self._print(d.symbols)
        else:
            return self._print(None)

    def _print_FiniteSet(self, s):
        items = sorted(s.args, key=default_sort_key)
        return self._print_set(items)

    def _print_set(self, s):
        items = sorted(s, key=default_sort_key)
        if self._settings['decimal_separator'] == 'comma':
            items = "; ".join(map(self._print, items))
        elif self._settings['decimal_separator'] == 'period':
            items = ", ".join(map(self._print, items))
        else:
            raise ValueError('Unknown Decimal Separator')
        return r"\left\{%s\right\}" % items


    _print_frozenset = _print_set

    def _print_Range(self, s):
        def _print_symbolic_range():
            # Symbolic Range that cannot be resolved
            if s.args[0] == 0:
                if s.args[2] == 1:
                    cont = self._print(s.args[1])
                else:
                    cont = ", ".join(self._print(arg) for arg in s.args)
            else:
                if s.args[2] == 1:
                    cont = ", ".join(self._print(arg) for arg in s.args[:2])
                else:
                    cont = ", ".join(self._print(arg) for arg in s.args)

            return(f"\\text{{Range}}\\left({cont}\\right)")

        dots = object()

        if s.start.is_infinite and s.stop.is_infinite:
            if s.step.is_positive:
                printset = dots, -1, 0, 1, dots
            else:
                printset = dots, 1, 0, -1, dots
        elif s.start.is_infinite:
            printset = dots, s[-1] - s.step, s[-1]
        elif s.stop.is_infinite:
            it = iter(s)
            printset = next(it), next(it), dots
        elif s.is_empty is not None:
            if (s.size < 4) == True:
                printset = tuple(s)
            elif s.is_iterable:
                it = iter(s)
                printset = next(it), next(it), dots, s[-1]
            else:
                return _print_symbolic_range()
        else:
            return _print_symbolic_range()
        return (r"\left\{" +
                r", ".join(self._print(el) if el is not dots else r'\ldots' for el in printset) +
                r"\right\}")

    def __print_number_polynomial(self, expr, letter, exp=None):
        if len(expr.args) == 2:
            if exp is not None:
                return r"%s_{%s}^{%s}\left(%s\right)" % (letter,
                            self._print(expr.args[0]), exp,
                            self._print(expr.args[1]))
            return r"%s_{%s}\left(%s\right)" % (letter,
                        self._print(expr.args[0]), self._print(expr.args[1]))

        tex = r"%s_{%s}" % (letter, self._print(expr.args[0]))
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_bernoulli(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "B", exp)

    def _print_genocchi(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "G", exp)

    def _print_bell(self, expr, exp=None):
        if len(expr.args) == 3:
            tex1 = r"B_{%s, %s}" % (self._print(expr.args[0]),
                                self._print(expr.args[1]))
            tex2 = r"\left(%s\right)" % r", ".join(self._print(el) for
                                               el in expr.args[2])
            if exp is not None:
                tex = r"%s^{%s}%s" % (tex1, exp, tex2)
            else:
                tex = tex1 + tex2
            return tex
        return self.__print_number_polynomial(expr, "B", exp)

    def _print_fibonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "F", exp)

    def _print_lucas(self, expr, exp=None):
        tex = r"L_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_tribonacci(self, expr, exp=None):
        return self.__print_number_polynomial(expr, "T", exp)

    def _print_mobius(self, expr, exp=None):
        if exp is None:
            return r'\mu\left(%s\right)' % self._print(expr.args[0])
        return r'\mu^{%s}\left(%s\right)' % (exp, self._print(expr.args[0]))

    def _print_SeqFormula(self, s):
        dots = object()
        if len(s.start.free_symbols) > 0 or len(s.stop.free_symbols) > 0:
            return r"\left\{%s\right\}_{%s=%s}^{%s}" % (
                self._print(s.formula),
                self._print(s.variables[0]),
                self._print(s.start),
                self._print(s.stop)
            )
        if s.start is S.NegativeInfinity:
            stop = s.stop
            printset = (dots, s.coeff(stop - 3), s.coeff(stop - 2),
                        s.coeff(stop - 1), s.coeff(stop))
        elif s.stop is S.Infinity or s.length > 4:
            printset = s[:4]
            printset.append(dots)
        else:
            printset = tuple(s)

        return (r"\left[" +
                r", ".join(self._print(el) if el is not dots else r'\ldots' for el in printset) +
                r"\right]")

    _print_SeqPer = _print_SeqFormula
    _print_SeqAdd = _print_SeqFormula
    _print_SeqMul = _print_SeqFormula

    def _print_Interval(self, i):
        if i.start == i.end:
            return r"\left\{%s\right\}" % self._print(i.start)

        else:
            if i.left_open:
                left = '('
            else:
                left = '['

            if i.right_open:
                right = ')'
            else:
                right = ']'

            return r"\left%s%s, %s\right%s" % \
                   (left, self._print(i.start), self._print(i.end), right)

    def _print_AccumulationBounds(self, i):
        return r"\left\langle %s, %s\right\rangle" % \
                (self._print(i.min), self._print(i.max))

    def _print_Union(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cup ".join(args_str)

    def _print_Complement(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \setminus ".join(args_str)

    def _print_Intersection(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \cap ".join(args_str)

    def _print_SymmetricDifference(self, u):
        prec = precedence_traditional(u)
        args_str = [self.parenthesize(i, prec) for i in u.args]
        return r" \triangle ".join(args_str)

    def _print_ProductSet(self, p):
        prec = precedence_traditional(p)
        if len(p.sets) >= 1 and not has_variety(p.sets):
            return self.parenthesize(p.sets[0], prec) + "^{%d}" % len(p.sets)
        return r" \times ".join(
            self.parenthesize(set, prec) for set in p.sets)

    def _print_EmptySet(self, e):
        return r"\emptyset"

    def _print_Naturals(self, n):
        return r"\mathbb{N}"

    def _print_Naturals0(self, n):
        return r"\mathbb{N}_0"

    def _print_Integers(self, i):
        return r"\mathbb{Z}"

    def _print_Rationals(self, i):
        return r"\mathbb{Q}"

    def _print_Reals(self, i):
        return r"\mathbb{R}"

    def _print_Complexes(self, i):
        return r"\mathbb{C}"

    def _print_ImageSet(self, s):
        expr = s.lamda.expr
        sig = s.lamda.signature
        xys = ((self._print(x), self._print(y)) for x, y in zip(sig, s.base_sets))
        xinys = r", ".join(r"%s \in %s" % xy for xy in xys)
        return r"\left\{%s\; \middle|\; %s\right\}" % (self._print(expr), xinys)

    def _print_ConditionSet(self, s):
        vars_print = ', '.join([self._print(var) for var in Tuple(s.sym)])
        if s.base_set is S.UniversalSet:
            return r"\left\{%s\; \middle|\; %s \right\}" % \
                (vars_print, self._print(s.condition))

        return r"\left\{%s\; \middle|\; %s \in %s \wedge %s \right\}" % (
            vars_print,
            vars_print,
            self._print(s.base_set),
            self._print(s.condition))

    def _print_PowerSet(self, expr):
        arg_print = self._print(expr.args[0])
        return r"\mathcal{{P}}\left({}\right)".format(arg_print)

    def _print_ComplexRegion(self, s):
        vars_print = ', '.join([self._print(var) for var in s.variables])
        return r"\left\{%s\; \middle|\; %s \in %s \right\}" % (
            self._print(s.expr),
            vars_print,
            self._print(s.sets))

    def _print_Contains(self, e):
        return r"%s \in %s" % tuple(self._print(a) for a in e.args)

    def _print_FourierSeries(self, s):
        if s.an.formula is S.Zero and s.bn.formula is S.Zero:
            return self._print(s.a0)
        return self._print_Add(s.truncate()) + r' + \ldots'

    def _print_FormalPowerSeries(self, s):
        return self._print_Add(s.infinite)

    def _print_FiniteField(self, expr):
        return r"\mathbb{F}_{%s}" % expr.mod

    def _print_IntegerRing(self, expr):
        return r"\mathbb{Z}"

    def _print_RationalField(self, expr):
        return r"\mathbb{Q}"

    def _print_RealField(self, expr):
        return r"\mathbb{R}"

    def _print_ComplexField(self, expr):
        return r"\mathbb{C}"

    def _print_PolynomialRing(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left[%s\right]" % (domain, symbols)

    def _print_FractionField(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        return r"%s\left(%s\right)" % (domain, symbols)

    def _print_PolynomialRingBase(self, expr):
        domain = self._print(expr.domain)
        symbols = ", ".join(map(self._print, expr.symbols))
        inv = ""
        if not expr.is_Poly:
            inv = r"S_<^{-1}"
        return r"%s%s\left[%s\right]" % (inv, domain, symbols)

    def _print_Poly(self, poly):
        cls = poly.__class__.__name__
        terms = []
        for monom, coeff in poly.terms():
            s_monom = ''
            for i, exp in enumerate(monom):
                if exp > 0:
                    if exp == 1:
                        s_monom += self._print(poly.gens[i])
                    else:
                        s_monom += self._print(pow(poly.gens[i], exp))

            if coeff.is_Add:
                if s_monom:
                    s_coeff = r"\left(%s\right)" % self._print(coeff)
                else:
                    s_coeff = self._print(coeff)
            else:
                if s_monom:
                    if coeff is S.One:
                        terms.extend(['+', s_monom])
                        continue

                    if coeff is S.NegativeOne:
                        terms.extend(['-', s_monom])
                        continue

                s_coeff = self._print(coeff)

            if not s_monom:
                s_term = s_coeff
            else:
                s_term = s_coeff + " " + s_monom

            if s_term.startswith('-'):
                terms.extend(['-', s_term[1:]])
            else:
                terms.extend(['+', s_term])

        if terms[0] in ('-', '+'):
            modifier = terms.pop(0)

            if modifier == '-':
                terms[0] = '-' + terms[0]

        expr = ' '.join(terms)
        gens = list(map(self._print, poly.gens))
        domain = "domain=%s" % self._print(poly.get_domain())

        args = ", ".join([expr] + gens + [domain])
        if cls in accepted_latex_functions:
            tex = r"\%s {\left(%s \right)}" % (cls, args)
        else:
            tex = r"\operatorname{%s}{\left( %s \right)}" % (cls, args)

        return tex

    def _print_ComplexRootOf(self, root):
        cls = root.__class__.__name__
        if cls == "ComplexRootOf":
            cls = "CRootOf"
        expr = self._print(root.expr)
        index = root.index
        if cls in accepted_latex_functions:
            return r"\%s {\left(%s, %d\right)}" % (cls, expr, index)
        else:
            return r"\operatorname{%s} {\left(%s, %d\right)}" % (cls, expr,
                                                                 index)

    def _print_RootSum(self, expr):
        cls = expr.__class__.__name__
        args = [self._print(expr.expr)]

        if expr.fun is not S.IdentityFunction:
            args.append(self._print(expr.fun))

        if cls in accepted_latex_functions:
            return r"\%s {\left(%s\right)}" % (cls, ", ".join(args))
        else:
            return r"\operatorname{%s} {\left(%s\right)}" % (cls,
                                                             ", ".join(args))

    def _print_OrdinalOmega(self, expr):
        return r"\omega"

    def _print_OmegaPower(self, expr):
        exp, mul = expr.args
        if mul != 1:
            if exp != 1:
                return r"{} \omega^{{{}}}".format(mul, exp)
            else:
                return r"{} \omega".format(mul)
        else:
            if exp != 1:
                return r"\omega^{{{}}}".format(exp)
            else:
                return r"\omega"

    def _print_Ordinal(self, expr):
        return " + ".join([self._print(arg) for arg in expr.args])

    def _print_PolyElement(self, poly):
        mul_symbol = self._settings['mul_symbol_latex']
        return poly.str(self, PRECEDENCE, "{%s}^{%d}", mul_symbol)

    def _print_FracElement(self, frac):
        if frac.denom == 1:
            return self._print(frac.numer)
        else:
            numer = self._print(frac.numer)
            denom = self._print(frac.denom)
            return r"\frac{%s}{%s}" % (numer, denom)

    def _print_euler(self, expr, exp=None):
        m, x = (expr.args[0], None) if len(expr.args) == 1 else expr.args
        tex = r"E_{%s}" % self._print(m)
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        if x is not None:
            tex = r"%s\left(%s\right)" % (tex, self._print(x))
        return tex

    def _print_catalan(self, expr, exp=None):
        tex = r"C_{%s}" % self._print(expr.args[0])
        if exp is not None:
            tex = r"%s^{%s}" % (tex, exp)
        return tex

    def _print_UnifiedTransform(self, expr, s, inverse=False):
        return r"\mathcal{{{}}}{}_{{{}}}\left[{}\right]\left({}\right)".format(s, '^{-1}' if inverse else '', self._print(expr.args[1]), self._print(expr.args[0]), self._print(expr.args[2]))

    def _print_MellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M')

    def _print_InverseMellinTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'M', True)

    def _print_LaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L')

    def _print_InverseLaplaceTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'L', True)

    def _print_FourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F')

    def _print_InverseFourierTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'F', True)

    def _print_SineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN')

    def _print_InverseSineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'SIN', True)

    def _print_CosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS')

    def _print_InverseCosineTransform(self, expr):
        return self._print_UnifiedTransform(expr, 'COS', True)

    def _print_DMP(self, p):
        try:
            if p.ring is not None:
                # TODO incorporate order
                return self._print(p.ring.to_sympy(p))
        except SympifyError:
            pass
        return self._print(repr(p))

    def _print_DMF(self, p):
        return self._print_DMP(p)

    def _print_Object(self, object):
        return self._print(Symbol(object.name))

    def _print_LambertW(self, expr, exp=None):
        arg0 = self._print(expr.args[0])
        exp = r"^{%s}" % (exp,) if exp is not None else ""
        if len(expr.args) == 1:
            result = r"W%s\left(%s\right)" % (exp, arg0)
        else:
            arg1 = self._print(expr.args[1])
            result = "W{0}_{{{1}}}\\left({2}\\right)".format(exp, arg1, arg0)
        return result

    def _print_Expectation(self, expr):
        return r"\operatorname{{E}}\left[{}\right]".format(self._print(expr.args[0]))

    def _print_Variance(self, expr):
        return r"\operatorname{{Var}}\left({}\right)".format(self._print(expr.args[0]))

    def _print_Covariance(self, expr):
        return r"\operatorname{{Cov}}\left({}\right)".format(", ".join(self._print(arg) for arg in expr.args))

    def _print_Probability(self, expr):
        return r"\operatorname{{P}}\left({}\right)".format(self._print(expr.args[0]))

    def _print_Morphism(self, morphism):
        domain = self._print(morphism.domain)
        codomain = self._print(morphism.codomain)
        return "%s\\rightarrow %s" % (domain, codomain)

    def _print_TransferFunction(self, expr):
        num, den = self._print(expr.num), self._print(expr.den)
        return r"\frac{%s}{%s}" % (num, den)

    def _print_Series(self, expr):
        args = list(expr.args)
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
                                            False)
        return ' '.join(map(parens, args))

    def _print_MIMOSeries(self, expr):
        from sympy.physics.control.lti import MIMOParallel
        args = list(expr.args)[::-1]
        parens = lambda x: self.parenthesize(x, precedence_traditional(expr),
                                             False) if isinstance(x, MIMOParallel) else self._print(x)
        return r"\cdot".join(map(parens, args))

    def _print_Parallel(self, expr):
        return ' + '.join(map(self._print, expr.args))

    def _print_MIMOParallel(self, expr):
        return ' + '.join(map(self._print, expr.args))

    def _print_Feedback(self, expr):
        from sympy.physics.control import TransferFunction, Series

        num, tf = expr.sys1, TransferFunction(1, 1, expr.var)
        num_arg_list = list(num.args) if isinstance(num, Series) else [num]
        den_arg_list = list(expr.sys2.args) if \
            isinstance(expr.sys2, Series) else [expr.sys2]
        den_term_1 = tf

        if isinstance(num, Series) and isinstance(expr.sys2, Series):
            den_term_2 = Series(*num_arg_list, *den_arg_list)
        elif isinstance(num, Series) and isinstance(expr.sys2, TransferFunction):
            if expr.sys2 == tf:
                den_term_2 = Series(*num_arg_list)
            else:
                den_term_2 = tf, Series(*num_arg_list, expr.sys2)
        elif isinstance(num, TransferFunction) and isinstance(expr.sys2, Series):
            if num == tf:
                den_term_2 = Series(*den_arg_list)
            else:
                den_term_2 = Series(num, *den_arg_list)
        else:
            if num == tf:
                den_term_2 = Series(*den_arg_list)
            elif expr.sys2 == tf:
                den_term_2 = Series(*num_arg_list)
            else:
                den_term_2 = Series(*num_arg_list, *den_arg_list)

        numer = self._print(num)
        denom_1 = self._print(den_term_1)
        denom_2 = self._print(den_term_2)
        _sign = "+" if expr.sign == -1 else "-"

        return r"\frac{%s}{%s %s %s}" % (numer, denom_1, _sign, denom_2)

    def _print_MIMOFeedback(self, expr):
        from sympy.physics.control import MIMOSeries
        inv_mat = self._print(MIMOSeries(expr.sys2, expr.sys1))
        sys1 = self._print(expr.sys1)
        _sign = "+" if expr.sign == -1 else "-"
        return r"\left(I_{\tau} %s %s\right)^{-1} \cdot %s" % (_sign, inv_mat, sys1)

    def _print_TransferFunctionMatrix(self, expr):
        mat = self._print(expr._expr_mat)
        return r"%s_\tau" % mat

    def _print_DFT(self, expr):
        return r"\text{{{}}}_{{{}}}".format(expr.__class__.__name__, expr.n)
    _print_IDFT = _print_DFT

    def _print_NamedMorphism(self, morphism):
        pretty_name = self._print(Symbol(morphism.name))
        pretty_morphism = self._print_Morphism(morphism)
        return "%s:%s" % (pretty_name, pretty_morphism)

    def _print_IdentityMorphism(self, morphism):
        from sympy.categories import NamedMorphism
        return self._print_NamedMorphism(NamedMorphism(
            morphism.domain, morphism.codomain, "id"))

    def _print_CompositeMorphism(self, morphism):
        # All components of the morphism have names and it is thus
        # possible to build the name of the composite.
        component_names_list = [self._print(Symbol(component.name)) for
                                component in morphism.components]
        component_names_list.reverse()
        component_names = "\\circ ".join(component_names_list) + ":"

        pretty_morphism = self._print_Morphism(morphism)
        return component_names + pretty_morphism

    def _print_Category(self, morphism):
        return r"\mathbf{{{}}}".format(self._print(Symbol(morphism.name)))

    def _print_Diagram(self, diagram):
        if not diagram.premises:
            # This is an empty diagram.
            return self._print(S.EmptySet)

        latex_result = self._print(diagram.premises)
        if diagram.conclusions:
            latex_result += "\\Longrightarrow %s" % \
                            self._print(diagram.conclusions)

        return latex_result

    def _print_DiagramGrid(self, grid):
        latex_result = "\\begin{array}{%s}\n" % ("c" * grid.width)

        for i in range(grid.height):
            for j in range(grid.width):
                if grid[i, j]:
                    latex_result += latex(grid[i, j])
                latex_result += " "
                if j != grid.width - 1:
                    latex_result += "& "

            if i != grid.height - 1:
                latex_result += "\\\\"
            latex_result += "\n"

        latex_result += "\\end{array}\n"
        return latex_result

    def _print_FreeModule(self, M):
        return '{{{}}}^{{{}}}'.format(self._print(M.ring), self._print(M.rank))

    def _print_FreeModuleElement(self, m):
        # Print as row vector for convenience, for now.
        return r"\left[ {} \right]".format(",".join(
            '{' + self._print(x) + '}' for x in m))

    def _print_SubModule(self, m):
        gens = [[self._print(m.ring.to_sympy(x)) for x in g] for g in m.gens]
        curly = lambda o: r"{" + o + r"}"
        square = lambda o: r"\left[ " + o + r" \right]"
        gens_latex = ",".join(curly(square(",".join(curly(x) for x in g))) for g in gens)
        return r"\left\langle {} \right\rangle".format(gens_latex)

    def _print_SubQuotientModule(self, m):
        gens_latex = ",".join(["{" + self._print(g) + "}" for g in m.gens])
        return r"\left\langle {} \right\rangle".format(gens_latex)

    def _print_ModuleImplementedIdeal(self, m):
        gens = [m.ring.to_sympy(x) for [x] in m._module.gens]
        gens_latex = ",".join('{' + self._print(x) + '}' for x in gens)
        return r"\left\langle {} \right\rangle".format(gens_latex)

    def _print_Quaternion(self, expr):
        # TODO: This expression is potentially confusing,
        # shall we print it as `Quaternion( ... )`?
        s = [self.parenthesize(i, PRECEDENCE["Mul"], strict=True)
             for i in expr.args]
        a = [s[0]] + [i+" "+j for i, j in zip(s[1:], "ijk")]
        return " + ".join(a)

    def _print_QuotientRing(self, R):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(R.ring),
                 self._print(R.base_ideal))

    def _print_QuotientRingElement(self, x):
        x_latex = self._print(x.ring.to_sympy(x))
        return r"{{{}}} + {{{}}}".format(x_latex,
                 self._print(x.ring.base_ideal))

    def _print_QuotientModuleElement(self, m):
        data = [m.module.ring.to_sympy(x) for x in m.data]
        data_latex = r"\left[ {} \right]".format(",".join(
            '{' + self._print(x) + '}' for x in data))
        return r"{{{}}} + {{{}}}".format(data_latex,
                 self._print(m.module.killed_module))

    def _print_QuotientModule(self, M):
        # TODO nicer fractions for few generators...
        return r"\frac{{{}}}{{{}}}".format(self._print(M.base),
                 self._print(M.killed_module))

    def _print_MatrixHomomorphism(self, h):
        return r"{{{}}} : {{{}}} \to {{{}}}".format(self._print(h._sympy_matrix()),
            self._print(h.domain), self._print(h.codomain))

    def _print_Manifold(self, manifold):
        name, supers, subs = self._split_super_sub(manifold.name.name)

        name = r'\text{%s}' % name
        if supers:
            name += "^{%s}" % " ".join(supers)
        if subs:
            name += "_{%s}" % " ".join(subs)

        return name

    def _print_Patch(self, patch):
        return r'\text{%s}_{%s}' % (self._print(patch.name), self._print(patch.manifold))

    def _print_CoordSystem(self, coordsys):
        return r'\text{%s}^{\text{%s}}_{%s}' % (
            self._print(coordsys.name), self._print(coordsys.patch.name), self._print(coordsys.manifold)
        )

    def _print_CovarDerivativeOp(self, cvd):
        return r'\mathbb{\nabla}_{%s}' % self._print(cvd._wrt)

    def _print_BaseScalarField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return r'\mathbf{{{}}}'.format(self._print(Symbol(string)))

    def _print_BaseVectorField(self, field):
        string = field._coord_sys.symbols[field._index].name
        return r'\partial_{{{}}}'.format(self._print(Symbol(string)))

    def _print_Differential(self, diff):
        field = diff._form_field
        if hasattr(field, '_coord_sys'):
            string = field._coord_sys.symbols[field._index].name
            return r'\operatorname{{d}}{}'.format(self._print(Symbol(string)))
        else:
            string = self._print(field)
            return r'\operatorname{{d}}\left({}\right)'.format(string)

    def _print_Tr(self, p):
        # TODO: Handle indices
        contents = self._print(p.args[0])
        return r'\operatorname{{tr}}\left({}\right)'.format(contents)

    def _print_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\phi\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\phi\left(%s\right)' % self._print(expr.args[0])

    def _print_reduced_totient(self, expr, exp=None):
        if exp is not None:
            return r'\left(\lambda\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\lambda\left(%s\right)' % self._print(expr.args[0])

    def _print_divisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^{%s}%s" % (exp, tex)
        return r"\sigma%s" % tex

    def _print_udivisor_sigma(self, expr, exp=None):
        if len(expr.args) == 2:
            tex = r"_%s\left(%s\right)" % tuple(map(self._print,
                                                (expr.args[1], expr.args[0])))
        else:
            tex = r"\left(%s\right)" % self._print(expr.args[0])
        if exp is not None:
            return r"\sigma^*^{%s}%s" % (exp, tex)
        return r"\sigma^*%s" % tex

    def _print_primenu(self, expr, exp=None):
        if exp is not None:
            return r'\left(\nu\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\nu\left(%s\right)' % self._print(expr.args[0])

    def _print_primeomega(self, expr, exp=None):
        if exp is not None:
            return r'\left(\Omega\left(%s\right)\right)^{%s}' % \
                (self._print(expr.args[0]), exp)
        return r'\Omega\left(%s\right)' % self._print(expr.args[0])

    def _print_Str(self, s):
        return str(s.name)

    def _print_float(self, expr):
        return self._print(Float(expr))

    def _print_int(self, expr):
        return str(expr)

    def _print_mpz(self, expr):
        return str(expr)

    def _print_mpq(self, expr):
        return str(expr)

    def _print_fmpz(self, expr):
        return str(expr)

    def _print_fmpq(self, expr):
        return str(expr)

    def _print_Predicate(self, expr):
        return r"\operatorname{{Q}}_{{\text{{{}}}}}".format(latex_escape(str(expr.name)))

    def _print_AppliedPredicate(self, expr):
        pred = expr.function
        args = expr.arguments
        pred_latex = self._print(pred)
        args_latex = ', '.join([self._print(a) for a in args])
        return '%s(%s)' % (pred_latex, args_latex)

    def emptyPrinter(self, expr):
        # default to just printing as monospace, like would normally be shown
        s = super().emptyPrinter(expr)

        return r"\mathtt{\text{%s}}" % latex_escape(s)


def translate(s: str) -> str:
    r'''
    Check for a modifier ending the string.  If present, convert the
    modifier to latex and translate the rest recursively.

    Given a description of a Greek letter or other special character,
    return the appropriate latex.

    Let everything else pass as given.

    >>> from sympy.printing.latex import translate
    >>> translate('alphahatdotprime')
    "{\\dot{\\hat{\\alpha}}}'"
    '''
    # Process the rest
    tex = tex_greek_dictionary.get(s)
    if tex:
        return tex
    elif s.lower() in greek_letters_set:
        return "\\" + s.lower()
    elif s in other_symbols:
        return "\\" + s
    else:
        # Process modifiers, if any, and recurse
        for key in sorted(modifier_dict.keys(), key=len, reverse=True):
            if s.lower().endswith(key) and len(s) > len(key):
                return modifier_dict[key](translate(s[:-len(key)]))
        return s



@print_function(LatexPrinter)
def latex(expr, **settings):
    r"""Convert the given expression to LaTeX string representation.

    Parameters
    ==========
    full_prec: boolean, optional
        If set to True, a floating point number is printed with full precision.
    fold_frac_powers : boolean, optional
        Emit ``^{p/q}`` instead of ``^{\frac{p}{q}}`` for fractional powers.
    fold_func_brackets : boolean, optional
        Fold function brackets where applicable.
    fold_short_frac : boolean, optional
        Emit ``p / q`` instead of ``\frac{p}{q}`` when the denominator is
        simple enough (at most two terms and no powers). The default value is
        ``True`` for inline mode, ``False`` otherwise.
    inv_trig_style : string, optional
        How inverse trig functions should be displayed. Can be one of
        ``'abbreviated'``, ``'full'``, or ``'power'``. Defaults to
        ``'abbreviated'``.
    itex : boolean, optional
        Specifies if itex-specific syntax is used, including emitting
        ``$$...$$``.
    ln_notation : boolean, optional
        If set to ``True``, ``\ln`` is used instead of default ``\log``.
    long_frac_ratio : float or None, optional
        The allowed ratio of the width of the numerator to the width of the
        denominator before the printer breaks off long fractions. If ``None``
        (the default value), long fractions are not broken up.
    mat_delim : string, optional
        The delimiter to wrap around matrices. Can be one of ``'['``, ``'('``,
        or the empty string ``''``. Defaults to ``'['``.
    mat_str : string, optional
        Which matrix environment string to emit. ``'smallmatrix'``,
        ``'matrix'``, ``'array'``, etc. Defaults to ``'smallmatrix'`` for
        inline mode, ``'matrix'`` for matrices of no more than 10 columns, and
        ``'array'`` otherwise.
    mode: string, optional
        Specifies how the generated code will be delimited. ``mode`` can be one
        of ``'plain'``, ``'inline'``, ``'equation'`` or ``'equation*'``.  If
        ``mode`` is set to ``'plain'``, then the resulting code will not be
        delimited at all (this is the default). If ``mode`` is set to
        ``'inline'`` then inline LaTeX ``$...$`` will be used. If ``mode`` is
        set to ``'equation'`` or ``'equation*'``, the resulting code will be
        enclosed in the ``equation`` or ``equation*`` environment (remember to
        import ``amsmath`` for ``equation*``), unless the ``itex`` option is
        set. In the latter case, the ``$$...$$`` syntax is used.
    mul_symbol : string or None, optional
        The symbol to use for multiplication. Can be one of ``None``,
        ``'ldot'``, ``'dot'``, or ``'times'``.
    order: string, optional
        Any of the supported monomial orderings (currently ``'lex'``,
        ``'grlex'``, or ``'grevlex'``), ``'old'``, and ``'none'``. This
        parameter does nothing for `~.Mul` objects. Setting order to ``'old'``
        uses the compatibility ordering for ``~.Add`` defined in Printer. For
        very large expressions, set the ``order`` keyword to ``'none'`` if
        speed is a concern.
    symbol_names : dictionary of strings mapped to symbols, optional
        Dictionary of symbols and the custom strings they should be emitted as.
    root_notation : boolean, optional
        If set to ``False``, exponents of the form 1/n are printed in fractonal
        form. Default is ``True``, to print exponent in root form.
    mat_symbol_style : string, optional
        Can be either ``'plain'`` (default) or ``'bold'``. If set to
        ``'bold'``, a `~.MatrixSymbol` A will be printed as ``\mathbf{A}``,
        otherwise as ``A``.
    imaginary_unit : string, optional
        String to use for the imaginary unit. Defined options are ``'i'``
        (default) and ``'j'``. Adding ``r`` or ``t`` in front gives ``\mathrm``
        or ``\text``, so ``'ri'`` leads to ``\mathrm{i}`` which gives
        `\mathrm{i}`.
    gothic_re_im : boolean, optional
        If set to ``True``, `\Re` and `\Im` is used for ``re`` and ``im``, respectively.
        The default is ``False`` leading to `\operatorname{re}` and `\operatorname{im}`.
    decimal_separator : string, optional
        Specifies what separator to use to separate the whole and fractional parts of a
        floating point number as in `2.5` for the default, ``period`` or `2{,}5`
        when ``comma`` is specified. Lists, sets, and tuple are printed with semicolon
        separating the elements when ``comma`` is chosen. For example, [1; 2; 3] when
        ``comma`` is chosen and [1,2,3] for when ``period`` is chosen.
    parenthesize_super : boolean, optional
        If set to ``False``, superscripted expressions will not be parenthesized when
        powered. Default is ``True``, which parenthesizes the expression when powered.
    min: Integer or None, optional
        Sets the lower bound for the exponent to print floating point numbers in
        fixed-point format.
    max: Integer or None, optional
        Sets the upper bound for the exponent to print floating point numbers in
        fixed-point format.
    diff_operator: string, optional
        String to use for differential operator. Default is ``'d'``, to print in italic
        form. ``'rd'``, ``'td'`` are shortcuts for ``\mathrm{d}`` and ``\text{d}``.
    adjoint_style: string, optional
        String to use for the adjoint symbol. Defined options are ``'dagger'``
        (default),``'star'``, and ``'hermitian'``.

    Notes
    =====

    Not using a print statement for printing, results in double backslashes for
    latex commands since that's the way Python escapes backslashes in strings.

    >>> from sympy import latex, Rational
    >>> from sympy.abc import tau
    >>> latex((2*tau)**Rational(7,2))
    '8 \\sqrt{2} \\tau^{\\frac{7}{2}}'
    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    Examples
    ========

    >>> from sympy import latex, pi, sin, asin, Integral, Matrix, Rational, log
    >>> from sympy.abc import x, y, mu, r, tau

    Basic usage:

    >>> print(latex((2*tau)**Rational(7,2)))
    8 \sqrt{2} \tau^{\frac{7}{2}}

    ``mode`` and ``itex`` options:

    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$
    >>> print(latex((2*mu)**Rational(7,2), mode='plain'))
    8 \sqrt{2} \mu^{\frac{7}{2}}
    >>> print(latex((2*tau)**Rational(7,2), mode='inline'))
    $8 \sqrt{2} \tau^{7 / 2}$
    >>> print(latex((2*mu)**Rational(7,2), mode='equation*'))
    \begin{equation*}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation*}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation'))
    \begin{equation}8 \sqrt{2} \mu^{\frac{7}{2}}\end{equation}
    >>> print(latex((2*mu)**Rational(7,2), mode='equation', itex=True))
    $$8 \sqrt{2} \mu^{\frac{7}{2}}$$

    Fraction options:

    >>> print(latex((2*tau)**Rational(7,2), fold_frac_powers=True))
    8 \sqrt{2} \tau^{7/2}
    >>> print(latex((2*tau)**sin(Rational(7,2))))
    \left(2 \tau\right)^{\sin{\left(\frac{7}{2} \right)}}
    >>> print(latex((2*tau)**sin(Rational(7,2)), fold_func_brackets=True))
    \left(2 \tau\right)^{\sin {\frac{7}{2}}}
    >>> print(latex(3*x**2/y))
    \frac{3 x^{2}}{y}
    >>> print(latex(3*x**2/y, fold_short_frac=True))
    3 x^{2} / y
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=2))
    \frac{\int r\, dr}{2 \pi}
    >>> print(latex(Integral(r, r)/2/pi, long_frac_ratio=0))
    \frac{1}{2 \pi} \int r\, dr

    Multiplication options:

    >>> print(latex((2*tau)**sin(Rational(7,2)), mul_symbol="times"))
    \left(2 \times \tau\right)^{\sin{\left(\frac{7}{2} \right)}}

    Trig options:

    >>> print(latex(asin(Rational(7,2))))
    \operatorname{asin}{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="full"))
    \arcsin{\left(\frac{7}{2} \right)}
    >>> print(latex(asin(Rational(7,2)), inv_trig_style="power"))
    \sin^{-1}{\left(\frac{7}{2} \right)}

    Matrix options:

    >>> print(latex(Matrix(2, 1, [x, y])))
    \left[\begin{matrix}x\\y\end{matrix}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_str = "array"))
    \left[\begin{array}{c}x\\y\end{array}\right]
    >>> print(latex(Matrix(2, 1, [x, y]), mat_delim="("))
    \left(\begin{matrix}x\\y\end{matrix}\right)

    Custom printing of symbols:

    >>> print(latex(x**2, symbol_names={x: 'x_i'}))
    x_i^{2}

    Logarithms:

    >>> print(latex(log(10)))
    \log{\left(10 \right)}
    >>> print(latex(log(10), ln_notation=True))
    \ln{\left(10 \right)}

    ``latex()`` also supports the builtin container types :class:`list`,
    :class:`tuple`, and :class:`dict`:

    >>> print(latex([2/x, y], mode='inline'))
    $\left[ 2 / x, \  y\right]$

    Unsupported types are rendered as monospaced plaintext:

    >>> print(latex(int))
    \mathtt{\text{<class 'int'>}}
    >>> print(latex("plain % text"))
    \mathtt{\text{plain \% text}}

    See :ref:`printer_method_example` for an example of how to override
    this behavior for your own types by implementing ``_latex``.

    .. versionchanged:: 1.7.0
        Unsupported types no longer have their ``str`` representation treated as valid latex.

    """
    return LatexPrinter(settings).doprint(expr)


def print_latex(expr, **settings):
    """Prints LaTeX representation of the given expression. Takes the same
    settings as ``latex()``."""

    print(latex(expr, **settings))


def multiline_latex(lhs, rhs, terms_per_line=1, environment="align*", use_dots=False, **settings):
    r"""
    This function generates a LaTeX equation with a multiline right-hand side
    in an ``align*``, ``eqnarray`` or ``IEEEeqnarray`` environment.

    Parameters
    ==========

    lhs : Expr
        Left-hand side of equation

    rhs : Expr
        Right-hand side of equation

    terms_per_line : integer, optional
        Number of terms per line to print. Default is 1.

    environment : "string", optional
        Which LaTeX wnvironment to use for the output. Options are "align*"
        (default), "eqnarray", and "IEEEeqnarray".

    use_dots : boolean, optional
        If ``True``, ``\\dots`` is added to the end of each line. Default is ``False``.

    Examples
    ========

    >>> from sympy import multiline_latex, symbols, sin, cos, exp, log, I
    >>> x, y, alpha = symbols('x y alpha')
    >>> expr = sin(alpha*y) + exp(I*alpha) - cos(log(y))
    >>> print(multiline_latex(x, expr))
    \begin{align*}
    x = & e^{i \alpha} \\
    & + \sin{\left(\alpha y \right)} \\
    & - \cos{\left(\log{\left(y \right)} \right)}
    \end{align*}

    Using at most two terms per line:
    >>> print(multiline_latex(x, expr, 2))
    \begin{align*}
    x = & e^{i \alpha} + \sin{\left(\alpha y \right)} \\
    & - \cos{\left(\log{\left(y \right)} \right)}
    \end{align*}

    Using ``eqnarray`` and dots:
    >>> print(multiline_latex(x, expr, terms_per_line=2, environment="eqnarray", use_dots=True))
    \begin{eqnarray}
    x & = & e^{i \alpha} + \sin{\left(\alpha y \right)} \dots\nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)}
    \end{eqnarray}

    Using ``IEEEeqnarray``:
    >>> print(multiline_latex(x, expr, environment="IEEEeqnarray"))
    \begin{IEEEeqnarray}{rCl}
    x & = & e^{i \alpha} \nonumber\\
    & & + \sin{\left(\alpha y \right)} \nonumber\\
    & & - \cos{\left(\log{\left(y \right)} \right)}
    \end{IEEEeqnarray}

    Notes
    =====

    All optional parameters from ``latex`` can also be used.

    """

    # Based on code from https://github.com/sympy/sympy/issues/3001
    l = LatexPrinter(**settings)
    if environment == "eqnarray":
        result = r'\begin{eqnarray}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{eqnarray}'
        doubleet = True
    elif environment == "IEEEeqnarray":
        result = r'\begin{IEEEeqnarray}{rCl}' + '\n'
        first_term = '& = &'
        nonumber = r'\nonumber'
        end_term = '\n\\end{IEEEeqnarray}'
        doubleet = True
    elif environment == "align*":
        result = r'\begin{align*}' + '\n'
        first_term = '= &'
        nonumber = ''
        end_term =  '\n\\end{align*}'
        doubleet = False
    else:
        raise ValueError("Unknown environment: {}".format(environment))
    dots = ''
    if use_dots:
        dots=r'\dots'
    terms = rhs.as_ordered_terms()
    n_terms = len(terms)
    term_count = 1
    for i in range(n_terms):
        term = terms[i]
        term_start = ''
        term_end = ''
        sign = '+'
        if term_count > terms_per_line:
            if doubleet:
                term_start = '& & '
            else:
                term_start = '& '
            term_count = 1
        if term_count == terms_per_line:
            # End of line
            if i < n_terms-1:
                # There are terms remaining
                term_end = dots + nonumber + r'\\' + '\n'
            else:
                term_end = ''

        if term.as_ordered_factors()[0] == -1:
            term = -1*term
            sign = r'-'
        if i == 0: # beginning
            if sign == '+':
                sign = ''
            result += r'{:s} {:s}{:s} {:s} {:s}'.format(l.doprint(lhs),
                        first_term, sign, l.doprint(term), term_end)
        else:
            result += r'{:s}{:s} {:s} {:s}'.format(term_start, sign,
                        l.doprint(term), term_end)
        term_count += 1
    result += end_term
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\suite\test_reflection.py ===
# testing/suite/test_reflection.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

import contextlib
import operator
import re

import sqlalchemy as sa
from .. import config
from .. import engines
from .. import eq_
from .. import eq_regex
from .. import expect_raises
from .. import expect_raises_message
from .. import expect_warnings
from .. import fixtures
from .. import is_
from ..provision import get_temp_table_name
from ..provision import temp_table_keyword_args
from ..schema import Column
from ..schema import Table
from ... import Boolean
from ... import DateTime
from ... import event
from ... import ForeignKey
from ... import func
from ... import Identity
from ... import inspect
from ... import Integer
from ... import MetaData
from ... import String
from ... import testing
from ... import types as sql_types
from ...engine import Inspector
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...exc import NoSuchTableError
from ...exc import UnreflectableTableError
from ...schema import DDL
from ...schema import Index
from ...sql.elements import quoted_name
from ...sql.schema import BLANK_SCHEMA
from ...testing import ComparesIndexes
from ...testing import ComparesTables
from ...testing import is_false
from ...testing import is_true
from ...testing import mock


metadata, users = None, None


class OneConnectionTablesTest(fixtures.TablesTest):
    @classmethod
    def setup_bind(cls):
        # TODO: when temp tables are subject to server reset,
        # this will also have to disable that server reset from
        # happening
        if config.requirements.independent_connections.enabled:
            from sqlalchemy import pool

            return engines.testing_engine(
                options=dict(poolclass=pool.StaticPool, scope="class"),
            )
        else:
            return config.db


class HasTableTest(OneConnectionTablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "test_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String(50)),
        )
        if testing.requires.schemas.enabled:
            Table(
                "test_table_s",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("data", String(50)),
                schema=config.test_schema,
            )

        if testing.requires.view_reflection:
            cls.define_views(metadata)
        if testing.requires.has_temp_table.enabled:
            cls.define_temp_tables(metadata)

    @classmethod
    def define_views(cls, metadata):
        query = "CREATE VIEW vv AS SELECT id, data FROM test_table"

        event.listen(metadata, "after_create", DDL(query))
        event.listen(metadata, "before_drop", DDL("DROP VIEW vv"))

        if testing.requires.schemas.enabled:
            query = (
                "CREATE VIEW %s.vv AS SELECT id, data FROM %s.test_table_s"
                % (
                    config.test_schema,
                    config.test_schema,
                )
            )
            event.listen(metadata, "after_create", DDL(query))
            event.listen(
                metadata,
                "before_drop",
                DDL("DROP VIEW %s.vv" % (config.test_schema)),
            )

    @classmethod
    def temp_table_name(cls):
        return get_temp_table_name(
            config, config.db, f"user_tmp_{config.ident}"
        )

    @classmethod
    def define_temp_tables(cls, metadata):
        kw = temp_table_keyword_args(config, config.db)
        table_name = cls.temp_table_name()
        user_tmp = Table(
            table_name,
            metadata,
            Column("id", sa.INT, primary_key=True),
            Column("name", sa.VARCHAR(50)),
            **kw,
        )
        if (
            testing.requires.view_reflection.enabled
            and testing.requires.temporary_views.enabled
        ):
            event.listen(
                user_tmp,
                "after_create",
                DDL(
                    "create temporary view user_tmp_v as "
                    "select * from user_tmp_%s" % config.ident
                ),
            )
            event.listen(user_tmp, "before_drop", DDL("drop view user_tmp_v"))

    def test_has_table(self):
        with config.db.begin() as conn:
            is_true(config.db.dialect.has_table(conn, "test_table"))
            is_false(config.db.dialect.has_table(conn, "test_table_s"))
            is_false(config.db.dialect.has_table(conn, "nonexistent_table"))

    def test_has_table_cache(self, metadata):
        insp = inspect(config.db)
        is_true(insp.has_table("test_table"))
        nt = Table("new_table", metadata, Column("col", Integer))
        is_false(insp.has_table("new_table"))
        nt.create(config.db)
        try:
            is_false(insp.has_table("new_table"))
            insp.clear_cache()
            is_true(insp.has_table("new_table"))
        finally:
            nt.drop(config.db)

    @testing.requires.schemas
    def test_has_table_schema(self):
        with config.db.begin() as conn:
            is_false(
                config.db.dialect.has_table(
                    conn, "test_table", schema=config.test_schema
                )
            )
            is_true(
                config.db.dialect.has_table(
                    conn, "test_table_s", schema=config.test_schema
                )
            )
            is_false(
                config.db.dialect.has_table(
                    conn, "nonexistent_table", schema=config.test_schema
                )
            )

    @testing.requires.schemas
    def test_has_table_nonexistent_schema(self):
        with config.db.begin() as conn:
            is_false(
                config.db.dialect.has_table(
                    conn, "test_table", schema="nonexistent_schema"
                )
            )

    @testing.requires.views
    def test_has_table_view(self, connection):
        insp = inspect(connection)
        is_true(insp.has_table("vv"))

    @testing.requires.has_temp_table
    def test_has_table_temp_table(self, connection):
        insp = inspect(connection)
        temp_table_name = self.temp_table_name()
        is_true(insp.has_table(temp_table_name))

    @testing.requires.has_temp_table
    @testing.requires.view_reflection
    @testing.requires.temporary_views
    def test_has_table_temp_view(self, connection):
        insp = inspect(connection)
        is_true(insp.has_table("user_tmp_v"))

    @testing.requires.views
    @testing.requires.schemas
    def test_has_table_view_schema(self, connection):
        insp = inspect(connection)
        is_true(insp.has_table("vv", config.test_schema))


class HasIndexTest(fixtures.TablesTest):
    __backend__ = True
    __requires__ = ("index_reflection",)

    @classmethod
    def define_tables(cls, metadata):
        tt = Table(
            "test_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", String(50)),
            Column("data2", String(50)),
        )
        Index("my_idx", tt.c.data)

        if testing.requires.schemas.enabled:
            tt = Table(
                "test_table",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("data", String(50)),
                schema=config.test_schema,
            )
            Index("my_idx_s", tt.c.data)

    kind = testing.combinations("dialect", "inspector", argnames="kind")

    def _has_index(self, kind, conn):
        if kind == "dialect":
            return lambda *a, **k: config.db.dialect.has_index(conn, *a, **k)
        else:
            return inspect(conn).has_index

    @kind
    def test_has_index(self, kind, connection, metadata):
        meth = self._has_index(kind, connection)
        assert meth("test_table", "my_idx")
        assert not meth("test_table", "my_idx_s")
        assert not meth("nonexistent_table", "my_idx")
        assert not meth("test_table", "nonexistent_idx")

        assert not meth("test_table", "my_idx_2")
        assert not meth("test_table_2", "my_idx_3")
        idx = Index("my_idx_2", self.tables.test_table.c.data2)
        tbl = Table(
            "test_table_2",
            metadata,
            Column("foo", Integer),
            Index("my_idx_3", "foo"),
        )
        idx.create(connection)
        tbl.create(connection)
        try:
            if kind == "inspector":
                assert not meth("test_table", "my_idx_2")
                assert not meth("test_table_2", "my_idx_3")
                meth.__self__.clear_cache()
            assert meth("test_table", "my_idx_2") is True
            assert meth("test_table_2", "my_idx_3") is True
        finally:
            tbl.drop(connection)
            idx.drop(connection)

    @testing.requires.schemas
    @kind
    def test_has_index_schema(self, kind, connection):
        meth = self._has_index(kind, connection)
        assert meth("test_table", "my_idx_s", schema=config.test_schema)
        assert not meth("test_table", "my_idx", schema=config.test_schema)
        assert not meth(
            "nonexistent_table", "my_idx_s", schema=config.test_schema
        )
        assert not meth(
            "test_table", "nonexistent_idx_s", schema=config.test_schema
        )


class BizarroCharacterFKResolutionTest(fixtures.TestBase):
    """tests for #10275"""

    __backend__ = True
    __requires__ = ("foreign_key_constraint_reflection",)

    @testing.combinations(
        ("id",), ("(3)",), ("col%p",), ("[brack]",), argnames="columnname"
    )
    @testing.variation("use_composite", [True, False])
    @testing.combinations(
        ("plain",),
        ("(2)",),
        ("per % cent",),
        ("[brackets]",),
        argnames="tablename",
    )
    def test_fk_ref(
        self, connection, metadata, use_composite, tablename, columnname
    ):
        tt = Table(
            tablename,
            metadata,
            Column(columnname, Integer, key="id", primary_key=True),
            test_needs_fk=True,
        )
        if use_composite:
            tt.append_column(Column("id2", Integer, primary_key=True))

        if use_composite:
            Table(
                "other",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("ref", Integer),
                Column("ref2", Integer),
                sa.ForeignKeyConstraint(["ref", "ref2"], [tt.c.id, tt.c.id2]),
                test_needs_fk=True,
            )
        else:
            Table(
                "other",
                metadata,
                Column("id", Integer, primary_key=True),
                Column("ref", ForeignKey(tt.c.id)),
                test_needs_fk=True,
            )

        metadata.create_all(connection)

        m2 = MetaData()

        o2 = Table("other", m2, autoload_with=connection)
        t1 = m2.tables[tablename]

        assert o2.c.ref.references(t1.c[0])
        if use_composite:
            assert o2.c.ref2.references(t1.c[1])


class QuotedNameArgumentTest(fixtures.TablesTest):
    run_create_tables = "once"
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "quote ' one",
            metadata,
            Column("id", Integer),
            Column("name", String(50)),
            Column("data", String(50)),
            Column("related_id", Integer),
            sa.PrimaryKeyConstraint("id", name="pk quote ' one"),
            sa.Index("ix quote ' one", "name"),
            sa.UniqueConstraint(
                "data",
                name="uq quote' one",
            ),
            sa.ForeignKeyConstraint(
                ["id"], ["related.id"], name="fk quote ' one"
            ),
            sa.CheckConstraint("name != 'foo'", name="ck quote ' one"),
            comment=r"""quote ' one comment""",
            test_needs_fk=True,
        )

        if testing.requires.symbol_names_w_double_quote.enabled:
            Table(
                'quote " two',
                metadata,
                Column("id", Integer),
                Column("name", String(50)),
                Column("data", String(50)),
                Column("related_id", Integer),
                sa.PrimaryKeyConstraint("id", name='pk quote " two'),
                sa.Index('ix quote " two', "name"),
                sa.UniqueConstraint(
                    "data",
                    name='uq quote" two',
                ),
                sa.ForeignKeyConstraint(
                    ["id"], ["related.id"], name='fk quote " two'
                ),
                sa.CheckConstraint("name != 'foo'", name='ck quote " two '),
                comment=r"""quote " two comment""",
                test_needs_fk=True,
            )

        Table(
            "related",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("related", Integer),
            test_needs_fk=True,
        )

        if testing.requires.view_column_reflection.enabled:
            if testing.requires.symbol_names_w_double_quote.enabled:
                names = [
                    "quote ' one",
                    'quote " two',
                ]
            else:
                names = [
                    "quote ' one",
                ]
            for name in names:
                query = "CREATE VIEW %s AS SELECT * FROM %s" % (
                    config.db.dialect.identifier_preparer.quote(
                        "view %s" % name
                    ),
                    config.db.dialect.identifier_preparer.quote(name),
                )

                event.listen(metadata, "after_create", DDL(query))
                event.listen(
                    metadata,
                    "before_drop",
                    DDL(
                        "DROP VIEW %s"
                        % config.db.dialect.identifier_preparer.quote(
                            "view %s" % name
                        )
                    ),
                )

    def quote_fixtures(fn):
        return testing.combinations(
            ("quote ' one",),
            ('quote " two', testing.requires.symbol_names_w_double_quote),
        )(fn)

    @quote_fixtures
    def test_get_table_options(self, name):
        insp = inspect(config.db)

        if testing.requires.reflect_table_options.enabled:
            res = insp.get_table_options(name)
            is_true(isinstance(res, dict))
        else:
            with expect_raises(NotImplementedError):
                insp.get_table_options(name)

    @quote_fixtures
    @testing.requires.view_column_reflection
    def test_get_view_definition(self, name):
        insp = inspect(config.db)
        assert insp.get_view_definition("view %s" % name)

    @quote_fixtures
    def test_get_columns(self, name):
        insp = inspect(config.db)
        assert insp.get_columns(name)

    @quote_fixtures
    def test_get_pk_constraint(self, name):
        insp = inspect(config.db)
        assert insp.get_pk_constraint(name)

    @quote_fixtures
    @testing.requires.foreign_key_constraint_reflection
    def test_get_foreign_keys(self, name):
        insp = inspect(config.db)
        assert insp.get_foreign_keys(name)

    @quote_fixtures
    @testing.requires.index_reflection
    def test_get_indexes(self, name):
        insp = inspect(config.db)
        assert insp.get_indexes(name)

    @quote_fixtures
    @testing.requires.unique_constraint_reflection
    def test_get_unique_constraints(self, name):
        insp = inspect(config.db)
        assert insp.get_unique_constraints(name)

    @quote_fixtures
    @testing.requires.comment_reflection
    def test_get_table_comment(self, name):
        insp = inspect(config.db)
        assert insp.get_table_comment(name)

    @quote_fixtures
    @testing.requires.check_constraint_reflection
    def test_get_check_constraints(self, name):
        insp = inspect(config.db)
        assert insp.get_check_constraints(name)


def _multi_combination(fn):
    schema = testing.combinations(
        None,
        (
            lambda: config.test_schema,
            testing.requires.schemas,
        ),
        argnames="schema",
    )
    scope = testing.combinations(
        ObjectScope.DEFAULT,
        ObjectScope.TEMPORARY,
        ObjectScope.ANY,
        argnames="scope",
    )
    kind = testing.combinations(
        ObjectKind.TABLE,
        ObjectKind.VIEW,
        ObjectKind.MATERIALIZED_VIEW,
        ObjectKind.ANY,
        ObjectKind.ANY_VIEW,
        ObjectKind.TABLE | ObjectKind.VIEW,
        ObjectKind.TABLE | ObjectKind.MATERIALIZED_VIEW,
        argnames="kind",
    )
    filter_names = testing.combinations(True, False, argnames="use_filter")

    return schema(scope(kind(filter_names(fn))))


class ComponentReflectionTest(ComparesTables, OneConnectionTablesTest):
    run_inserts = run_deletes = None

    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        cls.define_reflected_tables(metadata, None)
        if testing.requires.schemas.enabled:
            cls.define_reflected_tables(metadata, testing.config.test_schema)

    @classmethod
    def define_reflected_tables(cls, metadata, schema):
        if schema:
            schema_prefix = schema + "."
        else:
            schema_prefix = ""

        if testing.requires.self_referential_foreign_keys.enabled:
            parent_id_args = (
                ForeignKey(
                    "%susers.user_id" % schema_prefix, name="user_id_fk"
                ),
            )
        else:
            parent_id_args = ()
        users = Table(
            "users",
            metadata,
            Column("user_id", sa.INT, primary_key=True),
            Column("test1", sa.CHAR(5), nullable=False),
            Column("test2", sa.Float(), nullable=False),
            Column("parent_user_id", sa.Integer, *parent_id_args),
            sa.CheckConstraint(
                "test2 > 0",
                name="zz_test2_gt_zero",
                comment="users check constraint",
            ),
            sa.CheckConstraint("test2 <= 1000"),
            schema=schema,
            test_needs_fk=True,
        )

        Table(
            "dingalings",
            metadata,
            Column("dingaling_id", sa.Integer, primary_key=True),
            Column(
                "address_id",
                sa.Integer,
                ForeignKey(
                    "%semail_addresses.address_id" % schema_prefix,
                    name="zz_email_add_id_fg",
                    comment="di fk comment",
                ),
            ),
            Column(
                "id_user",
                sa.Integer,
                ForeignKey("%susers.user_id" % schema_prefix),
            ),
            Column("data", sa.String(30), unique=True),
            sa.CheckConstraint(
                "address_id > 0 AND address_id < 1000",
                name="address_id_gt_zero",
            ),
            sa.UniqueConstraint(
                "address_id",
                "dingaling_id",
                name="zz_dingalings_multiple",
                comment="di unique comment",
            ),
            schema=schema,
            test_needs_fk=True,
        )
        Table(
            "email_addresses",
            metadata,
            Column("address_id", sa.Integer),
            Column("remote_user_id", sa.Integer, ForeignKey(users.c.user_id)),
            Column("email_address", sa.String(20), index=True),
            sa.PrimaryKeyConstraint(
                "address_id", name="email_ad_pk", comment="ea pk comment"
            ),
            schema=schema,
            test_needs_fk=True,
        )
        Table(
            "comment_test",
            metadata,
            Column("id", sa.Integer, primary_key=True, comment="id comment"),
            Column("data", sa.String(20), comment="data % comment"),
            Column(
                "d2",
                sa.String(20),
                comment=r"""Comment types type speedily ' " \ '' Fun!""",
            ),
            Column("d3", sa.String(42), comment="Comment\nwith\rescapes"),
            schema=schema,
            comment=r"""the test % ' " \ table comment""",
        )
        Table(
            "no_constraints",
            metadata,
            Column("data", sa.String(20)),
            schema=schema,
            comment="no\nconstraints\rhas\fescaped\vcomment",
        )

        if testing.requires.cross_schema_fk_reflection.enabled:
            if schema is None:
                Table(
                    "local_table",
                    metadata,
                    Column("id", sa.Integer, primary_key=True),
                    Column("data", sa.String(20)),
                    Column(
                        "remote_id",
                        ForeignKey(
                            "%s.remote_table_2.id" % testing.config.test_schema
                        ),
                    ),
                    test_needs_fk=True,
                    schema=config.db.dialect.default_schema_name,
                )
            else:
                Table(
                    "remote_table",
                    metadata,
                    Column("id", sa.Integer, primary_key=True),
                    Column(
                        "local_id",
                        ForeignKey(
                            "%s.local_table.id"
                            % config.db.dialect.default_schema_name
                        ),
                    ),
                    Column("data", sa.String(20)),
                    schema=schema,
                    test_needs_fk=True,
                )
                Table(
                    "remote_table_2",
                    metadata,
                    Column("id", sa.Integer, primary_key=True),
                    Column("data", sa.String(20)),
                    schema=schema,
                    test_needs_fk=True,
                )

        if testing.requires.index_reflection.enabled:
            Index("users_t_idx", users.c.test1, users.c.test2, unique=True)
            Index(
                "users_all_idx", users.c.user_id, users.c.test2, users.c.test1
            )

            if not schema:
                # test_needs_fk is at the moment to force MySQL InnoDB
                noncol_idx_test_nopk = Table(
                    "noncol_idx_test_nopk",
                    metadata,
                    Column("q", sa.String(5)),
                    test_needs_fk=True,
                )

                noncol_idx_test_pk = Table(
                    "noncol_idx_test_pk",
                    metadata,
                    Column("id", sa.Integer, primary_key=True),
                    Column("q", sa.String(5)),
                    test_needs_fk=True,
                )

                if (
                    testing.requires.indexes_with_ascdesc.enabled
                    and testing.requires.reflect_indexes_with_ascdesc.enabled
                ):
                    Index("noncol_idx_nopk", noncol_idx_test_nopk.c.q.desc())
                    Index("noncol_idx_pk", noncol_idx_test_pk.c.q.desc())

        if testing.requires.view_column_reflection.enabled:
            cls.define_views(metadata, schema)
        if not schema and testing.requires.temp_table_reflection.enabled:
            cls.define_temp_tables(metadata)

    @classmethod
    def temp_table_name(cls):
        return get_temp_table_name(
            config, config.db, f"user_tmp_{config.ident}"
        )

    @classmethod
    def define_temp_tables(cls, metadata):
        kw = temp_table_keyword_args(config, config.db)
        table_name = cls.temp_table_name()
        user_tmp = Table(
            table_name,
            metadata,
            Column("id", sa.INT, primary_key=True),
            Column("name", sa.VARCHAR(50)),
            Column("foo", sa.INT),
            # disambiguate temp table unique constraint names.  this is
            # pretty arbitrary for a generic dialect however we are doing
            # it to suit SQL Server which will produce name conflicts for
            # unique constraints created against temp tables in different
            # databases.
            # https://www.arbinada.com/en/node/1645
            sa.UniqueConstraint("name", name=f"user_tmp_uq_{config.ident}"),
            sa.Index("user_tmp_ix", "foo"),
            **kw,
        )
        if (
            testing.requires.view_reflection.enabled
            and testing.requires.temporary_views.enabled
        ):
            event.listen(
                user_tmp,
                "after_create",
                DDL(
                    "create temporary view user_tmp_v as "
                    "select * from user_tmp_%s" % config.ident
                ),
            )
            event.listen(user_tmp, "before_drop", DDL("drop view user_tmp_v"))

    @classmethod
    def define_views(cls, metadata, schema):
        if testing.requires.materialized_views.enabled:
            materialized = {"dingalings"}
        else:
            materialized = set()
        for table_name in ("users", "email_addresses", "dingalings"):
            fullname = table_name
            if schema:
                fullname = f"{schema}.{table_name}"
            view_name = fullname + "_v"
            prefix = "MATERIALIZED " if table_name in materialized else ""
            query = (
                f"CREATE {prefix}VIEW {view_name} AS SELECT * FROM {fullname}"
            )

            event.listen(metadata, "after_create", DDL(query))
            if table_name in materialized:
                index_name = "mat_index"
                if schema and testing.against("oracle"):
                    index_name = f"{schema}.{index_name}"
                idx = f"CREATE INDEX {index_name} ON {view_name}(data)"
                event.listen(metadata, "after_create", DDL(idx))
            event.listen(
                metadata, "before_drop", DDL(f"DROP {prefix}VIEW {view_name}")
            )

    def _resolve_kind(self, kind, tables, views, materialized):
        res = {}
        if ObjectKind.TABLE in kind:
            res.update(tables)
        if ObjectKind.VIEW in kind:
            res.update(views)
        if ObjectKind.MATERIALIZED_VIEW in kind:
            res.update(materialized)
        return res

    def _resolve_views(self, views, materialized):
        if not testing.requires.view_column_reflection.enabled:
            materialized.clear()
            views.clear()
        elif not testing.requires.materialized_views.enabled:
            views.update(materialized)
            materialized.clear()

    def _resolve_names(self, schema, scope, filter_names, values):
        scope_filter = lambda _: True  # noqa: E731
        if scope is ObjectScope.DEFAULT:
            scope_filter = lambda k: "tmp" not in k[1]  # noqa: E731
        if scope is ObjectScope.TEMPORARY:
            scope_filter = lambda k: "tmp" in k[1]  # noqa: E731

        removed = {
            None: {"remote_table", "remote_table_2"},
            testing.config.test_schema: {
                "local_table",
                "noncol_idx_test_nopk",
                "noncol_idx_test_pk",
                "user_tmp_v",
                self.temp_table_name(),
            },
        }
        if not testing.requires.cross_schema_fk_reflection.enabled:
            removed[None].add("local_table")
            removed[testing.config.test_schema].update(
                ["remote_table", "remote_table_2"]
            )
        if not testing.requires.index_reflection.enabled:
            removed[None].update(
                ["noncol_idx_test_nopk", "noncol_idx_test_pk"]
            )
        if (
            not testing.requires.temp_table_reflection.enabled
            or not testing.requires.temp_table_names.enabled
        ):
            removed[None].update(["user_tmp_v", self.temp_table_name()])
        if not testing.requires.temporary_views.enabled:
            removed[None].update(["user_tmp_v"])

        res = {
            k: v
            for k, v in values.items()
            if scope_filter(k)
            and k[1] not in removed[schema]
            and (not filter_names or k[1] in filter_names)
        }
        return res

    def exp_options(
        self,
        schema=None,
        scope=ObjectScope.ANY,
        kind=ObjectKind.ANY,
        filter_names=None,
    ):
        materialized = {(schema, "dingalings_v"): mock.ANY}
        views = {
            (schema, "email_addresses_v"): mock.ANY,
            (schema, "users_v"): mock.ANY,
            (schema, "user_tmp_v"): mock.ANY,
        }
        self._resolve_views(views, materialized)
        tables = {
            (schema, "users"): mock.ANY,
            (schema, "dingalings"): mock.ANY,
            (schema, "email_addresses"): mock.ANY,
            (schema, "comment_test"): mock.ANY,
            (schema, "no_constraints"): mock.ANY,
            (schema, "local_table"): mock.ANY,
            (schema, "remote_table"): mock.ANY,
            (schema, "remote_table_2"): mock.ANY,
            (schema, "noncol_idx_test_nopk"): mock.ANY,
            (schema, "noncol_idx_test_pk"): mock.ANY,
            (schema, self.temp_table_name()): mock.ANY,
        }
        res = self._resolve_kind(kind, tables, views, materialized)
        res = self._resolve_names(schema, scope, filter_names, res)
        return res

    def exp_comments(
        self,
        schema=None,
        scope=ObjectScope.ANY,
        kind=ObjectKind.ANY,
        filter_names=None,
    ):
        empty = {"text": None}
        materialized = {(schema, "dingalings_v"): empty}
        views = {
            (schema, "email_addresses_v"): empty,
            (schema, "users_v"): empty,
            (schema, "user_tmp_v"): empty,
        }
        self._resolve_views(views, materialized)
        tables = {
            (schema, "users"): empty,
            (schema, "dingalings"): empty,
            (schema, "email_addresses"): empty,
            (schema, "comment_test"): {
                "text": r"""the test % ' " \ table comment"""
            },
            (schema, "no_constraints"): {
                "text": "no\nconstraints\rhas\fescaped\vcomment"
            },
            (schema, "local_table"): empty,
            (schema, "remote_table"): empty,
            (schema, "remote_table_2"): empty,
            (schema, "noncol_idx_test_nopk"): empty,
            (schema, "noncol_idx_test_pk"): empty,
            (schema, self.temp_table_name()): empty,
        }
        res = self._resolve_kind(kind, tables, views, materialized)
        res = self._resolve_names(schema, scope, filter_names, res)
        return res

    def exp_columns(
        self,
        schema=None,
        scope=ObjectScope.ANY,
        kind=ObjectKind.ANY,
        filter_names=None,
    ):
        def col(
            name, auto=False, default=mock.ANY, comment=None, nullable=True
        ):
            res = {
                "name": name,
                "autoincrement": auto,
                "type": mock.ANY,
                "default": default,
                "comment": comment,
                "nullable": nullable,
            }
            if auto == "omit":
                res.pop("autoincrement")
            return res

        def pk(name, **kw):
            kw = {"auto": True, "default": mock.ANY, "nullable": False, **kw}
            return col(name, **kw)

        materialized = {
            (schema, "dingalings_v"): [
                col("dingaling_id", auto="omit", nullable=mock.ANY),
                col("address_id"),
                col("id_user"),
                col("data"),
            ]
        }
        views = {
            (schema, "email_addresses_v"): [
                col("address_id", auto="omit", nullable=mock.ANY),
                col("remote_user_id"),
                col("email_address"),
            ],
            (schema, "users_v"): [
                col("user_id", auto="omit", nullable=mock.ANY),
                col("test1", nullable=mock.ANY),
                col("test2", nullable=mock.ANY),
                col("parent_user_id"),
            ],
            (schema, "user_tmp_v"): [
                col("id", auto="omit", nullable=mock.ANY),
                col("name"),
                col("foo"),
            ],
        }
        self._resolve_views(views, materialized)
        tables = {
            (schema, "users"): [
                pk("user_id"),
                col("test1", nullable=False),
                col("test2", nullable=False),
                col("parent_user_id"),
            ],
            (schema, "dingalings"): [
                pk("dingaling_id"),
                col("address_id"),
                col("id_user"),
                col("data"),
            ],
            (schema, "email_addresses"): [
                pk("address_id"),
                col("remote_user_id"),
                col("email_address"),
            ],
            (schema, "comment_test"): [
                pk("id", comment="id comment"),
                col("data", comment="data % comment"),
                col(
                    "d2",
                    comment=r"""Comment types type speedily ' " \ '' Fun!""",
                ),
                col("d3", comment="Comment\nwith\rescapes"),
            ],
            (schema, "no_constraints"): [col("data")],
            (schema, "local_table"): [pk("id"), col("data"), col("remote_id")],
            (schema, "remote_table"): [pk("id"), col("local_id"), col("data")],
            (schema, "remote_table_2"): [pk("id"), col("data")],
            (schema, "noncol_idx_test_nopk"): [col("q")],
            (schema, "noncol_idx_test_pk"): [pk("id"), col("q")],
            (schema, self.temp_table_name()): [
                pk("id"),
                col("name"),
                col("foo"),
            ],
        }
        res = self._resolve_kind(kind, tables, views, materialized)
        res = self._resolve_names(schema, scope, filter_names, res)
        return res

    @property
    def _required_column_keys(self):
        return {"name", "type", "nullable", "default"}

    def exp_pks(
        self,
        schema=None,
        scope=ObjectScope.ANY,
        kind=ObjectKind.ANY,
        filter_names=None,
    ):
        def pk(*cols, name=mock.ANY, comment=None):
            return {
                "constrained_columns": list(cols),
                "name": name,
                "comment": comment,
            }

        empty = pk(name=None)
        if testing.requires.materialized_views_reflect_pk.enabled:
            materialized = {(schema, "dingalings_v"): pk("dingaling_id")}
        else:
            materialized = {(schema, "dingalings_v"): empty}
        views = {
            (schema, "email_addresses_v"): empty,
            (schema, "users_v"): empty,
            (schema, "user_tmp_v"): empty,
        }
        self._resolve_views(views, materialized)
        tables = {
            (schema, "users"): pk("user_id"),
            (schema, "dingalings"): pk("dingaling_id"),
            (schema, "email_addresses"): pk(
                "address_id", name="email_ad_pk", comment="ea pk comment"
            ),
            (schema, "comment_test"): pk("id"),
            (schema, "no_constraints"): empty,
            (schema, "local_table"): pk("id"),
            (schema, "remote_table"): pk("id"),
            (schema, "remote_table_2"): pk("id"),
            (schema, "noncol_idx_test_nopk"): empty,
            (schema, "noncol_idx_test_pk"): pk("id"),
            (schema, self.temp_table_name()): pk("id"),
        }
        if not testing.requires.reflects_pk_names.enabled:
            for val in tables.values():
                if val["name"] is not None:
                    val["name"] = mock.ANY
        res = self._resolve_kind(kind, tables, views, materialized)
        res = self._resolve_names(schema, scope, filter_names, res)
        return res

    @property
    def _required_pk_keys(self):
        return {"name", "constrained_columns"}

    def exp_fks(
        self,
        schema=None,
        scope=ObjectScope.ANY,
        kind=ObjectKind.ANY,
        filter_names=None,
    ):
        class tt:
            def __eq__(self, other):
                return (
                    other is None
                    or config.db.dialect.default_schema_name == other
                )

        def fk(
            cols,
            ref_col,
            ref_table,
            ref_schema=schema,
            name=mock.ANY,
            comment=None,
        ):
            return {
                "constrained_columns": cols,
                "referred_columns": ref_col,
                "name": name,
                "options": mock.ANY,
                "referred_schema": (
                    ref_schema if ref_schema is not None else tt()
                ),
                "referred_table": ref_table,
                "comment": comment,
            }

        materialized = {(schema, "dingalings_v"): []}
        views = {
            (schema, "email_addresses_v"): [],
            (schema, "users_v"): [],
            (schema, "user_tmp_v"): [],
        }
        self._resolve_views(views, materialized)
        tables = {
            (schema, "users"): [
                fk(["parent_user_id"], ["user_id"], "users", name="user_id_fk")
            ],
            (schema, "dingalings"): [
                fk(["id_user"], ["user_id"], "users"),
                fk(
                    ["address_id"],
                    ["address_id"],
                    "email_addresses",
                    name="zz_email_add_id_fg",
                    comment="di fk comment",
                ),
            ],
            (schema, "email_addresses"): [
                fk(["remote_user_id"], ["user_id"], "users")
            ],
            (schema, "comment_test"): [],
            (schema, "no_constraints"): [],
            (schema, "local_table"): [
                fk(
                    ["remote_id"],
                    ["id"],
                    "remote_table_2",
                    ref_schema=config.test_schema,
                )
            ],
            (schema, "remote_table"): [
                fk(["local_id"], ["id"], "local_table", ref_schema=None)
            ],
            (schema, "remote_table_2"): [],
            (schema, "noncol_idx_test_nopk"): [],
            (schema, "noncol_idx_test_pk"): [],
            (schema, self.temp_table_name()): [],
        }
        if not testing.requires.self_referential_foreign_keys.enabled:
            tables[(schema, "users")].clear()
        if not testing.requires.named_constraints.enabled:
            for vals in tables.values():
                for val in vals:
                    if val["name"] is not mock.ANY:
                        val["name"] = mock.ANY

        res = self._resolve_kind(kind, tables, views, materialized)
        res = self._resolve_names(schema, scope, filter_names, res)
        return res

    @property
    def _required_fk_keys(self):
        return {
            "name",
            "constrained_columns",
            "referred_schema",
            "referred_table",
            "referred_columns",
        }

    def exp_indexes(
        self,
        schema=None,
        scope=ObjectScope.ANY,
        kind=ObjectKind.ANY,
        filter_names=None,
    ):
        def idx(
            *cols,
            name,
            unique=False,
            column_sorting=None,
            duplicates=False,
            fk=False,
        ):
            fk_req = testing.requires.foreign_keys_reflect_as_index
            dup_req = testing.requires.unique_constraints_reflect_as_index
            sorting_expression = (
                testing.requires.reflect_indexes_with_ascdesc_as_expression
            )

            if (fk and not fk_req.enabled) or (
                duplicates and not dup_req.enabled
            ):
                return ()
            res = {
                "unique": unique,
                "column_names": list(cols),
                "name": name,
                "dialect_options": mock.ANY,
                "include_columns": [],
            }
            if column_sorting:
                res["column_sorting"] = column_sorting
                if sorting_expression.enabled:
                    res["expressions"] = orig = res["column_names"]
                    res["column_names"] = [
                        None if c in column_sorting else c for c in orig
                    ]

            if duplicates:
                res["duplicates_constraint"] = name
            return [res]

        materialized = {(schema, "dingalings_v"): []}
        views = {
            (schema, "email_addresses_v"): [],
            (schema, "users_v"): [],
            (schema, "user_tmp_v"): [],
        }
        self._resolve_views(views, materialized)
        if materialized:
            materialized[(schema, "dingalings_v")].extend(
                idx("data", name="mat_index")
            )
        tables = {
            (schema, "users"): [
                *idx("parent_user_id", name="user_id_fk", fk=True),
                *idx("user_id", "test2", "test1", name="users_all_idx"),
                *idx("test1", "test2", name="users_t_idx", unique=True),
            ],
            (schema, "dingalings"): [
                *idx("data", name=mock.ANY, unique=True, duplicates=True),
                *idx("id_user", name=mock.ANY, fk=True),
                *idx(
                    "address_id",
                    "dingaling_id",
                    name="zz_dingalings_multiple",
                    unique=True,
                    duplicates=True,
                ),
            ],
            (schema, "email_addresses"): [
                *idx("email_address", name=mock.ANY),
                *idx("remote_user_id", name=mock.ANY, fk=True),
            ],
            (schema, "comment_test"): [],
            (schema, "no_constraints"): [],
            (schema, "local_table"): [
                *idx("remote_id", name=mock.ANY, fk=True)
            ],
            (schema, "remote_table"): [
                *idx("local_id", name=mock.ANY, fk=True)
            ],
            (schema, "remote_table_2"): [],
            (schema, "noncol_idx_test_nopk"): [
                *idx(
                    "q",
                    name="noncol_idx_nopk",
                    column_sorting={"q": ("desc",)},
                )
            ],
            (schema, "noncol_idx_test_pk"): [
                *idx(
                    "q", name="noncol_idx_pk", column_sorting={"q": ("desc",)}
                )
            ],
            (schema, self.temp_table_name()): [
                *idx("foo", name="user_tmp_ix"),
                *idx(
                    "name",
                    name=f"user_tmp_uq_{config.ident}",
                    duplicates=True,
                    unique=True,
                ),
            ],
        }
        if (
            not testing.requires.indexes_with_ascdesc.enabled
            or not testing.requires.reflect_indexes_with_ascdesc.enabled
        ):
            tables[(schema, "noncol_idx_test_nopk")].clear()
            tables[(schema, "noncol_idx_test_pk")].clear()
        res = self._resolve_kind(kind, tables, views, materialized)
        res = self._resolve_names(schema, scope, filter_names, res)
        return res

    @property
    def _required_index_keys(self):
        return {"name", "column_names", "unique"}

    def exp_ucs(
        self,
        schema=None,
        scope=ObjectScope.ANY,
        kind=ObjectKind.ANY,
        filter_names=None,
        all_=False,
    ):
        def uc(
            *cols, name, duplicates_index=None, is_index=False, comment=None
        ):
            req = testing.requires.unique_index_reflect_as_unique_constraints
            if is_index and not req.enabled:
                return ()
            res = {
                "column_names": list(cols),
                "name": name,
                "comment": comment,
            }
            if duplicates_index:
                res["duplicates_index"] = duplicates_index
            return [res]

        materialized = {(schema, "dingalings_v"): []}
        views = {
            (schema, "email_addresses_v"): [],
            (schema, "users_v"): [],
            (schema, "user_tmp_v"): [],
        }
        self._resolve_views(views, materialized)
        tables = {
            (schema, "users"): [
                *uc(
                    "test1",
                    "test2",
                    name="users_t_idx",
                    duplicates_index="users_t_idx",
                    is_index=True,
                )
            ],
            (schema, "dingalings"): [
                *uc("data", name=mock.ANY, duplicates_index=mock.ANY),
                *uc(
                    "address_id",
                    "dingaling_id",
                    name="zz_dingalings_multiple",
                    duplicates_index="zz_dingalings_multiple",
                    comment="di unique comment",
                ),
            ],
            (schema, "email_addresses"): [],
            (schema, "comment_test"): [],
            (schema, "no_constraints"): [],
            (schema, "local_table"): [],
            (schema, "remote_table"): [],
            (schema, "remote_table_2"): [],
            (schema, "noncol_idx_test_nopk"): [],
            (schema, "noncol_idx_test_pk"): [],
            (schema, self.temp_table_name()): [
                *uc("name", name=f"user_tmp_uq_{config.ident}")
            ],
        }
        if all_:
            return {**materialized, **views, **tables}
        else:
            res = self._resolve_kind(kind, tables, views, materialized)
            res = self._resolve_names(schema, scope, filter_names, res)
            return res

    @property
    def _required_unique_cst_keys(self):
        return {"name", "column_names"}

    def exp_ccs(
        self,
        schema=None,
        scope=ObjectScope.ANY,
        kind=ObjectKind.ANY,
        filter_names=None,
    ):
        class tt(str):
            def __eq__(self, other):
                res = (
                    other.lower()
                    .replace("(", "")
                    .replace(")", "")
                    .replace("`", "")
                )
                return self in res

        def cc(text, name, comment=None):
            return {"sqltext": tt(text), "name": name, "comment": comment}

        # print({1: "test2 > (0)::double precision"} == {1: tt("test2 > 0")})
        # assert 0
        materialized = {(schema, "dingalings_v"): []}
        views = {
            (schema, "email_addresses_v"): [],
            (schema, "users_v"): [],
            (schema, "user_tmp_v"): [],
        }
        self._resolve_views(views, materialized)
        tables = {
            (schema, "users"): [
                cc("test2 <= 1000", mock.ANY),
                cc(
                    "test2 > 0",
                    "zz_test2_gt_zero",
                    comment="users check constraint",
                ),
            ],
            (schema, "dingalings"): [
                cc(
                    "address_id > 0 and address_id < 1000",
                    name="address_id_gt_zero",
                ),
            ],
            (schema, "email_addresses"): [],
            (schema, "comment_test"): [],
            (schema, "no_constraints"): [],
            (schema, "local_table"): [],
            (schema, "remote_table"): [],
            (schema, "remote_table_2"): [],
            (schema, "noncol_idx_test_nopk"): [],
            (schema, "noncol_idx_test_pk"): [],
            (schema, self.temp_table_name()): [],
        }
        res = self._resolve_kind(kind, tables, views, materialized)
        res = self._resolve_names(schema, scope, filter_names, res)
        return res

    @property
    def _required_cc_keys(self):
        return {"name", "sqltext"}

    @testing.requires.schema_reflection
    def test_get_schema_names(self, connection):
        insp = inspect(connection)

        is_true(testing.config.test_schema in insp.get_schema_names())

    @testing.requires.schema_reflection
    def test_has_schema(self, connection):
        insp = inspect(connection)

        is_true(insp.has_schema(testing.config.test_schema))
        is_false(insp.has_schema("sa_fake_schema_foo"))

    @testing.requires.schema_reflection
    def test_get_schema_names_w_translate_map(self, connection):
        """test #7300"""

        connection = connection.execution_options(
            schema_translate_map={
                "foo": "bar",
                BLANK_SCHEMA: testing.config.test_schema,
            }
        )
        insp = inspect(connection)

        is_true(testing.config.test_schema in insp.get_schema_names())

    @testing.requires.schema_reflection
    def test_has_schema_w_translate_map(self, connection):
        connection = connection.execution_options(
            schema_translate_map={
                "foo": "bar",
                BLANK_SCHEMA: testing.config.test_schema,
            }
        )
        insp = inspect(connection)

        is_true(insp.has_schema(testing.config.test_schema))
        is_false(insp.has_schema("sa_fake_schema_foo"))

    @testing.requires.schema_reflection
    @testing.requires.schema_create_delete
    def test_schema_cache(self, connection):
        insp = inspect(connection)

        is_false("foo_bar" in insp.get_schema_names())
        is_false(insp.has_schema("foo_bar"))
        connection.execute(DDL("CREATE SCHEMA foo_bar"))
        try:
            is_false("foo_bar" in insp.get_schema_names())
            is_false(insp.has_schema("foo_bar"))
            insp.clear_cache()
            is_true("foo_bar" in insp.get_schema_names())
            is_true(insp.has_schema("foo_bar"))
        finally:
            connection.execute(DDL("DROP SCHEMA foo_bar"))

    @testing.requires.schema_reflection
    def test_dialect_initialize(self):
        engine = engines.testing_engine()
        inspect(engine)
        assert hasattr(engine.dialect, "default_schema_name")

    @testing.requires.schema_reflection
    def test_get_default_schema_name(self, connection):
        insp = inspect(connection)
        eq_(insp.default_schema_name, connection.dialect.default_schema_name)

    @testing.combinations(
        None,
        ("foreign_key", testing.requires.foreign_key_constraint_reflection),
        argnames="order_by",
    )
    @testing.combinations(
        (True, testing.requires.schemas), False, argnames="use_schema"
    )
    def test_get_table_names(self, connection, order_by, use_schema):
        if use_schema:
            schema = config.test_schema
        else:
            schema = None

        _ignore_tables = {
            "comment_test",
            "noncol_idx_test_pk",
            "noncol_idx_test_nopk",
            "local_table",
            "remote_table",
            "remote_table_2",
            "no_constraints",
        }

        insp = inspect(connection)

        if order_by:
            tables = [
                rec[0]
                for rec in insp.get_sorted_table_and_fkc_names(schema)
                if rec[0]
            ]
        else:
            tables = insp.get_table_names(schema)
        table_names = [t for t in tables if t not in _ignore_tables]

        if order_by == "foreign_key":
            answer = ["users", "email_addresses", "dingalings"]
            eq_(table_names, answer)
        else:
            answer = ["dingalings", "email_addresses", "users"]
            eq_(sorted(table_names), answer)

    @testing.combinations(
        (True, testing.requires.schemas), False, argnames="use_schema"
    )
    def test_get_view_names(self, connection, use_schema):
        insp = inspect(connection)
        if use_schema:
            schema = config.test_schema
        else:
            schema = None
        table_names = insp.get_view_names(schema)
        if testing.requires.materialized_views.enabled:
            eq_(sorted(table_names), ["email_addresses_v", "users_v"])
            eq_(insp.get_materialized_view_names(schema), ["dingalings_v"])
        else:
            answer = ["dingalings_v", "email_addresses_v", "users_v"]
            eq_(sorted(table_names), answer)

    @testing.requires.temp_table_names
    def test_get_temp_table_names(self, connection):
        insp = inspect(connection)
        temp_table_names = insp.get_temp_table_names()
        eq_(sorted(temp_table_names), [f"user_tmp_{config.ident}"])

    @testing.requires.view_reflection
    @testing.requires.temporary_views
    def test_get_temp_view_names(self, connection):
        insp = inspect(connection)
        temp_table_names = insp.get_temp_view_names()
        eq_(sorted(temp_table_names), ["user_tmp_v"])

    @testing.requires.comment_reflection
    def test_get_comments(self, connection):
        self._test_get_comments(connection)

    @testing.requires.comment_reflection
    @testing.requires.schemas
    def test_get_comments_with_schema(self, connection):
        self._test_get_comments(connection, testing.config.test_schema)

    def _test_get_comments(self, connection, schema=None):
        insp = inspect(connection)
        exp = self.exp_comments(schema=schema)
        eq_(
            insp.get_table_comment("comment_test", schema=schema),
            exp[(schema, "comment_test")],
        )

        eq_(
            insp.get_table_comment("users", schema=schema),
            exp[(schema, "users")],
        )

        eq_(
            insp.get_table_comment("comment_test", schema=schema),
            exp[(schema, "comment_test")],
        )

        no_cst = self.tables.no_constraints.name
        eq_(
            insp.get_table_comment(no_cst, schema=schema),
            exp[(schema, no_cst)],
        )

    @testing.combinations(
        (False, False),
        (False, True, testing.requires.schemas),
        (True, False, testing.requires.view_reflection),
        (
            True,
            True,
            testing.requires.schemas + testing.requires.view_reflection,
        ),
        argnames="use_views,use_schema",
    )
    def test_get_columns(self, connection, use_views, use_schema):
        if use_schema:
            schema = config.test_schema
        else:
            schema = None

        users, addresses = (self.tables.users, self.tables.email_addresses)
        if use_views:
            table_names = ["users_v", "email_addresses_v", "dingalings_v"]
        else:
            table_names = ["users", "email_addresses"]

        insp = inspect(connection)
        for table_name, table in zip(table_names, (users, addresses)):
            schema_name = schema
            cols = insp.get_columns(table_name, schema=schema_name)
            is_true(len(cols) > 0, len(cols))

            # should be in order

            for i, col in enumerate(table.columns):
                eq_(col.name, cols[i]["name"])
                ctype = cols[i]["type"].__class__
                ctype_def = col.type
                if isinstance(ctype_def, sa.types.TypeEngine):
                    ctype_def = ctype_def.__class__

                # Oracle returns Date for DateTime.

                if testing.against("oracle") and ctype_def in (
                    sql_types.Date,
                    sql_types.DateTime,
                ):
                    ctype_def = sql_types.Date

                # assert that the desired type and return type share
                # a base within one of the generic types.

                is_true(
                    len(
                        set(ctype.__mro__)
                        .intersection(ctype_def.__mro__)
                        .intersection(
                            [
                                sql_types.Integer,
                                sql_types.Numeric,
                                sql_types.DateTime,
                                sql_types.Date,
                                sql_types.Time,
                                sql_types.String,
                                sql_types._Binary,
                            ]
                        )
                    )
                    > 0,
                    "%s(%s), %s(%s)"
                    % (col.name, col.type, cols[i]["name"], ctype),
                )

                if not col.primary_key:
                    assert cols[i]["default"] is None

        # The case of a table with no column
        # is tested below in TableNoColumnsTest

    @testing.requires.temp_table_reflection
    def test_reflect_table_temp_table(self, connection):
        table_name = self.temp_table_name()
        user_tmp = self.tables[table_name]

        reflected_user_tmp = Table(
            table_name, MetaData(), autoload_with=connection
        )
        self.assert_tables_equal(
            user_tmp, reflected_user_tmp, strict_constraints=False
        )

    @testing.requires.temp_table_reflection
    def test_get_temp_table_columns(self, connection):
        table_name = self.temp_table_name()
        user_tmp = self.tables[table_name]
        insp = inspect(connection)
        cols = insp.get_columns(table_name)
        is_true(len(cols) > 0, len(cols))

        for i, col in enumerate(user_tmp.columns):
            eq_(col.name, cols[i]["name"])

    @testing.requires.temp_table_reflection
    @testing.requires.view_column_reflection
    @testing.requires.temporary_views
    def test_get_temp_view_columns(self, connection):
        insp = inspect(connection)
        cols = insp.get_columns("user_tmp_v")
        eq_([col["name"] for col in cols], ["id", "name", "foo"])

    @testing.combinations(
        (False,), (True, testing.requires.schemas), argnames="use_schema"
    )
    @testing.requires.primary_key_constraint_reflection
    def test_get_pk_constraint(self, connection, use_schema):
        if use_schema:
            schema = testing.config.test_schema
        else:
            schema = None

        users, addresses = self.tables.users, self.tables.email_addresses
        insp = inspect(connection)
        exp = self.exp_pks(schema=schema)

        users_cons = insp.get_pk_constraint(users.name, schema=schema)
        self._check_list(
            [users_cons], [exp[(schema, users.name)]], self._required_pk_keys
        )

        addr_cons = insp.get_pk_constraint(addresses.name, schema=schema)
        exp_cols = exp[(schema, addresses.name)]["constrained_columns"]
        eq_(addr_cons["constrained_columns"], exp_cols)

        with testing.requires.reflects_pk_names.fail_if():
            eq_(addr_cons["name"], "email_ad_pk")

        no_cst = self.tables.no_constraints.name
        self._check_list(
            [insp.get_pk_constraint(no_cst, schema=schema)],
            [exp[(schema, no_cst)]],
            self._required_pk_keys,
        )

    @testing.combinations(
        (False,), (True, testing.requires.schemas), argnames="use_schema"
    )
    @testing.requires.foreign_key_constraint_reflection
    def test_get_foreign_keys(self, connection, use_schema):
        if use_schema:
            schema = config.test_schema
        else:
            schema = None

        users, addresses = (self.tables.users, self.tables.email_addresses)
        insp = inspect(connection)
        expected_schema = schema
        # users

        if testing.requires.self_referential_foreign_keys.enabled:
            users_fkeys = insp.get_foreign_keys(users.name, schema=schema)
            fkey1 = users_fkeys[0]

            with testing.requires.named_constraints.fail_if():
                eq_(fkey1["name"], "user_id_fk")

            eq_(fkey1["referred_schema"], expected_schema)
            eq_(fkey1["referred_table"], users.name)
            eq_(fkey1["referred_columns"], ["user_id"])
            eq_(fkey1["constrained_columns"], ["parent_user_id"])

        # addresses
        addr_fkeys = insp.get_foreign_keys(addresses.name, schema=schema)
        fkey1 = addr_fkeys[0]

        with testing.requires.implicitly_named_constraints.fail_if():
            is_true(fkey1["name"] is not None)

        eq_(fkey1["referred_schema"], expected_schema)
        eq_(fkey1["referred_table"], users.name)
        eq_(fkey1["referred_columns"], ["user_id"])
        eq_(fkey1["constrained_columns"], ["remote_user_id"])

        no_cst = self.tables.no_constraints.name
        eq_(insp.get_foreign_keys(no_cst, schema=schema), [])

    @testing.requires.cross_schema_fk_reflection
    @testing.requires.schemas
    def test_get_inter_schema_foreign_keys(self, connection):
        local_table, remote_table, remote_table_2 = self.tables(
            "%s.local_table" % connection.dialect.default_schema_name,
            "%s.remote_table" % testing.config.test_schema,
            "%s.remote_table_2" % testing.config.test_schema,
        )

        insp = inspect(connection)

        local_fkeys = insp.get_foreign_keys(local_table.name)
        eq_(len(local_fkeys), 1)

        fkey1 = local_fkeys[0]
        eq_(fkey1["referred_schema"], testing.config.test_schema)
        eq_(fkey1["referred_table"], remote_table_2.name)
        eq_(fkey1["referred_columns"], ["id"])
        eq_(fkey1["constrained_columns"], ["remote_id"])

        remote_fkeys = insp.get_foreign_keys(
            remote_table.name, schema=testing.config.test_schema
        )
        eq_(len(remote_fkeys), 1)

        fkey2 = remote_fkeys[0]

        is_true(
            fkey2["referred_schema"]
            in (
                None,
                connection.dialect.default_schema_name,
            )
        )
        eq_(fkey2["referred_table"], local_table.name)
        eq_(fkey2["referred_columns"], ["id"])
        eq_(fkey2["constrained_columns"], ["local_id"])

    @testing.combinations(
        (False,), (True, testing.requires.schemas), argnames="use_schema"
    )
    @testing.requires.index_reflection
    def test_get_indexes(self, connection, use_schema):
        if use_schema:
            schema = config.test_schema
        else:
            schema = None

        # The database may decide to create indexes for foreign keys, etc.
        # so there may be more indexes than expected.
        insp = inspect(connection)
        indexes = insp.get_indexes("users", schema=schema)
        exp = self.exp_indexes(schema=schema)
        self._check_list(
            indexes, exp[(schema, "users")], self._required_index_keys
        )

        no_cst = self.tables.no_constraints.name
        self._check_list(
            insp.get_indexes(no_cst, schema=schema),
            exp[(schema, no_cst)],
            self._required_index_keys,
        )

    @testing.combinations(
        ("noncol_idx_test_nopk", "noncol_idx_nopk"),
        ("noncol_idx_test_pk", "noncol_idx_pk"),
        argnames="tname,ixname",
    )
    @testing.requires.index_reflection
    @testing.requires.indexes_with_ascdesc
    @testing.requires.reflect_indexes_with_ascdesc
    def test_get_noncol_index(self, connection, tname, ixname):
        insp = inspect(connection)
        indexes = insp.get_indexes(tname)
        # reflecting an index that has "x DESC" in it as the column.
        # the DB may or may not give us "x", but make sure we get the index
        # back, it has a name, it's connected to the table.
        expected_indexes = self.exp_indexes()[(None, tname)]
        self._check_list(indexes, expected_indexes, self._required_index_keys)

        t = Table(tname, MetaData(), autoload_with=connection)
        eq_(len(t.indexes), 1)
        is_(list(t.indexes)[0].table, t)
        eq_(list(t.indexes)[0].name, ixname)

    @testing.requires.temp_table_reflection
    @testing.requires.unique_constraint_reflection
    def test_get_temp_table_unique_constraints(self, connection):
        insp = inspect(connection)
        name = self.temp_table_name()
        reflected = insp.get_unique_constraints(name)
        exp = self.exp_ucs(all_=True)[(None, name)]
        self._check_list(reflected, exp, self._required_index_keys)

    @testing.requires.temp_table_reflect_indexes
    def test_get_temp_table_indexes(self, connection):
        insp = inspect(connection)
        table_name = self.temp_table_name()
        indexes = insp.get_indexes(table_name)
        for ind in indexes:
            ind.pop("dialect_options", None)
        expected = [
            {"unique": False, "column_names": ["foo"], "name": "user_tmp_ix"}
        ]
        if testing.requires.index_reflects_included_columns.enabled:
            expected[0]["include_columns"] = []
        eq_(
            [idx for idx in indexes if idx["name"] == "user_tmp_ix"],
            expected,
        )

    @testing.combinations(
        (True, testing.requires.schemas), (False,), argnames="use_schema"
    )
    @testing.requires.unique_constraint_reflection
    def test_get_unique_constraints(self, metadata, connection, use_schema):
        # SQLite dialect needs to parse the names of the constraints
        # separately from what it gets from PRAGMA index_list(), and
        # then matches them up.  so same set of column_names in two
        # constraints will confuse it.    Perhaps we should no longer
        # bother with index_list() here since we have the whole
        # CREATE TABLE?

        if use_schema:
            schema = config.test_schema
        else:
            schema = None
        uniques = sorted(
            [
                {"name": "unique_a", "column_names": ["a"]},
                {"name": "unique_a_b_c", "column_names": ["a", "b", "c"]},
                {"name": "unique_c_a_b", "column_names": ["c", "a", "b"]},
                {"name": "unique_asc_key", "column_names": ["asc", "key"]},
                {"name": "i.have.dots", "column_names": ["b"]},
                {"name": "i have spaces", "column_names": ["c"]},
            ],
            key=operator.itemgetter("name"),
        )
        table = Table(
            "testtbl",
            metadata,
            Column("a", sa.String(20)),
            Column("b", sa.String(30)),
            Column("c", sa.Integer),
            # reserved identifiers
            Column("asc", sa.String(30)),
            Column("key", sa.String(30)),
            schema=schema,
        )
        for uc in uniques:
            table.append_constraint(
                sa.UniqueConstraint(*uc["column_names"], name=uc["name"])
            )
        table.create(connection)

        insp = inspect(connection)
        reflected = sorted(
            insp.get_unique_constraints("testtbl", schema=schema),
            key=operator.itemgetter("name"),
        )

        names_that_duplicate_index = set()

        eq_(len(uniques), len(reflected))

        for orig, refl in zip(uniques, reflected):
            # Different dialects handle duplicate index and constraints
            # differently, so ignore this flag
            dupe = refl.pop("duplicates_index", None)
            if dupe:
                names_that_duplicate_index.add(dupe)
            eq_(refl.pop("comment", None), None)
            # ignore dialect_options
            refl.pop("dialect_options", None)
            eq_(orig, refl)

        reflected_metadata = MetaData()
        reflected = Table(
            "testtbl",
            reflected_metadata,
            autoload_with=connection,
            schema=schema,
        )

        # test "deduplicates for index" logic.   MySQL and Oracle
        # "unique constraints" are actually unique indexes (with possible
        # exception of a unique that is a dupe of another one in the case
        # of Oracle).  make sure # they aren't duplicated.
        idx_names = {idx.name for idx in reflected.indexes}
        uq_names = {
            uq.name
            for uq in reflected.constraints
            if isinstance(uq, sa.UniqueConstraint)
        }.difference(["unique_c_a_b"])

        assert not idx_names.intersection(uq_names)
        if names_that_duplicate_index:
            eq_(names_that_duplicate_index, idx_names)
            eq_(uq_names, set())

        no_cst = self.tables.no_constraints.name
        eq_(insp.get_unique_constraints(no_cst, schema=schema), [])

    @testing.requires.view_reflection
    @testing.combinations(
        (False,), (True, testing.requires.schemas), argnames="use_schema"
    )
    def test_get_view_definition(self, connection, use_schema):
        if use_schema:
            schema = config.test_schema
        else:
            schema = None
        insp = inspect(connection)
        for view in ["users_v", "email_addresses_v", "dingalings_v"]:
            v = insp.get_view_definition(view, schema=schema)
            is_true(bool(v))

    @testing.requires.view_reflection
    def test_get_view_definition_does_not_exist(self, connection):
        insp = inspect(connection)
        with expect_raises(NoSuchTableError):
            insp.get_view_definition("view_does_not_exist")
        with expect_raises(NoSuchTableError):
            insp.get_view_definition("users")  # a table

    @testing.requires.table_reflection
    def test_autoincrement_col(self, connection):
        """test that 'autoincrement' is reflected according to sqla's policy.

        Don't mark this test as unsupported for any backend !

        (technically it fails with MySQL InnoDB since "id" comes before "id2")

        A backend is better off not returning "autoincrement" at all,
        instead of potentially returning "False" for an auto-incrementing
        primary key column.

        """

        insp = inspect(connection)

        for tname, cname in [
            ("users", "user_id"),
            ("email_addresses", "address_id"),
            ("dingalings", "dingaling_id"),
        ]:
            cols = insp.get_columns(tname)
            id_ = {c["name"]: c for c in cols}[cname]
            assert id_.get("autoincrement", True)

    @testing.combinations(
        (True, testing.requires.schemas), (False,), argnames="use_schema"
    )
    def test_get_table_options(self, use_schema):
        insp = inspect(config.db)
        schema = config.test_schema if use_schema else None

        if testing.requires.reflect_table_options.enabled:
            res = insp.get_table_options("users", schema=schema)
            is_true(isinstance(res, dict))
            # NOTE: can't really create a table with no option
            res = insp.get_table_options("no_constraints", schema=schema)
            is_true(isinstance(res, dict))
        else:
            with expect_raises(NotImplementedError):
                insp.get_table_options("users", schema=schema)

    @testing.combinations((True, testing.requires.schemas), False)
    def test_multi_get_table_options(self, use_schema):
        insp = inspect(config.db)
        if testing.requires.reflect_table_options.enabled:
            schema = config.test_schema if use_schema else None
            res = insp.get_multi_table_options(schema=schema)

            exp = {
                (schema, table): insp.get_table_options(table, schema=schema)
                for table in insp.get_table_names(schema=schema)
            }
            eq_(res, exp)
        else:
            with expect_raises(NotImplementedError):
                insp.get_multi_table_options()

    @testing.fixture
    def get_multi_exp(self, connection):
        def provide_fixture(
            schema, scope, kind, use_filter, single_reflect_fn, exp_method
        ):
            insp = inspect(connection)
            # call the reflection function at least once to avoid
            # "Unexpected success" errors if the result is actually empty
            # and NotImplementedError is not raised
            single_reflect_fn(insp, "email_addresses")
            kw = {"scope": scope, "kind": kind}
            if schema:
                schema = schema()

            filter_names = []

            if ObjectKind.TABLE in kind:
                filter_names.extend(
                    ["comment_test", "users", "does-not-exist"]
                )
            if ObjectKind.VIEW in kind:
                filter_names.extend(["email_addresses_v", "does-not-exist"])
            if ObjectKind.MATERIALIZED_VIEW in kind:
                filter_names.extend(["dingalings_v", "does-not-exist"])

            if schema:
                kw["schema"] = schema
            if use_filter:
                kw["filter_names"] = filter_names

            exp = exp_method(
                schema=schema,
                scope=scope,
                kind=kind,
                filter_names=kw.get("filter_names"),
            )
            kws = [kw]
            if scope == ObjectScope.DEFAULT:
                nkw = kw.copy()
                nkw.pop("scope")
                kws.append(nkw)
            if kind == ObjectKind.TABLE:
                nkw = kw.copy()
                nkw.pop("kind")
                kws.append(nkw)

            return inspect(connection), kws, exp

        return provide_fixture

    @testing.requires.reflect_table_options
    @_multi_combination
    def test_multi_get_table_options_tables(
        self, get_multi_exp, schema, scope, kind, use_filter
    ):
        insp, kws, exp = get_multi_exp(
            schema,
            scope,
            kind,
            use_filter,
            Inspector.get_table_options,
            self.exp_options,
        )
        for kw in kws:
            insp.clear_cache()
            result = insp.get_multi_table_options(**kw)
            eq_(result, exp)

    @testing.requires.comment_reflection
    @_multi_combination
    def test_get_multi_table_comment(
        self, get_multi_exp, schema, scope, kind, use_filter
    ):
        insp, kws, exp = get_multi_exp(
            schema,
            scope,
            kind,
            use_filter,
            Inspector.get_table_comment,
            self.exp_comments,
        )
        for kw in kws:
            insp.clear_cache()
            eq_(insp.get_multi_table_comment(**kw), exp)

    def _check_expressions(self, result, exp, err_msg):
        def _clean(text: str):
            return re.sub(r"['\" ]", "", text).lower()

        if isinstance(exp, dict):
            eq_({_clean(e): v for e, v in result.items()}, exp, err_msg)
        else:
            eq_([_clean(e) for e in result], exp, err_msg)

    def _check_list(self, result, exp, req_keys=None, msg=None):
        if req_keys is None:
            eq_(result, exp, msg)
        else:
            eq_(len(result), len(exp), msg)
            for r, e in zip(result, exp):
                for k in set(r) | set(e):
                    if k in req_keys or (k in r and k in e):
                        err_msg = f"{msg} - {k} - {r}"
                        if k in ("expressions", "column_sorting"):
                            self._check_expressions(r[k], e[k], err_msg)
                        else:
                            eq_(r[k], e[k], err_msg)

    def _check_table_dict(self, result, exp, req_keys=None, make_lists=False):
        eq_(set(result.keys()), set(exp.keys()))
        for k in result:
            r, e = result[k], exp[k]
            if make_lists:
                r, e = [r], [e]
            self._check_list(r, e, req_keys, k)

    @_multi_combination
    def test_get_multi_columns(
        self, get_multi_exp, schema, scope, kind, use_filter
    ):
        insp, kws, exp = get_multi_exp(
            schema,
            scope,
            kind,
            use_filter,
            Inspector.get_columns,
            self.exp_columns,
        )

        for kw in kws:
            insp.clear_cache()
            result = insp.get_multi_columns(**kw)
            self._check_table_dict(result, exp, self._required_column_keys)

    @testing.requires.primary_key_constraint_reflection
    @_multi_combination
    def test_get_multi_pk_constraint(
        self, get_multi_exp, schema, scope, kind, use_filter
    ):
        insp, kws, exp = get_multi_exp(
            schema,
            scope,
            kind,
            use_filter,
            Inspector.get_pk_constraint,
            self.exp_pks,
        )
        for kw in kws:
            insp.clear_cache()
            result = insp.get_multi_pk_constraint(**kw)
            self._check_table_dict(
                result, exp, self._required_pk_keys, make_lists=True
            )

    def _adjust_sort(self, result, expected, key):
        if not testing.requires.implicitly_named_constraints.enabled:
            for obj in [result, expected]:
                for val in obj.values():
                    if len(val) > 1 and any(
                        v.get("name") in (None, mock.ANY) for v in val
                    ):
                        val.sort(key=key)

    @testing.requires.foreign_key_constraint_reflection
    @_multi_combination
    def test_get_multi_foreign_keys(
        self, get_multi_exp, schema, scope, kind, use_filter
    ):
        insp, kws, exp = get_multi_exp(
            schema,
            scope,
            kind,
            use_filter,
            Inspector.get_foreign_keys,
            self.exp_fks,
        )
        for kw in kws:
            insp.clear_cache()
            result = insp.get_multi_foreign_keys(**kw)
            self._adjust_sort(
                result, exp, lambda d: tuple(d["constrained_columns"])
            )
            self._check_table_dict(result, exp, self._required_fk_keys)

    @testing.requires.index_reflection
    @_multi_combination
    def test_get_multi_indexes(
        self, get_multi_exp, schema, scope, kind, use_filter
    ):
        insp, kws, exp = get_multi_exp(
            schema,
            scope,
            kind,
            use_filter,
            Inspector.get_indexes,
            self.exp_indexes,
        )
        for kw in kws:
            insp.clear_cache()
            result = insp.get_multi_indexes(**kw)
            self._check_table_dict(result, exp, self._required_index_keys)

    @testing.requires.unique_constraint_reflection
    @_multi_combination
    def test_get_multi_unique_constraints(
        self, get_multi_exp, schema, scope, kind, use_filter
    ):
        insp, kws, exp = get_multi_exp(
            schema,
            scope,
            kind,
            use_filter,
            Inspector.get_unique_constraints,
            self.exp_ucs,
        )
        for kw in kws:
            insp.clear_cache()
            result = insp.get_multi_unique_constraints(**kw)
            self._adjust_sort(result, exp, lambda d: tuple(d["column_names"]))
            self._check_table_dict(result, exp, self._required_unique_cst_keys)

    @testing.requires.check_constraint_reflection
    @_multi_combination
    def test_get_multi_check_constraints(
        self, get_multi_exp, schema, scope, kind, use_filter
    ):
        insp, kws, exp = get_multi_exp(
            schema,
            scope,
            kind,
            use_filter,
            Inspector.get_check_constraints,
            self.exp_ccs,
        )
        for kw in kws:
            insp.clear_cache()
            result = insp.get_multi_check_constraints(**kw)
            self._adjust_sort(result, exp, lambda d: tuple(d["sqltext"]))
            self._check_table_dict(result, exp, self._required_cc_keys)

    @testing.combinations(
        ("get_table_options", testing.requires.reflect_table_options),
        "get_columns",
        (
            "get_pk_constraint",
            testing.requires.primary_key_constraint_reflection,
        ),
        (
            "get_foreign_keys",
            testing.requires.foreign_key_constraint_reflection,
        ),
        ("get_indexes", testing.requires.index_reflection),
        (
            "get_unique_constraints",
            testing.requires.unique_constraint_reflection,
        ),
        (
            "get_check_constraints",
            testing.requires.check_constraint_reflection,
        ),
        ("get_table_comment", testing.requires.comment_reflection),
        argnames="method",
    )
    def test_not_existing_table(self, method, connection):
        insp = inspect(connection)
        meth = getattr(insp, method)
        with expect_raises(NoSuchTableError):
            meth("table_does_not_exists")

    def test_unreflectable(self, connection):
        mc = Inspector.get_multi_columns

        def patched(*a, **k):
            ur = k.setdefault("unreflectable", {})
            ur[(None, "some_table")] = UnreflectableTableError("err")
            return mc(*a, **k)

        with mock.patch.object(Inspector, "get_multi_columns", patched):
            with expect_raises_message(UnreflectableTableError, "err"):
                inspect(connection).reflect_table(
                    Table("some_table", MetaData()), None
                )

    @testing.combinations(True, False, argnames="use_schema")
    @testing.combinations(
        (True, testing.requires.views), False, argnames="views"
    )
    def test_metadata(self, connection, use_schema, views):
        m = MetaData()
        schema = config.test_schema if use_schema else None
        m.reflect(connection, schema=schema, views=views, resolve_fks=False)

        insp = inspect(connection)
        tables = insp.get_table_names(schema)
        if views:
            tables += insp.get_view_names(schema)
            try:
                tables += insp.get_materialized_view_names(schema)
            except NotImplementedError:
                pass
        if schema:
            tables = [f"{schema}.{t}" for t in tables]
        eq_(sorted(m.tables), sorted(tables))

    @testing.requires.comment_reflection
    def test_comments_unicode(self, connection, metadata):
        Table(
            "unicode_comments",
            metadata,
            Column("unicode", Integer, comment="é試蛇ẟΩ"),
            Column("emoji", Integer, comment="☁️✨"),
            comment="試蛇ẟΩ✨",
        )

        metadata.create_all(connection)

        insp = inspect(connection)
        tc = insp.get_table_comment("unicode_comments")
        eq_(tc, {"text": "試蛇ẟΩ✨"})

        cols = insp.get_columns("unicode_comments")
        value = {c["name"]: c["comment"] for c in cols}
        exp = {"unicode": "é試蛇ẟΩ", "emoji": "☁️✨"}
        eq_(value, exp)

    @testing.requires.comment_reflection_full_unicode
    def test_comments_unicode_full(self, connection, metadata):
        Table(
            "unicode_comments",
            metadata,
            Column("emoji", Integer, comment="🐍🧙🝝🧙‍♂️🧙‍♀️"),
            comment="🎩🁰🝑🤷‍♀️🤷‍♂️",
        )

        metadata.create_all(connection)

        insp = inspect(connection)
        tc = insp.get_table_comment("unicode_comments")
        eq_(tc, {"text": "🎩🁰🝑🤷‍♀️🤷‍♂️"})
        c = insp.get_columns("unicode_comments")[0]
        eq_({c["name"]: c["comment"]}, {"emoji": "🐍🧙🝝🧙‍♂️🧙‍♀️"})


class TableNoColumnsTest(fixtures.TestBase):
    __requires__ = ("reflect_tables_no_columns",)
    __backend__ = True

    @testing.fixture
    def table_no_columns(self, connection, metadata):
        Table("empty", metadata)
        metadata.create_all(connection)

    @testing.fixture
    def view_no_columns(self, connection, metadata):
        Table("empty", metadata)
        event.listen(
            metadata,
            "after_create",
            DDL("CREATE VIEW empty_v AS SELECT * FROM empty"),
        )

        # for transactional DDL the transaction is rolled back before this
        # drop statement is invoked
        event.listen(
            metadata, "before_drop", DDL("DROP VIEW IF EXISTS empty_v")
        )
        metadata.create_all(connection)

    def test_reflect_table_no_columns(self, connection, table_no_columns):
        t2 = Table("empty", MetaData(), autoload_with=connection)
        eq_(list(t2.c), [])

    def test_get_columns_table_no_columns(self, connection, table_no_columns):
        insp = inspect(connection)
        eq_(insp.get_columns("empty"), [])
        multi = insp.get_multi_columns()
        eq_(multi, {(None, "empty"): []})

    def test_reflect_incl_table_no_columns(self, connection, table_no_columns):
        m = MetaData()
        m.reflect(connection)
        assert set(m.tables).intersection(["empty"])

    @testing.requires.views
    def test_reflect_view_no_columns(self, connection, view_no_columns):
        t2 = Table("empty_v", MetaData(), autoload_with=connection)
        eq_(list(t2.c), [])

    @testing.requires.views
    def test_get_columns_view_no_columns(self, connection, view_no_columns):
        insp = inspect(connection)
        eq_(insp.get_columns("empty_v"), [])
        multi = insp.get_multi_columns(kind=ObjectKind.VIEW)
        eq_(multi, {(None, "empty_v"): []})


class ComponentReflectionTestExtra(ComparesIndexes, fixtures.TestBase):
    __backend__ = True

    @testing.fixture(params=[True, False])
    def use_schema_fixture(self, request):
        if request.param:
            return config.test_schema
        else:
            return None

    @testing.fixture()
    def inspect_for_table(self, metadata, connection, use_schema_fixture):
        @contextlib.contextmanager
        def go(tablename):
            yield use_schema_fixture, inspect(connection)

            metadata.create_all(connection)

        return go

    def ck_eq(self, reflected, expected):
        # trying to minimize effect of quoting, parenthesis, etc.
        # may need to add more to this as new dialects get CHECK
        # constraint reflection support
        def normalize(sqltext):
            return " ".join(
                re.findall(r"and|\d|=|a|b|c|or|<|>", sqltext.lower(), re.I)
            )

        reflected = sorted(
            [
                {"name": item["name"], "sqltext": normalize(item["sqltext"])}
                for item in reflected
            ],
            key=lambda item: (item["sqltext"]),
        )

        expected = sorted(
            expected,
            key=lambda item: (item["sqltext"]),
        )
        eq_(reflected, expected)

    @testing.requires.check_constraint_reflection
    def test_check_constraint_no_constraint(self, metadata, inspect_for_table):
        with inspect_for_table("no_constraints") as (schema, inspector):
            Table(
                "no_constraints",
                metadata,
                Column("data", sa.String(20)),
                schema=schema,
            )

        self.ck_eq(
            inspector.get_check_constraints("no_constraints", schema=schema),
            [],
        )

    @testing.requires.inline_check_constraint_reflection
    @testing.combinations(
        "my_inline", "MyInline", None, argnames="constraint_name"
    )
    def test_check_constraint_inline(
        self, metadata, inspect_for_table, constraint_name
    ):

        with inspect_for_table("sa_cc") as (schema, inspector):
            Table(
                "sa_cc",
                metadata,
                Column("id", Integer(), primary_key=True),
                Column(
                    "a",
                    Integer(),
                    sa.CheckConstraint(
                        "a > 1 AND a < 5", name=constraint_name
                    ),
                ),
                Column("data", String(50)),
                schema=schema,
            )

        reflected = inspector.get_check_constraints("sa_cc", schema=schema)

        self.ck_eq(
            reflected,
            [
                {
                    "name": constraint_name or mock.ANY,
                    "sqltext": "a > 1 and a < 5",
                },
            ],
        )

    @testing.requires.check_constraint_reflection
    @testing.combinations(
        "my_ck_const", "MyCkConst", None, argnames="constraint_name"
    )
    def test_check_constraint_standalone(
        self, metadata, inspect_for_table, constraint_name
    ):
        with inspect_for_table("sa_cc") as (schema, inspector):
            Table(
                "sa_cc",
                metadata,
                Column("a", Integer()),
                sa.CheckConstraint(
                    "a = 1 OR (a > 2 AND a < 5)", name=constraint_name
                ),
                schema=schema,
            )

        reflected = inspector.get_check_constraints("sa_cc", schema=schema)

        self.ck_eq(
            reflected,
            [
                {
                    "name": constraint_name or mock.ANY,
                    "sqltext": "a = 1 or a > 2 and a < 5",
                },
            ],
        )

    @testing.requires.inline_check_constraint_reflection
    def test_check_constraint_mixed(self, metadata, inspect_for_table):
        with inspect_for_table("sa_cc") as (schema, inspector):
            Table(
                "sa_cc",
                metadata,
                Column("id", Integer(), primary_key=True),
                Column("a", Integer(), sa.CheckConstraint("a > 1 AND a < 5")),
                Column(
                    "b",
                    Integer(),
                    sa.CheckConstraint("b > 1 AND b < 5", name="my_inline"),
                ),
                Column("c", Integer()),
                Column("data", String(50)),
                sa.UniqueConstraint("data", name="some_uq"),
                sa.CheckConstraint("c > 1 AND c < 5", name="cc1"),
                sa.UniqueConstraint("c", name="some_c_uq"),
                schema=schema,
            )

        reflected = inspector.get_check_constraints("sa_cc", schema=schema)

        self.ck_eq(
            reflected,
            [
                {"name": "cc1", "sqltext": "c > 1 and c < 5"},
                {"name": "my_inline", "sqltext": "b > 1 and b < 5"},
                {"name": mock.ANY, "sqltext": "a > 1 and a < 5"},
            ],
        )

    @testing.requires.indexes_with_expressions
    def test_reflect_expression_based_indexes(self, metadata, connection):
        t = Table(
            "t",
            metadata,
            Column("x", String(30)),
            Column("y", String(30)),
            Column("z", String(30)),
        )

        Index("t_idx", func.lower(t.c.x), t.c.z, func.lower(t.c.y))
        long_str = "long string " * 100
        Index("t_idx_long", func.coalesce(t.c.x, long_str))
        Index("t_idx_2", t.c.x)

        metadata.create_all(connection)

        insp = inspect(connection)

        expected = [
            {
                "name": "t_idx_2",
                "column_names": ["x"],
                "unique": False,
                "dialect_options": {},
            }
        ]

        def completeIndex(entry):
            if testing.requires.index_reflects_included_columns.enabled:
                entry["include_columns"] = []
                entry["dialect_options"] = {
                    f"{connection.engine.name}_include": []
                }
            else:
                entry.setdefault("dialect_options", {})

        completeIndex(expected[0])

        class lower_index_str(str):
            def __eq__(self, other):
                ol = other.lower()
                # test that lower and x or y are in the string
                return "lower" in ol and ("x" in ol or "y" in ol)

        class coalesce_index_str(str):
            def __eq__(self, other):
                # test that coalesce and the string is in other
                return "coalesce" in other.lower() and long_str in other

        if testing.requires.reflect_indexes_with_expressions.enabled:
            expr_index = {
                "name": "t_idx",
                "column_names": [None, "z", None],
                "expressions": [
                    lower_index_str("lower(x)"),
                    "z",
                    lower_index_str("lower(y)"),
                ],
                "unique": False,
            }
            completeIndex(expr_index)
            expected.insert(0, expr_index)

            expr_index_long = {
                "name": "t_idx_long",
                "column_names": [None],
                "expressions": [
                    coalesce_index_str(f"coalesce(x, '{long_str}')")
                ],
                "unique": False,
            }
            completeIndex(expr_index_long)
            expected.append(expr_index_long)

            eq_(insp.get_indexes("t"), expected)
            m2 = MetaData()
            t2 = Table("t", m2, autoload_with=connection)
        else:
            with expect_warnings(
                "Skipped unsupported reflection of expression-based "
                "index t_idx"
            ):
                eq_(insp.get_indexes("t"), expected)
                m2 = MetaData()
                t2 = Table("t", m2, autoload_with=connection)

        self.compare_table_index_with_expected(
            t2, expected, connection.engine.name
        )

    @testing.requires.index_reflects_included_columns
    def test_reflect_covering_index(self, metadata, connection):
        t = Table(
            "t",
            metadata,
            Column("x", String(30)),
            Column("y", String(30)),
        )
        idx = Index("t_idx", t.c.x)
        idx.dialect_options[connection.engine.name]["include"] = ["y"]

        metadata.create_all(connection)

        insp = inspect(connection)

        get_indexes = insp.get_indexes("t")
        eq_(
            get_indexes,
            [
                {
                    "name": "t_idx",
                    "column_names": ["x"],
                    "include_columns": ["y"],
                    "unique": False,
                    "dialect_options": mock.ANY,
                }
            ],
        )
        eq_(
            get_indexes[0]["dialect_options"][
                "%s_include" % connection.engine.name
            ],
            ["y"],
        )

        t2 = Table("t", MetaData(), autoload_with=connection)
        eq_(
            list(t2.indexes)[0].dialect_options[connection.engine.name][
                "include"
            ],
            ["y"],
        )

    def _type_round_trip(self, connection, metadata, *types):
        t = Table(
            "t",
            metadata,
            *[Column("t%d" % i, type_) for i, type_ in enumerate(types)],
        )
        t.create(connection)

        return [c["type"] for c in inspect(connection).get_columns("t")]

    @testing.requires.table_reflection
    def test_numeric_reflection(self, connection, metadata):
        for typ in self._type_round_trip(
            connection, metadata, sql_types.Numeric(18, 5)
        ):
            assert isinstance(typ, sql_types.Numeric)
            eq_(typ.precision, 18)
            eq_(typ.scale, 5)

    @testing.requires.table_reflection
    def test_varchar_reflection(self, connection, metadata):
        typ = self._type_round_trip(
            connection, metadata, sql_types.String(52)
        )[0]
        assert isinstance(typ, sql_types.String)
        eq_(typ.length, 52)

    @testing.requires.table_reflection
    def test_nullable_reflection(self, connection, metadata):
        t = Table(
            "t",
            metadata,
            Column("a", Integer, nullable=True),
            Column("b", Integer, nullable=False),
        )
        t.create(connection)
        eq_(
            {
                col["name"]: col["nullable"]
                for col in inspect(connection).get_columns("t")
            },
            {"a": True, "b": False},
        )

    @testing.combinations(
        (
            None,
            "CASCADE",
            None,
            testing.requires.foreign_key_constraint_option_reflection_ondelete,
        ),
        (
            None,
            None,
            "SET NULL",
            testing.requires.foreign_key_constraint_option_reflection_onupdate,
        ),
        (
            {},
            None,
            "NO ACTION",
            testing.requires.foreign_key_constraint_option_reflection_onupdate,
        ),
        (
            {},
            "NO ACTION",
            None,
            testing.requires.fk_constraint_option_reflection_ondelete_noaction,
        ),
        (
            None,
            None,
            "RESTRICT",
            testing.requires.fk_constraint_option_reflection_onupdate_restrict,
        ),
        (
            None,
            "RESTRICT",
            None,
            testing.requires.fk_constraint_option_reflection_ondelete_restrict,
        ),
        argnames="expected,ondelete,onupdate",
    )
    def test_get_foreign_key_options(
        self, connection, metadata, expected, ondelete, onupdate
    ):
        options = {}
        if ondelete:
            options["ondelete"] = ondelete
        if onupdate:
            options["onupdate"] = onupdate

        if expected is None:
            expected = options

        Table(
            "x",
            metadata,
            Column("id", Integer, primary_key=True),
            test_needs_fk=True,
        )

        Table(
            "table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("x_id", Integer, ForeignKey("x.id", name="xid")),
            Column("test", String(10)),
            test_needs_fk=True,
        )

        Table(
            "user",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("name", String(50), nullable=False),
            Column("tid", Integer),
            sa.ForeignKeyConstraint(
                ["tid"], ["table.id"], name="myfk", **options
            ),
            test_needs_fk=True,
        )

        metadata.create_all(connection)

        insp = inspect(connection)

        # test 'options' is always present for a backend
        # that can reflect these, since alembic looks for this
        opts = insp.get_foreign_keys("table")[0]["options"]

        eq_({k: opts[k] for k in opts if opts[k]}, {})

        opts = insp.get_foreign_keys("user")[0]["options"]
        eq_(opts, expected)
        # eq_(dict((k, opts[k]) for k in opts if opts[k]), expected)

    @testing.combinations(
        (Integer, sa.text("10"), r"'?10'?"),
        (Integer, "10", r"'?10'?"),
        (Boolean, sa.true(), r"1|true"),
        (
            Integer,
            sa.text("3 + 5"),
            r"3\+5",
            testing.requires.expression_server_defaults,
        ),
        (
            Integer,
            sa.text("(3 * 5)"),
            r"3\*5",
            testing.requires.expression_server_defaults,
        ),
        (DateTime, func.now(), r"current_timestamp|now|getdate"),
        (
            Integer,
            sa.literal_column("3") + sa.literal_column("5"),
            r"3\+5",
            testing.requires.expression_server_defaults,
        ),
        argnames="datatype, default, expected_reg",
    )
    @testing.requires.server_defaults
    def test_server_defaults(
        self, metadata, connection, datatype, default, expected_reg
    ):
        t = Table(
            "t",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("thecol", datatype, server_default=default),
        )
        t.create(connection)

        reflected = inspect(connection).get_columns("t")[1]["default"]
        reflected_sanitized = re.sub(r"[\(\) \']", "", reflected)
        eq_regex(reflected_sanitized, expected_reg, flags=re.IGNORECASE)


class NormalizedNameTest(fixtures.TablesTest):
    __requires__ = ("denormalized_names",)
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        Table(
            quoted_name("t1", quote=True),
            metadata,
            Column("id", Integer, primary_key=True),
        )
        Table(
            quoted_name("t2", quote=True),
            metadata,
            Column("id", Integer, primary_key=True),
            Column("t1id", ForeignKey("t1.id")),
        )

    def test_reflect_lowercase_forced_tables(self):
        m2 = MetaData()
        t2_ref = Table(
            quoted_name("t2", quote=True), m2, autoload_with=config.db
        )
        t1_ref = m2.tables["t1"]
        assert t2_ref.c.t1id.references(t1_ref.c.id)

        m3 = MetaData()
        m3.reflect(
            config.db, only=lambda name, m: name.lower() in ("t1", "t2")
        )
        assert m3.tables["t2"].c.t1id.references(m3.tables["t1"].c.id)

    def test_get_table_names(self):
        tablenames = [
            t
            for t in inspect(config.db).get_table_names()
            if t.lower() in ("t1", "t2")
        ]

        eq_(tablenames[0].upper(), tablenames[0].lower())
        eq_(tablenames[1].upper(), tablenames[1].lower())


class ComputedReflectionTest(fixtures.ComputedReflectionFixtureTest):
    def test_computed_col_default_not_set(self):
        insp = inspect(config.db)

        cols = insp.get_columns("computed_default_table")
        col_data = {c["name"]: c for c in cols}
        is_true("42" in col_data["with_default"]["default"])
        is_(col_data["normal"]["default"], None)
        is_(col_data["computed_col"]["default"], None)

    def test_get_column_returns_computed(self):
        insp = inspect(config.db)

        cols = insp.get_columns("computed_default_table")
        data = {c["name"]: c for c in cols}
        for key in ("id", "normal", "with_default"):
            is_true("computed" not in data[key])
        compData = data["computed_col"]
        is_true("computed" in compData)
        is_true("sqltext" in compData["computed"])
        eq_(self.normalize(compData["computed"]["sqltext"]), "normal+42")
        eq_(
            "persisted" in compData["computed"],
            testing.requires.computed_columns_reflect_persisted.enabled,
        )
        if testing.requires.computed_columns_reflect_persisted.enabled:
            eq_(
                compData["computed"]["persisted"],
                testing.requires.computed_columns_default_persisted.enabled,
            )

    def check_column(self, data, column, sqltext, persisted):
        is_true("computed" in data[column])
        compData = data[column]["computed"]
        eq_(self.normalize(compData["sqltext"]), sqltext)
        if testing.requires.computed_columns_reflect_persisted.enabled:
            is_true("persisted" in compData)
            is_(compData["persisted"], persisted)

    def test_get_column_returns_persisted(self):
        insp = inspect(config.db)

        cols = insp.get_columns("computed_column_table")
        data = {c["name"]: c for c in cols}

        self.check_column(
            data,
            "computed_no_flag",
            "normal+42",
            testing.requires.computed_columns_default_persisted.enabled,
        )
        if testing.requires.computed_columns_virtual.enabled:
            self.check_column(
                data,
                "computed_virtual",
                "normal+2",
                False,
            )
        if testing.requires.computed_columns_stored.enabled:
            self.check_column(
                data,
                "computed_stored",
                "normal-42",
                True,
            )

    @testing.requires.schemas
    def test_get_column_returns_persisted_with_schema(self):
        insp = inspect(config.db)

        cols = insp.get_columns(
            "computed_column_table", schema=config.test_schema
        )
        data = {c["name"]: c for c in cols}

        self.check_column(
            data,
            "computed_no_flag",
            "normal/42",
            testing.requires.computed_columns_default_persisted.enabled,
        )
        if testing.requires.computed_columns_virtual.enabled:
            self.check_column(
                data,
                "computed_virtual",
                "normal/2",
                False,
            )
        if testing.requires.computed_columns_stored.enabled:
            self.check_column(
                data,
                "computed_stored",
                "normal*42",
                True,
            )


class IdentityReflectionTest(fixtures.TablesTest):
    run_inserts = run_deletes = None

    __backend__ = True
    __requires__ = ("identity_columns", "table_reflection")

    @classmethod
    def define_tables(cls, metadata):
        Table(
            "t1",
            metadata,
            Column("normal", Integer),
            Column("id1", Integer, Identity()),
        )
        Table(
            "t2",
            metadata,
            Column(
                "id2",
                Integer,
                Identity(
                    always=True,
                    start=2,
                    increment=3,
                    minvalue=-2,
                    maxvalue=42,
                    cycle=True,
                    cache=4,
                ),
            ),
        )
        if testing.requires.schemas.enabled:
            Table(
                "t1",
                metadata,
                Column("normal", Integer),
                Column("id1", Integer, Identity(always=True, start=20)),
                schema=config.test_schema,
            )

    def check(self, value, exp, approx):
        if testing.requires.identity_columns_standard.enabled:
            common_keys = (
                "always",
                "start",
                "increment",
                "minvalue",
                "maxvalue",
                "cycle",
                "cache",
            )
            for k in list(value):
                if k not in common_keys:
                    value.pop(k)
            if approx:
                eq_(len(value), len(exp))
                for k in value:
                    if k == "minvalue":
                        is_true(value[k] <= exp[k])
                    elif k in {"maxvalue", "cache"}:
                        is_true(value[k] >= exp[k])
                    else:
                        eq_(value[k], exp[k], k)
            else:
                eq_(value, exp)
        else:
            eq_(value["start"], exp["start"])
            eq_(value["increment"], exp["increment"])

    def test_reflect_identity(self):
        insp = inspect(config.db)

        cols = insp.get_columns("t1") + insp.get_columns("t2")
        for col in cols:
            if col["name"] == "normal":
                is_false("identity" in col)
            elif col["name"] == "id1":
                if "autoincrement" in col:
                    is_true(col["autoincrement"])
                eq_(col["default"], None)
                is_true("identity" in col)
                self.check(
                    col["identity"],
                    dict(
                        always=False,
                        start=1,
                        increment=1,
                        minvalue=1,
                        maxvalue=2147483647,
                        cycle=False,
                        cache=1,
                    ),
                    approx=True,
                )
            elif col["name"] == "id2":
                if "autoincrement" in col:
                    is_true(col["autoincrement"])
                eq_(col["default"], None)
                is_true("identity" in col)
                self.check(
                    col["identity"],
                    dict(
                        always=True,
                        start=2,
                        increment=3,
                        minvalue=-2,
                        maxvalue=42,
                        cycle=True,
                        cache=4,
                    ),
                    approx=False,
                )

    @testing.requires.schemas
    def test_reflect_identity_schema(self):
        insp = inspect(config.db)

        cols = insp.get_columns("t1", schema=config.test_schema)
        for col in cols:
            if col["name"] == "normal":
                is_false("identity" in col)
            elif col["name"] == "id1":
                if "autoincrement" in col:
                    is_true(col["autoincrement"])
                eq_(col["default"], None)
                is_true("identity" in col)
                self.check(
                    col["identity"],
                    dict(
                        always=True,
                        start=20,
                        increment=1,
                        minvalue=1,
                        maxvalue=2147483647,
                        cycle=False,
                        cache=1,
                    ),
                    approx=True,
                )


class CompositeKeyReflectionTest(fixtures.TablesTest):
    __backend__ = True

    @classmethod
    def define_tables(cls, metadata):
        tb1 = Table(
            "tb1",
            metadata,
            Column("id", Integer),
            Column("attr", Integer),
            Column("name", sql_types.VARCHAR(20)),
            sa.PrimaryKeyConstraint("name", "id", "attr", name="pk_tb1"),
            schema=None,
            test_needs_fk=True,
        )
        Table(
            "tb2",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("pid", Integer),
            Column("pattr", Integer),
            Column("pname", sql_types.VARCHAR(20)),
            sa.ForeignKeyConstraint(
                ["pname", "pid", "pattr"],
                [tb1.c.name, tb1.c.id, tb1.c.attr],
                name="fk_tb1_name_id_attr",
            ),
            schema=None,
            test_needs_fk=True,
        )

    @testing.requires.primary_key_constraint_reflection
    def test_pk_column_order(self, connection):
        # test for issue #5661
        insp = inspect(connection)
        primary_key = insp.get_pk_constraint(self.tables.tb1.name)
        eq_(primary_key.get("constrained_columns"), ["name", "id", "attr"])

    @testing.requires.foreign_key_constraint_reflection
    def test_fk_column_order(self, connection):
        # test for issue #5661
        insp = inspect(connection)
        foreign_keys = insp.get_foreign_keys(self.tables.tb2.name)
        eq_(len(foreign_keys), 1)
        fkey1 = foreign_keys[0]
        eq_(fkey1.get("referred_columns"), ["name", "id", "attr"])
        eq_(fkey1.get("constrained_columns"), ["pname", "pid", "pattr"])


__all__ = (
    "ComponentReflectionTest",
    "ComponentReflectionTestExtra",
    "TableNoColumnsTest",
    "QuotedNameArgumentTest",
    "BizarroCharacterFKResolutionTest",
    "HasTableTest",
    "HasIndexTest",
    "NormalizedNameTest",
    "ComputedReflectionTest",
    "IdentityReflectionTest",
    "CompositeKeyReflectionTest",
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\events.py ===
# orm/events.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""ORM event interfaces.

"""
from __future__ import annotations

from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generic
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from . import instrumentation
from . import interfaces
from . import mapperlib
from .attributes import QueryableAttribute
from .base import _mapper_or_none
from .base import NO_KEY
from .instrumentation import ClassManager
from .instrumentation import InstrumentationFactory
from .query import BulkDelete
from .query import BulkUpdate
from .query import Query
from .scoping import scoped_session
from .session import Session
from .session import sessionmaker
from .. import event
from .. import exc
from .. import util
from ..event import EventTarget
from ..event.registry import _ET
from ..util.compat import inspect_getfullargspec

if TYPE_CHECKING:
    from weakref import ReferenceType

    from ._typing import _InstanceDict
    from ._typing import _InternalEntityType
    from ._typing import _O
    from ._typing import _T
    from .attributes import Event
    from .base import EventConstants
    from .session import ORMExecuteState
    from .session import SessionTransaction
    from .unitofwork import UOWTransaction
    from ..engine import Connection
    from ..event.base import _Dispatch
    from ..event.base import _HasEventsDispatch
    from ..event.registry import _EventKey
    from ..orm.collections import CollectionAdapter
    from ..orm.context import QueryContext
    from ..orm.decl_api import DeclarativeAttributeIntercept
    from ..orm.decl_api import DeclarativeMeta
    from ..orm.mapper import Mapper
    from ..orm.state import InstanceState

_KT = TypeVar("_KT", bound=Any)
_ET2 = TypeVar("_ET2", bound=EventTarget)


class InstrumentationEvents(event.Events[InstrumentationFactory]):
    """Events related to class instrumentation events.

    The listeners here support being established against
    any new style class, that is any object that is a subclass
    of 'type'.  Events will then be fired off for events
    against that class.  If the "propagate=True" flag is passed
    to event.listen(), the event will fire off for subclasses
    of that class as well.

    The Python ``type`` builtin is also accepted as a target,
    which when used has the effect of events being emitted
    for all classes.

    Note the "propagate" flag here is defaulted to ``True``,
    unlike the other class level events where it defaults
    to ``False``.  This means that new subclasses will also
    be the subject of these events, when a listener
    is established on a superclass.

    """

    _target_class_doc = "SomeBaseClass"
    _dispatch_target = InstrumentationFactory

    @classmethod
    def _accept_with(
        cls,
        target: Union[
            InstrumentationFactory,
            Type[InstrumentationFactory],
        ],
        identifier: str,
    ) -> Optional[
        Union[
            InstrumentationFactory,
            Type[InstrumentationFactory],
        ]
    ]:
        if isinstance(target, type):
            return _InstrumentationEventsHold(target)  # type: ignore [return-value] # noqa: E501
        else:
            return None

    @classmethod
    def _listen(
        cls, event_key: _EventKey[_T], propagate: bool = True, **kw: Any
    ) -> None:
        target, identifier, fn = (
            event_key.dispatch_target,
            event_key.identifier,
            event_key._listen_fn,
        )

        def listen(target_cls: type, *arg: Any) -> Optional[Any]:
            listen_cls = target()

            # if weakref were collected, however this is not something
            # that normally happens.   it was occurring during test teardown
            # between mapper/registry/instrumentation_manager, however this
            # interaction was changed to not rely upon the event system.
            if listen_cls is None:
                return None

            if propagate and issubclass(target_cls, listen_cls):
                return fn(target_cls, *arg)
            elif not propagate and target_cls is listen_cls:
                return fn(target_cls, *arg)
            else:
                return None

        def remove(ref: ReferenceType[_T]) -> None:
            key = event.registry._EventKey(  # type: ignore [type-var]
                None,
                identifier,
                listen,
                instrumentation._instrumentation_factory,
            )
            getattr(
                instrumentation._instrumentation_factory.dispatch, identifier
            ).remove(key)

        target = weakref.ref(target.class_, remove)

        event_key.with_dispatch_target(
            instrumentation._instrumentation_factory
        ).with_wrapper(listen).base_listen(**kw)

    @classmethod
    def _clear(cls) -> None:
        super()._clear()
        instrumentation._instrumentation_factory.dispatch._clear()

    def class_instrument(self, cls: ClassManager[_O]) -> None:
        """Called after the given class is instrumented.

        To get at the :class:`.ClassManager`, use
        :func:`.manager_of_class`.

        """

    def class_uninstrument(self, cls: ClassManager[_O]) -> None:
        """Called before the given class is uninstrumented.

        To get at the :class:`.ClassManager`, use
        :func:`.manager_of_class`.

        """

    def attribute_instrument(
        self, cls: ClassManager[_O], key: _KT, inst: _O
    ) -> None:
        """Called when an attribute is instrumented."""


class _InstrumentationEventsHold:
    """temporary marker object used to transfer from _accept_with() to
    _listen() on the InstrumentationEvents class.

    """

    def __init__(self, class_: type) -> None:
        self.class_ = class_

    dispatch = event.dispatcher(InstrumentationEvents)


class InstanceEvents(event.Events[ClassManager[Any]]):
    """Define events specific to object lifecycle.

    e.g.::

        from sqlalchemy import event


        def my_load_listener(target, context):
            print("on load!")


        event.listen(SomeClass, "load", my_load_listener)

    Available targets include:

    * mapped classes
    * unmapped superclasses of mapped or to-be-mapped classes
      (using the ``propagate=True`` flag)
    * :class:`_orm.Mapper` objects
    * the :class:`_orm.Mapper` class itself indicates listening for all
      mappers.

    Instance events are closely related to mapper events, but
    are more specific to the instance and its instrumentation,
    rather than its system of persistence.

    When using :class:`.InstanceEvents`, several modifiers are
    available to the :func:`.event.listen` function.

    :param propagate=False: When True, the event listener should
       be applied to all inheriting classes as well as the
       class which is the target of this listener.
    :param raw=False: When True, the "target" argument passed
       to applicable event listener functions will be the
       instance's :class:`.InstanceState` management
       object, rather than the mapped instance itself.
    :param restore_load_context=False: Applies to the
       :meth:`.InstanceEvents.load` and :meth:`.InstanceEvents.refresh`
       events.  Restores the loader context of the object when the event
       hook is complete, so that ongoing eager load operations continue
       to target the object appropriately.  A warning is emitted if the
       object is moved to a new loader context from within one of these
       events if this flag is not set.

       .. versionadded:: 1.3.14


    """

    _target_class_doc = "SomeClass"

    _dispatch_target = ClassManager

    @classmethod
    def _new_classmanager_instance(
        cls,
        class_: Union[DeclarativeAttributeIntercept, DeclarativeMeta, type],
        classmanager: ClassManager[_O],
    ) -> None:
        _InstanceEventsHold.populate(class_, classmanager)

    @classmethod
    @util.preload_module("sqlalchemy.orm")
    def _accept_with(
        cls,
        target: Union[
            ClassManager[Any],
            Type[ClassManager[Any]],
        ],
        identifier: str,
    ) -> Optional[Union[ClassManager[Any], Type[ClassManager[Any]]]]:
        orm = util.preloaded.orm

        if isinstance(target, ClassManager):
            return target
        elif isinstance(target, mapperlib.Mapper):
            return target.class_manager
        elif target is orm.mapper:  # type: ignore [attr-defined]
            util.warn_deprecated(
                "The `sqlalchemy.orm.mapper()` symbol is deprecated and "
                "will be removed in a future release. For the mapper-wide "
                "event target, use the 'sqlalchemy.orm.Mapper' class.",
                "2.0",
            )
            return ClassManager
        elif isinstance(target, type):
            if issubclass(target, mapperlib.Mapper):
                return ClassManager
            else:
                manager = instrumentation.opt_manager_of_class(target)
                if manager:
                    return manager
                else:
                    return _InstanceEventsHold(target)  # type: ignore [return-value] # noqa: E501
        return None

    @classmethod
    def _listen(
        cls,
        event_key: _EventKey[ClassManager[Any]],
        raw: bool = False,
        propagate: bool = False,
        restore_load_context: bool = False,
        **kw: Any,
    ) -> None:
        target, fn = (event_key.dispatch_target, event_key._listen_fn)

        if not raw or restore_load_context:

            def wrap(
                state: InstanceState[_O], *arg: Any, **kw: Any
            ) -> Optional[Any]:
                if not raw:
                    target: Any = state.obj()
                else:
                    target = state
                if restore_load_context:
                    runid = state.runid
                try:
                    return fn(target, *arg, **kw)
                finally:
                    if restore_load_context:
                        state.runid = runid

            event_key = event_key.with_wrapper(wrap)

        event_key.base_listen(propagate=propagate, **kw)

        if propagate:
            for mgr in target.subclass_managers(True):
                event_key.with_dispatch_target(mgr).base_listen(propagate=True)

    @classmethod
    def _clear(cls) -> None:
        super()._clear()
        _InstanceEventsHold._clear()

    def first_init(self, manager: ClassManager[_O], cls: Type[_O]) -> None:
        """Called when the first instance of a particular mapping is called.

        This event is called when the ``__init__`` method of a class
        is called the first time for that particular class.    The event
        invokes before ``__init__`` actually proceeds as well as before
        the :meth:`.InstanceEvents.init` event is invoked.

        """

    def init(self, target: _O, args: Any, kwargs: Any) -> None:
        """Receive an instance when its constructor is called.

        This method is only called during a userland construction of
        an object, in conjunction with the object's constructor, e.g.
        its ``__init__`` method.  It is not called when an object is
        loaded from the database; see the :meth:`.InstanceEvents.load`
        event in order to intercept a database load.

        The event is called before the actual ``__init__`` constructor
        of the object is called.  The ``kwargs`` dictionary may be
        modified in-place in order to affect what is passed to
        ``__init__``.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param args: positional arguments passed to the ``__init__`` method.
         This is passed as a tuple and is currently immutable.
        :param kwargs: keyword arguments passed to the ``__init__`` method.
         This structure *can* be altered in place.

        .. seealso::

            :meth:`.InstanceEvents.init_failure`

            :meth:`.InstanceEvents.load`

        """

    def init_failure(self, target: _O, args: Any, kwargs: Any) -> None:
        """Receive an instance when its constructor has been called,
        and raised an exception.

        This method is only called during a userland construction of
        an object, in conjunction with the object's constructor, e.g.
        its ``__init__`` method. It is not called when an object is loaded
        from the database.

        The event is invoked after an exception raised by the ``__init__``
        method is caught.  After the event
        is invoked, the original exception is re-raised outwards, so that
        the construction of the object still raises an exception.   The
        actual exception and stack trace raised should be present in
        ``sys.exc_info()``.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param args: positional arguments that were passed to the ``__init__``
         method.
        :param kwargs: keyword arguments that were passed to the ``__init__``
         method.

        .. seealso::

            :meth:`.InstanceEvents.init`

            :meth:`.InstanceEvents.load`

        """

    def _sa_event_merge_wo_load(
        self, target: _O, context: QueryContext
    ) -> None:
        """receive an object instance after it was the subject of a merge()
        call, when load=False was passed.

        The target would be the already-loaded object in the Session which
        would have had its attributes overwritten by the incoming object. This
        overwrite operation does not use attribute events, instead just
        populating dict directly. Therefore the purpose of this event is so
        that extensions like sqlalchemy.ext.mutable know that object state has
        changed and incoming state needs to be set up for "parents" etc.

        This functionality is acceptable to be made public in a later release.

        .. versionadded:: 1.4.41

        """

    def load(self, target: _O, context: QueryContext) -> None:
        """Receive an object instance after it has been created via
        ``__new__``, and after initial attribute population has
        occurred.

        This typically occurs when the instance is created based on
        incoming result rows, and is only called once for that
        instance's lifetime.

        .. warning::

            During a result-row load, this event is invoked when the
            first row received for this instance is processed.  When using
            eager loading with collection-oriented attributes, the additional
            rows that are to be loaded / processed in order to load subsequent
            collection items have not occurred yet.   This has the effect
            both that collections will not be fully loaded, as well as that
            if an operation occurs within this event handler that emits
            another database load operation for the object, the "loading
            context" for the object can change and interfere with the
            existing eager loaders still in progress.

            Examples of what can cause the "loading context" to change within
            the event handler include, but are not necessarily limited to:

            * accessing deferred attributes that weren't part of the row,
              will trigger an "undefer" operation and refresh the object

            * accessing attributes on a joined-inheritance subclass that
              weren't part of the row, will trigger a refresh operation.

            As of SQLAlchemy 1.3.14, a warning is emitted when this occurs. The
            :paramref:`.InstanceEvents.restore_load_context` option may  be
            used on the event to prevent this warning; this will ensure that
            the existing loading context is maintained for the object after the
            event is called::

                @event.listens_for(SomeClass, "load", restore_load_context=True)
                def on_load(instance, context):
                    instance.some_unloaded_attribute

            .. versionchanged:: 1.3.14 Added
               :paramref:`.InstanceEvents.restore_load_context`
               and :paramref:`.SessionEvents.restore_load_context` flags which
               apply to "on load" events, which will ensure that the loading
               context for an object is restored when the event hook is
               complete; a warning is emitted if the load context of the object
               changes without this flag being set.


        The :meth:`.InstanceEvents.load` event is also available in a
        class-method decorator format called :func:`_orm.reconstructor`.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param context: the :class:`.QueryContext` corresponding to the
         current :class:`_query.Query` in progress.  This argument may be
         ``None`` if the load does not correspond to a :class:`_query.Query`,
         such as during :meth:`.Session.merge`.

        .. seealso::

            :ref:`mapped_class_load_events`

            :meth:`.InstanceEvents.init`

            :meth:`.InstanceEvents.refresh`

            :meth:`.SessionEvents.loaded_as_persistent`

        """  # noqa: E501

    def refresh(
        self, target: _O, context: QueryContext, attrs: Optional[Iterable[str]]
    ) -> None:
        """Receive an object instance after one or more attributes have
        been refreshed from a query.

        Contrast this to the :meth:`.InstanceEvents.load` method, which
        is invoked when the object is first loaded from a query.

        .. note:: This event is invoked within the loader process before
           eager loaders may have been completed, and the object's state may
           not be complete.  Additionally, invoking row-level refresh
           operations on the object will place the object into a new loader
           context, interfering with the existing load context.   See the note
           on :meth:`.InstanceEvents.load` for background on making use of the
           :paramref:`.InstanceEvents.restore_load_context` parameter, in
           order to resolve this scenario.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param context: the :class:`.QueryContext` corresponding to the
         current :class:`_query.Query` in progress.
        :param attrs: sequence of attribute names which
         were populated, or None if all column-mapped, non-deferred
         attributes were populated.

        .. seealso::

            :ref:`mapped_class_load_events`

            :meth:`.InstanceEvents.load`

        """

    def refresh_flush(
        self,
        target: _O,
        flush_context: UOWTransaction,
        attrs: Optional[Iterable[str]],
    ) -> None:
        """Receive an object instance after one or more attributes that
        contain a column-level default or onupdate handler have been refreshed
        during persistence of the object's state.

        This event is the same as :meth:`.InstanceEvents.refresh` except
        it is invoked within the unit of work flush process, and includes
        only non-primary-key columns that have column level default or
        onupdate handlers, including Python callables as well as server side
        defaults and triggers which may be fetched via the RETURNING clause.

        .. note::

            While the :meth:`.InstanceEvents.refresh_flush` event is triggered
            for an object that was INSERTed as well as for an object that was
            UPDATEd, the event is geared primarily  towards the UPDATE process;
            it is mostly an internal artifact that INSERT actions can also
            trigger this event, and note that **primary key columns for an
            INSERTed row are explicitly omitted** from this event.  In order to
            intercept the newly INSERTed state of an object, the
            :meth:`.SessionEvents.pending_to_persistent` and
            :meth:`.MapperEvents.after_insert` are better choices.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.
        :param attrs: sequence of attribute names which
         were populated.

        .. seealso::

            :ref:`mapped_class_load_events`

            :ref:`orm_server_defaults`

            :ref:`metadata_defaults_toplevel`

        """

    def expire(self, target: _O, attrs: Optional[Iterable[str]]) -> None:
        """Receive an object instance after its attributes or some subset
        have been expired.

        'keys' is a list of attribute names.  If None, the entire
        state was expired.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param attrs: sequence of attribute
         names which were expired, or None if all attributes were
         expired.

        """

    def pickle(self, target: _O, state_dict: _InstanceDict) -> None:
        """Receive an object instance when its associated state is
        being pickled.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param state_dict: the dictionary returned by
         :class:`.InstanceState.__getstate__`, containing the state
         to be pickled.

        """

    def unpickle(self, target: _O, state_dict: _InstanceDict) -> None:
        """Receive an object instance after its associated state has
        been unpickled.

        :param target: the mapped instance.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :param state_dict: the dictionary sent to
         :class:`.InstanceState.__setstate__`, containing the state
         dictionary which was pickled.

        """


class _EventsHold(event.RefCollection[_ET]):
    """Hold onto listeners against unmapped, uninstrumented classes.

    Establish _listen() for that class' mapper/instrumentation when
    those objects are created for that class.

    """

    all_holds: weakref.WeakKeyDictionary[Any, Any]

    def __init__(
        self,
        class_: Union[DeclarativeAttributeIntercept, DeclarativeMeta, type],
    ) -> None:
        self.class_ = class_

    @classmethod
    def _clear(cls) -> None:
        cls.all_holds.clear()

    class HoldEvents(Generic[_ET2]):
        _dispatch_target: Optional[Type[_ET2]] = None

        @classmethod
        def _listen(
            cls,
            event_key: _EventKey[_ET2],
            raw: bool = False,
            propagate: bool = False,
            retval: bool = False,
            **kw: Any,
        ) -> None:
            target = event_key.dispatch_target

            if target.class_ in target.all_holds:
                collection = target.all_holds[target.class_]
            else:
                collection = target.all_holds[target.class_] = {}

            event.registry._stored_in_collection(event_key, target)
            collection[event_key._key] = (
                event_key,
                raw,
                propagate,
                retval,
                kw,
            )

            if propagate:
                stack = list(target.class_.__subclasses__())
                while stack:
                    subclass = stack.pop(0)
                    stack.extend(subclass.__subclasses__())
                    subject = target.resolve(subclass)
                    if subject is not None:
                        # we are already going through __subclasses__()
                        # so leave generic propagate flag False
                        event_key.with_dispatch_target(subject).listen(
                            raw=raw, propagate=False, retval=retval, **kw
                        )

    def remove(self, event_key: _EventKey[_ET]) -> None:
        target = event_key.dispatch_target

        if isinstance(target, _EventsHold):
            collection = target.all_holds[target.class_]
            del collection[event_key._key]

    @classmethod
    def populate(
        cls,
        class_: Union[DeclarativeAttributeIntercept, DeclarativeMeta, type],
        subject: Union[ClassManager[_O], Mapper[_O]],
    ) -> None:
        for subclass in class_.__mro__:
            if subclass in cls.all_holds:
                collection = cls.all_holds[subclass]
                for (
                    event_key,
                    raw,
                    propagate,
                    retval,
                    kw,
                ) in collection.values():
                    if propagate or subclass is class_:
                        # since we can't be sure in what order different
                        # classes in a hierarchy are triggered with
                        # populate(), we rely upon _EventsHold for all event
                        # assignment, instead of using the generic propagate
                        # flag.
                        event_key.with_dispatch_target(subject).listen(
                            raw=raw, propagate=False, retval=retval, **kw
                        )


class _InstanceEventsHold(_EventsHold[_ET]):
    all_holds: weakref.WeakKeyDictionary[Any, Any] = (
        weakref.WeakKeyDictionary()
    )

    def resolve(self, class_: Type[_O]) -> Optional[ClassManager[_O]]:
        return instrumentation.opt_manager_of_class(class_)

    class HoldInstanceEvents(_EventsHold.HoldEvents[_ET], InstanceEvents):  # type: ignore [misc] # noqa: E501
        pass

    dispatch = event.dispatcher(HoldInstanceEvents)


class MapperEvents(event.Events[mapperlib.Mapper[Any]]):
    """Define events specific to mappings.

    e.g.::

        from sqlalchemy import event


        def my_before_insert_listener(mapper, connection, target):
            # execute a stored procedure upon INSERT,
            # apply the value to the row to be inserted
            target.calculated_value = connection.execute(
                text("select my_special_function(%d)" % target.special_number)
            ).scalar()


        # associate the listener function with SomeClass,
        # to execute during the "before_insert" hook
        event.listen(SomeClass, "before_insert", my_before_insert_listener)

    Available targets include:

    * mapped classes
    * unmapped superclasses of mapped or to-be-mapped classes
      (using the ``propagate=True`` flag)
    * :class:`_orm.Mapper` objects
    * the :class:`_orm.Mapper` class itself indicates listening for all
      mappers.

    Mapper events provide hooks into critical sections of the
    mapper, including those related to object instrumentation,
    object loading, and object persistence. In particular, the
    persistence methods :meth:`~.MapperEvents.before_insert`,
    and :meth:`~.MapperEvents.before_update` are popular
    places to augment the state being persisted - however, these
    methods operate with several significant restrictions. The
    user is encouraged to evaluate the
    :meth:`.SessionEvents.before_flush` and
    :meth:`.SessionEvents.after_flush` methods as more
    flexible and user-friendly hooks in which to apply
    additional database state during a flush.

    When using :class:`.MapperEvents`, several modifiers are
    available to the :func:`.event.listen` function.

    :param propagate=False: When True, the event listener should
       be applied to all inheriting mappers and/or the mappers of
       inheriting classes, as well as any
       mapper which is the target of this listener.
    :param raw=False: When True, the "target" argument passed
       to applicable event listener functions will be the
       instance's :class:`.InstanceState` management
       object, rather than the mapped instance itself.
    :param retval=False: when True, the user-defined event function
       must have a return value, the purpose of which is either to
       control subsequent event propagation, or to otherwise alter
       the operation in progress by the mapper.   Possible return
       values are:

       * ``sqlalchemy.orm.interfaces.EXT_CONTINUE`` - continue event
         processing normally.
       * ``sqlalchemy.orm.interfaces.EXT_STOP`` - cancel all subsequent
         event handlers in the chain.
       * other values - the return value specified by specific listeners.

    """

    _target_class_doc = "SomeClass"
    _dispatch_target = mapperlib.Mapper

    @classmethod
    def _new_mapper_instance(
        cls,
        class_: Union[DeclarativeAttributeIntercept, DeclarativeMeta, type],
        mapper: Mapper[_O],
    ) -> None:
        _MapperEventsHold.populate(class_, mapper)

    @classmethod
    @util.preload_module("sqlalchemy.orm")
    def _accept_with(
        cls,
        target: Union[mapperlib.Mapper[Any], Type[mapperlib.Mapper[Any]]],
        identifier: str,
    ) -> Optional[Union[mapperlib.Mapper[Any], Type[mapperlib.Mapper[Any]]]]:
        orm = util.preloaded.orm

        if target is orm.mapper:  # type: ignore [attr-defined]
            util.warn_deprecated(
                "The `sqlalchemy.orm.mapper()` symbol is deprecated and "
                "will be removed in a future release. For the mapper-wide "
                "event target, use the 'sqlalchemy.orm.Mapper' class.",
                "2.0",
            )
            return mapperlib.Mapper
        elif isinstance(target, type):
            if issubclass(target, mapperlib.Mapper):
                return target
            else:
                mapper = _mapper_or_none(target)
                if mapper is not None:
                    return mapper
                else:
                    return _MapperEventsHold(target)
        else:
            return target

    @classmethod
    def _listen(
        cls,
        event_key: _EventKey[_ET],
        raw: bool = False,
        retval: bool = False,
        propagate: bool = False,
        **kw: Any,
    ) -> None:
        target, identifier, fn = (
            event_key.dispatch_target,
            event_key.identifier,
            event_key._listen_fn,
        )

        if (
            identifier in ("before_configured", "after_configured")
            and target is not mapperlib.Mapper
        ):
            util.warn(
                "'before_configured' and 'after_configured' ORM events "
                "only invoke with the Mapper class "
                "as the target."
            )

        if not raw or not retval:
            if not raw:
                meth = getattr(cls, identifier)
                try:
                    target_index = (
                        inspect_getfullargspec(meth)[0].index("target") - 1
                    )
                except ValueError:
                    target_index = None

            def wrap(*arg: Any, **kw: Any) -> Any:
                if not raw and target_index is not None:
                    arg = list(arg)  # type: ignore [assignment]
                    arg[target_index] = arg[target_index].obj()  # type: ignore [index] # noqa: E501
                if not retval:
                    fn(*arg, **kw)
                    return interfaces.EXT_CONTINUE
                else:
                    return fn(*arg, **kw)

            event_key = event_key.with_wrapper(wrap)

        if propagate:
            for mapper in target.self_and_descendants:
                event_key.with_dispatch_target(mapper).base_listen(
                    propagate=True, **kw
                )
        else:
            event_key.base_listen(**kw)

    @classmethod
    def _clear(cls) -> None:
        super()._clear()
        _MapperEventsHold._clear()

    def instrument_class(self, mapper: Mapper[_O], class_: Type[_O]) -> None:
        r"""Receive a class when the mapper is first constructed,
        before instrumentation is applied to the mapped class.

        This event is the earliest phase of mapper construction.
        Most attributes of the mapper are not yet initialized.   To
        receive an event within initial mapper construction where basic
        state is available such as the :attr:`_orm.Mapper.attrs` collection,
        the :meth:`_orm.MapperEvents.after_mapper_constructed` event may
        be a better choice.

        This listener can either be applied to the :class:`_orm.Mapper`
        class overall, or to any un-mapped class which serves as a base
        for classes that will be mapped (using the ``propagate=True`` flag)::

            Base = declarative_base()


            @event.listens_for(Base, "instrument_class", propagate=True)
            def on_new_class(mapper, cls_):
                "..."

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param class\_: the mapped class.

        .. seealso::

            :meth:`_orm.MapperEvents.after_mapper_constructed`

        """

    def after_mapper_constructed(
        self, mapper: Mapper[_O], class_: Type[_O]
    ) -> None:
        """Receive a class and mapper when the :class:`_orm.Mapper` has been
        fully constructed.

        This event is called after the initial constructor for
        :class:`_orm.Mapper` completes.  This occurs after the
        :meth:`_orm.MapperEvents.instrument_class` event and after the
        :class:`_orm.Mapper` has done an initial pass of its arguments
        to generate its collection of :class:`_orm.MapperProperty` objects,
        which are accessible via the :meth:`_orm.Mapper.get_property`
        method and the :attr:`_orm.Mapper.iterate_properties` attribute.

        This event differs from the
        :meth:`_orm.MapperEvents.before_mapper_configured` event in that it
        is invoked within the constructor for :class:`_orm.Mapper`, rather
        than within the :meth:`_orm.registry.configure` process.   Currently,
        this event is the only one which is appropriate for handlers that
        wish to create additional mapped classes in response to the
        construction of this :class:`_orm.Mapper`, which will be part of the
        same configure step when :meth:`_orm.registry.configure` next runs.

        .. versionadded:: 2.0.2

        .. seealso::

            :ref:`examples_versioning` - an example which illustrates the use
            of the :meth:`_orm.MapperEvents.before_mapper_configured`
            event to create new mappers to record change-audit histories on
            objects.

        """

    def before_mapper_configured(
        self, mapper: Mapper[_O], class_: Type[_O]
    ) -> None:
        """Called right before a specific mapper is to be configured.

        This event is intended to allow a specific mapper to be skipped during
        the configure step, by returning the :attr:`.orm.interfaces.EXT_SKIP`
        symbol which indicates to the :func:`.configure_mappers` call that this
        particular mapper (or hierarchy of mappers, if ``propagate=True`` is
        used) should be skipped in the current configuration run. When one or
        more mappers are skipped, the "new mappers" flag will remain set,
        meaning the :func:`.configure_mappers` function will continue to be
        called when mappers are used, to continue to try to configure all
        available mappers.

        In comparison to the other configure-level events,
        :meth:`.MapperEvents.before_configured`,
        :meth:`.MapperEvents.after_configured`, and
        :meth:`.MapperEvents.mapper_configured`, the
        :meth:`.MapperEvents.before_mapper_configured` event provides for a
        meaningful return value when it is registered with the ``retval=True``
        parameter.

        .. versionadded:: 1.3

        e.g.::

            from sqlalchemy.orm import EXT_SKIP

            Base = declarative_base()

            DontConfigureBase = declarative_base()


            @event.listens_for(
                DontConfigureBase,
                "before_mapper_configured",
                retval=True,
                propagate=True,
            )
            def dont_configure(mapper, cls):
                return EXT_SKIP

        .. seealso::

            :meth:`.MapperEvents.before_configured`

            :meth:`.MapperEvents.after_configured`

            :meth:`.MapperEvents.mapper_configured`

        """

    def mapper_configured(self, mapper: Mapper[_O], class_: Type[_O]) -> None:
        r"""Called when a specific mapper has completed its own configuration
        within the scope of the :func:`.configure_mappers` call.

        The :meth:`.MapperEvents.mapper_configured` event is invoked
        for each mapper that is encountered when the
        :func:`_orm.configure_mappers` function proceeds through the current
        list of not-yet-configured mappers.
        :func:`_orm.configure_mappers` is typically invoked
        automatically as mappings are first used, as well as each time
        new mappers have been made available and new mapper use is
        detected.

        When the event is called, the mapper should be in its final
        state, but **not including backrefs** that may be invoked from
        other mappers; they might still be pending within the
        configuration operation.    Bidirectional relationships that
        are instead configured via the
        :paramref:`.orm.relationship.back_populates` argument
        *will* be fully available, since this style of relationship does not
        rely upon other possibly-not-configured mappers to know that they
        exist.

        For an event that is guaranteed to have **all** mappers ready
        to go including backrefs that are defined only on other
        mappings, use the :meth:`.MapperEvents.after_configured`
        event; this event invokes only after all known mappings have been
        fully configured.

        The :meth:`.MapperEvents.mapper_configured` event, unlike
        :meth:`.MapperEvents.before_configured` or
        :meth:`.MapperEvents.after_configured`,
        is called for each mapper/class individually, and the mapper is
        passed to the event itself.  It also is called exactly once for
        a particular mapper.  The event is therefore useful for
        configurational steps that benefit from being invoked just once
        on a specific mapper basis, which don't require that "backref"
        configurations are necessarily ready yet.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param class\_: the mapped class.

        .. seealso::

            :meth:`.MapperEvents.before_configured`

            :meth:`.MapperEvents.after_configured`

            :meth:`.MapperEvents.before_mapper_configured`

        """
        # TODO: need coverage for this event

    def before_configured(self) -> None:
        """Called before a series of mappers have been configured.

        The :meth:`.MapperEvents.before_configured` event is invoked
        each time the :func:`_orm.configure_mappers` function is
        invoked, before the function has done any of its work.
        :func:`_orm.configure_mappers` is typically invoked
        automatically as mappings are first used, as well as each time
        new mappers have been made available and new mapper use is
        detected.

        This event can **only** be applied to the :class:`_orm.Mapper` class,
        and not to individual mappings or mapped classes. It is only invoked
        for all mappings as a whole::

            from sqlalchemy.orm import Mapper


            @event.listens_for(Mapper, "before_configured")
            def go(): ...

        Contrast this event to :meth:`.MapperEvents.after_configured`,
        which is invoked after the series of mappers has been configured,
        as well as :meth:`.MapperEvents.before_mapper_configured`
        and :meth:`.MapperEvents.mapper_configured`, which are both invoked
        on a per-mapper basis.

        Theoretically this event is called once per
        application, but is actually called any time new mappers
        are to be affected by a :func:`_orm.configure_mappers`
        call.   If new mappings are constructed after existing ones have
        already been used, this event will likely be called again.  To ensure
        that a particular event is only called once and no further, the
        ``once=True`` argument (new in 0.9.4) can be applied::

            from sqlalchemy.orm import mapper


            @event.listens_for(mapper, "before_configured", once=True)
            def go(): ...

        .. seealso::

            :meth:`.MapperEvents.before_mapper_configured`

            :meth:`.MapperEvents.mapper_configured`

            :meth:`.MapperEvents.after_configured`

        """

    def after_configured(self) -> None:
        """Called after a series of mappers have been configured.

        The :meth:`.MapperEvents.after_configured` event is invoked
        each time the :func:`_orm.configure_mappers` function is
        invoked, after the function has completed its work.
        :func:`_orm.configure_mappers` is typically invoked
        automatically as mappings are first used, as well as each time
        new mappers have been made available and new mapper use is
        detected.

        Contrast this event to the :meth:`.MapperEvents.mapper_configured`
        event, which is called on a per-mapper basis while the configuration
        operation proceeds; unlike that event, when this event is invoked,
        all cross-configurations (e.g. backrefs) will also have been made
        available for any mappers that were pending.
        Also contrast to :meth:`.MapperEvents.before_configured`,
        which is invoked before the series of mappers has been configured.

        This event can **only** be applied to the :class:`_orm.Mapper` class,
        and not to individual mappings or
        mapped classes.  It is only invoked for all mappings as a whole::

            from sqlalchemy.orm import Mapper


            @event.listens_for(Mapper, "after_configured")
            def go(): ...

        Theoretically this event is called once per
        application, but is actually called any time new mappers
        have been affected by a :func:`_orm.configure_mappers`
        call.   If new mappings are constructed after existing ones have
        already been used, this event will likely be called again.  To ensure
        that a particular event is only called once and no further, the
        ``once=True`` argument (new in 0.9.4) can be applied::

            from sqlalchemy.orm import mapper


            @event.listens_for(mapper, "after_configured", once=True)
            def go(): ...

        .. seealso::

            :meth:`.MapperEvents.before_mapper_configured`

            :meth:`.MapperEvents.mapper_configured`

            :meth:`.MapperEvents.before_configured`

        """

    def before_insert(
        self, mapper: Mapper[_O], connection: Connection, target: _O
    ) -> None:
        """Receive an object instance before an INSERT statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to modify local, non-object related
        attributes on the instance before an INSERT occurs, as well
        as to emit additional SQL statements on the given
        connection.

        The event is often called for a batch of objects of the
        same class before their INSERT statements are emitted at
        once in a later step. In the extremely rare case that
        this is not desirable, the :class:`_orm.Mapper` object can be
        configured with ``batch=False``, which will cause
        batches of instances to be broken up into individual
        (and more poorly performing) event->persist->event
        steps.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit INSERT statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being persisted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def after_insert(
        self, mapper: Mapper[_O], connection: Connection, target: _O
    ) -> None:
        """Receive an object instance after an INSERT statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to modify in-Python-only
        state on the instance after an INSERT occurs, as well
        as to emit additional SQL statements on the given
        connection.

        The event is often called for a batch of objects of the
        same class after their INSERT statements have been
        emitted at once in a previous step. In the extremely
        rare case that this is not desirable, the
        :class:`_orm.Mapper` object can be configured with ``batch=False``,
        which will cause batches of instances to be broken up
        into individual (and more poorly performing)
        event->persist->event steps.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit INSERT statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being persisted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def before_update(
        self, mapper: Mapper[_O], connection: Connection, target: _O
    ) -> None:
        """Receive an object instance before an UPDATE statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to modify local, non-object related
        attributes on the instance before an UPDATE occurs, as well
        as to emit additional SQL statements on the given
        connection.

        This method is called for all instances that are
        marked as "dirty", *even those which have no net changes
        to their column-based attributes*. An object is marked
        as dirty when any of its column-based attributes have a
        "set attribute" operation called or when any of its
        collections are modified. If, at update time, no
        column-based attributes have any net changes, no UPDATE
        statement will be issued. This means that an instance
        being sent to :meth:`~.MapperEvents.before_update` is
        *not* a guarantee that an UPDATE statement will be
        issued, although you can affect the outcome here by
        modifying attributes so that a net change in value does
        exist.

        To detect if the column-based attributes on the object have net
        changes, and will therefore generate an UPDATE statement, use
        ``object_session(instance).is_modified(instance,
        include_collections=False)``.

        The event is often called for a batch of objects of the
        same class before their UPDATE statements are emitted at
        once in a later step. In the extremely rare case that
        this is not desirable, the :class:`_orm.Mapper` can be
        configured with ``batch=False``, which will cause
        batches of instances to be broken up into individual
        (and more poorly performing) event->persist->event
        steps.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit UPDATE statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being persisted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def after_update(
        self, mapper: Mapper[_O], connection: Connection, target: _O
    ) -> None:
        """Receive an object instance after an UPDATE statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to modify in-Python-only
        state on the instance after an UPDATE occurs, as well
        as to emit additional SQL statements on the given
        connection.

        This method is called for all instances that are
        marked as "dirty", *even those which have no net changes
        to their column-based attributes*, and for which
        no UPDATE statement has proceeded. An object is marked
        as dirty when any of its column-based attributes have a
        "set attribute" operation called or when any of its
        collections are modified. If, at update time, no
        column-based attributes have any net changes, no UPDATE
        statement will be issued. This means that an instance
        being sent to :meth:`~.MapperEvents.after_update` is
        *not* a guarantee that an UPDATE statement has been
        issued.

        To detect if the column-based attributes on the object have net
        changes, and therefore resulted in an UPDATE statement, use
        ``object_session(instance).is_modified(instance,
        include_collections=False)``.

        The event is often called for a batch of objects of the
        same class after their UPDATE statements have been emitted at
        once in a previous step. In the extremely rare case that
        this is not desirable, the :class:`_orm.Mapper` can be
        configured with ``batch=False``, which will cause
        batches of instances to be broken up into individual
        (and more poorly performing) event->persist->event
        steps.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit UPDATE statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being persisted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def before_delete(
        self, mapper: Mapper[_O], connection: Connection, target: _O
    ) -> None:
        """Receive an object instance before a DELETE statement
        is emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to emit additional SQL statements on
        the given connection as well as to perform application
        specific bookkeeping related to a deletion event.

        The event is often called for a batch of objects of the
        same class before their DELETE statements are emitted at
        once in a later step.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit DELETE statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being deleted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """

    def after_delete(
        self, mapper: Mapper[_O], connection: Connection, target: _O
    ) -> None:
        """Receive an object instance after a DELETE statement
        has been emitted corresponding to that instance.

        .. note:: this event **only** applies to the
           :ref:`session flush operation <session_flushing>`
           and does **not** apply to the ORM DML operations described at
           :ref:`orm_expression_update_delete`.  To intercept ORM
           DML events, use :meth:`_orm.SessionEvents.do_orm_execute`.

        This event is used to emit additional SQL statements on
        the given connection as well as to perform application
        specific bookkeeping related to a deletion event.

        The event is often called for a batch of objects of the
        same class after their DELETE statements have been emitted at
        once in a previous step.

        .. warning::

            Mapper-level flush events only allow **very limited operations**,
            on attributes local to the row being operated upon only,
            as well as allowing any SQL to be emitted on the given
            :class:`_engine.Connection`.  **Please read fully** the notes
            at :ref:`session_persistence_mapper` for guidelines on using
            these methods; generally, the :meth:`.SessionEvents.before_flush`
            method should be preferred for general on-flush changes.

        :param mapper: the :class:`_orm.Mapper` which is the target
         of this event.
        :param connection: the :class:`_engine.Connection` being used to
         emit DELETE statements for this instance.  This
         provides a handle into the current transaction on the
         target database specific to this instance.
        :param target: the mapped instance being deleted.  If
         the event is configured with ``raw=True``, this will
         instead be the :class:`.InstanceState` state-management
         object associated with the instance.
        :return: No return value is supported by this event.

        .. seealso::

            :ref:`session_persistence_events`

        """


class _MapperEventsHold(_EventsHold[_ET]):
    all_holds = weakref.WeakKeyDictionary()

    def resolve(
        self, class_: Union[Type[_T], _InternalEntityType[_T]]
    ) -> Optional[Mapper[_T]]:
        return _mapper_or_none(class_)

    class HoldMapperEvents(_EventsHold.HoldEvents[_ET], MapperEvents):  # type: ignore [misc] # noqa: E501
        pass

    dispatch = event.dispatcher(HoldMapperEvents)


_sessionevents_lifecycle_event_names: Set[str] = set()


class SessionEvents(event.Events[Session]):
    """Define events specific to :class:`.Session` lifecycle.

    e.g.::

        from sqlalchemy import event
        from sqlalchemy.orm import sessionmaker


        def my_before_commit(session):
            print("before commit!")


        Session = sessionmaker()

        event.listen(Session, "before_commit", my_before_commit)

    The :func:`~.event.listen` function will accept
    :class:`.Session` objects as well as the return result
    of :class:`~.sessionmaker()` and :class:`~.scoped_session()`.

    Additionally, it accepts the :class:`.Session` class which
    will apply listeners to all :class:`.Session` instances
    globally.

    :param raw=False: When True, the "target" argument passed
       to applicable event listener functions that work on individual
       objects will be the instance's :class:`.InstanceState` management
       object, rather than the mapped instance itself.

       .. versionadded:: 1.3.14

    :param restore_load_context=False: Applies to the
       :meth:`.SessionEvents.loaded_as_persistent` event.  Restores the loader
       context of the object when the event hook is complete, so that ongoing
       eager load operations continue to target the object appropriately.  A
       warning is emitted if the object is moved to a new loader context from
       within this event if this flag is not set.

       .. versionadded:: 1.3.14

    """

    _target_class_doc = "SomeSessionClassOrObject"

    _dispatch_target = Session

    def _lifecycle_event(  # type: ignore [misc]
        fn: Callable[[SessionEvents, Session, Any], None]
    ) -> Callable[[SessionEvents, Session, Any], None]:
        _sessionevents_lifecycle_event_names.add(fn.__name__)
        return fn

    @classmethod
    def _accept_with(  # type: ignore [return]
        cls, target: Any, identifier: str
    ) -> Union[Session, type]:
        if isinstance(target, scoped_session):
            target = target.session_factory
            if not isinstance(target, sessionmaker) and (
                not isinstance(target, type) or not issubclass(target, Session)
            ):
                raise exc.ArgumentError(
                    "Session event listen on a scoped_session "
                    "requires that its creation callable "
                    "is associated with the Session class."
                )

        if isinstance(target, sessionmaker):
            return target.class_
        elif isinstance(target, type):
            if issubclass(target, scoped_session):
                return Session
            elif issubclass(target, Session):
                return target
        elif isinstance(target, Session):
            return target
        elif hasattr(target, "_no_async_engine_events"):
            target._no_async_engine_events()
        else:
            # allows alternate SessionEvents-like-classes to be consulted
            return event.Events._accept_with(target, identifier)  # type: ignore [return-value] # noqa: E501

    @classmethod
    def _listen(
        cls,
        event_key: Any,
        *,
        raw: bool = False,
        restore_load_context: bool = False,
        **kw: Any,
    ) -> None:
        is_instance_event = (
            event_key.identifier in _sessionevents_lifecycle_event_names
        )

        if is_instance_event:
            if not raw or restore_load_context:
                fn = event_key._listen_fn

                def wrap(
                    session: Session,
                    state: InstanceState[_O],
                    *arg: Any,
                    **kw: Any,
                ) -> Optional[Any]:
                    if not raw:
                        target = state.obj()
                        if target is None:
                            # existing behavior is that if the object is
                            # garbage collected, no event is emitted
                            return None
                    else:
                        target = state  # type: ignore [assignment]
                    if restore_load_context:
                        runid = state.runid
                    try:
                        return fn(session, target, *arg, **kw)
                    finally:
                        if restore_load_context:
                            state.runid = runid

                event_key = event_key.with_wrapper(wrap)

        event_key.base_listen(**kw)

    def do_orm_execute(self, orm_execute_state: ORMExecuteState) -> None:
        """Intercept statement executions that occur on behalf of an
        ORM :class:`.Session` object.

        This event is invoked for all top-level SQL statements invoked from the
        :meth:`_orm.Session.execute` method, as well as related methods such as
        :meth:`_orm.Session.scalars` and :meth:`_orm.Session.scalar`. As of
        SQLAlchemy 1.4, all ORM queries that run through the
        :meth:`_orm.Session.execute` method as well as related methods
        :meth:`_orm.Session.scalars`, :meth:`_orm.Session.scalar` etc.
        will participate in this event.
        This event hook does **not** apply to the queries that are
        emitted internally within the ORM flush process, i.e. the
        process described at :ref:`session_flushing`.

        .. note::  The :meth:`_orm.SessionEvents.do_orm_execute` event hook
           is triggered **for ORM statement executions only**, meaning those
           invoked via the :meth:`_orm.Session.execute` and similar methods on
           the :class:`_orm.Session` object. It does **not** trigger for
           statements that are invoked by SQLAlchemy Core only, i.e. statements
           invoked directly using :meth:`_engine.Connection.execute` or
           otherwise originating from an :class:`_engine.Engine` object without
           any :class:`_orm.Session` involved. To intercept **all** SQL
           executions regardless of whether the Core or ORM APIs are in use,
           see the event hooks at :class:`.ConnectionEvents`, such as
           :meth:`.ConnectionEvents.before_execute` and
           :meth:`.ConnectionEvents.before_cursor_execute`.

           Also, this event hook does **not** apply to queries that are
           emitted internally within the ORM flush process,
           i.e. the process described at :ref:`session_flushing`; to
           intercept steps within the flush process, see the event
           hooks described at :ref:`session_persistence_events` as
           well as :ref:`session_persistence_mapper`.

        This event is a ``do_`` event, meaning it has the capability to replace
        the operation that the :meth:`_orm.Session.execute` method normally
        performs.  The intended use for this includes sharding and
        result-caching schemes which may seek to invoke the same statement
        across  multiple database connections, returning a result that is
        merged from each of them, or which don't invoke the statement at all,
        instead returning data from a cache.

        The hook intends to replace the use of the
        ``Query._execute_and_instances`` method that could be subclassed prior
        to SQLAlchemy 1.4.

        :param orm_execute_state: an instance of :class:`.ORMExecuteState`
         which contains all information about the current execution, as well
         as helper functions used to derive other commonly required
         information.   See that object for details.

        .. seealso::

            :ref:`session_execute_events` - top level documentation on how
            to use :meth:`_orm.SessionEvents.do_orm_execute`

            :class:`.ORMExecuteState` - the object passed to the
            :meth:`_orm.SessionEvents.do_orm_execute` event which contains
            all information about the statement to be invoked.  It also
            provides an interface to extend the current statement, options,
            and parameters as well as an option that allows programmatic
            invocation of the statement at any point.

            :ref:`examples_session_orm_events` - includes examples of using
            :meth:`_orm.SessionEvents.do_orm_execute`

            :ref:`examples_caching` - an example of how to integrate
            Dogpile caching with the ORM :class:`_orm.Session` making use
            of the :meth:`_orm.SessionEvents.do_orm_execute` event hook.

            :ref:`examples_sharding` - the Horizontal Sharding example /
            extension relies upon the
            :meth:`_orm.SessionEvents.do_orm_execute` event hook to invoke a
            SQL statement on multiple backends and return a merged result.


        .. versionadded:: 1.4

        """

    def after_transaction_create(
        self, session: Session, transaction: SessionTransaction
    ) -> None:
        """Execute when a new :class:`.SessionTransaction` is created.

        This event differs from :meth:`~.SessionEvents.after_begin`
        in that it occurs for each :class:`.SessionTransaction`
        overall, as opposed to when transactions are begun
        on individual database connections.  It is also invoked
        for nested transactions and subtransactions, and is always
        matched by a corresponding
        :meth:`~.SessionEvents.after_transaction_end` event
        (assuming normal operation of the :class:`.Session`).

        :param session: the target :class:`.Session`.
        :param transaction: the target :class:`.SessionTransaction`.

         To detect if this is the outermost
         :class:`.SessionTransaction`, as opposed to a "subtransaction" or a
         SAVEPOINT, test that the :attr:`.SessionTransaction.parent` attribute
         is ``None``::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_create(session, transaction):
                    if transaction.parent is None:
                        ...  # work with top-level transaction

         To detect if the :class:`.SessionTransaction` is a SAVEPOINT, use the
         :attr:`.SessionTransaction.nested` attribute::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_create(session, transaction):
                    if transaction.nested:
                        ...  # work with SAVEPOINT transaction

        .. seealso::

            :class:`.SessionTransaction`

            :meth:`~.SessionEvents.after_transaction_end`

        """

    def after_transaction_end(
        self, session: Session, transaction: SessionTransaction
    ) -> None:
        """Execute when the span of a :class:`.SessionTransaction` ends.

        This event differs from :meth:`~.SessionEvents.after_commit`
        in that it corresponds to all :class:`.SessionTransaction`
        objects in use, including those for nested transactions
        and subtransactions, and is always matched by a corresponding
        :meth:`~.SessionEvents.after_transaction_create` event.

        :param session: the target :class:`.Session`.
        :param transaction: the target :class:`.SessionTransaction`.

         To detect if this is the outermost
         :class:`.SessionTransaction`, as opposed to a "subtransaction" or a
         SAVEPOINT, test that the :attr:`.SessionTransaction.parent` attribute
         is ``None``::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_end(session, transaction):
                    if transaction.parent is None:
                        ...  # work with top-level transaction

         To detect if the :class:`.SessionTransaction` is a SAVEPOINT, use the
         :attr:`.SessionTransaction.nested` attribute::

                @event.listens_for(session, "after_transaction_create")
                def after_transaction_end(session, transaction):
                    if transaction.nested:
                        ...  # work with SAVEPOINT transaction

        .. seealso::

            :class:`.SessionTransaction`

            :meth:`~.SessionEvents.after_transaction_create`

        """

    def before_commit(self, session: Session) -> None:
        """Execute before commit is called.

        .. note::

            The :meth:`~.SessionEvents.before_commit` hook is *not* per-flush,
            that is, the :class:`.Session` can emit SQL to the database
            many times within the scope of a transaction.
            For interception of these events, use the
            :meth:`~.SessionEvents.before_flush`,
            :meth:`~.SessionEvents.after_flush`, or
            :meth:`~.SessionEvents.after_flush_postexec`
            events.

        :param session: The target :class:`.Session`.

        .. seealso::

            :meth:`~.SessionEvents.after_commit`

            :meth:`~.SessionEvents.after_begin`

            :meth:`~.SessionEvents.after_transaction_create`

            :meth:`~.SessionEvents.after_transaction_end`

        """

    def after_commit(self, session: Session) -> None:
        """Execute after a commit has occurred.

        .. note::

            The :meth:`~.SessionEvents.after_commit` hook is *not* per-flush,
            that is, the :class:`.Session` can emit SQL to the database
            many times within the scope of a transaction.
            For interception of these events, use the
            :meth:`~.SessionEvents.before_flush`,
            :meth:`~.SessionEvents.after_flush`, or
            :meth:`~.SessionEvents.after_flush_postexec`
            events.

        .. note::

            The :class:`.Session` is not in an active transaction
            when the :meth:`~.SessionEvents.after_commit` event is invoked,
            and therefore can not emit SQL.  To emit SQL corresponding to
            every transaction, use the :meth:`~.SessionEvents.before_commit`
            event.

        :param session: The target :class:`.Session`.

        .. seealso::

            :meth:`~.SessionEvents.before_commit`

            :meth:`~.SessionEvents.after_begin`

            :meth:`~.SessionEvents.after_transaction_create`

            :meth:`~.SessionEvents.after_transaction_end`

        """

    def after_rollback(self, session: Session) -> None:
        """Execute after a real DBAPI rollback has occurred.

        Note that this event only fires when the *actual* rollback against
        the database occurs - it does *not* fire each time the
        :meth:`.Session.rollback` method is called, if the underlying
        DBAPI transaction has already been rolled back.  In many
        cases, the :class:`.Session` will not be in
        an "active" state during this event, as the current
        transaction is not valid.   To acquire a :class:`.Session`
        which is active after the outermost rollback has proceeded,
        use the :meth:`.SessionEvents.after_soft_rollback` event, checking the
        :attr:`.Session.is_active` flag.

        :param session: The target :class:`.Session`.

        """

    def after_soft_rollback(
        self, session: Session, previous_transaction: SessionTransaction
    ) -> None:
        """Execute after any rollback has occurred, including "soft"
        rollbacks that don't actually emit at the DBAPI level.

        This corresponds to both nested and outer rollbacks, i.e.
        the innermost rollback that calls the DBAPI's
        rollback() method, as well as the enclosing rollback
        calls that only pop themselves from the transaction stack.

        The given :class:`.Session` can be used to invoke SQL and
        :meth:`.Session.query` operations after an outermost rollback
        by first checking the :attr:`.Session.is_active` flag::

            @event.listens_for(Session, "after_soft_rollback")
            def do_something(session, previous_transaction):
                if session.is_active:
                    session.execute(text("select * from some_table"))

        :param session: The target :class:`.Session`.
        :param previous_transaction: The :class:`.SessionTransaction`
         transactional marker object which was just closed.   The current
         :class:`.SessionTransaction` for the given :class:`.Session` is
         available via the :attr:`.Session.transaction` attribute.

        """

    def before_flush(
        self,
        session: Session,
        flush_context: UOWTransaction,
        instances: Optional[Sequence[_O]],
    ) -> None:
        """Execute before flush process has started.

        :param session: The target :class:`.Session`.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.
        :param instances: Usually ``None``, this is the collection of
         objects which can be passed to the :meth:`.Session.flush` method
         (note this usage is deprecated).

        .. seealso::

            :meth:`~.SessionEvents.after_flush`

            :meth:`~.SessionEvents.after_flush_postexec`

            :ref:`session_persistence_events`

        """

    def after_flush(
        self, session: Session, flush_context: UOWTransaction
    ) -> None:
        """Execute after flush has completed, but before commit has been
        called.

        Note that the session's state is still in pre-flush, i.e. 'new',
        'dirty', and 'deleted' lists still show pre-flush state as well
        as the history settings on instance attributes.

        .. warning:: This event runs after the :class:`.Session` has emitted
           SQL to modify the database, but **before** it has altered its
           internal state to reflect those changes, including that newly
           inserted objects are placed into the identity map.  ORM operations
           emitted within this event such as loads of related items
           may produce new identity map entries that will immediately
           be replaced, sometimes causing confusing results.  SQLAlchemy will
           emit a warning for this condition as of version 1.3.9.

        :param session: The target :class:`.Session`.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.

        .. seealso::

            :meth:`~.SessionEvents.before_flush`

            :meth:`~.SessionEvents.after_flush_postexec`

            :ref:`session_persistence_events`

        """

    def after_flush_postexec(
        self, session: Session, flush_context: UOWTransaction
    ) -> None:
        """Execute after flush has completed, and after the post-exec
        state occurs.

        This will be when the 'new', 'dirty', and 'deleted' lists are in
        their final state.  An actual commit() may or may not have
        occurred, depending on whether or not the flush started its own
        transaction or participated in a larger transaction.

        :param session: The target :class:`.Session`.
        :param flush_context: Internal :class:`.UOWTransaction` object
         which handles the details of the flush.


        .. seealso::

            :meth:`~.SessionEvents.before_flush`

            :meth:`~.SessionEvents.after_flush`

            :ref:`session_persistence_events`

        """

    def after_begin(
        self,
        session: Session,
        transaction: SessionTransaction,
        connection: Connection,
    ) -> None:
        """Execute after a transaction is begun on a connection.

        .. note:: This event is called within the process of the
          :class:`_orm.Session` modifying its own internal state.
          To invoke SQL operations within this hook, use the
          :class:`_engine.Connection` provided to the event;
          do not run SQL operations using the :class:`_orm.Session`
          directly.

        :param session: The target :class:`.Session`.
        :param transaction: The :class:`.SessionTransaction`.
        :param connection: The :class:`_engine.Connection` object
         which will be used for SQL statements.

        .. seealso::

            :meth:`~.SessionEvents.before_commit`

            :meth:`~.SessionEvents.after_commit`

            :meth:`~.SessionEvents.after_transaction_create`

            :meth:`~.SessionEvents.after_transaction_end`

        """

    @_lifecycle_event
    def before_attach(self, session: Session, instance: _O) -> None:
        """Execute before an instance is attached to a session.

        This is called before an add, delete or merge causes
        the object to be part of the session.

        .. seealso::

            :meth:`~.SessionEvents.after_attach`

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def after_attach(self, session: Session, instance: _O) -> None:
        """Execute after an instance is attached to a session.

        This is called after an add, delete or merge.

        .. note::

           As of 0.8, this event fires off *after* the item
           has been fully associated with the session, which is
           different than previous releases.  For event
           handlers that require the object not yet
           be part of session state (such as handlers which
           may autoflush while the target object is not
           yet complete) consider the
           new :meth:`.before_attach` event.

        .. seealso::

            :meth:`~.SessionEvents.before_attach`

            :ref:`session_lifecycle_events`

        """

    @event._legacy_signature(
        "0.9",
        ["session", "query", "query_context", "result"],
        lambda update_context: (
            update_context.session,
            update_context.query,
            None,
            update_context.result,
        ),
    )
    def after_bulk_update(self, update_context: _O) -> None:
        """Event for after the legacy :meth:`_orm.Query.update` method
        has been called.

        .. legacy:: The :meth:`_orm.SessionEvents.after_bulk_update` method
           is a legacy event hook as of SQLAlchemy 2.0.   The event
           **does not participate** in :term:`2.0 style` invocations
           using :func:`_dml.update` documented at
           :ref:`orm_queryguide_update_delete_where`.  For 2.0 style use,
           the :meth:`_orm.SessionEvents.do_orm_execute` hook will intercept
           these calls.

        :param update_context: an "update context" object which contains
         details about the update, including these attributes:

            * ``session`` - the :class:`.Session` involved
            * ``query`` -the :class:`_query.Query`
              object that this update operation
              was called upon.
            * ``values`` The "values" dictionary that was passed to
              :meth:`_query.Query.update`.
            * ``result`` the :class:`_engine.CursorResult`
              returned as a result of the
              bulk UPDATE operation.

        .. versionchanged:: 1.4 the update_context no longer has a
           ``QueryContext`` object associated with it.

        .. seealso::

            :meth:`.QueryEvents.before_compile_update`

            :meth:`.SessionEvents.after_bulk_delete`

        """

    @event._legacy_signature(
        "0.9",
        ["session", "query", "query_context", "result"],
        lambda delete_context: (
            delete_context.session,
            delete_context.query,
            None,
            delete_context.result,
        ),
    )
    def after_bulk_delete(self, delete_context: _O) -> None:
        """Event for after the legacy :meth:`_orm.Query.delete` method
        has been called.

        .. legacy:: The :meth:`_orm.SessionEvents.after_bulk_delete` method
           is a legacy event hook as of SQLAlchemy 2.0.   The event
           **does not participate** in :term:`2.0 style` invocations
           using :func:`_dml.delete` documented at
           :ref:`orm_queryguide_update_delete_where`.  For 2.0 style use,
           the :meth:`_orm.SessionEvents.do_orm_execute` hook will intercept
           these calls.

        :param delete_context: a "delete context" object which contains
         details about the update, including these attributes:

            * ``session`` - the :class:`.Session` involved
            * ``query`` -the :class:`_query.Query`
              object that this update operation
              was called upon.
            * ``result`` the :class:`_engine.CursorResult`
              returned as a result of the
              bulk DELETE operation.

        .. versionchanged:: 1.4 the update_context no longer has a
           ``QueryContext`` object associated with it.

        .. seealso::

            :meth:`.QueryEvents.before_compile_delete`

            :meth:`.SessionEvents.after_bulk_update`

        """

    @_lifecycle_event
    def transient_to_pending(self, session: Session, instance: _O) -> None:
        """Intercept the "transient to pending" transition for a specific
        object.

        This event is a specialization of the
        :meth:`.SessionEvents.after_attach` event which is only invoked
        for this specific transition.  It is invoked typically during the
        :meth:`.Session.add` call.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def pending_to_transient(self, session: Session, instance: _O) -> None:
        """Intercept the "pending to transient" transition for a specific
        object.

        This less common transition occurs when an pending object that has
        not been flushed is evicted from the session; this can occur
        when the :meth:`.Session.rollback` method rolls back the transaction,
        or when the :meth:`.Session.expunge` method is used.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def persistent_to_transient(self, session: Session, instance: _O) -> None:
        """Intercept the "persistent to transient" transition for a specific
        object.

        This less common transition occurs when an pending object that has
        has been flushed is evicted from the session; this can occur
        when the :meth:`.Session.rollback` method rolls back the transaction.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def pending_to_persistent(self, session: Session, instance: _O) -> None:
        """Intercept the "pending to persistent"" transition for a specific
        object.

        This event is invoked within the flush process, and is
        similar to scanning the :attr:`.Session.new` collection within
        the :meth:`.SessionEvents.after_flush` event.  However, in this
        case the object has already been moved to the persistent state
        when the event is called.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def detached_to_persistent(self, session: Session, instance: _O) -> None:
        """Intercept the "detached to persistent" transition for a specific
        object.

        This event is a specialization of the
        :meth:`.SessionEvents.after_attach` event which is only invoked
        for this specific transition.  It is invoked typically during the
        :meth:`.Session.add` call, as well as during the
        :meth:`.Session.delete` call if the object was not previously
        associated with the
        :class:`.Session` (note that an object marked as "deleted" remains
        in the "persistent" state until the flush proceeds).

        .. note::

            If the object becomes persistent as part of a call to
            :meth:`.Session.delete`, the object is **not** yet marked as
            deleted when this event is called.  To detect deleted objects,
            check the ``deleted`` flag sent to the
            :meth:`.SessionEvents.persistent_to_detached` to event after the
            flush proceeds, or check the :attr:`.Session.deleted` collection
            within the :meth:`.SessionEvents.before_flush` event if deleted
            objects need to be intercepted before the flush.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def loaded_as_persistent(self, session: Session, instance: _O) -> None:
        """Intercept the "loaded as persistent" transition for a specific
        object.

        This event is invoked within the ORM loading process, and is invoked
        very similarly to the :meth:`.InstanceEvents.load` event.  However,
        the event here is linkable to a :class:`.Session` class or instance,
        rather than to a mapper or class hierarchy, and integrates
        with the other session lifecycle events smoothly.  The object
        is guaranteed to be present in the session's identity map when
        this event is called.

        .. note:: This event is invoked within the loader process before
           eager loaders may have been completed, and the object's state may
           not be complete.  Additionally, invoking row-level refresh
           operations on the object will place the object into a new loader
           context, interfering with the existing load context.   See the note
           on :meth:`.InstanceEvents.load` for background on making use of the
           :paramref:`.SessionEvents.restore_load_context` parameter, which
           works in the same manner as that of
           :paramref:`.InstanceEvents.restore_load_context`, in  order to
           resolve this scenario.

        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def persistent_to_deleted(self, session: Session, instance: _O) -> None:
        """Intercept the "persistent to deleted" transition for a specific
        object.

        This event is invoked when a persistent object's identity
        is deleted from the database within a flush, however the object
        still remains associated with the :class:`.Session` until the
        transaction completes.

        If the transaction is rolled back, the object moves again
        to the persistent state, and the
        :meth:`.SessionEvents.deleted_to_persistent` event is called.
        If the transaction is committed, the object becomes detached,
        which will emit the :meth:`.SessionEvents.deleted_to_detached`
        event.

        Note that while the :meth:`.Session.delete` method is the primary
        public interface to mark an object as deleted, many objects
        get deleted due to cascade rules, which are not always determined
        until flush time.  Therefore, there's no way to catch
        every object that will be deleted until the flush has proceeded.
        the :meth:`.SessionEvents.persistent_to_deleted` event is therefore
        invoked at the end of a flush.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def deleted_to_persistent(self, session: Session, instance: _O) -> None:
        """Intercept the "deleted to persistent" transition for a specific
        object.

        This transition occurs only when an object that's been deleted
        successfully in a flush is restored due to a call to
        :meth:`.Session.rollback`.   The event is not called under
        any other circumstances.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def deleted_to_detached(self, session: Session, instance: _O) -> None:
        """Intercept the "deleted to detached" transition for a specific
        object.

        This event is invoked when a deleted object is evicted
        from the session.   The typical case when this occurs is when
        the transaction for a :class:`.Session` in which the object
        was deleted is committed; the object moves from the deleted
        state to the detached state.

        It is also invoked for objects that were deleted in a flush
        when the :meth:`.Session.expunge_all` or :meth:`.Session.close`
        events are called, as well as if the object is individually
        expunged from its deleted state via :meth:`.Session.expunge`.

        .. seealso::

            :ref:`session_lifecycle_events`

        """

    @_lifecycle_event
    def persistent_to_detached(self, session: Session, instance: _O) -> None:
        """Intercept the "persistent to detached" transition for a specific
        object.

        This event is invoked when a persistent object is evicted
        from the session.  There are many conditions that cause this
        to happen, including:

        * using a method such as :meth:`.Session.expunge`
          or :meth:`.Session.close`

        * Calling the :meth:`.Session.rollback` method, when the object
          was part of an INSERT statement for that session's transaction


        :param session: target :class:`.Session`

        :param instance: the ORM-mapped instance being operated upon.

        :param deleted: boolean.  If True, indicates this object moved
         to the detached state because it was marked as deleted and flushed.


        .. seealso::

            :ref:`session_lifecycle_events`

        """


class AttributeEvents(event.Events[QueryableAttribute[Any]]):
    r"""Define events for object attributes.

    These are typically defined on the class-bound descriptor for the
    target class.

    For example, to register a listener that will receive the
    :meth:`_orm.AttributeEvents.append` event::

        from sqlalchemy import event


        @event.listens_for(MyClass.collection, "append", propagate=True)
        def my_append_listener(target, value, initiator):
            print("received append event for target: %s" % target)

    Listeners have the option to return a possibly modified version of the
    value, when the :paramref:`.AttributeEvents.retval` flag is passed to
    :func:`.event.listen` or :func:`.event.listens_for`, such as below,
    illustrated using the :meth:`_orm.AttributeEvents.set` event::

        def validate_phone(target, value, oldvalue, initiator):
            "Strip non-numeric characters from a phone number"

            return re.sub(r"\D", "", value)


        # setup listener on UserContact.phone attribute, instructing
        # it to use the return value
        listen(UserContact.phone, "set", validate_phone, retval=True)

    A validation function like the above can also raise an exception
    such as :exc:`ValueError` to halt the operation.

    The :paramref:`.AttributeEvents.propagate` flag is also important when
    applying listeners to mapped classes that also have mapped subclasses,
    as when using mapper inheritance patterns::


        @event.listens_for(MySuperClass.attr, "set", propagate=True)
        def receive_set(target, value, initiator):
            print("value set: %s" % target)

    The full list of modifiers available to the :func:`.event.listen`
    and :func:`.event.listens_for` functions are below.

    :param active_history=False: When True, indicates that the
      "set" event would like to receive the "old" value being
      replaced unconditionally, even if this requires firing off
      database loads. Note that ``active_history`` can also be
      set directly via :func:`.column_property` and
      :func:`_orm.relationship`.

    :param propagate=False: When True, the listener function will
      be established not just for the class attribute given, but
      for attributes of the same name on all current subclasses
      of that class, as well as all future subclasses of that
      class, using an additional listener that listens for
      instrumentation events.
    :param raw=False: When True, the "target" argument to the
      event will be the :class:`.InstanceState` management
      object, rather than the mapped instance itself.
    :param retval=False: when True, the user-defined event
      listening must return the "value" argument from the
      function.  This gives the listening function the opportunity
      to change the value that is ultimately used for a "set"
      or "append" event.

    """

    _target_class_doc = "SomeClass.some_attribute"
    _dispatch_target = QueryableAttribute

    @staticmethod
    def _set_dispatch(
        cls: Type[_HasEventsDispatch[Any]], dispatch_cls: Type[_Dispatch[Any]]
    ) -> _Dispatch[Any]:
        dispatch = event.Events._set_dispatch(cls, dispatch_cls)
        dispatch_cls._active_history = False
        return dispatch

    @classmethod
    def _accept_with(
        cls,
        target: Union[QueryableAttribute[Any], Type[QueryableAttribute[Any]]],
        identifier: str,
    ) -> Union[QueryableAttribute[Any], Type[QueryableAttribute[Any]]]:
        # TODO: coverage
        if isinstance(target, interfaces.MapperProperty):
            return getattr(target.parent.class_, target.key)
        else:
            return target

    @classmethod
    def _listen(  # type: ignore [override]
        cls,
        event_key: _EventKey[QueryableAttribute[Any]],
        active_history: bool = False,
        raw: bool = False,
        retval: bool = False,
        propagate: bool = False,
        include_key: bool = False,
    ) -> None:
        target, fn = event_key.dispatch_target, event_key._listen_fn

        if active_history:
            target.dispatch._active_history = True

        if not raw or not retval or not include_key:

            def wrap(target: InstanceState[_O], *arg: Any, **kw: Any) -> Any:
                if not raw:
                    target = target.obj()  # type: ignore [assignment]
                if not retval:
                    if arg:
                        value = arg[0]
                    else:
                        value = None
                    if include_key:
                        fn(target, *arg, **kw)
                    else:
                        fn(target, *arg)
                    return value
                else:
                    if include_key:
                        return fn(target, *arg, **kw)
                    else:
                        return fn(target, *arg)

            event_key = event_key.with_wrapper(wrap)

        event_key.base_listen(propagate=propagate)

        if propagate:
            manager = instrumentation.manager_of_class(target.class_)

            for mgr in manager.subclass_managers(True):  # type: ignore [no-untyped-call] # noqa: E501
                event_key.with_dispatch_target(mgr[target.key]).base_listen(
                    propagate=True
                )
                if active_history:
                    mgr[target.key].dispatch._active_history = True

    def append(
        self,
        target: _O,
        value: _T,
        initiator: Event,
        *,
        key: EventConstants = NO_KEY,
    ) -> Optional[_T]:
        """Receive a collection append event.

        The append event is invoked for each element as it is appended
        to the collection.  This occurs for single-item appends as well
        as for a "bulk replace" operation.

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: the value being appended.  If this listener
          is registered with ``retval=True``, the listener
          function must return this value, or a new value which
          replaces it.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.  May be modified
          from its original value by backref handlers in order to control
          chained event propagation, as well as be inspected for information
          about the source of the event.
        :param key: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the key used in the operation, such as
         ``collection[some_key_or_index] = value``.
         The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        :return: if the event was registered with ``retval=True``,
         the given value, or a new effective value, should be returned.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

            :meth:`.AttributeEvents.bulk_replace`

        """

    def append_wo_mutation(
        self,
        target: _O,
        value: _T,
        initiator: Event,
        *,
        key: EventConstants = NO_KEY,
    ) -> None:
        """Receive a collection append event where the collection was not
        actually mutated.

        This event differs from :meth:`_orm.AttributeEvents.append` in that
        it is fired off for de-duplicating collections such as sets and
        dictionaries, when the object already exists in the target collection.
        The event does not have a return value and the identity of the
        given object cannot be changed.

        The event is used for cascading objects into a :class:`_orm.Session`
        when the collection has already been mutated via a backref event.

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: the value that would be appended if the object did not
          already exist in the collection.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.  May be modified
          from its original value by backref handlers in order to control
          chained event propagation, as well as be inspected for information
          about the source of the event.
        :param key: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the key used in the operation, such as
         ``collection[some_key_or_index] = value``.
         The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        :return: No return value is defined for this event.

        .. versionadded:: 1.4.15

        """

    def bulk_replace(
        self,
        target: _O,
        values: Iterable[_T],
        initiator: Event,
        *,
        keys: Optional[Iterable[EventConstants]] = None,
    ) -> None:
        """Receive a collection 'bulk replace' event.

        This event is invoked for a sequence of values as they are incoming
        to a bulk collection set operation, which can be
        modified in place before the values are treated as ORM objects.
        This is an "early hook" that runs before the bulk replace routine
        attempts to reconcile which objects are already present in the
        collection and which are being removed by the net replace operation.

        It is typical that this method be combined with use of the
        :meth:`.AttributeEvents.append` event.    When using both of these
        events, note that a bulk replace operation will invoke
        the :meth:`.AttributeEvents.append` event for all new items,
        even after :meth:`.AttributeEvents.bulk_replace` has been invoked
        for the collection as a whole.  In order to determine if an
        :meth:`.AttributeEvents.append` event is part of a bulk replace,
        use the symbol :attr:`~.attributes.OP_BULK_REPLACE` to test the
        incoming initiator::

            from sqlalchemy.orm.attributes import OP_BULK_REPLACE


            @event.listens_for(SomeObject.collection, "bulk_replace")
            def process_collection(target, values, initiator):
                values[:] = [_make_value(value) for value in values]


            @event.listens_for(SomeObject.collection, "append", retval=True)
            def process_collection(target, value, initiator):
                # make sure bulk_replace didn't already do it
                if initiator is None or initiator.op is not OP_BULK_REPLACE:
                    return _make_value(value)
                else:
                    return value

        .. versionadded:: 1.2

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: a sequence (e.g. a list) of the values being set.  The
          handler can modify this list in place.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.
        :param keys: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the sequence of keys used in the operation,
         typically only for a dictionary update.  The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.


        """

    def remove(
        self,
        target: _O,
        value: _T,
        initiator: Event,
        *,
        key: EventConstants = NO_KEY,
    ) -> None:
        """Receive a collection remove event.

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: the value being removed.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.  May be modified
          from its original value by backref handlers in order to control
          chained event propagation.

        :param key: When the event is established using the
         :paramref:`.AttributeEvents.include_key` parameter set to
         True, this will be the key used in the operation, such as
         ``del collection[some_key_or_index]``.  The parameter is not passed
         to the event at all if the the
         :paramref:`.AttributeEvents.include_key`
         was not used to set up the event; this is to allow backwards
         compatibility with existing event handlers that don't include the
         ``key`` parameter.

         .. versionadded:: 2.0

        :return: No return value is defined for this event.


        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

        """

    def set(
        self, target: _O, value: _T, oldvalue: _T, initiator: Event
    ) -> None:
        """Receive a scalar set event.

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.
        :param value: the value being set.  If this listener
          is registered with ``retval=True``, the listener
          function must return this value, or a new value which
          replaces it.
        :param oldvalue: the previous value being replaced.  This
          may also be the symbol ``NEVER_SET`` or ``NO_VALUE``.
          If the listener is registered with ``active_history=True``,
          the previous value of the attribute will be loaded from
          the database if the existing value is currently unloaded
          or expired.
        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.  May be modified
          from its original value by backref handlers in order to control
          chained event propagation.

        :return: if the event was registered with ``retval=True``,
         the given value, or a new effective value, should be returned.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

        """

    def init_scalar(
        self, target: _O, value: _T, dict_: Dict[Any, Any]
    ) -> None:
        r"""Receive a scalar "init" event.

        This event is invoked when an uninitialized, unpersisted scalar
        attribute is accessed, e.g. read::


            x = my_object.some_attribute

        The ORM's default behavior when this occurs for an un-initialized
        attribute is to return the value ``None``; note this differs from
        Python's usual behavior of raising ``AttributeError``.    The
        event here can be used to customize what value is actually returned,
        with the assumption that the event listener would be mirroring
        a default generator that is configured on the Core
        :class:`_schema.Column`
        object as well.

        Since a default generator on a :class:`_schema.Column`
        might also produce
        a changing value such as a timestamp, the
        :meth:`.AttributeEvents.init_scalar`
        event handler can also be used to **set** the newly returned value, so
        that a Core-level default generation function effectively fires off
        only once, but at the moment the attribute is accessed on the
        non-persisted object.   Normally, no change to the object's state
        is made when an uninitialized attribute is accessed (much older
        SQLAlchemy versions did in fact change the object's state).

        If a default generator on a column returned a particular constant,
        a handler might be used as follows::

            SOME_CONSTANT = 3.1415926


            class MyClass(Base):
                # ...

                some_attribute = Column(Numeric, default=SOME_CONSTANT)


            @event.listens_for(
                MyClass.some_attribute, "init_scalar", retval=True, propagate=True
            )
            def _init_some_attribute(target, dict_, value):
                dict_["some_attribute"] = SOME_CONSTANT
                return SOME_CONSTANT

        Above, we initialize the attribute ``MyClass.some_attribute`` to the
        value of ``SOME_CONSTANT``.   The above code includes the following
        features:

        * By setting the value ``SOME_CONSTANT`` in the given ``dict_``,
          we indicate that this value is to be persisted to the database.
          This supersedes the use of ``SOME_CONSTANT`` in the default generator
          for the :class:`_schema.Column`.  The ``active_column_defaults.py``
          example given at :ref:`examples_instrumentation` illustrates using
          the same approach for a changing default, e.g. a timestamp
          generator.    In this particular example, it is not strictly
          necessary to do this since ``SOME_CONSTANT`` would be part of the
          INSERT statement in either case.

        * By establishing the ``retval=True`` flag, the value we return
          from the function will be returned by the attribute getter.
          Without this flag, the event is assumed to be a passive observer
          and the return value of our function is ignored.

        * The ``propagate=True`` flag is significant if the mapped class
          includes inheriting subclasses, which would also make use of this
          event listener.  Without this flag, an inheriting subclass will
          not use our event handler.

        In the above example, the attribute set event
        :meth:`.AttributeEvents.set` as well as the related validation feature
        provided by :obj:`_orm.validates` is **not** invoked when we apply our
        value to the given ``dict_``.  To have these events to invoke in
        response to our newly generated value, apply the value to the given
        object as a normal attribute set operation::

            SOME_CONSTANT = 3.1415926


            @event.listens_for(
                MyClass.some_attribute, "init_scalar", retval=True, propagate=True
            )
            def _init_some_attribute(target, dict_, value):
                # will also fire off attribute set events
                target.some_attribute = SOME_CONSTANT
                return SOME_CONSTANT

        When multiple listeners are set up, the generation of the value
        is "chained" from one listener to the next by passing the value
        returned by the previous listener that specifies ``retval=True``
        as the ``value`` argument of the next listener.

        :param target: the object instance receiving the event.
         If the listener is registered with ``raw=True``, this will
         be the :class:`.InstanceState` object.
        :param value: the value that is to be returned before this event
         listener were invoked.  This value begins as the value ``None``,
         however will be the return value of the previous event handler
         function if multiple listeners are present.
        :param dict\_: the attribute dictionary of this mapped object.
         This is normally the ``__dict__`` of the object, but in all cases
         represents the destination that the attribute system uses to get
         at the actual value of this attribute.  Placing the value in this
         dictionary has the effect that the value will be used in the
         INSERT statement generated by the unit of work.


        .. seealso::

            :meth:`.AttributeEvents.init_collection` - collection version
            of this event

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

            :ref:`examples_instrumentation` - see the
            ``active_column_defaults.py`` example.

        """  # noqa: E501

    def init_collection(
        self,
        target: _O,
        collection: Type[Collection[Any]],
        collection_adapter: CollectionAdapter,
    ) -> None:
        """Receive a 'collection init' event.

        This event is triggered for a collection-based attribute, when
        the initial "empty collection" is first generated for a blank
        attribute, as well as for when the collection is replaced with
        a new one, such as via a set event.

        E.g., given that ``User.addresses`` is a relationship-based
        collection, the event is triggered here::

            u1 = User()
            u1.addresses.append(a1)  #  <- new collection

        and also during replace operations::

            u1.addresses = [a2, a3]  #  <- new collection

        :param target: the object instance receiving the event.
         If the listener is registered with ``raw=True``, this will
         be the :class:`.InstanceState` object.
        :param collection: the new collection.  This will always be generated
         from what was specified as
         :paramref:`_orm.relationship.collection_class`, and will always
         be empty.
        :param collection_adapter: the :class:`.CollectionAdapter` that will
         mediate internal access to the collection.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

            :meth:`.AttributeEvents.init_scalar` - "scalar" version of this
            event.

        """

    def dispose_collection(
        self,
        target: _O,
        collection: Collection[Any],
        collection_adapter: CollectionAdapter,
    ) -> None:
        """Receive a 'collection dispose' event.

        This event is triggered for a collection-based attribute when
        a collection is replaced, that is::

            u1.addresses.append(a1)

            u1.addresses = [a2, a3]  # <- old collection is disposed

        The old collection received will contain its previous contents.

        .. versionchanged:: 1.2 The collection passed to
           :meth:`.AttributeEvents.dispose_collection` will now have its
           contents before the dispose intact; previously, the collection
           would be empty.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

        """

    def modified(self, target: _O, initiator: Event) -> None:
        """Receive a 'modified' event.

        This event is triggered when the :func:`.attributes.flag_modified`
        function is used to trigger a modify event on an attribute without
        any specific value being set.

        .. versionadded:: 1.2

        :param target: the object instance receiving the event.
          If the listener is registered with ``raw=True``, this will
          be the :class:`.InstanceState` object.

        :param initiator: An instance of :class:`.attributes.Event`
          representing the initiation of the event.

        .. seealso::

            :class:`.AttributeEvents` - background on listener options such
            as propagation to subclasses.

        """


class QueryEvents(event.Events[Query[Any]]):
    """Represent events within the construction of a :class:`_query.Query`
    object.

    .. legacy:: The :class:`_orm.QueryEvents` event methods are legacy
        as of SQLAlchemy 2.0, and only apply to direct use of the
        :class:`_orm.Query` object. They are not used for :term:`2.0 style`
        statements. For events to intercept and modify 2.0 style ORM use,
        use the :meth:`_orm.SessionEvents.do_orm_execute` hook.


    The :class:`_orm.QueryEvents` hooks are now superseded by the
    :meth:`_orm.SessionEvents.do_orm_execute` event hook.

    """

    _target_class_doc = "SomeQuery"
    _dispatch_target = Query

    def before_compile(self, query: Query[Any]) -> None:
        """Receive the :class:`_query.Query`
        object before it is composed into a
        core :class:`_expression.Select` object.

        .. deprecated:: 1.4  The :meth:`_orm.QueryEvents.before_compile` event
           is superseded by the much more capable
           :meth:`_orm.SessionEvents.do_orm_execute` hook.   In version 1.4,
           the :meth:`_orm.QueryEvents.before_compile` event is **no longer
           used** for ORM-level attribute loads, such as loads of deferred
           or expired attributes as well as relationship loaders.   See the
           new examples in :ref:`examples_session_orm_events` which
           illustrate new ways of intercepting and modifying ORM queries
           for the most common purpose of adding arbitrary filter criteria.


        This event is intended to allow changes to the query given::

            @event.listens_for(Query, "before_compile", retval=True)
            def no_deleted(query):
                for desc in query.column_descriptions:
                    if desc["type"] is User:
                        entity = desc["entity"]
                        query = query.filter(entity.deleted == False)
                return query

        The event should normally be listened with the ``retval=True``
        parameter set, so that the modified query may be returned.

        The :meth:`.QueryEvents.before_compile` event by default
        will disallow "baked" queries from caching a query, if the event
        hook returns a new :class:`_query.Query` object.
        This affects both direct
        use of the baked query extension as well as its operation within
        lazy loaders and eager loaders for relationships.  In order to
        re-establish the query being cached, apply the event adding the
        ``bake_ok`` flag::

            @event.listens_for(Query, "before_compile", retval=True, bake_ok=True)
            def my_event(query):
                for desc in query.column_descriptions:
                    if desc["type"] is User:
                        entity = desc["entity"]
                        query = query.filter(entity.deleted == False)
                return query

        When ``bake_ok`` is set to True, the event hook will only be invoked
        once, and not called for subsequent invocations of a particular query
        that is being cached.

        .. versionadded:: 1.3.11  - added the "bake_ok" flag to the
           :meth:`.QueryEvents.before_compile` event and disallowed caching via
           the "baked" extension from occurring for event handlers that
           return  a new :class:`_query.Query` object if this flag is not set.

        .. seealso::

            :meth:`.QueryEvents.before_compile_update`

            :meth:`.QueryEvents.before_compile_delete`

            :ref:`baked_with_before_compile`

        """  # noqa: E501

    def before_compile_update(
        self, query: Query[Any], update_context: BulkUpdate
    ) -> None:
        """Allow modifications to the :class:`_query.Query` object within
        :meth:`_query.Query.update`.

        .. deprecated:: 1.4  The :meth:`_orm.QueryEvents.before_compile_update`
           event is superseded by the much more capable
           :meth:`_orm.SessionEvents.do_orm_execute` hook.

        Like the :meth:`.QueryEvents.before_compile` event, if the event
        is to be used to alter the :class:`_query.Query` object, it should
        be configured with ``retval=True``, and the modified
        :class:`_query.Query` object returned, as in ::

            @event.listens_for(Query, "before_compile_update", retval=True)
            def no_deleted(query, update_context):
                for desc in query.column_descriptions:
                    if desc["type"] is User:
                        entity = desc["entity"]
                        query = query.filter(entity.deleted == False)

                        update_context.values["timestamp"] = datetime.datetime.now(
                            datetime.UTC
                        )
                return query

        The ``.values`` dictionary of the "update context" object can also
        be modified in place as illustrated above.

        :param query: a :class:`_query.Query` instance; this is also
         the ``.query`` attribute of the given "update context"
         object.

        :param update_context: an "update context" object which is
         the same kind of object as described in
         :paramref:`.QueryEvents.after_bulk_update.update_context`.
         The object has a ``.values`` attribute in an UPDATE context which is
         the dictionary of parameters passed to :meth:`_query.Query.update`.
         This
         dictionary can be modified to alter the VALUES clause of the
         resulting UPDATE statement.

        .. versionadded:: 1.2.17

        .. seealso::

            :meth:`.QueryEvents.before_compile`

            :meth:`.QueryEvents.before_compile_delete`


        """  # noqa: E501

    def before_compile_delete(
        self, query: Query[Any], delete_context: BulkDelete
    ) -> None:
        """Allow modifications to the :class:`_query.Query` object within
        :meth:`_query.Query.delete`.

        .. deprecated:: 1.4  The :meth:`_orm.QueryEvents.before_compile_delete`
           event is superseded by the much more capable
           :meth:`_orm.SessionEvents.do_orm_execute` hook.

        Like the :meth:`.QueryEvents.before_compile` event, this event
        should be configured with ``retval=True``, and the modified
        :class:`_query.Query` object returned, as in ::

            @event.listens_for(Query, "before_compile_delete", retval=True)
            def no_deleted(query, delete_context):
                for desc in query.column_descriptions:
                    if desc["type"] is User:
                        entity = desc["entity"]
                        query = query.filter(entity.deleted == False)
                return query

        :param query: a :class:`_query.Query` instance; this is also
         the ``.query`` attribute of the given "delete context"
         object.

        :param delete_context: a "delete context" object which is
         the same kind of object as described in
         :paramref:`.QueryEvents.after_bulk_delete.delete_context`.

        .. versionadded:: 1.2.17

        .. seealso::

            :meth:`.QueryEvents.before_compile`

            :meth:`.QueryEvents.before_compile_update`


        """

    @classmethod
    def _listen(
        cls,
        event_key: _EventKey[_ET],
        retval: bool = False,
        bake_ok: bool = False,
        **kw: Any,
    ) -> None:
        fn = event_key._listen_fn

        if not retval:

            def wrap(*arg: Any, **kw: Any) -> Any:
                if not retval:
                    query = arg[0]
                    fn(*arg, **kw)
                    return query
                else:
                    return fn(*arg, **kw)

            event_key = event_key.with_wrapper(wrap)
        else:
            # don't assume we can apply an attribute to the callable
            def wrap(*arg: Any, **kw: Any) -> Any:
                return fn(*arg, **kw)

            event_key = event_key.with_wrapper(wrap)

        wrap._bake_ok = bake_ok  # type: ignore [attr-defined]

        event_key.base_listen(**kw)