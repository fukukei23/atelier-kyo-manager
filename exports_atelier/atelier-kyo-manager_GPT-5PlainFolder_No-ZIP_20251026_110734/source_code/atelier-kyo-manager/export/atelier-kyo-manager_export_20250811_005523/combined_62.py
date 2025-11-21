
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\dml.py ===
# sql/dml.py
# Copyright (C) 2009-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
"""
Provide :class:`_expression.Insert`, :class:`_expression.Update` and
:class:`_expression.Delete`.

"""
from __future__ import annotations

import collections.abc as collections_abc
import operator
from typing import Any
from typing import cast
from typing import Dict
from typing import Iterable
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import coercions
from . import roles
from . import util as sql_util
from ._typing import _TP
from ._typing import _unexpected_kw
from ._typing import is_column_element
from ._typing import is_named_from_clause
from .base import _entity_namespace_key
from .base import _exclusive_against
from .base import _from_objects
from .base import _generative
from .base import _select_iterables
from .base import ColumnCollection
from .base import ColumnSet
from .base import CompileState
from .base import DialectKWArgs
from .base import Executable
from .base import Generative
from .base import HasCompileState
from .elements import BooleanClauseList
from .elements import ClauseElement
from .elements import ColumnClause
from .elements import ColumnElement
from .elements import Null
from .selectable import Alias
from .selectable import ExecutableReturnsRows
from .selectable import FromClause
from .selectable import HasCTE
from .selectable import HasPrefixes
from .selectable import Join
from .selectable import SelectLabelStyle
from .selectable import TableClause
from .selectable import TypedReturnsRows
from .sqltypes import NullType
from .visitors import InternalTraversal
from .. import exc
from .. import util
from ..util.typing import Self
from ..util.typing import TypeGuard

if TYPE_CHECKING:
    from ._typing import _ColumnExpressionArgument
    from ._typing import _ColumnsClauseArgument
    from ._typing import _DMLColumnArgument
    from ._typing import _DMLColumnKeyMapping
    from ._typing import _DMLTableArgument
    from ._typing import _T0  # noqa
    from ._typing import _T1  # noqa
    from ._typing import _T2  # noqa
    from ._typing import _T3  # noqa
    from ._typing import _T4  # noqa
    from ._typing import _T5  # noqa
    from ._typing import _T6  # noqa
    from ._typing import _T7  # noqa
    from ._typing import _TypedColumnClauseArgument as _TCCA  # noqa
    from .base import ReadOnlyColumnCollection
    from .compiler import SQLCompiler
    from .elements import KeyedColumnElement
    from .selectable import _ColumnsClauseElement
    from .selectable import _SelectIterable
    from .selectable import Select
    from .selectable import Selectable

    def isupdate(dml: DMLState) -> TypeGuard[UpdateDMLState]: ...

    def isdelete(dml: DMLState) -> TypeGuard[DeleteDMLState]: ...

    def isinsert(dml: DMLState) -> TypeGuard[InsertDMLState]: ...

else:
    isupdate = operator.attrgetter("isupdate")
    isdelete = operator.attrgetter("isdelete")
    isinsert = operator.attrgetter("isinsert")


_T = TypeVar("_T", bound=Any)

_DMLColumnElement = Union[str, ColumnClause[Any]]
_DMLTableElement = Union[TableClause, Alias, Join]


class DMLState(CompileState):
    _no_parameters = True
    _dict_parameters: Optional[MutableMapping[_DMLColumnElement, Any]] = None
    _multi_parameters: Optional[
        List[MutableMapping[_DMLColumnElement, Any]]
    ] = None
    _ordered_values: Optional[List[Tuple[_DMLColumnElement, Any]]] = None
    _parameter_ordering: Optional[List[_DMLColumnElement]] = None
    _primary_table: FromClause
    _supports_implicit_returning = True

    isupdate = False
    isdelete = False
    isinsert = False

    statement: UpdateBase

    def __init__(
        self, statement: UpdateBase, compiler: SQLCompiler, **kw: Any
    ):
        raise NotImplementedError()

    @classmethod
    def get_entity_description(cls, statement: UpdateBase) -> Dict[str, Any]:
        return {
            "name": (
                statement.table.name
                if is_named_from_clause(statement.table)
                else None
            ),
            "table": statement.table,
        }

    @classmethod
    def get_returning_column_descriptions(
        cls, statement: UpdateBase
    ) -> List[Dict[str, Any]]:
        return [
            {
                "name": c.key,
                "type": c.type,
                "expr": c,
            }
            for c in statement._all_selected_columns
        ]

    @property
    def dml_table(self) -> _DMLTableElement:
        return self.statement.table

    if TYPE_CHECKING:

        @classmethod
        def get_plugin_class(cls, statement: Executable) -> Type[DMLState]: ...

    @classmethod
    def _get_multi_crud_kv_pairs(
        cls,
        statement: UpdateBase,
        multi_kv_iterator: Iterable[Dict[_DMLColumnArgument, Any]],
    ) -> List[Dict[_DMLColumnElement, Any]]:
        return [
            {
                coercions.expect(roles.DMLColumnRole, k): v
                for k, v in mapping.items()
            }
            for mapping in multi_kv_iterator
        ]

    @classmethod
    def _get_crud_kv_pairs(
        cls,
        statement: UpdateBase,
        kv_iterator: Iterable[Tuple[_DMLColumnArgument, Any]],
        needs_to_be_cacheable: bool,
    ) -> List[Tuple[_DMLColumnElement, Any]]:
        return [
            (
                coercions.expect(roles.DMLColumnRole, k),
                (
                    v
                    if not needs_to_be_cacheable
                    else coercions.expect(
                        roles.ExpressionElementRole,
                        v,
                        type_=NullType(),
                        is_crud=True,
                    )
                ),
            )
            for k, v in kv_iterator
        ]

    def _make_extra_froms(
        self, statement: DMLWhereBase
    ) -> Tuple[FromClause, List[FromClause]]:
        froms: List[FromClause] = []

        all_tables = list(sql_util.tables_from_leftmost(statement.table))
        primary_table = all_tables[0]
        seen = {primary_table}

        consider = statement._where_criteria
        if self._dict_parameters:
            consider += tuple(self._dict_parameters.values())

        for crit in consider:
            for item in _from_objects(crit):
                if not seen.intersection(item._cloned_set):
                    froms.append(item)
                seen.update(item._cloned_set)

        froms.extend(all_tables[1:])
        return primary_table, froms

    def _process_values(self, statement: ValuesBase) -> None:
        if self._no_parameters:
            self._dict_parameters = statement._values
            self._no_parameters = False

    def _process_select_values(self, statement: ValuesBase) -> None:
        assert statement._select_names is not None
        parameters: MutableMapping[_DMLColumnElement, Any] = {
            name: Null() for name in statement._select_names
        }

        if self._no_parameters:
            self._no_parameters = False
            self._dict_parameters = parameters
        else:
            # this condition normally not reachable as the Insert
            # does not allow this construction to occur
            assert False, "This statement already has parameters"

    def _no_multi_values_supported(self, statement: ValuesBase) -> NoReturn:
        raise exc.InvalidRequestError(
            "%s construct does not support "
            "multiple parameter sets." % statement.__visit_name__.upper()
        )

    def _cant_mix_formats_error(self) -> NoReturn:
        raise exc.InvalidRequestError(
            "Can't mix single and multiple VALUES "
            "formats in one INSERT statement; one style appends to a "
            "list while the other replaces values, so the intent is "
            "ambiguous."
        )


@CompileState.plugin_for("default", "insert")
class InsertDMLState(DMLState):
    isinsert = True

    include_table_with_column_exprs = False

    _has_multi_parameters = False

    def __init__(
        self,
        statement: Insert,
        compiler: SQLCompiler,
        disable_implicit_returning: bool = False,
        **kw: Any,
    ):
        self.statement = statement
        self._primary_table = statement.table

        if disable_implicit_returning:
            self._supports_implicit_returning = False

        self.isinsert = True
        if statement._select_names:
            self._process_select_values(statement)
        if statement._values is not None:
            self._process_values(statement)
        if statement._multi_values:
            self._process_multi_values(statement)

    @util.memoized_property
    def _insert_col_keys(self) -> List[str]:
        # this is also done in crud.py -> _key_getters_for_crud_column
        return [
            coercions.expect(roles.DMLColumnRole, col, as_key=True)
            for col in self._dict_parameters or ()
        ]

    def _process_values(self, statement: ValuesBase) -> None:
        if self._no_parameters:
            self._has_multi_parameters = False
            self._dict_parameters = statement._values
            self._no_parameters = False
        elif self._has_multi_parameters:
            self._cant_mix_formats_error()

    def _process_multi_values(self, statement: ValuesBase) -> None:
        for parameters in statement._multi_values:
            multi_parameters: List[MutableMapping[_DMLColumnElement, Any]] = [
                (
                    {
                        c.key: value
                        for c, value in zip(statement.table.c, parameter_set)
                    }
                    if isinstance(parameter_set, collections_abc.Sequence)
                    else parameter_set
                )
                for parameter_set in parameters
            ]

            if self._no_parameters:
                self._no_parameters = False
                self._has_multi_parameters = True
                self._multi_parameters = multi_parameters
                self._dict_parameters = self._multi_parameters[0]
            elif not self._has_multi_parameters:
                self._cant_mix_formats_error()
            else:
                assert self._multi_parameters
                self._multi_parameters.extend(multi_parameters)


@CompileState.plugin_for("default", "update")
class UpdateDMLState(DMLState):
    isupdate = True

    include_table_with_column_exprs = False

    def __init__(self, statement: Update, compiler: SQLCompiler, **kw: Any):
        self.statement = statement

        self.isupdate = True
        if statement._ordered_values is not None:
            self._process_ordered_values(statement)
        elif statement._values is not None:
            self._process_values(statement)
        elif statement._multi_values:
            self._no_multi_values_supported(statement)
        t, ef = self._make_extra_froms(statement)
        self._primary_table = t
        self._extra_froms = ef

        self.is_multitable = mt = ef
        self.include_table_with_column_exprs = bool(
            mt and compiler.render_table_with_column_in_update_from
        )

    def _process_ordered_values(self, statement: ValuesBase) -> None:
        parameters = statement._ordered_values

        if self._no_parameters:
            self._no_parameters = False
            assert parameters is not None
            self._dict_parameters = dict(parameters)
            self._ordered_values = parameters
            self._parameter_ordering = [key for key, value in parameters]
        else:
            raise exc.InvalidRequestError(
                "Can only invoke ordered_values() once, and not mixed "
                "with any other values() call"
            )


@CompileState.plugin_for("default", "delete")
class DeleteDMLState(DMLState):
    isdelete = True

    def __init__(self, statement: Delete, compiler: SQLCompiler, **kw: Any):
        self.statement = statement

        self.isdelete = True
        t, ef = self._make_extra_froms(statement)
        self._primary_table = t
        self._extra_froms = ef
        self.is_multitable = ef


class UpdateBase(
    roles.DMLRole,
    HasCTE,
    HasCompileState,
    DialectKWArgs,
    HasPrefixes,
    Generative,
    ExecutableReturnsRows,
    ClauseElement,
):
    """Form the base for ``INSERT``, ``UPDATE``, and ``DELETE`` statements."""

    __visit_name__ = "update_base"

    _hints: util.immutabledict[Tuple[_DMLTableElement, str], str] = (
        util.EMPTY_DICT
    )
    named_with_column = False

    _label_style: SelectLabelStyle = (
        SelectLabelStyle.LABEL_STYLE_DISAMBIGUATE_ONLY
    )
    table: _DMLTableElement

    _return_defaults = False
    _return_defaults_columns: Optional[Tuple[_ColumnsClauseElement, ...]] = (
        None
    )
    _supplemental_returning: Optional[Tuple[_ColumnsClauseElement, ...]] = None
    _returning: Tuple[_ColumnsClauseElement, ...] = ()

    is_dml = True

    def _generate_fromclause_column_proxies(
        self,
        fromclause: FromClause,
        columns: ColumnCollection[str, KeyedColumnElement[Any]],
        primary_key: ColumnSet,
        foreign_keys: Set[KeyedColumnElement[Any]],
    ) -> None:
        columns._populate_separate_keys(
            col._make_proxy(
                fromclause, primary_key=primary_key, foreign_keys=foreign_keys
            )
            for col in self._all_selected_columns
            if is_column_element(col)
        )

    def params(self, *arg: Any, **kw: Any) -> NoReturn:
        """Set the parameters for the statement.

        This method raises ``NotImplementedError`` on the base class,
        and is overridden by :class:`.ValuesBase` to provide the
        SET/VALUES clause of UPDATE and INSERT.

        """
        raise NotImplementedError(
            "params() is not supported for INSERT/UPDATE/DELETE statements."
            " To set the values for an INSERT or UPDATE statement, use"
            " stmt.values(**parameters)."
        )

    @_generative
    def with_dialect_options(self, **opt: Any) -> Self:
        """Add dialect options to this INSERT/UPDATE/DELETE object.

        e.g.::

            upd = table.update().dialect_options(mysql_limit=10)

        .. versionadded: 1.4 - this method supersedes the dialect options
           associated with the constructor.


        """
        self._validate_dialect_kwargs(opt)
        return self

    @_generative
    def return_defaults(
        self,
        *cols: _DMLColumnArgument,
        supplemental_cols: Optional[Iterable[_DMLColumnArgument]] = None,
        sort_by_parameter_order: bool = False,
    ) -> Self:
        """Make use of a :term:`RETURNING` clause for the purpose
        of fetching server-side expressions and defaults, for supporting
        backends only.

        .. deepalchemy::

            The :meth:`.UpdateBase.return_defaults` method is used by the ORM
            for its internal work in fetching newly generated primary key
            and server default values, in particular to provide the underyling
            implementation of the :paramref:`_orm.Mapper.eager_defaults`
            ORM feature as well as to allow RETURNING support with bulk
            ORM inserts.  Its behavior is fairly idiosyncratic
            and is not really intended for general use.  End users should
            stick with using :meth:`.UpdateBase.returning` in order to
            add RETURNING clauses to their INSERT, UPDATE and DELETE
            statements.

        Normally, a single row INSERT statement will automatically populate the
        :attr:`.CursorResult.inserted_primary_key` attribute when executed,
        which stores the primary key of the row that was just inserted in the
        form of a :class:`.Row` object with column names as named tuple keys
        (and the :attr:`.Row._mapping` view fully populated as well). The
        dialect in use chooses the strategy to use in order to populate this
        data; if it was generated using server-side defaults and / or SQL
        expressions, dialect-specific approaches such as ``cursor.lastrowid``
        or ``RETURNING`` are typically used to acquire the new primary key
        value.

        However, when the statement is modified by calling
        :meth:`.UpdateBase.return_defaults` before executing the statement,
        additional behaviors take place **only** for backends that support
        RETURNING and for :class:`.Table` objects that maintain the
        :paramref:`.Table.implicit_returning` parameter at its default value of
        ``True``. In these cases, when the :class:`.CursorResult` is returned
        from the statement's execution, not only will
        :attr:`.CursorResult.inserted_primary_key` be populated as always, the
        :attr:`.CursorResult.returned_defaults` attribute will also be
        populated with a :class:`.Row` named-tuple representing the full range
        of server generated
        values from that single row, including values for any columns that
        specify :paramref:`_schema.Column.server_default` or which make use of
        :paramref:`_schema.Column.default` using a SQL expression.

        When invoking INSERT statements with multiple rows using
        :ref:`insertmanyvalues <engine_insertmanyvalues>`, the
        :meth:`.UpdateBase.return_defaults` modifier will have the effect of
        the :attr:`_engine.CursorResult.inserted_primary_key_rows` and
        :attr:`_engine.CursorResult.returned_defaults_rows` attributes being
        fully populated with lists of :class:`.Row` objects representing newly
        inserted primary key values as well as newly inserted server generated
        values for each row inserted. The
        :attr:`.CursorResult.inserted_primary_key` and
        :attr:`.CursorResult.returned_defaults` attributes will also continue
        to be populated with the first row of these two collections.

        If the backend does not support RETURNING or the :class:`.Table` in use
        has disabled :paramref:`.Table.implicit_returning`, then no RETURNING
        clause is added and no additional data is fetched, however the
        INSERT, UPDATE or DELETE statement proceeds normally.

        E.g.::

            stmt = table.insert().values(data="newdata").return_defaults()

            result = connection.execute(stmt)

            server_created_at = result.returned_defaults["created_at"]

        When used against an UPDATE statement
        :meth:`.UpdateBase.return_defaults` instead looks for columns that
        include :paramref:`_schema.Column.onupdate` or
        :paramref:`_schema.Column.server_onupdate` parameters assigned, when
        constructing the columns that will be included in the RETURNING clause
        by default if explicit columns were not specified. When used against a
        DELETE statement, no columns are included in RETURNING by default, they
        instead must be specified explicitly as there are no columns that
        normally change values when a DELETE statement proceeds.

        .. versionadded:: 2.0  :meth:`.UpdateBase.return_defaults` is supported
           for DELETE statements also and has been moved from
           :class:`.ValuesBase` to :class:`.UpdateBase`.

        The :meth:`.UpdateBase.return_defaults` method is mutually exclusive
        against the :meth:`.UpdateBase.returning` method and errors will be
        raised during the SQL compilation process if both are used at the same
        time on one statement. The RETURNING clause of the INSERT, UPDATE or
        DELETE statement is therefore controlled by only one of these methods
        at a time.

        The :meth:`.UpdateBase.return_defaults` method differs from
        :meth:`.UpdateBase.returning` in these ways:

        1. :meth:`.UpdateBase.return_defaults` method causes the
           :attr:`.CursorResult.returned_defaults` collection to be populated
           with the first row from the RETURNING result. This attribute is not
           populated when using :meth:`.UpdateBase.returning`.

        2. :meth:`.UpdateBase.return_defaults` is compatible with existing
           logic used to fetch auto-generated primary key values that are then
           populated into the :attr:`.CursorResult.inserted_primary_key`
           attribute. By contrast, using :meth:`.UpdateBase.returning` will
           have the effect of the :attr:`.CursorResult.inserted_primary_key`
           attribute being left unpopulated.

        3. :meth:`.UpdateBase.return_defaults` can be called against any
           backend. Backends that don't support RETURNING will skip the usage
           of the feature, rather than raising an exception, *unless*
           ``supplemental_cols`` is passed. The return value
           of :attr:`_engine.CursorResult.returned_defaults` will be ``None``
           for backends that don't support RETURNING or for which the target
           :class:`.Table` sets :paramref:`.Table.implicit_returning` to
           ``False``.

        4. An INSERT statement invoked with executemany() is supported if the
           backend database driver supports the
           :ref:`insertmanyvalues <engine_insertmanyvalues>`
           feature which is now supported by most SQLAlchemy-included backends.
           When executemany is used, the
           :attr:`_engine.CursorResult.returned_defaults_rows` and
           :attr:`_engine.CursorResult.inserted_primary_key_rows` accessors
           will return the inserted defaults and primary keys.

           .. versionadded:: 1.4 Added
              :attr:`_engine.CursorResult.returned_defaults_rows` and
              :attr:`_engine.CursorResult.inserted_primary_key_rows` accessors.
              In version 2.0, the underlying implementation which fetches and
              populates the data for these attributes was generalized to be
              supported by most backends, whereas in 1.4 they were only
              supported by the ``psycopg2`` driver.


        :param cols: optional list of column key names or
         :class:`_schema.Column` that acts as a filter for those columns that
         will be fetched.
        :param supplemental_cols: optional list of RETURNING expressions,
          in the same form as one would pass to the
          :meth:`.UpdateBase.returning` method. When present, the additional
          columns will be included in the RETURNING clause, and the
          :class:`.CursorResult` object will be "rewound" when returned, so
          that methods like :meth:`.CursorResult.all` will return new rows
          mostly as though the statement used :meth:`.UpdateBase.returning`
          directly. However, unlike when using :meth:`.UpdateBase.returning`
          directly, the **order of the columns is undefined**, so can only be
          targeted using names or :attr:`.Row._mapping` keys; they cannot
          reliably be targeted positionally.

          .. versionadded:: 2.0

        :param sort_by_parameter_order: for a batch INSERT that is being
         executed against multiple parameter sets, organize the results of
         RETURNING so that the returned rows correspond to the order of
         parameter sets passed in.  This applies only to an :term:`executemany`
         execution for supporting dialects and typically makes use of the
         :term:`insertmanyvalues` feature.

         .. versionadded:: 2.0.10

         .. seealso::

            :ref:`engine_insertmanyvalues_returning_order` - background on
            sorting of RETURNING rows for bulk INSERT

        .. seealso::

            :meth:`.UpdateBase.returning`

            :attr:`_engine.CursorResult.returned_defaults`

            :attr:`_engine.CursorResult.returned_defaults_rows`

            :attr:`_engine.CursorResult.inserted_primary_key`

            :attr:`_engine.CursorResult.inserted_primary_key_rows`

        """

        if self._return_defaults:
            # note _return_defaults_columns = () means return all columns,
            # so if we have been here before, only update collection if there
            # are columns in the collection
            if self._return_defaults_columns and cols:
                self._return_defaults_columns = tuple(
                    util.OrderedSet(self._return_defaults_columns).union(
                        coercions.expect(roles.ColumnsClauseRole, c)
                        for c in cols
                    )
                )
            else:
                # set for all columns
                self._return_defaults_columns = ()
        else:
            self._return_defaults_columns = tuple(
                coercions.expect(roles.ColumnsClauseRole, c) for c in cols
            )
        self._return_defaults = True
        if sort_by_parameter_order:
            if not self.is_insert:
                raise exc.ArgumentError(
                    "The 'sort_by_parameter_order' argument to "
                    "return_defaults() only applies to INSERT statements"
                )
            self._sort_by_parameter_order = True
        if supplemental_cols:
            # uniquifying while also maintaining order (the maintain of order
            # is for test suites but also for vertical splicing
            supplemental_col_tup = (
                coercions.expect(roles.ColumnsClauseRole, c)
                for c in supplemental_cols
            )

            if self._supplemental_returning is None:
                self._supplemental_returning = tuple(
                    util.unique_list(supplemental_col_tup)
                )
            else:
                self._supplemental_returning = tuple(
                    util.unique_list(
                        self._supplemental_returning
                        + tuple(supplemental_col_tup)
                    )
                )

        return self

    def is_derived_from(self, fromclause: Optional[FromClause]) -> bool:
        """Return ``True`` if this :class:`.ReturnsRows` is
        'derived' from the given :class:`.FromClause`.

        Since these are DMLs, we dont want such statements ever being adapted
        so we return False for derives.

        """
        return False

    @_generative
    def returning(
        self,
        *cols: _ColumnsClauseArgument[Any],
        sort_by_parameter_order: bool = False,
        **__kw: Any,
    ) -> UpdateBase:
        r"""Add a :term:`RETURNING` or equivalent clause to this statement.

        e.g.:

        .. sourcecode:: pycon+sql

            >>> stmt = (
            ...     table.update()
            ...     .where(table.c.data == "value")
            ...     .values(status="X")
            ...     .returning(table.c.server_flag, table.c.updated_timestamp)
            ... )
            >>> print(stmt)
            {printsql}UPDATE some_table SET status=:status
            WHERE some_table.data = :data_1
            RETURNING some_table.server_flag, some_table.updated_timestamp

        The method may be invoked multiple times to add new entries to the
        list of expressions to be returned.

        .. versionadded:: 1.4.0b2 The method may be invoked multiple times to
         add new entries to the list of expressions to be returned.

        The given collection of column expressions should be derived from the
        table that is the target of the INSERT, UPDATE, or DELETE.  While
        :class:`_schema.Column` objects are typical, the elements can also be
        expressions:

        .. sourcecode:: pycon+sql

            >>> stmt = table.insert().returning(
            ...     (table.c.first_name + " " + table.c.last_name).label("fullname")
            ... )
            >>> print(stmt)
            {printsql}INSERT INTO some_table (first_name, last_name)
            VALUES (:first_name, :last_name)
            RETURNING some_table.first_name || :first_name_1 || some_table.last_name AS fullname

        Upon compilation, a RETURNING clause, or database equivalent,
        will be rendered within the statement.   For INSERT and UPDATE,
        the values are the newly inserted/updated values.  For DELETE,
        the values are those of the rows which were deleted.

        Upon execution, the values of the columns to be returned are made
        available via the result set and can be iterated using
        :meth:`_engine.CursorResult.fetchone` and similar.
        For DBAPIs which do not
        natively support returning values (i.e. cx_oracle), SQLAlchemy will
        approximate this behavior at the result level so that a reasonable
        amount of behavioral neutrality is provided.

        Note that not all databases/DBAPIs
        support RETURNING.   For those backends with no support,
        an exception is raised upon compilation and/or execution.
        For those who do support it, the functionality across backends
        varies greatly, including restrictions on executemany()
        and other statements which return multiple rows. Please
        read the documentation notes for the database in use in
        order to determine the availability of RETURNING.

        :param \*cols: series of columns, SQL expressions, or whole tables
         entities to be returned.
        :param sort_by_parameter_order: for a batch INSERT that is being
         executed against multiple parameter sets, organize the results of
         RETURNING so that the returned rows correspond to the order of
         parameter sets passed in.  This applies only to an :term:`executemany`
         execution for supporting dialects and typically makes use of the
         :term:`insertmanyvalues` feature.

         .. versionadded:: 2.0.10

         .. seealso::

            :ref:`engine_insertmanyvalues_returning_order` - background on
            sorting of RETURNING rows for bulk INSERT (Core level discussion)

            :ref:`orm_queryguide_bulk_insert_returning_ordered` - example of
            use with :ref:`orm_queryguide_bulk_insert` (ORM level discussion)

        .. seealso::

          :meth:`.UpdateBase.return_defaults` - an alternative method tailored
          towards efficient fetching of server-side defaults and triggers
          for single-row INSERTs or UPDATEs.

          :ref:`tutorial_insert_returning` - in the :ref:`unified_tutorial`

        """  # noqa: E501
        if __kw:
            raise _unexpected_kw("UpdateBase.returning()", __kw)
        if self._return_defaults:
            raise exc.InvalidRequestError(
                "return_defaults() is already configured on this statement"
            )
        self._returning += tuple(
            coercions.expect(roles.ColumnsClauseRole, c) for c in cols
        )
        if sort_by_parameter_order:
            if not self.is_insert:
                raise exc.ArgumentError(
                    "The 'sort_by_parameter_order' argument to returning() "
                    "only applies to INSERT statements"
                )
            self._sort_by_parameter_order = True
        return self

    def corresponding_column(
        self, column: KeyedColumnElement[Any], require_embedded: bool = False
    ) -> Optional[ColumnElement[Any]]:
        return self.exported_columns.corresponding_column(
            column, require_embedded=require_embedded
        )

    @util.ro_memoized_property
    def _all_selected_columns(self) -> _SelectIterable:
        return [c for c in _select_iterables(self._returning)]

    @util.ro_memoized_property
    def exported_columns(
        self,
    ) -> ReadOnlyColumnCollection[Optional[str], ColumnElement[Any]]:
        """Return the RETURNING columns as a column collection for this
        statement.

        .. versionadded:: 1.4

        """
        return ColumnCollection(
            (c.key, c)
            for c in self._all_selected_columns
            if is_column_element(c)
        ).as_readonly()

    @_generative
    def with_hint(
        self,
        text: str,
        selectable: Optional[_DMLTableArgument] = None,
        dialect_name: str = "*",
    ) -> Self:
        """Add a table hint for a single table to this
        INSERT/UPDATE/DELETE statement.

        .. note::

         :meth:`.UpdateBase.with_hint` currently applies only to
         Microsoft SQL Server.  For MySQL INSERT/UPDATE/DELETE hints, use
         :meth:`.UpdateBase.prefix_with`.

        The text of the hint is rendered in the appropriate
        location for the database backend in use, relative
        to the :class:`_schema.Table` that is the subject of this
        statement, or optionally to that of the given
        :class:`_schema.Table` passed as the ``selectable`` argument.

        The ``dialect_name`` option will limit the rendering of a particular
        hint to a particular backend. Such as, to add a hint
        that only takes effect for SQL Server::

            mytable.insert().with_hint("WITH (PAGLOCK)", dialect_name="mssql")

        :param text: Text of the hint.
        :param selectable: optional :class:`_schema.Table` that specifies
         an element of the FROM clause within an UPDATE or DELETE
         to be the subject of the hint - applies only to certain backends.
        :param dialect_name: defaults to ``*``, if specified as the name
         of a particular dialect, will apply these hints only when
         that dialect is in use.
        """
        if selectable is None:
            selectable = self.table
        else:
            selectable = coercions.expect(roles.DMLTableRole, selectable)
        self._hints = self._hints.union({(selectable, dialect_name): text})
        return self

    @property
    def entity_description(self) -> Dict[str, Any]:
        """Return a :term:`plugin-enabled` description of the table and/or
        entity which this DML construct is operating against.

        This attribute is generally useful when using the ORM, as an
        extended structure which includes information about mapped
        entities is returned.  The section :ref:`queryguide_inspection`
        contains more background.

        For a Core statement, the structure returned by this accessor
        is derived from the :attr:`.UpdateBase.table` attribute, and
        refers to the :class:`.Table` being inserted, updated, or deleted::

            >>> stmt = insert(user_table)
            >>> stmt.entity_description
            {
                "name": "user_table",
                "table": Table("user_table", ...)
            }

        .. versionadded:: 1.4.33

        .. seealso::

            :attr:`.UpdateBase.returning_column_descriptions`

            :attr:`.Select.column_descriptions` - entity information for
            a :func:`.select` construct

            :ref:`queryguide_inspection` - ORM background

        """
        meth = DMLState.get_plugin_class(self).get_entity_description
        return meth(self)

    @property
    def returning_column_descriptions(self) -> List[Dict[str, Any]]:
        """Return a :term:`plugin-enabled` description of the columns
        which this DML construct is RETURNING against, in other words
        the expressions established as part of :meth:`.UpdateBase.returning`.

        This attribute is generally useful when using the ORM, as an
        extended structure which includes information about mapped
        entities is returned.  The section :ref:`queryguide_inspection`
        contains more background.

        For a Core statement, the structure returned by this accessor is
        derived from the same objects that are returned by the
        :attr:`.UpdateBase.exported_columns` accessor::

            >>> stmt = insert(user_table).returning(user_table.c.id, user_table.c.name)
            >>> stmt.entity_description
            [
                {
                    "name": "id",
                    "type": Integer,
                    "expr": Column("id", Integer(), table=<user>, ...)
                },
                {
                    "name": "name",
                    "type": String(),
                    "expr": Column("name", String(), table=<user>, ...)
                },
            ]

        .. versionadded:: 1.4.33

        .. seealso::

            :attr:`.UpdateBase.entity_description`

            :attr:`.Select.column_descriptions` - entity information for
            a :func:`.select` construct

            :ref:`queryguide_inspection` - ORM background

        """  # noqa: E501
        meth = DMLState.get_plugin_class(
            self
        ).get_returning_column_descriptions
        return meth(self)


class ValuesBase(UpdateBase):
    """Supplies support for :meth:`.ValuesBase.values` to
    INSERT and UPDATE constructs."""

    __visit_name__ = "values_base"

    _supports_multi_parameters = False

    select: Optional[Select[Any]] = None
    """SELECT statement for INSERT .. FROM SELECT"""

    _post_values_clause: Optional[ClauseElement] = None
    """used by extensions to Insert etc. to add additional syntacitcal
    constructs, e.g. ON CONFLICT etc."""

    _values: Optional[util.immutabledict[_DMLColumnElement, Any]] = None
    _multi_values: Tuple[
        Union[
            Sequence[Dict[_DMLColumnElement, Any]],
            Sequence[Sequence[Any]],
        ],
        ...,
    ] = ()

    _ordered_values: Optional[List[Tuple[_DMLColumnElement, Any]]] = None

    _select_names: Optional[List[str]] = None
    _inline: bool = False

    def __init__(self, table: _DMLTableArgument):
        self.table = coercions.expect(
            roles.DMLTableRole, table, apply_propagate_attrs=self
        )

    @_generative
    @_exclusive_against(
        "_select_names",
        "_ordered_values",
        msgs={
            "_select_names": "This construct already inserts from a SELECT",
            "_ordered_values": "This statement already has ordered "
            "values present",
        },
    )
    def values(
        self,
        *args: Union[
            _DMLColumnKeyMapping[Any],
            Sequence[Any],
        ],
        **kwargs: Any,
    ) -> Self:
        r"""Specify a fixed VALUES clause for an INSERT statement, or the SET
        clause for an UPDATE.

        Note that the :class:`_expression.Insert` and
        :class:`_expression.Update`
        constructs support
        per-execution time formatting of the VALUES and/or SET clauses,
        based on the arguments passed to :meth:`_engine.Connection.execute`.
        However, the :meth:`.ValuesBase.values` method can be used to "fix" a
        particular set of parameters into the statement.

        Multiple calls to :meth:`.ValuesBase.values` will produce a new
        construct, each one with the parameter list modified to include
        the new parameters sent.  In the typical case of a single
        dictionary of parameters, the newly passed keys will replace
        the same keys in the previous construct.  In the case of a list-based
        "multiple values" construct, each new list of values is extended
        onto the existing list of values.

        :param \**kwargs: key value pairs representing the string key
          of a :class:`_schema.Column`
          mapped to the value to be rendered into the
          VALUES or SET clause::

                users.insert().values(name="some name")

                users.update().where(users.c.id == 5).values(name="some name")

        :param \*args: As an alternative to passing key/value parameters,
         a dictionary, tuple, or list of dictionaries or tuples can be passed
         as a single positional argument in order to form the VALUES or
         SET clause of the statement.  The forms that are accepted vary
         based on whether this is an :class:`_expression.Insert` or an
         :class:`_expression.Update` construct.

         For either an :class:`_expression.Insert` or
         :class:`_expression.Update`
         construct, a single dictionary can be passed, which works the same as
         that of the kwargs form::

            users.insert().values({"name": "some name"})

            users.update().values({"name": "some new name"})

         Also for either form but more typically for the
         :class:`_expression.Insert` construct, a tuple that contains an
         entry for every column in the table is also accepted::

            users.insert().values((5, "some name"))

         The :class:`_expression.Insert` construct also supports being
         passed a list of dictionaries or full-table-tuples, which on the
         server will render the less common SQL syntax of "multiple values" -
         this syntax is supported on backends such as SQLite, PostgreSQL,
         MySQL, but not necessarily others::

            users.insert().values(
                [
                    {"name": "some name"},
                    {"name": "some other name"},
                    {"name": "yet another name"},
                ]
            )

         The above form would render a multiple VALUES statement similar to:

         .. sourcecode:: sql

                INSERT INTO users (name) VALUES
                                (:name_1),
                                (:name_2),
                                (:name_3)

         It is essential to note that **passing multiple values is
         NOT the same as using traditional executemany() form**.  The above
         syntax is a **special** syntax not typically used.  To emit an
         INSERT statement against multiple rows, the normal method is
         to pass a multiple values list to the
         :meth:`_engine.Connection.execute`
         method, which is supported by all database backends and is generally
         more efficient for a very large number of parameters.

           .. seealso::

               :ref:`tutorial_multiple_parameters` - an introduction to
               the traditional Core method of multiple parameter set
               invocation for INSERTs and other statements.

          The UPDATE construct also supports rendering the SET parameters
          in a specific order.  For this feature refer to the
          :meth:`_expression.Update.ordered_values` method.

           .. seealso::

              :meth:`_expression.Update.ordered_values`


        """
        if args:
            # positional case.  this is currently expensive.   we don't
            # yet have positional-only args so we have to check the length.
            # then we need to check multiparams vs. single dictionary.
            # since the parameter format is needed in order to determine
            # a cache key, we need to determine this up front.
            arg = args[0]

            if kwargs:
                raise exc.ArgumentError(
                    "Can't pass positional and kwargs to values() "
                    "simultaneously"
                )
            elif len(args) > 1:
                raise exc.ArgumentError(
                    "Only a single dictionary/tuple or list of "
                    "dictionaries/tuples is accepted positionally."
                )

            elif isinstance(arg, collections_abc.Sequence):
                if arg and isinstance(arg[0], dict):
                    multi_kv_generator = DMLState.get_plugin_class(
                        self
                    )._get_multi_crud_kv_pairs
                    self._multi_values += (multi_kv_generator(self, arg),)
                    return self

                if arg and isinstance(arg[0], (list, tuple)):
                    self._multi_values += (arg,)
                    return self

                if TYPE_CHECKING:
                    # crud.py raises during compilation if this is not the
                    # case
                    assert isinstance(self, Insert)

                # tuple values
                arg = {c.key: value for c, value in zip(self.table.c, arg)}

        else:
            # kwarg path.  this is the most common path for non-multi-params
            # so this is fairly quick.
            arg = cast("Dict[_DMLColumnArgument, Any]", kwargs)
            if args:
                raise exc.ArgumentError(
                    "Only a single dictionary/tuple or list of "
                    "dictionaries/tuples is accepted positionally."
                )

        # for top level values(), convert literals to anonymous bound
        # parameters at statement construction time, so that these values can
        # participate in the cache key process like any other ClauseElement.
        # crud.py now intercepts bound parameters with unique=True from here
        # and ensures they get the "crud"-style name when rendered.

        kv_generator = DMLState.get_plugin_class(self)._get_crud_kv_pairs
        coerced_arg = dict(kv_generator(self, arg.items(), True))
        if self._values:
            self._values = self._values.union(coerced_arg)
        else:
            self._values = util.immutabledict(coerced_arg)
        return self


class Insert(ValuesBase):
    """Represent an INSERT construct.

    The :class:`_expression.Insert` object is created using the
    :func:`_expression.insert()` function.

    """

    __visit_name__ = "insert"

    _supports_multi_parameters = True

    select = None
    include_insert_from_select_defaults = False

    _sort_by_parameter_order: bool = False

    is_insert = True

    table: TableClause

    _traverse_internals = (
        [
            ("table", InternalTraversal.dp_clauseelement),
            ("_inline", InternalTraversal.dp_boolean),
            ("_select_names", InternalTraversal.dp_string_list),
            ("_values", InternalTraversal.dp_dml_values),
            ("_multi_values", InternalTraversal.dp_dml_multi_values),
            ("select", InternalTraversal.dp_clauseelement),
            ("_post_values_clause", InternalTraversal.dp_clauseelement),
            ("_returning", InternalTraversal.dp_clauseelement_tuple),
            ("_hints", InternalTraversal.dp_table_hint_list),
            ("_return_defaults", InternalTraversal.dp_boolean),
            (
                "_return_defaults_columns",
                InternalTraversal.dp_clauseelement_tuple,
            ),
            ("_sort_by_parameter_order", InternalTraversal.dp_boolean),
        ]
        + HasPrefixes._has_prefixes_traverse_internals
        + DialectKWArgs._dialect_kwargs_traverse_internals
        + Executable._executable_traverse_internals
        + HasCTE._has_ctes_traverse_internals
    )

    def __init__(self, table: _DMLTableArgument):
        super().__init__(table)

    @_generative
    def inline(self) -> Self:
        """Make this :class:`_expression.Insert` construct "inline" .

        When set, no attempt will be made to retrieve the
        SQL-generated default values to be provided within the statement;
        in particular,
        this allows SQL expressions to be rendered 'inline' within the
        statement without the need to pre-execute them beforehand; for
        backends that support "returning", this turns off the "implicit
        returning" feature for the statement.


        .. versionchanged:: 1.4 the :paramref:`_expression.Insert.inline`
           parameter
           is now superseded by the :meth:`_expression.Insert.inline` method.

        """
        self._inline = True
        return self

    @_generative
    def from_select(
        self,
        names: Sequence[_DMLColumnArgument],
        select: Selectable,
        include_defaults: bool = True,
    ) -> Self:
        """Return a new :class:`_expression.Insert` construct which represents
        an ``INSERT...FROM SELECT`` statement.

        e.g.::

            sel = select(table1.c.a, table1.c.b).where(table1.c.c > 5)
            ins = table2.insert().from_select(["a", "b"], sel)

        :param names: a sequence of string column names or
         :class:`_schema.Column`
         objects representing the target columns.
        :param select: a :func:`_expression.select` construct,
         :class:`_expression.FromClause`
         or other construct which resolves into a
         :class:`_expression.FromClause`,
         such as an ORM :class:`_query.Query` object, etc.  The order of
         columns returned from this FROM clause should correspond to the
         order of columns sent as the ``names`` parameter;  while this
         is not checked before passing along to the database, the database
         would normally raise an exception if these column lists don't
         correspond.
        :param include_defaults: if True, non-server default values and
         SQL expressions as specified on :class:`_schema.Column` objects
         (as documented in :ref:`metadata_defaults_toplevel`) not
         otherwise specified in the list of names will be rendered
         into the INSERT and SELECT statements, so that these values are also
         included in the data to be inserted.

         .. note:: A Python-side default that uses a Python callable function
            will only be invoked **once** for the whole statement, and **not
            per row**.

        """

        if self._values:
            raise exc.InvalidRequestError(
                "This construct already inserts value expressions"
            )

        self._select_names = [
            coercions.expect(roles.DMLColumnRole, name, as_key=True)
            for name in names
        ]
        self._inline = True
        self.include_insert_from_select_defaults = include_defaults
        self.select = coercions.expect(roles.DMLSelectRole, select)
        return self

    if TYPE_CHECKING:
        # START OVERLOADED FUNCTIONS self.returning ReturningInsert 1-8 ", *, sort_by_parameter_order: bool = False"  # noqa: E501

        # code within this block is **programmatically,
        # statically generated** by tools/generate_tuple_map_overloads.py

        @overload
        def returning(
            self, __ent0: _TCCA[_T0], *, sort_by_parameter_order: bool = False
        ) -> ReturningInsert[Tuple[_T0]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            *,
            sort_by_parameter_order: bool = False,
        ) -> ReturningInsert[Tuple[_T0, _T1]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            *,
            sort_by_parameter_order: bool = False,
        ) -> ReturningInsert[Tuple[_T0, _T1, _T2]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            *,
            sort_by_parameter_order: bool = False,
        ) -> ReturningInsert[Tuple[_T0, _T1, _T2, _T3]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            *,
            sort_by_parameter_order: bool = False,
        ) -> ReturningInsert[Tuple[_T0, _T1, _T2, _T3, _T4]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            __ent5: _TCCA[_T5],
            *,
            sort_by_parameter_order: bool = False,
        ) -> ReturningInsert[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            __ent5: _TCCA[_T5],
            __ent6: _TCCA[_T6],
            *,
            sort_by_parameter_order: bool = False,
        ) -> ReturningInsert[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            __ent5: _TCCA[_T5],
            __ent6: _TCCA[_T6],
            __ent7: _TCCA[_T7],
            *,
            sort_by_parameter_order: bool = False,
        ) -> ReturningInsert[
            Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]
        ]: ...

        # END OVERLOADED FUNCTIONS self.returning

        @overload
        def returning(
            self,
            *cols: _ColumnsClauseArgument[Any],
            sort_by_parameter_order: bool = False,
            **__kw: Any,
        ) -> ReturningInsert[Any]: ...

        def returning(
            self,
            *cols: _ColumnsClauseArgument[Any],
            sort_by_parameter_order: bool = False,
            **__kw: Any,
        ) -> ReturningInsert[Any]: ...


class ReturningInsert(Insert, TypedReturnsRows[_TP]):
    """Typing-only class that establishes a generic type form of
    :class:`.Insert` which tracks returned column types.

    This datatype is delivered when calling the
    :meth:`.Insert.returning` method.

    .. versionadded:: 2.0

    """


class DMLWhereBase:
    table: _DMLTableElement
    _where_criteria: Tuple[ColumnElement[Any], ...] = ()

    @_generative
    def where(self, *whereclause: _ColumnExpressionArgument[bool]) -> Self:
        """Return a new construct with the given expression(s) added to
        its WHERE clause, joined to the existing clause via AND, if any.

        Both :meth:`_dml.Update.where` and :meth:`_dml.Delete.where`
        support multiple-table forms, including database-specific
        ``UPDATE...FROM`` as well as ``DELETE..USING``.  For backends that
        don't have multiple-table support, a backend agnostic approach
        to using multiple tables is to make use of correlated subqueries.
        See the linked tutorial sections below for examples.

        .. seealso::

            :ref:`tutorial_correlated_updates`

            :ref:`tutorial_update_from`

            :ref:`tutorial_multi_table_deletes`

        """

        for criterion in whereclause:
            where_criteria: ColumnElement[Any] = coercions.expect(
                roles.WhereHavingRole, criterion, apply_propagate_attrs=self
            )
            self._where_criteria += (where_criteria,)
        return self

    def filter(self, *criteria: roles.ExpressionElementRole[Any]) -> Self:
        """A synonym for the :meth:`_dml.DMLWhereBase.where` method.

        .. versionadded:: 1.4

        """

        return self.where(*criteria)

    def _filter_by_zero(self) -> _DMLTableElement:
        return self.table

    def filter_by(self, **kwargs: Any) -> Self:
        r"""apply the given filtering criterion as a WHERE clause
        to this select.

        """
        from_entity = self._filter_by_zero()

        clauses = [
            _entity_namespace_key(from_entity, key) == value
            for key, value in kwargs.items()
        ]
        return self.filter(*clauses)

    @property
    def whereclause(self) -> Optional[ColumnElement[Any]]:
        """Return the completed WHERE clause for this :class:`.DMLWhereBase`
        statement.

        This assembles the current collection of WHERE criteria
        into a single :class:`_expression.BooleanClauseList` construct.


        .. versionadded:: 1.4

        """

        return BooleanClauseList._construct_for_whereclause(
            self._where_criteria
        )


class Update(DMLWhereBase, ValuesBase):
    """Represent an Update construct.

    The :class:`_expression.Update` object is created using the
    :func:`_expression.update()` function.

    """

    __visit_name__ = "update"

    is_update = True

    _traverse_internals = (
        [
            ("table", InternalTraversal.dp_clauseelement),
            ("_where_criteria", InternalTraversal.dp_clauseelement_tuple),
            ("_inline", InternalTraversal.dp_boolean),
            ("_ordered_values", InternalTraversal.dp_dml_ordered_values),
            ("_values", InternalTraversal.dp_dml_values),
            ("_returning", InternalTraversal.dp_clauseelement_tuple),
            ("_hints", InternalTraversal.dp_table_hint_list),
            ("_return_defaults", InternalTraversal.dp_boolean),
            (
                "_return_defaults_columns",
                InternalTraversal.dp_clauseelement_tuple,
            ),
        ]
        + HasPrefixes._has_prefixes_traverse_internals
        + DialectKWArgs._dialect_kwargs_traverse_internals
        + Executable._executable_traverse_internals
        + HasCTE._has_ctes_traverse_internals
    )

    def __init__(self, table: _DMLTableArgument):
        super().__init__(table)

    @_generative
    def ordered_values(self, *args: Tuple[_DMLColumnArgument, Any]) -> Self:
        """Specify the VALUES clause of this UPDATE statement with an explicit
        parameter ordering that will be maintained in the SET clause of the
        resulting UPDATE statement.

        E.g.::

            stmt = table.update().ordered_values(("name", "ed"), ("ident", "foo"))

        .. seealso::

           :ref:`tutorial_parameter_ordered_updates` - full example of the
           :meth:`_expression.Update.ordered_values` method.

        .. versionchanged:: 1.4 The :meth:`_expression.Update.ordered_values`
           method
           supersedes the
           :paramref:`_expression.update.preserve_parameter_order`
           parameter, which will be removed in SQLAlchemy 2.0.

        """  # noqa: E501
        if self._values:
            raise exc.ArgumentError(
                "This statement already has values present"
            )
        elif self._ordered_values:
            raise exc.ArgumentError(
                "This statement already has ordered values present"
            )

        kv_generator = DMLState.get_plugin_class(self)._get_crud_kv_pairs
        self._ordered_values = kv_generator(self, args, True)
        return self

    @_generative
    def inline(self) -> Self:
        """Make this :class:`_expression.Update` construct "inline" .

        When set, SQL defaults present on :class:`_schema.Column`
        objects via the
        ``default`` keyword will be compiled 'inline' into the statement and
        not pre-executed.  This means that their values will not be available
        in the dictionary returned from
        :meth:`_engine.CursorResult.last_updated_params`.

        .. versionchanged:: 1.4 the :paramref:`_expression.update.inline`
           parameter
           is now superseded by the :meth:`_expression.Update.inline` method.

        """
        self._inline = True
        return self

    if TYPE_CHECKING:
        # START OVERLOADED FUNCTIONS self.returning ReturningUpdate 1-8

        # code within this block is **programmatically,
        # statically generated** by tools/generate_tuple_map_overloads.py

        @overload
        def returning(
            self, __ent0: _TCCA[_T0]
        ) -> ReturningUpdate[Tuple[_T0]]: ...

        @overload
        def returning(
            self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1]
        ) -> ReturningUpdate[Tuple[_T0, _T1]]: ...

        @overload
        def returning(
            self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2]
        ) -> ReturningUpdate[Tuple[_T0, _T1, _T2]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
        ) -> ReturningUpdate[Tuple[_T0, _T1, _T2, _T3]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
        ) -> ReturningUpdate[Tuple[_T0, _T1, _T2, _T3, _T4]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            __ent5: _TCCA[_T5],
        ) -> ReturningUpdate[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            __ent5: _TCCA[_T5],
            __ent6: _TCCA[_T6],
        ) -> ReturningUpdate[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            __ent5: _TCCA[_T5],
            __ent6: _TCCA[_T6],
            __ent7: _TCCA[_T7],
        ) -> ReturningUpdate[
            Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]
        ]: ...

        # END OVERLOADED FUNCTIONS self.returning

        @overload
        def returning(
            self, *cols: _ColumnsClauseArgument[Any], **__kw: Any
        ) -> ReturningUpdate[Any]: ...

        def returning(
            self, *cols: _ColumnsClauseArgument[Any], **__kw: Any
        ) -> ReturningUpdate[Any]: ...


class ReturningUpdate(Update, TypedReturnsRows[_TP]):
    """Typing-only class that establishes a generic type form of
    :class:`.Update` which tracks returned column types.

    This datatype is delivered when calling the
    :meth:`.Update.returning` method.

    .. versionadded:: 2.0

    """


class Delete(DMLWhereBase, UpdateBase):
    """Represent a DELETE construct.

    The :class:`_expression.Delete` object is created using the
    :func:`_expression.delete()` function.

    """

    __visit_name__ = "delete"

    is_delete = True

    _traverse_internals = (
        [
            ("table", InternalTraversal.dp_clauseelement),
            ("_where_criteria", InternalTraversal.dp_clauseelement_tuple),
            ("_returning", InternalTraversal.dp_clauseelement_tuple),
            ("_hints", InternalTraversal.dp_table_hint_list),
        ]
        + HasPrefixes._has_prefixes_traverse_internals
        + DialectKWArgs._dialect_kwargs_traverse_internals
        + Executable._executable_traverse_internals
        + HasCTE._has_ctes_traverse_internals
    )

    def __init__(self, table: _DMLTableArgument):
        self.table = coercions.expect(
            roles.DMLTableRole, table, apply_propagate_attrs=self
        )

    if TYPE_CHECKING:
        # START OVERLOADED FUNCTIONS self.returning ReturningDelete 1-8

        # code within this block is **programmatically,
        # statically generated** by tools/generate_tuple_map_overloads.py

        @overload
        def returning(
            self, __ent0: _TCCA[_T0]
        ) -> ReturningDelete[Tuple[_T0]]: ...

        @overload
        def returning(
            self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1]
        ) -> ReturningDelete[Tuple[_T0, _T1]]: ...

        @overload
        def returning(
            self, __ent0: _TCCA[_T0], __ent1: _TCCA[_T1], __ent2: _TCCA[_T2]
        ) -> ReturningDelete[Tuple[_T0, _T1, _T2]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
        ) -> ReturningDelete[Tuple[_T0, _T1, _T2, _T3]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
        ) -> ReturningDelete[Tuple[_T0, _T1, _T2, _T3, _T4]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            __ent5: _TCCA[_T5],
        ) -> ReturningDelete[Tuple[_T0, _T1, _T2, _T3, _T4, _T5]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            __ent5: _TCCA[_T5],
            __ent6: _TCCA[_T6],
        ) -> ReturningDelete[Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6]]: ...

        @overload
        def returning(
            self,
            __ent0: _TCCA[_T0],
            __ent1: _TCCA[_T1],
            __ent2: _TCCA[_T2],
            __ent3: _TCCA[_T3],
            __ent4: _TCCA[_T4],
            __ent5: _TCCA[_T5],
            __ent6: _TCCA[_T6],
            __ent7: _TCCA[_T7],
        ) -> ReturningDelete[
            Tuple[_T0, _T1, _T2, _T3, _T4, _T5, _T6, _T7]
        ]: ...

        # END OVERLOADED FUNCTIONS self.returning

        @overload
        def returning(
            self, *cols: _ColumnsClauseArgument[Any], **__kw: Any
        ) -> ReturningDelete[Any]: ...

        def returning(
            self, *cols: _ColumnsClauseArgument[Any], **__kw: Any
        ) -> ReturningDelete[Any]: ...


class ReturningDelete(Update, TypedReturnsRows[_TP]):
    """Typing-only class that establishes a generic type form of
    :class:`.Delete` which tracks returned column types.

    This datatype is delivered when calling the
    :meth:`.Delete.returning` method.

    .. versionadded:: 2.0

    """

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\strings.py ===
"""
This module contains a set of functions for vectorized string
operations.
"""

import functools
import sys

import numpy as np
from numpy import (
    add,
    equal,
    greater,
    greater_equal,
    less,
    less_equal,
    not_equal,
)
from numpy import (
    multiply as _multiply_ufunc,
)
from numpy._core.multiarray import _vec_string
from numpy._core.overrides import array_function_dispatch, set_module
from numpy._core.umath import (
    _center,
    _expandtabs,
    _expandtabs_length,
    _ljust,
    _lstrip_chars,
    _lstrip_whitespace,
    _partition,
    _partition_index,
    _replace,
    _rjust,
    _rpartition,
    _rpartition_index,
    _rstrip_chars,
    _rstrip_whitespace,
    _slice,
    _strip_chars,
    _strip_whitespace,
    _zfill,
    isalnum,
    isalpha,
    isdecimal,
    isdigit,
    islower,
    isnumeric,
    isspace,
    istitle,
    isupper,
    str_len,
)
from numpy._core.umath import (
    count as _count_ufunc,
)
from numpy._core.umath import (
    endswith as _endswith_ufunc,
)
from numpy._core.umath import (
    find as _find_ufunc,
)
from numpy._core.umath import (
    index as _index_ufunc,
)
from numpy._core.umath import (
    rfind as _rfind_ufunc,
)
from numpy._core.umath import (
    rindex as _rindex_ufunc,
)
from numpy._core.umath import (
    startswith as _startswith_ufunc,
)


def _override___module__():
    for ufunc in [
        isalnum, isalpha, isdecimal, isdigit, islower, isnumeric, isspace,
        istitle, isupper, str_len,
    ]:
        ufunc.__module__ = "numpy.strings"
        ufunc.__qualname__ = ufunc.__name__


_override___module__()


__all__ = [
    # UFuncs
    "equal", "not_equal", "less", "less_equal", "greater", "greater_equal",
    "add", "multiply", "isalpha", "isdigit", "isspace", "isalnum", "islower",
    "isupper", "istitle", "isdecimal", "isnumeric", "str_len", "find",
    "rfind", "index", "rindex", "count", "startswith", "endswith", "lstrip",
    "rstrip", "strip", "replace", "expandtabs", "center", "ljust", "rjust",
    "zfill", "partition", "rpartition", "slice",

    # _vec_string - Will gradually become ufuncs as well
    "upper", "lower", "swapcase", "capitalize", "title",

    # _vec_string - Will probably not become ufuncs
    "mod", "decode", "encode", "translate",

    # Removed from namespace until behavior has been crystallized
    # "join", "split", "rsplit", "splitlines",
]


MAX = np.iinfo(np.int64).max

array_function_dispatch = functools.partial(
    array_function_dispatch, module='numpy.strings')


def _get_num_chars(a):
    """
    Helper function that returns the number of characters per field in
    a string or unicode array.  This is to abstract out the fact that
    for a unicode array this is itemsize / 4.
    """
    if issubclass(a.dtype.type, np.str_):
        return a.itemsize // 4
    return a.itemsize


def _to_bytes_or_str_array(result, output_dtype_like):
    """
    Helper function to cast a result back into an array
    with the appropriate dtype if an object array must be used
    as an intermediary.
    """
    output_dtype_like = np.asarray(output_dtype_like)
    if result.size == 0:
        # Calling asarray & tolist in an empty array would result
        # in losing shape information
        return result.astype(output_dtype_like.dtype)
    ret = np.asarray(result.tolist())
    if isinstance(output_dtype_like.dtype, np.dtypes.StringDType):
        return ret.astype(type(output_dtype_like.dtype))
    return ret.astype(type(output_dtype_like.dtype)(_get_num_chars(ret)))


def _clean_args(*args):
    """
    Helper function for delegating arguments to Python string
    functions.

    Many of the Python string operations that have optional arguments
    do not use 'None' to indicate a default value.  In these cases,
    we need to remove all None arguments, and those following them.
    """
    newargs = []
    for chk in args:
        if chk is None:
            break
        newargs.append(chk)
    return newargs


def _multiply_dispatcher(a, i):
    return (a,)


@set_module("numpy.strings")
@array_function_dispatch(_multiply_dispatcher)
def multiply(a, i):
    """
    Return (a * i), that is string multiple concatenation,
    element-wise.

    Values in ``i`` of less than 0 are treated as 0 (which yields an
    empty string).

    Parameters
    ----------
    a : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype

    i : array_like, with any integer dtype

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["a", "b", "c"])
    >>> np.strings.multiply(a, 3)
    array(['aaa', 'bbb', 'ccc'], dtype='<U3')
    >>> i = np.array([1, 2, 3])
    >>> np.strings.multiply(a, i)
    array(['a', 'bb', 'ccc'], dtype='<U3')
    >>> np.strings.multiply(np.array(['a']), i)
    array(['a', 'aa', 'aaa'], dtype='<U3')
    >>> a = np.array(['a', 'b', 'c', 'd', 'e', 'f']).reshape((2, 3))
    >>> np.strings.multiply(a, 3)
    array([['aaa', 'bbb', 'ccc'],
           ['ddd', 'eee', 'fff']], dtype='<U3')
    >>> np.strings.multiply(a, i)
    array([['a', 'bb', 'ccc'],
           ['d', 'ee', 'fff']], dtype='<U3')

    """
    a = np.asanyarray(a)

    i = np.asanyarray(i)
    if not np.issubdtype(i.dtype, np.integer):
        raise TypeError(f"unsupported type {i.dtype} for operand 'i'")
    i = np.maximum(i, 0)

    # delegate to stringdtype loops that also do overflow checking
    if a.dtype.char == "T":
        return a * i

    a_len = str_len(a)

    # Ensure we can do a_len * i without overflow.
    if np.any(a_len > sys.maxsize / np.maximum(i, 1)):
        raise OverflowError("Overflow encountered in string multiply")

    buffersizes = a_len * i
    out_dtype = f"{a.dtype.char}{buffersizes.max()}"
    out = np.empty_like(a, shape=buffersizes.shape, dtype=out_dtype)
    return _multiply_ufunc(a, i, out=out)


def _mod_dispatcher(a, values):
    return (a, values)


@set_module("numpy.strings")
@array_function_dispatch(_mod_dispatcher)
def mod(a, values):
    """
    Return (a % i), that is pre-Python 2.6 string formatting
    (interpolation), element-wise for a pair of array_likes of str
    or unicode.

    Parameters
    ----------
    a : array_like, with `np.bytes_` or `np.str_` dtype

    values : array_like of values
       These values will be element-wise interpolated into the string.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["NumPy is a %s library"])
    >>> np.strings.mod(a, values=["Python"])
    array(['NumPy is a Python library'], dtype='<U25')

    >>> a = np.array([b'%d bytes', b'%d bits'])
    >>> values = np.array([8, 64])
    >>> np.strings.mod(a, values)
    array([b'8 bytes', b'64 bits'], dtype='|S7')

    """
    return _to_bytes_or_str_array(
        _vec_string(a, np.object_, '__mod__', (values,)), a)


@set_module("numpy.strings")
def find(a, sub, start=0, end=None):
    """
    For each element, return the lowest index in the string where
    substring ``sub`` is found, such that ``sub`` is contained in the
    range [``start``, ``end``).

    Parameters
    ----------
    a : array_like, with ``StringDType``, ``bytes_`` or ``str_`` dtype

    sub : array_like, with `np.bytes_` or `np.str_` dtype
        The substring to search for.

    start, end : array_like, with any integer dtype
        The range to look in, interpreted as in slice notation.

    Returns
    -------
    y : ndarray
        Output array of ints

    See Also
    --------
    str.find

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["NumPy is a Python library"])
    >>> np.strings.find(a, "Python")
    array([11])

    """
    end = end if end is not None else MAX
    return _find_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def rfind(a, sub, start=0, end=None):
    """
    For each element, return the highest index in the string where
    substring ``sub`` is found, such that ``sub`` is contained in the
    range [``start``, ``end``).

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sub : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        The substring to search for.

    start, end : array_like, with any integer dtype
        The range to look in, interpreted as in slice notation.

    Returns
    -------
    y : ndarray
        Output array of ints

    See Also
    --------
    str.rfind

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["Computer Science"])
    >>> np.strings.rfind(a, "Science", start=0, end=None)
    array([9])
    >>> np.strings.rfind(a, "Science", start=0, end=8)
    array([-1])
    >>> b = np.array(["Computer Science", "Science"])
    >>> np.strings.rfind(b, "Science", start=0, end=None)
    array([9, 0])

    """
    end = end if end is not None else MAX
    return _rfind_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def index(a, sub, start=0, end=None):
    """
    Like `find`, but raises :exc:`ValueError` when the substring is not found.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sub : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    start, end : array_like, with any integer dtype, optional

    Returns
    -------
    out : ndarray
        Output array of ints.

    See Also
    --------
    find, str.index

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["Computer Science"])
    >>> np.strings.index(a, "Science", start=0, end=None)
    array([9])

    """
    end = end if end is not None else MAX
    return _index_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def rindex(a, sub, start=0, end=None):
    """
    Like `rfind`, but raises :exc:`ValueError` when the substring `sub` is
    not found.

    Parameters
    ----------
    a : array-like, with `np.bytes_` or `np.str_` dtype

    sub : array-like, with `np.bytes_` or `np.str_` dtype

    start, end : array-like, with any integer dtype, optional

    Returns
    -------
    out : ndarray
        Output array of ints.

    See Also
    --------
    rfind, str.rindex

    Examples
    --------
    >>> a = np.array(["Computer Science"])
    >>> np.strings.rindex(a, "Science", start=0, end=None)
    array([9])

    """
    end = end if end is not None else MAX
    return _rindex_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def count(a, sub, start=0, end=None):
    """
    Returns an array with the number of non-overlapping occurrences of
    substring ``sub`` in the range [``start``, ``end``).

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sub : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
       The substring to search for.

    start, end : array_like, with any integer dtype
        The range to look in, interpreted as in slice notation.

    Returns
    -------
    y : ndarray
        Output array of ints

    See Also
    --------
    str.count

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.count(c, 'A')
    array([3, 1, 1])
    >>> np.strings.count(c, 'aA')
    array([3, 1, 0])
    >>> np.strings.count(c, 'A', start=1, end=4)
    array([2, 1, 1])
    >>> np.strings.count(c, 'A', start=1, end=3)
    array([1, 0, 0])

    """
    end = end if end is not None else MAX
    return _count_ufunc(a, sub, start, end)


@set_module("numpy.strings")
def startswith(a, prefix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in ``a`` starts with ``prefix``, otherwise `False`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    prefix : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    start, end : array_like, with any integer dtype
        With ``start``, test beginning at that position. With ``end``,
        stop comparing at that position.

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.startswith

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array(['foo', 'bar'])
    >>> s
    array(['foo', 'bar'], dtype='<U3')
    >>> np.strings.startswith(s, 'fo')
    array([True,  False])
    >>> np.strings.startswith(s, 'o', start=1, end=2)
    array([True,  False])

    """
    end = end if end is not None else MAX
    return _startswith_ufunc(a, prefix, start, end)


@set_module("numpy.strings")
def endswith(a, suffix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in ``a`` ends with ``suffix``, otherwise `False`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    suffix : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    start, end : array_like, with any integer dtype
        With ``start``, test beginning at that position. With ``end``,
        stop comparing at that position.

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.endswith

    Examples
    --------
    >>> import numpy as np
    >>> s = np.array(['foo', 'bar'])
    >>> s
    array(['foo', 'bar'], dtype='<U3')
    >>> np.strings.endswith(s, 'ar')
    array([False,  True])
    >>> np.strings.endswith(s, 'a', start=1, end=2)
    array([False,  True])

    """
    end = end if end is not None else MAX
    return _endswith_ufunc(a, suffix, start, end)


def _code_dispatcher(a, encoding=None, errors=None):
    return (a,)


@set_module("numpy.strings")
@array_function_dispatch(_code_dispatcher)
def decode(a, encoding=None, errors=None):
    r"""
    Calls :meth:`bytes.decode` element-wise.

    The set of available codecs comes from the Python standard library,
    and may be extended at runtime.  For more information, see the
    :mod:`codecs` module.

    Parameters
    ----------
    a : array_like, with ``bytes_`` dtype

    encoding : str, optional
       The name of an encoding

    errors : str, optional
       Specifies how to handle encoding errors

    Returns
    -------
    out : ndarray

    See Also
    --------
    :py:meth:`bytes.decode`

    Notes
    -----
    The type of the result will depend on the encoding specified.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
    ...               b'\x81\x82\xc2\xc1\xc2\x82\x81'])
    >>> c
    array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
           b'\x81\x82\xc2\xc1\xc2\x82\x81'], dtype='|S7')
    >>> np.strings.decode(c, encoding='cp037')
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')

    """
    return _to_bytes_or_str_array(
        _vec_string(a, np.object_, 'decode', _clean_args(encoding, errors)),
        np.str_(''))


@set_module("numpy.strings")
@array_function_dispatch(_code_dispatcher)
def encode(a, encoding=None, errors=None):
    """
    Calls :meth:`str.encode` element-wise.

    The set of available codecs comes from the Python standard library,
    and may be extended at runtime. For more information, see the
    :mod:`codecs` module.

    Parameters
    ----------
    a : array_like, with ``StringDType`` or ``str_`` dtype

    encoding : str, optional
       The name of an encoding

    errors : str, optional
       Specifies how to handle encoding errors

    Returns
    -------
    out : ndarray

    See Also
    --------
    str.encode

    Notes
    -----
    The type of the result will depend on the encoding specified.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.encode(a, encoding='cp037')
    array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
       b'\x81\x82\xc2\xc1\xc2\x82\x81'], dtype='|S7')

    """
    return _to_bytes_or_str_array(
        _vec_string(a, np.object_, 'encode', _clean_args(encoding, errors)),
        np.bytes_(b''))


def _expandtabs_dispatcher(a, tabsize=None):
    return (a,)


@set_module("numpy.strings")
@array_function_dispatch(_expandtabs_dispatcher)
def expandtabs(a, tabsize=8):
    """
    Return a copy of each string element where all tab characters are
    replaced by one or more spaces.

    Calls :meth:`str.expandtabs` element-wise.

    Return a copy of each string element where all tab characters are
    replaced by one or more spaces, depending on the current column
    and the given `tabsize`. The column number is reset to zero after
    each newline occurring in the string. This doesn't understand other
    non-printing characters or escape sequences.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    tabsize : int, optional
        Replace tabs with `tabsize` number of spaces.  If not given defaults
        to 8 spaces.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input type

    See Also
    --------
    str.expandtabs

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['\t\tHello\tworld'])
    >>> np.strings.expandtabs(a, tabsize=4)  # doctest: +SKIP
    array(['        Hello   world'], dtype='<U21')  # doctest: +SKIP

    """
    a = np.asanyarray(a)
    tabsize = np.asanyarray(tabsize)

    if a.dtype.char == "T":
        return _expandtabs(a, tabsize)

    buffersizes = _expandtabs_length(a, tabsize)
    out_dtype = f"{a.dtype.char}{buffersizes.max()}"
    out = np.empty_like(a, shape=buffersizes.shape, dtype=out_dtype)
    return _expandtabs(a, tabsize, out=out)


def _just_dispatcher(a, width, fillchar=None):
    return (a,)


@set_module("numpy.strings")
@array_function_dispatch(_just_dispatcher)
def center(a, width, fillchar=' '):
    """
    Return a copy of `a` with its elements centered in a string of
    length `width`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    width : array_like, with any integer dtype
        The length of the resulting strings, unless ``width < str_len(a)``.
    fillchar : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Optional padding character to use (default is space).

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.center

    Notes
    -----
    While it is possible for ``a`` and ``fillchar`` to have different dtypes,
    passing a non-ASCII character in ``fillchar`` when ``a`` is of dtype "S"
    is not allowed, and a ``ValueError`` is raised.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b']); c
    array(['a1b2', '1b2a', 'b2a1', '2a1b'], dtype='<U4')
    >>> np.strings.center(c, width=9)
    array(['   a1b2  ', '   1b2a  ', '   b2a1  ', '   2a1b  '], dtype='<U9')
    >>> np.strings.center(c, width=9, fillchar='*')
    array(['***a1b2**', '***1b2a**', '***b2a1**', '***2a1b**'], dtype='<U9')
    >>> np.strings.center(c, width=1)
    array(['a1b2', '1b2a', 'b2a1', '2a1b'], dtype='<U4')

    """
    width = np.asanyarray(width)

    if not np.issubdtype(width.dtype, np.integer):
        raise TypeError(f"unsupported type {width.dtype} for operand 'width'")

    a = np.asanyarray(a)
    fillchar = np.asanyarray(fillchar)

    if np.any(str_len(fillchar) != 1):
        raise TypeError(
            "The fill character must be exactly one character long")

    if np.result_type(a, fillchar).char == "T":
        return _center(a, width, fillchar)

    fillchar = fillchar.astype(a.dtype, copy=False)
    width = np.maximum(str_len(a), width)
    out_dtype = f"{a.dtype.char}{width.max()}"
    shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)
    out = np.empty_like(a, shape=shape, dtype=out_dtype)

    return _center(a, width, fillchar, out=out)


@set_module("numpy.strings")
@array_function_dispatch(_just_dispatcher)
def ljust(a, width, fillchar=' '):
    """
    Return an array with the elements of `a` left-justified in a
    string of length `width`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    width : array_like, with any integer dtype
        The length of the resulting strings, unless ``width < str_len(a)``.
    fillchar : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Optional character to use for padding (default is space).

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.ljust

    Notes
    -----
    While it is possible for ``a`` and ``fillchar`` to have different dtypes,
    passing a non-ASCII character in ``fillchar`` when ``a`` is of dtype "S"
    is not allowed, and a ``ValueError`` is raised.

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.ljust(c, width=3)
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.ljust(c, width=9)
    array(['aAaAaA   ', '  aA     ', 'abBABba  '], dtype='<U9')

    """
    width = np.asanyarray(width)
    if not np.issubdtype(width.dtype, np.integer):
        raise TypeError(f"unsupported type {width.dtype} for operand 'width'")

    a = np.asanyarray(a)
    fillchar = np.asanyarray(fillchar)

    if np.any(str_len(fillchar) != 1):
        raise TypeError(
            "The fill character must be exactly one character long")

    if np.result_type(a, fillchar).char == "T":
        return _ljust(a, width, fillchar)

    fillchar = fillchar.astype(a.dtype, copy=False)
    width = np.maximum(str_len(a), width)
    shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)
    out_dtype = f"{a.dtype.char}{width.max()}"
    out = np.empty_like(a, shape=shape, dtype=out_dtype)

    return _ljust(a, width, fillchar, out=out)


@set_module("numpy.strings")
@array_function_dispatch(_just_dispatcher)
def rjust(a, width, fillchar=' '):
    """
    Return an array with the elements of `a` right-justified in a
    string of length `width`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    width : array_like, with any integer dtype
        The length of the resulting strings, unless ``width < str_len(a)``.
    fillchar : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Optional padding character to use (default is space).

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.rjust

    Notes
    -----
    While it is possible for ``a`` and ``fillchar`` to have different dtypes,
    passing a non-ASCII character in ``fillchar`` when ``a`` is of dtype "S"
    is not allowed, and a ``ValueError`` is raised.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.rjust(a, width=3)
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.rjust(a, width=9)
    array(['   aAaAaA', '     aA  ', '  abBABba'], dtype='<U9')

    """
    width = np.asanyarray(width)
    if not np.issubdtype(width.dtype, np.integer):
        raise TypeError(f"unsupported type {width.dtype} for operand 'width'")

    a = np.asanyarray(a)
    fillchar = np.asanyarray(fillchar)

    if np.any(str_len(fillchar) != 1):
        raise TypeError(
            "The fill character must be exactly one character long")

    if np.result_type(a, fillchar).char == "T":
        return _rjust(a, width, fillchar)

    fillchar = fillchar.astype(a.dtype, copy=False)
    width = np.maximum(str_len(a), width)
    shape = np.broadcast_shapes(a.shape, width.shape, fillchar.shape)
    out_dtype = f"{a.dtype.char}{width.max()}"
    out = np.empty_like(a, shape=shape, dtype=out_dtype)

    return _rjust(a, width, fillchar, out=out)


def _zfill_dispatcher(a, width):
    return (a,)


@set_module("numpy.strings")
@array_function_dispatch(_zfill_dispatcher)
def zfill(a, width):
    """
    Return the numeric string left-filled with zeros. A leading
    sign prefix (``+``/``-``) is handled by inserting the padding
    after the sign character rather than before.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    width : array_like, with any integer dtype
        Width of string to left-fill elements in `a`.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input type

    See Also
    --------
    str.zfill

    Examples
    --------
    >>> import numpy as np
    >>> np.strings.zfill(['1', '-1', '+1'], 3)
    array(['001', '-01', '+01'], dtype='<U3')

    """
    width = np.asanyarray(width)
    if not np.issubdtype(width.dtype, np.integer):
        raise TypeError(f"unsupported type {width.dtype} for operand 'width'")

    a = np.asanyarray(a)

    if a.dtype.char == "T":
        return _zfill(a, width)

    width = np.maximum(str_len(a), width)
    shape = np.broadcast_shapes(a.shape, width.shape)
    out_dtype = f"{a.dtype.char}{width.max()}"
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    return _zfill(a, width, out=out)


@set_module("numpy.strings")
def lstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading characters
    removed.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.lstrip

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    # The 'a' variable is unstripped from c[1] because of leading whitespace.
    >>> np.strings.lstrip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABba'], dtype='<U7')
    >>> np.strings.lstrip(c, 'A') # leaves c unchanged
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> (np.strings.lstrip(c, ' ') == np.strings.lstrip(c, '')).all()
    np.False_
    >>> (np.strings.lstrip(c, ' ') == np.strings.lstrip(c)).all()
    np.True_

    """
    if chars is None:
        return _lstrip_whitespace(a)
    return _lstrip_chars(a, chars)


@set_module("numpy.strings")
def rstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the trailing characters
    removed.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.rstrip

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', 'abBABba'])
    >>> c
    array(['aAaAaA', 'abBABba'], dtype='<U7')
    >>> np.strings.rstrip(c, 'a')
    array(['aAaAaA', 'abBABb'], dtype='<U7')
    >>> np.strings.rstrip(c, 'A')
    array(['aAaAa', 'abBABba'], dtype='<U7')

    """
    if chars is None:
        return _rstrip_whitespace(a)
    return _rstrip_chars(a, chars)


@set_module("numpy.strings")
def strip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading and
    trailing characters removed.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    chars : scalar with the same dtype as ``a``, optional
       The ``chars`` argument is a string specifying the set of
       characters to be removed. If ``None``, the ``chars``
       argument defaults to removing whitespace. The ``chars`` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.strip

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.strings.strip(c)
    array(['aAaAaA', 'aA', 'abBABba'], dtype='<U7')
    # 'a' unstripped from c[1] because of leading whitespace.
    >>> np.strings.strip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABb'], dtype='<U7')
    # 'A' unstripped from c[1] because of trailing whitespace.
    >>> np.strings.strip(c, 'A')
    array(['aAaAa', '  aA  ', 'abBABba'], dtype='<U7')

    """
    if chars is None:
        return _strip_whitespace(a)
    return _strip_chars(a, chars)


def _unary_op_dispatcher(a):
    return (a,)


@set_module("numpy.strings")
@array_function_dispatch(_unary_op_dispatcher)
def upper(a):
    """
    Return an array with the elements converted to uppercase.

    Calls :meth:`str.upper` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.upper

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['a1b c', '1bca', 'bca1']); c
    array(['a1b c', '1bca', 'bca1'], dtype='<U5')
    >>> np.strings.upper(c)
    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'upper')


@set_module("numpy.strings")
@array_function_dispatch(_unary_op_dispatcher)
def lower(a):
    """
    Return an array with the elements converted to lowercase.

    Call :meth:`str.lower` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.lower

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['A1B C', '1BCA', 'BCA1']); c
    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')
    >>> np.strings.lower(c)
    array(['a1b c', '1bca', 'bca1'], dtype='<U5')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'lower')


@set_module("numpy.strings")
@array_function_dispatch(_unary_op_dispatcher)
def swapcase(a):
    """
    Return element-wise a copy of the string with
    uppercase characters converted to lowercase and vice versa.

    Calls :meth:`str.swapcase` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.swapcase

    Examples
    --------
    >>> import numpy as np
    >>> c=np.array(['a1B c','1b Ca','b Ca1','cA1b'],'S5'); c
    array(['a1B c', '1b Ca', 'b Ca1', 'cA1b'],
        dtype='|S5')
    >>> np.strings.swapcase(c)
    array(['A1b C', '1B cA', 'B cA1', 'Ca1B'],
        dtype='|S5')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'swapcase')


@set_module("numpy.strings")
@array_function_dispatch(_unary_op_dispatcher)
def capitalize(a):
    """
    Return a copy of ``a`` with only the first character of each element
    capitalized.

    Calls :meth:`str.capitalize` element-wise.

    For byte strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array of strings to capitalize.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.capitalize

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b'],'S4'); c
    array(['a1b2', '1b2a', 'b2a1', '2a1b'],
        dtype='|S4')
    >>> np.strings.capitalize(c)
    array(['A1b2', '1b2a', 'B2a1', '2a1b'],
        dtype='|S4')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'capitalize')


@set_module("numpy.strings")
@array_function_dispatch(_unary_op_dispatcher)
def title(a):
    """
    Return element-wise title cased version of string or unicode.

    Title case words start with uppercase characters, all remaining cased
    characters are lowercase.

    Calls :meth:`str.title` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.title

    Examples
    --------
    >>> import numpy as np
    >>> c=np.array(['a1b c','1b ca','b ca1','ca1b'],'S5'); c
    array(['a1b c', '1b ca', 'b ca1', 'ca1b'],
        dtype='|S5')
    >>> np.strings.title(c)
    array(['A1B C', '1B Ca', 'B Ca1', 'Ca1B'],
        dtype='|S5')

    """
    a_arr = np.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'title')


def _replace_dispatcher(a, old, new, count=None):
    return (a,)


@set_module("numpy.strings")
@array_function_dispatch(_replace_dispatcher)
def replace(a, old, new, count=-1):
    """
    For each element in ``a``, return a copy of the string with
    occurrences of substring ``old`` replaced by ``new``.

    Parameters
    ----------
    a : array_like, with ``bytes_`` or ``str_`` dtype

    old, new : array_like, with ``bytes_`` or ``str_`` dtype

    count : array_like, with ``int_`` dtype
        If the optional argument ``count`` is given, only the first
        ``count`` occurrences are replaced.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.replace

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(["That is a mango", "Monkeys eat mangos"])
    >>> np.strings.replace(a, 'mango', 'banana')
    array(['That is a banana', 'Monkeys eat bananas'], dtype='<U19')

    >>> a = np.array(["The dish is fresh", "This is it"])
    >>> np.strings.replace(a, 'is', 'was')
    array(['The dwash was fresh', 'Thwas was it'], dtype='<U19')

    """
    count = np.asanyarray(count)
    if not np.issubdtype(count.dtype, np.integer):
        raise TypeError(f"unsupported type {count.dtype} for operand 'count'")

    arr = np.asanyarray(a)
    old_dtype = getattr(old, 'dtype', None)
    old = np.asanyarray(old)
    new_dtype = getattr(new, 'dtype', None)
    new = np.asanyarray(new)

    if np.result_type(arr, old, new).char == "T":
        return _replace(arr, old, new, count)

    a_dt = arr.dtype
    old = old.astype(old_dtype or a_dt, copy=False)
    new = new.astype(new_dtype or a_dt, copy=False)
    max_int64 = np.iinfo(np.int64).max
    counts = _count_ufunc(arr, old, 0, max_int64)
    counts = np.where(count < 0, counts, np.minimum(counts, count))
    buffersizes = str_len(arr) + counts * (str_len(new) - str_len(old))
    out_dtype = f"{arr.dtype.char}{buffersizes.max()}"
    out = np.empty_like(arr, shape=buffersizes.shape, dtype=out_dtype)

    return _replace(arr, old, new, counts, out=out)


def _join_dispatcher(sep, seq):
    return (sep, seq)


@array_function_dispatch(_join_dispatcher)
def _join(sep, seq):
    """
    Return a string which is the concatenation of the strings in the
    sequence `seq`.

    Calls :meth:`str.join` element-wise.

    Parameters
    ----------
    sep : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    seq : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.join

    Examples
    --------
    >>> import numpy as np
    >>> np.strings.join('-', 'osd')  # doctest: +SKIP
    array('o-s-d', dtype='<U5')  # doctest: +SKIP

    >>> np.strings.join(['-', '.'], ['ghc', 'osd'])  # doctest: +SKIP
    array(['g-h-c', 'o.s.d'], dtype='<U5')  # doctest: +SKIP

    """
    return _to_bytes_or_str_array(
        _vec_string(sep, np.object_, 'join', (seq,)), seq)


def _split_dispatcher(a, sep=None, maxsplit=None):
    return (a,)


@array_function_dispatch(_split_dispatcher)
def _split(a, sep=None, maxsplit=None):
    """
    For each element in `a`, return a list of the words in the
    string, using `sep` as the delimiter string.

    Calls :meth:`str.split` element-wise.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sep : str or unicode, optional
       If `sep` is not specified or None, any whitespace string is a
       separator.

    maxsplit : int, optional
        If `maxsplit` is given, at most `maxsplit` splits are done.

    Returns
    -------
    out : ndarray
        Array of list objects

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array("Numpy is nice!")
    >>> np.strings.split(x, " ")  # doctest: +SKIP
    array(list(['Numpy', 'is', 'nice!']), dtype=object)  # doctest: +SKIP

    >>> np.strings.split(x, " ", 1)  # doctest: +SKIP
    array(list(['Numpy', 'is nice!']), dtype=object)  # doctest: +SKIP

    See Also
    --------
    str.split, rsplit

    """
    # This will return an array of lists of different sizes, so we
    # leave it as an object array
    return _vec_string(
        a, np.object_, 'split', [sep] + _clean_args(maxsplit))


@array_function_dispatch(_split_dispatcher)
def _rsplit(a, sep=None, maxsplit=None):
    """
    For each element in `a`, return a list of the words in the
    string, using `sep` as the delimiter string.

    Calls :meth:`str.rsplit` element-wise.

    Except for splitting from the right, `rsplit`
    behaves like `split`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    sep : str or unicode, optional
        If `sep` is not specified or None, any whitespace string
        is a separator.
    maxsplit : int, optional
        If `maxsplit` is given, at most `maxsplit` splits are done,
        the rightmost ones.

    Returns
    -------
    out : ndarray
        Array of list objects

    See Also
    --------
    str.rsplit, split

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', 'abBABba'])
    >>> np.strings.rsplit(a, 'A')  # doctest: +SKIP
    array([list(['a', 'a', 'a', '']),  # doctest: +SKIP
           list(['abB', 'Bba'])], dtype=object)  # doctest: +SKIP

    """
    # This will return an array of lists of different sizes, so we
    # leave it as an object array
    return _vec_string(
        a, np.object_, 'rsplit', [sep] + _clean_args(maxsplit))


def _splitlines_dispatcher(a, keepends=None):
    return (a,)


@array_function_dispatch(_splitlines_dispatcher)
def _splitlines(a, keepends=None):
    """
    For each element in `a`, return a list of the lines in the
    element, breaking at line boundaries.

    Calls :meth:`str.splitlines` element-wise.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    keepends : bool, optional
        Line breaks are not included in the resulting list unless
        keepends is given and true.

    Returns
    -------
    out : ndarray
        Array of list objects

    See Also
    --------
    str.splitlines

    Examples
    --------
    >>> np.char.splitlines("first line\\nsecond line")
    array(list(['first line', 'second line']), dtype=object)
    >>> a = np.array(["first\\nsecond", "third\\nfourth"])
    >>> np.char.splitlines(a)
    array([list(['first', 'second']), list(['third', 'fourth'])], dtype=object)

    """
    return _vec_string(
        a, np.object_, 'splitlines', _clean_args(keepends))


def _partition_dispatcher(a, sep):
    return (a,)


@set_module("numpy.strings")
@array_function_dispatch(_partition_dispatcher)
def partition(a, sep):
    """
    Partition each element in ``a`` around ``sep``.

    For each element in ``a``, split the element at the first
    occurrence of ``sep``, and return a 3-tuple containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, the first item of
    the tuple will contain the whole string, and the second and third
    ones will be the empty string.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    sep : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Separator to split each string element in ``a``.

    Returns
    -------
    out : 3-tuple:
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part before the separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part after the separator

    See Also
    --------
    str.partition

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array(["Numpy is nice!"])
    >>> np.strings.partition(x, " ")
    (array(['Numpy'], dtype='<U5'),
     array([' '], dtype='<U1'),
     array(['is nice!'], dtype='<U8'))

    """
    a = np.asanyarray(a)
    sep = np.asanyarray(sep)

    if np.result_type(a, sep).char == "T":
        return _partition(a, sep)

    sep = sep.astype(a.dtype, copy=False)
    pos = _find_ufunc(a, sep, 0, MAX)
    a_len = str_len(a)
    sep_len = str_len(sep)

    not_found = pos < 0
    buffersizes1 = np.where(not_found, a_len, pos)
    buffersizes3 = np.where(not_found, 0, a_len - pos - sep_len)

    out_dtype = ",".join([f"{a.dtype.char}{n}" for n in (
        buffersizes1.max(),
        1 if np.all(not_found) else sep_len.max(),
        buffersizes3.max(),
    )])
    shape = np.broadcast_shapes(a.shape, sep.shape)
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    return _partition_index(a, sep, pos, out=(out["f0"], out["f1"], out["f2"]))


@set_module("numpy.strings")
@array_function_dispatch(_partition_dispatcher)
def rpartition(a, sep):
    """
    Partition (split) each element around the right-most separator.

    For each element in ``a``, split the element at the last
    occurrence of ``sep``, and return a 3-tuple containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, the third item of
    the tuple will contain the whole string, and the first and second
    ones will be the empty string.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array
    sep : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Separator to split each string element in ``a``.

    Returns
    -------
    out : 3-tuple:
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part before the separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          separator
        - array with ``StringDType``, ``bytes_`` or ``str_`` dtype with the
          part after the separator

    See Also
    --------
    str.rpartition

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> np.strings.rpartition(a, 'A')
    (array(['aAaAa', '  a', 'abB'], dtype='<U5'),
     array(['A', 'A', 'A'], dtype='<U1'),
     array(['', '  ', 'Bba'], dtype='<U3'))

    """
    a = np.asanyarray(a)
    sep = np.asanyarray(sep)

    if np.result_type(a, sep).char == "T":
        return _rpartition(a, sep)

    sep = sep.astype(a.dtype, copy=False)
    pos = _rfind_ufunc(a, sep, 0, MAX)
    a_len = str_len(a)
    sep_len = str_len(sep)

    not_found = pos < 0
    buffersizes1 = np.where(not_found, 0, pos)
    buffersizes3 = np.where(not_found, a_len, a_len - pos - sep_len)

    out_dtype = ",".join([f"{a.dtype.char}{n}" for n in (
        buffersizes1.max(),
        1 if np.all(not_found) else sep_len.max(),
        buffersizes3.max(),
    )])
    shape = np.broadcast_shapes(a.shape, sep.shape)
    out = np.empty_like(a, shape=shape, dtype=out_dtype)
    return _rpartition_index(
        a, sep, pos, out=(out["f0"], out["f1"], out["f2"]))


def _translate_dispatcher(a, table, deletechars=None):
    return (a,)


@set_module("numpy.strings")
@array_function_dispatch(_translate_dispatcher)
def translate(a, table, deletechars=None):
    """
    For each element in `a`, return a copy of the string where all
    characters occurring in the optional argument `deletechars` are
    removed, and the remaining characters have been mapped through the
    given translation table.

    Calls :meth:`str.translate` element-wise.

    Parameters
    ----------
    a : array-like, with `np.bytes_` or `np.str_` dtype

    table : str of length 256

    deletechars : str

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.translate

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['a1b c', '1bca', 'bca1'])
    >>> table = a[0].maketrans('abc', '123')
    >>> deletechars = ' '
    >>> np.char.translate(a, table, deletechars)
    array(['112 3', '1231', '2311'], dtype='<U5')

    """
    a_arr = np.asarray(a)
    if issubclass(a_arr.dtype.type, np.str_):
        return _vec_string(
            a_arr, a_arr.dtype, 'translate', (table,))
    else:
        return _vec_string(
            a_arr,
            a_arr.dtype,
            'translate',
            [table] + _clean_args(deletechars)
        )

@set_module("numpy.strings")
def slice(a, start=None, stop=None, step=None, /):
    """
    Slice the strings in `a` by slices specified by `start`, `stop`, `step`.
    Like in the regular Python `slice` object, if only `start` is
    specified then it is interpreted as the `stop`.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array

    start : None, an integer or an array of integers
        The start of the slice, broadcasted to `a`'s shape

    stop : None, an integer or an array of integers
        The end of the slice, broadcasted to `a`'s shape

    step : None, an integer or an array of integers
        The step for the slice, broadcasted to `a`'s shape

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input type

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['hello', 'world'])
    >>> np.strings.slice(a, 2)
    array(['he', 'wo'], dtype='<U5')

    >>> np.strings.slice(a, 1, 5, 2)
    array(['el', 'ol'], dtype='<U5')

    One can specify different start/stop/step for different array entries:

    >>> np.strings.slice(a, np.array([1, 2]), np.array([4, 5]))
    array(['ell', 'rld'], dtype='<U5')

    Negative slices have the same meaning as in regular Python:

    >>> b = np.array(['hello world', 'γεια σου κόσμε', '你好世界', '👋 🌍'],
    ...              dtype=np.dtypes.StringDType())
    >>> np.strings.slice(b, -2)
    array(['hello wor', 'γεια σου κόσ', '你好', '👋'], dtype=StringDType())

    >>> np.strings.slice(b, [3, -10, 2, -3], [-1, -2, -1, 3])
    array(['lo worl', ' σου κόσ', '世', '👋 🌍'], dtype=StringDType())

    >>> np.strings.slice(b, None, None, -1)
    array(['dlrow olleh', 'εμσόκ υοσ αιεγ', '界世好你', '🌍 👋'],
          dtype=StringDType())

    """
    # Just like in the construction of a regular slice object, if only start
    # is specified then start will become stop, see logic in slice_new.
    if stop is None:
        stop = start
        start = None

    # adjust start, stop, step to be integers, see logic in PySlice_Unpack
    if step is None:
        step = 1
    step = np.asanyarray(step)
    if not np.issubdtype(step.dtype, np.integer):
        raise TypeError(f"unsupported type {step.dtype} for operand 'step'")
    if np.any(step == 0):
        raise ValueError("slice step cannot be zero")

    if start is None:
        start = np.where(step < 0, np.iinfo(np.intp).max, 0)

    if stop is None:
        stop = np.where(step < 0, np.iinfo(np.intp).min, np.iinfo(np.intp).max)

    return _slice(a, start, stop, step)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\evalf.py ===
"""
Adaptive numerical evaluation of SymPy expressions, using mpmath
for mathematical functions.
"""
from __future__ import annotations
from typing import Callable, TYPE_CHECKING, Any, overload, Type

import math

import mpmath.libmp as libmp
from mpmath import (
    make_mpc, make_mpf, mp, mpc, mpf, nsum, quadts, quadosc, workprec)
from mpmath import inf as mpmath_inf
from mpmath.libmp import (from_int, from_man_exp, from_rational, fhalf,
                          fnan, finf, fninf, fnone, fone, fzero, mpf_abs, mpf_add,
                          mpf_atan, mpf_atan2, mpf_cmp, mpf_cos, mpf_e, mpf_exp, mpf_log, mpf_lt,
                          mpf_mul, mpf_neg, mpf_pi, mpf_pow, mpf_pow_int, mpf_shift, mpf_sin,
                          mpf_sqrt, normalize, round_nearest, to_int, to_str, mpf_tan)
from mpmath.libmp import bitcount as mpmath_bitcount
from mpmath.libmp.backend import MPZ
from mpmath.libmp.libmpc import _infs_nan
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps

from .sympify import sympify
from .singleton import S
from sympy.external.gmpy import SYMPY_INTS
from sympy.utilities.iterables import is_sequence
from sympy.utilities.lambdify import lambdify
from sympy.utilities.misc import as_int

if TYPE_CHECKING:
    from sympy.core.expr import Expr
    from sympy.core.add import Add
    from sympy.core.mul import Mul
    from sympy.core.power import Pow
    from sympy.core.symbol import Symbol
    from sympy.integrals.integrals import Integral
    from sympy.concrete.summations import Sum
    from sympy.concrete.products import Product
    from sympy.functions.elementary.exponential import exp, log
    from sympy.functions.elementary.complexes import Abs, re, im
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.trigonometric import atan
    from .numbers import Float, Rational, Integer, AlgebraicNumber, Number

LG10 = math.log2(10)
rnd = round_nearest


def bitcount(n):
    """Return smallest integer, b, such that |n|/2**b < 1.
    """
    return mpmath_bitcount(abs(int(n)))

# Used in a few places as placeholder values to denote exponents and
# precision levels, e.g. of exact numbers. Must be careful to avoid
# passing these to mpmath functions or returning them in final results.
INF = float(mpmath_inf)
MINUS_INF = float(-mpmath_inf)

# ~= 100 digits. Real men set this to INF.
DEFAULT_MAXPREC = 333


class PrecisionExhausted(ArithmeticError):
    pass

#----------------------------------------------------------------------------#
#                                                                            #
#              Helper functions for arithmetic and complex parts             #
#                                                                            #
#----------------------------------------------------------------------------#

"""
An mpf value tuple is a tuple of integers (sign, man, exp, bc)
representing a floating-point number: [1, -1][sign]*man*2**exp where
sign is 0 or 1 and bc should correspond to the number of bits used to
represent the mantissa (man) in binary notation, e.g.
"""

MPF_TUP = tuple[int, int, int, int]  # mpf value tuple

"""
Explanation
===========

>>> from sympy.core.evalf import bitcount
>>> sign, man, exp, bc = 0, 5, 1, 3
>>> n = [1, -1][sign]*man*2**exp
>>> n, bitcount(man)
(10, 3)

A temporary result is a tuple (re, im, re_acc, im_acc) where
re and im are nonzero mpf value tuples representing approximate
numbers, or None to denote exact zeros.

re_acc, im_acc are integers denoting log2(e) where e is the estimated
relative accuracy of the respective complex part, but may be anything
if the corresponding complex part is None.

"""
TMP_RES = Any  # temporary result, should be some variant of
# tUnion[tTuple[Optional[MPF_TUP], Optional[MPF_TUP],
#               Optional[int], Optional[int]],
#        'ComplexInfinity']
# but mypy reports error because it doesn't know as we know
# 1. re and re_acc are either both None or both MPF_TUP
# 2. sometimes the result can't be zoo

# type of the "options" parameter in internal evalf functions
OPT_DICT = dict[str, Any]


def fastlog(x: MPF_TUP | None) -> int | Any:
    """Fast approximation of log2(x) for an mpf value tuple x.

    Explanation
    ===========

    Calculated as exponent + width of mantissa. This is an
    approximation for two reasons: 1) it gives the ceil(log2(abs(x)))
    value and 2) it is too high by 1 in the case that x is an exact
    power of 2. Although this is easy to remedy by testing to see if
    the odd mpf mantissa is 1 (indicating that one was dealing with
    an exact power of 2) that would decrease the speed and is not
    necessary as this is only being used as an approximation for the
    number of bits in x. The correct return value could be written as
    "x[2] + (x[3] if x[1] != 1 else 0)".
        Since mpf tuples always have an odd mantissa, no check is done
    to see if the mantissa is a multiple of 2 (in which case the
    result would be too large by 1).

    Examples
    ========

    >>> from sympy import log
    >>> from sympy.core.evalf import fastlog, bitcount
    >>> s, m, e = 0, 5, 1
    >>> bc = bitcount(m)
    >>> n = [1, -1][s]*m*2**e
    >>> n, (log(n)/log(2)).evalf(2), fastlog((s, m, e, bc))
    (10, 3.3, 4)
    """

    if not x or x == fzero:
        return MINUS_INF
    return x[2] + x[3]


def pure_complex(v: Expr, or_real=False) -> tuple[Number, Number] | None:
    """Return a and b if v matches a + I*b where b is not zero and
    a and b are Numbers, else None. If `or_real` is True then 0 will
    be returned for `b` if `v` is a real number.

    Examples
    ========

    >>> from sympy.core.evalf import pure_complex
    >>> from sympy import sqrt, I, S
    >>> a, b, surd = S(2), S(3), sqrt(2)
    >>> pure_complex(a)
    >>> pure_complex(a, or_real=True)
    (2, 0)
    >>> pure_complex(surd)
    >>> pure_complex(a + b*I)
    (2, 3)
    >>> pure_complex(I)
    (0, 1)
    """
    h, t = v.as_coeff_Add()
    if t:
        c, i = t.as_coeff_Mul()
        if i is S.ImaginaryUnit:
            return h, c
    elif or_real:
        return h, S.Zero
    return None


# I don't know what this is, see function scaled_zero below
SCALED_ZERO_TUP = tuple[list[int], int, int, int]



@overload
def scaled_zero(mag: SCALED_ZERO_TUP, sign=1) -> MPF_TUP:
    ...
@overload
def scaled_zero(mag: int, sign=1) -> tuple[SCALED_ZERO_TUP, int]:
    ...
def scaled_zero(mag: SCALED_ZERO_TUP | int, sign=1) -> \
        MPF_TUP | tuple[SCALED_ZERO_TUP, int]:
    """Return an mpf representing a power of two with magnitude ``mag``
    and -1 for precision. Or, if ``mag`` is a scaled_zero tuple, then just
    remove the sign from within the list that it was initially wrapped
    in.

    Examples
    ========

    >>> from sympy.core.evalf import scaled_zero
    >>> from sympy import Float
    >>> z, p = scaled_zero(100)
    >>> z, p
    (([0], 1, 100, 1), -1)
    >>> ok = scaled_zero(z)
    >>> ok
    (0, 1, 100, 1)
    >>> Float(ok)
    1.26765060022823e+30
    >>> Float(ok, p)
    0.e+30
    >>> ok, p = scaled_zero(100, -1)
    >>> Float(scaled_zero(ok), p)
    -0.e+30
    """
    if isinstance(mag, tuple) and len(mag) == 4 and iszero(mag, scaled=True):
        return (mag[0][0],) + mag[1:]
    elif isinstance(mag, SYMPY_INTS):
        if sign not in [-1, 1]:
            raise ValueError('sign must be +/-1')
        rv, p = mpf_shift(fone, mag), -1
        s = 0 if sign == 1 else 1
        rv = ([s],) + rv[1:]
        return rv, p
    else:
        raise ValueError('scaled zero expects int or scaled_zero tuple.')


def iszero(mpf: MPF_TUP | SCALED_ZERO_TUP | None, scaled=False) -> bool | None:
    if not scaled:
        return not mpf or not mpf[1] and not mpf[-1]
    return mpf and isinstance(mpf[0], list) and mpf[1] == mpf[-1] == 1


def complex_accuracy(result: TMP_RES) -> int | Any:
    """
    Returns relative accuracy of a complex number with given accuracies
    for the real and imaginary parts. The relative accuracy is defined
    in the complex norm sense as ||z|+|error|| / |z| where error
    is equal to (real absolute error) + (imag absolute error)*i.

    The full expression for the (logarithmic) error can be approximated
    easily by using the max norm to approximate the complex norm.

    In the worst case (re and im equal), this is wrong by a factor
    sqrt(2), or by log2(sqrt(2)) = 0.5 bit.
    """
    if result is S.ComplexInfinity:
        return INF
    re, im, re_acc, im_acc = result
    if not im:
        if not re:
            return INF
        return re_acc
    if not re:
        return im_acc
    re_size = fastlog(re)
    im_size = fastlog(im)
    absolute_error = max(re_size - re_acc, im_size - im_acc)
    relative_error = absolute_error - max(re_size, im_size)
    return -relative_error


def get_abs(expr: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    result = evalf(expr, prec + 2, options)
    if result is S.ComplexInfinity:
        return finf, None, prec, None
    re, im, re_acc, im_acc = result
    if not re:
        re, re_acc, im, im_acc = im, im_acc, re, re_acc
    if im:
        if expr.is_number:
            abs_expr, _, acc, _ = evalf(abs(N(expr, prec + 2)),
                                        prec + 2, options)
            return abs_expr, None, acc, None
        else:
            if 'subs' in options:
                return libmp.mpc_abs((re, im), prec), None, re_acc, None
            return abs(expr), None, prec, None
    elif re:
        return mpf_abs(re), None, re_acc, None
    else:
        return None, None, None, None


def get_complex_part(expr: Expr, no: int, prec: int, options: OPT_DICT) -> TMP_RES:
    """no = 0 for real part, no = 1 for imaginary part"""
    workprec = prec
    i = 0
    while 1:
        res = evalf(expr, workprec, options)
        if res is S.ComplexInfinity:
            return fnan, None, prec, None
        value, accuracy = res[no::2]
        # XXX is the last one correct? Consider re((1+I)**2).n()
        if (not value) or accuracy >= prec or -value[2] > prec:
            return value, None, accuracy, None
        workprec += max(30, 2**i)
        i += 1


def evalf_abs(expr: 'Abs', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_abs(expr.args[0], prec, options)


def evalf_re(expr: 're', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_complex_part(expr.args[0], 0, prec, options)


def evalf_im(expr: 'im', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_complex_part(expr.args[0], 1, prec, options)


def finalize_complex(re: MPF_TUP, im: MPF_TUP, prec: int) -> TMP_RES:
    if re == fzero and im == fzero:
        raise ValueError("got complex zero with unknown accuracy")
    elif re == fzero:
        return None, im, None, prec
    elif im == fzero:
        return re, None, prec, None

    size_re = fastlog(re)
    size_im = fastlog(im)
    if size_re > size_im:
        re_acc = prec
        im_acc = prec + min(-(size_re - size_im), 0)
    else:
        im_acc = prec
        re_acc = prec + min(-(size_im - size_re), 0)
    return re, im, re_acc, im_acc


def chop_parts(value: TMP_RES, prec: int) -> TMP_RES:
    """
    Chop off tiny real or complex parts.
    """
    if value is S.ComplexInfinity:
        return value
    re, im, re_acc, im_acc = value
    # Method 1: chop based on absolute value
    if re and re not in _infs_nan and (fastlog(re) < -prec + 4):
        re, re_acc = None, None
    if im and im not in _infs_nan and (fastlog(im) < -prec + 4):
        im, im_acc = None, None
    # Method 2: chop if inaccurate and relatively small
    if re and im:
        delta = fastlog(re) - fastlog(im)
        if re_acc < 2 and (delta - re_acc <= -prec + 4):
            re, re_acc = None, None
        if im_acc < 2 and (delta - im_acc >= prec - 4):
            im, im_acc = None, None
    return re, im, re_acc, im_acc


def check_target(expr: Expr, result: TMP_RES, prec: int):
    a = complex_accuracy(result)
    if a < prec:
        raise PrecisionExhausted("Failed to distinguish the expression: \n\n%s\n\n"
            "from zero. Try simplifying the input, using chop=True, or providing "
            "a higher maxn for evalf" % (expr))


def get_integer_part(expr: Expr, no: int, options: OPT_DICT, return_ints=False) -> \
        TMP_RES | tuple[int, int]:
    """
    With no = 1, computes ceiling(expr)
    With no = -1, computes floor(expr)

    Note: this function either gives the exact result or signals failure.
    """
    from sympy.functions.elementary.complexes import re, im
    # The expression is likely less than 2^30 or so
    assumed_size = 30
    result = evalf(expr, assumed_size, options)
    if result is S.ComplexInfinity:
        raise ValueError("Cannot get integer part of Complex Infinity")
    ire, iim, ire_acc, iim_acc = result

    # We now know the size, so we can calculate how much extra precision
    # (if any) is needed to get within the nearest integer
    if ire and iim:
        gap = max(fastlog(ire) - ire_acc, fastlog(iim) - iim_acc)
    elif ire:
        gap = fastlog(ire) - ire_acc
    elif iim:
        gap = fastlog(iim) - iim_acc
    else:
        # ... or maybe the expression was exactly zero
        if return_ints:
            return 0, 0
        else:
            return None, None, None, None

    margin = 10

    if gap >= -margin:
        prec = margin + assumed_size + gap
        ire, iim, ire_acc, iim_acc = evalf(
            expr, prec, options)
    else:
        prec = assumed_size

    # We can now easily find the nearest integer, but to find floor/ceil, we
    # must also calculate whether the difference to the nearest integer is
    # positive or negative (which may fail if very close).
    def calc_part(re_im: Expr, nexpr: MPF_TUP):
        from .add import Add
        _, _, exponent, _ = nexpr
        is_int = exponent == 0
        nint = int(to_int(nexpr, rnd))
        if is_int:
            # make sure that we had enough precision to distinguish
            # between nint and the re or im part (re_im) of expr that
            # was passed to calc_part
            ire, iim, ire_acc, iim_acc = evalf(
                re_im - nint, 10, options)  # don't need much precision
            assert not iim
            size = -fastlog(ire) + 2  # -ve b/c ire is less than 1
            if size > prec:
                ire, iim, ire_acc, iim_acc = evalf(
                    re_im, size, options)
                assert not iim
                nexpr = ire
            nint = int(to_int(nexpr, rnd))
            _, _, new_exp, _ = ire
            is_int = new_exp == 0
        if not is_int:
            # if there are subs and they all contain integer re/im parts
            # then we can (hopefully) safely substitute them into the
            # expression
            s = options.get('subs', False)
            if s:
                # use strict=False with as_int because we take
                # 2.0 == 2
                def is_int_reim(x):
                    """Check for integer or integer + I*integer."""
                    try:
                        as_int(x, strict=False)
                        return True
                    except ValueError:
                        try:
                            [as_int(i, strict=False) for i in x.as_real_imag()]
                            return True
                        except ValueError:
                            return False

                if all(is_int_reim(v) for v in s.values()):
                    re_im = re_im.subs(s)

            re_im = Add(re_im, -nint, evaluate=False)
            x, _, x_acc, _ = evalf(re_im, 10, options)
            try:
                check_target(re_im, (x, None, x_acc, None), 3)
            except PrecisionExhausted:
                if not re_im.equals(0):
                    raise PrecisionExhausted
                x = fzero
            nint += int(no*(mpf_cmp(x or fzero, fzero) == no))
        nint = from_int(nint)
        return nint, INF

    re_, im_, re_acc, im_acc = None, None, None, None

    if ire is not None and ire != fzero:
        re_, re_acc = calc_part(re(expr, evaluate=False), ire)
    if iim is not None and iim != fzero:
        im_, im_acc = calc_part(im(expr, evaluate=False), iim)

    if return_ints:
        return int(to_int(re_ or fzero)), int(to_int(im_ or fzero))
    return re_, im_, re_acc, im_acc


def evalf_ceiling(expr: 'ceiling', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_integer_part(expr.args[0], 1, options)


def evalf_floor(expr: 'floor', prec: int, options: OPT_DICT) -> TMP_RES:
    return get_integer_part(expr.args[0], -1, options)


def evalf_float(expr: 'Float', prec: int, options: OPT_DICT) -> TMP_RES:
    return expr._mpf_, None, prec, None


def evalf_rational(expr: 'Rational', prec: int, options: OPT_DICT) -> TMP_RES:
    return from_rational(expr.p, expr.q, prec), None, prec, None


def evalf_integer(expr: 'Integer', prec: int, options: OPT_DICT) -> TMP_RES:
    return from_int(expr.p, prec), None, prec, None

#----------------------------------------------------------------------------#
#                                                                            #
#                            Arithmetic operations                           #
#                                                                            #
#----------------------------------------------------------------------------#


def add_terms(terms: list, prec: int, target_prec: int) -> \
        tuple[MPF_TUP | SCALED_ZERO_TUP | None, int | None]:
    """
    Helper for evalf_add. Adds a list of (mpfval, accuracy) terms.

    Returns
    =======

    - None, None if there are no non-zero terms;
    - terms[0] if there is only 1 term;
    - scaled_zero if the sum of the terms produces a zero by cancellation
      e.g. mpfs representing 1 and -1 would produce a scaled zero which need
      special handling since they are not actually zero and they are purposely
      malformed to ensure that they cannot be used in anything but accuracy
      calculations;
    - a tuple that is scaled to target_prec that corresponds to the
      sum of the terms.

    The returned mpf tuple will be normalized to target_prec; the input
    prec is used to define the working precision.

    XXX explain why this is needed and why one cannot just loop using mpf_add
    """

    terms = [t for t in terms if not iszero(t[0])]
    if not terms:
        return None, None
    elif len(terms) == 1:
        return terms[0]

    # see if any argument is NaN or oo and thus warrants a special return
    special = []
    from .numbers import Float
    for t in terms:
        arg = Float._new(t[0], 1)
        if arg is S.NaN or arg.is_infinite:
            special.append(arg)
    if special:
        from .add import Add
        rv = evalf(Add(*special), prec + 4, {})
        return rv[0], rv[2]

    working_prec = 2*prec
    sum_man, sum_exp = 0, 0
    absolute_err: list[int] = []

    for x, accuracy in terms:
        sign, man, exp, bc = x
        if sign:
            man = -man
        absolute_err.append(bc + exp - accuracy)
        delta = exp - sum_exp
        if exp >= sum_exp:
            # x much larger than existing sum?
            # first: quick test
            if ((delta > working_prec) and
                ((not sum_man) or
                 delta - bitcount(abs(sum_man)) > working_prec)):
                sum_man = man
                sum_exp = exp
            else:
                sum_man += (man << delta)
        else:
            delta = -delta
            # x much smaller than existing sum?
            if delta - bc > working_prec:
                if not sum_man:
                    sum_man, sum_exp = man, exp
            else:
                sum_man = (sum_man << delta) + man
                sum_exp = exp
    absolute_error = max(absolute_err)
    if not sum_man:
        return scaled_zero(absolute_error)
    if sum_man < 0:
        sum_sign = 1
        sum_man = -sum_man
    else:
        sum_sign = 0
    sum_bc = bitcount(sum_man)
    sum_accuracy = sum_exp + sum_bc - absolute_error
    r = normalize(sum_sign, sum_man, sum_exp, sum_bc, target_prec,
        rnd), sum_accuracy
    return r


def evalf_add(v: 'Add', prec: int, options: OPT_DICT) -> TMP_RES:
    res = pure_complex(v)
    if res:
        h, c = res
        re, _, re_acc, _ = evalf(h, prec, options)
        im, _, im_acc, _ = evalf(c, prec, options)
        return re, im, re_acc, im_acc

    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)

    i = 0
    target_prec = prec
    while 1:
        options['maxprec'] = min(oldmaxprec, 2*prec)

        terms = [evalf(arg, prec + 10, options) for arg in v.args]
        n = terms.count(S.ComplexInfinity)
        if n >= 2:
            return fnan, None, prec, None
        re, re_acc = add_terms(
            [a[0::2] for a in terms if isinstance(a, tuple) and a[0]], prec, target_prec)
        im, im_acc = add_terms(
            [a[1::2] for a in terms if isinstance(a, tuple) and a[1]], prec, target_prec)
        if n == 1:
            if re in (finf, fninf, fnan) or im in (finf, fninf, fnan):
                return fnan, None, prec, None
            return S.ComplexInfinity
        acc = complex_accuracy((re, im, re_acc, im_acc))
        if acc >= target_prec:
            if options.get('verbose'):
                print("ADD: wanted", target_prec, "accurate bits, got", re_acc, im_acc)
            break
        else:
            if (prec - target_prec) > options['maxprec']:
                break

            prec = prec + max(10 + 2**i, target_prec - acc)
            i += 1
            if options.get('verbose'):
                print("ADD: restarting with prec", prec)

    options['maxprec'] = oldmaxprec
    if iszero(re, scaled=True):
        re = scaled_zero(re)
    if iszero(im, scaled=True):
        im = scaled_zero(im)
    return re, im, re_acc, im_acc


def evalf_mul(v: 'Mul', prec: int, options: OPT_DICT) -> TMP_RES:
    res = pure_complex(v)
    if res:
        # the only pure complex that is a mul is h*I
        _, h = res
        im, _, im_acc, _ = evalf(h, prec, options)
        return None, im, None, im_acc
    args = list(v.args)

    # see if any argument is NaN or oo and thus warrants a special return
    has_zero = False
    special = []
    from .numbers import Float
    for arg in args:
        result = evalf(arg, prec, options)
        if result is S.ComplexInfinity:
            special.append(result)
            continue
        if result[0] is None:
            if result[1] is None:
                has_zero = True
            continue
        num = Float._new(result[0], 1)
        if num is S.NaN:
            return fnan, None, prec, None
        if num.is_infinite:
            special.append(num)
    if special:
        if has_zero:
            return fnan, None, prec, None
        from .mul import Mul
        return evalf(Mul(*special), prec + 4, {})
    if has_zero:
        return None, None, None, None

    # With guard digits, multiplication in the real case does not destroy
    # accuracy. This is also true in the complex case when considering the
    # total accuracy; however accuracy for the real or imaginary parts
    # separately may be lower.
    acc = prec

    # XXX: big overestimate
    working_prec = prec + len(args) + 5

    # Empty product is 1
    start = man, exp, bc = MPZ(1), 0, 1

    # First, we multiply all pure real or pure imaginary numbers.
    # direction tells us that the result should be multiplied by
    # I**direction; all other numbers get put into complex_factors
    # to be multiplied out after the first phase.
    last = len(args)
    direction = 0
    args.append(S.One)
    complex_factors = []

    for i, arg in enumerate(args):
        if i != last and pure_complex(arg):
            args[-1] = (args[-1]*arg).expand()
            continue
        elif i == last and arg is S.One:
            continue
        re, im, re_acc, im_acc = evalf(arg, working_prec, options)
        if re and im:
            complex_factors.append((re, im, re_acc, im_acc))
            continue
        elif re:
            (s, m, e, b), w_acc = re, re_acc
        elif im:
            (s, m, e, b), w_acc = im, im_acc
            direction += 1
        else:
            return None, None, None, None
        direction += 2*s
        man *= m
        exp += e
        bc += b
        while bc > 3*working_prec:
            man >>= working_prec
            exp += working_prec
            bc -= working_prec
        acc = min(acc, w_acc)
    sign = (direction & 2) >> 1
    if not complex_factors:
        v = normalize(sign, man, exp, bitcount(man), prec, rnd)
        # multiply by i
        if direction & 1:
            return None, v, None, acc
        else:
            return v, None, acc, None
    else:
        # initialize with the first term
        if (man, exp, bc) != start:
            # there was a real part; give it an imaginary part
            re, im = (sign, man, exp, bitcount(man)), (0, MPZ(0), 0, 0)
            i0 = 0
        else:
            # there is no real part to start (other than the starting 1)
            wre, wim, wre_acc, wim_acc = complex_factors[0]
            acc = min(acc,
                      complex_accuracy((wre, wim, wre_acc, wim_acc)))
            re = wre
            im = wim
            i0 = 1

        for wre, wim, wre_acc, wim_acc in complex_factors[i0:]:
            # acc is the overall accuracy of the product; we aren't
            # computing exact accuracies of the product.
            acc = min(acc,
                      complex_accuracy((wre, wim, wre_acc, wim_acc)))

            use_prec = working_prec
            A = mpf_mul(re, wre, use_prec)
            B = mpf_mul(mpf_neg(im), wim, use_prec)
            C = mpf_mul(re, wim, use_prec)
            D = mpf_mul(im, wre, use_prec)
            re = mpf_add(A, B, use_prec)
            im = mpf_add(C, D, use_prec)
        if options.get('verbose'):
            print("MUL: wanted", prec, "accurate bits, got", acc)
        # multiply by I
        if direction & 1:
            re, im = mpf_neg(im), re
        return re, im, acc, acc


def evalf_pow(v: 'Pow', prec: int, options) -> TMP_RES:

    target_prec = prec
    base, exp = v.args

    # We handle x**n separately. This has two purposes: 1) it is much
    # faster, because we avoid calling evalf on the exponent, and 2) it
    # allows better handling of real/imaginary parts that are exactly zero
    if exp.is_Integer:
        p: int = exp.p  # type: ignore
        # Exact
        if not p:
            return fone, None, prec, None
        # Exponentiation by p magnifies relative error by |p|, so the
        # base must be evaluated with increased precision if p is large
        prec += int(math.log2(abs(p)))
        result = evalf(base, prec + 5, options)
        if result is S.ComplexInfinity:
            if p < 0:
                return None, None, None, None
            return result
        re, im, re_acc, im_acc = result
        # Real to integer power
        if re and not im:
            return mpf_pow_int(re, p, target_prec), None, target_prec, None
        # (x*I)**n = I**n * x**n
        if im and not re:
            z = mpf_pow_int(im, p, target_prec)
            case = p % 4
            if case == 0:
                return z, None, target_prec, None
            if case == 1:
                return None, z, None, target_prec
            if case == 2:
                return mpf_neg(z), None, target_prec, None
            if case == 3:
                return None, mpf_neg(z), None, target_prec
        # Zero raised to an integer power
        if not re:
            if p < 0:
                return S.ComplexInfinity
            return None, None, None, None
        # General complex number to arbitrary integer power
        re, im = libmp.mpc_pow_int((re, im), p, prec)
        # Assumes full accuracy in input
        return finalize_complex(re, im, target_prec)

    result = evalf(base, prec + 5, options)
    if result is S.ComplexInfinity:
        if exp.is_Rational:
            if exp < 0:
                return None, None, None, None
            return result
        raise NotImplementedError

    # Pure square root
    if exp is S.Half:
        xre, xim, _, _ = result
        # General complex square root
        if xim:
            re, im = libmp.mpc_sqrt((xre or fzero, xim), prec)
            return finalize_complex(re, im, prec)
        if not xre:
            return None, None, None, None
        # Square root of a negative real number
        if mpf_lt(xre, fzero):
            return None, mpf_sqrt(mpf_neg(xre), prec), None, prec
        # Positive square root
        return mpf_sqrt(xre, prec), None, prec, None

    # We first evaluate the exponent to find its magnitude
    # This determines the working precision that must be used
    prec += 10
    result = evalf(exp, prec, options)
    if result is S.ComplexInfinity:
        return fnan, None, prec, None
    yre, yim, _, _ = result
    # Special cases: x**0
    if not (yre or yim):
        return fone, None, prec, None

    ysize = fastlog(yre)
    # Restart if too big
    # XXX: prec + ysize might exceed maxprec
    if ysize > 5:
        prec += ysize
        yre, yim, _, _ = evalf(exp, prec, options)

    # Pure exponential function; no need to evalf the base
    if base is S.Exp1:
        if yim:
            re, im = libmp.mpc_exp((yre or fzero, yim), prec)
            return finalize_complex(re, im, target_prec)
        return mpf_exp(yre, target_prec), None, target_prec, None

    xre, xim, _, _ = evalf(base, prec + 5, options)
    # 0**y
    if not (xre or xim):
        if yim:
            return fnan, None, prec, None
        if yre[0] == 1:  # y < 0
            return S.ComplexInfinity
        return None, None, None, None

    # (real ** complex) or (complex ** complex)
    if yim:
        re, im = libmp.mpc_pow(
            (xre or fzero, xim or fzero), (yre or fzero, yim),
            target_prec)
        return finalize_complex(re, im, target_prec)
    # complex ** real
    if xim:
        re, im = libmp.mpc_pow_mpf((xre or fzero, xim), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    # negative ** real
    elif mpf_lt(xre, fzero):
        re, im = libmp.mpc_pow_mpf((xre, fzero), yre, target_prec)
        return finalize_complex(re, im, target_prec)
    # positive ** real
    else:
        return mpf_pow(xre, yre, target_prec), None, target_prec, None


#----------------------------------------------------------------------------#
#                                                                            #
#                            Special functions                               #
#                                                                            #
#----------------------------------------------------------------------------#


def evalf_exp(expr: 'exp', prec: int, options: OPT_DICT) -> TMP_RES:
    from .power import Pow
    return evalf_pow(Pow(S.Exp1, expr.exp, evaluate=False), prec, options)


def evalf_trig(v: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    """
    This function handles sin , cos and tan of complex arguments.

    """
    from sympy.functions.elementary.trigonometric import cos, sin, tan
    if isinstance(v, cos):
        func = mpf_cos
    elif isinstance(v, sin):
        func = mpf_sin
    elif isinstance(v,tan):
        func = mpf_tan
    else:
        raise NotImplementedError
    arg = v.args[0]
    # 20 extra bits is possibly overkill. It does make the need
    # to restart very unlikely
    xprec = prec + 20
    re, im, re_acc, im_acc = evalf(arg, xprec, options)
    if im:
        if 'subs' in options:
            v = v.subs(options['subs'])
        return evalf(v._eval_evalf(prec), prec, options)
    if not re:
        if isinstance(v, cos):
            return fone, None, prec, None
        elif isinstance(v, sin):
            return None, None, None, None
        elif isinstance(v,tan):
            return None, None, None, None
        else:
            raise NotImplementedError
    # For trigonometric functions, we are interested in the
    # fixed-point (absolute) accuracy of the argument.
    xsize = fastlog(re)
    # Magnitude <= 1.0. OK to compute directly, because there is no
    # danger of hitting the first root of cos (with sin, magnitude
    # <= 2.0 would actually be ok)
    if xsize < 1:
        return func(re, prec, rnd), None, prec, None
    # Very large
    if xsize >= 10:
        xprec = prec + xsize
        re, im, re_acc, im_acc = evalf(arg, xprec, options)
    # Need to repeat in case the argument is very close to a
    # multiple of pi (or pi/2), hitting close to a root
    while 1:
        y = func(re, prec, rnd)
        ysize = fastlog(y)
        gap = -ysize
        accuracy = (xprec - xsize) - gap
        if accuracy < prec:
            if options.get('verbose'):
                print("SIN/COS/TAN", accuracy, "wanted", prec, "gap", gap)
                print(to_str(y, 10))
            if xprec > options.get('maxprec', DEFAULT_MAXPREC):
                return y, None, accuracy, None
            xprec += gap
            re, im, re_acc, im_acc = evalf(arg, xprec, options)
            continue
        else:
            return y, None, prec, None


def evalf_log(expr: 'log', prec: int, options: OPT_DICT) -> TMP_RES:
    if len(expr.args)>1:
        expr = expr.doit()
        return evalf(expr, prec, options)
    arg = expr.args[0]
    workprec = prec + 10
    result = evalf(arg, workprec, options)
    if result is S.ComplexInfinity:
        return result
    xre, xim, xacc, _ = result

    # evalf can return NoneTypes if chop=True
    # issue 18516, 19623
    if xre is xim is None:
        # Dear reviewer, I do not know what -inf is;
        # it looks to be (1, 0, -789, -3)
        # but I'm not sure in general,
        # so we just let mpmath figure
        # it out by taking log of 0 directly.
        # It would be better to return -inf instead.
        xre = fzero

    if xim:
        from sympy.functions.elementary.complexes import Abs
        from sympy.functions.elementary.exponential import log

        # XXX: use get_abs etc instead
        re = evalf_log(
            log(Abs(arg, evaluate=False), evaluate=False), prec, options)
        im = mpf_atan2(xim, xre or fzero, prec)
        return re[0], im, re[2], prec

    imaginary_term = (mpf_cmp(xre, fzero) < 0)

    re = mpf_log(mpf_abs(xre), prec, rnd)
    size = fastlog(re)
    if prec - size > workprec and re != fzero:
        from .add import Add
        # We actually need to compute 1+x accurately, not x
        add = Add(S.NegativeOne, arg, evaluate=False)
        xre, xim, _, _ = evalf_add(add, prec, options)
        prec2 = workprec - fastlog(xre)
        # xre is now x - 1 so we add 1 back here to calculate x
        re = mpf_log(mpf_abs(mpf_add(xre, fone, prec2)), prec, rnd)

    re_acc = prec

    if imaginary_term:
        return re, mpf_pi(prec), re_acc, prec
    else:
        return re, None, re_acc, None


def evalf_atan(v: 'atan', prec: int, options: OPT_DICT) -> TMP_RES:
    arg = v.args[0]
    xre, xim, reacc, imacc = evalf(arg, prec + 5, options)
    if xre is xim is None:
        return (None,)*4
    if xim:
        raise NotImplementedError
    return mpf_atan(xre, prec, rnd), None, prec, None


def evalf_subs(prec: int, subs: dict) -> dict:
    """ Change all Float entries in `subs` to have precision prec. """
    newsubs = {}
    for a, b in subs.items():
        b = S(b)
        if b.is_Float:
            b = b._eval_evalf(prec)
        newsubs[a] = b
    return newsubs


def evalf_piecewise(expr: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    from .numbers import Float, Integer
    if 'subs' in options:
        expr = expr.subs(evalf_subs(prec, options['subs']))
        newopts = options.copy()
        del newopts['subs']
        if hasattr(expr, 'func'):
            return evalf(expr, prec, newopts)
        if isinstance(expr, float):
            return evalf(Float(expr), prec, newopts)
        if isinstance(expr, int):
            return evalf(Integer(expr), prec, newopts)

    # We still have undefined symbols
    raise NotImplementedError


def evalf_alg_num(a: 'AlgebraicNumber', prec: int, options: OPT_DICT) -> TMP_RES:
    return evalf(a.to_root(), prec, options)

#----------------------------------------------------------------------------#
#                                                                            #
#                            High-level operations                           #
#                                                                            #
#----------------------------------------------------------------------------#


def as_mpmath(x: Any, prec: int, options: OPT_DICT) -> mpc | mpf:
    from .numbers import Infinity, NegativeInfinity, Zero
    x = sympify(x)
    if isinstance(x, Zero) or x == 0.0:
        return mpf(0)
    if isinstance(x, Infinity):
        return mpf('inf')
    if isinstance(x, NegativeInfinity):
        return mpf('-inf')
    # XXX
    result = evalf(x, prec, options)
    return quad_to_mpmath(result)


def do_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    func = expr.args[0]
    x, xlow, xhigh = expr.args[1]
    if xlow == xhigh:
        xlow = xhigh = 0
    elif x not in func.free_symbols:
        # only the difference in limits matters in this case
        # so if there is a symbol in common that will cancel
        # out when taking the difference, then use that
        # difference
        if xhigh.free_symbols & xlow.free_symbols:
            diff = xhigh - xlow
            if diff.is_number:
                xlow, xhigh = 0, diff

    oldmaxprec = options.get('maxprec', DEFAULT_MAXPREC)
    options['maxprec'] = min(oldmaxprec, 2*prec)

    with workprec(prec + 5):
        xlow = as_mpmath(xlow, prec + 15, options)
        xhigh = as_mpmath(xhigh, prec + 15, options)

        # Integration is like summation, and we can phone home from
        # the integrand function to update accuracy summation style
        # Note that this accuracy is inaccurate, since it fails
        # to account for the variable quadrature weights,
        # but it is better than nothing

        from sympy.functions.elementary.trigonometric import cos, sin
        from .symbol import Wild

        have_part = [False, False]
        max_real_term: float | int = MINUS_INF
        max_imag_term: float | int = MINUS_INF

        def f(t: Expr) -> mpc | mpf:
            nonlocal max_real_term, max_imag_term
            re, im, re_acc, im_acc = evalf(func, mp.prec, {'subs': {x: t}})

            have_part[0] = re or have_part[0]
            have_part[1] = im or have_part[1]

            max_real_term = max(max_real_term, fastlog(re))
            max_imag_term = max(max_imag_term, fastlog(im))

            if im:
                return mpc(re or fzero, im)
            return mpf(re or fzero)

        if options.get('quad') == 'osc':
            A = Wild('A', exclude=[x])
            B = Wild('B', exclude=[x])
            D = Wild('D')
            m = func.match(cos(A*x + B)*D)
            if not m:
                m = func.match(sin(A*x + B)*D)
            if not m:
                raise ValueError("An integrand of the form sin(A*x+B)*f(x) "
                  "or cos(A*x+B)*f(x) is required for oscillatory quadrature")
            period = as_mpmath(2*S.Pi/m[A], prec + 15, options)
            result = quadosc(f, [xlow, xhigh], period=period)
            # XXX: quadosc does not do error detection yet
            quadrature_error = MINUS_INF
        else:
            result, quadrature_err = quadts(f, [xlow, xhigh], error=1)
            quadrature_error = fastlog(quadrature_err._mpf_)

    options['maxprec'] = oldmaxprec

    if have_part[0]:
        re: MPF_TUP | None = result.real._mpf_
        re_acc: int | None
        if re == fzero:
            re_s, re_acc = scaled_zero(int(-max(prec, max_real_term, quadrature_error)))
            re = scaled_zero(re_s)  # handled ok in evalf_integral
        else:
            re_acc = int(-max(max_real_term - fastlog(re) - prec, quadrature_error))
    else:
        re, re_acc = None, None

    if have_part[1]:
        im: MPF_TUP | None = result.imag._mpf_
        im_acc: int | None
        if im == fzero:
            im_s, im_acc = scaled_zero(int(-max(prec, max_imag_term, quadrature_error)))
            im = scaled_zero(im_s)  # handled ok in evalf_integral
        else:
            im_acc = int(-max(max_imag_term - fastlog(im) - prec, quadrature_error))
    else:
        im, im_acc = None, None

    result = re, im, re_acc, im_acc
    return result


def evalf_integral(expr: 'Integral', prec: int, options: OPT_DICT) -> TMP_RES:
    limits = expr.limits
    if len(limits) != 1 or len(limits[0]) != 3:
        raise NotImplementedError
    workprec = prec
    i = 0
    maxprec = options.get('maxprec', INF)
    while 1:
        result = do_integral(expr, workprec, options)
        accuracy = complex_accuracy(result)
        if accuracy >= prec:  # achieved desired precision
            break
        if workprec >= maxprec:  # can't increase accuracy any more
            break
        if accuracy == -1:
            # maybe the answer really is zero and maybe we just haven't increased
            # the precision enough. So increase by doubling to not take too long
            # to get to maxprec.
            workprec *= 2
        else:
            workprec += max(prec, 2**i)
        workprec = min(workprec, maxprec)
        i += 1
    return result


def check_convergence(numer: Expr, denom: Expr, n: Symbol) -> tuple[int, Any, Any]:
    """
    Returns
    =======

    (h, g, p) where
    -- h is:
        > 0 for convergence of rate 1/factorial(n)**h
        < 0 for divergence of rate factorial(n)**(-h)
        = 0 for geometric or polynomial convergence or divergence

    -- abs(g) is:
        > 1 for geometric convergence of rate 1/h**n
        < 1 for geometric divergence of rate h**n
        = 1 for polynomial convergence or divergence

        (g < 0 indicates an alternating series)

    -- p is:
        > 1 for polynomial convergence of rate 1/n**h
        <= 1 for polynomial divergence of rate n**(-h)

    """
    from sympy.polys.polytools import Poly
    npol = Poly(numer, n)
    dpol = Poly(denom, n)
    p = npol.degree()
    q = dpol.degree()
    rate = q - p
    if rate:
        return rate, None, None
    constant = dpol.LC() / npol.LC()
    from .numbers import equal_valued
    if not equal_valued(abs(constant), 1):
        return rate, constant, None
    if npol.degree() == dpol.degree() == 0:
        return rate, constant, 0
    pc = npol.all_coeffs()[1]
    qc = dpol.all_coeffs()[1]
    return rate, constant, (qc - pc)/dpol.LC()


def hypsum(expr: Expr, n: Symbol, start: int, prec: int) -> mpf:
    """
    Sum a rapidly convergent infinite hypergeometric series with
    given general term, e.g. e = hypsum(1/factorial(n), n). The
    quotient between successive terms must be a quotient of integer
    polynomials.
    """
    from .numbers import Float, equal_valued
    from sympy.simplify.simplify import hypersimp

    if prec == float('inf'):
        raise NotImplementedError('does not support inf prec')

    if start:
        expr = expr.subs(n, n + start)
    hs = hypersimp(expr, n)
    if hs is None:
        raise NotImplementedError("a hypergeometric series is required")
    num, den = hs.as_numer_denom()

    func1 = lambdify(n, num)
    func2 = lambdify(n, den)

    h, g, p = check_convergence(num, den, n)

    if h < 0:
        raise ValueError("Sum diverges like (n!)^%i" % (-h))

    eterm = expr.subs(n, 0)
    if not eterm.is_Rational:
        raise NotImplementedError("Non rational term functionality is not implemented.")

    term: Rational = eterm # type: ignore

    # Direct summation if geometric or faster
    if h > 0 or (h == 0 and abs(g) > 1):
        term = (MPZ(term.p) << prec) // term.q
        s = term
        k = 1
        while abs(term) > 5:
            term *= MPZ(func1(k - 1))
            term //= MPZ(func2(k - 1))
            s += term
            k += 1
        return from_man_exp(s, -prec)
    else:
        alt = g < 0
        if abs(g) < 1:
            raise ValueError("Sum diverges like (%i)^n" % abs(1/g))
        if p < 1 or (equal_valued(p, 1) and not alt):
            raise ValueError("Sum diverges like n^%i" % (-p))
        # We have polynomial convergence: use Richardson extrapolation
        vold = None
        ndig = prec_to_dps(prec)
        while True:
            # Need to use at least quad precision because a lot of cancellation
            # might occur in the extrapolation process; we check the answer to
            # make sure that the desired precision has been reached, too.
            prec2 = 4*prec
            term0 = (MPZ(term.p) << prec2) // term.q

            def summand(k, _term=[term0]):
                if k:
                    k = int(k)
                    _term[0] *= MPZ(func1(k - 1))
                    _term[0] //= MPZ(func2(k - 1))
                return make_mpf(from_man_exp(_term[0], -prec2))

            with workprec(prec):
                v = nsum(summand, [0, mpmath_inf], method='richardson')
            vf = Float(v, ndig)
            if vold is not None and vold == vf:
                break
            prec += prec  # double precision each time
            vold = vf

        return v._mpf_


def evalf_prod(expr: 'Product', prec: int, options: OPT_DICT) -> TMP_RES:
    if all((l[1] - l[2]).is_Integer for l in expr.limits):
        result = evalf(expr.doit(), prec=prec, options=options)
    else:
        from sympy.concrete.summations import Sum
        result = evalf(expr.rewrite(Sum), prec=prec, options=options)
    return result


def evalf_sum(expr: 'Sum', prec: int, options: OPT_DICT) -> TMP_RES:
    from .numbers import Float
    if 'subs' in options:
        expr = expr.subs(options['subs']) # type: ignore
    func = expr.function
    limits = expr.limits
    if len(limits) != 1 or len(limits[0]) != 3:
        raise NotImplementedError
    if func.is_zero:
        return None, None, prec, None
    prec2 = prec + 10
    try:
        n, a, b = limits[0]
        if b is not S.Infinity or a is S.NegativeInfinity or a != int(a):
            raise NotImplementedError
        # Use fast hypergeometric summation if possible
        v = hypsum(func, n, int(a), prec2)
        delta = prec - fastlog(v)
        if fastlog(v) < -10:
            v = hypsum(func, n, int(a), delta)
        return v, None, min(prec, delta), None
    except NotImplementedError:
        # Euler-Maclaurin summation for general series
        eps = Float(2.0)**(-prec)
        for i in range(1, 5):
            m = n = 2**i * prec
            s, err = expr.euler_maclaurin(m=m, n=n, eps=eps,
                eval_integral=False)
            err = err.evalf()
            if err is S.NaN:
                raise NotImplementedError
            if err <= eps:
                break
        err = fastlog(evalf(abs(err), 20, options)[0])
        re, im, re_acc, im_acc = evalf(s, prec2, options)
        if re_acc is None:
            re_acc = -err
        if im_acc is None:
            im_acc = -err
        return re, im, re_acc, im_acc


#----------------------------------------------------------------------------#
#                                                                            #
#                            Symbolic interface                              #
#                                                                            #
#----------------------------------------------------------------------------#

def evalf_symbol(x: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    val = options['subs'][x]
    if isinstance(val, mpf):
        if not val:
            return None, None, None, None
        return val._mpf_, None, prec, None
    else:
        if '_cache' not in options:
            options['_cache'] = {}
        cache = options['_cache']
        cached, cached_prec = cache.get(x, (None, MINUS_INF))
        if cached_prec >= prec:
            return cached
        v = evalf(sympify(val), prec, options)
        cache[x] = (v, prec)
        return v

evalf_table: dict[Type[Expr], Callable[[Expr, int, OPT_DICT], TMP_RES]] = {}


def _create_evalf_table():
    global evalf_table
    from sympy.concrete.products import Product
    from sympy.concrete.summations import Sum
    from .add import Add
    from .mul import Mul
    from .numbers import Exp1, Float, Half, ImaginaryUnit, Integer, NaN, NegativeOne, One, Pi, Rational, \
        Zero, ComplexInfinity, AlgebraicNumber
    from .power import Pow
    from .symbol import Dummy, Symbol
    from sympy.functions.elementary.complexes import Abs, im, re
    from sympy.functions.elementary.exponential import exp, log
    from sympy.functions.elementary.integers import ceiling, floor
    from sympy.functions.elementary.piecewise import Piecewise
    from sympy.functions.elementary.trigonometric import atan, cos, sin, tan
    from sympy.integrals.integrals import Integral
    evalf_table = {
        Symbol: evalf_symbol,
        Dummy: evalf_symbol,
        Float: evalf_float,
        Rational: evalf_rational,
        Integer: evalf_integer,
        Zero: lambda x, prec, options: (None, None, prec, None),
        One: lambda x, prec, options: (fone, None, prec, None),
        Half: lambda x, prec, options: (fhalf, None, prec, None),
        Pi: lambda x, prec, options: (mpf_pi(prec), None, prec, None),
        Exp1: lambda x, prec, options: (mpf_e(prec), None, prec, None),
        ImaginaryUnit: lambda x, prec, options: (None, fone, None, prec),
        NegativeOne: lambda x, prec, options: (fnone, None, prec, None),
        ComplexInfinity: lambda x, prec, options: S.ComplexInfinity,
        NaN: lambda x, prec, options: (fnan, None, prec, None),

        exp: evalf_exp,

        cos: evalf_trig,
        sin: evalf_trig,
        tan: evalf_trig,

        Add: evalf_add,
        Mul: evalf_mul,
        Pow: evalf_pow,

        log: evalf_log,
        atan: evalf_atan,
        Abs: evalf_abs,

        re: evalf_re,
        im: evalf_im,
        floor: evalf_floor,
        ceiling: evalf_ceiling,

        Integral: evalf_integral,
        Sum: evalf_sum,
        Product: evalf_prod,
        Piecewise: evalf_piecewise,

        AlgebraicNumber: evalf_alg_num,
    }


def evalf(x: Expr, prec: int, options: OPT_DICT) -> TMP_RES:
    """
    Evaluate the ``Expr`` instance, ``x``
    to a binary precision of ``prec``. This
    function is supposed to be used internally.

    Parameters
    ==========

    x : Expr
        The formula to evaluate to a float.
    prec : int
        The binary precision that the output should have.
    options : dict
        A dictionary with the same entries as
        ``EvalfMixin.evalf`` and in addition,
        ``maxprec`` which is the maximum working precision.

    Returns
    =======

    An optional tuple, ``(re, im, re_acc, im_acc)``
    which are the real, imaginary, real accuracy
    and imaginary accuracy respectively. ``re`` is
    an mpf value tuple and so is ``im``. ``re_acc``
    and ``im_acc`` are ints.

    NB: all these return values can be ``None``.
    If all values are ``None``, then that represents 0.
    Note that 0 is also represented as ``fzero = (0, 0, 0, 0)``.
    """
    from sympy.functions.elementary.complexes import re as re_, im as im_
    try:
        rf = evalf_table[type(x)]
        r = rf(x, prec, options)
    except KeyError:
        # Fall back to ordinary evalf if possible
        if 'subs' in options:
            x = x.subs(evalf_subs(prec, options['subs']))
        xe = x._eval_evalf(prec)
        if xe is None:
            raise NotImplementedError
        as_real_imag = getattr(xe, "as_real_imag", None)
        if as_real_imag is None:
            raise NotImplementedError # e.g. FiniteSet(-1.0, 1.0).evalf()
        re, im = as_real_imag()
        if re.has(re_) or im.has(im_):
            raise NotImplementedError
        if not re:
            re = None
            reprec = None
        elif re.is_number:
            re = re._to_mpmath(prec, allow_ints=False)._mpf_
            reprec = prec
        else:
            raise NotImplementedError
        if not im:
            im = None
            imprec = None
        elif im.is_number:
            im = im._to_mpmath(prec, allow_ints=False)._mpf_
            imprec = prec
        else:
            raise NotImplementedError
        r = re, im, reprec, imprec

    if options.get("verbose"):
        print("### input", x)
        print("### output", to_str(r[0] or fzero, 50) if isinstance(r, tuple) else r)
        print("### raw", r) # r[0], r[2]
        print()
    chop = options.get('chop', False)
    if chop:
        if chop is True:
            chop_prec = prec
        else:
            # convert (approximately) from given tolerance;
            # the formula here will will make 1e-i rounds to 0 for
            # i in the range +/-27 while 2e-i will not be chopped
            chop_prec = int(round(-3.321*math.log10(chop) + 2.5))
            if chop_prec == 3:
                chop_prec -= 1
        r = chop_parts(r, chop_prec)
    if options.get("strict"):
        check_target(x, r, prec)
    return r


def quad_to_mpmath(q, ctx=None):
    """Turn the quad returned by ``evalf`` into an ``mpf`` or ``mpc``. """
    mpc = make_mpc if ctx is None else ctx.make_mpc
    mpf = make_mpf if ctx is None else ctx.make_mpf
    if q is S.ComplexInfinity:
        raise NotImplementedError
    re, im, _, _ = q
    if im:
        if not re:
            re = fzero
        return mpc((re, im))
    elif re:
        return mpf(re)
    else:
        return mpf(fzero)


class EvalfMixin:
    """Mixin class adding evalf capability."""

    __slots__: tuple[str, ...] = ()

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """
        Evaluate the given formula to an accuracy of *n* digits.

        Parameters
        ==========

        subs : dict, optional
            Substitute numerical values for symbols, e.g.
            ``subs={x:3, y:1+pi}``. The substitutions must be given as a
            dictionary.

        maxn : int, optional
            Allow a maximum temporary working precision of maxn digits.

        chop : bool or number, optional
            Specifies how to replace tiny real or imaginary parts in
            subresults by exact zeros.

            When ``True`` the chop value defaults to standard precision.

            Otherwise the chop value is used to determine the
            magnitude of "small" for purposes of chopping.

            >>> from sympy import N
            >>> x = 1e-4
            >>> N(x, chop=True)
            0.000100000000000000
            >>> N(x, chop=1e-5)
            0.000100000000000000
            >>> N(x, chop=1e-4)
            0

        strict : bool, optional
            Raise ``PrecisionExhausted`` if any subresult fails to
            evaluate to full accuracy, given the available maxprec.

        quad : str, optional
            Choose algorithm for numerical quadrature. By default,
            tanh-sinh quadrature is used. For oscillatory
            integrals on an infinite interval, try ``quad='osc'``.

        verbose : bool, optional
            Print debug information.

        Notes
        =====

        When Floats are naively substituted into an expression,
        precision errors may adversely affect the result. For example,
        adding 1e16 (a Float) to 1 will truncate to 1e16; if 1e16 is
        then subtracted, the result will be 0.
        That is exactly what happens in the following:

        >>> from sympy.abc import x, y, z
        >>> values = {x: 1e16, y: 1, z: 1e16}
        >>> (x + y - z).subs(values)
        0

        Using the subs argument for evalf is the accurate way to
        evaluate such an expression:

        >>> (x + y - z).evalf(subs=values)
        1.00000000000000
        """
        from .numbers import Float, Number
        n = n if n is not None else 15

        if subs and is_sequence(subs):
            raise TypeError('subs must be given as a dictionary')

        # for sake of sage that doesn't like evalf(1)
        if n == 1 and isinstance(self, Number):
            from .expr import _mag
            rv = self.evalf(2, subs, maxn, chop, strict, quad, verbose)
            m = _mag(rv)
            rv = rv.round(1 - m)
            return rv

        if not evalf_table:
            _create_evalf_table()
        prec = dps_to_prec(n)
        options = {'maxprec': max(prec, int(maxn*LG10)), 'chop': chop,
               'strict': strict, 'verbose': verbose}
        if subs is not None:
            options['subs'] = subs
        if quad is not None:
            options['quad'] = quad
        try:
            result = evalf(self, prec + 4, options)
        except NotImplementedError:
            # Fall back to the ordinary evalf
            if hasattr(self, 'subs') and subs is not None:  # issue 20291
                v = self.subs(subs)._eval_evalf(prec)
            else:
                v = self._eval_evalf(prec)
            if v is None:
                return self
            elif not v.is_number:
                return v
            try:
                # If the result is numerical, normalize it
                result = evalf(v, prec, options)
            except NotImplementedError:
                # Probably contains symbols or unknown functions
                return v
        if result is S.ComplexInfinity:
            return result
        re, im, re_acc, im_acc = result
        if re is S.NaN or im is S.NaN:
            return S.NaN
        if re:
            p = max(min(prec, re_acc), 1)
            re = Float._new(re, p)
        else:
            re = S.Zero
        if im:
            p = max(min(prec, im_acc), 1)
            im = Float._new(im, p)
            return re + im*S.ImaginaryUnit
        else:
            return re

    n = evalf

    def _evalf(self, prec: int) -> Expr:
        """Helper for evalf. Does the same thing but takes binary precision"""
        r = self._eval_evalf(prec)
        if r is None:
            r = self # type: ignore
        return r # type: ignore

    def _eval_evalf(self, prec: int) -> Expr | None:
        return None

    def _to_mpmath(self, prec, allow_ints=True):
        # mpmath functions accept ints as input
        errmsg = "cannot convert to mpmath number"
        if allow_ints and self.is_Integer:
            return self.p
        if hasattr(self, '_as_mpf_val'):
            return make_mpf(self._as_mpf_val(prec))
        try:
            result = evalf(self, prec, {})
            return quad_to_mpmath(result)
        except NotImplementedError:
            v = self._eval_evalf(prec)
            if v is None:
                raise ValueError(errmsg)
            if v.is_Float:
                return make_mpf(v._mpf_)
            # Number + Number*I is also fine
            re, im = v.as_real_imag()
            if allow_ints and re.is_Integer:
                re = from_int(re.p)
            elif re.is_Float:
                re = re._mpf_
            else:
                raise ValueError(errmsg)
            if allow_ints and im.is_Integer:
                im = from_int(im.p)
            elif im.is_Float:
                im = im._mpf_
            else:
                raise ValueError(errmsg)
            return make_mpc((re, im))


def N(x, n=15, **options):
    r"""
    Calls x.evalf(n, \*\*options).

    Explanations
    ============

    Both .n() and N() are equivalent to .evalf(); use the one that you like better.
    See also the docstring of .evalf() for information on the options.

    Examples
    ========

    >>> from sympy import Sum, oo, N
    >>> from sympy.abc import k
    >>> Sum(1/k**k, (k, 1, oo))
    Sum(k**(-k), (k, 1, oo))
    >>> N(_, 4)
    1.291

    """
    # by using rational=True, any evaluation of a string
    # will be done using exact values for the Floats
    return sympify(x, rational=True).evalf(n, **options)


def _evalf_with_bounded_error(x: Expr, eps: Expr | None = None,
                              m: int = 0,
                              options: OPT_DICT | None = None) -> TMP_RES:
    """
    Evaluate *x* to within a bounded absolute error.

    Parameters
    ==========

    x : Expr
        The quantity to be evaluated.
    eps : Expr, None, optional (default=None)
        Positive real upper bound on the acceptable error.
    m : int, optional (default=0)
        If *eps* is None, then use 2**(-m) as the upper bound on the error.
    options: OPT_DICT
        As in the ``evalf`` function.

    Returns
    =======

    A tuple ``(re, im, re_acc, im_acc)``, as returned by ``evalf``.

    See Also
    ========

    evalf

    """
    if eps is not None:
        if not (eps.is_Rational or eps.is_Float) or not eps > 0:
            raise ValueError("eps must be positive")
        r, _, _, _ = evalf(1/eps, 1, {})
        m = fastlog(r)

    c, d, _, _ = evalf(x, 1, {})
    # Note: If x = a + b*I, then |a| <= 2|c| and |b| <= 2|d|, with equality
    # only in the zero case.
    # If a is non-zero, then |c| = 2**nc for some integer nc, and c has
    # bitcount 1. Therefore 2**fastlog(c) = 2**(nc+1) = 2|c| is an upper bound
    # on |a|. Likewise for b and d.
    nr, ni = fastlog(c), fastlog(d)
    n = max(nr, ni) + 1
    # If x is 0, then n is MINUS_INF, and p will be 1. Otherwise,
    # n - 1 bits get us past the integer parts of a and b, and +1 accounts for
    # the factor of <= sqrt(2) that is |x|/max(|a|, |b|).
    p = max(1, m + n + 1)

    options = options or {}
    return evalf(x, p, options)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\matrices\eigen_symmetric.py ===
#!/usr/bin/python
# -*- coding: utf-8 -*-

##################################################################################################
#     module for the symmetric eigenvalue problem
#       Copyright 2013 Timo Hartmann (thartmann15 at gmail.com)
#
# todo:
#  - implement balancing
#
##################################################################################################

"""
The symmetric eigenvalue problem.
---------------------------------

This file contains routines for the symmetric eigenvalue problem.

high level routines:

  eigsy : real symmetric (ordinary) eigenvalue problem
  eighe : complex hermitian (ordinary) eigenvalue problem
  eigh  : unified interface for eigsy and eighe
  svd_r : singular value decomposition for real matrices
  svd_c : singular value decomposition for complex matrices
  svd   : unified interface for svd_r and svd_c


low level routines:

  r_sy_tridiag : reduction of real symmetric matrix to real symmetric tridiagonal matrix
  c_he_tridiag_0 : reduction of complex hermitian matrix to real symmetric tridiagonal matrix
  c_he_tridiag_1 : auxiliary routine to c_he_tridiag_0
  c_he_tridiag_2 : auxiliary routine to c_he_tridiag_0
  tridiag_eigen : solves the real symmetric tridiagonal matrix eigenvalue problem
  svd_r_raw : raw singular value decomposition for real matrices
  svd_c_raw : raw singular value decomposition for complex matrices
"""

from ..libmp.backend import xrange
from .eigen import defun


def r_sy_tridiag(ctx, A, D, E, calc_ev = True):
    """
    This routine transforms a real symmetric matrix A to a real symmetric
    tridiagonal matrix T using an orthogonal similarity transformation:
          Q' * A * Q = T     (here ' denotes the matrix transpose).
    The orthogonal matrix Q is build up from Householder reflectors.

    parameters:
      A         (input/output) On input, A contains the real symmetric matrix of
                dimension (n,n). On output, if calc_ev is true, A contains the
                orthogonal matrix Q, otherwise A is destroyed.

      D         (output) real array of length n, contains the diagonal elements
                of the tridiagonal matrix

      E         (output) real array of length n, contains the offdiagonal elements
                of the tridiagonal matrix in E[0:(n-1)] where is the dimension of
                the matrix A. E[n-1] is undefined.

      calc_ev   (input) If calc_ev is true, this routine explicitly calculates the
                orthogonal matrix Q which is then returned in A. If calc_ev is
                false, Q is not explicitly calculated resulting in a shorter run time.

    This routine is a python translation of the fortran routine tred2.f in the
    software library EISPACK (see netlib.org) which itself is based on the algol
    procedure tred2 described in:
      - Num. Math. 11, p.181-195 (1968) by Martin, Reinsch and Wilkonson
      - Handbook for auto. comp., Vol II, Linear Algebra, p.212-226 (1971)

    For a good introduction to Householder reflections, see also
      Stoer, Bulirsch - Introduction to Numerical Analysis.
    """

    # note : the vector v of the i-th houshoulder reflector is stored in a[(i+1):,i]
    #        whereas v/<v,v> is stored in a[i,(i+1):]

    n = A.rows
    for i in xrange(n - 1, 0, -1):
        # scale the vector

        scale = 0
        for k in xrange(0, i):
            scale += abs(A[k,i])

        scale_inv = 0
        if scale != 0:
            scale_inv = 1/scale

        # sadly there are floating point numbers not equal to zero whose reciprocal is infinity

        if i == 1 or scale == 0 or ctx.isinf(scale_inv):
            E[i] = A[i-1,i]        # nothing to do
            D[i] = 0
            continue

        # calculate parameters for housholder transformation

        H = 0
        for k in xrange(0, i):
            A[k,i] *= scale_inv
            H += A[k,i] * A[k,i]

        F = A[i-1,i]
        G = ctx.sqrt(H)
        if F > 0:
            G = -G
        E[i] = scale * G
        H -= F * G
        A[i-1,i] = F - G
        F = 0

        # apply housholder transformation

        for j in xrange(0, i):
            if calc_ev:
                A[i,j] = A[j,i] / H

            G = 0                  # calculate A*U
            for k in xrange(0, j + 1):
                G += A[k,j] * A[k,i]
            for k in xrange(j + 1, i):
                G += A[j,k] * A[k,i]

            E[j] = G / H           # calculate P
            F += E[j] * A[j,i]

        HH = F / (2 * H)

        for j in xrange(0, i):     # calculate reduced A
            F = A[j,i]
            G = E[j] - HH * F      # calculate Q
            E[j] = G

            for k in xrange(0, j + 1):
                A[k,j] -= F * E[k] + G * A[k,i]

        D[i] = H

    for i in xrange(1, n):         # better for compatibility
        E[i-1] = E[i]
    E[n-1] = 0

    if calc_ev:
        D[0] = 0
        for i in xrange(0, n):
            if D[i] != 0:
                for j in xrange(0, i):     # accumulate transformation matrices
                    G = 0
                    for k in xrange(0, i):
                        G += A[i,k] * A[k,j]
                    for k in xrange(0, i):
                        A[k,j] -= G * A[k,i]

            D[i] = A[i,i]
            A[i,i] = 1

            for j in xrange(0, i):
                A[j,i] = A[i,j] = 0
    else:
        for i in xrange(0, n):
            D[i] = A[i,i]





def c_he_tridiag_0(ctx, A, D, E, T):
    """
    This routine transforms a complex hermitian matrix A to a real symmetric
    tridiagonal matrix T using an unitary similarity transformation:
          Q' * A * Q = T     (here ' denotes the hermitian matrix transpose,
                              i.e. transposition und conjugation).
    The unitary matrix Q is build up from Householder reflectors and
    an unitary diagonal matrix.

    parameters:
      A         (input/output) On input, A contains the complex hermitian matrix
                of dimension (n,n). On output, A contains the unitary matrix Q
                in compressed form.

      D         (output) real array of length n, contains the diagonal elements
                of the tridiagonal matrix.

      E         (output) real array of length n, contains the offdiagonal elements
                of the tridiagonal matrix in E[0:(n-1)] where is the dimension of
                the matrix A. E[n-1] is undefined.

      T         (output) complex array of length n, contains a unitary diagonal
                matrix.

    This routine is a python translation (in slightly modified form) of the fortran
    routine htridi.f in the software library EISPACK (see netlib.org) which itself
    is a complex version of the algol procedure tred1 described in:
      - Num. Math. 11, p.181-195 (1968) by Martin, Reinsch and Wilkonson
      - Handbook for auto. comp., Vol II, Linear Algebra, p.212-226 (1971)

    For a good introduction to Householder reflections, see also
      Stoer, Bulirsch - Introduction to Numerical Analysis.
    """

    n = A.rows
    T[n-1] = 1
    for i in xrange(n - 1, 0, -1):

        # scale the vector

        scale = 0
        for k in xrange(0, i):
            scale += abs(ctx.re(A[k,i])) + abs(ctx.im(A[k,i]))

        scale_inv = 0
        if scale != 0:
            scale_inv = 1 / scale

        # sadly there are floating point numbers not equal to zero whose reciprocal is infinity

        if scale == 0 or ctx.isinf(scale_inv):
            E[i] = 0
            D[i] = 0
            T[i-1] = 1
            continue

        if i == 1:
            F = A[i-1,i]
            f = abs(F)
            E[i] = f
            D[i] = 0
            if f != 0:
                T[i-1] = T[i] * F / f
            else:
                T[i-1] = T[i]
            continue

        # calculate parameters for housholder transformation

        H = 0
        for k in xrange(0, i):
            A[k,i] *= scale_inv
            rr = ctx.re(A[k,i])
            ii = ctx.im(A[k,i])
            H += rr * rr + ii * ii

        F = A[i-1,i]
        f = abs(F)
        G = ctx.sqrt(H)
        H += G * f
        E[i] = scale * G
        if f != 0:
            F = F / f
            TZ = - T[i] * F              # T[i-1]=-T[i]*F, but we need T[i-1] as temporary storage
            G *= F
        else:
            TZ = -T[i]                   # T[i-1]=-T[i]
        A[i-1,i] += G
        F = 0

        # apply housholder transformation

        for j in xrange(0, i):
            A[i,j] = A[j,i] / H

            G = 0                        # calculate A*U
            for k in xrange(0, j + 1):
                G += ctx.conj(A[k,j]) * A[k,i]
            for k in xrange(j + 1, i):
                G += A[j,k] * A[k,i]

            T[j] = G / H                 # calculate P
            F += ctx.conj(T[j]) * A[j,i]

        HH = F / (2 * H)

        for j in xrange(0, i):           # calculate reduced A
            F = A[j,i]
            G = T[j] - HH * F            # calculate Q
            T[j] = G

            for k in xrange(0, j + 1):
                A[k,j] -= ctx.conj(F) * T[k] + ctx.conj(G) * A[k,i]
                # as we use the lower left part for storage
                # we have to use the transpose of the normal formula

        T[i-1] = TZ
        D[i] = H

    for i in xrange(1, n):                # better for compatibility
        E[i-1] = E[i]
    E[n-1] = 0

    D[0] = 0
    for i in xrange(0, n):
        zw = D[i]
        D[i] = ctx.re(A[i,i])
        A[i,i] = zw







def c_he_tridiag_1(ctx, A, T):
    """
    This routine forms the unitary matrix Q described in c_he_tridiag_0.

    parameters:
      A    (input/output) On input, A is the same matrix as delivered by
           c_he_tridiag_0. On output, A is set to Q.

      T    (input) On input, T is the same array as delivered by c_he_tridiag_0.

    """

    n = A.rows

    for i in xrange(0, n):
        if A[i,i] != 0:
            for j in xrange(0, i):
                G = 0
                for k in xrange(0, i):
                    G += ctx.conj(A[i,k]) * A[k,j]
                for k in xrange(0, i):
                    A[k,j] -= G * A[k,i]

        A[i,i] = 1

        for j in xrange(0, i):
            A[j,i] = A[i,j] = 0

    for i in xrange(0, n):
        for k in xrange(0, n):
            A[i,k] *= T[k]




def c_he_tridiag_2(ctx, A, T, B):
    """
    This routine applied the unitary matrix Q described in c_he_tridiag_0
    onto the the matrix B, i.e. it forms Q*B.

    parameters:
      A    (input) On input, A is the same matrix as delivered by c_he_tridiag_0.

      T    (input) On input, T is the same array as delivered by c_he_tridiag_0.

      B    (input/output) On input, B is a complex matrix. On output B is replaced
           by Q*B.

    This routine is a python translation of the fortran routine htribk.f in the
    software library EISPACK (see netlib.org). See c_he_tridiag_0 for more
    references.
    """

    n = A.rows

    for i in xrange(0, n):
        for k in xrange(0, n):
            B[k,i] *= T[k]

    for i in xrange(0, n):
        if A[i,i] != 0:
            for j in xrange(0, n):
                G = 0
                for k in xrange(0, i):
                    G += ctx.conj(A[i,k]) * B[k,j]
                for k in xrange(0, i):
                    B[k,j] -= G * A[k,i]





def tridiag_eigen(ctx, d, e, z = False):
    """
    This subroutine find the eigenvalues and the first components of the
    eigenvectors of a real symmetric tridiagonal matrix using the implicit
    QL method.

    parameters:

      d (input/output) real array of length n. on input, d contains the diagonal
        elements of the input matrix. on output, d contains the eigenvalues in
        ascending order.

      e (input) real array of length n. on input, e contains the offdiagonal
        elements of the input matrix in e[0:(n-1)]. On output, e has been
        destroyed.

      z (input/output) If z is equal to False, no eigenvectors will be computed.
        Otherwise on input z should have the format z[0:m,0:n] (i.e. a real or
        complex matrix of dimension (m,n) ). On output this matrix will be
        multiplied by the matrix of the eigenvectors (i.e. the columns of this
        matrix are the eigenvectors): z --> z*EV
        That means if z[i,j]={1 if j==j; 0 otherwise} on input, then on output
        z will contain the first m components of the eigenvectors. That means
        if m is equal to n, the i-th eigenvector will be z[:,i].

    This routine is a python translation (in slightly modified form) of the
    fortran routine imtql2.f in the software library EISPACK (see netlib.org)
    which itself is based on the algol procudure imtql2 desribed in:
     - num. math. 12, p. 377-383(1968) by matrin and wilkinson
     - modified in num. math. 15, p. 450(1970) by dubrulle
     - handbook for auto. comp., vol. II-linear algebra, p. 241-248 (1971)
    See also the routine gaussq.f in netlog.org or acm algorithm 726.
    """

    n = len(d)
    e[n-1] = 0
    iterlim = 2 * ctx.dps

    for l in xrange(n):
        j = 0
        while 1:
            m = l
            while 1:
                # look for a small subdiagonal element
                if m + 1 == n:
                    break
                if abs(e[m]) <= ctx.eps * (abs(d[m]) + abs(d[m + 1])):
                    break
                m = m + 1
            if m == l:
                break

            if j >= iterlim:
                raise RuntimeError("tridiag_eigen: no convergence to an eigenvalue after %d iterations" % iterlim)

            j += 1

            # form shift

            p = d[l]
            g = (d[l + 1] - p) / (2 * e[l])
            r = ctx.hypot(g, 1)

            if g < 0:
                s = g - r
            else:
                s = g + r

            g = d[m] - p + e[l] / s

            s, c, p = 1, 1, 0

            for i in xrange(m - 1, l - 1, -1):
                f = s * e[i]
                b = c * e[i]
                if abs(f) > abs(g):             # this here is a slight improvement also used in gaussq.f or acm algorithm 726.
                    c = g / f
                    r = ctx.hypot(c, 1)
                    e[i + 1] = f * r
                    s = 1 / r
                    c = c * s
                else:
                    s = f / g
                    r = ctx.hypot(s, 1)
                    e[i + 1] = g * r
                    c = 1 / r
                    s = s * c
                g = d[i + 1] - p
                r = (d[i] - g) * s + 2 * c * b
                p = s * r
                d[i + 1] = g + p
                g = c * r - b

                if not isinstance(z, bool):
                    # calculate eigenvectors
                    for w in xrange(z.rows):
                        f = z[w,i+1]
                        z[w,i+1] = s * z[w,i] + c * f
                        z[w,i  ] = c * z[w,i] - s * f

            d[l] = d[l] - p
            e[l] = g
            e[m] = 0

    for ii in xrange(1, n):
        # sort eigenvalues and eigenvectors (bubble-sort)
        i = ii - 1
        k = i
        p = d[i]
        for j in xrange(ii, n):
            if d[j] >= p:
                continue
            k = j
            p = d[k]
        if k == i:
            continue
        d[k] = d[i]
        d[i] = p

        if not isinstance(z, bool):
            for w in xrange(z.rows):
                p = z[w,i]
                z[w,i] = z[w,k]
                z[w,k] = p

########################################################################################

@defun
def eigsy(ctx, A, eigvals_only = False, overwrite_a = False):
    """
    This routine solves the (ordinary) eigenvalue problem for a real symmetric
    square matrix A. Given A, an orthogonal matrix Q is calculated which
    diagonalizes A:

          Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) is a diagonal matrix whose diagonal is E.
    ' denotes the transpose.

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

          A Q[:,i] = E[i] Q[:,i]


    input:

      A: real matrix of format (n,n) which is symmetric
         (i.e. A=A' or A[i,j]=A[j,i])

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: orthogonal matrix of format (n,n). contains the eigenvectors
         of A as columns.

    return value:

          E          if eigvals_only is true
         (E, Q)      if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, 2], [2, 0]])
      >>> E = mp.eigsy(A, eigvals_only = True)
      >>> print(E)
      [-1.0]
      [ 4.0]

      >>> A = mp.matrix([[1, 2], [2, 3]])
      >>> E, Q = mp.eigsy(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eighe, eigh, eig
    """

    if not overwrite_a:
        A = A.copy()

    d = ctx.zeros(A.rows, 1)
    e = ctx.zeros(A.rows, 1)

    if eigvals_only:
        r_sy_tridiag(ctx, A, d, e, calc_ev = False)
        tridiag_eigen(ctx, d, e, False)
        return d
    else:
        r_sy_tridiag(ctx, A, d, e, calc_ev = True)
        tridiag_eigen(ctx, d, e, A)
        return (d, A)


@defun
def eighe(ctx, A, eigvals_only = False, overwrite_a = False):
    """
    This routine solves the (ordinary) eigenvalue problem for a complex
    hermitian square matrix A. Given A, an unitary matrix Q is calculated which
    diagonalizes A:

        Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) a is diagonal matrix whose diagonal is E.
    ' denotes the hermitian transpose (i.e. ordinary transposition and
    complex conjugation).

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

        A Q[:,i] = E[i] Q[:,i]


    input:

      A: complex matrix of format (n,n) which is hermitian
         (i.e. A=A' or A[i,j]=conj(A[j,i]))

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: unitary matrix of format (n,n). contains the eigenvectors
         of A as columns.

    return value:

           E         if eigvals_only is true
          (E, Q)     if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[1, -3 - 1j], [-3 + 1j, -2]])
      >>> E = mp.eighe(A, eigvals_only = True)
      >>> print(E)
      [-4.0]
      [ 3.0]

      >>> A = mp.matrix([[1, 2 + 5j], [2 - 5j, 3]])
      >>> E, Q = mp.eighe(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eigsy, eigh, eig
    """

    if not overwrite_a:
        A = A.copy()

    d = ctx.zeros(A.rows, 1)
    e = ctx.zeros(A.rows, 1)
    t = ctx.zeros(A.rows, 1)

    if eigvals_only:
        c_he_tridiag_0(ctx, A, d, e, t)
        tridiag_eigen(ctx, d, e, False)
        return d
    else:
        c_he_tridiag_0(ctx, A, d, e, t)
        B = ctx.eye(A.rows)
        tridiag_eigen(ctx, d, e, B)
        c_he_tridiag_2(ctx, A, t, B)
        return (d, B)

@defun
def eigh(ctx, A, eigvals_only = False, overwrite_a = False):
    """
    "eigh" is a unified interface for "eigsy" and "eighe". Depending on
    whether A is real or complex the appropriate function is called.

    This routine solves the (ordinary) eigenvalue problem for a real symmetric
    or complex hermitian square matrix A. Given A, an orthogonal (A real) or
    unitary (A complex) matrix Q is calculated which diagonalizes A:

        Q' A Q = diag(E)               and                Q Q' = Q' Q = 1

    Here diag(E) a is diagonal matrix whose diagonal is E.
    ' denotes the hermitian transpose (i.e. ordinary transposition and
    complex conjugation).

    The columns of Q are the eigenvectors of A and E contains the eigenvalues:

        A Q[:,i] = E[i] Q[:,i]

    input:

      A: a real or complex square matrix of format (n,n) which is symmetric
         (i.e. A[i,j]=A[j,i]) or hermitian (i.e. A[i,j]=conj(A[j,i])).

      eigvals_only: if true, calculates only the eigenvalues E.
                    if false, calculates both eigenvectors and eigenvalues.

      overwrite_a: if true, allows modification of A which may improve
                   performance. if false, A is not modified.

    output:

      E: vector of format (n). contains the eigenvalues of A in ascending order.

      Q: an orthogonal or unitary matrix of format (n,n). contains the
         eigenvectors of A as columns.

    return value:

          E         if eigvals_only is true
         (E, Q)     if eigvals_only is false

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, 2], [2, 0]])
      >>> E = mp.eigh(A, eigvals_only = True)
      >>> print(E)
      [-1.0]
      [ 4.0]

      >>> A = mp.matrix([[1, 2], [2, 3]])
      >>> E, Q = mp.eigh(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

      >>> A = mp.matrix([[1, 2 + 5j], [2 - 5j, 3]])
      >>> E, Q = mp.eigh(A)
      >>> print(mp.chop(A * Q[:,0] - E[0] * Q[:,0]))
      [0.0]
      [0.0]

    see also: eigsy, eighe, eig
    """

    iscomplex = any(type(x) is ctx.mpc for x in A)

    if iscomplex:
        return ctx.eighe(A, eigvals_only = eigvals_only, overwrite_a = overwrite_a)
    else:
        return ctx.eigsy(A, eigvals_only = eigvals_only, overwrite_a = overwrite_a)


@defun
def gauss_quadrature(ctx, n, qtype = "legendre", alpha = 0, beta = 0):
    """
    This routine calulates gaussian quadrature rules for different
    families of orthogonal polynomials. Let (a, b) be an interval,
    W(x) a positive weight function and n a positive integer.
    Then the purpose of this routine is to calculate pairs (x_k, w_k)
    for k=0, 1, 2, ... (n-1) which give

      int(W(x) * F(x), x = a..b) = sum(w_k * F(x_k),k = 0..(n-1))

    exact for all polynomials F(x) of degree (strictly) less than 2*n. For all
    integrable functions F(x) the sum is a (more or less) good approximation to
    the integral. The x_k are called nodes (which are the zeros of the
    related orthogonal polynomials) and the w_k are called the weights.

    parameters
       n        (input) The degree of the quadrature rule, i.e. its number of
                nodes.

       qtype    (input) The family of orthogonal polynmomials for which to
                compute the quadrature rule. See the list below.

       alpha    (input) real number, used as parameter for some orthogonal
                polynomials

       beta     (input) real number, used as parameter for some orthogonal
                polynomials.

    return value

      (X, W)    a pair of two real arrays where x_k = X[k] and w_k = W[k].


    orthogonal polynomials:

      qtype           polynomial
      -----           ----------

      "legendre"      Legendre polynomials, W(x)=1 on the interval (-1, +1)
      "legendre01"    shifted Legendre polynomials, W(x)=1 on the interval (0, +1)
      "hermite"       Hermite polynomials, W(x)=exp(-x*x) on (-infinity,+infinity)
      "laguerre"      Laguerre polynomials, W(x)=exp(-x) on (0,+infinity)
      "glaguerre"     generalized Laguerre polynomials, W(x)=exp(-x)*x**alpha
                      on (0, +infinity)
      "chebyshev1"    Chebyshev polynomials of the first kind, W(x)=1/sqrt(1-x*x)
                      on (-1, +1)
      "chebyshev2"    Chebyshev polynomials of the second kind, W(x)=sqrt(1-x*x)
                      on (-1, +1)
      "jacobi"        Jacobi polynomials, W(x)=(1-x)**alpha * (1+x)**beta on (-1, +1)
                      with alpha>-1 and beta>-1

    examples:
      >>> from mpmath import mp
      >>> f = lambda x: x**8 + 2 * x**6 - 3 * x**4 + 5 * x**2 - 7
      >>> X, W = mp.gauss_quadrature(5, "hermite")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> B = mp.sqrt(mp.pi) * 57 / 16
      >>> C = mp.quad(lambda x: mp.exp(- x * x) * f(x), [-mp.inf, +mp.inf])
      >>> mp.nprint((mp.chop(A-B, tol = 1e-10), mp.chop(A-C, tol = 1e-10)))
      (0.0, 0.0)

      >>> f = lambda x: x**5 - 2 * x**4 + 3 * x**3 - 5 * x**2 + 7 * x - 11
      >>> X, W = mp.gauss_quadrature(3, "laguerre")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> B = 76
      >>> C = mp.quad(lambda x: mp.exp(-x) * f(x), [0, +mp.inf])
      >>> mp.nprint(mp.chop(A-B, tol = 1e-10), mp.chop(A-C, tol = 1e-10))
      .0

      # orthogonality of the chebyshev polynomials:
      >>> f = lambda x: mp.chebyt(3, x) * mp.chebyt(2, x)
      >>> X, W = mp.gauss_quadrature(3, "chebyshev1")
      >>> A = mp.fdot([(f(x), w) for x, w in zip(X, W)])
      >>> print(mp.chop(A, tol = 1e-10))
      0.0

    references:
      - golub and welsch, "calculations of gaussian quadrature rules", mathematics of
        computation 23, p. 221-230 (1969)
      - golub, "some modified matrix eigenvalue problems", siam review 15, p. 318-334 (1973)
      - stroud and secrest, "gaussian quadrature formulas", prentice-hall (1966)

    See also the routine gaussq.f in netlog.org or ACM Transactions on
    Mathematical Software algorithm 726.
    """

    d = ctx.zeros(n, 1)
    e = ctx.zeros(n, 1)
    z = ctx.zeros(1, n)

    z[0,0] = 1

    if qtype == "legendre":
        # legendre on the range -1 +1 , abramowitz, table 25.4, p.916
        w = 2
        for i in xrange(n):
            j = i + 1
            e[i] = ctx.sqrt(j * j / (4 * j * j - ctx.mpf(1)))
    elif qtype == "legendre01":
        # legendre shifted to 0 1        , abramowitz, table 25.8, p.921
        w = 1
        for i in xrange(n):
            d[i] = 1 / ctx.mpf(2)
            j = i + 1
            e[i] = ctx.sqrt(j * j / (16 * j * j - ctx.mpf(4)))
    elif qtype == "hermite":
        # hermite on the range -inf +inf , abramowitz, table 25.10,p.924
        w = ctx.sqrt(ctx.pi)
        for i in xrange(n):
            j = i + 1
            e[i] = ctx.sqrt(j / ctx.mpf(2))
    elif qtype == "laguerre":
        # laguerre on the range 0 +inf , abramowitz, table 25.9, p. 923
        w = 1
        for i in xrange(n):
            j = i + 1
            d[i] = 2 * j - 1
            e[i] = j
    elif qtype=="chebyshev1":
        # chebyshev polynimials of the first kind
        w = ctx.pi
        for i in xrange(n):
            e[i] = 1 / ctx.mpf(2)
        e[0] = ctx.sqrt(1 / ctx.mpf(2))
    elif qtype == "chebyshev2":
        # chebyshev polynimials of the second kind
        w = ctx.pi / 2
        for i in xrange(n):
            e[i] = 1 / ctx.mpf(2)
    elif qtype == "glaguerre":
        # generalized laguerre on the range 0 +inf
        w = ctx.gamma(1 + alpha)
        for i in xrange(n):
            j = i + 1
            d[i] = 2 * j - 1 + alpha
            e[i] = ctx.sqrt(j * (j + alpha))
    elif qtype == "jacobi":
        # jacobi polynomials
        alpha = ctx.mpf(alpha)
        beta = ctx.mpf(beta)
        ab = alpha + beta
        abi = ab + 2
        w = (2**(ab+1)) * ctx.gamma(alpha + 1) * ctx.gamma(beta + 1) / ctx.gamma(abi)
        d[0] = (beta - alpha) / abi
        e[0] = ctx.sqrt(4 * (1 + alpha) * (1 + beta) / ((abi + 1) * (abi * abi)))
        a2b2 = beta * beta - alpha * alpha
        for i in xrange(1, n):
            j = i + 1
            abi = 2 * j + ab
            d[i] = a2b2 / ((abi - 2) * abi)
            e[i] = ctx.sqrt(4 * j * (j + alpha) * (j + beta) * (j + ab) / ((abi * abi - 1) * abi * abi))
    elif isinstance(qtype, str):
        raise ValueError("unknown quadrature rule \"%s\"" % qtype)
    elif not isinstance(qtype, str):
        w = qtype(d, e)
    else:
        assert 0

    tridiag_eigen(ctx, d, e, z)

    for i in xrange(len(z)):
        z[i] *= z[i]

    z = z.transpose()
    return (d, w * z)

##################################################################################################
##################################################################################################
##################################################################################################

def svd_r_raw(ctx, A, V = False, calc_u = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two orthogonal matrices U and V are calculated such that

                    A = U S V

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    The diagonal elements of S are the singular values of A, i.e. the
    squareroots of the eigenvalues of A' A or A A'. Here ' denotes the transpose.
    Householder bidiagonalization and a variant of the QR algorithm is used.

    overview of the matrices :

      A : m*n       A gets replaced by U
      U : m*n       U replaces A. If n>m then only the first m*m block of U is
                    non-zero. column-orthogonal: U' U = B
                    here B is a n*n matrix whose first min(m,n) diagonal
                    elements are 1 and all other elements are zero.
      S : n*n       diagonal matrix, only the diagonal elements are stored in
                    the array S. only the first min(m,n) diagonal elements are non-zero.
      V : n*n       orthogonal: V V' = V' V = 1

    parameters:
      A        (input/output) On input, A contains a real matrix of shape m*n.
               On output, if calc_u is true A contains the column-orthogonal
               matrix U; otherwise A is simply used as workspace and thus destroyed.

      V        (input/output) if false, the matrix V is not calculated. otherwise
               V must be a matrix of shape n*n.

      calc_u   (input) If true, the matrix U is calculated and replaces A.
               if false, U is not calculated and A is simply destroyed

    return value:
      S        an array of length n containing the singular values of A sorted by
               decreasing magnitude. only the first min(m,n) elements are non-zero.

    This routine is a python translation of the fortran routine svd.f in the
    software library EISPACK (see netlib.org) which itself is based on the
    algol procedure svd described in:
      - num. math. 14, 403-420(1970) by golub and reinsch.
      - wilkinson/reinsch: handbook for auto. comp., vol ii-linear algebra, 134-151(1971).

    """

    m, n = A.rows, A.cols

    S = ctx.zeros(n, 1)

    # work is a temporary array of size n
    work = ctx.zeros(n, 1)

    g = scale = anorm = 0
    maxits = 3 * ctx.dps

    for i in xrange(n):     # householder reduction to bidiagonal form
        work[i] = scale*g
        g = s = scale = 0
        if i < m:
            for k in xrange(i, m):
                scale += ctx.fabs(A[k,i])
            if scale != 0:
                for k in xrange(i, m):
                    A[k,i] /= scale
                    s += A[k,i] * A[k,i]
                f = A[i,i]
                g = -ctx.sqrt(s)
                if f < 0:
                    g = -g
                h = f * g - s
                A[i,i] = f - g
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i, m):
                        s += A[k,i] * A[k,j]
                    f = s / h
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for k in xrange(i,m):
                    A[k,i] *= scale

        S[i] = scale * g
        g = s = scale = 0

        if i < m and i != n - 1:
            for k in xrange(i+1, n):
                scale += ctx.fabs(A[i,k])
            if scale:
                for k in xrange(i+1, n):
                    A[i,k] /= scale
                    s += A[i,k] * A[i,k]
                f = A[i,i+1]
                g = -ctx.sqrt(s)
                if f < 0:
                    g = -g
                h = f * g - s
                A[i,i+1] = f - g

                for k in xrange(i+1, n):
                    work[k] = A[i,k] / h

                for j in xrange(i+1, m):
                    s = 0
                    for k in xrange(i+1, n):
                        s += A[j,k] * A[i,k]
                    for k in xrange(i+1, n):
                        A[j,k] += s * work[k]

                for k in xrange(i+1, n):
                    A[i,k] *= scale

        anorm = max(anorm, ctx.fabs(S[i]) + ctx.fabs(work[i]))

    if not isinstance(V, bool):
        for i in xrange(n-2, -1, -1):     # accumulation of right hand transformations
            V[i+1,i+1] = 1

            if work[i+1] != 0:
                for j in xrange(i+1, n):
                    V[i,j] = (A[i,j] / A[i,i+1]) / work[i+1]
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, n):
                        s += A[i,k] * V[j,k]
                    for k in xrange(i+1, n):
                        V[j,k] += s * V[i,k]

            for j in xrange(i+1, n):
                V[j,i] = V[i,j] = 0

        V[0,0] = 1

    if m<n : minnm = m
    else   : minnm = n

    if calc_u:
        for i in xrange(minnm-1, -1, -1): # accumulation of left hand transformations
            g = S[i]
            for j in xrange(i+1, n):
                A[i,j] = 0
            if g != 0:
                g = 1 / g
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, m):
                        s += A[k,i] * A[k,j]
                    f = (s / A[i,i]) * g
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for j in xrange(i, m):
                    A[j,i] *= g
            else:
                for j in xrange(i, m):
                    A[j,i] = 0
            A[i,i] += 1

    for k in xrange(n - 1, -1, -1):
        # diagonalization of the bidiagonal form:
        #   loop over singular values, and over allowed itations

        its = 0
        while 1:
            its += 1
            flag = True

            for l in xrange(k, -1, -1):
                nm = l-1

                if ctx.fabs(work[l]) + anorm == anorm:
                    flag = False
                    break

                if ctx.fabs(S[nm]) + anorm == anorm:
                    break

            if flag:
                c = 0
                s = 1
                for i in xrange(l, k + 1):
                    f = s * work[i]
                    work[i] *= c
                    if ctx.fabs(f) + anorm == anorm:
                        break
                    g = S[i]
                    h = ctx.hypot(f, g)
                    S[i] = h
                    h = 1 / h
                    c = g * h
                    s = - f * h

                    if calc_u:
                        for j in xrange(m):
                            y = A[j,nm]
                            z = A[j,i]
                            A[j,nm] = y * c + z * s
                            A[j,i]  = z * c - y * s

            z = S[k]

            if l == k:               # convergence
                if z < 0:            # singular value is made nonnegative
                    S[k] = -z
                    if not isinstance(V, bool):
                        for j in xrange(n):
                            V[k,j] = -V[k,j]
                break

            if its >= maxits:
                raise RuntimeError("svd: no convergence to an eigenvalue after %d iterations" % its)

            x = S[l]         # shift from bottom 2 by 2 minor
            nm = k-1
            y = S[nm]
            g = work[nm]
            h = work[k]
            f = ((y - z) * (y + z) + (g - h) * (g + h))/(2 * h * y)
            g = ctx.hypot(f, 1)
            if f >= 0: f = ((x - z) * (x + z) + h * ((y / (f + g)) - h)) / x
            else:      f = ((x - z) * (x + z) + h * ((y / (f - g)) - h)) / x

            c = s = 1         # next qt transformation

            for j in xrange(l, nm + 1):
                g = work[j+1]
                y = S[j+1]
                h = s * g
                g = c * g
                z = ctx.hypot(f, h)
                work[j] = z
                c = f / z
                s = h / z
                f = x * c + g * s
                g = g * c - x * s
                h = y * s
                y *= c
                if not isinstance(V, bool):
                    for jj in xrange(n):
                        x = V[j  ,jj]
                        z = V[j+1,jj]
                        V[j    ,jj]= x * c + z * s
                        V[j+1  ,jj]= z * c - x * s
                z = ctx.hypot(f, h)
                S[j] = z
                if z != 0:            # rotation can be arbitray if z=0
                    z = 1 / z
                    c = f * z
                    s = h * z
                f = c * g + s * y
                x = c * y - s * g

                if calc_u:
                    for jj in xrange(m):
                        y = A[jj,j  ]
                        z = A[jj,j+1]
                        A[jj,j    ] = y * c + z * s
                        A[jj,j+1  ] = z * c - y * s

            work[l] = 0
            work[k] = f
            S[k] = x

    ##########################

    # Sort singular values into decreasing order (bubble-sort)

    for i in xrange(n):
        imax = i
        s = ctx.fabs(S[i])         # s is the current maximal element

        for j in xrange(i + 1, n):
            c = ctx.fabs(S[j])
            if c > s:
                s = c
                imax = j

        if imax != i:
            # swap singular values

            z = S[i]
            S[i] = S[imax]
            S[imax] = z

            if calc_u:
                for j in xrange(m):
                    z = A[j,i]
                    A[j,i] = A[j,imax]
                    A[j,imax] = z

            if not isinstance(V, bool):
                for j in xrange(n):
                    z = V[i,j]
                    V[i,j] = V[imax,j]
                    V[imax,j] = z

    return S

#######################

def svd_c_raw(ctx, A, V = False, calc_u = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two unitary matrices U and V are calculated such that

                    A = U S V

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    The diagonal elements of S are the singular values of A, i.e. the
    squareroots of the eigenvalues of A' A or A A'. Here ' denotes the hermitian
    transpose (i.e. transposition and conjugation). Householder bidiagonalization
    and a variant of the QR algorithm is used.

    overview of the matrices :

      A : m*n       A gets replaced by U
      U : m*n       U replaces A. If n>m then only the first m*m block of U is
                    non-zero. column-unitary: U' U = B
                    here B is a n*n matrix whose first min(m,n) diagonal
                    elements are 1 and all other elements are zero.
      S : n*n       diagonal matrix, only the diagonal elements are stored in
                    the array S. only the first min(m,n) diagonal elements are non-zero.
      V : n*n       unitary: V V' = V' V = 1

    parameters:
      A        (input/output) On input, A contains a complex matrix of shape m*n.
               On output, if calc_u is true A contains the column-unitary
               matrix U; otherwise A is simply used as workspace and thus destroyed.

      V        (input/output) if false, the matrix V is not calculated. otherwise
               V must be a matrix of shape n*n.

      calc_u   (input) If true, the matrix U is calculated and replaces A.
               if false, U is not calculated and A is simply destroyed

    return value:
      S        an array of length n containing the singular values of A sorted by
               decreasing magnitude. only the first min(m,n) elements are non-zero.

    This routine is a python translation of the fortran routine svd.f in the
    software library EISPACK (see netlib.org) which itself is based on the
    algol procedure svd described in:
      - num. math. 14, 403-420(1970) by golub and reinsch.
      - wilkinson/reinsch: handbook for auto. comp., vol ii-linear algebra, 134-151(1971).

    """

    m, n = A.rows, A.cols

    S = ctx.zeros(n, 1)

    # work is a temporary array of size n
    work  = ctx.zeros(n, 1)
    lbeta = ctx.zeros(n, 1)
    rbeta = ctx.zeros(n, 1)
    dwork = ctx.zeros(n, 1)

    g = scale = anorm = 0
    maxits = 3 * ctx.dps

    for i in xrange(n):         # householder reduction to bidiagonal form
        dwork[i] = scale * g    # dwork are the side-diagonal elements
        g = s = scale = 0
        if i < m:
            for k in xrange(i, m):
                scale += ctx.fabs(ctx.re(A[k,i])) + ctx.fabs(ctx.im(A[k,i]))
            if scale != 0:
                for k in xrange(i, m):
                    A[k,i] /= scale
                    ar = ctx.re(A[k,i])
                    ai = ctx.im(A[k,i])
                    s += ar * ar + ai * ai
                f = A[i,i]
                g = -ctx.sqrt(s)
                if ctx.re(f) < 0:
                    beta = -g - ctx.conj(f)
                    g = -g
                else:
                    beta = -g + ctx.conj(f)
                beta /= ctx.conj(beta)
                beta += 1
                h = 2 * (ctx.re(f) * g - s)
                A[i,i] = f - g
                beta /= h
                lbeta[i] = (beta / scale) / scale
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i, m):
                        s += ctx.conj(A[k,i]) * A[k,j]
                    f = beta * s
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for k in xrange(i, m):
                    A[k,i] *= scale

        S[i] = scale * g     # S are the diagonal elements
        g = s = scale = 0

        if i < m and i != n - 1:
            for k in xrange(i+1, n):
                scale += ctx.fabs(ctx.re(A[i,k])) + ctx.fabs(ctx.im(A[i,k]))
            if scale:
                for k in xrange(i+1, n):
                    A[i,k] /= scale
                    ar = ctx.re(A[i,k])
                    ai = ctx.im(A[i,k])
                    s += ar * ar + ai * ai
                f = A[i,i+1]
                g = -ctx.sqrt(s)
                if ctx.re(f) < 0:
                    beta = -g - ctx.conj(f)
                    g = -g
                else:
                    beta = -g + ctx.conj(f)

                beta /= ctx.conj(beta)
                beta += 1

                h = 2 * (ctx.re(f) * g - s)
                A[i,i+1] = f - g

                beta /= h
                rbeta[i] = (beta / scale) / scale

                for k in xrange(i+1, n):
                    work[k] = A[i, k]

                for j in xrange(i+1, m):
                    s = 0
                    for k in xrange(i+1, n):
                        s += ctx.conj(A[i,k]) * A[j,k]
                    f = s * beta
                    for k in xrange(i+1,n):
                        A[j,k] += f * work[k]

                for k in xrange(i+1, n):
                    A[i,k] *= scale

        anorm = max(anorm,ctx.fabs(S[i]) + ctx.fabs(dwork[i]))

    if not isinstance(V, bool):
        for i in xrange(n-2, -1, -1):     # accumulation of right hand transformations
            V[i+1,i+1] = 1

            if dwork[i+1] != 0:
                f = ctx.conj(rbeta[i])
                for j in xrange(i+1, n):
                    V[i,j] = A[i,j] * f
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, n):
                        s += ctx.conj(A[i,k]) * V[j,k]
                    for k in xrange(i+1, n):
                        V[j,k] += s * V[i,k]

            for j in xrange(i+1,n):
                V[j,i] = V[i,j] = 0

        V[0,0] = 1

    if m < n : minnm = m
    else     : minnm = n

    if calc_u:
        for i in xrange(minnm-1, -1, -1): # accumulation of left hand transformations
            g = S[i]
            for j in xrange(i+1, n):
                A[i,j] = 0
            if g != 0:
                g = 1 / g
                for j in xrange(i+1, n):
                    s = 0
                    for k in xrange(i+1, m):
                        s += ctx.conj(A[k,i]) * A[k,j]
                    f = s * ctx.conj(lbeta[i])
                    for k in xrange(i, m):
                        A[k,j] += f * A[k,i]
                for j in xrange(i, m):
                    A[j,i] *= g
            else:
                for j in xrange(i, m):
                    A[j,i] = 0
            A[i,i] += 1

    for k in xrange(n-1, -1, -1):
        # diagonalization of the bidiagonal form:
        #   loop over singular values, and over allowed itations

        its = 0
        while 1:
            its += 1
            flag = True

            for l in xrange(k, -1, -1):
                nm = l - 1

                if ctx.fabs(dwork[l]) + anorm == anorm:
                    flag = False
                    break

                if ctx.fabs(S[nm]) + anorm == anorm:
                    break

            if flag:
                c = 0
                s = 1
                for i in xrange(l, k+1):
                    f = s * dwork[i]
                    dwork[i] *= c
                    if ctx.fabs(f) + anorm == anorm:
                        break
                    g = S[i]
                    h = ctx.hypot(f, g)
                    S[i] = h
                    h = 1 / h
                    c = g * h
                    s = -f * h

                    if calc_u:
                        for j in xrange(m):
                            y = A[j,nm]
                            z = A[j,i]
                            A[j,nm]= y * c + z * s
                            A[j,i] = z * c - y * s

            z = S[k]

            if l == k:         # convergence
                if z < 0:    # singular value is made nonnegative
                    S[k] = -z
                    if not isinstance(V, bool):
                        for j in xrange(n):
                            V[k,j] = -V[k,j]
                break

            if its >= maxits:
                raise RuntimeError("svd: no convergence to an eigenvalue after %d iterations" % its)

            x = S[l]         # shift from bottom 2 by 2 minor
            nm = k-1
            y = S[nm]
            g = dwork[nm]
            h = dwork[k]
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
            g = ctx.hypot(f, 1)
            if f >=0: f = (( x - z) *( x + z) + h *((y / (f + g)) - h)) / x
            else:     f = (( x - z) *( x + z) + h *((y / (f - g)) - h)) / x

            c = s = 1         # next qt transformation

            for j in xrange(l, nm + 1):
                g = dwork[j+1]
                y = S[j+1]
                h = s * g
                g = c * g
                z = ctx.hypot(f, h)
                dwork[j] = z
                c = f / z
                s = h / z
                f = x * c + g * s
                g = g * c - x * s
                h = y * s
                y *= c
                if not isinstance(V, bool):
                    for jj in xrange(n):
                        x = V[j  ,jj]
                        z = V[j+1,jj]
                        V[j    ,jj]= x * c + z * s
                        V[j+1,jj  ]= z * c - x * s
                z = ctx.hypot(f, h)
                S[j] = z
                if z != 0:            # rotation can be arbitray if z=0
                    z = 1 / z
                    c = f * z
                    s = h * z
                f = c * g + s * y
                x = c * y - s * g
                if calc_u:
                    for jj in xrange(m):
                        y = A[jj,j  ]
                        z = A[jj,j+1]
                        A[jj,j    ]= y * c + z * s
                        A[jj,j+1  ]= z * c - y * s

            dwork[l] = 0
            dwork[k] = f
            S[k] = x

    ##########################

    # Sort singular values into decreasing order (bubble-sort)

    for i in xrange(n):
        imax = i
        s = ctx.fabs(S[i])         # s is the current maximal element

        for j in xrange(i + 1, n):
            c = ctx.fabs(S[j])
            if c > s:
                s = c
                imax = j

        if imax != i:
            # swap singular values

            z = S[i]
            S[i] = S[imax]
            S[imax] = z

            if calc_u:
                for j in xrange(m):
                    z = A[j,i]
                    A[j,i] = A[j,imax]
                    A[j,imax] = z

            if not isinstance(V, bool):
                for j in xrange(n):
                    z = V[i,j]
                    V[i,j] = V[imax,j]
                    V[imax,j] = z

    return S

##################################################################################################

@defun
def svd_r(ctx, A, full_matrices = False, compute_uv = True, overwrite_a = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two orthogonal matrices U and V are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the transpose. The diagonal elements of S are the singular
    values of A, i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a real matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an orthogonal matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an orthogonal matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    examples:

       >>> from mpmath import mp
       >>> A = mp.matrix([[2, -2, -1], [3, 4, -2], [-2, -2, 0]])
       >>> S = mp.svd_r(A, compute_uv = False)
       >>> print(S)
       [6.0]
       [3.0]
       [1.0]

       >>> U, S, V = mp.svd_r(A)
       >>> print(mp.chop(A - U * mp.diag(S) * V))
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]


    see also: svd, svd_c
    """

    m, n = A.rows, A.cols

    if not compute_uv:
        if not overwrite_a:
            A = A.copy()
        S = svd_r_raw(ctx, A, V = False, calc_u = False)
        S = S[:min(m,n)]
        return S

    if full_matrices and n < m:
        V = ctx.zeros(m, m)
        A0 = ctx.zeros(m, m)
        A0[:,:n] = A
        S = svd_r_raw(ctx, A0, V, calc_u = True)

        S = S[:n]
        V = V[:n,:n]

        return (A0, S, V)
    else:
        if not overwrite_a:
            A = A.copy()
        V = ctx.zeros(n, n)
        S = svd_r_raw(ctx, A, V, calc_u = True)

        if n > m:
            if full_matrices == False:
                V = V[:m,:]

            S = S[:m]
            A = A[:,:m]

        return (A, S, V)

##############################

@defun
def svd_c(ctx, A, full_matrices = False, compute_uv = True, overwrite_a = False):
    """
    This routine computes the singular value decomposition of a matrix A.
    Given A, two unitary matrices U and V are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the hermitian transpose (i.e. transposition and complex
    conjugation). The diagonal elements of S are the singular values of A,
    i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a complex matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an unitary matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an unitary matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[-2j, -1-3j, -2+2j], [2-2j, -1-3j, 1], [-3+1j,-2j,0]])
      >>> S = mp.svd_c(A, compute_uv = False)
      >>> print(mp.chop(S - mp.matrix([mp.sqrt(34), mp.sqrt(15), mp.sqrt(6)])))
      [0.0]
      [0.0]
      [0.0]

      >>> U, S, V = mp.svd_c(A)
      >>> print(mp.chop(A - U * mp.diag(S) * V))
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]
      [0.0  0.0  0.0]

    see also: svd, svd_r
    """

    m, n = A.rows, A.cols

    if not compute_uv:
        if not overwrite_a:
            A = A.copy()
        S = svd_c_raw(ctx, A, V = False, calc_u = False)
        S = S[:min(m,n)]
        return S

    if full_matrices and n < m:
        V = ctx.zeros(m, m)
        A0 = ctx.zeros(m, m)
        A0[:,:n] = A
        S = svd_c_raw(ctx, A0, V, calc_u = True)

        S = S[:n]
        V = V[:n,:n]

        return (A0, S, V)
    else:
        if not overwrite_a:
            A = A.copy()
        V = ctx.zeros(n, n)
        S = svd_c_raw(ctx, A, V, calc_u = True)

        if n > m:
            if full_matrices == False:
                V = V[:m,:]

            S = S[:m]
            A = A[:,:m]

        return (A, S, V)

@defun
def svd(ctx, A, full_matrices = False, compute_uv = True, overwrite_a = False):
    """
    "svd" is a unified interface for "svd_r" and "svd_c". Depending on
    whether A is real or complex the appropriate function is called.

    This routine computes the singular value decomposition of a matrix A.
    Given A, two orthogonal (A real) or unitary (A complex) matrices U and V
    are calculated such that

           A = U S V        and        U' U = 1         and         V V' = 1

    where S is a suitable shaped matrix whose off-diagonal elements are zero.
    Here ' denotes the hermitian transpose (i.e. transposition and complex
    conjugation). The diagonal elements of S are the singular values of A,
    i.e. the squareroots of the eigenvalues of A' A or A A'.

    input:
      A             : a real or complex matrix of shape (m, n)
      full_matrices : if true, U and V are of shape (m, m) and (n, n).
                      if false, U and V are of shape (m, min(m, n)) and (min(m, n), n).
      compute_uv    : if true, U and V are calculated. if false, only S is calculated.
      overwrite_a   : if true, allows modification of A which may improve
                      performance. if false, A is not modified.

    output:
      U : an orthogonal or unitary matrix: U' U = 1. if full_matrices is true, U is of
          shape (m, m). ortherwise it is of shape (m, min(m, n)).

      S : an array of length min(m, n) containing the singular values of A sorted by
          decreasing magnitude.

      V : an orthogonal or unitary matrix: V V' = 1. if full_matrices is true, V is of
          shape (n, n). ortherwise it is of shape (min(m, n), n).

    return value:

           S          if compute_uv is false
       (U, S, V)      if compute_uv is true

    overview of the matrices:

      full_matrices true:
        A           : m*n
        U           : m*m     U' U  = 1
        S as matrix : m*n
        V           : n*n     V  V' = 1

     full_matrices false:
        A           : m*n
        U           : m*min(n,m)             U' U  = 1
        S as matrix : min(m,n)*min(m,n)
        V           : min(m,n)*n             V  V' = 1

    examples:

       >>> from mpmath import mp
       >>> A = mp.matrix([[2, -2, -1], [3, 4, -2], [-2, -2, 0]])
       >>> S = mp.svd(A, compute_uv = False)
       >>> print(S)
       [6.0]
       [3.0]
       [1.0]

       >>> U, S, V = mp.svd(A)
       >>> print(mp.chop(A - U * mp.diag(S) * V))
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]
       [0.0  0.0  0.0]

    see also: svd_r, svd_c
    """

    iscomplex = any(type(x) is ctx.mpc for x in A)

    if iscomplex:
        return ctx.svd_c(A, full_matrices = full_matrices, compute_uv = compute_uv, overwrite_a = overwrite_a)
    else:
        return ctx.svd_r(A, full_matrices = full_matrices, compute_uv = compute_uv, overwrite_a = overwrite_a)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio_websocket\_impl.py ===
from __future__ import annotations

import sys
from collections import OrderedDict
from contextlib import asynccontextmanager, AbstractAsyncContextManager
from functools import partial
from ipaddress import ip_address
import itertools
import logging
import random
import ssl
import struct
import urllib.parse
from typing import Any, List, NoReturn, Optional, Union, TypeVar, TYPE_CHECKING, Generic, cast
from importlib.metadata import version

import outcome
import trio
import trio.abc
from wsproto import ConnectionType, WSConnection
from wsproto.connection import ConnectionState
import wsproto.frame_protocol as wsframeproto
from wsproto.events import (
    AcceptConnection,
    BytesMessage,
    CloseConnection,
    Ping,
    Pong,
    RejectConnection,
    RejectData,
    Request,
    TextMessage,
)
import wsproto.utilities

if sys.version_info < (3, 11):  # pragma: no cover
    # pylint doesn't care about the version_info check, so need to ignore the warning
    from exceptiongroup import BaseExceptionGroup  # pylint: disable=redefined-builtin

if TYPE_CHECKING:
    from types import TracebackType
    from typing_extensions import Final
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Coroutine, Sequence

_IS_TRIO_MULTI_ERROR: Final = tuple(map(int, version("trio").split(".")[:2])) < (0, 22)

if _IS_TRIO_MULTI_ERROR:
    _TRIO_EXC_GROUP_TYPE = trio.MultiError  # type: ignore[attr-defined] # pylint: disable=no-member
else:
    _TRIO_EXC_GROUP_TYPE = BaseExceptionGroup  # pylint: disable=possibly-used-before-assignment

CONN_TIMEOUT: Final = 60 # default connect & disconnect timeout, in seconds
MESSAGE_QUEUE_SIZE: Final = 1
MAX_MESSAGE_SIZE: Final = 2 ** 20 # 1 MiB
RECEIVE_BYTES: Final = 4 * 2 ** 10 # 4 KiB
logger: Final = logging.getLogger('trio-websocket')

T = TypeVar("T")
E = TypeVar("E", bound=BaseException)


class TrioWebsocketInternalError(Exception):
    """Raised as a fallback when open_websocket is unable to unwind an exceptiongroup
    into a single preferred exception. This should never happen, if it does then
    underlying assumptions about the internal code are incorrect.
    """


def _ignore_cancel(exc: E) -> E | None:
    return None if isinstance(exc, trio.Cancelled) else exc


class _preserve_current_exception:
    """A context manager which should surround an ``__exit__`` or
    ``__aexit__`` handler or the contents of a ``finally:``
    block. It ensures that any exception that was being handled
    upon entry is not masked by a `trio.Cancelled` raised within
    the body of the context manager.

    https://github.com/python-trio/trio/issues/1559
    https://gitter.im/python-trio/general?at=5faf2293d37a1a13d6a582cf
    """
    __slots__ = ("_armed",)

    def __init__(self) -> None:
        self._armed = False

    def __enter__(self) -> None:
        self._armed = sys.exc_info()[1] is not None

    def __exit__(
        self,
        ty: type[BaseException] | None,
        value: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if value is None or not self._armed:
            return False

        if _IS_TRIO_MULTI_ERROR:  # pragma: no cover
            filtered_exception = trio.MultiError.filter(_ignore_cancel, value)  # type: ignore[attr-defined]  # pylint: disable=no-member
        elif isinstance(value, BaseExceptionGroup):  # pylint: disable=possibly-used-before-assignment
            filtered_exception = value.subgroup(lambda exc: not isinstance(exc, trio.Cancelled))
        else:
            filtered_exception = _ignore_cancel(value)
        return filtered_exception is None


@asynccontextmanager
async def open_websocket(
    host: str,
    port: int,
    resource: str,
    *,
    use_ssl: Union[bool, ssl.SSLContext],
    subprotocols: Optional[Iterable[str]] = None,
    extra_headers: Optional[list[tuple[bytes,bytes]]] = None,
    message_queue_size: int = MESSAGE_QUEUE_SIZE,
    max_message_size: int = MAX_MESSAGE_SIZE,
    receive_buffer_size: Union[None, int] = RECEIVE_BYTES,
    connect_timeout: float = CONN_TIMEOUT,
    disconnect_timeout: float = CONN_TIMEOUT
) -> AsyncGenerator[WebSocketConnection, None]:
    '''
    Open a WebSocket client connection to a host.

    This async context manager connects when entering the context manager and
    disconnects when exiting. It yields a
    :class:`WebSocketConnection` instance.

    :param str host: The host to connect to.
    :param int port: The port to connect to.
    :param str resource: The resource, i.e. URL path.
    :param Union[bool, ssl.SSLContext] use_ssl: If this is an SSL context, then
        use that context. If this is ``True`` then use default SSL context. If
        this is ``False`` then disable SSL.
    :param subprotocols: An iterable of strings representing preferred
        subprotocols.
    :param list[tuple[bytes,bytes]] extra_headers: A list of 2-tuples containing
        HTTP header key/value pairs to send with the connection request. Note
        that headers used by the WebSocket protocol (e.g.
        ``Sec-WebSocket-Accept``) will be overwritten.
    :param int message_queue_size: The maximum number of messages that will be
        buffered in the library's internal message queue.
    :param int max_message_size: The maximum message size as measured by
        ``len()``. If a message is received that is larger than this size,
        then the connection is closed with code 1009 (Message Too Big).
    :param Optional[int] receive_buffer_size: The buffer size we use to
        receive messages internally. None to let trio choose. Defaults
        to 4 KiB.
    :param float connect_timeout: The number of seconds to wait for the
        connection before timing out.
    :param float disconnect_timeout: The number of seconds to wait when closing
        the connection before timing out.
    :raises HandshakeError: for any networking error,
        client-side timeout (:exc:`ConnectionTimeout`, :exc:`DisconnectionTimeout`),
        or server rejection (:exc:`ConnectionRejected`) during handshakes.
    '''

    # This context manager tries very very hard not to raise an exceptiongroup
    # in order to be as transparent as possible for the end user.
    # In the trivial case, this means that if user code inside the cm raises
    # we make sure that it doesn't get wrapped.

    # If opening the connection fails, then we will raise that exception. User
    # code is never executed, so we will never have multiple exceptions.

    # After opening the connection, we spawn _reader_task in the background and
    # yield to user code. If only one of those raise a non-cancelled exception
    # we will raise that non-cancelled exception.
    # If we get multiple cancelled, we raise the user's cancelled.
    # If both raise exceptions, we raise the user code's exception with __context__
    # set to a group containing internal exception(s) + any user exception __context__
    # If we somehow get multiple exceptions, but no user exception, then we raise
    # TrioWebsocketInternalError.

    # If closing the connection fails, then that will be raised as the top
    # exception in the last `finally`. If we encountered exceptions in user code
    # or in reader task then they will be set as the `__context__`.


    async def _open_connection(nursery: trio.Nursery) -> WebSocketConnection:
        try:
            with trio.fail_after(connect_timeout):
                return await connect_websocket(nursery, host, port,
                    resource, use_ssl=use_ssl, subprotocols=subprotocols,
                    extra_headers=extra_headers,
                    message_queue_size=message_queue_size,
                    max_message_size=max_message_size,
                    receive_buffer_size=receive_buffer_size)
        except trio.TooSlowError:
            raise ConnectionTimeout from None
        except OSError as e:
            raise HandshakeError from e

    async def _close_connection(connection: WebSocketConnection) -> None:
        try:
            with trio.fail_after(disconnect_timeout):
                await connection.aclose()
        except trio.TooSlowError:
            raise DisconnectionTimeout from None

    def _raise(exc: BaseException) -> NoReturn:
        """This helper allows re-raising an exception without __context__ being set."""
        # cause does not need special handlng, we simply avoid using `raise .. from ..`
        __tracebackhide__ = True
        context = exc.__context__
        try:
            raise exc
        finally:
            exc.__context__ = context
            del exc, context

    connection: WebSocketConnection|None=None
    close_result: outcome.Maybe[None] | None = None
    user_error = None

    # Unwrapping exception groups has a lot of pitfalls, one of them stemming from
    # the exception we raise also being inside the group that's set as the context.
    # This leads to loss of info unless properly handled.
    # See https://github.com/python-trio/flake8-async/issues/298
    # We therefore avoid having the exceptiongroup included as either cause or context

    try:
        async with trio.open_nursery() as new_nursery:
            result = await outcome.acapture(_open_connection, new_nursery)

            if isinstance(result, outcome.Value):
                connection = result.unwrap()
                try:
                    yield connection
                except BaseException as e:
                    user_error = e
                    raise
                finally:
                    close_result = await outcome.acapture(_close_connection, connection)
    # This exception handler should only be entered if either:
    # 1. The _reader_task started in connect_websocket raises
    # 2. User code raises an exception
    # I.e. open/close_connection are not included
    except _TRIO_EXC_GROUP_TYPE as e:
        # user_error, or exception bubbling up from _reader_task
        if len(e.exceptions) == 1:
            _raise(e.exceptions[0])

        # contains at most 1 non-cancelled exceptions
        exception_to_raise: BaseException|None = None
        for sub_exc in e.exceptions:
            if not isinstance(sub_exc, trio.Cancelled):
                if exception_to_raise is not None:
                    # multiple non-cancelled
                    break
                exception_to_raise = sub_exc
        else:
            if exception_to_raise is None:
                # all exceptions are cancelled
                # we reraise the user exception and throw out internal
                if user_error is not None:
                    _raise(user_error)
                # multiple internal Cancelled is not possible afaik
                # but if so we just raise one of them
                _raise(e.exceptions[0])  # pragma: no cover
            # raise the non-cancelled exception
            _raise(exception_to_raise)

        # if we have any KeyboardInterrupt in the group, raise a new KeyboardInterrupt
        # with the group as cause & context
        for sub_exc in e.exceptions:
            if isinstance(sub_exc, KeyboardInterrupt):
                raise KeyboardInterrupt from e

        # Both user code and internal code raised non-cancelled exceptions.
        # We set the context to be an exception group containing internal exceptions
        # and, if not None, `user_error.__context__`
        if user_error is not None:
            exceptions = [subexc for subexc in e.exceptions if subexc is not user_error]
            eg_substr = ''
            # there's technically loss of info here, with __suppress_context__=True you
            # still have original __context__ available, just not printed. But we delete
            # it completely because we can't partially suppress the group
            if user_error.__context__ is not None and not user_error.__suppress_context__:
                exceptions.append(user_error.__context__)
                eg_substr = ' and the context for the user exception'
            eg_str = (
                "Both internal and user exceptions encountered. This group contains "
                "the internal exception(s)" + eg_substr + "."
            )
            user_error.__context__ = BaseExceptionGroup(eg_str, exceptions)
            user_error.__suppress_context__ = False
            _raise(user_error)

        raise TrioWebsocketInternalError(
            "The trio-websocket API is not expected to raise multiple exceptions. "
            "Please report this as a bug to "
            "https://github.com/python-trio/trio-websocket"
        ) from e  # pragma: no cover

    finally:
        if close_result is not None:
            close_result.unwrap()


    # error setting up, unwrap that exception
    if connection is None:
        result.unwrap()


async def connect_websocket(
    nursery: trio.Nursery,
    host: str,
    port: int,
    resource: str,
    *,
    use_ssl: bool | ssl.SSLContext,
    subprotocols: Iterable[str] | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
    message_queue_size: int = MESSAGE_QUEUE_SIZE,
    max_message_size: int = MAX_MESSAGE_SIZE,
    receive_buffer_size: Union[None, int] = RECEIVE_BYTES,
) -> WebSocketConnection:
    '''
    Return an open WebSocket client connection to a host.

    This function is used to specify a custom nursery to run connection
    background tasks in. The caller is responsible for closing the connection.

    If you don't need a custom nursery, you should probably use
    :func:`open_websocket` instead.

    :param nursery: A Trio nursery to run background tasks in.
    :param str host: The host to connect to.
    :param int port: The port to connect to.
    :param str resource: The resource, i.e. URL path.
    :param Union[bool, ssl.SSLContext] use_ssl: If this is an SSL context, then
        use that context. If this is ``True`` then use default SSL context. If
        this is ``False`` then disable SSL.
    :param subprotocols: An iterable of strings representing preferred
        subprotocols.
    :param list[tuple[bytes,bytes]] extra_headers: A list of 2-tuples containing
        HTTP header key/value pairs to send with the connection request. Note
        that headers used by the WebSocket protocol (e.g.
        ``Sec-WebSocket-Accept``) will be overwritten.
    :param int message_queue_size: The maximum number of messages that will be
        buffered in the library's internal message queue.
    :param int max_message_size: The maximum message size as measured by
        ``len()``. If a message is received that is larger than this size,
        then the connection is closed with code 1009 (Message Too Big).
    :param Optional[int] receive_buffer_size: The buffer size we use to
        receive messages internally. None to let trio choose. Defaults
        to 4 KiB.
    :rtype: WebSocketConnection
    '''
    if use_ssl is True:
        ssl_context = ssl.create_default_context()
    elif use_ssl is False:
        ssl_context = None
    elif isinstance(use_ssl, ssl.SSLContext):
        ssl_context = use_ssl
    else:
        raise TypeError('`use_ssl` argument must be bool or ssl.SSLContext')

    logger.debug('Connecting to ws%s://%s:%d%s',
        '' if ssl_context is None else 's', host, port, resource)
    stream: trio.SSLStream[trio.SocketStream] | trio.SocketStream
    if ssl_context is None:
        stream = await trio.open_tcp_stream(host, port)
    else:
        stream = await trio.open_ssl_over_tcp_stream(host, port,
            ssl_context=ssl_context, https_compatible=True)
    if port in (80, 443):
        host_header = host
    else:
        host_header = f'{host}:{port}'
    connection = WebSocketConnection(stream,
        WSConnection(ConnectionType.CLIENT),
        host=host_header,
        path=resource,
        client_subprotocols=subprotocols, client_extra_headers=extra_headers,
        message_queue_size=message_queue_size,
        max_message_size=max_message_size,
        receive_buffer_size=receive_buffer_size)
    nursery.start_soon(connection._reader_task)
    await connection._open_handshake.wait()
    return connection


def open_websocket_url(
    url: str,
    ssl_context: ssl.SSLContext | None = None,
    *,
    subprotocols: Iterable[str] | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
    message_queue_size: int = MESSAGE_QUEUE_SIZE,
    max_message_size: int = MAX_MESSAGE_SIZE,
    connect_timeout: float = CONN_TIMEOUT,
    disconnect_timeout: float = CONN_TIMEOUT,
    receive_buffer_size: Union[None, int] = RECEIVE_BYTES,
) -> AbstractAsyncContextManager[WebSocketConnection]:
    '''
    Open a WebSocket client connection to a URL.

    This async context manager connects when entering the context manager and
    disconnects when exiting. It yields a
    :class:`WebSocketConnection` instance.

    :param str url: A WebSocket URL, i.e. `ws:` or `wss:` URL scheme.
    :param ssl_context: Optional SSL context used for ``wss:`` URLs. A default
        SSL context is used for ``wss:`` if this argument is ``None``.
    :type ssl_context: ssl.SSLContext or None
    :param subprotocols: An iterable of strings representing preferred
        subprotocols.
    :param list[tuple[bytes,bytes]] extra_headers: A list of 2-tuples containing
        HTTP header key/value pairs to send with the connection request. Note
        that headers used by the WebSocket protocol (e.g.
        ``Sec-WebSocket-Accept``) will be overwritten.
    :param int message_queue_size: The maximum number of messages that will be
        buffered in the library's internal message queue.
    :param int max_message_size: The maximum message size as measured by
        ``len()``. If a message is received that is larger than this size,
        then the connection is closed with code 1009 (Message Too Big).
    :param Optional[int] receive_buffer_size: The buffer size we use to
        receive messages internally. None to let trio choose. Defaults
        to 4 KiB.
    :param float connect_timeout: The number of seconds to wait for the
        connection before timing out.
    :param float disconnect_timeout: The number of seconds to wait when closing
        the connection before timing out.
    :raises HandshakeError: for any networking error,
        client-side timeout (:exc:`ConnectionTimeout`, :exc:`DisconnectionTimeout`),
        or server rejection (:exc:`ConnectionRejected`) during handshakes.
    '''
    host, port, resource, return_ssl_context = _url_to_host(url, ssl_context)
    return open_websocket(host, port, resource, use_ssl=return_ssl_context,
        subprotocols=subprotocols, extra_headers=extra_headers,
        message_queue_size=message_queue_size,
        max_message_size=max_message_size,
        receive_buffer_size=receive_buffer_size,
        connect_timeout=connect_timeout, disconnect_timeout=disconnect_timeout)


async def connect_websocket_url(
    nursery: trio.Nursery,
    url: str,
    ssl_context: ssl.SSLContext | None = None,
    *,
    subprotocols: Iterable[str] | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
    message_queue_size: int = MESSAGE_QUEUE_SIZE,
    max_message_size: int = MAX_MESSAGE_SIZE,
    receive_buffer_size: Union[None, int] = RECEIVE_BYTES,
) -> WebSocketConnection:
    '''
    Return an open WebSocket client connection to a URL.

    This function is used to specify a custom nursery to run connection
    background tasks in. The caller is responsible for closing the connection.

    If you don't need a custom nursery, you should probably use
    :func:`open_websocket_url` instead.

    :param nursery: A nursery to run background tasks in.
    :param str url: A WebSocket URL.
    :param ssl_context: Optional SSL context used for ``wss:`` URLs.
    :type ssl_context: ssl.SSLContext or None
    :param subprotocols: An iterable of strings representing preferred
        subprotocols.
    :param list[tuple[bytes,bytes]] extra_headers: A list of 2-tuples containing
        HTTP header key/value pairs to send with the connection request. Note
        that headers used by the WebSocket protocol (e.g.
        ``Sec-WebSocket-Accept``) will be overwritten.
    :param int message_queue_size: The maximum number of messages that will be
        buffered in the library's internal message queue.
    :param int max_message_size: The maximum message size as measured by
        ``len()``. If a message is received that is larger than this size,
        then the connection is closed with code 1009 (Message Too Big).
    :param Optional[int] receive_buffer_size: The buffer size we use to
        receive messages internally. None to let trio choose. Defaults
        to 4 KiB.
    :rtype: WebSocketConnection
    '''
    host, port, resource, return_ssl_context = _url_to_host(url, ssl_context)
    return await connect_websocket(nursery, host, port, resource,
        use_ssl=return_ssl_context, subprotocols=subprotocols,
        extra_headers=extra_headers, message_queue_size=message_queue_size,
        max_message_size=max_message_size,
        receive_buffer_size=receive_buffer_size)


def _url_to_host(
    url: str,
    ssl_context: ssl.SSLContext | None,
) -> tuple[str, int, str, ssl.SSLContext | bool]:
    '''
    Convert a WebSocket URL to a (host,port,resource) tuple.

    The returned ``ssl_context`` is either the same object that was passed in,
    or if ``ssl_context`` is None, then a bool indicating if a default SSL
    context needs to be created.

    :param str url: A WebSocket URL.
    :type ssl_context: ssl.SSLContext or None
    :returns: A tuple of ``(host, port, resource, ssl_context)``.
    '''
    url = str(url)  # For backward compat with isinstance(url, yarl.URL).
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ('ws', 'wss'):
        raise ValueError('WebSocket URL scheme must be "ws:" or "wss:"')
    return_ssl_context: ssl.SSLContext | bool
    if ssl_context is None:
        return_ssl_context = parts.scheme == 'wss'
    elif parts.scheme == 'ws':
        raise ValueError('SSL context must be None for ws: URL scheme')
    else:
        return_ssl_context = ssl_context
    host = parts.hostname
    if host is None:
        raise ValueError('URL host must not be None')
    if parts.port is not None:
        port = parts.port
    else:
        port = 443 if return_ssl_context else 80
    path_qs = parts.path
    # RFC 7230, Section 5.3.1:
    # If the target URI's path component is empty, the client MUST
    # send "/" as the path within the origin-form of request-target.
    if not path_qs:
        path_qs = '/'
    if '?' in url:
        path_qs += '?' + parts.query
    return host, port, path_qs, return_ssl_context


async def wrap_client_stream(
    nursery: trio.Nursery,
    stream: trio.SocketStream | trio.SSLStream[trio.SocketStream],
    host: str,
    resource: str,
    *,
    subprotocols: Iterable[str] | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
    message_queue_size: int = MESSAGE_QUEUE_SIZE,
    max_message_size: int = MAX_MESSAGE_SIZE,
    receive_buffer_size: Union[None, int] = RECEIVE_BYTES,
) -> WebSocketConnection:
    '''
    Wrap an arbitrary stream in a WebSocket connection.

    This is a low-level function only needed in rare cases. In most cases, you
    should use :func:`open_websocket` or :func:`open_websocket_url`.

    :param nursery: A Trio nursery to run background tasks in.
    :param stream: A Trio stream to be wrapped.
    :type stream: trio.abc.Stream
    :param str host: A host string that will be sent in the ``Host:`` header.
    :param str resource: A resource string, i.e. the path component to be
        accessed on the server.
    :param subprotocols: An iterable of strings representing preferred
        subprotocols.
    :param list[tuple[bytes,bytes]] extra_headers: A list of 2-tuples containing
        HTTP header key/value pairs to send with the connection request. Note
        that headers used by the WebSocket protocol (e.g.
        ``Sec-WebSocket-Accept``) will be overwritten.
    :param int message_queue_size: The maximum number of messages that will be
        buffered in the library's internal message queue.
    :param int max_message_size: The maximum message size as measured by
        ``len()``. If a message is received that is larger than this size,
        then the connection is closed with code 1009 (Message Too Big).
    :param Optional[int] receive_buffer_size: The buffer size we use to
        receive messages internally. None to let trio choose. Defaults
        to 4 KiB.
    :rtype: WebSocketConnection
    '''
    connection = WebSocketConnection(stream,
        WSConnection(ConnectionType.CLIENT),
        host=host, path=resource,
        client_subprotocols=subprotocols, client_extra_headers=extra_headers,
        message_queue_size=message_queue_size,
        max_message_size=max_message_size,
        receive_buffer_size=receive_buffer_size)
    nursery.start_soon(connection._reader_task)
    await connection._open_handshake.wait()
    return connection


async def wrap_server_stream(
    nursery: trio.Nursery,
    stream: trio.abc.Stream,
    message_queue_size: int = MESSAGE_QUEUE_SIZE,
    max_message_size: int = MAX_MESSAGE_SIZE,
    receive_buffer_size: Union[None, int] = RECEIVE_BYTES,
) -> WebSocketRequest:
    '''
    Wrap an arbitrary stream in a server-side WebSocket.

    This is a low-level function only needed in rare cases. In most cases, you
    should use :func:`serve_websocket`.

    :param nursery: A nursery to run background tasks in.
    :param stream: A stream to be wrapped.
    :param int message_queue_size: The maximum number of messages that will be
        buffered in the library's internal message queue.
    :param int max_message_size: The maximum message size as measured by
        ``len()``. If a message is received that is larger than this size,
        then the connection is closed with code 1009 (Message Too Big).
    :param Optional[int] receive_buffer_size: The buffer size we use to
        receive messages internally. None to let trio choose. Defaults
        to 4 KiB.
    :type stream: trio.abc.Stream
    :rtype: WebSocketRequest
    '''
    connection = WebSocketConnection(
        stream,
        WSConnection(ConnectionType.SERVER),
        message_queue_size=message_queue_size,
        max_message_size=max_message_size,
        receive_buffer_size=receive_buffer_size)
    nursery.start_soon(connection._reader_task)
    request = await connection._get_request()
    return request



async def serve_websocket(
    handler: Callable[[WebSocketRequest], Awaitable[None]],
    host: str | bytes | None,
    port: int,
    ssl_context: ssl.SSLContext | None,
    *,
    handler_nursery: trio.Nursery | None = None,
    message_queue_size: int = MESSAGE_QUEUE_SIZE,
    max_message_size: int = MAX_MESSAGE_SIZE,
    receive_buffer_size: Union[None, int] = RECEIVE_BYTES,
    connect_timeout: float = CONN_TIMEOUT,
    disconnect_timeout: float = CONN_TIMEOUT,
    task_status: trio.TaskStatus[WebSocketServer] = trio.TASK_STATUS_IGNORED,
) -> NoReturn:
    """
    Serve a WebSocket over TCP.

    This function supports the Trio nursery start protocol: ``server = await
    nursery.start(serve_websocket, …)``. It will block until the server
    is accepting connections and then return a :class:`WebSocketServer` object.

    Note that if ``host`` is ``None`` and ``port`` is zero, then you may get
    multiple listeners that have *different port numbers!*

    :param handler: An async function that is invoked with a request
        for each new connection.
    :param host: The host interface to bind. This can be an address of an
        interface, a name that resolves to an interface address (e.g.
        ``localhost``), or a wildcard address like ``0.0.0.0`` for IPv4 or
        ``::`` for IPv6. If ``None``, then all local interfaces are bound.
    :type host: str, bytes, or None
    :param int port: The port to bind to.
    :param ssl_context: The SSL context to use for encrypted connections, or
        ``None`` for unencrypted connection.
    :type ssl_context: ssl.SSLContext or None
    :param handler_nursery: An optional nursery to spawn handlers and background
        tasks in. If not specified, a new nursery will be created internally.
    :param int message_queue_size: The maximum number of messages that will be
        buffered in the library's internal message queue.
    :param int max_message_size: The maximum message size as measured by
        ``len()``. If a message is received that is larger than this size,
        then the connection is closed with code 1009 (Message Too Big).
    :param Optional[int] receive_buffer_size: The buffer size we use to
        receive messages internally. None to let trio choose. Defaults
        to 4 KiB.
    :param float connect_timeout: The number of seconds to wait for a client
        to finish connection handshake before timing out.
    :param float disconnect_timeout: The number of seconds to wait for a client
        to finish the closing handshake before timing out.
    :param task_status: Part of Trio nursery start protocol.
    :returns: This function runs until cancelled.
    """
    open_tcp_listeners: (
        partial[Coroutine[Any, Any, list[trio.SocketListener]]]
        | partial[Coroutine[Any, Any, list[trio.SSLListener[trio.SocketStream]]]]
    )
    if ssl_context is None:
        open_tcp_listeners = partial(trio.open_tcp_listeners, port, host=host)
    else:
        open_tcp_listeners = partial(
            trio.open_ssl_over_tcp_listeners,
            port,
            ssl_context,
            host=host,
            https_compatible=True,
        )
    listeners = await open_tcp_listeners()
    server = WebSocketServer(
        handler,
        listeners,
        handler_nursery=handler_nursery,
        message_queue_size=message_queue_size,
        max_message_size=max_message_size,
        receive_buffer_size=receive_buffer_size,
        connect_timeout=connect_timeout,
        disconnect_timeout=disconnect_timeout,
    )
    await server.run(task_status=task_status)


class HandshakeError(Exception):
    '''
    There was an error during connection or disconnection with the websocket
    server.
    '''

class ConnectionTimeout(HandshakeError):
    '''There was a timeout when connecting to the websocket server.'''

class DisconnectionTimeout(HandshakeError):
    '''There was a timeout when disconnecting from the websocket server.'''

class ConnectionClosed(Exception):
    '''
    A WebSocket operation cannot be completed because the connection is closed
    or in the process of closing.
    '''
    def __init__(self, reason: CloseReason | None) -> None:
        '''
        Constructor.

        :param reason:
        :type reason: CloseReason
        '''
        super().__init__(reason)
        self.reason = reason

    def __repr__(self) -> str:
        ''' Return representation. '''
        return f'{self.__class__.__name__}<{self.reason}>'


class ConnectionRejected(HandshakeError):
    '''
    A WebSocket connection could not be established because the server rejected
    the connection attempt.
    '''
    def __init__(
        self,
        status_code: int,
        headers: tuple[tuple[bytes, bytes], ...],
        body: bytes | None,
    ) -> None:
        '''
        Constructor.

        :param reason:
        :type reason: CloseReason
        '''
        super().__init__(status_code, headers, body)
        #: a 3 digit HTTP status code
        self.status_code = status_code
        #: a tuple of 2-tuples containing header key/value pairs
        self.headers = headers
        #: an optional ``bytes`` response body
        self.body = body

    def __repr__(self) -> str:
        ''' Return representation. '''
        return f'{self.__class__.__name__}<status_code={self.status_code}>'


class CloseReason:
    ''' Contains information about why a WebSocket was closed. '''
    def __init__(self, code: int, reason: str | None) -> None:
        '''
        Constructor.

        :param int code:
        :param Optional[str] reason:
        '''
        self._code = code
        try:
            self._name = wsframeproto.CloseReason(code).name
        except ValueError:
            if 1000 <= code <= 2999:
                self._name = 'RFC_RESERVED'
            elif 3000 <= code <= 3999:
                self._name = 'IANA_RESERVED'
            elif 4000 <= code <= 4999:
                self._name = 'PRIVATE_RESERVED'
            else:
                self._name = 'INVALID_CODE'
        self._reason = reason

    @property
    def code(self) -> int:
        ''' (Read-only) The numeric close code. '''
        return self._code

    @property
    def name(self) -> str:
        ''' (Read-only) The human-readable close code. '''
        return self._name

    @property
    def reason(self) -> str | None:
        ''' (Read-only) An arbitrary reason string. '''
        return self._reason

    def __repr__(self) -> str:
        ''' Show close code, name, and reason. '''
        return f'{self.__class__.__name__}' \
               f'<code={self.code}, name={self.name}, reason={self.reason}>'


NULL: Final = object()


class Future(Generic[T]):
    ''' Represents a value that will be available in the future. '''
    def __init__(self) -> None:
        ''' Constructor. '''
        # We do some type shenanigins
        # Would do `T | Literal[NULL]` but that's not right apparently.
        self._value: T = cast(T, NULL)
        self._value_event = trio.Event()

    def set_value(self, value: T) -> None:
        '''
        Set a value, which will notify any waiters.

        :param value:
        '''
        self._value = value
        self._value_event.set()

    async def wait_value(self) -> T:
        '''
        Wait for this future to have a value, then return it.

        :returns: The value set by ``set_value()``.
        '''
        await self._value_event.wait()
        assert self._value is not NULL
        return self._value


class WebSocketRequest:
    '''
    Represents a handshake presented by a client to a server.

    The server may modify the handshake or leave it as is. The server should
    call ``accept()`` to finish the handshake and obtain a connection object.
    '''
    def __init__(
        self,
        connection: WebSocketConnection,
        event: wsproto.events.Request,
    ) -> None:
        '''
        Constructor.

        :param WebSocketConnection connection:
        :type event: wsproto.events.Request
        '''
        self._connection = connection
        self._event = event

    @property
    def headers(self) -> list[tuple[bytes, bytes]]:
        '''
        HTTP headers represented as a list of (name, value) pairs.

        :rtype: list[tuple]
        '''
        return self._event.extra_headers

    @property
    def path(self) -> str:
        '''
        The requested URL path.

        :rtype: str
        '''
        return self._event.target

    @property
    def proposed_subprotocols(self) -> tuple[str, ...]:
        '''
        A tuple of protocols proposed by the client.

        :rtype: tuple[str]
        '''
        return tuple(self._event.subprotocols)

    @property
    def local(self) -> Endpoint | str:
        '''
        The connection's local endpoint.

        :rtype: Endpoint or str
        '''
        return self._connection.local

    @property
    def remote(self) -> Endpoint | str:
        '''
        The connection's remote endpoint.

        :rtype: Endpoint or str
        '''
        return self._connection.remote

    async def accept(
        self,
        *,
        subprotocol: str | None = None,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
    ) -> WebSocketConnection:
        '''
        Accept the request and return a connection object.

        :param subprotocol: The selected subprotocol for this connection.
        :type subprotocol: str or None
        :param extra_headers: A list of 2-tuples containing key/value pairs to
            send as HTTP headers.
        :type extra_headers: list[tuple[bytes,bytes]] or None
        :rtype: WebSocketConnection
        '''
        if extra_headers is None:
            extra_headers = []
        await self._connection._accept(self._event, subprotocol, extra_headers)
        return self._connection

    async def reject(
        self,
        status_code: int,
        *,
        extra_headers: list[tuple[bytes, bytes]] | None = None,
        body: bytes | None = None,
    ) -> None:
        '''
        Reject the handshake.

        :param int status_code: The 3 digit HTTP status code. In order to be
            RFC-compliant, this should NOT be 101, and would ideally be an
            appropriate code in the range 300-599.
        :param list[tuple[bytes,bytes]] extra_headers: A list of 2-tuples
            containing key/value pairs to send as HTTP headers.
        :param body: If provided, this data will be sent in the response
            body, otherwise no response body will be sent.
        :type body: bytes or None
        '''
        extra_headers = extra_headers or []
        body = body or b''
        await self._connection._reject(status_code, extra_headers, body)


def _get_stream_endpoint(
    stream: trio.abc.Stream,
    *,
    local: bool,
) -> Endpoint | str:
    '''
    Construct an endpoint from a stream.

    :param trio.Stream stream:
    :param bool local: If true, return local endpoint. Otherwise return remote.
    :returns: An endpoint instance or ``repr()`` for streams that cannot be
        represented as an endpoint.
    :rtype: Endpoint or str
    '''
    socket, is_ssl = None, False
    if isinstance(stream, trio.SocketStream):
        socket = stream.socket
    elif isinstance(stream, trio.SSLStream):
        socket = stream.transport_stream.socket
        is_ssl = True
    endpoint: Endpoint | str
    if socket:
        addr, port, *_ = socket.getsockname() if local else socket.getpeername()
        endpoint = Endpoint(addr, port, is_ssl)
    else:
        endpoint = repr(stream)
    return endpoint


class WebSocketConnection(trio.abc.AsyncResource):
    ''' A WebSocket connection. '''

    CONNECTION_ID = itertools.count()

    def __init__(
        self,
        stream: trio.abc.Stream,
        ws_connection: wsproto.WSConnection,
        *,
        host: str | None = None,
        path: str | None = None,
        client_subprotocols: Iterable[str] | None = None,
        client_extra_headers: list[tuple[bytes, bytes]] | None = None,
        message_queue_size: int = MESSAGE_QUEUE_SIZE,
        max_message_size: int = MAX_MESSAGE_SIZE,
        receive_buffer_size: Union[None, int] = RECEIVE_BYTES,
    ) -> None:
        '''
        Constructor.

        Generally speaking, users are discouraged from directly instantiating a
        ``WebSocketConnection`` and should instead use one of the convenience
        functions in this module, e.g. ``open_websocket()`` or
        ``serve_websocket()``. This class has some tricky internal logic and
        timing that depends on whether it is an instance of a client connection
        or a server connection. The convenience functions handle this complexity
        for you.

        :param SocketStream stream:
        :param ws_connection wsproto.WSConnection:
        :param str host: The hostname to send in the HTTP request headers. Only
            used for client connections.
        :param str path: The URL path for this connection.
        :param list client_subprotocols: A list of desired subprotocols. Only
            used for client connections.
        :param list[tuple[bytes,bytes]] client_extra_headers: Extra headers to
            send with the connection request. Only used for client connections.
        :param int message_queue_size: The maximum number of messages that will be
            buffered in the library's internal message queue.
        :param int max_message_size: The maximum message size as measured by
            ``len()``. If a message is received that is larger than this size,
            then the connection is closed with code 1009 (Message Too Big).
        :param Optional[int] receive_buffer_size: The buffer size we use to
            receive messages internally. None to let trio choose. Defaults
            to 4 KiB.
        '''
        # NOTE: The implementation uses _close_reason for more than an advisory
        #   purpose.  It's critical internal state, indicating when the
        #   connection is closed or closing.
        self._close_reason: Optional[CloseReason] = None
        self._id = next(self.__class__.CONNECTION_ID)
        self._stream = stream
        self._stream_lock = trio.StrictFIFOLock()
        self._wsproto = ws_connection
        self._message_size = 0
        self._message_parts: List[Union[bytes, str]] = []
        self._max_message_size = max_message_size
        self._receive_buffer_size: Optional[int] = receive_buffer_size
        self._reader_running = True
        if ws_connection.client:
            assert host is not None
            assert path is not None
            self._initial_request: Optional[Request] = Request(host=host, target=path,
                subprotocols=list(client_subprotocols or ()),
                extra_headers=client_extra_headers or [])
        else:
            self._initial_request = None
        self._path = path
        self._subprotocol: Optional[str] = None
        self._handshake_headers: tuple[tuple[bytes, bytes], ...] = ()
        self._reject_status = 0
        self._reject_headers: tuple[tuple[bytes, bytes], ...] = ()
        self._reject_body = b''
        self._send_channel, self._recv_channel = trio.open_memory_channel[
            Union[bytes, str]
        ](message_queue_size)
        self._pings: OrderedDict[bytes, trio.Event] = OrderedDict()
        # Set when the server has received a connection request event. This
        # future is never set on client connections.
        self._connection_proposal: Future[WebSocketRequest] | None = Future[WebSocketRequest]()
        # Set once the WebSocket open handshake takes place, i.e.
        # ConnectionRequested for server or ConnectedEstablished for client.
        self._open_handshake = trio.Event()
        # Set once a WebSocket closed handshake takes place, i.e after a close
        # frame has been sent and a close frame has been received.
        self._close_handshake = trio.Event()
        # Set upon receiving CloseConnection from peer.
        # Used to test close race conditions between client and server.
        self._for_testing_peer_closed_connection = trio.Event()

    @property
    def closed(self) -> CloseReason | None:
        '''
        (Read-only) The reason why the connection was or is being closed,
        else ``None``.

        :rtype: Optional[CloseReason]
        '''
        return self._close_reason

    @property
    def is_client(self) -> bool:
        ''' (Read-only) Is this a client instance? '''
        return self._wsproto.client

    @property
    def is_server(self) -> bool:
        ''' (Read-only) Is this a server instance? '''
        return not self._wsproto.client

    @property
    def local(self) -> Endpoint | str:
        '''
        The local endpoint of the connection.

        :rtype: Endpoint or str
        '''
        return _get_stream_endpoint(self._stream, local=True)

    @property
    def remote(self) -> Endpoint | str:
        '''
        The remote endpoint of the connection.

        :rtype: Endpoint or str
        '''
        return _get_stream_endpoint(self._stream, local=False)

    @property
    def path(self) -> str | None:
        '''
        The requested URL path. For clients, this is set when the connection is
        instantiated. For servers, it is set after the handshake completes.

        :rtype: str or None
        '''
        return self._path

    @property
    def subprotocol(self) -> str | None:
        '''
        (Read-only) The negotiated subprotocol, or ``None`` if there is no
        subprotocol.

        This is only valid after the opening handshake is complete.

        :rtype: str or None
        '''
        return self._subprotocol

    @property
    def handshake_headers(self) -> tuple[tuple[bytes, bytes], ...]:
        '''
        The HTTP headers that were sent by the remote during the handshake,
        stored as 2-tuples containing key/value pairs. Header keys are always
        lower case.

        :rtype: tuple[tuple[str,str]]
        '''
        return self._handshake_headers

    async def aclose(self, code: int = 1000, reason: str | None = None) -> None:  # pylint: disable=arguments-differ
        '''
        Close the WebSocket connection.

        This sends a closing frame and suspends until the connection is closed.
        After calling this method, any further I/O on this WebSocket (such as
        ``get_message()`` or ``send_message()``) will raise
        ``ConnectionClosed``.

        This method is idempotent: it may be called multiple times on the same
        connection without any errors.

        :param int code: A 4-digit code number indicating the type of closure.
        :param str reason: An optional string describing the closure.
        '''
        with _preserve_current_exception():
            await self._aclose(code, reason)

    async def _aclose(self, code: int, reason: str | None) -> None:
        if self._close_reason:
            # Per AsyncResource interface, calling aclose() on a closed resource
            # should succeed.
            return
        try:
            if self._wsproto.state == ConnectionState.OPEN:
                # Our side is initiating the close, so send a close connection
                # event to peer, while setting the local close reason to normal.
                self._close_reason = CloseReason(1000, None)
                await self._send(CloseConnection(code=code, reason=reason))
            elif self._wsproto.state in (ConnectionState.CONNECTING,
                    ConnectionState.REJECTING):
                self._close_handshake.set()
            # TODO: shouldn't the receive channel be closed earlier, so that
            #  get_message() during send of the CloseConneciton event fails?
            await self._recv_channel.aclose()
            await self._close_handshake.wait()
        except ConnectionClosed:
            # If _send() raised ConnectionClosed, then we can bail out.
            pass
        finally:
            # If cancelled during WebSocket close, make sure that the underlying
            # stream is closed.
            await self._close_stream()

    async def get_message(self) -> str | bytes:
        '''
        Receive the next WebSocket message.

        If no message is available immediately, then this function blocks until
        a message is ready.

        If the remote endpoint closes the connection, then the caller can still
        get messages sent prior to closing. Once all pending messages have been
        retrieved, additional calls to this method will raise
        ``ConnectionClosed``. If the local endpoint closes the connection, then
        pending messages are discarded and calls to this method will immediately
        raise ``ConnectionClosed``.

        :rtype: str or bytes
        :raises ConnectionClosed: if the connection is closed.
        '''
        try:
            message = await self._recv_channel.receive()
        except (trio.ClosedResourceError, trio.EndOfChannel):
            raise ConnectionClosed(self._close_reason) from None
        return message

    async def ping(self, payload: bytes | None = None) -> None:
        '''
        Send WebSocket ping to remote endpoint and wait for a correspoding pong.

        Each in-flight ping must include a unique payload. This function sends
        the ping and then waits for a corresponding pong from the remote
        endpoint.

        *Note: If the remote endpoint recieves multiple pings, it is allowed to
        send a single pong. Therefore, the order of calls to ``ping()`` is
        tracked, and a pong will wake up its corresponding ping as well as all
        previous in-flight pings.*

        :param payload: The payload to send. If ``None`` then a random 32-bit
            payload is created.
        :type payload: bytes or None
        :raises ConnectionClosed: if connection is closed.
        :raises ValueError: if ``payload`` is identical to another in-flight
            ping.
        '''
        if self._close_reason:
            raise ConnectionClosed(self._close_reason)
        if payload in self._pings:
            raise ValueError(f'Payload value {payload!r} is already in flight.')
        if payload is None:
            payload = struct.pack('!I', random.getrandbits(32))
        event = trio.Event()
        self._pings[payload] = event
        await self._send(Ping(payload=payload))
        await event.wait()

    async def pong(self, payload: bytes | None = None) -> None:
        '''
        Send an unsolicted pong.

        :param payload: The pong's payload. If ``None``, then no payload is
            sent.
        :type payload: bytes or None
        :raises ConnectionClosed: if connection is closed
        '''
        if self._close_reason:
            raise ConnectionClosed(self._close_reason)
        await self._send(Pong(payload=payload or b''))

    async def send_message(self, message: str | bytes) -> None:
        '''
        Send a WebSocket message.

        :param message: The message to send.
        :type message: str or bytes
        :raises ConnectionClosed: if connection is closed, or being closed
        '''
        if self._close_reason:
            raise ConnectionClosed(self._close_reason)
        event: TextMessage | BytesMessage
        if isinstance(message, str):
            event = TextMessage(data=message)
        elif isinstance(message, bytes):
            event = BytesMessage(data=message)
        else:
            raise ValueError('message must be str or bytes')
        await self._send(event)

    def __str__(self) -> str:
        ''' Connection ID and type. '''
        type_ = 'client' if self.is_client else 'server'
        return f'{type_}-{self._id}'

    async def _accept(
        self,
        request: Request,
        subprotocol: str | None,
        extra_headers: list[tuple[bytes, bytes]],
    ) -> None:
        '''
        Accept the handshake.

        This method is only applicable to server-side connections.

        :param wsproto.events.Request request:
        :param subprotocol:
        :type subprotocol: str or None
        :param list[tuple[bytes,bytes]] extra_headers: A list of 2-tuples
            containing key/value pairs to send as HTTP headers.
        '''
        self._subprotocol = subprotocol
        self._path = request.target
        await self._send(AcceptConnection(subprotocol=self._subprotocol,
            extra_headers=extra_headers))
        self._open_handshake.set()

    async def _reject(
        self,
        status_code: int,
        headers: list[tuple[bytes, bytes]],
        body: bytes,
    ) -> None:
        '''
        Reject the handshake.

        :param int status_code: The 3 digit HTTP status code. In order to be
            RFC-compliant, this must not be 101, and should be an appropriate
            code in the range 300-599.
        :param list[tuple[bytes,bytes]] headers: A list of 2-tuples containing
            key/value pairs to send as HTTP headers.
        :param bytes body: An optional response body.
        '''
        if body:
            headers.append((b'Content-length', str(len(body)).encode('ascii')))
        reject_conn = RejectConnection(status_code=status_code, headers=headers,
            has_body=bool(body))
        await self._send(reject_conn)
        if body:
            reject_body = RejectData(data=body)
            await self._send(reject_body)
        self._close_reason = CloseReason(1006, 'Rejected WebSocket handshake')
        self._close_handshake.set()

    async def _abort_web_socket(self) -> None:
        '''
        If a stream is closed outside of this class, e.g. due to network
        conditions or because some other code closed our stream object, then we
        cannot perform the close handshake. We just need to clean up internal
        state.
        '''
        close_reason = wsframeproto.CloseReason.ABNORMAL_CLOSURE
        if self._wsproto.state == ConnectionState.OPEN:
            self._wsproto.send(CloseConnection(code=close_reason.value))
        if self._close_reason is None:
            await self._close_web_socket(close_reason)
        self._reader_running = False
        # We didn't really handshake, but we want any task waiting on this event
        # (e.g. self.aclose()) to resume.
        self._close_handshake.set()

    async def _close_stream(self) -> None:
        ''' Close the TCP connection. '''
        self._reader_running = False
        try:
            with _preserve_current_exception():
                await self._stream.aclose()
        except trio.BrokenResourceError:
            # This means the TCP connection is already dead.
            pass

    async def _close_web_socket(self, code: int, reason: str | None = None) -> None:
        '''
        Mark the WebSocket as closed. Close the message channel so that if any
        tasks are suspended in get_message(), they will wake up with a
        ConnectionClosed exception.
        '''
        self._close_reason = CloseReason(code, reason)
        exc = ConnectionClosed(self._close_reason)
        logger.debug('%s websocket closed %r', self, exc)
        await self._send_channel.aclose()

    async def _get_request(self) -> WebSocketRequest:
        '''
        Return a proposal for a WebSocket handshake.

        This method can only be called on server connections and it may only be
        called one time.

        :rtype: WebSocketRequest
        '''
        if not self.is_server:
            raise RuntimeError('This method is only valid for server connections.')
        if self._connection_proposal is None:
            raise RuntimeError('No proposal available. Did you call this method'
                ' multiple times or at the wrong time?')
        proposal = await self._connection_proposal.wait_value()
        self._connection_proposal = None
        return proposal

    async def _handle_request_event(self, event: wsproto.events.Request) -> None:
        '''
        Handle a connection request.

        This method is async even though it never awaits, because the event
        dispatch requires an async function.

        :param event:
        '''
        proposal = WebSocketRequest(self, event)
        assert self._connection_proposal is not None
        self._connection_proposal.set_value(proposal)

    async def _handle_accept_connection_event(self, event: wsproto.events.AcceptConnection) -> None:
        '''
        Handle an AcceptConnection event.

        :param wsproto.eventsAcceptConnection event:
        '''
        self._subprotocol = event.subprotocol
        self._handshake_headers = tuple(event.extra_headers)
        self._open_handshake.set()

    async def _handle_reject_connection_event(self, event: wsproto.events.RejectConnection) -> None:
        '''
        Handle a RejectConnection event.

        :param event:
        '''
        self._reject_status = event.status_code
        self._reject_headers = tuple(event.headers)
        if not event.has_body:
            raise ConnectionRejected(self._reject_status, self._reject_headers,
                body=None)

    async def _handle_reject_data_event(self, event: wsproto.events.RejectData) -> None:
        '''
        Handle a RejectData event.

        :param event:
        '''
        self._reject_body += event.data
        if event.body_finished:
            raise ConnectionRejected(self._reject_status, self._reject_headers,
                body=self._reject_body)

    async def _handle_close_connection_event(self, event: wsproto.events.CloseConnection) -> None:
        '''
        Handle a close event.

        :param wsproto.events.CloseConnection event:
        '''
        if self._wsproto.state == ConnectionState.REMOTE_CLOSING:
            # Set _close_reason in advance, so that send_message() will raise
            # ConnectionClosed during the close handshake.
            self._close_reason = CloseReason(event.code, event.reason or None)
            self._for_testing_peer_closed_connection.set()
            await self._send(event.response())
        await self._close_web_socket(event.code, event.reason or None)
        self._close_handshake.set()
        # RFC: "When a server is instructed to Close the WebSocket Connection
        #   it SHOULD initiate a TCP Close immediately, and when a client is
        #   instructed to do the same, it SHOULD wait for a TCP Close from the
        #   server."
        if self.is_server:
            await self._close_stream()

    async def _handle_message_event(
        self,
        event: wsproto.events.BytesMessage | wsproto.events.TextMessage,
    ) -> None:
        '''
        Handle a message event.

        :param event:
        :type event: wsproto.events.BytesMessage or wsproto.events.TextMessage
        '''
        self._message_size += len(event.data)
        self._message_parts.append(event.data)
        if self._message_size > self._max_message_size:
            err = f'Exceeded maximum message size: {self._max_message_size} bytes'
            self._message_size = 0
            self._message_parts = []
            self._close_reason = CloseReason(1009, err)
            await self._send(CloseConnection(code=1009, reason=err))
            await self._recv_channel.aclose()
            self._reader_running = False
        elif event.message_finished:
            msg: str | bytes
            # Type checker does not understand `_message_parts`
            if isinstance(event, BytesMessage):
                msg = b''.join(cast("list[bytes]", self._message_parts))
            else:
                msg = ''.join(cast("list[str]", self._message_parts))
            self._message_size = 0
            self._message_parts = []
            try:
                await self._send_channel.send(msg)
            except (trio.ClosedResourceError, trio.BrokenResourceError):
                # The receive channel is closed, probably because somebody
                # called ``aclose()``. We don't want to abort the reader task,
                # and there's no useful cleanup that we can do here.
                pass

    async def _handle_ping_event(self, event: wsproto.events.Ping) -> None:
        '''
        Handle a PingReceived event.

        Wsproto queues a pong frame automatically, so this handler just needs to
        send it.

        :param wsproto.events.Ping event:
        '''
        logger.debug('%s ping %r', self, event.payload)
        await self._send(event.response())

    async def _handle_pong_event(self, event: wsproto.events.Pong) -> None:
        '''
        Handle a PongReceived event.

        When a pong is received, check if we have any ping requests waiting for
        this pong response. If the remote endpoint skipped any earlier pings,
        then we wake up those skipped pings, too.

        This function is async even though it never awaits, because the other
        event handlers are async, too, and event dispatch would be more
        complicated if some handlers were sync.

        :param event:
        '''
        payload = bytes(event.payload)
        try:
            ping_event = self._pings[payload]
        except KeyError:
            # We received a pong that doesn't match any in-flight pongs. Nothing
            # we can do with it, so ignore it.
            return
        while self._pings:
            key, ping_event = self._pings.popitem(False)
            skipped = ' [skipped] ' if payload != key else ' '
            logger.debug('%s pong%s%r', self, skipped, key)
            ping_event.set()
            if payload == key:
                break

    async def _reader_task(self) -> None:
        ''' A background task that reads network data and generates events. '''
        handlers = {
            AcceptConnection: self._handle_accept_connection_event,
            BytesMessage: self._handle_message_event,
            CloseConnection: self._handle_close_connection_event,
            Ping: self._handle_ping_event,
            Pong: self._handle_pong_event,
            RejectConnection: self._handle_reject_connection_event,
            RejectData: self._handle_reject_data_event,
            Request: self._handle_request_event,
            TextMessage: self._handle_message_event,
        }

        # Clients need to initiate the opening handshake.
        if self._initial_request:
            try:
                await self._send(self._initial_request)
            except ConnectionClosed:
                self._reader_running = False

        async with self._send_channel:
            while self._reader_running:
                # Process events.
                for event in self._wsproto.events():
                    event_type = type(event)
                    try:
                        handler = handlers[event_type]
                        logger.debug('%s received event: %s', self,
                            event_type)
                        # Type checkers don't understand looking up type in handlers.
                        # If we wanted to fix, best I can figure is we'd need a huge
                        # if-else or case block for every type individually.
                        await handler(event)  # type: ignore[operator]
                    except KeyError:
                        logger.warning('%s received unknown event type: "%s"', self,
                            event_type)
                    except ConnectionClosed:
                        self._reader_running = False
                        break

                # Get network data.
                try:
                    data = await self._stream.receive_some(self._receive_buffer_size)
                except (trio.BrokenResourceError, trio.ClosedResourceError):
                    await self._abort_web_socket()
                    break
                if len(data) == 0:
                    logger.debug('%s received zero bytes (connection closed)',
                        self)
                    # If TCP closed before WebSocket, then record it as an abnormal
                    # closure.
                    if self._wsproto.state != ConnectionState.CLOSED:
                        await self._abort_web_socket()
                    break
                logger.debug('%s received %d bytes', self, len(data))
                if self._wsproto.state != ConnectionState.CLOSED:
                    try:
                        self._wsproto.receive_data(data)
                    except wsproto.utilities.RemoteProtocolError as err:
                        logger.debug('%s remote protocol error: %s', self, err)
                        if err.event_hint:
                            await self._send(err.event_hint)
                        await self._close_stream()

        logger.debug('%s reader task finished', self)

    async def _send(self, event: wsproto.events.Event) -> None:
        '''
        Send an event to the remote WebSocket.

        The reader task and one or more writers might try to send messages at
        the same time, so this method uses an internal lock to serialize
        requests to send data.

        :param wsproto.events.Event event:
        '''
        data = self._wsproto.send(event)
        async with self._stream_lock:
            logger.debug('%s sending %d bytes', self, len(data))
            try:
                await self._stream.send_all(data)
            except (trio.BrokenResourceError, trio.ClosedResourceError):
                await self._abort_web_socket()
                assert self._close_reason is not None
                raise ConnectionClosed(self._close_reason) from None


class Endpoint:
    ''' Represents a connection endpoint. '''
    def __init__(self, address: str | int, port: int, is_ssl: bool) -> None:
        #: IP address :class:`ipaddress.ip_address`
        self.address = ip_address(address)
        #: TCP port
        self.port = port
        #: Whether SSL is in use
        self.is_ssl = is_ssl

    @property
    def url(self) -> str:
        ''' Return a URL representation of a TCP endpoint, e.g.
        ``ws://127.0.0.1:80``. '''
        scheme = 'wss' if self.is_ssl else 'ws'
        if (self.port == 80 and not self.is_ssl) or \
           (self.port == 443 and self.is_ssl):
            port_str = ''
        else:
            port_str = ':' + str(self.port)
        if self.address.version == 4:
            return f'{scheme}://{self.address}{port_str}'
        return f'{scheme}://[{self.address}]{port_str}'

    def __repr__(self) -> str:
        ''' Return endpoint info as string. '''
        return f'Endpoint(address="{self.address}", port={self.port}, is_ssl={self.is_ssl})'


class WebSocketServer:
    '''
    WebSocket server.

    The server class handles incoming connections on one or more ``Listener``
    objects. For each incoming connection, it creates a ``WebSocketConnection``
    instance and starts some background tasks,
    '''

    def __init__(
        self,
        handler: Callable[[WebSocketRequest], Awaitable[None]],
        listeners: Sequence[trio.SSLListener[trio.SocketStream] | trio.SocketListener],
        *,
        handler_nursery: trio.Nursery | None = None,
        message_queue_size: int = MESSAGE_QUEUE_SIZE,
        max_message_size: int = MAX_MESSAGE_SIZE,
        receive_buffer_size: Union[None, int] = RECEIVE_BYTES,
        connect_timeout: float = CONN_TIMEOUT,
        disconnect_timeout: float = CONN_TIMEOUT,
    ) -> None:
        '''
        Constructor.

        Note that if ``host`` is ``None`` and ``port`` is zero, then you may get
        multiple listeners that have _different port numbers!_ See the
        ``listeners`` property.

        :param handler: the async function called with a :class:`WebSocketRequest`
            on each new connection.  The call will be made
            once the HTTP handshake completes, which notably implies that the
            connection's `path` property will be valid.
        :param listeners: The WebSocket will be served on each of the listeners.
        :param handler_nursery: An optional nursery to spawn connection tasks
            inside of. If ``None``, then a new nursery will be created
            internally.
        :param Optional[int] receive_buffer_size: The buffer size we use to
            receive messages internally. None to let trio choose. Defaults
            to 4 KiB.
        :param float connect_timeout: The number of seconds to wait for a client
            to finish connection handshake before timing out.
        :param float disconnect_timeout: The number of seconds to wait for a client
            to finish the closing handshake before timing out.
        '''
        if len(listeners) == 0:
            raise ValueError('Listeners must contain at least one item.')
        self._handler = handler
        self._handler_nursery = handler_nursery
        self._listeners = listeners
        self._message_queue_size = message_queue_size
        self._max_message_size = max_message_size
        self._receive_buffer_size = receive_buffer_size
        self._connect_timeout = connect_timeout
        self._disconnect_timeout = disconnect_timeout

    @property
    def port(self) -> int:
        """Returns the requested or kernel-assigned port number.

        In the case of kernel-assigned port (requested with port=0 in the
        constructor), the assigned port will be reflected after calling
        starting the `listen` task.  (Technically, once listen reaches the
        "started" state.)

        This property only works if you have a single listener, and that
        listener must be socket-based.
        """
        if len(self._listeners) > 1:
            raise RuntimeError('Cannot get port because this server has'
                ' more than 1 listener.')
        listener = self.listeners[0]
        try:
            return listener.port  # type: ignore[union-attr]
        except AttributeError:
            raise RuntimeError(f'This socket does not have a port: {repr(listener)}') from None

    @property
    def listeners(self) -> list[Endpoint | str]:
        '''
        Return a list of listener metadata. Each TCP listener is represented as
        an ``Endpoint`` instance. Other listener types are represented by their
        ``repr()``.

        :returns: Listeners
        :rtype list[Endpoint or str]:
        '''
        listeners: list[Endpoint | str] = []
        for listener in self._listeners:
            socket, is_ssl = None, False
            if isinstance(listener, trio.SocketListener):
                socket = listener.socket
            elif isinstance(listener, trio.SSLListener):
                internal_listener = listener.transport_listener
                assert isinstance(internal_listener, trio.SocketListener)
                socket = internal_listener.socket
                is_ssl = True
            if socket:
                sockname = socket.getsockname()
                listeners.append(Endpoint(sockname[0], sockname[1], is_ssl))
            else:
                listeners.append(repr(listener))
        return listeners

    # Type ignore is because type checker does not think NoReturn is
    # real for Trio 0.25.1 (current version used in requirements file as
    # of writing). Not a problem for newer versions however, which is
    # why we have unused-ignore as well.
    async def run(  # type: ignore[misc,unused-ignore]
        self,
        *,
        task_status: trio.TaskStatus[WebSocketServer] = trio.TASK_STATUS_IGNORED,
    ) -> NoReturn:
        '''
        Start serving incoming connections requests.

        This method supports the Trio nursery start protocol: ``server = await
        nursery.start(server.run, …)``. It will block until the server is
        accepting connections and then return a :class:`WebSocketServer` object.

        :param task_status: Part of the Trio nursery start protocol.
        :returns: This method never returns unless cancelled.
        '''
        async with trio.open_nursery() as nursery:
            serve_listeners = partial(trio.serve_listeners,
                self._handle_connection, list(self._listeners),
                handler_nursery=self._handler_nursery)
            await nursery.start(serve_listeners)
            logger.debug('Listening on %s',
                ','.join([str(l) for l in self.listeners]))
            task_status.started(self)
            await trio.sleep_forever()

    async def _handle_connection(self, stream: trio.abc.Stream) -> None:
        '''
        Handle an incoming connection by spawning a connection background task
        and a handler task inside a new nursery.

        :param stream:
        :type stream: trio.abc.Stream
        '''
        async with trio.open_nursery() as nursery:
            connection = WebSocketConnection(stream,
                WSConnection(ConnectionType.SERVER),
                message_queue_size=self._message_queue_size,
                max_message_size=self._max_message_size,
                receive_buffer_size=self._receive_buffer_size)
            nursery.start_soon(connection._reader_task)
            with trio.move_on_after(self._connect_timeout) as connect_scope:
                request = await connection._get_request()
            if connect_scope.cancelled_caught:
                nursery.cancel_scope.cancel()
                await stream.aclose()
                return
            try:
                await self._handler(request)
            finally:
                with trio.move_on_after(self._disconnect_timeout):
                    # aclose() will shut down the reader task even if it's
                    # cancelled:
                    await connection.aclose()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\rv.py ===
"""
Main Random Variables Module

Defines abstract random variable type.
Contains interfaces for probability space object (PSpace) as well as standard
operators, P, E, sample, density, where, quantile

See Also
========

sympy.stats.crv
sympy.stats.frv
sympy.stats.rv_interface
"""

from __future__ import annotations
from functools import singledispatch
from math import prod

from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import (Function, Lambda)
from sympy.core.logic import fuzzy_and
from sympy.core.mul import Mul
from sympy.core.relational import (Eq, Ne)
from sympy.core.singleton import S
from sympy.core.symbol import (Dummy, Symbol)
from sympy.core.sympify import sympify
from sympy.functions.special.delta_functions import DiracDelta
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.logic.boolalg import (And, Or)
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.tensor.indexed import Indexed
from sympy.utilities.lambdify import lambdify
from sympy.core.relational import Relational
from sympy.core.sympify import _sympify
from sympy.sets.sets import FiniteSet, ProductSet, Intersection
from sympy.solvers.solveset import solveset
from sympy.external import import_module
from sympy.utilities.decorator import doctest_depends_on
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import iterable


__doctest_requires__ = {('sample',): ['scipy']}


x = Symbol('x')


@singledispatch
def is_random(x):
    return False


@is_random.register(Basic)
def _(x):
    atoms = x.free_symbols
    return any(is_random(i) for i in atoms)


class RandomDomain(Basic):
    """
    Represents a set of variables and the values which they can take.

    See Also
    ========

    sympy.stats.crv.ContinuousDomain
    sympy.stats.frv.FiniteDomain
    """

    is_ProductDomain = False
    is_Finite = False
    is_Continuous = False
    is_Discrete = False

    def __new__(cls, symbols, *args):
        symbols = FiniteSet(*symbols)
        return Basic.__new__(cls, symbols, *args)

    @property
    def symbols(self):
        return self.args[0]

    @property
    def set(self):
        return self.args[1]

    def __contains__(self, other):
        raise NotImplementedError()

    def compute_expectation(self, expr):
        raise NotImplementedError()


class SingleDomain(RandomDomain):
    """
    A single variable and its domain.

    See Also
    ========

    sympy.stats.crv.SingleContinuousDomain
    sympy.stats.frv.SingleFiniteDomain
    """
    def __new__(cls, symbol, set):
        assert symbol.is_Symbol
        return Basic.__new__(cls, symbol, set)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def symbols(self):
        return FiniteSet(self.symbol)

    def __contains__(self, other):
        if len(other) != 1:
            return False
        sym, val = tuple(other)[0]
        return self.symbol == sym and val in self.set


class MatrixDomain(RandomDomain):
    """
    A Random Matrix variable and its domain.

    """
    def __new__(cls, symbol, set):
        symbol, set = _symbol_converter(symbol), _sympify(set)
        return Basic.__new__(cls, symbol, set)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def symbols(self):
        return FiniteSet(self.symbol)


class ConditionalDomain(RandomDomain):
    """
    A RandomDomain with an attached condition.

    See Also
    ========

    sympy.stats.crv.ConditionalContinuousDomain
    sympy.stats.frv.ConditionalFiniteDomain
    """
    def __new__(cls, fulldomain, condition):
        condition = condition.xreplace({rs: rs.symbol
            for rs in random_symbols(condition)})
        return Basic.__new__(cls, fulldomain, condition)

    @property
    def symbols(self):
        return self.fulldomain.symbols

    @property
    def fulldomain(self):
        return self.args[0]

    @property
    def condition(self):
        return self.args[1]

    @property
    def set(self):
        raise NotImplementedError("Set of Conditional Domain not Implemented")

    def as_boolean(self):
        return And(self.fulldomain.as_boolean(), self.condition)


class PSpace(Basic):
    """
    A Probability Space.

    Explanation
    ===========

    Probability Spaces encode processes that equal different values
    probabilistically. These underly Random Symbols which occur in SymPy
    expressions and contain the mechanics to evaluate statistical statements.

    See Also
    ========

    sympy.stats.crv.ContinuousPSpace
    sympy.stats.frv.FinitePSpace
    """

    is_Finite: bool | None = None  # Fails test if not set to None
    is_Continuous: bool | None = None  # Fails test if not set to None
    is_Discrete: bool | None = None  # Fails test if not set to None
    is_real: bool | None

    @property
    def domain(self):
        return self.args[0]

    @property
    def density(self):
        return self.args[1]

    @property
    def values(self):
        return frozenset(RandomSymbol(sym, self) for sym in self.symbols)

    @property
    def symbols(self):
        return self.domain.symbols

    def where(self, condition):
        raise NotImplementedError()

    def compute_density(self, expr):
        raise NotImplementedError()

    def sample(self, size=(), library='scipy', seed=None):
        raise NotImplementedError()

    def probability(self, condition):
        raise NotImplementedError()

    def compute_expectation(self, expr):
        raise NotImplementedError()


class SinglePSpace(PSpace):
    """
    Represents the probabilities of a set of random events that can be
    attributed to a single variable/symbol.
    """
    def __new__(cls, s, distribution):
        s = _symbol_converter(s)
        return Basic.__new__(cls, s, distribution)

    @property
    def value(self):
        return RandomSymbol(self.symbol, self)

    @property
    def symbol(self):
        return self.args[0]

    @property
    def distribution(self):
        return self.args[1]

    @property
    def pdf(self):
        return self.distribution.pdf(self.symbol)


class RandomSymbol(Expr):
    """
    Random Symbols represent ProbabilitySpaces in SymPy Expressions.
    In principle they can take on any value that their symbol can take on
    within the associated PSpace with probability determined by the PSpace
    Density.

    Explanation
    ===========

    Random Symbols contain pspace and symbol properties.
    The pspace property points to the represented Probability Space
    The symbol is a standard SymPy Symbol that is used in that probability space
    for example in defining a density.

    You can form normal SymPy expressions using RandomSymbols and operate on
    those expressions with the Functions

    E - Expectation of a random expression
    P - Probability of a condition
    density - Probability Density of an expression
    given - A new random expression (with new random symbols) given a condition

    An object of the RandomSymbol type should almost never be created by the
    user. They tend to be created instead by the PSpace class's value method.
    Traditionally a user does not even do this but instead calls one of the
    convenience functions Normal, Exponential, Coin, Die, FiniteRV, etc....
    """

    def __new__(cls, symbol, pspace=None):
        from sympy.stats.joint_rv import JointRandomSymbol
        if pspace is None:
            # Allow single arg, representing pspace == PSpace()
            pspace = PSpace()
        symbol = _symbol_converter(symbol)
        if not isinstance(pspace, PSpace):
            raise TypeError("pspace variable should be of type PSpace")
        if cls == JointRandomSymbol and isinstance(pspace, SinglePSpace):
            cls = RandomSymbol
        return Basic.__new__(cls, symbol, pspace)

    is_finite = True
    is_symbol = True
    is_Atom = True

    _diff_wrt = True

    pspace = property(lambda self: self.args[1])
    symbol = property(lambda self: self.args[0])
    name   = property(lambda self: self.symbol.name)

    def _eval_is_positive(self):
        return self.symbol.is_positive

    def _eval_is_integer(self):
        return self.symbol.is_integer

    def _eval_is_real(self):
        return self.symbol.is_real or self.pspace.is_real

    @property
    def is_commutative(self):
        return self.symbol.is_commutative

    @property
    def free_symbols(self):
        return {self}

class RandomIndexedSymbol(RandomSymbol):

    def __new__(cls, idx_obj, pspace=None):
        if pspace is None:
            # Allow single arg, representing pspace == PSpace()
            pspace = PSpace()
        if not isinstance(idx_obj, (Indexed, Function)):
            raise TypeError("An Function or Indexed object is expected not %s"%(idx_obj))
        return Basic.__new__(cls, idx_obj, pspace)

    symbol = property(lambda self: self.args[0])
    name = property(lambda self: str(self.args[0]))

    @property
    def key(self):
        if isinstance(self.symbol, Indexed):
            return self.symbol.args[1]
        elif isinstance(self.symbol, Function):
            return self.symbol.args[0]

    @property
    def free_symbols(self):
        if self.key.free_symbols:
            free_syms = self.key.free_symbols
            free_syms.add(self)
            return free_syms
        return {self}

    @property
    def pspace(self):
        return self.args[1]

class RandomMatrixSymbol(RandomSymbol, MatrixSymbol): # type: ignore
    def __new__(cls, symbol, n, m, pspace=None):
        n, m = _sympify(n), _sympify(m)
        symbol = _symbol_converter(symbol)
        if pspace is None:
            # Allow single arg, representing pspace == PSpace()
            pspace = PSpace()
        return Basic.__new__(cls, symbol, n, m, pspace)

    symbol = property(lambda self: self.args[0])
    pspace = property(lambda self: self.args[3])


class ProductPSpace(PSpace):
    """
    Abstract class for representing probability spaces with multiple random
    variables.

    See Also
    ========

    sympy.stats.rv.IndependentProductPSpace
    sympy.stats.joint_rv.JointPSpace
    """
    pass

class IndependentProductPSpace(ProductPSpace):
    """
    A probability space resulting from the merger of two independent probability
    spaces.

    Often created using the function, pspace.
    """

    def __new__(cls, *spaces):
        rs_space_dict = {}
        for space in spaces:
            for value in space.values:
                rs_space_dict[value] = space

        symbols = FiniteSet(*[val.symbol for val in rs_space_dict.keys()])

        # Overlapping symbols
        from sympy.stats.joint_rv import MarginalDistribution
        from sympy.stats.compound_rv import CompoundDistribution
        if len(symbols) < sum(len(space.symbols) for space in spaces if not
         isinstance(space.distribution, (
            CompoundDistribution, MarginalDistribution))):
            raise ValueError("Overlapping Random Variables")

        if all(space.is_Finite for space in spaces):
            from sympy.stats.frv import ProductFinitePSpace
            cls = ProductFinitePSpace

        obj = Basic.__new__(cls, *FiniteSet(*spaces))

        return obj

    @property
    def pdf(self):
        p = Mul(*[space.pdf for space in self.spaces])
        return p.subs({rv: rv.symbol for rv in self.values})

    @property
    def rs_space_dict(self):
        d = {}
        for space in self.spaces:
            for value in space.values:
                d[value] = space
        return d

    @property
    def symbols(self):
        return FiniteSet(*[val.symbol for val in self.rs_space_dict.keys()])

    @property
    def spaces(self):
        return FiniteSet(*self.args)

    @property
    def values(self):
        return sumsets(space.values for space in self.spaces)

    def compute_expectation(self, expr, rvs=None, evaluate=False, **kwargs):
        rvs = rvs or self.values
        rvs = frozenset(rvs)
        for space in self.spaces:
            expr = space.compute_expectation(expr, rvs & space.values, evaluate=False, **kwargs)
        if evaluate and hasattr(expr, 'doit'):
            return expr.doit(**kwargs)
        return expr

    @property
    def domain(self):
        return ProductDomain(*[space.domain for space in self.spaces])

    @property
    def density(self):
        raise NotImplementedError("Density not available for ProductSpaces")

    def sample(self, size=(), library='scipy', seed=None):
        return {k: v for space in self.spaces
            for k, v in space.sample(size=size, library=library, seed=seed).items()}


    def probability(self, condition, **kwargs):
        cond_inv = False
        if isinstance(condition, Ne):
            condition = Eq(condition.args[0], condition.args[1])
            cond_inv = True
        elif isinstance(condition, And): # they are independent
            return Mul(*[self.probability(arg) for arg in condition.args])
        elif isinstance(condition, Or): # they are independent
            return Add(*[self.probability(arg) for arg in condition.args])
        expr = condition.lhs - condition.rhs
        rvs = random_symbols(expr)
        dens = self.compute_density(expr)
        if any(pspace(rv).is_Continuous for rv in rvs):
            from sympy.stats.crv import SingleContinuousPSpace
            from sympy.stats.crv_types import ContinuousDistributionHandmade
            if expr in self.values:
                # Marginalize all other random symbols out of the density
                randomsymbols = tuple(set(self.values) - frozenset([expr]))
                symbols = tuple(rs.symbol for rs in randomsymbols)
                pdf = self.domain.integrate(self.pdf, symbols, **kwargs)
                return Lambda(expr.symbol, pdf)
            dens = ContinuousDistributionHandmade(dens)
            z = Dummy('z', real=True)
            space = SingleContinuousPSpace(z, dens)
            result = space.probability(condition.__class__(space.value, 0))
        else:
            from sympy.stats.drv import SingleDiscretePSpace
            from sympy.stats.drv_types import DiscreteDistributionHandmade
            dens = DiscreteDistributionHandmade(dens)
            z = Dummy('z', integer=True)
            space = SingleDiscretePSpace(z, dens)
            result = space.probability(condition.__class__(space.value, 0))
        return result if not cond_inv else S.One - result

    def compute_density(self, expr, **kwargs):
        rvs = random_symbols(expr)
        if any(pspace(rv).is_Continuous for rv in rvs):
            z = Dummy('z', real=True)
            expr = self.compute_expectation(DiracDelta(expr - z),
             **kwargs)
        else:
            z = Dummy('z', integer=True)
            expr = self.compute_expectation(KroneckerDelta(expr, z),
             **kwargs)
        return Lambda(z, expr)

    def compute_cdf(self, expr, **kwargs):
        raise ValueError("CDF not well defined on multivariate expressions")

    def conditional_space(self, condition, normalize=True, **kwargs):
        rvs = random_symbols(condition)
        condition = condition.xreplace({rv: rv.symbol for rv in self.values})
        pspaces = [pspace(rv) for rv in rvs]
        if any(ps.is_Continuous for ps in pspaces):
            from sympy.stats.crv import (ConditionalContinuousDomain,
                ContinuousPSpace)
            space = ContinuousPSpace
            domain = ConditionalContinuousDomain(self.domain, condition)
        elif any(ps.is_Discrete for ps in pspaces):
            from sympy.stats.drv import (ConditionalDiscreteDomain,
                DiscretePSpace)
            space = DiscretePSpace
            domain = ConditionalDiscreteDomain(self.domain, condition)
        elif all(ps.is_Finite for ps in pspaces):
            from sympy.stats.frv import FinitePSpace
            return FinitePSpace.conditional_space(self, condition)
        if normalize:
            replacement  = {rv: Dummy(str(rv)) for rv in self.symbols}
            norm = domain.compute_expectation(self.pdf, **kwargs)
            pdf = self.pdf / norm.xreplace(replacement)
            # XXX: Converting symbols from set to tuple. The order matters to
            # Lambda though so we shouldn't be starting with a set here...
            density = Lambda(tuple(domain.symbols), pdf)

        return space(domain, density)

class ProductDomain(RandomDomain):
    """
    A domain resulting from the merger of two independent domains.

    See Also
    ========
    sympy.stats.crv.ProductContinuousDomain
    sympy.stats.frv.ProductFiniteDomain
    """
    is_ProductDomain = True

    def __new__(cls, *domains):
        # Flatten any product of products
        domains2 = []
        for domain in domains:
            if not domain.is_ProductDomain:
                domains2.append(domain)
            else:
                domains2.extend(domain.domains)
        domains2 = FiniteSet(*domains2)

        if all(domain.is_Finite for domain in domains2):
            from sympy.stats.frv import ProductFiniteDomain
            cls = ProductFiniteDomain
        if all(domain.is_Continuous for domain in domains2):
            from sympy.stats.crv import ProductContinuousDomain
            cls = ProductContinuousDomain
        if all(domain.is_Discrete for domain in domains2):
            from sympy.stats.drv import ProductDiscreteDomain
            cls = ProductDiscreteDomain

        return Basic.__new__(cls, *domains2)

    @property
    def sym_domain_dict(self):
        return {symbol: domain for domain in self.domains
                                     for symbol in domain.symbols}

    @property
    def symbols(self):
        return FiniteSet(*[sym for domain in self.domains
                               for sym    in domain.symbols])

    @property
    def domains(self):
        return self.args

    @property
    def set(self):
        return ProductSet(*(domain.set for domain in self.domains))

    def __contains__(self, other):
        # Split event into each subdomain
        for domain in self.domains:
            # Collect the parts of this event which associate to this domain
            elem = frozenset([item for item in other
                              if sympify(domain.symbols.contains(item[0]))
                              is S.true])
            # Test this sub-event
            if elem not in domain:
                return False
        # All subevents passed
        return True

    def as_boolean(self):
        return And(*[domain.as_boolean() for domain in self.domains])


def random_symbols(expr):
    """
    Returns all RandomSymbols within a SymPy Expression.
    """
    atoms = getattr(expr, 'atoms', None)
    if atoms is not None:
        comp = lambda rv: rv.symbol.name
        l = list(atoms(RandomSymbol))
        return sorted(l, key=comp)
    else:
        return []


def pspace(expr):
    """
    Returns the underlying Probability Space of a random expression.

    For internal use.

    Examples
    ========

    >>> from sympy.stats import pspace, Normal
    >>> X = Normal('X', 0, 1)
    >>> pspace(2*X + 1) == X.pspace
    True
    """
    expr = sympify(expr)
    if isinstance(expr, RandomSymbol) and expr.pspace is not None:
        return expr.pspace
    if expr.has(RandomMatrixSymbol):
        rm = list(expr.atoms(RandomMatrixSymbol))[0]
        return rm.pspace

    rvs = random_symbols(expr)
    if not rvs:
        raise ValueError("Expression containing Random Variable expected, not %s" % (expr))
    # If only one space present
    if all(rv.pspace == rvs[0].pspace for rv in rvs):
        return rvs[0].pspace
    from sympy.stats.compound_rv import CompoundPSpace
    from sympy.stats.stochastic_process import StochasticPSpace
    for rv in rvs:
        if isinstance(rv.pspace, (CompoundPSpace, StochasticPSpace)):
            return rv.pspace
    # Otherwise make a product space
    return IndependentProductPSpace(*[rv.pspace for rv in rvs])


def sumsets(sets):
    """
    Union of sets
    """
    return frozenset().union(*sets)


def rs_swap(a, b):
    """
    Build a dictionary to swap RandomSymbols based on their underlying symbol.

    i.e.
    if    ``X = ('x', pspace1)``
    and   ``Y = ('x', pspace2)``
    then ``X`` and ``Y`` match and the key, value pair
    ``{X:Y}`` will appear in the result

    Inputs: collections a and b of random variables which share common symbols
    Output: dict mapping RVs in a to RVs in b
    """
    d = {}
    for rsa in a:
        d[rsa] = [rsb for rsb in b if rsa.symbol == rsb.symbol][0]
    return d


def given(expr, condition=None, **kwargs):
    r""" Conditional Random Expression.

    Explanation
    ===========

    From a random expression and a condition on that expression creates a new
    probability space from the condition and returns the same expression on that
    conditional probability space.

    Examples
    ========

    >>> from sympy.stats import given, density, Die
    >>> X = Die('X', 6)
    >>> Y = given(X, X > 3)
    >>> density(Y).dict
    {4: 1/3, 5: 1/3, 6: 1/3}

    Following convention, if the condition is a random symbol then that symbol
    is considered fixed.

    >>> from sympy.stats import Normal
    >>> from sympy import pprint
    >>> from sympy.abc import z

    >>> X = Normal('X', 0, 1)
    >>> Y = Normal('Y', 0, 1)
    >>> pprint(density(X + Y, Y)(z), use_unicode=False)
                    2
           -(-Y + z)
           -----------
      ___       2
    \/ 2 *e
    ------------------
             ____
         2*\/ pi
    """

    if not is_random(condition) or pspace_independent(expr, condition):
        return expr

    if isinstance(condition, RandomSymbol):
        condition = Eq(condition, condition.symbol)

    condsymbols = random_symbols(condition)
    if (isinstance(condition, Eq) and len(condsymbols) == 1 and
        not isinstance(pspace(expr).domain, ConditionalDomain)):
        rv = tuple(condsymbols)[0]

        results = solveset(condition, rv)
        if isinstance(results, Intersection) and S.Reals in results.args:
            results = list(results.args[1])

        sums = 0
        for res in results:
            temp = expr.subs(rv, res)
            if temp == True:
                return True
            if temp != False:
                # XXX: This seems nonsensical but preserves existing behaviour
                # after the change that Relational is no longer a subclass of
                # Expr. Here expr is sometimes Relational and sometimes Expr
                # but we are trying to add them with +=. This needs to be
                # fixed somehow.
                if sums == 0 and isinstance(expr, Relational):
                    sums = expr.subs(rv, res)
                else:
                    sums += expr.subs(rv, res)
        if sums == 0:
            return False
        return sums

    # Get full probability space of both the expression and the condition
    fullspace = pspace(Tuple(expr, condition))
    # Build new space given the condition
    space = fullspace.conditional_space(condition, **kwargs)
    # Dictionary to swap out RandomSymbols in expr with new RandomSymbols
    # That point to the new conditional space
    swapdict = rs_swap(fullspace.values, space.values)
    # Swap random variables in the expression
    expr = expr.xreplace(swapdict)
    return expr


def expectation(expr, condition=None, numsamples=None, evaluate=True, **kwargs):
    """
    Returns the expected value of a random expression.

    Parameters
    ==========

    expr : Expr containing RandomSymbols
        The expression of which you want to compute the expectation value
    given : Expr containing RandomSymbols
        A conditional expression. E(X, X>0) is expectation of X given X > 0
    numsamples : int
        Enables sampling and approximates the expectation with this many samples
    evalf : Bool (defaults to True)
        If sampling return a number rather than a complex expression
    evaluate : Bool (defaults to True)
        In case of continuous systems return unevaluated integral

    Examples
    ========

    >>> from sympy.stats import E, Die
    >>> X = Die('X', 6)
    >>> E(X)
    7/2
    >>> E(2*X + 1)
    8

    >>> E(X, X > 3) # Expectation of X given that it is above 3
    5
    """

    if not is_random(expr):  # expr isn't random?
        return expr
    kwargs['numsamples'] = numsamples
    from sympy.stats.symbolic_probability import Expectation
    if evaluate:
        return Expectation(expr, condition).doit(**kwargs)
    return Expectation(expr, condition)


def probability(condition, given_condition=None, numsamples=None,
                evaluate=True, **kwargs):
    """
    Probability that a condition is true, optionally given a second condition.

    Parameters
    ==========

    condition : Combination of Relationals containing RandomSymbols
        The condition of which you want to compute the probability
    given_condition : Combination of Relationals containing RandomSymbols
        A conditional expression. P(X > 1, X > 0) is expectation of X > 1
        given X > 0
    numsamples : int
        Enables sampling and approximates the probability with this many samples
    evaluate : Bool (defaults to True)
        In case of continuous systems return unevaluated integral

    Examples
    ========

    >>> from sympy.stats import P, Die
    >>> from sympy import Eq
    >>> X, Y = Die('X', 6), Die('Y', 6)
    >>> P(X > 3)
    1/2
    >>> P(Eq(X, 5), X > 2) # Probability that X == 5 given that X > 2
    1/4
    >>> P(X > Y)
    5/12
    """

    kwargs['numsamples'] = numsamples
    from sympy.stats.symbolic_probability import Probability
    if evaluate:
        return Probability(condition, given_condition).doit(**kwargs)
    return Probability(condition, given_condition)


class Density(Basic):
    expr = property(lambda self: self.args[0])

    def __new__(cls, expr, condition = None):
        expr = _sympify(expr)
        if condition is None:
            obj = Basic.__new__(cls, expr)
        else:
            condition = _sympify(condition)
            obj = Basic.__new__(cls, expr, condition)
        return obj

    @property
    def condition(self):
        if len(self.args) > 1:
            return self.args[1]
        else:
            return None

    def doit(self, evaluate=True, **kwargs):
        from sympy.stats.random_matrix import RandomMatrixPSpace
        from sympy.stats.joint_rv import JointPSpace
        from sympy.stats.matrix_distributions import MatrixPSpace
        from sympy.stats.compound_rv import CompoundPSpace
        from sympy.stats.frv import SingleFiniteDistribution
        expr, condition = self.expr, self.condition

        if isinstance(expr, SingleFiniteDistribution):
            return expr.dict
        if condition is not None:
            # Recompute on new conditional expr
            expr = given(expr, condition, **kwargs)
        if not random_symbols(expr):
            return Lambda(x, DiracDelta(x - expr))
        if isinstance(expr, RandomSymbol):
            if isinstance(expr.pspace, (SinglePSpace, JointPSpace, MatrixPSpace)) and \
                hasattr(expr.pspace, 'distribution'):
                return expr.pspace.distribution
            elif isinstance(expr.pspace, RandomMatrixPSpace):
                return expr.pspace.model
        if isinstance(pspace(expr), CompoundPSpace):
            kwargs['compound_evaluate'] = evaluate
        result = pspace(expr).compute_density(expr, **kwargs)

        if evaluate and hasattr(result, 'doit'):
            return result.doit()
        else:
            return result


def density(expr, condition=None, evaluate=True, numsamples=None, **kwargs):
    """
    Probability density of a random expression, optionally given a second
    condition.

    Explanation
    ===========

    This density will take on different forms for different types of
    probability spaces. Discrete variables produce Dicts. Continuous
    variables produce Lambdas.

    Parameters
    ==========

    expr : Expr containing RandomSymbols
        The expression of which you want to compute the density value
    condition : Relational containing RandomSymbols
        A conditional expression. density(X > 1, X > 0) is density of X > 1
        given X > 0
    numsamples : int
        Enables sampling and approximates the density with this many samples

    Examples
    ========

    >>> from sympy.stats import density, Die, Normal
    >>> from sympy import Symbol

    >>> x = Symbol('x')
    >>> D = Die('D', 6)
    >>> X = Normal(x, 0, 1)

    >>> density(D).dict
    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
    >>> density(2*D).dict
    {2: 1/6, 4: 1/6, 6: 1/6, 8: 1/6, 10: 1/6, 12: 1/6}
    >>> density(X)(x)
    sqrt(2)*exp(-x**2/2)/(2*sqrt(pi))
    """

    if numsamples:
        return sampling_density(expr, condition, numsamples=numsamples,
                **kwargs)

    return Density(expr, condition).doit(evaluate=evaluate, **kwargs)


def cdf(expr, condition=None, evaluate=True, **kwargs):
    """
    Cumulative Distribution Function of a random expression.

    optionally given a second condition.

    Explanation
    ===========

    This density will take on different forms for different types of
    probability spaces.
    Discrete variables produce Dicts.
    Continuous variables produce Lambdas.

    Examples
    ========

    >>> from sympy.stats import density, Die, Normal, cdf

    >>> D = Die('D', 6)
    >>> X = Normal('X', 0, 1)

    >>> density(D).dict
    {1: 1/6, 2: 1/6, 3: 1/6, 4: 1/6, 5: 1/6, 6: 1/6}
    >>> cdf(D)
    {1: 1/6, 2: 1/3, 3: 1/2, 4: 2/3, 5: 5/6, 6: 1}
    >>> cdf(3*D, D > 2)
    {9: 1/4, 12: 1/2, 15: 3/4, 18: 1}

    >>> cdf(X)
    Lambda(_z, erf(sqrt(2)*_z/2)/2 + 1/2)
    """
    if condition is not None:  # If there is a condition
        # Recompute on new conditional expr
        return cdf(given(expr, condition, **kwargs), **kwargs)

    # Otherwise pass work off to the ProbabilitySpace
    result = pspace(expr).compute_cdf(expr, **kwargs)

    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result


def characteristic_function(expr, condition=None, evaluate=True, **kwargs):
    """
    Characteristic function of a random expression, optionally given a second condition.

    Returns a Lambda.

    Examples
    ========

    >>> from sympy.stats import Normal, DiscreteUniform, Poisson, characteristic_function

    >>> X = Normal('X', 0, 1)
    >>> characteristic_function(X)
    Lambda(_t, exp(-_t**2/2))

    >>> Y = DiscreteUniform('Y', [1, 2, 7])
    >>> characteristic_function(Y)
    Lambda(_t, exp(7*_t*I)/3 + exp(2*_t*I)/3 + exp(_t*I)/3)

    >>> Z = Poisson('Z', 2)
    >>> characteristic_function(Z)
    Lambda(_t, exp(2*exp(_t*I) - 2))
    """
    if condition is not None:
        return characteristic_function(given(expr, condition, **kwargs), **kwargs)

    result = pspace(expr).compute_characteristic_function(expr, **kwargs)

    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result

def moment_generating_function(expr, condition=None, evaluate=True, **kwargs):
    if condition is not None:
        return moment_generating_function(given(expr, condition, **kwargs), **kwargs)

    result = pspace(expr).compute_moment_generating_function(expr, **kwargs)

    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result

def where(condition, given_condition=None, **kwargs):
    """
    Returns the domain where a condition is True.

    Examples
    ========

    >>> from sympy.stats import where, Die, Normal
    >>> from sympy import And

    >>> D1, D2 = Die('a', 6), Die('b', 6)
    >>> a, b = D1.symbol, D2.symbol
    >>> X = Normal('x', 0, 1)

    >>> where(X**2<1)
    Domain: (-1 < x) & (x < 1)

    >>> where(X**2<1).set
    Interval.open(-1, 1)

    >>> where(And(D1<=D2, D2<3))
    Domain: (Eq(a, 1) & Eq(b, 1)) | (Eq(a, 1) & Eq(b, 2)) | (Eq(a, 2) & Eq(b, 2))
    """
    if given_condition is not None:  # If there is a condition
        # Recompute on new conditional expr
        return where(given(condition, given_condition, **kwargs), **kwargs)

    # Otherwise pass work off to the ProbabilitySpace
    return pspace(condition).where(condition, **kwargs)


@doctest_depends_on(modules=('scipy',))
def sample(expr, condition=None, size=(), library='scipy',
           numsamples=1, seed=None, **kwargs):
    """
    A realization of the random expression.

    Parameters
    ==========

    expr : Expression of random variables
        Expression from which sample is extracted
    condition : Expr containing RandomSymbols
        A conditional expression
    size : int, tuple
        Represents size of each sample in numsamples
    library : str
        - 'scipy' : Sample using scipy
        - 'numpy' : Sample using numpy
        - 'pymc'  : Sample using PyMC

        Choose any of the available options to sample from as string,
        by default is 'scipy'
    numsamples : int
        Number of samples, each with size as ``size``.

        .. deprecated:: 1.9

        The ``numsamples`` parameter is deprecated and is only provided for
        compatibility with v1.8. Use a list comprehension or an additional
        dimension in ``size`` instead. See
        :ref:`deprecated-sympy-stats-numsamples` for details.

    seed :
        An object to be used as seed by the given external library for sampling `expr`.
        Following is the list of possible types of object for the supported libraries,

        - 'scipy': int, numpy.random.RandomState, numpy.random.Generator
        - 'numpy': int, numpy.random.RandomState, numpy.random.Generator
        - 'pymc': int

        Optional, by default None, in which case seed settings
        related to the given library will be used.
        No modifications to environment's global seed settings
        are done by this argument.

    Returns
    =======

    sample: float/list/numpy.ndarray
        one sample or a collection of samples of the random expression.

        - sample(X) returns float/numpy.float64/numpy.int64 object.
        - sample(X, size=int/tuple) returns numpy.ndarray object.

    Examples
    ========

    >>> from sympy.stats import Die, sample, Normal, Geometric
    >>> X, Y, Z = Die('X', 6), Die('Y', 6), Die('Z', 6) # Finite Random Variable
    >>> die_roll = sample(X + Y + Z)
    >>> die_roll # doctest: +SKIP
    3
    >>> N = Normal('N', 3, 4) # Continuous Random Variable
    >>> samp = sample(N)
    >>> samp in N.pspace.domain.set
    True
    >>> samp = sample(N, N>0)
    >>> samp > 0
    True
    >>> samp_list = sample(N, size=4)
    >>> [sam in N.pspace.domain.set for sam in samp_list]
    [True, True, True, True]
    >>> sample(N, size = (2,3)) # doctest: +SKIP
    array([[5.42519758, 6.40207856, 4.94991743],
       [1.85819627, 6.83403519, 1.9412172 ]])
    >>> G = Geometric('G', 0.5) # Discrete Random Variable
    >>> samp_list = sample(G, size=3)
    >>> samp_list # doctest: +SKIP
    [1, 3, 2]
    >>> [sam in G.pspace.domain.set for sam in samp_list]
    [True, True, True]
    >>> MN = Normal("MN", [3, 4], [[2, 1], [1, 2]]) # Joint Random Variable
    >>> samp_list = sample(MN, size=4)
    >>> samp_list # doctest: +SKIP
    [array([2.85768055, 3.38954165]),
     array([4.11163337, 4.3176591 ]),
     array([0.79115232, 1.63232916]),
     array([4.01747268, 3.96716083])]
    >>> [tuple(sam) in MN.pspace.domain.set for sam in samp_list]
    [True, True, True, True]

    .. versionchanged:: 1.7.0
        sample used to return an iterator containing the samples instead of value.

    .. versionchanged:: 1.9.0
        sample returns values or array of values instead of an iterator and numsamples is deprecated.

    """

    iterator = sample_iter(expr, condition, size=size, library=library,
                                                        numsamples=numsamples, seed=seed)

    if numsamples != 1:
        sympy_deprecation_warning(
            f"""
            The numsamples parameter to sympy.stats.sample() is deprecated.
            Either use a list comprehension, like

            [sample(...) for i in range({numsamples})]

            or add a dimension to size, like

            sample(..., size={(numsamples,) + size})
            """,
            deprecated_since_version="1.9",
            active_deprecations_target="deprecated-sympy-stats-numsamples",
        )
        return [next(iterator) for i in range(numsamples)]

    return next(iterator)


def quantile(expr, evaluate=True, **kwargs):
    r"""
    Return the :math:`p^{th}` order quantile of a probability distribution.

    Explanation
    ===========

    Quantile is defined as the value at which the probability of the random
    variable is less than or equal to the given probability.

    .. math::
        Q(p) = \inf\{x \in (-\infty, \infty) : p \le F(x)\}

    Examples
    ========

    >>> from sympy.stats import quantile, Die, Exponential
    >>> from sympy import Symbol, pprint
    >>> p = Symbol("p")

    >>> l = Symbol("lambda", positive=True)
    >>> X = Exponential("x", l)
    >>> quantile(X)(p)
    -log(1 - p)/lambda

    >>> D = Die("d", 6)
    >>> pprint(quantile(D)(p), use_unicode=False)
    /nan  for Or(p > 1, p < 0)
    |
    | 1       for p <= 1/6
    |
    | 2       for p <= 1/3
    |
    < 3       for p <= 1/2
    |
    | 4       for p <= 2/3
    |
    | 5       for p <= 5/6
    |
    \ 6        for p <= 1

    """
    result = pspace(expr).compute_quantile(expr, **kwargs)

    if evaluate and hasattr(result, 'doit'):
        return result.doit()
    else:
        return result

def sample_iter(expr, condition=None, size=(), library='scipy',
                    numsamples=S.Infinity, seed=None, **kwargs):

    """
    Returns an iterator of realizations from the expression given a condition.

    Parameters
    ==========

    expr: Expr
        Random expression to be realized
    condition: Expr, optional
        A conditional expression
    size : int, tuple
        Represents size of each sample in numsamples
    numsamples: integer, optional
        Length of the iterator (defaults to infinity)
    seed :
        An object to be used as seed by the given external library for sampling `expr`.
        Following is the list of possible types of object for the supported libraries,

        - 'scipy': int, numpy.random.RandomState, numpy.random.Generator
        - 'numpy': int, numpy.random.RandomState, numpy.random.Generator
        - 'pymc': int

        Optional, by default None, in which case seed settings
        related to the given library will be used.
        No modifications to environment's global seed settings
        are done by this argument.

    Examples
    ========

    >>> from sympy.stats import Normal, sample_iter
    >>> X = Normal('X', 0, 1)
    >>> expr = X*X + 3
    >>> iterator = sample_iter(expr, numsamples=3) # doctest: +SKIP
    >>> list(iterator) # doctest: +SKIP
    [12, 4, 7]

    Returns
    =======

    sample_iter: iterator object
        iterator object containing the sample/samples of given expr

    See Also
    ========

    sample
    sampling_P
    sampling_E

    """
    from sympy.stats.joint_rv import JointRandomSymbol
    if not import_module(library):
        raise ValueError("Failed to import %s" % library)

    if condition is not None:
        ps = pspace(Tuple(expr, condition))
    else:
        ps = pspace(expr)

    rvs = list(ps.values)
    if isinstance(expr, JointRandomSymbol):
        expr = expr.subs({expr: RandomSymbol(expr.symbol, expr.pspace)})
    else:
        sub = {}
        for arg in expr.args:
            if isinstance(arg, JointRandomSymbol):
                sub[arg] = RandomSymbol(arg.symbol, arg.pspace)
        expr = expr.subs(sub)

    def fn_subs(*args):
        return expr.subs(dict(zip(rvs, args)))

    def given_fn_subs(*args):
        if condition is not None:
            return condition.subs(dict(zip(rvs, args)))
        return False

    if library in ('pymc', 'pymc3'):
        # Currently unable to lambdify in pymc
        # TODO : Remove when lambdify accepts 'pymc' as module
        fn = lambdify(rvs, expr, **kwargs)
    else:
        fn = lambdify(rvs, expr, modules=library, **kwargs)


    if condition is not None:
        given_fn = lambdify(rvs, condition, **kwargs)

    def return_generator_infinite():
        count = 0
        _size = (1,)+((size,) if isinstance(size, int) else size)
        while count < numsamples:
            d = ps.sample(size=_size, library=library, seed=seed)  # a dictionary that maps RVs to values
            args = [d[rv][0] for rv in rvs]

            if condition is not None:  # Check that these values satisfy the condition
                # TODO: Replace the try-except block with only given_fn(*args)
                # once lambdify works with unevaluated SymPy objects.
                try:
                    gd = given_fn(*args)
                except (NameError, TypeError):
                    gd = given_fn_subs(*args)
                if gd != True and gd != False:
                    raise ValueError(
                        "Conditions must not contain free symbols")
                if not gd:  # If the values don't satisfy then try again
                    continue

            yield fn(*args)
            count += 1

    def return_generator_finite():
        faulty = True
        while faulty:
            d = ps.sample(size=(numsamples,) + ((size,) if isinstance(size, int) else size),
                          library=library, seed=seed) # a dictionary that maps RVs to values

            faulty = False
            count = 0
            while count < numsamples and not faulty:
                args = [d[rv][count] for rv in rvs]
                if condition is not None:  # Check that these values satisfy the condition
                    # TODO: Replace the try-except block with only given_fn(*args)
                    # once lambdify works with unevaluated SymPy objects.
                    try:
                        gd = given_fn(*args)
                    except (NameError, TypeError):
                        gd = given_fn_subs(*args)
                    if gd != True and gd != False:
                        raise ValueError(
                            "Conditions must not contain free symbols")
                    if not gd:  # If the values don't satisfy then try again
                        faulty = True

                count += 1

        count = 0
        while count < numsamples:
            args = [d[rv][count] for rv in rvs]
            # TODO: Replace the try-except block with only fn(*args)
            # once lambdify works with unevaluated SymPy objects.
            try:
                yield fn(*args)
            except (NameError, TypeError):
                yield fn_subs(*args)
            count += 1

    if numsamples is S.Infinity:
        return return_generator_infinite()

    return return_generator_finite()

def sample_iter_lambdify(expr, condition=None, size=(),
                         numsamples=S.Infinity, seed=None, **kwargs):

    return sample_iter(expr, condition=condition, size=size,
                       numsamples=numsamples, seed=seed, **kwargs)

def sample_iter_subs(expr, condition=None, size=(),
                     numsamples=S.Infinity, seed=None, **kwargs):

    return sample_iter(expr, condition=condition, size=size,
                       numsamples=numsamples, seed=seed, **kwargs)


def sampling_P(condition, given_condition=None, library='scipy', numsamples=1,
               evalf=True, seed=None, **kwargs):
    """
    Sampling version of P.

    See Also
    ========

    P
    sampling_E
    sampling_density

    """

    count_true = 0
    count_false = 0
    samples = sample_iter(condition, given_condition, library=library,
                          numsamples=numsamples, seed=seed, **kwargs)

    for sample in samples:
        if sample:
            count_true += 1
        else:
            count_false += 1

    result = S(count_true) / numsamples
    if evalf:
        return result.evalf()
    else:
        return result


def sampling_E(expr, given_condition=None, library='scipy', numsamples=1,
               evalf=True, seed=None, **kwargs):
    """
    Sampling version of E.

    See Also
    ========

    P
    sampling_P
    sampling_density
    """
    samples = list(sample_iter(expr, given_condition, library=library,
                          numsamples=numsamples, seed=seed, **kwargs))
    result = Add(*samples) / numsamples

    if evalf:
        return result.evalf()
    else:
        return result

def sampling_density(expr, given_condition=None, library='scipy',
                    numsamples=1, seed=None, **kwargs):
    """
    Sampling version of density.

    See Also
    ========
    density
    sampling_P
    sampling_E
    """

    results = {}
    for result in sample_iter(expr, given_condition, library=library,
                              numsamples=numsamples, seed=seed, **kwargs):
        results[result] = results.get(result, 0) + 1

    return results


def dependent(a, b):
    """
    Dependence of two random expressions.

    Two expressions are independent if knowledge of one does not change
    computations on the other.

    Examples
    ========

    >>> from sympy.stats import Normal, dependent, given
    >>> from sympy import Tuple, Eq

    >>> X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    >>> dependent(X, Y)
    False
    >>> dependent(2*X + Y, -Y)
    True
    >>> X, Y = given(Tuple(X, Y), Eq(X + Y, 3))
    >>> dependent(X, Y)
    True

    See Also
    ========

    independent
    """
    if pspace_independent(a, b):
        return False

    z = Symbol('z', real=True)
    # Dependent if density is unchanged when one is given information about
    # the other
    return (density(a, Eq(b, z)) != density(a) or
            density(b, Eq(a, z)) != density(b))


def independent(a, b):
    """
    Independence of two random expressions.

    Two expressions are independent if knowledge of one does not change
    computations on the other.

    Examples
    ========

    >>> from sympy.stats import Normal, independent, given
    >>> from sympy import Tuple, Eq

    >>> X, Y = Normal('X', 0, 1), Normal('Y', 0, 1)
    >>> independent(X, Y)
    True
    >>> independent(2*X + Y, -Y)
    False
    >>> X, Y = given(Tuple(X, Y), Eq(X + Y, 3))
    >>> independent(X, Y)
    False

    See Also
    ========

    dependent
    """
    return not dependent(a, b)


def pspace_independent(a, b):
    """
    Tests for independence between a and b by checking if their PSpaces have
    overlapping symbols. This is a sufficient but not necessary condition for
    independence and is intended to be used internally.

    Notes
    =====

    pspace_independent(a, b) implies independent(a, b)
    independent(a, b) does not imply pspace_independent(a, b)
    """
    a_symbols = set(pspace(b).symbols)
    b_symbols = set(pspace(a).symbols)

    if len(set(random_symbols(a)).intersection(random_symbols(b))) != 0:
        return False

    if len(a_symbols.intersection(b_symbols)) == 0:
        return True
    return None


def rv_subs(expr, symbols=None):
    """
    Given a random expression replace all random variables with their symbols.

    If symbols keyword is given restrict the swap to only the symbols listed.
    """
    if symbols is None:
        symbols = random_symbols(expr)
    if not symbols:
        return expr
    swapdict = {rv: rv.symbol for rv in symbols}
    return expr.subs(swapdict)


class NamedArgsMixin:
    _argnames: tuple[str, ...] = ()

    def __getattr__(self, attr):
        try:
            return self.args[self._argnames.index(attr)]
        except ValueError:
            raise AttributeError("'%s' object has no attribute '%s'" % (
                type(self).__name__, attr))


class Distribution(Basic):

    def sample(self, size=(), library='scipy', seed=None):
        """ A random realization from the distribution """

        module = import_module(library)
        if library in {'scipy', 'numpy', 'pymc3', 'pymc'} and module is None:
            raise ValueError("Failed to import %s" % library)

        if library == 'scipy':
            # scipy does not require map as it can handle using custom distributions.
            # However, we will still use a map where we can.

            # TODO: do this for drv.py and frv.py if necessary.
            # TODO: add more distributions here if there are more
            # See links below referring to sections beginning with "A common parametrization..."
            # I will remove all these comments if everything is ok.

            from sympy.stats.sampling.sample_scipy import do_sample_scipy
            import numpy
            if seed is None or isinstance(seed, int):
                rand_state = numpy.random.default_rng(seed=seed)
            else:
                rand_state = seed
            samps = do_sample_scipy(self, size, rand_state)

        elif library == 'numpy':
            from sympy.stats.sampling.sample_numpy import do_sample_numpy
            import numpy
            if seed is None or isinstance(seed, int):
                rand_state = numpy.random.default_rng(seed=seed)
            else:
                rand_state = seed
            _size = None if size == () else size
            samps = do_sample_numpy(self, _size, rand_state)
        elif library in ('pymc', 'pymc3'):
            from sympy.stats.sampling.sample_pymc import do_sample_pymc
            import logging
            logging.getLogger("pymc").setLevel(logging.ERROR)
            try:
                import pymc
            except ImportError:
                import pymc3 as pymc

            with pymc.Model():
                if do_sample_pymc(self) is not None:
                    samps = pymc.sample(draws=prod(size), chains=1, compute_convergence_checks=False,
                            progressbar=False, random_seed=seed, return_inferencedata=False)[:]['X']
                    samps = samps.reshape(size)
                else:
                    samps = None

        else:
            raise NotImplementedError("Sampling from %s is not supported yet."
                                      % str(library))

        if samps is not None:
            return samps
        raise NotImplementedError(
            "Sampling for %s is not currently implemented from %s"
            % (self, library))


def _value_check(condition, message):
    """
    Raise a ValueError with message if condition is False, else
    return True if all conditions were True, else False.

    Examples
    ========

    >>> from sympy.stats.rv import _value_check
    >>> from sympy.abc import a, b, c
    >>> from sympy import And, Dummy

    >>> _value_check(2 < 3, '')
    True

    Here, the condition is not False, but it does not evaluate to True
    so False is returned (but no error is raised). So checking if the
    return value is True or False will tell you if all conditions were
    evaluated.

    >>> _value_check(a < b, '')
    False

    In this case the condition is False so an error is raised:

    >>> r = Dummy(real=True)
    >>> _value_check(r < r - 1, 'condition is not true')
    Traceback (most recent call last):
    ...
    ValueError: condition is not true

    If no condition of many conditions must be False, they can be
    checked by passing them as an iterable:

    >>> _value_check((a < 0, b < 0, c < 0), '')
    False

    The iterable can be a generator, too:

    >>> _value_check((i < 0 for i in (a, b, c)), '')
    False

    The following are equivalent to the above but do not pass
    an iterable:

    >>> all(_value_check(i < 0, '') for i in (a, b, c))
    False
    >>> _value_check(And(a < 0, b < 0, c < 0), '')
    False
    """
    if not iterable(condition):
        condition = [condition]
    truth = fuzzy_and(condition)
    if truth == False:
        raise ValueError(message)
    return truth == True

def _symbol_converter(sym):
    """
    Casts the parameter to Symbol if it is 'str'
    otherwise no operation is performed on it.

    Parameters
    ==========

    sym
        The parameter to be converted.

    Returns
    =======

    Symbol
        the parameter converted to Symbol.

    Raises
    ======

    TypeError
        If the parameter is not an instance of both str and
        Symbol.

    Examples
    ========

    >>> from sympy import Symbol
    >>> from sympy.stats.rv import _symbol_converter
    >>> s = _symbol_converter('s')
    >>> isinstance(s, Symbol)
    True
    >>> _symbol_converter(1)
    Traceback (most recent call last):
    ...
    TypeError: 1 is neither a Symbol nor a string
    >>> r = Symbol('r')
    >>> isinstance(r, Symbol)
    True
    """
    if isinstance(sym, str):
        sym = Symbol(sym)
    if not isinstance(sym, Symbol):
        raise TypeError("%s is neither a Symbol nor a string"%(sym))
    return sym

def sample_stochastic_process(process):
    """
    This function is used to sample from stochastic process.

    Parameters
    ==========

    process: StochasticProcess
        Process used to extract the samples. It must be an instance of
        StochasticProcess

    Examples
    ========

    >>> from sympy.stats import sample_stochastic_process, DiscreteMarkovChain
    >>> from sympy import Matrix
    >>> T = Matrix([[0.5, 0.2, 0.3],[0.2, 0.5, 0.3],[0.2, 0.3, 0.5]])
    >>> Y = DiscreteMarkovChain("Y", [0, 1, 2], T)
    >>> next(sample_stochastic_process(Y)) in Y.state_space
    True
    >>> next(sample_stochastic_process(Y))  # doctest: +SKIP
    0
    >>> next(sample_stochastic_process(Y)) # doctest: +SKIP
    2

    Returns
    =======

    sample: iterator object
        iterator object containing the sample of given process

    """
    from sympy.stats.stochastic_process_types import StochasticProcess
    if not isinstance(process, StochasticProcess):
        raise ValueError("Process must be an instance of Stochastic Process")
    return process.sample()