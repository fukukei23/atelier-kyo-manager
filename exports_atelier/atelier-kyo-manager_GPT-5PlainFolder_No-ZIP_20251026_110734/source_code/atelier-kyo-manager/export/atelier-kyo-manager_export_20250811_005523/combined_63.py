
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\persistence.py ===
# orm/persistence.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""private module containing functions used to emit INSERT, UPDATE
and DELETE statements on behalf of a :class:`_orm.Mapper` and its descending
mappers.

The functions here are called only by the unit of work functions
in unitofwork.py.

"""
from __future__ import annotations

from itertools import chain
from itertools import groupby
from itertools import zip_longest
import operator

from . import attributes
from . import exc as orm_exc
from . import loading
from . import sync
from .base import state_str
from .. import exc as sa_exc
from .. import future
from .. import sql
from .. import util
from ..engine import cursor as _cursor
from ..sql import operators
from ..sql.elements import BooleanClauseList
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL


def save_obj(base_mapper, states, uowtransaction, single=False):
    """Issue ``INSERT`` and/or ``UPDATE`` statements for a list
    of objects.

    This is called within the context of a UOWTransaction during a
    flush operation, given a list of states to be flushed.  The
    base mapper in an inheritance hierarchy handles the inserts/
    updates for all descendant mappers.

    """

    # if batch=false, call _save_obj separately for each object
    if not single and not base_mapper.batch:
        for state in _sort_states(base_mapper, states):
            save_obj(base_mapper, [state], uowtransaction, single=True)
        return

    states_to_update = []
    states_to_insert = []

    for (
        state,
        dict_,
        mapper,
        connection,
        has_identity,
        row_switch,
        update_version_id,
    ) in _organize_states_for_save(base_mapper, states, uowtransaction):
        if has_identity or row_switch:
            states_to_update.append(
                (state, dict_, mapper, connection, update_version_id)
            )
        else:
            states_to_insert.append((state, dict_, mapper, connection))

    for table, mapper in base_mapper._sorted_tables.items():
        if table not in mapper._pks_by_table:
            continue
        insert = _collect_insert_commands(table, states_to_insert)

        update = _collect_update_commands(
            uowtransaction, table, states_to_update
        )

        _emit_update_statements(
            base_mapper,
            uowtransaction,
            mapper,
            table,
            update,
        )

        _emit_insert_statements(
            base_mapper,
            uowtransaction,
            mapper,
            table,
            insert,
        )

    _finalize_insert_update_commands(
        base_mapper,
        uowtransaction,
        chain(
            (
                (state, state_dict, mapper, connection, False)
                for (state, state_dict, mapper, connection) in states_to_insert
            ),
            (
                (state, state_dict, mapper, connection, True)
                for (
                    state,
                    state_dict,
                    mapper,
                    connection,
                    update_version_id,
                ) in states_to_update
            ),
        ),
    )


def post_update(base_mapper, states, uowtransaction, post_update_cols):
    """Issue UPDATE statements on behalf of a relationship() which
    specifies post_update.

    """

    states_to_update = list(
        _organize_states_for_post_update(base_mapper, states, uowtransaction)
    )

    for table, mapper in base_mapper._sorted_tables.items():
        if table not in mapper._pks_by_table:
            continue

        update = (
            (
                state,
                state_dict,
                sub_mapper,
                connection,
                (
                    mapper._get_committed_state_attr_by_column(
                        state, state_dict, mapper.version_id_col
                    )
                    if mapper.version_id_col is not None
                    else None
                ),
            )
            for state, state_dict, sub_mapper, connection in states_to_update
            if table in sub_mapper._pks_by_table
        )

        update = _collect_post_update_commands(
            base_mapper, uowtransaction, table, update, post_update_cols
        )

        _emit_post_update_statements(
            base_mapper,
            uowtransaction,
            mapper,
            table,
            update,
        )


def delete_obj(base_mapper, states, uowtransaction):
    """Issue ``DELETE`` statements for a list of objects.

    This is called within the context of a UOWTransaction during a
    flush operation.

    """

    states_to_delete = list(
        _organize_states_for_delete(base_mapper, states, uowtransaction)
    )

    table_to_mapper = base_mapper._sorted_tables

    for table in reversed(list(table_to_mapper.keys())):
        mapper = table_to_mapper[table]
        if table not in mapper._pks_by_table:
            continue
        elif mapper.inherits and mapper.passive_deletes:
            continue

        delete = _collect_delete_commands(
            base_mapper, uowtransaction, table, states_to_delete
        )

        _emit_delete_statements(
            base_mapper,
            uowtransaction,
            mapper,
            table,
            delete,
        )

    for (
        state,
        state_dict,
        mapper,
        connection,
        update_version_id,
    ) in states_to_delete:
        mapper.dispatch.after_delete(mapper, connection, state)


def _organize_states_for_save(base_mapper, states, uowtransaction):
    """Make an initial pass across a set of states for INSERT or
    UPDATE.

    This includes splitting out into distinct lists for
    each, calling before_insert/before_update, obtaining
    key information for each state including its dictionary,
    mapper, the connection to use for the execution per state,
    and the identity flag.

    """

    for state, dict_, mapper, connection in _connections_for_states(
        base_mapper, uowtransaction, states
    ):
        has_identity = bool(state.key)

        instance_key = state.key or mapper._identity_key_from_state(state)

        row_switch = update_version_id = None

        # call before_XXX extensions
        if not has_identity:
            mapper.dispatch.before_insert(mapper, connection, state)
        else:
            mapper.dispatch.before_update(mapper, connection, state)

        if mapper._validate_polymorphic_identity:
            mapper._validate_polymorphic_identity(mapper, state, dict_)

        # detect if we have a "pending" instance (i.e. has
        # no instance_key attached to it), and another instance
        # with the same identity key already exists as persistent.
        # convert to an UPDATE if so.
        if (
            not has_identity
            and instance_key in uowtransaction.session.identity_map
        ):
            instance = uowtransaction.session.identity_map[instance_key]
            existing = attributes.instance_state(instance)

            if not uowtransaction.was_already_deleted(existing):
                if not uowtransaction.is_deleted(existing):
                    util.warn(
                        "New instance %s with identity key %s conflicts "
                        "with persistent instance %s"
                        % (state_str(state), instance_key, state_str(existing))
                    )
                else:
                    base_mapper._log_debug(
                        "detected row switch for identity %s.  "
                        "will update %s, remove %s from "
                        "transaction",
                        instance_key,
                        state_str(state),
                        state_str(existing),
                    )

                    # remove the "delete" flag from the existing element
                    uowtransaction.remove_state_actions(existing)
                    row_switch = existing

        if (has_identity or row_switch) and mapper.version_id_col is not None:
            update_version_id = mapper._get_committed_state_attr_by_column(
                row_switch if row_switch else state,
                row_switch.dict if row_switch else dict_,
                mapper.version_id_col,
            )

        yield (
            state,
            dict_,
            mapper,
            connection,
            has_identity,
            row_switch,
            update_version_id,
        )


def _organize_states_for_post_update(base_mapper, states, uowtransaction):
    """Make an initial pass across a set of states for UPDATE
    corresponding to post_update.

    This includes obtaining key information for each state
    including its dictionary, mapper, the connection to use for
    the execution per state.

    """
    return _connections_for_states(base_mapper, uowtransaction, states)


def _organize_states_for_delete(base_mapper, states, uowtransaction):
    """Make an initial pass across a set of states for DELETE.

    This includes calling out before_delete and obtaining
    key information for each state including its dictionary,
    mapper, the connection to use for the execution per state.

    """
    for state, dict_, mapper, connection in _connections_for_states(
        base_mapper, uowtransaction, states
    ):
        mapper.dispatch.before_delete(mapper, connection, state)

        if mapper.version_id_col is not None:
            update_version_id = mapper._get_committed_state_attr_by_column(
                state, dict_, mapper.version_id_col
            )
        else:
            update_version_id = None

        yield (state, dict_, mapper, connection, update_version_id)


def _collect_insert_commands(
    table,
    states_to_insert,
    *,
    bulk=False,
    return_defaults=False,
    render_nulls=False,
    include_bulk_keys=(),
):
    """Identify sets of values to use in INSERT statements for a
    list of states.

    """
    for state, state_dict, mapper, connection in states_to_insert:
        if table not in mapper._pks_by_table:
            continue

        params = {}
        value_params = {}

        propkey_to_col = mapper._propkey_to_col[table]

        eval_none = mapper._insert_cols_evaluating_none[table]

        for propkey in set(propkey_to_col).intersection(state_dict):
            value = state_dict[propkey]
            col = propkey_to_col[propkey]
            if value is None and col not in eval_none and not render_nulls:
                continue
            elif not bulk and (
                hasattr(value, "__clause_element__")
                or isinstance(value, sql.ClauseElement)
            ):
                value_params[col] = (
                    value.__clause_element__()
                    if hasattr(value, "__clause_element__")
                    else value
                )
            else:
                params[col.key] = value

        if not bulk:
            # for all the columns that have no default and we don't have
            # a value and where "None" is not a special value, add
            # explicit None to the INSERT.   This is a legacy behavior
            # which might be worth removing, as it should not be necessary
            # and also produces confusion, given that "missing" and None
            # now have distinct meanings
            for colkey in (
                mapper._insert_cols_as_none[table]
                .difference(params)
                .difference([c.key for c in value_params])
            ):
                params[colkey] = None

        if not bulk or return_defaults:
            # params are in terms of Column key objects, so
            # compare to pk_keys_by_table
            has_all_pks = mapper._pk_keys_by_table[table].issubset(params)

            if mapper.base_mapper._prefer_eager_defaults(
                connection.dialect, table
            ):
                has_all_defaults = mapper._server_default_col_keys[
                    table
                ].issubset(params)
            else:
                has_all_defaults = True
        else:
            has_all_defaults = has_all_pks = True

        if (
            mapper.version_id_generator is not False
            and mapper.version_id_col is not None
            and mapper.version_id_col in mapper._cols_by_table[table]
        ):
            params[mapper.version_id_col.key] = mapper.version_id_generator(
                None
            )

        if bulk:
            if mapper._set_polymorphic_identity:
                params.setdefault(
                    mapper._polymorphic_attr_key, mapper.polymorphic_identity
                )

            if include_bulk_keys:
                params.update((k, state_dict[k]) for k in include_bulk_keys)

        yield (
            state,
            state_dict,
            params,
            mapper,
            connection,
            value_params,
            has_all_pks,
            has_all_defaults,
        )


def _collect_update_commands(
    uowtransaction,
    table,
    states_to_update,
    *,
    bulk=False,
    use_orm_update_stmt=None,
    include_bulk_keys=(),
):
    """Identify sets of values to use in UPDATE statements for a
    list of states.

    This function works intricately with the history system
    to determine exactly what values should be updated
    as well as how the row should be matched within an UPDATE
    statement.  Includes some tricky scenarios where the primary
    key of an object might have been changed.

    """

    for (
        state,
        state_dict,
        mapper,
        connection,
        update_version_id,
    ) in states_to_update:
        if table not in mapper._pks_by_table:
            continue

        pks = mapper._pks_by_table[table]

        if use_orm_update_stmt is not None:
            # TODO: ordered values, etc
            value_params = use_orm_update_stmt._values
        else:
            value_params = {}

        propkey_to_col = mapper._propkey_to_col[table]

        if bulk:
            # keys here are mapped attribute keys, so
            # look at mapper attribute keys for pk
            params = {
                propkey_to_col[propkey].key: state_dict[propkey]
                for propkey in set(propkey_to_col)
                .intersection(state_dict)
                .difference(mapper._pk_attr_keys_by_table[table])
            }
            has_all_defaults = True
        else:
            params = {}
            for propkey in set(propkey_to_col).intersection(
                state.committed_state
            ):
                value = state_dict[propkey]
                col = propkey_to_col[propkey]

                if hasattr(value, "__clause_element__") or isinstance(
                    value, sql.ClauseElement
                ):
                    value_params[col] = (
                        value.__clause_element__()
                        if hasattr(value, "__clause_element__")
                        else value
                    )
                # guard against values that generate non-__nonzero__
                # objects for __eq__()
                elif (
                    state.manager[propkey].impl.is_equal(
                        value, state.committed_state[propkey]
                    )
                    is not True
                ):
                    params[col.key] = value

            if mapper.base_mapper.eager_defaults is True:
                has_all_defaults = (
                    mapper._server_onupdate_default_col_keys[table]
                ).issubset(params)
            else:
                has_all_defaults = True

        if (
            update_version_id is not None
            and mapper.version_id_col in mapper._cols_by_table[table]
        ):
            if not bulk and not (params or value_params):
                # HACK: check for history in other tables, in case the
                # history is only in a different table than the one
                # where the version_id_col is.  This logic was lost
                # from 0.9 -> 1.0.0 and restored in 1.0.6.
                for prop in mapper._columntoproperty.values():
                    history = state.manager[prop.key].impl.get_history(
                        state, state_dict, attributes.PASSIVE_NO_INITIALIZE
                    )
                    if history.added:
                        break
                else:
                    # no net change, break
                    continue

            col = mapper.version_id_col
            no_params = not params and not value_params
            params[col._label] = update_version_id

            if (
                bulk or col.key not in params
            ) and mapper.version_id_generator is not False:
                val = mapper.version_id_generator(update_version_id)
                params[col.key] = val
            elif mapper.version_id_generator is False and no_params:
                # no version id generator, no values set on the table,
                # and version id wasn't manually incremented.
                # set version id to itself so we get an UPDATE
                # statement
                params[col.key] = update_version_id

        elif not (params or value_params):
            continue

        has_all_pks = True
        expect_pk_cascaded = False
        if bulk:
            # keys here are mapped attribute keys, so
            # look at mapper attribute keys for pk
            pk_params = {
                propkey_to_col[propkey]._label: state_dict.get(propkey)
                for propkey in set(propkey_to_col).intersection(
                    mapper._pk_attr_keys_by_table[table]
                )
            }
            if util.NONE_SET.intersection(pk_params.values()):
                raise sa_exc.InvalidRequestError(
                    f"No primary key value supplied for column(s) "
                    f"""{
                        ', '.join(
                            str(c) for c in pks if pk_params[c._label] is None
                        )
                    }; """
                    "per-row ORM Bulk UPDATE by Primary Key requires that "
                    "records contain primary key values",
                    code="bupq",
                )

        else:
            pk_params = {}
            for col in pks:
                propkey = mapper._columntoproperty[col].key

                history = state.manager[propkey].impl.get_history(
                    state, state_dict, attributes.PASSIVE_OFF
                )

                if history.added:
                    if (
                        not history.deleted
                        or ("pk_cascaded", state, col)
                        in uowtransaction.attributes
                    ):
                        expect_pk_cascaded = True
                        pk_params[col._label] = history.added[0]
                        params.pop(col.key, None)
                    else:
                        # else, use the old value to locate the row
                        pk_params[col._label] = history.deleted[0]
                        if col in value_params:
                            has_all_pks = False
                else:
                    pk_params[col._label] = history.unchanged[0]
                if pk_params[col._label] is None:
                    raise orm_exc.FlushError(
                        "Can't update table %s using NULL for primary "
                        "key value on column %s" % (table, col)
                    )

        if include_bulk_keys:
            params.update((k, state_dict[k]) for k in include_bulk_keys)

        if params or value_params:
            params.update(pk_params)
            yield (
                state,
                state_dict,
                params,
                mapper,
                connection,
                value_params,
                has_all_defaults,
                has_all_pks,
            )
        elif expect_pk_cascaded:
            # no UPDATE occurs on this table, but we expect that CASCADE rules
            # have changed the primary key of the row; propagate this event to
            # other columns that expect to have been modified. this normally
            # occurs after the UPDATE is emitted however we invoke it here
            # explicitly in the absence of our invoking an UPDATE
            for m, equated_pairs in mapper._table_to_equated[table]:
                sync.populate(
                    state,
                    m,
                    state,
                    m,
                    equated_pairs,
                    uowtransaction,
                    mapper.passive_updates,
                )


def _collect_post_update_commands(
    base_mapper, uowtransaction, table, states_to_update, post_update_cols
):
    """Identify sets of values to use in UPDATE statements for a
    list of states within a post_update operation.

    """

    for (
        state,
        state_dict,
        mapper,
        connection,
        update_version_id,
    ) in states_to_update:
        # assert table in mapper._pks_by_table

        pks = mapper._pks_by_table[table]
        params = {}
        hasdata = False

        for col in mapper._cols_by_table[table]:
            if col in pks:
                params[col._label] = mapper._get_state_attr_by_column(
                    state, state_dict, col, passive=attributes.PASSIVE_OFF
                )

            elif col in post_update_cols or col.onupdate is not None:
                prop = mapper._columntoproperty[col]
                history = state.manager[prop.key].impl.get_history(
                    state, state_dict, attributes.PASSIVE_NO_INITIALIZE
                )
                if history.added:
                    value = history.added[0]
                    params[col.key] = value
                    hasdata = True
        if hasdata:
            if (
                update_version_id is not None
                and mapper.version_id_col in mapper._cols_by_table[table]
            ):
                col = mapper.version_id_col
                params[col._label] = update_version_id

                if (
                    bool(state.key)
                    and col.key not in params
                    and mapper.version_id_generator is not False
                ):
                    val = mapper.version_id_generator(update_version_id)
                    params[col.key] = val
            yield state, state_dict, mapper, connection, params


def _collect_delete_commands(
    base_mapper, uowtransaction, table, states_to_delete
):
    """Identify values to use in DELETE statements for a list of
    states to be deleted."""

    for (
        state,
        state_dict,
        mapper,
        connection,
        update_version_id,
    ) in states_to_delete:
        if table not in mapper._pks_by_table:
            continue

        params = {}
        for col in mapper._pks_by_table[table]:
            params[col.key] = value = (
                mapper._get_committed_state_attr_by_column(
                    state, state_dict, col
                )
            )
            if value is None:
                raise orm_exc.FlushError(
                    "Can't delete from table %s "
                    "using NULL for primary "
                    "key value on column %s" % (table, col)
                )

        if (
            update_version_id is not None
            and mapper.version_id_col in mapper._cols_by_table[table]
        ):
            params[mapper.version_id_col.key] = update_version_id
        yield params, connection


def _emit_update_statements(
    base_mapper,
    uowtransaction,
    mapper,
    table,
    update,
    *,
    bookkeeping=True,
    use_orm_update_stmt=None,
    enable_check_rowcount=True,
):
    """Emit UPDATE statements corresponding to value lists collected
    by _collect_update_commands()."""

    needs_version_id = (
        mapper.version_id_col is not None
        and mapper.version_id_col in mapper._cols_by_table[table]
    )

    execution_options = {"compiled_cache": base_mapper._compiled_cache}

    def update_stmt(existing_stmt=None):
        clauses = BooleanClauseList._construct_raw(operators.and_)

        for col in mapper._pks_by_table[table]:
            clauses._append_inplace(
                col == sql.bindparam(col._label, type_=col.type)
            )

        if needs_version_id:
            clauses._append_inplace(
                mapper.version_id_col
                == sql.bindparam(
                    mapper.version_id_col._label,
                    type_=mapper.version_id_col.type,
                )
            )

        if existing_stmt is not None:
            stmt = existing_stmt.where(clauses)
        else:
            stmt = table.update().where(clauses)
        return stmt

    if use_orm_update_stmt is not None:
        cached_stmt = update_stmt(use_orm_update_stmt)

    else:
        cached_stmt = base_mapper._memo(("update", table), update_stmt)

    for (
        (connection, paramkeys, hasvalue, has_all_defaults, has_all_pks),
        records,
    ) in groupby(
        update,
        lambda rec: (
            rec[4],  # connection
            set(rec[2]),  # set of parameter keys
            bool(rec[5]),  # whether or not we have "value" parameters
            rec[6],  # has_all_defaults
            rec[7],  # has all pks
        ),
    ):
        rows = 0
        records = list(records)

        statement = cached_stmt

        if use_orm_update_stmt is not None:
            statement = statement._annotate(
                {
                    "_emit_update_table": table,
                    "_emit_update_mapper": mapper,
                }
            )

        return_defaults = False

        if not has_all_pks:
            statement = statement.return_defaults(*mapper._pks_by_table[table])
            return_defaults = True

        if (
            bookkeeping
            and not has_all_defaults
            and mapper.base_mapper.eager_defaults is True
            # change as of #8889 - if RETURNING is not going to be used anyway,
            # (applies to MySQL, MariaDB which lack UPDATE RETURNING) ensure
            # we can do an executemany UPDATE which is more efficient
            and table.implicit_returning
            and connection.dialect.update_returning
        ):
            statement = statement.return_defaults(
                *mapper._server_onupdate_default_cols[table]
            )
            return_defaults = True

        if mapper._version_id_has_server_side_value:
            statement = statement.return_defaults(mapper.version_id_col)
            return_defaults = True

        assert_singlerow = connection.dialect.supports_sane_rowcount

        assert_multirow = (
            assert_singlerow
            and connection.dialect.supports_sane_multi_rowcount
        )

        # change as of #8889 - if RETURNING is not going to be used anyway,
        # (applies to MySQL, MariaDB which lack UPDATE RETURNING) ensure
        # we can do an executemany UPDATE which is more efficient
        allow_executemany = not return_defaults and not needs_version_id

        if hasvalue:
            for (
                state,
                state_dict,
                params,
                mapper,
                connection,
                value_params,
                has_all_defaults,
                has_all_pks,
            ) in records:
                c = connection.execute(
                    statement.values(value_params),
                    params,
                    execution_options=execution_options,
                )
                if bookkeeping:
                    _postfetch(
                        mapper,
                        uowtransaction,
                        table,
                        state,
                        state_dict,
                        c,
                        c.context.compiled_parameters[0],
                        value_params,
                        True,
                        c.returned_defaults,
                    )
                rows += c.rowcount
                check_rowcount = enable_check_rowcount and assert_singlerow
        else:
            if not allow_executemany:
                check_rowcount = enable_check_rowcount and assert_singlerow
                for (
                    state,
                    state_dict,
                    params,
                    mapper,
                    connection,
                    value_params,
                    has_all_defaults,
                    has_all_pks,
                ) in records:
                    c = connection.execute(
                        statement, params, execution_options=execution_options
                    )

                    # TODO: why with bookkeeping=False?
                    if bookkeeping:
                        _postfetch(
                            mapper,
                            uowtransaction,
                            table,
                            state,
                            state_dict,
                            c,
                            c.context.compiled_parameters[0],
                            value_params,
                            True,
                            c.returned_defaults,
                        )
                    rows += c.rowcount
            else:
                multiparams = [rec[2] for rec in records]

                check_rowcount = enable_check_rowcount and (
                    assert_multirow
                    or (assert_singlerow and len(multiparams) == 1)
                )

                c = connection.execute(
                    statement, multiparams, execution_options=execution_options
                )

                rows += c.rowcount

                for (
                    state,
                    state_dict,
                    params,
                    mapper,
                    connection,
                    value_params,
                    has_all_defaults,
                    has_all_pks,
                ) in records:
                    if bookkeeping:
                        _postfetch(
                            mapper,
                            uowtransaction,
                            table,
                            state,
                            state_dict,
                            c,
                            c.context.compiled_parameters[0],
                            value_params,
                            True,
                            (
                                c.returned_defaults
                                if not c.context.executemany
                                else None
                            ),
                        )

        if check_rowcount:
            if rows != len(records):
                raise orm_exc.StaleDataError(
                    "UPDATE statement on table '%s' expected to "
                    "update %d row(s); %d were matched."
                    % (table.description, len(records), rows)
                )

        elif needs_version_id:
            util.warn(
                "Dialect %s does not support updated rowcount "
                "- versioning cannot be verified."
                % c.dialect.dialect_description
            )


def _emit_insert_statements(
    base_mapper,
    uowtransaction,
    mapper,
    table,
    insert,
    *,
    bookkeeping=True,
    use_orm_insert_stmt=None,
    execution_options=None,
):
    """Emit INSERT statements corresponding to value lists collected
    by _collect_insert_commands()."""

    if use_orm_insert_stmt is not None:
        cached_stmt = use_orm_insert_stmt
        exec_opt = util.EMPTY_DICT

        # if a user query with RETURNING was passed, we definitely need
        # to use RETURNING.
        returning_is_required_anyway = bool(use_orm_insert_stmt._returning)
        deterministic_results_reqd = (
            returning_is_required_anyway
            and use_orm_insert_stmt._sort_by_parameter_order
        ) or bookkeeping
    else:
        returning_is_required_anyway = False
        deterministic_results_reqd = bookkeeping
        cached_stmt = base_mapper._memo(("insert", table), table.insert)
        exec_opt = {"compiled_cache": base_mapper._compiled_cache}

    if execution_options:
        execution_options = util.EMPTY_DICT.merge_with(
            exec_opt, execution_options
        )
    else:
        execution_options = exec_opt

    return_result = None

    for (
        (connection, _, hasvalue, has_all_pks, has_all_defaults),
        records,
    ) in groupby(
        insert,
        lambda rec: (
            rec[4],  # connection
            set(rec[2]),  # parameter keys
            bool(rec[5]),  # whether we have "value" parameters
            rec[6],
            rec[7],
        ),
    ):
        statement = cached_stmt

        if use_orm_insert_stmt is not None:
            statement = statement._annotate(
                {
                    "_emit_insert_table": table,
                    "_emit_insert_mapper": mapper,
                }
            )

        if (
            (
                not bookkeeping
                or (
                    has_all_defaults
                    or not base_mapper._prefer_eager_defaults(
                        connection.dialect, table
                    )
                    or not table.implicit_returning
                    or not connection.dialect.insert_returning
                )
            )
            and not returning_is_required_anyway
            and has_all_pks
            and not hasvalue
        ):
            # the "we don't need newly generated values back" section.
            # here we have all the PKs, all the defaults or we don't want
            # to fetch them, or the dialect doesn't support RETURNING at all
            # so we have to post-fetch / use lastrowid anyway.
            records = list(records)
            multiparams = [rec[2] for rec in records]

            result = connection.execute(
                statement, multiparams, execution_options=execution_options
            )
            if bookkeeping:
                for (
                    (
                        state,
                        state_dict,
                        params,
                        mapper_rec,
                        conn,
                        value_params,
                        has_all_pks,
                        has_all_defaults,
                    ),
                    last_inserted_params,
                ) in zip(records, result.context.compiled_parameters):
                    if state:
                        _postfetch(
                            mapper_rec,
                            uowtransaction,
                            table,
                            state,
                            state_dict,
                            result,
                            last_inserted_params,
                            value_params,
                            False,
                            (
                                result.returned_defaults
                                if not result.context.executemany
                                else None
                            ),
                        )
                    else:
                        _postfetch_bulk_save(mapper_rec, state_dict, table)

        else:
            # here, we need defaults and/or pk values back or we otherwise
            # know that we are using RETURNING in any case

            records = list(records)

            if returning_is_required_anyway or (
                table.implicit_returning and not hasvalue and len(records) > 1
            ):
                if (
                    deterministic_results_reqd
                    and connection.dialect.insert_executemany_returning_sort_by_parameter_order  # noqa: E501
                ) or (
                    not deterministic_results_reqd
                    and connection.dialect.insert_executemany_returning
                ):
                    do_executemany = True
                elif returning_is_required_anyway:
                    if deterministic_results_reqd:
                        dt = " with RETURNING and sort by parameter order"
                    else:
                        dt = " with RETURNING"
                    raise sa_exc.InvalidRequestError(
                        f"Can't use explicit RETURNING for bulk INSERT "
                        f"operation with "
                        f"{connection.dialect.dialect_description} backend; "
                        f"executemany{dt} is not enabled for this dialect."
                    )
                else:
                    do_executemany = False
            else:
                do_executemany = False

            if use_orm_insert_stmt is None:
                if (
                    not has_all_defaults
                    and base_mapper._prefer_eager_defaults(
                        connection.dialect, table
                    )
                ):
                    statement = statement.return_defaults(
                        *mapper._server_default_cols[table],
                        sort_by_parameter_order=bookkeeping,
                    )

            if mapper.version_id_col is not None:
                statement = statement.return_defaults(
                    mapper.version_id_col,
                    sort_by_parameter_order=bookkeeping,
                )
            elif do_executemany:
                statement = statement.return_defaults(
                    *table.primary_key, sort_by_parameter_order=bookkeeping
                )

            if do_executemany:
                multiparams = [rec[2] for rec in records]

                result = connection.execute(
                    statement, multiparams, execution_options=execution_options
                )

                if use_orm_insert_stmt is not None:
                    if return_result is None:
                        return_result = result
                    else:
                        return_result = return_result.splice_vertically(result)

                if bookkeeping:
                    for (
                        (
                            state,
                            state_dict,
                            params,
                            mapper_rec,
                            conn,
                            value_params,
                            has_all_pks,
                            has_all_defaults,
                        ),
                        last_inserted_params,
                        inserted_primary_key,
                        returned_defaults,
                    ) in zip_longest(
                        records,
                        result.context.compiled_parameters,
                        result.inserted_primary_key_rows,
                        result.returned_defaults_rows or (),
                    ):
                        if inserted_primary_key is None:
                            # this is a real problem and means that we didn't
                            # get back as many PK rows.  we can't continue
                            # since this indicates PK rows were missing, which
                            # means we likely mis-populated records starting
                            # at that point with incorrectly matched PK
                            # values.
                            raise orm_exc.FlushError(
                                "Multi-row INSERT statement for %s did not "
                                "produce "
                                "the correct number of INSERTed rows for "
                                "RETURNING.  Ensure there are no triggers or "
                                "special driver issues preventing INSERT from "
                                "functioning properly." % mapper_rec
                            )

                        for pk, col in zip(
                            inserted_primary_key,
                            mapper._pks_by_table[table],
                        ):
                            prop = mapper_rec._columntoproperty[col]
                            if state_dict.get(prop.key) is None:
                                state_dict[prop.key] = pk

                        if state:
                            _postfetch(
                                mapper_rec,
                                uowtransaction,
                                table,
                                state,
                                state_dict,
                                result,
                                last_inserted_params,
                                value_params,
                                False,
                                returned_defaults,
                            )
                        else:
                            _postfetch_bulk_save(mapper_rec, state_dict, table)
            else:
                assert not returning_is_required_anyway

                for (
                    state,
                    state_dict,
                    params,
                    mapper_rec,
                    connection,
                    value_params,
                    has_all_pks,
                    has_all_defaults,
                ) in records:
                    if value_params:
                        result = connection.execute(
                            statement.values(value_params),
                            params,
                            execution_options=execution_options,
                        )
                    else:
                        result = connection.execute(
                            statement,
                            params,
                            execution_options=execution_options,
                        )

                    primary_key = result.inserted_primary_key
                    if primary_key is None:
                        raise orm_exc.FlushError(
                            "Single-row INSERT statement for %s "
                            "did not produce a "
                            "new primary key result "
                            "being invoked.  Ensure there are no triggers or "
                            "special driver issues preventing INSERT from "
                            "functioning properly." % (mapper_rec,)
                        )
                    for pk, col in zip(
                        primary_key, mapper._pks_by_table[table]
                    ):
                        prop = mapper_rec._columntoproperty[col]
                        if (
                            col in value_params
                            or state_dict.get(prop.key) is None
                        ):
                            state_dict[prop.key] = pk
                    if bookkeeping:
                        if state:
                            _postfetch(
                                mapper_rec,
                                uowtransaction,
                                table,
                                state,
                                state_dict,
                                result,
                                result.context.compiled_parameters[0],
                                value_params,
                                False,
                                (
                                    result.returned_defaults
                                    if not result.context.executemany
                                    else None
                                ),
                            )
                        else:
                            _postfetch_bulk_save(mapper_rec, state_dict, table)

    if use_orm_insert_stmt is not None:
        if return_result is None:
            return _cursor.null_dml_result()
        else:
            return return_result


def _emit_post_update_statements(
    base_mapper, uowtransaction, mapper, table, update
):
    """Emit UPDATE statements corresponding to value lists collected
    by _collect_post_update_commands()."""

    execution_options = {"compiled_cache": base_mapper._compiled_cache}

    needs_version_id = (
        mapper.version_id_col is not None
        and mapper.version_id_col in mapper._cols_by_table[table]
    )

    def update_stmt():
        clauses = BooleanClauseList._construct_raw(operators.and_)

        for col in mapper._pks_by_table[table]:
            clauses._append_inplace(
                col == sql.bindparam(col._label, type_=col.type)
            )

        if needs_version_id:
            clauses._append_inplace(
                mapper.version_id_col
                == sql.bindparam(
                    mapper.version_id_col._label,
                    type_=mapper.version_id_col.type,
                )
            )

        stmt = table.update().where(clauses)

        return stmt

    statement = base_mapper._memo(("post_update", table), update_stmt)

    if mapper._version_id_has_server_side_value:
        statement = statement.return_defaults(mapper.version_id_col)

    # execute each UPDATE in the order according to the original
    # list of states to guarantee row access order, but
    # also group them into common (connection, cols) sets
    # to support executemany().
    for key, records in groupby(
        update,
        lambda rec: (rec[3], set(rec[4])),  # connection  # parameter keys
    ):
        rows = 0

        records = list(records)
        connection = key[0]

        assert_singlerow = connection.dialect.supports_sane_rowcount
        assert_multirow = (
            assert_singlerow
            and connection.dialect.supports_sane_multi_rowcount
        )
        allow_executemany = not needs_version_id or assert_multirow

        if not allow_executemany:
            check_rowcount = assert_singlerow
            for state, state_dict, mapper_rec, connection, params in records:
                c = connection.execute(
                    statement, params, execution_options=execution_options
                )

                _postfetch_post_update(
                    mapper_rec,
                    uowtransaction,
                    table,
                    state,
                    state_dict,
                    c,
                    c.context.compiled_parameters[0],
                )
                rows += c.rowcount
        else:
            multiparams = [
                params
                for state, state_dict, mapper_rec, conn, params in records
            ]

            check_rowcount = assert_multirow or (
                assert_singlerow and len(multiparams) == 1
            )

            c = connection.execute(
                statement, multiparams, execution_options=execution_options
            )

            rows += c.rowcount
            for state, state_dict, mapper_rec, connection, params in records:
                _postfetch_post_update(
                    mapper_rec,
                    uowtransaction,
                    table,
                    state,
                    state_dict,
                    c,
                    c.context.compiled_parameters[0],
                )

        if check_rowcount:
            if rows != len(records):
                raise orm_exc.StaleDataError(
                    "UPDATE statement on table '%s' expected to "
                    "update %d row(s); %d were matched."
                    % (table.description, len(records), rows)
                )

        elif needs_version_id:
            util.warn(
                "Dialect %s does not support updated rowcount "
                "- versioning cannot be verified."
                % c.dialect.dialect_description
            )


def _emit_delete_statements(
    base_mapper, uowtransaction, mapper, table, delete
):
    """Emit DELETE statements corresponding to value lists collected
    by _collect_delete_commands()."""

    need_version_id = (
        mapper.version_id_col is not None
        and mapper.version_id_col in mapper._cols_by_table[table]
    )

    def delete_stmt():
        clauses = BooleanClauseList._construct_raw(operators.and_)

        for col in mapper._pks_by_table[table]:
            clauses._append_inplace(
                col == sql.bindparam(col.key, type_=col.type)
            )

        if need_version_id:
            clauses._append_inplace(
                mapper.version_id_col
                == sql.bindparam(
                    mapper.version_id_col.key, type_=mapper.version_id_col.type
                )
            )

        return table.delete().where(clauses)

    statement = base_mapper._memo(("delete", table), delete_stmt)
    for connection, recs in groupby(delete, lambda rec: rec[1]):  # connection
        del_objects = [params for params, connection in recs]

        execution_options = {"compiled_cache": base_mapper._compiled_cache}
        expected = len(del_objects)
        rows_matched = -1
        only_warn = False

        if (
            need_version_id
            and not connection.dialect.supports_sane_multi_rowcount
        ):
            if connection.dialect.supports_sane_rowcount:
                rows_matched = 0
                # execute deletes individually so that versioned
                # rows can be verified
                for params in del_objects:
                    c = connection.execute(
                        statement, params, execution_options=execution_options
                    )
                    rows_matched += c.rowcount
            else:
                util.warn(
                    "Dialect %s does not support deleted rowcount "
                    "- versioning cannot be verified."
                    % connection.dialect.dialect_description
                )
                connection.execute(
                    statement, del_objects, execution_options=execution_options
                )
        else:
            c = connection.execute(
                statement, del_objects, execution_options=execution_options
            )

            if not need_version_id:
                only_warn = True

            rows_matched = c.rowcount

        if (
            base_mapper.confirm_deleted_rows
            and rows_matched > -1
            and expected != rows_matched
            and (
                connection.dialect.supports_sane_multi_rowcount
                or len(del_objects) == 1
            )
        ):
            # TODO: why does this "only warn" if versioning is turned off,
            # whereas the UPDATE raises?
            if only_warn:
                util.warn(
                    "DELETE statement on table '%s' expected to "
                    "delete %d row(s); %d were matched.  Please set "
                    "confirm_deleted_rows=False within the mapper "
                    "configuration to prevent this warning."
                    % (table.description, expected, rows_matched)
                )
            else:
                raise orm_exc.StaleDataError(
                    "DELETE statement on table '%s' expected to "
                    "delete %d row(s); %d were matched.  Please set "
                    "confirm_deleted_rows=False within the mapper "
                    "configuration to prevent this warning."
                    % (table.description, expected, rows_matched)
                )


def _finalize_insert_update_commands(base_mapper, uowtransaction, states):
    """finalize state on states that have been inserted or updated,
    including calling after_insert/after_update events.

    """
    for state, state_dict, mapper, connection, has_identity in states:
        if mapper._readonly_props:
            readonly = state.unmodified_intersection(
                [
                    p.key
                    for p in mapper._readonly_props
                    if (
                        p.expire_on_flush
                        and (not p.deferred or p.key in state.dict)
                    )
                    or (
                        not p.expire_on_flush
                        and not p.deferred
                        and p.key not in state.dict
                    )
                ]
            )
            if readonly:
                state._expire_attributes(state.dict, readonly)

        # if eager_defaults option is enabled, load
        # all expired cols.  Else if we have a version_id_col, make sure
        # it isn't expired.
        toload_now = []

        # this is specifically to emit a second SELECT for eager_defaults,
        # so only if it's set to True, not "auto"
        if base_mapper.eager_defaults is True:
            toload_now.extend(
                state._unloaded_non_object.intersection(
                    mapper._server_default_plus_onupdate_propkeys
                )
            )

        if (
            mapper.version_id_col is not None
            and mapper.version_id_generator is False
        ):
            if mapper._version_id_prop.key in state.unloaded:
                toload_now.extend([mapper._version_id_prop.key])

        if toload_now:
            state.key = base_mapper._identity_key_from_state(state)
            stmt = future.select(mapper).set_label_style(
                LABEL_STYLE_TABLENAME_PLUS_COL
            )
            loading.load_on_ident(
                uowtransaction.session,
                stmt,
                state.key,
                refresh_state=state,
                only_load_props=toload_now,
            )

        # call after_XXX extensions
        if not has_identity:
            mapper.dispatch.after_insert(mapper, connection, state)
        else:
            mapper.dispatch.after_update(mapper, connection, state)

        if (
            mapper.version_id_generator is False
            and mapper.version_id_col is not None
        ):
            if state_dict[mapper._version_id_prop.key] is None:
                raise orm_exc.FlushError(
                    "Instance does not contain a non-NULL version value"
                )


def _postfetch_post_update(
    mapper, uowtransaction, table, state, dict_, result, params
):
    needs_version_id = (
        mapper.version_id_col is not None
        and mapper.version_id_col in mapper._cols_by_table[table]
    )

    if not uowtransaction.is_deleted(state):
        # post updating after a regular INSERT or UPDATE, do a full postfetch
        prefetch_cols = result.context.compiled.prefetch
        postfetch_cols = result.context.compiled.postfetch
    elif needs_version_id:
        # post updating before a DELETE with a version_id_col, need to
        # postfetch just version_id_col
        prefetch_cols = postfetch_cols = ()
    else:
        # post updating before a DELETE without a version_id_col,
        # don't need to postfetch
        return

    if needs_version_id:
        prefetch_cols = list(prefetch_cols) + [mapper.version_id_col]

    refresh_flush = bool(mapper.class_manager.dispatch.refresh_flush)
    if refresh_flush:
        load_evt_attrs = []

    for c in prefetch_cols:
        if c.key in params and c in mapper._columntoproperty:
            dict_[mapper._columntoproperty[c].key] = params[c.key]
            if refresh_flush:
                load_evt_attrs.append(mapper._columntoproperty[c].key)

    if refresh_flush and load_evt_attrs:
        mapper.class_manager.dispatch.refresh_flush(
            state, uowtransaction, load_evt_attrs
        )

    if postfetch_cols:
        state._expire_attributes(
            state.dict,
            [
                mapper._columntoproperty[c].key
                for c in postfetch_cols
                if c in mapper._columntoproperty
            ],
        )


def _postfetch(
    mapper,
    uowtransaction,
    table,
    state,
    dict_,
    result,
    params,
    value_params,
    isupdate,
    returned_defaults,
):
    """Expire attributes in need of newly persisted database state,
    after an INSERT or UPDATE statement has proceeded for that
    state."""

    prefetch_cols = result.context.compiled.prefetch
    postfetch_cols = result.context.compiled.postfetch
    returning_cols = result.context.compiled.effective_returning

    if (
        mapper.version_id_col is not None
        and mapper.version_id_col in mapper._cols_by_table[table]
    ):
        prefetch_cols = list(prefetch_cols) + [mapper.version_id_col]

    refresh_flush = bool(mapper.class_manager.dispatch.refresh_flush)
    if refresh_flush:
        load_evt_attrs = []

    if returning_cols:
        row = returned_defaults
        if row is not None:
            for row_value, col in zip(row, returning_cols):
                # pk cols returned from insert are handled
                # distinctly, don't step on the values here
                if col.primary_key and result.context.isinsert:
                    continue

                # note that columns can be in the "return defaults" that are
                # not mapped to this mapper, typically because they are
                # "excluded", which can be specified directly or also occurs
                # when using declarative w/ single table inheritance
                prop = mapper._columntoproperty.get(col)
                if prop:
                    dict_[prop.key] = row_value
                    if refresh_flush:
                        load_evt_attrs.append(prop.key)

    for c in prefetch_cols:
        if c.key in params and c in mapper._columntoproperty:
            pkey = mapper._columntoproperty[c].key

            # set prefetched value in dict and also pop from committed_state,
            # since this is new database state that replaces whatever might
            # have previously been fetched (see #10800).  this is essentially a
            # shorthand version of set_committed_value(), which could also be
            # used here directly (with more overhead)
            dict_[pkey] = params[c.key]
            state.committed_state.pop(pkey, None)

            if refresh_flush:
                load_evt_attrs.append(pkey)

    if refresh_flush and load_evt_attrs:
        mapper.class_manager.dispatch.refresh_flush(
            state, uowtransaction, load_evt_attrs
        )

    if isupdate and value_params:
        # explicitly suit the use case specified by
        # [ticket:3801], PK SQL expressions for UPDATE on non-RETURNING
        # database which are set to themselves in order to do a version bump.
        postfetch_cols.extend(
            [
                col
                for col in value_params
                if col.primary_key and col not in returning_cols
            ]
        )

    if postfetch_cols:
        state._expire_attributes(
            state.dict,
            [
                mapper._columntoproperty[c].key
                for c in postfetch_cols
                if c in mapper._columntoproperty
            ],
        )

    # synchronize newly inserted ids from one table to the next
    # TODO: this still goes a little too often.  would be nice to
    # have definitive list of "columns that changed" here
    for m, equated_pairs in mapper._table_to_equated[table]:
        sync.populate(
            state,
            m,
            state,
            m,
            equated_pairs,
            uowtransaction,
            mapper.passive_updates,
        )


def _postfetch_bulk_save(mapper, dict_, table):
    for m, equated_pairs in mapper._table_to_equated[table]:
        sync.bulk_populate_inherit_keys(dict_, m, equated_pairs)


def _connections_for_states(base_mapper, uowtransaction, states):
    """Return an iterator of (state, state.dict, mapper, connection).

    The states are sorted according to _sort_states, then paired
    with the connection they should be using for the given
    unit of work transaction.

    """
    # if session has a connection callable,
    # organize individual states with the connection
    # to use for update
    if uowtransaction.session.connection_callable:
        connection_callable = uowtransaction.session.connection_callable
    else:
        connection = uowtransaction.transaction.connection(base_mapper)
        connection_callable = None

    for state in _sort_states(base_mapper, states):
        if connection_callable:
            connection = connection_callable(base_mapper, state.obj())

        mapper = state.manager.mapper

        yield state, state.dict, mapper, connection


def _sort_states(mapper, states):
    pending = set(states)
    persistent = {s for s in pending if s.key is not None}
    pending.difference_update(persistent)

    try:
        persistent_sorted = sorted(
            persistent, key=mapper._persistent_sortkey_fn
        )
    except TypeError as err:
        raise sa_exc.InvalidRequestError(
            "Could not sort objects by primary key; primary key "
            "values must be sortable in Python (was: %s)" % err
        ) from err
    return (
        sorted(pending, key=operator.attrgetter("insert_order"))
        + persistent_sorted
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\arrayprint.py ===
"""Array printing function

$Id: arrayprint.py,v 1.9 2005/09/13 13:58:44 teoliphant Exp $

"""
__all__ = ["array2string", "array_str", "array_repr",
           "set_printoptions", "get_printoptions", "printoptions",
           "format_float_positional", "format_float_scientific"]
__docformat__ = 'restructuredtext'

#
# Written by Konrad Hinsen <hinsenk@ere.umontreal.ca>
# last revision: 1996-3-13
# modified by Jim Hugunin 1997-3-3 for repr's and str's (and other details)
# and by Perry Greenfield 2000-4-1 for numarray
# and by Travis Oliphant  2005-8-22 for numpy


# Note: Both scalartypes.c.src and arrayprint.py implement strs for numpy
# scalars but for different purposes. scalartypes.c.src has str/reprs for when
# the scalar is printed on its own, while arrayprint.py has strs for when
# scalars are printed inside an ndarray. Only the latter strs are currently
# user-customizable.

import functools
import numbers
import sys

try:
    from _thread import get_ident
except ImportError:
    from _dummy_thread import get_ident

import contextlib
import operator
import warnings

import numpy as np

from . import numerictypes as _nt
from .fromnumeric import any
from .multiarray import (
    array,
    datetime_as_string,
    datetime_data,
    dragon4_positional,
    dragon4_scientific,
    ndarray,
)
from .numeric import asarray, concatenate, errstate
from .numerictypes import complex128, flexible, float64, int_
from .overrides import array_function_dispatch, set_module
from .printoptions import format_options
from .umath import absolute, isfinite, isinf, isnat


def _make_options_dict(precision=None, threshold=None, edgeitems=None,
                       linewidth=None, suppress=None, nanstr=None, infstr=None,
                       sign=None, formatter=None, floatmode=None, legacy=None,
                       override_repr=None):
    """
    Make a dictionary out of the non-None arguments, plus conversion of
    *legacy* and sanity checks.
    """

    options = {k: v for k, v in list(locals().items()) if v is not None}

    if suppress is not None:
        options['suppress'] = bool(suppress)

    modes = ['fixed', 'unique', 'maxprec', 'maxprec_equal']
    if floatmode not in modes + [None]:
        raise ValueError("floatmode option must be one of " +
                         ", ".join(f'"{m}"' for m in modes))

    if sign not in [None, '-', '+', ' ']:
        raise ValueError("sign option must be one of ' ', '+', or '-'")

    if legacy is False:
        options['legacy'] = sys.maxsize
    elif legacy == False:  # noqa: E712
        warnings.warn(
            f"Passing `legacy={legacy!r}` is deprecated.",
            FutureWarning, stacklevel=3
        )
        options['legacy'] = sys.maxsize
    elif legacy == '1.13':
        options['legacy'] = 113
    elif legacy == '1.21':
        options['legacy'] = 121
    elif legacy == '1.25':
        options['legacy'] = 125
    elif legacy == '2.1':
        options['legacy'] = 201
    elif legacy == '2.2':
        options['legacy'] = 202
    elif legacy is None:
        pass  # OK, do nothing.
    else:
        warnings.warn(
            "legacy printing option can currently only be '1.13', '1.21', "
            "'1.25', '2.1', '2.2' or `False`", stacklevel=3)

    if threshold is not None:
        # forbid the bad threshold arg suggested by stack overflow, gh-12351
        if not isinstance(threshold, numbers.Number):
            raise TypeError("threshold must be numeric")
        if np.isnan(threshold):
            raise ValueError("threshold must be non-NAN, try "
                             "sys.maxsize for untruncated representation")

    if precision is not None:
        # forbid the bad precision arg as suggested by issue #18254
        try:
            options['precision'] = operator.index(precision)
        except TypeError as e:
            raise TypeError('precision must be an integer') from e

    return options


@set_module('numpy')
def set_printoptions(precision=None, threshold=None, edgeitems=None,
                     linewidth=None, suppress=None, nanstr=None,
                     infstr=None, formatter=None, sign=None, floatmode=None,
                     *, legacy=None, override_repr=None):
    """
    Set printing options.

    These options determine the way floating point numbers, arrays and
    other NumPy objects are displayed.

    Parameters
    ----------
    precision : int or None, optional
        Number of digits of precision for floating point output (default 8).
        May be None if `floatmode` is not `fixed`, to print as many digits as
        necessary to uniquely specify the value.
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr (default 1000).
        To always use the full repr without summarization, pass `sys.maxsize`.
    edgeitems : int, optional
        Number of array items in summary at beginning and end of
        each dimension (default 3).
    linewidth : int, optional
        The number of characters per line for the purpose of inserting
        line breaks (default 75).
    suppress : bool, optional
        If True, always print floating point numbers using fixed point
        notation, in which case numbers equal to zero in the current precision
        will print as zero.  If False, then scientific notation is used when
        absolute value of the smallest number is < 1e-4 or the ratio of the
        maximum absolute value to the minimum is > 1e3. The default is False.
    nanstr : str, optional
        String representation of floating point not-a-number (default nan).
    infstr : str, optional
        String representation of floating point infinity (default inf).
    sign : string, either '-', '+', or ' ', optional
        Controls printing of the sign of floating-point types. If '+', always
        print the sign of positive values. If ' ', always prints a space
        (whitespace character) in the sign position of positive values.  If
        '-', omit the sign character of positive values. (default '-')

        .. versionchanged:: 2.0
             The sign parameter can now be an integer type, previously
             types were floating-point types.

    formatter : dict of callables, optional
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are:

        - 'bool'
        - 'int'
        - 'timedelta' : a `numpy.timedelta64`
        - 'datetime' : a `numpy.datetime64`
        - 'float'
        - 'longfloat' : 128-bit floats
        - 'complexfloat'
        - 'longcomplexfloat' : composed of two 128-bit floats
        - 'numpystr' : types `numpy.bytes_` and `numpy.str_`
        - 'object' : `np.object_` arrays

        Other keys that can be used to set a group of types at once are:

        - 'all' : sets all types
        - 'int_kind' : sets 'int'
        - 'float_kind' : sets 'float' and 'longfloat'
        - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
        - 'str_kind' : sets 'numpystr'
    floatmode : str, optional
        Controls the interpretation of the `precision` option for
        floating-point types. Can take the following values
        (default maxprec_equal):

        * 'fixed': Always print exactly `precision` fractional digits,
                even if this would print more or fewer digits than
                necessary to specify the value uniquely.
        * 'unique': Print the minimum number of fractional digits necessary
                to represent each value uniquely. Different elements may
                have a different number of digits. The value of the
                `precision` option is ignored.
        * 'maxprec': Print at most `precision` fractional digits, but if
                an element can be uniquely represented with fewer digits
                only print it with that many.
        * 'maxprec_equal': Print at most `precision` fractional digits,
                but if every element in the array can be uniquely
                represented with an equal number of fewer digits, use that
                many digits for all elements.
    legacy : string or `False`, optional
        If set to the string ``'1.13'`` enables 1.13 legacy printing mode. This
        approximates numpy 1.13 print output by including a space in the sign
        position of floats and different behavior for 0d arrays. This also
        enables 1.21 legacy printing mode (described below).

        If set to the string ``'1.21'`` enables 1.21 legacy printing mode. This
        approximates numpy 1.21 print output of complex structured dtypes
        by not inserting spaces after commas that separate fields and after
        colons.

        If set to ``'1.25'`` approximates printing of 1.25 which mainly means
        that numeric scalars are printed without their type information, e.g.
        as ``3.0`` rather than ``np.float64(3.0)``.

        If set to ``'2.1'``, shape information is not given when arrays are
        summarized (i.e., multiple elements replaced with ``...``).

        If set to ``'2.2'``, the transition to use scientific notation for
        printing ``np.float16`` and ``np.float32`` types may happen later or
        not at all for larger values.

        If set to `False`, disables legacy mode.

        Unrecognized strings will be ignored with a warning for forward
        compatibility.

        .. versionchanged:: 1.22.0
        .. versionchanged:: 2.2

    override_repr: callable, optional
        If set a passed function will be used for generating arrays' repr.
        Other options will be ignored.

    See Also
    --------
    get_printoptions, printoptions, array2string

    Notes
    -----
    `formatter` is always reset with a call to `set_printoptions`.

    Use `printoptions` as a context manager to set the values temporarily.

    Examples
    --------
    Floating point precision can be set:

    >>> import numpy as np
    >>> np.set_printoptions(precision=4)
    >>> np.array([1.123456789])
    [1.1235]

    Long arrays can be summarised:

    >>> np.set_printoptions(threshold=5)
    >>> np.arange(10)
    array([0, 1, 2, ..., 7, 8, 9], shape=(10,))

    Small results can be suppressed:

    >>> eps = np.finfo(float).eps
    >>> x = np.arange(4.)
    >>> x**2 - (x + eps)**2
    array([-4.9304e-32, -4.4409e-16,  0.0000e+00,  0.0000e+00])
    >>> np.set_printoptions(suppress=True)
    >>> x**2 - (x + eps)**2
    array([-0., -0.,  0.,  0.])

    A custom formatter can be used to display array elements as desired:

    >>> np.set_printoptions(formatter={'all':lambda x: 'int: '+str(-x)})
    >>> x = np.arange(3)
    >>> x
    array([int: 0, int: -1, int: -2])
    >>> np.set_printoptions()  # formatter gets reset
    >>> x
    array([0, 1, 2])

    To put back the default options, you can use:

    >>> np.set_printoptions(edgeitems=3, infstr='inf',
    ... linewidth=75, nanstr='nan', precision=8,
    ... suppress=False, threshold=1000, formatter=None)

    Also to temporarily override options, use `printoptions`
    as a context manager:

    >>> with np.printoptions(precision=2, suppress=True, threshold=5):
    ...     np.linspace(0, 10, 10)
    array([ 0.  ,  1.11,  2.22, ...,  7.78,  8.89, 10.  ], shape=(10,))

    """
    _set_printoptions(precision, threshold, edgeitems, linewidth, suppress,
                      nanstr, infstr, formatter, sign, floatmode,
                      legacy=legacy, override_repr=override_repr)


def _set_printoptions(precision=None, threshold=None, edgeitems=None,
                      linewidth=None, suppress=None, nanstr=None,
                      infstr=None, formatter=None, sign=None, floatmode=None,
                      *, legacy=None, override_repr=None):
    new_opt = _make_options_dict(precision, threshold, edgeitems, linewidth,
                                 suppress, nanstr, infstr, sign, formatter,
                                 floatmode, legacy)
    # formatter and override_repr are always reset
    new_opt['formatter'] = formatter
    new_opt['override_repr'] = override_repr

    updated_opt = format_options.get() | new_opt
    updated_opt.update(new_opt)

    if updated_opt['legacy'] == 113:
        updated_opt['sign'] = '-'

    return format_options.set(updated_opt)


@set_module('numpy')
def get_printoptions():
    """
    Return the current print options.

    Returns
    -------
    print_opts : dict
        Dictionary of current print options with keys

        - precision : int
        - threshold : int
        - edgeitems : int
        - linewidth : int
        - suppress : bool
        - nanstr : str
        - infstr : str
        - sign : str
        - formatter : dict of callables
        - floatmode : str
        - legacy : str or False

        For a full description of these options, see `set_printoptions`.

    See Also
    --------
    set_printoptions, printoptions

    Examples
    --------
    >>> import numpy as np

    >>> np.get_printoptions()
    {'edgeitems': 3, 'threshold': 1000, ..., 'override_repr': None}

    >>> np.get_printoptions()['linewidth']
    75
    >>> np.set_printoptions(linewidth=100)
    >>> np.get_printoptions()['linewidth']
    100

    """
    opts = format_options.get().copy()
    opts['legacy'] = {
        113: '1.13', 121: '1.21', 125: '1.25', 201: '2.1',
        202: '2.2', sys.maxsize: False,
    }[opts['legacy']]
    return opts


def _get_legacy_print_mode():
    """Return the legacy print mode as an int."""
    return format_options.get()['legacy']


@set_module('numpy')
@contextlib.contextmanager
def printoptions(*args, **kwargs):
    """Context manager for setting print options.

    Set print options for the scope of the `with` block, and restore the old
    options at the end. See `set_printoptions` for the full description of
    available options.

    Examples
    --------
    >>> import numpy as np

    >>> from numpy.testing import assert_equal
    >>> with np.printoptions(precision=2):
    ...     np.array([2.0]) / 3
    array([0.67])

    The `as`-clause of the `with`-statement gives the current print options:

    >>> with np.printoptions(precision=2) as opts:
    ...      assert_equal(opts, np.get_printoptions())

    See Also
    --------
    set_printoptions, get_printoptions

    """
    token = _set_printoptions(*args, **kwargs)

    try:
        yield get_printoptions()
    finally:
        format_options.reset(token)


def _leading_trailing(a, edgeitems, index=()):
    """
    Keep only the N-D corners (leading and trailing edges) of an array.

    Should be passed a base-class ndarray, since it makes no guarantees about
    preserving subclasses.
    """
    axis = len(index)
    if axis == a.ndim:
        return a[index]

    if a.shape[axis] > 2 * edgeitems:
        return concatenate((
            _leading_trailing(a, edgeitems, index + np.index_exp[:edgeitems]),
            _leading_trailing(a, edgeitems, index + np.index_exp[-edgeitems:])
        ), axis=axis)
    else:
        return _leading_trailing(a, edgeitems, index + np.index_exp[:])


def _object_format(o):
    """ Object arrays containing lists should be printed unambiguously """
    if type(o) is list:
        fmt = 'list({!r})'
    else:
        fmt = '{!r}'
    return fmt.format(o)

def repr_format(x):
    if isinstance(x, (np.str_, np.bytes_)):
        return repr(x.item())
    return repr(x)

def str_format(x):
    if isinstance(x, (np.str_, np.bytes_)):
        return str(x.item())
    return str(x)

def _get_formatdict(data, *, precision, floatmode, suppress, sign, legacy,
                    formatter, **kwargs):
    # note: extra arguments in kwargs are ignored

    # wrapped in lambdas to avoid taking a code path
    # with the wrong type of data
    formatdict = {
        'bool': lambda: BoolFormat(data),
        'int': lambda: IntegerFormat(data, sign),
        'float': lambda: FloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),
        'longfloat': lambda: FloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),
        'complexfloat': lambda: ComplexFloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),
        'longcomplexfloat': lambda: ComplexFloatingFormat(
            data, precision, floatmode, suppress, sign, legacy=legacy),
        'datetime': lambda: DatetimeFormat(data, legacy=legacy),
        'timedelta': lambda: TimedeltaFormat(data),
        'object': lambda: _object_format,
        'void': lambda: str_format,
        'numpystr': lambda: repr_format}

    # we need to wrap values in `formatter` in a lambda, so that the interface
    # is the same as the above values.
    def indirect(x):
        return lambda: x

    if formatter is not None:
        fkeys = [k for k in formatter.keys() if formatter[k] is not None]
        if 'all' in fkeys:
            for key in formatdict.keys():
                formatdict[key] = indirect(formatter['all'])
        if 'int_kind' in fkeys:
            for key in ['int']:
                formatdict[key] = indirect(formatter['int_kind'])
        if 'float_kind' in fkeys:
            for key in ['float', 'longfloat']:
                formatdict[key] = indirect(formatter['float_kind'])
        if 'complex_kind' in fkeys:
            for key in ['complexfloat', 'longcomplexfloat']:
                formatdict[key] = indirect(formatter['complex_kind'])
        if 'str_kind' in fkeys:
            formatdict['numpystr'] = indirect(formatter['str_kind'])
        for key in formatdict.keys():
            if key in fkeys:
                formatdict[key] = indirect(formatter[key])

    return formatdict

def _get_format_function(data, **options):
    """
    find the right formatting function for the dtype_
    """
    dtype_ = data.dtype
    dtypeobj = dtype_.type
    formatdict = _get_formatdict(data, **options)
    if dtypeobj is None:
        return formatdict["numpystr"]()
    elif issubclass(dtypeobj, _nt.bool):
        return formatdict['bool']()
    elif issubclass(dtypeobj, _nt.integer):
        if issubclass(dtypeobj, _nt.timedelta64):
            return formatdict['timedelta']()
        else:
            return formatdict['int']()
    elif issubclass(dtypeobj, _nt.floating):
        if issubclass(dtypeobj, _nt.longdouble):
            return formatdict['longfloat']()
        else:
            return formatdict['float']()
    elif issubclass(dtypeobj, _nt.complexfloating):
        if issubclass(dtypeobj, _nt.clongdouble):
            return formatdict['longcomplexfloat']()
        else:
            return formatdict['complexfloat']()
    elif issubclass(dtypeobj, (_nt.str_, _nt.bytes_)):
        return formatdict['numpystr']()
    elif issubclass(dtypeobj, _nt.datetime64):
        return formatdict['datetime']()
    elif issubclass(dtypeobj, _nt.object_):
        return formatdict['object']()
    elif issubclass(dtypeobj, _nt.void):
        if dtype_.names is not None:
            return StructuredVoidFormat.from_data(data, **options)
        else:
            return formatdict['void']()
    else:
        return formatdict['numpystr']()


def _recursive_guard(fillvalue='...'):
    """
    Like the python 3.2 reprlib.recursive_repr, but forwards *args and **kwargs

    Decorates a function such that if it calls itself with the same first
    argument, it returns `fillvalue` instead of recursing.

    Largely copied from reprlib.recursive_repr
    """

    def decorating_function(f):
        repr_running = set()

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            key = id(self), get_ident()
            if key in repr_running:
                return fillvalue
            repr_running.add(key)
            try:
                return f(self, *args, **kwargs)
            finally:
                repr_running.discard(key)

        return wrapper

    return decorating_function


# gracefully handle recursive calls, when object arrays contain themselves
@_recursive_guard()
def _array2string(a, options, separator=' ', prefix=""):
    # The formatter __init__s in _get_format_function cannot deal with
    # subclasses yet, and we also need to avoid recursion issues in
    # _formatArray with subclasses which return 0d arrays in place of scalars
    data = asarray(a)
    if a.shape == ():
        a = data

    if a.size > options['threshold']:
        summary_insert = "..."
        data = _leading_trailing(data, options['edgeitems'])
    else:
        summary_insert = ""

    # find the right formatting function for the array
    format_function = _get_format_function(data, **options)

    # skip over "["
    next_line_prefix = " "
    # skip over array(
    next_line_prefix += " " * len(prefix)

    lst = _formatArray(a, format_function, options['linewidth'],
                       next_line_prefix, separator, options['edgeitems'],
                       summary_insert, options['legacy'])
    return lst


def _array2string_dispatcher(
        a, max_line_width=None, precision=None,
        suppress_small=None, separator=None, prefix=None,
        style=None, formatter=None, threshold=None,
        edgeitems=None, sign=None, floatmode=None, suffix=None,
        *, legacy=None):
    return (a,)


@array_function_dispatch(_array2string_dispatcher, module='numpy')
def array2string(a, max_line_width=None, precision=None,
                 suppress_small=None, separator=' ', prefix="",
                 style=np._NoValue, formatter=None, threshold=None,
                 edgeitems=None, sign=None, floatmode=None, suffix="",
                 *, legacy=None):
    """
    Return a string representation of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int or None, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.
    separator : str, optional
        Inserted between elements.
    prefix : str, optional
    suffix : str, optional
        The length of the prefix and suffix strings are used to respectively
        align and wrap the output. An array is typically printed as::

          prefix + array2string(a) + suffix

        The output is left-padded by the length of the prefix string, and
        wrapping is forced at the column ``max_line_width - len(suffix)``.
        It should be noted that the content of prefix and suffix strings are
        not included in the output.
    style : _NoValue, optional
        Has no effect, do not use.

        .. deprecated:: 1.14.0
    formatter : dict of callables, optional
        If not None, the keys should indicate the type(s) that the respective
        formatting function applies to.  Callables should return a string.
        Types that are not specified (by their corresponding keys) are handled
        by the default formatters.  Individual types for which a formatter
        can be set are:

        - 'bool'
        - 'int'
        - 'timedelta' : a `numpy.timedelta64`
        - 'datetime' : a `numpy.datetime64`
        - 'float'
        - 'longfloat' : 128-bit floats
        - 'complexfloat'
        - 'longcomplexfloat' : composed of two 128-bit floats
        - 'void' : type `numpy.void`
        - 'numpystr' : types `numpy.bytes_` and `numpy.str_`

        Other keys that can be used to set a group of types at once are:

        - 'all' : sets all types
        - 'int_kind' : sets 'int'
        - 'float_kind' : sets 'float' and 'longfloat'
        - 'complex_kind' : sets 'complexfloat' and 'longcomplexfloat'
        - 'str_kind' : sets 'numpystr'
    threshold : int, optional
        Total number of array elements which trigger summarization
        rather than full repr.
        Defaults to ``numpy.get_printoptions()['threshold']``.
    edgeitems : int, optional
        Number of array items in summary at beginning and end of
        each dimension.
        Defaults to ``numpy.get_printoptions()['edgeitems']``.
    sign : string, either '-', '+', or ' ', optional
        Controls printing of the sign of floating-point types. If '+', always
        print the sign of positive values. If ' ', always prints a space
        (whitespace character) in the sign position of positive values.  If
        '-', omit the sign character of positive values.
        Defaults to ``numpy.get_printoptions()['sign']``.

        .. versionchanged:: 2.0
             The sign parameter can now be an integer type, previously
             types were floating-point types.

    floatmode : str, optional
        Controls the interpretation of the `precision` option for
        floating-point types.
        Defaults to ``numpy.get_printoptions()['floatmode']``.
        Can take the following values:

        - 'fixed': Always print exactly `precision` fractional digits,
          even if this would print more or fewer digits than
          necessary to specify the value uniquely.
        - 'unique': Print the minimum number of fractional digits necessary
          to represent each value uniquely. Different elements may
          have a different number of digits.  The value of the
          `precision` option is ignored.
        - 'maxprec': Print at most `precision` fractional digits, but if
          an element can be uniquely represented with fewer digits
          only print it with that many.
        - 'maxprec_equal': Print at most `precision` fractional digits,
          but if every element in the array can be uniquely
          represented with an equal number of fewer digits, use that
          many digits for all elements.
    legacy : string or `False`, optional
        If set to the string ``'1.13'`` enables 1.13 legacy printing mode. This
        approximates numpy 1.13 print output by including a space in the sign
        position of floats and different behavior for 0d arrays. If set to
        `False`, disables legacy mode. Unrecognized strings will be ignored
        with a warning for forward compatibility.

    Returns
    -------
    array_str : str
        String representation of the array.

    Raises
    ------
    TypeError
        if a callable in `formatter` does not return a string.

    See Also
    --------
    array_str, array_repr, set_printoptions, get_printoptions

    Notes
    -----
    If a formatter is specified for a certain type, the `precision` keyword is
    ignored for that type.

    This is a very flexible function; `array_repr` and `array_str` are using
    `array2string` internally so keywords with the same name should work
    identically in all three functions.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1e-16,1,2,3])
    >>> np.array2string(x, precision=2, separator=',',
    ...                       suppress_small=True)
    '[0.,1.,2.,3.]'

    >>> x  = np.arange(3.)
    >>> np.array2string(x, formatter={'float_kind':lambda x: "%.2f" % x})
    '[0.00 1.00 2.00]'

    >>> x  = np.arange(3)
    >>> np.array2string(x, formatter={'int':lambda x: hex(x)})
    '[0x0 0x1 0x2]'

    """

    overrides = _make_options_dict(precision, threshold, edgeitems,
                                   max_line_width, suppress_small, None, None,
                                   sign, formatter, floatmode, legacy)
    options = format_options.get().copy()
    options.update(overrides)

    if options['legacy'] <= 113:
        if style is np._NoValue:
            style = repr

        if a.shape == () and a.dtype.names is None:
            return style(a.item())
    elif style is not np._NoValue:
        # Deprecation 11-9-2017  v1.14
        warnings.warn("'style' argument is deprecated and no longer functional"
                      " except in 1.13 'legacy' mode",
                      DeprecationWarning, stacklevel=2)

    if options['legacy'] > 113:
        options['linewidth'] -= len(suffix)

    # treat as a null array if any of shape elements == 0
    if a.size == 0:
        return "[]"

    return _array2string(a, options, separator, prefix)


def _extendLine(s, line, word, line_width, next_line_prefix, legacy):
    needs_wrap = len(line) + len(word) > line_width
    if legacy > 113:
        # don't wrap lines if it won't help
        if len(line) <= len(next_line_prefix):
            needs_wrap = False

    if needs_wrap:
        s += line.rstrip() + "\n"
        line = next_line_prefix
    line += word
    return s, line


def _extendLine_pretty(s, line, word, line_width, next_line_prefix, legacy):
    """
    Extends line with nicely formatted (possibly multi-line) string ``word``.
    """
    words = word.splitlines()
    if len(words) == 1 or legacy <= 113:
        return _extendLine(s, line, word, line_width, next_line_prefix, legacy)

    max_word_length = max(len(word) for word in words)
    if (len(line) + max_word_length > line_width and
            len(line) > len(next_line_prefix)):
        s += line.rstrip() + '\n'
        line = next_line_prefix + words[0]
        indent = next_line_prefix
    else:
        indent = len(line) * ' '
        line += words[0]

    for word in words[1::]:
        s += line.rstrip() + '\n'
        line = indent + word

    suffix_length = max_word_length - len(words[-1])
    line += suffix_length * ' '

    return s, line

def _formatArray(a, format_function, line_width, next_line_prefix,
                 separator, edge_items, summary_insert, legacy):
    """formatArray is designed for two modes of operation:

    1. Full output

    2. Summarized output

    """
    def recurser(index, hanging_indent, curr_width):
        """
        By using this local function, we don't need to recurse with all the
        arguments. Since this function is not created recursively, the cost is
        not significant
        """
        axis = len(index)
        axes_left = a.ndim - axis

        if axes_left == 0:
            return format_function(a[index])

        # when recursing, add a space to align with the [ added, and reduce the
        # length of the line by 1
        next_hanging_indent = hanging_indent + ' '
        if legacy <= 113:
            next_width = curr_width
        else:
            next_width = curr_width - len(']')

        a_len = a.shape[axis]
        show_summary = summary_insert and 2 * edge_items < a_len
        if show_summary:
            leading_items = edge_items
            trailing_items = edge_items
        else:
            leading_items = 0
            trailing_items = a_len

        # stringify the array with the hanging indent on the first line too
        s = ''

        # last axis (rows) - wrap elements if they would not fit on one line
        if axes_left == 1:
            # the length up until the beginning of the separator / bracket
            if legacy <= 113:
                elem_width = curr_width - len(separator.rstrip())
            else:
                elem_width = curr_width - max(
                    len(separator.rstrip()), len(']')
                )

            line = hanging_indent
            for i in range(leading_items):
                word = recurser(index + (i,), next_hanging_indent, next_width)
                s, line = _extendLine_pretty(
                    s, line, word, elem_width, hanging_indent, legacy)
                line += separator

            if show_summary:
                s, line = _extendLine(
                    s, line, summary_insert, elem_width, hanging_indent, legacy
                )
                if legacy <= 113:
                    line += ", "
                else:
                    line += separator

            for i in range(trailing_items, 1, -1):
                word = recurser(index + (-i,), next_hanging_indent, next_width)
                s, line = _extendLine_pretty(
                    s, line, word, elem_width, hanging_indent, legacy)
                line += separator

            if legacy <= 113:
                # width of the separator is not considered on 1.13
                elem_width = curr_width
            word = recurser(index + (-1,), next_hanging_indent, next_width)
            s, line = _extendLine_pretty(
                s, line, word, elem_width, hanging_indent, legacy)

            s += line

        # other axes - insert newlines between rows
        else:
            s = ''
            line_sep = separator.rstrip() + '\n' * (axes_left - 1)

            for i in range(leading_items):
                nested = recurser(
                    index + (i,), next_hanging_indent, next_width
                )
                s += hanging_indent + nested + line_sep

            if show_summary:
                if legacy <= 113:
                    # trailing space, fixed nbr of newlines,
                    # and fixed separator
                    s += hanging_indent + summary_insert + ", \n"
                else:
                    s += hanging_indent + summary_insert + line_sep

            for i in range(trailing_items, 1, -1):
                nested = recurser(index + (-i,), next_hanging_indent,
                                  next_width)
                s += hanging_indent + nested + line_sep

            nested = recurser(index + (-1,), next_hanging_indent, next_width)
            s += hanging_indent + nested

        # remove the hanging indent, and wrap in []
        s = '[' + s[len(hanging_indent):] + ']'
        return s

    try:
        # invoke the recursive part with an initial index and prefix
        return recurser(index=(),
                        hanging_indent=next_line_prefix,
                        curr_width=line_width)
    finally:
        # recursive closures have a cyclic reference to themselves, which
        # requires gc to collect (gh-10620). To avoid this problem, for
        # performance and PyPy friendliness, we break the cycle:
        recurser = None

def _none_or_positive_arg(x, name):
    if x is None:
        return -1
    if x < 0:
        raise ValueError(f"{name} must be >= 0")
    return x

class FloatingFormat:
    """ Formatter for subtypes of np.floating """
    def __init__(self, data, precision, floatmode, suppress_small, sign=False,
                 *, legacy=None):
        # for backcompatibility, accept bools
        if isinstance(sign, bool):
            sign = '+' if sign else '-'

        self._legacy = legacy
        if self._legacy <= 113:
            # when not 0d, legacy does not support '-'
            if data.shape != () and sign == '-':
                sign = ' '

        self.floatmode = floatmode
        if floatmode == 'unique':
            self.precision = None
        else:
            self.precision = precision

        self.precision = _none_or_positive_arg(self.precision, 'precision')

        self.suppress_small = suppress_small
        self.sign = sign
        self.exp_format = False
        self.large_exponent = False
        self.fillFormat(data)

    def fillFormat(self, data):
        # only the finite values are used to compute the number of digits
        finite_vals = data[isfinite(data)]

        # choose exponential mode based on the non-zero finite values:
        abs_non_zero = absolute(finite_vals[finite_vals != 0])
        if len(abs_non_zero) != 0:
            max_val = np.max(abs_non_zero)
            min_val = np.min(abs_non_zero)
            if self._legacy <= 202:
                exp_cutoff_max = 1.e8
            else:
                # consider data type while deciding the max cutoff for exp format
                exp_cutoff_max = 10.**min(8, np.finfo(data.dtype).precision)
            with errstate(over='ignore'):  # division can overflow
                if max_val >= exp_cutoff_max or (not self.suppress_small and
                        (min_val < 0.0001 or max_val / min_val > 1000.)):
                    self.exp_format = True

        # do a first pass of printing all the numbers, to determine sizes
        if len(finite_vals) == 0:
            self.pad_left = 0
            self.pad_right = 0
            self.trim = '.'
            self.exp_size = -1
            self.unique = True
            self.min_digits = None
        elif self.exp_format:
            trim, unique = '.', True
            if self.floatmode == 'fixed' or self._legacy <= 113:
                trim, unique = 'k', False
            strs = (dragon4_scientific(x, precision=self.precision,
                               unique=unique, trim=trim, sign=self.sign == '+')
                    for x in finite_vals)
            frac_strs, _, exp_strs = zip(*(s.partition('e') for s in strs))
            int_part, frac_part = zip(*(s.split('.') for s in frac_strs))
            self.exp_size = max(len(s) for s in exp_strs) - 1

            self.trim = 'k'
            self.precision = max(len(s) for s in frac_part)
            self.min_digits = self.precision
            self.unique = unique

            # for back-compat with np 1.13, use 2 spaces & sign and full prec
            if self._legacy <= 113:
                self.pad_left = 3
            else:
                # this should be only 1 or 2. Can be calculated from sign.
                self.pad_left = max(len(s) for s in int_part)
            # pad_right is only needed for nan length calculation
            self.pad_right = self.exp_size + 2 + self.precision
        else:
            trim, unique = '.', True
            if self.floatmode == 'fixed':
                trim, unique = 'k', False
            strs = (dragon4_positional(x, precision=self.precision,
                                       fractional=True,
                                       unique=unique, trim=trim,
                                       sign=self.sign == '+')
                    for x in finite_vals)
            int_part, frac_part = zip(*(s.split('.') for s in strs))
            if self._legacy <= 113:
                self.pad_left = 1 + max(len(s.lstrip('-+')) for s in int_part)
            else:
                self.pad_left = max(len(s) for s in int_part)
            self.pad_right = max(len(s) for s in frac_part)
            self.exp_size = -1
            self.unique = unique

            if self.floatmode in ['fixed', 'maxprec_equal']:
                self.precision = self.min_digits = self.pad_right
                self.trim = 'k'
            else:
                self.trim = '.'
                self.min_digits = 0

        if self._legacy > 113:
            # account for sign = ' ' by adding one to pad_left
            if self.sign == ' ' and not any(np.signbit(finite_vals)):
                self.pad_left += 1

        # if there are non-finite values, may need to increase pad_left
        if data.size != finite_vals.size:
            neginf = self.sign != '-' or any(data[isinf(data)] < 0)
            offset = self.pad_right + 1  # +1 for decimal pt
            current_options = format_options.get()
            self.pad_left = max(
                self.pad_left, len(current_options['nanstr']) - offset,
                len(current_options['infstr']) + neginf - offset
            )

    def __call__(self, x):
        if not np.isfinite(x):
            with errstate(invalid='ignore'):
                current_options = format_options.get()
                if np.isnan(x):
                    sign = '+' if self.sign == '+' else ''
                    ret = sign + current_options['nanstr']
                else:  # isinf
                    sign = '-' if x < 0 else '+' if self.sign == '+' else ''
                    ret = sign + current_options['infstr']
                return ' ' * (
                    self.pad_left + self.pad_right + 1 - len(ret)
                ) + ret

        if self.exp_format:
            return dragon4_scientific(x,
                                      precision=self.precision,
                                      min_digits=self.min_digits,
                                      unique=self.unique,
                                      trim=self.trim,
                                      sign=self.sign == '+',
                                      pad_left=self.pad_left,
                                      exp_digits=self.exp_size)
        else:
            return dragon4_positional(x,
                                      precision=self.precision,
                                      min_digits=self.min_digits,
                                      unique=self.unique,
                                      fractional=True,
                                      trim=self.trim,
                                      sign=self.sign == '+',
                                      pad_left=self.pad_left,
                                      pad_right=self.pad_right)


@set_module('numpy')
def format_float_scientific(x, precision=None, unique=True, trim='k',
                            sign=False, pad_left=None, exp_digits=None,
                            min_digits=None):
    """
    Format a floating-point scalar as a decimal string in scientific notation.

    Provides control over rounding, trimming and padding. Uses and assumes
    IEEE unbiased rounding. Uses the "Dragon4" algorithm.

    Parameters
    ----------
    x : python float or numpy floating scalar
        Value to format.
    precision : non-negative integer or None, optional
        Maximum number of digits to print. May be None if `unique` is
        `True`, but must be an integer if unique is `False`.
    unique : boolean, optional
        If `True`, use a digit-generation strategy which gives the shortest
        representation which uniquely identifies the floating-point number from
        other values of the same type, by judicious rounding. If `precision`
        is given fewer digits than necessary can be printed. If `min_digits`
        is given more can be printed, in which cases the last digit is rounded
        with unbiased rounding.
        If `False`, digits are generated as if printing an infinite-precision
        value and stopping after `precision` digits, rounding the remaining
        value with unbiased rounding
    trim : one of 'k', '.', '0', '-', optional
        Controls post-processing trimming of trailing digits, as follows:

        * 'k' : keep trailing zeros, keep decimal point (no trimming)
        * '.' : trim all trailing zeros, leave decimal point
        * '0' : trim all but the zero before the decimal point. Insert the
          zero if it is missing.
        * '-' : trim trailing zeros and any trailing decimal point
    sign : boolean, optional
        Whether to show the sign for positive values.
    pad_left : non-negative integer, optional
        Pad the left side of the string with whitespace until at least that
        many characters are to the left of the decimal point.
    exp_digits : non-negative integer, optional
        Pad the exponent with zeros until it contains at least this
        many digits. If omitted, the exponent will be at least 2 digits.
    min_digits : non-negative integer or None, optional
        Minimum number of digits to print. This only has an effect for
        `unique=True`. In that case more digits than necessary to uniquely
        identify the value may be printed and rounded unbiased.

        .. versionadded:: 1.21.0

    Returns
    -------
    rep : string
        The string representation of the floating point value

    See Also
    --------
    format_float_positional

    Examples
    --------
    >>> import numpy as np
    >>> np.format_float_scientific(np.float32(np.pi))
    '3.1415927e+00'
    >>> s = np.float32(1.23e24)
    >>> np.format_float_scientific(s, unique=False, precision=15)
    '1.230000071797338e+24'
    >>> np.format_float_scientific(s, exp_digits=4)
    '1.23e+0024'
    """
    precision = _none_or_positive_arg(precision, 'precision')
    pad_left = _none_or_positive_arg(pad_left, 'pad_left')
    exp_digits = _none_or_positive_arg(exp_digits, 'exp_digits')
    min_digits = _none_or_positive_arg(min_digits, 'min_digits')
    if min_digits > 0 and precision > 0 and min_digits > precision:
        raise ValueError("min_digits must be less than or equal to precision")
    return dragon4_scientific(x, precision=precision, unique=unique,
                              trim=trim, sign=sign, pad_left=pad_left,
                              exp_digits=exp_digits, min_digits=min_digits)


@set_module('numpy')
def format_float_positional(x, precision=None, unique=True,
                            fractional=True, trim='k', sign=False,
                            pad_left=None, pad_right=None, min_digits=None):
    """
    Format a floating-point scalar as a decimal string in positional notation.

    Provides control over rounding, trimming and padding. Uses and assumes
    IEEE unbiased rounding. Uses the "Dragon4" algorithm.

    Parameters
    ----------
    x : python float or numpy floating scalar
        Value to format.
    precision : non-negative integer or None, optional
        Maximum number of digits to print. May be None if `unique` is
        `True`, but must be an integer if unique is `False`.
    unique : boolean, optional
        If `True`, use a digit-generation strategy which gives the shortest
        representation which uniquely identifies the floating-point number from
        other values of the same type, by judicious rounding. If `precision`
        is given fewer digits than necessary can be printed, or if `min_digits`
        is given more can be printed, in which cases the last digit is rounded
        with unbiased rounding.
        If `False`, digits are generated as if printing an infinite-precision
        value and stopping after `precision` digits, rounding the remaining
        value with unbiased rounding
    fractional : boolean, optional
        If `True`, the cutoffs of `precision` and `min_digits` refer to the
        total number of digits after the decimal point, including leading
        zeros.
        If `False`, `precision` and `min_digits` refer to the total number of
        significant digits, before or after the decimal point, ignoring leading
        zeros.
    trim : one of 'k', '.', '0', '-', optional
        Controls post-processing trimming of trailing digits, as follows:

        * 'k' : keep trailing zeros, keep decimal point (no trimming)
        * '.' : trim all trailing zeros, leave decimal point
        * '0' : trim all but the zero before the decimal point. Insert the
          zero if it is missing.
        * '-' : trim trailing zeros and any trailing decimal point
    sign : boolean, optional
        Whether to show the sign for positive values.
    pad_left : non-negative integer, optional
        Pad the left side of the string with whitespace until at least that
        many characters are to the left of the decimal point.
    pad_right : non-negative integer, optional
        Pad the right side of the string with whitespace until at least that
        many characters are to the right of the decimal point.
    min_digits : non-negative integer or None, optional
        Minimum number of digits to print. Only has an effect if `unique=True`
        in which case additional digits past those necessary to uniquely
        identify the value may be printed, rounding the last additional digit.

        .. versionadded:: 1.21.0

    Returns
    -------
    rep : string
        The string representation of the floating point value

    See Also
    --------
    format_float_scientific

    Examples
    --------
    >>> import numpy as np
    >>> np.format_float_positional(np.float32(np.pi))
    '3.1415927'
    >>> np.format_float_positional(np.float16(np.pi))
    '3.14'
    >>> np.format_float_positional(np.float16(0.3))
    '0.3'
    >>> np.format_float_positional(np.float16(0.3), unique=False, precision=10)
    '0.3000488281'
    """
    precision = _none_or_positive_arg(precision, 'precision')
    pad_left = _none_or_positive_arg(pad_left, 'pad_left')
    pad_right = _none_or_positive_arg(pad_right, 'pad_right')
    min_digits = _none_or_positive_arg(min_digits, 'min_digits')
    if not fractional and precision == 0:
        raise ValueError("precision must be greater than 0 if "
                         "fractional=False")
    if min_digits > 0 and precision > 0 and min_digits > precision:
        raise ValueError("min_digits must be less than or equal to precision")
    return dragon4_positional(x, precision=precision, unique=unique,
                              fractional=fractional, trim=trim,
                              sign=sign, pad_left=pad_left,
                              pad_right=pad_right, min_digits=min_digits)

class IntegerFormat:
    def __init__(self, data, sign='-'):
        if data.size > 0:
            data_max = np.max(data)
            data_min = np.min(data)
            data_max_str_len = len(str(data_max))
            if sign == ' ' and data_min < 0:
                sign = '-'
            if data_max >= 0 and sign in "+ ":
                data_max_str_len += 1
            max_str_len = max(data_max_str_len,
                              len(str(data_min)))
        else:
            max_str_len = 0
        self.format = f'{{:{sign}{max_str_len}d}}'

    def __call__(self, x):
        return self.format.format(x)

class BoolFormat:
    def __init__(self, data, **kwargs):
        # add an extra space so " True" and "False" have the same length and
        # array elements align nicely when printed, except in 0d arrays
        self.truestr = ' True' if data.shape != () else 'True'

    def __call__(self, x):
        return self.truestr if x else "False"


class ComplexFloatingFormat:
    """ Formatter for subtypes of np.complexfloating """
    def __init__(self, x, precision, floatmode, suppress_small,
                 sign=False, *, legacy=None):
        # for backcompatibility, accept bools
        if isinstance(sign, bool):
            sign = '+' if sign else '-'

        floatmode_real = floatmode_imag = floatmode
        if legacy <= 113:
            floatmode_real = 'maxprec_equal'
            floatmode_imag = 'maxprec'

        self.real_format = FloatingFormat(
            x.real, precision, floatmode_real, suppress_small,
            sign=sign, legacy=legacy
        )
        self.imag_format = FloatingFormat(
            x.imag, precision, floatmode_imag, suppress_small,
            sign='+', legacy=legacy
        )

    def __call__(self, x):
        r = self.real_format(x.real)
        i = self.imag_format(x.imag)

        # add the 'j' before the terminal whitespace in i
        sp = len(i.rstrip())
        i = i[:sp] + 'j' + i[sp:]

        return r + i


class _TimelikeFormat:
    def __init__(self, data):
        non_nat = data[~isnat(data)]
        if len(non_nat) > 0:
            # Max str length of non-NaT elements
            max_str_len = max(len(self._format_non_nat(np.max(non_nat))),
                              len(self._format_non_nat(np.min(non_nat))))
        else:
            max_str_len = 0
        if len(non_nat) < data.size:
            # data contains a NaT
            max_str_len = max(max_str_len, 5)
        self._format = f'%{max_str_len}s'
        self._nat = "'NaT'".rjust(max_str_len)

    def _format_non_nat(self, x):
        # override in subclass
        raise NotImplementedError

    def __call__(self, x):
        if isnat(x):
            return self._nat
        else:
            return self._format % self._format_non_nat(x)


class DatetimeFormat(_TimelikeFormat):
    def __init__(self, x, unit=None, timezone=None, casting='same_kind',
                 legacy=False):
        # Get the unit from the dtype
        if unit is None:
            if x.dtype.kind == 'M':
                unit = datetime_data(x.dtype)[0]
            else:
                unit = 's'

        if timezone is None:
            timezone = 'naive'
        self.timezone = timezone
        self.unit = unit
        self.casting = casting
        self.legacy = legacy

        # must be called after the above are configured
        super().__init__(x)

    def __call__(self, x):
        if self.legacy <= 113:
            return self._format_non_nat(x)
        return super().__call__(x)

    def _format_non_nat(self, x):
        return "'%s'" % datetime_as_string(x,
                                    unit=self.unit,
                                    timezone=self.timezone,
                                    casting=self.casting)


class TimedeltaFormat(_TimelikeFormat):
    def _format_non_nat(self, x):
        return str(x.astype('i8'))


class SubArrayFormat:
    def __init__(self, format_function, **options):
        self.format_function = format_function
        self.threshold = options['threshold']
        self.edge_items = options['edgeitems']

    def __call__(self, a):
        self.summary_insert = "..." if a.size > self.threshold else ""
        return self.format_array(a)

    def format_array(self, a):
        if np.ndim(a) == 0:
            return self.format_function(a)

        if self.summary_insert and a.shape[0] > 2 * self.edge_items:
            formatted = (
                [self.format_array(a_) for a_ in a[:self.edge_items]]
                + [self.summary_insert]
                + [self.format_array(a_) for a_ in a[-self.edge_items:]]
            )
        else:
            formatted = [self.format_array(a_) for a_ in a]

        return "[" + ", ".join(formatted) + "]"


class StructuredVoidFormat:
    """
    Formatter for structured np.void objects.

    This does not work on structured alias types like
    np.dtype(('i4', 'i2,i2')), as alias scalars lose their field information,
    and the implementation relies upon np.void.__getitem__.
    """
    def __init__(self, format_functions):
        self.format_functions = format_functions

    @classmethod
    def from_data(cls, data, **options):
        """
        This is a second way to initialize StructuredVoidFormat,
        using the raw data as input. Added to avoid changing
        the signature of __init__.
        """
        format_functions = []
        for field_name in data.dtype.names:
            format_function = _get_format_function(data[field_name], **options)
            if data.dtype[field_name].shape != ():
                format_function = SubArrayFormat(format_function, **options)
            format_functions.append(format_function)
        return cls(format_functions)

    def __call__(self, x):
        str_fields = [
            format_function(field)
            for field, format_function in zip(x, self.format_functions)
        ]
        if len(str_fields) == 1:
            return f"({str_fields[0]},)"
        else:
            return f"({', '.join(str_fields)})"


def _void_scalar_to_string(x, is_repr=True):
    """
    Implements the repr for structured-void scalars. It is called from the
    scalartypes.c.src code, and is placed here because it uses the elementwise
    formatters defined above.
    """
    options = format_options.get().copy()

    if options["legacy"] <= 125:
        return StructuredVoidFormat.from_data(array(x), **options)(x)

    if options.get('formatter') is None:
        options['formatter'] = {}
    options['formatter'].setdefault('float_kind', str)
    val_repr = StructuredVoidFormat.from_data(array(x), **options)(x)
    if not is_repr:
        return val_repr
    cls = type(x)
    cls_fqn = cls.__module__.replace("numpy", "np") + "." + cls.__name__
    void_dtype = np.dtype((np.void, x.dtype))
    return f"{cls_fqn}({val_repr}, dtype={void_dtype!s})"


_typelessdata = [int_, float64, complex128, _nt.bool]


def dtype_is_implied(dtype):
    """
    Determine if the given dtype is implied by the representation
    of its values.

    Parameters
    ----------
    dtype : dtype
        Data type

    Returns
    -------
    implied : bool
        True if the dtype is implied by the representation of its values.

    Examples
    --------
    >>> import numpy as np
    >>> np._core.arrayprint.dtype_is_implied(int)
    True
    >>> np.array([1, 2, 3], int)
    array([1, 2, 3])
    >>> np._core.arrayprint.dtype_is_implied(np.int8)
    False
    >>> np.array([1, 2, 3], np.int8)
    array([1, 2, 3], dtype=int8)
    """
    dtype = np.dtype(dtype)
    if format_options.get()['legacy'] <= 113 and dtype.type == np.bool:
        return False

    # not just void types can be structured, and names are not part of the repr
    if dtype.names is not None:
        return False

    # should care about endianness *unless size is 1* (e.g., int8, bool)
    if not dtype.isnative:
        return False

    return dtype.type in _typelessdata


def dtype_short_repr(dtype):
    """
    Convert a dtype to a short form which evaluates to the same dtype.

    The intent is roughly that the following holds

    >>> from numpy import *
    >>> dt = np.int64([1, 2]).dtype
    >>> assert eval(dtype_short_repr(dt)) == dt
    """
    if type(dtype).__repr__ != np.dtype.__repr__:
        # TODO: Custom repr for user DTypes, logic should likely move.
        return repr(dtype)
    if dtype.names is not None:
        # structured dtypes give a list or tuple repr
        return str(dtype)
    elif issubclass(dtype.type, flexible):
        # handle these separately so they don't give garbage like str256
        return f"'{str(dtype)}'"

    typename = dtype.name
    if not dtype.isnative:
        # deal with cases like dtype('<u2') that are identical to an
        # established dtype (in this case uint16)
        # except that they have a different endianness.
        return f"'{str(dtype)}'"
    # quote typenames which can't be represented as python variable names
    if typename and not (typename[0].isalpha() and typename.isalnum()):
        typename = repr(typename)
    return typename


def _array_repr_implementation(
        arr, max_line_width=None, precision=None, suppress_small=None,
        array2string=array2string):
    """Internal version of array_repr() that allows overriding array2string."""
    current_options = format_options.get()
    override_repr = current_options["override_repr"]
    if override_repr is not None:
        return override_repr(arr)

    if max_line_width is None:
        max_line_width = current_options['linewidth']

    if type(arr) is not ndarray:
        class_name = type(arr).__name__
    else:
        class_name = "array"

    prefix = class_name + "("
    if (current_options['legacy'] <= 113 and
            arr.shape == () and not arr.dtype.names):
        lst = repr(arr.item())
    else:
        lst = array2string(arr, max_line_width, precision, suppress_small,
                           ', ', prefix, suffix=")")

    # Add dtype and shape information if these cannot be inferred from
    # the array string.
    extras = []
    if ((arr.size == 0 and arr.shape != (0,))
            or (current_options['legacy'] > 210
            and arr.size > current_options['threshold'])):
        extras.append(f"shape={arr.shape}")
    if not dtype_is_implied(arr.dtype) or arr.size == 0:
        extras.append(f"dtype={dtype_short_repr(arr.dtype)}")

    if not extras:
        return prefix + lst + ")"

    arr_str = prefix + lst + ","
    extra_str = ", ".join(extras) + ")"
    # compute whether we should put extras on a new line: Do so if adding the
    # extras would extend the last line past max_line_width.
    # Note: This line gives the correct result even when rfind returns -1.
    last_line_len = len(arr_str) - (arr_str.rfind('\n') + 1)
    spacer = " "
    if current_options['legacy'] <= 113:
        if issubclass(arr.dtype.type, flexible):
            spacer = '\n' + ' ' * len(prefix)
    elif last_line_len + len(extra_str) + 1 > max_line_width:
        spacer = '\n' + ' ' * len(prefix)

    return arr_str + spacer + extra_str


def _array_repr_dispatcher(
        arr, max_line_width=None, precision=None, suppress_small=None):
    return (arr,)


@array_function_dispatch(_array_repr_dispatcher, module='numpy')
def array_repr(arr, max_line_width=None, precision=None, suppress_small=None):
    """
    Return the string representation of an array.

    Parameters
    ----------
    arr : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.

    Returns
    -------
    string : str
      The string representation of an array.

    See Also
    --------
    array_str, array2string, set_printoptions

    Examples
    --------
    >>> import numpy as np
    >>> np.array_repr(np.array([1,2]))
    'array([1, 2])'
    >>> np.array_repr(np.ma.array([0.]))
    'MaskedArray([0.])'
    >>> np.array_repr(np.array([], np.int32))
    'array([], dtype=int32)'

    >>> x = np.array([1e-6, 4e-7, 2, 3])
    >>> np.array_repr(x, precision=6, suppress_small=True)
    'array([0.000001,  0.      ,  2.      ,  3.      ])'

    """
    return _array_repr_implementation(
        arr, max_line_width, precision, suppress_small)


@_recursive_guard()
def _guarded_repr_or_str(v):
    if isinstance(v, bytes):
        return repr(v)
    return str(v)


def _array_str_implementation(
        a, max_line_width=None, precision=None, suppress_small=None,
        array2string=array2string):
    """Internal version of array_str() that allows overriding array2string."""
    if (format_options.get()['legacy'] <= 113 and
            a.shape == () and not a.dtype.names):
        return str(a.item())

    # the str of 0d arrays is a special case: It should appear like a scalar,
    # so floats are not truncated by `precision`, and strings are not wrapped
    # in quotes. So we return the str of the scalar value.
    if a.shape == ():
        # obtain a scalar and call str on it, avoiding problems for subclasses
        # for which indexing with () returns a 0d instead of a scalar by using
        # ndarray's getindex. Also guard against recursive 0d object arrays.
        return _guarded_repr_or_str(np.ndarray.__getitem__(a, ()))

    return array2string(a, max_line_width, precision, suppress_small, ' ', "")


def _array_str_dispatcher(
        a, max_line_width=None, precision=None, suppress_small=None):
    return (a,)


@array_function_dispatch(_array_str_dispatcher, module='numpy')
def array_str(a, max_line_width=None, precision=None, suppress_small=None):
    """
    Return a string representation of the data in an array.

    The data in the array is returned as a single string.  This function is
    similar to `array_repr`, the difference being that `array_repr` also
    returns information on the kind of array and its data type.

    Parameters
    ----------
    a : ndarray
        Input array.
    max_line_width : int, optional
        Inserts newlines if text is longer than `max_line_width`.
        Defaults to ``numpy.get_printoptions()['linewidth']``.
    precision : int, optional
        Floating point precision.
        Defaults to ``numpy.get_printoptions()['precision']``.
    suppress_small : bool, optional
        Represent numbers "very close" to zero as zero; default is False.
        Very close is defined by precision: if the precision is 8, e.g.,
        numbers smaller (in absolute value) than 5e-9 are represented as
        zero.
        Defaults to ``numpy.get_printoptions()['suppress']``.

    See Also
    --------
    array2string, array_repr, set_printoptions

    Examples
    --------
    >>> import numpy as np
    >>> np.array_str(np.arange(3))
    '[0 1 2]'

    """
    return _array_str_implementation(
        a, max_line_width, precision, suppress_small)


# needed if __array_function__ is disabled
_array2string_impl = getattr(array2string, '__wrapped__', array2string)
_default_array_str = functools.partial(_array_str_implementation,
                                       array2string=_array2string_impl)
_default_array_repr = functools.partial(_array_repr_implementation,
                                        array2string=_array2string_impl)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\ellipse.py ===
"""Elliptical geometrical entities.

Contains
* Ellipse
* Circle

"""

from sympy.core.expr import Expr
from sympy.core.relational import Eq
from sympy.core import S, pi, sympify
from sympy.core.evalf import N
from sympy.core.parameters import global_parameters
from sympy.core.logic import fuzzy_bool
from sympy.core.numbers import Rational, oo
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, uniquely_named_symbol, _symbol
from sympy.simplify.simplify import simplify
from sympy.simplify.trigsimp import trigsimp
from sympy.functions.elementary.miscellaneous import sqrt, Max
from sympy.functions.elementary.trigonometric import cos, sin
from sympy.functions.special.elliptic_integrals import elliptic_e
from .entity import GeometryEntity, GeometrySet
from .exceptions import GeometryError
from .line import Line, Segment, Ray2D, Segment2D, Line2D, LinearEntity3D
from .point import Point, Point2D, Point3D
from .util import idiff, find
from sympy.polys import DomainError, Poly, PolynomialError
from sympy.polys.polyutils import _not_a_coeff, _nsort
from sympy.solvers import solve
from sympy.solvers.solveset import linear_coeffs
from sympy.utilities.misc import filldedent, func_name

from mpmath.libmp.libmpf import prec_to_dps

import random

x, y = [Dummy('ellipse_dummy', real=True) for i in range(2)]


class Ellipse(GeometrySet):
    """An elliptical GeometryEntity.

    Parameters
    ==========

    center : Point, optional
        Default value is Point(0, 0)
    hradius : number or SymPy expression, optional
    vradius : number or SymPy expression, optional
    eccentricity : number or SymPy expression, optional
        Two of `hradius`, `vradius` and `eccentricity` must be supplied to
        create an Ellipse. The third is derived from the two supplied.

    Attributes
    ==========

    center
    hradius
    vradius
    area
    circumference
    eccentricity
    periapsis
    apoapsis
    focus_distance
    foci

    Raises
    ======

    GeometryError
        When `hradius`, `vradius` and `eccentricity` are incorrectly supplied
        as parameters.
    TypeError
        When `center` is not a Point.

    See Also
    ========

    Circle

    Notes
    -----
    Constructed from a center and two radii, the first being the horizontal
    radius (along the x-axis) and the second being the vertical radius (along
    the y-axis).

    When symbolic value for hradius and vradius are used, any calculation that
    refers to the foci or the major or minor axis will assume that the ellipse
    has its major radius on the x-axis. If this is not true then a manual
    rotation is necessary.

    Examples
    ========

    >>> from sympy import Ellipse, Point, Rational
    >>> e1 = Ellipse(Point(0, 0), 5, 1)
    >>> e1.hradius, e1.vradius
    (5, 1)
    >>> e2 = Ellipse(Point(3, 1), hradius=3, eccentricity=Rational(4, 5))
    >>> e2
    Ellipse(Point2D(3, 1), 3, 9/5)

    """

    def __contains__(self, o):
        if isinstance(o, Point):
            res = self.equation(x, y).subs({x: o.x, y: o.y})
            return trigsimp(simplify(res)) is S.Zero
        elif isinstance(o, Ellipse):
            return self == o
        return False

    def __eq__(self, o):
        """Is the other GeometryEntity the same as this ellipse?"""
        return isinstance(o, Ellipse) and (self.center == o.center and
                                           self.hradius == o.hradius and
                                           self.vradius == o.vradius)

    def __hash__(self):
        return super().__hash__()

    def __new__(
        cls, center=None, hradius=None, vradius=None, eccentricity=None, **kwargs):

        hradius = sympify(hradius)
        vradius = sympify(vradius)

        if center is None:
            center = Point(0, 0)
        else:
            if len(center) != 2:
                raise ValueError('The center of "{}" must be a two dimensional point'.format(cls))
            center = Point(center, dim=2)

        if len(list(filter(lambda x: x is not None, (hradius, vradius, eccentricity)))) != 2:
            raise ValueError(filldedent('''
                Exactly two arguments of "hradius", "vradius", and
                "eccentricity" must not be None.'''))

        if eccentricity is not None:
            eccentricity = sympify(eccentricity)
            if eccentricity.is_negative:
                raise GeometryError("Eccentricity of ellipse/circle should lie between [0, 1)")
            elif hradius is None:
                hradius = vradius / sqrt(1 - eccentricity**2)
            elif vradius is None:
                vradius = hradius * sqrt(1 - eccentricity**2)

        if hradius == vradius:
            return Circle(center, hradius, **kwargs)

        if S.Zero in (hradius, vradius):
            return Segment(Point(center[0] - hradius, center[1] - vradius), Point(center[0] + hradius, center[1] + vradius))

        if hradius.is_real is False or vradius.is_real is False:
            raise GeometryError("Invalid value encountered when computing hradius / vradius.")

        return GeometryEntity.__new__(cls, center, hradius, vradius, **kwargs)

    def _svg(self, scale_factor=1., fill_color="#66cc99"):
        """Returns SVG ellipse element for the Ellipse.

        Parameters
        ==========

        scale_factor : float
            Multiplication factor for the SVG stroke-width.  Default is 1.
        fill_color : str, optional
            Hex string for fill color. Default is "#66cc99".
        """

        c = N(self.center)
        h, v = N(self.hradius), N(self.vradius)
        return (
            '<ellipse fill="{1}" stroke="#555555" '
            'stroke-width="{0}" opacity="0.6" cx="{2}" cy="{3}" rx="{4}" ry="{5}"/>'
        ).format(2. * scale_factor, fill_color, c.x, c.y, h, v)

    @property
    def ambient_dimension(self):
        return 2

    @property
    def apoapsis(self):
        """The apoapsis of the ellipse.

        The greatest distance between the focus and the contour.

        Returns
        =======

        apoapsis : number

        See Also
        ========

        periapsis : Returns shortest distance between foci and contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.apoapsis
        2*sqrt(2) + 3

        """
        return self.major * (1 + self.eccentricity)

    def arbitrary_point(self, parameter='t'):
        """A parameterized point on the ellipse.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        arbitrary_point : Point

        Raises
        ======

        ValueError
            When `parameter` already appears in the functions.

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.arbitrary_point()
        Point2D(3*cos(t), 2*sin(t))

        """
        t = _symbol(parameter, real=True)
        if t.name in (f.name for f in self.free_symbols):
            raise ValueError(filldedent('Symbol %s already appears in object '
                                        'and cannot be used as a parameter.' % t.name))
        return Point(self.center.x + self.hradius*cos(t),
                     self.center.y + self.vradius*sin(t))

    @property
    def area(self):
        """The area of the ellipse.

        Returns
        =======

        area : number

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.area
        3*pi

        """
        return simplify(S.Pi * self.hradius * self.vradius)

    @property
    def bounds(self):
        """Return a tuple (xmin, ymin, xmax, ymax) representing the bounding
        rectangle for the geometric figure.

        """

        h, v = self.hradius, self.vradius
        return (self.center.x - h, self.center.y - v, self.center.x + h, self.center.y + v)

    @property
    def center(self):
        """The center of the ellipse.

        Returns
        =======

        center : number

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.center
        Point2D(0, 0)

        """
        return self.args[0]

    @property
    def circumference(self):
        """The circumference of the ellipse.

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.circumference
        12*elliptic_e(8/9)

        """
        if self.eccentricity == 1:
            # degenerate
            return 4*self.major
        elif self.eccentricity == 0:
            # circle
            return 2*pi*self.hradius
        else:
            return 4*self.major*elliptic_e(self.eccentricity**2)

    @property
    def eccentricity(self):
        """The eccentricity of the ellipse.

        Returns
        =======

        eccentricity : number

        Examples
        ========

        >>> from sympy import Point, Ellipse, sqrt
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, sqrt(2))
        >>> e1.eccentricity
        sqrt(7)/3

        """
        return self.focus_distance / self.major

    def encloses_point(self, p):
        """
        Return True if p is enclosed by (is inside of) self.

        Notes
        -----
        Being on the border of self is considered False.

        Parameters
        ==========

        p : Point

        Returns
        =======

        encloses_point : True, False or None

        See Also
        ========

        sympy.geometry.point.Point

        Examples
        ========

        >>> from sympy import Ellipse, S
        >>> from sympy.abc import t
        >>> e = Ellipse((0, 0), 3, 2)
        >>> e.encloses_point((0, 0))
        True
        >>> e.encloses_point(e.arbitrary_point(t).subs(t, S.Half))
        False
        >>> e.encloses_point((4, 0))
        False

        """
        p = Point(p, dim=2)
        if p in self:
            return False

        if len(self.foci) == 2:
            # if the combined distance from the foci to p (h1 + h2) is less
            # than the combined distance from the foci to the minor axis
            # (which is the same as the major axis length) then p is inside
            # the ellipse
            h1, h2 = [f.distance(p) for f in self.foci]
            test = 2*self.major - (h1 + h2)
        else:
            test = self.radius - self.center.distance(p)

        return fuzzy_bool(test.is_positive)

    def equation(self, x='x', y='y', _slope=None):
        """
        Returns the equation of an ellipse aligned with the x and y axes;
        when slope is given, the equation returned corresponds to an ellipse
        with a major axis having that slope.

        Parameters
        ==========

        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.
        _slope : Expr, optional
                The slope of the major axis. Ignored when 'None'.

        Returns
        =======

        equation : SymPy expression

        See Also
        ========

        arbitrary_point : Returns parameterized point on ellipse

        Examples
        ========

        >>> from sympy import Point, Ellipse, pi
        >>> from sympy.abc import x, y
        >>> e1 = Ellipse(Point(1, 0), 3, 2)
        >>> eq1 = e1.equation(x, y); eq1
        y**2/4 + (x/3 - 1/3)**2 - 1
        >>> eq2 = e1.equation(x, y, _slope=1); eq2
        (-x + y + 1)**2/8 + (x + y - 1)**2/18 - 1

        A point on e1 satisfies eq1. Let's use one on the x-axis:

        >>> p1 = e1.center + Point(e1.major, 0)
        >>> assert eq1.subs(x, p1.x).subs(y, p1.y) == 0

        When rotated the same as the rotated ellipse, about the center
        point of the ellipse, it will satisfy the rotated ellipse's
        equation, too:

        >>> r1 = p1.rotate(pi/4, e1.center)
        >>> assert eq2.subs(x, r1.x).subs(y, r1.y) == 0

        References
        ==========

        .. [1] https://math.stackexchange.com/questions/108270/what-is-the-equation-of-an-ellipse-that-is-not-aligned-with-the-axis
        .. [2] https://en.wikipedia.org/wiki/Ellipse#Shifted_ellipse

        """

        x = _symbol(x, real=True)
        y = _symbol(y, real=True)

        dx = x - self.center.x
        dy = y - self.center.y

        if _slope is not None:
            L = (dy - _slope*dx)**2
            l = (_slope*dy + dx)**2
            h = 1 + _slope**2
            b = h*self.major**2
            a = h*self.minor**2
            return l/b + L/a - 1

        else:
            t1 = (dx/self.hradius)**2
            t2 = (dy/self.vradius)**2
            return t1 + t2 - 1

    def evolute(self, x='x', y='y'):
        """The equation of evolute of the ellipse.

        Parameters
        ==========

        x : str, optional
            Label for the x-axis. Default value is 'x'.
        y : str, optional
            Label for the y-axis. Default value is 'y'.

        Returns
        =======

        equation : SymPy expression

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(1, 0), 3, 2)
        >>> e1.evolute()
        2**(2/3)*y**(2/3) + (3*x - 3)**(2/3) - 5**(2/3)
        """
        if len(self.args) != 3:
            raise NotImplementedError('Evolute of arbitrary Ellipse is not supported.')
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        t1 = (self.hradius*(x - self.center.x))**Rational(2, 3)
        t2 = (self.vradius*(y - self.center.y))**Rational(2, 3)
        return t1 + t2 - (self.hradius**2 - self.vradius**2)**Rational(2, 3)

    @property
    def foci(self):
        """The foci of the ellipse.

        Notes
        -----
        The foci can only be calculated if the major/minor axes are known.

        Raises
        ======

        ValueError
            When the major and minor axis cannot be determined.

        See Also
        ========

        sympy.geometry.point.Point
        focus_distance : Returns the distance between focus and center

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.foci
        (Point2D(-2*sqrt(2), 0), Point2D(2*sqrt(2), 0))

        """
        c = self.center
        hr, vr = self.hradius, self.vradius
        if hr == vr:
            return (c, c)

        # calculate focus distance manually, since focus_distance calls this
        # routine
        fd = sqrt(self.major**2 - self.minor**2)
        if hr == self.minor:
            # foci on the y-axis
            return (c + Point(0, -fd), c + Point(0, fd))
        elif hr == self.major:
            # foci on the x-axis
            return (c + Point(-fd, 0), c + Point(fd, 0))

    @property
    def focus_distance(self):
        """The focal distance of the ellipse.

        The distance between the center and one focus.

        Returns
        =======

        focus_distance : number

        See Also
        ========

        foci

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.focus_distance
        2*sqrt(2)

        """
        return Point.distance(self.center, self.foci[0])

    @property
    def hradius(self):
        """The horizontal radius of the ellipse.

        Returns
        =======

        hradius : number

        See Also
        ========

        vradius, major, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.hradius
        3

        """
        return self.args[1]

    def intersection(self, o):
        """The intersection of this ellipse and another geometrical entity
        `o`.

        Parameters
        ==========

        o : GeometryEntity

        Returns
        =======

        intersection : list of GeometryEntity objects

        Notes
        -----
        Currently supports intersections with Point, Line, Segment, Ray,
        Circle and Ellipse types.

        See Also
        ========

        sympy.geometry.entity.GeometryEntity

        Examples
        ========

        >>> from sympy import Ellipse, Point, Line
        >>> e = Ellipse(Point(0, 0), 5, 7)
        >>> e.intersection(Point(0, 0))
        []
        >>> e.intersection(Point(5, 0))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(0,0), Point(0, 1)))
        [Point2D(0, -7), Point2D(0, 7)]
        >>> e.intersection(Line(Point(5,0), Point(5, 1)))
        [Point2D(5, 0)]
        >>> e.intersection(Line(Point(6,0), Point(6, 1)))
        []
        >>> e = Ellipse(Point(-1, 0), 4, 3)
        >>> e.intersection(Ellipse(Point(1, 0), 4, 3))
        [Point2D(0, -3*sqrt(15)/4), Point2D(0, 3*sqrt(15)/4)]
        >>> e.intersection(Ellipse(Point(5, 0), 4, 3))
        [Point2D(2, -3*sqrt(7)/4), Point2D(2, 3*sqrt(7)/4)]
        >>> e.intersection(Ellipse(Point(100500, 0), 4, 3))
        []
        >>> e.intersection(Ellipse(Point(0, 0), 3, 4))
        [Point2D(3, 0), Point2D(-363/175, -48*sqrt(111)/175), Point2D(-363/175, 48*sqrt(111)/175)]
        >>> e.intersection(Ellipse(Point(-1, 0), 3, 4))
        [Point2D(-17/5, -12/5), Point2D(-17/5, 12/5), Point2D(7/5, -12/5), Point2D(7/5, 12/5)]
        """
        # TODO: Replace solve with nonlinsolve, when nonlinsolve will be able to solve in real domain

        if isinstance(o, Point):
            if o in self:
                return [o]
            else:
                return []

        elif isinstance(o, (Segment2D, Ray2D)):
            ellipse_equation = self.equation(x, y)
            result = solve([ellipse_equation, Line(
                o.points[0], o.points[1]).equation(x, y)], [x, y],
                set=True)[1]
            return list(ordered([Point(i) for i in result if i in o]))

        elif isinstance(o, Polygon):
            return o.intersection(self)

        elif isinstance(o, (Ellipse, Line2D)):
            if o == self:
                return self
            else:
                ellipse_equation = self.equation(x, y)
                return list(ordered([Point(i) for i in solve(
                    [ellipse_equation, o.equation(x, y)], [x, y],
                    set=True)[1]]))
        elif isinstance(o, LinearEntity3D):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Intersection not handled for %s' % func_name(o))

    def is_tangent(self, o):
        """Is `o` tangent to the ellipse?

        Parameters
        ==========

        o : GeometryEntity
            An Ellipse, LinearEntity or Polygon

        Raises
        ======

        NotImplementedError
            When the wrong type of argument is supplied.

        Returns
        =======

        is_tangent: boolean
            True if o is tangent to the ellipse, False otherwise.

        See Also
        ========

        tangent_lines

        Examples
        ========

        >>> from sympy import Point, Ellipse, Line
        >>> p0, p1, p2 = Point(0, 0), Point(3, 0), Point(3, 3)
        >>> e1 = Ellipse(p0, 3, 2)
        >>> l1 = Line(p1, p2)
        >>> e1.is_tangent(l1)
        True

        """
        if isinstance(o, Point2D):
            return False
        elif isinstance(o, Ellipse):
            intersect = self.intersection(o)
            if isinstance(intersect, Ellipse):
                return True
            elif intersect:
                return all((self.tangent_lines(i)[0]).equals(o.tangent_lines(i)[0]) for i in intersect)
            else:
                return False
        elif isinstance(o, Line2D):
            hit = self.intersection(o)
            if not hit:
                return False
            if len(hit) == 1:
                return True
            # might return None if it can't decide
            return hit[0].equals(hit[1])
        elif isinstance(o, (Segment2D, Ray2D)):
            intersect = self.intersection(o)
            if len(intersect) == 1:
                return o in self.tangent_lines(intersect[0])[0]
            else:
                return False
        elif isinstance(o, Polygon):
            return all(self.is_tangent(s) for s in o.sides)
        elif isinstance(o, (LinearEntity3D, Point3D)):
            raise TypeError('Entity must be two dimensional, not three dimensional')
        else:
            raise TypeError('Is_tangent not handled for %s' % func_name(o))

    @property
    def major(self):
        """Longer axis of the ellipse (if it can be determined) else hradius.

        Returns
        =======

        major : number or expression

        See Also
        ========

        hradius, vradius, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse, Symbol
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.major
        3

        >>> a = Symbol('a')
        >>> b = Symbol('b')
        >>> Ellipse(p1, a, b).major
        a
        >>> Ellipse(p1, b, a).major
        b

        >>> m = Symbol('m')
        >>> M = m + 1
        >>> Ellipse(p1, m, M).major
        m + 1

        """
        ab = self.args[1:3]
        if len(ab) == 1:
            return ab[0]
        a, b = ab
        o = b - a < 0
        if o == True:
            return a
        elif o == False:
            return b
        return self.hradius

    @property
    def minor(self):
        """Shorter axis of the ellipse (if it can be determined) else vradius.

        Returns
        =======

        minor : number or expression

        See Also
        ========

        hradius, vradius, major

        Examples
        ========

        >>> from sympy import Point, Ellipse, Symbol
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.minor
        1

        >>> a = Symbol('a')
        >>> b = Symbol('b')
        >>> Ellipse(p1, a, b).minor
        b
        >>> Ellipse(p1, b, a).minor
        a

        >>> m = Symbol('m')
        >>> M = m + 1
        >>> Ellipse(p1, m, M).minor
        m

        """
        ab = self.args[1:3]
        if len(ab) == 1:
            return ab[0]
        a, b = ab
        o = a - b < 0
        if o == True:
            return a
        elif o == False:
            return b
        return self.vradius

    def normal_lines(self, p, prec=None):
        """Normal lines between `p` and the ellipse.

        Parameters
        ==========

        p : Point

        Returns
        =======

        normal_lines : list with 1, 2 or 4 Lines

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e = Ellipse((0, 0), 2, 3)
        >>> c = e.center
        >>> e.normal_lines(c + Point(1, 0))
        [Line2D(Point2D(0, 0), Point2D(1, 0))]
        >>> e.normal_lines(c)
        [Line2D(Point2D(0, 0), Point2D(0, 1)), Line2D(Point2D(0, 0), Point2D(1, 0))]

        Off-axis points require the solution of a quartic equation. This
        often leads to very large expressions that may be of little practical
        use. An approximate solution of `prec` digits can be obtained by
        passing in the desired value:

        >>> e.normal_lines((3, 3), prec=2)
        [Line2D(Point2D(-0.81, -2.7), Point2D(0.19, -1.2)),
        Line2D(Point2D(1.5, -2.0), Point2D(2.5, -2.7))]

        Whereas the above solution has an operation count of 12, the exact
        solution has an operation count of 2020.
        """
        p = Point(p, dim=2)

        # XXX change True to something like self.angle == 0 if the arbitrarily
        # rotated ellipse is introduced.
        # https://github.com/sympy/sympy/issues/2815)
        if True:
            rv = []
            if p.x == self.center.x:
                rv.append(Line(self.center, slope=oo))
            if p.y == self.center.y:
                rv.append(Line(self.center, slope=0))
            if rv:
                # at these special orientations of p either 1 or 2 normals
                # exist and we are done
                return rv

        # find the 4 normal points and construct lines through them with
        # the corresponding slope
        eq = self.equation(x, y)
        dydx = idiff(eq, y, x)
        norm = -1/dydx
        slope = Line(p, (x, y)).slope
        seq = slope - norm

        # TODO: Replace solve with solveset, when this line is tested
        yis = solve(seq, y)[0]
        xeq = eq.subs(y, yis).as_numer_denom()[0].expand()
        if len(xeq.free_symbols) == 1:
            try:
                # this is so much faster, it's worth a try
                xsol = Poly(xeq, x).real_roots()
            except (DomainError, PolynomialError, NotImplementedError):
                # TODO: Replace solve with solveset, when these lines are tested
                xsol = _nsort(solve(xeq, x), separated=True)[0]
            points = [Point(i, solve(eq.subs(x, i), y)[0]) for i in xsol]
        else:
            raise NotImplementedError(
                'intersections for the general ellipse are not supported')
        slopes = [norm.subs(zip((x, y), pt.args)) for pt in points]
        if prec is not None:
            points = [pt.n(prec) for pt in points]
            slopes = [i if _not_a_coeff(i) else i.n(prec) for i in slopes]
        return [Line(pt, slope=s) for pt, s in zip(points, slopes)]

    @property
    def periapsis(self):
        """The periapsis of the ellipse.

        The shortest distance between the focus and the contour.

        Returns
        =======

        periapsis : number

        See Also
        ========

        apoapsis : Returns greatest distance between focus and contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.periapsis
        3 - 2*sqrt(2)

        """
        return self.major * (1 - self.eccentricity)

    @property
    def semilatus_rectum(self):
        """
        Calculates the semi-latus rectum of the Ellipse.

        Semi-latus rectum is defined as one half of the chord through a
        focus parallel to the conic section directrix of a conic section.

        Returns
        =======

        semilatus_rectum : number

        See Also
        ========

        apoapsis : Returns greatest distance between focus and contour

        periapsis : The shortest distance between the focus and the contour

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.semilatus_rectum
        1/3

        References
        ==========

        .. [1] https://mathworld.wolfram.com/SemilatusRectum.html
        .. [2] https://en.wikipedia.org/wiki/Ellipse#Semi-latus_rectum

        """
        return self.major * (1 - self.eccentricity ** 2)

    def auxiliary_circle(self):
        """Returns a Circle whose diameter is the major axis of the ellipse.

        Examples
        ========

        >>> from sympy import Ellipse, Point, symbols
        >>> c = Point(1, 2)
        >>> Ellipse(c, 8, 7).auxiliary_circle()
        Circle(Point2D(1, 2), 8)
        >>> a, b = symbols('a b')
        >>> Ellipse(c, a, b).auxiliary_circle()
        Circle(Point2D(1, 2), Max(a, b))
        """
        return Circle(self.center, Max(self.hradius, self.vradius))

    def director_circle(self):
        """
        Returns a Circle consisting of all points where two perpendicular
        tangent lines to the ellipse cross each other.

        Returns
        =======

        Circle
            A director circle returned as a geometric object.

        Examples
        ========

        >>> from sympy import Ellipse, Point, symbols
        >>> c = Point(3,8)
        >>> Ellipse(c, 7, 9).director_circle()
        Circle(Point2D(3, 8), sqrt(130))
        >>> a, b = symbols('a b')
        >>> Ellipse(c, a, b).director_circle()
        Circle(Point2D(3, 8), sqrt(a**2 + b**2))

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Director_circle

        """
        return Circle(self.center, sqrt(self.hradius**2 + self.vradius**2))

    def plot_interval(self, parameter='t'):
        """The plot interval for the default geometric plot of the Ellipse.

        Parameters
        ==========

        parameter : str, optional
            Default value is 't'.

        Returns
        =======

        plot_interval : list
            [parameter, lower_bound, upper_bound]

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.plot_interval()
        [t, -pi, pi]

        """
        t = _symbol(parameter, real=True)
        return [t, -S.Pi, S.Pi]

    def random_point(self, seed=None):
        """A random point on the ellipse.

        Returns
        =======

        point : Point

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.random_point() # gives some random point
        Point2D(...)
        >>> p1 = e1.random_point(seed=0); p1.n(2)
        Point2D(2.1, 1.4)

        Notes
        =====

        When creating a random point, one may simply replace the
        parameter with a random number. When doing so, however, the
        random number should be made a Rational or else the point
        may not test as being in the ellipse:

        >>> from sympy.abc import t
        >>> from sympy import Rational
        >>> arb = e1.arbitrary_point(t); arb
        Point2D(3*cos(t), 2*sin(t))
        >>> arb.subs(t, .1) in e1
        False
        >>> arb.subs(t, Rational(.1)) in e1
        True
        >>> arb.subs(t, Rational('.1')) in e1
        True

        See Also
        ========
        sympy.geometry.point.Point
        arbitrary_point : Returns parameterized point on ellipse
        """
        t = _symbol('t', real=True)
        x, y = self.arbitrary_point(t).args
        # get a random value in [-1, 1) corresponding to cos(t)
        # and confirm that it will test as being in the ellipse
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        # simplify this now or else the Float will turn s into a Float
        r = Rational(rng.random())
        c = 2*r - 1
        s = sqrt(1 - c**2)
        return Point(x.subs(cos(t), c), y.subs(sin(t), s))

    def reflect(self, line):
        """Override GeometryEntity.reflect since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle, Line
        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))
        Circle(Point2D(1, 0), -1)
        >>> from sympy import Ellipse, Line, Point
        >>> Ellipse(Point(3, 4), 1, 3).reflect(Line(Point(0, -4), Point(5, 0)))
        Traceback (most recent call last):
        ...
        NotImplementedError:
        General Ellipse is not supported but the equation of the reflected
        Ellipse is given by the zeros of: f(x, y) = (9*x/41 + 40*y/41 +
        37/41)**2 + (40*x/123 - 3*y/41 - 364/123)**2 - 1

        Notes
        =====

        Until the general ellipse (with no axis parallel to the x-axis) is
        supported a NotImplemented error is raised and the equation whose
        zeros define the rotated ellipse is given.

        """

        if line.slope in (0, oo):
            c = self.center
            c = c.reflect(line)
            return self.func(c, -self.hradius, self.vradius)
        else:
            x, y = [uniquely_named_symbol(
                name, (self, line), modify=lambda s: '_' + s, real=True)
                for name in 'xy']
            expr = self.equation(x, y)
            p = Point(x, y).reflect(line)
            result = expr.subs(zip((x, y), p.args
                                   ), simultaneous=True)
            raise NotImplementedError(filldedent(
                'General Ellipse is not supported but the equation '
                'of the reflected Ellipse is given by the zeros of: ' +
                "f(%s, %s) = %s" % (str(x), str(y), str(result))))

    def rotate(self, angle=0, pt=None):
        """Rotate ``angle`` radians counterclockwise about Point ``pt``.

        Note: since the general ellipse is not supported, only rotations that
        are integer multiples of pi/2 are allowed.

        Examples
        ========

        >>> from sympy import Ellipse, pi
        >>> Ellipse((1, 0), 2, 1).rotate(pi/2)
        Ellipse(Point2D(0, 1), 1, 2)
        >>> Ellipse((1, 0), 2, 1).rotate(pi)
        Ellipse(Point2D(-1, 0), 2, 1)
        """
        if self.hradius == self.vradius:
            return self.func(self.center.rotate(angle, pt), self.hradius)
        if (angle/S.Pi).is_integer:
            return super().rotate(angle, pt)
        if (2*angle/S.Pi).is_integer:
            return self.func(self.center.rotate(angle, pt), self.vradius, self.hradius)
        # XXX see https://github.com/sympy/sympy/issues/2815 for general ellipes
        raise NotImplementedError('Only rotations of pi/2 are currently supported for Ellipse.')

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since it is the major and minor
        axes which must be scaled and they are not GeometryEntities.

        Examples
        ========

        >>> from sympy import Ellipse
        >>> Ellipse((0, 0), 2, 1).scale(2, 4)
        Circle(Point2D(0, 0), 4)
        >>> Ellipse((0, 0), 2, 1).scale(2)
        Ellipse(Point2D(0, 0), 4, 1)
        """
        c = self.center
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        h = self.hradius
        v = self.vradius
        return self.func(c.scale(x, y), hradius=h*x, vradius=v*y)

    def tangent_lines(self, p):
        """Tangent lines between `p` and the ellipse.

        If `p` is on the ellipse, returns the tangent line through point `p`.
        Otherwise, returns the tangent line(s) from `p` to the ellipse, or
        None if no tangent line is possible (e.g., `p` inside ellipse).

        Parameters
        ==========

        p : Point

        Returns
        =======

        tangent_lines : list with 1 or 2 Lines

        Raises
        ======

        NotImplementedError
            Can only find tangent lines for a point, `p`, on the ellipse.

        See Also
        ========

        sympy.geometry.point.Point, sympy.geometry.line.Line

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> e1 = Ellipse(Point(0, 0), 3, 2)
        >>> e1.tangent_lines(Point(3, 0))
        [Line2D(Point2D(3, 0), Point2D(3, -12))]

        """
        p = Point(p, dim=2)
        if self.encloses_point(p):
            return []

        if p in self:
            delta = self.center - p
            rise = (self.vradius**2)*delta.x
            run = -(self.hradius**2)*delta.y
            p2 = Point(simplify(p.x + run),
                       simplify(p.y + rise))
            return [Line(p, p2)]
        else:
            if len(self.foci) == 2:
                f1, f2 = self.foci
                maj = self.hradius
                test = (2*maj -
                        Point.distance(f1, p) -
                        Point.distance(f2, p))
            else:
                test = self.radius - Point.distance(self.center, p)
            if test.is_number and test.is_positive:
                return []
            # else p is outside the ellipse or we can't tell. In case of the
            # latter, the solutions returned will only be valid if
            # the point is not inside the ellipse; if it is, nan will result.
            eq = self.equation(x, y)
            dydx = idiff(eq, y, x)
            slope = Line(p, Point(x, y)).slope

            # TODO: Replace solve with solveset, when this line is tested
            tangent_points = solve([slope - dydx, eq], [x, y])

            # handle horizontal and vertical tangent lines
            if len(tangent_points) == 1:
                if tangent_points[0][
                           0] == p.x or tangent_points[0][1] == p.y:
                    return [Line(p, p + Point(1, 0)), Line(p, p + Point(0, 1))]
                else:
                    return [Line(p, p + Point(0, 1)), Line(p, tangent_points[0])]

            # others
            return [Line(p, tangent_points[0]), Line(p, tangent_points[1])]

    @property
    def vradius(self):
        """The vertical radius of the ellipse.

        Returns
        =======

        vradius : number

        See Also
        ========

        hradius, major, minor

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.vradius
        1

        """
        return self.args[2]


    def second_moment_of_area(self, point=None):
        """Returns the second moment and product moment area of an ellipse.

        Parameters
        ==========

        point : Point, two-tuple of sympifiable objects, or None(default=None)
            point is the point about which second moment of area is to be found.
            If "point=None" it will be calculated about the axis passing through the
            centroid of the ellipse.

        Returns
        =======

        I_xx, I_yy, I_xy : number or SymPy expression
            I_xx, I_yy are second moment of area of an ellise.
            I_xy is product moment of area of an ellipse.

        Examples
        ========

        >>> from sympy import Point, Ellipse
        >>> p1 = Point(0, 0)
        >>> e1 = Ellipse(p1, 3, 1)
        >>> e1.second_moment_of_area()
        (3*pi/4, 27*pi/4, 0)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/List_of_second_moments_of_area

        """

        I_xx = (S.Pi*(self.hradius)*(self.vradius**3))/4
        I_yy = (S.Pi*(self.hradius**3)*(self.vradius))/4
        I_xy = 0

        if point is None:
            return I_xx, I_yy, I_xy

        # parallel axis theorem
        I_xx = I_xx + self.area*((point[1] - self.center.y)**2)
        I_yy = I_yy + self.area*((point[0] - self.center.x)**2)
        I_xy = I_xy + self.area*(point[0] - self.center.x)*(point[1] - self.center.y)

        return I_xx, I_yy, I_xy


    def polar_second_moment_of_area(self):
        """Returns the polar second moment of area of an Ellipse

        It is a constituent of the second moment of area, linked through
        the perpendicular axis theorem. While the planar second moment of
        area describes an object's resistance to deflection (bending) when
        subjected to a force applied to a plane parallel to the central
        axis, the polar second moment of area describes an object's
        resistance to deflection when subjected to a moment applied in a
        plane perpendicular to the object's central axis (i.e. parallel to
        the cross-section)

        Examples
        ========

        >>> from sympy import symbols, Circle, Ellipse
        >>> c = Circle((5, 5), 4)
        >>> c.polar_second_moment_of_area()
        128*pi
        >>> a, b = symbols('a, b')
        >>> e = Ellipse((0, 0), a, b)
        >>> e.polar_second_moment_of_area()
        pi*a**3*b/4 + pi*a*b**3/4

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Polar_moment_of_inertia

        """
        second_moment = self.second_moment_of_area()
        return second_moment[0] + second_moment[1]


    def section_modulus(self, point=None):
        """Returns a tuple with the section modulus of an ellipse

        Section modulus is a geometric property of an ellipse defined as the
        ratio of second moment of area to the distance of the extreme end of
        the ellipse from the centroidal axis.

        Parameters
        ==========

        point : Point, two-tuple of sympifyable objects, or None(default=None)
            point is the point at which section modulus is to be found.
            If "point=None" section modulus will be calculated for the
            point farthest from the centroidal axis of the ellipse.

        Returns
        =======

        S_x, S_y: numbers or SymPy expressions
                  S_x is the section modulus with respect to the x-axis
                  S_y is the section modulus with respect to the y-axis
                  A negative sign indicates that the section modulus is
                  determined for a point below the centroidal axis.

        Examples
        ========

        >>> from sympy import Symbol, Ellipse, Circle, Point2D
        >>> d = Symbol('d', positive=True)
        >>> c = Circle((0, 0), d/2)
        >>> c.section_modulus()
        (pi*d**3/32, pi*d**3/32)
        >>> e = Ellipse(Point2D(0, 0), 2, 4)
        >>> e.section_modulus()
        (8*pi, 4*pi)
        >>> e.section_modulus((2, 2))
        (16*pi, 4*pi)

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Section_modulus

        """
        x_c, y_c = self.center
        if point is None:
            # taking x and y as maximum distances from centroid
            x_min, y_min, x_max, y_max = self.bounds
            y = max(y_c - y_min, y_max - y_c)
            x = max(x_c - x_min, x_max - x_c)
        else:
            # taking x and y as distances of the given point from the center
            point = Point2D(point)
            y = point.y - y_c
            x = point.x - x_c

        second_moment = self.second_moment_of_area()
        S_x = second_moment[0]/y
        S_y = second_moment[1]/x

        return S_x, S_y


class Circle(Ellipse):
    r"""A circle in space.

    Constructed simply from a center and a radius, from three
    non-collinear points, or the equation of a circle.

    Parameters
    ==========

    center : Point
    radius : number or SymPy expression
    points : sequence of three Points
    equation : equation of a circle

    Attributes
    ==========

    radius (synonymous with hradius, vradius, major and minor)
    circumference
    equation

    Raises
    ======

    GeometryError
        When the given equation is not that of a circle.
        When trying to construct circle from incorrect parameters.

    See Also
    ========

    Ellipse, sympy.geometry.point.Point

    Examples
    ========

    >>> from sympy import Point, Circle, Eq
    >>> from sympy.abc import x, y, a, b

    A circle constructed from a center and radius:

    >>> c1 = Circle(Point(0, 0), 5)
    >>> c1.hradius, c1.vradius, c1.radius
    (5, 5, 5)

    A circle constructed from three points:

    >>> c2 = Circle(Point(0, 0), Point(1, 1), Point(1, 0))
    >>> c2.hradius, c2.vradius, c2.radius, c2.center
    (sqrt(2)/2, sqrt(2)/2, sqrt(2)/2, Point2D(1/2, 1/2))

    A circle can be constructed from an equation in the form
    `ax^2 + by^2 + gx + hy + c = 0`, too:

    >>> Circle(x**2 + y**2 - 25)
    Circle(Point2D(0, 0), 5)

    If the variables corresponding to x and y are named something
    else, their name or symbol can be supplied:

    >>> Circle(Eq(a**2 + b**2, 25), x='a', y=b)
    Circle(Point2D(0, 0), 5)
    """

    def __new__(cls, *args, **kwargs):
        evaluate = kwargs.get('evaluate', global_parameters.evaluate)
        if len(args) == 1 and isinstance(args[0], (Expr, Eq)):
            x = kwargs.get('x', 'x')
            y = kwargs.get('y', 'y')
            equation = args[0].expand()
            if isinstance(equation, Eq):
                equation = equation.lhs - equation.rhs
            x = find(x, equation)
            y = find(y, equation)

            try:
                a, b, c, d, e = linear_coeffs(equation, x**2, y**2, x, y)
            except ValueError:
                raise GeometryError("The given equation is not that of a circle.")

            if S.Zero in (a, b) or a != b:
                raise GeometryError("The given equation is not that of a circle.")

            center_x = -c/a/2
            center_y = -d/b/2
            r2 = (center_x**2) + (center_y**2) - e/a

            return Circle((center_x, center_y), sqrt(r2), evaluate=evaluate)

        else:
            c, r = None, None
            if len(args) == 3:
                args = [Point(a, dim=2, evaluate=evaluate) for a in args]
                t = Triangle(*args)
                if not isinstance(t, Triangle):
                    return t
                c = t.circumcenter
                r = t.circumradius
            elif len(args) == 2:
                # Assume (center, radius) pair
                c = Point(args[0], dim=2, evaluate=evaluate)
                r = args[1]
                # this will prohibit imaginary radius
                try:
                    r = Point(r, 0, evaluate=evaluate).x
                except ValueError:
                    raise GeometryError("Circle with imaginary radius is not permitted")

            if not (c is None or r is None):
                if r == 0:
                    return c
                return GeometryEntity.__new__(cls, c, r, **kwargs)

            raise GeometryError("Circle.__new__ received unknown arguments")

    def _eval_evalf(self, prec=15, **options):
        pt, r = self.args
        dps = prec_to_dps(prec)
        pt = pt.evalf(n=dps, **options)
        r = r.evalf(n=dps, **options)
        return self.func(pt, r, evaluate=False)

    @property
    def circumference(self):
        """The circumference of the circle.

        Returns
        =======

        circumference : number or SymPy expression

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.circumference
        12*pi

        """
        return 2 * S.Pi * self.radius

    def equation(self, x='x', y='y'):
        """The equation of the circle.

        Parameters
        ==========

        x : str or Symbol, optional
            Default value is 'x'.
        y : str or Symbol, optional
            Default value is 'y'.

        Returns
        =======

        equation : SymPy expression

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(0, 0), 5)
        >>> c1.equation()
        x**2 + y**2 - 25

        """
        x = _symbol(x, real=True)
        y = _symbol(y, real=True)
        t1 = (x - self.center.x)**2
        t2 = (y - self.center.y)**2
        return t1 + t2 - self.major**2

    def intersection(self, o):
        """The intersection of this circle with another geometrical entity.

        Parameters
        ==========

        o : GeometryEntity

        Returns
        =======

        intersection : list of GeometryEntities

        Examples
        ========

        >>> from sympy import Point, Circle, Line, Ray
        >>> p1, p2, p3 = Point(0, 0), Point(5, 5), Point(6, 0)
        >>> p4 = Point(5, 0)
        >>> c1 = Circle(p1, 5)
        >>> c1.intersection(p2)
        []
        >>> c1.intersection(p4)
        [Point2D(5, 0)]
        >>> c1.intersection(Ray(p1, p2))
        [Point2D(5*sqrt(2)/2, 5*sqrt(2)/2)]
        >>> c1.intersection(Line(p2, p3))
        []

        """
        return Ellipse.intersection(self, o)

    @property
    def radius(self):
        """The radius of the circle.

        Returns
        =======

        radius : number or SymPy expression

        See Also
        ========

        Ellipse.major, Ellipse.minor, Ellipse.hradius, Ellipse.vradius

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.radius
        6

        """
        return self.args[1]

    def reflect(self, line):
        """Override GeometryEntity.reflect since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle, Line
        >>> Circle((0, 1), 1).reflect(Line((0, 0), (1, 1)))
        Circle(Point2D(1, 0), -1)
        """
        c = self.center
        c = c.reflect(line)
        return self.func(c, -self.radius)

    def scale(self, x=1, y=1, pt=None):
        """Override GeometryEntity.scale since the radius
        is not a GeometryEntity.

        Examples
        ========

        >>> from sympy import Circle
        >>> Circle((0, 0), 1).scale(2, 2)
        Circle(Point2D(0, 0), 2)
        >>> Circle((0, 0), 1).scale(2, 4)
        Ellipse(Point2D(0, 0), 2, 4)
        """
        c = self.center
        if pt:
            pt = Point(pt, dim=2)
            return self.translate(*(-pt).args).scale(x, y).translate(*pt.args)
        c = c.scale(x, y)
        x, y = [abs(i) for i in (x, y)]
        if x == y:
            return self.func(c, x*self.radius)
        h = v = self.radius
        return Ellipse(c, hradius=h*x, vradius=v*y)

    @property
    def vradius(self):
        """
        This Ellipse property is an alias for the Circle's radius.

        Whereas hradius, major and minor can use Ellipse's conventions,
        the vradius does not exist for a circle. It is always a positive
        value in order that the Circle, like Polygons, will have an
        area that can be positive or negative as determined by the sign
        of the hradius.

        Examples
        ========

        >>> from sympy import Point, Circle
        >>> c1 = Circle(Point(3, 4), 6)
        >>> c1.vradius
        6
        """
        return abs(self.radius)


from .polygon import Polygon, Triangle

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\dtypes\common.py ===
"""
Common type operations.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
)
import warnings

import numpy as np

from pandas._config import using_string_dtype

from pandas._libs import (
    Interval,
    Period,
    algos,
    lib,
)
from pandas._libs.tslibs import conversion
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.base import _registry as registry
from pandas.core.dtypes.dtypes import (
    CategoricalDtype,
    DatetimeTZDtype,
    ExtensionDtype,
    IntervalDtype,
    PeriodDtype,
    SparseDtype,
)
from pandas.core.dtypes.generic import ABCIndex
from pandas.core.dtypes.inference import (
    is_array_like,
    is_bool,
    is_complex,
    is_dataclass,
    is_decimal,
    is_dict_like,
    is_file_like,
    is_float,
    is_hashable,
    is_integer,
    is_interval,
    is_iterator,
    is_list_like,
    is_named_tuple,
    is_nested_list_like,
    is_number,
    is_re,
    is_re_compilable,
    is_scalar,
    is_sequence,
)

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        DtypeObj,
    )

DT64NS_DTYPE = conversion.DT64NS_DTYPE
TD64NS_DTYPE = conversion.TD64NS_DTYPE
INT64_DTYPE = np.dtype(np.int64)

# oh the troubles to reduce import time
_is_scipy_sparse = None

ensure_float64 = algos.ensure_float64
ensure_int64 = algos.ensure_int64
ensure_int32 = algos.ensure_int32
ensure_int16 = algos.ensure_int16
ensure_int8 = algos.ensure_int8
ensure_platform_int = algos.ensure_platform_int
ensure_object = algos.ensure_object
ensure_uint64 = algos.ensure_uint64


def ensure_str(value: bytes | Any) -> str:
    """
    Ensure that bytes and non-strings get converted into ``str`` objects.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif not isinstance(value, str):
        value = str(value)
    return value


def ensure_python_int(value: int | np.integer) -> int:
    """
    Ensure that a value is a python int.

    Parameters
    ----------
    value: int or numpy.integer

    Returns
    -------
    int

    Raises
    ------
    TypeError: if the value isn't an int or can't be converted to one.
    """
    if not (is_integer(value) or is_float(value)):
        if not is_scalar(value):
            raise TypeError(
                f"Value needs to be a scalar value, was type {type(value).__name__}"
            )
        raise TypeError(f"Wrong type {type(value)} for value {value}")
    try:
        new_value = int(value)
        assert new_value == value
    except (TypeError, ValueError, AssertionError) as err:
        raise TypeError(f"Wrong type {type(value)} for value {value}") from err
    return new_value


def classes(*klasses) -> Callable:
    """Evaluate if the tipo is a subclass of the klasses."""
    return lambda tipo: issubclass(tipo, klasses)


def _classes_and_not_datetimelike(*klasses) -> Callable:
    """
    Evaluate if the tipo is a subclass of the klasses
    and not a datetimelike.
    """
    return lambda tipo: (
        issubclass(tipo, klasses)
        and not issubclass(tipo, (np.datetime64, np.timedelta64))
    )


def is_object_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the object dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the object dtype.

    Examples
    --------
    >>> from pandas.api.types import is_object_dtype
    >>> is_object_dtype(object)
    True
    >>> is_object_dtype(int)
    False
    >>> is_object_dtype(np.array([], dtype=object))
    True
    >>> is_object_dtype(np.array([], dtype=int))
    False
    >>> is_object_dtype([1, 2, 3])
    False
    """
    return _is_dtype_type(arr_or_dtype, classes(np.object_))


def is_sparse(arr) -> bool:
    """
    Check whether an array-like is a 1-D pandas sparse array.

    .. deprecated:: 2.1.0
        Use isinstance(dtype, pd.SparseDtype) instead.

    Check that the one-dimensional array-like is a pandas sparse array.
    Returns True if it is a pandas sparse array, not another type of
    sparse array.

    Parameters
    ----------
    arr : array-like
        Array-like to check.

    Returns
    -------
    bool
        Whether or not the array-like is a pandas sparse array.

    Examples
    --------
    Returns `True` if the parameter is a 1-D pandas sparse array.

    >>> from pandas.api.types import is_sparse
    >>> is_sparse(pd.arrays.SparseArray([0, 0, 1, 0]))
    True
    >>> is_sparse(pd.Series(pd.arrays.SparseArray([0, 0, 1, 0])))
    True

    Returns `False` if the parameter is not sparse.

    >>> is_sparse(np.array([0, 0, 1, 0]))
    False
    >>> is_sparse(pd.Series([0, 1, 0, 0]))
    False

    Returns `False` if the parameter is not a pandas sparse array.

    >>> from scipy.sparse import bsr_matrix
    >>> is_sparse(bsr_matrix([0, 1, 0, 0]))
    False

    Returns `False` if the parameter has more than one dimension.
    """
    warnings.warn(
        "is_sparse is deprecated and will be removed in a future "
        "version. Check `isinstance(dtype, pd.SparseDtype)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    dtype = getattr(arr, "dtype", arr)
    return isinstance(dtype, SparseDtype)


def is_scipy_sparse(arr) -> bool:
    """
    Check whether an array-like is a scipy.sparse.spmatrix instance.

    Parameters
    ----------
    arr : array-like
        The array-like to check.

    Returns
    -------
    boolean
        Whether or not the array-like is a scipy.sparse.spmatrix instance.

    Notes
    -----
    If scipy is not installed, this function will always return False.

    Examples
    --------
    >>> from scipy.sparse import bsr_matrix
    >>> is_scipy_sparse(bsr_matrix([1, 2, 3]))
    True
    >>> is_scipy_sparse(pd.arrays.SparseArray([1, 2, 3]))
    False
    """
    global _is_scipy_sparse

    if _is_scipy_sparse is None:  # pylint: disable=used-before-assignment
        try:
            from scipy.sparse import issparse as _is_scipy_sparse
        except ImportError:
            _is_scipy_sparse = lambda _: False

    assert _is_scipy_sparse is not None
    return _is_scipy_sparse(arr)


def is_datetime64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64_dtype
    >>> is_datetime64_dtype(object)
    False
    >>> is_datetime64_dtype(np.datetime64)
    True
    >>> is_datetime64_dtype(np.array([], dtype=int))
    False
    >>> is_datetime64_dtype(np.array([], dtype=np.datetime64))
    True
    >>> is_datetime64_dtype([1, 2, 3])
    False
    """
    if isinstance(arr_or_dtype, np.dtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "M"
    return _is_dtype_type(arr_or_dtype, classes(np.datetime64))


def is_datetime64tz_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of a DatetimeTZDtype dtype.

    .. deprecated:: 2.1.0
        Use isinstance(dtype, pd.DatetimeTZDtype) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of a DatetimeTZDtype dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64tz_dtype
    >>> is_datetime64tz_dtype(object)
    False
    >>> is_datetime64tz_dtype([1, 2, 3])
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3]))  # tz-naive
    False
    >>> is_datetime64tz_dtype(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    True

    >>> from pandas.core.dtypes.dtypes import DatetimeTZDtype
    >>> dtype = DatetimeTZDtype("ns", tz="US/Eastern")
    >>> s = pd.Series([], dtype=dtype)
    >>> is_datetime64tz_dtype(dtype)
    True
    >>> is_datetime64tz_dtype(s)
    True
    """
    # GH#52607
    warnings.warn(
        "is_datetime64tz_dtype is deprecated and will be removed in a future "
        "version. Check `isinstance(dtype, pd.DatetimeTZDtype)` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr_or_dtype, DatetimeTZDtype):
        # GH#33400 fastpath for dtype object
        # GH 34986
        return True

    if arr_or_dtype is None:
        return False
    return DatetimeTZDtype.is_dtype(arr_or_dtype)


def is_timedelta64_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the timedelta64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the timedelta64 dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_timedelta64_dtype
    >>> is_timedelta64_dtype(object)
    False
    >>> is_timedelta64_dtype(np.timedelta64)
    True
    >>> is_timedelta64_dtype([1, 2, 3])
    False
    >>> is_timedelta64_dtype(pd.Series([], dtype="timedelta64[ns]"))
    True
    >>> is_timedelta64_dtype('0 days')
    False
    """
    if isinstance(arr_or_dtype, np.dtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "m"

    return _is_dtype_type(arr_or_dtype, classes(np.timedelta64))


def is_period_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Period dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.Period) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Period dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_period_dtype
    >>> is_period_dtype(object)
    False
    >>> is_period_dtype(pd.PeriodDtype(freq="D"))
    True
    >>> is_period_dtype([1, 2, 3])
    False
    >>> is_period_dtype(pd.Period("2017-01-01"))
    False
    >>> is_period_dtype(pd.PeriodIndex([], freq="Y"))
    True
    """
    warnings.warn(
        "is_period_dtype is deprecated and will be removed in a future version. "
        "Use `isinstance(dtype, pd.PeriodDtype)` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.type is Period

    if arr_or_dtype is None:
        return False
    return PeriodDtype.is_dtype(arr_or_dtype)


def is_interval_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Interval dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.IntervalDtype) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Interval dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_interval_dtype
    >>> is_interval_dtype(object)
    False
    >>> is_interval_dtype(pd.IntervalDtype())
    True
    >>> is_interval_dtype([1, 2, 3])
    False
    >>>
    >>> interval = pd.Interval(1, 2, closed="right")
    >>> is_interval_dtype(interval)
    False
    >>> is_interval_dtype(pd.IntervalIndex([interval]))
    True
    """
    # GH#52607
    warnings.warn(
        "is_interval_dtype is deprecated and will be removed in a future version. "
        "Use `isinstance(dtype, pd.IntervalDtype)` instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.type is Interval

    if arr_or_dtype is None:
        return False
    return IntervalDtype.is_dtype(arr_or_dtype)


def is_categorical_dtype(arr_or_dtype) -> bool:
    """
    Check whether an array-like or dtype is of the Categorical dtype.

    .. deprecated:: 2.2.0
        Use isinstance(dtype, pd.CategoricalDtype) instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array-like or dtype is of the Categorical dtype.

    Examples
    --------
    >>> from pandas.api.types import is_categorical_dtype
    >>> from pandas import CategoricalDtype
    >>> is_categorical_dtype(object)
    False
    >>> is_categorical_dtype(CategoricalDtype())
    True
    >>> is_categorical_dtype([1, 2, 3])
    False
    >>> is_categorical_dtype(pd.Categorical([1, 2, 3]))
    True
    >>> is_categorical_dtype(pd.CategoricalIndex([1, 2, 3]))
    True
    """
    # GH#52527
    warnings.warn(
        "is_categorical_dtype is deprecated and will be removed in a future "
        "version. Use isinstance(dtype, pd.CategoricalDtype) instead",
        DeprecationWarning,
        stacklevel=2,
    )
    if isinstance(arr_or_dtype, ExtensionDtype):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.name == "category"

    if arr_or_dtype is None:
        return False
    return CategoricalDtype.is_dtype(arr_or_dtype)


def is_string_or_object_np_dtype(dtype: np.dtype) -> bool:
    """
    Faster alternative to is_string_dtype, assumes we have a np.dtype object.
    """
    return dtype == object or dtype.kind in "SU"


def is_string_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the string dtype.

    If an array is passed with an object dtype, the elements must be
    inferred as strings.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the string dtype.

    Examples
    --------
    >>> from pandas.api.types import is_string_dtype
    >>> is_string_dtype(str)
    True
    >>> is_string_dtype(object)
    True
    >>> is_string_dtype(int)
    False
    >>> is_string_dtype(np.array(['a', 'b']))
    True
    >>> is_string_dtype(pd.Series([1, 2]))
    False
    >>> is_string_dtype(pd.Series([1, 2], dtype=object))
    False
    """
    if hasattr(arr_or_dtype, "dtype") and _get_dtype(arr_or_dtype).kind == "O":
        return is_all_strings(arr_or_dtype)

    def condition(dtype) -> bool:
        if is_string_or_object_np_dtype(dtype):
            return True
        try:
            return dtype == "string"
        except TypeError:
            return False

    return _is_dtype(arr_or_dtype, condition)


def is_dtype_equal(source, target) -> bool:
    """
    Check if two dtypes are equal.

    Parameters
    ----------
    source : The first dtype to compare
    target : The second dtype to compare

    Returns
    -------
    boolean
        Whether or not the two dtypes are equal.

    Examples
    --------
    >>> is_dtype_equal(int, float)
    False
    >>> is_dtype_equal("int", int)
    True
    >>> is_dtype_equal(object, "category")
    False
    >>> is_dtype_equal(CategoricalDtype(), "category")
    True
    >>> is_dtype_equal(DatetimeTZDtype(tz="UTC"), "datetime64")
    False
    """
    if isinstance(target, str):
        if not isinstance(source, str):
            # GH#38516 ensure we get the same behavior from
            #  is_dtype_equal(CDT, "category") and CDT == "category"
            try:
                src = _get_dtype(source)
                if isinstance(src, ExtensionDtype):
                    return src == target
            except (TypeError, AttributeError, ImportError):
                return False
    elif isinstance(source, str):
        return is_dtype_equal(target, source)

    try:
        source = _get_dtype(source)
        target = _get_dtype(target)
        return source == target
    except (TypeError, AttributeError, ImportError):
        # invalid comparison
        # object == category will hit this
        return False


def is_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an integer dtype.

    Unlike in `is_any_int_dtype`, timedelta64 instances will return False.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an integer dtype and
        not an instance of timedelta64.

    Examples
    --------
    >>> from pandas.api.types import is_integer_dtype
    >>> is_integer_dtype(str)
    False
    >>> is_integer_dtype(int)
    True
    >>> is_integer_dtype(float)
    False
    >>> is_integer_dtype(np.uint64)
    True
    >>> is_integer_dtype('int8')
    True
    >>> is_integer_dtype('Int8')
    True
    >>> is_integer_dtype(pd.Int8Dtype)
    True
    >>> is_integer_dtype(np.datetime64)
    False
    >>> is_integer_dtype(np.timedelta64)
    False
    >>> is_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_integer_dtype(pd.Index([1, 2.]))  # float
    False
    """
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.integer)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind in "iu"
    )


def is_signed_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a signed integer dtype.

    Unlike in `is_any_int_dtype`, timedelta64 instances will return False.

    The nullable Integer dtypes (e.g. pandas.Int64Dtype) are also considered
    as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a signed integer dtype
        and not an instance of timedelta64.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_signed_integer_dtype
    >>> is_signed_integer_dtype(str)
    False
    >>> is_signed_integer_dtype(int)
    True
    >>> is_signed_integer_dtype(float)
    False
    >>> is_signed_integer_dtype(np.uint64)  # unsigned
    False
    >>> is_signed_integer_dtype('int8')
    True
    >>> is_signed_integer_dtype('Int8')
    True
    >>> is_signed_integer_dtype(pd.Int8Dtype)
    True
    >>> is_signed_integer_dtype(np.datetime64)
    False
    >>> is_signed_integer_dtype(np.timedelta64)
    False
    >>> is_signed_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_signed_integer_dtype(pd.Series([1, 2]))
    True
    >>> is_signed_integer_dtype(np.array([], dtype=np.timedelta64))
    False
    >>> is_signed_integer_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_signed_integer_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned
    False
    """
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.signedinteger)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind == "i"
    )


def is_unsigned_integer_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of an unsigned integer dtype.

    The nullable Integer dtypes (e.g. pandas.UInt64Dtype) are also
    considered as integer by this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of an unsigned integer dtype.

    Examples
    --------
    >>> from pandas.api.types import is_unsigned_integer_dtype
    >>> is_unsigned_integer_dtype(str)
    False
    >>> is_unsigned_integer_dtype(int)  # signed
    False
    >>> is_unsigned_integer_dtype(float)
    False
    >>> is_unsigned_integer_dtype(np.uint64)
    True
    >>> is_unsigned_integer_dtype('uint8')
    True
    >>> is_unsigned_integer_dtype('UInt8')
    True
    >>> is_unsigned_integer_dtype(pd.UInt8Dtype)
    True
    >>> is_unsigned_integer_dtype(np.array(['a', 'b']))
    False
    >>> is_unsigned_integer_dtype(pd.Series([1, 2]))  # signed
    False
    >>> is_unsigned_integer_dtype(pd.Index([1, 2.]))  # float
    False
    >>> is_unsigned_integer_dtype(np.array([1, 2], dtype=np.uint32))
    True
    """
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.unsignedinteger)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind == "u"
    )


def is_int64_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the int64 dtype.

    .. deprecated:: 2.1.0

       is_int64_dtype is deprecated and will be removed in a future
       version. Use dtype == np.int64 instead.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the int64 dtype.

    Notes
    -----
    Depending on system architecture, the return value of `is_int64_dtype(
    int)` will be True if the OS uses 64-bit integers and False if the OS
    uses 32-bit integers.

    Examples
    --------
    >>> from pandas.api.types import is_int64_dtype
    >>> is_int64_dtype(str)  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.int32)  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.int64)  # doctest: +SKIP
    True
    >>> is_int64_dtype('int8')  # doctest: +SKIP
    False
    >>> is_int64_dtype('Int8')  # doctest: +SKIP
    False
    >>> is_int64_dtype(pd.Int64Dtype)  # doctest: +SKIP
    True
    >>> is_int64_dtype(float)  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.uint64)  # unsigned  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.array(['a', 'b']))  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.array([1, 2], dtype=np.int64))  # doctest: +SKIP
    True
    >>> is_int64_dtype(pd.Index([1, 2.]))  # float  # doctest: +SKIP
    False
    >>> is_int64_dtype(np.array([1, 2], dtype=np.uint32))  # unsigned  # doctest: +SKIP
    False
    """
    # GH#52564
    warnings.warn(
        "is_int64_dtype is deprecated and will be removed in a future "
        "version. Use dtype == np.int64 instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _is_dtype_type(arr_or_dtype, classes(np.int64))


def is_datetime64_any_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the datetime64 dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the datetime64 dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64_any_dtype
    >>> from pandas.core.dtypes.dtypes import DatetimeTZDtype
    >>> is_datetime64_any_dtype(str)
    False
    >>> is_datetime64_any_dtype(int)
    False
    >>> is_datetime64_any_dtype(np.datetime64)  # can be tz-naive
    True
    >>> is_datetime64_any_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    True
    >>> is_datetime64_any_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime64_any_dtype(np.array([1, 2]))
    False
    >>> is_datetime64_any_dtype(np.array([], dtype="datetime64[ns]"))
    True
    >>> is_datetime64_any_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))
    True
    """
    if isinstance(arr_or_dtype, (np.dtype, ExtensionDtype)):
        # GH#33400 fastpath for dtype object
        return arr_or_dtype.kind == "M"

    if arr_or_dtype is None:
        return False

    try:
        tipo = _get_dtype(arr_or_dtype)
    except TypeError:
        return False
    return lib.is_np_dtype(tipo, "M") or isinstance(tipo, DatetimeTZDtype)


def is_datetime64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the datetime64[ns] dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    bool
        Whether or not the array or dtype is of the datetime64[ns] dtype.

    Examples
    --------
    >>> from pandas.api.types import is_datetime64_ns_dtype
    >>> from pandas.core.dtypes.dtypes import DatetimeTZDtype
    >>> is_datetime64_ns_dtype(str)
    False
    >>> is_datetime64_ns_dtype(int)
    False
    >>> is_datetime64_ns_dtype(np.datetime64)  # no unit
    False
    >>> is_datetime64_ns_dtype(DatetimeTZDtype("ns", "US/Eastern"))
    True
    >>> is_datetime64_ns_dtype(np.array(['a', 'b']))
    False
    >>> is_datetime64_ns_dtype(np.array([1, 2]))
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64"))  # no unit
    False
    >>> is_datetime64_ns_dtype(np.array([], dtype="datetime64[ps]"))  # wrong unit
    False
    >>> is_datetime64_ns_dtype(pd.DatetimeIndex([1, 2, 3], dtype="datetime64[ns]"))
    True
    """
    if arr_or_dtype is None:
        return False
    try:
        tipo = _get_dtype(arr_or_dtype)
    except TypeError:
        return False
    return tipo == DT64NS_DTYPE or (
        isinstance(tipo, DatetimeTZDtype) and tipo.unit == "ns"
    )


def is_timedelta64_ns_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of the timedelta64[ns] dtype.

    This is a very specific dtype, so generic ones like `np.timedelta64`
    will return False if passed into this function.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of the timedelta64[ns] dtype.

    Examples
    --------
    >>> from pandas.core.dtypes.common import is_timedelta64_ns_dtype
    >>> is_timedelta64_ns_dtype(np.dtype('m8[ns]'))
    True
    >>> is_timedelta64_ns_dtype(np.dtype('m8[ps]'))  # Wrong frequency
    False
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype='m8[ns]'))
    True
    >>> is_timedelta64_ns_dtype(np.array([1, 2], dtype=np.timedelta64))
    False
    """
    return _is_dtype(arr_or_dtype, lambda dtype: dtype == TD64NS_DTYPE)


# This exists to silence numpy deprecation warnings, see GH#29553
def is_numeric_v_string_like(a: ArrayLike, b) -> bool:
    """
    Check if we are comparing a string-like object to a numeric ndarray.
    NumPy doesn't like to compare such objects, especially numeric arrays
    and scalar string-likes.

    Parameters
    ----------
    a : array-like, scalar
        The first object to check.
    b : array-like, scalar
        The second object to check.

    Returns
    -------
    boolean
        Whether we return a comparing a string-like object to a numeric array.

    Examples
    --------
    >>> is_numeric_v_string_like(np.array([1]), "foo")
    True
    >>> is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
    True
    >>> is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))
    True
    >>> is_numeric_v_string_like(np.array([1]), np.array([2]))
    False
    >>> is_numeric_v_string_like(np.array(["foo"]), np.array(["foo"]))
    False
    """
    is_a_array = isinstance(a, np.ndarray)
    is_b_array = isinstance(b, np.ndarray)

    is_a_numeric_array = is_a_array and a.dtype.kind in ("u", "i", "f", "c", "b")
    is_b_numeric_array = is_b_array and b.dtype.kind in ("u", "i", "f", "c", "b")
    is_a_string_array = is_a_array and a.dtype.kind in ("S", "U")
    is_b_string_array = is_b_array and b.dtype.kind in ("S", "U")

    is_b_scalar_string_like = not is_b_array and isinstance(b, str)

    return (
        (is_a_numeric_array and is_b_scalar_string_like)
        or (is_a_numeric_array and is_b_string_array)
        or (is_b_numeric_array and is_a_string_array)
    )


def needs_i8_conversion(dtype: DtypeObj | None) -> bool:
    """
    Check whether the dtype should be converted to int64.

    Dtype "needs" such a conversion if the dtype is of a datetime-like dtype

    Parameters
    ----------
    dtype : np.dtype, ExtensionDtype, or None

    Returns
    -------
    boolean
        Whether or not the dtype should be converted to int64.

    Examples
    --------
    >>> needs_i8_conversion(str)
    False
    >>> needs_i8_conversion(np.int64)
    False
    >>> needs_i8_conversion(np.datetime64)
    False
    >>> needs_i8_conversion(np.dtype(np.datetime64))
    True
    >>> needs_i8_conversion(np.array(['a', 'b']))
    False
    >>> needs_i8_conversion(pd.Series([1, 2]))
    False
    >>> needs_i8_conversion(pd.Series([], dtype="timedelta64[ns]"))
    False
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern"))
    False
    >>> needs_i8_conversion(pd.DatetimeIndex([1, 2, 3], tz="US/Eastern").dtype)
    True
    """
    if isinstance(dtype, np.dtype):
        return dtype.kind in "mM"
    return isinstance(dtype, (PeriodDtype, DatetimeTZDtype))


def is_numeric_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a numeric dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a numeric dtype.

    Examples
    --------
    >>> from pandas.api.types import is_numeric_dtype
    >>> is_numeric_dtype(str)
    False
    >>> is_numeric_dtype(int)
    True
    >>> is_numeric_dtype(float)
    True
    >>> is_numeric_dtype(np.uint64)
    True
    >>> is_numeric_dtype(np.datetime64)
    False
    >>> is_numeric_dtype(np.timedelta64)
    False
    >>> is_numeric_dtype(np.array(['a', 'b']))
    False
    >>> is_numeric_dtype(pd.Series([1, 2]))
    True
    >>> is_numeric_dtype(pd.Index([1, 2.]))
    True
    >>> is_numeric_dtype(np.array([], dtype=np.timedelta64))
    False
    """
    return _is_dtype_type(
        arr_or_dtype, _classes_and_not_datetimelike(np.number, np.bool_)
    ) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ._is_numeric
    )


def is_any_real_numeric_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a real number dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a real number dtype.

    Examples
    --------
    >>> from pandas.api.types import is_any_real_numeric_dtype
    >>> is_any_real_numeric_dtype(int)
    True
    >>> is_any_real_numeric_dtype(float)
    True
    >>> is_any_real_numeric_dtype(object)
    False
    >>> is_any_real_numeric_dtype(str)
    False
    >>> is_any_real_numeric_dtype(complex(1, 2))
    False
    >>> is_any_real_numeric_dtype(bool)
    False
    """
    return (
        is_numeric_dtype(arr_or_dtype)
        and not is_complex_dtype(arr_or_dtype)
        and not is_bool_dtype(arr_or_dtype)
    )


def is_float_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a float dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a float dtype.

    Examples
    --------
    >>> from pandas.api.types import is_float_dtype
    >>> is_float_dtype(str)
    False
    >>> is_float_dtype(int)
    False
    >>> is_float_dtype(float)
    True
    >>> is_float_dtype(np.array(['a', 'b']))
    False
    >>> is_float_dtype(pd.Series([1, 2]))
    False
    >>> is_float_dtype(pd.Index([1, 2.]))
    True
    """
    return _is_dtype_type(arr_or_dtype, classes(np.floating)) or _is_dtype(
        arr_or_dtype, lambda typ: isinstance(typ, ExtensionDtype) and typ.kind in "f"
    )


def is_bool_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a boolean dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a boolean dtype.

    Notes
    -----
    An ExtensionArray is considered boolean when the ``_is_boolean``
    attribute is set to True.

    Examples
    --------
    >>> from pandas.api.types import is_bool_dtype
    >>> is_bool_dtype(str)
    False
    >>> is_bool_dtype(int)
    False
    >>> is_bool_dtype(bool)
    True
    >>> is_bool_dtype(np.bool_)
    True
    >>> is_bool_dtype(np.array(['a', 'b']))
    False
    >>> is_bool_dtype(pd.Series([1, 2]))
    False
    >>> is_bool_dtype(np.array([True, False]))
    True
    >>> is_bool_dtype(pd.Categorical([True, False]))
    True
    >>> is_bool_dtype(pd.arrays.SparseArray([True, False]))
    True
    """
    if arr_or_dtype is None:
        return False
    try:
        dtype = _get_dtype(arr_or_dtype)
    except (TypeError, ValueError):
        return False

    if isinstance(dtype, CategoricalDtype):
        arr_or_dtype = dtype.categories
        # now we use the special definition for Index

    if isinstance(arr_or_dtype, ABCIndex):
        # Allow Index[object] that is all-bools or Index["boolean"]
        if arr_or_dtype.inferred_type == "boolean":
            if not is_bool_dtype(arr_or_dtype.dtype):
                # GH#52680
                warnings.warn(
                    "The behavior of is_bool_dtype with an object-dtype Index "
                    "of bool objects is deprecated. In a future version, "
                    "this will return False. Cast the Index to a bool dtype instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return True
        return False
    elif isinstance(dtype, ExtensionDtype):
        return getattr(dtype, "_is_boolean", False)

    return issubclass(dtype.type, np.bool_)


def is_1d_only_ea_dtype(dtype: DtypeObj | None) -> bool:
    """
    Analogue to is_extension_array_dtype but excluding DatetimeTZDtype.
    """
    return isinstance(dtype, ExtensionDtype) and not dtype._supports_2d


def is_extension_array_dtype(arr_or_dtype) -> bool:
    """
    Check if an object is a pandas extension array type.

    See the :ref:`Use Guide <extending.extension-types>` for more.

    Parameters
    ----------
    arr_or_dtype : object
        For array-like input, the ``.dtype`` attribute will
        be extracted.

    Returns
    -------
    bool
        Whether the `arr_or_dtype` is an extension array type.

    Notes
    -----
    This checks whether an object implements the pandas extension
    array interface. In pandas, this includes:

    * Categorical
    * Sparse
    * Interval
    * Period
    * DatetimeArray
    * TimedeltaArray

    Third-party libraries may implement arrays or types satisfying
    this interface as well.

    Examples
    --------
    >>> from pandas.api.types import is_extension_array_dtype
    >>> arr = pd.Categorical(['a', 'b'])
    >>> is_extension_array_dtype(arr)
    True
    >>> is_extension_array_dtype(arr.dtype)
    True

    >>> arr = np.array(['a', 'b'])
    >>> is_extension_array_dtype(arr.dtype)
    False
    """
    dtype = getattr(arr_or_dtype, "dtype", arr_or_dtype)
    if isinstance(dtype, ExtensionDtype):
        return True
    elif isinstance(dtype, np.dtype):
        return False
    else:
        try:
            with warnings.catch_warnings():
                # pandas_dtype(..) can raise UserWarning for class input
                warnings.simplefilter("ignore", UserWarning)
                dtype = pandas_dtype(dtype)
        except (TypeError, ValueError):
            # np.dtype(..) can raise ValueError
            return False
        return isinstance(dtype, ExtensionDtype)


def is_ea_or_datetimelike_dtype(dtype: DtypeObj | None) -> bool:
    """
    Check for ExtensionDtype, datetime64 dtype, or timedelta64 dtype.

    Notes
    -----
    Checks only for dtype objects, not dtype-castable strings or types.
    """
    return isinstance(dtype, ExtensionDtype) or (lib.is_np_dtype(dtype, "mM"))


def is_complex_dtype(arr_or_dtype) -> bool:
    """
    Check whether the provided array or dtype is of a complex dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array or dtype to check.

    Returns
    -------
    boolean
        Whether or not the array or dtype is of a complex dtype.

    Examples
    --------
    >>> from pandas.api.types import is_complex_dtype
    >>> is_complex_dtype(str)
    False
    >>> is_complex_dtype(int)
    False
    >>> is_complex_dtype(np.complex128)
    True
    >>> is_complex_dtype(np.array(['a', 'b']))
    False
    >>> is_complex_dtype(pd.Series([1, 2]))
    False
    >>> is_complex_dtype(np.array([1 + 1j, 5]))
    True
    """
    return _is_dtype_type(arr_or_dtype, classes(np.complexfloating))


def _is_dtype(arr_or_dtype, condition) -> bool:
    """
    Return true if the condition is satisfied for the arr_or_dtype.

    Parameters
    ----------
    arr_or_dtype : array-like, str, np.dtype, or ExtensionArrayType
        The array-like or dtype object whose dtype we want to extract.
    condition : callable[Union[np.dtype, ExtensionDtype]]

    Returns
    -------
    bool

    """
    if arr_or_dtype is None:
        return False
    try:
        dtype = _get_dtype(arr_or_dtype)
    except (TypeError, ValueError):
        return False
    return condition(dtype)


def _get_dtype(arr_or_dtype) -> DtypeObj:
    """
    Get the dtype instance associated with an array
    or dtype object.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype object whose dtype we want to extract.

    Returns
    -------
    obj_dtype : The extract dtype instance from the
                passed in array or dtype object.

    Raises
    ------
    TypeError : The passed in object is None.
    """
    if arr_or_dtype is None:
        raise TypeError("Cannot deduce dtype from null object")

    # fastpath
    if isinstance(arr_or_dtype, np.dtype):
        return arr_or_dtype
    elif isinstance(arr_or_dtype, type):
        return np.dtype(arr_or_dtype)

    # if we have an array-like
    elif hasattr(arr_or_dtype, "dtype"):
        arr_or_dtype = arr_or_dtype.dtype

    return pandas_dtype(arr_or_dtype)


def _is_dtype_type(arr_or_dtype, condition) -> bool:
    """
    Return true if the condition is satisfied for the arr_or_dtype.

    Parameters
    ----------
    arr_or_dtype : array-like or dtype
        The array-like or dtype object whose dtype we want to extract.
    condition : callable[Union[np.dtype, ExtensionDtypeType]]

    Returns
    -------
    bool : if the condition is satisfied for the arr_or_dtype
    """
    if arr_or_dtype is None:
        return condition(type(None))

    # fastpath
    if isinstance(arr_or_dtype, np.dtype):
        return condition(arr_or_dtype.type)
    elif isinstance(arr_or_dtype, type):
        if issubclass(arr_or_dtype, ExtensionDtype):
            arr_or_dtype = arr_or_dtype.type
        return condition(np.dtype(arr_or_dtype).type)

    # if we have an array-like
    if hasattr(arr_or_dtype, "dtype"):
        arr_or_dtype = arr_or_dtype.dtype

    # we are not possibly a dtype
    elif is_list_like(arr_or_dtype):
        return condition(type(None))

    try:
        tipo = pandas_dtype(arr_or_dtype).type
    except (TypeError, ValueError):
        if is_scalar(arr_or_dtype):
            return condition(type(None))

        return False

    return condition(tipo)


def infer_dtype_from_object(dtype) -> type:
    """
    Get a numpy dtype.type-style object for a dtype object.

    This methods also includes handling of the datetime64[ns] and
    datetime64[ns, TZ] objects.

    If no dtype can be found, we return ``object``.

    Parameters
    ----------
    dtype : dtype, type
        The dtype object whose numpy dtype.type-style
        object we want to extract.

    Returns
    -------
    type
    """
    if isinstance(dtype, type) and issubclass(dtype, np.generic):
        # Type object from a dtype

        return dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        # dtype object
        try:
            _validate_date_like_dtype(dtype)
        except TypeError:
            # Should still pass if we don't have a date-like
            pass
        if hasattr(dtype, "numpy_dtype"):
            # TODO: Implement this properly
            # https://github.com/pandas-dev/pandas/issues/52576
            return dtype.numpy_dtype.type
        return dtype.type

    try:
        dtype = pandas_dtype(dtype)
    except TypeError:
        pass

    if isinstance(dtype, ExtensionDtype):
        return dtype.type
    elif isinstance(dtype, str):
        # TODO(jreback)
        # should deprecate these
        if dtype in ["datetimetz", "datetime64tz"]:
            return DatetimeTZDtype.type
        elif dtype in ["period"]:
            raise NotImplementedError

        if dtype in ["datetime", "timedelta"]:
            dtype += "64"
        try:
            return infer_dtype_from_object(getattr(np, dtype))
        except (AttributeError, TypeError):
            # Handles cases like _get_dtype(int) i.e.,
            # Python objects that are valid dtypes
            # (unlike user-defined types, in general)
            #
            # TypeError handles the float16 type code of 'e'
            # further handle internal types
            pass

    return infer_dtype_from_object(np.dtype(dtype))


def _validate_date_like_dtype(dtype) -> None:
    """
    Check whether the dtype is a date-like dtype. Raises an error if invalid.

    Parameters
    ----------
    dtype : dtype, type
        The dtype to check.

    Raises
    ------
    TypeError : The dtype could not be casted to a date-like dtype.
    ValueError : The dtype is an illegal date-like dtype (e.g. the
                 frequency provided is too specific)
    """
    try:
        typ = np.datetime_data(dtype)[0]
    except ValueError as e:
        raise TypeError(e) from e
    if typ not in ["generic", "ns"]:
        raise ValueError(
            f"{repr(dtype.name)} is too specific of a frequency, "
            f"try passing {repr(dtype.type.__name__)}"
        )


def validate_all_hashable(*args, error_name: str | None = None) -> None:
    """
    Return None if all args are hashable, else raise a TypeError.

    Parameters
    ----------
    *args
        Arguments to validate.
    error_name : str, optional
        The name to use if error

    Raises
    ------
    TypeError : If an argument is not hashable

    Returns
    -------
    None
    """
    if not all(is_hashable(arg) for arg in args):
        if error_name:
            raise TypeError(f"{error_name} must be a hashable type")
        raise TypeError("All elements must be hashable")


def pandas_dtype(dtype) -> DtypeObj:
    """
    Convert input into a pandas only dtype object or a numpy dtype object.

    Parameters
    ----------
    dtype : object to be converted

    Returns
    -------
    np.dtype or a pandas dtype

    Raises
    ------
    TypeError if not a dtype

    Examples
    --------
    >>> pd.api.types.pandas_dtype(int)
    dtype('int64')
    """
    # short-circuit
    if isinstance(dtype, np.ndarray):
        return dtype.dtype
    elif isinstance(dtype, (np.dtype, ExtensionDtype)):
        return dtype

    # builtin aliases
    if dtype is str and using_string_dtype():
        from pandas.core.arrays.string_ import StringDtype

        return StringDtype(na_value=np.nan)

    # registered extension types
    result = registry.find(dtype)
    if result is not None:
        if isinstance(result, type):
            # GH 31356, GH 54592
            warnings.warn(
                f"Instantiating {result.__name__} without any arguments."
                f"Pass a {result.__name__} instance to silence this warning.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            result = result()
        return result

    # try a numpy dtype
    # raise a consistent TypeError if failed
    try:
        with warnings.catch_warnings():
            # TODO: warnings.catch_warnings can be removed when numpy>2.3.0
            # is the minimum version
            # GH#51523 - Series.astype(np.integer) doesn't show
            # numpy deprecation warning of np.integer
            # Hence enabling DeprecationWarning
            warnings.simplefilter("always", DeprecationWarning)
            npdtype = np.dtype(dtype)
    except SyntaxError as err:
        # np.dtype uses `eval` which can raise SyntaxError
        raise TypeError(f"data type '{dtype}' not understood") from err

    # Any invalid dtype (such as pd.Timestamp) should raise an error.
    # np.dtype(invalid_type).kind = 0 for such objects. However, this will
    # also catch some valid dtypes such as object, np.object_ and 'object'
    # which we safeguard against by catching them earlier and returning
    # np.dtype(valid_dtype) before this condition is evaluated.
    if is_hashable(dtype) and dtype in [
        object,
        np.object_,
        "object",
        "O",
        "object_",
    ]:
        # check hashability to avoid errors/DeprecationWarning when we get
        # here and `dtype` is an array
        return npdtype
    elif npdtype.kind == "O":
        raise TypeError(f"dtype '{dtype}' not understood")

    return npdtype


def is_all_strings(value: ArrayLike) -> bool:
    """
    Check if this is an array of strings that we should try parsing.

    Includes object-dtype ndarray containing all-strings, StringArray,
    and Categorical with all-string categories.
    Does not include numpy string dtypes.
    """
    dtype = value.dtype

    if isinstance(dtype, np.dtype):
        if len(value) == 0:
            return dtype == np.dtype("object")
        else:
            return dtype == np.dtype("object") and lib.is_string_array(
                np.asarray(value), skipna=False
            )
    elif isinstance(dtype, CategoricalDtype):
        return dtype.categories.inferred_type == "string"
    return dtype == "string"


__all__ = [
    "classes",
    "DT64NS_DTYPE",
    "ensure_float64",
    "ensure_python_int",
    "ensure_str",
    "infer_dtype_from_object",
    "INT64_DTYPE",
    "is_1d_only_ea_dtype",
    "is_all_strings",
    "is_any_real_numeric_dtype",
    "is_array_like",
    "is_bool",
    "is_bool_dtype",
    "is_categorical_dtype",
    "is_complex",
    "is_complex_dtype",
    "is_dataclass",
    "is_datetime64_any_dtype",
    "is_datetime64_dtype",
    "is_datetime64_ns_dtype",
    "is_datetime64tz_dtype",
    "is_decimal",
    "is_dict_like",
    "is_dtype_equal",
    "is_ea_or_datetimelike_dtype",
    "is_extension_array_dtype",
    "is_file_like",
    "is_float_dtype",
    "is_int64_dtype",
    "is_integer_dtype",
    "is_interval",
    "is_interval_dtype",
    "is_iterator",
    "is_named_tuple",
    "is_nested_list_like",
    "is_number",
    "is_numeric_dtype",
    "is_object_dtype",
    "is_period_dtype",
    "is_re",
    "is_re_compilable",
    "is_scipy_sparse",
    "is_sequence",
    "is_signed_integer_dtype",
    "is_sparse",
    "is_string_dtype",
    "is_string_or_object_np_dtype",
    "is_timedelta64_dtype",
    "is_timedelta64_ns_dtype",
    "is_unsigned_integer_dtype",
    "needs_i8_conversion",
    "pandas_dtype",
    "TD64NS_DTYPE",
    "validate_all_hashable",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\biomechanics\curve.py ===
"""Implementations of characteristic curves for musculotendon models."""

from dataclasses import dataclass

from sympy.core.expr import UnevaluatedExpr
from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Float, Integer
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.hyperbolic import cosh, sinh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.printing.precedence import PRECEDENCE


__all__ = [
    'CharacteristicCurveCollection',
    'CharacteristicCurveFunction',
    'FiberForceLengthActiveDeGroote2016',
    'FiberForceLengthPassiveDeGroote2016',
    'FiberForceLengthPassiveInverseDeGroote2016',
    'FiberForceVelocityDeGroote2016',
    'FiberForceVelocityInverseDeGroote2016',
    'TendonForceLengthDeGroote2016',
    'TendonForceLengthInverseDeGroote2016',
]


class CharacteristicCurveFunction(Function):
    """Base class for all musculotendon characteristic curve functions."""

    @classmethod
    def eval(cls):
        msg = (
            f'Cannot directly instantiate {cls.__name__!r}, instances of '
            f'characteristic curves must be of a concrete subclass.'

        )
        raise TypeError(msg)

    def _print_code(self, printer):
        """Print code for the function defining the curve using a printer.

        Explanation
        ===========

        The order of operations may need to be controlled as constant folding
        the numeric terms within the equations of a musculotendon
        characteristic curve can sometimes results in a numerically-unstable
        expression.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print a string representation of the
            characteristic curve as valid code in the target language.

        """
        return printer._print(printer.parenthesize(
            self.doit(deep=False, evaluate=False), PRECEDENCE['Atom'],
        ))

    _ccode = _print_code
    _cupycode = _print_code
    _cxxcode = _print_code
    _fcode = _print_code
    _jaxcode = _print_code
    _lambdacode = _print_code
    _mpmathcode = _print_code
    _octave = _print_code
    _pythoncode = _print_code
    _numpycode = _print_code
    _scipycode = _print_code


class TendonForceLengthDeGroote2016(CharacteristicCurveFunction):
    r"""Tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon force produced as a function of normalized
    tendon length.

    The function is defined by the equation:

    $fl^T = c_0 \exp{c_3 \left( \tilde{l}^T - c_1 \right)} - c_2$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon length. We'll create a
    :class:`~.Symbol` called ``l_T_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthDeGroote2016
    >>> l_T_tilde = Symbol('l_T_tilde')
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fl_T = TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_T`` and
    ``l_T_slack``, representing tendon length and tendon slack length
    respectively. We can then represent ``l_T_tilde`` as an expression, the
    ratio of these.

    >>> l_T, l_T_slack = symbols('l_T l_T_slack')
    >>> l_T_tilde = l_T/l_T_slack
    >>> fl_T = TendonForceLengthDeGroote2016.with_defaults(l_T_tilde)
    >>> fl_T
    TendonForceLengthDeGroote2016(l_T/l_T_slack, 0.2, 0.995, 0.25,
    33.93669377311689)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_T.doit(evaluate=False)
    -0.25 + 0.2*exp(33.93669377311689*(l_T/l_T_slack - 0.995))

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> fl_T.diff(l_T)
    6.787338754623378*exp(33.93669377311689*(l_T/l_T_slack - 0.995))/l_T_slack

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_T_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the tendon force-length function using the
        four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
        $c_3 = 33.93669377311689$

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.

        """
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        return cls(l_T_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, l_T_tilde, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_T_tilde : Any (sympifiable)
            Normalized tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        l_T_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_T_tilde = l_T_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return c0*exp(c3*(l_T_tilde - c1)) - c2

        return c0*exp(c3*UnevaluatedExpr(l_T_tilde - c1)) - c2

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        l_T_tilde, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return c0*c3*exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 2:
            return exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 3:
            return -c0*c3*exp(c3*UnevaluatedExpr(l_T_tilde - c1))
        elif argindex == 4:
            return Integer(-1)
        elif argindex == 5:
            return c0*(l_T_tilde - c1)*exp(c3*UnevaluatedExpr(l_T_tilde - c1))

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return TendonForceLengthInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        l_T_tilde = self.args[0]
        _l_T_tilde = printer._print(l_T_tilde)
        return r'\operatorname{fl}^T \left( %s \right)' % _l_T_tilde


class TendonForceLengthInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse tendon force-length curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized tendon length that produces a specific normalized
    tendon force.

    The function is defined by the equation:

    ${fl^T}^{-1} = frac{\log{\frac{fl^T + c_2}{c_0}}}{c_3} + c_1$

    with constant values of $c_0 = 0.2$, $c_1 = 0.995$, $c_2 = 0.25$, and
    $c_3 = 33.93669377311689$. This function is the exact analytical inverse
    of the related tendon force-length curve ``TendonForceLengthDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces no
    force when the tendon is in an unstrained state. It also produces a force
    of 1 normalized unit when the tendon is under a 5% strain.

    Examples
    ========

    The preferred way to instantiate :class:`TendonForceLengthInverseDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized tendon force-length, which is
    equal to the tendon force. We'll create a :class:`~.Symbol` called ``fl_T`` to
    represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import TendonForceLengthInverseDeGroote2016
    >>> fl_T = Symbol('fl_T')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016.with_defaults(fl_T)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, 0.2, 0.995, 0.25,
    33.93669377311689)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> l_T_tilde = TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)
    >>> l_T_tilde
    TendonForceLengthInverseDeGroote2016(fl_T, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_T_tilde.doit(evaluate=False)
    c1 + log((c2 + fl_T)/c0)/c3

    The function can also be differentiated. We'll differentiate with respect
    to l_T using the ``diff`` method on an instance with the single positional
    argument ``l_T``.

    >>> l_T_tilde.diff(fl_T)
    1/(c3*(c2 + fl_T))

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fl_T):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse tendon force-length function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = 0.2$
        $c_1 = 0.995$
        $c_2 = 0.25$
        $c_3 = 33.93669377311689$

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.

        """
        c0 = Float('0.2')
        c1 = Float('0.995')
        c2 = Float('0.25')
        c3 = Float('33.93669377311689')
        return cls(fl_T, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fl_T, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_T : Any (sympifiable)
            Normalized tendon force as a function of tendon length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.2``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``0.995``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.25``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``33.93669377311689``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        fl_T, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fl_T = fl_T.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return log((fl_T + c2)/c0)/c3 + c1

        return log(UnevaluatedExpr((fl_T + c2)/c0))/c3 + c1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        fl_T, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return 1/(c3*(fl_T + c2))
        elif argindex == 2:
            return -1/(c0*c3)
        elif argindex == 3:
            return Integer(1)
        elif argindex == 4:
            return 1/(c3*(fl_T + c2))
        elif argindex == 5:
            return -log(UnevaluatedExpr((fl_T + c2)/c0))/c3**2

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return TendonForceLengthDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        fl_T = self.args[0]
        _fl_T = printer._print(fl_T)
        return r'\left( \operatorname{fl}^T \right)^{-1} \left( %s \right)' % _fl_T


class FiberForceLengthPassiveDeGroote2016(CharacteristicCurveFunction):
    r"""Passive muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl^M_{pas} = \frac{\frac{\exp{c_1 \left(\tilde{l^M} - 1\right)}}{c_0} - 1}{\exp{c_1} - 1}$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthPassiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> fl_M = FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M_tilde, c0, c1)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthPassiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthPassiveDeGroote2016(l_M/l_M_opt, 0.6, 4.0)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.0186573603637741*(-1 + exp(6.66666666666667*(l_M/l_M_opt - 1)))

    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    0.12438240242516*exp(6.66666666666667*(l_M/l_M_opt - 1))/l_M_opt

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.

        """
        c0 = Float('0.6')
        c1 = Float('4.0')
        return cls(l_M_tilde, c0, c1)

    @classmethod
    def eval(cls, l_M_tilde, c0, c1):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        l_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1 = constants

        if evaluate:
            return (exp((c1*(l_M_tilde - 1))/c0) - 1)/(exp(c1) - 1)

        return (exp((c1*UnevaluatedExpr(l_M_tilde - 1))/c0) - 1)/(exp(c1) - 1)

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        l_M_tilde, c0, c1 = self.args
        if argindex == 1:
            return c1*exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0)/(c0*(exp(c1) - 1))
        elif argindex == 2:
            return (
                -c1*exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0)
                *UnevaluatedExpr(l_M_tilde - 1)/(c0**2*(exp(c1) - 1))
            )
        elif argindex == 3:
            return (
                -exp(c1)*(-1 + exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0))/(exp(c1) - 1)**2
                + exp(c1*UnevaluatedExpr(l_M_tilde - 1)/c0)*(l_M_tilde - 1)/(c0*(exp(c1) - 1))
            )

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceLengthPassiveInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        l_M_tilde = self.args[0]
        _l_M_tilde = printer._print(l_M_tilde)
        return r'\operatorname{fl}^M_{pas} \left( %s \right)' % _l_M_tilde


class FiberForceLengthPassiveInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse passive muscle fiber force-length curve based on De Groote et
    al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber length that produces a specific normalized
    passive muscle fiber force.

    The function is defined by the equation:

    ${fl^M_{pas}}^{-1} = \frac{c_0 \log{\left(\exp{c_1} - 1\right)fl^M_pas + 1}}{c_1} + 1$

    with constant values of $c_0 = 0.6$ and $c_1 = 4.0$. This function is the
    exact analytical inverse of the related tendon force-length curve
    ``FiberForceLengthPassiveDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    passive fiber force very close to 0 for all normalized fiber lengths
    between 0 and 1.

    Examples
    ========

    The preferred way to instantiate
    :class:`FiberForceLengthPassiveInverseDeGroote2016` is using the
    :meth:`~.with_defaults` constructor because this will automatically populate the
    constants within the characteristic curve equation with the floating point
    values from the original publication. This constructor takes a single
    argument corresponding to the normalized passive muscle fiber length-force
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fl_M_pas`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthPassiveInverseDeGroote2016
    >>> fl_M_pas = Symbol('fl_M_pas')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016.with_defaults(fl_M_pas)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, 0.6, 4.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1 = symbols('c0 c1')
    >>> l_M_tilde = FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)
    >>> l_M_tilde
    FiberForceLengthPassiveInverseDeGroote2016(fl_M_pas, c0, c1)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> l_M_tilde.doit(evaluate=False)
    c0*log(1 + fl_M_pas*(exp(c1) - 1))/c1 + 1

    The function can also be differentiated. We'll differentiate with respect
    to fl_M_pas using the ``diff`` method on an instance with the single positional
    argument ``fl_M_pas``.

    >>> l_M_tilde.diff(fl_M_pas)
    c0*(exp(c1) - 1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fl_M_pas):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber passive force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = 0.6$
        $c_1 = 4.0$

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
        c0 = Float('0.6')
        c1 = Float('4.0')
        return cls(fl_M_pas, c0, c1)

    @classmethod
    def eval(cls, fl_M_pas, c0, c1):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fl_M_pas : Any (sympifiable)
            Normalized passive muscle fiber force.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.6``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``4.0``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_T_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        fl_M_pas, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fl_M_pas = fl_M_pas.doit(deep=deep, **hints)
            c0, c1 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1 = constants

        if evaluate:
            return c0*log(fl_M_pas*(exp(c1) - 1) + 1)/c1 + 1

        return c0*log(UnevaluatedExpr(fl_M_pas*(exp(c1) - 1)) + 1)/c1 + 1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        fl_M_pas, c0, c1 = self.args
        if argindex == 1:
            return c0*(exp(c1) - 1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))
        elif argindex == 2:
            return log(fl_M_pas*(exp(c1) - 1) + 1)/c1
        elif argindex == 3:
            return (
                c0*fl_M_pas*exp(c1)/(c1*(fl_M_pas*(exp(c1) - 1) + 1))
                - c0*log(fl_M_pas*(exp(c1) - 1) + 1)/c1**2
            )

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceLengthPassiveDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        fl_M_pas = self.args[0]
        _fl_M_pas = printer._print(fl_M_pas)
        return r'\left( \operatorname{fl}^M_{pas} \right)^{-1} \left( %s \right)' % _fl_M_pas


class FiberForceLengthActiveDeGroote2016(CharacteristicCurveFunction):
    r"""Active muscle fiber force-length curve based on De Groote et al., 2016
    [1]_.

    Explanation
    ===========

    The function is defined by the equation:

    $fl_{\text{act}}^M = c_0 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_1}{c_2 + c_3 \tilde{l}^M}\right)^2\right)
    + c_4 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_5}{c_6 + c_7 \tilde{l}^M}\right)^2\right)
    + c_8 \exp\left(-\frac{1}{2}\left(\frac{\tilde{l}^M - c_9}{c_{10} + c_{11} \tilde{l}^M}\right)^2\right)$

    with constant values of $c0 = 0.814$, $c1 = 1.06$, $c2 = 0.162$,
    $c3 = 0.0633$, $c4 = 0.433$, $c5 = 0.717$, $c6 = -0.0299$, $c7 = 0.2$,
    $c8 = 0.1$, $c9 = 1.0$, $c10 = 0.354$, and $c11 = 0.0$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    active fiber force of 1 at a normalized fiber length of 1, and an active
    fiber force of 0 at normalized fiber lengths of 0 and 2.

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceLengthActiveDeGroote2016` is
    using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber length. We'll
    create a :class:`~.Symbol` called ``l_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceLengthActiveDeGroote2016
    >>> l_M_tilde = Symbol('l_M_tilde')
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

    It's also possible to populate the two constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = symbols('c0:12')
    >>> fl_M = FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3,
    ...     c4, c5, c6, c7, c8, c9, c10, c11)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M_tilde, c0, c1, c2, c3, c4, c5, c6,
    c7, c8, c9, c10, c11)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``l_M`` and
    ``l_M_opt``, representing muscle fiber length and optimal muscle fiber
    length respectively. We can then represent ``l_M_tilde`` as an expression,
    the ratio of these.

    >>> l_M, l_M_opt = symbols('l_M l_M_opt')
    >>> l_M_tilde = l_M/l_M_opt
    >>> fl_M = FiberForceLengthActiveDeGroote2016.with_defaults(l_M_tilde)
    >>> fl_M
    FiberForceLengthActiveDeGroote2016(l_M/l_M_opt, 0.814, 1.06, 0.162, 0.0633,
    0.433, 0.717, -0.0299, 0.2, 0.1, 1.0, 0.354, 0.0)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fl_M.doit(evaluate=False)
    0.814*exp(-(l_M/l_M_opt
    - 1.06)**2/(2*(0.0633*l_M/l_M_opt + 0.162)**2))
    + 0.433*exp(-(l_M/l_M_opt - 0.717)**2/(2*(0.2*l_M/l_M_opt - 0.0299)**2))
    + 0.1*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)

    The function can also be differentiated. We'll differentiate with respect
    to l_M using the ``diff`` method on an instance with the single positional
    argument ``l_M``.

    >>> fl_M.diff(l_M)
    ((-0.79798269973507*l_M/l_M_opt
    + 0.79798269973507)*exp(-3.98991349867535*(l_M/l_M_opt - 1.0)**2)
    + (0.433*(-l_M/l_M_opt + 0.717)/(0.2*l_M/l_M_opt - 0.0299)**2
    + 0.0866*(l_M/l_M_opt - 0.717)**2/(0.2*l_M/l_M_opt
    - 0.0299)**3)*exp(-(l_M/l_M_opt - 0.717)**2/(2*(0.2*l_M/l_M_opt - 0.0299)**2))
    + (0.814*(-l_M/l_M_opt + 1.06)/(0.0633*l_M/l_M_opt
    + 0.162)**2 + 0.0515262*(l_M/l_M_opt
    - 1.06)**2/(0.0633*l_M/l_M_opt
    + 0.162)**3)*exp(-(l_M/l_M_opt
    - 1.06)**2/(2*(0.0633*l_M/l_M_opt + 0.162)**2)))/l_M_opt

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, l_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber act force-length
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c0 = 0.814$
        $c1 = 1.06$
        $c2 = 0.162$
        $c3 = 0.0633$
        $c4 = 0.433$
        $c5 = 0.717$
        $c6 = -0.0299$
        $c7 = 0.2$
        $c8 = 0.1$
        $c9 = 1.0$
        $c10 = 0.354$
        $c11 = 0.0$

        Parameters
        ==========

        fl_M_act : Any (sympifiable)
            Normalized passive muscle fiber force as a function of muscle fiber
            length.

        """
        c0 = Float('0.814')
        c1 = Float('1.06')
        c2 = Float('0.162')
        c3 = Float('0.0633')
        c4 = Float('0.433')
        c5 = Float('0.717')
        c6 = Float('-0.0299')
        c7 = Float('0.2')
        c8 = Float('0.1')
        c9 = Float('1.0')
        c10 = Float('0.354')
        c11 = Float('0.0')
        return cls(l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11)

    @classmethod
    def eval(cls, l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11):
        """Evaluation of basic inputs.

        Parameters
        ==========

        l_M_tilde : Any (sympifiable)
            Normalized muscle fiber length.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``0.814``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``1.06``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``0.162``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.0633``.
        c4 : Any (sympifiable)
            The fifth constant in the characteristic equation. The published
            value is ``0.433``.
        c5 : Any (sympifiable)
            The sixth constant in the characteristic equation. The published
            value is ``0.717``.
        c6 : Any (sympifiable)
            The seventh constant in the characteristic equation. The published
            value is ``-0.0299``.
        c7 : Any (sympifiable)
            The eighth constant in the characteristic equation. The published
            value is ``0.2``.
        c8 : Any (sympifiable)
            The ninth constant in the characteristic equation. The published
            value is ``0.1``.
        c9 : Any (sympifiable)
            The tenth constant in the characteristic equation. The published
            value is ``1.0``.
        c10 : Any (sympifiable)
            The eleventh constant in the characteristic equation. The published
            value is ``0.354``.
        c11 : Any (sympifiable)
            The tweflth constant in the characteristic equation. The published
            value is ``0.0``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``l_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        l_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            l_M_tilde = l_M_tilde.doit(deep=deep, **hints)
            constants = [c.doit(deep=deep, **hints) for c in constants]
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = constants

        if evaluate:
            return (
                c0*exp(-(((l_M_tilde - c1)/(c2 + c3*l_M_tilde))**2)/2)
                + c4*exp(-(((l_M_tilde - c5)/(c6 + c7*l_M_tilde))**2)/2)
                + c8*exp(-(((l_M_tilde - c9)/(c10 + c11*l_M_tilde))**2)/2)
            )

        return (
            c0*exp(-((UnevaluatedExpr(l_M_tilde - c1)/(c2 + c3*l_M_tilde))**2)/2)
            + c4*exp(-((UnevaluatedExpr(l_M_tilde - c5)/(c6 + c7*l_M_tilde))**2)/2)
            + c8*exp(-((UnevaluatedExpr(l_M_tilde - c9)/(c10 + c11*l_M_tilde))**2)/2)
        )

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        l_M_tilde, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = self.args
        if argindex == 1:
            return (
                c0*(
                    c3*(l_M_tilde - c1)**2/(c2 + c3*l_M_tilde)**3
                    + (c1 - l_M_tilde)/((c2 + c3*l_M_tilde)**2)
                )*exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
                + c4*(
                    c7*(l_M_tilde - c5)**2/(c6 + c7*l_M_tilde)**3
                    + (c5 - l_M_tilde)/((c6 + c7*l_M_tilde)**2)
                )*exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
                + c8*(
                    c11*(l_M_tilde - c9)**2/(c10 + c11*l_M_tilde)**3
                    + (c9 - l_M_tilde)/((c10 + c11*l_M_tilde)**2)
                )*exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
            )
        elif argindex == 2:
            return exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
        elif argindex == 3:
            return (
                c0*(l_M_tilde - c1)/(c2 + c3*l_M_tilde)**2
                *exp(-(l_M_tilde - c1)**2 /(2*(c2 + c3*l_M_tilde)**2))
            )
        elif argindex == 4:
            return (
                c0*(l_M_tilde - c1)**2/(c2 + c3*l_M_tilde)**3
                *exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
            )
        elif argindex == 5:
            return (
                c0*l_M_tilde*(l_M_tilde - c1)**2/(c2 + c3*l_M_tilde)**3
                *exp(-(l_M_tilde - c1)**2/(2*(c2 + c3*l_M_tilde)**2))
            )
        elif argindex == 6:
            return exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
        elif argindex == 7:
            return (
                c4*(l_M_tilde - c5)/(c6 + c7*l_M_tilde)**2
                *exp(-(l_M_tilde - c5)**2 /(2*(c6 + c7*l_M_tilde)**2))
            )
        elif argindex == 8:
            return (
                c4*(l_M_tilde - c5)**2/(c6 + c7*l_M_tilde)**3
                *exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
            )
        elif argindex == 9:
            return (
                c4*l_M_tilde*(l_M_tilde - c5)**2/(c6 + c7*l_M_tilde)**3
                *exp(-(l_M_tilde - c5)**2/(2*(c6 + c7*l_M_tilde)**2))
            )
        elif argindex == 10:
            return exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
        elif argindex == 11:
            return (
                c8*(l_M_tilde - c9)/(c10 + c11*l_M_tilde)**2
                *exp(-(l_M_tilde - c9)**2 /(2*(c10 + c11*l_M_tilde)**2))
            )
        elif argindex == 12:
            return (
                c8*(l_M_tilde - c9)**2/(c10 + c11*l_M_tilde)**3
                *exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
            )
        elif argindex == 13:
            return (
                c8*l_M_tilde*(l_M_tilde - c9)**2/(c10 + c11*l_M_tilde)**3
                *exp(-(l_M_tilde - c9)**2/(2*(c10 + c11*l_M_tilde)**2))
            )

        raise ArgumentIndexError(self, argindex)

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        l_M_tilde = self.args[0]
        _l_M_tilde = printer._print(l_M_tilde)
        return r'\operatorname{fl}^M_{act} \left( %s \right)' % _l_M_tilde


class FiberForceVelocityDeGroote2016(CharacteristicCurveFunction):
    r"""Muscle fiber force-velocity curve based on De Groote et al., 2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber force produced as a function of
    normalized tendon velocity.

    The function is defined by the equation:

    $fv^M = c_0 \log{\left(c_1 \tilde{v}_m + c_2\right) + \sqrt{\left(c_1 \tilde{v}_m + c_2\right)^2 + 1}} + c_3$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityDeGroote2016` is using
    the :meth:`~.with_defaults` constructor because this will automatically populate
    the constants within the characteristic curve equation with the floating
    point values from the original publication. This constructor takes a single
    argument corresponding to normalized muscle fiber extension velocity. We'll
    create a :class:`~.Symbol` called ``v_M_tilde`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityDeGroote2016
    >>> v_M_tilde = Symbol('v_M_tilde')
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> fv_M = FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M_tilde, c0, c1, c2, c3)

    You don't just have to use symbols as the arguments, it's also possible to
    use expressions. Let's create a new pair of symbols, ``v_M`` and
    ``v_M_max``, representing muscle fiber extension velocity and maximum
    muscle fiber extension velocity respectively. We can then represent
    ``v_M_tilde`` as an expression, the ratio of these.

    >>> v_M, v_M_max = symbols('v_M v_M_max')
    >>> v_M_tilde = v_M/v_M_max
    >>> fv_M = FiberForceVelocityDeGroote2016.with_defaults(v_M_tilde)
    >>> fv_M
    FiberForceVelocityDeGroote2016(v_M/v_M_max, -0.318, -8.149, -0.374, 0.886)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> fv_M.doit(evaluate=False)
    0.886 - 0.318*log(-8.149*v_M/v_M_max - 0.374 + sqrt(1 + (-8.149*v_M/v_M_max
    - 0.374)**2))

    The function can also be differentiated. We'll differentiate with respect
    to v_M using the ``diff`` method on an instance with the single positional
    argument ``v_M``.

    >>> fv_M.diff(v_M)
    2.591382*(1 + (-8.149*v_M/v_M_max - 0.374)**2)**(-1/2)/v_M_max

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, v_M_tilde):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the muscle fiber force-velocity function
        using the four constant values specified in the original publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        return cls(v_M_tilde, c0, c1, c2, c3)

    @classmethod
    def eval(cls, v_M_tilde, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        v_M_tilde : Any (sympifiable)
            Normalized muscle fiber extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``v_M_tilde`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        v_M_tilde, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            v_M_tilde = v_M_tilde.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return c0*log(c1*v_M_tilde + c2 + sqrt((c1*v_M_tilde + c2)**2 + 1)) + c3

        return c0*log(c1*v_M_tilde + c2 + sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)) + c3

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        v_M_tilde, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return c0*c1/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 2:
            return log(
                c1*v_M_tilde + c2
                + sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
            )
        elif argindex == 3:
            return c0*v_M_tilde/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 4:
            return c0/sqrt(UnevaluatedExpr(c1*v_M_tilde + c2)**2 + 1)
        elif argindex == 5:
            return Integer(1)

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceVelocityInverseDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        v_M_tilde = self.args[0]
        _v_M_tilde = printer._print(v_M_tilde)
        return r'\operatorname{fv}^M \left( %s \right)' % _v_M_tilde


class FiberForceVelocityInverseDeGroote2016(CharacteristicCurveFunction):
    r"""Inverse muscle fiber force-velocity curve based on De Groote et al.,
    2016 [1]_.

    Explanation
    ===========

    Gives the normalized muscle fiber velocity that produces a specific
    normalized muscle fiber force.

    The function is defined by the equation:

    ${fv^M}^{-1} = \frac{\sinh{\frac{fv^M - c_3}{c_0}} - c_2}{c_1}$

    with constant values of $c_0 = -0.318$, $c_1 = -8.149$, $c_2 = -0.374$, and
    $c_3 = 0.886$. This function is the exact analytical inverse of the related
    muscle fiber force-velocity curve ``FiberForceVelocityDeGroote2016``.

    While it is possible to change the constant values, these were carefully
    selected in the original publication to give the characteristic curve
    specific and required properties. For example, the function produces a
    normalized muscle fiber force of 1 when the muscle fibers are contracting
    isometrically (they have an extension rate of 0).

    Examples
    ========

    The preferred way to instantiate :class:`FiberForceVelocityInverseDeGroote2016`
    is using the :meth:`~.with_defaults` constructor because this will automatically
    populate the constants within the characteristic curve equation with the
    floating point values from the original publication. This constructor takes
    a single argument corresponding to normalized muscle fiber force-velocity
    component of the muscle fiber force. We'll create a :class:`~.Symbol` called
    ``fv_M`` to represent this.

    >>> from sympy import Symbol
    >>> from sympy.physics.biomechanics import FiberForceVelocityInverseDeGroote2016
    >>> fv_M = Symbol('fv_M')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016.with_defaults(fv_M)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, -0.318, -8.149, -0.374, 0.886)

    It's also possible to populate the four constants with your own values too.

    >>> from sympy import symbols
    >>> c0, c1, c2, c3 = symbols('c0 c1 c2 c3')
    >>> v_M_tilde = FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)
    >>> v_M_tilde
    FiberForceVelocityInverseDeGroote2016(fv_M, c0, c1, c2, c3)

    To inspect the actual symbolic expression that this function represents,
    we can call the :meth:`~.doit` method on an instance. We'll use the keyword
    argument ``evaluate=False`` as this will keep the expression in its
    canonical form and won't simplify any constants.

    >>> v_M_tilde.doit(evaluate=False)
    (-c2 + sinh((-c3 + fv_M)/c0))/c1

    The function can also be differentiated. We'll differentiate with respect
    to fv_M using the ``diff`` method on an instance with the single positional
    argument ``fv_M``.

    >>> v_M_tilde.diff(fv_M)
    cosh((-c3 + fv_M)/c0)/(c0*c1)

    References
    ==========

    .. [1] De Groote, F., Kinney, A. L., Rao, A. V., & Fregly, B. J., Evaluation
           of direct collocation optimal control problem formulations for
           solving the muscle redundancy problem, Annals of biomedical
           engineering, 44(10), (2016) pp. 2922-2936

    """

    @classmethod
    def with_defaults(cls, fv_M):
        r"""Recommended constructor that will use the published constants.

        Explanation
        ===========

        Returns a new instance of the inverse muscle fiber force-velocity
        function using the four constant values specified in the original
        publication.

        These have the values:

        $c_0 = -0.318$
        $c_1 = -8.149$
        $c_2 = -0.374$
        $c_3 = 0.886$

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber extension velocity.

        """
        c0 = Float('-0.318')
        c1 = Float('-8.149')
        c2 = Float('-0.374')
        c3 = Float('0.886')
        return cls(fv_M, c0, c1, c2, c3)

    @classmethod
    def eval(cls, fv_M, c0, c1, c2, c3):
        """Evaluation of basic inputs.

        Parameters
        ==========

        fv_M : Any (sympifiable)
            Normalized muscle fiber force as a function of muscle fiber
            extension velocity.
        c0 : Any (sympifiable)
            The first constant in the characteristic equation. The published
            value is ``-0.318``.
        c1 : Any (sympifiable)
            The second constant in the characteristic equation. The published
            value is ``-8.149``.
        c2 : Any (sympifiable)
            The third constant in the characteristic equation. The published
            value is ``-0.374``.
        c3 : Any (sympifiable)
            The fourth constant in the characteristic equation. The published
            value is ``0.886``.

        """
        pass

    def _eval_evalf(self, prec):
        """Evaluate the expression numerically using ``evalf``."""
        return self.doit(deep=False, evaluate=False)._eval_evalf(prec)

    def doit(self, deep=True, evaluate=True, **hints):
        """Evaluate the expression defining the function.

        Parameters
        ==========

        deep : bool
            Whether ``doit`` should be recursively called. Default is ``True``.
        evaluate : bool.
            Whether the SymPy expression should be evaluated as it is
            constructed. If ``False``, then no constant folding will be
            conducted which will leave the expression in a more numerically-
            stable for values of ``fv_M`` that correspond to a sensible
            operating range for a musculotendon. Default is ``True``.
        **kwargs : dict[str, Any]
            Additional keyword argument pairs to be recursively passed to
            ``doit``.

        """
        fv_M, *constants = self.args
        if deep:
            hints['evaluate'] = evaluate
            fv_M = fv_M.doit(deep=deep, **hints)
            c0, c1, c2, c3 = [c.doit(deep=deep, **hints) for c in constants]
        else:
            c0, c1, c2, c3 = constants

        if evaluate:
            return (sinh((fv_M - c3)/c0) - c2)/c1

        return (sinh(UnevaluatedExpr(fv_M - c3)/c0) - c2)/c1

    def fdiff(self, argindex=1):
        """Derivative of the function with respect to a single argument.

        Parameters
        ==========

        argindex : int
            The index of the function's arguments with respect to which the
            derivative should be taken. Argument indexes start at ``1``.
            Default is ``1``.

        """
        fv_M, c0, c1, c2, c3 = self.args
        if argindex == 1:
            return cosh((fv_M - c3)/c0)/(c0*c1)
        elif argindex == 2:
            return (c3 - fv_M)*cosh((fv_M - c3)/c0)/(c0**2*c1)
        elif argindex == 3:
            return (c2 - sinh((fv_M - c3)/c0))/c1**2
        elif argindex == 4:
            return -1/c1
        elif argindex == 5:
            return -cosh((fv_M - c3)/c0)/(c0*c1)

        raise ArgumentIndexError(self, argindex)

    def inverse(self, argindex=1):
        """Inverse function.

        Parameters
        ==========

        argindex : int
            Value to start indexing the arguments at. Default is ``1``.

        """
        return FiberForceVelocityDeGroote2016

    def _latex(self, printer):
        """Print a LaTeX representation of the function defining the curve.

        Parameters
        ==========

        printer : Printer
            The printer to be used to print the LaTeX string representation.

        """
        fv_M = self.args[0]
        _fv_M = printer._print(fv_M)
        return r'\left( \operatorname{fv}^M \right)^{-1} \left( %s \right)' % _fv_M


@dataclass(frozen=True)
class CharacteristicCurveCollection:
    """Simple data container to group together related characteristic curves."""
    tendon_force_length: CharacteristicCurveFunction
    tendon_force_length_inverse: CharacteristicCurveFunction
    fiber_force_length_passive: CharacteristicCurveFunction
    fiber_force_length_passive_inverse: CharacteristicCurveFunction
    fiber_force_length_active: CharacteristicCurveFunction
    fiber_force_velocity: CharacteristicCurveFunction
    fiber_force_velocity_inverse: CharacteristicCurveFunction

    def __iter__(self):
        """Iterator support for ``CharacteristicCurveCollection``."""
        yield self.tendon_force_length
        yield self.tendon_force_length_inverse
        yield self.fiber_force_length_passive
        yield self.fiber_force_length_passive_inverse
        yield self.fiber_force_length_active
        yield self.fiber_force_velocity
        yield self.fiber_force_velocity_inverse

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\multiarray.py ===
"""
Create the numpy._core.multiarray namespace for backward compatibility.
In v1.16 the multiarray and umath c-extension modules were merged into
a single _multiarray_umath extension module. So we replicate the old
namespace by importing from the extension module.

"""

import functools

from . import _multiarray_umath, overrides
from ._multiarray_umath import *  # noqa: F403

# These imports are needed for backward compatibility,
# do not change them. issue gh-15518
# _get_ndarray_c_version is semi-public, on purpose not added to __all__
from ._multiarray_umath import (  # noqa: F401
    _ARRAY_API,
    _flagdict,
    _get_madvise_hugepage,
    _get_ndarray_c_version,
    _monotonicity,
    _place,
    _reconstruct,
    _set_madvise_hugepage,
    _vec_string,
    from_dlpack,
)

__all__ = [
    '_ARRAY_API', 'ALLOW_THREADS', 'BUFSIZE', 'CLIP', 'DATETIMEUNITS',
    'ITEM_HASOBJECT', 'ITEM_IS_POINTER', 'LIST_PICKLE', 'MAXDIMS',
    'MAY_SHARE_BOUNDS', 'MAY_SHARE_EXACT', 'NEEDS_INIT', 'NEEDS_PYAPI',
    'RAISE', 'USE_GETITEM', 'USE_SETITEM', 'WRAP',
    '_flagdict', 'from_dlpack', '_place', '_reconstruct', '_vec_string',
    '_monotonicity', 'add_docstring', 'arange', 'array', 'asarray',
    'asanyarray', 'ascontiguousarray', 'asfortranarray', 'bincount',
    'broadcast', 'busday_count', 'busday_offset', 'busdaycalendar', 'can_cast',
    'compare_chararrays', 'concatenate', 'copyto', 'correlate', 'correlate2',
    'count_nonzero', 'c_einsum', 'datetime_as_string', 'datetime_data',
    'dot', 'dragon4_positional', 'dragon4_scientific', 'dtype',
    'empty', 'empty_like', 'error', 'flagsobj', 'flatiter', 'format_longfloat',
    'frombuffer', 'fromfile', 'fromiter', 'fromstring',
    'get_handler_name', 'get_handler_version', 'inner', 'interp',
    'interp_complex', 'is_busday', 'lexsort', 'matmul', 'vecdot',
    'may_share_memory', 'min_scalar_type', 'ndarray', 'nditer', 'nested_iters',
    'normalize_axis_index', 'packbits', 'promote_types', 'putmask',
    'ravel_multi_index', 'result_type', 'scalar', 'set_datetimeparse_function',
    'set_typeDict', 'shares_memory', 'typeinfo',
    'unpackbits', 'unravel_index', 'vdot', 'where', 'zeros']

# For backward compatibility, make sure pickle imports
# these functions from here
_reconstruct.__module__ = 'numpy._core.multiarray'
scalar.__module__ = 'numpy._core.multiarray'


from_dlpack.__module__ = 'numpy'
arange.__module__ = 'numpy'
array.__module__ = 'numpy'
asarray.__module__ = 'numpy'
asanyarray.__module__ = 'numpy'
ascontiguousarray.__module__ = 'numpy'
asfortranarray.__module__ = 'numpy'
datetime_data.__module__ = 'numpy'
empty.__module__ = 'numpy'
frombuffer.__module__ = 'numpy'
fromfile.__module__ = 'numpy'
fromiter.__module__ = 'numpy'
frompyfunc.__module__ = 'numpy'
fromstring.__module__ = 'numpy'
may_share_memory.__module__ = 'numpy'
nested_iters.__module__ = 'numpy'
promote_types.__module__ = 'numpy'
zeros.__module__ = 'numpy'
normalize_axis_index.__module__ = 'numpy.lib.array_utils'
add_docstring.__module__ = 'numpy.lib'
compare_chararrays.__module__ = 'numpy.char'


def _override___module__():
    namespace_names = globals()
    for ufunc_name in [
        'absolute', 'arccos', 'arccosh', 'add', 'arcsin', 'arcsinh', 'arctan',
        'arctan2', 'arctanh', 'bitwise_and', 'bitwise_count', 'invert',
        'left_shift', 'bitwise_or', 'right_shift', 'bitwise_xor', 'cbrt',
        'ceil', 'conjugate', 'copysign', 'cos', 'cosh', 'deg2rad', 'degrees',
        'divide', 'divmod', 'equal', 'exp', 'exp2', 'expm1', 'fabs',
        'float_power', 'floor', 'floor_divide', 'fmax', 'fmin', 'fmod',
        'frexp', 'gcd', 'greater', 'greater_equal', 'heaviside', 'hypot',
        'isfinite', 'isinf', 'isnan', 'isnat', 'lcm', 'ldexp', 'less',
        'less_equal', 'log', 'log10', 'log1p', 'log2', 'logaddexp',
        'logaddexp2', 'logical_and', 'logical_not', 'logical_or',
        'logical_xor', 'matmul', 'matvec', 'maximum', 'minimum', 'remainder',
        'modf', 'multiply', 'negative', 'nextafter', 'not_equal', 'positive',
        'power', 'rad2deg', 'radians', 'reciprocal', 'rint', 'sign', 'signbit',
        'sin', 'sinh', 'spacing', 'sqrt', 'square', 'subtract', 'tan', 'tanh',
        'trunc', 'vecdot', 'vecmat',
    ]:
        ufunc = namespace_names[ufunc_name]
        ufunc.__module__ = "numpy"
        ufunc.__qualname__ = ufunc_name


_override___module__()


# We can't verify dispatcher signatures because NumPy's C functions don't
# support introspection.
array_function_from_c_func_and_dispatcher = functools.partial(
    overrides.array_function_from_dispatcher,
    module='numpy', docs_from_dispatcher=True, verify=False)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.empty_like)
def empty_like(
    prototype, dtype=None, order=None, subok=None, shape=None, *, device=None
):
    """
    empty_like(prototype, dtype=None, order='K', subok=True, shape=None, *,
               device=None)

    Return a new array with the same shape and type as a given array.

    Parameters
    ----------
    prototype : array_like
        The shape and data-type of `prototype` define these same attributes
        of the returned array.
    dtype : data-type, optional
        Overrides the data type of the result.
    order : {'C', 'F', 'A', or 'K'}, optional
        Overrides the memory layout of the result. 'C' means C-order,
        'F' means F-order, 'A' means 'F' if `prototype` is Fortran
        contiguous, 'C' otherwise. 'K' means match the layout of `prototype`
        as closely as possible.
    subok : bool, optional.
        If True, then the newly created array will use the sub-class
        type of `prototype`, otherwise it will be a base-class array. Defaults
        to True.
    shape : int or sequence of ints, optional.
        Overrides the shape of the result. If order='K' and the number of
        dimensions is unchanged, will try to keep order, otherwise,
        order='C' is implied.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    out : ndarray
        Array of uninitialized (arbitrary) data with the same
        shape and type as `prototype`.

    See Also
    --------
    ones_like : Return an array of ones with shape and type of input.
    zeros_like : Return an array of zeros with shape and type of input.
    full_like : Return a new array with shape of input filled with value.
    empty : Return a new uninitialized array.

    Notes
    -----
    Unlike other array creation functions (e.g. `zeros_like`, `ones_like`,
    `full_like`), `empty_like` does not initialize the values of the array,
    and may therefore be marginally faster. However, the values stored in the
    newly allocated array are arbitrary. For reproducible behavior, be sure
    to set each element of the array before reading.

    Examples
    --------
    >>> import numpy as np
    >>> a = ([1,2,3], [4,5,6])                         # a is array-like
    >>> np.empty_like(a)
    array([[-1073741821, -1073741821,           3],    # uninitialized
           [          0,           0, -1073741821]])
    >>> a = np.array([[1., 2., 3.],[4.,5.,6.]])
    >>> np.empty_like(a)
    array([[ -2.00000715e+000,   1.48219694e-323,  -2.00000572e+000], # uninitialized
           [  4.38791518e-305,  -2.00000715e+000,   4.17269252e-309]])

    """
    return (prototype,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.concatenate)
def concatenate(arrays, axis=None, out=None, *, dtype=None, casting=None):
    """
    concatenate(
        (a1, a2, ...),
        axis=0,
        out=None,
        dtype=None,
        casting="same_kind"
    )

    Join a sequence of arrays along an existing axis.

    Parameters
    ----------
    a1, a2, ... : sequence of array_like
        The arrays must have the same shape, except in the dimension
        corresponding to `axis` (the first, by default).
    axis : int, optional
        The axis along which the arrays will be joined.  If axis is None,
        arrays are flattened before use.  Default is 0.
    out : ndarray, optional
        If provided, the destination to place the result. The shape must be
        correct, matching that of what concatenate would have returned if no
        out argument were specified.
    dtype : str or dtype
        If provided, the destination array will have this dtype. Cannot be
        provided together with `out`.

        .. versionadded:: 1.20.0

    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur. Defaults to 'same_kind'.
        For a description of the options, please see :term:`casting`.

        .. versionadded:: 1.20.0

    Returns
    -------
    res : ndarray
        The concatenated array.

    See Also
    --------
    ma.concatenate : Concatenate function that preserves input masks.
    array_split : Split an array into multiple sub-arrays of equal or
                  near-equal size.
    split : Split array into a list of multiple sub-arrays of equal size.
    hsplit : Split array into multiple sub-arrays horizontally (column wise).
    vsplit : Split array into multiple sub-arrays vertically (row wise).
    dsplit : Split array into multiple sub-arrays along the 3rd axis (depth).
    stack : Stack a sequence of arrays along a new axis.
    block : Assemble arrays from blocks.
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).
    column_stack : Stack 1-D arrays as columns into a 2-D array.

    Notes
    -----
    When one or more of the arrays to be concatenated is a MaskedArray,
    this function will return a MaskedArray object instead of an ndarray,
    but the input masks are *not* preserved. In cases where a MaskedArray
    is expected as input, use the ma.concatenate function from the masked
    array module instead.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6]])
    >>> np.concatenate((a, b), axis=0)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> np.concatenate((a, b.T), axis=1)
    array([[1, 2, 5],
           [3, 4, 6]])
    >>> np.concatenate((a, b), axis=None)
    array([1, 2, 3, 4, 5, 6])

    This function will not preserve masking of MaskedArray inputs.

    >>> a = np.ma.arange(3)
    >>> a[1] = np.ma.masked
    >>> b = np.arange(2, 5)
    >>> a
    masked_array(data=[0, --, 2],
                 mask=[False,  True, False],
           fill_value=999999)
    >>> b
    array([2, 3, 4])
    >>> np.concatenate([a, b])
    masked_array(data=[0, 1, 2, 2, 3, 4],
                 mask=False,
           fill_value=999999)
    >>> np.ma.concatenate([a, b])
    masked_array(data=[0, --, 2, 2, 3, 4],
                 mask=[False,  True, False, False, False, False],
           fill_value=999999)

    """
    if out is not None:
        # optimize for the typical case where only arrays is provided
        arrays = list(arrays)
        arrays.append(out)
    return arrays


@array_function_from_c_func_and_dispatcher(_multiarray_umath.inner)
def inner(a, b):
    """
    inner(a, b, /)

    Inner product of two arrays.

    Ordinary inner product of vectors for 1-D arrays (without complex
    conjugation), in higher dimensions a sum product over the last axes.

    Parameters
    ----------
    a, b : array_like
        If `a` and `b` are nonscalar, their last dimensions must match.

    Returns
    -------
    out : ndarray
        If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        ``out.shape = (*a.shape[:-1], *b.shape[:-1])``

    Raises
    ------
    ValueError
        If both `a` and `b` are nonscalar and their last dimensions have
        different sizes.

    See Also
    --------
    tensordot : Sum products over arbitrary axes.
    dot : Generalised matrix product, using second last dimension of `b`.
    vecdot : Vector dot product of two arrays.
    einsum : Einstein summation convention.

    Notes
    -----
    For vectors (1-D arrays) it computes the ordinary inner-product::

        np.inner(a, b) = sum(a[:]*b[:])

    More generally, if ``ndim(a) = r > 0`` and ``ndim(b) = s > 0``::

        np.inner(a, b) = np.tensordot(a, b, axes=(-1,-1))

    or explicitly::

        np.inner(a, b)[i0,...,ir-2,j0,...,js-2]
             = sum(a[i0,...,ir-2,:]*b[j0,...,js-2,:])

    In addition `a` or `b` may be scalars, in which case::

       np.inner(a,b) = a*b

    Examples
    --------
    Ordinary inner product for vectors:

    >>> import numpy as np
    >>> a = np.array([1,2,3])
    >>> b = np.array([0,1,0])
    >>> np.inner(a, b)
    2

    Some multidimensional examples:

    >>> a = np.arange(24).reshape((2,3,4))
    >>> b = np.arange(4)
    >>> c = np.inner(a, b)
    >>> c.shape
    (2, 3)
    >>> c
    array([[ 14,  38,  62],
           [ 86, 110, 134]])

    >>> a = np.arange(2).reshape((1,1,2))
    >>> b = np.arange(6).reshape((3,2))
    >>> c = np.inner(a, b)
    >>> c.shape
    (1, 1, 3)
    >>> c
    array([[[1, 3, 5]]])

    An example where `b` is a scalar:

    >>> np.inner(np.eye(2), 7)
    array([[7., 0.],
           [0., 7.]])

    """
    return (a, b)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.where)
def where(condition, x=None, y=None):
    """
    where(condition, [x, y], /)

    Return elements chosen from `x` or `y` depending on `condition`.

    .. note::
        When only `condition` is provided, this function is a shorthand for
        ``np.asarray(condition).nonzero()``. Using `nonzero` directly should be
        preferred, as it behaves correctly for subclasses. The rest of this
        documentation covers only the case where all three arguments are
        provided.

    Parameters
    ----------
    condition : array_like, bool
        Where True, yield `x`, otherwise yield `y`.
    x, y : array_like
        Values from which to choose. `x`, `y` and `condition` need to be
        broadcastable to some shape.

    Returns
    -------
    out : ndarray
        An array with elements from `x` where `condition` is True, and elements
        from `y` elsewhere.

    See Also
    --------
    choose
    nonzero : The function that is called when x and y are omitted

    Notes
    -----
    If all the arrays are 1-D, `where` is equivalent to::

        [xv if c else yv
         for c, xv, yv in zip(condition, x, y)]

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.where(a < 5, a, 10*a)
    array([ 0,  1,  2,  3,  4, 50, 60, 70, 80, 90])

    This can be used on multidimensional arrays too:

    >>> np.where([[True, False], [True, True]],
    ...          [[1, 2], [3, 4]],
    ...          [[9, 8], [7, 6]])
    array([[1, 8],
           [3, 4]])

    The shapes of x, y, and the condition are broadcast together:

    >>> x, y = np.ogrid[:3, :4]
    >>> np.where(x < y, x, 10 + y)  # both x and 10+y are broadcast
    array([[10,  0,  0,  0],
           [10, 11,  1,  1],
           [10, 11, 12,  2]])

    >>> a = np.array([[0, 1, 2],
    ...               [0, 2, 4],
    ...               [0, 3, 6]])
    >>> np.where(a < 4, a, -1)  # -1 is broadcast
    array([[ 0,  1,  2],
           [ 0,  2, -1],
           [ 0,  3, -1]])
    """
    return (condition, x, y)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.lexsort)
def lexsort(keys, axis=None):
    """
    lexsort(keys, axis=-1)

    Perform an indirect stable sort using a sequence of keys.

    Given multiple sorting keys, lexsort returns an array of integer indices
    that describes the sort order by multiple keys. The last key in the
    sequence is used for the primary sort order, ties are broken by the
    second-to-last key, and so on.

    Parameters
    ----------
    keys : (k, m, n, ...) array-like
        The `k` keys to be sorted. The *last* key (e.g, the last
        row if `keys` is a 2D array) is the primary sort key.
        Each element of `keys` along the zeroth axis must be
        an array-like object of the same shape.
    axis : int, optional
        Axis to be indirectly sorted. By default, sort over the last axis
        of each sequence. Separate slices along `axis` sorted over
        independently; see last example.

    Returns
    -------
    indices : (m, n, ...) ndarray of ints
        Array of indices that sort the keys along the specified axis.

    See Also
    --------
    argsort : Indirect sort.
    ndarray.sort : In-place sort.
    sort : Return a sorted copy of an array.

    Examples
    --------
    Sort names: first by surname, then by name.

    >>> import numpy as np
    >>> surnames =    ('Hertz',    'Galilei', 'Hertz')
    >>> first_names = ('Heinrich', 'Galileo', 'Gustav')
    >>> ind = np.lexsort((first_names, surnames))
    >>> ind
    array([1, 2, 0])

    >>> [surnames[i] + ", " + first_names[i] for i in ind]
    ['Galilei, Galileo', 'Hertz, Gustav', 'Hertz, Heinrich']

    Sort according to two numerical keys, first by elements
    of ``a``, then breaking ties according to elements of ``b``:

    >>> a = [1, 5, 1, 4, 3, 4, 4]  # First sequence
    >>> b = [9, 4, 0, 4, 0, 2, 1]  # Second sequence
    >>> ind = np.lexsort((b, a))  # Sort by `a`, then by `b`
    >>> ind
    array([2, 0, 4, 6, 5, 3, 1])
    >>> [(a[i], b[i]) for i in ind]
    [(1, 0), (1, 9), (3, 0), (4, 1), (4, 2), (4, 4), (5, 4)]

    Compare against `argsort`, which would sort each key independently.

    >>> np.argsort((b, a), kind='stable')
    array([[2, 4, 6, 5, 1, 3, 0],
           [0, 2, 4, 3, 5, 6, 1]])

    To sort lexicographically with `argsort`, we would need to provide a
    structured array.

    >>> x = np.array([(ai, bi) for ai, bi in zip(a, b)],
    ...              dtype = np.dtype([('x', int), ('y', int)]))
    >>> np.argsort(x)  # or np.argsort(x, order=('x', 'y'))
    array([2, 0, 4, 6, 5, 3, 1])

    The zeroth axis of `keys` always corresponds with the sequence of keys,
    so 2D arrays are treated just like other sequences of keys.

    >>> arr = np.asarray([b, a])
    >>> ind2 = np.lexsort(arr)
    >>> np.testing.assert_equal(ind2, ind)

    Accordingly, the `axis` parameter refers to an axis of *each* key, not of
    the `keys` argument itself. For instance, the array ``arr`` is treated as
    a sequence of two 1-D keys, so specifying ``axis=0`` is equivalent to
    using the default axis, ``axis=-1``.

    >>> np.testing.assert_equal(np.lexsort(arr, axis=0),
    ...                         np.lexsort(arr, axis=-1))

    For higher-dimensional arrays, the axis parameter begins to matter. The
    resulting array has the same shape as each key, and the values are what
    we would expect if `lexsort` were performed on corresponding slices
    of the keys independently. For instance,

    >>> x = [[1, 2, 3, 4],
    ...      [4, 3, 2, 1],
    ...      [2, 1, 4, 3]]
    >>> y = [[2, 2, 1, 1],
    ...      [1, 2, 1, 2],
    ...      [1, 1, 2, 1]]
    >>> np.lexsort((x, y), axis=1)
    array([[2, 3, 0, 1],
           [2, 0, 3, 1],
           [1, 0, 3, 2]])

    Each row of the result is what we would expect if we were to perform
    `lexsort` on the corresponding row of the keys:

    >>> for i in range(3):
    ...     print(np.lexsort((x[i], y[i])))
    [2 3 0 1]
    [2 0 3 1]
    [1 0 3 2]

    """
    if isinstance(keys, tuple):
        return keys
    else:
        return (keys,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.can_cast)
def can_cast(from_, to, casting=None):
    """
    can_cast(from_, to, casting='safe')

    Returns True if cast between data types can occur according to the
    casting rule.

    Parameters
    ----------
    from_ : dtype, dtype specifier, NumPy scalar, or array
        Data type, NumPy scalar, or array to cast from.
    to : dtype or dtype specifier
        Data type to cast to.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.

    Returns
    -------
    out : bool
        True if cast can occur according to the casting rule.

    Notes
    -----
    .. versionchanged:: 2.0
       This function does not support Python scalars anymore and does not
       apply any value-based logic for 0-D arrays and NumPy scalars.

    See also
    --------
    dtype, result_type

    Examples
    --------
    Basic examples

    >>> import numpy as np
    >>> np.can_cast(np.int32, np.int64)
    True
    >>> np.can_cast(np.float64, complex)
    True
    >>> np.can_cast(complex, float)
    False

    >>> np.can_cast('i8', 'f8')
    True
    >>> np.can_cast('i8', 'f4')
    False
    >>> np.can_cast('i4', 'S4')
    False

    """
    return (from_,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.min_scalar_type)
def min_scalar_type(a):
    """
    min_scalar_type(a, /)

    For scalar ``a``, returns the data type with the smallest size
    and smallest scalar kind which can hold its value.  For non-scalar
    array ``a``, returns the vector's dtype unmodified.

    Floating point values are not demoted to integers,
    and complex values are not demoted to floats.

    Parameters
    ----------
    a : scalar or array_like
        The value whose minimal data type is to be found.

    Returns
    -------
    out : dtype
        The minimal data type.

    See Also
    --------
    result_type, promote_types, dtype, can_cast

    Examples
    --------
    >>> import numpy as np
    >>> np.min_scalar_type(10)
    dtype('uint8')

    >>> np.min_scalar_type(-260)
    dtype('int16')

    >>> np.min_scalar_type(3.1)
    dtype('float16')

    >>> np.min_scalar_type(1e50)
    dtype('float64')

    >>> np.min_scalar_type(np.arange(4,dtype='f8'))
    dtype('float64')

    """
    return (a,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.result_type)
def result_type(*arrays_and_dtypes):
    """
    result_type(*arrays_and_dtypes)

    Returns the type that results from applying the NumPy
    type promotion rules to the arguments.

    Type promotion in NumPy works similarly to the rules in languages
    like C++, with some slight differences.  When both scalars and
    arrays are used, the array's type takes precedence and the actual value
    of the scalar is taken into account.

    For example, calculating 3*a, where a is an array of 32-bit floats,
    intuitively should result in a 32-bit float output.  If the 3 is a
    32-bit integer, the NumPy rules indicate it can't convert losslessly
    into a 32-bit float, so a 64-bit float should be the result type.
    By examining the value of the constant, '3', we see that it fits in
    an 8-bit integer, which can be cast losslessly into the 32-bit float.

    Parameters
    ----------
    arrays_and_dtypes : list of arrays and dtypes
        The operands of some operation whose result type is needed.

    Returns
    -------
    out : dtype
        The result type.

    See also
    --------
    dtype, promote_types, min_scalar_type, can_cast

    Notes
    -----
    The specific algorithm used is as follows.

    Categories are determined by first checking which of boolean,
    integer (int/uint), or floating point (float/complex) the maximum
    kind of all the arrays and the scalars are.

    If there are only scalars or the maximum category of the scalars
    is higher than the maximum category of the arrays,
    the data types are combined with :func:`promote_types`
    to produce the return value.

    Otherwise, `min_scalar_type` is called on each scalar, and
    the resulting data types are all combined with :func:`promote_types`
    to produce the return value.

    The set of int values is not a subset of the uint values for types
    with the same number of bits, something not reflected in
    :func:`min_scalar_type`, but handled as a special case in `result_type`.

    Examples
    --------
    >>> import numpy as np
    >>> np.result_type(3, np.arange(7, dtype='i1'))
    dtype('int8')

    >>> np.result_type('i4', 'c8')
    dtype('complex128')

    >>> np.result_type(3.0, -2)
    dtype('float64')

    """
    return arrays_and_dtypes


@array_function_from_c_func_and_dispatcher(_multiarray_umath.dot)
def dot(a, b, out=None):
    """
    dot(a, b, out=None)

    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors
      (without complex conjugation).

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,
      but using :func:`matmul` or ``a @ b`` is preferred.

    - If either `a` or `b` is 0-D (scalar), it is equivalent to
      :func:`multiply` and using ``numpy.multiply(a, b)`` or ``a * b`` is
      preferred.

    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.

    - If `a` is an N-D array and `b` is an M-D array (where ``M>=2``), it is a
      sum product over the last axis of `a` and the second-to-last axis of
      `b`::

        dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])

    It uses an optimized BLAS library when possible (see `numpy.linalg`).

    Parameters
    ----------
    a : array_like
        First argument.
    b : array_like
        Second argument.
    out : ndarray, optional
        Output argument. This must have the exact kind that would be returned
        if it was not used. In particular, it must have the right type, must be
        C-contiguous, and its dtype must be the dtype that would be returned
        for `dot(a,b)`. This is a performance feature. Therefore, if these
        conditions are not met, an exception is raised, instead of attempting
        to be flexible.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        If `out` is given, then it is returned.

    Raises
    ------
    ValueError
        If the last dimension of `a` is not the same size as
        the second-to-last dimension of `b`.

    See Also
    --------
    vdot : Complex-conjugating dot product.
    vecdot : Vector dot product of two arrays.
    tensordot : Sum products over arbitrary axes.
    einsum : Einstein summation convention.
    matmul : '@' operator as method with out parameter.
    linalg.multi_dot : Chained dot product.

    Examples
    --------
    >>> import numpy as np
    >>> np.dot(3, 4)
    12

    Neither argument is complex-conjugated:

    >>> np.dot([2j, 3j], [2j, 3j])
    (-13+0j)

    For 2-D arrays it is the matrix product:

    >>> a = [[1, 0], [0, 1]]
    >>> b = [[4, 1], [2, 2]]
    >>> np.dot(a, b)
    array([[4, 1],
           [2, 2]])

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
    >>> np.dot(a, b)[2,3,2,1,2,2]
    499128
    >>> sum(a[2,3,2,:] * b[1,2,:,2])
    499128

    """
    return (a, b, out)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.vdot)
def vdot(a, b):
    r"""
    vdot(a, b, /)

    Return the dot product of two vectors.

    The `vdot` function handles complex numbers differently than `dot`:
    if the first argument is complex, it is replaced by its complex conjugate
    in the dot product calculation. `vdot` also handles multidimensional
    arrays differently than `dot`: it does not perform a matrix product, but
    flattens the arguments to 1-D arrays before taking a vector dot product.

    Consequently, when the arguments are 2-D arrays of the same shape, this
    function effectively returns their
    `Frobenius inner product <https://en.wikipedia.org/wiki/Frobenius_inner_product>`_
    (also known as the *trace inner product* or the *standard inner product*
    on a vector space of matrices).

    Parameters
    ----------
    a : array_like
        If `a` is complex the complex conjugate is taken before calculation
        of the dot product.
    b : array_like
        Second argument to the dot product.

    Returns
    -------
    output : ndarray
        Dot product of `a` and `b`.  Can be an int, float, or
        complex depending on the types of `a` and `b`.

    See Also
    --------
    dot : Return the dot product without using the complex conjugate of the
          first argument.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1+2j,3+4j])
    >>> b = np.array([5+6j,7+8j])
    >>> np.vdot(a, b)
    (70-8j)
    >>> np.vdot(b, a)
    (70+8j)

    Note that higher-dimensional arrays are flattened!

    >>> a = np.array([[1, 4], [5, 6]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.vdot(a, b)
    30
    >>> np.vdot(b, a)
    30
    >>> 1*4 + 4*1 + 5*2 + 6*2
    30

    """  # noqa: E501
    return (a, b)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.bincount)
def bincount(x, weights=None, minlength=None):
    """
    bincount(x, /, weights=None, minlength=0)

    Count number of occurrences of each value in array of non-negative ints.

    The number of bins (of size 1) is one larger than the largest value in
    `x`. If `minlength` is specified, there will be at least this number
    of bins in the output array (though it will be longer if necessary,
    depending on the contents of `x`).
    Each bin gives the number of occurrences of its index value in `x`.
    If `weights` is specified the input array is weighted by it, i.e. if a
    value ``n`` is found at position ``i``, ``out[n] += weight[i]`` instead
    of ``out[n] += 1``.

    Parameters
    ----------
    x : array_like, 1 dimension, nonnegative ints
        Input array.
    weights : array_like, optional
        Weights, array of the same shape as `x`.
    minlength : int, optional
        A minimum number of bins for the output array.

    Returns
    -------
    out : ndarray of ints
        The result of binning the input array.
        The length of `out` is equal to ``np.amax(x)+1``.

    Raises
    ------
    ValueError
        If the input is not 1-dimensional, or contains elements with negative
        values, or if `minlength` is negative.
    TypeError
        If the type of the input is float or complex.

    See Also
    --------
    histogram, digitize, unique

    Examples
    --------
    >>> import numpy as np
    >>> np.bincount(np.arange(5))
    array([1, 1, 1, 1, 1])
    >>> np.bincount(np.array([0, 1, 1, 3, 2, 1, 7]))
    array([1, 3, 1, 1, 0, 0, 0, 1])

    >>> x = np.array([0, 1, 1, 3, 2, 1, 7, 23])
    >>> np.bincount(x).size == np.amax(x)+1
    True

    The input array needs to be of integer dtype, otherwise a
    TypeError is raised:

    >>> np.bincount(np.arange(5, dtype=float))
    Traceback (most recent call last):
      ...
    TypeError: Cannot cast array data from dtype('float64') to dtype('int64')
    according to the rule 'safe'

    A possible use of ``bincount`` is to perform sums over
    variable-size chunks of an array, using the ``weights`` keyword.

    >>> w = np.array([0.3, 0.5, 0.2, 0.7, 1., -0.6]) # weights
    >>> x = np.array([0, 1, 1, 2, 2, 2])
    >>> np.bincount(x,  weights=w)
    array([ 0.3,  0.7,  1.1])

    """
    return (x, weights)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.ravel_multi_index)
def ravel_multi_index(multi_index, dims, mode=None, order=None):
    """
    ravel_multi_index(multi_index, dims, mode='raise', order='C')

    Converts a tuple of index arrays into an array of flat
    indices, applying boundary modes to the multi-index.

    Parameters
    ----------
    multi_index : tuple of array_like
        A tuple of integer arrays, one array for each dimension.
    dims : tuple of ints
        The shape of array into which the indices from ``multi_index`` apply.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices are handled.  Can specify
        either one mode or a tuple of modes, one mode per index.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        In 'clip' mode, a negative index which would normally
        wrap will clip to 0 instead.
    order : {'C', 'F'}, optional
        Determines whether the multi-index should be viewed as
        indexing in row-major (C-style) or column-major
        (Fortran-style) order.

    Returns
    -------
    raveled_indices : ndarray
        An array of indices into the flattened version of an array
        of dimensions ``dims``.

    See Also
    --------
    unravel_index

    Examples
    --------
    >>> import numpy as np
    >>> arr = np.array([[3,6,6],[4,5,1]])
    >>> np.ravel_multi_index(arr, (7,6))
    array([22, 41, 37])
    >>> np.ravel_multi_index(arr, (7,6), order='F')
    array([31, 41, 13])
    >>> np.ravel_multi_index(arr, (4,6), mode='clip')
    array([22, 23, 19])
    >>> np.ravel_multi_index(arr, (4,4), mode=('clip','wrap'))
    array([12, 13, 13])

    >>> np.ravel_multi_index((3,1,4,1), (6,7,8,9))
    1621
    """
    return multi_index


@array_function_from_c_func_and_dispatcher(_multiarray_umath.unravel_index)
def unravel_index(indices, shape=None, order=None):
    """
    unravel_index(indices, shape, order='C')

    Converts a flat index or array of flat indices into a tuple
    of coordinate arrays.

    Parameters
    ----------
    indices : array_like
        An integer array whose elements are indices into the flattened
        version of an array of dimensions ``shape``. Before version 1.6.0,
        this function accepted just one index value.
    shape : tuple of ints
        The shape of the array to use for unraveling ``indices``.
    order : {'C', 'F'}, optional
        Determines whether the indices should be viewed as indexing in
        row-major (C-style) or column-major (Fortran-style) order.

    Returns
    -------
    unraveled_coords : tuple of ndarray
        Each array in the tuple has the same shape as the ``indices``
        array.

    See Also
    --------
    ravel_multi_index

    Examples
    --------
    >>> import numpy as np
    >>> np.unravel_index([22, 41, 37], (7,6))
    (array([3, 6, 6]), array([4, 5, 1]))
    >>> np.unravel_index([31, 41, 13], (7,6), order='F')
    (array([3, 6, 6]), array([4, 5, 1]))

    >>> np.unravel_index(1621, (6,7,8,9))
    (3, 1, 4, 1)

    """
    return (indices,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.copyto)
def copyto(dst, src, casting=None, where=None):
    """
    copyto(dst, src, casting='same_kind', where=True)

    Copies values from one array to another, broadcasting as necessary.

    Raises a TypeError if the `casting` rule is violated, and if
    `where` is provided, it selects which elements to copy.

    Parameters
    ----------
    dst : ndarray
        The array into which values are copied.
    src : array_like
        The array from which values are copied.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}, optional
        Controls what kind of data casting may occur when copying.

        * 'no' means the data types should not be cast at all.
        * 'equiv' means only byte-order changes are allowed.
        * 'safe' means only casts which can preserve values are allowed.
        * 'same_kind' means only safe casts or casts within a kind,
          like float64 to float32, are allowed.
        * 'unsafe' means any data conversions may be done.
    where : array_like of bool, optional
        A boolean array which is broadcasted to match the dimensions
        of `dst`, and selects elements to copy from `src` to `dst`
        wherever it contains the value True.

    Examples
    --------
    >>> import numpy as np
    >>> A = np.array([4, 5, 6])
    >>> B = [1, 2, 3]
    >>> np.copyto(A, B)
    >>> A
    array([1, 2, 3])

    >>> A = np.array([[1, 2, 3], [4, 5, 6]])
    >>> B = [[4, 5, 6], [7, 8, 9]]
    >>> np.copyto(A, B)
    >>> A
    array([[4, 5, 6],
           [7, 8, 9]])

    """
    return (dst, src, where)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.putmask)
def putmask(a, /, mask, values):
    """
    putmask(a, mask, values)

    Changes elements of an array based on conditional and input values.

    Sets ``a.flat[n] = values[n]`` for each n where ``mask.flat[n]==True``.

    If `values` is not the same size as `a` and `mask` then it will repeat.
    This gives behavior different from ``a[mask] = values``.

    Parameters
    ----------
    a : ndarray
        Target array.
    mask : array_like
        Boolean mask array. It has to be the same shape as `a`.
    values : array_like
        Values to put into `a` where `mask` is True. If `values` is smaller
        than `a` it will be repeated.

    See Also
    --------
    place, put, take, copyto

    Examples
    --------
    >>> import numpy as np
    >>> x = np.arange(6).reshape(2, 3)
    >>> np.putmask(x, x>2, x**2)
    >>> x
    array([[ 0,  1,  2],
           [ 9, 16, 25]])

    If `values` is smaller than `a` it is repeated:

    >>> x = np.arange(5)
    >>> np.putmask(x, x>1, [-33, -44])
    >>> x
    array([  0,   1, -33, -44, -33])

    """
    return (a, mask, values)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.packbits)
def packbits(a, axis=None, bitorder='big'):
    """
    packbits(a, /, axis=None, bitorder='big')

    Packs the elements of a binary-valued array into bits in a uint8 array.

    The result is padded to full bytes by inserting zero bits at the end.

    Parameters
    ----------
    a : array_like
        An array of integers or booleans whose elements should be packed to
        bits.
    axis : int, optional
        The dimension over which bit-packing is done.
        ``None`` implies packing the flattened array.
    bitorder : {'big', 'little'}, optional
        The order of the input bits. 'big' will mimic bin(val),
        ``[0, 0, 0, 0, 0, 0, 1, 1] => 3 = 0b00000011``, 'little' will
        reverse the order so ``[1, 1, 0, 0, 0, 0, 0, 0] => 3``.
        Defaults to 'big'.

    Returns
    -------
    packed : ndarray
        Array of type uint8 whose elements represent bits corresponding to the
        logical (0 or nonzero) value of the input elements. The shape of
        `packed` has the same number of dimensions as the input (unless `axis`
        is None, in which case the output is 1-D).

    See Also
    --------
    unpackbits: Unpacks elements of a uint8 array into a binary-valued output
                array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[[1,0,1],
    ...                [0,1,0]],
    ...               [[1,1,0],
    ...                [0,0,1]]])
    >>> b = np.packbits(a, axis=-1)
    >>> b
    array([[[160],
            [ 64]],
           [[192],
            [ 32]]], dtype=uint8)

    Note that in binary 160 = 1010 0000, 64 = 0100 0000, 192 = 1100 0000,
    and 32 = 0010 0000.

    """
    return (a,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.unpackbits)
def unpackbits(a, axis=None, count=None, bitorder='big'):
    """
    unpackbits(a, /, axis=None, count=None, bitorder='big')

    Unpacks elements of a uint8 array into a binary-valued output array.

    Each element of `a` represents a bit-field that should be unpacked
    into a binary-valued output array. The shape of the output array is
    either 1-D (if `axis` is ``None``) or the same shape as the input
    array with unpacking done along the axis specified.

    Parameters
    ----------
    a : ndarray, uint8 type
       Input array.
    axis : int, optional
        The dimension over which bit-unpacking is done.
        ``None`` implies unpacking the flattened array.
    count : int or None, optional
        The number of elements to unpack along `axis`, provided as a way
        of undoing the effect of packing a size that is not a multiple
        of eight. A non-negative number means to only unpack `count`
        bits. A negative number means to trim off that many bits from
        the end. ``None`` means to unpack the entire array (the
        default). Counts larger than the available number of bits will
        add zero padding to the output. Negative counts must not
        exceed the available number of bits.
    bitorder : {'big', 'little'}, optional
        The order of the returned bits. 'big' will mimic bin(val),
        ``3 = 0b00000011 => [0, 0, 0, 0, 0, 0, 1, 1]``, 'little' will reverse
        the order to ``[1, 1, 0, 0, 0, 0, 0, 0]``.
        Defaults to 'big'.

    Returns
    -------
    unpacked : ndarray, uint8 type
       The elements are binary-valued (0 or 1).

    See Also
    --------
    packbits : Packs the elements of a binary-valued array into bits in
               a uint8 array.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[2], [7], [23]], dtype=np.uint8)
    >>> a
    array([[ 2],
           [ 7],
           [23]], dtype=uint8)
    >>> b = np.unpackbits(a, axis=1)
    >>> b
    array([[0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 0, 1, 1, 1]], dtype=uint8)
    >>> c = np.unpackbits(a, axis=1, count=-3)
    >>> c
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0]], dtype=uint8)

    >>> p = np.packbits(b, axis=0)
    >>> np.unpackbits(p, axis=0)
    array([[0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1],
           [0, 0, 0, 1, 0, 1, 1, 1],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> np.array_equal(b, np.unpackbits(p, axis=0, count=b.shape[0]))
    True

    """
    return (a,)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.shares_memory)
def shares_memory(a, b, max_work=None):
    """
    shares_memory(a, b, /, max_work=None)

    Determine if two arrays share memory.

    .. warning::

       This function can be exponentially slow for some inputs, unless
       `max_work` is set to zero or a positive integer.
       If in doubt, use `numpy.may_share_memory` instead.

    Parameters
    ----------
    a, b : ndarray
        Input arrays
    max_work : int, optional
        Effort to spend on solving the overlap problem (maximum number
        of candidate solutions to consider). The following special
        values are recognized:

        max_work=-1 (default)
            The problem is solved exactly. In this case, the function returns
            True only if there is an element shared between the arrays. Finding
            the exact solution may take extremely long in some cases.
        max_work=0
            Only the memory bounds of a and b are checked.
            This is equivalent to using ``may_share_memory()``.

    Raises
    ------
    numpy.exceptions.TooHardError
        Exceeded max_work.

    Returns
    -------
    out : bool

    See Also
    --------
    may_share_memory

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1, 2, 3, 4])
    >>> np.shares_memory(x, np.array([5, 6, 7]))
    False
    >>> np.shares_memory(x[::2], x)
    True
    >>> np.shares_memory(x[::2], x[1::2])
    False

    Checking whether two arrays share memory is NP-complete, and
    runtime may increase exponentially in the number of
    dimensions. Hence, `max_work` should generally be set to a finite
    number, as it is possible to construct examples that take
    extremely long to run:

    >>> from numpy.lib.stride_tricks import as_strided
    >>> x = np.zeros([192163377], dtype=np.int8)
    >>> x1 = as_strided(
    ...     x, strides=(36674, 61119, 85569), shape=(1049, 1049, 1049))
    >>> x2 = as_strided(
    ...     x[64023025:], strides=(12223, 12224, 1), shape=(1049, 1049, 1))
    >>> np.shares_memory(x1, x2, max_work=1000)
    Traceback (most recent call last):
    ...
    numpy.exceptions.TooHardError: Exceeded max_work

    Running ``np.shares_memory(x1, x2)`` without `max_work` set takes
    around 1 minute for this case. It is possible to find problems
    that take still significantly longer.

    """
    return (a, b)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.may_share_memory)
def may_share_memory(a, b, max_work=None):
    """
    may_share_memory(a, b, /, max_work=None)

    Determine if two arrays might share memory

    A return of True does not necessarily mean that the two arrays
    share any element.  It just means that they *might*.

    Only the memory bounds of a and b are checked by default.

    Parameters
    ----------
    a, b : ndarray
        Input arrays
    max_work : int, optional
        Effort to spend on solving the overlap problem.  See
        `shares_memory` for details.  Default for ``may_share_memory``
        is to do a bounds check.

    Returns
    -------
    out : bool

    See Also
    --------
    shares_memory

    Examples
    --------
    >>> import numpy as np
    >>> np.may_share_memory(np.array([1,2]), np.array([5,8,9]))
    False
    >>> x = np.zeros([3, 4])
    >>> np.may_share_memory(x[:,0], x[:,1])
    True

    """
    return (a, b)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.is_busday)
def is_busday(dates, weekmask=None, holidays=None, busdaycal=None, out=None):
    """
    is_busday(
        dates,
        weekmask='1111100',
        holidays=None,
        busdaycal=None,
        out=None
    )

    Calculates which of the given dates are valid days, and which are not.

    Parameters
    ----------
    dates : array_like of datetime64[D]
        The array of dates to process.
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates.  They may be
        specified in any order, and NaT (not-a-time) dates are ignored.
        This list is saved in a normalized form that is suited for
        fast calculations of valid days.
    busdaycal : busdaycalendar, optional
        A `busdaycalendar` object which specifies the valid days. If this
        parameter is provided, neither weekmask nor holidays may be
        provided.
    out : array of bool, optional
        If provided, this array is filled with the result.

    Returns
    -------
    out : array of bool
        An array with the same shape as ``dates``, containing True for
        each valid day, and False for each invalid day.

    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
    busday_offset : Applies an offset counted in valid days.
    busday_count : Counts how many valid days are in a half-open date range.

    Examples
    --------
    >>> import numpy as np
    >>> # The weekdays are Friday, Saturday, and Monday
    ... np.is_busday(['2011-07-01', '2011-07-02', '2011-07-18'],
    ...                 holidays=['2011-07-01', '2011-07-04', '2011-07-17'])
    array([False, False,  True])
    """
    return (dates, weekmask, holidays, out)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.busday_offset)
def busday_offset(dates, offsets, roll=None, weekmask=None, holidays=None,
                  busdaycal=None, out=None):
    """
    busday_offset(
        dates,
        offsets,
        roll='raise',
        weekmask='1111100',
        holidays=None,
        busdaycal=None,
        out=None
    )

    First adjusts the date to fall on a valid day according to
    the ``roll`` rule, then applies offsets to the given dates
    counted in valid days.

    Parameters
    ----------
    dates : array_like of datetime64[D]
        The array of dates to process.
    offsets : array_like of int
        The array of offsets, which is broadcast with ``dates``.
    roll : {'raise', 'nat', 'forward', 'following', 'backward', 'preceding', \
        'modifiedfollowing', 'modifiedpreceding'}, optional
        How to treat dates that do not fall on a valid day. The default
        is 'raise'.

        * 'raise' means to raise an exception for an invalid day.
        * 'nat' means to return a NaT (not-a-time) for an invalid day.
        * 'forward' and 'following' mean to take the first valid day
          later in time.
        * 'backward' and 'preceding' mean to take the first valid day
          earlier in time.
        * 'modifiedfollowing' means to take the first valid day
          later in time unless it is across a Month boundary, in which
          case to take the first valid day earlier in time.
        * 'modifiedpreceding' means to take the first valid day
          earlier in time unless it is across a Month boundary, in which
          case to take the first valid day later in time.
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates.  They may be
        specified in any order, and NaT (not-a-time) dates are ignored.
        This list is saved in a normalized form that is suited for
        fast calculations of valid days.
    busdaycal : busdaycalendar, optional
        A `busdaycalendar` object which specifies the valid days. If this
        parameter is provided, neither weekmask nor holidays may be
        provided.
    out : array of datetime64[D], optional
        If provided, this array is filled with the result.

    Returns
    -------
    out : array of datetime64[D]
        An array with a shape from broadcasting ``dates`` and ``offsets``
        together, containing the dates with offsets applied.

    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
    is_busday : Returns a boolean array indicating valid days.
    busday_count : Counts how many valid days are in a half-open date range.

    Examples
    --------
    >>> import numpy as np
    >>> # First business day in October 2011 (not accounting for holidays)
    ... np.busday_offset('2011-10', 0, roll='forward')
    np.datetime64('2011-10-03')
    >>> # Last business day in February 2012 (not accounting for holidays)
    ... np.busday_offset('2012-03', -1, roll='forward')
    np.datetime64('2012-02-29')
    >>> # Third Wednesday in January 2011
    ... np.busday_offset('2011-01', 2, roll='forward', weekmask='Wed')
    np.datetime64('2011-01-19')
    >>> # 2012 Mother's Day in Canada and the U.S.
    ... np.busday_offset('2012-05', 1, roll='forward', weekmask='Sun')
    np.datetime64('2012-05-13')

    >>> # First business day on or after a date
    ... np.busday_offset('2011-03-20', 0, roll='forward')
    np.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 0, roll='forward')
    np.datetime64('2011-03-22')
    >>> # First business day after a date
    ... np.busday_offset('2011-03-20', 1, roll='backward')
    np.datetime64('2011-03-21')
    >>> np.busday_offset('2011-03-22', 1, roll='backward')
    np.datetime64('2011-03-23')
    """
    return (dates, offsets, weekmask, holidays, out)


@array_function_from_c_func_and_dispatcher(_multiarray_umath.busday_count)
def busday_count(begindates, enddates, weekmask=None, holidays=None,
                 busdaycal=None, out=None):
    """
    busday_count(
        begindates,
        enddates,
        weekmask='1111100',
        holidays=[],
        busdaycal=None,
        out=None
    )

    Counts the number of valid days between `begindates` and
    `enddates`, not including the day of `enddates`.

    If ``enddates`` specifies a date value that is earlier than the
    corresponding ``begindates`` date value, the count will be negative.

    Parameters
    ----------
    begindates : array_like of datetime64[D]
        The array of the first dates for counting.
    enddates : array_like of datetime64[D]
        The array of the end dates for counting, which are excluded
        from the count themselves.
    weekmask : str or array_like of bool, optional
        A seven-element array indicating which of Monday through Sunday are
        valid days. May be specified as a length-seven list or array, like
        [1,1,1,1,1,0,0]; a length-seven string, like '1111100'; or a string
        like "Mon Tue Wed Thu Fri", made up of 3-character abbreviations for
        weekdays, optionally separated by white space. Valid abbreviations
        are: Mon Tue Wed Thu Fri Sat Sun
    holidays : array_like of datetime64[D], optional
        An array of dates to consider as invalid dates.  They may be
        specified in any order, and NaT (not-a-time) dates are ignored.
        This list is saved in a normalized form that is suited for
        fast calculations of valid days.
    busdaycal : busdaycalendar, optional
        A `busdaycalendar` object which specifies the valid days. If this
        parameter is provided, neither weekmask nor holidays may be
        provided.
    out : array of int, optional
        If provided, this array is filled with the result.

    Returns
    -------
    out : array of int
        An array with a shape from broadcasting ``begindates`` and ``enddates``
        together, containing the number of valid days between
        the begin and end dates.

    See Also
    --------
    busdaycalendar : An object that specifies a custom set of valid days.
    is_busday : Returns a boolean array indicating valid days.
    busday_offset : Applies an offset counted in valid days.

    Examples
    --------
    >>> import numpy as np
    >>> # Number of weekdays in January 2011
    ... np.busday_count('2011-01', '2011-02')
    21
    >>> # Number of weekdays in 2011
    >>> np.busday_count('2011', '2012')
    260
    >>> # Number of Saturdays in 2011
    ... np.busday_count('2011', '2012', weekmask='Sat')
    53
    """
    return (begindates, enddates, weekmask, holidays, out)


@array_function_from_c_func_and_dispatcher(
    _multiarray_umath.datetime_as_string)
def datetime_as_string(arr, unit=None, timezone=None, casting=None):
    """
    datetime_as_string(arr, unit=None, timezone='naive', casting='same_kind')

    Convert an array of datetimes into an array of strings.

    Parameters
    ----------
    arr : array_like of datetime64
        The array of UTC timestamps to format.
    unit : str
        One of None, 'auto', or
        a :ref:`datetime unit <arrays.dtypes.dateunits>`.
    timezone : {'naive', 'UTC', 'local'} or tzinfo
        Timezone information to use when displaying the datetime. If 'UTC',
        end with a Z to indicate UTC time. If 'local', convert to the local
        timezone first, and suffix with a +-#### timezone offset. If a tzinfo
        object, then do as with 'local', but use the specified timezone.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}
        Casting to allow when changing between datetime units.

    Returns
    -------
    str_arr : ndarray
        An array of strings the same shape as `arr`.

    Examples
    --------
    >>> import numpy as np
    >>> import pytz
    >>> d = np.arange('2002-10-27T04:30', 4*60, 60, dtype='M8[m]')
    >>> d
    array(['2002-10-27T04:30', '2002-10-27T05:30', '2002-10-27T06:30',
           '2002-10-27T07:30'], dtype='datetime64[m]')

    Setting the timezone to UTC shows the same information, but with a Z suffix

    >>> np.datetime_as_string(d, timezone='UTC')
    array(['2002-10-27T04:30Z', '2002-10-27T05:30Z', '2002-10-27T06:30Z',
           '2002-10-27T07:30Z'], dtype='<U35')

    Note that we picked datetimes that cross a DST boundary. Passing in a
    ``pytz`` timezone object will print the appropriate offset

    >>> np.datetime_as_string(d, timezone=pytz.timezone('US/Eastern'))
    array(['2002-10-27T00:30-0400', '2002-10-27T01:30-0400',
           '2002-10-27T01:30-0500', '2002-10-27T02:30-0500'], dtype='<U39')

    Passing in a unit will change the precision

    >>> np.datetime_as_string(d, unit='h')
    array(['2002-10-27T04', '2002-10-27T05', '2002-10-27T06', '2002-10-27T07'],
          dtype='<U32')
    >>> np.datetime_as_string(d, unit='s')
    array(['2002-10-27T04:30:00', '2002-10-27T05:30:00', '2002-10-27T06:30:00',
           '2002-10-27T07:30:00'], dtype='<U38')

    'casting' can be used to specify whether precision can be changed

    >>> np.datetime_as_string(d, unit='h', casting='safe')
    Traceback (most recent call last):
        ...
    TypeError: Cannot create a datetime string as units 'h' from a NumPy
    datetime with units 'm' according to the rule 'safe'
    """
    return (arr,)