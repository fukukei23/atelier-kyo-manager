
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\_utils.py ===
from typing import Union

"""
_url.py
websocket - WebSocket client library for Python

Copyright 2024 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
__all__ = ["NoLock", "validate_utf8", "extract_err_message", "extract_error_code"]


class NoLock:
    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        pass


try:
    # If wsaccel is available we use compiled routines to validate UTF-8
    # strings.
    from wsaccel.utf8validator import Utf8Validator

    def _validate_utf8(utfbytes: Union[str, bytes]) -> bool:
        result: bool = Utf8Validator().validate(utfbytes)[0]
        return result

except ImportError:
    # UTF-8 validator
    # python implementation of http://bjoern.hoehrmann.de/utf-8/decoder/dfa/

    _UTF8_ACCEPT = 0
    _UTF8_REJECT = 12

    _UTF8D = [
        # The first part of the table maps bytes to character classes that
        # to reduce the size of the transition table and create bitmasks.
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        9,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        7,
        8,
        8,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        10,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        3,
        4,
        3,
        3,
        11,
        6,
        6,
        6,
        5,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        8,
        # The second part is a transition table that maps a combination
        # of a state of the automaton and a character class to a state.
        0,
        12,
        24,
        36,
        60,
        96,
        84,
        12,
        12,
        12,
        48,
        72,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        0,
        12,
        12,
        12,
        12,
        12,
        0,
        12,
        0,
        12,
        12,
        12,
        24,
        12,
        12,
        12,
        12,
        12,
        24,
        12,
        24,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        24,
        12,
        12,
        12,
        12,
        12,
        24,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        24,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        36,
        12,
        36,
        12,
        12,
        12,
        36,
        12,
        12,
        12,
        12,
        12,
        36,
        12,
        36,
        12,
        12,
        12,
        36,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
        12,
    ]

    def _decode(state: int, codep: int, ch: int) -> tuple:
        tp = _UTF8D[ch]

        codep = (
            (ch & 0x3F) | (codep << 6) if (state != _UTF8_ACCEPT) else (0xFF >> tp) & ch
        )
        state = _UTF8D[256 + state + tp]

        return state, codep

    def _validate_utf8(utfbytes: Union[str, bytes]) -> bool:
        state = _UTF8_ACCEPT
        codep = 0
        for i in utfbytes:
            state, codep = _decode(state, codep, int(i))
            if state == _UTF8_REJECT:
                return False

        return True


def validate_utf8(utfbytes: Union[str, bytes]) -> bool:
    """
    validate utf8 byte string.
    utfbytes: utf byte string to check.
    return value: if valid utf8 string, return true. Otherwise, return false.
    """
    return _validate_utf8(utfbytes)


def extract_err_message(exception: Exception) -> Union[str, None]:
    if exception.args:
        exception_message: str = exception.args[0]
        return exception_message
    else:
        return None


def extract_error_code(exception: Exception) -> Union[int, None]:
    if exception.args and len(exception.args) > 1:
        return exception.args[0] if isinstance(exception.args[0], int) else None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\events.py ===
# sql/events.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from .base import SchemaEventTarget
from .. import event

if TYPE_CHECKING:
    from .schema import Column
    from .schema import Constraint
    from .schema import SchemaItem
    from .schema import Table
    from ..engine.base import Connection
    from ..engine.interfaces import ReflectedColumn
    from ..engine.reflection import Inspector


class DDLEvents(event.Events[SchemaEventTarget]):
    """
    Define event listeners for schema objects,
    that is, :class:`.SchemaItem` and other :class:`.SchemaEventTarget`
    subclasses, including :class:`_schema.MetaData`, :class:`_schema.Table`,
    :class:`_schema.Column`, etc.

    **Create / Drop Events**

    Events emitted when CREATE and DROP commands are emitted to the database.
    The event hooks in this category include :meth:`.DDLEvents.before_create`,
    :meth:`.DDLEvents.after_create`, :meth:`.DDLEvents.before_drop`, and
    :meth:`.DDLEvents.after_drop`.

    These events are emitted when using schema-level methods such as
    :meth:`.MetaData.create_all` and :meth:`.MetaData.drop_all`. Per-object
    create/drop methods such as :meth:`.Table.create`, :meth:`.Table.drop`,
    :meth:`.Index.create` are also included, as well as dialect-specific
    methods such as :meth:`_postgresql.ENUM.create`.

    .. versionadded:: 2.0 :class:`.DDLEvents` event hooks now take place
       for non-table objects including constraints, indexes, and
       dialect-specific schema types.

    Event hooks may be attached directly to a :class:`_schema.Table` object or
    to a :class:`_schema.MetaData` collection, as well as to any
    :class:`.SchemaItem` class or object that can be individually created and
    dropped using a distinct SQL command. Such classes include :class:`.Index`,
    :class:`.Sequence`, and dialect-specific classes such as
    :class:`_postgresql.ENUM`.

    Example using the :meth:`.DDLEvents.after_create` event, where a custom
    event hook will emit an ``ALTER TABLE`` command on the current connection,
    after ``CREATE TABLE`` is emitted::

        from sqlalchemy import create_engine
        from sqlalchemy import event
        from sqlalchemy import Table, Column, Metadata, Integer

        m = MetaData()
        some_table = Table("some_table", m, Column("data", Integer))


        @event.listens_for(some_table, "after_create")
        def after_create(target, connection, **kw):
            connection.execute(
                text("ALTER TABLE %s SET name=foo_%s" % (target.name, target.name))
            )


        some_engine = create_engine("postgresql://scott:tiger@host/test")

        # will emit "CREATE TABLE some_table" as well as the above
        # "ALTER TABLE" statement afterwards
        m.create_all(some_engine)

    Constraint objects such as :class:`.ForeignKeyConstraint`,
    :class:`.UniqueConstraint`, :class:`.CheckConstraint` may also be
    subscribed to these events, however they will **not** normally produce
    events as these objects are usually rendered inline within an
    enclosing ``CREATE TABLE`` statement and implicitly dropped from a
    ``DROP TABLE`` statement.

    For the :class:`.Index` construct, the event hook will be emitted
    for ``CREATE INDEX``, however SQLAlchemy does not normally emit
    ``DROP INDEX`` when dropping tables as this is again implicit within the
    ``DROP TABLE`` statement.

    .. versionadded:: 2.0 Support for :class:`.SchemaItem` objects
       for create/drop events was expanded from its previous support for
       :class:`.MetaData` and :class:`.Table` to also include
       :class:`.Constraint` and all subclasses, :class:`.Index`,
       :class:`.Sequence` and some type-related constructs such as
       :class:`_postgresql.ENUM`.

    .. note:: These event hooks are only emitted within the scope of
       SQLAlchemy's create/drop methods; they are not necessarily supported
       by tools such as `alembic <https://alembic.sqlalchemy.org>`_.


    **Attachment Events**

    Attachment events are provided to customize
    behavior whenever a child schema element is associated
    with a parent, such as when a :class:`_schema.Column` is associated
    with its :class:`_schema.Table`, when a
    :class:`_schema.ForeignKeyConstraint`
    is associated with a :class:`_schema.Table`, etc.  These events include
    :meth:`.DDLEvents.before_parent_attach` and
    :meth:`.DDLEvents.after_parent_attach`.

    **Reflection Events**

    The :meth:`.DDLEvents.column_reflect` event is used to intercept
    and modify the in-Python definition of database columns when
    :term:`reflection` of database tables proceeds.

    **Use with Generic DDL**

    DDL events integrate closely with the
    :class:`.DDL` class and the :class:`.ExecutableDDLElement` hierarchy
    of DDL clause constructs, which are themselves appropriate
    as listener callables::

        from sqlalchemy import DDL

        event.listen(
            some_table,
            "after_create",
            DDL("ALTER TABLE %(table)s SET name=foo_%(table)s"),
        )

    **Event Propagation to MetaData Copies**

    For all :class:`.DDLEvent` events, the ``propagate=True`` keyword argument
    will ensure that a given event handler is propagated to copies of the
    object, which are made when using the :meth:`_schema.Table.to_metadata`
    method::

        from sqlalchemy import DDL

        metadata = MetaData()
        some_table = Table("some_table", metadata, Column("data", Integer))

        event.listen(
            some_table,
            "after_create",
            DDL("ALTER TABLE %(table)s SET name=foo_%(table)s"),
            propagate=True,
        )

        new_metadata = MetaData()
        new_table = some_table.to_metadata(new_metadata)

    The above :class:`.DDL` object will be associated with the
    :meth:`.DDLEvents.after_create` event for both the ``some_table`` and
    the ``new_table`` :class:`.Table` objects.

    .. seealso::

        :ref:`event_toplevel`

        :class:`.ExecutableDDLElement`

        :class:`.DDL`

        :ref:`schema_ddl_sequences`

    """  # noqa: E501

    _target_class_doc = "SomeSchemaClassOrObject"
    _dispatch_target = SchemaEventTarget

    def before_create(
        self, target: SchemaEventTarget, connection: Connection, **kw: Any
    ) -> None:
        r"""Called before CREATE statements are emitted.

        :param target: the :class:`.SchemaObject`, such as a
         :class:`_schema.MetaData` or :class:`_schema.Table`
         but also including all create/drop objects such as
         :class:`.Index`, :class:`.Sequence`, etc.,
         object which is the target of the event.

         .. versionadded:: 2.0 Support for all :class:`.SchemaItem` objects
            was added.

        :param connection: the :class:`_engine.Connection` where the
         CREATE statement or statements will be emitted.
        :param \**kw: additional keyword arguments relevant
         to the event.  The contents of this dictionary
         may vary across releases, and include the
         list of tables being generated for a metadata-level
         event, the checkfirst flag, and other
         elements used by internal events.

        :func:`.event.listen` accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        :func:`.event.listen` accepts the ``insert=True``
        modifier for this event; when True, the listener function will
        be prepended to the internal list of events upon discovery, and execute
        before registered listener functions that do not pass this argument.

        """

    def after_create(
        self, target: SchemaEventTarget, connection: Connection, **kw: Any
    ) -> None:
        r"""Called after CREATE statements are emitted.

        :param target: the :class:`.SchemaObject`, such as a
         :class:`_schema.MetaData` or :class:`_schema.Table`
         but also including all create/drop objects such as
         :class:`.Index`, :class:`.Sequence`, etc.,
         object which is the target of the event.

         .. versionadded:: 2.0 Support for all :class:`.SchemaItem` objects
            was added.

        :param connection: the :class:`_engine.Connection` where the
         CREATE statement or statements have been emitted.
        :param \**kw: additional keyword arguments relevant
         to the event.  The contents of this dictionary
         may vary across releases, and include the
         list of tables being generated for a metadata-level
         event, the checkfirst flag, and other
         elements used by internal events.

        :func:`.event.listen` also accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        """

    def before_drop(
        self, target: SchemaEventTarget, connection: Connection, **kw: Any
    ) -> None:
        r"""Called before DROP statements are emitted.

        :param target: the :class:`.SchemaObject`, such as a
         :class:`_schema.MetaData` or :class:`_schema.Table`
         but also including all create/drop objects such as
         :class:`.Index`, :class:`.Sequence`, etc.,
         object which is the target of the event.

         .. versionadded:: 2.0 Support for all :class:`.SchemaItem` objects
            was added.

        :param connection: the :class:`_engine.Connection` where the
         DROP statement or statements will be emitted.
        :param \**kw: additional keyword arguments relevant
         to the event.  The contents of this dictionary
         may vary across releases, and include the
         list of tables being generated for a metadata-level
         event, the checkfirst flag, and other
         elements used by internal events.

        :func:`.event.listen` also accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        """

    def after_drop(
        self, target: SchemaEventTarget, connection: Connection, **kw: Any
    ) -> None:
        r"""Called after DROP statements are emitted.

        :param target: the :class:`.SchemaObject`, such as a
         :class:`_schema.MetaData` or :class:`_schema.Table`
         but also including all create/drop objects such as
         :class:`.Index`, :class:`.Sequence`, etc.,
         object which is the target of the event.

         .. versionadded:: 2.0 Support for all :class:`.SchemaItem` objects
            was added.

        :param connection: the :class:`_engine.Connection` where the
         DROP statement or statements have been emitted.
        :param \**kw: additional keyword arguments relevant
         to the event.  The contents of this dictionary
         may vary across releases, and include the
         list of tables being generated for a metadata-level
         event, the checkfirst flag, and other
         elements used by internal events.

        :func:`.event.listen` also accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        """

    def before_parent_attach(
        self, target: SchemaEventTarget, parent: SchemaItem
    ) -> None:
        """Called before a :class:`.SchemaItem` is associated with
        a parent :class:`.SchemaItem`.

        :param target: the target object
        :param parent: the parent to which the target is being attached.

        :func:`.event.listen` also accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        """

    def after_parent_attach(
        self, target: SchemaEventTarget, parent: SchemaItem
    ) -> None:
        """Called after a :class:`.SchemaItem` is associated with
        a parent :class:`.SchemaItem`.

        :param target: the target object
        :param parent: the parent to which the target is being attached.

        :func:`.event.listen` also accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        """

    def _sa_event_column_added_to_pk_constraint(
        self, const: Constraint, col: Column[Any]
    ) -> None:
        """internal event hook used for primary key naming convention
        updates.

        """

    def column_reflect(
        self, inspector: Inspector, table: Table, column_info: ReflectedColumn
    ) -> None:
        """Called for each unit of 'column info' retrieved when
        a :class:`_schema.Table` is being reflected.

        This event is most easily used by applying it to a specific
        :class:`_schema.MetaData` instance, where it will take effect for
        all :class:`_schema.Table` objects within that
        :class:`_schema.MetaData` that undergo reflection::

            metadata = MetaData()


            @event.listens_for(metadata, "column_reflect")
            def receive_column_reflect(inspector, table, column_info):
                # receives for all Table objects that are reflected
                # under this MetaData
                ...


            # will use the above event hook
            my_table = Table("my_table", metadata, autoload_with=some_engine)

        .. versionadded:: 1.4.0b2 The :meth:`_events.DDLEvents.column_reflect`
           hook may now be applied to a :class:`_schema.MetaData` object as
           well as the :class:`_schema.MetaData` class itself where it will
           take place for all :class:`_schema.Table` objects associated with
           the targeted :class:`_schema.MetaData`.

        It may also be applied to the :class:`_schema.Table` class across
        the board::

            from sqlalchemy import Table


            @event.listens_for(Table, "column_reflect")
            def receive_column_reflect(inspector, table, column_info):
                # receives for all Table objects that are reflected
                ...

        It can also be applied to a specific :class:`_schema.Table` at the
        point that one is being reflected using the
        :paramref:`_schema.Table.listeners` parameter::

            t1 = Table(
                "my_table",
                autoload_with=some_engine,
                listeners=[("column_reflect", receive_column_reflect)],
            )

        The dictionary of column information as returned by the
        dialect is passed, and can be modified.  The dictionary
        is that returned in each element of the list returned
        by :meth:`.reflection.Inspector.get_columns`:

            * ``name`` - the column's name, is applied to the
              :paramref:`_schema.Column.name` parameter

            * ``type`` - the type of this column, which should be an instance
              of :class:`~sqlalchemy.types.TypeEngine`, is applied to the
              :paramref:`_schema.Column.type` parameter

            * ``nullable`` - boolean flag if the column is NULL or NOT NULL,
              is applied to the :paramref:`_schema.Column.nullable` parameter

            * ``default`` - the column's server default value.  This is
              normally specified as a plain string SQL expression, however the
              event can pass a :class:`.FetchedValue`, :class:`.DefaultClause`,
              or :func:`_expression.text` object as well.  Is applied to the
              :paramref:`_schema.Column.server_default` parameter

        The event is called before any action is taken against
        this dictionary, and the contents can be modified; the following
        additional keys may be added to the dictionary to further modify
        how the :class:`_schema.Column` is constructed:


            * ``key`` - the string key that will be used to access this
              :class:`_schema.Column` in the ``.c`` collection; will be applied
              to the :paramref:`_schema.Column.key` parameter. Is also used
              for ORM mapping.  See the section
              :ref:`mapper_automated_reflection_schemes` for an example.

            * ``quote`` - force or un-force quoting on the column name;
              is applied to the :paramref:`_schema.Column.quote` parameter.

            * ``info`` - a dictionary of arbitrary data to follow along with
              the :class:`_schema.Column`, is applied to the
              :paramref:`_schema.Column.info` parameter.

        :func:`.event.listen` also accepts the ``propagate=True``
        modifier for this event; when True, the listener function will
        be established for any copies made of the target object,
        i.e. those copies that are generated when
        :meth:`_schema.Table.to_metadata` is used.

        .. seealso::

            :ref:`mapper_automated_reflection_schemes` -
            in the ORM mapping documentation

            :ref:`automap_intercepting_columns` -
            in the :ref:`automap_toplevel` documentation

            :ref:`metadata_reflection_dbagnostic_types` - in
            the :ref:`metadata_reflection_toplevel` documentation

        """

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\random_matrix_models.py ===
from sympy.concrete.products import Product
from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.integrals import Integral
from sympy.matrices.expressions.matexpr import MatrixSymbol
from sympy.matrices.expressions.trace import Trace
from sympy.tensor.indexed import IndexedBase
from sympy.core.sympify import _sympify
from sympy.stats.rv import _symbol_converter, Density, RandomMatrixSymbol, is_random
from sympy.stats.joint_rv_types import JointDistributionHandmade
from sympy.stats.random_matrix import RandomMatrixPSpace
from sympy.tensor.array import ArrayComprehension

__all__ = [
    'CircularEnsemble',
    'CircularUnitaryEnsemble',
    'CircularOrthogonalEnsemble',
    'CircularSymplecticEnsemble',
    'GaussianEnsemble',
    'GaussianUnitaryEnsemble',
    'GaussianOrthogonalEnsemble',
    'GaussianSymplecticEnsemble',
    'joint_eigen_distribution',
    'JointEigenDistribution',
    'level_spacing_distribution'
]

@is_random.register(RandomMatrixSymbol)
def _(x):
    return True


class RandomMatrixEnsembleModel(Basic):
    """
    Base class for random matrix ensembles.
    It acts as an umbrella and contains
    the methods common to all the ensembles
    defined in sympy.stats.random_matrix_models.
    """
    def __new__(cls, sym, dim=None):
        sym, dim = _symbol_converter(sym), _sympify(dim)
        if dim.is_integer == False:
            raise ValueError("Dimension of the random matrices must be "
                                "integers, received %s instead."%(dim))
        return Basic.__new__(cls, sym, dim)

    symbol = property(lambda self: self.args[0])
    dimension = property(lambda self: self.args[1])

    def density(self, expr):
        return Density(expr)

    def __call__(self, expr):
        return self.density(expr)

class GaussianEnsembleModel(RandomMatrixEnsembleModel):
    """
    Abstract class for Gaussian ensembles.
    Contains the properties common to all the
    gaussian ensembles.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Random_matrix#Gaussian_ensembles
    .. [2] https://arxiv.org/pdf/1712.07903.pdf
    """
    def _compute_normalization_constant(self, beta, n):
        """
        Helper function for computing normalization
        constant for joint probability density of eigen
        values of Gaussian ensembles.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Selberg_integral#Mehta's_integral
        """
        n = S(n)
        prod_term = lambda j: gamma(1 + beta*S(j)/2)/gamma(S.One + beta/S(2))
        j = Dummy('j', integer=True, positive=True)
        term1 = Product(prod_term(j), (j, 1, n)).doit()
        term2 = (2/(beta*n))**(beta*n*(n - 1)/4 + n/2)
        term3 = (2*pi)**(n/2)
        return term1 * term2 * term3

    def _compute_joint_eigen_distribution(self, beta):
        """
        Helper function for computing the joint
        probability distribution of eigen values
        of the random matrix.
        """
        n = self.dimension
        Zbn = self._compute_normalization_constant(beta, n)
        l = IndexedBase('l')
        i = Dummy('i', integer=True, positive=True)
        j = Dummy('j', integer=True, positive=True)
        k = Dummy('k', integer=True, positive=True)
        term1 = exp((-S(n)/2) * Sum(l[k]**2, (k, 1, n)).doit())
        sub_term = Lambda(i, Product(Abs(l[j] - l[i])**beta, (j, i + 1, n)))
        term2 = Product(sub_term(i).doit(), (i, 1, n - 1)).doit()
        syms = ArrayComprehension(l[k], (k, 1, n)).doit()
        return Lambda(tuple(syms), (term1 * term2)/Zbn)

class GaussianUnitaryEnsembleModel(GaussianEnsembleModel):
    @property
    def normalization_constant(self):
        n = self.dimension
        return 2**(S(n)/2) * pi**(S(n**2)/2)

    def density(self, expr):
        n, ZGUE = self.dimension, self.normalization_constant
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n)/2 * Trace(H**2))/ZGUE)(expr)

    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S(2))

    def level_spacing_distribution(self):
        s = Dummy('s')
        f = (32/pi**2)*(s**2)*exp((-4/pi)*s**2)
        return Lambda(s, f)

class GaussianOrthogonalEnsembleModel(GaussianEnsembleModel):
    @property
    def normalization_constant(self):
        n = self.dimension
        _H = MatrixSymbol('_H', n, n)
        return Integral(exp(-S(n)/4 * Trace(_H**2)))

    def density(self, expr):
        n, ZGOE = self.dimension, self.normalization_constant
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n)/4 * Trace(H**2))/ZGOE)(expr)

    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S.One)

    def level_spacing_distribution(self):
        s = Dummy('s')
        f = (pi/2)*s*exp((-pi/4)*s**2)
        return Lambda(s, f)

class GaussianSymplecticEnsembleModel(GaussianEnsembleModel):
    @property
    def normalization_constant(self):
        n = self.dimension
        _H = MatrixSymbol('_H', n, n)
        return Integral(exp(-S(n) * Trace(_H**2)))

    def density(self, expr):
        n, ZGSE = self.dimension, self.normalization_constant
        h_pspace = RandomMatrixPSpace('P', model=self)
        H = RandomMatrixSymbol('H', n, n, pspace=h_pspace)
        return Lambda(H, exp(-S(n) * Trace(H**2))/ZGSE)(expr)

    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S(4))

    def level_spacing_distribution(self):
        s = Dummy('s')
        f = ((S(2)**18)/((S(3)**6)*(pi**3)))*(s**4)*exp((-64/(9*pi))*s**2)
        return Lambda(s, f)

def GaussianEnsemble(sym, dim):
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianUnitaryEnsemble(sym, dim):
    """
    Represents Gaussian Unitary Ensembles.

    Examples
    ========

    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE, density
    >>> from sympy import MatrixSymbol
    >>> G = GUE('U', 2)
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(G)(X)
    exp(-Trace(X**2))/(2*pi**2)
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianUnitaryEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianOrthogonalEnsemble(sym, dim):
    """
    Represents Gaussian Orthogonal Ensembles.

    Examples
    ========

    >>> from sympy.stats import GaussianOrthogonalEnsemble as GOE, density
    >>> from sympy import MatrixSymbol
    >>> G = GOE('U', 2)
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(G)(X)
    exp(-Trace(X**2)/2)/Integral(exp(-Trace(_H**2)/2), _H)
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianOrthogonalEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def GaussianSymplecticEnsemble(sym, dim):
    """
    Represents Gaussian Symplectic Ensembles.

    Examples
    ========

    >>> from sympy.stats import GaussianSymplecticEnsemble as GSE, density
    >>> from sympy import MatrixSymbol
    >>> G = GSE('U', 2)
    >>> X = MatrixSymbol('X', 2, 2)
    >>> density(G)(X)
    exp(-2*Trace(X**2))/Integral(exp(-2*Trace(_H**2)), _H)
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = GaussianSymplecticEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

class CircularEnsembleModel(RandomMatrixEnsembleModel):
    """
    Abstract class for Circular ensembles.
    Contains the properties and methods
    common to all the circular ensembles.

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Circular_ensemble
    """
    def density(self, expr):
        # TODO : Add support for Lie groups(as extensions of sympy.diffgeom)
        #        and define measures on them
        raise NotImplementedError("Support for Haar measure hasn't been "
                                  "implemented yet, therefore the density of "
                                  "%s cannot be computed."%(self))

    def _compute_joint_eigen_distribution(self, beta):
        """
        Helper function to compute the joint distribution of phases
        of the complex eigen values of matrices belonging to any
        circular ensembles.
        """
        n = self.dimension
        Zbn = ((2*pi)**n)*(gamma(beta*n/2 + 1)/S(gamma(beta/2 + 1))**n)
        t = IndexedBase('t')
        i, j, k = (Dummy('i', integer=True), Dummy('j', integer=True),
                   Dummy('k', integer=True))
        syms = ArrayComprehension(t[i], (i, 1, n)).doit()
        f = Product(Product(Abs(exp(I*t[k]) - exp(I*t[j]))**beta, (j, k + 1, n)).doit(),
                    (k, 1, n - 1)).doit()
        return Lambda(tuple(syms), f/Zbn)

class CircularUnitaryEnsembleModel(CircularEnsembleModel):
    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S(2))

class CircularOrthogonalEnsembleModel(CircularEnsembleModel):
    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S.One)

class CircularSymplecticEnsembleModel(CircularEnsembleModel):
    def joint_eigen_distribution(self):
        return self._compute_joint_eigen_distribution(S(4))

def CircularEnsemble(sym, dim):
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = CircularEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def CircularUnitaryEnsemble(sym, dim):
    """
    Represents Circular Unitary Ensembles.

    Examples
    ========

    >>> from sympy.stats import CircularUnitaryEnsemble as CUE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = CUE('U', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k]))**2, (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    As can be seen above in the example, density of CiruclarUnitaryEnsemble
    is not evaluated because the exact definition is based on haar measure of
    unitary group which is not unique.
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = CircularUnitaryEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def CircularOrthogonalEnsemble(sym, dim):
    """
    Represents Circular Orthogonal Ensembles.

    Examples
    ========

    >>> from sympy.stats import CircularOrthogonalEnsemble as COE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = COE('O', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k])), (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    As can be seen above in the example, density of CiruclarOrthogonalEnsemble
    is not evaluated because the exact definition is based on haar measure of
    unitary group which is not unique.
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = CircularOrthogonalEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def CircularSymplecticEnsemble(sym, dim):
    """
    Represents Circular Symplectic Ensembles.

    Examples
    ========

    >>> from sympy.stats import CircularSymplecticEnsemble as CSE
    >>> from sympy.stats import joint_eigen_distribution
    >>> C = CSE('S', 1)
    >>> joint_eigen_distribution(C)
    Lambda(t[1], Product(Abs(exp(I*t[_j]) - exp(I*t[_k]))**4, (_j, _k + 1, 1), (_k, 1, 0))/(2*pi))

    Note
    ====

    As can be seen above in the example, density of CiruclarSymplecticEnsemble
    is not evaluated because the exact definition is based on haar measure of
    unitary group which is not unique.
    """
    sym, dim = _symbol_converter(sym), _sympify(dim)
    model = CircularSymplecticEnsembleModel(sym, dim)
    rmp = RandomMatrixPSpace(sym, model=model)
    return RandomMatrixSymbol(sym, dim, dim, pspace=rmp)

def joint_eigen_distribution(mat):
    """
    For obtaining joint probability distribution
    of eigen values of random matrix.

    Parameters
    ==========

    mat: RandomMatrixSymbol
        The matrix symbol whose eigen values are to be considered.

    Returns
    =======

    Lambda

    Examples
    ========

    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE
    >>> from sympy.stats import joint_eigen_distribution
    >>> U = GUE('U', 2)
    >>> joint_eigen_distribution(U)
    Lambda((l[1], l[2]), exp(-l[1]**2 - l[2]**2)*Product(Abs(l[_i] - l[_j])**2, (_j, _i + 1, 2), (_i, 1, 1))/pi)
    """
    if not isinstance(mat, RandomMatrixSymbol):
        raise ValueError("%s is not of type, RandomMatrixSymbol."%(mat))
    return mat.pspace.model.joint_eigen_distribution()

def JointEigenDistribution(mat):
    """
    Creates joint distribution of eigen values of matrices with random
    expressions.

    Parameters
    ==========

    mat: Matrix
        The matrix under consideration.

    Returns
    =======

    JointDistributionHandmade

    Examples
    ========

    >>> from sympy.stats import Normal, JointEigenDistribution
    >>> from sympy import Matrix
    >>> A = [[Normal('A00', 0, 1), Normal('A01', 0, 1)],
    ... [Normal('A10', 0, 1), Normal('A11', 0, 1)]]
    >>> JointEigenDistribution(Matrix(A))
    JointDistributionHandmade(-sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2
    + A00/2 + A11/2, sqrt(A00**2 - 2*A00*A11 + 4*A01*A10 + A11**2)/2 + A00/2 + A11/2)

    """
    eigenvals = mat.eigenvals(multiple=True)
    if not all(is_random(eigenval) for eigenval in set(eigenvals)):
        raise ValueError("Eigen values do not have any random expression, "
                         "joint distribution cannot be generated.")
    return JointDistributionHandmade(*eigenvals)

def level_spacing_distribution(mat):
    """
    For obtaining distribution of level spacings.

    Parameters
    ==========

    mat: RandomMatrixSymbol
        The random matrix symbol whose eigen values are
        to be considered for finding the level spacings.

    Returns
    =======

    Lambda

    Examples
    ========

    >>> from sympy.stats import GaussianUnitaryEnsemble as GUE
    >>> from sympy.stats import level_spacing_distribution
    >>> U = GUE('U', 2)
    >>> level_spacing_distribution(U)
    Lambda(_s, 32*_s**2*exp(-4*_s**2/pi)/pi**2)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Random_matrix#Distribution_of_level_spacings
    """
    return mat.pspace.model.level_spacing_distribution()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\util\_validators.py ===
"""
Module that contains many useful utilities
for validating data or function arguments
"""
from __future__ import annotations

from collections.abc import (
    Iterable,
    Sequence,
)
from typing import (
    TypeVar,
    overload,
)

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.common import (
    is_bool,
    is_integer,
)

BoolishT = TypeVar("BoolishT", bool, int)
BoolishNoneT = TypeVar("BoolishNoneT", bool, int, None)


def _check_arg_length(fname, args, max_fname_arg_count, compat_args) -> None:
    """
    Checks whether 'args' has length of at most 'compat_args'. Raises
    a TypeError if that is not the case, similar to in Python when a
    function is called with too many arguments.
    """
    if max_fname_arg_count < 0:
        raise ValueError("'max_fname_arg_count' must be non-negative")

    if len(args) > len(compat_args):
        max_arg_count = len(compat_args) + max_fname_arg_count
        actual_arg_count = len(args) + max_fname_arg_count
        argument = "argument" if max_arg_count == 1 else "arguments"

        raise TypeError(
            f"{fname}() takes at most {max_arg_count} {argument} "
            f"({actual_arg_count} given)"
        )


def _check_for_default_values(fname, arg_val_dict, compat_args) -> None:
    """
    Check that the keys in `arg_val_dict` are mapped to their
    default values as specified in `compat_args`.

    Note that this function is to be called only when it has been
    checked that arg_val_dict.keys() is a subset of compat_args
    """
    for key in arg_val_dict:
        # try checking equality directly with '=' operator,
        # as comparison may have been overridden for the left
        # hand object
        try:
            v1 = arg_val_dict[key]
            v2 = compat_args[key]

            # check for None-ness otherwise we could end up
            # comparing a numpy array vs None
            if (v1 is not None and v2 is None) or (v1 is None and v2 is not None):
                match = False
            else:
                match = v1 == v2

            if not is_bool(match):
                raise ValueError("'match' is not a boolean")

        # could not compare them directly, so try comparison
        # using the 'is' operator
        except ValueError:
            match = arg_val_dict[key] is compat_args[key]

        if not match:
            raise ValueError(
                f"the '{key}' parameter is not supported in "
                f"the pandas implementation of {fname}()"
            )


def validate_args(fname, args, max_fname_arg_count, compat_args) -> None:
    """
    Checks whether the length of the `*args` argument passed into a function
    has at most `len(compat_args)` arguments and whether or not all of these
    elements in `args` are set to their default values.

    Parameters
    ----------
    fname : str
        The name of the function being passed the `*args` parameter
    args : tuple
        The `*args` parameter passed into a function
    max_fname_arg_count : int
        The maximum number of arguments that the function `fname`
        can accept, excluding those in `args`. Used for displaying
        appropriate error messages. Must be non-negative.
    compat_args : dict
        A dictionary of keys and their associated default values.
        In order to accommodate buggy behaviour in some versions of `numpy`,
        where a signature displayed keyword arguments but then passed those
        arguments **positionally** internally when calling downstream
        implementations, a dict ensures that the original
        order of the keyword arguments is enforced.

    Raises
    ------
    TypeError
        If `args` contains more values than there are `compat_args`
    ValueError
        If `args` contains values that do not correspond to those
        of the default values specified in `compat_args`
    """
    _check_arg_length(fname, args, max_fname_arg_count, compat_args)

    # We do this so that we can provide a more informative
    # error message about the parameters that we are not
    # supporting in the pandas implementation of 'fname'
    kwargs = dict(zip(compat_args, args))
    _check_for_default_values(fname, kwargs, compat_args)


def _check_for_invalid_keys(fname, kwargs, compat_args) -> None:
    """
    Checks whether 'kwargs' contains any keys that are not
    in 'compat_args' and raises a TypeError if there is one.
    """
    # set(dict) --> set of the dictionary's keys
    diff = set(kwargs) - set(compat_args)

    if diff:
        bad_arg = next(iter(diff))
        raise TypeError(f"{fname}() got an unexpected keyword argument '{bad_arg}'")


def validate_kwargs(fname, kwargs, compat_args) -> None:
    """
    Checks whether parameters passed to the **kwargs argument in a
    function `fname` are valid parameters as specified in `*compat_args`
    and whether or not they are set to their default values.

    Parameters
    ----------
    fname : str
        The name of the function being passed the `**kwargs` parameter
    kwargs : dict
        The `**kwargs` parameter passed into `fname`
    compat_args: dict
        A dictionary of keys that `kwargs` is allowed to have and their
        associated default values

    Raises
    ------
    TypeError if `kwargs` contains keys not in `compat_args`
    ValueError if `kwargs` contains keys in `compat_args` that do not
    map to the default values specified in `compat_args`
    """
    kwds = kwargs.copy()
    _check_for_invalid_keys(fname, kwargs, compat_args)
    _check_for_default_values(fname, kwds, compat_args)


def validate_args_and_kwargs(
    fname, args, kwargs, max_fname_arg_count, compat_args
) -> None:
    """
    Checks whether parameters passed to the *args and **kwargs argument in a
    function `fname` are valid parameters as specified in `*compat_args`
    and whether or not they are set to their default values.

    Parameters
    ----------
    fname: str
        The name of the function being passed the `**kwargs` parameter
    args: tuple
        The `*args` parameter passed into a function
    kwargs: dict
        The `**kwargs` parameter passed into `fname`
    max_fname_arg_count: int
        The minimum number of arguments that the function `fname`
        requires, excluding those in `args`. Used for displaying
        appropriate error messages. Must be non-negative.
    compat_args: dict
        A dictionary of keys that `kwargs` is allowed to
        have and their associated default values.

    Raises
    ------
    TypeError if `args` contains more values than there are
    `compat_args` OR `kwargs` contains keys not in `compat_args`
    ValueError if `args` contains values not at the default value (`None`)
    `kwargs` contains keys in `compat_args` that do not map to the default
    value as specified in `compat_args`

    See Also
    --------
    validate_args : Purely args validation.
    validate_kwargs : Purely kwargs validation.

    """
    # Check that the total number of arguments passed in (i.e.
    # args and kwargs) does not exceed the length of compat_args
    _check_arg_length(
        fname, args + tuple(kwargs.values()), max_fname_arg_count, compat_args
    )

    # Check there is no overlap with the positional and keyword
    # arguments, similar to what is done in actual Python functions
    args_dict = dict(zip(compat_args, args))

    for key in args_dict:
        if key in kwargs:
            raise TypeError(
                f"{fname}() got multiple values for keyword argument '{key}'"
            )

    kwargs.update(args_dict)
    validate_kwargs(fname, kwargs, compat_args)


def validate_bool_kwarg(
    value: BoolishNoneT,
    arg_name: str,
    none_allowed: bool = True,
    int_allowed: bool = False,
) -> BoolishNoneT:
    """
    Ensure that argument passed in arg_name can be interpreted as boolean.

    Parameters
    ----------
    value : bool
        Value to be validated.
    arg_name : str
        Name of the argument. To be reflected in the error message.
    none_allowed : bool, default True
        Whether to consider None to be a valid boolean.
    int_allowed : bool, default False
        Whether to consider integer value to be a valid boolean.

    Returns
    -------
    value
        The same value as input.

    Raises
    ------
    ValueError
        If the value is not a valid boolean.
    """
    good_value = is_bool(value)
    if none_allowed:
        good_value = good_value or (value is None)

    if int_allowed:
        good_value = good_value or isinstance(value, int)

    if not good_value:
        raise ValueError(
            f'For argument "{arg_name}" expected type bool, received '
            f"type {type(value).__name__}."
        )
    return value  # pyright: ignore[reportGeneralTypeIssues]


def validate_fillna_kwargs(value, method, validate_scalar_dict_value: bool = True):
    """
    Validate the keyword arguments to 'fillna'.

    This checks that exactly one of 'value' and 'method' is specified.
    If 'method' is specified, this validates that it's a valid method.

    Parameters
    ----------
    value, method : object
        The 'value' and 'method' keyword arguments for 'fillna'.
    validate_scalar_dict_value : bool, default True
        Whether to validate that 'value' is a scalar or dict. Specifically,
        validate that it is not a list or tuple.

    Returns
    -------
    value, method : object
    """
    from pandas.core.missing import clean_fill_method

    if value is None and method is None:
        raise ValueError("Must specify a fill 'value' or 'method'.")
    if value is None and method is not None:
        method = clean_fill_method(method)

    elif value is not None and method is None:
        if validate_scalar_dict_value and isinstance(value, (list, tuple)):
            raise TypeError(
                '"value" parameter must be a scalar or dict, but '
                f'you passed a "{type(value).__name__}"'
            )

    elif value is not None and method is not None:
        raise ValueError("Cannot specify both 'value' and 'method'.")

    return value, method


def validate_percentile(q: float | Iterable[float]) -> np.ndarray:
    """
    Validate percentiles (used by describe and quantile).

    This function checks if the given float or iterable of floats is a valid percentile
    otherwise raises a ValueError.

    Parameters
    ----------
    q: float or iterable of floats
        A single percentile or an iterable of percentiles.

    Returns
    -------
    ndarray
        An ndarray of the percentiles if valid.

    Raises
    ------
    ValueError if percentiles are not in given interval([0, 1]).
    """
    q_arr = np.asarray(q)
    # Don't change this to an f-string. The string formatting
    # is too expensive for cases where we don't need it.
    msg = "percentiles should all be in the interval [0, 1]"
    if q_arr.ndim == 0:
        if not 0 <= q_arr <= 1:
            raise ValueError(msg)
    else:
        if not all(0 <= qs <= 1 for qs in q_arr):
            raise ValueError(msg)
    return q_arr


@overload
def validate_ascending(ascending: BoolishT) -> BoolishT:
    ...


@overload
def validate_ascending(ascending: Sequence[BoolishT]) -> list[BoolishT]:
    ...


def validate_ascending(
    ascending: bool | int | Sequence[BoolishT],
) -> bool | int | list[BoolishT]:
    """Validate ``ascending`` kwargs for ``sort_index`` method."""
    kwargs = {"none_allowed": False, "int_allowed": True}
    if not isinstance(ascending, Sequence):
        return validate_bool_kwarg(ascending, "ascending", **kwargs)

    return [validate_bool_kwarg(item, "ascending", **kwargs) for item in ascending]


def validate_endpoints(closed: str | None) -> tuple[bool, bool]:
    """
    Check that the `closed` argument is among [None, "left", "right"]

    Parameters
    ----------
    closed : {None, "left", "right"}

    Returns
    -------
    left_closed : bool
    right_closed : bool

    Raises
    ------
    ValueError : if argument is not among valid values
    """
    left_closed = False
    right_closed = False

    if closed is None:
        left_closed = True
        right_closed = True
    elif closed == "left":
        left_closed = True
    elif closed == "right":
        right_closed = True
    else:
        raise ValueError("Closed has to be either 'left', 'right' or None")

    return left_closed, right_closed


def validate_inclusive(inclusive: str | None) -> tuple[bool, bool]:
    """
    Check that the `inclusive` argument is among {"both", "neither", "left", "right"}.

    Parameters
    ----------
    inclusive : {"both", "neither", "left", "right"}

    Returns
    -------
    left_right_inclusive : tuple[bool, bool]

    Raises
    ------
    ValueError : if argument is not among valid values
    """
    left_right_inclusive: tuple[bool, bool] | None = None

    if isinstance(inclusive, str):
        left_right_inclusive = {
            "both": (True, True),
            "left": (True, False),
            "right": (False, True),
            "neither": (False, False),
        }.get(inclusive)

    if left_right_inclusive is None:
        raise ValueError(
            "Inclusive has to be either 'both', 'neither', 'left' or 'right'"
        )

    return left_right_inclusive


def validate_insert_loc(loc: int, length: int) -> int:
    """
    Check that we have an integer between -length and length, inclusive.

    Standardize negative loc to within [0, length].

    The exceptions we raise on failure match np.insert.
    """
    if not is_integer(loc):
        raise TypeError(f"loc must be an integer between -{length} and {length}")

    if loc < 0:
        loc += length
    if not 0 <= loc <= length:
        raise IndexError(f"loc must be an integer between -{length} and {length}")
    return loc  # pyright: ignore[reportGeneralTypeIssues]


def check_dtype_backend(dtype_backend) -> None:
    if dtype_backend is not lib.no_default:
        if dtype_backend not in ["numpy_nullable", "pyarrow"]:
            raise ValueError(
                f"dtype_backend {dtype_backend} is invalid, only 'numpy_nullable' and "
                f"'pyarrow' are allowed.",
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\numberfields\resolvent_lookup.py ===
"""Lookup table for Galois resolvents for polys of degree 4 through 6. """
# This table was generated by a call to
# `sympy.polys.numberfields.galois_resolvents.generate_lambda_lookup()`.
# The entire job took 543.23s.
# Of this, Case (6, 1) took 539.03s.
# The final polynomial of Case (6, 1) alone took 455.09s.
resolvent_coeff_lambdas = {
    (4, 0): [
        lambda s1, s2, s3, s4: (-2*s1*s2 + 6*s3),
        lambda s1, s2, s3, s4: (2*s1**3*s3 + s1**2*s2**2 + s1**2*s4 - 17*s1*s2*s3 + 2*s2**3 - 8*s2*s4 + 24*s3**2),
        lambda s1, s2, s3, s4: (-2*s1**5*s4 - 2*s1**4*s2*s3 + 10*s1**3*s2*s4 + 8*s1**3*s3**2 + 10*s1**2*s2**2*s3 -
12*s1**2*s3*s4 - 2*s1*s2**4 - 54*s1*s2*s3**2 + 32*s1*s4**2 + 8*s2**3*s3 - 32*s2*s3*s4
+ 56*s3**3),
        lambda s1, s2, s3, s4: (2*s1**6*s2*s4 + s1**6*s3**2 - 5*s1**5*s3*s4 - 11*s1**4*s2**2*s4 - 13*s1**4*s2*s3**2
+ 7*s1**4*s4**2 + 3*s1**3*s2**3*s3 + 30*s1**3*s2*s3*s4 + 22*s1**3*s3**3 + 10*s1**2*s2**3*s4
+ 33*s1**2*s2**2*s3**2 - 72*s1**2*s2*s4**2 - 36*s1**2*s3**2*s4 - 13*s1*s2**4*s3 +
48*s1*s2**2*s3*s4 - 116*s1*s2*s3**3 + 144*s1*s3*s4**2 + s2**6 - 12*s2**4*s4 + 22*s2**3*s3**2
+ 48*s2**2*s4**2 - 120*s2*s3**2*s4 + 96*s3**4 - 64*s4**3),
        lambda s1, s2, s3, s4: (-2*s1**8*s3*s4 - s1**7*s4**2 + 22*s1**6*s2*s3*s4 + 2*s1**6*s3**3 - 2*s1**5*s2**3*s4
- s1**5*s2**2*s3**2 - 29*s1**5*s3**2*s4 - 60*s1**4*s2**2*s3*s4 - 19*s1**4*s2*s3**3
+ 38*s1**4*s3*s4**2 + 9*s1**3*s2**4*s4 + 10*s1**3*s2**3*s3**2 + 24*s1**3*s2**2*s4**2
+ 134*s1**3*s2*s3**2*s4 + 28*s1**3*s3**4 + 16*s1**3*s4**3 - s1**2*s2**5*s3 - 4*s1**2*s2**3*s3*s4
+ 34*s1**2*s2**2*s3**3 - 288*s1**2*s2*s3*s4**2 - 104*s1**2*s3**3*s4 - 19*s1*s2**4*s3**2
+ 120*s1*s2**2*s3**2*s4 - 128*s1*s2*s3**4 + 336*s1*s3**2*s4**2 + 2*s2**6*s3 - 24*s2**4*s3*s4
+ 28*s2**3*s3**3 + 96*s2**2*s3*s4**2 - 176*s2*s3**3*s4 + 96*s3**5 - 128*s3*s4**3),
        lambda s1, s2, s3, s4: (s1**10*s4**2 - 11*s1**8*s2*s4**2 - 2*s1**8*s3**2*s4 + s1**7*s2**2*s3*s4 + 15*s1**7*s3*s4**2
+ 45*s1**6*s2**2*s4**2 + 17*s1**6*s2*s3**2*s4 + s1**6*s3**4 - 5*s1**6*s4**3 - 12*s1**5*s2**3*s3*s4
- 133*s1**5*s2*s3*s4**2 - 22*s1**5*s3**3*s4 + s1**4*s2**5*s4 - 76*s1**4*s2**3*s4**2
- 6*s1**4*s2**2*s3**2*s4 - 12*s1**4*s2*s3**4 + 32*s1**4*s2*s4**3 + 128*s1**4*s3**2*s4**2
+ 29*s1**3*s2**4*s3*s4 + 2*s1**3*s2**3*s3**3 + 344*s1**3*s2**2*s3*s4**2 + 48*s1**3*s2*s3**3*s4
+ 16*s1**3*s3**5 - 48*s1**3*s3*s4**3 - 4*s1**2*s2**6*s4 + 32*s1**2*s2**4*s4**2 - 134*s1**2*s2**3*s3**2*s4
+ 36*s1**2*s2**2*s3**4 - 64*s1**2*s2**2*s4**3 - 648*s1**2*s2*s3**2*s4**2 - 48*s1**2*s3**4*s4
+ 16*s1*s2**5*s3*s4 - 12*s1*s2**4*s3**3 - 128*s1*s2**3*s3*s4**2 + 296*s1*s2**2*s3**3*s4
- 96*s1*s2*s3**5 + 256*s1*s2*s3*s4**3 + 416*s1*s3**3*s4**2 + s2**6*s3**2 - 28*s2**4*s3**2*s4
+ 16*s2**3*s3**4 + 176*s2**2*s3**2*s4**2 - 224*s2*s3**4*s4 + 64*s3**6 - 320*s3**2*s4**3)
    ],
    (4, 1): [
        lambda s1, s2, s3, s4: (-s2),
        lambda s1, s2, s3, s4: (s1*s3 - 4*s4),
        lambda s1, s2, s3, s4: (-s1**2*s4 + 4*s2*s4 - s3**2)
    ],
    (5, 1): [
        lambda s1, s2, s3, s4, s5: (-2*s1*s3 + 8*s4),
        lambda s1, s2, s3, s4, s5: (-8*s1**3*s5 + 2*s1**2*s2*s4 + s1**2*s3**2 + 30*s1*s2*s5 - 14*s1*s3*s4 - 6*s2**2*s4
+ 2*s2*s3**2 - 50*s3*s5 + 40*s4**2),
        lambda s1, s2, s3, s4, s5: (16*s1**4*s3*s5 - 2*s1**4*s4**2 - 2*s1**3*s2**2*s5 - 2*s1**3*s2*s3*s4 - 44*s1**3*s4*s5
- 66*s1**2*s2*s3*s5 + 21*s1**2*s2*s4**2 + 6*s1**2*s3**2*s4 - 50*s1**2*s5**2 + 9*s1*s2**3*s5
+ 5*s1*s2**2*s3*s4 - 2*s1*s2*s3**3 + 190*s1*s2*s4*s5 + 120*s1*s3**2*s5 - 80*s1*s3*s4**2
- 15*s2**2*s3*s5 - 40*s2**2*s4**2 + 21*s2*s3**2*s4 + 125*s2*s5**2 - 2*s3**4 - 400*s3*s4*s5
+ 160*s4**3),
        lambda s1, s2, s3, s4, s5: (16*s1**6*s5**2 - 8*s1**5*s2*s4*s5 - 8*s1**5*s3**2*s5 + 2*s1**5*s3*s4**2 + 2*s1**4*s2**2*s3*s5
+ s1**4*s2**2*s4**2 - 120*s1**4*s2*s5**2 + 68*s1**4*s3*s4*s5 - 8*s1**4*s4**3 + 46*s1**3*s2**2*s4*s5
+ 28*s1**3*s2*s3**2*s5 - 19*s1**3*s2*s3*s4**2 + 250*s1**3*s3*s5**2 - 144*s1**3*s4**2*s5
- 9*s1**2*s2**3*s3*s5 - 6*s1**2*s2**3*s4**2 + 3*s1**2*s2**2*s3**2*s4 + 225*s1**2*s2**2*s5**2
- 354*s1**2*s2*s3*s4*s5 + 76*s1**2*s2*s4**3 - 70*s1**2*s3**3*s5 + 41*s1**2*s3**2*s4**2
- 200*s1**2*s4*s5**2 - 54*s1*s2**3*s4*s5 + 45*s1*s2**2*s3**2*s5 + 30*s1*s2**2*s3*s4**2
- 19*s1*s2*s3**3*s4 - 875*s1*s2*s3*s5**2 + 640*s1*s2*s4**2*s5 + 2*s1*s3**5 + 630*s1*s3**2*s4*s5
- 264*s1*s3*s4**3 + 9*s2**4*s4**2 - 6*s2**3*s3**2*s4 + s2**2*s3**4 + 90*s2**2*s3*s4*s5
- 136*s2**2*s4**3 - 50*s2*s3**3*s5 + 76*s2*s3**2*s4**2 + 500*s2*s4*s5**2 - 8*s3**4*s4
+ 625*s3**2*s5**2 - 1400*s3*s4**2*s5 + 400*s4**4),
        lambda s1, s2, s3, s4, s5: (-32*s1**7*s3*s5**2 + 8*s1**7*s4**2*s5 + 8*s1**6*s2**2*s5**2 + 8*s1**6*s2*s3*s4*s5
- 2*s1**6*s2*s4**3 + 48*s1**6*s4*s5**2 - 2*s1**5*s2**3*s4*s5 + 264*s1**5*s2*s3*s5**2
- 94*s1**5*s2*s4**2*s5 - 24*s1**5*s3**2*s4*s5 + 6*s1**5*s3*s4**3 - 56*s1**5*s5**3
- 66*s1**4*s2**3*s5**2 - 50*s1**4*s2**2*s3*s4*s5 + 19*s1**4*s2**2*s4**3 + 8*s1**4*s2*s3**3*s5
- 2*s1**4*s2*s3**2*s4**2 - 318*s1**4*s2*s4*s5**2 - 352*s1**4*s3**2*s5**2 + 166*s1**4*s3*s4**2*s5
+ 3*s1**4*s4**4 + 15*s1**3*s2**4*s4*s5 - 2*s1**3*s2**3*s3**2*s5 - s1**3*s2**3*s3*s4**2
- 574*s1**3*s2**2*s3*s5**2 + 347*s1**3*s2**2*s4**2*s5 + 194*s1**3*s2*s3**2*s4*s5 -
89*s1**3*s2*s3*s4**3 + 350*s1**3*s2*s5**3 - 8*s1**3*s3**4*s5 + 4*s1**3*s3**3*s4**2
+ 1090*s1**3*s3*s4*s5**2 - 364*s1**3*s4**3*s5 + 162*s1**2*s2**4*s5**2 + 33*s1**2*s2**3*s3*s4*s5
- 51*s1**2*s2**3*s4**3 - 32*s1**2*s2**2*s3**3*s5 + 28*s1**2*s2**2*s3**2*s4**2 + 305*s1**2*s2**2*s4*s5**2
- 2*s1**2*s2*s3**4*s4 + 1340*s1**2*s2*s3**2*s5**2 - 901*s1**2*s2*s3*s4**2*s5 + 76*s1**2*s2*s4**4
- 234*s1**2*s3**3*s4*s5 + 102*s1**2*s3**2*s4**3 - 750*s1**2*s3*s5**3 - 550*s1**2*s4**2*s5**2
- 27*s1*s2**5*s4*s5 + 9*s1*s2**4*s3**2*s5 + 3*s1*s2**4*s3*s4**2 - s1*s2**3*s3**3*s4
+ 180*s1*s2**3*s3*s5**2 - 366*s1*s2**3*s4**2*s5 - 231*s1*s2**2*s3**2*s4*s5 + 212*s1*s2**2*s3*s4**3
- 375*s1*s2**2*s5**3 + 112*s1*s2*s3**4*s5 - 89*s1*s2*s3**3*s4**2 - 3075*s1*s2*s3*s4*s5**2
+ 1640*s1*s2*s4**3*s5 + 6*s1*s3**5*s4 - 850*s1*s3**3*s5**2 + 1220*s1*s3**2*s4**2*s5
- 384*s1*s3*s4**4 + 2500*s1*s4*s5**3 - 108*s2**5*s5**2 + 117*s2**4*s3*s4*s5 + 32*s2**4*s4**3
- 31*s2**3*s3**3*s5 - 51*s2**3*s3**2*s4**2 + 525*s2**3*s4*s5**2 + 19*s2**2*s3**4*s4
- 325*s2**2*s3**2*s5**2 + 260*s2**2*s3*s4**2*s5 - 256*s2**2*s4**4 - 2*s2*s3**6 + 105*s2*s3**3*s4*s5
+ 76*s2*s3**2*s4**3 + 625*s2*s3*s5**3 - 500*s2*s4**2*s5**2 - 58*s3**5*s5 + 3*s3**4*s4**2
+ 2750*s3**2*s4*s5**2 - 2400*s3*s4**3*s5 + 512*s4**5 - 3125*s5**4),
        lambda s1, s2, s3, s4, s5: (16*s1**8*s3**2*s5**2 - 8*s1**8*s3*s4**2*s5 + s1**8*s4**4 - 8*s1**7*s2**2*s3*s5**2
+ 2*s1**7*s2**2*s4**2*s5 - 48*s1**7*s3*s4*s5**2 + 12*s1**7*s4**3*s5 + s1**6*s2**4*s5**2
+ 12*s1**6*s2**2*s4*s5**2 - 144*s1**6*s2*s3**2*s5**2 + 88*s1**6*s2*s3*s4**2*s5 - 13*s1**6*s2*s4**4
+ 56*s1**6*s3*s5**3 + 86*s1**6*s4**2*s5**2 + 72*s1**5*s2**3*s3*s5**2 - 22*s1**5*s2**3*s4**2*s5
- 4*s1**5*s2**2*s3**2*s4*s5 + s1**5*s2**2*s3*s4**3 - 14*s1**5*s2**2*s5**3 + 304*s1**5*s2*s3*s4*s5**2
- 148*s1**5*s2*s4**3*s5 + 152*s1**5*s3**3*s5**2 - 54*s1**5*s3**2*s4**2*s5 + 5*s1**5*s3*s4**4
- 468*s1**5*s4*s5**3 - 9*s1**4*s2**5*s5**2 + s1**4*s2**4*s3*s4*s5 - 76*s1**4*s2**3*s4*s5**2
+ 370*s1**4*s2**2*s3**2*s5**2 - 287*s1**4*s2**2*s3*s4**2*s5 + 65*s1**4*s2**2*s4**4
- 28*s1**4*s2*s3**3*s4*s5 + 5*s1**4*s2*s3**2*s4**3 - 200*s1**4*s2*s3*s5**3 - 294*s1**4*s2*s4**2*s5**2
+ 8*s1**4*s3**5*s5 - 2*s1**4*s3**4*s4**2 - 676*s1**4*s3**2*s4*s5**2 + 180*s1**4*s3*s4**3*s5
+ 17*s1**4*s4**5 + 625*s1**4*s5**4 - 210*s1**3*s2**4*s3*s5**2 + 76*s1**3*s2**4*s4**2*s5
+ 43*s1**3*s2**3*s3**2*s4*s5 - 15*s1**3*s2**3*s3*s4**3 + 50*s1**3*s2**3*s5**3 - 6*s1**3*s2**2*s3**4*s5
+ 2*s1**3*s2**2*s3**3*s4**2 - 397*s1**3*s2**2*s3*s4*s5**2 + 514*s1**3*s2**2*s4**3*s5
- 700*s1**3*s2*s3**3*s5**2 + 447*s1**3*s2*s3**2*s4**2*s5 - 118*s1**3*s2*s3*s4**4 +
2300*s1**3*s2*s4*s5**3 - 12*s1**3*s3**4*s4*s5 + 6*s1**3*s3**3*s4**3 + 250*s1**3*s3**2*s5**3
+ 1470*s1**3*s3*s4**2*s5**2 - 276*s1**3*s4**4*s5 + 27*s1**2*s2**6*s5**2 - 9*s1**2*s2**5*s3*s4*s5
+ s1**2*s2**5*s4**3 + s1**2*s2**4*s3**3*s5 + 141*s1**2*s2**4*s4*s5**2 - 185*s1**2*s2**3*s3**2*s5**2
+ 168*s1**2*s2**3*s3*s4**2*s5 - 128*s1**2*s2**3*s4**4 + 93*s1**2*s2**2*s3**3*s4*s5
+ 19*s1**2*s2**2*s3**2*s4**3 - 125*s1**2*s2**2*s3*s5**3 - 610*s1**2*s2**2*s4**2*s5**2
- 36*s1**2*s2*s3**5*s5 + 5*s1**2*s2*s3**4*s4**2 + 1995*s1**2*s2*s3**2*s4*s5**2 - 1174*s1**2*s2*s3*s4**3*s5
- 16*s1**2*s2*s4**5 - 3125*s1**2*s2*s5**4 + 375*s1**2*s3**4*s5**2 - 172*s1**2*s3**3*s4**2*s5
+ 82*s1**2*s3**2*s4**4 - 3500*s1**2*s3*s4*s5**3 - 1450*s1**2*s4**3*s5**2 + 198*s1*s2**5*s3*s5**2
- 78*s1*s2**5*s4**2*s5 - 95*s1*s2**4*s3**2*s4*s5 + 44*s1*s2**4*s3*s4**3 + 25*s1*s2**3*s3**4*s5
- 15*s1*s2**3*s3**3*s4**2 + 15*s1*s2**3*s3*s4*s5**2 - 384*s1*s2**3*s4**3*s5 + s1*s2**2*s3**5*s4
+ 525*s1*s2**2*s3**3*s5**2 - 528*s1*s2**2*s3**2*s4**2*s5 + 384*s1*s2**2*s3*s4**4 -
1750*s1*s2**2*s4*s5**3 - 29*s1*s2*s3**4*s4*s5 - 118*s1*s2*s3**3*s4**3 + 625*s1*s2*s3**2*s5**3
- 850*s1*s2*s3*s4**2*s5**2 + 1760*s1*s2*s4**4*s5 + 38*s1*s3**6*s5 + 5*s1*s3**5*s4**2
- 2050*s1*s3**3*s4*s5**2 + 780*s1*s3**2*s4**3*s5 - 192*s1*s3*s4**5 + 3125*s1*s3*s5**4
+ 7500*s1*s4**2*s5**3 - 27*s2**7*s5**2 + 18*s2**6*s3*s4*s5 - 4*s2**6*s4**3 - 4*s2**5*s3**3*s5
+ s2**5*s3**2*s4**2 - 99*s2**5*s4*s5**2 - 150*s2**4*s3**2*s5**2 + 196*s2**4*s3*s4**2*s5
+ 48*s2**4*s4**4 + 12*s2**3*s3**3*s4*s5 - 128*s2**3*s3**2*s4**3 + 1200*s2**3*s4**2*s5**2
- 12*s2**2*s3**5*s5 + 65*s2**2*s3**4*s4**2 - 725*s2**2*s3**2*s4*s5**2 - 160*s2**2*s3*s4**3*s5
- 192*s2**2*s4**5 + 3125*s2**2*s5**4 - 13*s2*s3**6*s4 - 125*s2*s3**4*s5**2 + 590*s2*s3**3*s4**2*s5
- 16*s2*s3**2*s4**4 - 1250*s2*s3*s4*s5**3 - 2000*s2*s4**3*s5**2 + s3**8 - 124*s3**5*s4*s5
+ 17*s3**4*s4**3 + 3250*s3**2*s4**2*s5**2 - 1600*s3*s4**4*s5 + 256*s4**6 - 9375*s4*s5**4)
    ],
    (6, 1): [
        lambda s1, s2, s3, s4, s5, s6: (8*s1*s5 - 2*s2*s4 - 18*s6),
        lambda s1, s2, s3, s4, s5, s6: (-50*s1**2*s4*s6 + 40*s1**2*s5**2 + 30*s1*s2*s3*s6 - 14*s1*s2*s4*s5 - 6*s1*s3**2*s5
+ 2*s1*s3*s4**2 - 30*s1*s5*s6 - 8*s2**3*s6 + 2*s2**2*s3*s5 + s2**2*s4**2 + 114*s2*s4*s6
- 50*s2*s5**2 - 54*s3**2*s6 + 30*s3*s4*s5 - 8*s4**3 - 135*s6**2),
        lambda s1, s2, s3, s4, s5, s6: (125*s1**3*s3*s6**2 - 400*s1**3*s4*s5*s6 + 160*s1**3*s5**3 - 50*s1**2*s2**2*s6**2 +
190*s1**2*s2*s3*s5*s6 + 120*s1**2*s2*s4**2*s6 - 80*s1**2*s2*s4*s5**2 - 15*s1**2*s3**2*s4*s6
- 40*s1**2*s3**2*s5**2 + 21*s1**2*s3*s4**2*s5 - 2*s1**2*s4**4 + 900*s1**2*s4*s6**2
- 80*s1**2*s5**2*s6 - 44*s1*s2**3*s5*s6 - 66*s1*s2**2*s3*s4*s6 + 21*s1*s2**2*s3*s5**2
+ 6*s1*s2**2*s4**2*s5 + 9*s1*s2*s3**3*s6 + 5*s1*s2*s3**2*s4*s5 - 2*s1*s2*s3*s4**3
- 990*s1*s2*s3*s6**2 + 920*s1*s2*s4*s5*s6 - 400*s1*s2*s5**3 - 135*s1*s3**2*s5*s6 -
126*s1*s3*s4**2*s6 + 190*s1*s3*s4*s5**2 - 44*s1*s4**3*s5 - 2070*s1*s5*s6**2 + 16*s2**4*s4*s6
- 2*s2**4*s5**2 - 2*s2**3*s3**2*s6 - 2*s2**3*s3*s4*s5 + 304*s2**3*s6**2 - 126*s2**2*s3*s5*s6
- 232*s2**2*s4**2*s6 + 120*s2**2*s4*s5**2 + 198*s2*s3**2*s4*s6 - 15*s2*s3**2*s5**2
- 66*s2*s3*s4**2*s5 + 16*s2*s4**4 - 1440*s2*s4*s6**2 + 900*s2*s5**2*s6 - 27*s3**4*s6
+ 9*s3**3*s4*s5 - 2*s3**2*s4**3 + 1350*s3**2*s6**2 - 990*s3*s4*s5*s6 + 125*s3*s5**3
+ 304*s4**3*s6 - 50*s4**2*s5**2 + 3240*s6**3),
        lambda s1, s2, s3, s4, s5, s6: (500*s1**4*s3*s5*s6**2 + 625*s1**4*s4**2*s6**2 - 1400*s1**4*s4*s5**2*s6 + 400*s1**4*s5**4
- 200*s1**3*s2**2*s5*s6**2 - 875*s1**3*s2*s3*s4*s6**2 + 640*s1**3*s2*s3*s5**2*s6 +
630*s1**3*s2*s4**2*s5*s6 - 264*s1**3*s2*s4*s5**3 + 90*s1**3*s3**2*s4*s5*s6 - 136*s1**3*s3**2*s5**3
- 50*s1**3*s3*s4**3*s6 + 76*s1**3*s3*s4**2*s5**2 - 1125*s1**3*s3*s6**3 - 8*s1**3*s4**4*s5
+ 2550*s1**3*s4*s5*s6**2 - 200*s1**3*s5**3*s6 + 250*s1**2*s2**3*s4*s6**2 - 144*s1**2*s2**3*s5**2*s6
+ 225*s1**2*s2**2*s3**2*s6**2 - 354*s1**2*s2**2*s3*s4*s5*s6 + 76*s1**2*s2**2*s3*s5**3
- 70*s1**2*s2**2*s4**3*s6 + 41*s1**2*s2**2*s4**2*s5**2 + 450*s1**2*s2**2*s6**3 - 54*s1**2*s2*s3**3*s5*s6
+ 45*s1**2*s2*s3**2*s4**2*s6 + 30*s1**2*s2*s3**2*s4*s5**2 - 19*s1**2*s2*s3*s4**3*s5
- 2880*s1**2*s2*s3*s5*s6**2 + 2*s1**2*s2*s4**5 - 3480*s1**2*s2*s4**2*s6**2 + 4692*s1**2*s2*s4*s5**2*s6
- 1400*s1**2*s2*s5**4 + 9*s1**2*s3**4*s5**2 - 6*s1**2*s3**3*s4**2*s5 + s1**2*s3**2*s4**4
+ 1485*s1**2*s3**2*s4*s6**2 - 522*s1**2*s3**2*s5**2*s6 - 1257*s1**2*s3*s4**2*s5*s6
+ 640*s1**2*s3*s4*s5**3 + 218*s1**2*s4**4*s6 - 144*s1**2*s4**3*s5**2 + 1350*s1**2*s4*s6**3
- 5175*s1**2*s5**2*s6**2 - 120*s1*s2**4*s3*s6**2 + 68*s1*s2**4*s4*s5*s6 - 8*s1*s2**4*s5**3
+ 46*s1*s2**3*s3**2*s5*s6 + 28*s1*s2**3*s3*s4**2*s6 - 19*s1*s2**3*s3*s4*s5**2 + 868*s1*s2**3*s5*s6**2
- 9*s1*s2**2*s3**3*s4*s6 - 6*s1*s2**2*s3**3*s5**2 + 3*s1*s2**2*s3**2*s4**2*s5 + 2484*s1*s2**2*s3*s4*s6**2
- 1257*s1*s2**2*s3*s5**2*s6 - 1356*s1*s2**2*s4**2*s5*s6 + 630*s1*s2**2*s4*s5**3 -
891*s1*s2*s3**3*s6**2 + 882*s1*s2*s3**2*s4*s5*s6 + 90*s1*s2*s3**2*s5**3 + 84*s1*s2*s3*s4**3*s6
- 354*s1*s2*s3*s4**2*s5**2 + 3240*s1*s2*s3*s6**3 + 68*s1*s2*s4**4*s5 - 4392*s1*s2*s4*s5*s6**2
+ 2550*s1*s2*s5**3*s6 + 54*s1*s3**4*s5*s6 - 54*s1*s3**3*s4**2*s6 - 54*s1*s3**3*s4*s5**2
+ 46*s1*s3**2*s4**3*s5 + 2727*s1*s3**2*s5*s6**2 - 8*s1*s3*s4**5 + 756*s1*s3*s4**2*s6**2
- 2880*s1*s3*s4*s5**2*s6 + 500*s1*s3*s5**4 + 868*s1*s4**3*s5*s6 - 200*s1*s4**2*s5**3
+ 8100*s1*s5*s6**3 + 16*s2**6*s6**2 - 8*s2**5*s3*s5*s6 - 8*s2**5*s4**2*s6 + 2*s2**5*s4*s5**2
+ 2*s2**4*s3**2*s4*s6 + s2**4*s3**2*s5**2 - 688*s2**4*s4*s6**2 + 218*s2**4*s5**2*s6
+ 234*s2**3*s3**2*s6**2 + 84*s2**3*s3*s4*s5*s6 - 50*s2**3*s3*s5**3 + 168*s2**3*s4**3*s6
- 70*s2**3*s4**2*s5**2 - 1224*s2**3*s6**3 - 54*s2**2*s3**3*s5*s6 - 144*s2**2*s3**2*s4**2*s6
+ 45*s2**2*s3**2*s4*s5**2 + 28*s2**2*s3*s4**3*s5 + 756*s2**2*s3*s5*s6**2 - 8*s2**2*s4**5
+ 4320*s2**2*s4**2*s6**2 - 3480*s2**2*s4*s5**2*s6 + 625*s2**2*s5**4 + 27*s2*s3**4*s4*s6
- 9*s2*s3**3*s4**2*s5 + 2*s2*s3**2*s4**4 - 4752*s2*s3**2*s4*s6**2 + 1485*s2*s3**2*s5**2*s6
+ 2484*s2*s3*s4**2*s5*s6 - 875*s2*s3*s4*s5**3 - 688*s2*s4**4*s6 + 250*s2*s4**3*s5**2
- 4536*s2*s4*s6**3 + 1350*s2*s5**2*s6**2 + 972*s3**4*s6**2 - 891*s3**3*s4*s5*s6 +
234*s3**2*s4**3*s6 + 225*s3**2*s4**2*s5**2 - 1944*s3**2*s6**3 - 120*s3*s4**4*s5 +
3240*s3*s4*s5*s6**2 - 1125*s3*s5**3*s6 + 16*s4**6 - 1224*s4**3*s6**2 + 450*s4**2*s5**2*s6),
        lambda s1, s2, s3, s4, s5, s6: (-3125*s1**6*s6**4 + 2500*s1**5*s2*s5*s6**3 + 625*s1**5*s3*s4*s6**3 - 500*s1**5*s3*s5**2*s6**2
+ 2750*s1**5*s4**2*s5*s6**2 - 2400*s1**5*s4*s5**3*s6 + 512*s1**5*s5**5 - 750*s1**4*s2**2*s4*s6**3
- 550*s1**4*s2**2*s5**2*s6**2 - 375*s1**4*s2*s3**2*s6**3 - 3075*s1**4*s2*s3*s4*s5*s6**2
+ 1640*s1**4*s2*s3*s5**3*s6 - 850*s1**4*s2*s4**3*s6**2 + 1220*s1**4*s2*s4**2*s5**2*s6
- 384*s1**4*s2*s4*s5**4 + 22500*s1**4*s2*s6**4 + 525*s1**4*s3**3*s5*s6**2 - 325*s1**4*s3**2*s4**2*s6**2
+ 260*s1**4*s3**2*s4*s5**2*s6 - 256*s1**4*s3**2*s5**4 + 105*s1**4*s3*s4**3*s5*s6 +
76*s1**4*s3*s4**2*s5**3 + 375*s1**4*s3*s5*s6**3 - 58*s1**4*s4**5*s6 + 3*s1**4*s4**4*s5**2
- 12750*s1**4*s4**2*s6**3 + 3700*s1**4*s4*s5**2*s6**2 + 640*s1**4*s5**4*s6 + 350*s1**3*s2**3*s3*s6**3
+ 1090*s1**3*s2**3*s4*s5*s6**2 - 364*s1**3*s2**3*s5**3*s6 + 305*s1**3*s2**2*s3**2*s5*s6**2
+ 1340*s1**3*s2**2*s3*s4**2*s6**2 - 901*s1**3*s2**2*s3*s4*s5**2*s6 + 76*s1**3*s2**2*s3*s5**4
- 234*s1**3*s2**2*s4**3*s5*s6 + 102*s1**3*s2**2*s4**2*s5**3 - 16650*s1**3*s2**2*s5*s6**3
+ 180*s1**3*s2*s3**3*s4*s6**2 - 366*s1**3*s2*s3**3*s5**2*s6 - 231*s1**3*s2*s3**2*s4**2*s5*s6
+ 212*s1**3*s2*s3**2*s4*s5**3 + 112*s1**3*s2*s3*s4**4*s6 - 89*s1**3*s2*s3*s4**3*s5**2
+ 10950*s1**3*s2*s3*s4*s6**3 + 1555*s1**3*s2*s3*s5**2*s6**2 + 6*s1**3*s2*s4**5*s5
- 9540*s1**3*s2*s4**2*s5*s6**2 + 9016*s1**3*s2*s4*s5**3*s6 - 2400*s1**3*s2*s5**5 -
108*s1**3*s3**5*s6**2 + 117*s1**3*s3**4*s4*s5*s6 + 32*s1**3*s3**4*s5**3 - 31*s1**3*s3**3*s4**3*s6
- 51*s1**3*s3**3*s4**2*s5**2 - 2025*s1**3*s3**3*s6**3 + 19*s1**3*s3**2*s4**4*s5 +
2955*s1**3*s3**2*s4*s5*s6**2 - 1436*s1**3*s3**2*s5**3*s6 - 2*s1**3*s3*s4**6 + 2770*s1**3*s3*s4**3*s6**2
- 5123*s1**3*s3*s4**2*s5**2*s6 + 1640*s1**3*s3*s4*s5**4 - 40500*s1**3*s3*s6**4 + 914*s1**3*s4**4*s5*s6
- 364*s1**3*s4**3*s5**3 + 53550*s1**3*s4*s5*s6**3 - 17930*s1**3*s5**3*s6**2 - 56*s1**2*s2**5*s6**3
- 318*s1**2*s2**4*s3*s5*s6**2 - 352*s1**2*s2**4*s4**2*s6**2 + 166*s1**2*s2**4*s4*s5**2*s6
+ 3*s1**2*s2**4*s5**4 - 574*s1**2*s2**3*s3**2*s4*s6**2 + 347*s1**2*s2**3*s3**2*s5**2*s6
+ 194*s1**2*s2**3*s3*s4**2*s5*s6 - 89*s1**2*s2**3*s3*s4*s5**3 - 8*s1**2*s2**3*s4**4*s6
+ 4*s1**2*s2**3*s4**3*s5**2 + 560*s1**2*s2**3*s4*s6**3 + 3662*s1**2*s2**3*s5**2*s6**2
+ 162*s1**2*s2**2*s3**4*s6**2 + 33*s1**2*s2**2*s3**3*s4*s5*s6 - 51*s1**2*s2**2*s3**3*s5**3
- 32*s1**2*s2**2*s3**2*s4**3*s6 + 28*s1**2*s2**2*s3**2*s4**2*s5**2 + 270*s1**2*s2**2*s3**2*s6**3
- 2*s1**2*s2**2*s3*s4**4*s5 + 4872*s1**2*s2**2*s3*s4*s5*s6**2 - 5123*s1**2*s2**2*s3*s5**3*s6
+ 2144*s1**2*s2**2*s4**3*s6**2 - 2812*s1**2*s2**2*s4**2*s5**2*s6 + 1220*s1**2*s2**2*s4*s5**4
- 37800*s1**2*s2**2*s6**4 - 27*s1**2*s2*s3**5*s5*s6 + 9*s1**2*s2*s3**4*s4**2*s6 +
3*s1**2*s2*s3**4*s4*s5**2 - s1**2*s2*s3**3*s4**3*s5 - 3078*s1**2*s2*s3**3*s5*s6**2
- 4014*s1**2*s2*s3**2*s4**2*s6**2 + 5412*s1**2*s2*s3**2*s4*s5**2*s6 + 260*s1**2*s2*s3**2*s5**4
- 310*s1**2*s2*s3*s4**3*s5*s6 - 901*s1**2*s2*s3*s4**2*s5**3 - 3780*s1**2*s2*s3*s5*s6**3
+ 166*s1**2*s2*s4**4*s5**2 + 40320*s1**2*s2*s4**2*s6**3 - 25344*s1**2*s2*s4*s5**2*s6**2
+ 3700*s1**2*s2*s5**4*s6 + 918*s1**2*s3**4*s4*s6**2 + 27*s1**2*s3**4*s5**2*s6 - 342*s1**2*s3**3*s4**2*s5*s6
- 366*s1**2*s3**3*s4*s5**3 + 32*s1**2*s3**2*s4**4*s6 + 347*s1**2*s3**2*s4**3*s5**2
- 4590*s1**2*s3**2*s4*s6**3 + 594*s1**2*s3**2*s5**2*s6**2 - 94*s1**2*s3*s4**5*s5 +
3618*s1**2*s3*s4**2*s5*s6**2 + 1555*s1**2*s3*s4*s5**3*s6 - 500*s1**2*s3*s5**5 + 8*s1**2*s4**7
- 7192*s1**2*s4**4*s6**2 + 3662*s1**2*s4**3*s5**2*s6 - 550*s1**2*s4**2*s5**4 - 48600*s1**2*s4*s6**4
+ 1080*s1**2*s5**2*s6**3 + 48*s1*s2**6*s5*s6**2 + 264*s1*s2**5*s3*s4*s6**2 - 94*s1*s2**5*s3*s5**2*s6
- 24*s1*s2**5*s4**2*s5*s6 + 6*s1*s2**5*s4*s5**3 - 66*s1*s2**4*s3**3*s6**2 - 50*s1*s2**4*s3**2*s4*s5*s6
+ 19*s1*s2**4*s3**2*s5**3 + 8*s1*s2**4*s3*s4**3*s6 - 2*s1*s2**4*s3*s4**2*s5**2 - 552*s1*s2**4*s3*s6**3
- 2560*s1*s2**4*s4*s5*s6**2 + 914*s1*s2**4*s5**3*s6 + 15*s1*s2**3*s3**4*s5*s6 - 2*s1*s2**3*s3**3*s4**2*s6
- s1*s2**3*s3**3*s4*s5**2 + 1602*s1*s2**3*s3**2*s5*s6**2 - 608*s1*s2**3*s3*s4**2*s6**2
- 310*s1*s2**3*s3*s4*s5**2*s6 + 105*s1*s2**3*s3*s5**4 + 600*s1*s2**3*s4**3*s5*s6 -
234*s1*s2**3*s4**2*s5**3 + 31368*s1*s2**3*s5*s6**3 + 756*s1*s2**2*s3**3*s4*s6**2 -
342*s1*s2**2*s3**3*s5**2*s6 + 216*s1*s2**2*s3**2*s4**2*s5*s6 - 231*s1*s2**2*s3**2*s4*s5**3
- 192*s1*s2**2*s3*s4**4*s6 + 194*s1*s2**2*s3*s4**3*s5**2 - 39096*s1*s2**2*s3*s4*s6**3
+ 3618*s1*s2**2*s3*s5**2*s6**2 - 24*s1*s2**2*s4**5*s5 + 9408*s1*s2**2*s4**2*s5*s6**2
- 9540*s1*s2**2*s4*s5**3*s6 + 2750*s1*s2**2*s5**5 - 162*s1*s2*s3**5*s6**2 - 378*s1*s2*s3**4*s4*s5*s6
+ 117*s1*s2*s3**4*s5**3 + 150*s1*s2*s3**3*s4**3*s6 + 33*s1*s2*s3**3*s4**2*s5**2 +
10044*s1*s2*s3**3*s6**3 - 50*s1*s2*s3**2*s4**4*s5 - 8640*s1*s2*s3**2*s4*s5*s6**2 +
2955*s1*s2*s3**2*s5**3*s6 + 8*s1*s2*s3*s4**6 + 6144*s1*s2*s3*s4**3*s6**2 + 4872*s1*s2*s3*s4**2*s5**2*s6
- 3075*s1*s2*s3*s4*s5**4 + 174960*s1*s2*s3*s6**4 - 2560*s1*s2*s4**4*s5*s6 + 1090*s1*s2*s4**3*s5**3
- 148824*s1*s2*s4*s5*s6**3 + 53550*s1*s2*s5**3*s6**2 + 81*s1*s3**6*s5*s6 - 27*s1*s3**5*s4**2*s6
- 27*s1*s3**5*s4*s5**2 + 15*s1*s3**4*s4**3*s5 + 2430*s1*s3**4*s5*s6**2 - 2*s1*s3**3*s4**5
- 2052*s1*s3**3*s4**2*s6**2 - 3078*s1*s3**3*s4*s5**2*s6 + 525*s1*s3**3*s5**4 + 1602*s1*s3**2*s4**3*s5*s6
+ 305*s1*s3**2*s4**2*s5**3 + 18144*s1*s3**2*s5*s6**3 - 104*s1*s3*s4**5*s6 - 318*s1*s3*s4**4*s5**2
- 33696*s1*s3*s4**2*s6**3 - 3780*s1*s3*s4*s5**2*s6**2 + 375*s1*s3*s5**4*s6 + 48*s1*s4**6*s5
+ 31368*s1*s4**3*s5*s6**2 - 16650*s1*s4**2*s5**3*s6 + 2500*s1*s4*s5**5 + 77760*s1*s5*s6**4
- 32*s2**7*s4*s6**2 + 8*s2**7*s5**2*s6 + 8*s2**6*s3**2*s6**2 + 8*s2**6*s3*s4*s5*s6
- 2*s2**6*s3*s5**3 + 96*s2**6*s6**3 - 2*s2**5*s3**3*s5*s6 - 104*s2**5*s3*s5*s6**2
+ 416*s2**5*s4**2*s6**2 - 58*s2**5*s5**4 - 312*s2**4*s3**2*s4*s6**2 + 32*s2**4*s3**2*s5**2*s6
- 192*s2**4*s3*s4**2*s5*s6 + 112*s2**4*s3*s4*s5**3 - 8*s2**4*s4**3*s5**2 + 4224*s2**4*s4*s6**3
- 7192*s2**4*s5**2*s6**2 + 54*s2**3*s3**4*s6**2 + 150*s2**3*s3**3*s4*s5*s6 - 31*s2**3*s3**3*s5**3
- 32*s2**3*s3**2*s4**2*s5**2 - 864*s2**3*s3**2*s6**3 + 8*s2**3*s3*s4**4*s5 + 6144*s2**3*s3*s4*s5*s6**2
+ 2770*s2**3*s3*s5**3*s6 - 4032*s2**3*s4**3*s6**2 + 2144*s2**3*s4**2*s5**2*s6 - 850*s2**3*s4*s5**4
- 16416*s2**3*s6**4 - 27*s2**2*s3**5*s5*s6 + 9*s2**2*s3**4*s4*s5**2 - 2*s2**2*s3**3*s4**3*s5
- 2052*s2**2*s3**3*s5*s6**2 + 2376*s2**2*s3**2*s4**2*s6**2 - 4014*s2**2*s3**2*s4*s5**2*s6
- 325*s2**2*s3**2*s5**4 - 608*s2**2*s3*s4**3*s5*s6 + 1340*s2**2*s3*s4**2*s5**3 - 33696*s2**2*s3*s5*s6**3
+ 416*s2**2*s4**5*s6 - 352*s2**2*s4**4*s5**2 - 6048*s2**2*s4**2*s6**3 + 40320*s2**2*s4*s5**2*s6**2
- 12750*s2**2*s5**4*s6 - 324*s2*s3**4*s4*s6**2 + 918*s2*s3**4*s5**2*s6 + 756*s2*s3**3*s4**2*s5*s6
+ 180*s2*s3**3*s4*s5**3 - 312*s2*s3**2*s4**4*s6 - 574*s2*s3**2*s4**3*s5**2 + 43416*s2*s3**2*s4*s6**3
- 4590*s2*s3**2*s5**2*s6**2 + 264*s2*s3*s4**5*s5 - 39096*s2*s3*s4**2*s5*s6**2 + 10950*s2*s3*s4*s5**3*s6
+ 625*s2*s3*s5**5 - 32*s2*s4**7 + 4224*s2*s4**4*s6**2 + 560*s2*s4**3*s5**2*s6 - 750*s2*s4**2*s5**4
+ 85536*s2*s4*s6**4 - 48600*s2*s5**2*s6**3 - 162*s3**5*s4*s5*s6 - 108*s3**5*s5**3
+ 54*s3**4*s4**3*s6 + 162*s3**4*s4**2*s5**2 - 11664*s3**4*s6**3 - 66*s3**3*s4**4*s5
+ 10044*s3**3*s4*s5*s6**2 - 2025*s3**3*s5**3*s6 + 8*s3**2*s4**6 - 864*s3**2*s4**3*s6**2
+ 270*s3**2*s4**2*s5**2*s6 - 375*s3**2*s4*s5**4 - 163296*s3**2*s6**4 - 552*s3*s4**4*s5*s6
+ 350*s3*s4**3*s5**3 + 174960*s3*s4*s5*s6**3 - 40500*s3*s5**3*s6**2 + 96*s4**6*s6
- 56*s4**5*s5**2 - 16416*s4**3*s6**3 - 37800*s4**2*s5**2*s6**2 + 22500*s4*s5**4*s6
- 3125*s5**6 - 93312*s6**5),
        lambda s1, s2, s3, s4, s5, s6: (-9375*s1**7*s5*s6**4 + 3125*s1**6*s2*s4*s6**4 + 7500*s1**6*s2*s5**2*s6**3 + 3125*s1**6*s3**2*s6**4
- 1250*s1**6*s3*s4*s5*s6**3 - 2000*s1**6*s3*s5**3*s6**2 + 3250*s1**6*s4**2*s5**2*s6**2
- 1600*s1**6*s4*s5**4*s6 + 256*s1**6*s5**6 + 40625*s1**6*s6**5 - 3125*s1**5*s2**2*s3*s6**4
- 3500*s1**5*s2**2*s4*s5*s6**3 - 1450*s1**5*s2**2*s5**3*s6**2 - 1750*s1**5*s2*s3**2*s5*s6**3
+ 625*s1**5*s2*s3*s4**2*s6**3 - 850*s1**5*s2*s3*s4*s5**2*s6**2 + 1760*s1**5*s2*s3*s5**4*s6
- 2050*s1**5*s2*s4**3*s5*s6**2 + 780*s1**5*s2*s4**2*s5**3*s6 - 192*s1**5*s2*s4*s5**5
+ 35000*s1**5*s2*s5*s6**4 + 1200*s1**5*s3**3*s5**2*s6**2 - 725*s1**5*s3**2*s4**2*s5*s6**2
- 160*s1**5*s3**2*s4*s5**3*s6 - 192*s1**5*s3**2*s5**5 - 125*s1**5*s3*s4**4*s6**2 +
590*s1**5*s3*s4**3*s5**2*s6 - 16*s1**5*s3*s4**2*s5**4 - 20625*s1**5*s3*s4*s6**4 +
17250*s1**5*s3*s5**2*s6**3 - 124*s1**5*s4**5*s5*s6 + 17*s1**5*s4**4*s5**3 - 20250*s1**5*s4**2*s5*s6**3
+ 1900*s1**5*s4*s5**3*s6**2 + 1344*s1**5*s5**5*s6 + 625*s1**4*s2**4*s6**4 + 2300*s1**4*s2**3*s3*s5*s6**3
+ 250*s1**4*s2**3*s4**2*s6**3 + 1470*s1**4*s2**3*s4*s5**2*s6**2 - 276*s1**4*s2**3*s5**4*s6
- 125*s1**4*s2**2*s3**2*s4*s6**3 - 610*s1**4*s2**2*s3**2*s5**2*s6**2 + 1995*s1**4*s2**2*s3*s4**2*s5*s6**2
- 1174*s1**4*s2**2*s3*s4*s5**3*s6 - 16*s1**4*s2**2*s3*s5**5 + 375*s1**4*s2**2*s4**4*s6**2
- 172*s1**4*s2**2*s4**3*s5**2*s6 + 82*s1**4*s2**2*s4**2*s5**4 - 7750*s1**4*s2**2*s4*s6**4
- 46650*s1**4*s2**2*s5**2*s6**3 + 15*s1**4*s2*s3**3*s4*s5*s6**2 - 384*s1**4*s2*s3**3*s5**3*s6
+ 525*s1**4*s2*s3**2*s4**3*s6**2 - 528*s1**4*s2*s3**2*s4**2*s5**2*s6 + 384*s1**4*s2*s3**2*s4*s5**4
- 10125*s1**4*s2*s3**2*s6**4 - 29*s1**4*s2*s3*s4**4*s5*s6 - 118*s1**4*s2*s3*s4**3*s5**3
+ 36700*s1**4*s2*s3*s4*s5*s6**3 + 2410*s1**4*s2*s3*s5**3*s6**2 + 38*s1**4*s2*s4**6*s6
+ 5*s1**4*s2*s4**5*s5**2 + 5550*s1**4*s2*s4**3*s6**3 - 10040*s1**4*s2*s4**2*s5**2*s6**2
+ 5800*s1**4*s2*s4*s5**4*s6 - 1600*s1**4*s2*s5**6 - 292500*s1**4*s2*s6**5 - 99*s1**4*s3**5*s5*s6**2
- 150*s1**4*s3**4*s4**2*s6**2 + 196*s1**4*s3**4*s4*s5**2*s6 + 48*s1**4*s3**4*s5**4
+ 12*s1**4*s3**3*s4**3*s5*s6 - 128*s1**4*s3**3*s4**2*s5**3 - 6525*s1**4*s3**3*s5*s6**3
- 12*s1**4*s3**2*s4**5*s6 + 65*s1**4*s3**2*s4**4*s5**2 + 225*s1**4*s3**2*s4**2*s6**3
+ 80*s1**4*s3**2*s4*s5**2*s6**2 - 13*s1**4*s3*s4**6*s5 + 5145*s1**4*s3*s4**3*s5*s6**2
- 6746*s1**4*s3*s4**2*s5**3*s6 + 1760*s1**4*s3*s4*s5**5 - 103500*s1**4*s3*s5*s6**4
+ s1**4*s4**8 + 954*s1**4*s4**5*s6**2 + 449*s1**4*s4**4*s5**2*s6 - 276*s1**4*s4**3*s5**4
+ 70125*s1**4*s4**2*s6**4 + 58900*s1**4*s4*s5**2*s6**3 - 23310*s1**4*s5**4*s6**2 -
468*s1**3*s2**5*s5*s6**3 - 200*s1**3*s2**4*s3*s4*s6**3 - 294*s1**3*s2**4*s3*s5**2*s6**2
- 676*s1**3*s2**4*s4**2*s5*s6**2 + 180*s1**3*s2**4*s4*s5**3*s6 + 17*s1**3*s2**4*s5**5
+ 50*s1**3*s2**3*s3**3*s6**3 - 397*s1**3*s2**3*s3**2*s4*s5*s6**2 + 514*s1**3*s2**3*s3**2*s5**3*s6
- 700*s1**3*s2**3*s3*s4**3*s6**2 + 447*s1**3*s2**3*s3*s4**2*s5**2*s6 - 118*s1**3*s2**3*s3*s4*s5**4
+ 11700*s1**3*s2**3*s3*s6**4 - 12*s1**3*s2**3*s4**4*s5*s6 + 6*s1**3*s2**3*s4**3*s5**3
+ 10360*s1**3*s2**3*s4*s5*s6**3 + 11404*s1**3*s2**3*s5**3*s6**2 + 141*s1**3*s2**2*s3**4*s5*s6**2
- 185*s1**3*s2**2*s3**3*s4**2*s6**2 + 168*s1**3*s2**2*s3**3*s4*s5**2*s6 - 128*s1**3*s2**2*s3**3*s5**4
+ 93*s1**3*s2**2*s3**2*s4**3*s5*s6 + 19*s1**3*s2**2*s3**2*s4**2*s5**3 + 5895*s1**3*s2**2*s3**2*s5*s6**3
- 36*s1**3*s2**2*s3*s4**5*s6 + 5*s1**3*s2**2*s3*s4**4*s5**2 - 12020*s1**3*s2**2*s3*s4**2*s6**3
- 5698*s1**3*s2**2*s3*s4*s5**2*s6**2 - 6746*s1**3*s2**2*s3*s5**4*s6 + 5064*s1**3*s2**2*s4**3*s5*s6**2
- 762*s1**3*s2**2*s4**2*s5**3*s6 + 780*s1**3*s2**2*s4*s5**5 + 93900*s1**3*s2**2*s5*s6**4
+ 198*s1**3*s2*s3**5*s4*s6**2 - 78*s1**3*s2*s3**5*s5**2*s6 - 95*s1**3*s2*s3**4*s4**2*s5*s6
+ 44*s1**3*s2*s3**4*s4*s5**3 + 25*s1**3*s2*s3**3*s4**4*s6 - 15*s1**3*s2*s3**3*s4**3*s5**2
+ 1935*s1**3*s2*s3**3*s4*s6**3 - 2808*s1**3*s2*s3**3*s5**2*s6**2 + s1**3*s2*s3**2*s4**5*s5
- 4844*s1**3*s2*s3**2*s4**2*s5*s6**2 + 8996*s1**3*s2*s3**2*s4*s5**3*s6 - 160*s1**3*s2*s3**2*s5**5
- 3616*s1**3*s2*s3*s4**4*s6**2 + 500*s1**3*s2*s3*s4**3*s5**2*s6 - 1174*s1**3*s2*s3*s4**2*s5**4
+ 72900*s1**3*s2*s3*s4*s6**4 - 55665*s1**3*s2*s3*s5**2*s6**3 + 128*s1**3*s2*s4**5*s5*s6
+ 180*s1**3*s2*s4**4*s5**3 + 16240*s1**3*s2*s4**2*s5*s6**3 - 9330*s1**3*s2*s4*s5**3*s6**2
+ 1900*s1**3*s2*s5**5*s6 - 27*s1**3*s3**7*s6**2 + 18*s1**3*s3**6*s4*s5*s6 - 4*s1**3*s3**6*s5**3
- 4*s1**3*s3**5*s4**3*s6 + s1**3*s3**5*s4**2*s5**2 + 54*s1**3*s3**5*s6**3 + 1143*s1**3*s3**4*s4*s5*s6**2
- 820*s1**3*s3**4*s5**3*s6 + 923*s1**3*s3**3*s4**3*s6**2 + 57*s1**3*s3**3*s4**2*s5**2*s6
- 384*s1**3*s3**3*s4*s5**4 + 29700*s1**3*s3**3*s6**4 - 547*s1**3*s3**2*s4**4*s5*s6
+ 514*s1**3*s3**2*s4**3*s5**3 - 10305*s1**3*s3**2*s4*s5*s6**3 - 7405*s1**3*s3**2*s5**3*s6**2
+ 108*s1**3*s3*s4**6*s6 - 148*s1**3*s3*s4**5*s5**2 - 11360*s1**3*s3*s4**3*s6**3 +
22209*s1**3*s3*s4**2*s5**2*s6**2 + 2410*s1**3*s3*s4*s5**4*s6 - 2000*s1**3*s3*s5**6
+ 432000*s1**3*s3*s6**5 + 12*s1**3*s4**7*s5 - 22624*s1**3*s4**4*s5*s6**2 + 11404*s1**3*s4**3*s5**3*s6
- 1450*s1**3*s4**2*s5**5 - 242100*s1**3*s4*s5*s6**4 + 58430*s1**3*s5**3*s6**3 + 56*s1**2*s2**6*s4*s6**3
+ 86*s1**2*s2**6*s5**2*s6**2 - 14*s1**2*s2**5*s3**2*s6**3 + 304*s1**2*s2**5*s3*s4*s5*s6**2
- 148*s1**2*s2**5*s3*s5**3*s6 + 152*s1**2*s2**5*s4**3*s6**2 - 54*s1**2*s2**5*s4**2*s5**2*s6
+ 5*s1**2*s2**5*s4*s5**4 - 2472*s1**2*s2**5*s6**4 - 76*s1**2*s2**4*s3**3*s5*s6**2
+ 370*s1**2*s2**4*s3**2*s4**2*s6**2 - 287*s1**2*s2**4*s3**2*s4*s5**2*s6 + 65*s1**2*s2**4*s3**2*s5**4
- 28*s1**2*s2**4*s3*s4**3*s5*s6 + 5*s1**2*s2**4*s3*s4**2*s5**3 - 8092*s1**2*s2**4*s3*s5*s6**3
+ 8*s1**2*s2**4*s4**5*s6 - 2*s1**2*s2**4*s4**4*s5**2 + 1096*s1**2*s2**4*s4**2*s6**3
- 5144*s1**2*s2**4*s4*s5**2*s6**2 + 449*s1**2*s2**4*s5**4*s6 - 210*s1**2*s2**3*s3**4*s4*s6**2
+ 76*s1**2*s2**3*s3**4*s5**2*s6 + 43*s1**2*s2**3*s3**3*s4**2*s5*s6 - 15*s1**2*s2**3*s3**3*s4*s5**3
- 6*s1**2*s2**3*s3**2*s4**4*s6 + 2*s1**2*s2**3*s3**2*s4**3*s5**2 + 1962*s1**2*s2**3*s3**2*s4*s6**3
+ 3181*s1**2*s2**3*s3**2*s5**2*s6**2 + 1684*s1**2*s2**3*s3*s4**2*s5*s6**2 + 500*s1**2*s2**3*s3*s4*s5**3*s6
+ 590*s1**2*s2**3*s3*s5**5 - 168*s1**2*s2**3*s4**4*s6**2 - 494*s1**2*s2**3*s4**3*s5**2*s6
- 172*s1**2*s2**3*s4**2*s5**4 - 22080*s1**2*s2**3*s4*s6**4 + 58894*s1**2*s2**3*s5**2*s6**3
+ 27*s1**2*s2**2*s3**6*s6**2 - 9*s1**2*s2**2*s3**5*s4*s5*s6 + s1**2*s2**2*s3**5*s5**3
+ s1**2*s2**2*s3**4*s4**3*s6 - 486*s1**2*s2**2*s3**4*s6**3 + 1071*s1**2*s2**2*s3**3*s4*s5*s6**2
+ 57*s1**2*s2**2*s3**3*s5**3*s6 + 2262*s1**2*s2**2*s3**2*s4**3*s6**2 - 2742*s1**2*s2**2*s3**2*s4**2*s5**2*s6
- 528*s1**2*s2**2*s3**2*s4*s5**4 - 29160*s1**2*s2**2*s3**2*s6**4 + 772*s1**2*s2**2*s3*s4**4*s5*s6
+ 447*s1**2*s2**2*s3*s4**3*s5**3 - 96732*s1**2*s2**2*s3*s4*s5*s6**3 + 22209*s1**2*s2**2*s3*s5**3*s6**2
- 160*s1**2*s2**2*s4**6*s6 - 54*s1**2*s2**2*s4**5*s5**2 - 7992*s1**2*s2**2*s4**3*s6**3
+ 8634*s1**2*s2**2*s4**2*s5**2*s6**2 - 10040*s1**2*s2**2*s4*s5**4*s6 + 3250*s1**2*s2**2*s5**6
+ 529200*s1**2*s2**2*s6**5 - 351*s1**2*s2*s3**5*s5*s6**2 - 1215*s1**2*s2*s3**4*s4**2*s6**2
- 360*s1**2*s2*s3**4*s4*s5**2*s6 + 196*s1**2*s2*s3**4*s5**4 + 741*s1**2*s2*s3**3*s4**3*s5*s6
+ 168*s1**2*s2*s3**3*s4**2*s5**3 + 11718*s1**2*s2*s3**3*s5*s6**3 - 106*s1**2*s2*s3**2*s4**5*s6
- 287*s1**2*s2*s3**2*s4**4*s5**2 + 22572*s1**2*s2*s3**2*s4**2*s6**3 - 8892*s1**2*s2*s3**2*s4*s5**2*s6**2
+ 80*s1**2*s2*s3**2*s5**4*s6 + 88*s1**2*s2*s3*s4**6*s5 + 22144*s1**2*s2*s3*s4**3*s5*s6**2
- 5698*s1**2*s2*s3*s4**2*s5**3*s6 - 850*s1**2*s2*s3*s4*s5**5 + 169560*s1**2*s2*s3*s5*s6**4
- 8*s1**2*s2*s4**8 + 3032*s1**2*s2*s4**5*s6**2 - 5144*s1**2*s2*s4**4*s5**2*s6 + 1470*s1**2*s2*s4**3*s5**4
- 249480*s1**2*s2*s4**2*s6**4 - 105390*s1**2*s2*s4*s5**2*s6**3 + 58900*s1**2*s2*s5**4*s6**2
+ 162*s1**2*s3**6*s4*s6**2 + 216*s1**2*s3**6*s5**2*s6 - 216*s1**2*s3**5*s4**2*s5*s6
- 78*s1**2*s3**5*s4*s5**3 + 36*s1**2*s3**4*s4**4*s6 + 76*s1**2*s3**4*s4**3*s5**2 -
3564*s1**2*s3**4*s4*s6**3 + 8802*s1**2*s3**4*s5**2*s6**2 - 22*s1**2*s3**3*s4**5*s5
- 11475*s1**2*s3**3*s4**2*s5*s6**2 - 2808*s1**2*s3**3*s4*s5**3*s6 + 1200*s1**2*s3**3*s5**5
+ 2*s1**2*s3**2*s4**7 + 222*s1**2*s3**2*s4**4*s6**2 + 3181*s1**2*s3**2*s4**3*s5**2*s6
- 610*s1**2*s3**2*s4**2*s5**4 - 165240*s1**2*s3**2*s4*s6**4 + 118260*s1**2*s3**2*s5**2*s6**3
+ 572*s1**2*s3*s4**5*s5*s6 - 294*s1**2*s3*s4**4*s5**3 - 32616*s1**2*s3*s4**2*s5*s6**3
- 55665*s1**2*s3*s4*s5**3*s6**2 + 17250*s1**2*s3*s5**5*s6 - 232*s1**2*s4**7*s6 + 86*s1**2*s4**6*s5**2
+ 48408*s1**2*s4**4*s6**3 + 58894*s1**2*s4**3*s5**2*s6**2 - 46650*s1**2*s4**2*s5**4*s6
+ 7500*s1**2*s4*s5**6 - 129600*s1**2*s4*s6**5 + 41040*s1**2*s5**2*s6**4 - 48*s1*s2**7*s4*s5*s6**2
+ 12*s1*s2**7*s5**3*s6 + 12*s1*s2**6*s3**2*s5*s6**2 - 144*s1*s2**6*s3*s4**2*s6**2
+ 88*s1*s2**6*s3*s4*s5**2*s6 - 13*s1*s2**6*s3*s5**4 + 1680*s1*s2**6*s5*s6**3 + 72*s1*s2**5*s3**3*s4*s6**2
- 22*s1*s2**5*s3**3*s5**2*s6 - 4*s1*s2**5*s3**2*s4**2*s5*s6 + s1*s2**5*s3**2*s4*s5**3
- 144*s1*s2**5*s3*s4*s6**3 + 572*s1*s2**5*s3*s5**2*s6**2 + 736*s1*s2**5*s4**2*s5*s6**2
+ 128*s1*s2**5*s4*s5**3*s6 - 124*s1*s2**5*s5**5 - 9*s1*s2**4*s3**5*s6**2 + s1*s2**4*s3**4*s4*s5*s6
+ 36*s1*s2**4*s3**3*s6**3 - 2028*s1*s2**4*s3**2*s4*s5*s6**2 - 547*s1*s2**4*s3**2*s5**3*s6
- 480*s1*s2**4*s3*s4**3*s6**2 + 772*s1*s2**4*s3*s4**2*s5**2*s6 - 29*s1*s2**4*s3*s4*s5**4
+ 6336*s1*s2**4*s3*s6**4 - 12*s1*s2**4*s4**3*s5**3 + 4368*s1*s2**4*s4*s5*s6**3 - 22624*s1*s2**4*s5**3*s6**2
+ 441*s1*s2**3*s3**4*s5*s6**2 + 336*s1*s2**3*s3**3*s4**2*s6**2 + 741*s1*s2**3*s3**3*s4*s5**2*s6
+ 12*s1*s2**3*s3**3*s5**4 - 868*s1*s2**3*s3**2*s4**3*s5*s6 + 93*s1*s2**3*s3**2*s4**2*s5**3
+ 11016*s1*s2**3*s3**2*s5*s6**3 + 176*s1*s2**3*s3*s4**5*s6 - 28*s1*s2**3*s3*s4**4*s5**2
+ 14784*s1*s2**3*s3*s4**2*s6**3 + 22144*s1*s2**3*s3*s4*s5**2*s6**2 + 5145*s1*s2**3*s3*s5**4*s6
- 11344*s1*s2**3*s4**3*s5*s6**2 + 5064*s1*s2**3*s4**2*s5**3*s6 - 2050*s1*s2**3*s4*s5**5
- 346896*s1*s2**3*s5*s6**4 - 54*s1*s2**2*s3**5*s4*s6**2 - 216*s1*s2**2*s3**5*s5**2*s6
+ 324*s1*s2**2*s3**4*s4**2*s5*s6 - 95*s1*s2**2*s3**4*s4*s5**3 - 80*s1*s2**2*s3**3*s4**4*s6
+ 43*s1*s2**2*s3**3*s4**3*s5**2 - 12204*s1*s2**2*s3**3*s4*s6**3 - 11475*s1*s2**2*s3**3*s5**2*s6**2
- 4*s1*s2**2*s3**2*s4**5*s5 - 3888*s1*s2**2*s3**2*s4**2*s5*s6**2 - 4844*s1*s2**2*s3**2*s4*s5**3*s6
- 725*s1*s2**2*s3**2*s5**5 - 1312*s1*s2**2*s3*s4**4*s6**2 + 1684*s1*s2**2*s3*s4**3*s5**2*s6
+ 1995*s1*s2**2*s3*s4**2*s5**4 + 139104*s1*s2**2*s3*s4*s6**4 - 32616*s1*s2**2*s3*s5**2*s6**3
+ 736*s1*s2**2*s4**5*s5*s6 - 676*s1*s2**2*s4**4*s5**3 + 131040*s1*s2**2*s4**2*s5*s6**3
+ 16240*s1*s2**2*s4*s5**3*s6**2 - 20250*s1*s2**2*s5**5*s6 - 27*s1*s2*s3**6*s4*s5*s6
+ 18*s1*s2*s3**6*s5**3 + 9*s1*s2*s3**5*s4**3*s6 - 9*s1*s2*s3**5*s4**2*s5**2 + 1944*s1*s2*s3**5*s6**3
+ s1*s2*s3**4*s4**4*s5 + 6156*s1*s2*s3**4*s4*s5*s6**2 + 1143*s1*s2*s3**4*s5**3*s6
+ 324*s1*s2*s3**3*s4**3*s6**2 + 1071*s1*s2*s3**3*s4**2*s5**2*s6 + 15*s1*s2*s3**3*s4*s5**4
- 7776*s1*s2*s3**3*s6**4 - 2028*s1*s2*s3**2*s4**4*s5*s6 - 397*s1*s2*s3**2*s4**3*s5**3
+ 112860*s1*s2*s3**2*s4*s5*s6**3 - 10305*s1*s2*s3**2*s5**3*s6**2 + 336*s1*s2*s3*s4**6*s6
+ 304*s1*s2*s3*s4**5*s5**2 - 68976*s1*s2*s3*s4**3*s6**3 - 96732*s1*s2*s3*s4**2*s5**2*s6**2
+ 36700*s1*s2*s3*s4*s5**4*s6 - 1250*s1*s2*s3*s5**6 - 1477440*s1*s2*s3*s6**5 - 48*s1*s2*s4**7*s5
+ 4368*s1*s2*s4**4*s5*s6**2 + 10360*s1*s2*s4**3*s5**3*s6 - 3500*s1*s2*s4**2*s5**5
+ 935280*s1*s2*s4*s5*s6**4 - 242100*s1*s2*s5**3*s6**3 - 972*s1*s3**6*s5*s6**2 - 351*s1*s3**5*s4*s5**2*s6
- 99*s1*s3**5*s5**4 + 441*s1*s3**4*s4**3*s5*s6 + 141*s1*s3**4*s4**2*s5**3 - 36936*s1*s3**4*s5*s6**3
- 84*s1*s3**3*s4**5*s6 - 76*s1*s3**3*s4**4*s5**2 + 17496*s1*s3**3*s4**2*s6**3 + 11718*s1*s3**3*s4*s5**2*s6**2
- 6525*s1*s3**3*s5**4*s6 + 12*s1*s3**2*s4**6*s5 + 11016*s1*s3**2*s4**3*s5*s6**2 +
5895*s1*s3**2*s4**2*s5**3*s6 - 1750*s1*s3**2*s4*s5**5 - 252720*s1*s3**2*s5*s6**4 -
2544*s1*s3*s4**5*s6**2 - 8092*s1*s3*s4**4*s5**2*s6 + 2300*s1*s3*s4**3*s5**4 + 536544*s1*s3*s4**2*s6**4
+ 169560*s1*s3*s4*s5**2*s6**3 - 103500*s1*s3*s5**4*s6**2 + 1680*s1*s4**6*s5*s6 - 468*s1*s4**5*s5**3
- 346896*s1*s4**3*s5*s6**3 + 93900*s1*s4**2*s5**3*s6**2 + 35000*s1*s4*s5**5*s6 - 9375*s1*s5**7
+ 108864*s1*s5*s6**5 + 16*s2**8*s4**2*s6**2 - 8*s2**8*s4*s5**2*s6 + s2**8*s5**4 -
8*s2**7*s3**2*s4*s6**2 + 2*s2**7*s3**2*s5**2*s6 - 96*s2**7*s4*s6**3 - 232*s2**7*s5**2*s6**2
+ s2**6*s3**4*s6**2 + 24*s2**6*s3**2*s6**3 + 336*s2**6*s3*s4*s5*s6**2 + 108*s2**6*s3*s5**3*s6
- 32*s2**6*s4**3*s6**2 - 160*s2**6*s4**2*s5**2*s6 + 38*s2**6*s4*s5**4 + 144*s2**6*s6**4
- 84*s2**5*s3**3*s5*s6**2 + 8*s2**5*s3**2*s4**2*s6**2 - 106*s2**5*s3**2*s4*s5**2*s6
- 12*s2**5*s3**2*s5**4 + 176*s2**5*s3*s4**3*s5*s6 - 36*s2**5*s3*s4**2*s5**3 - 2544*s2**5*s3*s5*s6**3
- 32*s2**5*s4**5*s6 + 8*s2**5*s4**4*s5**2 - 3072*s2**5*s4**2*s6**3 + 3032*s2**5*s4*s5**2*s6**2
+ 954*s2**5*s5**4*s6 + 36*s2**4*s3**4*s5**2*s6 - 80*s2**4*s3**3*s4**2*s5*s6 + 25*s2**4*s3**3*s4*s5**3
+ 16*s2**4*s3**2*s4**4*s6 - 6*s2**4*s3**2*s4**3*s5**2 + 2520*s2**4*s3**2*s4*s6**3
+ 222*s2**4*s3**2*s5**2*s6**2 - 1312*s2**4*s3*s4**2*s5*s6**2 - 3616*s2**4*s3*s4*s5**3*s6
- 125*s2**4*s3*s5**5 + 1296*s2**4*s4**4*s6**2 - 168*s2**4*s4**3*s5**2*s6 + 375*s2**4*s4**2*s5**4
+ 19296*s2**4*s4*s6**4 + 48408*s2**4*s5**2*s6**3 + 9*s2**3*s3**5*s4*s5*s6 - 4*s2**3*s3**5*s5**3
- 2*s2**3*s3**4*s4**3*s6 + s2**3*s3**4*s4**2*s5**2 - 432*s2**3*s3**4*s6**3 + 324*s2**3*s3**3*s4*s5*s6**2
+ 923*s2**3*s3**3*s5**3*s6 - 752*s2**3*s3**2*s4**3*s6**2 + 2262*s2**3*s3**2*s4**2*s5**2*s6
+ 525*s2**3*s3**2*s4*s5**4 - 9936*s2**3*s3**2*s6**4 - 480*s2**3*s3*s4**4*s5*s6 - 700*s2**3*s3*s4**3*s5**3
- 68976*s2**3*s3*s4*s5*s6**3 - 11360*s2**3*s3*s5**3*s6**2 - 32*s2**3*s4**6*s6 + 152*s2**3*s4**5*s5**2
+ 6912*s2**3*s4**3*s6**3 - 7992*s2**3*s4**2*s5**2*s6**2 + 5550*s2**3*s4*s5**4*s6 -
29376*s2**3*s6**5 + 108*s2**2*s3**4*s4**2*s6**2 - 1215*s2**2*s3**4*s4*s5**2*s6 - 150*s2**2*s3**4*s5**4
+ 336*s2**2*s3**3*s4**3*s5*s6 - 185*s2**2*s3**3*s4**2*s5**3 + 17496*s2**2*s3**3*s5*s6**3
+ 8*s2**2*s3**2*s4**5*s6 + 370*s2**2*s3**2*s4**4*s5**2 - 864*s2**2*s3**2*s4**2*s6**3
+ 22572*s2**2*s3**2*s4*s5**2*s6**2 + 225*s2**2*s3**2*s5**4*s6 - 144*s2**2*s3*s4**6*s5
+ 14784*s2**2*s3*s4**3*s5*s6**2 - 12020*s2**2*s3*s4**2*s5**3*s6 + 625*s2**2*s3*s4*s5**5
+ 536544*s2**2*s3*s5*s6**4 + 16*s2**2*s4**8 - 3072*s2**2*s4**5*s6**2 + 1096*s2**2*s4**4*s5**2*s6
+ 250*s2**2*s4**3*s5**4 - 93744*s2**2*s4**2*s6**4 - 249480*s2**2*s4*s5**2*s6**3 +
70125*s2**2*s5**4*s6**2 + 162*s2*s3**6*s5**2*s6 - 54*s2*s3**5*s4**2*s5*s6 + 198*s2*s3**5*s4*s5**3
- 210*s2*s3**4*s4**3*s5**2 - 3564*s2*s3**4*s5**2*s6**2 + 72*s2*s3**3*s4**5*s5 - 12204*s2*s3**3*s4**2*s5*s6**2
+ 1935*s2*s3**3*s4*s5**3*s6 - 8*s2*s3**2*s4**7 + 2520*s2*s3**2*s4**4*s6**2 + 1962*s2*s3**2*s4**3*s5**2*s6
- 125*s2*s3**2*s4**2*s5**4 - 178848*s2*s3**2*s4*s6**4 - 165240*s2*s3**2*s5**2*s6**3
- 144*s2*s3*s4**5*s5*s6 - 200*s2*s3*s4**4*s5**3 + 139104*s2*s3*s4**2*s5*s6**3 + 72900*s2*s3*s4*s5**3*s6**2
- 20625*s2*s3*s5**5*s6 - 96*s2*s4**7*s6 + 56*s2*s4**6*s5**2 + 19296*s2*s4**4*s6**3
- 22080*s2*s4**3*s5**2*s6**2 - 7750*s2*s4**2*s5**4*s6 + 3125*s2*s4*s5**6 + 248832*s2*s4*s6**5
- 129600*s2*s5**2*s6**4 - 27*s3**7*s5**3 + 27*s3**6*s4**2*s5**2 - 9*s3**5*s4**4*s5
+ 1944*s3**5*s4*s5*s6**2 + 54*s3**5*s5**3*s6 + s3**4*s4**6 - 432*s3**4*s4**3*s6**2
- 486*s3**4*s4**2*s5**2*s6 + 46656*s3**4*s6**4 + 36*s3**3*s4**4*s5*s6 + 50*s3**3*s4**3*s5**3
- 7776*s3**3*s4*s5*s6**3 + 29700*s3**3*s5**3*s6**2 + 24*s3**2*s4**6*s6 - 14*s3**2*s4**5*s5**2
- 9936*s3**2*s4**3*s6**3 - 29160*s3**2*s4**2*s5**2*s6**2 - 10125*s3**2*s4*s5**4*s6
+ 3125*s3**2*s5**6 + 1026432*s3**2*s6**5 + 6336*s3*s4**4*s5*s6**2 + 11700*s3*s4**3*s5**3*s6
- 3125*s3*s4**2*s5**5 - 1477440*s3*s4*s5*s6**4 + 432000*s3*s5**3*s6**3 + 144*s4**6*s6**2
- 2472*s4**5*s5**2*s6 + 625*s4**4*s5**4 - 29376*s4**3*s6**4 + 529200*s4**2*s5**2*s6**3
- 292500*s4*s5**4*s6**2 + 40625*s5**6*s6 - 186624*s6**6)
    ],
    (6, 2): [
        lambda s1, s2, s3, s4, s5, s6: (-s3),
        lambda s1, s2, s3, s4, s5, s6: (-s1*s5 + s2*s4 - 9*s6),
        lambda s1, s2, s3, s4, s5, s6: (s1*s2*s6 + 2*s1*s3*s5 - s1*s4**2 - s2**2*s5 + 6*s3*s6 + s4*s5),
        lambda s1, s2, s3, s4, s5, s6: (s1**2*s4*s6 - s1**2*s5**2 - 3*s1*s2*s3*s6 + s1*s2*s4*s5 + 9*s1*s5*s6 + s2**3*s6 -
9*s2*s4*s6 + s2*s5**2 + 3*s3**2*s6 - 3*s3*s4*s5 + s4**3 + 27*s6**2),
        lambda s1, s2, s3, s4, s5, s6: (-2*s1**3*s6**2 + 2*s1**2*s2*s5*s6 + 2*s1**2*s3*s4*s6 - s1**2*s3*s5**2 - s1*s2**2*s4*s6
- 3*s1*s2*s6**2 - 16*s1*s3*s5*s6 + 4*s1*s4**2*s6 + 2*s1*s4*s5**2 + 4*s2**2*s5*s6 +
s2*s3*s4*s6 + 2*s2*s3*s5**2 - s2*s4**2*s5 - 9*s3*s6**2 - 3*s4*s5*s6 - 2*s5**3),
        lambda s1, s2, s3, s4, s5, s6: (s1**3*s3*s6**2 - 3*s1**3*s4*s5*s6 + s1**3*s5**3 - s1**2*s2**2*s6**2 + s1**2*s2*s3*s5*s6
- 2*s1**2*s4*s6**2 + 6*s1**2*s5**2*s6 + 16*s1*s2*s3*s6**2 - 3*s1*s2*s5**3 - s1*s3**2*s5*s6
- 2*s1*s3*s4**2*s6 + s1*s3*s4*s5**2 - 30*s1*s5*s6**2 - 4*s2**3*s6**2 - 2*s2**2*s3*s5*s6
+ s2**2*s4**2*s6 + 18*s2*s4*s6**2 - 2*s2*s5**2*s6 - 15*s3**2*s6**2 + 16*s3*s4*s5*s6
+ s3*s5**3 - 4*s4**3*s6 - s4**2*s5**2 - 27*s6**3),
        lambda s1, s2, s3, s4, s5, s6: (s1**4*s5*s6**2 + 2*s1**3*s2*s4*s6**2 - s1**3*s2*s5**2*s6 - s1**3*s3**2*s6**2 + 9*s1**3*s6**3
- 14*s1**2*s2*s5*s6**2 - 11*s1**2*s3*s4*s6**2 + 6*s1**2*s3*s5**2*s6 + 3*s1**2*s4**2*s5*s6
- s1**2*s4*s5**3 + 3*s1*s2**2*s5**2*s6 + 3*s1*s2*s3**2*s6**2 - s1*s2*s3*s4*s5*s6 +
39*s1*s3*s5*s6**2 - 14*s1*s4*s5**2*s6 + s1*s5**4 - 11*s2*s3*s5**2*s6 + 2*s2*s4*s5**3
- 3*s3**3*s6**2 + 3*s3**2*s4*s5*s6 - s3**2*s5**3 + 9*s5**3*s6),
        lambda s1, s2, s3, s4, s5, s6: (-s1**4*s2*s6**3 + s1**4*s3*s5*s6**2 - 4*s1**3*s3*s6**3 + 10*s1**3*s4*s5*s6**2 - 4*s1**3*s5**3*s6
+ 8*s1**2*s2**2*s6**3 - 8*s1**2*s2*s3*s5*s6**2 - 2*s1**2*s2*s4**2*s6**2 + s1**2*s2*s4*s5**2*s6
+ s1**2*s3**2*s4*s6**2 - 6*s1**2*s4*s6**3 - 7*s1**2*s5**2*s6**2 - 24*s1*s2*s3*s6**3
- 4*s1*s2*s4*s5*s6**2 + 10*s1*s2*s5**3*s6 + 8*s1*s3**2*s5*s6**2 + 8*s1*s3*s4**2*s6**2
- 8*s1*s3*s4*s5**2*s6 + s1*s3*s5**4 + 36*s1*s5*s6**3 + 8*s2**2*s3*s5*s6**2 - 2*s2**2*s4*s5**2*s6
- 2*s2*s3**2*s4*s6**2 + s2*s3**2*s5**2*s6 - 6*s2*s5**2*s6**2 + 18*s3**2*s6**3 - 24*s3*s4*s5*s6**2
- 4*s3*s5**3*s6 + 8*s4**2*s5**2*s6 - s4*s5**4),
        lambda s1, s2, s3, s4, s5, s6: (-s1**5*s4*s6**3 - 2*s1**4*s5*s6**3 + 3*s1**3*s2*s5**2*s6**2 + 3*s1**3*s3**2*s6**3
- s1**3*s3*s4*s5*s6**2 - 8*s1**3*s6**4 + 16*s1**2*s2*s5*s6**3 + 8*s1**2*s3*s4*s6**3
- 6*s1**2*s3*s5**2*s6**2 - 8*s1**2*s4**2*s5*s6**2 + 3*s1**2*s4*s5**3*s6 - 8*s1*s2**2*s5**2*s6**2
- 8*s1*s2*s3**2*s6**3 + 8*s1*s2*s3*s4*s5*s6**2 - s1*s2*s3*s5**3*s6 - s1*s3**3*s5*s6**2
- 24*s1*s3*s5*s6**3 + 16*s1*s4*s5**2*s6**2 - 2*s1*s5**4*s6 + 8*s2*s3*s5**2*s6**2 -
s2*s5**5 + 8*s3**3*s6**3 - 8*s3**2*s4*s5*s6**2 + 3*s3**2*s5**3*s6 - 8*s5**3*s6**2),
        lambda s1, s2, s3, s4, s5, s6: (s1**6*s6**4 - 4*s1**4*s2*s6**4 - 2*s1**4*s3*s5*s6**3 + s1**4*s4**2*s6**3 + 8*s1**3*s3*s6**4
- 4*s1**3*s4*s5*s6**3 + 2*s1**3*s5**3*s6**2 + 8*s1**2*s2*s3*s5*s6**3 - 2*s1**2*s2*s4*s5**2*s6**2
- 2*s1**2*s3**2*s4*s6**3 + s1**2*s3**2*s5**2*s6**2 - 4*s1*s2*s5**3*s6**2 - 12*s1*s3**2*s5*s6**3
+ 8*s1*s3*s4*s5**2*s6**2 - 2*s1*s3*s5**4*s6 + s2**2*s5**4*s6 - 2*s2*s3**2*s5**2*s6**2
+ s3**4*s6**3 + 8*s3*s5**3*s6**2 - 4*s4*s5**4*s6 + s5**6)
    ],
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_cell_widths.py ===
# Auto generated by make_terminal_widths.py

CELL_WIDTHS = [
    (0, 0, 0),
    (1, 31, -1),
    (127, 159, -1),
    (173, 173, 0),
    (768, 879, 0),
    (1155, 1161, 0),
    (1425, 1469, 0),
    (1471, 1471, 0),
    (1473, 1474, 0),
    (1476, 1477, 0),
    (1479, 1479, 0),
    (1536, 1541, 0),
    (1552, 1562, 0),
    (1564, 1564, 0),
    (1611, 1631, 0),
    (1648, 1648, 0),
    (1750, 1757, 0),
    (1759, 1764, 0),
    (1767, 1768, 0),
    (1770, 1773, 0),
    (1807, 1807, 0),
    (1809, 1809, 0),
    (1840, 1866, 0),
    (1958, 1968, 0),
    (2027, 2035, 0),
    (2045, 2045, 0),
    (2070, 2073, 0),
    (2075, 2083, 0),
    (2085, 2087, 0),
    (2089, 2093, 0),
    (2137, 2139, 0),
    (2192, 2193, 0),
    (2200, 2207, 0),
    (2250, 2307, 0),
    (2362, 2364, 0),
    (2366, 2383, 0),
    (2385, 2391, 0),
    (2402, 2403, 0),
    (2433, 2435, 0),
    (2492, 2492, 0),
    (2494, 2500, 0),
    (2503, 2504, 0),
    (2507, 2509, 0),
    (2519, 2519, 0),
    (2530, 2531, 0),
    (2558, 2558, 0),
    (2561, 2563, 0),
    (2620, 2620, 0),
    (2622, 2626, 0),
    (2631, 2632, 0),
    (2635, 2637, 0),
    (2641, 2641, 0),
    (2672, 2673, 0),
    (2677, 2677, 0),
    (2689, 2691, 0),
    (2748, 2748, 0),
    (2750, 2757, 0),
    (2759, 2761, 0),
    (2763, 2765, 0),
    (2786, 2787, 0),
    (2810, 2815, 0),
    (2817, 2819, 0),
    (2876, 2876, 0),
    (2878, 2884, 0),
    (2887, 2888, 0),
    (2891, 2893, 0),
    (2901, 2903, 0),
    (2914, 2915, 0),
    (2946, 2946, 0),
    (3006, 3010, 0),
    (3014, 3016, 0),
    (3018, 3021, 0),
    (3031, 3031, 0),
    (3072, 3076, 0),
    (3132, 3132, 0),
    (3134, 3140, 0),
    (3142, 3144, 0),
    (3146, 3149, 0),
    (3157, 3158, 0),
    (3170, 3171, 0),
    (3201, 3203, 0),
    (3260, 3260, 0),
    (3262, 3268, 0),
    (3270, 3272, 0),
    (3274, 3277, 0),
    (3285, 3286, 0),
    (3298, 3299, 0),
    (3315, 3315, 0),
    (3328, 3331, 0),
    (3387, 3388, 0),
    (3390, 3396, 0),
    (3398, 3400, 0),
    (3402, 3405, 0),
    (3415, 3415, 0),
    (3426, 3427, 0),
    (3457, 3459, 0),
    (3530, 3530, 0),
    (3535, 3540, 0),
    (3542, 3542, 0),
    (3544, 3551, 0),
    (3570, 3571, 0),
    (3633, 3633, 0),
    (3636, 3642, 0),
    (3655, 3662, 0),
    (3761, 3761, 0),
    (3764, 3772, 0),
    (3784, 3790, 0),
    (3864, 3865, 0),
    (3893, 3893, 0),
    (3895, 3895, 0),
    (3897, 3897, 0),
    (3902, 3903, 0),
    (3953, 3972, 0),
    (3974, 3975, 0),
    (3981, 3991, 0),
    (3993, 4028, 0),
    (4038, 4038, 0),
    (4139, 4158, 0),
    (4182, 4185, 0),
    (4190, 4192, 0),
    (4194, 4196, 0),
    (4199, 4205, 0),
    (4209, 4212, 0),
    (4226, 4237, 0),
    (4239, 4239, 0),
    (4250, 4253, 0),
    (4352, 4447, 2),
    (4448, 4607, 0),
    (4957, 4959, 0),
    (5906, 5909, 0),
    (5938, 5940, 0),
    (5970, 5971, 0),
    (6002, 6003, 0),
    (6068, 6099, 0),
    (6109, 6109, 0),
    (6155, 6159, 0),
    (6277, 6278, 0),
    (6313, 6313, 0),
    (6432, 6443, 0),
    (6448, 6459, 0),
    (6679, 6683, 0),
    (6741, 6750, 0),
    (6752, 6780, 0),
    (6783, 6783, 0),
    (6832, 6862, 0),
    (6912, 6916, 0),
    (6964, 6980, 0),
    (7019, 7027, 0),
    (7040, 7042, 0),
    (7073, 7085, 0),
    (7142, 7155, 0),
    (7204, 7223, 0),
    (7376, 7378, 0),
    (7380, 7400, 0),
    (7405, 7405, 0),
    (7412, 7412, 0),
    (7415, 7417, 0),
    (7616, 7679, 0),
    (8203, 8207, 0),
    (8232, 8238, 0),
    (8288, 8292, 0),
    (8294, 8303, 0),
    (8400, 8432, 0),
    (8986, 8987, 2),
    (9001, 9002, 2),
    (9193, 9196, 2),
    (9200, 9200, 2),
    (9203, 9203, 2),
    (9725, 9726, 2),
    (9748, 9749, 2),
    (9800, 9811, 2),
    (9855, 9855, 2),
    (9875, 9875, 2),
    (9889, 9889, 2),
    (9898, 9899, 2),
    (9917, 9918, 2),
    (9924, 9925, 2),
    (9934, 9934, 2),
    (9940, 9940, 2),
    (9962, 9962, 2),
    (9970, 9971, 2),
    (9973, 9973, 2),
    (9978, 9978, 2),
    (9981, 9981, 2),
    (9989, 9989, 2),
    (9994, 9995, 2),
    (10024, 10024, 2),
    (10060, 10060, 2),
    (10062, 10062, 2),
    (10067, 10069, 2),
    (10071, 10071, 2),
    (10133, 10135, 2),
    (10160, 10160, 2),
    (10175, 10175, 2),
    (11035, 11036, 2),
    (11088, 11088, 2),
    (11093, 11093, 2),
    (11503, 11505, 0),
    (11647, 11647, 0),
    (11744, 11775, 0),
    (11904, 11929, 2),
    (11931, 12019, 2),
    (12032, 12245, 2),
    (12272, 12329, 2),
    (12330, 12335, 0),
    (12336, 12350, 2),
    (12353, 12438, 2),
    (12441, 12442, 0),
    (12443, 12543, 2),
    (12549, 12591, 2),
    (12593, 12686, 2),
    (12688, 12771, 2),
    (12783, 12830, 2),
    (12832, 12871, 2),
    (12880, 19903, 2),
    (19968, 42124, 2),
    (42128, 42182, 2),
    (42607, 42610, 0),
    (42612, 42621, 0),
    (42654, 42655, 0),
    (42736, 42737, 0),
    (43010, 43010, 0),
    (43014, 43014, 0),
    (43019, 43019, 0),
    (43043, 43047, 0),
    (43052, 43052, 0),
    (43136, 43137, 0),
    (43188, 43205, 0),
    (43232, 43249, 0),
    (43263, 43263, 0),
    (43302, 43309, 0),
    (43335, 43347, 0),
    (43360, 43388, 2),
    (43392, 43395, 0),
    (43443, 43456, 0),
    (43493, 43493, 0),
    (43561, 43574, 0),
    (43587, 43587, 0),
    (43596, 43597, 0),
    (43643, 43645, 0),
    (43696, 43696, 0),
    (43698, 43700, 0),
    (43703, 43704, 0),
    (43710, 43711, 0),
    (43713, 43713, 0),
    (43755, 43759, 0),
    (43765, 43766, 0),
    (44003, 44010, 0),
    (44012, 44013, 0),
    (44032, 55203, 2),
    (55216, 55295, 0),
    (63744, 64255, 2),
    (64286, 64286, 0),
    (65024, 65039, 0),
    (65040, 65049, 2),
    (65056, 65071, 0),
    (65072, 65106, 2),
    (65108, 65126, 2),
    (65128, 65131, 2),
    (65279, 65279, 0),
    (65281, 65376, 2),
    (65504, 65510, 2),
    (65529, 65531, 0),
    (66045, 66045, 0),
    (66272, 66272, 0),
    (66422, 66426, 0),
    (68097, 68099, 0),
    (68101, 68102, 0),
    (68108, 68111, 0),
    (68152, 68154, 0),
    (68159, 68159, 0),
    (68325, 68326, 0),
    (68900, 68903, 0),
    (69291, 69292, 0),
    (69373, 69375, 0),
    (69446, 69456, 0),
    (69506, 69509, 0),
    (69632, 69634, 0),
    (69688, 69702, 0),
    (69744, 69744, 0),
    (69747, 69748, 0),
    (69759, 69762, 0),
    (69808, 69818, 0),
    (69821, 69821, 0),
    (69826, 69826, 0),
    (69837, 69837, 0),
    (69888, 69890, 0),
    (69927, 69940, 0),
    (69957, 69958, 0),
    (70003, 70003, 0),
    (70016, 70018, 0),
    (70067, 70080, 0),
    (70089, 70092, 0),
    (70094, 70095, 0),
    (70188, 70199, 0),
    (70206, 70206, 0),
    (70209, 70209, 0),
    (70367, 70378, 0),
    (70400, 70403, 0),
    (70459, 70460, 0),
    (70462, 70468, 0),
    (70471, 70472, 0),
    (70475, 70477, 0),
    (70487, 70487, 0),
    (70498, 70499, 0),
    (70502, 70508, 0),
    (70512, 70516, 0),
    (70709, 70726, 0),
    (70750, 70750, 0),
    (70832, 70851, 0),
    (71087, 71093, 0),
    (71096, 71104, 0),
    (71132, 71133, 0),
    (71216, 71232, 0),
    (71339, 71351, 0),
    (71453, 71467, 0),
    (71724, 71738, 0),
    (71984, 71989, 0),
    (71991, 71992, 0),
    (71995, 71998, 0),
    (72000, 72000, 0),
    (72002, 72003, 0),
    (72145, 72151, 0),
    (72154, 72160, 0),
    (72164, 72164, 0),
    (72193, 72202, 0),
    (72243, 72249, 0),
    (72251, 72254, 0),
    (72263, 72263, 0),
    (72273, 72283, 0),
    (72330, 72345, 0),
    (72751, 72758, 0),
    (72760, 72767, 0),
    (72850, 72871, 0),
    (72873, 72886, 0),
    (73009, 73014, 0),
    (73018, 73018, 0),
    (73020, 73021, 0),
    (73023, 73029, 0),
    (73031, 73031, 0),
    (73098, 73102, 0),
    (73104, 73105, 0),
    (73107, 73111, 0),
    (73459, 73462, 0),
    (73472, 73473, 0),
    (73475, 73475, 0),
    (73524, 73530, 0),
    (73534, 73538, 0),
    (78896, 78912, 0),
    (78919, 78933, 0),
    (92912, 92916, 0),
    (92976, 92982, 0),
    (94031, 94031, 0),
    (94033, 94087, 0),
    (94095, 94098, 0),
    (94176, 94179, 2),
    (94180, 94180, 0),
    (94192, 94193, 0),
    (94208, 100343, 2),
    (100352, 101589, 2),
    (101632, 101640, 2),
    (110576, 110579, 2),
    (110581, 110587, 2),
    (110589, 110590, 2),
    (110592, 110882, 2),
    (110898, 110898, 2),
    (110928, 110930, 2),
    (110933, 110933, 2),
    (110948, 110951, 2),
    (110960, 111355, 2),
    (113821, 113822, 0),
    (113824, 113827, 0),
    (118528, 118573, 0),
    (118576, 118598, 0),
    (119141, 119145, 0),
    (119149, 119170, 0),
    (119173, 119179, 0),
    (119210, 119213, 0),
    (119362, 119364, 0),
    (121344, 121398, 0),
    (121403, 121452, 0),
    (121461, 121461, 0),
    (121476, 121476, 0),
    (121499, 121503, 0),
    (121505, 121519, 0),
    (122880, 122886, 0),
    (122888, 122904, 0),
    (122907, 122913, 0),
    (122915, 122916, 0),
    (122918, 122922, 0),
    (123023, 123023, 0),
    (123184, 123190, 0),
    (123566, 123566, 0),
    (123628, 123631, 0),
    (124140, 124143, 0),
    (125136, 125142, 0),
    (125252, 125258, 0),
    (126980, 126980, 2),
    (127183, 127183, 2),
    (127374, 127374, 2),
    (127377, 127386, 2),
    (127488, 127490, 2),
    (127504, 127547, 2),
    (127552, 127560, 2),
    (127568, 127569, 2),
    (127584, 127589, 2),
    (127744, 127776, 2),
    (127789, 127797, 2),
    (127799, 127868, 2),
    (127870, 127891, 2),
    (127904, 127946, 2),
    (127951, 127955, 2),
    (127968, 127984, 2),
    (127988, 127988, 2),
    (127992, 127994, 2),
    (127995, 127999, 0),
    (128000, 128062, 2),
    (128064, 128064, 2),
    (128066, 128252, 2),
    (128255, 128317, 2),
    (128331, 128334, 2),
    (128336, 128359, 2),
    (128378, 128378, 2),
    (128405, 128406, 2),
    (128420, 128420, 2),
    (128507, 128591, 2),
    (128640, 128709, 2),
    (128716, 128716, 2),
    (128720, 128722, 2),
    (128725, 128727, 2),
    (128732, 128735, 2),
    (128747, 128748, 2),
    (128756, 128764, 2),
    (128992, 129003, 2),
    (129008, 129008, 2),
    (129292, 129338, 2),
    (129340, 129349, 2),
    (129351, 129535, 2),
    (129648, 129660, 2),
    (129664, 129672, 2),
    (129680, 129725, 2),
    (129727, 129733, 2),
    (129742, 129755, 2),
    (129760, 129768, 2),
    (129776, 129784, 2),
    (131072, 196605, 2),
    (196608, 262141, 2),
    (917505, 917505, 0),
    (917536, 917631, 0),
    (917760, 917999, 0),
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexers\objects.py ===
"""Indexer objects for computing start/end window bounds for rolling operations"""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from pandas._libs.tslibs import BaseOffset
from pandas._libs.window.indexers import calculate_variable_window_bounds
from pandas.util._decorators import Appender

from pandas.core.dtypes.common import ensure_platform_int

from pandas.core.indexes.datetimes import DatetimeIndex

from pandas.tseries.offsets import Nano

get_window_bounds_doc = """
Computes the bounds of a window.

Parameters
----------
num_values : int, default 0
    number of values that will be aggregated over
window_size : int, default 0
    the number of rows in a window
min_periods : int, default None
    min_periods passed from the top level rolling API
center : bool, default None
    center passed from the top level rolling API
closed : str, default None
    closed passed from the top level rolling API
step : int, default None
    step passed from the top level rolling API
    .. versionadded:: 1.5
win_type : str, default None
    win_type passed from the top level rolling API

Returns
-------
A tuple of ndarray[int64]s, indicating the boundaries of each
window
"""


class BaseIndexer:
    """
    Base class for window bounds calculations.

    Examples
    --------
    >>> from pandas.api.indexers import BaseIndexer
    >>> class CustomIndexer(BaseIndexer):
    ...     def get_window_bounds(self, num_values, min_periods, center, closed, step):
    ...         start = np.empty(num_values, dtype=np.int64)
    ...         end = np.empty(num_values, dtype=np.int64)
    ...         for i in range(num_values):
    ...             start[i] = i
    ...             end[i] = i + self.window_size
    ...         return start, end
    >>> df = pd.DataFrame({"values": range(5)})
    >>> indexer = CustomIndexer(window_size=2)
    >>> df.rolling(indexer).sum()
        values
    0	1.0
    1	3.0
    2	5.0
    3	7.0
    4	4.0
    """

    def __init__(
        self, index_array: np.ndarray | None = None, window_size: int = 0, **kwargs
    ) -> None:
        self.index_array = index_array
        self.window_size = window_size
        # Set user defined kwargs as attributes that can be used in get_window_bounds
        for key, value in kwargs.items():
            setattr(self, key, value)

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class FixedWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of fixed length."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if center or self.window_size == 0:
            offset = (self.window_size - 1) // 2
        else:
            offset = 0

        end = np.arange(1 + offset, num_values + 1 + offset, step, dtype="int64")
        start = end - self.window_size
        if closed in ["left", "both"]:
            start -= 1
        if closed in ["left", "neither"]:
            end -= 1

        end = np.clip(end, 0, num_values)
        start = np.clip(start, 0, num_values)

        return start, end


class VariableWindowIndexer(BaseIndexer):
    """Creates window boundaries that are of variable length, namely for time series."""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # error: Argument 4 to "calculate_variable_window_bounds" has incompatible
        # type "Optional[bool]"; expected "bool"
        # error: Argument 6 to "calculate_variable_window_bounds" has incompatible
        # type "Optional[ndarray]"; expected "ndarray"
        return calculate_variable_window_bounds(
            num_values,
            self.window_size,
            min_periods,
            center,  # type: ignore[arg-type]
            closed,
            self.index_array,  # type: ignore[arg-type]
        )


class VariableOffsetWindowIndexer(BaseIndexer):
    """
    Calculate window boundaries based on a non-fixed offset such as a BusinessDay.

    Examples
    --------
    >>> from pandas.api.indexers import VariableOffsetWindowIndexer
    >>> df = pd.DataFrame(range(10), index=pd.date_range("2020", periods=10))
    >>> offset = pd.offsets.BDay(1)
    >>> indexer = VariableOffsetWindowIndexer(index=df.index, offset=offset)
    >>> df
                0
    2020-01-01  0
    2020-01-02  1
    2020-01-03  2
    2020-01-04  3
    2020-01-05  4
    2020-01-06  5
    2020-01-07  6
    2020-01-08  7
    2020-01-09  8
    2020-01-10  9
    >>> df.rolling(indexer).sum()
                   0
    2020-01-01   0.0
    2020-01-02   1.0
    2020-01-03   2.0
    2020-01-04   3.0
    2020-01-05   7.0
    2020-01-06  12.0
    2020-01-07   6.0
    2020-01-08   7.0
    2020-01-09   8.0
    2020-01-10   9.0
    """

    def __init__(
        self,
        index_array: np.ndarray | None = None,
        window_size: int = 0,
        index: DatetimeIndex | None = None,
        offset: BaseOffset | None = None,
        **kwargs,
    ) -> None:
        super().__init__(index_array, window_size, **kwargs)
        if not isinstance(index, DatetimeIndex):
            raise ValueError("index must be a DatetimeIndex.")
        self.index = index
        if not isinstance(offset, BaseOffset):
            raise ValueError("offset must be a DateOffset-like object.")
        self.offset = offset

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if step is not None:
            raise NotImplementedError("step not implemented for variable offset window")
        if num_values <= 0:
            return np.empty(0, dtype="int64"), np.empty(0, dtype="int64")

        # if windows is variable, default is 'right', otherwise default is 'both'
        if closed is None:
            closed = "right" if self.index is not None else "both"

        right_closed = closed in ["right", "both"]
        left_closed = closed in ["left", "both"]

        if self.index[num_values - 1] < self.index[0]:
            index_growth_sign = -1
        else:
            index_growth_sign = 1
        offset_diff = index_growth_sign * self.offset

        start = np.empty(num_values, dtype="int64")
        start.fill(-1)
        end = np.empty(num_values, dtype="int64")
        end.fill(-1)

        start[0] = 0

        # right endpoint is closed
        if right_closed:
            end[0] = 1
        # right endpoint is open
        else:
            end[0] = 0

        zero = timedelta(0)
        # start is start of slice interval (including)
        # end is end of slice interval (not including)
        for i in range(1, num_values):
            end_bound = self.index[i]
            start_bound = end_bound - offset_diff

            # left endpoint is closed
            if left_closed:
                start_bound -= Nano(1)

            # advance the start bound until we are
            # within the constraint
            start[i] = i
            for j in range(start[i - 1], i):
                start_diff = (self.index[j] - start_bound) * index_growth_sign
                if start_diff > zero:
                    start[i] = j
                    break

            # end bound is previous end
            # or current index
            end_diff = (self.index[end[i - 1]] - end_bound) * index_growth_sign
            if end_diff == zero and not right_closed:
                end[i] = end[i - 1] + 1
            elif end_diff <= zero:
                end[i] = i + 1
            else:
                end[i] = end[i - 1]

            # right endpoint is open
            if not right_closed:
                end[i] -= 1

        return start, end


class ExpandingIndexer(BaseIndexer):
    """Calculate expanding window bounds, mimicking df.expanding()"""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros(num_values, dtype=np.int64),
            np.arange(1, num_values + 1, dtype=np.int64),
        )


class FixedForwardWindowIndexer(BaseIndexer):
    """
    Creates window boundaries for fixed-length windows that include the current row.

    Examples
    --------
    >>> df = pd.DataFrame({'B': [0, 1, 2, np.nan, 4]})
    >>> df
         B
    0  0.0
    1  1.0
    2  2.0
    3  NaN
    4  4.0

    >>> indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=2)
    >>> df.rolling(window=indexer, min_periods=1).sum()
         B
    0  1.0
    1  3.0
    2  2.0
    3  4.0
    4  4.0
    """

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if center:
            raise ValueError("Forward-looking windows can't have center=True")
        if closed is not None:
            raise ValueError(
                "Forward-looking windows don't support setting the closed argument"
            )
        if step is None:
            step = 1

        start = np.arange(0, num_values, step, dtype="int64")
        end = start + self.window_size
        if self.window_size:
            end = np.clip(end, 0, num_values)

        return start, end


class GroupbyIndexer(BaseIndexer):
    """Calculate bounds to compute groupby rolling, mimicking df.groupby().rolling()"""

    def __init__(
        self,
        index_array: np.ndarray | None = None,
        window_size: int | BaseIndexer = 0,
        groupby_indices: dict | None = None,
        window_indexer: type[BaseIndexer] = BaseIndexer,
        indexer_kwargs: dict | None = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ----------
        index_array : np.ndarray or None
            np.ndarray of the index of the original object that we are performing
            a chained groupby operation over. This index has been pre-sorted relative to
            the groups
        window_size : int or BaseIndexer
            window size during the windowing operation
        groupby_indices : dict or None
            dict of {group label: [positional index of rows belonging to the group]}
        window_indexer : BaseIndexer
            BaseIndexer class determining the start and end bounds of each group
        indexer_kwargs : dict or None
            Custom kwargs to be passed to window_indexer
        **kwargs :
            keyword arguments that will be available when get_window_bounds is called
        """
        self.groupby_indices = groupby_indices or {}
        self.window_indexer = window_indexer
        self.indexer_kwargs = indexer_kwargs.copy() if indexer_kwargs else {}
        super().__init__(
            index_array=index_array,
            window_size=self.indexer_kwargs.pop("window_size", window_size),
            **kwargs,
        )

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 1) For each group, get the indices that belong to the group
        # 2) Use the indices to calculate the start & end bounds of the window
        # 3) Append the window bounds in group order
        start_arrays = []
        end_arrays = []
        window_indices_start = 0
        for key, indices in self.groupby_indices.items():
            index_array: np.ndarray | None

            if self.index_array is not None:
                index_array = self.index_array.take(ensure_platform_int(indices))
            else:
                index_array = self.index_array
            indexer = self.window_indexer(
                index_array=index_array,
                window_size=self.window_size,
                **self.indexer_kwargs,
            )
            start, end = indexer.get_window_bounds(
                len(indices), min_periods, center, closed, step
            )
            start = start.astype(np.int64)
            end = end.astype(np.int64)
            assert len(start) == len(
                end
            ), "these should be equal in length from get_window_bounds"
            # Cannot use groupby_indices as they might not be monotonic with the object
            # we're rolling over
            window_indices = np.arange(
                window_indices_start, window_indices_start + len(indices)
            )
            window_indices_start += len(indices)
            # Extend as we'll be slicing window like [start, end)
            window_indices = np.append(window_indices, [window_indices[-1] + 1]).astype(
                np.int64, copy=False
            )
            start_arrays.append(window_indices.take(ensure_platform_int(start)))
            end_arrays.append(window_indices.take(ensure_platform_int(end)))
        if len(start_arrays) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        start = np.concatenate(start_arrays)
        end = np.concatenate(end_arrays)
        return start, end


class ExponentialMovingWindowIndexer(BaseIndexer):
    """Calculate ewm window bounds (the entire window)"""

    @Appender(get_window_bounds_doc)
    def get_window_bounds(
        self,
        num_values: int = 0,
        min_periods: int | None = None,
        center: bool | None = None,
        closed: str | None = None,
        step: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        return np.array([0], dtype=np.int64), np.array([num_values], dtype=np.int64)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\rewritingsystem.py ===
from collections import deque
from sympy.combinatorics.rewritingsystem_fsm import StateMachine

class RewritingSystem:
    '''
    A class implementing rewriting systems for `FpGroup`s.

    References
    ==========
    .. [1] Epstein, D., Holt, D. and Rees, S. (1991).
           The use of Knuth-Bendix methods to solve the word problem in automatic groups.
           Journal of Symbolic Computation, 12(4-5), pp.397-414.

    .. [2] GAP's Manual on its KBMAG package
           https://www.gap-system.org/Manuals/pkg/kbmag-1.5.3/doc/manual.pdf

    '''
    def __init__(self, group):
        self.group = group
        self.alphabet = group.generators
        self._is_confluent = None

        # these values are taken from [2]
        self.maxeqns = 32767 # max rules
        self.tidyint = 100 # rules before tidying

        # _max_exceeded is True if maxeqns is exceeded
        # at any point
        self._max_exceeded = False

        # Reduction automaton
        self.reduction_automaton = None
        self._new_rules = {}

        # dictionary of reductions
        self.rules = {}
        self.rules_cache = deque([], 50)
        self._init_rules()


        # All the transition symbols in the automaton
        generators = list(self.alphabet)
        generators += [gen**-1 for gen in generators]
        # Create a finite state machine as an instance of the StateMachine object
        self.reduction_automaton = StateMachine('Reduction automaton for '+ repr(self.group), generators)
        self.construct_automaton()

    def set_max(self, n):
        '''
        Set the maximum number of rules that can be defined

        '''
        if n > self.maxeqns:
            self._max_exceeded = False
        self.maxeqns = n
        return

    @property
    def is_confluent(self):
        '''
        Return `True` if the system is confluent

        '''
        if self._is_confluent is None:
            self._is_confluent = self._check_confluence()
        return self._is_confluent

    def _init_rules(self):
        identity = self.group.free_group.identity
        for r in self.group.relators:
            self.add_rule(r, identity)
        self._remove_redundancies()
        return

    def _add_rule(self, r1, r2):
        '''
        Add the rule r1 -> r2 with no checking or further
        deductions

        '''
        if len(self.rules) + 1 > self.maxeqns:
            self._is_confluent = self._check_confluence()
            self._max_exceeded = True
            raise RuntimeError("Too many rules were defined.")
        self.rules[r1] = r2
        # Add the newly added rule to the `new_rules` dictionary.
        if self.reduction_automaton:
            self._new_rules[r1] = r2

    def add_rule(self, w1, w2, check=False):
        new_keys = set()

        if w1 == w2:
            return new_keys

        if w1 < w2:
            w1, w2 = w2, w1

        if (w1, w2) in self.rules_cache:
            return new_keys
        self.rules_cache.append((w1, w2))

        s1, s2 = w1, w2

        # The following is the equivalent of checking
        # s1 for overlaps with the implicit reductions
        # {g*g**-1 -> <identity>} and {g**-1*g -> <identity>}
        # for any generator g without installing the
        # redundant rules that would result from processing
        # the overlaps. See [1], Section 3 for details.

        if len(s1) - len(s2) < 3:
            if s1 not in self.rules:
                new_keys.add(s1)
                if not check:
                    self._add_rule(s1, s2)
            if s2**-1 > s1**-1 and s2**-1 not in self.rules:
                new_keys.add(s2**-1)
                if not check:
                    self._add_rule(s2**-1, s1**-1)

        # overlaps on the right
        while len(s1) - len(s2) > -1:
            g = s1[len(s1)-1]
            s1 = s1.subword(0, len(s1)-1)
            s2 = s2*g**-1
            if len(s1) - len(s2) < 0:
                if s2 not in self.rules:
                    if not check:
                        self._add_rule(s2, s1)
                    new_keys.add(s2)
            elif len(s1) - len(s2) < 3:
                new = self.add_rule(s1, s2, check)
                new_keys.update(new)

        # overlaps on the left
        while len(w1) - len(w2) > -1:
            g = w1[0]
            w1 = w1.subword(1, len(w1))
            w2 = g**-1*w2
            if len(w1) - len(w2) < 0:
                if w2 not in self.rules:
                    if not check:
                        self._add_rule(w2, w1)
                    new_keys.add(w2)
            elif len(w1) - len(w2) < 3:
                new = self.add_rule(w1, w2, check)
                new_keys.update(new)

        return new_keys

    def _remove_redundancies(self, changes=False):
        '''
        Reduce left- and right-hand sides of reduction rules
        and remove redundant equations (i.e. those for which
        lhs == rhs). If `changes` is `True`, return a set
        containing the removed keys and a set containing the
        added keys

        '''
        removed = set()
        added = set()
        rules = self.rules.copy()
        for r in rules:
            v = self.reduce(r, exclude=r)
            w = self.reduce(rules[r])
            if v != r:
                del self.rules[r]
                removed.add(r)
                if v > w:
                    added.add(v)
                    self.rules[v] = w
                elif v < w:
                    added.add(w)
                    self.rules[w] = v
            else:
                self.rules[v] = w
        if changes:
            return removed, added
        return

    def make_confluent(self, check=False):
        '''
        Try to make the system confluent using the Knuth-Bendix
        completion algorithm

        '''
        if self._max_exceeded:
            return self._is_confluent
        lhs = list(self.rules.keys())

        def _overlaps(r1, r2):
            len1 = len(r1)
            len2 = len(r2)
            result = []
            for j in range(1, len1 + len2):
                if (r1.subword(len1 - j, len1 + len2 - j, strict=False)
                       == r2.subword(j - len1, j, strict=False)):
                    a = r1.subword(0, len1-j, strict=False)
                    a = a*r2.subword(0, j-len1, strict=False)
                    b = r2.subword(j-len1, j, strict=False)
                    c = r2.subword(j, len2, strict=False)
                    c = c*r1.subword(len1 + len2 - j, len1, strict=False)
                    result.append(a*b*c)
            return result

        def _process_overlap(w, r1, r2, check):
                s = w.eliminate_word(r1, self.rules[r1])
                s = self.reduce(s)
                t = w.eliminate_word(r2, self.rules[r2])
                t = self.reduce(t)
                if s != t:
                    if check:
                        # system not confluent
                        return [0]
                    try:
                        new_keys = self.add_rule(t, s, check)
                        return new_keys
                    except RuntimeError:
                        return False
                return

        added = 0
        i = 0
        while i < len(lhs):
            r1 = lhs[i]
            i += 1
            # j could be i+1 to not
            # check each pair twice but lhs
            # is extended in the loop and the new
            # elements have to be checked with the
            # preceding ones. there is probably a better way
            # to handle this
            j = 0
            while j < len(lhs):
                r2 = lhs[j]
                j += 1
                if r1 == r2:
                    continue
                overlaps = _overlaps(r1, r2)
                overlaps.extend(_overlaps(r1**-1, r2))
                if not overlaps:
                    continue
                for w in overlaps:
                    new_keys = _process_overlap(w, r1, r2, check)
                    if new_keys:
                        if check:
                            return False
                        lhs.extend(new_keys)
                        added += len(new_keys)
                    elif new_keys == False:
                        # too many rules were added so the process
                        # couldn't complete
                        return self._is_confluent

                if added > self.tidyint and not check:
                    # tidy up
                    r, a = self._remove_redundancies(changes=True)
                    added = 0
                    if r:
                        # reset i since some elements were removed
                        i = min(lhs.index(s) for s in r)
                    lhs = [l for l in lhs if l not in r]
                    lhs.extend(a)
                    if r1 in r:
                        # r1 was removed as redundant
                        break

        self._is_confluent = True
        if not check:
            self._remove_redundancies()
        return True

    def _check_confluence(self):
        return self.make_confluent(check=True)

    def reduce(self, word, exclude=None):
        '''
        Apply reduction rules to `word` excluding the reduction rule
        for the lhs equal to `exclude`

        '''
        rules = {r: self.rules[r] for r in self.rules if r != exclude}
        # the following is essentially `eliminate_words()` code from the
        # `FreeGroupElement` class, the only difference being the first
        # "if" statement
        again = True
        new = word
        while again:
            again = False
            for r in rules:
                prev = new
                if rules[r]**-1 > r**-1:
                    new = new.eliminate_word(r, rules[r], _all=True, inverse=False)
                else:
                    new = new.eliminate_word(r, rules[r], _all=True)
                if new != prev:
                    again = True
        return new

    def _compute_inverse_rules(self, rules):
        '''
        Compute the inverse rules for a given set of rules.
        The inverse rules are used in the automaton for word reduction.

        Arguments:
            rules (dictionary): Rules for which the inverse rules are to computed.

        Returns:
            Dictionary of inverse_rules.

        '''
        inverse_rules = {}
        for r in rules:
            rule_key_inverse = r**-1
            rule_value_inverse = (rules[r])**-1
            if (rule_value_inverse < rule_key_inverse):
                inverse_rules[rule_key_inverse] = rule_value_inverse
            else:
                inverse_rules[rule_value_inverse] = rule_key_inverse
        return inverse_rules

    def construct_automaton(self):
        '''
        Construct the automaton based on the set of reduction rules of the system.

        Automata Design:
        The accept states of the automaton are the proper prefixes of the left hand side of the rules.
        The complete left hand side of the rules are the dead states of the automaton.

        '''
        self._add_to_automaton(self.rules)

    def _add_to_automaton(self, rules):
        '''
        Add new states and transitions to the automaton.

        Summary:
        States corresponding to the new rules added to the system are computed and added to the automaton.
        Transitions in the previously added states are also modified if necessary.

        Arguments:
            rules (dictionary) -- Dictionary of the newly added rules.

        '''
        # Automaton variables
        automaton_alphabet = []
        proper_prefixes = {}

        # compute the inverses of all the new rules added
        all_rules = rules
        inverse_rules = self._compute_inverse_rules(all_rules)
        all_rules.update(inverse_rules)

        # Keep track of the accept_states.
        accept_states = []

        for rule in all_rules:
            # The symbols present in the new rules are the symbols to be verified at each state.
            # computes the automaton_alphabet, as the transitions solely depend upon the new states.
            automaton_alphabet += rule.letter_form_elm
            # Compute the proper prefixes for every rule.
            proper_prefixes[rule] = []
            letter_word_array = list(rule.letter_form_elm)
            len_letter_word_array = len(letter_word_array)
            for i in range (1, len_letter_word_array):
                letter_word_array[i] = letter_word_array[i-1]*letter_word_array[i]
                # Add accept states.
                elem = letter_word_array[i-1]
                if elem not in self.reduction_automaton.states:
                    self.reduction_automaton.add_state(elem, state_type='a')
                    accept_states.append(elem)
            proper_prefixes[rule] = letter_word_array
            # Check for overlaps between dead and accept states.
            if rule in accept_states:
                self.reduction_automaton.states[rule].state_type = 'd'
                self.reduction_automaton.states[rule].rh_rule = all_rules[rule]
                accept_states.remove(rule)
            # Add dead states
            if rule not in self.reduction_automaton.states:
                self.reduction_automaton.add_state(rule, state_type='d', rh_rule=all_rules[rule])

        automaton_alphabet = set(automaton_alphabet)

        # Add new transitions for every state.
        for state in self.reduction_automaton.states:
            current_state_name = state
            current_state_type = self.reduction_automaton.states[state].state_type
            # Transitions will be modified only when suffixes of the current_state
            # belongs to the proper_prefixes of the new rules.
            # The rest are ignored if they cannot lead to a dead state after a finite number of transisitons.
            if current_state_type == 's':
                for letter in automaton_alphabet:
                    if letter in self.reduction_automaton.states:
                        self.reduction_automaton.states[state].add_transition(letter, letter)
                    else:
                        self.reduction_automaton.states[state].add_transition(letter, current_state_name)
            elif current_state_type == 'a':
                # Check if the transition to any new state in possible.
                for letter in automaton_alphabet:
                    _next = current_state_name*letter
                    while len(_next) and _next not in self.reduction_automaton.states:
                        _next = _next.subword(1, len(_next))
                    if not len(_next):
                        _next = 'start'
                    self.reduction_automaton.states[state].add_transition(letter, _next)

        # Add transitions for new states. All symbols used in the automaton are considered here.
        # Ignore this if `reduction_automaton.automaton_alphabet` = `automaton_alphabet`.
        if len(self.reduction_automaton.automaton_alphabet) != len(automaton_alphabet):
            for state in accept_states:
                current_state_name = state
                for letter in self.reduction_automaton.automaton_alphabet:
                    _next = current_state_name*letter
                    while len(_next) and _next not in self.reduction_automaton.states:
                        _next = _next.subword(1, len(_next))
                    if not len(_next):
                        _next = 'start'
                    self.reduction_automaton.states[state].add_transition(letter, _next)

    def reduce_using_automaton(self, word):
        '''
        Reduce a word using an automaton.

        Summary:
        All the symbols of the word are stored in an array and are given as the input to the automaton.
        If the automaton reaches a dead state that subword is replaced and the automaton is run from the beginning.
        The complete word has to be replaced when the word is read and the automaton reaches a dead state.
        So, this process is repeated until the word is read completely and the automaton reaches the accept state.

        Arguments:
            word (instance of FreeGroupElement) -- Word that needs to be reduced.

        '''
        # Modify the automaton if new rules are found.
        if self._new_rules:
            self._add_to_automaton(self._new_rules)
            self._new_rules = {}

        flag = 1
        while flag:
            flag = 0
            current_state = self.reduction_automaton.states['start']
            for i, s in enumerate(word.letter_form_elm):
                next_state_name = current_state.transitions[s]
                next_state = self.reduction_automaton.states[next_state_name]
                if next_state.state_type == 'd':
                    subst = next_state.rh_rule
                    word = word.substituted_word(i - len(next_state_name) + 1, i+1, subst)
                    flag = 1
                    break
                current_state = next_state
        return word

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\__init__.py ===
# isort:skip_file
"""
Dimensional analysis and unit systems.

This module defines dimension/unit systems and physical quantities. It is
based on a group-theoretical construction where dimensions are represented as
vectors (coefficients being the exponents), and units are defined as a dimension
to which we added a scale.

Quantities are built from a factor and a unit, and are the basic objects that
one will use when doing computations.

All objects except systems and prefixes can be used in SymPy expressions.
Note that as part of a CAS, various objects do not combine automatically
under operations.

Details about the implementation can be found in the documentation, and we
will not repeat all the explanations we gave there concerning our approach.
Ideas about future developments can be found on the `Github wiki
<https://github.com/sympy/sympy/wiki/Unit-systems>`_, and you should consult
this page if you are willing to help.

Useful functions:

- ``find_unit``: easily lookup pre-defined units.
- ``convert_to(expr, newunit)``: converts an expression into the same
    expression expressed in another unit.

"""

from .dimensions import Dimension, DimensionSystem
from .unitsystem import UnitSystem
from .util import convert_to
from .quantities import Quantity

from .definitions.dimension_definitions import (
    amount_of_substance, acceleration, action, area,
    capacitance, charge, conductance, current, energy,
    force, frequency, impedance, inductance, length,
    luminous_intensity, magnetic_density,
    magnetic_flux, mass, momentum, power, pressure, temperature, time,
    velocity, voltage, volume
)

Unit = Quantity

speed = velocity
luminosity = luminous_intensity
magnetic_flux_density = magnetic_density
amount = amount_of_substance

from .prefixes import (
    # 10-power based:
    yotta,
    zetta,
    exa,
    peta,
    tera,
    giga,
    mega,
    kilo,
    hecto,
    deca,
    deci,
    centi,
    milli,
    micro,
    nano,
    pico,
    femto,
    atto,
    zepto,
    yocto,
    # 2-power based:
    kibi,
    mebi,
    gibi,
    tebi,
    pebi,
    exbi,
)

from .definitions import (
    percent, percents,
    permille,
    rad, radian, radians,
    deg, degree, degrees,
    sr, steradian, steradians,
    mil, angular_mil, angular_mils,
    m, meter, meters,
    kg, kilogram, kilograms,
    s, second, seconds,
    A, ampere, amperes,
    K, kelvin, kelvins,
    mol, mole, moles,
    cd, candela, candelas,
    g, gram, grams,
    mg, milligram, milligrams,
    ug, microgram, micrograms,
    t, tonne, metric_ton,
    newton, newtons, N,
    joule, joules, J,
    watt, watts, W,
    pascal, pascals, Pa, pa,
    hertz, hz, Hz,
    coulomb, coulombs, C,
    volt, volts, v, V,
    ohm, ohms,
    siemens, S, mho, mhos,
    farad, farads, F,
    henry, henrys, H,
    tesla, teslas, T,
    weber, webers, Wb, wb,
    optical_power, dioptre, D,
    lux, lx,
    katal, kat,
    gray, Gy,
    becquerel, Bq,
    km, kilometer, kilometers,
    dm, decimeter, decimeters,
    cm, centimeter, centimeters,
    mm, millimeter, millimeters,
    um, micrometer, micrometers, micron, microns,
    nm, nanometer, nanometers,
    pm, picometer, picometers,
    ft, foot, feet,
    inch, inches,
    yd, yard, yards,
    mi, mile, miles,
    nmi, nautical_mile, nautical_miles,
    angstrom, angstroms,
    ha, hectare,
    l, L, liter, liters,
    dl, dL, deciliter, deciliters,
    cl, cL, centiliter, centiliters,
    ml, mL, milliliter, milliliters,
    ms, millisecond, milliseconds,
    us, microsecond, microseconds,
    ns, nanosecond, nanoseconds,
    ps, picosecond, picoseconds,
    minute, minutes,
    h, hour, hours,
    day, days,
    anomalistic_year, anomalistic_years,
    sidereal_year, sidereal_years,
    tropical_year, tropical_years,
    common_year, common_years,
    julian_year, julian_years,
    draconic_year, draconic_years,
    gaussian_year, gaussian_years,
    full_moon_cycle, full_moon_cycles,
    year, years,
    G, gravitational_constant,
    c, speed_of_light,
    elementary_charge,
    hbar,
    planck,
    eV, electronvolt, electronvolts,
    avogadro_number,
    avogadro, avogadro_constant,
    boltzmann, boltzmann_constant,
    stefan, stefan_boltzmann_constant,
    R, molar_gas_constant,
    faraday_constant,
    josephson_constant,
    von_klitzing_constant,
    Da, dalton, amu, amus, atomic_mass_unit, atomic_mass_constant,
    me, electron_rest_mass,
    gee, gees, acceleration_due_to_gravity,
    u0, magnetic_constant, vacuum_permeability,
    e0, electric_constant, vacuum_permittivity,
    Z0, vacuum_impedance,
    coulomb_constant, electric_force_constant,
    atmosphere, atmospheres, atm,
    kPa,
    bar, bars,
    pound, pounds,
    psi,
    dHg0,
    mmHg, torr,
    mmu, mmus, milli_mass_unit,
    quart, quarts,
    ly, lightyear, lightyears,
    au, astronomical_unit, astronomical_units,
    planck_mass,
    planck_time,
    planck_temperature,
    planck_length,
    planck_charge,
    planck_area,
    planck_volume,
    planck_momentum,
    planck_energy,
    planck_force,
    planck_power,
    planck_density,
    planck_energy_density,
    planck_intensity,
    planck_angular_frequency,
    planck_pressure,
    planck_current,
    planck_voltage,
    planck_impedance,
    planck_acceleration,
    bit, bits,
    byte,
    kibibyte, kibibytes,
    mebibyte, mebibytes,
    gibibyte, gibibytes,
    tebibyte, tebibytes,
    pebibyte, pebibytes,
    exbibyte, exbibytes,
)

from .systems import (
    mks, mksa, si
)


def find_unit(quantity, unit_system="SI"):
    """
    Return a list of matching units or dimension names.

    - If ``quantity`` is a string -- units/dimensions containing the string
    `quantity`.
    - If ``quantity`` is a unit or dimension -- units having matching base
    units or dimensions.

    Examples
    ========

    >>> from sympy.physics import units as u
    >>> u.find_unit('charge')
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit(u.charge)
    ['C', 'coulomb', 'coulombs', 'planck_charge', 'elementary_charge']
    >>> u.find_unit("ampere")
    ['ampere', 'amperes']
    >>> u.find_unit('angstrom')
    ['angstrom', 'angstroms']
    >>> u.find_unit('volt')
    ['volt', 'volts', 'electronvolt', 'electronvolts', 'planck_voltage']
    >>> u.find_unit(u.inch**3)[:9]
    ['L', 'l', 'cL', 'cl', 'dL', 'dl', 'mL', 'ml', 'liter']
    """
    unit_system = UnitSystem.get_unit_system(unit_system)

    import sympy.physics.units as u
    rv = []
    if isinstance(quantity, str):
        rv = [i for i in dir(u) if quantity in i and isinstance(getattr(u, i), Quantity)]
        dim = getattr(u, quantity)
        if isinstance(dim, Dimension):
            rv.extend(find_unit(dim))
    else:
        for i in sorted(dir(u)):
            other = getattr(u, i)
            if not isinstance(other, Quantity):
                continue
            if isinstance(quantity, Quantity):
                if quantity.dimension == other.dimension:
                    rv.append(str(i))
            elif isinstance(quantity, Dimension):
                if other.dimension == quantity:
                    rv.append(str(i))
            elif other.dimension == Dimension(unit_system.get_dimensional_expr(quantity)):
                rv.append(str(i))
    return sorted(set(rv), key=lambda x: (len(x), x))

# NOTE: the old units module had additional variables:
# 'density', 'illuminance', 'resistance'.
# They were not dimensions, but units (old Unit class).

__all__ = [
    'Dimension', 'DimensionSystem',
    'UnitSystem',
    'convert_to',
    'Quantity',

    'amount_of_substance', 'acceleration', 'action', 'area',
    'capacitance', 'charge', 'conductance', 'current', 'energy',
    'force', 'frequency', 'impedance', 'inductance', 'length',
    'luminous_intensity', 'magnetic_density',
    'magnetic_flux', 'mass', 'momentum', 'power', 'pressure', 'temperature', 'time',
    'velocity', 'voltage', 'volume',

    'Unit',

    'speed',
    'luminosity',
    'magnetic_flux_density',
    'amount',

    'yotta',
    'zetta',
    'exa',
    'peta',
    'tera',
    'giga',
    'mega',
    'kilo',
    'hecto',
    'deca',
    'deci',
    'centi',
    'milli',
    'micro',
    'nano',
    'pico',
    'femto',
    'atto',
    'zepto',
    'yocto',

    'kibi',
    'mebi',
    'gibi',
    'tebi',
    'pebi',
    'exbi',

    'percent', 'percents',
    'permille',
    'rad', 'radian', 'radians',
    'deg', 'degree', 'degrees',
    'sr', 'steradian', 'steradians',
    'mil', 'angular_mil', 'angular_mils',
    'm', 'meter', 'meters',
    'kg', 'kilogram', 'kilograms',
    's', 'second', 'seconds',
    'A', 'ampere', 'amperes',
    'K', 'kelvin', 'kelvins',
    'mol', 'mole', 'moles',
    'cd', 'candela', 'candelas',
    'g', 'gram', 'grams',
    'mg', 'milligram', 'milligrams',
    'ug', 'microgram', 'micrograms',
    't', 'tonne', 'metric_ton',
    'newton', 'newtons', 'N',
    'joule', 'joules', 'J',
    'watt', 'watts', 'W',
    'pascal', 'pascals', 'Pa', 'pa',
    'hertz', 'hz', 'Hz',
    'coulomb', 'coulombs', 'C',
    'volt', 'volts', 'v', 'V',
    'ohm', 'ohms',
    'siemens', 'S', 'mho', 'mhos',
    'farad', 'farads', 'F',
    'henry', 'henrys', 'H',
    'tesla', 'teslas', 'T',
    'weber', 'webers', 'Wb', 'wb',
    'optical_power', 'dioptre', 'D',
    'lux', 'lx',
    'katal', 'kat',
    'gray', 'Gy',
    'becquerel', 'Bq',
    'km', 'kilometer', 'kilometers',
    'dm', 'decimeter', 'decimeters',
    'cm', 'centimeter', 'centimeters',
    'mm', 'millimeter', 'millimeters',
    'um', 'micrometer', 'micrometers', 'micron', 'microns',
    'nm', 'nanometer', 'nanometers',
    'pm', 'picometer', 'picometers',
    'ft', 'foot', 'feet',
    'inch', 'inches',
    'yd', 'yard', 'yards',
    'mi', 'mile', 'miles',
    'nmi', 'nautical_mile', 'nautical_miles',
    'angstrom', 'angstroms',
    'ha', 'hectare',
    'l', 'L', 'liter', 'liters',
    'dl', 'dL', 'deciliter', 'deciliters',
    'cl', 'cL', 'centiliter', 'centiliters',
    'ml', 'mL', 'milliliter', 'milliliters',
    'ms', 'millisecond', 'milliseconds',
    'us', 'microsecond', 'microseconds',
    'ns', 'nanosecond', 'nanoseconds',
    'ps', 'picosecond', 'picoseconds',
    'minute', 'minutes',
    'h', 'hour', 'hours',
    'day', 'days',
    'anomalistic_year', 'anomalistic_years',
    'sidereal_year', 'sidereal_years',
    'tropical_year', 'tropical_years',
    'common_year', 'common_years',
    'julian_year', 'julian_years',
    'draconic_year', 'draconic_years',
    'gaussian_year', 'gaussian_years',
    'full_moon_cycle', 'full_moon_cycles',
    'year', 'years',
    'G', 'gravitational_constant',
    'c', 'speed_of_light',
    'elementary_charge',
    'hbar',
    'planck',
    'eV', 'electronvolt', 'electronvolts',
    'avogadro_number',
    'avogadro', 'avogadro_constant',
    'boltzmann', 'boltzmann_constant',
    'stefan', 'stefan_boltzmann_constant',
    'R', 'molar_gas_constant',
    'faraday_constant',
    'josephson_constant',
    'von_klitzing_constant',
    'Da', 'dalton', 'amu', 'amus', 'atomic_mass_unit', 'atomic_mass_constant',
    'me', 'electron_rest_mass',
    'gee', 'gees', 'acceleration_due_to_gravity',
    'u0', 'magnetic_constant', 'vacuum_permeability',
    'e0', 'electric_constant', 'vacuum_permittivity',
    'Z0', 'vacuum_impedance',
    'coulomb_constant', 'electric_force_constant',
    'atmosphere', 'atmospheres', 'atm',
    'kPa',
    'bar', 'bars',
    'pound', 'pounds',
    'psi',
    'dHg0',
    'mmHg', 'torr',
    'mmu', 'mmus', 'milli_mass_unit',
    'quart', 'quarts',
    'ly', 'lightyear', 'lightyears',
    'au', 'astronomical_unit', 'astronomical_units',
    'planck_mass',
    'planck_time',
    'planck_temperature',
    'planck_length',
    'planck_charge',
    'planck_area',
    'planck_volume',
    'planck_momentum',
    'planck_energy',
    'planck_force',
    'planck_power',
    'planck_density',
    'planck_energy_density',
    'planck_intensity',
    'planck_angular_frequency',
    'planck_pressure',
    'planck_current',
    'planck_voltage',
    'planck_impedance',
    'planck_acceleration',
    'bit', 'bits',
    'byte',
    'kibibyte', 'kibibytes',
    'mebibyte', 'mebibytes',
    'gibibyte', 'gibibytes',
    'tebibyte', 'tebibytes',
    'pebibyte', 'pebibytes',
    'exbibyte', 'exbibytes',

    'mks', 'mksa', 'si',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\_abnf.py ===
import array
import os
import struct
import sys
from threading import Lock
from typing import Callable, Optional, Union

from ._exceptions import WebSocketPayloadException, WebSocketProtocolException
from ._utils import validate_utf8

"""
_abnf.py
websocket - WebSocket client library for Python

Copyright 2024 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

try:
    # If wsaccel is available, use compiled routines to mask data.
    # wsaccel only provides around a 10% speed boost compared
    # to the websocket-client _mask() implementation.
    # Note that wsaccel is unmaintained.
    from wsaccel.xormask import XorMaskerSimple

    def _mask(mask_value: array.array, data_value: array.array) -> bytes:
        mask_result: bytes = XorMaskerSimple(mask_value).process(data_value)
        return mask_result

except ImportError:
    # wsaccel is not available, use websocket-client _mask()
    native_byteorder = sys.byteorder

    def _mask(mask_value: array.array, data_value: array.array) -> bytes:
        datalen = len(data_value)
        int_data_value = int.from_bytes(data_value, native_byteorder)
        int_mask_value = int.from_bytes(
            mask_value * (datalen // 4) + mask_value[: datalen % 4], native_byteorder
        )
        return (int_data_value ^ int_mask_value).to_bytes(datalen, native_byteorder)


__all__ = [
    "ABNF",
    "continuous_frame",
    "frame_buffer",
    "STATUS_NORMAL",
    "STATUS_GOING_AWAY",
    "STATUS_PROTOCOL_ERROR",
    "STATUS_UNSUPPORTED_DATA_TYPE",
    "STATUS_STATUS_NOT_AVAILABLE",
    "STATUS_ABNORMAL_CLOSED",
    "STATUS_INVALID_PAYLOAD",
    "STATUS_POLICY_VIOLATION",
    "STATUS_MESSAGE_TOO_BIG",
    "STATUS_INVALID_EXTENSION",
    "STATUS_UNEXPECTED_CONDITION",
    "STATUS_BAD_GATEWAY",
    "STATUS_TLS_HANDSHAKE_ERROR",
]

# closing frame status codes.
STATUS_NORMAL = 1000
STATUS_GOING_AWAY = 1001
STATUS_PROTOCOL_ERROR = 1002
STATUS_UNSUPPORTED_DATA_TYPE = 1003
STATUS_STATUS_NOT_AVAILABLE = 1005
STATUS_ABNORMAL_CLOSED = 1006
STATUS_INVALID_PAYLOAD = 1007
STATUS_POLICY_VIOLATION = 1008
STATUS_MESSAGE_TOO_BIG = 1009
STATUS_INVALID_EXTENSION = 1010
STATUS_UNEXPECTED_CONDITION = 1011
STATUS_SERVICE_RESTART = 1012
STATUS_TRY_AGAIN_LATER = 1013
STATUS_BAD_GATEWAY = 1014
STATUS_TLS_HANDSHAKE_ERROR = 1015

VALID_CLOSE_STATUS = (
    STATUS_NORMAL,
    STATUS_GOING_AWAY,
    STATUS_PROTOCOL_ERROR,
    STATUS_UNSUPPORTED_DATA_TYPE,
    STATUS_INVALID_PAYLOAD,
    STATUS_POLICY_VIOLATION,
    STATUS_MESSAGE_TOO_BIG,
    STATUS_INVALID_EXTENSION,
    STATUS_UNEXPECTED_CONDITION,
    STATUS_SERVICE_RESTART,
    STATUS_TRY_AGAIN_LATER,
    STATUS_BAD_GATEWAY,
)


class ABNF:
    """
    ABNF frame class.
    See http://tools.ietf.org/html/rfc5234
    and http://tools.ietf.org/html/rfc6455#section-5.2
    """

    # operation code values.
    OPCODE_CONT = 0x0
    OPCODE_TEXT = 0x1
    OPCODE_BINARY = 0x2
    OPCODE_CLOSE = 0x8
    OPCODE_PING = 0x9
    OPCODE_PONG = 0xA

    # available operation code value tuple
    OPCODES = (
        OPCODE_CONT,
        OPCODE_TEXT,
        OPCODE_BINARY,
        OPCODE_CLOSE,
        OPCODE_PING,
        OPCODE_PONG,
    )

    # opcode human readable string
    OPCODE_MAP = {
        OPCODE_CONT: "cont",
        OPCODE_TEXT: "text",
        OPCODE_BINARY: "binary",
        OPCODE_CLOSE: "close",
        OPCODE_PING: "ping",
        OPCODE_PONG: "pong",
    }

    # data length threshold.
    LENGTH_7 = 0x7E
    LENGTH_16 = 1 << 16
    LENGTH_63 = 1 << 63

    def __init__(
        self,
        fin: int = 0,
        rsv1: int = 0,
        rsv2: int = 0,
        rsv3: int = 0,
        opcode: int = OPCODE_TEXT,
        mask_value: int = 1,
        data: Union[str, bytes, None] = "",
    ) -> None:
        """
        Constructor for ABNF. Please check RFC for arguments.
        """
        self.fin = fin
        self.rsv1 = rsv1
        self.rsv2 = rsv2
        self.rsv3 = rsv3
        self.opcode = opcode
        self.mask_value = mask_value
        if data is None:
            data = ""
        self.data = data
        self.get_mask_key = os.urandom

    def validate(self, skip_utf8_validation: bool = False) -> None:
        """
        Validate the ABNF frame.

        Parameters
        ----------
        skip_utf8_validation: skip utf8 validation.
        """
        if self.rsv1 or self.rsv2 or self.rsv3:
            raise WebSocketProtocolException("rsv is not implemented, yet")

        if self.opcode not in ABNF.OPCODES:
            raise WebSocketProtocolException("Invalid opcode %r", self.opcode)

        if self.opcode == ABNF.OPCODE_PING and not self.fin:
            raise WebSocketProtocolException("Invalid ping frame.")

        if self.opcode == ABNF.OPCODE_CLOSE:
            l = len(self.data)
            if not l:
                return
            if l == 1 or l >= 126:
                raise WebSocketProtocolException("Invalid close frame.")
            if l > 2 and not skip_utf8_validation and not validate_utf8(self.data[2:]):
                raise WebSocketProtocolException("Invalid close frame.")

            code = 256 * int(self.data[0]) + int(self.data[1])
            if not self._is_valid_close_status(code):
                raise WebSocketProtocolException("Invalid close opcode %r", code)

    @staticmethod
    def _is_valid_close_status(code: int) -> bool:
        return code in VALID_CLOSE_STATUS or (3000 <= code < 5000)

    def __str__(self) -> str:
        return f"fin={self.fin} opcode={self.opcode} data={self.data}"

    @staticmethod
    def create_frame(data: Union[bytes, str], opcode: int, fin: int = 1) -> "ABNF":
        """
        Create frame to send text, binary and other data.

        Parameters
        ----------
        data: str
            data to send. This is string value(byte array).
            If opcode is OPCODE_TEXT and this value is unicode,
            data value is converted into unicode string, automatically.
        opcode: int
            operation code. please see OPCODE_MAP.
        fin: int
            fin flag. if set to 0, create continue fragmentation.
        """
        if opcode == ABNF.OPCODE_TEXT and isinstance(data, str):
            data = data.encode("utf-8")
        # mask must be set if send data from client
        return ABNF(fin, 0, 0, 0, opcode, 1, data)

    def format(self) -> bytes:
        """
        Format this object to string(byte array) to send data to server.
        """
        if any(x not in (0, 1) for x in [self.fin, self.rsv1, self.rsv2, self.rsv3]):
            raise ValueError("not 0 or 1")
        if self.opcode not in ABNF.OPCODES:
            raise ValueError("Invalid OPCODE")
        length = len(self.data)
        if length >= ABNF.LENGTH_63:
            raise ValueError("data is too long")

        frame_header = chr(
            self.fin << 7
            | self.rsv1 << 6
            | self.rsv2 << 5
            | self.rsv3 << 4
            | self.opcode
        ).encode("latin-1")
        if length < ABNF.LENGTH_7:
            frame_header += chr(self.mask_value << 7 | length).encode("latin-1")
        elif length < ABNF.LENGTH_16:
            frame_header += chr(self.mask_value << 7 | 0x7E).encode("latin-1")
            frame_header += struct.pack("!H", length)
        else:
            frame_header += chr(self.mask_value << 7 | 0x7F).encode("latin-1")
            frame_header += struct.pack("!Q", length)

        if not self.mask_value:
            if isinstance(self.data, str):
                self.data = self.data.encode("utf-8")
            return frame_header + self.data
        mask_key = self.get_mask_key(4)
        return frame_header + self._get_masked(mask_key)

    def _get_masked(self, mask_key: Union[str, bytes]) -> bytes:
        s = ABNF.mask(mask_key, self.data)

        if isinstance(mask_key, str):
            mask_key = mask_key.encode("utf-8")

        return mask_key + s

    @staticmethod
    def mask(mask_key: Union[str, bytes], data: Union[str, bytes]) -> bytes:
        """
        Mask or unmask data. Just do xor for each byte

        Parameters
        ----------
        mask_key: bytes or str
            4 byte mask.
        data: bytes or str
            data to mask/unmask.
        """
        if data is None:
            data = ""

        if isinstance(mask_key, str):
            mask_key = mask_key.encode("latin-1")

        if isinstance(data, str):
            data = data.encode("latin-1")

        return _mask(array.array("B", mask_key), array.array("B", data))


class frame_buffer:
    _HEADER_MASK_INDEX = 5
    _HEADER_LENGTH_INDEX = 6

    def __init__(
        self, recv_fn: Callable[[int], int], skip_utf8_validation: bool
    ) -> None:
        self.recv = recv_fn
        self.skip_utf8_validation = skip_utf8_validation
        # Buffers over the packets from the layer beneath until desired amount
        # bytes of bytes are received.
        self.recv_buffer: list = []
        self.clear()
        self.lock = Lock()

    def clear(self) -> None:
        self.header: Optional[tuple] = None
        self.length: Optional[int] = None
        self.mask_value: Union[bytes, str, None] = None

    def has_received_header(self) -> bool:
        return self.header is None

    def recv_header(self) -> None:
        header = self.recv_strict(2)
        b1 = header[0]
        fin = b1 >> 7 & 1
        rsv1 = b1 >> 6 & 1
        rsv2 = b1 >> 5 & 1
        rsv3 = b1 >> 4 & 1
        opcode = b1 & 0xF
        b2 = header[1]
        has_mask = b2 >> 7 & 1
        length_bits = b2 & 0x7F

        self.header = (fin, rsv1, rsv2, rsv3, opcode, has_mask, length_bits)

    def has_mask(self) -> Union[bool, int]:
        if not self.header:
            return False
        header_val: int = self.header[frame_buffer._HEADER_MASK_INDEX]
        return header_val

    def has_received_length(self) -> bool:
        return self.length is None

    def recv_length(self) -> None:
        bits = self.header[frame_buffer._HEADER_LENGTH_INDEX]
        length_bits = bits & 0x7F
        if length_bits == 0x7E:
            v = self.recv_strict(2)
            self.length = struct.unpack("!H", v)[0]
        elif length_bits == 0x7F:
            v = self.recv_strict(8)
            self.length = struct.unpack("!Q", v)[0]
        else:
            self.length = length_bits

    def has_received_mask(self) -> bool:
        return self.mask_value is None

    def recv_mask(self) -> None:
        self.mask_value = self.recv_strict(4) if self.has_mask() else ""

    def recv_frame(self) -> ABNF:
        with self.lock:
            # Header
            if self.has_received_header():
                self.recv_header()
            (fin, rsv1, rsv2, rsv3, opcode, has_mask, _) = self.header

            # Frame length
            if self.has_received_length():
                self.recv_length()
            length = self.length

            # Mask
            if self.has_received_mask():
                self.recv_mask()
            mask_value = self.mask_value

            # Payload
            payload = self.recv_strict(length)
            if has_mask:
                payload = ABNF.mask(mask_value, payload)

            # Reset for next frame
            self.clear()

            frame = ABNF(fin, rsv1, rsv2, rsv3, opcode, has_mask, payload)
            frame.validate(self.skip_utf8_validation)

        return frame

    def recv_strict(self, bufsize: int) -> bytes:
        shortage = bufsize - sum(map(len, self.recv_buffer))
        while shortage > 0:
            # Limit buffer size that we pass to socket.recv() to avoid
            # fragmenting the heap -- the number of bytes recv() actually
            # reads is limited by socket buffer and is relatively small,
            # yet passing large numbers repeatedly causes lots of large
            # buffers allocated and then shrunk, which results in
            # fragmentation.
            bytes_ = self.recv(min(16384, shortage))
            self.recv_buffer.append(bytes_)
            shortage -= len(bytes_)

        unified = b"".join(self.recv_buffer)

        if shortage == 0:
            self.recv_buffer = []
            return unified
        else:
            self.recv_buffer = [unified[bufsize:]]
            return unified[:bufsize]


class continuous_frame:
    def __init__(self, fire_cont_frame: bool, skip_utf8_validation: bool) -> None:
        self.fire_cont_frame = fire_cont_frame
        self.skip_utf8_validation = skip_utf8_validation
        self.cont_data: Optional[list] = None
        self.recving_frames: Optional[int] = None

    def validate(self, frame: ABNF) -> None:
        if not self.recving_frames and frame.opcode == ABNF.OPCODE_CONT:
            raise WebSocketProtocolException("Illegal frame")
        if self.recving_frames and frame.opcode in (
            ABNF.OPCODE_TEXT,
            ABNF.OPCODE_BINARY,
        ):
            raise WebSocketProtocolException("Illegal frame")

    def add(self, frame: ABNF) -> None:
        if self.cont_data:
            self.cont_data[1] += frame.data
        else:
            if frame.opcode in (ABNF.OPCODE_TEXT, ABNF.OPCODE_BINARY):
                self.recving_frames = frame.opcode
            self.cont_data = [frame.opcode, frame.data]

        if frame.fin:
            self.recving_frames = None

    def is_fire(self, frame: ABNF) -> Union[bool, int]:
        return frame.fin or self.fire_cont_frame

    def extract(self, frame: ABNF) -> tuple:
        data = self.cont_data
        self.cont_data = None
        frame.data = data[1]
        if (
            not self.fire_cont_frame
            and data[0] == ABNF.OPCODE_TEXT
            and not self.skip_utf8_validation
            and not validate_utf8(frame.data)
        ):
            raise WebSocketPayloadException(f"cannot decode: {repr(frame.data)}")
        return data[0], frame

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\intervalmath\lib_interval.py ===
""" The module contains implemented functions for interval arithmetic."""
from functools import reduce

from sympy.plotting.intervalmath import interval
from sympy.external import import_module


def Abs(x):
    if isinstance(x, (int, float)):
        return interval(abs(x))
    elif isinstance(x, interval):
        if x.start < 0 and x.end > 0:
            return interval(0, max(abs(x.start), abs(x.end)), is_valid=x.is_valid)
        else:
            return interval(abs(x.start), abs(x.end))
    else:
        raise NotImplementedError

#Monotonic


def exp(x):
    """evaluates the exponential of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.exp(x), np.exp(x))
    elif isinstance(x, interval):
        return interval(np.exp(x.start), np.exp(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


#Monotonic
def log(x):
    """evaluates the natural logarithm of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.log(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-np.inf, np.inf, is_valid=x.is_valid)
        elif x.end <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        elif x.start <= 0:
            return interval(-np.inf, np.inf, is_valid=None)

        return interval(np.log(x.start), np.log(x.end))
    else:
        raise NotImplementedError


#Monotonic
def log10(x):
    """evaluates the logarithm to the base 10 of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.log10(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-np.inf, np.inf, is_valid=x.is_valid)
        elif x.end <= 0:
            return interval(-np.inf, np.inf, is_valid=False)
        elif x.start <= 0:
            return interval(-np.inf, np.inf, is_valid=None)
        return interval(np.log10(x.start), np.log10(x.end))
    else:
        raise NotImplementedError


#Monotonic
def atan(x):
    """evaluates the tan inverse of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.arctan(x))
    elif isinstance(x, interval):
        start = np.arctan(x.start)
        end = np.arctan(x.end)
        return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


#periodic
def sin(x):
    """evaluates the sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sin(x))
    elif isinstance(x, interval):
        if not x.is_valid:
            return interval(-1, 1, is_valid=x.is_valid)
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        start = min(np.sin(x.start), np.sin(x.end))
        end = max(np.sin(x.start), np.sin(x.end))
        if nb - na > 4:
            return interval(-1, 1, is_valid=x.is_valid)
        elif na == nb:
            return interval(start, end, is_valid=x.is_valid)
        else:
            if (na - 1) // 4 != (nb - 1) // 4:
                #sin has max
                end = 1
            if (na - 3) // 4 != (nb - 3) // 4:
                #sin has min
                start = -1
            return interval(start, end)
    else:
        raise NotImplementedError


#periodic
def cos(x):
    """Evaluates the cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sin(x))
    elif isinstance(x, interval):
        if not (np.isfinite(x.start) and np.isfinite(x.end)):
            return interval(-1, 1, is_valid=x.is_valid)
        na, __ = divmod(x.start, np.pi / 2.0)
        nb, __ = divmod(x.end, np.pi / 2.0)
        start = min(np.cos(x.start), np.cos(x.end))
        end = max(np.cos(x.start), np.cos(x.end))
        if nb - na > 4:
            #differ more than 2*pi
            return interval(-1, 1, is_valid=x.is_valid)
        elif na == nb:
            #in the same quadarant
            return interval(start, end, is_valid=x.is_valid)
        else:
            if (na) // 4 != (nb) // 4:
                #cos has max
                end = 1
            if (na - 2) // 4 != (nb - 2) // 4:
                #cos has min
                start = -1
            return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


def tan(x):
    """Evaluates the tan of an interval"""
    return sin(x) / cos(x)


#Monotonic
def sqrt(x):
    """Evaluates the square root of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if x > 0:
            return interval(np.sqrt(x))
        else:
            return interval(-np.inf, np.inf, is_valid=False)
    elif isinstance(x, interval):
        #Outside the domain
        if x.end < 0:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partially outside the domain
        elif x.start < 0:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            return interval(np.sqrt(x.start), np.sqrt(x.end),
                    is_valid=x.is_valid)
    else:
        raise NotImplementedError


def imin(*args):
    """Evaluates the minimum of a list of intervals"""
    np = import_module('numpy')
    if not all(isinstance(arg, (int, float, interval)) for arg in args):
        return NotImplementedError
    else:
        new_args = [a for a in args if isinstance(a, (int, float))
                    or a.is_valid]
        if len(new_args) == 0:
            if all(a.is_valid is False for a in args):
                return interval(-np.inf, np.inf, is_valid=False)
            else:
                return interval(-np.inf, np.inf, is_valid=None)
        start_array = [a if isinstance(a, (int, float)) else a.start
                       for a in new_args]

        end_array = [a if isinstance(a, (int, float)) else a.end
                     for a in new_args]
        return interval(min(start_array), min(end_array))


def imax(*args):
    """Evaluates the maximum of a list of intervals"""
    np = import_module('numpy')
    if not all(isinstance(arg, (int, float, interval)) for arg in args):
        return NotImplementedError
    else:
        new_args = [a for a in args if isinstance(a, (int, float))
                    or a.is_valid]
        if len(new_args) == 0:
            if all(a.is_valid is False for a in args):
                return interval(-np.inf, np.inf, is_valid=False)
            else:
                return interval(-np.inf, np.inf, is_valid=None)
        start_array = [a if isinstance(a, (int, float)) else a.start
                       for a in new_args]

        end_array = [a if isinstance(a, (int, float)) else a.end
                     for a in new_args]

        return interval(max(start_array), max(end_array))


#Monotonic
def sinh(x):
    """Evaluates the hyperbolic sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.sinh(x), np.sinh(x))
    elif isinstance(x, interval):
        return interval(np.sinh(x.start), np.sinh(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


def cosh(x):
    """Evaluates the hyperbolic cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.cosh(x), np.cosh(x))
    elif isinstance(x, interval):
        #both signs
        if x.start < 0 and x.end > 0:
            end = max(np.cosh(x.start), np.cosh(x.end))
            return interval(1, end, is_valid=x.is_valid)
        else:
            #Monotonic
            start = np.cosh(x.start)
            end = np.cosh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        raise NotImplementedError


#Monotonic
def tanh(x):
    """Evaluates the hyperbolic tan of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.tanh(x), np.tanh(x))
    elif isinstance(x, interval):
        return interval(np.tanh(x.start), np.tanh(x.end), is_valid=x.is_valid)
    else:
        raise NotImplementedError


def asin(x):
    """Evaluates the inverse sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        #Outside the domain
        if abs(x) > 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arcsin(x), np.arcsin(x))
    elif isinstance(x, interval):
        #Outside the domain
        if x.is_valid is False or x.start > 1 or x.end < -1:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partially outside the domain
        elif x.start < -1 or x.end > 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arcsin(x.start)
            end = np.arcsin(x.end)
            return interval(start, end, is_valid=x.is_valid)


def acos(x):
    """Evaluates the inverse cos of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        if abs(x) > 1:
            #Outside the domain
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arccos(x), np.arccos(x))
    elif isinstance(x, interval):
        #Outside the domain
        if x.is_valid is False or x.start > 1 or x.end < -1:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partially outside the domain
        elif x.start < -1 or x.end > 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arccos(x.start)
            end = np.arccos(x.end)
            return interval(start, end, is_valid=x.is_valid)


def ceil(x):
    """Evaluates the ceiling of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.ceil(x))
    elif isinstance(x, interval):
        if x.is_valid is False:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            start = np.ceil(x.start)
            end = np.ceil(x.end)
            #Continuous over the interval
            if start == end:
                return interval(start, end, is_valid=x.is_valid)
            else:
                #Not continuous over the interval
                return interval(start, end, is_valid=None)
    else:
        return NotImplementedError


def floor(x):
    """Evaluates the floor of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.floor(x))
    elif isinstance(x, interval):
        if x.is_valid is False:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            start = np.floor(x.start)
            end = np.floor(x.end)
            #continuous over the argument
            if start == end:
                return interval(start, end, is_valid=x.is_valid)
            else:
                #not continuous over the interval
                return interval(start, end, is_valid=None)
    else:
        return NotImplementedError


def acosh(x):
    """Evaluates the inverse hyperbolic cosine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        #Outside the domain
        if x < 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arccosh(x))
    elif isinstance(x, interval):
        #Outside the domain
        if x.end < 1:
            return interval(-np.inf, np.inf, is_valid=False)
        #Partly outside the domain
        elif x.start < 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arccosh(x.start)
            end = np.arccosh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        return NotImplementedError


#Monotonic
def asinh(x):
    """Evaluates the inverse hyperbolic sine of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        return interval(np.arcsinh(x))
    elif isinstance(x, interval):
        start = np.arcsinh(x.start)
        end = np.arcsinh(x.end)
        return interval(start, end, is_valid=x.is_valid)
    else:
        return NotImplementedError


def atanh(x):
    """Evaluates the inverse hyperbolic tangent of an interval"""
    np = import_module('numpy')
    if isinstance(x, (int, float)):
        #Outside the domain
        if abs(x) >= 1:
            return interval(-np.inf, np.inf, is_valid=False)
        else:
            return interval(np.arctanh(x))
    elif isinstance(x, interval):
        #outside the domain
        if x.is_valid is False or x.start >= 1 or x.end <= -1:
            return interval(-np.inf, np.inf, is_valid=False)
        #partly outside the domain
        elif x.start <= -1 or x.end >= 1:
            return interval(-np.inf, np.inf, is_valid=None)
        else:
            start = np.arctanh(x.start)
            end = np.arctanh(x.end)
            return interval(start, end, is_valid=x.is_valid)
    else:
        return NotImplementedError


#Three valued logic for interval plotting.

def And(*args):
    """Defines the three valued ``And`` behaviour for a 2-tuple of
     three valued logic values"""
    def reduce_and(cmp_intervala, cmp_intervalb):
        if cmp_intervala[0] is False or cmp_intervalb[0] is False:
            first = False
        elif cmp_intervala[0] is None or cmp_intervalb[0] is None:
            first = None
        else:
            first = True
        if cmp_intervala[1] is False or cmp_intervalb[1] is False:
            second = False
        elif cmp_intervala[1] is None or cmp_intervalb[1] is None:
            second = None
        else:
            second = True
        return (first, second)
    return reduce(reduce_and, args)


def Or(*args):
    """Defines the three valued ``Or`` behaviour for a 2-tuple of
     three valued logic values"""
    def reduce_or(cmp_intervala, cmp_intervalb):
        if cmp_intervala[0] is True or cmp_intervalb[0] is True:
            first = True
        elif cmp_intervala[0] is None or cmp_intervalb[0] is None:
            first = None
        else:
            first = False

        if cmp_intervala[1] is True or cmp_intervalb[1] is True:
            second = True
        elif cmp_intervala[1] is None or cmp_intervalb[1] is None:
            second = None
        else:
            second = False
        return (first, second)
    return reduce(reduce_or, args)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\ntheory\qs.py ===
from math import exp, log
from sympy.core.random import _randint
from sympy.external.gmpy import bit_scan1, gcd, invert, sqrt as isqrt
from sympy.ntheory.factor_ import _perfect_power
from sympy.ntheory.primetest import isprime
from sympy.ntheory.residue_ntheory import _sqrt_mod_prime_power


class SievePolynomial:
    def __init__(self, a, b, N):
        """This class denotes the sieve polynomial.
        Provide methods to compute `(a*x + b)**2 - N` and
        `a*x + b` when given `x`.

        Parameters
        ==========

        a : parameter of the sieve polynomial
        b : parameter of the sieve polynomial
        N : number to be factored

        """
        self.a = a
        self.b = b
        self.a2 = a**2
        self.ab = 2*a*b
        self.b2 = b**2 - N

    def eval_u(self, x):
        return self.a*x + self.b

    def eval_v(self, x):
        return (self.a2*x + self.ab)*x + self.b2


class FactorBaseElem:
    """This class stores an element of the `factor_base`.
    """
    def __init__(self, prime, tmem_p, log_p):
        """
        Initialization of factor_base_elem.

        Parameters
        ==========

        prime : prime number of the factor_base
        tmem_p : Integer square root of x**2 = n mod prime
        log_p : Compute Natural Logarithm of the prime
        """
        self.prime = prime
        self.tmem_p = tmem_p
        self.log_p = log_p
        # `soln1` and `soln2` are solutions to
        # the equation `(a*x + b)**2 - N = 0 (mod p)`.
        self.soln1 = None
        self.soln2 = None
        self.b_ainv = None


def _generate_factor_base(prime_bound, n):
    """Generate `factor_base` for Quadratic Sieve. The `factor_base`
    consists of all the points whose ``legendre_symbol(n, p) == 1``
    and ``p < num_primes``. Along with the prime `factor_base` also stores
    natural logarithm of prime and the residue n modulo p.
    It also returns the of primes numbers in the `factor_base` which are
    close to 1000 and 5000.

    Parameters
    ==========

    prime_bound : upper prime bound of the factor_base
    n : integer to be factored
    """
    from sympy.ntheory.generate import sieve
    factor_base = []
    idx_1000, idx_5000 = None, None
    for prime in sieve.primerange(1, prime_bound):
        if pow(n, (prime - 1) // 2, prime) == 1:
            if prime > 1000 and idx_1000 is None:
                idx_1000 = len(factor_base) - 1
            if prime > 5000 and idx_5000 is None:
                idx_5000 = len(factor_base) - 1
            residue = _sqrt_mod_prime_power(n, prime, 1)[0]
            log_p = round(log(prime)*2**10)
            factor_base.append(FactorBaseElem(prime, residue, log_p))
    return idx_1000, idx_5000, factor_base


def _generate_polynomial(N, M, factor_base, idx_1000, idx_5000, randint):
    """ Generate sieve polynomials indefinitely.
    Information such as `soln1` in the `factor_base` associated with
    the polynomial is modified in place.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    idx_1000 : index of prime number in the factor_base near 1000
    idx_5000 : index of prime number in the factor_base near to 5000
    randint : A callable that takes two integers (a, b) and returns a random integer
              n such that a <= n <= b, similar to `random.randint`.
    """
    approx_val = log(2*N)/2 - log(M)
    start = idx_1000 or 0
    end = idx_5000 or (len(factor_base) - 1)
    while True:
        # Choose `a` that is close to `sqrt(2*N) / M`
        best_a, best_q, best_ratio = None, None, None
        for _ in range(50):
            a = 1
            q = []
            while log(a) < approx_val:
                rand_p = 0
                while(rand_p == 0 or rand_p in q):
                    rand_p = randint(start, end)
                p = factor_base[rand_p].prime
                a *= p
                q.append(rand_p)
            ratio = exp(log(a) - approx_val)
            if best_ratio is None or abs(ratio - 1) < abs(best_ratio - 1):
                best_q = q
                best_a = a
                best_ratio = ratio

        # Set `b` using the Chinese remainder theorem
        a = best_a
        q = best_q
        B = []
        for val in q:
            q_l = factor_base[val].prime
            gamma = factor_base[val].tmem_p * invert(a // q_l, q_l) % q_l
            if 2*gamma > q_l:
                gamma = q_l - gamma
            B.append(a//q_l*gamma)
        b = sum(B)
        g = SievePolynomial(a, b, N)
        for fb in factor_base:
            if a % fb.prime == 0:
                fb.soln1 = None
                continue
            a_inv = invert(a, fb.prime)
            fb.b_ainv = [2*b_elem*a_inv % fb.prime for b_elem in B]
            fb.soln1 = (a_inv*(fb.tmem_p - b)) % fb.prime
            fb.soln2 = (a_inv*(-fb.tmem_p - b)) % fb.prime
        yield g

        # Update `b` with Gray code
        for i in range(1, 2**(len(B)-1)):
            v = bit_scan1(i)
            neg_pow = 2*((i >> (v + 1)) % 2) - 1
            b = g.b + 2*neg_pow*B[v]
            a = g.a
            g = SievePolynomial(a, b, N)
            for fb in factor_base:
                if fb.soln1 is None:
                    continue
                fb.soln1 = (fb.soln1 - neg_pow*fb.b_ainv[v]) % fb.prime
                fb.soln2 = (fb.soln2 - neg_pow*fb.b_ainv[v]) % fb.prime
            yield g


def _gen_sieve_array(M, factor_base):
    """Sieve Stage of the Quadratic Sieve. For every prime in the factor_base
    that does not divide the coefficient `a` we add log_p over the sieve_array
    such that ``-M <= soln1 + i*p <=  M`` and ``-M <= soln2 + i*p <=  M`` where `i`
    is an integer. When p = 2 then log_p is only added using
    ``-M <= soln1 + i*p <=  M``.

    Parameters
    ==========

    M : sieve interval
    factor_base : factor_base primes
    """
    sieve_array = [0]*(2*M + 1)
    for factor in factor_base:
        if factor.soln1 is None: #The prime does not divides a
            continue
        for idx in range((M + factor.soln1) % factor.prime, 2*M, factor.prime):
            sieve_array[idx] += factor.log_p
        if factor.prime == 2:
            continue
        #if prime is 2 then sieve only with soln_1_p
        for idx in range((M + factor.soln2) % factor.prime, 2*M, factor.prime):
            sieve_array[idx] += factor.log_p
    return sieve_array


def _check_smoothness(num, factor_base):
    r""" Check if `num` is smooth with respect to the given `factor_base`
    and compute its factorization vector.

    Parameters
    ==========

    num : integer whose smootheness is to be checked
    factor_base : factor_base primes
    """
    if num < 0:
        num *= -1
        vec = 1
    else:
        vec = 0
    for i, fb in enumerate(factor_base, 1):
        if num % fb.prime:
            continue
        e = 1
        num //= fb.prime
        while num % fb.prime == 0:
            e += 1
            num //= fb.prime
        if e % 2:
            vec += 1 << i
    return vec, num


def _trial_division_stage(N, M, factor_base, sieve_array, sieve_poly, partial_relations, ERROR_TERM):
    """Trial division stage. Here we trial divide the values generetated
    by sieve_poly in the sieve interval and if it is a smooth number then
    it is stored in `smooth_relations`. Moreover, if we find two partial relations
    with same large prime then they are combined to form a smooth relation.
    First we iterate over sieve array and look for values which are greater
    than accumulated_val, as these values have a high chance of being smooth
    number. Then using these values we find smooth relations.
    In general, let ``t**2 = u*p modN`` and ``r**2 = v*p modN`` be two partial relations
    with the same large prime p. Then they can be combined ``(t*r/p)**2 = u*v modN``
    to form a smooth relation.

    Parameters
    ==========

    N : Number to be factored
    M : sieve interval
    factor_base : factor_base primes
    sieve_array : stores log_p values
    sieve_poly : polynomial from which we find smooth relations
    partial_relations : stores partial relations with one large prime
    ERROR_TERM : error term for accumulated_val
    """
    accumulated_val = (log(M) + log(N)/2 - ERROR_TERM) * 2**10
    smooth_relations = []
    proper_factor = set()
    partial_relation_upper_bound = 128*factor_base[-1].prime
    for x, val in enumerate(sieve_array, -M):
        if val < accumulated_val:
            continue
        v = sieve_poly.eval_v(x)
        vec, num = _check_smoothness(v, factor_base)
        if num == 1:
            smooth_relations.append((sieve_poly.eval_u(x), v, vec))
        elif num < partial_relation_upper_bound and isprime(num):
            if N % num == 0:
                proper_factor.add(num)
                continue
            u = sieve_poly.eval_u(x)
            if num in partial_relations:
                u_prev, v_prev, vec_prev = partial_relations.pop(num)
                u = u*u_prev*invert(num, N) % N
                v = v*v_prev // num**2
                vec ^= vec_prev
                smooth_relations.append((u, v, vec))
            else:
                partial_relations[num] = (u, v, vec)
    return smooth_relations, proper_factor


def _find_factor(N, smooth_relations, col):
    """ Finds proper factor of N using fast gaussian reduction for modulo 2 matrix.

    Parameters
    ==========

    N : Number to be factored
    smooth_relations : Smooth relations vectors matrix
    col : Number of columns in the matrix

    Reference
    ==========

    .. [1] A fast algorithm for gaussian elimination over GF(2) and
    its implementation on the GAPP. Cetin K.Koc, Sarath N.Arachchige
    """
    matrix = [s_relation[2] for s_relation in smooth_relations]
    row = len(matrix)
    mark = [False] * row
    for pos in range(col):
        m = 1 << pos
        for i in range(row):
            if p := matrix[i] & m:
                add_col = p ^ matrix[i]
                matrix[i] = m
                mark[i] = True
                for j in range(i + 1, row):
                    if matrix[j] & m:
                        matrix[j] ^= add_col
                break

    for m, mat, rel in zip(mark, matrix, smooth_relations):
        if m:
            continue
        u, v = rel[0], rel[1]
        for m1, mat1, rel1 in zip(mark, matrix, smooth_relations):
            if m1 and mat & mat1:
                u *= rel1[0]
                v *= rel1[1]
        # assert is_square(v)
        v = isqrt(v)
        if 1 < (g := gcd(u - v, N)) < N:
            yield g


def qs(N, prime_bound, M, ERROR_TERM=25, seed=1234):
    """Performs factorization using Self-Initializing Quadratic Sieve.
    In SIQS, let N be a number to be factored, and this N should not be a
    perfect power. If we find two integers such that ``X**2 = Y**2 modN`` and
    ``X != +-Y modN``, then `gcd(X + Y, N)` will reveal a proper factor of N.
    In order to find these integers X and Y we try to find relations of form
    t**2 = u modN where u is a product of small primes. If we have enough of
    these relations then we can form ``(t1*t2...ti)**2 = u1*u2...ui modN`` such that
    the right hand side is a square, thus we found a relation of ``X**2 = Y**2 modN``.

    Here, several optimizations are done like using multiple polynomials for
    sieving, fast changing between polynomials and using partial relations.
    The use of partial relations can speeds up the factoring by 2 times.

    Parameters
    ==========

    N : Number to be Factored
    prime_bound : upper bound for primes in the factor base
    M : Sieve Interval
    ERROR_TERM : Error term for checking smoothness
    seed : seed of random number generator

    Returns
    =======

    set(int) : A set of factors of N without considering multiplicity.
               Returns ``{N}`` if factorization fails.

    Examples
    ========

    >>> from sympy.ntheory import qs
    >>> qs(25645121643901801, 2000, 10000)
    {5394769, 4753701529}
    >>> qs(9804659461513846513, 2000, 10000)
    {4641991, 2112166839943}

    See Also
    ========

    qs_factor

    References
    ==========

    .. [1] https://pdfs.semanticscholar.org/5c52/8a975c1405bd35c65993abf5a4edb667c1db.pdf
    .. [2] https://www.rieselprime.de/ziki/Self-initializing_quadratic_sieve
    """
    return set(qs_factor(N, prime_bound, M, ERROR_TERM, seed))


def qs_factor(N, prime_bound, M, ERROR_TERM=25, seed=1234):
    """ Performs factorization using Self-Initializing Quadratic Sieve.

    Parameters
    ==========

    N : Number to be Factored
    prime_bound : upper bound for primes in the factor base
    M : Sieve Interval
    ERROR_TERM : Error term for checking smoothness
    seed : seed of random number generator

    Returns
    =======

    dict[int, int] : Factors of N.
                     Returns ``{N: 1}`` if factorization fails.
                     Note that the key is not always a prime number.

    Examples
    ========

    >>> from sympy.ntheory import qs_factor
    >>> qs_factor(1009 * 100003, 2000, 10000)
    {1009: 1, 100003: 1}

    See Also
    ========

    qs

    """
    if N < 2:
        raise ValueError("N should be greater than 1")
    factors = {}
    smooth_relations = []
    partial_relations = {}
    # Eliminate the possibility of even numbers,
    # prime numbers, and perfect powers.
    if N % 2 == 0:
        e = 1
        N //= 2
        while N % 2 == 0:
            N //= 2
            e += 1
        factors[2] = e
    if isprime(N):
        factors[N] = 1
        return factors
    if result := _perfect_power(N, 3):
        n, e = result
        factors[n] = e
        return factors
    N_copy = N
    randint = _randint(seed)
    idx_1000, idx_5000, factor_base = _generate_factor_base(prime_bound, N)
    threshold = len(factor_base) * 105//100
    for g in _generate_polynomial(N, M, factor_base, idx_1000, idx_5000, randint):
        sieve_array = _gen_sieve_array(M, factor_base)
        s_rel, p_f = _trial_division_stage(N, M, factor_base, sieve_array, g, partial_relations, ERROR_TERM)
        smooth_relations += s_rel
        for p in p_f:
            if N_copy % p:
                continue
            e = 1
            N_copy //= p
            while N_copy % p == 0:
                N_copy //= p
                e += 1
            factors[p] = e
        if threshold <= len(smooth_relations):
            break

    for factor in _find_factor(N, smooth_relations, len(factor_base) + 1):
        if N_copy % factor == 0:
            e = 1
            N_copy //= factor
            while N_copy % factor == 0:
                N_copy //= factor
                e += 1
            factors[factor] = e
            if N_copy == 1 or isprime(N_copy):
                break
    if N_copy != 1:
        factors[N_copy] = 1
    return factors

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\instrumentation.py ===
# ext/instrumentation.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

"""Extensible class instrumentation.

The :mod:`sqlalchemy.ext.instrumentation` package provides for alternate
systems of class instrumentation within the ORM.  Class instrumentation
refers to how the ORM places attributes on the class which maintain
data and track changes to that data, as well as event hooks installed
on the class.

.. note::
    The extension package is provided for the benefit of integration
    with other object management packages, which already perform
    their own instrumentation.  It is not intended for general use.

For examples of how the instrumentation extension is used,
see the example :ref:`examples_instrumentation`.

"""
import weakref

from .. import util
from ..orm import attributes
from ..orm import base as orm_base
from ..orm import collections
from ..orm import exc as orm_exc
from ..orm import instrumentation as orm_instrumentation
from ..orm import util as orm_util
from ..orm.instrumentation import _default_dict_getter
from ..orm.instrumentation import _default_manager_getter
from ..orm.instrumentation import _default_opt_manager_getter
from ..orm.instrumentation import _default_state_getter
from ..orm.instrumentation import ClassManager
from ..orm.instrumentation import InstrumentationFactory


INSTRUMENTATION_MANAGER = "__sa_instrumentation_manager__"
"""Attribute, elects custom instrumentation when present on a mapped class.

Allows a class to specify a slightly or wildly different technique for
tracking changes made to mapped attributes and collections.

Only one instrumentation implementation is allowed in a given object
inheritance hierarchy.

The value of this attribute must be a callable and will be passed a class
object.  The callable must return one of:

  - An instance of an :class:`.InstrumentationManager` or subclass
  - An object implementing all or some of InstrumentationManager (TODO)
  - A dictionary of callables, implementing all or some of the above (TODO)
  - An instance of a :class:`.ClassManager` or subclass

This attribute is consulted by SQLAlchemy instrumentation
resolution, once the :mod:`sqlalchemy.ext.instrumentation` module
has been imported.  If custom finders are installed in the global
instrumentation_finders list, they may or may not choose to honor this
attribute.

"""


def find_native_user_instrumentation_hook(cls):
    """Find user-specified instrumentation management for a class."""
    return getattr(cls, INSTRUMENTATION_MANAGER, None)


instrumentation_finders = [find_native_user_instrumentation_hook]
"""An extensible sequence of callables which return instrumentation
implementations

When a class is registered, each callable will be passed a class object.
If None is returned, the
next finder in the sequence is consulted.  Otherwise the return must be an
instrumentation factory that follows the same guidelines as
sqlalchemy.ext.instrumentation.INSTRUMENTATION_MANAGER.

By default, the only finder is find_native_user_instrumentation_hook, which
searches for INSTRUMENTATION_MANAGER.  If all finders return None, standard
ClassManager instrumentation is used.

"""


class ExtendedInstrumentationRegistry(InstrumentationFactory):
    """Extends :class:`.InstrumentationFactory` with additional
    bookkeeping, to accommodate multiple types of
    class managers.

    """

    _manager_finders = weakref.WeakKeyDictionary()
    _state_finders = weakref.WeakKeyDictionary()
    _dict_finders = weakref.WeakKeyDictionary()
    _extended = False

    def _locate_extended_factory(self, class_):
        for finder in instrumentation_finders:
            factory = finder(class_)
            if factory is not None:
                manager = self._extended_class_manager(class_, factory)
                return manager, factory
        else:
            return None, None

    def _check_conflicts(self, class_, factory):
        existing_factories = self._collect_management_factories_for(
            class_
        ).difference([factory])
        if existing_factories:
            raise TypeError(
                "multiple instrumentation implementations specified "
                "in %s inheritance hierarchy: %r"
                % (class_.__name__, list(existing_factories))
            )

    def _extended_class_manager(self, class_, factory):
        manager = factory(class_)
        if not isinstance(manager, ClassManager):
            manager = _ClassInstrumentationAdapter(class_, manager)

        if factory != ClassManager and not self._extended:
            # somebody invoked a custom ClassManager.
            # reinstall global "getter" functions with the more
            # expensive ones.
            self._extended = True
            _install_instrumented_lookups()

        self._manager_finders[class_] = manager.manager_getter()
        self._state_finders[class_] = manager.state_getter()
        self._dict_finders[class_] = manager.dict_getter()
        return manager

    def _collect_management_factories_for(self, cls):
        """Return a collection of factories in play or specified for a
        hierarchy.

        Traverses the entire inheritance graph of a cls and returns a
        collection of instrumentation factories for those classes. Factories
        are extracted from active ClassManagers, if available, otherwise
        instrumentation_finders is consulted.

        """
        hierarchy = util.class_hierarchy(cls)
        factories = set()
        for member in hierarchy:
            manager = self.opt_manager_of_class(member)
            if manager is not None:
                factories.add(manager.factory)
            else:
                for finder in instrumentation_finders:
                    factory = finder(member)
                    if factory is not None:
                        break
                else:
                    factory = None
                factories.add(factory)
        factories.discard(None)
        return factories

    def unregister(self, class_):
        super().unregister(class_)
        if class_ in self._manager_finders:
            del self._manager_finders[class_]
            del self._state_finders[class_]
            del self._dict_finders[class_]

    def opt_manager_of_class(self, cls):
        try:
            finder = self._manager_finders.get(
                cls, _default_opt_manager_getter
            )
        except TypeError:
            # due to weakref lookup on invalid object
            return None
        else:
            return finder(cls)

    def manager_of_class(self, cls):
        try:
            finder = self._manager_finders.get(cls, _default_manager_getter)
        except TypeError:
            # due to weakref lookup on invalid object
            raise orm_exc.UnmappedClassError(
                cls, f"Can't locate an instrumentation manager for class {cls}"
            )
        else:
            manager = finder(cls)
            if manager is None:
                raise orm_exc.UnmappedClassError(
                    cls,
                    f"Can't locate an instrumentation manager for class {cls}",
                )
            return manager

    def state_of(self, instance):
        if instance is None:
            raise AttributeError("None has no persistent state.")
        return self._state_finders.get(
            instance.__class__, _default_state_getter
        )(instance)

    def dict_of(self, instance):
        if instance is None:
            raise AttributeError("None has no persistent state.")
        return self._dict_finders.get(
            instance.__class__, _default_dict_getter
        )(instance)


orm_instrumentation._instrumentation_factory = _instrumentation_factory = (
    ExtendedInstrumentationRegistry()
)
orm_instrumentation.instrumentation_finders = instrumentation_finders


class InstrumentationManager:
    """User-defined class instrumentation extension.

    :class:`.InstrumentationManager` can be subclassed in order
    to change
    how class instrumentation proceeds. This class exists for
    the purposes of integration with other object management
    frameworks which would like to entirely modify the
    instrumentation methodology of the ORM, and is not intended
    for regular usage.  For interception of class instrumentation
    events, see :class:`.InstrumentationEvents`.

    The API for this class should be considered as semi-stable,
    and may change slightly with new releases.

    """

    # r4361 added a mandatory (cls) constructor to this interface.
    # given that, perhaps class_ should be dropped from all of these
    # signatures.

    def __init__(self, class_):
        pass

    def manage(self, class_, manager):
        setattr(class_, "_default_class_manager", manager)

    def unregister(self, class_, manager):
        delattr(class_, "_default_class_manager")

    def manager_getter(self, class_):
        def get(cls):
            return cls._default_class_manager

        return get

    def instrument_attribute(self, class_, key, inst):
        pass

    def post_configure_attribute(self, class_, key, inst):
        pass

    def install_descriptor(self, class_, key, inst):
        setattr(class_, key, inst)

    def uninstall_descriptor(self, class_, key):
        delattr(class_, key)

    def install_member(self, class_, key, implementation):
        setattr(class_, key, implementation)

    def uninstall_member(self, class_, key):
        delattr(class_, key)

    def instrument_collection_class(self, class_, key, collection_class):
        return collections.prepare_instrumentation(collection_class)

    def get_instance_dict(self, class_, instance):
        return instance.__dict__

    def initialize_instance_dict(self, class_, instance):
        pass

    def install_state(self, class_, instance, state):
        setattr(instance, "_default_state", state)

    def remove_state(self, class_, instance):
        delattr(instance, "_default_state")

    def state_getter(self, class_):
        return lambda instance: getattr(instance, "_default_state")

    def dict_getter(self, class_):
        return lambda inst: self.get_instance_dict(class_, inst)


class _ClassInstrumentationAdapter(ClassManager):
    """Adapts a user-defined InstrumentationManager to a ClassManager."""

    def __init__(self, class_, override):
        self._adapted = override
        self._get_state = self._adapted.state_getter(class_)
        self._get_dict = self._adapted.dict_getter(class_)

        ClassManager.__init__(self, class_)

    def manage(self):
        self._adapted.manage(self.class_, self)

    def unregister(self):
        self._adapted.unregister(self.class_, self)

    def manager_getter(self):
        return self._adapted.manager_getter(self.class_)

    def instrument_attribute(self, key, inst, propagated=False):
        ClassManager.instrument_attribute(self, key, inst, propagated)
        if not propagated:
            self._adapted.instrument_attribute(self.class_, key, inst)

    def post_configure_attribute(self, key):
        super().post_configure_attribute(key)
        self._adapted.post_configure_attribute(self.class_, key, self[key])

    def install_descriptor(self, key, inst):
        self._adapted.install_descriptor(self.class_, key, inst)

    def uninstall_descriptor(self, key):
        self._adapted.uninstall_descriptor(self.class_, key)

    def install_member(self, key, implementation):
        self._adapted.install_member(self.class_, key, implementation)

    def uninstall_member(self, key):
        self._adapted.uninstall_member(self.class_, key)

    def instrument_collection_class(self, key, collection_class):
        return self._adapted.instrument_collection_class(
            self.class_, key, collection_class
        )

    def initialize_collection(self, key, state, factory):
        delegate = getattr(self._adapted, "initialize_collection", None)
        if delegate:
            return delegate(key, state, factory)
        else:
            return ClassManager.initialize_collection(
                self, key, state, factory
            )

    def new_instance(self, state=None):
        instance = self.class_.__new__(self.class_)
        self.setup_instance(instance, state)
        return instance

    def _new_state_if_none(self, instance):
        """Install a default InstanceState if none is present.

        A private convenience method used by the __init__ decorator.
        """
        if self.has_state(instance):
            return False
        else:
            return self.setup_instance(instance)

    def setup_instance(self, instance, state=None):
        self._adapted.initialize_instance_dict(self.class_, instance)

        if state is None:
            state = self._state_constructor(instance, self)

        # the given instance is assumed to have no state
        self._adapted.install_state(self.class_, instance, state)
        return state

    def teardown_instance(self, instance):
        self._adapted.remove_state(self.class_, instance)

    def has_state(self, instance):
        try:
            self._get_state(instance)
        except orm_exc.NO_STATE:
            return False
        else:
            return True

    def state_getter(self):
        return self._get_state

    def dict_getter(self):
        return self._get_dict


def _install_instrumented_lookups():
    """Replace global class/object management functions
    with ExtendedInstrumentationRegistry implementations, which
    allow multiple types of class managers to be present,
    at the cost of performance.

    This function is called only by ExtendedInstrumentationRegistry
    and unit tests specific to this behavior.

    The _reinstall_default_lookups() function can be called
    after this one to re-establish the default functions.

    """
    _install_lookups(
        dict(
            instance_state=_instrumentation_factory.state_of,
            instance_dict=_instrumentation_factory.dict_of,
            manager_of_class=_instrumentation_factory.manager_of_class,
            opt_manager_of_class=_instrumentation_factory.opt_manager_of_class,
        )
    )


def _reinstall_default_lookups():
    """Restore simplified lookups."""
    _install_lookups(
        dict(
            instance_state=_default_state_getter,
            instance_dict=_default_dict_getter,
            manager_of_class=_default_manager_getter,
            opt_manager_of_class=_default_opt_manager_getter,
        )
    )
    _instrumentation_factory._extended = False


def _install_lookups(lookups):
    global instance_state, instance_dict
    global manager_of_class, opt_manager_of_class
    instance_state = lookups["instance_state"]
    instance_dict = lookups["instance_dict"]
    manager_of_class = lookups["manager_of_class"]
    opt_manager_of_class = lookups["opt_manager_of_class"]
    orm_base.instance_state = attributes.instance_state = (
        orm_instrumentation.instance_state
    ) = instance_state
    orm_base.instance_dict = attributes.instance_dict = (
        orm_instrumentation.instance_dict
    ) = instance_dict
    orm_base.manager_of_class = attributes.manager_of_class = (
        orm_instrumentation.manager_of_class
    ) = manager_of_class
    orm_base.opt_manager_of_class = orm_util.opt_manager_of_class = (
        attributes.opt_manager_of_class
    ) = orm_instrumentation.opt_manager_of_class = opt_manager_of_class

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\debug\tbtools.py ===
from __future__ import annotations

import itertools
import linecache
import os
import re
import sys
import sysconfig
import traceback
import typing as t

from markupsafe import escape

from ..utils import cached_property
from .console import Console

HEADER = """\
<!doctype html>
<html lang=en>
  <head>
    <title>%(title)s // Werkzeug Debugger</title>
    <link rel="stylesheet" href="?__debugger__=yes&amp;cmd=resource&amp;f=style.css">
    <link rel="shortcut icon"
        href="?__debugger__=yes&amp;cmd=resource&amp;f=console.png">
    <script src="?__debugger__=yes&amp;cmd=resource&amp;f=debugger.js"></script>
    <script>
      var CONSOLE_MODE = %(console)s,
          EVALEX = %(evalex)s,
          EVALEX_TRUSTED = %(evalex_trusted)s,
          SECRET = "%(secret)s";
    </script>
  </head>
  <body style="background-color: #fff">
    <div class="debugger">
"""

FOOTER = """\
      <div class="footer">
        Brought to you by <strong class="arthur">DON'T PANIC</strong>, your
        friendly Werkzeug powered traceback interpreter.
      </div>
    </div>

    <div class="pin-prompt">
      <div class="inner">
        <h3>Console Locked</h3>
        <p>
          The console is locked and needs to be unlocked by entering the PIN.
          You can find the PIN printed out on the standard output of your
          shell that runs the server.
        <form>
          <p>PIN:
            <input type=text name=pin size=14>
            <input type=submit name=btn value="Confirm Pin">
        </form>
      </div>
    </div>
  </body>
</html>
"""

PAGE_HTML = (
    HEADER
    + """\
<h1>%(exception_type)s</h1>
<div class="detail">
  <p class="errormsg">%(exception)s</p>
</div>
<h2 class="traceback">Traceback <em>(most recent call last)</em></h2>
%(summary)s
<div class="plain">
    <p>
      This is the Copy/Paste friendly version of the traceback.
    </p>
    <textarea cols="50" rows="10" name="code" readonly>%(plaintext)s</textarea>
</div>
<div class="explanation">
  The debugger caught an exception in your WSGI application.  You can now
  look at the traceback which led to the error.  <span class="nojavascript">
  If you enable JavaScript you can also use additional features such as code
  execution (if the evalex feature is enabled), automatic pasting of the
  exceptions and much more.</span>
</div>
"""
    + FOOTER
    + """
<!--

%(plaintext_cs)s

-->
"""
)

CONSOLE_HTML = (
    HEADER
    + """\
<h1>Interactive Console</h1>
<div class="explanation">
In this console you can execute Python expressions in the context of the
application.  The initial namespace was created by the debugger automatically.
</div>
<div class="console"><div class="inner">The Console requires JavaScript.</div></div>
"""
    + FOOTER
)

SUMMARY_HTML = """\
<div class="%(classes)s">
  %(title)s
  <ul>%(frames)s</ul>
  %(description)s
</div>
"""

FRAME_HTML = """\
<div class="frame" id="frame-%(id)d">
  <h4>File <cite class="filename">"%(filename)s"</cite>,
      line <em class="line">%(lineno)s</em>,
      in <code class="function">%(function_name)s</code></h4>
  <div class="source %(library)s">%(lines)s</div>
</div>
"""


def _process_traceback(
    exc: BaseException,
    te: traceback.TracebackException | None = None,
    *,
    skip: int = 0,
    hide: bool = True,
) -> traceback.TracebackException:
    if te is None:
        te = traceback.TracebackException.from_exception(exc, lookup_lines=False)

    # Get the frames the same way StackSummary.extract did, in order
    # to match each frame with the FrameSummary to augment.
    frame_gen = traceback.walk_tb(exc.__traceback__)
    limit = getattr(sys, "tracebacklimit", None)

    if limit is not None:
        if limit < 0:
            limit = 0

        frame_gen = itertools.islice(frame_gen, limit)

    if skip:
        frame_gen = itertools.islice(frame_gen, skip, None)
        del te.stack[:skip]

    new_stack: list[DebugFrameSummary] = []
    hidden = False

    # Match each frame with the FrameSummary that was generated.
    # Hide frames using Paste's __traceback_hide__ rules. Replace
    # all visible FrameSummary with DebugFrameSummary.
    for (f, _), fs in zip(frame_gen, te.stack):
        if hide:
            hide_value = f.f_locals.get("__traceback_hide__", False)

            if hide_value in {"before", "before_and_this"}:
                new_stack = []
                hidden = False

                if hide_value == "before_and_this":
                    continue
            elif hide_value in {"reset", "reset_and_this"}:
                hidden = False

                if hide_value == "reset_and_this":
                    continue
            elif hide_value in {"after", "after_and_this"}:
                hidden = True

                if hide_value == "after_and_this":
                    continue
            elif hide_value or hidden:
                continue

        frame_args: dict[str, t.Any] = {
            "filename": fs.filename,
            "lineno": fs.lineno,
            "name": fs.name,
            "locals": f.f_locals,
            "globals": f.f_globals,
        }

        if sys.version_info >= (3, 11):
            frame_args["colno"] = fs.colno
            frame_args["end_colno"] = fs.end_colno

        new_stack.append(DebugFrameSummary(**frame_args))

    # The codeop module is used to compile code from the interactive
    # debugger. Hide any codeop frames from the bottom of the traceback.
    while new_stack:
        module = new_stack[0].global_ns.get("__name__")

        if module is None:
            module = new_stack[0].local_ns.get("__name__")

        if module == "codeop":
            del new_stack[0]
        else:
            break

    te.stack[:] = new_stack

    if te.__context__:
        context_exc = t.cast(BaseException, exc.__context__)
        te.__context__ = _process_traceback(context_exc, te.__context__, hide=hide)

    if te.__cause__:
        cause_exc = t.cast(BaseException, exc.__cause__)
        te.__cause__ = _process_traceback(cause_exc, te.__cause__, hide=hide)

    return te


class DebugTraceback:
    __slots__ = ("_te", "_cache_all_tracebacks", "_cache_all_frames")

    def __init__(
        self,
        exc: BaseException,
        te: traceback.TracebackException | None = None,
        *,
        skip: int = 0,
        hide: bool = True,
    ) -> None:
        self._te = _process_traceback(exc, te, skip=skip, hide=hide)

    def __str__(self) -> str:
        return f"<{type(self).__name__} {self._te}>"

    @cached_property
    def all_tracebacks(
        self,
    ) -> list[tuple[str | None, traceback.TracebackException]]:
        out = []
        current = self._te

        while current is not None:
            if current.__cause__ is not None:
                chained_msg = (
                    "The above exception was the direct cause of the"
                    " following exception"
                )
                chained_exc = current.__cause__
            elif current.__context__ is not None and not current.__suppress_context__:
                chained_msg = (
                    "During handling of the above exception, another"
                    " exception occurred"
                )
                chained_exc = current.__context__
            else:
                chained_msg = None
                chained_exc = None

            out.append((chained_msg, current))
            current = chained_exc

        return out

    @cached_property
    def all_frames(self) -> list[DebugFrameSummary]:
        return [
            f  # type: ignore[misc]
            for _, te in self.all_tracebacks
            for f in te.stack
        ]

    def render_traceback_text(self) -> str:
        return "".join(self._te.format())

    def render_traceback_html(self, include_title: bool = True) -> str:
        library_frames = [f.is_library for f in self.all_frames]
        mark_library = 0 < sum(library_frames) < len(library_frames)
        rows = []

        if not library_frames:
            classes = "traceback noframe-traceback"
        else:
            classes = "traceback"

            for msg, current in reversed(self.all_tracebacks):
                row_parts = []

                if msg is not None:
                    row_parts.append(f'<li><div class="exc-divider">{msg}:</div>')

                for frame in current.stack:
                    frame = t.cast(DebugFrameSummary, frame)
                    info = f' title="{escape(frame.info)}"' if frame.info else ""
                    row_parts.append(f"<li{info}>{frame.render_html(mark_library)}")

                rows.append("\n".join(row_parts))

        if sys.version_info < (3, 13):
            exc_type_str = self._te.exc_type.__name__
        else:
            exc_type_str = self._te.exc_type_str

        is_syntax_error = exc_type_str == "SyntaxError"

        if include_title:
            if is_syntax_error:
                title = "Syntax Error"
            else:
                title = "Traceback <em>(most recent call last)</em>:"
        else:
            title = ""

        exc_full = escape("".join(self._te.format_exception_only()))

        if is_syntax_error:
            description = f"<pre class=syntaxerror>{exc_full}</pre>"
        else:
            description = f"<blockquote>{exc_full}</blockquote>"

        return SUMMARY_HTML % {
            "classes": classes,
            "title": f"<h3>{title}</h3>",
            "frames": "\n".join(rows),
            "description": description,
        }

    def render_debugger_html(
        self, evalex: bool, secret: str, evalex_trusted: bool
    ) -> str:
        exc_lines = list(self._te.format_exception_only())
        plaintext = "".join(self._te.format())

        if sys.version_info < (3, 13):
            exc_type_str = self._te.exc_type.__name__
        else:
            exc_type_str = self._te.exc_type_str

        return PAGE_HTML % {
            "evalex": "true" if evalex else "false",
            "evalex_trusted": "true" if evalex_trusted else "false",
            "console": "false",
            "title": escape(exc_lines[0]),
            "exception": escape("".join(exc_lines)),
            "exception_type": escape(exc_type_str),
            "summary": self.render_traceback_html(include_title=False),
            "plaintext": escape(plaintext),
            "plaintext_cs": re.sub("-{2,}", "-", plaintext),
            "secret": secret,
        }


class DebugFrameSummary(traceback.FrameSummary):
    """A :class:`traceback.FrameSummary` that can evaluate code in the
    frame's namespace.
    """

    __slots__ = (
        "local_ns",
        "global_ns",
        "_cache_info",
        "_cache_is_library",
        "_cache_console",
    )

    def __init__(
        self,
        *,
        locals: dict[str, t.Any],
        globals: dict[str, t.Any],
        **kwargs: t.Any,
    ) -> None:
        super().__init__(locals=None, **kwargs)
        self.local_ns = locals
        self.global_ns = globals

    @cached_property
    def info(self) -> str | None:
        return self.local_ns.get("__traceback_info__")

    @cached_property
    def is_library(self) -> bool:
        return any(
            self.filename.startswith((path, os.path.realpath(path)))
            for path in sysconfig.get_paths().values()
        )

    @cached_property
    def console(self) -> Console:
        return Console(self.global_ns, self.local_ns)

    def eval(self, code: str) -> t.Any:
        return self.console.eval(code)

    def render_html(self, mark_library: bool) -> str:
        context = 5
        lines = linecache.getlines(self.filename)
        line_idx = self.lineno - 1  # type: ignore[operator]
        start_idx = max(0, line_idx - context)
        stop_idx = min(len(lines), line_idx + context + 1)
        rendered_lines = []

        def render_line(line: str, cls: str) -> None:
            line = line.expandtabs().rstrip()
            stripped_line = line.strip()
            prefix = len(line) - len(stripped_line)
            colno = getattr(self, "colno", 0)
            end_colno = getattr(self, "end_colno", 0)

            if cls == "current" and colno and end_colno:
                arrow = (
                    f'\n<span class="ws">{" " * prefix}</span>'
                    f'{" " * (colno - prefix)}{"^" * (end_colno - colno)}'
                )
            else:
                arrow = ""

            rendered_lines.append(
                f'<pre class="line {cls}"><span class="ws">{" " * prefix}</span>'
                f"{escape(stripped_line) if stripped_line else ' '}"
                f"{arrow if arrow else ''}</pre>"
            )

        if lines:
            for line in lines[start_idx:line_idx]:
                render_line(line, "before")

            render_line(lines[line_idx], "current")

            for line in lines[line_idx + 1 : stop_idx]:
                render_line(line, "after")

        return FRAME_HTML % {
            "id": id(self),
            "filename": escape(self.filename),
            "lineno": self.lineno,
            "function_name": escape(self.name),
            "lines": "\n".join(rendered_lines),
            "library": "library" if mark_library and self.is_library else "",
        }


def render_console_html(secret: str, evalex_trusted: bool) -> str:
    return CONSOLE_HTML % {
        "evalex": "true",
        "evalex_trusted": "true" if evalex_trusted else "false",
        "console": "true",
        "title": "Console",
        "secret": secret,
    }

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\ctx.py ===
from __future__ import annotations

import contextvars
import sys
import typing as t
from functools import update_wrapper
from types import TracebackType

from werkzeug.exceptions import HTTPException

from . import typing as ft
from .globals import _cv_app
from .globals import _cv_request
from .signals import appcontext_popped
from .signals import appcontext_pushed

if t.TYPE_CHECKING:  # pragma: no cover
    from _typeshed.wsgi import WSGIEnvironment

    from .app import Flask
    from .sessions import SessionMixin
    from .wrappers import Request


# a singleton sentinel value for parameter defaults
_sentinel = object()


class _AppCtxGlobals:
    """A plain object. Used as a namespace for storing data during an
    application context.

    Creating an app context automatically creates this object, which is
    made available as the :data:`g` proxy.

    .. describe:: 'key' in g

        Check whether an attribute is present.

        .. versionadded:: 0.10

    .. describe:: iter(g)

        Return an iterator over the attribute names.

        .. versionadded:: 0.10
    """

    # Define attr methods to let mypy know this is a namespace object
    # that has arbitrary attributes.

    def __getattr__(self, name: str) -> t.Any:
        try:
            return self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name: str, value: t.Any) -> None:
        self.__dict__[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self.__dict__[name]
        except KeyError:
            raise AttributeError(name) from None

    def get(self, name: str, default: t.Any | None = None) -> t.Any:
        """Get an attribute by name, or a default value. Like
        :meth:`dict.get`.

        :param name: Name of attribute to get.
        :param default: Value to return if the attribute is not present.

        .. versionadded:: 0.10
        """
        return self.__dict__.get(name, default)

    def pop(self, name: str, default: t.Any = _sentinel) -> t.Any:
        """Get and remove an attribute by name. Like :meth:`dict.pop`.

        :param name: Name of attribute to pop.
        :param default: Value to return if the attribute is not present,
            instead of raising a ``KeyError``.

        .. versionadded:: 0.11
        """
        if default is _sentinel:
            return self.__dict__.pop(name)
        else:
            return self.__dict__.pop(name, default)

    def setdefault(self, name: str, default: t.Any = None) -> t.Any:
        """Get the value of an attribute if it is present, otherwise
        set and return a default value. Like :meth:`dict.setdefault`.

        :param name: Name of attribute to get.
        :param default: Value to set and return if the attribute is not
            present.

        .. versionadded:: 0.11
        """
        return self.__dict__.setdefault(name, default)

    def __contains__(self, item: str) -> bool:
        return item in self.__dict__

    def __iter__(self) -> t.Iterator[str]:
        return iter(self.__dict__)

    def __repr__(self) -> str:
        ctx = _cv_app.get(None)
        if ctx is not None:
            return f"<flask.g of '{ctx.app.name}'>"
        return object.__repr__(self)


def after_this_request(
    f: ft.AfterRequestCallable[t.Any],
) -> ft.AfterRequestCallable[t.Any]:
    """Executes a function after this request.  This is useful to modify
    response objects.  The function is passed the response object and has
    to return the same or a new one.

    Example::

        @app.route('/')
        def index():
            @after_this_request
            def add_header(response):
                response.headers['X-Foo'] = 'Parachute'
                return response
            return 'Hello World!'

    This is more useful if a function other than the view function wants to
    modify a response.  For instance think of a decorator that wants to add
    some headers without converting the return value into a response object.

    .. versionadded:: 0.9
    """
    ctx = _cv_request.get(None)

    if ctx is None:
        raise RuntimeError(
            "'after_this_request' can only be used when a request"
            " context is active, such as in a view function."
        )

    ctx._after_request_functions.append(f)
    return f


F = t.TypeVar("F", bound=t.Callable[..., t.Any])


def copy_current_request_context(f: F) -> F:
    """A helper function that decorates a function to retain the current
    request context.  This is useful when working with greenlets.  The moment
    the function is decorated a copy of the request context is created and
    then pushed when the function is called.  The current session is also
    included in the copied request context.

    Example::

        import gevent
        from flask import copy_current_request_context

        @app.route('/')
        def index():
            @copy_current_request_context
            def do_some_work():
                # do some work here, it can access flask.request or
                # flask.session like you would otherwise in the view function.
                ...
            gevent.spawn(do_some_work)
            return 'Regular response'

    .. versionadded:: 0.10
    """
    ctx = _cv_request.get(None)

    if ctx is None:
        raise RuntimeError(
            "'copy_current_request_context' can only be used when a"
            " request context is active, such as in a view function."
        )

    ctx = ctx.copy()

    def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
        with ctx:  # type: ignore[union-attr]
            return ctx.app.ensure_sync(f)(*args, **kwargs)  # type: ignore[union-attr]

    return update_wrapper(wrapper, f)  # type: ignore[return-value]


def has_request_context() -> bool:
    """If you have code that wants to test if a request context is there or
    not this function can be used.  For instance, you may want to take advantage
    of request information if the request object is available, but fail
    silently if it is unavailable.

    ::

        class User(db.Model):

            def __init__(self, username, remote_addr=None):
                self.username = username
                if remote_addr is None and has_request_context():
                    remote_addr = request.remote_addr
                self.remote_addr = remote_addr

    Alternatively you can also just test any of the context bound objects
    (such as :class:`request` or :class:`g`) for truthness::

        class User(db.Model):

            def __init__(self, username, remote_addr=None):
                self.username = username
                if remote_addr is None and request:
                    remote_addr = request.remote_addr
                self.remote_addr = remote_addr

    .. versionadded:: 0.7
    """
    return _cv_request.get(None) is not None


def has_app_context() -> bool:
    """Works like :func:`has_request_context` but for the application
    context.  You can also just do a boolean check on the
    :data:`current_app` object instead.

    .. versionadded:: 0.9
    """
    return _cv_app.get(None) is not None


class AppContext:
    """The app context contains application-specific information. An app
    context is created and pushed at the beginning of each request if
    one is not already active. An app context is also pushed when
    running CLI commands.
    """

    def __init__(self, app: Flask) -> None:
        self.app = app
        self.url_adapter = app.create_url_adapter(None)
        self.g: _AppCtxGlobals = app.app_ctx_globals_class()
        self._cv_tokens: list[contextvars.Token[AppContext]] = []

    def push(self) -> None:
        """Binds the app context to the current context."""
        self._cv_tokens.append(_cv_app.set(self))
        appcontext_pushed.send(self.app, _async_wrapper=self.app.ensure_sync)

    def pop(self, exc: BaseException | None = _sentinel) -> None:  # type: ignore
        """Pops the app context."""
        try:
            if len(self._cv_tokens) == 1:
                if exc is _sentinel:
                    exc = sys.exc_info()[1]
                self.app.do_teardown_appcontext(exc)
        finally:
            ctx = _cv_app.get()
            _cv_app.reset(self._cv_tokens.pop())

        if ctx is not self:
            raise AssertionError(
                f"Popped wrong app context. ({ctx!r} instead of {self!r})"
            )

        appcontext_popped.send(self.app, _async_wrapper=self.app.ensure_sync)

    def __enter__(self) -> AppContext:
        self.push()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.pop(exc_value)


class RequestContext:
    """The request context contains per-request information. The Flask
    app creates and pushes it at the beginning of the request, then pops
    it at the end of the request. It will create the URL adapter and
    request object for the WSGI environment provided.

    Do not attempt to use this class directly, instead use
    :meth:`~flask.Flask.test_request_context` and
    :meth:`~flask.Flask.request_context` to create this object.

    When the request context is popped, it will evaluate all the
    functions registered on the application for teardown execution
    (:meth:`~flask.Flask.teardown_request`).

    The request context is automatically popped at the end of the
    request. When using the interactive debugger, the context will be
    restored so ``request`` is still accessible. Similarly, the test
    client can preserve the context after the request ends. However,
    teardown functions may already have closed some resources such as
    database connections.
    """

    def __init__(
        self,
        app: Flask,
        environ: WSGIEnvironment,
        request: Request | None = None,
        session: SessionMixin | None = None,
    ) -> None:
        self.app = app
        if request is None:
            request = app.request_class(environ)
            request.json_module = app.json
        self.request: Request = request
        self.url_adapter = None
        try:
            self.url_adapter = app.create_url_adapter(self.request)
        except HTTPException as e:
            self.request.routing_exception = e
        self.flashes: list[tuple[str, str]] | None = None
        self.session: SessionMixin | None = session
        # Functions that should be executed after the request on the response
        # object.  These will be called before the regular "after_request"
        # functions.
        self._after_request_functions: list[ft.AfterRequestCallable[t.Any]] = []

        self._cv_tokens: list[
            tuple[contextvars.Token[RequestContext], AppContext | None]
        ] = []

    def copy(self) -> RequestContext:
        """Creates a copy of this request context with the same request object.
        This can be used to move a request context to a different greenlet.
        Because the actual request object is the same this cannot be used to
        move a request context to a different thread unless access to the
        request object is locked.

        .. versionadded:: 0.10

        .. versionchanged:: 1.1
           The current session object is used instead of reloading the original
           data. This prevents `flask.session` pointing to an out-of-date object.
        """
        return self.__class__(
            self.app,
            environ=self.request.environ,
            request=self.request,
            session=self.session,
        )

    def match_request(self) -> None:
        """Can be overridden by a subclass to hook into the matching
        of the request.
        """
        try:
            result = self.url_adapter.match(return_rule=True)  # type: ignore
            self.request.url_rule, self.request.view_args = result  # type: ignore
        except HTTPException as e:
            self.request.routing_exception = e

    def push(self) -> None:
        # Before we push the request context we have to ensure that there
        # is an application context.
        app_ctx = _cv_app.get(None)

        if app_ctx is None or app_ctx.app is not self.app:
            app_ctx = self.app.app_context()
            app_ctx.push()
        else:
            app_ctx = None

        self._cv_tokens.append((_cv_request.set(self), app_ctx))

        # Open the session at the moment that the request context is available.
        # This allows a custom open_session method to use the request context.
        # Only open a new session if this is the first time the request was
        # pushed, otherwise stream_with_context loses the session.
        if self.session is None:
            session_interface = self.app.session_interface
            self.session = session_interface.open_session(self.app, self.request)

            if self.session is None:
                self.session = session_interface.make_null_session(self.app)

        # Match the request URL after loading the session, so that the
        # session is available in custom URL converters.
        if self.url_adapter is not None:
            self.match_request()

    def pop(self, exc: BaseException | None = _sentinel) -> None:  # type: ignore
        """Pops the request context and unbinds it by doing that.  This will
        also trigger the execution of functions registered by the
        :meth:`~flask.Flask.teardown_request` decorator.

        .. versionchanged:: 0.9
           Added the `exc` argument.
        """
        clear_request = len(self._cv_tokens) == 1

        try:
            if clear_request:
                if exc is _sentinel:
                    exc = sys.exc_info()[1]
                self.app.do_teardown_request(exc)

                request_close = getattr(self.request, "close", None)
                if request_close is not None:
                    request_close()
        finally:
            ctx = _cv_request.get()
            token, app_ctx = self._cv_tokens.pop()
            _cv_request.reset(token)

            # get rid of circular dependencies at the end of the request
            # so that we don't require the GC to be active.
            if clear_request:
                ctx.request.environ["werkzeug.request"] = None

            if app_ctx is not None:
                app_ctx.pop(exc)

            if ctx is not self:
                raise AssertionError(
                    f"Popped wrong request context. ({ctx!r} instead of {self!r})"
                )

    def __enter__(self) -> RequestContext:
        self.push()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_value: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.pop(exc_value)

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} {self.request.url!r}"
            f" [{self.request.method}] of {self.app.name}>"
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\text.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: December 1, 2020
# URL: https://humanfriendly.readthedocs.io

"""
Simple text manipulation functions.

The :mod:`~humanfriendly.text` module contains simple functions to manipulate text:

- The :func:`concatenate()` and :func:`pluralize()` functions make it easy to
  generate human friendly output.

- The :func:`format()`, :func:`compact()` and :func:`dedent()` functions
  provide a clean and simple to use syntax for composing large text fragments
  with interpolated variables.

- The :func:`tokenize()` function parses simple user input.
"""

# Standard library modules.
import numbers
import random
import re
import string
import textwrap

# Public identifiers that require documentation.
__all__ = (
    'compact',
    'compact_empty_lines',
    'concatenate',
    'dedent',
    'format',
    'generate_slug',
    'is_empty_line',
    'join_lines',
    'pluralize',
    'pluralize_raw',
    'random_string',
    'split',
    'split_paragraphs',
    'tokenize',
    'trim_empty_lines',
)


def compact(text, *args, **kw):
    '''
    Compact whitespace in a string.

    Trims leading and trailing whitespace, replaces runs of whitespace
    characters with a single space and interpolates any arguments using
    :func:`format()`.

    :param text: The text to compact (a string).
    :param args: Any positional arguments are interpolated using :func:`format()`.
    :param kw: Any keyword arguments are interpolated using :func:`format()`.
    :returns: The compacted text (a string).

    Here's an example of how I like to use the :func:`compact()` function, this
    is an example from a random unrelated project I'm working on at the moment::

        raise PortDiscoveryError(compact("""
            Failed to discover port(s) that Apache is listening on!
            Maybe I'm parsing the wrong configuration file? ({filename})
        """, filename=self.ports_config))

    The combination of :func:`compact()` and Python's multi line strings allows
    me to write long text fragments with interpolated variables that are easy
    to write, easy to read and work well with Python's whitespace
    sensitivity.
    '''
    non_whitespace_tokens = text.split()
    compacted_text = ' '.join(non_whitespace_tokens)
    return format(compacted_text, *args, **kw)


def compact_empty_lines(text):
    """
    Replace repeating empty lines with a single empty line (similar to ``cat -s``).

    :param text: The text in which to compact empty lines (a string).
    :returns: The text with empty lines compacted (a string).
    """
    i = 0
    lines = text.splitlines(True)
    while i < len(lines):
        if i > 0 and is_empty_line(lines[i - 1]) and is_empty_line(lines[i]):
            lines.pop(i)
        else:
            i += 1
    return ''.join(lines)


def concatenate(items, conjunction='and', serial_comma=False):
    """
    Concatenate a list of items in a human friendly way.

    :param items:

        A sequence of strings.

    :param conjunction:

        The word to use before the last item (a string, defaults to "and").

    :param serial_comma:

        :data:`True` to use a `serial comma`_, :data:`False` otherwise
        (defaults to :data:`False`).

    :returns:

        A single string.

    >>> from humanfriendly.text import concatenate
    >>> concatenate(["eggs", "milk", "bread"])
    'eggs, milk and bread'

    .. _serial comma: https://en.wikipedia.org/wiki/Serial_comma
    """
    items = list(items)
    if len(items) > 1:
        final_item = items.pop()
        formatted = ', '.join(items)
        if serial_comma:
            formatted += ','
        return ' '.join([formatted, conjunction, final_item])
    elif items:
        return items[0]
    else:
        return ''


def dedent(text, *args, **kw):
    """
    Dedent a string (remove common leading whitespace from all lines).

    Removes common leading whitespace from all lines in the string using
    :func:`textwrap.dedent()`, removes leading and trailing empty lines using
    :func:`trim_empty_lines()` and interpolates any arguments using
    :func:`format()`.

    :param text: The text to dedent (a string).
    :param args: Any positional arguments are interpolated using :func:`format()`.
    :param kw: Any keyword arguments are interpolated using :func:`format()`.
    :returns: The dedented text (a string).

    The :func:`compact()` function's documentation contains an example of how I
    like to use the :func:`compact()` and :func:`dedent()` functions. The main
    difference is that I use :func:`compact()` for text that will be presented
    to the user (where whitespace is not so significant) and :func:`dedent()`
    for data file and code generation tasks (where newlines and indentation are
    very significant).
    """
    dedented_text = textwrap.dedent(text)
    trimmed_text = trim_empty_lines(dedented_text)
    return format(trimmed_text, *args, **kw)


def format(text, *args, **kw):
    """
    Format a string using the string formatting operator and/or :meth:`str.format()`.

    :param text: The text to format (a string).
    :param args: Any positional arguments are interpolated into the text using
                 the string formatting operator (``%``). If no positional
                 arguments are given no interpolation is done.
    :param kw: Any keyword arguments are interpolated into the text using the
               :meth:`str.format()` function. If no keyword arguments are given
               no interpolation is done.
    :returns: The text with any positional and/or keyword arguments
              interpolated (a string).

    The implementation of this function is so trivial that it seems silly to
    even bother writing and documenting it. Justifying this requires some
    context :-).

    **Why format() instead of the string formatting operator?**

    For really simple string interpolation Python's string formatting operator
    is ideal, but it does have some strange quirks:

    - When you switch from interpolating a single value to interpolating
      multiple values you have to wrap them in tuple syntax. Because
      :func:`format()` takes a `variable number of arguments`_ it always
      receives a tuple (which saves me a context switch :-). Here's an
      example:

      >>> from humanfriendly.text import format
      >>> # The string formatting operator.
      >>> print('the magic number is %s' % 42)
      the magic number is 42
      >>> print('the magic numbers are %s and %s' % (12, 42))
      the magic numbers are 12 and 42
      >>> # The format() function.
      >>> print(format('the magic number is %s', 42))
      the magic number is 42
      >>> print(format('the magic numbers are %s and %s', 12, 42))
      the magic numbers are 12 and 42

    - When you interpolate a single value and someone accidentally passes in a
      tuple your code raises a :exc:`~exceptions.TypeError`. Because
      :func:`format()` takes a `variable number of arguments`_ it always
      receives a tuple so this can never happen. Here's an example:

      >>> # How expecting to interpolate a single value can fail.
      >>> value = (12, 42)
      >>> print('the magic value is %s' % value)
      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
      TypeError: not all arguments converted during string formatting
      >>> # The following line works as intended, no surprises here!
      >>> print(format('the magic value is %s', value))
      the magic value is (12, 42)

    **Why format() instead of the str.format() method?**

    When you're doing complex string interpolation the :meth:`str.format()`
    function results in more readable code, however I frequently find myself
    adding parentheses to force evaluation order. The :func:`format()` function
    avoids this because of the relative priority between the comma and dot
    operators. Here's an example:

    >>> "{adjective} example" + " " + "(can't think of anything less {adjective})".format(adjective='silly')
    "{adjective} example (can't think of anything less silly)"
    >>> ("{adjective} example" + " " + "(can't think of anything less {adjective})").format(adjective='silly')
    "silly example (can't think of anything less silly)"
    >>> format("{adjective} example" + " " + "(can't think of anything less {adjective})", adjective='silly')
    "silly example (can't think of anything less silly)"

    The :func:`compact()` and :func:`dedent()` functions are wrappers that
    combine :func:`format()` with whitespace manipulation to make it easy to
    write nice to read Python code.

    .. _variable number of arguments: https://docs.python.org/2/tutorial/controlflow.html#arbitrary-argument-lists
    """
    if args:
        text %= args
    if kw:
        text = text.format(**kw)
    return text


def generate_slug(text, delimiter="-"):
    """
    Convert text to a normalized "slug" without whitespace.

    :param text: The original text, for example ``Some Random Text!``.
    :param delimiter: The delimiter used to separate words
                      (defaults to the ``-`` character).
    :returns: The slug text, for example ``some-random-text``.
    :raises: :exc:`~exceptions.ValueError` when the provided
             text is nonempty but results in an empty slug.
    """
    slug = text.lower()
    escaped = delimiter.replace("\\", "\\\\")
    slug = re.sub("[^a-z0-9]+", escaped, slug)
    slug = slug.strip(delimiter)
    if text and not slug:
        msg = "The provided text %r results in an empty slug!"
        raise ValueError(format(msg, text))
    return slug


def is_empty_line(text):
    """
    Check if a text is empty or contains only whitespace.

    :param text: The text to check for "emptiness" (a string).
    :returns: :data:`True` if the text is empty or contains only whitespace,
              :data:`False` otherwise.
    """
    return len(text) == 0 or text.isspace()


def join_lines(text):
    """
    Remove "hard wrapping" from the paragraphs in a string.

    :param text: The text to reformat (a string).
    :returns: The text without hard wrapping (a string).

    This function works by removing line breaks when the last character before
    a line break and the first character after the line break are both
    non-whitespace characters. This means that common leading indentation will
    break :func:`join_lines()` (in that case you can use :func:`dedent()`
    before calling :func:`join_lines()`).
    """
    return re.sub(r'(\S)\n(\S)', r'\1 \2', text)


def pluralize(count, singular, plural=None):
    """
    Combine a count with the singular or plural form of a word.

    :param count: The count (a number).
    :param singular: The singular form of the word (a string).
    :param plural: The plural form of the word (a string or :data:`None`).
    :returns: The count and singular or plural word concatenated (a string).

    See :func:`pluralize_raw()` for the logic underneath :func:`pluralize()`.
    """
    return '%s %s' % (count, pluralize_raw(count, singular, plural))


def pluralize_raw(count, singular, plural=None):
    """
    Select the singular or plural form of a word based on a count.

    :param count: The count (a number).
    :param singular: The singular form of the word (a string).
    :param plural: The plural form of the word (a string or :data:`None`).
    :returns: The singular or plural form of the word (a string).

    When the given count is exactly 1.0 the singular form of the word is
    selected, in all other cases the plural form of the word is selected.

    If the plural form of the word is not provided it is obtained by
    concatenating the singular form of the word with the letter "s". Of course
    this will not always be correct, which is why you have the option to
    specify both forms.
    """
    if not plural:
        plural = singular + 's'
    return singular if float(count) == 1.0 else plural


def random_string(length=(25, 100), characters=string.ascii_letters):
    """random_string(length=(25, 100), characters=string.ascii_letters)
    Generate a random string.

    :param length: The length of the string to be generated (a number or a
                   tuple with two numbers). If this is a tuple then a random
                   number between the two numbers given in the tuple is used.
    :param characters: The characters to be used (a string, defaults
                       to :data:`string.ascii_letters`).
    :returns: A random string.

    The :func:`random_string()` function is very useful in test suites; by the
    time I included it in :mod:`humanfriendly.text` I had already included
    variants of this function in seven different test suites :-).
    """
    if not isinstance(length, numbers.Number):
        length = random.randint(length[0], length[1])
    return ''.join(random.choice(characters) for _ in range(length))


def split(text, delimiter=','):
    """
    Split a comma-separated list of strings.

    :param text: The text to split (a string).
    :param delimiter: The delimiter to split on (a string).
    :returns: A list of zero or more nonempty strings.

    Here's the default behavior of Python's built in :meth:`str.split()`
    function:

    >>> 'foo,bar, baz,'.split(',')
    ['foo', 'bar', ' baz', '']

    In contrast here's the default behavior of the :func:`split()` function:

    >>> from humanfriendly.text import split
    >>> split('foo,bar, baz,')
    ['foo', 'bar', 'baz']

    Here is an example that parses a nested data structure (a mapping of
    logging level names to one or more styles per level) that's encoded in a
    string so it can be set as an environment variable:

    >>> from pprint import pprint
    >>> encoded_data = 'debug=green;warning=yellow;error=red;critical=red,bold'
    >>> parsed_data = dict((k, split(v, ',')) for k, v in (split(kv, '=') for kv in split(encoded_data, ';')))
    >>> pprint(parsed_data)
    {'debug': ['green'],
     'warning': ['yellow'],
     'error': ['red'],
     'critical': ['red', 'bold']}
    """
    return [token.strip() for token in text.split(delimiter) if token and not token.isspace()]


def split_paragraphs(text):
    """
    Split a string into paragraphs (one or more lines delimited by an empty line).

    :param text: The text to split into paragraphs (a string).
    :returns: A list of strings.
    """
    paragraphs = []
    for chunk in text.split('\n\n'):
        chunk = trim_empty_lines(chunk)
        if chunk and not chunk.isspace():
            paragraphs.append(chunk)
    return paragraphs


def tokenize(text):
    """
    Tokenize a text into numbers and strings.

    :param text: The text to tokenize (a string).
    :returns: A list of strings and/or numbers.

    This function is used to implement robust tokenization of user input in
    functions like :func:`.parse_size()` and :func:`.parse_timespan()`. It
    automatically coerces integer and floating point numbers, ignores
    whitespace and knows how to separate numbers from strings even without
    whitespace. Some examples to make this more concrete:

    >>> from humanfriendly.text import tokenize
    >>> tokenize('42')
    [42]
    >>> tokenize('42MB')
    [42, 'MB']
    >>> tokenize('42.5MB')
    [42.5, 'MB']
    >>> tokenize('42.5 MB')
    [42.5, 'MB']
    """
    tokenized_input = []
    for token in re.split(r'(\d+(?:\.\d+)?)', text):
        token = token.strip()
        if re.match(r'\d+\.\d+', token):
            tokenized_input.append(float(token))
        elif token.isdigit():
            tokenized_input.append(int(token))
        elif token:
            tokenized_input.append(token)
    return tokenized_input


def trim_empty_lines(text):
    """
    Trim leading and trailing empty lines from the given text.

    :param text: The text to trim (a string).
    :returns: The trimmed text (a string).
    """
    lines = text.splitlines(True)
    while lines and is_empty_line(lines[0]):
        lines.pop(0)
    while lines and is_empty_line(lines[-1]):
        lines.pop(-1)
    return ''.join(lines)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\message.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

# TODO: We should just make these methods all "pure-virtual" and move
# all implementation out, into reflection.py for now.


"""Contains an abstract base class for protocol messages."""

__author__ = 'robinson@google.com (Will Robinson)'

_INCONSISTENT_MESSAGE_ATTRIBUTES = ('Extensions',)


class Error(Exception):
  """Base error type for this module."""
  pass


class DecodeError(Error):
  """Exception raised when deserializing messages."""
  pass


class EncodeError(Error):
  """Exception raised when serializing messages."""
  pass


class Message(object):

  """Abstract base class for protocol messages.

  Protocol message classes are almost always generated by the protocol
  compiler.  These generated types subclass Message and implement the methods
  shown below.
  """

  # TODO: Link to an HTML document here.

  # TODO: Document that instances of this class will also
  # have an Extensions attribute with __getitem__ and __setitem__.
  # Again, not sure how to best convey this.

  # TODO: Document these fields and methods.

  __slots__ = []

  #: The :class:`google.protobuf.Descriptor`
  # for this message type.
  DESCRIPTOR = None

  def __deepcopy__(self, memo=None):
    clone = type(self)()
    clone.MergeFrom(self)
    return clone

  def __dir__(self):
    """Provides the list of all accessible Message attributes."""
    message_attributes = set(super().__dir__())

    # TODO: Remove this once the UPB implementation is improved.
    # The UPB proto implementation currently doesn't provide proto fields as
    # attributes and they have to added.
    if self.DESCRIPTOR is not None:
      for field in self.DESCRIPTOR.fields:
        message_attributes.add(field.name)

    # The Fast C++ proto implementation provides inaccessible attributes that
    # have to be removed.
    for attribute in _INCONSISTENT_MESSAGE_ATTRIBUTES:
      if attribute not in message_attributes:
        continue
      try:
        getattr(self, attribute)
      except AttributeError:
        message_attributes.remove(attribute)

    return sorted(message_attributes)

  def __eq__(self, other_msg):
    """Recursively compares two messages by value and structure."""
    raise NotImplementedError

  def __ne__(self, other_msg):
    # Can't just say self != other_msg, since that would infinitely recurse. :)
    return not self == other_msg

  def __hash__(self):
    raise TypeError('unhashable object')

  def __str__(self):
    """Outputs a human-readable representation of the message."""
    raise NotImplementedError

  def __unicode__(self):
    """Outputs a human-readable representation of the message."""
    raise NotImplementedError

  def __contains__(self, field_name_or_key):
    """Checks if a certain field is set for the message.

    Has presence fields return true if the field is set, false if the field is
    not set. Fields without presence do raise `ValueError` (this includes
    repeated fields, map fields, and implicit presence fields).

    If field_name is not defined in the message descriptor, `ValueError` will
    be raised.
    Note: WKT Struct checks if the key is contained in fields. ListValue checks
    if the item is contained in the list.

    Args:
      field_name_or_key: For Struct, the key (str) of the fields map. For
        ListValue, any type that may be contained in the list. For other
        messages, name of the field (str) to check for presence.

    Returns:
      bool: For Struct, whether the item is contained in fields. For ListValue,
            whether the item is contained in the list. For other message,
            whether a value has been set for the named field.

    Raises:
      ValueError: For normal messages,  if the `field_name_or_key` is not a
                  member of this message or `field_name_or_key` is not a string.
    """
    raise NotImplementedError

  def MergeFrom(self, other_msg):
    """Merges the contents of the specified message into current message.

    This method merges the contents of the specified message into the current
    message. Singular fields that are set in the specified message overwrite
    the corresponding fields in the current message. Repeated fields are
    appended. Singular sub-messages and groups are recursively merged.

    Args:
      other_msg (Message): A message to merge into the current message.
    """
    raise NotImplementedError

  def CopyFrom(self, other_msg):
    """Copies the content of the specified message into the current message.

    The method clears the current message and then merges the specified
    message using MergeFrom.

    Args:
      other_msg (Message): A message to copy into the current one.
    """
    if self is other_msg:
      return
    self.Clear()
    self.MergeFrom(other_msg)

  def Clear(self):
    """Clears all data that was set in the message."""
    raise NotImplementedError

  def SetInParent(self):
    """Mark this as present in the parent.

    This normally happens automatically when you assign a field of a
    sub-message, but sometimes you want to make the sub-message
    present while keeping it empty.  If you find yourself using this,
    you may want to reconsider your design.
    """
    raise NotImplementedError

  def IsInitialized(self):
    """Checks if the message is initialized.

    Returns:
      bool: The method returns True if the message is initialized (i.e. all of
      its required fields are set).
    """
    raise NotImplementedError

  # TODO: MergeFromString() should probably return None and be
  # implemented in terms of a helper that returns the # of bytes read.  Our
  # deserialization routines would use the helper when recursively
  # deserializing, but the end user would almost always just want the no-return
  # MergeFromString().

  def MergeFromString(self, serialized):
    """Merges serialized protocol buffer data into this message.

    When we find a field in `serialized` that is already present
    in this message:

    -   If it's a "repeated" field, we append to the end of our list.
    -   Else, if it's a scalar, we overwrite our field.
    -   Else, (it's a nonrepeated composite), we recursively merge
        into the existing composite.

    Args:
      serialized (bytes): Any object that allows us to call
        ``memoryview(serialized)`` to access a string of bytes using the
        buffer interface.

    Returns:
      int: The number of bytes read from `serialized`.
      For non-group messages, this will always be `len(serialized)`,
      but for messages which are actually groups, this will
      generally be less than `len(serialized)`, since we must
      stop when we reach an ``END_GROUP`` tag.  Note that if
      we *do* stop because of an ``END_GROUP`` tag, the number
      of bytes returned does not include the bytes
      for the ``END_GROUP`` tag information.

    Raises:
      DecodeError: if the input cannot be parsed.
    """
    # TODO: Document handling of unknown fields.
    # TODO: When we switch to a helper, this will return None.
    raise NotImplementedError

  def ParseFromString(self, serialized):
    """Parse serialized protocol buffer data in binary form into this message.

    Like :func:`MergeFromString()`, except we clear the object first.

    Raises:
      message.DecodeError if the input cannot be parsed.
    """
    self.Clear()
    return self.MergeFromString(serialized)

  def SerializeToString(self, **kwargs):
    """Serializes the protocol message to a binary string.

    Keyword Args:
      deterministic (bool): If true, requests deterministic serialization
        of the protobuf, with predictable ordering of map keys.

    Returns:
      A binary string representation of the message if all of the required
      fields in the message are set (i.e. the message is initialized).

    Raises:
      EncodeError: if the message isn't initialized (see :func:`IsInitialized`).
    """
    raise NotImplementedError

  def SerializePartialToString(self, **kwargs):
    """Serializes the protocol message to a binary string.

    This method is similar to SerializeToString but doesn't check if the
    message is initialized.

    Keyword Args:
      deterministic (bool): If true, requests deterministic serialization
        of the protobuf, with predictable ordering of map keys.

    Returns:
      bytes: A serialized representation of the partial message.
    """
    raise NotImplementedError

  # TODO: Decide whether we like these better
  # than auto-generated has_foo() and clear_foo() methods
  # on the instances themselves.  This way is less consistent
  # with C++, but it makes reflection-type access easier and
  # reduces the number of magically autogenerated things.
  #
  # TODO: Be sure to document (and test) exactly
  # which field names are accepted here.  Are we case-sensitive?
  # What do we do with fields that share names with Python keywords
  # like 'lambda' and 'yield'?
  #
  # nnorwitz says:
  # """
  # Typically (in python), an underscore is appended to names that are
  # keywords. So they would become lambda_ or yield_.
  # """
  def ListFields(self):
    """Returns a list of (FieldDescriptor, value) tuples for present fields.

    A message field is non-empty if HasField() would return true. A singular
    primitive field is non-empty if HasField() would return true in proto2 or it
    is non zero in proto3. A repeated field is non-empty if it contains at least
    one element. The fields are ordered by field number.

    Returns:
      list[tuple(FieldDescriptor, value)]: field descriptors and values
      for all fields in the message which are not empty. The values vary by
      field type.
    """
    raise NotImplementedError

  def HasField(self, field_name):
    """Checks if a certain field is set for the message.

    For a oneof group, checks if any field inside is set. Note that if the
    field_name is not defined in the message descriptor, :exc:`ValueError` will
    be raised.

    Args:
      field_name (str): The name of the field to check for presence.

    Returns:
      bool: Whether a value has been set for the named field.

    Raises:
      ValueError: if the `field_name` is not a member of this message.
    """
    raise NotImplementedError

  def ClearField(self, field_name):
    """Clears the contents of a given field.

    Inside a oneof group, clears the field set. If the name neither refers to a
    defined field or oneof group, :exc:`ValueError` is raised.

    Args:
      field_name (str): The name of the field to check for presence.

    Raises:
      ValueError: if the `field_name` is not a member of this message.
    """
    raise NotImplementedError

  def WhichOneof(self, oneof_group):
    """Returns the name of the field that is set inside a oneof group.

    If no field is set, returns None.

    Args:
      oneof_group (str): the name of the oneof group to check.

    Returns:
      str or None: The name of the group that is set, or None.

    Raises:
      ValueError: no group with the given name exists
    """
    raise NotImplementedError

  def HasExtension(self, field_descriptor):
    """Checks if a certain extension is present for this message.

    Extensions are retrieved using the :attr:`Extensions` mapping (if present).

    Args:
      field_descriptor: The field descriptor for the extension to check.

    Returns:
      bool: Whether the extension is present for this message.

    Raises:
      KeyError: if the extension is repeated. Similar to repeated fields,
        there is no separate notion of presence: a "not present" repeated
        extension is an empty list.
    """
    raise NotImplementedError

  def ClearExtension(self, field_descriptor):
    """Clears the contents of a given extension.

    Args:
      field_descriptor: The field descriptor for the extension to clear.
    """
    raise NotImplementedError

  def UnknownFields(self):
    """Returns the UnknownFieldSet.

    Returns:
      UnknownFieldSet: The unknown fields stored in this message.
    """
    raise NotImplementedError

  def DiscardUnknownFields(self):
    """Clears all fields in the :class:`UnknownFieldSet`.

    This operation is recursive for nested message.
    """
    raise NotImplementedError

  def ByteSize(self):
    """Returns the serialized size of this message.

    Recursively calls ByteSize() on all contained messages.

    Returns:
      int: The number of bytes required to serialize this message.
    """
    raise NotImplementedError

  @classmethod
  def FromString(cls, s):
    raise NotImplementedError

  def _SetListener(self, message_listener):
    """Internal method used by the protocol message implementation.
    Clients should not call this directly.

    Sets a listener that this message will call on certain state transitions.

    The purpose of this method is to register back-edges from children to
    parents at runtime, for the purpose of setting "has" bits and
    byte-size-dirty bits in the parent and ancestor objects whenever a child or
    descendant object is modified.

    If the client wants to disconnect this Message from the object tree, she
    explicitly sets callback to None.

    If message_listener is None, unregisters any existing listener.  Otherwise,
    message_listener must implement the MessageListener interface in
    internal/message_listener.py, and we discard any listener registered
    via a previous _SetListener() call.
    """
    raise NotImplementedError

  def __getstate__(self):
    """Support the pickle protocol."""
    return dict(serialized=self.SerializePartialToString())

  def __setstate__(self, state):
    """Support the pickle protocol."""
    self.__init__()
    serialized = state['serialized']
    # On Python 3, using encoding='latin1' is required for unpickling
    # protos pickled by Python 2.
    if not isinstance(serialized, bytes):
      serialized = serialized.encode('latin1')
    self.ParseFromString(serialized)

  def __reduce__(self):
    message_descriptor = self.DESCRIPTOR
    if message_descriptor.containing_type is None:
      return type(self), (), self.__getstate__()
    # the message type must be nested.
    # Python does not pickle nested classes; use the symbol_database on the
    # receiving end.
    container = message_descriptor
    return (_InternalConstructMessage, (container.full_name,),
            self.__getstate__())


def _InternalConstructMessage(full_name):
  """Constructs a nested message."""
  from google.protobuf import symbol_database  # pylint:disable=g-import-not-at-top

  return symbol_database.Default().GetSymbol(full_name)()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\fields\core.py ===
import inspect
import itertools

from markupsafe import escape
from markupsafe import Markup

from wtforms import widgets
from wtforms.i18n import DummyTranslations
from wtforms.utils import unset_value
from wtforms.validators import StopValidation
from wtforms.validators import ValidationError


class Field:
    """
    Field base class
    """

    errors = tuple()
    process_errors = tuple()
    raw_data = None
    validators = tuple()
    widget = None
    _formfield = True
    _translations = DummyTranslations()
    do_not_call_in_templates = True  # Allow Django 1.4 traversal

    def __new__(cls, *args, **kwargs):
        if "_form" in kwargs:
            return super().__new__(cls)
        else:
            return UnboundField(cls, *args, **kwargs)

    def __init__(
        self,
        label=None,
        validators=None,
        filters=(),
        description="",
        id=None,
        default=None,
        widget=None,
        render_kw=None,
        name=None,
        _form=None,
        _prefix="",
        _translations=None,
        _meta=None,
    ):
        """
        Construct a new field.

        :param label:
            The label of the field.
        :param validators:
            A sequence of validators to call when `validate` is called.
        :param filters:
            A sequence of callable which are run by :meth:`~Field.process`
            to filter or transform the input data. For example
            ``StringForm(filters=[str.strip, str.upper])``.
            Note that filters are applied after processing the default and
            incoming data, but before validation.
        :param description:
            A description for the field, typically used for help text.
        :param id:
            An id to use for the field. A reasonable default is set by the form,
            and you shouldn't need to set this manually.
        :param default:
            The default value to assign to the field, if no form or object
            input is provided. May be a callable.
        :param widget:
            If provided, overrides the widget used to render the field.
        :param dict render_kw:
            If provided, a dictionary which provides default keywords that
            will be given to the widget at render time.
        :param name:
            The HTML name of this field. The default value is the Python
            attribute name.
        :param _form:
            The form holding this field. It is passed by the form itself during
            construction. You should never pass this value yourself.
        :param _prefix:
            The prefix to prepend to the form name of this field, passed by
            the enclosing form during construction.
        :param _translations:
            A translations object providing message translations. Usually
            passed by the enclosing form during construction. See
            :doc:`I18n docs <i18n>` for information on message translations.
        :param _meta:
            If provided, this is the 'meta' instance from the form. You usually
            don't pass this yourself.

        If `_form` isn't provided, an :class:`UnboundField` will be
        returned instead. Call its :func:`bind` method with a form instance and
        a name to construct the field.
        """
        if _translations is not None:
            self._translations = _translations

        if _meta is not None:
            self.meta = _meta
        elif _form is not None:
            self.meta = _form.meta
        else:
            raise TypeError("Must provide one of _form or _meta")

        self.default = default
        self.description = description
        self.render_kw = render_kw
        self.filters = filters
        self.flags = Flags()
        self.name = _prefix + name
        self.short_name = name
        self.type = type(self).__name__

        self.check_validators(validators)
        self.validators = validators or self.validators

        self.id = id or self.name
        self.label = Label(
            self.id,
            label
            if label is not None
            else self.gettext(name.replace("_", " ").title()),
        )

        if widget is not None:
            self.widget = widget

        for v in itertools.chain(self.validators, [self.widget]):
            flags = getattr(v, "field_flags", {})

            for k, v in flags.items():
                setattr(self.flags, k, v)

    def __str__(self):
        """
        Returns a HTML representation of the field. For more powerful rendering,
        see the `__call__` method.
        """
        return self()

    def __html__(self):
        """
        Returns a HTML representation of the field. For more powerful rendering,
        see the :meth:`__call__` method.
        """
        return self()

    def __call__(self, **kwargs):
        """
        Render this field as HTML, using keyword args as additional attributes.

        This delegates rendering to
        :meth:`meta.render_field <wtforms.meta.DefaultMeta.render_field>`
        whose default behavior is to call the field's widget, passing any
        keyword arguments from this call along to the widget.

        In all of the WTForms HTML widgets, keyword arguments are turned to
        HTML attributes, though in theory a widget is free to do anything it
        wants with the supplied keyword arguments, and widgets don't have to
        even do anything related to HTML.
        """
        return self.meta.render_field(self, kwargs)

    @classmethod
    def check_validators(cls, validators):
        if validators is not None:
            for validator in validators:
                if not callable(validator):
                    raise TypeError(
                        f"{validator} is not a valid validator because it is not "
                        "callable"
                    )

                if inspect.isclass(validator):
                    raise TypeError(
                        f"{validator} is not a valid validator because it is a class, "
                        "it should be an instance"
                    )

    def gettext(self, string):
        """
        Get a translation for the given message.

        This proxies for the internal translations object.

        :param string: A string to be translated.
        :return: A string which is the translated output.
        """
        return self._translations.gettext(string)

    def ngettext(self, singular, plural, n):
        """
        Get a translation for a message which can be pluralized.

        :param str singular: The singular form of the message.
        :param str plural: The plural form of the message.
        :param int n: The number of elements this message is referring to
        """
        return self._translations.ngettext(singular, plural, n)

    def validate(self, form, extra_validators=()):
        """
        Validates the field and returns True or False. `self.errors` will
        contain any errors raised during validation. This is usually only
        called by `Form.validate`.

        Subfields shouldn't override this, but rather override either
        `pre_validate`, `post_validate` or both, depending on needs.

        :param form: The form the field belongs to.
        :param extra_validators: A sequence of extra validators to run.
        """
        self.errors = list(self.process_errors)
        stop_validation = False

        # Check the type of extra_validators
        self.check_validators(extra_validators)

        # Call pre_validate
        try:
            self.pre_validate(form)
        except StopValidation as e:
            if e.args and e.args[0]:
                self.errors.append(e.args[0])
            stop_validation = True
        except ValidationError as e:
            self.errors.append(e.args[0])

        # Run validators
        if not stop_validation:
            chain = itertools.chain(self.validators, extra_validators)
            stop_validation = self._run_validation_chain(form, chain)

        # Call post_validate
        try:
            self.post_validate(form, stop_validation)
        except ValidationError as e:
            self.errors.append(e.args[0])

        return len(self.errors) == 0

    def _run_validation_chain(self, form, validators):
        """
        Run a validation chain, stopping if any validator raises StopValidation.

        :param form: The Form instance this field belongs to.
        :param validators: a sequence or iterable of validator callables.
        :return: True if validation was stopped, False otherwise.
        """
        for validator in validators:
            try:
                validator(form, self)
            except StopValidation as e:
                if e.args and e.args[0]:
                    self.errors.append(e.args[0])
                return True
            except ValidationError as e:
                self.errors.append(e.args[0])

        return False

    def pre_validate(self, form):
        """
        Override if you need field-level validation. Runs before any other
        validators.

        :param form: The form the field belongs to.
        """
        pass

    def post_validate(self, form, validation_stopped):
        """
        Override if you need to run any field-level validation tasks after
        normal validation. This shouldn't be needed in most cases.

        :param form: The form the field belongs to.
        :param validation_stopped:
            `True` if any validator raised StopValidation.
        """
        pass

    def process(self, formdata, data=unset_value, extra_filters=None):
        """
        Process incoming data, calling process_data, process_formdata as needed,
        and run filters.

        If `data` is not provided, process_data will be called on the field's
        default.

        Field subclasses usually won't override this, instead overriding the
        process_formdata and process_data methods. Only override this for
        special advanced processing, such as when a field encapsulates many
        inputs.

        :param extra_filters: A sequence of extra filters to run.
        """
        self.process_errors = []
        if data is unset_value:
            try:
                data = self.default()
            except TypeError:
                data = self.default

        self.object_data = data

        try:
            self.process_data(data)
        except ValueError as e:
            self.process_errors.append(e.args[0])

        if formdata is not None:
            if self.name in formdata:
                self.raw_data = formdata.getlist(self.name)
            else:
                self.raw_data = []

            try:
                self.process_formdata(self.raw_data)
            except ValueError as e:
                self.process_errors.append(e.args[0])

        try:
            for filter in itertools.chain(self.filters, extra_filters or []):
                self.data = filter(self.data)
        except ValueError as e:
            self.process_errors.append(e.args[0])

    def process_data(self, value):
        """
        Process the Python data applied to this field and store the result.

        This will be called during form construction by the form's `kwargs` or
        `obj` argument.

        :param value: The python object containing the value to process.
        """
        self.data = value

    def process_formdata(self, valuelist):
        """
        Process data received over the wire from a form.

        This will be called during form construction with data supplied
        through the `formdata` argument.

        :param valuelist: A list of strings to process.
        """
        if valuelist:
            self.data = valuelist[0]

    def populate_obj(self, obj, name):
        """
        Populates `obj.<name>` with the field's data.

        :note: This is a destructive operation. If `obj.<name>` already exists,
               it will be overridden. Use with caution.
        """
        setattr(obj, name, self.data)


class UnboundField:
    _formfield = True
    creation_counter = 0

    def __init__(self, field_class, *args, name=None, **kwargs):
        UnboundField.creation_counter += 1
        self.field_class = field_class
        self.args = args
        self.name = name
        self.kwargs = kwargs
        self.creation_counter = UnboundField.creation_counter
        validators = kwargs.get("validators")
        if validators:
            self.field_class.check_validators(validators)

    def bind(self, form, name, prefix="", translations=None, **kwargs):
        kw = dict(
            self.kwargs,
            name=name,
            _form=form,
            _prefix=prefix,
            _translations=translations,
            **kwargs,
        )
        return self.field_class(*self.args, **kw)

    def __repr__(self):
        return (
            "<UnboundField("
            f"{self.field_class.__name__}, {self.args!r}, {self.kwargs!r}"
            ")>"
        )


class Flags:
    """
    Holds a set of flags as attributes.

    Accessing a non-existing attribute returns None for its value.
    """

    def __getattr__(self, name):
        if name.startswith("_"):
            return super().__getattr__(name)
        return None

    def __contains__(self, name):
        return getattr(self, name)

    def __repr__(self):
        flags = (
            f"{name}={getattr(self, name)}"
            for name in dir(self)
            if not name.startswith("_")
        )
        flags = ", ".join(flags)
        return f"<wtforms.fields.Flags: {{{flags}}}>"


class Label:
    """
    An HTML form label.
    """

    def __init__(self, field_id, text):
        self.field_id = field_id
        self.text = text

    def __str__(self):
        return self()

    def __html__(self):
        return self()

    def __call__(self, text=None, **kwargs):
        if "for_" in kwargs:
            kwargs["for"] = kwargs.pop("for_")
        else:
            kwargs.setdefault("for", self.field_id)

        attributes = widgets.html_params(**kwargs)
        text = escape(text or self.text)
        return Markup(f"<label {attributes}>{text}</label>")

    def __repr__(self):
        return f"Label({self.field_id!r}, {self.text!r})"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\lib\scimath.py ===
from ._scimath_impl import (  # noqa: F401
    __all__,
    __doc__,
    arccos,
    arcsin,
    arctanh,
    log,
    log2,
    log10,
    logn,
    power,
    sqrt,
)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\io_binding_helper.py ===
import copy
import logging
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy
import torch

from onnxruntime import InferenceSession, RunOptions

# Type alias
ShapeDict = Mapping[str, tuple | list[int]]

logger = logging.getLogger(__name__)


class TypeHelper:
    @staticmethod
    def get_input_type(ort_session: InferenceSession, name: str) -> str:
        for _i, input in enumerate(ort_session.get_inputs()):
            if input.name == name:
                return input.type
        raise ValueError(f"input name {name} not found")

    @staticmethod
    def get_output_type(ort_session, name: str) -> str:
        for _i, output in enumerate(ort_session.get_outputs()):
            if output.name == name:
                return output.type

        raise ValueError(f"output name {name} not found")

    @staticmethod
    def ort_type_to_numpy_type(ort_type: str):
        ort_type_to_numpy_type_map = {
            "tensor(int64)": numpy.longlong,
            "tensor(int32)": numpy.intc,
            "tensor(float)": numpy.float32,
            "tensor(float16)": numpy.float16,
            "tensor(bool)": bool,
            "tensor(uint8)": numpy.uint8,
        }
        if ort_type not in ort_type_to_numpy_type_map:
            raise ValueError(f"{ort_type} not found in map")

        return ort_type_to_numpy_type_map[ort_type]

    @staticmethod
    def ort_type_to_torch_type(ort_type: str):
        ort_type_to_torch_type_map = {
            "tensor(int64)": torch.int64,
            "tensor(int32)": torch.int32,
            "tensor(float)": torch.float32,
            "tensor(float16)": torch.float16,
            "tensor(bool)": torch.bool,
            "tensor(uint8)": torch.uint8,
        }
        if ort_type not in ort_type_to_torch_type_map:
            raise ValueError(f"{ort_type} not found in map")

        return ort_type_to_torch_type_map[ort_type]

    @staticmethod
    def numpy_type_to_torch_type(numpy_type: numpy.dtype):
        numpy_type_to_torch_type_map = {
            numpy.longlong: torch.int64,
            numpy.intc: torch.int32,
            numpy.int32: torch.int32,
            numpy.float32: torch.float32,
            numpy.float16: torch.float16,
            bool: torch.bool,
            numpy.uint8: torch.uint8,
        }
        if numpy_type not in numpy_type_to_torch_type_map:
            raise ValueError(f"{numpy_type} not found in map")

        return numpy_type_to_torch_type_map[numpy_type]

    @staticmethod
    def torch_type_to_numpy_type(torch_type: torch.dtype):
        torch_type_to_numpy_type_map = {
            torch.int64: numpy.longlong,
            torch.int32: numpy.intc,
            torch.float32: numpy.float32,
            torch.float16: numpy.float16,
            torch.bool: bool,
            torch.uint8: numpy.uint8,
        }
        if torch_type not in torch_type_to_numpy_type_map:
            raise ValueError(f"{torch_type} not found in map")

        return torch_type_to_numpy_type_map[torch_type]

    @staticmethod
    def get_io_numpy_type_map(ort_session: InferenceSession) -> dict[str, numpy.dtype]:
        """Create a mapping from input/output name to numpy data type"""
        name_to_numpy_type = {}
        for input in ort_session.get_inputs():
            name_to_numpy_type[input.name] = TypeHelper.ort_type_to_numpy_type(input.type)

        for output in ort_session.get_outputs():
            name_to_numpy_type[output.name] = TypeHelper.ort_type_to_numpy_type(output.type)
        return name_to_numpy_type


class IOBindingHelper:
    @staticmethod
    def get_output_buffers(ort_session: InferenceSession, output_shapes, device):
        """Returns a dictionary of output name as key, and 1D tensor as value. The tensor has enough space for given shape."""
        output_buffers = {}
        for name, shape in output_shapes.items():
            ort_type = TypeHelper.get_output_type(ort_session, name)
            torch_type = TypeHelper.ort_type_to_torch_type(ort_type)
            output_buffers[name] = torch.empty(numpy.prod(shape), dtype=torch_type, device=device)
        return output_buffers

    @staticmethod
    def prepare_io_binding(
        ort_session,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past: list[torch.Tensor],
        output_buffers,
        output_shapes,
        name_to_np_type=None,
    ):
        """Returnas IO binding object for a session."""
        if name_to_np_type is None:
            name_to_np_type = TypeHelper.get_io_numpy_type_map(ort_session)

        # Bind inputs and outputs to onnxruntime session
        io_binding = ort_session.io_binding()

        # Bind inputs
        assert input_ids.is_contiguous()
        io_binding.bind_input(
            "input_ids",
            input_ids.device.type,
            0,
            name_to_np_type["input_ids"],
            list(input_ids.size()),
            input_ids.data_ptr(),
        )

        if past is not None:
            for i, past_i in enumerate(past):
                assert past_i.is_contiguous()

                data_ptr = past_i.data_ptr()
                if data_ptr == 0:
                    # When past_sequence_length is 0, its data_ptr will be zero. IO Binding asserts that data_ptr shall not be zero.
                    # Here we workaround and pass data pointer of input_ids. Actual data is not used for past so it does not matter.
                    data_ptr = input_ids.data_ptr()

                io_binding.bind_input(
                    f"past_{i}",
                    past_i.device.type,
                    0,
                    name_to_np_type[f"past_{i}"],
                    list(past_i.size()),
                    data_ptr,
                )

        if attention_mask is not None:
            assert attention_mask.is_contiguous()
            io_binding.bind_input(
                "attention_mask",
                attention_mask.device.type,
                0,
                name_to_np_type["attention_mask"],
                list(attention_mask.size()),
                attention_mask.data_ptr(),
            )

        if position_ids is not None:
            assert position_ids.is_contiguous()
            io_binding.bind_input(
                "position_ids",
                position_ids.device.type,
                0,
                name_to_np_type["position_ids"],
                list(position_ids.size()),
                position_ids.data_ptr(),
            )

        # Bind outputs
        for output in ort_session.get_outputs():
            output_name = output.name
            output_buffer = output_buffers[output_name]
            logger.debug(f"{output_name} device type={output_buffer.device.type} shape={list(output_buffer.size())}")
            io_binding.bind_output(
                output_name,
                output_buffer.device.type,
                0,
                name_to_np_type[output_name],
                output_shapes[output_name],
                output_buffer.data_ptr(),
            )

        return io_binding

    @staticmethod
    def get_outputs_from_io_binding_buffer(ort_session, output_buffers, output_shapes, return_numpy=True):
        """Copy results to cpu. Returns a list of numpy array."""
        ort_outputs = []
        for output in ort_session.get_outputs():
            output_name = output.name
            buffer = output_buffers[output_name]
            shape = output_shapes[output_name]
            copy_tensor = buffer[0 : numpy.prod(shape)].reshape(shape).clone().detach()
            if return_numpy:
                ort_outputs.append(copy_tensor.cpu().numpy())
            else:
                ort_outputs.append(copy_tensor)
        return ort_outputs


class CudaSession:
    """Inference Session with IO Binding for ONNX Runtime CUDA or TensorRT provider"""

    def __init__(self, ort_session: InferenceSession, device: torch.device, enable_cuda_graph=False):
        self.ort_session = ort_session
        self.input_names = [input.name for input in self.ort_session.get_inputs()]
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        self.io_name_to_numpy_type = TypeHelper.get_io_numpy_type_map(self.ort_session)
        self.io_binding = self.ort_session.io_binding()
        self.enable_cuda_graph = enable_cuda_graph

        self.input_tensors = OrderedDict()
        self.output_tensors = OrderedDict()
        self.device = device

        # Pairs of input and output names that share the same buffer.
        self.buffer_sharing: dict[str, str] = {}

    def set_buffer_sharing(self, input_name: str, output_name: str):
        assert input_name in self.input_names
        assert output_name in self.output_names
        self.buffer_sharing[input_name] = output_name
        self.buffer_sharing[output_name] = input_name

    def __del__(self):
        del self.input_tensors
        del self.output_tensors
        del self.io_binding

    def bind_input_and_buffer_sharing(self, name: str, tensor: torch.Tensor):
        device_id = tensor.device.index if tensor.device.index is not None else 0
        tensor_shape = [1] if len(tensor.shape) == 0 else list(tensor.shape)

        self.io_binding.bind_input(
            name,
            tensor.device.type,
            device_id,
            self.io_name_to_numpy_type[name],
            tensor_shape,
            tensor.data_ptr(),
        )

        if name in self.buffer_sharing:
            self.io_binding.bind_output(
                self.buffer_sharing[name],
                tensor.device.type,
                device_id,
                self.io_name_to_numpy_type[name],
                tensor_shape,
                tensor.data_ptr(),
            )
            self.output_tensors[self.buffer_sharing[name]] = tensor

    def allocate_buffers(self, shape_dict: ShapeDict):
        """Allocate tensors for I/O Binding"""
        if self.enable_cuda_graph:
            for name, shape in shape_dict.items():
                if name in self.input_names:
                    # Reuse allocated buffer when the shape is same
                    if name in self.input_tensors:
                        if tuple(self.input_tensors[name].shape) == tuple(shape):
                            continue
                        raise RuntimeError("Expect static input shape for cuda graph")

                    numpy_dtype = self.io_name_to_numpy_type[name]
                    tensor = torch.empty(tuple(shape), dtype=TypeHelper.numpy_type_to_torch_type(numpy_dtype)).to(
                        device=self.device
                    )
                    self.input_tensors[name] = tensor
                    self.bind_input_and_buffer_sharing(name, tensor)

        for name, shape in shape_dict.items():
            if name in self.output_names:
                # Reuse allocated buffer when the shape is same
                if name in self.output_tensors and tuple(self.output_tensors[name].shape) == tuple(shape):
                    continue

                if name in self.buffer_sharing:
                    continue

                numpy_dtype = self.io_name_to_numpy_type[name]
                tensor = torch.empty(tuple(shape), dtype=TypeHelper.numpy_type_to_torch_type(numpy_dtype)).to(
                    device=self.device
                )
                self.output_tensors[name] = tensor

                self.io_binding.bind_output(
                    name,
                    tensor.device.type,
                    tensor.device.index if tensor.device.index is not None else 0,
                    numpy_dtype,
                    list(tensor.size()),
                    tensor.data_ptr(),
                )

    def infer(self, feed_dict: dict[str, torch.Tensor], run_options: RunOptions = None, synchronize: bool = True):
        """Bind input tensors and run inference"""
        for name, tensor in feed_dict.items():
            assert isinstance(tensor, torch.Tensor) and tensor.is_contiguous()
            if name in self.input_names:
                if self.enable_cuda_graph:
                    assert self.input_tensors[name].nelement() == tensor.nelement()
                    assert self.input_tensors[name].dtype == tensor.dtype
                    assert tensor.device.type == "cuda"
                    self.input_tensors[name].copy_(tensor)
                else:
                    self.bind_input_and_buffer_sharing(name, tensor)

        if synchronize:
            self.io_binding.synchronize_inputs()
            self.ort_session.run_with_iobinding(self.io_binding, run_options)
            self.io_binding.synchronize_outputs()
        else:
            self.ort_session.run_with_iobinding(self.io_binding, run_options)

        return self.output_tensors

    @staticmethod
    def get_cuda_provider_options(device_id: int, enable_cuda_graph: bool, stream: int = 0) -> dict[str, Any]:
        options = {
            "device_id": device_id,
            "arena_extend_strategy": "kSameAsRequested",
            "enable_cuda_graph": enable_cuda_graph,
        }

        # Stream is address of a CUDA stream. 0 means the default stream.
        if stream != 0:
            options["user_compute_stream"] = str(stream)

        return options


class GpuBinding(CudaSession):
    def __init__(
        self,
        ort_session: InferenceSession,
        device: torch.device,
        shape_dict: ShapeDict,
        enable_gpu_graph: bool = False,
        gpu_graph_id: int = -1,
        stream: int = 0,
        buffer_sharing: dict[str, str] | None = None,
    ):
        super().__init__(ort_session, device, enable_gpu_graph)
        if buffer_sharing:
            for input_name, output_name in buffer_sharing.items():
                self.set_buffer_sharing(input_name, output_name)

        self.allocate_buffers(shape_dict)
        self.gpu_graph_id = gpu_graph_id
        # For cuda graph, we need to keep a copy of shape_dict to check if the shape is same in inference later.
        self.shape_dict = copy.deepcopy(shape_dict) if enable_gpu_graph else None
        self.stream = stream
        # The gpu graph id of last run. It will be saved to image metadata.
        self.last_run_gpu_graph_id = None

    def get_run_options(self, disable_cuda_graph_in_run: bool = False) -> RunOptions:
        options = RunOptions()

        gpu_graph_id = -1 if disable_cuda_graph_in_run else self.gpu_graph_id

        options.add_run_config_entry("gpu_graph_id", str(gpu_graph_id))

        self.last_run_gpu_graph_id = gpu_graph_id

        return options

    def infer(self, feed_dict: dict[str, torch.Tensor], disable_cuda_graph_in_run: bool = False):
        run_options = self.get_run_options(disable_cuda_graph_in_run)

        if self.stream:
            run_options.add_run_config_entry("disable_synchronize_execution_providers", "1")

        return super().infer(feed_dict, run_options)


class GpuBindingManager:
    """A manager for I/O bindings that support multiple CUDA Graphs.
    One cuda graph is reused for same input shape. Automatically add a new cuda graph for new input shape.
    """

    def __init__(self, ort_session: InferenceSession, device: torch.device, stream: int = 0, max_cuda_graphs: int = 1):
        self.ort_session = ort_session
        self.device = device

        # Binding supports cuda graphs. For a binding, it is able to disable cuda graph for a specific run.
        self.graph_bindings = []

        # Binding for not using cuda graph.
        self.no_graph_binding = None

        self.stream = stream

        self.max_cuda_graphs = max_cuda_graphs

    def get_binding(
        self,
        shape_dict: ShapeDict,
        use_cuda_graph: bool = False,
        buffer_sharing: dict[str, str] | None = None,
    ) -> GpuBinding:
        for gpu_graph_binding in self.graph_bindings:
            # Found a cuda graph that captured with the same shape
            if gpu_graph_binding.shape_dict == shape_dict:
                return gpu_graph_binding

        # Reached the maximum number of cuda graphs. Return a binding without cuda graph.
        if len(self.graph_bindings) >= self.max_cuda_graphs or (not use_cuda_graph):
            if self.no_graph_binding is None:
                self.no_graph_binding = GpuBinding(
                    self.ort_session, self.device, shape_dict, stream=self.stream, buffer_sharing=buffer_sharing
                )
            else:
                self.no_graph_binding.allocate_buffers(shape_dict)
            return self.no_graph_binding

        # This is a new input shape, create a new cuda graph
        gpu_graph_binding = GpuBinding(
            self.ort_session,
            self.device,
            shape_dict,
            enable_gpu_graph=True,
            gpu_graph_id=len(self.graph_bindings),
            stream=self.stream,
            buffer_sharing=buffer_sharing,
        )
        self.graph_bindings.append(gpu_graph_binding)
        return gpu_graph_binding

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\scripts.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2013-2023 Vinay Sajip.
# Licensed to the Python Software Foundation under a contributor agreement.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
from io import BytesIO
import logging
import os
import re
import struct
import sys
import time
from zipfile import ZipInfo

from .compat import sysconfig, detect_encoding, ZipFile
from .resources import finder
from .util import (FileOperator, get_export_entry, convert_path, get_executable, get_platform, in_venv)

logger = logging.getLogger(__name__)

_DEFAULT_MANIFEST = '''
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
 <assemblyIdentity version="1.0.0.0"
 processorArchitecture="X86"
 name="%s"
 type="win32"/>

 <!-- Identify the application security requirements. -->
 <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
 <security>
 <requestedPrivileges>
 <requestedExecutionLevel level="asInvoker" uiAccess="false"/>
 </requestedPrivileges>
 </security>
 </trustInfo>
</assembly>'''.strip()

# check if Python is called on the first line with this expression
FIRST_LINE_RE = re.compile(b'^#!.*pythonw?[0-9.]*([ \t].*)?$')
SCRIPT_TEMPLATE = r'''# -*- coding: utf-8 -*-
import re
import sys
from %(module)s import %(import_name)s
if __name__ == '__main__':
    sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0])
    sys.exit(%(func)s())
'''

# Pre-fetch the contents of all executable wrapper stubs.
# This is to address https://github.com/pypa/pip/issues/12666.
# When updating pip, we rename the old pip in place before installing the
# new version. If we try to fetch a wrapper *after* that rename, the finder
# machinery will be confused as the package is no longer available at the
# location where it was imported from. So we load everything into memory in
# advance.

if os.name == 'nt' or (os.name == 'java' and os._name == 'nt'):
    # Issue 31: don't hardcode an absolute package name, but
    # determine it relative to the current package
    DISTLIB_PACKAGE = __name__.rsplit('.', 1)[0]

    WRAPPERS = {
        r.name: r.bytes
        for r in finder(DISTLIB_PACKAGE).iterator("")
        if r.name.endswith(".exe")
    }


def enquote_executable(executable):
    if ' ' in executable:
        # make sure we quote only the executable in case of env
        # for example /usr/bin/env "/dir with spaces/bin/jython"
        # instead of "/usr/bin/env /dir with spaces/bin/jython"
        # otherwise whole
        if executable.startswith('/usr/bin/env '):
            env, _executable = executable.split(' ', 1)
            if ' ' in _executable and not _executable.startswith('"'):
                executable = '%s "%s"' % (env, _executable)
        else:
            if not executable.startswith('"'):
                executable = '"%s"' % executable
    return executable


# Keep the old name around (for now), as there is at least one project using it!
_enquote_executable = enquote_executable


class ScriptMaker(object):
    """
    A class to copy or create scripts from source scripts or callable
    specifications.
    """
    script_template = SCRIPT_TEMPLATE

    executable = None  # for shebangs

    def __init__(self, source_dir, target_dir, add_launchers=True, dry_run=False, fileop=None):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.add_launchers = add_launchers
        self.force = False
        self.clobber = False
        # It only makes sense to set mode bits on POSIX.
        self.set_mode = (os.name == 'posix') or (os.name == 'java' and os._name == 'posix')
        self.variants = set(('', 'X.Y'))
        self._fileop = fileop or FileOperator(dry_run)

        self._is_nt = os.name == 'nt' or (os.name == 'java' and os._name == 'nt')
        self.version_info = sys.version_info

    def _get_alternate_executable(self, executable, options):
        if options.get('gui', False) and self._is_nt:  # pragma: no cover
            dn, fn = os.path.split(executable)
            fn = fn.replace('python', 'pythonw')
            executable = os.path.join(dn, fn)
        return executable

    if sys.platform.startswith('java'):  # pragma: no cover

        def _is_shell(self, executable):
            """
            Determine if the specified executable is a script
            (contains a #! line)
            """
            try:
                with open(executable) as fp:
                    return fp.read(2) == '#!'
            except (OSError, IOError):
                logger.warning('Failed to open %s', executable)
                return False

        def _fix_jython_executable(self, executable):
            if self._is_shell(executable):
                # Workaround for Jython is not needed on Linux systems.
                import java

                if java.lang.System.getProperty('os.name') == 'Linux':
                    return executable
            elif executable.lower().endswith('jython.exe'):
                # Use wrapper exe for Jython on Windows
                return executable
            return '/usr/bin/env %s' % executable

    def _build_shebang(self, executable, post_interp):
        """
        Build a shebang line. In the simple case (on Windows, or a shebang line
        which is not too long or contains spaces) use a simple formulation for
        the shebang. Otherwise, use /bin/sh as the executable, with a contrived
        shebang which allows the script to run either under Python or sh, using
        suitable quoting. Thanks to Harald Nordgren for his input.

        See also: http://www.in-ulm.de/~mascheck/various/shebang/#length
                  https://hg.mozilla.org/mozilla-central/file/tip/mach
        """
        if os.name != 'posix':
            simple_shebang = True
        elif getattr(sys, "cross_compiling", False):
            # In a cross-compiling environment, the shebang will likely be a
            # script; this *must* be invoked with the "safe" version of the
            # shebang, or else using os.exec() to run the entry script will
            # fail, raising "OSError 8 [Errno 8] Exec format error".
            simple_shebang = False
        else:
            # Add 3 for '#!' prefix and newline suffix.
            shebang_length = len(executable) + len(post_interp) + 3
            if sys.platform == 'darwin':
                max_shebang_length = 512
            else:
                max_shebang_length = 127
            simple_shebang = ((b' ' not in executable) and (shebang_length <= max_shebang_length))

        if simple_shebang:
            result = b'#!' + executable + post_interp + b'\n'
        else:
            result = b'#!/bin/sh\n'
            result += b"'''exec' " + executable + post_interp + b' "$0" "$@"\n'
            result += b"' '''\n"
        return result

    def _get_shebang(self, encoding, post_interp=b'', options=None):
        enquote = True
        if self.executable:
            executable = self.executable
            enquote = False  # assume this will be taken care of
        elif not sysconfig.is_python_build():
            executable = get_executable()
        elif in_venv():  # pragma: no cover
            executable = os.path.join(sysconfig.get_path('scripts'), 'python%s' % sysconfig.get_config_var('EXE'))
        else:  # pragma: no cover
            if os.name == 'nt':
                # for Python builds from source on Windows, no Python executables with
                # a version suffix are created, so we use python.exe
                executable = os.path.join(sysconfig.get_config_var('BINDIR'),
                                          'python%s' % (sysconfig.get_config_var('EXE')))
            else:
                executable = os.path.join(
                    sysconfig.get_config_var('BINDIR'),
                    'python%s%s' % (sysconfig.get_config_var('VERSION'), sysconfig.get_config_var('EXE')))
        if options:
            executable = self._get_alternate_executable(executable, options)

        if sys.platform.startswith('java'):  # pragma: no cover
            executable = self._fix_jython_executable(executable)

        # Normalise case for Windows - COMMENTED OUT
        # executable = os.path.normcase(executable)
        # N.B. The normalising operation above has been commented out: See
        # issue #124. Although paths in Windows are generally case-insensitive,
        # they aren't always. For example, a path containing a ẞ (which is a
        # LATIN CAPITAL LETTER SHARP S - U+1E9E) is normcased to ß (which is a
        # LATIN SMALL LETTER SHARP S' - U+00DF). The two are not considered by
        # Windows as equivalent in path names.

        # If the user didn't specify an executable, it may be necessary to
        # cater for executable paths with spaces (not uncommon on Windows)
        if enquote:
            executable = enquote_executable(executable)
        # Issue #51: don't use fsencode, since we later try to
        # check that the shebang is decodable using utf-8.
        executable = executable.encode('utf-8')
        # in case of IronPython, play safe and enable frames support
        if (sys.platform == 'cli' and '-X:Frames' not in post_interp and
                '-X:FullFrames' not in post_interp):  # pragma: no cover
            post_interp += b' -X:Frames'
        shebang = self._build_shebang(executable, post_interp)
        # Python parser starts to read a script using UTF-8 until
        # it gets a #coding:xxx cookie. The shebang has to be the
        # first line of a file, the #coding:xxx cookie cannot be
        # written before. So the shebang has to be decodable from
        # UTF-8.
        try:
            shebang.decode('utf-8')
        except UnicodeDecodeError:  # pragma: no cover
            raise ValueError('The shebang (%r) is not decodable from utf-8' % shebang)
        # If the script is encoded to a custom encoding (use a
        # #coding:xxx cookie), the shebang has to be decodable from
        # the script encoding too.
        if encoding != 'utf-8':
            try:
                shebang.decode(encoding)
            except UnicodeDecodeError:  # pragma: no cover
                raise ValueError('The shebang (%r) is not decodable '
                                 'from the script encoding (%r)' % (shebang, encoding))
        return shebang

    def _get_script_text(self, entry):
        return self.script_template % dict(
            module=entry.prefix, import_name=entry.suffix.split('.')[0], func=entry.suffix)

    manifest = _DEFAULT_MANIFEST

    def get_manifest(self, exename):
        base = os.path.basename(exename)
        return self.manifest % base

    def _write_script(self, names, shebang, script_bytes, filenames, ext):
        use_launcher = self.add_launchers and self._is_nt
        if not use_launcher:
            script_bytes = shebang + script_bytes
        else:  # pragma: no cover
            if ext == 'py':
                launcher = self._get_launcher('t')
            else:
                launcher = self._get_launcher('w')
            stream = BytesIO()
            with ZipFile(stream, 'w') as zf:
                source_date_epoch = os.environ.get('SOURCE_DATE_EPOCH')
                if source_date_epoch:
                    date_time = time.gmtime(int(source_date_epoch))[:6]
                    zinfo = ZipInfo(filename='__main__.py', date_time=date_time)
                    zf.writestr(zinfo, script_bytes)
                else:
                    zf.writestr('__main__.py', script_bytes)
            zip_data = stream.getvalue()
            script_bytes = launcher + shebang + zip_data
        for name in names:
            outname = os.path.join(self.target_dir, name)
            if use_launcher:  # pragma: no cover
                n, e = os.path.splitext(outname)
                if e.startswith('.py'):
                    outname = n
                outname = '%s.exe' % outname
                try:
                    self._fileop.write_binary_file(outname, script_bytes)
                except Exception:
                    # Failed writing an executable - it might be in use.
                    logger.warning('Failed to write executable - trying to '
                                   'use .deleteme logic')
                    dfname = '%s.deleteme' % outname
                    if os.path.exists(dfname):
                        os.remove(dfname)  # Not allowed to fail here
                    os.rename(outname, dfname)  # nor here
                    self._fileop.write_binary_file(outname, script_bytes)
                    logger.debug('Able to replace executable using '
                                 '.deleteme logic')
                    try:
                        os.remove(dfname)
                    except Exception:
                        pass  # still in use - ignore error
            else:
                if self._is_nt and not outname.endswith('.' + ext):  # pragma: no cover
                    outname = '%s.%s' % (outname, ext)
                if os.path.exists(outname) and not self.clobber:
                    logger.warning('Skipping existing file %s', outname)
                    continue
                self._fileop.write_binary_file(outname, script_bytes)
                if self.set_mode:
                    self._fileop.set_executable_mode([outname])
            filenames.append(outname)

    variant_separator = '-'

    def get_script_filenames(self, name):
        result = set()
        if '' in self.variants:
            result.add(name)
        if 'X' in self.variants:
            result.add('%s%s' % (name, self.version_info[0]))
        if 'X.Y' in self.variants:
            result.add('%s%s%s.%s' % (name, self.variant_separator, self.version_info[0], self.version_info[1]))
        return result

    def _make_script(self, entry, filenames, options=None):
        post_interp = b''
        if options:
            args = options.get('interpreter_args', [])
            if args:
                args = ' %s' % ' '.join(args)
                post_interp = args.encode('utf-8')
        shebang = self._get_shebang('utf-8', post_interp, options=options)
        script = self._get_script_text(entry).encode('utf-8')
        scriptnames = self.get_script_filenames(entry.name)
        if options and options.get('gui', False):
            ext = 'pyw'
        else:
            ext = 'py'
        self._write_script(scriptnames, shebang, script, filenames, ext)

    def _copy_script(self, script, filenames):
        adjust = False
        script = os.path.join(self.source_dir, convert_path(script))
        outname = os.path.join(self.target_dir, os.path.basename(script))
        if not self.force and not self._fileop.newer(script, outname):
            logger.debug('not copying %s (up-to-date)', script)
            return

        # Always open the file, but ignore failures in dry-run mode --
        # that way, we'll get accurate feedback if we can read the
        # script.
        try:
            f = open(script, 'rb')
        except IOError:  # pragma: no cover
            if not self.dry_run:
                raise
            f = None
        else:
            first_line = f.readline()
            if not first_line:  # pragma: no cover
                logger.warning('%s is an empty file (skipping)', script)
                return

            match = FIRST_LINE_RE.match(first_line.replace(b'\r\n', b'\n'))
            if match:
                adjust = True
                post_interp = match.group(1) or b''

        if not adjust:
            if f:
                f.close()
            self._fileop.copy_file(script, outname)
            if self.set_mode:
                self._fileop.set_executable_mode([outname])
            filenames.append(outname)
        else:
            logger.info('copying and adjusting %s -> %s', script, self.target_dir)
            if not self._fileop.dry_run:
                encoding, lines = detect_encoding(f.readline)
                f.seek(0)
                shebang = self._get_shebang(encoding, post_interp)
                if b'pythonw' in first_line:  # pragma: no cover
                    ext = 'pyw'
                else:
                    ext = 'py'
                n = os.path.basename(outname)
                self._write_script([n], shebang, f.read(), filenames, ext)
            if f:
                f.close()

    @property
    def dry_run(self):
        return self._fileop.dry_run

    @dry_run.setter
    def dry_run(self, value):
        self._fileop.dry_run = value

    if os.name == 'nt' or (os.name == 'java' and os._name == 'nt'):  # pragma: no cover
        # Executable launcher support.
        # Launchers are from https://bitbucket.org/vinay.sajip/simple_launcher/

        def _get_launcher(self, kind):
            if struct.calcsize('P') == 8:  # 64-bit
                bits = '64'
            else:
                bits = '32'
            platform_suffix = '-arm' if get_platform() == 'win-arm64' else ''
            name = '%s%s%s.exe' % (kind, bits, platform_suffix)
            if name not in WRAPPERS:
                msg = ('Unable to find resource %s in package %s' %
                       (name, DISTLIB_PACKAGE))
                raise ValueError(msg)
            return WRAPPERS[name]

    # Public API follows

    def make(self, specification, options=None):
        """
        Make a script.

        :param specification: The specification, which is either a valid export
                              entry specification (to make a script from a
                              callable) or a filename (to make a script by
                              copying from a source location).
        :param options: A dictionary of options controlling script generation.
        :return: A list of all absolute pathnames written to.
        """
        filenames = []
        entry = get_export_entry(specification)
        if entry is None:
            self._copy_script(specification, filenames)
        else:
            self._make_script(entry, filenames, options=options)
        return filenames

    def make_multiple(self, specifications, options=None):
        """
        Take a list of specifications and make scripts from them,
        :param specifications: A list of specifications.
        :return: A list of all absolute pathnames written to,
        """
        filenames = []
        for specification in specifications:
            filenames.extend(self.make(specification, options))
        return filenames

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\formula\tokenizer.py ===
"""
This module contains a tokenizer for Excel formulae.

The tokenizer is based on the Javascript tokenizer found at
http://ewbi.blogs.com/develops/2004/12/excel_formula_p.html written by Eric
Bachtal
"""

import re


class TokenizerError(Exception):
    """Base class for all Tokenizer errors."""


class Tokenizer:

    """
    A tokenizer for Excel worksheet formulae.

    Converts a str string representing an Excel formula (in A1 notation)
    into a sequence of `Token` objects.

    `formula`: The str string to tokenize

    Tokenizer defines a method `._parse()` to parse the formula into tokens,
    which can then be accessed through the `.items` attribute.

    """

    SN_RE = re.compile("^[1-9](\\.[0-9]+)?[Ee]$")  # Scientific notation
    WSPACE_RE = re.compile(r"[ \n]+")
    STRING_REGEXES = {
        # Inside a string, all characters are treated as literals, except for
        # the quote character used to start the string. That character, when
        # doubled is treated as a single character in the string. If an
        # unmatched quote appears, the string is terminated.
        '"': re.compile('"(?:[^"]*"")*[^"]*"(?!")'),
        "'": re.compile("'(?:[^']*'')*[^']*'(?!')"),
    }
    ERROR_CODES = ("#NULL!", "#DIV/0!", "#VALUE!", "#REF!", "#NAME?",
                   "#NUM!", "#N/A", "#GETTING_DATA")
    TOKEN_ENDERS = ',;}) +-*/^&=><%'  # Each of these characters, marks the
                                       # end of an operand token

    def __init__(self, formula):
        self.formula = formula
        self.items = []
        self.token_stack = []  # Used to keep track of arrays, functions, and
                               # parentheses
        self.offset = 0  # How many chars have we read
        self.token = []  # Used to build up token values char by char
        self._parse()

    def _parse(self):
        """Populate self.items with the tokens from the formula."""
        if self.offset:
            return  # Already parsed!
        if not self.formula:
            return
        elif self.formula[0] == '=':
            self.offset += 1
        else:
            self.items.append(Token(self.formula, Token.LITERAL))
            return
        consumers = (
            ('"\'', self._parse_string),
            ('[', self._parse_brackets),
            ('#', self._parse_error),
            (' ', self._parse_whitespace),
            ('\n', self._parse_whitespace),
            ('+-*/^&=><%', self._parse_operator),
            ('{(', self._parse_opener),
            (')}', self._parse_closer),
            (';,', self._parse_separator),
        )
        dispatcher = {}  # maps chars to the specific parsing function
        for chars, consumer in consumers:
            dispatcher.update(dict.fromkeys(chars, consumer))
        while self.offset < len(self.formula):
            if self.check_scientific_notation():  # May consume one character
                continue
            curr_char = self.formula[self.offset]
            if curr_char in self.TOKEN_ENDERS:
                self.save_token()
            if curr_char in dispatcher:
                self.offset += dispatcher[curr_char]()
            else:
                # TODO: this can probably be sped up using a regex to get to
                # the next interesting character
                self.token.append(curr_char)
                self.offset += 1
        self.save_token()

    def _parse_string(self):
        """
        Parse a "-delimited string or '-delimited link.

        The offset must be pointing to either a single quote ("'") or double
        quote ('"') character. The strings are parsed according to Excel
        rules where to escape the delimiter you just double it up. E.g.,
        "abc""def" in Excel is parsed as 'abc"def' in Python.

        Returns the number of characters matched. (Does not update
        self.offset)

        """
        self.assert_empty_token(can_follow=':')
        delim = self.formula[self.offset]
        assert delim in ('"', "'")
        regex = self.STRING_REGEXES[delim]
        match = regex.match(self.formula[self.offset:])
        if match is None:
            subtype = "string" if delim == '"' else 'link'
            raise TokenizerError(f"Reached end of formula while parsing {subtype} in {self.formula}")
        match = match.group(0)
        if delim == '"':
            self.items.append(Token.make_operand(match))
        else:
            self.token.append(match)
        return len(match)

    def _parse_brackets(self):
        """
        Consume all the text between square brackets [].

        Returns the number of characters matched. (Does not update
        self.offset)

        """
        assert self.formula[self.offset] == '['
        lefts = [(t.start(), 1) for t in
                 re.finditer(r"\[", self.formula[self.offset:])]
        rights = [(t.start(), -1) for t in
                  re.finditer(r"\]", self.formula[self.offset:])]

        open_count = 0
        for idx, open_close in sorted(lefts + rights):
            open_count += open_close
            if open_count == 0:
                outer_right = idx + 1
                self.token.append(
                    self.formula[self.offset:self.offset + outer_right])
                return outer_right

        raise TokenizerError(f"Encountered unmatched '[' in {self.formula}")

    def _parse_error(self):
        """
        Consume the text following a '#' as an error.

        Looks for a match in self.ERROR_CODES and returns the number of
        characters matched. (Does not update self.offset)

        """
        self.assert_empty_token(can_follow='!')
        assert self.formula[self.offset] == '#'
        subformula = self.formula[self.offset:]
        for err in self.ERROR_CODES:
            if subformula.startswith(err):
                self.items.append(Token.make_operand(''.join(self.token) + err))
                del self.token[:]
                return len(err)
        raise TokenizerError(f"Invalid error code at position {self.offset} in '{self.formula}'")

    def _parse_whitespace(self):
        """
        Consume a string of consecutive spaces.

        Returns the number of spaces found. (Does not update self.offset).

        """
        assert self.formula[self.offset] in (' ', '\n')
        self.items.append(Token(self.formula[self.offset], Token.WSPACE))
        return self.WSPACE_RE.match(self.formula[self.offset:]).end()

    def _parse_operator(self):
        """
        Consume the characters constituting an operator.

        Returns the number of characters consumed. (Does not update
        self.offset)

        """
        if self.formula[self.offset:self.offset + 2] in ('>=', '<=', '<>'):
            self.items.append(Token(
                self.formula[self.offset:self.offset + 2],
                Token.OP_IN
            ))
            return 2
        curr_char = self.formula[self.offset]  # guaranteed to be 1 char
        assert curr_char in '%*/^&=><+-'
        if curr_char == '%':
            token = Token('%', Token.OP_POST)
        elif curr_char in "*/^&=><":
            token = Token(curr_char, Token.OP_IN)
        # From here on, curr_char is guaranteed to be in '+-'
        elif not self.items:
            token = Token(curr_char, Token.OP_PRE)
        else:
            prev = next((i for i in reversed(self.items)
                         if i.type != Token.WSPACE), None)
            is_infix = prev and (
                prev.subtype == Token.CLOSE
                or prev.type == Token.OP_POST
                or prev.type == Token.OPERAND
            )
            if is_infix:
                token = Token(curr_char, Token.OP_IN)
            else:
                token = Token(curr_char, Token.OP_PRE)
        self.items.append(token)
        return 1

    def _parse_opener(self):
        """
        Consumes a ( or { character.

        Returns the number of characters consumed. (Does not update
        self.offset)

        """
        assert self.formula[self.offset] in ('(', '{')
        if self.formula[self.offset] == '{':
            self.assert_empty_token()
            token = Token.make_subexp("{")
        elif self.token:
            token_value = "".join(self.token) + '('
            del self.token[:]
            token = Token.make_subexp(token_value)
        else:
            token = Token.make_subexp("(")
        self.items.append(token)
        self.token_stack.append(token)
        return 1

    def _parse_closer(self):
        """
        Consumes a } or ) character.

        Returns the number of characters consumed. (Does not update
        self.offset)

        """
        assert self.formula[self.offset] in (')', '}')
        token = self.token_stack.pop().get_closer()
        if token.value != self.formula[self.offset]:
            raise TokenizerError(
                "Mismatched ( and { pair in '%s'" % self.formula)
        self.items.append(token)
        return 1

    def _parse_separator(self):
        """
        Consumes a ; or , character.

        Returns the number of characters consumed. (Does not update
        self.offset)

        """
        curr_char = self.formula[self.offset]
        assert curr_char in (';', ',')
        if curr_char == ';':
            token = Token.make_separator(";")
        else:
            try:
                top_type = self.token_stack[-1].type
            except IndexError:
                token = Token(",", Token.OP_IN)  # Range Union operator
            else:
                if top_type == Token.PAREN:
                    token = Token(",", Token.OP_IN)  # Range Union operator
                else:
                    token = Token.make_separator(",")
        self.items.append(token)
        return 1

    def check_scientific_notation(self):
        """
        Consumes a + or - character if part of a number in sci. notation.

        Returns True if the character was consumed and self.offset was
        updated, False otherwise.

        """
        curr_char = self.formula[self.offset]
        if (curr_char in '+-'
                and len(self.token) >= 1
                and self.SN_RE.match("".join(self.token))):
            self.token.append(curr_char)
            self.offset += 1
            return True
        return False

    def assert_empty_token(self, can_follow=()):
        """
        Ensure that there's no token currently being parsed.

        Or if there is a token being parsed, it must end with a character in
        can_follow.

        If there are unconsumed token contents, it means we hit an unexpected
        token transition. In this case, we raise a TokenizerError

        """
        if self.token and self.token[-1] not in can_follow:
            raise TokenizerError(f"Unexpected character at position {self.offset} in '{self.formula}'")

    def save_token(self):
        """If there's a token being parsed, add it to the item list."""
        if self.token:
            self.items.append(Token.make_operand("".join(self.token)))
            del self.token[:]

    def render(self):
        """Convert the parsed tokens back to a string."""
        if not self.items:
            return ""
        elif self.items[0].type == Token.LITERAL:
            return self.items[0].value
        return "=" + "".join(token.value for token in self.items)


class Token:

    """
    A token in an Excel formula.

    Tokens have three attributes:

    * `value`: The string value parsed that led to this token
    * `type`: A string identifying the type of token
    * `subtype`: A string identifying subtype of the token (optional, and
                 defaults to "")

    """

    __slots__ = ['value', 'type', 'subtype']

    LITERAL = "LITERAL"
    OPERAND = "OPERAND"
    FUNC = "FUNC"
    ARRAY = "ARRAY"
    PAREN = "PAREN"
    SEP = "SEP"
    OP_PRE = "OPERATOR-PREFIX"
    OP_IN = "OPERATOR-INFIX"
    OP_POST = "OPERATOR-POSTFIX"
    WSPACE = "WHITE-SPACE"

    def __init__(self, value, type_, subtype=""):
        self.value = value
        self.type = type_
        self.subtype = subtype

    # Literal operands:
    #
    # Literal operands are always of type 'OPERAND' and can be of subtype
    # 'TEXT' (for text strings), 'NUMBER' (for all numeric types), 'LOGICAL'
    # (for TRUE and FALSE), 'ERROR' (for literal error values), or 'RANGE'
    # (for all range references).

    TEXT = 'TEXT'
    NUMBER = 'NUMBER'
    LOGICAL = 'LOGICAL'
    ERROR = 'ERROR'
    RANGE = 'RANGE'

    def __repr__(self):
        return u"{0} {1} {2}:".format(self.type, self.subtype, self.value)

    @classmethod
    def make_operand(cls, value):
        """Create an operand token."""
        if value.startswith('"'):
            subtype = cls.TEXT
        elif value.startswith('#'):
            subtype = cls.ERROR
        elif value in ('TRUE', 'FALSE'):
            subtype = cls.LOGICAL
        else:
            try:
                float(value)
                subtype = cls.NUMBER
            except ValueError:
                subtype = cls.RANGE
        return cls(value, cls.OPERAND, subtype)


    # Subexpresssions
    #
    # There are 3 types of `Subexpressions`: functions, array literals, and
    # parentheticals. Subexpressions have 'OPEN' and 'CLOSE' tokens. 'OPEN'
    # is used when parsing the initial expression token (i.e., '(' or '{')
    # and 'CLOSE' is used when parsing the closing expression token ('}' or
    # ')').

    OPEN = "OPEN"
    CLOSE = "CLOSE"

    @classmethod
    def make_subexp(cls, value, func=False):
        """
        Create a subexpression token.

        `value`: The value of the token
        `func`: If True, force the token to be of type FUNC

        """
        assert value[-1] in ('{', '}', '(', ')')
        if func:
            assert re.match('.+\\(|\\)', value)
            type_ = Token.FUNC
        elif value in '{}':
            type_ = Token.ARRAY
        elif value in '()':
            type_ = Token.PAREN
        else:
            type_ = Token.FUNC
        subtype = cls.CLOSE if value in ')}' else cls.OPEN
        return cls(value, type_, subtype)

    def get_closer(self):
        """Return a closing token that matches this token's type."""
        assert self.type in (self.FUNC, self.ARRAY, self.PAREN)
        assert self.subtype == self.OPEN
        value = "}" if self.type == self.ARRAY else ")"
        return self.make_subexp(value, func=self.type == self.FUNC)

    # Separator tokens
    #
    # Argument separators always have type 'SEP' and can have one of two
    # subtypes: 'ARG', 'ROW'. 'ARG' is used for the ',' token, when used to
    # delimit either function arguments or array elements. 'ROW' is used for
    # the ';' token, which is always used to delimit rows in an array
    # literal.

    ARG = "ARG"
    ROW = "ROW"

    @classmethod
    def make_separator(cls, value):
        """Create a separator token"""
        assert value in (',', ';')
        subtype = cls.ARG if value == ',' else cls.ROW
        return cls(value, cls.SEP, subtype)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\cnf.py ===
"""
The classes used here are for the internal use of assumptions system
only and should not be used anywhere else as these do not possess the
signatures common to SymPy objects. For general use of logic constructs
please refer to sympy.logic classes And, Or, Not, etc.
"""
from itertools import combinations, product, zip_longest
from sympy.assumptions.assume import AppliedPredicate, Predicate
from sympy.core.relational import Eq, Ne, Gt, Lt, Ge, Le
from sympy.core.singleton import S
from sympy.logic.boolalg import Or, And, Not, Xnor
from sympy.logic.boolalg import (Equivalent, ITE, Implies, Nand, Nor, Xor)


class Literal:
    """
    The smallest element of a CNF object.

    Parameters
    ==========

    lit : Boolean expression

    is_Not : bool

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import Literal
    >>> from sympy.abc import x
    >>> Literal(Q.even(x))
    Literal(Q.even(x), False)
    >>> Literal(~Q.even(x))
    Literal(Q.even(x), True)
    """

    def __new__(cls, lit, is_Not=False):
        if isinstance(lit, Not):
            lit = lit.args[0]
            is_Not = True
        elif isinstance(lit, (AND, OR, Literal)):
            return ~lit if is_Not else lit
        obj = super().__new__(cls)
        obj.lit = lit
        obj.is_Not = is_Not
        return obj

    @property
    def arg(self):
        return self.lit

    def rcall(self, expr):
        if callable(self.lit):
            lit = self.lit(expr)
        else:
            lit = self.lit.apply(expr)
        return type(self)(lit, self.is_Not)

    def __invert__(self):
        is_Not = not self.is_Not
        return Literal(self.lit, is_Not)

    def __str__(self):
        return '{}({}, {})'.format(type(self).__name__, self.lit, self.is_Not)

    __repr__ = __str__

    def __eq__(self, other):
        return self.arg == other.arg and self.is_Not == other.is_Not

    def __hash__(self):
        h = hash((type(self).__name__, self.arg, self.is_Not))
        return h


class OR:
    """
    A low-level implementation for Or
    """
    def __init__(self, *args):
        self._args = args

    @property
    def args(self):
        return sorted(self._args, key=str)

    def rcall(self, expr):
        return type(self)(*[arg.rcall(expr)
                            for arg in self._args
                            ])

    def __invert__(self):
        return AND(*[~arg for arg in self._args])

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(self, other):
        return self.args == other.args

    def __str__(self):
        s = '(' + ' | '.join([str(arg) for arg in self.args]) + ')'
        return s

    __repr__ = __str__


class AND:
    """
    A low-level implementation for And
    """
    def __init__(self, *args):
        self._args = args

    def __invert__(self):
        return OR(*[~arg for arg in self._args])

    @property
    def args(self):
        return sorted(self._args, key=str)

    def rcall(self, expr):
        return type(self)(*[arg.rcall(expr)
                            for arg in self._args
                            ])

    def __hash__(self):
        return hash((type(self).__name__,) + tuple(self.args))

    def __eq__(self, other):
        return self.args == other.args

    def __str__(self):
        s = '('+' & '.join([str(arg) for arg in self.args])+')'
        return s

    __repr__ = __str__


def to_NNF(expr, composite_map=None):
    """
    Generates the Negation Normal Form of any boolean expression in terms
    of AND, OR, and Literal objects.

    Examples
    ========

    >>> from sympy import Q, Eq
    >>> from sympy.assumptions.cnf import to_NNF
    >>> from sympy.abc import x, y
    >>> expr = Q.even(x) & ~Q.positive(x)
    >>> to_NNF(expr)
    (Literal(Q.even(x), False) & Literal(Q.positive(x), True))

    Supported boolean objects are converted to corresponding predicates.

    >>> to_NNF(Eq(x, y))
    Literal(Q.eq(x, y), False)

    If ``composite_map`` argument is given, ``to_NNF`` decomposes the
    specified predicate into a combination of primitive predicates.

    >>> cmap = {Q.nonpositive: Q.negative | Q.zero}
    >>> to_NNF(Q.nonpositive, cmap)
    (Literal(Q.negative, False) | Literal(Q.zero, False))
    >>> to_NNF(Q.nonpositive(x), cmap)
    (Literal(Q.negative(x), False) | Literal(Q.zero(x), False))
    """
    from sympy.assumptions.ask import Q

    if composite_map is None:
        composite_map = {}


    binrelpreds = {Eq: Q.eq, Ne: Q.ne, Gt: Q.gt, Lt: Q.lt, Ge: Q.ge, Le: Q.le}
    if type(expr) in binrelpreds:
        pred = binrelpreds[type(expr)]
        expr = pred(*expr.args)

    if isinstance(expr, Not):
        arg = expr.args[0]
        tmp = to_NNF(arg, composite_map)  # Strategy: negate the NNF of expr
        return ~tmp

    if isinstance(expr, Or):
        return OR(*[to_NNF(x, composite_map) for x in Or.make_args(expr)])

    if isinstance(expr, And):
        return AND(*[to_NNF(x, composite_map) for x in And.make_args(expr)])

    if isinstance(expr, Nand):
        tmp = AND(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp

    if isinstance(expr, Nor):
        tmp = OR(*[to_NNF(x, composite_map) for x in expr.args])
        return ~tmp

    if isinstance(expr, Xor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map)
                          for s in expr.args]
                cnfs.append(OR(*clause))
        return AND(*cnfs)

    if isinstance(expr, Xnor):
        cnfs = []
        for i in range(0, len(expr.args) + 1, 2):
            for neg in combinations(expr.args, i):
                clause = [~to_NNF(s, composite_map) if s in neg else to_NNF(s, composite_map)
                          for s in expr.args]
                cnfs.append(OR(*clause))
        return ~AND(*cnfs)

    if isinstance(expr, Implies):
        L, R = to_NNF(expr.args[0], composite_map), to_NNF(expr.args[1], composite_map)
        return OR(~L, R)

    if isinstance(expr, Equivalent):
        cnfs = []
        for a, b in zip_longest(expr.args, expr.args[1:], fillvalue=expr.args[0]):
            a = to_NNF(a, composite_map)
            b = to_NNF(b, composite_map)
            cnfs.append(OR(~a, b))
        return AND(*cnfs)

    if isinstance(expr, ITE):
        L = to_NNF(expr.args[0], composite_map)
        M = to_NNF(expr.args[1], composite_map)
        R = to_NNF(expr.args[2], composite_map)
        return AND(OR(~L, M), OR(L, R))

    if isinstance(expr, AppliedPredicate):
        pred, args = expr.function, expr.arguments
        newpred = composite_map.get(pred, None)
        if newpred is not None:
            return to_NNF(newpred.rcall(*args), composite_map)

    if isinstance(expr, Predicate):
        newpred = composite_map.get(expr, None)
        if newpred is not None:
            return to_NNF(newpred, composite_map)

    return Literal(expr)


def distribute_AND_over_OR(expr):
    """
    Distributes AND over OR in the NNF expression.
    Returns the result( Conjunctive Normal Form of expression)
    as a CNF object.
    """
    if not isinstance(expr, (AND, OR)):
        tmp = set()
        tmp.add(frozenset((expr,)))
        return CNF(tmp)

    if isinstance(expr, OR):
        return CNF.all_or(*[distribute_AND_over_OR(arg)
                            for arg in expr._args])

    if isinstance(expr, AND):
        return CNF.all_and(*[distribute_AND_over_OR(arg)
                             for arg in expr._args])


class CNF:
    """
    Class to represent CNF of a Boolean expression.
    Consists of set of clauses, which themselves are stored as
    frozenset of Literal objects.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.cnf import CNF
    >>> from sympy.abc import x
    >>> cnf = CNF.from_prop(Q.real(x) & ~Q.zero(x))
    >>> cnf.clauses
    {frozenset({Literal(Q.zero(x), True)}),
    frozenset({Literal(Q.negative(x), False),
    Literal(Q.positive(x), False), Literal(Q.zero(x), False)})}
    """
    def __init__(self, clauses=None):
        if not clauses:
            clauses = set()
        self.clauses = clauses

    def add(self, prop):
        clauses = CNF.to_CNF(prop).clauses
        self.add_clauses(clauses)

    def __str__(self):
        s = ' & '.join(
            ['(' + ' | '.join([str(lit) for lit in clause]) +')'
            for clause in self.clauses]
        )
        return s

    def extend(self, props):
        for p in props:
            self.add(p)
        return self

    def copy(self):
        return CNF(set(self.clauses))

    def add_clauses(self, clauses):
        self.clauses |= clauses

    @classmethod
    def from_prop(cls, prop):
        res = cls()
        res.add(prop)
        return res

    def __iand__(self, other):
        self.add_clauses(other.clauses)
        return self

    def all_predicates(self):
        predicates = set()
        for c in self.clauses:
            predicates |= {arg.lit for arg in c}
        return predicates

    def _or(self, cnf):
        clauses = set()
        for a, b in product(self.clauses, cnf.clauses):
            tmp = set(a)
            tmp.update(b)
            clauses.add(frozenset(tmp))
        return CNF(clauses)

    def _and(self, cnf):
        clauses = self.clauses.union(cnf.clauses)
        return CNF(clauses)

    def _not(self):
        clss = list(self.clauses)
        ll = {frozenset((~x,)) for x in clss[-1]}
        ll = CNF(ll)

        for rest in clss[:-1]:
            p = {frozenset((~x,)) for x in rest}
            ll = ll._or(CNF(p))
        return ll

    def rcall(self, expr):
        clause_list = []
        for clause in self.clauses:
            lits = [arg.rcall(expr) for arg in clause]
            clause_list.append(OR(*lits))
        expr = AND(*clause_list)
        return distribute_AND_over_OR(expr)

    @classmethod
    def all_or(cls, *cnfs):
        b = cnfs[0].copy()
        for rest in cnfs[1:]:
            b = b._or(rest)
        return b

    @classmethod
    def all_and(cls, *cnfs):
        b = cnfs[0].copy()
        for rest in cnfs[1:]:
            b = b._and(rest)
        return b

    @classmethod
    def to_CNF(cls, expr):
        from sympy.assumptions.facts import get_composite_predicates
        expr = to_NNF(expr, get_composite_predicates())
        expr = distribute_AND_over_OR(expr)
        return expr

    @classmethod
    def CNF_to_cnf(cls, cnf):
        """
        Converts CNF object to SymPy's boolean expression
        retaining the form of expression.
        """
        def remove_literal(arg):
            return Not(arg.lit) if arg.is_Not else arg.lit

        return And(*(Or(*(remove_literal(arg) for arg in clause)) for clause in cnf.clauses))


class EncodedCNF:
    """
    Class for encoding the CNF expression.
    """
    def __init__(self, data=None, encoding=None):
        if not data and not encoding:
            data = []
            encoding = {}
        self.data = data
        self.encoding = encoding
        self._symbols = list(encoding.keys())

    def from_cnf(self, cnf):
        self._symbols = list(cnf.all_predicates())
        n = len(self._symbols)
        self.encoding = dict(zip(self._symbols, range(1, n + 1)))
        self.data = [self.encode(clause) for clause in cnf.clauses]

    @property
    def symbols(self):
        return self._symbols

    @property
    def variables(self):
        return range(1, len(self._symbols) + 1)

    def copy(self):
        new_data = [set(clause) for clause in self.data]
        return EncodedCNF(new_data, dict(self.encoding))

    def add_prop(self, prop):
        cnf = CNF.from_prop(prop)
        self.add_from_cnf(cnf)

    def add_from_cnf(self, cnf):
        clauses = [self.encode(clause) for clause in cnf.clauses]
        self.data += clauses

    def encode_arg(self, arg):
        literal = arg.lit
        value = self.encoding.get(literal, None)
        if value is None:
            n = len(self._symbols)
            self._symbols.append(literal)
            value = self.encoding[literal] = n + 1
        if arg.is_Not:
            return -value
        else:
            return value

    def encode(self, clause):
        return {self.encode_arg(arg) if not arg.lit == S.false else 0 for arg in clause}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\functions\special\elliptic_integrals.py ===
""" Elliptic Integrals. """

from sympy.core import S, pi, I, Rational
from sympy.core.function import DefinedFunction, ArgumentIndexError
from sympy.core.symbol import Dummy,uniquely_named_symbol
from sympy.functions.elementary.complexes import sign
from sympy.functions.elementary.hyperbolic import atanh
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import sin, tan
from sympy.functions.special.gamma_functions import gamma
from sympy.functions.special.hyper import hyper, meijerg

class elliptic_k(DefinedFunction):
    r"""
    The complete elliptic integral of the first kind, defined by

    .. math:: K(m) = F\left(\tfrac{\pi}{2}\middle| m\right)

    where $F\left(z\middle| m\right)$ is the Legendre incomplete
    elliptic integral of the first kind.

    Explanation
    ===========

    The function $K(m)$ is a single-valued function on the complex
    plane with branch cut along the interval $(1, \infty)$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_k, I
    >>> from sympy.abc import m
    >>> elliptic_k(0)
    pi/2
    >>> elliptic_k(1.0 + I)
    1.50923695405127 + 0.625146415202697*I
    >>> elliptic_k(m).series(n=3)
    pi/2 + pi*m/8 + 9*pi*m**2/128 + O(m**3)

    See Also
    ========

    elliptic_f

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticK

    """

    @classmethod
    def eval(cls, m):
        if m.is_zero:
            return pi*S.Half
        elif m is S.Half:
            return 8*pi**Rational(3, 2)/gamma(Rational(-1, 4))**2
        elif m is S.One:
            return S.ComplexInfinity
        elif m is S.NegativeOne:
            return gamma(Rational(1, 4))**2/(4*sqrt(2*pi))
        elif m in (S.Infinity, S.NegativeInfinity, I*S.Infinity,
                   I*S.NegativeInfinity, S.ComplexInfinity):
            return S.Zero

    def fdiff(self, argindex=1):
        m = self.args[0]
        return (elliptic_e(m) - (1 - m)*elliptic_k(m))/(2*m*(1 - m))

    def _eval_conjugate(self):
        m = self.args[0]
        if (m.is_real and (m - 1).is_positive) is False:
            return self.func(m.conjugate())

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.simplify import hyperexpand
        return hyperexpand(self.rewrite(hyper)._eval_nseries(x, n=n, logx=logx))

    def _eval_rewrite_as_hyper(self, m, **kwargs):
        return pi*S.Half*hyper((S.Half, S.Half), (S.One,), m)

    def _eval_rewrite_as_meijerg(self, m, **kwargs):
        return meijerg(((S.Half, S.Half), []), ((S.Zero,), (S.Zero,)), -m)/2

    def _eval_is_zero(self):
        m = self.args[0]
        if m.is_infinite:
            return True

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', args).name)
        m = self.args[0]
        return Integral(1/sqrt(1 - m*sin(t)**2), (t, 0, pi/2))


class elliptic_f(DefinedFunction):
    r"""
    The Legendre incomplete elliptic integral of the first
    kind, defined by

    .. math:: F\left(z\middle| m\right) =
              \int_0^z \frac{dt}{\sqrt{1 - m \sin^2 t}}

    Explanation
    ===========

    This function reduces to a complete elliptic integral of
    the first kind, $K(m)$, when $z = \pi/2$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_f, I
    >>> from sympy.abc import z, m
    >>> elliptic_f(z, m).series(z)
    z + z**5*(3*m**2/40 - m/30) + m*z**3/6 + O(z**6)
    >>> elliptic_f(3.0 + I/2, 1.0 + I)
    2.909449841483 + 1.74720545502474*I

    See Also
    ========

    elliptic_k

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticF

    """

    @classmethod
    def eval(cls, z, m):
        if z.is_zero:
            return S.Zero
        if m.is_zero:
            return z
        k = 2*z/pi
        if k.is_integer:
            return k*elliptic_k(m)
        elif m in (S.Infinity, S.NegativeInfinity):
            return S.Zero
        elif z.could_extract_minus_sign():
            return -elliptic_f(-z, m)

    def fdiff(self, argindex=1):
        z, m = self.args
        fm = sqrt(1 - m*sin(z)**2)
        if argindex == 1:
            return 1/fm
        elif argindex == 2:
            return (elliptic_e(z, m)/(2*m*(1 - m)) - elliptic_f(z, m)/(2*m) -
                    sin(2*z)/(4*(1 - m)*fm))
        raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        z, m = self.args
        if (m.is_real and (m - 1).is_positive) is False:
            return self.func(z.conjugate(), m.conjugate())

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        t = Dummy(uniquely_named_symbol('t', args).name)
        z, m = self.args[0], self.args[1]
        return Integral(1/(sqrt(1 - m*sin(t)**2)), (t, 0, z))

    def _eval_is_zero(self):
        z, m = self.args
        if z.is_zero:
            return True
        if m.is_extended_real and m.is_infinite:
            return True


class elliptic_e(DefinedFunction):
    r"""
    Called with two arguments $z$ and $m$, evaluates the
    incomplete elliptic integral of the second kind, defined by

    .. math:: E\left(z\middle| m\right) = \int_0^z \sqrt{1 - m \sin^2 t} dt

    Called with a single argument $m$, evaluates the Legendre complete
    elliptic integral of the second kind

    .. math:: E(m) = E\left(\tfrac{\pi}{2}\middle| m\right)

    Explanation
    ===========

    The function $E(m)$ is a single-valued function on the complex
    plane with branch cut along the interval $(1, \infty)$.

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_e, I
    >>> from sympy.abc import z, m
    >>> elliptic_e(z, m).series(z)
    z + z**5*(-m**2/40 + m/30) - m*z**3/6 + O(z**6)
    >>> elliptic_e(m).series(n=4)
    pi/2 - pi*m/8 - 3*pi*m**2/128 - 5*pi*m**3/512 + O(m**4)
    >>> elliptic_e(1 + I, 2 - I/2).n()
    1.55203744279187 + 0.290764986058437*I
    >>> elliptic_e(0)
    pi/2
    >>> elliptic_e(2.0 - I)
    0.991052601328069 + 0.81879421395609*I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticE2
    .. [3] https://functions.wolfram.com/EllipticIntegrals/EllipticE

    """

    @classmethod
    def eval(cls, m, z=None):
        if z is not None:
            z, m = m, z
            k = 2*z/pi
            if m.is_zero:
                return z
            if z.is_zero:
                return S.Zero
            elif k.is_integer:
                return k*elliptic_e(m)
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.ComplexInfinity
            elif z.could_extract_minus_sign():
                return -elliptic_e(-z, m)
        else:
            if m.is_zero:
                return pi/2
            elif m is S.One:
                return S.One
            elif m is S.Infinity:
                return I*S.Infinity
            elif m is S.NegativeInfinity:
                return S.Infinity
            elif m is S.ComplexInfinity:
                return S.ComplexInfinity

    def fdiff(self, argindex=1):
        if len(self.args) == 2:
            z, m = self.args
            if argindex == 1:
                return sqrt(1 - m*sin(z)**2)
            elif argindex == 2:
                return (elliptic_e(z, m) - elliptic_f(z, m))/(2*m)
        else:
            m = self.args[0]
            if argindex == 1:
                return (elliptic_e(m) - elliptic_k(m))/(2*m)
        raise ArgumentIndexError(self, argindex)

    def _eval_conjugate(self):
        if len(self.args) == 2:
            z, m = self.args
            if (m.is_real and (m - 1).is_positive) is False:
                return self.func(z.conjugate(), m.conjugate())
        else:
            m = self.args[0]
            if (m.is_real and (m - 1).is_positive) is False:
                return self.func(m.conjugate())

    def _eval_nseries(self, x, n, logx, cdir=0):
        from sympy.simplify import hyperexpand
        if len(self.args) == 1:
            return hyperexpand(self.rewrite(hyper)._eval_nseries(x, n=n, logx=logx))
        return super()._eval_nseries(x, n=n, logx=logx)

    def _eval_rewrite_as_hyper(self, *args, **kwargs):
        if len(args) == 1:
            m = args[0]
            return (pi/2)*hyper((Rational(-1, 2), S.Half), (S.One,), m)

    def _eval_rewrite_as_meijerg(self, *args, **kwargs):
        if len(args) == 1:
            m = args[0]
            return -meijerg(((S.Half, Rational(3, 2)), []), \
                            ((S.Zero,), (S.Zero,)), -m)/4

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        z, m = (pi/2, self.args[0]) if len(self.args) == 1 else self.args
        t = Dummy(uniquely_named_symbol('t', args).name)
        return Integral(sqrt(1 - m*sin(t)**2), (t, 0, z))


class elliptic_pi(DefinedFunction):
    r"""
    Called with three arguments $n$, $z$ and $m$, evaluates the
    Legendre incomplete elliptic integral of the third kind, defined by

    .. math:: \Pi\left(n; z\middle| m\right) = \int_0^z \frac{dt}
              {\left(1 - n \sin^2 t\right) \sqrt{1 - m \sin^2 t}}

    Called with two arguments $n$ and $m$, evaluates the complete
    elliptic integral of the third kind:

    .. math:: \Pi\left(n\middle| m\right) =
              \Pi\left(n; \tfrac{\pi}{2}\middle| m\right)

    Explanation
    ===========

    Note that our notation defines the incomplete elliptic integral
    in terms of the parameter $m$ instead of the elliptic modulus
    (eccentricity) $k$.
    In this case, the parameter $m$ is defined as $m=k^2$.

    Examples
    ========

    >>> from sympy import elliptic_pi, I
    >>> from sympy.abc import z, n, m
    >>> elliptic_pi(n, z, m).series(z, n=4)
    z + z**3*(m/6 + n/3) + O(z**4)
    >>> elliptic_pi(0.5 + I, 1.0 - I, 1.2)
    2.50232379629182 - 0.760939574180767*I
    >>> elliptic_pi(0, 0)
    pi/2
    >>> elliptic_pi(1.0 - I/3, 2.0 + I)
    3.29136443417283 + 0.32555634906645*I

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Elliptic_integrals
    .. [2] https://functions.wolfram.com/EllipticIntegrals/EllipticPi3
    .. [3] https://functions.wolfram.com/EllipticIntegrals/EllipticPi

    """

    @classmethod
    def eval(cls, n, m, z=None):
        if z is not None:
            z, m = m, z
            if n.is_zero:
                return elliptic_f(z, m)
            elif n is S.One:
                return (elliptic_f(z, m) +
                        (sqrt(1 - m*sin(z)**2)*tan(z) -
                         elliptic_e(z, m))/(1 - m))
            k = 2*z/pi
            if k.is_integer:
                return k*elliptic_pi(n, m)
            elif m.is_zero:
                return atanh(sqrt(n - 1)*tan(z))/sqrt(n - 1)
            elif n == m:
                return (elliptic_f(z, n) - elliptic_pi(1, z, n) +
                        tan(z)/sqrt(1 - n*sin(z)**2))
            elif n in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif z.could_extract_minus_sign():
                return -elliptic_pi(n, -z, m)
            if n.is_zero:
                return elliptic_f(z, m)
            if m.is_extended_real and m.is_infinite or \
                    n.is_extended_real and n.is_infinite:
                return S.Zero
        else:
            if n.is_zero:
                return elliptic_k(m)
            elif n is S.One:
                return S.ComplexInfinity
            elif m.is_zero:
                return pi/(2*sqrt(1 - n))
            elif m == S.One:
                return S.NegativeInfinity/sign(n - 1)
            elif n == m:
                return elliptic_e(n)/(1 - n)
            elif n in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            elif m in (S.Infinity, S.NegativeInfinity):
                return S.Zero
            if n.is_zero:
                return elliptic_k(m)
            if m.is_extended_real and m.is_infinite or \
                    n.is_extended_real and n.is_infinite:
                return S.Zero

    def _eval_conjugate(self):
        if len(self.args) == 3:
            n, z, m = self.args
            if (n.is_real and (n - 1).is_positive) is False and \
               (m.is_real and (m - 1).is_positive) is False:
                return self.func(n.conjugate(), z.conjugate(), m.conjugate())
        else:
            n, m = self.args
            return self.func(n.conjugate(), m.conjugate())

    def fdiff(self, argindex=1):
        if len(self.args) == 3:
            n, z, m = self.args
            fm, fn = sqrt(1 - m*sin(z)**2), 1 - n*sin(z)**2
            if argindex == 1:
                return (elliptic_e(z, m) + (m - n)*elliptic_f(z, m)/n +
                        (n**2 - m)*elliptic_pi(n, z, m)/n -
                        n*fm*sin(2*z)/(2*fn))/(2*(m - n)*(n - 1))
            elif argindex == 2:
                return 1/(fm*fn)
            elif argindex == 3:
                return (elliptic_e(z, m)/(m - 1) +
                        elliptic_pi(n, z, m) -
                        m*sin(2*z)/(2*(m - 1)*fm))/(2*(n - m))
        else:
            n, m = self.args
            if argindex == 1:
                return (elliptic_e(m) + (m - n)*elliptic_k(m)/n +
                        (n**2 - m)*elliptic_pi(n, m)/n)/(2*(m - n)*(n - 1))
            elif argindex == 2:
                return (elliptic_e(m)/(m - 1) + elliptic_pi(n, m))/(2*(n - m))
        raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Integral(self, *args, **kwargs):
        from sympy.integrals.integrals import Integral
        if len(self.args) == 2:
            n, m, z = self.args[0], self.args[1], pi/2
        else:
            n, z, m = self.args
        t = Dummy(uniquely_named_symbol('t', args).name)
        return Integral(1/((1 - n*sin(t)**2)*sqrt(1 - m*sin(t)**2)), (t, 0, z))