
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\mapper.py ===
# orm/mapper.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Logic to map Python classes to and from selectables.

Defines the :class:`~sqlalchemy.orm.mapper.Mapper` class, the central
configurational unit which associates a class with a database table.

This is a semi-private module; the main configurational API of the ORM is
available in :class:`~sqlalchemy.orm.`.

"""
from __future__ import annotations

from collections import deque
from functools import reduce
from itertools import chain
import sys
import threading
from typing import Any
from typing import Callable
from typing import cast
from typing import Collection
from typing import Deque
from typing import Dict
from typing import FrozenSet
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import weakref

from . import attributes
from . import exc as orm_exc
from . import instrumentation
from . import loading
from . import properties
from . import util as orm_util
from ._typing import _O
from .base import _class_to_mapper
from .base import _parse_mapper_argument
from .base import _state_mapper
from .base import PassiveFlag
from .base import state_str
from .interfaces import _MappedAttribute
from .interfaces import EXT_SKIP
from .interfaces import InspectionAttr
from .interfaces import MapperProperty
from .interfaces import ORMEntityColumnsClauseRole
from .interfaces import ORMFromClauseRole
from .interfaces import StrategizedProperty
from .path_registry import PathRegistry
from .. import event
from .. import exc as sa_exc
from .. import inspection
from .. import log
from .. import schema
from .. import sql
from .. import util
from ..event import dispatcher
from ..event import EventTarget
from ..sql import base as sql_base
from ..sql import coercions
from ..sql import expression
from ..sql import operators
from ..sql import roles
from ..sql import TableClause
from ..sql import util as sql_util
from ..sql import visitors
from ..sql.cache_key import MemoizedHasCacheKey
from ..sql.elements import KeyedColumnElement
from ..sql.schema import Column
from ..sql.schema import Table
from ..sql.selectable import LABEL_STYLE_TABLENAME_PLUS_COL
from ..util import HasMemoized
from ..util import HasMemoized_ro_memoized_attribute
from ..util.typing import Literal

if TYPE_CHECKING:
    from ._typing import _IdentityKeyType
    from ._typing import _InstanceDict
    from ._typing import _ORMColumnExprArgument
    from ._typing import _RegistryType
    from .decl_api import registry
    from .dependency import DependencyProcessor
    from .descriptor_props import CompositeProperty
    from .descriptor_props import SynonymProperty
    from .events import MapperEvents
    from .instrumentation import ClassManager
    from .path_registry import CachingEntityRegistry
    from .properties import ColumnProperty
    from .relationships import RelationshipProperty
    from .state import InstanceState
    from .util import ORMAdapter
    from ..engine import Row
    from ..engine import RowMapping
    from ..sql._typing import _ColumnExpressionArgument
    from ..sql._typing import _EquivalentColumnMap
    from ..sql.base import ReadOnlyColumnCollection
    from ..sql.elements import ColumnClause
    from ..sql.elements import ColumnElement
    from ..sql.selectable import FromClause
    from ..util import OrderedSet


_T = TypeVar("_T", bound=Any)
_MP = TypeVar("_MP", bound="MapperProperty[Any]")
_Fn = TypeVar("_Fn", bound="Callable[..., Any]")


_WithPolymorphicArg = Union[
    Literal["*"],
    Tuple[
        Union[Literal["*"], Sequence[Union["Mapper[Any]", Type[Any]]]],
        Optional["FromClause"],
    ],
    Sequence[Union["Mapper[Any]", Type[Any]]],
]


_mapper_registries: weakref.WeakKeyDictionary[_RegistryType, bool] = (
    weakref.WeakKeyDictionary()
)


def _all_registries() -> Set[registry]:
    with _CONFIGURE_MUTEX:
        return set(_mapper_registries)


def _unconfigured_mappers() -> Iterator[Mapper[Any]]:
    for reg in _all_registries():
        yield from reg._mappers_to_configure()


_already_compiling = False


# a constant returned by _get_attr_by_column to indicate
# this mapper is not handling an attribute for a particular
# column
NO_ATTRIBUTE = util.symbol("NO_ATTRIBUTE")

# lock used to synchronize the "mapper configure" step
_CONFIGURE_MUTEX = threading.RLock()


@inspection._self_inspects
@log.class_logger
class Mapper(
    ORMFromClauseRole,
    ORMEntityColumnsClauseRole[_O],
    MemoizedHasCacheKey,
    InspectionAttr,
    log.Identified,
    inspection.Inspectable["Mapper[_O]"],
    EventTarget,
    Generic[_O],
):
    """Defines an association between a Python class and a database table or
    other relational structure, so that ORM operations against the class may
    proceed.

    The :class:`_orm.Mapper` object is instantiated using mapping methods
    present on the :class:`_orm.registry` object.  For information
    about instantiating new :class:`_orm.Mapper` objects, see
    :ref:`orm_mapping_classes_toplevel`.

    """

    dispatch: dispatcher[Mapper[_O]]

    _dispose_called = False
    _configure_failed: Any = False
    _ready_for_configure = False

    @util.deprecated_params(
        non_primary=(
            "1.3",
            "The :paramref:`.mapper.non_primary` parameter is deprecated, "
            "and will be removed in a future release.  The functionality "
            "of non primary mappers is now better suited using the "
            ":class:`.AliasedClass` construct, which can also be used "
            "as the target of a :func:`_orm.relationship` in 1.3.",
        ),
    )
    def __init__(
        self,
        class_: Type[_O],
        local_table: Optional[FromClause] = None,
        properties: Optional[Mapping[str, MapperProperty[Any]]] = None,
        primary_key: Optional[Iterable[_ORMColumnExprArgument[Any]]] = None,
        non_primary: bool = False,
        inherits: Optional[Union[Mapper[Any], Type[Any]]] = None,
        inherit_condition: Optional[_ColumnExpressionArgument[bool]] = None,
        inherit_foreign_keys: Optional[
            Sequence[_ORMColumnExprArgument[Any]]
        ] = None,
        always_refresh: bool = False,
        version_id_col: Optional[_ORMColumnExprArgument[Any]] = None,
        version_id_generator: Optional[
            Union[Literal[False], Callable[[Any], Any]]
        ] = None,
        polymorphic_on: Optional[
            Union[_ORMColumnExprArgument[Any], str, MapperProperty[Any]]
        ] = None,
        _polymorphic_map: Optional[Dict[Any, Mapper[Any]]] = None,
        polymorphic_identity: Optional[Any] = None,
        concrete: bool = False,
        with_polymorphic: Optional[_WithPolymorphicArg] = None,
        polymorphic_abstract: bool = False,
        polymorphic_load: Optional[Literal["selectin", "inline"]] = None,
        allow_partial_pks: bool = True,
        batch: bool = True,
        column_prefix: Optional[str] = None,
        include_properties: Optional[Sequence[str]] = None,
        exclude_properties: Optional[Sequence[str]] = None,
        passive_updates: bool = True,
        passive_deletes: bool = False,
        confirm_deleted_rows: bool = True,
        eager_defaults: Literal[True, False, "auto"] = "auto",
        legacy_is_orphan: bool = False,
        _compiled_cache_size: int = 100,
    ):
        r"""Direct constructor for a new :class:`_orm.Mapper` object.

        The :class:`_orm.Mapper` constructor is not called directly, and
        is normally invoked through the
        use of the :class:`_orm.registry` object through either the
        :ref:`Declarative <orm_declarative_mapping>` or
        :ref:`Imperative <orm_imperative_mapping>` mapping styles.

        .. versionchanged:: 2.0 The public facing ``mapper()`` function is
           removed; for a classical mapping configuration, use the
           :meth:`_orm.registry.map_imperatively` method.

        Parameters documented below may be passed to either the
        :meth:`_orm.registry.map_imperatively` method, or may be passed in the
        ``__mapper_args__`` declarative class attribute described at
        :ref:`orm_declarative_mapper_options`.

        :param class\_: The class to be mapped.  When using Declarative,
          this argument is automatically passed as the declared class
          itself.

        :param local_table: The :class:`_schema.Table` or other
           :class:`_sql.FromClause` (i.e. selectable) to which the class is
           mapped. May be ``None`` if this mapper inherits from another mapper
           using single-table inheritance. When using Declarative, this
           argument is automatically passed by the extension, based on what is
           configured via the :attr:`_orm.DeclarativeBase.__table__` attribute
           or via the :class:`_schema.Table` produced as a result of
           the :attr:`_orm.DeclarativeBase.__tablename__` attribute being
           present.

        :param polymorphic_abstract: Indicates this class will be mapped in a
            polymorphic hierarchy, but not directly instantiated. The class is
            mapped normally, except that it has no requirement for a
            :paramref:`_orm.Mapper.polymorphic_identity` within an inheritance
            hierarchy. The class however must be part of a polymorphic
            inheritance scheme which uses
            :paramref:`_orm.Mapper.polymorphic_on` at the base.

            .. versionadded:: 2.0

            .. seealso::

                :ref:`orm_inheritance_abstract_poly`

        :param always_refresh: If True, all query operations for this mapped
           class will overwrite all data within object instances that already
           exist within the session, erasing any in-memory changes with
           whatever information was loaded from the database. Usage of this
           flag is highly discouraged; as an alternative, see the method
           :meth:`_query.Query.populate_existing`.

        :param allow_partial_pks: Defaults to True.  Indicates that a
           composite primary key with some NULL values should be considered as
           possibly existing within the database. This affects whether a
           mapper will assign an incoming row to an existing identity, as well
           as if :meth:`.Session.merge` will check the database first for a
           particular primary key value. A "partial primary key" can occur if
           one has mapped to an OUTER JOIN, for example.

           The :paramref:`.orm.Mapper.allow_partial_pks` parameter also
           indicates to the ORM relationship lazy loader, when loading a
           many-to-one related object, if a composite primary key that has
           partial NULL values should result in an attempt to load from the
           database, or if a load attempt is not necessary.

           .. versionadded:: 2.0.36 :paramref:`.orm.Mapper.allow_partial_pks`
              is consulted by the relationship lazy loader strategy, such that
              when set to False, a SELECT for a composite primary key that
              has partial NULL values will not be emitted.

        :param batch: Defaults to ``True``, indicating that save operations
           of multiple entities can be batched together for efficiency.
           Setting to False indicates
           that an instance will be fully saved before saving the next
           instance.  This is used in the extremely rare case that a
           :class:`.MapperEvents` listener requires being called
           in between individual row persistence operations.

        :param column_prefix: A string which will be prepended
           to the mapped attribute name when :class:`_schema.Column`
           objects are automatically assigned as attributes to the
           mapped class.  Does not affect :class:`.Column` objects that
           are mapped explicitly in the :paramref:`.Mapper.properties`
           dictionary.

           This parameter is typically useful with imperative mappings
           that keep the :class:`.Table` object separate.  Below, assuming
           the ``user_table`` :class:`.Table` object has columns named
           ``user_id``, ``user_name``, and ``password``::

                class User(Base):
                    __table__ = user_table
                    __mapper_args__ = {"column_prefix": "_"}

           The above mapping will assign the ``user_id``, ``user_name``, and
           ``password`` columns to attributes named ``_user_id``,
           ``_user_name``, and ``_password`` on the mapped ``User`` class.

           The :paramref:`.Mapper.column_prefix` parameter is uncommon in
           modern use. For dealing with reflected tables, a more flexible
           approach to automating a naming scheme is to intercept the
           :class:`.Column` objects as they are reflected; see the section
           :ref:`mapper_automated_reflection_schemes` for notes on this usage
           pattern.

        :param concrete: If True, indicates this mapper should use concrete
           table inheritance with its parent mapper.

           See the section :ref:`concrete_inheritance` for an example.

        :param confirm_deleted_rows: defaults to True; when a DELETE occurs
          of one more rows based on specific primary keys, a warning is
          emitted when the number of rows matched does not equal the number
          of rows expected.  This parameter may be set to False to handle the
          case where database ON DELETE CASCADE rules may be deleting some of
          those rows automatically.  The warning may be changed to an
          exception in a future release.

        :param eager_defaults: if True, the ORM will immediately fetch the
          value of server-generated default values after an INSERT or UPDATE,
          rather than leaving them as expired to be fetched on next access.
          This can be used for event schemes where the server-generated values
          are needed immediately before the flush completes.

          The fetch of values occurs either by using ``RETURNING`` inline
          with the ``INSERT`` or ``UPDATE`` statement, or by adding an
          additional ``SELECT`` statement subsequent to the ``INSERT`` or
          ``UPDATE``, if the backend does not support ``RETURNING``.

          The use of ``RETURNING`` is extremely performant in particular for
          ``INSERT`` statements where SQLAlchemy can take advantage of
          :ref:`insertmanyvalues <engine_insertmanyvalues>`, whereas the use of
          an additional ``SELECT`` is relatively poor performing, adding
          additional SQL round trips which would be unnecessary if these new
          attributes are not to be accessed in any case.

          For this reason, :paramref:`.Mapper.eager_defaults` defaults to the
          string value ``"auto"``, which indicates that server defaults for
          INSERT should be fetched using ``RETURNING`` if the backing database
          supports it and if the dialect in use supports "insertmanyreturning"
          for an INSERT statement. If the backing database does not support
          ``RETURNING`` or "insertmanyreturning" is not available, server
          defaults will not be fetched.

          .. versionchanged:: 2.0.0rc1 added the "auto" option for
             :paramref:`.Mapper.eager_defaults`

          .. seealso::

                :ref:`orm_server_defaults`

          .. versionchanged:: 2.0.0  RETURNING now works with multiple rows
             INSERTed at once using the
             :ref:`insertmanyvalues <engine_insertmanyvalues>` feature, which
             among other things allows the :paramref:`.Mapper.eager_defaults`
             feature to be very performant on supporting backends.

        :param exclude_properties: A list or set of string column names to
          be excluded from mapping.

          .. seealso::

            :ref:`include_exclude_cols`

        :param include_properties: An inclusive list or set of string column
          names to map.

          .. seealso::

            :ref:`include_exclude_cols`

        :param inherits: A mapped class or the corresponding
          :class:`_orm.Mapper`
          of one indicating a superclass to which this :class:`_orm.Mapper`
          should *inherit* from.   The mapped class here must be a subclass
          of the other mapper's class.   When using Declarative, this argument
          is passed automatically as a result of the natural class
          hierarchy of the declared classes.

          .. seealso::

            :ref:`inheritance_toplevel`

        :param inherit_condition: For joined table inheritance, a SQL
           expression which will
           define how the two tables are joined; defaults to a natural join
           between the two tables.

        :param inherit_foreign_keys: When ``inherit_condition`` is used and
           the columns present are missing a :class:`_schema.ForeignKey`
           configuration, this parameter can be used to specify which columns
           are "foreign".  In most cases can be left as ``None``.

        :param legacy_is_orphan: Boolean, defaults to ``False``.
          When ``True``, specifies that "legacy" orphan consideration
          is to be applied to objects mapped by this mapper, which means
          that a pending (that is, not persistent) object is auto-expunged
          from an owning :class:`.Session` only when it is de-associated
          from *all* parents that specify a ``delete-orphan`` cascade towards
          this mapper.  The new default behavior is that the object is
          auto-expunged when it is de-associated with *any* of its parents
          that specify ``delete-orphan`` cascade.  This behavior is more
          consistent with that of a persistent object, and allows behavior to
          be consistent in more scenarios independently of whether or not an
          orphan object has been flushed yet or not.

          See the change note and example at :ref:`legacy_is_orphan_addition`
          for more detail on this change.

        :param non_primary: Specify that this :class:`_orm.Mapper`
          is in addition
          to the "primary" mapper, that is, the one used for persistence.
          The :class:`_orm.Mapper` created here may be used for ad-hoc
          mapping of the class to an alternate selectable, for loading
          only.

          .. seealso::

            :ref:`relationship_aliased_class` - the new pattern that removes
            the need for the :paramref:`_orm.Mapper.non_primary` flag.

        :param passive_deletes: Indicates DELETE behavior of foreign key
           columns when a joined-table inheritance entity is being deleted.
           Defaults to ``False`` for a base mapper; for an inheriting mapper,
           defaults to ``False`` unless the value is set to ``True``
           on the superclass mapper.

           When ``True``, it is assumed that ON DELETE CASCADE is configured
           on the foreign key relationships that link this mapper's table
           to its superclass table, so that when the unit of work attempts
           to delete the entity, it need only emit a DELETE statement for the
           superclass table, and not this table.

           When ``False``, a DELETE statement is emitted for this mapper's
           table individually.  If the primary key attributes local to this
           table are unloaded, then a SELECT must be emitted in order to
           validate these attributes; note that the primary key columns
           of a joined-table subclass are not part of the "primary key" of
           the object as a whole.

           Note that a value of ``True`` is **always** forced onto the
           subclass mappers; that is, it's not possible for a superclass
           to specify passive_deletes without this taking effect for
           all subclass mappers.

           .. seealso::

               :ref:`passive_deletes` - description of similar feature as
               used with :func:`_orm.relationship`

               :paramref:`.mapper.passive_updates` - supporting ON UPDATE
               CASCADE for joined-table inheritance mappers

        :param passive_updates: Indicates UPDATE behavior of foreign key
           columns when a primary key column changes on a joined-table
           inheritance mapping.   Defaults to ``True``.

           When True, it is assumed that ON UPDATE CASCADE is configured on
           the foreign key in the database, and that the database will handle
           propagation of an UPDATE from a source column to dependent columns
           on joined-table rows.

           When False, it is assumed that the database does not enforce
           referential integrity and will not be issuing its own CASCADE
           operation for an update.  The unit of work process will
           emit an UPDATE statement for the dependent columns during a
           primary key change.

           .. seealso::

               :ref:`passive_updates` - description of a similar feature as
               used with :func:`_orm.relationship`

               :paramref:`.mapper.passive_deletes` - supporting ON DELETE
               CASCADE for joined-table inheritance mappers

        :param polymorphic_load: Specifies "polymorphic loading" behavior
         for a subclass in an inheritance hierarchy (joined and single
         table inheritance only).   Valid values are:

          * "'inline'" - specifies this class should be part of
            the "with_polymorphic" mappers, e.g. its columns will be included
            in a SELECT query against the base.

          * "'selectin'" - specifies that when instances of this class
            are loaded, an additional SELECT will be emitted to retrieve
            the columns specific to this subclass.  The SELECT uses
            IN to fetch multiple subclasses at once.

         .. versionadded:: 1.2

         .. seealso::

            :ref:`with_polymorphic_mapper_config`

            :ref:`polymorphic_selectin`

        :param polymorphic_on: Specifies the column, attribute, or
          SQL expression used to determine the target class for an
          incoming row, when inheriting classes are present.

          May be specified as a string attribute name, or as a SQL
          expression such as a :class:`_schema.Column` or in a Declarative
          mapping a :func:`_orm.mapped_column` object.  It is typically
          expected that the SQL expression corresponds to a column in the
          base-most mapped :class:`.Table`::

            class Employee(Base):
                __tablename__ = "employee"

                id: Mapped[int] = mapped_column(primary_key=True)
                discriminator: Mapped[str] = mapped_column(String(50))

                __mapper_args__ = {
                    "polymorphic_on": discriminator,
                    "polymorphic_identity": "employee",
                }

          It may also be specified
          as a SQL expression, as in this example where we
          use the :func:`.case` construct to provide a conditional
          approach::

            class Employee(Base):
                __tablename__ = "employee"

                id: Mapped[int] = mapped_column(primary_key=True)
                discriminator: Mapped[str] = mapped_column(String(50))

                __mapper_args__ = {
                    "polymorphic_on": case(
                        (discriminator == "EN", "engineer"),
                        (discriminator == "MA", "manager"),
                        else_="employee",
                    ),
                    "polymorphic_identity": "employee",
                }

          It may also refer to any attribute using its string name,
          which is of particular use when using annotated column
          configurations::

                class Employee(Base):
                    __tablename__ = "employee"

                    id: Mapped[int] = mapped_column(primary_key=True)
                    discriminator: Mapped[str]

                    __mapper_args__ = {
                        "polymorphic_on": "discriminator",
                        "polymorphic_identity": "employee",
                    }

          When setting ``polymorphic_on`` to reference an
          attribute or expression that's not present in the
          locally mapped :class:`_schema.Table`, yet the value
          of the discriminator should be persisted to the database,
          the value of the
          discriminator is not automatically set on new
          instances; this must be handled by the user,
          either through manual means or via event listeners.
          A typical approach to establishing such a listener
          looks like::

                from sqlalchemy import event
                from sqlalchemy.orm import object_mapper


                @event.listens_for(Employee, "init", propagate=True)
                def set_identity(instance, *arg, **kw):
                    mapper = object_mapper(instance)
                    instance.discriminator = mapper.polymorphic_identity

          Where above, we assign the value of ``polymorphic_identity``
          for the mapped class to the ``discriminator`` attribute,
          thus persisting the value to the ``discriminator`` column
          in the database.

          .. warning::

             Currently, **only one discriminator column may be set**, typically
             on the base-most class in the hierarchy. "Cascading" polymorphic
             columns are not yet supported.

          .. seealso::

            :ref:`inheritance_toplevel`

        :param polymorphic_identity: Specifies the value which
          identifies this particular class as returned by the column expression
          referred to by the :paramref:`_orm.Mapper.polymorphic_on` setting. As
          rows are received, the value corresponding to the
          :paramref:`_orm.Mapper.polymorphic_on` column expression is compared
          to this value, indicating which subclass should be used for the newly
          reconstructed object.

          .. seealso::

            :ref:`inheritance_toplevel`

        :param properties: A dictionary mapping the string names of object
           attributes to :class:`.MapperProperty` instances, which define the
           persistence behavior of that attribute.  Note that
           :class:`_schema.Column`
           objects present in
           the mapped :class:`_schema.Table` are automatically placed into
           ``ColumnProperty`` instances upon mapping, unless overridden.
           When using Declarative, this argument is passed automatically,
           based on all those :class:`.MapperProperty` instances declared
           in the declared class body.

           .. seealso::

               :ref:`orm_mapping_properties` - in the
               :ref:`orm_mapping_classes_toplevel`

        :param primary_key: A list of :class:`_schema.Column`
           objects, or alternatively string names of attribute names which
           refer to :class:`_schema.Column`, which define
           the primary key to be used against this mapper's selectable unit.
           This is normally simply the primary key of the ``local_table``, but
           can be overridden here.

           .. versionchanged:: 2.0.2 :paramref:`_orm.Mapper.primary_key`
              arguments may be indicated as string attribute names as well.

           .. seealso::

                :ref:`mapper_primary_key` - background and example use

        :param version_id_col: A :class:`_schema.Column`
           that will be used to keep a running version id of rows
           in the table.  This is used to detect concurrent updates or
           the presence of stale data in a flush.  The methodology is to
           detect if an UPDATE statement does not match the last known
           version id, a
           :class:`~sqlalchemy.orm.exc.StaleDataError` exception is
           thrown.
           By default, the column must be of :class:`.Integer` type,
           unless ``version_id_generator`` specifies an alternative version
           generator.

           .. seealso::

              :ref:`mapper_version_counter` - discussion of version counting
              and rationale.

        :param version_id_generator: Define how new version ids should
          be generated.  Defaults to ``None``, which indicates that
          a simple integer counting scheme be employed.  To provide a custom
          versioning scheme, provide a callable function of the form::

              def generate_version(version):
                  return next_version

          Alternatively, server-side versioning functions such as triggers,
          or programmatic versioning schemes outside of the version id
          generator may be used, by specifying the value ``False``.
          Please see :ref:`server_side_version_counter` for a discussion
          of important points when using this option.

          .. seealso::

             :ref:`custom_version_counter`

             :ref:`server_side_version_counter`


        :param with_polymorphic: A tuple in the form ``(<classes>,
            <selectable>)`` indicating the default style of "polymorphic"
            loading, that is, which tables are queried at once. <classes> is
            any single or list of mappers and/or classes indicating the
            inherited classes that should be loaded at once. The special value
            ``'*'`` may be used to indicate all descending classes should be
            loaded immediately. The second tuple argument <selectable>
            indicates a selectable that will be used to query for multiple
            classes.

            The :paramref:`_orm.Mapper.polymorphic_load` parameter may be
            preferable over the use of :paramref:`_orm.Mapper.with_polymorphic`
            in modern mappings to indicate a per-subclass technique of
            indicating polymorphic loading styles.

            .. seealso::

                :ref:`with_polymorphic_mapper_config`

        """
        self.class_ = util.assert_arg_type(class_, type, "class_")
        self._sort_key = "%s.%s" % (
            self.class_.__module__,
            self.class_.__name__,
        )

        self._primary_key_argument = util.to_list(primary_key)
        self.non_primary = non_primary

        self.always_refresh = always_refresh

        if isinstance(version_id_col, MapperProperty):
            self.version_id_prop = version_id_col
            self.version_id_col = None
        else:
            self.version_id_col = (
                coercions.expect(
                    roles.ColumnArgumentOrKeyRole,
                    version_id_col,
                    argname="version_id_col",
                )
                if version_id_col is not None
                else None
            )

        if version_id_generator is False:
            self.version_id_generator = False
        elif version_id_generator is None:
            self.version_id_generator = lambda x: (x or 0) + 1
        else:
            self.version_id_generator = version_id_generator

        self.concrete = concrete
        self.single = False

        if inherits is not None:
            self.inherits = _parse_mapper_argument(inherits)
        else:
            self.inherits = None

        if local_table is not None:
            self.local_table = coercions.expect(
                roles.StrictFromClauseRole,
                local_table,
                disable_inspection=True,
                argname="local_table",
            )
        elif self.inherits:
            # note this is a new flow as of 2.0 so that
            # .local_table need not be Optional
            self.local_table = self.inherits.local_table
            self.single = True
        else:
            raise sa_exc.ArgumentError(
                f"Mapper[{self.class_.__name__}(None)] has None for a "
                "primary table argument and does not specify 'inherits'"
            )

        if inherit_condition is not None:
            self.inherit_condition = coercions.expect(
                roles.OnClauseRole, inherit_condition
            )
        else:
            self.inherit_condition = None

        self.inherit_foreign_keys = inherit_foreign_keys
        self._init_properties = dict(properties) if properties else {}
        self._delete_orphans = []
        self.batch = batch
        self.eager_defaults = eager_defaults
        self.column_prefix = column_prefix

        # interim - polymorphic_on is further refined in
        # _configure_polymorphic_setter
        self.polymorphic_on = (
            coercions.expect(  # type: ignore
                roles.ColumnArgumentOrKeyRole,
                polymorphic_on,
                argname="polymorphic_on",
            )
            if polymorphic_on is not None
            else None
        )
        self.polymorphic_abstract = polymorphic_abstract
        self._dependency_processors = []
        self.validators = util.EMPTY_DICT
        self.passive_updates = passive_updates
        self.passive_deletes = passive_deletes
        self.legacy_is_orphan = legacy_is_orphan
        self._clause_adapter = None
        self._requires_row_aliasing = False
        self._inherits_equated_pairs = None
        self._memoized_values = {}
        self._compiled_cache_size = _compiled_cache_size
        self._reconstructor = None
        self.allow_partial_pks = allow_partial_pks

        if self.inherits and not self.concrete:
            self.confirm_deleted_rows = False
        else:
            self.confirm_deleted_rows = confirm_deleted_rows

        self._set_with_polymorphic(with_polymorphic)
        self.polymorphic_load = polymorphic_load

        # our 'polymorphic identity', a string name that when located in a
        #  result set row indicates this Mapper should be used to construct
        # the object instance for that row.
        self.polymorphic_identity = polymorphic_identity

        # a dictionary of 'polymorphic identity' names, associating those
        # names with Mappers that will be used to construct object instances
        # upon a select operation.
        if _polymorphic_map is None:
            self.polymorphic_map = {}
        else:
            self.polymorphic_map = _polymorphic_map

        if include_properties is not None:
            self.include_properties = util.to_set(include_properties)
        else:
            self.include_properties = None
        if exclude_properties:
            self.exclude_properties = util.to_set(exclude_properties)
        else:
            self.exclude_properties = None

        # prevent this mapper from being constructed
        # while a configure_mappers() is occurring (and defer a
        # configure_mappers() until construction succeeds)
        with _CONFIGURE_MUTEX:
            cast("MapperEvents", self.dispatch._events)._new_mapper_instance(
                class_, self
            )
            self._configure_inheritance()
            self._configure_class_instrumentation()
            self._configure_properties()
            self._configure_polymorphic_setter()
            self._configure_pks()
            self.registry._flag_new_mapper(self)
            self._log("constructed")
            self._expire_memoizations()

        self.dispatch.after_mapper_constructed(self, self.class_)

    def _prefer_eager_defaults(self, dialect, table):
        if self.eager_defaults == "auto":
            if not table.implicit_returning:
                return False

            return (
                table in self._server_default_col_keys
                and dialect.insert_executemany_returning
            )
        else:
            return self.eager_defaults

    def _gen_cache_key(self, anon_map, bindparams):
        return (self,)

    # ### BEGIN
    # ATTRIBUTE DECLARATIONS START HERE

    is_mapper = True
    """Part of the inspection API."""

    represents_outer_join = False

    registry: _RegistryType

    @property
    def mapper(self) -> Mapper[_O]:
        """Part of the inspection API.

        Returns self.

        """
        return self

    @property
    def entity(self):
        r"""Part of the inspection API.

        Returns self.class\_.

        """
        return self.class_

    class_: Type[_O]
    """The class to which this :class:`_orm.Mapper` is mapped."""

    _identity_class: Type[_O]

    _delete_orphans: List[Tuple[str, Type[Any]]]
    _dependency_processors: List[DependencyProcessor]
    _memoized_values: Dict[Any, Callable[[], Any]]
    _inheriting_mappers: util.WeakSequence[Mapper[Any]]
    _all_tables: Set[TableClause]
    _polymorphic_attr_key: Optional[str]

    _pks_by_table: Dict[FromClause, OrderedSet[ColumnClause[Any]]]
    _cols_by_table: Dict[FromClause, OrderedSet[ColumnElement[Any]]]

    _props: util.OrderedDict[str, MapperProperty[Any]]
    _init_properties: Dict[str, MapperProperty[Any]]

    _columntoproperty: _ColumnMapping

    _set_polymorphic_identity: Optional[Callable[[InstanceState[_O]], None]]
    _validate_polymorphic_identity: Optional[
        Callable[[Mapper[_O], InstanceState[_O], _InstanceDict], None]
    ]

    tables: Sequence[TableClause]
    """A sequence containing the collection of :class:`_schema.Table`
    or :class:`_schema.TableClause` objects which this :class:`_orm.Mapper`
    is aware of.

    If the mapper is mapped to a :class:`_expression.Join`, or an
    :class:`_expression.Alias`
    representing a :class:`_expression.Select`, the individual
    :class:`_schema.Table`
    objects that comprise the full construct will be represented here.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    validators: util.immutabledict[str, Tuple[str, Dict[str, Any]]]
    """An immutable dictionary of attributes which have been decorated
    using the :func:`_orm.validates` decorator.

    The dictionary contains string attribute names as keys
    mapped to the actual validation method.

    """

    always_refresh: bool
    allow_partial_pks: bool
    version_id_col: Optional[ColumnElement[Any]]

    with_polymorphic: Optional[
        Tuple[
            Union[Literal["*"], Sequence[Union[Mapper[Any], Type[Any]]]],
            Optional[FromClause],
        ]
    ]

    version_id_generator: Optional[Union[Literal[False], Callable[[Any], Any]]]

    local_table: FromClause
    """The immediate :class:`_expression.FromClause` to which this
    :class:`_orm.Mapper` refers.

    Typically is an instance of :class:`_schema.Table`, may be any
    :class:`.FromClause`.

    The "local" table is the
    selectable that the :class:`_orm.Mapper` is directly responsible for
    managing from an attribute access and flush perspective.   For
    non-inheriting mappers, :attr:`.Mapper.local_table` will be the same
    as :attr:`.Mapper.persist_selectable`.  For inheriting mappers,
    :attr:`.Mapper.local_table` refers to the specific portion of
    :attr:`.Mapper.persist_selectable` that includes the columns to which
    this :class:`.Mapper` is loading/persisting, such as a particular
    :class:`.Table` within a join.

    .. seealso::

        :attr:`_orm.Mapper.persist_selectable`.

        :attr:`_orm.Mapper.selectable`.

    """

    persist_selectable: FromClause
    """The :class:`_expression.FromClause` to which this :class:`_orm.Mapper`
    is mapped.

    Typically is an instance of :class:`_schema.Table`, may be any
    :class:`.FromClause`.

    The :attr:`_orm.Mapper.persist_selectable` is similar to
    :attr:`.Mapper.local_table`, but represents the :class:`.FromClause` that
    represents the inheriting class hierarchy overall in an inheritance
    scenario.

    :attr.`.Mapper.persist_selectable` is also separate from the
    :attr:`.Mapper.selectable` attribute, the latter of which may be an
    alternate subquery used for selecting columns.
    :attr.`.Mapper.persist_selectable` is oriented towards columns that
    will be written on a persist operation.

    .. seealso::

        :attr:`_orm.Mapper.selectable`.

        :attr:`_orm.Mapper.local_table`.

    """

    inherits: Optional[Mapper[Any]]
    """References the :class:`_orm.Mapper` which this :class:`_orm.Mapper`
    inherits from, if any.

    """

    inherit_condition: Optional[ColumnElement[bool]]

    configured: bool = False
    """Represent ``True`` if this :class:`_orm.Mapper` has been configured.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    .. seealso::

        :func:`.configure_mappers`.

    """

    concrete: bool
    """Represent ``True`` if this :class:`_orm.Mapper` is a concrete
    inheritance mapper.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    primary_key: Tuple[Column[Any], ...]
    """An iterable containing the collection of :class:`_schema.Column`
    objects
    which comprise the 'primary key' of the mapped table, from the
    perspective of this :class:`_orm.Mapper`.

    This list is against the selectable in
    :attr:`_orm.Mapper.persist_selectable`.
    In the case of inheriting mappers, some columns may be managed by a
    superclass mapper.  For example, in the case of a
    :class:`_expression.Join`, the
    primary key is determined by all of the primary key columns across all
    tables referenced by the :class:`_expression.Join`.

    The list is also not necessarily the same as the primary key column
    collection associated with the underlying tables; the :class:`_orm.Mapper`
    features a ``primary_key`` argument that can override what the
    :class:`_orm.Mapper` considers as primary key columns.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    class_manager: ClassManager[_O]
    """The :class:`.ClassManager` which maintains event listeners
    and class-bound descriptors for this :class:`_orm.Mapper`.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    single: bool
    """Represent ``True`` if this :class:`_orm.Mapper` is a single table
    inheritance mapper.

    :attr:`_orm.Mapper.local_table` will be ``None`` if this flag is set.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    non_primary: bool
    """Represent ``True`` if this :class:`_orm.Mapper` is a "non-primary"
    mapper, e.g. a mapper that is used only to select rows but not for
    persistence management.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    polymorphic_on: Optional[KeyedColumnElement[Any]]
    """The :class:`_schema.Column` or SQL expression specified as the
    ``polymorphic_on`` argument
    for this :class:`_orm.Mapper`, within an inheritance scenario.

    This attribute is normally a :class:`_schema.Column` instance but
    may also be an expression, such as one derived from
    :func:`.cast`.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    polymorphic_map: Dict[Any, Mapper[Any]]
    """A mapping of "polymorphic identity" identifiers mapped to
    :class:`_orm.Mapper` instances, within an inheritance scenario.

    The identifiers can be of any type which is comparable to the
    type of column represented by :attr:`_orm.Mapper.polymorphic_on`.

    An inheritance chain of mappers will all reference the same
    polymorphic map object.  The object is used to correlate incoming
    result rows to target mappers.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    polymorphic_identity: Optional[Any]
    """Represent an identifier which is matched against the
    :attr:`_orm.Mapper.polymorphic_on` column during result row loading.

    Used only with inheritance, this object can be of any type which is
    comparable to the type of column represented by
    :attr:`_orm.Mapper.polymorphic_on`.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    base_mapper: Mapper[Any]
    """The base-most :class:`_orm.Mapper` in an inheritance chain.

    In a non-inheriting scenario, this attribute will always be this
    :class:`_orm.Mapper`.   In an inheritance scenario, it references
    the :class:`_orm.Mapper` which is parent to all other :class:`_orm.Mapper`
    objects in the inheritance chain.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    columns: ReadOnlyColumnCollection[str, Column[Any]]
    """A collection of :class:`_schema.Column` or other scalar expression
    objects maintained by this :class:`_orm.Mapper`.

    The collection behaves the same as that of the ``c`` attribute on
    any :class:`_schema.Table` object,
    except that only those columns included in
    this mapping are present, and are keyed based on the attribute name
    defined in the mapping, not necessarily the ``key`` attribute of the
    :class:`_schema.Column` itself.   Additionally, scalar expressions mapped
    by :func:`.column_property` are also present here.

    This is a *read only* attribute determined during mapper construction.
    Behavior is undefined if directly modified.

    """

    c: ReadOnlyColumnCollection[str, Column[Any]]
    """A synonym for :attr:`_orm.Mapper.columns`."""

    @util.non_memoized_property
    @util.deprecated("1.3", "Use .persist_selectable")
    def mapped_table(self):
        return self.persist_selectable

    @util.memoized_property
    def _path_registry(self) -> CachingEntityRegistry:
        return PathRegistry.per_mapper(self)

    def _configure_inheritance(self):
        """Configure settings related to inheriting and/or inherited mappers
        being present."""

        # a set of all mappers which inherit from this one.
        self._inheriting_mappers = util.WeakSequence()

        if self.inherits:
            if not issubclass(self.class_, self.inherits.class_):
                raise sa_exc.ArgumentError(
                    "Class '%s' does not inherit from '%s'"
                    % (self.class_.__name__, self.inherits.class_.__name__)
                )

            self.dispatch._update(self.inherits.dispatch)

            if self.non_primary != self.inherits.non_primary:
                np = not self.non_primary and "primary" or "non-primary"
                raise sa_exc.ArgumentError(
                    "Inheritance of %s mapper for class '%s' is "
                    "only allowed from a %s mapper"
                    % (np, self.class_.__name__, np)
                )

            if self.single:
                self.persist_selectable = self.inherits.persist_selectable
            elif self.local_table is not self.inherits.local_table:
                if self.concrete:
                    self.persist_selectable = self.local_table
                    for mapper in self.iterate_to_root():
                        if mapper.polymorphic_on is not None:
                            mapper._requires_row_aliasing = True
                else:
                    if self.inherit_condition is None:
                        # figure out inherit condition from our table to the
                        # immediate table of the inherited mapper, not its
                        # full table which could pull in other stuff we don't
                        # want (allows test/inheritance.InheritTest4 to pass)
                        try:
                            self.inherit_condition = sql_util.join_condition(
                                self.inherits.local_table, self.local_table
                            )
                        except sa_exc.NoForeignKeysError as nfe:
                            assert self.inherits.local_table is not None
                            assert self.local_table is not None
                            raise sa_exc.NoForeignKeysError(
                                "Can't determine the inherit condition "
                                "between inherited table '%s' and "
                                "inheriting "
                                "table '%s'; tables have no "
                                "foreign key relationships established.  "
                                "Please ensure the inheriting table has "
                                "a foreign key relationship to the "
                                "inherited "
                                "table, or provide an "
                                "'on clause' using "
                                "the 'inherit_condition' mapper argument."
                                % (
                                    self.inherits.local_table.description,
                                    self.local_table.description,
                                )
                            ) from nfe
                        except sa_exc.AmbiguousForeignKeysError as afe:
                            assert self.inherits.local_table is not None
                            assert self.local_table is not None
                            raise sa_exc.AmbiguousForeignKeysError(
                                "Can't determine the inherit condition "
                                "between inherited table '%s' and "
                                "inheriting "
                                "table '%s'; tables have more than one "
                                "foreign key relationship established.  "
                                "Please specify the 'on clause' using "
                                "the 'inherit_condition' mapper argument."
                                % (
                                    self.inherits.local_table.description,
                                    self.local_table.description,
                                )
                            ) from afe
                    assert self.inherits.persist_selectable is not None
                    self.persist_selectable = sql.join(
                        self.inherits.persist_selectable,
                        self.local_table,
                        self.inherit_condition,
                    )

                    fks = util.to_set(self.inherit_foreign_keys)
                    self._inherits_equated_pairs = sql_util.criterion_as_pairs(
                        self.persist_selectable.onclause,
                        consider_as_foreign_keys=fks,
                    )
            else:
                self.persist_selectable = self.local_table

            if self.polymorphic_identity is None:
                self._identity_class = self.class_

                if (
                    not self.polymorphic_abstract
                    and self.inherits.base_mapper.polymorphic_on is not None
                ):
                    util.warn(
                        f"{self} does not indicate a 'polymorphic_identity', "
                        "yet is part of an inheritance hierarchy that has a "
                        f"'polymorphic_on' column of "
                        f"'{self.inherits.base_mapper.polymorphic_on}'. "
                        "If this is an intermediary class that should not be "
                        "instantiated, the class may either be left unmapped, "
                        "or may include the 'polymorphic_abstract=True' "
                        "parameter in its Mapper arguments. To leave the "
                        "class unmapped when using Declarative, set the "
                        "'__abstract__ = True' attribute on the class."
                    )
            elif self.concrete:
                self._identity_class = self.class_
            else:
                self._identity_class = self.inherits._identity_class

            if self.version_id_col is None:
                self.version_id_col = self.inherits.version_id_col
                self.version_id_generator = self.inherits.version_id_generator
            elif (
                self.inherits.version_id_col is not None
                and self.version_id_col is not self.inherits.version_id_col
            ):
                util.warn(
                    "Inheriting version_id_col '%s' does not match inherited "
                    "version_id_col '%s' and will not automatically populate "
                    "the inherited versioning column. "
                    "version_id_col should only be specified on "
                    "the base-most mapper that includes versioning."
                    % (
                        self.version_id_col.description,
                        self.inherits.version_id_col.description,
                    )
                )

            self.polymorphic_map = self.inherits.polymorphic_map
            self.batch = self.inherits.batch
            self.inherits._inheriting_mappers.append(self)
            self.base_mapper = self.inherits.base_mapper
            self.passive_updates = self.inherits.passive_updates
            self.passive_deletes = (
                self.inherits.passive_deletes or self.passive_deletes
            )
            self._all_tables = self.inherits._all_tables

            if self.polymorphic_identity is not None:
                if self.polymorphic_identity in self.polymorphic_map:
                    util.warn(
                        "Reassigning polymorphic association for identity %r "
                        "from %r to %r: Check for duplicate use of %r as "
                        "value for polymorphic_identity."
                        % (
                            self.polymorphic_identity,
                            self.polymorphic_map[self.polymorphic_identity],
                            self,
                            self.polymorphic_identity,
                        )
                    )
                self.polymorphic_map[self.polymorphic_identity] = self

            if self.polymorphic_load and self.concrete:
                raise sa_exc.ArgumentError(
                    "polymorphic_load is not currently supported "
                    "with concrete table inheritance"
                )
            if self.polymorphic_load == "inline":
                self.inherits._add_with_polymorphic_subclass(self)
            elif self.polymorphic_load == "selectin":
                pass
            elif self.polymorphic_load is not None:
                raise sa_exc.ArgumentError(
                    "unknown argument for polymorphic_load: %r"
                    % self.polymorphic_load
                )

        else:
            self._all_tables = set()
            self.base_mapper = self
            assert self.local_table is not None
            self.persist_selectable = self.local_table
            if self.polymorphic_identity is not None:
                self.polymorphic_map[self.polymorphic_identity] = self
            self._identity_class = self.class_

        if self.persist_selectable is None:
            raise sa_exc.ArgumentError(
                "Mapper '%s' does not have a persist_selectable specified."
                % self
            )

    def _set_with_polymorphic(
        self, with_polymorphic: Optional[_WithPolymorphicArg]
    ) -> None:
        if with_polymorphic == "*":
            self.with_polymorphic = ("*", None)
        elif isinstance(with_polymorphic, (tuple, list)):
            if isinstance(with_polymorphic[0], (str, tuple, list)):
                self.with_polymorphic = cast(
                    """Tuple[
                        Union[
                            Literal["*"],
                            Sequence[Union["Mapper[Any]", Type[Any]]],
                        ],
                        Optional["FromClause"],
                    ]""",
                    with_polymorphic,
                )
            else:
                self.with_polymorphic = (with_polymorphic, None)
        elif with_polymorphic is not None:
            raise sa_exc.ArgumentError(
                f"Invalid setting for with_polymorphic: {with_polymorphic!r}"
            )
        else:
            self.with_polymorphic = None

        if self.with_polymorphic and self.with_polymorphic[1] is not None:
            self.with_polymorphic = (
                self.with_polymorphic[0],
                coercions.expect(
                    roles.StrictFromClauseRole,
                    self.with_polymorphic[1],
                    allow_select=True,
                ),
            )

        if self.configured:
            self._expire_memoizations()

    def _add_with_polymorphic_subclass(self, mapper):
        subcl = mapper.class_
        if self.with_polymorphic is None:
            self._set_with_polymorphic((subcl,))
        elif self.with_polymorphic[0] != "*":
            assert isinstance(self.with_polymorphic[0], tuple)
            self._set_with_polymorphic(
                (self.with_polymorphic[0] + (subcl,), self.with_polymorphic[1])
            )

    def _set_concrete_base(self, mapper):
        """Set the given :class:`_orm.Mapper` as the 'inherits' for this
        :class:`_orm.Mapper`, assuming this :class:`_orm.Mapper` is concrete
        and does not already have an inherits."""

        assert self.concrete
        assert not self.inherits
        assert isinstance(mapper, Mapper)
        self.inherits = mapper
        self.inherits.polymorphic_map.update(self.polymorphic_map)
        self.polymorphic_map = self.inherits.polymorphic_map
        for mapper in self.iterate_to_root():
            if mapper.polymorphic_on is not None:
                mapper._requires_row_aliasing = True
        self.batch = self.inherits.batch
        for mp in self.self_and_descendants:
            mp.base_mapper = self.inherits.base_mapper
        self.inherits._inheriting_mappers.append(self)
        self.passive_updates = self.inherits.passive_updates
        self._all_tables = self.inherits._all_tables

        for key, prop in mapper._props.items():
            if key not in self._props and not self._should_exclude(
                key, key, local=False, column=None
            ):
                self._adapt_inherited_property(key, prop, False)

    def _set_polymorphic_on(self, polymorphic_on):
        self.polymorphic_on = polymorphic_on
        self._configure_polymorphic_setter(True)

    def _configure_class_instrumentation(self):
        """If this mapper is to be a primary mapper (i.e. the
        non_primary flag is not set), associate this Mapper with the
        given class and entity name.

        Subsequent calls to ``class_mapper()`` for the ``class_`` / ``entity``
        name combination will return this mapper.  Also decorate the
        `__init__` method on the mapped class to include optional
        auto-session attachment logic.

        """

        # we expect that declarative has applied the class manager
        # already and set up a registry.  if this is None,
        # this raises as of 2.0.
        manager = attributes.opt_manager_of_class(self.class_)

        if self.non_primary:
            if not manager or not manager.is_mapped:
                raise sa_exc.InvalidRequestError(
                    "Class %s has no primary mapper configured.  Configure "
                    "a primary mapper first before setting up a non primary "
                    "Mapper." % self.class_
                )
            self.class_manager = manager

            assert manager.registry is not None
            self.registry = manager.registry
            self._identity_class = manager.mapper._identity_class
            manager.registry._add_non_primary_mapper(self)
            return

        if manager is None or not manager.registry:
            raise sa_exc.InvalidRequestError(
                "The _mapper() function and Mapper() constructor may not be "
                "invoked directly outside of a declarative registry."
                " Please use the sqlalchemy.orm.registry.map_imperatively() "
                "function for a classical mapping."
            )

        self.dispatch.instrument_class(self, self.class_)

        # this invokes the class_instrument event and sets up
        # the __init__ method.  documented behavior is that this must
        # occur after the instrument_class event above.
        # yes two events with the same two words reversed and different APIs.
        # :(

        manager = instrumentation.register_class(
            self.class_,
            mapper=self,
            expired_attribute_loader=util.partial(
                loading.load_scalar_attributes, self
            ),
            # finalize flag means instrument the __init__ method
            # and call the class_instrument event
            finalize=True,
        )

        self.class_manager = manager

        assert manager.registry is not None
        self.registry = manager.registry

        # The remaining members can be added by any mapper,
        # e_name None or not.
        if manager.mapper is None:
            return

        event.listen(manager, "init", _event_on_init, raw=True)

        for key, method in util.iterate_attributes(self.class_):
            if key == "__init__" and hasattr(method, "_sa_original_init"):
                method = method._sa_original_init
                if hasattr(method, "__func__"):
                    method = method.__func__
            if callable(method):
                if hasattr(method, "__sa_reconstructor__"):
                    self._reconstructor = method
                    event.listen(manager, "load", _event_on_load, raw=True)
                elif hasattr(method, "__sa_validators__"):
                    validation_opts = method.__sa_validation_opts__
                    for name in method.__sa_validators__:
                        if name in self.validators:
                            raise sa_exc.InvalidRequestError(
                                "A validation function for mapped "
                                "attribute %r on mapper %s already exists."
                                % (name, self)
                            )
                        self.validators = self.validators.union(
                            {name: (method, validation_opts)}
                        )

    def _set_dispose_flags(self) -> None:
        self.configured = True
        self._ready_for_configure = True
        self._dispose_called = True

        self.__dict__.pop("_configure_failed", None)

    def _str_arg_to_mapped_col(self, argname: str, key: str) -> Column[Any]:
        try:
            prop = self._props[key]
        except KeyError as err:
            raise sa_exc.ArgumentError(
                f"Can't determine {argname} column '{key}' - "
                "no attribute is mapped to this name."
            ) from err
        try:
            expr = prop.expression
        except AttributeError as ae:
            raise sa_exc.ArgumentError(
                f"Can't determine {argname} column '{key}'; "
                "property does not refer to a single mapped Column"
            ) from ae
        if not isinstance(expr, Column):
            raise sa_exc.ArgumentError(
                f"Can't determine {argname} column '{key}'; "
                "property does not refer to a single "
                "mapped Column"
            )
        return expr

    def _configure_pks(self) -> None:
        self.tables = sql_util.find_tables(self.persist_selectable)

        self._all_tables.update(t for t in self.tables)

        self._pks_by_table = {}
        self._cols_by_table = {}

        all_cols = util.column_set(
            chain(*[col.proxy_set for col in self._columntoproperty])
        )

        pk_cols = util.column_set(c for c in all_cols if c.primary_key)

        # identify primary key columns which are also mapped by this mapper.
        for fc in set(self.tables).union([self.persist_selectable]):
            if fc.primary_key and pk_cols.issuperset(fc.primary_key):
                # ordering is important since it determines the ordering of
                # mapper.primary_key (and therefore query.get())
                self._pks_by_table[fc] = util.ordered_column_set(  # type: ignore  # noqa: E501
                    fc.primary_key
                ).intersection(
                    pk_cols
                )
            self._cols_by_table[fc] = util.ordered_column_set(fc.c).intersection(  # type: ignore  # noqa: E501
                all_cols
            )

        if self._primary_key_argument:
            coerced_pk_arg = [
                (
                    self._str_arg_to_mapped_col("primary_key", c)
                    if isinstance(c, str)
                    else c
                )
                for c in (
                    coercions.expect(
                        roles.DDLConstraintColumnRole,
                        coerce_pk,
                        argname="primary_key",
                    )
                    for coerce_pk in self._primary_key_argument
                )
            ]
        else:
            coerced_pk_arg = None

        # if explicit PK argument sent, add those columns to the
        # primary key mappings
        if coerced_pk_arg:
            for k in coerced_pk_arg:
                if k.table not in self._pks_by_table:
                    self._pks_by_table[k.table] = util.OrderedSet()
                self._pks_by_table[k.table].add(k)

        # otherwise, see that we got a full PK for the mapped table
        elif (
            self.persist_selectable not in self._pks_by_table
            or len(self._pks_by_table[self.persist_selectable]) == 0
        ):
            raise sa_exc.ArgumentError(
                "Mapper %s could not assemble any primary "
                "key columns for mapped table '%s'"
                % (self, self.persist_selectable.description)
            )
        elif self.local_table not in self._pks_by_table and isinstance(
            self.local_table, schema.Table
        ):
            util.warn(
                "Could not assemble any primary "
                "keys for locally mapped table '%s' - "
                "no rows will be persisted in this Table."
                % self.local_table.description
            )

        if (
            self.inherits
            and not self.concrete
            and not self._primary_key_argument
        ):
            # if inheriting, the "primary key" for this mapper is
            # that of the inheriting (unless concrete or explicit)
            self.primary_key = self.inherits.primary_key
        else:
            # determine primary key from argument or persist_selectable pks
            primary_key: Collection[ColumnElement[Any]]

            if coerced_pk_arg:
                primary_key = [
                    cc if cc is not None else c
                    for cc, c in (
                        (self.persist_selectable.corresponding_column(c), c)
                        for c in coerced_pk_arg
                    )
                ]
            else:
                # if heuristically determined PKs, reduce to the minimal set
                # of columns by eliminating FK->PK pairs for a multi-table
                # expression.   May over-reduce for some kinds of UNIONs
                # / CTEs; use explicit PK argument for these special cases
                primary_key = sql_util.reduce_columns(
                    self._pks_by_table[self.persist_selectable],
                    ignore_nonexistent_tables=True,
                )

            if len(primary_key) == 0:
                raise sa_exc.ArgumentError(
                    "Mapper %s could not assemble any primary "
                    "key columns for mapped table '%s'"
                    % (self, self.persist_selectable.description)
                )

            self.primary_key = tuple(primary_key)
            self._log("Identified primary key columns: %s", primary_key)

        # determine cols that aren't expressed within our tables; mark these
        # as "read only" properties which are refreshed upon INSERT/UPDATE
        self._readonly_props = {
            self._columntoproperty[col]
            for col in self._columntoproperty
            if self._columntoproperty[col] not in self._identity_key_props
            and (
                not hasattr(col, "table")
                or col.table not in self._cols_by_table
            )
        }

    def _configure_properties(self) -> None:
        self.columns = self.c = sql_base.ColumnCollection()  # type: ignore

        # object attribute names mapped to MapperProperty objects
        self._props = util.OrderedDict()

        # table columns mapped to MapperProperty
        self._columntoproperty = _ColumnMapping(self)

        explicit_col_props_by_column: Dict[
            KeyedColumnElement[Any], Tuple[str, ColumnProperty[Any]]
        ] = {}
        explicit_col_props_by_key: Dict[str, ColumnProperty[Any]] = {}

        # step 1: go through properties that were explicitly passed
        # in the properties dictionary.  For Columns that are local, put them
        # aside in a separate collection we will reconcile with the Table
        # that's given.  For other properties, set them up in _props now.
        if self._init_properties:
            for key, prop_arg in self._init_properties.items():
                if not isinstance(prop_arg, MapperProperty):
                    possible_col_prop = self._make_prop_from_column(
                        key, prop_arg
                    )
                else:
                    possible_col_prop = prop_arg

                # issue #8705.  if the explicit property is actually a
                # Column that is local to the local Table, don't set it up
                # in ._props yet, integrate it into the order given within
                # the Table.

                _map_as_property_now = True
                if isinstance(possible_col_prop, properties.ColumnProperty):
                    for given_col in possible_col_prop.columns:
                        if self.local_table.c.contains_column(given_col):
                            _map_as_property_now = False
                            explicit_col_props_by_key[key] = possible_col_prop
                            explicit_col_props_by_column[given_col] = (
                                key,
                                possible_col_prop,
                            )

                if _map_as_property_now:
                    self._configure_property(
                        key,
                        possible_col_prop,
                        init=False,
                    )

        # step 2: pull properties from the inherited mapper.  reconcile
        # columns with those which are explicit above.  for properties that
        # are only in the inheriting mapper, set them up as local props
        if self.inherits:
            for key, inherited_prop in self.inherits._props.items():
                if self._should_exclude(key, key, local=False, column=None):
                    continue

                incoming_prop = explicit_col_props_by_key.get(key)
                if incoming_prop:
                    new_prop = self._reconcile_prop_with_incoming_columns(
                        key,
                        inherited_prop,
                        warn_only=False,
                        incoming_prop=incoming_prop,
                    )
                    explicit_col_props_by_key[key] = new_prop

                    for inc_col in incoming_prop.columns:
                        explicit_col_props_by_column[inc_col] = (
                            key,
                            new_prop,
                        )
                elif key not in self._props:
                    self._adapt_inherited_property(key, inherited_prop, False)

        # step 3.  Iterate through all columns in the persist selectable.
        # this includes not only columns in the local table / fromclause,
        # but also those columns in the superclass table if we are joined
        # inh or single inh mapper.  map these columns as well. additional
        # reconciliation against inherited columns occurs here also.

        for column in self.persist_selectable.columns:
            if column in explicit_col_props_by_column:
                # column was explicitly passed to properties; configure
                # it now in the order in which it corresponds to the
                # Table / selectable
                key, prop = explicit_col_props_by_column[column]
                self._configure_property(key, prop, init=False)
                continue

            elif column in self._columntoproperty:
                continue

            column_key = (self.column_prefix or "") + column.key
            if self._should_exclude(
                column.key,
                column_key,
                local=self.local_table.c.contains_column(column),
                column=column,
            ):
                continue

            # adjust the "key" used for this column to that
            # of the inheriting mapper
            for mapper in self.iterate_to_root():
                if column in mapper._columntoproperty:
                    column_key = mapper._columntoproperty[column].key

            self._configure_property(
                column_key,
                column,
                init=False,
                setparent=True,
            )

    def _configure_polymorphic_setter(self, init=False):
        """Configure an attribute on the mapper representing the
        'polymorphic_on' column, if applicable, and not
        already generated by _configure_properties (which is typical).

        Also create a setter function which will assign this
        attribute to the value of the 'polymorphic_identity'
        upon instance construction, also if applicable.  This
        routine will run when an instance is created.

        """
        setter = False
        polymorphic_key: Optional[str] = None

        if self.polymorphic_on is not None:
            setter = True

            if isinstance(self.polymorphic_on, str):
                # polymorphic_on specified as a string - link
                # it to mapped ColumnProperty
                try:
                    self.polymorphic_on = self._props[self.polymorphic_on]
                except KeyError as err:
                    raise sa_exc.ArgumentError(
                        "Can't determine polymorphic_on "
                        "value '%s' - no attribute is "
                        "mapped to this name." % self.polymorphic_on
                    ) from err

            if self.polymorphic_on in self._columntoproperty:
                # polymorphic_on is a column that is already mapped
                # to a ColumnProperty
                prop = self._columntoproperty[self.polymorphic_on]
            elif isinstance(self.polymorphic_on, MapperProperty):
                # polymorphic_on is directly a MapperProperty,
                # ensure it's a ColumnProperty
                if not isinstance(
                    self.polymorphic_on, properties.ColumnProperty
                ):
                    raise sa_exc.ArgumentError(
                        "Only direct column-mapped "
                        "property or SQL expression "
                        "can be passed for polymorphic_on"
                    )
                prop = self.polymorphic_on
            else:
                # polymorphic_on is a Column or SQL expression and
                # doesn't appear to be mapped. this means it can be 1.
                # only present in the with_polymorphic selectable or
                # 2. a totally standalone SQL expression which we'd
                # hope is compatible with this mapper's persist_selectable
                col = self.persist_selectable.corresponding_column(
                    self.polymorphic_on
                )
                if col is None:
                    # polymorphic_on doesn't derive from any
                    # column/expression isn't present in the mapped
                    # table. we will make a "hidden" ColumnProperty
                    # for it. Just check that if it's directly a
                    # schema.Column and we have with_polymorphic, it's
                    # likely a user error if the schema.Column isn't
                    # represented somehow in either persist_selectable or
                    # with_polymorphic.   Otherwise as of 0.7.4 we
                    # just go with it and assume the user wants it
                    # that way (i.e. a CASE statement)
                    setter = False
                    instrument = False
                    col = self.polymorphic_on
                    if isinstance(col, schema.Column) and (
                        self.with_polymorphic is None
                        or self.with_polymorphic[1] is None
                        or self.with_polymorphic[1].corresponding_column(col)
                        is None
                    ):
                        raise sa_exc.InvalidRequestError(
                            "Could not map polymorphic_on column "
                            "'%s' to the mapped table - polymorphic "
                            "loads will not function properly"
                            % col.description
                        )
                else:
                    # column/expression that polymorphic_on derives from
                    # is present in our mapped table
                    # and is probably mapped, but polymorphic_on itself
                    # is not.  This happens when
                    # the polymorphic_on is only directly present in the
                    # with_polymorphic selectable, as when use
                    # polymorphic_union.
                    # we'll make a separate ColumnProperty for it.
                    instrument = True
                key = getattr(col, "key", None)
                if key:
                    if self._should_exclude(key, key, False, col):
                        raise sa_exc.InvalidRequestError(
                            "Cannot exclude or override the "
                            "discriminator column %r" % key
                        )
                else:
                    self.polymorphic_on = col = col.label("_sa_polymorphic_on")
                    key = col.key

                prop = properties.ColumnProperty(col, _instrument=instrument)
                self._configure_property(key, prop, init=init, setparent=True)

            # the actual polymorphic_on should be the first public-facing
            # column in the property
            self.polymorphic_on = prop.columns[0]
            polymorphic_key = prop.key
        else:
            # no polymorphic_on was set.
            # check inheriting mappers for one.
            for mapper in self.iterate_to_root():
                # determine if polymorphic_on of the parent
                # should be propagated here.   If the col
                # is present in our mapped table, or if our mapped
                # table is the same as the parent (i.e. single table
                # inheritance), we can use it
                if mapper.polymorphic_on is not None:
                    if self.persist_selectable is mapper.persist_selectable:
                        self.polymorphic_on = mapper.polymorphic_on
                    else:
                        self.polymorphic_on = (
                            self.persist_selectable
                        ).corresponding_column(mapper.polymorphic_on)
                    # we can use the parent mapper's _set_polymorphic_identity
                    # directly; it ensures the polymorphic_identity of the
                    # instance's mapper is used so is portable to subclasses.
                    if self.polymorphic_on is not None:
                        self._set_polymorphic_identity = (
                            mapper._set_polymorphic_identity
                        )
                        self._polymorphic_attr_key = (
                            mapper._polymorphic_attr_key
                        )
                        self._validate_polymorphic_identity = (
                            mapper._validate_polymorphic_identity
                        )
                    else:
                        self._set_polymorphic_identity = None
                        self._polymorphic_attr_key = None
                    return

        if self.polymorphic_abstract and self.polymorphic_on is None:
            raise sa_exc.InvalidRequestError(
                "The Mapper.polymorphic_abstract parameter may only be used "
                "on a mapper hierarchy which includes the "
                "Mapper.polymorphic_on parameter at the base of the hierarchy."
            )

        if setter:

            def _set_polymorphic_identity(state):
                dict_ = state.dict
                # TODO: what happens if polymorphic_on column attribute name
                # does not match .key?

                polymorphic_identity = (
                    state.manager.mapper.polymorphic_identity
                )
                if (
                    polymorphic_identity is None
                    and state.manager.mapper.polymorphic_abstract
                ):
                    raise sa_exc.InvalidRequestError(
                        f"Can't instantiate class for {state.manager.mapper}; "
                        "mapper is marked polymorphic_abstract=True"
                    )

                state.get_impl(polymorphic_key).set(
                    state,
                    dict_,
                    polymorphic_identity,
                    None,
                )

            self._polymorphic_attr_key = polymorphic_key

            def _validate_polymorphic_identity(mapper, state, dict_):
                if (
                    polymorphic_key in dict_
                    and dict_[polymorphic_key]
                    not in mapper._acceptable_polymorphic_identities
                ):
                    util.warn_limited(
                        "Flushing object %s with "
                        "incompatible polymorphic identity %r; the "
                        "object may not refresh and/or load correctly",
                        (state_str(state), dict_[polymorphic_key]),
                    )

            self._set_polymorphic_identity = _set_polymorphic_identity
            self._validate_polymorphic_identity = (
                _validate_polymorphic_identity
            )
        else:
            self._polymorphic_attr_key = None
            self._set_polymorphic_identity = None

    _validate_polymorphic_identity = None

    @HasMemoized.memoized_attribute
    def _version_id_prop(self):
        if self.version_id_col is not None:
            return self._columntoproperty[self.version_id_col]
        else:
            return None

    @HasMemoized.memoized_attribute
    def _acceptable_polymorphic_identities(self):
        identities = set()

        stack = deque([self])
        while stack:
            item = stack.popleft()
            if item.persist_selectable is self.persist_selectable:
                identities.add(item.polymorphic_identity)
                stack.extend(item._inheriting_mappers)

        return identities

    @HasMemoized.memoized_attribute
    def _prop_set(self):
        return frozenset(self._props.values())

    @util.preload_module("sqlalchemy.orm.descriptor_props")
    def _adapt_inherited_property(self, key, prop, init):
        descriptor_props = util.preloaded.orm_descriptor_props

        if not self.concrete:
            self._configure_property(key, prop, init=False, setparent=False)
        elif key not in self._props:
            # determine if the class implements this attribute; if not,
            # or if it is implemented by the attribute that is handling the
            # given superclass-mapped property, then we need to report that we
            # can't use this at the instance level since we are a concrete
            # mapper and we don't map this.  don't trip user-defined
            # descriptors that might have side effects when invoked.
            implementing_attribute = self.class_manager._get_class_attr_mro(
                key, prop
            )
            if implementing_attribute is prop or (
                isinstance(
                    implementing_attribute, attributes.InstrumentedAttribute
                )
                and implementing_attribute._parententity is prop.parent
            ):
                self._configure_property(
                    key,
                    descriptor_props.ConcreteInheritedProperty(),
                    init=init,
                    setparent=True,
                )

    @util.preload_module("sqlalchemy.orm.descriptor_props")
    def _configure_property(
        self,
        key: str,
        prop_arg: Union[KeyedColumnElement[Any], MapperProperty[Any]],
        *,
        init: bool = True,
        setparent: bool = True,
        warn_for_existing: bool = False,
    ) -> MapperProperty[Any]:
        descriptor_props = util.preloaded.orm_descriptor_props
        self._log(
            "_configure_property(%s, %s)", key, prop_arg.__class__.__name__
        )

        if not isinstance(prop_arg, MapperProperty):
            prop: MapperProperty[Any] = self._property_from_column(
                key, prop_arg
            )
        else:
            prop = prop_arg

        if isinstance(prop, properties.ColumnProperty):
            col = self.persist_selectable.corresponding_column(prop.columns[0])

            # if the column is not present in the mapped table,
            # test if a column has been added after the fact to the
            # parent table (or their parent, etc.) [ticket:1570]
            if col is None and self.inherits:
                path = [self]
                for m in self.inherits.iterate_to_root():
                    col = m.local_table.corresponding_column(prop.columns[0])
                    if col is not None:
                        for m2 in path:
                            m2.persist_selectable._refresh_for_new_column(col)
                        col = self.persist_selectable.corresponding_column(
                            prop.columns[0]
                        )
                        break
                    path.append(m)

            # subquery expression, column not present in the mapped
            # selectable.
            if col is None:
                col = prop.columns[0]

                # column is coming in after _readonly_props was
                # initialized; check for 'readonly'
                if hasattr(self, "_readonly_props") and (
                    not hasattr(col, "table")
                    or col.table not in self._cols_by_table
                ):
                    self._readonly_props.add(prop)

            else:
                # if column is coming in after _cols_by_table was
                # initialized, ensure the col is in the right set
                if (
                    hasattr(self, "_cols_by_table")
                    and col.table in self._cols_by_table
                    and col not in self._cols_by_table[col.table]
                ):
                    self._cols_by_table[col.table].add(col)

            # if this properties.ColumnProperty represents the "polymorphic
            # discriminator" column, mark it.  We'll need this when rendering
            # columns in SELECT statements.
            if not hasattr(prop, "_is_polymorphic_discriminator"):
                prop._is_polymorphic_discriminator = (
                    col is self.polymorphic_on
                    or prop.columns[0] is self.polymorphic_on
                )

            if isinstance(col, expression.Label):
                # new in 1.4, get column property against expressions
                # to be addressable in subqueries
                col.key = col._tq_key_label = key

            self.columns.add(col, key)

            for col in prop.columns:
                for proxy_col in col.proxy_set:
                    self._columntoproperty[proxy_col] = prop

        if getattr(prop, "key", key) != key:
            util.warn(
                f"ORM mapped property {self.class_.__name__}.{prop.key} being "
                "assigned to attribute "
                f"{key!r} is already associated with "
                f"attribute {prop.key!r}. The attribute will be de-associated "
                f"from {prop.key!r}."
            )

        prop.key = key

        if setparent:
            prop.set_parent(self, init)

        if key in self._props and getattr(
            self._props[key], "_mapped_by_synonym", False
        ):
            syn = self._props[key]._mapped_by_synonym
            raise sa_exc.ArgumentError(
                "Can't call map_column=True for synonym %r=%r, "
                "a ColumnProperty already exists keyed to the name "
                "%r for column %r" % (syn, key, key, syn)
            )

        # replacement cases

        # case one: prop is replacing a prop that we have mapped.  this is
        # independent of whatever might be in the actual class dictionary
        if (
            key in self._props
            and not isinstance(
                self._props[key], descriptor_props.ConcreteInheritedProperty
            )
            and not isinstance(prop, descriptor_props.SynonymProperty)
        ):
            if warn_for_existing:
                util.warn_deprecated(
                    f"User-placed attribute {self.class_.__name__}.{key} on "
                    f"{self} is replacing an existing ORM-mapped attribute.  "
                    "Behavior is not fully defined in this case.  This "
                    "use is deprecated and will raise an error in a future "
                    "release",
                    "2.0",
                )
            oldprop = self._props[key]
            self._path_registry.pop(oldprop, None)

        # case two: prop is replacing an attribute on the class of some kind.
        # we have to be more careful here since it's normal when using
        # Declarative that all the "declared attributes" on the class
        # get replaced.
        elif (
            warn_for_existing
            and self.class_.__dict__.get(key, None) is not None
            and not isinstance(prop, descriptor_props.SynonymProperty)
            and not isinstance(
                self._props.get(key, None),
                descriptor_props.ConcreteInheritedProperty,
            )
        ):
            util.warn_deprecated(
                f"User-placed attribute {self.class_.__name__}.{key} on "
                f"{self} is replacing an existing class-bound "
                "attribute of the same name.  "
                "Behavior is not fully defined in this case.  This "
                "use is deprecated and will raise an error in a future "
                "release",
                "2.0",
            )

        self._props[key] = prop

        if not self.non_primary:
            prop.instrument_class(self)

        for mapper in self._inheriting_mappers:
            mapper._adapt_inherited_property(key, prop, init)

        if init:
            prop.init()
            prop.post_instrument_class(self)

        if self.configured:
            self._expire_memoizations()

        return prop

    def _make_prop_from_column(
        self,
        key: str,
        column: Union[
            Sequence[KeyedColumnElement[Any]], KeyedColumnElement[Any]
        ],
    ) -> ColumnProperty[Any]:
        columns = util.to_list(column)
        mapped_column = []
        for c in columns:
            mc = self.persist_selectable.corresponding_column(c)
            if mc is None:
                mc = self.local_table.corresponding_column(c)
                if mc is not None:
                    # if the column is in the local table but not the
                    # mapped table, this corresponds to adding a
                    # column after the fact to the local table.
                    # [ticket:1523]
                    self.persist_selectable._refresh_for_new_column(mc)
                mc = self.persist_selectable.corresponding_column(c)
                if mc is None:
                    raise sa_exc.ArgumentError(
                        "When configuring property '%s' on %s, "
                        "column '%s' is not represented in the mapper's "
                        "table. Use the `column_property()` function to "
                        "force this column to be mapped as a read-only "
                        "attribute." % (key, self, c)
                    )
            mapped_column.append(mc)
        return properties.ColumnProperty(*mapped_column)

    def _reconcile_prop_with_incoming_columns(
        self,
        key: str,
        existing_prop: MapperProperty[Any],
        warn_only: bool,
        incoming_prop: Optional[ColumnProperty[Any]] = None,
        single_column: Optional[KeyedColumnElement[Any]] = None,
    ) -> ColumnProperty[Any]:
        if incoming_prop and (
            self.concrete
            or not isinstance(existing_prop, properties.ColumnProperty)
        ):
            return incoming_prop

        existing_column = existing_prop.columns[0]

        if incoming_prop and existing_column in incoming_prop.columns:
            return incoming_prop

        if incoming_prop is None:
            assert single_column is not None
            incoming_column = single_column
            equated_pair_key = (existing_prop.columns[0], incoming_column)
        else:
            assert single_column is None
            incoming_column = incoming_prop.columns[0]
            equated_pair_key = (incoming_column, existing_prop.columns[0])

        if (
            (
                not self._inherits_equated_pairs
                or (equated_pair_key not in self._inherits_equated_pairs)
            )
            and not existing_column.shares_lineage(incoming_column)
            and existing_column is not self.version_id_col
            and incoming_column is not self.version_id_col
        ):
            msg = (
                "Implicitly combining column %s with column "
                "%s under attribute '%s'.  Please configure one "
                "or more attributes for these same-named columns "
                "explicitly."
                % (
                    existing_prop.columns[-1],
                    incoming_column,
                    key,
                )
            )
            if warn_only:
                util.warn(msg)
            else:
                raise sa_exc.InvalidRequestError(msg)

        # existing properties.ColumnProperty from an inheriting
        # mapper. make a copy and append our column to it
        # breakpoint()
        new_prop = existing_prop.copy()

        new_prop.columns.insert(0, incoming_column)
        self._log(
            "inserting column to existing list "
            "in properties.ColumnProperty %s",
            key,
        )
        return new_prop  # type: ignore

    @util.preload_module("sqlalchemy.orm.descriptor_props")
    def _property_from_column(
        self,
        key: str,
        column: KeyedColumnElement[Any],
    ) -> ColumnProperty[Any]:
        """generate/update a :class:`.ColumnProperty` given a
        :class:`_schema.Column` or other SQL expression object."""

        descriptor_props = util.preloaded.orm_descriptor_props

        prop = self._props.get(key)

        if isinstance(prop, properties.ColumnProperty):
            return self._reconcile_prop_with_incoming_columns(
                key,
                prop,
                single_column=column,
                warn_only=prop.parent is not self,
            )
        elif prop is None or isinstance(
            prop, descriptor_props.ConcreteInheritedProperty
        ):
            return self._make_prop_from_column(key, column)
        else:
            raise sa_exc.ArgumentError(
                "WARNING: when configuring property '%s' on %s, "
                "column '%s' conflicts with property '%r'. "
                "To resolve this, map the column to the class under a "
                "different name in the 'properties' dictionary.  Or, "
                "to remove all awareness of the column entirely "
                "(including its availability as a foreign key), "
                "use the 'include_properties' or 'exclude_properties' "
                "mapper arguments to control specifically which table "
                "columns get mapped." % (key, self, column.key, prop)
            )

    @util.langhelpers.tag_method_for_warnings(
        "This warning originated from the `configure_mappers()` process, "
        "which was invoked automatically in response to a user-initiated "
        "operation.",
        sa_exc.SAWarning,
    )
    def _check_configure(self) -> None:
        if self.registry._new_mappers:
            _configure_registries({self.registry}, cascade=True)

    def _post_configure_properties(self) -> None:
        """Call the ``init()`` method on all ``MapperProperties``
        attached to this mapper.

        This is a deferred configuration step which is intended
        to execute once all mappers have been constructed.

        """

        self._log("_post_configure_properties() started")
        l = [(key, prop) for key, prop in self._props.items()]
        for key, prop in l:
            self._log("initialize prop %s", key)

            if prop.parent is self and not prop._configure_started:
                prop.init()

            if prop._configure_finished:
                prop.post_instrument_class(self)

        self._log("_post_configure_properties() complete")
        self.configured = True

    def add_properties(self, dict_of_properties):
        """Add the given dictionary of properties to this mapper,
        using `add_property`.

        """
        for key, value in dict_of_properties.items():
            self.add_property(key, value)

    def add_property(
        self, key: str, prop: Union[Column[Any], MapperProperty[Any]]
    ) -> None:
        """Add an individual MapperProperty to this mapper.

        If the mapper has not been configured yet, just adds the
        property to the initial properties dictionary sent to the
        constructor.  If this Mapper has already been configured, then
        the given MapperProperty is configured immediately.

        """
        prop = self._configure_property(
            key, prop, init=self.configured, warn_for_existing=True
        )
        assert isinstance(prop, MapperProperty)
        self._init_properties[key] = prop

    def _expire_memoizations(self) -> None:
        for mapper in self.iterate_to_root():
            mapper._reset_memoizations()

    @property
    def _log_desc(self) -> str:
        return (
            "("
            + self.class_.__name__
            + "|"
            + (
                self.local_table is not None
                and self.local_table.description
                or str(self.local_table)
            )
            + (self.non_primary and "|non-primary" or "")
            + ")"
        )

    def _log(self, msg: str, *args: Any) -> None:
        self.logger.info("%s " + msg, *((self._log_desc,) + args))

    def _log_debug(self, msg: str, *args: Any) -> None:
        self.logger.debug("%s " + msg, *((self._log_desc,) + args))

    def __repr__(self) -> str:
        return "<Mapper at 0x%x; %s>" % (id(self), self.class_.__name__)

    def __str__(self) -> str:
        return "Mapper[%s%s(%s)]" % (
            self.class_.__name__,
            self.non_primary and " (non-primary)" or "",
            (
                self.local_table.description
                if self.local_table is not None
                else self.persist_selectable.description
            ),
        )

    def _is_orphan(self, state: InstanceState[_O]) -> bool:
        orphan_possible = False
        for mapper in self.iterate_to_root():
            for key, cls in mapper._delete_orphans:
                orphan_possible = True

                has_parent = attributes.manager_of_class(cls).has_parent(
                    state, key, optimistic=state.has_identity
                )

                if self.legacy_is_orphan and has_parent:
                    return False
                elif not self.legacy_is_orphan and not has_parent:
                    return True

        if self.legacy_is_orphan:
            return orphan_possible
        else:
            return False

    def has_property(self, key: str) -> bool:
        return key in self._props

    def get_property(
        self, key: str, _configure_mappers: bool = False
    ) -> MapperProperty[Any]:
        """return a MapperProperty associated with the given key."""

        if _configure_mappers:
            self._check_configure()

        try:
            return self._props[key]
        except KeyError as err:
            raise sa_exc.InvalidRequestError(
                f"Mapper '{self}' has no property '{key}'.  If this property "
                "was indicated from other mappers or configure events, ensure "
                "registry.configure() has been called."
            ) from err

    def get_property_by_column(
        self, column: ColumnElement[_T]
    ) -> MapperProperty[_T]:
        """Given a :class:`_schema.Column` object, return the
        :class:`.MapperProperty` which maps this column."""

        return self._columntoproperty[column]

    @property
    def iterate_properties(self):
        """return an iterator of all MapperProperty objects."""

        return iter(self._props.values())

    def _mappers_from_spec(
        self, spec: Any, selectable: Optional[FromClause]
    ) -> Sequence[Mapper[Any]]:
        """given a with_polymorphic() argument, return the set of mappers it
        represents.

        Trims the list of mappers to just those represented within the given
        selectable, if present. This helps some more legacy-ish mappings.

        """
        if spec == "*":
            mappers = list(self.self_and_descendants)
        elif spec:
            mapper_set = set()
            for m in util.to_list(spec):
                m = _class_to_mapper(m)
                if not m.isa(self):
                    raise sa_exc.InvalidRequestError(
                        "%r does not inherit from %r" % (m, self)
                    )

                if selectable is None:
                    mapper_set.update(m.iterate_to_root())
                else:
                    mapper_set.add(m)
            mappers = [m for m in self.self_and_descendants if m in mapper_set]
        else:
            mappers = []

        if selectable is not None:
            tables = set(
                sql_util.find_tables(selectable, include_aliases=True)
            )
            mappers = [m for m in mappers if m.local_table in tables]
        return mappers

    def _selectable_from_mappers(
        self, mappers: Iterable[Mapper[Any]], innerjoin: bool
    ) -> FromClause:
        """given a list of mappers (assumed to be within this mapper's
        inheritance hierarchy), construct an outerjoin amongst those mapper's
        mapped tables.

        """
        from_obj = self.persist_selectable
        for m in mappers:
            if m is self:
                continue
            if m.concrete:
                raise sa_exc.InvalidRequestError(
                    "'with_polymorphic()' requires 'selectable' argument "
                    "when concrete-inheriting mappers are used."
                )
            elif not m.single:
                if innerjoin:
                    from_obj = from_obj.join(
                        m.local_table, m.inherit_condition
                    )
                else:
                    from_obj = from_obj.outerjoin(
                        m.local_table, m.inherit_condition
                    )

        return from_obj

    @HasMemoized.memoized_attribute
    def _version_id_has_server_side_value(self) -> bool:
        vid_col = self.version_id_col

        if vid_col is None:
            return False

        elif not isinstance(vid_col, Column):
            return True
        else:
            return vid_col.server_default is not None or (
                vid_col.default is not None
                and (
                    not vid_col.default.is_scalar
                    and not vid_col.default.is_callable
                )
            )

    @HasMemoized.memoized_attribute
    def _single_table_criterion(self):
        if self.single and self.inherits and self.polymorphic_on is not None:
            return self.polymorphic_on._annotate(
                {"parententity": self, "parentmapper": self}
            ).in_(
                [
                    m.polymorphic_identity
                    for m in self.self_and_descendants
                    if not m.polymorphic_abstract
                ]
            )
        else:
            return None

    @HasMemoized.memoized_attribute
    def _has_aliased_polymorphic_fromclause(self):
        """return True if with_polymorphic[1] is an aliased fromclause,
        like a subquery.

        As of #8168, polymorphic adaption with ORMAdapter is used only
        if this is present.

        """
        return self.with_polymorphic and isinstance(
            self.with_polymorphic[1],
            expression.AliasedReturnsRows,
        )

    @HasMemoized.memoized_attribute
    def _should_select_with_poly_adapter(self):
        """determine if _MapperEntity or _ORMColumnEntity will need to use
        polymorphic adaption when setting up a SELECT as well as fetching
        rows for mapped classes and subclasses against this Mapper.

        moved here from context.py for #8456 to generalize the ruleset
        for this condition.

        """

        # this has been simplified as of #8456.
        # rule is: if we have a with_polymorphic or a concrete-style
        # polymorphic selectable, *or* if the base mapper has either of those,
        # we turn on the adaption thing.  if not, we do *no* adaption.
        #
        # (UPDATE for #8168: the above comment was not accurate, as we were
        # still saying "do polymorphic" if we were using an auto-generated
        # flattened JOIN for with_polymorphic.)
        #
        # this splits the behavior among the "regular" joined inheritance
        # and single inheritance mappers, vs. the "weird / difficult"
        # concrete and joined inh mappings that use a with_polymorphic of
        # some kind or polymorphic_union.
        #
        # note we have some tests in test_polymorphic_rel that query against
        # a subclass, then refer to the superclass that has a with_polymorphic
        # on it (such as test_join_from_polymorphic_explicit_aliased_three).
        # these tests actually adapt the polymorphic selectable (like, the
        # UNION or the SELECT subquery with JOIN in it) to be just the simple
        # subclass table.   Hence even if we are a "plain" inheriting mapper
        # but our base has a wpoly on it, we turn on adaption.  This is a
        # legacy case we should probably disable.
        #
        #
        # UPDATE: simplified way more as of #8168.   polymorphic adaption
        # is turned off even if with_polymorphic is set, as long as there
        # is no user-defined aliased selectable / subquery configured.
        # this scales back the use of polymorphic adaption in practice
        # to basically no cases except for concrete inheritance with a
        # polymorphic base class.
        #
        return (
            self._has_aliased_polymorphic_fromclause
            or self._requires_row_aliasing
            or (self.base_mapper._has_aliased_polymorphic_fromclause)
            or self.base_mapper._requires_row_aliasing
        )

    @HasMemoized.memoized_attribute
    def _with_polymorphic_mappers(self) -> Sequence[Mapper[Any]]:
        self._check_configure()

        if not self.with_polymorphic:
            return []
        return self._mappers_from_spec(*self.with_polymorphic)

    @HasMemoized.memoized_attribute
    def _post_inspect(self):
        """This hook is invoked by attribute inspection.

        E.g. when Query calls:

            coercions.expect(roles.ColumnsClauseRole, ent, keep_inspect=True)

        This allows the inspection process run a configure mappers hook.

        """
        self._check_configure()

    @HasMemoized_ro_memoized_attribute
    def _with_polymorphic_selectable(self) -> FromClause:
        if not self.with_polymorphic:
            return self.persist_selectable

        spec, selectable = self.with_polymorphic
        if selectable is not None:
            return selectable
        else:
            return self._selectable_from_mappers(
                self._mappers_from_spec(spec, selectable), False
            )

    with_polymorphic_mappers = _with_polymorphic_mappers
    """The list of :class:`_orm.Mapper` objects included in the
    default "polymorphic" query.

    """

    @HasMemoized_ro_memoized_attribute
    def _insert_cols_evaluating_none(self):
        return {
            table: frozenset(
                col for col in columns if col.type.should_evaluate_none
            )
            for table, columns in self._cols_by_table.items()
        }

    @HasMemoized.memoized_attribute
    def _insert_cols_as_none(self):
        return {
            table: frozenset(
                col.key
                for col in columns
                if not col.primary_key
                and not col.server_default
                and not col.default
                and not col.type.should_evaluate_none
            )
            for table, columns in self._cols_by_table.items()
        }

    @HasMemoized.memoized_attribute
    def _propkey_to_col(self):
        return {
            table: {self._columntoproperty[col].key: col for col in columns}
            for table, columns in self._cols_by_table.items()
        }

    @HasMemoized.memoized_attribute
    def _pk_keys_by_table(self):
        return {
            table: frozenset([col.key for col in pks])
            for table, pks in self._pks_by_table.items()
        }

    @HasMemoized.memoized_attribute
    def _pk_attr_keys_by_table(self):
        return {
            table: frozenset([self._columntoproperty[col].key for col in pks])
            for table, pks in self._pks_by_table.items()
        }

    @HasMemoized.memoized_attribute
    def _server_default_cols(
        self,
    ) -> Mapping[FromClause, FrozenSet[Column[Any]]]:
        return {
            table: frozenset(
                [
                    col
                    for col in cast("Iterable[Column[Any]]", columns)
                    if col.server_default is not None
                    or (
                        col.default is not None
                        and col.default.is_clause_element
                    )
                ]
            )
            for table, columns in self._cols_by_table.items()
        }

    @HasMemoized.memoized_attribute
    def _server_onupdate_default_cols(
        self,
    ) -> Mapping[FromClause, FrozenSet[Column[Any]]]:
        return {
            table: frozenset(
                [
                    col
                    for col in cast("Iterable[Column[Any]]", columns)
                    if col.server_onupdate is not None
                    or (
                        col.onupdate is not None
                        and col.onupdate.is_clause_element
                    )
                ]
            )
            for table, columns in self._cols_by_table.items()
        }

    @HasMemoized.memoized_attribute
    def _server_default_col_keys(self) -> Mapping[FromClause, FrozenSet[str]]:
        return {
            table: frozenset(col.key for col in cols if col.key is not None)
            for table, cols in self._server_default_cols.items()
        }

    @HasMemoized.memoized_attribute
    def _server_onupdate_default_col_keys(
        self,
    ) -> Mapping[FromClause, FrozenSet[str]]:
        return {
            table: frozenset(col.key for col in cols if col.key is not None)
            for table, cols in self._server_onupdate_default_cols.items()
        }

    @HasMemoized.memoized_attribute
    def _server_default_plus_onupdate_propkeys(self) -> Set[str]:
        result: Set[str] = set()

        col_to_property = self._columntoproperty
        for table, columns in self._server_default_cols.items():
            result.update(
                col_to_property[col].key
                for col in columns.intersection(col_to_property)
            )
        for table, columns in self._server_onupdate_default_cols.items():
            result.update(
                col_to_property[col].key
                for col in columns.intersection(col_to_property)
            )
        return result

    @HasMemoized.memoized_instancemethod
    def __clause_element__(self):
        annotations: Dict[str, Any] = {
            "entity_namespace": self,
            "parententity": self,
            "parentmapper": self,
        }
        if self.persist_selectable is not self.local_table:
            # joined table inheritance, with polymorphic selectable,
            # etc.
            annotations["dml_table"] = self.local_table._annotate(
                {
                    "entity_namespace": self,
                    "parententity": self,
                    "parentmapper": self,
                }
            )._set_propagate_attrs(
                {"compile_state_plugin": "orm", "plugin_subject": self}
            )

        return self.selectable._annotate(annotations)._set_propagate_attrs(
            {"compile_state_plugin": "orm", "plugin_subject": self}
        )

    @util.memoized_property
    def select_identity_token(self):
        return (
            expression.null()
            ._annotate(
                {
                    "entity_namespace": self,
                    "parententity": self,
                    "parentmapper": self,
                    "identity_token": True,
                }
            )
            ._set_propagate_attrs(
                {"compile_state_plugin": "orm", "plugin_subject": self}
            )
        )

    @property
    def selectable(self) -> FromClause:
        """The :class:`_schema.FromClause` construct this
        :class:`_orm.Mapper` selects from by default.

        Normally, this is equivalent to :attr:`.persist_selectable`, unless
        the ``with_polymorphic`` feature is in use, in which case the
        full "polymorphic" selectable is returned.

        """
        return self._with_polymorphic_selectable

    def _with_polymorphic_args(
        self,
        spec: Any = None,
        selectable: Union[Literal[False, None], FromClause] = False,
        innerjoin: bool = False,
    ) -> Tuple[Sequence[Mapper[Any]], FromClause]:
        if selectable not in (None, False):
            selectable = coercions.expect(
                roles.StrictFromClauseRole, selectable, allow_select=True
            )

        if self.with_polymorphic:
            if not spec:
                spec = self.with_polymorphic[0]
            if selectable is False:
                selectable = self.with_polymorphic[1]
        elif selectable is False:
            selectable = None
        mappers = self._mappers_from_spec(spec, selectable)
        if selectable is not None:
            return mappers, selectable
        else:
            return mappers, self._selectable_from_mappers(mappers, innerjoin)

    @HasMemoized.memoized_attribute
    def _polymorphic_properties(self):
        return list(
            self._iterate_polymorphic_properties(
                self._with_polymorphic_mappers
            )
        )

    @property
    def _all_column_expressions(self):
        poly_properties = self._polymorphic_properties
        adapter = self._polymorphic_adapter

        return [
            adapter.columns[c] if adapter else c
            for prop in poly_properties
            if isinstance(prop, properties.ColumnProperty)
            and prop._renders_in_subqueries
            for c in prop.columns
        ]

    def _columns_plus_keys(self, polymorphic_mappers=()):
        if polymorphic_mappers:
            poly_properties = self._iterate_polymorphic_properties(
                polymorphic_mappers
            )
        else:
            poly_properties = self._polymorphic_properties

        return [
            (prop.key, prop.columns[0])
            for prop in poly_properties
            if isinstance(prop, properties.ColumnProperty)
        ]

    @HasMemoized.memoized_attribute
    def _polymorphic_adapter(self) -> Optional[orm_util.ORMAdapter]:
        if self._has_aliased_polymorphic_fromclause:
            return orm_util.ORMAdapter(
                orm_util._TraceAdaptRole.MAPPER_POLYMORPHIC_ADAPTER,
                self,
                selectable=self.selectable,
                equivalents=self._equivalent_columns,
                limit_on_entity=False,
            )
        else:
            return None

    def _iterate_polymorphic_properties(self, mappers=None):
        """Return an iterator of MapperProperty objects which will render into
        a SELECT."""
        if mappers is None:
            mappers = self._with_polymorphic_mappers

        if not mappers:
            for c in self.iterate_properties:
                yield c
        else:
            # in the polymorphic case, filter out discriminator columns
            # from other mappers, as these are sometimes dependent on that
            # mapper's polymorphic selectable (which we don't want rendered)
            for c in util.unique_list(
                chain(
                    *[
                        list(mapper.iterate_properties)
                        for mapper in [self] + mappers
                    ]
                )
            ):
                if getattr(c, "_is_polymorphic_discriminator", False) and (
                    self.polymorphic_on is None
                    or c.columns[0] is not self.polymorphic_on
                ):
                    continue
                yield c

    @HasMemoized.memoized_attribute
    def attrs(self) -> util.ReadOnlyProperties[MapperProperty[Any]]:
        """A namespace of all :class:`.MapperProperty` objects
        associated this mapper.

        This is an object that provides each property based on
        its key name.  For instance, the mapper for a
        ``User`` class which has ``User.name`` attribute would
        provide ``mapper.attrs.name``, which would be the
        :class:`.ColumnProperty` representing the ``name``
        column.   The namespace object can also be iterated,
        which would yield each :class:`.MapperProperty`.

        :class:`_orm.Mapper` has several pre-filtered views
        of this attribute which limit the types of properties
        returned, including :attr:`.synonyms`, :attr:`.column_attrs`,
        :attr:`.relationships`, and :attr:`.composites`.

        .. warning::

            The :attr:`_orm.Mapper.attrs` accessor namespace is an
            instance of :class:`.OrderedProperties`.  This is
            a dictionary-like object which includes a small number of
            named methods such as :meth:`.OrderedProperties.items`
            and :meth:`.OrderedProperties.values`.  When
            accessing attributes dynamically, favor using the dict-access
            scheme, e.g. ``mapper.attrs[somename]`` over
            ``getattr(mapper.attrs, somename)`` to avoid name collisions.

        .. seealso::

            :attr:`_orm.Mapper.all_orm_descriptors`

        """

        self._check_configure()
        return util.ReadOnlyProperties(self._props)

    @HasMemoized.memoized_attribute
    def all_orm_descriptors(self) -> util.ReadOnlyProperties[InspectionAttr]:
        """A namespace of all :class:`.InspectionAttr` attributes associated
        with the mapped class.

        These attributes are in all cases Python :term:`descriptors`
        associated with the mapped class or its superclasses.

        This namespace includes attributes that are mapped to the class
        as well as attributes declared by extension modules.
        It includes any Python descriptor type that inherits from
        :class:`.InspectionAttr`.  This includes
        :class:`.QueryableAttribute`, as well as extension types such as
        :class:`.hybrid_property`, :class:`.hybrid_method` and
        :class:`.AssociationProxy`.

        To distinguish between mapped attributes and extension attributes,
        the attribute :attr:`.InspectionAttr.extension_type` will refer
        to a constant that distinguishes between different extension types.

        The sorting of the attributes is based on the following rules:

        1. Iterate through the class and its superclasses in order from
           subclass to superclass (i.e. iterate through ``cls.__mro__``)

        2. For each class, yield the attributes in the order in which they
           appear in ``__dict__``, with the exception of those in step
           3 below.  In Python 3.6 and above this ordering will be the
           same as that of the class' construction, with the exception
           of attributes that were added after the fact by the application
           or the mapper.

        3. If a certain attribute key is also in the superclass ``__dict__``,
           then it's included in the iteration for that class, and not the
           class in which it first appeared.

        The above process produces an ordering that is deterministic in terms
        of the order in which attributes were assigned to the class.

        .. versionchanged:: 1.3.19 ensured deterministic ordering for
           :meth:`_orm.Mapper.all_orm_descriptors`.

        When dealing with a :class:`.QueryableAttribute`, the
        :attr:`.QueryableAttribute.property` attribute refers to the
        :class:`.MapperProperty` property, which is what you get when
        referring to the collection of mapped properties via
        :attr:`_orm.Mapper.attrs`.

        .. warning::

            The :attr:`_orm.Mapper.all_orm_descriptors`
            accessor namespace is an
            instance of :class:`.OrderedProperties`.  This is
            a dictionary-like object which includes a small number of
            named methods such as :meth:`.OrderedProperties.items`
            and :meth:`.OrderedProperties.values`.  When
            accessing attributes dynamically, favor using the dict-access
            scheme, e.g. ``mapper.all_orm_descriptors[somename]`` over
            ``getattr(mapper.all_orm_descriptors, somename)`` to avoid name
            collisions.

        .. seealso::

            :attr:`_orm.Mapper.attrs`

        """
        return util.ReadOnlyProperties(
            dict(self.class_manager._all_sqla_attributes())
        )

    @HasMemoized.memoized_attribute
    @util.preload_module("sqlalchemy.orm.descriptor_props")
    def _pk_synonyms(self) -> Dict[str, str]:
        """return a dictionary of {syn_attribute_name: pk_attr_name} for
        all synonyms that refer to primary key columns

        """
        descriptor_props = util.preloaded.orm_descriptor_props

        pk_keys = {prop.key for prop in self._identity_key_props}

        return {
            syn.key: syn.name
            for k, syn in self._props.items()
            if isinstance(syn, descriptor_props.SynonymProperty)
            and syn.name in pk_keys
        }

    @HasMemoized.memoized_attribute
    @util.preload_module("sqlalchemy.orm.descriptor_props")
    def synonyms(self) -> util.ReadOnlyProperties[SynonymProperty[Any]]:
        """Return a namespace of all :class:`.Synonym`
        properties maintained by this :class:`_orm.Mapper`.

        .. seealso::

            :attr:`_orm.Mapper.attrs` - namespace of all
            :class:`.MapperProperty`
            objects.

        """
        descriptor_props = util.preloaded.orm_descriptor_props

        return self._filter_properties(descriptor_props.SynonymProperty)

    @property
    def entity_namespace(self):
        return self.class_

    @HasMemoized.memoized_attribute
    def column_attrs(self) -> util.ReadOnlyProperties[ColumnProperty[Any]]:
        """Return a namespace of all :class:`.ColumnProperty`
        properties maintained by this :class:`_orm.Mapper`.

        .. seealso::

            :attr:`_orm.Mapper.attrs` - namespace of all
            :class:`.MapperProperty`
            objects.

        """
        return self._filter_properties(properties.ColumnProperty)

    @HasMemoized.memoized_attribute
    @util.preload_module("sqlalchemy.orm.relationships")
    def relationships(
        self,
    ) -> util.ReadOnlyProperties[RelationshipProperty[Any]]:
        """A namespace of all :class:`.Relationship` properties
        maintained by this :class:`_orm.Mapper`.

        .. warning::

            the :attr:`_orm.Mapper.relationships` accessor namespace is an
            instance of :class:`.OrderedProperties`.  This is
            a dictionary-like object which includes a small number of
            named methods such as :meth:`.OrderedProperties.items`
            and :meth:`.OrderedProperties.values`.  When
            accessing attributes dynamically, favor using the dict-access
            scheme, e.g. ``mapper.relationships[somename]`` over
            ``getattr(mapper.relationships, somename)`` to avoid name
            collisions.

        .. seealso::

            :attr:`_orm.Mapper.attrs` - namespace of all
            :class:`.MapperProperty`
            objects.

        """
        return self._filter_properties(
            util.preloaded.orm_relationships.RelationshipProperty
        )

    @HasMemoized.memoized_attribute
    @util.preload_module("sqlalchemy.orm.descriptor_props")
    def composites(self) -> util.ReadOnlyProperties[CompositeProperty[Any]]:
        """Return a namespace of all :class:`.Composite`
        properties maintained by this :class:`_orm.Mapper`.

        .. seealso::

            :attr:`_orm.Mapper.attrs` - namespace of all
            :class:`.MapperProperty`
            objects.

        """
        return self._filter_properties(
            util.preloaded.orm_descriptor_props.CompositeProperty
        )

    def _filter_properties(
        self, type_: Type[_MP]
    ) -> util.ReadOnlyProperties[_MP]:
        self._check_configure()
        return util.ReadOnlyProperties(
            util.OrderedDict(
                (k, v) for k, v in self._props.items() if isinstance(v, type_)
            )
        )

    @HasMemoized.memoized_attribute
    def _get_clause(self):
        """create a "get clause" based on the primary key.  this is used
        by query.get() and many-to-one lazyloads to load this item
        by primary key.

        """
        params = [
            (
                primary_key,
                sql.bindparam("pk_%d" % idx, type_=primary_key.type),
            )
            for idx, primary_key in enumerate(self.primary_key, 1)
        ]
        return (
            sql.and_(*[k == v for (k, v) in params]),
            util.column_dict(params),
        )

    @HasMemoized.memoized_attribute
    def _equivalent_columns(self) -> _EquivalentColumnMap:
        """Create a map of all equivalent columns, based on
        the determination of column pairs that are equated to
        one another based on inherit condition.  This is designed
        to work with the queries that util.polymorphic_union
        comes up with, which often don't include the columns from
        the base table directly (including the subclass table columns
        only).

        The resulting structure is a dictionary of columns mapped
        to lists of equivalent columns, e.g.::

            {tablea.col1: {tableb.col1, tablec.col1}, tablea.col2: {tabled.col2}}

        """  # noqa: E501
        result: _EquivalentColumnMap = {}

        def visit_binary(binary):
            if binary.operator == operators.eq:
                if binary.left in result:
                    result[binary.left].add(binary.right)
                else:
                    result[binary.left] = {binary.right}
                if binary.right in result:
                    result[binary.right].add(binary.left)
                else:
                    result[binary.right] = {binary.left}

        for mapper in self.base_mapper.self_and_descendants:
            if mapper.inherit_condition is not None:
                visitors.traverse(
                    mapper.inherit_condition, {}, {"binary": visit_binary}
                )

        return result

    def _is_userland_descriptor(self, assigned_name: str, obj: Any) -> bool:
        if isinstance(
            obj,
            (
                _MappedAttribute,
                instrumentation.ClassManager,
                expression.ColumnElement,
            ),
        ):
            return False
        else:
            return assigned_name not in self._dataclass_fields

    @HasMemoized.memoized_attribute
    def _dataclass_fields(self):
        return [f.name for f in util.dataclass_fields(self.class_)]

    def _should_exclude(self, name, assigned_name, local, column):
        """determine whether a particular property should be implicitly
        present on the class.

        This occurs when properties are propagated from an inherited class, or
        are applied from the columns present in the mapped table.

        """

        if column is not None and sql_base._never_select_column(column):
            return True

        # check for class-bound attributes and/or descriptors,
        # either local or from an inherited class
        # ignore dataclass field default values
        if local:
            if self.class_.__dict__.get(
                assigned_name, None
            ) is not None and self._is_userland_descriptor(
                assigned_name, self.class_.__dict__[assigned_name]
            ):
                return True
        else:
            attr = self.class_manager._get_class_attr_mro(assigned_name, None)
            if attr is not None and self._is_userland_descriptor(
                assigned_name, attr
            ):
                return True

        if (
            self.include_properties is not None
            and name not in self.include_properties
            and (column is None or column not in self.include_properties)
        ):
            self._log("not including property %s" % (name))
            return True

        if self.exclude_properties is not None and (
            name in self.exclude_properties
            or (column is not None and column in self.exclude_properties)
        ):
            self._log("excluding property %s" % (name))
            return True

        return False

    def common_parent(self, other: Mapper[Any]) -> bool:
        """Return true if the given mapper shares a
        common inherited parent as this mapper."""

        return self.base_mapper is other.base_mapper

    def is_sibling(self, other: Mapper[Any]) -> bool:
        """return true if the other mapper is an inheriting sibling to this
        one.  common parent but different branch

        """
        return (
            self.base_mapper is other.base_mapper
            and not self.isa(other)
            and not other.isa(self)
        )

    def _canload(
        self, state: InstanceState[Any], allow_subtypes: bool
    ) -> bool:
        s = self.primary_mapper()
        if self.polymorphic_on is not None or allow_subtypes:
            return _state_mapper(state).isa(s)
        else:
            return _state_mapper(state) is s

    def isa(self, other: Mapper[Any]) -> bool:
        """Return True if the this mapper inherits from the given mapper."""

        m: Optional[Mapper[Any]] = self
        while m and m is not other:
            m = m.inherits
        return bool(m)

    def iterate_to_root(self) -> Iterator[Mapper[Any]]:
        m: Optional[Mapper[Any]] = self
        while m:
            yield m
            m = m.inherits

    @HasMemoized.memoized_attribute
    def self_and_descendants(self) -> Sequence[Mapper[Any]]:
        """The collection including this mapper and all descendant mappers.

        This includes not just the immediately inheriting mappers but
        all their inheriting mappers as well.

        """
        descendants = []
        stack = deque([self])
        while stack:
            item = stack.popleft()
            descendants.append(item)
            stack.extend(item._inheriting_mappers)
        return util.WeakSequence(descendants)

    def polymorphic_iterator(self) -> Iterator[Mapper[Any]]:
        """Iterate through the collection including this mapper and
        all descendant mappers.

        This includes not just the immediately inheriting mappers but
        all their inheriting mappers as well.

        To iterate through an entire hierarchy, use
        ``mapper.base_mapper.polymorphic_iterator()``.

        """
        return iter(self.self_and_descendants)

    def primary_mapper(self) -> Mapper[Any]:
        """Return the primary mapper corresponding to this mapper's class key
        (class)."""

        return self.class_manager.mapper

    @property
    def primary_base_mapper(self) -> Mapper[Any]:
        return self.class_manager.mapper.base_mapper

    def _result_has_identity_key(self, result, adapter=None):
        pk_cols: Sequence[ColumnClause[Any]] = self.primary_key
        if adapter:
            pk_cols = [adapter.columns[c] for c in pk_cols]
        rk = result.keys()
        for col in pk_cols:
            if col not in rk:
                return False
        else:
            return True

    def identity_key_from_row(
        self,
        row: Union[Row[Any], RowMapping],
        identity_token: Optional[Any] = None,
        adapter: Optional[ORMAdapter] = None,
    ) -> _IdentityKeyType[_O]:
        """Return an identity-map key for use in storing/retrieving an
        item from the identity map.

        :param row: A :class:`.Row` or :class:`.RowMapping` produced from a
         result set that selected from the ORM mapped primary key columns.

         .. versionchanged:: 2.0
            :class:`.Row` or :class:`.RowMapping` are accepted
            for the "row" argument

        """
        pk_cols: Sequence[ColumnClause[Any]] = self.primary_key
        if adapter:
            pk_cols = [adapter.columns[c] for c in pk_cols]

        mapping: RowMapping
        if hasattr(row, "_mapping"):
            mapping = row._mapping
        else:
            mapping = row  # type: ignore[assignment]

        return (
            self._identity_class,
            tuple(mapping[column] for column in pk_cols),
            identity_token,
        )

    def identity_key_from_primary_key(
        self,
        primary_key: Tuple[Any, ...],
        identity_token: Optional[Any] = None,
    ) -> _IdentityKeyType[_O]:
        """Return an identity-map key for use in storing/retrieving an
        item from an identity map.

        :param primary_key: A list of values indicating the identifier.

        """
        return (
            self._identity_class,
            tuple(primary_key),
            identity_token,
        )

    def identity_key_from_instance(self, instance: _O) -> _IdentityKeyType[_O]:
        """Return the identity key for the given instance, based on
        its primary key attributes.

        If the instance's state is expired, calling this method
        will result in a database check to see if the object has been deleted.
        If the row no longer exists,
        :class:`~sqlalchemy.orm.exc.ObjectDeletedError` is raised.

        This value is typically also found on the instance state under the
        attribute name `key`.

        """
        state = attributes.instance_state(instance)
        return self._identity_key_from_state(state, PassiveFlag.PASSIVE_OFF)

    def _identity_key_from_state(
        self,
        state: InstanceState[_O],
        passive: PassiveFlag = PassiveFlag.PASSIVE_RETURN_NO_VALUE,
    ) -> _IdentityKeyType[_O]:
        dict_ = state.dict
        manager = state.manager
        return (
            self._identity_class,
            tuple(
                [
                    manager[prop.key].impl.get(state, dict_, passive)
                    for prop in self._identity_key_props
                ]
            ),
            state.identity_token,
        )

    def primary_key_from_instance(self, instance: _O) -> Tuple[Any, ...]:
        """Return the list of primary key values for the given
        instance.

        If the instance's state is expired, calling this method
        will result in a database check to see if the object has been deleted.
        If the row no longer exists,
        :class:`~sqlalchemy.orm.exc.ObjectDeletedError` is raised.

        """
        state = attributes.instance_state(instance)
        identity_key = self._identity_key_from_state(
            state, PassiveFlag.PASSIVE_OFF
        )
        return identity_key[1]

    @HasMemoized.memoized_attribute
    def _persistent_sortkey_fn(self):
        key_fns = [col.type.sort_key_function for col in self.primary_key]

        if set(key_fns).difference([None]):

            def key(state):
                return tuple(
                    key_fn(val) if key_fn is not None else val
                    for key_fn, val in zip(key_fns, state.key[1])
                )

        else:

            def key(state):
                return state.key[1]

        return key

    @HasMemoized.memoized_attribute
    def _identity_key_props(self):
        return [self._columntoproperty[col] for col in self.primary_key]

    @HasMemoized.memoized_attribute
    def _all_pk_cols(self):
        collection: Set[ColumnClause[Any]] = set()
        for table in self.tables:
            collection.update(self._pks_by_table[table])
        return collection

    @HasMemoized.memoized_attribute
    def _should_undefer_in_wildcard(self):
        cols: Set[ColumnElement[Any]] = set(self.primary_key)
        if self.polymorphic_on is not None:
            cols.add(self.polymorphic_on)
        return cols

    @HasMemoized.memoized_attribute
    def _primary_key_propkeys(self):
        return {self._columntoproperty[col].key for col in self._all_pk_cols}

    def _get_state_attr_by_column(
        self,
        state: InstanceState[_O],
        dict_: _InstanceDict,
        column: ColumnElement[Any],
        passive: PassiveFlag = PassiveFlag.PASSIVE_RETURN_NO_VALUE,
    ) -> Any:
        prop = self._columntoproperty[column]
        return state.manager[prop.key].impl.get(state, dict_, passive=passive)

    def _set_committed_state_attr_by_column(self, state, dict_, column, value):
        prop = self._columntoproperty[column]
        state.manager[prop.key].impl.set_committed_value(state, dict_, value)

    def _set_state_attr_by_column(self, state, dict_, column, value):
        prop = self._columntoproperty[column]
        state.manager[prop.key].impl.set(state, dict_, value, None)

    def _get_committed_attr_by_column(self, obj, column):
        state = attributes.instance_state(obj)
        dict_ = attributes.instance_dict(obj)
        return self._get_committed_state_attr_by_column(
            state, dict_, column, passive=PassiveFlag.PASSIVE_OFF
        )

    def _get_committed_state_attr_by_column(
        self, state, dict_, column, passive=PassiveFlag.PASSIVE_RETURN_NO_VALUE
    ):
        prop = self._columntoproperty[column]
        return state.manager[prop.key].impl.get_committed_value(
            state, dict_, passive=passive
        )

    def _optimized_get_statement(self, state, attribute_names):
        """assemble a WHERE clause which retrieves a given state by primary
        key, using a minimized set of tables.

        Applies to a joined-table inheritance mapper where the
        requested attribute names are only present on joined tables,
        not the base table.  The WHERE clause attempts to include
        only those tables to minimize joins.

        """
        props = self._props

        col_attribute_names = set(attribute_names).intersection(
            state.mapper.column_attrs.keys()
        )
        tables: Set[FromClause] = set(
            chain(
                *[
                    sql_util.find_tables(c, check_columns=True)
                    for key in col_attribute_names
                    for c in props[key].columns
                ]
            )
        )

        if self.base_mapper.local_table in tables:
            return None

        def visit_binary(binary):
            leftcol = binary.left
            rightcol = binary.right
            if leftcol is None or rightcol is None:
                return

            if leftcol.table not in tables:
                leftval = self._get_committed_state_attr_by_column(
                    state,
                    state.dict,
                    leftcol,
                    passive=PassiveFlag.PASSIVE_NO_INITIALIZE,
                )
                if leftval in orm_util._none_set:
                    raise _OptGetColumnsNotAvailable()
                binary.left = sql.bindparam(
                    None, leftval, type_=binary.right.type
                )
            elif rightcol.table not in tables:
                rightval = self._get_committed_state_attr_by_column(
                    state,
                    state.dict,
                    rightcol,
                    passive=PassiveFlag.PASSIVE_NO_INITIALIZE,
                )
                if rightval in orm_util._none_set:
                    raise _OptGetColumnsNotAvailable()
                binary.right = sql.bindparam(
                    None, rightval, type_=binary.right.type
                )

        allconds: List[ColumnElement[bool]] = []

        start = False

        # as of #7507, from the lowest base table on upwards,
        # we include all intermediary tables.

        for mapper in reversed(list(self.iterate_to_root())):
            if mapper.local_table in tables:
                start = True
            elif not isinstance(mapper.local_table, expression.TableClause):
                return None
            if start and not mapper.single:
                assert mapper.inherits
                assert not mapper.concrete
                assert mapper.inherit_condition is not None
                allconds.append(mapper.inherit_condition)
                tables.add(mapper.local_table)

        # only the bottom table needs its criteria to be altered to fit
        # the primary key ident - the rest of the tables upwards to the
        # descendant-most class should all be present and joined to each
        # other.
        try:
            _traversed = visitors.cloned_traverse(
                allconds[0], {}, {"binary": visit_binary}
            )
        except _OptGetColumnsNotAvailable:
            return None
        else:
            allconds[0] = _traversed

        cond = sql.and_(*allconds)

        cols = []
        for key in col_attribute_names:
            cols.extend(props[key].columns)
        return (
            sql.select(*cols)
            .where(cond)
            .set_label_style(LABEL_STYLE_TABLENAME_PLUS_COL)
        )

    def _iterate_to_target_viawpoly(self, mapper):
        if self.isa(mapper):
            prev = self
            for m in self.iterate_to_root():
                yield m

                if m is not prev and prev not in m._with_polymorphic_mappers:
                    break

                prev = m
                if m is mapper:
                    break

    @HasMemoized.memoized_attribute
    def _would_selectinload_combinations_cache(self):
        return {}

    def _would_selectin_load_only_from_given_mapper(self, super_mapper):
        """return True if this mapper would "selectin" polymorphic load based
        on the given super mapper, and not from a setting from a subclass.

        given::

            class A: ...


            class B(A):
                __mapper_args__ = {"polymorphic_load": "selectin"}


            class C(B): ...


            class D(B):
                __mapper_args__ = {"polymorphic_load": "selectin"}

        ``inspect(C)._would_selectin_load_only_from_given_mapper(inspect(B))``
        returns True, because C does selectin loading because of B's setting.

        OTOH, ``inspect(D)
        ._would_selectin_load_only_from_given_mapper(inspect(B))``
        returns False, because D does selectin loading because of its own
        setting; when we are doing a selectin poly load from B, we want to
        filter out D because it would already have its own selectin poly load
        set up separately.

        Added as part of #9373.

        """
        cache = self._would_selectinload_combinations_cache

        try:
            return cache[super_mapper]
        except KeyError:
            pass

        # assert that given object is a supermapper, meaning we already
        # strong reference it directly or indirectly.  this allows us
        # to not worry that we are creating new strongrefs to unrelated
        # mappers or other objects.
        assert self.isa(super_mapper)

        mapper = super_mapper
        for m in self._iterate_to_target_viawpoly(mapper):
            if m.polymorphic_load == "selectin":
                retval = m is super_mapper
                break
        else:
            retval = False

        cache[super_mapper] = retval
        return retval

    def _should_selectin_load(self, enabled_via_opt, polymorphic_from):
        if not enabled_via_opt:
            # common case, takes place for all polymorphic loads
            mapper = polymorphic_from
            for m in self._iterate_to_target_viawpoly(mapper):
                if m.polymorphic_load == "selectin":
                    return m
        else:
            # uncommon case, selectin load options were used
            enabled_via_opt = set(enabled_via_opt)
            enabled_via_opt_mappers = {e.mapper: e for e in enabled_via_opt}
            for entity in enabled_via_opt.union([polymorphic_from]):
                mapper = entity.mapper
                for m in self._iterate_to_target_viawpoly(mapper):
                    if (
                        m.polymorphic_load == "selectin"
                        or m in enabled_via_opt_mappers
                    ):
                        return enabled_via_opt_mappers.get(m, m)

        return None

    @util.preload_module("sqlalchemy.orm.strategy_options")
    def _subclass_load_via_in(self, entity, polymorphic_from):
        """Assemble a that can load the columns local to
        this subclass as a SELECT with IN.

        """

        strategy_options = util.preloaded.orm_strategy_options

        assert self.inherits

        if self.polymorphic_on is not None:
            polymorphic_prop = self._columntoproperty[self.polymorphic_on]
            keep_props = set([polymorphic_prop] + self._identity_key_props)
        else:
            keep_props = set(self._identity_key_props)

        disable_opt = strategy_options.Load(entity)
        enable_opt = strategy_options.Load(entity)

        classes_to_include = {self}
        m: Optional[Mapper[Any]] = self.inherits
        while (
            m is not None
            and m is not polymorphic_from
            and m.polymorphic_load == "selectin"
        ):
            classes_to_include.add(m)
            m = m.inherits

        for prop in self.column_attrs + self.relationships:
            # skip prop keys that are not instrumented on the mapped class.
            # this is primarily the "_sa_polymorphic_on" property that gets
            # created for an ad-hoc polymorphic_on SQL expression, issue #8704
            if prop.key not in self.class_manager:
                continue

            if prop.parent in classes_to_include or prop in keep_props:
                # "enable" options, to turn on the properties that we want to
                # load by default (subject to options from the query)
                if not isinstance(prop, StrategizedProperty):
                    continue

                enable_opt = enable_opt._set_generic_strategy(
                    # convert string name to an attribute before passing
                    # to loader strategy.   note this must be in terms
                    # of given entity, such as AliasedClass, etc.
                    (getattr(entity.entity_namespace, prop.key),),
                    dict(prop.strategy_key),
                    _reconcile_to_other=True,
                )
            else:
                # "disable" options, to turn off the properties from the
                # superclass that we *don't* want to load, applied after
                # the options from the query to override them
                disable_opt = disable_opt._set_generic_strategy(
                    # convert string name to an attribute before passing
                    # to loader strategy.   note this must be in terms
                    # of given entity, such as AliasedClass, etc.
                    (getattr(entity.entity_namespace, prop.key),),
                    {"do_nothing": True},
                    _reconcile_to_other=False,
                )

        primary_key = [
            sql_util._deep_annotate(pk, {"_orm_adapt": True})
            for pk in self.primary_key
        ]

        in_expr: ColumnElement[Any]

        if len(primary_key) > 1:
            in_expr = sql.tuple_(*primary_key)
        else:
            in_expr = primary_key[0]

        if entity.is_aliased_class:
            assert entity.mapper is self

            q = sql.select(entity).set_label_style(
                LABEL_STYLE_TABLENAME_PLUS_COL
            )

            in_expr = entity._adapter.traverse(in_expr)
            primary_key = [entity._adapter.traverse(k) for k in primary_key]
            q = q.where(
                in_expr.in_(sql.bindparam("primary_keys", expanding=True))
            ).order_by(*primary_key)
        else:
            q = sql.select(self).set_label_style(
                LABEL_STYLE_TABLENAME_PLUS_COL
            )
            q = q.where(
                in_expr.in_(sql.bindparam("primary_keys", expanding=True))
            ).order_by(*primary_key)

        return q, enable_opt, disable_opt

    @HasMemoized.memoized_attribute
    def _subclass_load_via_in_mapper(self):
        # the default is loading this mapper against the basemost mapper
        return self._subclass_load_via_in(self, self.base_mapper)

    def cascade_iterator(
        self,
        type_: str,
        state: InstanceState[_O],
        halt_on: Optional[Callable[[InstanceState[Any]], bool]] = None,
    ) -> Iterator[
        Tuple[object, Mapper[Any], InstanceState[Any], _InstanceDict]
    ]:
        r"""Iterate each element and its mapper in an object graph,
        for all relationships that meet the given cascade rule.

        :param type\_:
          The name of the cascade rule (i.e. ``"save-update"``, ``"delete"``,
          etc.).

          .. note::  the ``"all"`` cascade is not accepted here.  For a generic
             object traversal function, see :ref:`faq_walk_objects`.

        :param state:
          The lead InstanceState.  child items will be processed per
          the relationships defined for this object's mapper.

        :return: the method yields individual object instances.

        .. seealso::

            :ref:`unitofwork_cascades`

            :ref:`faq_walk_objects` - illustrates a generic function to
            traverse all objects without relying on cascades.

        """
        visited_states: Set[InstanceState[Any]] = set()
        prp, mpp = object(), object()

        assert state.mapper.isa(self)

        # this is actually a recursive structure, fully typing it seems
        # a little too difficult for what it's worth here
        visitables: Deque[
            Tuple[
                Deque[Any],
                object,
                Optional[InstanceState[Any]],
                Optional[_InstanceDict],
            ]
        ]

        visitables = deque(
            [(deque(state.mapper._props.values()), prp, state, state.dict)]
        )

        while visitables:
            iterator, item_type, parent_state, parent_dict = visitables[-1]
            if not iterator:
                visitables.pop()
                continue

            if item_type is prp:
                prop = iterator.popleft()
                if not prop.cascade or type_ not in prop.cascade:
                    continue
                assert parent_state is not None
                assert parent_dict is not None
                queue = deque(
                    prop.cascade_iterator(
                        type_,
                        parent_state,
                        parent_dict,
                        visited_states,
                        halt_on,
                    )
                )
                if queue:
                    visitables.append((queue, mpp, None, None))
            elif item_type is mpp:
                (
                    instance,
                    instance_mapper,
                    corresponding_state,
                    corresponding_dict,
                ) = iterator.popleft()
                yield (
                    instance,
                    instance_mapper,
                    corresponding_state,
                    corresponding_dict,
                )
                visitables.append(
                    (
                        deque(instance_mapper._props.values()),
                        prp,
                        corresponding_state,
                        corresponding_dict,
                    )
                )

    @HasMemoized.memoized_attribute
    def _compiled_cache(self):
        return util.LRUCache(self._compiled_cache_size)

    @HasMemoized.memoized_attribute
    def _multiple_persistence_tables(self):
        return len(self.tables) > 1

    @HasMemoized.memoized_attribute
    def _sorted_tables(self):
        table_to_mapper: Dict[TableClause, Mapper[Any]] = {}

        for mapper in self.base_mapper.self_and_descendants:
            for t in mapper.tables:
                table_to_mapper.setdefault(t, mapper)

        extra_dependencies = []
        for table, mapper in table_to_mapper.items():
            super_ = mapper.inherits
            if super_:
                extra_dependencies.extend(
                    [(super_table, table) for super_table in super_.tables]
                )

        def skip(fk):
            # attempt to skip dependencies that are not
            # significant to the inheritance chain
            # for two tables that are related by inheritance.
            # while that dependency may be important, it's technically
            # not what we mean to sort on here.
            parent = table_to_mapper.get(fk.parent.table)
            dep = table_to_mapper.get(fk.column.table)
            if (
                parent is not None
                and dep is not None
                and dep is not parent
                and dep.inherit_condition is not None
            ):
                cols = set(sql_util._find_columns(dep.inherit_condition))
                if parent.inherit_condition is not None:
                    cols = cols.union(
                        sql_util._find_columns(parent.inherit_condition)
                    )
                    return fk.parent not in cols and fk.column not in cols
                else:
                    return fk.parent not in cols
            return False

        sorted_ = sql_util.sort_tables(
            table_to_mapper,
            skip_fn=skip,
            extra_dependencies=extra_dependencies,
        )

        ret = util.OrderedDict()
        for t in sorted_:
            ret[t] = table_to_mapper[t]
        return ret

    def _memo(self, key: Any, callable_: Callable[[], _T]) -> _T:
        if key in self._memoized_values:
            return cast(_T, self._memoized_values[key])
        else:
            self._memoized_values[key] = value = callable_()
            return value

    @util.memoized_property
    def _table_to_equated(self):
        """memoized map of tables to collections of columns to be
        synchronized upwards to the base mapper."""

        result: util.defaultdict[
            Table,
            List[
                Tuple[
                    Mapper[Any],
                    List[Tuple[ColumnElement[Any], ColumnElement[Any]]],
                ]
            ],
        ] = util.defaultdict(list)

        def set_union(x, y):
            return x.union(y)

        for table in self._sorted_tables:
            cols = set(table.c)

            for m in self.iterate_to_root():
                if m._inherits_equated_pairs and cols.intersection(
                    reduce(
                        set_union,
                        [l.proxy_set for l, r in m._inherits_equated_pairs],
                    )
                ):
                    result[table].append((m, m._inherits_equated_pairs))

        return result


class _OptGetColumnsNotAvailable(Exception):
    pass


def configure_mappers() -> None:
    """Initialize the inter-mapper relationships of all mappers that
    have been constructed thus far across all :class:`_orm.registry`
    collections.

    The configure step is used to reconcile and initialize the
    :func:`_orm.relationship` linkages between mapped classes, as well as to
    invoke configuration events such as the
    :meth:`_orm.MapperEvents.before_configured` and
    :meth:`_orm.MapperEvents.after_configured`, which may be used by ORM
    extensions or user-defined extension hooks.

    Mapper configuration is normally invoked automatically, the first time
    mappings from a particular :class:`_orm.registry` are used, as well as
    whenever mappings are used and additional not-yet-configured mappers have
    been constructed. The automatic configuration process however is local only
    to the :class:`_orm.registry` involving the target mapper and any related
    :class:`_orm.registry` objects which it may depend on; this is
    equivalent to invoking the :meth:`_orm.registry.configure` method
    on a particular :class:`_orm.registry`.

    By contrast, the :func:`_orm.configure_mappers` function will invoke the
    configuration process on all :class:`_orm.registry` objects that
    exist in memory, and may be useful for scenarios where many individual
    :class:`_orm.registry` objects that are nonetheless interrelated are
    in use.

    .. versionchanged:: 1.4

        As of SQLAlchemy 1.4.0b2, this function works on a
        per-:class:`_orm.registry` basis, locating all :class:`_orm.registry`
        objects present and invoking the :meth:`_orm.registry.configure` method
        on each. The :meth:`_orm.registry.configure` method may be preferred to
        limit the configuration of mappers to those local to a particular
        :class:`_orm.registry` and/or declarative base class.

    Points at which automatic configuration is invoked include when a mapped
    class is instantiated into an instance, as well as when ORM queries
    are emitted using :meth:`.Session.query` or :meth:`_orm.Session.execute`
    with an ORM-enabled statement.

    The mapper configure process, whether invoked by
    :func:`_orm.configure_mappers` or from :meth:`_orm.registry.configure`,
    provides several event hooks that can be used to augment the mapper
    configuration step. These hooks include:

    * :meth:`.MapperEvents.before_configured` - called once before
      :func:`.configure_mappers` or :meth:`_orm.registry.configure` does any
      work; this can be used to establish additional options, properties, or
      related mappings before the operation proceeds.

    * :meth:`.MapperEvents.mapper_configured` - called as each individual
      :class:`_orm.Mapper` is configured within the process; will include all
      mapper state except for backrefs set up by other mappers that are still
      to be configured.

    * :meth:`.MapperEvents.after_configured` - called once after
      :func:`.configure_mappers` or :meth:`_orm.registry.configure` is
      complete; at this stage, all :class:`_orm.Mapper` objects that fall
      within the scope of the configuration operation will be fully configured.
      Note that the calling application may still have other mappings that
      haven't been produced yet, such as if they are in modules as yet
      unimported, and may also have mappings that are still to be configured,
      if they are in other :class:`_orm.registry` collections not part of the
      current scope of configuration.

    """

    _configure_registries(_all_registries(), cascade=True)


def _configure_registries(
    registries: Set[_RegistryType], cascade: bool
) -> None:
    for reg in registries:
        if reg._new_mappers:
            break
    else:
        return

    with _CONFIGURE_MUTEX:
        global _already_compiling
        if _already_compiling:
            return
        _already_compiling = True
        try:
            # double-check inside mutex
            for reg in registries:
                if reg._new_mappers:
                    break
            else:
                return

            Mapper.dispatch._for_class(Mapper).before_configured()  # type: ignore # noqa: E501
            # initialize properties on all mappers
            # note that _mapper_registry is unordered, which
            # may randomly conceal/reveal issues related to
            # the order of mapper compilation

            _do_configure_registries(registries, cascade)
        finally:
            _already_compiling = False
    Mapper.dispatch._for_class(Mapper).after_configured()  # type: ignore


@util.preload_module("sqlalchemy.orm.decl_api")
def _do_configure_registries(
    registries: Set[_RegistryType], cascade: bool
) -> None:
    registry = util.preloaded.orm_decl_api.registry

    orig = set(registries)

    for reg in registry._recurse_with_dependencies(registries):
        has_skip = False

        for mapper in reg._mappers_to_configure():
            run_configure = None

            for fn in mapper.dispatch.before_mapper_configured:
                run_configure = fn(mapper, mapper.class_)
                if run_configure is EXT_SKIP:
                    has_skip = True
                    break
            if run_configure is EXT_SKIP:
                continue

            if getattr(mapper, "_configure_failed", False):
                e = sa_exc.InvalidRequestError(
                    "One or more mappers failed to initialize - "
                    "can't proceed with initialization of other "
                    "mappers. Triggering mapper: '%s'. "
                    "Original exception was: %s"
                    % (mapper, mapper._configure_failed)
                )
                e._configure_failed = mapper._configure_failed  # type: ignore
                raise e

            if not mapper.configured:
                try:
                    mapper._post_configure_properties()
                    mapper._expire_memoizations()
                    mapper.dispatch.mapper_configured(mapper, mapper.class_)
                except Exception:
                    exc = sys.exc_info()[1]
                    if not hasattr(exc, "_configure_failed"):
                        mapper._configure_failed = exc
                    raise
        if not has_skip:
            reg._new_mappers = False

        if not cascade and reg._dependencies.difference(orig):
            raise sa_exc.InvalidRequestError(
                "configure was called with cascade=False but "
                "additional registries remain"
            )


@util.preload_module("sqlalchemy.orm.decl_api")
def _dispose_registries(registries: Set[_RegistryType], cascade: bool) -> None:
    registry = util.preloaded.orm_decl_api.registry

    orig = set(registries)

    for reg in registry._recurse_with_dependents(registries):
        if not cascade and reg._dependents.difference(orig):
            raise sa_exc.InvalidRequestError(
                "Registry has dependent registries that are not disposed; "
                "pass cascade=True to clear these also"
            )

        while reg._managers:
            try:
                manager, _ = reg._managers.popitem()
            except KeyError:
                # guard against race between while and popitem
                pass
            else:
                reg._dispose_manager_and_mapper(manager)

        reg._non_primary_mappers.clear()
        reg._dependents.clear()
        for dep in reg._dependencies:
            dep._dependents.discard(reg)
        reg._dependencies.clear()
        # this wasn't done in the 1.3 clear_mappers() and in fact it
        # was a bug, as it could cause configure_mappers() to invoke
        # the "before_configured" event even though mappers had all been
        # disposed.
        reg._new_mappers = False


def reconstructor(fn: _Fn) -> _Fn:
    """Decorate a method as the 'reconstructor' hook.

    Designates a single method as the "reconstructor", an ``__init__``-like
    method that will be called by the ORM after the instance has been
    loaded from the database or otherwise reconstituted.

    .. tip::

        The :func:`_orm.reconstructor` decorator makes use of the
        :meth:`_orm.InstanceEvents.load` event hook, which can be
        used directly.

    The reconstructor will be invoked with no arguments.  Scalar
    (non-collection) database-mapped attributes of the instance will
    be available for use within the function.  Eagerly-loaded
    collections are generally not yet available and will usually only
    contain the first element.  ORM state changes made to objects at
    this stage will not be recorded for the next flush() operation, so
    the activity within a reconstructor should be conservative.

    .. seealso::

        :meth:`.InstanceEvents.load`

    """
    fn.__sa_reconstructor__ = True  # type: ignore[attr-defined]
    return fn


def validates(
    *names: str, include_removes: bool = False, include_backrefs: bool = True
) -> Callable[[_Fn], _Fn]:
    r"""Decorate a method as a 'validator' for one or more named properties.

    Designates a method as a validator, a method which receives the
    name of the attribute as well as a value to be assigned, or in the
    case of a collection, the value to be added to the collection.
    The function can then raise validation exceptions to halt the
    process from continuing (where Python's built-in ``ValueError``
    and ``AssertionError`` exceptions are reasonable choices), or can
    modify or replace the value before proceeding. The function should
    otherwise return the given value.

    Note that a validator for a collection **cannot** issue a load of that
    collection within the validation routine - this usage raises
    an assertion to avoid recursion overflows.  This is a reentrant
    condition which is not supported.

    :param \*names: list of attribute names to be validated.
    :param include_removes: if True, "remove" events will be
     sent as well - the validation function must accept an additional
     argument "is_remove" which will be a boolean.

    :param include_backrefs: defaults to ``True``; if ``False``, the
     validation function will not emit if the originator is an attribute
     event related via a backref.  This can be used for bi-directional
     :func:`.validates` usage where only one validator should emit per
     attribute operation.

     .. versionchanged:: 2.0.16 This paramter inadvertently defaulted to
        ``False`` for releases 2.0.0 through 2.0.15.  Its correct default
        of ``True`` is restored in 2.0.16.

    .. seealso::

      :ref:`simple_validators` - usage examples for :func:`.validates`

    """

    def wrap(fn: _Fn) -> _Fn:
        fn.__sa_validators__ = names  # type: ignore[attr-defined]
        fn.__sa_validation_opts__ = {  # type: ignore[attr-defined]
            "include_removes": include_removes,
            "include_backrefs": include_backrefs,
        }
        return fn

    return wrap


def _event_on_load(state, ctx):
    instrumenting_mapper = state.manager.mapper

    if instrumenting_mapper._reconstructor:
        instrumenting_mapper._reconstructor(state.obj())


def _event_on_init(state, args, kwargs):
    """Run init_instance hooks.

    This also includes mapper compilation, normally not needed
    here but helps with some piecemeal configuration
    scenarios (such as in the ORM tutorial).

    """

    instrumenting_mapper = state.manager.mapper
    if instrumenting_mapper:
        instrumenting_mapper._check_configure()
        if instrumenting_mapper._set_polymorphic_identity:
            instrumenting_mapper._set_polymorphic_identity(state)


class _ColumnMapping(Dict["ColumnElement[Any]", "MapperProperty[Any]"]):
    """Error reporting helper for mapper._columntoproperty."""

    __slots__ = ("mapper",)

    def __init__(self, mapper):
        # TODO: weakref would be a good idea here
        self.mapper = mapper

    def __missing__(self, column):
        prop = self.mapper._props.get(column)
        if prop:
            raise orm_exc.UnmappedColumnError(
                "Column '%s.%s' is not available, due to "
                "conflicting property '%s':%r"
                % (column.table.name, column.name, column.key, prop)
            )
        raise orm_exc.UnmappedColumnError(
            "No column %s is configured on mapper %s..."
            % (column, self.mapper)
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\network.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Network
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import emulation
from . import io
from . import page
from . import runtime
from . import security


class ResourceType(enum.Enum):
    '''
    Resource type as it was perceived by the rendering engine.
    '''
    DOCUMENT = "Document"
    STYLESHEET = "Stylesheet"
    IMAGE = "Image"
    MEDIA = "Media"
    FONT = "Font"
    SCRIPT = "Script"
    TEXT_TRACK = "TextTrack"
    XHR = "XHR"
    FETCH = "Fetch"
    PREFETCH = "Prefetch"
    EVENT_SOURCE = "EventSource"
    WEB_SOCKET = "WebSocket"
    MANIFEST = "Manifest"
    SIGNED_EXCHANGE = "SignedExchange"
    PING = "Ping"
    CSP_VIOLATION_REPORT = "CSPViolationReport"
    PREFLIGHT = "Preflight"
    OTHER = "Other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class LoaderId(str):
    '''
    Unique loader identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> LoaderId:
        return cls(json)

    def __repr__(self):
        return 'LoaderId({})'.format(super().__repr__())


class RequestId(str):
    '''
    Unique network request identifier.
    Note that this does not identify individual HTTP requests that are part of
    a network request.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RequestId:
        return cls(json)

    def __repr__(self):
        return 'RequestId({})'.format(super().__repr__())


class InterceptionId(str):
    '''
    Unique intercepted request identifier.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> InterceptionId:
        return cls(json)

    def __repr__(self):
        return 'InterceptionId({})'.format(super().__repr__())


class ErrorReason(enum.Enum):
    '''
    Network level fetch failure reason.
    '''
    FAILED = "Failed"
    ABORTED = "Aborted"
    TIMED_OUT = "TimedOut"
    ACCESS_DENIED = "AccessDenied"
    CONNECTION_CLOSED = "ConnectionClosed"
    CONNECTION_RESET = "ConnectionReset"
    CONNECTION_REFUSED = "ConnectionRefused"
    CONNECTION_ABORTED = "ConnectionAborted"
    CONNECTION_FAILED = "ConnectionFailed"
    NAME_NOT_RESOLVED = "NameNotResolved"
    INTERNET_DISCONNECTED = "InternetDisconnected"
    ADDRESS_UNREACHABLE = "AddressUnreachable"
    BLOCKED_BY_CLIENT = "BlockedByClient"
    BLOCKED_BY_RESPONSE = "BlockedByResponse"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class TimeSinceEpoch(float):
    '''
    UTC time in seconds, counted from January 1, 1970.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> TimeSinceEpoch:
        return cls(json)

    def __repr__(self):
        return 'TimeSinceEpoch({})'.format(super().__repr__())


class MonotonicTime(float):
    '''
    Monotonically increasing time in seconds since an arbitrary point in the past.
    '''
    def to_json(self) -> float:
        return self

    @classmethod
    def from_json(cls, json: float) -> MonotonicTime:
        return cls(json)

    def __repr__(self):
        return 'MonotonicTime({})'.format(super().__repr__())


class Headers(dict):
    '''
    Request / response headers as keys / values of JSON object.
    '''
    def to_json(self) -> dict:
        return self

    @classmethod
    def from_json(cls, json: dict) -> Headers:
        return cls(json)

    def __repr__(self):
        return 'Headers({})'.format(super().__repr__())


class ConnectionType(enum.Enum):
    '''
    The underlying connection technology that the browser is supposedly using.
    '''
    NONE = "none"
    CELLULAR2G = "cellular2g"
    CELLULAR3G = "cellular3g"
    CELLULAR4G = "cellular4g"
    BLUETOOTH = "bluetooth"
    ETHERNET = "ethernet"
    WIFI = "wifi"
    WIMAX = "wimax"
    OTHER = "other"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieSameSite(enum.Enum):
    '''
    Represents the cookie's 'SameSite' status:
    https://tools.ietf.org/html/draft-west-first-party-cookies
    '''
    STRICT = "Strict"
    LAX = "Lax"
    NONE = "None"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookiePriority(enum.Enum):
    '''
    Represents the cookie's 'Priority' status:
    https://tools.ietf.org/html/draft-west-cookie-priority-00
    '''
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieSourceScheme(enum.Enum):
    '''
    Represents the source scheme of the origin that originally set the cookie.
    A value of "Unset" allows protocol clients to emulate legacy cookie scope for the scheme.
    This is a temporary ability and it will be removed in the future.
    '''
    UNSET = "Unset"
    NON_SECURE = "NonSecure"
    SECURE = "Secure"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ResourceTiming:
    '''
    Timing information for the request.
    '''
    #: Timing's requestTime is a baseline in seconds, while the other numbers are ticks in
    #: milliseconds relatively to this requestTime.
    request_time: float

    #: Started resolving proxy.
    proxy_start: float

    #: Finished resolving proxy.
    proxy_end: float

    #: Started DNS address resolve.
    dns_start: float

    #: Finished DNS address resolve.
    dns_end: float

    #: Started connecting to the remote host.
    connect_start: float

    #: Connected to the remote host.
    connect_end: float

    #: Started SSL handshake.
    ssl_start: float

    #: Finished SSL handshake.
    ssl_end: float

    #: Started running ServiceWorker.
    worker_start: float

    #: Finished Starting ServiceWorker.
    worker_ready: float

    #: Started fetch event.
    worker_fetch_start: float

    #: Settled fetch event respondWith promise.
    worker_respond_with_settled: float

    #: Started sending request.
    send_start: float

    #: Finished sending request.
    send_end: float

    #: Time the server started pushing request.
    push_start: float

    #: Time the server finished pushing request.
    push_end: float

    #: Started receiving response headers.
    receive_headers_start: float

    #: Finished receiving response headers.
    receive_headers_end: float

    #: Started ServiceWorker static routing source evaluation.
    worker_router_evaluation_start: typing.Optional[float] = None

    #: Started cache lookup when the source was evaluated to ``cache``.
    worker_cache_lookup_start: typing.Optional[float] = None

    def to_json(self):
        json = dict()
        json['requestTime'] = self.request_time
        json['proxyStart'] = self.proxy_start
        json['proxyEnd'] = self.proxy_end
        json['dnsStart'] = self.dns_start
        json['dnsEnd'] = self.dns_end
        json['connectStart'] = self.connect_start
        json['connectEnd'] = self.connect_end
        json['sslStart'] = self.ssl_start
        json['sslEnd'] = self.ssl_end
        json['workerStart'] = self.worker_start
        json['workerReady'] = self.worker_ready
        json['workerFetchStart'] = self.worker_fetch_start
        json['workerRespondWithSettled'] = self.worker_respond_with_settled
        json['sendStart'] = self.send_start
        json['sendEnd'] = self.send_end
        json['pushStart'] = self.push_start
        json['pushEnd'] = self.push_end
        json['receiveHeadersStart'] = self.receive_headers_start
        json['receiveHeadersEnd'] = self.receive_headers_end
        if self.worker_router_evaluation_start is not None:
            json['workerRouterEvaluationStart'] = self.worker_router_evaluation_start
        if self.worker_cache_lookup_start is not None:
            json['workerCacheLookupStart'] = self.worker_cache_lookup_start
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request_time=float(json['requestTime']),
            proxy_start=float(json['proxyStart']),
            proxy_end=float(json['proxyEnd']),
            dns_start=float(json['dnsStart']),
            dns_end=float(json['dnsEnd']),
            connect_start=float(json['connectStart']),
            connect_end=float(json['connectEnd']),
            ssl_start=float(json['sslStart']),
            ssl_end=float(json['sslEnd']),
            worker_start=float(json['workerStart']),
            worker_ready=float(json['workerReady']),
            worker_fetch_start=float(json['workerFetchStart']),
            worker_respond_with_settled=float(json['workerRespondWithSettled']),
            send_start=float(json['sendStart']),
            send_end=float(json['sendEnd']),
            push_start=float(json['pushStart']),
            push_end=float(json['pushEnd']),
            receive_headers_start=float(json['receiveHeadersStart']),
            receive_headers_end=float(json['receiveHeadersEnd']),
            worker_router_evaluation_start=float(json['workerRouterEvaluationStart']) if 'workerRouterEvaluationStart' in json else None,
            worker_cache_lookup_start=float(json['workerCacheLookupStart']) if 'workerCacheLookupStart' in json else None,
        )


class ResourcePriority(enum.Enum):
    '''
    Loading priority of a resource request.
    '''
    VERY_LOW = "VeryLow"
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "VeryHigh"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PostDataEntry:
    '''
    Post data entry for HTTP request
    '''
    bytes_: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        if self.bytes_ is not None:
            json['bytes'] = self.bytes_
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            bytes_=str(json['bytes']) if 'bytes' in json else None,
        )


@dataclass
class Request:
    '''
    HTTP request data.
    '''
    #: Request URL (without fragment).
    url: str

    #: HTTP request method.
    method: str

    #: HTTP request headers.
    headers: Headers

    #: Priority of the resource request at the time request is sent.
    initial_priority: ResourcePriority

    #: The referrer policy of the request, as defined in https://www.w3.org/TR/referrer-policy/
    referrer_policy: str

    #: Fragment of the requested URL starting with hash, if present.
    url_fragment: typing.Optional[str] = None

    #: HTTP POST request data.
    #: Use postDataEntries instead.
    post_data: typing.Optional[str] = None

    #: True when the request has POST data. Note that postData might still be omitted when this flag is true when the data is too long.
    has_post_data: typing.Optional[bool] = None

    #: Request body elements (post data broken into individual entries).
    post_data_entries: typing.Optional[typing.List[PostDataEntry]] = None

    #: The mixed content type of the request.
    mixed_content_type: typing.Optional[security.MixedContentType] = None

    #: Whether is loaded via link preload.
    is_link_preload: typing.Optional[bool] = None

    #: Set for requests when the TrustToken API is used. Contains the parameters
    #: passed by the developer (e.g. via "fetch") as understood by the backend.
    trust_token_params: typing.Optional[TrustTokenParams] = None

    #: True if this resource request is considered to be the 'same site' as the
    #: request corresponding to the main frame.
    is_same_site: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['method'] = self.method
        json['headers'] = self.headers.to_json()
        json['initialPriority'] = self.initial_priority.to_json()
        json['referrerPolicy'] = self.referrer_policy
        if self.url_fragment is not None:
            json['urlFragment'] = self.url_fragment
        if self.post_data is not None:
            json['postData'] = self.post_data
        if self.has_post_data is not None:
            json['hasPostData'] = self.has_post_data
        if self.post_data_entries is not None:
            json['postDataEntries'] = [i.to_json() for i in self.post_data_entries]
        if self.mixed_content_type is not None:
            json['mixedContentType'] = self.mixed_content_type.to_json()
        if self.is_link_preload is not None:
            json['isLinkPreload'] = self.is_link_preload
        if self.trust_token_params is not None:
            json['trustTokenParams'] = self.trust_token_params.to_json()
        if self.is_same_site is not None:
            json['isSameSite'] = self.is_same_site
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            method=str(json['method']),
            headers=Headers.from_json(json['headers']),
            initial_priority=ResourcePriority.from_json(json['initialPriority']),
            referrer_policy=str(json['referrerPolicy']),
            url_fragment=str(json['urlFragment']) if 'urlFragment' in json else None,
            post_data=str(json['postData']) if 'postData' in json else None,
            has_post_data=bool(json['hasPostData']) if 'hasPostData' in json else None,
            post_data_entries=[PostDataEntry.from_json(i) for i in json['postDataEntries']] if 'postDataEntries' in json else None,
            mixed_content_type=security.MixedContentType.from_json(json['mixedContentType']) if 'mixedContentType' in json else None,
            is_link_preload=bool(json['isLinkPreload']) if 'isLinkPreload' in json else None,
            trust_token_params=TrustTokenParams.from_json(json['trustTokenParams']) if 'trustTokenParams' in json else None,
            is_same_site=bool(json['isSameSite']) if 'isSameSite' in json else None,
        )


@dataclass
class SignedCertificateTimestamp:
    '''
    Details of a signed certificate timestamp (SCT).
    '''
    #: Validation status.
    status: str

    #: Origin.
    origin: str

    #: Log name / description.
    log_description: str

    #: Log ID.
    log_id: str

    #: Issuance date. Unlike TimeSinceEpoch, this contains the number of
    #: milliseconds since January 1, 1970, UTC, not the number of seconds.
    timestamp: float

    #: Hash algorithm.
    hash_algorithm: str

    #: Signature algorithm.
    signature_algorithm: str

    #: Signature data.
    signature_data: str

    def to_json(self):
        json = dict()
        json['status'] = self.status
        json['origin'] = self.origin
        json['logDescription'] = self.log_description
        json['logId'] = self.log_id
        json['timestamp'] = self.timestamp
        json['hashAlgorithm'] = self.hash_algorithm
        json['signatureAlgorithm'] = self.signature_algorithm
        json['signatureData'] = self.signature_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            status=str(json['status']),
            origin=str(json['origin']),
            log_description=str(json['logDescription']),
            log_id=str(json['logId']),
            timestamp=float(json['timestamp']),
            hash_algorithm=str(json['hashAlgorithm']),
            signature_algorithm=str(json['signatureAlgorithm']),
            signature_data=str(json['signatureData']),
        )


@dataclass
class SecurityDetails:
    '''
    Security details about a request.
    '''
    #: Protocol name (e.g. "TLS 1.2" or "QUIC").
    protocol: str

    #: Key Exchange used by the connection, or the empty string if not applicable.
    key_exchange: str

    #: Cipher name.
    cipher: str

    #: Certificate ID value.
    certificate_id: security.CertificateId

    #: Certificate subject name.
    subject_name: str

    #: Subject Alternative Name (SAN) DNS names and IP addresses.
    san_list: typing.List[str]

    #: Name of the issuing CA.
    issuer: str

    #: Certificate valid from date.
    valid_from: TimeSinceEpoch

    #: Certificate valid to (expiration) date
    valid_to: TimeSinceEpoch

    #: List of signed certificate timestamps (SCTs).
    signed_certificate_timestamp_list: typing.List[SignedCertificateTimestamp]

    #: Whether the request complied with Certificate Transparency policy
    certificate_transparency_compliance: CertificateTransparencyCompliance

    #: Whether the connection used Encrypted ClientHello
    encrypted_client_hello: bool

    #: (EC)DH group used by the connection, if applicable.
    key_exchange_group: typing.Optional[str] = None

    #: TLS MAC. Note that AEAD ciphers do not have separate MACs.
    mac: typing.Optional[str] = None

    #: The signature algorithm used by the server in the TLS server signature,
    #: represented as a TLS SignatureScheme code point. Omitted if not
    #: applicable or not known.
    server_signature_algorithm: typing.Optional[int] = None

    def to_json(self):
        json = dict()
        json['protocol'] = self.protocol
        json['keyExchange'] = self.key_exchange
        json['cipher'] = self.cipher
        json['certificateId'] = self.certificate_id.to_json()
        json['subjectName'] = self.subject_name
        json['sanList'] = [i for i in self.san_list]
        json['issuer'] = self.issuer
        json['validFrom'] = self.valid_from.to_json()
        json['validTo'] = self.valid_to.to_json()
        json['signedCertificateTimestampList'] = [i.to_json() for i in self.signed_certificate_timestamp_list]
        json['certificateTransparencyCompliance'] = self.certificate_transparency_compliance.to_json()
        json['encryptedClientHello'] = self.encrypted_client_hello
        if self.key_exchange_group is not None:
            json['keyExchangeGroup'] = self.key_exchange_group
        if self.mac is not None:
            json['mac'] = self.mac
        if self.server_signature_algorithm is not None:
            json['serverSignatureAlgorithm'] = self.server_signature_algorithm
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            protocol=str(json['protocol']),
            key_exchange=str(json['keyExchange']),
            cipher=str(json['cipher']),
            certificate_id=security.CertificateId.from_json(json['certificateId']),
            subject_name=str(json['subjectName']),
            san_list=[str(i) for i in json['sanList']],
            issuer=str(json['issuer']),
            valid_from=TimeSinceEpoch.from_json(json['validFrom']),
            valid_to=TimeSinceEpoch.from_json(json['validTo']),
            signed_certificate_timestamp_list=[SignedCertificateTimestamp.from_json(i) for i in json['signedCertificateTimestampList']],
            certificate_transparency_compliance=CertificateTransparencyCompliance.from_json(json['certificateTransparencyCompliance']),
            encrypted_client_hello=bool(json['encryptedClientHello']),
            key_exchange_group=str(json['keyExchangeGroup']) if 'keyExchangeGroup' in json else None,
            mac=str(json['mac']) if 'mac' in json else None,
            server_signature_algorithm=int(json['serverSignatureAlgorithm']) if 'serverSignatureAlgorithm' in json else None,
        )


class CertificateTransparencyCompliance(enum.Enum):
    '''
    Whether the request complied with Certificate Transparency policy.
    '''
    UNKNOWN = "unknown"
    NOT_COMPLIANT = "not-compliant"
    COMPLIANT = "compliant"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class BlockedReason(enum.Enum):
    '''
    The reason why request was blocked.
    '''
    OTHER = "other"
    CSP = "csp"
    MIXED_CONTENT = "mixed-content"
    ORIGIN = "origin"
    INSPECTOR = "inspector"
    SUBRESOURCE_FILTER = "subresource-filter"
    CONTENT_TYPE = "content-type"
    COEP_FRAME_RESOURCE_NEEDS_COEP_HEADER = "coep-frame-resource-needs-coep-header"
    COOP_SANDBOXED_IFRAME_CANNOT_NAVIGATE_TO_COOP_PAGE = "coop-sandboxed-iframe-cannot-navigate-to-coop-page"
    CORP_NOT_SAME_ORIGIN = "corp-not-same-origin"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP = "corp-not-same-origin-after-defaulted-to-same-origin-by-coep"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_DIP = "corp-not-same-origin-after-defaulted-to-same-origin-by-dip"
    CORP_NOT_SAME_ORIGIN_AFTER_DEFAULTED_TO_SAME_ORIGIN_BY_COEP_AND_DIP = "corp-not-same-origin-after-defaulted-to-same-origin-by-coep-and-dip"
    CORP_NOT_SAME_SITE = "corp-not-same-site"
    SRI_MESSAGE_SIGNATURE_MISMATCH = "sri-message-signature-mismatch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CorsError(enum.Enum):
    '''
    The reason why request was blocked.
    '''
    DISALLOWED_BY_MODE = "DisallowedByMode"
    INVALID_RESPONSE = "InvalidResponse"
    WILDCARD_ORIGIN_NOT_ALLOWED = "WildcardOriginNotAllowed"
    MISSING_ALLOW_ORIGIN_HEADER = "MissingAllowOriginHeader"
    MULTIPLE_ALLOW_ORIGIN_VALUES = "MultipleAllowOriginValues"
    INVALID_ALLOW_ORIGIN_VALUE = "InvalidAllowOriginValue"
    ALLOW_ORIGIN_MISMATCH = "AllowOriginMismatch"
    INVALID_ALLOW_CREDENTIALS = "InvalidAllowCredentials"
    CORS_DISABLED_SCHEME = "CorsDisabledScheme"
    PREFLIGHT_INVALID_STATUS = "PreflightInvalidStatus"
    PREFLIGHT_DISALLOWED_REDIRECT = "PreflightDisallowedRedirect"
    PREFLIGHT_WILDCARD_ORIGIN_NOT_ALLOWED = "PreflightWildcardOriginNotAllowed"
    PREFLIGHT_MISSING_ALLOW_ORIGIN_HEADER = "PreflightMissingAllowOriginHeader"
    PREFLIGHT_MULTIPLE_ALLOW_ORIGIN_VALUES = "PreflightMultipleAllowOriginValues"
    PREFLIGHT_INVALID_ALLOW_ORIGIN_VALUE = "PreflightInvalidAllowOriginValue"
    PREFLIGHT_ALLOW_ORIGIN_MISMATCH = "PreflightAllowOriginMismatch"
    PREFLIGHT_INVALID_ALLOW_CREDENTIALS = "PreflightInvalidAllowCredentials"
    PREFLIGHT_MISSING_ALLOW_EXTERNAL = "PreflightMissingAllowExternal"
    PREFLIGHT_INVALID_ALLOW_EXTERNAL = "PreflightInvalidAllowExternal"
    PREFLIGHT_MISSING_ALLOW_PRIVATE_NETWORK = "PreflightMissingAllowPrivateNetwork"
    PREFLIGHT_INVALID_ALLOW_PRIVATE_NETWORK = "PreflightInvalidAllowPrivateNetwork"
    INVALID_ALLOW_METHODS_PREFLIGHT_RESPONSE = "InvalidAllowMethodsPreflightResponse"
    INVALID_ALLOW_HEADERS_PREFLIGHT_RESPONSE = "InvalidAllowHeadersPreflightResponse"
    METHOD_DISALLOWED_BY_PREFLIGHT_RESPONSE = "MethodDisallowedByPreflightResponse"
    HEADER_DISALLOWED_BY_PREFLIGHT_RESPONSE = "HeaderDisallowedByPreflightResponse"
    REDIRECT_CONTAINS_CREDENTIALS = "RedirectContainsCredentials"
    INSECURE_PRIVATE_NETWORK = "InsecurePrivateNetwork"
    INVALID_PRIVATE_NETWORK_ACCESS = "InvalidPrivateNetworkAccess"
    UNEXPECTED_PRIVATE_NETWORK_ACCESS = "UnexpectedPrivateNetworkAccess"
    NO_CORS_REDIRECT_MODE_NOT_FOLLOW = "NoCorsRedirectModeNotFollow"
    PREFLIGHT_MISSING_PRIVATE_NETWORK_ACCESS_ID = "PreflightMissingPrivateNetworkAccessId"
    PREFLIGHT_MISSING_PRIVATE_NETWORK_ACCESS_NAME = "PreflightMissingPrivateNetworkAccessName"
    PRIVATE_NETWORK_ACCESS_PERMISSION_UNAVAILABLE = "PrivateNetworkAccessPermissionUnavailable"
    PRIVATE_NETWORK_ACCESS_PERMISSION_DENIED = "PrivateNetworkAccessPermissionDenied"
    LOCAL_NETWORK_ACCESS_PERMISSION_DENIED = "LocalNetworkAccessPermissionDenied"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CorsErrorStatus:
    cors_error: CorsError

    failed_parameter: str

    def to_json(self):
        json = dict()
        json['corsError'] = self.cors_error.to_json()
        json['failedParameter'] = self.failed_parameter
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cors_error=CorsError.from_json(json['corsError']),
            failed_parameter=str(json['failedParameter']),
        )


class ServiceWorkerResponseSource(enum.Enum):
    '''
    Source of serviceworker response.
    '''
    CACHE_STORAGE = "cache-storage"
    HTTP_CACHE = "http-cache"
    FALLBACK_CODE = "fallback-code"
    NETWORK = "network"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class TrustTokenParams:
    '''
    Determines what type of Trust Token operation is executed and
    depending on the type, some additional parameters. The values
    are specified in third_party/blink/renderer/core/fetch/trust_token.idl.
    '''
    operation: TrustTokenOperationType

    #: Only set for "token-redemption" operation and determine whether
    #: to request a fresh SRR or use a still valid cached SRR.
    refresh_policy: str

    #: Origins of issuers from whom to request tokens or redemption
    #: records.
    issuers: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['operation'] = self.operation.to_json()
        json['refreshPolicy'] = self.refresh_policy
        if self.issuers is not None:
            json['issuers'] = [i for i in self.issuers]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            operation=TrustTokenOperationType.from_json(json['operation']),
            refresh_policy=str(json['refreshPolicy']),
            issuers=[str(i) for i in json['issuers']] if 'issuers' in json else None,
        )


class TrustTokenOperationType(enum.Enum):
    ISSUANCE = "Issuance"
    REDEMPTION = "Redemption"
    SIGNING = "Signing"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class AlternateProtocolUsage(enum.Enum):
    '''
    The reason why Chrome uses a specific transport protocol for HTTP semantics.
    '''
    ALTERNATIVE_JOB_WON_WITHOUT_RACE = "alternativeJobWonWithoutRace"
    ALTERNATIVE_JOB_WON_RACE = "alternativeJobWonRace"
    MAIN_JOB_WON_RACE = "mainJobWonRace"
    MAPPING_MISSING = "mappingMissing"
    BROKEN = "broken"
    DNS_ALPN_H3_JOB_WON_WITHOUT_RACE = "dnsAlpnH3JobWonWithoutRace"
    DNS_ALPN_H3_JOB_WON_RACE = "dnsAlpnH3JobWonRace"
    UNSPECIFIED_REASON = "unspecifiedReason"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ServiceWorkerRouterSource(enum.Enum):
    '''
    Source of service worker router.
    '''
    NETWORK = "network"
    CACHE = "cache"
    FETCH_EVENT = "fetch-event"
    RACE_NETWORK_AND_FETCH_HANDLER = "race-network-and-fetch-handler"
    RACE_NETWORK_AND_CACHE = "race-network-and-cache"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ServiceWorkerRouterInfo:
    #: ID of the rule matched. If there is a matched rule, this field will
    #: be set, otherwiser no value will be set.
    rule_id_matched: typing.Optional[int] = None

    #: The router source of the matched rule. If there is a matched rule, this
    #: field will be set, otherwise no value will be set.
    matched_source_type: typing.Optional[ServiceWorkerRouterSource] = None

    #: The actual router source used.
    actual_source_type: typing.Optional[ServiceWorkerRouterSource] = None

    def to_json(self):
        json = dict()
        if self.rule_id_matched is not None:
            json['ruleIdMatched'] = self.rule_id_matched
        if self.matched_source_type is not None:
            json['matchedSourceType'] = self.matched_source_type.to_json()
        if self.actual_source_type is not None:
            json['actualSourceType'] = self.actual_source_type.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            rule_id_matched=int(json['ruleIdMatched']) if 'ruleIdMatched' in json else None,
            matched_source_type=ServiceWorkerRouterSource.from_json(json['matchedSourceType']) if 'matchedSourceType' in json else None,
            actual_source_type=ServiceWorkerRouterSource.from_json(json['actualSourceType']) if 'actualSourceType' in json else None,
        )


@dataclass
class Response:
    '''
    HTTP response data.
    '''
    #: Response URL. This URL can be different from CachedResource.url in case of redirect.
    url: str

    #: HTTP response status code.
    status: int

    #: HTTP response status text.
    status_text: str

    #: HTTP response headers.
    headers: Headers

    #: Resource mimeType as determined by the browser.
    mime_type: str

    #: Resource charset as determined by the browser (if applicable).
    charset: str

    #: Specifies whether physical connection was actually reused for this request.
    connection_reused: bool

    #: Physical connection id that was actually used for this request.
    connection_id: float

    #: Total number of bytes received for this request so far.
    encoded_data_length: float

    #: Security state of the request resource.
    security_state: security.SecurityState

    #: HTTP response headers text. This has been replaced by the headers in Network.responseReceivedExtraInfo.
    headers_text: typing.Optional[str] = None

    #: Refined HTTP request headers that were actually transmitted over the network.
    request_headers: typing.Optional[Headers] = None

    #: HTTP request headers text. This has been replaced by the headers in Network.requestWillBeSentExtraInfo.
    request_headers_text: typing.Optional[str] = None

    #: Remote IP address.
    remote_ip_address: typing.Optional[str] = None

    #: Remote port.
    remote_port: typing.Optional[int] = None

    #: Specifies that the request was served from the disk cache.
    from_disk_cache: typing.Optional[bool] = None

    #: Specifies that the request was served from the ServiceWorker.
    from_service_worker: typing.Optional[bool] = None

    #: Specifies that the request was served from the prefetch cache.
    from_prefetch_cache: typing.Optional[bool] = None

    #: Specifies that the request was served from the prefetch cache.
    from_early_hints: typing.Optional[bool] = None

    #: Information about how ServiceWorker Static Router API was used. If this
    #: field is set with ``matchedSourceType`` field, a matching rule is found.
    #: If this field is set without ``matchedSource``, no matching rule is found.
    #: Otherwise, the API is not used.
    service_worker_router_info: typing.Optional[ServiceWorkerRouterInfo] = None

    #: Timing information for the given request.
    timing: typing.Optional[ResourceTiming] = None

    #: Response source of response from ServiceWorker.
    service_worker_response_source: typing.Optional[ServiceWorkerResponseSource] = None

    #: The time at which the returned response was generated.
    response_time: typing.Optional[TimeSinceEpoch] = None

    #: Cache Storage Cache Name.
    cache_storage_cache_name: typing.Optional[str] = None

    #: Protocol used to fetch this request.
    protocol: typing.Optional[str] = None

    #: The reason why Chrome uses a specific transport protocol for HTTP semantics.
    alternate_protocol_usage: typing.Optional[AlternateProtocolUsage] = None

    #: Security details for the request.
    security_details: typing.Optional[SecurityDetails] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['status'] = self.status
        json['statusText'] = self.status_text
        json['headers'] = self.headers.to_json()
        json['mimeType'] = self.mime_type
        json['charset'] = self.charset
        json['connectionReused'] = self.connection_reused
        json['connectionId'] = self.connection_id
        json['encodedDataLength'] = self.encoded_data_length
        json['securityState'] = self.security_state.to_json()
        if self.headers_text is not None:
            json['headersText'] = self.headers_text
        if self.request_headers is not None:
            json['requestHeaders'] = self.request_headers.to_json()
        if self.request_headers_text is not None:
            json['requestHeadersText'] = self.request_headers_text
        if self.remote_ip_address is not None:
            json['remoteIPAddress'] = self.remote_ip_address
        if self.remote_port is not None:
            json['remotePort'] = self.remote_port
        if self.from_disk_cache is not None:
            json['fromDiskCache'] = self.from_disk_cache
        if self.from_service_worker is not None:
            json['fromServiceWorker'] = self.from_service_worker
        if self.from_prefetch_cache is not None:
            json['fromPrefetchCache'] = self.from_prefetch_cache
        if self.from_early_hints is not None:
            json['fromEarlyHints'] = self.from_early_hints
        if self.service_worker_router_info is not None:
            json['serviceWorkerRouterInfo'] = self.service_worker_router_info.to_json()
        if self.timing is not None:
            json['timing'] = self.timing.to_json()
        if self.service_worker_response_source is not None:
            json['serviceWorkerResponseSource'] = self.service_worker_response_source.to_json()
        if self.response_time is not None:
            json['responseTime'] = self.response_time.to_json()
        if self.cache_storage_cache_name is not None:
            json['cacheStorageCacheName'] = self.cache_storage_cache_name
        if self.protocol is not None:
            json['protocol'] = self.protocol
        if self.alternate_protocol_usage is not None:
            json['alternateProtocolUsage'] = self.alternate_protocol_usage.to_json()
        if self.security_details is not None:
            json['securityDetails'] = self.security_details.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            status=int(json['status']),
            status_text=str(json['statusText']),
            headers=Headers.from_json(json['headers']),
            mime_type=str(json['mimeType']),
            charset=str(json['charset']),
            connection_reused=bool(json['connectionReused']),
            connection_id=float(json['connectionId']),
            encoded_data_length=float(json['encodedDataLength']),
            security_state=security.SecurityState.from_json(json['securityState']),
            headers_text=str(json['headersText']) if 'headersText' in json else None,
            request_headers=Headers.from_json(json['requestHeaders']) if 'requestHeaders' in json else None,
            request_headers_text=str(json['requestHeadersText']) if 'requestHeadersText' in json else None,
            remote_ip_address=str(json['remoteIPAddress']) if 'remoteIPAddress' in json else None,
            remote_port=int(json['remotePort']) if 'remotePort' in json else None,
            from_disk_cache=bool(json['fromDiskCache']) if 'fromDiskCache' in json else None,
            from_service_worker=bool(json['fromServiceWorker']) if 'fromServiceWorker' in json else None,
            from_prefetch_cache=bool(json['fromPrefetchCache']) if 'fromPrefetchCache' in json else None,
            from_early_hints=bool(json['fromEarlyHints']) if 'fromEarlyHints' in json else None,
            service_worker_router_info=ServiceWorkerRouterInfo.from_json(json['serviceWorkerRouterInfo']) if 'serviceWorkerRouterInfo' in json else None,
            timing=ResourceTiming.from_json(json['timing']) if 'timing' in json else None,
            service_worker_response_source=ServiceWorkerResponseSource.from_json(json['serviceWorkerResponseSource']) if 'serviceWorkerResponseSource' in json else None,
            response_time=TimeSinceEpoch.from_json(json['responseTime']) if 'responseTime' in json else None,
            cache_storage_cache_name=str(json['cacheStorageCacheName']) if 'cacheStorageCacheName' in json else None,
            protocol=str(json['protocol']) if 'protocol' in json else None,
            alternate_protocol_usage=AlternateProtocolUsage.from_json(json['alternateProtocolUsage']) if 'alternateProtocolUsage' in json else None,
            security_details=SecurityDetails.from_json(json['securityDetails']) if 'securityDetails' in json else None,
        )


@dataclass
class WebSocketRequest:
    '''
    WebSocket request data.
    '''
    #: HTTP request headers.
    headers: Headers

    def to_json(self):
        json = dict()
        json['headers'] = self.headers.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            headers=Headers.from_json(json['headers']),
        )


@dataclass
class WebSocketResponse:
    '''
    WebSocket response data.
    '''
    #: HTTP response status code.
    status: int

    #: HTTP response status text.
    status_text: str

    #: HTTP response headers.
    headers: Headers

    #: HTTP response headers text.
    headers_text: typing.Optional[str] = None

    #: HTTP request headers.
    request_headers: typing.Optional[Headers] = None

    #: HTTP request headers text.
    request_headers_text: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['status'] = self.status
        json['statusText'] = self.status_text
        json['headers'] = self.headers.to_json()
        if self.headers_text is not None:
            json['headersText'] = self.headers_text
        if self.request_headers is not None:
            json['requestHeaders'] = self.request_headers.to_json()
        if self.request_headers_text is not None:
            json['requestHeadersText'] = self.request_headers_text
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            status=int(json['status']),
            status_text=str(json['statusText']),
            headers=Headers.from_json(json['headers']),
            headers_text=str(json['headersText']) if 'headersText' in json else None,
            request_headers=Headers.from_json(json['requestHeaders']) if 'requestHeaders' in json else None,
            request_headers_text=str(json['requestHeadersText']) if 'requestHeadersText' in json else None,
        )


@dataclass
class WebSocketFrame:
    '''
    WebSocket message data. This represents an entire WebSocket message, not just a fragmented frame as the name suggests.
    '''
    #: WebSocket message opcode.
    opcode: float

    #: WebSocket message mask.
    mask: bool

    #: WebSocket message payload data.
    #: If the opcode is 1, this is a text message and payloadData is a UTF-8 string.
    #: If the opcode isn't 1, then payloadData is a base64 encoded string representing binary data.
    payload_data: str

    def to_json(self):
        json = dict()
        json['opcode'] = self.opcode
        json['mask'] = self.mask
        json['payloadData'] = self.payload_data
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            opcode=float(json['opcode']),
            mask=bool(json['mask']),
            payload_data=str(json['payloadData']),
        )


@dataclass
class CachedResource:
    '''
    Information about the cached resource.
    '''
    #: Resource URL. This is the url of the original network request.
    url: str

    #: Type of this resource.
    type_: ResourceType

    #: Cached response body size.
    body_size: float

    #: Cached response data.
    response: typing.Optional[Response] = None

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['type'] = self.type_.to_json()
        json['bodySize'] = self.body_size
        if self.response is not None:
            json['response'] = self.response.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            type_=ResourceType.from_json(json['type']),
            body_size=float(json['bodySize']),
            response=Response.from_json(json['response']) if 'response' in json else None,
        )


@dataclass
class Initiator:
    '''
    Information about the request initiator.
    '''
    #: Type of this initiator.
    type_: str

    #: Initiator JavaScript stack trace, set for Script only.
    #: Requires the Debugger domain to be enabled.
    stack: typing.Optional[runtime.StackTrace] = None

    #: Initiator URL, set for Parser type or for Script type (when script is importing module) or for SignedExchange type.
    url: typing.Optional[str] = None

    #: Initiator line number, set for Parser type or for Script type (when script is importing
    #: module) (0-based).
    line_number: typing.Optional[float] = None

    #: Initiator column number, set for Parser type or for Script type (when script is importing
    #: module) (0-based).
    column_number: typing.Optional[float] = None

    #: Set if another request triggered this request (e.g. preflight).
    request_id: typing.Optional[RequestId] = None

    def to_json(self):
        json = dict()
        json['type'] = self.type_
        if self.stack is not None:
            json['stack'] = self.stack.to_json()
        if self.url is not None:
            json['url'] = self.url
        if self.line_number is not None:
            json['lineNumber'] = self.line_number
        if self.column_number is not None:
            json['columnNumber'] = self.column_number
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            type_=str(json['type']),
            stack=runtime.StackTrace.from_json(json['stack']) if 'stack' in json else None,
            url=str(json['url']) if 'url' in json else None,
            line_number=float(json['lineNumber']) if 'lineNumber' in json else None,
            column_number=float(json['columnNumber']) if 'columnNumber' in json else None,
            request_id=RequestId.from_json(json['requestId']) if 'requestId' in json else None,
        )


@dataclass
class CookiePartitionKey:
    '''
    cookiePartitionKey object
    The representation of the components of the key that are created by the cookiePartitionKey class contained in net/cookies/cookie_partition_key.h.
    '''
    #: The site of the top-level URL the browser was visiting at the start
    #: of the request to the endpoint that set the cookie.
    top_level_site: str

    #: Indicates if the cookie has any ancestors that are cross-site to the topLevelSite.
    has_cross_site_ancestor: bool

    def to_json(self):
        json = dict()
        json['topLevelSite'] = self.top_level_site
        json['hasCrossSiteAncestor'] = self.has_cross_site_ancestor
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            top_level_site=str(json['topLevelSite']),
            has_cross_site_ancestor=bool(json['hasCrossSiteAncestor']),
        )


@dataclass
class Cookie:
    '''
    Cookie object
    '''
    #: Cookie name.
    name: str

    #: Cookie value.
    value: str

    #: Cookie domain.
    domain: str

    #: Cookie path.
    path: str

    #: Cookie expiration date as the number of seconds since the UNIX epoch.
    expires: float

    #: Cookie size.
    size: int

    #: True if cookie is http-only.
    http_only: bool

    #: True if cookie is secure.
    secure: bool

    #: True in case of session cookie.
    session: bool

    #: Cookie Priority
    priority: CookiePriority

    #: True if cookie is SameParty.
    same_party: bool

    #: Cookie source scheme type.
    source_scheme: CookieSourceScheme

    #: Cookie source port. Valid values are {-1, [1, 65535]}, -1 indicates an unspecified port.
    #: An unspecified port value allows protocol clients to emulate legacy cookie scope for the port.
    #: This is a temporary ability and it will be removed in the future.
    source_port: int

    #: Cookie SameSite type.
    same_site: typing.Optional[CookieSameSite] = None

    #: Cookie partition key.
    partition_key: typing.Optional[CookiePartitionKey] = None

    #: True if cookie partition key is opaque.
    partition_key_opaque: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        json['domain'] = self.domain
        json['path'] = self.path
        json['expires'] = self.expires
        json['size'] = self.size
        json['httpOnly'] = self.http_only
        json['secure'] = self.secure
        json['session'] = self.session
        json['priority'] = self.priority.to_json()
        json['sameParty'] = self.same_party
        json['sourceScheme'] = self.source_scheme.to_json()
        json['sourcePort'] = self.source_port
        if self.same_site is not None:
            json['sameSite'] = self.same_site.to_json()
        if self.partition_key is not None:
            json['partitionKey'] = self.partition_key.to_json()
        if self.partition_key_opaque is not None:
            json['partitionKeyOpaque'] = self.partition_key_opaque
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
            domain=str(json['domain']),
            path=str(json['path']),
            expires=float(json['expires']),
            size=int(json['size']),
            http_only=bool(json['httpOnly']),
            secure=bool(json['secure']),
            session=bool(json['session']),
            priority=CookiePriority.from_json(json['priority']),
            same_party=bool(json['sameParty']),
            source_scheme=CookieSourceScheme.from_json(json['sourceScheme']),
            source_port=int(json['sourcePort']),
            same_site=CookieSameSite.from_json(json['sameSite']) if 'sameSite' in json else None,
            partition_key=CookiePartitionKey.from_json(json['partitionKey']) if 'partitionKey' in json else None,
            partition_key_opaque=bool(json['partitionKeyOpaque']) if 'partitionKeyOpaque' in json else None,
        )


class SetCookieBlockedReason(enum.Enum):
    '''
    Types of reasons why a cookie may not be stored from a response.
    '''
    SECURE_ONLY = "SecureOnly"
    SAME_SITE_STRICT = "SameSiteStrict"
    SAME_SITE_LAX = "SameSiteLax"
    SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "SameSiteUnspecifiedTreatedAsLax"
    SAME_SITE_NONE_INSECURE = "SameSiteNoneInsecure"
    USER_PREFERENCES = "UserPreferences"
    THIRD_PARTY_PHASEOUT = "ThirdPartyPhaseout"
    THIRD_PARTY_BLOCKED_IN_FIRST_PARTY_SET = "ThirdPartyBlockedInFirstPartySet"
    SYNTAX_ERROR = "SyntaxError"
    SCHEME_NOT_SUPPORTED = "SchemeNotSupported"
    OVERWRITE_SECURE = "OverwriteSecure"
    INVALID_DOMAIN = "InvalidDomain"
    INVALID_PREFIX = "InvalidPrefix"
    UNKNOWN_ERROR = "UnknownError"
    SCHEMEFUL_SAME_SITE_STRICT = "SchemefulSameSiteStrict"
    SCHEMEFUL_SAME_SITE_LAX = "SchemefulSameSiteLax"
    SCHEMEFUL_SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "SchemefulSameSiteUnspecifiedTreatedAsLax"
    SAME_PARTY_FROM_CROSS_PARTY_CONTEXT = "SamePartyFromCrossPartyContext"
    SAME_PARTY_CONFLICTS_WITH_OTHER_ATTRIBUTES = "SamePartyConflictsWithOtherAttributes"
    NAME_VALUE_PAIR_EXCEEDS_MAX_SIZE = "NameValuePairExceedsMaxSize"
    DISALLOWED_CHARACTER = "DisallowedCharacter"
    NO_COOKIE_CONTENT = "NoCookieContent"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieBlockedReason(enum.Enum):
    '''
    Types of reasons why a cookie may not be sent with a request.
    '''
    SECURE_ONLY = "SecureOnly"
    NOT_ON_PATH = "NotOnPath"
    DOMAIN_MISMATCH = "DomainMismatch"
    SAME_SITE_STRICT = "SameSiteStrict"
    SAME_SITE_LAX = "SameSiteLax"
    SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "SameSiteUnspecifiedTreatedAsLax"
    SAME_SITE_NONE_INSECURE = "SameSiteNoneInsecure"
    USER_PREFERENCES = "UserPreferences"
    THIRD_PARTY_PHASEOUT = "ThirdPartyPhaseout"
    THIRD_PARTY_BLOCKED_IN_FIRST_PARTY_SET = "ThirdPartyBlockedInFirstPartySet"
    UNKNOWN_ERROR = "UnknownError"
    SCHEMEFUL_SAME_SITE_STRICT = "SchemefulSameSiteStrict"
    SCHEMEFUL_SAME_SITE_LAX = "SchemefulSameSiteLax"
    SCHEMEFUL_SAME_SITE_UNSPECIFIED_TREATED_AS_LAX = "SchemefulSameSiteUnspecifiedTreatedAsLax"
    SAME_PARTY_FROM_CROSS_PARTY_CONTEXT = "SamePartyFromCrossPartyContext"
    NAME_VALUE_PAIR_EXCEEDS_MAX_SIZE = "NameValuePairExceedsMaxSize"
    PORT_MISMATCH = "PortMismatch"
    SCHEME_MISMATCH = "SchemeMismatch"
    ANONYMOUS_CONTEXT = "AnonymousContext"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class CookieExemptionReason(enum.Enum):
    '''
    Types of reasons why a cookie should have been blocked by 3PCD but is exempted for the request.
    '''
    NONE = "None"
    USER_SETTING = "UserSetting"
    TPCD_METADATA = "TPCDMetadata"
    TPCD_DEPRECATION_TRIAL = "TPCDDeprecationTrial"
    TOP_LEVEL_TPCD_DEPRECATION_TRIAL = "TopLevelTPCDDeprecationTrial"
    TPCD_HEURISTICS = "TPCDHeuristics"
    ENTERPRISE_POLICY = "EnterprisePolicy"
    STORAGE_ACCESS = "StorageAccess"
    TOP_LEVEL_STORAGE_ACCESS = "TopLevelStorageAccess"
    SCHEME = "Scheme"
    SAME_SITE_NONE_COOKIES_IN_SANDBOX = "SameSiteNoneCookiesInSandbox"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class BlockedSetCookieWithReason:
    '''
    A cookie which was not stored from a response with the corresponding reason.
    '''
    #: The reason(s) this cookie was blocked.
    blocked_reasons: typing.List[SetCookieBlockedReason]

    #: The string representing this individual cookie as it would appear in the header.
    #: This is not the entire "cookie" or "set-cookie" header which could have multiple cookies.
    cookie_line: str

    #: The cookie object which represents the cookie which was not stored. It is optional because
    #: sometimes complete cookie information is not available, such as in the case of parsing
    #: errors.
    cookie: typing.Optional[Cookie] = None

    def to_json(self):
        json = dict()
        json['blockedReasons'] = [i.to_json() for i in self.blocked_reasons]
        json['cookieLine'] = self.cookie_line
        if self.cookie is not None:
            json['cookie'] = self.cookie.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            blocked_reasons=[SetCookieBlockedReason.from_json(i) for i in json['blockedReasons']],
            cookie_line=str(json['cookieLine']),
            cookie=Cookie.from_json(json['cookie']) if 'cookie' in json else None,
        )


@dataclass
class ExemptedSetCookieWithReason:
    '''
    A cookie should have been blocked by 3PCD but is exempted and stored from a response with the
    corresponding reason. A cookie could only have at most one exemption reason.
    '''
    #: The reason the cookie was exempted.
    exemption_reason: CookieExemptionReason

    #: The string representing this individual cookie as it would appear in the header.
    cookie_line: str

    #: The cookie object representing the cookie.
    cookie: Cookie

    def to_json(self):
        json = dict()
        json['exemptionReason'] = self.exemption_reason.to_json()
        json['cookieLine'] = self.cookie_line
        json['cookie'] = self.cookie.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            exemption_reason=CookieExemptionReason.from_json(json['exemptionReason']),
            cookie_line=str(json['cookieLine']),
            cookie=Cookie.from_json(json['cookie']),
        )


@dataclass
class AssociatedCookie:
    '''
    A cookie associated with the request which may or may not be sent with it.
    Includes the cookies itself and reasons for blocking or exemption.
    '''
    #: The cookie object representing the cookie which was not sent.
    cookie: Cookie

    #: The reason(s) the cookie was blocked. If empty means the cookie is included.
    blocked_reasons: typing.List[CookieBlockedReason]

    #: The reason the cookie should have been blocked by 3PCD but is exempted. A cookie could
    #: only have at most one exemption reason.
    exemption_reason: typing.Optional[CookieExemptionReason] = None

    def to_json(self):
        json = dict()
        json['cookie'] = self.cookie.to_json()
        json['blockedReasons'] = [i.to_json() for i in self.blocked_reasons]
        if self.exemption_reason is not None:
            json['exemptionReason'] = self.exemption_reason.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            cookie=Cookie.from_json(json['cookie']),
            blocked_reasons=[CookieBlockedReason.from_json(i) for i in json['blockedReasons']],
            exemption_reason=CookieExemptionReason.from_json(json['exemptionReason']) if 'exemptionReason' in json else None,
        )


@dataclass
class CookieParam:
    '''
    Cookie parameter object
    '''
    #: Cookie name.
    name: str

    #: Cookie value.
    value: str

    #: The request-URI to associate with the setting of the cookie. This value can affect the
    #: default domain, path, source port, and source scheme values of the created cookie.
    url: typing.Optional[str] = None

    #: Cookie domain.
    domain: typing.Optional[str] = None

    #: Cookie path.
    path: typing.Optional[str] = None

    #: True if cookie is secure.
    secure: typing.Optional[bool] = None

    #: True if cookie is http-only.
    http_only: typing.Optional[bool] = None

    #: Cookie SameSite type.
    same_site: typing.Optional[CookieSameSite] = None

    #: Cookie expiration date, session cookie if not set
    expires: typing.Optional[TimeSinceEpoch] = None

    #: Cookie Priority.
    priority: typing.Optional[CookiePriority] = None

    #: True if cookie is SameParty.
    same_party: typing.Optional[bool] = None

    #: Cookie source scheme type.
    source_scheme: typing.Optional[CookieSourceScheme] = None

    #: Cookie source port. Valid values are {-1, [1, 65535]}, -1 indicates an unspecified port.
    #: An unspecified port value allows protocol clients to emulate legacy cookie scope for the port.
    #: This is a temporary ability and it will be removed in the future.
    source_port: typing.Optional[int] = None

    #: Cookie partition key. If not set, the cookie will be set as not partitioned.
    partition_key: typing.Optional[CookiePartitionKey] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['value'] = self.value
        if self.url is not None:
            json['url'] = self.url
        if self.domain is not None:
            json['domain'] = self.domain
        if self.path is not None:
            json['path'] = self.path
        if self.secure is not None:
            json['secure'] = self.secure
        if self.http_only is not None:
            json['httpOnly'] = self.http_only
        if self.same_site is not None:
            json['sameSite'] = self.same_site.to_json()
        if self.expires is not None:
            json['expires'] = self.expires.to_json()
        if self.priority is not None:
            json['priority'] = self.priority.to_json()
        if self.same_party is not None:
            json['sameParty'] = self.same_party
        if self.source_scheme is not None:
            json['sourceScheme'] = self.source_scheme.to_json()
        if self.source_port is not None:
            json['sourcePort'] = self.source_port
        if self.partition_key is not None:
            json['partitionKey'] = self.partition_key.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            value=str(json['value']),
            url=str(json['url']) if 'url' in json else None,
            domain=str(json['domain']) if 'domain' in json else None,
            path=str(json['path']) if 'path' in json else None,
            secure=bool(json['secure']) if 'secure' in json else None,
            http_only=bool(json['httpOnly']) if 'httpOnly' in json else None,
            same_site=CookieSameSite.from_json(json['sameSite']) if 'sameSite' in json else None,
            expires=TimeSinceEpoch.from_json(json['expires']) if 'expires' in json else None,
            priority=CookiePriority.from_json(json['priority']) if 'priority' in json else None,
            same_party=bool(json['sameParty']) if 'sameParty' in json else None,
            source_scheme=CookieSourceScheme.from_json(json['sourceScheme']) if 'sourceScheme' in json else None,
            source_port=int(json['sourcePort']) if 'sourcePort' in json else None,
            partition_key=CookiePartitionKey.from_json(json['partitionKey']) if 'partitionKey' in json else None,
        )


@dataclass
class AuthChallenge:
    '''
    Authorization challenge for HTTP status code 401 or 407.
    '''
    #: Origin of the challenger.
    origin: str

    #: The authentication scheme used, such as basic or digest
    scheme: str

    #: The realm of the challenge. May be empty.
    realm: str

    #: Source of the authentication challenge.
    source: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['origin'] = self.origin
        json['scheme'] = self.scheme
        json['realm'] = self.realm
        if self.source is not None:
            json['source'] = self.source
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            origin=str(json['origin']),
            scheme=str(json['scheme']),
            realm=str(json['realm']),
            source=str(json['source']) if 'source' in json else None,
        )


@dataclass
class AuthChallengeResponse:
    '''
    Response to an AuthChallenge.
    '''
    #: The decision on what to do in response to the authorization challenge.  Default means
    #: deferring to the default behavior of the net stack, which will likely either the Cancel
    #: authentication or display a popup dialog box.
    response: str

    #: The username to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    username: typing.Optional[str] = None

    #: The password to provide, possibly empty. Should only be set if response is
    #: ProvideCredentials.
    password: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['response'] = self.response
        if self.username is not None:
            json['username'] = self.username
        if self.password is not None:
            json['password'] = self.password
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            response=str(json['response']),
            username=str(json['username']) if 'username' in json else None,
            password=str(json['password']) if 'password' in json else None,
        )


class InterceptionStage(enum.Enum):
    '''
    Stages of the interception to begin intercepting. Request will intercept before the request is
    sent. Response will intercept after the response is received.
    '''
    REQUEST = "Request"
    HEADERS_RECEIVED = "HeadersReceived"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class RequestPattern:
    '''
    Request pattern for interception.
    '''
    #: Wildcards (``'*'`` -> zero or more, ``'?'`` -> exactly one) are allowed. Escape character is
    #: backslash. Omitting is equivalent to ``"*"``.
    url_pattern: typing.Optional[str] = None

    #: If set, only requests for matching resource types will be intercepted.
    resource_type: typing.Optional[ResourceType] = None

    #: Stage at which to begin intercepting requests. Default is Request.
    interception_stage: typing.Optional[InterceptionStage] = None

    def to_json(self):
        json = dict()
        if self.url_pattern is not None:
            json['urlPattern'] = self.url_pattern
        if self.resource_type is not None:
            json['resourceType'] = self.resource_type.to_json()
        if self.interception_stage is not None:
            json['interceptionStage'] = self.interception_stage.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url_pattern=str(json['urlPattern']) if 'urlPattern' in json else None,
            resource_type=ResourceType.from_json(json['resourceType']) if 'resourceType' in json else None,
            interception_stage=InterceptionStage.from_json(json['interceptionStage']) if 'interceptionStage' in json else None,
        )


@dataclass
class SignedExchangeSignature:
    '''
    Information about a signed exchange signature.
    https://wicg.github.io/webpackage/draft-yasskin-httpbis-origin-signed-exchanges-impl.html#rfc.section.3.1
    '''
    #: Signed exchange signature label.
    label: str

    #: The hex string of signed exchange signature.
    signature: str

    #: Signed exchange signature integrity.
    integrity: str

    #: Signed exchange signature validity Url.
    validity_url: str

    #: Signed exchange signature date.
    date: int

    #: Signed exchange signature expires.
    expires: int

    #: Signed exchange signature cert Url.
    cert_url: typing.Optional[str] = None

    #: The hex string of signed exchange signature cert sha256.
    cert_sha256: typing.Optional[str] = None

    #: The encoded certificates.
    certificates: typing.Optional[typing.List[str]] = None

    def to_json(self):
        json = dict()
        json['label'] = self.label
        json['signature'] = self.signature
        json['integrity'] = self.integrity
        json['validityUrl'] = self.validity_url
        json['date'] = self.date
        json['expires'] = self.expires
        if self.cert_url is not None:
            json['certUrl'] = self.cert_url
        if self.cert_sha256 is not None:
            json['certSha256'] = self.cert_sha256
        if self.certificates is not None:
            json['certificates'] = [i for i in self.certificates]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            label=str(json['label']),
            signature=str(json['signature']),
            integrity=str(json['integrity']),
            validity_url=str(json['validityUrl']),
            date=int(json['date']),
            expires=int(json['expires']),
            cert_url=str(json['certUrl']) if 'certUrl' in json else None,
            cert_sha256=str(json['certSha256']) if 'certSha256' in json else None,
            certificates=[str(i) for i in json['certificates']] if 'certificates' in json else None,
        )


@dataclass
class SignedExchangeHeader:
    '''
    Information about a signed exchange header.
    https://wicg.github.io/webpackage/draft-yasskin-httpbis-origin-signed-exchanges-impl.html#cbor-representation
    '''
    #: Signed exchange request URL.
    request_url: str

    #: Signed exchange response code.
    response_code: int

    #: Signed exchange response headers.
    response_headers: Headers

    #: Signed exchange response signature.
    signatures: typing.List[SignedExchangeSignature]

    #: Signed exchange header integrity hash in the form of ``sha256-<base64-hash-value>``.
    header_integrity: str

    def to_json(self):
        json = dict()
        json['requestUrl'] = self.request_url
        json['responseCode'] = self.response_code
        json['responseHeaders'] = self.response_headers.to_json()
        json['signatures'] = [i.to_json() for i in self.signatures]
        json['headerIntegrity'] = self.header_integrity
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request_url=str(json['requestUrl']),
            response_code=int(json['responseCode']),
            response_headers=Headers.from_json(json['responseHeaders']),
            signatures=[SignedExchangeSignature.from_json(i) for i in json['signatures']],
            header_integrity=str(json['headerIntegrity']),
        )


class SignedExchangeErrorField(enum.Enum):
    '''
    Field type for a signed exchange related error.
    '''
    SIGNATURE_SIG = "signatureSig"
    SIGNATURE_INTEGRITY = "signatureIntegrity"
    SIGNATURE_CERT_URL = "signatureCertUrl"
    SIGNATURE_CERT_SHA256 = "signatureCertSha256"
    SIGNATURE_VALIDITY_URL = "signatureValidityUrl"
    SIGNATURE_TIMESTAMPS = "signatureTimestamps"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class SignedExchangeError:
    '''
    Information about a signed exchange response.
    '''
    #: Error message.
    message: str

    #: The index of the signature which caused the error.
    signature_index: typing.Optional[int] = None

    #: The field which caused the error.
    error_field: typing.Optional[SignedExchangeErrorField] = None

    def to_json(self):
        json = dict()
        json['message'] = self.message
        if self.signature_index is not None:
            json['signatureIndex'] = self.signature_index
        if self.error_field is not None:
            json['errorField'] = self.error_field.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            message=str(json['message']),
            signature_index=int(json['signatureIndex']) if 'signatureIndex' in json else None,
            error_field=SignedExchangeErrorField.from_json(json['errorField']) if 'errorField' in json else None,
        )


@dataclass
class SignedExchangeInfo:
    '''
    Information about a signed exchange response.
    '''
    #: The outer response of signed HTTP exchange which was received from network.
    outer_response: Response

    #: Information about the signed exchange header.
    header: typing.Optional[SignedExchangeHeader] = None

    #: Security details for the signed exchange header.
    security_details: typing.Optional[SecurityDetails] = None

    #: Errors occurred while handling the signed exchange.
    errors: typing.Optional[typing.List[SignedExchangeError]] = None

    def to_json(self):
        json = dict()
        json['outerResponse'] = self.outer_response.to_json()
        if self.header is not None:
            json['header'] = self.header.to_json()
        if self.security_details is not None:
            json['securityDetails'] = self.security_details.to_json()
        if self.errors is not None:
            json['errors'] = [i.to_json() for i in self.errors]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            outer_response=Response.from_json(json['outerResponse']),
            header=SignedExchangeHeader.from_json(json['header']) if 'header' in json else None,
            security_details=SecurityDetails.from_json(json['securityDetails']) if 'securityDetails' in json else None,
            errors=[SignedExchangeError.from_json(i) for i in json['errors']] if 'errors' in json else None,
        )


class ContentEncoding(enum.Enum):
    '''
    List of content encodings supported by the backend.
    '''
    DEFLATE = "deflate"
    GZIP = "gzip"
    BR = "br"
    ZSTD = "zstd"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class DirectSocketDnsQueryType(enum.Enum):
    IPV4 = "ipv4"
    IPV6 = "ipv6"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class DirectTCPSocketOptions:
    #: TCP_NODELAY option
    no_delay: bool

    #: Expected to be unsigned integer.
    keep_alive_delay: typing.Optional[float] = None

    #: Expected to be unsigned integer.
    send_buffer_size: typing.Optional[float] = None

    #: Expected to be unsigned integer.
    receive_buffer_size: typing.Optional[float] = None

    dns_query_type: typing.Optional[DirectSocketDnsQueryType] = None

    def to_json(self):
        json = dict()
        json['noDelay'] = self.no_delay
        if self.keep_alive_delay is not None:
            json['keepAliveDelay'] = self.keep_alive_delay
        if self.send_buffer_size is not None:
            json['sendBufferSize'] = self.send_buffer_size
        if self.receive_buffer_size is not None:
            json['receiveBufferSize'] = self.receive_buffer_size
        if self.dns_query_type is not None:
            json['dnsQueryType'] = self.dns_query_type.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            no_delay=bool(json['noDelay']),
            keep_alive_delay=float(json['keepAliveDelay']) if 'keepAliveDelay' in json else None,
            send_buffer_size=float(json['sendBufferSize']) if 'sendBufferSize' in json else None,
            receive_buffer_size=float(json['receiveBufferSize']) if 'receiveBufferSize' in json else None,
            dns_query_type=DirectSocketDnsQueryType.from_json(json['dnsQueryType']) if 'dnsQueryType' in json else None,
        )


class PrivateNetworkRequestPolicy(enum.Enum):
    ALLOW = "Allow"
    BLOCK_FROM_INSECURE_TO_MORE_PRIVATE = "BlockFromInsecureToMorePrivate"
    WARN_FROM_INSECURE_TO_MORE_PRIVATE = "WarnFromInsecureToMorePrivate"
    PREFLIGHT_BLOCK = "PreflightBlock"
    PREFLIGHT_WARN = "PreflightWarn"
    PERMISSION_BLOCK = "PermissionBlock"
    PERMISSION_WARN = "PermissionWarn"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class IPAddressSpace(enum.Enum):
    LOCAL = "Local"
    PRIVATE = "Private"
    PUBLIC = "Public"
    UNKNOWN = "Unknown"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ConnectTiming:
    #: Timing's requestTime is a baseline in seconds, while the other numbers are ticks in
    #: milliseconds relatively to this requestTime. Matches ResourceTiming's requestTime for
    #: the same request (but not for redirected requests).
    request_time: float

    def to_json(self):
        json = dict()
        json['requestTime'] = self.request_time
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            request_time=float(json['requestTime']),
        )


@dataclass
class ClientSecurityState:
    initiator_is_secure_context: bool

    initiator_ip_address_space: IPAddressSpace

    private_network_request_policy: PrivateNetworkRequestPolicy

    def to_json(self):
        json = dict()
        json['initiatorIsSecureContext'] = self.initiator_is_secure_context
        json['initiatorIPAddressSpace'] = self.initiator_ip_address_space.to_json()
        json['privateNetworkRequestPolicy'] = self.private_network_request_policy.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            initiator_is_secure_context=bool(json['initiatorIsSecureContext']),
            initiator_ip_address_space=IPAddressSpace.from_json(json['initiatorIPAddressSpace']),
            private_network_request_policy=PrivateNetworkRequestPolicy.from_json(json['privateNetworkRequestPolicy']),
        )


class CrossOriginOpenerPolicyValue(enum.Enum):
    SAME_ORIGIN = "SameOrigin"
    SAME_ORIGIN_ALLOW_POPUPS = "SameOriginAllowPopups"
    RESTRICT_PROPERTIES = "RestrictProperties"
    UNSAFE_NONE = "UnsafeNone"
    SAME_ORIGIN_PLUS_COEP = "SameOriginPlusCoep"
    RESTRICT_PROPERTIES_PLUS_COEP = "RestrictPropertiesPlusCoep"
    NOOPENER_ALLOW_POPUPS = "NoopenerAllowPopups"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CrossOriginOpenerPolicyStatus:
    value: CrossOriginOpenerPolicyValue

    report_only_value: CrossOriginOpenerPolicyValue

    reporting_endpoint: typing.Optional[str] = None

    report_only_reporting_endpoint: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['value'] = self.value.to_json()
        json['reportOnlyValue'] = self.report_only_value.to_json()
        if self.reporting_endpoint is not None:
            json['reportingEndpoint'] = self.reporting_endpoint
        if self.report_only_reporting_endpoint is not None:
            json['reportOnlyReportingEndpoint'] = self.report_only_reporting_endpoint
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=CrossOriginOpenerPolicyValue.from_json(json['value']),
            report_only_value=CrossOriginOpenerPolicyValue.from_json(json['reportOnlyValue']),
            reporting_endpoint=str(json['reportingEndpoint']) if 'reportingEndpoint' in json else None,
            report_only_reporting_endpoint=str(json['reportOnlyReportingEndpoint']) if 'reportOnlyReportingEndpoint' in json else None,
        )


class CrossOriginEmbedderPolicyValue(enum.Enum):
    NONE = "None"
    CREDENTIALLESS = "Credentialless"
    REQUIRE_CORP = "RequireCorp"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class CrossOriginEmbedderPolicyStatus:
    value: CrossOriginEmbedderPolicyValue

    report_only_value: CrossOriginEmbedderPolicyValue

    reporting_endpoint: typing.Optional[str] = None

    report_only_reporting_endpoint: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['value'] = self.value.to_json()
        json['reportOnlyValue'] = self.report_only_value.to_json()
        if self.reporting_endpoint is not None:
            json['reportingEndpoint'] = self.reporting_endpoint
        if self.report_only_reporting_endpoint is not None:
            json['reportOnlyReportingEndpoint'] = self.report_only_reporting_endpoint
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            value=CrossOriginEmbedderPolicyValue.from_json(json['value']),
            report_only_value=CrossOriginEmbedderPolicyValue.from_json(json['reportOnlyValue']),
            reporting_endpoint=str(json['reportingEndpoint']) if 'reportingEndpoint' in json else None,
            report_only_reporting_endpoint=str(json['reportOnlyReportingEndpoint']) if 'reportOnlyReportingEndpoint' in json else None,
        )


class ContentSecurityPolicySource(enum.Enum):
    HTTP = "HTTP"
    META = "Meta"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class ContentSecurityPolicyStatus:
    effective_directives: str

    is_enforced: bool

    source: ContentSecurityPolicySource

    def to_json(self):
        json = dict()
        json['effectiveDirectives'] = self.effective_directives
        json['isEnforced'] = self.is_enforced
        json['source'] = self.source.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            effective_directives=str(json['effectiveDirectives']),
            is_enforced=bool(json['isEnforced']),
            source=ContentSecurityPolicySource.from_json(json['source']),
        )


@dataclass
class SecurityIsolationStatus:
    coop: typing.Optional[CrossOriginOpenerPolicyStatus] = None

    coep: typing.Optional[CrossOriginEmbedderPolicyStatus] = None

    csp: typing.Optional[typing.List[ContentSecurityPolicyStatus]] = None

    def to_json(self):
        json = dict()
        if self.coop is not None:
            json['coop'] = self.coop.to_json()
        if self.coep is not None:
            json['coep'] = self.coep.to_json()
        if self.csp is not None:
            json['csp'] = [i.to_json() for i in self.csp]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            coop=CrossOriginOpenerPolicyStatus.from_json(json['coop']) if 'coop' in json else None,
            coep=CrossOriginEmbedderPolicyStatus.from_json(json['coep']) if 'coep' in json else None,
            csp=[ContentSecurityPolicyStatus.from_json(i) for i in json['csp']] if 'csp' in json else None,
        )


class ReportStatus(enum.Enum):
    '''
    The status of a Reporting API report.
    '''
    QUEUED = "Queued"
    PENDING = "Pending"
    MARKED_FOR_REMOVAL = "MarkedForRemoval"
    SUCCESS = "Success"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class ReportId(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> ReportId:
        return cls(json)

    def __repr__(self):
        return 'ReportId({})'.format(super().__repr__())


@dataclass
class ReportingApiReport:
    '''
    An object representing a report generated by the Reporting API.
    '''
    id_: ReportId

    #: The URL of the document that triggered the report.
    initiator_url: str

    #: The name of the endpoint group that should be used to deliver the report.
    destination: str

    #: The type of the report (specifies the set of data that is contained in the report body).
    type_: str

    #: When the report was generated.
    timestamp: network.TimeSinceEpoch

    #: How many uploads deep the related request was.
    depth: int

    #: The number of delivery attempts made so far, not including an active attempt.
    completed_attempts: int

    body: dict

    status: ReportStatus

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['initiatorUrl'] = self.initiator_url
        json['destination'] = self.destination
        json['type'] = self.type_
        json['timestamp'] = self.timestamp.to_json()
        json['depth'] = self.depth
        json['completedAttempts'] = self.completed_attempts
        json['body'] = self.body
        json['status'] = self.status.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=ReportId.from_json(json['id']),
            initiator_url=str(json['initiatorUrl']),
            destination=str(json['destination']),
            type_=str(json['type']),
            timestamp=network.TimeSinceEpoch.from_json(json['timestamp']),
            depth=int(json['depth']),
            completed_attempts=int(json['completedAttempts']),
            body=dict(json['body']),
            status=ReportStatus.from_json(json['status']),
        )


@dataclass
class ReportingApiEndpoint:
    #: The URL of the endpoint to which reports may be delivered.
    url: str

    #: Name of the endpoint group.
    group_name: str

    def to_json(self):
        json = dict()
        json['url'] = self.url
        json['groupName'] = self.group_name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            url=str(json['url']),
            group_name=str(json['groupName']),
        )


@dataclass
class LoadNetworkResourcePageResult:
    '''
    An object providing the result of a network resource load.
    '''
    success: bool

    #: Optional values used for error reporting.
    net_error: typing.Optional[float] = None

    net_error_name: typing.Optional[str] = None

    http_status_code: typing.Optional[float] = None

    #: If successful, one of the following two fields holds the result.
    stream: typing.Optional[io.StreamHandle] = None

    #: Response headers.
    headers: typing.Optional[network.Headers] = None

    def to_json(self):
        json = dict()
        json['success'] = self.success
        if self.net_error is not None:
            json['netError'] = self.net_error
        if self.net_error_name is not None:
            json['netErrorName'] = self.net_error_name
        if self.http_status_code is not None:
            json['httpStatusCode'] = self.http_status_code
        if self.stream is not None:
            json['stream'] = self.stream.to_json()
        if self.headers is not None:
            json['headers'] = self.headers.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            success=bool(json['success']),
            net_error=float(json['netError']) if 'netError' in json else None,
            net_error_name=str(json['netErrorName']) if 'netErrorName' in json else None,
            http_status_code=float(json['httpStatusCode']) if 'httpStatusCode' in json else None,
            stream=io.StreamHandle.from_json(json['stream']) if 'stream' in json else None,
            headers=network.Headers.from_json(json['headers']) if 'headers' in json else None,
        )


@dataclass
class LoadNetworkResourceOptions:
    '''
    An options object that may be extended later to better support CORS,
    CORB and streaming.
    '''
    disable_cache: bool

    include_credentials: bool

    def to_json(self):
        json = dict()
        json['disableCache'] = self.disable_cache
        json['includeCredentials'] = self.include_credentials
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            disable_cache=bool(json['disableCache']),
            include_credentials=bool(json['includeCredentials']),
        )


def set_accepted_encodings(
        encodings: typing.List[ContentEncoding]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets a list of content encodings that will be accepted. Empty list means no encoding is accepted.

    **EXPERIMENTAL**

    :param encodings: List of accepted content encodings.
    '''
    params: T_JSON_DICT = dict()
    params['encodings'] = [i.to_json() for i in encodings]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setAcceptedEncodings',
        'params': params,
    }
    json = yield cmd_dict


def clear_accepted_encodings_override() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears accepted encodings set by setAcceptedEncodings

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.clearAcceptedEncodingsOverride',
    }
    json = yield cmd_dict


def can_clear_browser_cache() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Tells whether clearing browser cache is supported.

    :returns: True if browser cache can be cleared.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.canClearBrowserCache',
    }
    json = yield cmd_dict
    return bool(json['result'])


def can_clear_browser_cookies() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Tells whether clearing browser cookies is supported.

    :returns: True if browser cookies can be cleared.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.canClearBrowserCookies',
    }
    json = yield cmd_dict
    return bool(json['result'])


def can_emulate_network_conditions() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Tells whether emulation of network conditions is supported.

    :returns: True if emulation of network conditions is supported.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.canEmulateNetworkConditions',
    }
    json = yield cmd_dict
    return bool(json['result'])


def clear_browser_cache() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears browser cache.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.clearBrowserCache',
    }
    json = yield cmd_dict


def clear_browser_cookies() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears browser cookies.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.clearBrowserCookies',
    }
    json = yield cmd_dict


def continue_intercepted_request(
        interception_id: InterceptionId,
        error_reason: typing.Optional[ErrorReason] = None,
        raw_response: typing.Optional[str] = None,
        url: typing.Optional[str] = None,
        method: typing.Optional[str] = None,
        post_data: typing.Optional[str] = None,
        headers: typing.Optional[Headers] = None,
        auth_challenge_response: typing.Optional[AuthChallengeResponse] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Response to Network.requestIntercepted which either modifies the request to continue with any
    modifications, or blocks it, or completes it with the provided response bytes. If a network
    fetch occurs as a result which encounters a redirect an additional Network.requestIntercepted
    event will be sent with the same InterceptionId.
    Deprecated, use Fetch.continueRequest, Fetch.fulfillRequest and Fetch.failRequest instead.

    **EXPERIMENTAL**

    :param interception_id:
    :param error_reason: *(Optional)* If set this causes the request to fail with the given reason. Passing ```Aborted```` for requests marked with ````isNavigationRequest``` also cancels the navigation. Must not be set in response to an authChallenge.
    :param raw_response: *(Optional)* If set the requests completes using with the provided base64 encoded raw response, including HTTP status line and headers etc... Must not be set in response to an authChallenge.
    :param url: *(Optional)* If set the request url will be modified in a way that's not observable by page. Must not be set in response to an authChallenge.
    :param method: *(Optional)* If set this allows the request method to be overridden. Must not be set in response to an authChallenge.
    :param post_data: *(Optional)* If set this allows postData to be set. Must not be set in response to an authChallenge.
    :param headers: *(Optional)* If set this allows the request headers to be changed. Must not be set in response to an authChallenge.
    :param auth_challenge_response: *(Optional)* Response to a requestIntercepted with an authChallenge. Must not be set otherwise.
    '''
    params: T_JSON_DICT = dict()
    params['interceptionId'] = interception_id.to_json()
    if error_reason is not None:
        params['errorReason'] = error_reason.to_json()
    if raw_response is not None:
        params['rawResponse'] = raw_response
    if url is not None:
        params['url'] = url
    if method is not None:
        params['method'] = method
    if post_data is not None:
        params['postData'] = post_data
    if headers is not None:
        params['headers'] = headers.to_json()
    if auth_challenge_response is not None:
        params['authChallengeResponse'] = auth_challenge_response.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.continueInterceptedRequest',
        'params': params,
    }
    json = yield cmd_dict


def delete_cookies(
        name: str,
        url: typing.Optional[str] = None,
        domain: typing.Optional[str] = None,
        path: typing.Optional[str] = None,
        partition_key: typing.Optional[CookiePartitionKey] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Deletes browser cookies with matching name and url or domain/path/partitionKey pair.

    :param name: Name of the cookies to remove.
    :param url: *(Optional)* If specified, deletes all the cookies with the given name where domain and path match provided URL.
    :param domain: *(Optional)* If specified, deletes only cookies with the exact domain.
    :param path: *(Optional)* If specified, deletes only cookies with the exact path.
    :param partition_key: **(EXPERIMENTAL)** *(Optional)* If specified, deletes only cookies with the the given name and partitionKey where all partition key attributes match the cookie partition key attribute.
    '''
    params: T_JSON_DICT = dict()
    params['name'] = name
    if url is not None:
        params['url'] = url
    if domain is not None:
        params['domain'] = domain
    if path is not None:
        params['path'] = path
    if partition_key is not None:
        params['partitionKey'] = partition_key.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.deleteCookies',
        'params': params,
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables network tracking, prevents network events from being sent to the client.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.disable',
    }
    json = yield cmd_dict


def emulate_network_conditions(
        offline: bool,
        latency: float,
        download_throughput: float,
        upload_throughput: float,
        connection_type: typing.Optional[ConnectionType] = None,
        packet_loss: typing.Optional[float] = None,
        packet_queue_length: typing.Optional[int] = None,
        packet_reordering: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Activates emulation of network conditions.

    :param offline: True to emulate internet disconnection.
    :param latency: Minimum latency from request sent to response headers received (ms).
    :param download_throughput: Maximal aggregated download throughput (bytes/sec). -1 disables download throttling.
    :param upload_throughput: Maximal aggregated upload throughput (bytes/sec).  -1 disables upload throttling.
    :param connection_type: *(Optional)* Connection type if known.
    :param packet_loss: **(EXPERIMENTAL)** *(Optional)* WebRTC packet loss (percent, 0-100). 0 disables packet loss emulation, 100 drops all the packets.
    :param packet_queue_length: **(EXPERIMENTAL)** *(Optional)* WebRTC packet queue length (packet). 0 removes any queue length limitations.
    :param packet_reordering: **(EXPERIMENTAL)** *(Optional)* WebRTC packetReordering feature.
    '''
    params: T_JSON_DICT = dict()
    params['offline'] = offline
    params['latency'] = latency
    params['downloadThroughput'] = download_throughput
    params['uploadThroughput'] = upload_throughput
    if connection_type is not None:
        params['connectionType'] = connection_type.to_json()
    if packet_loss is not None:
        params['packetLoss'] = packet_loss
    if packet_queue_length is not None:
        params['packetQueueLength'] = packet_queue_length
    if packet_reordering is not None:
        params['packetReordering'] = packet_reordering
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.emulateNetworkConditions',
        'params': params,
    }
    json = yield cmd_dict


def enable(
        max_total_buffer_size: typing.Optional[int] = None,
        max_resource_buffer_size: typing.Optional[int] = None,
        max_post_data_size: typing.Optional[int] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables network tracking, network events will now be delivered to the client.

    :param max_total_buffer_size: **(EXPERIMENTAL)** *(Optional)* Buffer size in bytes to use when preserving network payloads (XHRs, etc).
    :param max_resource_buffer_size: **(EXPERIMENTAL)** *(Optional)* Per-resource buffer size in bytes to use when preserving network payloads (XHRs, etc).
    :param max_post_data_size: *(Optional)* Longest post body size (in bytes) that would be included in requestWillBeSent notification
    '''
    params: T_JSON_DICT = dict()
    if max_total_buffer_size is not None:
        params['maxTotalBufferSize'] = max_total_buffer_size
    if max_resource_buffer_size is not None:
        params['maxResourceBufferSize'] = max_resource_buffer_size
    if max_post_data_size is not None:
        params['maxPostDataSize'] = max_post_data_size
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.enable',
        'params': params,
    }
    json = yield cmd_dict


def get_all_cookies() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Cookie]]:
    '''
    Returns all browser cookies. Depending on the backend support, will return detailed cookie
    information in the ``cookies`` field.
    Deprecated. Use Storage.getCookies instead.

    :returns: Array of cookie objects.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getAllCookies',
    }
    json = yield cmd_dict
    return [Cookie.from_json(i) for i in json['cookies']]


def get_certificate(
        origin: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Returns the DER-encoded certificate.

    **EXPERIMENTAL**

    :param origin: Origin to get certificate for.
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getCertificate',
        'params': params,
    }
    json = yield cmd_dict
    return [str(i) for i in json['tableNames']]


def get_cookies(
        urls: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Cookie]]:
    '''
    Returns all browser cookies for the current URL. Depending on the backend support, will return
    detailed cookie information in the ``cookies`` field.

    :param urls: *(Optional)* The list of URLs for which applicable cookies will be fetched. If not specified, it's assumed to be set to the list containing the URLs of the page and all of its subframes.
    :returns: Array of cookie objects.
    '''
    params: T_JSON_DICT = dict()
    if urls is not None:
        params['urls'] = [i for i in urls]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getCookies',
        'params': params,
    }
    json = yield cmd_dict
    return [Cookie.from_json(i) for i in json['cookies']]


def get_response_body(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Returns content served for the given request.

    :param request_id: Identifier of the network request to get content for.
    :returns: A tuple with the following items:

        0. **body** - Response body.
        1. **base64Encoded** - True, if content was sent as base64.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getResponseBody',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['body']),
        bool(json['base64Encoded'])
    )


def get_request_post_data(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Returns post data sent with the request. Returns an error when no data was sent with the request.

    :param request_id: Identifier of the network request to get content for.
    :returns: Request body string, omitting files from multipart requests
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getRequestPostData',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['postData'])


def get_response_body_for_interception(
        interception_id: InterceptionId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, bool]]:
    '''
    Returns content served for the given currently intercepted request.

    **EXPERIMENTAL**

    :param interception_id: Identifier for the intercepted request to get body for.
    :returns: A tuple with the following items:

        0. **body** - Response body.
        1. **base64Encoded** - True, if content was sent as base64.
    '''
    params: T_JSON_DICT = dict()
    params['interceptionId'] = interception_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getResponseBodyForInterception',
        'params': params,
    }
    json = yield cmd_dict
    return (
        str(json['body']),
        bool(json['base64Encoded'])
    )


def take_response_body_for_interception_as_stream(
        interception_id: InterceptionId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,io.StreamHandle]:
    '''
    Returns a handle to the stream representing the response body. Note that after this command,
    the intercepted request can't be continued as is -- you either need to cancel it or to provide
    the response body. The stream only supports sequential read, IO.read will fail if the position
    is specified.

    **EXPERIMENTAL**

    :param interception_id:
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['interceptionId'] = interception_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.takeResponseBodyForInterceptionAsStream',
        'params': params,
    }
    json = yield cmd_dict
    return io.StreamHandle.from_json(json['stream'])


def replay_xhr(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    This method sends a new XMLHttpRequest which is identical to the original one. The following
    parameters should be identical: method, url, async, request body, extra headers, withCredentials
    attribute, user, password.

    **EXPERIMENTAL**

    :param request_id: Identifier of XHR to replay.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.replayXHR',
        'params': params,
    }
    json = yield cmd_dict


def search_in_response_body(
        request_id: RequestId,
        query: str,
        case_sensitive: typing.Optional[bool] = None,
        is_regex: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[debugger.SearchMatch]]:
    '''
    Searches for given string in response content.

    **EXPERIMENTAL**

    :param request_id: Identifier of the network response to search.
    :param query: String to search for.
    :param case_sensitive: *(Optional)* If true, search is case sensitive.
    :param is_regex: *(Optional)* If true, treats string parameter as regex.
    :returns: List of search matches.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    params['query'] = query
    if case_sensitive is not None:
        params['caseSensitive'] = case_sensitive
    if is_regex is not None:
        params['isRegex'] = is_regex
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.searchInResponseBody',
        'params': params,
    }
    json = yield cmd_dict
    return [debugger.SearchMatch.from_json(i) for i in json['result']]


def set_blocked_ur_ls(
        urls: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Blocks URLs from loading.

    **EXPERIMENTAL**

    :param urls: URL patterns to block. Wildcards ('*') are allowed.
    '''
    params: T_JSON_DICT = dict()
    params['urls'] = [i for i in urls]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setBlockedURLs',
        'params': params,
    }
    json = yield cmd_dict


def set_bypass_service_worker(
        bypass: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Toggles ignoring of service worker for each request.

    :param bypass: Bypass service worker and load from network.
    '''
    params: T_JSON_DICT = dict()
    params['bypass'] = bypass
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setBypassServiceWorker',
        'params': params,
    }
    json = yield cmd_dict


def set_cache_disabled(
        cache_disabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Toggles ignoring cache for each request. If ``true``, cache will not be used.

    :param cache_disabled: Cache disabled state.
    '''
    params: T_JSON_DICT = dict()
    params['cacheDisabled'] = cache_disabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setCacheDisabled',
        'params': params,
    }
    json = yield cmd_dict


def set_cookie(
        name: str,
        value: str,
        url: typing.Optional[str] = None,
        domain: typing.Optional[str] = None,
        path: typing.Optional[str] = None,
        secure: typing.Optional[bool] = None,
        http_only: typing.Optional[bool] = None,
        same_site: typing.Optional[CookieSameSite] = None,
        expires: typing.Optional[TimeSinceEpoch] = None,
        priority: typing.Optional[CookiePriority] = None,
        same_party: typing.Optional[bool] = None,
        source_scheme: typing.Optional[CookieSourceScheme] = None,
        source_port: typing.Optional[int] = None,
        partition_key: typing.Optional[CookiePartitionKey] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,bool]:
    '''
    Sets a cookie with the given cookie data; may overwrite equivalent cookies if they exist.

    :param name: Cookie name.
    :param value: Cookie value.
    :param url: *(Optional)* The request-URI to associate with the setting of the cookie. This value can affect the default domain, path, source port, and source scheme values of the created cookie.
    :param domain: *(Optional)* Cookie domain.
    :param path: *(Optional)* Cookie path.
    :param secure: *(Optional)* True if cookie is secure.
    :param http_only: *(Optional)* True if cookie is http-only.
    :param same_site: *(Optional)* Cookie SameSite type.
    :param expires: *(Optional)* Cookie expiration date, session cookie if not set
    :param priority: **(EXPERIMENTAL)** *(Optional)* Cookie Priority type.
    :param same_party: **(EXPERIMENTAL)** *(Optional)* True if cookie is SameParty.
    :param source_scheme: **(EXPERIMENTAL)** *(Optional)* Cookie source scheme type.
    :param source_port: **(EXPERIMENTAL)** *(Optional)* Cookie source port. Valid values are {-1, [1, 65535]}, -1 indicates an unspecified port. An unspecified port value allows protocol clients to emulate legacy cookie scope for the port. This is a temporary ability and it will be removed in the future.
    :param partition_key: **(EXPERIMENTAL)** *(Optional)* Cookie partition key. If not set, the cookie will be set as not partitioned.
    :returns: Always set to true. If an error occurs, the response indicates protocol error.
    '''
    params: T_JSON_DICT = dict()
    params['name'] = name
    params['value'] = value
    if url is not None:
        params['url'] = url
    if domain is not None:
        params['domain'] = domain
    if path is not None:
        params['path'] = path
    if secure is not None:
        params['secure'] = secure
    if http_only is not None:
        params['httpOnly'] = http_only
    if same_site is not None:
        params['sameSite'] = same_site.to_json()
    if expires is not None:
        params['expires'] = expires.to_json()
    if priority is not None:
        params['priority'] = priority.to_json()
    if same_party is not None:
        params['sameParty'] = same_party
    if source_scheme is not None:
        params['sourceScheme'] = source_scheme.to_json()
    if source_port is not None:
        params['sourcePort'] = source_port
    if partition_key is not None:
        params['partitionKey'] = partition_key.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setCookie',
        'params': params,
    }
    json = yield cmd_dict
    return bool(json['success'])


def set_cookies(
        cookies: typing.List[CookieParam]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets given cookies.

    :param cookies: Cookies to be set.
    '''
    params: T_JSON_DICT = dict()
    params['cookies'] = [i.to_json() for i in cookies]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setCookies',
        'params': params,
    }
    json = yield cmd_dict


def set_extra_http_headers(
        headers: Headers
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Specifies whether to always send extra HTTP headers with the requests from this page.

    :param headers: Map with extra HTTP headers.
    '''
    params: T_JSON_DICT = dict()
    params['headers'] = headers.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setExtraHTTPHeaders',
        'params': params,
    }
    json = yield cmd_dict


def set_attach_debug_stack(
        enabled: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Specifies whether to attach a page script stack id in requests

    **EXPERIMENTAL**

    :param enabled: Whether to attach a page script stack for debugging purpose.
    '''
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setAttachDebugStack',
        'params': params,
    }
    json = yield cmd_dict


def set_request_interception(
        patterns: typing.List[RequestPattern]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the requests to intercept that match the provided patterns and optionally resource types.
    Deprecated, please use Fetch.enable instead.

    **EXPERIMENTAL**

    :param patterns: Requests matching any of these patterns will be forwarded and wait for the corresponding continueInterceptedRequest call.
    '''
    params: T_JSON_DICT = dict()
    params['patterns'] = [i.to_json() for i in patterns]
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setRequestInterception',
        'params': params,
    }
    json = yield cmd_dict


def set_user_agent_override(
        user_agent: str,
        accept_language: typing.Optional[str] = None,
        platform: typing.Optional[str] = None,
        user_agent_metadata: typing.Optional[emulation.UserAgentMetadata] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Allows overriding user agent with the given string.

    :param user_agent: User agent to use.
    :param accept_language: *(Optional)* Browser language to emulate.
    :param platform: *(Optional)* The platform navigator.platform should return.
    :param user_agent_metadata: **(EXPERIMENTAL)** *(Optional)* To be sent in Sec-CH-UA-* headers and returned in navigator.userAgentData
    '''
    params: T_JSON_DICT = dict()
    params['userAgent'] = user_agent
    if accept_language is not None:
        params['acceptLanguage'] = accept_language
    if platform is not None:
        params['platform'] = platform
    if user_agent_metadata is not None:
        params['userAgentMetadata'] = user_agent_metadata.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setUserAgentOverride',
        'params': params,
    }
    json = yield cmd_dict


def stream_resource_content(
        request_id: RequestId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Enables streaming of the response for the given requestId.
    If enabled, the dataReceived event contains the data that was received during streaming.

    **EXPERIMENTAL**

    :param request_id: Identifier of the request to stream.
    :returns: Data that has been buffered until streaming is enabled.
    '''
    params: T_JSON_DICT = dict()
    params['requestId'] = request_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.streamResourceContent',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['bufferedData'])


def get_security_isolation_status(
        frame_id: typing.Optional[page.FrameId] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,SecurityIsolationStatus]:
    '''
    Returns information about the COEP/COOP isolation status.

    **EXPERIMENTAL**

    :param frame_id: *(Optional)* If no frameId is provided, the status of the target is provided.
    :returns:
    '''
    params: T_JSON_DICT = dict()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.getSecurityIsolationStatus',
        'params': params,
    }
    json = yield cmd_dict
    return SecurityIsolationStatus.from_json(json['status'])


def enable_reporting_api(
        enable: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables tracking for the Reporting API, events generated by the Reporting API will now be delivered to the client.
    Enabling triggers 'reportingApiReportAdded' for all existing reports.

    **EXPERIMENTAL**

    :param enable: Whether to enable or disable events for the Reporting API
    '''
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.enableReportingApi',
        'params': params,
    }
    json = yield cmd_dict


def load_network_resource(
        frame_id: typing.Optional[page.FrameId] = None,
        url: str = None,
        options: LoadNetworkResourceOptions = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,LoadNetworkResourcePageResult]:
    '''
    Fetches the resource and returns the content.

    **EXPERIMENTAL**

    :param frame_id: *(Optional)* Frame id to get the resource for. Mandatory for frame targets, and should be omitted for worker targets.
    :param url: URL of the resource to get content for.
    :param options: Options for the request.
    :returns:
    '''
    params: T_JSON_DICT = dict()
    if frame_id is not None:
        params['frameId'] = frame_id.to_json()
    params['url'] = url
    params['options'] = options.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.loadNetworkResource',
        'params': params,
    }
    json = yield cmd_dict
    return LoadNetworkResourcePageResult.from_json(json['resource'])


def set_cookie_controls(
        enable_third_party_cookie_restriction: bool,
        disable_third_party_cookie_metadata: bool,
        disable_third_party_cookie_heuristics: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets Controls for third-party cookie access
    Page reload is required before the new cookie bahavior will be observed

    **EXPERIMENTAL**

    :param enable_third_party_cookie_restriction: Whether 3pc restriction is enabled.
    :param disable_third_party_cookie_metadata: Whether 3pc grace period exception should be enabled; false by default.
    :param disable_third_party_cookie_heuristics: Whether 3pc heuristics exceptions should be enabled; false by default.
    '''
    params: T_JSON_DICT = dict()
    params['enableThirdPartyCookieRestriction'] = enable_third_party_cookie_restriction
    params['disableThirdPartyCookieMetadata'] = disable_third_party_cookie_metadata
    params['disableThirdPartyCookieHeuristics'] = disable_third_party_cookie_heuristics
    cmd_dict: T_JSON_DICT = {
        'method': 'Network.setCookieControls',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Network.dataReceived')
@dataclass
class DataReceived:
    '''
    Fired when data chunk was received over the network.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Data chunk length.
    data_length: int
    #: Actual bytes received (might be less than dataLength for compressed encodings).
    encoded_data_length: int
    #: Data that was received.
    data: typing.Optional[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DataReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            data_length=int(json['dataLength']),
            encoded_data_length=int(json['encodedDataLength']),
            data=str(json['data']) if 'data' in json else None
        )


@event_class('Network.eventSourceMessageReceived')
@dataclass
class EventSourceMessageReceived:
    '''
    Fired when EventSource message is received.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Message type.
    event_name: str
    #: Message identifier.
    event_id: str
    #: Message content.
    data: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> EventSourceMessageReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            event_name=str(json['eventName']),
            event_id=str(json['eventId']),
            data=str(json['data'])
        )


@event_class('Network.loadingFailed')
@dataclass
class LoadingFailed:
    '''
    Fired when HTTP request has failed to load.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Resource type.
    type_: ResourceType
    #: Error message. List of network errors: https://cs.chromium.org/chromium/src/net/base/net_error_list.h
    error_text: str
    #: True if loading was canceled.
    canceled: typing.Optional[bool]
    #: The reason why loading was blocked, if any.
    blocked_reason: typing.Optional[BlockedReason]
    #: The reason why loading was blocked by CORS, if any.
    cors_error_status: typing.Optional[CorsErrorStatus]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadingFailed:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            type_=ResourceType.from_json(json['type']),
            error_text=str(json['errorText']),
            canceled=bool(json['canceled']) if 'canceled' in json else None,
            blocked_reason=BlockedReason.from_json(json['blockedReason']) if 'blockedReason' in json else None,
            cors_error_status=CorsErrorStatus.from_json(json['corsErrorStatus']) if 'corsErrorStatus' in json else None
        )


@event_class('Network.loadingFinished')
@dataclass
class LoadingFinished:
    '''
    Fired when HTTP request has finished loading.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Total number of bytes received for this request.
    encoded_data_length: float

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> LoadingFinished:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            encoded_data_length=float(json['encodedDataLength'])
        )


@event_class('Network.requestIntercepted')
@dataclass
class RequestIntercepted:
    '''
    **EXPERIMENTAL**

    Details of an intercepted HTTP request, which must be either allowed, blocked, modified or
    mocked.
    Deprecated, use Fetch.requestPaused instead.
    '''
    #: Each request the page makes will have a unique id, however if any redirects are encountered
    #: while processing that fetch, they will be reported with the same id as the original fetch.
    #: Likewise if HTTP authentication is needed then the same fetch id will be used.
    interception_id: InterceptionId
    request: Request
    #: The id of the frame that initiated the request.
    frame_id: page.FrameId
    #: How the requested resource will be used.
    resource_type: ResourceType
    #: Whether this is a navigation request, which can abort the navigation completely.
    is_navigation_request: bool
    #: Set if the request is a navigation that will result in a download.
    #: Only present after response is received from the server (i.e. HeadersReceived stage).
    is_download: typing.Optional[bool]
    #: Redirect location, only sent if a redirect was intercepted.
    redirect_url: typing.Optional[str]
    #: Details of the Authorization Challenge encountered. If this is set then
    #: continueInterceptedRequest must contain an authChallengeResponse.
    auth_challenge: typing.Optional[AuthChallenge]
    #: Response error if intercepted at response stage or if redirect occurred while intercepting
    #: request.
    response_error_reason: typing.Optional[ErrorReason]
    #: Response code if intercepted at response stage or if redirect occurred while intercepting
    #: request or auth retry occurred.
    response_status_code: typing.Optional[int]
    #: Response headers if intercepted at the response stage or if redirect occurred while
    #: intercepting request or auth retry occurred.
    response_headers: typing.Optional[Headers]
    #: If the intercepted request had a corresponding requestWillBeSent event fired for it, then
    #: this requestId will be the same as the requestId present in the requestWillBeSent event.
    request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestIntercepted:
        return cls(
            interception_id=InterceptionId.from_json(json['interceptionId']),
            request=Request.from_json(json['request']),
            frame_id=page.FrameId.from_json(json['frameId']),
            resource_type=ResourceType.from_json(json['resourceType']),
            is_navigation_request=bool(json['isNavigationRequest']),
            is_download=bool(json['isDownload']) if 'isDownload' in json else None,
            redirect_url=str(json['redirectUrl']) if 'redirectUrl' in json else None,
            auth_challenge=AuthChallenge.from_json(json['authChallenge']) if 'authChallenge' in json else None,
            response_error_reason=ErrorReason.from_json(json['responseErrorReason']) if 'responseErrorReason' in json else None,
            response_status_code=int(json['responseStatusCode']) if 'responseStatusCode' in json else None,
            response_headers=Headers.from_json(json['responseHeaders']) if 'responseHeaders' in json else None,
            request_id=RequestId.from_json(json['requestId']) if 'requestId' in json else None
        )


@event_class('Network.requestServedFromCache')
@dataclass
class RequestServedFromCache:
    '''
    Fired if request ended up loading from cache.
    '''
    #: Request identifier.
    request_id: RequestId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestServedFromCache:
        return cls(
            request_id=RequestId.from_json(json['requestId'])
        )


@event_class('Network.requestWillBeSent')
@dataclass
class RequestWillBeSent:
    '''
    Fired when page is about to send HTTP request.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Loader identifier. Empty string if the request is fetched from worker.
    loader_id: LoaderId
    #: URL of the document this request is loaded for.
    document_url: str
    #: Request data.
    request: Request
    #: Timestamp.
    timestamp: MonotonicTime
    #: Timestamp.
    wall_time: TimeSinceEpoch
    #: Request initiator.
    initiator: Initiator
    #: In the case that redirectResponse is populated, this flag indicates whether
    #: requestWillBeSentExtraInfo and responseReceivedExtraInfo events will be or were emitted
    #: for the request which was just redirected.
    redirect_has_extra_info: bool
    #: Redirect response data.
    redirect_response: typing.Optional[Response]
    #: Type of this resource.
    type_: typing.Optional[ResourceType]
    #: Frame identifier.
    frame_id: typing.Optional[page.FrameId]
    #: Whether the request is initiated by a user gesture. Defaults to false.
    has_user_gesture: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestWillBeSent:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            loader_id=LoaderId.from_json(json['loaderId']),
            document_url=str(json['documentURL']),
            request=Request.from_json(json['request']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            wall_time=TimeSinceEpoch.from_json(json['wallTime']),
            initiator=Initiator.from_json(json['initiator']),
            redirect_has_extra_info=bool(json['redirectHasExtraInfo']),
            redirect_response=Response.from_json(json['redirectResponse']) if 'redirectResponse' in json else None,
            type_=ResourceType.from_json(json['type']) if 'type' in json else None,
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None,
            has_user_gesture=bool(json['hasUserGesture']) if 'hasUserGesture' in json else None
        )


@event_class('Network.resourceChangedPriority')
@dataclass
class ResourceChangedPriority:
    '''
    **EXPERIMENTAL**

    Fired when resource loading priority is changed
    '''
    #: Request identifier.
    request_id: RequestId
    #: New priority
    new_priority: ResourcePriority
    #: Timestamp.
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResourceChangedPriority:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            new_priority=ResourcePriority.from_json(json['newPriority']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.signedExchangeReceived')
@dataclass
class SignedExchangeReceived:
    '''
    **EXPERIMENTAL**

    Fired when a signed exchange was received over the network
    '''
    #: Request identifier.
    request_id: RequestId
    #: Information about the signed exchange response.
    info: SignedExchangeInfo

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SignedExchangeReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            info=SignedExchangeInfo.from_json(json['info'])
        )


@event_class('Network.responseReceived')
@dataclass
class ResponseReceived:
    '''
    Fired when HTTP response is available.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Loader identifier. Empty string if the request is fetched from worker.
    loader_id: LoaderId
    #: Timestamp.
    timestamp: MonotonicTime
    #: Resource type.
    type_: ResourceType
    #: Response data.
    response: Response
    #: Indicates whether requestWillBeSentExtraInfo and responseReceivedExtraInfo events will be
    #: or were emitted for this request.
    has_extra_info: bool
    #: Frame identifier.
    frame_id: typing.Optional[page.FrameId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResponseReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            loader_id=LoaderId.from_json(json['loaderId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            type_=ResourceType.from_json(json['type']),
            response=Response.from_json(json['response']),
            has_extra_info=bool(json['hasExtraInfo']),
            frame_id=page.FrameId.from_json(json['frameId']) if 'frameId' in json else None
        )


@event_class('Network.webSocketClosed')
@dataclass
class WebSocketClosed:
    '''
    Fired when WebSocket is closed.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketClosed:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.webSocketCreated')
@dataclass
class WebSocketCreated:
    '''
    Fired upon WebSocket creation.
    '''
    #: Request identifier.
    request_id: RequestId
    #: WebSocket request URL.
    url: str
    #: Request initiator.
    initiator: typing.Optional[Initiator]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketCreated:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            url=str(json['url']),
            initiator=Initiator.from_json(json['initiator']) if 'initiator' in json else None
        )


@event_class('Network.webSocketFrameError')
@dataclass
class WebSocketFrameError:
    '''
    Fired when WebSocket message error occurs.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: WebSocket error message.
    error_message: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketFrameError:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            error_message=str(json['errorMessage'])
        )


@event_class('Network.webSocketFrameReceived')
@dataclass
class WebSocketFrameReceived:
    '''
    Fired when WebSocket message is received.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: WebSocket response data.
    response: WebSocketFrame

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketFrameReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            response=WebSocketFrame.from_json(json['response'])
        )


@event_class('Network.webSocketFrameSent')
@dataclass
class WebSocketFrameSent:
    '''
    Fired when WebSocket message is sent.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: WebSocket response data.
    response: WebSocketFrame

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketFrameSent:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            response=WebSocketFrame.from_json(json['response'])
        )


@event_class('Network.webSocketHandshakeResponseReceived')
@dataclass
class WebSocketHandshakeResponseReceived:
    '''
    Fired when WebSocket handshake response becomes available.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: WebSocket response data.
    response: WebSocketResponse

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketHandshakeResponseReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            response=WebSocketResponse.from_json(json['response'])
        )


@event_class('Network.webSocketWillSendHandshakeRequest')
@dataclass
class WebSocketWillSendHandshakeRequest:
    '''
    Fired when WebSocket is about to initiate handshake.
    '''
    #: Request identifier.
    request_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime
    #: UTC Timestamp.
    wall_time: TimeSinceEpoch
    #: WebSocket request data.
    request: WebSocketRequest

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebSocketWillSendHandshakeRequest:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            wall_time=TimeSinceEpoch.from_json(json['wallTime']),
            request=WebSocketRequest.from_json(json['request'])
        )


@event_class('Network.webTransportCreated')
@dataclass
class WebTransportCreated:
    '''
    Fired upon WebTransport creation.
    '''
    #: WebTransport identifier.
    transport_id: RequestId
    #: WebTransport request URL.
    url: str
    #: Timestamp.
    timestamp: MonotonicTime
    #: Request initiator.
    initiator: typing.Optional[Initiator]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebTransportCreated:
        return cls(
            transport_id=RequestId.from_json(json['transportId']),
            url=str(json['url']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            initiator=Initiator.from_json(json['initiator']) if 'initiator' in json else None
        )


@event_class('Network.webTransportConnectionEstablished')
@dataclass
class WebTransportConnectionEstablished:
    '''
    Fired when WebTransport handshake is finished.
    '''
    #: WebTransport identifier.
    transport_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebTransportConnectionEstablished:
        return cls(
            transport_id=RequestId.from_json(json['transportId']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.webTransportClosed')
@dataclass
class WebTransportClosed:
    '''
    Fired when WebTransport is disposed.
    '''
    #: WebTransport identifier.
    transport_id: RequestId
    #: Timestamp.
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> WebTransportClosed:
        return cls(
            transport_id=RequestId.from_json(json['transportId']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.directTCPSocketCreated')
@dataclass
class DirectTCPSocketCreated:
    '''
    **EXPERIMENTAL**

    Fired upon direct_socket.TCPSocket creation.
    '''
    identifier: RequestId
    remote_addr: str
    #: Unsigned int 16.
    remote_port: int
    options: DirectTCPSocketOptions
    timestamp: MonotonicTime
    initiator: typing.Optional[Initiator]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DirectTCPSocketCreated:
        return cls(
            identifier=RequestId.from_json(json['identifier']),
            remote_addr=str(json['remoteAddr']),
            remote_port=int(json['remotePort']),
            options=DirectTCPSocketOptions.from_json(json['options']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            initiator=Initiator.from_json(json['initiator']) if 'initiator' in json else None
        )


@event_class('Network.directTCPSocketOpened')
@dataclass
class DirectTCPSocketOpened:
    '''
    **EXPERIMENTAL**

    Fired when direct_socket.TCPSocket connection is opened.
    '''
    identifier: RequestId
    remote_addr: str
    #: Expected to be unsigned integer.
    remote_port: int
    timestamp: MonotonicTime
    local_addr: typing.Optional[str]
    #: Expected to be unsigned integer.
    local_port: typing.Optional[int]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DirectTCPSocketOpened:
        return cls(
            identifier=RequestId.from_json(json['identifier']),
            remote_addr=str(json['remoteAddr']),
            remote_port=int(json['remotePort']),
            timestamp=MonotonicTime.from_json(json['timestamp']),
            local_addr=str(json['localAddr']) if 'localAddr' in json else None,
            local_port=int(json['localPort']) if 'localPort' in json else None
        )


@event_class('Network.directTCPSocketAborted')
@dataclass
class DirectTCPSocketAborted:
    '''
    **EXPERIMENTAL**

    Fired when direct_socket.TCPSocket is aborted.
    '''
    identifier: RequestId
    error_message: str
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DirectTCPSocketAborted:
        return cls(
            identifier=RequestId.from_json(json['identifier']),
            error_message=str(json['errorMessage']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.directTCPSocketClosed')
@dataclass
class DirectTCPSocketClosed:
    '''
    **EXPERIMENTAL**

    Fired when direct_socket.TCPSocket is closed.
    '''
    identifier: RequestId
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DirectTCPSocketClosed:
        return cls(
            identifier=RequestId.from_json(json['identifier']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.directTCPSocketChunkSent')
@dataclass
class DirectTCPSocketChunkSent:
    '''
    **EXPERIMENTAL**

    Fired when data is sent to tcp direct socket stream.
    '''
    identifier: RequestId
    data: str
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DirectTCPSocketChunkSent:
        return cls(
            identifier=RequestId.from_json(json['identifier']),
            data=str(json['data']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.directTCPSocketChunkReceived')
@dataclass
class DirectTCPSocketChunkReceived:
    '''
    **EXPERIMENTAL**

    Fired when data is received from tcp direct socket stream.
    '''
    identifier: RequestId
    data: str
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DirectTCPSocketChunkReceived:
        return cls(
            identifier=RequestId.from_json(json['identifier']),
            data=str(json['data']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.directTCPSocketChunkError')
@dataclass
class DirectTCPSocketChunkError:
    '''
    **EXPERIMENTAL**

    Fired when there is an error
    when writing to tcp direct socket stream.
    For example, if user writes illegal type like string
    instead of ArrayBuffer or ArrayBufferView.
    There's no reporting for reading, because
    we cannot know errors on the other side.
    '''
    identifier: RequestId
    error_message: str
    timestamp: MonotonicTime

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DirectTCPSocketChunkError:
        return cls(
            identifier=RequestId.from_json(json['identifier']),
            error_message=str(json['errorMessage']),
            timestamp=MonotonicTime.from_json(json['timestamp'])
        )


@event_class('Network.requestWillBeSentExtraInfo')
@dataclass
class RequestWillBeSentExtraInfo:
    '''
    **EXPERIMENTAL**

    Fired when additional information about a requestWillBeSent event is available from the
    network stack. Not every requestWillBeSent event will have an additional
    requestWillBeSentExtraInfo fired for it, and there is no guarantee whether requestWillBeSent
    or requestWillBeSentExtraInfo will be fired first for the same request.
    '''
    #: Request identifier. Used to match this information to an existing requestWillBeSent event.
    request_id: RequestId
    #: A list of cookies potentially associated to the requested URL. This includes both cookies sent with
    #: the request and the ones not sent; the latter are distinguished by having blockedReasons field set.
    associated_cookies: typing.List[AssociatedCookie]
    #: Raw request headers as they will be sent over the wire.
    headers: Headers
    #: Connection timing information for the request.
    connect_timing: ConnectTiming
    #: The client security state set for the request.
    client_security_state: typing.Optional[ClientSecurityState]
    #: Whether the site has partitioned cookies stored in a partition different than the current one.
    site_has_cookie_in_other_partition: typing.Optional[bool]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RequestWillBeSentExtraInfo:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            associated_cookies=[AssociatedCookie.from_json(i) for i in json['associatedCookies']],
            headers=Headers.from_json(json['headers']),
            connect_timing=ConnectTiming.from_json(json['connectTiming']),
            client_security_state=ClientSecurityState.from_json(json['clientSecurityState']) if 'clientSecurityState' in json else None,
            site_has_cookie_in_other_partition=bool(json['siteHasCookieInOtherPartition']) if 'siteHasCookieInOtherPartition' in json else None
        )


@event_class('Network.responseReceivedExtraInfo')
@dataclass
class ResponseReceivedExtraInfo:
    '''
    **EXPERIMENTAL**

    Fired when additional information about a responseReceived event is available from the network
    stack. Not every responseReceived event will have an additional responseReceivedExtraInfo for
    it, and responseReceivedExtraInfo may be fired before or after responseReceived.
    '''
    #: Request identifier. Used to match this information to another responseReceived event.
    request_id: RequestId
    #: A list of cookies which were not stored from the response along with the corresponding
    #: reasons for blocking. The cookies here may not be valid due to syntax errors, which
    #: are represented by the invalid cookie line string instead of a proper cookie.
    blocked_cookies: typing.List[BlockedSetCookieWithReason]
    #: Raw response headers as they were received over the wire.
    #: Duplicate headers in the response are represented as a single key with their values
    #: concatentated using ``\n`` as the separator.
    #: See also ``headersText`` that contains verbatim text for HTTP/1.*.
    headers: Headers
    #: The IP address space of the resource. The address space can only be determined once the transport
    #: established the connection, so we can't send it in ``requestWillBeSentExtraInfo``.
    resource_ip_address_space: IPAddressSpace
    #: The status code of the response. This is useful in cases the request failed and no responseReceived
    #: event is triggered, which is the case for, e.g., CORS errors. This is also the correct status code
    #: for cached requests, where the status in responseReceived is a 200 and this will be 304.
    status_code: int
    #: Raw response header text as it was received over the wire. The raw text may not always be
    #: available, such as in the case of HTTP/2 or QUIC.
    headers_text: typing.Optional[str]
    #: The cookie partition key that will be used to store partitioned cookies set in this response.
    #: Only sent when partitioned cookies are enabled.
    cookie_partition_key: typing.Optional[CookiePartitionKey]
    #: True if partitioned cookies are enabled, but the partition key is not serializable to string.
    cookie_partition_key_opaque: typing.Optional[bool]
    #: A list of cookies which should have been blocked by 3PCD but are exempted and stored from
    #: the response with the corresponding reason.
    exempted_cookies: typing.Optional[typing.List[ExemptedSetCookieWithReason]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResponseReceivedExtraInfo:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            blocked_cookies=[BlockedSetCookieWithReason.from_json(i) for i in json['blockedCookies']],
            headers=Headers.from_json(json['headers']),
            resource_ip_address_space=IPAddressSpace.from_json(json['resourceIPAddressSpace']),
            status_code=int(json['statusCode']),
            headers_text=str(json['headersText']) if 'headersText' in json else None,
            cookie_partition_key=CookiePartitionKey.from_json(json['cookiePartitionKey']) if 'cookiePartitionKey' in json else None,
            cookie_partition_key_opaque=bool(json['cookiePartitionKeyOpaque']) if 'cookiePartitionKeyOpaque' in json else None,
            exempted_cookies=[ExemptedSetCookieWithReason.from_json(i) for i in json['exemptedCookies']] if 'exemptedCookies' in json else None
        )


@event_class('Network.responseReceivedEarlyHints')
@dataclass
class ResponseReceivedEarlyHints:
    '''
    **EXPERIMENTAL**

    Fired when 103 Early Hints headers is received in addition to the common response.
    Not every responseReceived event will have an responseReceivedEarlyHints fired.
    Only one responseReceivedEarlyHints may be fired for eached responseReceived event.
    '''
    #: Request identifier. Used to match this information to another responseReceived event.
    request_id: RequestId
    #: Raw response headers as they were received over the wire.
    #: Duplicate headers in the response are represented as a single key with their values
    #: concatentated using ``\n`` as the separator.
    #: See also ``headersText`` that contains verbatim text for HTTP/1.*.
    headers: Headers

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ResponseReceivedEarlyHints:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            headers=Headers.from_json(json['headers'])
        )


@event_class('Network.trustTokenOperationDone')
@dataclass
class TrustTokenOperationDone:
    '''
    **EXPERIMENTAL**

    Fired exactly once for each Trust Token operation. Depending on
    the type of the operation and whether the operation succeeded or
    failed, the event is fired before the corresponding request was sent
    or after the response was received.
    '''
    #: Detailed success or error status of the operation.
    #: 'AlreadyExists' also signifies a successful operation, as the result
    #: of the operation already exists und thus, the operation was abort
    #: preemptively (e.g. a cache hit).
    status: str
    type_: TrustTokenOperationType
    request_id: RequestId
    #: Top level origin. The context in which the operation was attempted.
    top_level_origin: typing.Optional[str]
    #: Origin of the issuer in case of a "Issuance" or "Redemption" operation.
    issuer_origin: typing.Optional[str]
    #: The number of obtained Trust Tokens on a successful "Issuance" operation.
    issued_token_count: typing.Optional[int]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> TrustTokenOperationDone:
        return cls(
            status=str(json['status']),
            type_=TrustTokenOperationType.from_json(json['type']),
            request_id=RequestId.from_json(json['requestId']),
            top_level_origin=str(json['topLevelOrigin']) if 'topLevelOrigin' in json else None,
            issuer_origin=str(json['issuerOrigin']) if 'issuerOrigin' in json else None,
            issued_token_count=int(json['issuedTokenCount']) if 'issuedTokenCount' in json else None
        )


@event_class('Network.policyUpdated')
@dataclass
class PolicyUpdated:
    '''
    **EXPERIMENTAL**

    Fired once security policy has been updated.
    '''


    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PolicyUpdated:
        return cls(

        )


@event_class('Network.subresourceWebBundleMetadataReceived')
@dataclass
class SubresourceWebBundleMetadataReceived:
    '''
    **EXPERIMENTAL**

    Fired once when parsing the .wbn file has succeeded.
    The event contains the information about the web bundle contents.
    '''
    #: Request identifier. Used to match this information to another event.
    request_id: RequestId
    #: A list of URLs of resources in the subresource Web Bundle.
    urls: typing.List[str]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleMetadataReceived:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            urls=[str(i) for i in json['urls']]
        )


@event_class('Network.subresourceWebBundleMetadataError')
@dataclass
class SubresourceWebBundleMetadataError:
    '''
    **EXPERIMENTAL**

    Fired once when parsing the .wbn file has failed.
    '''
    #: Request identifier. Used to match this information to another event.
    request_id: RequestId
    #: Error message
    error_message: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleMetadataError:
        return cls(
            request_id=RequestId.from_json(json['requestId']),
            error_message=str(json['errorMessage'])
        )


@event_class('Network.subresourceWebBundleInnerResponseParsed')
@dataclass
class SubresourceWebBundleInnerResponseParsed:
    '''
    **EXPERIMENTAL**

    Fired when handling requests for resources within a .wbn file.
    Note: this will only be fired for resources that are requested by the webpage.
    '''
    #: Request identifier of the subresource request
    inner_request_id: RequestId
    #: URL of the subresource resource.
    inner_request_url: str
    #: Bundle request identifier. Used to match this information to another event.
    #: This made be absent in case when the instrumentation was enabled only
    #: after webbundle was parsed.
    bundle_request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleInnerResponseParsed:
        return cls(
            inner_request_id=RequestId.from_json(json['innerRequestId']),
            inner_request_url=str(json['innerRequestURL']),
            bundle_request_id=RequestId.from_json(json['bundleRequestId']) if 'bundleRequestId' in json else None
        )


@event_class('Network.subresourceWebBundleInnerResponseError')
@dataclass
class SubresourceWebBundleInnerResponseError:
    '''
    **EXPERIMENTAL**

    Fired when request for resources within a .wbn file failed.
    '''
    #: Request identifier of the subresource request
    inner_request_id: RequestId
    #: URL of the subresource resource.
    inner_request_url: str
    #: Error message
    error_message: str
    #: Bundle request identifier. Used to match this information to another event.
    #: This made be absent in case when the instrumentation was enabled only
    #: after webbundle was parsed.
    bundle_request_id: typing.Optional[RequestId]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> SubresourceWebBundleInnerResponseError:
        return cls(
            inner_request_id=RequestId.from_json(json['innerRequestId']),
            inner_request_url=str(json['innerRequestURL']),
            error_message=str(json['errorMessage']),
            bundle_request_id=RequestId.from_json(json['bundleRequestId']) if 'bundleRequestId' in json else None
        )


@event_class('Network.reportingApiReportAdded')
@dataclass
class ReportingApiReportAdded:
    '''
    **EXPERIMENTAL**

    Is sent whenever a new report is added.
    And after 'enableReportingApi' for all existing reports.
    '''
    report: ReportingApiReport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportingApiReportAdded:
        return cls(
            report=ReportingApiReport.from_json(json['report'])
        )


@event_class('Network.reportingApiReportUpdated')
@dataclass
class ReportingApiReportUpdated:
    '''
    **EXPERIMENTAL**


    '''
    report: ReportingApiReport

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportingApiReportUpdated:
        return cls(
            report=ReportingApiReport.from_json(json['report'])
        )


@event_class('Network.reportingApiEndpointsChangedForOrigin')
@dataclass
class ReportingApiEndpointsChangedForOrigin:
    '''
    **EXPERIMENTAL**


    '''
    #: Origin of the document(s) which configured the endpoints.
    origin: str
    endpoints: typing.List[ReportingApiEndpoint]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> ReportingApiEndpointsChangedForOrigin:
        return cls(
            origin=str(json['origin']),
            endpoints=[ReportingApiEndpoint.from_json(i) for i in json['endpoints']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\fromnumeric.py ===
"""Module containing non-deprecated functions borrowed from Numeric.

"""
import functools
import types
import warnings

import numpy as np
from numpy._utils import set_module

from . import _methods, overrides
from . import multiarray as mu
from . import numerictypes as nt
from . import umath as um
from ._multiarray_umath import _array_converter
from .multiarray import asanyarray, asarray, concatenate

_dt_ = nt.sctype2char

# functions that are methods
__all__ = [
    'all', 'amax', 'amin', 'any', 'argmax',
    'argmin', 'argpartition', 'argsort', 'around', 'choose', 'clip',
    'compress', 'cumprod', 'cumsum', 'cumulative_prod', 'cumulative_sum',
    'diagonal', 'mean', 'max', 'min', 'matrix_transpose',
    'ndim', 'nonzero', 'partition', 'prod', 'ptp', 'put',
    'ravel', 'repeat', 'reshape', 'resize', 'round',
    'searchsorted', 'shape', 'size', 'sort', 'squeeze',
    'std', 'sum', 'swapaxes', 'take', 'trace', 'transpose', 'var',
]

_gentype = types.GeneratorType
# save away Python sum
_sum_ = sum

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


# functions that are now methods
def _wrapit(obj, method, *args, **kwds):
    conv = _array_converter(obj)
    # As this already tried the method, subok is maybe quite reasonable here
    # but this follows what was done before. TODO: revisit this.
    arr, = conv.as_arrays(subok=False)
    result = getattr(arr, method)(*args, **kwds)

    return conv.wrap(result, to_scalar=False)


def _wrapfunc(obj, method, *args, **kwds):
    bound = getattr(obj, method, None)
    if bound is None:
        return _wrapit(obj, method, *args, **kwds)

    try:
        return bound(*args, **kwds)
    except TypeError:
        # A TypeError occurs if the object does have such a method in its
        # class, but its signature is not identical to that of NumPy's. This
        # situation has occurred in the case of a downstream library like
        # 'pandas'.
        #
        # Call _wrapit from within the except clause to ensure a potential
        # exception has a traceback chain.
        return _wrapit(obj, method, *args, **kwds)


def _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs):
    passkwargs = {k: v for k, v in kwargs.items()
                  if v is not np._NoValue}

    if type(obj) is not mu.ndarray:
        try:
            reduction = getattr(obj, method)
        except AttributeError:
            pass
        else:
            # This branch is needed for reductions like any which don't
            # support a dtype.
            if dtype is not None:
                return reduction(axis=axis, dtype=dtype, out=out, **passkwargs)
            else:
                return reduction(axis=axis, out=out, **passkwargs)

    return ufunc.reduce(obj, axis, dtype, out, **passkwargs)


def _wrapreduction_any_all(obj, ufunc, method, axis, out, **kwargs):
    # Same as above function, but dtype is always bool (but never passed on)
    passkwargs = {k: v for k, v in kwargs.items()
                  if v is not np._NoValue}

    if type(obj) is not mu.ndarray:
        try:
            reduction = getattr(obj, method)
        except AttributeError:
            pass
        else:
            return reduction(axis=axis, out=out, **passkwargs)

    return ufunc.reduce(obj, axis, bool, out, **passkwargs)


def _take_dispatcher(a, indices, axis=None, out=None, mode=None):
    return (a, out)


@array_function_dispatch(_take_dispatcher)
def take(a, indices, axis=None, out=None, mode='raise'):
    """
    Take elements from an array along an axis.

    When axis is not None, this function does the same thing as "fancy"
    indexing (indexing arrays using arrays); however, it can be easier to use
    if you need elements along a given axis. A call such as
    ``np.take(arr, indices, axis=3)`` is equivalent to
    ``arr[:,:,:,indices,...]``.

    Explained without fancy indexing, this is equivalent to the following use
    of `ndindex`, which sets each of ``ii``, ``jj``, and ``kk`` to a tuple of
    indices::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        Nj = indices.shape
        for ii in ndindex(Ni):
            for jj in ndindex(Nj):
                for kk in ndindex(Nk):
                    out[ii + jj + kk] = a[ii + (indices[jj],) + kk]

    Parameters
    ----------
    a : array_like (Ni..., M, Nk...)
        The source array.
    indices : array_like (Nj...)
        The indices of the values to extract.
        Also allow scalars for indices.
    axis : int, optional
        The axis over which to select values. By default, the flattened
        input array is used.
    out : ndarray, optional (Ni..., Nj..., Nk...)
        If provided, the result will be placed in this array. It should
        be of the appropriate shape and dtype. Note that `out` is always
        buffered if `mode='raise'`; use other modes for better performance.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers.

    Returns
    -------
    out : ndarray (Ni..., Nj..., Nk...)
        The returned array has the same type as `a`.

    See Also
    --------
    compress : Take elements using a boolean mask
    ndarray.take : equivalent method
    take_along_axis : Take elements by matching the array and the index arrays

    Notes
    -----
    By eliminating the inner loop in the description above, and using `s_` to
    build simple slice objects, `take` can be expressed  in terms of applying
    fancy indexing to each 1-d slice::

        Ni, Nk = a.shape[:axis], a.shape[axis+1:]
        for ii in ndindex(Ni):
            for kk in ndindex(Nj):
                out[ii + s_[...,] + kk] = a[ii + s_[:,] + kk][indices]

    For this reason, it is equivalent to (but faster than) the following use
    of `apply_along_axis`::

        out = np.apply_along_axis(lambda a_1d: a_1d[indices], axis, a)

    Examples
    --------
    >>> import numpy as np
    >>> a = [4, 3, 5, 7, 6, 8]
    >>> indices = [0, 1, 4]
    >>> np.take(a, indices)
    array([4, 3, 6])

    In this example if `a` is an ndarray, "fancy" indexing can be used.

    >>> a = np.array(a)
    >>> a[indices]
    array([4, 3, 6])

    If `indices` is not one dimensional, the output also has these dimensions.

    >>> np.take(a, [[0, 1], [2, 3]])
    array([[4, 3],
           [5, 7]])
    """
    return _wrapfunc(a, 'take', indices, axis=axis, out=out, mode=mode)


def _reshape_dispatcher(a, /, shape=None, order=None, *, newshape=None,
                        copy=None):
    return (a,)


@array_function_dispatch(_reshape_dispatcher)
def reshape(a, /, shape=None, order='C', *, newshape=None, copy=None):
    """
    Gives a new shape to an array without changing its data.

    Parameters
    ----------
    a : array_like
        Array to be reshaped.
    shape : int or tuple of ints
        The new shape should be compatible with the original shape. If
        an integer, then the result will be a 1-D array of that length.
        One shape dimension can be -1. In this case, the value is
        inferred from the length of the array and remaining dimensions.
    order : {'C', 'F', 'A'}, optional
        Read the elements of ``a`` using this index order, and place the
        elements into the reshaped array using this index order. 'C'
        means to read / write the elements using C-like index order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest. 'F' means to read / write the
        elements using Fortran-like index order, with the first index
        changing fastest, and the last index changing slowest. Note that
        the 'C' and 'F' options take no account of the memory layout of
        the underlying array, and only refer to the order of indexing.
        'A' means to read / write the elements in Fortran-like index
        order if ``a`` is Fortran *contiguous* in memory, C-like order
        otherwise.
    newshape : int or tuple of ints
        .. deprecated:: 2.1
            Replaced by ``shape`` argument. Retained for backward
            compatibility.
    copy : bool, optional
        If ``True``, then the array data is copied. If ``None``, a copy will
        only be made if it's required by ``order``. For ``False`` it raises
        a ``ValueError`` if a copy cannot be avoided. Default: ``None``.

    Returns
    -------
    reshaped_array : ndarray
        This will be a new view object if possible; otherwise, it will
        be a copy.  Note there is no guarantee of the *memory layout* (C- or
        Fortran- contiguous) of the returned array.

    See Also
    --------
    ndarray.reshape : Equivalent method.

    Notes
    -----
    It is not always possible to change the shape of an array without copying
    the data.

    The ``order`` keyword gives the index ordering both for *fetching*
    the values from ``a``, and then *placing* the values into the output
    array. For example, let's say you have an array:

    >>> a = np.arange(6).reshape((3, 2))
    >>> a
    array([[0, 1],
           [2, 3],
           [4, 5]])

    You can think of reshaping as first raveling the array (using the given
    index order), then inserting the elements from the raveled array into the
    new array using the same kind of index ordering as was used for the
    raveling.

    >>> np.reshape(a, (2, 3)) # C-like index ordering
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(np.ravel(a), (2, 3)) # equivalent to C ravel then C reshape
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> np.reshape(a, (2, 3), order='F') # Fortran-like index ordering
    array([[0, 4, 3],
           [2, 1, 5]])
    >>> np.reshape(np.ravel(a, order='F'), (2, 3), order='F')
    array([[0, 4, 3],
           [2, 1, 5]])

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> np.reshape(a, 6)
    array([1, 2, 3, 4, 5, 6])
    >>> np.reshape(a, 6, order='F')
    array([1, 4, 2, 5, 3, 6])

    >>> np.reshape(a, (3,-1))       # the unspecified value is inferred to be 2
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """
    if newshape is None and shape is None:
        raise TypeError(
            "reshape() missing 1 required positional argument: 'shape'")
    if newshape is not None:
        if shape is not None:
            raise TypeError(
                "You cannot specify 'newshape' and 'shape' arguments "
                "at the same time.")
        # Deprecated in NumPy 2.1, 2024-04-18
        warnings.warn(
            "`newshape` keyword argument is deprecated, "
            "use `shape=...` or pass shape positionally instead. "
            "(deprecated in NumPy 2.1)",
            DeprecationWarning,
            stacklevel=2,
        )
        shape = newshape
    if copy is not None:
        return _wrapfunc(a, 'reshape', shape, order=order, copy=copy)
    return _wrapfunc(a, 'reshape', shape, order=order)


def _choose_dispatcher(a, choices, out=None, mode=None):
    yield a
    yield from choices
    yield out


@array_function_dispatch(_choose_dispatcher)
def choose(a, choices, out=None, mode='raise'):
    """
    Construct an array from an index array and a list of arrays to choose from.

    First of all, if confused or uncertain, definitely look at the Examples -
    in its full generality, this function is less simple than it might
    seem from the following code description::

        np.choose(a,c) == np.array([c[a[I]][I] for I in np.ndindex(a.shape)])

    But this omits some subtleties.  Here is a fully general summary:

    Given an "index" array (`a`) of integers and a sequence of ``n`` arrays
    (`choices`), `a` and each choice array are first broadcast, as necessary,
    to arrays of a common shape; calling these *Ba* and *Bchoices[i], i =
    0,...,n-1* we have that, necessarily, ``Ba.shape == Bchoices[i].shape``
    for each ``i``.  Then, a new array with shape ``Ba.shape`` is created as
    follows:

    * if ``mode='raise'`` (the default), then, first of all, each element of
      ``a`` (and thus ``Ba``) must be in the range ``[0, n-1]``; now, suppose
      that ``i`` (in that range) is the value at the ``(j0, j1, ..., jm)``
      position in ``Ba`` - then the value at the same position in the new array
      is the value in ``Bchoices[i]`` at that same position;

    * if ``mode='wrap'``, values in `a` (and thus `Ba`) may be any (signed)
      integer; modular arithmetic is used to map integers outside the range
      `[0, n-1]` back into that range; and then the new array is constructed
      as above;

    * if ``mode='clip'``, values in `a` (and thus ``Ba``) may be any (signed)
      integer; negative integers are mapped to 0; values greater than ``n-1``
      are mapped to ``n-1``; and then the new array is constructed as above.

    Parameters
    ----------
    a : int array
        This array must contain integers in ``[0, n-1]``, where ``n`` is the
        number of choices, unless ``mode=wrap`` or ``mode=clip``, in which
        cases any integers are permissible.
    choices : sequence of arrays
        Choice arrays. `a` and all of the choices must be broadcastable to the
        same shape.  If `choices` is itself an array (not recommended), then
        its outermost dimension (i.e., the one corresponding to
        ``choices.shape[0]``) is taken as defining the "sequence".
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype. Note that `out` is always
        buffered if ``mode='raise'``; use other modes for better performance.
    mode : {'raise' (default), 'wrap', 'clip'}, optional
        Specifies how indices outside ``[0, n-1]`` will be treated:

        * 'raise' : an exception is raised
        * 'wrap' : value becomes value mod ``n``
        * 'clip' : values < 0 are mapped to 0, values > n-1 are mapped to n-1

    Returns
    -------
    merged_array : array
        The merged result.

    Raises
    ------
    ValueError: shape mismatch
        If `a` and each choice array are not all broadcastable to the same
        shape.

    See Also
    --------
    ndarray.choose : equivalent method
    numpy.take_along_axis : Preferable if `choices` is an array

    Notes
    -----
    To reduce the chance of misinterpretation, even though the following
    "abuse" is nominally supported, `choices` should neither be, nor be
    thought of as, a single array, i.e., the outermost sequence-like container
    should be either a list or a tuple.

    Examples
    --------

    >>> import numpy as np
    >>> choices = [[0, 1, 2, 3], [10, 11, 12, 13],
    ...   [20, 21, 22, 23], [30, 31, 32, 33]]
    >>> np.choose([2, 3, 1, 0], choices
    ... # the first element of the result will be the first element of the
    ... # third (2+1) "array" in choices, namely, 20; the second element
    ... # will be the second element of the fourth (3+1) choice array, i.e.,
    ... # 31, etc.
    ... )
    array([20, 31, 12,  3])
    >>> np.choose([2, 4, 1, 0], choices, mode='clip') # 4 goes to 3 (4-1)
    array([20, 31, 12,  3])
    >>> # because there are 4 choice arrays
    >>> np.choose([2, 4, 1, 0], choices, mode='wrap') # 4 goes to (4 mod 4)
    array([20,  1, 12,  3])
    >>> # i.e., 0

    A couple examples illustrating how choose broadcasts:

    >>> a = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
    >>> choices = [-10, 10]
    >>> np.choose(a, choices)
    array([[ 10, -10,  10],
           [-10,  10, -10],
           [ 10, -10,  10]])

    >>> # With thanks to Anne Archibald
    >>> a = np.array([0, 1]).reshape((2,1,1))
    >>> c1 = np.array([1, 2, 3]).reshape((1,3,1))
    >>> c2 = np.array([-1, -2, -3, -4, -5]).reshape((1,1,5))
    >>> np.choose(a, (c1, c2)) # result is 2x3x5, res[0,:,:]=c1, res[1,:,:]=c2
    array([[[ 1,  1,  1,  1,  1],
            [ 2,  2,  2,  2,  2],
            [ 3,  3,  3,  3,  3]],
           [[-1, -2, -3, -4, -5],
            [-1, -2, -3, -4, -5],
            [-1, -2, -3, -4, -5]]])

    """
    return _wrapfunc(a, 'choose', choices, out=out, mode=mode)


def _repeat_dispatcher(a, repeats, axis=None):
    return (a,)


@array_function_dispatch(_repeat_dispatcher)
def repeat(a, repeats, axis=None):
    """
    Repeat each element of an array after themselves

    Parameters
    ----------
    a : array_like
        Input array.
    repeats : int or array of ints
        The number of repetitions for each element.  `repeats` is broadcasted
        to fit the shape of the given axis.
    axis : int, optional
        The axis along which to repeat values.  By default, use the
        flattened input array, and return a flat output array.

    Returns
    -------
    repeated_array : ndarray
        Output array which has the same shape as `a`, except along
        the given axis.

    See Also
    --------
    tile : Tile an array.
    unique : Find the unique elements of an array.

    Examples
    --------
    >>> import numpy as np
    >>> np.repeat(3, 4)
    array([3, 3, 3, 3])
    >>> x = np.array([[1,2],[3,4]])
    >>> np.repeat(x, 2)
    array([1, 1, 2, 2, 3, 3, 4, 4])
    >>> np.repeat(x, 3, axis=1)
    array([[1, 1, 1, 2, 2, 2],
           [3, 3, 3, 4, 4, 4]])
    >>> np.repeat(x, [1, 2], axis=0)
    array([[1, 2],
           [3, 4],
           [3, 4]])

    """
    return _wrapfunc(a, 'repeat', repeats, axis=axis)


def _put_dispatcher(a, ind, v, mode=None):
    return (a, ind, v)


@array_function_dispatch(_put_dispatcher)
def put(a, ind, v, mode='raise'):
    """
    Replaces specified elements of an array with given values.

    The indexing works on the flattened target array. `put` is roughly
    equivalent to:

    ::

        a.flat[ind] = v

    Parameters
    ----------
    a : ndarray
        Target array.
    ind : array_like
        Target indices, interpreted as integers.
    v : array_like
        Values to place in `a` at target indices. If `v` is shorter than
        `ind` it will be repeated as necessary.
    mode : {'raise', 'wrap', 'clip'}, optional
        Specifies how out-of-bounds indices will behave.

        * 'raise' -- raise an error (default)
        * 'wrap' -- wrap around
        * 'clip' -- clip to the range

        'clip' mode means that all indices that are too large are replaced
        by the index that addresses the last element along that axis. Note
        that this disables indexing with negative numbers. In 'raise' mode,
        if an exception occurs the target array may still be modified.

    See Also
    --------
    putmask, place
    put_along_axis : Put elements by matching the array and the index arrays

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(5)
    >>> np.put(a, [0, 2], [-44, -55])
    >>> a
    array([-44,   1, -55,   3,   4])

    >>> a = np.arange(5)
    >>> np.put(a, 22, -5, mode='clip')
    >>> a
    array([ 0,  1,  2,  3, -5])

    """
    try:
        put = a.put
    except AttributeError as e:
        raise TypeError(f"argument 1 must be numpy.ndarray, not {type(a)}") from e

    return put(ind, v, mode=mode)


def _swapaxes_dispatcher(a, axis1, axis2):
    return (a,)


@array_function_dispatch(_swapaxes_dispatcher)
def swapaxes(a, axis1, axis2):
    """
    Interchange two axes of an array.

    Parameters
    ----------
    a : array_like
        Input array.
    axis1 : int
        First axis.
    axis2 : int
        Second axis.

    Returns
    -------
    a_swapped : ndarray
        For NumPy >= 1.10.0, if `a` is an ndarray, then a view of `a` is
        returned; otherwise a new array is created. For earlier NumPy
        versions a view of `a` is returned only if the order of the
        axes is changed, otherwise the input array is returned.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[1,2,3]])
    >>> np.swapaxes(x,0,1)
    array([[1],
           [2],
           [3]])

    >>> x = np.array([[[0,1],[2,3]],[[4,5],[6,7]]])
    >>> x
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])

    >>> np.swapaxes(x,0,2)
    array([[[0, 4],
            [2, 6]],
           [[1, 5],
            [3, 7]]])

    """
    return _wrapfunc(a, 'swapaxes', axis1, axis2)


def _transpose_dispatcher(a, axes=None):
    return (a,)


@array_function_dispatch(_transpose_dispatcher)
def transpose(a, axes=None):
    """
    Returns an array with axes transposed.

    For a 1-D array, this returns an unchanged view of the original array, as a
    transposed vector is simply the same vector.
    To convert a 1-D array into a 2-D column vector, an additional dimension
    must be added, e.g., ``np.atleast_2d(a).T`` achieves this, as does
    ``a[:, np.newaxis]``.
    For a 2-D array, this is the standard matrix transpose.
    For an n-D array, if axes are given, their order indicates how the
    axes are permuted (see Examples). If axes are not provided, then
    ``transpose(a).shape == a.shape[::-1]``.

    Parameters
    ----------
    a : array_like
        Input array.
    axes : tuple or list of ints, optional
        If specified, it must be a tuple or list which contains a permutation
        of [0, 1, ..., N-1] where N is the number of axes of `a`. Negative
        indices can also be used to specify axes. The i-th axis of the returned
        array will correspond to the axis numbered ``axes[i]`` of the input.
        If not specified, defaults to ``range(a.ndim)[::-1]``, which reverses
        the order of the axes.

    Returns
    -------
    p : ndarray
        `a` with its axes permuted. A view is returned whenever possible.

    See Also
    --------
    ndarray.transpose : Equivalent method.
    moveaxis : Move axes of an array to new positions.
    argsort : Return the indices that would sort an array.

    Notes
    -----
    Use ``transpose(a, argsort(axes))`` to invert the transposition of tensors
    when using the `axes` keyword argument.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> a
    array([[1, 2],
           [3, 4]])
    >>> np.transpose(a)
    array([[1, 3],
           [2, 4]])

    >>> a = np.array([1, 2, 3, 4])
    >>> a
    array([1, 2, 3, 4])
    >>> np.transpose(a)
    array([1, 2, 3, 4])

    >>> a = np.ones((1, 2, 3))
    >>> np.transpose(a, (1, 0, 2)).shape
    (2, 1, 3)

    >>> a = np.ones((2, 3, 4, 5))
    >>> np.transpose(a).shape
    (5, 4, 3, 2)

    >>> a = np.arange(3*4*5).reshape((3, 4, 5))
    >>> np.transpose(a, (-1, 0, -2)).shape
    (5, 3, 4)

    """
    return _wrapfunc(a, 'transpose', axes)


def _matrix_transpose_dispatcher(x):
    return (x,)

@array_function_dispatch(_matrix_transpose_dispatcher)
def matrix_transpose(x, /):
    """
    Transposes a matrix (or a stack of matrices) ``x``.

    This function is Array API compatible.

    Parameters
    ----------
    x : array_like
        Input array having shape (..., M, N) and whose two innermost
        dimensions form ``MxN`` matrices.

    Returns
    -------
    out : ndarray
        An array containing the transpose for each matrix and having shape
        (..., N, M).

    See Also
    --------
    transpose : Generic transpose method.

    Examples
    --------
    >>> import numpy as np
    >>> np.matrix_transpose([[1, 2], [3, 4]])
    array([[1, 3],
           [2, 4]])

    >>> np.matrix_transpose([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    array([[[1, 3],
            [2, 4]],
           [[5, 7],
            [6, 8]]])

    """
    x = asanyarray(x)
    if x.ndim < 2:
        raise ValueError(
            f"Input array must be at least 2-dimensional, but it is {x.ndim}"
        )
    return swapaxes(x, -1, -2)


def _partition_dispatcher(a, kth, axis=None, kind=None, order=None):
    return (a,)


@array_function_dispatch(_partition_dispatcher)
def partition(a, kth, axis=-1, kind='introselect', order=None):
    """
    Return a partitioned copy of an array.

    Creates a copy of the array and partially sorts it in such a way that
    the value of the element in k-th position is in the position it would be
    in a sorted array. In the output array, all elements smaller than the k-th
    element are located to the left of this element and all equal or greater
    are located to its right. The ordering of the elements in the two
    partitions on the either side of the k-th element in the output array is
    undefined.

    Parameters
    ----------
    a : array_like
        Array to be sorted.
    kth : int or sequence of ints
        Element index to partition by. The k-th value of the element
        will be in its final sorted position and all smaller elements
        will be moved before it and all equal or greater elements behind
        it. The order of all elements in the partitions is undefined. If
        provided with a sequence of k-th it will partition all elements
        indexed by k-th  of them into their sorted position at once.

        .. deprecated:: 1.22.0
            Passing booleans as index is deprecated.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument
        specifies which fields to compare first, second, etc.  A single
        field can be specified as a string.  Not all fields need be
        specified, but unspecified fields will still be used, in the
        order in which they come up in the dtype, to break ties.

    Returns
    -------
    partitioned_array : ndarray
        Array of the same type and shape as `a`.

    See Also
    --------
    ndarray.partition : Method to sort an array in-place.
    argpartition : Indirect partition.
    sort : Full sorting

    Notes
    -----
    The various selection algorithms are characterized by their average
    speed, worst case performance, work space size, and whether they are
    stable. A stable sort keeps items with the same key in the same
    relative order. The available algorithms have the following
    properties:

    ================= ======= ============= ============ =======
       kind            speed   worst case    work space  stable
    ================= ======= ============= ============ =======
    'introselect'        1        O(n)           0         no
    ================= ======= ============= ============ =======

    All the partition algorithms make temporary copies of the data when
    partitioning along any but the last axis.  Consequently,
    partitioning along the last axis is faster and uses less space than
    partitioning along any other axis.

    The sort order for complex numbers is lexicographic. If both the
    real and imaginary parts are non-nan then the order is determined by
    the real parts except when they are equal, in which case the order
    is determined by the imaginary parts.

    The sort order of ``np.nan`` is bigger than ``np.inf``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([7, 1, 7, 7, 1, 5, 7, 2, 3, 2, 6, 2, 3, 0])
    >>> p = np.partition(a, 4)
    >>> p
    array([0, 1, 2, 1, 2, 5, 2, 3, 3, 6, 7, 7, 7, 7]) # may vary

    ``p[4]`` is 2;  all elements in ``p[:4]`` are less than or equal
    to ``p[4]``, and all elements in ``p[5:]`` are greater than or
    equal to ``p[4]``.  The partition is::

        [0, 1, 2, 1], [2], [5, 2, 3, 3, 6, 7, 7, 7, 7]

    The next example shows the use of multiple values passed to `kth`.

    >>> p2 = np.partition(a, (4, 8))
    >>> p2
    array([0, 1, 2, 1, 2, 3, 3, 2, 5, 6, 7, 7, 7, 7])

    ``p2[4]`` is 2  and ``p2[8]`` is 5.  All elements in ``p2[:4]``
    are less than or equal to ``p2[4]``, all elements in ``p2[5:8]``
    are greater than or equal to ``p2[4]`` and less than or equal to
    ``p2[8]``, and all elements in ``p2[9:]`` are greater than or
    equal to ``p2[8]``.  The partition is::

        [0, 1, 2, 1], [2], [3, 3, 2], [5], [6, 7, 7, 7, 7]
    """
    if axis is None:
        # flatten returns (1, N) for np.matrix, so always use the last axis
        a = asanyarray(a).flatten()
        axis = -1
    else:
        a = asanyarray(a).copy(order="K")
    a.partition(kth, axis=axis, kind=kind, order=order)
    return a


def _argpartition_dispatcher(a, kth, axis=None, kind=None, order=None):
    return (a,)


@array_function_dispatch(_argpartition_dispatcher)
def argpartition(a, kth, axis=-1, kind='introselect', order=None):
    """
    Perform an indirect partition along the given axis using the
    algorithm specified by the `kind` keyword. It returns an array of
    indices of the same shape as `a` that index data along the given
    axis in partitioned order.

    Parameters
    ----------
    a : array_like
        Array to sort.
    kth : int or sequence of ints
        Element index to partition by. The k-th element will be in its
        final sorted position and all smaller elements will be moved
        before it and all larger elements behind it. The order of all
        elements in the partitions is undefined. If provided with a
        sequence of k-th it will partition all of them into their sorted
        position at once.

        .. deprecated:: 1.22.0
            Passing booleans as index is deprecated.
    axis : int or None, optional
        Axis along which to sort. The default is -1 (the last axis). If
        None, the flattened array is used.
    kind : {'introselect'}, optional
        Selection algorithm. Default is 'introselect'
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument
        specifies which fields to compare first, second, etc. A single
        field can be specified as a string, and not all fields need be
        specified, but unspecified fields will still be used, in the
        order in which they come up in the dtype, to break ties.

    Returns
    -------
    index_array : ndarray, int
        Array of indices that partition `a` along the specified axis.
        If `a` is one-dimensional, ``a[index_array]`` yields a partitioned `a`.
        More generally, ``np.take_along_axis(a, index_array, axis=axis)``
        always yields the partitioned `a`, irrespective of dimensionality.

    See Also
    --------
    partition : Describes partition algorithms used.
    ndarray.partition : Inplace partition.
    argsort : Full indirect sort.
    take_along_axis : Apply ``index_array`` from argpartition
                      to an array as if by calling partition.

    Notes
    -----
    The returned indices are not guaranteed to be sorted according to
    the values. Furthermore, the default selection algorithm ``introselect``
    is unstable, and hence the returned indices are not guaranteed
    to be the earliest/latest occurrence of the element.

    `argpartition` works for real/complex inputs with nan values,
    see `partition` for notes on the enhanced sort order and
    different selection algorithms.

    Examples
    --------
    One dimensional array:

    >>> import numpy as np
    >>> x = np.array([3, 4, 2, 1])
    >>> x[np.argpartition(x, 3)]
    array([2, 1, 3, 4]) # may vary
    >>> x[np.argpartition(x, (1, 3))]
    array([1, 2, 3, 4]) # may vary

    >>> x = [3, 4, 2, 1]
    >>> np.array(x)[np.argpartition(x, 3)]
    array([2, 1, 3, 4]) # may vary

    Multi-dimensional array:

    >>> x = np.array([[3, 4, 2], [1, 3, 1]])
    >>> index_array = np.argpartition(x, kth=1, axis=-1)
    >>> # below is the same as np.partition(x, kth=1)
    >>> np.take_along_axis(x, index_array, axis=-1)
    array([[2, 3, 4],
           [1, 1, 3]])

    """
    return _wrapfunc(a, 'argpartition', kth, axis=axis, kind=kind, order=order)


def _sort_dispatcher(a, axis=None, kind=None, order=None, *, stable=None):
    return (a,)


@array_function_dispatch(_sort_dispatcher)
def sort(a, axis=-1, kind=None, order=None, *, stable=None):
    """
    Return a sorted copy of an array.

    Parameters
    ----------
    a : array_like
        Array to be sorted.
    axis : int or None, optional
        Axis along which to sort. If None, the array is flattened before
        sorting. The default is -1, which sorts along the last axis.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort or radix sort under the covers and,
        in general, the actual implementation will vary with data type.
        The 'mergesort' option is retained for backwards compatibility.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    stable : bool, optional
        Sort stability. If ``True``, the returned array will maintain
        the relative order of ``a`` values which compare as equal.
        If ``False`` or ``None``, this is not guaranteed. Internally,
        this option selects ``kind='stable'``. Default: ``None``.

        .. versionadded:: 2.0.0

    Returns
    -------
    sorted_array : ndarray
        Array of the same type and shape as `a`.

    See Also
    --------
    ndarray.sort : Method to sort an array in-place.
    argsort : Indirect sort.
    lexsort : Indirect stable sort on multiple keys.
    searchsorted : Find elements in a sorted array.
    partition : Partial sort.

    Notes
    -----
    The various sorting algorithms are characterized by their average speed,
    worst case performance, work space size, and whether they are stable. A
    stable sort keeps items with the same key in the same relative
    order. The four algorithms implemented in NumPy have the following
    properties:

    =========== ======= ============= ============ ========
       kind      speed   worst case    work space   stable
    =========== ======= ============= ============ ========
    'quicksort'    1     O(n^2)            0          no
    'heapsort'     3     O(n*log(n))       0          no
    'mergesort'    2     O(n*log(n))      ~n/2        yes
    'timsort'      2     O(n*log(n))      ~n/2        yes
    =========== ======= ============= ============ ========

    .. note:: The datatype determines which of 'mergesort' or 'timsort'
       is actually used, even if 'mergesort' is specified. User selection
       at a finer scale is not currently available.

    For performance, ``sort`` makes a temporary copy if needed to make the data
    `contiguous <https://numpy.org/doc/stable/glossary.html#term-contiguous>`_
    in memory along the sort axis. For even better performance and reduced
    memory consumption, ensure that the array is already contiguous along the
    sort axis.

    The sort order for complex numbers is lexicographic. If both the real
    and imaginary parts are non-nan then the order is determined by the
    real parts except when they are equal, in which case the order is
    determined by the imaginary parts.

    Previous to numpy 1.4.0 sorting real and complex arrays containing nan
    values led to undefined behaviour. In numpy versions >= 1.4.0 nan
    values are sorted to the end. The extended sort order is:

      * Real: [R, nan]
      * Complex: [R + Rj, R + nanj, nan + Rj, nan + nanj]

    where R is a non-nan real value. Complex values with the same nan
    placements are sorted according to the non-nan part if it exists.
    Non-nan values are sorted as before.

    quicksort has been changed to:
    `introsort <https://en.wikipedia.org/wiki/Introsort>`_.
    When sorting does not make enough progress it switches to
    `heapsort <https://en.wikipedia.org/wiki/Heapsort>`_.
    This implementation makes quicksort O(n*log(n)) in the worst case.

    'stable' automatically chooses the best stable sorting algorithm
    for the data type being sorted.
    It, along with 'mergesort' is currently mapped to
    `timsort <https://en.wikipedia.org/wiki/Timsort>`_
    or `radix sort <https://en.wikipedia.org/wiki/Radix_sort>`_
    depending on the data type.
    API forward compatibility currently limits the
    ability to select the implementation and it is hardwired for the different
    data types.

    Timsort is added for better performance on already or nearly
    sorted data. On random data timsort is almost identical to
    mergesort. It is now used for stable sort while quicksort is still the
    default sort if none is chosen. For timsort details, refer to
    `CPython listsort.txt
    <https://github.com/python/cpython/blob/3.7/Objects/listsort.txt>`_
    'mergesort' and 'stable' are mapped to radix sort for integer data types.
    Radix sort is an O(n) sort instead of O(n log n).

    NaT now sorts to the end of arrays for consistency with NaN.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,4],[3,1]])
    >>> np.sort(a)                # sort along the last axis
    array([[1, 4],
           [1, 3]])
    >>> np.sort(a, axis=None)     # sort the flattened array
    array([1, 1, 3, 4])
    >>> np.sort(a, axis=0)        # sort along the first axis
    array([[1, 1],
           [3, 4]])

    Use the `order` keyword to specify a field to use when sorting a
    structured array:

    >>> dtype = [('name', 'S10'), ('height', float), ('age', int)]
    >>> values = [('Arthur', 1.8, 41), ('Lancelot', 1.9, 38),
    ...           ('Galahad', 1.7, 38)]
    >>> a = np.array(values, dtype=dtype)       # create a structured array
    >>> np.sort(a, order='height')                        # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Arthur', 1.8, 41),
           ('Lancelot', 1.8999999999999999, 38)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

    Sort by age, then height if ages are equal:

    >>> np.sort(a, order=['age', 'height'])               # doctest: +SKIP
    array([('Galahad', 1.7, 38), ('Lancelot', 1.8999999999999999, 38),
           ('Arthur', 1.8, 41)],
          dtype=[('name', '|S10'), ('height', '<f8'), ('age', '<i4')])

    """
    if axis is None:
        # flatten returns (1, N) for np.matrix, so always use the last axis
        a = asanyarray(a).flatten()
        axis = -1
    else:
        a = asanyarray(a).copy(order="K")
    a.sort(axis=axis, kind=kind, order=order, stable=stable)
    return a


def _argsort_dispatcher(a, axis=None, kind=None, order=None, *, stable=None):
    return (a,)


@array_function_dispatch(_argsort_dispatcher)
def argsort(a, axis=-1, kind=None, order=None, *, stable=None):
    """
    Returns the indices that would sort an array.

    Perform an indirect sort along the given axis using the algorithm specified
    by the `kind` keyword. It returns an array of indices of the same shape as
    `a` that index data along the given axis in sorted order.

    Parameters
    ----------
    a : array_like
        Array to sort.
    axis : int or None, optional
        Axis along which to sort.  The default is -1 (the last axis). If None,
        the flattened array is used.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is 'quicksort'. Note that both 'stable'
        and 'mergesort' use timsort under the covers and, in general, the
        actual implementation will vary with data type. The 'mergesort' option
        is retained for backwards compatibility.
    order : str or list of str, optional
        When `a` is an array with fields defined, this argument specifies
        which fields to compare first, second, etc.  A single field can
        be specified as a string, and not all fields need be specified,
        but unspecified fields will still be used, in the order in which
        they come up in the dtype, to break ties.
    stable : bool, optional
        Sort stability. If ``True``, the returned array will maintain
        the relative order of ``a`` values which compare as equal.
        If ``False`` or ``None``, this is not guaranteed. Internally,
        this option selects ``kind='stable'``. Default: ``None``.

        .. versionadded:: 2.0.0

    Returns
    -------
    index_array : ndarray, int
        Array of indices that sort `a` along the specified `axis`.
        If `a` is one-dimensional, ``a[index_array]`` yields a sorted `a`.
        More generally, ``np.take_along_axis(a, index_array, axis=axis)``
        always yields the sorted `a`, irrespective of dimensionality.

    See Also
    --------
    sort : Describes sorting algorithms used.
    lexsort : Indirect stable sort with multiple keys.
    ndarray.sort : Inplace sort.
    argpartition : Indirect partial sort.
    take_along_axis : Apply ``index_array`` from argsort
                      to an array as if by calling sort.

    Notes
    -----
    See `sort` for notes on the different sorting algorithms.

    As of NumPy 1.4.0 `argsort` works with real/complex arrays containing
    nan values. The enhanced sort order is documented in `sort`.

    Examples
    --------
    One dimensional array:

    >>> import numpy as np
    >>> x = np.array([3, 1, 2])
    >>> np.argsort(x)
    array([1, 2, 0])

    Two-dimensional array:

    >>> x = np.array([[0, 3], [2, 2]])
    >>> x
    array([[0, 3],
           [2, 2]])

    >>> ind = np.argsort(x, axis=0)  # sorts along first axis (down)
    >>> ind
    array([[0, 1],
           [1, 0]])
    >>> np.take_along_axis(x, ind, axis=0)  # same as np.sort(x, axis=0)
    array([[0, 2],
           [2, 3]])

    >>> ind = np.argsort(x, axis=1)  # sorts along last axis (across)
    >>> ind
    array([[0, 1],
           [0, 1]])
    >>> np.take_along_axis(x, ind, axis=1)  # same as np.sort(x, axis=1)
    array([[0, 3],
           [2, 2]])

    Indices of the sorted elements of a N-dimensional array:

    >>> ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
    >>> ind
    (array([0, 1, 1, 0]), array([0, 0, 1, 1]))
    >>> x[ind]  # same as np.sort(x, axis=None)
    array([0, 2, 2, 3])

    Sorting with keys:

    >>> x = np.array([(1, 0), (0, 1)], dtype=[('x', '<i4'), ('y', '<i4')])
    >>> x
    array([(1, 0), (0, 1)],
          dtype=[('x', '<i4'), ('y', '<i4')])

    >>> np.argsort(x, order=('x','y'))
    array([1, 0])

    >>> np.argsort(x, order=('y','x'))
    array([0, 1])

    """
    return _wrapfunc(
        a, 'argsort', axis=axis, kind=kind, order=order, stable=stable
    )

def _argmax_dispatcher(a, axis=None, out=None, *, keepdims=np._NoValue):
    return (a, out)


@array_function_dispatch(_argmax_dispatcher)
def argmax(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    Returns the indices of the maximum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as ``a.shape``
        with the dimension along `axis` removed. If `keepdims` is set to True,
        then the size of `axis` will be 1 with the resulting array having same
        shape as ``a.shape``.

    See Also
    --------
    ndarray.argmax, argmin
    amax : The maximum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.
    take_along_axis : Apply ``np.expand_dims(index_array, axis)``
                      from argmax to an array as if by calling max.

    Notes
    -----
    In case of multiple occurrences of the maximum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(6).reshape(2,3) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> np.argmax(a)
    5
    >>> np.argmax(a, axis=0)
    array([1, 1, 1])
    >>> np.argmax(a, axis=1)
    array([2, 2])

    Indexes of the maximal elements of a N-dimensional array:

    >>> ind = np.unravel_index(np.argmax(a, axis=None), a.shape)
    >>> ind
    (1, 2)
    >>> a[ind]
    15

    >>> b = np.arange(6)
    >>> b[1] = 5
    >>> b
    array([0, 5, 2, 3, 4, 5])
    >>> np.argmax(b)  # Only the first occurrence is returned.
    1

    >>> x = np.array([[4,2,3], [1,0,3]])
    >>> index_array = np.argmax(x, axis=-1)
    >>> # Same as np.amax(x, axis=-1, keepdims=True)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
    array([[4],
           [3]])
    >>> # Same as np.amax(x, axis=-1)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1),
    ...     axis=-1).squeeze(axis=-1)
    array([4, 3])

    Setting `keepdims` to `True`,

    >>> x = np.arange(24).reshape((2, 3, 4))
    >>> res = np.argmax(x, axis=1, keepdims=True)
    >>> res.shape
    (2, 1, 4)
    """
    kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
    return _wrapfunc(a, 'argmax', axis=axis, out=out, **kwds)


def _argmin_dispatcher(a, axis=None, out=None, *, keepdims=np._NoValue):
    return (a, out)


@array_function_dispatch(_argmin_dispatcher)
def argmin(a, axis=None, out=None, *, keepdims=np._NoValue):
    """
    Returns the indices of the minimum values along an axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        By default, the index is into the flattened array, otherwise
        along the specified axis.
    out : array, optional
        If provided, the result will be inserted into this array. It should
        be of the appropriate shape and dtype.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the array.

        .. versionadded:: 1.22.0

    Returns
    -------
    index_array : ndarray of ints
        Array of indices into the array. It has the same shape as `a.shape`
        with the dimension along `axis` removed. If `keepdims` is set to True,
        then the size of `axis` will be 1 with the resulting array having same
        shape as `a.shape`.

    See Also
    --------
    ndarray.argmin, argmax
    amin : The minimum value along a given axis.
    unravel_index : Convert a flat index into an index tuple.
    take_along_axis : Apply ``np.expand_dims(index_array, axis)``
                      from argmin to an array as if by calling min.

    Notes
    -----
    In case of multiple occurrences of the minimum values, the indices
    corresponding to the first occurrence are returned.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(6).reshape(2,3) + 10
    >>> a
    array([[10, 11, 12],
           [13, 14, 15]])
    >>> np.argmin(a)
    0
    >>> np.argmin(a, axis=0)
    array([0, 0, 0])
    >>> np.argmin(a, axis=1)
    array([0, 0])

    Indices of the minimum elements of a N-dimensional array:

    >>> ind = np.unravel_index(np.argmin(a, axis=None), a.shape)
    >>> ind
    (0, 0)
    >>> a[ind]
    10

    >>> b = np.arange(6) + 10
    >>> b[4] = 10
    >>> b
    array([10, 11, 12, 13, 10, 15])
    >>> np.argmin(b)  # Only the first occurrence is returned.
    0

    >>> x = np.array([[4,2,3], [1,0,3]])
    >>> index_array = np.argmin(x, axis=-1)
    >>> # Same as np.amin(x, axis=-1, keepdims=True)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1), axis=-1)
    array([[2],
           [0]])
    >>> # Same as np.amax(x, axis=-1)
    >>> np.take_along_axis(x, np.expand_dims(index_array, axis=-1),
    ...     axis=-1).squeeze(axis=-1)
    array([2, 0])

    Setting `keepdims` to `True`,

    >>> x = np.arange(24).reshape((2, 3, 4))
    >>> res = np.argmin(x, axis=1, keepdims=True)
    >>> res.shape
    (2, 1, 4)
    """
    kwds = {'keepdims': keepdims} if keepdims is not np._NoValue else {}
    return _wrapfunc(a, 'argmin', axis=axis, out=out, **kwds)


def _searchsorted_dispatcher(a, v, side=None, sorter=None):
    return (a, v, sorter)


@array_function_dispatch(_searchsorted_dispatcher)
def searchsorted(a, v, side='left', sorter=None):
    """
    Find indices where elements should be inserted to maintain order.

    Find the indices into a sorted array `a` such that, if the
    corresponding elements in `v` were inserted before the indices, the
    order of `a` would be preserved.

    Assuming that `a` is sorted:

    ======  ============================
    `side`  returned index `i` satisfies
    ======  ============================
    left    ``a[i-1] < v <= a[i]``
    right   ``a[i-1] <= v < a[i]``
    ======  ============================

    Parameters
    ----------
    a : 1-D array_like
        Input array. If `sorter` is None, then it must be sorted in
        ascending order, otherwise `sorter` must be an array of indices
        that sort it.
    v : array_like
        Values to insert into `a`.
    side : {'left', 'right'}, optional
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.  If there is no suitable
        index, return either 0 or N (where N is the length of `a`).
    sorter : 1-D array_like, optional
        Optional array of integer indices that sort array a into ascending
        order. They are typically the result of argsort.

    Returns
    -------
    indices : int or array of ints
        Array of insertion points with the same shape as `v`,
        or an integer if `v` is a scalar.

    See Also
    --------
    sort : Return a sorted copy of an array.
    histogram : Produce histogram from 1-D data.

    Notes
    -----
    Binary search is used to find the required insertion points.

    As of NumPy 1.4.0 `searchsorted` works with real/complex arrays containing
    `nan` values. The enhanced sort order is documented in `sort`.

    This function uses the same algorithm as the builtin python
    `bisect.bisect_left` (``side='left'``) and `bisect.bisect_right`
    (``side='right'``) functions, which is also vectorized
    in the `v` argument.

    Examples
    --------
    >>> import numpy as np
    >>> np.searchsorted([11,12,13,14,15], 13)
    2
    >>> np.searchsorted([11,12,13,14,15], 13, side='right')
    3
    >>> np.searchsorted([11,12,13,14,15], [-10, 20, 12, 13])
    array([0, 5, 1, 2])

    When `sorter` is used, the returned indices refer to the sorted
    array of `a` and not `a` itself:

    >>> a = np.array([40, 10, 20, 30])
    >>> sorter = np.argsort(a)
    >>> sorter
    array([1, 2, 3, 0])  # Indices that would sort the array 'a'
    >>> result = np.searchsorted(a, 25, sorter=sorter)
    >>> result
    2
    >>> a[sorter[result]]
    30  # The element at index 2 of the sorted array is 30.
    """
    return _wrapfunc(a, 'searchsorted', v, side=side, sorter=sorter)


def _resize_dispatcher(a, new_shape):
    return (a,)


@array_function_dispatch(_resize_dispatcher)
def resize(a, new_shape):
    """
    Return a new array with the specified shape.

    If the new array is larger than the original array, then the new
    array is filled with repeated copies of `a`.  Note that this behavior
    is different from a.resize(new_shape) which fills with zeros instead
    of repeated copies of `a`.

    Parameters
    ----------
    a : array_like
        Array to be resized.

    new_shape : int or tuple of int
        Shape of resized array.

    Returns
    -------
    reshaped_array : ndarray
        The new array is formed from the data in the old array, repeated
        if necessary to fill out the required number of elements.  The
        data are repeated iterating over the array in C-order.

    See Also
    --------
    numpy.reshape : Reshape an array without changing the total size.
    numpy.pad : Enlarge and pad an array.
    numpy.repeat : Repeat elements of an array.
    ndarray.resize : resize an array in-place.

    Notes
    -----
    When the total size of the array does not change `~numpy.reshape` should
    be used.  In most other cases either indexing (to reduce the size)
    or padding (to increase the size) may be a more appropriate solution.

    Warning: This functionality does **not** consider axes separately,
    i.e. it does not apply interpolation/extrapolation.
    It fills the return array with the required number of elements, iterating
    over `a` in C-order, disregarding axes (and cycling back from the start if
    the new shape is larger).  This functionality is therefore not suitable to
    resize images, or data where each axis represents a separate and distinct
    entity.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[0,1],[2,3]])
    >>> np.resize(a,(2,3))
    array([[0, 1, 2],
           [3, 0, 1]])
    >>> np.resize(a,(1,4))
    array([[0, 1, 2, 3]])
    >>> np.resize(a,(2,4))
    array([[0, 1, 2, 3],
           [0, 1, 2, 3]])

    """
    if isinstance(new_shape, (int, nt.integer)):
        new_shape = (new_shape,)

    a = ravel(a)

    new_size = 1
    for dim_length in new_shape:
        new_size *= dim_length
        if dim_length < 0:
            raise ValueError(
                'all elements of `new_shape` must be non-negative'
            )

    if a.size == 0 or new_size == 0:
        # First case must zero fill. The second would have repeats == 0.
        return np.zeros_like(a, shape=new_shape)

    repeats = -(-new_size // a.size)  # ceil division
    a = concatenate((a,) * repeats)[:new_size]

    return reshape(a, new_shape)


def _squeeze_dispatcher(a, axis=None):
    return (a,)


@array_function_dispatch(_squeeze_dispatcher)
def squeeze(a, axis=None):
    """
    Remove axes of length one from `a`.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Selects a subset of the entries of length one in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. This is always `a` itself
        or a view into `a`. Note that if all axes are squeezed,
        the result is a 0d array and not a scalar.

    Raises
    ------
    ValueError
        If `axis` is not None, and an axis being squeezed is not of length 1

    See Also
    --------
    expand_dims : The inverse operation, adding entries of length one
    reshape : Insert, remove, and combine dimensions, and resize existing ones

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> np.squeeze(x).shape
    (3,)
    >>> np.squeeze(x, axis=0).shape
    (3, 1)
    >>> np.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    ValueError: cannot select an axis to squeeze out which has size
    not equal to one
    >>> np.squeeze(x, axis=2).shape
    (1, 3)
    >>> x = np.array([[1234]])
    >>> x.shape
    (1, 1)
    >>> np.squeeze(x)
    array(1234)  # 0d array
    >>> np.squeeze(x).shape
    ()
    >>> np.squeeze(x)[()]
    1234

    """
    try:
        squeeze = a.squeeze
    except AttributeError:
        return _wrapit(a, 'squeeze', axis=axis)
    if axis is None:
        return squeeze()
    else:
        return squeeze(axis=axis)


def _diagonal_dispatcher(a, offset=None, axis1=None, axis2=None):
    return (a,)


@array_function_dispatch(_diagonal_dispatcher)
def diagonal(a, offset=0, axis1=0, axis2=1):
    """
    Return specified diagonals.

    If `a` is 2-D, returns the diagonal of `a` with the given offset,
    i.e., the collection of elements of the form ``a[i, i+offset]``.  If
    `a` has more than two dimensions, then the axes specified by `axis1`
    and `axis2` are used to determine the 2-D sub-array whose diagonal is
    returned.  The shape of the resulting array can be determined by
    removing `axis1` and `axis2` and appending an index to the right equal
    to the size of the resulting diagonals.

    In versions of NumPy prior to 1.7, this function always returned a new,
    independent array containing a copy of the values in the diagonal.

    In NumPy 1.7 and 1.8, it continues to return a copy of the diagonal,
    but depending on this fact is deprecated. Writing to the resulting
    array continues to work as it used to, but a FutureWarning is issued.

    Starting in NumPy 1.9 it returns a read-only view on the original array.
    Attempting to write to the resulting array will produce an error.

    In some future release, it will return a read/write view and writing to
    the returned array will alter your original array.  The returned array
    will have the same type as the input array.

    If you don't write to the array returned by this function, then you can
    just ignore all of the above.

    If you depend on the current behavior, then we suggest copying the
    returned array explicitly, i.e., use ``np.diagonal(a).copy()`` instead
    of just ``np.diagonal(a)``. This will work with both past and future
    versions of NumPy.

    Parameters
    ----------
    a : array_like
        Array from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal.  Can be positive or
        negative.  Defaults to main diagonal (0).
    axis1 : int, optional
        Axis to be used as the first axis of the 2-D sub-arrays from which
        the diagonals should be taken.  Defaults to first axis (0).
    axis2 : int, optional
        Axis to be used as the second axis of the 2-D sub-arrays from
        which the diagonals should be taken. Defaults to second axis (1).

    Returns
    -------
    array_of_diagonals : ndarray
        If `a` is 2-D, then a 1-D array containing the diagonal and of the
        same type as `a` is returned unless `a` is a `matrix`, in which case
        a 1-D array rather than a (2-D) `matrix` is returned in order to
        maintain backward compatibility.

        If ``a.ndim > 2``, then the dimensions specified by `axis1` and `axis2`
        are removed, and a new axis inserted at the end corresponding to the
        diagonal.

    Raises
    ------
    ValueError
        If the dimension of `a` is less than 2.

    See Also
    --------
    diag : MATLAB work-a-like for 1-D and 2-D arrays.
    diagflat : Create diagonal arrays.
    trace : Sum along diagonals.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(4).reshape(2,2)
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> a.diagonal()
    array([0, 3])
    >>> a.diagonal(1)
    array([1])

    A 3-D example:

    >>> a = np.arange(8).reshape(2,2,2); a
    array([[[0, 1],
            [2, 3]],
           [[4, 5],
            [6, 7]]])
    >>> a.diagonal(0,  # Main diagonals of two arrays created by skipping
    ...            0,  # across the outer(left)-most axis last and
    ...            1)  # the "middle" (row) axis first.
    array([[0, 6],
           [1, 7]])

    The sub-arrays whose main diagonals we just obtained; note that each
    corresponds to fixing the right-most (column) axis, and that the
    diagonals are "packed" in rows.

    >>> a[:,:,0]  # main diagonal is [0 6]
    array([[0, 2],
           [4, 6]])
    >>> a[:,:,1]  # main diagonal is [1 7]
    array([[1, 3],
           [5, 7]])

    The anti-diagonal can be obtained by reversing the order of elements
    using either `numpy.flipud` or `numpy.fliplr`.

    >>> a = np.arange(9).reshape(3, 3)
    >>> a
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> np.fliplr(a).diagonal()  # Horizontal flip
    array([2, 4, 6])
    >>> np.flipud(a).diagonal()  # Vertical flip
    array([6, 4, 2])

    Note that the order in which the diagonal is retrieved varies depending
    on the flip function.
    """
    if isinstance(a, np.matrix):
        # Make diagonal of matrix 1-D to preserve backward compatibility.
        return asarray(a).diagonal(offset=offset, axis1=axis1, axis2=axis2)
    else:
        return asanyarray(a).diagonal(offset=offset, axis1=axis1, axis2=axis2)


def _trace_dispatcher(
        a, offset=None, axis1=None, axis2=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_trace_dispatcher)
def trace(a, offset=0, axis1=0, axis2=1, dtype=None, out=None):
    """
    Return the sum along diagonals of the array.

    If `a` is 2-D, the sum along its diagonal with the given offset
    is returned, i.e., the sum of elements ``a[i,i+offset]`` for all i.

    If `a` has more than two dimensions, then the axes specified by axis1 and
    axis2 are used to determine the 2-D sub-arrays whose traces are returned.
    The shape of the resulting array is the same as that of `a` with `axis1`
    and `axis2` removed.

    Parameters
    ----------
    a : array_like
        Input array, from which the diagonals are taken.
    offset : int, optional
        Offset of the diagonal from the main diagonal. Can be both positive
        and negative. Defaults to 0.
    axis1, axis2 : int, optional
        Axes to be used as the first and second axis of the 2-D sub-arrays
        from which the diagonals should be taken. Defaults are the first two
        axes of `a`.
    dtype : dtype, optional
        Determines the data-type of the returned array and of the accumulator
        where the elements are summed. If dtype has the value None and `a` is
        of integer type of precision less than the default integer
        precision, then the default integer precision is used. Otherwise,
        the precision is the same as that of `a`.
    out : ndarray, optional
        Array into which the output is placed. Its type is preserved and
        it must be of the right shape to hold the output.

    Returns
    -------
    sum_along_diagonals : ndarray
        If `a` is 2-D, the sum along the diagonal is returned.  If `a` has
        larger dimensions, then an array of sums along diagonals is returned.

    See Also
    --------
    diag, diagonal, diagflat

    Examples
    --------
    >>> import numpy as np
    >>> np.trace(np.eye(3))
    3.0
    >>> a = np.arange(8).reshape((2,2,2))
    >>> np.trace(a)
    array([6, 8])

    >>> a = np.arange(24).reshape((2,2,2,3))
    >>> np.trace(a).shape
    (2, 3)

    """
    if isinstance(a, np.matrix):
        # Get trace of matrix via an array to preserve backward compatibility.
        return asarray(a).trace(
            offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
        )
    else:
        return asanyarray(a).trace(
            offset=offset, axis1=axis1, axis2=axis2, dtype=dtype, out=out
        )


def _ravel_dispatcher(a, order=None):
    return (a,)


@array_function_dispatch(_ravel_dispatcher)
def ravel(a, order='C'):
    """Return a contiguous flattened array.

    A 1-D array, containing the elements of the input, is returned.  A copy is
    made only if needed.

    As of NumPy 1.10, the returned array will have the same type as the input
    array. (for example, a masked array will be returned for a masked array
    input)

    Parameters
    ----------
    a : array_like
        Input array.  The elements in `a` are read in the order specified by
        `order`, and packed as a 1-D array.
    order : {'C','F', 'A', 'K'}, optional

        The elements of `a` are read using this index order. 'C' means
        to index the elements in row-major, C-style order,
        with the last axis index changing fastest, back to the first
        axis index changing slowest.  'F' means to index the elements
        in column-major, Fortran-style order, with the
        first index changing fastest, and the last index changing
        slowest. Note that the 'C' and 'F' options take no account of
        the memory layout of the underlying array, and only refer to
        the order of axis indexing.  'A' means to read the elements in
        Fortran-like index order if `a` is Fortran *contiguous* in
        memory, C-like order otherwise.  'K' means to read the
        elements in the order they occur in memory, except for
        reversing the data when strides are negative.  By default, 'C'
        index order is used.

    Returns
    -------
    y : array_like
        y is a contiguous 1-D array of the same subtype as `a`,
        with shape ``(a.size,)``.
        Note that matrices are special cased for backward compatibility,
        if `a` is a matrix, then y is a 1-D ndarray.

    See Also
    --------
    ndarray.flat : 1-D iterator over an array.
    ndarray.flatten : 1-D array copy of the elements of an array
                      in row-major order.
    ndarray.reshape : Change the shape of an array without changing its data.

    Notes
    -----
    In row-major, C-style order, in two dimensions, the row index
    varies the slowest, and the column index the quickest.  This can
    be generalized to multiple dimensions, where row-major order
    implies that the index along the first axis varies slowest, and
    the index along the last quickest.  The opposite holds for
    column-major, Fortran-style index ordering.

    When a view is desired in as many cases as possible, ``arr.reshape(-1)``
    may be preferable. However, ``ravel`` supports ``K`` in the optional
    ``order`` argument while ``reshape`` does not.

    Examples
    --------
    It is equivalent to ``reshape(-1, order=order)``.

    >>> import numpy as np
    >>> x = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.ravel(x)
    array([1, 2, 3, 4, 5, 6])

    >>> x.reshape(-1)
    array([1, 2, 3, 4, 5, 6])

    >>> np.ravel(x, order='F')
    array([1, 4, 2, 5, 3, 6])

    When ``order`` is 'A', it will preserve the array's 'C' or 'F' ordering:

    >>> np.ravel(x.T)
    array([1, 4, 2, 5, 3, 6])
    >>> np.ravel(x.T, order='A')
    array([1, 2, 3, 4, 5, 6])

    When ``order`` is 'K', it will preserve orderings that are neither 'C'
    nor 'F', but won't reverse axes:

    >>> a = np.arange(3)[::-1]; a
    array([2, 1, 0])
    >>> a.ravel(order='C')
    array([2, 1, 0])
    >>> a.ravel(order='K')
    array([2, 1, 0])

    >>> a = np.arange(12).reshape(2,3,2).swapaxes(1,2); a
    array([[[ 0,  2,  4],
            [ 1,  3,  5]],
           [[ 6,  8, 10],
            [ 7,  9, 11]]])
    >>> a.ravel(order='C')
    array([ 0,  2,  4,  1,  3,  5,  6,  8, 10,  7,  9, 11])
    >>> a.ravel(order='K')
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11])

    """
    if isinstance(a, np.matrix):
        return asarray(a).ravel(order=order)
    else:
        return asanyarray(a).ravel(order=order)


def _nonzero_dispatcher(a):
    return (a,)


@array_function_dispatch(_nonzero_dispatcher)
def nonzero(a):
    """
    Return the indices of the elements that are non-zero.

    Returns a tuple of arrays, one for each dimension of `a`,
    containing the indices of the non-zero elements in that
    dimension. The values in `a` are always tested and returned in
    row-major, C-style order.

    To group the indices by element, rather than dimension, use `argwhere`,
    which returns a row for each non-zero element.

    .. note::

       When called on a zero-d array or scalar, ``nonzero(a)`` is treated
       as ``nonzero(atleast_1d(a))``.

       .. deprecated:: 1.17.0

          Use `atleast_1d` explicitly if this behavior is deliberate.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    tuple_of_arrays : tuple
        Indices of elements that are non-zero.

    See Also
    --------
    flatnonzero :
        Return indices that are non-zero in the flattened version of the input
        array.
    ndarray.nonzero :
        Equivalent ndarray method.
    count_nonzero :
        Counts the number of non-zero elements in the input array.

    Notes
    -----
    While the nonzero values can be obtained with ``a[nonzero(a)]``, it is
    recommended to use ``x[x.astype(bool)]`` or ``x[x != 0]`` instead, which
    will correctly handle 0-d arrays.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    >>> x
    array([[3, 0, 0],
           [0, 4, 0],
           [5, 6, 0]])
    >>> np.nonzero(x)
    (array([0, 1, 2, 2]), array([0, 1, 0, 1]))

    >>> x[np.nonzero(x)]
    array([3, 4, 5, 6])
    >>> np.transpose(np.nonzero(x))
    array([[0, 0],
           [1, 1],
           [2, 0],
           [2, 1]])

    A common use for ``nonzero`` is to find the indices of an array, where
    a condition is True.  Given an array `a`, the condition `a` > 3 is a
    boolean array and since False is interpreted as 0, np.nonzero(a > 3)
    yields the indices of the `a` where the condition is true.

    >>> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> a > 3
    array([[False, False, False],
           [ True,  True,  True],
           [ True,  True,  True]])
    >>> np.nonzero(a > 3)
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    Using this result to index `a` is equivalent to using the mask directly:

    >>> a[np.nonzero(a > 3)]
    array([4, 5, 6, 7, 8, 9])
    >>> a[a > 3]  # prefer this spelling
    array([4, 5, 6, 7, 8, 9])

    ``nonzero`` can also be called as a method of the array.

    >>> (a > 3).nonzero()
    (array([1, 1, 1, 2, 2, 2]), array([0, 1, 2, 0, 1, 2]))

    """
    return _wrapfunc(a, 'nonzero')


def _shape_dispatcher(a):
    return (a,)


@array_function_dispatch(_shape_dispatcher)
def shape(a):
    """
    Return the shape of an array.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    shape : tuple of ints
        The elements of the shape tuple give the lengths of the
        corresponding array dimensions.

    See Also
    --------
    len : ``len(a)`` is equivalent to ``np.shape(a)[0]`` for N-D arrays with
          ``N>=1``.
    ndarray.shape : Equivalent array method.

    Examples
    --------
    >>> import numpy as np
    >>> np.shape(np.eye(3))
    (3, 3)
    >>> np.shape([[1, 3]])
    (1, 2)
    >>> np.shape([0])
    (1,)
    >>> np.shape(0)
    ()

    >>> a = np.array([(1, 2), (3, 4), (5, 6)],
    ...              dtype=[('x', 'i4'), ('y', 'i4')])
    >>> np.shape(a)
    (3,)
    >>> a.shape
    (3,)

    """
    try:
        result = a.shape
    except AttributeError:
        result = asarray(a).shape
    return result


def _compress_dispatcher(condition, a, axis=None, out=None):
    return (condition, a, out)


@array_function_dispatch(_compress_dispatcher)
def compress(condition, a, axis=None, out=None):
    """
    Return selected slices of an array along given axis.

    When working along a given axis, a slice along that axis is returned in
    `output` for each index where `condition` evaluates to True. When
    working on a 1-D array, `compress` is equivalent to `extract`.

    Parameters
    ----------
    condition : 1-D array of bools
        Array that selects which entries to return. If len(condition)
        is less than the size of `a` along the given axis, then output is
        truncated to the length of the condition array.
    a : array_like
        Array from which to extract a part.
    axis : int, optional
        Axis along which to take slices. If None (default), work on the
        flattened array.
    out : ndarray, optional
        Output array.  Its type is preserved and it must be of the right
        shape to hold the output.

    Returns
    -------
    compressed_array : ndarray
        A copy of `a` without the slices along axis for which `condition`
        is false.

    See Also
    --------
    take, choose, diag, diagonal, select
    ndarray.compress : Equivalent method in ndarray
    extract : Equivalent method when working on 1-D arrays
    :ref:`ufuncs-output-type`

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4], [5, 6]])
    >>> a
    array([[1, 2],
           [3, 4],
           [5, 6]])
    >>> np.compress([0, 1], a, axis=0)
    array([[3, 4]])
    >>> np.compress([False, True, True], a, axis=0)
    array([[3, 4],
           [5, 6]])
    >>> np.compress([False, True], a, axis=1)
    array([[2],
           [4],
           [6]])

    Working on the flattened array does not return slices along an axis but
    selects elements.

    >>> np.compress([False, True], a)
    array([2])

    """
    return _wrapfunc(a, 'compress', condition, axis=axis, out=out)


def _clip_dispatcher(a, a_min=None, a_max=None, out=None, *, min=None,
                     max=None, **kwargs):
    return (a, a_min, a_max, out, min, max)


@array_function_dispatch(_clip_dispatcher)
def clip(a, a_min=np._NoValue, a_max=np._NoValue, out=None, *,
         min=np._NoValue, max=np._NoValue, **kwargs):
    """
    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Equivalent to but faster than ``np.minimum(a_max, np.maximum(a, a_min))``.

    No check is performed to ensure ``a_min < a_max``.

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min, a_max : array_like or None
        Minimum and maximum value. If ``None``, clipping is not performed on
        the corresponding edge. If both ``a_min`` and ``a_max`` are ``None``,
        the elements of the returned array stay the same. Both are broadcasted
        against ``a``.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  `out` must be of the right shape
        to hold the output.  Its type is preserved.
    min, max : array_like or None
        Array API compatible alternatives for ``a_min`` and ``a_max``
        arguments. Either ``a_min`` and ``a_max`` or ``min`` and ``max``
        can be passed at the same time. Default: ``None``.

        .. versionadded:: 2.1.0
    **kwargs
        For other keyword-only arguments, see the
        :ref:`ufunc docs <ufuncs.kwargs>`.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of `a`, but where values
        < `a_min` are replaced with `a_min`, and those > `a_max`
        with `a_max`.

    See Also
    --------
    :ref:`ufuncs-output-type`

    Notes
    -----
    When `a_min` is greater than `a_max`, `clip` returns an
    array in which all values are equal to `a_max`,
    as shown in the second example.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, 1, 8)
    array([1, 1, 2, 3, 4, 5, 6, 7, 8, 8])
    >>> np.clip(a, 8, 1)
    array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> np.clip(a, 3, 6, out=a)
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a
    array([3, 3, 3, 3, 4, 5, 6, 6, 6, 6])
    >>> a = np.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> np.clip(a, [3, 4, 1, 1, 1, 4, 4, 4, 4, 4], 8)
    array([3, 4, 2, 3, 4, 5, 6, 7, 8, 8])

    """
    if a_min is np._NoValue and a_max is np._NoValue:
        a_min = None if min is np._NoValue else min
        a_max = None if max is np._NoValue else max
    elif a_min is np._NoValue:
        raise TypeError("clip() missing 1 required positional "
                        "argument: 'a_min'")
    elif a_max is np._NoValue:
        raise TypeError("clip() missing 1 required positional "
                        "argument: 'a_max'")
    elif min is not np._NoValue or max is not np._NoValue:
        raise ValueError("Passing `min` or `max` keyword argument when "
                         "`a_min` and `a_max` are provided is forbidden.")

    return _wrapfunc(a, 'clip', a_min, a_max, out=out, **kwargs)


def _sum_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                    initial=None, where=None):
    return (a, out)


@array_function_dispatch(_sum_dispatcher)
def sum(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
        initial=np._NoValue, where=np._NoValue):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Elements to sum.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis. If
        axis is a tuple of ints, a sum is performed on all of the axes
        specified in the tuple instead of a single axis or all the axes as
        before.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed.  The dtype of `a` is used by default unless `a`
        has an integer dtype of less precision than the default platform
        integer.  In that case, if `a` is signed then the platform integer
        is used while if `a` is unsigned then an unsigned integer of the
        same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        Starting value for the sum. See `~numpy.ufunc.reduce` for details.
    where : array_like of bool, optional
        Elements to include in the sum. See `~numpy.ufunc.reduce` for details.

    Returns
    -------
    sum_along_axis : ndarray
        An array with the same shape as `a`, with the specified
        axis removed.   If `a` is a 0-d array, or if `axis` is None, a scalar
        is returned.  If an output array is specified, a reference to
        `out` is returned.

    See Also
    --------
    ndarray.sum : Equivalent method.
    add: ``numpy.add.reduce`` equivalent function.
    cumsum : Cumulative sum of array elements.
    trapezoid : Integration of array values using composite trapezoidal rule.

    mean, average

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    The sum of an empty array is the neutral element 0:

    >>> np.sum([])
    0.0

    For floating point numbers the numerical precision of sum (and
    ``np.add.reduce``) is in general limited by directly adding each number
    individually to the result causing rounding errors in every step.
    However, often numpy will use a  numerically better approach (partial
    pairwise summation) leading to improved precision in many use-cases.
    This improved precision is always provided when no ``axis`` is given.
    When ``axis`` is given, it will depend on which axis is summed.
    Technically, to provide the best speed possible, the improved precision
    is only used when the summation is along the fast axis in memory.
    Note that the exact precision may vary depending on other parameters.
    In contrast to NumPy, Python's ``math.fsum`` function uses a slower but
    more precise approach to summation.
    Especially when summing a large number of lower precision floating point
    numbers, such as ``float32``, numerical errors can become significant.
    In such cases it can be advisable to use `dtype="float64"` to use a higher
    precision for the output.

    Examples
    --------
    >>> import numpy as np
    >>> np.sum([0.5, 1.5])
    2.0
    >>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
    np.int32(1)
    >>> np.sum([[0, 1], [0, 5]])
    6
    >>> np.sum([[0, 1], [0, 5]], axis=0)
    array([0, 6])
    >>> np.sum([[0, 1], [0, 5]], axis=1)
    array([1, 5])
    >>> np.sum([[0, 1], [np.nan, 5]], where=[False, True], axis=1)
    array([1., 5.])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    np.int8(-128)

    You can also start the sum with a value other than zero:

    >>> np.sum([10], initial=5)
    15
    """
    if isinstance(a, _gentype):
        # 2018-02-25, 1.15.0
        warnings.warn(
            "Calling np.sum(generator) is deprecated, and in the future will "
            "give a different result. Use np.sum(np.fromiter(generator)) or "
            "the python sum builtin instead.",
            DeprecationWarning, stacklevel=2
        )

        res = _sum_(a)
        if out is not None:
            out[...] = res
            return out
        return res

    return _wrapreduction(
        a, np.add, 'sum', axis, dtype, out,
        keepdims=keepdims, initial=initial, where=where
    )


def _any_dispatcher(a, axis=None, out=None, keepdims=None, *,
                    where=np._NoValue):
    return (a, where, out)


@array_function_dispatch(_any_dispatcher)
def any(a, axis=None, out=None, keepdims=np._NoValue, *, where=np._NoValue):
    """
    Test whether any array element along a given axis evaluates to True.

    Returns single boolean if `axis` is ``None``

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical OR reduction is performed.
        The default (``axis=None``) is to perform a logical OR over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis. If this
        is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output and its type is preserved
        (e.g., if it is of type float, then it will remain so, returning
        1.0 for True and 0.0 for False, regardless of the type of `a`).
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `any` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in checking for any `True` values.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    any : bool or ndarray
        A new boolean or `ndarray` is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    ndarray.any : equivalent method

    all : Test whether all elements along a given axis evaluate to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity evaluate
    to `True` because these are not equal to zero.

    .. versionchanged:: 2.0
       Before NumPy 2.0, ``any`` did not return booleans for object dtype
       input arrays.
       This behavior is still available via ``np.logical_or.reduce``.

    Examples
    --------
    >>> import numpy as np
    >>> np.any([[True, False], [True, True]])
    True

    >>> np.any([[True,  False, True ],
    ...         [False, False, False]], axis=0)
    array([ True, False, True])

    >>> np.any([-1, 0, 5])
    True

    >>> np.any([[np.nan], [np.inf]], axis=1, keepdims=True)
    array([[ True],
           [ True]])

    >>> np.any([[True, False], [False, False]], where=[[False], [True]])
    False

    >>> a = np.array([[1, 0, 0],
    ...               [0, 0, 1],
    ...               [0, 0, 0]])
    >>> np.any(a, axis=0)
    array([ True, False,  True])
    >>> np.any(a, axis=1)
    array([ True,  True, False])

    >>> o=np.array(False)
    >>> z=np.any([-1, 4, 5], out=o)
    >>> z, o
    (array(True), array(True))
    >>> # Check now that z is a reference to o
    >>> z is o
    True
    >>> id(z), id(o) # identity of z and o              # doctest: +SKIP
    (191614240, 191614240)

    """
    return _wrapreduction_any_all(a, np.logical_or, 'any', axis, out,
                                  keepdims=keepdims, where=where)


def _all_dispatcher(a, axis=None, out=None, keepdims=None, *,
                    where=None):
    return (a, where, out)


@array_function_dispatch(_all_dispatcher)
def all(a, axis=None, out=None, keepdims=np._NoValue, *, where=np._NoValue):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (``axis=None``) is to perform a logical AND over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis. If this
        is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternate output array in which to place the result.
        It must have the same shape as the expected output and its
        type is preserved (e.g., if ``dtype(out)`` is float, the result
        will consist of 0.0's and 1.0's). See :ref:`ufuncs-output-type`
        for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `all` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in checking for all `True` values.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    all : ndarray, bool
        A new boolean or array is returned unless `out` is specified,
        in which case a reference to `out` is returned.

    See Also
    --------
    ndarray.all : equivalent method

    any : Test whether any element along a given axis evaluates to True.

    Notes
    -----
    Not a Number (NaN), positive infinity and negative infinity
    evaluate to `True` because these are not equal to zero.

    .. versionchanged:: 2.0
       Before NumPy 2.0, ``all`` did not return booleans for object dtype
       input arrays.
       This behavior is still available via ``np.logical_and.reduce``.

    Examples
    --------
    >>> import numpy as np
    >>> np.all([[True,False],[True,True]])
    False

    >>> np.all([[True,False],[True,True]], axis=0)
    array([ True, False])

    >>> np.all([-1, 4, 5])
    True

    >>> np.all([1.0, np.nan])
    True

    >>> np.all([[True, True], [False, True]], where=[[True], [False]])
    True

    >>> o=np.array(False)
    >>> z=np.all([-1, 4, 5], out=o)
    >>> id(z), id(o), z
    (28293632, 28293632, array(True)) # may vary

    """
    return _wrapreduction_any_all(a, np.logical_and, 'all', axis, out,
                                  keepdims=keepdims, where=where)


def _cumulative_func(x, func, axis, dtype, out, include_initial):
    x = np.atleast_1d(x)
    x_ndim = x.ndim
    if axis is None:
        if x_ndim >= 2:
            raise ValueError("For arrays which have more than one dimension "
                            "``axis`` argument is required.")
        axis = 0

    if out is not None and include_initial:
        item = [slice(None)] * x_ndim
        item[axis] = slice(1, None)
        func.accumulate(x, axis=axis, dtype=dtype, out=out[tuple(item)])
        item[axis] = 0
        out[tuple(item)] = func.identity
        return out

    res = func.accumulate(x, axis=axis, dtype=dtype, out=out)
    if include_initial:
        initial_shape = list(x.shape)
        initial_shape[axis] = 1
        res = np.concat(
            [np.full_like(res, func.identity, shape=initial_shape), res],
            axis=axis,
        )

    return res


def _cumulative_prod_dispatcher(x, /, *, axis=None, dtype=None, out=None,
                                include_initial=None):
    return (x, out)


@array_function_dispatch(_cumulative_prod_dispatcher)
def cumulative_prod(x, /, *, axis=None, dtype=None, out=None,
                    include_initial=False):
    """
    Return the cumulative product of elements along a given axis.

    This function is an Array API compatible alternative to `numpy.cumprod`.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed. The default
        (None) is only allowed for one-dimensional arrays. For arrays
        with more than one dimension ``axis`` is required.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.  If ``dtype`` is not specified, it
        defaults to the dtype of ``x``, unless ``x`` has an integer dtype
        with a precision less than that of the default platform integer.
        In that case, the default platform integer is used instead.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
    include_initial : bool, optional
        Boolean indicating whether to include the initial value (ones) as
        the first value in the output. With ``include_initial=True``
        the shape of the output is different than the shape of the input.
        Default: ``False``.

    Returns
    -------
    cumulative_prod_along_axis : ndarray
        A new array holding the result is returned unless ``out`` is
        specified, in which case a reference to ``out`` is returned. The
        result has the same shape as ``x`` if ``include_initial=False``.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> np.cumulative_prod(a)  # intermediate results 1, 1*2
    ...                        # total product 1*2*3 = 6
    array([1, 2, 6])
    >>> a = np.array([1, 2, 3, 4, 5, 6])
    >>> np.cumulative_prod(a, dtype=float) # specify type of output
    array([   1.,    2.,    6.,   24.,  120.,  720.])

    The cumulative product for each column (i.e., over the rows) of ``b``:

    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.cumulative_prod(b, axis=0)
    array([[ 1,  2,  3],
           [ 4, 10, 18]])

    The cumulative product for each row (i.e. over the columns) of ``b``:

    >>> np.cumulative_prod(b, axis=1)
    array([[  1,   2,   6],
           [  4,  20, 120]])

    """
    return _cumulative_func(x, um.multiply, axis, dtype, out, include_initial)


def _cumulative_sum_dispatcher(x, /, *, axis=None, dtype=None, out=None,
                               include_initial=None):
    return (x, out)


@array_function_dispatch(_cumulative_sum_dispatcher)
def cumulative_sum(x, /, *, axis=None, dtype=None, out=None,
                   include_initial=False):
    """
    Return the cumulative sum of the elements along a given axis.

    This function is an Array API compatible alternative to `numpy.cumsum`.

    Parameters
    ----------
    x : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is only allowed for one-dimensional arrays. For arrays
        with more than one dimension ``axis`` is required.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If ``dtype`` is not specified, it defaults
        to the dtype of ``x``, unless ``x`` has an integer dtype with
        a precision less than that of the default platform integer.
        In that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See :ref:`ufuncs-output-type`
        for more details.
    include_initial : bool, optional
        Boolean indicating whether to include the initial value (zeros) as
        the first value in the output. With ``include_initial=True``
        the shape of the output is different than the shape of the input.
        Default: ``False``.

    Returns
    -------
    cumulative_sum_along_axis : ndarray
        A new array holding the result is returned unless ``out`` is
        specified, in which case a reference to ``out`` is returned. The
        result has the same shape as ``x`` if ``include_initial=False``.

    See Also
    --------
    sum : Sum array elements.
    trapezoid : Integration of array values using composite trapezoidal rule.
    diff : Calculate the n-th discrete difference along given axis.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    ``cumulative_sum(a)[-1]`` may not be equal to ``sum(a)`` for
    floating-point values since ``sum`` may use a pairwise summation routine,
    reducing the roundoff-error. See `sum` for more information.

    Examples
    --------
    >>> a = np.array([1, 2, 3, 4, 5, 6])
    >>> a
    array([1, 2, 3, 4, 5, 6])
    >>> np.cumulative_sum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> np.cumulative_sum(a, dtype=float)  # specifies type of output value(s)
    array([  1.,   3.,   6.,  10.,  15.,  21.])

    >>> b = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.cumulative_sum(b,axis=0)  # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> np.cumulative_sum(b,axis=1)  # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    ``cumulative_sum(c)[-1]`` may not be equal to ``sum(c)``

    >>> c = np.array([1, 2e-9, 3e-9] * 1000000)
    >>> np.cumulative_sum(c)[-1]
    1000000.0050045159
    >>> c.sum()
    1000000.0050000029

    """
    return _cumulative_func(x, um.add, axis, dtype, out, include_initial)


def _cumsum_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_cumsum_dispatcher)
def cumsum(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative sum of the elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative sum is computed. The default
        (None) is to compute the cumsum over the flattened array.
    dtype : dtype, optional
        Type of the returned array and of the accumulator in which the
        elements are summed.  If `dtype` is not specified, it defaults
        to the dtype of `a`, unless `a` has an integer dtype with a
        precision less than that of the default platform integer.  In
        that case, the default platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type will be cast if necessary. See :ref:`ufuncs-output-type`
        for more details.

    Returns
    -------
    cumsum_along_axis : ndarray.
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to `out` is returned. The
        result has the same size as `a`, and the same shape as `a` if
        `axis` is not None or `a` is a 1-d array.

    See Also
    --------
    cumulative_sum : Array API compatible alternative for ``cumsum``.
    sum : Sum array elements.
    trapezoid : Integration of array values using composite trapezoidal rule.
    diff : Calculate the n-th discrete difference along given axis.

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    ``cumsum(a)[-1]`` may not be equal to ``sum(a)`` for floating-point
    values since ``sum`` may use a pairwise summation routine, reducing
    the roundoff-error. See `sum` for more information.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,2,3], [4,5,6]])
    >>> a
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> np.cumsum(a)
    array([ 1,  3,  6, 10, 15, 21])
    >>> np.cumsum(a, dtype=float)     # specifies type of output value(s)
    array([  1.,   3.,   6.,  10.,  15.,  21.])

    >>> np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
    array([[1, 2, 3],
           [5, 7, 9]])
    >>> np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows
    array([[ 1,  3,  6],
           [ 4,  9, 15]])

    ``cumsum(b)[-1]`` may not be equal to ``sum(b)``

    >>> b = np.array([1, 2e-9, 3e-9] * 1000000)
    >>> b.cumsum()[-1]
    1000000.0050045159
    >>> b.sum()
    1000000.0050000029

    """
    return _wrapfunc(a, 'cumsum', axis=axis, dtype=dtype, out=out)


def _ptp_dispatcher(a, axis=None, out=None, keepdims=None):
    return (a, out)


@array_function_dispatch(_ptp_dispatcher)
def ptp(a, axis=None, out=None, keepdims=np._NoValue):
    """
    Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for 'peak to peak'.

    .. warning::
        `ptp` preserves the data type of the array. This means the
        return value for an input of signed integers with n bits
        (e.g. `numpy.int8`, `numpy.int16`, etc) is also a signed integer
        with n bits.  In that case, peak-to-peak values greater than
        ``2**(n-1)-1`` will be returned as negative values. An example
        with a work-around is shown below.

    Parameters
    ----------
    a : array_like
        Input values.
    axis : None or int or tuple of ints, optional
        Axis along which to find the peaks.  By default, flatten the
        array.  `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a reduction is performed on multiple
        axes, instead of a single axis or all the axes as before.
    out : array_like
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output,
        but the type of the output values will be cast if necessary.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `ptp` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    Returns
    -------
    ptp : ndarray or scalar
        The range of a given array - `scalar` if array is one-dimensional
        or a new array holding the result along the given axis

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([[4, 9, 2, 10],
    ...               [6, 9, 7, 12]])

    >>> np.ptp(x, axis=1)
    array([8, 6])

    >>> np.ptp(x, axis=0)
    array([2, 0, 5, 2])

    >>> np.ptp(x)
    10

    This example shows that a negative value can be returned when
    the input is an array of signed integers.

    >>> y = np.array([[1, 127],
    ...               [0, 127],
    ...               [-1, 127],
    ...               [-2, 127]], dtype=np.int8)
    >>> np.ptp(y, axis=1)
    array([ 126,  127, -128, -127], dtype=int8)

    A work-around is to use the `view()` method to view the result as
    unsigned integers with the same bit width:

    >>> np.ptp(y, axis=1).view(np.uint8)
    array([126, 127, 128, 129], dtype=uint8)

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    return _methods._ptp(a, axis=axis, out=out, **kwargs)


def _max_dispatcher(a, axis=None, out=None, keepdims=None, initial=None,
                    where=None):
    return (a, out)


@array_function_dispatch(_max_dispatcher)
@set_module('numpy')
def max(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
         where=np._NoValue):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used. If this is a tuple of ints, the maximum is selected over
        multiple axes, instead of a single axis or all the axes as before.

    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the ``max`` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The minimum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

    where : array_like of bool, optional
        Elements to compare for the maximum. See `~numpy.ufunc.reduce`
        for details.

    Returns
    -------
    max : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is an int, the result is an array of dimension
        ``a.ndim - 1``. If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    See Also
    --------
    amin :
        The minimum value of an array along a given axis, propagating any NaNs.
    nanmax :
        The maximum value of an array along a given axis, ignoring any NaNs.
    maximum :
        Element-wise maximum of two arrays, propagating any NaNs.
    fmax :
        Element-wise maximum of two arrays, ignoring any NaNs.
    argmax :
        Return the indices of the maximum values.

    nanmin, minimum, fmin

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding max value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmax.

    Don't use `~numpy.max` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    ``max(a, axis=0)``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.max(a)           # Maximum of the flattened array
    3
    >>> np.max(a, axis=0)   # Maxima along the first axis
    array([2, 3])
    >>> np.max(a, axis=1)   # Maxima along the second axis
    array([1, 3])
    >>> np.max(a, where=[False, True], initial=-1, axis=0)
    array([-1,  3])
    >>> b = np.arange(5, dtype=float)
    >>> b[2] = np.nan
    >>> np.max(b)
    np.float64(nan)
    >>> np.max(b, where=~np.isnan(b), initial=-1)
    4.0
    >>> np.nanmax(b)
    4.0

    You can use an initial value to compute the maximum of an empty slice, or
    to initialize it to a different value:

    >>> np.max([[-50], [10]], axis=-1, initial=0)
    array([ 0, 10])

    Notice that the initial value is used as one of the elements for which the
    maximum is determined, unlike for the default argument Python's max
    function, which is only used for empty iterables.

    >>> np.max([5], initial=6)
    6
    >>> max([5], default=6)
    5
    """
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


@array_function_dispatch(_max_dispatcher)
def amax(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
         where=np._NoValue):
    """
    Return the maximum of an array or maximum along an axis.

    `amax` is an alias of `~numpy.max`.

    See Also
    --------
    max : alias of this function
    ndarray.max : equivalent method
    """
    return _wrapreduction(a, np.maximum, 'max', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


def _min_dispatcher(a, axis=None, out=None, keepdims=None, initial=None,
                    where=None):
    return (a, out)


@array_function_dispatch(_min_dispatcher)
def min(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
        where=np._NoValue):
    """
    Return the minimum of an array or minimum along an axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which to operate.  By default, flattened input is
        used.

        If this is a tuple of ints, the minimum is selected over multiple axes,
        instead of a single axis or all the axes as before.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the ``min`` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    initial : scalar, optional
        The maximum value of an output element. Must be present to allow
        computation on empty slice. See `~numpy.ufunc.reduce` for details.

    where : array_like of bool, optional
        Elements to compare for the minimum. See `~numpy.ufunc.reduce`
        for details.

    Returns
    -------
    min : ndarray or scalar
        Minimum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is an int, the result is an array of dimension
        ``a.ndim - 1``.  If `axis` is a tuple, the result is an array of
        dimension ``a.ndim - len(axis)``.

    See Also
    --------
    amax :
        The maximum value of an array along a given axis, propagating any NaNs.
    nanmin :
        The minimum value of an array along a given axis, ignoring any NaNs.
    minimum :
        Element-wise minimum of two arrays, propagating any NaNs.
    fmin :
        Element-wise minimum of two arrays, ignoring any NaNs.
    argmin :
        Return the indices of the minimum values.

    nanmax, maximum, fmax

    Notes
    -----
    NaN values are propagated, that is if at least one item is NaN, the
    corresponding min value will be NaN as well. To ignore NaN values
    (MATLAB behavior), please use nanmin.

    Don't use `~numpy.min` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``minimum(a[0], a[1])`` is faster than
    ``min(a, axis=0)``.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0, 1],
           [2, 3]])
    >>> np.min(a)           # Minimum of the flattened array
    0
    >>> np.min(a, axis=0)   # Minima along the first axis
    array([0, 1])
    >>> np.min(a, axis=1)   # Minima along the second axis
    array([0, 2])
    >>> np.min(a, where=[False, True], initial=10, axis=0)
    array([10,  1])

    >>> b = np.arange(5, dtype=float)
    >>> b[2] = np.nan
    >>> np.min(b)
    np.float64(nan)
    >>> np.min(b, where=~np.isnan(b), initial=10)
    0.0
    >>> np.nanmin(b)
    0.0

    >>> np.min([[-50], [10]], axis=-1, initial=0)
    array([-50,   0])

    Notice that the initial value is used as one of the elements for which the
    minimum is determined, unlike for the default argument Python's max
    function, which is only used for empty iterables.

    Notice that this isn't the same as Python's ``default`` argument.

    >>> np.min([6], initial=5)
    5
    >>> min([6], default=5)
    6
    """
    return _wrapreduction(a, np.minimum, 'min', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


@array_function_dispatch(_min_dispatcher)
def amin(a, axis=None, out=None, keepdims=np._NoValue, initial=np._NoValue,
         where=np._NoValue):
    """
    Return the minimum of an array or minimum along an axis.

    `amin` is an alias of `~numpy.min`.

    See Also
    --------
    min : alias of this function
    ndarray.min : equivalent method
    """
    return _wrapreduction(a, np.minimum, 'min', axis, None, out,
                          keepdims=keepdims, initial=initial, where=where)


def _prod_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None,
                     initial=None, where=None):
    return (a, out)


@array_function_dispatch(_prod_dispatcher)
def prod(a, axis=None, dtype=None, out=None, keepdims=np._NoValue,
         initial=np._NoValue, where=np._NoValue):
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed.  The default,
        axis=None, will calculate the product of all the elements in the
        input array. If axis is negative it counts from the last to the
        first axis.

        If axis is a tuple of ints, a product is performed on all of the
        axes specified in the tuple instead of a single axis or all the
        axes as before.
    dtype : dtype, optional
        The type of the returned array, as well as of the accumulator in
        which the elements are multiplied.  The dtype of `a` is used by
        default unless `a` has an integer dtype of less precision than the
        default platform integer.  In that case, if `a` is signed then the
        platform integer is used while if `a` is unsigned then an unsigned
        integer of the same precision as the platform integer is used.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in the
        result as dimensions with size one. With this option, the result
        will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `prod` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    initial : scalar, optional
        The starting value for this product. See `~numpy.ufunc.reduce`
        for details.
    where : array_like of bool, optional
        Elements to include in the product. See `~numpy.ufunc.reduce`
        for details.

    Returns
    -------
    product_along_axis : ndarray, see `dtype` parameter above.
        An array shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    See Also
    --------
    ndarray.prod : equivalent method
    :ref:`ufuncs-output-type`

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.  That means that, on a 32-bit platform:

    >>> x = np.array([536870910, 536870910, 536870910, 536870910])
    >>> np.prod(x)
    16 # may vary

    The product of an empty array is the neutral element 1:

    >>> np.prod([])
    1.0

    Examples
    --------
    By default, calculate the product of all elements:

    >>> import numpy as np
    >>> np.prod([1.,2.])
    2.0

    Even when the input array is two-dimensional:

    >>> a = np.array([[1., 2.], [3., 4.]])
    >>> np.prod(a)
    24.0

    But we can also specify the axis over which to multiply:

    >>> np.prod(a, axis=1)
    array([  2.,  12.])
    >>> np.prod(a, axis=0)
    array([3., 8.])

    Or select specific elements to include:

    >>> np.prod([1., np.nan, 3.], where=[True, False, True])
    3.0

    If the type of `x` is unsigned, then the output type is
    the unsigned platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.uint8)
    >>> np.prod(x).dtype == np.uint
    True

    If `x` is of a signed integer type, then the output type
    is the default platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.int8)
    >>> np.prod(x).dtype == int
    True

    You can also start the product with a value other than one:

    >>> np.prod([1, 2], initial=5)
    10
    """
    return _wrapreduction(a, np.multiply, 'prod', axis, dtype, out,
                          keepdims=keepdims, initial=initial, where=where)


def _cumprod_dispatcher(a, axis=None, dtype=None, out=None):
    return (a, out)


@array_function_dispatch(_cumprod_dispatcher)
def cumprod(a, axis=None, dtype=None, out=None):
    """
    Return the cumulative product of elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input array.
    axis : int, optional
        Axis along which the cumulative product is computed.  By default
        the input is flattened.
    dtype : dtype, optional
        Type of the returned array, as well as of the accumulator in which
        the elements are multiplied.  If *dtype* is not specified, it
        defaults to the dtype of `a`, unless `a` has an integer dtype with
        a precision less than that of the default platform integer.  In
        that case, the default platform integer is used instead.
    out : ndarray, optional
        Alternative output array in which to place the result. It must
        have the same shape and buffer length as the expected output
        but the type of the resulting values will be cast if necessary.

    Returns
    -------
    cumprod : ndarray
        A new array holding the result is returned unless `out` is
        specified, in which case a reference to out is returned.

    See Also
    --------
    cumulative_prod : Array API compatible alternative for ``cumprod``.
    :ref:`ufuncs-output-type`

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([1,2,3])
    >>> np.cumprod(a) # intermediate results 1, 1*2
    ...               # total product 1*2*3 = 6
    array([1, 2, 6])
    >>> a = np.array([[1, 2, 3], [4, 5, 6]])
    >>> np.cumprod(a, dtype=float) # specify type of output
    array([   1.,    2.,    6.,   24.,  120.,  720.])

    The cumulative product for each column (i.e., over the rows) of `a`:

    >>> np.cumprod(a, axis=0)
    array([[ 1,  2,  3],
           [ 4, 10, 18]])

    The cumulative product for each row (i.e. over the columns) of `a`:

    >>> np.cumprod(a,axis=1)
    array([[  1,   2,   6],
           [  4,  20, 120]])

    """
    return _wrapfunc(a, 'cumprod', axis=axis, dtype=dtype, out=out)


def _ndim_dispatcher(a):
    return (a,)


@array_function_dispatch(_ndim_dispatcher)
def ndim(a):
    """
    Return the number of dimensions of an array.

    Parameters
    ----------
    a : array_like
        Input array.  If it is not already an ndarray, a conversion is
        attempted.

    Returns
    -------
    number_of_dimensions : int
        The number of dimensions in `a`.  Scalars are zero-dimensional.

    See Also
    --------
    ndarray.ndim : equivalent method
    shape : dimensions of array
    ndarray.shape : dimensions of array

    Examples
    --------
    >>> import numpy as np
    >>> np.ndim([[1,2,3],[4,5,6]])
    2
    >>> np.ndim(np.array([[1,2,3],[4,5,6]]))
    2
    >>> np.ndim(1)
    0

    """
    try:
        return a.ndim
    except AttributeError:
        return asarray(a).ndim


def _size_dispatcher(a, axis=None):
    return (a,)


@array_function_dispatch(_size_dispatcher)
def size(a, axis=None):
    """
    Return the number of elements along a given axis.

    Parameters
    ----------
    a : array_like
        Input data.
    axis : int, optional
        Axis along which the elements are counted.  By default, give
        the total number of elements.

    Returns
    -------
    element_count : int
        Number of elements along the specified axis.

    See Also
    --------
    shape : dimensions of array
    ndarray.shape : dimensions of array
    ndarray.size : number of elements in array

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1,2,3],[4,5,6]])
    >>> np.size(a)
    6
    >>> np.size(a,1)
    3
    >>> np.size(a,0)
    2

    """
    if axis is None:
        try:
            return a.size
        except AttributeError:
            return asarray(a).size
    else:
        try:
            return a.shape[axis]
        except AttributeError:
            return asarray(a).shape[axis]


def _round_dispatcher(a, decimals=None, out=None):
    return (a, out)


@array_function_dispatch(_round_dispatcher)
def round(a, decimals=0, out=None):
    """
    Evenly round to the given number of decimals.

    Parameters
    ----------
    a : array_like
        Input data.
    decimals : int, optional
        Number of decimal places to round to (default: 0).  If
        decimals is negative, it specifies the number of positions to
        the left of the decimal point.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the output
        values will be cast if necessary. See :ref:`ufuncs-output-type`
        for more details.

    Returns
    -------
    rounded_array : ndarray
        An array of the same type as `a`, containing the rounded values.
        Unless `out` was specified, a new array is created.  A reference to
        the result is returned.

        The real and imaginary parts of complex numbers are rounded
        separately.  The result of rounding a float is a float.

    See Also
    --------
    ndarray.round : equivalent method
    around : an alias for this function
    ceil, fix, floor, rint, trunc


    Notes
    -----
    For values exactly halfway between rounded decimal values, NumPy
    rounds to the nearest even value. Thus 1.5 and 2.5 round to 2.0,
    -0.5 and 0.5 round to 0.0, etc.

    ``np.round`` uses a fast but sometimes inexact algorithm to round
    floating-point datatypes. For positive `decimals` it is equivalent to
    ``np.true_divide(np.rint(a * 10**decimals), 10**decimals)``, which has
    error due to the inexact representation of decimal fractions in the IEEE
    floating point standard [1]_ and errors introduced when scaling by powers
    of ten. For instance, note the extra "1" in the following:

        >>> np.round(56294995342131.5, 3)
        56294995342131.51

    If your goal is to print such values with a fixed number of decimals, it is
    preferable to use numpy's float printing routines to limit the number of
    printed decimals:

        >>> np.format_float_positional(56294995342131.5, precision=3)
        '56294995342131.5'

    The float printing routines use an accurate but much more computationally
    demanding algorithm to compute the number of digits after the decimal
    point.

    Alternatively, Python's builtin `round` function uses a more accurate
    but slower algorithm for 64-bit floating point values:

        >>> round(56294995342131.5, 3)
        56294995342131.5
        >>> np.round(16.055, 2), round(16.055, 2)  # equals 16.0549999999999997
        (16.06, 16.05)


    References
    ----------
    .. [1] "Lecture Notes on the Status of IEEE 754", William Kahan,
           https://people.eecs.berkeley.edu/~wkahan/ieee754status/IEEE754.PDF

    Examples
    --------
    >>> import numpy as np
    >>> np.round([0.37, 1.64])
    array([0., 2.])
    >>> np.round([0.37, 1.64], decimals=1)
    array([0.4, 1.6])
    >>> np.round([.5, 1.5, 2.5, 3.5, 4.5]) # rounds to nearest even value
    array([0., 2., 2., 4., 4.])
    >>> np.round([1,2,3,11], decimals=1) # ndarray of ints is returned
    array([ 1,  2,  3, 11])
    >>> np.round([1,2,3,11], decimals=-1)
    array([ 0,  0,  0, 10])

    """
    return _wrapfunc(a, 'round', decimals=decimals, out=out)


@array_function_dispatch(_round_dispatcher)
def around(a, decimals=0, out=None):
    """
    Round an array to the given number of decimals.

    `around` is an alias of `~numpy.round`.

    See Also
    --------
    ndarray.round : equivalent method
    round : alias for this function
    ceil, fix, floor, rint, trunc

    """
    return _wrapfunc(a, 'round', decimals=decimals, out=out)


def _mean_dispatcher(a, axis=None, dtype=None, out=None, keepdims=None, *,
                     where=None):
    return (a, where, out)


@array_function_dispatch(_mean_dispatcher)
def mean(a, axis=None, dtype=None, out=None, keepdims=np._NoValue, *,
         where=np._NoValue):
    """
    Compute the arithmetic mean along the specified axis.

    Returns the average of the array elements.  The average is taken over
    the flattened array by default, otherwise over the specified axis.
    `float64` intermediate and return values are used for integer inputs.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose mean is desired. If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the means are computed. The default is to
        compute the mean of the flattened array.

        If this is a tuple of ints, a mean is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the mean.  For integer inputs, the default
        is `float64`; for floating point inputs, it is the same as the
        input dtype.
    out : ndarray, optional
        Alternate output array in which to place the result.  The default
        is ``None``; if provided, it must have the same shape as the
        expected output, but the type will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
        See :ref:`ufuncs-output-type` for more details.

    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `mean` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.

    where : array_like of bool, optional
        Elements to include in the mean. See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    Returns
    -------
    m : ndarray, see dtype parameter above
        If `out=None`, returns a new array containing the mean values,
        otherwise a reference to the output array is returned.

    See Also
    --------
    average : Weighted average
    std, var, nanmean, nanstd, nanvar

    Notes
    -----
    The arithmetic mean is the sum of the elements along the axis divided
    by the number of elements.

    Note that for floating-point input, the mean is computed using the
    same precision the input has.  Depending on the input data, this can
    cause the results to be inaccurate, especially for `float32` (see
    example below).  Specifying a higher-precision accumulator using the
    `dtype` keyword can alleviate this issue.

    By default, `float16` results are computed using `float32` intermediates
    for extra precision.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.mean(a)
    2.5
    >>> np.mean(a, axis=0)
    array([2., 3.])
    >>> np.mean(a, axis=1)
    array([1.5, 3.5])

    In single precision, `mean` can be inaccurate:

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.mean(a)
    np.float32(0.54999924)

    Computing the mean in float64 is more accurate:

    >>> np.mean(a, dtype=np.float64)
    0.55000000074505806 # may vary

    Computing the mean in timedelta64 is available:

    >>> b = np.array([1, 3], dtype="timedelta64[D]")
    >>> np.mean(b)
    np.timedelta64(2,'D')

    Specifying a where argument:

    >>> a = np.array([[5, 9, 13], [14, 10, 12], [11, 15, 19]])
    >>> np.mean(a)
    12.0
    >>> np.mean(a, where=[[True], [False], [False]])
    9.0

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if where is not np._NoValue:
        kwargs['where'] = where
    if type(a) is not mu.ndarray:
        try:
            mean = a.mean
        except AttributeError:
            pass
        else:
            return mean(axis=axis, dtype=dtype, out=out, **kwargs)

    return _methods._mean(a, axis=axis, dtype=dtype,
                          out=out, **kwargs)


def _std_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                    keepdims=None, *, where=None, mean=None, correction=None):
    return (a, where, out, mean)


@array_function_dispatch(_std_dispatcher)
def std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, *,
        where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    r"""
    Compute the standard deviation along the specified axis.

    Returns the standard deviation, a measure of the spread of a distribution,
    of the array elements. The standard deviation is computed for the
    flattened array by default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Calculate the standard deviation of these values.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the standard deviation is computed. The
        default is to compute the standard deviation of the flattened array.
        If this is a tuple of ints, a standard deviation is performed over
        multiple axes, instead of a single axis or all the axes as before.
    dtype : dtype, optional
        Type to use in computing the standard deviation. For arrays of
        integer type the default is float64, for arrays of float types it is
        the same as the array type.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output but the type (of the calculated
        values) will be cast if necessary.
        See :ref:`ufuncs-output-type` for more details.
    ddof : {int, float}, optional
        Means Delta Degrees of Freedom.  The divisor used in calculations
        is ``N - ddof``, where ``N`` represents the number of elements.
        By default `ddof` is zero. See Notes for details about use of `ddof`.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `std` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the standard deviation.
        See `~numpy.ufunc.reduce` for details.

        .. versionadded:: 1.20.0

    mean : array_like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this std function.

        .. versionadded:: 2.0.0

    correction : {int, float}, optional
        Array API compatible name for the ``ddof`` parameter. Only one of them
        can be provided at the same time.

        .. versionadded:: 2.0.0

    Returns
    -------
    standard_deviation : ndarray, see dtype parameter above.
        If `out` is None, return a new array containing the standard deviation,
        otherwise return a reference to the output array.

    See Also
    --------
    var, mean, nanmean, nanstd, nanvar
    :ref:`ufuncs-output-type`

    Notes
    -----
    There are several common variants of the array standard deviation
    calculation. Assuming the input `a` is a one-dimensional NumPy array
    and ``mean`` is either provided as an argument or computed as
    ``a.mean()``, NumPy computes the standard deviation of an array as::

        N = len(a)
        d2 = abs(a - mean)**2  # abs is for complex `a`
        var = d2.sum() / (N - ddof)  # note use of `ddof`
        std = var**0.5

    Different values of the argument `ddof` are useful in different
    contexts. NumPy's default ``ddof=0`` corresponds with the expression:

    .. math::

        \sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N}}

    which is sometimes called the "population standard deviation" in the field
    of statistics because it applies the definition of standard deviation to
    `a` as if `a` were a complete population of possible observations.

    Many other libraries define the standard deviation of an array
    differently, e.g.:

    .. math::

        \sqrt{\frac{\sum_i{|a_i - \bar{a}|^2 }}{N - 1}}

    In statistics, the resulting quantity is sometimes called the "sample
    standard deviation" because if `a` is a random sample from a larger
    population, this calculation provides the square root of an unbiased
    estimate of the variance of the population. The use of :math:`N-1` in the
    denominator is often called "Bessel's correction" because it corrects for
    bias (toward lower values) in the variance estimate introduced when the
    sample mean of `a` is used in place of the true mean of the population.
    The resulting estimate of the standard deviation is still biased, but less
    than it would have been without the correction. For this quantity, use
    ``ddof=1``.

    Note that, for complex numbers, `std` takes the absolute
    value before squaring, so that the result is always real and nonnegative.

    For floating-point input, the standard deviation is computed using the same
    precision the input has. Depending on the input data, this can cause
    the results to be inaccurate, especially for float32 (see example below).
    Specifying a higher-accuracy accumulator using the `dtype` keyword can
    alleviate this issue.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.std(a)
    1.1180339887498949 # may vary
    >>> np.std(a, axis=0)
    array([1.,  1.])
    >>> np.std(a, axis=1)
    array([0.5,  0.5])

    In single precision, std() can be inaccurate:

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.std(a)
    np.float32(0.45000005)

    Computing the standard deviation in float64 is more accurate:

    >>> np.std(a, dtype=np.float64)
    0.44999999925494177 # may vary

    Specifying a where argument:

    >>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    >>> np.std(a)
    2.614064523559687 # may vary
    >>> np.std(a, where=[[True], [True], [False]])
    2.0

    Using the mean keyword to save computation time:

    >>> import numpy as np
    >>> from timeit import timeit
    >>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    >>> mean = np.mean(a, axis=1, keepdims=True)
    >>>
    >>> g = globals()
    >>> n = 10000
    >>> t1 = timeit("std = np.std(a, axis=1, mean=mean)", globals=g, number=n)
    >>> t2 = timeit("std = np.std(a, axis=1)", globals=g, number=n)
    >>> print(f'Percentage execution time saved {100*(t2-t1)/t2:.0f}%')
    #doctest: +SKIP
    Percentage execution time saved 30%

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if where is not np._NoValue:
        kwargs['where'] = where
    if mean is not np._NoValue:
        kwargs['mean'] = mean

    if correction != np._NoValue:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously."
            )
        else:
            ddof = correction

    if type(a) is not mu.ndarray:
        try:
            std = a.std
        except AttributeError:
            pass
        else:
            return std(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)

    return _methods._std(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                         **kwargs)


def _var_dispatcher(a, axis=None, dtype=None, out=None, ddof=None,
                    keepdims=None, *, where=None, mean=None, correction=None):
    return (a, where, out, mean)


@array_function_dispatch(_var_dispatcher)
def var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue, *,
        where=np._NoValue, mean=np._NoValue, correction=np._NoValue):
    r"""
    Compute the variance along the specified axis.

    Returns the variance of the array elements, a measure of the spread of a
    distribution.  The variance is computed for the flattened array by
    default, otherwise over the specified axis.

    Parameters
    ----------
    a : array_like
        Array containing numbers whose variance is desired.  If `a` is not an
        array, a conversion is attempted.
    axis : None or int or tuple of ints, optional
        Axis or axes along which the variance is computed.  The default is to
        compute the variance of the flattened array.
        If this is a tuple of ints, a variance is performed over multiple axes,
        instead of a single axis or all the axes as before.
    dtype : data-type, optional
        Type to use in computing the variance.  For arrays of integer type
        the default is `float64`; for arrays of float types it is the same as
        the array type.
    out : ndarray, optional
        Alternate output array in which to place the result.  It must have
        the same shape as the expected output, but the type is cast if
        necessary.
    ddof : {int, float}, optional
        "Delta Degrees of Freedom": the divisor used in the calculation is
        ``N - ddof``, where ``N`` represents the number of elements. By
        default `ddof` is zero. See notes for details about use of `ddof`.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `var` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-class' method does not implement `keepdims` any
        exceptions will be raised.
    where : array_like of bool, optional
        Elements to include in the variance. See `~numpy.ufunc.reduce` for
        details.

        .. versionadded:: 1.20.0

    mean : array like, optional
        Provide the mean to prevent its recalculation. The mean should have
        a shape as if it was calculated with ``keepdims=True``.
        The axis for the calculation of the mean should be the same as used in
        the call to this var function.

        .. versionadded:: 2.0.0

    correction : {int, float}, optional
        Array API compatible name for the ``ddof`` parameter. Only one of them
        can be provided at the same time.

        .. versionadded:: 2.0.0

    Returns
    -------
    variance : ndarray, see dtype parameter above
        If ``out=None``, returns a new array containing the variance;
        otherwise, a reference to the output array is returned.

    See Also
    --------
    std, mean, nanmean, nanstd, nanvar
    :ref:`ufuncs-output-type`

    Notes
    -----
    There are several common variants of the array variance calculation.
    Assuming the input `a` is a one-dimensional NumPy array and ``mean`` is
    either provided as an argument or computed as ``a.mean()``, NumPy
    computes the variance of an array as::

        N = len(a)
        d2 = abs(a - mean)**2  # abs is for complex `a`
        var = d2.sum() / (N - ddof)  # note use of `ddof`

    Different values of the argument `ddof` are useful in different
    contexts. NumPy's default ``ddof=0`` corresponds with the expression:

    .. math::

        \frac{\sum_i{|a_i - \bar{a}|^2 }}{N}

    which is sometimes called the "population variance" in the field of
    statistics because it applies the definition of variance to `a` as if `a`
    were a complete population of possible observations.

    Many other libraries define the variance of an array differently, e.g.:

    .. math::

        \frac{\sum_i{|a_i - \bar{a}|^2}}{N - 1}

    In statistics, the resulting quantity is sometimes called the "sample
    variance" because if `a` is a random sample from a larger population,
    this calculation provides an unbiased estimate of the variance of the
    population.  The use of :math:`N-1` in the denominator is often called
    "Bessel's correction" because it corrects for bias (toward lower values)
    in the variance estimate introduced when the sample mean of `a` is used
    in place of the true mean of the population. For this quantity, use
    ``ddof=1``.

    Note that for complex numbers, the absolute value is taken before
    squaring, so that the result is always real and nonnegative.

    For floating-point input, the variance is computed using the same
    precision the input has.  Depending on the input data, this can cause
    the results to be inaccurate, especially for `float32` (see example
    below).  Specifying a higher-accuracy accumulator using the ``dtype``
    keyword can alleviate this issue.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array([[1, 2], [3, 4]])
    >>> np.var(a)
    1.25
    >>> np.var(a, axis=0)
    array([1.,  1.])
    >>> np.var(a, axis=1)
    array([0.25,  0.25])

    In single precision, var() can be inaccurate:

    >>> a = np.zeros((2, 512*512), dtype=np.float32)
    >>> a[0, :] = 1.0
    >>> a[1, :] = 0.1
    >>> np.var(a)
    np.float32(0.20250003)

    Computing the variance in float64 is more accurate:

    >>> np.var(a, dtype=np.float64)
    0.20249999932944759 # may vary
    >>> ((1-0.55)**2 + (0.1-0.55)**2)/2
    0.2025

    Specifying a where argument:

    >>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    >>> np.var(a)
    6.833333333333333 # may vary
    >>> np.var(a, where=[[True], [True], [False]])
    4.0

    Using the mean keyword to save computation time:

    >>> import numpy as np
    >>> from timeit import timeit
    >>>
    >>> a = np.array([[14, 8, 11, 10], [7, 9, 10, 11], [10, 15, 5, 10]])
    >>> mean = np.mean(a, axis=1, keepdims=True)
    >>>
    >>> g = globals()
    >>> n = 10000
    >>> t1 = timeit("var = np.var(a, axis=1, mean=mean)", globals=g, number=n)
    >>> t2 = timeit("var = np.var(a, axis=1)", globals=g, number=n)
    >>> print(f'Percentage execution time saved {100*(t2-t1)/t2:.0f}%')
    #doctest: +SKIP
    Percentage execution time saved 32%

    """
    kwargs = {}
    if keepdims is not np._NoValue:
        kwargs['keepdims'] = keepdims
    if where is not np._NoValue:
        kwargs['where'] = where
    if mean is not np._NoValue:
        kwargs['mean'] = mean

    if correction != np._NoValue:
        if ddof != 0:
            raise ValueError(
                "ddof and correction can't be provided simultaneously."
            )
        else:
            ddof = correction

    if type(a) is not mu.ndarray:
        try:
            var = a.var

        except AttributeError:
            pass
        else:
            return var(axis=axis, dtype=dtype, out=out, ddof=ddof, **kwargs)

    return _methods._var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
                         **kwargs)