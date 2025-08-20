
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\soupsieve\css_types.py ===
"""CSS selector structure items."""
from __future__ import annotations
import copyreg
from .pretty import pretty
from typing import Any, Iterator, Hashable, Pattern, Iterable, Mapping

__all__ = (
    'Selector',
    'SelectorNull',
    'SelectorTag',
    'SelectorAttribute',
    'SelectorContains',
    'SelectorNth',
    'SelectorLang',
    'SelectorList',
    'Namespaces',
    'CustomSelectors'
)


SEL_EMPTY = 0x1
SEL_ROOT = 0x2
SEL_DEFAULT = 0x4
SEL_INDETERMINATE = 0x8
SEL_SCOPE = 0x10
SEL_DIR_LTR = 0x20
SEL_DIR_RTL = 0x40
SEL_IN_RANGE = 0x80
SEL_OUT_OF_RANGE = 0x100
SEL_DEFINED = 0x200
SEL_PLACEHOLDER_SHOWN = 0x400


class Immutable:
    """Immutable."""

    __slots__: tuple[str, ...] = ('_hash',)

    _hash: int

    def __init__(self, **kwargs: Any) -> None:
        """Initialize."""

        temp = []
        for k, v in kwargs.items():
            temp.append(type(v))
            temp.append(v)
            super().__setattr__(k, v)
        super().__setattr__('_hash', hash(tuple(temp)))

    @classmethod
    def __base__(cls) -> type[Immutable]:
        """Get base class."""

        return cls

    def __eq__(self, other: Any) -> bool:
        """Equal."""

        return (
            isinstance(other, self.__base__()) and
            all(getattr(other, key) == getattr(self, key) for key in self.__slots__ if key != '_hash')
        )

    def __ne__(self, other: Any) -> bool:
        """Equal."""

        return (
            not isinstance(other, self.__base__()) or
            any(getattr(other, key) != getattr(self, key) for key in self.__slots__ if key != '_hash')
        )

    def __hash__(self) -> int:
        """Hash."""

        return self._hash

    def __setattr__(self, name: str, value: Any) -> None:
        """Prevent mutability."""

        raise AttributeError(f"'{self.__class__.__name__}' is immutable")

    def __repr__(self) -> str:  # pragma: no cover
        """Representation."""

        r = ', '.join([f"{k}={getattr(self, k)!r}" for k in self.__slots__[:-1]])
        return f"{self.__class__.__name__}({r})"

    __str__ = __repr__

    def pretty(self) -> None:  # pragma: no cover
        """Pretty print."""

        print(pretty(self))


class ImmutableDict(Mapping[Any, Any]):
    """Hashable, immutable dictionary."""

    def __init__(
        self,
        arg: dict[Any, Any] | Iterable[tuple[Any, Any]]
    ) -> None:
        """Initialize."""

        self._validate(arg)
        self._d = dict(arg)
        self._hash = hash(tuple([(type(x), x, type(y), y) for x, y in sorted(self._d.items())]))

    def _validate(self, arg: dict[Any, Any] | Iterable[tuple[Any, Any]]) -> None:
        """Validate arguments."""

        if isinstance(arg, dict):
            if not all(isinstance(v, Hashable) for v in arg.values()):
                raise TypeError(f'{self.__class__.__name__} values must be hashable')
        elif not all(isinstance(k, Hashable) and isinstance(v, Hashable) for k, v in arg):
            raise TypeError(f'{self.__class__.__name__} values must be hashable')

    def __iter__(self) -> Iterator[Any]:
        """Iterator."""

        return iter(self._d)

    def __len__(self) -> int:
        """Length."""

        return len(self._d)

    def __getitem__(self, key: Any) -> Any:
        """Get item: `namespace['key']`."""

        return self._d[key]

    def __hash__(self) -> int:
        """Hash."""

        return self._hash

    def __repr__(self) -> str:  # pragma: no cover
        """Representation."""

        return f"{self._d!r}"

    __str__ = __repr__


class Namespaces(ImmutableDict):
    """Namespaces."""

    def __init__(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        """Initialize."""

        super().__init__(arg)

    def _validate(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        """Validate arguments."""

        if isinstance(arg, dict):
            if not all(isinstance(v, str) for v in arg.values()):
                raise TypeError(f'{self.__class__.__name__} values must be hashable')
        elif not all(isinstance(k, str) and isinstance(v, str) for k, v in arg):
            raise TypeError(f'{self.__class__.__name__} keys and values must be Unicode strings')


class CustomSelectors(ImmutableDict):
    """Custom selectors."""

    def __init__(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        """Initialize."""

        super().__init__(arg)

    def _validate(self, arg: dict[str, str] | Iterable[tuple[str, str]]) -> None:
        """Validate arguments."""

        if isinstance(arg, dict):
            if not all(isinstance(v, str) for v in arg.values()):
                raise TypeError(f'{self.__class__.__name__} values must be hashable')
        elif not all(isinstance(k, str) and isinstance(v, str) for k, v in arg):
            raise TypeError(f'{self.__class__.__name__} keys and values must be Unicode strings')


class Selector(Immutable):
    """Selector."""

    __slots__ = (
        'tag', 'ids', 'classes', 'attributes', 'nth', 'selectors',
        'relation', 'rel_type', 'contains', 'lang', 'flags', '_hash'
    )

    tag: SelectorTag | None
    ids: tuple[str, ...]
    classes: tuple[str, ...]
    attributes: tuple[SelectorAttribute, ...]
    nth: tuple[SelectorNth, ...]
    selectors: tuple[SelectorList, ...]
    relation: SelectorList
    rel_type: str | None
    contains: tuple[SelectorContains, ...]
    lang: tuple[SelectorLang, ...]
    flags: int

    def __init__(
        self,
        tag: SelectorTag | None,
        ids: tuple[str, ...],
        classes: tuple[str, ...],
        attributes: tuple[SelectorAttribute, ...],
        nth: tuple[SelectorNth, ...],
        selectors: tuple[SelectorList, ...],
        relation: SelectorList,
        rel_type: str | None,
        contains: tuple[SelectorContains, ...],
        lang: tuple[SelectorLang, ...],
        flags: int
    ):
        """Initialize."""

        super().__init__(
            tag=tag,
            ids=ids,
            classes=classes,
            attributes=attributes,
            nth=nth,
            selectors=selectors,
            relation=relation,
            rel_type=rel_type,
            contains=contains,
            lang=lang,
            flags=flags
        )


class SelectorNull(Immutable):
    """Null Selector."""

    def __init__(self) -> None:
        """Initialize."""

        super().__init__()


class SelectorTag(Immutable):
    """Selector tag."""

    __slots__ = ("name", "prefix", "_hash")

    name: str
    prefix: str | None

    def __init__(self, name: str, prefix: str | None) -> None:
        """Initialize."""

        super().__init__(name=name, prefix=prefix)


class SelectorAttribute(Immutable):
    """Selector attribute rule."""

    __slots__ = ("attribute", "prefix", "pattern", "xml_type_pattern", "_hash")

    attribute: str
    prefix: str
    pattern: Pattern[str] | None
    xml_type_pattern: Pattern[str] | None

    def __init__(
        self,
        attribute: str,
        prefix: str,
        pattern: Pattern[str] | None,
        xml_type_pattern: Pattern[str] | None
    ) -> None:
        """Initialize."""

        super().__init__(
            attribute=attribute,
            prefix=prefix,
            pattern=pattern,
            xml_type_pattern=xml_type_pattern
        )


class SelectorContains(Immutable):
    """Selector contains rule."""

    __slots__ = ("text", "own", "_hash")

    text: tuple[str, ...]
    own: bool

    def __init__(self, text: Iterable[str], own: bool) -> None:
        """Initialize."""

        super().__init__(text=tuple(text), own=own)


class SelectorNth(Immutable):
    """Selector nth type."""

    __slots__ = ("a", "n", "b", "of_type", "last", "selectors", "_hash")

    a: int
    n: bool
    b: int
    of_type: bool
    last: bool
    selectors: SelectorList

    def __init__(self, a: int, n: bool, b: int, of_type: bool, last: bool, selectors: SelectorList) -> None:
        """Initialize."""

        super().__init__(
            a=a,
            n=n,
            b=b,
            of_type=of_type,
            last=last,
            selectors=selectors
        )


class SelectorLang(Immutable):
    """Selector language rules."""

    __slots__ = ("languages", "_hash",)

    languages: tuple[str, ...]

    def __init__(self, languages: Iterable[str]):
        """Initialize."""

        super().__init__(languages=tuple(languages))

    def __iter__(self) -> Iterator[str]:
        """Iterator."""

        return iter(self.languages)

    def __len__(self) -> int:  # pragma: no cover
        """Length."""

        return len(self.languages)

    def __getitem__(self, index: int) -> str:  # pragma: no cover
        """Get item."""

        return self.languages[index]


class SelectorList(Immutable):
    """Selector list."""

    __slots__ = ("selectors", "is_not", "is_html", "_hash")

    selectors: tuple[Selector | SelectorNull, ...]
    is_not: bool
    is_html: bool

    def __init__(
        self,
        selectors: Iterable[Selector | SelectorNull] | None = None,
        is_not: bool = False,
        is_html: bool = False
    ) -> None:
        """Initialize."""

        super().__init__(
            selectors=tuple(selectors) if selectors is not None else (),
            is_not=is_not,
            is_html=is_html
        )

    def __iter__(self) -> Iterator[Selector | SelectorNull]:
        """Iterator."""

        return iter(self.selectors)

    def __len__(self) -> int:
        """Length."""

        return len(self.selectors)

    def __getitem__(self, index: int) -> Selector | SelectorNull:
        """Get item."""

        return self.selectors[index]


def _pickle(p: Any) -> Any:
    return p.__base__(), tuple([getattr(p, s) for s in p.__slots__[:-1]])


def pickle_register(obj: Any) -> None:
    """Allow object to be pickled."""

    copyreg.pickle(obj, _pickle)


pickle_register(Selector)
pickle_register(SelectorNull)
pickle_register(SelectorTag)
pickle_register(SelectorAttribute)
pickle_register(SelectorContains)
pickle_register(SelectorNth)
pickle_register(SelectorLang)
pickle_register(SelectorList)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\definitions\unit_definitions.py ===
from sympy.physics.units.definitions.dimension_definitions import current, temperature, amount_of_substance, \
    luminous_intensity, angle, charge, voltage, impedance, conductance, capacitance, inductance, magnetic_density, \
    magnetic_flux, information

from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S as S_singleton
from sympy.physics.units.prefixes import kilo, mega, milli, micro, deci, centi, nano, pico, kibi, mebi, gibi, tebi, pebi, exbi
from sympy.physics.units.quantities import PhysicalConstant, Quantity

One = S_singleton.One

#### UNITS ####

# Dimensionless:
percent = percents = Quantity("percent", latex_repr=r"\%")
percent.set_global_relative_scale_factor(Rational(1, 100), One)

permille = Quantity("permille")
permille.set_global_relative_scale_factor(Rational(1, 1000), One)


# Angular units (dimensionless)
rad = radian = radians = Quantity("radian", abbrev="rad")
radian.set_global_dimension(angle)
deg = degree = degrees = Quantity("degree", abbrev="deg", latex_repr=r"^\circ")
degree.set_global_relative_scale_factor(pi/180, radian)
sr = steradian = steradians = Quantity("steradian", abbrev="sr")
mil = angular_mil = angular_mils = Quantity("angular_mil", abbrev="mil")

# Base units:
m = meter = meters = Quantity("meter", abbrev="m")

# gram; used to define its prefixed units
g = gram = grams = Quantity("gram", abbrev="g")

# NOTE: the `kilogram` has scale factor 1000. In SI, kg is a base unit, but
# nonetheless we are trying to be compatible with the `kilo` prefix. In a
# similar manner, people using CGS or gaussian units could argue that the
# `centimeter` rather than `meter` is the fundamental unit for length, but the
# scale factor of `centimeter` will be kept as 1/100 to be compatible with the
# `centi` prefix.  The current state of the code assumes SI unit dimensions, in
# the future this module will be modified in order to be unit system-neutral
# (that is, support all kinds of unit systems).
kg = kilogram = kilograms = Quantity("kilogram", abbrev="kg")
kg.set_global_relative_scale_factor(kilo, gram)

s = second = seconds = Quantity("second", abbrev="s")
A = ampere = amperes = Quantity("ampere", abbrev='A')
ampere.set_global_dimension(current)
K = kelvin = kelvins = Quantity("kelvin", abbrev='K')
kelvin.set_global_dimension(temperature)
mol = mole = moles = Quantity("mole", abbrev="mol")
mole.set_global_dimension(amount_of_substance)
cd = candela = candelas = Quantity("candela", abbrev="cd")
candela.set_global_dimension(luminous_intensity)

# derived units
newton = newtons = N = Quantity("newton", abbrev="N")

kilonewton = kilonewtons = kN = Quantity("kilonewton", abbrev="kN")
kilonewton.set_global_relative_scale_factor(kilo, newton)

meganewton = meganewtons = MN = Quantity("meganewton", abbrev="MN")
meganewton.set_global_relative_scale_factor(mega, newton)

joule = joules = J = Quantity("joule", abbrev="J")
watt = watts = W = Quantity("watt", abbrev="W")
pascal = pascals = Pa = pa = Quantity("pascal", abbrev="Pa")
hertz = hz = Hz = Quantity("hertz", abbrev="Hz")

# CGS derived units:
dyne = Quantity("dyne")
dyne.set_global_relative_scale_factor(One/10**5, newton)
erg = Quantity("erg")
erg.set_global_relative_scale_factor(One/10**7, joule)

# MKSA extension to MKS: derived units
coulomb = coulombs = C = Quantity("coulomb", abbrev='C')
coulomb.set_global_dimension(charge)
volt = volts = v = V = Quantity("volt", abbrev='V')
volt.set_global_dimension(voltage)
ohm = ohms = Quantity("ohm", abbrev='ohm', latex_repr=r"\Omega")
ohm.set_global_dimension(impedance)
siemens = S = mho = mhos = Quantity("siemens", abbrev='S')
siemens.set_global_dimension(conductance)
farad = farads = F = Quantity("farad", abbrev='F')
farad.set_global_dimension(capacitance)
henry = henrys = H = Quantity("henry", abbrev='H')
henry.set_global_dimension(inductance)
tesla = teslas = T = Quantity("tesla", abbrev='T')
tesla.set_global_dimension(magnetic_density)
weber = webers = Wb = wb = Quantity("weber", abbrev='Wb')
weber.set_global_dimension(magnetic_flux)

# CGS units for electromagnetic quantities:
statampere = Quantity("statampere")
statcoulomb = statC = franklin = Quantity("statcoulomb", abbrev="statC")
statvolt = Quantity("statvolt")
gauss = Quantity("gauss")
maxwell = Quantity("maxwell")
debye = Quantity("debye")
oersted = Quantity("oersted")

# Other derived units:
optical_power = dioptre = diopter = D = Quantity("dioptre")
lux = lx = Quantity("lux", abbrev="lx")

# katal is the SI unit of catalytic activity
katal = kat = Quantity("katal", abbrev="kat")

# gray is the SI unit of absorbed dose
gray = Gy = Quantity("gray")

# becquerel is the SI unit of radioactivity
becquerel = Bq = Quantity("becquerel", abbrev="Bq")


# Common mass units

mg = milligram = milligrams = Quantity("milligram", abbrev="mg")
mg.set_global_relative_scale_factor(milli, gram)

ug = microgram = micrograms = Quantity("microgram", abbrev="ug", latex_repr=r"\mu\text{g}")
ug.set_global_relative_scale_factor(micro, gram)

# Atomic mass constant
Da = dalton = amu = amus = atomic_mass_unit = atomic_mass_constant = PhysicalConstant("atomic_mass_constant")

t = metric_ton = tonne = Quantity("tonne", abbrev="t")
tonne.set_global_relative_scale_factor(mega, gram)

# Electron rest mass
me = electron_rest_mass = Quantity("electron_rest_mass", abbrev="me")


# Common length units

km = kilometer = kilometers = Quantity("kilometer", abbrev="km")
km.set_global_relative_scale_factor(kilo, meter)

dm = decimeter = decimeters = Quantity("decimeter", abbrev="dm")
dm.set_global_relative_scale_factor(deci, meter)

cm = centimeter = centimeters = Quantity("centimeter", abbrev="cm")
cm.set_global_relative_scale_factor(centi, meter)

mm = millimeter = millimeters = Quantity("millimeter", abbrev="mm")
mm.set_global_relative_scale_factor(milli, meter)

um = micrometer = micrometers = micron = microns = \
    Quantity("micrometer", abbrev="um", latex_repr=r'\mu\text{m}')
um.set_global_relative_scale_factor(micro, meter)

nm = nanometer = nanometers = Quantity("nanometer", abbrev="nm")
nm.set_global_relative_scale_factor(nano, meter)

pm = picometer = picometers = Quantity("picometer", abbrev="pm")
pm.set_global_relative_scale_factor(pico, meter)

ft = foot = feet = Quantity("foot", abbrev="ft")
ft.set_global_relative_scale_factor(Rational(3048, 10000), meter)

inch = inches = Quantity("inch")
inch.set_global_relative_scale_factor(Rational(1, 12), foot)

yd = yard = yards = Quantity("yard", abbrev="yd")
yd.set_global_relative_scale_factor(3, feet)

mi = mile = miles = Quantity("mile")
mi.set_global_relative_scale_factor(5280, feet)

nmi = nautical_mile = nautical_miles = Quantity("nautical_mile")
nmi.set_global_relative_scale_factor(6076, feet)

angstrom = angstroms = Quantity("angstrom", latex_repr=r'\r{A}')
angstrom.set_global_relative_scale_factor(Rational(1, 10**10), meter)


# Common volume and area units

ha = hectare = Quantity("hectare", abbrev="ha")

l = L = liter = liters = Quantity("liter", abbrev="l")

dl = dL = deciliter = deciliters = Quantity("deciliter", abbrev="dl")
dl.set_global_relative_scale_factor(Rational(1, 10), liter)

cl = cL = centiliter = centiliters = Quantity("centiliter", abbrev="cl")
cl.set_global_relative_scale_factor(Rational(1, 100), liter)

ml = mL = milliliter = milliliters = Quantity("milliliter", abbrev="ml")
ml.set_global_relative_scale_factor(Rational(1, 1000), liter)


# Common time units

ms = millisecond = milliseconds = Quantity("millisecond", abbrev="ms")
millisecond.set_global_relative_scale_factor(milli, second)

us = microsecond = microseconds = Quantity("microsecond", abbrev="us", latex_repr=r'\mu\text{s}')
microsecond.set_global_relative_scale_factor(micro, second)

ns = nanosecond = nanoseconds = Quantity("nanosecond", abbrev="ns")
nanosecond.set_global_relative_scale_factor(nano, second)

ps = picosecond = picoseconds = Quantity("picosecond", abbrev="ps")
picosecond.set_global_relative_scale_factor(pico, second)

minute = minutes = Quantity("minute")
minute.set_global_relative_scale_factor(60, second)

h = hour = hours = Quantity("hour")
hour.set_global_relative_scale_factor(60, minute)

day = days = Quantity("day")
day.set_global_relative_scale_factor(24, hour)

anomalistic_year = anomalistic_years = Quantity("anomalistic_year")
anomalistic_year.set_global_relative_scale_factor(365.259636, day)

sidereal_year = sidereal_years = Quantity("sidereal_year")
sidereal_year.set_global_relative_scale_factor(31558149.540, seconds)

tropical_year = tropical_years = Quantity("tropical_year")
tropical_year.set_global_relative_scale_factor(365.24219, day)

common_year = common_years = Quantity("common_year")
common_year.set_global_relative_scale_factor(365, day)

julian_year = julian_years = Quantity("julian_year")
julian_year.set_global_relative_scale_factor((365 + One/4), day)

draconic_year = draconic_years = Quantity("draconic_year")
draconic_year.set_global_relative_scale_factor(346.62, day)

gaussian_year = gaussian_years = Quantity("gaussian_year")
gaussian_year.set_global_relative_scale_factor(365.2568983, day)

full_moon_cycle = full_moon_cycles = Quantity("full_moon_cycle")
full_moon_cycle.set_global_relative_scale_factor(411.78443029, day)

year = years = tropical_year


#### CONSTANTS ####

# Newton constant
G = gravitational_constant = PhysicalConstant("gravitational_constant", abbrev="G")

# speed of light
c = speed_of_light = PhysicalConstant("speed_of_light", abbrev="c")

# elementary charge
elementary_charge = PhysicalConstant("elementary_charge", abbrev="e")

# Planck constant
planck = PhysicalConstant("planck", abbrev="h")

# Reduced Planck constant
hbar = PhysicalConstant("hbar", abbrev="hbar")

# Electronvolt
eV = electronvolt = electronvolts = PhysicalConstant("electronvolt", abbrev="eV")

# Avogadro number
avogadro_number = PhysicalConstant("avogadro_number")

# Avogadro constant
avogadro = avogadro_constant = PhysicalConstant("avogadro_constant")

# Boltzmann constant
boltzmann = boltzmann_constant = PhysicalConstant("boltzmann_constant")

# Stefan-Boltzmann constant
stefan = stefan_boltzmann_constant = PhysicalConstant("stefan_boltzmann_constant")

# Molar gas constant
R = molar_gas_constant = PhysicalConstant("molar_gas_constant", abbrev="R")

# Faraday constant
faraday_constant = PhysicalConstant("faraday_constant")

# Josephson constant
josephson_constant = PhysicalConstant("josephson_constant", abbrev="K_j")

# Von Klitzing constant
von_klitzing_constant = PhysicalConstant("von_klitzing_constant", abbrev="R_k")

# Acceleration due to gravity (on the Earth surface)
gee = gees = acceleration_due_to_gravity = PhysicalConstant("acceleration_due_to_gravity", abbrev="g")

# magnetic constant:
u0 = magnetic_constant = vacuum_permeability = PhysicalConstant("magnetic_constant")

# electric constat:
e0 = electric_constant = vacuum_permittivity = PhysicalConstant("vacuum_permittivity")

# vacuum impedance:
Z0 = vacuum_impedance = PhysicalConstant("vacuum_impedance", abbrev='Z_0', latex_repr=r'Z_{0}')

# Coulomb's constant:
coulomb_constant = coulombs_constant = electric_force_constant = \
    PhysicalConstant("coulomb_constant", abbrev="k_e")


atmosphere = atmospheres = atm = Quantity("atmosphere", abbrev="atm")

kPa = kilopascal = Quantity("kilopascal", abbrev="kPa")
kilopascal.set_global_relative_scale_factor(kilo, Pa)

bar = bars = Quantity("bar", abbrev="bar")

pound = pounds = Quantity("pound")  # exact

psi = Quantity("psi")

dHg0 = 13.5951  # approx value at 0 C
mmHg = torr = Quantity("mmHg")

atmosphere.set_global_relative_scale_factor(101325, pascal)
bar.set_global_relative_scale_factor(100, kPa)
pound.set_global_relative_scale_factor(Rational(45359237, 100000000), kg)

mmu = mmus = milli_mass_unit = Quantity("milli_mass_unit")

quart = quarts = Quantity("quart")


# Other convenient units and magnitudes

ly = lightyear = lightyears = Quantity("lightyear", abbrev="ly")

au = astronomical_unit = astronomical_units = Quantity("astronomical_unit", abbrev="AU")


# Fundamental Planck units:
planck_mass = Quantity("planck_mass", abbrev="m_P", latex_repr=r'm_\text{P}')

planck_time = Quantity("planck_time", abbrev="t_P", latex_repr=r't_\text{P}')

planck_temperature = Quantity("planck_temperature", abbrev="T_P",
                              latex_repr=r'T_\text{P}')

planck_length = Quantity("planck_length", abbrev="l_P", latex_repr=r'l_\text{P}')

planck_charge = Quantity("planck_charge", abbrev="q_P", latex_repr=r'q_\text{P}')


# Derived Planck units:
planck_area = Quantity("planck_area")

planck_volume = Quantity("planck_volume")

planck_momentum = Quantity("planck_momentum")

planck_energy = Quantity("planck_energy", abbrev="E_P", latex_repr=r'E_\text{P}')

planck_force = Quantity("planck_force", abbrev="F_P", latex_repr=r'F_\text{P}')

planck_power = Quantity("planck_power", abbrev="P_P", latex_repr=r'P_\text{P}')

planck_density = Quantity("planck_density", abbrev="rho_P", latex_repr=r'\rho_\text{P}')

planck_energy_density = Quantity("planck_energy_density", abbrev="rho^E_P")

planck_intensity = Quantity("planck_intensity", abbrev="I_P", latex_repr=r'I_\text{P}')

planck_angular_frequency = Quantity("planck_angular_frequency", abbrev="omega_P",
                                    latex_repr=r'\omega_\text{P}')

planck_pressure = Quantity("planck_pressure", abbrev="p_P", latex_repr=r'p_\text{P}')

planck_current = Quantity("planck_current", abbrev="I_P", latex_repr=r'I_\text{P}')

planck_voltage = Quantity("planck_voltage", abbrev="V_P", latex_repr=r'V_\text{P}')

planck_impedance = Quantity("planck_impedance", abbrev="Z_P", latex_repr=r'Z_\text{P}')

planck_acceleration = Quantity("planck_acceleration", abbrev="a_P",
                               latex_repr=r'a_\text{P}')


# Information theory units:
bit = bits = Quantity("bit")
bit.set_global_dimension(information)

byte = bytes = Quantity("byte")

kibibyte = kibibytes = Quantity("kibibyte")
mebibyte = mebibytes = Quantity("mebibyte")
gibibyte = gibibytes = Quantity("gibibyte")
tebibyte = tebibytes = Quantity("tebibyte")
pebibyte = pebibytes = Quantity("pebibyte")
exbibyte = exbibytes = Quantity("exbibyte")

byte.set_global_relative_scale_factor(8, bit)
kibibyte.set_global_relative_scale_factor(kibi, byte)
mebibyte.set_global_relative_scale_factor(mebi, byte)
gibibyte.set_global_relative_scale_factor(gibi, byte)
tebibyte.set_global_relative_scale_factor(tebi, byte)
pebibyte.set_global_relative_scale_factor(pebi, byte)
exbibyte.set_global_relative_scale_factor(exbi, byte)

# Older units for radioactivity
curie = Ci = Quantity("curie", abbrev="Ci")

rutherford = Rd = Quantity("rutherford", abbrev="Rd")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\itsdangerous\serializer.py ===
from __future__ import annotations

import collections.abc as cabc
import json
import typing as t

from .encoding import want_bytes
from .exc import BadPayload
from .exc import BadSignature
from .signer import _make_keys_list
from .signer import Signer

if t.TYPE_CHECKING:
    import typing_extensions as te

    # This should be either be str or bytes. To avoid having to specify the
    # bound type, it falls back to a union if structural matching fails.
    _TSerialized = te.TypeVar(
        "_TSerialized", bound=t.Union[str, bytes], default=t.Union[str, bytes]
    )
else:
    # Still available at runtime on Python < 3.13, but without the default.
    _TSerialized = t.TypeVar("_TSerialized", bound=t.Union[str, bytes])


class _PDataSerializer(t.Protocol[_TSerialized]):
    def loads(self, payload: _TSerialized, /) -> t.Any: ...
    # A signature with additional arguments is not handled correctly by type
    # checkers right now, so an overload is used below for serializers that
    # don't match this strict protocol.
    def dumps(self, obj: t.Any, /) -> _TSerialized: ...


# Use TypeIs once it's available in typing_extensions or 3.13.
def is_text_serializer(
    serializer: _PDataSerializer[t.Any],
) -> te.TypeGuard[_PDataSerializer[str]]:
    """Checks whether a serializer generates text or binary."""
    return isinstance(serializer.dumps({}), str)


class Serializer(t.Generic[_TSerialized]):
    """A serializer wraps a :class:`~itsdangerous.signer.Signer` to
    enable serializing and securely signing data other than bytes. It
    can unsign to verify that the data hasn't been changed.

    The serializer provides :meth:`dumps` and :meth:`loads`, similar to
    :mod:`json`, and by default uses :mod:`json` internally to serialize
    the data to bytes.

    The secret key should be a random string of ``bytes`` and should not
    be saved to code or version control. Different salts should be used
    to distinguish signing in different contexts. See :doc:`/concepts`
    for information about the security of the secret key and salt.

    :param secret_key: The secret key to sign and verify with. Can be a
        list of keys, oldest to newest, to support key rotation.
    :param salt: Extra key to combine with ``secret_key`` to distinguish
        signatures in different contexts.
    :param serializer: An object that provides ``dumps`` and ``loads``
        methods for serializing data to a string. Defaults to
        :attr:`default_serializer`, which defaults to :mod:`json`.
    :param serializer_kwargs: Keyword arguments to pass when calling
        ``serializer.dumps``.
    :param signer: A ``Signer`` class to instantiate when signing data.
        Defaults to :attr:`default_signer`, which defaults to
        :class:`~itsdangerous.signer.Signer`.
    :param signer_kwargs: Keyword arguments to pass when instantiating
        the ``Signer`` class.
    :param fallback_signers: List of signer parameters to try when
        unsigning with the default signer fails. Each item can be a dict
        of ``signer_kwargs``, a ``Signer`` class, or a tuple of
        ``(signer, signer_kwargs)``. Defaults to
        :attr:`default_fallback_signers`.

    .. versionchanged:: 2.0
        Added support for key rotation by passing a list to
        ``secret_key``.

    .. versionchanged:: 2.0
        Removed the default SHA-512 fallback signer from
        ``default_fallback_signers``.

    .. versionchanged:: 1.1
        Added support for ``fallback_signers`` and configured a default
        SHA-512 fallback. This fallback is for users who used the yanked
        1.0.0 release which defaulted to SHA-512.

    .. versionchanged:: 0.14
        The ``signer`` and ``signer_kwargs`` parameters were added to
        the constructor.
    """

    #: The default serialization module to use to serialize data to a
    #: string internally. The default is :mod:`json`, but can be changed
    #: to any object that provides ``dumps`` and ``loads`` methods.
    default_serializer: _PDataSerializer[t.Any] = json

    #: The default ``Signer`` class to instantiate when signing data.
    #: The default is :class:`itsdangerous.signer.Signer`.
    default_signer: type[Signer] = Signer

    #: The default fallback signers to try when unsigning fails.
    default_fallback_signers: list[
        dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
    ] = []

    # Serializer[str] if no data serializer is provided, or if it returns str.
    @t.overload
    def __init__(
        self: Serializer[str],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        serializer: None | _PDataSerializer[str] = None,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    # Serializer[bytes] with a bytes data serializer positional argument.
    @t.overload
    def __init__(
        self: Serializer[bytes],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None,
        serializer: _PDataSerializer[bytes],
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    # Serializer[bytes] with a bytes data serializer keyword argument.
    @t.overload
    def __init__(
        self: Serializer[bytes],
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        *,
        serializer: _PDataSerializer[bytes],
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    # Fall back with a positional argument. If the strict signature of
    # _PDataSerializer doesn't match, fall back to a union, requiring the user
    # to specify the type.
    @t.overload
    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None,
        serializer: t.Any,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    # Fall back with a keyword argument.
    @t.overload
    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        *,
        serializer: t.Any,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ): ...

    def __init__(
        self,
        secret_key: str | bytes | cabc.Iterable[str] | cabc.Iterable[bytes],
        salt: str | bytes | None = b"itsdangerous",
        serializer: t.Any | None = None,
        serializer_kwargs: dict[str, t.Any] | None = None,
        signer: type[Signer] | None = None,
        signer_kwargs: dict[str, t.Any] | None = None,
        fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ]
        | None = None,
    ):
        #: The list of secret keys to try for verifying signatures, from
        #: oldest to newest. The newest (last) key is used for signing.
        #:
        #: This allows a key rotation system to keep a list of allowed
        #: keys and remove expired ones.
        self.secret_keys: list[bytes] = _make_keys_list(secret_key)

        if salt is not None:
            salt = want_bytes(salt)
            # if salt is None then the signer's default is used

        self.salt = salt

        if serializer is None:
            serializer = self.default_serializer

        self.serializer: _PDataSerializer[_TSerialized] = serializer
        self.is_text_serializer: bool = is_text_serializer(serializer)

        if signer is None:
            signer = self.default_signer

        self.signer: type[Signer] = signer
        self.signer_kwargs: dict[str, t.Any] = signer_kwargs or {}

        if fallback_signers is None:
            fallback_signers = list(self.default_fallback_signers)

        self.fallback_signers: list[
            dict[str, t.Any] | tuple[type[Signer], dict[str, t.Any]] | type[Signer]
        ] = fallback_signers
        self.serializer_kwargs: dict[str, t.Any] = serializer_kwargs or {}

    @property
    def secret_key(self) -> bytes:
        """The newest (last) entry in the :attr:`secret_keys` list. This
        is for compatibility from before key rotation support was added.
        """
        return self.secret_keys[-1]

    def load_payload(
        self, payload: bytes, serializer: _PDataSerializer[t.Any] | None = None
    ) -> t.Any:
        """Loads the encoded object. This function raises
        :class:`.BadPayload` if the payload is not valid. The
        ``serializer`` parameter can be used to override the serializer
        stored on the class. The encoded ``payload`` should always be
        bytes.
        """
        if serializer is None:
            use_serializer = self.serializer
            is_text = self.is_text_serializer
        else:
            use_serializer = serializer
            is_text = is_text_serializer(serializer)

        try:
            if is_text:
                return use_serializer.loads(payload.decode("utf-8"))  # type: ignore[arg-type]

            return use_serializer.loads(payload)  # type: ignore[arg-type]
        except Exception as e:
            raise BadPayload(
                "Could not load the payload because an exception"
                " occurred on unserializing the data.",
                original_error=e,
            ) from e

    def dump_payload(self, obj: t.Any) -> bytes:
        """Dumps the encoded object. The return value is always bytes.
        If the internal serializer returns text, the value will be
        encoded as UTF-8.
        """
        return want_bytes(self.serializer.dumps(obj, **self.serializer_kwargs))

    def make_signer(self, salt: str | bytes | None = None) -> Signer:
        """Creates a new instance of the signer to be used. The default
        implementation uses the :class:`.Signer` base class.
        """
        if salt is None:
            salt = self.salt

        return self.signer(self.secret_keys, salt=salt, **self.signer_kwargs)

    def iter_unsigners(self, salt: str | bytes | None = None) -> cabc.Iterator[Signer]:
        """Iterates over all signers to be tried for unsigning. Starts
        with the configured signer, then constructs each signer
        specified in ``fallback_signers``.
        """
        if salt is None:
            salt = self.salt

        yield self.make_signer(salt)

        for fallback in self.fallback_signers:
            if isinstance(fallback, dict):
                kwargs = fallback
                fallback = self.signer
            elif isinstance(fallback, tuple):
                fallback, kwargs = fallback
            else:
                kwargs = self.signer_kwargs

            for secret_key in self.secret_keys:
                yield fallback(secret_key, salt=salt, **kwargs)

    def dumps(self, obj: t.Any, salt: str | bytes | None = None) -> _TSerialized:
        """Returns a signed string serialized with the internal
        serializer. The return value can be either a byte or unicode
        string depending on the format of the internal serializer.
        """
        payload = want_bytes(self.dump_payload(obj))
        rv = self.make_signer(salt).sign(payload)

        if self.is_text_serializer:
            return rv.decode("utf-8")  # type: ignore[return-value]

        return rv  # type: ignore[return-value]

    def dump(self, obj: t.Any, f: t.IO[t.Any], salt: str | bytes | None = None) -> None:
        """Like :meth:`dumps` but dumps into a file. The file handle has
        to be compatible with what the internal serializer expects.
        """
        f.write(self.dumps(obj, salt))

    def loads(
        self, s: str | bytes, salt: str | bytes | None = None, **kwargs: t.Any
    ) -> t.Any:
        """Reverse of :meth:`dumps`. Raises :exc:`.BadSignature` if the
        signature validation fails.
        """
        s = want_bytes(s)
        last_exception = None

        for signer in self.iter_unsigners(salt):
            try:
                return self.load_payload(signer.unsign(s))
            except BadSignature as err:
                last_exception = err

        raise t.cast(BadSignature, last_exception)

    def load(self, f: t.IO[t.Any], salt: str | bytes | None = None) -> t.Any:
        """Like :meth:`loads` but loads from a file."""
        return self.loads(f.read(), salt)

    def loads_unsafe(
        self, s: str | bytes, salt: str | bytes | None = None
    ) -> tuple[bool, t.Any]:
        """Like :meth:`loads` but without verifying the signature. This
        is potentially very dangerous to use depending on how your
        serializer works. The return value is ``(signature_valid,
        payload)`` instead of just the payload. The first item will be a
        boolean that indicates if the signature is valid. This function
        never fails.

        Use it for debugging only and if you know that your serializer
        module is not exploitable (for example, do not use it with a
        pickle serializer).

        .. versionadded:: 0.15
        """
        return self._loads_unsafe_impl(s, salt)

    def _loads_unsafe_impl(
        self,
        s: str | bytes,
        salt: str | bytes | None,
        load_kwargs: dict[str, t.Any] | None = None,
        load_payload_kwargs: dict[str, t.Any] | None = None,
    ) -> tuple[bool, t.Any]:
        """Low level helper function to implement :meth:`loads_unsafe`
        in serializer subclasses.
        """
        if load_kwargs is None:
            load_kwargs = {}

        try:
            return True, self.loads(s, salt=salt, **load_kwargs)
        except BadSignature as e:
            if e.payload is None:
                return False, None

            if load_payload_kwargs is None:
                load_payload_kwargs = {}

            try:
                return (
                    False,
                    self.load_payload(e.payload, **load_payload_kwargs),
                )
            except BadPayload:
                return False, None

    def load_unsafe(
        self, f: t.IO[t.Any], salt: str | bytes | None = None
    ) -> tuple[bool, t.Any]:
        """Like :meth:`loads_unsafe` but loads from a file.

        .. versionadded:: 0.15
        """
        return self.loads_unsafe(f.read(), salt=salt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\quantization\execution_providers\qnn\quant_config.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnx

from ...calibrate import CalibrationDataReader, CalibrationMethod
from ...quant_utils import QuantType
from ...quantize import StaticQuantConfig
from ...tensor_quant_overrides import TensorQuantOverridesHelper
from .mixed_precision_overrides_utils import MixedPrecisionTensorQuantOverridesFixer

Q16_TYPES = {QuantType.QInt16, QuantType.QUInt16}
Q8_TYPES = {QuantType.QInt8, QuantType.QUInt8}
Q4_TYPES = {QuantType.QInt4, QuantType.QUInt4}
OP_TYPES_TO_EXCLUDE = {"Cast"}
MODEL_SIZE_THRESHOLD = 2147483648  # Quant model should use external data if >= 2GB


def warn_unable_to_override(
    node: onnx.NodeProto,
    what_str: str,
    tensor_name: str,
    io_kind: str,
):
    logging.warning(
        f"Unable to override {what_str} for {node.op_type} node's {io_kind} "
        "because it has already been overridden! Check the initial quantization overrides provided "
        "to get_qnn_qdq_config() if the generated QDQ model does not run on QNN EP. "
        f"Node name: {node.name}, {io_kind} name: {tensor_name}"
    )


def get_qnn_qdq_config(
    model_input: str | Path | onnx.ModelProto,
    calibration_data_reader: CalibrationDataReader,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    activation_type: QuantType = QuantType.QUInt8,
    weight_type: QuantType = QuantType.QUInt8,
    per_channel: bool = False,
    init_overrides: dict[str, list[dict[str, Any]]] | None = None,
    add_qtype_converts: bool = True,
    activation_symmetric: bool = False,
    weight_symmetric: bool | None = None,
    keep_removable_activations: bool = False,
    stride: int | None = None,
    calibration_providers: list[str] | None = None,
    op_types_to_quantize: list[str] | None = None,
    nodes_to_exclude: list[str] | None = None,
) -> StaticQuantConfig:
    """
    Returns a static quantization configuration suitable for running QDQ models on QNN EP.
    This is done primarily by setting tensor-level quantization overrides.

    Params:
        model_input: Path to the input model file or ModelProto.
        calibration_data_reader: Calibration data reader.
        calibrate_methode: The calibration method. Defaults to MinMax.
        activation_type: The default activation quantization type. Defaults to QUInt8.
        weight_type: The default weight quantization type. Defaults to QUInt8.
        per_channel: Global option that determines if a fixed set of operator types should be quantized per-channel.
            Defaults to false. Alternatively, use the tensor-level `init_overrides` to select individual operators
            and their quantization axes.

            If set, the quantization tool uses per-channel quantization for the following operator types and inputs:
                - Conv:
                    - input[1] on axis 0
                    - input[2] (bias) on axis 0
                - ConvTranspose:
                    - input[1] on axis 1
                    - input[2] (bias) on axis 0
        init_overrides: Initial tensor-level quantization overrides. Defaults to None. This function updates of a copy
            of these overrides with any necessary adjustments and includes them in the returned
            configuration object (i.e., config.extra_options['TensorQuantOverrides']).

            The key is a tensor name and the value is a list of dictionaries. For per-tensor quantization, the list
            contains a single dictionary. For per-channel quantization, the list contains either a dictionary for
            each channel in the tensor or a single dictionary that is assumed to apply to all channels. An 'axis'
            key must be present in the first dictionary for per-channel quantization.

            Each dictionary contains optional overrides with the following keys and values.
                'quant_type' = QuantType : The tensor's quantization data type.
                'axis' = Int             : The per-channel axis. Must be present for per-channel weights.
                'scale' =  Float         : The scale value to use. Must also specify `zero_point` if set.
                'zero_point' = Int       : The zero-point value to use. Must also specify `scale` is set.
                'symmetric' = Bool       : If the tensor should use symmetric quantization. Invalid if also
                                            set `scale` or `zero_point`.
                'reduce_range' = Bool    : If the quantization range should be reduced. Invalid if also
                                            set `scale` or `zero_point`. Only valid for initializers.
                'rmax' = Float           : Override the maximum real tensor value in calibration data.
                                            Invalid if also set `scale` or `zero_point`.
                'rmin' = Float           : Override the minimum real tensor value in calibration data.
                                            Invalid if also set `scale` or `zero_point`.
                'convert' = Dict         : A nested dictionary with the same keys for an activation
                                           tensor that should be converted to another quantization type.
                'convert["recv_nodes"] = Set : Set of node names that consume the converted activation,
                                               other nodes get the original type. If not specified,
                                               assume all consumer nodes get the converted type.
        add_qtype_converts: True if this function should automatically add "convert" entries to the provided
            `init_overrides` to ensure that operators use valid input/output types (activations only).
            Ex: if you override the output of an Add to 16-bit, this option ensures that the activation inputs
            of the Add are also up-converted to 16-bit and that data types for surrounding ops are converted
            appropriately. Refer to the documentation in mixed_precision_overrides_utils.py for additional details.
        activation_symmetric: True if activations should be quantized symmetrically (i.e, rmax == -rmin) by default.
            Defaults to false. For int8 and int16, this results in zero-point values of 0. For uint8 and uin16,
            the zero-point values are 128 and 32,768, respectively.
        weight_symmetric: True if weights should be quantized symmetrically (i.e., rmax == -rmin) by default.
            Defaults to None. If set to None, weight_symmetric is assumed true if the weight_type is a signed int.
        keep_removable_activations: Defaults to false. If true, "removable" activations (e.g., Clip or Relu) will not
                        be removed, and will be explicitly represented in the QDQ model. If false, these activations
                        are automatically removed if activations are asymmetrically quantized. Keeping these activations
                        is necessary if optimizations or EP transformations will later remove
                        QuantizeLinear/DequantizeLinear operators from the model.
        calibration_providers: Execution providers to run the session during calibration. Default is None which uses
            [ "CPUExecutionProvider" ].
        op_types_to_quantize: If set to None, all operator types will be quantized except for OP_TYPES_TO_EXCLUDE
        nodes_to_exclude: List of nodes names to exclude from quantization. The nodes in this list will be excluded from
            quantization when it is not None.

    Returns:
        A StaticQuantConfig object
    """
    if weight_symmetric is None:
        weight_symmetric = weight_type in {QuantType.QInt8, QuantType.QInt16}

    model = (
        model_input
        if isinstance(model_input, onnx.ModelProto)
        else onnx.load_model(model_input, load_external_data=False)
    )

    op_types = set()
    model_has_external_data = False
    name_to_initializer = {}

    # Build map of initializers (name -> initializer) and
    # check if the model has external data.
    for initializer in model.graph.initializer:
        name_to_initializer[initializer.name] = initializer
        if onnx.external_data_helper.uses_external_data(initializer):
            model_has_external_data = True

    overrides_helper = TensorQuantOverridesHelper(copy.deepcopy(init_overrides) if init_overrides else {})

    if not overrides_helper.empty() and add_qtype_converts:
        # Fix mixed-precision overrides.
        overrides_fixer = MixedPrecisionTensorQuantOverridesFixer.create_from_model(
            overrides_helper, model, activation_type
        )
        overrides_fixer.apply(activation_type, activation_symmetric)

    # Setup quantization overrides for specific operator types to ensure compatibility with QNN EP.
    qnn_compat = QnnCompatibilityOverrides(
        activation_type,
        weight_type,
        activation_symmetric,
        weight_symmetric,
        per_channel,
        overrides_helper,
        name_to_initializer,
    )

    op_types_to_quantize_set = set(op_types_to_quantize) if op_types_to_quantize else None
    nodes_to_exclude_set = set(nodes_to_exclude) if nodes_to_exclude else None

    for node in model.graph.node:
        if op_types_to_quantize_set and node.op_type not in op_types_to_quantize_set:
            continue
        if nodes_to_exclude_set and node.name in nodes_to_exclude_set:
            continue
        op_types.add(node.op_type)
        qnn_compat.process_node(node)

    extra_options = {
        "MinimumRealRange": 0.0001,
        "DedicatedQDQPair": False,  # Let ORT optimizer duplicate DQ nodes
        "QDQKeepRemovableActivations": keep_removable_activations,
        "TensorQuantOverrides": overrides_helper.get_dict(),
        "ActivationSymmetric": activation_symmetric,
        "WeightSymmetric": weight_symmetric,
        "CalibStridedMinMax": stride,
    }

    # ONNX opset < 21 does not support 16-bit quantization, so must use 'com.microsoft' domain
    # on Q/DQ operators if using 16-bit or 4-bit quantization.
    onnx_opset = next(x for x in model.opset_import if x.domain == "" or x.domain == "ai.onnx")
    if onnx_opset.version < 21:
        opset21_types = Q16_TYPES.union(Q4_TYPES)
        overrides_have_opset21_types = any(t in opset21_types for t in overrides_helper.get_quant_types())
        if activation_type in opset21_types or weight_type in opset21_types or overrides_have_opset21_types:
            extra_options["UseQDQContribOps"] = True

    return StaticQuantConfig(
        calibration_data_reader,
        calibrate_method=calibrate_method,
        activation_type=activation_type,
        weight_type=weight_type,
        op_types_to_quantize=(
            op_types_to_quantize if op_types_to_quantize else list(op_types.difference(OP_TYPES_TO_EXCLUDE))
        ),
        nodes_to_exclude=nodes_to_exclude,
        per_channel=per_channel,
        use_external_data_format=(model_has_external_data or model.ByteSize() >= MODEL_SIZE_THRESHOLD),
        calibration_providers=calibration_providers,
        extra_options=extra_options,
    )


class QnnCompatibilityOverrides:
    """
    Helper that processes nodes to generate quantization overrides that make the resulting QDQ model
    compatible with QNN EP.
    """

    def __init__(
        self,
        default_activation_qtype: QuantType,
        default_weight_qtype: QuantType,
        activation_symmetric: bool,
        weight_symmetric: bool,
        per_channel: bool,
        overrides: TensorQuantOverridesHelper,
        initializers: dict[str, onnx.TensorProto],
    ):
        self.default_activation_qtype = default_activation_qtype
        self.default_weight_qtype = default_weight_qtype
        self.activation_symmetric = activation_symmetric
        self.weight_symmetric = weight_symmetric
        self.per_channel = per_channel
        self.overrides = overrides
        self.initializers = initializers

        self.process_fns = {
            "MatMul": self._process_matmul,
            "LayerNormalization": self._process_layernorm,
            "Sigmoid": self._process_sigmoid,
            "Tanh": self._process_tanh,
        }

    def process_node(self, node: onnx.NodeProto):
        process_fn = self.process_fns.get(node.op_type)

        if process_fn is not None:
            process_fn(node)

    def _make_static_inputs_use_default_weight_type(self, node: onnx.NodeProto):
        """
        Overrides initializer input(s) to use the default weight type if:
        - The default weight type is 8-bit
        - One of the inputs is a 16-bit activation
        - The other input is an initializer (per-tensor quantized)

        This is necessary because the quantization tool does not assign MatMul or LayerNorm initializer
        inputs the default weight type. Instead, it assigns the default activation type.
        """
        if self.default_weight_qtype not in Q8_TYPES:
            return

        input_16bit_act_name = None
        input_weight_name = None

        # Loop through first 2 inputs to find a 16-bit activation and a (per-tensor) weight.
        for i in range(2):
            input_name = node.input[i]
            if not input_name:
                continue

            is_weight = input_name in self.initializers
            qtype_info = self.overrides.get_node_input_qtype_info(
                input_name,
                node.name,
                default_qtype=None if is_weight else self.default_activation_qtype,
            )

            if qtype_info.axis is not None:
                return  # Don't process MatMul with a per-channel quantized input.

            if (
                is_weight
                and qtype_info.quant_type == self.default_weight_qtype
                and qtype_info.symmetric == self.weight_symmetric
            ):
                return  # Return. Weight is already overridden to use the desired weight type.

            if is_weight:
                input_weight_name = input_name
            elif qtype_info.quant_type in Q16_TYPES:
                input_16bit_act_name = input_name

        # Override initializer input to use the default weight type.
        if input_16bit_act_name and input_weight_name:
            did_update = self.overrides.update_tensor_overrides(
                input_weight_name,
                {"quant_type": self.default_weight_qtype, "symmetric": self.weight_symmetric},
                overwrite=False,
            )

            if not did_update:
                warn_unable_to_override(node, "quant_type/symmetric", input_weight_name, "input weight")

    def _process_matmul(self, node: onnx.NodeProto):
        assert node.op_type == "MatMul", f"Expected MatMul, but got {node.op_type}"

        if not self.per_channel:
            self._make_static_inputs_use_default_weight_type(node)
            return

        # QNN does not support per-channel MatMul. However, the ORT quantization tool attempts to use per-channel
        # quantization for MatMul by default *if* the global per_channel setting is enabled. So, we need to
        # provide explicit per-tensor quantization overrides for MatMul if per_channel is enabled and
        # the user did not provide any other overrides.
        for input_name in node.input:
            is_weight_no_overrides = input_name in self.initializers and input_name not in self.overrides
            if is_weight_no_overrides:
                self.overrides.update_tensor_overrides(
                    input_name,
                    {"quant_type": self.default_weight_qtype, "symmetric": self.weight_symmetric},
                )

    def _process_layernorm(self, node: onnx.NodeProto):
        assert node.op_type == "LayerNormalization", f"Expected LayerNormalization, but got {node.op_type}"

        if not self.per_channel:
            self._make_static_inputs_use_default_weight_type(node)
            return

        has_weight_no_overrides = node.input[1] in self.initializers and node.input[1] not in self.overrides
        has_bias_no_overrides = (
            len(node.input) > 2
            and node.input[2]
            and node.input[2] in self.initializers
            and node.input[2] not in self.overrides
        )

        if has_weight_no_overrides or has_bias_no_overrides:
            # TODO: Make bias input not per-channel. QNN needs it to be per-tensor, but quantizer
            # tries to makes it per-channel if the weight is also per-channel.
            raise ValueError(
                "get_qnn_qdq_config() does not currently support the global per_channel option with LayerNormalization."
                " Please try using custom overrides that make bias per-tensor quantized."
            )

    def _process_sigmoid(self, node: onnx.NodeProto):
        """
        Overrides 16-bit Sigmoid's output scale and zero-point as per QNN requirements.
        """
        assert node.op_type == "Sigmoid", f"Expected Sigmoid, but got {node.op_type}"
        output_type = self.overrides.get_node_output_qtype_info(
            node.output[0], self.default_activation_qtype
        ).quant_type

        if output_type == QuantType.QUInt16:
            self.overrides.update_tensor_overrides(
                node.output[0],
                {
                    "quant_type": output_type,
                    "scale": np.array(1.0 / 65536.0, dtype=np.float32),
                    "zero_point": np.array(0, dtype=np.uint16),
                },
            )
        elif output_type == QuantType.QInt16:
            self.overrides.update_tensor_overrides(
                node.output[0],
                {
                    "quant_type": output_type,
                    "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                    "zero_point": np.array(0, dtype=np.int16),
                },
            )

    def _process_tanh(self, node: onnx.NodeProto):
        """
        Overrides 16-bit Tanh's output scale and zero-point as per QNN requirements.
        """
        assert node.op_type == "Tanh", f"Expected Tanh, but got {node.op_type}"
        output_type = self.overrides.get_node_output_qtype_info(
            node.output[0], self.default_activation_qtype
        ).quant_type

        if output_type == QuantType.QUInt16:
            self.overrides.update_tensor_overrides(
                node.output[0],
                {
                    "quant_type": output_type,
                    "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                    "zero_point": np.array(32768, dtype=np.uint16),
                },
            )
        elif output_type == QuantType.QInt16:
            self.overrides.update_tensor_overrides(
                node.output[0],
                {
                    "quant_type": output_type,
                    "scale": np.array(1.0 / 32768.0, dtype=np.float32),
                    "zero_point": np.array(0, dtype=np.int16),
                },
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\hstore.py ===
# dialects/postgresql/hstore.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


import re

from .array import ARRAY
from .operators import CONTAINED_BY
from .operators import CONTAINS
from .operators import GETITEM
from .operators import HAS_ALL
from .operators import HAS_ANY
from .operators import HAS_KEY
from ... import types as sqltypes
from ...sql import functions as sqlfunc


__all__ = ("HSTORE", "hstore")


class HSTORE(sqltypes.Indexable, sqltypes.Concatenable, sqltypes.TypeEngine):
    """Represent the PostgreSQL HSTORE type.

    The :class:`.HSTORE` type stores dictionaries containing strings, e.g.::

        data_table = Table(
            "data_table",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("data", HSTORE),
        )

        with engine.connect() as conn:
            conn.execute(
                data_table.insert(), data={"key1": "value1", "key2": "value2"}
            )

    :class:`.HSTORE` provides for a wide range of operations, including:

    * Index operations::

        data_table.c.data["some key"] == "some value"

    * Containment operations::

        data_table.c.data.has_key("some key")

        data_table.c.data.has_all(["one", "two", "three"])

    * Concatenation::

        data_table.c.data + {"k1": "v1"}

    For a full list of special methods see
    :class:`.HSTORE.comparator_factory`.

    .. container:: topic

        **Detecting Changes in HSTORE columns when using the ORM**

        For usage with the SQLAlchemy ORM, it may be desirable to combine the
        usage of :class:`.HSTORE` with :class:`.MutableDict` dictionary now
        part of the :mod:`sqlalchemy.ext.mutable` extension. This extension
        will allow "in-place" changes to the dictionary, e.g. addition of new
        keys or replacement/removal of existing keys to/from the current
        dictionary, to produce events which will be detected by the unit of
        work::

            from sqlalchemy.ext.mutable import MutableDict


            class MyClass(Base):
                __tablename__ = "data_table"

                id = Column(Integer, primary_key=True)
                data = Column(MutableDict.as_mutable(HSTORE))


            my_object = session.query(MyClass).one()

            # in-place mutation, requires Mutable extension
            # in order for the ORM to detect
            my_object.data["some_key"] = "some value"

            session.commit()

        When the :mod:`sqlalchemy.ext.mutable` extension is not used, the ORM
        will not be alerted to any changes to the contents of an existing
        dictionary, unless that dictionary value is re-assigned to the
        HSTORE-attribute itself, thus generating a change event.

    .. seealso::

        :class:`.hstore` - render the PostgreSQL ``hstore()`` function.


    """  # noqa: E501

    __visit_name__ = "HSTORE"
    hashable = False
    text_type = sqltypes.Text()

    def __init__(self, text_type=None):
        """Construct a new :class:`.HSTORE`.

        :param text_type: the type that should be used for indexed values.
         Defaults to :class:`_types.Text`.

        """
        if text_type is not None:
            self.text_type = text_type

    class Comparator(
        sqltypes.Indexable.Comparator, sqltypes.Concatenable.Comparator
    ):
        """Define comparison operations for :class:`.HSTORE`."""

        def has_key(self, other):
            """Boolean expression.  Test for presence of a key.  Note that the
            key may be a SQLA expression.
            """
            return self.operate(HAS_KEY, other, result_type=sqltypes.Boolean)

        def has_all(self, other):
            """Boolean expression.  Test for presence of all keys in jsonb"""
            return self.operate(HAS_ALL, other, result_type=sqltypes.Boolean)

        def has_any(self, other):
            """Boolean expression.  Test for presence of any key in jsonb"""
            return self.operate(HAS_ANY, other, result_type=sqltypes.Boolean)

        def contains(self, other, **kwargs):
            """Boolean expression.  Test if keys (or array) are a superset
            of/contained the keys of the argument jsonb expression.

            kwargs may be ignored by this operator but are required for API
            conformance.
            """
            return self.operate(CONTAINS, other, result_type=sqltypes.Boolean)

        def contained_by(self, other):
            """Boolean expression.  Test if keys are a proper subset of the
            keys of the argument jsonb expression.
            """
            return self.operate(
                CONTAINED_BY, other, result_type=sqltypes.Boolean
            )

        def _setup_getitem(self, index):
            return GETITEM, index, self.type.text_type

        def defined(self, key):
            """Boolean expression.  Test for presence of a non-NULL value for
            the key.  Note that the key may be a SQLA expression.
            """
            return _HStoreDefinedFunction(self.expr, key)

        def delete(self, key):
            """HStore expression.  Returns the contents of this hstore with the
            given key deleted.  Note that the key may be a SQLA expression.
            """
            if isinstance(key, dict):
                key = _serialize_hstore(key)
            return _HStoreDeleteFunction(self.expr, key)

        def slice(self, array):
            """HStore expression.  Returns a subset of an hstore defined by
            array of keys.
            """
            return _HStoreSliceFunction(self.expr, array)

        def keys(self):
            """Text array expression.  Returns array of keys."""
            return _HStoreKeysFunction(self.expr)

        def vals(self):
            """Text array expression.  Returns array of values."""
            return _HStoreValsFunction(self.expr)

        def array(self):
            """Text array expression.  Returns array of alternating keys and
            values.
            """
            return _HStoreArrayFunction(self.expr)

        def matrix(self):
            """Text array expression.  Returns array of [key, value] pairs."""
            return _HStoreMatrixFunction(self.expr)

    comparator_factory = Comparator

    def bind_processor(self, dialect):
        # note that dialect-specific types like that of psycopg and
        # psycopg2 will override this method to allow driver-level conversion
        # instead, see _PsycopgHStore
        def process(value):
            if isinstance(value, dict):
                return _serialize_hstore(value)
            else:
                return value

        return process

    def result_processor(self, dialect, coltype):
        # note that dialect-specific types like that of psycopg and
        # psycopg2 will override this method to allow driver-level conversion
        # instead, see _PsycopgHStore
        def process(value):
            if value is not None:
                return _parse_hstore(value)
            else:
                return value

        return process


class hstore(sqlfunc.GenericFunction):
    """Construct an hstore value within a SQL expression using the
    PostgreSQL ``hstore()`` function.

    The :class:`.hstore` function accepts one or two arguments as described
    in the PostgreSQL documentation.

    E.g.::

        from sqlalchemy.dialects.postgresql import array, hstore

        select(hstore("key1", "value1"))

        select(
            hstore(
                array(["key1", "key2", "key3"]),
                array(["value1", "value2", "value3"]),
            )
        )

    .. seealso::

        :class:`.HSTORE` - the PostgreSQL ``HSTORE`` datatype.

    """

    type = HSTORE
    name = "hstore"
    inherit_cache = True


class _HStoreDefinedFunction(sqlfunc.GenericFunction):
    type = sqltypes.Boolean
    name = "defined"
    inherit_cache = True


class _HStoreDeleteFunction(sqlfunc.GenericFunction):
    type = HSTORE
    name = "delete"
    inherit_cache = True


class _HStoreSliceFunction(sqlfunc.GenericFunction):
    type = HSTORE
    name = "slice"
    inherit_cache = True


class _HStoreKeysFunction(sqlfunc.GenericFunction):
    type = ARRAY(sqltypes.Text)
    name = "akeys"
    inherit_cache = True


class _HStoreValsFunction(sqlfunc.GenericFunction):
    type = ARRAY(sqltypes.Text)
    name = "avals"
    inherit_cache = True


class _HStoreArrayFunction(sqlfunc.GenericFunction):
    type = ARRAY(sqltypes.Text)
    name = "hstore_to_array"
    inherit_cache = True


class _HStoreMatrixFunction(sqlfunc.GenericFunction):
    type = ARRAY(sqltypes.Text)
    name = "hstore_to_matrix"
    inherit_cache = True


#
# parsing.  note that none of this is used with the psycopg2 backend,
# which provides its own native extensions.
#

# My best guess at the parsing rules of hstore literals, since no formal
# grammar is given.  This is mostly reverse engineered from PG's input parser
# behavior.
HSTORE_PAIR_RE = re.compile(
    r"""
(
  "(?P<key> (\\ . | [^"])* )"       # Quoted key
)
[ ]* => [ ]*    # Pair operator, optional adjoining whitespace
(
    (?P<value_null> NULL )          # NULL value
  | "(?P<value> (\\ . | [^"])* )"   # Quoted value
)
""",
    re.VERBOSE,
)

HSTORE_DELIMITER_RE = re.compile(
    r"""
[ ]* , [ ]*
""",
    re.VERBOSE,
)


def _parse_error(hstore_str, pos):
    """format an unmarshalling error."""

    ctx = 20
    hslen = len(hstore_str)

    parsed_tail = hstore_str[max(pos - ctx - 1, 0) : min(pos, hslen)]
    residual = hstore_str[min(pos, hslen) : min(pos + ctx + 1, hslen)]

    if len(parsed_tail) > ctx:
        parsed_tail = "[...]" + parsed_tail[1:]
    if len(residual) > ctx:
        residual = residual[:-1] + "[...]"

    return "After %r, could not parse residual at position %d: %r" % (
        parsed_tail,
        pos,
        residual,
    )


def _parse_hstore(hstore_str):
    """Parse an hstore from its literal string representation.

    Attempts to approximate PG's hstore input parsing rules as closely as
    possible. Although currently this is not strictly necessary, since the
    current implementation of hstore's output syntax is stricter than what it
    accepts as input, the documentation makes no guarantees that will always
    be the case.



    """
    result = {}
    pos = 0
    pair_match = HSTORE_PAIR_RE.match(hstore_str)

    while pair_match is not None:
        key = pair_match.group("key").replace(r"\"", '"').replace("\\\\", "\\")
        if pair_match.group("value_null"):
            value = None
        else:
            value = (
                pair_match.group("value")
                .replace(r"\"", '"')
                .replace("\\\\", "\\")
            )
        result[key] = value

        pos += pair_match.end()

        delim_match = HSTORE_DELIMITER_RE.match(hstore_str[pos:])
        if delim_match is not None:
            pos += delim_match.end()

        pair_match = HSTORE_PAIR_RE.match(hstore_str[pos:])

    if pos != len(hstore_str):
        raise ValueError(_parse_error(hstore_str, pos))

    return result


def _serialize_hstore(val):
    """Serialize a dictionary into an hstore literal.  Keys and values must
    both be strings (except None for values).

    """

    def esc(s, position):
        if position == "value" and s is None:
            return "NULL"
        elif isinstance(s, str):
            return '"%s"' % s.replace("\\", "\\\\").replace('"', r"\"")
        else:
            raise ValueError(
                "%r in %s position is not a string." % (s, position)
            )

    return ", ".join(
        "%s=>%s" % (esc(k, "key"), esc(v, "value")) for k, v in val.items()
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\calculus\singularities.py ===
"""
Singularities
=============

This module implements algorithms for finding singularities for a function
and identifying types of functions.

The differential calculus methods in this module include methods to identify
the following function types in the given ``Interval``:
- Increasing
- Strictly Increasing
- Decreasing
- Strictly Decreasing
- Monotonic

"""

from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.exponential import log
from sympy.functions.elementary.trigonometric import sec, csc, cot, tan, cos
from sympy.functions.elementary.hyperbolic import (
    sech, csch, coth, tanh, cosh, asech, acsch, atanh, acoth)
from sympy.utilities.misc import filldedent


def singularities(expression, symbol, domain=None):
    """
    Find singularities of a given function.

    Parameters
    ==========

    expression : Expr
        The target function in which singularities need to be found.
    symbol : Symbol
        The symbol over the values of which the singularity in
        expression in being searched for.

    Returns
    =======

    Set
        A set of values for ``symbol`` for which ``expression`` has a
        singularity. An ``EmptySet`` is returned if ``expression`` has no
        singularities for any given value of ``Symbol``.

    Raises
    ======

    NotImplementedError
        Methods for determining the singularities of this function have
        not been developed.

    Notes
    =====

    This function does not find non-isolated singularities
    nor does it find branch points of the expression.

    Currently supported functions are:
        - univariate continuous (real or complex) functions

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Mathematical_singularity

    Examples
    ========

    >>> from sympy import singularities, Symbol, log
    >>> x = Symbol('x', real=True)
    >>> y = Symbol('y', real=False)
    >>> singularities(x**2 + x + 1, x)
    EmptySet
    >>> singularities(1/(x + 1), x)
    {-1}
    >>> singularities(1/(y**2 + 1), y)
    {-I, I}
    >>> singularities(1/(y**3 + 1), y)
    {-1, 1/2 - sqrt(3)*I/2, 1/2 + sqrt(3)*I/2}
    >>> singularities(log(x), x)
    {0}

    """
    from sympy.solvers.solveset import solveset

    if domain is None:
        domain = S.Reals if symbol.is_real else S.Complexes
    try:
        sings = S.EmptySet
        e = expression.rewrite([sec, csc, cot, tan], cos)
        e = e.rewrite([sech, csch, coth, tanh], cosh)
        for i in e.atoms(Pow):
            if i.exp.is_infinite:
                raise NotImplementedError
            if i.exp.is_negative:
                # XXX: exponent of varying sign not handled
                sings += solveset(i.base, symbol, domain)
        for i in expression.atoms(log, asech, acsch):
            sings += solveset(i.args[0], symbol, domain)
        for i in expression.atoms(atanh, acoth):
            sings += solveset(i.args[0] - 1, symbol, domain)
            sings += solveset(i.args[0] + 1, symbol, domain)
        return sings
    except NotImplementedError:
        raise NotImplementedError(filldedent('''
            Methods for determining the singularities
            of this function have not been developed.'''))


###########################################################################
#                      DIFFERENTIAL CALCULUS METHODS                      #
###########################################################################


def monotonicity_helper(expression, predicate, interval=S.Reals, symbol=None):
    """
    Helper function for functions checking function monotonicity.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked
    predicate : function
        The property being tested for. The function takes in an integer
        and returns a boolean. The integer input is the derivative and
        the boolean result should be true if the property is being held,
        and false otherwise.
    interval : Set, optional
        The range of values in which we are testing, defaults to all reals.
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    It returns a boolean indicating whether the interval in which
    the function's derivative satisfies given predicate is a superset
    of the given interval.

    Returns
    =======

    Boolean
        True if ``predicate`` is true for all the derivatives when ``symbol``
        is varied in ``range``, False otherwise.

    """
    from sympy.solvers.solveset import solveset

    expression = sympify(expression)
    free = expression.free_symbols

    if symbol is None:
        if len(free) > 1:
            raise NotImplementedError(
                'The function has not yet been implemented'
                ' for all multivariate expressions.'
            )

    variable = symbol or (free.pop() if free else Symbol('x'))
    derivative = expression.diff(variable)
    predicate_interval = solveset(predicate(derivative), variable, S.Reals)
    return interval.is_subset(predicate_interval)


def is_increasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is increasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is increasing (either strictly increasing or
        constant) in the given ``interval``, False otherwise.

    Examples
    ========

    >>> from sympy import is_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_increasing(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_increasing(-x**2, Interval(-oo, 0))
    True
    >>> is_increasing(-x**2, Interval(0, oo))
    False
    >>> is_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval(-2, 3))
    False
    >>> is_increasing(x**2 + y, Interval(1, 2), x)
    True

    """
    return monotonicity_helper(expression, lambda x: x >= 0, interval, symbol)


def is_strictly_increasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is strictly increasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is strictly increasing in the given ``interval``,
        False otherwise.

    Examples
    ========

    >>> from sympy import is_strictly_increasing
    >>> from sympy.abc import x, y
    >>> from sympy import Interval, oo
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Ropen(-oo, -2))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.Lopen(3, oo))
    True
    >>> is_strictly_increasing(4*x**3 - 6*x**2 - 72*x + 30, Interval.open(-2, 3))
    False
    >>> is_strictly_increasing(-x**2, Interval(0, oo))
    False
    >>> is_strictly_increasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x > 0, interval, symbol)


def is_decreasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is decreasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is decreasing (either strictly decreasing or
        constant) in the given ``interval``, False otherwise.

    Examples
    ========

    >>> from sympy import is_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))
    False
    >>> is_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x <= 0, interval, symbol)


def is_strictly_decreasing(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is strictly decreasing in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is strictly decreasing in the given ``interval``,
        False otherwise.

    Examples
    ========

    >>> from sympy import is_strictly_decreasing
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, S(3)/2))
    False
    >>> is_strictly_decreasing(1/(x**2 - 3*x), Interval.Ropen(-oo, 1.5))
    False
    >>> is_strictly_decreasing(-x**2, Interval(-oo, 0))
    False
    >>> is_strictly_decreasing(-x**2 + y, Interval(-oo, 0), x)
    False

    """
    return monotonicity_helper(expression, lambda x: x < 0, interval, symbol)


def is_monotonic(expression, interval=S.Reals, symbol=None):
    """
    Return whether the function is monotonic in the given interval.

    Parameters
    ==========

    expression : Expr
        The target function which is being checked.
    interval : Set, optional
        The range of values in which we are testing (defaults to set of
        all real numbers).
    symbol : Symbol, optional
        The symbol present in expression which gets varied over the given range.

    Returns
    =======

    Boolean
        True if ``expression`` is monotonic in the given ``interval``,
        False otherwise.

    Raises
    ======

    NotImplementedError
        Monotonicity check has not been implemented for the queried function.

    Examples
    ========

    >>> from sympy import is_monotonic
    >>> from sympy.abc import x, y
    >>> from sympy import S, Interval, oo
    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(S(3)/2, 3))
    True
    >>> is_monotonic(1/(x**2 - 3*x), Interval.open(1.5, 3))
    True
    >>> is_monotonic(1/(x**2 - 3*x), Interval.Lopen(3, oo))
    True
    >>> is_monotonic(x**3 - 3*x**2 + 4*x, S.Reals)
    True
    >>> is_monotonic(-x**2, S.Reals)
    False
    >>> is_monotonic(x**2 + y + 1, Interval(1, 2), x)
    True

    """
    from sympy.solvers.solveset import solveset

    expression = sympify(expression)

    free = expression.free_symbols
    if symbol is None and len(free) > 1:
        raise NotImplementedError(
            'is_monotonic has not yet been implemented'
            ' for all multivariate expressions.'
        )

    variable = symbol or (free.pop() if free else Symbol('x'))
    turning_points = solveset(expression.diff(variable), variable, interval)
    return interval.intersection(turning_points) is S.EmptySet

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\ie\options.py ===
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
from enum import Enum
from typing import Any, Dict

from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.webdriver.common.options import ArgOptions


class ElementScrollBehavior(Enum):
    TOP = 0
    BOTTOM = 1


class _IeOptionsDescriptor:
    """_IeOptionsDescriptor is an implementation of Descriptor Protocol:

    : Any look-up or assignment to the below attributes in `Options` class will be intercepted
    by `__get__` and `__set__` method respectively.

    - `browser_attach_timeout`
    - `element_scroll_behavior`
    - `ensure_clean_session`
    - `file_upload_dialog_timeout`
    - `force_create_process_api`
    - `force_shell_windows_api`
    - `full_page_screenshot`
    - `ignore_protected_mode_settings`
    - `ignore_zoom_level`
    - `initial_browser_url`
    - `native_events`
    - `persistent_hover`
    - `require_window_focus`
    - `use_per_process_proxy`
    - `use_legacy_file_upload_dialog_handling`
    - `attach_to_edge_chrome`
    - `edge_executable_path`


    : When an attribute lookup happens,
    Example:
        `self. browser_attach_timeout`
        `__get__` method does a dictionary look up in the dictionary `_options` in `Options` class
        and returns the value of key `browserAttachTimeout`
    : When an attribute assignment happens,
    Example:
        `self.browser_attach_timeout` = 30
        `__set__` method sets/updates the value of the key `browserAttachTimeout` in `_options`
        dictionary in `Options` class.
    """

    def __init__(self, name, expected_type):
        self.name = name
        self.expected_type = expected_type

    def __get__(self, obj, cls):
        return obj._options.get(self.name)

    def __set__(self, obj, value) -> None:
        if not isinstance(value, self.expected_type):
            raise ValueError(f"{self.name} should be of type {self.expected_type.__name__}")

        if self.name == "elementScrollBehavior" and value not in [
            ElementScrollBehavior.TOP,
            ElementScrollBehavior.BOTTOM,
        ]:
            raise ValueError("Element Scroll Behavior out of range.")
        obj._options[self.name] = value


class Options(ArgOptions):
    KEY = "se:ieOptions"
    SWITCHES = "ie.browserCommandLineSwitches"

    BROWSER_ATTACH_TIMEOUT = "browserAttachTimeout"
    ELEMENT_SCROLL_BEHAVIOR = "elementScrollBehavior"
    ENSURE_CLEAN_SESSION = "ie.ensureCleanSession"
    FILE_UPLOAD_DIALOG_TIMEOUT = "ie.fileUploadDialogTimeout"
    FORCE_CREATE_PROCESS_API = "ie.forceCreateProcessApi"
    FORCE_SHELL_WINDOWS_API = "ie.forceShellWindowsApi"
    FULL_PAGE_SCREENSHOT = "ie.enableFullPageScreenshot"
    IGNORE_PROTECTED_MODE_SETTINGS = "ignoreProtectedModeSettings"
    IGNORE_ZOOM_LEVEL = "ignoreZoomSetting"
    INITIAL_BROWSER_URL = "initialBrowserUrl"
    NATIVE_EVENTS = "nativeEvents"
    PERSISTENT_HOVER = "enablePersistentHover"
    REQUIRE_WINDOW_FOCUS = "requireWindowFocus"
    USE_PER_PROCESS_PROXY = "ie.usePerProcessProxy"
    USE_LEGACY_FILE_UPLOAD_DIALOG_HANDLING = "ie.useLegacyFileUploadDialogHandling"
    ATTACH_TO_EDGE_CHROME = "ie.edgechromium"
    EDGE_EXECUTABLE_PATH = "ie.edgepath"
    IGNORE_PROCESS_MATCH = "ie.ignoreprocessmatch"

    # Creating descriptor objects for each of the above IE options
    browser_attach_timeout = _IeOptionsDescriptor(BROWSER_ATTACH_TIMEOUT, int)
    """Gets and Sets `browser_attach_timeout`

    Usage:
    ------
    - Get
        - `self.browser_attach_timeout`
    - Set
        - `self.browser_attach_timeout` = `value`

    Parameters:
    -----------
    `value`: `int` (Timeout) in milliseconds
    """

    element_scroll_behavior = _IeOptionsDescriptor(ELEMENT_SCROLL_BEHAVIOR, Enum)
    """Gets and Sets `element_scroll_behavior`

    Usage:
    ------
    - Get
        - `self.element_scroll_behavior`
    - Set
        - `self.element_scroll_behavior` = `value`

    Parameters:
    -----------
    `value`: `int` either 0 - Top, 1 - Bottom
    """

    ensure_clean_session = _IeOptionsDescriptor(ENSURE_CLEAN_SESSION, bool)
    """Gets and Sets `ensure_clean_session`

    Usage:
    ------
    - Get
        - `self.ensure_clean_session`
    - Set
        - `self.ensure_clean_session` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    file_upload_dialog_timeout = _IeOptionsDescriptor(FILE_UPLOAD_DIALOG_TIMEOUT, int)
    """Gets and Sets `file_upload_dialog_timeout`

    Usage:
    ------
    - Get
        - `self.file_upload_dialog_timeout`
    - Set
        - `self.file_upload_dialog_timeout` = `value`

    Parameters:
    -----------
    `value`: `int` (Timeout) in milliseconds
    """

    force_create_process_api = _IeOptionsDescriptor(FORCE_CREATE_PROCESS_API, bool)
    """Gets and Sets `force_create_process_api`

    Usage:
    ------
    - Get
        - `self.force_create_process_api`
    - Set
        - `self.force_create_process_api` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    force_shell_windows_api = _IeOptionsDescriptor(FORCE_SHELL_WINDOWS_API, bool)
    """Gets and Sets `force_shell_windows_api`

    Usage:
    ------
    - Get
        - `self.force_shell_windows_api`
    - Set
        - `self.force_shell_windows_api` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    full_page_screenshot = _IeOptionsDescriptor(FULL_PAGE_SCREENSHOT, bool)
    """Gets and Sets `full_page_screenshot`

    Usage:
    ------
    - Get
        - `self.full_page_screenshot`
    - Set
        - `self.full_page_screenshot` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    ignore_protected_mode_settings = _IeOptionsDescriptor(IGNORE_PROTECTED_MODE_SETTINGS, bool)
    """Gets and Sets `ignore_protected_mode_settings`

    Usage:
    ------
    - Get
        - `self.ignore_protected_mode_settings`
    - Set
        - `self.ignore_protected_mode_settings` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    ignore_zoom_level = _IeOptionsDescriptor(IGNORE_ZOOM_LEVEL, bool)
    """Gets and Sets `ignore_zoom_level`

    Usage:
    ------
    - Get
        - `self.ignore_zoom_level`
    - Set
        - `self.ignore_zoom_level` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    initial_browser_url = _IeOptionsDescriptor(INITIAL_BROWSER_URL, str)
    """Gets and Sets `initial_browser_url`

    Usage:
    ------
    - Get
        - `self.initial_browser_url`
    - Set
        - `self.initial_browser_url` = `value`

    Parameters:
    -----------
    `value`: `str`
    """

    native_events = _IeOptionsDescriptor(NATIVE_EVENTS, bool)
    """Gets and Sets `native_events`

    Usage:
    ------
    - Get
        - `self.native_events`
    - Set
        - `self.native_events` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    persistent_hover = _IeOptionsDescriptor(PERSISTENT_HOVER, bool)
    """Gets and Sets `persistent_hover`

    Usage:
    ------
    - Get
        - `self.persistent_hover`
    - Set
        - `self.persistent_hover` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    require_window_focus = _IeOptionsDescriptor(REQUIRE_WINDOW_FOCUS, bool)
    """Gets and Sets `require_window_focus`

    Usage:
    ------
    - Get
        - `self.require_window_focus`
    - Set
        - `self.require_window_focus` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    use_per_process_proxy = _IeOptionsDescriptor(USE_PER_PROCESS_PROXY, bool)
    """Gets and Sets `use_per_process_proxy`

    Usage:
    ------
    - Get
        - `self.use_per_process_proxy`
    - Set
        - `self.use_per_process_proxy` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    use_legacy_file_upload_dialog_handling = _IeOptionsDescriptor(USE_LEGACY_FILE_UPLOAD_DIALOG_HANDLING, bool)
    """Gets and Sets `use_legacy_file_upload_dialog_handling`

    Usage:
    ------
    - Get
        - `self.use_legacy_file_upload_dialog_handling`
    - Set
        - `self.use_legacy_file_upload_dialog_handling` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    attach_to_edge_chrome = _IeOptionsDescriptor(ATTACH_TO_EDGE_CHROME, bool)
    """Gets and Sets `attach_to_edge_chrome`

    Usage:
    ------
    - Get
        - `self.attach_to_edge_chrome`
    - Set
        - `self.attach_to_edge_chrome` = `value`

    Parameters:
    -----------
    `value`: `bool`
    """

    edge_executable_path = _IeOptionsDescriptor(EDGE_EXECUTABLE_PATH, str)
    """Gets and Sets `edge_executable_path`

    Usage:
    ------
    - Get
        - `self.edge_executable_path`
    - Set
        - `self.edge_executable_path` = `value`

    Parameters:
    -----------
    `value`: `str`
    """

    def __init__(self) -> None:
        super().__init__()
        self._options: Dict[str, Any] = {}
        self._additional: Dict[str, Any] = {}

    @property
    def options(self) -> dict:
        """:Returns: A dictionary of browser options."""
        return self._options

    @property
    def additional_options(self) -> dict:
        """:Returns: The additional options."""
        return self._additional

    def add_additional_option(self, name: str, value) -> None:
        """Adds an additional option not yet added as a safe option for IE.

        :Args:
         - name: name of the option to add
         - value: value of the option to add
        """
        self._additional[name] = value

    def to_capabilities(self) -> dict:
        """Marshals the IE options to the correct object."""
        caps = self._caps

        opts = self._options.copy()
        if self._arguments:
            opts[self.SWITCHES] = " ".join(self._arguments)

        if self._additional:
            opts.update(self._additional)

        if opts:
            caps[Options.KEY] = opts
        return caps

    @property
    def default_capabilities(self) -> dict:
        return DesiredCapabilities.INTERNETEXPLORER.copy()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\refine.py ===
from __future__ import annotations
from typing import Callable

from sympy.core import S, Add, Expr, Basic, Mul, Pow, Rational
from sympy.core.logic import fuzzy_not
from sympy.logic.boolalg import Boolean

from sympy.assumptions import ask, Q  # type: ignore


def refine(expr, assumptions=True):
    """
    Simplify an expression using assumptions.

    Explanation
    ===========

    Unlike :func:`~.simplify` which performs structural simplification
    without any assumption, this function transforms the expression into
    the form which is only valid under certain assumptions. Note that
    ``simplify()`` is generally not done in refining process.

    Refining boolean expression involves reducing it to ``S.true`` or
    ``S.false``. Unlike :func:`~.ask`, the expression will not be reduced
    if the truth value cannot be determined.

    Examples
    ========

    >>> from sympy import refine, sqrt, Q
    >>> from sympy.abc import x
    >>> refine(sqrt(x**2), Q.real(x))
    Abs(x)
    >>> refine(sqrt(x**2), Q.positive(x))
    x

    >>> refine(Q.real(x), Q.positive(x))
    True
    >>> refine(Q.positive(x), Q.real(x))
    Q.positive(x)

    See Also
    ========

    sympy.simplify.simplify.simplify : Structural simplification without assumptions.
    sympy.assumptions.ask.ask : Query for boolean expressions using assumptions.
    """
    if not isinstance(expr, Basic):
        return expr

    if not expr.is_Atom:
        args = [refine(arg, assumptions) for arg in expr.args]
        # TODO: this will probably not work with Integral or Polynomial
        expr = expr.func(*args)
    if hasattr(expr, '_eval_refine'):
        ref_expr = expr._eval_refine(assumptions)
        if ref_expr is not None:
            return ref_expr
    name = expr.__class__.__name__
    handler = handlers_dict.get(name, None)
    if handler is None:
        return expr
    new_expr = handler(expr, assumptions)
    if (new_expr is None) or (expr == new_expr):
        return expr
    if not isinstance(new_expr, Expr):
        return new_expr
    return refine(new_expr, assumptions)


def refine_abs(expr, assumptions):
    """
    Handler for the absolute value.

    Examples
    ========

    >>> from sympy import Q, Abs
    >>> from sympy.assumptions.refine import refine_abs
    >>> from sympy.abc import x
    >>> refine_abs(Abs(x), Q.real(x))
    >>> refine_abs(Abs(x), Q.positive(x))
    x
    >>> refine_abs(Abs(x), Q.negative(x))
    -x

    """
    from sympy.functions.elementary.complexes import Abs
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions) and \
            fuzzy_not(ask(Q.negative(arg), assumptions)):
        # if it's nonnegative
        return arg
    if ask(Q.negative(arg), assumptions):
        return -arg
    # arg is Mul
    if isinstance(arg, Mul):
        r = [refine(abs(a), assumptions) for a in arg.args]
        non_abs = []
        in_abs = []
        for i in r:
            if isinstance(i, Abs):
                in_abs.append(i.args[0])
            else:
                non_abs.append(i)
        return Mul(*non_abs) * Abs(Mul(*in_abs))


def refine_Pow(expr, assumptions):
    """
    Handler for instances of Pow.

    Examples
    ========

    >>> from sympy import Q
    >>> from sympy.assumptions.refine import refine_Pow
    >>> from sympy.abc import x,y,z
    >>> refine_Pow((-1)**x, Q.real(x))
    >>> refine_Pow((-1)**x, Q.even(x))
    1
    >>> refine_Pow((-1)**x, Q.odd(x))
    -1

    For powers of -1, even parts of the exponent can be simplified:

    >>> refine_Pow((-1)**(x+y), Q.even(x))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+z), Q.odd(x) & Q.odd(z))
    (-1)**y
    >>> refine_Pow((-1)**(x+y+2), Q.odd(x))
    (-1)**(y + 1)
    >>> refine_Pow((-1)**(x+3), True)
    (-1)**(x + 1)

    """
    from sympy.functions.elementary.complexes import Abs
    from sympy.functions import sign
    if isinstance(expr.base, Abs):
        if ask(Q.real(expr.base.args[0]), assumptions) and \
                ask(Q.even(expr.exp), assumptions):
            return expr.base.args[0] ** expr.exp
    if ask(Q.real(expr.base), assumptions):
        if expr.base.is_number:
            if ask(Q.even(expr.exp), assumptions):
                return abs(expr.base) ** expr.exp
            if ask(Q.odd(expr.exp), assumptions):
                return sign(expr.base) * abs(expr.base) ** expr.exp
        if isinstance(expr.exp, Rational):
            if isinstance(expr.base, Pow):
                return abs(expr.base.base) ** (expr.base.exp * expr.exp)

        if expr.base is S.NegativeOne:
            if expr.exp.is_Add:

                old = expr

                # For powers of (-1) we can remove
                #  - even terms
                #  - pairs of odd terms
                #  - a single odd term + 1
                #  - A numerical constant N can be replaced with mod(N,2)

                coeff, terms = expr.exp.as_coeff_add()
                terms = set(terms)
                even_terms = set()
                odd_terms = set()
                initial_number_of_terms = len(terms)

                for t in terms:
                    if ask(Q.even(t), assumptions):
                        even_terms.add(t)
                    elif ask(Q.odd(t), assumptions):
                        odd_terms.add(t)

                terms -= even_terms
                if len(odd_terms) % 2:
                    terms -= odd_terms
                    new_coeff = (coeff + S.One) % 2
                else:
                    terms -= odd_terms
                    new_coeff = coeff % 2

                if new_coeff != coeff or len(terms) < initial_number_of_terms:
                    terms.add(new_coeff)
                    expr = expr.base**(Add(*terms))

                # Handle (-1)**((-1)**n/2 + m/2)
                e2 = 2*expr.exp
                if ask(Q.even(e2), assumptions):
                    if e2.could_extract_minus_sign():
                        e2 *= expr.base
                if e2.is_Add:
                    i, p = e2.as_two_terms()
                    if p.is_Pow and p.base is S.NegativeOne:
                        if ask(Q.integer(p.exp), assumptions):
                            i = (i + 1)/2
                            if ask(Q.even(i), assumptions):
                                return expr.base**p.exp
                            elif ask(Q.odd(i), assumptions):
                                return expr.base**(p.exp + 1)
                            else:
                                return expr.base**(p.exp + i)

                if old != expr:
                    return expr


def refine_atan2(expr, assumptions):
    """
    Handler for the atan2 function.

    Examples
    ========

    >>> from sympy import Q, atan2
    >>> from sympy.assumptions.refine import refine_atan2
    >>> from sympy.abc import x, y
    >>> refine_atan2(atan2(y,x), Q.real(y) & Q.positive(x))
    atan(y/x)
    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.negative(x))
    atan(y/x) - pi
    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.negative(x))
    atan(y/x) + pi
    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.negative(x))
    pi
    >>> refine_atan2(atan2(y,x), Q.positive(y) & Q.zero(x))
    pi/2
    >>> refine_atan2(atan2(y,x), Q.negative(y) & Q.zero(x))
    -pi/2
    >>> refine_atan2(atan2(y,x), Q.zero(y) & Q.zero(x))
    nan
    """
    from sympy.functions.elementary.trigonometric import atan
    y, x = expr.args
    if ask(Q.real(y) & Q.positive(x), assumptions):
        return atan(y / x)
    elif ask(Q.negative(y) & Q.negative(x), assumptions):
        return atan(y / x) - S.Pi
    elif ask(Q.positive(y) & Q.negative(x), assumptions):
        return atan(y / x) + S.Pi
    elif ask(Q.zero(y) & Q.negative(x), assumptions):
        return S.Pi
    elif ask(Q.positive(y) & Q.zero(x), assumptions):
        return S.Pi/2
    elif ask(Q.negative(y) & Q.zero(x), assumptions):
        return -S.Pi/2
    elif ask(Q.zero(y) & Q.zero(x), assumptions):
        return S.NaN
    else:
        return expr


def refine_re(expr, assumptions):
    """
    Handler for real part.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_re
    >>> from sympy import Q, re
    >>> from sympy.abc import x
    >>> refine_re(re(x), Q.real(x))
    x
    >>> refine_re(re(x), Q.imaginary(x))
    0
    """
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return arg
    if ask(Q.imaginary(arg), assumptions):
        return S.Zero
    return _refine_reim(expr, assumptions)


def refine_im(expr, assumptions):
    """
    Handler for imaginary part.

    Explanation
    ===========

    >>> from sympy.assumptions.refine import refine_im
    >>> from sympy import Q, im
    >>> from sympy.abc import x
    >>> refine_im(im(x), Q.real(x))
    0
    >>> refine_im(im(x), Q.imaginary(x))
    -I*x
    """
    arg = expr.args[0]
    if ask(Q.real(arg), assumptions):
        return S.Zero
    if ask(Q.imaginary(arg), assumptions):
        return - S.ImaginaryUnit * arg
    return _refine_reim(expr, assumptions)

def refine_arg(expr, assumptions):
    """
    Handler for complex argument

    Explanation
    ===========

    >>> from sympy.assumptions.refine import refine_arg
    >>> from sympy import Q, arg
    >>> from sympy.abc import x
    >>> refine_arg(arg(x), Q.positive(x))
    0
    >>> refine_arg(arg(x), Q.negative(x))
    pi
    """
    rg = expr.args[0]
    if ask(Q.positive(rg), assumptions):
        return S.Zero
    if ask(Q.negative(rg), assumptions):
        return S.Pi
    return None


def _refine_reim(expr, assumptions):
    # Helper function for refine_re & refine_im
    expanded = expr.expand(complex = True)
    if expanded != expr:
        refined = refine(expanded, assumptions)
        if refined != expanded:
            return refined
    # Best to leave the expression as is
    return None


def refine_sign(expr, assumptions):
    """
    Handler for sign.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_sign
    >>> from sympy import Symbol, Q, sign, im
    >>> x = Symbol('x', real = True)
    >>> expr = sign(x)
    >>> refine_sign(expr, Q.positive(x) & Q.nonzero(x))
    1
    >>> refine_sign(expr, Q.negative(x) & Q.nonzero(x))
    -1
    >>> refine_sign(expr, Q.zero(x))
    0
    >>> y = Symbol('y', imaginary = True)
    >>> expr = sign(y)
    >>> refine_sign(expr, Q.positive(im(y)))
    I
    >>> refine_sign(expr, Q.negative(im(y)))
    -I
    """
    arg = expr.args[0]
    if ask(Q.zero(arg), assumptions):
        return S.Zero
    if ask(Q.real(arg)):
        if ask(Q.positive(arg), assumptions):
            return S.One
        if ask(Q.negative(arg), assumptions):
            return S.NegativeOne
    if ask(Q.imaginary(arg)):
        arg_re, arg_im = arg.as_real_imag()
        if ask(Q.positive(arg_im), assumptions):
            return S.ImaginaryUnit
        if ask(Q.negative(arg_im), assumptions):
            return -S.ImaginaryUnit
    return expr


def refine_matrixelement(expr, assumptions):
    """
    Handler for symmetric part.

    Examples
    ========

    >>> from sympy.assumptions.refine import refine_matrixelement
    >>> from sympy import MatrixSymbol, Q
    >>> X = MatrixSymbol('X', 3, 3)
    >>> refine_matrixelement(X[0, 1], Q.symmetric(X))
    X[0, 1]
    >>> refine_matrixelement(X[1, 0], Q.symmetric(X))
    X[0, 1]
    """
    from sympy.matrices.expressions.matexpr import MatrixElement
    matrix, i, j = expr.args
    if ask(Q.symmetric(matrix), assumptions):
        if (i - j).could_extract_minus_sign():
            return expr
        return MatrixElement(matrix, j, i)

handlers_dict: dict[str, Callable[[Expr, Boolean], Expr]] = {
    'Abs': refine_abs,
    'Pow': refine_Pow,
    'atan2': refine_atan2,
    're': refine_re,
    'im': refine_im,
    'arg': refine_arg,
    'sign': refine_sign,
    'MatrixElement': refine_matrixelement
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_highlevel_open_tcp_stream.py ===
from __future__ import annotations

import sys
from contextlib import contextmanager, suppress
from typing import TYPE_CHECKING, Any

import trio
from trio.socket import SOCK_STREAM, SocketType, getaddrinfo, socket

if TYPE_CHECKING:
    from collections.abc import Generator, MutableSequence
    from socket import AddressFamily, SocketKind

    from trio._socket import AddressFormat

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup, ExceptionGroup


# Implementation of RFC 6555 "Happy eyeballs"
# https://tools.ietf.org/html/rfc6555
#
# Basically, the problem here is that if we want to connect to some host, and
# DNS returns multiple IP addresses, then we don't know which of them will
# actually work -- it can happen that some of them are reachable, and some of
# them are not. One particularly common situation where this happens is on a
# host that thinks it has ipv6 connectivity, but really doesn't. But in
# principle this could happen for any kind of multi-home situation (e.g. the
# route to one mirror is down but another is up).
#
# The naive algorithm (e.g. the stdlib's socket.create_connection) would be to
# pick one of the IP addresses and try to connect; if that fails, try the
# next; etc. The problem with this is that TCP is stubborn, and if the first
# address is a blackhole then it might take a very long time (tens of seconds)
# before that connection attempt fails.
#
# That's where RFC 6555 comes in. It tells us that what we do is:
# - get the list of IPs from getaddrinfo, trusting the order it gives us (with
#   one exception noted in section 5.4)
# - start a connection attempt to the first IP
# - when this fails OR if it's still going after DELAY seconds, then start a
#   connection attempt to the second IP
# - when this fails OR if it's still going after another DELAY seconds, then
#   start a connection attempt to the third IP
# - ... repeat until we run out of IPs.
#
# Our implementation is similarly straightforward: we spawn a chain of tasks,
# where each one (a) waits until the previous connection has failed or DELAY
# seconds have passed, (b) spawns the next task, (c) attempts to connect. As
# soon as any task crashes or succeeds, we cancel all the tasks and return.
#
# Note: this currently doesn't attempt to cache any results, so if you make
# multiple connections to the same host it'll re-run the happy-eyeballs
# algorithm each time. RFC 6555 is pretty confusing about whether this is
# allowed. Section 4 describes an algorithm that attempts ipv4 and ipv6
# simultaneously, and then says "The client MUST cache information regarding
# the outcome of each connection attempt, and it uses that information to
# avoid thrashing the network with subsequent attempts." Then section 4.2 says
# "implementations MUST prefer the first IP address family returned by the
# host's address preference policy, unless implementing a stateful
# algorithm". Here "stateful" means "one that caches information about
# previous attempts". So my reading of this is that IF you're starting ipv4
# and ipv6 at the same time then you MUST cache the result for ~ten minutes,
# but IF you're "preferring" one protocol by trying it first (like we are),
# then you don't need to cache.
#
# Caching is quite tricky: to get it right you need to do things like detect
# when the network interfaces are reconfigured, and if you get it wrong then
# connection attempts basically just don't work. So we don't even try.

# "Firefox and Chrome use 300 ms"
# https://tools.ietf.org/html/rfc6555#section-6
# Though
#   https://www.researchgate.net/profile/Vaibhav_Bajpai3/publication/304568993_Measuring_the_Effects_of_Happy_Eyeballs/links/5773848e08ae6f328f6c284c/Measuring-the-Effects-of-Happy-Eyeballs.pdf
# claims that Firefox actually uses 0 ms, unless an about:config option is
# toggled and then it uses 250 ms.
DEFAULT_DELAY = 0.250

# How should we call getaddrinfo? In particular, should we use AI_ADDRCONFIG?
#
# The idea of AI_ADDRCONFIG is that it only returns addresses that might
# work. E.g., if getaddrinfo knows that you don't have any IPv6 connectivity,
# then it doesn't return any IPv6 addresses. And this is kinda nice, because
# it means maybe you can skip sending AAAA requests entirely. But in practice,
# it doesn't really work right.
#
# - on Linux/glibc, empirically, the default is to return all addresses, and
# with AI_ADDRCONFIG then it only returns IPv6 addresses if there is at least
# one non-loopback IPv6 address configured... but this can be a link-local
# address, so in practice I guess this is basically always configured if IPv6
# is enabled at all. OTOH if you pass in "::1" as the target address with
# AI_ADDRCONFIG and there's no *external* IPv6 address configured, you get an
# error. So AI_ADDRCONFIG mostly doesn't do anything, even when you would want
# it to, and when it does do something it might break things that would have
# worked.
#
# - on Windows 10, empirically, if no IPv6 address is configured then by
# default they are also suppressed from getaddrinfo (flags=0 and
# flags=AI_ADDRCONFIG seem to do the same thing). If you pass AI_ALL, then you
# get the full list.
# ...except for localhost! getaddrinfo("localhost", "80") gives me ::1, even
# though there's no ipv6 and other queries only return ipv4.
# If you pass in and IPv6 IP address as the target address, then that's always
# returned OK, even with AI_ADDRCONFIG set and no IPv6 configured.
#
# But I guess other versions of windows messed this up, judging from these bug
# reports:
# https://bugs.chromium.org/p/chromium/issues/detail?id=5234
# https://bugs.chromium.org/p/chromium/issues/detail?id=32522#c50
#
# So basically the options are either to use AI_ADDRCONFIG and then add some
# complicated special cases to work around its brokenness, or else don't use
# AI_ADDRCONFIG and accept that sometimes on legacy/misconfigured networks
# we'll waste 300 ms trying to connect to a blackholed destination.
#
# Twisted and Tornado always uses default flags. I think we'll do the same.


@contextmanager
def close_all() -> Generator[set[SocketType], None, None]:
    sockets_to_close: set[SocketType] = set()
    try:
        yield sockets_to_close
    finally:
        errs = []
        for sock in sockets_to_close:
            try:
                sock.close()
            except BaseException as exc:
                errs.append(exc)
        if len(errs) == 1:
            raise errs[0]
        elif errs:
            raise BaseExceptionGroup("", errs)


def reorder_for_rfc_6555_section_5_4(  # type: ignore[explicit-any]
    targets: MutableSequence[tuple[AddressFamily, SocketKind, int, str, Any]],
) -> None:
    # RFC 6555 section 5.4 says that if getaddrinfo returns multiple address
    # families (e.g. IPv4 and IPv6), then you should make sure that your first
    # and second attempts use different families:
    #
    #    https://tools.ietf.org/html/rfc6555#section-5.4
    #
    # This function post-processes the results from getaddrinfo, in-place, to
    # satisfy this requirement.
    for i in range(1, len(targets)):
        if targets[i][0] != targets[0][0]:
            # Found the first entry with a different address family; move it
            # so that it becomes the second item on the list.
            if i != 1:
                targets.insert(1, targets.pop(i))
            break


def format_host_port(host: str | bytes, port: int | str) -> str:
    host = host.decode("ascii") if isinstance(host, bytes) else host
    if ":" in host:
        return f"[{host}]:{port}"
    else:
        return f"{host}:{port}"


# Twisted's HostnameEndpoint has a good set of configurables:
#   https://twistedmatrix.com/documents/current/api/twisted.internet.endpoints.HostnameEndpoint.html
#
# - per-connection timeout
#   this doesn't seem useful -- we let you set a timeout on the whole thing
#   using Trio's normal mechanisms, and that seems like enough
# - delay between attempts
# - bind address (but not port!)
#   they *don't* support multiple address bindings, like giving the ipv4 and
#   ipv6 addresses of the host.
#   I think maybe our semantics should be: we accept a list of bind addresses,
#   and we bind to the first one that is compatible with the
#   connection attempt we want to make, and if none are compatible then we
#   don't try to connect to that target.
#
# XX TODO: implement bind address support
#
# Actually, the best option is probably to be explicit: {AF_INET: "...",
#   AF_INET6: "..."}
# this might be simpler after
async def open_tcp_stream(
    host: str | bytes,
    port: int,
    *,
    happy_eyeballs_delay: float | None = DEFAULT_DELAY,
    local_address: str | None = None,
) -> trio.SocketStream:
    """Connect to the given host and port over TCP.

    If the given ``host`` has multiple IP addresses associated with it, then
    we have a problem: which one do we use?

    One approach would be to attempt to connect to the first one, and then if
    that fails, attempt to connect to the second one ... until we've tried all
    of them. But the problem with this is that if the first IP address is
    unreachable (for example, because it's an IPv6 address and our network
    discards IPv6 packets), then we might end up waiting tens of seconds for
    the first connection attempt to timeout before we try the second address.

    Another approach would be to attempt to connect to all of the addresses at
    the same time, in parallel, and then use whichever connection succeeds
    first, abandoning the others. This would be fast, but create a lot of
    unnecessary load on the network and the remote server.

    This function strikes a balance between these two extremes: it works its
    way through the available addresses one at a time, like the first
    approach; but, if ``happy_eyeballs_delay`` seconds have passed and it's
    still waiting for an attempt to succeed or fail, then it gets impatient
    and starts the next connection attempt in parallel. As soon as any one
    connection attempt succeeds, all the other attempts are cancelled. This
    avoids unnecessary load because most connections will succeed after just
    one or two attempts, but if one of the addresses is unreachable then it
    doesn't slow us down too much.

    This is known as a "happy eyeballs" algorithm, and our particular variant
    is modelled after how Chrome connects to webservers; see `RFC 6555
    <https://tools.ietf.org/html/rfc6555>`__ for more details.

    Args:
      host (str or bytes): The host to connect to. Can be an IPv4 address,
          IPv6 address, or a hostname.

      port (int): The port to connect to.

      happy_eyeballs_delay (float or None): How many seconds to wait for each
          connection attempt to succeed or fail before getting impatient and
          starting another one in parallel. Set to `None` if you want
          to limit to only one connection attempt at a time (like
          :func:`socket.create_connection`). Default: 0.25 (250 ms).

      local_address (None or str): The local IP address or hostname to use as
          the source for outgoing connections. If ``None``, we let the OS pick
          the source IP.

          This is useful in some exotic networking configurations where your
          host has multiple IP addresses, and you want to force the use of a
          specific one.

          Note that if you pass an IPv4 ``local_address``, then you won't be
          able to connect to IPv6 hosts, and vice-versa. If you want to take
          advantage of this to force the use of IPv4 or IPv6 without
          specifying an exact source address, you can use the IPv4 wildcard
          address ``local_address="0.0.0.0"``, or the IPv6 wildcard address
          ``local_address="::"``.

    Returns:
      SocketStream: a :class:`~trio.abc.Stream` connected to the given server.

    Raises:
      OSError: if the connection fails.

    See also:
      open_ssl_over_tcp_stream

    """

    # To keep our public API surface smaller, rule out some cases that
    # getaddrinfo will accept in some circumstances, but that act weird or
    # have non-portable behavior or are just plain not useful.
    if not isinstance(host, (str, bytes)):
        raise ValueError(f"host must be str or bytes, not {host!r}")
    if not isinstance(port, int):
        raise TypeError(f"port must be int, not {port!r}")

    if happy_eyeballs_delay is None:
        happy_eyeballs_delay = DEFAULT_DELAY

    targets = await getaddrinfo(host, port, type=SOCK_STREAM)

    # I don't think this can actually happen -- if there are no results,
    # getaddrinfo should have raised OSError instead of returning an empty
    # list. But let's be paranoid and handle it anyway:
    if not targets:
        msg = f"no results found for hostname lookup: {format_host_port(host, port)}"
        raise OSError(msg)

    reorder_for_rfc_6555_section_5_4(targets)

    # This list records all the connection failures that we ignored.
    oserrors: list[OSError] = []

    # Keeps track of the socket that we're going to complete with,
    # need to make sure this isn't automatically closed
    winning_socket: SocketType | None = None

    # Try connecting to the specified address. Possible outcomes:
    # - success: record connected socket in winning_socket and cancel
    #   concurrent attempts
    # - failure: record exception in oserrors, set attempt_failed allowing
    #   the next connection attempt to start early
    # code needs to ensure sockets can be closed appropriately in the
    # face of crash or cancellation
    async def attempt_connect(
        socket_args: tuple[AddressFamily, SocketKind, int],
        sockaddr: AddressFormat,
        attempt_failed: trio.Event,
    ) -> None:
        nonlocal winning_socket

        try:
            sock = socket(*socket_args)
            open_sockets.add(sock)

            if local_address is not None:
                # TCP connections are identified by a 4-tuple:
                #
                #   (local IP, local port, remote IP, remote port)
                #
                # So if a single local IP wants to make multiple connections
                # to the same (remote IP, remote port) pair, then those
                # connections have to use different local ports, or else TCP
                # won't be able to tell them apart. OTOH, if you have multiple
                # connections to different remote IP/ports, then those
                # connections can share a local port.
                #
                # Normally, when you call bind(), the kernel will immediately
                # assign a specific local port to your socket. At this point
                # the kernel doesn't know which (remote IP, remote port)
                # you're going to use, so it has to pick a local port that
                # *no* other connection is using. That's the only way to
                # guarantee that this local port will be usable later when we
                # call connect(). (Alternatively, you can set SO_REUSEADDR to
                # allow multiple nascent connections to share the same port,
                # but then connect() might fail with EADDRNOTAVAIL if we get
                # unlucky and our TCP 4-tuple ends up colliding with another
                # unrelated connection.)
                #
                # So calling bind() before connect() works, but it disables
                # sharing of local ports. This is inefficient: it makes you
                # more likely to run out of local ports.
                #
                # But on some versions of Linux, we can re-enable sharing of
                # local ports by setting a special flag. This flag tells
                # bind() to only bind the IP, and not the port. That way,
                # connect() is allowed to pick the the port, and it can do a
                # better job of it because it knows the remote IP/port.
                with suppress(OSError, AttributeError):
                    sock.setsockopt(
                        trio.socket.IPPROTO_IP,
                        trio.socket.IP_BIND_ADDRESS_NO_PORT,
                        1,
                    )
                try:
                    await sock.bind((local_address, 0))
                except OSError:
                    raise OSError(
                        f"local_address={local_address!r} is incompatible "
                        f"with remote address {sockaddr!r}",
                    ) from None

            await sock.connect(sockaddr)

            # Success! Save the winning socket and cancel all outstanding
            # connection attempts.
            winning_socket = sock
            nursery.cancel_scope.cancel()
        except OSError as exc:
            # This connection attempt failed, but the next one might
            # succeed. Save the error for later so we can report it if
            # everything fails, and tell the next attempt that it should go
            # ahead (if it hasn't already).
            oserrors.append(exc)
            attempt_failed.set()

    with close_all() as open_sockets:
        # nursery spawns a task for each connection attempt, will be
        # cancelled by the task that gets a successful connection
        async with trio.open_nursery() as nursery:
            for address_family, socket_type, proto, _, addr in targets:
                # create an event to indicate connection failure,
                # allowing the next target to be tried early
                attempt_failed = trio.Event()

                # workaround to check types until typing of nursery.start_soon improved
                if TYPE_CHECKING:
                    await attempt_connect(
                        (address_family, socket_type, proto),
                        addr,
                        attempt_failed,
                    )

                nursery.start_soon(
                    attempt_connect,
                    (address_family, socket_type, proto),
                    addr,
                    attempt_failed,
                )

                # give this attempt at most this time before moving on
                with trio.move_on_after(happy_eyeballs_delay):
                    await attempt_failed.wait()

        # nothing succeeded
        if winning_socket is None:
            assert len(oserrors) == len(targets)
            msg = f"all attempts to connect to {format_host_port(host, port)} failed"
            raise OSError(msg) from ExceptionGroup(msg, oserrors)
        else:
            stream = trio.SocketStream(winning_socket)
            open_sockets.remove(winning_socket)
            return stream

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\coloredlogs\converter\__init__.py ===
# Program to convert text with ANSI escape sequences to HTML.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: February 14, 2020
# URL: https://coloredlogs.readthedocs.io

"""Convert text with ANSI escape sequences to HTML."""

# Standard library modules.
import codecs
import os
import pipes
import re
import subprocess
import tempfile

# External dependencies.
from humanfriendly.terminal import (
    ANSI_CSI,
    ANSI_TEXT_STYLES,
    clean_terminal_output,
    output,
)

# Modules included in our package.
from coloredlogs.converter.colors import (
    BRIGHT_COLOR_PALETTE,
    EIGHT_COLOR_PALETTE,
    EXTENDED_COLOR_PALETTE,
)

# Compiled regular expression that matches leading spaces (indentation).
INDENT_PATTERN = re.compile('^ +', re.MULTILINE)

# Compiled regular expression that matches a tag followed by a space at the start of a line.
TAG_INDENT_PATTERN = re.compile('^(<[^>]+>) ', re.MULTILINE)

# Compiled regular expression that matches strings we want to convert. Used to
# separate all special strings and literal output in a single pass (this allows
# us to properly encode the output without resorting to nasty hacks).
TOKEN_PATTERN = re.compile(r'''
    # Wrap the pattern in a capture group so that re.split() includes the
    # substrings that match the pattern in the resulting list of strings.
    (
        # Match URLs with supported schemes and domain names.
        (?: https?:// | www\\. )
        # Scan until the end of the URL by matching non-whitespace characters
        # that are also not escape characters.
        [^\s\x1b]+
        # Alternatively ...
        |
        # Match (what looks like) ANSI escape sequences.
        \x1b \[ .*? m
    )
''', re.UNICODE | re.VERBOSE)


def capture(command, encoding='UTF-8'):
    """
    Capture the output of an external command as if it runs in an interactive terminal.

    :param command: The command name and its arguments (a list of strings).
    :param encoding: The encoding to use to decode the output (a string).
    :returns: The output of the command.

    This function runs an external command under ``script`` (emulating an
    interactive terminal) to capture the output of the command as if it was
    running in an interactive terminal (including ANSI escape sequences).
    """
    with open(os.devnull, 'wb') as dev_null:
        # We start by invoking the `script' program in a form that is supported
        # by the Linux implementation [1] but fails command line validation on
        # the MacOS (BSD) implementation [2]: The command is specified using
        # the -c option and the typescript file is /dev/null.
        #
        # [1] http://man7.org/linux/man-pages/man1/script.1.html
        # [2] https://developer.apple.com/legacy/library/documentation/Darwin/Reference/ManPages/man1/script.1.html
        command_line = ['script', '-qc', ' '.join(map(pipes.quote, command)), '/dev/null']
        script = subprocess.Popen(command_line, stdout=subprocess.PIPE, stderr=dev_null)
        stdout, stderr = script.communicate()
        if script.returncode == 0:
            # If `script' succeeded we assume that it understood our command line
            # invocation which means it's the Linux implementation (in this case
            # we can use standard output instead of a temporary file).
            output = stdout.decode(encoding)
        else:
            # If `script' failed we assume that it didn't understand our command
            # line invocation which means it's the MacOS (BSD) implementation
            # (in this case we need a temporary file because the command line
            # interface requires it).
            fd, temporary_file = tempfile.mkstemp(prefix='coloredlogs-', suffix='-capture.txt')
            try:
                command_line = ['script', '-q', temporary_file] + list(command)
                subprocess.Popen(command_line, stdout=dev_null, stderr=dev_null).wait()
                with codecs.open(temporary_file, 'rb') as handle:
                    output = handle.read()
            finally:
                os.unlink(temporary_file)
            # On MacOS when standard input is /dev/null I've observed
            # the captured output starting with the characters '^D':
            #
            #   $ script -q capture.txt echo example </dev/null
            #   example
            #   $ xxd capture.txt
            #   00000000: 5e44 0808 6578 616d 706c 650d 0a         ^D..example..
            #
            # I'm not sure why this is here, although I suppose it has to do
            # with ^D in caret notation signifying end-of-file [1]. What I do
            # know is that this is an implementation detail that callers of the
            # capture() function shouldn't be bothered with, so we strip it.
            #
            # [1] https://en.wikipedia.org/wiki/End-of-file
            if output.startswith(b'^D'):
                output = output[2:]
            output = output.decode(encoding)
    # Clean up backspace and carriage return characters and the 'erase line'
    # ANSI escape sequence and return the output as a Unicode string.
    return u'\n'.join(clean_terminal_output(output))


def convert(text, code=True, tabsize=4):
    """
    Convert text with ANSI escape sequences to HTML.

    :param text: The text with ANSI escape sequences (a string).
    :param code: Whether to wrap the returned HTML fragment in a
                 ``<code>...</code>`` element (a boolean, defaults
                 to :data:`True`).
    :param tabsize: Refer to :func:`str.expandtabs()` for details.
    :returns: The text converted to HTML (a string).
    """
    output = []
    in_span = False
    compatible_text_styles = {
        # The following ANSI text styles have an obvious mapping to CSS.
        ANSI_TEXT_STYLES['bold']: {'font-weight': 'bold'},
        ANSI_TEXT_STYLES['strike_through']: {'text-decoration': 'line-through'},
        ANSI_TEXT_STYLES['underline']: {'text-decoration': 'underline'},
    }
    for token in TOKEN_PATTERN.split(text):
        if token.startswith(('http://', 'https://', 'www.')):
            url = token if '://' in token else ('http://' + token)
            token = u'<a href="%s" style="color:inherit">%s</a>' % (html_encode(url), html_encode(token))
        elif token.startswith(ANSI_CSI):
            ansi_codes = token[len(ANSI_CSI):-1].split(';')
            if all(c.isdigit() for c in ansi_codes):
                ansi_codes = list(map(int, ansi_codes))
            # First we check for a reset code to close the previous <span>
            # element. As explained on Wikipedia [1] an absence of codes
            # implies a reset code as well: "No parameters at all in ESC[m acts
            # like a 0 reset code".
            # [1] https://en.wikipedia.org/wiki/ANSI_escape_code#CSI_sequences
            if in_span and (0 in ansi_codes or not ansi_codes):
                output.append('</span>')
                in_span = False
            # Now we're ready to generate the next <span> element (if any) in
            # the knowledge that we're emitting opening <span> and closing
            # </span> tags in the correct order.
            styles = {}
            is_faint = (ANSI_TEXT_STYLES['faint'] in ansi_codes)
            is_inverse = (ANSI_TEXT_STYLES['inverse'] in ansi_codes)
            while ansi_codes:
                number = ansi_codes.pop(0)
                # Try to match a compatible text style.
                if number in compatible_text_styles:
                    styles.update(compatible_text_styles[number])
                    continue
                # Try to extract a text and/or background color.
                text_color = None
                background_color = None
                if 30 <= number <= 37:
                    # 30-37 sets the text color from the eight color palette.
                    text_color = EIGHT_COLOR_PALETTE[number - 30]
                elif 40 <= number <= 47:
                    # 40-47 sets the background color from the eight color palette.
                    background_color = EIGHT_COLOR_PALETTE[number - 40]
                elif 90 <= number <= 97:
                    # 90-97 sets the text color from the high-intensity eight color palette.
                    text_color = BRIGHT_COLOR_PALETTE[number - 90]
                elif 100 <= number <= 107:
                    # 100-107 sets the background color from the high-intensity eight color palette.
                    background_color = BRIGHT_COLOR_PALETTE[number - 100]
                elif number in (38, 39) and len(ansi_codes) >= 2 and ansi_codes[0] == 5:
                    # 38;5;N is a text color in the 256 color mode palette,
                    # 39;5;N is a background color in the 256 color mode palette.
                    try:
                        # Consume the 5 following 38 or 39.
                        ansi_codes.pop(0)
                        # Consume the 256 color mode color index.
                        color_index = ansi_codes.pop(0)
                        # Set the variable to the corresponding HTML/CSS color.
                        if number == 38:
                            text_color = EXTENDED_COLOR_PALETTE[color_index]
                        elif number == 39:
                            background_color = EXTENDED_COLOR_PALETTE[color_index]
                    except (ValueError, IndexError):
                        pass
                # Apply the 'faint' or 'inverse' text style
                # by manipulating the selected color(s).
                if text_color and is_inverse:
                    # Use the text color as the background color and pick a
                    # text color that will be visible on the resulting
                    # background color.
                    background_color = text_color
                    text_color = select_text_color(*parse_hex_color(text_color))
                if text_color and is_faint:
                    # Because I wasn't sure how to implement faint colors
                    # based on normal colors I looked at how gnome-terminal
                    # (my terminal of choice) handles this and it appears
                    # to just pick a somewhat darker color.
                    text_color = '#%02X%02X%02X' % tuple(
                        max(0, n - 40) for n in parse_hex_color(text_color)
                    )
                if text_color:
                    styles['color'] = text_color
                if background_color:
                    styles['background-color'] = background_color
            if styles:
                token = '<span style="%s">' % ';'.join(k + ':' + v for k, v in sorted(styles.items()))
                in_span = True
            else:
                token = ''
        else:
            token = html_encode(token)
        output.append(token)
    html = ''.join(output)
    html = encode_whitespace(html, tabsize)
    if code:
        html = '<code>%s</code>' % html
    return html


def encode_whitespace(text, tabsize=4):
    """
    Encode whitespace so that web browsers properly render it.

    :param text: The plain text (a string).
    :param tabsize: Refer to :func:`str.expandtabs()` for details.
    :returns: The text converted to HTML (a string).

    The purpose of this function is to encode whitespace in such a way that web
    browsers render the same whitespace regardless of whether 'preformatted'
    styling is used (by wrapping the text in a ``<pre>...</pre>`` element).

    .. note:: While the string manipulation performed by this function is
              specifically intended not to corrupt the HTML generated by
              :func:`convert()` it definitely does have the potential to
              corrupt HTML from other sources. You have been warned :-).
    """
    # Convert Windows line endings (CR+LF) to UNIX line endings (LF).
    text = text.replace('\r\n', '\n')
    # Convert UNIX line endings (LF) to HTML line endings (<br>).
    text = text.replace('\n', '<br>\n')
    # Convert tabs to spaces.
    text = text.expandtabs(tabsize)
    # Convert leading spaces (that is to say spaces at the start of the string
    # and/or directly after a line ending) into non-breaking spaces, otherwise
    # HTML rendering engines will simply ignore these spaces.
    text = re.sub(INDENT_PATTERN, encode_whitespace_cb, text)
    # The conversion of leading spaces we just did misses a corner case where a
    # line starts with an HTML tag but the first visible text is a space. Web
    # browsers seem to ignore these spaces, so we need to convert them.
    text = re.sub(TAG_INDENT_PATTERN, r'\1&nbsp;', text)
    # Convert runs of multiple spaces into non-breaking spaces to avoid HTML
    # rendering engines from visually collapsing runs of spaces into a single
    # space. We specifically don't replace single spaces for several reasons:
    # 1. We'd break the HTML emitted by convert() by replacing spaces
    #    inside HTML elements (for example the spaces that separate
    #    element names from attribute names).
    # 2. If every single space is replaced by a non-breaking space,
    #    web browsers perform awkwardly unintuitive word wrapping.
    # 3. The HTML output would be bloated for no good reason.
    text = re.sub(' {2,}', encode_whitespace_cb, text)
    return text


def encode_whitespace_cb(match):
    """
    Replace runs of multiple spaces with non-breaking spaces.

    :param match: A regular expression match object.
    :returns: The replacement string.

    This function is used by func:`encode_whitespace()` as a callback for
    replacement using a regular expression pattern.
    """
    return '&nbsp;' * len(match.group(0))


def html_encode(text):
    """
    Encode characters with a special meaning as HTML.

    :param text: The plain text (a string).
    :returns: The text converted to HTML (a string).
    """
    text = text.replace('&', '&amp;')
    text = text.replace('<', '&lt;')
    text = text.replace('>', '&gt;')
    text = text.replace('"', '&quot;')
    return text


def parse_hex_color(value):
    """
    Convert a CSS color in hexadecimal notation into its R, G, B components.

    :param value: A CSS color in hexadecimal notation (a string like '#000000').
    :return: A tuple with three integers (with values between 0 and 255)
             corresponding to the R, G and B components of the color.
    :raises: :exc:`~exceptions.ValueError` on values that can't be parsed.
    """
    if value.startswith('#'):
        value = value[1:]
    if len(value) == 3:
        return (
            int(value[0] * 2, 16),
            int(value[1] * 2, 16),
            int(value[2] * 2, 16),
        )
    elif len(value) == 6:
        return (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
        )
    else:
        raise ValueError()


def select_text_color(r, g, b):
    """
    Choose a suitable color for the inverse text style.

    :param r: The amount of red (an integer between 0 and 255).
    :param g: The amount of green (an integer between 0 and 255).
    :param b: The amount of blue (an integer between 0 and 255).
    :returns: A CSS color in hexadecimal notation (a string).

    In inverse mode the color that is normally used for the text is instead
    used for the background, however this can render the text unreadable. The
    purpose of :func:`select_text_color()` is to make an effort to select a
    suitable text color. Based on http://stackoverflow.com/a/3943023/112731.
    """
    return '#000' if (r * 0.299 + g * 0.587 + b * 0.114) > 186 else '#FFF'


class ColoredCronMailer(object):

    """
    Easy to use integration between :mod:`coloredlogs` and the UNIX ``cron`` daemon.

    By using :class:`ColoredCronMailer` as a context manager in the command
    line interface of your Python program you make it trivially easy for users
    of your program to opt in to HTML output under ``cron``: The only thing the
    user needs to do is set ``CONTENT_TYPE="text/html"`` in their crontab!

    Under the hood this requires quite a bit of magic and I must admit that I
    developed this code simply because I was curious whether it could even be
    done :-). It requires my :mod:`capturer` package which you can install
    using ``pip install 'coloredlogs[cron]'``. The ``[cron]`` extra will pull
    in the :mod:`capturer` 2.4 or newer which is required to capture the output
    while silencing it - otherwise you'd get duplicate output in the emails
    sent by ``cron``.
    """

    def __init__(self):
        """Initialize output capturing when running under ``cron`` with the correct configuration."""
        self.is_enabled = 'text/html' in os.environ.get('CONTENT_TYPE', 'text/plain')
        self.is_silent = False
        if self.is_enabled:
            # We import capturer here so that the coloredlogs[cron] extra
            # isn't required to use the other functions in this module.
            from capturer import CaptureOutput
            self.capturer = CaptureOutput(merged=True, relay=False)

    def __enter__(self):
        """Start capturing output (when applicable)."""
        if self.is_enabled:
            self.capturer.__enter__()
        return self

    def __exit__(self, exc_type=None, exc_value=None, traceback=None):
        """Stop capturing output and convert the output to HTML (when applicable)."""
        if self.is_enabled:
            if not self.is_silent:
                # Only call output() when we captured something useful.
                text = self.capturer.get_text()
                if text and not text.isspace():
                    output(convert(text))
            self.capturer.__exit__(exc_type, exc_value, traceback)

    def silence(self):
        """
        Tell :func:`__exit__()` to swallow all output (things will be silent).

        This can be useful when a Python program is written in such a way that
        it has already produced output by the time it becomes apparent that
        nothing useful can be done (say in a cron job that runs every few
        minutes :-p). By calling :func:`silence()` the output can be swallowed
        retroactively, avoiding useless emails from ``cron``.
        """
        self.is_silent = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\weyl_group.py ===
# -*- coding: utf-8 -*-

from .cartan_type import CartanType
from mpmath import fac
from sympy.core.backend import Matrix, eye, Rational, igcd
from sympy.core.basic import Atom

class WeylGroup(Atom):

    """
    For each semisimple Lie group, we have a Weyl group.  It is a subgroup of
    the isometry group of the root system.  Specifically, it's the subgroup
    that is generated by reflections through the hyperplanes orthogonal to
    the roots.  Therefore, Weyl groups are reflection groups, and so a Weyl
    group is a finite Coxeter group.

    """

    def __new__(cls, cartantype):
        obj = Atom.__new__(cls)
        obj.cartan_type = CartanType(cartantype)
        return obj

    def generators(self):
        """
        This method creates the generating reflections of the Weyl group for
        a given Lie algebra.  For a Lie algebra of rank n, there are n
        different generating reflections.  This function returns them as
        a list.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("F4")
        >>> c.generators()
        ['r1', 'r2', 'r3', 'r4']
        """
        n = self.cartan_type.rank()
        generators = []
        for i in range(1, n+1):
            reflection = "r"+str(i)
            generators.append(reflection)
        return generators

    def group_order(self):
        """
        This method returns the order of the Weyl group.
        For types A, B, C, D, and E the order depends on
        the rank of the Lie algebra.  For types F and G,
        the order is fixed.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("D4")
        >>> c.group_order()
        192.0
        """
        n = self.cartan_type.rank()
        if self.cartan_type.series == "A":
            return fac(n+1)

        if self.cartan_type.series in ("B", "C"):
            return fac(n)*(2**n)

        if self.cartan_type.series == "D":
            return fac(n)*(2**(n-1))

        if self.cartan_type.series == "E":
            if n == 6:
                return 51840
            if n == 7:
                return 2903040
            if n == 8:
                return 696729600
        if self.cartan_type.series == "F":
            return 1152

        if self.cartan_type.series == "G":
            return 12

    def group_name(self):
        """
        This method returns some general information about the Weyl group for
        a given Lie algebra.  It returns the name of the group and the elements
        it acts on, if relevant.
        """
        n = self.cartan_type.rank()
        if self.cartan_type.series == "A":
            return "S"+str(n+1) + ": the symmetric group acting on " + str(n+1) + " elements."

        if self.cartan_type.series in ("B", "C"):
            return "The hyperoctahedral group acting on " + str(2*n) + " elements."

        if self.cartan_type.series == "D":
            return "The symmetry group of the " + str(n) + "-dimensional demihypercube."

        if self.cartan_type.series == "E":
            if n == 6:
                return "The symmetry group of the 6-polytope."

            if n == 7:
                return "The symmetry group of the 7-polytope."

            if n == 8:
                return "The symmetry group of the 8-polytope."

        if self.cartan_type.series == "F":
            return "The symmetry group of the 24-cell, or icositetrachoron."

        if self.cartan_type.series == "G":
            return "D6, the dihedral group of order 12, and symmetry group of the hexagon."

    def element_order(self, weylelt):
        """
        This method returns the order of a given Weyl group element, which should
        be specified by the user in the form of products of the generating
        reflections, i.e. of the form r1*r2 etc.

        For types A-F, this method current works by taking the matrix form of
        the specified element, and then finding what power of the matrix is the
        identity.  It then returns this power.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> b = WeylGroup("B4")
        >>> b.element_order('r1*r4*r2')
        4
        """
        n = self.cartan_type.rank()
        if self.cartan_type.series == "A":
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(n+1):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        if self.cartan_type.series == "D":
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(n):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        if self.cartan_type.series == "E":
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(8):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

        if self.cartan_type.series == "G":
            elts = list(weylelt)
            reflections = elts[1::3]
            m = self.delete_doubles(reflections)
            while self.delete_doubles(m) != m:
                m = self.delete_doubles(m)
                reflections = m
            if len(reflections) % 2 == 1:
                return 2

            elif len(reflections) == 0:
                return 1

            else:
                if len(reflections) == 1:
                    return 2
                else:
                    m = len(reflections) // 2
                    lcm = (6 * m)/ igcd(m, 6)
                order = lcm / m
                return order


        if self.cartan_type.series == 'F':
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(4):
                a *= self.matrix_form(weylelt)
                order += 1
            return order


        if self.cartan_type.series in ("B", "C"):
            a = self.matrix_form(weylelt)
            order = 1
            while a != eye(n):
                a *= self.matrix_form(weylelt)
                order += 1
            return order

    def delete_doubles(self, reflections):
        """
        This is a helper method for determining the order of an element in the
        Weyl group of G2.  It takes a Weyl element and if repeated simple reflections
        in it, it deletes them.
        """
        counter = 0
        copy = list(reflections)
        for elt in copy:
            if counter < len(copy)-1:
                if copy[counter + 1] == elt:
                    del copy[counter]
                    del copy[counter]
            counter += 1


        return copy


    def matrix_form(self, weylelt):
        """
        This method takes input from the user in the form of products of the
        generating reflections, and returns the matrix corresponding to the
        element of the Weyl group.  Since each element of the Weyl group is
        a reflection of some type, there is a corresponding matrix representation.
        This method uses the standard representation for all the generating
        reflections.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> f = WeylGroup("F4")
        >>> f.matrix_form('r2*r3')
        Matrix([
        [1, 0, 0,  0],
        [0, 1, 0,  0],
        [0, 0, 0, -1],
        [0, 0, 1,  0]])

        """
        elts = list(weylelt)
        reflections = elts[1::3]
        n = self.cartan_type.rank()
        if self.cartan_type.series == 'A':
            matrixform = eye(n+1)
            for elt in reflections:
                a = int(elt)
                mat = eye(n+1)
                mat[a-1, a-1] = 0
                mat[a-1, a] = 1
                mat[a, a-1] = 1
                mat[a, a] = 0
                matrixform *= mat
            return matrixform

        if self.cartan_type.series == 'D':
            matrixform = eye(n)
            for elt in reflections:
                a = int(elt)
                mat = eye(n)
                if a < n:
                    mat[a-1, a-1] = 0
                    mat[a-1, a] = 1
                    mat[a, a-1] = 1
                    mat[a, a] = 0
                    matrixform *= mat
                else:
                    mat[n-2, n-1] = -1
                    mat[n-2, n-2] = 0
                    mat[n-1, n-2] = -1
                    mat[n-1, n-1] = 0
                    matrixform *= mat
            return matrixform

        if self.cartan_type.series == 'G':
            matrixform = eye(3)
            for elt in reflections:
                a = int(elt)
                if a == 1:
                    gen1 = Matrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
                    matrixform *= gen1
                else:
                    gen2 = Matrix([[Rational(2, 3), Rational(2, 3), Rational(-1, 3)],
                                   [Rational(2, 3), Rational(-1, 3), Rational(2, 3)],
                                   [Rational(-1, 3), Rational(2, 3), Rational(2, 3)]])
                    matrixform *= gen2
            return matrixform

        if self.cartan_type.series == 'F':
            matrixform = eye(4)
            for elt in reflections:
                a = int(elt)
                if a == 1:
                    mat = Matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
                    matrixform *= mat
                elif a == 2:
                    mat = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
                    matrixform *= mat
                elif a == 3:
                    mat = Matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]])
                    matrixform *= mat
                else:

                    mat = Matrix([[Rational(1, 2), Rational(1, 2), Rational(1, 2), Rational(1, 2)],
                        [Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2)],
                        [Rational(1, 2), Rational(-1, 2), Rational(1, 2), Rational(-1, 2)],
                        [Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]])
                    matrixform *= mat
            return matrixform

        if self.cartan_type.series == 'E':
            matrixform = eye(8)
            for elt in reflections:
                a = int(elt)
                if a == 1:
                    mat = Matrix([[Rational(3, 4), Rational(1, 4), Rational(1, 4), Rational(1, 4),
                        Rational(1, 4), Rational(1, 4), Rational(1, 4), Rational(-1, 4)],
                        [Rational(1, 4), Rational(3, 4), Rational(-1, 4), Rational(-1, 4),
                            Rational(-1, 4), Rational(-1, 4), Rational(1, 4), Rational(-1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(3, 4), Rational(-1, 4),
                        Rational(-1, 4), Rational(-1, 4), Rational(-1, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(3, 4),
                        Rational(-1, 4), Rational(-1, 4), Rational(-1, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4),
                        Rational(3, 4), Rational(-1, 4), Rational(-1, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4),
                        Rational(-1, 4), Rational(3, 4), Rational(-1, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4),
                        Rational(-1, 4), Rational(-1, 4), Rational(-3, 4), Rational(1, 4)],
                        [Rational(1, 4), Rational(-1, 4), Rational(-1, 4), Rational(-1, 4),
                        Rational(-1, 4), Rational(-1, 4), Rational(-1, 4), Rational(3, 4)]])
                    matrixform *= mat
                elif a == 2:
                    mat = eye(8)
                    mat[0, 0] = 0
                    mat[0, 1] = -1
                    mat[1, 0] = -1
                    mat[1, 1] = 0
                    matrixform *= mat
                else:
                    mat = eye(8)
                    mat[a-3, a-3] = 0
                    mat[a-3, a-2] = 1
                    mat[a-2, a-3] = 1
                    mat[a-2, a-2] = 0
                    matrixform *= mat
            return matrixform


        if self.cartan_type.series in ("B", "C"):
            matrixform = eye(n)
            for elt in reflections:
                a = int(elt)
                mat = eye(n)
                if a == 1:
                    mat[0, 0] = -1
                    matrixform *= mat
                else:
                    mat[a - 2, a - 2] = 0
                    mat[a-2, a-1] = 1
                    mat[a - 1, a - 2] = 1
                    mat[a -1, a - 1] = 0
                    matrixform *= mat
            return matrixform



    def coxeter_diagram(self):
        """
        This method returns the Coxeter diagram corresponding to a Weyl group.
        The Coxeter diagram can be obtained from a Lie algebra's Dynkin diagram
        by deleting all arrows; the Coxeter diagram is the undirected graph.
        The vertices of the Coxeter diagram represent the generating reflections
        of the Weyl group, $s_i$.  An edge is drawn between $s_i$ and $s_j$ if the order
        $m(i, j)$ of $s_is_j$ is greater than two.  If there is one edge, the order
        $m(i, j)$ is 3.  If there are two edges, the order $m(i, j)$ is 4, and if there
        are three edges, the order $m(i, j)$ is 6.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> c = WeylGroup("B3")
        >>> print(c.coxeter_diagram())
        0---0===0
        1   2   3
        """
        n = self.cartan_type.rank()
        if self.cartan_type.series in ("A", "D", "E"):
            return self.cartan_type.dynkin_diagram()

        if self.cartan_type.series in ("B", "C"):
            diag = "---".join("0" for i in range(1, n)) + "===0\n"
            diag += "   ".join(str(i) for i in range(1, n+1))
            return diag

        if self.cartan_type.series == "F":
            diag = "0---0===0---0\n"
            diag += "   ".join(str(i) for i in range(1, 5))
            return diag

        if self.cartan_type.series == "G":
            diag = "0≡≡≡0\n1   2"
            return diag

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_tools\gen_exports.py ===
#! /usr/bin/env python3
"""
Code generation script for class methods
to be exported as public API
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from pathlib import Path
from textwrap import indent
from typing import TYPE_CHECKING

import attrs

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import TypeGuard

# keep these imports up to date with conditional imports in test_gen_exports
# isort: split
import astor

PREFIX = "_generated"

HEADER = """# ***********************************************************
# ******* WARNING: AUTOGENERATED! ALL EDITS WILL BE LOST ******
# *************************************************************
from __future__ import annotations

import sys

from ._ki import enable_ki_protection
from ._run import GLOBAL_RUN_CONTEXT
"""

TEMPLATE = """try:
    return{}GLOBAL_RUN_CONTEXT.{}.{}
except AttributeError:
    raise RuntimeError("must be called from async context") from None
"""


@attrs.define
class File:
    path: Path
    modname: str
    platform: str = attrs.field(default="", kw_only=True)
    imports: str = attrs.field(default="", kw_only=True)


def is_function(node: ast.AST) -> TypeGuard[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Check if the AST node is either a function
    or an async function
    """
    return isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))


def is_public(node: ast.AST) -> TypeGuard[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Check if the AST node has a _public decorator"""
    if is_function(node):
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "_public":
                return True
    return False


def get_public_methods(
    tree: ast.AST,
) -> Iterator[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return a list of methods marked as public.
    The function walks the given tree and extracts
    all objects that are functions which are marked
    public.
    """
    for node in ast.walk(tree):
        if is_public(node):
            yield node


def create_passthrough_args(funcdef: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Given a function definition, create a string that represents taking all
    the arguments from the function, and passing them through to another
    invocation of the same function.

    Example input: ast.parse("def f(a, *, b): ...")
    Example output: "(a, b=b)"
    """
    call_args = [arg.arg for arg in funcdef.args.args]
    if funcdef.args.vararg:
        call_args.append("*" + funcdef.args.vararg.arg)
    for arg in funcdef.args.kwonlyargs:
        call_args.append(arg.arg + "=" + arg.arg)  # noqa: PERF401  # clarity
    if funcdef.args.kwarg:
        call_args.append("**" + funcdef.args.kwarg.arg)
    return "({})".format(", ".join(call_args))


def run_black(file: File, source: str) -> tuple[bool, str]:
    """Run black on the specified file.

    Returns:
      Tuple of success and result string.
      ex.:
        (False, "Failed to run black!\nerror: cannot format ...")
        (True, "<formatted source>")

    Raises:
      ImportError: If black is not installed.
    """
    # imported to check that `subprocess` calls will succeed
    import black  # noqa: F401

    # Black has an undocumented API, but it doesn't easily allow reading configuration from
    # pyproject.toml, and simultaneously pass in / receive the code as a string.
    # https://github.com/psf/black/issues/779
    result = subprocess.run(
        # "-" as a filename = use stdin, return on stdout.
        [sys.executable, "-m", "black", "--stdin-filename", file.path, "-"],
        input=source,
        capture_output=True,
        encoding="utf8",
    )

    if result.returncode != 0:
        return False, f"Failed to run black!\n{result.stderr}"
    return True, result.stdout


def run_ruff(file: File, source: str) -> tuple[bool, str]:
    """Run ruff on the specified file.

    Returns:
      Tuple of success and result string.
      ex.:
        (False, "Failed to run ruff!\nerror: Failed to parse ...")
        (True, "<formatted source>")

    Raises:
      ImportError: If ruff is not installed.
    """
    # imported to check that `subprocess` calls will succeed
    import ruff  # noqa: F401

    result = subprocess.run(
        # "-" as a filename = use stdin, return on stdout.
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--fix",
            "--unsafe-fixes",
            "--stdin-filename",
            file.path,
            "-",
        ],
        input=source,
        capture_output=True,
        encoding="utf8",
    )

    if result.returncode != 0:
        return False, f"Failed to run ruff!\n{result.stderr}"
    return True, result.stdout


def run_linters(file: File, source: str) -> str:
    """Format the specified file using black and ruff.

    Returns:
      Formatted source code.

    Raises:
      ImportError: If either is not installed.
      SystemExit: If either failed.
    """

    for fn in (run_black, run_ruff):
        success, source = fn(file, source)
        if not success:
            print(source)
            sys.exit(1)

    return source


def gen_public_wrappers_source(file: File) -> str:
    """Scan the given .py file for @_public decorators, and generate wrapper
    functions.

    """
    header = [HEADER]
    header.append(file.imports)
    if file.platform:
        # Simple checks to avoid repeating imports. If this messes up, type checkers/tests will
        # just give errors.
        if "TYPE_CHECKING" not in file.imports:
            header.append("from typing import TYPE_CHECKING\n")
        if "import sys" not in file.imports:  # pragma: no cover
            header.append("import sys\n")
        header.append(
            f'\nassert not TYPE_CHECKING or sys.platform=="{file.platform}"\n',
        )

    generated = ["".join(header)]

    source = astor.code_to_ast.parse_file(file.path)
    method_names = []
    for method in get_public_methods(source):
        # Remove self from arguments
        assert method.args.args[0].arg == "self"
        del method.args.args[0]
        method_names.append(method.name)

        for dec in method.decorator_list:  # pragma: no cover
            if isinstance(dec, ast.Name) and dec.id == "contextmanager":
                is_cm = True
                break
        else:
            is_cm = False

        # Remove decorators
        method.decorator_list = [ast.Name("enable_ki_protection")]

        # Create pass through arguments
        new_args = create_passthrough_args(method)

        # Remove method body without the docstring
        if ast.get_docstring(method) is None:
            del method.body[:]
        else:
            # The first entry is always the docstring
            del method.body[1:]

        # Create the function definition including the body
        func = astor.to_source(method, indent_with=" " * 4)

        if is_cm:  # pragma: no cover
            func = func.replace("->Iterator", "->AbstractContextManager")

        # Create export function body
        template = TEMPLATE.format(
            " await " if isinstance(method, ast.AsyncFunctionDef) else " ",
            file.modname,
            method.name + new_args,
        )

        # Assemble function definition arguments and body
        snippet = func + indent(template, " " * 4)

        # Append the snippet to the corresponding module
        generated.append(snippet)

    method_names.sort()
    # Insert after the header, before function definitions
    generated.insert(1, f"__all__ = {method_names!r}")
    return "\n\n".join(generated)


def matches_disk_files(new_files: dict[str, str]) -> bool:
    for new_path, new_source in new_files.items():
        if not os.path.exists(new_path):
            return False
        old_source = Path(new_path).read_text(encoding="utf-8")
        if old_source != new_source:
            return False
    return True


def process(files: Iterable[File], *, do_test: bool) -> None:
    new_files = {}
    for file in files:
        print("Scanning:", file.path)
        new_source = gen_public_wrappers_source(file)
        new_source = run_linters(file, new_source)
        dirname, basename = os.path.split(file.path)
        new_path = os.path.join(dirname, PREFIX + basename)
        new_files[new_path] = new_source
    matches_disk = matches_disk_files(new_files)
    if do_test:
        if not matches_disk:
            print("Generated sources are outdated. Please regenerate.")
            sys.exit(1)
        else:
            print("Generated sources are up to date.")
    else:
        for new_path, new_source in new_files.items():
            with open(new_path, "w", encoding="utf-8", newline="\n") as fp:
                fp.write(new_source)
        print("Regenerated sources successfully.")
        if not matches_disk:  # TODO: test this branch
            # With pre-commit integration, show that we edited files.
            sys.exit(1)


# This is in fact run in CI, but only in the formatting check job, which
# doesn't collect coverage.
def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Generate python code for public api wrappers",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="test if code is still up to date",
    )
    parsed_args = parser.parse_args()

    source_root = Path.cwd()
    # Double-check we found the right directory
    assert (source_root / "LICENSE").exists()
    core = source_root / "src/trio/_core"
    to_wrap = [
        File(core / "_run.py", "runner", imports=IMPORTS_RUN),
        File(
            core / "_instrumentation.py",
            "runner.instruments",
            imports=IMPORTS_INSTRUMENT,
        ),
        File(
            core / "_io_windows.py",
            "runner.io_manager",
            platform="win32",
            imports=IMPORTS_WINDOWS,
        ),
        File(
            core / "_io_epoll.py",
            "runner.io_manager",
            platform="linux",
            imports=IMPORTS_EPOLL,
        ),
        File(
            core / "_io_kqueue.py",
            "runner.io_manager",
            platform="darwin",
            imports=IMPORTS_KQUEUE,
        ),
    ]

    process(to_wrap, do_test=parsed_args.test)


IMPORTS_RUN = """\
from collections.abc import Awaitable, Callable
from typing import Any, TYPE_CHECKING

from outcome import Outcome
import contextvars

from ._run import _NO_SEND, RunStatistics, Task
from ._entry_queue import TrioToken
from .._abc import Clock

if TYPE_CHECKING:
    from typing_extensions import Unpack
    from ._run import PosArgT
"""
IMPORTS_INSTRUMENT = """\
from ._instrumentation import Instrument
"""

IMPORTS_EPOLL = """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._file_io import _HasFileNo
"""

IMPORTS_KQUEUE = """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import select
    from collections.abc import Callable
    from contextlib import AbstractContextManager

    from .. import _core
    from .._file_io import _HasFileNo
    from ._traps import Abort, RaiseCancelT
"""

IMPORTS_WINDOWS = """\
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from typing_extensions import Buffer

    from .._file_io import _HasFileNo
    from ._unbounded_queue import UnboundedQueue
    from ._windows_cffi import Handle, CData
"""


if __name__ == "__main__":  # pragma: no cover
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\rcode.py ===
"""
R code printer

The RCodePrinter converts single SymPy expressions into single R expressions,
using the functions defined in math.h where possible.



"""

from __future__ import annotations
from typing import Any

from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence, PRECEDENCE
from sympy.sets.fancysets import Range

# dictionary mapping SymPy function to (argument_conditions, C_function).
# Used in RCodePrinter._print_Function(self)
known_functions = {
    #"Abs": [(lambda x: not x.is_integer, "fabs")],
    "Abs": "abs",
    "sin": "sin",
    "cos": "cos",
    "tan": "tan",
    "asin": "asin",
    "acos": "acos",
    "atan": "atan",
    "atan2": "atan2",
    "exp": "exp",
    "log": "log",
    "erf": "erf",
    "sinh": "sinh",
    "cosh": "cosh",
    "tanh": "tanh",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "floor": "floor",
    "ceiling": "ceiling",
    "sign": "sign",
    "Max": "max",
    "Min": "min",
    "factorial": "factorial",
    "gamma": "gamma",
    "digamma": "digamma",
    "trigamma": "trigamma",
    "beta": "beta",
    "sqrt": "sqrt",  # To enable automatic rewrite
}

# These are the core reserved words in the R language. Taken from:
# https://cran.r-project.org/doc/manuals/r-release/R-lang.html#Reserved-words

reserved_words = ['if',
                  'else',
                  'repeat',
                  'while',
                  'function',
                  'for',
                  'in',
                  'next',
                  'break',
                  'TRUE',
                  'FALSE',
                  'NULL',
                  'Inf',
                  'NaN',
                  'NA',
                  'NA_integer_',
                  'NA_real_',
                  'NA_complex_',
                  'NA_character_',
                  'volatile']


class RCodePrinter(CodePrinter):
    """A printer to convert SymPy expressions to strings of R code"""
    printmethod = "_rcode"
    language = "R"

    _default_settings: dict[str, Any] = dict(CodePrinter._default_settings, **{
        'precision': 15,
        'user_functions': {},
        'contract': True,
        'dereference': set(),
    })
    _operators = {
       'and': '&',
        'or': '|',
       'not': '!',
    }

    _relationals: dict[str, str] = {}

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)
        self._dereference = set(settings.get('dereference', []))
        self.reserved_words = set(reserved_words)

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// {}".format(text)

    def _declare_number_const(self, name, value):
        return "{} = {};".format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def _traverse_matrix_indices(self, mat):
        rows, cols = mat.shape
        return ((i, j) for i in range(rows) for j in range(cols))

    def _get_loop_opening_ending(self, indices):
        """Returns a tuple (open_lines, close_lines) containing lists of codelines
        """
        open_lines = []
        close_lines = []
        loopstart = "for (%(var)s in %(start)s:%(end)s){"
        for i in indices:
            # R arrays start at 1 and end at dimension
            open_lines.append(loopstart % {
                'var': self._print(i.label),
                'start': self._print(i.lower+1),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    def _print_Pow(self, expr):
        if "Pow" in self.known_functions:
            return self._print_Function(expr)
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            return '%s^%s' % (self.parenthesize(expr.base, PREC),
                                 self.parenthesize(expr.exp, PREC))


    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d.0/%d.0' % (p, q)

    def _print_Indexed(self, expr):
        inds = [ self._print(i) for i in expr.indices ]
        return "%s[%s]" % (self._print(expr.base.label), ", ".join(inds))

    def _print_Exp1(self, expr):
        return "exp(1)"

    def _print_Pi(self, expr):
        return 'pi'

    def _print_Infinity(self, expr):
        return 'Inf'

    def _print_NegativeInfinity(self, expr):
        return '-Inf'

    def _print_Assignment(self, expr):
        from sympy.codegen.ast import Assignment

        from sympy.matrices.expressions.matexpr import MatrixSymbol
        from sympy.tensor.indexed import IndexedBase
        lhs = expr.lhs
        rhs = expr.rhs
        # We special case assignments that take multiple lines
        #if isinstance(expr.rhs, Piecewise):
        #    from sympy.functions.elementary.piecewise import Piecewise
        #    # Here we modify Piecewise so each expression is now
        #    # an Assignment, and then continue on the print.
        #    expressions = []
        #    conditions = []
        #    for (e, c) in rhs.args:
        #        expressions.append(Assignment(lhs, e))
        #        conditions.append(c)
        #    temp = Piecewise(*zip(expressions, conditions))
        #    return self._print(temp)
        #elif isinstance(lhs, MatrixSymbol):
        if isinstance(lhs, MatrixSymbol):
            # Here we form an Assignment for each element in the array,
            # printing each one.
            lines = []
            for (i, j) in self._traverse_matrix_indices(lhs):
                temp = Assignment(lhs[i, j], rhs[i, j])
                code0 = self._print(temp)
                lines.append(code0)
            return "\n".join(lines)
        elif self._settings["contract"] and (lhs.has(IndexedBase) or
                rhs.has(IndexedBase)):
            # Here we check if there is looping to be done, and if so
            # print the required loops.
            return self._doprint_loops(rhs, lhs)
        else:
            lhs_code = self._print(lhs)
            rhs_code = self._print(rhs)
            return self._get_statement("%s = %s" % (lhs_code, rhs_code))

    def _print_Piecewise(self, expr):
        # This method is called only for inline if constructs
        # Top level piecewise is handled in doprint()
        if expr.args[-1].cond == True:
            last_line = "%s" % self._print(expr.args[-1].expr)
        else:
            last_line = "ifelse(%s,%s,NA)" % (self._print(expr.args[-1].cond), self._print(expr.args[-1].expr))
        code=last_line
        for e, c in reversed(expr.args[:-1]):
            code= "ifelse(%s,%s," % (self._print(c), self._print(e))+code+")"
        return(code)

    def _print_ITE(self, expr):
        from sympy.functions import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_MatrixElement(self, expr):
        return "{}[{}]".format(self.parenthesize(expr.parent, PRECEDENCE["Atom"],
            strict=True), expr.j + expr.i*expr.parent.shape[1])

    def _print_Symbol(self, expr):
        name = super()._print_Symbol(expr)
        if expr in self._dereference:
            return '(*{})'.format(name)
        else:
            return name

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_AugmentedAssignment(self, expr):
        lhs_code = self._print(expr.lhs)
        op = expr.op
        rhs_code = self._print(expr.rhs)
        return "{} {} {};".format(lhs_code, op, rhs_code)

    def _print_For(self, expr):
        target = self._print(expr.target)
        if isinstance(expr.iterable, Range):
            start, stop, step = expr.iterable.args
        else:
            raise NotImplementedError("Only iterable currently supported is Range")
        body = self._print(expr.body)
        return 'for({target} in seq(from={start}, to={stop}, by={step}){{\n{body}\n}}'.format(target=target, start=start,
                stop=stop-1, step=step, body=body)


    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [ line.lstrip(' \t') for line in code ]

        increase = [ int(any(map(line.endswith, inc_token))) for line in code ]
        decrease = [ int(any(map(line.startswith, dec_token)))
                     for line in code ]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty


def rcode(expr, assign_to=None, **settings):
    """Converts an expr to a string of r code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used as the name of the variable to which
        the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol``, or ``Indexed`` type. This is helpful in case of
        line-wrapping, or for expressions that generate multi-line statements.
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where the keys are string representations of either
        ``FunctionClass`` or ``UndefinedFunction`` instances and the values
        are their desired R string representations. Alternatively, the
        dictionary value can be a list of tuples i.e. [(argument_test,
        rfunction_string)] or [(argument_test, rfunction_formater)]. See below
        for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import rcode, symbols, Rational, sin, ceiling, Abs, Function
    >>> x, tau = symbols("x, tau")
    >>> rcode((2*tau)**Rational(7, 2))
    '8*sqrt(2)*tau^(7.0/2.0)'
    >>> rcode(sin(x), assign_to="s")
    's = sin(x);'

    Simple custom printing can be defined for certain types by passing a
    dictionary of {"type" : "function"} to the ``user_functions`` kwarg.
    Alternatively, the dictionary value can be a list of tuples i.e.
    [(argument_test, cfunction_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")],
    ...   "func": "f"
    ... }
    >>> func = Function('func')
    >>> rcode(func(Abs(x) + ceiling(x)), user_functions=custom_functions)
    'f(fabs(x) + CEIL(x))'

    or if the R-function takes a subset of the original arguments:

    >>> rcode(2**x + 3**x, user_functions={'Pow': [
    ...   (lambda b, e: b == 2, lambda b, e: 'exp2(%s)' % e),
    ...   (lambda b, e: b != 2, 'pow')]})
    'exp2(x) + pow(3, x)'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(rcode(expr, assign_to=tau))
    tau = ifelse(x > 0,x + 1,x);

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> rcode(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    Matrices are also supported, but a ``MatrixSymbol`` of the same dimensions
    must be provided to ``assign_to``. Note that any expression that can be
    generated normally can also exist inside a Matrix:

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(rcode(mat, A))
    A[0] = x^2;
    A[1] = ifelse(x > 0,x + 1,x);
    A[2] = sin(x);

    """

    return RCodePrinter(settings).doprint(expr, assign_to)


def print_rcode(expr, **settings):
    """Prints R representation of the given expression."""
    print(rcode(expr, **settings))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\axis.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Float,
    NoneSet,
    Bool,
    Integer,
    MinMax,
    NoneSet,
    Set,
    String,
    Alias,
)

from openpyxl.descriptors.excel import (
    ExtensionList,
    Percentage,
    _explicit_none,
)
from openpyxl.descriptors.nested import (
    NestedValue,
    NestedSet,
    NestedBool,
    NestedNoneSet,
    NestedFloat,
    NestedInteger,
    NestedMinMax,
)
from openpyxl.xml.constants import CHART_NS

from .descriptors import NumberFormatDescriptor
from .layout import Layout
from .text import Text, RichText
from .shapes import GraphicalProperties
from .title import Title, TitleDescriptor


class ChartLines(Serialisable):

    tagname = "chartLines"

    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias('spPr')

    def __init__(self, spPr=None):
        self.spPr = spPr


class Scaling(Serialisable):

    tagname = "scaling"

    logBase = NestedFloat(allow_none=True)
    orientation = NestedSet(values=(['maxMin', 'minMax']))
    max = NestedFloat(allow_none=True)
    min = NestedFloat(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('logBase', 'orientation', 'max', 'min',)

    def __init__(self,
                 logBase=None,
                 orientation="minMax",
                 max=None,
                 min=None,
                 extLst=None,
                ):
        self.logBase = logBase
        self.orientation = orientation
        self.max = max
        self.min = min


class _BaseAxis(Serialisable):

    axId = NestedInteger(expected_type=int)
    scaling = Typed(expected_type=Scaling)
    delete = NestedBool(allow_none=True)
    axPos = NestedSet(values=(['b', 'l', 'r', 't']))
    majorGridlines = Typed(expected_type=ChartLines, allow_none=True)
    minorGridlines = Typed(expected_type=ChartLines, allow_none=True)
    title = TitleDescriptor()
    numFmt = NumberFormatDescriptor()
    number_format = Alias("numFmt")
    majorTickMark = NestedNoneSet(values=(['cross', 'in', 'out']), to_tree=_explicit_none)
    minorTickMark = NestedNoneSet(values=(['cross', 'in', 'out']), to_tree=_explicit_none)
    tickLblPos = NestedNoneSet(values=(['high', 'low', 'nextTo']))
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias('spPr')
    txPr = Typed(expected_type=RichText, allow_none=True)
    textProperties = Alias('txPr')
    crossAx = NestedInteger(expected_type=int) # references other axis
    crosses = NestedNoneSet(values=(['autoZero', 'max', 'min']))
    crossesAt = NestedFloat(allow_none=True)

    # crosses & crossesAt are mutually exclusive

    __elements__ = ('axId', 'scaling', 'delete', 'axPos', 'majorGridlines',
                    'minorGridlines', 'title', 'numFmt', 'majorTickMark', 'minorTickMark',
                    'tickLblPos', 'spPr', 'txPr', 'crossAx', 'crosses', 'crossesAt')

    def __init__(self,
                 axId=None,
                 scaling=None,
                 delete=None,
                 axPos='l',
                 majorGridlines=None,
                 minorGridlines=None,
                 title=None,
                 numFmt=None,
                 majorTickMark=None,
                 minorTickMark=None,
                 tickLblPos=None,
                 spPr=None,
                 txPr= None,
                 crossAx=None,
                 crosses=None,
                 crossesAt=None,
                ):
        self.axId = axId
        if scaling is None:
            scaling = Scaling()
        self.scaling = scaling
        self.delete = delete
        self.axPos = axPos
        self.majorGridlines = majorGridlines
        self.minorGridlines = minorGridlines
        self.title = title
        self.numFmt = numFmt
        self.majorTickMark = majorTickMark
        self.minorTickMark = minorTickMark
        self.tickLblPos = tickLblPos
        self.spPr = spPr
        self.txPr = txPr
        self.crossAx = crossAx
        self.crosses = crosses
        self.crossesAt = crossesAt


class DisplayUnitsLabel(Serialisable):

    tagname = "dispUnitsLbl"

    layout = Typed(expected_type=Layout, allow_none=True)
    tx = Typed(expected_type=Text, allow_none=True)
    text = Alias("tx")
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias("spPr")
    txPr = Typed(expected_type=RichText, allow_none=True)
    textPropertes = Alias("txPr")

    __elements__ = ('layout', 'tx', 'spPr', 'txPr')

    def __init__(self,
                 layout=None,
                 tx=None,
                 spPr=None,
                 txPr=None,
                ):
        self.layout = layout
        self.tx = tx
        self.spPr = spPr
        self.txPr = txPr


class DisplayUnitsLabelList(Serialisable):

    tagname = "dispUnits"

    custUnit = NestedFloat(allow_none=True)
    builtInUnit = NestedNoneSet(values=(['hundreds', 'thousands',
                                         'tenThousands', 'hundredThousands', 'millions', 'tenMillions',
                                         'hundredMillions', 'billions', 'trillions']))
    dispUnitsLbl = Typed(expected_type=DisplayUnitsLabel, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('custUnit', 'builtInUnit', 'dispUnitsLbl',)

    def __init__(self,
                 custUnit=None,
                 builtInUnit=None,
                 dispUnitsLbl=None,
                 extLst=None,
                ):
        self.custUnit = custUnit
        self.builtInUnit = builtInUnit
        self.dispUnitsLbl = dispUnitsLbl


class NumericAxis(_BaseAxis):

    tagname = "valAx"

    axId = _BaseAxis.axId
    scaling = _BaseAxis.scaling
    delete = _BaseAxis.delete
    axPos = _BaseAxis.axPos
    majorGridlines = _BaseAxis.majorGridlines
    minorGridlines = _BaseAxis.minorGridlines
    title = _BaseAxis.title
    numFmt = _BaseAxis.numFmt
    majorTickMark = _BaseAxis.majorTickMark
    minorTickMark = _BaseAxis.minorTickMark
    tickLblPos = _BaseAxis.tickLblPos
    spPr = _BaseAxis.spPr
    txPr = _BaseAxis.txPr
    crossAx = _BaseAxis.crossAx
    crosses = _BaseAxis.crosses
    crossesAt = _BaseAxis.crossesAt

    crossBetween = NestedNoneSet(values=(['between', 'midCat']))
    majorUnit = NestedFloat(allow_none=True)
    minorUnit = NestedFloat(allow_none=True)
    dispUnits = Typed(expected_type=DisplayUnitsLabelList, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = _BaseAxis.__elements__ + ('crossBetween', 'majorUnit',
                                             'minorUnit', 'dispUnits',)


    def __init__(self,
                 crossBetween=None,
                 majorUnit=None,
                 minorUnit=None,
                 dispUnits=None,
                 extLst=None,
                 **kw
                ):
        self.crossBetween = crossBetween
        self.majorUnit = majorUnit
        self.minorUnit = minorUnit
        self.dispUnits = dispUnits
        kw.setdefault('majorGridlines', ChartLines())
        kw.setdefault('axId', 100)
        kw.setdefault('crossAx', 10)
        super().__init__(**kw)


    @classmethod
    def from_tree(cls, node):
        """
        Special case value axes with no gridlines
        """
        self = super().from_tree(node)
        gridlines = node.find("{%s}majorGridlines" % CHART_NS)
        if gridlines is None:
            self.majorGridlines = None
        return self



class TextAxis(_BaseAxis):

    tagname = "catAx"

    axId = _BaseAxis.axId
    scaling = _BaseAxis.scaling
    delete = _BaseAxis.delete
    axPos = _BaseAxis.axPos
    majorGridlines = _BaseAxis.majorGridlines
    minorGridlines = _BaseAxis.minorGridlines
    title = _BaseAxis.title
    numFmt = _BaseAxis.numFmt
    majorTickMark = _BaseAxis.majorTickMark
    minorTickMark = _BaseAxis.minorTickMark
    tickLblPos = _BaseAxis.tickLblPos
    spPr = _BaseAxis.spPr
    txPr = _BaseAxis.txPr
    crossAx = _BaseAxis.crossAx
    crosses = _BaseAxis.crosses
    crossesAt = _BaseAxis.crossesAt

    auto = NestedBool(allow_none=True)
    lblAlgn = NestedNoneSet(values=(['ctr', 'l', 'r']))
    lblOffset = NestedMinMax(min=0, max=1000)
    tickLblSkip = NestedInteger(allow_none=True)
    tickMarkSkip = NestedInteger(allow_none=True)
    noMultiLvlLbl = NestedBool(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = _BaseAxis.__elements__ + ('auto', 'lblAlgn', 'lblOffset',
                                             'tickLblSkip', 'tickMarkSkip', 'noMultiLvlLbl')

    def __init__(self,
                 auto=None,
                 lblAlgn=None,
                 lblOffset=100,
                 tickLblSkip=None,
                 tickMarkSkip=None,
                 noMultiLvlLbl=None,
                 extLst=None,
                 **kw
                ):
        self.auto = auto
        self.lblAlgn = lblAlgn
        self.lblOffset = lblOffset
        self.tickLblSkip = tickLblSkip
        self.tickMarkSkip = tickMarkSkip
        self.noMultiLvlLbl = noMultiLvlLbl
        kw.setdefault('axId', 10)
        kw.setdefault('crossAx', 100)
        super().__init__(**kw)


class DateAxis(TextAxis):

    tagname = "dateAx"

    axId = _BaseAxis.axId
    scaling = _BaseAxis.scaling
    delete = _BaseAxis.delete
    axPos = _BaseAxis.axPos
    majorGridlines = _BaseAxis.majorGridlines
    minorGridlines = _BaseAxis.minorGridlines
    title = _BaseAxis.title
    numFmt = _BaseAxis.numFmt
    majorTickMark = _BaseAxis.majorTickMark
    minorTickMark = _BaseAxis.minorTickMark
    tickLblPos = _BaseAxis.tickLblPos
    spPr = _BaseAxis.spPr
    txPr = _BaseAxis.txPr
    crossAx = _BaseAxis.crossAx
    crosses = _BaseAxis.crosses
    crossesAt = _BaseAxis.crossesAt

    auto = NestedBool(allow_none=True)
    lblOffset = NestedInteger(allow_none=True)
    baseTimeUnit = NestedNoneSet(values=(['days', 'months', 'years']))
    majorUnit = NestedFloat(allow_none=True)
    majorTimeUnit = NestedNoneSet(values=(['days', 'months', 'years']))
    minorUnit = NestedFloat(allow_none=True)
    minorTimeUnit = NestedNoneSet(values=(['days', 'months', 'years']))
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = _BaseAxis.__elements__ + ('auto', 'lblOffset',
                                             'baseTimeUnit', 'majorUnit', 'majorTimeUnit', 'minorUnit',
                                             'minorTimeUnit')

    def __init__(self,
                 auto=None,
                 lblOffset=None,
                 baseTimeUnit=None,
                 majorUnit=None,
                 majorTimeUnit=None,
                 minorUnit=None,
                 minorTimeUnit=None,
                 extLst=None,
                 **kw
                ):
        self.auto = auto
        self.lblOffset = lblOffset
        self.baseTimeUnit = baseTimeUnit
        self.majorUnit = majorUnit
        self.majorTimeUnit = majorTimeUnit
        self.minorUnit = minorUnit
        self.minorTimeUnit = minorTimeUnit
        kw.setdefault('axId', 500)
        kw.setdefault('lblOffset', lblOffset)
        super().__init__(**kw)


class SeriesAxis(_BaseAxis):

    tagname = "serAx"

    axId = _BaseAxis.axId
    scaling = _BaseAxis.scaling
    delete = _BaseAxis.delete
    axPos = _BaseAxis.axPos
    majorGridlines = _BaseAxis.majorGridlines
    minorGridlines = _BaseAxis.minorGridlines
    title = _BaseAxis.title
    numFmt = _BaseAxis.numFmt
    majorTickMark = _BaseAxis.majorTickMark
    minorTickMark = _BaseAxis.minorTickMark
    tickLblPos = _BaseAxis.tickLblPos
    spPr = _BaseAxis.spPr
    txPr = _BaseAxis.txPr
    crossAx = _BaseAxis.crossAx
    crosses = _BaseAxis.crosses
    crossesAt = _BaseAxis.crossesAt

    tickLblSkip = NestedInteger(allow_none=True)
    tickMarkSkip = NestedInteger(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = _BaseAxis.__elements__ + ('tickLblSkip', 'tickMarkSkip')

    def __init__(self,
                 tickLblSkip=None,
                 tickMarkSkip=None,
                 extLst=None,
                 **kw
                ):
        self.tickLblSkip = tickLblSkip
        self.tickMarkSkip = tickMarkSkip
        kw.setdefault('axId', 1000)
        kw.setdefault('crossAx', 10)
        super().__init__(**kw)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\deprecations.py ===
# util/deprecations.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

"""Helpers related to deprecation of functions, methods, classes, other
functionality."""

from __future__ import annotations

import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import Match
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

from . import compat
from .langhelpers import _hash_limit_string
from .langhelpers import _warnings_warn
from .langhelpers import decorator
from .langhelpers import inject_docstring_text
from .langhelpers import inject_param_text
from .. import exc

_T = TypeVar("_T", bound=Any)


# https://mypy.readthedocs.io/en/stable/generics.html#declaring-decorators
_F = TypeVar("_F", bound="Callable[..., Any]")


def _warn_with_version(
    msg: str,
    version: str,
    type_: Type[exc.SADeprecationWarning],
    stacklevel: int,
    code: Optional[str] = None,
) -> None:
    warn = type_(msg, code=code)
    warn.deprecated_since = version

    _warnings_warn(warn, stacklevel=stacklevel + 1)


def warn_deprecated(
    msg: str, version: str, stacklevel: int = 3, code: Optional[str] = None
) -> None:
    _warn_with_version(
        msg, version, exc.SADeprecationWarning, stacklevel, code=code
    )


def warn_deprecated_limited(
    msg: str,
    args: Sequence[Any],
    version: str,
    stacklevel: int = 3,
    code: Optional[str] = None,
) -> None:
    """Issue a deprecation warning with a parameterized string,
    limiting the number of registrations.

    """
    if args:
        msg = _hash_limit_string(msg, 10, args)
    _warn_with_version(
        msg, version, exc.SADeprecationWarning, stacklevel, code=code
    )


def deprecated_cls(
    version: str, message: str, constructor: Optional[str] = "__init__"
) -> Callable[[Type[_T]], Type[_T]]:
    header = ".. deprecated:: %s %s" % (version, (message or ""))

    def decorate(cls: Type[_T]) -> Type[_T]:
        return _decorate_cls_with_warning(
            cls,
            constructor,
            exc.SADeprecationWarning,
            message % dict(func=constructor),
            version,
            header,
        )

    return decorate


def deprecated(
    version: str,
    message: Optional[str] = None,
    add_deprecation_to_docstring: bool = True,
    warning: Optional[Type[exc.SADeprecationWarning]] = None,
    enable_warnings: bool = True,
) -> Callable[[_F], _F]:
    """Decorates a function and issues a deprecation warning on use.

    :param version:
      Issue version in the warning.

    :param message:
      If provided, issue message in the warning.  A sensible default
      is used if not provided.

    :param add_deprecation_to_docstring:
      Default True.  If False, the wrapped function's __doc__ is left
      as-is.  If True, the 'message' is prepended to the docs if
      provided, or sensible default if message is omitted.

    """

    if add_deprecation_to_docstring:
        header = ".. deprecated:: %s %s" % (
            version,
            (message or ""),
        )
    else:
        header = None

    if message is None:
        message = "Call to deprecated function %(func)s"

    if warning is None:
        warning = exc.SADeprecationWarning

    message += " (deprecated since: %s)" % version

    def decorate(fn: _F) -> _F:
        assert message is not None
        assert warning is not None
        return _decorate_with_warning(
            fn,
            warning,
            message % dict(func=fn.__name__),
            version,
            header,
            enable_warnings=enable_warnings,
        )

    return decorate


def moved_20(
    message: str, **kw: Any
) -> Callable[[Callable[..., _T]], Callable[..., _T]]:
    return deprecated(
        "2.0", message=message, warning=exc.MovedIn20Warning, **kw
    )


def became_legacy_20(
    api_name: str, alternative: Optional[str] = None, **kw: Any
) -> Callable[[_F], _F]:
    type_reg = re.match("^:(attr|func|meth):", api_name)
    if type_reg:
        type_ = {"attr": "attribute", "func": "function", "meth": "method"}[
            type_reg.group(1)
        ]
    else:
        type_ = "construct"
    message = (
        "The %s %s is considered legacy as of the "
        "1.x series of SQLAlchemy and %s in 2.0."
        % (
            api_name,
            type_,
            "becomes a legacy construct",
        )
    )

    if ":attr:" in api_name:
        attribute_ok = kw.pop("warn_on_attribute_access", False)
        if not attribute_ok:
            assert kw.get("enable_warnings") is False, (
                "attribute %s will emit a warning on read access.  "
                "If you *really* want this, "
                "add warn_on_attribute_access=True.  Otherwise please add "
                "enable_warnings=False." % api_name
            )

    if alternative:
        message += " " + alternative

    warning_cls = exc.LegacyAPIWarning

    return deprecated("2.0", message=message, warning=warning_cls, **kw)


def deprecated_params(**specs: Tuple[str, str]) -> Callable[[_F], _F]:
    """Decorates a function to warn on use of certain parameters.

    e.g. ::

        @deprecated_params(
            weak_identity_map=(
                "0.7",
                "the :paramref:`.Session.weak_identity_map parameter "
                "is deprecated.",
            )
        )
        def some_function(**kwargs): ...

    """

    messages: Dict[str, str] = {}
    versions: Dict[str, str] = {}
    version_warnings: Dict[str, Type[exc.SADeprecationWarning]] = {}

    for param, (version, message) in specs.items():
        versions[param] = version
        messages[param] = _sanitize_restructured_text(message)
        version_warnings[param] = exc.SADeprecationWarning

    def decorate(fn: _F) -> _F:
        spec = compat.inspect_getfullargspec(fn)

        check_defaults: Union[Set[str], Tuple[()]]
        if spec.defaults is not None:
            defaults = dict(
                zip(
                    spec.args[(len(spec.args) - len(spec.defaults)) :],
                    spec.defaults,
                )
            )
            check_defaults = set(defaults).intersection(messages)
            check_kw = set(messages).difference(defaults)
        elif spec.kwonlydefaults is not None:
            defaults = spec.kwonlydefaults
            check_defaults = set(defaults).intersection(messages)
            check_kw = set(messages).difference(defaults)
        else:
            check_defaults = ()
            check_kw = set(messages)

        check_any_kw = spec.varkw

        # latest mypy has opinions here, not sure if they implemented
        # Concatenate or something
        @decorator
        def warned(fn: _F, *args: Any, **kwargs: Any) -> _F:
            for m in check_defaults:
                if (defaults[m] is None and kwargs[m] is not None) or (
                    defaults[m] is not None and kwargs[m] != defaults[m]
                ):
                    _warn_with_version(
                        messages[m],
                        versions[m],
                        version_warnings[m],
                        stacklevel=3,
                    )

            if check_any_kw in messages and set(kwargs).difference(
                check_defaults
            ):
                assert check_any_kw is not None
                _warn_with_version(
                    messages[check_any_kw],
                    versions[check_any_kw],
                    version_warnings[check_any_kw],
                    stacklevel=3,
                )

            for m in check_kw:
                if m in kwargs:
                    _warn_with_version(
                        messages[m],
                        versions[m],
                        version_warnings[m],
                        stacklevel=3,
                    )
            return fn(*args, **kwargs)  # type: ignore[no-any-return]

        doc = fn.__doc__ is not None and fn.__doc__ or ""
        if doc:
            doc = inject_param_text(
                doc,
                {
                    param: ".. deprecated:: %s %s"
                    % ("1.4" if version == "2.0" else version, (message or ""))
                    for param, (version, message) in specs.items()
                },
            )
        decorated = warned(fn)
        decorated.__doc__ = doc
        return decorated

    return decorate


def _sanitize_restructured_text(text: str) -> str:
    def repl(m: Match[str]) -> str:
        type_, name = m.group(1, 2)
        if type_ in ("func", "meth"):
            name += "()"
        return name

    text = re.sub(r":ref:`(.+) <.*>`", lambda m: '"%s"' % m.group(1), text)
    return re.sub(r"\:(\w+)\:`~?(?:_\w+)?\.?(.+?)`", repl, text)


def _decorate_cls_with_warning(
    cls: Type[_T],
    constructor: Optional[str],
    wtype: Type[exc.SADeprecationWarning],
    message: str,
    version: str,
    docstring_header: Optional[str] = None,
) -> Type[_T]:
    doc = cls.__doc__ is not None and cls.__doc__ or ""
    if docstring_header is not None:
        if constructor is not None:
            docstring_header %= dict(func=constructor)

        if issubclass(wtype, exc.Base20DeprecationWarning):
            docstring_header += (
                " (Background on SQLAlchemy 2.0 at: "
                ":ref:`migration_20_toplevel`)"
            )
        doc = inject_docstring_text(doc, docstring_header, 1)

        constructor_fn = None
        if type(cls) is type:
            clsdict = dict(cls.__dict__)
            clsdict["__doc__"] = doc
            clsdict.pop("__dict__", None)
            clsdict.pop("__weakref__", None)
            cls = type(cls.__name__, cls.__bases__, clsdict)
            if constructor is not None:
                constructor_fn = clsdict[constructor]

        else:
            cls.__doc__ = doc
            if constructor is not None:
                constructor_fn = getattr(cls, constructor)

        if constructor is not None:
            assert constructor_fn is not None
            assert wtype is not None
            setattr(
                cls,
                constructor,
                _decorate_with_warning(
                    constructor_fn, wtype, message, version, None
                ),
            )
    return cls


def _decorate_with_warning(
    func: _F,
    wtype: Type[exc.SADeprecationWarning],
    message: str,
    version: str,
    docstring_header: Optional[str] = None,
    enable_warnings: bool = True,
) -> _F:
    """Wrap a function with a warnings.warn and augmented docstring."""

    message = _sanitize_restructured_text(message)

    if issubclass(wtype, exc.Base20DeprecationWarning):
        doc_only = (
            " (Background on SQLAlchemy 2.0 at: "
            ":ref:`migration_20_toplevel`)"
        )
    else:
        doc_only = ""

    @decorator
    def warned(fn: _F, *args: Any, **kwargs: Any) -> _F:
        skip_warning = not enable_warnings or kwargs.pop(
            "_sa_skip_warning", False
        )
        if not skip_warning:
            _warn_with_version(message, version, wtype, stacklevel=3)
        return fn(*args, **kwargs)  # type: ignore[no-any-return]

    doc = func.__doc__ is not None and func.__doc__ or ""
    if docstring_header is not None:
        docstring_header %= dict(func=func.__name__)

        docstring_header += doc_only

        doc = inject_docstring_text(doc, docstring_header, 1)

    decorated = warned(func)
    decorated.__doc__ = doc
    decorated._sa_warn = lambda: _warn_with_version(  # type: ignore
        message, version, wtype, stacklevel=3
    )
    return decorated

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\shape_optimizer.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This tool is not used directly in bert optimization. It could assist developing the optimization script on the following scenarios:
# (1) It could simplify graph by removing many sub-graphs related to reshape.
# (2) It could reduce extra inputs and outputs to fit other tools. The script compare_bert_results.py or bert_perf_test.py requires 3 inputs.

import argparse
import logging
import os
import re  # noqa: F401
import sys
import tempfile
from collections import deque  # noqa: F401
from datetime import datetime
from pathlib import Path  # noqa: F401

import numpy as np
import onnx
from onnx import ModelProto, TensorProto, numpy_helper
from onnx_model import OnnxModel

import onnxruntime

logger = logging.getLogger(__name__)

CONSTANT_SHAPE_NAME_PREFIX = "constant_shape_opt__"
RESHAPE_INPUT_SHAPE_PREFIX = "reshape_input_shape__"


class BertOnnxModelShapeOptimizer(OnnxModel):
    """
    This optimizer will replace Shape output or the shape input of Reshape node by initializer. Currently, it requires
    model inputs to have static shape.
    """

    def __init__(self, onnx_model):
        super().__init__(onnx_model.model)

    def add_shape_initializer(self, shape):
        """
        Add an initializer for constant shape.
        """
        shape_value = np.asarray(shape, dtype=np.int64)
        constant_shape_name = self.create_node_name("Constant", CONSTANT_SHAPE_NAME_PREFIX)
        tensor = onnx.helper.make_tensor(
            name=constant_shape_name,
            data_type=TensorProto.INT64,
            dims=shape_value.shape,
            vals=shape_value,
        )
        self.add_initializer(tensor)
        return tensor

    def get_shape_outputs(self):
        """
        Returns a list of output names of all Shape nodes.
        """
        input_name_to_nodes = self.input_name_to_nodes()

        outputs = []
        for node in self.model.graph.node:
            if node.op_type == "Shape":
                if node.output[0] in input_name_to_nodes:
                    outputs.append(node.output[0])

        return outputs

    def get_reshape_shape_inputs(self):
        """
        Returns a list of shape input names of Reshape nodes.
        """
        self.output_name_to_node()

        shape_inputs = []
        for node in self.model.graph.node:
            if node.op_type == "Reshape":
                shape_inputs.append(node.input[1])

        return shape_inputs

    def add_shape_for_reshape_input(self):
        """
        For each Reshape node, create a Shape node for its first input.
        Returns the output names of these Shape nodes.
        """
        output_names = []
        nodes_to_add = []
        for node in self.model.graph.node:
            if node.op_type == "Reshape":
                input = node.input[0]
                output_name = self.create_node_name("Reshape_Input", RESHAPE_INPUT_SHAPE_PREFIX)
                shape_node = onnx.helper.make_node("Shape", inputs=[input], outputs=[output_name])
                nodes_to_add.append(shape_node)
                output_names.append(output_name)

        self.add_nodes(nodes_to_add)
        return output_names

    def add_extra_graph_output(self, extra_outputs):
        """
        Add a list of output names to graph output.
        """
        names_to_evaluate = []
        output_names = [output.name for output in self.model.graph.output]
        for name in extra_outputs:
            if self.get_initializer(name) is not None:  # already a constant
                continue
            names_to_evaluate.append(name)

            if name not in output_names:
                output_info = onnx.helper.ValueInfoProto()
                output_info.name = name
                self.model.graph.output.extend([output_info])
                output_names.append(name)

        return names_to_evaluate

    # Update input and output shape to be static
    def use_static_input(self, inputs, batch_size=1, max_seq_len=128):
        """
        Update the model to use static axes instead of dynamic axes for graph inputs.
        """
        for input in self.model.graph.input:
            if input.name in inputs:
                dim_proto = input.type.tensor_type.shape.dim[0]
                dim_proto.dim_value = batch_size
                dim_proto = input.type.tensor_type.shape.dim[1]
                if dim_proto.HasField("dim_param"):
                    dim_proto.dim_value = max_seq_len
                elif dim_proto.HasField("dim_value") and dim_proto.dim_value != max_seq_len:
                    raise ValueError(
                        f"Unable to set dimension value to {max_seq_len} for axis {1} of {input.name}. Contradicts existing dimension value {dim_proto.dim_value}."
                    )

    def create_dummy_inputs(
        self,
        input_ids,
        segment_ids,
        input_mask,
        batch_size,
        sequence_length,
        elem_type,
        dictionary_size=8,
    ):
        """
        Create dummy data for model inputs. If the model has more than 3 inputs, please update this function accordingly before running the tool.
        """
        assert elem_type in [1, 6, 7]  # only int32, int64 and float32 are supported.

        # Create dummy inputs
        input_1 = np.random.randint(dictionary_size, size=(batch_size, sequence_length), dtype=np.int32)
        input_2 = np.ones((batch_size, sequence_length), dtype=np.int32)
        input_3 = np.zeros((batch_size, sequence_length), dtype=np.int32)

        # Here we assume that 3 inputs have same data type
        if elem_type == 1:  # float32
            input_1 = np.float32(input_1)
            input_2 = np.float32(input_2)
            input_3 = np.float32(input_3)
        elif elem_type == 7:  # int64
            input_1 = np.int64(input_1)
            input_2 = np.int64(input_2)
            input_3 = np.int64(input_3)

        inputs = {input_ids: input_1, input_mask: input_2, segment_ids: input_3}
        return inputs

    def shape_optimization(
        self,
        temp_model_path,
        input_ids,
        segment_ids,
        input_mask,
        output_names,
        batch_size,
        sequence_length,
        enable_shape_opt,
        enable_reshape_opt,
        verbose,
    ):
        self.bert_inputs = [input_ids, segment_ids, input_mask]

        extra_outputs = []
        if enable_shape_opt:
            extra_outputs.extend(self.get_shape_outputs())

        if enable_reshape_opt:
            reshape_shape_inputs = self.get_reshape_shape_inputs()
            reshape_input_shapes = self.add_shape_for_reshape_input()
            extra_outputs.extend(reshape_shape_inputs)
            extra_outputs.extend(reshape_input_shapes)

        if len(extra_outputs) == 0:
            return

        names_to_evaluate = self.add_extra_graph_output(extra_outputs)

        # This tool does not support dynamic axes right now.
        self.use_static_input(self.bert_inputs, batch_size, sequence_length)

        with open(temp_model_path, "wb") as out:
            out.write(self.model.SerializeToString())
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        session = onnxruntime.InferenceSession(
            temp_model_path,
            sess_options,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

        elem_type = 7
        for input in self.model.graph.input:
            if input.name == input_ids:
                elem_type = input.type.tensor_type.elem_type
        inputs = self.create_dummy_inputs(input_ids, segment_ids, input_mask, batch_size, sequence_length, elem_type)

        outputs = session.run(names_to_evaluate, inputs)
        shapes = {}
        for i, name in enumerate(names_to_evaluate):
            shapes[name] = outputs[i]

        logger.debug(f"shapes={shapes}")

        if enable_reshape_opt:
            for i, shape_input in enumerate(reshape_shape_inputs):
                input_shape = reshape_input_shapes[i]
                self.update_target_shape(shapes, shape_input, input_shape, verbose)

        for name, shape in shapes.items():
            tensor = self.add_shape_initializer(shape)
            self.replace_input_of_all_nodes(name, tensor.name)

        # Remove extra outputs, and prune all nodes not linked to output.
        self.prune_graph(output_names)

    def update_target_shape(self, shapes, shape_input, input_shape, verbose):
        """
        Update the target shape to use 0 to represent that dimension value does not change.
        For example, shape of source data is (2, 5, 8) and target shape is (2, 5, 4, 2), the target shape will be updated to (0, 0, 4, 2).
        """
        if shape_input in shapes:
            target_shape = shapes[shape_input]
        else:
            initializer = self.get_initializer(shape_input)
            assert initializer is not None
            target_shape = numpy_helper.to_array(initializer)

        if input_shape in shapes:
            source_shape = shapes[input_shape]
        else:
            initializer = self.get_initializer(input_shape)
            assert initializer is not None
            source_shape = numpy_helper.to_array(initializer)

        new_target_shape = []
        for i, dim_value in enumerate(target_shape):
            if i < len(source_shape) and source_shape[i] == dim_value:
                new_target_shape.append(0)
            else:
                new_target_shape.append(dim_value)
        shapes[shape_input] = new_target_shape

        logger.debug(f"source_shape={source_shape}, target_shape={target_shape}, new_target_shape={new_target_shape}")

    def validate_input(self, input: str):
        if not self.find_graph_input(input):
            valid_names = [input.name for input in self.model.graph.input]
            raise Exception(f"Input {input} does not exist in the graph inputs: {valid_names}")

    def validate_outputs(self, output_names: list[str]):
        valid_names = [output.name for output in self.model.graph.output]
        for name in output_names:
            if name not in valid_names:
                raise Exception(f"Output {name} does not exist in the graph outputs: {valid_names}")

    def optimize(
        self,
        output_path: str,
        input_ids: str,
        segment_ids: str,
        input_mask: str,
        enable_shape_opt: bool,
        enable_reshape_opt: bool,
        output_names: list[str] | None = None,
        batch_size=1,
        sequence_length=128,
        verbose=False,
    ):
        # Skip if shape optimization has been done before.
        for tensor in self.model.graph.initializer:
            if tensor.name.startswith(CONSTANT_SHAPE_NAME_PREFIX):
                logger.info("Skip shape optimization since it has been done before")
                return

        self.validate_input(input_ids)
        self.validate_input(segment_ids)
        self.validate_input(input_mask)

        if output_names is not None:
            self.validate_outputs(output_names)
            self.prune_graph(output_names)

        remaining_outputs = [output.name for output in self.model.graph.output]

        if enable_shape_opt or enable_reshape_opt:
            if len(self.get_graph_inputs_excluding_initializers()) != 3:
                logger.info("Skip shape optimization since graph input number is not 3")
                return

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file_name = "temp_{}.onnx".format(datetime.now().strftime("%m_%d-%H_%M_%S"))
                dir = "." if verbose else temp_dir
                temp_file = os.path.join(dir, temp_file_name)
                self.shape_optimization(
                    temp_file,
                    input_ids,
                    segment_ids,
                    input_mask,
                    remaining_outputs,
                    batch_size,
                    sequence_length,
                    enable_shape_opt,
                    enable_reshape_opt,
                    verbose,
                )
            logger.debug(f"Temp model with additional outputs: {temp_file}")
            logger.warning(
                f"Shape optimization is done. The optimized model might only work for input with batch_size={batch_size} sequence_length={sequence_length}"
            )

        if output_path is not None:
            with open(output_path, "wb") as out:
                out.write(self.model.SerializeToString())


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--input_ids", required=True, type=str)
    parser.add_argument("--segment_ids", required=True, type=str)
    parser.add_argument("--input_mask", required=True, type=str)
    parser.add_argument("--output_names", required=False, type=str, default=None)
    parser.add_argument("--batch_size", required=False, type=int, default=1)
    parser.add_argument("--sequence_length", required=False, type=int, default=128)
    parser.add_argument("--enable_shape_opt", required=False, action="store_true")
    parser.set_defaults(enable_shape_opt=False)
    parser.add_argument("--enable_reshape_opt", required=False, action="store_true")
    parser.set_defaults(enable_reshape_opt=False)
    parser.add_argument("--verbose", required=False, action="store_true")
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    return args


def setup_logging(verbose):
    log_handler = logging.StreamHandler(sys.stdout)
    if verbose:
        log_handler.setFormatter(logging.Formatter("[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s"))
        logging_level = logging.DEBUG
    else:
        log_handler.setFormatter(logging.Formatter("%(filename)20s: %(message)s"))
        logging_level = logging.INFO
    log_handler.setLevel(logging_level)
    logger.addHandler(log_handler)
    logger.setLevel(logging_level)


def main():
    args = parse_arguments()
    setup_logging(args.verbose)

    output_names = None if args.output_names is None else args.output_names.split(";")

    model = ModelProto()
    with open(args.input, "rb") as input_file:
        model.ParseFromString(input_file.read())
    onnx_model = OnnxModel(model)

    optimizer = BertOnnxModelShapeOptimizer(onnx_model)

    optimizer.optimize(
        args.output,
        args.input_ids,
        args.segment_ids,
        args.input_mask,
        args.enable_shape_opt,
        args.enable_reshape_opt,
        output_names,
        args.batch_size,
        args.sequence_length,
        args.verbose,
    )


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\prompt.py ===
from typing import Any, Generic, List, Optional, TextIO, TypeVar, Union, overload

from . import get_console
from .console import Console
from .text import Text, TextType

PromptType = TypeVar("PromptType")
DefaultType = TypeVar("DefaultType")


class PromptError(Exception):
    """Exception base class for prompt related errors."""


class InvalidResponse(PromptError):
    """Exception to indicate a response was invalid. Raise this within process_response() to indicate an error
    and provide an error message.

    Args:
        message (Union[str, Text]): Error message.
    """

    def __init__(self, message: TextType) -> None:
        self.message = message

    def __rich__(self) -> TextType:
        return self.message


class PromptBase(Generic[PromptType]):
    """Ask the user for input until a valid response is received. This is the base class, see one of
    the concrete classes for examples.

    Args:
        prompt (TextType, optional): Prompt text. Defaults to "".
        console (Console, optional): A Console instance or None to use global console. Defaults to None.
        password (bool, optional): Enable password input. Defaults to False.
        choices (List[str], optional): A list of valid choices. Defaults to None.
        case_sensitive (bool, optional): Matching of choices should be case-sensitive. Defaults to True.
        show_default (bool, optional): Show default in prompt. Defaults to True.
        show_choices (bool, optional): Show choices in prompt. Defaults to True.
    """

    response_type: type = str

    validate_error_message = "[prompt.invalid]Please enter a valid value"
    illegal_choice_message = (
        "[prompt.invalid.choice]Please select one of the available options"
    )
    prompt_suffix = ": "

    choices: Optional[List[str]] = None

    def __init__(
        self,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
    ) -> None:
        self.console = console or get_console()
        self.prompt = (
            Text.from_markup(prompt, style="prompt")
            if isinstance(prompt, str)
            else prompt
        )
        self.password = password
        if choices is not None:
            self.choices = choices
        self.case_sensitive = case_sensitive
        self.show_default = show_default
        self.show_choices = show_choices

    @classmethod
    @overload
    def ask(
        cls,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        default: DefaultType,
        stream: Optional[TextIO] = None,
    ) -> Union[DefaultType, PromptType]:
        ...

    @classmethod
    @overload
    def ask(
        cls,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        stream: Optional[TextIO] = None,
    ) -> PromptType:
        ...

    @classmethod
    def ask(
        cls,
        prompt: TextType = "",
        *,
        console: Optional[Console] = None,
        password: bool = False,
        choices: Optional[List[str]] = None,
        case_sensitive: bool = True,
        show_default: bool = True,
        show_choices: bool = True,
        default: Any = ...,
        stream: Optional[TextIO] = None,
    ) -> Any:
        """Shortcut to construct and run a prompt loop and return the result.

        Example:
            >>> filename = Prompt.ask("Enter a filename")

        Args:
            prompt (TextType, optional): Prompt text. Defaults to "".
            console (Console, optional): A Console instance or None to use global console. Defaults to None.
            password (bool, optional): Enable password input. Defaults to False.
            choices (List[str], optional): A list of valid choices. Defaults to None.
            case_sensitive (bool, optional): Matching of choices should be case-sensitive. Defaults to True.
            show_default (bool, optional): Show default in prompt. Defaults to True.
            show_choices (bool, optional): Show choices in prompt. Defaults to True.
            stream (TextIO, optional): Optional text file open for reading to get input. Defaults to None.
        """
        _prompt = cls(
            prompt,
            console=console,
            password=password,
            choices=choices,
            case_sensitive=case_sensitive,
            show_default=show_default,
            show_choices=show_choices,
        )
        return _prompt(default=default, stream=stream)

    def render_default(self, default: DefaultType) -> Text:
        """Turn the supplied default in to a Text instance.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text: Text containing rendering of default value.
        """
        return Text(f"({default})", "prompt.default")

    def make_prompt(self, default: DefaultType) -> Text:
        """Make prompt text.

        Args:
            default (DefaultType): Default value.

        Returns:
            Text: Text to display in prompt.
        """
        prompt = self.prompt.copy()
        prompt.end = ""

        if self.show_choices and self.choices:
            _choices = "/".join(self.choices)
            choices = f"[{_choices}]"
            prompt.append(" ")
            prompt.append(choices, "prompt.choices")

        if (
            default != ...
            and self.show_default
            and isinstance(default, (str, self.response_type))
        ):
            prompt.append(" ")
            _default = self.render_default(default)
            prompt.append(_default)

        prompt.append(self.prompt_suffix)

        return prompt

    @classmethod
    def get_input(
        cls,
        console: Console,
        prompt: TextType,
        password: bool,
        stream: Optional[TextIO] = None,
    ) -> str:
        """Get input from user.

        Args:
            console (Console): Console instance.
            prompt (TextType): Prompt text.
            password (bool): Enable password entry.

        Returns:
            str: String from user.
        """
        return console.input(prompt, password=password, stream=stream)

    def check_choice(self, value: str) -> bool:
        """Check value is in the list of valid choices.

        Args:
            value (str): Value entered by user.

        Returns:
            bool: True if choice was valid, otherwise False.
        """
        assert self.choices is not None
        if self.case_sensitive:
            return value.strip() in self.choices
        return value.strip().lower() in [choice.lower() for choice in self.choices]

    def process_response(self, value: str) -> PromptType:
        """Process response from user, convert to prompt type.

        Args:
            value (str): String typed by user.

        Raises:
            InvalidResponse: If ``value`` is invalid.

        Returns:
            PromptType: The value to be returned from ask method.
        """
        value = value.strip()
        try:
            return_value: PromptType = self.response_type(value)
        except ValueError:
            raise InvalidResponse(self.validate_error_message)

        if self.choices is not None:
            if not self.check_choice(value):
                raise InvalidResponse(self.illegal_choice_message)

            if not self.case_sensitive:
                # return the original choice, not the lower case version
                return_value = self.response_type(
                    self.choices[
                        [choice.lower() for choice in self.choices].index(value.lower())
                    ]
                )
        return return_value

    def on_validate_error(self, value: str, error: InvalidResponse) -> None:
        """Called to handle validation error.

        Args:
            value (str): String entered by user.
            error (InvalidResponse): Exception instance the initiated the error.
        """
        self.console.print(error)

    def pre_prompt(self) -> None:
        """Hook to display something before the prompt."""

    @overload
    def __call__(self, *, stream: Optional[TextIO] = None) -> PromptType:
        ...

    @overload
    def __call__(
        self, *, default: DefaultType, stream: Optional[TextIO] = None
    ) -> Union[PromptType, DefaultType]:
        ...

    def __call__(self, *, default: Any = ..., stream: Optional[TextIO] = None) -> Any:
        """Run the prompt loop.

        Args:
            default (Any, optional): Optional default value.

        Returns:
            PromptType: Processed value.
        """
        while True:
            self.pre_prompt()
            prompt = self.make_prompt(default)
            value = self.get_input(self.console, prompt, self.password, stream=stream)
            if value == "" and default != ...:
                return default
            try:
                return_value = self.process_response(value)
            except InvalidResponse as error:
                self.on_validate_error(value, error)
                continue
            else:
                return return_value


class Prompt(PromptBase[str]):
    """A prompt that returns a str.

    Example:
        >>> name = Prompt.ask("Enter your name")


    """

    response_type = str


class IntPrompt(PromptBase[int]):
    """A prompt that returns an integer.

    Example:
        >>> burrito_count = IntPrompt.ask("How many burritos do you want to order")

    """

    response_type = int
    validate_error_message = "[prompt.invalid]Please enter a valid integer number"


class FloatPrompt(PromptBase[float]):
    """A prompt that returns a float.

    Example:
        >>> temperature = FloatPrompt.ask("Enter desired temperature")

    """

    response_type = float
    validate_error_message = "[prompt.invalid]Please enter a number"


class Confirm(PromptBase[bool]):
    """A yes / no confirmation prompt.

    Example:
        >>> if Confirm.ask("Continue"):
                run_job()

    """

    response_type = bool
    validate_error_message = "[prompt.invalid]Please enter Y or N"
    choices: List[str] = ["y", "n"]

    def render_default(self, default: DefaultType) -> Text:
        """Render the default as (y) or (n) rather than True/False."""
        yes, no = self.choices
        return Text(f"({yes})" if default else f"({no})", style="prompt.default")

    def process_response(self, value: str) -> bool:
        """Convert choices to a bool."""
        value = value.strip().lower()
        if value not in self.choices:
            raise InvalidResponse(self.validate_error_message)
        return value == self.choices[0]


if __name__ == "__main__":  # pragma: no cover
    from pip._vendor.rich import print

    if Confirm.ask("Run [i]prompt[/i] tests?", default=True):
        while True:
            result = IntPrompt.ask(
                ":rocket: Enter a number between [b]1[/b] and [b]10[/b]", default=5
            )
            if result >= 1 and result <= 10:
                break
            print(":pile_of_poo: [prompt.invalid]Number must be between 1 and 10")
        print(f"number={result}")

        while True:
            password = Prompt.ask(
                "Please enter a password [cyan](must be at least 5 characters)",
                password=True,
            )
            if len(password) >= 5:
                break
            print("[prompt.invalid]password too short")
        print(f"password={password!r}")

        fruit = Prompt.ask("Enter a fruit", choices=["apple", "orange", "pear"])
        print(f"fruit={fruit!r}")

        doggie = Prompt.ask(
            "What's the best Dog? (Case INSENSITIVE)",
            choices=["Border Terrier", "Collie", "Labradoodle"],
            case_sensitive=False,
        )
        print(f"doggie={doggie!r}")

    else:
        print("[b]OK :loudly_crying_face:")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\row.py ===
# engine/row.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Define row constructs including :class:`.Row`."""

from __future__ import annotations

from abc import ABC
import collections.abc as collections_abc
import operator
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import Iterator
from typing import List
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from ..sql import util as sql_util
from ..util import deprecated
from ..util._has_cy import HAS_CYEXTENSION

if TYPE_CHECKING or not HAS_CYEXTENSION:
    from ._py_row import BaseRow as BaseRow
else:
    from sqlalchemy.cyextension.resultproxy import BaseRow as BaseRow

if TYPE_CHECKING:
    from .result import _KeyType
    from .result import _ProcessorsType
    from .result import RMKeyView

_T = TypeVar("_T", bound=Any)
_TP = TypeVar("_TP", bound=Tuple[Any, ...])


class Row(BaseRow, Sequence[Any], Generic[_TP]):
    """Represent a single result row.

    The :class:`.Row` object represents a row of a database result.  It is
    typically associated in the 1.x series of SQLAlchemy with the
    :class:`_engine.CursorResult` object, however is also used by the ORM for
    tuple-like results as of SQLAlchemy 1.4.

    The :class:`.Row` object seeks to act as much like a Python named
    tuple as possible.   For mapping (i.e. dictionary) behavior on a row,
    such as testing for containment of keys, refer to the :attr:`.Row._mapping`
    attribute.

    .. seealso::

        :ref:`tutorial_selecting_data` - includes examples of selecting
        rows from SELECT statements.

    .. versionchanged:: 1.4

        Renamed ``RowProxy`` to :class:`.Row`. :class:`.Row` is no longer a
        "proxy" object in that it contains the final form of data within it,
        and now acts mostly like a named tuple. Mapping-like functionality is
        moved to the :attr:`.Row._mapping` attribute. See
        :ref:`change_4710_core` for background on this change.

    """

    __slots__ = ()

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise AttributeError("can't set attribute")

    def __delattr__(self, name: str) -> NoReturn:
        raise AttributeError("can't delete attribute")

    def _tuple(self) -> _TP:
        """Return a 'tuple' form of this :class:`.Row`.

        At runtime, this method returns "self"; the :class:`.Row` object is
        already a named tuple. However, at the typing level, if this
        :class:`.Row` is typed, the "tuple" return type will be a :pep:`484`
        ``Tuple`` datatype that contains typing information about individual
        elements, supporting typed unpacking and attribute access.

        .. versionadded:: 2.0.19 - The :meth:`.Row._tuple` method supersedes
           the previous :meth:`.Row.tuple` method, which is now underscored
           to avoid name conflicts with column names in the same way as other
           named-tuple methods on :class:`.Row`.

        .. seealso::

            :attr:`.Row._t` - shorthand attribute notation

            :meth:`.Result.tuples`


        """
        return self  # type: ignore

    @deprecated(
        "2.0.19",
        "The :meth:`.Row.tuple` method is deprecated in favor of "
        ":meth:`.Row._tuple`; all :class:`.Row` "
        "methods and library-level attributes are intended to be underscored "
        "to avoid name conflicts.  Please use :meth:`Row._tuple`.",
    )
    def tuple(self) -> _TP:
        """Return a 'tuple' form of this :class:`.Row`.

        .. versionadded:: 2.0

        """
        return self._tuple()

    @property
    def _t(self) -> _TP:
        """A synonym for :meth:`.Row._tuple`.

        .. versionadded:: 2.0.19 - The :attr:`.Row._t` attribute supersedes
           the previous :attr:`.Row.t` attribute, which is now underscored
           to avoid name conflicts with column names in the same way as other
           named-tuple methods on :class:`.Row`.

        .. seealso::

            :attr:`.Result.t`
        """
        return self  # type: ignore

    @property
    @deprecated(
        "2.0.19",
        "The :attr:`.Row.t` attribute is deprecated in favor of "
        ":attr:`.Row._t`; all :class:`.Row` "
        "methods and library-level attributes are intended to be underscored "
        "to avoid name conflicts.  Please use :attr:`Row._t`.",
    )
    def t(self) -> _TP:
        """A synonym for :meth:`.Row._tuple`.

        .. versionadded:: 2.0

        """
        return self._t

    @property
    def _mapping(self) -> RowMapping:
        """Return a :class:`.RowMapping` for this :class:`.Row`.

        This object provides a consistent Python mapping (i.e. dictionary)
        interface for the data contained within the row.   The :class:`.Row`
        by itself behaves like a named tuple.

        .. seealso::

            :attr:`.Row._fields`

        .. versionadded:: 1.4

        """
        return RowMapping(self._parent, None, self._key_to_index, self._data)

    def _filter_on_values(
        self, processor: Optional[_ProcessorsType]
    ) -> Row[Any]:
        return Row(self._parent, processor, self._key_to_index, self._data)

    if not TYPE_CHECKING:

        def _special_name_accessor(name: str) -> Any:
            """Handle ambiguous names such as "count" and "index" """

            @property
            def go(self: Row) -> Any:
                if self._parent._has_key(name):
                    return self.__getattr__(name)
                else:

                    def meth(*arg: Any, **kw: Any) -> Any:
                        return getattr(collections_abc.Sequence, name)(
                            self, *arg, **kw
                        )

                    return meth

            return go

        count = _special_name_accessor("count")
        index = _special_name_accessor("index")

    def __contains__(self, key: Any) -> bool:
        return key in self._data

    def _op(self, other: Any, op: Callable[[Any, Any], bool]) -> bool:
        return (
            op(self._to_tuple_instance(), other._to_tuple_instance())
            if isinstance(other, Row)
            else op(self._to_tuple_instance(), other)
        )

    __hash__ = BaseRow.__hash__

    if TYPE_CHECKING:

        @overload
        def __getitem__(self, index: int) -> Any: ...

        @overload
        def __getitem__(self, index: slice) -> Sequence[Any]: ...

        def __getitem__(self, index: Union[int, slice]) -> Any: ...

    def __lt__(self, other: Any) -> bool:
        return self._op(other, operator.lt)

    def __le__(self, other: Any) -> bool:
        return self._op(other, operator.le)

    def __ge__(self, other: Any) -> bool:
        return self._op(other, operator.ge)

    def __gt__(self, other: Any) -> bool:
        return self._op(other, operator.gt)

    def __eq__(self, other: Any) -> bool:
        return self._op(other, operator.eq)

    def __ne__(self, other: Any) -> bool:
        return self._op(other, operator.ne)

    def __repr__(self) -> str:
        return repr(sql_util._repr_row(self))

    @property
    def _fields(self) -> Tuple[str, ...]:
        """Return a tuple of string keys as represented by this
        :class:`.Row`.

        The keys can represent the labels of the columns returned by a core
        statement or the names of the orm classes returned by an orm
        execution.

        This attribute is analogous to the Python named tuple ``._fields``
        attribute.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`.Row._mapping`

        """
        return tuple([k for k in self._parent.keys if k is not None])

    def _asdict(self) -> Dict[str, Any]:
        """Return a new dict which maps field names to their corresponding
        values.

        This method is analogous to the Python named tuple ``._asdict()``
        method, and works by applying the ``dict()`` constructor to the
        :attr:`.Row._mapping` attribute.

        .. versionadded:: 1.4

        .. seealso::

            :attr:`.Row._mapping`

        """
        return dict(self._mapping)


BaseRowProxy = BaseRow
RowProxy = Row


class ROMappingView(ABC):
    __slots__ = ()

    _items: Sequence[Any]
    _mapping: Mapping["_KeyType", Any]

    def __init__(
        self, mapping: Mapping["_KeyType", Any], items: Sequence[Any]
    ):
        self._mapping = mapping  # type: ignore[misc]
        self._items = items  # type: ignore[misc]

    def __len__(self) -> int:
        return len(self._items)

    def __repr__(self) -> str:
        return "{0.__class__.__name__}({0._mapping!r})".format(self)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._items)

    def __contains__(self, item: Any) -> bool:
        return item in self._items

    def __eq__(self, other: Any) -> bool:
        return list(other) == list(self)

    def __ne__(self, other: Any) -> bool:
        return list(other) != list(self)


class ROMappingKeysValuesView(
    ROMappingView, typing.KeysView["_KeyType"], typing.ValuesView[Any]
):
    __slots__ = ("_items",)  # mapping slot is provided by KeysView


class ROMappingItemsView(ROMappingView, typing.ItemsView["_KeyType", Any]):
    __slots__ = ("_items",)  # mapping slot is provided by ItemsView


class RowMapping(BaseRow, typing.Mapping["_KeyType", Any]):
    """A ``Mapping`` that maps column names and objects to :class:`.Row`
    values.

    The :class:`.RowMapping` is available from a :class:`.Row` via the
    :attr:`.Row._mapping` attribute, as well as from the iterable interface
    provided by the :class:`.MappingResult` object returned by the
    :meth:`_engine.Result.mappings` method.

    :class:`.RowMapping` supplies Python mapping (i.e. dictionary) access to
    the  contents of the row.   This includes support for testing of
    containment of specific keys (string column names or objects), as well
    as iteration of keys, values, and items::

        for row in result:
            if "a" in row._mapping:
                print("Column 'a': %s" % row._mapping["a"])

            print("Column b: %s" % row._mapping[table.c.b])

    .. versionadded:: 1.4 The :class:`.RowMapping` object replaces the
       mapping-like access previously provided by a database result row,
       which now seeks to behave mostly like a named tuple.

    """

    __slots__ = ()

    if TYPE_CHECKING:

        def __getitem__(self, key: _KeyType) -> Any: ...

    else:
        __getitem__ = BaseRow._get_by_key_impl_mapping

    def _values_impl(self) -> List[Any]:
        return list(self._data)

    def __iter__(self) -> Iterator[str]:
        return (k for k in self._parent.keys if k is not None)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return self._parent._has_key(key)

    def __repr__(self) -> str:
        return repr(dict(self))

    def items(self) -> ROMappingItemsView:
        """Return a view of key/value tuples for the elements in the
        underlying :class:`.Row`.

        """
        return ROMappingItemsView(
            self, [(key, self[key]) for key in self.keys()]
        )

    def keys(self) -> RMKeyView:
        """Return a view of 'keys' for string column names represented
        by the underlying :class:`.Row`.

        """

        return self._parent.keys

    def values(self) -> ROMappingKeysValuesView:
        """Return a view of values for the values represented in the
        underlying :class:`.Row`.

        """
        return ROMappingKeysValuesView(self, self._values_impl())

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\plot_mode.py ===
from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence


class PlotMode(PlotObject):
    """
    Grandparent class for plotting
    modes. Serves as interface for
    registration, lookup, and init
    of modes.

    To create a new plot mode,
    inherit from PlotModeBase
    or one of its children, such
    as PlotSurface or PlotCurve.
    """

    ## Class-level attributes
    ## used to register and lookup
    ## plot modes. See PlotModeBase
    ## for descriptions and usage.

    i_vars, d_vars = '', ''
    intervals = []
    aliases = []
    is_default = False

    ## Draw is the only method here which
    ## is meant to be overridden in child
    ## classes, and PlotModeBase provides
    ## a base implementation.
    def draw(self):
        raise NotImplementedError()

    ## Everything else in this file has to
    ## do with registration and retrieval
    ## of plot modes. This is where I've
    ## hidden much of the ugliness of automatic
    ## plot mode divination...

    ## Plot mode registry data structures
    _mode_alias_list = []
    _mode_map = {
        1: {1: {}, 2: {}},
        2: {1: {}, 2: {}},
        3: {1: {}, 2: {}},
    }  # [d][i][alias_str]: class
    _mode_default_map = {
        1: {},
        2: {},
        3: {},
    }  # [d][i]: class
    _i_var_max, _d_var_max = 2, 3

    def __new__(cls, *args, **kwargs):
        """
        This is the function which interprets
        arguments given to Plot.__init__ and
        Plot.__setattr__. Returns an initialized
        instance of the appropriate child class.
        """

        newargs, newkwargs = PlotMode._extract_options(args, kwargs)
        mode_arg = newkwargs.get('mode', '')

        # Interpret the arguments
        d_vars, intervals = PlotMode._interpret_args(newargs)
        i_vars = PlotMode._find_i_vars(d_vars, intervals)
        i, d = max([len(i_vars), len(intervals)]), len(d_vars)

        # Find the appropriate mode
        subcls = PlotMode._get_mode(mode_arg, i, d)

        # Create the object
        o = object.__new__(subcls)

        # Do some setup for the mode instance
        o.d_vars = d_vars
        o._fill_i_vars(i_vars)
        o._fill_intervals(intervals)
        o.options = newkwargs

        return o

    @staticmethod
    def _get_mode(mode_arg, i_var_count, d_var_count):
        """
        Tries to return an appropriate mode class.
        Intended to be called only by __new__.

        mode_arg
            Can be a string or a class. If it is a
            PlotMode subclass, it is simply returned.
            If it is a string, it can an alias for
            a mode or an empty string. In the latter
            case, we try to find a default mode for
            the i_var_count and d_var_count.

        i_var_count
            The number of independent variables
            needed to evaluate the d_vars.

        d_var_count
            The number of dependent variables;
            usually the number of functions to
            be evaluated in plotting.

        For example, a Cartesian function y = f(x) has
        one i_var (x) and one d_var (y). A parametric
        form x,y,z = f(u,v), f(u,v), f(u,v) has two
        two i_vars (u,v) and three d_vars (x,y,z).
        """
        # if the mode_arg is simply a PlotMode class,
        # check that the mode supports the numbers
        # of independent and dependent vars, then
        # return it
        try:
            m = None
            if issubclass(mode_arg, PlotMode):
                m = mode_arg
        except TypeError:
            pass
        if m:
            if not m._was_initialized:
                raise ValueError(("To use unregistered plot mode %s "
                                  "you must first call %s._init_mode().")
                                 % (m.__name__, m.__name__))
            if d_var_count != m.d_var_count:
                raise ValueError(("%s can only plot functions "
                                  "with %i dependent variables.")
                                 % (m.__name__,
                                     m.d_var_count))
            if i_var_count > m.i_var_count:
                raise ValueError(("%s cannot plot functions "
                                  "with more than %i independent "
                                  "variables.")
                                 % (m.__name__,
                                     m.i_var_count))
            return m
        # If it is a string, there are two possibilities.
        if isinstance(mode_arg, str):
            i, d = i_var_count, d_var_count
            if i > PlotMode._i_var_max:
                raise ValueError(var_count_error(True, True))
            if d > PlotMode._d_var_max:
                raise ValueError(var_count_error(False, True))
            # If the string is '', try to find a suitable
            # default mode
            if not mode_arg:
                return PlotMode._get_default_mode(i, d)
            # Otherwise, interpret the string as a mode
            # alias (e.g. 'cartesian', 'parametric', etc)
            else:
                return PlotMode._get_aliased_mode(mode_arg, i, d)
        else:
            raise ValueError("PlotMode argument must be "
                             "a class or a string")

    @staticmethod
    def _get_default_mode(i, d, i_vars=-1):
        if i_vars == -1:
            i_vars = i
        try:
            return PlotMode._mode_default_map[d][i]
        except KeyError:
            # Keep looking for modes in higher i var counts
            # which support the given d var count until we
            # reach the max i_var count.
            if i < PlotMode._i_var_max:
                return PlotMode._get_default_mode(i + 1, d, i_vars)
            else:
                raise ValueError(("Couldn't find a default mode "
                                  "for %i independent and %i "
                                  "dependent variables.") % (i_vars, d))

    @staticmethod
    def _get_aliased_mode(alias, i, d, i_vars=-1):
        if i_vars == -1:
            i_vars = i
        if alias not in PlotMode._mode_alias_list:
            raise ValueError(("Couldn't find a mode called"
                              " %s. Known modes: %s.")
                             % (alias, ", ".join(PlotMode._mode_alias_list)))
        try:
            return PlotMode._mode_map[d][i][alias]
        except TypeError:
            # Keep looking for modes in higher i var counts
            # which support the given d var count and alias
            # until we reach the max i_var count.
            if i < PlotMode._i_var_max:
                return PlotMode._get_aliased_mode(alias, i + 1, d, i_vars)
            else:
                raise ValueError(("Couldn't find a %s mode "
                                  "for %i independent and %i "
                                  "dependent variables.")
                                 % (alias, i_vars, d))

    @classmethod
    def _register(cls):
        """
        Called once for each user-usable plot mode.
        For Cartesian2D, it is invoked after the
        class definition: Cartesian2D._register()
        """
        name = cls.__name__
        cls._init_mode()

        try:
            i, d = cls.i_var_count, cls.d_var_count
            # Add the mode to _mode_map under all
            # given aliases
            for a in cls.aliases:
                if a not in PlotMode._mode_alias_list:
                    # Also track valid aliases, so
                    # we can quickly know when given
                    # an invalid one in _get_mode.
                    PlotMode._mode_alias_list.append(a)
                PlotMode._mode_map[d][i][a] = cls
            if cls.is_default:
                # If this mode was marked as the
                # default for this d,i combination,
                # also set that.
                PlotMode._mode_default_map[d][i] = cls

        except Exception as e:
            raise RuntimeError(("Failed to register "
                              "plot mode %s. Reason: %s")
                               % (name, (str(e))))

    @classmethod
    def _init_mode(cls):
        """
        Initializes the plot mode based on
        the 'mode-specific parameters' above.
        Only intended to be called by
        PlotMode._register(). To use a mode without
        registering it, you can directly call
        ModeSubclass._init_mode().
        """
        def symbols_list(symbol_str):
            return [Symbol(s) for s in symbol_str]

        # Convert the vars strs into
        # lists of symbols.
        cls.i_vars = symbols_list(cls.i_vars)
        cls.d_vars = symbols_list(cls.d_vars)

        # Var count is used often, calculate
        # it once here
        cls.i_var_count = len(cls.i_vars)
        cls.d_var_count = len(cls.d_vars)

        if cls.i_var_count > PlotMode._i_var_max:
            raise ValueError(var_count_error(True, False))
        if cls.d_var_count > PlotMode._d_var_max:
            raise ValueError(var_count_error(False, False))

        # Try to use first alias as primary_alias
        if len(cls.aliases) > 0:
            cls.primary_alias = cls.aliases[0]
        else:
            cls.primary_alias = cls.__name__

        di = cls.intervals
        if len(di) != cls.i_var_count:
            raise ValueError("Plot mode must provide a "
                             "default interval for each i_var.")
        for i in range(cls.i_var_count):
            # default intervals must be given [min,max,steps]
            # (no var, but they must be in the same order as i_vars)
            if len(di[i]) != 3:
                raise ValueError("length should be equal to 3")

            # Initialize an incomplete interval,
            # to later be filled with a var when
            # the mode is instantiated.
            di[i] = PlotInterval(None, *di[i])

        # To prevent people from using modes
        # without these required fields set up.
        cls._was_initialized = True

    _was_initialized = False

    ## Initializer Helper Methods

    @staticmethod
    def _find_i_vars(functions, intervals):
        i_vars = []

        # First, collect i_vars in the
        # order they are given in any
        # intervals.
        for i in intervals:
            if i.v is None:
                continue
            elif i.v in i_vars:
                raise ValueError(("Multiple intervals given "
                                  "for %s.") % (str(i.v)))
            i_vars.append(i.v)

        # Then, find any remaining
        # i_vars in given functions
        # (aka d_vars)
        for f in functions:
            for a in f.free_symbols:
                if a not in i_vars:
                    i_vars.append(a)

        return i_vars

    def _fill_i_vars(self, i_vars):
        # copy default i_vars
        self.i_vars = [Symbol(str(i)) for i in self.i_vars]
        # replace with given i_vars
        for i in range(len(i_vars)):
            self.i_vars[i] = i_vars[i]

    def _fill_intervals(self, intervals):
        # copy default intervals
        self.intervals = [PlotInterval(i) for i in self.intervals]
        # track i_vars used so far
        v_used = []
        # fill copy of default
        # intervals with given info
        for i in range(len(intervals)):
            self.intervals[i].fill_from(intervals[i])
            if self.intervals[i].v is not None:
                v_used.append(self.intervals[i].v)
        # Find any orphan intervals and
        # assign them i_vars
        for i in range(len(self.intervals)):
            if self.intervals[i].v is None:
                u = [v for v in self.i_vars if v not in v_used]
                if len(u) == 0:
                    raise ValueError("length should not be equal to 0")
                self.intervals[i].v = u[0]
                v_used.append(u[0])

    @staticmethod
    def _interpret_args(args):
        interval_wrong_order = "PlotInterval %s was given before any function(s)."
        interpret_error = "Could not interpret %s as a function or interval."

        functions, intervals = [], []
        if isinstance(args[0], GeometryEntity):
            for coords in list(args[0].arbitrary_point()):
                functions.append(coords)
            intervals.append(PlotInterval.try_parse(args[0].plot_interval()))
        else:
            for a in args:
                i = PlotInterval.try_parse(a)
                if i is not None:
                    if len(functions) == 0:
                        raise ValueError(interval_wrong_order % (str(i)))
                    else:
                        intervals.append(i)
                else:
                    if is_sequence(a, include=str):
                        raise ValueError(interpret_error % (str(a)))
                    try:
                        f = sympify(a)
                        functions.append(f)
                    except TypeError:
                        raise ValueError(interpret_error % str(a))

        return functions, intervals

    @staticmethod
    def _extract_options(args, kwargs):
        newkwargs, newargs = {}, []
        for a in args:
            if isinstance(a, str):
                newkwargs = dict(newkwargs, **parse_option_string(a))
            else:
                newargs.append(a)
        newkwargs = dict(newkwargs, **kwargs)
        return newargs, newkwargs


def var_count_error(is_independent, is_plotting):
    """
    Used to format an error message which differs
    slightly in 4 places.
    """
    if is_plotting:
        v = "Plotting"
    else:
        v = "Registering plot modes"
    if is_independent:
        n, s = PlotMode._i_var_max, "independent"
    else:
        n, s = PlotMode._d_var_max, "dependent"
    return ("%s with more than %i %s variables "
            "is not supported.") % (v, n, s)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\flask\sessions.py ===
from __future__ import annotations

import collections.abc as c
import hashlib
import typing as t
from collections.abc import MutableMapping
from datetime import datetime
from datetime import timezone

from itsdangerous import BadSignature
from itsdangerous import URLSafeTimedSerializer
from werkzeug.datastructures import CallbackDict

from .json.tag import TaggedJSONSerializer

if t.TYPE_CHECKING:  # pragma: no cover
    import typing_extensions as te

    from .app import Flask
    from .wrappers import Request
    from .wrappers import Response


class SessionMixin(MutableMapping[str, t.Any]):
    """Expands a basic dictionary with session attributes."""

    @property
    def permanent(self) -> bool:
        """This reflects the ``'_permanent'`` key in the dict."""
        return self.get("_permanent", False)

    @permanent.setter
    def permanent(self, value: bool) -> None:
        self["_permanent"] = bool(value)

    #: Some implementations can detect whether a session is newly
    #: created, but that is not guaranteed. Use with caution. The mixin
    # default is hard-coded ``False``.
    new = False

    #: Some implementations can detect changes to the session and set
    #: this when that happens. The mixin default is hard coded to
    #: ``True``.
    modified = True

    #: Some implementations can detect when session data is read or
    #: written and set this when that happens. The mixin default is hard
    #: coded to ``True``.
    accessed = True


class SecureCookieSession(CallbackDict[str, t.Any], SessionMixin):
    """Base class for sessions based on signed cookies.

    This session backend will set the :attr:`modified` and
    :attr:`accessed` attributes. It cannot reliably track whether a
    session is new (vs. empty), so :attr:`new` remains hard coded to
    ``False``.
    """

    #: When data is changed, this is set to ``True``. Only the session
    #: dictionary itself is tracked; if the session contains mutable
    #: data (for example a nested dict) then this must be set to
    #: ``True`` manually when modifying that data. The session cookie
    #: will only be written to the response if this is ``True``.
    modified = False

    #: When data is read or written, this is set to ``True``. Used by
    # :class:`.SecureCookieSessionInterface` to add a ``Vary: Cookie``
    #: header, which allows caching proxies to cache different pages for
    #: different users.
    accessed = False

    def __init__(
        self,
        initial: c.Mapping[str, t.Any] | c.Iterable[tuple[str, t.Any]] | None = None,
    ) -> None:
        def on_update(self: te.Self) -> None:
            self.modified = True
            self.accessed = True

        super().__init__(initial, on_update)

    def __getitem__(self, key: str) -> t.Any:
        self.accessed = True
        return super().__getitem__(key)

    def get(self, key: str, default: t.Any = None) -> t.Any:
        self.accessed = True
        return super().get(key, default)

    def setdefault(self, key: str, default: t.Any = None) -> t.Any:
        self.accessed = True
        return super().setdefault(key, default)


class NullSession(SecureCookieSession):
    """Class used to generate nicer error messages if sessions are not
    available.  Will still allow read-only access to the empty session
    but fail on setting.
    """

    def _fail(self, *args: t.Any, **kwargs: t.Any) -> t.NoReturn:
        raise RuntimeError(
            "The session is unavailable because no secret "
            "key was set.  Set the secret_key on the "
            "application to something unique and secret."
        )

    __setitem__ = __delitem__ = clear = pop = popitem = update = setdefault = _fail  # type: ignore # noqa: B950
    del _fail


class SessionInterface:
    """The basic interface you have to implement in order to replace the
    default session interface which uses werkzeug's securecookie
    implementation.  The only methods you have to implement are
    :meth:`open_session` and :meth:`save_session`, the others have
    useful defaults which you don't need to change.

    The session object returned by the :meth:`open_session` method has to
    provide a dictionary like interface plus the properties and methods
    from the :class:`SessionMixin`.  We recommend just subclassing a dict
    and adding that mixin::

        class Session(dict, SessionMixin):
            pass

    If :meth:`open_session` returns ``None`` Flask will call into
    :meth:`make_null_session` to create a session that acts as replacement
    if the session support cannot work because some requirement is not
    fulfilled.  The default :class:`NullSession` class that is created
    will complain that the secret key was not set.

    To replace the session interface on an application all you have to do
    is to assign :attr:`flask.Flask.session_interface`::

        app = Flask(__name__)
        app.session_interface = MySessionInterface()

    Multiple requests with the same session may be sent and handled
    concurrently. When implementing a new session interface, consider
    whether reads or writes to the backing store must be synchronized.
    There is no guarantee on the order in which the session for each
    request is opened or saved, it will occur in the order that requests
    begin and end processing.

    .. versionadded:: 0.8
    """

    #: :meth:`make_null_session` will look here for the class that should
    #: be created when a null session is requested.  Likewise the
    #: :meth:`is_null_session` method will perform a typecheck against
    #: this type.
    null_session_class = NullSession

    #: A flag that indicates if the session interface is pickle based.
    #: This can be used by Flask extensions to make a decision in regards
    #: to how to deal with the session object.
    #:
    #: .. versionadded:: 0.10
    pickle_based = False

    def make_null_session(self, app: Flask) -> NullSession:
        """Creates a null session which acts as a replacement object if the
        real session support could not be loaded due to a configuration
        error.  This mainly aids the user experience because the job of the
        null session is to still support lookup without complaining but
        modifications are answered with a helpful error message of what
        failed.

        This creates an instance of :attr:`null_session_class` by default.
        """
        return self.null_session_class()

    def is_null_session(self, obj: object) -> bool:
        """Checks if a given object is a null session.  Null sessions are
        not asked to be saved.

        This checks if the object is an instance of :attr:`null_session_class`
        by default.
        """
        return isinstance(obj, self.null_session_class)

    def get_cookie_name(self, app: Flask) -> str:
        """The name of the session cookie. Uses``app.config["SESSION_COOKIE_NAME"]``."""
        return app.config["SESSION_COOKIE_NAME"]  # type: ignore[no-any-return]

    def get_cookie_domain(self, app: Flask) -> str | None:
        """The value of the ``Domain`` parameter on the session cookie. If not set,
        browsers will only send the cookie to the exact domain it was set from.
        Otherwise, they will send it to any subdomain of the given value as well.

        Uses the :data:`SESSION_COOKIE_DOMAIN` config.

        .. versionchanged:: 2.3
            Not set by default, does not fall back to ``SERVER_NAME``.
        """
        return app.config["SESSION_COOKIE_DOMAIN"]  # type: ignore[no-any-return]

    def get_cookie_path(self, app: Flask) -> str:
        """Returns the path for which the cookie should be valid.  The
        default implementation uses the value from the ``SESSION_COOKIE_PATH``
        config var if it's set, and falls back to ``APPLICATION_ROOT`` or
        uses ``/`` if it's ``None``.
        """
        return app.config["SESSION_COOKIE_PATH"] or app.config["APPLICATION_ROOT"]  # type: ignore[no-any-return]

    def get_cookie_httponly(self, app: Flask) -> bool:
        """Returns True if the session cookie should be httponly.  This
        currently just returns the value of the ``SESSION_COOKIE_HTTPONLY``
        config var.
        """
        return app.config["SESSION_COOKIE_HTTPONLY"]  # type: ignore[no-any-return]

    def get_cookie_secure(self, app: Flask) -> bool:
        """Returns True if the cookie should be secure.  This currently
        just returns the value of the ``SESSION_COOKIE_SECURE`` setting.
        """
        return app.config["SESSION_COOKIE_SECURE"]  # type: ignore[no-any-return]

    def get_cookie_samesite(self, app: Flask) -> str | None:
        """Return ``'Strict'`` or ``'Lax'`` if the cookie should use the
        ``SameSite`` attribute. This currently just returns the value of
        the :data:`SESSION_COOKIE_SAMESITE` setting.
        """
        return app.config["SESSION_COOKIE_SAMESITE"]  # type: ignore[no-any-return]

    def get_cookie_partitioned(self, app: Flask) -> bool:
        """Returns True if the cookie should be partitioned. By default, uses
        the value of :data:`SESSION_COOKIE_PARTITIONED`.

        .. versionadded:: 3.1
        """
        return app.config["SESSION_COOKIE_PARTITIONED"]  # type: ignore[no-any-return]

    def get_expiration_time(self, app: Flask, session: SessionMixin) -> datetime | None:
        """A helper method that returns an expiration date for the session
        or ``None`` if the session is linked to the browser session.  The
        default implementation returns now + the permanent session
        lifetime configured on the application.
        """
        if session.permanent:
            return datetime.now(timezone.utc) + app.permanent_session_lifetime
        return None

    def should_set_cookie(self, app: Flask, session: SessionMixin) -> bool:
        """Used by session backends to determine if a ``Set-Cookie`` header
        should be set for this session cookie for this response. If the session
        has been modified, the cookie is set. If the session is permanent and
        the ``SESSION_REFRESH_EACH_REQUEST`` config is true, the cookie is
        always set.

        This check is usually skipped if the session was deleted.

        .. versionadded:: 0.11
        """

        return session.modified or (
            session.permanent and app.config["SESSION_REFRESH_EACH_REQUEST"]
        )

    def open_session(self, app: Flask, request: Request) -> SessionMixin | None:
        """This is called at the beginning of each request, after
        pushing the request context, before matching the URL.

        This must return an object which implements a dictionary-like
        interface as well as the :class:`SessionMixin` interface.

        This will return ``None`` to indicate that loading failed in
        some way that is not immediately an error. The request
        context will fall back to using :meth:`make_null_session`
        in this case.
        """
        raise NotImplementedError()

    def save_session(
        self, app: Flask, session: SessionMixin, response: Response
    ) -> None:
        """This is called at the end of each request, after generating
        a response, before removing the request context. It is skipped
        if :meth:`is_null_session` returns ``True``.
        """
        raise NotImplementedError()


session_json_serializer = TaggedJSONSerializer()


def _lazy_sha1(string: bytes = b"") -> t.Any:
    """Don't access ``hashlib.sha1`` until runtime. FIPS builds may not include
    SHA-1, in which case the import and use as a default would fail before the
    developer can configure something else.
    """
    return hashlib.sha1(string)


class SecureCookieSessionInterface(SessionInterface):
    """The default session interface that stores sessions in signed cookies
    through the :mod:`itsdangerous` module.
    """

    #: the salt that should be applied on top of the secret key for the
    #: signing of cookie based sessions.
    salt = "cookie-session"
    #: the hash function to use for the signature.  The default is sha1
    digest_method = staticmethod(_lazy_sha1)
    #: the name of the itsdangerous supported key derivation.  The default
    #: is hmac.
    key_derivation = "hmac"
    #: A python serializer for the payload.  The default is a compact
    #: JSON derived serializer with support for some extra Python types
    #: such as datetime objects or tuples.
    serializer = session_json_serializer
    session_class = SecureCookieSession

    def get_signing_serializer(self, app: Flask) -> URLSafeTimedSerializer | None:
        if not app.secret_key:
            return None

        keys: list[str | bytes] = []

        if fallbacks := app.config["SECRET_KEY_FALLBACKS"]:
            keys.extend(fallbacks)

        keys.append(app.secret_key)  # itsdangerous expects current key at top
        return URLSafeTimedSerializer(
            keys,  # type: ignore[arg-type]
            salt=self.salt,
            serializer=self.serializer,
            signer_kwargs={
                "key_derivation": self.key_derivation,
                "digest_method": self.digest_method,
            },
        )

    def open_session(self, app: Flask, request: Request) -> SecureCookieSession | None:
        s = self.get_signing_serializer(app)
        if s is None:
            return None
        val = request.cookies.get(self.get_cookie_name(app))
        if not val:
            return self.session_class()
        max_age = int(app.permanent_session_lifetime.total_seconds())
        try:
            data = s.loads(val, max_age=max_age)
            return self.session_class(data)
        except BadSignature:
            return self.session_class()

    def save_session(
        self, app: Flask, session: SessionMixin, response: Response
    ) -> None:
        name = self.get_cookie_name(app)
        domain = self.get_cookie_domain(app)
        path = self.get_cookie_path(app)
        secure = self.get_cookie_secure(app)
        partitioned = self.get_cookie_partitioned(app)
        samesite = self.get_cookie_samesite(app)
        httponly = self.get_cookie_httponly(app)

        # Add a "Vary: Cookie" header if the session was accessed at all.
        if session.accessed:
            response.vary.add("Cookie")

        # If the session is modified to be empty, remove the cookie.
        # If the session is empty, return without setting the cookie.
        if not session:
            if session.modified:
                response.delete_cookie(
                    name,
                    domain=domain,
                    path=path,
                    secure=secure,
                    partitioned=partitioned,
                    samesite=samesite,
                    httponly=httponly,
                )
                response.vary.add("Cookie")

            return

        if not self.should_set_cookie(app, session):
            return

        expires = self.get_expiration_time(app, session)
        val = self.get_signing_serializer(app).dumps(dict(session))  # type: ignore[union-attr]
        response.set_cookie(
            name,
            val,
            expires=expires,
            httponly=httponly,
            domain=domain,
            path=path,
            secure=secure,
            partitioned=partitioned,
            samesite=samesite,
        )
        response.vary.add("Cookie")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\assumptions\predicates\sets.py ===
from sympy.assumptions import Predicate
from sympy.multipledispatch import Dispatcher


class IntegerPredicate(Predicate):
    """
    Integer predicate.

    Explanation
    ===========

    ``Q.integer(x)`` is true iff ``x`` belongs to the set of integer
    numbers.

    Examples
    ========

    >>> from sympy import Q, ask, S
    >>> ask(Q.integer(5))
    True
    >>> ask(Q.integer(S(1)/2))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Integer

    """
    name = 'integer'
    handler = Dispatcher(
        "IntegerHandler",
        doc=("Handler for Q.integer.\n\n"
        "Test that an expression belongs to the field of integer numbers.")
    )


class NonIntegerPredicate(Predicate):
    """
    Non-integer extended real predicate.
    """
    name = 'noninteger'
    handler = Dispatcher(
        "NonIntegerHandler",
        doc=("Handler for Q.noninteger.\n\n"
        "Test that an expression is a non-integer extended real number.")
    )


class RationalPredicate(Predicate):
    """
    Rational number predicate.

    Explanation
    ===========

    ``Q.rational(x)`` is true iff ``x`` belongs to the set of
    rational numbers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S
    >>> ask(Q.rational(0))
    True
    >>> ask(Q.rational(S(1)/2))
    True
    >>> ask(Q.rational(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Rational_number

    """
    name = 'rational'
    handler = Dispatcher(
        "RationalHandler",
        doc=("Handler for Q.rational.\n\n"
        "Test that an expression belongs to the field of rational numbers.")
    )


class IrrationalPredicate(Predicate):
    """
    Irrational number predicate.

    Explanation
    ===========

    ``Q.irrational(x)`` is true iff ``x``  is any real number that
    cannot be expressed as a ratio of integers.

    Examples
    ========

    >>> from sympy import ask, Q, pi, S, I
    >>> ask(Q.irrational(0))
    False
    >>> ask(Q.irrational(S(1)/2))
    False
    >>> ask(Q.irrational(pi))
    True
    >>> ask(Q.irrational(I))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Irrational_number

    """
    name = 'irrational'
    handler = Dispatcher(
        "IrrationalHandler",
        doc=("Handler for Q.irrational.\n\n"
        "Test that an expression is irrational numbers.")
    )


class RealPredicate(Predicate):
    r"""
    Real number predicate.

    Explanation
    ===========

    ``Q.real(x)`` is true iff ``x`` is a real number, i.e., it is in the
    interval `(-\infty, \infty)`.  Note that, in particular the
    infinities are not real. Use ``Q.extended_real`` if you want to
    consider those as well.

    A few important facts about reals:

    - Every real number is positive, negative, or zero.  Furthermore,
        because these sets are pairwise disjoint, each real number is
        exactly one of those three.

    - Every real number is also complex.

    - Every real number is finite.

    - Every real number is either rational or irrational.

    - Every real number is either algebraic or transcendental.

    - The facts ``Q.negative``, ``Q.zero``, ``Q.positive``,
        ``Q.nonnegative``, ``Q.nonpositive``, ``Q.nonzero``,
        ``Q.integer``, ``Q.rational``, and ``Q.irrational`` all imply
        ``Q.real``, as do all facts that imply those facts.

    - The facts ``Q.algebraic``, and ``Q.transcendental`` do not imply
        ``Q.real``; they imply ``Q.complex``. An algebraic or
        transcendental number may or may not be real.

    - The "non" facts (i.e., ``Q.nonnegative``, ``Q.nonzero``,
        ``Q.nonpositive`` and ``Q.noninteger``) are not equivalent to
        not the fact, but rather, not the fact *and* ``Q.real``.
        For example, ``Q.nonnegative`` means ``~Q.negative & Q.real``.
        So for example, ``I`` is not nonnegative, nonzero, or
        nonpositive.

    Examples
    ========

    >>> from sympy import Q, ask, symbols
    >>> x = symbols('x')
    >>> ask(Q.real(x), Q.positive(x))
    True
    >>> ask(Q.real(0))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Real_number

    """
    name = 'real'
    handler = Dispatcher(
        "RealHandler",
        doc=("Handler for Q.real.\n\n"
        "Test that an expression belongs to the field of real numbers.")
    )


class ExtendedRealPredicate(Predicate):
    r"""
    Extended real predicate.

    Explanation
    ===========

    ``Q.extended_real(x)`` is true iff ``x`` is a real number or
    `\{-\infty, \infty\}`.

    See documentation of ``Q.real`` for more information about related
    facts.

    Examples
    ========

    >>> from sympy import ask, Q, oo, I
    >>> ask(Q.extended_real(1))
    True
    >>> ask(Q.extended_real(I))
    False
    >>> ask(Q.extended_real(oo))
    True

    """
    name = 'extended_real'
    handler = Dispatcher(
        "ExtendedRealHandler",
        doc=("Handler for Q.extended_real.\n\n"
        "Test that an expression belongs to the field of extended real\n"
        "numbers, that is real numbers union {Infinity, -Infinity}.")
    )


class HermitianPredicate(Predicate):
    """
    Hermitian predicate.

    Explanation
    ===========

    ``ask(Q.hermitian(x))`` is true iff ``x`` belongs to the set of
    Hermitian operators.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HermitianOperator.html

    """
    # TODO: Add examples
    name = 'hermitian'
    handler = Dispatcher(
        "HermitianHandler",
        doc=("Handler for Q.hermitian.\n\n"
        "Test that an expression belongs to the field of Hermitian operators.")
    )


class ComplexPredicate(Predicate):
    """
    Complex number predicate.

    Explanation
    ===========

    ``Q.complex(x)`` is true iff ``x`` belongs to the set of complex
    numbers. Note that every complex number is finite.

    Examples
    ========

    >>> from sympy import Q, Symbol, ask, I, oo
    >>> x = Symbol('x')
    >>> ask(Q.complex(0))
    True
    >>> ask(Q.complex(2 + 3*I))
    True
    >>> ask(Q.complex(oo))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Complex_number

    """
    name = 'complex'
    handler = Dispatcher(
        "ComplexHandler",
        doc=("Handler for Q.complex.\n\n"
        "Test that an expression belongs to the field of complex numbers.")
    )


class ImaginaryPredicate(Predicate):
    """
    Imaginary number predicate.

    Explanation
    ===========

    ``Q.imaginary(x)`` is true iff ``x`` can be written as a real
    number multiplied by the imaginary unit ``I``. Please note that ``0``
    is not considered to be an imaginary number.

    Examples
    ========

    >>> from sympy import Q, ask, I
    >>> ask(Q.imaginary(3*I))
    True
    >>> ask(Q.imaginary(2 + 3*I))
    False
    >>> ask(Q.imaginary(0))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Imaginary_number

    """
    name = 'imaginary'
    handler = Dispatcher(
        "ImaginaryHandler",
        doc=("Handler for Q.imaginary.\n\n"
        "Test that an expression belongs to the field of imaginary numbers,\n"
        "that is, numbers in the form x*I, where x is real.")
    )


class AntihermitianPredicate(Predicate):
    """
    Antihermitian predicate.

    Explanation
    ===========

    ``Q.antihermitian(x)`` is true iff ``x`` belongs to the field of
    antihermitian operators, i.e., operators in the form ``x*I``, where
    ``x`` is Hermitian.

    References
    ==========

    .. [1] https://mathworld.wolfram.com/HermitianOperator.html

    """
    # TODO: Add examples
    name = 'antihermitian'
    handler = Dispatcher(
        "AntiHermitianHandler",
        doc=("Handler for Q.antihermitian.\n\n"
        "Test that an expression belongs to the field of anti-Hermitian\n"
        "operators, that is, operators in the form x*I, where x is Hermitian.")
    )


class AlgebraicPredicate(Predicate):
    r"""
    Algebraic number predicate.

    Explanation
    ===========

    ``Q.algebraic(x)`` is true iff ``x`` belongs to the set of
    algebraic numbers. ``x`` is algebraic if there is some polynomial
    in ``p(x)\in \mathbb\{Q\}[x]`` such that ``p(x) = 0``.

    Examples
    ========

    >>> from sympy import ask, Q, sqrt, I, pi
    >>> ask(Q.algebraic(sqrt(2)))
    True
    >>> ask(Q.algebraic(I))
    True
    >>> ask(Q.algebraic(pi))
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Algebraic_number

    """
    name = 'algebraic'
    AlgebraicHandler = Dispatcher(
        "AlgebraicHandler",
        doc="""Handler for Q.algebraic key."""
    )


class TranscendentalPredicate(Predicate):
    """
    Transcedental number predicate.

    Explanation
    ===========

    ``Q.transcendental(x)`` is true iff ``x`` belongs to the set of
    transcendental numbers. A transcendental number is a real
    or complex number that is not algebraic.

    """
    # TODO: Add examples
    name = 'transcendental'
    handler = Dispatcher(
        "Transcendental",
        doc="""Handler for Q.transcendental key."""
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\array\array_comprehension.py ===
import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer


class ArrayComprehension(Basic):
    """
    Generate a list comprehension.

    Explanation
    ===========

    If there is a symbolic dimension, for example, say [i for i in range(1, N)] where
    N is a Symbol, then the expression will not be expanded to an array. Otherwise,
    calling the doit() function will launch the expansion.

    Examples
    ========

    >>> from sympy.tensor.array import ArrayComprehension
    >>> from sympy import symbols
    >>> i, j, k = symbols('i j k')
    >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
    >>> a
    ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
    >>> a.doit()
    [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]
    >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k))
    >>> b.doit()
    ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k))
    """
    def __new__(cls, function, *symbols, **assumptions):
        if any(len(l) != 3 or None for l in symbols):
            raise ValueError('ArrayComprehension requires values lower and upper bound'
                              ' for the expression')
        arglist = [sympify(function)]
        arglist.extend(cls._check_limits_validity(function, symbols))
        obj = Basic.__new__(cls, *arglist, **assumptions)
        obj._limits = obj._args[1:]
        obj._shape = cls._calculate_shape_from_limits(obj._limits)
        obj._rank = len(obj._shape)
        obj._loop_size = cls._calculate_loop_size(obj._shape)
        return obj

    @property
    def function(self):
        """The function applied across limits.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.function
        10*i + j
        """
        return self._args[0]

    @property
    def limits(self):
        """
        The list of limits that will be applied while expanding the array.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.limits
        ((i, 1, 4), (j, 1, 3))
        """
        return self._limits

    @property
    def free_symbols(self):
        """
        The set of the free_symbols in the array.
        Variables appeared in the bounds are supposed to be excluded
        from the free symbol set.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.free_symbols
        set()
        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))
        >>> b.free_symbols
        {k}
        """
        expr_free_sym = self.function.free_symbols
        for var, inf, sup in self._limits:
            expr_free_sym.discard(var)
            curr_free_syms = inf.free_symbols.union(sup.free_symbols)
            expr_free_sym = expr_free_sym.union(curr_free_syms)
        return expr_free_sym

    @property
    def variables(self):
        """The tuples of the variables in the limits.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.variables
        [i, j]
        """
        return [l[0] for l in self._limits]

    @property
    def bound_symbols(self):
        """The list of dummy variables.

        Note
        ====

        Note that all variables are dummy variables since a limit without
        lower bound or upper bound is not accepted.
        """
        return [l[0] for l in self._limits if len(l) != 1]

    @property
    def shape(self):
        """
        The shape of the expanded array, which may have symbols.

        Note
        ====

        Both the lower and the upper bounds are included while
        calculating the shape.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.shape
        (4, 3)
        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))
        >>> b.shape
        (4, k + 3)
        """
        return self._shape

    @property
    def is_shape_numeric(self):
        """
        Test if the array is shape-numeric which means there is no symbolic
        dimension.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.is_shape_numeric
        True
        >>> b = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, k+3))
        >>> b.is_shape_numeric
        False
        """
        for _, inf, sup in self._limits:
            if Basic(inf, sup).atoms(Symbol):
                return False
        return True

    def rank(self):
        """The rank of the expanded array.

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j, k = symbols('i j k')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.rank()
        2
        """
        return self._rank

    def __len__(self):
        """
        The length of the expanded array which means the number
        of elements in the array.

        Raises
        ======

        ValueError : When the length of the array is symbolic

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> len(a)
        12
        """
        if self._loop_size.free_symbols:
            raise ValueError('Symbolic length is not supported')
        return self._loop_size

    @classmethod
    def _check_limits_validity(cls, function, limits):
        #limits = sympify(limits)
        new_limits = []
        for var, inf, sup in limits:
            var = _sympify(var)
            inf = _sympify(inf)
            #since this is stored as an argument, it should be
            #a Tuple
            if isinstance(sup, list):
                sup = Tuple(*sup)
            else:
                sup = _sympify(sup)
            new_limits.append(Tuple(var, inf, sup))
            if any((not isinstance(i, Expr)) or i.atoms(Symbol, Integer) != i.atoms()
                                                                for i in [inf, sup]):
                raise TypeError('Bounds should be an Expression(combination of Integer and Symbol)')
            if (inf > sup) == True:
                raise ValueError('Lower bound should be inferior to upper bound')
            if var in inf.free_symbols or var in sup.free_symbols:
                raise ValueError('Variable should not be part of its bounds')
        return new_limits

    @classmethod
    def _calculate_shape_from_limits(cls, limits):
        return tuple([sup - inf + 1 for _, inf, sup in limits])

    @classmethod
    def _calculate_loop_size(cls, shape):
        if not shape:
            return 0
        loop_size = 1
        for l in shape:
            loop_size = loop_size * l

        return loop_size

    def doit(self, **hints):
        if not self.is_shape_numeric:
            return self

        return self._expand_array()

    def _expand_array(self):
        res = []
        for values in itertools.product(*[range(inf, sup+1)
                                        for var, inf, sup
                                        in self._limits]):
            res.append(self._get_element(values))

        return ImmutableDenseNDimArray(res, self.shape)

    def _get_element(self, values):
        temp = self.function
        for var, val in zip(self.variables, values):
            temp = temp.subs(var, val)
        return temp

    def tolist(self):
        """Transform the expanded array to a list.

        Raises
        ======

        ValueError : When there is a symbolic dimension

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.tolist()
        [[11, 12, 13], [21, 22, 23], [31, 32, 33], [41, 42, 43]]
        """
        if self.is_shape_numeric:
            return self._expand_array().tolist()

        raise ValueError("A symbolic array cannot be expanded to a list")

    def tomatrix(self):
        """Transform the expanded array to a matrix.

        Raises
        ======

        ValueError : When there is a symbolic dimension
        ValueError : When the rank of the expanded array is not equal to 2

        Examples
        ========

        >>> from sympy.tensor.array import ArrayComprehension
        >>> from sympy import symbols
        >>> i, j = symbols('i j')
        >>> a = ArrayComprehension(10*i + j, (i, 1, 4), (j, 1, 3))
        >>> a.tomatrix()
        Matrix([
        [11, 12, 13],
        [21, 22, 23],
        [31, 32, 33],
        [41, 42, 43]])
        """
        from sympy.matrices import Matrix

        if not self.is_shape_numeric:
            raise ValueError("A symbolic array cannot be expanded to a matrix")
        if self._rank != 2:
            raise ValueError('Dimensions must be of size of 2')

        return Matrix(self._expand_array().tomatrix())


def isLambda(v):
    LAMBDA = lambda: 0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

class ArrayComprehensionMap(ArrayComprehension):
    '''
    A subclass of ArrayComprehension dedicated to map external function lambda.

    Notes
    =====

    Only the lambda function is considered.
    At most one argument in lambda function is accepted in order to avoid ambiguity
    in value assignment.

    Examples
    ========

    >>> from sympy.tensor.array import ArrayComprehensionMap
    >>> from sympy import symbols
    >>> i, j, k = symbols('i j k')
    >>> a = ArrayComprehensionMap(lambda: 1, (i, 1, 4))
    >>> a.doit()
    [1, 1, 1, 1]
    >>> b = ArrayComprehensionMap(lambda a: a+1, (j, 1, 4))
    >>> b.doit()
    [2, 3, 4, 5]

    '''
    def __new__(cls, function, *symbols, **assumptions):
        if any(len(l) != 3 or None for l in symbols):
            raise ValueError('ArrayComprehension requires values lower and upper bound'
                              ' for the expression')

        if not isLambda(function):
            raise ValueError('Data type not supported')

        arglist = cls._check_limits_validity(function, symbols)
        obj = Basic.__new__(cls, *arglist, **assumptions)
        obj._limits = obj._args
        obj._shape = cls._calculate_shape_from_limits(obj._limits)
        obj._rank = len(obj._shape)
        obj._loop_size = cls._calculate_loop_size(obj._shape)
        obj._lambda = function
        return obj

    @property
    def func(self):
        class _(ArrayComprehensionMap):
            def __new__(cls, *args, **kwargs):
                return ArrayComprehensionMap(self._lambda, *args, **kwargs)
        return _

    def _get_element(self, values):
        temp = self._lambda
        if self._lambda.__code__.co_argcount == 0:
            temp = temp()
        elif self._lambda.__code__.co_argcount == 1:
            temp = temp(functools.reduce(lambda a, b: a*b, values))
        return temp

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dotenv\main.py ===
import io
import logging
import os
import pathlib
import shutil
import sys
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from typing import IO, Dict, Iterable, Iterator, Mapping, Optional, Tuple, Union

from .parser import Binding, parse_stream
from .variables import parse_variables

# A type alias for a string path to be used for the paths in this file.
# These paths may flow to `open()` and `shutil.move()`; `shutil.move()`
# only accepts string paths, not byte paths or file descriptors. See
# https://github.com/python/typeshed/pull/6832.
StrPath = Union[str, "os.PathLike[str]"]

logger = logging.getLogger(__name__)


def with_warn_for_invalid_lines(mappings: Iterator[Binding]) -> Iterator[Binding]:
    for mapping in mappings:
        if mapping.error:
            logger.warning(
                "python-dotenv could not parse statement starting at line %s",
                mapping.original.line,
            )
        yield mapping


class DotEnv:
    def __init__(
        self,
        dotenv_path: Optional[StrPath],
        stream: Optional[IO[str]] = None,
        verbose: bool = False,
        encoding: Optional[str] = None,
        interpolate: bool = True,
        override: bool = True,
    ) -> None:
        self.dotenv_path: Optional[StrPath] = dotenv_path
        self.stream: Optional[IO[str]] = stream
        self._dict: Optional[Dict[str, Optional[str]]] = None
        self.verbose: bool = verbose
        self.encoding: Optional[str] = encoding
        self.interpolate: bool = interpolate
        self.override: bool = override

    @contextmanager
    def _get_stream(self) -> Iterator[IO[str]]:
        if self.dotenv_path and os.path.isfile(self.dotenv_path):
            with open(self.dotenv_path, encoding=self.encoding) as stream:
                yield stream
        elif self.stream is not None:
            yield self.stream
        else:
            if self.verbose:
                logger.info(
                    "python-dotenv could not find configuration file %s.",
                    self.dotenv_path or ".env",
                )
            yield io.StringIO("")

    def dict(self) -> Dict[str, Optional[str]]:
        """Return dotenv as dict"""
        if self._dict:
            return self._dict

        raw_values = self.parse()

        if self.interpolate:
            self._dict = OrderedDict(
                resolve_variables(raw_values, override=self.override)
            )
        else:
            self._dict = OrderedDict(raw_values)

        return self._dict

    def parse(self) -> Iterator[Tuple[str, Optional[str]]]:
        with self._get_stream() as stream:
            for mapping in with_warn_for_invalid_lines(parse_stream(stream)):
                if mapping.key is not None:
                    yield mapping.key, mapping.value

    def set_as_environment_variables(self) -> bool:
        """
        Load the current dotenv as system environment variable.
        """
        if not self.dict():
            return False

        for k, v in self.dict().items():
            if k in os.environ and not self.override:
                continue
            if v is not None:
                os.environ[k] = v

        return True

    def get(self, key: str) -> Optional[str]:
        """ """
        data = self.dict()

        if key in data:
            return data[key]

        if self.verbose:
            logger.warning("Key %s not found in %s.", key, self.dotenv_path)

        return None


def get_key(
    dotenv_path: StrPath,
    key_to_get: str,
    encoding: Optional[str] = "utf-8",
) -> Optional[str]:
    """
    Get the value of a given key from the given .env.

    Returns `None` if the key isn't found or doesn't have a value.
    """
    return DotEnv(dotenv_path, verbose=True, encoding=encoding).get(key_to_get)


@contextmanager
def rewrite(
    path: StrPath,
    encoding: Optional[str],
) -> Iterator[Tuple[IO[str], IO[str]]]:
    pathlib.Path(path).touch()

    with tempfile.NamedTemporaryFile(mode="w", encoding=encoding, delete=False) as dest:
        error = None
        try:
            with open(path, encoding=encoding) as source:
                yield (source, dest)
        except BaseException as err:
            error = err

    if error is None:
        shutil.move(dest.name, path)
    else:
        os.unlink(dest.name)
        raise error from None


def set_key(
    dotenv_path: StrPath,
    key_to_set: str,
    value_to_set: str,
    quote_mode: str = "always",
    export: bool = False,
    encoding: Optional[str] = "utf-8",
) -> Tuple[Optional[bool], str, str]:
    """
    Adds or Updates a key/value to the given .env

    If the .env path given doesn't exist, fails instead of risking creating
    an orphan .env somewhere in the filesystem
    """
    if quote_mode not in ("always", "auto", "never"):
        raise ValueError(f"Unknown quote_mode: {quote_mode}")

    quote = quote_mode == "always" or (
        quote_mode == "auto" and not value_to_set.isalnum()
    )

    if quote:
        value_out = "'{}'".format(value_to_set.replace("'", "\\'"))
    else:
        value_out = value_to_set
    if export:
        line_out = f"export {key_to_set}={value_out}\n"
    else:
        line_out = f"{key_to_set}={value_out}\n"

    with rewrite(dotenv_path, encoding=encoding) as (source, dest):
        replaced = False
        missing_newline = False
        for mapping in with_warn_for_invalid_lines(parse_stream(source)):
            if mapping.key == key_to_set:
                dest.write(line_out)
                replaced = True
            else:
                dest.write(mapping.original.string)
                missing_newline = not mapping.original.string.endswith("\n")
        if not replaced:
            if missing_newline:
                dest.write("\n")
            dest.write(line_out)

    return True, key_to_set, value_to_set


def unset_key(
    dotenv_path: StrPath,
    key_to_unset: str,
    quote_mode: str = "always",
    encoding: Optional[str] = "utf-8",
) -> Tuple[Optional[bool], str]:
    """
    Removes a given key from the given `.env` file.

    If the .env path given doesn't exist, fails.
    If the given key doesn't exist in the .env, fails.
    """
    if not os.path.exists(dotenv_path):
        logger.warning("Can't delete from %s - it doesn't exist.", dotenv_path)
        return None, key_to_unset

    removed = False
    with rewrite(dotenv_path, encoding=encoding) as (source, dest):
        for mapping in with_warn_for_invalid_lines(parse_stream(source)):
            if mapping.key == key_to_unset:
                removed = True
            else:
                dest.write(mapping.original.string)

    if not removed:
        logger.warning(
            "Key %s not removed from %s - key doesn't exist.", key_to_unset, dotenv_path
        )
        return None, key_to_unset

    return removed, key_to_unset


def resolve_variables(
    values: Iterable[Tuple[str, Optional[str]]],
    override: bool,
) -> Mapping[str, Optional[str]]:
    new_values: Dict[str, Optional[str]] = {}

    for name, value in values:
        if value is None:
            result = None
        else:
            atoms = parse_variables(value)
            env: Dict[str, Optional[str]] = {}
            if override:
                env.update(os.environ)  # type: ignore
                env.update(new_values)
            else:
                env.update(new_values)
                env.update(os.environ)  # type: ignore
            result = "".join(atom.resolve(env) for atom in atoms)

        new_values[name] = result

    return new_values


def _walk_to_root(path: str) -> Iterator[str]:
    """
    Yield directories starting from the given directory up to the root
    """
    if not os.path.exists(path):
        raise IOError("Starting path not found")

    if os.path.isfile(path):
        path = os.path.dirname(path)

    last_dir = None
    current_dir = os.path.abspath(path)
    while last_dir != current_dir:
        yield current_dir
        parent_dir = os.path.abspath(os.path.join(current_dir, os.path.pardir))
        last_dir, current_dir = current_dir, parent_dir


def find_dotenv(
    filename: str = ".env",
    raise_error_if_not_found: bool = False,
    usecwd: bool = False,
) -> str:
    """
    Search in increasingly higher folders for the given file

    Returns path to the file if found, or an empty string otherwise
    """

    def _is_interactive():
        """Decide whether this is running in a REPL or IPython notebook"""
        try:
            main = __import__("__main__", None, None, fromlist=["__file__"])
        except ModuleNotFoundError:
            return False
        return not hasattr(main, "__file__")

    def _is_debugger():
        return sys.gettrace() is not None

    if usecwd or _is_interactive() or _is_debugger() or getattr(sys, "frozen", False):
        # Should work without __file__, e.g. in REPL or IPython notebook.
        path = os.getcwd()
    else:
        # will work for .py files
        frame = sys._getframe()
        current_file = __file__

        while frame.f_code.co_filename == current_file or not os.path.exists(
            frame.f_code.co_filename
        ):
            assert frame.f_back is not None
            frame = frame.f_back
        frame_filename = frame.f_code.co_filename
        path = os.path.dirname(os.path.abspath(frame_filename))

    for dirname in _walk_to_root(path):
        check_path = os.path.join(dirname, filename)
        if os.path.isfile(check_path):
            return check_path

    if raise_error_if_not_found:
        raise IOError("File not found")

    return ""


def load_dotenv(
    dotenv_path: Optional[StrPath] = None,
    stream: Optional[IO[str]] = None,
    verbose: bool = False,
    override: bool = False,
    interpolate: bool = True,
    encoding: Optional[str] = "utf-8",
) -> bool:
    """Parse a .env file and then load all the variables found as environment variables.

    Parameters:
        dotenv_path: Absolute or relative path to .env file.
        stream: Text stream (such as `io.StringIO`) with .env content, used if
            `dotenv_path` is `None`.
        verbose: Whether to output a warning the .env file is missing.
        override: Whether to override the system environment variables with the variables
            from the `.env` file.
        encoding: Encoding to be used to read the file.
    Returns:
        Bool: True if at least one environment variable is set else False

    If both `dotenv_path` and `stream` are `None`, `find_dotenv()` is used to find the
    .env file with it's default parameters. If you need to change the default parameters
    of `find_dotenv()`, you can explicitly call `find_dotenv()` and pass the result
    to this function as `dotenv_path`.
    """
    if dotenv_path is None and stream is None:
        dotenv_path = find_dotenv()

    dotenv = DotEnv(
        dotenv_path=dotenv_path,
        stream=stream,
        verbose=verbose,
        interpolate=interpolate,
        override=override,
        encoding=encoding,
    )
    return dotenv.set_as_environment_variables()


def dotenv_values(
    dotenv_path: Optional[StrPath] = None,
    stream: Optional[IO[str]] = None,
    verbose: bool = False,
    interpolate: bool = True,
    encoding: Optional[str] = "utf-8",
) -> Dict[str, Optional[str]]:
    """
    Parse a .env file and return its content as a dict.

    The returned dict will have `None` values for keys without values in the .env file.
    For example, `foo=bar` results in `{"foo": "bar"}` whereas `foo` alone results in
    `{"foo": None}`

    Parameters:
        dotenv_path: Absolute or relative path to the .env file.
        stream: `StringIO` object with .env content, used if `dotenv_path` is `None`.
        verbose: Whether to output a warning if the .env file is missing.
        encoding: Encoding to be used to read the file.

    If both `dotenv_path` and `stream` are `None`, `find_dotenv()` is used to find the
    .env file.
    """
    if dotenv_path is None and stream is None:
        dotenv_path = find_dotenv()

    return DotEnv(
        dotenv_path=dotenv_path,
        stream=stream,
        verbose=verbose,
        interpolate=interpolate,
        override=True,
        encoding=encoding,
    ).dict()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\sqlite\aiosqlite.py ===
# dialects/sqlite/aiosqlite.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


r"""

.. dialect:: sqlite+aiosqlite
    :name: aiosqlite
    :dbapi: aiosqlite
    :connectstring: sqlite+aiosqlite:///file_path
    :url: https://pypi.org/project/aiosqlite/

The aiosqlite dialect provides support for the SQLAlchemy asyncio interface
running on top of pysqlite.

aiosqlite is a wrapper around pysqlite that uses a background thread for
each connection.   It does not actually use non-blocking IO, as SQLite
databases are not socket-based.  However it does provide a working asyncio
interface that's useful for testing and prototyping purposes.

Using a special asyncio mediation layer, the aiosqlite dialect is usable
as the backend for the :ref:`SQLAlchemy asyncio <asyncio_toplevel>`
extension package.

This dialect should normally be used only with the
:func:`_asyncio.create_async_engine` engine creation function::

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine("sqlite+aiosqlite:///filename")

The URL passes through all arguments to the ``pysqlite`` driver, so all
connection arguments are the same as they are for that of :ref:`pysqlite`.

.. _aiosqlite_udfs:

User-Defined Functions
----------------------

aiosqlite extends pysqlite to support async, so we can create our own user-defined functions (UDFs)
in Python and use them directly in SQLite queries as described here: :ref:`pysqlite_udfs`.

.. _aiosqlite_serializable:

Serializable isolation / Savepoints / Transactional DDL (asyncio version)
-------------------------------------------------------------------------

A newly revised version of this important section is now available
at the top level of the SQLAlchemy SQLite documentation, in the section
:ref:`sqlite_transactions`.


.. _aiosqlite_pooling:

Pooling Behavior
----------------

The SQLAlchemy ``aiosqlite`` DBAPI establishes the connection pool differently
based on the kind of SQLite database that's requested:

* When a ``:memory:`` SQLite database is specified, the dialect by default
  will use :class:`.StaticPool`. This pool maintains a single
  connection, so that all access to the engine
  use the same ``:memory:`` database.
* When a file-based database is specified, the dialect will use
  :class:`.AsyncAdaptedQueuePool` as the source of connections.

  .. versionchanged:: 2.0.38

    SQLite file database engines now use :class:`.AsyncAdaptedQueuePool` by default.
    Previously, :class:`.NullPool` were used.  The :class:`.NullPool` class
    may be used by specifying it via the
    :paramref:`_sa.create_engine.poolclass` parameter.

"""  # noqa

import asyncio
from collections import deque
from functools import partial

from .base import SQLiteExecutionContext
from .pysqlite import SQLiteDialect_pysqlite
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only


class AsyncAdapt_aiosqlite_cursor:
    # TODO: base on connectors/asyncio.py
    # see #10415

    __slots__ = (
        "_adapt_connection",
        "_connection",
        "description",
        "await_",
        "_rows",
        "arraysize",
        "rowcount",
        "lastrowid",
    )

    server_side = False

    def __init__(self, adapt_connection):
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self.await_ = adapt_connection.await_
        self.arraysize = 1
        self.rowcount = -1
        self.description = None
        self._rows = deque()

    def close(self):
        self._rows.clear()

    def execute(self, operation, parameters=None):
        try:
            _cursor = self.await_(self._connection.cursor())

            if parameters is None:
                self.await_(_cursor.execute(operation))
            else:
                self.await_(_cursor.execute(operation, parameters))

            if _cursor.description:
                self.description = _cursor.description
                self.lastrowid = self.rowcount = -1

                if not self.server_side:
                    self._rows = deque(self.await_(_cursor.fetchall()))
            else:
                self.description = None
                self.lastrowid = _cursor.lastrowid
                self.rowcount = _cursor.rowcount

            if not self.server_side:
                self.await_(_cursor.close())
            else:
                self._cursor = _cursor
        except Exception as error:
            self._adapt_connection._handle_exception(error)

    def executemany(self, operation, seq_of_parameters):
        try:
            _cursor = self.await_(self._connection.cursor())
            self.await_(_cursor.executemany(operation, seq_of_parameters))
            self.description = None
            self.lastrowid = _cursor.lastrowid
            self.rowcount = _cursor.rowcount
            self.await_(_cursor.close())
        except Exception as error:
            self._adapt_connection._handle_exception(error)

    def setinputsizes(self, *inputsizes):
        pass

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


class AsyncAdapt_aiosqlite_ss_cursor(AsyncAdapt_aiosqlite_cursor):
    # TODO: base on connectors/asyncio.py
    # see #10415
    __slots__ = "_cursor"

    server_side = True

    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)
        self._cursor = None

    def close(self):
        if self._cursor is not None:
            self.await_(self._cursor.close())
            self._cursor = None

    def fetchone(self):
        return self.await_(self._cursor.fetchone())

    def fetchmany(self, size=None):
        if size is None:
            size = self.arraysize
        return self.await_(self._cursor.fetchmany(size=size))

    def fetchall(self):
        return self.await_(self._cursor.fetchall())


class AsyncAdapt_aiosqlite_connection(AdaptedConnection):
    await_ = staticmethod(await_only)
    __slots__ = ("dbapi",)

    def __init__(self, dbapi, connection):
        self.dbapi = dbapi
        self._connection = connection

    @property
    def isolation_level(self):
        return self._connection.isolation_level

    @isolation_level.setter
    def isolation_level(self, value):
        # aiosqlite's isolation_level setter works outside the Thread
        # that it's supposed to, necessitating setting check_same_thread=False.
        # for improved stability, we instead invent our own awaitable version
        # using aiosqlite's async queue directly.

        def set_iso(connection, value):
            connection.isolation_level = value

        function = partial(set_iso, self._connection._conn, value)
        future = asyncio.get_event_loop().create_future()

        self._connection._tx.put_nowait((future, function))

        try:
            return self.await_(future)
        except Exception as error:
            self._handle_exception(error)

    def create_function(self, *args, **kw):
        try:
            self.await_(self._connection.create_function(*args, **kw))
        except Exception as error:
            self._handle_exception(error)

    def cursor(self, server_side=False):
        if server_side:
            return AsyncAdapt_aiosqlite_ss_cursor(self)
        else:
            return AsyncAdapt_aiosqlite_cursor(self)

    def execute(self, *args, **kw):
        return self.await_(self._connection.execute(*args, **kw))

    def rollback(self):
        try:
            self.await_(self._connection.rollback())
        except Exception as error:
            self._handle_exception(error)

    def commit(self):
        try:
            self.await_(self._connection.commit())
        except Exception as error:
            self._handle_exception(error)

    def close(self):
        try:
            self.await_(self._connection.close())
        except ValueError:
            # this is undocumented for aiosqlite, that ValueError
            # was raised if .close() was called more than once, which is
            # both not customary for DBAPI and is also not a DBAPI.Error
            # exception. This is now fixed in aiosqlite via my PR
            # https://github.com/omnilib/aiosqlite/pull/238, so we can be
            # assured this will not become some other kind of exception,
            # since it doesn't raise anymore.

            pass
        except Exception as error:
            self._handle_exception(error)

    def _handle_exception(self, error):
        if (
            isinstance(error, ValueError)
            and error.args[0] == "no active connection"
        ):
            raise self.dbapi.sqlite.OperationalError(
                "no active connection"
            ) from error
        else:
            raise error


class AsyncAdaptFallback_aiosqlite_connection(AsyncAdapt_aiosqlite_connection):
    __slots__ = ()

    await_ = staticmethod(await_fallback)


class AsyncAdapt_aiosqlite_dbapi:
    def __init__(self, aiosqlite, sqlite):
        self.aiosqlite = aiosqlite
        self.sqlite = sqlite
        self.paramstyle = "qmark"
        self._init_dbapi_attributes()

    def _init_dbapi_attributes(self):
        for name in (
            "DatabaseError",
            "Error",
            "IntegrityError",
            "NotSupportedError",
            "OperationalError",
            "ProgrammingError",
            "sqlite_version",
            "sqlite_version_info",
        ):
            setattr(self, name, getattr(self.aiosqlite, name))

        for name in ("PARSE_COLNAMES", "PARSE_DECLTYPES"):
            setattr(self, name, getattr(self.sqlite, name))

        for name in ("Binary",):
            setattr(self, name, getattr(self.sqlite, name))

    def connect(self, *arg, **kw):
        async_fallback = kw.pop("async_fallback", False)

        creator_fn = kw.pop("async_creator_fn", None)
        if creator_fn:
            connection = creator_fn(*arg, **kw)
        else:
            connection = self.aiosqlite.connect(*arg, **kw)
            # it's a Thread.   you'll thank us later
            connection.daemon = True

        if util.asbool(async_fallback):
            return AsyncAdaptFallback_aiosqlite_connection(
                self,
                await_fallback(connection),
            )
        else:
            return AsyncAdapt_aiosqlite_connection(
                self,
                await_only(connection),
            )


class SQLiteExecutionContext_aiosqlite(SQLiteExecutionContext):
    def create_server_side_cursor(self):
        return self._dbapi_connection.cursor(server_side=True)


class SQLiteDialect_aiosqlite(SQLiteDialect_pysqlite):
    driver = "aiosqlite"
    supports_statement_cache = True

    is_async = True

    supports_server_side_cursors = True

    execution_ctx_cls = SQLiteExecutionContext_aiosqlite

    @classmethod
    def import_dbapi(cls):
        return AsyncAdapt_aiosqlite_dbapi(
            __import__("aiosqlite"), __import__("sqlite3")
        )

    @classmethod
    def get_pool_class(cls, url):
        if cls._is_url_file_db(url):
            return pool.AsyncAdaptedQueuePool
        else:
            return pool.StaticPool

    def is_disconnect(self, e, connection, cursor):
        if isinstance(
            e, self.dbapi.OperationalError
        ) and "no active connection" in str(e):
            return True

        return super().is_disconnect(e, connection, cursor)

    def get_driver_connection(self, connection):
        return connection._connection


dialect = SQLiteDialect_aiosqlite

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\orienters.py ===
from sympy.core.basic import Basic
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import (eye, rot_axis1, rot_axis2, rot_axis3)
from sympy.matrices.immutable import ImmutableDenseMatrix as Matrix
from sympy.core.cache import cacheit
from sympy.core.symbol import Str
import sympy.vector


class Orienter(Basic):
    """
    Super-class for all orienter classes.
    """

    def rotation_matrix(self):
        """
        The rotation matrix corresponding to this orienter
        instance.
        """
        return self._parent_orient


class AxisOrienter(Orienter):
    """
    Class to denote an axis orienter.
    """

    def __new__(cls, angle, axis):
        if not isinstance(axis, sympy.vector.Vector):
            raise TypeError("axis should be a Vector")
        angle = sympify(angle)

        obj = super().__new__(cls, angle, axis)
        obj._angle = angle
        obj._axis = axis

        return obj

    def __init__(self, angle, axis):
        """
        Axis rotation is a rotation about an arbitrary axis by
        some angle. The angle is supplied as a SymPy expr scalar, and
        the axis is supplied as a Vector.

        Parameters
        ==========

        angle : Expr
            The angle by which the new system is to be rotated

        axis : Vector
            The axis around which the rotation has to be performed

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q1 = symbols('q1')
        >>> N = CoordSys3D('N')
        >>> from sympy.vector import AxisOrienter
        >>> orienter = AxisOrienter(q1, N.i + 2 * N.j)
        >>> B = N.orient_new('B', (orienter, ))

        """
        # Dummy initializer for docstrings
        pass

    @cacheit
    def rotation_matrix(self, system):
        """
        The rotation matrix corresponding to this orienter
        instance.

        Parameters
        ==========

        system : CoordSys3D
            The coordinate system wrt which the rotation matrix
            is to be computed
        """

        axis = sympy.vector.express(self.axis, system).normalize()
        axis = axis.to_matrix(system)
        theta = self.angle
        parent_orient = ((eye(3) - axis * axis.T) * cos(theta) +
                         Matrix([[0, -axis[2], axis[1]],
                                 [axis[2], 0, -axis[0]],
                                 [-axis[1], axis[0], 0]]) * sin(theta) +
                         axis * axis.T)
        parent_orient = parent_orient.T
        return parent_orient

    @property
    def angle(self):
        return self._angle

    @property
    def axis(self):
        return self._axis


class ThreeAngleOrienter(Orienter):
    """
    Super-class for Body and Space orienters.
    """

    def __new__(cls, angle1, angle2, angle3, rot_order):
        if isinstance(rot_order, Str):
            rot_order = rot_order.name

        approved_orders = ('123', '231', '312', '132', '213',
                           '321', '121', '131', '212', '232',
                           '313', '323', '')
        original_rot_order = rot_order
        rot_order = str(rot_order).upper()
        if not (len(rot_order) == 3):
            raise TypeError('rot_order should be a str of length 3')
        rot_order = [i.replace('X', '1') for i in rot_order]
        rot_order = [i.replace('Y', '2') for i in rot_order]
        rot_order = [i.replace('Z', '3') for i in rot_order]
        rot_order = ''.join(rot_order)
        if rot_order not in approved_orders:
            raise TypeError('Invalid rot_type parameter')
        a1 = int(rot_order[0])
        a2 = int(rot_order[1])
        a3 = int(rot_order[2])
        angle1 = sympify(angle1)
        angle2 = sympify(angle2)
        angle3 = sympify(angle3)
        if cls._in_order:
            parent_orient = (_rot(a1, angle1) *
                             _rot(a2, angle2) *
                             _rot(a3, angle3))
        else:
            parent_orient = (_rot(a3, angle3) *
                             _rot(a2, angle2) *
                             _rot(a1, angle1))
        parent_orient = parent_orient.T

        obj = super().__new__(
            cls, angle1, angle2, angle3, Str(rot_order))
        obj._angle1 = angle1
        obj._angle2 = angle2
        obj._angle3 = angle3
        obj._rot_order = original_rot_order
        obj._parent_orient = parent_orient

        return obj

    @property
    def angle1(self):
        return self._angle1

    @property
    def angle2(self):
        return self._angle2

    @property
    def angle3(self):
        return self._angle3

    @property
    def rot_order(self):
        return self._rot_order


class BodyOrienter(ThreeAngleOrienter):
    """
    Class to denote a body-orienter.
    """

    _in_order = True

    def __new__(cls, angle1, angle2, angle3, rot_order):
        obj = ThreeAngleOrienter.__new__(cls, angle1, angle2, angle3,
                                         rot_order)
        return obj

    def __init__(self, angle1, angle2, angle3, rot_order):
        """
        Body orientation takes this coordinate system through three
        successive simple rotations.

        Body fixed rotations include both Euler Angles and
        Tait-Bryan Angles, see https://en.wikipedia.org/wiki/Euler_angles.

        Parameters
        ==========

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, BodyOrienter
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        A 'Body' fixed rotation is described by three angles and
        three body-fixed rotation axes. To orient a coordinate system D
        with respect to N, each sequential rotation is always about
        the orthogonal unit vectors fixed to D. For example, a '123'
        rotation will specify rotations about N.i, then D.j, then
        D.k. (Initially, D.i is same as N.i)
        Therefore,

        >>> body_orienter = BodyOrienter(q1, q2, q3, '123')
        >>> D = N.orient_new('D', (body_orienter, ))

        is same as

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter1 = AxisOrienter(q1, N.i)
        >>> D = N.orient_new('D', (axis_orienter1, ))
        >>> axis_orienter2 = AxisOrienter(q2, D.j)
        >>> D = D.orient_new('D', (axis_orienter2, ))
        >>> axis_orienter3 = AxisOrienter(q3, D.k)
        >>> D = D.orient_new('D', (axis_orienter3, ))

        Acceptable rotation orders are of length 3, expressed in XYZ or
        123, and cannot have a rotation about about an axis twice in a row.

        >>> body_orienter1 = BodyOrienter(q1, q2, q3, '123')
        >>> body_orienter2 = BodyOrienter(q1, q2, 0, 'ZXZ')
        >>> body_orienter3 = BodyOrienter(0, 0, 0, 'XYX')

        """
        # Dummy initializer for docstrings
        pass


class SpaceOrienter(ThreeAngleOrienter):
    """
    Class to denote a space-orienter.
    """

    _in_order = False

    def __new__(cls, angle1, angle2, angle3, rot_order):
        obj = ThreeAngleOrienter.__new__(cls, angle1, angle2, angle3,
                                         rot_order)
        return obj

    def __init__(self, angle1, angle2, angle3, rot_order):
        """
        Space rotation is similar to Body rotation, but the rotations
        are applied in the opposite order.

        Parameters
        ==========

        angle1, angle2, angle3 : Expr
            Three successive angles to rotate the coordinate system by

        rotation_order : string
            String defining the order of axes for rotation

        See Also
        ========

        BodyOrienter : Orienter to orient systems wrt Euler angles.

        Examples
        ========

        >>> from sympy.vector import CoordSys3D, SpaceOrienter
        >>> from sympy import symbols
        >>> q1, q2, q3 = symbols('q1 q2 q3')
        >>> N = CoordSys3D('N')

        To orient a coordinate system D with respect to N, each
        sequential rotation is always about N's orthogonal unit vectors.
        For example, a '123' rotation will specify rotations about
        N.i, then N.j, then N.k.
        Therefore,

        >>> space_orienter = SpaceOrienter(q1, q2, q3, '312')
        >>> D = N.orient_new('D', (space_orienter, ))

        is same as

        >>> from sympy.vector import AxisOrienter
        >>> axis_orienter1 = AxisOrienter(q1, N.i)
        >>> B = N.orient_new('B', (axis_orienter1, ))
        >>> axis_orienter2 = AxisOrienter(q2, N.j)
        >>> C = B.orient_new('C', (axis_orienter2, ))
        >>> axis_orienter3 = AxisOrienter(q3, N.k)
        >>> D = C.orient_new('C', (axis_orienter3, ))

        """
        # Dummy initializer for docstrings
        pass


class QuaternionOrienter(Orienter):
    """
    Class to denote a quaternion-orienter.
    """

    def __new__(cls, q0, q1, q2, q3):
        q0 = sympify(q0)
        q1 = sympify(q1)
        q2 = sympify(q2)
        q3 = sympify(q3)
        parent_orient = (Matrix([[q0 ** 2 + q1 ** 2 - q2 ** 2 -
                                  q3 ** 2,
                                  2 * (q1 * q2 - q0 * q3),
                                  2 * (q0 * q2 + q1 * q3)],
                                 [2 * (q1 * q2 + q0 * q3),
                                  q0 ** 2 - q1 ** 2 +
                                  q2 ** 2 - q3 ** 2,
                                  2 * (q2 * q3 - q0 * q1)],
                                 [2 * (q1 * q3 - q0 * q2),
                                  2 * (q0 * q1 + q2 * q3),
                                  q0 ** 2 - q1 ** 2 -
                                  q2 ** 2 + q3 ** 2]]))
        parent_orient = parent_orient.T

        obj = super().__new__(cls, q0, q1, q2, q3)
        obj._q0 = q0
        obj._q1 = q1
        obj._q2 = q2
        obj._q3 = q3
        obj._parent_orient = parent_orient

        return obj

    def __init__(self, angle1, angle2, angle3, rot_order):
        """
        Quaternion orientation orients the new CoordSys3D with
        Quaternions, defined as a finite rotation about lambda, a unit
        vector, by some amount theta.

        This orientation is described by four parameters:

        q0 = cos(theta/2)

        q1 = lambda_x sin(theta/2)

        q2 = lambda_y sin(theta/2)

        q3 = lambda_z sin(theta/2)

        Quaternion does not take in a rotation order.

        Parameters
        ==========

        q0, q1, q2, q3 : Expr
            The quaternions to rotate the coordinate system by

        Examples
        ========

        >>> from sympy.vector import CoordSys3D
        >>> from sympy import symbols
        >>> q0, q1, q2, q3 = symbols('q0 q1 q2 q3')
        >>> N = CoordSys3D('N')
        >>> from sympy.vector import QuaternionOrienter
        >>> q_orienter = QuaternionOrienter(q0, q1, q2, q3)
        >>> B = N.orient_new('B', (q_orienter, ))

        """
        # Dummy initializer for docstrings
        pass

    @property
    def q0(self):
        return self._q0

    @property
    def q1(self):
        return self._q1

    @property
    def q2(self):
        return self._q2

    @property
    def q3(self):
        return self._q3


def _rot(axis, angle):
    """DCM for simple axis 1, 2 or 3 rotations. """
    if axis == 1:
        return Matrix(rot_axis1(angle).T)
    elif axis == 2:
        return Matrix(rot_axis2(angle).T)
    elif axis == 3:
        return Matrix(rot_axis3(angle).T)