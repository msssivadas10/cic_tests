from typing import Union
from pycosmo._bases import Cosmology, OverDensity
from math import pi

class FoF(OverDensity):
    r"""
    Represents a Friends of Friends (FoF) halo. Since this is charecterised in terms of the linking length, 
    not the overdensity, it gives None as value of overdensity. Only defined for completeness of the module. 
    """

    def __init__(self) -> None:
        self._value = 'FoF'

    def value(self, z: float, cm: Cosmology) -> int:
        return None

class SO(OverDensity):
    r"""
    Represents a spherical overdensity (SO) halo. This is charecterised by the value of the overdensity, 
    denoted :math:`\Delta`.
    """
    ...

class _vir(SO):
    r"""
    Virial overdensity. This gives the spherical overdensity of a virialised object, which is approximately 
    :math:`18\pi^2 \approx 178` for a matter deominated universe. For a flat universe 

    .. math::
        \Delta_c = 18\pi^2 + 82x - 39x^2

    and for a universe without dark-energy, 

    .. math::
        \Delta_c = 18\pi^2 + 60x - 32x^2

    where :math:`x = \Omega_m - 1`

    """

    def __init__(self) -> None:
        self._value = None

    def __repr__(self) -> str:
        return "Overdensity('vir')"

    def value(self, z: float, cm: Cosmology) -> int:
        x = cm.Om( z ) - 1

        Delta_c = 18*pi**2
        if cm.flat:
            Delta_c += 82.0*x - 39.0*x**2
        elif cm.Ode0 == 0.0:
            Delta_c += 60.0*x - 32.0*x**2
        return round( round( Delta_c ) * cm.criticalDensity( z ) / cm.rho_m( z ) )

class _m(SO):
    r"""
    Spherical overdensity expressed in terms of the mean background density. It defines the suffix `m := _m(1)`, 
    which can be used to define overdensities in the form `value*m` as in `200m` (e.g., `200m` is specified by 
    `200*m`).
    """

    def __repr__(self) -> str:
        return f"Overdensity('{ self._value }m')"

    def value(self, z: float, cm: Cosmology) -> int:
        return round( self._value )

    def __mul__(self, __o: int) -> SO:
        if not isinstance( __o, int ):
            return NotImplemented
        return _m( __o * self._value )
    
    __rmul__ = __mul__

class _c(SO):
    r"""
    Spherical overdensity expressed in terms of the critical density. It defines the suffix `c := _c(1)`, which 
    can be used to define overdensities in the form `value*c` as in `500c` (e.g., `500c` is specified by `500*c`).
    """

    def __repr__(self) -> str:
        return f"Overdensity('{ self._value }c')"

    def value(self, z: float, cm: Cosmology) -> int:
        return round( round( self._value ) * cm.criticalDensity( z ) / cm.rho_m( z ) )

    def __mul__(self, __o: int) -> SO:
        if not isinstance( __o, int ):
            return NotImplemented
        return _m( __o * self._value )
    
    __rmul__ = __mul__


fof = FoF()  # FoF halo indicator
vir = _vir() # virial overdensity
m   = _m()   # spherical overdensity w.r.to the mean: *m
c   = _c()   # spherical overdensity w.r.to critical density: *c


def overdensity(value: Union[str, int, OverDensity]) -> OverDensity:
    r"""
    Take an overdensity-like object and return the corresponding :class:`OverDensity` object.

    Parameters
    ----------
    value: int, str, OverDensity
        Overdensity specifier. If an integer, it represents the overdensity w.r.to the mean background density. 
        If is a string, it must be either 'fof', 'vir' or, of the form `*m` or `*c`, where `*` is a number and 
        the suffux represents the reference. If it is an :class:`OverDensity` object, then return itself.
    
    Returns
    -------
    Delta: OverDensity
        Overdensity as an :class:`OverDensity` object.

    Examples
    --------
    
    """
    if isinstance( value, OverDensity ):
        return value
    
    if isinstance( value, int ):
        return value * m

    if not isinstance( value, str ):
        raise TypeError("value must be a 'str', 'int' or 'OverDensity'")

    import re

    tmp   = value
    value = value.lower()
    match = re.match( r'(\d*)([mc]|[fovir]{3})', value )
    if match is None:
        raise ValueError("cannot parse mass definition: '{}'".format( value ))

    value, ref = match.groups()

    if value == 'fof':
        return fof

    if value == 'vir':
        return vir

    if value:
        if ref == 'm':
            return int( value ) * m
        if ref == 'c':
            return int( value ) * c
    raise ValueError(f"invalid overdensity value: '{ tmp }'")


    
