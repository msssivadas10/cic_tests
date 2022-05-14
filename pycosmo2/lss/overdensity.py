from typing import Union
from pycosmo2._bases import Cosmology, OverDensity
from math import pi

class FoF(OverDensity):

    def __init__(self) -> None:
        self._value = 'FoF'

    def value(self, z: float, cm: Cosmology) -> int:
        return None

class SO(OverDensity):
    ...

class _vir(SO):

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

    def __repr__(self) -> str:
        return f"Overdensity('{ self._value }c')"

    def value(self, z: float, cm: Cosmology) -> int:
        return round( round( self._value ) * cm.criticalDensity( z ) / cm.rho_m( z ) )

    def __mul__(self, __o: int) -> SO:
        if not isinstance( __o, int ):
            return NotImplemented
        return _m( __o * self._value )
    
    __rmul__ = __mul__


fof = FoF()
vir = _vir()
m   = _m()
c   = _c()


def overdensity(value: Union[str, int, OverDensity]) -> OverDensity:
    if isinstance( value, OverDensity ):
        return value
    
    if isinstance( value, int ):
        return value * m

    if not isinstance( value, str ):
        raise TypeError("value must be a 'str', 'int' or 'OverDensity'")

    import re

    tmp   = value
    value = value.lower()
    m     = re.match( r'(\d*)([mc]|[fovir]{3})', value )
    if m is None:
        raise ValueError("cannot parse mass definition: '{}'".format( value ))

    value, ref = m.groups()

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


    
