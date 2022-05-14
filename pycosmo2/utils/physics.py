from typing import Any, Union
import re

class Dimension:

    __slots__ = 'T', 'L', 'M', 'I', 'K', 'N', 'J'

    def __init__(self, t: int = 0, l: int = 0, m: int = 0, i: int = 0, k: int = 0, n: int = 0, j: int = 0) -> None:
        self.T = t
        self.L = l
        self.M = m
        self.I = i
        self.K = k
        self.N = n
        self.J = j

    def __repr__(self) -> str:
        dim = { __name: getattr( self, __name ) for __name in self.__slots__ }
        dim = [ f'{ key }={ value }' for key, value in dim.items() if value ]
        return f"Dimension({ ', '.join( dim ) if dim else 'None' })"

    def __eq__(self, __o: object) -> bool:
        if not isinstance( __o, Dimension ):
            return NotImplemented
        return (
                    ( self.T == __o.T )
                        and ( self.L == __o.L )
                        and ( self.M == __o.M )
                        and ( self.I == __o.I )
                        and ( self.K == __o.K )
                        and ( self.N == __o.N )
                        and ( self.J == __o.J )
               )

    def __mul__(self, __o: object) -> 'Dimension':
        if not isinstance( __o, Dimension ):
            return NotImplemented
        return Dimension(
                            t = self.T + __o.T,
                            l = self.L + __o.L,
                            m = self.M + __o.M,
                            i = self.I + __o.I,
                            k = self.K + __o.K,
                            n = self.N + __o.N,
                            j = self.J + __o.J,
                        )

    def __truediv__(self, __o: object) -> 'Dimension':
        if not isinstance( __o, Dimension ):
            return NotImplemented
        return Dimension(
                            t = self.T - __o.T,
                            l = self.L - __o.L,
                            m = self.M - __o.M,
                            i = self.I - __o.I,
                            k = self.K - __o.K,
                            n = self.N - __o.N,
                            j = self.J - __o.J,
                        )
    
    def __pow__(self, __o: object) -> 'Dimension':
        if not isinstance( __o, int ):
            return NotImplemented
        return Dimension(
                            t = self.T * __o,
                            l = self.L * __o,
                            m = self.M * __o,
                            i = self.I * __o,
                            k = self.K * __o,
                            n = self.N * __o,
                            j = self.J * __o,
                        )

class Unit:

    __slots__ = 'base_value', 'dim', 'id', 

    def __init__(self, id: str, dim: Dimension, base_value: float = 1.0) -> None:
        self.id = id 
        self.dim = dim 
        self.base_value = base_value

    def __repr__(self) -> str:
        id = self.id
        if not id:
            id = 'None'
        return f"Unit('{ id }')"
    
    def scale(self, scale: float, id: str) -> 'Unit':
        return Unit(id, self.dim, self.base_value * scale)

    @property
    def parsed_id(self) -> dict:
        id  = {}
        for x in [ re.match( r'(\w*)\^?([+-]?\d*)', xi ).groups() for xi in self.id.split( '*' ) ]:
            if not x[0]:
                continue
            id[ x[0] ] = int( x[1] ) if x[1] else 1 
        return id

    def __mul__(self, __o: object) -> 'Unit':
        if not isinstance( __o, Unit ):
            return NotImplemented

        id1 = self.parsed_id
        id2 = __o.parsed_id

        all = { *id1, *id2 }
        id  = []
        for symbol in all:
            exp = 0
            if symbol in id1:
                exp += id1[ symbol ]
            if symbol in id2:
                exp += id2[ symbol ]
            if exp:
                id.append( f"{ symbol }^{ exp }" if exp != 1 else symbol )
        id = '*'.join( id )

        dim = self.dim * __o.dim
        return Unit( id, dim, self.base_value * __o.base_value )

    def __rmul__(self, __o: object) -> 'Unit':
        return Unit( '', self.dim, __o * self.base_value )

    def __truediv__(self, __o: object) -> 'Unit':
        if not isinstance( __o, Unit ):
            return NotImplemented

        id1 = self.parsed_id
        id2 = __o.parsed_id
        all = { *id1, *id2 }
        id  = []
        for symbol in all:
            exp = 0
            if symbol in id1:
                exp += id1[ symbol ]
            if symbol in id2:
                exp -= id2[ symbol ]
            if exp:
                id.append( f"{ symbol }^{ exp }" if exp != 1 else symbol )
        id = '*'.join( id )
        
        dim = self.dim / __o.dim
        return Unit( id, dim, self.base_value / __o.base_value )
    
    def __rtruediv__(self, __o: object) -> 'Unit':
        return Unit( '', self.dim**-1, __o / self.base_value )

    def __pow__(self, __o: object) -> 'Unit':
        if not isinstance( __o, int ):
            return NotImplemented

        id1 = self.parsed_id
        id  = []
        for symbol in id1:
            exp = id1[ symbol ] * __o
            id.append( f"{ symbol }^{ exp }" if exp != 1 else symbol )
        id = '*'.join( id )

        return Unit( id, self.dim**__o, self.base_value**__o )

    def getConversionFactor(self, other: 'Unit') -> float:
        if self.dim != other.dim:
            raise ValueError("cannot convert unit: dimension mismatch")
        return other.base_value / self.base_value


class Quantity:

    __slots__ = 'value', 'unit', 

    def __init__(self, value: Any, unit: Unit = None) -> None:
        self.value = value
        self.unit  = Unit( '', Dimension() ) if unit is None else unit

    def __repr__(self) -> str:
        return f"Quantity(value={ repr( self.value ) }, unit={ self.unit })"
    
    def convert2(self, unit: Unit) -> 'Quantity':
        fac = self.unit.getConversionFactor( unit )
        return Quantity( self.value / fac, unit )

def unit(__o: Union[Unit, Quantity], name: str) -> Unit:
    r"""
    Create a unit from an instance of :class:`Unit` or :class:`Quantity`.
    """
    if isinstance( __o, Unit ):
        return Unit( name, __o.dim, __o.base_value )
    
    if not isinstance( __o, Quantity ):
        raise TypeError("argument must be a 'Unit' or 'Quantity'")  

    base_value = __o.value * __o.unit.base_value
    return Unit( name, __o.unit.dim, base_value )


#################################################################################################


class dimensions:
    """ 
    A table of some basic dimensions, such as the base dimensions. 
    """

    # ==========================================
    # Base dimensions
    # ==========================================

    O = Dimension() # dimensionless quatity

    T = Dimension(t = 1) # time
    L = Dimension(l = 1) # length
    M = Dimension(m = 1) # mass
    I = Dimension(i = 1) # current
    K = Dimension(k = 1) # temperature
    N = Dimension(n = 1) # amount of substance
    J = Dimension(j = 1) # luminous intensity

    # ===========================================
    # Other derived quatities
    # =========================================== 

    VELOCITY     = L*T**-1
    ACCELERATION = L*T**-2
    FORCE        = M*L*T**-2
    ENERGY       = M*L**2*T**-2
    DENSITY      = M*L**-3 
    CHARGE       = I*T


_d = dimensions()

class units:
    """
    A table of units for some physical quatities (SI system is used for base units).
    """

    # =================================================
    # Base units (SI system)
    # ================================================= 

    s   = Unit( 's'  , _d.T) # second, time
    kg  = Unit( 'kg' , _d.M) # kilogram, mass
    m   = Unit( 'm'  , _d.L) # meter, length
    A   = Unit( 'A'  , _d.I) # ampere, current
    K   = Unit( 'K'  , _d.K) # kelvin, temperature
    mol = Unit( 'mol', _d.N) # mole, amount of substance

    # ================================================
    # Units of base quantities ( in base units )
    # ================================================

    # time: 
    min = unit( 60.0*s, 'min' )      # minute = 60 sec
    hr  = unit( 60*min, 'hr' )       # hour = 60 min
    day = unit( 24*hr,  'day' )      # day = 24 hours
    yr  = unit( 31558149.8*s, 'yr' ) # (sidereal) year = 365.25 days approx. 
    Gyr = unit( 1e+09*yr, 'Gyr' )    # giga year = 10^9 year

    # mass:
    g    = unit( 0.001*kg, 'g' )          # gram
    Msun = unit( 1.98842e+30*kg, 'Msun' ) # solar mass

    # length:
    angstrum = unit( 1e-10*m, 'angstrum' ) # angstrum

    pm = unit( 1e-12*m, 'pm' ) # picometer
    nm = unit( 1e-09*m, 'nm' ) # nanometer
    um = unit( 1e-06*m, 'um' ) # micrometer
    mm = unit( 1e-03*m, 'mm' ) # millimeter
    cm = unit( 1e-02*m, 'cm' ) # centimeter
    km = unit( 1e+03*m, 'km' ) # kilometer

    AU  = unit( 1.49597870700e+11*m, 'AU' ) # astronomical unit
    pc  = unit( 3.08567758149e+16*m, 'kc' ) # parsec
    kpc = unit( 1000.0*pc,  'kpc' )         # kilo-parsec
    Mpc = unit( 1000.0*kpc, 'Mpc' )         # mega-parsec


#################################################################################################

# some physical quantities

class Density(Quantity):
    r"""
    Represents the density of something in some unit. Default unit is the astrophysical unit, 
    :math:`{ \rm Msun }/{ \rm MPc }^3`. 
    """

    def __init__(self, value: Any, unit: Unit = units.Msun/units.Mpc**3) -> None:
        super().__init__(value, unit)

    def convert2(self, unit: Unit) -> 'Quantity':
        if unit.dim != _d.DENSITY:
            raise ValueError("cannot convert to a non-density unit")
        return super().convert2(unit)

class Time(Quantity):
    r"""
    Represent the time in some unit. Default unit is the year.
    """

    def __init__(self, value: Any, unit: Unit = units.yr) -> None:
        super().__init__(value, unit)

    def convert2(self, unit: Unit) -> 'Quantity':
        if unit.dim != _d.T:
            raise ValueError("cannot convert to a non-time unit")
        return super().convert2(unit)

class Length(Quantity):
    r"""
    Represent the length or distance in some unit. Default unit is the Mpc (mega parsec).
    """

    def __init__(self, value: Any, unit: Unit = units.Mpc) -> None:
        super().__init__(value, unit)

    def convert2(self, unit: Unit) -> 'Quantity':
        if unit.dim != _d.L:
            raise ValueError("cannot convert to a non-length unit")
        return super().convert2(unit)
