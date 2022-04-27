from abc import ABC, abstractmethod, abstractproperty
from typing import Any

import numpy as np 

class Component( ABC ):

    __slots__ = 'density0', 'wparam'

    def __init__(self, density0: float, w: Any) -> None:
        self.density0, self.wparam = density0, w

    @abstractmethod
    def density(self, z: Any) -> Any:
        ...

    @abstractmethod
    def w(self, z: Any) -> Any:
        ...

class ComponentCollection( Component ):

    __slots__ = ()

    def __init__(self, density0: float, w: Any) -> None:
        super().__init__( density0, w )

    @abstractproperty
    def children(self) -> tuple:
        ...

    @property
    def totalDensity(self) -> float:
        total = 0.0
        for child in self.children:
            child =  getattr( self, child )
            if not isinstance( child, Component ):
                continue
            total += child.density0
        return total
    
    @property
    def remainingDensity(self) -> float:
        return self.density0 - self.totalDensity


###################################################################################################


class MatterComponent( Component ):

    def __init__(self, density0: float ) -> None:
        if density0 < 0.0:
            raise ValueError( "density of a matter component must be positive" )
        super().__init__( density0, w = 0.0 )

    def density(self, z: Any) -> Any:
        zp1 = np.asfarray( z ) + 1
        return self.density0 * zp1**3

    def w(self, z: Any) -> Any:
        return np.zeros_like( z, dtype = 'float' )

class Baryon( MatterComponent ):

    def __init__(self, density0: float) -> None:
        super().__init__( density0 )

class ColdDarkMatter( MatterComponent ):

    def __init__(self, density0: float) -> None:
        super().__init__( density0 )

class MassiveNeutrino( MatterComponent ):

    __slots__ = 'Nnu', 'Mnu'

    def __init__(self, density0: float, Nnu: float) -> None:
        if not ( Nnu > 0.0 ):
            raise ValueError( "number must be non-zero and positive" )
        self.Nnu, self.Mnu = Nnu, None

        super().__init__( density0 )

    def computeMass(self, h: float) -> None:
        self.Mnu = 91.5*self.density0 * h**2 / self.Nnu

class Matter( MatterComponent, ComponentCollection ):

    __slots__ = 'baryon', 'cdm', 'massive_nu', 

    def __init__(self, density0: float ) -> None:
        super().__init__( density0 )

        self.baryon     = None
        self.cdm        = None
        self.massive_nu = None

    @property
    def children(self) -> tuple:
        return self.__slots__

class DarkEnergy( Component ):

    def __init__(self, density0: float, w: Any) -> None:
        if density0 < 0.0:
            raise ValueError( "density of a dark-energy component must be positive" )
        super().__init__(density0, w) 

    def density(self, z: Any) -> Any:
        zp1 = np.asfarray( z ) + 1
        return self.density0 * zp1**self.w( z )

class LinearDarkEnergy( DarkEnergy ):

    def __init__(self, density0: float, w0: float = -1.0, wa: float = 0.0) -> None:
        super().__init__(density0, w = ( w0, wa ))

    @property
    def w0(self) -> float:
        return self.wparam[0]

    @property
    def wa(self) -> float:
        return self.wparam[1]

    def w(self, z: Any) -> Any:
        z = np.asfarray( z )
        return self.wa + self.wa * z / ( z + 1 )

class Curvature( Component ):

    def __init__(self, density0: float) -> None:
        super().__init__( density0, w = -1./3 )

    def density(self, z: Any) -> Any:
        zp1 = np.asfarray( z ) + 1
        return self.density0 * zp1**2

    def w(self, z: Any) -> Any:
        return self.wparam * np.ones_like( z, dtype = 'float' )


#####################################################################################################

class Universe( ComponentCollection ):
    
    __slots__ = 'matter', 'dark_energy', 'curvature', 'flat'

    def __init__(self, flat: float = True) -> None:
        super().__init__( density0 = 1.0, w = None )

# m = Matter(0.4)
# print(m.wparam)