from abc import ABC, abstractmethod
from typing import Any, Type
import warnings
import numpy as np

class ComponentError( Exception ):
    """
    Base class of exceptions used by :class:`Component` objects.
    """

class Component( ABC ):
    """
    Represents a component of the universe. 
    """
    __slots__ = '_density0', '_w', 'negative_density', 'children', 'name'

    def __init__(self, density0: float, w: Any = None, negative_density: bool = False, name: str = None) -> None:
        if not isinstance( name, str ):
            raise TypeError("name must be an 'str'")
        self.name             = name
        self.negative_density = negative_density

        if not negative_density and density0 < 0.0:
            raise ValueError(f"density of { self.name } cannot be negative")
        self._density0, self._w = density0, w

        self.children = {}

    @property
    def density0(self) -> float:
        """
        Density of the component at redshift 0.
        """
        return self._density0
    
    @property
    def w(self) -> Any:
        """
        Equation of state parameter(s).
        """
        return self._w

    @property
    def totalAssignedDensity(self) -> float:
        """
        Total density assigned to the children.
        """
        return sum( child.density0 for child in self.children.values() )

    @property
    def reminder(self) -> float:
        """
        Remaining density, after assigning to the children.
        """
        return self.density0 - self.totalAssignedDensity

    @abstractmethod
    def density(self, z: Any) -> Any:
        """
        Density of the component at redshift z.
        """
        ...

    @abstractmethod
    def wz(self, z: Any) -> Any:
        """
        Equation of state parameter at redshift z.
        """
        ...

    def pressure(self, z: Any) -> Any:
        """
        Pressure of the component as a function of redshift.
        """
        return self.wz( z ) * self.density( z )

    def addchild(self, o: object, raise_error: bool = True) -> None:
        """
        Add a child component to this.
        """
        if not isinstance( o, Component ):
            raise TypeError("object must be a 'Component'")

        # check if the component already a child
        if o.name in self.children:
            raise ValueError(f"'{ o.name }' already presnt in children")

        # check if the total density of the children not greater than parent density
        total_density = self.totalAssignedDensity
        
        if total_density + o.density0 > self.density0 and raise_error:
            child_list = ', '.join([ child for child in self.children ])
            raise ValueError( f"total density of { child_list } and { o.name } cannot be greater than the density of { self.name }" )
            
        # add the child component
        self.children[ o.name ] = o

    def fill(self, cls: Type['Component']) -> None:
        """
        Fill the remaining density with another component.
        """
        if not issubclass( cls, Component ):
            raise TypeError("cls must be a subclass of 'Component'")
        
        o = cls( self.reminder )
        return self.addchild( o )
        
    def __getitem__(self, name: str) -> object:
        """
        Get a child component with specified name.
        """
        if name not in self.children:
            raise KeyError(f"no child with name '{ name }'")
        return self.children[ name ]

    def __repr__(self) -> str:
        children = ", ".join([ f"'{ c }'" for c in self.children ])
        return f"{ self.name }(density0={ self.density0 }, w={ self.w }, children=[{ children }])"


#######################################################################################################

class Matter( Component ):
    """
    Represent any non-relativistic matter component in the universe.
    """

    def __init__(self, Om0: float) -> None:
        super().__init__( density0 = Om0, w = 0, negative_density = False, name = 'matter' )

    def density(self, z: Any) -> Any:
        return self.density0 * ( 1 + np.asfarray( z ) )**3

    def wz(self, z: Any) -> Any:
        return np.zeros_like( z, dtype = 'float' )

class Baryon( Matter ):
    """
    Represent the baryon component in the universe.
    """

    def __init__(self, Ob0: float) -> None:
        super().__init__( Ob0 )
        
        self.name = 'baryon' # change the name of the component

class ColdDarkMatter( Matter ):
    """
    Represent cold dark-matter component in the universe.
    """

    def __init__(self, Oc0: float) -> None:
        super().__init__( Oc0 )

        self.name = 'cdm' # change the name of the component

class MassiveNeutrino( Matter ):
    """
    Represent non-relativistic massive neutrinos in the universe.
    """
    __slots__ = '_number', '_mass'

    def __init__(self, Onu0: float, Nnu: float, ) -> None:
        super().__init__( Onu0 )

        self.name = 'massive_nu' # change the name of the component
        
        if Onu0:
            if Nnu <= 0.0:
                raise ValueError("number of neutrinos must be postive and non-zero")
            self._mass   = None
            self._number = Nnu

    def getmass(self, h: float) -> None:
        """
        Compute the mass of the neutrino.
        """
        self._mass = 91.5 * self.density0 * h**2 / self._number

    @property
    def mnu(self) -> float:
        if self._mass is None:
            raise ComponentError( "neutrino mass is not calculated" )
        return self._mass

    @property
    def Nnu(self) -> float:
        return self._number

class DarkEnergy( Component ):
    """
    Represent the dark energy component in the universe.
    """
    __slots__ = 'model_name'

    def __init__(self, Ode0: float, w: Any, name: str = 'dark_energy') -> None:
        super().__init__( Ode0, w, negative_density = False, name = 'dark_energy' )
        self.model_name = name

class LinearDarkEnergy( DarkEnergy ):
    r"""
    Represent a dark-energy component with the equation of state parameter linearly varying 
    with time or scale factor :math:`a(t)`. It is given by 

    .. math::

        w(z) = w_0 + w_a (1-a)  w_0 + w_a \frac{z}{z+1}

    :math:`w_0` is the constant part and `w_a` defines the time variation. For the cosmological 
    constant :math:`\Lambda`, :math:`w_0 = -1` and :math:`w_a = 0`.
    """

    def __init__(self, Ode0: float, w0: float = -1.0, wa: float = 0.0) -> None:
        super().__init__(Ode0, w = ( w0, wa ), name = 'dark_energy_lw')

    def density(self, z: Any) -> Any:
        return self.density0 * ( np.asfarray( z ) + 1 )**self.wz( z )

    def wz(self, z: Any) -> Any:
        z = np.asfarray( z )
        return self.w[0] + self.w[1] * z / ( z + 1 )

    @property
    def w0(self) -> float:
        """
        Constant part of the equation of state parameter.
        """
        return self.w[0]

    @property
    def wa(self) -> float:
        """
        Coefficient of the time varying part of the equation of state parameter.
        """
        return self.w[1]

class Curvature( Component ):
    """
    Curvature component of the universe.
    """

    def __init__(self, Ok0: float) -> None:
        super().__init__( Ok0, w = -1./3, negative_density = True, name = 'curvature')

    def density(self, z: Any) -> Any:
        return self.density0 * ( np.asfarray( z ) + 1 )**2

    def wz(self, z: Any) -> Any:
        return -np.ones_like( z, dtype = 'float' ) / 3.0


############################################################################################################


class Universe( Component ):
    """
    Represent the universe.
    """
    __slots__ = 'flat', 'dark_energy', 'locked'
    
    def __init__(self, flat: bool = True, name: str = 'universe', dark_energy: Type[DarkEnergy] = LinearDarkEnergy) -> None:
        super().__init__( density0 = 1.0, w = None, negative_density = False, name = name )

        self.flat   = flat
        self.locked = False

        if not issubclass( dark_energy, DarkEnergy ):
            raise ComponentError("dark_energy must be a subclass of 'DarkEnergy")
        self.dark_energy = dark_energy

    def density(self, z: Any) -> Any:
        density = 0.0
        for child in self.children.values():
            density += child.density( z )
        return density

    def wz(self, z: Any) -> Any:
        return None

    def addchild(self, o: object) -> None:
        if self.locked:
            return warnings.warn("universe is finalized and no changes are made")

        # for flat universe, dark energy and otherwise curvature are added such that the 
        # total density is 1. these cannot be added manually. 
        forbidden_name = 'dark_energy' if self.flat else 'curvature'
        if o.name == forbidden_name:
            raise ComponentError(f"cannot add component: '{ forbidden_name }'")

        return super().addchild( o, raise_error = self.flat )

    def finalize(self) -> None:
        """
        Finalize and lock the object.
        """
        density_left = self.density0 - self.totalAssignedDensity

        if self.flat:
            self.children[ 'curvature' ]   =  Curvature( 0.0 ) 
            self.children[ 'dark_energy' ] = self.dark_energy( density_left ) 
        else:
            self.children[ 'curvature' ]   = Curvature( density_left ) 

        self.locked = True # lock the universe

