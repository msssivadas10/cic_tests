#!/usr/bin/python3

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable


# =====================================================
# Modelling the components in universe
# =====================================================

class Component( ABC ):
    """
    Represents a component of the universe.
    """
    __slots__ = 'name', 'Omega0', 'childComponents', 'negOmega', 'w', 'OmegaID', 'variableW'

    def __init__(self, name: str, Omega0: float, w: float = None, negOmega: bool = False, varw: bool = False, OmegaID: str = 'Omega0') -> None:
        self.name            = name
        self.childComponents = {}
        self.negOmega        = negOmega
        self.variableW       = varw

        if not isinstance( OmegaID, str ):
            raise TypeError("'OmegaID' must be an 'str'")
        self.OmegaID = OmegaID

        if not negOmega and Omega0 < 0.0:
            raise ValueError(f"'{ OmegaID }' cannot be negative")
        self.Omega0, self.w = Omega0, w 
    
    @abstractmethod
    def zDensity(self, z: Any) -> Any:
        """
        Density of this component as function of redshift `z`.
        """
        ...

    @abstractmethod
    def wz(self, z: Any) -> Any:
        """
        Equation of state parameter evolution.
        """
        ...
    
    def zPressure(self, z: Any) -> Any:
        """
        Pressure of this component as a function of redshift 'z'
        """
        return self.wz( z ) * self.zDensity( z ) # with c = 1
    
    def addchild(self, c: 'Component', checkOmega: bool = True) -> 'Component':
        """
        Add a child component to this component.
        """
        if not isinstance( c, Component ):
            raise TypeError( "component must be a 'Component' object" )
        
        if checkOmega:
            # check if the total child Omega0 is less than parent Omega0
            totalOmega0 = c.Omega0 + sum( _c.Omega0 for _c in self.childComponents.values() )
            if totalOmega0 > self.Omega0:
                expr = '+'.join( [ _c.OmegaID for _c in self.childComponents ] + [ c.OmegaID ] )
                raise ValueError(f"'{ expr }' cannot be greater than '{ self.OmegaID }'")
        
        self.childComponents[ c.name ] = c
        return self
    
    def child(self, key: str) -> 'Component':
        """
        Get a child component of this component.
        """
        if key not in self.childComponents:
            raise KeyError(f"'{ key }'")
        return self.childComponents[ key ]

    @property
    def remOmega0(self) -> float:
        """
        Get the remaining `Omega0`, after taking the child components.
        """
        return self.Omega0 - sum( _c.Omega0 for _c in self.childComponents.values() )

    def __repr__(self) -> str:
        childlist = "[{}]".format( ", ".join( map( lambda s: f"'{s}'", self.childComponents ) ) )
        return f"<Component '{ self.name }': { self.OmegaID }={ self.Omega0 }, w={ self.w }, child={ childlist }>"


# ====================================================================
# Models for matter: baryons, massive neutrino and cold dark-matter
# ====================================================================

class Matter( Component ):
    """
    Represnts a matter component, like dark-matter or baryon.
    """

    def __init__(self, Om0: float, name: str = 'matter', OmegaID: str = 'Om0') -> None:
        # matter will have equation of state w = 0 and positive omega
        super().__init__(name, Om0, 0.0, False, False, OmegaID)

    @property 
    def Om0(self) -> float: return self.Omega0

    def wz(self, z: Any) -> Any:
        return np.zeros( np.shape( z ) )

    def zDensity(self, z: Any) -> Any:
        z = np.asfarray( z )
        return self.Omega0 * ( 1+z )**3

class Baryon( Matter ):
    """
    Represents the baryons.
    """

    def __init__(self, Ob0: float) -> None:
        super().__init__(Ob0, 'baryon', 'Ob0')

    @property
    def Ob0(self) -> float: return self.Omega0

class MassiveNeutrino( Matter ):
    """
    Represents the massive neutrinos.
    """
    __slots__ = 'Nnu', 'mnu',  

    def __init__(self, Onu0: float, Nnu: float = None, h: float = None) -> None:
        super().__init__(Onu0, 'massive_nu', 'Onu0')
        
        self.Nnu, self.mnu = 0.0, 0.0
        if Onu0:
            if Nnu <= 0.0:
                raise ValueError("number of neutrinos cannot be negative or zero")
            self.mnu = 91.5*Onu0*h**2 / Nnu
            self.Nnu = Nnu

    @property
    def Onu0(self) -> float: return self.Omega0    

class ColdDarkMatter( Matter ):
    """ 
    Represents the cold dark-matter.
    """

    def __init__(self, Oc0: float) -> None:
        super().__init__(Oc0, 'cdm', 'Oc0')

    @property
    def Oc0(self) -> float: return self.Omega0


# ==================================================================
# dark energy models: cosmological constant and co.
# ==================================================================

class DarkEnergy( Component ):
    """
    Represents the dark-energy.
    """

    def __init__(self, name: str, Ode0: float, w: float) -> None:
        super().__init__(name, Ode0, w, False, 'Ode0')

    @property
    def Ode0(self) -> float: return self.Omega0

class LinearWDarkEnergy( DarkEnergy ):
    """
    Represents a dark-energy component with linearly varying w.
    """

    def __init__(self, Ode0: float, w0: float = -1, wa: float = 0.0) -> None:
        super().__init__( 'de_linear_w', Ode0, [ w0, wa ] )

    @property
    def w0(self) -> float: 
        """
        Constant part of w(z).
        """
        return self.w[0]

    @property
    def wa(self) -> float: 
        """
        Coefficent of the variable part of w(z).
        """
        return self.w[1]

    def wz(self, z: Any) -> Any:
        z = np.asfarray( z )
        w = self.w0 * np.ones_like( z )
        if self.wa:
            w += self.wa * z / ( z+1 )
        return w
    
    def zDensity(self, z: Any) -> Any:
        z = np.asfarray( z )
        return self.Ode0 * ( 1+z )**( 3 + 3*self.wz( z ) )

# =============================================================================
# Curvature component
# =============================================================================
    
class Curvature( Component ):
    """
    Represents the curvature.
    """

    def __init__(self, Omega0: float) -> None:
        super().__init__( 'curvature', Omega0, -1.0/3, True, False, 'Ok0' )

    @property
    def Ok0(self) -> float: return self.Omega0

    def wz(self, z: Any) -> Any:
        return self.w * np.ones( np.shape( z ) )

    def zDensity(self, z: Any) -> Any:
        z = np.asfarray( z )
        if self.Ok0:
            return self.Ok0 * ( z+1 )**2
        return np.zeros_like( z )

# =============================================================================
# The universe!
# =============================================================================

class Universe( Component ):
    """
    Represents a model for universe (Omega = 1).
    """

    def __init__(self, name: str = 'universe') -> None:
        super().__init__( name, 1.0 )

    @property
    def flat(self) -> bool: return not self.child('curvature').Ok0
        
    def zDensity(self, z: Any) -> Any:
        z   = np.asfarray( z )
        out = np.zeros_like( z )
        for c in self.childComponents.values():
            out += c.zDensity( z )
        return out

    def zPressure(self, z: Any) -> Any:
        return NotImplemented

    def wz(self, z: Any) -> Any:
        return None

    def getCurvature(self, flat: bool = True) -> 'Universe':
        """
        Compute the curvature if `flat` is False, else check flatness.
        """
        totalOmega = sum( c.Omega0 for c in self.childComponents.values() )

        if flat:
            if totalOmega != 1.0:
                raise ValueError("total Omega0 must be 1 for a flat universe")
            Ok0 = 0.0
        else:
            Ok0 = self.Omega0 - totalOmega
        
        return self.addchild( Curvature( Ok0 ) )

