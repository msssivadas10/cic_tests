#!\usr\bin\python3

from typing import Any, Callable, List
from pycosmo.cosmology import Cosmology
from scipy.interpolate import CubicSpline
import numpy as np
import pycosmo.constants as const

def press74(sigma: Any):
    r"""
    Fitting function by Press and Schechter (1974). 
    """
    nu = const.DELTA_C / np.asfarray( sigma )
    f  = np.sqrt( 2 / np.pi ) * nu * np.exp( -0.5 * nu**2 )
    return f

def sheth01(sigma: Any):
    r"""
    Fitting function by Sheth et. al. in 2001.
    """
    A = 0.3222
    a = 0.707
    p = 0.3
    nu = const.DELTA_C / np.asarray( sigma )
    f = A * np.sqrt( 2*a / np.pi ) * nu * np.exp( -0.5 * a * nu**2 ) * ( 1.0 + ( 1.0/a/nu**2 )**p )
    return f

def tinker08(sigma: Any, z : float, Delta: float):
    r""" 
    Fitting function by J. Tinker et. al. in 2008[1]. 
    """
    sigma  = np.asarray( sigma )

    if np.ndim( z ):
        raise ValueError("parameter 'z' should be a scalar")
    elif z < -1:
        raise ValueError("redshift 'z' must be greater than -1")

    if Delta < 200 or Delta > 3200:
        raise ValueError('`Delta` value is out of bound. must be within 200 and 3200.')

    # find interpolated values from 0-redshift parameter table : table 2
    A  = CubicSpline(
                        [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                        [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260],
                    )( Delta )
    a  = CubicSpline(
                        [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                        [1.47,  1.52,  1.56,  1.61,  1.87,  2.13,  2.30,  2.53,  2.66 ],
                    )( Delta )
    b  = CubicSpline(
                        [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                        [2.57,  2.25,  2.05,  1.87,  1.59,  1.51,  1.46,  1.44,  1.41 ],
                    )( Delta )
    c  = CubicSpline(
                        [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                        [1.19,  1.27,  1.34,  1.45,  1.58,  1.80,  1.97,  2.24,  2.44 ],
                    )( Delta ) # `c` is time independent

    # redshift evolution of parameters : 
    zp1   = 1 + z
    A     = A / zp1**0.14 # eqn 5
    a     = a / zp1**0.06 # eqn 6  
    alpha = 10.0**( -( 0.75 / np.log10( Delta/75 ) )**1.2 ) # eqn 8    
    b     = b / zp1**alpha # eqn 7 
    
    # eqn 3
    f = A * ( 1 + ( b / sigma )**a ) * np.exp( -c / sigma**2 )
    return f


def mdefParser(value: str, z: float = None, cm: Cosmology = None) -> tuple:
    """
    Parse a mass definition string.
    """
    import re

    m = re.match( r'(\d*)([mc]|[fovir]{3})', value )
    if m is None:
        raise ValueError("cannot parse mass definition: '{}'".format( value ))

    delta, ref = m.groups()
    if ref == 'fof':
        return ( None, 'fof' )

    if ref == 'vir':
        x = cm.Omz( z ) - 1.0
        if cm.flat:
            delta_c = 18.0*np.pi**2 + 82.0*x - 39.0*x**2
        elif cm.Ode0 == 0.0:
            delta_c = 18.0*np.pi**2 + 60.0*x - 32.0*x**2
        else:
            raise ValueError("cannot use 'vir' mass definition")
        return ( round( delta_c ) * cm.criticalDensity( z ), 'so' )
    
    if delta:
        if ref == 'm':
            return ( int( delta ) * cm.Omz( z ), 'so' )
        elif ref == 'c':
            return ( int( delta ) * cm.criticalDensity( z ), 'so' )

    raise ValueError("incorrect mass definition: '{}'".format( value ))

class MassFunction:
    """
    Represents a mass function model.
    """
    __slots__ = '_f', 'allowed_mdef', 'z_dependent', 'cosmology_dependent', 'attrs', 
    
    def __init__(self, f: Callable, zdep: bool = False, mdef: List[str] = [ 'fof' ], cdep: bool = False, **attrs) -> None:
        if not callable( f ):
            raise TypeError("'f' must be a callable")
        self._f    = f

        self.z_dependent         = bool( zdep )
        self.cosmology_dependent = bool( cdep )
        self.allowed_mdef        = mdef
        self.attrs               = attrs
        
    def f(self, sigma: Any, cm: Cosmology = None, z: float = 0.0, mdef: str = 'fof') -> Any:
        """
        Calculate the fitting function for mass function.
        """
        sigma = np.asfarray( sigma )

        if not ( isinstance( cm, Cosmology ) or cm is None ):
            raise TypeError("'cm' must be a 'Cosmology'")

        args = { 'sigma': sigma }

        if self.z_dependent:
            if z < -1.0:
                raise ValueError("redshift 'z' must be greater than -1")
            args[ 'z' ] = z

        delta, mdef = mdefParser( mdef, z, cm )
        if mdef not in self.allowed_mdef:
            raise ValueError("function is not defined for '{}' mass definitions".format( mdef ))

        if mdef != 'fof':
            args[ 'Delta' ] = round( delta / cm.rho_m( z ) )

        if self.cosmology_dependent:
            args[ 'cm' ] = cm

        return self._f( **args )

    def fr(self, r: Any, cm: Cosmology, z: float = 0.0, mdef: str = 'fof', **kwargs) -> Any:
        """
        Calculate the fitting function for mass function.
        """
        if not isinstance( cm, Cosmology ):
            raise TypeError("'cm' must be a 'Cosmology'")
        return self.f( np.sqrt( cm.variance( r, z, **kwargs ) ), cm, z, mdef )

    def fm(self, m: Any, cm: Cosmology = None, z: float = 0.0, mdef: str = 'fof', **kwargs) -> Any:
        """
        Calculate the fitting function for mass function.
        """
        return self.fr( cm.lagrangianR( m ), cm, z, mdef, **kwargs )

    def dndlnm(self, m: Any, cm: Cosmology, z: float = 0.0, mdef: str = 'fof', **kwargs) -> Any:
        """
        Calculate the halo mass-function.
        """
        m = np.asfarray( m )
        r = cm.lagrangianR( m )
        
        return (
                    self.fr( r, cm, z, mdef, **kwargs )
                        * ( cm.rho_m( z ) / m )
                        * ( -cm.dlnsdlnm( r, z, **kwargs ) )
               )

    def dndm(self, m: Any, cm: Cosmology, z: float = 0.0, mdef: str = 'fof', **kwargs) -> Any:
        """
        Calculate the halo mass-function.
        """
        return self.dndlnm( m, cm, z, mdef ) / np.asfarray( m )

    def __call__(self, m: Any, cm: Cosmology, z: float = 0.0, mdef: str = 'fof', mode: str = 'dndlnm', **kwargs) -> Any:
        """
        Calculate the halo mass-function.
        """
        if mode == 'f':
            return self.fm( m, cm, z, mdef, **kwargs )
        elif mode == 'dndlnm':
            return self.dndlnm( m, cm, z, mdef, **kwargs )
        elif mode == 'dndm':
            return self.dndm( m, cm, z, mdef, **kwargs )
        raise ValueError("invalid value for 'mode': '{}'".format( mode ))

_functab = {
                'press74' : MassFunction( 
                                            press74, 
                                            zdep = False, 
                                            mdef = [ 'fof' ], 
                                            cdep = False 
                                        ),
                'sheth01' : MassFunction( 
                                            sheth01, 
                                            zdep = False, 
                                            mdef = [ 'fof' ], 
                                            cdep = False 
                                        ),
                'tinker08': MassFunction( 
                                            tinker08, 
                                            zdep = True, 
                                            mdef = [ 'so' ], 
                                            cdep = False 
                                        ),
           }

def available(model: str) -> bool:
    """
    Check if a mass-function model is available.
    """
    return ( model in _functab.keys() )

def massFunction(m: Any, cm: Cosmology, z: float = 0.0, mdef: str = 'fof', model: str = 'tinker08', mode: str = 'dndlnm', **kwargs) -> Any:
    """
    Compute the halo mass-function of given model.
    """
    if not available( model ):
        raise ValueError("model not available: '{}'".format( model ))
    mf = _functab[ model ]
    return mf( m, cm, z, mdef, mode, **kwargs )
        


        

        


    

