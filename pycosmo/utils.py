#!\usr\bin\python3

import numpy as np
from typing import Any, Callable 
from itertools import repeat

# from scipy.interpolate import CubicSpline
# from scipy.optimize import newton

def simps(y: Any, dx: Any = 1.0, axis: int = -1) -> Any:
    """
    Simpsons rule integration.
    """
    y   = np.asfarray( y )
    if np.ndim( y ) < 1:
        raise TypeError("'y' should be at least 1 dimensional")

    pts = y.shape[ axis ]
    if ( pts < 2 ) or not ( pts & 1 ):
        raise ValueError("number of points must be an odd number greater than 3: '{}'".format( pts ))
    
    i1, i2, i3 = repeat( list( repeat( slice( None ), y.ndim ) ), 3 )

    i1[ axis ] = slice( None, -1,   2 )
    i2[ axis ] = slice( 1,    None, 2 )
    i3[ axis ] = slice( 2,    None, 2 )

    i1, i2, i3 = tuple( i1 ), tuple( i2 ), tuple( i3 )
    
    return (
                y[ i1 ].sum( axis ) + 4.0*y[ i2 ].sum( axis ) + y[ i3 ].sum( axis )
           ) * ( dx / 3.0 )


########################################################################################################


class PyCosmoFunction:
    """
    A function with specific attributes. 
    """
    __slots__ = 'f', 'attrs', 'fclass', 'z_dependent', 'cosmo_dependent', 'mdef', 

    _fclasses = ( 'any', 'bias', 'power', 'hmf' )

    def __init__(self, f: Callable, fclass: str = 'any', z: bool = False, cosmo: bool = False, mdef: list = None, **attrs) -> None:
        if not callable( f ):
            raise TypeError("'f' must be a callable")
        self.f     = f

        if fclass not in self._fclasses:
            raise ValueError("invalid function class: '{}'".format( fclass ))
        self.fclass = fclass

        self.z_dependent = z

        self.cosmo_dependent = cosmo
        if fclass == 'power':
            self.cosmo_dependent = True

        self.mdef = None
        if mdef is not None:
            if fclass not in [ 'bias', 'hmf' ]:
                raise KeyError("'mdef' attribute is only used with 'bias' and 'hmf' functions")

            if not isinstance( mdef, list ):
                raise TypeError("mdef must be a 'list'")
            elif False in map( lambda o: isinstance( o, str ), mdef ):
                raise TypeError("mdef must be a list of 'str'")
            self.mdef = mdef

        self.attrs = attrs

    def __call__(self, *args, **kwargs) -> Any:
        return self.f( *args, **kwargs )

class FunctionTable:
    """
    A function table storing functions as key-value pairs.
    """
    __slots__ = 'name', 'ftab', 'map',

    def __init__(self, name: str) -> None:
        self.ftab = {}
        self.map  = {}
        self.name = name

    def keys(self) -> tuple:
        """
        Get the keys in the table.
        """
        return tuple( self.ftab.keys() )

    def exists(self, key: str) -> bool:
        """
        Check if a key exist or not.
        """
        return ( key in self.keys() ) or ( key in self.map.keys() )

    def __getitem__(self, key: str) -> PyCosmoFunction:
        if key in self.map.keys():
            key = self.map[ key ]
        if not self.exists( key ):
            raise KeyError("'{}'".format( key ))
        return self.ftab[ key ]

    def __setitem__(self, key: str, f: PyCosmoFunction) -> None:
        if not isinstance( key, str ):
            raise TypeError("key must be an 'str'")
        if not isinstance( f, PyCosmoFunction ):
            raise TypeError("'f' must be a 'PyCosmoFuncion'")
        if key in self.map.keys():
            key = self.map[ key ]
        self.ftab[ key ] = f

    def setMapping(self, mapping: dict) -> None:
        """
        Set mapping for keys.
        """
        for key1, key2 in mapping.items():
            if not ( isinstance( key1, str ) and isinstance( key2, str ) ):
                raise TypeError("keys must be 'str'")
            if not self.exists( key2 ):
                raise KeyError("'{}'".format( key2 ))
            self.map[ key1 ] = key2
            

def addfunction(key: str, to: FunctionTable, fclass: str = 'any', z: bool = False, cosmo: bool = False, mdef: list = None, **attrs) -> Callable:
    """
    Decorator to add a function to a function table.
    """
    def _addfunction(f: Callable) -> Callable:
        if not to.exists( key ):
            to[ key ] = PyCosmoFunction(f, fclass, z, cosmo, mdef, **attrs)
        return f

    return _addfunction
