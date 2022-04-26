#!/usr/bin/python3

import numpy as np
from typing import Any
from itertools import repeat

def simps(y: Any, dx: Any = None, axis: int = -1) -> Any:
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
    
    retval = (
                y[ i1 ].sum( axis ) + 4.0*y[ i2 ].sum( axis ) + y[ i3 ].sum( axis )
             ) / 3.0
    if dx is not None:
        return retval * dx
    return retval
