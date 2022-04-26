#!/usr/bin/python3

import numpy as np
from typing import Any

def tophat(x: Any, j: int = 0) -> Any:
    """
    A spherical tophat filter in k-space.
    """
    x = np.asfarray( x )
    if j == 0:
        return ( np.sin( x ) - x * np.cos( x ) ) * 3.0 / x**3 
    elif j == 1:
        return ( ( x**2 - 3.0 ) * np.sin( x ) + 3.0 * x * np.cos( x ) ) * 3.0 / x**4
    elif j == 2:
        return ( ( x**2 - 12.0 ) * x * np.cos( x ) - ( 5*x**2 - 12.0 ) * np.sin( x ) ) * 3.0 / x**5
    raise ValueError(f"invalid value for 'j': { j }")

def gauss(x: Any, j: int = 0) -> Any:
    """
    Gaussian filter in k-space.
    """
    x = np.asfarray( x )
    if j == 0:
        return np.exp( -0.5*x**2 )
    elif j == 1:
        return -x*np.exp( -0.5*x**2 )
    elif j == 2:
        return ( x**2 - 1 )*np.exp( -0.5*x**2 )
    raise ValueError(f"invalid value for 'j': { j }")

def sharpk(x: Any, j: int = 0) -> Any:
    """
    Sharp-k filter in k-space.
    """
    raise NotImplementedError("function not implemented!")

def filter(x: Any, j: int = 0, model: str = 'tophat') -> Any:
    """
    Filter function in k-space.
    """
    x = np.asfarray( x )
    if model == 'tophat':
        return tophat( x, j )
    elif model == 'gauss':
        return gauss( x, j )
    elif model == 'sharpk':
        return sharpk( x, j )
    raise ValueError(f"invalid filter: '{ model }'")