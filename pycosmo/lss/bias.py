#!/usr/bin/python3

import numpy as np
import pycosmo.utils.constants as const
from typing import Any

def bias_Cole89(nu: Any):
    r"""
    Fitting function given by Cole & Kaiser (1989) and Mo & White (1996).
    """
    nu = np.asfarray( nu )
    return 1.0 + ( nu**2 - 1.0 ) / const.DELTA_C

def bias_Sheth01(nu: Any):
    r"""
    Fitting function given by Sheth et al. (2001).
    """
    nu = np.asfarray( nu )
    a  = 0.707
    b  = 0.5
    c  = 0.6
    sqrt_a = np.sqrt( a )
    anu2   = a * nu**2
    return (
                1.0 + 1.0 / sqrt_a / const.DELTA_C 
                    * ( 
                            sqrt_a * anu2 + sqrt_a * b * anu2**( 1-c ) 
                                - anu2**c / ( anu2**c + b * ( 1-c ) * ( 1-0.5*c ) )
                      )
           )

def bias_Tinker10(nu: Any, Delta: float):
    r"""
    Fitting function given by Tinker et al. (2010).
    """
    nu = np.asfarray( nu )
    y  = np.log10( Delta )
    A  = 1.0 + 0.24 * y * np.exp( -( 4.0/y )**4 )
    a  = 0.44 * y - 0.88
    B  = 0.183
    b  = 1.5
    C  = 0.019 + 0.107 * y + 0.19 * np.exp( -( 4.0/y )**4 )
    c  = 2.4
    return 1.0 - A * nu**a / ( nu**a + const.DELTA_C**a ) + B * nu**b + C * nu**c

def bias_Jing98(nu: Any, z: float, cm: object) -> Any:
    """
    Fitting function by Jing (1998).
    """
    nu   = np.asfarray( nu )
    r    = cm.radius( const.DELTA_C / nu, z )
    neff = cm.effectiveIndex( 2*np.pi / r, z )
    return ( 0.5 / nu**4 + 1 )**( 0.06 - 0.02*neff ) * ( 1 + ( nu**2 - 1 ) / const.DELTA_C )


def bias_Seljak04(nu: Any, z: float, cm: object, correction: bool = False) -> Any:
    """
    Fitting function by Seljak et al (2004).
    """
    nu = np.asfarray( nu )

    r, rstar  = cm.radius( nu / const.DELTA_C, z ), cm.radius( 1.0, z )

    x = ( r / rstar )**3
    b = 0.53 + 0.39*x**0.45 + 0.13 / ( 40.0*x + 1 ) + 5E-4*x**1.5

    if correction:
        alpha_s = 0.0
        if cm.sigma8 is None:
            raise ValueError("'sigma8' parameter is not set")

        b += np.log10( x ) * (
                0.4*( cm.Om0 - 0.3 + cm.ns - 1.0 ) + 0.3*( cm.sigma8 - 0.9 + cm.h - 0.7 ) + 0.8*alpha_s
        )
    
    return b


