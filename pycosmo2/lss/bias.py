from typing import Any, Union 
from pycosmo2.lss._flags import *
import numpy as np
import pycosmo2._bases as base
import pycosmo2.lss.overdensity as od

OverDensityType = Union[int, str, od.OverDensity]

class LinearBias(base.LinearBias):
    r"""
    Class representing a linear bias model.
    """

    def __init__(self, cm: base.Cosmology) -> None:
        if not isinstance( cm, base.Cosmology ):
            raise TypeError("cm must be a 'Cosmology' object")
        self.cosmology = cm

        self.flags       = 0
        self.corrections = False


###############################################################################################

class Cole89(LinearBias):
    r"""
    Linear bias model given by Cole & Kaiser (1989) and Mo & White (1996).
    """

    def b(self, nu: Any, z: float = 0, overdensity: OverDensityType = None) -> Any:
        delta_c = self.cosmology.collapseOverdensity( z )

        nu   = np.asfarray( nu )
        bias = 1.0 + ( nu**2 - 1.0 ) / self.cosmology.collapseOverdensity( z )
        return bias

class Sheth01(LinearBias):
    r"""
    Linear bias model given by Sheth et al. (2001).
    """

    def b(self, nu: Any, z: float = 0, overdensity: OverDensityType = None) -> Any:
        delta_c = self.cosmology.collapseOverdensity( z )

        nu = np.asfarray( nu )
        a  = 0.707
        b  = 0.5
        c  = 0.6
        sqrt_a = np.sqrt( a )
        anu2   = a * nu**2
        
        bias = (
                    1.0 + 1.0 / sqrt_a / delta_c
                        * ( 
                                sqrt_a * anu2 
                                    + sqrt_a * b * anu2**( 1-c ) 
                                    - anu2**c / ( anu2**c + b * ( 1-c ) * ( 1-0.5*c ) )
                          )
               )
        return bias

class Jing98(LinearBias):
    """
    Linear bias model by Jing (1998).
    """

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = Z_DEPENDENT | COSMO_DEPENDENT

    def b(self, nu: Any, z: float = 0, overdensity: OverDensityType = None) -> Any:
        delta_c = self.cosmology.collapseOverdensity( z )

        nu   = np.asfarray( nu )
        r    = self.cosmology.radius( delta_c / nu, z )
        neff = self.cosmology.effectiveIndex( 2*np.pi / r, z )
        bias = ( 0.5 / nu**4 + 1 )**( 0.06 - 0.02*neff ) * ( 1 + ( nu**2 - 1 ) / delta_c )
        return bias

class Seljak04(LinearBias):
    """
    Linear bias model by Seljak et al (2004).
    """

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = Z_DEPENDENT | COSMO_DEPENDENT | HAS_CORRECTION

    def b(self, nu: Any, z: float = 0, overdensity: OverDensityType = None) -> Any:
        delta_c = self.cosmology.collapseOverdensity( z )

        nu    = np.asfarray( nu )
        r     = self.cosmology.radius( nu / delta_c, z )
        rstar = self.cosmology.radius( 1.0, z )
        x     = ( r / rstar )**3
        
        bias  = 0.53 + 0.39*x**0.45 + 0.13 / ( 40.0*x + 1 ) + 5E-4*x**1.5

        if self.corrections:
            Om0    = self.cosmology.Om0
            h      = self.cosmology.h
            ns     = self.cosmology.ns
            sigma8 = self.cosmology.sigma8

            alpha_s = 0.0
            bias   += np.log10( x ) * (
                                            0.4*( Om0 - 0.3 + ns - 1.0 ) 
                                                + 0.3*( sigma8 - 0.9 + h - 0.7 ) 
                                                + 0.8*alpha_s
                                      )
        return bias

class Tinker10(LinearBias):
    r"""
    Linear bias model given by Tinker et al. (2010).
    """

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = SO_OVERDENSITY 

    def b(self, nu: Any, z: float = 0, overdensity: OverDensityType = None) -> Any:
        delta_c = self.cosmology.collapseOverdensity( z )

        if overdensity is None:
            overdensity = '200m'
        overdensity = od.overdensity( overdensity ).value

        nu = np.asfarray( nu )
        y  = np.log10( overdensity )
        A  = 1.0 + 0.24 * y * np.exp( -( 4.0/y )**4 )
        a  = 0.44 * y - 0.88
        B  = 0.183
        b  = 1.5
        C  = 0.019 + 0.107 * y + 0.19 * np.exp( -( 4.0/y )**4 )
        c  = 2.4
        
        bias = 1.0 - A * nu**a / ( nu**a + delta_c**a ) + B * nu**b + C * nu**c
        return bias

models = {
            'cole89'  : Cole89, 
            'sheth01' : Sheth01,
            'jing98'  : Jing98,
            'seljak04': Seljak04,
            'tinker10': Tinker10,
         }