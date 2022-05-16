r"""

Bias Module: Linear Bias Models
===============================

This module defines the abstract class :class:`LinearBias`, which will be base for all linear bias models. 
Also, some models available in the literature are implemented in this module. Linear bias :math:`b_1(M)` 
relates the overdensity in the matter field to that of halos as :math:`\delta_h = b_1 \delta`.

Available linear bias models are

.. table::

    =================== =====================================================
    Key                 Model
    =================== =====================================================
    `cole89`            Cole & Kaiser (1989), Mo & White (1996)
    `sheth01`           Sheth et al (2001)
    `jing98`            Jing (1998)
    `seljak04`          Seljak et al (2004)
    `tinker10`          Tinker et al (2010)
    =================== =====================================================

"""

from typing import Any, Union 
from pycosmo.lss._flags import *
import numpy as np
import pycosmo._bases as base
import pycosmo.lss.overdensity as od

OverDensityType = Union[int, str, od.OverDensity]

class LinearBias(base.LinearBias):
    r"""
    Class representing a linear bias model.

    Parameters
    ----------
    cm: Cosmology
        Cosmology model to use.

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
    Linear bias model given by Cole & Kaiser (1989) [1]_ and Mo & White (1996) [2]_.

    Parameters
    ----------
    cm: Cosmology
        Cosmology model to use.

    References
    ----------
    .. [1] Shaun Cole and Nick Kaiser. Biased clustering in the cold dark matter cosmogony. Mon. Not.R. astr. Soc. 
            237, 1127-1146 (1989).
    .. [2] H. J. Mo, Y. P. Jing and S. D. M. White. High-order correlations of peaks and haloes: a step towards
            understanding galaxy biasing. Mon. Not. R. Astron. Soc. 284, 189-201 (1997).

    """

    def b(self, nu: Any, z: float = 0, overdensity: OverDensityType = None) -> Any:
        delta_c = self.cosmology.collapseOverdensity( z )

        nu   = np.asfarray( nu )
        bias = 1.0 + ( nu**2 - 1.0 ) / self.cosmology.collapseOverdensity( z )
        return bias

class Sheth01(LinearBias):
    r"""
    Linear bias model given by Sheth et al. (2001). For the functional form, see, for example [1]_. 

    Parameters
    ----------
    cm: Cosmology
        Cosmology model to use.

    References
    ----------
    .. [1] Jeremy L. Tinker et al. The large scale bias of dark matter halos: Numerical calibration and model tests. 
            <http://arxiv.org/abs/1001.3162v2> (2010).
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

    Parameters
    ----------
    cm: Cosmology
        Cosmology model to use.

    References
    ----------
    .. [1] Y. P. Jing. Accurate fitting formula for the two-point correlation function of the dark matter halos. 
            <http://arXiv.org/abs/astro-ph/9805202v2> (1998).

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
    Linear bias model by Seljak et al (2004) [1]_.

    Parameters
    ----------
    cm: Cosmology
        Cosmology model to use.

    References
    ----------
    .. [1] Uroš Seljak & Michael S. Warren. Large scale bias and stochasticity of halos and dark matter. 
            <http://arxiv.org/abs/astro-ph/0403698v3> (2004).

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

    Parameters
    ----------
    cm: Cosmology
        Cosmology model to use.

    References
    ----------
    .. [1] Jeremy L. Tinker et al. The large scale bias of dark matter halos: Numerical calibration and model tests. 
            <http://arxiv.org/abs/1001.3162v2> (2010).

    """

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = SO_OVERDENSITY 

    def b(self, nu: Any, z: float = 0, overdensity: OverDensityType = None) -> Any:
        delta_c = self.cosmology.collapseOverdensity( z )

        if overdensity is None:
            overdensity = '200m'
        overdensity = od.overdensity( overdensity ).value( z, self.cosmology )

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