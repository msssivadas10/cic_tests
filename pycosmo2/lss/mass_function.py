from typing import Any
import numpy as np
import pycosmo2.utils.constants as const
import pycosmo2._bases as base

def mfmodelPress74(sigma: Any):
    r"""
    Fitting function by Press and Schechter (1974). 
    """
    nu = const.DELTA_C / np.asfarray( sigma )
    f  = np.sqrt( 2 / np.pi ) * nu * np.exp( -0.5 * nu**2 )
    return f

def mfmodelSheth01(sigma: Any):
    r"""
    Fitting function by Sheth et al (2001).
    """
    A = 0.3222
    a = 0.707
    p = 0.3
    nu = const.DELTA_C / np.asarray( sigma )
    f = A * np.sqrt( 2*a / np.pi ) * nu * np.exp( -0.5 * a * nu**2 ) * ( 1.0 + ( nu**2 / a )**-p )
    return f

def mfmodelTinker08(sigma: Any, z : float, Delta: float):
    r""" 
    Fitting function by J. Tinker et al(2008). 
    """
    sigma  = np.asarray( sigma )

    if np.ndim( z ):
        raise ValueError("parameter 'z' should be a scalar")
    elif z < -1:
        raise ValueError("redshift 'z' must be greater than -1")

    if Delta < 200 or Delta > 3200:
        raise ValueError('`Delta` value is out of bound. must be within 200 and 3200.')

    # find interpolated values from 0-redshift parameter table : table 2
    from scipy.interpolate import CubicSpline

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

def mfmodelJenkins01(sigma: Any) -> Any:
    """
    Fitting function by Jenkins et al. (2001).
    """
    sigma = np.asfarray( sigma )
    return 0.315*( -np.abs( np.log( sigma**-1 ) + 0.61 )**3.8 )

def mfmodelReed03(sigma: Any) -> Any:
    """
    Fitting function by Reed et al. (2003).
    """
    sigma = np.asfarray( sigma )
    return mfmodelSheth01( sigma ) * np.exp( -0.7 / ( sigma * np.cosh( 2*sigma )**5 ) )

def mfmodelReed07(sigma: Any, z: float, cm: base.Cosmology) -> Any:
    """
    Fitting function by Reed et al. (2007).
    """
    A, c, ca, p = 0.310, 1.08, 0.764, 0.3

    sigma = np.asfarray( sigma )
    omega = np.sqrt( ca ) * const.DELTA_C / sigma

    G1    = np.exp( -0.5*( np.log( omega ) - 0.788 )**2 / 0.6**2 )
    G2    = np.exp( -0.5*( np.log( omega ) - 1.138 )**2 / 0.2**2 )

    r     = cm.radius( sigma, z )
    neff  = -6.0*cm.dlnsdlnm( r, 0.0 ) - 3.0
    
    # eqn. 12
    f = (
            A * omega * np.sqrt( 2.0/np.pi ) 
                * np.exp( -0.5*omega - 0.0325*omega**p / ( neff + 3 )**2 )
                * ( 1.0 + 1.02*omega**( 2*p ) + 0.6*G1 + 0.4*G2 )
        )
    return f

def mfmodelWarren06(sigma: Any) -> Any:
    """
    Fitting function by Warren et al (2006).
    """
    A, a, b, c = 0.7234, 1.625, 0.2538, 1.1982
    sigma = np.asfarray( sigma )
    return A * ( sigma**-a + b ) * np.exp( -c / sigma**2 )

def mfmodelCrocce10(sigma: Any, z: float) -> Any:
    """
    Fitting function by Crocce et al (2010).
    """
    Az = 0.580*( z + 1 )**-0.130
    az = 1.370*( z + 1 )**-0.150
    bz = 0.300*( z + 1 )**-0.084
    cz = 1.036*( z + 1 )**-0.024
    return Az * ( sigma**-az + bz ) * np.exp( -cz / sigma**2 )

def mfmodelCourtin10(sigma: Any) -> Any:
    """
    Fitting function by Courtin et al (2010).
    """
    A  = 0.348
    a  = 0.695
    p  = 0.1
    nu = const.DELTA_C / np.asarray( sigma )
    f  = A * np.sqrt( 2*a / np.pi ) * nu * np.exp( -0.5 * a * nu**2 ) * ( 1.0 + ( nu**2 / a )**-p )
    return f