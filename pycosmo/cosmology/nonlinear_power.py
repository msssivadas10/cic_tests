#!/usr/bin/python3

import numpy as np
from typing import Any

def nlpmodelHalofit(cm: Any, k: Any, z: float = 0.0) -> Any:
    """
    Non-linear power spectrum using **halofit**.
    """
    rstar = cm.radius( 1.0, z, lin = True ) # 1/k_sigma in eqn. A4 ( with default filter )

    # eqn. A5
    neff  = -2.0 * cm.dlnsdlnr( rstar, z, lin = True ) - 3.0 
    C     = -2.0 * cm.d2lnsdlnr2( rstar, z, lin = True )     

    Om, Ow, w = cm.Om( z ), cm.Ode( z ), cm.wde( z )

    # best-fit parameters: eqn A6-13
    an = 10**( 
                1.5222 + 2.8553 * neff + 2.3706 * neff**2 + 0.9903 * neff**3 + 0.2250 * neff**4 
                    - 0.6038 * C + 0.1749 * Ow * ( 1 + w )
             )
    bn = 10**(
                -0.5642 + 0.5864 * neff + 0.5716 * neff**2 - 1.5474 * C + 0.2279 * Ow * ( 1 + w )
             )
    cn = 10**(
                0.3698 +2.0404 * neff + 0.8161 * neff**2 + 0.5869 * C
             )
    
    gamma_n = 0.1971 - 0.0843 * neff + 0.8460 * C 
    alpha_n = np.abs( 6.0835 + 1.3373 * neff - 0.1959 * neff**2 - 5.5274 * C )
    beta_n  = 2.0379 - 0.7354 * neff + 0.3157 * neff**2 + 1.2490 * neff**3 + 0.3980 * neff**4 - 0.1682 * C 
    mu_n    = 0.0
    nu_n    = 10**( 5.2105 + 3.6902 * neff )

    f1, f2, f3 = Om**-0.0307, Om**-0.0585, Om**0.0743 # eqn. A14

    y     = np.asfarray( k ) * rstar

    # two-halo term: eqn. A2
    dlin    = cm.linearPowerSpectrum( k, z, dim = False )
    fy      = 0.25 * y + 0.125 * y**2
    delta2Q = dlin * ( ( 1 + dlin )**beta_n / ( 1 + alpha_n * dlin ) ) * np.exp( -fy )

    # one-halo term: eqn. A3
    delta2H = an * y**( 3 * f1 ) / ( 1 + bn * y**f2 + ( cn * f3 * y )**( 3 - gamma_n ) )
    delta2H = delta2H / ( 1 + mu_n * y**-1 + nu_n * y**-2 ) 

    return delta2Q + delta2H

def nlpmodelPeacock(cm: Any, k: Any, z: float) -> Any:
    """
    Non-linear power spectrum by Peacock and Dodds formula.
    """
    neff = cm.effectiveIndex( k, z, lin = True )
    n3p1 = 1 + neff / 3
    mask = ( n3p1 > 0 ) # else, for neff < -3, power becomes complex

    dnl  = np.zeros_like( neff )
    g    = cm.gz( z )
    
    dnl[ mask ] = cm.linearPowerSpectrum( k[ mask ], z, dim = False )

    # best-fit parameters: eqn. 23-27
    A     = 0.486 * n3p1[ mask ]**-0.947
    B     = 0.266 * n3p1[ mask ]**-1.778
    alpha = 3.310 * n3p1[ mask ]**-0.244
    beta  = 0.862 * n3p1[ mask ]**-0.287
    V     = 11.55 * n3p1[ mask ]**-0.423

    dnl[ mask ] = dnl[ mask ] * (
                                    ( 1 + B * beta * dnl[ mask ] + ( A * dnl[ mask ] )**( alpha * beta ) )
                                        / ( 1 + ( 
                                                    ( A * dnl[ mask ] )**alpha * g**3 / ( V * dnl[ mask ]**0.5 ) 
                                                )**beta 
                                          )
                                )**( 1/beta )
    return dnl

