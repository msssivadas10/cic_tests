from typing import Any 
import numpy as np
import numpy.fft as fft
import pycosmo.nbodytools.estimators.density as density

from scipy.stats import binned_statistic

def powerSpectrum(pos: Any, boxsize: float, gridsize: int, mass: Any = 1.0, bins: int = 21) -> Any:
    r"""
    Estimate the power spectrum from particles position data.

    Parameters
    ----------
    pos: array_like of shape (N, 3)
        Particle positions.
    boxsize: float  
        Size of the bounding box of the particles.
    gridsize: int
        Size of the density mesh on each dimension.
    mass: array_like, optional  
        Particle mass. Must be a scalar or 1D array of same length as particles.
    bins: int, optional
        Number bins in k-space (default is 21).

    Returns
    -------
    k: array_like
        Wavenumbers in (unit of boxsize)^-1. Has length equal to `bins`.
    power: array_like
        Power spectrum values correspond to wavenumbers. 

    """
    delta = density.densityCloudInCell( pos, boxsize, gridsize, mass )
    delta = delta / np.mean( delta ) - 1.0
    delta = fft.rfftn( delta ).flatten()

    # make the k grid
    kf    = 2*np.pi / boxsize
    kn    = gridsize * np.pi / boxsize
    nHalf = gridsize // 2

    kx = np.fromfunction( lambda i,j,k: i, shape = ( gridsize, gridsize, nHalf+1 ) ).flatten()
    ky = np.fromfunction( lambda i,j,k: j, shape = ( gridsize, gridsize, nHalf+1 ) ).flatten()
    kz = np.fromfunction( lambda i,j,k: k, shape = ( gridsize, gridsize, nHalf+1 ) ).flatten()

    kx[ kx > nHalf ] -= gridsize
    ky[ ky > nHalf ] -= gridsize
    kz[ kz > nHalf ] -= gridsize
    kx, ky, kz        = kx*kf, ky*kf, kz*kf

    # cic deconvolution
    # w     = np.sinc( kx / kf / gridsize ) * np.sinc( ky / kf / gridsize ) * np.sinc( kz / kf / gridsize )
    # delta = delta / w**2

    power = np.abs( delta )**2 / boxsize**3

    lnk   =  kx**2 + ky**2 + kz**2
    mask  = ( lnk != 0.0 )
    lnk   = 0.5 * np.log( lnk[ mask ] )
    power = power[ mask ]

    lnkn  = np.log( kn )

    mask       = ( lnk <= lnkn )
    lnk, power = lnk[ mask ], power[ mask ]

    power, lnk, _ = binned_statistic( lnk, power, statistic = 'mean', bins = bins )
    
    k = np.exp( 0.5 * ( lnk[:-1] + lnk[1:] ) )

    return k, power

