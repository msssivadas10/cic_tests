#!/usr/bin/python3
r"""

Density Estimators
==================

to do
"""

import numpy as np, numpy.fft as fft
from scipy.stats import binned_statistic
from typing import Any 


############################################################################################
# Density estimators
############################################################################################

def ngpDensity(pos: Any, boxsize: float, gridsize: int, mass: Any = 1.0) -> Any:
    r"""
    Nearest grid point density estimation. Assume the particles as points and each particle contribute 
    only to the cell it lies.

    Parameters
    ----------
    pos: array_like of shape (N,3)
        Particle positions.
    boxsize: float
        Size of the bounding box for particles.
    gridsize: int
        Size of the density mesh along each dimension (G).
    mass: array_like, optional
        Mass of the perticle. If not scalar, must be a 1D array of length N.

    Returns
    -------
    density: array_like of shape (G,G,G)
        Density mesh. This will be a 3D array with the value at index (i,j,k) correspond to the density 
        at that cell.

    """

    pos = np.asfarray( pos )
    if np.ndim( pos ) != 2:
        raise TypeError("'pos' must be a 2 dimensional array")
    if pos.shape[1] != 3:
        raise TypeError("'pos' should have 3 columns")
    
    mass = np.asfarray( mass )
    if np.ndim( mass ) == 1:
        if len( mass ) != pos.shape[0]:
            raise TypeError("'mass' array should have same length as particles")
    elif np.ndim( mass ) > 1:
        raise TypeError("'mass' must be a scalar or 1D array")

    cellsize = np.asfarray( boxsize ) / gridsize

    i = np.floor( pos / cellsize )

    _range = [ (0., gridsize), (0., gridsize), (0., gridsize) ]

    dens = np.histogramdd(
                            np.hstack([ i[:,0:1], i[:,1:2], i[:,2:3] ]),
                            bins    = gridsize,
                            range   = _range,
                            weights = mass,
                         )[0]
    return dens

def cicDensity(pos: Any, boxsize: float, gridsize: int, mass: Any = 1.0) -> Any:
    r"""
    Cloud in cells density estimation. Assume the particles as boxes of one grid cell size and add 
    the contributions of each particle to the cell it lies and nearby cells.

    Parameters
    ----------
    pos: array_like of shape (N,3)
        Particle positions.
    boxsize: float
        Size of the bounding box for particles.
    gridsize: int
        Size of the density mesh along each dimension (G).
    mass: array_like, optional
        Mass of the perticle. If not scalar, must be a 1D array of length N.

    Returns
    -------
    density: array_like of shape (G,G,G)
        Density mesh. This will be a 3D array with the value at index (i,j,k) correspond to the density 
        at that cell.

    """

    pos = np.asfarray( pos )
    if np.ndim( pos ) != 2:
        raise TypeError("'pos' must be a 2 dimensional array")
    if pos.shape[1] != 3:
        raise TypeError("'pos' should have 3 columns")
    
    mass = np.asfarray( mass )
    if np.ndim( mass ) == 1:
        if len( mass ) != pos.shape[0]:
            raise TypeError("'mass' array should have same length as particles")
    elif np.ndim( mass ) > 1:
        raise TypeError("'mass' must be a scalar or 1D array")
        

    cellsize = np.asfarray( boxsize ) / gridsize

    d = pos / cellsize
    i = np.floor( d )
    d = d - i
    t = 1.0 - d
    j = np.where( d > 0.5, i+1, i-1 ) % gridsize

    _range = [ (0., gridsize), (0., gridsize), (0., gridsize) ]

    dens = (
                np.histogramdd(
                                    np.hstack([ i[:,0:1], i[:,1:2], i[:,2:3] ]),
                                    bins    = gridsize,
                                    range   = _range,
                                    weights = mass * t[:,0] * t[:,1] * t[:,2],
                              )[0]
                + np.histogramdd(
                                    np.hstack([ i[:,0:1], i[:,1:2], j[:,2:3] ]),
                                    bins    = gridsize,
                                    range   = _range,
                                    weights = mass * t[:,0] * t[:,1] * d[:,2],
                                )[0]
                + np.histogramdd(
                                    np.hstack([ i[:,0:1], j[:,1:2], i[:,2:3] ]),
                                    bins    = gridsize,
                                    range   = _range,
                                    weights = mass * t[:,0] * d[:,1] * t[:,2],
                                )[0]
                + np.histogramdd(
                                    np.hstack([ i[:,0:1], j[:,1:2], j[:,2:3] ]),
                                    bins    = gridsize,
                                    range   = _range,
                                    weights = mass * t[:,0] * d[:,1] * d[:,2],
                                )[0]
                + np.histogramdd(
                                    np.hstack([ j[:,0:1], i[:,1:2], i[:,2:3] ]),
                                    bins    = gridsize,
                                    range   = _range,
                                    weights = mass * d[:,0] * t[:,1] * t[:,2],
                                )[0]
                + np.histogramdd(
                                    np.hstack([ j[:,0:1], i[:,1:2], j[:,2:3] ]),
                                    bins    = gridsize,
                                    range   = _range,
                                    weights = mass * d[:,0] * t[:,1] * d[:,2],
                                )[0]
                + np.histogramdd(
                                    np.hstack([ j[:,0:1], j[:,1:2], i[:,2:3] ]),
                                    bins    = gridsize,
                                    range   = _range,
                                    weights = mass * d[:,0] * d[:,1] * t[:,2],
                                )[0]
                + np.histogramdd(
                                    np.hstack([ j[:,0:1], j[:,1:2], j[:,2:3] ]),
                                    bins    = gridsize,
                                    range   = _range,
                                    weights = mass * d[:,0] * d[:,1] * d[:,2],
                                )[0]
            )

    return dens


##################################################################################################
# Interpolation methods
##################################################################################################

def ngpInterpolate(value: Any, pos: Any, boxsize: float) -> Any:
    r"""
    Interpolate the values at grid points to given positions using nearest grid point scheme. 

    Parameters
    ----------
    value: array_like of shape (G,G,G)
        Values of the quantity to be interpolated, on grid positions. Must be a 3D array.
    pos: array_like of shape (N,3)
        Particle positions.
    boxsize: float
        Size of bounding box of the particles.

    Returns
    -------
    val: array_like of length N
        Values interpolated to particle positions. Will be a 1D array.
    
    """

    pos = np.asfarray( pos )
    if np.ndim( pos ) != 2:
        raise TypeError("'pos' must be a 2 dimensional array")
    if pos.shape[1] != 3:
        raise TypeError("'pos' should have 3 columns")
    
    value = np.asfarray( value )
    if np.ndim( value ) != 3:
        raise TypeError("'value' must be a 3 dimensional array")
    
    gridsize = value.shape[0]
    if False in map( lambda n: n == gridsize, value.shape[1:] ):
        raise TypeError("all dimensions of 'value' should be same")

    cellsize = np.asfarray( boxsize ) / gridsize

    i = np.floor( pos / cellsize ).astype( 'int' )

    interpval = value[ i[:,0], i[:,1], i[:,2] ]

    return interpval

def cicInterpolate(value: Any, pos: Any, boxsize: float) -> Any:
    r"""
    Interpolate the values at grid points to given positions using cloud in cells scheme. 

    Parameters
    ----------
    value: array_like of shape (G,G,G)
        Values of the quantity to be interpolated, on grid positions. Must be a 3D array.
    pos: array_like of shape (N,3)
        Particle positions.
    boxsize: float
        Size of bounding box of the particles.

    Returns
    -------
    val: array_like of length N
        Values interpolated to particle positions. Will be a 1D array.
    
    """

    pos = np.asfarray( pos )
    if np.ndim( pos ) != 2:
        raise TypeError("'pos' must be a 2 dimensional array")
    if pos.shape[1] != 3:
        raise TypeError("'pos' should have 3 columns")
    
    value = np.asfarray( value )
    if np.ndim( value ) != 3:
        raise TypeError("'value' must be a 3 dimensional array")
    
    gridsize = value.shape[0]
    if False in map( lambda n: n == gridsize, value.shape[1:] ):
        raise TypeError("all dimensions of 'value' should be same")

    cellsize = np.asfarray( boxsize ) / gridsize

    d = pos / cellsize
    i = np.floor( d )
    d = d - i
    t = 1.0 - d
    j = np.where( d > 0.5, i+1, i-1 ) % gridsize

    i, j = i.astype( 'int' ), j.astype( 'int' )

    interpval = (
                    value[ i[:,0], i[:,1], i[:,2] ] * ( t[:,0] * t[:,1] * t[:,2] )
                        + value[ i[:,0], i[:,1], j[:,2] ] * ( t[:,0] * t[:,1] * d[:,2] )
                        + value[ i[:,0], j[:,1], i[:,2] ] * ( t[:,0] * d[:,1] * t[:,2] )
                        + value[ i[:,0], j[:,1], j[:,2] ] * ( t[:,0] * d[:,1] * d[:,2] )
                        + value[ j[:,0], i[:,1], i[:,2] ] * ( d[:,0] * t[:,1] * t[:,2] )
                        + value[ j[:,0], i[:,1], j[:,2] ] * ( d[:,0] * t[:,1] * d[:,2] )
                        + value[ j[:,0], j[:,1], i[:,2] ] * ( d[:,0] * d[:,1] * t[:,2] )
                        + value[ j[:,0], j[:,1], j[:,2] ] * ( d[:,0] * d[:,1] * d[:,2] )
                )
    
    return interpval

###############################################################################################
# Power spectrum from density
###############################################################################################

def powerSpectrum(delta: Any, boxsize: float, bins: int = 21, cic_convolve: bool = False) -> Any:
    r"""
    Estimate the power spectrum from particles density data.

    Parameters
    ----------
    delta: array_like of shape (gridsize, gridsize, gridsize)
        Particle positions.
    boxsize: float  
        Size of the bounding box of the particles.
    bins: int, optional
        Number bins in k-space (default is 21).
    cic_convolve: bool, optional
        If true, apply a CIC deconvolution step. Default is false.

    Returns
    -------
    k: array_like
        Wavenumbers in (unit of boxsize)^-1. Has length equal to `bins`.
    power: array_like
        Power spectrum values correspond to wavenumbers. 

    """
    delta = np.asfarray(delta)

    assert np.ndim(delta) == 3 

    gridsize = delta.shape[0]
    if not np.equal(delta.shape, gridsize).min():
        raise ValueError("delta should be a square grid (all dimensions should have equal size)")

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

    if cic_convolve:
        w = (np.sinc( kx / kf / gridsize ) 
                * np.sinc( ky / kf / gridsize ) 
                * np.sinc( kz / kf / gridsize ))
        delta = delta / w**2

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

