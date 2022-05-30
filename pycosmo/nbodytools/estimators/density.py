r"""

Density Estimators
==================

The `nbody.estimators.density` module contains techniques for density estimation from particle data, such as the cloud-
in-cells interpolation scheme and grid to particles interpolation schemes.

"""

from typing import Any 
import numpy as np

def densityCloudInCell(pos: Any, boxsize: float, gridsize: int, mass: Any = 1.0) -> Any:
    r"""
    Cloud in cells density estimation. Assume the particles as boxes of one grid cell size and add the contributions of 
    each particle to the cell it lies and nearby cells.

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
        Density mesh. This will be a 3D array with the value at index (i,j,k) correspond to the density at that cell.

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
                                    weights = mass * t[:,0] * j[:,1] * t[:,2],
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
                                    weights = mass * d[:,0] * j[:,1] * t[:,2],
                                )[0]
                + np.histogramdd(
                                    np.hstack([ j[:,0:1], j[:,1:2], j[:,2:3] ]),
                                    bins    = gridsize,
                                    range   = _range,
                                    weights = mass * d[:,0] * d[:,1] * d[:,2],
                                )[0]
            )

    return dens

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

