#!/usr/bin/python3

from typing import Any
import numpy as np

class CICError(Exception):
    """
    Error raised by functions and classes related to CIC computations.
    """
    ...


def cart2redshift(x: Any, v: Any, z: float, los: Any = ...) -> Any:
    r"""
    Convert a real (cartetian) space galaxy catalog to a redshift space catalog 
    using the plane parallel transformation.

    .. math::
        {\bf s} = {\bf x} + ({\bf v} \cdot \hat{\bf l}) \frac{(z + 1)\hat{\bf l}}{H}

    where, :math:`{\bf s}` and :math:`{\bf x}` are the comoving position of the 
    galaxies in redshift and real spaces, respectively. :math:`{\bf l}` is the 
    line-of-sight vector.

    Parameters
    ----------
    x: array_like
        Comoving position coordinates of the galaxies. Must be an array with 3 
        columns (corrsponding to 3 cartesian coordinates of the position).
    v: array_like
        Peculiar velocities of the galaxies in units of the Hubble parameter 
        (i.e., now both the distance and velocity will be in same units). It should 
        have the same shape as the position array.
    z: float
        Redshift corresponding to the present configuration.
    los: array_like, str
        Line-of-sight vector. Position corrdinates are transformed along this direction.
        It can be a 3-vector or any of `x`, `y` or `z` (directions). If not specified, 
        use the radial direction.

    Returns
    -------
    s: array_like
        Comoving position of galaxies in redshift space. This will have the same shape 
        as the input position coordinates.

    Examples
    --------
    Let us create a random position catalog catalog in a 500 unit box, with randomized 
    velocity in the range [-10, 10]. Then the transformed position with z-direction as
    the line-of will be 

    >>> x = np.random.uniform(0., 500., (8, 3))
    >>> v = np.random.uniform(-10., 10., (8, 3))
    >>> s = pycic.cart2redshift(x, v, 0., 'z')

    """
    if z < -1: # check redshift
        raise CICError("redshift must not be less than -1")

    # check input data:
    x, v = np.asarray(x), np.asarray(v)
    if x.ndim != 2 or v.ndim != 2:
        raise CICError("position and velocity should be a 2D array")
    elif x.shape[1] != 3 or v.shape[1] != 3:
        raise CICError("position and velocity should have 3 components (3D vectors)")
    elif x.shape[0] != v.shape[0]:
        raise CICError("size position and velocity arrays must be the same")

    # get line-of-sight vector:
    if los is ... :
        los = x
    elif isinstance(los, str):
        if len(los) == 1 and los in 'xyz':
            los = np.where(
                            np.array(['x', 'y', 'z']) == los,
                            1., 0.
                          )
        else:
            raise CICError(f"invalid key for los, `{los}`")
    else:
        los = np.asarray(los)
        if los.ndim != 1:
            raise CICError("los must be 1D array (vector)")
        elif los.shape[0] != 3:
            raise CICError("los must be a 3-vector")
    los = los / np.sum(los**2, axis = -1) 

    # plane parallel transformation:
    s = x + los * np.sum(v * los, axis = -1)[:, np.newaxis] * (1. + z)

    return s








