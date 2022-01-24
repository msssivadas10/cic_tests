#!/usr/bin/python3

from typing import Any
import numpy as np

class CICError(Exception):
    """
    Error raised by functions and classes related to CIC computations.
    """
    ...


def cart2redshift(x: Any, v: Any, z: float, los: Any = ..., Om0: float = 0.3, Ode0: float = 0.7, H0: float = 70.0):
    r"""
    Convert a real (cartetian) space galaxy catalog to a redshift space catalog 
    using the plane parallel transformation.

    .. math::
        {\bf s} = {\bf x} + ({\bf v} \cdot \hat{\bf l}) \frac{(z + 1)\hat{\bf l}}{H}

    where, :math:`{\bf s}` and :math:`{\bf x}` are the comoving position of the 
    galaxies in redshift and real spaces, respectively. :math:`{\bf l}` is the line
    of sight vector.

    Parameters
    ----------

    Returns
    -------

    """
    # get the hubble parameter:
    zp1 = 1 + z
    Hz  = H0 * np.sqrt(Om0 * zp1**3 + (1. - Om0 - Ode0) * zp1**2 + Ode0)

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
        if los == 'x':
            los = np.array([1., 0., 0.])
        elif los == 'y':
            los = np.array([0., 1., 0.])
        elif los == 'z':
            los = np.array([0., 0., 1.])
        else:
            raise CICError(f"invalid key for los, `{los}`")
    else:
        los = np.asarray(los)
        if los.ndim != 1:
            raise CICError("los must be 1D array (vector)")
        elif los.shape[0] != 3:
            raise CICError("los must be a 3-vector")
