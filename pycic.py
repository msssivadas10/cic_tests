#!/usr/bin/python3

import numpy as np
from typing import Any
from scipy.interpolate import CubicSpline

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

class cicDistribution:
    r"""
    Implementation of the theoretical count-in-cells distribution given in Repp
    and Szapudi (2020). 

    Parameters
    ----------
    pk_table: array_like
        Power spectrum table. This must be a 2D array with two columns - :math:`\ln k`
        in the first and :math:`\ln P(k)` in the second. The table should have enough 
        resolution to include the details, if present.
    z: float
        Redshift value. Must be greater than -1. 
    Om0: float
        Present value of the normalized matter density, :math:`\Omega_{\rm m}`.
    Ode0: float
        Present value of the normalized dark-energy density, :math:`\Omega_{\rm de}`.
    h: float
        Presnt value of the Hubble parameter in units of 100 km/sec/Mpc.

    """
    __slots__ = "pk_spline", "z", "Om0", "Ode0", "Ok0", "h", 

    def __init__(self, pk_table: Any, z: float, Om0: float, Ode0: float, h: float) -> None:
        if z < -1.:
            raise CICError("redshift cannot be less than -1")
        self.z = z

        if Om0 < 0.:
            raise CICError("Om0 cannot be negative")
        self.Om0 = Om0

        if Ode0 < 0.: 
            raise CICError("Ode0 cannot be negative")
        self.Ode0 = Ode0
        self.Ok0  = 1. - Om0 - Ode0
        if abs(self.Ok0) < 1.e-08:
            self.Ok0 = 0.

        if h < 0.:
            raise CICError("h cannot be neative")
        self.h = h

        pk_table = np.asarray(pk_table)
        if pk_table.ndim != 2:
            raise CICError("power spectrum should be given as a table")
        elif pk_table.shape[1] != 2:
            raise CICError("power table should have two columns")
        lnk, lnpk      = pk_table.T
        self.pk_spline = CubicSpline(lnk, lnpk)

    def Ez(self, z: Any) -> Any:
        r"""
        Evaluate the function :math:`E(z) := H(z) / H_0` as a function of the 
        redshift :math:`z`.

        .. math::
            E(z) = \sqrt{\Omega_{\rm m} (1 + z)^3 + \Omega_{\rm k} (1 + z)^2 + 
                         \Omega_{\rm de}}

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        Ez: array_like
            Value of the function at z.

        Examples
        --------
        TODO

        """
        zp1 = 1. + np.asarray(z)
        return np.sqrt(self.Om0 * zp1**3 + self.Ok0 * zp1**2 + self.Ode0)

    def Omz(self, z: Any) -> Any:
        r"""
        Evaluate the normalized density of (dark) matter at redshift :math:`z`. 
        
        .. math::
            \Omega_{\rm m}(z) = \frac{\Omega_{\rm m} (z + 1)^3}{E^2(z)}
        
        Parameters
        ----------
        z: array_like
            Redshift

        Returns
        -------
        Omz: array_like
            Normalized matter density at z.

        Examples
        --------

        """
        zp1 = 1 + np.asarray(z)
        Omz = self.Om0 * zp1**3
        return Omz / (Omz + self.Ok0 * zp1**2 + self.Ode0)

    def _Pdelta(self, delta: Any, mu: float, sigma: float, xi: float) -> Any:
        r"""
        PDF of :math:`\delta` as given in Repp and Szapudi (2020).

        .. math::
            P(\delta) = \frac{1}{(1 + \delta) \sigma}t^{1 + \xi} e^{-t}

        where, :math:`t \equiv t(\delta)` is given by

        .. math::
            t(\delta) = \left( 1 + \frac{\ln \delta - \mu}{\sigma} \xi \right)^{-1/\xi}

        :math:`\mu, \sigma, \xi` are the location, scale and shape parameters.

        Parameters
        ----------
        delta: array_like
            :math:`\delta` value - density fluctuation.
        mu: float
            Lccation parameter.
        sigma: float
            Scale parameter.
        xi: float
            Shape parameter.

        Returns
        -------
        Pdelta: array_like
            Probability density for the delta value.

        Examples
        --------
        TODO

        """
        t = (1. + (np.log(delta) - mu) * xi / sigma)**(-1./xi)
        return t**(1 + xi) * np.exp(-t) / (1. + delta) / sigma





