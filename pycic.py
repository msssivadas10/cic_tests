#!/usr/bin/python3

import numpy as np
from typing import Any, Tuple
from itertools import product, repeat
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from scipy.special import gamma
from scipy.optimize import newton

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
    n: float
        Slope of the linear power spectrum.
    pixsize: float
        Size of a pixel (cell). Th entire space is divided into cells of this size.

    """
    __slots__ = "pk_spline", "z", "Om0", "Ode0", "Ok0", "h", "n", "pixsize", "kn", 

    def __init__(self, pk_table: Any, z: float, Om0: float, Ode0: float, h: float, n: float, pixsize: float) -> None:
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
        self.n = n

        if not isinstance(pixsize, (float, int)):
            raise CICError("pixsize should be a number")
        self.pixsize = pixsize
        self.kn      = np.pi / self.pixsize # nyquist wavenumber 

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

    def powerLin(self, lnk: Any) -> Any:
        r"""
        Get the power spectrum, calculated by interpolating the values in the power
        spectrum table.

        Parameters
        ----------
        lnk: array_like
            Natural logarithm of the wavenumber.

        Returns
        -------
        pk: array_like
            Value of the power spectrum.

        """
        return np.exp(self.pk_spline(lnk))

    def _Pdelta(self, delta: Any, mu: float, sigma: float, xi: float) -> Any:
        r"""
        PDF of :math:`\delta` as given in Repp and Szapudi (2020).

        .. math::
            P(\delta) = \frac{1}{(1 + \delta) \sigma}t^{1 + \xi} e^{-t}

        where, :math:`t \equiv t(\delta)` is given by

        .. math::
            t(\delta) = \left( 1 + \frac{\ln \delta - \mu}{\sigma} \xi \right)^{-1/\xi}

        and :math:`\mu, \sigma, \xi` are the location, scale and shape parameters.

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

    def _var_lin_integrand(self, lnk: Any) -> Any:
        r"""
        Function to be integrated in order to find the linear variance in the cell,
        :math:`\sigma^2_{\rm lin}`, given by :math:`k^3 P_{\rm lin}(k)`, where the 
        integration variable is :math:`\ln k`. The :math:`1/2\pi^2` factor is not 
        included here. 

        Parameters
        ----------
        lnk: array_like
            Integration variable, natural logarithm of wavenumber.

        Returns
        -------
        fk: array_like
            Value of the function.

        """
        return np.exp(lnk)**3 * self.powerLin(lnk)

    def varLin(self, ) -> float:
        r"""
        Compute the value of the linear variance in the cell. It is computed from
        the linear power spectrum as the integral

        .. math::
            \sigma_{\rm lin}^2 = \int_0^{k_N} \frac{{\rm d}k k^2}{2 \pi^2} P_{\rm lin}(k) 

        where :math:`k_N` is the Nyquist wavenumber.

        Returns
        -------
        sigma2_lin: float
            Value of linear variance.
        """
        retval, err = quad(self._var_lin_integrand, -8., np.log(self.kn))
        return retval / 2. / np.pi**2

    def biasA(self, vlin: float) -> float:
        r"""
        Get the A-bias factor. It is given by 

        .. math::
            b_A^2 = \frac{\sigma^2_a (k_N)}{\sigma^2_{\rm lin} (k_N)}

        Parameters
        ----------
        vlin: float
            Linear variance.
        
        Returns
        -------
        b: float
            Bias value.

        """
        return np.sqrt(self.varA(vlin) / vlin)

    def powerA(self, lnk: Any) -> Any:
        r"""
        Power spectrum of the log density. It is related to the linear power spectrum 
        by a bias factor as :math:`P_A(k) = b_A^2 P_{\rm lin}(k)`. The bias factor is 
        the ratio of log to linear variances.

        Parameters
        ----------
        k: array_like
            Natural logarith of wavenumber.

        Returns
        -------
        Pk: array_like
            Value of power spectrum.


        """
        vlin = self.varLin()           # linear variance
        b2   = self.varA(vlin) / vlin # bias squared
        return self.powerLin(lnk) * b2

    def varA(self, vlin: float) -> float:
        r"""
        Get the A-variance in the cell, where :math:`A = \ln (1 + \delta)`. This uses 
        the fit given in Repp & Szapudi (2018).

        .. math::
            \sigma_A^2 = \mu \ln \left(1 + \frac{\sigma_{\rm lin}^2}{\mu} \right)
        
        Both the variances are evaluated at the Nyquist wavenumber :math:`k_N` and the
        value of :math:`\mu` is taken as 0.73 (best-fit value).

        Parameters
        ----------
        vlin: float
            Linear variance.

        Returns
        -------
        sigma: float
            Value of variance.

        """
        mu = 0.73
        return mu * np.log(1. + vlin / mu)

    def meanA(self, vlin: float) -> float:
        r"""
        Return the fitted value of mean of A as a function the linear variance. This 
        uses the fit given in Repp & Szapudi (2018)

        .. math::
            <A> = - \lambda \ln \left(1 + \frac{\sigma_{\rm lin}^2}{2\lambda} \right)

        where the best fitting value is :math:`\lambda = 0.65`.

        Parameters
        ----------
        vlin: float
            Linear variance.

        Returns
        -------
        avA: float
            Mean value of A.

        """
        lamda = 0.65
        return -lamda * np.log(1. + vlin / 2. / lamda)

    def skewA(self, vl: float) -> float:
        r"""
        Return the skewness of the A distribution as a function the count-in-cells 
        variance. The value if calculated by a fit given by Repp & Szapudi (2018),

        .. math::
            T_3 = T(n) [\sigma_A^2(l)]^{-p(n)}

        where :math:`n` is the slope of the power spectrum and, 

        .. math::
            T(n) = a(n + 3) + b \quad {\rm and} \quad p(n) = d + c \ln(n + 3)

        The best fit values are a = -0.70, b = 1.25, c = -0.26 and d = 0.06.

        Parameters
        ----------
        vl: float
            Count in cell variance. This is not the fitted value, but calculated from
            the *measured* A power spectrum.

        Returns
        -------
        T3: float
            Value of skewness.

        """
        a  = -0.70
        b  =  1.25
        c  = -0.26
        d  =  0.06

        np3 = self.n + 3.
        Tn  = a * np3 + b
        pn  = d + c * np.log(np3)
        return Tn * vl**(-pn)

    def getParameters(self, vlin: float, vl: float) -> Tuple[float, float, float]:
        r"""
        Get the location, scale and shape parameters of the probability distribution 
        corrsponding to the linear and count-in-cell variance value.

        Parameters
        ----------
        vlin: float
            Linear Variance.
        vl: float
            Count-in-cells variance.

        Returns
        -------
        mu: float
            Value of location parameter, :math:`\mu`.
        sigma: float
            Value of scale parameter, :math:`\sigma`.
        xi: float
            Value of shape parameter, :math:`\xi`. 

        """    
        # finding for xi:
        r1 = vl * self.skewA(vl) # Pearson’s moment coefficient (\gamma_1)

        def fxi(xi: float) -> float:
            """ function whose roort is xi """
            fn = gamma(1 - 3*xi) - 3*gamma(1 - xi)*gamma(1 - 2*xi) + 2*gamma(1 - xi)**3
            fd = (gamma(1 - 2*xi) - gamma(1 - xi)**2)**1.5
            return r1 + fn / fd

        # TODO: using r1 as an initial guess root. need to find a better one.
        xi = newton(fxi, r1, )

        # finding sigma:
        sigma = xi * vl / np.sqrt(gamma(1 - 2*xi) - gamma(1 - xi)**2)

        # finding mu:
        mu = self.meanA(vlin) - sigma * (gamma(1 - xi) - 1) / xi

        return mu, sigma, xi

    def _powerA_meas(self, kx: Any, ky: Any, kz: Any) -> Any:
        r"""
        Measured A power spectrum. This is valid only for :math:`\bf k` vectors 
        such that :math:`\sqrt{k_x^2 + k_y^2 + k_z^2} \le k_N`. No error will be
        raised otherwise.

        Parameters
        ----------
        kx: array_like
            X component of the k vectors.
        ky: array_like
            Y component of the k vectors.
        kz: array_like
            Z component of the k vectors.

        Returns
        -------
        Pk: array_like
            Power spectrum values.

        """
        # XXX: using the cloud in cell weight function as W(k)
        p    = 2.
        nmax = 3 
        kn   = self.kn


        def _powerA_vec(kx: Any, ky: Any, kz: Any) -> Any:
            """ A power for vector inputs """
            k = np.sqrt(kx**2 + ky**2 + kz**2)
            return self.powerA(np.log(k))

        def _weight(kx: Any, ky: Any, kz: Any) -> Any:
            """ mass assignment function (squared) """
            wx = np.sinc(kx / kn / 2.)
            wy = np.sinc(ky / kn / 2.)
            wz = np.sinc(kz / kn / 2.)
            return (wx * wy * wz)**(2.*p)

        def _power_term(kx: Any, ky: Any, kz: Any) -> Any:
            """ a term in the power sum """
            return _powerA_vec(kx, ky, kz) * _weight(kx, ky, kz)

        Pk = 0.
        for nx, ny, nz in product(*repeat(range(nmax), 3)):
            if nx**2 + ny**2 + nz**2 >= 9.:
                continue # only abs(n) < 3 are needed
            Pk += _power_term(
                                kx + 2. * nx * kn, 
                                ky + 2. * ny * kn, 
                                kz + 2. * nz * kn
                             )
        
        return Pk

    def powerA_meas(self, kx: Any, ky: Any, kz: Any) -> Any:
        r"""
        Measured power spectrum of A. It is computed as the sum

        .. math::
            P_{A, \rm meas} = \sum_{\bf n} P_A ({\bf k} + 2 k_N * {\bf n})
                                W^2 ({\bf k} + 2 k_N * {\bf n})

        where :math:`{\bf n}` is a vector of three integers with length less
        than 3. This is valid for :math:`\vert {\bf k} \vert \le k_N`. After 
        that limit, it is power law continued. 

        Parameters
        ----------
        kx: array_like
            X component of the k vectors.
        ky: array_like
            Y component of the k vectors.
        kz: array_like
            Z component of the k vectors.

        Returns
        -------
        Pk: array_like
            Power spectrum values.

        """
        return NotImplemented



