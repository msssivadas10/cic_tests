#!/usr/bin/python3

import numpy as np
from typing import Any 
from cosmo import Cosmology
from itertools import product, repeat
from collections import namedtuple
from scipy.special import gamma
from scipy.optimize import newton

class DistributionError(Exception):
    """ Base class of exceptions used by distribution objects. """
    ...

class Distribution:
    """ One point distribution of density perturbations. """
    __slots__ = 'cellsize', 'cosmo', 'z', 

    def __init__(self, cellsize: float, cosmo: Cosmology, z: float = 0) -> None:
        if not isinstance(cosmo, Cosmology):
            raise TypeError("cosmo should be a 'Cosmology' object")
        self.cosmo = cosmo

        if cellsize <= 0:
            raise ValueError("cellsize must be positive")
        self.cellsize = cellsize

        self.z = z

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """ Value of distribution function. """
        return NotImplemented

    def matterPowerSpectrum(self, k: Any, z: float, normalize: bool = True) -> Any:
        r""" 
        Linear matter power spectrum.
        """
        return self.cosmo.matterPowerSpectrum(k, z, normalize)


class GEVLogDistribution(Distribution):
    r""" 
    One point GEV distribution for the density perturbations.
    """
    __slots__ = 'kn', '_a', '_b', '_kcont', '_settings',

    Settings  = namedtuple('settings', ['ka', 'kb', 'n', 'n3']) 

    def __init__(self, cellsize: float, cosmo: Cosmology, z: float = 0) -> None:
        super().__init__(cellsize, cosmo, z)

        self.kn = np.pi / self.cellsize # nyquist wavenumber
        
        self._a, self._b = 0.0, 0.0     # power law parameters 
        self._kcont      = ...
        self._settings   = self.Settings(1e-8, 1e+8, 1001, 101)

    def set(self, *, ka: float = ..., kb: float = ..., n: int = ..., n3d: int = ..., z: float = 0) -> None:
        if n is not ... :
            if n < 3 or not n%2:
                raise ValueError("n must be and odd number greater than 2")
            self._settings['n'] = n
        if ka is not ... :
            if ka < 0:
                raise ValueError("ka must be positive")
            elif self.kn <= ka:
                raise ValueError("ka must be less than the nyquist wavenumber")
            self._settings['ka'] = ka
        return

    def linearCellVariance(self) -> float:
        r"""
        Linear variance in a cell.
        """
        ka, n = self._settings.ka, self._settings.n

        lnka, lnkb = np.log(ka), np.log(self.kn)
        dlnk       = (lnkb - lnka) / (n-1)
        k          = np.exp(np.linspace(lnka, lnkb, n)) # nodes

        # integration done in log(k) variable
        y   = k**3 * self.cosmo.matterPowerSpectrum(k, self.z)
        var = (y[:-1:2].sum(-1) + 4 * y[1::2].sum(-1) + y[2::2].sum(-1)) * dlnk / 3

        return var / 2. / np.pi**2 

    def logCellVariance(self, vlin: float, mu: float = 0.73) -> float:
        """ Fitted value of the log-field variance. """
        return mu * np.log(1 + vlin / mu)

    def _measuredPowerInsideSphere(self, kx: Any, ky: Any, kz: Any) -> Any:
        r""" 
        Measured power spectrum from the cell, for k vectors with length 
        less than the nyquist wavenumber, :math:`k_N`. 
        """
        def _power(__kx: Any, __ky: Any, __kz: Any) -> Any:
            """ matter power spectrum. """
            k = np.sqrt(__kx**2 + __ky**2 + __kz**2)
            return self.cosmo.matterPowerSpectrum(k, self.z)

        def _weight(__kx: Any, __ky: Any, __kz: Any) -> Any:
            """ weight function """
            w = np.sinc(__kx / 2. / self.kn)
            w = w * np.sinc(__ky / 2. / self.kn)
            w = w * np.sinc(__kz / 2. / self.kn)
            return w**2

        def _factor(__kx: Any, __ky: Any, __kz: Any) -> Any:
            return _power(__kx, __ky, __kz) * _weight(__kx, __ky, __kz)

        kx, ky, kz = np.asarray(kx), np.asarray(ky), np.asarray(kz)

        retval  = 0
        for nx, ny, nz in product(*repeat(range(3), 3)):
            if nx**2 + ny**2 + nz**2 >= 9:
                continue
            retval += _factor(kx + 2*self.kn*nx, ky + 2*self.kn*ny, kz + 2*self.kn*nz)
        return retval

    def _measuredPowerOutsideSphere(self, k: Any) -> Any:
        r"""
        Measured power spectrum from the cell, for k vectors with length 
        greater than the nyquist wavenumber, :math:`k_N`. 
        """
        return self._a * np.asarray(k)**self._b

    def makeContinuation(self, npts: int = 100_000, krange: slice = slice(0.5, 0.7)) -> None:
        """
        Get the power law continuation of the power spectrum.
        """
        if not isinstance(krange, slice):
            raise TypeError("krange must be a slice")
        ka, kb, kn = krange.start, krange.stop, self.kn
        if not (0.0 <= ka <= kb <= 1.0):
            raise ValueError("kb must be greater than ka and both in range [0, 1]")
        self._kcont = ka

        # generate random k vectors with length in the k-range
        lnk   = np.random.uniform(np.log(ka), np.log(kb), npts) + np.log(kn)
        theta = np.random.uniform(0,   np.pi, npts)
        phi   = np.random.uniform(0, 2*np.pi, npts)

        # polar to cartesian conversion: 
        kx, ky, kz = np.exp(lnk) * np.array([
                                                np.sin(theta) * np.cos(phi), 
                                                np.sin(theta) * np.sin(phi), 
                                                np.cos(theta)
                                            ])
        lnpm       = np.log(self._measuredPowerInsideSphere(kx, ky, kz))

        # find a linear regression of the form lnpm ~ lna + b* lnk
        b, lna = np.polyfit(lnk, lnpm, deg = 1)

        self._a, self._b = np.exp(lna), b

    def measuredPowerSpectrum(self, kx: Any, ky: Any, kz: Any, kcont: float = 0.6) -> Any:
        r"""
        Measured power spectrum from the cell.
        """
        if not self._kcont <= kcont <= 1.0:
            raise ValueError(f'kcont must be in the range [{self._kcont}, 1]')
        kx, ky, kz = np.asarray(kx), np.asarray(ky), np.asarray(kz)
        out        = np.sqrt(kx**2 + ky**2 + kz**2) # k, overtwritten by pk
        mask       = (out <= self.kn * kcont)
        out[~mask] = self._measuredPowerOutsideSphere(out[~mask])
        out[mask]  = self._measuredPowerInsideSphere(kx[mask], ky[mask], kz[mask])
        return out
        
    def measuredVariance(self, n: int = 101, ka: float = 1e-8) -> float:
        """
        Measure count-in-cell variance in the cell.
        """
        if n < 3 or not n%2:
            raise ValueError("n must be and odd number greater than 2")
        elif ka < 0:
            raise ValueError("ka must be positive")
        elif self.kn <= ka:
            raise ValueError("ka must be less than the nyquist wavenumber")
        
        # integration is done in the positive octant
        # generate k-grid:
        lnka, lnkb = np.log(1e-8), np.log(self.kn) 
        dlnk       = (lnkb - lnka) / (n-1)              
        f          = np.exp(np.linspace(lnka, lnkb, n)) 
        f, ky, kz  = np.meshgrid(f, f, f)               

        f = f * ky * kz * self.measuredPowerSpectrum(f, ky, kz)
        
        del ky, kz # delete variables ky and kz

        f = f[:,:,:-1:2].sum(-1) + 4 * f[:,:,1::2].sum(-1) + f[:,:,2::2].sum(-1)
        f = f[:,:-1:2].sum(-1) + 4 * f[:,1::2].sum(-1) + f[:,2::2].sum(-1)
        f = f[:-1:2].sum(-1) + 4 * f[1::2].sum(-1) + f[2::2].sum(-1)
        
        return 8 * f * (dlnk / 3 / 2 / np.pi)**3

    def _getParameters(self, sigma8: float) -> tuple:
        """ Compute distribution parameters. """
        ka, kb = self._settings.ka, self._settings.kb

        #re-normalise power spectrum
        self.cosmo.normalize(sigma8, ka, kb, self._settings.n) 

        vlin = self.linearCellVariance() # linear variance
        vlog = self.logCellVariance(vlin)        # log field variance

        # calculate the measured cic varaince:
        blog = vlog / vlin                       # log field bias^2
        vcic = self.measuredVariance(self._settings.n3, ka) * blog

        # log field mean value (fit):
        lamda = 0.65
        mlog  = -lamda * np.log(1 + vlin / lamda / 2)

        # skewness of the log field:
        a, b, c, d = -0.7, 1.25, -0.26, 0.06 # fit parameters
        np3        = self.cosmo.ns + 3
        slog       = (a * np3 + b) * vcic**-(d + c * np.log(np3))
            
        # finding shape parameter:
        r = slog * np.sqrt(vcic) # pearson moment coeff.

        def shapefn(xi: float) -> float:
            g1mx  = gamma(1 - xi)
            g1m2x = gamma(1 - 2*xi)
            num   = gamma(1 - 3*xi) - 3*g1mx * g1m2x + 2*g1mx**3
            return r + num / (g1m2x - g1mx**2)**1.5

        xi = newton(shapefn, r) # shape param.

        # scale parameters;
        g1mx  = gamma(1 - xi)
        sigma = np.sqrt(xi**2 * vcic / (gamma(1-2*xi) - g1mx**2))

        # location parameter:
        mu = mlog - sigma * (g1mx - 1) / xi

        return mu, sigma, xi

    def func(self, x: Any, sigma8: float) -> Any:
        """ Value of the distribution function. """
        x = np.asarray(x)
        y = np.empty_like(x)

        # get parameters:
        mu, sigma, xi = self._getParameters(sigma8)

        mask = (x <= mu - sigma / xi) # x has upper limit

        y[~mask] = 0.0

        t       = (1 + (x[mask] + mu) * xi / sigma)**(-1/xi)
        y[mask] = t**(1 + xi) * np.exp(-t) / sigma
        return y
        
class GEVDistribution(GEVLogDistribution):
    r""" 
    One point GEV distribution for the density perturbations.
    """

    def __init__(self, cellsize: float, cosmo: Cosmology, z: float = 0) -> None:
        super().__init__(cellsize, cosmo, z)

    def func(self, x: Any, sigma8: float) -> Any:
        return NotImplemented
        return super().func(x, sigma8)

        


