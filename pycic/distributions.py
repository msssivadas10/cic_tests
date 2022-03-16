#!/usr/bin/python3
r"""
Distributions
=============

One point probability distribution functions for density fluctuation 
fields.

"""

import numpy as np
from typing import Any 
from .cosmo import Cosmology
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

    def __init__(self, cellsize: float, cosmo: Cosmology, z: float) -> None:
        if not isinstance(cosmo, Cosmology):
            raise TypeError("cosmo should be a 'Cosmology' object")
        self.cosmo = cosmo

        if cellsize <= 0:
            raise ValueError("cellsize must be positive")
        self.cellsize = cellsize

        if z < -1:
            raise ValueError("z must be greater than -1")
        self.z = z

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        """ Value of distribution function. """
        return NotImplemented

    def set(self, **kwargs) -> None:
        """ Set values for the settings attributes. """
        ...

    def matterPowerSpectrum(self, k: Any, z: float, normalize: bool = True) -> Any:
        r""" 
        Linear matter power spectrum.
        """
        return self.cosmo.matterPowerSpectrum(k, z, normalize)

    def parametrize(self, *args, **kwargs) -> tuple:
        """ Compute distribution parameters. """
        ...

    def parameters(self) -> dict:
        """ Get the free parameters. """
        ...

class GEVDistribution(Distribution):
    r""" 
    One point GEV distribution for the density perturbations.
    """
    __slots__  = 'kn', '_a', '_b', '_kcont', '_settings', '_params'

    Settings   = namedtuple(
                                'settings', 
                                ['ka', 'kb', 'n', 'n3d', 'n4cont', 'krange', 'ksplit'],
                            )

    Parameters = namedtuple(
                                'Parameters', 
                                ['mu', 'sigma', 'xi', 'sigma8', 'bias']
                           )  

    def __init__(self, cellsize: float, cosmo: Cosmology, z: float) -> None:
        super().__init__(cellsize, cosmo, z)

        self.kn = np.pi / self.cellsize # nyquist wavenumber
        
        self._a, self._b = 0.0, 0.0     # power law parameters 
        self._params     = None
        self._settings   = self.Settings(
                                            ka     = 1e-8, 
                                            kb     = 1e+8, 
                                            n      = 1001, 
                                            n3d    = 101,
                                            n4cont = 100_000,
                                            krange = slice(0.5, 0.7),
                                            ksplit = 0.6,
                                        )

        self.makeContinuation()

    def set(self, **kwargs) -> None:
        """ Set values for the settings attributes. """

        for key, value in kwargs.items():
            if key == 'n':
                # number of nodes for 1d k-integration (simpson rule)
                if value < 3 or not value%2:
                    raise ValueError("n must be and odd number greater than 2")
                self._settings._replace(n = value)
            elif key == 'n3d':
                # number of nodes for 2d k-integration (simpson rule)
                if value < 3 or not value%2:
                    raise ValueError("n3d must be and odd number greater than 2")
                self._settings._replace(n3d = value)
            elif key == 'n4cont':
                # number of values used for continuation
                self._settings._replace(n4cont = value)
            elif key == 'krange':
                # range of values to use for continuation
                if not isinstance(value, slice):
                    raise TypeError("krange must be a slice")
                if not ( 0 <= value.start <= 1 and 0 <= value.stop <= 1):
                    raise ValueError("krange values must be in range [0,1]")
                self._settings._replace(krange = value)
            elif key == 'ksplit':
                # point after which continuation is used
                if not 0 <= value <= 1:
                    raise ValueError("ksplit must be in range [0,1]")
                self._settings._replace(ksplit = value)
            elif key == 'ka':
                # lower limit for k integration
                if value < 0:
                    raise ValueError("ka must be positive")
                elif 1e-4 <= value:
                    raise ValueError("ka must be less than the 1e-4")
                self._settings._replace(ka = value)
            elif key == 'kb':
                # upper limit for k integration
                if value < 0:
                    raise ValueError("kb must be positive")
                elif 1e+4 > value:
                    raise ValueError("kb must be greater than 1e+4")
                self._settings._replace(kb = value)
            elif key == 'z':
                # redshift
                if value < -1:
                    raise ValueError("z must be greater than -1")
                self.z = value
            else:
                raise TypeError(f"invalid keyword argument '{key}'")
            self.makeContinuation()

    def matterPowerSpectrum(self, k: Any, z: float, normalize: bool = True) -> Any:
        """
        Linear matter power spectrum
        """
        return super().matterPowerSpectrum(k, z, normalize)

    def linearCellVariance(self) -> float:
        r"""
        Linear variance in a cell.
        """
        ka, n = self._settings.ka, self._settings.n

        lnka, lnkb = np.log(ka), np.log(self.kn)
        dlnk       = (lnkb - lnka) / (n-1)
        k          = np.exp(np.linspace(lnka, lnkb, n)) # nodes

        # integration done in log(k) variable
        y   = k**3 * self.matterPowerSpectrum(k, self.z)
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
            return self.matterPowerSpectrum(k, self.z, False)

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

    def makeContinuation(self) -> None:
        """
        Get the power law continuation of the power spectrum.
        """
        npts, krange = self._settings.n4cont, self._settings.krange
        ka, kb, kn   = krange.start, krange.stop, self.kn
        if not (0.0 <= ka <= kb <= 1.0):
            raise ValueError("kb must be greater than ka and both in range [0, 1]")

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

    def measuredPowerSpectrum(self, kx: Any, ky: Any, kz: Any) -> Any:
        r"""
        Measured power spectrum from the cell.
        """
        frac = self._settings.ksplit # mark the point to use continuation 

        kx, ky, kz = np.asarray(kx), np.asarray(ky), np.asarray(kz)
        out        = np.sqrt(kx**2 + ky**2 + kz**2) # k, overtwritten by pk
        mask       = (out <= self.kn * frac)
        out[~mask] = self._measuredPowerOutsideSphere(out[~mask])
        out[mask]  = self._measuredPowerInsideSphere(kx[mask], ky[mask], kz[mask])
        return out * self.cosmo._pknorm
        
    def measuredVariance(self) -> float:
        """
        Measure count-in-cell variance in the cell.
        """
        ka, n = self._settings.ka, self._settings.n3d
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

    def parametrize(self, sigma8: float, bias: float = ..., **kwargs) -> None:
        """ Compute distribution parameters. """
        ka, kb = self._settings.ka, self._settings.kb

        #re-normalise power spectrum
        self.cosmo.normalize(sigma8, ka, kb, self._settings.n) 

        vlin = self.linearCellVariance() # linear variance
        vlog = self.logCellVariance(vlin)        # log field variance

        # calculate the measured cic varaince:
        blog = vlog / vlin                       # log field bias^2
        vcic = self.measuredVariance() * blog

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

        # return mu, sigma, xi
        self._params = self.Parameters(mu, sigma, xi, sigma8, bias)

    def parameters(self) -> dict:
        return {
                    'sigma8': self._params.sigma8,
                    'bias'  : self._params.bias, 
               }

    @property
    def xmax(self) -> float:
        r""" Maximum value for :math:`\ln(\delta+1)` """
        # get parameters:
        mu, sigma, xi, _, _ = self._params
        return mu - sigma / xi

    def pdf_log(self, x: Any) -> Any:
        r""" Value of the distribution function, :math:`P(A)`. """
        x = np.asarray(x) # log field, ln(a+delta)
        y = np.zeros_like(x)

        # get parameters:
        mu, sigma, xi, _, _ = self._params

        mask    = (x <= self.xmax) # x has upper limit
        t       = (1 + (x[mask] + mu) * xi / sigma)**(-1/xi)
        y[mask] = t**(1 + xi) * np.exp(-t) / sigma
        return y

    def pdf(self, x: Any) -> Any:
        r""" Value of the distribution function, :math:`P(\delta)` """
        x = np.asarray(x)
        return self.pdf_log(np.log(x+1)) / (x+1)

    def __call__(self, x: Any, log: bool = False) -> Any:
        return self.pdf_log(x) if log else self.pdf(x)
        

class CICDistribution:
    """
    Count-in-cells distribution.
    """
    __slots__    = 'pdf1pt', 'bias', 'sigma8', 
    _all_models  = {
                        "gev" : GEVDistribution, 
                   }

    def __init__(self, model: str, cellsize: float, cosmo: Cosmology, z: float) -> None:
        if model not in self._all_models:
            raise TypeError(f"invalid value for model_1pt, '{model}'")
        
        self.pdf1pt = self._all_models[model](cellsize, cosmo, z)
        
        self.bias: float   = ...
        self.sigma8: float = ...

    @property
    def cosmo(self) -> Cosmology:
        """ Get the curent cosmology model. """
        return self.pdf1pt.cosmo
    
    @property
    def z(self) -> float:
        """ Get the redshift. """
        return self.pdf1pt.z
    
    @property
    def cellsize(self) -> float:
        """ Get the cellsize. """
        return self.pdf1pt.cellsize

    def set(self, **kwargs) -> None:
        """ Set values for the settings attributes. """
        return self.pdf1pt.set(**kwargs)

    def parametrize(self, **kwargs) -> None:
        """ Parameterize the model. """
        if 'bias' not in kwargs.keys():
            raise TypeError("'bias' is a required argument")
        self.bias = kwargs['bias']
        if 'sigma8' in kwargs.keys():
            self.sigma8 = kwargs['sigma8']
        return self.pdf1pt.parametrize(**kwargs)

    def pdf(self, n: Any) -> Any:
        """ Distribution function. """
        return NotImplemented
        

    

    

