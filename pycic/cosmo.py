#!/usr/bin/python3
r"""
`cosmo.py`: Cosmology
=====================

Cosmology models and related computations.

"""

from typing import Any
import pycic.transfer as transfer
import numpy as np

class CosmologyError(Exception):
    """ Base class of exceptions used by cosmology objects. """
    ...

class Cosmology:
    r"""
    A flat, Lambda-CDM cosmology class. 
    """
    __slots__ = (
                    'Om0', 'Ob0', 'Ode0', 'Onu0', 'h', 'ns', 'sigma8', 'Tcmb0', 'Nnu', 
                    'Mnu', '_pknorm', 'psmodel', '_transfer', 
                )

    def __init__(self, Om0: float, Ob0: float, h: float, sigma8: float = ..., ns: float = 1.0, Tcmb0: float = 2.725, Nnu: float = ..., Mnu: float = ..., psmodel: str = 'eisenstein98_zb') -> None:
        if Om0 < 0:
            raise CosmologyError("Om0 cannot be negative")
        elif Ob0 < 0 or Ob0 > Om0:
            raise CosmologyError("Ob0 cannot be negative or greater than Om0")
        elif h < 0:
            raise CosmologyError("h cannot be negative")
        elif Tcmb0 <= 0:
            raise CosmologyError("Tcmb0 must be positive")
        elif psmodel not in transfer.available.keys():
            raise CosmologyError(f"invalid power spectrum model {psmodel}")

        # cosmology model parameters:
        self.h             = h
        self.Om0, self.Ob0 = Om0, Ob0
        self.Ode0          = 1 - Om0   # rest of the density is dark-energy density
        
        self.Tcmb0         = Tcmb0

        self.Onu0 = 0
        if Nnu is not ... :
            if Nnu <= 0:
                raise CosmologyError("Nnu must be positive")
            elif Mnu <= 0:
                raise CosmologyError("Mnu must be positive")
            
            # find the neutrino density from the total mass, Mnu:
            Onu0 = Mnu / 91.5 / h**2
            if (Onu0 + self.Ob0) > self.Om0:
                raise CosmologyError("baryon + neutrino content cannot exceed total matter content")
            self.Onu0 = Onu0
        elif psmodel in ['eisenstein98_mdm', ]:
            raise CosmologyError(f"cannot use `{psmodel}` for cosmologies without massive neutrinos")
        self.Mnu, self.Nnu = Mnu, Nnu

        # power spectrum specifications:
        self.psmodel       = psmodel
        self._transfer     = transfer.available[psmodel]
        self.ns            = ns 
        self._pknorm       = 1.0 # power spectrum normalization factor 
        self.sigma8        = ...

        self.normalize(sigma8)   # normalize the power spectrum

    @property
    def H0(self) -> float:
        """ Hubble parameter in units of km/sec/Mpc """
        return self.h * 100.0           

    def __repr__(self) -> str:
        return f"Cosmology(Om0={self.Om0}, Ob0={self.Ob0}, Ode0={self.Ode0}, Onu0={self.Onu0}, h={self.h}, Tcmb0={self.Tcmb0}K, ns={self.ns}, sigma8={'...' if self.sigma8 is ... else self.sigma8})"

    def Ez(self, z: Any) -> Any:
        r"""
        Compute the function :math:`E(z)=\sqrt{\Omega_{\rm m}(z+1)^3+\Omega_{\rm de}}`.
        """
        zp1 = np.asarray(z) + 1
        return np.sqrt(self.Om0 * zp1**3 + self.Ode0)

    def Omz(self, z: Any) -> Any:
        r""" 
        Normalised matter density as function of redshift.
        """
        zp1 = np.asarray(z) + 1
        _Om = self.Om0 * zp1**3
        return _Om / (_Om + self.Ode0)

    def Odez(self, z: Any) -> Any:
        r""" 
        Normalised dark-energy density as function of redshift.
        """
        zp1 = np.asarray(z) + 1
        return self.Ode0 / (self.Om0 * zp1**3 + self.Ode0)

    def Dz(self, z: Any, normalize: bool = True) -> Any:
        r"""
        Linear growth factor.
        """
        def _Dz(z: Any) -> Any:
            """ Un-normalised linear growth factor. """
            zp1  = np.asarray(z) + 1
            Omz  = self.Om0 * zp1**3
            Odez = Omz + self.Ode0   # E^2(z), will be overwritten by Ode(z)
            Omz  = Omz / Odez
            Odez = self.Ode0 / Odez

            # growth factor using the approximation be carroll et al (1992)
            Dz = 2.5 * Omz / zp1 / (Omz**(4./7.) - Odez + (1 + Omz / 2) * (1 + Odez / 70))
            return Dz

        if normalize:
            return _Dz(z) / _Dz(0)
        return _Dz(z)

    def fz(self, z: Any) -> Any:
        r"""
        Linear growth rate.
        """
        return self.Omz(z)**0.6

    def transfer(self, k: Any, z: float = 0) -> Any:
        r"""
        Linear transfer function.
        """
        Dz = ...
        if self.psmodel == 'eisenstein98_mdm':
            Dz = self.Dz(z)
        return self._transfer(
                                k, 
                                h     = self.h, 
                                Om0   = self.Om0,
                                Ob0   = self.Ob0,
                                Ode0  = self.Ode0,
                                Onu0  = self.Onu0,
                                Nnu   = self.Nnu,
                                Tcmb0 = self.Tcmb0,
                                z     = ...,
                                Dz    = Dz,
                             )

    def _unn_matterPowerSpectrum(self, k: Any, z: float = 0) -> Any:
        """
        Un-normalised linear matter power spectrum.
        """
        k = np.asarray(k)
        return self.transfer(k, z)**2 * k**self.ns

    def _unn_variance(self, r: Any, z: float = 0, ka: float = 1e-8, kb: float = 1e+8, n: int = 1001) -> Any:
        """
        Un-normalised linear variance.
        """
        def filt(x):
            """ spherical top-hat filter. """
            return (np.sin(x) - x * np.cos(x)) * 3. / x**3 

        # setup integration: simpson's rule is used
        if n < 3 or not n%2:
            raise ValueError("n must be an odd integer greater than 2")
        elif kb <= ka:
            raise ValueError("kb must be greater than ka")
        elif ka < 0:
            raise ValueError("ka and kb must be positive")
        
        lnka, lnkb = np.log(ka), np.log(kb)
        dlnk       = (lnkb - lnka) / (n-1)
        k          = np.exp(np.linspace(lnka, lnkb, n)) # nodes

        # integration done in log(k) variable
        kr  = np.outer(r, k)
        y   = k**3 * self._unn_matterPowerSpectrum(k, z) * filt(kr)**2
        var = (y[:, :-1:2].sum(-1) + 4 * y[:, 1::2].sum(-1) + y[:, 2::2].sum(-1)) * dlnk / 3

        return (var if np.ndim(r) else var[0]) / 2. / np.pi**2

    def matterPowerSpectrum(self, k: Any, z: float = 0, normalize: bool = True) -> Any:
        r""" 
        Linear matter power spectrum.
        """
        if np.ndim(z):
            raise TypeError("z must be a scalar")
        pk = self._unn_matterPowerSpectrum(k, z) * self.Dz(z)**2
        if normalize:
            return pk * self._pknorm
        return pk

    def variance(self, r: Any, z: float = 0, normalize: bool = True, ka: float = 1e-8, kb: float = 1e+8, n: int = 1001) -> Any:
        r""" 
        Linear variance of density perturbations.
        """
        if np.ndim(z):
            raise TypeError("z must be a scalar")
        var = self._unn_variance(r, z, ka, kb, n) * self.Dz(z)**2
        if normalize:
            return var * self._pknorm
        return var

    def normalize(self, sigma8: float = ..., ka: float = 1e-8, kb: float = 1e+8, n: int = 1001) -> None:
        """ 
        Normalise the power spectrum with a :math:`\sigma_8` value.
        """
        sigma8 = self.sigma8 if sigma8 is ... else sigma8
        if sigma8 is ... :
            # raise CosmologyError("sigma8 is not set")
            self._pknorm = 1.0
            return
        elif sigma8 < 0:
            raise CosmologyError("sigma8 must be postive")
        self._pknorm = sigma8**2 / self._unn_variance(8, 0, ka, kb, n)
        self.sigma8  = sigma8
        

