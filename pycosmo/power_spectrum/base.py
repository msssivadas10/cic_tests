from typing import Any 
from abc import ABC, abstractmethod
import numpy as np
import pycosmo.utils.settings as settings
import pycosmo.utils.numeric as numeric
import pycosmo.cosmology.cosmo as cosmo
import pycosmo.power_spectrum.filters as filters

class PowerSpectrumError(Exception):
    r"""
    Base class of exception used by  power spectrum objects.
    """
    ...

class PowerSpectrum(ABC):
    r"""
    Base power spectrum class. 
    """

    __slots__ = 'filter', 'cosmology', 'A', 'use_exact_growth'

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        
        if not isinstance( cm, cosmo.Cosmology ):
            raise PowerSpectrumError("cm must be a 'Cosmology' object")
        self.cosmology        = cm
        self.use_exact_growth = False # whether to use exact (integrated) growth factors

        if filter not in filters.filters:
            raise PowerSpectrumError(f"invalid filter: { filter }")
        self.filter = filters.filters[ filter ] # filter to use for smoothing

        self.normalize()

    def Dplus(self, z: Any) -> Any:
        r"""
        Linear growth factor.

        Parameter
        ---------
        z: array_like
            Redshift. Must be a value greater than -1.

        Returns
        --------
        Dz: array_like
            Value of the growth factor.
        """
        return self.cosmology.Dplus( z, exact = self.use_exact_growth )
    
    @property 
    def ns(self) -> float:
        r"""
        Power spectrum index, :math:`n_s`.
        """
        return self.cosmology.ns

    @property 
    def sigma8(self) -> float:
        r"""
        RMS variance of matter fluctuations at 8 Mpc/h scale, :math:`\sigma_8`
        """
        return self.cosmology.sigma8
    
    @abstractmethod
    def transferFunction(self, k: Any, z: float = 0) -> Any:
        """
        Linear transfer function.

        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift (default is 0)

        Returns
        -------
        tk: array_like
            Value of linear transfer function.
        """
        ...

    def linearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        r"""
        Linear matter power spectrum.

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        dim: bool, optional
            If true (default), return the usual power spectrum, else give the dimenssionless one.
        
        Returns
        -------
        pk: array_like
            Linear (dimenssionless) power spectrum values.

        """
        k = np.asfarray( k )

        if np.ndim( z ):
            raise PowerSpectrumError("z must be a scalar")
        if z + 1 < 0:
            raise ValueError("redshift cannot be less than -1")

        Pk = self.A * k**self.ns * self.transferFunction( k, z )**2 * self.Dplus( z )**2
        if not dim:
            return k**3 * Pk / ( 2*np.pi**2 )
        return Pk  

    # non-linear power and scale will be implemented later in a sub-class

    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        r"""
        Non-liinear matter power spectrum. 

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        dim: bool, optional
            If true (default), return the usual power spectrum, else give the dimenssionless one.
        
        Returns
        -------
        pk: array_like
            Non-linear (dimenssionless) power spectrum values.

        See Also
        --------
        `PowerSpectrum.linearPowerSpectrum`;
            Linear matter power spectrum.
            
        """
        raise NotImplementedError("non-linear models are implemented in a general power spectrum subclass")

    def nonlineark(self, k: Any, z: float = 0) -> Any:
        r"""
        Compute the non-liinear wavenumber corresponding to the linear one. 

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        
        Returns
        -------
        knl: array_like
            Non-linear wavenumber in h/Mpc.
            
        """
        raise NotImplementedError("non-linear models are implemented in a general power spectrum subclass")

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, linear: bool = True) -> Any:
        r"""
        Compute the linear or non-linear matter power spectrum. Linear power spectrum is given in 
        in terms of the linear transfer function and growth factor as 

        .. math::
            P_{\rm lin}(k, z) = A k^{n_s} T(k)^2 D_+(z)^2

        Non-linear power spectrum can be related to the linear power by some transformation. The 
        dimenssionless power spectrum is then defined as

        .. math::
            \Delta^2(k, z) = \frac{1}{2\pi^2} k63 P(k, z)

        where the power spectrum could be linear or non-linear.

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        dim: bool, optional
            If true (default), return the usual power spectrum, else give the dimenssionless one.
        linear: bool, optional
            If true (default) return the linear power spectrum, else the non-linear power spectrum.
        
        Returns
        -------
        pk: array_like
            Matter (dimenssionless) power spectrum values.

        """
        if linear:
            return self.linearPowerSpectrum( k, z, dim ) 
        return self.nonlinearPowerSpectrum( k, z, dim )

    def matterCorrelation(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the linear or non-linear 2-point matter correlation function.

        Parameters
        ----------
        r: array_like
            Seperation between the two points in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the linear correlation, else the non-linear correlation.
        
        Returns
        -------
        xr: array_like
            Matter correlation function values.

        """
        return filters.j0convolution(  self.matterPowerSpectrum, r, args = ( z, False, linear ) )

    def variance(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the linear or non-linear matter fluctuations variance.

        Parameters
        ----------
        r: array_like
            Smoothing radius in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the linear variance, else the non-linear variance.
        
        Returns
        -------
        var: array_like
            Matter fluctuation variance.

        """
        return self.filter.convolution( self.matterPowerSpectrum, r, args = ( z, False, linear ) )

    def dlnsdlnr(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the first logarithmic derivative of matter fluctuations variance w.r.to radius.

        Parameters
        ----------
        r: array_like
            Smoothing radius in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the value for linear variance, else for non-linear variance.
        
        Returns
        -------
        y: array_like
            Values of the derivative.

        """
        r  = np.asfarray( r )
        y0 = self.variance( r, z, linear )
        y1 = self.filter.dcdr( self.matterPowerSpectrum, r, args = ( z, False, linear ) )
        return 0.5 * r * y1 / y0

    def d2lnsdlnr2(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the second logarithmic derivative of matter fluctuations variance w.r.to radius.

        Parameters
        ----------
        r: array_like
            Smoothing radius in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the value for linear variance, else for non-linear variance.
        
        Returns
        -------
        y: array_like
            Values of the derivative.

        """
        h     = settings.DEFAULT_H
        r     = np.asfarray( r )

        df    = (
                    -self.dlnsdlnr( ( 1+2*h )*r, z, linear )
                        + 8*self.dlnsdlnr( ( 1+h )*r, z, linear )
                        - 8*self.dlnsdlnr( ( 1-h )*r, z, linear )
                        +   self.dlnsdlnr( ( 1-2*h )*r, z, linear )
                ) # f := dlns/dlnr
               
        dlnr = 6.0 * ( np.log( (1+h)*r ) - np.log( (1-h)*r ) )
        
        return df / dlnr

    def radius(self, sigma: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Invert the variance to find the smoothing radius.

        Parameters
        ----------
        sigma: array_like
            Variance values (linear or non-linear, specified by `linear` argument), to be exact, their 
            square roots.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) use the linear variance, else the non-linear variance.
        
        Returns
        -------
        r: array_like
            Smoothing radius in Mpc/h.

        """
        def f(lnr: Any, v: Any, z: float, linear: bool) -> Any:
            r = np.exp( lnr )
            return self.variance( r, z, linear ) - v

        v   = np.asfarray( sigma )**2
        lnr = numeric.solve( 
                                f, a = np.log( 1e-04 ), b = np.log( 1e+04 ), 
                                args = ( v, z, linear ), tol = settings.RELTOL 
                           )
        return np.exp( lnr )

    def effectiveIndex(self, k: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the effective power spectrum index (effective slope).

        .. math::
            n_{]\rm eff}(k) = \frac{ {\rm d}\ln P(k) }{ {\rm d}\ln k }

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the index for linear power spectrum, else the non-linear power spectrum.
        
        Returns
        -------
        neff: array_like
            Power spectrum index values.
        """
        def lnPower(k: Any, z: float, linear: bool) -> Any:
            return np.log( self.matterPowerSpectrum( k, z, dim = True, linear = linear ) )

        h    = settings.DEFAULT_H
        k    = np.asfarray( k )
        dlnp = (
                    -lnPower( (1+2*h)*k, z, linear ) 
                        + 8*lnPower( (1+h)*k,   z, linear ) 
                        - 8*lnPower( (1-h)*k,   z, linear ) 
                        +   lnPower( (1-2*h)*k, z, linear )
               )
               
        dlnk = 6.0 * ( np.log( (1+h)*k ) - np.log( (1-h)*k ) )
        
        return dlnp / dlnk

    def normalize(self) -> None:
        r"""
        Normalize the power spectrum using the value of :math:`\sigma_8` parameter.
        """
        self.A = 1.0 # power spectrum normalization factor
        self.A = self.sigma8**2 / self.variance( 8.0 ) 
