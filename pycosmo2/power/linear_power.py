from typing import Any 
from abc import ABC, abstractmethod
import numpy as np
import pycosmo2.utils.settings as settings
import pycosmo2.utils.numeric as numeric
import pycosmo2.cosmology.cosmo as cosmo
import pycosmo2.power.filters as filters

class LinearPowerSpectrumError(Exception):
    ...

class LinearPowerSpectrum(ABC):

    __slots__ = 'filter', 'cosmology', 'A', 'use_exact_growth'

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        
        if not isinstance( cm, cosmo.Cosmology ):
            raise LinearPowerSpectrumError("cm must be a 'Cosmology' object")
        self.cosmology        = cm
        self.use_exact_growth = False # whether to use exact (integrated) growth factors

        if filter not in filters.filters:
            raise LinearPowerSpectrumError(f"invalid filter: { filter }")
        self.filter = filters.filters[ filter ] # filter to use for smoothing

        self.normalize()

    def Dplus(self, z: Any) -> Any:
        return self.cosmology.Dplus( z, exact = self.use_exact_growth )
    
    @property 
    def ns(self) -> float:
        return self.cosmology.ns

    @property 
    def sigma8(self) -> float:
        return self.cosmology.sigma8
    
    @abstractmethod
    def transferFunction(self, k: Any, z: float = 0) -> Any:
        ...

    def linearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        k = np.asfarray( k )

        if np.ndim( z ):
            raise LinearPowerSpectrumError("z must be a scalar")
        if z + 1 < 0:
            raise ValueError("redshift cannot be less than -1")

        Pk = self.A * k**self.ns * self.transferFunction( k, z )**2 * self.Dplus( z )**2
        if not dim:
            return k**3 * Pk / ( 2*np.pi**2 )
        return Pk  

    # non-linear power and scale will be implemented later in a sub-class

    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        raise NotImplementedError("non-linear models are implemented in a general power spectrum subclass")

    def nonlineark(self, k: Any, z: float = 0) -> Any:
        raise NotImplementedError("non-linear models are implemented in a general power spectrum subclass")

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, linear: bool = True) -> Any:
        if linear:
            return self.linearPowerSpectrum( k, z, dim ) 
        return self.nonlinearPowerSpectrum( k, z, dim )

    def matterCorrelation(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        return filters.j0convolution(  self.matterPowerSpectrum, r, args = ( z, False, linear ) )

    def variance(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        return self.filter.convolution( self.matterPowerSpectrum, r, args = ( z, False, linear ) )

    def dlnsdlnr(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r  = np.asfarray( r )
        y0 = self.variance( r, z, linear )
        y1 = self.filter.dcdr( self.matterPowerSpectrum, r, args = ( z, False, linear ) )
        return 0.5 * r * y1 / y0

    def d2lnsdlnr2(self, r: Any, z: float = 0, linear: bool = True) -> Any:
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
        self.A = 1.0 # power spectrum normalization factor
        self.A = self.sigma8**2 / self.variance( 8.0 ) 
