from typing import Any 
from abc import ABC, abstractmethod
import numpy as np
import pycosmo2.utils.constants as const
import pycosmo2.utils.numeric as numeric
import pycosmo2.utils.settings as settings
import pycosmo2.cosmology.cosmo as cosmo
import pycosmo2.power.filters as filters
import pycosmo2.power.transfer_functions as tf

class PowerSpectrumError(Exception):
    ...

class PowerSpectrum(ABC):

    __slots__ = 'linear_model', 'nonlinear_model', 'filter', 'cosmology', 'A', 'use_exact_growth'

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        
        if not isinstance( cm, cosmo.Cosmology ):
            raise PowerSpectrumError("cm must be a 'Cosmology' object")
        self.cosmology        = cm
        self.use_exact_growth = False # whether to use exact (integrated) growth factors

        if filter not in filters.filters:
            raise PowerSpectrumError(f"invalid filter: { filter }")
        self.filter = filters.filters[ filter ] # filter to use for smoothing

        self.linear_model    = None      # linear power spectrum model name 
        self.nonlinear_model = 'halofit' # non-linear power spectrum model name 

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
            raise PowerSpectrumError("z must be a scalar")
        if z + 1 < 0:
            raise ValueError("redshift cannot be less than -1")

        Pk = self.A * k**self.ns * self.transferFunction( k, z )**2 * self.Dplus( z )**2
        if not dim:
            return k**3 * Pk / ( 2*np.pi**2 )
        return Pk

    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        ...

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, linear: bool = True) -> Any:
        if not linear:
            return NotImplemented
        return self.linearPowerSpectrum( k, z, dim )        

    def variance(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        return self.filter.convolution( self.matterPowerSpectrum, r, args = ( z, False, linear ) )

    def dlnsdlnr(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r  = np.asfarray( r )
        y0 = self.variance( r, z, linear )
        y1 = self.filter.dcdr( self.matterPowerSpectrum, r, args = ( z, False, linear ) )
        return 0.5 * r * y1 / y0

    def d2lnsdlnr2(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        raise NotImplementedError()

    def neff(self, k: Any, z: float = 0, linear: bool = True) -> Any:
        def lnP(k: Any, z: float, linear: bool) -> Any:
            return np.log( self.matterPowerSpectrum( k, z, dim = True, linear = linear ) )

        h    = settings.DEFAULT_H
        k    = np.asfarray( k )
        dlnp = (
                    -lnP( (1+2*h)*k, z, linear ) 
                        + 8.0 * lnP( (1+h)*k, z, linear ) 
                        - 8.0 * lnP( (1-h)*k, z, linear ) 
                        + lnP( (1-2*h)*k, z, linear )
               )
               
        dlnk = 12.0 * ( np.log( (1+h)*k ) - np.log( (1-h)*k ) )
        
        return dlnp / dlnk

    def nonlineark(self, lineark: Any, z: float = 0) -> Any:
        ...

    def normalize(self) -> None:
        self.A = 1.0 # power spectrum normalization factor
        self.A = self.sigma8**2 / self.variance( 8.0 ) 




class Sugiyama96( PowerSpectrum ):

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return tf.psmodelSugiyama96( self.cosmology, k, z )