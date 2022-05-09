from typing import Any 
from abc import ABC, abstractmethod
import numpy as np
import pycosmo2.utils.constants as const
import pycosmo2.utils.numeric as numeric
import pycosmo2.utils.settings as settings
import pycosmo2.cosmology.cosmo as cosmo

class PowerSpectrumError(Exception):
    ...

class PowerSpectrum(ABC):

    __slots__ = 'linear_model', 'nonlinear_model', 'filter', 'cosmology', 'A'

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        
        if not isinstance( cm, cosmo.Cosmology ):
            raise PowerSpectrumError("cm must be a 'Cosmology' object")
        self.cosmology = cm

        self.filter = filter # filter to use for smoothing

        self.linear_model    = None      # linear power spectrum model name 
        self.nonlinear_model = 'halofit' # non-linear power spectrum model name 

        self.A = 1.0 # power spectrum normalization factor
    
    @abstractmethod
    def transferFunction(self, k: Any, z: float = 0) -> Any:
        ...

    def linearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        ...

    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        ...

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, linear: bool = True) -> Any:
        ...

    def variance(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        ...

    def dlnsdlnr(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        ...

    def d2lnsdlnr2(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        ...

    def neff(self, k: Any, z: float = 0, linear: bool = True) -> Any:
        ...

    def nonlineark(self, lineark: Any, z: float = 0) -> Any:
        ...

    def normalize(self) -> None:
        ...

