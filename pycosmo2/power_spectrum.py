from abc import ABC, abstractmethod
from typing import Any
import numpy as np
import pycosmo2.constants as const

class PowerSpectrumError( Exception ):
    r"""
    Base class of exceptions used by power spectrum objects.
    """
    ...

class PowerSpectrum( ABC ):
    r"""
    A class representing the matter power spectrum.
    """
    
    __slots__ = 'transfer_function', 'nonlinear_model', 'filter', 'A', 'cosmology', 'attrs'

    def __init__(self, cm: object, transfer_function: str, nonlinear_model: str, filter: str) -> None:
        self.transfer_function = transfer_function
        self.nonlinear_model   = nonlinear_model
        self.filter            = filter

        import cosmology
        if not isinstance( cm, cosmology.Cosmology ):
            raise PowerSpectrumError( "cm must be a cosmology model object" )
        self.cosmology = cm

        self.A     = 1.0   # normalization of the power spectrum
        self.attrs = set() # set of attributes

    @abstractmethod
    def transferFunction(self, k: Any, *args, **kwargs) -> Any:
        r"""
        Compute the linear transfer function as function of wavenumber. The wavenumber 
        k must be in units of h/Mpc.
        """
        ...

    @abstractmethod
    def matterPowerSpectrum(self, k: Any, z: float, dim: bool = True) -> Any:
        r"""
        Compute the matter power spectrum as function of wavenumber. The wavenumber k should 
        be in units of h/Mpc and the power spectrum computed will have units of :math:`h^3/{\rm Mpc}^3`.
        """
        ...

    @abstractmethod
    def linearPowerSpectrum(self, k: Any, z: float, dim: bool = True) -> Any:
        r"""
        Compute the linear matter power spectrum as function of wavenumber. Units are similar 
        to :meth:`matterPowerSpectrum` function.
        """
        ...

    @abstractmethod
    def nonlinearPowerSpectrum(self, k: Any, z: float, dim: bool = True) -> Any:
        r"""
        Compute the non-linear power spectrum as function of wavenumber. Units are similar 
        to :meth:`matterPowerSpectrum` function.
        """
        ...

    @abstractmethod
    def variance(self, r: Any, z: float, linear: bool = True) -> Any:
        r"""
        Compute the matter fluctuations variance as function of smoothing radius r. Radius 
        should be in units of Mpc/h and the result is unitless.
        """
        ...

    