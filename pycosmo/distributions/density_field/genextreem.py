from typing import Any 
from pycosmo.cosmology import Cosmology
from pycosmo.distributions.base import Distribution, DistributionError
import pycosmo.utils.numeric as numeric 
import pycosmo.utils.settings as settings
import numpy as np

class GenExtremeDistribution(Distribution):

    __slots__ = 'cosmology', 'z', 'r', 'kn', 'b_log'

    def __init__(self, cm: Cosmology, r: float, z: float = 0) -> None:
        
        if not isinstance( cm, Cosmology ):
            raise TypeError("cm must be a 'Cosmology' object")
        self.cosmology = cm

        self.r  = r         # size of the box (Mpc/h)
        self.kn = np.pi / r # nyquist wavenumber (h/Mpc)

        self.z  = z

        self.b_log = 1.0 # log field bias 
        
    def sigma2Linear(self, z: float = 0) -> float:
        r"""
        Linear variance in the box, using a k-space tophat filter (sharp-k filter) :math:`\sigma^2_{\rm lin}( k_N )`.

        Parameters
        ----------
        z: float, optional
            Redshift.

        Returns
        -------
        var: float
            Linear variance.

        """
        def Delta2(k: Any, z: float) -> Any:
            return self.cosmology.linearPowerSpectrum( k, z, dim = False )
        
        var = numeric.integrate1( 
                                    Delta2, 
                                    a = np.log( settings.ZERO ), b = np.log( self.kn ), 
                                    args = (z, ), 
                                    subdiv = settings.DEFAULT_SUBDIV 
                                )
        return var

    def sigma2A(self, arg: float) -> float:
        r"""
        Return the fitted value of the log-field variance, :math:`\sigma^2_A( k_N )`.

        Parameters
        ----------
        arg: float
            Linear variance value.
        
        Returns
        -------
        var: float
            Best fiiting value of logarithmic field variance.

        """
        mu = 0.73
        return mu * np.log( 1 + arg / mu )
    
    def measuredPowerSepctrum(self, kx: Any, ky: Any, kz: Any) -> Any:
        r"""
        Return the measured log-field power spectrum.
        """





