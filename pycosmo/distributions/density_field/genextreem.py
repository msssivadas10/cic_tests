from typing import Any 
from itertools import product, repeat
from scipy.interpolate import CubicSpline
from pycosmo.cosmology import Cosmology
from pycosmo.distributions.base import Distribution, DistributionError
import pycosmo.utils.numeric as numeric 
import pycosmo.utils.settings as settings
import numpy as np

class GenExtremeDistribution(Distribution):

    # global setting for objects
    INTERP_N     = 51    # number of interpolation points
    MEAN_N       = 501   # number of samples for averaging
    EXACT_GROWTH = False # use exact growth factor 

    __slots__ = 'cosmology', 'z', 'r', 'kn', 'b2_log', 'meas_power_spectrum'

    def __init__(self, cm: Cosmology, r: float, z: float = 0) -> None:
        
        if not isinstance( cm, Cosmology ):
            raise TypeError("cm must be a 'Cosmology' object")
        self.cosmology = cm

        self.r  = r         # size of the box (Mpc/h)
        self.kn = np.pi / r # nyquist wavenumber (h/Mpc)

        self.z  = z

        self.b2_log    = 1.0 # log field bias 

        self._prepareSpline()

    def pdf(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def setup(self, *args: Any, **kwargs: Any) -> Any:
        ...

    def supportInterval(self) -> tuple:
        ...
    
    def measuredPowerSepctrum(self, kx: Any, ky: Any, kz: Any, z: float = 0) -> Any:
        r"""
        Return the measured log-field power spectrum.
        """
        kx, ky, kz = np.asfarray( kx ), np.asfarray( ky ), np.asfarray( kz )
        
        lnk = 0.5*np.log( kx**2 + ky**2 + kz**2 )
        pk  = np.exp( self.meas_power_spectrum( lnk ) )
        return pk * self.cosmology.Dplus( z, exact = self.EXACT_GROWTH ) * self.b2_log

    def _measuredPowerSpectrum(self, kx: Any, ky: Any, kz: Any) -> Any:

        def weightedPowerTerm(kx: Any, ky: Any, kz: Any) -> Any:
            p = 1
            k  = np.sqrt( kx**2 + ky**2 + kz**2 )

            # log field power spectrum
            Pk = self.cosmology.linearPowerSpectrum( k, z = 0, dim = True )

            # mass-assignment function
            Wk = ( np.sinc( 0.5*kx / self.kn ) * np.sinc( 0.5*ky / self.kn ) * np.sinc( 0.5*kz / self.kn ) )**p

            return Pk * Wk**2 

        kx, ky, kz = np.asfarray( kx ), np.asfarray( ky ), np.asfarray( kz )

        y = 0.0
        for nx, ny, nz in product( *repeat( range(3), 3 ) ):

            # need only n's with length < 3
            if nx**2 + ny**2 + nz**2 > 9:
                continue

            y += weightedPowerTerm( kx + 2*nx*self.kn, ky + 2*ny*self.kn, kz + 2*nz*self.kn )
        
        return y 

    def _prepareSpline(self) -> None:

        k     = np.logspace( -4, np.log10( self.kn ), self.INTERP_N )
        theta = np.random.uniform( 0.0,   np.pi, ( self.MEAN_N, self.INTERP_N ) )
        phi   = np.random.uniform( 0.0, 2*np.pi, ( self.MEAN_N, self.INTERP_N ) )

        Pk = self._measuredPowerSpectrum( 
                                            kx = k * np.sin( theta ) * np.cos( phi ),
                                            ky = k * np.sin( theta ) * np.sin( phi ),
                                            kz = k * np.cos( theta )
                                        ).mean( axis = 0 ) 

        self.meas_power_spectrum = CubicSpline( np.log(k), np.log(Pk), bc_type = 'natural' )
        return

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

    def sigma2Box(self, z: float = 0) -> float:
        r"""
        Return the measured value of log-field variance, :math:`\sigma^2_A(l)`.

        Parameters
        ----------
        z: float, optional
            Redshift.

        Returns
        -------
        var: float
            Log field variance.

        """
    









