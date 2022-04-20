import numpy as np
import pycosmo.power as pm
from typing import Any


class Cosmology:
    """
    A flat Lambda-cdm cosmology model.
    """
    __slots__ = (
                    'h', 'Om0', 'Ob0', 'Ode0', 'Onu0', 'Nnu', 'mnu', 'ns', 'Tcmb0', 'A', 
                    'sigma8', 'psmodel', 
                )

    def __init__(self, h: float, Om0: float, Ob0: float, ns: float, sigma8: float = None, Nnu: float = 0.0, mnu: float = 0.0) -> None:
        if h <= 0:
            raise ValueError("'h' must be a positive value")
        elif Om0 < 0 or Om0 > 1:
            raise ValueError("'Om0' must be a value between 0 and 1")

        self.Onu0, self.Nnu, self.mnu = 0.0, 0.0, 0.0
        if mnu < 0 or Nnu < 0:
            raise ValueError("'Nnu' and 'mnu' must be positive numbers")
        Onu0 = Nnu * mnu / 91.5 / h**2
        if Onu0 > Om0:
            raise ValueError("'Onu0' cannot be greater than 'Om0'")

        if Ob0 < 0:
            raise ValueError("'Ob0' must be a positive value")
        elif Ob0 + Onu0 > Om0:
            raise ValueError("sum of 'Ob0' and 'Onu0' cannot be greater than 'Om0'")

        self.h, self.Om0, self.Ob0    = h, Om0, Ob0
        self.Onu0, self.Nnu, self.mnu = Onu0, Nnu, mnu
        
        self.Ode0 = 1 - self.Om0
        self.ns   = ns

        self.A, self.Tcmb0 = 1.0, 2.725
        self.psmodel       = 'eisenstein98_zb'

        self.sigma8        = None
        if sigma8 is not None:
            if sigma8 <= 0:
                raise ValueError("'sigma8' must be a positive value")
            self.normalize( sigma8 )

    def Ez(self, z: Any) -> Any:
        """
        Evolution of hubble parameter.
        """
        zp1 = np.asfarray( z ) + 1
        return np.sqrt( self.Om0 * zp1**3 + self.Ode0 )

    def Hz(self, z: Any) -> Any:
        """
        Evolution of hubble parameter.
        """
        return 100.0 * self.h * self.Ez( z )

    def Omz(self, z: Any) -> Any:
        """
        Evolution of matter density.
        """
        zp1 = np.asfarray( z ) + 1
        y   = self.Om0 * zp1**3
        return y / ( y + self.Ode0 )

    def Odez(self, z: Any) -> Any:
        """
        Evolution of dark-energy density.
        """
        zp1 = np.asfarray( z ) + 1
        return self.Ode0 / ( self.Om0 * zp1**3 + self.Ode0 )

    def gz(self, z: Any) -> Any:
        """
        Fitting function for linear growth factor.
        """
        Om, Ode = self.Omz( z ), self.Odez( z )
        return 2.5 * Om * ( 
                            Om**(4.0/7.0) 
                                - Ode 
                                + ( 1 + Om / 2.0 ) * ( 1 + Ode / 70.0 )
                          )**( -1 )

    def Dz(self, z: Any, fac: float = None) -> Any:
        """
        Linear growth factor.
        """
        def _Dz(z: Any) -> Any:
            return self.gz( z ) / ( np.asfarray( z ) + 1 )
        
        if fac is None:
            fac = 1.0 / _Dz( 0.0 )
        return _Dz( z ) * fac 

    def fz(self, z: Any) -> Any:
        """
        Linear growth rate.
        """
        return self.Omz( z )**0.6

    def growthSuppressionFactor(self, q: Any, z: float, nu: bool = False, fac: float = None) -> Any:
        """
        Suppression of growth of fluctuations in presence of neutrinos.
        """
        q   = np.asfarray( q )    

        if self.Onu0 < 1.0E-08:
            return np.ones_like( q )

        fnu = self.Onu0 / self.Om0
        fcb = 1 - fnu
        pcb = 0.25*( 5 - np.sqrt( 1 + 24.0*fcb ) )
        yfs = 17.2 * fnu * ( 1 + 0.488*fnu**(-7.0/6.0) ) * ( self.Nnu*q / fnu )**2
        D1  = self.Dz( z, fac )    

        x   = ( D1 / ( 1 + yfs ) )**0.7
        if nu:
            return ( fcb**( 0.7 / pcb ) + x )**( pcb / 0.7 ) * D1**( -pcb )
        return ( 1 + x )**( pcb / 0.7 ) * D1**( -pcb )

    def transfer(self, k: Any, z: float = 0, model: str = None) -> Any:
        """
        Transfer function.
        """
        if model is None:
            model = self.psmodel
        return pm.transfer( model, self, k, z )

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, model: str = None) -> Any:
        """
        Compute linear matter power spectrum. 
        """
        k  = np.asfarray( k )
        pk = self.A * k**self.ns * self.transfer( k, z, model )**2 * self.Dz( z )**2
        if dim:
            return pk
        return pk * k**3 / ( 2*np.pi**2 )

    def variance(self, r: Any, z: float = 0, ka: float = 1.0E-08, kb: float = 1.0E+08, pts: int = 10001, model: str = None) -> Any:
        """ 
        Compute the linear matter variance.
        """

        def filt(x: Any) -> Any:
            return ( np.sin( x ) - x * np.cos( x )) * 3. / x**3 

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr  = np.outer(r, k)
        y   = self.matterPowerSpectrum( k, z, dim = False, model = model ) * filt( kr )**2 
        var = ( 
                    y[ ..., :-1:2 ].sum(-1) + 4 * y[ ..., 1::2 ].sum(-1) + y[ ..., 2::2 ].sum(-1)
              ) * dlnk / 3

        return var if np.ndim(r) else var[0] 

    def correlation(self, r: Any, z: float = 0, ka: float = 1.0E-08, kb: float = 1.0E+08, pts: int = 10001, model: str = None) -> Any:
        """
        Linear matter correlation function.
        """

        def sinc(x: Any) -> Any:
            return np.sinc( x / np.pi )

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr  = np.outer(r, k)
        y   = self.matterPowerSpectrum( k, z, dim = False, model = model ) * sinc( kr )
        cor = ( 
                    y[ ..., :-1:2 ].sum(-1) + 4 * y[ ..., 1::2 ].sum(-1) + y[ ..., 2::2 ].sum(-1)
              ) * dlnk / 3

        return cor if np.ndim(r) else cor[0]

    def powerNorm(self, sigma8: float, **kwargs: Any) -> float:
        """
        Get the power spectrum normalization without setting it.
        """ 
        return sigma8**2 / self.variance( 8.0, **kwargs )

    def normalize(self, sigma8: float, **kwargs: Any) -> None:
        """
        Normalize the power spectrum.
        """
        self.A      = 1.0
        self.sigma8 = sigma8
        self.A      = self.powerNorm( sigma8, **kwargs )

