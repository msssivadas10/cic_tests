#!\usr\bin\python3

from typing import Any
import numpy as np
import pycosmo.cosmology.power as pm
import pycosmo.constants as const

class Cosmology:
    """
    A general cosmology model.
    """
    __slots__ = (
                    'h', 'Om0', 'Ob0', 'Ode0', 'Ok0', 'Onu0', 'Nnu', 'mnu', 'ns', 'Tcmb0', 
                    'A', 'sigma8', 'psmodel', 'flat'
                )

    def __init__(self, flat: bool, h: float, Om0: float, Ob0: float, ns: float, Ode0: float = None, sigma8: float = None, Nnu: float = 0.0, Onu0: float = 0.0, psmodel: str = 'eisenstein98_zb', Tcmb0: float = 2.725) -> None:
        if h <= 0:
            raise ValueError("parameter 'h' must be positive")
        self.h = h

        self.Nnu, self.mnu, self.Onu0 = 0.0, 0.0, 0.0
        if ( Nnu != 0 ) and ( Onu0 != 0 ): 
            if ( Nnu < 0.0 ) or ( Onu0 < 0.0 ):
                raise ValueError("parameters 'Nnu' and 'Onu0' must be positive")
            self.Nnu  = Nnu
            self.Onu0 = Onu0
            self.mnu  = 91.5*Onu0*h**2 / Nnu

        if ( Om0 < 0.0 ):
            raise ValueError("parameters 'Om0' must be positive")
        if ( Ob0 < 0.0 ) or ( Ob0 > Om0 ):
            raise ValueError("parameter 'Ob0' must be positive and cannot be greater than 'Om0'")
        if ( self.Onu0 > Om0 ):
            raise ValueError("'Onu0' cannot be greater than 'Om0'")

        self.Om0, self.Ob0 = Om0, Ob0
        
        self.flat = bool( flat )
        if not flat:
            if Ode0 is None:
                raise ValueError("parameter 'Ode0' is required for non-flat cosmologies")
            if ( Ode0 < 0.0 ):
                raise ValueError("parameter 'Ode0' must be positive")
            
            self.Ode0 = Ode0
            self.Ok0  = 1.0 - self.Om0 - self.Ode0
        else:
            if ( self.Om0 > 1 ):
                raise ValueError("parameter 'Om0' cannot be greater than 1")

            self.Ode0 = 1.0 - self.Om0
            self.Ok0  = 0.0
        
        if ( self.Ob0 + self.Onu0 > self.Om0 ):
            raise ValueError("total baryon and neutrino content cannot be greater than that of matter")

        self.ns = ns
        self.A  = 1.0

        if ( Tcmb0 <= 0.0 ):
            raise ValueError("parameter 'Tcmb0' must be a positive value")
        self.Tcmb0 = Tcmb0

        if not pm.available( psmodel ):
            raise ValueError("model not available: '{}'".format( psmodel ))
        self.psmodel = psmodel

        self.sigma8 = None
        if sigma8 is not None:
            self.normalize( sigma8 )       

    def Ez(self, z: Any) -> Any:
        """
        Evolution of hubble parameter.
        """
        zp1 = np.asfarray( z ) + 1
        y   = self.Om0 * zp1**3 + self.Ode0
        if not self.flat:
            y += self.Ok0 * zp1**2
        return np.sqrt( y )

    def Hz(self, z: Any) -> Any:
        """
        Evolution of hubble parameter.
        """
        return 100.0 * self.h * self.Ez( z )
    
    def criticalDensity(self, z: Any) -> Any:
        """
        Critical density of the universe in kg/m^3.
        """
        return const.RHO_CRIT0_ASTRO * self.Ez( z )

    def rho_m(self, z: Any) -> Any:
        """
        Evolution of matter density.
        """
        return self.Om0 * ( np.asfarray( z ) + 1 )**3 * const.RHO_CRIT0_ASTRO

    def rho_de(self, z: Any) -> Any:
        """
        Evolution of dark-energy density.
        """
        return self.Ode0 * ( np.asfarray( z ) + 1 )**3 * const.RHO_CRIT0_ASTRO

    def Omz(self, z: Any) -> Any:
        """
        Evolution of matter density.
        """
        zp1 = np.asfarray( z ) + 1
        y1  = self.Om0 * zp1**3
        y2  = y1 + self.Ode0
        if not self.flat:
            y2 += self.Ok0 * zp1**2
        return y1 / y2

    def Odez(self, z: Any) -> Any:
        """
        Evolution of dark-energy density.
        """
        zp1 = np.asfarray( z ) + 1
        y   = self.Om0 * zp1**3 + self.Ode0
        if not self.flat:
            y += self.Ok0 * zp1**2
        return self.Ode0 / y

    def gz(self, z: Any) -> Any:
        """
        Fitting function for linear growth factor.
        """
        zp1     = np.asfarray( z ) + 1
        Om, Ode = self.Om0 * zp1**3, self.Ode0
        
        y = Om + Ode
        if not self.flat:
            y += self.Ok0 * zp1**2

        Om, Ode = Om / y, Ode / y
        return 2.5 * Om * ( 
                            Om**(4.0/7.0) 
                                - Ode 
                                + ( 1 + Om / 2.0 ) * ( 1 + Ode / 70.0 )
                          )**( -1 )

    def Dz(self, z: Any, fac: float = None, exact: bool = False, zb: float = 1.0E+8, pts: int = 10001) -> Any:
        """
        Linear growth factor.
        """
        def _Dz(z: Any, exact: bool, zb: float, pts: int) -> Any:
            zp1 = z + 1

            if not exact:
                return self.gz( z ) / zp1

            if zb < -1:
                raise ValueError("redshift 'zb' cannot be less than -1")
            if pts < 3:
                raise ValueError("'pts' must be greater than 2")
            elif not ( pts % 2 ):
                pts = pts + 1
            
            xp1, dlnxp1 = np.linspace( np.log( zp1 ), np.log( zb + 1 ), pts, retstep = True )
            xp1         = np.exp( xp1 )

            y   = self.Om0 * xp1**3 + self.Ode0
            if not self.flat:
                y += self.Ok0 * xp1**2
            y   = xp1**2 / y**1.5
            y   = ( 
                        y[ :-1:2,... ].sum(0) + 4 * y[ 1::2,... ].sum(0) + y[ 2::2,... ].sum(0)
                    ) * dlnxp1 / 3

            w   = self.Om0 * zp1**3 + self.Ode0
            if not self.flat:
                w += self.Ok0 * zp1**2
                
            return 2.5 * self.Om0 * y * np.sqrt( w )
        
        z = np.asfarray( z )
        if np.ndim( z ) > 1:
            raise TypeError("array dimension should be less than 2")
        if np.any( z < -1 ):
            raise ValueError("redshift 'z' cannot be less than -1")

        if fac is None:
            fac = 1.0 / _Dz( 0.0, exact, zb, pts )
        return _Dz( z, exact, zb, pts ) * fac 

    def fz(self, z: Any, exact: bool = False, zb: float = 1.0E+8, pts: int = 10001) -> Any:
        """
        Linear growth rate.
        """
        if not exact:
            return self.Omz( z )**0.6

        zp1    = np.asfarray( z ) + 1
        y1, y2 = self.Om0*zp1**3, None
        
        y = y1 + self.Ode0
        if not self.flat:
            y2 = self.Ok0*zp1**2
            y += y2
        
        y1 = y1 * ( 2.5 / ( zp1*self.Dz( z, 1.0, exact, zb, pts ) ) - 1.5 )
        if y2 is not None:
            y1 += y2
        return ( y1 / y )

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
        if np.ndim( r ) > 1:
            raise TypeError("array dimension should be less than 2")

        if pts < 3:
            raise ValueError("'pts' must be greater than 2")
        elif ( pts % 2 ):
            pts = pts + 1
        
        if ka < 0.0 or kb < 0.0:
            raise ValueError("'ka' and 'kb' should be positive")
        elif kb <= ka:
            raise ValueError("'ka' should be less than 'kb'")

        def filt(x: Any) -> Any:
            return ( np.sin( x ) - x * np.cos( x )) * 3. / x**3 

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr  = np.outer(r, k)
        y   = self.matterPowerSpectrum( k, z, dim = False, model = model ) * filt( kr )**2 
        var = ( 
                    y[ ...,:-1:2 ].sum(-1) + 4 * y[ ...,1::2 ].sum(-1) + y[ ...,2::2 ].sum(-1)
              ) * dlnk / 3

        return var if np.ndim(r) else var[0] 

    def dlnsdlnr(self, r: Any, z: float = 0, ka: float = 1.0E-08, kb: float = 1.0E+08, pts: int = 10001, model: str = None) -> Any:
        """ 
        Compute the logarithmic derivative of linear matter variance.
        """
        if np.ndim( r ) > 1:
            raise TypeError("array dimension should be less than 2")

        if pts < 3:
            raise ValueError("'pts' must be greater than 2")
        elif ( pts % 2 ):
            pts = pts + 1
        
        if ka < 0.0 or kb < 0.0:
            raise ValueError("'ka' and 'kb' should be positive")
        elif kb <= ka:
            raise ValueError("'ka' should be less than 'kb'")

        def filt(x: Any) -> Any:
            return ( np.sin( x ) - x * np.cos( x ) ) * 3.0 / x**3 

        def dfilt(x: Any) -> Any:
            return ( ( x**2 - 3.0 ) * np.sin( x ) + 3.0 * x * np.cos( x ) ) * 3.0 / x**4

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr = np.outer(r, k)
        y1 = self.matterPowerSpectrum( k, z, dim = False, model = model ) * filt( kr )
        y2 = y1 * k * dfilt( kr )
        y1 = y1 * filt( kr )

        q1 = y1[ ..., :-1:2 ].sum(-1) + 4 * y1[ ..., 1::2 ].sum(-1) + y1[ ..., 2::2 ].sum(-1)
        q2 = y2[ ..., :-1:2 ].sum(-1) + 4 * y2[ ..., 1::2 ].sum(-1) + y2[ ..., 2::2 ].sum(-1)        

        return r * ( q2 / q1 if np.ndim(r) else q2[0] / q1[0] ) 

    def dlnsdlnm(self, r: Any, z: float = 0, ka: float = 1.0E-08, kb: float = 1.0E+08, pts: int = 10001, model: str = None) -> Any:
        """ 
        Compute the logarithmic derivative of linear matter variance.
        """
        return self.dlnsdlnr( r, z, ka, kb, pts, model ) / 3.0

    def correlation(self, r: Any, z: float = 0, ka: float = 1.0E-08, kb: float = 1.0E+08, pts: int = 10001, model: str = None) -> Any:
        """
        Linear matter correlation function.
        """
        if np.ndim( r ) > 1:
            raise TypeError("array dimension should be less than 2")

        if pts < 3:
            raise ValueError("'pts' must be greater than 2")
        elif ( pts % 2 ):
            pts = pts + 1
        
        if ka < 0.0 or kb < 0.0:
            raise ValueError("'ka' and 'kb' should be positive")
        elif kb <= ka:
            raise ValueError("'ka' should be less than 'kb'")

        def sinc(x: Any) -> Any:
            return np.sinc( x / np.pi )

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr  = np.outer(r, k)
        y   = self.matterPowerSpectrum( k, z, dim = False, model = model ) * sinc( kr )
        var = ( 
                    y[ ...,:-1:2 ].sum(-1) + 4 * y[ ...,1::2 ].sum(-1) + y[ ...,2::2 ].sum(-1)
              ) * dlnk / 3

        return var if np.ndim(r) else var[0] 
    
    def _correlation(self, r: Any, z: float = 0, ka: float = 1.0E-08, kb: float = 1.0E+08, pts: int = 101, model: str = None) -> Any:
        """
        Linear matter correlation function (note: slow).
        """
        if pts < 3:
            raise ValueError("'pts' must be greater than 2")
        elif ( pts % 2 ):
            pts = pts + 1
        
        if ka < 0.0 or kb < 0.0:
            raise ValueError("'ka' and 'kb' should be positive")
        elif kb <= ka:
            raise ValueError("'ka' should be less than 'kb'")

        def sinc(x: Any) -> Any:
            return np.sinc( x / np.pi )

        def _correlation(r: float) -> float:
            kai, kbi = ka, np.pi / r
            cor      = 0.0
            while 1:
                k, dlnk = np.linspace( np.log( kai ), np.log( kbi ), pts, retstep = True )
                k       = np.exp( k )    

                # integration done in log(k) variable
                y = self.matterPowerSpectrum( k, z, dim = False, model = model ) * sinc( k*r )
                q = ( 
                        y[ :-1:2 ].sum(-1) + 4 * y[ 1::2 ].sum(-1) + y[ 2::2 ].sum(-1)
                    ) * dlnk / 3
                cor  += q
                kai   = kbi
                kbi  += np.pi / r
                if kbi > kb:
                    break
                if cor and np.abs( q / cor ) < 1.0E-4:
                    break
            return cor 

        if np.ndim( r ):
            if np.ndim( r ) > 1:
                raise TypeError("array dimension should be less than 2")
            return np.asfarray( list( map( _correlation, r ) ) )
        return _correlation( r )

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

    def lagrangianR(self, m: Any) -> Any:
        """
        Lagrangian radius (in Mpc/h) corresponding to a mass (in Msun/h).
        """
        m    = np.asfarray( m )                 # Msun/h
        rho0 = self.Om0 * const.RHO_CRIT0_ASTRO # h^2 Msun/Mpc^3
        return np.cbrt( 0.75*m / ( np.pi * rho0 ) )

    def lagrangianM(self, r: Any) -> Any:
        """
        Lagrangian mass (in Msun/h) corresponding to a radius (in Mpc/h)
        """
        r    = np.asfarray( r )                 # Mpc/h
        rho0 = self.Om0 * const.RHO_CRIT0_ASTRO # h^2 Msun/Mpc^3
        return ( 4*np.pi / 3.0 ) * r**3 * rho0

class LambdaCDM( Cosmology ):
    """
    A Lambda-cdm cosmology model.
    """

    def __init__(self, flat: bool, h: float, Om0: float, Ob0: float, ns: float, Ode0: float = None, sigma8: float = None, psmodel: str = 'eisenstein98_zb', Tcmb0: float = 2.725) -> None:
        super().__init__(   
                            flat    = flat, 
                            h       = h, 
                            Om0     = Om0, 
                            Ob0     = Ob0, 
                            ns      = ns, 
                            Ode0    = Ode0, 
                            sigma8  = sigma8,
                            Nnu     = 0.0, 
                            Onu0    = 0.0, 
                            psmodel = psmodel, 
                            Tcmb0   = Tcmb0,
                        )

class FlatLambdaCDM( LambdaCDM ):
    """
    A flat Lambda-cdm cosmology model.
    """

    def __init__(self, h: float, Om0: float, Ob0: float, ns: float, sigma8: float = None, psmodel: str = 'eisenstein98_zb', Tcmb0: float = 2.725) -> None:
        super().__init__(
                            flat    = True, 
                            h       = h, 
                            Om0     = Om0, 
                            Ob0     = Ob0, 
                            ns      = ns, 
                            sigma8  = sigma8, 
                            psmodel = psmodel, 
                            Tcmb0   = Tcmb0,
                        )




