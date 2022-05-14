from typing import Any, Callable
import numpy as np
import pycosmo2.utils.constants as const
import pycosmo2.utils.numeric as numeric
import pycosmo2.utils.settings as settings
import pycosmo2.power_spectrum as ps
import pycosmo2.lss.mass_function as mf
import pycosmo2._bases as base

from pycosmo2._bases import CosmologyError


class Cosmology(base.Cosmology):
    
    def __init__(
                    self, h: float, Om0: float, Ob0: float, sigma8: float, ns: float, flat: bool = True, 
                    relspecies: bool = False, Ode0: float = None, Omnu0: float = 0.0, Nmnu: float = None, 
                    Tcmb0: float = 2.725, w0: float = -1.0, wa: float = 0.0, Nnu: float = 3.0,
                    power_spectrum: str = 'eisenstein98_zb', filter: str = 'tophat', 
                    mass_function: str = 'tinker08', linear_bias: str = None,
                ) -> None:

        # check parameters h, sigma8 and ns
        if h <= 0:
            raise CosmologyError("hubble parameter 'h' cannot be negative or zero")
        elif sigma8 <= 0:
            raise CosmologyError("'sigma8' cannot be negative or zero")
        
        self.h, self.sigma8, self.ns = h, sigma8, ns

        # total neutrino number
        if Nnu < 0:
            raise CosmologyError("total neutrino number cannot be negative")
        self.Nnu  = Nnu
            
        # initialise the matter components
        self._init_matter( Om0, Ob0, Omnu0, Nmnu )    

        self.Nrnu = self.Nnu - self.Nmnu 

        # initialise radiation (CMB + relativistic neutrino)
        if Tcmb0 <= 0:
            raise CosmologyError("CMB temperature 'Tcmb0' cannot be negative or zero")
        self.Tcmb0 = Tcmb0
        self.Tnu0  = 0.7137658555036082 * Tcmb0 # temperature of CnuB, (4/11)^(1/3) * Tcmb0 ( TODO: check this )

        self._init_relspecies( relspecies )

        # initialize dark-energy and curvature
        if flat:
            Ode0 = 1 - self.Om0 - self.Or0
            if Ode0 < 0.0:
                raise CosmologyError("dark energy density cannot be negative, adjust matter density")
            self.Ode0, Ok0 = Ode0, 0.0
        else:
            if Ode0 is None:
                raise CosmologyError("Ode0 is a required argument for non-flat cosmology")
            elif Ode0 < 0.0:
                raise CosmologyError("dark-energy density cannot be negative")
            self.Ode0 = Ode0
            
            # calculate the curvature
            Ok0 = 1 - self.Om0 - self.Or0 - self.Ode0
            if abs( Ok0 ) < 1E-15:
                Ok0, flat = 0.0, True
            
        self.flat, self.Ok0 = flat, Ok0

        # dark-energy equation of state parameterization:
        self.w0, self.wa = w0, wa 

        # initialiing power spectrum
        if isinstance(power_spectrum, str):
            if power_spectrum not in ps.models:
                raise ValueError(f"invalid value for power spectrum: '{ power_spectrum }'")
            power_spectrum = ps.models[ power_spectrum ]( self, filter = filter )
        elif not isinstance(power_spectrum, ps.PowerSpectrum):
            raise TypeError("power spectrum must be a 'str' or 'PowerSpectrum' object")
        self.power_spectrum = power_spectrum

        # initialising mass function
        if isinstance(mass_function, str):
            if mass_function not in mf.models:
                raise ValueError(f"invalid value for mass-function: '{ mass_function }'")
            mass_function = mf.models[ mass_function ]( self )
        elif not isinstance(mass_function, mf.HaloMassFunction):
            raise TypeError("mass function must be a 'str' of 'HaloMassFunction' object")
        self.mass_function = mass_function

    def _init_matter(self, Om0: float, Ob0: float, Omnu0: float = 0.0, Nmnu: float = None):
        if Om0 < 0:
            raise CosmologyError("matter density cannot be negative")
        elif Ob0 < 0:
            raise CosmologyError("baryon density cannot be negative")

        if Omnu0:
            if Omnu0 < 0:
                raise CosmologyError("neutrino density must be within 0 and matter density")
            if Nmnu is None or Nmnu <= 0:
                raise CosmologyError("number of massive neutrinos density must be within 0 and matter density")
            elif Nmnu > self.Nnu:
                raise CosmologyError("number of massive nuetrinos cannot exceed total nuetrino number")
        else:
            Omnu0, Nmnu = 0.0, 0.0
        
        if Ob0 + Omnu0 > Om0:
            raise CosmologyError("baryon + massive neutrino density cannot exceed total matter density")
        
        self.Om0, self.Ob0    = Om0, Ob0
        self.Omnu0, self.Nmnu = Omnu0, Nmnu
        self.Oc0              = self.Om0 - self.Ob0 - self.Omnu0  # cold dark matter density

        self.Mmnu = 0.0
        if self.Omnu0:
            self.Mmnu = 91.5 * self.Omnu0 / self.Nmnu * self.h**2 # mass of one massive neutrino

    def _init_relspecies(self, value: bool) -> None:
        if not value:
            self.relspecies = False

            # set all relativistic species densities to zero
            self.Oph0, self.Ornu0, self.Or0 = 0.0, 0.0, 0.0
            
            self.Mrnu = 0.0 # neutrino mass
            return

        self.relspecies = True
        
        # calculate the photon and neutrino density from cmb temperature:
        # using the stephans law to compute the energy density of the photons and converting it into 
        # mass density using E = mc^2. then, 
        # 
        #   Oph0 = 4 * stephan_boltzmann_const * Tcmb0^4 / c^3 / critical_density
        #
        # i.e., Oph0 = const. * Tcmb^4 / h^2, where the constant is `stephan_boltzmann_const / c^3 / critical_density` 
        # with stephans const. in kg/sec^3/K^4, c in m/sec and critical density in h^2 kg/m^2 
        self.Oph0  = 4.4816066598134054e-07 * self.Tcmb0**4 / self.h**2

        # neutrino density is N * (7/8) * (4/11)^(4/3) * photon density
        self.Ornu0 = self.Nrnu * 0.22710731766023898 * self.Oph0

        # mass of relativistic neutrino
        self.Mrnu  = 0.0
        if self.Ornu0:
            self.Mrnu = 91.5 * self.Ornu0 / self.Nrnu * self.h**2

        # total relativistic species density
        self.Or0   = self.Oph0 + self.Ornu0
        return

    def __repr__(self) -> str:
        items = [ f'flat={ self.flat }' , f'h={ self.h }', f'Om0={ self.Om0 }', f'Ob0={ self.Ob0 }', f'Ode0={ self.Ode0 }' ]
        if self.Omnu0:
            items.append( f'Omnu0={ self.Onu0 }' )
        if self.relspecies:
            items.append( f'Or0={ self.Or0 }' )
        items = items + [ f'sigma8={ self.sigma8 }',  f'ns={ self.ns }', f'Tcmb0={ self.Tcmb0 }K', f'w0={ self.w0 }',  f'wa={ self.wa }' ]
        return f'Cosmology({ ", ".join( items ) })'

    def wde(self, z: Any, deriv: bool = False) -> Any:
        z = np.asfarray( z )
        if deriv:
            return self.wa / ( z + 1 )**2
        return self.w0 + self.wa * z / ( z + 1 )

    # hubble parameter:

    def E(self, z: Any, square: bool = False) -> Any:
        zp1 = np.asfarray( z ) + 1
        res = self.Om0 * zp1**3 + self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        if not self.flat:
            res = res + self.Ok0 * zp1**2
        if self.relspecies:
            res = res + self.Or0 * zp1**4
        if square:
            return res
        return np.sqrt( res )

    @property
    def H0(self) -> float:
        return self.h * 100.0

    def H(self, z: Any) -> Any:
        return self.H0 * self.E( z )

    def dlnEdlnzp1(self, z: Any) -> Any:
        zp1 = np.asfarray( z ) + 1

        # add matter contribution to numerator and denominator
        y   = self.Om0 * zp1**3
        y1  = 3*y

        # add curvature contribution (if not flat)
        if not self.flat:
            tmp = self.Ok0 * zp1**2 
            y   = y  + tmp
            y1  = y1 + 2*tmp

        # add radiation contribution
        if self.relspecies:
            tmp = self.Or0 * zp1**4
            y   = y  + tmp
            y1  = y1 + 4*tmp

        # add dark-energy contribution
        b   = 3 + 3*self.wde( z ) 
        tmp = self.Ode0 * zp1**b
        y   = y  + tmp 
        y1  = y1 + tmp * ( 
                            b + ( 3*self.wde( z, deriv = True ) ) * zp1 * np.log( zp1 ) 
                         )

        return ( 0.5 * y1 / y )

    # densities 

    def Om(self, z: Any) -> Any:
        zp1 = np.asfarray( z ) + 1
        res = self.Om0 * zp1**3
        y   = res + self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        if not self.flat:
            y = y + self.Ok0 * zp1**2
        if self.relspecies:
            y = y + self.Or0 * zp1**4
        return res / y

    def Ob(self, z: Any) -> Any:
        return self.Om( z ) * ( self.Ob0 / self.Om0 )

    def Oc(self, z: Any) -> Any:
        return self.Om( z ) * ( self.Oc0 / self.Om0 )

    def Omnu(self, z: Any) -> Any:
        return self.Om( z ) * ( self.Omnu0 / self.Om0 )

    def Ode(self, z: Any) -> Any:
        zp1 = np.asfarray( z ) + 1
        res = self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        y   = self.Om0 * zp1**3 + res
        if not self.flat:
            y = y + self.Ok0 * zp1**2
        if self.relspecies:
            y = y + self.Or0 * zp1**4
        return res / y

    def Ok(self, z: Any) -> Any:
        if self.flat:
            return np.zeros_like( z, dtype = 'float' )

        zp1 = np.asfarray( z ) + 1
        res = self.Ok0 * zp1**2
        y   = res + self.Om0 * zp1**3 + self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        if self.relspecies:
            y = y + self.Or0 * zp1**4
        return res / y

    def Or(self, z: Any) -> Any:
        if not self.relspecies:
            return np.zeros_like( z, 'float' )
        
        zp1 = np.asfarray( z ) + 1
        res = self.Or0 * zp1**4
        y   = res + self.Om0 * zp1**3 + self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        if not self.flat:
            y = y + self.Ok0 * zp1**2
        return res / y

    def Oph(self, z: Any) -> Any:
        if not self.relspecies:
            return self.Or( z )
        return self.Or( z ) * ( self.Oph0 / self.Or0 )

    def Ornu(self, z: Any) -> Any:
        if not self.relspecies:
            return self.Or( z )
        return self.Or( z ) * ( self.Ornu0 / self.Or0 )

    def criticalDensity(self, z: Any) -> Any:
        return const.RHO_CRIT0_ASTRO * self.E( z, square = True )

    def rho_m(self, z: Any) -> Any:
        zp1 = np.asfarray(z) + 1
        return self.criticalDensity(0) * self.Om0 * zp1**3

    def rho_b(self, z: Any) -> Any:
        return self.rho_m( z ) * ( self.Ob0 / self.Om0 )

    def rho_c(self, z: Any) -> Any:
        return self.rho_m( z ) * ( self.Oc0 / self.Om0 )

    def rho_mnu(self, z: Any) -> Any:
        return self.rho_m( z ) * ( self.Omnu0 / self.Om0 )

    def rho_de(self, z: Any) -> Any:
        zp1 = np.asfarray(z) + 1
        return self.criticalDensity(0) * self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
    
    def rho_r(self, z: Any) -> Any:
        if not self.relspecies:
            return np.zeros_like( z, 'float' )
        
        zp1 = np.asfarray( z ) + 1
        return self.criticalDensity(0) * self.Or0 * zp1**4

    def rho_ph(self, z: Any) -> Any:
        if not self.relspecies:
            return self.rho_r( z )
        return self.rho_r( z ) * ( self.Oph0 / self.Or0 )

    def rho_rnu(self, z: Any) -> Any:
        if not self.relspecies:
            return self.rho_r( z )
        return self.rho_r( z ) * ( self.Ornu0 / self.Or0 )

    # temperature of cmb and cnub:

    def Tcmb(self, z: Any) -> Any:
        return self.Tcmb0 * ( np.asfarray( z ) + 1 )

    def Tnu(self, z: Any) -> Any:
        return self.Tnu0 * ( np.asfarray( z ) + 1 )

    # deceleration parameter

    def q(self, z: Any) -> Any:
        zp1 = np.asfarray( z )
        return zp1 * self.dlnEdlnzp1( z ) - 1 # TODO: check this eqn.
        
    # z-integrals: integrals of z-functions 

    def zIntegral(self, f: Callable, za: Any, zb: Any) -> Any:
        za, zb = np.asfarray( za ), np.asfarray( zb )

        if not callable( f ):
            raise TypeError("f must be a callable")
        if np.any( za+1 < 0 ) or np.any( zb+1 < 0 ):
            raise CosmologyError("redshift values must be greater than -1")

        def zfunc(lnzp1: Any) -> Any:
            z = np.exp( lnzp1 ) - 1
            return f( z ) * ( z + 1 )

        return numeric.integrate1( zfunc, np.log( za+1 ), np.log( zb+1 ), subdiv = settings.DEFAULT_SUBDIV  )

    def zIntegral_zp1_over_Ez3(self, za: Any, zb: Any) -> Any:
        def zfunc(z: Any) -> Any:
            return ( z + 1 ) / self.E( z )**3

        return self.zIntegral( zfunc, za, zb )

    def zIntegral_1_over_zp1_Ez(self, za: Any, zb: Any) -> Any:
        def zfunc(z: Any) -> Any:
            return 1.0 / ( self.E( z ) * ( z + 1 ) )
        
        return self.zIntegral( zfunc, za, zb )

    def zIntegral_1_over_Ez(self, za: Any, zb: Any) -> Any:
        def zfunc(z: Any) -> Any:
            return 1.0 / self.E( z )
        
        return self.zIntegral( zfunc, za, zb )

    # time and distances

    def universeAge(self, z: Any) -> Any:
        inf = settings.INF
        t0  = self.zIntegral_1_over_zp1_Ez( z, inf ) * self.hubbleTime( 0 )
        return t0
    
    def lookbackTime(self, z: Any) -> Any:
        return self.universeAge( 0.0 ) - self.universeAge( z )

    def hubbleTime(self, z: Any) -> Any:
        Hz = self.H( z ) * ( 1000.0 / const.MPC * const.YEAR ) # in 1/yr
        return 1.0 / Hz
    
    def comovingDistance(self, z: Any) -> Any:
        fac = const.C_SI / self.H0 / 1000.0 # c/H0 in Mpc
        return self.zIntegral_1_over_Ez( 0.0, z ) * fac

    def comovingCorrdinate(self, z: Any) -> Any:
        x = self.comovingCorrdinate( z )
        if self.Ok0:
            K = np.sqrt( abs( self.Ok0 ) ) * ( self.H0 / const.C_SI * 1000 ) 

            if self.Ok0 < 0.0:
                return np.sin( K*x ) / K # k > 0 : closed/spherical
            return np.sinh( K*x ) / K    # k < 0 : open / hyperbolic
        return x
    
    def angularDiamaterDistance(self, z: Any) -> Any:
        return NotImplemented
    
    def luminocityDistance(self, z: Any) -> Any:
        return NotImplemented

    # horizons

    def hubbleHorizon(self, z: Any) -> Any:
        c = const.C_SI / 1000.0 # speed of light in km/sec
        return c / self.Hz( z ) # Mpc
    
    def eventHorizon(self, z: Any) -> Any:
        return NotImplemented
    
    def particleHorizon(self, z: Any) -> Any:
        return NotImplemented

    # linear growth

    def g(self, z: Any, exact: bool = False) -> Any:
        
        def gzFit(z: Any) -> Any:
            Om, Ode = self.Om( z ), self.Ode( z )
            return 2.5*Om*(
                            Om**(4./7.) - Ode + ( 1 + Om/2 ) * ( 1 + Ode/70 )
                          )**( -1 )

        def gzExact(z: Any) -> Any:
            z, inf = np.asfarray( z ), settings.INF
            if np.ndim( z ):
                z  = z.flatten()
                
            y = self.zIntegral_zp1_over_Ez3( z, inf )
            return 2.5 * self.Om0 * self.E( z ) * y * ( z + 1 )
            
        return gzExact( z ) if exact else gzFit( z )

    def Dplus(self, z: Any, exact: bool = False, fac: float = None):

        def _Dplus(z: Any, exact: bool) -> Any:
            gz = self.g( z, exact )
            return gz / ( z + 1 )

        if fac is None:
            fac = 1.0 / _Dplus( 0, exact )
        return _Dplus( z, exact ) * fac

    def f(self, z: Any, exact: bool = False) -> Any:

        def fzFit(z: Any) -> Any:
            return self.Om( z )**0.55

        def fzExact(z: Any) -> Any:
            return (
                        2.5*self.Om( z ) / self.g( z, exact = True ) 
                            - self.dlnEdlnzp1( z )
                   )
        
        return fzExact( z ) if exact else fzFit( z )
    
    def _DplusFreeStream(self, q: Any, Dz: Any, include_nu: bool = False) -> Any:
        q, Dz = np.asfarray( q ), np.asfarray( Dz )
        if np.ndim( q ):
            q = q.flatten()
        if np.ndim( Dz ):
            Dz = Dz.flatten()[ :, None ]
        
        if not self.Omnu:
            return np.repeat( 
                                Dz, q.shape[0], 
                                axis = 1 if np.ndim( Dz ) else 0 
                            )

        fnu = self.Omnu0 / self.Om0 # fraction of massive neutrino
        fcb = 1 - fnu
        pcb = 0.25*( 5 - np.sqrt( 1 + 24.0*fcb ) )
        yfs = 17.2 * fnu * ( 1 + 0.488*fnu**(-7./6.) ) * ( self.Nmnu*q / fnu )**2
        
        x = ( Dz / ( 1 + yfs ) )**0.7     
        y = fcb**( 0.7 / pcb ) if include_nu else 1.0
        return ( y + x )**( pcb / 0.7 ) * Dz**( 1 - pcb )

    def DplusFreeStream(self, q: Any, z: Any, include_nu: bool = False, exact: bool = False, fac: float = None) -> Any:
        Dz = self.Dplus( z, exact, fac ) # growth without free streaming
        return self._DplusFreeStream( q, Dz, include_nu )

    # power spectrum

    def linearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        return self.power_spectrum.linearPowerSpectrum( k, z, dim )

    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        return self.power_spectrum.nonlinearPowerSpectrum( k, z, dim )

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, linear: bool = True) -> Any:
        return self.power_spectrum.matterPowerSpectrum( k, z, dim, linear )

    def matterCorreleation(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        return self.power_spectrum.matterCorrelation( r, z, linear )

    def variance(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        return self.power_spectrum.variance(r, z, linear)
    
    def radius(self, sigma: Any, z: float = 0, linear: bool = True) -> Any:
        return self.power_spectrum.radius( sigma, z, linear )
    
    def dlnsdlnr(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        return self.power_spectrum.dlnsdlnr( r, z, linear )
    
    def dlnsdlnm(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        return self.dlnsdlnr( r, z, linear ) / 3.0

    def d2lnsdlnr2(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        return self.power_spectrum.d2lnsdlnr2( r, z, linear )

    def effectiveIndex(self, k: Any, z: float = 0, linear: bool = True) -> Any:
        return self.power_spectrum.effectiveIndex( k, z, linear )

    def nonlineark(self, k: Any, z: float = 0) -> Any:
        return self.power_spectrum.nonlineark( k, z )
        
    # halo mass function and related calculations

    def lagrangianR(self, m: Any) -> Any:
        m = np.asfarray( m ) # Msun/h
        return np.cbrt( 0.75*m / ( np.pi * self.rho_m( 0 ) ) )

    def lagrangianM(self, r: Any) -> Any:
        r = np.asfarray( r ) # Mpc/h
        return ( 4*np.pi / 3.0 ) * r**3 * self.rho_m( 0 )

    def collapseOverdensity(self, z: Any) -> float:
        return const.DELTA_C * np.ones_like( z, 'float' )

    def massFunction(self, m: Any, z: float = 0, overdensity: Any = None, out: str = 'dndlnm') -> Any:
        return self.mass_function.massFunction( m, z, overdensity, out )