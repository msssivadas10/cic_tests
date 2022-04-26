
import numpy as np
import pycosmo.core.filters as filters
import pycosmo.power.linear_power as lp
import pycosmo.power.nonlinear_power as nlp
import pycosmo.lss.massfunction as hmf
from pycosmo.core.components import ( 
                                        Universe, Matter, Baryon, ColdDarkMatter, MassiveNeutrino, 
                                        DarkEnergy, LinearWDarkEnergy, Curvature 
                                    )
import pycosmo.utils.constants as const
from pycosmo.utils.integrate import simps
from typing import Any, Callable, Type


class settings:
    """
    Settings for cosmology calculations.
    """
    # end-points for integration on half-real line [a, b], a --> 0 and b --> inf
    a, b = 1E-08, 1E+08

    # number of points used for integration
    pts  = 10001

    # step-size used for numerical differentiation
    step = 0.01

    # whether to use exact (integral) form for growth factor/rate
    exactGrowth = False

    # filter to use : tophat / gauss / sharpk
    filter = 'tophat'

    # non-linear power spectrum model
    nlpower = 'pd'


class Cosmology:
    """
    Represents a basic cosmology model.
    """
    __slots__ = (
                    'name', 'univ', 'h', 'sigma8', 'ns', 'Tcmb0', 'A', 'psmodel', 'ptable', 'flat',
                    'demodel', 'hmfmodel', 'biasmodel',
                )

    def __init__(self, flat: bool = True, h: float = None, Om0: float = None, Ob0: float = None, Ode0: float = None, Onu0: float = 0.0, Nnu: float = None, sigma8: float = None, ns: float = 1.0, w0: float = -1.0, wa: float = 0.0, Tcmb0: float = 2.725, power_spectrum: str = 'eisenstein98_zb', filter: str = 'tophat', hmfmodel: str = None, bias_model: str = None) -> None:
        if h is None:
            raise ValueError("required argument: 'h'")
        elif h <= 0.0:
            raise ValueError("'h' must be strictly positive")
        self.h = h

        if sigma8 is None:
            raise ValueError("required argument: 'sigma8'")
        elif sigma8 <= 0.0:
            raise ValueError("'sigma8' must be strictly positive")
        self.sigma8 = sigma8

        if ns is None:
            raise ValueError("required argument: 'ns'")
        self.ns = ns

        if Tcmb0 <= 0.0:
            raise ValueError("temperature'Tcmb0' must be strictly positive")
        self.Tcmb0 = Tcmb0

        # Create the universe model:
        self._createUniverse( flat, Om0, Ob0, Ode0, [ w0, wa ], Onu0, Nnu )

        # set filter type:
        if filter not in [ 'tophat', 'gauss', 'sharpk' ]:
            raise ValueError(f"invalid filter key: '{ filter }")

        # set the power spectrum model:
        self.setPowerSpectrum( power_spectrum )

        # set the mass-function model:
        self.sethmf( hmfmodel )

        # set linear bias model: 
        self.setbias( bias_model )  

    def _createUniverse(self, flat: bool, Om0: float, Ob0: float, Ode0: float, wde: Any, Onu0: float, Nnu: float) -> None:
        """
        Create the universe model.
        """
        univ = Universe() # root component: universe

        if Om0 is None:
            raise ValueError("required argument: 'Om0'")
        elif Om0 <= 0.0:
            raise ValueError("'Om0' must be strictly positive")
        matter = Matter( Om0 ) # matter component

        if Ob0 is None:
            raise ValueError("required argument: 'h'")
        elif Ob0 <= 0.0:
            raise ValueError("'h' must be strictly positive")
        matter.addchild( Baryon( Ob0 ) ) # add baryons to matter

        neutrino = MassiveNeutrino( 0.0 )
        if Onu0:
            if Onu0 < 0.0:
                raise ValueError("'Onu0' must be strictly positive")
            if Nnu is None:
                raise ValueError("required argument: 'Nnu'")
            neutrino = MassiveNeutrino( Onu0, Nnu, self.h )
        matter.addchild( neutrino ) # add massive neutrinos to matter 

        matter.addchild( ColdDarkMatter( matter.remOmega0 ) ) # fill the rest with cold dark-matter

        univ.addchild( matter, checkOmega = False ) # add matter to universe

        w0, wa = wde
        de     = LinearWDarkEnergy( univ.remOmega0, w0, wa ) 
        if not flat:
            if Ode0 is None:
                raise ValueError("required argument: 'Ode0'")
            de = LinearWDarkEnergy( Ode0, w0, wa )
        univ.addchild( de, checkOmega = False ) # add dark energy to universe

        univ.getCurvature( flat ) # get the geometry of universe

        self.univ    = univ
        self.flat    = flat or univ.flat
        self.demodel = de.name

    def setPowerSpectrum(self, value: Any) -> None:
        """
        Set a power spectrum model.
        """
        self.ptable = None

        if isinstance( value, str ):
            # value is a valid string name of a (linear) model

            if not lp.available( value ):
                raise ValueError(f"power spectrum model not available: '{ value }'")
            self.psmodel = value
        else:
            # value is a tabulated power spectrum

            if np.ndim( value ) != 2:
                raise TypeError("power spectrum table must be a 2D array")
            value = np.asfarray( value )
            if value.shape[1] != 2:
                raise TypeError("power spectrum table should only have 2 coulmns: 'lnk' and 'lnPk'")

            from scipy.interpolate import CubicSpline

            lnk, lnPk   = value.T
            self.ptable = CubicSpline( lnk, lnPk )
            self.psmodel = 'rawdata'   

        return self.normalize()

    def sethmf(self, value: str) -> None:
        """
        Set the halo mass-function model to use.
        """
        if not hmf.available( value ):
            raise ValueError(f"mass-function model not available: '{ value }'")
        self.hmfmodel = value

    def setbias(self, value: str) -> None:
        """
        Set the linear bias model to use.
        """
        ...

    # Components:

    @property
    def matter(self) -> Matter: 
        """
        Matter content in the universe.
        """
        return self.univ.child( 'matter' )

    @property
    def baryon(self) -> Baryon: 
        """
        Baryon content in the universe.
        """
        return self.matter.child( 'baryon' )

    @property
    def massive_nu(self) -> MassiveNeutrino: 
        """
        Maasive neutrino content in the universe.
        """
        return self.matter.child( 'massive_nu' )

    @property
    def cdm(self) -> ColdDarkMatter: 
        """
        Cold dark-matter content in the universe.
        """
        return self.univ.child( 'matter' ).child( 'cdm' )

    @property
    def dark_energy(self) -> DarkEnergy: 
        """
        Dark-energy content in the universe.
        """
        return self.univ.child( self.demodel )

    @property
    def curvature(self) -> Curvature: 
        """
        Curvature in the universe.
        """
        return self.univ.child( 'curvature' )

    # Component density:

    @property
    def Om0(self) -> float: 
        return self.matter.Om0

    @property
    def Ob0(self) -> float: 
        return self.baryon.Ob0

    @property
    def Onu0(self) -> float: 
        return self.massive_nu.Onu0

    @property
    def Nnu(self) -> float: 
        return self.massive_nu.Nnu

    @property
    def mnu(self) -> float: 
        return self.massive_nu.mnu

    @property
    def Oc0(self) -> float: 
        return self.cdm.Oc0

    @property
    def Ode0(self) -> float: 
        return self.dark_energy.Ode0
    
    @property
    def Ok0(self) -> float: 
        return self.curvature.Ok0

    @property
    def H0(self) -> float:
        return self.h * 100.0

    # Redshift functions:

    def Ez(self, z: Any) -> Any:
        """
        Evolution of hubble parameter.
        """
        y   = self.matter.zDensity( z ) + self.dark_energy.zDensity( z )
        if not self.flat:
            y += self.curvature( z )
        return np.sqrt( y )

    def Hz(self, z: Any) -> Any:
        """
        Evolution of hubble parameter.
        """
        return self.H0 * self.Ez( z )

    def lnzp1Integral(self, zfunc: Callable[[Any], Any], za: Any, zb: Any) -> Any:
        """
        Evaluate the integral of a function over redshift variable.
        """
        zp1_a  = np.asfarray( za ) + 1
        zp1_b  = np.asfarray( zb ) + 1

        if not np.any( zp1_a ) or not np.any( zp1_b ):
            raise ValueError("redshift cannot be less than -1")

        pts    = settings.pts
        if pts < 3:
            raise ValueError("'pts' must be greater than 2")
        elif not ( pts % 2 ):
            pts = pts + 1
        
        z, dlnzp1 = np.linspace( np.log( zp1_a ), np.log( zp1_b ), pts, retstep = True )
        z         = np.exp( z ) - 1
        y         = zfunc( z )
        return simps( y, dlnzp1 )

    def zIntegral_zp1_over_Ez3(self, za: Any, zb: Any) -> Any:
        """
        Evaluate the integral over redshift.
        """
        def zfunc(z: Any) -> Any:
            return ( z + 1 )**2 / self.Ez( z )**3

        return self.lnzp1Integral( zfunc, za, zb )

    def zIntegral_1_over_zp1_Ez(self, za: Any, zb: Any) -> Any:
        """
        Evaluate the integral over redshift.
        """
        def zfunc(z: Any) -> Any:
            return 1.0 / self.Ez( z )
        
        return self.lnzp1Integral( zfunc, za, zb )

    def zIntegral_1_over_Ez(self, za: Any, zb: Any) -> Any:
        """
        Evaluate the integral over redshift.
        """
        def zfunc(z: Any) -> Any:
            return ( z + 1 ) / self.Ez( z )
        
        return self.lnzp1Integral( zfunc, za, zb )

    # Time and distances:

    def universeAge(self, z: Any) -> Any:
        """
        Compute the age of universe at redshift z.
        """
        t0 = self.zIntegral_1_over_zp1_Ez( z, settings.b ) / self.H0 * ( 1.0E-03 * const.MPC / const.YEAR ) # year 
        return np.log( t0 )

    def lookbackTime(self, z: Any) -> Any:
        """
        Compute the lookback time at redshift z.
        """
        return self.universeAge( 0.0 ) - self.universeAge( z )

    def comovingDistance(self, z: Any) -> Any:
        """
        Comoving distance in Mpc corresponding to a redshift.
        """
        x = self.zIntegral_1_over_Ez( 0.0, z ) * ( 1.0E-03 * const.C_SI / self.H0 ) # Mpc
        return x

    def comovingCorrdinate(self, z: Any) -> Any:
        """
        Comoving coordinate corresponding to a redshift.
        """
        x = self.comovingCorrdinate( z )
        if self.Ok0:
            K = np.sqrt( abs( self.Ok0 ) ) * ( self.H0 / const.C_SI * 1.0E+03 ) # Mpc^-1
            if self.Ok0 < 0.0:
                # k > 0 : closed / spherical universe
                return np.sin( K*x ) / K

            # k < 0 : open / hyperbolic universe
            return np.sinh( K*x ) / K
        return x
    
    def angularDiamaterDistance(self, z: Any) -> Any:
        """
        Angular diameter distance corresponding to a redshift.
        """
        return NotImplemented
    
    def luminocityDistance(self, z: Any) -> Any:
        """
        Luminocity distance corresponding to a redshift.
        """
        return NotImplemented

    def hubbleHorizon(self, z: Any) -> Any:
        """
        Hubble horizon in Mpc at redshift z.
        """
        return 1.0E-03 * const.C_SI / self.Hz( z ) # Mpc
    
    def eventHorizon(self, z: Any) -> Any:
        """
        Event horizon at redshift z.
        """
        return NotImplemented
    
    def particleHorizon(self, z: Any) -> Any:
        """
        Particle horizon at redshift z.
        """
        return NotImplemented

    # CMB temperature:

    def Tcmb(self, z: Any) -> Any:
        """
        Temperature of the cosmic microwave background.  
        """
        return self.Tcmb0 * ( np.asfarray( z ) + 1 )

    # Densities:

    def criticalDensity(self, z: Any) -> Any:
        """
        Critical density of the universe in kg/m^3.
        """
        return const.RHO_CRIT0_ASTRO * self.Ez( z )

    def rho_m(self, z: Any) -> Any:
        """
        Evolution of matter density.
        """
        return self.matter.zDensity( z ) * const.RHO_CRIT0_ASTRO

    def rho_de(self, z: Any) -> Any:
        """
        Evolution of dark-energy density.
        """
        return self.dark_energy.zDensity( z ) * const.RHO_CRIT0_ASTRO

    def Om(self, z: Any) -> Any:
        """
        Evolution of matter density.
        """
        y1 = self.matter.zDensity( z )
        y2 = y1 + self.dark_energy.zDensity( z )
        if not self.flat:
            y2 += self.curvature.zDensity( z )
        return ( y1 / y2 )

    def Ode(self, z: Any) -> Any:
        """
        Evolution of dark-energy density.
        """
        y1 = self.dark_energy.zDensity( z )
        y2 = self.matter.zDensity( z ) + y1
        if not self.flat:
            y2 += self.curvature.zDensity( z )
        return ( y1 / y2 )

    def wde(self, z: Any) -> Any:
        """
        Evolution of dark-energy equation of state.
        """
        return self.dark_energy.wz( z )

    # Growth factor:

    def gz(self, z: Any) -> Any:
        """
        Fitting function for linear growth factor.
        """
        Om, Ode = self.matter.zDensity( z ), self.dark_energy.zDensity( z )
        y       = Om + Ode
        if not self.flat:
            y += self.curvature.zDensity( z )

        Om, Ode = Om / y, Ode / y
        return 2.5*Om*( 
                            Om**(4./7.) - Ode + ( 1 + Om/2 ) * ( 1 + Ode/70 )
                      )**( -1 )

    def Dz(self, z: Any, fac: float = None) -> Any:
        """
        Linear growth factor.
        """
        def _Dz(z: Any, exact: bool) -> Any:
            if not exact:
                return self.gz( z ) / ( z+1 )

            y = self.zIntegral_zp1_over_Ez3( z, settings.b )
            w = self.matter.zDensity( z ) + self.dark_energy.zDensity( z )
            if not self.flat:
                w += self.curvature.zDensity( z )
                
            return 2.5 * self.Om0 * y * np.sqrt( w )
        
        z = np.asfarray( z )
        if np.ndim( z ) > 1:
            raise TypeError("array dimension should be less than 2")
        if np.any( z < -1 ):
            raise ValueError("redshift 'z' cannot be less than -1")

        exact = settings.exactGrowth

        if fac is None:
            fac = 1.0 / _Dz( 0.0, exact )
        return _Dz( z, exact ) * fac 

    def fz(self, z: Any) -> Any:
        """
        Linear growth rate.
        """
        exact = settings.exactGrowth
            
        if not exact:
            return self.Om( z )**0.6

        z      = np.asfarray( z )
        y1, y2 = self.matter.zDensity( z ), None
        y      = y1 + self.dark_energy.zDensity( z )
        if not self.flat:
            y2 = self.curvature.zDensity( z )
            y += y2
        
        y1 = y1 * ( 2.5 / ( ( z+1 )*self.Dz( z, 1.0, exact ) ) - 1.5 )
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

    # Lagrangian radius / mass:

    def lagrangianR(self, m: Any) -> Any:
        """
        Lagrangian radius (in Mpc/h) corresponding to a mass (in Msun/h).
        """
        m = np.asfarray( m ) # Msun/h
        return np.cbrt( 0.75*m / ( np.pi * self.rho_m( 0 ) ) )

    def lagrangianM(self, r: Any) -> Any:
        """
        Lagrangian mass (in Msun/h) corresponding to a radius (in Mpc/h)
        """
        r = np.asfarray( r ) # Mpc/h
        return ( 4*np.pi / 3.0 ) * r**3 * self.rho_m( 0 )

    # Power spectrum calculations:

    def filter(self, x: Any, j: int = 0) -> Any:
        """
        Filter function.
        """
        filt = settings.filter

        x = np.asfarray( x )
        if filt == 'tophat':
            return filters.tophat( x, j )
        elif filt == 'gauss':
            return filters.gauss( x, j )
        elif filt == 'sharpk':
            return filters.sharpk( x, j )
        raise ValueError(f"invalid filter: '{ filt }'")

    def transfer(self, k: Any, z: float = 0) -> Any:
        """
        Transfer function.
        """
        model = self.psmodel

        if model == 'raw':
            k = np.asfarray( k )
            return np.sqrt( 
                            np.exp( self.pspline( np.log( k ) ) ) / k**self.ns 
                          )
            
        return lp.transfer( self, k, z, model )

    def linearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        """
        Compute linear matter power spectrum. 
        """
        model = self.psmodel
        if model == 'raw':
            return self.A * np.exp( self.pspline( np.log( k ) ) )
        return lp.linearPowerSpectrum( self, k, z, dim, model )
    
    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        """
        Compute the non-linear power spectrum.
        """
        model = settings.nlpower
        return nlp.nonlinearPowerSpectrum( self, k, z, dim, model )

    def knl(self, k: Any, z: float = 0.0) -> Any:
        """
        Compute the non-linear wavenumber.
        """
        k   = np.asfarray( k )
        dnl = self.nonlinearPowerSpectrum( k, z, dim = False )
        return k * np.cbrt( 1.0 + dnl )

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, lin: bool = True) -> Any:
        """
        Compute matter power spectrum. 
        """
        if lin:
            return self.linearPowerSpectrum( k, z, dim )
        return self.nonlinearPowerSpectrum( k, z, dim )

    def variance(self, r: Any, z: float = 0, lin: bool = True) -> Any:
        """ 
        Compute the linear matter variance.
        """
        if np.ndim( r ) > 1:
            raise TypeError("array dimension should be less than 2")

        ka, kb, pts = settings.a, settings.b, settings.pts
        
        if ka <= 0.0 or kb <= 0.0:
            raise ValueError("'ka' and 'kb' must be positive") 
        elif kb <= ka:
            raise ValueError("'ka' should be less than 'kb'")

        if pts < 3:
                raise ValueError("'pts' must be greater than 2")
        elif not ( pts % 2 ):
            pts = pts + 1

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr  = np.outer(r, k)
        var = simps( 
                        self.matterPowerSpectrum( k, z, dim = False, lin = lin ) * self.filter( kr )**2 , 
                        dlnk
                   )

        return var if np.ndim(r) else var[0] 

    def radius(self, sigma: Any, z: float = 0.0, lin: bool = True) -> Any:
        """
        Invert the variance equation to get the value of radius.
        """        
        from scipy.optimize import toms748

        def f(r: Any, z: float, lin: bool, var: Any) -> Any:
            return self.variance( r, z, lin ) - var

        def _radius(var: Any) -> Any:
            return toms748( f, 1.0E-05, 1.0E+05, args = ( z, lin, var ) )

        return np.asfarray( 
                            list( 
                                    map( _radius, np.asfarray( sigma )**2 ) 
                                ) 
                          )

    def dlnsdlnr(self, r: Any, z: float = 0, lin: bool = True) -> Any:
        """ 
        Compute the logarithmic derivative of linear matter variance.
        """
        if np.ndim( r ) > 1:
            raise TypeError("array dimension should be less than 2")

        ka, kb, pts = settings.a, settings.b, settings.pts
        
        if ka <= 0.0 or kb <= 0.0:
            raise ValueError("'ka' and 'kb' must be positive") 
        elif kb <= ka:
            raise ValueError("'ka' should be less than 'kb'")
            
        if pts < 3:
                raise ValueError("'pts' must be greater than 2")
        elif not ( pts % 2 ):
            pts = pts + 1

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr  = np.outer(r, k)

        y0  = self.matterPowerSpectrum( k, z, dim = False, lin = lin ) * self.filter( kr )
        y1  = y0 * k * self.filter( kr, j = 1 )
        y0  = y0 * self.filter( kr )

        y0  = simps( y0, dlnk )
        y1  = simps( y1, dlnk )
        
        out = y1 / y0 * r
        return ( out if np.ndim(r) else out[0] ) 

    def d2lnsdlnr2(self, r: Any, z: float = 0, lin: bool = True) -> Any:
        """ 
        Compute the logarithmic second derivative of linear matter variance.
        """
        if np.ndim( r ) > 1:
            raise TypeError("array dimension should be less than 2")

        ka, kb, pts = settings.a, settings.b, settings.pts
        
        if ka <= 0.0 or kb <= 0.0:
            raise ValueError("'ka' and 'kb' must be positive") 
        elif kb <= ka:
            raise ValueError("'ka' should be less than 'kb'")
            
        if pts < 3:
                raise ValueError("'pts' must be greater than 2")
        elif not ( pts % 2 ):
            pts = pts + 1

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr = np.outer(r, k)

        y0 = self.matterPowerSpectrum( k, z, dim = False, lin = lin )
        y1 = self.filter( kr ) * k
        y2 = y1 * self.filter( kr, j = 2 ) * k + self.filter( kr, j = 1 )**2 * k**2
        y2 = y0 * y2                        # delta^2 * ( w * d2w + dw^2 ) * k^2 [2]
        y1 = y0 * self.filter( kr, j = 1 )  # delta^2 * w * dw * k [2]
        y0 = y0 * self.filter( kr )         # delta^2 * w^2 [2]

        y0 = simps( y0, dlnk ) # sigma^2
        y1 = simps( y1, dlnk ) # d sigma^2 
        y2 = simps( y2, dlnk ) # d2 sigma^2
        
        out = ( y0 * y2 - 2 * y1**2 ) / y0**2 * r**2
        return ( out if np.ndim(r) else out[0] )

    def dlnsdlnm(self, r: Any, z: float = 0, lin: bool = True) -> Any:
        """ 
        Compute the logarithmic derivative of linear matter variance.
        """
        return self.dlnsdlnr( r, z, lin ) / 3.0

    def correlation(self, r: Any, z: float = 0, lin: bool = True) -> Any:
        """
        Linear matter correlation function.
        """
        if np.ndim( r ) > 1:
            raise TypeError("array dimension should be less than 2")

        ka, kb, pts = settings.a, settings.b, settings.pts
        
        if ka <= 0.0 or kb <= 0.0:
            raise ValueError("'ka' and 'kb' must be positive") 
        elif kb <= ka:
            raise ValueError("'ka' should be less than 'kb'")
            
        if pts < 3:
                raise ValueError("'pts' must be greater than 2")
        elif not ( pts % 2 ):
            pts = pts + 1

        def sinc(x: Any) -> Any:
            return np.sinc( x / np.pi )

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr  = np.outer(r, k)
        var = simps(
                        self.matterPowerSpectrum( k, z, dim = False, lin = lin ) * sinc( kr ),
                        dlnk
                   )

        return var if np.ndim(r) else var[0] 
    
    def powerNorm(self, sigma8: float, lin: bool = True) -> float:
        """
        Get the power spectrum normalization without setting it.
        """ 
        return sigma8**2 / self.variance( 8.0, 0.0, lin )

    def normalize(self, sigma8: float = None, lin: bool = True) -> None:
        """
        Normalize the power spectrum.
        """
        if sigma8 is None:
            sigma8 = self.sigma8

        self.A      = 1.0
        self.sigma8 = sigma8
        self.A      = self.powerNorm( sigma8, lin )
    
    def effectiveIndex(self, k: Any, z: float = 0.0, lin: bool = True) -> Any:
        """
        Compute the slope of the power spectrum.
        """
        h = settings.step
        if h <= 0.0:
            raise ValueError("'h' cannot be negative or zero")

        k    = np.asfarray( k )
        dlnp = (
                    -np.log( 
                                self.matterPowerSpectrum( (1+2*h)*k, z, dim = True, lin = lin ) 
                           )
                        + 8.0*np.log( 
                                        self.matterPowerSpectrum( (1+h)*k, z, dim = True, lin = lin ) 
                                    )
                        - 8.0*np.log( 
                                        self.matterPowerSpectrum( (1-h)*k, z, dim = True, lin = lin ) 
                                    )
                        + np.log( 
                                    self.matterPowerSpectrum( (1-2*h)*k, z, dim = True, lin = lin ) 
                                )
                )
        dlnk = 12.0 * ( np.log( (1+h)*k ) - np.log( (1-h)*k ) )
        
        return dlnp / dlnk

    # Halo Mass-function:

    def haloMassFunction(self, m: Any, z: float = 0.0, mdef: str = None, out: str = 'dndlnm') -> Any:
        """
        Compute the halo mass-function.
        """
        return hmf.massFunction( self, m, z, model = self.hmfmodel, mdef = mdef, out = out )

