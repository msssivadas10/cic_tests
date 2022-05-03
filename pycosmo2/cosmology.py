import warnings
import numpy as np
import pycosmo2.integrate as integrate
import pycosmo2.constants as const
from typing import Any, Callable, Union 

class CosmologyError( Exception ):
    r"""
    Base class of exceptions used by cosmology objects.
    """
    ...

class CosmologySettings:
    r"""
    Settings for a cosmology object.
    """
    __slots__ = 'ka', 'kb', 'za', 'zb', 'integrator'

    def __init__(self) -> None:
        # limits for k-space integration:
        self.ka, self.kb = 1E-08, 1E+08

        # limits for z-space integration:
        self.za, self.zb = 1E-08, 1E+08

        # default integrator to use
        self.integrator  = integrate.SimpsonIntegrator(adaptive = True)

    def setkspace(self, a: float = None, b: float = None) -> None:
        """
        Set the limits for k-space integration.
        """
        a = self.ka if a is None else a
        b = self.kb if b is None else b

        if b <= a:
            raise ValueError( "lower limit must be less than the upper limit" )

        # if a is too large or b is too small, put a warning
        if a > 1.0E-04:
            warnings.warn( "for an accurate result, lower limit may be less than 10^-4" )
        elif b < 1.0E+04:
            warnings.warn( "for an accurate result, upper limit may be larger than 10^4" )

        self.ka, self.kb = a, b

    def setzspace(self, a: float = None, b: float = None) -> None:
        """
        Set the limits for z-space integration.
        """
        a = self.za if a is None else a
        b = self.zb if b is None else b

        if b <= a:
            raise ValueError( "lower limit must be less than the upper limit" )

        # if a is too large or b is too small, put a warning
        if a > 1.0E-04:
            warnings.warn( "for an accurate result, lower limit may be less than 10^-4" )
        elif b < 1.0E+04:
            warnings.warn( "for an accurate result, upper limit may be larger than 10^4" )

        self.za, self.zb = a, b

    def setIntegrator(self, integrator: integrate.Integrator) -> None:
        r"""
        Set the integrator to use.
        """
        if isinstance( integrator, integrate.Integrator ):
            self.integrator = integrator
        raise TypeError( "argument must be an 'Integrator' object" )

class Cosmology:
    r"""
    Object to store a cosmology model.
    """
    __slots__ = (
                    '_Om0', '_Ob0', '_Oc0', '_Onu0', '_Ode0', '_Ok0', '_Nnu', '_Mnu', 
                    '_h', '_sigma8', '_ns', '_flat',  '_Tcmb0', '_w0', '_wa', 'settings'
                )

    def __init__(self, flat: bool = True, h: float = None, Om0: float = None, Ob0: float = None, Ode0: float = None, Onu0: float = 0, Nnu: float = None, sigma8: float = None, ns: float = None, Tcmb0: float = 2.725, w0: float = -1.0, wa: float = 0.0) -> None:

        # settings:
        self.settings = CosmologySettings()
        
        # hubble parameter, h:
        if h is None:
            raise CosmologyError( "required parameter: 'h'" )
        elif not ( h > 0.0 ):
            raise CosmologyError( "hubble parameter must be positive" )
        self._h = h

        # rms matter fluctuation, sigma_8:
        if sigma8 is None:
            raise CosmologyError( "required parameter: 'sigma8'" )
        elif not ( sigma8 > 0.0 ):
            raise CosmologyError( "sigma_8 parameter must be positive" )
        self._sigma8 = sigma8

        # power spectrum index:
        if ns is None:
            raise CosmologyError( "required parameter: 'ns'" )
        self._ns = ns

        # cmb temperature:
        if not ( Tcmb0 > 0.0 ):
            raise CosmologyError( "cmb temperature cannot be zero or negative" )
        self._Tcmb0 = Tcmb0


        # initialize components in the universe:
        if Om0 is None:
            raise CosmologyError( "required parameter: 'Om0'" )
        elif Ob0 is None:
            raise CosmologyError( "required parameter: 'Ob0'" )
        elif not flat and Ode0 is None:
            raise CosmologyError( "required parameter: 'Ode0'" )         
        
        if Om0 < 0.0:
            raise CosmologyError( "matter density must be positive" )
        
        if Ob0 < 0.0:
            raise CosmologyError( "baryon density must be positive" )
        elif Ob0 > Om0:
            raise CosmologyError( "baryon density cannot exceed total matter density" )
        self._Om0, self._Ob0 = Om0, Ob0

        if Onu0:
            if Onu0 < 0.0:
                raise CosmologyError( "massive neutrino density must be positive" )
            elif Nnu is None:
                raise CosmologyError( "required parameter: 'Nnu'" )
            elif not ( Nnu > 0.0 ):
                raise CosmologyError( "number of nuetrinos must be positive" )

            self._Nnu  = Nnu
            self._Mnu  =  91.5 * Onu0 * h**2 / Nnu 
            self._Onu0 = Onu0
        else:
            self._Onu0, self._Nnu, self._Mnu = 0.0, 0.0, 0.0

        totalMatter = self._Ob0 + self._Onu0
        if totalMatter > Om0:
            raise CosmologyError( "total density of baryon and neutrino cannot exceed matter density" )
        self._Oc0 = Om0 - totalMatter

        if flat:
            if Om0 > 1.0:
                raise CosmologyError( "matter density cannot exceed 1 for flat universe" )
            
            self._Ode0 = 1.0 - Om0
            self._Ok0  = 0.0
        else:
            if Ode0 < 0.0:
                raise CosmologyError( "dark-energy density cannot be negative" )
            
            totalDensity          = Om0 + Ode0
            self._Ode0, self._Ok0 = Ode0, 1.0 - totalDensity

        self._flat = not self._Ok0


        # dark energy parametrizations. w0-wa model of dark-energy is used
        self._w0, self._wa = w0, wa    

    def __repr__(self) -> str:
        string = f"Cosmology(flat={ self.flat }, h={ self.h }, Om0={ self.Om0 }, Ob0={ self.Ob0 }, Ode0={ self.Ode0 }, "
        if self.Onu0:
            string = string + f"Onu0={ self.Onu0 }, "
        string = string + f"sigma8={ self.sigma8 }, ns={ self.ns }, Tcmb0={ self.Tcmb0 }K, "
        string = string + f"w0={ self.w0 }, wa={ self.wa })"
        return string

    @property
    def Om0(self) -> float: 
        r"""
        Present value of the density parameter for matter.
        """
        return self._Om0

    @property
    def Ob0(self) -> float: 
        r"""
        Present value of the density parameter for baryons.
        """
        return self._Ob0

    @property
    def Oc0(self) -> float: 
        r"""
        Present value of the density parameter for cold dark-matter.
        """
        return self._Oc0

    @property
    def Onu0(self) -> float: 
        r"""
        Present value of the density parameter for massive neutrinos.
        """
        return self._Onu0

    @property
    def Ode0(self) -> float: 
        r"""
        Present value of the density parameter fordark-energy.
        """
        return self._Ode0

    @property
    def Ok0(self) -> float: 
        r"""
        Present value of the density parameter for curvature.
        """
        return self._Ok0

    @property
    def h(self) -> float: 
        r"""
        Present value of the Hubble parameter in units of 100 km/sec/Mpc.
        """
        return self._h

    @property
    def H0(self) -> float:
        r"""
        Present value of the Hubble parameter in units of km/sec/Mpc.
        """
        return ( 100.0 * self.h )

    @property
    def Nnu(self) -> float: 
        r"""
        Number of massive neutrino species.
        """
        return self._Nnu

    @property
    def Mnu(self) -> float: 
        r"""
        Total mass of the massive neutrinos.
        """
        return self._Mnu

    @property
    def sigma8(self) -> float:
        r"""
        RMS variance of matter density fluctuations at 8 Mpc/h scale. 
        """ 
        return self._sigma8

    @property 
    def ns(self) -> float:
        r"""
        Index of the early power spectrum.
        """
        return self._ns

    @property
    def Tcmb0(self) -> float:
        r"""
        Present value of the CMB temperature in Kelvin.
        """
        return self._Tcmb0

    @property
    def flat(self) -> bool:
        r"""
        Geometry of the space - whether flat or not.
        """
        return self._flat

    @property
    def w0(self) -> float:
        r"""
        Constant part of the dark-energy equation of state parameter.
        """
        return self._w0

    @property
    def wa(self) -> float:
        r"""
        Coefficient of the time-dependent part of dark-energy equation of state parameter.
        """
        return self._wa

    # densities of the components:

    def Om(self, z: Any) -> Any:
        r"""
        Density parameter for matter at redshift z.
        """
        zp1 = np.asfarray( z ) + 1
        res = self.Om0 * zp1**3
        y   = res + self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        if not self.flat:
            y = y + self.Ok0 * zp1**2
        return res / y

    def Ob(self, z: Any) -> Any:
        r"""
        Density parameter for baryons at redshift z.
        """
        return self.Om( z ) * ( self.Ob0 / self.Om0 )

    def Oc(self, z: Any) -> Any:
        r"""
        Density parameter for cold dark-matter at redshift z.
        """
        return self.Om( z ) * ( self.Oc0 / self.Om0 )

    def Onu(self, z: Any) -> Any:
        r"""
        Density parameter for massive neutrinos at redshift z.
        """
        return self.Om( z ) * ( self.Onu0 / self.Om0 )

    def Ode(self, z: Any) -> Any:
        r"""
        Density parameter for dark-energy at redshift z.
        """
        zp1 = np.asfarray( z ) + 1
        res = self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        y   = self.Om0 * zp1**3 + res
        if not self.flat:
            y = y + self.Ok0 * zp1**2
        return res / y

    def Ok0(self, z: Any) -> Any:
        r"""
        Density parameter for curvature at redshift z.
        """
        if self.flat:
            return np.zeros_like( z, dtype = 'float' )

        zp1 = np.asfarray( z ) + 1
        res = self.Ok0 * zp1**2
        y   = res + self.Om0 * zp1**3 + self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        return res / y

    def wde(self, z: Any, deriv: bool = False) -> Any:
        r"""
        Equation of state parameter for dark-energy at redshift z. The model used here is a 
        linearly varying equation of state with time / scale factor. i.e., 

        .. math ::

            w( z ) = w_0 + w_a \frac{ z }{ z+1 }

        For :math:`w_0 = -1` and :math:`w_a = 0`, this model becomes time independent (i.e., 
        the cosmological constant, :math:`\Lambda`). 
        """
        z = np.asfarray( z )
        if deriv:
            return self.wa / ( z + 1 )**2
        return self.w0 + self.wa * z / ( z + 1 )

    def criticalDensity(self, z: Any) -> Any:
        r"""
        Critical density if the universe at redshift z. Unit is :math:`h^2 {\rm Msun}/{\rm Mpc}^3`.

        Critical density is the density of a flat universe and is given by 

        .. math ::

            \rho_{\rm crit}(z) = \frac{3H(z)^2}{8\pi G}

        """
        zp1 = np.asfarray( z ) + 1
        res = self.Om0 * zp1**3 + self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        if not self.flat:
            res = res + self.Ok0 * zp1**2
        return res * const.RHO_CRIT0_ASTRO

    def rho_m(self, z: Any) -> Any:
        r"""
        Density of matter at redshift z. Unit is :math:`h^2 {\rm Msun}/{\rm Mpc}^3` 
        """
        return self.criticalDensity(0) * self.Om0 * ( np.asfarray(z) + 1 )**3

    def rho_b(self, z: Any) -> Any:
        r"""
        Density of baryons at redshift z. Unit is :math:`h^2 {\rm Msun}/{\rm Mpc}^3` 
        """
        return self.rho_m( z ) * ( self.Ob0 / self.Om0 )

    def rho_c(self, z: Any) -> Any:
        r"""
        Density of cold dark-matter at redshift z. Unit is :math:`h^2 {\rm Msun}/{\rm Mpc}^3` 
        """
        return self.rho_m( z ) * ( self.Oc0 / self.Om0 )

    def rho_nu(self, z: Any) -> Any:
        r"""
        Density of massive neutrinos at redshift z. Unit is :math:`h^2 {\rm Msun}/{\rm Mpc}^3` 
        """
        return self.rho_m( z ) * ( self.Onu0 / self.Om0 )

    def rho_de(self, z: Any) -> Any:
        r"""
        Density of dark-energy at redshift z. Unit is :math:`h^2 {\rm Msun}/{\rm Mpc}^3` 
        """
        return self.criticalDensity(0) * self.Ode0 * ( np.asfarray(z) + 1 )**( 3 + 3*self.wde( z ) )

    # temperature of cmb:

    def Tcmb(self, z: Any) -> Any:
        r"""
        Temperature of the cosmic microwave background at redshift z. Unit is Kelvin. 
        """
        return self.Tcmb0 * ( np.asfarray( z ) + 1 )

    # hubble parameter:

    def E(self, z: Any) -> Any:
        r"""
        Compute the function

        .. math ::

            E(z) = \sqrt{ \Omega_{\rm } (z+1)^3 + \Omega_{\rm k} (z+1)^2 + \Omega_{\rm de} f(z) }

        where the function :math:`f(z) = (1+z)^{3 + 3w(z)}` tells the evolution of the dark-energy 
        as function of redshift.
        """
        zp1 = np.asfarray( z ) + 1
        res = self.Om0 * zp1**3 + self.Ode0 * zp1**( 3 + 3*self.wde( z ) )
        if not self.flat:
            res = res + self.Ok0 * zp1**2
        return np.sqrt( res )

    def H(self, z: Any) -> Any:
        r"""
        Hubble parameter at redshift z.
        """
        return self.H0 * self.E( z )

    def dlnEdlnzp1(self, z: Any) -> Any:
        r"""
        Logarithmic deivative of the Hubble parameter function.
        """
        zp1 = np.asfarray( z ) + 1

        # add matter contribution to numerator and denominator
        y   = self.Om0 * zp1**3
        y1  = 3.0 * y

        # add curvature contribution (if not flat)
        if not self.flat:
            tmp =  self.Ok0 * zp1**2 
            y   = y  + tmp
            y1  = y1 + 2.0 * tmp

        # add dark-energy contribution
        b   = 3 + 3*self.wde( z ) 
        tmp = self.Ode0 * zp1**b
        y   = y  + tmp 
        y1  = y1 + tmp * ( 
                            b + ( 3*self.wde( z, deriv = True ) ) * zp1 * np.log( zp1 ) 
                         )

        return ( 0.5 * y1 / y )

    def q(self, z: Any) -> Any:
        r"""
        Deceleration parameter at redshift z. In terms of the scale factor,

        .. math ::

            q = -\frac{ a \ddot{a} }{ \dot{a}^2 }

        """
        zp1 = np.asfarray( z )
        return zp1 * self.dlnEdlnzp1( z ) - 1 # TODO: check this eqn.

    # linear growth factors :

    def Dplus(self, z: Any, exact: bool = False, fac: float = None) -> Any:
        r"""
        Compute the linear growth factor.
        """
        z = np.asfarray(z)
        if np.ndim( z ) > 1:
            z = z.flatten()
        if ( z + 1 ) < 0:
            raise CosmologyError( "redshift cannot be less than -1" )

        def _Dplus_exact(z: Any) -> Any:
            raise NotImplementedError()

        def _Dplus_approx(z: Any) -> Any:
            return self.g( z, exact = False ) / ( z + 1 )
        
        if fac is None:
            fac = 1.0 / ( _Dplus_exact( 0 ) if exact else _Dplus_approx( 0 ) )

        return ( _Dplus_exact( z ) if exact else _Dplus_approx( z ) ) * fac

    def g(self, z: Any, exact: bool = False) -> Any:
        r"""
        Linear growth factor, suppressed with respect to a matter dominated universe.
        """
        def _g_exact(z: Any) -> Any:
            raise NotImplementedError()

        def _g_approx(z: Any) -> Any:
            Om, Ode = self.Om( z ), self.Ode( z )
            return 2.5 * Om * ( Om**(4./7.) - Ode + ( 1 + Om/2 ) * ( 1 + Ode/70 ) )**( -1 )
        
        return _g_exact( z ) if exact else _g_approx( z )         

    def f(self, z: Any, exact: bool = False) -> Any:
        r"""
        Logarithmic derivative of linear growth factor with respect to scale factor.
        """
        def _f_exact(z: Any) -> Any:
            raise NotImplementedError()

        def _f_approx(z: Any) -> Any:
            return self.Om( z )**0.6

        return ( _f_exact( z ) if exact else _f_approx( z ) )

    def growthSuppressionFactor(self, q: Any, z: float, nu: bool = False, exact: bool = False, fac: float = None) -> Any:
        r"""
        Compute the suppression of growth of fluctuations in presence of neutrinos.
        """
        q   = np.asfarray( q )    
        
        if self.Onu0 < 1.0E-08:
            return np.ones_like( q )

        fnu = self.Onu0 / self.Om0
        fcb = 1 - fnu
        pcb = 0.25*( 5 - np.sqrt( 1 + 24.0*fcb ) )
        yfs = 17.2 * fnu * ( 1 + 0.488*fnu**(-7.0/6.0) ) * ( self.Nnu*q / fnu )**2
        D1  = self.Dplus( z, exact, fac )    

        x   = ( D1 / ( 1 + yfs ) )**0.7
        if nu:
            return ( fcb**( 0.7 / pcb ) + x )**( pcb / 0.7 ) * D1**( -pcb )
        return ( 1 + x )**( pcb / 0.7 ) * D1**( -pcb )

    # other z functions:

    def zIntegral(self, fz: Callable[[Any], Any], za: Any, zb: Any) -> Any:
        r"""
        Evaluate the integral of a function redshift, given by

        .. math::

            \int_{z_a}^{z_b} f(z) {\rm d}z = \int_{z=z_a}^{z=z_b} (z+1)f(z) {\rm d}\ln(z+1)

        """
        zp1_a  = np.asfarray( za ) + 1
        zp1_b  = np.asfarray( zb ) + 1

        if np.any( zp1_a < 0.0 ) or np.any( zp1_b < 0.0 ):
            raise ValueError("redshift cannot be less than -1")
        
        integrator = self.settings.integrator
        return integrator( lambda z: ( z + 1 ) * fz( z ), za, zb )
        
