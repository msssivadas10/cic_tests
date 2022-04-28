from abc import ABC, abstractmethod
from typing import Any, Type, Union
import numpy as np

####################################################################################################

class CosmologyError( Exception ):
    r"""
    Base class of exceptions used by cosmology objects.
    """
    ...

class CosmologyBase( ABC ):
    r"""
    Object to store a cosmology model.
    """
    __slots__ = (
                    '_Om0', '_Ob0', '_Oc0', '_Onu0', '_Ode0', '_Ok0', '_Nnu', '_Mnu', 
                    '_h', '_sigma8', '_ns', '_flat',  '_Tcmb0', '_w0', '_wa'
                )

    def __init__(self, flat: bool = True, h: float = None, Om0: float = None, Ob0: float = None, Ode0: float = None, Onu0: float = 0, Nnu: float = None, sigma8: float = None, ns: float = None, Tcmb0: float = 2.725, w0: float = -1.0, wa: float = 0.0) -> None:
        
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
        return res * 2.77536627E+11

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

    # abstract methods;

    @abstractmethod
    def Dz(self, z: Any) -> Any:
        r"""
        Linear growth factor at redshift z.
        """
        ...

    @abstractmethod
    def fz(self, z: Any) -> Any:
        r"""
        Logarithmic rate of linear growth at redshift z.
        """
        ...

    @abstractmethod
    def matterPowerSpectrum(self, k: Any, *args, **kwargs) -> Any:
        r"""
        Compute the matter power spectrum as function of wave number k (unit: h/Mpc).
        """
        ...

    @abstractmethod
    def haloMassFunction(self, m: Any, *args, **kwargs) -> Any:
        r"""
        Compute the halo mass-function, given mass m (unit: Msun/h).
        """
        ...

####################################################################################################

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

    def __init__(self, cm: CosmologyBase, transfer_function: str, nonlinear_model: str, filter: str) -> None:
        self.cosmology = cm

        self.transfer_function = transfer_function
        self.nonlinear_model   = nonlinear_model
        self.filter            = filter

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

    
####################################################################################################

class MassFunctionError( Exception ):
    r"""
    Base class of exceptions used by halo mass-function objects.
    """
    ...

class MassFunction( ABC ):
    r"""
    A class representing a halo mass-function model.
    """

    __slots__ = 'mass_function', 'depend_z', 'depend_cosmology', 'mdefs', 'cosmology'

    def __init__(self, model: str, depend_z: bool, depend_cosmo: bool, mdefs: list) -> None:
        self.mass_function    = model # name of the models

        # model specific attributes
        self.depend_z         = depend_z     # model depends on redshift 
        self.depend_cosmology = depend_cosmo # depends on a cosmology model
        self.mdefs            = mdefs        # allowed mass definition types

        self.cosmology = None # cosmology model to use

    @abstractmethod
    def f(self, sigma: Any, *args, **kwargs) -> Any:
        r"""
        Compute the fitting function.
        """
        ...

    @abstractmethod
    def dndlnm(self, m: Any, *args, **kwargs) -> Any:
        r"""
        Compute the halo mass-function.
        """
        ...

    @abstractmethod
    def dndm(self, m: Any, *args, **kwargs) -> Any:
        r"""
        Compute the halo mass-function.
        """
        ...












