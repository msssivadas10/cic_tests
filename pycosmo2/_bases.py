from typing import Any, Callable, Tuple, Union 
from abc import ABC, abstractmethod, abstractproperty
from pycosmo2.utils.physics import Quantity

# base cosmology object

class CosmologyError(Exception):
    ...

class Cosmology:
    r"""
    Base cosmology class.
    """
    __slots__ = (
                    'Om0', 'Ob0', 'Omnu0', 'Oc0', 'Ode0', 'Ok0', 'Or0', 'Oph0', 'Ornu0', 'Tcmb0', 'Tnu0',
                    'h', 'sigma8', 'ns', 'Nmnu', 'Nrnu', 'Mmnu', 'Mrnu', 'Nnu', 'Mnu', 'w0', 'wa', 'flat', 
                    'relspecies', 'power_spectrum', 'mass_function', 'linear_bias',
                )
    
    def __init__(
                    self, h: float, Om0: float, Ob0: float, sigma8: float, ns: float, flat: bool = True, 
                    relspecies: bool = False, Ode0: float = None, Omnu0: float = 0.0, Nmnu: float = None, 
                    Tcmb0: float = 2.725, w0: float = -1.0, wa: float = 0.0, Nnu: float = 3.0, 
                    power_spectrum: str = None, filter: str = None, mass_function: str = None, 
                    linear_bias: str = None,
                ) -> None:
        self.Om0           : float
        self.Ob0           : float
        self.Omnu0         : float
        self.Oc0           : float
        self.Ode0          : float
        self.Or0           : float
        self.Oph0          : float 
        self.Ornu0         : float
        self.Nmnu          : float
        self.Nnu           : float
        self.Nrnu          : float
        self.Mmnu          : float
        self.Mnu           : float
        self.Mrnu          : float
        self.w0            : float 
        self.wa            : float
        self.Tcmb0         : float
        self.Tnu0          : float
        self.h             : float
        self.sigma8        : float
        self.ns            : float
        self.flat          : bool
        self.relspecies    : bool
        self.power_spectrum: PowerSpectrum
        self.mass_function : HaloMassFunction
        self.linear_bias   : LinearBias

    def __repr__(self) -> str:
        items = [ f'flat={ self.flat }' , f'h={ self.h }', f'Om0={ self.Om0 }', f'Ob0={ self.Ob0 }', f'Ode0={ self.Ode0 }' ]
        if self.Omnu0:
            items.append( f'Omnu0={ self.Onu0 }' )
        if self.relspecies:
            items.append( f'Or0={ self.Or0 }' )
        items = items + [ f'sigma8={ self.sigma8 }',  f'ns={ self.ns }', f'Tcmb0={ self.Tcmb0 }K', f'w0={ self.w0 }',  f'wa={ self.wa }' ]
        return f'Cosmology({ ", ".join( items ) })'

    def wde(self, z: Any, deriv: bool = False) -> Any:
        r"""
        Evolution of equation of state parameter for dark-energy. In general, the dark-energy model 
        is given as

        .. math::
            w(z) = w_0 + w_a \frac{ z }{ 1+z }

        :math:`w_0 = 1` and :math:`w_a = 0` is the cosmological constant.

        Parameters
        ----------
        z: array_like
            Redshift
        deriv: bool, optional
            If true. returns the derivative. Default is false.

        Returns
        -------
        w: array_like
            Value of the equation of state parameter.
        
        Examples
        -------

        """
        ...

    # hubble parameter:

    def E(self, z: Any, square: bool = False) -> Any:
        r"""
        Evolution of Hubble parameter. 

        .. math ::
            E( z ) = \frac{H( z )}{H_0} 
                = \sqrt{ \Omega_m (z+1)^3 + \Omega_k (z+1)^2 + \Omega_r (z+1)^4 + \Omega_{de}(z) }

        Parameters
        ----------
        z: array_like
            Redshift.
        square: bool, optional
            If set true, return the squared value (default is false).

        Returns
        -------
        Ez: array_like
            Value of the function. 

        Examples
        --------

        """
        ...

    @property
    def H0(self) -> float:
        """
        Present value of the Hubble parameter.
        """
        return self.h * 100.0

    def H(self, z: Any) -> Any:
        r"""
        Evolution of Hubble parameter. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        Hz: array_like
            Value of the Hubble parameter. 

        Examples
        --------

        """
        ...

    def dlnEdlnzp1(self, z: Any) -> Any:
        r"""
        Logarithmic derivative of Hubble parameter. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the derivative. 

        Examples
        --------

        """
        ...

    # densities 

    def Om(self, z: Any) -> Any:
        r"""
        Evolution of the density parameter for matter. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the density parameter. 

        Examples
        --------

        """
        ...

    def Ob(self, z: Any) -> Any:
        r"""
        Evolution of the density parameter for baryonic matter. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the density parameter. 

        Examples
        --------

        """
        ...

    def Oc(self, z: Any) -> Any:
        r"""
        Evolution of the density parameter for cold dark-matter. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the density parameter. 

        Examples
        --------

        """
        ...

    def Omnu(self, z: Any) -> Any:
        r"""
        Evolution of the density parameter for warm dark-matter (massive neutrino). 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the density parameter. 

        Examples
        --------

        """
        ...

    def Ode(self, z: Any) -> Any:
        r"""
        Evolution of the density parameter for dark-energy. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the density parameter. 

        Examples
        --------

        """
        ...

    def Ok(self, z: Any) -> Any:
        r"""
        Evolution of the density parameter for curvature. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the density parameter. 

        Examples
        --------

        """
        ...

    def Or(self, z: Any) -> Any:
        r"""
        Evolution of the density parameter for relativistic components. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the density parameter. 

        Examples
        --------

        """
        ...

    def Oph(self, z: Any) -> Any:
        r"""
        Evolution of the density parameter for photons. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the density parameter. 

        Examples
        --------

        """
        ...

    def Ornu(self, z: Any) -> Any:
        r"""
        Evolution of the density parameter for relativistic neutrino. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the density parameter. 

        Examples
        --------

        """
        ...

    def criticalDensity(self, z: Any) -> Quantity:
        r"""
        Evolution of the critical density for the universe. Critical density is the density for the 
        universe to be flat.

        .. math::
            \rho_{\rm crit}(z) = \frac{ 3H(z)^2 }{ 8\pi G } 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: Quantity
            Value of the density. 

        Examples
        --------

        """
        ...

    def rho_m(self, z: Any) -> Quantity:
        r"""
        Evolution of the density for matter. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: Quantity
            Value of the density. 

        Examples
        --------

        """
        ...

    def rho_b(self, z: Any) -> Quantity:
        r"""
        Evolution of the density for baryonic matter. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: Quantity
            Value of the density. 

        Examples
        --------

        """
        ...

    def rho_c(self, z: Any) -> Quantity:
        r"""
        Evolution of the density for cold dark-matter. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: Quantity
            Value of the density. 

        Examples
        --------

        """
        ...

    def rho_mnu(self, z: Any) -> Quantity:
        r"""
        Evolution of the density for warm dark-matter (massive neutrino). 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: Quantity
            Value of the density. 

        Examples
        --------

        """
        ...

    def rho_de(self, z: Any) -> Quantity:
        r"""
        Evolution of the density for dark-energy. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: Quantity
            Value of the density. 

        Examples
        --------

        """
        ...
    
    def rho_r(self, z: Any) -> Quantity:
        r"""
        Evolution of the density for relativistic components. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: Quantity
            Value of the density. 

        Examples
        --------

        """
        ...

    def rho_ph(self, z: Any) -> Quantity:
        r"""
        Evolution of the density for photons. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: Quantity
            Value of the density. 

        Examples
        --------

        """
        ...

    def rho_rnu(self, z: Any) -> Quantity:
        r"""
        Evolution of the density for relativistic neutrinos. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: Quantity
            Value of the density. 

        Examples
        --------

        """
        ...

    # temperature of cmb and cnub:

    def Tcmb(self, z: Any) -> Any:
        r"""
        Evolution of the temperature of cosmic microwave background. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the temperature in Kelvin. 

        Examples
        --------

        """
        ...

    def Tnu(self, z: Any) -> Any:
        r"""
        Evolution of the temperature of cosmic neutrino background. 

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the temperature in Kelvin. 

        Examples
        --------

        """
        ...

    # deceleration parameter

    def q(self, z: Any) -> Any:
        r"""
        Evolution of the deceleration parameter. 

        .. math::
            q( z ) = \frac{ a \ddot{a} }{ \dot{a}^2 }

        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Value of the deceleration parameter. 

        Examples
        --------

        """
        ...
        
    # z-integrals: integrals of z-functions 

    def zIntegral(self, f: Callable, za: Any, zb: Any) -> Any:
        r"""
        Evaluate the definite integral of a function of redshift.

        .. math::
            I = \int_{ z_a }^{ z_b } f(z) {\rm d}z
        
        Parameters
        ----------
        f: callable
            Function to integrate. Must be a callable python function of single argument.
        za, zb: array_like
            Lower and upper limits of integration. Can be any value greater than -1, including `inf`.

        Returns
        -------
        y: array_like
            Values of the integrals.
        
        Examples
        --------

        """
        ...

    def zIntegral_zp1_over_Ez3(self, za: Any, zb: Any) -> Any:
        r"""
        Evaluate the integral

        .. math::
            I = \int_{ z_a }^{ z_b } \frac{ z+1 }{ E(z)^3 } {\rm d}z
        
        Parameters
        ----------
        za, zb: array_like
            Lower and upper limits of integration. Can be any value greater than -1, including `inf`.

        Returns
        -------
        y: array_like
            Values of the integrals.
        
        Examples
        --------
        
        """
        ...

    def zIntegral_1_over_zp1_Ez(self, za: Any, zb: Any) -> Any:
        r"""
        Evaluate the integral

        .. math::
            I = \int_{ z_a }^{ z_b } \frac{ 1 }{ (z+1)E(z) } {\rm d}z
        
        Parameters
        ----------
        za, zb: array_like
            Lower and upper limits of integration. Can be any value greater than -1, including `inf`.

        Returns
        -------
        y: array_like
            Values of the integrals.
        
        Examples
        --------
        
        """
        ...

    def zIntegral_1_over_Ez(self, za: Any, zb: Any) -> Any:
        r"""
        Evaluate the integral

        .. math::
            I = \int_{ z_a }^{ z_b } \frac{ 1 }{ E(z) } {\rm d}z
        
        Parameters
        ----------
        za, zb: array_like
            Lower and upper limits of integration. Can be any value greater than -1, including `inf`.

        Returns
        -------
        y: array_like
            Values of the integrals.
        
        Examples
        --------
        
        """
        ...

    # time and distances

    def universeAge(self, z: Any) -> Any:
        r"""
        Return the age of the universe at redshift z.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Age in years.
        
        Examples
        --------
        
        """
        ...
    
    def lookbackTime(self, z: Any) -> Any:
        r"""
        Return the lookback time at redshift z.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            lookback time in years.
        
        Examples
        --------
        
        """
        ...

    def hubbleTime(self, z: Any) -> Any:
        r"""
        Return the Hubble time (inverse Hubble parameter) at redshift z.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            Hubble time in years.
        
        Examples
        --------
        
        """
        ...
    
    def comovingDistance(self, z: Any) -> Any:
        r"""
        Return the comoving distance corresponding to the redshift z.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
            comoving distance in Mpc.
        
        Examples
        --------
        
        """
        ...

    def comovingCorrdinate(self, z: Any) -> Any:
        r"""
        Return the comving coordinate corresponding to the redshift z.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
           Comving coordinate.
        
        Examples
        --------
        
        """
        ...
    
    def angularDiamaterDistance(self, z: Any) -> Any:
        r"""
        Return the angular diameter distance corresponding to the redshift z.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
           Angular diameter distnace.
        
        Examples
        --------
        
        """
        ...
    
    def luminocityDistance(self, z: Any) -> Any:
        r"""
        Return the luminocity distance corresponding to the redshift z.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
           Luminocity distnace.
        
        Examples
        --------
        
        """
        ...

    # horizons

    def hubbleHorizon(self, z: Any) -> Any:
        r"""
        Return the Hubble horizon at redshift z.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
           Hubble horizon in Mpc.
        
        Examples
        --------
        
        """
        ...
    
    def eventHorizon(self, z: Any) -> Any:
        r"""
        Return the event horizon at redshift z. Event horizon is the maximum comoving distance at 
        which light emitted now could reach.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
           Event horizon in Mpc.
        
        Examples
        --------
        
        """
        ...
    
    def particleHorizon(self, z: Any) -> Any:
        r"""
        Return the particle horizon at redshift z. Particle horizon is the maximum comoving distance from 
        which light could reach the observer within a specific time.
        
        Parameters
        ----------
        z: array_like
            Redshift.

        Returns
        -------
        y: array_like
           Particle horizon in Mpc.
        
        Examples
        --------
        
        """
        ...

    # linear growth

    def g(self, z: Any, exact: bool = False) -> Any:
        r"""
        Linear growth factor, suppressed with respect to that of a matter dominated universe. 
        Actual growth factor would be 

        .. math::
            D_+( z ) = \frac{ 1 }{ z+1 } g(z)

        Parameters
        ----------
        z: array_like
            Redshift
        exact: bool, optional
            If true, compute the growth factor exactly by evaluating an integral expression. 
            Otherwise, use the approximate form given by Carroll et al.

        Returns
        -------
        gz: array_like
            Growth factor values.

        Examples
        --------

        """
        ...

    def Dplus(self, z: Any, exact: bool = False, fac: float = None):
        r"""
        Linear growth factor.

        Parameters
        ----------
        z: array_like
            Redshift
        exact: bool, optional
            If true, compute the growth factor exactly by evaluating an integral expression. 
            Otherwise, use the approximate form given by Carroll et al.
        fac: float, optional
            Normalization of the growth factor. If not given, it is found such that the present 
            value is 1.

        Returns
        -------
        Dz: array_like
            Growth factor values.

        Examples
        --------
        
        """
        ...

    def f(self, z: Any, exact: bool = False) -> Any:
        r"""
        Logarithmic rate of linear growth factor with respect to scale factor.

        .. math::
            f( z ) = - \frac{ {\rm d}\ln D(z) }{ {\rm d}\ln (z+1) } \approx \Omega_m(z)^{ 0.55 }

        Parameters
        ----------
        z: array_like
            Redshift
        exact: bool, optional
            If true, compute the growth factor exactly by evaluating an integral expression. 
            Otherwise, use the approximate form.

        Returns
        -------
        Dz: array_like
            Growth factor values.

        Examples
        --------
        
        """
        ...
    
    def _DplusFreeStream(self, q: Any, Dz: Any, include_nu: bool = False) -> Any:
        r"""
        Growth factor in the presence of free streaming.

        Parameters
        ----------
        q: array_like
            Dimnsionless scale. If multi dimensional array, it will be flattened.
        Dz: array_like
            Linear growth factor. If multi dimensional array, it will be flattened.
        include_nu: bool, optional
            If true, returns the growth factor of fluctuations including massive neutrinos. Else, 
            return that of only baryons and cold dark-matter.
        
        Returns
        -------
        Dz: array_like
            Growth factor. If no neutrinos are presnt, then this will be same as the input growth 
            factor. 

        Examples
        --------

        """
        ...

    def DplusFreeStream(self, q: Any, z: Any, include_nu: bool = False, exact: bool = False, fac: float = None) -> Any:
        r"""
        Growth factor in the presence of free streaming.

        Parameters
        ----------
        q: array_like
            Dimnsionless scale. If multi dimensional array, it will be flattened.
        z: array_like
            Redshift. If multi dimensional array, it will be flattened.
        include_nu: bool, optional
            If true, returns the growth factor of fluctuations including massive neutrinos. Else, 
            return that of only baryons and cold dark-matter.
        
        Returns
        -------
        Dz: array_like
            Growth factor. If no neutrinos are presnt, then this will be same as the linear growth 
            factor. 

        Examples
        --------
        
        """
        ...

    # power spectrum 

    def linearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        r"""
        Compute the linear matter power spectrum. This power spectrum will correspond to the specified 
        transfer function model. 

        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift (default is 0).
        dim: bool, optional
            If true (default) return the power spectrum in :math:`h^{-3}/{\rm Mpc}^3`, otherwise gives the 
            dimensionless power spectrum.
        
        Returns
        -------
        pk: array_like
            Values of power spectrum.

        Examples
        --------

        """
        ...

    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        r"""
        Compute the non-linear matter power spectrum. This power spectrum will correspond to the specified 
        model. 

        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift (default is 0).
        dim: bool, optional
            If true (default) return the power spectrum in :math:`h^{-3}/{\rm Mpc}^3`, otherwise gives the 
            dimensionless power spectrum.
        
        Returns
        -------
        pk: array_like
            Values of power spectrum.

        Examples
        --------
        
        """
        ...

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, linear: bool= True) -> Any:
        r"""
        Compute the linear or non-linear matter power spectrum. This power spectrum will correspond to the 
        specified transfer function model. The linear power spectrum is given in terms of the transfer function 
        :math:`T(k)` as

        .. math::
            P_{\rm lin}(k, z) = A k^{ n_s } T(k)^2 D_+(z)^2

        where :math:`A` is the normalization factor. Non-linear power spectrum is modelled as some function of 
        the linear power spectrum. Thes will have units :math:`h^{-3}/{\rm Mpc}^3`. The dimensionless power 
        spectrum is defined as 

        .. math ::
            \Delta^2(k, z) = \frac{ k^3 P(k, z) }{ 2\pi^2 }

        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift (default is 0).
        dim: bool, optional
            If true (default) return the power spectrum in :math:`h^{-3}/{\rm Mpc}^3`, otherwise gives the 
            dimensionless power spectrum.
        linear: bool, optional
            If true (default) return the linear model, else non-linear.
        
        Returns
        -------
        pk: array_like
            Values of power spectrum.

        Examples
        --------
        
        """
        ...

    def matterCorreleation(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the two-point correlation function for matter.

        .. math::
            \xi(r, z) = \int_0^\infty \Delta^2(k, z) \frac{ \sin kr }{ kr } {\rm d}\ln k

        Parameters
        ----------
        r: array_like
           Seperation between two spacial points in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the linear model, else non-linear.
        
        Returns
        -------
        xi: array_like
            Values of power spectrum.

        Examples
        --------
        
        """
        ...

    def variance(self, r: Any, z: float = 0, linear: bool = True, j: int = 0) -> Any:
        r"""
        Compute the variance of matter fluctuations, smoothed at radius :math:`r` using a filter function,
        :math:`w (kr )`, such as a spherical top-hat. 

        .. math::
            \\sigma^2(r, z) = \int_0^\infty \Delta^2(k, z) w( kr )^2 {\rm d}\ln k

        Parameters
        ----------
        r: array_like
           Seperation between two spacial points in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the linear model, else non-linear.
        j: int, optional
            Indicates the moment, j = 0 (default) moment is the variance.
        
        Returns
        -------
        var: array_like
            Values of variance.

        Examples
        --------
        
        """
        ...
    
    def radius(self, sigma: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Invert the variance to find the smoothing radius.

        Parameters
        ----------
        sigma: array_like
            Variance values (linear or non-linear, specified by `linear` argument), to be exact, their 
            square roots.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) use the linear variance, else the non-linear variance.
        
        Returns
        -------
        r: array_like
            Smoothing radius in Mpc/h.

        Examples
        --------

        """
        ...

    def dlnsdlnr(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the first logarithmic derivative of smoothed variance.

        Parameters
        ----------
        r: array_like
           Seperation between two spacial points in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the linear model, else non-linear.
        
        Returns
        -------
        y: array_like
            Values of derivative of variance.

        Examples
        --------
        
        """
        ...

    def dlnsdlnm(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the first logarithmic derivative of matter fluctuations variance w.r.to mass.

        Parameters
        ----------
        m: array_like
            Smoothing radius in Mpc/h, corresponding to the mass.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the value for linear variance, else for non-linear variance.
        
        Returns
        -------
        y: array_like
            Values of the derivative.

        Examples
        --------

        """
        ...

    def d2lnsdlnr2(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the second logarithmic derivative of smoothed variance.

        Parameters
        ----------
        r: array_like
           Seperation between two spacial points in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the linear model, else non-linear.
        
        Returns
        -------
        y: array_like
            Values of derivative of variance.

        Examples
        --------
        
        """
        ...

    def effectiveIndex(self, k: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the effective slope or index of the power spectrum.

        .. math::
            n_{\rm eff}(k, z) = \frac{ {\rm d}\ln P(k,z) }{ {\rm d}\ln k }

        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the slope for linear model, else non-linear.

        Returns
        -------
        neff: array_like
            Effective slope at given wavenumber.

        Examples
        --------

        """
        ...
    
    def nonlineark(self, k: Any, z: float = 0) -> Any:
        r"""
        Compute the non-linear wavenumber corresponding to a linear one.

        Parameters
        ----------
        k: array_like
            linear wavenumber in h/Mpc.
        z: float, optional
            Redshift (default is 0).

        Returns
        -------
        knl: array_like
            Non-linear wavenumber in h/Mpc.

        Examples
        --------
        
        """
        ...

    # halo mass function, bias and related calculations

    def lagrangianR(self, m: Any) -> Any:
        r"""
        Compute the Lagrangian radius corresponding to a mass.

        Parameters
        ----------
        m: array_like
            Mass in Msun/h.

        Returns
        -------
        r: array_like
            Lagrangian radius in Mpc/h.

        Examples
        --------

        """
        ...

    def lagrangianM(self, r: Any) -> Any:
        r"""
        Compute the mass corresponding to a Lagrangian radius.

        Parameters
        ----------
        r: array_like
            Lagrangian radius in Mpc/h.
        
        Returns
        -------
        m: array_like
            Mass in Msun/h.

        Examples
        --------
        
        """
        ...

    def collapseOverdensity(self, z: Any) -> float:
        """
        Critical value for the spherical collapse overdensity. It's value is approximately 1.686.

        Parameters
        ----------
        z: array_like
            Redshift. In general, it has a very small redshift dependence. But, here it is taken as a 
            constant in time. So, value of this argument is unused, except the shape.

        Returns
        -------
        delta_c: array_like
            Value of the collapse overdensity.

        Examples
        --------

        """
        ...

    def massFunction(self, m: Any, z: float = 0, overdensity: Union[int, str, 'OverDensity'] = ..., out: str = 'dndlnm') -> Any:
        r"""
        Compute the halo mass-function and return the value in a specified format.

        Parameters
        ----------
        m: array_like
            Mass of the halo in Msun/h.
        z: float, optional
            Redshift (default is 0).
        overdensity: str, int, OverDensity, optional
            Value of the overdensity. It could be an integer value of overdensity w.r.to the mean background density, 
            a string indicating the value such as `200m`, `vir` etc., or an :class:`OverDensity` object. For FoF type 
            halos, its default value is `fof` and for spherical overdensity (SO) halos, it is 200.
        out: str, optional
            Output format for the mass function. Must be either of `f`, `dndm`, `dndlnm` (default) or `dndlog10m`.

        Returns
        -------
        mf: array_like
            Halo mass-function in specified format.

        Examples
        --------

        """
        ...

    def linearBias(self, m: Any, z: float = 0, overdensity: Union[int, str, 'OverDensity'] = None) -> Any:
        r"""
        Compute the linear halo bias of a given model.

        Parameters
        ----------
        m: array_like
            Mass of the halo in Msun/h.
        z: float, optional
            Redshift (default is 0).
        overdensity: str, int, OverDensity, optional
            Value of the overdensity. It could be an integer value of overdensity w.r.to the mean background density, 
            a string indicating the value such as `200m`, `vir` etc., or an :class:`OverDensity` object. For FoF type 
            halos, its default value is None (not used) and for spherical overdensity (SO) halos, it is 200.

        Returns
        -------
        bias: array_like
            Linear halo bias values.
        
        Examples
        --------

        """


# smoothing filters

class Filter(ABC):
    r"""
    Base class representing a smoothing filter in k-space.
    """

    @abstractmethod
    def filter(self, x: Any, j: int = 0) -> Any:
        r"""
        Functional form of the filter.

        Parameters
        ----------
        x: array_like
            Argument.
        j: int, optional
            Specifies the n-the derivative. Default is 0, meaning the function itself.

        Returns
        -------
        wx: array_like
            Value of the function.
        """
        ...

    def convolution(self, f: Callable, r: Any, args: tuple = (), ) -> Any:
        r"""
        Convolve a function with the filter. i.e., smooth the function. The convolution of a function 
        :math:`f(k)` with wthe filter is given by the integral

        .. math::
            F(r) = \int_0^\infty f(k) w(kr)^2 {\rm d}k

        Parameters
        ----------
        f: callable
            Function to convolve with the filter.
        r: array_like
            Smoothing radius or convolution argument.
        args: tuple, optional
            Other arguments to be passed to the function call.

        Returns
        -------
        F: array_like
            Value of the convolution.
        """
        ...

    def dcdr(self, f: Callable, r: Any, args: tuple = (), ) -> Any:
        r"""
        Compute the first derivative of the convolution.

        Parameters
        ----------
        f: callable
            Function to convolve with the filter.
        r: array_like
            Smoothing radius or convolution argument.
        args: tuple, optional
            Other arguments to be passed to the function call.

        Returns
        -------
        dF: array_like
            Value of the derivative of convolution.
        """
        ...

    def d2cdr2(self, f: Callable, r: Any, args: tuple = (), ) -> Any:
        r"""
        Compute the second derivative of the convolution.

        Parameters
        ----------
        f: callable
            Function to convolve with the filter.
        r: array_like
            Smoothing radius or convolution argument.
        args: tuple, optional
            Other arguments to be passed to the function call.

        Returns
        -------
        d2F: array_like
            Value of the derivative of convolution.
        """
        ...


# base power spectrum object

class PowerSpectrumError(Exception):
    r"""
    Base class of exception used by  power spectrum objects.
    """
    ...

class PowerSpectrum(ABC):
    r"""
    Base power spectrum class. 
    """

    __slots__ = 'filter', 'cosmology', 'A', 'use_exact_growth', 'nonlinear_model', 'linear_model',

    def __init__(self, cm: Cosmology, filter: str = 'tophat') -> None:
        self.filter          : Filter
        self.cosmology       : Cosmology
        self.A               : float
        self.use_exact_growth: bool
        self.nonlinear_model : str
        self.linear_model    : str 

    def Dplus(self, z: Any) -> Any:
        r"""
        Linear growth factor.

        Parameter
        ---------
        z: array_like
            Redshift. Must be a value greater than -1.

        Returns
        --------
        Dz: array_like
            Value of the growth factor.
        """
        return self.cosmology.Dplus( z, exact = self.use_exact_growth )
    
    @property 
    def ns(self) -> float:
        r"""
        Power spectrum index, :math:`n_s`.
        """
        return self.cosmology.ns

    @property 
    def sigma8(self) -> float:
        r"""
        RMS variance of matter fluctuations at 8 Mpc/h scale, :math:`\sigma_8`
        """
        return self.cosmology.sigma8
    
    @abstractmethod
    def transferFunction(self, k: Any, z: float = 0) -> Any:
        """
        Linear transfer function.

        Parameters
        ----------
        k: array_like
            Wavenumber in h/Mpc.
        z: float, optional
            Redshift (default is 0)

        Returns
        -------
        tk: array_like
            Value of linear transfer function.
        """
        ...

    def linearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        r"""
        Linear matter power spectrum.

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        dim: bool, optional
            If true (default), return the usual power spectrum, else give the dimenssionless one.
        
        Returns
        -------
        pk: array_like
            Linear (dimenssionless) power spectrum values.

        """
        ...

    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        r"""
        Non-liinear matter power spectrum. 

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        dim: bool, optional
            If true (default), return the usual power spectrum, else give the dimenssionless one.
        
        Returns
        -------
        pk: array_like
            Non-linear (dimenssionless) power spectrum values.

        See Also
        --------
        `PowerSpectrum.linearPowerSpectrum`;
            Linear matter power spectrum.
            
        """
        ...

    def nonlineark(self, k: Any, z: float = 0) -> Any:
        r"""
        Compute the non-liinear wavenumber corresponding to the linear one. 

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        
        Returns
        -------
        knl: array_like
            Non-linear wavenumber in h/Mpc.
            
        """
        ...

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, linear: bool = True) -> Any:
        r"""
        Compute the linear or non-linear matter power spectrum. Linear power spectrum is given in 
        in terms of the linear transfer function and growth factor as 

        .. math::
            P_{\rm lin}(k, z) = A k^{n_s} T(k)^2 D_+(z)^2

        Non-linear power spectrum can be related to the linear power by some transformation. The 
        dimenssionless power spectrum is then defined as

        .. math::
            \Delta^2(k, z) = \frac{1}{2\pi^2} k63 P(k, z)

        where the power spectrum could be linear or non-linear.

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        dim: bool, optional
            If true (default), return the usual power spectrum, else give the dimenssionless one.
        linear: bool, optional
            If true (default) return the linear power spectrum, else the non-linear power spectrum.
        
        Returns
        -------
        pk: array_like
            Matter (dimenssionless) power spectrum values.

        """
        ...

    def matterCorrelation(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the linear or non-linear 2-point matter correlation function.

        Parameters
        ----------
        r: array_like
            Seperation between the two points in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the linear correlation, else the non-linear correlation.
        
        Returns
        -------
        xr: array_like
            Matter correlation function values.

        """
        ...

    def variance(self, r: Any, z: float = 0, linear: bool = True, j: int = 0) -> Any:
        r"""
        Compute the linear or non-linear matter fluctuations variance.

        Parameters
        ----------
        r: array_like
            Smoothing radius in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the linear variance, else the non-linear variance.
        j: int, optional
            Indicates the moment, j = 0 (default) moment is the variance.
        
        Returns
        -------
        var: array_like
            Matter fluctuation variance.

        """
        ...

    def dlnsdlnr(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the first logarithmic derivative of matter fluctuations variance w.r.to radius.

        Parameters
        ----------
        r: array_like
            Smoothing radius in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the value for linear variance, else for non-linear variance.
        
        Returns
        -------
        y: array_like
            Values of the derivative.

        """
        ...

    def d2lnsdlnr2(self, r: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the second logarithmic derivative of matter fluctuations variance w.r.to radius.

        Parameters
        ----------
        r: array_like
            Smoothing radius in Mpc/h.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the value for linear variance, else for non-linear variance.
        
        Returns
        -------
        y: array_like
            Values of the derivative.

        """
        ...

    def radius(self, sigma: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Invert the variance to find the smoothing radius.

        Parameters
        ----------
        sigma: array_like
            Variance values (linear or non-linear, specified by `linear` argument), to be exact, their 
            square roots.
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) use the linear variance, else the non-linear variance.
        
        Returns
        -------
        r: array_like
            Smoothing radius in Mpc/h.

        """
        ...

    def effectiveIndex(self, k: Any, z: float = 0, linear: bool = True) -> Any:
        r"""
        Compute the effective power spectrum index (effective slope).

        .. math::
            n_{]\rm eff}(k) = \frac{ {\rm d}\ln P(k) }{ {\rm d}\ln k }

        Parameters
        ----------
        k: array_like
            Wavenumbers in h/Mpc
        z: float, optional
            Redshift (default is 0).
        linear: bool, optional
            If true (default) return the index for linear power spectrum, else the non-linear power spectrum.
        
        Returns
        -------
        neff: array_like
            Power spectrum index values.
        """
        ...

    def normalize(self) -> None:
        r"""
        Normalize the power spectrum using the value of :math:`\sigma_8` parameter.
        """
        ...


# overdensity models

class OverDensity(ABC):
    r"""
    Represent the value of overdensity with respect to some reference. Halos are formed by the collapse of 
    over-dense regions and the overdensity is the density at that place, with respect to mean background 
    density (denoted by :math:`Delta`). Spherical overdenisties are specified w.r.to the mean density or 
    critical density (denoted by an integer, followed by `m` or `c` respectively, eg., `200m` or `500c`). 
    """

    __slots__ = '_value'

    def __init__(self, value: int = 1) -> None:
        self._value = int( value )

    def __repr__(self) -> str:
        return f"Overdensity('{ self._value }')"

    @abstractmethod
    def value(self, z: float, cm: Cosmology) -> int:
        r"""
        Value of the overdensity with respect to mean matter density.

        Parameters
        ----------
        z: float
            Redshift
        cm: Cosmology
            Cosmology model to use.

        Returns
        -------
        overdensity: int
            Overdensity value with respect to background density.

        """
        ...

# halo mass function models

class HaloMassFunctionError(Exception):
    r"""
    Base class of exceptions used by halo mass function objects.
    """

class HaloMassFunction(ABC):
    r"""
    Base halo mass-function class. 
    """
    
    __slots__ = 'model', 'flags', 'cosmology'

    def __init__(self, cm: Cosmology) -> None:
        self.cosmology : Cosmology
        self.model     : str
        self.flags     : int 
    
    @abstractmethod
    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, OverDensity] = None) -> Any:
        r"""
        Fitting function for the mass-function.

        Parameters
        ----------
        sigma: array_like
            Density field fluctuation variance.
        z: float, optional
            Redshift (default is 0).
        overdensity: str, int, OverDensity, optional
            Value of the overdensity. For FoF type halos, its default value is `fof` and for spherical overdensity 
            (SO) halos, it is 200.

        Returns
        -------
        f: array_like
            Halo mass function values.
        """
        ...

    def massFunction(self, m: Any, z: float = 0, overdensity: Union[int, str, OverDensity] = None, out: str = 'dndlnm') -> Any:
        r"""
        Compute the halo mass-function and return the value in a specified format.

        Parameters
        ----------
        m: array_like
            Mass of the halo in Msun/h.
        z: float, optional
            Redshift (default is 0).
        overdensity: str, int, OverDensity, optional
            Value of the overdensity. It could be an integer value of overdensity w.r.to the mean background density, 
            a string indicating the value such as `200m`, `vir` etc., or an :class:`OverDensity` object. For FoF type 
            halos, its default value is `fof` and for spherical overdensity (SO) halos, it is 200.
        out: str, optional
            Output format for the mass function. Must be either of `f`, `dndm`, `dndlnm` (default) or `dndlog10m`.

        Returns
        -------
        mf: array_like
            Halo mass-function in specified format.
        """
        ...

    @abstractproperty
    def zDependent(self) -> bool:
        """
        Whether the model is redshift-dependent?
        """
        ...

    @abstractproperty
    def cosmoDependent(self) -> bool:
        """
        Whether the model is cosmology dependent?
        """
        ...

    @abstractproperty
    def mdefs(self) -> Tuple[str]:
        """
        Valid mass definitions/overdensities.
        """
        ...

# linear bias function

class LinearBias(ABC):
    r"""
    Base class for linear bias function models.
    """

    __slots__ = 'flags', 'cosmology', 'corrections'

    def __init__(self) -> None:
        self.flags      : int
        self.cosmology  : Cosmology
        self.corrections: bool 

    @abstractmethod
    def b(self, nu: Any, z: float = 0, overdensity: Union[int, str, OverDensity] = None) -> Any:
        r"""
        Linear bias function.

        Parameters
        ----------
        nu: array_like
            Input argument. :math:`\nu(M) = \delta_c / \sigma(M)`.
        z: float, optional
            Redshift (default is 0).
        overdensity: str, int, OverDensity
            Overdensity value. Only needed for models based on spherical overdensity.

        Returns
        -------
        b: array_like
            Bias function values.
        """
        ...