from typing import Any 
import numpy as np
import pycosmo2.cosmology.cosmo as cosmo
import pycosmo2._bases as base
import pycosmo2.utils.settings as settings
import pycosmo2.utils.numeric as numeric
import pycosmo2.power_spectrum.filters as filters
import pycosmo2.power_spectrum.nonlinear_power as nlp
import pycosmo2.power_spectrum.transfer_functions as tf

from pycosmo2._bases import PowerSpectrumError

class PowerSpectrum(base.PowerSpectrum):
    r"""
    An abstract power spectrum class. A properly initialised power spectrum object cann be 
    used to compute the linear and non-linear power spectra, matter variance etc.

    Parameters
    ----------
    cm: Cosmology
        Working cosmology object. Power spectrum objects extract cosmology parameters from 
        these objects.
    filter: str. optional
        Filter to use for smoothing the density field. Its allowed values are `tophat` (default), 
        `gauss` and `sharpk`.
    
    Raises
    ------
    PowerSpectrumError

    """

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        self.linear_model    = None
        self.nonlinear_model = 'halofit'

        if not isinstance( cm, cosmo.Cosmology ):
            raise PowerSpectrumError("cm must be a 'Cosmology' object")
        self.cosmology        = cm
        self.use_exact_growth = False # whether to use exact (integrated) growth factors

        if filter not in filters.filters:
            raise PowerSpectrumError(f"invalid filter: { filter }")
        self.filter = filters.filters[ filter ] # filter to use for smoothing

        self.normalize()

    def Dplus(self, z: Any) -> Any:
        return self.cosmology.Dplus( z, exact = self.use_exact_growth )

    def linearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        k = np.asfarray( k )

        if np.ndim( z ):
            raise PowerSpectrumError("z must be a scalar")
        if z + 1 < 0:
            raise ValueError("redshift cannot be less than -1")

        Pk = self.A * k**self.ns * self.transferFunction( k, z )**2 * self.Dplus( z )**2
        if not dim:
            return k**3 * Pk / ( 2*np.pi**2 )
        return Pk 

    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        return nlp.nonlinearPowerSpectrum( self, k, z, dim, model = self.nonlinear_model )

    def nonlineark(self, k: Any, z: float = 0) -> Any:
        k   = np.asfarray( k )
        dnl = self.nonlinearPowerSpectrum( k, z, dim = False )
        return k * np.cbrt( 1.0 + dnl )
    
    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True, linear: bool = True) -> Any:
        if linear:
            return self.linearPowerSpectrum( k, z, dim ) 
        return self.nonlinearPowerSpectrum( k, z, dim )

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
        return filters.j0convolution(  self.matterPowerSpectrum, r, args = ( z, False, linear ) )

    def variance(self, r: Any, z: float = 0, linear: bool = True) -> Any:
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
        
        Returns
        -------
        var: array_like
            Matter fluctuation variance.

        """
        return self.filter.convolution( self.matterPowerSpectrum, r, args = ( z, False, linear ) )

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
        r  = np.asfarray( r )
        y0 = self.variance( r, z, linear )
        y1 = self.filter.dcdr( self.matterPowerSpectrum, r, args = ( z, False, linear ) )
        return 0.5 * r * y1 / y0

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
        h     = settings.DEFAULT_H
        r     = np.asfarray( r )

        df    = (
                    -self.dlnsdlnr( ( 1+2*h )*r, z, linear )
                        + 8*self.dlnsdlnr( ( 1+h )*r, z, linear )
                        - 8*self.dlnsdlnr( ( 1-h )*r, z, linear )
                        +   self.dlnsdlnr( ( 1-2*h )*r, z, linear )
                ) # f := dlns/dlnr
               
        dlnr = 6.0 * ( np.log( (1+h)*r ) - np.log( (1-h)*r ) )
        
        return df / dlnr

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
        def f(lnr: Any, v: Any, z: float, linear: bool) -> Any:
            r = np.exp( lnr )
            return self.variance( r, z, linear ) - v

        v   = np.asfarray( sigma )**2
        lnr = numeric.solve( 
                                f, a = np.log( 1e-04 ), b = np.log( 1e+04 ), 
                                args = ( v, z, linear ), tol = settings.RELTOL 
                           )
        return np.exp( lnr )

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
        def lnPower(k: Any, z: float, linear: bool) -> Any:
            return np.log( self.matterPowerSpectrum( k, z, dim = True, linear = linear ) )

        h    = settings.DEFAULT_H
        k    = np.asfarray( k )
        dlnp = (
                    -lnPower( (1+2*h)*k, z, linear ) 
                        + 8*lnPower( (1+h)*k,   z, linear ) 
                        - 8*lnPower( (1-h)*k,   z, linear ) 
                        +   lnPower( (1-2*h)*k, z, linear )
               )
               
        dlnk = 6.0 * ( np.log( (1+h)*k ) - np.log( (1-h)*k ) )
        
        return dlnp / dlnk

    def normalize(self) -> None:
        r"""
        Normalize the power spectrum using the value of :math:`\sigma_8` parameter.
        """
        self.A = 1.0 # power spectrum normalization factor
        self.A = self.sigma8**2 / self.variance( 8.0 ) 



#######################################################################################################

# predefined models

class Sugiyama96(PowerSpectrum):
    r"""
    Matter power spectrum based on the transfer function given by Bardeen et al.(1986), with correction 
    given by Sugiyama(1995) [1]_.

    Parameters
    ----------
    cm: Cosmology
        Working cosmology object. Power spectrum objects extract cosmology parameters from 
        these objects.
    filter: str. optional
        Filter to use for smoothing the density field. Its allowed values are `tophat` (default), 
        `gauss` and `sharpk`.
    
    Raises
    ------
    PowerSpectrumError

    References
    ----------
    .. [1] A. Meiksin, matrin White and J. A. Peacock. Baryonic signatures in large-scale structure, 
            Mon. Not. R. Astron. Soc. 304, 851-864, 1999.  
    """

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'sugiyama96'

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return tf.psmodelSugiyama96( self.cosmology, k, z )

class BBKS(Sugiyama96):
    r"""
    Same as :class:`Sugiyama96`.
    """

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'bbks'

class Eisenstein98_zeroBaryon(PowerSpectrum):
    r"""
    Matter power spectrum based on the transfer function given by Eisenstein and Hu (1997), without including any 
    baryon oscillations [1]_.

    Parameters
    ----------
    cm: Cosmology
        Working cosmology object. Power spectrum objects extract cosmology parameters from 
        these objects.
    filter: str. optional
        Filter to use for smoothing the density field. Its allowed values are `tophat` (default), 
        `gauss` and `sharpk`.
    
    Raises
    ------
    PowerSpectrumError

    References
    ----------
    .. [1] Daniel J. Eisenstein and Wayne Hu. Baryonic Features in the Matter Transfer Function, 
            `arXive:astro-ph/9709112v1, <http://arXiv.org/abs/astro-ph/9709112v1>`_, 1997.
    """

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'eisenstein98_zb'

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return tf.psmodelEisenstein98_zeroBaryon( self.cosmology, k, z )

class Eisenstein98_withBaryon(PowerSpectrum):
    r"""
    Matter power spectrum based on the transfer function given by Eisenstein and Hu (1997), including the 
    baryon oscillations [1]_.

    Parameters
    ----------
    cm: Cosmology
        Working cosmology object. Power spectrum objects extract cosmology parameters from 
        these objects.
    filter: str. optional
        Filter to use for smoothing the density field. Its allowed values are `tophat` (default), 
        `gauss` and `sharpk`.
    
    Raises
    ------
    PowerSpectrumError

    References
    ----------
    .. [1] Daniel J. Eisenstein and Wayne Hu. Baryonic Features in the Matter Transfer Function, 
            `arXive:astro-ph/9709112v1, <http://arXiv.org/abs/astro-ph/9709112v1>`_, 1997.
    """

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'eisenstein98_wb'

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return tf.psmodelEisenstein98_withBaryon( self.cosmology, k, z )

class Eisenstein98_withNeutrino(PowerSpectrum):
    r"""
    Matter power spectrum based on the transfer function given by Eisenstein and Hu (1997), including massive 
    neutrinos  [1]_.

    Parameters
    ----------
    cm: Cosmology
        Working cosmology object. Power spectrum objects extract cosmology parameters from 
        these objects.
    filter: str. optional
        Filter to use for smoothing the density field. Its allowed values are `tophat` (default), 
        `gauss` and `sharpk`.
    
    Raises
    ------
    PowerSpectrumError

    References
    ----------
    .. [1] Daniel J. Eisenstein and Wayne Hu. Power Spectra for Cold Dark Matter and its Variants, 
            `arXive:astro-ph/9710252v1, <http://arXiv.org/abs/astro-ph/9710252v1>`_, 1997.
    """

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'eisenstein98_nu'

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return tf.psmodelEisenstein98_withNeutrino( self.cosmology, k, z, self.use_exact_growth )


# power spectrum from raw data:

class PowerSpectrumTable(PowerSpectrum):
    r"""
    Matter power spectrum based on a tabulated transfer function. This can be used to create power spectrum 
    objects based on transfer function computed by external software etc. Input data must be given as logarithm 
    of wavenumbers (in h/Mpc) and transfer function. i.e., :math:`\ln(k)` and :math:`\ln T(k)`.

    Parameters
    ----------
    lnk: array_like
        Natural logarithm of the wavenumber data. Must be an 1D array of valid size. Wavenumbers must be in h/Mpc.
    lnt: array_like
        Natural logarithm of the transfer function data. Must be an 1D array of valid size.
    cm: Cosmology
        Working cosmology object. Power spectrum objects extract cosmology parameters from 
        these objects.
    filter: str. optional
        Filter to use for smoothing the density field. Its allowed values are `tophat` (default), 
        `gauss` and `sharpk`.
    
    Raises
    ------
    PowerSpectrumError

    """

    __slots__ = 'spline', 

    def __init__(self, lnk: Any, lnt: Any, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'rawdata'

        lnk, lnt = np.asfarray( lnk ), np.asfarray( lnt )
        if np.ndim( lnk ) != 1 or np.ndim( lnt ) != 1:
            raise base.PowerSpectrumError("both lnk and lnt must be 1-dimensional arrays")
        
        from scipy.interpolate import CubicSpline

        self.spline = CubicSpline( lnk, lnt )

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return np.exp( self.spline( np.log( k ) ) )


