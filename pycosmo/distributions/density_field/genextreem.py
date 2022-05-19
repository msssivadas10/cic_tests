from typing import Any 
from itertools import product, repeat
import warnings
from scipy.interpolate import CubicSpline
from scipy.special import gamma
from scipy.optimize import newton
from pycosmo.cosmology import Cosmology, CosmologyError
from pycosmo.distributions.base import Distribution, DistributionError
import pycosmo.utils.numeric as numeric 
import pycosmo.utils.gaussrules as gaussrules
import pycosmo.utils.settings as settings
import numpy as np

class GenExtremeParameters:

    __slots__ = 'loc', 'scale', 'shape'

    def __init__(self, loc: float, scale: float, shape: float) -> None:
        self.loc   = loc
        self.scale = scale
        self.shape = shape

    def __repr__(self) -> str:
        return f"GenExtremeParameters(loc={ self.loc }, scale={ self.scale }, shape={ self.shape })"

class GenExtremeDistribution(Distribution):
    r"""
    Generalised extreme value distribution for the overdensity field. This distribution is based on the reversed 
    Wiebull subclass [1]_.

    .. math::   
        P( A ) = \frac{1}{ \sigma } t(A)^{ 1 + \xi } \exp[ -t(A) ]

    where :math:`A = \ln (\delta+1)` is the logarithmic overdensity field, 

    .. math::
        t(A) = \left[ 1 + \frac{A - \mu}{\sigma} \xi \right]^{-1/\xi}

    This distribution is charecterised by the location parameter :math:`\mu`, positive scale parameter :math:`\sigma` 
    and the negative shape parameter :math:`\xi`.

    Parameters
    ----------
    cm: Cosmology
        Cosmology object to use.
    r: float
        Pixel size or smoothing size of the overdenisty field in Mpc/h.

    References
    ----------
    .. [1] Andrew Repp & István Szapudi. Precision Prediction for the Cosmological Density Distribution. arXive: 1705.08015v2 (2018).  
    
    """

    # global setting for objects
    INTERP_N     = 101   # number of interpolation points
    MEAN_N       = 501   # number of samples for averaging
    EXACT_GROWTH = False # use exact growth factor 

    __slots__ = 'cosmology', 'z', 'r', 'kn', 'b2_log', 'meas_power_spectrum', 'param'

    def __init__(self, cm: Cosmology, r: float) -> None:
        
        if not isinstance( cm, Cosmology ):
            raise TypeError("cm must be a 'Cosmology' object")
        self.cosmology = cm

        self.r  = r         # size of the box (Mpc/h)
        self.kn = np.pi / r # nyquist wavenumber (h/Mpc)

        self.b2_log = 1.0 # log field bias 
        self.param  : GenExtremeParameters

        self.setup( self.cosmology.sigma8 )
    
    def measuredPowerSepctrum(self, kx: Any, ky: Any, kz: Any, z: float = 0) -> Any:
        r"""
        Return the measured log-field power spectrum.
        """
        kx, ky, kz = np.asfarray( kx ), np.asfarray( ky ), np.asfarray( kz )
        
        lnk = 0.5*np.log( kx**2 + ky**2 + kz**2 )
        pk  = np.exp( self.meas_power_spectrum( lnk ) )
        return pk * self.cosmology.Dplus( z, exact = self.EXACT_GROWTH )**2 * self.b2_log

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
        def Delta2(lnk: Any, z: float) -> Any:
            return self.cosmology.linearPowerSpectrum( np.exp( lnk ), z, dim = False )
        
        var = numeric.integrate2( 
                                    Delta2,
                                    a = np.log( settings.ZERO ), b = np.log( self.kn ), 
                                    args = (z, ),
                                    eps = settings.RELTOL,
                                    n = settings.DEFAULT_N
                                ) # Eqn. 3
        return var

    def sigma2A(self, arg: float, invert: bool = False) -> float:
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
        if invert:
            return mu * np.exp( arg / mu ) - mu
        return mu * np.log( 1 + arg / mu ) # Eqn. 5

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
        def integrand(lnkx: Any, lnky: Any, lnkz: Any, z: float = 0) -> Any:
            kx, ky, kz = np.exp( lnkx ), np.exp( lnky ), np.exp( lnkz )
            return kx * ky * kz * self.measuredPowerSepctrum( kx, ky, kz, z )

        nodes, wg, wk = gaussrules.legendrerule( 64 )

        # transform interval
        a, b   = np.log( settings.ZERO ), np.log( self.kn )
        m      = 0.5*( b - a )
        nodes  = nodes * m + ( a + m )
        wg, wk = wg * m, wk * m

        y = integrand( *np.meshgrid( nodes, nodes, nodes ), z )

        # gauss integral
        weight = np.prod( np.meshgrid( wg, wg, wg ), axis = 0 )
        Ig     = np.sum( y[ 1:-1:2, 1:-1:2, 1:-1:2 ] * weight ) / ( np.pi )**3

        # konrod integral
        weight = np.prod( np.meshgrid( wk, wk, wk ), axis = 0 )
        Ik     = np.sum( y * weight ) / ( np.pi )**3

        if not np.allclose( Ik, Ig, settings.RELTOL, settings.ABSTOL ):
            warnings.warn("Integral is not converged")

        return Ik
    
    def averageA(self, arg: float, invert: bool = False) -> float:
        r"""
        Return the fitted value of the log-field average, :math:`\langle A \rangle`.

        Parameters
        ----------
        arg: float
            Linear variance value.
        
        Returns
        -------
        mean: float
            Best fiiting value of logarithmic field mean.

        """
        lamda = 0.65
        if invert:
            return 2*lamda*np.exp( -arg / lamda ) - 2*lamda
        return -lamda * np.log( 1 + 0.5*arg / lamda ) # Eqn. 8

    def skewnessA(self, arg: float, invert: bool = False) -> float:
        r"""
        Return the fitted value of the log-field skewnesss, :math:`T_3`.

        Parameters
        ----------
        arg: float
            Log field variance value (measured).
        
        Returns
        -------
        mean: float
            Best fiiting value of logarithmic field mean.

        """    
        a, b, c, d = -0.70, 1.25, -0.26, 0.06 # Eqn. 15

        np3 = -6*self.cosmology.dlnsdlnm( self.r ) # Lukic et al (2018). Eqn. 42
        # np3 = self.cosmology.effectiveIndex( self.kn ) + 3
        Tn  = a*np3 + b           # Eqn. 13
        pn  = d + c*np.log( np3 ) # Eqn. 14
        if invert:
            return ( arg / Tn )**( -1/( pn+0.5 ) )
        return Tn * arg**( -pn ) * np.sqrt( arg ) # Eqn. 12 & 11

    def setup(self, sigma8: float, z: float = 0) -> Any:
        r"""
        Setup the distribution by setting the parameters.

        Parameters
        ----------
        sigma8: float
            RMS value of fluctuations at 8 Mpc/h scale.
        z: float
            Redshift.
            
        """
        if sigma8 <= 0:
            raise CosmologyError("sigma8 must be positive")
        self.cosmology.sigma8 = sigma8
        self.cosmology.power_spectrum.normalize()

        sigma2Lin   = self.sigma2Linear( z )
        sigma2Log   = self.sigma2A( sigma2Lin )
        self.b2_log = sigma2Log / sigma2Lin

        self._prepareSpline() # TODO: use power spectrum re-scaling

        # compute the location, scale and shape parameters

        sigma2Box = self.sigma2Box( z )

        def shapeEquation(shape: float, r1: float) -> float:
            g1 = gamma( 1-shape )
            g2 = gamma( 1-shape*2 )
            g3 = gamma( 1-shape*3 )

            return r1 + ( g3 - 3*g1*g2 + 2*g1**3 ) / ( g2 - g1**2 )**1.5 # Eqn. 18

        r1    = self.skewnessA( sigma2Box )  # pearson's moment coefficient, Eqn. 11
        shape = newton( shapeEquation, -0.001, args = ( r1, ), )

        # NOTE: Eqn. 19 gives negative scale parameter. but scale parameter must be positive
        # and using the absolute value of shape parameter here. 
        g1, g2 = gamma( 1-shape ), gamma( 1-shape*2 )
        scale  = np.abs( shape ) * np.sqrt( sigma2Box ) * ( g2 - g1**2 )**( -0.5 ) # Eqn. 19

        loc    = self.averageA( sigma2Lin ) - scale * ( g1 - 1 ) / shape # Eqn. 20

        self.param = GenExtremeParameters( loc, scale, shape )

    @property
    def supportInterval(self) -> tuple:
        return ( -np.inf, self.param.loc - self.param.scale / self.param.shape ) 

    def pdf(self, arg: Any, z: float = None, sigma8: float = None, log_field: bool = True) -> Any:
        r"""
        Return the probability density function for the overdensity.

        Parameters
        ----------
        arg: array_like
            Argument to the distribution function. It could be the logarithmic overdensity, :math:`A=\ln(\delta+1)` 
            or the 'linear' overdensity :math:`\delta` (specified by `log_field` parameter).
        z: float, optional
            Redshift. Default value if 0.
        sigma8: float, optional
            RMS fluctuation value at 8 Mpc/h scale (:math:`\sigma_8` parameter). If not given, use the value stored 
            in the cosmology object, otherwise overwrite the value.
        log_field: bool, optional
            Tell whether the argument is the linear or logarithmic field.

        Returns
        -------
        y: arrya_like
            Value of probability density function.

        Examples
        --------

        """
        # setup:
        if ( sigma8 is not None ) or ( z is not None ) :
            sigma8 = self.cosmology.sigma8 if sigma8 is None else sigma8
            z      = 0 if z is None else z
            self.setup( sigma8, z )

        # linear field:
        if not log_field:
            argp1 = np.asfarray( arg ) + 1
            return self.pdf( np.log( argp1 ) ) / argp1

        # log filed:
        if not np.ndim( arg ):
            return self.pdf( [ arg ], log_field )[0]

        arg = np.asfarray( arg )
        sup = ( arg < self.supportInterval[1] )

        shape = self.param.shape
        if shape >= 0:
            raise DistributionError("shape must be negative")

        loc, scale = self.param.loc, self.param.scale

        y        = np.zeros_like( arg )
        y[ sup ] = ( 1 + ( arg[ sup ] - loc ) * shape / scale )**( -1/shape )
        y[ sup ] = y[ sup ]**( 1 + shape ) * np.exp( -y[ sup ] ) / scale
        return y


    

    









