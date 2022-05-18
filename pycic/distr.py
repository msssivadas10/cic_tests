#!/usr/bin/python3
r"""
Distributions
=============

One point probability distribution functions for density fluctuation fields.

"""
import numpy as np
import numpy.random as rnd
from pycic.cosmo import Cosmology, Constants
from itertools import product, repeat
from typing import Any, Tuple, Union 
from scipy.special import gamma, gammaln, zeta, erf
from scipy.optimize import newton


qsettings = {
                'ka': 1.0E-08,  # lower limit
                'kb': 1.0E+08,  # upper limit
                'n' : 10001,    # number of points for 1D
                'n3': 101,      # number of points for 3D
            }

class IntervalError(Exception):
    """
    Base class of exceptions used by interval objects.
    """
    ...

class Interval:
    """
    A class representing an interval.
    """
    __slots__ = 'a', 'b', 'lopen', 'ropen',

    def __init__(self, a: float, b:float, lopen: bool = False, ropen: bool = False) -> None:
        if np.ndim( a ) or np.ndim( b ):
            raise IntervalError("a and b must be scalars")
        elif b < a:
            raise IntervalError("b cannot be smaller than a")
        elif a == b:
            raise IntervalError("zero width interval")

        self.a, self.b         = a, b
        self.lopen, self.ropen = bool( lopen ), bool( ropen )

        if self.a == -np.inf:
            self.lopen = True
        if self.b == np.inf:
            self.ropen = True

    def __repr__(self) -> str:
        return "{}{},{}{}".format( 
                                    '(' if self.lopen else '[',
                                    self.a, self.b,
                                    ')' if self.ropen else ']',
                                 )
    
    def _points(self, n: int = 51, log: bool = False, base: float = 10.0) -> Any:
        if self.lopen and self.ropen:
            n += 2
        elif self.lopen or self.ropen:
            n += 1

        start, end = 0, n
        if self.lopen:
            start += 1
        if self.ropen:
            end   -= 1

        if log:
            loga = np.log( self.a ) / np.log( base )
            logb = np.log( self.b ) / np.log( base )
            return np.logspace( loga, logb, n, base = base )[ start:end ]
        return np.linspace( self.a, self.b, n )[ start:end ]

    def linspace(self, n: int = 51) -> Any:
        """
        Return n linearly spaced points in the interval.
        """
        return self._points( n, False, )

    def logspace(self, n: int = 51, base: float = 10.0) -> Any:
        """
        Return n logarithmic spaced points in the interval.
        """
        return self._points( n, True, base )

    def astuple(self) -> tuple:
        """
        Return the interval ends as a tuple.
        """
        return self.a, self.b

class SettingsError(Exception):
    """
    Base class of exceptions used for settings errors.
    """
    ...

# =============================================================================
# Base distribution class
# ============================================================================== 

class DistributionError(Exception):
    """
    Base class of exceptions used by distribution function objects.
    """
    ...

class Distribution:
    """
    Base class for distribution function objects.
    """
    __slots__ = 'attrs', 'b', 'z', 'cm', 'kn', 'zspace', 

    def pdf(x: Any, *args, **kwargs) -> Any:
        """
        Probability distribution function.
        """
        return NotImplemented
    
    def cdf(x: Any, *args, **kwargs) -> Any:
        """
        Cumulative density function.
        """
        return NotImplemented

    def __init__(self, b: float, z: float, cm: Cosmology, params: Union[list, dict], zspace: bool = False) -> None:
        self.attrs = {}

        if np.ndim( b ):
            raise TypeError("cellsize 'b' must be a number")
        elif not b > 0:
            raise ValueError("cellsize 'b' must be positive")
        
        if np.ndim( z ):
            raise TypeError("redshift 'z' must be a number")
        elif z < -1:
            raise ValueError("redshift 'z' must be greater than -1")
        
        if not isinstance( cm, Cosmology ):
            raise TypeError("'cm' must be a 'Cosmology' object")

        self.b, self.z, self.cm = b, z, cm
        self.zspace             = bool( zspace )

        self.kn = np.pi / self.b  # nyquist wavenumber

        if isinstance( params, list ):
            if False in map( lambda o: isinstance( o, str ), params ):
                raise DistributionError("params must be a 'list' of 'str'")
            self.attrs = dict( 
                                zip( 
                                        params, repeat( None, len( params ) ) 
                                   ) 
                             )
        elif isinstance( params, dict ):
            if False in map( lambda o: isinstance( o, str ), params.keys() ):
                raise DistributionError("keys of params must be 'str'")
            self.attrs = { **params }
        else:
            raise TypeError("params must be a 'list' or 'dict'")

    def zcfactor(self) -> float:
        """
        Redshift space correction factor.
        """
        bias = self.param( 'bias' )
        if bias is None:
            raise DistributionError("distribution is not parametrised")
        if self.zspace:
            beta = self.cm.fz( self.z ) / bias
            return ( 1 + beta * 2.0 / 3.0 + beta**2 / 5.0 ) * bias**2
        return 1.0

    def support(self) -> Interval:
        """
        Support interval for the distribution function.
        """
        return NotImplemented

    def mean(self) -> float:
        """
        Distribution specific mean value.
        """
        return NotImplemented

    def var(self) -> float:
        """
        Distribution specific variance.
        """
        return NotImplemented

    def skew(self) -> float:
        """
        Distribution specific skewness value.
        """
        return NotImplemented

    def parameters(self) -> dict:
        """
        Return a copy of the parameters dictionary.  
        """
        return self.attrs.copy()

    def param(self, key: str) -> Any:
        """ 
        Return the value of a parameter with name `key`.
        """
        if key in self.attrs.keys():
            return self.attrs[ key ]
        raise DistributionError("invalid parameter key: '{}'".format( key ))

    def parametrize(self, **kwargs) -> None:
        """
        Set parameter values to the distribution. Each key-value pair should be a 
        valid parameter name and its value.
        """
        for key, value in kwargs.items():
            if key not in self.attrs.keys():
                raise DistributionError("invalid parameter key: '{}'".format( key ))
            self.attrs[ key ] = value
        
    def f(self, x: Any, log: bool = False) -> Any:
        """
        Distribution function.
        """
        return NotImplemented

    def fstd(self, x: Any) -> Any:
        """
        Standard form of the distribution function.
        """
        return NotImplemented
    
    def fcount(self, n: Any, lamda: Any) -> Any:
        r"""
        Distribution function for the galaxy counts. By default, a Poisson 
        distribution is used with :math:`\lambda = (1+\delta_g) \mu`.

        Parameters
        ----------
        n: array_like
            Value of counts.
        lamda: array_like
            Rate parameter, :math:`\lambda`.

        Returns
        -------
        f: array_like
            Value of the distribution function, :math:`f(n; \delta_g, \mu)`.
            
        """
        return np.exp(
                        n * np.log( lamda ) - lamda - gammaln( n + 1 )
                     ) 

    def fcic(self, n: Any, mu: float, xa: float = -50.0, xb: float = 50.0, pts: int = 10001) -> Any:
        """
        Distribution function for galaxy count-in-cells.

        Parameters
        ----------
        n: array_like
            Counts in cell.
        bias: float
            Galaxy bias (linear).
        mu: float
            Average count.
        xa, xb: float
            Integration limits
        pts: int
            Number of points for integration.
        
        Returns
        -------
        y: array_like
            Value of count-in-cells distribution function.
        """
        if self.bias is None:
            raise DistributionError("distribution is not parametrised")

        x, h = np.linspace( -1.0 + 1.0E-08, xb, pts, retstep = True ) # delta_g

        lamda = mu * ( 1 + x )

        fx    = self.f( x / self.bias, log = False ) 
        fnx   = self.fcount( n, lamda[:,None] ) 
        y     = fx[:, None] * fnx
        y = ( 
                y[ :-1:2,: ].sum(0) + 4*y[ 1::2,: ].sum(0) + y[ 2::2,: ].sum(0) 
            ) * h / 3.0 / self.bias
        return y

    def __call__(self, x: Any) -> Any:
        return self.f(x)

    def linvar(self) -> float:
        """
        Linear variance in a cell.
        """
        ka, n = qsettings[ 'ka' ], qsettings[ 'n' ]

        k, dlnk    = np.linspace( 
                                    np.log( ka ),
                                    np.log( self.kn ),
                                    n,
                                    retstep = True
                                ) 
        k          = np.exp( k )

        # integration variable: ln(k)
        y = k**3 * self.cm.matterPowerSpectrum( k, self.z )
        I = ( 
                y[ :-1:2 ].sum(-1) + 4*y[ 1::2 ].sum(-1) + y[ 2::2 ].sum(-1) 
            ) * dlnk / 3.0 
        
        return I / 2.0 / np.pi**2

# ==============================================================================
# Specialized distribution classes
# ============================================================================== 

class GEV(Distribution):
    """
    Generalized extreem value distribution.
    """
    __slots__ = '_powerlaw', 'zspace', 'csettings', 'sigma8', 'bias'

    def pdf(x: Any, xi: float, mu: float = 0.0, sigma: float = 1.0) -> Any:
        """
        GEV distribution probability density function.
        """
        x = np.asfarray( x )
        y = np.zeros_like( x )

        if xi != 0:
            lim = mu - sigma / xi
            if xi > 0:
                support  = ( x > lim )
            else:
                support  = ( x < lim )
            t            = ( 1 + xi * ( x[ support ] - mu ) / sigma )**( -1 / xi )
            y[ support ] = t**( xi + 1 ) * np.exp( -t ) / sigma
            return y
        t = np.exp( -( x - mu ) / sigma )
        y = t * np.exp( -t ) / sigma
        return y

    def cdf(x: Any, xi: float, mu: float = 0.0, sigma: float = 1.0) -> Any:
        """
        GEV distribution cumulative density function.
        """
        x = np.asfarray( x )
        y = np.zeros_like( x )

        if xi != 0:
            lim = mu - sigma / xi
            if xi > 0:
                support = ( x > lim )
            else:
                support       = ( x < lim )
                y[ ~support ] = 1.0
            t            = ( 1 + xi * ( x[ support ] - mu ) / sigma )**( -1 / xi )
            y[ support ] = np.exp( -t ) 
            return y
        t = np.exp( -( x - mu ) / sigma )
        y = np.exp( -t )
        return y
    
    def __init__(self, b: float, z: float, cm: Cosmology, zspace: bool = False) -> None:
        super().__init__(b, z, cm, [ 'xi', 'mu', 'sigma' ], zspace)

        self.sigma8, self.bias = None, None

        self._powerlaw = ( None, None )

        self.csettings = {
                            'start'  : 0.6,
                            'ka'     : 0.5,
                            'kb'     : 0.7,
                            'npoints': 100_000,
                         }
        
        self.getContinuation() # get power spectrum continuation

    def mean(self) -> float:
        p = self.param( 'xi' ), self.param( 'mu' ), self.param( 'sigma' )
        if None in p:
            raise DistributionError("distribution is not parametrised")
        xi, mu, sigma = p

        if xi == 0:
            return mu + sigma * 0.577215664901532
        elif xi < 1:
            return mu + gamma( 1.0 - xi ) * sigma / xi
        return np.inf

    def var(self) -> float:
        p = self.param( 'xi' ), self.param( 'sigma' )
        if None in p:
            raise DistributionError("distribution is not parametrised")
        xi, sigma = p

        if xi == 0:
            return sigma**2 * np.pi**2 / 6.0
        elif 2 * xi < 1:
            g1, g2 = gamma( 1 - np.array([1, 2]) * xi )
            return sigma**2 * ( g2 - g1**2 ) / xi**2
        return np.inf
    
    def skew(self) -> float:
        xi = self.param( 'xi' )
        if xi is None:
            raise DistributionError("distribution is not parametrised")

        if xi == 0:
            return 12 * np.sqrt(6) * zeta(3) / np.pi**3
        elif 3 * xi < 1:
            xisign     = -1 if xi < 0 else 1
            g1, g2, g3 = gamma( 1 - np.array([1, 2]) * xi )
            return xisign * ( g3 - 3 * g2 * g1 + 2 * g1**3 ) / ( g2 - g1**2 )**1.5
        return np.inf

    def parametrize(self, sigma8: float, bias: float) -> None:
        """
        Parametrize the GEV distribution.
        """
        self.cm.normalize(
                            sigma8,
                            ka = qsettings[ 'ka' ],
                            kb = qsettings[ 'kb' ],
                            n  = qsettings[ 'n' ]
                         )
        
        vlin = self.linvar() # linear field variance
        
        # log field variance (fit) :
        u    = 0.73
        vlog = u * np.log( 1.0 + vlin / u ) 

        blog = vlog / vlin # log field bias^2

        vcic = self.measvar() * blog # measured cic variance

        # log field mean (fit) : 
        lamda = 0.65
        mlog  = -lamda * np.log( 1.0 + 0.5 * vlin / lamda )

        # log field skewness (fit) :
        a, b = -0.70, 1.25
        c, d = -0.26, 0.06
        np3  = self.cm.ns + 3
        slog = ( a * np3 + b ) * vcic**( -d - c * np.log( np3 ) )

        r = slog * np.sqrt( vcic ) # pearson moment coeficent

        def g(x: float, n: int) -> float:
            return gamma( 1.0 - n*x )

        def f(x: float, y: float) -> float:
            g1, g2, g3 = g(x, 1), g(x, 2), g(x, 3)
            return y + ( g3 - 3.0 * g1 * g2 + 2 * g1**3 ) / ( g2 - g1**2 )**1.5
        
        xi     = newton( f, r, args = (r, ) ) # shape

        g1, g2 = g(xi, 1), g(xi, 2)
        sigma  = np.sqrt( xi**2 * vcic / ( g2 - g1**2 ) ) # scale

        mu     = mlog - sigma * g1 / xi

        self.sigma8, self.bias = sigma8, bias

        print( vlin, vlog, vcic )
        return super().parametrize(
                                    xi     = xi,
                                    mu     = mu,
                                    sigma  = sigma,
                                  )
    
    def support(self) -> Interval:
        p = self.param( 'xi' ), self.param( 'mu' ), self.param( 'sigma' )
        if None in p:
            raise DistributionError("distribution is not parametrised")
        xi, mu, sigma = p

        lim = mu - sigma / xi
        if xi != 0:
            if xi > 0:
                return Interval( lim, np.inf )
            return Interval( -np.inf, lim )
        return Interval( -np.inf, np.inf )
    
    def linpower(self, k: Any) -> Any:
        """
        Linear matter power spectrum.
        """
        return self.cm.matterPowerSpectrum( k, self.z )
    
    def _measpowerInside(self, kx: Any, ky: Any, kz: Any) -> Any:
        r"""
        Measured cell power spectrum for :math:`\vert { \bf k } \vert < k_n`.
        """
        def fk(_kx: Any, _ky: Any, _kz: Any) -> Any:
            wk = (
                    np.sinc( _kx / 2.0 / self.kn )
                        * np.sinc( _ky / 2.0 / self.kn )
                        * np.sinc( _kz / 2.0 / self.kn )
                 )**2

            k  = np.sqrt( _kx**2 + _ky**2 + _kz**2 )
            pk = self.linpower( k )
            return pk * wk**2

        kx = np.asfarray( kx ) 
        ky = np.asfarray( ky ) 
        kz = np.asfarray( kz )

        retval = 0.0
        for nx, ny, nz in product( *repeat( range( 3 ), 3 ) ):
            if nx**2 + ny**2 + nz**2 >= 9:
                continue
            retval += fk( 
                            kx + 2 * nx * self.kn,
                            ky + 2 * ny * self.kn,
                            kz + 2 * nz * self.kn,
                        ) 
        
        return retval

    def _measpowerOutside(self, k: Any) -> Any:
        r"""
        Measured cell power spectrum for :math:`\vert { \bf k } \vert > k_n`.
        """
        a, b = self._powerlaw
        if a is None or b is None:
            raise DistributionError("power law continuation not found")
        return a * np.asfarray( k )**b

    def measpower(self, kx: Any, ky: Any, kz: Any) -> Any:
        """
        Measured power spectrum in a cell.
        """
        kpart = self.kn * self.csettings[ 'start' ]

        kx = np.asfarray( kx )
        ky = np.asfarray( ky )
        kz = np.asfarray( kz )

        retval = np.sqrt( kx**2 + ky**2 + kz**2 )
        mask   = ( retval < kpart )

        retval[ mask ]  = self._measpowerInside( 
                                                    kx[ mask ],
                                                    ky[ mask ],
                                                    kz[ mask ],
                                               )
        retval[ ~mask ] = self._measpowerOutside( retval[ ~mask ] )
        return retval

    def measvar(self) -> float:
        """ 
        Measured variance in a cell.
        """
        ka, n = qsettings[ 'ka' ], qsettings[ 'n3' ]

        x, dlnk = np.linspace( 
                                np.log( ka ),
                                np.log( self.kn ),
                                n,
                                retstep = True
                             ) 
        
        x       = np.exp( x ) 
        x, y, z = np.meshgrid( x, x, x )
        y          = x * y * z * self.measpower( x, y, z )
        
        del x, z

        for _ in range(3):
            y = y[ ...,:-1:2 ].sum(-1) + 4 * y[ ...,1::2 ].sum(-1) + y[ ...,2::2 ].sum(-1)
        
        return 8.0 * y * ( dlnk / np.pi / 6.0 )**3

    def getContinuation(self) -> None:
        """
        Get a power law continuation for measured power spectrum.
        """

        def sph2cart(r: Any, t: Any, p: Any) -> tuple:
            return (
                        r * np.sin( t ) * np.cos( p ),
                        r * np.sin( t ) * np.sin( p ),
                        r * np.cos( t )
                   )

        npts = self.csettings[ 'npoints' ]
        ka   = self.csettings[ 'ka' ] * self.kn
        kb   = self.csettings[ 'kb' ] * self.kn

        # generate n random k-vectors:
        lnk = rnd.uniform( np.log( ka ), np.log( kb ), npts )

        kx, ky, kz = sph2cart( 
                                np.exp( lnk ),
                                rnd.uniform( size = npts ) * np.pi,
                                rnd.uniform( size = npts ) * 2 * np.pi,
                             ) 
        
        lnp = np.log(
                        self._measpowerInside( kx, ky, kz )
                    )

        # getting the power-law fit:
        b, lna         = np.polyfit( lnk, lnp, deg = 1 )
        self._powerlaw = ( np.exp( lna ), b )

    def f(self, x: Any, log: bool = False) -> Any:
        p = self.param( 'xi' ), self.param( 'mu' ), self.param( 'sigma' )
        if None in p:
            raise DistributionError("distribution is not parametrised")
        xi, mu, sigma = p

        if log:
            return GEV.pdf( x, xi, mu, sigma ) # input is x := log( 1 + delta )

        x      = 1 + np.asfarray( x ) 
        y      = np.zeros_like( x )
        m      = ( x > 0 )
        y[ m ] = GEV.pdf( np.log( x[ m ] ), xi, mu, sigma ) / x[ m ]
        return y
    
    def fcic(self, n: Any, mu: float, xa: float = -50, xb: float = 50, pts: int = 10001) -> Any:
        return super().fcic(n, mu, xa, self.support().b, pts)

class Lognormal(Distribution):
    """
    Lognormal distribution.
    """        
    __slots__ = 'sigma8', 'bias' 

    def pdf(x: Any, mu: float = 0.0, sigma: float = 1.0) -> Any:
        """
        Lognormal probability density function.
        """
        if sigma <= 0.0:
            raise DistributionError("sigma must be positive")
        x = np.asfarray( x )
        y = np.zeros_like( x )

        support      = ( x > 0.0 )
        s            = ( np.log( x[ support ] ) - mu ) / sigma
        y[ support ] = np.exp( -0.5 * s**2 ) / x[ support ] / np.sqrt( 2 * np.pi ) / sigma
        return y

    def cdf(x: Any, mu: float = 0.0, sigma: float = 1.0) -> Any:
        """ 
        Lognormal cumulative distribution function.
        """
        if sigma <= 0.0:
            raise DistributionError("sigma must be positive")
        x = np.asfarray( x )
        y = np.zeros_like( x )

        support      = ( x > 0.0 )
        s            = ( np.log( x[ support ] ) - mu ) / sigma
        y[ support ] = 0.5 + 0.5 * erf( s / np.sqrt( 2 ) )
        return y

    def __init__(self, b: float, z: float, cm: Cosmology, zspace: bool = False) -> None:
        super().__init__(b, z, cm, [ 's' ], zspace)

        self.sigma8, self.bias = None, None

    def support(self) -> Interval:
        return NotImplemented
    
    def parametrize(self, sigma8: float, bias: float) -> None:
        if sigma8 <= 0 or bias <= 0:
            raise DistributionError("parameter should be positive")

        self.cm.normalize( sigma8 )

        delta_c = Constants.DELTA_C
        var     = delta_c**2 / ( ( bias - 1 ) * delta_c + 1 ) # get variance corresponding to bias

        # get radius correspond to this variance
        def f(x: float) -> float:
            return self.cm.nonlinearVariance( x, self.z ) - var

        r   = newton( f, 1 )    

        var = self.cm.nonlinearVariance(r, self.z)

        self.sigma8, self.bias = sigma8, bias
        return super().parametrize( 
                                        s = np.log( 1 + var )
                                  )

    def f(self, x: Any, log: bool = False) -> Any:
        s = self.param( 's' )
        if s is None:
            raise DistributionError("distribution is not parametrised")

        x = np.asfarray( x ) # x is log( delta + 1 )
        if log:
            y = np.exp( -0.5 * ( x / s + 0.5 * s )**2 ) / np.exp( x ) / s / np.sqrt( 2*np.pi )
            return y
        
        # x is delta
        x = x + 1
        m = ( x > 0 )
        y = np.zeros_like( x )
        
        y[ m ] = np.log( x[ m ] ) 
        y[ m ] = np.exp( -0.5 * ( y[ m ] / s + 0.5 * s )**2 ) / x[ m ] / s / np.sqrt( 2*np.pi )
        return y

    

        

    