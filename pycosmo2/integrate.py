import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Callable 

class IntegrationError( Exception ):
    """
    Base class of exceptions used by functions in `integrate` module.
    """
    ...

class IntegrationWarning( Warning ):
    """
    Base class of warnigs used by functions in `integrate` module.
    """
    ...

class Integrator( ABC ):

    @abstractmethod
    def __call__(self, f: Callable, a: Any, b: Any, args: tuple = ()) -> Any:
        """
        Integrate the function f(x) from a to b.
        """
        ...

# simpson rule:

class SimpsonIntegrator( Integrator ):
    """
    Simpson's rule integration.
    """

    @staticmethod
    def simps(f: Callable, a: Any, b: Any, args: tuple = (), pts: int = 1001) -> Any:
        """
        Simpsons rule integration.
        """
        if ( pts < 2 ) or not ( pts & 1 ):
            raise ValueError("number of points must be an odd number greater than 3: '{}'".format( pts ))

        x, dx = np.linspace( a, b, pts, retstep = True )
        y     = f( x, *args )
        
        retval = (
                    y[ ..., :-1:2 ].sum( -1 ) 
                        + y[ ..., 1::2 ].sum( -1 ) * 4.0
                        + y[ ..., 2::2 ].sum( -1 )
                ) / 3.0
        return retval * dx

    @staticmethod
    def adatpiveSimps(f: Callable, a: Any, b: Any, args: tuple = (), reltol: float = 1E-06, abstol: float = 1E-06) -> Any:
        """
        Integrate a function using the adaptive simpson rule.
        """
        def _simps_3pt(f: Callable, a: Any, fa: Any, b: Any, fb: Any, args: tuple) -> tuple:
            h  = b - a
            m  = a + 0.5*h 
            fm = f( m, *args )
            return m, fm, ( fa + 4*fm + fb ) * h / 6.0

        def _simps_rec(
                        f: Callable, 
                        a: Any, fa: Any, 
                        b: Any, fb: Any, 
                        m: Any, fm: Any, 
                        prev_sum: Any, 
                        args: tuple, 
                        reltol: float, abstol: float, 
                        rec: int = 0,
                    ) -> Any:
            lm, flm, left  = _simps_3pt( f, a, fa, m, fm, args )
            rm, frm, right = _simps_3pt( f, m, fm, b, fb, args )

            reltol, abstol = max( reltol, 1E-08 ), max( abstol, 1E-08 )

            current_sum = left + right
            if np.allclose( current_sum, prev_sum, rtol = reltol, atol = abstol ):
                return current_sum
            
            return (
                        _simps_rec( f, a, fa, m, fm, lm, flm, left,  args, 0.5*reltol, 0.5*abstol, rec+1 ) + 
                        _simps_rec( f, m, fm, b, fb, rm, frm, right, args, 0.5*reltol, 0.5*abstol, rec+1 )
                   )
        
        fa, fb = f( a, *args ), f( b, *args )
        
        m, fm, fsum = _simps_3pt( f, a, fa, b, fb, args )
        return _simps_rec( f, a, fa, b, fb, m, fm, fsum, args, reltol, abstol, 0 )

    __slots__ = 'adaptive', 'pts', 'reltol', 'abstol', 'limit'

    def __init__(self, adaptive: bool = True, pts: int = 10001, reltol: float = 1E-06, abstol: float = 1E-06) -> None:
        self.adaptive = adaptive
        self.reltol   = reltol
        self.abstol   = abstol
        self.pts      = pts
    
    def __call__(self, f: Callable, a: Any, b: Any, args: tuple = (), ) -> Any:
        if self.adaptive:
            return self.adatpiveSimps( f, a, b, args, self.reltol, self.abstol )
        return self.simps( f, a, b, args, self.pts )

# double exponential transforms:

class DetIntegrator( Integrator ):
    """
    Double exponential transformation based integrator.
    """

    @staticmethod
    def detIntegral_finite(f: Callable, a: float = -1.0, b: float = 1.0, args: tuple = (), eps: float = 1E-6, limit: int = 50) -> Any:
        """
        Double exponential transformation based integral of a function over the finite interval [a, b]. 
        The integrand should have a singularity at atleast one of the endpoints, for this integral to 
        work well. 
        """
        def gt(t: Any) -> Any:
            u1, u2 = np.sinh( t ) * np.pi*0.5, np.cosh( t ) * np.pi*0.5
            g      = np.tanh( u1 )
            dgdt   = u2 / np.cosh( u1 )**2
            return g, dgdt
        
        def costN(N: int, h: float, f: Callable, args: tuple, transform: tuple):
            m, c    = transform
            g, dgdt = gt( N*h )
            return np.abs( m * dgdt * f( m*g + c , *args ) )

        def solveN(h: float, eps: float, f: Callable, args: tuple, transform: tuple) -> int:
            Nlimit = 1000_000
            for N in range( 1, Nlimit ):
                if np.all( costN( N, h, f, args, transform ) < eps ):
                    return N
            raise IntegrationError( f"cannot find N between 1 and { Nlimit }" )

        def detIntegral(f: Callable, h: float, args: tuple, transform: tuple) -> Any:
            N = solveN( h, eps, f, args, transform )
            t = np.arange( -N, N+1 )*h

            m, c    = transform
            g, dgdt = gt( t )
            return m * h * np.trapz( dgdt * f( m*g + c, *args ) )
        
        m = 0.5 * ( b - a )
        c = a + m
        h = 0.5
        y = detIntegral( f, h, args, ( m, c ) )
        for step in range( 1, limit ):
            h  = h / 2.0
            y1 = detIntegral( f, h, args, ( m, c ) )
            if np.allclose( y, y1, rtol = eps ):
                # print( step )
                return y1
            y  = y1

        raise IntegrationError( f"integral failed to converge after { limit } steps" )

    @staticmethod
    def detIntegral_ainf(f: Callable, a: float = 0.0, args: tuple = (), eps: float = 1E-6, limit: int = 50) -> Any:
        r"""
        Double exponential transformation based integral of a function over the interval :math:`[a, \infty]`. 
        The integrand should have a singularity at atleast one of the endpoints, for this integral to 
        work well. 
        """
        def gt(t: Any) -> Any:
            u1, u2 = np.sinh( t ) * np.pi*0.5, np.cosh( t ) * np.pi*0.5
            g      = np.exp( u1 )
            dgdt   = u2 * g
            return g, dgdt
        
        def costN(N: int, h: float, f: Callable, args: tuple, a: Any):
            g, dgdt = gt( N*h )
            return np.abs( dgdt * f( g + a , *args ) )

        def solveN(h: float, eps: float, f: Callable, args: tuple, a: Any) -> int:
            Nlimit = 1000_000
            for N in range( 1, Nlimit ):
                if np.all( costN( N, h, f, args, a ) < eps ):
                    return N
            raise IntegrationError( f"cannot find N between 1 and { Nlimit }" )

        def detIntegral(f: Callable, h: float, args: tuple, a: Any) -> Any:
            N = solveN( h, eps, f, args, a )
            t = np.arange( -N, N+1 )*h

            g, dgdt = gt( t )
            return h * np.trapz( dgdt * f( g + a, *args ) )
        
        h = 0.5
        y = detIntegral( f, h, args, a )
        for step in range( 1, limit ):
            h  = h / 2.0
            y1 = detIntegral( f, h, args, a )
            if np.allclose( y, y1, rtol = eps ):
                # print( step )
                return y1
            y  = y1

        raise IntegrationError( f"integral failed to converge after { limit } steps" )

    @staticmethod
    def detIntegral_inf(f: Callable, args: tuple = (), eps: float = 1E-6, limit: int = 50) -> Any:
        """
        Double exponential transformation based integral of a function over the entire real line. 
        The integrand should have a singularity at atleast one of the endpoints, for this integral to 
        work well. 
        """
        def gt(t: Any) -> Any:
            u1, u2 = np.sinh( t ) * np.pi*0.5, np.cosh( t ) * np.pi*0.5
            g      = np.sinh( u1 )
            dgdt   = u2 * np.cosh( u1 )
            return g, dgdt
        
        def costN(N: int, h: float, f: Callable, args: tuple):
            g, dgdt = gt( N*h )
            return np.abs( dgdt * f( g , *args ) )

        def solveN(h: float, eps: float, f: Callable, args: tuple) -> int:
            Nlimit = 1000_000
            for N in range( 1, Nlimit ):
                if np.all( costN( N, h, f, args ) < eps ):
                    return N
            raise IntegrationError( f"cannot find N between 1 and { Nlimit }" )

        def detIntegral(f: Callable, h: float, args: tuple) -> Any:
            N = solveN( h, eps, f, args )
            t = np.arange( -N, N+1 )*h

            g, dgdt = gt( t )
            return h * np.trapz( dgdt * f( g, *args ) )
        
        h = 0.5
        y = detIntegral( f, h, args )
        for step in range( 1, limit ):
            h  = h / 2.0
            y1 = detIntegral( f, h, args )
            if np.allclose( y, y1, rtol = eps ):
                # print( step )
                return y1
            y  = y1

        raise IntegrationError( f"integral failed to converge after { limit } steps" )

    @staticmethod
    def detIntegral_fourier(f: Callable, omega: Any = 1.0, args: tuple = (), eps: float = 1E-5, limit: int = 50) -> Any:
        r"""
        Double exponential transformation based method to do Fourier integrations. i.e., integrals of the 
        form 

        .. math ::

            F(\omega) = \int_0^\infty f(x) \sin( \omega x )

        where the function :math:`f(x)` is non-oscillatory and slowly decaying.
        """
        def gt(t: Any, M: Any, omega: Any) -> Any:
            M_omega = M / omega 
            u1, u2 = 2*np.pi * np.sinh( t ), 2*np.pi * np.cosh( t )
            g      = np.outer( M_omega, t / ( 1.0 - np.exp( -u1 ) ) )
            dgdt   = np.outer( M_omega, np.exp( u1 ) * ( np.exp( u1 ) - u2 * t - 1.0 ) / ( np.exp( u1 ) - 1.0 )**2 )
            return g, dgdt
        
        def costN(N: int, h: float, f: Callable, args: tuple, omega: Any):
            M       = np.pi / h
            g, dgdt = gt( N*h, M, omega )
            return np.abs( dgdt * f( g , *args ) )

        def solveN(h: float, eps: float, f: Callable, args: tuple, omega: Any) -> int:
            Nlimit = 1000_000
            for N in range( 1, Nlimit ):
                if np.all( costN( N, h, f, args, omega ) < eps ):
                    return N
            raise IntegrationError( f"cannot find N between 1 and { Nlimit }" )

        def detIntegral(f: Callable, h: float, args: tuple, omega: Any) -> Any:
            N = solveN( h, eps, f, args, omega )
            t = np.arange( -N, N+1 )*h
            t =  t[ t != 0.0 ]
            M = np.pi / h

            g, dgdt = gt( t, M, omega )
            return h * np.trapz( dgdt * f( g, *args ) )

        h = 0.5
        y = detIntegral( f, h, args, omega )
        for step in range( 1, limit ):
            h  = h / 2.0
            y1 = detIntegral( f, h, args, omega )
            # print( y, y1 )
            if np.allclose( y, y1, rtol = eps, atol = eps ):
                # print( step )
                return y1
            y  = y1

        raise IntegrationError( f"integral failed to converge after { limit } steps" )

    __slots__ = 'eps', 'limit', 'fourier'

    def __init__(self, eps: float = 1E-06, limit: int = 50, fourier: bool = False) -> None:
        self.eps     = eps
        self.limit   = limit
        self.fourier = fourier

    def __call__(self, f: Callable, a: Any, b: Any, args: tuple = ()) -> Any:
        if b == np.inf:
            if a == -np.inf:
                return self.detIntegral_inf( f, args, self.eps, self.limit )
            return self.detIntegral_ainf( f, a, args, self.eps, self.limit )
        return self.detIntegral_finite( f, a, b, args, self.eps, self.limit )

    def sineIntegral(self, f: Callable, args: tuple = (), omega: Any = 1.0) -> Any:
        """
        Fourier sine integral.
        """
        return self.detIntegral_fourier( f, omega, args, self.eps, self.limit )

# clenshaw-curtis integration :

# TODO: