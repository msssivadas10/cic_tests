from typing import Any, Callable
from pycosmo2.utils.gaussrules import legendrerule
from itertools import repeat
import warnings
import numpy as np

class NumericError(Exception):
    """ 
    Base class of exceptions raised by `numeric` functions.
    """
    ...

class NumericWarning(Warning):
    """ 
    Base class of warnings raised by `numeric` functions.
    """
    ...

def erf(x: Any) -> Any:
    r"""
    Return an approximate value for the error function. This approximation has a maximum accuracy 
    of :math:`1.7 \times 10^{-7}`. Argument `x` must be real.    
    """
    x = np.asfarray( x )
    t = 1 / ( 1 + 0.3275911*np.abs( x ) )
    y = 1 - ( 
                0.254829592*t 
                    - 0.284496736*t**2 
                    + 1.421413741*t**3 
                    - 1.453152027*t**4 
                    + 1.061405429*t**5 
            ) * np.exp( -x**2 )
    return np.where( x < 0, -y, y )

def erfc(x: Any) -> Any:
    r"""
    Compute the approximate value of :math:`{\rm erfc}(x) = 1 - {\rm erf}(x)` for real argument.
    """
    return 1 - erf( x )

def integrate1(f: Callable, a: Any, b: Any, args: tuple = (), subdiv: int = 20) -> Any:
    r"""
    Compute the integral of a real valued function :math:`f(x)` using adaptive simpsons rule.

    Parameters
    ----------
    f: callable
        Function to integrate. Must have the signature `f(x, *args)`.
    a, b: array_like
        Lower and upper limits of integration. Both `a` and `b` should have the same dimension.
    args: tuple, optional  
        Other arguments to pass to the function.
    subdiv: int, optional
        Number of subdivisions of the integration domain. This correspond to :math:`2^{subdiv}+1` 
        points. Default is 20.

    Returns
    -------
    result: array_like
        Computed integral of the function.

    Examples
    --------
    For a power law function, 

    .. math::
        \int_0^1 x^a {\rm d}x = \left[ \frac{x^{a+1}}{a+1} \right]_0^1 = \frac{1}{a+1}


    """
    pts  = int( 2**subdiv + 1 )
    x, h = np.linspace( a, b, pts, retstep = True, axis = -1 )
    y    = f( x, *args )
    return ( y[...,:-1:2].sum(-1) + 4*y[...,1::2].sum(-1) + y[...,2::2].sum(-1) ) * h/3

def integrate2(f: Callable, a: Any, b: Any, args: tuple = (), eps: float = 1e-06, n: int = 64, no_warnings: bool = True) -> Any:
    r"""
    Compute the integral of a function using Gauss-Konrod quadrature rule.

    Parameters
    ----------
    f: callable
        Function to integrate. Must have the signature `f(x, *args)`.
    a, b: array_like
        Lower and upper limits of integration. Both `a` and `b` should have the same dimension.
    args: tuple, optional  
        Other arguments to pass to the function.
    eps: float, optional
        Tolerance (default: 1E-6).
    n: int, optional. 
        Order of the integration rule or number of points to use (default: 64).
    
    Returns
    -------
    result: array_like
        Computed integral of the function.

    Examples
    --------
    For a power law function, 

    .. math::
        \int_0^1 x^a {\rm d}x = \left[ \frac{x^{a+1}}{a+1} \right]_0^1 = \frac{1}{a+1}

    >>> a = np.array([2.0, 4.0, 6.0])
    >>> 1/(a+1) # eaxct values
    array([0.33333333, 0.2       , 0.14285714])
    >>> integrate2(lambda x: x**a, 0, 1) 
    array([0.33333333, 0.2       , 0.14285714])

    Limits can also be arrays (then, both should have the same dimension)

    >>> a, b = np.asfarray([0, 1]), np.asfarray([1, 2])
    >>> b**7/7 - a**7/7 # eaxact values
    array([ 0.14285714, 18.14285714])
    >>> integrate2( lambda x: x**6, a, b )
    array([ 0.14285714, 18.14285714])

    """
    
    node, wg, wk = legendrerule( n )

    a, b = np.asfarray( a ), np.asfarray( b )
    if np.ndim( a ) != np.ndim( b ):
        if np.ndim( a ) == 0:
            a = a * np.ones_like( b )
        elif np.ndim( b ) == 0:
            b = b * np.ones_like( a )
        else:
            raise NumericError("a and b should have same dimension")

    m = 0.5*( b - a )
    c = a + m
    if np.ndim( m ):
        x = m[:,None] * node + c[:,None]
    else:
        x = m * node + c
    y = f( x, *args )

    Ig = m * np.dot( y[..., 1:-1:2], wg )
    Ik = m * np.dot( y, wk )

    error = np.abs( Ig - Ik )
    if np.any( error > eps ) and not no_warnings:
        if np.ndim( error ):
            warnings.warn("atleast some integrals are not converged to specified accuracy", NumericWarning)
        else:
            warnings.warn("integral is not converged to specified accuracy", NumericWarning)
    return Ik
        

def solve(f: Callable, a: Any, b: Any, args: tuple = (), tol: float = 1e-06) -> Any:
    r"""
    Solve for the root of a function using Ridders' method.

    Parameters
    ----------
    f: callable
        Function to solve for the root. Must have the signature `f(x, *args)`.
    a, b: array_like
        Lower and upper bounds for the root. Both `a` and `b` should have the same dimension.
    args: tuple, optional  
        Other arguments to pass to the function.
    tol: float, optional
        Tolerance (default: 1E-6).

    Returns
    -------
    r: array_like
        Root of the function, calculated up to specfied accuracy.

    Examples
    --------

    Roots of the function :math:`f(x) = \cos(\pi x)` are the half-integers.

    >>> solve( lambda x: np.cos( np.pi * x ), [0., 1.], [1., 2.] )
    array([0.50000048, 1.50000048])
    >>> solve( lambda x: np.cos( np.pi * x ), 0., 1. )
    0.5000004768371582

    """

    def recurse(f: Callable, a: Any, fa: Any, b: Any, fb: Any, args: tuple, tol: float, conv: Any, j: int = 0) -> Any:
        if j > 1000:
            raise NumericError("root does not converge after maximum iterations")

        h = b - a
        c = a + h * 0.5

        conv = conv | ( np.abs( h ) < tol )
        if np.min( conv ):
            return c

        fc = f( c, *args )

        delta = ( c - a ) * fc / np.sqrt( fc*fc - fa*fb )
        d     = np.where( fa < fb , c - delta, c + delta)
        fd    = f( d, *args )

        # [choice] : [0] conv | [1] a, b = c, d | [2] a, b = a, d | [3] a, b = d, b
        choice = np.where( 
                            conv,
                            0,
                            np.where(
                                        fc*fd < 0,
                                        1,
                                        np.where(
                                                    fa*fd < 0,
                                                    2,
                                                    3
                                                )
                                    )

                         )

        a  = np.where( 
                        ( choice == 0 ) | ( choice == 2 ), 
                        a,
                        np.where(
                                    choice == 1,
                                    c,
                                    d
                                )  
                     )
        fa = np.where( 
                        ( choice == 0 ) | ( choice == 2 ), 
                        fa,
                        np.where(
                                    choice == 1,
                                    fc,
                                    fd
                                )  
                     )

        b  = np.where( 
                        ( choice == 0 ) | ( choice == 3 ), 
                        b,
                        d 
                     )
        fb = np.where( 
                        ( choice == 0 ) | ( choice == 3 ), 
                        fb,
                        fd  
                     )

        return recurse( f, a, fa, b, fb, args, tol, conv, j+1 )

    if np.ndim( a ) != np.ndim( b ):
        raise NumericError("a and b should have same dimension")

    a, b   = np.asfarray( a ).copy(), np.asfarray( b ).copy()
    fa, fb = f( a, *args ), f( b, *args )

    if np.any( fa * fb >= 0 ):
        raise NumericError("interval does not contain a root")

    conv = np.zeros_like( fa, 'bool' )
    return recurse( f, a, fa, b, fb, args, tol, conv, 0 )



if __name__ == '__main__':
    pass