#!/usr/bin/python3
r"""

Filters
=======

to do
"""

import numpy as np
import pycosmo.utils.settings as settings
from scipy.integrate import simpson
from typing import Any, Callable 
from abc import ABC, abstractmethod

#################################################################################
# Base filter function class
#################################################################################


class Filter(ABC):
    r"""
    Base class representing a smoothing filter in k-space. A filter function object must be a subclass of this. 
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

    def __call__(self, x: Any, j: int = 0) -> Any:
        return self.filter(x, j)

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
        
        def integrand(lnk: Any, r: Any, *args):
            k  = np.exp( lnk )
            kr = np.outer( r, k )
            return f( k, *args ) * self.filter( kr )**2

        r    = np.asfarray( r )
        args = ( r, *args )
        a, b = np.log( settings.ZERO ), np.log( settings.INF )
        out = _integrate( integrand, a, b, args = args )
        return out if np.ndim( r ) else out[0]

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
        
        def integrand(lnk: Any, r: Any, *args):
            k  = np.exp( lnk )
            kr = np.outer( r, k )
            return f( k, *args ) * k * self.filter( kr ) * self.filter( kr, 1 )

        args = ( r, *args )
        a, b = np.log( settings.ZERO ), np.log( settings.INF )
        y1   = 2 * _integrate( integrand, a, b, args = args )
        return y1 if np.ndim( r ) else y1[0]

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
        
        def integrand(lnk: Any, r: Any, *args):
            k  = np.exp( lnk )
            kr = np.outer( r, k )

            wfactor = self.filter( kr ) * self.filter( kr, 2 ) + self.filter( kr, 1 )**2
            return f( k, *args ) * k**2 * self.filter( kr ) * wfactor

        args = ( r, *args )
        a, b = np.log( settings.ZERO ), np.log( settings.INF )
        y2   = 2 * _integrate( integrand, a, b, args = args )
        return y2 if np.ndim( r ) else y2[0]


###################################################################################
# Most used filter funcions
###################################################################################

class Tophat(Filter):
    r"""
    Spherical tophat filter in k-space.

    .. math::
        w(x) = 3 \frac{ \sin(x) - x\cos(x) }{ x^3 }

    """
    
    def filter(self, x: Any, j: int = 0) -> Any:
        x = np.asfarray( x )

        if j == 0:
            return ( np.sin( x ) - x * np.cos( x ) ) * 3.0 / x**3 
        elif j == 1:
            return ( ( x**2 - 3.0 ) * np.sin( x ) + 3.0 * x * np.cos( x ) ) * 3.0 / x**4
        elif j == 2:
            return ( ( x**2 - 12.0 ) * x * np.cos( x ) - ( 5*x**2 - 12.0 ) * np.sin( x ) ) * 3.0 / x**5
        return NotImplemented

class Gaussian(Filter):
    r"""
    Gaussian filter in k-space.

    .. math::
        w(x) = e^{ -x^2 / 2 }
        
    """

    def filter(x: Any, j: int = 0) -> Any:
        x = np.asfarray( x )

        if j == 0:
            return np.exp( -0.5*x**2 )
        elif j == 1:
            return -x*np.exp( -0.5*x**2 )
        elif j == 2:
            return ( x**2 - 1 )*np.exp( -0.5*x**2 )
        return NotImplemented


class SharpK(Filter):
    r"""
    Sharp k filter in k-space. It is the Fourier transform of the spherical tophat filter 
    in real space. It's value is 0 for x > 0 and 1 otherwise.
        
    Note: derivative of the convolution is not implemented.
    """

    def filter(self, x: Any, j: int = 0) -> Any:
        x = np.asfarray( x )
        
        if j == 0:
            return np.where( x > 1.0, 0.0, 1.0 )
        return NotImplemented

    def convolution(self, f: Callable, r: Any, args: tuple = ()) -> Any:

        def integrand(lnk: Any, *args):
            k  = np.exp( lnk )
            return f( k, *args )

        r    = np.asfarray( r )
        a, b = np.log( settings.ZERO ), np.log( 1/r )
        out  = _integrate( integrand, a, b, args = args )
        return out if np.ndim( r ) else out[0]

    def dcdr(self, f: Callable, r: Any, args: tuple = ()) -> Any:
        raise NotImplementedError()

    def d2cdr2(self, f: Callable, r: Any, args: tuple = ()) -> Any:
        raise NotImplementedError()
        

available = ['tophat', 'gauss', 'sharpk']

def get(key: str) -> Filter:

    if key == 'tophat':
        return Tophat()
    if key in ['gauss', 'gaussian']:
        return Gaussian()
    if key in ['sharpk', 'sharp-k']:
        return SharpK()

    raise ValueError(f"cannor find filter function '{key}'") 
    

def j0convolution(f: Callable, r: Any, args: tuple = (), ) -> Any:
    r"""
    Compute the convolution with sinc function (spherical bessel function :math:`j_0`) filter.

    .. math::
        F(r) = \int_0^\infty f(k) \frac{ \sin(kr) }{ kr }

    Note: sinc function is highly oscillating for large arguments, it is exponentilly suppressed 
    at x = 100. i.e., an additional factor of :math:`\exp[(x/100)^2]` is introduced.

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
    
    def integrand(lnk: Any, r: Any, *args):
        k  = np.exp( lnk )
        kr = np.outer( r, k )
        return f( k, *args ) * np.sinc( kr / np.pi ) * np.exp( -(kr / 100)**2 )

    r    = np.asfarray( r )
    args = ( r, *args )
    a, b = np.log( settings.ZERO ), np.log( settings.INF )
    out = _integrate( integrand, a, b, args = args )
    return out if np.ndim( r ) else out[0]


def _integrate(f: Callable, a: Any, b: Any, args: tuple = ()) -> Any:
    """
    Integrate a function using simpsons rule.
    """

    subdiv = settings.DEFAULT_SUBDIV

    pts  = int( 2**subdiv + 1 )
    x, h = np.linspace( a, b, pts, retstep = True, axis = -1 )
    y    = f( x, *args )
    return simpson(y, dx = h, axis = -1)