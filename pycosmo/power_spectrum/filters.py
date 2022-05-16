from typing import Any, Callable 
import numpy as np
import pycosmo.utils.numeric as numeric
import pycosmo.utils.settings as settings
import pycosmo._bases as base

class Filter(base.Filter):
    r"""
    Base class representing a smoothing filter in k-space. A filter function object must be a subclass of this. 
    """

    def convolution(self, f: Callable, r: Any, args: tuple = (), ) -> Any:
        
        def integrand(lnk: Any, r: Any, *args):
            k  = np.exp( lnk )
            kr = np.outer( r, k )
            return f( k, *args ) * self.filter( kr )**2

        r    = np.asfarray( r )
        args = ( r, *args )
        a, b = np.log( settings.ZERO ), np.log( settings.INF )
        out = numeric.integrate1( integrand, a, b, args = args, subdiv = settings.DEFAULT_SUBDIV )
        return out if np.ndim( r ) else out[0]

    def dcdr(self, f: Callable, r: Any, args: tuple = (), ) -> Any:
        
        def integrand(lnk: Any, r: Any, *args):
            k  = np.exp( lnk )
            kr = np.outer( r, k )
            return f( k, *args ) * k * self.filter( kr ) * self.filter( kr, 1 )

        args = ( r, *args )
        a, b = np.log( settings.ZERO ), np.log( settings.INF )
        y1   = 2 * numeric.integrate1( integrand, a, b, args = args, subdiv = settings.DEFAULT_SUBDIV )
        return y1 if np.ndim( r ) else y1[0]

    def d2cdr2(self, f: Callable, r: Any, args: tuple = (), ) -> Any:
        
        def integrand(lnk: Any, r: Any, *args):
            k  = np.exp( lnk )
            kr = np.outer( r, k )

            wfactor = self.filter( kr ) * self.filter( kr, 2 ) + self.filter( kr, 1 )**2
            return f( k, *args ) * k**2 * self.filter( kr ) * wfactor

        args = ( r, *args )
        a, b = np.log( settings.ZERO ), np.log( settings.INF )
        y2   = 2 * numeric.integrate1( integrand, a, b, args = args, subdiv = settings.DEFAULT_SUBDIV )
        return y2 if np.ndim( r ) else y2[0]


##################################################################################################


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
        out = numeric.integrate1( integrand, a, b, args = args, subdiv = settings.DEFAULT_SUBDIV )
        return out if np.ndim( r ) else out[0]

    def dcdr(self, f: Callable, r: Any, args: tuple = ()) -> Any:
        raise NotImplementedError()

    def d2cdr2(self, f: Callable, r: Any, args: tuple = ()) -> Any:
        raise NotImplementedError()


filters = {
                'tophat': Tophat(),
                'gauss' : Gaussian(),
                'sharpk': SharpK(),
          }

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
    out = numeric.integrate1( integrand, a, b, args = args, subdiv = settings.DEFAULT_SUBDIV )
    return out if np.ndim( r ) else out[0]