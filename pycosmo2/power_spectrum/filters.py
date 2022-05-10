from typing import Any, Callable 
from abc import ABC, abstractmethod
from scipy.special import erfc
import numpy as np
import pycosmo2.utils.numeric as numeric
import pycosmo2.utils.settings as settings

class Filter(ABC):
    
    @abstractmethod
    def filter(self, x: Any, j: int = 0) -> Any:
        ...

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
          }

def j0convolution(f: Callable, r: Any, args: tuple = (), ) -> Any:
    
    def integrand(lnk: Any, r: Any, *args):
        k  = np.exp( lnk )
        kr = np.outer( r, k )
        return f( k, *args ) * np.sinc( kr / np.pi ) * np.exp( -(kr / 100)**2 )

    r    = np.asfarray( r )
    args = ( r, *args )
    a, b = np.log( settings.ZERO ), np.log( settings.INF )
    out = numeric.integrate1( integrand, a, b, args = args, subdiv = settings.DEFAULT_SUBDIV )
    return out if np.ndim( r ) else out[0]