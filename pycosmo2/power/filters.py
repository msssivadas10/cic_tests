from typing import Any, Callable 
from abc import ABC, abstractmethod
import numpy as np
import pycosmo2.utils.numeric as numeric
import pycosmo2.utils.settings as settings

def tophat(x: Any, j: int = 0) -> Any:
    x = np.asfarray( x )
    if j == 0:
        return ( np.sin( x ) - x * np.cos( x ) ) * 3.0 / x**3 
    elif j == 1:
        return ( ( x**2 - 3.0 ) * np.sin( x ) + 3.0 * x * np.cos( x ) ) * 3.0 / x**4
    elif j == 2:
        return ( ( x**2 - 12.0 ) * x * np.cos( x ) - ( 5*x**2 - 12.0 ) * np.sin( x ) ) * 3.0 / x**5
    return NotImplemented

def gauss(x: Any, j: int = 0) -> Any:
    x = np.asfarray( x )
    if j == 0:
        return np.exp( -0.5*x**2 )
    elif j == 1:
        return -x*np.exp( -0.5*x**2 )
    elif j == 2:
        return ( x**2 - 1 )*np.exp( -0.5*x**2 )
    return NotImplemented

def sharpk(x: Any, j: int = 0) -> Any:
    raise NotImplementedError("function not implemented!")


#####################################################################################################

class Filter(ABC):
    
    @abstractmethod
    def w(self, x: Any, j: int = 0) -> Any:
        ...

    def smoothIntegral(self, f: Callable, r: Any, **kwargs: Any) -> Any:
        ...

    def smoothIntegral_firstLogDerivative(self, f: Callable, r: Any, **kwargs: Any) -> Any:
        ...

    def smoothIntegral_secondLogDerivative(self, f: Callable, r: Any, **kwargs: Any) -> Any:
        ...