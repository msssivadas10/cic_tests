from abc import ABC, abstractmethod
from typing import Any, Type, Union

####################################################################################################

class CosmologyError( Exception ):
    """
    Base class of exceptions used by cosmology objects.
    """
    ...

class CosmologyData( ABC ):
    """
    Object to store a cosmology model.
    """
    __slots__ = (
                    '_Om0', '_Ob0', '_Oc0', '_Onu0', '_Ode0', '_Ok0', '_Oph0', '_Nnu', '_Mnu', 
                    '_h', '_sigma8', '_ns', '_flat',  
                )

    def __init__(self, flat: bool = True, h: float = None, Om0: float = None, Ob0: float = None, Ode0: float = None, Onu0: float = 0, Nnu: float = None, sigma8: float = None, ns: float = None, w0: float = -1.0, wa: float = 0.0) -> None:
        # hubble parameter, h:
        if h is None:
            raise CosmologyError( "required parameter: 'h'" )
        elif not ( h > 0.0 ):
            raise CosmologyError( "hubble parameter must be positive" )
        self._h = h

        # rms matter fluctuation, sigma_8:
        if sigma8 is None:
            raise CosmologyError( "required parameter: 'sigma8'" )
        elif not ( sigma8 > 0.0 ):
            raise CosmologyError( "sigma_8 parameter must be positive" )
        self._sigma8 = sigma8

        # power spectrum index:
        if ns is None:
            raise CosmologyError( "required parameter: 'ns'" )
        self._ns = ns


        if Om0 is None:
            raise CosmologyError( "required parameter: 'Om0'" )
        elif Ob0 is None:
            raise CosmologyError( "required parameter: 'Ob0'" )
        elif not flat and Ode0 is None:
            raise CosmologyError( "required parameter: 'Ode0'" ) 

        # initialize components in the universe:
        
        

    