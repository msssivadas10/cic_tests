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
                    '_h', '_sigma8', '_ns', '_flat',  '_Tcmb0', '_w0', '_wa'
                )

    def __init__(self, flat: bool = True, h: float = None, Om0: float = None, Ob0: float = None, Ode0: float = None, Onu0: float = 0, Nnu: float = None, sigma8: float = None, ns: float = None, Tcmb0: float = 2.725, w0: float = -1.0, wa: float = 0.0) -> None:
        
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

        # cmb temperature:
        if not ( Tcmb0 > 0.0 ):
            raise CosmologyError( "cmb temperature cannot be zero or negative" )
        self._Tcmb0 = Tcmb0


        # initialize components in the universe:
        if Om0 is None:
            raise CosmologyError( "required parameter: 'Om0'" )
        elif Ob0 is None:
            raise CosmologyError( "required parameter: 'Ob0'" )
        elif not flat and Ode0 is None:
            raise CosmologyError( "required parameter: 'Ode0'" )         
        
        if Om0 < 0.0:
            raise CosmologyError( "matter density must be positive" )
        
        if Ob0 < 0.0:
            raise CosmologyError( "baryon density must be positive" )
        elif Ob0 > Om0:
            raise CosmologyError( "baryon density cannot exceed total matter density" )
        self._Om0, self._Ob0 = Om0, Ob0

        if Onu0:
            if Onu0 < 0.0:
                raise CosmologyError( "massive neutrino density must be positive" )
            elif Nnu is None:
                raise CosmologyError( "required parameter: 'Nnu'" )
            elif not ( Nnu > 0.0 ):
                raise CosmologyError( "number of nuetrinos must be positive" )

            self._Nnu  = Nnu
            self._Mnu  =  91.5 * Onu0 * h**2 / Nnu 
            self._Onu0 = Onu0
        else:
            self._Onu0, self._Nnu, self._Mnu = 0.0, 0.0, 0.0

        totalMatter = self._Ob0 + self._Onu0
        if totalMatter > Om0:
            raise CosmologyError( "total density of baryon and neutrino cannot exceed matter density" )
        self._Oc0 = Om0 - totalMatter

        if flat:
            if Om0 > 1.0:
                raise CosmologyError( "matter density cannot exceed 1 for flat universe" )
            
            self._Ode0 = 1.0 - Om0
            self._Ok0  = 0.0
        else:
            if Ode0 < 0.0:
                raise CosmologyError( "dark-energy density cannot be negative" )
            
            totalDensity          = Om0 + Ode0
            self._Ode0, self._Ok0 = Ode0, 1.0 - totalDensity

        self.flat = not self._Ok0


        # dark energy parametrizations
        self._w0, self._wa = w0, wa    

    