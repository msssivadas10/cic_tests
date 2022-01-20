#!/usr/bin/python3
from typing import Any, Callable, Union
import numpy as np 
import warnings


class CatalogError(Exception):
    """
    Exceptions used by catalog objects.
    """
    ...

class Catalog:
    r"""
    An object to hold details of galaxy/halo objects (catalog). 

    Parameters
    ----------
    pos: array_like
        Object positions as a list/array of 3D vectors.
    vel: array_like, optional
        Object velocity as list/array of 3D vectors. This should correspond to the 
        peculiar velocity of the objects with with respect to the conformal time, 
        :math:`{\rm d}t / a(t)`.
    z: float. optional
        Redshift property of the catalog.
    space: str, optional
        Which space the position coordinates are specified. It has two possible values, 
        `real` for real space and `z` for redshift space.
    coord_sys: str, optional
        Which coordinate system to use. Only `cart` (for cartetian system) is currently 
        available.
    **prop: , optional
        Other properties of the objects as name-value pairs. 

    
    """
    __slots__ = (
                    'pos', 'vel', 'prop', 'redshift', 'Hz'
                )
    
    def __init__(self, pos: Any, vel: Any = ..., z: float = ..., space: str = "real", coord_sys: str = "cart", **prop) -> None:
        pos = np.asarray(pos)
        if pos.shape[1] != 3:
            raise CatalogError("data should correspond to 3 dimensionsss")
        self.pos      = pos
        
        if vel is not ... :
            vel = np.asarray(vel)
            if pos.shape[0] != vel.shape[0] or pos.shape[1] != vel.shape[1]:
                raise CatalogError("position and velocity data have different size/shape")
        self.vel      = vel

        self.redshift = z

        if space not in ["real", "z"]:
            raise CatalogError(f"invalid space ({space}), only `real` and `z` are the allowed values")
        if coord_sys not in ["cart", ]:
            raise CatalogError(f"invalid coordinate system ({coord_sys})")

        # create property table:
        self.prop = {
                        "space"     : space, 
                        "coord_sys" : coord_sys, 
                        "nobj"      : pos.shape[0], 
                        "los"       : ...,
                        **prop
                    }

        self.Hz = None

    def __repr__(self) -> str:
        return f"<Catalog z: {self.redshift}, nobj: {self.prop['nobj']}, space: '{self.prop['space']}'>"

    def properties(self, ) -> list:
        """
        Return a list of all the properties of the catalog.
        """
        return list(self.prop.keys())

    def propertyValue(self, key: str) -> Any:
        """
        Get the value of the given property, if it exists, else :class:`CatalogError` is raised.

        Parameters
        ----------
        key: str
            Name of the property.

        Returns
        -------
        val: Any
            Value of the named property.

        """
        if not isinstance(key, str):
            raise TypeError("key should be a 'str'")
        if key not in self.prop.keys():
            raise CatalogError(f"catalog has no property called `{key}`")
        return self.prop[key]

    def __getitem__(self, key: str) -> Any:
        """
        Alias for `propertyValue`: get the named property.
        """
        return self.propertyValue(key)

    def setHfunc(self, hfunc: Callable[[float], float] = ..., Om0: float = ..., Ode0: float = ..., H0: float = ...) -> None:
        r"""
        Set the functional form of the Hubble parameter  :math:`H(z)` as a function of 
        redshift :math:`z`. Alternatively, it creates a one if the cosmology parameters
        :math:`H_0`, :math:`\Omega_{\rm m}` and :math:`\Omega_{\rm de}` are given.

        .. math::
            H(z) = H_0 \sqrt{\Omega_{\rm m} (1 + z)^3 + (1 - \Omega_{\rm m} - \Omega_{\rm de})(1 + z)^2 + \Omega_{\rm de}}

        Parameters
        ----------
        hfunc: callable
            Function to be called to get the value of Hubble parameter. It must have 
            the signature `hfunc(z: float) -> float`.
        Om0: float, optional
            Normalised matter density. Used when no function is given.
        Ode0: float, optional
            Normalised dark-energy density. Used when no function is given.
        H0: float, optional
            Present value of Hubble parameter given in in km/s/Mpc. Used when no function 
            is given.

        """
        if hfunc is not ...:
            if not callable(hfunc):
                raise CatalogError("hfunc should be a callable")
        else:
            if Om0 is ... :
                raise CatalogError("Om0 should be given if no function is given")
            elif not isinstance(Om0, (float, int)):
                raise CatalogError("Om0 should be a real number")
            elif Om0 < 0.:
                raise CatalogError("Om0 should be positive")

            if Ode0 is ... :
                raise CatalogError("Ode0 should be given if no function is given")
            elif not isinstance(Ode0, (float, int)):
                raise CatalogError("Ode0 should be a real number")
            elif Ode0 < 0.:
                raise CatalogError("Ode0 should be positive")

            if H0 is ... :
                raise CatalogError("H0 should be given if no function is given")
            elif not isinstance(H0, (float, int)):
                raise CatalogError("H0 should be a real number")
            elif H0 < 0.:
                raise CatalogError("H0 should be positive")

            from collections import namedtuple
            
            cparams = namedtuple("cparams", ["Om0", "Ode0", "H0"], )
            self.prop['cparams'] = cparams(Om0, Ode0, H0)

            def hfunc(z: float) -> float:
                zp1 = 1. + np.asarray(z)
                Ok0 = 1. - Om0 - Ode0
                return H0 * np.sqrt(Om0 * zp1**3 + Ok0 * zp1**2 + Ode0)

        self.Hz = hfunc
        return

    def spaceConversion(self, action: str = "r2z") -> bool:
        r"""
        Convert object position between real and redshift spaces. The conversion
        is done by the relation (see Mo and White, Galaxy formation and evolution)
        
        .. math::
            {\bf s} = {\bf r} + \frac{{\bf v} \cdot hat{\bf r}}{H} hat{\bf r}

        Parameters
        ----------
        action: str
            Either `r2z` for real-to-redshift space or `z2r` for redshift-to-real 
            space transformation.

        """
        if action == "r2z":
            if self.prop["space"] == "z":
                warnings.warn("catalog already in redshift space")
                return False
            # convert from real to redshift: 
            self.pos           = self._transform_r2s()
            self.prop['space'] = "z"
            return True
        elif action == "z2r":
            if self.prop["space"] == "real":
                warnings.warn("catalog already in real space")
                return False
            # convert from redshift to real: 
            self.pos           = self._transform_s2r()
            self.prop['space'] = "real"
            return True
        raise CatalogError(f"invalid action {action}")

    def _transform_r2s(self, ) -> Any:
        """ 
        Real to redshift space transformation 
        """
        if "H" not in self.prop.keys():
            raise CatalogError("no H property for conversion")
        Hz = self.prop['H']
        
        if self.redshift is ... :
            raise CatalogError("no redshift given conversion")
        z = self.redshift

        if self.vel is ... :
            raise CatalogError("no velocity information")

        if self.prop["los"] is ... :
            los = self.pos
        else:
            los = np.asarray(self.prop["los"])
            
        if self.prop['coord_sys'] == 'cart':
            ds = np.sum(self.vel * los, axis = -1) / np.sum(los**2, axis = -1)
            ds = los * ds[:, np.newaxis]

            return self.pos + ds * (1 + z) / Hz
        else:
            raise NotImplementedError("function not implemented")

    def _transform_r2s(self, ) -> Any:
        """ 
        Redshift to real space transformation 
        """
        raise NotImplementedError("function not implemented")
        

            
        


