#!/usr/bin/python3
from typing import Any
import numpy as np 


class CatalogError(Exception):
    """
    Exceptions used by catalog objects.
    """
    ...

class Catalog:
    """
    An object to hold details of galaxy/halo objects (catalog). 

    Parameters
    ----------
    pos: array_like
        Object positions as a list/array of 3D vectors.
    vel: array_like
        Object velocity as list/array of 3D vectors.
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
                    'pos', 'vel', 'prop', 'redshift', 
                )
    
    def __init__(self, pos: Any, vel: Any, z: float = ..., space: str = "real", coord_sys: str = "cart", **prop) -> None:
        pos, vel = np.asarray(pos), np.asarray(vel)
        if pos.shape[1] != 3:
            raise CatalogError("data should correspond to 3 dimensionsss")
        if pos.shape[0] != vel.shape[0] or pos.shape[1] != vel.shape[1]:
            raise CatalogError("position and velocity data have different size/shape")

        nobj = pos.shape[0]

        self.pos      = pos
        self.vel      = vel
        self.redshift = z

        if space not in ["real", "z"]:
            raise CatalogError(f"invalid space ({space}), only `real` and `z` are the allowed values")
        if coord_sys not in ["cart", ]:
            raise CatalogError(f"invalid coordinate system ({coord_sys})")

        # create property table:
        self.prop = {
                        "space": space, "coord_sys": coord_sys, "nobj": nobj, **prop
                    }

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
        if key not in self.prop.keys():
            raise CatalogError(f"catalog has no property called `{key}`")
        return self.prop[key]

    def __getitem__(self, key: str) -> Any:
        """
        Alias for `propertyValue`: get the named property.
        """
        return self.propertyValue(key)

    def countCells(self, ) -> None:
        """
        Divide the entire space into cells and count the number of objects in each cell.
        This is stored in the attribute `cell_count` and can be accessed by the method
        'getCount`.

        Parameters
        ----------
        subdiv: int, optional
            Number of cell divisions to use.
        """



