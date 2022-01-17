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
    mass: array_like, optional
        Object mass.
    mag: array_like, optional
        Object magnitudes.
    z: float
        Redshift at the time of configuration.

    
    """
    __slots__ = (
                    'x', 'y', 'z', 'vx', 'vy', 'vz', 'm', 'mag', 
                    'redshift', "nobj", "space"
                )
    
    def __init__(self, pos: Any, vel: Any, mass: Any = ..., mag: Any = ..., z: float = ..., space: str = "real") -> None:
        pos, vel = np.asarray(pos), np.asarray(vel)
        if pos.shape[1] != 3:
            raise CatalogError("data should correspond to 3 dimensionsss")
        if pos.shape[0] != vel.shape[0] or pos.shape[1] != vel.shape[1]:
            raise CatalogError("position and velocity data have different size/shape")

        nobj = pos.shape[0]

        if mass is not ... :
            mass = np.asarray(mass)
            if mass.shape[0] != nobj or mass.shape[1] != 3:
                raise CatalogError("mass data have different size/shape")
        
        if mag is not ... :
            mag = np.asarray(mag)
            if mag.shape[0] != nobj or mag.shape[1] != 3:
                raise CatalogError("magnitude data have different size/shape")

        self.nobj = nobj
        self.x,  self.y,  self.z  = pos.T
        self.vx, self.vy, self.vz = vel.T
        self.m, self.mag          = mass, mag
        self.redshift             = z
        self.space                = space

    def __repr__(self) -> str:
        return f"<Catalog z: {self.redshift}, nobjects: {self.nobj}, space: '{self.space}'>"

    
