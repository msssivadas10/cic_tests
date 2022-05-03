from abc import ABC, abstractmethod
from typing import Any


class MassFunctionError( Exception ):
    r"""
    Base class of exceptions used by halo mass-function objects.
    """
    ...

class MassFunction( ABC ):
    r"""
    A class representing a halo mass-function model.
    """

    __slots__ = 'mass_function', 'depend_z', 'depend_cosmology', 'mdefs', 'cosmology'

    def __init__(self, model: str, depend_z: bool, depend_cosmo: bool, mdefs: list, cm: object) -> None:
        self.mass_function    = model # name of the models

        # model specific attributes
        self.depend_z         = depend_z     # model depends on redshift 
        self.depend_cosmology = depend_cosmo # depends on a cosmology model
        self.mdefs            = mdefs        # allowed mass definition types

        import cosmology
        if not isinstance( cm, cosmology.Cosmology ):
            raise MassFunctionError( "cm must be a cosmology model object" )
        self.cosmology = cm # cosmology model to use

    @abstractmethod
    def f(self, sigma: Any, *args, **kwargs) -> Any:
        r"""
        Compute the fitting function.
        """
        ...

    @abstractmethod
    def dndlnm(self, m: Any, *args, **kwargs) -> Any:
        r"""
        Compute the halo mass-function.
        """
        ...

    @abstractmethod
    def dndm(self, m: Any, *args, **kwargs) -> Any:
        r"""
        Compute the halo mass-function.
        """
        ...

    











