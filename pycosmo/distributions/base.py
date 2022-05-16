from typing import Any 
from abc import ABC, abstractmethod, abstractproperty


class DistributionError(Exception):
    r"""
    Base class of exceptions used by distribution objects.
    """
    ...

class Distribution(ABC):
    r"""
    Probability distribution base class.
    """

    @abstractmethod
    def pdf(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        Return the probability density function. 
        """
        ...

    @abstractproperty
    def supportInterval(self) -> tuple:
        r"""
        Return the interval of support.
        """
        ...

    @abstractmethod
    def setup(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        Setup the distribution function.
        """
        ...

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.pdf(*args, **kwargs)
        
