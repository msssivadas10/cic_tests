#!/usr/bin/python3

import numpy as np
from pycosmo.utils import settings
from scipy.optimize import curve_fit
from scipy.special import gammaln
from scipy.integrate import simpson
from typing import Any 
from abc import ABC, abstractmethod, abstractproperty

######################################################################################
# Base class for distribution
######################################################################################

class DistributionError(Exception):
    """
    Base class of exceptions raised by distributions.
    """

class Distribution(ABC):
    """
    Base class for probability distributions.
    """

    __slots__ = '_p', 'nparams', '_pcov'

    def __init__(self, nparams: int, p: tuple = None) -> None:

        assert isinstance(nparams, int)
        assert nparams >= 0

        self.nparams, self._p = nparams, None

        if p is None:
            return

        self._checkParameters(p)
        self._p = p

    def _checkParameters(self, p: tuple) -> None:
        """
        Check the given parameter set and raise exceptions.
        """

        nparams = self.nparams
        if len(p) != nparams:
            raise DistributionError(f"number of parameters must be {nparams}")

        bounds = self.bounds
        for i in range(nparams):
            p_lower, p_upper = bounds[i]
            if p[i] is None:
                continue
            if not p_lower <= p[i] <= p_upper:
                raise DistributionError(f"{i}-th parameter is out of bound")
        
        return

    @property
    def param(self) -> tuple:
        """
        Parameters for the distribution
        """
        return self._p

    @property
    def pcov(self) -> Any:
        """
        Parameter covariance matrix (in case of fitted distribution)
        """
        return self._pcov

    @abstractproperty
    def support(self) -> tuple:
        """
        Support interval for the distribution.
        """
        ...

    @abstractproperty
    def bounds(self) -> list:
        """
        Bounds for the distribution parameters. This will be list of 2-tuples, where each 
        tuple is the lower and upper bound for the corresponding parameter.
        """
        ...

    @staticmethod
    @abstractmethod
    def pdf(x: Any, *p: float) -> Any:
        """
        Distribution function.

        Parameters
        ----------
        x: array_like
            Argument for the function.
        *p: float
            A set of parameters for the distribution function as positional arguments.

        Returns
        ------
        y: array_like
            Values of the distribution function.

        """
        ...

    def fit(self, xdata: Any, ydata: Any, p0: tuple, **kwargs) -> 'Distribution':
        """
        Fit the distribution to the given data.
        """
        
        self._checkParameters(p0)
        bounds = tuple(np.transpose(np.asfarray(self.bounds)))

        self._p, self._pcov = curve_fit( self.pdf, xdata, ydata, p0, bounds = bounds, **kwargs )
        return self

    def call(self, x: Any) -> Any:
        """
        Return the value of the distribution function with given parameters.
        """

        if None is self.param:
            raise DistributionError("distribution is not initialized properly")

        x    = np.asfarray(x)
        y    = np.empty_like(x)
        y[:] = np.nan

        valid    = (self.support[0] <= x) & (x <= self.support[1])
        y[valid] = self.pdf( x[valid], *self.param )
        
        return y

    def __call__(self, x: Any) -> Any:
        return self.call(x)

    
######################################################################################
# some common distributions
######################################################################################


class Poisson(Distribution):
    r"""
    Poisson ditribution. This is defined for non-negative integer arguments. For this distribution, 
    the mean and standard deviation are the same as the distribution parameter :math:`\lambda`.

    .. math::
        f(n) = \frac{\lambda^n e^{-\lambda}}{n!}

    """

    def __init__(self, lam: float = None) -> None:
        super().__init__(nparams = 1, p = (lam, ))

    @property
    def support(self) -> tuple:
        return (0., np.inf)

    @property
    def bounds(self) -> list:
        return [(0., np.inf)]

    @staticmethod
    def pdf(x: Any, lam: float) -> Any:

        x = np.asfarray(x)
        y = x * np.log(lam) - lam - gammaln(x+1)
        y = np.exp(y)
        return y


class GQED(Distribution):
    r"""
    The Gravitational Quasi-Equilibrium Distribution (GQED). It is used as a model for galaxy count-in-
    cells distribution. It comes from the statistcal mechanics of a set of interacting particles.

    .. math::
        f(n) = \frac{a(1-b)}{n!} ( a(1-b) + nb )^{n-1} \exp(-a(1-b) - nb)

    where :math:`a = \bar{n}V` is the average expected number of galaxies in a cell of volume :math:`V` 
    and :math:`\bar{n}` is the average number density of galaxies. :math:`b` is related to the volume 
    integral of the correlation function :math:`b = 1 - (a \bar{\xi_2} + 1)^{-1/2}`.

    """

    def __init__(self, a: float = None, b: float = 0.) -> None:
        super().__init__(nparams = 2, p = (a, b))

    @property
    def support(self) -> tuple:
        return (0., np.inf)

    @property
    def bounds(self) -> list:
        return [(0., np.inf), (0., 1.)]

    @staticmethod
    def pdf(x: Any, a: float, b: float) -> Any:

        x = np.asfarray(x)

        if abs(b - 1.0) < 1e-08 or abs(a) < 1e-08:
            return np.zeros_like(x)

        c = a*(1 - b)
        y = c + b*x
        y = np.log(c) - gammaln(x+1) + (x-1)*np.log(y) - y
        y = np.exp(y)
        return y


class NegativeBinomial(Distribution):
    r"""
    Negative Binomial Distribution (NBD) for galaxy count in cells.
    """
    
    def __init__(self, a: float = None, g: float = 1.) -> None:
        super().__init__(nparams = 2, p = (a, g))

    @property
    def support(self) -> tuple:
        return (0., np.inf)

    @property
    def bounds(self) -> list:
        return [(0., np.inf), (0., np.inf)]

    @staticmethod
    def pdf(x: Any, a: float, g: float) -> Any:

        x  = np.asfarray(x)
        gi = 1. / g

        y = (gammaln(x + gi) + x*np.log(a) + gi*np.log(gi) 
                - gammaln(gi) - gammaln(x+1) - (x + gi)*np.log(a + gi))
        y = np.exp(y)
        return y


class BiasedLognormal(Distribution):
    r"""
    Log-normal ditribution for galaxy count in cells (with halo bias). The matter density is 
    assumed as log-normal distributed. Discrete Poisson sampling of galaxies will include a 
    shot noise term, resulting in the final distribution to be a convolution of this with the 
    Poisson distribution. The parameter :math:`a` is the average expected number of galaxies in 
    a cell, :math:`s` is the matter density variance parameter and :math:`b` is the halo bias 
    term. 
    
    """
    
    def __init__(self, a: float = None, s: float = None, b: float = 1.) -> None:
        super().__init__(nparams = 3, p = (a, s, b))

    @property
    def support(self) -> tuple:
        return (0., np.inf)

    @property
    def bounds(self) -> list:
        return [(0., np.inf), (0., np.inf), (0., np.inf)]

    @staticmethod
    def pdf(x: Any, a: float, s: float, b: float) -> Any:

        delta_max, npts  = settings.DELTA_MAX, (2**settings.DEFAULT_SUBDIV + 1)
        delta, step_size = np.linspace(-1., delta_max, npts, retstep = True)
        delta  = delta[:, None]

        x = np.asfarray(x)

        # poisson distribution
        y   = a*(delta + 1)
        lny = np.zeros_like(y)
        j   = (y > 0.)
        lny[j] = np.log(y[j])
        poiss  = np.exp( x*lny - y - gammaln(x+1) )

        # log-normal part
        c, dp1 = np.log(s**2 + 1), delta + b
        j      = (dp1 != 0.)
        lnorm  = np.zeros_like(dp1)
        lnorm[j] = 0.5*c + np.log(dp1[j])
        lnorm[j] = 1./np.sqrt(2*np.pi*c) * np.exp(-0.5*lnorm[j]**2/c) / dp1[j]

        # conolution
        y = simpson(lnorm*poiss, dx = step_size, axis = 0)
        return y

    
class Lognormal(Distribution):
    r"""
    Log-normal ditribution for galaxy count in cells (without halo bias). The parameter :math:`a` is 
    the average expected number of galaxies in a cell, :math:`s` is the matter density variance parameter.

    See Also
    -------
    BiasedLognormal: Log-normal distribution, including a halo bias term.

    """
    
    def __init__(self, a: float = None, s: float = 1.) -> None:
        super().__init__(nparams = 2, p = (a, s))

    @property
    def support(self) -> tuple:
        return (0., np.inf)

    @property
    def bounds(self) -> list:
        return [(0., np.inf), (0., np.inf)]

    @staticmethod
    def pdf(x: Any, a: float, s: float) -> Any:
        return BiasedLognormal.pdf(x, a, s, b = 1.)


class PowerLaw(Distribution):
    r"""
    A power law distribution. This is supported for the half real line and can be used for 
    fitting correlation functions.

    .. math::
        f(x) = \left( \frac{x}{a} \right)^{-b}

    """

    def __init__(self, a: float = None, b: float = None) -> None:
        super().__init__(nparams = 2, p = (a, b))

    @property
    def support(self) -> tuple:
        return (0., np.inf)

    @property
    def bounds(self) -> list:
        return [(0., np.inf), (0., np.inf)]

    @staticmethod
    def pdf(x: Any, a: float, b: float) -> Any:

        x = np.asfarray(x)
        return (x / a)**(-b)


class GEV(Distribution):
    """
    Generalized Extreme Value (GEV) distribution for galaxy count in cells.
    """
    ...

