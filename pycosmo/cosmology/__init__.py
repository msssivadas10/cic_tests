r"""

Cosmology Module
================

The cosmology module, `pycosmo.cosmology` defines the :class:`Cosmology` object. This object is the main 
tool for all the cosmology related calculations. This object stores a specific cosmology model and do all 
computations using that model. Also, there will be some specific models such as :math:`Lambda`-CDM model, 
which are implemented as subclasses of the base :class:`Cosmology` class.

**Todo**: Predefined cosmology model objects.


Cosmological Functions
-----------------------

A :class:`Cosmology` object has some methods for calculations related to cosmology, such as the linear growth 
factor and rate, component density functions, Hubble parameters etc. Also, there will be methods to compute 
the horizons (if exists) and distances, times etc.


Power Spectrum and Related Calculations
----------------------------------------

:class:`Cosmology` object can compute the matter power spectrum (linear or non-linear) of a specified model. 
It also has the ability to use tabulated transfer functions. Also, it can accept a user defined power spectrum 
model, a subclass of :class:`pycosmo.objects.PowerSpectrum` and use that model for power spectrum.

Related quantities to the power spectrum are the 2-point correlation function, variance etc.


Halo Massfunction and Linear Bias
----------------------------------

A :class:`Cosmology` object can be used to compute the halo massfunction and linear halo bias of a specific model. 
One can specify a pre-defined model or use a user defined model. A halo massfunction model should be subclass of 
:class:`pycosmo.objects.HaloMassFunction` and linear bias model, a subclass of:class:`pycosmo.objects.LinearBias`.

"""

__all__ = [ 'cosmo', 'models' ]

# base model
from pycosmo.cosmology.models import Cosmology, CosmologyError

# other derived models
from pycosmo.cosmology.models import (
                                        Cosmology_wMDM,         
                                        Cosmology_flat_wMDM,
                                        Cosmology_LambdaCDM, 
                                        Cosmology_flat_LambdaCDM
                                     )