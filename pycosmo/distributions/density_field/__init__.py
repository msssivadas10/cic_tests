r"""

Distribution Functions for Over-density Field
=============================================

The `distributions.density_field` module contains some distribution functions for the over-density 
values, :math:`\delta`, such as generalized extreme value distribution. Distributions are available 
for the logarithmic field, :math:`\ln(1+\delta)` and the 'linear' field.

"""

from pycosmo.distributions.density_field.genextreem import GenExtremeDistribution

__all__ = [ 'genextreme',  ]