r"""
Power Spectrum Module
=====================

The `power_spectrum` module defines the :class:`PowerSpectrum` type, which is the base for all power 
spectrum models. These models are then used to compute the linear and non-linear power spectra and 
related quantities.

Available power spectrum/linear tranfer function models are

======================= ===============================================================
Key                     Model 
======================= ===============================================================
`sugiyama96`/`bbks`     Bardeen et al (1986), with correction by Sugiyama (1996)
`eisenstein98_zb`       Eisenstein & Hu (1997) without BAO 
`eisenstein98_wb`       Eisenstein & Hu (1997) with BAO
`eisenstein98_nu`       Eisenstein & Hu (1997) with massive neutrinos
======================= ===============================================================

Available non-linear power spectrum models are `halofit` and `peacock_dodds` (Peacock and Dodds, 1996).
By default, the halofit model is used, but can be changed by setting the `nonlinear_model` attribute of 
the :class:`PowerSpectrum` object to a valid model name.

Available filters are `tophat` (spherical top-hat), `gauss` (Gaussian) and `sharpk`. 

"""

__all__ = [ 'filters', 'linear_power', 'nonlinear_power', 'power', 'transfer_functions' ]

from pycosmo2.power_spectrum.core import PowerSpectrum
from pycosmo2.power_spectrum.base import PowerSpectrumError

# predefined power spectrum models
from pycosmo2.power_spectrum.core import (
											Eisenstein98_withBaryon, 
											Eisenstein98_zeroBaryon, 
											Eisenstein98_withNeutrino, 
											Sugiyama96, BBKS,
											PowerSpectrumTable
										)

models = {
            'sugiyama96'     : Sugiyama96,
            'bbks'           : BBKS,
            'eisenstein98_zb': Eisenstein98_zeroBaryon,
            'eisenstein98_wb': Eisenstein98_withBaryon,
            'eisenstein98_nu': Eisenstein98_withNeutrino,
         }