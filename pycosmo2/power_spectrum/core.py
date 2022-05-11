from typing import Any 
import numpy as np
import pycosmo2.cosmology.cosmo as cosmo
import pycosmo2.power_spectrum.base as base
import pycosmo2.power_spectrum.nonlinear_power as nlp
import pycosmo2.power_spectrum.transfer_functions as tf


class PowerSpectrum(base.PowerSpectrum):
    r"""
    An abstract power spectrum class. A properly initialised power spectrum object cann be 
    used to compute the linear and non-linear power spectra, matter variance etc.

    Parameters
    ----------
    cm: Cosmology
        Working cosmology object. Power spectrum objects extract cosmology parameters from 
        these objects.
    filter: str. optional
        Filter to use for smoothing the density field. Its allowed values are `tophat` (default), 
        `gauss` and `sharpk`.
    
    Raises
    ------
    PowerSpectrumError

    """

    __slots__ = 'nonlinear_model', 'linear_model', 

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)

        self.linear_model    = None
        self.nonlinear_model = 'halofit'

    def nonlinearPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        return nlp.nonlinearPowerSpectrum( self, k, z, dim, model = self.nonlinear_model )

    def nonlineark(self, k: Any, z: float = 0) -> Any:
        k   = np.asfarray( k )
        dnl = self.nonlinearPowerSpectrum( k, z, dim = False )
        return k * np.cbrt( 1.0 + dnl )


#######################################################################################################

# predefined models

class Sugiyama96(PowerSpectrum):

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'sugiyama96'

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return tf.psmodelSugiyama96( self.cosmology, k, z )

class BBKS(Sugiyama96):

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'bbks'

class Eisenstein98_zeroBaryon(PowerSpectrum):

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'eisenstein98_zb'

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return tf.psmodelEisenstein98_zeroBaryon( self.cosmology, k, z )

class Eisenstein98_withBaryon(PowerSpectrum):

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'eisenstein98_wb'

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return tf.psmodelEisenstein98_withBaryon( self.cosmology, k, z )

class Eisenstein98_withNeutrino(PowerSpectrum):

    def __init__(self, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'eisenstein98_nu'

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return tf.psmodelEisenstein98_withNeutrino( self.cosmology, k, z, self.use_exact_growth )


# power spectrum from raw data:

class PowerSpectrumTable(PowerSpectrum):

    __slots__ = 'spline', 

    def __init__(self, lnk: Any, lnt: Any, cm: cosmo.Cosmology, filter: str = 'tophat') -> None:
        super().__init__(cm, filter)
        self.linear_model = 'rawdata'

        lnk, lnt = np.asfarray( lnk ), np.asfarray( lnt )
        if np.ndim( lnk ) != 1 or np.ndim( lnt ) != 1:
            raise base.PowerSpectrumError("both lnk and lnt must be 1-dimensional arrays")
        
        from scipy.interpolate import CubicSpline

        self.spline = CubicSpline( lnk, lnt )

    def transferFunction(self, k: Any, z: float = 0) -> Any:
        return np.exp( self.spline( np.log( k ) ) )


