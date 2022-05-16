import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


from pycosmo.cosmology import Cosmology
from pycosmo.distributions.density_field.genextreem import GenExtremeDistribution

def test1():
    c = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )

    p = GenExtremeDistribution( c, 1.95, 0  )

    a = -0.2

    kx, ky = np.mgrid[ -a:a:501j, -a:a:501j ]
    kz     = 0

    y = p._measuredPowerSpectrum( kx, ky, kz )


    plt.figure()

    plt.contourf( kx, ky, y )

    plt.show()

    return



if __name__ == '__main__':
    test1()
