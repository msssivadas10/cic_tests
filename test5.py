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

    fig, ( ( ax1, ax2 ), ( ax3, ax4 ) ) = plt.subplots(2, 2, )

    y = p._measuredPowerSpectrum( kx, ky, kz )
    ax1.contourf( kx, ky, y )

    y = p._measuredPowerSpectrum( kx, ky, kz )
    ax2.contourf( kx, ky, y )

    y = p._measuredPowerSpectrum( kx, ky, kz )
    ax3.contourf( kx, ky, y )

    ###################################################################

    x = np.logspace( -3, np.log10(p.kn), 201 )

    ax4.loglog()

    y = p._measuredPowerSpectrum( x, 0, 0 )
    ax4.plot(x, y)

    y = p._measuredPowerSpectrum( 0, x, 0 )
    ax4.plot(2*x, y)

    y = p._measuredPowerSpectrum( 0, 0, x )
    ax4.plot(4*x, y)

    plt.show()

    return


def test2():
    c = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )

    p = GenExtremeDistribution( c, 1.95, 0  )

    plt.figure()
    plt.loglog()  

    n1, n2 = 51, 501
    k = np.logspace( -4, np.log10(p.kn), n1 )

    t = np.random.uniform( 0, np.pi, (n2, n1) )
    s = np.random.uniform( 0, 2*np.pi, (n2, n1) )
    
    kx = k*np.sin(t)*np.cos(s)
    ky = k*np.sin(t)*np.sin(s)
    kz = k*np.cos(t)

    y = np.mean( p._measuredPowerSpectrum(kx, ky, kz), axis = 0 )

    plt.plot( k/p.kn, y  )

    k1 = np.sqrt( kx**2 + ky**2 + kz**2 ).flatten()
    y1 = p.measuredPowerSepctrum( kx, ky, kz ).flatten()

    plt.plot( k1/p.kn, y1, 'o', ms = 3 )


    plt.show()

    return


if __name__ == '__main__':
    test2()
