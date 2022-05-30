import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pycosmo.cosmology import Cosmology
from pycosmo.distributions.density_field import GenExtremeDistribution

def test3():
    c = Cosmology( 0.7, 0.25, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )

    plt.figure()

    p = GenExtremeDistribution( c, 1.95  )
    x = np.logspace( 0, np.log10( 4 ), 51 ) - 1.95
    y = p.pdf( x, log_field = False )
    plt.plot( x, np.log10( y ), 'o-', ms = 5, label = f'R=2.0 h/Mpc' )

    p = GenExtremeDistribution( c, 15.6  )
    x = np.logspace( 0, np.log10( 4 ), 51 ) - 1.85
    y = p.pdf( x, log_field = False )
    plt.plot( x, np.log10( y ), 'o-', ms = 5, label = f'R=15.6 h/Mpc' )

    plt.legend()
    plt.xlabel('$\\delta$')
    plt.ylabel('$\\log_{10} ~ P(\\delta)$')
    plt.show()

    return

def test2():
    c = Cosmology( 0.7, 0.25, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )
    p = GenExtremeDistribution( c, 1.95  )

    plt.figure()

    for z in ( 2, 1, 0 ):
        # print( p.supportInterval[1] ) 

        x = np.linspace( -6.0, 8.0, 51 )
        y = p.pdf( x, z = z )

        plt.plot( x, y, 'o-', ms = 5, label = f'z={z}' )

    plt.legend()
    plt.xlabel('A = $\\ln (1+\\delta)$')
    plt.ylabel('P(A)')
    plt.show()

    return

def test1():
    from scipy.special import gamma

    def f(shape: float) -> float:
        g1 = gamma( 1-shape )
        g2 = gamma( 1-shape*2 )
        g3 = gamma( 1-shape*3 )

        return ( g3 - 3*g1*g2 + 2*g1**3 ) / ( g2 - g1**2 )**1.5

    plt.figure()

    x = np.linspace(-1, 0.0001, 101)
    y = f(x)

    plt.plot(x, y)
    plt.show()



if __name__ == '__main__':
    test2()
