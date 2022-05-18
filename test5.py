import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from pycosmo.cosmology import Cosmology
from pycosmo.distributions.density_field.genextreem import GenExtremeDistribution

def test2():
    c = Cosmology( 0.7, 0.25, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )

    p = GenExtremeDistribution( c, 1.95  )

    plt.figure()

    for z in ( 0, 1, 2 ):
        p.setup( 0.8, z )

        # print( p.param, p.supportInterval[1] ) 

        x = np.linspace( -6.0, 8.0, 501 )
        y = p.pdf( x )

        plt.plot( x, y, label = f'z={z}' )

    plt.legend()
    plt.xlabel('A')
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

    x = np.linspace(-0.4, 0.0001, 101)
    y = f(x)

    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    test2()
