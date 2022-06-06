import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from pycosmo.cosmology import Cosmology
from pycosmo.distributions.density_field import GenExtremeDistribution

color = [ 'tab:blue', 'tab:green', 'orange' ]

def test3():
    c = Cosmology( 0.7, 0.25, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )

    plt.figure(figsize = [8,5])

    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    p = GenExtremeDistribution( c, 1.95  )
    x = np.logspace( 0, np.log10( 4 ), 31 ) - 1.95
    y = p.pdf( x, log_field = False )
    plt.plot( x, np.log10( y ), 'o-', ms = 4, color = color[0], label = f'2.0 h/Mpc' )

    p = GenExtremeDistribution( c, 15.6  )
    x = np.logspace( 0, np.log10( 4 ), 21 ) - 1.85
    y = p.pdf( x, log_field = False )
    plt.plot( x, np.log10( y ), 'o-', ms = 4, color = color[1], label = f'15.6 h/Mpc' )

    plt.legend(title = "Cellsize", fontsize = 14, title_fontsize = 14)
    plt.xlabel('$\\delta$', fontsize = 14)
    plt.ylabel('$\\log_{10} ~ P(\\delta)$', fontsize = 14)
    plt.show()

    return

def test2():
    c = Cosmology( 0.7, 0.25, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )
    p = GenExtremeDistribution( c, 1.95  )

    plt.figure(figsize = [8,5])

    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    for i, z in enumerate( [ 2, 1, 0 ] ):
        x = np.linspace( -4.0, 6.0, 51 )
        y = p.pdf( x, z = z )

        plt.plot( x, y, 'o-', ms = 4, color = color[i], label = f'{z:.1f}' )

    plt.legend(title = "Redshift", fontsize = 14, title_fontsize = 14)
    plt.xlabel('A = $\\ln (1+\\delta)$', fontsize = 14)
    plt.ylabel('P(A)', fontsize = 14)
    plt.show()

    return


if __name__ == '__main__':
    test2()
