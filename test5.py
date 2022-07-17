import numpy as np
import matplotlib.pyplot as plt
from rich import Any
plt.style.use('ggplot')

from pycosmo.cosmology import Cosmology
from pycosmo.distributions.density_field import GenExtremeDistribution

from scipy.integrate import simps

color = [ 'tab:blue', 'tab:green', 'orange' ]

def cicpdf( n: Any, p: GenExtremeDistribution, nbar: int, b: float ):

    a = np.linspace( -10, 10, 10001 ) # log( b*delta + 1 )
    
    
    delta = ( np.exp( a ) - 1 ) / b
    lamda = ( b * delta + 1 ) * nbar

    from scipy.special import gammaln
    y = np.exp( n * np.log( lamda[:,None] ) - lamda[:,None] - gammaln( n+1 ) ) 
    y = y * p.pdf( a )[:,None]

    return simps( y, dx = a[1] - a[0], axis = 0 )






def test3():
    c = Cosmology( 0.7, 0.25, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )

    plt.figure(figsize = [8,5])

    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    n = np.floor( np.logspace( 0, np.log10(300), 21 ) )

    p = GenExtremeDistribution( c, 1.95  )
    # x = np.logspace( 0, np.log10( 4 ), 31 ) - 1.95
    # y = p.pdf( x, log_field = False )
    # plt.plot( x, np.log10( y ), 'o-', ms = 4, color = color[0], label = f'2.0 h/Mpc' )
    y = cicpdf( n, p, 100, 1.4 )
    plt.plot( n, y, 'o-', ms = 4, color = color[0], label = f'2.0 h/Mpc' )

    p = GenExtremeDistribution( c, 15.6  )
    # x = np.logspace( 0, np.log10( 4 ), 21 ) - 1.85
    # y = p.pdf( x, log_field = False )
    # plt.plot( x, np.log10( y ), 'o-', ms = 4, color = color[1], label = f'15.6 h/Mpc' )
    y = cicpdf( n, p, 100, 1.4 )
    plt.plot( n, y, 'o-', ms = 4, color = color[1], label = f'15.6 h/Mpc' )

    plt.legend(title = "Cellsize", fontsize = 14, title_fontsize = 14)
    # plt.xlabel('$\\delta$', fontsize = 14)
    # plt.ylabel('$\\log_{10} ~ P(\\delta)$', fontsize = 14)
    plt.xlabel('N', fontsize = 14)
    plt.ylabel('P(N)', fontsize = 14)
    plt.show()

    return

def test2():
    c = Cosmology( 0.7, 0.25, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )
    p = GenExtremeDistribution( c, 1.95  )

    plt.figure(figsize = [8,5])

    # plt.loglog()

    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    x = np.linspace( -4.0, 6.0, 51 )
    # x = np.logspace( -4.0, 2.0, 21 )
    for i, z in enumerate( [ 2, 1, 0 ] ):
        y = p.pdf( x, z = z )
        # b = c.linearBias( 2000.0, z )
        # y = c.matterCorreleation( x, z ) #* b**2

        plt.plot( x, y, 'o-', ms = 4, color = color[i], label = f'{z:.1f}' )

    plt.legend(title = "Redshift", fontsize = 14, title_fontsize = 14)
    plt.xlabel('A = $\\ln (1+\\delta)$', fontsize = 14)
    plt.ylabel('P(A)', fontsize = 14)
    # plt.xlabel('r in Mpc/h', fontsize = 14)
    # plt.ylabel('$\\xi$(r)', fontsize = 14)
    plt.show()

    return


if __name__ == '__main__':
    test3()
