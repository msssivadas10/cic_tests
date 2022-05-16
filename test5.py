from pycosmo.cosmology.cosmo import Cosmology
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def test2():
    c = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'sugiyama96' )

    x  = np.logspace( 8, 16, 21 )
    # x = np.logspace( -3, 3, 21 )
    

    plt.figure()
    # plt.semilogx()
    plt.loglog()

    y1 = c.linearBias( x, 0, 200 )
    plt.plot( x, y1, color = 'tab:blue' )

    y2 = c.lin( x, 0, 200 )
    plt.plot( x, y2, 'o', ms = 4, color = 'green')


    plt.show()

    return


if __name__ == '__main__':
    # integrate_osc_test()
    test2()
