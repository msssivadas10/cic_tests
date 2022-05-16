from pycosmo.cosmology import Cosmology
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def test1():
    c = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'sugiyama96' )

    x  = np.logspace( 8, 16, 21 )
    # x = np.logspace( -3, 3, 21 )

    plt.figure()
    # plt.semilogx()
    plt.loglog()

    y1 = c.massFunction( x, 0, 200 )
    plt.plot( x, y1, color = 'tab:blue' )

    y2 = c.massFunction( x, 0, 200 )
    plt.plot( x, y2, 'o', ms = 4, color = 'green')

    plt.show()

    return



if __name__ == '__main__':
    test1()
