from pycosmo2.cosmology.cosmo import Cosmology
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def test2():
    c = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'sugiyama96' )

    import pycosmo.core.cosmology as cm
    d = cm.Cosmology( h=0.7, Om0=0.3, Ob0=0.05, sigma8=0.8, ns=1.0, transfer_function='sugiyama96' )

    # x  = np.logspace(-3, 3, 501)
    # y2 = p.variance( x )**0.5
    # from scipy.interpolate import CubicSpline
    # f = CubicSpline( np.log(x), np.log(y2) )

    x  = np.logspace( -3, 3, 51 )
    y1 = d.matterPowerSpectrum( x, dim=0, lin=1 )
    # # y1 = f(np.log(x), nu = 2) 
    y2 = c.matterPowerSpectrum( x, dim=0, linear=1 )

    plt.figure()
    # plt.semilogx()
    plt.loglog()
    plt.plot( x, y1, color = 'tab:blue' )
    plt.plot( x, y2, 'o', ms = 4, color = 'green')
    plt.show()

    return


if __name__ == '__main__':
    # integrate_osc_test()
    test2()