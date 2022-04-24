from pycic.cosmo import Cosmology
from pycic.distr import GEV, Lognormal

import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'ggplot' )

def timeit(func):
    def _timeit(*args, **kwargs):
        import time 
        t = time.time()
        y = func(*args, **kwargs)
        print( "execution time: {} sec".format( time.time() - t ) )
        return y
    return _timeit

def plotit(scale: str = None):
    def _plotit(func):
        def __plotit(*args, **kwargs):
            out = func(*args, **kwargs)
            if out is None:
                return
            if len( out ) < 2 :
                return out
            x = out[0]
            plt.figure()
            if scale is not None:
                eval( 'plt.{}()'.format( scale ) )
            for y in out[1:]:
                plt.plot(x, y)
            plt.show()
            return x, y
        return __plotit
    return _plotit

@timeit
def gev(b: float, z: float, cm: Cosmology, sigma8: float, bias: float):
    d = GEV( b, z, cm )
    d.parametrize( sigma8, bias )

    # x = np.linspace(1, 5000, 51)
    # y = d.fcount( x, 5.0, 100 )

    # x = np.linspace( -5.0, d.support().b, 201)
    # y = d.f( x, True )

    x = np.arange(500)
    y = d.fcic( x, 100 )

    return x, y

@timeit
def lognorm(b: float, z: float, cm: Cosmology, sigma8: float, bias: float):
    d = Lognormal( b, z, cm )
    d.parametrize( sigma8, bias )


    # x = np.linspace(-1, 5, 501) 
    # y = d.fcount( 100, 50 * ( x * d.bias  + 1 ) )
    # x = np.linspace( -d.bias, 5.0, 201)
    # y2 = d.f( x )

    x = np.arange(500)
    y = d.fcic( x, 250 )

    return x, y

@plotit(scale = 'semilogx')
def main1():
    cm   = Cosmology( Om0 = 0.3, Ob0 = 0.05, h = 0.70 )
    # x, y = gev( 2.0, 0.0, cm, 1.0, 0.8 )
    # x, y = lognorm( 2.0, 0, cm, 1.0, 0.8 )
    x = np.logspace(-3, 3, 21)
    y = cm.neff( x )
    return x, y


@plotit(scale = 'loglog')
@timeit
def main2():
    from pycic.catalog import LognormalCatalog

    cm  = Cosmology( Om0 = 0.3, Ob0 = 0.05, h = 0.70, sigma8 = 0.8 )
    cat = LognormalCatalog(1000, 10000, 64, 0.0, 1.4, cm )

    x = np.logspace( -4, 4, 101 )

    y = cat.gfieldPowerSpectrum( x, False )
    z = cat._gfieldPowerSpectrum( x, True )

    # gg, gm = cat.gfield()

    # print(gg.max(), gm.max())

    return x, y, z






main1()