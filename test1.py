from pycic.cosmo import Cosmology
from pycic.distr import GEV

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

@timeit
def pdf(b: float, z: float, cm: Cosmology, sigma8: float, bias: float):
    d = GEV( b, z, cm )
    d.parametrize( sigma8, bias )

    # x = np.linspace(1, 5000, 51)
    # y = d.fcount( x, 5.0, 100 )

    # print( d.support(), d.param('xi') )
    
    # x = np.linspace( -5.0, d.support().b, 201)
    # y = d.f( x, True )

    x = np.linspace(10, 500, 201)
    y = d.fcic( x, 100 )

    return x, y


# x = np.linspace( -5.0, 5.0, 201 )
# y = GEV.pdf( x, -1.0 )

cm   = Cosmology( Om0 = 0.3, Ob0 = 0.05, h = 0.70 )

plt.figure()

x, y = pdf( 2.0, 0.0, cm, 1.0, 0.5 )
plt.plot( x, y )
plt.show()