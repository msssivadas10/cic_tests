from pycosmo2.cosmo import Cosmology
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

# print(plt.style.available)

def integrate_osc_test():
    from pycosmo2.utils.numeric import integrate2, erfc

    def f(x):
        y = np.zeros_like(x, 'float')
        y[ x!= 0 ] = np.sin(x[x!=0]) / np.log(1+x[x!=0])
        return y

    print("oscillatory integral test:")
    eps = 1e-07
    q   = int( np.ceil( np.sqrt( -np.log( eps ) ) ) )
    p   = 2*q
    L   = 2*p*q
    print( 
            integrate2( 
                        lambda x: f(x)*erfc(x/p-q)*0.5, 0, L 
                      ), 
                        2.0410186151477098866 
         )

def test1():
    c = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0 )
    # print(c)

    x  = np.logspace(-3, 1, 21)
    y1 = c.f( x, False )
    y2 = c.f( x, True )

    plt.figure()
    plt.semilogx()
    plt.plot( x, y1, color = 'tab:blue' )
    plt.plot( x, y2, 'o', ms = 4, color = 'green')
    plt.show()


if __name__ == '__main__':
    # integrate_osc_test()
    test1()