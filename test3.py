import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'ggplot' )

from pycosmo.cosmology import FlatLambdaCDM

c = FlatLambdaCDM(0.70, 0.3, 0.05, 1.0, 0.8)

plt.figure()

plt.semilogx()
# plt.semilogx()

x = np.logspace( -3, 3, 51)
y = c.dlnsdlnr( x )

h=0.01

x1= (1+h)*x
z = ( np.log( c.variance(x1) ) - np.log( c.variance(x) ) ) / ( np.log(x1) - np.log(x) ) / 2

plt.plot( x, y, '-o', ms = 3)
plt.plot( x, z, '--' )

plt.show()