import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'ggplot' )

from pycosmo.cosmology import FlatLambdaCDM

c = FlatLambdaCDM(0.70, 0.3, 0.05, 1.0, 0.8)

plt.figure()

plt.loglog()
# plt.semilogx()

x = np.logspace( -3, 3, 11)
y = c.correlation( x )

plt.plot( x, y, '-o', ms = 3)

plt.show()

