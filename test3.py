import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'ggplot' )

from pycosmo.cosmo import Cosmology


c = Cosmology(0.70, 0.3, 0.05, 1.0, 0.8)

plt.figure()

x = np.logspace( -3, 3, 101)

y = c.matterPowerSpectrum( x, model = '' ) 

plt.loglog( x, y )

plt.show()