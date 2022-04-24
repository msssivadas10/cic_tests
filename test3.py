import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'ggplot' )

from pycosmo.cosmology.models import Cosmology

c1 = Cosmology(True, 0.7, 0.3, 0.05, sigma8 = 0.8)

def printtree(root, lv = 0):
    print( "\t"*lv + repr( root ) )
    for c in root.childComponents.values():
        printtree(c, lv+1)

# printtree( c1.univ )
# print( c1.Om0, c1.Ob0, c1.Onu0, c1.Oc0, c1.Ode0, c1.Ok0, c1.Nnu, c1.mnu )

plt.figure()

plt.loglog()
# plt.semilogx()

x = np.logspace( -3, 3, 21)
y = c1.variance( x )

plt.plot( x, y, '-o', ms = 3)

plt.show()

# print( np.exp(c1.universeAge( 0 )) / 1e9 )


