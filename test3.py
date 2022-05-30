import numpy as np
import matplotlib.pyplot as plt
from pycosmo.cosmology import Cosmology
from pycosmo.nbodytools.simulation import InitialCondition, ParticleMeshSimulation
from pycosmo.nbodytools.estimators.density import densityCloudInCell
from pycosmo.nbodytools.estimators.power_spectrum import powerSpectrum

c   = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )
pd  = InitialCondition(300.0, 48, c)(100)
sim = ParticleMeshSimulation( pd, cm = c  )

xout, aout = [], []

a1 = np.logspace( np.log10(sim.a), 0, 101 )
a2 = ( 1 + np.asfarray([50.0, 20.0, 0.0]) )**-1
a  = np.unique( np.hstack([ a1, a2 ]) )
a.sort()

i = np.searchsorted( a, a2 ) - 1

for j in range( a.shape[0]-1 ):
    da = a[j+1] - a[j]

    sim.updateParticles(da)

    if j in i:
        dens = densityCloudInCell( sim.currentPos, sim.boxsize, sim.gridsize )
        dens = dens / dens.mean()

        dens = np.log( dens[ dens != 0.0 ] )

        xout.append( dens.flatten() )

        # k, p = powerSpectrum( sim.currentPos, sim.boxsize, sim.gridsize, bins = 50 )
        # xout.append( (k, p) )

        aout.append( sim.a )

plt.figure(  )

for x, a in zip( xout, aout ):
    # plt.loglog()
    # plt.plot( k, p, label = f"a = {a:.3f}" )

    plt.hist( x, bins = 101, density = True, range = [-10, 10], rwidth = 0.9, histtype = 'step', lw = 1.5, label = f"a = {a:.3f}" )

plt.legend()
plt.show()