import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycosmo.cosmology import Cosmology
from pycosmo.nbodytools.simulation import InitialCondition, ParticleMeshSimulation, ParticleParticleSimulation
from pycosmo.nbodytools.estimators.density import densityCloudInCell, cicInterpolate

plt.style.use('ggplot')

def testpp():
    c  = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )
    pd = InitialCondition(100.0, 20, c)(100)

    sim = ParticleParticleSimulation( pd, cm = c )

    fig = plt.figure(figsize = [6,6])
    ax = fig.add_subplot(111, projection='3d')
    
    col = [[0.0, 0.0, 0.0]]
    ax.scatter(sim.currentPos[:,0], sim.currentPos[:,1], sim.currentPos[:,2], s = 2, cmap = 'hot', c = col, alpha = 0.2)
    plt.pause(0.01)

    a  = np.logspace( np.log10(sim.a), 0, 21 )

    for da in np.diff( a ):
        sim.updateParticles(da)

        ax.cla()
        col = cicInterpolate( 
                                densityCloudInCell( sim.currentPos, sim.boxsize, sim.gridsize ), 
                                sim.currentPos, 
                                sim.boxsize 
                            )
        ax.scatter(sim.currentPos[:,0], sim.currentPos[:,1], sim.currentPos[:,2], s = 1, cmap = 'hot', c = col, alpha = 0.2)

        for __edge in pd.boundindgBox.edges:
            ax.plot( __edge[:,0], __edge[:,1], __edge[:,2], color = 'black', lw = 1 )

        ax.set_title( f"{sim.a:.3f}" )

        plt.pause(0.01)

        # break

    # d = densityCloudInCell( sim.currentPos, sim.boxsize, 128 )

    # plt.pcolor(d[..., 45], cmap = 'Spectral_r')

    plt.show()

    return

def testpm():
    c  = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )
    pd = InitialCondition(100.0, 48, c)(5)

    sim = ParticleMeshSimulation( pd, cm = c  )

    fig = plt.figure(figsize = [8,8])
    ax = fig.add_subplot(111, projection='3d')

    col = [[0.0, 0.0, 0.0]]#sim.dens
    ax.scatter(sim.currentPos[:,0], sim.currentPos[:,1], sim.currentPos[:,2], s = 1, cmap = 'hot', c = col, alpha = 0.2)
    plt.pause(0.01)

    a  = np.logspace( np.log10(sim.a), 0, 201 )

    for da in np.diff( a ):
        sim.updateParticles(da)

        ax.cla()
        col = sim.dens
        ax.scatter(sim.currentPos[:,0], sim.currentPos[:,1], sim.currentPos[:,2], s = 1, cmap = 'hot', c = col, alpha = 0.2)

        for __edge in pd.boundindgBox.edges:
            ax.plot( __edge[:,0], __edge[:,1], __edge[:,2], color = 'black', lw = 1 )

        ax.set_title( f"{sim.a:.3f}" )

        plt.pause(0.01)

        # break

    # d = densityCloudInCell( sim.currentPos, sim.boxsize, 128 )

    # plt.pcolor(d[..., 45], cmap = 'Spectral_r')

    plt.show()

    return

testpm()
# testpp()