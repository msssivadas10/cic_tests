import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pycosmo.cosmology import Cosmology
from pycosmo.nbodytools.simulation import InitialCondition, ParticleMeshSimulation
from pycosmo.nbodytools.estimators.density import densityCloudInCell
# from pycosmo.nbodytools.estimators.power_spectrum import powerSpectrum

plt.style.use( 'ggplot' )


def test1():
    c   = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )
    pd  = InitialCondition(300.0, 32, c)(100)
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



    plt.figure(figsize = [8,5])

    plt.gca().tick_params(axis='both', which='major', labelsize=14)

    color = [ 'tab:blue', 'tab:green', 'orange' ]

    for i, ( x, a ) in enumerate( zip( xout, aout ) ):
        # plt.loglog()
        # plt.plot( k, p, label = f"a = {a:.3f}" )

        plt.hist( 
                    -x, bins = 51, density = True, range = [-4, 6], rwidth = 0.9, 
                    histtype = 'step', 
                    color = color[i], lw = 1.5, label = f"{1/a-1:.1f}" 
                )

    plt.legend(title = "Redshift", fontsize = 14, title_fontsize = 14)
    plt.xlabel('A = $\\ln (1+\\delta)$', fontsize = 14)
    plt.ylabel('P(A)', fontsize = 14)
    plt.show()

def test2():
    c   = Cosmology( 0.7, 0.3, 0.05, 0.8, 1.0, power_spectrum = 'eisenstein98_zb' )
    pd  = InitialCondition(300.0, 64, c)(100)
    sim = ParticleMeshSimulation( pd, cm = c  )

    xout, aout = [], []

    a1 = np.logspace( np.log10(sim.a), 0, 101 )
    a2 = ( 1 + np.asfarray([50.0, 20.0, 0.0]) )**-1
    a  = np.unique( np.hstack([ a1, a2 ]) )
    a.sort()

    i = np.searchsorted( a, a2 ) - 1

    # fig = plt.figure(figsize = [8,8])
    # ax = fig.add_subplot(111, projection='3d')

    # col = [[0.0, 0.0, 0.0]]#sim.dens
    # ax.scatter(
    #                 sim.currentPos[:,0], sim.currentPos[:,1], sim.currentPos[:,2], 
    #                 s = 1, cmap = 'hot', c = col, alpha = 0.2
    #           )
    # plt.pause(0.01)

    for j in range( a.shape[0]-1 ):
        da = a[j+1] - a[j]

        sim.updateParticles(da)

        print( a[j] )

        # ax.cla()
        # col = sim.dens
        # ax.scatter(
        #                 sim.currentPos[:,0], sim.currentPos[:,1], sim.currentPos[:,2], 
        #                 s = 1, cmap = 'hot', c = col, alpha = 0.2
        #         )

        # for __edge in pd.boundindgBox.edges:
        #     ax.plot( __edge[:,0], __edge[:,1], __edge[:,2], color = 'black', lw = 1 )

        # ax.set_title( f"{sim.a:.3f}" )
        # ax.set( xlim = [0, sim.boxsize], ylim = [0, sim.boxsize], zlim = [0, sim.boxsize] )

        # plt.pause(0.01)

        if j in i:
            ...


    # plt.show()

    fig, (ax1, ax2) = plt.subplots( 1, 2, figsize = [12,6] )

    ax1.tick_params(axis='both', which='major', labelsize=14)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    x, y = np.mgrid[ 0.0:sim.boxsize:128j, 0.0:sim.boxsize:128j ]
    dens = densityCloudInCell( sim.currentPos, sim.boxsize, 128 )
    ax1.pcolor( x, y, dens[..., 64], cmap = 'Spectral_r' )

    x, y = np.mgrid[ 0.0:sim.boxsize:64j, 0.0:sim.boxsize:64j ]
    dens = densityCloudInCell( sim.currentPos, sim.boxsize, 64 )
    ax2.pcolor( x, y, dens[..., 32], cmap = 'Spectral_r' )

    plt.show()


test2()