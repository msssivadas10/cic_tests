import numpy as np
import h5py, os

path = os.path.join( os.path.split( __file__ )[0], 'plank18' )

with h5py.File( os.path.join( path, 'snapshot_004.hdf5' ), 'r' ) as f:
    pos = np.asfarray( f['PartType1']['Coordinates'] )
    boxsize = f['Header'].attrs['BoxSize'] 
    redshift = f['Header'].attrs['Redshift'] 

from pycosmo.nbodytools.estimators.density import densityCloudInCell
from pycosmo.cosmology import Cosmology
from pycosmo.distributions.density_field import GenExtremeDistribution, genextreem

gridsize = 50


density = densityCloudInCell( pos, boxsize, gridsize )

dens = density.overdensity + 1
dens = dens.flatten()
dens = np.log( dens[ dens > 0.0 ] )

import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


# fig = plt.figure(figsize = [8,8])
# ax = fig.add_subplot(111, projection='3d')

# col = [[0.0, 0.0, 0.0]]#sim.dens
# ax.scatter(
#                 pos[:,0], pos[:,1], pos[:,2], 
#                 s = 1, cmap = 'hot', c = col, alpha = 0.2
#           )
# plt.show()

# print( redshift )

plt.figure()
plt.hist( dens, bins = 31, range = [-2, 2], histtype = 'step', density = True )


# c = Cosmology( 0.7, 0.30, 0.048, 1.0, 0.9661, power_spectrum = 'eisenstein98_zb' )
# p = GenExtremeDistribution( c, density.cellsize  )
# x = np.linspace( -3, 3, 51 )
# y = p.pdf( x, z = redshift )
# plt.plot( x, y, 'o-', ms = 4 )

# print( p.param )

mean, var = np.mean( dens ), np.var( dens ), 
skew = np.mean( ( dens - mean )**3 ) / var**1.5
# print( mean, var, skew )

from scipy.stats import genextreme as gev, lognorm as testdist
from scipy.special import gamma

a, b, c = testdist.fit( dens )

shape, loc, scale = gev.fit( dens )
shape = -shape

# print( shape, loc, scale )

# print( mean, loc + scale / shape * ( gamma(1-shape) - 1 ) )
# print( var, scale**2 / shape**2 * ( gamma(1-2*shape) - gamma(1-shape)**2 ) )
# print( skew, -( gamma(1-3*shape) - 3*gamma(1-shape)*gamma(1-2*shape) + 2*gamma(1-shape)**3 ) / ( gamma(1-2*shape) - gamma(1-shape)**2 )**1.5 )


print( mean, var, skew )
print( gev.stats( -shape, loc, scale, moments = 'mvs' ) )
print( testdist.stats( a, b, c, moments = 'mvs' ) )


xx = np.linspace(-2, 2, 201)
yy = gev.pdf(xx, -shape, loc, scale)
zz = testdist.pdf( xx, a, b, c )

plt.plot( xx, yy, '-', label = 'GEV' )
plt.plot( xx, zz, '-', label = 'testdist' )

plt.legend()
plt.show()

