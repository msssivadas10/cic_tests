import numpy as np
import h5py, os
from scipy.stats import genextreme 
from pycosmo.nbodytools.estimators.density import densityCloudInCell
from dataclasses import dataclass

@dataclass
class ParticleData:
    pos: np.ndarray
    boxsize: float
    redshift: float

@dataclass
class DistributionData:
    shape: float
    loc: float
    scale: float
    mean: float 
    var: float
    skew: float
    redshift: float
    cellsize: float
    meanL: float
    varL: float

def readParticleData(fnum):
    path = os.path.join( os.path.split( __file__ )[0], 'plank18' )
    with h5py.File( os.path.join( path, f'snapshot_00{ fnum }.hdf5' ), 'r' ) as f:
        pos = np.asfarray( f['PartType1']['Coordinates'] )
        boxsize = f['Header'].attrs['BoxSize'] 
        redshift = f['Header'].attrs['Redshift'] 
    return ParticleData(pos, boxsize, redshift)

def getBestfit(pdata, gridsize):
    density = densityCloudInCell( pdata.pos, pdata.boxsize, gridsize )
    dens    = ( density.overdensity ).flatten()

    meanL, varL = np.mean( dens ), np.var( var )

    dens    = np.log( dens[ dens > 0.0 ] + 1 ) # log field

    # best fitting statistics for `genextreme`:
    shape, loc, scale = genextreme.fit( dens ) 
    mean, var, skew   = genextreme.stats( shape, loc, scale, moments = 'mvs' )
    return DistributionData(shape, loc, scale, mean, var, skew, pdata.redshift, density.cellsize, meanL, varL)


def main():
    table = []
    for fnum in range(6): 
        pdata = readParticleData(fnum)
        for gridsize in [ 8, 16, 24, 32, 48 ]:
            print(f"DataFile: { fnum } Gridsize: { gridsize }")
            d = getBestfit(pdata, gridsize)
            table.append([d.redshift, d.cellsize, d.shape, d.loc, d.scale, d.mean, d.var, d.skew])


    table  = np.asfarray(table)
    header = '# Redshift, Cellsize, Shape, Loc, Scale, Mean, Variance, Skewness'
    np.savetxt('density.txt', table, delimiter = ',', header = header)

if __name__ == "__main__":
    main()


