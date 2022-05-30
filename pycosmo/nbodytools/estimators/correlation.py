from typing import Any 
import numpy as np
import numpy.random as rnd

from itertools import product, repeat

def correlationPH(pos: Any, boxsize: float, bins: int = 20, subdiv: int = 10, weight: Any = 1.0) -> Any:

    def createPointGrid(pos: Any, cellsize: float, subdiv: int):
        npart = pos.shape[0]
        pid   = np.arange( npart ) 

        # create grid from particles
        i = np.floor( pos / cellsize ).astype( 'int' )
        i = i[:,2] + subdiv * ( i[:,1] + subdiv * i[:,0] )

        G = { __i: pid[ i == __i ] for __i in np.unique(i) }
        return G

    def countPairs(x: Any, G: dict, pos: Any, bins2: Any, cellsize: float, subdiv: int, weight: Any) -> Any:
        i0, i1, i2 = np.floor( x / cellsize ).astype( 'int' )

        count = np.zeros( len(bins2)-1 )
        for j0, j1, j2 in product( *repeat( range(-1, 2), 3 ) ):
            j0, j1, j2 = i0 + j0, i1 + j1, i2 + j2
            j          = j2 + subdiv * ( j1 + subdiv * j0 )

            if j not in G:
                continue

            dist2  = np.sum( ( x - pos[ G[j], : ] )**2, axis = -1 )
            count += np.histogram( dist2, bins2, weights = weight[ G[j] ] )[0]

        return count

    pos = np.asfarray( pos )  
    if np.ndim( pos ) != 2:
        raise TypeError("pos must be a 2D array")
    elif pos.shape[1] != 3:
        raise TypeError("pos should have 3 columns")
    npart = pos.shape[0]

    cellsize = boxsize / subdiv

    if np.ndim( weight ) == 0:
        weight = np.repeat( weight, npart )
    else:
        if np.ndim( weight ) == 1 and weight.shape[0] != npart:
            raise TypeError("weight should have same length as particles")
        elif np.ndim( weight ) > 1:
            raise TypeError("weight must be a 1D array")
    
    mean_weight = np.repeat( np.mean( weight ), npart )

    # create signal grid
    D = createPointGrid( pos, cellsize, subdiv )
    
    # create random grid
    pos_r = rnd.uniform( 0.0, boxsize, ( npart, 3 ) )
    R     = createPointGrid( pos_r, cellsize, subdiv )

    r = np.linspace( 0.0, cellsize, bins+1 )

    DD = np.zeros( bins )
    for i in range( npart ):
        DD += countPairs( pos[i,:], D, pos, r**2, cellsize, subdiv, weight )

    RR = np.zeros( bins )
    for i in range( npart ):
        RR += countPairs( pos_r[i,:], R, pos_r, r**2, cellsize, subdiv, mean_weight )

    xi = 1 - DD / RR
    return r, xi

