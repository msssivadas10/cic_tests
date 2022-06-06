from typing import Any 
import numpy as np
import numpy.random as rnd

from scipy.spatial import cKDTree

def correlation(r: Any, pos: Any, pos_r: Any = None, weight: Any = None, mode: str = 'PH') -> Any:
    pos   = np.asfarray( pos )
    treeD = cKDTree( pos )

    if pos_r is None:
        N       = pos.shape[0]
        boxsize = np.max( pos, axis = 0 )
        pos_r   = rnd.uniform( 0.0, 1.0, pos.shape ) * boxsize 
    treeR = cKDTree( pos_r )

    DD = treeD.count_neighbors( treeD, r, weights = weight, cumulative = False )

    RR = treeR.count_neighbors( treeR, r, weights = weight, cumulative = False )

    f  = pos.shape[0] / pos_r.shape[0]

    if mode == 'PH':
        return DD / RR / f**2 - 1

    DR = treeD.count_neighbors( treeR, r, weights = weight, cumulative = False )

    if mode == 'LS':
        return ( DD - 2*f*DR + f**2 * RR ) / ( f**2 * RR )

    raise ValueError(f"invalid mode: '{mode}'")


