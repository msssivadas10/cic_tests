#!/usr/bin/python3

import numpy as np, pandas as pd
import warnings
from scipy.optimize import newton
from sklearn.neighbors import BallTree
from pycosmo.analysis.utils import angle
from pycosmo.analysis.cic.objects import BoundingBox, Circle, Square
from typing import Any

##################################################################################################
# Count in cells in 2D
##################################################################################################

# tiling cells
def tile_circles2d(region: BoundingBox, radius: float, packing: str = 'cubic', filter: bool = False, 
                   df: pd.DataFrame = None, *, ra: str = None, dec: str = None, mask: str = None, 
                   selection_frac: float = 0.9, angle_unit: str = 'radian', bt_leafsize: int = 2, 
                   check_overlap: bool = False) -> list:
    r"""
    Tile the 2D region with circular cells of given radius. 

    Parameters
    ----------
    region: BoundingBox
        Region to tile the cells. Must be a 2D `BoundingBox` object, with coordinates in the order of 
        dec, ra. 
    radius: float
        Radius of the circles in radians.
    packing: str, optional
        Tells which way to pack the circles. `cubic` is the only availabe option now, which is for cubical 
        packing.
    filter: bool, optional
        Tells whether to filter the cells based on the availablity of data. This is used for regions with 
        holes. In that case, cells overlapping the valid region are selected only.
    df: pandas.DataFrame, optional
        A dataframe containing region information. This will be a set of random points in the region, where 
        the points in the holes are labelled with a mask value. This should be given if filter option is 
        set.
    ra: str
        Name of the right ascension/longitude column in dataframe df.
    dec: str
        Name of the declination/lattitude column in dataframe df.
    mask: str
        Name of the mask value column in dataframe df.
    selection_frac: float, optional
        Fraction of non-masked points that a cell must have to be selected. This should be a number between 
        0 and 1 (default is 0.9).
    angle_unit: str, optional
        Unit for angles in the dataframe df. Should be either `radian` or `degree`.
    bt_leafsize: int, optional
        Leaf size used for the ball tree in internal nearest neighbour search. Default is 2.
    check_overlap: bool, optional
        If set true, check if the cell overlap with any other cells in the region. This is time consuming 
        and is turned off by default.  

    Returns
    -------
    cells: list
        Cells in the region.

    """

    # creating the cells
    assert radius > 0.
    assert isinstance(region, BoundingBox)
    assert region.ndim == 2

    dec1, ra1 = region.lower
    dec2, ra2 = region.upper

    cells = []
    if packing == 'cubic':
        cd    = dec1 + radius
        while cd < dec2:

            dm = newton(Circle._f, cd, args = (cd, radius))
            da = Circle._dra(dm, cd, radius)
            ca = ra1 + da
            while ca < ra2:

                cell = Circle( center = [cd, ca], radius = radius )

                select_cell = True
                if not cell.is_inside_region( region ):
                    select_cell = False 
                if check_overlap:
                    if cell.intersect_any(cells):
                        select_cell = False
                if select_cell:
                    cells.append(cell)

                ca += 2*da
            cd += 2*radius

    elif packing == 'hexagonal':
        raise NotImplementedError("hexagonal packing is not implemented")

    else:
        raise ValueError("packing must be either 'cubic' or 'hexagonal'")

    if not filter:
        return cells

    # selecting the cells
    if not isinstance(df, pd.DataFrame):
        raise TypeError("object data 'df' must be a 'pandas.DataFrame' object")

    for _key, _value in {'ra': ra, 'dec': dec, 'mask': mask}.items():
        if _value is None:
            raise ValueError(f"name of the '{_key}' column should be given")
        elif not isinstance(_value, str):
            raise TypeError(f"{_key} must be a 'str'")

    coords   = df[[dec, ra]].to_numpy() 
    mask_val = df[mask].to_numpy().astype('bool').flatten()

    if not 0. <= selection_frac <= 1.:
        raise ValueError("selection fraction must be a number between 0 and 1")
    rejection_frac = 1. - selection_frac

    if angle_unit == 'degree':
        coords = angle.radian(coords)
    elif angle_unit != 'radian':
        raise ValueError(f"invalid angle unit: '{angle_unit}'")

    bt = BallTree(coords, leaf_size = bt_leafsize, metric = 'haversine')
    id = bt.query_radius([cell.center for cell in cells],
                        [cell.radius for cell in cells])


    selected_cells = []
    for cell, i in zip(cells, id):
        m = mask_val[i]

        if np.nan in m:
            warnings.warn("NaN in mask values (taken as 'False')")

        if np.mean(m) > rejection_frac:
            continue
        selected_cells.append(cell)

    return selected_cells

# tiling square cells
def tile_squares2d(*args, **kwargs) -> Any:
    raise NotImplementedError()

# get count in cells
def get_counts2d(cells: list, cell_geom: str, df: pd.DataFrame, *, ra: str = None, dec: str = None, 
                 angle_unit: str = 'radian', bt_leafsize: int = 2, return_index: bool = False) -> Any:
    r"""
    Find the count in cells of 2D (sky) objects in a given dataset.

    Parameters
    ----------
    cells: list of :class:`Cell`
        Cells to find the counts.
    cell_geom: str
        Cell geometry - either `circle` or `square` (not implemented yet).
    df: pandas.DataFrame
        A dataframe containing the objects in the given region.
    ra: str
        Name of the right ascension/longitude column in dataframe df.
    dec: str
        Name of the declination/lattitude column in dataframe df.
    angle_unit: str, optional
        Unit for angles in the dataframe df. Should be either `radian` or `degree`.
    bt_leafsize: int, optional
        Leaf size used for the ball tree in internal nearest neighbour search. Default is 2.
    return_index: bool, optional
        If set true, return the list of indices of objects in each cell also.

    Returns
    -------
    counts: array_like
        Count of objects in each cells.
    id: list of array_like
        Indices of objects in each cells. 
    
    """
    
    if not isinstance(df, pd.DataFrame):
        raise TypeError("object data 'df' must be a 'pandas.DataFrame' object")

    if not isinstance(ra, str):
        raise ValueError("name of the ra column should be a 'str'")
    if not isinstance(dec, str):
        raise ValueError("name of the dec column should be a 'str'")

    if not df.shape[0]:
        raise ValueError("empty data")

    coords = df[[dec, ra]].to_numpy()

    if angle_unit == 'degree':
        coords = angle.radian(coords)
    elif angle_unit != 'radian':
        raise ValueError(f"invalid angle unit: '{angle_unit}'")

    if not len(cells):
        raise ValueError("no cells are found in the list")

    if cell_geom == 'circle':
        bt = BallTree(coords, leaf_size = bt_leafsize, metric = 'haversine')
        id = bt.query_radius([cell.center for cell in cells],
                            [cell.radius for cell in cells])
                            
    elif cell_geom == 'square':
        raise NotImplementedError()

    else:
        raise ValueError(f"invalid type of cells '{cell_geom}'")
    
    counts = np.array([len(i) for i in id])

    if return_index:
        return counts, id
    return counts