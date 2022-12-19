#!/usr/bin/python3

import numpy as np, pandas as pd
import warnings
from sklearn.neighbors import BallTree
from pycosmo.analysis.cic.objects import BoundingBox, Sphere, Cube
from typing import Any

##################################################################################################
# Count in cells in 3D (real space)
##################################################################################################

# tiling cells
def tile_cubes3d(region: BoundingBox, size: float, packing: str = 'cubic', filter: bool = False, 
                 df: pd.DataFrame = None, *, x: str = None, y: str = None, z: str = None, 
                 mask: str = None, selection_frac: float = 0.9, check_overlap: bool = False) -> list:
    r"""
    Tile the 3D region with cubic cells/
    """

    if packing != 'cubic':
        raise ValueError("only 'cubic' packing is supported for cubes")

    # creating the cells
    

    # selecting the cells
    

    raise NotImplementedError()

def tile_spheres3d(region: BoundingBox, radius: float, packing: str = 'cubic', filter: bool = False, 
                 df: pd.DataFrame = None, *, x: str = None, y: str = None, z: str = None, mask: str = None, 
                 selection_frac: float = 0.9, bt_leafsize: int = 2, check_overlap: bool = False) -> list:
    r"""
    Tile the 3D region with sphere cells.

    Parameters
    ----------
    region: BoundingBox
        Region to tile the cells. Must be a 3D `BoundingBox` object, with coordinates in the order of 
        x, y, z. 
    radius: float
        Radius of the spheres.
    packing: str, optional
        Tells which way to pack the spheres. `cubic` is the only availabe option now, which is for cubical 
        packing.
    filter: bool, optional
        Tells whether to filter the cells based on the availablity of data. This is used for regions with 
        holes. In that case, cells overlapping the valid region are selected only.
    df: pandas.DataFrame, optional
        A dataframe containing region information. This will be a set of random points in the region, where 
        the points in the holes are labelled with a mask value. This should be given if filter option is 
        set.
    x, y, z: str
        Names of the x, y, z coordinates column in dataframe df.
    mask: str
        Name of the mask value column in dataframe df.
    selection_frac: float, optional
        Fraction of non-masked points that a cell must have to be selected. This should be a number between 
        0 and 1 (default is 0.9).
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
    assert region.ndim == 3

    x1, y1, z1 = region.lower
    x2, y2, z2 = region.upper

    cells = []
    if packing == 'cubic':

        x = np.arange(x1, x2, 2*radius) + radius
        y = np.arange(y1, y2, 2*radius) + radius
        z = np.arange(z1, z2, 2*radius) + radius

        x, y, z = np.meshgrid(x, y, z)
        centers = np.stack([x.flatten(), y.flatten(), z.flatten()], axis=1)

        for center in centers:

            cell = Sphere( center = center, radius = radius )

            select_cell = True
            if not cell.is_inside_region( region ):
                select_cell = False
            if check_overlap:
                if cell.intersect_any(cells):
                    select_cell = False
            if select_cell:
                cells.append(cell)
    
    elif packing == 'hexagonal':
        raise NotImplementedError("hexagonal packing is not implemented")

    else:
        raise ValueError("packing must be either 'cubic' or 'hexagonal'")

    if not filter:
        return cells

    # selecting the cells
    if not isinstance(df, pd.DataFrame):
        raise TypeError("object data 'df' must be a 'pandas.DataFrame' object")

    for _key, _value in {'x': x, 'y': y, 'z': z, 'mask': mask}.items():
        if _value is None:
            raise ValueError(f"name of the '{_key}' column should be given")
        elif not isinstance(_value, str):
            raise TypeError(f"{_key} must be a 'str'")

    coords   = df[[x, y, z]].to_numpy() 
    mask_val = df[mask].to_numpy().astype('bool').flatten()

    if not 0. <= selection_frac <= 1.:
        raise ValueError("selection fraction must be a number between 0 and 1")
    rejection_frac = 1. - selection_frac

    bt = BallTree(coords, leaf_size = bt_leafsize, metric = 'euclidean')
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

# get count in cells
def get_counts3d():
    raise NotImplementedError()
