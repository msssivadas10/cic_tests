#!/usr/bin/python3

import numpy as np, pandas as pd
import warnings
from scipy.optimize import newton
from sklearn.neighbors import BallTree
from pycosmo.analysis.helpers import angle, distances, BoundingBox
from typing import Any, Iterable, Callable
from abc import ABC, abstractmethod

###############################################################################################
# base class for cell objects
###############################################################################################

class Cell(ABC):
    """
    An abstract cell class for count in cells.
    """

    @property
    @abstractmethod
    def distance(self) -> Callable:
        r"""
        Distnce function. A callable accepting two tuples ans arguments.
        """
        ...
    
    @abstractmethod
    def intersect(self, other: 'Cell') -> bool:
        """
        Check where this cell intersect with another cell.
        """
        ...

    def intersect_any(self, others: Iterable['Cell']) -> bool:     
        """
        Check if this cell intersect with any other cell in the given list.
        """
        for other in others:
            if self.intersect(other):
                return True
        return False

    @abstractmethod
    def is_inside(self, point: tuple) -> bool:
        """
        Tell if a point is inside or outside the cell.
        """
        ...

    @property
    @abstractmethod
    def bbox(self) -> BoundingBox:
        """
        Get the bounding box of the cell.
        """
        ...

    def is_inside_region(self, region: BoundingBox) -> bool:
        """
        Tell if the region is inside a given region.
        """
        return self.bbox.is_inside( region )

    @abstractmethod
    def verts(self, res: int = 50) -> Any:
        """
        Get the vertices of the polygon representing the cell. The `res` parameter tells 
        the resolution of the curve.
        """
        ...


######################################################################################################
# Cell in 2D: projected or angular space / sky with haversine metric
######################################################################################################

class Circle(Cell):
    r"""
    A circular cell in the sky sphere. The circle is defined the lattitude-longitude space (or 
    declination-right ascension space) and all the angles are in radians. haversine distance is 
    used as the metric.

    Parameters
    ----------
    center: tuple
        Center of the circle as lattitude, longitude (or dec, ra) pair. Angles are in radians. The 
        lattitude (or dec) must be in the range :math:`[-\pi/2, \pi/2]` and the longitude in the 
        range :math:`[0, 2\pi]`.
    radius: float
        Angular radius in radians. Must be a positive value, less than :math:`\pi/2`.

    """

    __slots__ = 'center', 'radius', 

    @staticmethod
    def _f(d: Any, d0: float, r: float) -> Any:

        y = 0.5*np.sin(d-d0) - np.tan(d)*( np.sin(0.5*r)**2 - np.sin(0.5*(d-d0))**2 )
        return y

    @staticmethod
    def _dra(d: Any, d0: float, r: float) -> Any:

        da = ( np.sin(0.5*r)**2 - np.sin(0.5*(d-d0))**2 ) / ( np.cos(d)*np.cos(d0) )
        da = np.where( np.abs(da) < 1e-12, 0., da )
        da = np.sqrt(da)
        da = 2*np.arcsin(da)
        return da

    def __init__(self, center: tuple, radius: float) -> None:
        center = tuple(center)

        assert radius > 0.
        assert len(center) == 2
        assert 0 <= center[1] <= 2*np.pi 
        assert -0.5*np.pi <= center[0] <= 0.5*np.pi

        self.center, self.radius = center, radius

    def __repr__(self) -> str:
        return f"Circle(center={self.center}, radius={self.radius})"

    @property
    def distance(self) -> Callable:
        return distances.haversine

    @property
    def bbox(self) -> BoundingBox:
        
        (cd, ca), radius = self.center, self.radius

        dm = newton(self._f, cd, args=(cd, radius))
        al, ar = self.ra(dm)

        return BoundingBox(ndim = 2, lower = [cd-radius, al], upper = [cd+radius, ar])

    def intersect(self, other: 'Circle') -> bool:
        d = self.distance(self.center, other.center) - (self.radius + other.radius)
        return np.where( np.abs(d) < 1.0e-12, False, d < 0. )

    def is_inside(self, point: tuple) -> bool:
        return self.distance(self.center, point) <= self.radius

    def ra(self, dec: Any) -> tuple:
        r"""
        Get the right ascension or longitude as a function of declination or lattitude.

        Parameters
        ----------
        dec: array_like
            Declination or lattitude in radians.
        
        Returns
        -------
        ra_upper, ra_lower: array_like
            Upper and lower right ascension values.

        """

        (cd, ca), radius = self.center, self.radius

        dec = np.asfarray(dec)
        da  = self._dra(dec, cd, radius)
        return ca - da, ca + da

    def verts(self, res: int = 50) -> Any:
        
        (cd, ca), radius = self.center, self.radius

        if radius > np.pi/2:
            raise ValueError("radius must be less than pi/2 for verts to calculate")

        # TODO: verts for center dec == pi/2

        d = np.linspace( cd-radius, cd+radius, res )
        al, ar = self.ra(d)

        d = np.hstack([d, d[::-1]])
        a = np.hstack([al, ar[::-1]])
        return np.stack( (d, a), axis=1 )


class Square(Cell):
    r"""
    A square cell on a sphere.
    """
    ...

######################################################################################################
# Cells in 3D: real space with euclidian metric
######################################################################################################

class Sphere(Cell):
    r"""
    A spherical cell in real (3D cartetian) space.
    """

    __slots__ = 'center', 'radius', 

    def __init__(self, center: tuple, radius: float) -> None:
    
        center = tuple(center)

        assert radius > 0.
        assert len(center) == 3

        self.center, self.radius = center, radius

    def __repr__(self) -> str:
        return f"Sphere(center={self.center}, radius={self.radius})"

    @property
    def distance(self) -> Callable:
        return distances.euclidian

    @property
    def bbox(self) -> BoundingBox:
        (cx, cy, cz), r = self.center, self.radius
        return BoundingBox(ndim = 3, lower = [cx-r, cy-r, cz-r], upper = [cx+r, cy+r, cz+r])
    
    def intersect(self, other: 'Sphere') -> bool:
        d = self.distance(self.center, other.center) - (self.radius + other.radius)
        return np.where( np.abs(d) < 1.0e-12, False, d < 0. )

    def is_inside(self, point: tuple) -> bool:
        return self.distance(self.center, point) <= self.radius

    def verts(self, res: int = 50) -> Any:
        raise NotImplementedError("verts method is not implemented for 3D cells")


class Cube(Cell):
    r"""
    A Cubic cell in real (3D cartetian) space. 
    """
    
    __slots__ = 'center', 'size'

    def __init__(self, center: tuple, size: float) -> None:
        
        center = tuple(center)

        assert size > 0.
        assert len(center) == 3

        self.center, self.size = center, size
    
    def __repr__(self) -> str:
        return f"Cube(center={self.center}, size={self.size})"

    @property
    def distance(self) -> Callable:
        return distances.euclidian

    @property
    def bbox(self) -> BoundingBox:
        (cx, cy, cz), r = self.center, self.size
        return BoundingBox(ndim = 3, lower = [cx-r, cy-r, cz-r], upper = [cx+r, cy+r, cz+r])
    
    def intersect(self, other: 'Cube') -> bool:

        self_c, self_r = self.center, self.size

        self_xl, self_xr = self_c[0] - self_r, self_c[0] + self_r
        self_yl, self_yr = self_c[1] - self_r, self_c[1] + self_r
        self_zl, self_zr = self_c[2] - self_r, self_c[2] + self_r

        other_c, other_r = other.center, other.size

        other_xl, other_xr = other_c[0] - other_r, other_c[0] + other_r
        other_yl, other_yr = other_c[1] - other_r, other_c[1] + other_r
        other_zl, other_zr = other_c[2] - other_r, other_c[2] + other_r

        return (other_xl < self_xr and other_xr > self_xl
                    and other_yl < self_yr and other_yr > self_yl
                    and other_zl < self_zr and other_zr > self_zl)

    def is_inside(self, point: tuple) -> bool:

        assert len(point) == 3

        delta = np.abs(np.subtract(point, self.center))
        return min(delta <= 0.5*self.size)

    def verts(self, res: int = 50) -> Any:
        raise NotImplementedError("verts method is not implemented for 3D cells")


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
    ...
