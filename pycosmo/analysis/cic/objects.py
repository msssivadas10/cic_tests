#!/usr/bin/python3
import numpy as np
from scipy.optimize import newton
from pycosmo.analysis.utils import distances
from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable


#############################################################################################
# base class for object bounding boxes
#############################################################################################

class BoundingBox:
    """
    A rectangular bounding box, defined by the lower and upper limits of the coordinates.
    """

    __slots__ = 'lower', 'upper', 'ndim'

    def __init__(self, ndim: int, lower: tuple, upper: tuple) -> None:

        ndim = int(ndim)
        assert ndim > 1

        self.ndim = ndim
        
        if len(lower) != ndim or len(upper) != ndim:
            raise ValueError(f"lower and upper should have size {ndim}")

        for i in range(ndim):
            if lower[i] > upper [i]:
                raise ValueError(f"lower limit cannot be greater than upper limit (coordinate {i})")

        self.lower, self.upper = lower, upper

    def __repr__(self) -> str:

        r = []
        for i in range(self.ndim):
            r.append( f"c{i}_min={self.lower[i]}, c{i}_max={self.upper[i]}" )
        r = ', '.join(r)
        return f"BoundingBox({r})"

    def is_inside(self, other: 'BoundingBox') -> bool:
        """
        Tell if the bounding box object is inside another.
        """

        if self.ndim != other.ndim:
            raise ValueError("both objects should have same dimension")

        res = True
        for i in range(self.ndim):
            res = res and (self.lower[i] >= other.lower[i] and self.upper[i] <= other.upper[i])

        return res


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


##########################################################################################
# cell generation methods
##########################################################################################

def create_cell(geom: str, center: tuple, size: float) -> Cell:
    r"""
    Create a cell with given parameters.

    Parameters
    ----------
    geom: str
        Name of the cell. Available cells are `circle` in 2D and `sphere` and `cube` in 3D.
    center: tuple
        Center of the cell. For 2D cells, it should be a 2-tuple and for 3D, a 3-tuple.
    size: float
        Size of the cell (radius for sphere / circle and side length for boxes).

    Returns
    -------
    cell: Cell or its subclasses
        A cell object.

    """
    
    if geom == 'circle':
        return Circle(center, size)
    
    # if geom == 'square':
    #     return Square(center, size)

    if geom == 'sphere':
        return Sphere(center, size)

    if geom == 'cube':
        return Cube(center, size)

    raise NotImplementedError(f"cell not implemented: '{geom}'")

