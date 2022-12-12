#!/usr/bin/python3

import numpy as np, pandas as pd
from typing import Any, Callable, Union

# module: angle
class angle:
    """
    Angle conversion functions.
    """
    
    def radian(deg: float = 0., arcmin: float = 0., arcsec: float = 0.) -> float:
        r"""
        Convert and angle from degree to radians.

        Parameters
        ----------
        deg: float
            Angle to convert in degree, or the degree part of the angle to convert.
        arcmin: float
            Arcminut part of the angle to convert.
        arcsec: float
            Arcsecond part of the angle to convert.

        Returns
        -------
        rad: float
            Angle in radians.

        """

        deg = deg + arcmin / 60. + arcsec / 3600.
        rad = deg * np.pi / 180.
        return rad

    def degree(rad: float = 0.) -> float:
        r"""
        Convert angle from radians to degree.

        Parameters
        ----------
        rad: float
            Angle to convert in radians.

        Returns
        -------
        deg: float
            Angle in degrees.

        """

        deg = rad * 180. / np.pi
        return deg

    def asfraction(deg: float) -> tuple:
        r"""
        Split an angle in degrees to degree, minute and second parts
        """

        value = deg

        degree = np.floor(value)
        value  = 60*(value - degree)
        arcmin = np.floor(value)
        arcsec = 60*(value - arcmin)

        return degree, arcmin, arcsec





# module: distances
class distances:
    """
    Distance functions.
    """
    
    def euclidian(x: tuple, y: tuple) -> float:
        r"""
        Calculate the Euclidian distance between two points in 2D or 3D.

        Parameters
        ----------
        x, y: tuple
            Coordinates of the points. Must tuples of length 2 or 3.

        Returns
        -------
        dist: float
            Distance between the points.

        """

        ndim = len(x)
        if len(y) != ndim:
            raise TypeError("x and y should have same size")
        if ndim not in [2, 3]:
            print(x, y)
            raise TypeError("x and y should be 2 or 3-tuples")

        dist = (x[0] - y[0])**2 + (x[1] - y[1])**2
        if ndim == 3:
            dist += (x[2] - y[2])**2
        dist = np.sqrt( dist )
        return dist

    def haversine(x: tuple, y: tuple) -> float:
        r"""
        Calculate the haversine distance between two points on a sphere, given the 
        angular coordinates in radians. For the surface of earth the coordinates 
        will be lattitude and longitude, and for sky it is declination and right 
        ascension, in that order.

        Parameters
        ----------
        x, y: tuple
            Coordinates of the points. Must be a tuple of length 2 and the angles are in
            radians.

        Returns
        -------
        dist: float
            Angular distance between the points in radians.

        """

        if len(x) != 2 or len(y) != 2:
            raise TypeError("x and y should be 2-tuples")

        s0, s1 = np.sin( 0.5*(x[0] - y[0]) ), np.sin( 0.5*(x[1] - y[1]) )
        cx, cy = np.cos(x[0]), np.cos(y[0])

        dist = s0**2 + cx * cy * s1**2
        dist = np.sqrt(dist)
        dist = 2*np.arcsin(dist)
        return dist

    def haversine_deg(x: tuple, y: tuple) -> float:
        r"""
        Calculate the haversine distance between two points, given the angles in degree. 
        See `distances.haversine` for order of corrdinates.

        Parameters
        ----------
        x, y: tuple
            Coordinates of the points. Must be a tuple of length 2 and the angles are in
            degrees.
        
        Returns
        -------
        dist: float
            Angular distance between the points in degrees.
        """

        x, y = tuple( map( angle.radian, x ) ), tuple( map( angle.radian, y ) )
        dist = distances.haversine(x, y)
        return angle.degree(dist)

    available = ['euclidian', 'haversine', 'haversine_deg']

    def pairwise(pts: Any, metric: Union[str, Callable] = 'euclidian') -> Any:
        r"""
        Calculate the pairwise distance array for a list of points.

        Parameters
        ----------
        pts: array_like
            List of points. Should be an array of shape (N,2) or (N,3).
        metric: str, callable, optional
            Distance function. It can be a string name of any available function or a 
            python callable of two tuple arguments. Default value is `euclidian`.
        """

        if isinstance(metric, str):

            if metric not in distances.available:
                raise ValueError(f"metric '{metric}' is not available")

            metric = distances.__dict__[metric]

        if not callable(metric):
            raise TypeError("'metric' argument should be a 'str' or 'callable'")

        pts = np.asfarray(pts)

        assert np.ndim(pts) == 2 

        dxy = np.asfarray([[metric(x, y) for x in pts] for y in pts])
        return dxy

    def get(metric: str) -> Callable:
        r"""
        Get the distance function with given key.
        """

        if metric == 'euclidian':
            return distances.euclidian

        if metric == 'haversine':
            return distances.haversine
        
        if metric == 'haversine_deg':
            return distances.haversine_deg
        
        raise ValueError(f"metric '{metric}' is not available")


###############################################################################################################

def filterCatalog(df: pd.DataFrame, *, c1: str = None, c2: str = None, c3: str = None, mask: str = None, 
                  z: str = None, mag: str = None, zrange: tuple = None, magnitude_range: tuple = None,
                  mask_val: bool = None, c1_range: tuple = None, c2_range: tuple = None, c3_range: tuple = None) -> pd.DataFrame:
    r"""
    Filter a galaxy / object catalog.

    Parameters
    ----------
    df: pandas.DataFrame
        A table containing object data.
    c1, c2, c3: str, optional
        Columns specifiying the coordinates of the objects. e.g., x, y, z for cartetian and ra, dec for sky 
        coordinates.
    mask: str, optional
        If given, specifies the mask value column for the objects. The column values must be boolean.
    z: str, optional
        Column specifying the redshift.
    mag: str, optional
        Column specifying the magnitude values.
    zrange: tuple, optional 
        Range of redshift values of the objects. `z` argument should be given in this case.
    magnitude_range: tuple, optional
        Range of magnitude values of the objects. `mag` argument should be given in this case.
    mask_val: bool, optional
        Mask values of the objects in the returned dataframe. `mask` argument should be given in this case.
    c1_range, c2_range, c3_range: tuple, optional
        Range of object's position. `c1`, `c2`, `c3` arguments should be given in this case.

    Returns
    -------
    df: pandas.DataFrame
        New dataframe with filtered values.

    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a 'pandas.DataFrame' object")

    conditions = []

    if c1_range is not None:

        if not isinstance(c1, str):
            raise TypeError("c1 must be 'str'")
        if c1 not in df.columns:
            raise ValueError(f"no column with name '{c1}' in df")
        if np.size(c1_range) != 2:
            raise ValueError("c1_range should be a 2-tuple")

        conditions.append(f"( @c1_range[0] <= {c1} <= @c1_range[1] )")

    if c2_range is not None:

        if not isinstance(c2, str):
            raise TypeError("c2 must be 'str'")
        if c2 not in df.columns:
            raise ValueError(f"no column with name '{c2}' in df")
        if np.size(c2_range) != 2:
            raise ValueError("c2_range should be a 2-tuple")

        conditions.append(f"( @c2_range[0] <= {c2} <= @c2_range[1] )")

    if c3_range is not None:

        if not isinstance(c3, str):
            raise TypeError("c3 must be 'str'")
        if c3 not in df.columns:
            raise ValueError(f"no column with name '{c3}' in df")
        if np.size(c3_range) != 2:
            raise ValueError("c3_range should be a 2-tuple")

        conditions.append(f"( @c3_range[0] <= {c3} <= @c3_range[1] )")

    if zrange is not None:

        if not isinstance(z, str):
            raise TypeError("z must be 'str'")
        if z not in df.columns:
            raise ValueError(f"no column with name '{z}' in df")
        if np.size(zrange) != 2:
            raise ValueError("zrange should be a 2-tuple")

        conditions.append(f"( @zrange[0] <= {z} <= @zrange[1] )")

    if magnitude_range is not None:

        if not isinstance(mag, str):
            raise TypeError("mag must be 'str'")
        if mag not in df.columns:
            raise ValueError(f"no column with name '{mag}' in df")
        if np.size(magnitude_range) != 2:
            raise ValueError("magnitude_range should be a 2-tuple")

        conditions.append(f"( @magnitude_range[0] <= {mag} <= @magnitude_range[1] )")

    if mask_val is not None:

        if not isinstance(mask, str):
            raise TypeError("mask must be 'str'")
        if mask not in df.columns:
            raise ValueError(f"no column with name '{mask}' in df")
        if mask not in df.columns:
            raise ValueError(f"no column with name '{mask}' in df")

        conditions.append(f"( {mask} == {mask_val} )")

    if not len(conditions):
        return df

    q  = ' and '.join(conditions)
    df = df.query(q)
    return df
        

###############################################################################################################


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



