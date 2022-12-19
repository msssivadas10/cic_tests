import numpy as np
from typing import Any, Callable, Union
from pycosmo.analysis.utils import angle


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
    dist = haversine(x, y)
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

        if metric not in available:
            raise ValueError(f"metric '{metric}' is not available")

        metric = get_metric(metric)

    if not callable(metric):
        raise TypeError("'metric' argument should be a 'str' or 'callable'")

    pts = np.asfarray(pts)

    assert np.ndim(pts) == 2 

    dxy = np.asfarray([[metric(x, y) for x in pts] for y in pts])
    return dxy

def get_metric(metric: str) -> Callable:
    r"""
    Get the distance function with given key.
    """

    if metric == 'euclidian':
        return euclidian

    if metric == 'haversine':
        return haversine
    
    if metric == 'haversine_deg':
        return haversine_deg
    
    raise ValueError(f"metric '{metric}' is not available")