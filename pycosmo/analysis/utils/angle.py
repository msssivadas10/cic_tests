import numpy as np
from collections import namedtuple

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

def degree(rad: float = 0., split: bool = False) -> float:
    r"""
    Convert angle from radians to degree.

    Parameters
    ----------
    rad: float
        Angle to convert in radians.
    split: bool
        Tell whether to split the value in degree, arcminute and arcsecond.

    Returns
    -------
    deg: float or namedtuple
        Angle in degrees. This will be a float if split option is not set. Otherwise, it 
        will be namedtuple with fields `deg`, `arcmin` and `arcsec`.

    """

    deg = rad * 180. / np.pi

    if not split:
        return deg

    degree_angle = namedtuple('degree_angle', ['deg', 'arcmin', 'arcsec'])

    value = deg

    degree = np.floor(value)
    value  = 60*(value - degree)
    arcmin = np.floor(value)
    arcsec = 60*(value - arcmin)

    return degree_angle(degree, arcmin, arcsec)