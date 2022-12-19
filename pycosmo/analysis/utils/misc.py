#!/usr/bin/python3
import numpy as np, pandas as pd


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
        
        