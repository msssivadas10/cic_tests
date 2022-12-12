#!/usr/bin/python3

import numpy as np, pandas as pd
from pycosmo.analysis.cic import tile_circles2d, get_counts2d
from pycosmo.analysis.helpers import BoundingBox, filterCatalog
from typing import Any


class SkyRegion(BoundingBox):
    r"""
    Specify a region in the sky, defined by the right ascension and declination. 
    """

    __slots__ = '_cells', '_cell_geom', 'odf', 'rdf', 'attrs'

    def __init__(self, dec1: float, dec2: float, ra1: float, ra2: float) -> None:

        assert -np.pi/2 <= dec1 <= np.pi/2 and -np.pi/2 <= dec2 <= np.pi/2
        assert 0 <= ra1 <= 2*np.pi and 0 <= ra2 <= 2*np.pi

        super().__init__(ndim = 2, lower = (dec1, ra1), upper = (dec2, ra2))

        # cells 
        self._cells     = None
        self._cell_geom = None

        # object data
        self.odf, self.rdf = None, None
        self.attrs = {'ra': None, 'dec': None, 'mask': None, 'redshift': None, 'mag': None, 'angle_unit': 'radian'}

    @property
    def dec1(self) -> float: return self.lower[0]

    @property
    def ra1(self) -> float: return self.lower[1]

    @property
    def dec2(self) -> float: return self.upper[0]

    @property
    def ra2(self) -> float: return self.upper[1]

    def __repr__(self) -> str:
        return f"SkyRegion(dec1={self.dec1}, dec2={self.dec2}, ra1={self.ra1}, ra2={self.ra2})"

    # link data to the region

    def setdata(self, odf: pd.DataFrame, rdf: pd.DataFrame, ra: str = 'ra', dec: str = 'dec', mask: str = 'mask',
                redshift: str = 'redshift', mag: str = 'mag', angle_unit: str = 'radian') -> None:
        r"""
        Link object and random data to the region. Both tables should have same column names. Object 
        data should be linked to the region to get count in cells or correlations.

        Parameters
        ----------
        odf: pandas.DataFrame
            A table or dataframe containing object data in the region.
        rdf: pandas.DataFrame
            A table or dataframe containing random data in the region.
        ra: str, optional
            Name of the column specifying the right-ascension values. Default is 'ra'.
        dec: str, optional
            Name of the column specifying the declination values. Default is 'dec'.
        mask: str, optional
            Name of the column specifying the bright star mask values to use. Default is 'mask'. 
            This column is only needed in the object dataset and optional for random dataset.
        redshift: str, optional
            Name of the column specifying the object redshift values. Default is 'redshift'. This 
            column is only needed in the object dataset and optional for random dataset.
        mag: str, optional
            Name of the column specifying the object magnitude values. Default is 'magnitude'. This 
            column is only needed in the object dataset and optional for random dataset.
        angle_unit: str, optional
            Unit of angles in the data. It can be either 'radian' (default) or 'degree'.

        """

        if not isinstance(odf, pd.DataFrame):
            raise TypeError("odf must be a 'pandas.DataFrame' object")
        if not isinstance(rdf, pd.DataFrame):
            raise TypeError("rdf must be a 'pandas.DataFrame' object")

        self.odf, self.rdf = odf, rdf 

        # checking excistence of columns: ra, dec and mask in both, redshift and mag in odf 
        if ra not in odf.columns or ra not in rdf.columns:
            raise ValueError(f"right-ascension column '{ra}' should be presesnt in both rdf and odf")
        self.attrs['ra'] = ra

        if dec not in odf.columns or ra not in rdf.columns:
            raise ValueError(f"declination column '{dec}' should be presesnt in both rdf and odf")
        self.attrs['dec'] = dec
            
        if mask not in odf.columns or ra not in rdf.columns:
            raise ValueError(f"mask column '{mask}' should be presesnt in both rdf and odf")
        self.attrs['mask'] = mask

        if redshift not in odf.columns:
            raise ValueError(f"redshift column '{redshift}' should be presesnt in odf")
        self.attrs['redshift'] = redshift

        if mag not in odf.columns:
            raise ValueError(f"magnitude column '{mag}' should be presesnt in odf")
        self.attrs['mag'] = mag

        if angle_unit not in ['radian', 'degree']:
            raise ValueError(f"invalid angle unit '{angle_unit}'")
        self.attrs['angle_unit'] = angle_unit

        return

    # count in cells
    
    def tileCells(self, radius: float, packing: str = 'cubic', filter: bool = False, geom: str = 'circle',
                  selection_frac: float = 0.9, bt_leafsize: int = 2, check_overlap: bool = False) -> None:
        r"""
        Tile the 2D region with circular cells of given radius. The cells can be then accessed by the `cells` 
        property of the object.

        Parameters
        ----------
        radius: float
            Radius of the circles in radians.
        packing: str, optional
            Tells which way to pack the circles. `cubic` is the only availabe option now, which is for cubical 
            packing.
        filter: bool, optional
            Tells whether to filter the cells based on the availablity of data. This is used for regions with 
            holes. In that case, cells overlapping the valid region are selected only.
        geom: str, optional
            Cell geometry. Default is 'circle' for circular cells with haversine metric.
        selection_frac: float, optional
            Fraction of non-masked points that a cell must have to be selected. This should be a number between 0 
            and 1 (default is 0.9).
        bt_leafsize: int, optional
            Leaf size used for the ball tree in internal nearest neighbour search. Default is 2.
        check_overlap: bool, optional
            If set true, check if the cell overlap with any other cells in the region. This is time consuming and 
            is turned off by default.  

        """

        rdf, attrs = self.rdf, self.attrs
        if filter and ( rdf is None or None in attrs.values() ):
            raise ValueError("no random data is linked to the region")

        if geom == 'circle':
            self._cells = tile_circles2d(self, radius, packing, filter = filter, df = rdf, ra = attrs['ra'], dec = attrs['dec'], 
                                         mask = attrs['mask'], selection_frac = selection_frac, angle_unit = attrs['angle_unit'], 
                                         bt_leafsize = bt_leafsize, check_overlap = check_overlap)
            self._cell_geom = 'circle'
        else:
            raise NotImplementedError(f"cell geometry '{geom}' is not implemented")
        return

    def getCounts(self, mask: bool = False, zrange: tuple = None, magnitude_coutoff: float = None, 
                   bt_leafsize: int = 2, return_index: bool = False) -> Any:
        r"""
        Find the count in cells of objects in the region.

        Parameters
        ----------
        mask: bool, optional
            Mask value for objects. Default is False, means only objects without mask are used. For not 
            using any mask value, set it to None.
        zrange: tuple, optional
            Redshift range of objects to use for counting. By default, no range is applied.
        magnitude_cutoff: float, optional
            Lower cutoff magnitude for objects to use for counting. By default, no limit is applied.
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

        odf, attrs = self.odf, self.attrs
        if odf is None or None in attrs.values():
            raise ValueError("no object data is linked to the region")

        # filtereing the data
        args = {}
        if mask is not None:
            args['mask']     = attrs['mask']
            args['mask_val'] = mask
        if zrange is not None:
            args['z']      = attrs['redshift']
            args['zrange'] = zrange
        if magnitude_coutoff is not None:
            args['mag'] = attrs['mag']
            args['magnitude_range'] = (magnitude_coutoff, np.inf)
        odf = filterCatalog(odf, **args)

        return get_counts2d(self._cells, self._cell_geom, df = odf, ra = attrs['ra'], dec = attrs['dec'], 
                            angle_unit = attrs['angle_unit'], bt_leafsize = bt_leafsize, return_index = return_index)


