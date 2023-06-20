#!/usr/bin/python3

import os
import numpy as np, pandas as pd
import re
import struct
from dataclasses import dataclass
from string import Template
from typing import Any, TypeVar 


# error codes: warning (2), error/failure (1) or success (0)
WARN, ERROR, SUCCESS = 2, 1, 0

EPSILON  = 1e-06 # tolerence to match two floats


class CICError(Exception):
    r"""
    A class of exception raised on failing count-in-cells calculation.
    """
    ...
    

def jackknife(obs: Any, bias_correction: bool = False) -> tuple:
    r"""
    Estimate mean and error using jackknife resampling.
    """

    n_obs = obs.shape[-1] # number of observations

    mean_jk  = np.mean( obs, axis = -1 )  # jackknife mean: same as sample mean
    error_jk = np.zeros( obs.shape[:-1] ) # jackknife error
    for p in range( n_obs ):
        error_jk += ( np.mean( np.delete( obs, p, axis = -1 ), axis = -1 ) - mean_jk )**2
    error_jk = np.sqrt( error_jk * (n_obs - 1) / n_obs )

    # TODO: apply bias correction

    return mean_jk, error_jk


def replace_fields(__str: str, __mapper: dict) -> str:
    r"""
    Replace fields marked with a `$` at begining (e.g., `$a`) with a value in the mapper. 
    """

    fields = re.findall( r'\$(\w+)', __str )
    for field in fields:
        if field not in __mapper.keys():
            raise CICError( f"field '{ field }' has no replacement in the mapper" )
        
    return Template( __str ).substitute( __mapper )


def get_typename(__type: type) -> str:
    r"""
    Get the typename of the type `__type`.
    """
    if not isinstance(__type, type):
        raise CICError("__type must be a python 'type' object")
    __t, = re.match(r'\<class \'([\w\.]+)\'\>', str(__type)).groups()
    return __t

def get_typename_of(__obj: Any) -> str:
    r"""
    Get the typename of the object `__obj`.
    """
    return get_typename( type(__obj) )


def check_datafile(path: str, required_cols: list[str] = [], expressions: list[str] = [], 
                   compression: str = 'infer', chunk_size: int = 1_000_000, header: int = 0, 
                   delimiter: str = ',', comment: str = '#', colnames: list[str] = None) -> None:
    r"""
    Check if the data file exist and can be loaded properly. If not, 'CICError' exception is 
    raised. 
    """

    # check file path
    if not os.path.exists( path ):
        raise CICError( f"catalog file does not exist: '{path}'" )
    
    # check file compression
    if compression not in ['infer', None, 'bz2', 'gzip', 'tar', 'xz', 'zip', 'zstd']:
        raise CICError( f"unsupported compression: '{compression}'" )
    
    # check chunksize
    if chunk_size is not None:
        if not isinstance(chunk_size, int):
            raise CICError( f"chunk_size must be an integer, got value {chunk_size}" )
        elif chunk_size < 1:
            raise CICError( f"chunk_size must be an integer >= 1, got value {chunk_size}" )
    
    # check if the data can be loaded and apply filters correctly
    with pd.read_csv(path, 
                     header      = header,
                     delimiter   = delimiter,
                     comment     = comment, 
                     names       = colnames, 
                     compression = compression, 
                     chunksize   = chunk_size ) as df_iter:
        
        _df = df_iter.get_chunk(10)

        # check for columns
        _df_cols = _df.columns
        for col in required_cols:

            if not isinstance(col, str):
                raise CICError( "column names must be of type 'str', got '%s'" % get_typename_of(col) )
            
            if not col:
                raise CICError( "column name cannot be an empty string" )
            
            if col not in _df_cols:
                raise CICError( "missing required column '%s'" % col )
            
        # check expressions
        for expr in expressions:

            if not isinstance(expr, str):
                raise CICError( "expression must be of type 'str', got '%s'" % get_typename_of(expr) )
            
            if not expr:
                raise CICError( "expression cannot be an empty string" )
            
            try:
                _ = _df.eval( expr )
            except Exception as e:
                raise CICError(f"cannot evaluate expression: '{expr}': {e}")
    
    return


Rectangle_like = TypeVar('Rectangle_like')

@dataclass(slots = True, frozen = True)
class Rectangle:
    r"""
    A rectangle object used to specify rectangular regions for cic calculations.
    """

    xmin: float
    xmax: float
    ymin: float
    ymax: float

    @classmethod
    def make(cls, r: Rectangle_like):
        if isinstance(r, Rectangle):
            return r
        try:
            return Rectangle(*map(float, r))
        except Exception:
            raise CICError( f"cannot convert a '{r}' to a 'Rectangle' object" )

    def __post__init__(self) -> None:
        if self.xmax <= self.xmin or self.ymax <= self.ymin:
            raise CICError( f"bad rectangle (non-positive area)")  
    
    def aslist(self):
        return [self.xmin, self.xmax, self.ymin, self.ymax]
        
    def intersect(self, other) -> bool:
        r"""
        Check if the rectangle intersects with other.
        """
        return not (self.xmin >= other.xmax 
                    or self.xmax <= other.xmin 
                    or self.ymax <= other.ymin 
                    or self.ymin >= other.ymax )
    
    def combine(self, other):
        r"""
        Union with another rectangle.
        """
        return Rectangle([min(self.xmin, other.xmin), 
                          max(self.xmax, other.xmax), 
                          min(self.ymin, other.ymin), 
                          max(self.ymax, other.ymax)])
    
    def __eq__(self, other):
        if not isinstance(other, Rectangle):
            return NotImplemented
        return (abs(self.xmin - other.xmin) < EPSILON
                and abs(self.xmax - other.xmax) < EPSILON
                and abs(self.ymin - other.ymin) < EPSILON
                and abs(self.ymax - other.ymax) < EPSILON )


@dataclass(slots = True, frozen = True)
class _CountHeader:
    data_shape: tuple     # shape of the each data arrays: (# of ra cells, # of dec cells, # of patches)
    max_subdiv: int       # maximum subdivisions used for image generation 
    pixsize   : float     # pixel size for the current data (smallest possible value)
    region    : Rectangle # bounding box of the region as the list [xmin, xmax, ymin, ymax]
    ndata     : int       # number of arrays stored (all should have same shape)
    
    # patchsizes along x and y direction
    patchsize_x: float
    patchsize_y: float


class CountData:
    r"""
    Store count data along with patches, flags etc.
    """

    __slots__ = 'header', 'patch_llcoords', 'patch_flags', 'data', 

    def __init__(self, *data: Any, patch_llcoords: list, patch_flags: list[bool], max_subdiv: int,
                 pixsize: float, patchsize_x: float, patchsize_y: float, region: Rectangle) -> None:
        
        self.data = []
        shape, ndata = None, 0
        for data_i in data:
            data_i = np.asfarray(data_i)
            
            self.data.append( data_i )
            ndata += 1

            if shape is None:
                shape = data_i.shape
                continue
            
            if np.ndim(data_i) != 3:
                raise CICError( f"all data should be of dimension 3, got {np.ndim(data_i)}" )
            if not np.allclose(data_i.shape, shape):
                raise CICError( f"all data should have same size, got {shape} and {data_i.shape}" )
            
        self.patch_llcoords = np.asfarray(patch_llcoords)
        self.patch_flags    = np.array(patch_flags).astype('bool')

         
        if len(self.patch_flags) != len(self.patch_llcoords):
            raise CICError( f"number of flags and patch coordinates must be same, got {len(self.patch_flags)} and {len(self.patch_llcoords)}" )
        
        n_patches = len(self.patch_flags) - sum(self.patch_flags) # number of good patches
        if n_patches != shape[2]:
            raise CICError( f"number of patches should be {n_patches}, got {shape[2]}" )
            
        if ndata < 1:
            raise CICError( f"no data is available" ) 

        self.header = _CountHeader(data_shape  = shape, 
                                   max_subdiv  = max_subdiv,
                                   pixsize     = pixsize, 
                                   region      = region,
                                   ndata       = ndata,
                                   patchsize_x = patchsize_x,
                                   patchsize_y = patchsize_y)
        
    def assert_similar(self, other, ignore: list[str] = []) -> None:
        r"""
        Check if the two data has similar conditions.
        """

        if not isinstance(other, CountData):
            raise CICError("other should be a 'CountData' object")

        for attr in ['max_subdiv', 'pixsize', 'ndata', 'patchsize_x', 'patchsize_y']:
            if attr in ignore:
                continue
            value1, value2 = getattr(self.header, attr), getattr(other.header, attr)
            if abs( value1 - value2 ) > EPSILON:
                raise CICError(f"got different values for '{attr}' ({value1} and {value2})")
        
        for attr in ['data_shape', 'region']:
            if attr in ignore:
                continue
            value1, value2 = getattr(self.header, attr), getattr(other.header, attr)
            if value1 != value2:
                raise CICError(f"got different values for '{attr}' ({value1} and {value2})")
            
        return
        
    def save(self, file: str) -> None:
        r"""
        Save data as a binary file.
        """

        # header      : { ndata::int, shape::int[3], subdiv::int, pixsize::float, patchsize::float[2], region::float[4] }
        # data        : float[shape_0*shape_1*shape_2]
        # patch coords: float[2*shape_2]
        # patch_flags : bool[shape_2] 
        #
        # i.e., total format { int[5], float[7], float[shape_0*shape_1*shape_2*ndata], float[2*shape_2], bool[shape_2] }

        ndata, shape  = self.header.ndata, self.header.data_shape
        n_patches     = int(len(self.patch_flags))
        n_bad_patches = int(sum(self.patch_flags)) 

        __fmt = f'6q7d{ ndata * shape[0] * shape[1] * shape[2] }d{ 2*n_patches }d{ n_patches }?'
        buf   = struct.pack(__fmt,
                            ndata, 
                            *shape, 
                            n_bad_patches,
                            self.header.max_subdiv,
                            self.header.pixsize,
                            self.header.patchsize_x,
                            self.header.patchsize_y,
                            *self.header.region.aslist(),
                            *np.concatenate([ xi.flatten() for xi in self.data ]),
                            *self.patch_llcoords.flatten(),
                            *self.patch_flags.flatten(),
                            )
        
        with open(file, 'wb') as f:
            f.write( buf )

        return
    
    def extend(self, *others):
        r"""
        Extend count data using the data from others.
        """
        
        region = self.header.region
        shape1 = self.header.data_shape[:2]
        nimag  = self.header.data_shape[2]
        
        for other in others:

            if not isinstance(other, CountData):
                raise TypeError("can only extend with another 'CountData' object")
                    
            # check header
            for attr in ['max_subdiv', 'pixsize', 'ndata', 'patchsize_x', 'patchsize_y']:
                value1, value2 = getattr(self.header, attr), getattr(other.header, attr)
                if abs( value1 - value2 ) > EPSILON:
                    raise CICError(f"got different values for '{attr}' ({value1} and {value2})")
                
            shape2 = other.header.data_shape[:2]
            if shape1 != shape2:
                raise CICError(f"got different values for 'data_shape'({shape1} and {shape2})")
            
            region = region.combine(other.header.region)
            nimag  = nimag + other.header.data_shape[2]

            for i in range(self.header.ndata):
                self.data[i] = np.concatenate( [self.data[i], other.data[i]], axis = -1 )

            self.patch_flags    = np.concatenate([self.patch_flags,    other.patch_flags],    axis = -1)
            self.patch_llcoords = np.concatenate([self.patch_llcoords, other.patch_llcoords], axis = -1)

        # update header
        self.header = _CountHeader(data_shape  = (shape1[0], shape1[1], nimag), 
                                   max_subdiv  = self.header.max_subdiv,
                                   pixsize     = self.header.pixsize, 
                                   region      = region,
                                   ndata       = self.header.ndata,
                                   patchsize_x = self.header.patchsize_x,
                                   patchsize_y = self.header.patchsize_y     )
        return
    
    @staticmethod
    def load(file: str):
        r"""
        Load counts data from a binary file.
        """

        if not os.path.exists(file):
            raise CICError(f"unable to load data, file does not exist: '{file}'")

        with open(file, 'rb') as f:
            buf = f.read()

        __fmt = '6q'
        start = 0
        stop  = start + struct.calcsize(__fmt)
        ndata, ncells_x, ncells_y, n_good_patches, n_bad_patches, max_subdiv = struct.unpack(__fmt, buf[start:stop])
        n_patches = n_good_patches + n_bad_patches

        __fmt = '7d'
        start = stop
        stop  = start + struct.calcsize(__fmt)
        pixsize, patchsize_x, patchsize_y, xmin, xmax, ymin, ymax = struct.unpack(__fmt, buf[start:stop])

        __fmt = f'{ndata * ncells_x * ncells_y * n_good_patches}d'
        start = stop
        stop  = start + struct.calcsize(__fmt)
        data  = list(
                        np.asfarray(struct.unpack(__fmt, buf[start:stop]) 
                                    ).reshape((ndata, ncells_x, ncells_y, n_good_patches))
                    )

        __fmt = f'{2*n_patches}d'
        start = stop
        stop  = start + struct.calcsize(__fmt)
        patch_llcoords = np.asfarray( struct.unpack(__fmt, buf[start:stop]) ).reshape((n_patches, 2))

        __fmt = f'{n_patches}?'
        start = stop
        stop  = start + struct.calcsize(__fmt)
        patch_flags = np.array( struct.unpack(__fmt, buf[start:stop]) ).astype('bool')

        return CountData(*data, 
                         patch_llcoords = patch_llcoords,
                         patch_flags    = patch_flags,
                         max_subdiv     = max_subdiv,
                         pixsize        = pixsize,
                         patchsize_x    = patchsize_x,
                         patchsize_y    = patchsize_y,
                         region         = Rectangle(xmin, xmax, ymin, ymax)
                         )
    
    @staticmethod
    def merge_load(files: list[str]):
        r"""
        Load count data from multiple files and merge.
        """

        if len(files) < 1:
            raise CICError("at leat one file should be given")
        
        res = CountData.load( files[0] )
        res.extend(*[CountData.load(file) for file in files[1:]])
        return res
    