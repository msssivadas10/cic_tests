#!/usr/bin/python3

import os
import numpy as np, pandas as pd
import logging
import re
from string import Template
from typing import Any 


# error codes: warning (2), error/failure (1) or success (0)
WARN, ERROR, SUCCESS = 2, 1, 0


def check_datafile(path: str, compression: str, chunk_size: int, required_cols: list = [], 
                   data_filters: list = []) -> int:
    r"""
    Check if the data file exist and loaded properly. 
    """

    # check file path
    if not os.path.exists( path ):
        logging.error( "catalog file does not exist: '%s'", path )
        return 1
    
    # check file compression
    if compression not in ['infer', None, 'bz2', 'gzip', 'tar', 'xz', 'zip', 'zstd']:
        logging.error( "unsupported compression: '%s'", str( compression ) )
        return 1
    
    # check chunksize
    if not isinstance( chunk_size, int ):
        logging.error( "chunk_size must be an integer" )
        return 1
    elif chunk_size < 1:
        logging.error( "chunk_size must be an integer >= 1" )
        return 1
    
    # check if the data can be loaded and apply filters correctly
    try:
        with pd.read_csv( path, header = 0, compression = compression, chunksize = chunk_size ) as df_iter:
            
            __df = df_iter.get_chunk(10)

            logging.info( "checking for required columns..." )
            __df_cols = __df.columns
            for col in required_cols:
                if col not in __df_cols:
                    logging.error( "missing required column '%s'", col )
                    return 1
            logging.info( "all column requirements satisfied :)" )

            logging.info( "checking filtering conditions..." )
            if data_filters:
                conditions = "&".join( data_filters )
                _ = __df.query( conditions )
            logging.info( "all filtering conditions are valid :)" )

    except Exception as __e:
        logging.error( "loading/filtering data raised an exception: %s", __e )
        return 1
    
    return 0


def intersect(r1: list, r2: list) -> bool:
    r"""
    Check if two rectangles intersect.
    """
    return not (r1[0] >= r2[1] or r1[1] <= r2[0] or r1[3] <= r2[2] or r1[2] >= r2[3])

def is_bad_rect(r: list) -> int:
    r"""
    Check if `r` is a valid rectangle structure. i.e., a sequence of four numbers.
    """
    try:
        size = len(r)
        if size != 4:
            logging.error( "rect must be a sequence of 4 numbers, got %d", size )
            return ERROR 
        
        if any( map( lambda ri: not isinstance( ri, (float, int) ), r ) ):
            logging.error( "rect must be a sequence of 4 numbers" )
            return ERROR
        
    except Exception as e:
        print(e)
        logging.error( "rect must be a sequence (list, tuple or numpy.array)" )
        return ERROR
    
    return SUCCESS


def replace_fields(__str: str, __mapper: dict) -> str:
    r"""
    Replace fields marked with a `$` at begining (e.g., `$a`) with a value in the mapper. 
    """

    fields = re.findall( r'\$(\w+)', __str )
    for field in fields:
        if field not in __mapper.keys():
            logging.error( f"field '{ field }' has no replacement in the mapper" )
            return None
        
    return Template( __str ).substitute( __mapper )


def jackknife_error(obs: Any) -> tuple:
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
