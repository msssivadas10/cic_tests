

import numpy as np, pandas as pd
import json
import argparse
import os, sys
from dataclasses import dataclass
from mpi4py import MPI


# argument parser
parser = argparse.ArgumentParser(prog = 'meas_cic', description = 'Do count-in-cells analysis on data.')
parser.add_argument('param_file', type = str, help = 'path to the parameter file')
parser.add_argument('-r', '--restart', help = 'restart code', type = int)


# writes the log message 
def log(msg: str, exit_code: int = 0):

    sys.stdout.write(msg + '\n')
    sys.stdout.flush()

    if exit:
        sys.exit("process exited with code {}".format(exit_code))
    return


# load input parameters from file
def load_paramfile(param_file: str, restart: int = 0, check_values: bool = False):
    
    with open(param_file, 'r') as fp:
        params = json.load( fp )

    if check_values:
        
        # checking parameter values
        data_struct = {'cellsize': float,
                       'cell_subdivisions': int,
                       'chunk_size': int,
                       }
        for key, value in data_struct.items():
            if key not in params.keys():
                log(f"missing value for `{key}`", exit_code = 2)
            # if not isinstance(params[key], value):
            #     log(f"incorrect value for `{key}`: {params[key]}")


        # check file paths
        _paths = params.get('paths')
        if not _paths:
            log("missing value for `paths`", exit_code = 2)

        for _path in ['objects', 'randoms', 'patches']:
            if _path not in _paths.keys():
                log(f"missing value for `paths.{_path}`", exit_code = 2)
            
        #     if not os.path.exists( _paths[_path] ):
        #         log(f"path does not exist: `{ _paths[_path] }`", exit_code = 3)

        # if 'output_dir' not in _paths.keys():
        #     params['paths']['output_dir'] = './output'

        # output_dir = params['paths']['output_dir']
        # if not os.path.exists(output_dir):
        #     os.mkdir(output_dir)

        
    return params


# create patch mask images
def create_patch_images(patch_file_path: str, random_catalog_path: str, cellsize: float, nsubdiv: int = 1, chunk_size: int = 1_000_000, comm = None, rank: int = 0, size: int = 1):

    # load patches file

    # load random objects

    # save patch as image

    return 0


# step-2: estimate count-in-cells distribution, 1st, 2nd, 3rd moments
def estimate_cic(*args, **kwds):
    ...


def main():

    comm = MPI.COMM_WORLD
    rank = comm.rank 
    size = comm.size

    args = parser.parse_args()

    # load input parameter file
    params = load_paramfile( args.param_file, args.restart, check_values = (rank == 0) )
    

    # # create patch images
    patch_file_path     = params['paths']['patches']
    random_catalog_path = params['paths']['randoms']
    cellsize            = params['cellsize']
    cell_subdivisions   = params['cell_subdivisions']
    chunk_size          = params['chunk_size']

    __failure = create_patch_images(patch_file_path, random_catalog_path, cellsize, cell_subdivisions, chunk_size, comm, rank, size)
    if __failure:
        log("failed to create patch images", exit_code = 1)

    return


if __name__ == '__main__':
    
    __failure = main()
    if __failure:
        log('process failed!...')