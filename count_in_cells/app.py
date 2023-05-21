#!/usr/bin/python3
#
# measuring count in cells and its analysis
# @author 
#
    

import os, sys
import logging # for log messages
from argparse import ArgumentParser # for argument parsing
from options import load_options # for loading the options file
from patches import create_patches # for creating jackknife patches


try:
    
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK, SIZE = COMM.rank, COMM.size
    HAS_MPI    = 1

except ModuleNotFoundError:

    COMM, RANK, SIZE = None, 0, 1
    HAS_MPI          = 0


# log an error message and exit
def _exit():
    logging.error( "something wrong happened. process terminated!" )
    sys.exit(-1)

# syncing processes: wait for all process to finish
def _sync():
    if HAS_MPI:
        COMM.Barrier() # COMM.barrier()
    return


# argument parser object
parser = ArgumentParser(prog = 'meas_cic', 
                        description = 'Do count-in-cells analysis on data.')
parser.add_argument('param_file', help = 'path to the parameter file', type = str)
parser.add_argument('-r', '--restart', help = 'restart code', type = int, default = 0)


# temporary location for log files. 
if not os.path.exists(".logs") and RANK == 0:
    os.mkdir(".logs")
_sync()

log_file = f"rank-{ RANK }-output.log" 
log_path = os.path.join( '.logs', log_file )

# logger configuration
logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [
                        logging.FileHandler(log_path, mode = 'w'),
                        logging.StreamHandler()
                    ])

if not HAS_MPI:
    logging.warn( "no module named 'mpi4py', computations will be serial. " )


# parsing input arguments
args    = parser.parse_args()

# PART 1: load options from the file
options = load_options( args.param_file )

# if any error in the option loading, log and exit
if options.error is not None:
    logging.error( options.error )
    _exit()


options = options.value # actual options!

output_dir = options.output_path
if not os.path.exists( output_dir ) and RANK == 0:
    logging.info(f"output path { output_dir } does not exist: creating...")
    os.mkdir( output_dir ) # create the output directory


_sync()


# PART 2: create patch images and save to disc


# PART 3: calculate count-in-cells data


logging.info( "all calculations are completed successfully!" )

# PART 4: move log files into output directory
if RANK == 0:

    logging.info( "moving log files to output directory..." )

    log_path = os.path.join( output_dir, 'log' )
    if not os.path.exists( log_path ):
        os.mkdir( log_path )

    for file in os.listdir( '.logs' ):
        os.rename( src = os.path.join( '.logs', file ), dst = os.path.join( log_path, file ) )
    os.rmdir( '.logs' )