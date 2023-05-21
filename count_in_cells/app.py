#!/usr/bin/python3
#
# measuring count in cells and its analysis
# @author 
#
    

import os, sys
import logging # for log messages
from mpi4py import MPI
from argparse import ArgumentParser # for argument parsing
from options import load_options # for loading the options file
from patches import create_patches # for creating jackknife patches


COMM = MPI.COMM_WORLD
RANK, SIZE = COMM.rank, COMM.size


# argument parser object
parser = ArgumentParser(prog = 'meas_cic', 
                        description = 'Do count-in-cells analysis on data.')
parser.add_argument('param_file', help = 'path to the parameter file', type = str)
parser.add_argument('-r', '--restart', help = 'restart code', type = int)



# logger configuration
log_file = f"rank-{ RANK }-output.log"
logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s [%(levelname)s] %(message)s",
                    handlers = [
                        logging.FileHandler(log_file, mode = 'w'),
                        logging.StreamHandler()
                    ])