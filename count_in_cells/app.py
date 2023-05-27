#!/usr/bin/python3
#
# measuring count in cells and its analysis
# @author m. s. sūryan śivadās 
#
    

import os, sys
import logging # for log messages
from utils import WARN, ERROR, SUCCESS # status variables 
from utils import replace_fields # for string replacement 
from argparse import ArgumentParser # for argument parsing
from options import load_options # for loading the options file
from patches import create_patches # for creating jackknife patches
from count_in_cells import estimate_counts # for count-in-cells


PY_VERSION = sys.version_info
if not ( PY_VERSION.major >= 3 and PY_VERSION.minor >= 10 ):
    print( "requires python version >= 3.10, code may not run properly...", flush = True ) 


try:
    
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK, SIZE = COMM.rank, COMM.size
    HAS_MPI    = 1

except ModuleNotFoundError:

    COMM, RANK, SIZE = None, 0, 1
    HAS_MPI          = 0


# argument parser object
parser = ArgumentParser(prog = 'meas_cic', description = 'Do count-in-cells analysis on data.')
parser.add_argument('param_file', help = 'path to the parameter file', type = str)
# parser.add_argument('-r', '--restart', help = 'restart code', type = int, default = 0)


# syncing processes: wait for all process to finish
def __sync():

    if HAS_MPI:
        COMM.Barrier() # COMM.barrier()
    return


# setting up calculations: loading options and configuring logging
def __initialise():

    __logque  = [] 

    # parse input arguments
    args = parser.parse_args()

    # loading options from the file
    options, __msgs = load_options( args.param_file )

    # create output directory if not exist, otherwise use existing
    output_dir = options.output_dir
    if not os.path.exists( output_dir ) and RANK == 0:
        __logque.append( ( f"created output directory '{ output_dir }'", SUCCESS ) )
        os.mkdir( output_dir ) # create the output directory

    # create log file directory if not exist, otherwise use existing
    log_path = os.path.join( output_dir, 'logs' )
    if not os.path.exists( log_path ) and RANK == 0:
        __logque.append( ( f"created log directory '{ log_path }'", SUCCESS ) )
        os.mkdir( log_path ) # create the log directory
    log_path = os.path.join( log_path, f"rank-{ RANK }-output.log" )

    __sync() # sync processes so that the directories are availabe to all process 

    # logger configuration
    logging.basicConfig(level = logging.INFO,
                        format = "%(asctime)s [%(levelname)s] %(message)s",
                        handlers = [
                            logging.FileHandler(log_path, mode = 'w'),
                            logging.StreamHandler()
                        ])
    
    if not HAS_MPI:
        logging.warn( "no module named 'mpi4py', computations will be serial. " )
    

    # log messages from loading options
    __failed = 0
    if RANK == 0:

        if len( __msgs ):
            for __msg in __msgs:
                if __msg.status == ERROR:
                    logging.error( __msg.msg )
                    __failed = 1
                else:
                    logging.warning( __msg.msg )
        
        for __msg, __status in __logque:
            if __status == ERROR:
                logging.error( __msg )
            elif __status == WARN:
                logging.warning( __msg )
            else:
                logging.info( __msg )

        # try:
        optfile = os.path.join(output_dir, 'used_options.txt')
        options.save_as( optfile )
        logging.info( "used options are written to '%s'", optfile )
        # except Exception as e:
        #     logging.warning( f"cannot save used options file: {e}" )

    __sync() # syncing again
    return options, __failed


# divide region into jackknife patches and random object counting
def __create_and_save_patch_data(options):

    cell_subdivisions = options.cic_cell_num_subdiv
    if not isinstance( cell_subdivisions, int ):
        logging.error( f"cell subdivisions must be an integer" )
        return ERROR
    
    if cell_subdivisions < 0:
        logging.warning( f"cell subdivisions must be positive, will take absolute value" )
        cell_subdivisions = abs( cell_subdivisions )

    min_cellsize = options.cic_cellsize / 2**cell_subdivisions

    # masks to use
    use_masks = [ options.catalog_mask % {'band': band} for band in options.jackknife_use_mask ]

    __failed = create_patches(reg_rect        = options.jackknife_region_rect,
                              ra_size         = options.jackknife_patch_width_ra,
                              dec_size        = options.jackknife_patch_width_dec,
                              pixsize         = min_cellsize,
                              rdf_path        = options.catalog_random,
                              output_dir      = options.output_dir,
                              use_masks       = use_masks,
                              reg_to_remove   = options.jackknife_remove_regions,
                              rdf_compression = options.catalog_compression,
                              chunk_size      = options.catalog_chunk_size,
                              ra_shift        = options.catalog_ra_shift,
                              dec_shift       = options.catalog_dec_shift,
                              rdf_filters     = options.catalog_random_filter_conditions,
                              mpi_comm        = COMM,
                             )

    return __failed


# calculate count-in-cells data
def __calculate_and_save_cic_measurements(options):

    # column keys 
    mask             = options.catalog_mask
    magnitude        = options.catalog_magnitude
    magnitude_offset = options.catalog_magnitude_offset

    # field - column name mapping
    __mapper = {'redshift': options.catalog_redshift,
                'redshift_error': options.catalog_redshift_error,
            }
    for band in options.catalog_all_bands:
        __mapper[ band ] = magnitude % {'band': band}

    # masks to use
    use_masks = [ mask % {'band': band} for band in options.cic_use_mask ]

    # magnitude - offest pairs
    magnitude_offsets = { magnitude % {'band': band} : magnitude_offset % {'band': band} for band in options.catalog_magnitude_to_correct } 

    # magnitude filters
    magnitude_filters = [ replace_fields( __cond, __mapper ) for __cond in options.cic_magnitude_filter_conditions ]

    # redshift filters
    redshift_filters  = [ replace_fields( __cond, __mapper ) for __cond in options.cic_redshift_filter_conditions ]


    __failed = estimate_counts(output_dir        = options.output_dir,
                               odf_path          = options.catalog_object,
                               use_masks         = use_masks,
                               subdiv            = options.cic_cell_num_subdiv,
                               masked_frac       = options.cic_masked_frac,
                               odf_compression   = options.catalog_compression,
                               chunk_size        = options.catalog_chunk_size,
                               magnitude_filters = magnitude_filters,
                               redshift_filters  = redshift_filters,
                               odf_filters       = options.catalog_object_filter_conditions,
                               magnitude_offsets = magnitude_offsets,
                               save_counts       = options.cic_save_counts,
                               mpi_comm          = COMM
                               )
    return __failed


# initialize, patch generation, measurement; in that order!
def main():

    # initialisin calculations...
    options, __failed = __initialise()
    if __failed:
        logging.error("initialization failed, see the log files for more information :(")
        sys.exit(1)

    # calculating patch images...
    __failed = __create_and_save_patch_data(options)
    if __failed:
        logging.error("`create_patches` exited with non-zero status: patch generation failed! :(")
        sys.exit(1)

    # measuring count-in-cells...
    __failed = __calculate_and_save_cic_measurements(options)
    if __failed:
        logging.error("`estimate_counts` exited with non-zero status: cic measurement failled! :(")
        sys.exit(1)

    logging.info("all calculations completed successfully :)")
    return 

if __name__ == '__main__':
    main()
    