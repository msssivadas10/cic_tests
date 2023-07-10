#!/usr/bin/pyhton3
#
# application for measuring count-in-cells
# @author m. s. sūryan śivadās
#
__version__ = '1.0a'
prog_name   = 'meas_cic'
prog_info   = 'Do count-in-cells analysis on data.'

import sys

def __raise_error_and_exit(__msg: str):
    sys.stderr.write("\033[1m\033[91merror:\033[m %s\n" % __msg)
    sys.exit(1)

if sys.version_info < (3, 10):
    __raise_error_and_exit("app requires python version >= 3.10")

try:
    from mpi4py import MPI
except ModuleNotFoundError:
    __raise_error_and_exit("cannot import 'mpi4py', module not found.")

import os
import logging 
from argparse import ArgumentParser
from options import load_options, Options
from estimation import create_patches, estimate_counts, estimate_distribution, estimate_distribution2
from utils import replace_fields, get_typename, Rectangle, CICError


comm       = MPI.COMM_WORLD
rank, size = comm.rank, comm.size 


# argument parser object
parser = ArgumentParser(prog = prog_name, description = prog_info)
parser.add_argument('--opt',     help = 'path to the input options file',    type = str)
parser.add_argument('--alt-opt', help = 'path to an alternate options file', type = str)
parser.add_argument('--flag',    help = 'flags to control the execution',    type = int, default = 1)


# default values
default_mask           = "%(band)s_mask"
default_allbands       = ['g','r','i','z','y']
default_magnitude      = "%(band)s_magnitude"
default_mag_offset     = "%(band)s_magnitude_offset"
default_redshift       = "redshift"
default_redshift_error = "redshift_err"


# to check if the value is of the given type(s)
def __check_type(__value: object, __types: list[type], __key: str = 'value', __null_val: object = None):

    if __value is None and __null_val is not None:
        return __null_val
    
    if isinstance(__value, tuple(__types)):
        return __value
    
    allowed_types = ' or '.join( ', '.join( map(lambda o: f"'{get_typename(o)}'", __types) ).rsplit(', ', maxsplit = 1) )
    currect_type  = get_typename( type(__value) )
    raise ValueError(f"{__key} must be of type {allowed_types}, got '{currect_type}'")

def __check_array(__value: object, __item_types: list[type], __key: str = 'value', __null_val: object = None):

    if __value is None and __null_val is not None:
        return __null_val

    if isinstance(__value, (list, tuple)):
        if all( map( lambda o: isinstance(o, tuple(__item_types)), __value ) ):
            return __value
        
    allowed_types = ' or '.join( ', '.join( map(lambda o: f"'{get_typename(o)}'", __item_types) ).rsplit(', ', maxsplit = 1) )
    raise ValueError(f"{__key} must a sequence of {allowed_types}")

def __check_rect(r: object, __array: bool, __key: str = 'value',  __null_val: object = None):

    if r is None and __null_val is not None:
        return __null_val
    
    if not __array:
        try:
            return Rectangle.make(r)
        except Exception:
            raise ValueError(f"{__key} must be a valid rectangle-like object")
    
    if not isinstance(r, (list, tuple)):
        raise ValueError(f"{__key} must be a sequence of rectangle-like objects")
    
    rr = []
    for i, ri in enumerate(r):
        try:
            rr.append( Rectangle.make(ri) )
        except Exception:
            raise ValueError(f"{i}-th entry of {__key} is not a rectangle-like object")
    return rr

# initialising the calculations
def __initialize(opt_file: str, alt_opt_file: str = None) -> Options:

    # load options files
    if opt_file is None:
        __raise_error_and_exit("missing options file")
    if not os.path.exists(opt_file):
        __raise_error_and_exit("file '%s' does not exist." % opt_file)
    if alt_opt_file is not None:
        if not os.path.exists(alt_opt_file):
            __raise_error_and_exit("file '%s' does not exist." % alt_opt_file)
    
    try:
        options = load_options(file = opt_file, alt_file = alt_opt_file)
    except Exception as e:
        __raise_error_and_exit("loading options failed: %s" % e)

    output_dir = './output'
    if options.output_dir is not None:
        output_dir = options.output_dir
        if not isinstance(output_dir, str):
            __raise_error_and_exit("output_dir must be an 'str'")
    # output_dir = os.path.abspath(output_dir)
    
    # create output directory if not exist
    if not os.path.exists(output_dir):
        if rank == 0:
            try:
                os.mkdir(output_dir)
            except Exception as e:
                __raise_error_and_exit("creating output directory raised exception '%s'" % e)
    options.output_dir = output_dir

    # create directory for logs
    log_path = os.path.join(output_dir, 'logs')
    if not os.path.exists(log_path):
        if rank == 0:
            try:
                os.mkdir(log_path)
            except Exception as e:
                __raise_error_and_exit("creating log directory raised exception '%s'" % e)
    log_path = os.path.join(log_path, '%d.log' % rank)
    
    # FIXME: log files are not created properly
    comm.Barrier()
    if not os.path.exists(log_path):
        open(log_path, 'w').close()
    

    # configure logger
    logging.basicConfig(level = logging.INFO,
                        format = "%(asctime)s [%(levelname)s] %(message)s",
                        handlers = [
                            logging.FileHandler(log_path, mode = 'w'),
                            logging.StreamHandler()
                        ])
    
    # write used options to a file
    if rank == 0:
        options._save_as(file = os.path.join(output_dir, 'used_options.txt'))
    
    comm.Barrier()
    return options

# making patch calculations
def __calculate_patch_data(options: Options) -> None:
              
    try:

        # 
        # check options
        # 

        # total region to used for calculations
        region      = __check_rect(options.counting_region_rect, False, 'region_rect' )

        # list of bad regions to cut out from total
        bad_regions = __check_rect(options.counting_remove_regions, True,  'remove_regions', [])

        # patchsizes
        patchsize_x = __check_type(options.counting_patchsize_x, [float, int], 'patchsize_x' )
        patchsize_y = __check_type(options.counting_patchsize_y, [float, int], 'patchsize_x' )

        # maximum cellsize and number of subdivisions to apply
        pixsize     = __check_type(options.counting_cellsize,   [float, int], 'pixsize'      )
        max_subdiv  = __check_type(options.counting_max_subdiv, [int],        'max_subdiv', 0)

        # random catalog 
        catalog     = __check_type(options.random_catalog_path,  [str], 'random catalog path')
        mask        = __check_type(options.random_catalog_mask,    [str], 'mask key', default_mask)
        x_coord     = __check_type(options.random_catalog_x_coord, [str], 'x_coord',  'ra'        )
        y_coord     = __check_type(options.random_catalog_y_coord, [str], 'y_coord',  'dec'       )
        compression = options.random_catalog_compression
        chunk_size  = options.random_catalog_chunk_size
        delimiter   = options.random_catalog_delimiter
        comment     = options.random_catalog_comment
        header      = options.random_catalog_header
        colnames    = options.random_catalog_colnames

        if colnames is None and header is None:
            header = 0


        # all bands used in the survey
        all_bands   = __check_array(options.all_bands, [str], 'all_bands', default_allbands)

        # set of masks to use
        mask_bands  = __check_array(options.counting_random_mask, [str], 'use_masks', [])
        use_masks   = []
        for band in mask_bands:
            if band not in all_bands:
                raise ValueError("%s is not a valid band name" % band)
            use_masks.append(mask % {'band': band})

        # data filters for random
        filters = []
        for expr in __check_array(options.random_catalog_filters, [str], 'random_catalog_filters', []):
            if '%(band)s' in expr:
                for band in all_bands:
                    filters.append(expr % {'band': band})
            else:
                filters.append(expr)

        # additional expressions to evaluate
        expressions = [] # not availble

        output_dir = options.output_dir

        # 
        # creating the patches (raise CICError on failure)
        #
        create_patches(region       = region, 
                       patchsize_x  = patchsize_x, 
                       patchsize_y  = patchsize_y, 
                       pixsize      = pixsize, 
                       df_path      = catalog, 
                       output_dir   = output_dir, 
                       use_masks    = use_masks,
                       x_coord      = x_coord, 
                       y_coord      = y_coord, 
                       subdiv       = max_subdiv, 
                       compression  = compression, 
                       chunk_size   = chunk_size, 
                       filters      = filters, 
                       expressions  = expressions, 
                       bad_regions  = bad_regions, 
                       log          = True,
                       delimiter    = delimiter, 
                       comment      = comment, 
                       header       = header, 
                       colnames     = colnames    )
        return 

    except ValueError as e1: # failure in loading options 
        logging.error("%s", e1)

    except CICError as e2: # failure in calculations
        logging.error("create_patches: %s", e2)
    
    logging.error("patch calculation failed :(")
    __raise_error_and_exit("patch calculation failed :(")
    return

# counting objects
def __count_objects(options: Options) -> None:

    try:

        #
        # check options 
        #

        # object catalog
        catalog        = __check_type(options.object_catalog_path,             [str], 'object catalog path'                       )
        mask           = __check_type(options.object_catalog_mask,             [str], 'mask key',           default_mask          )
        magnitude      = __check_type(options.object_catalog_magnitude,        [str], 'magnitude key',      default_magnitude     )
        mag_offset     = __check_type(options.object_catalog_magnitude_offset, [str], 'magnitude key',      default_mag_offset    )
        redshift       = __check_type(options.object_catalog_redshift,         [str], 'redshift key',       default_redshift      )
        redshift_error = __check_type(options.object_catalog_redshift_error,   [str], 'redshift error key', default_redshift_error)
        x_coord        = __check_type(options.object_catalog_x_coord,          [str], 'x_coord',            'ra'                  )
        y_coord        = __check_type(options.object_catalog_y_coord,          [str], 'y_coord',            'dec'                 )
        compression    = options.object_catalog_compression
        chunk_size     = options.object_catalog_chunk_size
        delimiter      = options.object_catalog_delimiter
        comment        = options.object_catalog_comment
        header         = options.object_catalog_header
        colnames       = options.object_catalog_colnames

        if colnames is None and header is None:
            header = 0

        # all bands used in the survey
        all_bands   = __check_array(options.all_bands, [str], 'all_bands', default_allbands)

        # set of masks to use
        mask_bands  = __check_array(options.counting_object_mask, [str], 'use_masks', [])
        use_masks   = []
        for band in mask_bands:
            if band not in all_bands:
                raise ValueError("%s is not a valid band name" % band)
            use_masks.append(mask % {'band': band})

        # variable mapper for expressions
        mapper = {'redshift': redshift, 'redshift_err': redshift_error}
        for band in all_bands:
            mapper[band]            = magnitude  % {'band': band}
            mapper['%s_off' % band] = mag_offset % {'band': band} 

        # data filters for objects, incl. special filters for magnitude and redshift selections
        filters = []
        for expr in __check_array(options.object_catalog_filters, [str], 'object_catalog_filters', []):
            if '%(band)s' in expr:
                for band in all_bands:
                    filters.append(expr % {'band': band})
            else:
                # if the expression contains mappings (fields starting with $), apply them 
                filters.append( replace_fields(expr, mapper) )

        # expressions to evaluate on 
        expressions = []
        for expr in __check_array(options.object_catalog_expressions, [str], 'object_catalog_expressions', []):
            if '%(band)s' in expr:
                for band in all_bands:
                    expressions.append(expr % {'band': band})
            else:
                # if the expression contains mappings (fields starting with $), apply them 
                expressions.append( replace_fields(expr, mapper) )

        output_dir = options.output_dir
        patch_file = None # use file 'patch_data.dat' from the output dir


        #
        # counting objects on cells
        #
        estimate_counts(df_path     = catalog,
                        use_masks   = use_masks,
                        output_dir  = output_dir,
                        patch_file  = patch_file,
                        compression = compression,
                        chunk_size  = chunk_size,
                        filters     = filters,
                        expressions = expressions,
                        x_coord     = x_coord,
                        y_coord     = y_coord,
                        log         = True, 
                        delimiter    = delimiter, 
                        comment      = comment, 
                        header       = header, 
                        colnames     = colnames   )
        return 
    
    except ValueError as e1:
        logging.error("%s", e1)

    except CICError as e2:
        logging.error("estimate_counts: %s", e2)
    
    logging.error("object counting failed :(")
    __raise_error_and_exit("object counting failed :(")
    return 

# calculating count distribution
def __calculate_distribution(options: Options) -> None:
    
    try:
        
        #
        # check options
        #
        output_dir = options.output_dir

        # count and patch data files to use
        count_files = __check_array(options.distribution_count_files, [str], 'count_files', []) 
        patch_files = __check_array(options.distribution_patch_files, [str], 'patch_files', [])

        # maximum count values
        max_count   = __check_type(options.distribution_max_count, [int], 'max_count', 1000)

        # maximum allowed masked fraction
        masked_frac = __check_type(options.distribution_masked_frac, [float, int], 'masked_frac', 0.05)  

        #
        # estimate distribution
        #
        estimate_distribution2(output_dir  = output_dir,
                                count_files = count_files,
                                patch_files = patch_files,
                                max_count   = max_count,
                                masked_frac = masked_frac,
                                log         = True,        )

        return
    
    except ValueError as e1:
        logging.error('%s', e1)
    
    except CICError as e2:
        logging.error("estimate_distribution: %s", e2)

    logging.error("distribution estimation failed :(")
    __raise_error_and_exit("distribution estimation failed :(")
    return


def main():

    # parse arguments
    args = parser.parse_args()
    flag = args.flag

    # initialising...
    options = __initialize(opt_file = args.opt, alt_opt_file = args.alt_opt)
    if flag & 8:
        logging.info("exiting after successfull initialization :)")
        return
    
    # preparing patch data by counting randoms...
    if flag & 48:
        logging.warning("skipping patch calculations")
    else:
        __calculate_patch_data(options)
        if flag & 4:
            logging.info("exiting after successfull patch generation :)")
            return

    # counting objects... 
    if flag & 32:
        logging.warning("skipping object counting")
    else:
        __count_objects(options)
        if flag & 2:
            logging.info("exiting after successfull object counting :)")
            return 
    
    # measuring distribution...
    __calculate_distribution(options)    
    logging.info("exiting after successfull distribution estimation :)")
    return

if __name__ == '__main__':
    main()
