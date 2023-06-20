#!/usr/bin/python3

import os, time
import logging # for log messages
import numpy as np, pandas as pd
from scipy.stats import binned_statistic_dd, binned_statistic
from utils import CountData, Rectangle, CICError, check_datafile
from mpi4py import MPI



##################################################################################################
# functions for object counting
##################################################################################################

def _get_counts(df_path: str, region: Rectangle, patchsize_x: float, patchsize_y: float, x_cells: int, 
                y_cells: int, n_patches: int, pixsize: float, subdiv: int, patch_coord: list, 
                bad_patches: list[bool], use_masks: list[str], filters: list[str] = [], 
                expressions: list[str] = [], output_dir: str = '.', filename: str = 'counts.dat', 
                compression: str = 'infer', chunk_size: int = 1_000, x_coord: str = 'ra', y_coord: str = 'dec', 
                frac: bool = False, log: bool = True, delimiter: str = ',', comment: str = '#', header: int = 0,
                colnames: list[str] = None):
    r"""
    Divide the given patches into cells and count the total and masked number of objects in each cells. This is a 
    private function and is called by other public functions (just to make sure no error happen!...)
    """


    comm       = MPI.COMM_WORLD    
    RANK, SIZE = comm.rank, comm.size 

    #
    # count the number of objects in cells, not having mask flag 
    #
    rx_min, rx_max, ry_min, ry_max = region.xmin, region.xmax, region.ymin, region.ymax
    y_patches  = int((region.ymax - ry_min) / patchsize_y ) # number of patches along y direction

    # bin edges
    x_bins     = np.linspace( 0., patchsize_x, x_cells + 1 )
    y_bins     = np.linspace( 0., patchsize_y, y_cells + 1 )
    patch_bins = np.arange(n_patches + 1) - 0.5   # patch bins are centered at its index 
    img_shape  = (x_cells, y_cells, n_patches)

    if log:
        logging.info("counting with cellsize = %f, cell array shape = (%d, %d), %d patches.", 
                     pixsize, x_cells, y_cells, n_patches)
        
        
    # first, check the data files can be loaded properly
    if log:
        logging.info("checking catalog file: '%s'...", df_path)

    
    # checking data...
    required_cols = [x_coord, y_coord, *use_masks]
    check_datafile(df_path, 
                   required_cols, 
                   filters + expressions, 
                   compression, 
                   chunk_size, 
                   header, 
                   delimiter, 
                   comment, 
                   colnames,            ) # raise exception on failure

    filters   = [*filters, 
                 f"{x_coord} >= {rx_min}", f"{x_coord} <= {rx_max}", 
                 f"{y_coord} >= {ry_min}", f"{y_coord} <= {ry_max}" ] # adding boundary limits
    filters   = '&'.join( map( lambda __filt: "(%s)" % __filt, filters ) )          # join data filter conditions (logical and)
    mask_expr = '&'.join( map( lambda __msk: "(%s == False)" % __msk, use_masks ) ) # no mask flags are set
    if log:
        logging.info("using filtering condition: '%s'", filters)
        if mask_expr:
            logging.info("using mask expression: '%s'", mask_expr)
        if expressions:
            logging.info("using expressions: '%s'", ', '.join(expressions))


    if log:
        logging.info( "started counting objects in cell" )
        __t_init = time.time()

    total, exposed = np.zeros(img_shape), np.zeros(img_shape)
    with pd.read_csv(df_path, 
                     header      = header,
                     delimiter   = delimiter,
                     comment     = comment,
                     names       = colnames,  
                     compression = compression,
                     chunksize   = chunk_size   ) as df_iter:
        
        chunk = 0
        n_used, n_total = 0, 0
        for df in df_iter:
            
            # distribute chunks among different process
            if chunk % SIZE != RANK:
                chunk = chunk + 1
                continue
            chunk = chunk + 1
            
            n_total += df.shape[0]

            # evaluate expressions: mainly magnitude corrections
            for expression in expressions:
                df = df.eval( expression ) 

            df = df.query( filters ).reset_index(drop = True) # apply data filters
            if df.shape[0] == 0: # if no data is available, ignore
                continue 
            n_used += df.shape[0]

            if mask_expr:
                mask_weight = df.eval( mask_expr ).to_numpy().astype('float') # masked objects get weight 0 and others 1
            else: 
                mask_weight = np.ones( df.shape[0] )

            # 
            # convert from 2D physical coordinates (x, y) to 3D patch coordinates (x', y', patch) with 
            # x' in [0, patchsize_x], y' in [0, patchsize_y] and patch in [0, n_patches) 
            #  
            df[x_coord] = df[x_coord] - rx_min # coordinates relative to region
            df[y_coord] = df[y_coord] - ry_min

            xp, yp      = np.floor(df[x_coord] / patchsize_x), np.floor(df[y_coord] / patchsize_y)
            df['patch'] = y_patches * xp + yp # patch index

            df[x_coord] = df[x_coord] - xp * patchsize_x # coordinates relative to patch
            df[y_coord] = df[y_coord] - yp * patchsize_y

            # chunk counts
            total_c, exposed_c = binned_statistic_dd(df[[x_coord, y_coord, 'patch']].to_numpy(),
                                                     values    = [ np.ones(df.shape[0]), mask_weight ], # weights
                                                     statistic = 'sum',
                                                     bins      = [ x_bins, y_bins, patch_bins ], ).statistic
            
            total, exposed = total + total_c, exposed + exposed_c

    if log:
        logging.info( "used %d objects out of %d for counting", n_used, n_total )
        logging.info( "finished counting in %g seconds :)", time.time() - __t_init )
    

    #
    # since using multiple processes, combine counts from all processes
    #
    comm.Barrier() # FIXME: check if this needed...

    if log and SIZE > 1:
        logging.info( "starting communication...")

    if RANK != 0:

        if log:
            logging.info( "sending data to rank-0" )

        # send data to process-0
        comm.Send( total,   dest = 0, tag = 10 ) # total count
        comm.Send( exposed, dest = 0, tag = 11 ) # unmasked count

    else:

        # recieve data from other processes
        tmp = np.zeros( img_shape ) # temporary storage
        for src in range(1, SIZE):

            if log:
                logging.info( "recieving data from rank-%d", src )
            
            # total count
            comm.Recv( tmp, source = src, tag = 10,  )
            total = total + tmp

            # exposed count
            comm.Recv( tmp, source = src, tag = 11,  )
            exposed = exposed + tmp

    comm.Barrier()


    #
    # after counting, save the counts to use later (done at process 0)
    #
    if RANK == 0:

        # remove bad patches from the results
        good_patches = np.invert(bad_patches)
        total        = total[:,:,good_patches]
        exposed      = exposed[:,:,good_patches]

        data_to_save = [total, exposed]
        if frac: # save exposed fraction only
            non_empty = (total > 0.)
            exposed[non_empty] = exposed[non_empty] / total[non_empty]
            data_to_save       = [exposed] # now it is exposed fraction

        if not os.path.exists(output_dir):
            raise CICError(f"path does not exist: {output_dir}")
        
        file = os.path.join(output_dir, filename)
        CountData(*data_to_save,
                  patch_llcoords = patch_coord,
                  patch_flags    = bad_patches,
                  max_subdiv     = subdiv,
                  pixsize        = pixsize,
                  patchsize_x    = patchsize_x,
                  patchsize_y    = patchsize_y,
                  region         = region,        ).save( file )
        
        if log:
            logging.info( "count data saved to '%s'", file )


    # 
    # wait untill all process to reach this point and finish... :)
    #  
    comm.Barrier()

    if log:
        logging.info( "finished counting!" )
    return 


def create_patches(region: Rectangle | list[float], patchsize_x: float, patchsize_y: float, pixsize: float, 
                   df_path: str, output_dir: str, use_masks: list[str], x_coord: str = 'ra', y_coord: str = 'dec', 
                   subdiv: int = 0, compression: str = 'infer', chunk_size: int = 1_000, filters: list[str] = [], 
                   expressions: list[str] = [], bad_regions: list[Rectangle | list[float]] = [], log: bool = True, 
                   delimiter: str = ',', comment: str = '#', header: int = 0, colnames: list[str] = None):
    r"""
    Divide a rectangular region in to rectangular patches of same dimensions. Then, each of these 
    patches are divided into square cells of smallest possible size.
    """
    if log:
        logging.info( "started patch image computation..." )

    if patchsize_x <= 0. or patchsize_y <= 0.:
        raise CICError( f"patch sizes must be positive (x_size = {patchsize_x}, y_size = {patchsize_y})" )
    elif pixsize <= 0. or pixsize > min(patchsize_x, patchsize_y):
        raise CICError( f"pixsize (= {pixsize}) must be less than the patch sizes, min({patchsize_x}, {patchsize_y})" )
    
    if not isinstance(subdiv, int):
        raise CICError("number of subdivisions must be an integer")
    elif subdiv < 0:
        raise CICError("number of subdivisions must be zero or a positive integer")
    
    region = Rectangle.make(region)

    if bad_regions is None:
        bad_regions = []
    for __i, __reg in enumerate(bad_regions):
        bad_regions[__i] = Rectangle.make(__reg)
    
    # for the calculations to work, image shape must be (even, even). if pixsize does not make this, 
    # correct the pixsize and patchsizes
    x_cells = int( patchsize_x / pixsize )
    if x_cells % 2: 
        x_cells = x_cells - 1 
    patchsize_x = x_cells * pixsize # corrected patchsize along x dirra_sizeection

    y_cells = int( patchsize_y / pixsize )
    if y_cells % 2: 
        y_cells = y_cells - 1 
    patchsize_y = y_cells * pixsize # corrected patchsize along dec

    # after coorecting the size, apply subdivisions to get the smallest pixel
    __res   = 2**subdiv 
    pixsize = pixsize / __res 
    x_cells, y_cells = int( y_cells * __res ), int( y_cells * __res )

    rx_min, rx_max, ry_min, ry_max = region.xmin, region.xmax, region.ymin, region.ymax


    #
    # divide the area into patches of similar sizes
    #
    if log:
        logging.info( "creating patches of size_x = %f and size_y = %f. expected image shape = (%d, %d)", 
                     patchsize_x, 
                     patchsize_y, 
                     x_cells, 
                     y_cells     )
        
    patch_coord = [] # lower left coordinates of the patches
    bad_patches = [] # this flag is set True if a patch is bad (i.e., intersect with a bad region)
    x_min       = rx_min
    while 1:
        x_max = x_min + patchsize_x
        if x_max > rx_max:
            break

        y_min = ry_min
        while 1:
            y_max = y_min + patchsize_y
            if y_max > ry_max:
                break

            bad_patch = False
            for bad_region in bad_regions:
                bad_patch = bad_region.intersect( Rectangle(x_min, x_max, y_min, y_max) )
                if bad_patch:
                    break
            
            patch_coord.append([ x_min, y_min ])
            bad_patches.append( bad_patch )

            y_min = y_max
        
        x_min = x_max

    n_patches, n_badpatches = len(patch_coord), sum(bad_patches)
    if n_patches == 0: # no patches in this region
        raise CICError("no patches can be created in the region with given sizes")
    elif n_patches == n_badpatches: # all the patches are bad
        raise CICError("no good patches left in the region :(")
    
    if log:
        logging.info("created %d patches (%d bad patches)", n_patches, n_badpatches)
    
    #
    # counting objects in cells (total and exposed / non-masked)
    #
    _get_counts(df_path      = df_path, 
                region       = region, 
                patchsize_x  = patchsize_x, 
                patchsize_y  = patchsize_y, 
                x_cells      = x_cells, 
                y_cells      = y_cells, 
                n_patches    = n_patches, 
                pixsize      = pixsize, 
                subdiv       = subdiv, 
                patch_coord  = patch_coord, 
                bad_patches  = bad_patches, 
                use_masks    = use_masks, 
                filters      = filters, 
                expressions  = expressions, 
                output_dir   = output_dir,
                filename     = 'patch_data.dat', 
                compression  = compression, 
                chunk_size   = chunk_size, 
                x_coord      = x_coord, 
                y_coord      = y_coord, 
                frac         = True,
                log          = log,
                delimiter    = delimiter,
                comment      = comment, 
                header       = header,
                colnames     = colnames          )
    
    if log:
        logging.info( "finished patch image computation :)" )
    return    


def estimate_counts(df_path: str, use_masks: list[str], output_dir: str, patch_file: str = None, 
                    compression: str = 'infer', chunk_size: int = 1_000, filters: list[str] = [], 
                    expressions: list[str] = [], x_coord: str = 'ra', y_coord: str = 'dec', log: bool = True,
                    delimiter: str = ',', comment: str = '#', header: int = 0, colnames: list[str] = None):
    r"""
    Count the number of objects satisfying a set of conditions, in some pre-computed cells.
    """

    if log:
        logging.info( "started object counting..." )

    #
    # load jackknife patch data from file, incl. cellsize, patchsizes, subdivisions etc
    #
    if not os.path.exists(output_dir):
        raise CICError(f"path does not exist: {output_dir}")
    if patch_file is None:
        patch_file = os.path.join(output_dir, 'patch_data.dat')
    patch_data  = CountData.load(patch_file)
    patch_specs = patch_data.header
    patch_coord = patch_data.patch_llcoords
    bad_patches = patch_data.patch_flags
    del patch_data # to save space (patch image data is not needed here)
    if log:
        logging.info( "loaded patch data from file '%s'", patch_file )


    x_cells, y_cells         = patch_specs.data_shape[:2]
    patchsize_x, patchsize_y = patch_specs.patchsize_x, patch_specs.patchsize_y
    pixsize                  = patch_specs.pixsize    # smallest possible cellsize
    subdiv                   = patch_specs.max_subdiv # number of subdivisions applied to the cells
    region                   = patch_specs.region
    n_patches                = len(patch_coord)

    #
    # counting objects in cells (total and exposed / non-masked)
    #
    _get_counts(df_path      = df_path, 
                region       = region, 
                patchsize_x  = patchsize_x, 
                patchsize_y  = patchsize_y, 
                x_cells      = x_cells, 
                y_cells      = y_cells, 
                n_patches    = n_patches, 
                pixsize      = pixsize, 
                subdiv       = subdiv, 
                patch_coord  = patch_coord, 
                bad_patches  = bad_patches, 
                use_masks    = use_masks, 
                filters      = filters, 
                expressions  = expressions, 
                output_dir   = output_dir,
                filename     = 'count_data.dat', 
                compression  = compression, 
                chunk_size   = chunk_size, 
                x_coord      = x_coord, 
                y_coord      = y_coord, 
                frac         = False,
                log          = log,  
                delimiter    = delimiter,
                comment      = comment, 
                header       = header,
                colnames     = colnames          )
    return    


##################################################################################################
# functions to estimate count histograms
##################################################################################################

def estimate_distribution(output_dir: str, count_files: str | list[str] = None, 
                          patch_files: str | list[str] = None, max_count: int = 1_000, 
                          masked_frac: float = 0.05, log: bool = True):
    r"""
    Estimate count-in-cells distribution using pre-computed count image.
    """ 
    # import sys
    # sys.stderr.write("\033[91mwarning:\033[m using parellelised test version\n")

    comm       = MPI.COMM_WORLD    
    RANK, SIZE = comm.rank, comm.size 


    if not isinstance(max_count, int):
        raise CICError("max_count must be an integer")
    elif max_count < 10:
        raise CICError("max_count must be at least 10, got %d" % max_count)

    if masked_frac < 0. or masked_frac > 1.:
        raise CICError("masked_frac must be a number between 0 and 1")


    if log:
        logging.info( "started estimating distribution..." )

    #
    # load counts and patch data from files 
    #
    if RANK == 0:
        # data are loaded only in rank 0 and distributed to others
        if log:
            logging.info( "loading data files..." )
            __t_init = time.time()

        if not os.path.exists(output_dir):
            raise CICError(f"path does not exist: {output_dir}")
        
        if not count_files:
            count_files = os.path.join(output_dir, 'count_data.dat')
        if isinstance(count_files, str):
            count_files = [count_files]
        
        if not patch_files:
            patch_files = os.path.join(output_dir, 'patch_data.dat')
        if isinstance(patch_files, str):
            patch_files = [patch_files]

        if len(patch_files) != len(count_files):
            raise CICError(f"number of patch files {len(patch_files)} must be same as number of count files {len(count_files)}")
    
        # patch and count data
        count_data = CountData.merge_load(count_files)

        patch_data = CountData.merge_load(patch_files)
        patch_data.assert_similar(count_data, ignore = ['ndata'])

        exp_frac, exp_count = patch_data.data[0].T, count_data.data[1].T
        subdiv, pixsize = patch_data.header.max_subdiv, patch_data.header.pixsize
        x_cells, y_cells, n_patches = patch_data.header.data_shape

        if log:
            logging.info( "successfully loaded all data in %g seconds :)", time.time() - __t_init )
    else:
        # not setting the values in other processes
        exp_frac, exp_count         = None, None
        n_patches, x_cells, y_cells = None, None, None
        pixsize, subdiv             = None, None 

    # if using multiple processes, brodcast the values from 0
    n_patches = comm.bcast( n_patches, root = 0 )
    x_cells   = comm.bcast( x_cells,   root = 0 )
    y_cells   = comm.bcast( y_cells,   root = 0 )
    pixsize   = comm.bcast( pixsize,   root = 0 )
    subdiv    = comm.bcast( subdiv,    root = 0 )


    # split the data into chunks for each process
    _split_size, _rem = divmod(n_patches, SIZE)
    split_sizes       = np.repeat(_split_size, SIZE) + np.where(np.arange(SIZE) < _rem, 1, 0) # number of patches at each chunk
    split_sizes_inp   = split_sizes * y_cells * x_cells                  # size of the input chunk
    displacements_inp = np.insert(np.cumsum(split_sizes_inp), 0, 0)[:-1] # shift of input chunk beginning from array beginning


    #
    # estimating distribution of counts with a histogram
    #
    if log:
        logging.info("started estimation of count histogram...")
        __t_init = time.time()

    # buffers to store the chunk data
    exp_frac_c  = np.zeros((split_sizes[RANK], y_cells, x_cells))
    exp_count_c = np.zeros((split_sizes[RANK], y_cells, x_cells))

    # scatter data from process 0 to others
    comm.Scatterv([exp_frac,  split_sizes_inp, displacements_inp, MPI.DOUBLE], exp_frac_c,  root = 0)
    comm.Scatterv([exp_count, split_sizes_inp, displacements_inp, MPI.DOUBLE], exp_count_c, root = 0)

    comm.Barrier() # wait for all process to sync...

    
    # count the distribution...
    count_bins       = np.arange(max_count + 2) - 0.5 # bins centered at count values
    min_exposed_frac = 1.0 - masked_frac              # lowest exposed fraction allowed
    distr            = np.zeros((max_count + 1, subdiv + 1, split_sizes[RANK]))
    for level in range(subdiv + 1):

        # cells with exposed fraction greater than the minimum is marked good
        good_cells = (exp_frac_c > min_exposed_frac)  
        for p in range(split_sizes[RANK]):

            # using counts from good cells of this patch...
            good_counts = exp_count_c[p, good_cells[p,:,:]].flatten()
            if len(good_counts) < 1:
                continue

            distr[:,level,p] = binned_statistic(good_counts, 
                                                values    = None,
                                                statistic = 'count',
                                                bins      = count_bins ).statistic
            
        if level == subdiv:
            break
        
        # after each level, double the cellsize. i.e., merge two cells from each direction 
        # for fractions, take the average value as the new cell fraction
        exp_frac_c = 0.5*(exp_frac_c[:,0::2,:] + exp_frac_c[:,1::2,:]) # along y direction
        exp_frac_c = 0.5*(exp_frac_c[:,:,0::2] + exp_frac_c[:,:,1::2]) # along x direction
        
        #  for counts, take sum as the new cell count
        exp_count_c = exp_count_c[:,0::2,:] + exp_count_c[:,1::2,:] # along y direction
        exp_count_c = exp_count_c[:,:,0::2] + exp_count_c[:,:,1::2] # along x direction

        
    comm.Barrier() # wait untill all process to sync...

    if log:
        logging.info(f"finished estimation of count histogram in %g seconds! :)", time.time() - __t_init)


    if RANK != 0:
        if log:
            logging.info( "sending data to rank-0" )

        comm.Send(distr, dest = 0, tag = 13) # send data to process-0   

        comm.Barrier()
        if log:
            logging.info( "finished estimating distribution :)" )
        return


    # receieving data from others
    error = [distr]
    for src in range(1, SIZE):
        tmp = np.zeros((max_count + 1, subdiv + 1, split_sizes[src]))
        if log:
            logging.info( "recieving data from rank-%d", src )

        comm.Recv( tmp, source = src, tag = 13,  )
        error.append(tmp)
    error = np.concatenate(error, axis = -1) # now it is just count distribution


    # 
    # jackknife average and error
    #
    if log:
        logging.info("estimatimating jackknife error...")
        __t_init = time.time()    

    # average distribution:
    distr = np.sum(error, axis = -1) / n_patches 

    # std. error (now var error become the 'error'...)
    error = np.sqrt(np.sum((error - distr[...,None])**2, axis = -1) / (n_patches * (n_patches - 1))) 

    if log:
        logging.info(f"finished error estimation in %g seconds! :)", time.time() - __t_init)


    # 
    # save results to disk
    # 
    count     = np.arange(max_count + 1)[:,None]
    columns   = ['count'] + ['distr_%d' % level for level in range(subdiv + 1)] + ['error_%d' % level for level in range(subdiv + 1)]
    save_path = os.path.join(output_dir, 'count_histogram.csv')
    pd.DataFrame(np.concatenate([count, distr, error], axis = 1),
                 columns = columns).to_csv(save_path,
                                           index        = False,
                                           float_format = "%16.8e")
    if log:
        logging.info("estimated distribution saved to '%s'", save_path)


    save_path = os.path.join(output_dir, 'cellsizes.csv')
    np.savetxt(save_path, 
               pixsize * 2**np.arange(subdiv + 1), 
               fmt = "%16.8e")
    if log:
        logging.info("cellsizes are saved to '%s'", save_path)


    # finishing...
    comm.Barrier()
    if log:
            logging.info( "finished estimating distribution :)" )
    return
    
