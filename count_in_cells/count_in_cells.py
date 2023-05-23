#!/usr/bin/python3

import numpy as np, pandas as pd
import logging # for log messages
from utils import check_datafile # to check data 
from utils import ERROR, SUCCESS
from patches import PatchData
from dataclasses import dataclass
from scipy.stats import binned_statistic_dd, binned_statistic
from typing import Any 


def estimate_counts(patch_file: str, odf_path: str, use_masks: list, subdiv: int = 0, max_count: int = 100,
                    masked_frac: float = 0.05, odf_compression: str = 'gzip', chunk_size: int = 1_000, 
                    magnitude_filters: list = [], redshift_filters: list = [], odf_filters: list = [], 
                    magnitude_offsets: dict = {}, mpi_comm = None) -> int:
    r"""
    Estimate count-in-cells distribution and first moments from data.
    """

    RANK, SIZE = 0, 1
    USE_MPI    = False
    if mpi_comm is not None:
        RANK, SIZE = mpi_comm.rank, mpi_comm.size
        USE_MPI    = True

    # load jackknife patch images from the file
    patches = PatchData.load_from( patch_file )
    if patches is None:
        logging.error( "failed to load patch file '%s'", patch_file )
        return ERROR
    
    logging.info( "successfully loaded patch data" )

    # checking cell subdivisions value
    if not isinstance( subdiv, int ):
        logging.error( f"cell subdivisions must be an integer" )
        return 1
    if subdiv < 0:
        logging.warning( f"cell subdivisions must be positive, will take absolute value" )
        subdiv = abs( subdiv )

    # checking max_count
    if not isinstance( max_count, int ):
        logging.error( f"max_count must be an integer" )
        return 1
    if max_count < 10:
        logging.warning( f"max_count must be at least 10 (got {max_count})" )
        max_count = 10

    # check data file
    logging.info( "checking object catalog file: '%s'...", odf_path )

    required_cols = []
    required_cols.extend([ 'ra', 'dec' ])
    required_cols.extend( use_masks )
    for __mag, __off in magnitude_offsets.items():
        required_cols.append( __mag )
        required_cols.append( __off )

    odf_filters.extend( magnitude_filters )
    odf_filters.extend( redshift_filters )

    if check_datafile( odf_path, odf_compression, chunk_size, required_cols, odf_filters ):
        return ERROR

    # data filters: region limits and other specified filters
    reg_ra1, reg_ra2, reg_dec1, reg_dec2 = patches.header.region
    odf_filters.extend([ "ra >= @reg_ra1",  "ra <= @reg_ra2", "dec >= @reg_dec1", "dec <= @reg_dec2" ])

    # adding mask filters: select only unmasked objects
    odf_filters.extend([ "%s == False" % __m for __m in use_masks ])

    odf_filters = "&".join([ "(%s)" % __filter for __filter in odf_filters ])

    # 
    # counting objects in cells
    #

    total_rcount, masked_rcount = patches.total, patches.masked # for masked cell area

    ra_size, dec_size      = patches.header.ra_patchsize, patches.header.dec_patchsize # patch sizes
    ra_shift, dec_shift    = patches.header.ra_shift, patches.header.dec_shift # coordinate shifts
    n_ra, n_dec, n_patches = patches.header.ra_cells, patches.header.dec_cells, patches.header.n_patches

    # cell size
    min_pixsize = patches.header.pixsize # pixsize = {1, 2, 4, ..., 2**subdiv} x min_pixsize

    ra_bins = np.arange(0., ra_size + min_pixsize, min_pixsize)
    ra_bins = ra_bins[ ra_bins <= ra_size ] 

    dec_bins = np.arange(0., dec_size + min_pixsize, min_pixsize)
    dec_bins = dec_bins[ dec_bins <= dec_size ] 

    patch_bins = np.arange(0, n_patches + 1) - 0.5

    count_hist = np.zeros( (max_count, subdiv+1, n_patches), dtype = 'int' ) # distribution / histogram
    count_bins = np.arange( max_count + 1 )

    logging.info( "started counting unmasked objects in cell" )

    with pd.read_csv(odf_path, header = 0, compression = odf_compression, chunksize = chunk_size) as odf_iter:

        #
        #  iterate through all chunks and accumulate the counts
        #
        chunk_id = 0
        for odf in odf_iter:

            if chunk_id % SIZE != RANK:
                chunk_id += 1
                continue
            chunk_id += 1

            # apply all filters
            odf = odf.query( odf_filters ).reset_index(drop = True)
            if odf.shape[0] == 0:
                continue
            
            # shift the origin to the lower-left corner of the region 
            odf['ra']  = odf['ra']  - reg_ra1  - ra_shift
            odf['dec'] = odf['dec'] - reg_dec1 - dec_shift

            # get patch id
            i = np.floor( odf['ra']  / ra_size ).astype( 'int' ) 
            j = np.floor( odf['dec'] / dec_size ).astype( 'int' )
            odf['patch'] = n_dec * i + j

            # convert positions to patch coordinates
            odf['ra']  = odf['ra']  - i * ra_size
            odf['dec'] = odf['dec'] - j * dec_size

            # counting the number in each cell
            counts_i = binned_statistic_dd(odf[['ra', 'dec', 'patch']].to_numpy(), 
                                           values = None,
                                           statistic = 'count',
                                           bins = [ ra_bins, dec_bins, patch_bins ]
                                        ).statistic
            
            masked_frac = 0.5 # hardcoding, will become a parameter later
            
            #
            # get the distribution of counts
            #

            count_hist += get_distribution(counts_i,
                                           count_bins, 
                                           total_rcount,
                                           masked_rcount,
                                           n_patches,
                                           subdiv,
                                           masked_frac, )


    logging.info( "finished counting object histogram" )

    # wait untill all process are completed, if using multiple process 
    if USE_MPI:
        mpi_comm.Barrier() # comm.barrier()


    # combine the results from all process
    logging.info( "starting communication...")
    if RANK != 0 and USE_MPI:

        logging.info( "sent data to rank-0" )

        # send data to process-0
        mpi_comm.Send( count_hist, dest = 0, tag = 12 )

    else:

        # recieve data from other process, if using multiple process
        if USE_MPI:

            tmp = np.zeros( (max_count, subdiv+1, n_patches), dtype = 'int' ) # temporary storage
            for src in range(1, SIZE):

                logging.info( "recieving data from rank-%d", src )
                
                mpi_comm.Recv( tmp, source = src, tag = 12 )
                count_hist += tmp

        
        count_hist  = count_hist[..., patches.flags] # remove bad patches

        # moment calculations

        #
        # jackknife averaging over all patches ()
        #

        # save average results and variance

    # wait untill all process are completed, if using multiple process 
    if USE_MPI:
        mpi_comm.Barrier() # comm.barrier()

    logging.info( "finished counting histogram!" )
    
    return SUCCESS


def get_distribution(counts: Any, count_bins: Any, r_total: Any, r_mask: Any, 
                     n_patches: int, n_subdiv: int, mask_frac: float):
    r"""
    Count histogram of masked counts 
    """

    hist = np.empty( ( len(count_bins)-1, n_subdiv+1, n_patches ), dtype = 'int' ) # distribution / histogram

    total_l, mask_l, counts_l = r_total, r_mask, counts
    for level in range(n_subdiv, -1, -1): # subdivision levels

        # pixel goodness flag
        isgood_l = ( (total_l > 0) & (mask_l < mask_frac * total_l) )

        for p in range( n_patches ): # patches

            xlp = counts_l[..., p].flatten() # pixel values
            mlp = isgood_l[..., p].flatten() # is the pixel good

            xlp = xlp[ mlp ]
            hist[:, level, p] = binned_statistic(xlp,
                                                 values    = None, 
                                                 statistic = 'count',
                                                 bins      = count_bins, ).statistic
            
        #  
        # doubling pixsize: combine 4 cells each into 1 
        #

        if level == 0:
            break
        
        total_l = total_l[0::2,:,:] + total_l[1::2,:,:] # along ra
        total_l = total_l[:,0::2,:] + total_l[:,1::2,:] # along dec

        mask_l = mask_l[0::2,:,:] + mask_l[1::2,:,:] # along ra
        mask_l = mask_l[:,0::2,:] + mask_l[:,1::2,:] # along dec

        counts_l = counts_l[0::2,:,:] + counts_l[1::2,:,:] # along ra
        counts_l = counts_l[:,0::2,:] + counts_l[:,1::2,:] # along dec
        
    return hist
