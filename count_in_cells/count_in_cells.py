#!/usr/bin/python3

import os
import logging # for log messages
import numpy as np, pandas as pd
from utils import check_datafile, jackknife_error 
from utils import ERROR, SUCCESS
from patches import PatchData
from scipy.stats import binned_statistic_dd, binned_statistic, describe
from typing import Any 


def estimate_counts(output_dir: str, odf_path: str, use_masks: list, subdiv: int = 0, max_count: int = 100,
                    masked_frac: float = 0.05, odf_compression: str = 'gzip', chunk_size: int = 1_000, 
                    magnitude_filters: list = [], redshift_filters: list = [], odf_filters: list = [], 
                    magnitude_offsets: dict = {}, save_counts: bool = True, do_stats: bool = True, 
                    mpi_comm = None) -> int:
    r"""
    Estimate count-in-cells distribution and first moments from data.
    """

    RANK, SIZE = 0, 1
    USE_MPI    = False
    if mpi_comm is not None:
        RANK, SIZE = mpi_comm.rank, mpi_comm.size
        USE_MPI    = ( SIZE > 1 )

    # 
    # load jackknife patch images from the file
    #

    patch_file = os.path.join( output_dir, "patch_data.npz" )  # patch file for jackknife sampling
    patches    = PatchData.load_from( patch_file )
    if patches is None:
        logging.error( "failed to load patch file '%s'", patch_file )
        return ERROR
    
    logging.info( "successfully loaded patch data" )


    #
    # checking parameters values and data file
    #

    # checking cell subdivisions value
    if not isinstance( subdiv, int ):
        logging.error( f"cell subdivisions must be an integer" )
        return ERROR
    if subdiv < 0:
        logging.warning( f"cell subdivisions must be positive, will take absolute value" )
        subdiv = abs( subdiv )

    # checking max_count
    if not isinstance( max_count, int ):
        logging.error( f"max_count must be an integer" )
        return ERROR
    if max_count < 10:
        logging.warning( f"max_count must be at least 10 (got {max_count})" )

    # check masked fraction value
    if masked_frac < 0. or masked_frac > 1.:
        logging.error( f"masked_frac must be in between 0 and 1" )
        return ERROR

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
    # counting ( un-masked ) objects in cells
    #

    ra_size, dec_size   = patches.header.ra_patchsize, patches.header.dec_patchsize # patch sizes
    ra_shift, dec_shift = patches.header.ra_shift, patches.header.dec_shift # coordinate shifts
    n_dec, n_patches    = int( (reg_dec2 - reg_dec1) / patches.header.dec_patchsize ), patches.header.n_patches

    # cell size
    min_pixsize = patches.header.pixsize # pixsize = {1, 2, 4, ..., 2**subdiv} x min_pixsize

    ra_bins = np.arange(0., ra_size + min_pixsize, min_pixsize)
    ra_bins = ra_bins[ ra_bins <= ra_size ] 

    dec_bins = np.arange(0., dec_size + min_pixsize, min_pixsize)
    dec_bins = dec_bins[ dec_bins <= dec_size ] 

    patch_bins = np.arange(0, n_patches + 1) - 0.5


    logging.info( "started counting unmasked objects in cell" )

    counts = np.zeros(( len(ra_bins) - 1, len(dec_bins) - 1, n_patches ))
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

            # correct magnitudes
            for __mag, __off in magnitude_offsets.items():
                odf[ __mag ] = odf[ __mag ] - odf[ __off ]

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

            # counting the number of unmasked objects in each cell
            counts_i = binned_statistic_dd(odf[['ra', 'dec', 'patch']].to_numpy(), 
                                           values = None,
                                           statistic = 'count',
                                           bins = [ ra_bins, dec_bins, patch_bins ]
                                        ).statistic
            counts += counts_i


    logging.info( "finished counting objects" )

    #
    # data communication: combine the results from all process
    #
    if USE_MPI:
        mpi_comm.Barrier() # comm.barrier()

        logging.info( "starting communication...")
        if RANK != 0:

            logging.info( "sent data to rank-0" )

            mpi_comm.Send( counts, dest = 0, tag = 13 ) # send data to process-0

        else:

            # recieve data from other process, if using multiple process
            tmp = np.zeros((len(ra_bins)-1, len(dec_bins)-1, n_patches))
            for src in range(1, SIZE):

                logging.info( "recieving data from rank-%d", src )
                
                mpi_comm.Recv( tmp, source = src, tag = 13 )
                counts += tmp

        mpi_comm.Barrier() # comm.barrier()



    #
    # estimating count distribution and moments OR save the counts
    #
    if RANK == 0:

        # apply patch flags to select only good patches
        patch_flags = ~patches.flags.astype('bool') # will be `true` if patch is good
        counts      = counts[:,:,patch_flags]
        total       = patches.total[:,:,patch_flags]
        masked      = patches.masked[:,:,patch_flags]

        if save_counts:
            save_path = os.path.join( output_dir, 'counts.npz' )
            np.savez(
                     save_path, 
                     counts = counts,
                     total  = total,
                     masked = masked,
                     sizes  = np.asfarray([min_pixsize, patches.header.ra_patchsize, patches.header.dec_patchsize]),)
            logging.info( "counts saved to '%s'", save_path )

        if do_stats:
            __estimate_distribution_and_save(counts,
                                             total,
                                             masked, 
                                             max_count, 
                                             subdiv, 
                                             masked_frac,
                                             output_dir)


    # wait untill all process are completed, if using multiple process 
    if USE_MPI:
        mpi_comm.Barrier() # comm.barrier()

    logging.info( "finished count-in-cells estimation" )
    
    return SUCCESS


def __estimate_distribution_and_save(counts: Any, total: Any, masked: Any, max_count: int, subdiv: int,
                                     masked_frac: float, output_dir: str):

    logging.info( "started counting histogram..." )

    # estimating the count distribution
    count_bins = np.arange( max_count + 2 ) - 0.5 # count bin edges {-1/2, 1/2, 3/2, ..., max_count + 0.5} 
    count_hist = get_distribution(counts, 
                                  count_bins, 
                                  total,
                                  masked,
                                  subdiv, 
                                  masked_frac)


    # jackknife averaging over all patches
    count_hist, count_hist_err = jackknife_error( count_hist )

    logging.info( "finished counting histogram!" )



    # save average results and variance
    logging.info( "saving outputs..." )

    save_path = os.path.join( output_dir, 'count_histogram.csv' )
    pd.DataFrame(count_hist
                ).reset_index(names = 'count'
                            ).to_csv(save_path, 
                                    index = False)
    logging.info( "count distribution saved to '%s'", save_path )
    
    save_path = os.path.join( output_dir, 'count_histogram_error.csv' )
    pd.DataFrame(count_hist_err
                ).reset_index(names = 'count'
                            ).to_csv(save_path, 
                                    index = False)
    logging.info( "count distribution errors saved to '%s'", save_path )
    return 


def get_distribution(counts: Any, count_bins: Any, r_total: Any, r_mask: Any, n_subdiv: int, 
                     masked_frac: float) -> Any:
    r"""
    Count histogram of masked counts 
    """

    n_patches = counts.shape[2]

    hist = np.zeros( ( len(count_bins)-1, n_subdiv+1, n_patches ), dtype = 'int' ) # distribution / histogram
    # stat = np.zeros( ( 4, n_subdiv+1, n_patches ), dtype = 'float' ) # descriptive statistics

    total_l, mask_l, counts_l = r_total, r_mask, counts
    for level in range(n_subdiv + 1): # subdivision levels

        # pixel goodness flag
        isgood_l = ( (total_l > 0) & (mask_l < masked_frac * total_l) )

        for p in range( n_patches ): # patches

            xlp = counts_l[ isgood_l[..., p] , p].flatten() # good pixel values
            hist[:, level, p] = binned_statistic(xlp,
                                                 values    = None, 
                                                 statistic = 'count',
                                                 bins      = count_bins, ).statistic
            
            # __stat = describe( xlp )
            # stat[:, level, p] = __stat.mean, __stat.variance, __stat.skewness, __stat.kurtosis 
            
        #  
        # doubling pixsize: combine 4 cells each into 1 
        #
        if level == n_subdiv:
            break
        
        total_l = total_l[0::2,:,:] + total_l[1::2,:,:] # along ra
        total_l = total_l[:,0::2,:] + total_l[:,1::2,:] # along dec

        mask_l = mask_l[0::2,:,:] + mask_l[1::2,:,:] # along ra
        mask_l = mask_l[:,0::2,:] + mask_l[:,1::2,:] # along dec

        counts_l = counts_l[0::2,:,:] + counts_l[1::2,:,:] # along ra
        counts_l = counts_l[:,0::2,:] + counts_l[:,1::2,:] # along dec

    return hist


def estimate_total_ditribution(count_files: list, output_dir: str, max_count: int, subdiv: int, 
                               masked_frac: float)-> int:
    r"""
    Combine results from multiple non-overlapping, disconnected regions with same setup. That is, 
    the cellsize, patchsizes, data filtering conditions etc must be same in all to get correct 
    results. Note that not all these conditions are checked and the user should ensure this, if 
    want correct results...
    """

    #
    # checking parameters values and data file
    #

    # checking cell subdivisions value
    if not isinstance( subdiv, int ):
        logging.error( f"cell subdivisions must be an integer" )
        return ERROR
    if subdiv < 0:
        logging.warning( f"cell subdivisions must be positive, will take absolute value" )
        subdiv = abs( subdiv )

    # checking max_count
    if not isinstance( max_count, int ):
        logging.error( f"max_count must be an integer" )
        return ERROR
    if max_count < 10:
        logging.warning( f"max_count must be at least 10 (got {max_count})" )

    # check masked fraction value
    if masked_frac < 0. or masked_frac > 1.:
        logging.error( f"masked_frac must be in between 0 and 1" )
        return ERROR


    #
    # loading count-in-cells data from files
    #

    n_files = len(count_files)
    if n_files < 2:
        logging.error(f"at least 2 files are needed, got {n_files}")
        return ERROR

    counts, total, masked, __first_file = [], [], [], True
    min_pixsize, ra_patchsize, dec_patchsize, ra_cells, dec_cells = 0., 0., 0., 0, 0
    for file in count_files:
        
        if not os.path.exists(file):
            logging.error(f"file does not exist: '{file}'")
            return ERROR
        
        try:
            __cdata = np.load(file)
        except Exception as __e:
            logging.error("error loading counts file: %s", str(__e))
            return ERROR
        
        _min_pixsize, _ra_patchsize, _dec_patchsize = __cdata['sizes']
        _counts               = __cdata['counts']
        _total                = __cdata['total']
        _masked               = __cdata['masked']
        _ra_cells, _dec_cells = _counts.shape[:2]

        counts.append( _counts )
        total.append( _total )
        masked.append( _masked )

        if __first_file:
            min_pixsize, ra_patchsize, dec_patchsize =_min_pixsize, _ra_patchsize, _dec_patchsize
            ra_cells, dec_cells = _ra_cells, _dec_cells
            continue

        if abs(min_pixsize - _min_pixsize) > 1e-08:
            logging.error(f"regions has different cellsizes: {min_pixsize} and {_min_pixsize}")
            return ERROR
        if abs(ra_patchsize - _ra_patchsize) > 1e-08:
            logging.error(f"regions has different ra patchsizes: {ra_patchsize} and {_ra_patchsize}")
            return ERROR
        if abs(dec_patchsize - _dec_patchsize) > 1e-08:
            logging.error(f"regions has different dec patchsizez: {dec_patchsize} and {_dec_patchsize}")
            return ERROR
        if ra_cells != _ra_cells or dec_cells != _dec_cells:
            logging.error(f"regions has different image shapes: ({ra_cells}, {dec_cells}) and ({_ra_cells}, {_dec_cells})")
            return ERROR
        
    counts = np.concatenate( counts, axis = -1 )
    total  = np.concatenate( total,  axis = -1 )
    masked = np.concatenate( masked, axis = -1 )

    # 
    # estimating the count distribution and save results (serial)
    #

    __estimate_distribution_and_save(counts,
                                     total,
                                     masked, 
                                     max_count, 
                                     subdiv, 
                                     masked_frac,
                                     output_dir)

    return SUCCESS