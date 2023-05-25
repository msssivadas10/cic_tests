#!/usr/bin/python3

import numpy as np, pandas as pd
import struct # for patch image data structure
import logging # for log messages
from utils import check_datafile # to check data 
from utils import is_bad_rect, intersect # check rectangles
from utils import ERROR, SUCCESS
from dataclasses import dataclass
from scipy.stats import binned_statistic_dd
from typing import Any 



#
# =============================================================== 
# patch image data (.pid) file structure 
# =============================================================== 
# PID
#
# BLK {HEADER}
#   RA_SIZE DEC_SIZE PATCH_SIZE
#   PIXSIZE
#   RA_PATCHSIZE DEC_PATCHSIZE
#   REG_RA1 REG_RA2 REG_DEC1 REG_DEC2
#   RA_SHIFT DEC_SHIFT 
# END
#  
# BLK {PATCHES}
#   RA1_0  RA1_1  ... RA1_N 
#   DEC1_0 DEC1_1 ... DEC1_N
#   FLAG_0 FLAG_1 ... FLAG_N
# END
#
# BLK {DATA}
#   V_000 V_001 ... V_LMN
# END
#  

@dataclass(slots = True, frozen = True)
class _PatchImage_Header:
    
    # data shape
    ra_cells : int
    dec_cells: int
    n_patches: int
    # n_patches_ra: int  # no. of patches along ra 
    # n_patches_dec: int # no. of patches along dec

    pixsize      : float # size of a pixel (cell)
    ra_patchsize : float # size of a patch along ra direction
    dec_patchsize: float # size of a patch along dec direction
    
    # corners of the region rectangle
    reg_ra1 : float
    reg_ra2 : float
    reg_dec1: float
    reg_dec2: float

    # shift to apply for coordinates
    ra_shift : float = 0.
    dec_shift: float = 0.

    @property
    def region(self) -> list:
        return [ self.reg_ra1, self.reg_ra2, self.reg_dec1, self.reg_dec2 ]
    

class PatchData:
    r"""
    Store patch data such as the coordinated, sizes, flags and completeness scores.
    """

    __slots__ = 'header', 'patches', 'flags', 'masked', 'total'

    def __init__(self, patches: Any, flags: Any, masked: Any, total: Any, pixsize: float, ra_patchsize: float, 
                 dec_patchsize: float, region: list, ra_shift: float = 0., dec_shift: float = 0.) -> None:

        self.patches = np.asfarray( patches )
        self.flags   = np.array( flags, dtype = 'int' )
        self.masked  = np.array( masked )
        self.total   = np.array( total )


        ra_cells, dec_cells, n_patches       = total.shape
        reg_ra1, reg_ra2, reg_dec1, reg_dec2 = region
        self.header = _PatchImage_Header(ra_cells, 
                                         dec_cells, 
                                         n_patches, 
                                         pixsize, 
                                         ra_patchsize,
                                         dec_patchsize, 
                                         reg_ra1, 
                                         reg_ra2, 
                                         reg_dec1, 
                                         reg_dec2, 
                                         ra_shift, 
                                         dec_shift
                                        )
    
    @classmethod
    def load_from(cls, file: str):
        r"""
        Load patch image data from a file
        """
        try:
            with open( file, 'rb' ) as file:
                buf = file.read()
        except Exception as __e:
            logging.error( f"error loading file: %s", __e )
            return None

        sign, = struct.unpack( '3sx', buf[:4] )
        if sign != b'PID':
            logging.error( "invalid patch image file %s", file )
            return None
        

        # load header 
        header_struct  = struct.Struct( "3s3L9d3sx" )

        start = 4
        stop  = start + header_struct.size
        header = header_struct.unpack(buf[start:stop])[1:-1]

        ra_cells, dec_cells, n_patches       = header[:3]
        n_pixels                             = ra_cells * dec_cells * n_patches
        pixsize, ra_patchsize, dec_patchsize = header[3:6]
        region = header[6:10]
        ra_shift, dec_shift = header[10:12]

        # load patches coordinate and flag
        patches_struct = struct.Struct( "3s{n}d{n}d{n}i3sx".format(n = n_patches ) )

        start = stop
        stop  = start + patches_struct.size
        pdata = patches_struct.unpack(buf[start:stop])[1:-1]
        patches = np.asfarray( pdata[:2*n_patches] ).reshape((2, n_patches)).T
        flags   = np.array( pdata[2*n_patches:] )

        # load values 
        values_struct  = struct.Struct( "3s{m}d3sx".format(m = n_pixels) )

        start  = stop
        stop   = start + values_struct.size
        masked = np.reshape(values_struct.unpack(buf[start:stop])[1:-1], (ra_cells, dec_cells, n_patches))

        start  = stop
        stop   = start + values_struct.size
        total  = np.reshape(values_struct.unpack(buf[start:stop])[1:-1], (ra_cells, dec_cells, n_patches))

        return cls(patches, flags, masked, total, pixsize, ra_patchsize, dec_patchsize, 
                   region, ra_shift, dec_shift)

    def save_as(self, file: str) -> int:
        r"""
        Save the patch image as a binary file
        """

        n_patches = self.header.n_patches
        n_pixels  = self.header.ra_cells * self.header.dec_cells * n_patches
        patches   = self.patches.T

        header_struct  = struct.Struct( "3s3L9d3sx" )
        patches_struct = struct.Struct( "3s{n}d{n}d{n}i3sx".format(n = n_patches ) )
        values_struct  = struct.Struct( "3s{m}d3sx".format(m = n_pixels) )

        buf = (
                struct.pack( "3sx", b'PID' ) # signature block

                    # header block
                    + header_struct.pack(b'BLK', 
                                         self.header.ra_cells, 
                                         self.header.dec_cells,
                                         n_patches, 
                                         self.header.pixsize, 
                                         self.header.ra_patchsize, 
                                         self.header.dec_patchsize, 
                                         *self.header.region, 
                                         self.header.ra_shift, 
                                         self.header.dec_shift, 
                                         b'END' )

                    # patch data bloack
                    + patches_struct.pack( b'BLK', *patches[0,:], *patches[1,:], *self.flags, b'END' ) 

                    # masked count block
                    + values_struct.pack( b'BLK', *self.masked.flatten(), b'END' )

                    # total count block
                    + values_struct.pack( b'BLK', *self.total.flatten(), b'END' )
              )

        with open( file, 'wb' ) as file:
            file.write( buf )

        return SUCCESS
    
    def get_unmasked_fraction(self) -> Any:
        
        frac = 1. - self.get_masked_fraction()
        return frac
    
    def get_masked_fraction(self) -> Any:

        ra_cells, dec_cells, n_patches = self.header.ra_cells, self.header.dec_cells, self.header.n_patches

        frac       = np.ones( (ra_cells, dec_cells, n_patches) )
        _is_finite = ( self.total > 0 ) # mask to avoid nan values
        frac[ _is_finite ] = self.masked[ _is_finite ] / self.total[ _is_finite ]

        return frac
    

def create_patches(reg_rect: list, ra_size: float, dec_size: float, pixsize: float, rdf_path: str, save_path: str, 
                   use_masks: list, reg_to_remove: list = [], rdf_compression: str = 'gzip', chunk_size: int = 1_000, 
                   ra_shift: float = 0., dec_shift: float = 0., rdf_filters: list = [], mpi_comm = None) -> int:
    r"""
    Divide a rectangular region in to rectangular patches of same size. 
    """

    RANK, SIZE = 0, 1
    USE_MPI    = False
    if mpi_comm is not None:
        RANK, SIZE = mpi_comm.rank, mpi_comm.size
        USE_MPI    = ( SIZE > 1 )


    #
    # divide the region into patches
    # 

    if ra_size <= 0. or dec_size <= 0.:
        logging.error( "patch sizes must be positive (ra_size = %f, dec_size = %f)", ra_size, dec_size )
        return ERROR
    
    # check region rectangles
    logging.info( "checking the region rectangle..." )
    if is_bad_rect( reg_rect ):
        return ERROR
    
    logging.info( "checking regions to remove..." )
    for i, bad_region in enumerate( reg_to_remove ):
        if is_bad_rect( bad_region ):
            logging.error( "bad rectangle at %d :(", i )
            return ERROR

    reg_ra1, reg_ra2, reg_dec1, reg_dec2 = reg_rect

    patches, patch_flags = [], []

    ra1, n_ra = reg_ra1, 0
    while 1:

        ra2 = ra1 + ra_size
        if ra2 > reg_ra2:
            break

        dec1, n_dec = reg_dec1, 0
        while 1:

            dec2 = dec1 + dec_size
            if dec2 > reg_dec2:
                break

            bad_patch = False
            for bad_region in reg_to_remove:
                if intersect( [ra1, ra2, dec1, dec2], bad_region ):
                    bad_patch = True
                    break
            
            patches.append( [ra1, dec1] )
            patch_flags.append( bad_patch )
            # print( [ra1, ra2, dec1, dec2] )

            dec1, n_dec = dec2, n_dec + 1

        ra1, n_ra = ra2, n_ra + 1
    
    if not len(patches):
        logging.error( "no patches in the region with given sizes" )
        return ERROR


    #
    # generate patch completeness images
    #
    
    if pixsize <= 0. or pixsize > min( ra_size, dec_size ):
        logging.error( "pixsize (= %f) must be less than the patch sizes, min(%f, %f)", pixsize, ra_size, dec_size )
        return ERROR
    

    ra_bins = np.arange(0., ra_size + pixsize, pixsize)
    ra_bins = ra_bins[ ra_bins <= ra_size ] 

    dec_bins = np.arange(0., dec_size + pixsize, pixsize)
    dec_bins = dec_bins[ dec_bins <= dec_size ] 

    patch_bins = np.arange(0, len(patches) + 1) - 0.5

    image_shape   = ( len(ra_bins)-1, len(dec_bins)-1, len(patch_bins)-1 )

    logging.info( "creating patch images with pixsize = %f, image shape = (%d, %d), %d patches.", pixsize, *image_shape )

    # check data file
    logging.info( "checking random catalog file: '%s'...", rdf_path )

    if not isinstance(use_masks, list):
        logging.error( "use_masks must be a list of mask names to use" )
        return ERROR
    
    required_cols = ['ra', 'dec']
    required_cols.extend( use_masks )
    # for __m in use_masks:
    #     required_cols.append( mask_colname % {'band': __m} )
    if check_datafile( rdf_path, rdf_compression, chunk_size, required_cols, rdf_filters ):
        return ERROR
    
    # data filters: region limits and other specified filters
    rdf_filters.extend([ "ra >= @reg_ra1",  "ra <= @reg_ra2", "dec >= @reg_dec1", "dec <= @reg_dec2" ])
    rdf_filters = "&".join([ "(%s)" % __filter for __filter in rdf_filters ])

    logging.info( "started counting random objects in cell" )
    
    #
    # counting random objects
    #
    total, masked = np.zeros( image_shape ), np.zeros( image_shape )
    with pd.read_csv(rdf_path, header = 0, compression = rdf_compression, chunksize = chunk_size) as rdf_iter:

        # total mask filter: masked in any filter
        mask_filter = '|'.join([ "(%s == True)" % __m for __m in use_masks ]) 

        #
        #  iterate through all chunks and accumulate the counts
        #
        chunk_id = 0
        for rdf in rdf_iter:

            if chunk_id % SIZE != RANK:
                chunk_id += 1
                continue
            chunk_id += 1

            # apply all filters
            rdf = rdf.query( rdf_filters ).reset_index(drop = True)
            if rdf.shape[0] == 0:
                continue
            
            mask_weight = rdf.eval( mask_filter ).to_numpy().astype( 'float' ) # mask value: is masked in any filter?

            # shift the origin to the lower-left corner of the region 
            rdf['ra']  = rdf['ra']  - reg_ra1  - ra_shift
            rdf['dec'] = rdf['dec'] - reg_dec1 - dec_shift

            # get patch id
            i = np.floor( rdf['ra']  / ra_size ).astype( 'int' ) 
            j = np.floor( rdf['dec'] / dec_size ).astype( 'int' )
            rdf['patch'] = n_dec * i + j

            # convert positions to patch coordinates
            rdf['ra']  = rdf['ra']  - i * ra_size
            rdf['dec'] = rdf['dec'] - j * dec_size

            # counting the number in each cell
            total_i, masked_i = binned_statistic_dd(rdf[['ra', 'dec', 'patch']].to_numpy(), 
                                                    values = [ np.ones( rdf.shape[0] ), mask_weight ],
                                                    statistic = 'sum',
                                                    bins = [ ra_bins, dec_bins, patch_bins ]
                                                ).statistic
            
            total  = total + total_i
            masked = masked + masked_i
    

    logging.info( "finished counting random objects in cell" )

    #
    # data communication: combine the results from all process
    #
    if USE_MPI:
        mpi_comm.Barrier() # comm.barrier()

        logging.info( "starting communication...")
        if RANK != 0:

            logging.info( "sent data to rank-0" )

            # send data to process-0
            mpi_comm.Send( total, dest = 0, tag = 10 )  # total count
            mpi_comm.Send( masked, dest = 0, tag = 11 ) # unmasked count

        else:

            # recieve data from other process, if using multiple process
            tmp = np.zeros( image_shape ) # temporary storage
            for src in range(1, SIZE):

                logging.info( "recieving data from rank-%d", src )
                
                # total count
                mpi_comm.Recv( tmp, source = src, tag = 10,  )
                total = total + tmp

                # unmasked count
                mpi_comm.Recv( tmp, source = src, tag = 11,  )
                masked = masked + tmp

        mpi_comm.Barrier() # comm.barrier()
            
    #
    # write patch image files to disc (at process 0) 
    #
    if RANK == 0:

        _ = PatchData(patches       = patches, 
                      flags         = patch_flags, 
                      masked        = masked,
                      total         = total, 
                      pixsize       = pixsize, 
                      ra_patchsize  = ra_size, 
                      dec_patchsize = dec_size, 
                      region        = reg_rect, 
                      ra_shift      = ra_shift, 
                      dec_shift     = dec_shift,
                    ).save_as( save_path )
        
        logging.info( "patch data saved to '%s'", save_path )


    # wait untill all process are completed, if using multiple process 
    if USE_MPI:
        mpi_comm.Barrier() # comm.barrier()

    logging.info( "finished patch image computation job!" )

    return SUCCESS


