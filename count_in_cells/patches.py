#!/usr/bin/python3

import numpy as np, pandas as pd
import struct # for patch image data structure
import logging # for log messages
from dataclasses import dataclass
from scipy.stats import binned_statistic_dd
# from mpi4py import MPI
from typing import Any 



#
# =============================================================== 
# patch image data (.pid) file structure 
# =============================================================== 
# PID
#
# BLK {HEAD}
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

    __slots__ = 'header', 'patches', 'flags', 'values'

    def __init__(self, patches: Any, flags: Any, values: Any, pixsize: float, ra_patchsize: float, 
                 dec_patchsize: float, region: list, ra_shift: float = 0., dec_shift: float = 0.) -> None:

        self.patches = np.asfarray( patches )
        self.flags   = np.array( flags, dtype = 'int' )
        self.values  = np.asfarray( values )

        ra_cells, dec_cells, n_patches       = values.shape
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

        with open( file, 'rb' ) as file:
            buf = file.read()

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
        values = np.reshape(values_struct.unpack(buf[start:stop])[1:-1], (ra_cells, dec_cells, n_patches))

        return cls(patches, flags, values, pixsize, ra_patchsize, dec_patchsize, 
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

                    # values block
                    + values_struct.pack( b'BLK', *self.values.flatten(), b'END' )
              )

        with open( file, 'wb' ) as file:
            file.write( buf )

        return 0


def intersect(r1: list, r2: list) -> bool:
    r"""
    Check if two rectangular regions intersect.
    """
    return not (r1[0] >= r2[1] or r1[1] <= r2[0] or r1[3] <= r2[2] or r1[2] >= r2[3])


def create_patches(reg_rect: list, ra_size: float, dec_size: float, pixsize: float, rdf_path: str, 
                   save_path: str, use_masks: list, reg_to_remove: list = [], mask_colname: str = "{}_mask", 
                   rdf_compression: str = 'gzip', chunk_size: int = 1_000, ra_shift: float = 0.,
                   dec_shift: float = 0., mpi_comm = None) -> int:
    r"""
    Divide a rectangular region in to rectangular patches of same size. 
    """

    RANK, SIZE = 0, 1
    USE_MPI    = False
    if mpi_comm is not None:
        RANK, SIZE = mpi_comm.rank, mpi_comm.size
        USE_MPI    = True


    #
    # divide the region into patches
    # 

    if ra_size <= 0. or dec_size <= 0.:
        logging.error( "patch sizes must be positive (ra_size = %f, dec_size = %f)", ra_size, dec_size )
        return 1

    reg_ra1, reg_ra2, reg_dec1, reg_dec2 = reg_rect

    patches, patch_flags = [], []

    ra1 = reg_ra1
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
            for region in reg_to_remove:
                if intersect( [ra1, ra2, dec1, dec2], region ):
                    bad_patch = True
                    break
            
            patches.append( [ra1, dec1] )
            patch_flags.append( bad_patch )
            # print( [ra1, ra2, dec1, dec2] )

            dec1, n_dec = dec2, n_dec + 1

        ra1 = ra2
    
    if not len(patches):
        logging.error( "no patches in the region with given sizes" )
        return 1


    #
    # generate patch completeness images
    #
    
    if pixsize <= 0. or pixsize > min( ra_size, dec_size ):
        logging.error( "cell size must be less than the patch sizes" )
        return 1
    

    ra_bins = np.arange(0., ra_size + pixsize, pixsize)
    ra_bins = ra_bins[ ra_bins <= ra_size ] 

    dec_bins = np.arange(0., dec_size + pixsize, pixsize)
    dec_bins = dec_bins[ dec_bins <= dec_size ] 

    patch_bins = np.arange(0, len(patches) + 1) - 0.5

    image_shape     = ( len(ra_bins)-1, len(dec_bins)-1, len(patch_bins)-1 )
    total, unmasked = np.zeros( image_shape ), np.zeros( image_shape )


    with pd.read_csv(rdf_path, header = 0, compression = rdf_compression, chunksize = chunk_size) as rdf_iter:

        #
        # check if the mask columns are available, using a slice of the table with no rows
        #
        rdf_columns = rdf_iter.get_chunk(0).columns
        mask_filter = []
        for filt in use_masks:

            mask = mask_colname.format( filt )
            if mask not in rdf_columns:
                logging.error( "missing column: '%s'", mask )
            
            mask_filter.append( f"({mask} == False)" )

        mask_filter = '&'.join( mask_filter ) # total mask filter: masked in any filter

        #
        #  iterate through all chunks and accumulate the counts
        #
        chunk_id = 0
        for rdf in rdf_iter:

            if chunk_id % SIZE != RANK:
                chunk_id += 1
                continue
            chunk_id += 1

            # filter out data outside the region
            rdf = rdf.query(
                                "(ra >= @reg_ra1) & (ra <= @reg_ra2) & (dec >= @reg_dec1) & (dec <= @reg_dec2)"
                        ).reset_index(drop = True)

            
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
            total_i, unmasked_i = binned_statistic_dd(rdf[['ra', 'dec', 'patch']].to_numpy(), 
                                                      values = [ np.ones( rdf.shape[0] ), mask_weight ],
                                                      statistic = 'sum',
                                                      bins = [ ra_bins, dec_bins, patch_bins ]
                                                    ).statistic
            
            # print( RANK, chunk_id )

            total    = total + total_i
            unmasked = unmasked + unmasked_i


    logging.info( "finished counting random objects in cell" )

    # wait untill all process are completed, if using multiple process 
    if USE_MPI:
        mpi_comm.Barrier() # comm.barrier()


    # combine the results from all process
    logging.info( "starting communication...")
    if RANK != 0 and USE_MPI:

        logging.info( "sent data to rank 0" )

        # send data to process-0
        mpi_comm.Send( total, dest = 0, tag = 10 )    # total count
        mpi_comm.Send( unmasked, dest = 0, tag = 11 ) # unmasked count

    else:

        # recieve data from other process, if using multiple process
        if USE_MPI:

            tmp = np.zeros( image_shape ) # temporary storage
            for src in range(1, SIZE):

                logging.info( "recieving data from rank-%d", src )
                
                # total count
                mpi_comm.Recv( tmp, source = src, tag = 10,  )
                total = total + tmp

                # unmasked count
                mpi_comm.Recv( tmp, source = src, tag = 11,  )
                unmasked = unmasked + tmp

    
        # calculate the completeness score of the cells
        score     = np.zeros( image_shape )
        is_finite = (total > 0.)
        score[ is_finite ] = unmasked[ is_finite ] / total[ is_finite ]

        # write patch image files 
        _ = PatchData(patches, 
                      patch_flags, 
                      score, 
                      pixsize, 
                      ra_size, 
                      dec_size, 
                      reg_rect, 
                      ra_shift, 
                      dec_shift).save_as( save_path )
        
        logging.info( "patch data saved to %s", save_path )


    # wait untill all process are completed, if using multiple process 
    if USE_MPI:
        mpi_comm.Barrier() # comm.barrier()

    logging.info( "finished patch image computation job!" )
    
    return 0



# if __name__ == '__main__':
#     comm = MPI.COMM_WORLD
#     create_patches([0., 16., 0., 2.], 
#                    4., 
#                    2.,
#                    pixsize = 0.2, 
#                    rdf_path = '/home/ms3/Documents/phd/cosmo/codes/cosmology_codes/random.csv.gz', 
#                    save_path= 'patches.pid',
#                    rdf_compression = 'gzip',
#                    use_masks = ['g'], 
#                    mpi_comm = comm)
