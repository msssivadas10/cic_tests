#!/usr/bin/python3
# "/home/ms3/Documents/phd/cosmo/data/photoz/w2_obj.csv.gz"

#########################################################################################
# PART 4: move log files into output directory
##########################################################################################
# if RANK == 0:
#     logging.info( "moving log files to output directory..." )
#     log_path = os.path.join( output_dir, 'log' )
#     if not os.path.exists( log_path ):
#         os.mkdir( log_path )
#     for file in os.listdir( '.logs' ):
#         os.rename( src = os.path.join( '.logs', file ), dst = os.path.join( log_path, file ) )
#     os.rmdir( '.logs' )

# import matplotlib.pyplot as plt
# import numpy as np, pandas as pd
# from patches import PatchData

# x = PatchData.load_from( "output/patches.pid" )
# y = pd.read_csv("random.csv.gz")

# s = x.get_unmasked_fraction()
# a = np.vstack([s[..., i] for i in range(x.header.n_patches)]).T
# print(x.header.ra_cells, x.header.dec_cells, x.header.n_patches, x.total.shape)
# # print(np.min(b), np.max(b), np.min(a), np.max(a))

# plt.subplots(1, 1)
# plt.imshow(a, cmap = 'coolwarm')
# # plt.scatter(y['ra'], y['dec'], c = y['g_mask'])
# plt.show()

import numpy as np
from scipy.stats import binned_statistic

x = np.random.uniform(0, 1, 100)
y = binned_statistic( x, values=None, statistic='count', bins=[0., 0.25, 0.5, 0.75, 1.] )
print(y.statistic)