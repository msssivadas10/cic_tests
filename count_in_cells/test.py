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

import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from patches import PatchData


x = PatchData.load_from( "output/patches.pid" )
# y = pd.read_csv("random.csv.gz")

t, m = x.total, x.masked
u    = np.load("counts.npy")


# fig, (ax1, ax2) = plt.subplots(2, 1)

# a = np.vstack([t[..., i] - m[..., i] for i in range(x.header.n_patches)]).T
# ax1.imshow(a, cmap = 'coolwarm')

# b = np.vstack([u[..., i] for i in range(u.shape[2])]).T
# ax2.imshow(b, cmap = 'coolwarm')

# print( np.allclose(a, b) )

# plt.show()

print( "total  = ", t.sum() )
print( "masked = ", m.sum() )
print( "unmasked = ",  u.sum(), "(", t.sum() - m.sum(), ")" )


# import numpy as np
from scipy.stats import binned_statistic
from scipy.stats import describe

# x = np.random.uniform(0, 1, 100)
# y = binned_statistic( x, values=None, statistic='count', bins=[0., 0.25, 0.5, 0.75, 1.] )
# print(y.statistic)

# x = np.arange(10).reshape((2, 5))
# print(x[:,[True, False, False, True, True]])

n_patches = x.header.n_patches
cs = x.header.pixsize
subdiv = 1

f = 0.05

max_count = 20 #int(np.max(u))
print("max. count =", max_count)

bins = np.arange(max_count + 2) - 0.5
hist = np.zeros((max_count + 1, subdiv + 1, n_patches))
moms = np.zeros((5, subdiv + 1, n_patches))


for l in range(subdiv + 1):

    s = ( (t > 0) & (m < f*t) )

    print(l)
    for p in range(n_patches):
        
        xp = u[s[...,p],p].flatten()
        hist[:,l,p] = binned_statistic( xp, None, 'count', bins ).statistic

        dp = describe( xp )
        moms[:,l,p] = dp.nobs, dp.mean, dp.variance, dp.skewness, dp.kurtosis

    if l == subdiv:
        break

    t = t[::2,:,:] + t[1::2,:,:]
    t = t[:,::2,:] + t[:,1::2,:]

    m = m[::2,:,:] + m[1::2,:,:]
    m = m[:,::2,:] + m[:,1::2,:]

    u = u[::2,:,:] + u[1::2,:,:]
    u = u[:,::2,:] + u[:,1::2,:]



n = np.arange(max_count+1)

plt.figure()
for p in range(n_patches):
    plt.plot(n, hist[:,0,p], '-', color = 'k', alpha = 0.1)
plt.show()