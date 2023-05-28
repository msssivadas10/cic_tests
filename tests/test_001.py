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

# t, m = x.total, x.masked
# u    = np.load("counts.npy")


# fig, (ax1, ax2) = plt.subplots(2, 1)

# a = np.vstack([t[..., i] - m[..., i] for i in range(x.header.n_patches)]).T
# ax1.imshow(a, cmap = 'coolwarm')

# b = np.vstack([u[..., i] for i in range(u.shape[2])]).T
# ax2.imshow(b, cmap = 'coolwarm')

# print( np.allclose(a, b) )

# plt.show()

# print( "total  = ", t.sum() )
# print( "masked = ", m.sum() )
# print( "unmasked = ",  u.sum(), "(", t.sum() - m.sum(), ")" )


# import numpy as np
# from scipy.stats import binned_statistic
# from scipy.stats import describe

# x = np.random.uniform(0, 1, 100)
# y = binned_statistic( x, values=None, statistic='count', bins=[0., 0.25, 0.5, 0.75, 1.] )
# print(y.statistic)

# x = np.arange(10).reshape((2, 5))
# print(x[:,[True, False, False, True, True]])

# n_patches = x.header.n_patches
# cs = x.header.pixsize
# subdiv = 1

# f = 0.05

# max_count = 20 #int(np.max(u))
# print("max. count =", max_count)

# bins = np.arange(max_count + 2) - 0.5
# hist = np.zeros((max_count + 1, subdiv + 1, n_patches))
# moms = np.zeros((5, subdiv + 1, n_patches))


# for l in range(subdiv + 1):

#     s = ( (t > 0) & (m < f*t) )

#     print(l)
#     for p in range(n_patches):
        
#         xp = u[s[...,p],p].flatten()
#         hist[:,l,p] = binned_statistic( xp, None, 'count', bins ).statistic

#         dp = describe( xp )
#         moms[:,l,p] = dp.nobs, dp.mean, dp.variance, dp.skewness, dp.kurtosis

#     if l == subdiv:
#         break

#     t = t[::2,:,:] + t[1::2,:,:]
#     t = t[:,::2,:] + t[:,1::2,:]

#     m = m[::2,:,:] + m[1::2,:,:]
#     m = m[:,::2,:] + m[:,1::2,:]

#     u = u[::2,:,:] + u[1::2,:,:]
#     u = u[:,::2,:] + u[:,1::2,:]



# n = np.arange(max_count+1)

# plt.figure()
# for p in range(n_patches):
#     plt.plot(n, hist[:,0,p], '-', color = 'k', alpha = 0.1)
# plt.show()

# import yaml, pprint as pp

# with open("tests/param.template.yml", 'r') as file:
#     x = yaml.safe_load(file)
#     pp.pprint(x)

# from argparse import ArgumentParser

# parser = ArgumentParser(prog = 'meas_cic', description = 'Do count-in-cells analysis on data.')
# # parser.add_argument('param_file', help = 'path to the parameter file', type = str)
# parser.add_argument('--opt-file', help = 'argument', type = int, default = 0)


# args = parser.parse_args()
# print( args )

# from mpl_toolkits.basemap import Basemap
# import numpy as np
# import matplotlib.pyplot as plt


# R = 10000000
# m = Basemap(width = 2*R, height = R, projection = 'ortho', lat_0 = 0., lon_0 = 0.)


# parallels = np.arange(-80,81,20)
# m.drawparallels(parallels,labels=[False,True,True,False])

# meridians = np.arange(-180,180,20)
# m.drawmeridians(meridians,labels=[True,False,False,True])

# # draw a black dot at the center.

# lon = np.random.uniform(-90, 90, 100)
# lat = np.random.uniform(-10, 10, 100)
# xpt, ypt = m(lon, lat)
# m.plot(xpt, ypt, 'ko')

# # draw the title.
# plt.show()

# fig = plt.figure(figsize=(8,6))
# ax = fig.add_subplot(111, projection="aitoff")
# ax.scatter(lon*np.pi/180, lat*np.pi/180)
# ax.set_xticklabels(['14h','16h','18h','20h','22h','0h','2h','4h','6h','8h','10h'])
# ax.grid(True)
# plt.show()

# import numpy as np
# from mpl_toolkits.basemap import Basemap
# import matplotlib.pyplot as plt

# fig = plt.figure(figsize = (5,5))

# map = Basemap(projection = 'ortho', lon_0 = 15, lat_0 = 68, resolution = 'l')

# map.drawcoastlines(linewidth = 1)
# map.drawmeridians(np.arange(-180,180,1), linewidth = 1)
# map.drawparallels(np.arange(-80,80,.25), linewidth = 1)

# lllon = 7
# urlon = 21
# lllat = 67
# urlat = 70

# xmin, ymin = map(lllon, lllat)
# xmax, ymax = map(urlon, urlat)

# ax = plt.gca()

# # ax.set_xlim([xmin, xmax])
# # ax.set_ylim([ymin, ymax])

# plt.show()

