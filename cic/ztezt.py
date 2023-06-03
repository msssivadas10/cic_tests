import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# from utils import CountData
# pd = CountData.load('tests/output/patch_data.dat')
# pd = CountData.load('tests/output/count_data.dat')
# print(pd.header)


# x1 = np.zeros_like(pd.data[1])
# m = pd.data[0] > 0
# x1[m] = pd.data[1][m] / pd.data[0][m]
# x1 = pd.data[1]
# print(x1.shape)

# fig, (ax1, ax2) = plt.subplots(2,1)
# ax1.imshow(np.concatenate([x1[...,i] for i in range(2)],0).T, cmap = 'coolwarm')

# x1 = 0.5*(x1[0::2,:,:] + x1[1::2,:,:])
# x1 = 0.5*(x1[:,0::2,:] + x1[:,1::2,:])
# ax2.imshow(np.concatenate([x1[...,i] for i in range(2)],0).T, cmap = 'coolwarm')
# plt.imshow(np.concatenate(x1, 1))
# plt.show()


# df1 = pd.read_csv('tests/output/count_histogram_1.csv', header = 0)
# cs = np.loadtxt('tests/output/cellsizes.csv')

# plt.figure()
# plt.loglog()
# x = df1['count'].values
# for i in range(3):
#     y = df1['distr_%d' % i].values
#     Y = y.sum()
#     y = y / Y
#     yerr = df1['error_%d' % i].values / Y

#     plt.errorbar(1+x[::(i+2)], y[::(i+2)], yerr[::(i+2)], capsize = 4, marker = 's', ms = 4, label = "%.3f" % cs[i])
# plt.xlim(1, 200)
# plt.legend(title='pixsize')
# plt.show()