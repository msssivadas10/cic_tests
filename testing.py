#!/usr/bin/python3

import numpy as np
import pycic

x = np.random.uniform(0., 500., (512, 3))
v = np.random.uniform(0., 10.,  (512, 3))

s = pycic.cart2redshift(x, v, 0., 'z')

import matplotlib.pyplot as plt
plt.style.use('ggplot')

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x[:,0], x[:,2], 'o', ms = 2.)
ax2.plot(s[:,0], s[:,2], 'o', ms = 2.)
plt.show()