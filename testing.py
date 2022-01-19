#!/usr/bin/python3

import numpy as np
import objects as o

x = np.random.uniform(0., 500., (256, 3))
v = np.random.uniform(-10., 10., (256, 3))

cat = o.Catalog(x, v, z = 0., )

print(cat["nobj"])