## tests:

# testing bin merge

import numpy as np

# n = 10
# x = np.array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
# y = np.array([ 1,  0,  0,  6,  4,  1,  2,  5,  3,  2])
# cutoff = 3

# from typing import Any, Tuple

# def __mergeBins(x: Any, y: Any, cutoff: int) -> Tuple[Any, Any]:
#     start, stop = 0, len(y)
#     while start < stop-1:
#         for i in range(start, stop):
#             yi, start = y[i], i
#             if yi < cutoff:
#                 __ix, __iy = (i, i-1) if i == stop-1 else (i+1, i+1)
#                 y[__iy] += yi
#                 stop    -= 1

#                 y, x = np.delete(y, i), np.delete(x, __ix) 
#                 break
#     return x, y

# x, y = __mergeBins(x, y, cutoff)
# print(list(zip(x, y)), )

import pycic

pk_tab = np.loadtxt("./data/power.csv", delimiter = ',')

c = pycic.Cosmology(Om0 = 0.3, Ode0 = 0.7, h = 0.7, ns = 1., pk_tab = pk_tab)
# c.normalisePower(0.8)

pdf = pycic.DeltaDistribution(0., 1.95, c)

pdf._distrParameters()