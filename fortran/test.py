import numpy as np
import ctypes as ct

flib = ct.CDLL('./c.so')

setCosmology = flib.setCosmology
setCosmology.argtypes = [ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_double, ct.c_int]

Dz = flib.Dz
Dz.argtypes = [ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_int]

setCosmology(0.3, 0.05, 0.7, 1.0, 2.275, 1)

z = np.array([0.0, 1.0, 2.0])
dplus = np.empty_like(z)

Dz(
    z.ctypes.data_as(ct.POINTER(ct.c_double)),
    dplus.ctypes.data_as(ct.POINTER(ct.c_double)),
    ct.c_int(z.shape[0])
)

print(z, dplus / dplus[0])
