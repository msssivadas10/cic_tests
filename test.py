import numpy as np
from pycosmo.cosmology import Cosmology

c = Cosmology(0.7, 0.3, 0.05, 0.8, 1.0, True)
print(c.massFunction(1e+10))