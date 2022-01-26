#!/usr/bin/python3

import numpy as np
import pycic
import lss2

cs = lss2.CosmoStructure(Om0 = 0.3, Ob0 = 0.05, sigma8 = 0.8, n = 1., h = 0.7, psmodel = "eisenstein98_zb")

k = np.logspace(-6, 6, 201)
pk = cs.matterPowerSpectrum(k, 0.)

cd = pycic.cicDistribution(
    np.log(np.stack([k, pk], axis = -1)),
    0., 
    cs.Om0, cs.Ode0, cs.h, 
    500. / 8
)

# k1 = np.logspace(-8, 8, 501)
# fk = cd.power(np.log(k1))

# import matplotlib.pyplot as plt
# plt.style.use('ggplot')

# plt.figure()
# plt.loglog(k1, fk, k, pk)
# plt.show()

print(cd.varLin(), cd.varA(), cd.biasA())