#!/usr/bin/python3

import numpy as np
import pycic

def test1():
    """ testing `cicDistribution` """
    import lss2

    cs = lss2.CosmoStructure(Om0 = 0.3, Ob0 = 0.05, sigma8 = 0.8, n = 1., h = 0.7, psmodel = "eisenstein98_zb")

    k = np.logspace(-6, 6, 201)
    pk = cs.matterPowerSpectrum(k, 0.)

    cd = pycic.cicDistribution(
                                np.log(np.stack([k, pk], axis = -1)),
                                0., 
                                cs.Om0, cs.Ode0, cs.h, cs.n,
                                1.95
                              )

    # k1 = np.logspace(-8, 8, 501)
    # fk = cd.power(np.log(k1))

    # import matplotlib.pyplot as plt
    # plt.style.use('ggplot')

    # plt.figure()
    # plt.loglog(k1, fk, k, pk)
    # plt.show()

    vlin = cd.varLin()
    print(vlin, cd.varA(vlin), cd.biasA(vlin))

    return

def test2():
    """ testing `CartesianCatalog` """
    x = np.random.uniform(0., 500., (512, 3)) # uniform dist. positions in 500 unit box
    v = np.random.uniform(-10., 10., (512, 3))# uniform dist. velocity in [-10, 10]
    m = np.random.normal(1.e+5, 5., (512, ))  # normally dist. mass 
    # creating the catalog:
    cat = pycic.CartesianCatalog(x, v, mass = m, redshift = 0.)
    print(cat.n)
    
    return

def test3():
    """ tesing `CountMatrix` """
    x = np.random.uniform(0., 500., (512, 3))

    cm = pycic.CountMatrix(x, 4, 500., )
    print(cm.countVector())

    return


if __name__ == '__main__':
    test3()