import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'ggplot' )
from pycosmo.cosmology import models as cm
import pyccl


def test_bbks():
    c2 = cm.Cosmology(True, h=0.7, Om0=0.3, Ob0=0.05, ns=1.0, sigma8=0.8, psmodel='bbks')


    c1 = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=1.0, sigma8=0.8, transfer_function='bbks')

    k = np.logspace(-3, 3, 21)

    p1 = pyccl.linear_matter_power(c1, k*0.7, 1.0) * 0.7**3
    p2 = c2.matterPowerSpectrum(k, 0.0)

    err = np.abs(1 - p2/p1)*100

    fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[1.0, 0.3]})
    ax.loglog()
    ax.plot(k, p1, '-')
    ax.plot(k, p2, 'o', ms = 4)
    ax2.semilogx()
    ax2.plot(k, err, '-o', ms = 4)
    plt.show()

def test_eh():
    c2 = cm.Cosmology(True, h=0.7, Om0=0.3, Ob0=0.05, ns=1.0, sigma8=0.8, psmodel='eisenstein98_bao')


    c1 = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=1.0, sigma8=0.8, transfer_function='eisenstein_hu')

    k = np.logspace(-3, 3, 21)

    p1 = pyccl.linear_matter_power(c1, k*0.7, 1.0) * 0.7**3
    p2 = c2.matterPowerSpectrum(k, 0.0)

    err = np.abs(1 - p2/p1)*100

    fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[1.0, 0.3]})
    ax.loglog()
    ax.plot(k, p1, '-')
    ax.plot(k, p2, 'o', ms = 4)
    ax2.semilogx()
    ax2.plot(k, err, '-o', ms = 4)
    plt.show()


if __name__ == '__main__':
    test_bbks()
    # test_eh()