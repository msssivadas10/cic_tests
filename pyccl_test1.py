import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'ggplot' )
import pycosmo.core.cosmology as cm
# import pyccl

def compare(scale: str = None, xlab: str = '', ylab: str = '', col: str = 'C0', ms: int = 4, **subplot_args):
    def decorator(func):
        def _decorator(*args, **kwargs):
            x, y1, y2 = func(*args, **kwargs)
            fig, (ax, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios':[1.0, 0.3]}, **subplot_args)
            if scale:
                eval( f"ax.{ scale }()" )
            if scale in [ 'loglog', 'semilogx' ]:
                ax2.semilogx()
            ax.plot(x, y1, '-', color = col)
            ax.plot(x, y2, 'o', ms = ms, color = col)
            ax.set(xlabel=xlab, ylabel=ylab)
            err = np.abs(1-y1/y2)*100.0
            ax2.plot(x, err, '-o', ms = ms, color = col)
            ax2.set(xlabel=xlab, ylabel='% Error')
            plt.show()
            return x, y1, y2
        return _decorator
    return decorator 

@compare(scale='loglog', figsize=[8,6])
def test_bbks():
    c1 = cm.Cosmology(True, h=0.7, Om0=0.3, Ob0=0.05, ns=1.0, sigma8=0.8, transfer_function='bbks')
    c2 = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=1.0, sigma8=0.8, transfer_function='bbks')
    k  = np.logspace(-3, 3, 21)
    p2 = pyccl.linear_matter_power(c2, k*0.7, 1.0) * 0.7**3
    p1 = c1.matterPowerSpectrum(k, 0.0)
    return k, p1, p2

@compare(scale='loglog', figsize=[8,6])
def test_eh():
    c1 = cm.Cosmology(True, h=0.7, Om0=0.3, Ob0=0.05, ns=1.0, sigma8=0.8, transfer_function='eisenstein98_bao')
    c2 = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=1.0, sigma8=0.8, transfer_function='eisenstein_hu')
    k  = np.logspace(-3, 3, 21)
    p2 = pyccl.linear_matter_power(c2, k*0.7, 1.0) * 0.7**3
    p1 = c1.matterPowerSpectrum(k, 0.0)
    return k, p1, p2

@compare(scale='loglog', figsize=[8,6])
def test_sigma():
    c1 = cm.Cosmology(True, h=0.7, Om0=0.3, Ob0=0.05, ns=1.0, sigma8=0.8, transfer_function='eisenstein98_bao')
    c2 = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=1.0, sigma8=0.8, transfer_function='eisenstein_hu')
    r  = np.logspace(-3, 3, 21)
    p2 = pyccl.sigmaR(c2, r/0.7, 1.0)**2
    p1 = c1.variance(r, 0.0)
    return r, p1, p2


# @compare(scale='loglog', figsize=[8,6])
def test_massfunction():
    m = np.logspace( 6, 15, 501 )

    # from colossus.cosmology import cosmology
    # from colossus.lss import mass_function

    # cosmology.setCosmology('planck18', { 'relspecies': False})

    # n1 = mass_function.massFunction( m, 0, q_out = 'dndlnM', mdef = '200m', model='tinker08', ps_args={'model':'eisenstein98_zb'})

    from pycosmo.core import cosmology as cm
    from pycosmo.lss import massfunction

    # {'flat': True, 'H0': 67.36, 'Om0': 0.3153, 'Ob0': 0.0493, 'sigma8': 0.8111, 'ns': 0.9649}

    c = cm.Cosmology(True, h=0.6774, Om0=0.278, Ob0=0.049, ns=0.9667, sigma8=0.8159, Tcmb0=2.7255, transfer_function='eisenstein98_zb')
    n2 = massfunction.massFunction( c, m, 0, model = 'tinker08', mdef = '200m', out = 'dndlnm' )
    n3 = massfunction.massFunction( c, m, 4, model = 'tinker08', mdef = '200m', out = 'dndlnm' )

    with open( 'hmf.txt', 'w' ) as file:
        file.write( "# halo mass-function for plank-18 cosmology\n" )
        file.write( "# cosmology: Om0=0.278, Ob0=0.049,h=0.6774, ns=0.9667, sigma8=0.8159, Tcmb0=2.7255\n" )
        file.write( "# power spectrum: EH without BAO \n" )
        file.write( "# model: 'tinker08' with overdensity = 200m \n" )
        file.write( "# units: M(Msun/h); dndlnM(h^3 / Mpc^3) \n" )
        file.write( f"# {'M':15s}\t{'dndlnM(z=0)':15s}\t{'dndlnM(z=4)':15s}\n" )
        for mi, y1i, y2i in zip( m, n2, n3 ):
            file.write( f"{mi:15.8e}\t{y1i:15.8e}\t{y2i:15.8e}\n" )

    return #m, n2, n1

if __name__ == '__main__':
    # test_bbks()
    # test_eh()
    # test_sigma()
    test_massfunction()