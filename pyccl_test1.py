import numpy as np
import matplotlib.pyplot as plt
plt.style.use( 'ggplot' )
import pycosmo.core.cosmology as cm

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
    import pyccl
    c1 = cm.Cosmology(True, h=0.7, Om0=0.3, Ob0=0.05, ns=1.0, sigma8=0.8, transfer_function='bbks')
    c2 = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=1.0, sigma8=0.8, transfer_function='bbks')
    k  = np.logspace(-3, 3, 21)
    p2 = pyccl.linear_matter_power(c2, k*0.7, 1.0) * 0.7**3
    p1 = c1.matterPowerSpectrum(k, 0.0)
    return k, p1, p2

@compare(scale='loglog', figsize=[8,6])
def test_eh():
    import pyccl
    c1 = cm.Cosmology(True, h=0.7, Om0=0.3, Ob0=0.05, ns=1.0, sigma8=0.8, transfer_function='eisenstein98_bao')
    c2 = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=1.0, sigma8=0.8, transfer_function='eisenstein_hu')
    k  = np.logspace(-3, 3, 21)
    p2 = pyccl.linear_matter_power(c2, k*0.7, 1.0) * 0.7**3
    p1 = c1.matterPowerSpectrum(k, 0.0)
    return k, p1, p2

@compare(scale='loglog', figsize=[8,6])
def test_sigma():
    import pyccl
    c1 = cm.Cosmology(True, h=0.7, Om0=0.3, Ob0=0.05, ns=1.0, sigma8=0.8, transfer_function='eisenstein98_bao')
    c2 = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=1.0, sigma8=0.8, transfer_function='eisenstein_hu')
    r  = np.logspace(-3, 3, 21)
    p2 = pyccl.sigmaR(c2, r/0.7, 1.0)**2
    p1 = c1.variance(r, 0.0)
    return r, p1, p2


@compare(scale='loglog', figsize=[8,6])
def test_massfunction():
    import pyccl
    from pycosmo.lss import massfunction
    
    c1 = cm.Cosmology(True, h=0.7, Om0=0.3, Ob0=0.05, ns=1.0, sigma8=0.8, transfer_function='eisenstein98_bao')
    c2 = pyccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=1.0, sigma8=0.8, transfer_function='eisenstein_hu', mass_function='tinker')
    m  = np.logspace( 6, 15, 21 )
    n1 = massfunction.massFunction(c1, m, 0.0, mdef = '200m', out = 'dndlnm')

    # NOTE: pyccl accept mass in Msun/h and return dn/dlog10M in Mpc^-3. change this 
    # to dn/dlnM in h^3/Mpc^3 to compare  
    n2 = pyccl.massfunction.massfunc( c2, m/0.7, 1.0, 200 ) / np.log(10) / 0.7**3

    return m, n2, n1

if __name__ == '__main__':
    # test_bbks()
    # test_eh()
    # test_sigma()
    test_massfunction()