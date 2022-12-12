import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sbn
plt.style.use('seaborn')

def test1():

    from pycosmo.cosmology import Cosmology

    c = Cosmology(0.7, 0.3, 0.05, 0.8, 1.0, True)
    print(c.massFunction(1e+10))
    
    return

def test2():

    from pycosmo.analysis.distributions import GQED
    from pycosmo.analysis import SkyRegion, angle as a  
    
    odf = pd.read_csv('../work1/data/wide.csv', header=0, na_values='', skipinitialspace=True, comment='#')
    odf.eval("gri_mask = g_mask & r_mask & i_mask", inplace=True)

    rdf = pd.read_csv('../work1/data/random.csv', header=0, na_values='', skipinitialspace=True, comment='#')
    rdf.eval("gri_mask = g_mask & r_mask & i_mask", inplace=True)

    sky = SkyRegion(a.radian(-0.7), a.radian(0.30001), a.radian(30), a.radian(31.0001))
    sky.setdata(odf, rdf, ra='ra', dec='dec', mask='gri_mask', redshift='photoz_mean', mag='g_mag', angle_unit='degree')
    sky.tileCells(a.radian(arcsec=25), filter=True, selection_frac=0.9)
    count = sky.getCounts()

    px, x = np.histogram(count, bins=21, density=True)
    x  = 0.5*(x[:-1] + x[1:])
    f1 = GQED().fit(x, px, p0=[count.mean(), 0.])


    fig, ax = plt.subplots()

    # ax.set_aspect('equal')
    # sbn.scatterplot(sky.odf, x='ra', y='dec', hue='gri_mask', s=2)
    # for c, n in zip(sky._cells, count):
    #     v = a.degree( c.verts() )
    #     ax.plot(v[:,1], v[:,0], '-', color='black')
    #     ax.text(x=a.degree(c.center[1]), y=a.degree(c.center[0]), s=f'{n}', 
    #             horizontalalignment='center', verticalalignment='center',)

    ax.step(x, px, where='mid', lw=2, label='est.')
    ax.plot(x, f1(x), 'o-', ms=5, label='best fit')
    ax.legend()

    plt.show()
    
    return

