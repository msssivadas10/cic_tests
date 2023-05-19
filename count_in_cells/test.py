#!/usr/bin/python3
#
#

import numpy as np, pandas as pd
import json # to read configuration file
from astropy.io import fits # to read/write fits files




class Patch:

    __slots__ = ('ra1', 'ra2', 'dec1', 'dec2', 'ra_width', 'dec_width', 'ra_bins', 'dec_bins', 'mask')

    def __init__(self, ra1: float, dec1: float, ra_width: float, dec_width: float = None) -> None:

        if not dec_width:
            dec_width = ra_width
        
        self.ra1, self.dec1 = ra1, dec1
        self.ra2, self.dec2 = ra1 + ra_width, dec1 + dec_width

        self.ra_width, self.dec_width = ra_width, dec_width

        self.ra_bins, self.dec_bins = None, None
        self.mask = None

    def __repr__(self) -> str:
        return f'Patch(ra1={self.ra1}, dec1={self.dec1}, ra2={self.ra2}, dec2={self.dec2})'
    
    def make_cell_edges(self, ra_size: float, dec_size: float = None) -> None:
        
        if not dec_size:
            dec_size = ra_size

        assert dec_size <= self.dec_width
        assert ra_size  <= self.ra_width
        
        ra_bins = np.arange(0., self.ra_width + ra_size, ra_size) 
        self.ra_bins = ra_bins[ ra_bins <= self.ra_width ] + self.ra1

        dec_bins = np.arange(0., self.dec_width + dec_size, dec_size) 
        self.dec_bins = dec_bins[ dec_bins <= self.dec_width ] + self.dec1

        self.mask = np.zeros( shape = (len(ra_bins)-1, len(dec_bins)-1) )
        return
    
def main():

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    p = Patch(0., 0., 5., 4.)
    p.make_cell_edges(0.5)

    x = np.random.uniform(p.ra1, p.ra2, 1000)
    y = np.random.uniform(p.dec1, p.dec2, 1000)

    fig, ax = plt.subplots()

    plt.plot(x, y, 'o', ms = 2)

    rect = patches.Rectangle((p.ra1, p.dec1), p.ra_width, p.dec_width, lw=1, ec='k', fc='none')
    ax.add_patch(rect)

    for xi in p.ra_bins[:-1]:
        for yi in p.dec_bins[:-1]:
            rect = patches.Rectangle((xi, yi), 0.5, 0.5, lw=.1, ec='k', fc='none')
            ax.add_patch(rect)

    plt.hist2d(x, y, bins = [p.ra_bins, p.dec_bins], cmap = 'gray')[0]

    

    plt.show()

    return


if __name__ == '__main__':
    main()