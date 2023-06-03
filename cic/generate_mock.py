#!/usr/bin/python3

import os
import numpy as np
import numpy.random as rnd
import pandas as pd
from dataclasses import dataclass

@dataclass(slots = True)
class Mask:
    x: float
    y: float
    size: float

    def is_masked(self, x: float, y: float) -> bool:
        return (x - self.x)**2 + (y - self.y)**2 < self.size**2 

def generate_mock_catalog(region: list[float], pixsize: float, navg: float, nrand: int, n_masks: int, 
                          min_masksize: float, max_masksize: float, x_coord: str = 'ra', y_coord: str = 'dec',
                          mask: str = 'mask', seed: int = None, save_path: str = './catalog') -> None:
    
    rnd.seed(seed) # set random seed 

    xmin, xmax, ymin, ymax = region

    # generate random catalog
    xr = rnd.uniform(xmin, xmax, nrand)
    yr = rnd.uniform(ymin, ymax, nrand)


    # generate object catalog, with given poisson count distribution
    x_cells = int((xmax - xmin) / pixsize)
    y_cells = int((ymax - ymin) / pixsize)

    xo, yo = [], []
    for i in range(x_cells):
        for j in range(y_cells):
            no = rnd.poisson(navg)
            xo = np.append(xo, rnd.uniform(0, pixsize, no) + i * pixsize + xmin)
            yo = np.append(yo, rnd.uniform(0, pixsize, no) + j * pixsize + ymin)
    nobjs = xo.shape[0]

    
    # create and apply masks]
    min_masksize = np.asfarray(min_masksize).flatten()
    max_masksize = np.asfarray(max_masksize).flatten()
    mask         = np.array(mask).flatten()

    n_bands = len(max_masksize)
    assert len(min_masksize) == n_bands and len(mask) == n_bands

    mo = np.zeros([nobjs, n_bands], dtype = 'bool')
    mr = np.zeros([nrand, n_bands], dtype = 'bool')
    for i in range(n_masks):
        xm = rnd.uniform(xmin, xmax)
        ym = rnd.uniform(ymin, ymax)
        for j in range(n_bands):
            rm = rnd.uniform(min_masksize[j], max_masksize[j])
            m  = Mask(xm, ym, rm)
            mr[:,j] = mr[:,j] | m.is_masked(xr, yr)
            mo[:,j] = mo[:,j] | m.is_masked(xo, yo)

    head, tail = os.path.split(save_path)
    pd.DataFrame({x_coord: xo,
                  y_coord: yo,
                  **dict(zip(mask, mo.T))
                }).to_csv(os.path.join(head, 
                                       '_'.join([tail.rsplit('.', 1)[0], 'object.csv'])),
                          index = False)
    pd.DataFrame({x_coord: xr,
                  y_coord: yr,
                  **dict(zip(mask, mr.T))
                }).to_csv(os.path.join(head, 
                                       '_'.join([tail.rsplit('.', 1)[0], 'random.csv'])),
                          index = False)
    return 


generate_mock_catalog(region       = [0., 20., 0., 8.], 
                      pixsize      = 0.1, 
                      navg         = 100, 
                      nrand        = 1_000_000, 
                      n_masks      = 100, 
                      min_masksize = [0.02, 0.01, 0.03], 
                      max_masksize = [0.20, 0.06, 0.25], 
                      x_coord      = 'ra', 
                      y_coord      = 'dec', 
                      mask         = ['a_mask', 'b_mask', 'c_mask'],
                      save_path    = './tests/catalog')