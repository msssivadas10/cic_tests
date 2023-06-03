#!/usr/bin/python3

import os
import numpy as np
import numpy.random as rnd
import pandas as pd
from dataclasses import dataclass

@dataclass(slots = True)
class Circle:
    x: float
    y: float
    r: float

    def is_inside(self, x: float, y: float) -> bool:
        return (x - self.x)**2 + (y - self.y)**2 < self.r**2

def generate_mock_catalog(region: list[float], pixsize: float, navg: float, nrand: int, n_masks: int, 
                          min_masksize: float, max_masksize: float, x_coord: str = 'ra', y_coord: str = 'dec',
                          mask: str = 'mask', seed: int = None, save_path: str = './catalog', **features) -> None:
    
    assert isinstance(pixsize, (float, int))
    assert pixsize > 0.
    assert isinstance(navg, int)
    assert navg > 1
    assert isinstance(nrand, int)
    assert nrand > 1
    assert isinstance(x_coord, str)
    assert isinstance(y_coord, str)
    for feat_name, feat_gen in features.items():
        assert callable(feat_gen)
    
    rng = rnd.default_rng(seed) # set random seed 

    xmin, xmax, ymin, ymax = region

    # generate random catalog
    xr = rng.uniform(xmin, xmax, nrand)
    yr = rng.uniform(ymin, ymax, nrand)


    # generate object catalog, with given poisson count distribution
    x_cells = int((xmax - xmin) / pixsize)
    y_cells = int((ymax - ymin) / pixsize)

    n_features = len(features)
    xo, yo, fo = [], [], np.empty((0, n_features))
    for i in range(x_cells):
        for j in range(y_cells):
            no = rnd.poisson(navg)

            # coords
            xo = np.append(xo, 
                           rng.uniform(0, pixsize, no) + i * pixsize + xmin)
            yo = np.append(yo, 
                           rng.uniform(0, pixsize, no) + j * pixsize + ymin)

            # features
            if n_features:
                fo = np.append(fo,
                            np.stack([feat_gen(rng, no) for feat_name, feat_gen in features.items()], 
                                      axis = 1),
                            axis = 0)
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
        xm = rng.uniform(xmin, xmax)
        ym = rng.uniform(ymin, ymax)
        for j in range(n_bands):
            rm = rng.uniform(min_masksize[j], max_masksize[j])
            m  = Circle(xm, ym, rm)
            mr[:,j] = mr[:,j] | m.is_inside(xr, yr)
            mo[:,j] = mo[:,j] | m.is_inside(xo, yo)

    head, tail = os.path.split(save_path)
    objects = {x_coord: xo,
               y_coord: yo,
               **dict(zip(mask, mo.T))}
    if n_features:
        for i, feat_i in enumerate(features):
            objects[feat_i] = fo[:,i]
    pd.DataFrame(objects
                 ).to_csv(os.path.join(head, 
                                       '_'.join([tail.rsplit('.', 1)[0], 'object.csv'])),
                          index = False)
    pd.DataFrame({x_coord: xr,
                  y_coord: yr,
                  **dict(zip(mask, mr.T))
                }).to_csv(os.path.join(head, 
                                       '_'.join([tail.rsplit('.', 1)[0], 'random.csv'])),
                          index = False)
    return 


# generate_mock_catalog(region       = [0., 20., 0., 8.], 
#                       pixsize      = 0.1, 
#                       navg         = 10, 
#                       nrand        = 100_000, 
#                       n_masks      = 100, 
#                       min_masksize = [0.02, 0.01, 0.03], 
#                       max_masksize = [0.20, 0.06, 0.25], 
#                       x_coord      = 'ra', 
#                       y_coord      = 'dec', 
#                       mask         = ['a_mask', 'b_mask', 'c_mask'],
#                       save_path    = './tests/catalog',
#                       redshift     = lambda rng, n: rng.exponential(0.65, n),
#                       a_magnitude  = lambda rng, n: rng.normal(22.0, 1.7, n),)
