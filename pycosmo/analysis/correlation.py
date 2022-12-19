#!/usr/bin/python3

import numpy as np, pandas as pd
import warnings
from scipy.optimize import newton
from sklearn.neighbors import BallTree, KDTree
from pycosmo.analysis.utils import angle, distances
from typing import Any


def estimatorCode(estimator: str) -> int:

    estimator = estimator.lower()
    if estimator in ['ph', 'peebles-hauser']:
        estimator = 0
    elif estimator in ['ls', 'landy-szalay']:
        estimator = 1
    elif estimator in ['dp', 'davis-peebles']:
        estimator = 2
    elif estimator in ['ham', 'hamilton']:
        estimator = 3
    elif estimator in ['hew', 'hewett']:
        estimator = 4
    else:
        raise ValueError(f"invalid value for estimator '{estimator}'")
    
    return estimator

def correlationFromPairCount(nd: int, nr: int, dd: Any, rr: Any = None, dr: Any = None, estimator: str = 'ls') -> Any:
    r"""
    Calculate the correlation function, given the pair counts.
    """

    nd, nr = int(nd), int(nr)
    assert nd > 0 and nr > 0

    estimator = estimatorCode( estimator )

    dd = np.asfarray(dd) / (0.5*nd*(nd-1)) # normalization

    if rr is None and estimator != 2:
        raise ValueError("rr is required for estimators other than Davies-Peebles")
    else:
        rr = np.asfarray(dr) / (0.5*nr*(nr-1)) # normalization
        if np.size(dd) != np.size(rr):
            raise ValueError("dd and rr should have the same size")

    if dr is None and estimator != 0:
        raise ValueError("dr is required for estimators other than Peebles-Hauser")
    else:
        dr = np.asfarray(dr) / (nd*nr) # normalization
        if np.size(dr) != np.size(dd):
            raise ValueError("dd and dr should have the same size")

    y    = np.empty_like(dd)
    y[:] = np.nan

    if estimator == 0: # peebles-hauser

        mask = (rr != 0.)
        y[mask] = dd[mask] / rr[mask] - 1.
        return y

    if estimator == 1: # landy-szalay

        mask = (rr != 0.)
        y[mask] = (dd[mask] - 2*dr[mask] + rr[mask]) / rr[mask]
        return y

    if estimator == 2: # davies-peebles
        
        mask    = (dr != 0.)
        y[mask] = dd[mask] / dr[mask] - 1.
        return y

    if estimator == 3: # hamilton

        mask = (dr != 0.)
        y[mask] = dd[mask] * rr[mask] / dr[mask] - 1.
        return y

    if estimator == 4: # hewett

        mask = (rr != 0.)
        y[mask] = (dd[mask] - dr[mask]) / rr[mask]
        return y

    return


def correlation(sep: Any, odf: pd.DataFrame, rdf: pd.DataFrame, estimator: str = 'ls', c1: str = None, 
                c2: str = None, c3: str = None, mask: str = None, mag: str = None, zrange: tuple = None, 
                magnitude_cutoff: float = None, return_pairs: bool = False, return_corr: bool = True) -> Any: 
    ...


