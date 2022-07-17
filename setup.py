#!/usr/bin/python3

from distutils.core import setup

setup(
    name        = "PyCosmo",
    version     = "1.0",
    description = "A python module to do computations related to large-scale structure formation in cosmology",
    author      = "M. S. Suryan Sivadas",
    url         = "https://github.com/msssivadas10/lss2.git",
    py_modules  = [],
    packages    = [
                    'pycosmo', 
                    'pycosmo.cosmology', 
                    'pycosmo.lss',
                    'pycosmo.power_spectrum',
                    'pycosmo.utils', 
                    'pycosmo.utils.gaussrules',
                    'pycosmo.distributions', 
                    'pycosmo.distributions.density_field', 
                    'pycosmo.distributions.galaxy_count',
                    'pycosmo.nbodytools',
                    'pycosmo.nbodytools.estimators'
                  ]
)
