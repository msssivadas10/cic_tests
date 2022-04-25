#!\usr\bin\python3

import numpy as np

def parseMassDefinition(value: str, z: float = None, cm: object = None) -> tuple:
    """
    Parse a mass definition string.
    """
    import re

    m = re.match( r'(\d*)([mc]|[fovir]{3})', value )
    if m is None:
        raise ValueError("cannot parse mass definition: '{}'".format( value ))

    delta, ref = m.groups()
    if ref == 'fof':
        return ( None, 'fof' )

    if ref == 'vir':
        x = cm.Omz( z ) - 1.0
        if cm.flat:
            delta_c = 18.0*np.pi**2 + 82.0*x - 39.0*x**2
        elif cm.Ode0 == 0.0:
            delta_c = 18.0*np.pi**2 + 60.0*x - 32.0*x**2
        else:
            raise ValueError("cannot use 'vir' mass definition")
        return ( round( delta_c ) * cm.criticalDensity( z ), 'so' )
    
    if delta:
        if ref == 'm':
            return ( int( delta ) * cm.Omz( z ), 'so' )
        elif ref == 'c':
            return ( int( delta ) * cm.criticalDensity( z ), 'so' )

    raise ValueError("incorrect mass definition: '{}'".format( value ))

