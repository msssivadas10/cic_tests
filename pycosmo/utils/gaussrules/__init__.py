from typing import Any, Callable 
from scipy.linalg import eigh_tridiagonal
import numpy as np
import struct, os
import os.path as path

PATH = path.dirname( path.realpath( __file__ ) )

def _getJacobian(n: int, alpha: Callable, beta: Callable) -> tuple:
    """
    Compute the Jacobian matrix for n-point Gauss-Konrod quadrature rule.
    """
    # compute the jacobian matrix
    a      = np.zeros( 2*n+1, dtype = float )
    b      = np.zeros( 2*n+1, dtype = float )
    c      = np.zeros( ( n//2+2, 2 ), dtype = float )
    s, t   = 0, 1 # index of s/t array in c
    k      = np.arange( 3*n//2+1 )     
    a[k]   = alpha( k )                
    k      = np.arange( -(-3*n//2)+1 ) 
    b[k]   = beta( k )                 
    c[1,t] = b[n+1]    
    
    for m in range(n-1):
        k        = np.arange( ( m+1 )//2+1 )[::-1] # k = floor((m+1)/2), ..., 1, 0
        l        = m - k
        p        = k + n + 1
        c[k+1,s] = np.cumsum( ( a[p] - a[l] ) * c[k+1,t] + b[p] * c[k,s] - b[l] * c[k+1,s] ) 
        s, t     = t, s # swap `s` and `t` arrays

    j        = np.arange( n//2+1 )[::-1]
    c[j+1,s] = c[j,s]
    
    for m in range(n-1, 2*n-2):
        k        = np.arange( m+1-n, (m-1)//2+1 ) # k = m+1-n, ..., floor((m-1)/2)
        l        = m - k
        j        = n - l - 1
        p        = k + n + 1
        c[j+1,s] = np.cumsum( -(a[p] - a[l]) * c[j+1,t] - b[p] * c[j+1,s] + b[l] * c[j+2,s] )
        j        = j[-1] 
        k        = ( m+1 )//2
        p        = k + n + 1
        if m % 2 == 0: 
            a[p] = a[k] + ( c[j+1,s] - b[p] * c[j+2,s] )/c[j+2,t]
        else: 
            b[p] = c[j+1,s] / c[j+2,s]            
        
        s, t = t, s # swap `s` and `t` arrays
    
    a[2*n] = a[n-1] - b[2*n] * c[1,s] / c[1,t]
    
    return a, np.sqrt( b )

def computeRule(n: int, alpha: Callable, beta: Callable) -> tuple:
    """
    Compute the n-point Gauss-Konrod quadrature rule, for integer n > 1.
    """
    a, b  = _getJacobian( n, alpha, beta )
    beta0 = b[0]**2

    # solve for gauss rule:
    x, wg = eigh_tridiagonal( a[:n], b[1:n] )
    wg    = beta0*wg[0,:]**2
    i     = np.argsort( x )
    x, wg = x[i], wg[i]

    # solve for kronrod rule:
    x, wk = eigh_tridiagonal( a, b[1:] )
    wk    = beta0*wk[0,:]**2
    i     = np.argsort( x )
    x, wk = x[i], wk[i]

    return x, wg, wk

def gaussLegendre(n: int) -> tuple:
    """
    Compute the Gauss-Konrod quadrature rule based on Legendre polynomials.
    """
    def alpha(k: Any) -> Any:
        return np.zeros_like( k, 'float' )

    def beta(k: Any) -> Any:
        return np.where( k != 0, k**2 / (4*k**2 - 1), 2.0 )

    if n < 2:
        raise ValueError("n must be atleast 2")
    return computeRule( n, alpha, beta )

def _saverule(rule: tuple, file: str) -> None:
    """
    Save the quadrature rule to a file (full path must be given).
    """
    n, x, wg, wk = rule
    with open( file, 'wb' ) as f:
        fmt  = f'I{ 2*n+1 }d{ n }d{ 2*n+1 }d'
        f.write( struct.pack( fmt, n, *x, *wg, *wk ) )
    return

def _loadrule(n: int, file: str) -> tuple:
    """
    Load the quadrature rule from a file (full path must be given).
    """
    with open( file, 'rb' ) as f:
        fmt  = f'I{ 2*n+1 }d{ n }d{ 2*n+1 }d'
        data = struct.unpack( fmt, f.read() )
    
    n           = data[0]
    start       = 1
    stop        = start + ( 2*n+1 )
    x           = np.asfarray( data[ start:stop ] )
    start, stop = stop, stop + n
    wg          = np.asfarray( data[ start: stop ] )
    start, stop = stop, stop + ( 2*n+1 )
    wk          = np.asfarray( data[ start: stop ] )
    return x, wg, wk

def legendrerule(n: int) -> tuple:
    r"""
    Calculate the n-point Gauss-Konrod quadrature rule, based on the Legendre polynomials for an 
    integer :math:`n > 1`. It first tries to read the nodes and weights from a file. If the file 
    does not exist, compute the rule and save it.
    """
    file = path.join( PATH, f'gk{ n }' )
    if not path.isfile(file):
        # file not exist - create the file for future use:
        nodes, wg, wk = gaussLegendre( n )
        _saverule( [ n, nodes, wg, wk ], file )
        return nodes, wg, wk
    return _loadrule( n, file )

    
