from typing import Any
import numpy as np
import numpy.random as rnd
import numpy.fft as fft
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
from pycic.cosmo import Cosmology

class LognormalCatalog:
    """
    Log-normally distributed galaxy catalog.
    """

    def __init__(self, boxsize: float, ngal: int, ncells: int, z: float, bias: float, cm: Cosmology) -> None:
        self.cm = cm
        self.ngal = ngal
        self.ngrid = ncells
        self.z = z
        self.bias = bias
        self.boxsize = boxsize 

        k           = np.logspace( -4, 4, 1001 )
        self.logpgm = CubicSpline(
                                    np.log( k ),
                                    np.log( self._gfieldPowerSpectrum( k, galaxy = False ) )
                                 )
        self.logpgg = CubicSpline(
                                    np.log( k ),
                                    np.log( self._gfieldPowerSpectrum( k, galaxy = True ) )
                                 )

    def powerSpectrum(self, k: Any, galaxy: bool = False) -> Any:
        y = self.cm.matterPowerSpectrum( k, self.z )
        if galaxy:
            return y * self.bias**2
        return y
    
    def correlation(self, r: Any, galaxy: bool = False, ka: float = 1E-6, kb: float = 1E+6, pts: int = 10001) -> Any:
        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )
        kr      = np.outer( r, k )

        y = self.powerSpectrum( k, galaxy ) * self.sinc( kr ) * k**3
        x = ( y[:, :-1:2].sum(-1) + 4 * y[:, 1::2].sum(-1) + y[:, 2::2].sum(-1) ) * dlnk / 3

        return (x if np.ndim(r) else x[0]) / 2. / np.pi**2

    def gfieldCorrelation(self, r: Any, galaxy: bool = False, ka: float = 1E-6, kb: float = 1E+6, pts: int = 10001) -> Any:
        x = self.correlation( r, galaxy, ka, kb, pts )
        return np.log( 1 + x )

    def sinc(self, x: Any) -> Any:
        y = np.sinc( x / np.pi )
        y[ np.abs( y ) < 1E-8 ] = 0.0
        return y

    def _gfieldPowerSpectrum(self, k: Any, galaxy: bool = False, ra: float = 1E-6, rb: float = 1E+6, pts: int = 10001) -> Any:
        r, dlnr = np.linspace( np.log( ra ), np.log( rb ), pts, retstep = True )
        r       = np.exp( r )
        kr      = np.outer( k, r )

        y = self.gfieldCorrelation( r, galaxy, ra, rb, pts ) * self.sinc( kr ) * r**3
        p = ( y[:, :-1:2].sum(-1) + 4 * y[:, 1::2].sum(-1) + y[:, 2::2].sum(-1) ) * dlnr / 3

        return (p if np.ndim(k) else p[0]) * 4. * np.pi

    def gfieldPowerSpectrum(self, k: Any, galaxy: bool = False, exact: bool = False, *args, **kwargs) -> Any:
        if exact:
            return self._gfieldPowerSpectrum( k, galaxy, *args, **kwargs )
        if galaxy:
            return np.exp( self.logpgg( np.log( k ) ) )
        return np.exp( self.logpgm( np.log( k ) ) )

    def gfieldPhase(self) -> Any:
        nq = self.ngrid // 2
        nn = self.ngrid-nq-1
        t  = rnd.uniform( size = ( self.ngrid, self.ngrid, nq+1 ) )

        t[ nq+1:, nq+1:, 0 ] = -t[ nn:0:-1, nn:0:-1, 0 ] # kz = 0 plane
        t[ nq+1:, 0, 0 ]     = -t[ nn:0:-1, 0, 0 ]       # kz = ky = 0 line
        t[ 0, 0, 0 ]         =  0.0
        return t 

    def gfieldAmplitiude(self) -> Any:
        nq = self.ngrid // 2
        nn = self.ngrid-nq-1
        r  = rnd.uniform( size = ( self.ngrid, self.ngrid, nq+1 ) )

        r[ nq+1:, nq+1:, 0 ] = r[ nn:0:-1, nn:0:-1, 0 ] # kz = 0 plane
        r[ nq+1:, 0, 0 ]     = r[ nn:0:-1, 0, 0 ]       # kz = ky = 0 line
        return np.log( r ) 

    def gfield(self) -> tuple:
        nq = self.ngrid // 2
        kq = 2*np.pi / self.boxsize
        V  = ( self.boxsize / self.ngrid )**3

        kx, ky, kz = np.mgrid[ :self.ngrid, :self.ngrid, :nq+1 ] 
        kx         = np.where( kx > nq, kx - self.ngrid, kx ) * kq
        ky         = np.where( ky > nq, ky - self.ngrid, ky ) * kq
        kz         = np.where( kz > nq, kz - self.ngrid, kz ) * kq
        kk         = kx**2 + ky**2 + kz**2

        mask       = ( kk != 0.0 )

        t = self.gfieldPhase()

        # galaxy field:
        Gg = np.sqrt( kk )
        Gg[ mask ] = np.sqrt(
                                self.gfieldPowerSpectrum( Gg[ mask ], galaxy = True ) * V * 0.5
                            )
        
        Gg = (
                Gg * np.sqrt( -self.gfieldAmplitiude() )
                    * (
                            np.cos( t ) + 1j * np.sin( t )
                      )
             )

        Gg = fft.irfftn( Gg, s = ( self.ngrid, self.ngrid, self.ngrid ) )

        # matter field:
        Gm = np.sqrt( kk )
        Gm[ mask ] = np.sqrt(
                                self.gfieldPowerSpectrum( Gm[ mask ], galaxy = False ) * V * 0.5
                            )
        
        Gm = (
                Gm * np.sqrt( -self.gfieldAmplitiude() )
                    * (
                            np.cos( t ) + 1j * np.sin( t )
                      )
             )
            
        Gm = fft.irfftn( Gm, s = ( self.ngrid, self.ngrid, self.ngrid ) )

        return Gg, Gm
    
    def dfield(self) -> tuple:
        Dg, Dm = self.gfield()

        Dg = np.exp( -Dg.var() + Dg ) - 1
        Dm = np.exp( -Dm.var() + Dm ) - 1
        return Dg, Dm

    def makeCatalog(self) -> Any:
        Dg, Dm = self.dfield()
        
        Ng = self.ngal * ( 1 + Dg ) * ( self.boxsize / self.ngrid )

        pos = []
        for ng in Ng.flatten():
            pos.append(
                            rnd.poisson( ng )
                      )
        return ( pos )
