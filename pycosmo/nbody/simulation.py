from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np
import numpy.fft as fft
import numpy.random as rnd
from pycosmo.cosmology.cosmo import Cosmology
from pycosmo.nbody.particledata import ParticleData

class InitialCondition:
    r"""
    A N-body initial condition generator based on first order Lagrangian perturbation theory (Zeldovic 
    approximation).

    Parameters
    ----------
    boxsize: float
        Size of the simulation box. The box used is a cubical one.
    gridsize: int
        Number of grid points along each spatial dimension. 
    cm: Cosmology, optional
        Cosmology model to use. Another way to specify a model is to give the parameters as keyword arguments, 
        which will be passed to the :class:'Cosmology' constructor.

    """

    __slots__ = 'boxsize', 'gridsize', 'cm', 'disp1', 'disp2'

    def __init__(self, boxsize: float, gridsize: int, cm: Cosmology = None, **kwargs: Any) -> None:
        if boxsize <= 0.0:
            raise ValueError("boxsize must be a positive number")
        self.boxsize = boxsize

        if not isinstance(gridsize, int):
            raise TypeError("gridsize must be an integer")
        elif gridsize < 1:
            raise ValueError("gridsize must be positive")
        self.gridsize = gridsize

        if cm is None:
            cm = Cosmology( **kwargs )
        else:
            if not isinstance(cm, Cosmology):
                raise TypeError("cm must be a 'Cosmology' object")
        self.cm = cm

        self.disp1 = None
        self.disp2 = None

        self.generateDisplacementField()

    def matterPowerSpectrum(self, k: Any) -> Any:
        return self.cm.matterPowerSpectrum( k, 0.0, dim = True, linear = True )

    def generateDisplacementField(self) -> None:
        boxsize, gridsize = self.boxsize, self.gridsize

        kf    = 2*np.pi / boxsize
        nHalf = gridsize // 2
        nNeg  = gridsize - nHalf - 1

        qx = np.fromfunction( lambda i,j,k: i, shape = ( gridsize, gridsize, nHalf ) )
        qy = np.fromfunction( lambda i,j,k: j, shape = ( gridsize, gridsize, nHalf ) )
        qz = np.fromfunction( lambda i,j,k: k, shape = ( gridsize, gridsize, nHalf ) )

        qx[ qx > nHalf ] -= gridsize
        qy[ qy > nHalf ] -= gridsize
        qx, qy, qz        = qx*kf, qy*kf, qz*kf

        k2          = qx**2 + qy**2 + qz**2
        phi         = np.sqrt( k2 )
        mask        = ( phi != 0.0 )
        phi[ mask ] = np.sqrt( 
                                0.5*self.matterPowerSpectrum( phi[ mask ] )*boxsize**3 
                            ) / k2[ mask ]

        # generate random part of the field (hermitian)
        a = rnd.normal( 0.0, 1.0, ( gridsize, gridsize, nHalf ) )
        b = rnd.normal( 0.0, 1.0, ( gridsize, gridsize, nHalf ) )

        a[ 0, nHalf+1:, 0 ]  = a[ 0, nNeg:0:-1, 0 ]
        a[ nHalf+1:, 0, 0 ]  = a[ nNeg:0:-1, 0, 0 ]
        a[ 1:, nHalf+1:, 0 ] = a[ :0:-1, nNeg:0:-1, 0 ]

        b[ 0, nHalf+1:, 0 ]  = -b[ 0, nNeg:0:-1, 0 ]
        b[ nHalf+1:, 0, 0 ]  = -b[ nNeg:0:-1, 0, 0 ]
        b[ 1:, nHalf+1:, 0 ] = -b[ :0:-1, nNeg:0:-1, 0 ]
        b[ 0, 0, 0 ]         =  0.0

        phi = phi * ( a + b*1j )

        const = boxsize / gridsize

        # first order displacement field (space part)
        vx = fft.irfftn( 
                            1j*qx*phi, ( gridsize, gridsize, gridsize ) 
                    ).flatten() / const**3
        vy = fft.irfftn( 
                            1j*qy*phi, ( gridsize, gridsize, gridsize ) 
                    ).flatten() / const**3
        vz = fft.irfftn( 
                            1j*qz*phi, ( gridsize, gridsize, gridsize ) 
                    ).flatten() / const**3

        self.disp1 = ( vx, vy, vz )
        return

        phi1 = fft.irfftn( 
                            qx*qx*phi, ( gridsize, gridsize, gridsize ) 
                         ) / const**3
        phi2 = fft.irfftn( 
                            qy*qy*phi, ( gridsize, gridsize, gridsize ) 
                         ) / const**3
        phi3 = fft.irfftn( 
                            qz*qz*phi, ( gridsize, gridsize, gridsize ) 
                         ) / const**3
        
        psi  = phi2*phi1 + phi3*phi2 + phi1*phi3

        del phi1, phi2, phi3

        psi -= ( 
                    fft.irfftn( 
                                    qx*qy*phi, ( gridsize, gridsize, gridsize ) 
                              ) / const**3 
               )**2 
        psi -= ( 
                    fft.irfftn( 
                                    qy*qz*phi, ( gridsize, gridsize, gridsize ) 
                              ) / const**3 
               )**2 
        psi -= ( 
                    fft.irfftn( 
                                    qz*qx*phi, ( gridsize, gridsize, gridsize ) 
                              ) / const**3 
               )**2 

        psi         = fft.rfftn( psi )
        psi[ mask ] = psi[ mask ] / k2[ mask ]

        # second order displacement field (space part)
        vx = fft.irfftn( 
                            1j*qx*psi, ( gridsize, gridsize, gridsize ) 
                       ).flatten() / const**3
        vy = fft.irfftn( 
                            1j*qy*psi, ( gridsize, gridsize, gridsize ) 
                       ).flatten() / const**3
        vz = fft.irfftn( 
                            1j*qz*psi, ( gridsize, gridsize, gridsize ) 
                       ).flatten() / const**3
        
        self.disp2 = ( vx, vy, vz )

    def getInitialCondition(self, z: float, exact_growth: bool = False) -> tuple:
        gridsize = self.gridsize
        boxsize  = self.boxsize
        const    = boxsize / gridsize

        # lagrangian positions
        qx = np.fromfunction( 
                            lambda i,j,k: i, ( gridsize, gridsize, gridsize ) 
                        ).flatten() * const
        qy = np.fromfunction( 
                                lambda i,j,k: j, ( gridsize, gridsize, gridsize ) 
                            ).flatten() * const
        qz = np.fromfunction( 
                                lambda i,j,k: k, ( gridsize, gridsize, gridsize ) 
                            ).flatten() * const

        # first order displacement
        D1 = self.cm.Dplus( z, exact = exact_growth )
        F1 = self.cm.f( z, exact = exact_growth ) * 100*self.cm.E( z ) / ( z+1 )

        vx, vy, vz = [ D1 * __v for __v in self.disp1 ]

        qx += vx
        qy += vy
        qz += vz

        vx *= F1
        vy *= F1
        vz *= F1

        # second order displacement 
        if False:
            D2 = -3 * self.cm.Om( z )**( -1./143 ) / 7
            F2 = ( 2*self.cm.Om( z )**( 6./11 ) ) * 100*self.cm.E( z ) / ( z+1 )

            vx2, vy2, vz2 = [ D2 * __v for __v in self.disp2 ]

            qx += vx2
            qy += vy2
            qz += vz2

            vx += F2 * vx2
            vy += F2 * vy2
            vz += F2 * vz2

        qx, qy, qz = qx % boxsize, qy % boxsize, qz % boxsize

        return np.stack([qx, qy, qz]).T, np.stack([vx, vy, vz]).T

    def __call__(self, z: float, exact_growth: bool = False) -> ParticleData:
        x, v = self.getInitialCondition( z, exact_growth )
        return ParticleData( x, v, z, self.boxsize, self.gridsize )

def generateInitialCondition(boxsize: float, gridsize: int, z: float, cm: Cosmology):
    r"""
    Generate initial conditions.
    """
    ic = InitialCondition( boxsize, gridsize, cm )
    return ic( z )
        


class Simulation(ABC):

    __slots__  = 'currentPos', 'currentVel', 'currentAcc', 'a', 'boxsize', 'gridsize', 'cm', 'zi'

    def __init__(self, init: Union[InitialCondition, ParticleData], zi: float, cm: Cosmology = None, exact_growth: bool = False) -> None:
        self.zi = zi
        self.a  = 1./( 1+zi )

        if isinstance( init, InitialCondition ):
            init = init.cm if cm is None else cm
            init = init( zi, exact_growth )
        elif not isinstance( init, ParticleData ):
            raise TypeError("init must be an 'InitialCondition' or 'ParticleData'")

        self.boxsize  = init.boxsize
        self.gridsize = init.gridsize

        if not isinstance( cm, Cosmology ):
            raise TypeError("cm must be a 'Cosmology'")
        self.cm = cm

        self.currentPos, self.currentVel = init.position, init.velocity
        
        self.currentAcc = None

    @abstractmethod
    def getForceOnParticle(self) -> Any:
        ...

    def g(self) -> float:
        zp1 = 1./self.a
        return zp1 / self.cm.E( zp1-1 ) / 100.0

    def updateParticles(self, da: float) -> None:
        if self.currentAcc is None:
            self.currentAcc = self.getForceOnParticle()

        # kick
        self.currentVel += 0.5 * da * self.currentAcc * self.g
        self.a          += 0.5 * da

        # drift
        self.currentPos += da * self.currentVel * self.g / self.a**2
        self.currentAcc  = self.getForceOnParticle()

        # kick
        self.currentVel += 0.5 * da * self.currentAcc * self.g
        self.a          += 0.5 * da


class ParticleParticleSimulation(Simulation):

    def getForceOnParticle(self) -> Any:
        return 
