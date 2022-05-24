from typing import Any
import numpy as np
import numpy.fft as fft
import numpy.random as rnd
import pycosmo.cosmology as cosmo
from pycosmo.nbody.particledata import ParticleData

class InitialCondition:

    __slots__ = 'boxsize', 'gridsize', 'cm', 'disp1', 'disp2'

    def __init__(self, boxsize: float, gridsize: int, cm: cosmo.Cosmology = None, **kwargs: Any) -> None:
        if boxsize <= 0.0:
            raise ValueError("boxsize must be a positive number")
        self.boxsize = boxsize

        if not isinstance(gridsize, int):
            raise TypeError("gridsize must be an integer")
        elif gridsize < 1:
            raise ValueError("gridsize must be positive")
        self.gridsize = gridsize

        if cm is None:
            cm = cosmo.Cosmology( **kwargs )
        else:
            if not isinstance(cm, cosmo.Cosmology):
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

    def __call__(self, z: float, exact_growth: bool = False) -> ParticleData:
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
        Dz = self.cm.Dplus( z, exact = exact_growth )
        Fz = self.cm.f( z, exact = exact_growth ) * self.cm.H( z ) / ( z+1 )

        vx, vy, vz = [ Dz * __v for __v in self.disp1 ]

        qx += vx
        qy += vy
        qz += vz

        vx *= Fz
        vy *= Fz
        vz *= Fz

        qx, qy, qz = qx % boxsize, qy % boxsize, qz % boxsize

        return ParticleData(
                                np.stack([qx, qy, qz]).T,
                                np.stack([vx, vy, vz]).T,
                                z,
                                boxsize,
                                gridsize
                           )


def generateInitialCondition(boxsize: float, gridsize: int, z: float, cm: cosmo.Cosmology):
    r"""
    Generate initial conditions.
    """
    # if boxsize <= 0.0:
    #     raise ValueError("boxsize must be positive")

    # if not isinstance(gridsize, int):
    #     raise TypeError("gridsize must be an 'int'")
    # elif gridsize < 1:
    #     raise ValueError("gridsize must be positive")

    # if ( z + 1 ) < 0.0:
    #     raise ValueError("redshift must be greater than -1")

    # if not isinstance(cm, cosmo.Cosmology):
    #     raise TypeError("cm must be a 'Cosmology' object")

    # kf = 2*np.pi / boxsize
    
    # nHalf = gridsize // 2
    # nNeg  = gridsize - nHalf - 1

    # qx = np.fromfunction( lambda i,j,k: i, shape = ( gridsize, gridsize, nHalf ) )
    # qy = np.fromfunction( lambda i,j,k: j, shape = ( gridsize, gridsize, nHalf ) )
    # qz = np.fromfunction( lambda i,j,k: k, shape = ( gridsize, gridsize, nHalf ) )

    # qx[ qx > nHalf ] -= gridsize
    # qy[ qy > nHalf ] -= gridsize
    # qx, qy, qz        = qx*kf, qy*kf, qz*kf

    # phi         = np.sqrt( qx**2 + qy**2 + qz**2 )
    # mask        = ( phi != 0.0 )
    # phi[ mask ] = np.sqrt( 
    #                         0.5*cm.matterPowerSpectrum( phi[ mask ], z )*boxsize**3 
    #                      ) / phi[ mask ]**2

    # # generate random part of the field (hermitian)
    # a = rnd.normal( 0.0, 1.0, ( gridsize, gridsize, nHalf ) )
    # b = rnd.normal( 0.0, 1.0, ( gridsize, gridsize, nHalf ) )

    # a[ 0, nHalf+1:, 0 ]  = a[ 0, nNeg:0:-1, 0 ]
    # a[ nHalf+1:, 0, 0 ]  = a[ nNeg:0:-1, 0, 0 ]
    # a[ 1:, nHalf+1:, 0 ] = a[ :0:-1, nNeg:0:-1, 0 ]

    # b[ 0, nHalf+1:, 0 ]  = -b[ 0, nNeg:0:-1, 0 ]
    # b[ nHalf+1:, 0, 0 ]  = -b[ nNeg:0:-1, 0, 0 ]
    # b[ 1:, nHalf+1:, 0 ] = -b[ :0:-1, nNeg:0:-1, 0 ]
    # b[ 0, 0, 0 ]         =  0.0

    # phi = phi * ( a*1j - b )

    # const = boxsize / gridsize

    # # displacement field
    # vx = fft.irfftn( 
    #                     qx*phi, ( gridsize, gridsize, gridsize ) 
    #                ).flatten() / const**3
    # vy = fft.irfftn( 
    #                     qy*phi, ( gridsize, gridsize, gridsize ) 
    #                ).flatten() / const**3
    # vz = fft.irfftn( 
    #                     qz*phi, ( gridsize, gridsize, gridsize ) 
    #                ).flatten() / const**3

    # # positions
    # qx = np.fromfunction( 
    #                         lambda i,j,k: i, ( gridsize, gridsize, gridsize ) 
    #                     ).flatten() * const + vx
    # qy = np.fromfunction( 
    #                         lambda i,j,k: j, ( gridsize, gridsize, gridsize ) 
    #                     ).flatten() * const + vy
    # qz = np.fromfunction( 
    #                         lambda i,j,k: k, ( gridsize, gridsize, gridsize ) 
    #                     ).flatten() * const + vz
    
    # # velocities
    # vfact      = cm.f( z, cm.power_spectrum.use_exact_growth ) * cm.H( z ) / ( z+1 )
    # vx, vy, vz = vfact*vx, vfact*vy, vfact*vz 

    # return ParticleData(
    #                         np.stack([qx, qy, qz]).T,
    #                         np.stack([vx, vy, vz]).T,
    #                         z,
    #                         boxsize,
    #                         gridsize
    #                    )
    ic = InitialCondition( boxsize, gridsize, cm )
    return ic( z )
        

        
