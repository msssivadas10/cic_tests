r"""

N-Body Simulations
==================

The `nbody.simulation` module may be used for small scale N-body simulations. This is not a well optimized or accurate 
implementation of the N-body simulation techniques and not intented for any real-life applications. It is only a test 
of the simulation code, which may be fully functional in future!
"""

from abc import ABC, abstractmethod
from typing import Any, Union
import warnings
import numpy as np
import numpy.fft as fft
import numpy.random as rnd
import pycosmo.utils.constants as const
from pycosmo.cosmology.cosmo import Cosmology
from pycosmo.nbodytools.data_structures import ParticleData
from pycosmo.nbodytools.estimators import density

class InitialCondition:
    r"""
    A N-body initial condition generator based on first order Lagrangian perturbation theory (Zeldovic 
    approximation).

    Parameters
    ----------
    boxsize: float
        Size of the simulation box in Mpc/h. The box used is a cubical one.
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
        r"""
        Generate the space part of the displacement field.
        """
        boxsize, gridsize = self.boxsize, self.gridsize

        kf    = 2*np.pi / boxsize
        nHalf = gridsize // 2
        nNeg  = gridsize - nHalf - 1

        qx = np.fromfunction( lambda i,j,k: i, shape = ( gridsize, gridsize, nHalf+1 ) )
        qy = np.fromfunction( lambda i,j,k: j, shape = ( gridsize, gridsize, nHalf+1 ) )
        qz = np.fromfunction( lambda i,j,k: k, shape = ( gridsize, gridsize, nHalf+1 ) )

        qx[ qx > nHalf ] -= gridsize
        qy[ qy > nHalf ] -= gridsize
        qz[ qz > nHalf ] -= gridsize
        qx, qy, qz        = qx*kf, qy*kf, qz*kf

        k2          = qx**2 + qy**2 + qz**2
        phi         = np.sqrt( k2 )
        mask        = ( phi != 0.0 )
        phi[ mask ] = np.sqrt( 
                                0.5*self.matterPowerSpectrum( phi[ mask ] )*boxsize**3 
                            ) / k2[ mask ]

        # generate random part of the field (hermitian)
        a = rnd.normal( 0.0, 1.0, ( gridsize, gridsize, nHalf+1 ) )
        b = rnd.normal( 0.0, 1.0, ( gridsize, gridsize, nHalf+1 ) )

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
        r"""
        Compute the particle distribution at redshift :math:`z_i` based on Lagrangian perturbation theory. This can 
        be used to generate initial conditions for N-body simulations.

        Parameters
        ----------
        z: float
            Redshift.
        exact_growth: bool, optional
            Tells whether to use exact values for growth factor. Default is false.

        Returns
        -------
        q: array_like
            Particle positions in Mpc/h.
        v: array_like
            Particle velocities in km/s.

        """
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
        r"""
        Compute the particle distribution at redshift :math:`z_i` based on Lagrangian perturbation theory. This can 
        be used to generate initial conditions for N-body simulations.

        Parameters
        ----------
        z: float
            Redshift.
        exact_growth: bool, optional
            Tells whether to use exact values for growth factor. Default is false.

        Returns
        -------
        pd: ParticleData
            Particle distribution at given redshift.
            
        """
        x, v = self.getInitialCondition( z, exact_growth )
        return ParticleData( x, v, z, self.boxsize, self.gridsize )
        
##########################################################################################################################

class Simulation(ABC):
    r"""
    An abstract base class for N-body simulations. A :class:`Simulation` object can be used run small scale N-body 
    simulations. Any simulators can be created as subclasses of this class. It uses a second-order accurate *leapfrog* 
    integration scheme. Subclasses should define a method to compute the force on the particles.

    Parameters
    ----------
    init: InitialCondition, ParticleData
        Specify the initial condition. It must be initial condition generator object (:class:`InitialCondition`) or the 
        initial particle data (:class:`PartcleData`).
    zi: float
        Initial redshift. If not given, and `init` is a :class:`ParticleData`, take it from that. Otherwise, it should 
        be given.
    cm: Cosmology
        Cosmology model. If not given, and `init` is a :class:`InitialCondition`, take it from that. Otherwise, it should 
        be given.
    exact_growth: bool, optional
        If true, use exact values for the growth factors, calculated through integrating their differential equation. Else, 
        use the fitting forms (default).

    """

    __slots__  = 'currentPos', 'currentVel', 'currentAcc', 'a', 'boxsize', 'gridsize', 'cm', 'zi', 'mass'

    def __init__(self, init: Union[InitialCondition, ParticleData], zi: float = None, cm: Cosmology = None, exact_growth: bool = False) -> None:
        
        if isinstance( init, InitialCondition ):
            cm = init.cm if cm is None else cm
            if zi is None:
                raise ValueError("initial redshift must be given")

            init = init( zi, exact_growth )
        elif not isinstance( init, ParticleData ):
            raise TypeError("init must be an 'InitialCondition' or 'ParticleData'")

        self.zi = init.z
        self.a  = 1./( 1+self.zi )

        self.boxsize  = init.boxsize
        self.gridsize = init.gridsize

        if not isinstance( cm, Cosmology ):
            raise TypeError("cm must be a 'Cosmology'")
        self.cm = cm

        self.mass = ( self.boxsize / self.gridsize )**3 * self.cm.rho_m(0) # mass of a particle in Msun/h

        self.currentPos, self.currentVel = init.position, init.velocity
        
        self.currentAcc = None

    @abstractmethod
    def getForceOnParticle(self) -> None:
        r"""
        Compute the force on a particle, or the acceleration of the particle. This should be implemented in a subclass. 
        The way the force is calculated makes the simualtion schemes different. The value of the calculated force should 
        be assigned to the `currentAcc` attribute. Also, it should have units km/s^2, since the velocities are in km/s 
        and position in Mpc/h units.
        """
        ...

    @property
    def g(self) -> float:
        zp1 = 1./self.a
        return zp1 / self.cm.H( zp1-1 ) * ( 1000.0 / const.MPC ) # sec.

    def updateParticles(self, da: float) -> None:
        r"""
        Update the particle positions by taking a step. A second order accurate *leapfrog* integration scheme is used to 
        update the particle position and velocities. 

        Parameters
        ----------
        da: float
            Time step.

        """
        if self.currentAcc is None:
            self.getForceOnParticle()

        # kick
        self.currentVel += 0.5 * da * self.currentAcc * self.g # km/sec
        self.a          += 0.5 * da

        # drift
        self.currentPos += da * self.currentVel * self.g / self.a**2 * ( self.cm.h * const.MPC / 1000.0 ) # Mpc/h
        self.applyPeriodicBoundary()
        self.getForceOnParticle()

        # kick
        self.currentVel += 0.5 * da * self.currentAcc * self.g # km/sec
        self.a          += 0.5 * da

    def applyPeriodicBoundary(self) -> None:
        r"""
        Apply periodic boundary condition on particle positions.
        """
        # self.currentPos  = self.currentPos % self.boxsize
        
        while 1:
            mask = ( self.currentPos < 0.0 )
            if not np.sum( mask ):
                break
            self.currentPos[ mask ] = self.boxsize + self.currentPos[ mask ]

        while 1:
            mask = ( self.currentPos > self.boxsize )
            if not np.sum( mask ):
                break
            self.currentPos[ mask ] = self.currentPos[ mask ] - self.boxsize


class ParticleParticleSimulation(Simulation):
    r"""
    An N-body simulation based on the *particle-particle* force calculation scheme. This scheme is slow, as it has to 
    look for the force on each particle, due to every other particle. i.e., it has time complexity of the order of 
    :math:`N^2` for :math:`N` particles. The force on the :math:`j`-th particle is

    .. math::
        {\bf F}_j = \sum_{i} GM \frac{ {\bf x}_i - {\bf x}_j }{ \vert {\bf x}_i - {\bf x}_j + \epsilon \vert^3}

    Parameters
    ----------
    init: InitialCondition, ParticleData
        Specify the initial condition. It must be initial condition generator object (:class:`InitialCondition`) or the 
        initial particle data (:class:`PartcleData`).
    zi: float
        Initial redshift. If not given, and `init` is a :class:`ParticleData`, take it from that. Otherwise, it should 
        be given.
    cm: Cosmology
        Cosmology model. If not given, and `init` is a :class:`InitialCondition`, take it from that. Otherwise, it should 
        be given.
    exact_growth: bool, optional
        If true, use exact values for the growth factors, calculated through integrating their differential equation. Else, 
        use the fitting forms (default).

    """

    def getForceOnParticle(self) -> Any:

        GM   = self.mass * const.GMSUN * 1.0e-03 / const.MPC**2 * self.cm.h**2 # h^-2 Mpc^2 km/s^2
        SOFT = 0.01

        N = self.currentPos.shape[0]
        if N > 1e4:
            warnings.warn("force calculation may be very slow for large number of particles")

        def force_j(j: int):
            dx   = self.currentPos - self.currentPos[j,:]         # Mpc/h
            _r3  = ( np.sum( dx**2, axis = -1 ) + SOFT**2 )**-1.5 # ( Mpc/h )^-3
            return GM * np.sum( dx * _r3[:,None], axis = 0 )      # km/s^2

        self.currentAcc = np.asfarray( list( map( force_j, range( N ) ) ) )

class ParticleMeshSimulation(Simulation):
    r"""
    An N-body simulation based on the *particle-mesh* force calculation scheme. This scheme is relatively fast, and 
    use FFT based technique to calculate the force. i.e., by solving the Poisson equation for gravity on a mesh and 
    interpolating the values to the particles. 

    .. math::
        \nabla^2 \phi( {\bf x}, a ) = - \frac{3}{2} \Omega_m(a) H^2(a) a^2 \delta( {\bf x}, a )

    Parameters
    ----------
    init: InitialCondition, ParticleData
        Specify the initial condition. It must be initial condition generator object (:class:`InitialCondition`) or the 
        initial particle data (:class:`PartcleData`).
    zi: float
        Initial redshift. If not given, and `init` is a :class:`ParticleData`, take it from that. Otherwise, it should 
        be given.
    cm: Cosmology
        Cosmology model. If not given, and `init` is a :class:`InitialCondition`, take it from that. Otherwise, it should 
        be given.
    meshsize: int, optional
        Size of the mesh to solve Poisson equation. If not given, use the value of the gridsize from initial conditions.
    exact_growth: bool, optional
        If true, use exact values for the growth factors, calculated through integrating their differential equation. Else, 
        use the fitting forms (default).

    """

    __slots__ = 'meshsize', 'dens'

    def __init__(self, init: Union[InitialCondition, ParticleData], zi: float = None, cm: Cosmology = None, meshsize: int = None, exact_growth: bool = False) -> None:
        super().__init__(init, zi, cm, exact_growth)

        self.meshsize = self.gridsize if meshsize is None else meshsize

    def getForceOnParticle(self) -> None:
        
        gridsize, boxsize = self.meshsize, self.boxsize
        
        delta = density.densityCloudInCell( self.currentPos, boxsize, gridsize )
        delta = delta / delta.mean() - 1.0

        self.dens = density.cicInterpolate( delta, self.currentPos, self.boxsize )

        # fourier transform density
        delta = fft.rfftn( delta )

        # k grid
        kf    = 2*np.pi / boxsize
        nHalf = gridsize // 2

        gx = np.fromfunction( lambda i,j,k: i, shape = ( gridsize, gridsize, nHalf+1 ) )
        gy = np.fromfunction( lambda i,j,k: j, shape = ( gridsize, gridsize, nHalf+1 ) )
        gz = np.fromfunction( lambda i,j,k: k, shape = ( gridsize, gridsize, nHalf+1 ) )

        gx[ gx > nHalf ] -= gridsize
        gy[ gy > nHalf ] -= gridsize
        gz[ gz > nHalf ] -= gridsize
        gx, gy, gz        = gx * kf, gy * kf, gz * kf

        phi  = ( gx**2 + gy**2 + gz**2 ).astype( 'complex' )
        mask = ( phi != 0.0 )

        phi[ mask ] = delta[ mask ] / phi[ mask ]


        # acceleration at grid positions
        z   = 1./self.a - 1
        fac = (
                    1.5 * self.cm.Om( z ) 
                        * ( self.a * ( 100.0 * self.cm.E( z ) ) )**2 
              ) # in (h km/s/Mpc)^2
        
        gx = fft.irfftn( 
                            -1j * gx * phi * fac, ( gridsize, gridsize, gridsize ) 
                       ) 
        gy = fft.irfftn( 
                            -1j * gy * phi * fac, ( gridsize, gridsize, gridsize ) 
                       ) 
        gz = fft.irfftn( 
                            -1j * gz * phi * fac, ( gridsize, gridsize, gridsize ) 
                       ) 

        # interpolate acceleration from grid to points
        gx = density.cicInterpolate( gx, self.currentPos, boxsize ).flatten()
        gy = density.cicInterpolate( gy, self.currentPos, boxsize ).flatten()
        gz = density.cicInterpolate( gz, self.currentPos, boxsize ).flatten()

        self.currentAcc = np.stack([ gx, gy, gz ]).T * ( self.cm.h * 1000.0 / const.MPC ) # km/s^2


