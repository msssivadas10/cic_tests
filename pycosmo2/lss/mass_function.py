r"""

Halo Mass-Functions Module
==========================

This module defines the halo mass function abstract base class, :class:`HaloMassFunction` and some of 
the mass-function models available in the literature. All these models are implemented as subclasses 
to the base class. New models can also be created likewise and used with a cosmology model to get the 
mass function in that cosmology.

Available models are


=============== ================================================================
Key             Model                   
=============== ================================================================
`press74`       Press & Schehter (1974)  
`sheth01`       Sheth et al (2001)      
`jenkins01`     Jenkins et al (2001)    
`reed03`        Reed et al (2003)       
`warren06`      Warren et al (2006)     
`reed07`        Reed et al (2007) [2, 3]       
`tinker08`      Tinker et al (2008) [1, 2]
`crocce10`      Crocce et al (2010) [2]    
`courtin10`     Courtin et al (2010)    
=============== =================================================================

1. Only for SO halos. If not specified, mass-function is valid only for FoF halos.
2. Redshift dependent.
3. Cosmology dependent. 

"""

from typing import Any, Union, Tuple
import numpy as np
import warnings
import pycosmo2._bases as base
import pycosmo2.lss.overdensity as od

from pycosmo2._bases import HaloMassFunctionError

Z_DEPENDENT     = 0b0001
COSMO_DEPENDENT = 0b0010
FOF_OVERDENSITY = 0b0100
SO_OVERDENSITY  = 0b1000

class HaloMassFunction(base.HaloMassFunction):

    def __init__(self, cm: base.Cosmology) -> None:

        if not isinstance( cm, base.Cosmology ):
            raise HaloMassFunctionError("argument must be a 'Cosmology' object")
        self.cosmology = cm

        # flags destribing the mass-function properties
        self.flags     = 0
    
    @property
    def zDependent(self) -> bool:
        return self.flags & Z_DEPENDENT
    
    @property
    def cosmoDependent(self) -> bool:
        return self.flags & COSMO_DEPENDENT

    @property
    def mdefs(self) -> Tuple[str]:
        mdefs = []
        if self.flags & FOF_OVERDENSITY:
            mdefs.append( 'fof' )
        if self.flags & SO_OVERDENSITY:
            mdefs.append( 'so' )
        return tuple( mdefs )

    def massFunction(self, m: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = None, out: str = 'dndlnm') -> Any:

        mdefs = self.mdefs
        if overdensity is None:
            if 'fof' in mdefs:
                overdensity = 'fof'
            if 'so' in mdefs:
                overdensity = '200m'

            if len( mdefs ) > 1:
                warnings.warn(f"mass-function '{ self.model }' accepts multiple overdensities, using '{ overdensity }'")
        
        m = np.asfarray( m )

        if np.ndim( z ):
            raise TypeError("z must be a scalar")
        elif z + 1 < 0:
            raise ValueError("z must be greater than -1")

        r        = self.cosmology.lagrangianR( m )
        sigma    = self.cosmology.variance( r, z )
        dlnsdlnm = self.cosmology.dlnsdlnm( r, z )

        f = self.f( sigma, z, overdensity )

        if out == 'f':
            return f

        dndlnM = f * ( self.cosmology.rho_m( z ) / m ) * ( -dlnsdlnm )
        
        if out == 'dndlnm':
            return dndlnM
        if out == 'dndlog10m':
            return dndlnM * 2.302585092994046 # log(M) = ln(M) / ln(10)
        if out == 'dndm':
            return dndlnM / m

        raise HaloMassFunctionError(f"invalid output mode: '{ out }")

        
######################################################################################################

# pre-defined models


class Press74(HaloMassFunction):

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = FOF_OVERDENSITY
        self.model = 'press74'

    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = 'fof') -> Any:
        
        # create the overdensity object
        overdensity = od.overdensity( overdensity )
        if overdensity != od.fof:
            warnings.warn(f"'{ self.model }' mass function is defined for FoF halos")

        nu = self.cosmology.collapseOverdensity( z ) / np.asfarray( sigma )
        f  = np.sqrt( 2 / np.pi ) * nu * np.exp( -0.5 * nu**2 )
        return f

class Sheth01(HaloMassFunction):

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = FOF_OVERDENSITY
        self.model = 'sheth01'

    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = 'fof') -> Any:

        # create the overdensity object
        overdensity = od.overdensity( overdensity )
        if overdensity != od.fof:
            warnings.warn(f"'{ self.model }' mass function is defined for FoF halos")

        A = 0.3222
        a = 0.707
        p = 0.3
        nu = self.cosmology.collapseOverdensity( z ) / np.asarray( sigma )
        f = A * np.sqrt( 2*a / np.pi ) * nu * np.exp( -0.5 * a * nu**2 ) * ( 1.0 + ( nu**2 / a )**-p )
        return f

class Jenkins01(HaloMassFunction):

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = FOF_OVERDENSITY
        self.model = 'jenkins01'

    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = 'fof') -> Any:
        
        # create the overdensity object
        overdensity = od.overdensity( overdensity )
        if overdensity != od.fof:
            warnings.warn(f"'{ self.model }' mass function is defined for FoF halos")

        sigma = np.asfarray( sigma )
        f     = 0.315*( -np.abs( np.log( sigma**-1 ) + 0.61 )**3.8 )
        return f

class Reed03(Sheth01):

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = FOF_OVERDENSITY
        self.model = 'reed03'

    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = 'fof') -> Any:

        # create the overdensity object
        overdensity = od.overdensity( overdensity )
        if overdensity != od.fof:
            warnings.warn(f"'{ self.model }' mass function is defined for FoF halos")

        sigma = np.asfarray( sigma )
        f     = super().f(sigma, z, overdensity) * np.exp( -0.7 / ( sigma * np.cosh( 2*sigma )**5 ) )
        return f

class Warren06(HaloMassFunction):

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = FOF_OVERDENSITY
        self.model = 'warren06'

    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = '200m') -> Any:

        # create the overdensity object
        overdensity = od.overdensity( overdensity )
        if overdensity != od.fof:
            warnings.warn(f"'{ self.model }' mass function is defined for FoF halos")


        A, a, b, c = 0.7234, 1.625, 0.2538, 1.1982

        sigma = np.asfarray( sigma )
        f     = A * ( sigma**-a + b ) * np.exp( -c / sigma**2 )
        return f

class Reed07(HaloMassFunction):

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = FOF_OVERDENSITY | Z_DEPENDENT | COSMO_DEPENDENT
        self.model = 'reed07'

    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = 'fof') -> Any:

        # create the overdensity object
        overdensity = od.overdensity( overdensity )
        if overdensity != od.fof:
            warnings.warn(f"'{ self.model }' mass function is defined for FoF halos")

        cm = self.cosmology

        A, c, ca, p = 0.310, 1.08, 0.764, 0.3

        sigma = np.asfarray( sigma )
        omega = np.sqrt( ca ) * cm.collapseOverdensity( z ) / sigma

        G1    = np.exp( -0.5*( np.log( omega ) - 0.788 )**2 / 0.6**2 )
        G2    = np.exp( -0.5*( np.log( omega ) - 1.138 )**2 / 0.2**2 )

        r     = cm.radius( sigma, z )
        neff  = -6.0*cm.dlnsdlnm( r, 0.0 ) - 3.0
        
        # eqn. 12
        f = (
                A * omega * np.sqrt( 2.0/np.pi ) 
                    * np.exp( -0.5*omega - 0.0325*omega**p / ( neff + 3 )**2 )
                    * ( 1.0 + 1.02*omega**( 2*p ) + 0.6*G1 + 0.4*G2 )
            )
        return f

class Tinker08(HaloMassFunction):

    # find interpolated values from 0-redshift parameter table : table 2
    from scipy.interpolate import CubicSpline

    A  = CubicSpline(
                        [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                        [0.186, 0.200, 0.212, 0.218, 0.248, 0.255, 0.260, 0.260, 0.260],
                    )
    a  = CubicSpline(
                        [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                        [1.47,  1.52,  1.56,  1.61,  1.87,  2.13,  2.30,  2.53,  2.66 ],
                    )
    b  = CubicSpline(
                        [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                        [2.57,  2.25,  2.05,  1.87,  1.59,  1.51,  1.46,  1.44,  1.41 ],
                    )
    c  = CubicSpline(
                        [200,   300,   400,   600,   800,   1200,  1600,  2400,  3200 ],
                        [1.19,  1.27,  1.34,  1.45,  1.58,  1.80,  1.97,  2.24,  2.44 ],
                    )
    
    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = SO_OVERDENSITY | Z_DEPENDENT 
        self.model = 'tinker08'

    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = '200m') -> Any:
        
        sigma  = np.asarray( sigma )

        if np.ndim( z ):
            raise ValueError("parameter 'z' should be a scalar")
        elif z < -1:
            raise ValueError("redshift 'z' must be greater than -1")

        # create the overdensity object and extranct the value
        overdensity = od.overdensity( overdensity )
        if not isinstance( overdensity, od.SO ):
            warnings.warn(f"'{ self.model }' mass function is defined only for SO halos")
        overdensity = overdensity.value( z )

        if overdensity < 200 or overdensity > 3200:
            raise ValueError('`overdensity` value is out of bound. must be within 200 and 3200.')

        # redshift evolution of parameters : 
        zp1   = 1 + z
        A     = self.A( overdensity ) / zp1**0.14 # eqn 5
        a     = self.a( overdensity ) / zp1**0.06 # eqn 6  
        alpha = 10.0**( -( 0.75 / np.log10( overdensity/75 ) )**1.2 ) # eqn 8    
        b     = self.b( overdensity ) / zp1**alpha # eqn 7 
        c     = self.c( overdensity )
        
        f = A * ( 1 + ( b / sigma )**a ) * np.exp( -c / sigma**2 ) # eqn 3
        return f

class Crocce10(HaloMassFunction):

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = FOF_OVERDENSITY | Z_DEPENDENT
        self.model = 'crocce10'

    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = 'fof') -> Any:

        # create the overdensity object
        overdensity = od.overdensity( overdensity )
        if overdensity != od.fof:
            warnings.warn(f"'{ self.model }' mass function is defined for FoF halos")

        if np.ndim( z ):
            raise ValueError("parameter 'z' should be a scalar")
        
        zp1 = z + 1
        if z < -1:
            raise ValueError("redshift 'z' must be greater than -1")

        Az = 0.580 * zp1**-0.130
        az = 1.370 * zp1**-0.150
        bz = 0.300 * zp1**-0.084
        cz = 1.036 * zp1**-0.024
        return Az * ( sigma**-az + bz ) * np.exp( -cz / sigma**2 )

class Courtin10(HaloMassFunction):

    def __init__(self, cm: base.Cosmology) -> None:
        super().__init__(cm)

        self.flags = FOF_OVERDENSITY
        self.model = 'courtin10'

    def f(self, sigma: Any, z: float = 0, overdensity: Union[int, str, od.OverDensity] = 'fof') -> Any:

        # create the overdensity object
        overdensity = od.overdensity( overdensity )
        if overdensity != od.fof:
            warnings.warn(f"'{ self.model }' mass function is defined for FoF halos")

        A  = 0.348
        a  = 0.695
        p  = 0.1
        nu = self.cosmology.collapseOverdensity( z ) / np.asarray( sigma )
        f  = A * np.sqrt( 2*a / np.pi ) * nu * np.exp( -0.5 * a * nu**2 ) * ( 1.0 + ( nu**2 / a )**-p )
        return f


models = {
            'press74'  : Press74,
            'sheth01'  : Sheth01,
            'jenkins01': Jenkins01,
            'reed03'   : Reed03,
            'warren06' : Warren06,
            'reed07'   : Reed07,
            'tinker08' : Tinker08,
            'crocce10' : Crocce10,
            'courtin10': Courtin10,
         }
