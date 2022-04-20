import time, struct
import numpy as np
from scipy.special import gammaln
from scipy.optimize import curve_fit
from typing import Any, Type

import matplotlib.pyplot as plt
plt.style.use( 'ggplot' )

def timeit(func) -> Any:
    """ 
    Print the execution time for a function.
    """
    def _timeit(*args, **kwargs) -> Any:
        start = time.time()
        out   = func( *args, **kwargs )
        end   = time.time()

        print( 'Execution time: {} sec'.format( end - start ) )
        return out
    return _timeit

# ======================================================================
# cosmology
# ======================================================================

def eisenstein98_withBaryon(cm: 'Cosmology', k: Any, z: float = None) -> Any:
    """
    Transfer function by Eisenstein & Hu, with baryon oscillations.
    """
    theta, Om0, Ob0, h = cm.Tcmb0 / 2.7, cm.Om0, cm.Ob0, cm.h

    k = np.asarray(k) * h #  Mpc^-1

    Omh2, Obh2  = Om0 * h**2, Ob0 * h**2
    fb          = Ob0 / Om0 
    fc          = 1 - fb 
    
    # redshift at equality : eqn. 2 (corrected)
    zp1_eq = (2.50e+04)*Omh2 / theta**4

    # wavenumber at equality : eqn. 3
    k_eq = (7.46e-02)*Omh2 / theta**2

    # redshift at drag epoch : eqn 4
    c1  = 0.313*(1 + 0.607*Omh2**0.674) / Omh2**0.419
    c2  = 0.238*Omh2**0.223
    z_d = 1291.0*(Omh2**0.251)*(1 + c1*Obh2**c2) / (1 + 0.659*Omh2**0.828)

    # baryon - photon momentum density ratio : eqn. 5
    R_const = 31.5*(Obh2 / theta**4) * 1000
    R_eq    = R_const / zp1_eq     # ... at equality epoch
    R_d     = R_const / (1 + z_d)  # ... at drag epoch

    # sound horizon : eqn. 6
    s = (2/3/k_eq)*np.sqrt(6/R_eq)*np.log((np.sqrt(1 + R_d) + np.sqrt(R_eq + R_d)) / (1 + np.sqrt(R_eq)))

    # silk scale : eqn. 7
    k_silk = 1.6*(Obh2**0.52)*(Omh2**0.73)*(1 + (10.4*Omh2)**(-0.95))
    
    q = k/(13.41*k_eq)  # eqn. 10
    x = k*s             # new variable

    # eqn. 11
    a1      = (1 + (32.1*Omh2)**(-0.532))*(46.9*Omh2)**0.670
    a2      = (1 + (45.0*Omh2)**(-0.582))*(12.0*Omh2)**0.424
    alpha_c = (a1**(-fb)) * (a2**(-fb**3))

    # eqn. 12
    b1      = 0.944 / (1 + (458.0*Omh2)**(-0.708))
    b2      = (0.395*Omh2)**(-0.0266)
    beta_c  = 1 / (1 + b1*(fc**b2 - 1))

    # eqn. 18
    f = 1 / (1 + (x/5.4)**4)

    # eqn. 19 and 20
    l_beta     = np.log(np.e + 1.8*beta_c*q)

    c_no_alpha = 14.2           + 386.0 / (1 + 69.9*q**1.08)
    t_no_alpha = l_beta / (l_beta + c_no_alpha*q**2)

    c_alpha    = 14.2 / alpha_c + 386.0 / (1 + 69.9*q**1.08)
    t_alpha    = l_beta / (l_beta + c_alpha*q**2)

    # cold-dark matter part : eqn. 17
    tc = f*t_no_alpha + (1 - f)*t_alpha

    # eqn. 15
    y   = zp1_eq / (1 + z_d)
    y1  = np.sqrt(1 + y)
    Gy  = y*( -6*y1 + (2 + 3*y) * np.log((y1 + 1) / (y1 - 1)) )

    # eqn. 14
    alpha_b = 2.07*(k_eq*s)*Gy*(1 + R_d)**(-3/4)

    # eqn. 24
    beta_b  = 0.5 + fb + (3 - 2*fb)*np.sqrt((17.2*Omh2)**2 + 1)

    # eqn. 23
    beta_node = 8.41*Omh2**0.435

    # eqn. 22
    s_tilde   = s / (1 + (beta_node / x)**3)**(1/3)
    x_tilde   = k*s_tilde

    # eqn. 19 and 20 again
    l_no_beta = np.log(np.e + 1.8*q)
    t_nothing = l_no_beta / (l_no_beta + c_no_alpha*q**2)

    # baryonic part : eqn. 21
    j0 = np.sin(x_tilde) / x_tilde # zero order spherical bessel
    tb = (t_nothing / (1 + (x / 5.2)**2) + ( alpha_b / (1 + (beta_b / x)**3) ) * np.exp(-(k / k_silk)**1.4)) * j0

    return fb * tb + fc * tc # full transfer function : eqn. 16

def eisenstein98_zeroBaryon(cm: 'Cosmology', k: Any, z: float = None) -> Any:
    """
    Transfer function by Eisenstein & Hu, without baryon oscillations.
    """
    theta, Om0, Ob0, h = cm.Tcmb0 / 2.7, cm.Om0, cm.Ob0, cm.h
    Omh2, Obh2, fb     = Om0 * h**2, Ob0 * h**2, Ob0 / Om0

    s = (
            44.5*np.log( 9.83/Omh2 ) / np.sqrt( 1 + 10*Obh2**0.75 )
        ) # eqn. 26
    a_gamma   = (
                    1 - 0.328*np.log( 431*Omh2 ) * fb + 0.38*np.log( 22.3*Omh2 ) * fb**2
                ) # eqn. 31
    gamma_eff = Om0*h * ( 
                            a_gamma + ( 1 - a_gamma ) / ( 1 + ( 0.43*k*s )**4 ) 
                        ) # eqn. 30

    q = k * ( theta**2 / gamma_eff ) # eqn. 28
    l = np.log( 2*np.e + 1.8*q )
    c = 14.2 + 731.0 / ( 1 + 62.5*q )
    return l / ( l + c*q**2 )

def eisenstein98_withNeutrino(cm: 'Cosmology', k: Any, z: float) -> Any:
    """
    Transfer function by Eisenstein & Hu, with neutrino (no baryon oscillations).
    """
    theta, Om0, Ob0, h, Nnu  = cm.Tcmb0 / 2.7, cm.Om0, cm.Ob0, cm.h, cm.Nnu

    k = np.asfarray( k ) * h # Mpc^-1

    Omh2, Obh2 = Om0 * h**2, Ob0 * h**2
    fb, fnu    = Ob0 / Om0, cm.Onu0 / Om0
    fc         = 1.0 - fb - fnu
    fcb, fnb   = fc + fb, fnu + fc

    # redshift at matter-radiation equality: eqn. 1
    zp1_eq = 2.5e+4 * Omh2 / theta**4

    # redshift at drag epoch : eqn 2
    c1  = 0.313*(1 + 0.607*Omh2**0.674) / Omh2**0.419
    c2  = 0.238*Omh2**0.223
    z_d = 1291.0*(Omh2**0.251)*(1 + c1*Obh2**c2) / (1 + 0.659*Omh2**0.828)

    yd  = zp1_eq / (1 + z_d) # eqn 3

    # sound horizon : eqn. 4
    s = 44.5*np.log(9.83 / Omh2) / np.sqrt(1 + 10*Obh2**(3/4))

    q = k * theta**2 / Omh2 # eqn 5


    pc  = 0.25*( 5 - np.sqrt( 1 + 24.0*fc  ) ) # eqn. 14 
    pcb = 0.25*( 5 - np.sqrt( 1 + 24.0*fcb ) ) 

    Dcb = cm.growthSuppressionFactor( q, z, fac = zp1_eq ) # eqn. 12

    # small-scale suppression : eqn. 15
    alpha  = (fc / fcb) * (5 - 2 *(pc + pcb)) / (5 - 4 * pcb)
    alpha *= (1 - 0.533 * fnb + 0.126 * fnb**3) / (1 - 0.193 * np.sqrt(fnu * Nnu) + 0.169 * fnu * Nnu**0.2)
    alpha *= (1 + yd)**(pcb - pc)
    alpha *= (1 + 0.5 * (pc - pcb) * (1 + 1 / (3 - 4 * pc) / (7 - 4 * pcb)) / (1 + yd))

    Gamma_eff = Omh2 * (np.sqrt(alpha) + (1 - np.sqrt(alpha)) / (1 + (0.43 * k * s)**4)) # eqn. 16
    qeff      = k * theta**2 / Gamma_eff

    # transfer function T_sup :
    beta_c = (1 - 0.949 * fnb)**(-1) # eqn. 21
    L      = np.log(np.e + 1.84 * beta_c * np.sqrt(alpha) * qeff) # eqn. 19
    C      = 14.4 + 325 / (1 + 60.5 * qeff**1.08) # eqn. 20
    Tk_sup = L / (L + C * qeff**2) # eqn. 18

    # master function :
    qnu       = 3.92 * q * np.sqrt(Nnu / fnu) # eqn. 23
    Bk        = 1 + (1.24 * fnu**0.64 * Nnu**(0.3 + 0.6 * fnu)) / (qnu**(-1.6) + qnu**0.8) # eqn. 22
    Tk_master = Tk_sup * Bk # eqn. 24 

    return Tk_master * Dcb

def sugiyama96(cm: 'Cosmology', k: Any, z: float = None) -> Any:
    """
    Transfer function Bardeen et al, with correction by Sugiyama.
    """
    theta, Om0, Ob0, h = cm.Tcmb0 / 2.7, cm.Om0, cm.Ob0, cm.h

    q = (
            np.asfarray( k )
                * theta**2 
                / ( Om0*h ) * np.exp( Ob0 + np.sqrt( 2*h ) * Ob0 / Om0 )
        )
    tk = (
            np.log( 1 + 2.34*q ) / ( 2.34*q )
                * (
                        1 + 3.89*q + ( 16.1*q )**2 + ( 5.46*q )**3 + ( 6.71*q )**4
                    )**-0.25 \
         )
    return tk

class Cosmology:
    """
    A flat Lambda-cdm cosmology model.
    """
    __slots__ = (
                    'h', 'Om0', 'Ob0', 'Ode0', 'Onu0', 'Nnu', 'mnu', 'ns', 'Tcmb0', 'A', 
                    'sigma8', 'psmodel', 
                )

    def __init__(self, h: float, Om0: float, Ob0: float, ns: float, sigma8: float = None, Nnu: float = 0.0, mnu: float = 0.0) -> None:
        if h <= 0:
            raise ValueError("'h' must be a positive value")
        elif Om0 < 0 or Om0 > 1:
            raise ValueError("'Om0' must be a value between 0 and 1")

        self.Onu0, self.Nnu, self.mnu = 0.0, 0.0, 0.0
        if mnu < 0 or Nnu < 0:
            raise ValueError("'Nnu' and 'mnu' must be positive numbers")
        Onu0 = Nnu * mnu / 91.5 / h**2
        if Onu0 > Om0:
            raise ValueError("'Onu0' cannot be greater than 'Om0'")

        if Ob0 < 0:
            raise ValueError("'Ob0' must be a positive value")
        elif Ob0 + Onu0 > Om0:
            raise ValueError("sum of 'Ob0' and 'Onu0' cannot be greater than 'Om0'")

        self.h, self.Om0, self.Ob0    = h, Om0, Ob0
        self.Onu0, self.Nnu, self.mnu = Onu0, Nnu, mnu
        
        self.Ode0 = 1 - self.Om0
        self.ns   = ns

        self.A, self.Tcmb0 = 1.0, 2.725
        self.psmodel       = 'eisenstein98_zb'

        self.sigma8        = None
        if sigma8 is not None:
            if sigma8 <= 0:
                raise ValueError("'sigma8' must be a positive value")
            self.normalize( sigma8 )

    def Ez(self, z: Any) -> Any:
        """
        Evolution of hubble parameter.
        """
        zp1 = np.asfarray( z ) + 1
        return np.sqrt( self.Om0 * zp1**3 + self.Ode0 )

    def Hz(self, z: Any) -> Any:
        """
        Evolution of hubble parameter.
        """
        return 100.0 * self.h * self.Ez( z )

    def Omz(self, z: Any) -> Any:
        """
        Evolution of matter density.
        """
        zp1 = np.asfarray( z ) + 1
        y   = self.Om0 * zp1**3
        return y / ( y + self.Ode0 )

    def Odez(self, z: Any) -> Any:
        """
        Evolution of dark-energy density.
        """
        zp1 = np.asfarray( z ) + 1
        return self.Ode0 / ( self.Om0 * zp1**3 + self.Ode0 )

    def gz(self, z: Any) -> Any:
        """
        Fitting function for linear growth factor.
        """
        Om, Ode = self.Omz( z ), self.Odez( z )
        return 2.5 * Om * ( 
                            Om**(4.0/7.0) 
                                - Ode 
                                + ( 1 + Om / 2.0 ) * ( 1 + Ode / 70.0 )
                          )**( -1 )

    def Dz(self, z: Any, fac: float = None) -> Any:
        """
        Linear growth factor.
        """
        def _Dz(z: Any) -> Any:
            return self.gz( z ) / ( np.asfarray( z ) + 1 )
        
        if fac is None:
            fac = 1.0 / _Dz( 0.0 )
        return _Dz( z ) * fac 

    def fz(self, z: Any) -> Any:
        """
        Linear growth rate.
        """
        return self.Omz( z )**0.6

    def growthSuppressionFactor(self, q: Any, z: float, nu: bool = False, fac: float = None) -> Any:
        """
        Suppression of growth of fluctuations in presence of neutrinos.
        """
        q   = np.asfarray( q )    

        if self.Onu0 < 1.0E-08:
            return np.ones_like( q )

        fnu = self.Onu0 / self.Om0
        fcb = 1 - fnu
        pcb = 0.25*( 5 - np.sqrt( 1 + 24.0*fcb ) )
        yfs = 17.2 * fnu * ( 1 + 0.488*fnu**(-7.0/6.0) ) * ( self.Nnu*q / fnu )**2
        D1  = self.Dz( z, fac )    

        x   = ( D1 / ( 1 + yfs ) )**0.7
        if nu:
            return ( fcb**( 0.7 / pcb ) + x )**( pcb / 0.7 ) * D1**( -pcb )
        return ( 1 + x )**( pcb / 0.7 ) * D1**( -pcb )

    def transfer(self, k: Any, z: float = 0) -> Any:
        """
        Transfer function.
        """
        return eisenstein98_zeroBaryon( self, k, z )

    def matterPowerSpectrum(self, k: Any, z: float = 0, dim: bool = True) -> Any:
        """
        Compute linear matter power spectrum. 
        """
        k  = np.asfarray( k )
        pk = self.A * k**self.ns * self.transfer( k, z )**2 * self.Dz( z )**2
        if dim:
            return pk
        return pk * k**3 / ( 2*np.pi**2 )

    def variance(self, r: Any, z: float = 0, ka: float = 1.0E-08, kb: float = 1.0E+08, pts: int = 10001) -> Any:
        """ 
        Compute the linear matter variance.
        """

        def filt(x: Any) -> Any:
            return ( np.sin( x ) - x * np.cos( x )) * 3. / x**3 

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr  = np.outer(r, k)
        y   = self.matterPowerSpectrum( k, z, dim = False ) * filt( kr )**2 
        var = ( 
                    y[ ..., :-1:2 ].sum(-1) + 4 * y[ ..., 1::2 ].sum(-1) + y[ ..., 2::2 ].sum(-1)
              ) * dlnk / 3

        return var if np.ndim(r) else var[0] 

    def correlation(self, r: Any, z: float = 0, ka: float = 1.0E-08, kb: float = 1.0E+08, pts: int = 10001) -> Any:
        """
        Linear matter correlation function.
        """

        def sinc(x: Any) -> Any:
            return np.sinc( x / np.pi )

        k, dlnk = np.linspace( np.log( ka ), np.log( kb ), pts, retstep = True )
        k       = np.exp( k )    

        # integration done in log(k) variable
        kr  = np.outer(r, k)
        y   = self.matterPowerSpectrum( k, z, dim = False ) * sinc( kr )
        cor = ( 
                    y[ ..., :-1:2 ].sum(-1) + 4 * y[ ..., 1::2 ].sum(-1) + y[ ..., 2::2 ].sum(-1)
              ) * dlnk / 3

        return cor if np.ndim(r) else cor[0]

    def powerNorm(self, sigma8: float, **kwargs: Any) -> float:
        """
        Get the power spectrum normalization without setting it.
        """ 
        return sigma8**2 / self.variance( 8.0, **kwargs )

    def normalize(self, sigma8: float, **kwargs: Any) -> None:
        """
        Normalize the power spectrum.
        """
        self.A      = 1.0
        self.sigma8 = sigma8
        self.A      = self.powerNorm( sigma8, **kwargs )


# ======================================================================
# galaxy catalogs
# ====================================================================== 

class Catalog:
    """
    A galaxy catalog in real / redshift space.
    """
    __slots__ = 'position', 'velocity', 'box', 'ngalaxy', 'z', 'zspace'

    def __init__(self, pos: Any, vel: Any, box: Any = None, z: float = None, zspace: bool = False) -> None:
        self.position, self.velocity = np.asfarray( pos ), np.asfarray( vel )
        
        self.box     = np.asfarray( box )
        self.z       = z
        self.zspace  = bool( zspace )
        self.ngalaxy = self.position.shape[0]

    def __getitem__(self, key: str) -> Any:
        return getattr( self, key )

    @classmethod
    def loadCatalog(cls, file: str, z: float = None, zspace: bool = False) -> Type['Catalog']:
        """
        Load a galaxy catalog from file (binary).
        """
        with open(file, 'rb') as f:
            Lx, Ly, Lz, Ngal = struct.unpack('dddi', f.read(28))

            fmt = 'f' * (Ngal * 6)
            cat = struct.unpack(fmt, f.read(Ngal * 6 * 4))
            cat = np.asarray(cat).reshape((Ngal, 6))
            
            return cls( 
                            pos    = cat[ :,:3 ],
                            vel    = cat[ :,3: ],
                            box    = ( Lx, Ly, Lz ),
                            z      = z,
                            zspace = zspace                      
                      )

    def zspaceCatalog(self, cm: Cosmology, dir: int = 1) -> None:
        """
        Convert a catalog to redshift space.
        """
        dir = -1 if dir < 0 else 1
        if dir == ( 1 if self.zspace else -1 ):
            return

        zp1         = self.z + 1
        c           = zp1 / cm.Hz( self.z )
        self.zspace = not self.zspace

        self.position[ :,2 ] += dir * c * self.velocity[ :,3 ]


# ======================================================================
# count-in-cells distributions (log-normal)
# ======================================================================

settings  = {   
                'a'  : -50.0, 
                'b'  :  50.0, 
                'pts': 10001 
            }

def getDistribution(cat: Catalog, ncells: int, bins: int = 21) -> dict:
    """
    Get the count-in-cells distribution from a catalog.
    """
    Lx, Ly, Lz = cat.box

    cic, _ = np.histogramdd( 
                                cat.position,
                                bins  = ( ncells, ncells, ncells ),
                                range = [ ( 0, Lx ), (0, Ly ), ( 0, Lz ) ],
                           )
    cic    = cic.flatten()

    mu, mu_std     = cic.mean(), cic.std()
    varg           = ( ( cic - mu )**2 - mu ) / mu**2 
    varg, varg_std = varg.mean(), varg.std()

    p, edges = np.histogram( cic, bins = bins, density = True )
    return {
                'distr': ( p, edges ),
                'pest' : np.asfarray([ mu, varg ]),
                'pstd' : np.asfarray([ mu_std, varg_std ]),
           }

def bestFit(f: Any, n: Any, p: Any, init: tuple = None) -> tuple:
    """
    Find the best fitting model for the given distribution.
    """
    popt, pcov = curve_fit( f, n, p, init )
    return np.asfarray( popt ), np.sqrt( np.diag( pcov ) )

def cicLognormal(n: Any, nbar: float, varg: float) -> Any:
    """
    Lognormal CIC distribution.
    """
    xa, xb, pts = settings['a'], settings['b'], settings['pts']

    n     = np.asfarray( n )
    x, dx = np.linspace( xa, xb, pts, retstep = True )
    x     = x[:,None]

    lamda = nbar * np.exp( x ) 

    fn = np.exp( 
                    n * np.log( lamda ) - lamda - gammaln( n + 1 )
               )
    
    s  = np.sqrt( np.log( 1 + varg ) )
    fx = np.exp( -0.5 * ( x / s + 0.5 * s )**2 ) / np.sqrt( 2 * np.pi * s**2 )

    y = fn * fx * np.exp( x )
    y = ( 
            y[ :-1:2,: ].sum(0) + 4*y[ 1::2,: ].sum(0) + y[ 2::2,: ].sum(0) 
        ) * dx / 3.0 
    return y




def test_lognormal() -> None:
    distr = getDistribution(
                                Catalog.loadCatalog( 'data/example_lognormal_rlz0.bin' ), 16
                           )
    p, bins = distr[ 'distr' ]
    bins    = bins[:-1] + 0.5 * np.diff( bins )

    popt, err = bestFit( cicLognormal, bins, p, [150, 0.05] ) 

    # print( popt, err )
    # print( distr[ 'pest' ], distr[ 'pstd' ] )

    fig, ax = plt.subplots(figsize = [8,6])  

    ax.bar( bins, p, width = 8, align = 'center', color = 'gray', alpha = 0.3 )
    
    n = np.linspace( bins[0]-10, bins[-1]+10, 51 )
    q = cicLognormal( n, *popt )
    ax.plot( 
                n, q, '-', 
                label = "best: $\\bar N={:.3f}$, $\\sigma_g^2={:.3f}$".format( *popt ) 
           )

    ax.plot(
                [], [], '-',
                label = "cat.: $\\bar N={:.3f}$, $\\sigma_g^2={:.3f}$".format( *distr[ 'pest' ] ) 
           )

    ax.legend()
    plt.show()


if __name__ == '__main__':
    # test_lognormal()
    c = Cosmology(0.70, 0.3, 0.05, 1.0, 0.8)

    plt.figure()

    x = np.logspace( -3, 3, 101)

    y = c.correlation( x ) 
    # y = np.sinc( x*r/np.pi )

    plt.loglog( x, y )

    plt.show()