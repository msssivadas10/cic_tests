import numpy as np
from typing import Any
from pycosmo.utils.function_table import FunctionTable, addfunction

ftable = FunctionTable( 'transfer_functions' )

@addfunction('eisenstein98_bao', ftable, fclass = 'power', cosmo = True)
def psmodelEisenstein98_withBaryon(cm: object, k: Any, z: float = None) -> Any:
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

@addfunction('eisenstein98_zb', ftable, fclass = 'power', cosmo = True)
def psmodelEisenstein98_zeroBaryon(cm: object, k: Any, z: float = None) -> Any:
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

@addfunction('eisenstein98_nu', ftable, fclass = 'power', z = True, cosmo = True)
def psmodelEisenstein98_withNeutrino(cm: object, k: Any, z: float) -> Any:
    """
    Transfer function by Eisenstein & Hu, with neutrino (no baryon oscillations).
    """
    theta, Om0, Ob0, h, Nnu  = cm.Tcmb0 / 2.7, cm.Om0, cm.Ob0, cm.h, cm.Nnu

    k = np.asfarray( k ) * h # Mpc^-1

    Omh2, Obh2 = Om0 * h**2, Ob0 * h**2
    fb, fnu    = Ob0 / Om0, cm.Onu0 / Om0
    fc         = 1.0 - fb - fnu
    fcb, fnb   = fc + fb, fnu + fc

    if fnu == 0:
        raise ValueError("cannot use 'eisenstein98_withNeutrino' if the model has no neutrino")
    elif z is None:
        raise ValueError("redshift should be given")

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

@addfunction('sugiyama96', ftable, fclass = 'power', cosmo = True)
def psmodelSugiyama96(cm: object, k: Any, z: float = None) -> Any:
    """
    Transfer function Bardeen et al, with correction by Sugiyama.
    """
    theta, Om0, Ob0, h = cm.Tcmb0 / 2.7, cm.Om0, cm.Ob0, cm.h

    q  = (
            np.asfarray( k )
                * theta**2 
                / ( Om0*h ) * np.exp( Ob0 + np.sqrt( 2*h ) * Ob0 / Om0 )
         )
    Tk = (
            np.log( 1 + 2.34*q ) / ( 2.34*q )
                * (
                        1 + 3.89*q + ( 16.1*q )**2 + ( 5.46*q )**3 + ( 6.71*q )**4
                  )**-0.25
         )
    return Tk


########################################################################################################

ftable.setMapping( { 'bbks': 'sugiyama96',  } )

def available(model: str = None) -> Any:
    """
    Check if a model is available.
    """
    if model is None:
        return list( ftable.keys() )
    return ftable.exists( model )

def transfer(cm: object, k: Any, z: float = 0.0, model: str = 'eisenstein98_zb') -> Any:
    """
    Compute the transfer function of given model. 
    """
        
    if not ftable.exists( model ):
        _all = ', '.join( map( lambda name: f"'{ name }'", ftable.keys() ) ) 
        msg  = "transfer function model '{}' is not available. available models are [{}]".format( model, _all )
        raise ValueError( msg )
    
    tfunc = ftable[ model ]
    return tfunc( cm, k, z )

def linearPowerSpectrum(cm: Any, k: Any, z: float = 0.0, dim: bool = True, model: str = 'eisenstein98_zb') -> Any:
    """
    Compute the linear power spectrum.
    """
    k  = np.asfarray( k )
    Pk = cm.A * transfer( cm, k, z, model )**2 * k**cm.ns * cm.Dz( z )**2
    if dim:
        return Pk
    return Pk * k**3 / ( 2*np.pi**2 )
