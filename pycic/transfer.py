#!\usr\bin\python3
r"""
Transfer Functions
==================

This module contains some of the transfer function models. Models by Bardeen et al (1986) 
or BBKS with Sugiyama (1995) correction [1]_ and Eisentein-Hu models (with and without 
BAO [2]_, and mixed dark-matter model [3]_) are currently available.

References
----------
.. [1] A. Meiksin, matrin White and J. A. Peacock. Baryonic signatures in large-scale structure, 
       Mon. Not. R. Astron. Soc. 304, 851-864, 1999.
.. [2] Daniel J. Eisenstein and Wayne Hu. Baryonic Features in the Matter Transfer Function, 
       `arXive:astro-ph/9709112v1, <http://arXiv.org/abs/astro-ph/9709112v1>`_, 1997.
.. [3] Daniel J. Eisenstein and Wayne Hu. Power Spectra for Cold Dark Matter and its Variants, 
       `arXive:astro-ph/9710252v1, <http://arXiv.org/abs/astro-ph/9710252v1>`_, 1997.
"""

from typing import Any
import numpy as np

def modelSugiyama95(k: Any, *, h: float, Om0: float, Ob0: float, **kwargs) -> Any:
    r""" 
    Transfer function by Bardeen et al (1986) with correction given by Sugiyama (1995). 
    All arguments except the first are keyword arguments.

    Parameters
    ----------
    k: array_like
        Wavenumber in units of h/Mpc.
    h: float
        Present Hubble parameter in 100 km/sec/Mpc.
    Om0: float
        Present matter density.
    Ob0: float
        Present baryon density.
    **kwargs: ignored

    Returns
    -------
    tk: array_like
        Transfer function values. Has the same shape as `k`.

    Examples
    ---------
    >>> k = np.array([1.e-1, 1.e+0, 1.e+1])
    >>> modelSugiyama95(k, h = 0.7, Om0 = 0.3, Ob0 = 0.05)
    array([1.37520093e-01, 4.65026996e-03, 8.50707301e-05])

    """
    k = np.asarray(k)
    q = k / Om0 / h * np.exp(Ob0 + np.sqrt(2*h) * Ob0 / Om0) # eqn. 4

    # transfer function: eqn. 3
    tk = np.log(1 + 2.34*q) / (2.34*q) * (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)

    # for smaller k, transfer function shound return 1, instead of 0
    tk[k < 1e-8] = 1.0

    return tk

def modelEisenstein98_zeroBaryon(k: Any, *, h: float, Om0: float, Ob0: float, Tcmb0: float, **kwargs) -> Any:
    r"""
    Transfer function by Eisenstein and Hu (1998), without baryon oscillations. All 
    arguments except the first are keyword arguments.

    Parameters
    ----------
    k: array_like
        Wavenumber in units of h/Mpc.
    h: float
        Present Hubble parameter in 100 km/sec/Mpc.
    Om0: float
        Present matter density.
    Ob0: float
        Present baryon density.
    Tcmb0: float
        Present temperature of cosmic microwave background in kelvin.
    **kwargs: ignored

    Returns
    -------
    tk: array_like
        Transfer function values. Has the same shape as `k`.

    Examples
    ---------
    >>> k = np.array([1.e-1, 1.e+0, 1.e+1])
    >>> modelEisenstein98_zeroBaryon(k, h = 0.7, Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725)
    array([1.31173577e-01, 4.55592401e-03, 8.67666599e-05])

    """
    k     = np.asarray(k)
    h2    = h * h
    Omh2  = Om0 * h2
    Obh2  = Ob0 * h2
    theta = Tcmb0 / 2.7
    fb    = Ob0 / Om0

    # sound horizon: eqn. 26
    s = 44.5*np.log(9.83 / Omh2) / np.sqrt(1 + 10*Obh2**0.75)

    # eqn. 31
    a_gamma   = 1 - 0.328*np.log(431*Omh2)*fb + 0.38*np.log(22.3*Omh2)*fb**2

    # eqn. 30
    gamma_eff = Om0 * h * (a_gamma + (1 - a_gamma) / (1 + (0.43*k*s)**4))

    q = k * (theta**2 / gamma_eff) # eqn. 28
    l = np.log(2*np.e + 1.8*q)
    c = 14.2 + 731.0 / (1 + 62.5*q)
    return l / (l + c*q**2)

def modelEisenstein98_mixedDarkmatter(k: Any, *, h: float, Om0: float, Ob0: float, Ode0: float, Onu0: float, Nnu: float, Tcmb0: float, z: float = ..., Dz: float = ..., out: str = 'cb') -> Any:
    r"""
    Transfer function by Eisenstein and Hu (1998), with mixed dark-matter. This model 
    is redshift dependent. All arguments except the first are keyword arguments.

    Parameters
    ----------
    k: array_like
        Wavenumber in units of h/Mpc.
    h: float
        Present Hubble parameter in 100 km/sec/Mpc.
    Om0: float
        Present matter density.
    Ob0: float
        Present baryon density.
    Ode0: float
        Present dark-energy density.
    Onu0: float
        Present neutrino (massive) density.
    Nnu: float
        Number of (massive) neutrinos.
    Tcmb0: float
        Present temperature of cosmic microwave background in kelvin.
    z: float, optional
        Redshift. If given, find the growth factor using a fitting formula. If not given, 
        the growth factor should be given.
    Dz: float, optional
        Growth factor. This must be given if no redshift is specified.
    out: str, {'cb', 'cbnu', 'nu'}, optional
        Specify the combination, whose transfer function is returned. `cb` is for CDM-
        baryon combination (default), `cbnu` for CDM-baryon-neutrino and `nu` for only 
        neutrino.  
    **kwargs: ignored

    Returns
    -------
    tk: array_like
        Transfer function values. Has the same shape as `k`.

    Examples
    ---------
    If there is one massive neutrino species and its density :math:`\Omega_\nu = 0.1`, 
    then for the cold darkmatter - baryon combination, transfer function is 

    >>> k = np.array([1.e-1, 1.e+0, 1.e+1])
    >>> modelEisenstein98_mixedDarkmatter(k, h = 0.7, Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725, 
    ... Ode0 = 0.7, Onu0 = 0.1, Nnu = 1, z = 0)
    array([6.75956823e-02, 8.70765246e-04, 1.26614585e-05])

    For the CDM-baryon-neutrino combination, 
    >>> modelEisenstein98_mixedDarkmatter(k, h = 0.7, Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725, 
    ... Ode0 = 0.7, Onu0 = 0.1, Nnu = 1, z = 0, out = 'cbnu')
    array([6.69039059e-02, 7.46736180e-04, 8.66926371e-06])

    and, for neutrinos, 
    >>> modelEisenstein98_mixedDarkmatter(k, h = 0.7, Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725, 
    ... Ode0 = 0.7, Onu0 = 0.1, Nnu = 1, z = 0, out = 'nu')
    array([6.55203530e-02, 4.98678048e-04, 6.84874157e-07])

    """
    k     = np.asarray(k) * h # wavenumber in units of 1/Mpc
    h2    = h * h
    Omh2  = Om0 * h2
    Obh2  = Ob0 * h2
    theta = Tcmb0 / 2.7
    fb    = Ob0 / Om0    # baryon fraction   
    fnu   = Onu0 / Om0   # neutrino fraction
    fc    = 1 - fb - fnu # cold dark-matter fraction
    fcb   = fc + fb
    fnb   = fnu + fb

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

    # if z is given, find the growth factor. if Dz is given, use that.
    if z is not ... :
        zp1  = z + 1
        g2z  = Om0 * zp1**3 + (1 - Om0 - Ode0) * zp1**2 + Ode0 # eqn. 9
        Omz  = Om0 * zp1**3 / g2z # eqn. 10
        Odez = Ode0 / g2z
        Dz   = (2.5 * Omz) / zp1 / (Omz**(4/7) - Odez + (1 + Omz / 2.) * (1 + Odez / 70.))
    elif Dz is ... :
        raise ValueError("Dz must be given if z is not specified")
    Dz = Dz * zp1_eq # re-normalize the growth factor 

    # growth factor in presence of free-streaming : eqn. 11
    pc  = 0.25 * (5 - np.sqrt(1 + 24 * fc ))
    pcb = 0.25 * (5 - np.sqrt(1 + 24 * fcb)) 
    yfs = 17.2 * fnu * (1 + 0.488 / fnu**(7/6)) * (Nnu * q / fnu)**2 # eqn. 14

    _Dy   = Dz / (1 + yfs)
    Dcb   = (1 + _Dy**0.7)**(pcb / 0.7) * Dz**(1 - pcb)                # eqn. 12
    Dcbnu = (fcb**(0.7 / pcb) + _Dy**0.7)**(pcb / 0.7) * Dz**(1 - pcb) # eqn. 13


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

    Tkcb = Tk_master * Dcb / Dz  # cdm + baryon : eqn. 6
    if 'nu' in out:
        Tkcbnu = Tk_master * Dcbnu / Dz # cdm + baryon + neutrino : eqn. 7
        if out == 'cbnu':
            return Tkcbnu
        elif out == 'nu':
            return (Tkcbnu - fcb * Tkcb) / fnu # neutrino : eqn. 26
    elif out == 'cb':
        return Tkcb
    raise ValueError(f"invalid output combination `{out}`")
    
def modelEisenstein98(k: float, *, h: float, Om0: float, Ob0: float, Tcmb0: float) -> float:
    r"""
    Transfer function by Eisenstein and Hu (1998), with baryon oscillations. All 
    arguments except the first are keyword arguments.

    Parameters
    ----------
    k: array_like
        Wavenumber in units of h/Mpc.
    h: float
        Present Hubble parameter in 100 km/sec/Mpc.
    Om0: float
        Present matter density.
    Ob0: float
        Present baryon density.
    Tcmb0: float
        Present temperature of cosmic microwave background in kelvin.
    **kwargs: ignored

    Returns
    -------
    tk: array_like
        Transfer function values. Has the same shape as `k`.

    Examples
    ---------
    >>> k = np.array([1.e-1, 1.e+0, 1.e+1])
    >>> modelEisenstein98(k, h = 0.7, Om0 = 0.3, Ob0 = 0.05, Tcmb0 = 2.725)
    array([1.28680465e-01, 4.59061096e-03, 8.78252444e-05])

    """
    k           = np.asarray(k) * h # convert wavenumber from h/Mpc to 1/Mpc
    h2          = h * h
    Omh2, Obh2  = Om0 * h2, Ob0 * h2
    theta       = Tcmb0 / 2.7 # cmb temperature in units of 2.7 K
    fb          = Ob0 / Om0 # fraction of baryons
    fc          = 1 - fb  # fraction of cold dark matter
    
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


# a dictionary of all available models:
available = {
                'sugiyama95'      : modelSugiyama95,
                'eisenstein98'    : modelEisenstein98,
                'eisenstein98_zb' : modelEisenstein98_zeroBaryon,
                'eisenstein98_mdm': modelEisenstein98_mixedDarkmatter,
            }