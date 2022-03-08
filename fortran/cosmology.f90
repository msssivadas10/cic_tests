!! useful constants
module constants
    implicit none
    
    real(kind = 8), parameter :: M_E  = 2.718281828459045
    real(kind = 8), parameter :: M_PI = 3.141592653589793
    
end module constants

!! transfer function/power spectrum models
module power
    use constants
    implicit none
    public

    !! transfer function by bardeen et al, with sugiyama correction.
    interface modelSugiyama95
        module procedure modelSuiyama95_scalark, modelSugiyama95_arrayk
    end interface modelSugiyama95

    !! transfer function by eisenstien & hu, without baryon oscillations
    interface modelEisenstein98_zeroBaryon
        module procedure modelEisenstein98_zb_scalark, modelEisenstein98_zb_arrayk
    end interface modelEisenstein98_zeroBaryon
contains
    !! transfer function by bardeen et al, with sugiyama correction.
    !! for scalar k arguments
    subroutine modelSuiyama95_scalark(k, h, Om0, Ob0, Tcmb0) 
        implicit none
        
        ! wavenumber - overwritten as tk
        real(kind = 8), intent(inout) :: k  
        
        ! cosmology parameters         
        real(kind = 8), intent(in)           :: h, Om0, Ob0  
        real(kind = 8), intent(in), optional :: Tcmb0 

        if (k < 1e-8) then
            k = 1.0 ! for smaller k, transfer function is 1
        else
            k = k / Om0 / h * exp(Ob0 + sqrt(2.0*h) * Ob0 / Om0)

            !! transfer function
            k = log(1 + 2.34*k) / (2.34*k) * (1 + 3.89*k + (16.1*k)**2 + (5.46*k)**3 + (6.71*k)**4)**(-0.25)
        end if

    end subroutine modelSuiyama95_scalark

    !! for array k arguments
    subroutine modelSugiyama95_arrayk(k, h, Om0, Ob0, Tcmb0)
        implicit none

        ! wavenumber - overwritten as tk
        real(kind = 8), intent(inout) :: k(:)        

        ! cosmology parameters
        real(kind = 8), intent(in)           :: h, Om0, Ob0 
        real(kind = 8), intent(in), optional :: Tcmb0 

        where (k < 1e-8)
            k = 1.0 ! for smaller k, transfer function is 1
        elsewhere
            k = k / Om0 / h * exp(Ob0 + sqrt(2.0*h) * Ob0 / Om0)

            !! transfer function
            k = log(1 + 2.34*k) / (2.34*k) * (1 + 3.89*k + (16.1*k)**2 + (5.46*k)**3 + (6.71*k)**4)**(-0.25)
        end where
    
    end subroutine modelSugiyama95_arrayk

    !! ====================================================================

    !! transfer function by eisentein & hu, without baryon oscillations
    !! for scalar k arguments
    subroutine modelEisenstein98_zb_scalark(k, h, Om0, Ob0, Tcmb0)
        implicit none
        
        ! wavenumber - overwritten as tk
        real(kind = 8), intent(inout) :: k  
        
        ! cosmology parameters         
        real(kind = 8), intent(in)           :: h, Om0, Ob0  
        real(kind = 8), intent(in), optional :: Tcmb0 

        real(kind = 8) :: theta      ! Tcmb in units of 2.7 K
        real(kind = 8) :: Omh2, Obh2 ! density parameters
        real(kind = 8) :: fb         ! fraction of baryon
        real(kind = 8) :: s, geff, L

        if (present(Tcmb0)) then
            theta = Tcmb0
        else
            theta = 2.725
        end if
        theta = theta / 2.7
        Omh2  = Om0 * h**2
        Obh2  = Ob0 * h**2
        fb    = Ob0 / Om0

        ! sound horizon, s (eqn. 26)
        s = 44.5*log(9.83 / Omh2) / sqrt(1 + 10*Obh2**0.75)
        
        geff = 1 - 0.328*log(431*Omh2)*fb + 0.38*log(22.3*Omh2)*fb**2 ! alpha_gamma (eqn. 31)
        geff = Om0*h*(geff + (1-geff) / (1 + (0.43*k*s)**4))          ! gamma_eff (eqn. 30)

        k = k * (theta*theta / geff) ! q (eqn. 28)

        !! transfer function
        L = log(2*M_E + 1.8*k) 
        k = L / (L + (14.2 + 731.0 / (1 + 62.5*k))*k**2)

    end subroutine modelEisenstein98_zb_scalark

    !! for array k arguments
    subroutine modelEisenstein98_zb_arrayk(k, h, Om0, Ob0, Tcmb0)
        implicit none
        
        ! wavenumber - overwritten as tk
        real(kind = 8), intent(inout) :: k(:)  
        
        ! cosmology parameters         
        real(kind = 8), intent(in)           :: h, Om0, Ob0  
        real(kind = 8), intent(in), optional :: Tcmb0 

        real(kind = 8) :: theta      ! Tcmb in units of 2.7 K
        real(kind = 8) :: Omh2, Obh2 ! density parameters
        real(kind = 8) :: fb         ! fraction of baryon

        real(kind = 8), dimension(size(k)) :: s, geff, L

        if (present(Tcmb0)) then
            theta = Tcmb0
        else
            theta = 2.725
        end if
        theta = theta / 2.7
        Omh2  = Om0 * h**2
        Obh2  = Ob0 * h**2
        fb    = Ob0 / Om0

        ! sound horizon, s (eqn. 26)
        s = 44.5*log(9.83 / Omh2) / sqrt(1 + 10*Obh2**0.75)
        
        geff = 1 - 0.328*log(431*Omh2)*fb + 0.38*log(22.3*Omh2)*fb**2 ! alpha_gamma (eqn. 31)
        geff = Om0*h*(geff + (1-geff) / (1 + (0.43*k*s)**4)) ! gamma_eff (eqn. 30)

        k = k * (theta*theta / geff) ! q (eqn. 28)

        !! transfer function
        L = log(2*M_E + 1.8*k) 
        k = L / (L + (14.2 + 731.0 / (1 + 62.5*k))*k**2)

    end subroutine modelEisenstein98_zb_arrayk

    !! ====================================================================

    !! transfer function by eisentein & hu, with baryon oscillations
    !! to do.

    !! ====================================================================

    !! transfer function by eisentein & hu, with mixed dark-matter
    !! to do.

end module power

module cosmology
    use power
    implicit none
    public

    !! power spectrum model indicators:
    integer, parameter :: SUGIYAMA95       = 0
    integer, parameter :: EISENSTEIN95     = 1
    integer, parameter :: EISENSTEIN95_ZB  = 2
    integer, parameter :: EISENSTEIN95_MDM = 3

    !! global (flat) cosmology model parameters:
    real(kind = 8) :: G_Om0, G_Ob0, G_Ode0 ! density parametrs
    real(kind = 8) :: G_Tcmb0              ! cmb temperature
    real(kind = 8) :: G_h                  ! hubble parameter
    real(kind = 8) :: G_ns                 ! power spectrum slope
    real(kind = 8) :: G_sigma8             ! rms variance of density fluctuations at 8 Mpc/h
    real(kind = 8) :: G_Onu0, G_Nnu, G_Mnu ! neutrino parameters
    integer        :: G_psmodel            ! power spectrum model

    ! others
    real(kind = 8) :: G_PKNORM = 1.0       ! power spectrum normalisation
    logical        :: READY    = .false.   ! model is ready to use

    !! (unnormalised) growth factor
    interface Dz
        module procedure Dz_scalarz, Dz_arrayz
    end interface Dz

    !! matter density evolution
    interface Om
        module procedure Om_scalarz, Om_arrayz
    end interface Om

contains
    !! initialise a flat lambda-cdm cosmology model
    subroutine initCosmology(h, Om0, Ob0, ns, Tcmb0, Nnu, Mnu, psmodel)
        implicit none
        
        ! cosmology model parameters
        real(kind = 8), intent(in) :: Om0, Ob0, h, ns
        
        ! optional arguments
        real(kind = 8), intent(in), optional :: Tcmb0, Nnu, Mnu
        integer, intent(in), optional        :: psmodel

        ! initialise model:
        if (h <= 0) then
            print *, "h must be positive"
            call exit(1)
        else if (Om0 < 0) then
            print *, "matter density Om0 must be positive"
            call exit(1)
        else if ((Ob0 < 0) .or. (Om0 < Ob0)) then
            print *, "baryon density Ob0 must be in the range [0, Om0]"
            call exit(1)
        end if

        G_Om0 = Om0  ! matter density
        G_Ob0 = Ob0  ! baryon density
        G_h   = h    ! hubble parameter
        G_ns  = ns   ! power spectrum index

        ! calculate the dark-energy density:
        G_Ode0 = 1.0 - Om0 ! assuming flat cosmology
        if (G_Ode0 < 0) then
            print *, "dark-energy density Ode0 is negative, adjust matter density"
            call exit(1)
        end if

        ! initialise optional parameters:
        if ( present(Tcmb0) ) then
            if (Tcmb0 <= 0) then
                print *, "Tcmb0 must be positive"
                call exit(1)
            end if
            G_Tcmb0 = Tcmb0 ! cmb temperature
        else
            G_Tcmb0 = 2.275 ! default 
        end if

        if ( present(Nnu) ) then
            if (Nnu < 0) then
                print *, "Nnu must be positive"
                call exit(1)
            end if
            
            ! now, neutrino mass is required
            if ( .not. present(Mnu) ) then
                print *, "Mnu is required if heavy neutrino is present"
                call exit(1)
            else if (Mnu < 0) then
                print *, "neutrino mass must be positive"
                call exit(1)
            end if

            G_Nnu = Nnu ! number of heavy neutrinos
            G_Mnu = Mnu ! total mass of heavy neutrinos

        else
            G_Nnu = 0.0   ! default, no haevy neutrino
            G_Mnu = 0.0
        end if

        ! power spectrum model
        if ( present(psmodel) ) then
            if ((psmodel < 0) .or. (psmodel > 3)) then
                print *, "invalid power spectrum model"
                call exit(1)
            else if ((G_Nnu == 0.0) .and. psmodel == 3) then
                print *, "cannot use mixed dark-matter model without neutrinos"
                call exit(1)
            end if
            G_psmodel = psmodel ! power spectrum model
        else
            G_psmodel = 2       ! default (eisenstein & hu without bao)
        end if

        READY    = .true.
        
    end subroutine initCosmology

    !! E(z) function

    !! dark-matter density evolution
    !! for scalar z
    subroutine Om_scalarz(z)
        implicit none
        real(kind = 8),intent(inout) :: z ! redshift
        
        z = z + 1.0
        z = G_Om0 * z**3

        !! matter density 
        z = z / (z + G_Ode0)

    end subroutine Om_scalarz

    !! for array z
    subroutine Om_arrayz(z)
        implicit none
        real(kind = 8),intent(inout) :: z(:) ! redshift
        
        z = z + 1.0
        z = G_Om0 * z**3

        !! matter density 
        z = z / (z + G_Ode0)

    end subroutine Om_arrayz

    !! dark-energy density evolution

    !! ==========================================================

    !! linear growth factor (approx. by carroll et al 1992)
    !! for scalar z
    subroutine Dz_scalarz(z)
        implicit none
        real(kind = 8),intent(inout) :: z ! redshift
        real(kind = 8) :: Om, Ode 

        z   = z + 1.0
        Om  = G_Om0 * z**3 
        Ode = G_Ode0 + Om  ! now, E^2 = Om0 * (z+1)^3 + Ode0
        Om  = Om / Ode     ! Om(z)    = Om0 * (z+1)^3 / E^2(z)
        Ode = G_Ode0 / Ode ! Ode(z)   = Ode0 / E^2(z)

        !! growth factor
        z = 2.5 * Om / z / (Om**(4.0/7.0) - Ode + (1 + Om / 2.0) * (1 + Ode / 70.0))
        
    end subroutine Dz_scalarz

    !! for array z
    subroutine Dz_arrayz(z)
        implicit none
        real(kind = 8),intent(inout) :: z(:) ! redshift
        real(kind = 8), dimension(size(z)) :: Om, Ode 

        z   = z + 1.0
        Om  = G_Om0 * z**3 
        Ode = G_Ode0 + Om  ! now, E^2 = Om0 * (z+1)^3 + Ode0
        Om  = Om / Ode     ! Om(z)    = Om0 * (z+1)^3 / E^2(z)
        Ode = G_Ode0 / Ode ! Ode(z)   = Ode0 / E^2(z)

        !! growth factor
        z = 2.5 * Om / z / (Om**(4.0/7.0) - Ode + (1 + Om / 2.0) * (1 + Ode / 70.0))
        
    end subroutine Dz_arrayz
    
end module cosmology



program main
    use cosmology
    implicit none

    real(kind = 8), dimension(3) :: z, dplus

    call initCosmology(0.7d0, 0.3d0, 0.05d0, 1.0d0)  
    
    z = (/ 0.0d0, 0.1d0, 0.2d0 /)
    dplus = z
    call Om(dplus)

    write(*,*) "growth factor, D(", z, ") = ", dplus 
    
end program main