!! useful constants
module constants
    use iso_c_binding
    implicit none
    
    real(c_double), parameter :: M_E  = 2.718281828459045
    real(c_double), parameter :: M_PI = 3.141592653589793
    
end module constants

!! transfer function/power spectrum models
module power
    use constants
    use iso_c_binding
    implicit none
contains
    !! transfer function by bardeen et al, with sugiyama correction.
    !! for scalar k arguments
    subroutine modelSuiyama95(x, h, Om0, Ob0) 
        implicit none
        
        
        real(c_double), intent(inout) :: x  ! wavenumber, k - overwritten as T(k)
        real(c_double), intent(in)    :: h, Om0, Ob0  ! cosmology parameters       

        if (x < 1e-8) then
            x = 1.0 ! for smaller k, transfer function is 1
        else
            x = x / Om0 / h * exp(Ob0 + sqrt(2.0*h) * Ob0 / Om0)

            !! transfer function
            x = log(1 + 2.34*x) / (2.34*x) * (1 + 3.89*x + (16.1*x)**2 + (5.46*x)**3 + (6.71*x)**4)**(-0.25)
        end if

    end subroutine modelSuiyama95

    !! transfer function by eisentein & hu, without baryon oscillations
    subroutine modelEisenstein98_zb(x, h, Om0, Ob0, Tcmb0)
        implicit none
        
        real(c_double), intent(inout) :: x ! wavenumber, k - overwritten as T(k)
        real(c_double), intent(in)    :: h, Om0, Ob0, Tcmb0 ! cosmology parameters        
        real(c_double) :: theta      ! Tcmb in units of 2.7 K
        real(c_double) :: Omh2, Obh2 ! density parameters
        real(c_double) :: fb         ! fraction of baryon
        real(c_double) :: s, geff, L

        theta = Tcmb0 / 2.7
        Omh2  = Om0 * h**2
        Obh2  = Ob0 * h**2
        fb    = Ob0 / Om0

        ! sound horizon, s (eqn. 26)
        s = 44.5*log(9.83 / Omh2) / sqrt(1 + 10*Obh2**0.75)
        
        geff = 1 - 0.328*log(431*Omh2)*fb + 0.38*log(22.3*Omh2)*fb**2 ! alpha_gamma (eqn. 31)
        geff = Om0*h*(geff + (1-geff) / (1 + (0.43*x*s)**4))          ! gamma_eff (eqn. 30)

        x = x * (theta*theta / geff) ! q (eqn. 28)

        !! transfer function
        L = log(2*M_E + 1.8*x) 
        x = L / (L + (14.2 + 731.0 / (1 + 62.5*x))*x**2)

    end subroutine modelEisenstein98_zb


    !! transfer function by eisentein & hu, with baryon oscillations
    !! to do.


    !! transfer function by eisentein & hu, with mixed dark-matter
    !! to do.

end module power

module GEVLogDistribution
    use iso_c_binding
    implicit none

    private
        !! integration settings
        real(c_double) :: integ_kmin = 1.e-8 ! lower limit
        real(c_double) :: integ_kmax = 1.e+8 ! upper limit
        integer(c_int) :: integ_n    = 1001  ! order of integration (simpson rule)

        !! cosmology models
        real(c_double) :: Om0, Ob0, Ode0 ! dark-matter, baryon and dark-energy density
        real(c_double) :: h              ! hubble parameter
        real(c_double) :: ns             ! power spectrum slope 
        real(c_double) :: Tcmb0  = 2.275 ! cmb temperature
        real(c_double) :: sigma8 = 0.8d0 ! rms variance at 8 Mpc/h scale 
        real(c_double) :: pknorm = 1.d0  ! power spectrum normalization factor
        integer(c_int) :: pid    = 1     ! power spectrum model

    !! public functions
    public :: setCosmology, printCosmology                  ! cosmology model setup
    public :: printIntegrationSettings, configIntegration   ! integration setup
    public :: Dz, Om, Ode                                   ! cosmology functions
    public :: transferFunction                              ! transfer function
    public :: unn0_matterPowerSpectrum, matterPowerSpectrum ! power spectrum


contains
    !! configure integration settings
    subroutine configIntegration(user_ka, user_kb, user_n)
        implicit none
        real(c_double), intent(in), value, optional :: user_ka, user_kb ! limits
        integer(c_int), intent(in), value, optional :: user_n ! function eveluations

        if ( present(user_ka) ) then
            integ_kmin = user_ka
        end if

        if ( present(user_kb) ) then
            integ_kmax = user_kb
        end if

        if ( present(user_n) ) then
            ! n must be an odd integer, greater than 2
            if ( (mod(user_n, 2) == 0) .or. (user_n < 3)) then
                print *, "n must be an odd integer, greater than 2"
                call exit(1)
            end if
            integ_n = user_n
        end if

    end subroutine configIntegration

    !! print current integration setup
    subroutine printIntegrationSettings()
        implicit none
        print *, "krange = (", integ_kmin, integ_kmax, ")"
        print *, "order  = ", integ_n    
    end subroutine printIntegrationSettings

    !! set current cosmology model
    subroutine setCosmology(user_Om0, user_Ob0, user_h, user_ns, user_Tcmb0, user_psmodel)
        implicit none
        real(c_double), intent(in), value :: user_Om0, user_Ob0, user_h, user_ns, user_Tcmb0
        integer(c_int), intent(in), value :: user_psmodel

        !! check values
        if ( (user_Om0 < 0.d0) .or. (user_Om0 > 1.d0) ) then
            print *, "parameter Om0 Om0 must be in range [0, 1]"
            goto 1
        else if ( (user_Ob0 < 0.d0) .or. (user_Ob0 > user_Om0) ) then
            print *, "parameter Ob0 must be in range [0, Om0]"
            go to 1
        else if ( user_h < 0.d0 ) then
            print *, "parameters h must be positive"
            goto 1
        else if ( user_Tcmb0 < 0.d0 ) then
            print *, "parameters Tcmb0 must be positive"
            goto 1
        else if ( (user_psmodel < 0) .or. (user_psmodel > 1) ) then
            print *, "invalid power spectrum model, ", user_psmodel
            goto 1
        end if

        !! initialise cosmology
        Om0   = user_Om0     ! matter density
        Ob0   = user_Ob0     ! baryon density
        Ode0  = 1.d0 - Om0   ! dark-energy density (flat space)
        h     = user_h       ! hubble parameter in 100 Mpc/h
        Tcmb0 = user_Tcmb0   ! cmb temperature
        ns    = user_ns      ! power spectrum slope
        pid   = user_psmodel ! power spectrum model

        goto 2 ! successfully set cosmology

1       print *, "failed to set cosmology model"
        call exit(1)

2   end subroutine setCosmology

    subroutine printCosmology()
        implicit none
        print *, "cosmology( Om0 = ", Om0, ", Ob0 = ", Ob0, ", Ode0 = ", Ode0, &
                 ", h = ", h, ", ns   = ", ns, ", Tcmb0 = ", Tcmb0, ")"
    end subroutine printCosmology


    !! TODO: E(z) function

    !! dark-matter density evolution in a flat cosmology
    real(c_double) function Om(z) result(retval)
        implicit none
        real(c_double),intent(in) :: z

        !! eqn: Om(z) = Om0 * (z+1)^3 / (Ode0 + Om0 * (z+1)^3)
        
        retval = z + 1.d0
        retval = Om0 * retval**3
        retval = retval / (retval + Ode0) ! matter density at redshift z

    end function Om

    !! dark-energy density evolution
    real(c_double) function Ode(z) result(retval)
        implicit none
        real(c_double), intent(in) :: z

        !! eqn: Ode(z) = Ode0 / (Ode0 + Om0 * (z+1)^3)

        retval = Ode0 / (Ode0 + Om0 * (z + 1.d0)**3) ! dark-energy density at redshift z

    end function Ode

    !! linear growth factor (approx. by carroll et al 1992)
    real(c_double) function Dz(z) result(retval)
        implicit none
        real(c_double),intent(in) :: z
        real(c_double)            :: Omz, Odez, gz

        retval = z + 1.d0
        Omz    = Om0 * retval**3 
        gz     = Ode0 + Omz       ! g(z)   = Om0 * (z+1)^3 + Ode0 = E^2(z)
        Omz    = Omz / gz         ! matter density evolution, Om(z)  = Om0 * (z+1)^3 / g(z)
        Odez   = Ode0 / gz        ! dark-energy density evolution, Ode(z) = Ode0 / g(z)

        !! growth factor
        retval = 2.5 * Omz / retval / (Omz**(4.0/7.0) - Odez + (1 + Omz / 2.0) * (1 + Odez / 70.0))
        
    end function Dz

    !! transfer function model
    subroutine transferFunction(x)
        use power
        implicit none
        real(c_double), intent(inout) :: x ! wavenumber k - overwritten as T(k)

        !! get the trancfer function
        select case( pid )
            case(0)
                call modelSuiyama95(x, h, Om0, Ob0)    ! bbks with sugiyama correction
            case(1)
                call modelEisenstein98_zb(x, h, Om0, Ob0, Tcmb0) ! eisenstein & hu without baryon oscillations 
            case default
                stop
        end select

    end subroutine transferFunction

    !! un-normalised present matter power spectrum
    subroutine unn0_matterPowerSpectrum(x)
        implicit none

        real(c_double), intent(inout) :: x ! wavenumber, k - overwritten as P(k)
        real(c_double)                :: tk ! transfer function value


        ! compute the transfer function
        tk = x
        call transferFunction(tk)

        x = tk**2 * x**ns ! un-normalised power spectrum at z = 0
    
    end subroutine unn0_matterPowerSpectrum

    !! spherical top-hat filter
    real(c_double) function filt(x) result(retval)
        implicit none
        real(c_double), intent(in) :: x 

        retval = 3 * (sin(x) - x * cos(x)) / x**3

    end function

    !! unnormalised present linear matter variance (rms)
    subroutine unn0_variance(x)
        implicit none
        real(c_double), intent(inout) :: x ! smoothing radius, r - overwritten to var(r)

        ! create the 
    
    end subroutine unn0_variance

    !! normalised matter power spectrum
    subroutine matterPowerSpectrum(x, z)
        implicit none

        real(c_double), intent(inout) :: x ! wavenumber, k - overwritten as P(k)
        real(c_double), intent(in)    :: z ! redshift
        
        call unn0_matterPowerSpectrum(x) ! get the un-normalised power

        x = pknorm * x * (Dz(z) / Dz(0.d0))**2
    
    end subroutine matterPowerSpectrum

end module GEVLogDistribution

program main
    use GEVLogDistribution
    use iso_c_binding
    implicit none



    call setCosmology(0.3d0, 0.05d0, 0.7d0, 1.d0, 2.275d0, 1)

    call printCosmology()
    
    
end program main

! gfortran -shared cosmology.f90 -o c.so