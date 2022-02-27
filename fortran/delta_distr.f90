! cosmology module contains the cosmology data type, storing 
! parameters for a a lambda-cdm cosmology and related methods
! like the power spectrum and density.
module cosmology_m
    implicit none
    private
    public cosmology_t

    type cosmology_t
        real :: Om0                   ! matter density
        real :: Ob0                   ! baryon density
        real :: Ode0 = 0.0            ! dark-energy density
        real :: Ok0  = 0.0            ! curvature density
        real :: h                     ! hubble parameter
        real :: ns                    ! power spectrum index
        real :: sigma8                ! rms variance of fluctuation
        real :: Tcmb = 2.75           ! cmb temperature
        logical :: flat     = .true.  ! is the space flat
        logical :: checked_ = .false. ! object is checked good
    contains
        procedure :: init   ! check for correct values
        procedure :: Ez     ! calculate E(z)
        procedure :: Om     ! calculate matter density at z 
        procedure :: Ode    ! calculate dark-energy density at z
        procedure :: Dplus  ! linear growth factor
        procedure :: fz     ! linear growth rate
    end type
    
contains
    ! check if the values in the object are correct.
    subroutine init(self)
        class(cosmology_t), intent(inout) :: self 
        
        logical :: success = .true. ! check is success
        
        ! Om0 must be positive
        if (self%Om0 < 0.0) then
            print *, "Om0 cannot be negative."
            success = .false.
        end if 
        
        ! Ob0 must be positive and less than or equal to Om0
        if (self%Ob0 < 0.0) then
            print *, "Ob0 cannot be negative."
            success = .false.
        else if (self%Ob0 > self%Om0) then
            print *, "Ob0 must be less than Om0."
            success = .false.
        end if

        ! h must be positive
        if (self%h < 0.0) then
            print *, "h cannot be negative."
            success = .false.
        end if

        ! if flat cosmology, then calculate the value of Ode0
        if (self%flat) then
            self%Ode0 = 1.0 - self%Om0
            self%Ok0  = 0.0
        else
            self%Ok0 = 1.0 - self%Om0 - self%Ode0
        end if

        ! Ode0 must be positive
        if (self%Ode0 < 0.0) then
            print *, "Ode0 cannot be negative."
            success = .false.
        end if

        ! Tcmb must be positive
        if (self%Tcmb < 0.0) then
            print *, "Tcmb cannot be negative."
            success = .false.
        end if

        if (success) then
            self%checked_ = .true.
        else
            print *, "Invalid values of parameters."
            call exit(1)
        end if
        
    end subroutine

    ! calculate E(z) function at redshift z
    function Ez(self, z) result(retval)
        implicit none 
        class(cosmology_t), intent(in) :: self   
        real, intent(in)               :: z      
        real                           :: retval, zp1

        zp1    = z + 1
        retval = self%Om0 * zp1**3 + self%Ode0
        if (.not. self%flat) then
            retval = retval + self%Ok0 * zp1**2
        end if

        retval = sqrt(retval)
    end function Ez

    ! calculate the matter density at redshift z
    function Om(self, z) result(retval)
        implicit none
        class(cosmology_t), intent(in) :: self
        real, intent(in)               :: z
        real                           :: retval, zp1

        zp1    = z + 1
        retval = self%Ode0
        if (.not. self%flat) then
            retval = retval + self%Ok0 * zp1**2
        end if
        zp1    = self%Om0 * zp1**3
        retval = zp1 / (zp1 + retval)
    end function

    ! calculate the dark-energy density at redshift z
    function Ode(self, z) result(retval)
        implicit none
        class(cosmology_t), intent(in) :: self
        real, intent(in)               :: z
        real                           :: retval, zp1

        zp1    = z + 1
        retval = self%Ode0 + self%Om0 * zp1**3
        if (.not. self%flat) then
            retval = retval + self%Ok0 * zp1**2
        end if
        retval = self%Ode0 / retval
    end function

    ! calculate un-normalised growth factor using the approximate
    ! form by carroll et al (1992)
    function Dplus_u(self, z) result(retval)
        implicit none
        class(cosmology_t), intent(in) :: self
        real, intent(in)               :: z
        real                           :: retval
        real                           :: Omz, Odez, zp1

        zp1    = z + 1
        Omz    = self%Om0 * zp1**3
        Odez   = self%Ode0

        ! put the value of E(z)^2 on the variable retval
        retval = Omz + Odez
        if (.not. self%flat) then
            retval = retval + self%Ok0 * zp1**2
        end if

        ! actual values of Omz and Odez
        Omz    = Omz  / retval 
        Odez   = Odez / retval

        ! approximate growth factor
        retval = Omz**(4./7.) - Odez + (1 + Omz / 2.) * (1 + Odez / 70.)
        retval = 2.5 * Omz / zp1 / retval
    end function Dplus_u

    ! calculate the growth factor using the approximate form by
    ! carroll et al (1992), normalised so that tha present value
    ! is 1
    function Dplus(self, z) result(retval)
        implicit none
        class(cosmology_t), intent(in) :: self
        real, intent(in)               :: z
        real                           :: retval

        retval = Dplus_u(self, z) / Dplus_u(self, 0.0) 
    end function Dplus

    ! calculate the linear growth rate, f(z) = Om(z)^0.6 approximately
    function fz(self, z) result(retval)
        implicit none
        class(cosmology_t), intent(in) :: self
        real, intent(in)               :: z
        real                           :: retval

        retval = self%Om(z)**0.6
    end function
    
end module cosmology_m 


program main
    use cosmology_m
    implicit none

    type(cosmology_t) :: lcdm

    lcdm = cosmology_t(Om0 = 0.3, Ob0 = 0.045, h = 0.7, ns = 1.0, sigma8 = 0.8)
    call lcdm%init()

    print *, "Om(0) = ", lcdm%Om(0.0)
    
end program main