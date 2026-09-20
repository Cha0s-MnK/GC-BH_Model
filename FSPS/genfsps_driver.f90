program genfsps_driver
  use sps_vars
  use sps_utils
  implicit none

  integer :: n_age, imf_code
  integer :: stat, i, z, low
  real(sp) :: target_wavelength
  real(sp) :: mass_csp, lbol_csp, mdust_csp, lnu
  real(sp) :: fraction, low_wavelength, high_wavelength
  real(sp), dimension(10000) :: age_grid
  real(sp), dimension(nspec, ntfull) :: spec_ssp
  real(sp), dimension(ntfull) :: mass_ssp, lbol_ssp
  real(sp), dimension(nemline, ntfull) :: emlin_ssp
  real(sp), dimension(nspec) :: spec_csp
  real(sp), dimension(nemline) :: emlin_csp
  type(params) :: pset

  namelist /config/ n_age, target_wavelength, imf_code

  read(*, nml=config, iostat=stat)
  if (stat /= 0) then
     write(*, '(A,I0)') "DRIVER ERROR: invalid configuration, iostat=", stat
     stop 2
  endif
  if (n_age < 1 .or. n_age > 10000) then
     write(*, '(A,I0)') "DRIVER ERROR: invalid n_age=", n_age
     stop 2
  endif
  do i = 1, n_age
     read(*, *, iostat=stat) age_grid(i)
     if (stat /= 0) then
        write(*, '(A,I0)') "DRIVER ERROR: invalid age at index=", i
        stop 2
     endif
  enddo

  ! This driver always produces a pure-stellar spectrum.
  imf_type = imf_code
  dust_type = 0
  add_dust_emission = 0
  add_agn_dust = 0
  add_agb_dust_model = 0
  add_igm_absorption = 0
  add_xrb_emission = 0
  add_neb_emission = 0
  add_neb_continuum = 0
  cloudy_dust = 0
  nebemlineinspec = 0
  vactoair_flag = 0
  redshift_colors = 0
  compute_light_ages = 0
  smooth_lsf = 0

  call sps_setup(-1)

  low = 1
  do while (low < nspec .and. spec_lambda(low + 1) < target_wavelength)
     low = low + 1
  enddo
  if (target_wavelength < spec_lambda(1) .or. low >= nspec) then
     write(*, '(A,1X,ES16.8)') "DRIVER ERROR: wavelength outside grid=", &
          target_wavelength
     stop 3
  endif
  low_wavelength = spec_lambda(low)
  high_wavelength = spec_lambda(low + 1)
  fraction = (target_wavelength - low_wavelength) / &
       (high_wavelength - low_wavelength)
  write(*, '(A,1X,I0,1X,I0,1X,ES24.16E3,1X,ES24.16E3)') &
       "GRID", nz, nspec, low_wavelength, high_wavelength
  do z = 1, nz
     write(*, '(A,1X,I0,1X,ES24.16E3)') "NATIVE", z, &
          log10(zlegend(z) / zsol)
  enddo

  pset%sfh = 0
  pset%afeindx = 1
  emlin_ssp = 0.0

  do z = 1, nz
     pset%zmet = z
     call ssp_gen(pset, mass_ssp, lbol_ssp, spec_ssp)
     do i = 1, n_age
        call csp_gen(mass_ssp, lbol_ssp, spec_ssp, pset, age_grid(i), 1, &
             mass_csp, lbol_csp, spec_csp, mdust_csp, emlin_ssp, emlin_csp)
        lnu = (1.0 - fraction) * spec_csp(low) + &
             fraction * spec_csp(low + 1)
        if (lnu <= 0.0 .or. lnu /= lnu) then
           write(*, '(A,1X,I0,1X,I0)') &
                "DRIVER ERROR: non-positive spectrum at z/age=", z, i
           stop 4
        endif
        write(*, '(A,1X,I0,1X,I0,1X,ES24.16E3,1X,ES24.16E3)') &
             "VALUE", z, i, age_grid(i), lnu
     enddo
  enddo
end program genfsps_driver
