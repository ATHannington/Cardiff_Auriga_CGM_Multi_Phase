module CONSTANTS
    IMPLICIT NONE
                                                                                !  CONFIGURATION [CF]
    CHARACTER(LEN=20),parameter       :: CFgeom='Cylindrical1D'                 ! geometry of configuration
    REAL(KIND=8)            :: CFrho0=0.1000E-18                      ! central density (g/cm^3)
    REAL(KIND=8),parameter            :: CFw0=0.1500E+18                        ! core radius (cm)
    INTEGER,parameter                 :: CFschP=4                               ! radial density exponent forn Schuster profile
    REAL(KIND=8),parameter            :: CFwB=0.1500E+19                        ! boundary radius (cm)
    INTEGER,parameter                 :: CFcTOT=100                             ! number of (cylindrical) shells
              INTEGER,parameter                    :: CFprof=0                  ! flag to sanction diagnostics for profile
              INTEGER,parameter                    :: CFlist=0                  ! flag to sanction diagnostics for cells

                                                                                ! DUST GRAINS [DG]
    CHARACTER(LEN=20),parameter       :: DGsource='Draine'                      ! source of dust properties (e.g. 'Draine')
    CHARACTER(LEN=20),parameter       :: DGmodel='draine_rv3.1.dat'             ! dust model (e.g. 'draine_rv3.1.dat')
    INTEGER,parameter                 :: DGlMAX=560! ###560 ~= line 640###      ! line number where dust properties end
    INTEGER,parameter                 :: DGlMIN=66!66                           ! line number where dust properties start
    REAL(KIND=8),parameter            :: DGkapV=0.30000E-17                     ! mass opacity, only for pure scattering (cm^2/g)
    REAL(KIND=8),parameter            :: DGkapM=0.20000E+03                     ! volume opacity, only for pure scattering (1/cm)

                                                                                ! WAVELENGTHS [WL]
    REAL(KIND=8),parameter            :: WLdelta=1.0e-1                         ! logarithmic spacing of optical properties
    REAL(KIND=8),parameter            :: WLdcl=0.10                             ! weight of slope-change
              INTEGER,parameter                    :: WLprint=0                 ! flag to trigger printing of dust properties
              INTEGER,parameter                    :: WLplot=0                  ! flag to trigger plotting of dust properties

                                                                                ! TEMPERATURES [TE]
    INTEGER,parameter                 :: TEkTOT=100!100!1000                    ! number of discrete temperatures
    REAL(KIND=8),parameter            :: teTmin=2.725                           ! minimum discrete temperature
    REAL(KIND=8),parameter            :: teTmax=272.5!272.5!2725.0             ! maximum discrete temperature
                INTEGER,parameter                  :: TElist=0                  ! flag to print out some temperatures

                                                                                ! REFERENCE PROBABILITIES [PR]
    INTEGER,parameter                 :: PRnTOT=1000                            ! number of reference probabilities
                INTEGER,parameter                  :: WTpack=1000000            ! number of calls for plotting probabilities
                INTEGER,parameter                  :: WTplot=0                  ! flag to sanction plotting probabilities

                                                                                ! BACKGROUND RADIATION FIELD [BG]
    INTEGER,parameter                 :: BGkBB=29!29!80!865                    ! temperature-ID of background BB radiation field
    REAL(KIND=8),parameter            :: BGfBB=1.00E0                           ! dilution factor of background BB radiation field
    INTEGER,parameter                 :: BGkGO=23!23!70!696                    ! ID of temperature for cfLgo ---- ceiling(dble(BGkBB)*0.8d0)

                                                                                ! LUMINOSITY PACKETS [LP]
    INTEGER,parameter                 :: LPpTOT= int(1E7)                       ! number of luminosity packets

              INTEGER(Kind=4),parameter            :: DBTestFlag= 1             !Diagnostic test flag for tests and print statements



    Real(kind=8)                      :: TAUconst = 1.d0
    ! Real(kind=8)                      :: RHOconst = 1.d-18
    ! Real(kind=8)                      :: MUconst = 1.d+17
    ! Real(kind=8)                      :: SIGconst = 1.d-1
end module CONSTANTS

! The value of pi is 3.14159274
! 2^(-1/2) = 0.70710678
! 2^(+1/2) = 1.4142136
! (6/pi)^(1/3) = 1.2407007
! pc = (0.308568E+19) cm
! cm = (0.324078E-18) pc
! M_Sun = (0.198910E+34) g
! amu = (0.166054E-23) g
! g/cm = (0.155129E-15) M_Sun/pc
! M_Sun/pc = (0.644623E+15) g/cm
! H2/g = (0.210775E+24)
! g/H2 = (0.474444E-23)


module PHYSICAL_CONSTANTS
  IMPLICIT NONE

  Real(kind=8),parameter              :: pi = 3.14159274d0
  Real(kind=8),parameter              :: twopi = 6.28311855d0
  Real(kind=8),parameter              :: lightc = 2.997925d10                   ![cm s**-1]
  Real(kind=8),parameter              :: planckh = 6.626070d-27                 ![erg s]
  Real(kind=8),parameter              :: boltzkb = 1.380649d-16                 ![erg K**-1]
  Real(kind=8),parameter              :: sigmasb = 5.670515d-5                  ![erg s**-1 cm**-2 K**-4]
  Real(kind=8),parameter              :: hckb = 0.143878d5                      ![microns K**-1]
  Real(kind=8),parameter              :: hc2 = 5.9552219872d10                  ![erg s**-1 cm**-2 microns**4]
  Real(kind=8),parameter              :: h2c3kb = 8.568225728d14                ![erg s**-1 cm**-2 K**-1 microns**5 K**2]
  Real(kind=8),parameter              :: cmtopc = 0.324078d-18                  ![pc]
  Real(kind=8),parameter              :: pctocm = 3.085676905d18                ![cm]
  Real(kind=8),parameter              :: msol = 0.198910d+34                    ![g]
  Real(kind=8),parameter              :: amu = 0.166054d-23                     ![g] atomic mass unit
  Real(kind=8),parameter              :: msolpctogcm = 0.644623E+15             ![g cm**-1]
  Real(kind=8),parameter              :: gcmtomsolpc = 0.155129E-15             ![M_sol pc**-1]
  Real(kind=8),parameter              :: h2dens = 0.210775E+24                  ![g cm**-1]
  Real(kind=8),parameter              :: invh2dens = 0.474444E-23               ![cm g**-1]

end module PHYSICAL_CONSTANTS
