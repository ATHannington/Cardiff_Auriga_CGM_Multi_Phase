
! RadTrans_Cylindrical1D.F90
! ==========================

! Last updated: 2018.Dec.05 (1405)

! Desktop: gfortran -O3 -o RUN RadTrans_Cylindrical1D.F90 -L/star/lib `pgplot_link`
! Laptop:  gfortran -O0 -fbounds-check -o RUN RadTrans_Cylindrical1D.F90 -L/opt/local/lib -lpgplot -lX11

! Run with ./RUN

! The value of pi is 3.14159274
! 2^(-1/2) = 0.70710678
! 2^(+1/2) = 1.4142136
! (6/pi)^(1/3) = 1.2407007
! pc = (0.308568E+19) cm
! M_Sun = (0.198910E+34) g
! amu = (0.166054E-23) g

! g/cm = 0.155129E-15 M_Sun/pc;   M_Sun/pc = 0.644623E+16 g/cm

! SubRoutines:
!   RadTrans_Cylinder1D_GlobalParameters
!   RadTrans_DustPropertiesFromDraine
!   RadTrans_PlotDustProperties
!   RadTrans_Temperatures
!   RadTrans_EmissionProbabilities
!   RadTrans_PlotEmissionProbabilities [UD]
!   RadTrans_Cylindrical1D_LinearShellSpacing
!   RadTrans_InjectAndTrack0 (straight through)
!   RadTrans_InjectAndTrack1 (pure scattering, linear opacity)
!   RadTrans_InjectAndTrack2 (pure scattering, Plummer-like opacity)
!   
! Timing:
!   the zero-opacity case, with 10(100) shells, does 10^8(10^7) packets in 25(20) seconds;
!   the pure scattering case . . . . .

!*****************************
PROGRAM RadTrans_Cylindrical1D
!*****************************

IMPLICIT NONE                                            ! DECLARATIONS
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: alb       ! discrete albedos
INTEGER                                     :: CellList  ! flag to print out some cell boundaries
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: chi       ! discrete extinction opacities (in cm^2/g)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: dlam      ! discrete wavelength intervals (in microns)
REAL(KIND=8)                                :: DeltaX    ! spacing for dust optical properties
INTEGER                                     :: Draine    ! flag to sanction Draine optical properties
INTEGER                                     :: DustList  ! flag to print out some dust properties
INTEGER                                     :: DustPlot  ! flag to plot dust optical properties
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: eta       ! outer radius of shell, squared (in cm^2)
INTEGER                                     :: i_TOT     ! number of luminosity packets
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: J0        ! mean (angle-averaged) intensity in shell
INTEGER                                     :: k_TOT     ! number of discrete temperatures
REAL(KIND=8)                                :: kappa_M   ! mass scattering opacity (in cm^2/g)
REAL(KIND=8)                                :: kappa_V   ! volume scattering opacity (in 1/cm)
INTEGER                                     :: l_MAX     ! line number for longest wavelength needed
INTEGER                                     :: l_MIN     ! line number for shortest wavelength needed
INTEGER                                     :: l_TOT     ! number of interpolated wavelengths
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: lam       ! discrete wavelengths (in microns)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: LM        ! luminosity per unit mass (in erg/s.g) 
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: lPl_LOW   ! ID of highest lam w. Pl<(l-1)/l_TOT at T(k) 
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: lPl_UPP   ! ID of lowest lam w. Pl>l/l_TOT at T(k) 
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: lPr_LOW   ! ID of highest lam w. Pr<(l-1)/l_TOT at T(k) 
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: lPr_UPP   ! ID of lowest lam w. Pr>l/l_TOT at T(k) 
REAL(KIND=8)                                :: mu_J0     ! shell-mean J0
REAL(KIND=8)                                :: mu_O      ! line-density (in g/cm)
INTEGER                                     :: n_ADJ     ! flag to regulate shell adjustments
INTEGER                                     :: n_TOT     ! number of shells
INTEGER                                     :: p         ! radial density exponent in envelope
REAL(KIND=8),DIMENSION(:,:),ALLOCATABLE     :: Pl        ! Planck probability at lam(l) at T(k)
REAL(KIND=8),DIMENSION(:,:),ALLOCATABLE     :: Pr        ! probability of re-emitting at lam(l) at T(k)
INTEGER                                     :: ProbPlot  ! flag to plot re-emission probabilities
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: rho       ! mean density in shell (in g/cm^3)
REAL(KIND=8)                                :: rho_O     ! spine density (in g/cm^3)
INTEGER                                     :: Sanction  ! flag to trigger subroutine call
CHARACTER(LEN=100)                          :: Source    ! dummy character string
REAL(KIND=8)                                :: sigma_J0  ! shell-SD of J0
REAL(KIND=8)                                :: Sigma_O   ! column-density through spine (in g/cm^2)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: Store     ! array for re-scoping dust properties
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: T         ! discrete temperatures
REAL(KIND=8)                                :: T_MAX     ! maximum discrete temperature
REAL(KIND=8)                                :: T_MIN     ! minimum discrete temperature
REAL(KIND=8)                                :: tau_O     ! total optical depth through spine
INTEGER                                     :: TempList  ! flag to print out some temperatures
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: w         ! outer radius of shell (in cm)
REAL(KIND=8)                                :: W_O       ! radius of core of spine (in cm)
REAL(KIND=8)                                :: W_B       ! boundary radius (in cm)
REAL(KIND=8)                                :: x         ! x-coordinate (in cm)
REAL(KIND=8)                                :: y         ! y-coordinate (in cm)

                                                         ! GLOBAL INPUT PARAMETERS
                                                         ! CONFIGURATION
rho_O=3.0E+04                                            ! enter spine density (in H2/cm^3)
rho_O=rho_O*(4.77E-24)                                   ! convert spine density to g/cm^3
W_O=0.0200                                               ! enter spine radius (in pc)
W_O=W_O*(3.086E+18)                                      ! convert spine radius to cm
p=2                                                      ! set envelope density exponent
W_B=0.4000    ! pc                                       ! enter boundary radius (in pc)
W_B=W_B*(3.086E+18)                                      ! convert boundary radius to cm
n_TOT=100                                                ! set number of shells
n_ADJ=0                                                  ! set shell adjustment flag
CellList=1                                               ! set flag for diagnostic printout
                                                         ! WAVELENGTHS, DUST
Draine=1                                                 ! set source of dust optical properties
Source='draine_rv3.1.dat'                                ! set model for dust optical properties
l_MIN=66                                                 ! set line number where OptProps start
l_MAX=560                                                ! set line number where OptProps end
DeltaX=0.10                                              ! set spacing of optical properties
DustList=0                                               ! set flag to list some dust optical props.
DustPlot=0                                               ! set flag to plot optical properties
tau_O=003.00                                             ! enter total optical through spine
                                                         ! TEMPERATURES
k_TOT=100                                                ! set number of temperatures required
T_MIN=3.000                                              ! set minimum temperature
T_MAX=100.0                                              ! set maximum temperature
TempList=0                                               ! set flag to list temperatures
                                                         ! PROBABILITIES
ProbPlot=0                                               ! set number of LumPacks to test probs.
                                                         ! LUMINOSITY PACKETS
i_TOT=10000000                                           ! set number of luminosity packets

                                                         ! PRINTOUT
WRITE (6,"(/,3X,'CONFIGURATION PARAMETERS')")            ! header
WRITE (6,"(3X,'Spine density:',24X,E10.3,2X,'H_2 cm^-3',6X,'(rho_O)')") rho_O/(4.77E-24)
WRITE (6,"(3X,'Spine radius:',25X,F10.7,2X,'pc',13X,'(W_O)')") W_O/(3.086E+18)
WRITE (6,"(3X,'Radial density exponent:',22X,I2,17X,'(p)')") p
WRITE (6,"(3X,'Boundary radius:',22X,F10.7,2X,'pc',13X,'(W_B)')") W_B/(3.086E+18)
CALL RadTrans_Cylinder1D_GlobalParameters(rho_O,W_O,p,W_B,tau_O,Sigma_O,mu_O,kappa_V,kappa_M)
WRITE (6,"(3X,'Spine column-density:',17X,E10.3,'  H_2 cm^-2',6X,'(Sigma_O)')") Sigma_O/(4.77E-24)
WRITE (6,"(3X,'Spine optical depth:',18X,F10.5,17X,'(tau_O)')") tau_O
WRITE (6,"(3X,'Line-density:',25X,E10.3,2X,'M_Sun pc^-1',4X,'(mu_O)')") mu_O*(1.552E-15)
WRITE (6,"(3X,'Number of shells:',27X,I4,17X,'(n_TOT)')") n_TOT
WRITE (6,"(3X,'Number of luminosity packets:',9X,I10,17X,'(i_TOT)',/)") i_TOT

                                                         ! ALLOCATIONS 1
ALLOCATE (w(0:n_TOT))                                    ! allocate w array
ALLOCATE (eta(0:n_TOT))                                  ! allocate eta array
ALLOCATE (rho(1:n_TOT))                                  ! allocate rho array
ALLOCATE (J0(1:n_TOT))                                   ! allocate J0 array
ALLOCATE (T(0:k_TOT))                                    ! allocate T array
ALLOCATE (LM(0:k_TOT))                                   ! allocate LM array
ALLOCATE (lam(1:5000))                                   ! temporary ............
ALLOCATE (dlam(1:5000))                                  ! ... storage ..........
ALLOCATE (chi(1:5000))                                   ! ....... for dust .....
ALLOCATE (alb(1:5000))                                   ! ........... properties

                                                         ! IMPORT DUST PROPERTIES FROM DRAINE
IF (Draine==1) THEN                                      ! [IF] sanctioned, [THEN]
  CALL RadTrans_DustPropertiesFromDraine(Source,l_MIN,l_MAX,DeltaX,DustList,l_TOT,lam,dlam,chi,alb)
ENDIF                                                    ! [ENDIF] completed

                                                         ! RESCOPE THE DUST ARRAYS
ALLOCATE (Store(1:l_TOT))                                ! rescope ................
Store(1:l_TOT)=lam(1:l_TOT)                              ! ........................
DEALLOCATE (lam)                                         ! ........................
ALLOCATE (lam(1:l_TOT))                                  ! ........................
lam(1:l_TOT)=Store(1:l_TOT)                              ! ........................
Store(1:l_TOT)=dlam(1:l_TOT)                             ! .... the dust ..........
DEALLOCATE (dlam)                                        ! ........................
ALLOCATE (dlam(1:l_TOT))                                 ! ........................
dlam(1:l_TOT)=Store(1:l_TOT)                             ! ........................
Store(1:l_TOT)=chi(1:l_TOT)                              ! ........................
DEALLOCATE (chi)                                         ! ........................
ALLOCATE (chi(1:l_TOT))                                  ! .......... property ....
chi(1:l_TOT)=Store(1:l_TOT)                              ! ........................
Store(1:l_TOT)=alb(1:l_TOT)                              ! ........................
DEALLOCATE (alb)                                         ! ........................
ALLOCATE (alb(1:l_TOT))                                  ! ........................
alb(1:l_TOT)=Store(1:l_TOT)                              ! ................. arrays

                                                         ! ALLOCATIONS 2
ALLOCATE (lPl_LOW(1:l_TOT,1:k_TOT))                      ! allocate lPl_LOW array 
ALLOCATE (lPl_UPP(1:l_TOT,1:k_TOT))                      ! allocate lPl_UPP array
ALLOCATE (lPr_LOW(1:l_TOT,0:k_TOT))                      ! allocate lPr_LOW array
ALLOCATE (lPr_UPP(1:l_TOT,0:k_TOT))                      ! allocate lPr_UPP array
ALLOCATE (Pl(0:l_TOT,1:k_TOT))                           ! allocate Pl array 
ALLOCATE (Pr(0:l_TOT,0:k_TOT))                           ! allocate Pr array 

                                                         ! PLOT DUST PROPERTIES
IF (DustPlot==1) THEN                                    ! [IF] sanctioned, [THEN]
  CALL RadTrans_PlotDustProperties(l_TOT,lam,chi,alb)    !   plot dust properties and emissivities
ENDIF                                                    ! [ENDIF] completed

CALL RadTrans_Temperatures(k_TOT,T_MIN,T_MAX,TempList,T) ! COMPUTE DISCRETE TEMPERATURES

CALL RadTrans_EmissionProbabilities(k_TOT,T,l_TOT,lam,  &! COMPUTE ............
&dlam,chi,alb,ProbPlot,Pl,lPl_LOW,lPl_UPP,LM,Pr,        &! ... EMISSION .......
&lPr_LOW,lPr_UPP)                                        ! ...... PROBABILITIES

                                                         ! SET UP LINEAR SHELLS
IF (n_ADJ<2) CALL RadTrans_Cylindrical1D_LinearShellSpacing(rho_O,W_O,p,W_B,n_TOT,CellList,w,eta,rho) ! 

                                                         ! TEST CASE 0: ZERO OPACITY
Sanction=0                                               ! Sanction? 
IF (Sanction==1) THEN                                    ! [IF] sanctioned, [THEN]
  CALL RadTrans_InjectAndTrack0(W_B,n_TOT,w,eta,i_TOT)
ENDIF                                                    ! [ENDIF] completed

                                                         ! TEST CASE 1: PURE SCATTERING, UNIFORM OPACITY
Sanction=0                                               ! Sanction? 
IF (Sanction==1) THEN                                    ! [IF] sanctioned, [THEN]
  CALL RadTrans_InjectAndTrack1(W_B,n_TOT,w,eta,kappa_V,i_TOT)
ENDIF                                                    ! [ENDIF] completed

                                                         ! TEST CASE 2: PURE SCATTERING, PLUMMER-LIKE OPACITY
Sanction=1                                               ! Sanction? 
IF (Sanction==1) THEN                                    ! [IF] sanctioned, [THEN]
  CALL RadTrans_InjectAndTrack2(W_B,n_TOT,w,eta,rho,kappa_M,i_TOT)
ENDIF                                                    ! [ENDIF] completed

!*********************************
END PROGRAM RadTrans_Cylindrical1D
!*********************************



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RadTrans_Cylinder1D_GlobalParameters(rho_O,&
         &W_O,p,W_B,tau_O,Sigma_O,mu_O,kappa_V,kappa_M)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine is given:
!   the spine density                                 (rho_O); 
!   the spine radius                                  (W_O); 
!   the radial density exponent                       (p); 
!   the boundary radius                               (W_B);
!   and the total optical depth through the spine     (tau_O). 
! It returns:
!   the column-density through the spine              (Sigma_O); 
!   the line-density                                  (mu_O);
!   the volume scattering opacity                     (kappa_V); 
!   the mass scattering opacity                       (kappa_M). 

IMPLICIT NONE                                            ! DECLARATIONS
REAL(KIND=8)                                :: aux_R     ! auxiliary real
REAL(KIND=8),INTENT(OUT)                    :: kappa_M   ! mass scattering opacity (in cm^2/g)
REAL(KIND=8),INTENT(OUT)                    :: kappa_V   ! volume scattering opacity (in 1/cm)
REAL(KIND=8),INTENT(OUT)                    :: mu_O      ! line-density (in g/cm)
INTEGER,INTENT(IN)                          :: p         ! radial density exponent
REAL(KIND=8),INTENT(IN)                     :: rho_O     ! spine density (in g/cm^3)
REAL(KIND=8),INTENT(OUT)                    :: Sigma_O   ! spine column-density (in g/cm^2)
REAL(KIND=8),INTENT(IN)                     :: tau_O     ! total optical depth through spine
REAL(KIND=8),INTENT(IN)                     :: W_B       ! boundary radius (in cm)
REAL(KIND=8),INTENT(IN)                     :: W_O       ! spine radius (in cm)

IF (p==2) THEN                                           ! start Case: p=2
  aux_R=W_B/W_O                                          !   compute W_B/W_O
  Sigma_O=2.0000*rho_O*W_O*ATAN(aux_R)                   !   compute Sigma_O
  mu_O=1.5708*rho_O*W_O*W_O*LOG(1.+(aux_R*aux_R))        !   compute mu_O
ENDIF                                                    ! end Case: p = 2
kappa_M=tau_O/Sigma_O                                    ! compute mass scattering opacity
kappa_V=tau_O/(2.*W_B)                                   ! compute volume scattering opacity

! There are other cases for a rainy day (see Appendix A) !

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RadTrans_Cylinder1D_GlobalParameters
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RadTrans_DustPropertiesFromDraine(Source,l_MIN,l_MAX,&
         &DeltaX,DustList,l_TOT,lam_OUT,dlam_OUT,chi_OUT,alb_OUT)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine imports dust properties from one of the Draine (2003) tabulations, and 
! interpolates to find regularly spaced values in the range specified by the IDs [l_MIN,l_MAX].
! It is given:
!   the name of the Draine file                          (Source); 
!   the line number for the longest wavelength needed    (l_MIN); 
!   the line number for the shortest wavelength needed   (l_MAX); 
!   the spacing parameter                                (DeltaX);
!   and a flag to print out some optical properties      (DustList). 
! It reads in the data from the Draine file, viz.
!   wavelengths                                          (lam_IN(1:l_LST)); 
!   extinction opacities                                 (chi_IN(1:l_LST)); 
!   albedos                                              (alb_IN(1:l_LST)); 
!   and mean scattering cosines                          (msc_IN(1:l_LST)). 
! It returns the interpolated:
!   number of discrete wavelengths                       (l_TOT); 
!   wavelengths                                          (lam_OUT(1:l_TOT)); 
!   wavelength intervals                                 (dlam_OUT(1:l_TOT)); 
!   extinction opacities                                 (chi_OUT(1:l_TOT)); 
!   and albedos                                          (alb_OUT(1:l_TOT)).  
! The array receiving these interpolated values is then rescoped, to save memory. 

IMPLICIT NONE                                            ! DECLARATIONS
REAL(KIND=8),DIMENSION(1:l_MAX)             :: alb_IN    ! imported albedos
REAL(KIND=8)                                :: alb_NEW   ! new trial albedo
REAL(KIND=8)                                :: alb_OLD   ! old trial albedo
REAL(KIND=8),INTENT(OUT),DIMENSION(1:3000)  :: alb_OUT   ! exported albedos
REAL(KIND=8),DIMENSION(1:l_MAX)             :: chi_IN    ! imported extinction opacities
REAL(KIND=8)                                :: chi_NEW   ! new trial extinction opacity
REAL(KIND=8)                                :: chi_OLD   ! old trial extinction opacity
REAL(KIND=8),INTENT(OUT),DIMENSION(1:3000)  :: chi_OUT   ! exported extinction opacities
REAL(KIND=8)                                :: dDelta    ! spacing increment
REAL(KIND=8)                                :: Delta     ! accumulated spacing
REAL(KIND=8),INTENT(IN)                     :: DeltaX    ! desired spacing
REAL(KIND=8),INTENT(OUT),DIMENSION(1:3000)  :: dlam_OUT  ! exported wavelength intervals
REAL(KIND=8)                                :: dum_R     ! dummy real
INTEGER,INTENT(IN)                          :: DustList  ! flag to print out some dust optical props.
REAL(KIND=8)                                :: f_chi     ! spacing of long opacities
REAL(KIND=8)                                :: f_lam     ! spacing of long wavelengths
REAL(KIND=8)                                :: kappa     ! dummy absorption opacity
INTEGER                                     :: l,ll      ! dummy wavelength IDs
INTEGER                                     :: l_MAX     ! line number for longest wavelength needed
INTEGER                                     :: l_MIN     ! line number for shortest wavelength needed
INTEGER,INTENT(OUT)                         :: l_TOT     ! number of interpolated wavelengths
REAL(KIND=8),DIMENSION(1:l_MAX)             :: lam_IN    ! imported wavelengths
REAL(KIND=8)                                :: lam_NEW   ! new trial lambda
REAL(KIND=8)                                :: lam_OLD   ! old trial lambda
REAL(KIND=8),INTENT(OUT),DIMENSION(1:3000)  :: lam_OUT   ! exported wavelengths
REAL(KIND=8),DIMENSION(1:l_MAX)             :: msc_IN    ! imported mean scattering cosines
REAL(KIND=8)                                :: msc_SQD   ! dummy squared mean scattering cosine
CHARACTER(LEN=100)                          :: rhubarb   ! dummy character string
CHARACTER(LEN=100),INTENT(IN)               :: Source    ! Source of imported data
REAL(KIND=8)                                :: w_LOW     ! weight of lower interpolation point
REAL(KIND=8)                                :: w_UPP     ! weight of upper interpolation point

                                                         ! READ IN DUST PROPERTIES
OPEN (UNIT=5,FILE=Source,STATUS='old',ACTION='read')     ! open data file
DO l=-l_MIN,l_MAX                                        ! start loop over input file
  IF (l<1) THEN                                          !   [IF] in header-text part, [THEN] 
    READ(5,*) rhubarb                                    !     [READ] into dummy character string
  ELSE                                                   !   [ELSE]
    READ (5,"(E11.5,F7.4,F8.4,E10.3,E10.3,F8.5)") lam_IN(l),alb_IN(l),msc_IN(l),chi_IN(l),kappa,msc_SQD
  ENDIF                                                  !   [ENDIF]
ENDDO                                                    ! end loop over input file
IF (DustList==1) THEN
  WRITE (6,"(/,3X,'DUST PROPERTIES FROM DRAINE:')")
  WRITE (6,"(3X,'lam:',3X,5F10.1,6X,5F10.5)") lam_IN(1:5),lam_IN(l_MAX-4:l_MAX)
  WRITE (6,"(3X,'chi:',3X,5E10.3,6X,5E10.3)") chi_IN(1:5),chi_IN(l_MAX-4:l_MAX)
  WRITE (6,"(3X,'alb:',3X,5F10.5,6X,5F10.5)") alb_IN(1:5),alb_IN(l_MAX-4:l_MAX)
  WRITE (6,"(3X,'msc:',3X,5F10.5,6X,5F10.5)") msc_IN(1:5),msc_IN(l_MAX-4:l_MAX)
ENDIF
                                                         ! IMPLEMENT IRVING APPROXIMATION
DO l=1,l_MAX                                             ! start loop over input file
  dum_R=1.-(msc_IN(l)*alb_IN(l))                         !   compute factor
  IF (Source=='draine_rv3.1.dat')                       &!   adjust extinction opacity ... 
  &                chi_IN(l)=dum_R*chi_IN(l)/(1.870E-26) !   ..... and normalise for R=3.1
  IF (Source=='draine_rv4.0.dat')                       &!   adjust extinction opacity ... 
  &                chi_IN(l)=dum_R*chi_IN(l)/(1.969E-26) !   ..... and normalise for R=3.1
  IF (Source=='draine_rv5.5.dat')                       &!   adjust extinction opacity ... 
  &                chi_IN(l)=dum_R*chi_IN(l)/(2.199E-26) !   ..... and normalise for R=3.1
  alb_IN(l)=(1.-msc_IN(l))*alb_IN(l)/dum_R               !   adjust albedo
  msc_IN(l)=0.                                           !   set mean scattering cosine to zero
ENDDO                                                    ! end loop over input file
IF (DustList==1) THEN
  WRITE (6,"(/,3X,'DUST PROPERTIES POST IRVING:')")
  WRITE (6,"(3X,'lam:',3X,5F10.1,6X,5F10.5)") lam_IN(1:5),lam_IN(l_MAX-4:l_MAX)
  WRITE (6,"(3X,'chi:',3X,5E10.3,6X,5E10.3)") chi_IN(1:5),chi_IN(l_MAX-4:l_MAX)
  WRITE (6,"(3X,'alb:',3X,5F10.5,6X,5F10.5)") alb_IN(1:5),alb_IN(l_MAX-4:l_MAX)
  WRITE (6,"(3X,'msc:',3X,5F10.5,6X,5F10.5)") msc_IN(1:5),msc_IN(l_MAX-4:l_MAX)
ENDIF
  
                                                         ! INTERPOLATE
lam_OLD=lam_IN(1)                                        ! set lam_OLD to first imported lam_IN
chi_OLD=chi_IN(1)                                        ! set chi_OLD to first imported chi_IN
alb_OLD=alb_IN(1)                                        ! set alb_OLD to first imported alb_IN
l=1                                                      ! set l=1 (ID of first imported value)
ll=1                                                     ! set ll=1 (ID of first exported value)
DO WHILE (l<=l_MAX)                                      ! iterate until imported wavelengths exhausted
  Delta=0.                                               !   set Delta (accumulator) to zero
  DO WHILE ((Delta<DeltaX).AND.(l<l_MAX))                !   iterate until Delta big enough
    l=l+1                                                !     increment ID of imported point
    lam_NEW=lam_IN(l)                                    !     set lam_NEW to imported value
    dDelta=ABS(LOG10(lam_IN(l)/lam_OLD))                &!     compute .........
         &+ABS(LOG10(chi_IN(l)/chi_OLD))                &!     ... spacing .....
         &+ABS(LOG10((0.01+alb_IN(l))/(0.01+alb_OLD)))   !     ....... increment
    Delta=Delta+dDelta                                   !     increment Delta (accumulator)
  ! WRITE (6,"(3X,'trawl:  ',I6,I5,3X,2F14.5,4X,2F9.4,E13.3,F13.5)")&
  ! &l,ll,lam_OLD,lam_NEW,dDelta,Delta,chi_IN(l),alb_IN(l)
  ENDDO                                                  !   Delta now big enough
  IF (l==l_MAX) EXIT                                     !   terminate interpolation
  w_LOW=(Delta-DeltaX)/dDelta                            !   weight of upper imported wavelength
  w_UPP=1.-w_LOW                                         !   weight of lower imported wavelength
  lam_NEW=w_LOW*lam_IN(l-1)+w_UPP*lam_IN(l)              !   wavelength at upper end of interval
  chi_NEW=w_LOW*chi_IN(l-1)+w_UPP*chi_IN(l)              !   extinction at upper end of interval
  alb_NEW=w_LOW*alb_IN(l-1)+w_UPP*alb_IN(l)              !   albedo at upper end of interval
! WRITE (6,"(3X,'hone:  ',3X,2F10.5,5X,2F14.5,5X,2E10.3,5X,2F10.5,I7)")&
! &w_LOW,w_UPP,lam_OLD,lam_NEW,chi_OLD,chi_NEW,alb_OLD,alb_NEW,l
  lam_OUT(ll)=0.5*(lam_OLD+lam_NEW)                      !   exported wavelength
  dlam_OUT(ll)=lam_OLD-lam_NEW                           !   exported wavelength interval
  chi_OUT(ll)=0.5*(chi_OLD+chi_NEW)                      !   exported extinction
  alb_OUT(ll)=0.5*(alb_OLD+alb_NEW)                      !   exported albedo
!WRITE (6,"(I6,2F15.5,E15.3,F15.5)") ll,lam_OUT(ll),dlam_OUT(ll),chi_OUT(ll),alb_OUT(ll)
  ll=ll+1                                                !   increment ID of exported wavelength
  l=l-1                                                  !   take a step back
  lam_OLD=lam_NEW                                        !   update lam_OLD
  chi_OLD=chi_NEW                                        !   update chi_OLD
  alb_OLD=alb_NEW                                        !   update alb_OLD
ENDDO                                                    ! imported wavelengths exhausted
l_TOT=ll-1                                               ! record l_TOT

                                                         ! REVERSE ORDER OF WAVELENGTHS
DO l=1,l_TOT/2                                           ! start loop over lower half
  ll=l_TOT-l+1                                           !   compute ID of wavelength to swap with
  dum_R=lam_OUT(l)                                       !   swap ................
  lam_OUT(l)=lam_OUT(ll)                                 !   .... the ............
  lam_OUT(ll)=dum_R                                      !   ......... wavelengths
  dum_R=dlam_OUT(l)                                      !   swap .....................
  dlam_OUT(l)=dlam_OUT(ll)                               !   ...... the wavelength ....
  dlam_OUT(ll)=dum_R                                     !   ................ intervals
  dum_R=chi_OUT(l)                                       !   swap ...............
  chi_OUT(l)=chi_OUT(ll)                                 !   .... the ...........
  chi_OUT(ll)=dum_R                                      !   ........ extinctions
  dum_R=alb_OUT(l)                                       !   swap ...........
  alb_OUT(l)=alb_OUT(ll)                                 !   .... the .......
  alb_OUT(ll)=dum_R                                      !   ........ albedos
ENDDO                                                    ! end loop over lower half

IF (DustList==1) THEN
  WRITE (6,"(/,3X,'DISCRETE DUST PROPERTIES, l_TOT =',I4)") l_TOT
  WRITE (6,"(3X,'lam:',3X,5F10.5,6X,5F10.1)") lam_OUT(1:5),lam_OUT(l_TOT-4:l_TOT)
  WRITE (6,"(3X,'dlam:',2X,5F10.5,6X,5F10.1)") dlam_OUT(1:5),dlam_OUT(l_TOT-4:l_TOT) 
  WRITE (6,"(3X,'chi:',3X,5E10.3,6X,5E10.3)") chi_OUT(1:5),chi_OUT(l_TOT-4:l_TOT)
  WRITE (6,"(3X,'alb:',3X,5F10.5,6X,5F10.5)") alb_OUT(1:5),alb_OUT(l_TOT-4:l_TOT)
ENDIF

                                                         ! EXTRAPOLATION
f_lam=lam_OUT(l_TOT)/lam_OUT(l_TOT-1)                    ! compute ratio between successive wavelengths
f_chi=chi_OUT(l_TOT)/chi_OUT(l_TOT-1)                    ! compute ratio between successive opacities
DO WHILE (lam_OUT(l_TOT)<(0.1E+06))                      ! start increasing wavelength
  l_TOT=l_TOT+1                                          !   increment l_TOT
  lam_OUT(l_TOT)=f_lam*lam_OUT(l_TOT-1)                  !   increase lam_OUT
  dlam_OUT(l_TOT)=2.*(lam_OUT(l_TOT)-lam_OUT(l_TOT-1))  &!   compute ....
                                    &-dlam_OUT(l_TOT-1)  !   ... dlam_OUT
  chi_OUT(l_TOT)=f_chi*chi_OUT(l_TOT-1)                  !   decrease chi_OUT
  alb_OUT(l_TOT)=alb_OUT(l_TOT-1)                        !   leave alb_OUT the same
ENDDO                                                    ! stop increasing wavelength

IF (DustList==1) THEN
  WRITE (6,"(/,3X,'EXTENDED DUST PROPERTIES, l_TOT =',I4)") l_TOT
  WRITE (6,"(3X,'lam:',3X,5F10.1,6X,5F10.1)") lam_OUT(l_TOT-13:l_TOT-9),lam_OUT(l_TOT-4:l_TOT)
  WRITE (6,"(3X,'dlam:',2X,5F10.1,6X,5F10.1)") dlam_OUT(l_TOT-13:l_TOT-9),dlam_OUT(l_TOT-4:l_TOT) 
  WRITE (6,"(3X,'chi:',3X,5E10.3,6X,5E10.3)") chi_OUT(l_TOT-13:l_TOT-9),chi_OUT(l_TOT-4:l_TOT)
  WRITE (6,"(3X,'alb:',3X,5F10.5,6X,5F10.5)") alb_OUT(l_TOT-13:l_TOT-9),alb_OUT(l_TOT-4:l_TOT)
  WRITE (*,*) ' '
ENDIF
   
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RadTrans_DustPropertiesFromDraine
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RadTrans_PlotDustProperties(l_TOT,lam,chi,alb)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine plots the optical properties of the dust grains, and 
! modified Planck spectra at a selection of temperatures. It is given:
!   the number of wavelengths            (l_TOT);   
!   the discrete wavelengths             (lam(1:l_TOT)); 
!   the discrete extinction opacities    (chi(1:l_TOT)); 
!   and the discrete albedos             (alb(1:l_TOT)).

IMPLICIT NONE                                            ! DECLARATIONS
REAL(KIND=8),INTENT(IN),DIMENSION(1:l_TOT)  :: alb       ! the discrete albedos
REAL(KIND=8),INTENT(IN),DIMENSION(1:l_TOT)  :: chi       ! the discrete extinction opacities
REAL(KIND=8)                                :: dum_R     ! a dummy real
INTEGER                                     :: k         ! a dummy temperature ID
INTEGER                                     :: l         ! a dummy wavelength ID
INTEGER                                     :: l_TOT     ! the number of discrete wavelengths
REAL(KIND=8),INTENT(IN),DIMENSION(1:l_TOT)  :: lam       ! the discrete  wavelengths
REAL(KIND=8)                                :: lam_MAX   ! the maximum wavelength
REAL(KIND=8)                                :: lam_MIN   ! the minimum wavelength
REAL(KIND=8),DIMENSION(1:5)                 :: T         ! the prescribed temperatures
                                                         ! FOR PGPLOT
REAL(KIND=4),DIMENSION(1:l_TOT)             :: PGx       ! array for abscissa (log10[lam])
REAL(KIND=4)                                :: PGx_MAX   ! upper limit on abscissa
REAL(KIND=4)                                :: PGx_MIN   ! lower limit on abscissa
REAL(KIND=4),DIMENSION(1:l_TOT)             :: PGy       ! array for ordinate (log10[chi,PlanckFn])
REAL(KIND=4)                                :: PGy_MAX   ! upper limit on ordinate
REAL(KIND=4)                                :: PGy_MIN   ! lower limit on ordinate
REAL(KIND=4),DIMENSION(1:l_TOT)             :: PGz       ! array for ordinate (log10[alb,VolEm])
REAL(KIND=4)                                :: PGz_MAX   ! upper limit on ordinate
REAL(KIND=4)                                :: PGz_MIN   ! lower limit on ordinate

                                                         ! DUST PROPERTIES
PGx_MIN=+0.1E+11                                         ! set PGx_MIN to improbably high value
PGx_MAX=-0.1E+11                                         ! set PGx_MAX to improbably low value
PGy_MIN=+0.1E+11                                         ! set PGy_MIN to improbably high value
PGy_MAX=-0.1E+11                                         ! set PGy_MAX to improbably low value
DO l=1,l_TOT                                             ! start loop over wavelengths
  PGx(l)=LOG10(lam(l))                                   !   compute LOG10[lam]
  IF (PGx(l)<PGx_MIN) PGx_MIN=PGx(l)                     !   reduce PGx_MIN, as appropriate
  IF (PGx(l)>PGx_MAX) PGx_MAX=PGx(l)                     !   increase PGx_MAX, as appropriate
  PGy(l)=LOG10(chi(l))                                   !   compute LOG10[chi]
  IF (PGy(l)<PGy_MIN) PGy_MIN=PGy(l)                     !   reduce PGy_MIN, as appropriate
  IF (PGy(l)>PGy_MAX) PGy_MAX=PGy(l)                     !   increase PGy_MAX, as appropriate
  PGz(l)=10*alb(l)                                       !   compute LOG10[alb]
ENDDO                                                    ! end loop over wavelengths
dum_R=0.1*(PGx_MAX-PGx_MIN)                              ! compute margin for abscissa
PGx_MIN=PGx_MIN-dum_R                                    ! compute minimum abscissa
PGx_MAX=PGx_MAX+dum_R                                    ! compute maximum abscissa
dum_R=0.1*(PGy_MAX-PGy_MIN)                              ! compute margin for ordinate
PGy_MIN=PGy_MIN-dum_R                                    ! compute minimum ordinate
PGy_MAX=PGy_MAX+dum_R                                    ! compute maximum ordinate
CALL PGBEG(0,'/XWINDOW',1,1)                             ! open PGPLOT to display on screen
!CALL PGBEG(0,'/PS',1,2)                                  ! open PGPLOT to produce postscript
CALL PGSLW(1)                                            ! select line weight
CALL PGSCH(1.0)                                          ! select character height
CALL PGENV(PGx_MIN,PGx_MAX,PGy_MIN,PGy_MAX,0,0)          ! construct frame
CALL PGLAB('log\d10\u[\gl/\gmm]','log\d10\u[\gx/cm\u2\dg\u-1\d]  and  10\fia',&
     &'DUST EXTINCTION OPACITY, \gx, AND ALBEDO, \fia\fn, AS A FUNCTION OF WAVELENGTH, \gl.')
CALL PGSLS(1)                                            ! select full line
CALL PGLINE(l_TOT,PGx,PGy)                               ! plot extinction curve
CALL PGTEXT(1.7,2.4,'DUST EXTINCTION OPACITY')           ! label ..............
CALL PGTEXT(2.44,2.0,'log\d10\u[\gx/cm\u2\dg\u-1\d]')    ! ... extinction curve
CALL PGSLS(2)                                            ! select dashed line
CALL PGLINE(l_TOT,PGx,PGz)                               ! plot 10 x albedo
CALL PGTEXT(-0.4,0.80,'DUST ALBEDO')                     ! label ..........
CALL PGTEXT(0.37,0.45,'10\fia\fn')                       ! ... albedo curve
CALL PGEND                                               ! close PGPLOT

                                                         ! PLANCK FUNCTIONS AND VOLUME EMISSIVITIES
T(1)=3.16; T(2)=10.0; T(3)=31.6; T(4)=100.; T(5)=316.    ! input selected temperatures
CALL PGBEG(0,'/XWINDOW',1,1)                             ! open PGPLOT to display on screen
!CALL PGBEG(0,'/PS',1,2)                                  ! open PGPLOT to produce postscript
CALL PGENV(0.2,4.5,-4.2,+0.6,0,0)                        ! construct frame
CALL PGLAB('log\d10\u[\gl/\gmm]','log\d10\u[\fiB\fn\d\gl\u(\fiT\fn)]  and  log\d10\u[\fij\fn\d\gl\u(\fiT\fn)]',&
&'PLANCK FUNCTION, \fiB\fn\d\gl\u(\fiT\fn), AND VOLUME EMISSIVITY, \fij\fn\d\gl\u(\fiT\fn), AS A FUNCTION OF WAVELENGTH, \gl. ')
DO k=1,5                                                 ! start loop over temperatures
  dum_R=(0.143878E+05)/T(k)                              !   compute lambda_T=hc/kT
  lam_MIN=0.03*dum_R                                     !   compute minimum significant wavelength
  lam_MAX=10.0*dum_R                                     !   compute maximum significant wavelength
  PGy_MAX=-0.1E+21                                       !   set PGy_MAX to absurdly low value
  PGz_MAX=-0.1E+21                                       !   set PGz_MAX to absurdly low value
  PGy=-0.2E+21                                           !   set all PGy to even lower value
  PGz=-0.2E+21                                           !   set all PGz to even lower value
  DO l=1,l_TOT                                           !   start loop over wavelengths
    IF (lam(l)<lam_MIN) CYCLE                            !     [IF] wavelength very low, [CYCLE]
    IF (lam(l)>lam_MAX) CYCLE                            !     [IF] wavelength very high, [CYCLE]
    PGy(l)=1./(lam(l)**5*(EXP(dum_R/lam(l))-1.))         !     compute Planck Function
    PGz(l)=PGy(l)*chi(l)*(1.-alb(l))                     !     compute volume emissivity
    PGy(l)=LOG10(PGy(l))                                 !     compute LOG(PGy)
    PGz(l)=LOG10(PGz(l))                                 !     compute LOG(PGz)
    IF (PGy(l)>PGy_MAX) PGy_MAX=PGy(l)                   !     update PGy_MAX, as appropriate
    IF (PGz(l)>PGz_MAX) PGz_MAX=PGz(l)                   !     update PGz_MAX, as appropriate
  ENDDO                                                  !   end loop over wavelengths
  PGy=PGy-PGy_MAX                                        !   normalise PGy
  PGz=PGz-PGz_MAX                                        !   normalise PGz
  CALL PGSLS(2)                                          !   invoke dashed line
  CALL PGLINE(l_TOT,PGx,PGy)                             !   plot Planck Function
  CALL PGSLS(1)                                          !   invoke full line
  CALL PGLINE(l_TOT,PGx,PGz)                             !   plot volume emissivity
  CALL PGTEXT(+0.85,+0.10,'316.K')                       !   label 316.K plots
  CALL PGTEXT(+1.30,+0.10,'100.K')                       !   label 100.K plots
  CALL PGTEXT(+1.75,+0.10,'31.6K')                       !   label 31.6K plots
  CALL PGTEXT(+2.25,+0.10,'10.0K')                       !   label 10.0K plots
  CALL PGTEXT(+2.75,+0.10,'3.16K')                       !   label 3.16K plots
ENDDO                                                    ! end loop over temperatures
CALL PGEND                                               ! close PGPLOT

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RadTrans_PlotDustProperties
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RadTrans_Temperatures(k_TOT,T_MIN,T_MAX,TempList,T)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine computes the discrete temperatures.
! It is given:
!   the number of temperatures needed      (k_TOT); 
!   the minimum temperature                (T_MIN); 
!   the maximum temperature                (T_MAX);  
!   and a flag to trigger printout         (TempList). 
! It returns:
!   the discrete temperatures              (T(0:k_TOT)), 
!   where T(0) represents T<T(1).

IMPLICIT NONE                                            ! DECLARATIONS
REAL(KIND=8)                                :: dum_R     ! dummy real
INTEGER                                     :: k         ! dummy temperature ID
INTEGER,INTENT(IN)                          :: k_TOT     ! number of temperatures
REAL(KIND=8)                                :: L_MAX     ! LOG10(T_MAX)
REAL(KIND=8)                                :: L_MIN     ! LOG10(T_MIN)
REAL(KIND=8),INTENT(OUT),DIMENSION(0:k_TOT) :: T         ! discrete temperatures
REAL(KIND=8)                                :: T_MAX     ! maximum temperature
REAL(KIND=8)                                :: T_MIN     ! maximum temperature
INTEGER,INTENT(IN)                          :: TempList  ! flag to trigger printout

                                                         ! COMPUTE TEMPERATURES
T(0)=T_MIN                                               ! set T(0) to T_MIN
dum_R=1./DBLE(k_TOT-1)                                   ! compute 1/(k_TOT-1)
L_MIN=LOG10(T_MIN)                                       ! compute LOG10[T_MIN]
L_MAX=LOG10(T_MAX)                                       ! compute LOG10[T_MAX]
DO k=1,k_TOT                                             ! start loop over temperatures
  T(k)=10.**(dum_R*(DBLE(k_TOT-k)*L_MIN+DBLE(k-1)*L_MAX))!   compute discrete temperatures
ENDDO                                                    ! end loop over temperatures

                                                         ! CONDITIONAL DIAGNOSTIC PRINTOUT
IF (TempList==1) THEN                                    ! [IF] printout sanctioned, [THEN]
  WRITE (6,"(/,3X,'DISCRETE TEMPERATURES (k_TOT =',I4,'):')") k_TOT
  WRITE (6,"(6F9.3,8X,6F9.3,/)") T(1:6),T(k_TOT-5:k_TOT) !
ENDIF                                                    ! [ENDIF]

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RadTrans_Temperatures
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RadTrans_EmissionProbabilities(k_TOT,T,l_TOT,lam,&
&dlam,chi,alb,ProbPlot,Pl,lPl_LOW,lPl_UPP,LM,Pr,lPr_LOW,lPr_UPP)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine computes the probabilites needed for 
! re-emitting absorbed luminosity packets. It is given:
!   the number of discrete temperatures                    (k_TOT); 
!   the discrete temperatures                              (T(1:k_TOT)); 
!   the number of discrete wavelengths                     (l_TOT); 
!   the discrete wavelengths                               (lam(1:l_TOT));
!   the corresponding wavelength intervals                 (dlam(1:l_TOT));   
!   the corresponding extinction opacities                 (chi(1:l_TOT)); 
!   the corresponding albedos                              (alb(1:l_TOT)); 
!   and a flag to plot the probabilities                   (ProbPlot).

! It returns:
!   the Planck probability for   
!   wavelength interval [lam(l),dlam(l)] at T=T(k)         (Pl(1:l_TOT,1:k_TOT));
!   highest ID with Pl(ID,k)<=(l-1)/l_TOT                  (lPl_LOW(1:l_TOT,1:k_TOT)); 
!   lowest ID with Pl(ID,k)>=l/l_TOT                       (lPl_LOW(1:l_TOT,1:k_TOT)); 
!   the MBB luminosity per unit mass at T(k)               (LM(0:k_TOT)); 
!   the re-emission probability for 
!   wavelength interval [lam(l),dlam(l)] at T=T(k)         (Pr(1:l_TOT,0:k_TOT)); 
!   highest ID with Pr(ID,k)<=(l-1)/l_TOT                  (lPr_LOW(1:l_TOT,0:k_TOT)); 
!   and lowest ID with Pr(ID,k)>=l/l_TOT                   (lPr_LOW(1:l_TOT,0:k_TOT)); 

! ProbPlot=>0 sanctions plots of the probabilities and gives number of LumPacks.

IMPLICIT NONE                                            ! DECLARATIONS
INTEGER,DIMENSION(1:l_TOT)                  :: acc_I     ! accumlator  
REAL(KIND=8),INTENT(IN),DIMENSION(1:l_TOT)  :: alb       ! discrete albedos
REAL(KIND=8),INTENT(IN),DIMENSION(1:l_TOT)  :: chi       ! discrete extinction opacities
REAL(KIND=8)                                :: cut       ! bifurcation cut
REAL(KIND=8),INTENT(IN),DIMENSION(1:l_TOT)  :: dlam      ! discrete wavelengths
REAL(KIND=8)                                :: dPl       ! increment for Planck probabilities
REAL(KIND=8)                                :: dum_R     ! dummy real
INTEGER                                     :: k         ! dummy temperature ID
INTEGER,INTENT(IN)                          :: k_TOT     ! number of temperatures
INTEGER                                     :: l         ! dummy ID of wavelength & discrete prob.
INTEGER                                     :: l_LOW     ! lower wavelength ID in binary search
INTEGER,DIMENSION(0:k_TOT)                  :: l_MAX     ! ID of longest significant wavelength
INTEGER                                     :: l_MID     ! trial wavelength ID in binary search
INTEGER,DIMENSION(0:k_TOT)                  :: l_MIN     ! ID of shortest significant wavelength
INTEGER,INTENT(IN)                          :: l_TOT     ! number of wavelengths
INTEGER                                     :: l_UPP     ! upper wavelength ID in binary search
INTEGER                                     :: ll        ! dummy ID of wavelength and LumPack
REAL(KIND=8),INTENT(IN),DIMENSION(1:l_TOT)  :: lam       ! discrete wavelengths
REAL(KIND=8)                                :: lamT      ! hc/k
REAL(KIND=8),INTENT(OUT),DIMENSION(0:k_TOT) :: LM        ! MBB luminosity per unit mass at T(k)
REAL(KIND=8)                                :: LRD       ! linear random deviate
INTEGER,INTENT(OUT),                                    &! highest ID with ........
&                DIMENSION(1:l_TOT,1:k_TOT) :: lPl_LOW   ! ... Pl(ID,k)<(l-1)/l_TOT
INTEGER,INTENT(OUT),                                    &! lowest ID with .....
&                DIMENSION(1:l_TOT,1:k_TOT) :: lPl_UPP   ! ... Pl(ID,k)>l/l_TOT
INTEGER,INTENT(OUT),                                    &! highest ID with ........
&                DIMENSION(1:l_TOT,0:k_TOT) :: lPr_LOW   ! ... Pr(ID,k)<(l-1)/l_TOT
INTEGER,INTENT(OUT),                                    &! lowest ID with .....
&                DIMENSION(1:l_TOT,0:k_TOT) :: lPr_UPP   ! ... Pr(ID,k)>l/l_TOT
REAL(KIND=8),DIMENSION(0:l_TOT)             :: P         ! discrete probabilities (l/l_TOT)
REAL(KIND=8),INTENT(OUT),                               &! Planck probability for wavelength interval ...
&                DIMENSION(0:l_TOT,1:k_TOT) :: Pl        ! ......... [lam(l),dlam(l)] at temperature T(k)
REAL(KIND=8),INTENT(OUT),                               &! re-emission prob. for wavelength interval ...
&                DIMENSION(0:l_TOT,0:k_TOT) :: Pr        ! ........ [lam(l),dlam(l)] at temperature T(k)
INTEGER,INTENT(IN)                          :: ProbPlot  ! flag to sanction plots of probabilities
REAL(KIND=8),INTENT(IN),DIMENSION(0:k_TOT)  :: T         ! temperatures
                                                         ! FOR PGPLOT
REAL(KIND=4),DIMENSION(1:l_TOT)             :: PGx       ! array for abscissa (log10[lam])
REAL(KIND=4)                                :: PGx_MAX   ! upper limit on abscissa
REAL(KIND=4)                                :: PGx_MIN   ! lower limit on abscissa
REAL(KIND=4),DIMENSION(1:l_TOT)             :: PGy       ! array for ordinate (log10[chi,PlanckFn])
REAL(KIND=4)                                :: PGy_MAX   ! upper limit on ordinate
REAL(KIND=4)                                :: PGy_MIN   ! lower limit on ordinate
REAL(KIND=4),DIMENSION(1:l_TOT)             :: PGz       ! array for ordinate (log10[alb,VolEm])
REAL(KIND=4)                                :: PGz_MAX   ! upper limit on ordinate
REAL(KIND=4)                                :: PGz_MIN   ! lower limit on ordinate

                                                         ! PROBABILITIES
Pl=0.                                                    ! set Planck probabilities to zero
Pr=0.                                                    ! set emission probabilities to zero
LM=0.                                                    ! set luminosity per unit mass to zero
lPl_LOW=0                                                ! set lPl_LOW to zero
lPl_UPP=0                                                ! set lPl_UPP to zero
lPr_LOW=0                                                ! set lPr_LOW to zero
lPr_UPP=0                                                ! set lPr_UPP to zero
l_MIN(0:k_TOT)=0                                         ! set l_MIN to 0
l_MAX(0:k_TOT)=0                                         ! set l_MAX to 0
                                                         ! CASE 0: T<=T_MIN
lamT=(0.143878E+05)/T(1)                                 ! compute hc/k_BT_MIN
DO l=1,l_TOT                                             ! start loop over wavelengths
  IF (lam(l)<0.027*lamT) CYCLE                           !   [IF] wavelength too short, [CYCLE]
  l_MIN(0)=l                                             !   record l_MIN
  EXIT                                                   !   [EXIT] (l_MIN has been found)
ENDDO                                                    ! end loop over wavelengths
DO l=l_TOT,1,-1                                          ! start backwards loop over wavelengths
  IF (lam(l)>11.00*lamT) CYCLE                           !   [IF] wavelength too long, [CYCLE]
  l_MAX(0)=l                                             !   record l_MAX
  EXIT                                                   !   [EXIT] (l_MAX has been found)
ENDDO                                                    ! end backwards loop over wavelengths
DO l=l_MIN(0),l_MAX(0)                                   ! start loop over significant wavelengths
  dum_R=EXP(lamT/lam(l))                                 !   compute e^[hc/k_BT_MINLambda]
  Pr(l,0)=Pr(l-1,0)                                     &!   increment re-emission .....
  &+(chi(l)*(1.-alb(l))*dlam(l)/(lam(l)**5*(dum_R-1.)))  !   ... probability for T<T_MIN
ENDDO                                                    ! end loop over significant wavelengths
LM(0)=(0.149671E+13)*Pr(l_MAX(0),0)                      ! compute luminosity per unit mass at T_MIN
Pr(l_MIN(0):l_MAX(0),0)=Pr(l_MIN(0):l_MAX(0),0)         &! normalise emission ..........
     &/Pr(l_MAX(0),0)                                    ! ... probabilities for T<T_MIN
Pr(l_MAX(0)+1:l_TOT,0)=1.                                ! set higher integrated probs to unity
                                                         ! CASES 1 to k_TOT: T_MIN<=T<=T_MAX
DO k=1,k_TOT                                             ! start loop over discrete temperatures
  lamT=(0.143878E+05)/T(k)                               !   compute hc/k_BT(k)
  DO l=1,l_TOT                                           !   start forward loop over wavelengths
    IF (lam(l)<0.027*lamT) CYCLE                         !     [IF] wavelength too short, [CYCLE]
    l_MIN(k)=l                                           !     record l_MIN
    EXIT                                                 !     [EXIT] (l_MIN has been found)
  ENDDO                                                  !   end forward loop over wavelengths
  DO l=l_TOT,1,-1                                        !   start backwards loop over wavelengths
    IF (lam(l)>11.00*lamT) CYCLE                         !     [IF] wavelength too long, [CYCLE]
    l_MAX(k)=l                                           !     record l_MAX
    EXIT                                                 !     [EXIT] (l_MAX has been found)
  ENDDO                                                  !   end backwards loop over wavelengths
  DO l=l_MIN(k),l_MAX(k)                                 !   start loop over wavelengths
    dum_R=EXP(lamT/lam(l))                               !     compute e^[hc/k_BT_MINLambda]
    dPl=dlam(l)/(lam(l)**5*(dum_R-1.))                   !     differential Planck emission probability
    Pl(l,k)=Pl(l-1,k)+dPl                                !     increment Planck emission probability
    Pr(l,k)=Pr(l-1,k)+(chi(l)*(1.-alb(l))*dum_R*dlam(l) &!     increment re-emission .....
                           &/(lam(l)**6*(dum_R-1.)**2))  !     ... probability for T>T_MIN
    LM(k)=LM(k)+(chi(l)*(1.-alb(l))*dPl)                 !     increment luminosity per unit mass
  ENDDO                                                  !   end loop over wavelengths
  dum_R=(0.659867E+16)*Pl(l_MAX(k),k)/(T(k)**4)          !   check agreement with blackbody flux
  Pl(l_MIN(k):l_MAX(k),k)=Pl(l_MIN(k):l_MAX(k),k)       &!   normalise ..............
                                       &/Pl(l_MAX(k),k)  !   ... Planck probabilities
  Pl(l_MAX(k)+1:l_TOT,k)=1.                              !   set higher integrated probs to unity
  Pr(l_MIN(k):l_MAX(k),k)=Pr(l_MIN(k):l_MAX(k),k)       &!   normalise ....................
                                       &/Pr(l_MAX(k),k)  !   ...e re-emission probabilities
  Pr(l_MAX(k)+1:l_TOT,k)=1.                              !   set higher integrated probs to unity
  LM(k)=(0.149671E+13)*LM(k)                             !   normalise luminosity per unit mass
                                                         !   DIGITISATION
  dum_R=1./DBLE(l_TOT)                                   !   spacing of discrete probabilities
  DO l=0,l_TOT                                           !   start loop over discrete probabilities
    P(l)=DBLE(l)*dum_R                                   !     compute discrete probabilities 
  ENDDO                                                  !   end loop over discrete probabilities
  ll=l_MIN(k)                                            !   set dummy wavelength ID to l_MIN
  DO l=1,l_TOT-1                                         !   start loop over discrete probabilities
    DO WHILE (Pl(ll,k)<P(l))                             !     start search for upper wavelength
      ll=ll+1                                            !       increase wavelength ID
    ENDDO                                                !     upper wavelength found
    lPl_UPP(l,k)=ll                                      !     record ID of upper wavelength
    ll=ll-1                                              !     step back before continuing
  ENDDO                                                  !   end loop over discrete probabilities
  lPl_UPP(l_TOT,k)=l_MAX(k)                              !   special case
  ll=l_MAX(k)                                            !   set dummy wavelength ID to l_MAX
  DO l=l_TOT,2,-1                                        !   start loop over discrete probabilities
    DO WHILE (Pl(ll,k)>P(l-1))                           !     start search for lower wavelength
      ll=ll-1                                            !       decrease wavelength ID
    ENDDO                                                !     lower wavelength found
    lPl_LOW(l,k)=ll                                      !     record ID of lower wavelength
    ll=ll+1                                              !     step back before continuing
  ENDDO                                                  !   end loop over discrete probabilities
  lPl_LOW(1,k)=l_MIN(k)                                  !   special case
  ll=l_MIN(k)                                            !   set dummy wavelength ID to l_MIN
  DO l=1,l_TOT-1                                         !   start loop over discrete probabilities
    DO WHILE (Pr(ll,k)<P(l))                             !     start search for upper wavelength
      ll=ll+1                                            !       increase wavelength ID
    ENDDO                                                !     upper wavelength found
    lPr_UPP(l,k)=ll                                      !     record ID of upper wavelength
    ll=ll-1                                              !     step back before continuing
  ENDDO                                                  !   end loop over discrete probabilities
  lPr_UPP(l_TOT,k)=l_MAX(k)                              !   special case
  ll=l_MAX(k)                                            !   set dummy wavelength ID to l_MAX
  DO l=l_TOT,2,-1                                        !   start loop over discrete probabilities
    DO WHILE (Pr(ll,k)>P(l-1))                           !     start search for lower wavelength
      ll=ll-1                                            !       decrease wavelength ID
    ENDDO                                                !     lower wavelength found
    lPr_LOW(l,k)=ll                                      !     record ID of lower wavelength
    ll=ll+1                                              !     step back before continuing
  ENDDO                                                  !   end loop over discrete probabilities
  lPr_LOW(1,k)=l_MIN(k)                                  !   special case
  DO l=1,l_TOT                                           !   start loop over evenly spaced probs
    IF ((lPl_LOW(l,k)>=lPl_UPP(l,k)).OR.                &!     conditional ..........
    & (Pl(lPl_LOW(l,k),k)>P(l-1)+0.000001).OR.          &!     ... printout .........
    & (Pl(lPl_UPP(l,k),k)<P(l)))                        &!     ...... to check ......
    & WRITE(6,"(2I6,3X,2I6,3X,4F10.5,'HELP!')") k,l,    &!     ......... for ........
    & lPl_LOW(l,k),lPl_UPP(l,k),Pl(lPl_LOW(l,k),k),     &!     ............ faults ..
    & P(l-1),P(l),Pl(lPl_UPP(l,k),k)                     !     ............... in lPl
    IF ((lPr_LOW(l,k)>=lPr_UPP(l,k)).OR.                &!     conditional ..........
    & (Pr(lPr_LOW(l,k),k)>P(l-1)+0.000001).OR.          &!     ... printout .........
    & (Pr(lPr_UPP(l,k),k)<P(l)))                        &!     ...... to check ......
    & WRITE(6,"(2I6,3X,2I6,3X,4F10.5,'HELP!')") k,l,    &!     ......... for ........
    & lPr_LOW(l,k),lPr_UPP(l,k),Pr(lPr_LOW(l,k),k),     &!     ............ faults ..
    & P(l-1),P(l),Pr(lPr_UPP(l,k),k)                     !     ............... in lPl
  ENDDO                                                  !   end loop over evenly space probs
ENDDO                                                    ! end loop over discrete temperatures

IF (ProbPlot>0) THEN                                     ! CONDITIONAL DIAGNOSTIC PLOTS
  dum_R=DBLE(l_TOT)                                      !   compute real l_TOT
  DO k=1,k_TOT,49                                        !   start loop over temperatures
    PGx_MAX=LOG10(lam(l_MAX(k)))-0.9                     !     compute maximum abscissa
    PGx_MIN=LOG10(lam(l_MIN(k)))+0.1                     !     compute maximum abscissa
    PGy_MAX=-0.1E+11                                     !     set max ordinate to very low value
    DO l=1,l_TOT                                         !     start loop over wavelengths
      PGx(l)=LOG10(lam(l))                               !       compute abscissa
      PGy(l)=LOG10(Pr(l,k)-Pr(l-1,k))                    !       compute ordinate
      IF (PGy(l)>PGy_MAX) PGy_MAX=PGy(l)                 !       update max ordinate as appropriate
    ENDDO                                                !     end loop over wavelengths
    PGy_MAX=PGy_MAX+0.2                                  !     compute maximum ordinate
    PGy_MIN=PGy_MAX-2.6                                  !     compute minimum ordinate
    CALL PGBEG(0,'/XWINDOW',1,1)                         !     open PGPLOT to display on screen
    !CALL PGBEG(0,'/PS',1,2)                             !     open PGPLOT to produce postscript
    CALL PGENV(PGx_MIN,PGx_MAX,PGy_MIN,PGy_MAX,0,0)      !     construct frame
    CALL PGLAB('log\d10\u[\gl/\gmm]','log\d10\u[Pr]  and  log\d10\u[Pl]',&
    &'PLANCK AND RE-EMISSION PROBABILITIES, Pl and Pr, AS A FUNCTION OF WAVELENGTH, \gl. ')
    CALL PGSLS(2)                                        !     set line style to 'dashed'
    CALL PGLINE(l_TOT,PGx,PGy)                           !     plot discrete probabilities
    acc_I=0                                              !     set accumulator to zero
    DO ll=1,ProbPlot                                     !     start loop over luminosity packets
      CALL RANDOM_NUMBER(LRD)                            !!       generate linear random deviate
      l=CEILING(LRD*dum_R)                               !!       compute probability-bin ID
      l_LOW=lPr_LOW(l,k)                                 !!       register ID of largest lam(l) below bin
      l_UPP=lPr_UPP(l,k)                                 !!       register ID of smallest lam(l) above bin
      DO WHILE (l_UPP>l_LOW+1)                           !!       home in on wavelengths either side
        l_MID=(l_LOW+l_UPP)/2                            !!         compute middle wavelength ID
        IF (Pr(l_MID,k)<LRD) THEN                        !!         [IF] low, [THEN]
          l_LOW=l_MID                                    !!           increase l_LOW
        ELSE                                             !!         [ELSE] too high
          l_UPP=l_MID                                    !!           reduce l_UPP
        ENDIF                                            !!         [ENDIF] sorted
      ENDDO                                              !!       found the wavelengths either side
      cut=(Pr(l_UPP,k)-LRD)/(Pr(l_UPP,k)-Pr(l_LOW,k))    !!       location of cut relative to l_LOW
      CALL RANDOM_NUMBER(LRD)                            !!       generate linear random deviate
      IF (LRD<cut) THEN                                  !!       [IF] LRD below cut, [THEN]
        acc_I(l_LOW)=acc_I(l_LOW)+1                      !!         add LumPack to lower accumulator
      ELSE                                               !!       [ELSE] LRD above cut, so
        acc_I(l_UPP)=acc_I(l_UPP)+1                      !!         add LumPack to upper accumulator
      ENDIF                                              !!       [ENDIF]
    ENDDO                                                !     end loop over luminosity packets
    DO l=l_MIN(k),l_MAX(k)                               !     start loop over significant wavelengths
      PGx(l)=LOG10(lam(l)+0.5*dlam(l))                   !       compute abscissa
      PGy(l)=LOG10(REAL(acc_I(l))/REAL(ProbPlot))        !       compute ordinate
    ENDDO                                                !     end loop over significant wavelengths
    CALL PGSLS(1)                                        !     set line style to 'full'
    CALL PGLINE(l_TOT,PGx,PGy)                           !     plot estimated probabilities
    CALL PGEND                                           !     close PGPLOT
  ENDDO                                                  !   end loop over temperatures
  DO k=1,k_TOT,49                                        !   start loop over temperatures
    PGx_MAX=LOG10(lam(l_MAX(k)))-0.2                     !     compute maximum abscissa
    PGx_MIN=LOG10(lam(l_MIN(k)))+0.0                     !     compute minimum abscissa
    PGy_MAX=-0.1E+11                                     !     set max ordinate to very low value
    DO l=1,l_TOT                                         !     start loop over wavelengths
      PGx(l)=LOG10(lam(l))                               !       compute abscissa
      PGy(l)=LOG10(Pl(l,k)-Pl(l-1,k))                    !       compute ordinate
      IF (PGy(l)>PGy_MAX) PGy_MAX=PGy(l)                 !       update max ordinate as appropriate
    ENDDO                                                !     end loop over wavelengths
    PGy_MAX=PGy_MAX+0.2                                  !     compute maximum ordinate
    PGy_MIN=PGy_MAX-2.6                                  !     compute minimum ordinate
    CALL PGBEG(0,'/XWINDOW',1,1)                         !     open PGPLOT to display on screen
    !CALL PGBEG(0,'/PS',1,2)                             !     open PGPLOT to produce postscript
    CALL PGENV(PGx_MIN,PGx_MAX,PGy_MIN,PGy_MAX,0,0)      !     construct frame
    CALL PGLAB('log\d10\u[\gl/\gmm]','log\d10\u[Pr]  and  log\d10\u[Pl]',&
    &'PLANCK AND RE-EMISSION PROBABILITIES, Pl and Pr, AS A FUNCTION OF WAVELENGTH, \gl. ')
    CALL PGSLS(2)                                        !     set line style to 'dashed'
    CALL PGLINE(l_TOT,PGx,PGy)                           !     plot discrete probabilities
    acc_I=0                                              !     set accumulator to zero
    DO ll=1,ProbPlot                                     !     start loop over luminosity packets
      CALL RANDOM_NUMBER(LRD)                            !!       generate linear random deviate
      l=CEILING(LRD*dum_R)                               !!       compute probability-bin ID
      l_LOW=lPl_LOW(l,k)                                 !!       register ID of largest lam(l) below bin
      l_UPP=lPl_UPP(l,k)                                 !!       register ID of smallest lam(l) above bin
      DO WHILE (l_UPP>l_LOW+1)                           !!       home in on wavelengths either side
        l_MID=(l_LOW+l_UPP)/2                            !!         compute middle wavelength ID
        IF (Pl(l_MID,k)<LRD) THEN                        !!         [IF] low, [THEN]
          l_LOW=l_MID                                    !!           increase l_LOW
        ELSE                                             !!         [ELSE] too high
          l_UPP=l_MID                                    !!           reduce l_UPP
        ENDIF                                            !!         [ENDIF] sorted
      ENDDO                                              !!       found the wavelengths either side
      cut=(Pl(l_UPP,k)-LRD)/(Pl(l_UPP,k)-Pl(l_LOW,k))    !!       location of cut relative to l_LOW
      CALL RANDOM_NUMBER(LRD)                            !!       generate linear random deviate
      IF (LRD<cut) THEN                                  !!       [IF] LRD below cut, [THEN]
        acc_I(l_LOW)=acc_I(l_LOW)+1                      !!         add LumPack to lower accumulator
      ELSE                                               !!       [ELSE] LRD above cut, so
        acc_I(l_UPP)=acc_I(l_UPP)+1                      !!         add LumPack to upper accumulator
      ENDIF                                              !!       [ENDIF]
    ENDDO                                                !     end loop over luminosity packets
    DO l=l_MIN(k),l_MAX(k)                               !     start loop over significant wavelengths
      PGx(l)=LOG10(lam(l)+0.5*dlam(l))                   !       compute abscissa
      PGy(l)=LOG10(REAL(acc_I(l))/REAL(ProbPlot))        !       compute ordinate
    ENDDO                                                !     end loop over significant wavelengths
    CALL PGSLS(1)                                        !     set line style to 'full'
    CALL PGLINE(l_TOT,PGx,PGy)                           !     plot estimated probabilities
    CALL PGEND                                           !     close PGPLOT
  ENDDO                                                  !   end loop over temperatures
ENDIF                                                    ! end diagnostic plots

! The 19 linws marked with '!!' are basically the code to generate wavelengths for re-emitted LumPacks.
! Add lPl      <<<<<<<<<<       PRIORITY
  
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RadTrans_EmissionProbabilities
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RadTrans_Cylindrical1D_LinearShellSpacing(rho_O,W_O,p,W_B,n_TOT,CellList,w,eta,rho)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine is given:
!   the spinal density                                 (rho_O); 
!   the core radius                                    (W_O); 
!   the envelope density exponent                      (p); 
!   the boundary radius of the filament                (W_B); 
!   the number of shells                               (n_TOT);
!   and a flag for diagnostic printout                 (CellList).
! It returns :
!   the boundary radii of the shells                   (w(0:n_TOT)); 
!   the differential shell radius squared              (eta(0:n_tot));
!   and the mean density in each shell                 (rho(1:n_TOT)). 

IMPLICIT NONE                                            ! DECLARATIONS
REAL(KIND=8)                                :: dum_R     ! auxiliary real
INTEGER                                     :: CellList  ! flag for diagnistic printout
REAL(KIND=8),INTENT(OUT),DIMENSION(0:n_TOT) :: eta       ! shell boundary radius squared (in cm^2)
INTEGER                                     :: n         ! dummy ID of shell
INTEGER,INTENT(IN)                          :: n_TOT     ! number of shells
INTEGER,INTENT(IN)                          :: p         ! envelope density exponent
REAL(KIND=8),INTENT(OUT),DIMENSION(1:n_TOT) :: rho       ! mean density in shell
REAL(KIND=8),INTENT(IN)                     :: rho_O     ! spinal density (in g/cm^3)
REAL(KIND=8),INTENT(OUT),DIMENSION(0:n_TOT) :: w         ! shell boundary radius (in cm)
REAL(KIND=8),INTENT(IN)                     :: W_B       ! boundary radius of the filament (in cm)
REAL(KIND=8),INTENT(IN)                     :: W_O       ! core radius of the filament (in cm)
REAL(KIND=8)                                :: W_OSQD    ! core radius squared (in cm^2)

dum_R=W_B/DBLE(n_TOT)                                    ! compute shell width
W_OSQD=W_O*W_O                                           ! compute W_OSQD
w(0)=0.                                                  ! set w(0) to zero
eta(0)=0.                                                ! set eta(0) to zero
DO n=1,n_TOT                                             ! start loop over shells
  w(n)=DBLE(n)*dum_R                                     !   compute outer boundary radius
  eta(n)=w(n)*w(n)                                       !   compute outer shell radius squared
  IF (p==2) THEN                                         !   [IF] p=2, [THEN]
    rho(n)=LOG((W_OSQD+eta(n))/(W_OSQD+eta(n-1)))        !     compute logarithmic term
    rho(n)=rho(n)*rho_O*W_OSQD/(eta(n)-eta(n-1))         !     compute density
  ENDIF                                                  !   [ENDIF]
                                                         !!!!!  OTHER CASES FOR A RAINY DAY 
ENDDO                                                    ! end loop over shells

IF (CellList==1) THEN
  WRITE (6,"(3X,'SHELL BOUNDARIES, n_TOT =',I4)") n_TOT
  WRITE (6,"(3X,'w/(pc):',9X,5E10.3,6X,5E10.3)") w(1:5)/(3.086E+18),w(n_TOT-4:n_TOT)/(3.086E+18)
  WRITE (6,"(3X,'eta/(pc^2):',5X,5E10.3,6X,5E10.3)") eta(1:5)/(9.521E+36),eta(n_TOT-4:n_TOT)/(9.521E+36)
  WRITE (6,"(3X,'rho/(H2/cm^3):',2X,5E10.3,6X,5E10.3,/)") rho(1:5)/(4.77E-24),rho(n_TOT-4:n_TOT)/(4.77E-24)
ENDIF

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RadTrans_Cylindrical1D_LinearShellSpacing
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RadTrans_InjectAndTrack0(W_B,n_TOT,w,eta,i_TOT)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine deals with the zero-opacity case. It is given:
!   the boundary radius of the filament                   (W_B); 
!   the number of shells                                  (n_TOT); 
!   the boundary radii of the shells                      (w(0:n_TOT));
!   the boundary radii of the shells squared              (eta(0:N_tot)); 
!   and the number of luminosity packets to be injected   (i_TOT). 
! It prints out:
!   the normalised angle-mean intensity in each shell     (J0(0:n_TOT)) --- should be ~1; 
!   its mean                                              (mu_J0) --- should be ~1; 
!   and its standard deviation                            (sigma_J0) --- should be <<1.

IMPLICIT NONE                                            ! DECLARATIONS
REAL(KIND=8)                                :: alpha     ! 'outwardness' of packet (in pc)
REAL(KIND=8)                                :: beta      ! squared tangent-distance (in pc^2)
REAL(KIND=8),DIMENSION(1:3)                 :: e         ! direction of packet (unit vector)
REAL(KIND=8)                                :: e12_SQD   ! e_x^2+e_y^2
REAL(KIND=8),INTENT(IN),DIMENSION(0:n_TOT)  :: eta       ! shell boundary radius squared (in pc^2)
INTEGER                                     :: i         ! dummy ID of packet
INTEGER,INTENT(IN)                          :: i_TOT     ! number of packets to be injected
REAL(KIND=8),DIMENSION(1:n_TOT)             :: J0        ! mean intensity in shell
REAL(KIND=8)                                :: LRD       ! linear random deviate on [0,1]
REAL(KIND=8)                                :: mu_J0     ! shell-mean of J0
INTEGER                                     :: n,nn      ! dummy ID of shell
INTEGER,INTENT(IN)                          :: n_TOT     ! number of shells
REAL(KIND=8)                                :: phi       ! azimuthal angle of packet direction
REAL(KIND=8),DIMENSION(1:3)                 :: r         ! position of packet (vector; in pc)
REAL(KIND=8)                                :: r12_SQD   ! squared distance from spine (in pc^2)
REAL(KIND=8)                                :: s         ! distance to next boundary intercept (in pc)
REAL(KIND=8)                                :: sigma_J0  ! shell-SD of J0
REAL(KIND=8)                                :: sintheta  ! sine of polar angle of packet direction
REAL(KIND=8),INTENT(IN),DIMENSION(0:n_TOT)  :: w         ! shell boundary radius (in pc)
REAL(KIND=8),INTENT(IN)                     :: W_B       ! boundary radius of the filament (in pc)
REAL(KIND=8)                                :: W_SQD     ! squared distance radius of filament (in pc^2)

W_SQD=W_B**2                                             ! compute W_B squared
J0=0.                                                    ! set mean intensities to zero
DO i=1,i_TOT                                             ! start loop over packets
  r(1)=-W_B;  r(2)=0.;  r(3)=0.                          !   position packet on boundary of filament
  r12_SQD=W_SQD                                          !   compute squared distance from spine
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  e(1)=SQRT(LRD)                                         !   compute e(1)
  sintheta=SQRT(1.-LRD)                                  !   compute SIN(theta)
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  phi=6.28311855*LRD                                     !   compute phi
  e(2)=sintheta*COS(phi)                                 !   compute e_y
  e(3)=sintheta*SIN(phi)                                 !   compute e_z
  n=n_TOT                                                !   set shell ID to n_TOT (outermost shell)
  DO WHILE (n<=n_TOT)                                    !   keep going until packet exits filament
    nn=n                                                 !     record ID of shell being entered
    e12_SQD=e(1)**2+e(2)**2                              !     compute e_x^2+e_y^2
    alpha=((r(1)*e(1))+(r(2)*e(2)))/e12_SQD              !     compute 'outwardness'
    IF (alpha>0.) THEN                                   !     [IF] travelling outward, [THEN]
      s=-alpha+SQRT(alpha**2+((eta(n)-eta(n-1))/e12_SQD))!       compute path
      n=n+1                                              !       increase shell ID
    ELSE                                                 !     [ELSE] travelling inward, so
      beta=alpha**2-((eta(n)-eta(n-1))/e12_SQD)          !       compute beta, and
      IF (beta>0.) THEN                                  !       [IF] beta>0, [THEN] hits inner shell
        s=-alpha-SQRT(beta)                              !         compute path
        n=n-1                                            !         decrease shell ID
      ELSE                                               !       [ELSE] traverses shell, so
        s=-2.*alpha                                      !         compute path
        n=n+1                                            !         increase shell ID
      ENDIF                                              !       [ENDIF] inward cases done
    ENDIF                                                !     [ENDIF] all cases done
    r(1:2)=r(1:2)+(s*e(1:2))                             !     advance position of packet
    r12_SQD=(r(1)**2)+(r(2)**2)                          !     compute squared distance from spine
    J0(nn)=J0(nn)+s                                      !     increment mean intensity
  ENDDO                                                  !   packet exits filament
ENDDO                                                    ! end loop over packets
alpha=W_B/(2.*DBLE(i_TOT))                               ! use alpha as normalisation coefficient
mu_J0=0.                                                 ! set mu_J0 to zero
sigma_J0=0.                                              ! set sigma_J0 to zero
DO n=1,n_TOT                                             ! start loop over shells
  J0(n)=J0(n)*alpha/(eta(n)-eta(n-1))                    !   normalise J0
  mu_J0=mu_j0+J0(n)                                      !   increment mu_J0 accumulator
  sigma_J0=sigma_J0+(J0(n)**2)                           !   increment sigma_J0 accumulator
ENDDO                                                    ! end loop over shells
mu_J0=mu_J0/DBLE(n_TOT)                                  ! compute mu_J0
sigma_J0=SQRT((sigma_J0/DBLE(n_TOT))-(mu_J0**2))         ! compute sigma_J0
WRITE (6,"(/,3X,'TEST CASE 0: ZERO OPACITY, SHELL MEAN INTENSITIES (J0), n_TOT =',I4)") n_TOT
WRITE (6,"(3X,'J0:',4X,5F10.5,6X,5F10.5)") J0(1:5),J0(n_TOT-4:n_TOT)
WRITE (6,"(3X,'Shell-mean of angle-mean intensity:',3X,F10.7,17X,'(mu_J0)')") mu_J0
WRITE (6,"(3X,'Shell-SD of angle-mean intensity:',5X,F10.7,17X,'(sigma_J0)',/)") sigma_J0

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RadTrans_InjectAndTrack0
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RadTrans_InjectAndTrack1(W_B,n_TOT,w,eta,kappa_V,i_TOT)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine deals with the pure scattering case. It is given:
!   the boundary radius of the filament                   (W_B); 
!   the number of shells                                  (n_TOT); 
!   the boundary radii of the shells                      (w(0:n_TOT));
!   the boundary radii of the shells squared              (eta(0:n_tot));
!   the volume scattering opacity coefficient             (kappa_V); 
!   and the number of luminosity packets to be injected   (i_TOT). 
! It prints out:
!   the normalised angle-mean intensity in each shell     (J0(0:n_TOT)) --- should be ~1; 
!   its mean                                              (mu_J0) --- should be ~1; 
!   and its standard deviation                            (sigma_J0) --- should be <<1.

IMPLICIT NONE                                            ! DECLARATIONS
REAL(KIND=8)                                :: alpha     ! 'outwardness' of packet (in pc)
REAL(KIND=8)                                :: beta      ! squared tangent-distance (in pc^2)
REAL(KIND=8),DIMENSION(1:3)                 :: e         ! direction of packet (unit vector)
REAL(KIND=8)                                :: e12_SQD   ! e_x^2+e_y^2
REAL(KIND=8),INTENT(IN),DIMENSION(0:n_TOT)  :: eta       ! shell boundary radius squared (in pc^2)
INTEGER                                     :: i         ! dummy ID of packet
INTEGER,INTENT(IN)                          :: i_TOT     ! number of packets to be injected
REAL(KIND=8),DIMENSION(1:n_TOT)             :: J0        ! mean intensity in shell
REAL(KIND=8),INTENT(IN)                     :: kappa_V   ! volume scattering opacity coefficient
REAL(KIND=8)                                :: LRD       ! linear random deviate on [0,1]
REAL(KIND=8)                                :: mu_J0     ! shell-mean of J0
INTEGER                                     :: n,nn      ! dummy ID of shell
INTEGER,INTENT(IN)                          :: n_TOT     ! number of shells
REAL(KIND=8)                                :: phi       ! azimuthal angle of packet direction
REAL(KIND=8),DIMENSION(1:3)                 :: r         ! position of packet (vector; in pc)
REAL(KIND=8)                                :: r12_SQD   ! squared distance from spine (in pc^2)
REAL(KIND=8)                                :: s         ! distance to next boundary intercept (in pc)
REAL(KIND=8)                                :: sigma_J0  ! shell-SD of J0
REAL(KIND=8)                                :: sintheta  ! sine of polar angle of packet direction
REAL(KIND=8)                                :: tau       ! optical depth
REAL(KIND=8),INTENT(IN),DIMENSION(0:n_TOT)  :: w         ! shell boundary radius (in pc)
REAL(KIND=8),INTENT(IN)                     :: W_B       ! boundary radius of the filament (in pc)
REAL(KIND=8)                                :: W_SQD     ! squared distance radius of filament (in pc^2)
                                                         ! DIAGNOSTIC CHECKS
INTEGER                                     :: dir,eINNUM,eSCNUM,tauNUM
REAL(KIND=8),DIMENSION(1:3)                 :: eINBAR,eSCBAR,eINSIG,eSCSIG
REAL(KIND=8)                                :: tauBAR,tauSIG

W_SQD=W_B**2                                             ! compute W_B squared
J0=0.                                                    ! set mean intensities to zero
eINBAR=0.; eINSIG=0.; eINNUM=0                           !D
eSCBAR=0.; eSCSIG=0.; eSCNUM=0                           !D
tauBAR=0.; tauSIG=0.; tauNUM=0                           !D
!WRITE (6,"(6X,'i:',8X,'e1:',5X,'e2:',6X,'e:',8X,'LRD:',4X,'tau:',9X,'r1:',5X,'r2:',6X,'r:',9X,'wn:',6X,'s:',6X,'MODE:')")
DO i=1,i_TOT                                             ! start loop over packets
  r(1)=-W_B;  r(2)=0.;  r(3)=0.                          !   position packet on boundary of filament
  r12_SQD=W_SQD                                          !   compute squared distance from spine
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  e(1)=SQRT(LRD)                                         !   compute e(1)
  sintheta=SQRT(1.-LRD)                                  !   compute SIN(theta)
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  phi=6.2831853*LRD                                      !   compute phi
  e(2)=sintheta*COS(phi)                                 !   compute e(2)
  e(3)=sintheta*SIN(phi)                                 !   compute e(3)
  eINBAR(1)=eINBAR(1)+e(1)                               !D
  eINBAR(2:3)=eINBAR(2:3)+ABS(e(2:3))                    !D
  eINSIG(1:3)=eINSIG(1:3)+e(1:3)**2                      !D
  eINNUM=eINNUM+1                                        !D
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  tau=-LOG(LRD)                                          !   compute optical-depth
  tauBAR=tauBAR+tau                                      !D
  tauSIG=tauSIG+tau**2                                   !D
  tauNUM=tauNUM+1                                        !D
  tau=tau/kappa_V                                        !   convert optical depth to range
  n=n_TOT                                                !   set shell ID to n_TOT (outermost shell)
  IF (i==0) WRITE (6,"(I8,3X,3F8.4,4X,2F8.4)") i,e(1:2),SQRT(e(1)**2+e(2)**2+e(3)**2),LRD,(tau/W_B)
  DO WHILE (n<=n_TOT)                                    !   keep going until packet exits filament
    nn=n                                                 !     record ID of shell being entered
    e12_SQD=e(1)**2+e(2)**2                              !     compute e_x^2+e_y^2
    alpha=((r(1)*e(1))+(r(2)*e(2)))/e12_SQD              !     compute 'outwardness'
    IF (alpha>0.) THEN                                   !     [IF] travelling outward, [THEN]
      s=-alpha+SQRT(alpha**2+((eta(n)-r12_SQD)/e12_SQD)) !       compute path
      n=n+1                                              !       increase shell ID
      dir=+1                                             !       record direction relative to shells
    ELSE                                                 !     [ELSE] travelling inward, so
      beta=alpha**2+((eta(n-1)-r12_SQD)/e12_SQD)         !       compute beta, and
      IF (beta>0.) THEN                                  !       [IF] beta>0, [THEN] hits inner shell
        s=-alpha-SQRT(beta)                              !         compute path
        n=n-1                                            !         decrease shell ID
        dir=-1                                           !       record direction relative to shells
      ELSE                                               !       [ELSE] traverses shell, so
        s=-alpha+SQRT(alpha**2+(eta(n)-r12_SQD)/e12_SQD) !         compute path
        n=n+1                                            !         increase shell ID
        dir=0                                            !         record direction relative to shells
      ENDIF                                              !       [ENDIF] inward cases done
    ENDIF                                                !     [ENDIF] all cases done
    IF (tau<s) THEN                                      !     [IF] range of packet too small
      s=tau                                              !       set s to tau
      n=nn                                               !       record that still in same shell
      dir=2                                              !       record direction relative to shells
    ELSE                                                 !     [ELSE] 
      tau=tau-s                                          !       decrease range as appropriate
    ENDIF                                                !     [ENDIF]
    r(1:2)=r(1:2)+(s*e(1:2))                             !     advance position of packet
    r12_SQD=(r(1)**2)+(r(2)**2)                          !     compute squared distance from spine
    IF (dir==2) THEN                                     !     [IF] scattering adjustments needed
        CALL RANDOM_NUMBER(LRD)                          !       generate linear random deviate on [0,1]
        e(1)=2.*LRD-1.                                   !       compute e(1)
        sintheta=SQRT(1.-e(1)**2)                        !       compute SIN(theta)
        CALL RANDOM_NUMBER(LRD)                          !       generate linear random deviate on [0,1]
        phi=6.28311855*LRD                               !       compute phi
        e(2)=sintheta*COS(phi)                           !       compute e(2)
        e(3)=sintheta*SIN(phi)                           !       compute e(3)
        eSCBAR(1:3)=eSCBAR(1:3)+ABS(e(1:3))              !D
        eSCSIG(1:3)=eSCSIG(1:3)+e(1:3)**2                !D
        eSCNUM=eSCNUM+1                                  !D
        CALL RANDOM_NUMBER(LRD)                          !       generate linear random deviate on [0,1]
        tau=-LOG(LRD)                                    !       compute optical-depth
        tauBAR=tauBAR+tau                                !D
        tauSIG=tauSIG+tau**2                             !D
        tauNUM=tauNUM+1                                  !D
        tau=tau/kappa_V                                  !       convert optical depth to range
    ENDIF                                                !     [ENDIF] scattering adjustments completed
    IF (i==0) THEN                                       !D
      IF (dir==-1) WRITE (6,"(59X,3F8.4,4X,2F8.4,5X,'INWARD')") &
      &(r(1:2)/W_B),(SQRT(r12_SQD)/W_B),(w(n)/W_B),(s/W_B)
      IF (dir==0) WRITE (6,"(59X,3F8.4,4X,2F8.4,5X,'ACROSS')") &
      &(r(1:2)/W_B),(SQRT(r12_SQD)/W_B),(w(n-1)/W_B),(s/W_B)
      IF (dir==+1) WRITE (6,"(59X,3F8.4,4X,2F8.4,4X,'OUTWARD')") &
      &(r(1:2)/W_B),(SQRT(r12_SQD)/W_B),(w(n-1)/W_B),(s/W_B)
      IF (dir==+2) WRITE (6,"(11X,3F8.4,4X,2F8.4,4X,3F8.4,4X,2F8.4,4X,'SCATTER')") e(1:2),&
      &SQRT(e(1)**2+e(2)**2+e(3)**2),LRD,(tau/W_B),(r(1:2)/W_B),(SQRT(r12_SQD)/W_B),(w(n)/W_B),(s/W_B)
    ENDIF                                                !D
    J0(nn)=J0(nn)+s                                      !     increment mean intensity
  ENDDO                                                  !   packet exits filament
ENDDO                                                    ! end loop over packets
WRITE (6,"(/,3X,'TEST CASE 1: UNIFORM VOLUME SCATTERING OPACITY,  n_TOT =',I4,/)") n_TOT
alpha=1./DBLE(eINNUM)                                    ! use alpha as normalisation coefficient
eINBAR(1:3)=alpha*eINBAR(1:3)                            ! compute mean of injection directions
eINSIG(1:3)=SQRT(alpha*eINSIG(1:3)-eINBAR(1:3)**2)       ! compute StDev of injection directions
WRITE (6,"(3X,'MEAN and StDev OF ENTRY DIRECTION:',4X,3(3X,2F8.5))") &
&eINBAR(1),eINSIG(1),eINBAR(2),eINSIG(2),eINBAR(3),eINSIG(3)
WRITE (6,"(27X,'SHOULD BE:',8X,'0.66667 0.23570    0.42441 0.26433    0.42441 0.26433',/)")
alpha=1./DBLE(eSCNUM)                                    ! use alpha as normalisation coefficient
eSCBAR(1:3)=alpha*eSCBAR(1:3)                            ! compute mean of scattering directions
eSCSIG(1:3)=SQRT(alpha*eSCSIG(1:3)-eSCBAR(1:3)**2)       ! compute StDev of scattering directions
WRITE (6,"(3X,'MEAN and StDev OF SCATTERED DIRECTION:',3(3X,2F8.5))") &
     &eSCBAR(1),eSCSIG(1),eSCBAR(2),eSCSIG(2),eSCBAR(3),eSCSIG(3)
WRITE (6,"(31X,'SHOULD BE:',4X,'0.50000 0.28868    0.50000 0.28868    0.50000 0.28868',/)")
alpha=1./DBLE(tauNUM)                                    ! use alpha as normalisation coefficient
tauBAR=alpha*tauBAR                                      ! compute mean of optical depths
tauSIG=SQRT(alpha*tauSIG-tauBAR**2)                      ! compute StDev of oprical depths
WRITE (6,"(3X,'MEAN AND StDev OF OPTICAL DEPTH:',9X,2F8.5)") tauBAR,tauSIG
WRITE (6,"(25X,'SHOULD BE:',10X,'1.00000 1.00000',/)")
WRITE (6,"(3X,'i_TOT =',I8,'; eINNUM =',I8,'; eSCNUM =',I8,'; eINNUM+eSCNUM =',I8,'; tauNUM =',I8,'; paths =',F8.3)") &
&i_TOT,eINNUM,eSCNUM,(eINNUM+eSCNUM),tauNUM,DBLE(eINNUM+eSCNUM)/DBLE(i_TOT)
alpha=W_B/(2.*DBLE(i_TOT))                               ! use alpha as normalisation coefficient
mu_J0=0.                                                 ! set mu_J0 to zero
sigma_J0=0.                                              ! set sigma_J0 to zero
DO n=1,n_TOT                                             ! start loop over shells
  J0(n)=J0(n)*alpha/(eta(n)-eta(n-1))                    !   normalise J0
  mu_J0=mu_j0+J0(n)                                      !   increment mu_J0 accumulator
  sigma_J0=sigma_J0+(J0(n)**2)                           !   increment sigma_J0 accumulator
ENDDO                                                    ! end loop over shells
mu_J0=mu_J0/DBLE(n_TOT)                                  ! compute mu_J0
sigma_J0=SQRT((sigma_J0/DBLE(n_TOT))-(mu_J0**2))         ! compute sigma_J0
WRITE (6,"(/,3X,'SHELL MEAN INTENSITIES (J0):')")
WRITE (6,"(X,6F9.5,6X,6F9.5)") J0(1:6),J0(n_TOT-5:n_TOT)
WRITE (6,"(3X,'Shell-mean of angle-mean intensity:',3X,F10.7,17X,'(mu_J0)')") mu_J0
WRITE (6,"(3X,'Shell-SD of angle-mean intensity:',5X,F10.7,17X,'(sigma_J0)',/)") sigma_J0

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RadTrans_InjectAndTrack1
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RadTrans_InjectAndTrack2(W_B,n_TOT,w,eta,rho,kappa_M,i_TOT)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine deals with the pure scattering case. It is given: 
!   the boundary radius of the filament                   (W_B); 
!   the number of shells                                  (n_TOT); 
!   the boundary radii of the shells                      (w(0:n_TOT)); 
!   the boundary radii of the shells squared              (eta(0:n_tot)); 
!   the mean densities in the shells                      (rho(1:n_TOT)); 
!   the mass scattering opacity coefficient               (kappa_M); 
!   and the number of luminosity packets to be injected   (i_TOT). 
! It prints out:
!   the normalised angle-mean intensity in each shell     (J0(0:n_TOT)) --- should be ~1; 
!   its mean                                              (mu_J0) --- should be ~1; 
!   and its standard deviation                            (sigma_J0) --- should be <<1.

IMPLICIT NONE                                            ! DECLARATIONS
REAL(KIND=8)                                :: alpha     ! 'outwardness' of packet (in cm)
REAL(KIND=8)                                :: beta      ! squared tangent-distance (in cm^2)
REAL(KIND=8),DIMENSION(1:3)                 :: e         ! direction of packet (unit vector)
REAL(KIND=8)                                :: e12_SQD   ! e_x^2+e_y^2
REAL(KIND=8),INTENT(IN),DIMENSION(0:n_TOT)  :: eta       ! shell boundary radius squared (in cm^2)
INTEGER                                     :: i         ! dummy ID of packet
INTEGER,INTENT(IN)                          :: i_TOT     ! number of packets to be injected
REAL(KIND=8),DIMENSION(1:n_TOT)             :: J0        ! mean intensity in shell
REAL(KIND=8),INTENT(IN)                     :: kappa_M   ! mass scattering opacity coefficient
REAL(KIND=8)                                :: LRD       ! linear random deviate on [0,1]
REAL(KIND=8)                                :: mu_J0     ! shell-mean of J0
INTEGER                                     :: n,nn      ! dummy ID of shell
INTEGER,INTENT(IN)                          :: n_TOT     ! number of shells
REAL(KIND=8)                                :: phi       ! azimuthal angle of packet direction
REAL(KIND=8),DIMENSION(1:3)                 :: r         ! position of packet (vector; in cm)
REAL(KIND=8)                                :: r12_SQD   ! squared distance from spine (in cm^2)
REAL(KIND=8),INTENT(IN),DIMENSION(1:n_TOT)  :: rho       ! mean density in shell (in g/cm^2)
REAL(KIND=8)                                :: s         ! distance to next boundary intercept (in cm)
REAL(KIND=8)                                :: s_LIM     ! maximum distance lumpack can travel
REAL(KIND=8)                                :: sigma_J0  ! shell-SD of J0
REAL(KIND=8)                                :: sintheta  ! sine of polar angle of packet direction
REAL(KIND=8)                                :: tau       ! optical depth
REAL(KIND=8),INTENT(IN),DIMENSION(0:n_TOT)  :: w         ! shell boundary radius (in cm)
REAL(KIND=8),INTENT(IN)                     :: W_B       ! boundary radius of the filament (in cm)
REAL(KIND=8)                                :: W_SQD     ! squared distance radius of filament (in cm^2)
                                                         ! DIAGNOSTIC CHECKS
INTEGER                                     :: dir,eINNUM,eSCNUM,tauNUM
REAL(KIND=8),DIMENSION(1:3)                 :: eINBAR,eSCBAR,eINSIG,eSCSIG
REAL(KIND=8)                                :: tauBAR,tauSIG

W_SQD=W_B**2                                             ! compute W_B squared
J0=0.                                                    ! set mean intensities to zero
eINBAR=0.; eINSIG=0.; eINNUM=0                           !D
eSCBAR=0.; eSCSIG=0.; eSCNUM=0                           !D
tauBAR=0.; tauSIG=0.; tauNUM=0                           !D
!WRITE (6,"(6X,'i:',8X,'e1:',5X,'e2:',6X,'e:',8X,'LRD:',4X,'tau:',9X,'r1:',5X,'r2:',6X,'r:',9X,'wn:',6X,'s:',6X,'MODE:')")
DO i=1,i_TOT                                             ! start loop over packets
  r(1)=-W_B;  r(2)=0.;  r(3)=0.                          !   position packet on boundary of filament
  r12_SQD=W_SQD                                          !   compute squared distance from spine
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  e(1)=SQRT(LRD)                                         !   compute e(1)
  sintheta=SQRT(1.-LRD)                                  !   compute SIN(theta)
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  phi=6.2831853*LRD                                      !   compute phi
  e(2)=sintheta*COS(phi)                                 !   compute e(2)
  e(3)=sintheta*SIN(phi)                                 !   compute e(3)
  eINBAR(1)=eINBAR(1)+e(1)                               !D
  eINBAR(2:3)=eINBAR(2:3)+ABS(e(2:3))                    !D
  eINSIG(1:3)=eINSIG(1:3)+e(1:3)**2                      !D
  eINNUM=eINNUM+1                                        !D
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  tau=-LOG(LRD)                                          !   compute optical-depth
  tauBAR=tauBAR+tau                                      !D
  tauSIG=tauSIG+tau**2                                   !D
  tauNUM=tauNUM+1                                        !D
  n=n_TOT                                                !   set shell ID to n_TOT (outermost shell)
  IF (i==0) WRITE (6,"(I8,3X,3F8.4,4X,2F8.4)") i,e(1:2),SQRT(e(1)**2+e(2)**2+e(3)**2),LRD,(tau/W_B)
  DO WHILE (n<=n_TOT)                                    !   keep going until packet exits filament
    nn=n                                                 !     record ID of shell being entered
    e12_SQD=e(1)**2+e(2)**2                              !     compute e_x^2+e_y^2
    alpha=((r(1)*e(1))+(r(2)*e(2)))/e12_SQD              !     compute 'outwardness'
    IF (alpha>0.) THEN                                   !     [IF] travelling outward, [THEN]
      s=-alpha+SQRT(alpha**2+((eta(n)-r12_SQD)/e12_SQD)) !       compute path
      n=n+1                                              !       increase shell ID
      dir=+1                                             !       record direction relative to shells
    ELSE                                                 !     [ELSE] travelling inward, so
      beta=alpha**2+((eta(n-1)-r12_SQD)/e12_SQD)         !       compute beta, and
      IF (beta>0.) THEN                                  !       [IF] beta>0, [THEN] hits inner shell
        s=-alpha-SQRT(beta)                              !         compute path
        n=n-1                                            !         decrease shell ID
        dir=-1                                           !       record direction relative to shells
      ELSE                                               !       [ELSE] traverses shell, so
        s=-alpha+SQRT(alpha**2+(eta(n)-r12_SQD)/e12_SQD) !         compute path
        n=n+1                                            !         increase shell ID
        dir=0                                            !         record direction relative to shells
      ENDIF                                              !       [ENDIF] inward cases done
    ENDIF                                                !     [ENDIF] all cases done
    s_LIM=tau/(rho(nn)*kappa_M)
    IF (s_LIM<s) THEN                                      !     [IF] range of packet too small
      s=s_LIM                                             !       set s to tau
      n=nn                                               !       record that still in same shell
      dir=2                                              !       record direction relative to shells
    ELSE                                                 !     [ELSE] 
      tau=tau-s*rho(nn)*kappa_M                                          !       decrease range as appropriate
    ENDIF                                                !     [ENDIF]
    r(1:2)=r(1:2)+(s*e(1:2))                             !     advance position of packet
    r12_SQD=(r(1)**2)+(r(2)**2)                          !     compute squared distance from spine
    IF (dir==2) THEN                                     !     [IF] scattering adjustments needed
        CALL RANDOM_NUMBER(LRD)                          !       generate linear random deviate on [0,1]
        e(1)=2.*LRD-1.                                   !       compute e(1)
        sintheta=SQRT(1.-e(1)**2)                        !       compute SIN(theta)
        CALL RANDOM_NUMBER(LRD)                          !       generate linear random deviate on [0,1]
        phi=6.28311855*LRD                               !       compute phi
        e(2)=sintheta*COS(phi)                           !       compute e(2)
        e(3)=sintheta*SIN(phi)                           !       compute e(3)
        eSCBAR(1:3)=eSCBAR(1:3)+ABS(e(1:3))              !D
        eSCSIG(1:3)=eSCSIG(1:3)+e(1:3)**2                !D
        eSCNUM=eSCNUM+1                                  !D
        CALL RANDOM_NUMBER(LRD)                          !       generate linear random deviate on [0,1]
        tau=-LOG(LRD)                                    !       compute optical-depth
        tauBAR=tauBAR+tau                                !D
        tauSIG=tauSIG+tau**2                             !D
        tauNUM=tauNUM+1                                  !D
    ENDIF                                                !     [ENDIF] scattering adjustments completed
    IF (i==0) THEN                                       !D
      IF (dir==-1) WRITE (6,"(59X,3F8.4,4X,2F8.4,5X,'INWARD')") &
      &(r(1:2)/W_B),(SQRT(r12_SQD)/W_B),(w(n)/W_B),(s/W_B)
      IF (dir==0) WRITE (6,"(59X,3F8.4,4X,2F8.4,5X,'ACROSS')") &
      &(r(1:2)/W_B),(SQRT(r12_SQD)/W_B),(w(n-1)/W_B),(s/W_B)
      IF (dir==+1) WRITE (6,"(59X,3F8.4,4X,2F8.4,4X,'OUTWARD')") &
      &(r(1:2)/W_B),(SQRT(r12_SQD)/W_B),(w(n-1)/W_B),(s/W_B)
      IF (dir==+2) WRITE (6,"(11X,3F8.4,4X,2F8.4,4X,3F8.4,4X,2F8.4,4X,'SCATTER')") e(1:2),&
      &SQRT(e(1)**2+e(2)**2+e(3)**2),LRD,(tau/W_B),(r(1:2)/W_B),(SQRT(r12_SQD)/W_B),(w(n)/W_B),(s/W_B)
    ENDIF                                                !D
    J0(nn)=J0(nn)+s                                      !     increment mean intensity
  ENDDO                                                  !   packet exits filament
ENDDO                                                    ! end loop over packets
WRITE (6,"(/,3X,'TEST CASE 2: UNIFORM MASS SCATTERING OPACITY,  n_TOT =',I4,/)") n_TOT
alpha=1./DBLE(eINNUM)                                    ! use alpha as normalisation coefficient
eINBAR(1:3)=alpha*eINBAR(1:3)                            ! compute mean of injection directions
eINSIG(1:3)=SQRT(alpha*eINSIG(1:3)-eINBAR(1:3)**2)       ! compute StDev of injection directions
WRITE (6,"(3X,'MEAN and StDev OF ENTRY DIRECTION:',4X,3(3X,2F8.5))") &
&eINBAR(1),eINSIG(1),eINBAR(2),eINSIG(2),eINBAR(3),eINSIG(3)
WRITE (6,"(27X,'SHOULD BE:',8X,'0.66667 0.23570    0.42441 0.26433    0.42441 0.26433',/)")
alpha=1./DBLE(eSCNUM)                                    ! use alpha as normalisation coefficient
eSCBAR(1:3)=alpha*eSCBAR(1:3)                            ! compute mean of scattering directions
eSCSIG(1:3)=SQRT(alpha*eSCSIG(1:3)-eSCBAR(1:3)**2)       ! compute StDev of scattering directions
WRITE (6,"(3X,'MEAN and StDev OF SCATTERED DIRECTION:',3(3X,2F8.5))") &
     &eSCBAR(1),eSCSIG(1),eSCBAR(2),eSCSIG(2),eSCBAR(3),eSCSIG(3)
WRITE (6,"(31X,'SHOULD BE:',4X,'0.50000 0.28868    0.50000 0.28868    0.50000 0.28868',/)")
alpha=1./DBLE(tauNUM)                                    ! use alpha as normalisation coefficient
tauBAR=alpha*tauBAR                                      ! compute mean of optical depths
tauSIG=SQRT(alpha*tauSIG-tauBAR**2)                      ! compute StDev of oprical depths
WRITE (6,"(3X,'MEAN AND StDev OF OPTICAL DEPTH:',9X,2F8.5)") tauBAR,tauSIG
WRITE (6,"(25X,'SHOULD BE:',10X,'1.00000 1.00000',/)")
WRITE (6,"(3X,'i_TOT =',I8,'; eINNUM =',I8,'; eSCNUM =',I8,'; eINNUM+eSCNUM =',I8,'; tauNUM =',I8,'; paths =',F8.3)") &
&i_TOT,eINNUM,eSCNUM,(eINNUM+eSCNUM),tauNUM,DBLE(eINNUM+eSCNUM)/DBLE(i_TOT)
alpha=W_B/(2.*DBLE(i_TOT))                               ! use alpha as normalisation coefficient
mu_J0=0.                                                 ! set mu_J0 to zero
sigma_J0=0.                                              ! set sigma_J0 to zero
DO n=1,n_TOT                                             ! start loop over shells
  J0(n)=J0(n)*alpha/(eta(n)-eta(n-1))                    !   normalise J0
  mu_J0=mu_j0+J0(n)                                      !   increment mu_J0 accumulator
  sigma_J0=sigma_J0+(J0(n)**2)                           !   increment sigma_J0 accumulator
ENDDO                                                    ! end loop over shells
mu_J0=mu_J0/DBLE(n_TOT)                                  ! compute mu_J0
sigma_J0=SQRT((sigma_J0/DBLE(n_TOT))-(mu_J0**2))         ! compute sigma_J0
WRITE (6,"(/,3X,'SHELL MEAN INTENSITIES (J0):')")
WRITE (6,"(3X,'J0:',4X,5F9.5,6X,5F9.5)") J0(1:5),J0(n_TOT-4:n_TOT)
WRITE (6,"(3X,'Shell-mean of angle-mean intensity:',3X,F10.7,17X,'(mu_J0)')") mu_J0
WRITE (6,"(3X,'Shell-SD of angle-mean intensity:',5X,F10.7,17X,'(sigma_J0)',/)") sigma_J0

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RadTrans_InjectAndTrack2
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



