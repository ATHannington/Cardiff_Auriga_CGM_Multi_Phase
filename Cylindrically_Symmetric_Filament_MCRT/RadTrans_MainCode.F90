
! RadTrans_MainCode.F90
! =====================

! Last updated (Ant): 2019.Mar.13 (line 1879)
! Desktop: gfortran -O3 -o RUN RadTrans_MainCode.F90 -L/star/lib `pgplot_link`
! Laptop:  gfortran -O3 -o RUN RadTrans_MainCode.F90 -L/opt/local/lib -lpgplot -lX11
!          gfortran -O0 -fbounds-check -o RUN RadTrans_MainCode.F90 -L/opt/local/lib -lpgplot -lX11
! Run with ./RUN

!
!To complie:            gfortran -o2 -o run RadTrans_MainCode.F90
! or, for debugging:    gfortran -o2 -g -fbounds-check -fbacktrace -o run RadTrans_MainCode.F90
!
!To RUN:
!   ./run

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

! SubRoutines:
!   RT_DustPropertiesFromDraine
!   RT_PlotDustProperties
!   RT_Temperatures
!   RT_EmProbs_DMBB
!   RT_LumPack_BB
!   RT_LumPack_MB
!   RT_LumPack_DM
!   RT_Cyl1D_LinearShellSpacing
!   RT_Cyl1D_InjectIsotropic
!   RT_ReDirectIsotropic
!   RT_Cyl1D_InjectIsotropicAndTrack_ZeroOpacity []
!   RT_Cyl1D_InjectIsotropicAndTrack_UniformScatteringOpacity
!   RT_SchusterDensities
!   RT_Cyl1D_InjectIsotropicAndTrack_SchusterScatteringOpacity
!   ++++++++++++++++
!   RT_Cyl1D_GlobalParameters
!
! Timing:
!   TBD
!   TBD
!
!Notes:
! RT_Cyl1DSchuster_DetailedBalance not working!!
!
! All data to be written for Python:
!   Must allow for csv formatting and be SQUARE/RECTANGULAR
!   Must be outputted in Fx.y style (e.g. F10.3). Python cannot handle Fortran
!   Scientific notation
!
!
!************************
PROGRAM RadTrans_MainCode
!************************
USE CONSTANTS
USE PHYSICAL_CONSTANTS

IMPLICIT NONE                                            ! [] DECLARATIONS:                                                         ! Configuration (CF)

                                                         !  CONFIGURATION [CF]
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: cfL       ! line-luminosity absorbed by cell
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: CFmu      ! line-density of cell (g/cm)
REAL(KIND=8)                                :: CFmuTOT   ! line-density of filment (g/cm)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: CFrho     ! density in cell (g/cm^3)
REAL(KIND=8)                                :: CFsig     ! column through centre (g/cm^2)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: cfT       ! temperature in cell (K)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: CFw       ! cell outer boundary radius (cm)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: CFw2      ! squared cell outer boundary radius (cm^2)

                                                         ! [] LUMINOSITY PACKETS (LP)
REAL(KIND=8),DIMENSION(1:3)                 :: LPe       ! direction of luminosity packet
INTEGER                                     :: LPl       ! ID of luminosity packet's wavelength
INTEGER                                     :: LPp       ! dummy ID of luminosity packet
REAL(KIND=8),DIMENSION(1:3)                 :: LPr       ! position of luminosity packet
REAL(KIND=8)                                :: LPtau     ! opical depth of luminosity packet

                                                       ! [] TEMPERATURES (TE)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: teLMmb    ! MB luminosities per unit mass (cm^2/s^3)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: teLMTdm   ! DM lums per unit mass and unit temprtre (cm^2/s^3K)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: teT       ! discrete temperatures
REAL(KIND=8)                                :: teTcmb    ! temperature of cosmic microwave background
                                                         ! [] WAVELENGTHS (WL)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: WLalb     ! albedos at discrete wavelengths
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: WLchi     ! extinction opacities at dscrt wvlngths (cm^2/g)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: WLdlam    ! discrete wavelength intervals (in microns)
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: WLlam     ! discrete wavelengths (in microns)
INTEGER                                     :: WLlTOT    ! number of discrete wavelengths
REAL(KIND=8),DIMENSION(:),ALLOCATABLE       :: WLstore   ! array for re-scoping other WL arrays
INTEGER                                     :: WLl       ! dummy ID of discrete wavelength
                                                         ! [] PROBABILITIES DEPENDING ON lambda AND T (WT)
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: WTlBBlo   ! ID of longest wavelength with WTpBB(ID,k)<=(l-1)/lTOT
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: WTlBBup   ! ID of shortest wavelength with WTpBB(ID,k)>=l/lTOT
REAL(KIND=8),DIMENSION(:,:),ALLOCATABLE     :: WTpBB     ! BB emission probabilities
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: WTlMBlo   ! ID of longest wavelength with WTpMB(ID,k)<=(l-1)/l_TOT
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: WTlMBup   ! ID of shortest wavelength with WTpMB(ID,k)>=l/l_TOT
REAL(KIND=8),DIMENSION(:,:),ALLOCATABLE     :: WTpMB     ! MB emission probabilities
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: WTlDMlo   ! ID of longest wavelength with WTpDM(ID,k)<=(l-1)/l_TOT
INTEGER,DIMENSION(:,:),ALLOCATABLE          :: WTlDMup   ! ID of shortest wavelength with WTpDM(ID,k)>=l/l_TOT
REAL(KIND=8),DIMENSION(:,:),ALLOCATABLE     :: WTpDM     ! DM emission probabilities
                                                         ! [] RADIATION FIELD
REAL(KIND=8),DIMENSION(:,:),ALLOCATABLE     :: RFjLAM    ! mean intensity in wavelength interval in cell

Character(len=50)                           :: DustPropertiesFilename = &
                                              & "DustProperties.csv"
Integer*4                                   :: readcheck
Integer*4                                   :: i



PRINT*,""
print*,"***+++***"
print*,"[@RadTrans_MainCode]: Program Start!"
print*,"***+++***"
PRINT*,""

print*,
Print*,"Selected Geometry:"
print*,trim(CFgeom)                                      ! Write Selected geometry to screen

!! SANITY CHECK: Does the selected temperature make sense? !!

IF (BGkGO.ge.BGkBB) then
    print*,"WARNING [@RadTrans_MainCode]:  BGkGO cfLgo temperature .ge. background temp (BGkBB)!"
    print*,"  BGkGO MUST be less than BGkBB!"
endif

If (BGkBB .ge. int(ceiling(dble(TEkTOT)*0.99d0))) THEN
  print*,"WARNING [@RadTrans_MainCode]:  BGkBB background RF temp >=99% of maximum computation temp (TEkTOT)!"
else if (BGkBB .eq. TEkTOT) THEN
  print*,"WARNING [@RadTrans_MainCode]:  BGkBB background RF temp at 100% of maximum computation temp (TEkTOT)!"
  print*,"**Program Failure likely**!!"
else if (BGkBB .le. int(ceiling(dble(TEkTOT)*0.01d0))) THEN
  print*,"WARNING [@RadTrans_MainCode]:  BGkBB background RF temp <=1% of maximum computation temp (TEkTOT)! &
  & This is near computational minimum!"
else if (BGkBB .eq. 0) THEN
  print*,"WARNING [@RadTrans_MainCode]:  BGkBB background RF temp at 0% of maximum computation temp (TEkTOT)! &
  & This is at computational minimum!"
  print*,"**Program Failure likely**!!"
ENDIF

                                                         ! [] ALLOCATIONS 1
ALLOCATE (cfL(1:CFcTOT))                                 ! allocate cfL array
ALLOCATE (CFmu(1:CFcTOT))                                ! allocate CFmu array
ALLOCATE (CFrho(1:CFcTOT))                               ! allocate CFrho array
ALLOCATE (cfT(1:CFcTOT))                                 ! allocate cfT array
ALLOCATE (CFw(0:CFcTOT))                                 ! allocate CFw array
ALLOCATE (CFw2(0:CFcTOT))                                ! allocate CFw2 array
ALLOCATE (WLlam(1:1000))                                 ! temporary ............
ALLOCATE (WLdlam(1:1000))                                ! ... storage ..........
ALLOCATE (WLchi(1:1000))                                 ! ....... for dust .....
ALLOCATE (WLalb(1:1000))                                 ! ........... properties
ALLOCATE (teT(0:TEkTOT))                                 ! allocate teT array
ALLOCATE (teLMmb(0:TEkTOT))                              ! allocate teLMmb array
ALLOCATE (teLMTdm(0:TEkTOT))                             ! allocate teLMTdm array
ALLOCATE (WTlBBlo(1:PRnTOT,0:TEkTOT))                    ! allocate WTlBBlo array
ALLOCATE (WTlBBup(1:PRnTOT,0:TEkTOT))                    ! allocate WTlBBup array
ALLOCATE (WTlMBlo(1:PRnTOT,0:TEkTOT))                    ! allocate WTlMBlo array
ALLOCATE (WTlMBup(1:PRnTOT,0:TEkTOT))                    ! allocate WTlMBup array
ALLOCATE (WTlDMlo(1:PRnTOT,0:TEkTOT))                    ! allocate WTlDMlo array
ALLOCATE (WTlDMup(1:PRnTOT,0:TEkTOT))                    ! allocate WTlDMup array

IF (DGsource=='Draine') &
& CALL RT_DustPropertiesFromDraine(WLlTOT,WLlam,WLdlam,WLchi,WLalb)

                                                         ! [] RESCOPE DUST ARRAYS
ALLOCATE (WLstore(1:WLlTOT))                             ! rescope ..............
WLstore(1:WLlTOT)=WLlam(1:WLlTOT)                        ! ......................
DEALLOCATE (WLlam)                                       ! ......................
ALLOCATE (WLlam(1:WLlTOT))                               ! ......................
WLlam(1:WLlTOT)=WLstore(1:WLlTOT)                        ! ......................
WLstore(1:WLlTOT)=WLdlam(1:WLlTOT)                       ! .... the dust ........
DEALLOCATE (WLdlam)                                      ! ......................
ALLOCATE (WLdlam(1:WLlTOT))                              ! ......................
WLdlam(1:WLlTOT)=WLstore(1:WLlTOT)                       ! ......................
WLstore(1:WLlTOT)=WLchi(1:WLlTOT)                        ! ......................
DEALLOCATE (WLchi)                                       ! ......................
ALLOCATE (WLchi(1:WLlTOT))                               ! ......... property ...
WLchi(1:WLlTOT)=WLstore(1:WLlTOT)                        ! ......................
WLstore(1:WLlTOT)=WLalb(1:WLlTOT)                        ! ......................
DEALLOCATE (WLalb)                                       ! ......................
ALLOCATE (WLalb(1:WLlTOT))                               ! ......................
WLalb(1:WLlTOT)=WLstore(1:WLlTOT)                        ! ......................
DEALLOCATE (WLstore)                                     ! ............... arrays

                                                         ! [] ALLOCATIONS 2
ALLOCATE (WTpBB(0:WLlTOT,0:TEkTOT))                      ! allocate WTpBB array
ALLOCATE (WTpMB(0:WLlTOT,0:TEkTOT))                      ! allocate WTpMB array
ALLOCATE (WTpDM(0:WLlTOT,0:TEkTOT))                      ! allocate WTpDM array
ALLOCATE (RFjLAM(1:WLlTOT,1:CFcTOT))                     ! allocate RFjLAM array

IF (WLplot==1) then
    !CALL RT_PlotDustProperties(WLlTOT,WLlam,WLchi,WLalb)
    OPEN(1,file=trim(adjustl(DustPropertiesFilename)),&
    & iostat=readcheck)
    WRITE(1,"(A3,1x,A3,1x,A3)") (/"lam","chi","alb"/)
    do i = 1, WLlTOT
        WRITE(1,"(F10.3,1x,F10.3,1x,F10.3)") &
        &(/WLlam(i),WLchi(i),WLalb(i)/)
    enddo

    CLOSE(1)
ENDIF

!!!!!       Main Subroutine Calls:          !!!!!
CALL RT_Temperatures(teT)

CALL RT_EmProbs_DMBB(teT,WLlTOT,WLlam,WLdlam,WLchi,WLalb,&
&WTpBB,WTlBBlo,WTlBBup,WTpMB,WTlMBlo,WTlMBup,teLMmb,WTpDM,WTlDMlo,WTlDMup,teLMTdm)

CALL RT_Cyl1D_LinearShellSpacing(CFw,CFw2)

CALL RT_Cyl1D_SchusterDensities(CFw,CFrho,CFmu,CFmuTOT,CFsig)


!!!!!           TESTS:                  !!!!!!
! 
! CALL RT_Cyl1D_InjectIsotropicAndTrack_ZeroOpacity(CFw,CFw2)
!
! CALL RT_Cyl1D_InjectIsotropicAndTrack_UniformScatteringOpacity(CFw,CFw2)
!
! CALL RT_Cyl1D_InjectIsotropicAndTrack_SchusterScatteringOpacity(CFw,CFw2,CFrho,CFsig)

CALL RT_Cyl1DSchuster_DetailedBalance(CFw, &
&CFw2,CFrho,CFmu,CFsig,cfT,cfL,teT,WLlTOT,WLlam, &
&WLdlam,WLchi,WLalb,WTpBB,WTlBBlo,WTlBBup,WTpMB,WTlMBlo,&
&WTlMBup,teLMmb,WTpDM,WTlDMlo,WTlDMup,teLMTdm, &
&RFjLAM)



PRINT*,""
print*,"***+++***"
print*,"[@RadTrans_MainCode]: Program Complete!"
print*,"***+++***"
PRINT*,""
!****************************
END PROGRAM RadTrans_MainCode
!****************************
