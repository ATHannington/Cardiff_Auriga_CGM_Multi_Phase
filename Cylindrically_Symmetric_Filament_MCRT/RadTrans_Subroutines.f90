!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_DustPropertiesFromDraine(WLlTOT,WLlam,WLdlam,WLchi,WLalb)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine imports dust properties from one of the Draine (2003) tabulated models, and
! interpolates to find regularly spaced values in the range specified by the IDs [DGlMIN,DGlMAX].

! It is given:
!   the name of the Draine model file                     (DGmodel);
!   the line number for the longest wavelength needed     (DGlMIN);
!   the line number for the shortest wavelength needed    (DGlMAX);
!   the spacing parameter                                 (WLdelta);
!   the weight for the slope-change                       (WLdcl);
!   and a flag to print out some optical properties       (WLprint).

! It reads in the data from the Draine model file, viz.
!   wavelengths                                           (DGlam(1:DGlMAX));
!   extinction opacities                                  (DGchi(1:DGlMAX));
!   albedos                                               (DGalb(1:DGlMAX));
!   and mean scattering cosines                           (DGmsc(1:DGlMAX)).

! It returns the interpolated:
!   number of discrete wavelengths                        (WLlTOT);
!   wavelengths                                           (WLlam(1:1500));
!   wavelength intervals                                  (WLdlam(1:1500));
!   extinction opacities                                  (WLchi(1:1500));
!   and albedos                                           (WLalb(1:1500)).

! These have been adjusted according to the Irving Approximation, so the
! scattering can then be treated as isotropic. The array receiving these
! interpolated values is immediately rescoped, outside the subroutine, to
! save memory.
USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! [] DECLARATIONS
REAL(KIND=8),DIMENSION(1:DGlMAX)            :: DGlam     ! imported wavelengths (in microns)
REAL(KIND=8),DIMENSION(1:DGlMAX)            :: DGchi     ! imported extinction opacities (in cm^2/g)
REAL(KIND=8),DIMENSION(1:DGlMAX)            :: DGalb     ! imported albedos
REAL(KIND=8),DIMENSION(1:DGlMAX)            :: DGmsc     ! imported mean scattering cosines
INTEGER,     INTENT(OUT)                    :: WLlTOT    ! number of exported discrete wavelengths
REAL(KIND=8),INTENT(OUT),DIMENSION(1:5000)  :: WLlam     ! exported discrete wavelengths
REAL(KIND=8),INTENT(OUT),DIMENSION(1:5000)  :: WLdlam    ! exported wavelength intervals
REAL(KIND=8),INTENT(OUT),DIMENSION(1:5000)  :: WLchi     ! exported extinction opacities at desceret wavelengths
REAL(KIND=8),INTENT(OUT),DIMENSION(1:5000)  :: WLalb     ! exported albedos at discrete wavelengths
                                                         ! [] DRONES
REAL(KIND=8)                                :: dchidWLl  ! slope, dchi/dl, for extrapolation
REAL(KIND=8)                                :: ddelta    ! increment to logarithmic displacement
REAL(KIND=8)                                :: deltaLAM  ! contribution to spacing from lambda
REAL(KIND=8)                                :: deltaCHI  ! contribution to spacing from chi
REAL(KIND=8)                                :: deltaALB  ! contribution to spacing from albedo
REAL(KIND=8)                                :: deltaDCL  ! contribution to spacing from gradient
REAL(KIND=8),DIMENSION(1:DGlMAX)            :: DGdcl     ! logarithmic gradient
REAL(KIND=8)                                :: DGdclMAX
REAL(KIND=8)                                :: DGdclMIN
INTEGER                                     :: DGl       ! dummy ID for imported wavelengths
REAL(KIND=8)                                :: dlamdWLl  ! slope, dlambda/dl, for extrapolation
REAL(KIND=8)                                :: factor    ! factor in the Irving approximation
REAL(KIND=8)                                :: kappa     ! dummy absorption opacity
REAL(KIND=8)                                :: mscSQD    ! dummy mean squared scattering cosine
REAL(KIND=8)                                :: NEWlam    ! new trial wavelength
REAL(KIND=8)                                :: NEWchi    ! new trial extinction opacity
REAL(KIND=8)                                :: NEWalb    ! new trial albedo
REAL(KIND=8)                                :: NEWdcl    ! new trial albedo
REAL(KIND=8)                                :: NEWdelta
REAL(KIND=8)                                :: OLDlam    ! old trial wavelength
REAL(KIND=8)                                :: OLDchi    ! old trial extinction opacity
REAL(KIND=8)                                :: OLDalb    ! old trial albedo
REAL(KIND=8)                                :: OLDdcl    ! old trial albedo
REAL(KIND=8)                                :: OLDdelta
CHARACTER(LEN=100)                          :: rhubarb   ! dummy character string
REAL(KIND=8)                                :: store     ! real for reversing order
INTEGER                                     :: WLl       ! dummy ID of exported wavelength
REAL(KIND=8)                                :: Wlo       ! weight of lower (longer) wavelength
REAL(KIND=8)                                :: Wup       ! weight of upper (shorter) wavelength


                                                         ! [] READ IN TABULATED DATA
OPEN (UNIT=5,FILE=DGmodel,STATUS='old',ACTION='read')    ! open data file
DO DGl=-DGlMIN,DGlMAX                                    ! start loop over input file
  IF (DGl<1) THEN                                        !   [IF] in header-text part, [THEN]
    READ(5,*) rhubarb                                    !     [READ] into dummy character string
  ELSE                                                   !   [ELSE]
    READ (5,"(E11.5,F7.4,F8.4,E10.3,E10.3,F8.5)")       &!     [READ] ........
    &DGlam(DGl),DGalb(DGl),DGmsc(DGl),DGchi(DGl),       &!     .... into .....
    &kappa,mscSQD                                        !     ........ arrays
  ENDIF                                                  !   [ENDIF]
ENDDO                                                    ! end loop over input file
IF (WLprint==1) THEN                                     ! [IF] sanctioned, [THEN]
  WRITE (6,"(/,3X,'DUST PROPERTIES FROM DRAINE:')")      !   print out some dust properties
  WRITE (6,"(3X,'DGl:',3X,5I10,6X,5I10)") 1,2,3,4,5,DGlMAX-4,DGlMAX-3,DGlMAX-2,DGlMAX-1,DGlMAX
  WRITE (6,"(3X,'lam:',3X,5F10.1,6X,5F10.5)") DGlam(1:5),DGlam(DGlMAX-4:DGlMAX)
  WRITE (6,"(3X,'chi:',3X,5E10.3,6X,5E10.3)") DGchi(1:5),DGchi(DGlMAX-4:DGlMAX)
  WRITE (6,"(3X,'alb:',3X,5F10.5,6X,5F10.5)") DGalb(1:5),DGalb(DGlMAX-4:DGlMAX)
  WRITE (6,"(3X,'msc:',3X,5F10.5,6X,5F10.5)") DGmsc(1:5),DGmsc(DGlMAX-4:DGlMAX)
ENDIF                                                    ! [ENDIF]

                                                         ! [] IMPLEMENT IRVING APPROXIMATION
DO DGl=1,DGlMAX                                          ! start loop over input file
  factor=1.-(DGmsc(DGl)*DGalb(DGl))                      !   compute factor
  IF (DGmodel=='draine_rv3.1.dat')                      &!   adjust extinction opacity ...
  &             DGchi(DGl)=factor*DGchi(DGl)/(1.870E-26) !   ..... and normalise for R=3.1
  IF (DGmodel=='draine_rv4.0.dat')                      &!   adjust extinction opacity ...
  &             DGchi(DGl)=factor*DGchi(DGl)/(1.969E-26) !   ..... and normalise for R=4.0
  IF (DGmodel=='draine_rv5.5.dat')                      &!   adjust extinction opacity ...
  &             DGchi(DGl)=factor*DGchi(DGl)/(2.199E-26) !   ..... and normalise for R=5.5
  DGalb(DGl)=(1.-DGmsc(DGl))*DGalb(DGl)/factor           !   adjust albedo
  DGmsc(DGl)=0.                                          !   set mean scattering cosine to zero
ENDDO                                                    ! end loop over input file
IF (WLprint==1) THEN                                     ! [IF] sanctioned, [THEN]
  WRITE (6,"(/,3X,'DUST PROPERTIES POST IRVING:')")      !   print out some dust properties
  WRITE (6,"(3X,'DGl:',3X,5I10,6X,5I10)") 1,2,3,4,5,DGlMAX-4,DGlMAX-3,DGlMAX-2,DGlMAX-1,DGlMAX
  WRITE (6,"(3X,'lam:',3X,5F10.1,6X,5F10.5)") DGlam(1:5),DGlam(DGlMAX-4:DGlMAX)
  WRITE (6,"(3X,'chi:',3X,5E10.3,6X,5E10.3)") DGchi(1:5),DGchi(DGlMAX-4:DGlMAX)
  WRITE (6,"(3X,'alb:',3X,5F10.5,6X,5F10.5)") DGalb(1:5),DGalb(DGlMAX-4:DGlMAX)
  WRITE (6,"(3X,'msc:',3X,5F10.5,6X,5F10.5)") DGmsc(1:5),DGmsc(DGlMAX-4:DGlMAX)
ENDIF                                                    ! [ENDIF]

                                                         ! [] COMPUTE LOGARITHIC GRADIENTS
DGdcl(1)=(DGlam(1)+DGlam(2))*(DGchi(1)-DGchi(2))/       &! compute dcl for first ...
       &((DGlam(1)-DGlam(2))*(DGchi(1)+DGchi(2)))        ! ..... imported wavelength
DO DGL=2,DGlMAX-1                                        ! start loop over imported wavelengths
   DGdcl(DGl)=(DGlam(DGl-1)+DGlam(DGl+1))*              &! compute dcl ............
             &(DGchi(DGl-1)-DGchi(DGl+1))/              &! ... for intermediate ...
            &((DGlam(DGl-1)-DGlam(DGl+1))*              &! ........ imported ......
             &(DGchi(DGl-1)+DGchi(DGl+1)))               ! ............ wavelengths
ENDDO                                                    ! end loop over imported wavelengths
DGdcl(DGlMAX)=(DGlam(DGlMAX-1)+DGlam(DGlMAX))*          &! compute dcl ...........
             &(DGchi(DGlMAX-1)-DGchi(DGlMAX))/          &! ... for last ..........
            &((DGlam(DGlMAX-1)-DGlam(DGlMAX))*          &! ........ imported .....
             &(DGchi(DGlMAX-1)+DGchi(DGlMAX)))           ! ............ wavelength

                                                         ! [] INTERPOLATION
OLDlam=DGlam(1)                                          ! set OLDlam to first imported DGlam
OLDchi=DGchi(1)                                          ! set OLDchi to first imported DGchi
OLDalb=DGalb(1)                                          ! set OLDalb to first imported DGalb
OLDdcl=DGdcl(1)                                          ! set OLDdcl to first deriveded DGdcl
NEWdelta=0.                                              ! set NEWdelta to zero
DGl=1                                                    ! set DGl=1 (ID of first imported value)
WLl=1                                                    ! set WLl=1 (ID of first exported value)
IF (WLprint==1) WRITE &
&(6,"(/,'  WLl:',10X,'lam:',9X,'chi:',7X,'alb:',13X,'dlam:',5X,'dchi:',5X,'dalb:',5X,'ddcl:',4X,'delta:')")
DO WHILE (DGl<=DGlMAX)                                   ! iterate until imported wavelengths exhausted
  DO WHILE ((NEWdelta<WLdelta).AND.(DGl<DGlMAX))         !   iterate until NEWdelta big enough
    OLDdelta=NEWdelta                                    !     set OLDdelta to NEWdelta
    DGl=DGl+1                                            !     increment ID of imported point
    deltaLAM=ABS(LOG10(DGlam(DGl)/OLDlam))               !     compute deltaLAM
    deltaCHI=ABS(LOG10(DGchi(DGl)/OLDchi))               !     compute deltaCHI
    deltaALB=ABS(LOG10((0.01+DGalb(DGl))/(0.01+OLDalb))) !     compute deltaALB
    deltaDCL=WLdcl*ABS(DGdcl(DGl)-OLDdcl)                  !     compute deltaDCL
    NEWdelta=deltaLAM+deltaCHI+deltaALB+deltaDCL         !     compute NEWdelta
  ENDDO                                                  !   NEWdelta now big enough
  IF (DGl==DGlMAX) EXIT                                  !   terminate interpolation
  Wlo=(NEWdelta-WLdelta)/NEWdelta                        !   compute weighting of lower limiting values
  Wup=1.-Wlo                                             !   compute weighting of upper limiting values
  NEWlam=Wlo*OLDlam+Wup*DGlam(DGl)                       !   compute wavelength at upper end of interval
  NEWchi=Wlo*OLDchi+Wup*DGchi(DGl)                       !   compute extinction at upper end of interval
  NEWalb=Wlo*OLDalb+Wup*DGalb(DGl)                       !   compute albedo at upper end of interval
  NEWdcl=Wlo*OLDdcl+Wup*DGdcl(DGl)                       !   compute slope at upper end of interval
  WLlam(WLl)=0.5*(OLDlam+NEWlam)                         !   exported wavelength
  WLdlam(WLl)=OLDlam-NEWlam                              !   exported wavelength interval
  WLchi(WLl)=0.5*(OLDchi+NEWchi)                         !   exported extinction
  WLalb(WLl)=0.5*(OLDalb+NEWalb)                         !   exported albedo
  IF ((WLprint==1).AND.(MOD(WLl,10)==0)) THEN            !   [IF] printout sanctioned, and every tenth wavelength
    deltaLAM=ABS(LOG10(NEWlam/OLDlam))                   !     compute deltaLAM
    deltaCHI=ABS(LOG10(NEWchi/OLDchi))                   !     compute deltaCHI
    deltaALB=ABS(LOG10((0.01+NEWalb)/(0.01+OLDalb)))     !     compute deltaALB
    deltaDCL=WLdcl*ABS(NEWdcl-OLDdcl)                      !     compute deltaDCL
    NEWdelta=deltaLAM+deltaCHI+deltaALB+deltaDCL         !     compute NEWdelta
    WRITE (6,"(I6,F14.5,E13.3,F11.5,8X,5F10.5)")WLl,WLlam(WLl),&
    &WLchi(WLl),WLalb(WLl),deltaLAM,deltaCHI,deltaALB,deltaDCL,NEWdelta
  ENDIF                                                  !   [ENDIF] printout accomplished
  WLl=WLl+1                                              !   increment ID of exported wavelength
  DGl=DGl-1                                              !   take a step back
  OLDlam=NEWlam                                          !   update OLDlam
  OLDchi=NEWchi                                          !   update OLDchi
  OLDalb=NEWalb                                          !   update OLDalb
  OLDdcl=NEWdcl                                          !   update OLDdcl
  NEWdelta=0.                                            !   set NEWdelta to zero
ENDDO                                                    ! imported wavelengths exhausted
WLlTOT=WLl-1                                             ! record l_TOT
IF (WLprint==1) THEN                                     ! [IF] sanctioned, [THEN]
  WRITE (6,"(/,3X,'INTERPOLATED DUST PROPERTIES:')")     !   print out some dust properties
  WRITE (6,"(3X,'WLl:',3X,5I10,6X,5I10)") 1,2,3,4,5,WLlTOT-4,WLlTOT-3,WLlTOT-2,WLlTOT-1,WLlTOT
  WRITE (6,"(3X,'lam:',3X,5F10.1,6X,5F10.5)") WLlam(1:5),WLlam(WLlTOT-4:WLlTOT)
  WRITE (6,"(3X,'chi:',3X,5E10.3,6X,5E10.3)") WLchi(1:5),WLchi(WLlTOT-4:WLlTOT)
  WRITE (6,"(3X,'alb:',3X,5F10.5,6X,5F10.5)") WLalb(1:5),WLalb(WLlTOT-4:WLlTOT)
ENDIF                                                    ! [ENDIF]

                                                         ! [] REVERSE ORDER OF WAVELENGTHS
DO WLl=1,WLlTOT/2                                        ! start loop over lower half
  DGl=WLlTOT-WLl+1                                       !   compute ID of wavelength to swap with
  store=WLlam(WLl)                                       !   swap ................
  WLlam(WLl)=WLlam(DGl)                                  !   .... the ............
  WLlam(DGl)=store                                       !   ......... wavelengths
  store=WLdlam(WLl)                                      !   swap .....................
  WLdlam(WLl)=WLdlam(DGl)                                !   ...... the wavelength ....
  WLdlam(DGl)=store                                      !   ................ intervals
  store=WLchi(WLl)                                       !   swap ...............
  WLchi(WLl)=WLchi(DGl)                                  !   .... the ...........
  WLchi(DGl)=store                                       !   ........ extinctions
  store=WLalb(WLl)                                       !   swap ...........
  WLalb(WLl)=WLalb(DGl)                                  !   .... the .......
  WLalb(DGl)=store                                       !   ........ albedos
ENDDO                                                    ! end loop over lower half
IF (WLprint==1) THEN                                     ! [IF] sanctioned, [THEN]
  WRITE (6,"(/,3X,'REVERSED DUST PROPERTIES:')")      !   print out some dust properties
  WRITE (6,"(3X,'WLl:',3X,5I10,6X,5I10)") 1,2,3,4,5,WLlTOT-4,WLlTOT-3,WLlTOT-2,WLlTOT-1,WLlTOT
  WRITE (6,"(3X,'lam:',3X,5F10.5,6X,5F10.1)") WLlam(1:5),WLlam(WLlTOT-4:WLlTOT)
  WRITE (6,"(3X,'chi:',3X,5E10.3,6X,5E10.3)") WLchi(1:5),WLchi(WLlTOT-4:WLlTOT)
  WRITE (6,"(3X,'alb:',3X,5F10.5,6X,5F10.5)") WLalb(1:5),WLalb(WLlTOT-4:WLlTOT)
ENDIF                                                    ! [ENDIF]

                                                         ! [] EXTRAPOLATION
dlamdWLl=WLlam(WLlTOT)/WLlam(WLlTOT-1)                   ! compute ratio between successive wavelengths
dchidWLl=WLchi(WLlTOT)/WLchi(WLlTOT-1)                   ! compute ratio between successive opacities
DO WHILE (WLlam(WLlTOT)<(0.3E+06))                       ! start increasing wavelength
  WLlTOT=WLlTOT+1                                        !   increment l_TOT
  WLlam(WLlTOT)=dlamdWLl*WLlam(WLlTOT-1)                 !   increase lam_OUT
  WLdlam(WLlTOT)=2.*(WLlam(WLlTOT)-WLlam(WLlTOT-1))     &!   compute ....
                                     &-WLdlam(WLlTOT-1)  !   ... dlam_OUT
  WLchi(WLlTOT)=dchidWLl*WLchi(WLlTOT-1)                 !   decrease chi_OUT
  WLalb(WLlTOT)=WLalb(WLlTOT-1)                          !   leave alb_OUT the same
ENDDO                                                    ! stop increasing wavelength
IF (WLprint==1) THEN                                     ! [IF] sanctioned, [THEN]
  WRITE (6,"(/,3X,'EXTRAPOLATED DUST PROPERTIES:')")     !   print out some dust properties
  WRITE (6,"(3X,'WLl:',3X,5I10,6X,5I10)") WLlTOT-9,WLlTOT-8,WLlTOT-7,&
  &WLlTOT-6,WLlTOT-5,WLlTOT-4,WLlTOT-3,WLlTOT-2,WLlTOT-1,WLlTOT
  WRITE (6,"(3X,'lam:',3X,5F10.1,6X,5F10.1)") WLlam(WLlTOT-9:WLlTOT-5),WLlam(WLlTOT-4:WLlTOT)
  WRITE (6,"(3X,'chi:',3X,5E10.3,6X,5E10.3)") WLchi(WLlTOT-9:WLlTOT-5),WLchi(WLlTOT-4:WLlTOT)
  WRITE (6,"(3X,'alb:',3X,5F10.5,6X,5F10.5)") WLalb(WLlTOT-9:WLlTOT-5),WLalb(WLlTOT-4:WLlTOT)
  WRITE (*,*) ' '                                        !   blank line
ENDIF                                                    ! [ENDIF]


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_DustPropertiesFromDraine
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_EmProbs_DMBB(teT,WLlTOT,WLlam,WLdlam,WLchi,WLalb,&
&WTpBB,WTlBBlo,WTlBBup,WTpMB,WTlMBlo,WTlMBup,teLMmb,WTpDM,WTlDMlo,WTlDMup,teLMTdm)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine computes the emission probabilites for blackbody radiation (BB), modified
! blackbody radiation (MB) and temperature-differential modified blackbody radiation (DM).

! It is given:
!   the number of discrete temperatures                          (TEkTOT);
!   the discrete temperatures                                    (teT(0:TEkTOT));
!   the number of discrete wavelengths                           (WLlTOT);
!   the discrete wavelengths                                     (WLlam(1:WLlTOT));
!   the corresponding wavelength intervals                       (WLdlam(1:WLlTOT));
!   the discrete extinction opacities                            (WLchi(1:WLlTOT));
!   the discrete albedos                                         (WLalb(1:WLlTOT));
!   the number of reference probabilities                        (PRnTOT);
!   the number of calls for estimating the probabilities         (WTpack);
!   and a flag to plot the probabilities                         (WTplot).

! It returns:
!   the BB emission probability for [lam(l),dlam(l)] at T=T(k)   (WTpBB(0:WLlTOT,0:TEkTOT));
!   the ID of longest wavelength with WTpBB(ID,k)<=(l-1)/lTOT    (WTlBBlo(1:WLlTOT,0:TEkTOT));
!   the ID of shortest wavelength with WTpBB(ID,k)>=l/lTOT       (WTlBBup(1:WLlTOT,0:TEkTOT)).


!   the MB emission probability for [lam(l),dlam(l)] at T=T(k)   (WTpMB(0:WLlTOT,0:TEkTOT));
!   the ID of longest wavelength with WTpMB(ID,k)<=(l-1)/lTOT    (WTlMBlo(1:WLlTOT,0:TEkTOT));
!   the ID of shortest wavelength with WTpMB(ID,k)>=l/lTOT       (WTlMBup(1:WLlTOT,0:TEkTOT));
!   the MB luminosity per unit mass                              ( (0:TEkTOT));


!   the DM emission probability for [lam(l),dlam(l)] at T=T(k)   (WTpDM(0:WLlTOT,0:TEkTOT));
!   the ID of longest wavelength with WTpDM(ID,k)<=(l-1)/lTOT    (WTlDMlo(1:WLlTOT,0:TEkTOT));
!   the ID of shortest wavelength with WTpDM(ID,k)>=l/lTOT       (WTlDMup(1:WLlTOT,0:TEkTOT));
!   and the DM luminosity per unit mass per unit temperature     (teLMTdm(0:TEkTOT)).

USE CONSTANTS
USE PHYSICAL_CONSTANTS                                           ! [] DECLARATIONS
REAL(KIND=8),INTENT(IN),DIMENSION(0:TEkTOT) :: teT       ! discrete temperatures
INTEGER,     INTENT(IN)                     :: WLlTOT    ! number of discrete wavelengths
REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLlam     ! discrete wavelengths
REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLdlam    ! discrete wavelengths
REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLchi     ! discrete wavelengths
REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLalb     ! discrete wavelengths
REAL(KIND=8),INTENT(OUT),                               &! BB emission probabilities ........
               DIMENSION(0:WLlTOT,0:TEkTOT) :: WTpBB     ! ... for [lam(l),dlam(l)] at T=T(k)
INTEGER,     INTENT(OUT),                               &! ID of longest wavelength with ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlBBlo   ! ........ WTpBB(ID,k)<=(l-1)/l_TOT
INTEGER,     INTENT(OUT),                               &! ID of shortest wavelength ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlBBup   ! ... with WTpBB(ID,k)>=l/l_TOT
REAL(KIND=8),INTENT(OUT),                               &! MB emission probabilities ........
               DIMENSION(0:WLlTOT,0:TEkTOT) :: WTpMB     ! ... for [lam(l),dlam(l)] at T=T(k)
INTEGER,     INTENT(OUT),                               &! ID of longest wavelength with ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlMBlo   ! ........ WTpMB(ID,k)<=(l-1)/l_TOT
INTEGER,     INTENT(OUT),                               &! ID of shortest wavelength ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlMBup   ! ... with WTpMB(ID,k)>=l/l_TOT
REAL(KIND=8),INTENT(OUT),DIMENSION(0:TEkTOT):: teLMmb    ! MB luminosity to mass ratio
REAL(KIND=8),INTENT(OUT),                               &! DM emission probabilities ........
               DIMENSION(0:WLlTOT,0:TEkTOT) :: WTpDM     ! ... for [lam(l),dlam(l)] at T=T(k)
INTEGER,     INTENT(OUT),                               &! ID of longest wavelength with ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlDMlo   ! ........ WTpDM(ID,k)<=(l-1)/l_TOT
INTEGER,     INTENT(OUT),                               &! ID of shortest wavelength ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlDMup   ! ... with WTpDM(ID,k)>=l/l_TOT
REAL(KIND=8),INTENT(OUT),DIMENSION(0:TEkTOT):: teLMTdm   ! DM lum per unit mass per unit temperature
                                                         ! [] DRONES
INTEGER                                     :: PRn       ! dummy ID for reference probabilities
REAL(KIND=8)                                :: PRnTOTinv ! 1/REAL(RFnTOT), spacing
REAL(KIND=8)                                :: PRnTOTrea ! REAL(RFnTOT)
REAL(KIND=8),DIMENSION(0:PRnTOT)            :: PRp       ! reference probabilities
REAL(KIND=8)                                :: TEbbFL    ! check for BB integrated flux
INTEGER                                     :: TEk       ! dummy temperature ID
REAL(KIND=8)                                :: TElamT    ! hc/kT
REAL(KIND=8),DIMENSION(3:6,3:7)             :: WLfMIN    ! factor for minimum wavelength
REAL(KIND=8),DIMENSION(3:6,3:7)             :: WLfMAX    ! factor for maximum wavelength
INTEGER                                     :: WLl       ! dummy wavelength ID
INTEGER                                     :: WLlEM     ! wavelength ID returned by SR:RadTrans_LumPack_BB
REAL(KIND=8)                                :: WLlTOTrea ! REAL(WLlTOT)
REAL(KIND=8)                                :: WTbbEXP   ! exponential in Planck Function
INTEGER,DIMENSION(0:TEkTOT)                 :: WTlMAX    ! ID of longest significant wavelength
INTEGER,DIMENSION(0:TEkTOT)                 :: WTlMIN    ! ID of shortest significant wavelength
INTEGER,DIMENSION(1:WLlTOT)                 :: WTpACC    ! sampling accumulator
REAL(KIND=8)                                :: WTpackINV ! 1/REAL(WTpack)
                                                         ! [] PGPLOT
REAL(KIND=8),DIMENSION(1:WLlTOT)            :: PGx       ! array for log10[lam] (abscissa)
REAL(KIND=8),DIMENSION(1:WLlTOT)            :: PGxx      ! array for log10[lam+dlam] (abscissa)
REAL(KIND=8)                                :: PGxMAX    ! upper limit on log10[lam]
REAL(KIND=8)                                :: PGxMIN    ! lower limit on log10[lam]
REAL(KIND=8),DIMENSION(1:WLlTOT)            :: PGy       ! array for log10[PlanckFn] (ordinate)
REAL(KIND=8)                                :: PGyMAX    ! upper limit on log10[PlanckFn]
REAL(KIND=8)                                :: PGyMIN    ! lower limit on log10[PlanckFn]
REAL(KIND=8),DIMENSION(1:WLlTOT)            :: PGz       ! array for log10[VolEm] (ordinate)
REAL(KIND=8)                                :: PGzMAX    ! upper limit on log10[VolEm]
REAL(KIND=8)                                :: PGzMIN    ! lower limit on log10[VolEm]

!-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
!-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
!ATH Added variables:
Character(len=50) :: BBconstantsFile= "BBconstants.csv",&
               & BBanalyticFile = "BBAnalytic.csv", &
               & BBmcrtFile     = "BBMCRT.csv",     &
               & MBconstantsFile= "MBconstants.csv",&
               & MBanalyticFile = "MBAnalytic.csv", &
               & MBmcrtFile     = "MBMCRT.csv",     &
               & DMconstantsFile= "DMconstants.csv",&
               & DManalyticFile = "DMAnalytic.csv", &
               & DMmcrtFile     = "DMMCRT.csv",     &
               & LambdaDataFile = "WLData.csv"
Integer                                     :: readcheck
Integer                                     :: i
!-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
!-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-

                                                         ! [] INITIALISATION
WTlBBlo=0;   WTlMBlo=0;   WTlDMlo=0                      ! set WTl[BB,MB,DM]lo to zero
WTlBBup=0;   WTlMBup=0;   WTlDMup=0                      ! set WTl[BB,MB,DM]up to zero
WTpBB=0.;    WTpMB=0.;    WTpDM=0.                       ! set WTp[BB,MB,DM] to zero
WTlMIN(1:TEkTOT)=0;       WTlMAX(1:TEkTOT)=0             ! set all WTlMIN and WTlMAX to zero
PRnTOTrea=DBLE(PRnTOT);   PRnTOTinv=1./PRnTOTrea         ! compute REAL(PRnTOT) and 1/REAL(PRnTOT)
DO PRn=0,PRnTOT                                          ! start loop over reference probabilities
  PRp(PRn)=PRnTOTinv*DBLE(PRn)                           !   compute reference probability
ENDDO                                                    ! end loop over reference probabilities

                                                         ! [] TEMPERATURE LOOP
DO TEk=0,TEkTOT                                          ! start loop over discrete temperatures

                                                         !   [] RANGE OF WAVELENGTHS
  TElamT=(0.143878E+05)/teT(TEk)                         !   compute hc/kT(k)
  DO WLl=1,WLlTOT                                        !   start forward loop over wavelengths
    IF (WLlam(WLl)<0.027*TElamT) CYCLE                   !     [IF] wavelength too short, [CYCLE]
    WTlMIN(TEk)=WLl;   EXIT                              !     record WTlMIN and [EXIT]
  ENDDO                                                  !   end forward loop over wavelengths
  DO WLl=WLlTOT,1,-1                                     !   start backwards loop over wavelengths
    IF (WLlam(WLl)>30.00*TElamT) CYCLE                   !     [IF] wavelength too long, [CYCLE]
    WTlMAX(TEk)=WLl;   EXIT                              !     record WTlMAX and [EXIT]
  ENDDO                                                  !   end backwards loop over wavelengths

                                                         !   [] PROBABILITIES
  DO WLl=WTlMIN(TEk),WTlMAX(TEk)                         !   start loop over significant wavelengths
    WTbbEXP=EXP(TElamT/WLlam(WLl))                       !     compute e^[hc/kTminLambda]
    WTpBB(WLl,TEk)=WTpBB(WLl-1,TEk)                     &!     increment BB ...........
           &+(WLdlam(WLl)/(WLlam(WLl)**5*(WTbbEXP-1.)))  !     ... emission probability
    WTpMB(WLl,TEk)=WTpMB(WLl-1,TEk)                     &!     increment .........
              &+(WLchi(WLl)*(1.-WLalb(WLl))*WLdlam(WLl) &!     ... MB emission ...
                        &/(WLlam(WLl)**5*(WTbbEXP-1.)))  !     ....... probability
    WTpDM(WLl,TEk)=WTpDM(WLl-1,TEk)                     &!     increment .........
      &+(WLchi(WLl)*(1.-WLalb(WLl))*WTbbEXP*WLdlam(WLl) &!     ... DM emission ...
                     &/(WLlam(WLl)**6*(WTbbEXP-1.)**2))  !     ....... probability
  ENDDO                                                  !   end loop over significant wavelengths
  TEbbFL=(0.659867E+16)*WTpBB(WTlMAX(TEk),TEk)/         &!   check agreement .......
                                         &(teT(TEk)**4)  !   ... with blackbody flux
  IF (ABS(TEbbFL-1.)>0.1) WRITE (6,"(3X,'Blackbody flux suspect')")
  WTpBB(WTlMIN(TEk):WTlMAX(TEk),TEk)=                   &!   normalise BB .......
                   &WTpBB(WTlMIN(TEk):WTlMAX(TEk),TEk)/ &!   ... emission .......
                                &WTpBB(WTlMAX(TEk),TEk)  !   ...... probabilities
  WTpBB(WTlMAX(TEk)+1:WLlTOT,TEk)=1.                     !   set higher BB em. probs to unity
  teLMmb(TEk)=(0.149671E+13)*WTpMB(WTlMAX(TEk),TEk)      !   compute the luminosity per unit mass
  WTpMB(WTlMIN(TEk):WTlMAX(TEk),TEk)=                   &!   normalise MB .......
                   &WTpMB(WTlMIN(TEk):WTlMAX(TEk),TEk)/ &!   ... emission .......
                                &WTpMB(WTlMAX(TEk),TEk)  !   ...... probabilities
  WTpMB(WTlMAX(TEk)+1:WLlTOT,TEk)=1.                     !   set higher MB em. probs. to unity
  teLMTdm(TEk)=(0.215343E+17)*WTpDM(WTlMAX(TEk),TEk)/   &!   compute the luminosity per unit ...
                                          (teT(TEk)**2)  !   .........mass, per unit temperature

!!!!
!
!   print*,"emprobs-dmbb teLMTdm(TEk)",teLMTdm(TEk)
!
!!!!
  WTpDM(WTlMIN(TEk):WTlMAX(TEk),TEk)=                   &!   normalise DM .......
                   &WTpDM(WTlMIN(TEk):WTlMAX(TEk),TEk)/ &!   ... emission .......
                                &WTpDM(WTlMAX(TEk),TEk)  !   ...... probabilities
  WTpDM(WTlMAX(TEk)+1:WLlTOT,TEk)=1.                     !   set higher DM em. probs. to unity

                                                         !   [] IDs OF LOWER WAVELENGTHS FOR DMBB EMISSION
  WLl=WTlMAX(TEk)+1                                      !   set WLl just above highest significant ID
  DO PRn=PRnTOT,2,-1                                     !   scan ref.probs. downwards to penultimate
    DO WHILE (WTpBB(WLl,TEk)>PRp(PRn-1))                 !     [WHILE] BB prob. greater than lower ref.prob.
      WLl=WLl-1                                          !       decrease WLl
    ENDDO                                                !     [DONE] BB prob. just less than lower ref.prob.
    WTlBBlo(PRn,TEk)=WLl                                 !     record WTlBBlo
  ENDDO                                                  !   scan done
  WTlBBlo(1,TEk)=WTlMIN(TEk)-1                           !   special extreme case
  WLl=WTlMAX(TEk)+1                                      !   set WLl just above highest significant ID
  DO PRn=PRnTOT,2,-1                                     !   scan ref.probs. downwards to penultimate
    DO WHILE (WTpMB(WLl,TEk)>PRp(PRn-1))                 !     [WHILE] MB prob. greater than lower ref.prob.
      WLl=WLl-1                                          !       decrease WLl
    ENDDO                                                !     [DONE] MB prob. just less than lower ref.prob.
    WTlMBlo(PRn,TEk)=WLl                                 !     record WTlMBlo
  ENDDO                                                  !   scan done
  WTlMBlo(1,TEk)=WTlMIN(TEk)-1                           !   special extreme case
  WLl=WTlMAX(TEk)+1                                      !   set WLl just above highest significant ID
  DO PRn=PRnTOT,2,-1                                     !   scan ref.probs. downwards to penultimate
    DO WHILE (WTpDM(WLl,TEk)>PRp(PRn-1))                 !     [WHILE] DM prob. greater than lower ref.prob.
      WLl=WLl-1                                          !       decrease WLl
    ENDDO                                                !     [DONE] DM prob. just less than lower ref.prob.
    WTlDMlo(PRn,TEk)=WLl                                 !     record WTlDMlo
  ENDDO                                                  !   scan done
  WTlDMlo(1,TEk)=WTlMIN(TEk)-1                           !   special extreme case

                                                         !   [] IDs OF UPPER WAVELENGTHS FOR DMBB EMISSION
  WLl=WTlMIN(TEk)-1                                      !   set WLl just below highest significant ID
  DO PRn=1,PRnTOT-1                                      !   scan ref.probs. upwards to penultimate
    DO WHILE (WTpBB(WLl,TEk)<PRp(PRn))                   !     [WHILE] BB prob. less than upper ref.prob.
      WLl=WLl+1                                          !       increase WLl
    ENDDO                                                !     [DONE] BB prob. just greater than upper ref.prob.
    WTlBBup(PRn,TEk)=WLl                                 !     record WTlBBup
  ENDDO                                                  !   scan done
  WTlBBup(PRnTOT,TEk)=WTlMAX(TEk)+1                      !   special extreme case
  WLl=WTlMIN(TEk)-1                                      !   set WLl just below highest significant ID
  DO PRn=1,PRnTOT-1                                      !   scan ref.probs. upwards to penultimate
    DO WHILE (WTpMB(WLl,TEk)<PRp(PRn))                   !     [WHILE] BB prob. less than upper ref.prob.
      WLl=WLl+1                                          !       increase WLl
    ENDDO                                                !     [DONE] BB prob. just greater than upper ref.prob.
    WTlMBup(PRn,TEk)=WLl                                 !     record WTlBBup
  ENDDO                                                  !   scan done
  WTlMBup(PRnTOT,TEk)=WTlMAX(TEk)+1                      !   special extreme case
  WLl=WTlMIN(TEk)-1                                      !   set WLl just below highest significant ID
  DO PRn=1,PRnTOT-1                                      !   scan ref.probs. upwards to penultimate
    DO WHILE (WTpDM(WLl,TEk)<PRp(PRn))                   !     [WHILE] BB prob. less than upper ref.prob.
      WLl=WLl+1                                          !       increase WLl
    ENDDO                                                !     [DONE] BB prob. just greater than upper ref.prob.
    WTlDMup(PRn,TEk)=WLl                                 !     record WTlBBup
  ENDDO                                                  !   scan done
  WTlDMup(PRnTOT,TEk)=WTlMAX(TEk)+1                      !   special extreme case

  IF (WTplot<0) THEN                                     !   [] RAW DIAGNOSTICS
    PRINT*,"RT_EmProbs_DMBB Raw diagnostics Data"
    WRITE (6,"(/,2X,'WLl:',6X,'WLlam:',5X,'WLdlam:',8X,'WTpBB:',/)")
    DO WLl=111,115
      WRITE (6,"(I6,2F12.3,3F14.6)") WLl,WLlam(WLl),WLdlam(WLl),WTpBB(WLl,TEk),WTpMB(WLl,TEk),WTpDM(WLl,TEk)
    ENDDO
    WRITE (*,*) ' '
    DO WLl=176,186
      WRITE (6,"(I6,2F12.3,3F14.6)") WLl,WLlam(WLl),WLdlam(WLl),WTpBB(WLl,TEk),WTpMB(WLl,TEk),WTpDM(WLl,TEk)
    ENDDO
    WRITE (6,"(/,2X,'RFn:',10X,'RFp:',4X,'WTlBBlo:',4X,'WTlBBup:',/)")
    DO PRn=089,106
      WRITE (6,"(I6,F14.6,3(4X,2I12))") PRn,PRp(PRn),WTlBBlo(PRn,TEk),WTlBBup(PRn,TEk),&
      &WTlMBlo(PRn,TEk),WTlMBup(PRn,TEk),WTlDMlo(PRn,TEk),WTlDMup(PRn,TEk)
    ENDDO
    WRITE (*,*) ' '
    DO PRn=995,998
      WRITE (6,"(I6,F14.6,3(4X,2I12))") PRn,PRp(PRn),WTlBBlo(PRn,TEk),WTlBBup(PRn,TEk),&
      &WTlMBlo(PRn,TEk),WTlMBup(PRn,TEk),WTlDMlo(PRn,TEk),WTlDMup(PRn,TEk)
    ENDDO
    WRITE (*,*) ' '
  ENDIF

ENDDO                                                    ! end loop over discrete temperatures

!---------------------------------------------------------
!---------------------------------------------------------
!           Diagnostics plotting                         !
!               & relevant data output                   !
!---------------------------------------------------------
!---------------------------------------------------------


WTlMIN = 1
WTlMAX = WLlTOT


IF (WTplot==1) THEN                                      ! [] CONDITIONAL DIAGNOSTIC PLOTS
  WLlTOTrea=DBLE(WLlTOT)                                 !   compute REAL(WLlTOT)
  WTpackINV=1./DBLE(WTpack)                              !   compute WTpackINV=1/WTpack

   OPEN(1,file=trim(adjustl(LambdaDataFile)),&
  &iostat=readcheck)                                     !Open a seperate file for wavelength data. This
                                                         !..ensures we only write this data once.

  IF(readcheck .ne. 0) then                              !Same creation/"read" check as previous
      print*,(/LambdaDataFile, &
      & " not read successfully! Aborting program!"/)
      STOP
  ENDIF

  DO WLl=1,WLlTOT                                        !     start loop over wavelengths
    PGx(WLl)=LOG10(WLlam(WLl))                           !       compute boundary wavelength (abscissa)
    WRITE(1,"(F10.3)") (/PGx(WLl)/)
  ENDDO                                                  !     end loop over wavelengths

  OPEN(2,file=trim(adjustl(BBconstantsFile)),&
  &iostat=readcheck)                                         !Open file for BB constants
                                                         !..here trim and adjustl will remove any unecessary
                                                         !..white space at the start of the filename.
  IF(readcheck .ne. 0) then                              !..Check file was created/"read" correctly
      print*,(/BBconstantsFile,&
    &  " not read successfully! Aborting program!"/)     !....ELSE error code and STOP
      STOP
  ENDIF
  WRITE(2,"(A4,1x,A4,1x,A4,1x,A4,1x,A4)") &
  &(/"temp","xmin","xmax","ymin","ymax"/)                !Write header text to file

  OPEN(3,file=trim(adjustl(BBanalyticFile)),&
  & iostat=readcheck)                                    !Open a second file, for the analytic data of the
                                                          !..BB curve.
  IF(readcheck .ne. 0) then
      print*,(/BBanalyticFile,&
      & " not read successfully! Aborting program!"/)
      STOP
  ENDIF


  OPEN(4,file=trim(adjustl(BBmcrtFile)),iostat=readcheck)!Same process of opening file for MCRT data of
                                                         !..BB curve, sampled from RT_LumPack_BB
  IF(readcheck .ne. 0) then
    print*,(/BBmcrtFile, &
    & " not read successfully! Aborting program!"/)
    STOP
  ENDIF

  DO TEk=0,TEkTOT,(TEkTOT/2)                             !   start loop over temperatures
    PGxMIN=LOG10(WLlam(WTlMIN(TEk)))+0.2                 !     compute maximum abscissa
    PGxMAX=LOG10(WLlam(WTlMAX(TEk)))-1.1                 !     compute maximum abscissa
    PGyMAX=-0.1E+11                                      !     set PGyMAX to very low value
    PGy=(-1d-10)                                        !     set PGy to extremely low value
    DO WLl=WTlMIN(TEk),WTlMAX(TEk)                       !     start loop over wavelengths
      PGy(WLl)=LOG10(WTpBB(WLl,TEk)-WTpBB(WLl-1,TEk))-  &!       compute BB emission ......
                                    &LOG10(WLdlam(WLl))  !       ... probability (ordinate)
      WRITE(3,"(F10.3,1x)",advance="no") (/PGy(WLl)/)
      IF (PGy(WLl)>PGyMAX) PGyMAX=PGy(WLl)               !       update max ordinate as appropriate
    ENDDO                                                !     end loop over wavelengths
    WRITE(3,*)
    PGyMAX=PGyMAX+0.2                                    !     compute maximum Planck Function (ordinate)
    PGyMIN=PGyMAX-2.6                                    !     compute minimum Planck Function (ordinate)


    WRITE(2,"(E9.3,1x,E9.3,1x,E9.3,1x,E9.3,1x,E9.3)")&
    &(/teT(TEk),PGxMIN,PGxMAX,PGyMIN,PGyMAX/)                !..Write single value constants to file

    WTpACC=0                                             !     set accumulator to zero
    DO WLl=1,WTpack                                      !     start loop over luminosity packets
      CALL RT_LumPack_BB(TEk,WLlTOT,WTpBB,WTlBBlo,WTlBBup,WLlEM)
      WTpACC(WLlEM)=WTpACC(WLlEM)+1                      !       increment WTpACC
    ENDDO                                                !     end loop over luminosity packets
    PGz=-0.1E+31                                         !     set PGz to extremely low value
    DO WLl=WTlMIN(TEk),WTlMAX(TEk)                       !     start loop over significant wavelengths
      PGz(WLl)=LOG10(DBLE(WTpACC(WLl))*WTpackINV/       &!       compute BB emission ......
                                          &WLdlam(WLl))  !       ... probability (ordinate)
      WRITE(4,"(F10.3,1x)",advance="no") (/PGz(WLl)/)

      IF ((TEk==(TEkTOT/2)).AND.(MOD(WLl,10)==0)) then
        ! PRINT*,"((TEk==(TEkTOT/2)).AND.(MOD(WLl,10)==0))"
        ! PRINT*,"WLl,WLlam(WLl),DBLE(WTpACC(WLl)),PGz(WLl)[MCRT BB]"
        WRITE (*,*) WLl,WLlam(WLl),DBLE(WTpACC(WLl)),PGz(WLl)
      ENDIF
    ENDDO                                                !     end loop over significant wavelengths
    WRITE(4,*)
                                                         !     [] PLOT BB SPECTRA TO SCREEN
    WRITE (*,*) ' '                                      !     print blank line
    PRINT*,"RT_EmProbs_DMBB BB Plotting Data"
    PRINT*,"teT(TEk),WTlMIN(TEk),WLlam(WTlMIN(TEk)),WTlMAX(TEk),WLlam(WTlMAX(TEk))"
    WRITE (6,"(F11.3,2(5X,I5,F15.5))") teT(TEk),WTlMIN(TEk),WLlam(WTlMIN(TEk)),WTlMAX(TEk),WLlam(WTlMAX(TEk))

    ! CALL PGBEG(0,'/XWINDOW',1,1)                         !     open PGPLOT to display on screen
    ! CALL PGENV(PGxMIN,PGxMAX,PGyMIN,PGyMAX,0,0)          !     construct frame
    ! CALL PGLAB('log\d10\u[\gl/\gmm]','log\d10\u[\fiWTpBB\fn]',&
    ! &'BB EMISSION PROBABILITIES, \fiWTpBB\fn, AS A FUNCTION OF WAVELENGTH, \gl. ')
    ! CALL PGSLS(2)                                        !     set line style to 'dashed'
    ! CALL PGLINE(WLlTOT,PGx,PGy)                          !     plot discrete probabilities
    ! CALL PGSLS(1)                                        !     set line style to 'full'
    ! CALL PGLINE(WLlTOT,PGx,PGz)                          !     plot estimated probabilities
    ! CALL PGEND                                           !     close PGPLOT
                                                         ! !     [] SAVE TO POSTSCRIPT FILE



  ENDDO                                                  !   end loop over temperatures

  close(1)                                               !Close lambda data file
  close(2)                                               !Close BB constants file
  close(3)                                               !Close Analytic BB data file
  close(4)                                               !Close MCRT BB data file

  OPEN(2,file=trim(adjustl(MBconstantsFile)),&
  &iostat=readcheck)                                         !Open file for MB constants
                                                         !..here trim and adjustl will remove any unecessary
                                                         !..white space at the start of the filename.
  IF(readcheck .ne. 0) then                              !..Check file was created/"read" correctly
      print*,(/MBconstantsFile,&
    &  " not read successfully! Aborting program!"/)     !....ELSE error code and STOP
      STOP
  ENDIF
  WRITE(2,"(A4,1x,A4,1x,A4,1x,A4,1x,A4)") &
  &(/"temp","xmin","xmax","ymin","ymax"/)                !Write header text to file

  OPEN(3,file=trim(adjustl(MBanalyticFile)),&
  & iostat=readcheck)                                    !Open a second file, for the analytic data of the
                                                          !..MB curve.
  IF(readcheck .ne. 0) then
      print*,(/MBanalyticFile,&
      & " not read successfully! Aborting program!"/)
      STOP
  ENDIF


  OPEN(4,file=trim(adjustl(MBmcrtFile)),iostat=readcheck)!Same process of opening file for MCRT data of
                                                         !..MB curve, sampled from RT_LumPack_BB
  IF(readcheck .ne. 0) then
    print*,(/MBmcrtFile, &
    & " not read successfully! Aborting program!"/)
    STOP
  ENDIF


  DO TEk=0,TEkTOT,(TEkTOT/2)                             !   start loop over temperatures
    PGxMIN=LOG10(WLlam(WTlMIN(TEk)))+0.2                 !     compute maximum abscissa
    PGxMAX=LOG10(WLlam(WTlMAX(TEk)))-1.1                 !     compute maximum abscissa
    PGyMAX=-0.1E+11                                      !     set PGyMAX to very low value
    PGy=(-1d-10)                                        !     set PGy to extremely low value


    !! This section calculates data for the analytic result
    !! of the MB spectrum.
    !! We want Log Log space. WTpMB is the probability integrated,
    !! so here we are taking the numeric differential
    !! so as to check the MBB curve is as expected.
    DO WLl=1,WLlTOT                                      !     start loop over wavelengths
      PGy(WLl)=LOG10(WTpMB(WLl,TEk)-WTpMB(WLl-1,TEk))-  &!       compute BB emission ......
                                    &LOG10(WLdlam(WLl))  !       ... probability (ordinate)
      WRITE(3,"(F10.3,1x)",advance="no") (/PGy(WLl)/)
      IF (PGy(WLl)>PGyMAX) PGyMAX=PGy(WLl)               !       update max ordinate as appropriate
    ENDDO                                                !     end loop over wavelengths
    WRITE(3,*)

    !! This section calculates data for the MBB curve from
    !! the MCRT version, by calling RT_LumPack_MB.
    !! Here we are returned an array of times each bin is
    !! called. This gives us a measure of the probability
    !! when divided by the NPacketsInverse (frequentist)
    !! and again is numerically differentiated to check
    !! that the MBB curve works as expected.
    PGyMAX=PGyMAX+0.2                                    !     compute maximum Planck Function (ordinate)
    PGyMIN=PGyMAX-2.6                                    !     compute minimum Planck Function (ordinate)


    WRITE(2,"(E9.3,1x,E9.3,1x,E9.3,1x,E9.3,1x,E9.3)")&
    &(/teT(TEk),PGxMIN,PGxMAX,PGyMIN,PGyMAX/)                !..Write single value constants to file

    WTpACC=0                                             !     set accumulator to zero
    DO WLl=1,WTpack                                      !     start loop over luminosity packets
      CALL RT_LumPack_MB(TEk,WLlTOT,WTpMB,WTlMBlo,WTlMBup,WLlEM)
      WTpACC(WLlEM)=WTpACC(WLlEM)+1                      !       increment WTpACC
    ENDDO                                                !     end loop over luminosity packets
    PGz=-0.1E+31                                         !     set PGz to extremely low value
    DO WLl=WTlMIN(TEk),WTlMAX(TEk)                       !     start loop over significant wavelengths
       PGz(WLl)=LOG10(DBLE(WTpACC(WLl))*WTpackINV/      &!       compute MB emission ......
                                          &WLdlam(WLl))  !       ... probability (ordinate)
       WRITE(4,"(F10.3,1x)",advance="no") (/PGz(WLl)/)
    ENDDO                                                !     end loop over significant wavelengths
    WRITE(4,*)
                                                         !     [] PLOT MB SPECTRA TO SCREEN
    WRITE (*,*) ' '                                      !     print blank line
        PRINT*,"RT_EmProbs_DMBB MB Plotting Data"
    PRINT*,"teT(TEk),WTlMIN(TEk),WLlam(WTlMIN(TEk)),WTlMAX(TEk),WLlam(WTlMAX(TEk))"
    WRITE (6,"(F11.3,2(5X,I5,F15.5))") teT(TEk),WTlMIN(TEk),WLlam(WTlMIN(TEk)),WTlMAX(TEk),WLlam(WTlMAX(TEk))



    ! CALL PGBEG(0,'/XWINDOW',1,1)                         !     open PGPLOT to display on screen
    ! CALL PGENV(PGxMIN,PGxMAX,PGyMIN,PGyMAX,0,0)          !     construct frame
    ! CALL PGLAB('log\d10\u[\gl/\gmm]','log\d10\u[\fiWTpMB\fn]',&
    ! &'MB EMISSION PROBABILITIES, \fiWTpMB\fn, AS A FUNCTION OF WAVELENGTH, \gl. ')
    ! CALL PGSLS(2)                                        !     set line style to 'dashed'
    ! CALL PGLINE(WLlTOT,PGx,PGy)                          !     plot discrete probabilities
    ! CALL PGSLS(1)                                        !     set line style to 'full'
    ! CALL PGLINE(WLlTOT,PGx,PGz)                          !     plot estimated probabilities
    ! CALL PGEND                                           !     close PGPLOT
                                                         ! !     [] SAVE TO POSTSCRIPT FILE



  ENDDO                                                  !   end loop over temperatures
  close(2)
  close(3)
  close(4)

  OPEN(2,file=trim(adjustl(DMconstantsFile)),&
  &iostat=readcheck)                                     !Open file for DM constants
                                                         !..here trim and adjustl will remove any unecessary
                                                         !..white space at the start of the filename.
  IF(readcheck .ne. 0) then                              !..Check file was created/"read" correctly
      print*,(/DMconstantsFile,&
    &  " not read successfully! Aborting program!"/)     !....ELSE error code and STOP
      STOP
  ENDIF
  WRITE(2,"(A4,1x,A4,1x,A4,1x,A4,1x,A4)") &
  &(/"temp","xmin","xmax","ymin","ymax"/)                !Write header text to file

  OPEN(3,file=trim(adjustl(DManalyticFile)),&
  & iostat=readcheck)                                    !Open a second file, for the analytic data of the
                                                          !..DM curve.
  IF(readcheck .ne. 0) then
      print*,(/DManalyticFile,&
      & " not read successfully! Aborting program!"/)
      STOP
  ENDIF


  OPEN(4,file=trim(adjustl(DMmcrtFile)),iostat=readcheck)!Same process of opening file for MCRT data of
                                                         !..DM curve, sampled from RT_LumPack_BB
  IF(readcheck .ne. 0) then
    print*,(/DMmcrtFile, &
    & " not read successfully! Aborting program!"/)
    STOP
  ENDIF


  DO TEk=0,TEkTOT,(TEkTOT/2)                             !   start loop over temperatures
    PGxMIN=LOG10(WLlam(WTlMIN(TEk)))+0.2                 !     compute maximum abscissa
    PGxMAX=LOG10(WLlam(WTlMAX(TEk)))-1.1                 !     compute maximum abscissa
    PGyMAX=-0.1E+11                                      !     set PGyMAX to very low value
    PGy=(-1d-10)                                        !     set PGy to extremely low value
    DO WLl=1,WLlTOT                                      !     start loop over wavelengths
      PGy(WLl)=LOG10(WTpDM(WLl,TEk)-WTpDM(WLl-1,TEk))-  &!       compute DM emission ......
                                    &LOG10(WLdlam(WLl))  !       ... probability (ordinate)
      WRITE(3,"(F10.3,1x)",advance="no") (/PGy(WLl)/)
      IF (PGy(WLl)>PGyMAX) PGyMAX=PGy(WLl)               !       update max ordinate as appropriate
    ENDDO                                                !     end loop over wavelengths
    WRITE(3,*)
    PGyMAX=PGyMAX+0.2                                    !     compute maximum Planck Function (ordinate)
    PGyMIN=PGyMAX-2.6                                    !     compute minimum Planck Function (ordinate)

    WRITE(2,"(E9.3,1x,E9.3,1x,E9.3,1x,E9.3,1x,E9.3)")&
    &(/teT(TEk),PGxMIN,PGxMAX,PGyMIN,PGyMAX/)                !..Write single value constants to file

    WTpACC=0                                             !     set accumulator to zero
    DO WLl=1,WTpack                                      !     start loop over luminosity packets
      CALL RT_LumPack_DM(TEk,WLlTOT,WTpDM,WTlDMlo,WTlDMup,WLlEM)
      WTpACC(WLlEM)=WTpACC(WLlEM)+1                      !       increment WTpACC
    ENDDO                                                !     end loop over luminosity packets
    PGz=-0.1E+31                                         !     set PGz to extremely low value
    DO WLl=WTlMIN(TEk),WTlMAX(TEk)                       !     start loop over significant wavelengths
       PGz(WLl)=LOG10(DBLE(WTpACC(WLl))*WTpackINV/      &!       compute BB emission ......
                                          &WLdlam(WLl))  !       ... probability (ordinate)
       WRITE(4,"(F10.3,1x)",advance="no") (/PGz(WLl)/)
    ENDDO                                                !     end loop over significant wavelengths
    WRITE(4,*)
                                                         !     [] PLOT DM SPECTRA TO SCREEN
    WRITE (*,*) ' '                                      !     print blank line
    PRINT*,"RT_EmProbs_DMBB DMBB Plotting Data"
    PRINT*,"teT(TEk),WTlMIN(TEk),WLlam(WTlMIN(TEk)),WTlMAX(TEk),WLlam(WTlMAX(TEk))"
    WRITE (6,"(F11.3,2(5X,I5,F15.5))") teT(TEk),WTlMIN(TEk),WLlam(WTlMIN(TEk)),WTlMAX(TEk),WLlam(WTlMAX(TEk))

    ! CALL PGBEG(0,'/XWINDOW',1,1)                         !     open PGPLOT to display on screen
    ! CALL PGENV(PGxMIN,PGxMAX,PGyMIN,PGyMAX,0,0)          !     construct frame
    ! CALL PGLAB('log\d10\u[\gl/\gmm]','log\d10\u[\fiWTpDM\fn]',&
    ! &'DM EMISSION PROBABILITIES, \fiWTpDM\fn, AS A FUNCTION OF WAVELENGTH, \gl. ')
    ! CALL PGSLS(2)                                        !     set line style to 'dashed'
    ! CALL PGLINE(WLlTOT,PGx,PGy)                          !     plot discrete probabilities
    ! CALL PGSLS(1)                                        !     set line style to 'full'
    ! CALL PGLINE(WLlTOT,PGx,PGz)                          !     plot estimated probabilities
    ! CALL PGEND                                           !     close PGPLOT
                                                         ! !     [] SAVE TO POSTSCRIPT FILE



  ENDDO                                                  !   end loop over temperatures

  ! DO TEk=0,TEkTOT,(TEkTOT/2)                             !   start loop over temperatures
    ! PGxMIN=LOG10(WLlam(WTlMIN(TEk)))+0.2                 !     compute maximum abscissa
    ! PGxMAX=LOG10(WLlam(WTlMAX(TEk)))-1.1                 !     compute maximum abscissa
    ! PGyMAX=-0.1E+11                                      !     set PGyMAX to very low value
    ! PGy=(-1d-10)                                        !     set PGy to extremely low value
    ! DO WLl=1,WLlTOT                                      !     start loop over wavelengths
      ! PGy(WLl)=LOG10(WTpDM(WLl,TEk)-WTpDM(WLl-1,TEk))-  &!       compute DM emission ......
                                    ! &LOG10(WLdlam(WLl))  !       ... probability (ordinate)
      ! IF (PGy(WLl)>PGyMAX) PGyMAX=PGy(WLl)               !       update max ordinate as appropriate
      ! PGz(WLl)=LOG10(WTpMB(WLl,TEk)-WTpMB(WLl-1,TEk))-  &!       compute DM emission ......
                                    ! &LOG10(WLdlam(WLl))  !       ... probability (ordinate)
      ! IF (PGz(WLl)>PGyMAX) PGyMAX=PGy(WLl)               !       update max ordinate as appropriate
    ! ENDDO                                                !     end loop over wavelengths
    ! PGyMAX=PGyMAX+0.2                                    !     compute maximum Planck Function (ordinate)
    ! PGyMIN=PGyMAX-2.6                                    !     compute minimum Planck Function (ordinate)
                                                         ! !     [] PLOT TO SCREEN
    ! WRITE (*,*) ' '                                      !     print blank line
    ! WRITE (6,"(F11.3,2(5X,I5,F15.5))") teT(TEk),WTlMIN(TEk),WLlam(WTlMIN(TEk)),WTlMAX(TEk),WLlam(WTlMAX(TEk))



    ! ! CALL PGBEG(0,'/XWINDOW',1,1)                         !     open PGPLOT to display on screen
    ! ! CALL PGENV(PGxMIN,PGxMAX,PGyMIN,PGyMAX,0,0)          !     construct frame
    ! ! CALL PGLAB('log\d10\u[\gl/\gmm]','log\d10\u[\fiWTpDM\fn]',&
    ! ! &'DM EMISSION PROBABILITIES, \fiWTpDM\fn, AS A FUNCTION OF WAVELENGTH, \gl. ')
    ! ! CALL PGSLS(2)                                        !     set line style to 'dashed'
    ! ! CALL PGLINE(WLlTOT,PGx,PGy)                          !     plot discrete probabilities
    ! ! CALL PGSLS(1)                                        !     set line style to 'full'
    ! ! CALL PGLINE(WLlTOT,PGx,PGz)                          !     plot estimated probabilities
    ! ! CALL PGEND                                           !     close PGPLOT
                                                         ! ! !     [] SAVE TO POSTSCRIPT FILE
  ! ENDDO                                                  !   end loop over temperatures

ENDIF                                                    ! end diagnostic plots
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_EmProbs_DMBB
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_Cyl1D_InjectIsotropicAndTrack_ZeroOpacity(CFw,CFw2)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine treats the zero-opacity case. Note that it assumes all
! 'events' occur on shell boundaries, and this is not generally the case.
! The mean intensity should be the same in all shells.

! It is given:
!   the boundary radius of the filament                    (CFwB);
!   the number of shells                                   (CFcTOT);
!   the boundary radii of the shells                       (CFw(0:CFcTOT));
!   the boundary radii of the shells squared               (CFw2(0:CFcTOT));
!   and the number of luminosity packets to be injected    (LPpTOT).

! It prints out:
!   the angle-mean intensity in each shell                 (RFj(0:CFcTOT)) --- should be  ~1;
!   its mean                                               (RFmuJ)         --- should be  ~1;
!   and its standard deviation                             (RFsdJ)         --- should be <<1.

USE CONSTANTS
USE PHYSICAL_CONSTANTS

IMPLICIT NONE                                             ! [] DECLARATIONS
REAL(KIND=8),INTENT(IN),DIMENSION(0:CFcTOT) :: CFw       ! boundary radii of cells (cm)
REAL(KIND=8),INTENT(IN),DIMENSION(0:CFcTOT) :: CFw2      ! squared boundary radii of shells
                                                        ! [] DRONES
INTEGER                                     :: CFc,CFcc  ! dummy shell IDs
REAL(KIND=8)                                :: CFw2B     ! squared boundary radius (cm^2)
REAL(KIND=8)                                :: LPalfa    ! 'outwardness' of packet (cm)
REAL(KIND=8)                                :: LPbeta    ! squared tangent-distance (cm^2)
REAL(KIND=8),DIMENSION(1:3)                 :: LPe       ! direction of luminosity packet (unit vector)
REAL(KIND=8)                                :: LPe1122   ! e_x^2+e_y^2
INTEGER                                     :: LPp       ! dummy ID of luminosity packet
REAL(KIND=8),DIMENSION(1:3)                 :: LPr       ! position of luninosity packet
REAL(KIND=8)                                :: LPr1122   ! r_x^2+r_y^2
REAL(KIND=8)                                :: LPs       ! path-length of luminosity packet
REAL(KIND=8)                                :: LRD       ! linear random deviate on [0,1]
REAL(KIND=8),DIMENSION(1:CFcTOT)            :: RFj       ! angle-mean intensity in each shell
REAL(KIND=8)                                :: RFmuJ     ! volume-weighted mean angle-mean intensity
REAL(KIND=8)                                :: RFsdJ     ! volume-weighted stdev angle-mean intensity
REAL(KIND=8)                                :: alpha     ! 'outwardness' (pc)
REAL(KIND=8)                                :: beta      ! squared tangent-distance (pc^2)
REAL(KIND=8)                                :: phi       ! azimuthal angle
REAL(KIND=8)                                :: sintheta  ! sine of polar angle

                                                         ! [] COMPUTATION
CFw2B=CFwB**2                                            ! compute squared boundary radius (pc^2)
RFj=0.                                                   ! set RFj to zero
DO LPp=1,LPpTOT                                          ! start loop over luminosity packets
  LPr(1)=-CFwB;  LPr(2)=0.;  LPr(3)=0.                   !   position packet on boundary of filament
  LPr1122=CFw2B                                          !   compute squared distance from spine
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  LPe(1)=SQRT(LRD)                                       !   compute e(1)
  sintheta=SQRT(1.-LRD)                                  !   compute SIN(theta)
  CALL RANDOM_NUMBER(LRD)                                !   generate linear random deviate on [0,1]
  phi=twopi*LRD                                     !   compute phi
  LPe(2)=sintheta*COS(phi)                               !   compute e_y
  LPe(3)=sintheta*SIN(phi)                               !   compute e_z
  CFc=CFcTOT                                             !   set shell ID to n_TOT (outermost shell)
  DO WHILE (CFc<=CFcTOT)                                 !   keep going until packet exits filament
    CFcc=CFc                                             !     record ID of shell being entered
    LPe1122=LPe(1)**2+LPe(2)**2                          !     compute e_x^2+e_y^2
    alpha=((LPr(1)*LPe(1))+(LPr(2)*LPe(2)))/LPe1122      !     compute 'outwardness' (r_x*e_x)+(r_y*e_y)
    IF (alpha>0.) THEN                                   !     [IF] travelling outward, [THEN]
      LPs=-alpha+                                       &!       compute .......
      &SQRT(alpha**2+((CFw2(CFc)-CFw2(CFc-1))/LPe1122))  !       ... path-length
      CFc=CFc+1                                          !       increase shell ID
    ELSE                                                 !     [ELSE] travelling inward, so
      beta=alpha**2-((CFw2(CFc)-CFw2(CFc-1))/LPe1122)    !       compute beta, and
      IF (beta>0.) THEN                                  !       [IF] beta>0, [THEN] hits inner shell
        LPs=-alpha-SQRT(beta)                            !         compute path
        CFc=CFc-1                                        !         decrease shell ID
      ELSE                                               !       [ELSE] traverses shell, so
        LPs=-2.*alpha                                    !         compute path
        CFc=CFc+1                                        !         increase shell ID
      ENDIF                                              !       [ENDIF] inward cases done
    ENDIF                                                !     [ENDIF] all cases done
    LPr(1:2)=LPr(1:2)+(LPs*LPe(1:2))                     !     advance position of packet
    LPr1122=(LPr(1)**2)+(LPr(2)**2)                      !     compute squared distance from spine
    RFj(CFcc)=RFj(CFcc)+LPs                              !     increment mean intensity
  ENDDO                                                  !   packet exits filament
ENDDO                                                    ! end loop over luminosity packets
alpha=CFwB/(2.*DBLE(LPpTOT))                             ! use alpha as normalisation coefficient
RFmuJ=0.                                                 ! set RFmuJ to zero
RFsdJ=0.                                                 ! set RFsdJ to zero
DO CFc=1,CFcTOT                                          ! start loop over shells
  RFj(CFc)=RFj(CFc)*alpha/(CFw2(CFc)-CFw2(CFc-1))        !   normalise intensity in cell
  RFmuJ=RFmuJ+RFj(CFc)                                   !   increment mu_J0 accumulator
  RFsdJ=RFsdJ+(RFj(CFc)**2)                              !   increment sigma_J0 accumulator
ENDDO                                                    ! end loop over shells
RFmuJ=RFmuJ/DBLE(CFcTOT)                                 ! compute RFmuJ
RFsdJ=SQRT((RFsdJ/DBLE(CFcTOT))-(RFmuJ**2))              ! compute RFsdJ
WRITE (6,"(/,3X,'TEST CASE 0: ZERO OPACITY, SHELL ANGLE-MEAN INTENSITIES (RFj)')")
WRITE (6,"(3X,'RFj:',4X,5F10.5,4X,'(innermost five shells)')") RFj(1:5)
WRITE (6,"(3X,'RFj:',4X,5F10.5,4X,'(outermost five shells)')") RFj(CFcTOT-4:CFcTOT)
WRITE (6,"(3X,'Shell-mean of angle-mean intensity, RFmuJ:',6X,F10.7)") RFmuJ
WRITE (6,"(3X,'Shell-SD of angle-mean intensity,   RFsdJ:',6X,F10.7)") RFsdJ
WRITE (6,"(3X,'Number of shells:            ',18X,I11)") CFcTOT
WRITE (6,"(3X,'Number of luminosity packets:',18X,I11,/)") LPpTOT


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_Cyl1D_InjectIsotropicAndTrack_ZeroOpacity
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_Cyl1D_InjectIsotropicAndTrack_UniformScatteringOpacity&
&(CFw,CFw2)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine deals with the pure scattering case, with uniform scattering opacity.

! It is given:
!   the boundary radius of the filament                   (CFwB);
!   the number of shells                                  (CFcTOT);
!   the boundary radii of the shells                      (CFw(0:CFc_TOT));
!   the boundary radii of the shells squared              (CFw2(0:CFc_tot));
!   the volume scattering opacity coefficient             (DGkapV);
!   and the number of luminosity packets to be injected   (LPpTOT).

! It prints out:
!   the normalised angle-mean intensity in each shell     (RFj(0:CFcTOT)) --- should be  ~1;
!   its mean                                              (RFmuJ)         --- should be  ~1;
!   and its standard deviation                            (RFsdJ)         --- should be <<1.
USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! DECLARATIONS
REAL(KIND=8),INTENT(IN),DIMENSION(0:CFcTOT) :: CFw       ! boundary radii of shells
REAL(KIND=8),INTENT(IN),DIMENSION(0:CFcTOT) :: CFw2      ! squared boundary radii of shells
                                                         ! [] DRONES
INTEGER                                     :: CFc,CFcc  ! dummy shell IDs
REAL(KIND=8)                                :: CFtauTOT  ! total optical depth through filament
REAL(KIND=8)                                :: CFw2B     ! squared boundary radius (pc^2)
REAL(KIND=8)                                :: LPalfa    ! 'outwardness' of packet (pc)
REAL(KIND=8)                                :: LPbeta    ! squared tangent-distance (pc^2)
REAL(KIND=8),DIMENSION(1:3)                 :: LPe       ! direction of luminosity packet (unit vector)
REAL(KIND=8)                                :: LPe1122   ! e_x^2+e_y^2
INTEGER                                     :: LPnSCA    ! number of scatterings
INTEGER                                     :: LPp       ! dummy ID of luminosity packet
REAL(KIND=8),DIMENSION(1:3)                 :: LPr       ! position of luninosity packet
REAL(KIND=8)                                :: LPr1122   ! r_x^2+r_y^2
REAL(KIND=8)                                :: LPtau     ! optical depth
REAL(KIND=8)                                :: LPs       ! path-length of lum.pack. to shell boundary
REAL(KIND=8)                                :: LPsTAU    ! path-length of lum.pack. to scattering
REAL(KIND=8)                                :: LRD       ! linear random deviate on [0,1]
REAL(KIND=8),DIMENSION(1:CFcTOT)            :: RFj       ! angle-mean intensity in each shell
REAL(KIND=8)                                :: RFmuJ     ! volume-weighted mean angle-mean intensity
REAL(KIND=8)                                :: RFsdJ     ! volume-weighted stdev angle-mean intensity
REAL(KIND=8)                                :: alpha     ! 'outwardness' (pc)
REAL(KIND=8)                                :: beta      ! squared tangent-distance (pc^2)
REAL(KIND=8)                                :: phi       ! azimuthal angle
REAL(KIND=8)                                :: sintheta  ! sine of polar angle
INTEGER                                     :: NMeIN,NMeSC,NMtau
REAL(KIND=8),DIMENSION(1:3)                 :: MUeIN,SDeIN
REAL(KIND=8),DIMENSION(1:3)                 :: MUeSC,SDeSC
REAL(KIND=8)                                :: MUtau,SDtau

                                                         ! [] COMPUTATION
CFw2B=CFwB**2                                            ! compute W_B squared
RFj=0.                                                   ! set mean intensities to zero
LPnSCA=0                                                 ! set number of scatterings to zero
NMeIN=0;   MUeIN=0.;   SDeIN=0.                          ! *****
NMeSC=0;   MUeSC=0.;   SDeSC=0.                          ! *****
NMtau=0;   MUtau=0.;   SDtau=0.                          ! *****
DO LPp=1,LPpTOT                                          ! start loop over luminosity packets
  CALL RT_Cyl1D_InjectIsotropic                         &!   generate and inject ...
  &                  (LPr,LPr1122,LPe,LPtau)        !   ... a luminosity packet
  NMeIN=NMeIN+1;   MUeIN(1:3)=MUeIN(1:3)+ABS(LPe(1:3));   SDeIN(1:3)=SDeIN(1:3)+(LPe(1:3)**2) ! *****
  NMtau=NMtau+1;   MUtau=MUtau+LPtau;                     SDtau=SDtau+(LPtau**2)              ! *****
  LPsTAU=LPtau/DGkapV                                    !   convert optical depth to distance
  CFc=CFcTOT                                             !   set shell ID to CFcTOT (outermost shell)
  DO WHILE (CFc<=CFcTOT)                                 !   keep going until packet exits filament
    CFcc=CFc                                             !     record ID of shell being entered
    LPe1122=(LPe(1)**2)+(LPe(2)**2)                      !     compute e_x^2+e_y^2
    alpha=((LPr(1)*LPe(1))+(LPr(2)*LPe(2)))/LPe1122      !     compute 'outwardness'
    IF (alpha>0.) THEN                                   !     [IF] travelling outward, [THEN]
      LPs=-alpha+                                       &!       compute .......
      &SQRT(alpha**2+((CFw2(CFc)-LPr1122)/LPe1122))      !       ... path length
      CFc=CFc+1                                          !       increase shell ID
    ELSE                                                 !     [ELSE] travelling inward, so
      beta=alpha**2+((CFw2(CFc-1)-LPr1122)/LPe1122)      !       compute beta, and
      IF (beta>0.) THEN                                  !       [IF] beta>0, [THEN] hits inner shell
        LPs=-alpha-SQRT(beta)                            !         compute path
        CFc=CFc-1                                        !         decrease shell ID
      ELSE                                               !       [ELSE] traverses shell, so
        LPs=-alpha+                                     &!         compute .......
        &SQRT(alpha**2+(CFw2(CFc)-LPr1122)/LPe1122)      !         ... path length
        CFc=CFc+1                                        !         increase shell ID
      ENDIF                                              !       [ENDIF] inward cases done
    ENDIF                                                !     [ENDIF] all cases done
    IF (LPsTAU<LPs) THEN                                 !     [IF] range of packet too small
      LPr(1:2)=LPr(1:2)+LPsTAU*LPe(1:2)                  !       advance position
      RFj(CFcc)=RFj(CFcc)+LPsTAU                         !     increment sum on intercept lengths
      ! print*,"if"
      ! print*,"RFj(CFcc)",RFj(CFcc)
      CFc=CFcc                                           !       record that still in same shell
      CALL RT_ReDirectIsotropic(LPe,LPtau)               !       generate new direction and optical depth
      NMeSC=NMeSC+1;   MUeSC(1:3)=MUeSC(1:3)+ABS(LPe(1:3));   SDeSC(1:3)=SDeSC(1:3)+(LPe(1:3)**2) ! *****
      NMtau=NMtau+1;   MUtau=MUtau+LPtau;                     SDtau=SDtau+(LPtau**2)              ! *****
      LPsTAU=LPtau/DGkapV                                !       convert optical depth to distance
      LPnSCA=LPnSCA+1                                    !       increment number of scatterings
    ELSE                                                 !     [ELSE] exit shell
      LPr(1:2)=LPr(1:2)+LPs*LPe(1:2)                     !       advance position to shell boundary
      RFj(CFcc)=RFj(CFcc)+LPs                            !     increment sum on intercept lengths
      ! print*,"else"
      ! print*,"RFj(CFcc)",RFj(CFcc)
      LPsTAU=LPsTAU-LPs                                  !       reduce remaining distance
    ENDIF                                                !     [ENDIF] continue
    LPr1122=LPr(1)**2+LPr(2)**2                          !     compute distane from spine
  ENDDO                                                  !   packet exits filament
ENDDO                                                    ! end loop over luminosity packets
alpha=CFwB/(2.*DBLE(LPpTOT))                             ! use alpha as normalisation coefficient

RFmuJ=0.                                                 ! set RFmuJ to zero
RFsdJ=0.                                                 ! set RFsdJ to zero
DO CFc=1,CFcTOT                                          ! start loop over shells
  RFj(CFc)=RFj(CFc)*alpha/(CFw2(CFc)-CFw2(CFc-1))        !   normalise intensity in cell
  RFmuJ=RFmuJ+RFj(CFc)                                   !   increment mu_J0 accumulator
  RFsdJ=RFsdJ+(RFj(CFc)**2)                              !   increment sigma_J0 accumulator
ENDDO                                                    ! end loop over shells
RFmuJ=RFmuJ/DBLE(CFcTOT)                                 ! compute RFmuJ
RFsdJ=SQRT((RFsdJ/DBLE(CFcTOT))-(RFmuJ**2))              ! compute RFsdJ
CFtauTOT=2.*CFwB*DGkapV
WRITE (6,"(/,3X,'TEST CASE 1: UNIFORM SCATTERING OPACITY, SHELL ANGLE-MEAN INTENSITIES (RFj)')")
WRITE (6,"(3X,'RFj:',4X,5F10.5,4X,'(innermost five shells)')") RFj(1:5)
WRITE (6,"(3X,'RFj:',4X,5F10.5,4X,'(outermost five shells)')") RFj(CFcTOT-4:CFcTOT)
WRITE (6,"(3X,'Shell-mean of angle-mean intensity, RFmuJ:',6X,F10.7)") RFmuJ
WRITE (6,"(3X,'Shell-SD of angle-mean intensity,   RFsdJ:',6X,F10.7)") RFsdJ
WRITE (6,"(3X,'Number of shells:            ',18X,I11)") CFcTOT
WRITE (6,"(3X,'Number of luminosity packets:',18X,I11)") LPpTOT
WRITE (6,"(3X,'Number of scatterings:       ',18X,I11)") LPnSCA
WRITE (6,"(3X,'Total optical depth right through filament:',5X,F10.3,/)") CFtauTOT

MUeIN(1:3)=MUeIN(1:3)/DBLE(NMeIN)                         ! compute mean of injection directions
SDeIN(1:3)=SQRT((SDeIN(1:3)/DBLE(NMeIN))-(MUeIN(1:3)**2)) ! compute StDev of injection directions
WRITE (6,"(3X,'MEAN and StDev OF INJECTION DIRECTION:',5X,3(3X,2F8.5),5X,I11)") &
&MUeIN(1),SDeIN(1),MUeIN(2),SDeIN(2),MUeIN(3),SDeIN(3),NMeIN
WRITE (6,"(31X,'SHOULD BE:',9X,'0.66667 0.23570    0.42441 0.26433    0.42441 0.26433')")
MUeSC(1:3)=MUeSC(1:3)/DBLE(NMeSC)                         ! compute mean of scattering directions
SDeSC(1:3)=SQRT((SDeSC(1:3)/DBLE(NMeSC))-(MUeSC(1:3)**2)) ! compute StDev of scattering directions
WRITE (6,"(3X,'MEAN and StDev OF SCATTERING DIRECTION:',4X,3(3X,2F8.5),5X,I11)") &
&MUeSC(1),SDeSC(1),MUeSC(2),SDeSC(2),MUeSC(3),SDeSC(3),NMeSC
WRITE (6,"(32X,'SHOULD BE:',8X,'0.50000 0.28868    0.50000 0.28868    0.50000 0.28868')")
MUtau=MUtau/DBLE(NMtau)                                   ! compute mean of scattering directions
SDtau=SQRT((SDtau/DBLE(NMtau))-(MUtau**2))                ! compute StDev of scattering directions
WRITE (6,"(3X,'MEAN and StDev OF OPTICAL DEPTH:',14X,2F8.5,43X,I11)") &
&MUtau,SDtau,NMtau
WRITE (6,"(25X,'SHOULD BE:',15X,'1.00000 1.00000',/)")

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_Cyl1D_InjectIsotropicAndTrack_UniformScatteringOpacity
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_Cyl1D_SchusterDensities(CFw,CFrho,CFmu,CFmuTOT,CFsig)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine computes the cell densities, the total line mass, and the
! column-density through the spine for a filament with a Schuster profile.

! It is given:
!   the boundary radii of the shells (cm)                      (CFw(0:CFcTOT));
!   and a flag to sanction diagnostics                         (CFprof).

! It returns:
!   the volume-densities in the shells (g/cm^3)                (CFrho(1:CFcTOT));
!   the line-densities of the shells (g/cm)                    (CFmu(1:CFcTOT));
!   the total line-density of the filament (g/cm)              (CFmuTOT);
!   and the total column-density through the spine (g/cm^2)    (CFsig).
USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! DECLARATIONS
REAL(KIND=8),INTENT(IN),DIMENSION(0:CFcTOT) :: CFw       ! boundary radii of cells (cm)
REAL(KIND=8),INTENT(OUT),DIMENSION(1:CFcTOT):: CFrho     ! density in cells (g/cm^3)
REAL(KIND=8),INTENT(OUT),DIMENSION(1:CFcTOT):: CFmu      ! density in cells (g/cm)
REAL(KIND=8),INTENT(OUT)                    :: CFmuTOT   ! total line-density of filament (g/cm)
REAL(KIND=8),INTENT(OUT)                    :: CFsig     ! surface-density through spine (g/cm^2)
                                                         ! [] DRONES
REAL(KIND=8)                                :: coeff      ! coefficient
INTEGER                                     :: CFc       ! dummy ID of shell
REAL(KIND=8)                                :: CFzet2HI  ! zeta^2 outer
REAL(KIND=8)                                :: CFzet2LO  ! zeta^2 inner
                                                         ! [] PGPLOT
REAL(KIND=4),DIMENSION(1:CFcTOT)            :: PGx       ! array for log10[lam] (abscissa)
REAL(KIND=4)                                :: PGxMAX    ! upper limit on log10[lam]
REAL(KIND=4),DIMENSION(1:CFcTOT)            :: PGy       ! array for log10[PlanckFn] (ordinate)
REAL(KIND=4)                                :: PGyMAX    ! upper limit on log10[PlanckFn]
character(len=50)                           :: filename ="density_cell.csv"


open(1,file=trim(adjustl(filename)))

IF (CFprof==1) WRITE (6,"(5X,'c:',12X,'w/cm:',6X,'w/pc:',14X,'rho.cm^3/g:',4X,'rho.cm^3/H2:',15X,'mu.cm/g:',5X,'mu.pc/MSun:')")

IF (CFschP==0) THEN                                      ! [IF] p=1, [THEN]
  coeff=6.2831853*CFrho0*CFw0**2
  CFzet2LO=0.                                            !   set CFzet2LO to zero
  CFrho=CFrho0
  DO CFc=1,CFcTOT                                        !   start loop over cells
    CFzet2HI=(CFw(CFc)/CFw0)**2                          !     compute CFzet2HI (zeta^2)
    CFmu(CFc)=coeff*0.5d0*(CFzet2HI - CFzet2LO) !     compute CFm
    CFzet2LO=CFzet2HI                                    !     update CFzet2LO
    ! PGx(CFc)=(0.324078E-18)*CFw(CFc)                     !     rescale w to pc for PGPLOT
    ! PGy(CFc)=(0.210775E+24)*CFrho(CFc)                   !     rescale rho to nH2/cm^3
    IF ((CFprof==1).AND.(MOD(CFc,INT(DBLE(CFcTOT)/30.))==0))& ! print out .............
    &WRITE (6,"(I7,7X,E11.3,X,E16.5,15X,E11.3,6X,E11.3,13X,E11.3,6X,E16.5)")&          ! ... selected points ...
    &CFc,CFw(CFc),CFrho(CFc),CFmu(CFc),(0.155129E-15)*CFmu(CFc) ! ............ on profile
  ENDDO                                                  !   end loop over cells

  CFmuTOT=coeff*CFzet2HI*0.5d0                            !   compute total line-density
  CFsig=2.*CFrho0*CFw(CFcTOT)                             !   compute surface-density through spine
ENDIF

                                                         ! [] CASE p=1
IF (CFschP==1) THEN                                      ! [IF] p=1, [THEN]
  coeff=6.2831853*CFrho0*CFw0**2
  CFzet2LO=0.                                            !   set CFzet2LO to zero
  DO CFc=1,CFcTOT                                        !   start loop over cells
    CFzet2HI=(CFw(CFc)/CFw0)**2                          !     compute CFzet2HI (zeta^2)
    CFrho(CFc)=2.*CFrho0*(SQRT(1.+CFzet2HI)-SQRT(1.+CFzet2LO))/(CFzet2HI-CFzet2LO)
    CFmu(CFc)=coeff*(SQRT(1.+CFzet2HI)-SQRT(1.+CFzet2LO)) !     compute CFmu
    CFzet2LO=CFzet2HI                                    !     update CFzet2LO
    ! PGx(CFc)=(0.324078E-18)*CFw(CFc)                     !     rescale w to pc for PGPLOT
    ! PGy(CFc)=(0.210775E+24)*CFrho(CFc)                   !     rescale rho to nH2/cm^3
    IF ((CFprof==1).AND.(MOD(CFc,INT(DBLE(CFcTOT)/30.))==0))& ! print out .............
    &WRITE (6,"(I7,7X,E11.3,X,E16.5,15X,E11.3,6X,E11.3,13X,E11.3,6X,E16.5)")&          ! ... selected points ...
    &CFc,CFw(CFc),CFrho(CFc),CFmu(CFc),(0.155129E-15)*CFmu(CFc) ! ............ on profile
  ENDDO                                                  !   end loop over cells
  CFmuTOT=coeff*(SQRT(1.+CFzet2HI)-1.)                    !   compute total line-density
  CFsig=2.*CFrho0*CFw0*LOG((CFw(CFcTOT)/CFw0)+SQRT(1.+CFzet2HI))!   compute surface-density through spine
ENDIF                                                    ! [ENDIF]
                                                         ! [] CASE p=2
IF (CFschP==2) THEN                                      ! [IF] p=2, [THEN]
  coeff=3.14159274*CFrho0*CFw0**2
  CFzet2LO=0.                                            !   set CFzet2LO to zero
  DO CFc=1,CFcTOT                                        !   start loop over cells
    CFzet2HI=(CFw(CFc)/CFw0)**2                          !     compute CFzet2HI (zeta^2)
    CFrho(CFc)=CFrho0*LOG((1.+CFzet2HI)/(1.+CFzet2LO))/(CFzet2HI-CFzet2LO)
    CFmu(CFc)=coeff*LOG((1.+CFzet2HI)/(1.+CFzet2LO))      !     compute CFmu
    CFzet2LO=CFzet2HI                                    !     update CFzet2LO
    ! PGx(CFc)=(0.324078E-18)*CFw(CFc)                     !     rescale w to pc for PGPLOT
    ! PGy(CFc)=(0.210775E+24)*CFrho(CFc)                   !     rescale rho to nH2/cm^3
    IF (MOD(CFc,INT(DBLE(CFcTOT)/30.))==0) WRITE (6,"(I7,15X,E10.3,5X,F10.5,21X,E10.3,7X,F10.1)") &
    &CFc,CFw(CFc),CFrho(CFc)         !     print out selected points on profile
  ENDDO                                                  !   end loop over cells
  CFmuTOT=coeff*LOG(1.+CFzet2HI)                          !   compute total line-density
  CFsig=2.*CFrho0*CFw0*ATAN(CFw(CFcTOT)/CFw0)            !   compute surface-density through spine
ENDIF                                                    ! [ENDIF]
                                                         ! [] CASE p=3
IF (CFschP==3) THEN                                      ! [IF] p=3, [THEN]
  coeff=6.2831853*CFrho0*CFw0**2
  CFzet2LO=0.                                            !   set CFzet2LO to zero
  DO CFc=1,CFcTOT                                        !   start loop over cells
    CFzet2HI=(CFw(CFc)/CFw0)**2                          !     compute CFzet2HI (zeta^2)
    CFrho(CFc)=2.*CFrho0*((1./SQRT(1.+CFzet2LO))-(1./SQRT(1.+CFzet2HI)))/(CFzet2HI-CFzet2LO)
    CFmu(CFc)=coeff*((1./SQRT(1.+CFzet2LO))-(1./SQRT(1.+CFzet2HI)))
    CFzet2LO=CFzet2HI                                    !     update CFzet2LO
    ! PGx(CFc)=(0.324078E-18)*CFw(CFc)                     !     rescale w to pc for PGPLOT
    ! PGy(CFc)=(0.210775E+24)*CFrho(CFc)                   !     rescale rho to nH2/cm^3
    IF (MOD(CFc,INT(DBLE(CFcTOT)/30.))==0) WRITE (6,"(I7,15X,E10.3,5X,F10.5,21X,E10.3,7X,F10.1)") &
    &CFc,CFw(CFc),CFrho(CFc)         !     print out selected points on profile
  ENDDO                                                  !   end loop over cells
  CFmuTOT=coeff*(1.-(1./SQRT(1.+CFzet2HI))) !   compute total line-density
  CFsig=2.*CFrho0*CFw(CFcTOT)/SQRT(1.+CFzet2HI)          !   compute surface-density through spine
ENDIF                                                    ! [ENDIF]
                                                         ! [] CASE p=4
IF (CFschP==4) THEN                                      ! [IF] p=3, [THEN]
  coeff=3.14159274*CFrho0*CFw0**2
  CFzet2LO=0.                                            !   set CFzet2LO to zero
  DO CFc=1,CFcTOT                                        !   start loop over cells
    CFzet2HI=(CFw(CFc)/CFw0)**2                          !     compute CFzet2HI (zeta^2)
    CFrho(CFc)=CFrho0*((CFzet2HI/(1.+CFzet2HI))-(CFzet2LO/(1.+CFzet2LO)))/(CFzet2HI-CFzet2LO)
    CFmu(CFc)=coeff*((CFzet2HI/(1.+CFzet2HI))-(CFzet2LO/(1.+CFzet2LO)))
    CFzet2LO=CFzet2HI                                    !     update CFzet2LO
    ! PGx(CFc)=(0.324078E-18)*CFw(CFc)                     !     rescale w to pc for PGPLOT
    ! PGy(CFc)=(0.210775E+24)*CFrho(CFc)                   !     rescale rho to nH2/cm^3
    IF (MOD(CFc,INT(DBLE(CFcTOT)/30.))==0) WRITE (6,"(I7,15X,E10.3,5X,F10.5,21X,E10.3,7X,F10.1)") &
    &CFc,CFw(CFc),CFrho(CFc)      !     print out selected points on profile
  ENDDO                                                  !   end loop over cells
  CFmuTOT=coeff*CFzet2HI/(1.+CFzet2HI)                     !   compute total line-density
  CFsig=CFrho0*CFw0*(ATAN(CFw(CFcTOT)/CFw0)+(CFw(CFcTOT)/(CFw0*(1.+CFzet2HI)))) ! surface-density through spine
ENDIF                                                    ! [ENDIF]

write(1,"(4(A3,1x))") (/"rho","pos","shp","rh0"/)
do CFc=1,CFcTOT
  if (CFC ==1 ) THEN
    write(1,"(ES18.5,1x,ES18.5,1x,ES18.5,1x,ES18.5,1x)") (/CFrho(CFc),dble(CFc),dble(CFschP),CFrho0/)
  else
    write(1,"(ES18.5,1x,ES18.5,1x)") (/CFrho(CFc),dble(CFc)/)
  endif
enddo

close(1)
                                                         ! [] DIAGNOSTICS
! IF (CFprof==1) THEN                                      ! [IF] diagnostics sanctioned, [THEN]
  ! WRITE (6,"(/,3X,'TOTAL LINE-DENSITY:',16X,E10.3,' g/cm',7X,F10.3,' MSun/pc')") CFmuTOT,(0.155129E-15)*CFmuTOT
  ! WRITE (6,"(3X,'CENTRAL SURFACE-DENSITY:',9X,E10.3,' g/cm^2',7X,E10.3,' H2/cm^2',/)") CFsig,(0.210775E+24)*CFsig
  ! CALL PGBEG(0,'/XWINDOW',1,1)                           !   open PGPLOT to display on screen
  ! !CALL PGBEG(0,'/PS',1,2)                                !   open PGPLOT to produce postscript
  ! PGxMAX=1.1*PGx(CFcTOT)                                 !   compute maximum abscissa
  ! PGyMAX=1.1*PGy(1)                                      !   compute maximum ordinate
  ! CALL PGENV(0.0,PGxMAX,0.0,PGyMAX,0,0)                  !   construct frame
  ! CALL PGLAB('\fiw\fn/pc','         \fin\fn\dH\d2\u\u/cm\u-3\d','SCHUSTER DENSITY PROFILE')
  ! CALL PGLINE(CFcTOT,PGx,PGy)                            !   plot profile
  ! DO CFc=1,40                                            !   start loop over sample points
    ! PGx(CFc)=0.025*DBLE(CFc)*PGx(CFcTOT)                 !     compute abscissa of sample point
    ! PGy(CFc)=(0.210775E+24)*CFrho0/((1.+((0.308568E+19) &!     compute ordinate ...
              ! &*PGx(CFc)/CFw0)**2)**(DBLE(0.5*CFschP)))  !     .... of sample point
  ! ENDDO                                                  !   end loop over sample points
  ! CALL PGPT(40,PGx,PGy,0)                                !   plot sample points
  ! CALL PGEND                                             !   close PGPLOT
! ENDIF                                                    ! [ENDIF] plots made

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_Cyl1D_SchusterDensities
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_Cyl1D_InjectIsotropicAndTrack_SchusterScatteringOpacity&
&(CFw,CFw2,CFrho,CFsig)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine deals with the pure scattering case, with a Schuster density profile.

! It is given:
!   the boundary radius of the filament                   (CFwB);
!   the number of shells                                  (CFcTOT);

!   the boundary radii of the shells                      (CFw(0:CFc_TOT));
!   the boundary radii of the shells squared              (CFw2(0:CFc_tot));
!   the density in the shells                             (CFrho(1:CFcTOT));
!   the surface density through the spine                 (CFsig);
!   the mass scattering opacity coefficient               (DGkapM);
!   and the number of luminosity packets to be injected   (LPpTOT).

! It prints out:
!   the normalised angle-mean intensity in each shell     (RFj(0:CFcTOT)) --- should be  ~1;
!   its mean                                              (RFmuJ)         --- should be  ~1;
!   and its standard deviation                            (RFsdJ)         --- should be <<1.
USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! DECLARATIONS
REAL(KIND=8),INTENT(IN),DIMENSION(0:CFcTOT) :: CFw       ! boundary radii of shells
REAL(KIND=8),INTENT(IN),DIMENSION(0:CFcTOT) :: CFw2      ! squared boundary radii of shells
REAL(KIND=8),INTENT(IN),DIMENSION(1:CFcTOT) :: CFrho     ! densities in shells
REAL(KIND=8),INTENT(IN)                     :: CFsig     ! surface-density through spine (g/cm^2)
                                                         ! [] DRONES
INTEGER                                     :: CFc,CFcc  ! dummy shell IDs
REAL(KIND=8)                                :: CFtauTOT  ! total optical depth through filament
REAL(KIND=8)                                :: CFw2B     ! squared boundary radius (pc^2)
REAL(KIND=8)                                :: LPalfa    ! 'outwardness' of packet (pc)
REAL(KIND=8)                                :: LPbeta    ! squared tangent-distance (pc^2)
REAL(KIND=8),DIMENSION(1:3)                 :: LPe       ! direction of luminosity packet (unit vector)
REAL(KIND=8)                                :: LPe1122   ! e_x^2+e_y^2
INTEGER                                     :: LPnSCA    ! number of scatterings
INTEGER                                     :: LPp       ! dummy ID of luminosity packet
REAL(KIND=8),DIMENSION(1:3)                 :: LPr       ! position of luninosity packet
REAL(KIND=8)                                :: LPr1122   ! r_x^2+r_y^2
REAL(KIND=8)                                :: LPtau     ! optical depth
REAL(KIND=8)                                :: LPs       ! path-length of lum.pack. to shell boundary
REAL(KIND=8)                                :: LPsTAU    ! path-length of lum.pack. to scattering
REAL(KIND=8)                                :: LRD       ! linear random deviate on [0,1]
REAL(KIND=8),DIMENSION(1:CFcTOT)            :: RFj       ! angle-mean intensity in each shell
REAL(KIND=8)                                :: RFmuJ     ! volume-weighted mean angle-mean intensity
REAL(KIND=8)                                :: RFsdJ     ! volume-weighted stdev angle-mean intensity
REAL(KIND=8)                                :: alpha     ! 'outwardness' (pc)
REAL(KIND=8)                                :: beta      ! squared tangent-distance (pc^2)
REAL(KIND=8)                                :: phi       ! azimuthal angle
REAL(KIND=8)                                :: sintheta  ! sine of polar angle
INTEGER                                     :: NMeIN,NMeSC,NMtau
REAL(KIND=8),DIMENSION(1:3)                 :: MUeIN,SDeIN
REAL(KIND=8),DIMENSION(1:3)                 :: MUeSC,SDeSC
REAL(KIND=8)                                :: MUtau,SDtau

                                                         ! [] COMPUTATION
CFw2B=CFwB**2                                            ! compute W_B squared
RFj=0.                                                   ! set mean intensities to zero
LPnSCA=0                                                 ! set number of scatterings to zero
NMeIN=0;   MUeIN=0.;   SDeIN=0.                          ! *****
NMeSC=0;   MUeSC=0.;   SDeSC=0.                          ! *****
NMtau=0;   MUtau=0.;   SDtau=0.                          ! *****
DO LPp=1,LPpTOT                                          ! start loop over luminosity packets
  CALL RT_Cyl1D_InjectIsotropic                         &!   generate and inject ...
  &                  (LPr,LPr1122,LPe,LPtau)        !   ... a luminosity packet
  NMeIN=NMeIN+1;   MUeIN(1:3)=MUeIN(1:3)+ABS(LPe(1:3));   SDeIN(1:3)=SDeIN(1:3)+(LPe(1:3)**2) ! *****
  NMtau=NMtau+1;   MUtau=MUtau+LPtau;                     SDtau=SDtau+(LPtau**2)              ! *****
  CFc=CFcTOT                                             !   set shell ID to CFcTOT (outermost shell)
  DO WHILE (CFc<=CFcTOT)                                 !   keep going until packet exits filament
    LPsTAU=LPtau/(CFrho(CFc)*DGkapM)                     !     convert optical depth to distance
    CFcc=CFc                                             !     record ID of shell being entered
    LPe1122=(LPe(1)**2)+(LPe(2)**2)                      !     compute e_x^2+e_y^2
    alpha=((LPr(1)*LPe(1))+(LPr(2)*LPe(2)))/LPe1122      !     compute 'outwardness'
    IF (alpha>0.) THEN                                   !     [IF] travelling outward, [THEN]
      LPs=-alpha+                                       &!       compute .......
      &SQRT(alpha**2+((CFw2(CFc)-LPr1122)/LPe1122))      !       ... path length
      CFc=CFc+1                                          !       increase shell ID
    ELSE                                                 !     [ELSE] travelling inward, so
      beta=alpha**2+((CFw2(CFc-1)-LPr1122)/LPe1122)      !       compute beta, and
      IF (beta>0.) THEN                                  !       [IF] beta>0, [THEN] hits inner shell
        LPs=-alpha-SQRT(beta)                            !         compute path
        CFc=CFc-1                                        !         decrease shell ID
      ELSE                                               !       [ELSE] traverses shell, so
        LPs=-alpha+                                     &!         compute .......
        &SQRT(alpha**2+(CFw2(CFc)-LPr1122)/LPe1122)      !         ... path length
        CFc=CFc+1                                        !         increase shell ID
      ENDIF                                              !       [ENDIF] inward cases done
    ENDIF                                                !     [ENDIF] all cases done
    IF (LPsTAU<LPs) THEN                                 !     [IF] range of packet too small
      LPr(1:2)=LPr(1:2)+LPsTAU*LPe(1:2)                  !       advance position
      RFj(CFcc)=RFj(CFcc)+LPsTAU                         !     increment sum on intercept lengths
      CFc=CFcc                                           !       record that still in same shell
      CALL RT_ReDirectIsotropic(LPe,LPtau)               !       generate new direction and optical depth
      NMeSC=NMeSC+1;   MUeSC(1:3)=MUeSC(1:3)+ABS(LPe(1:3));   SDeSC(1:3)=SDeSC(1:3)+(LPe(1:3)**2) ! *****
      NMtau=NMtau+1;   MUtau=MUtau+LPtau;                     SDtau=SDtau+(LPtau**2)              ! *****
      LPnSCA=LPnSCA+1                                    !       increment number of scatterings
    ELSE                                                 !     [ELSE] exit shell
      LPr(1:2)=LPr(1:2)+LPs*LPe(1:2)                     !       advance position to shell boundary
      RFj(CFcc)=RFj(CFcc)+LPs                            !       increment sum of intercept lengths
      LPtau=LPtau-LPs*CFrho(CFcc)*DGkapM                 !       reduce remaining optical depth
    ENDIF                                                !     [ENDIF] continue
    LPr1122=LPr(1)**2+LPr(2)**2                          !     compute distane from spine
  ENDDO                                                  !   packet exits filament
ENDDO                                                    ! end loop over luminosity packets
alpha=CFwB/(2.*DBLE(LPpTOT))                             ! use alpha as normalisation coefficient
RFmuJ=0.                                                 ! set RFmuJ to zero
RFsdJ=0.                                                 ! set RFsdJ to zero
DO CFc=1,CFcTOT                                          ! start loop over shells
  RFj(CFc)=RFj(CFc)*alpha/(CFw2(CFc)-CFw2(CFc-1))        !   normalise intensity in cell
  RFmuJ=RFmuJ+RFj(CFc)                                   !   increment mu_J0 accumulator
  RFsdJ=RFsdJ+(RFj(CFc)**2)                              !   increment sigma_J0 accumulator
ENDDO                                                    ! end loop over shells
RFmuJ=RFmuJ/DBLE(CFcTOT)                                 ! compute RFmuJ
RFsdJ=SQRT((RFsdJ/DBLE(CFcTOT))-(RFmuJ**2))              ! compute RFsdJ
CFtauTOT=CFsig*DGkapM
WRITE (6,"(/,3X,'TEST CASE 2: SCHUSTER SCATTERING OPACITY, SHELL ANGLE-MEAN INTENSITIES (RFj)')")
WRITE (6,"(3X,'RFj:',4X,5F10.5,4X,'(innermost five shells)')") RFj(1:5)
WRITE (6,"(3X,'RFj:',4X,5F10.5,4X,'(outermost five shells)')") RFj(CFcTOT-4:CFcTOT)
WRITE (6,"(3X,'Shell-mean of angle-mean intensity, RFmuJ:',6X,F10.7)") RFmuJ
WRITE (6,"(3X,'Shell-SD of angle-mean intensity,   RFsdJ:',6X,F10.7)") RFsdJ
WRITE (6,"(3X,'Number of shells:            ',18X,I11)") CFcTOT
WRITE (6,"(3X,'Number of luminosity packets:',18X,I11)") LPpTOT
WRITE (6,"(3X,'Number of scatterings:       ',18X,I11)") LPnSCA
WRITE (6,"(3X,'Total optical depth right through filament:',5X,F10.3,/)") CFtauTOT

MUeIN(1:3)=MUeIN(1:3)/DBLE(NMeIN)                         ! compute mean of injection directions
SDeIN(1:3)=SQRT((SDeIN(1:3)/DBLE(NMeIN))-(MUeIN(1:3)**2)) ! compute StDev of injection directions
WRITE (6,"(3X,'MEAN and StDev OF INJECTION DIRECTION:',5X,3(3X,2F8.5),5X,I11)") &
&MUeIN(1),SDeIN(1),MUeIN(2),SDeIN(2),MUeIN(3),SDeIN(3),NMeIN
WRITE (6,"(31X,'SHOULD BE:',9X,'0.66667 0.23570    0.42441 0.26433    0.42441 0.26433')")
MUeSC(1:3)=MUeSC(1:3)/DBLE(NMeSC)                         ! compute mean of scattering directions
SDeSC(1:3)=SQRT((SDeSC(1:3)/DBLE(NMeSC))-(MUeSC(1:3)**2)) ! compute StDev of scattering directions
WRITE (6,"(3X,'MEAN and StDev OF SCATTERING DIRECTION:',4X,3(3X,2F8.5),5X,I11)") &
&MUeSC(1),SDeSC(1),MUeSC(2),SDeSC(2),MUeSC(3),SDeSC(3),NMeSC
WRITE (6,"(32X,'SHOULD BE:',8X,'0.50000 0.28868    0.50000 0.28868    0.50000 0.28868')")
MUtau=MUtau/DBLE(NMtau)                                   ! compute mean of scattering directions
SDtau=SQRT((SDtau/DBLE(NMtau))-(MUtau**2))                ! compute StDev of scattering directions
WRITE (6,"(3X,'MEAN and StDev OF OPTICAL DEPTH:',14X,2F8.5,43X,I11)") &
&MUtau,SDtau,NMtau
WRITE (6,"(25X,'SHOULD BE:',15X,'1.00000 1.00000',/)")

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_Cyl1D_InjectIsotropicAndTrack_SchusterScatteringOpacity
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_Cyl1DSchuster_DetailedBalance(CFw, &
&CFw2,CFrho,CFmu,CFsig,cfT,cfL,teT,WLlTOT,WLlam, &
&WLdlam,WLchi,WLalb,WTpBB,WTlBBlo,WTlBBup,WTpMB,WTlMBlo,&
&WTlMBup,teLMmb,WTpDM,WTlDMlo,WTlDMup,teLMTdm, &
&RFjLAM)

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine deals with the pure scattering case, with a Schuster density profile.

! It is given:
!   the boundary radius of the filament                               (CFwB);
!   the number of shells                                              (CFcTOT);
!   the boundary radii of the shells                                  (CFw(0:CFc_TOT));
!   the boundary radii of the shells squared                          (CFw2(0:CFc_tot));
!   the volume-densities in the shells                                (CFrho(1:CFcTOT));
!   the line-densities of the shells                                  (CFmu(1:CFcTOT));
!   the number of discrete temperatures                               (TEkTOT);
!   the discrete temperatures                                         (teT(0:TEkTOT));
!   the temperature-ID of the background blackbody radiation field    (BGkBB);
!   the dilution factor of the background blackbody radiation field   (BGfBB);
!   the number of discrete wavelengths                                (WLlTOT);
!   the discrete wavelengths                                          (WLlam(1:WLlTOT));
!   the corresponding wavelength intervals                            (WLdlam(1:WLlTOT));
!   the corresponding extinction opacities                            (WLchi(1:WLlTOT));
!   the corresponding albedos                                         (WLalb(1:WLlTOT));
!   the BB emission probability for [lam(l),dlam(l)] at T=T(k)        (WTpBB(0:WLlTOT,1:TEkTOT));
!   the ID of the longest wavelength with WTpBB(ID,k)<=(l-1)/lTOT     (WTlBBlo(1:WLlTOT,1:TEkTOT));
!   the ID of the shortest wavelength with WTpBB(ID,k)>=l/lTOT        (WTlBBup(1:WLlTOT,1:TEkTOT));
!   the MB emission probability for [lam(l),dlam(l)] at T=T(k)        (WTpMB(0:WLlTOT,1:TEkTOT));
!   the ID of the longest wavelength with WTpMB(ID,k)<=(l-1)/lTOT     (WTlMBlo(1:WLlTOT,1:TEkTOT));
!   the ID of the shortest wavelength with WTpMB(ID,k)>=l/lTOT        (WTlMBup(1:WLlTOT,1:TEkTOT));
!   the MB luminosity per unit mass at each temperature               (teLMmb(0:TEkTOT));
!   the DM emission probability for [lam(l),dlam(l)] at T=T(k)        (WTpDM(0:WLlTOT,1:TEkTOT));
!   the ID of the longest wavelength with WTpDM(ID,k)<=(l-1)/lTOT     (WTlDMlo(1:WLlTOT,1:TEkTOT));
!   the ID of the shortest wavelength with WTpDM(ID,k)>=l/lTOT        (WTlDMup(1:WLlTOT,1:TEkTOT));
!   the DM luminosity per unit mass per unit temperature              (teLMTdm(0:TEkTOT));
!   the number of reference probabilities                             (PRnTOT);
!   and the number of luminosity packets to be injected               (LPpTOT).

! It returns:
!   the mean intensities at each wavelength and in each cell          (RFjLAM(1:WLlTOT,1:CFcTOT));
!   the temperature of each cell                                      (cfT(1:CFcTOT));
!   the line-luninosity absorbed by each cell                         (cfL(1:CFcTOT)).

! It prints out
!   the cell-mean of the mean intensity (at selected wavelengths)     (RFmuJlam(1:WLlTOT));
!   and the cell-SD of the mean intensity (at selected wavelengths)   (RFsdJlam(1:WLlTOT));
!   the cell-mean of the temperature                                  (CFmuT(1:CFcTOT));
!   the cell-SD of the temperature                                    (CFsdT(1:CFcTOT));

USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! DECLARATIONS
REAL(KIND=8),INTENT(IN),DIMENSION(0:CFcTOT) :: CFw       ! boundary radii of shells (cm)
REAL(KIND=8),INTENT(IN),DIMENSION(0:CFcTOT) :: CFw2      ! squared boundary radii of shells (cm^2)
REAL(KIND=8),INTENT(IN),DIMENSION(1:CFcTOT) :: CFrho     ! volume-densities in shells (g/cm^3)
REAL(KIND=8),INTENT(IN),DIMENSION(1:CFcTOT) :: CFmu      ! line-densities in shells (g/cm)
REAL(KIND=8),INTENT(IN)                     :: CFsig     ! surface-density through spine (g/cm^2)
REAL(KIND=8),INTENT(IN),DIMENSION(0:TEkTOT) :: teT       ! discrete temperatures
INTEGER,     INTENT(IN)                     :: WLlTOT    ! the number of discrete wavelengths
REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLlam     ! the discrete wavelengths
REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLdlam    ! the corresponding wavelength intervals
REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLchi     ! the corresponding extinction opacities
REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT) :: WLalb     ! the corresponding albedos
REAL(KIND=8),INTENT(IN),                                &! BB emission probabilities ........
               DIMENSION(0:WLlTOT,0:TEkTOT) :: WTpBB     ! ... for [lam(l),dlam(l)] at T=T(k)
INTEGER,     INTENT(IN),                                &! ID of longest wavelength with ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlBBlo   ! ........ WTpBB(ID,k)<=(l-1)/l_TOT
INTEGER,     INTENT(IN),                                &! ID of shortest wavelength ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlBBup   ! ... with WTpBB(ID,k)>=l/l_TOT
REAL(KIND=8),INTENT(IN),                                &! MB emission probabilities ........
               DIMENSION(0:WLlTOT,0:TEkTOT) :: WTpMB     ! ... for [lam(l),dlam(l)] at T=T(k)
INTEGER,     INTENT(IN),                                &! ID of longest wavelength with ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlMBlo   ! ........ WTpMB(ID,k)<=(l-1)/l_TOT
INTEGER,     INTENT(IN),                                &! ID of shortest wavelength ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlMBup   ! ... with WTpMB(ID,k)>=l/l_TOT
REAL(KIND=8),INTENT(IN),DIMENSION(0:TEkTOT) :: teLMmb    ! MB luminosity per unit mass
REAL(KIND=8),INTENT(IN),                                &! DM emission probabilities ........
               DIMENSION(0:WLlTOT,0:TEkTOT) :: WTpDM     ! ... for [lam(l),dlam(l)] at T=T(k)
INTEGER,     INTENT(IN),                                &! ID of longest wavelength with ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlDMlo   ! ........ WTpDM(ID,k)<=(l-1)/l_TOT
INTEGER,     INTENT(IN),                                &! ID of shortest wavelength ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlDMup   ! ... with WTpDM(ID,k)>=l/l_TOT
REAL(KIND=8),INTENT(IN),DIMENSION(0:TEkTOT) :: teLMTdm   ! MB luminosity per unit mass

REAL(KIND=8),INTENT(OUT),                               &! the mean intensity in cell CFc, in wavelength ...
     DIMENSION(1:WLlTOT,1:CFcTOT)           :: RFjLAM    ! .... interval [WLlam(WLl),WLlam(WLl)+WLdlam(WLl)]
REAL(KIND=8),INTENT(OUT),DIMENSION(1:CFcTOT):: cfT       ! the temperature in each cell
REAL(KIND=8),INTENT(OUT),DIMENSION(1:CFcTOT):: cfL       ! the line-luminosity absorbed by each cell

INTEGER                                     :: CFc,CFcc  ! dummy shell IDs
!REAL(KIND=8)                                :: CFtauTOT  ! total optical depth through filament
INTEGER,DIMENSION(0:CFcTOT)                 :: CFk       ! ID of temperature just above cell temperature
REAL(KIND=8),DIMENSION(1:CFcTOT)            :: cfLgo     ! cell luminosity above which temperature is updated
REAL(KIND=8)                                :: CFw2B     ! squared boundary radius (pc^2)
REAL(KIND=8)                                :: LPalb     ! albedo of luminosity packet
REAL(KIND=8)                                :: LPalfa    ! 'outwardness' of packet (pc)
REAL(KIND=8)                                :: LPbeta    ! squared tangent-distance (pc^2)
REAL(KIND=8)                                :: LPchi     ! extinction opacity of luminosity packet
REAL(KIND=8)                                :: LPdeltaL    ! line-luminosity of a luminosity packet
REAL(KIND=8),DIMENSION(1:3)                 :: LPe       ! direction of luminosity packet (unit vector)
REAL(KIND=8)                                :: LPe1122   ! e_x^2+e_y^2
INTEGER                                     :: LPl       ! wavelength-ID of luminosity packet
INTEGER                                     :: LPnSCA    ! number of scatterings
INTEGER                                     :: LPp       ! dummy ID of luminosity packet
REAL(KIND=8),DIMENSION(1:3)                 :: LPr       ! position of luninosity packet
REAL(KIND=8)                                :: LPr1122   ! r_x^2+r_y^2
REAL(KIND=8)                                :: LPtau     ! optical depth
REAL(KIND=8)                                :: LPs       ! path-length of lum.pack. to shell boundary
REAL(KIND=8)                                :: LPsTAU    ! path-length of lum.pack. to scattering
REAL(KIND=8)                                :: LRD       ! linear random deviate on [0,1]
!REAL(KIND=8),DIMENSION(1:CFcTOT)            :: RFj       ! angle-mean intensity in each shell
!REAL(KIND=8)                                :: RFmuJ     ! volume-weighted mean angle-mean intensity
!REAL(KIND=8)                                :: RFsdJ     ! volume-weighted stdev angle-mean intensity
!REAL(KIND=8)                                :: alpha     ! 'outwardness' (pc)
!REAL(KIND=8)                                :: beta      ! squared tangent-distance (pc^2)
!REAL(KIND=8)                                :: phi       ! azimuthal angle
!REAL(KIND=8)                                :: sintheta  ! sine of polar angle
INTEGER                                     :: TEk        ! dummy temperature-ID
INTEGER                                     :: NMeIN,NMeSC,NMtau
REAL(KIND=8),DIMENSION(1:3)                 :: MUeIN,SDeIN
REAL(KIND=8),DIMENSION(1:3)                 :: MUeSC,SDeSC
REAL(KIND=8)                                :: MUtau,SDtau
REAL(KIND=8)                                :: TeRatio       ! weighting factor

Real(kind=8)                                ::  rfI
Real(kind=8)                                ::  rfH         !Heating term from Lucy 1999 method
REAL(KIND=8),DIMENSION(1:CFcTOT)            ::  rfTemp      !Lucy line addition temp
integer(KIND=8),DIMENSION(1:CFcTOT)         ::  LPnScatter, LPnAbsorb
Character(len=50)                           :: CellTempFilename = &
                                              & "detailed-balance-RT_cell_temperatures.csv"
integer                                     ::  readcheck
integer                                     :: i
Real(kind=8)                                :: LmRatio

character(len=40)                           :: tmpFilename = "db_N-a_S-i_M-c_V-c_T-e_P-t_W-f.csv"
Real(kind=8)                                ::CFv
integer(kind=4)                             :: LPlFixed     !Fixed wavelength index at roughly below wavelength value
Real(kind=8)                                :: WLfixed = 300.d0 !Fixed 300 microns wavelength


Real(kind=8),DIMENSION(1:CFcTOT)            :: errLucy,errStnd
Integer,DIMENSION(1:CFcTOT)                 :: RFinter
Real(kind=8)                                :: chiBar, lamMax, tauBar




LPdeltaL=((0.35628897E-3)*CFwB*BGfBB*teT(BGkBB)**4)        &! compute line-luminosity of ...
                                         &/DBLE(LPpTOT)  ! ... a single luminosity packet   (0.35628897E+03)

CFw2B=CFwB**2                                            ! compute W_B squared
RFjLAM=0.                                                ! set mean intensities to zero
cfT=teT(BGkGO)!teT(0)
CFk=BGkGO!1

cfL=0.                                                   ! set line-luminosity absorbed by each cell to zero
cfLgo(1:CFcTOT)=CFmu(1:CFcTOT)*teLMmb(BGkGO)


LPnScatter = 0
LPnAbsorb = 0
RFinter = 0


!Diagnostics print statements
!print*," "
!print*,"teT(BGkBB)",teT(BGkBB)
!print*,"teT(BGkGO) cfLgo temp = ", teT(BGkGO), " K"
!print*,"cell initialised temp cfT(1)",cfT(1)," K"
!print*," "
!print*,"cfLgo(1)",cfLgo(1)
!print*,"LPdeltaL",LPdeltaL
!print*," "

!Sanity Check: If this condition isn't met we will overshoot
! cfLgo when increasing temperature, and temperature errors will occur!!
IF(cfLgo(1).lt.LPdeltaL) then
print*,"WARNING[@Detailed-Balance]: Luminosity condition FALSE!!! CFlGO < LPdeltaL! Please Adjust BGkGO!"
ENDIF

LPnSCA=0                                                 ! set number of scatterings to zero
! NMeIN=0;   MUeIN=0.;   SDeIN=0.                          ! *****
! NMeSC=0;   MUeSC=0.;   SDeSC=0.                          ! *****
! NMtau=0;   MUtau=0.;   SDtau=0.                          ! *****
DO LPp=1,LPpTOT                                          ! start loop over luminosity packets
  CALL RT_Cyl1D_InjectIsotropic                         &!   generate and inject ...
  &                  (LPr,LPr1122,LPe,LPtau)        !   ... a luminosity packet
  ! NMeIN=NMeIN+1;   MUeIN(1:3)=MUeIN(1:3)+ABS(LPe(1:3));   SDeIN(1:3)=SDeIN(1:3)+(LPe(1:3)**2) ! *****
  ! NMtau=NMtau+1;   MUtau=MUtau+LPtau;                     SDtau=SDtau+(LPtau**2)              ! *****

  CFc=CFcTOT                                             !   set shell ID to CFcTOT (outermost shell)

  CALL RT_LumPack_BB(BGkBB,WLlTOT,WTpBB,WTlBBlo,WTlBBup,LPl)

  LPchi=WLchi(LPl)                                       !   record extinction opacity of luminosity packet
  LPalb=WLalb(LPl)                                       !   albedo of luminosity packet

  DO WHILE (CFc<=CFcTOT)                                 !   keep going until packet exits filament

    LPsTAU=LPtau/(CFrho(CFc)*LPchi)                      !     convert optical depth to distance
    CFcc=CFc                                             !     record ID of shell being entered
    LPe1122=(LPe(1)**2)+(LPe(2)**2)                      !     compute e_x^2+e_y^2
    LPalfa=((LPr(1)*LPe(1))+(LPr(2)*LPe(2)))/LPe1122     !     compute 'outwardness'
    IF (LPalfa>0.) THEN                                  !     [IF] travelling outward, [THEN]
      LPs=-LPalfa+                                      &!       compute .......
      &SQRT(LPalfa**2+((CFw2(CFc)-LPr1122)/LPe1122))     !       ... path length
      CFc=CFc+1                                          !       increase shell ID

    ELSE                                                 !     [ELSE] travelling inward, so
      LPbeta=LPalfa**2+((CFw2(CFc-1)-LPr1122)/LPe1122)   !       compute beta, and
      IF (LPbeta>0.) THEN                                !       [IF] LPbeta>0, [THEN] hits inner shell
        LPs=-LPalfa-SQRT(LPbeta)                         !         compute path
        CFc=CFc-1                                        !         decrease shell ID

      ELSE                                               !       [ELSE] traverses shell, so
        LPs=-LPalfa+                                    &!         compute .......
        &SQRT(LPalfa**2+(CFw2(CFc)-LPr1122)/LPe1122)     !         ... path length
        CFc=CFc+1                                        !         increase shell ID

      ENDIF                                              !       [ENDIF] inward cases done
    ENDIF                                                !     [ENDIF] all cases done

    IF (LPsTAU<LPs) THEN                                 !     [IF] range of packet too small
      LPr(1:2)=LPr(1:2)+LPsTAU*LPe(1:2)                  !       advance position
      CFc=CFcc                                           !       record that still in same shell

      RFjLAM(LPl,CFc)=RFjLAM(LPl,CFc)+LPsTAU             !     increment sum on intercept lengths
      RFinter(CFc) = RFinter(CFc) + 1
      LPnScatter(CFcc) = LPnScatter(CFcc) + 1
      LPnSCA=LPnSCA+1                                    !       increment number of scatterings

      CALL RT_ReDirectIsotropic(LPe,LPtau)               !       generate new direction and optical depth
      ! NMeSC=NMeSC+1;   MUeSC(1:3)=MUeSC(1:3)+ABS(LPe(1:3));   SDeSC(1:3)=SDeSC(1:3)+(LPe(1:3)**2) ! *****
      ! NMtau=NMtau+1;   MUtau=MUtau+LPtau;                     SDtau=SDtau+(LPtau**2)              ! *****

      CALL RANDOM_NUMBER(LRD)
      IF (LRD>LPalb) THEN

        LPnAbsorb(CFc) = LPnAbsorb(CFc) + 1

        cfL(CFc)=cfL(CFc)+LPdeltaL
        IF (cfL(CFc)<cfLgo(CFc)) THEN
          !CALL RT_LumPack_MB(0,TEkTOT,PRnTOT,WLlTOT, &
          !& WTpMB,WTlMBlo,WTlMBup,LPl)
          CALL RT_LumPack_MB(BGkGO,WLlTOT,WTpMB,WTlMBlo,WTlMBup,LPl)
        ELSE
        !- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        !Temperature Checks:
        !If temperature of cell has increased into next increment
        !in last loop, then increase temp. counter.
          Do while(cfT(CFc)>teT(CFk(CFc)))
            CFk(CFc)=CFk(CFc)+1
          enddo
        !TeRatio: Checks the closeness of continuous cell temp. (cfT) to
        !boundaries of discrete temp.. TeRatio -> 1 as cfT -> teT(CFk(CFc-1))
        ! i.e. the lower boundary of temperature.
        !TeRatio -> 0 as cfT -> teT(CFk(CFc)) i.e. upper bound
        !Therefore if LRD<TeRatio the temperature will randomly decrease with
        !high probability if cfT ~ teT(CFk(CFc)).

          TEk=CFk(CFc)
          TeRatio=(teT(TEk)-cfT(CFc))/(teT(TEk)-teT(TEk-1))
          CALL RANDOM_NUMBER(LRD)
          IF (LRD.le.TeRatio) TEk =TEk-1
        !- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        !Increase Cell temperature. See documentation
          cfT(CFc)=cfT(CFc)+(LPdeltaL/(CFmu(CFc)*teLMTdm(TEk)))
        !- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        !Randomly call new wavelength from cell Temp based DMBB.
          CALL RT_LumPack_DM(TEk,WLlTOT, WTpDM,WTlDMlo,WTlDMup,LPl)
        !- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        ENDIF

        LPchi=WLchi(LPl)
        LPalb=WLalb(LPl)
      ENDIF
    ELSE
      LPr(1:2)=LPr(1:2)+LPs*LPe(1:2)                     !       advance position to shell boundary
      RFjLAM(LPl,CFcc)=RFjLAM(LPl,CFcc)+LPs              !       increment sum of intercept lengths => This can be used for Lucy 1999 method of mean intensity tracking
      LPtau=LPtau-LPs*CFrho(CFcc)*LPchi                  !       reduce remaining optical depth
    ENDIF
    LPr1122=LPr(1)**2+LPr(2)**2                          !     compute distance from spine

  ENDDO                                                  !   packet exits filament

  If(mod(dble(LPp),(dble(LPpTOT)/10.d0)).eq. 0.d0) then
    print("(A7,1x,F7.2,1x,A3)"),"@DB: ", (dble(LPp)/dble(LPpTOT))*100.d0," %"
  endif

ENDDO                                                    ! end loop over luminosity packets

!------------------------------------------------------------------------------
do CFc=1,CFcTOT,1
    rfI = 0.d0
    do LPl=1,WLlTOT,1
        rfI = rfI+(WLchi(LPl)*(1.d0-WLalb(LPl))*RFjLAM(LPl,CFc))    !CGS. Lucy 1999
    enddo
    rfH=(LPdeltaL*rfI)/((CFw2(CFc)-CFw2(CFc-1))*(3.14159265359d0))
    do TEK=1,TEkTOT,1
        If(teLMmb(TEk).gt.rfH)then
            LmRatio=(rfH - teLMmb(TEk-1))/(teLMmb(TEk)-teLMmb(TEk-1))
            rfTemp(CFc) = teT(TEk-1) + (teT(TEk)-teT(TEk-1))*LmRatio
            EXIT
        endif
    enddo
    errLucy(CFc) = rfTemp(CFc)*(1.d0/(SQRT(dble(RFinter(CFc)))))
    errStnd(CFC) = cfT(CFc)*(1.d0/(SQRT(dble(LPnAbsorb(CFc)))))
enddo

!print*,
!print*,"Lucy intercept added temp: rfTemp(LPl,CFc):"
!do CFc=1,CFcTOT,1
!   IF(rfTemp(CFc).ne.0.d0) THEN
!       print*,"CFc",CFc
!       print*,"temp",rfTemp(CFc)
!   ENDIF
!enddo
!print*," "

lamMax = (0.288*(1.d4))/teT(BGkBB)

do i=1,WLlTOT,1
  If(WLlam(i) .ge. lamMax) THEN
    chiBar = WLchi(i)
    EXIT
  endif
enddo

tauBar = chiBar*CFsig



If (DBTestFlag == 1) then
  print*,"-+-+-+-+-"
  !print*,"Ant's Diagnostics:"
  !print*," "
  !Print*,"CFcTOT,CFwB,CFw(CFcTOT)"
  !WRITE (*,*) CFcTOT,CFwB,CFw(CFcTOT)
  !Print*,"TEkTOT,BGfBB,teT(BGkBB)"
  !WRITE (*,*) TEkTOT,BGfBB,teT(BGkBB)
  !Print*,"WLlTOT,WLlTOT/3,WLlam(WLlTOT/3),WLchi(WLlTOT/3),WLalb(WLlTOT/3)"
  !WRITE (*,*) WLlTOT,WLlTOT/3,WLlam(WLlTOT/3),WLchi(WLlTOT/3),WLalb(WLlTOT/3)
  !print*,"LPpTOT,LPpTOT"
  !WRITE (*,*) LPpTOT,LPpTOT

  print*," "
  print*,"ATH diag:"
  print*," "
  print*,"The temperature of each cell:"
  print*,"cfT(1),cfT(CFcTOT/2),cfT(CFcTOT)"
  print*,cfT(1),cfT(CFcTOT/2),cfT(CFcTOT)
  print*," "
  print*,"LPnSCA/LPpTOT",dble(LPnSCA)/dble(LPpTOT)
  print*,"CFsig",CFsig
  print*,"chiBar",chiBar
  print*,"lamMax",lamMax
  print*,"tauBar",tauBar
  print*,"LPnScatter(1), LPnScatter(CFcTOT/2), LPnScatter(CFcTOT)"
  print*,LPnScatter(1), LPnScatter(CFcTOT/2), LPnScatter(CFcTOT)
  print*," "
  print*,"LPnAbsorb(1), LPnAbsorb(CFcTOT/2), LPnAbsorb(CFcTOT)"
  print*,LPnAbsorb(1), LPnAbsorb(CFcTOT/2), LPnAbsorb(CFcTOT)
  print*," "



!-------------------------------------------------------------------------------
  !Find WLl nearest to WLfixed wavelength
  do i=1,WLlTOT
    if (WLlam(i) .le. WLfixed) THEN
        LPlFixed = i
    ELSE
        EXIT
    ENDIF
  enddo

  open(1, file=adjustl(trim(tmpFilename)))

  print*, "Writing: "
  print*, tmpFilename

  write(1,"(7(A3,1x))") (/"N-a","S-i","M-c","V-c","T-e","P-t","W-f"/)

  do i=1,CFcTOT
    CFv = 3.14159274d0 *(CFw2B)*(((dble(i)**2) - (dble(i-1)**2))/dble(CFcTOT**2))
    CFv = CFv*((0.324078E-18)**2)
    if (i==1) THEN
      write(1,"(7(F20.8,1x))") (/dble(LPnAbsorb(i)),dble(RFjLAM(LPlFixed,i))*(0.324078E-18),CFmu(i)*(0.155129E-15) &
      & ,CFv,teT(BGkBB),dble(LPpTOT),dble(WLlam(LPlFixed))/)
    ELSE
      write(1,"(4(F20.8,1x))") (/dble(LPnAbsorb(i)),dble(RFjLAM(LPlFixed,i))*(0.324078E-18),CFmu(i)*(0.155129E-15) &
      & ,CFv/)
    endif
  enddo
  close(1)
  print*,"Written!"
  print*,""
!-------------------------------------------------------------------------------

  print*,"Saving Temperatures versus cells:"
  print*,"Filename = ", CellTempFilename
  OPEN(1,file=trim(adjustl(CellTempFilename)),&
  & iostat=readcheck)
  IF (readcheck.ne.0) print*,"WARNING: [@detailed-balance Temp print] failed iostat check!"
  WRITE(1,"(8(A4,1x))") (/"cell","temp","lucy","ErrL","ErrS","EquT","PTOT"/)
  do i = 1, CFcTOT
      IF (i.eq.1) then
          WRITE(1,"(6(F18.10,1x),F10.0)") &
          &(/dble(i),cfT(i),rfTemp(i),errLucy(i),errStnd(i),teT(BGkBB),dble(LPpTOT)/)
      ELSE
          WRITE(1,"(5(F18.10,1x))") &
          &(/dble(i),cfT(i),rfTemp(i),errLucy(i),errStnd(i)/)
      ENDIF
  enddo
  close(1)
ENDIF
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_Cyl1DSchuster_DetailedBalance
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!==============================================================================!
!                       PyTest UNIT TESTED Subroutines
!==============================================================================!
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_Cyl1D_InjectIsotropic(LPr,LPr1122,LPe,LPtau)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine is given:
!   the radius of the domain boundary               (CFwB).

! It returns:
!   the injection position                          (LPr(1:3));
!   the injection distance from the spine           (LPr1122);
!   random injection direction-cosines              (LPe(1:3));
!   and a random optical depth                      (LPtau).

USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! DECLARATIONS:
REAL(KIND=8),INTENT(OUT),DIMENSION(1:3)     :: LPe       ! random injection direction-cosines
REAL(KIND=8),INTENT(OUT),DIMENSION(1:3)     :: LPr       ! injection position
REAL(KIND=8),INTENT(OUT)                    :: LPr1122   ! injection distance from spine
REAL(KIND=8),INTENT(OUT)                    :: LPtau     ! random optical depth
REAL(KIND=8)                                :: LRD       ! linear random deviate
REAL(KIND=8)                                :: phi       ! azimuthal angle (phi)
REAL(KIND=8)                                :: sintheta  ! sine of polar angle (theta)
                                                         ! COMPUTATIONS:
LPr(1)=-CFwB                                             ! set x-coordinate of injection position
LPr(2:3)=0.                                              ! set y and z-coordinates of injection position
LPr1122=CFwB**2                                          ! compute injection distance from spine
CALL RANDOM_NUMBER(LRD)                                  ! generate linear random deviate on [0,1]
LPe(1)=SQRT(LRD)                                         ! compute LPe(1)
sintheta=SQRT(1.-LRD)                                    ! compute sintheta
CALL RANDOM_NUMBER(LRD)                                  ! generate linear random deviate on [0,1]
phi=6.2831853*LRD                                        ! compute phi
LPe(2)=sintheta*COS(phi)                                 ! compute LPe(2)
LPe(3)=sintheta*SIN(phi)                                 ! compute LPe(3)
CALL RANDOM_NUMBER(LRD)                                  ! generate linear random deviate on [0,1]
LPtau=-LOG(LRD)                                          ! compute optical depth
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_Cyl1D_InjectIsotropic
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_ReDirectIsotropic(LPe,LPtau)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine returns:
!   isotropic random direction-cosines   (LPe(1:3));
!   and a random optical depth           (LPtau).

USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! DECLARATIONS:
REAL(KIND=8),INTENT(OUT),DIMENSION(1:3)     :: LPe       ! isotropic random direction-cosines
REAL(KIND=8),INTENT(OUT)                    :: LPtau     ! random optical depth
REAL(KIND=8)                                :: LRD       ! linear random deviate
REAL(KIND=8)                                :: phi       ! azimuthal angle (phi)
REAL(KIND=8)                                :: sintheta  ! sine of polar angle (theta)

! COMPUTATIONS:
CALL RANDOM_NUMBER(LRD)                                  ! generate linear random deviate on [0,1]
LPe(1)=2.*LRD-1.                                         ! compute LPe(1)
sintheta=SQRT(1.-LPe(1)**2)                              ! compute sintheta


CALL RANDOM_NUMBER(LRD)                                  ! generate linear random deviate on [0,1]
phi=6.2831853*LRD                                        ! compute phi
LPe(2)=sintheta*COS(phi)                                 ! compute LPe(2)
LPe(3)=sintheta*SIN(phi)                                 ! compute LPe(3)
CALL RANDOM_NUMBER(LRD)                                  ! generate linear random deviate on [0,1]
LPtau=-LOG(LRD)                                          ! compute optical depth


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_ReDirectIsotropic
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_Cyl1D_LinearShellSpacing(CFw,CFw2)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine is given:
!   the boundary radius of the filament             (CFwB);
!   the number of shells (aka cells)                (CFcTOT);
!   and a flag for diagnostic printout              (CFlist).

! It returns :
!   the boundary radii of the shells                (CFw(0:CFcTOT));
!   and the squared boundary radii of the shells    (CFw2(0:CFcTOT)).

USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! [] DECLARATIONS
REAL(KIND=8),INTENT(OUT),DIMENSION(0:CFcTOT):: CFw       ! shell boundary radius (in cm)
REAL(KIND=8),INTENT(OUT),DIMENSION(0:CFcTOT):: CFw2      ! squarred shell boundary radius (in cm^2)
                                                         ! [] DRONES
INTEGER                                     :: CFc       ! dummy ID of shell
REAL(KIND=8)                                :: CFdw      ! shell width


CFdw=CFwB/DBLE(CFcTOT)                                   ! compute shell width
DO CFc=0,CFcTOT                                          ! start loop over shells
  CFw(CFc)=CFdw*DBLE(CFc)                                !   compute outer shell radius
  CFw2(CFc)=CFw(CFc)**2                                  !   compute squared outer shell radius
ENDDO                                                    ! end loop over shells

IF (CFlist==1) THEN
  WRITE (6,"(/,3X,'SHELL BOUNDARIES, CFcTOT =',I4)") CFcTOT
  WRITE (6,"(3X,'CFw/(pc):',6X,5E11.4)") CFw(1:CFcTOT)/(3.086E+18)
  WRITE (6,"(3X,'CFw2/(pc^2):',3X,5E11.4)") CFw2(1:CFcTOT)/(9.521E+36)
  WRITE (*,*) ' '
ENDIF

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_Cyl1D_LinearShellSpacing
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_Temperatures(teT)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine computes the discrete temperatures.
! It is given:
!   the number of temperatures needed      (TEkTOT);
!   the minimum temperature                (teTmin);
!   the maximum temperature                (teTmax);
!   and a flag to trigger printout         (TElist).

! It returns:
!   the discrete temperatures              (teT(0:TEkTOT)),
!   where T(0)=teTmin and T(TEkTOT)=teTmax.

USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! [] DECLARATIONS
REAL(KIND=8),INTENT(OUT),DIMENSION(0:TEkTOT):: teT       ! discrete temperatures
INTEGER                                     :: TEk       ! dummy temperature ID
REAL(KIND=8)                                :: kTOTinv   ! REAL(1/kTOT)
REAL(KIND=8)                                :: logTmax   ! LOG10(Tmax)
REAL(KIND=8)                                :: logTmin   ! LOG10(Tmin)

! [] COMPUTE TEMPERATURES
kTOTinv=1./DBLE(TEkTOT)                                  ! compute 1/(k_TOT-1)
logTmin=LOG10(teTmin)                                    ! compute LOG10[Tmin]
logTmax=LOG10(teTmax)                                    ! compute LOG10[Tmax]
DO TEk=0,TEkTOT                                          ! start loop over temperatures
teT(TEk)=10.**(kTOTinv*(DBLE(TEkTOT-TEk)*logTmin      &!   compute .................
&+DBLE(TEk)*logTmax))  !   ... discrete temperatures
ENDDO                                                    ! end loop over temperatures

! [] CONDITIONAL DIAGNOSTIC PRINTOUT
IF (TElist==1) THEN                                      ! [IF] printout sanctioned, [THEN]
print*," "
PRINT*,"Temperature Diagnostics: "
PRINT*,"teT(TEk)"
WRITE (6,"(F11.3)") teT(0:TEkTOT)
ENDIF                                                        ! [ENDIF]

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_Temperatures
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_LumPack_BB(TEk,WLlTOT,WTpBB,WTlBBlo,WTlBBup,WLlEM)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine is given:
!   a temperature ID                                                 (TEk);
!   the number of temperatures                                       (TEkTOT);
!   the number of reference probabilities                            (PRnTOT);
!   the number of wavelengths                                        (WLlTOT);
!   the BB emission probability for [lam(l),dlam(l)] at T=T(k)       (WTpBB(0:WLlTOT,1:TEkTOT));
!   the ID of the longest wavelength with WTpBB(ID,k)<=(l-1)/lTOT    (WTlBBlo(1:WLlTOT,1:TEkTOT));
!   and the ID of the shortest wavelength with WTpBB(ID,k)>=l/lTOT   (WTlBBup(1:WLlTOT,1:TEkTOT)).

! It returns:
!   the wavelength ID of the emitted luminosity packet               (WLlEM).
USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! [] DECLARATIONS
INTEGER,     INTENT(IN)                     :: TEk       ! temperature ID
INTEGER,     INTENT(IN)                     :: WLlTOT    ! number of wavelengths
REAL(KIND=8),INTENT(IN),                                &! BB emission probabilities ........
               DIMENSION(0:WLlTOT,0:TEkTOT) :: WTpBB     ! ... for [lam(l),dlam(l)] at T=T(k)
INTEGER,     INTENT(IN),                                &! ID of longest wavelength with ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlBBlo   ! ........ WTpBB(ID,k)<=(l-1)/l_TOT
INTEGER,     INTENT(IN),                                &! ID of shortest wavelength ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlBBup   ! ... with WTpBB(ID,k)>=l/l_TOT
INTEGER,     INTENT(OUT)                    :: WLlEM     ! wavelength ID of emitted luminosity packet
                                                         ! [] DRONES
REAL(KIND=8)                                :: LRD       ! linear random deviate
INTEGER                                     :: WLl,WLll  ! dummy wavelength IDs
INTEGER                                     :: WLlLO     ! ID of lower wavelength in binary search
INTEGER                                     :: WLlTR     ! ID of trial wavelength in binary search
INTEGER                                     :: WLlUP     ! ID of upper wavelength in binary search
REAL(KIND=8)                                :: ZZcut     ! bifurcation cut

! [] COMPUTATION
CALL RANDOM_NUMBER(LRD)                                  ! generate linear random deviate
WLl=CEILING(LRD*DBLE(PRnTOT))                                  ! compute probability-bin ID
WLlLO=WTlBBlo(WLl,TEk)                                   ! register ID of largest lam(l) below bin
WLlUP=WTlBBup(WLl,TEk)                                   ! register ID of smallest lam(l) above bin
DO WHILE (WLlUP>WLlLO+1)                                 ! home in on wavelengths either side
WLlTR=(WLlLO+WLlUP)/2                                  !   compute middle wavelength ID
IF (WTpBB(WLlTR,TEk)<LRD) THEN                         !   [IF] low, [THEN]
WLlLO=WLlTR                                          !     increase WLlLO
ELSE                                                   !   [ELSE] too high
WLlUP=WLlTR                                          !     reduce WLlUP
ENDIF                                                  !  [ENDIF] sorted
ENDDO                                                    ! found the wavelengths either side
WLlEM=WLlUP                                              !   select upper wavelength ID

!~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_LumPack_BB
!~~~~~~~~~~~~~~~~~~~~~~~~~~~


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_LumPack_MB(TEk,WLlTOT,WTpMB,WTlMBlo,WTlMBup,WLlEM)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine is given:
!   a temperature ID                                                 (TEk);
!   the number of temperatures                                       (TEkTOT);
!   the number of reference probabilities                            (PRnTOT);
!   the number of wavelengths                                        (WLlTOT);
!   the MB emission probability for [lam(l),dlam(l)] at T=T(k)       (WTpMB(0:WLlTOT,1:TEkTOT));
!   the ID of the longest wavelength with WTpMB(ID,k)<=(l-1)/lTOT    (WTlMBlo(1:WLlTOT,1:TEkTOT));
!   and the ID of the shortest wavelength with WTpMB(ID,k)>=l/lTOT   (WTlMBup(1:WLlTOT,1:TEkTOT)).
! It returns:
!   the wavelength ID of the emitted luminosity packet               (WLlEM).

USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! [] DECLARATIONS
INTEGER,     INTENT(IN)                     :: TEk       ! temperature ID
INTEGER,     INTENT(IN)                     :: WLlTOT    ! number of wavelengths
REAL(KIND=8),INTENT(IN),                                &! MB emission probabilities ........
               DIMENSION(0:WLlTOT,0:TEkTOT) :: WTpMB     ! ... for [lam(l),dlam(l)] at T=T(k)
INTEGER,     INTENT(IN),                                &! ID of longest wavelength with ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlMBlo   ! ........ WTpMB(ID,k)<=(l-1)/l_TOT
INTEGER,     INTENT(IN),                                &! ID of shortest wavelength ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlMBup   ! ... with WTpMB(ID,k)>=l/l_TOT
INTEGER,     INTENT(OUT)                    :: WLlEM     ! wavelength ID of emitted luminosity packet
                                                         ! [] DRONES
REAL(KIND=8)                                :: LRD       ! linear random deviate
INTEGER                                     :: WLl,WLll  ! dummy wavelength IDs
INTEGER                                     :: WLlLO     ! ID of lower wavelength in binary search
INTEGER                                     :: WLlTR     ! ID of trial wavelength in binary search
INTEGER                                     :: WLlUP     ! ID of upper wavelength in binary search
REAL(KIND=8)                                :: ZZcut     ! bifurcation cut

! [] COMPUTATION
CALL RANDOM_NUMBER(LRD)                                  ! generate linear random deviate
WLl=CEILING(LRD*DBLE(PRnTOT))                                  ! compute probability-bin ID
WLlLO=WTlMBlo(WLl,TEk)                                   ! register ID of largest lam(l) below bin
WLlUP=WTlMBup(WLl,TEk)                                   ! register ID of smallest lam(l) above bin
DO WHILE (WLlUP>WLlLO+1)                                 ! home in on wavelengths either side
WLlTR=(WLlLO+WLlUP)/2                                  !   compute middle wavelength ID
IF (WTpMB(WLlTR,TEk)<LRD) THEN                         !   [IF] low, [THEN]
WLlLO=WLlTR                                          !     increase WLlLO
ELSE                                                   !   [ELSE] too high
WLlUP=WLlTR                                          !     reduce WLlUP
ENDIF                                                  !  [ENDIF] sorted
ENDDO                                                    ! found the wavelengths either side
WLlEM=WLlUP                                              !   select upper wavelength ID

!~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_LumPack_MB
!~~~~~~~~~~~~~~~~~~~~~~~~~~~


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE RT_LumPack_DM(TEk,WLlTOT,WTpDM,WTlDMlo,WTlDMup,WLlEM)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

! This subroutine is given:
!   a temperature ID                                                 (TEk);
!   the number of temperatures                                       (TEkTOT);
!   the number of reference probabilities                            (PRnTOT);
!   the number of wavelengths                                        (WLlTOT);
!   the MB emission probability for [lam(l),dlam(l)] at T=T(k)       (WTpDM(0:WLlTOT,1:TEkTOT));
!   the ID of the longest wavelength with WTpDM(ID,k)<=(l-1)/lTOT    (WTlDMlo(1:WLlTOT,1:TEkTOT));
!   and the ID of the shortest wavelength with WTpDM(ID,k)>=l/lTOT   (WTlDMup(1:WLlTOT,1:TEkTOT)).
! It returns:
!   the wavelength ID of the emitted luminosity packet               (WLlEM).


USE CONSTANTS
USE PHYSICAL_CONSTANTS
IMPLICIT NONE                                             ! [] DECLARATIONS
INTEGER,     INTENT(IN)                     :: TEk       ! temperature ID
INTEGER,     INTENT(IN)                     :: WLlTOT    ! number of wavelengths
REAL(KIND=8),INTENT(IN),                                &! MB emission probabilities ........
               DIMENSION(0:WLlTOT,0:TEkTOT) :: WTpDM     ! ... for [lam(l),dlam(l)] at T=T(k)
INTEGER,     INTENT(IN),                                &! ID of longest wavelength with ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlDMlo   ! ........ WTpMB(ID,k)<=(l-1)/l_TOT
INTEGER,     INTENT(IN),                                &! ID of shortest wavelength ...
               DIMENSION(1:PRnTOT,0:TEkTOT) :: WTlDMup   ! ... with WTpMB(ID,k)>=l/l_TOT
INTEGER,     INTENT(OUT)                    :: WLlEM     ! wavelength ID of emitted luminosity packet
                                                         ! [] DRONES
REAL(KIND=8)                                :: LRD       ! linear random deviate
INTEGER                                     :: WLl,WLll  ! dummy wavelength IDs
INTEGER                                     :: WLlLO     ! ID of lower wavelength in binary search
INTEGER                                     :: WLlTR     ! ID of trial wavelength in binary search
INTEGER                                     :: WLlUP     ! ID of upper wavelength in binary search
REAL(KIND=8)                                :: ZZcut     ! bifurcation cut

! [] COMPUTATION
CALL RANDOM_NUMBER(LRD)                                  ! generate linear random deviate
WLl=CEILING(LRD*DBLE(PRnTOT))                                  ! compute probability-bin ID
WLlLO=WTlDMlo(WLl,TEk)                                   ! register ID of largest lam(l) below bin
WLlUP=WTlDMup(WLl,TEk)                                   ! register ID of smallest lam(l) above bin
DO WHILE (WLlUP>WLlLO+1)                                 ! home in on wavelengths either side
WLlTR=(WLlLO+WLlUP)/2                                  !   compute middle wavelength ID
IF (WTpDM(WLlTR,TEk)<LRD) THEN                         !   [IF] low, [THEN]
WLlLO=WLlTR                                          !     increase WLlLO
ELSE                                                   !   [ELSE] too high
WLlUP=WLlTR                                          !     reduce WLlUP
ENDIF                                                  !  [ENDIF] sorted
ENDDO                                                    ! found the wavelengths either side
WLlEM=WLlUP                                              !   select upper wavelength ID

!~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE RT_LumPack_DM
!~~~~~~~~~~~~~~~~~~~~~~~~~~~


!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SUBROUTINE WL_cms_microns_convert(WLlTOT, WLlamIN, WLlamOUT, FLAG)
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
!This is a very basic subroutine to handle any conversion
! between cms and microns for wavelengths.

!IN:
!WLlamIN(1:WLlTOT)                                                  !Wavelengths array passed in, and to be converted
!FLAG                                                               !Flag to which units to convert to. EITHER 'cm' or 'mu' ONLY!!!

!OUT
!WLlamIN(1:WLlTOT)                                                  !Wavelengths array passed out, after having been converted

IMPLICIT NONE
INTEGER, INTENT(IN)                             :: WLlTOT           ! the number of discrete wavelengths
REAL(KIND=8),INTENT(IN),DIMENSION(1:WLlTOT)     :: WLlamIN
Character(len=2), intent(in)                    :: FLAG
REAL(KIND=8),INTENT(OUT),DIMENSION(1:WLlTOT)    :: WLlamOUT
REAL (kind=8)                                   :: ConversionConst

ConversionConst = 1.00d4

If(FLAG.eq."cm") then
    WLlamOUT = WLlamIN * ConversionConst
else if(FLAG.eq."mu") then
    WLlamOUT = WLlamIN * (1.d0/ConversionConst)
else
    print*,"WARNING! [@cms_to_microns]: Invalid conversion flag."
    print*,"Must be coverted to 'cm' [centimeters] or to 'mu' [microns]."
    print*,"No conversion has been completed! WLlamOUT = WLlamIN"

    WLlamOUT = WLlamIN
endif

!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
END SUBROUTINE WL_cms_microns_convert
!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
