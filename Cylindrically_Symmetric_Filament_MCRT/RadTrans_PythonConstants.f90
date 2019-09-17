subroutine PYTHON_CONSTANTS(CFgeomdum,CFrho0dum, CFw0dum, CFschPdum, CFwBdum, CFcTOTdum, CFprofdum,&
& CFlistdum, DGsourcedum, DGmodeldum, DGlMAXdum, DGlMINdum, DGkapMdum, DGkapVdum, WLdeltadum, WLdcldum,&
& WLprintdum, WLplotdum, TEkTOTdum, teTmindum, teTmaxdum, TElistdum, PRnTOTdum, WTpackdum, WTplotdum, BGkBBdum, &
& BGfBBdum, BGkGOdum, LPpTOTdum, DBTestFlagdum, pidum,twopidum, lightcdum, planckhdum, boltzkbdum,&
& sigmasbdum, hckbdum, hc2dum, h2c3kbdum, cmtopcdum, pctocmdum, msoldum, amudum, msolpctogcmdum,&
& gcmtomsolpcdum, h2densdum, invh2densdum)

    USE CONSTANTS
    USE PHYSICAL_CONSTANTS
    IMPLICIT NONE

    CHARACTER(LEN=20),intent(out)       :: CFgeomdum
    REAL(KIND=8),intent(out)            :: CFrho0dum
    REAL(KIND=8),intent(out)            :: CFw0dum
    INTEGER,intent(out)                 :: CFschPdum
    REAL(KIND=8),intent(out)            :: CFwBdum
    INTEGER,intent(out)                 :: CFcTOTdum
    INTEGER,intent(out)                 :: CFprofdum
    INTEGER,intent(out)                 :: CFlistdum
    CHARACTER(LEN=20),intent(out)       :: DGsourcedum
    CHARACTER(LEN=20),intent(out)       :: DGmodeldum
    INTEGER,intent(out)                 :: DGlMAXdum
    INTEGER,intent(out)                 :: DGlMINdum
    REAL(KIND=8),intent(out)            :: DGkapMdum
    REAL(KIND=8),intent(out)            :: DGkapVdum
    REAL(KIND=8),intent(out)            :: WLdeltadum
    REAL(KIND=8),intent(out)            :: WLdcldum
    INTEGER,intent(out)                 :: WLprintdum
    INTEGER,intent(out)                 :: WLplotdum
    INTEGER,intent(out)                 :: TEkTOTdum
    REAL(KIND=8),intent(out)            :: teTmindum
    REAL(KIND=8),intent(out)            :: teTmaxdum
    INTEGER,intent(out)                 :: TElistdum
    INTEGER,intent(out)                 :: PRnTOTdum
    INTEGER,intent(out)                 :: WTpackdum
    INTEGER,intent(out)                 :: WTplotdum
    INTEGER,intent(out)                 :: BGkBBdum
    REAL(KIND=8),intent(out)            :: BGfBBdum
    INTEGER,intent(out)                 :: BGkGOdum
    INTEGER,intent(out)                 :: LPpTOTdum
    INTEGER(Kind=4),intent(out)         :: DBTestFlagdum
    Real(kind=8),intent(out)            :: pidum
    Real(kind=8),intent(out)            :: twopidum
    Real(kind=8),intent(out)            :: lightcdum
    Real(kind=8),intent(out)            :: planckhdum
    Real(kind=8),intent(out)            :: boltzkbdum
    Real(kind=8),intent(out)            :: sigmasbdum
    Real(kind=8),intent(out)            :: hckbdum
    Real(kind=8),intent(out)            :: hc2dum
    Real(kind=8),intent(out)            :: h2c3kbdum
    Real(kind=8),intent(out)            :: cmtopcdum
    Real(kind=8),intent(out)            :: pctocmdum
    Real(kind=8),intent(out)            :: msoldum
    Real(kind=8),intent(out)            :: amudum
    Real(kind=8),intent(out)            :: msolpctogcmdum
    Real(kind=8),intent(out)            :: gcmtomsolpcdum
    Real(kind=8),intent(out)            :: h2densdum
    Real(kind=8),intent(out)            :: invh2densdum

    ! (CFgeomdum,CFrho0dum, CFw0dum, CFschPdum, CFwBdum, CFcTOTdum, CFprofdum,&
    ! & CFlistdum, DGsourcedum, DGmodeldum, DGlMAXdum, DGlMINdum, DGkapMdum, DGkapVdum, WLdeltadum, WLdcldum,&
    ! & WLprintdum, WLplotdum, TEkTOTdum, teTmindum, teTmaxdum, PRnTOTdum, WTpackdum, WTplotdum, BGkBBdum, &
    ! & BGfBBdum, BGkGOdum, LPpTOTdum, DBTestFlagdum, pidum,twopidum, lightcdum, planckhdum, boltzkbdum,&
    ! & sigmasbdum, hckbdum, hc2dum, h2c3kbdum, cmtopcdum, pctocmdum, msoldum, amudum, msolpctogcmdum,&
    ! & gcmtomsolpcdum, h2densdum, invh2densdum)
    !
    ! (CFgeom,CFrho0, CFw0, CFschP, CFwB, CFcTOT, CFprof,&
    ! & CFlist, DGsource, DGmodel, DGlMAX, DGlMIN, DGkapM, DGkapV, WLdelta, WLdcl,&
    ! & WLprint, WLplot, TEkTOT, teTmin, teTmax, PRnTOT, WTpack, WTplot, BGkBB, &
    ! & BGfBB, BGkGO, LPpTOT, DBTestFlag, pi,twopi, lightc, planckh, boltzkb,&
    ! & sigmasb, hckb, hc2, h2c3kb, cmtopc, pctocm, msol, amu, msolpctogcm,&
    ! & gcmtomsolpc, h2dens, invh2dens)

    CFgeomdum=CFgeom; CFrho0dum=CFrho0; CFw0dum=CFw0; CFschPdum=CFschP;
    CFwBdum=CFwB; CFcTOTdum=CFcTOT; CFprofdum=CFprof; TElistdum = TElist;
    CFlistdum=CFlist; DGsourcedum=DGsource; DGmodeldum=DGmodel; DGlMAXdum=DGlMAX;
    DGlMINdum=DGlMIN; DGkapMdum=DGkapM; DGkapVdum=DGkapV; WLdeltadum=WLdelta;
    WLdcldum=WLdcl; WLprintdum=WLprint; WLplotdum=WLplot; TEkTOTdum=TEkTOT;
    teTmindum=teTmin; teTmaxdum=teTmax; PRnTOTdum=PRnTOT; WTpackdum=WTpack;
    WTplotdum=WTplot; BGkBBdum=BGkBB; BGfBBdum=BGfBB; BGkGOdum=BGkGO;
    LPpTOTdum=LPpTOT; DBTestFlagdum=DBTestFlag; pidum=pi; twopidum=twopi;
    lightcdum=lightc; planckhdum=planckh; boltzkbdum=boltzkb
    sigmasbdum=sigmasb; hckbdum=hckb; hc2dum=hc2; h2c3kbdum=h2c3kb;
    cmtopcdum=cmtopc; pctocmdum=pctocm; msoldum=msol; amudum=amu;
    msolpctogcmdum=msolpctogcm; gcmtomsolpcdum=gcmtomsolpc;
    h2densdum=h2dens; invh2densdum=invh2dens;


end subroutine PYTHON_CONSTANTS
