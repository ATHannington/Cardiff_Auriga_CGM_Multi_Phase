import numpy as np
import RadTransSubroutinesf90 as f90Sub
import RadTransConstantsf90 as f90const
import math
import pytest
from astropy import units as u
# from astropy.modeling.blackbody import blackbody_lambda
import scipy.integrate as integrate

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#------------------------------------------------------------------------------#
# (CFgeomdum,CFrho0dum, CFw0dum, CFschPdum, CFwBdum, CFcTOTdum, CFprofdum,&
# & CFlistdum, DGsourcedum, DGmodeldum, DGlMAXdum, DGlMINdum, DGkapMdum, DGkapVdum, WLdeltadum, WLdcldum,&
# & WLprintdum, WLplotdum, TEkTOTdum, teTmindum, teTmaxdum, TElistdum, PRnTOTdum, WTpackdum, WTplotdum, BGkBBdum, &
# & BGfBBdum, BGkGOdum, LPpTOTdum, DBTestFlagdum, pidum,twopidum, lightcdum, planckhdum, boltzkbdum,&
# & sigmasbdum, hckbdum, hc2dum, h2c3kbdum, cmtopcdum, pctocmdum, msoldum, amudum, msolpctogcmdum,&
# & gcmtomsolpcdum, h2densdum, invh2densdum)
#------------------------------------------------------------------------------#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

CFgeom,CFrho0, CFw0, CFschP, CFwB, CFcTOT, CFprof,\
CFlist, DGsource, DGmodel, DGlMAX, DGlMIN, DGkapM, DGkapV, WLdelta, WLdcl,\
WLprint, WLplot, TEkTOT, teTmin, teTmax, TElistdum, PRnTOT, WTpack, WTplot, BGkBB, \
BGfBB, BGkGO, LPpTOT, DBTestFlag, pi,twopi, lightc, planckh, boltzkb,\
sigmasb, hckb, hc2, h2c3kb, cmtopc, pctocm, msol, amu, msolpctogcm,\
gcmtomsolpc, h2dens, invh2dens = f90const.python_constants()


# DIAGNOSTICFLAG = 1
#
# CFcTOT=10
WLlTOT = 10
# PRnTOT = 10
# TEkTOT = 10
# NPACKETS = 10
#
# #CFwB = 1.0
#
WLMIN=1.E0
WLMAX=1.E2
#
TESTWLHIGH = math.ceil((WLlTOT*2)/3)
TESTWLLOW = math.ceil((WLlTOT*1)/3)
#
# teTmax = 1.E2
# teTmin = 1.0
# TESTTEMP = TEkTOT/2
#
#                                                                                 #  CONFIGURATION [CF]
# CFgeom='Cylindrical1D'                                                          # geometry of configuration
# CFrho0=0.1000E-18                                                               # central density (g/cm^3)
# CFw0=0.1500E18                                                                  # core radius (cm)
# CFschP=1                                                                        # radial density exponent forn Schuster profile
# CFwB=0.1500E19                                                                  # boundary radius (cm)
# CFcTOT=100                                                                      # number of (cylindrical) shells
# CFprof=1                                                                        # flag to sanction diagnostics for profile
# CFlist=1                                                                        # flag to sanction diagnostics for cells
#
#                                                                                 # DUST GRAINS [DG]
# DGsource='Draine'                                                               # source of dust properties (e.g. 'Draine')
# DGmodel='draine_rv3.1.dat'                                                      # dust model (e.g. 'draine_rv3.1.dat')
# DGlMAX=66                                                                       # line number where dust properties end
# DGlMIN=560                                                                      # line number where dust properties start
# DGkapM=0.30000E-17                                                              # mass opacity, only for pure scattering (cm^2/g)
# DGkapV=0.20000E3                                                                # volume opacity, only for pure scattering (1/cm)
#
#                                                                                 # WAVELENGTHS [WL]
# WLdelta=0.10                                                                    # logarithmic spacing of optical properties
# WLdcl=0.10                                                                      # weight of slope-change
# WLprint=1                                                                        # flag to trigger printing of dust properties
# WLplot=1                                                                        # flag to trigger plotting of dust properties
#
#                                                                                 # TEMPERATURES [TE]
# TEkTOT=100                                                                      # number of discrete temperatures
# teTmin=2.725                                                                    # minimum discrete temperature
# teTmax=272.5                                                                    # maximum discrete temperature
# TElist=1                                                                        # flag to print out some temperatures
#
#                                                                                 # REFERENCE PROBABILITIES [PR]
# PRnTOT=1000                                                                     # number of reference probabilities
# WTpack=1000000                                                                  # number of calls for plotting probabilities
# WTplot=1                                                                        # flag to sanction plotting probabilities
#
#                                                                                 # BACKGROUND RADIATION FIELD [BG]
# BGkBB=29#29                                                                     # temperature-ID of background BB radiation field
# BGfBB=1.00E0#0.100E0                                                            # dilution factor of background BB radiation field
# BGkGO=24                                                                        # ID of temperature for cfLgo ---- ceiling(dble(BGkBB)*0.8d0)
#
#                                                                                 # LUMINOSITY PACKETS [LP]
# LPpTOT= int(1E6)                                                                # number of luminosity packets
#
# DBTestFlag= 1                                                                   #Diagnostic test flag for tests and print statements
#
#
# # # The value of pi is 3.14159274
# # # 2^(-1/2) = 0.70710678
# # # 2^(+1/2) = 1.4142136
# # # (6/pi)^(1/3) = 1.2407007
# # # pc = (0.308568E+19) cm
# # # cm = (0.324078E-18) pc
# # # M_Sun = (0.198910E+34) g
# # # amu = (0.166054E-23) g
# # # g/cm = (0.155129E-15) M_Sun/pc
# # # M_Sun/pc = (0.644623E+15) g/cm
# # # H2/g = (0.210775E+24)
# # # g/H2 = (0.474444E-23)
#
# pi = 3.14159274
# twopi = 6.28318548
# lightc = 2.997925e10                                                            #[cm s**-1]
# planckh = 6.626070e-27                                                          #[erg s]
# boltzkb = 1.380649e-16                                                          #[erg K**-1]
# sigmasb = 5.670515e-5                                                           #[erg s**-1 cm**-2 K**-4]
# hckb = 0.143878e5                                                               #[microns K**-1]
# hc2 = 5.9552219872e10                                                           #[erg s**-1 cm**-2 microns**4]
# h2c3kb = 8.568225728e14                                                         #[erg s**-1 cm**-2 K**-1 microns**5 K**2]
# cmtopc = 0.324078e-18                                                           #[pc]
# pctocm = 3.085676905e18                                                         #[cm]
# msol = 0.198910e34                                                              #[g]
# amu = 0.166054e-23                                                              #[g] atomic mass unit
# msolpctogcm = 0.644623E15                                                       #[g cm**-1]
# gcmtomsolpc = 0.155129E-15                                                      #[M_sol pc**-1]
# h2dens = 0.210775E24                                                            #[g cm**-1]
# invh2dens = 0.474444E-23                                                        #[cm g**-1]
#

#------------------------------------------------------------------------------#
#               PHYSICAL CONSTANTS IN CGS
#------------------------------------------------------------------------------#
#		Below are constants used by this program                               #
#			Here we have used the AstroPy module to allocate units and convert #
#				to CGS units.                                                  #
#                                                                              #
#------------------------------------------------------------------------------#

h = 6.62607004e-34*u.m**2 * u.kg / u.s											#Planck constant [m^2 kg / s]
h = h.cgs																		#Planck constant to CGS units [cm^2 g / s]
#print("TEST: Planck Constant in CGS =", h)

c  = 299792458 * u.m / u.s														#Speed of Light [m / s]
c = c.cgs																		#Speed of Light to CGS units [cm /s]
#print("TEST: C in CGS =", c)

kb = 1.3806852e-23 * u.m**2 * u.kg / (u.s**2 * u.K)								#Boltzmann constant [m^2 kg / s^2 /K ]
kb = kb.cgs
#print("TEST: kb in CGS =", kb)

#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#                          Set Up probabilites                                 #
#               These are CDFs of integrated Planck Functions                  #
#------------------------------------------------------------------------------#
# def blackbody_lambda_ath(wavelengths,temperature):
#     planck = [(lam**(-5))* \
#     ( ((np.exp((h*c)/(lam*kb*temperature)))-1.)**(-1) )\
#     for lam in wavelengths]
#
#     return planck
#
#
# PROBABILITIES = np.array([integrate.quad(\
# blackbody_lambda_ath(WAVELENGTHS,t)) for t in TEMPERATURES])
#


tmp = np.linspace(0.,1.,WLlTOT+1)
PROBABILITIES = np.array([tmp for i in range(0,TEkTOT+1)]).T
del tmp

tmp = [TESTWLLOW for i in range(0,PRnTOT)]
WAVELENGTHLOW = np.array([tmp for i in range(0,TEkTOT+1)]).T
del tmp

tmp = [TESTWLHIGH for i in range(0,PRnTOT)]
WAVELENGTHHIGH = np.array([tmp for i in range(0,TEkTOT+1)]).T
del tmp

#------------------------------------------------------------------------------#
#                   Set up other lists and arrays
#------------------------------------------------------------------------------#

TEMPERATURES = np.linspace(teTmin,teTmax,TEkTOT)

TESTWAVELENGTHS = np.linspace(WLMIN,WLMAX,WLlTOT)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#                               TESTS BELOW                                    #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


#------------------------------------------------------------------------------#
#   Test Wavelength converter between microns and cms_to_microns               #
#------------------------------------------------------------------------------#
def test_wl_cms_microns_convert():
    #Set expected wavelengths
    MUWAVELENGTHS = [x*1.e-4 for x in TESTWAVELENGTHS]
    CMWAVELENGTHS = [x*1.e+4 for x in TESTWAVELENGTHS]

    #Convert to microns
    out = f90Sub.wl_cms_microns_convert(wlltot=WLlTOT,\
     wllamin=TESTWAVELENGTHS, flag='mu')
    assert np.all(np.where(out == MUWAVELENGTHS, True, False)) == True,"Convert to microns failure"
    assert np.any(np.isinf(out)) == False,"Micron conversion is INF"
    assert np.any(np.isnan(out)) == False,"Micron conversion is NaN"

    #Convert to cms
    out = f90Sub.wl_cms_microns_convert(wlltot=WLlTOT,\
     wllamin=TESTWAVELENGTHS, flag='cm')
    assert np.all(np.where(out == CMWAVELENGTHS, True, False)) == True,"Convert to centimetres failure"
    assert np.any(np.isinf(out)) == False,"Centimetre conversion is INF"
    assert np.any(np.isnan(out)) == False,"Centimetre conversion is NaN"

    return
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#   Generalised test for optical depth and unit direction vector               #
#------------------------------------------------------------------------------#
def lptau_lpe(lptau,lpe):
    #Check each component of unit vector is physical
    for i in range(0,3):
        assert lpe[i] <=1.0,"Unit vector component is greater than 1"
        assert lpe[i] >=-1.0,"Unit vector component is less than -1"

    #Check unit vector is normalised
    assert math.sqrt((lpe[0]**2) + (lpe[1]**2) + (lpe[2]**2)) \
     == pytest.approx(1.0),"Unit vector component is NOT normalised"

    #Check optical depth is physical
    assert math.isnan(lptau) == False,"Optical depth is NaN"
    assert math.isinf(lptau) == False,"Optical depth is INF"
    assert lptau >= 0.0,"Optical depth is less than 0 => non-physical!"

    return
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#       Test Isotropic Redirection via random direction unit vector and        #
#              random optical depth                                            #
#------------------------------------------------------------------------------#
def test_rt_redirectisotropic():
    lpe, lptau = f90Sub.rt_redirectisotropic()

    #Perform lptau and lpe tests
    lptau_lpe(lptau,lpe)

    return
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#       Test Isotropic injection at (-CFwB,0,0)                                #
#------------------------------------------------------------------------------#
def test_rt_cyl1d_injectisotropic():

    lpr,lpr1122,lpe,lptau = f90Sub.rt_cyl1d_injectisotropic()

    #Check Photon packet is placed on boundary at (-1,0,0)
    assert lpr[0] == pytest.approx(-1.*CFwB),"Photon packet x position is NOT at (-1*CFwB,y,z)"
    assert lpr[1] == pytest.approx(0.0),"Photon packet y position is NOT at (x,0,z)"
    assert lpr[2] == pytest.approx(0.0),"Photon packet z position is NOT at (x,y,0)"

    #Check Photon packet placement is physical
    for i in range(0,3):
        assert math.isinf(lpr[i]) == False,"Photon packet position is INF"
        assert math.isnan(lpr[i]) == False,"Photon packet position is NaN"
        assert lpr[i] >= -1.*CFwB
        assert lpr[i] <= CFwB

    #Check square radial distance is 1**2 and is physical
    assert math.isinf(lpr1122) == False,"Photon packet square radial distance is INF"
    assert math.isnan(lpr1122) == False,"Photon packet square radial distance is NaN"
    assert lpr1122 == pytest.approx(CFwB**2),"Photon packet square radial distance is NOT square of radius as (-1*rad,0,0)"

    #Perform lptau and lpe tests
    lptau_lpe(lptau,lpe)

    return
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#   Test of linear shell spacing subroutine
#------------------------------------------------------------------------------#
def test_rt_cyl1d_linearshellspacing():

    cfw,cfw2 = f90Sub.rt_cyl1d_linearshellspacing()

    #Set up linearly spaced shells and squares of these values
    RADIUSTEST = np.linspace(0.,CFwB,CFcTOT+1)
    RADISUSSQUAREDTEST = np.array([r**2 for r in RADIUSTEST])

    #Check Radial boundaries
    assert np.all(np.where(cfw == RADIUSTEST, True, False)) == True,"1D cylindrical shell cell boundary failure"

    #Check SQUARED Radial boundaries
    assert np.all(np.where(cfw2 == RADISUSSQUAREDTEST, True, False)) == True,"1D cylindrical shell SQUARED cell boundary failure"

    return
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#   Test of logarithmically spaced temperatures boundaries (therefore ntemps
#               +1 boundaries)
#------------------------------------------------------------------------------#
def test_rt_temperatures():

    #Set logarithmically spaced temperatures list
    TEMPTEST = np.logspace(start=math.log10(teTmin),stop=math.log10(teTmax),num=TEkTOT+1,base=10.0)

    tet = f90Sub.rt_temperatures()

    #Check Temperatures are correctly logarithmically spaced
    assert np.all(np.where(tet == pytest.approx(TEMPTEST), True, False)) == True,"Logarithmically spaced temperature failure"
    assert np.any(np.isinf(tet)) == False,"Log spaced temperatures are INF"
    assert np.any(np.isnan(tet)) == False,"Log spaced temperatures are NaN"
    return
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#       Tests on the Black Body Subroutines
#               For a given temperature, can they produce a sensible wavelength#
#                   from the integrated BB function acting as a CDF            #
#                       sampled by a MCRT method.
#------------------------------------------------------------------------------#
def rt_lumpack(wllem):

    #Check wavelength returned is physical
    assert wllem <= WLlTOT,"Wavelength beyond maximum error"
    assert wllem >= 0,"Wavelength below minimum (0) error"
    assert np.any(np.isinf(wllem)) == False,"Wavelength is infinite error"
    assert np.any(np.isnan(wllem)) == False,"Wavelength is NaN error"

    #Check Wavelength sits between selected high and low wavelength indices
    assert wllem >= TESTWLLOW, "Wavelength below LOW wavelength index"
    assert wllem <= TESTWLHIGH, "Wavelength above HIGH wavelength index"

    return

def test_rt_lumpack():

    #Test BlackBody Subroutine (BB)
    wllem = f90Sub.rt_lumpack_bb(TEk,WLlTOT,WTpBB,WTlBBlo,WTlBBup)

    rt_lumpack(wllem)

    #Test Modified Black Body Subroutine (MB)
    wllem = f90Sub.rt_lumpack_mb(TEk,WLlTOT,WTpMB,WTlMBlo,WTlMBup)

    rt_lumpack(wllem)

    #Test Differential Modified Black Body Subroutine (DM)
    wllem = f90Sub.rt_lumpack_dm(TEk,WLlTOT,WTpDM,WTlDMlo,WTlDMup)

    rt_lumpack(wllem)

    return
