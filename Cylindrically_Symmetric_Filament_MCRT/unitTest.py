import numpy as np
import RadTransSubroutinesf90 as f90Sub
from RadTransConstantsf90 import constants as f90const
from RadTransConstantsf90 import physical_constants as f90phys_const
import math
import pytest
from astropy import units as u
# from astropy.modeling.blackbody import blackbody_lambda
import scipy.integrate as integrate

DIAGNOSTICFLAG = 1

NWAVELENGTHS = 10

NPACKETS = 10

WLMIN=1.E0
WLMAX=1.E2

TESTWLHIGH = math.ceil((NWAVELENGTHS*2)/3)
TESTWLLOW = math.ceil((NWAVELENGTHS*1)/3)

TESTTEMP = 1
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#                          Set Up probabilites                                 #
#               These are CDFs of integrated Planck Functions                  #
#------------------------------------------------------------------------------#

tmp = np.linspace(0.,1.,NWAVELENGTHS+1)
PROBABILITIES = np.array([tmp for i in range(0,f90const.tektot+1)]).T
print(np.shape(PROBABILITIES))
del tmp

tmp = [TESTWLLOW for i in range(0,f90const.prntot)]
WAVELENGTHLOW = np.array([tmp for i in range(0,f90const.tektot+1)]).T
del tmp
print(np.shape(WAVELENGTHLOW))

tmp = [TESTWLHIGH for i in range(0,f90const.prntot)]
WAVELENGTHHIGH = np.array([tmp for i in range(0,f90const.tektot+1)]).T
del tmp
print(np.shape(WAVELENGTHHIGH))
#------------------------------------------------------------------------------#
#                   Set up other lists and arrays
#------------------------------------------------------------------------------#

TEMPERATURES = np.linspace(f90const.tetmin,f90const.tetmax,f90const.tektot)
print(np.shape(TEMPERATURES))

TESTWAVELENGTHS = np.linspace(WLMIN,WLMAX,NWAVELENGTHS)
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
    out = f90Sub.wl_cms_microns_convert(wlltot=NWAVELENGTHS,\
     wllamin=TESTWAVELENGTHS, flag='mu')
    assert np.all(np.where(out == MUWAVELENGTHS, True, False)) == True,"Convert to microns failure"
    assert np.any(np.isinf(out)) == False,"Micron conversion is INF"
    assert np.any(np.isnan(out)) == False,"Micron conversion is NaN"

    #Convert to cms
    out = f90Sub.wl_cms_microns_convert(wlltot=NWAVELENGTHS,\
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
#       Test Isotropic injection at (-f90const.cfwb,0,0)                                #
#------------------------------------------------------------------------------#
def test_rt_cyl1d_injectisotropic():

    lpr,lpr1122,lpe,lptau = f90Sub.rt_cyl1d_injectisotropic()

    #Check Photon packet is placed on boundary at (-1,0,0)
    assert lpr[0] == pytest.approx(-1.*f90const.cfwb),"Photon packet x position is NOT at (-1*f90const.cfwb,y,z)"
    assert lpr[1] == pytest.approx(0.0),"Photon packet y position is NOT at (x,0,z)"
    assert lpr[2] == pytest.approx(0.0),"Photon packet z position is NOT at (x,y,0)"

    #Check Photon packet placement is physical
    for i in range(0,3):
        assert math.isinf(lpr[i]) == False,"Photon packet position is INF"
        assert math.isnan(lpr[i]) == False,"Photon packet position is NaN"
        assert lpr[i] >= -1.*f90const.cfwb
        assert lpr[i] <= f90const.cfwb

    #Check square radial distance is 1**2 and is physical
    assert math.isinf(lpr1122) == False,"Photon packet square radial distance is INF"
    assert math.isnan(lpr1122) == False,"Photon packet square radial distance is NaN"
    assert lpr1122 == pytest.approx(f90const.cfwb**2),"Photon packet square radial distance is NOT square of radius as (-1*rad,0,0)"

    #Perform lptau and lpe tests
    lptau_lpe(lptau,lpe)

    return



def set_radii():
    #Set up linearly spaced shells and squares of these values
    TESTRADII = np.linspace(0.,f90const.cfwb,f90const.cfctot+1)
    return TESTRADII
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#   Test of linear shell spacing subroutine
#------------------------------------------------------------------------------#
def test_rt_cyl1d_linearshellspacing():

    cfw,cfw2 = f90Sub.rt_cyl1d_linearshellspacing()

    RADIUSTEST = set_radii()

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
    TEMPTEST = np.logspace(start=math.log10(f90const.tetmin),stop=math.log10(f90const.tetmax),num=f90const.tektot+1,base=10.0)

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
    assert wllem <= NWAVELENGTHS,"Wavelength beyond maximum error"
    assert wllem >= 0,"Wavelength below minimum (0) error"
    assert np.any(np.isinf(wllem)) == False,"Wavelength is infinite error"
    assert np.any(np.isnan(wllem)) == False,"Wavelength is NaN error"

    #Check Wavelength sits between selected high and low wavelength indices
    assert wllem >= TESTWLLOW, "Wavelength below LOW wavelength index"
    assert wllem <= TESTWLHIGH, "Wavelength above HIGH wavelength index"

    return

def test_rt_lumpack():

    #Test BlackBody Subroutine (BB)
    wllem = f90Sub.rt_lumpack_bb(tek=TESTTEMP,wlltot=NWAVELENGTHS,wtpbb=PROBABILITIES,\
    wtlbblo=WAVELENGTHLOW,wtlbbup=WAVELENGTHHIGH)

    rt_lumpack(wllem)

    #Test Modified Black Body Subroutine (MB)
    wllem = f90Sub.rt_lumpack_mb(tek=TESTTEMP,wlltot=NWAVELENGTHS,wtpmb=PROBABILITIES,\
    wtlmblo=WAVELENGTHLOW,wtlmbup=WAVELENGTHHIGH)

    rt_lumpack(wllem)

    #Test Differential Modified Black Body Subroutine (DM)
    wllem = f90Sub.rt_lumpack_dm(tek=TESTTEMP,wlltot=NWAVELENGTHS,wtpdm=PROBABILITIES,\
    wtldmlo=WAVELENGTHLOW,wtldmup=WAVELENGTHHIGH)

    rt_lumpack(wllem)

    return

#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#       Tests on the Schuster Density Profiles Subroutine
#           Here we check the physicallity of the returned values and
#           check that the returned values for total line density and
#           surface density are what we expect.
#           We alsp check for that the density matches what is expected for
#           the selected Schuster p value.
#
#      Note: ideally this test ought to be re-run for every value of 0 <= p <= 4
#           which are varied upon compile time of the FORTRAN90 version using
#           f2py. i.e. vary p, compile with f2py, run again. repeat.
#------------------------------------------------------------------------------#
def test_schuster_p_check():
    p = int(f90const.cfschp)
    if ((p<0) or (p>4)):
        assert p >= 0,"Schuster p Failure! 0<=p<=4"
        assert p <= 4,"Schuster p Failure! 0<=p<=4" 

    return

def test_rt_cyl1d_schusterdensities():

    RADIUSTEST = set_radii()

    cfrho,cfmu,cfmutot,cfsig = f90Sub.rt_cyl1d_schusterdensities(cfw=RADIUSTEST)

    #### Perform sanity checks
        # Density returned is physical
    assert np.any(np.isinf(cfrho)) == False,"Density is infinite error"
    assert np.any(np.isnan(cfrho)) == False,"Density is NaN error"

        # Line Density returned is physical
    assert np.any(np.isinf(cfmu)) == False,"Line Density is infinite error"
    assert np.any(np.isnan(cfmu)) == False,"Line Density is NaN error"

        # Line Density Total returned is physical
    assert math.isinf(cfmutot) == False,"Line Density TOTAL is infinite error"
    assert math.isnan(cfmutot) == False,"Line Density TOTAL is NaN error"

        # Surface Density Total returned is physical
    assert math.isinf(cfsig) == False,"Surface Density TOTAL is infinite error"
    assert math.isnan(cfsig) == False,"Surface Density TOTAL is NaN error"


    ###### Test Density profiles #####

    ## Create Test list of density from definition of Schuster Profile
    TESTDENSITY = np.array([f90const.cfrho0*(1.+((x/f90const.cfw0)**2))**(-1.*(float(f90const.cfschp))/2.) \
    for x in RADIUSTEST[:len(RADIUSTEST)-1]])

    ## Check Density generated meets expected density
    assert np.all(np.where(cfrho == pytest.approx(TESTDENSITY,rel=1e-3), True, False)) == True,"Density does not meet expected function"

    #Create a temporary table of: rho * 2 pi  * w for line density integrand
    #       Note: The changes in indices are to account for the density being defined
    #           within a region, and so having 1 less element
    tmp_integrand_mu = np.array([TESTDENSITY[i]*2.0*math.pi*RADIUSTEST[i] for i in range(0,len(RADIUSTEST)-1)])




    ###### Test that line and surface density from  Python integrals match F90 result #####

    #Numerically integrate for line density accounting for first boundary
    testmu = integrate.simps(y=tmp_integrand_mu,x=RADIUSTEST[:len(RADIUSTEST)-1])
    #Test equality to 0.1% tolerance
    assert cfmutot == pytest.approx(testmu,rel=1e-3),"Line Density total does not meet tolerance: +/-0.1%"
    del tmp_integrand_mu

    #Numerically integrate for surface density accounting for first boundary
    testsigma = 2.0 * integrate.simps(y=TESTDENSITY,x=RADIUSTEST[:len(RADIUSTEST)-1])
    #Test equality to 0.1% tolerance
    assert cfsig == pytest.approx(testsigma,rel=1e-3),"Surface Density total does not meet tolerance: +/-0.1%"




    ##### Check F90 against analytic results #####

    zeta = float((f90const.cfwb)/(f90const.cfw0))
    p = int(f90const.cfschp)
    MUCONST = 2.*math.pi*(f90const.cfrho0)*((f90const.cfw0)**2)
    SIGCONST = 2.*(f90const.cfrho0)*(f90const.cfw0)
    if (p == 0):
        assert cfmutot == pytest.approx((MUCONST*(zeta**2)*0.5)),f"Analytic Check: Schuster P={p:.0f} cfmutot does not equal analytic within 0.1% tolerance"
        assert cfsig == pytest.approx(SIGCONST*zeta),f"Analytic Check: Schuster P={p:.0f} cfsig does not equal analytic within 0.1% tolerance"
    elif (p == 1):
        assert cfmutot == pytest.approx( (MUCONST*(((1.+ zeta**2)**(0.5))-1.)) ),f"Analytic Check: Schuster P={p:.0f} cfmutot does not equal analytic within 0.1% tolerance"
        assert cfsig == pytest.approx((SIGCONST*(math.log( zeta+((1.+ zeta**2)**(0.5)) )))),f"Analytic Check: Schuster P={p:.0f} cfsig does not equal analytic within 0.1% tolerance"
    elif (p == 2):
        assert cfmutot == pytest.approx((MUCONST*((math.log(1.+ zeta**2))*0.5))),f"Analytic Check: Schuster P={p:.0f} cfmutot does not equal analytic within 0.1% tolerance"
        assert cfsig == pytest.approx(SIGCONST*(math.atan(zeta))),f"Analytic Check: Schuster P={p:.0f} cfsig does not equal analytic within 0.1% tolerance"
    elif (p == 3):
        assert cfmutot == pytest.approx( (MUCONST*(1.-((1.+ zeta**2)**(-0.5)))) ),f"Analytic Check: Schuster P={p:.0f} cfmutot does not equal analytic within 0.1% tolerance"
        assert cfsig == pytest.approx(SIGCONST*(zeta/((1.+zeta**2)**(0.5)))),f"Analytic Check: Schuster P={p:.0f} cfsig does not equal analytic within 0.1% tolerance"
    elif (p == 4):
        assert cfmutot == pytest.approx((MUCONST*(0.5*(zeta**2)/(1.+zeta**2)))),f"Analytic Check: Schuster P={p:.0f} cfmutot does not equal analytic within 0.1% tolerance"
        assert cfsig == pytest.approx(SIGCONST*0.5*(math.atan(zeta) + (zeta/((1.+zeta**2))))),f"Analytic Check: Schuster P={p:.0f} cfsig does not equal analytic within 0.1% tolerance"

    return
