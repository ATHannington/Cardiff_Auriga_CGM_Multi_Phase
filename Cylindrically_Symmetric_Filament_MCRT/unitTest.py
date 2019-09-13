import numpy as np
import RadTransSubroutinesf90 as f90Sub
import math
import pytest

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
DIAGNOSTICFLAG = 1

TESTRADIUS = 1.0

NSHELLS=3

WAVELENGTHTOT = 3
TESTWAVELENGTHS = [0.5,1.0,2.0]

NTEMPERATURES = 10
TEMPMAX = 1.E2
TEMPMIN = 1.0

#------------------------------------------------------------------------------#
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
    out = f90Sub.wl_cms_microns_convert(wlltot=WAVELENGTHTOT,\
     wllamin=TESTWAVELENGTHS, flag='mu')
    assert np.all(np.where(out == MUWAVELENGTHS, True, False)) == True,"Convert to microns failure"

    #Convert to cms
    out = f90Sub.wl_cms_microns_convert(wlltot=WAVELENGTHTOT,\
     wllamin=TESTWAVELENGTHS, flag='cm')
    assert np.all(np.where(out == CMWAVELENGTHS, True, False)) == True,"Convert to centimetres failure"

    return
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#   Generalised test for optical depth and unit direction vector               #
#------------------------------------------------------------------------------#
def lptau_lpe(lptau,lpe):
    #Check each component of unit vector is physical
    for i in range(0,3):
        assert lpe[i] <=1.0
        assert lpe[i] >=-1.0

    #Check unit vector is normalised
    assert math.sqrt((lpe[0]**2) + (lpe[1]**2) + (lpe[2]**2)) \
     == pytest.approx(1.0)

    #Check optical depth is physical
    assert math.isnan(lptau) == False
    assert math.isinf(lptau) == False
    assert lptau >= 0.0

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

def test_rt_cyl1d_injectisotropic():

    lpr,lpr1122,lpe,lptau = f90Sub.rt_cyl1d_injectisotropic(cfwb=TESTRADIUS)

    #Check Photon packet is placed on boundary at (-1,0,0)
    assert lpr[0] == pytest.approx(-1.*TESTRADIUS)
    assert lpr[1] == pytest.approx(0.0)
    assert lpr[2] == pytest.approx(0.0)

    #Check Photon packet placement is physical
    for i in range(0,3):
        assert math.isinf(lpr[i]) == False
        assert math.isnan(lpr[i]) == False
        assert lpr[i] >= -1.*TESTRADIUS
        assert lpr[i] <= TESTRADIUS

    #Check square radial distance is 1**2 and is physical
    assert math.isinf(lpr1122) == False
    assert math.isnan(lpr1122) == False
    assert lpr1122 == pytest.approx(TESTRADIUS**2)

    #Perform lptau and lpe tests
    lptau_lpe(lptau,lpe)

    return
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#   Test of linear shell spacing subroutine
#------------------------------------------------------------------------------#
def test_rt_cyl1d_linearshellspacing():

    cfw,cfw2 = f90Sub.rt_cyl1d_linearshellspacing\
    (cfwb=TESTRADIUS,cfctot=NSHELLS,cflist=DIAGNOSTICFLAG)

    #Set up linearly spaced shells and squares of these values
    RADIUSTEST = np.linspace(0.,TESTRADIUS,NSHELLS+1)
    RADISUSSQUAREDTEST = np.array([r**2 for r in RADIUSTEST])

    #Check Radial boundaries
    assert np.all(np.where(cfw == RADIUSTEST, True, False)) == True,"1D cylindrical shell cell boundary failure"

    #Check SQUARED Radial boundaries
    assert np.all(np.where(cfw2 == RADISUSSQUAREDTEST, True, False)) == True,"1D cylindrical shell SQUARED cell boundary failure"

    return
#------------------------------------------------------------------------------#

#------------------------------------------------------------------------------#
#   Test of logarithmically spaced temperatures
#------------------------------------------------------------------------------#
def test_rt_temperatures():

    #Set logarithmically spaced temperatures list
    TEMPTEST = np.logspace(start=math.log10(TEMPMIN),stop=math.log10(TEMPMAX),num=NTEMPERATURES+1,base=10.0)

    tet = f90Sub.rt_temperatures(tektot=NTEMPERATURES,tetmin=TEMPMIN,\
    tetmax=TEMPMAX,telist=DIAGNOSTICFLAG)

    #Check Temperatures are correctly logarithmically spaced
    assert np.all(np.where(tet == pytest.approx(TEMPTEST), True, False)) == True,"Logarithmically spaced temperature failure"
