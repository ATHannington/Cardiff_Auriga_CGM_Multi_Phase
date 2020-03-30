"""
Author: A. T. Hannington
Created: 19/03/2020
Known Bugs:
    None
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import gadget_readsnap
from gadget_subfind import load_subfind, sum
from Snapper import Snapper
from Tracers_Subroutines import GetTracersFromCells, GetCellsFromTracers, GetIndividualCellFromTracer
import pytest

#==============================================================================#

simfile='/home/universe/spxtd1-shared/ISOTOPES/output/' # set paths to simulation
snapnum=127 # set snapshot to look at

#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
elements_mass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

Zsolar = 0.0127

omegabaryon0 = 0.048
#==============================================================================#


# Single FvdV Projection:
# load in the subfind group files
snap_subfind = load_subfind(snapnum,dir=simfile)

# load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
snapGas     = gadget_readsnap(snapnum, simfile, hdf5=True, loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
snapTracers = gadget_readsnap(snapnum, simfile, hdf5=True, loadonlytype = [6], lazy_load=True)

print(f" SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

Snapper1 = Snapper()
snapGas     = Snapper1.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=0)

#Convert Units
## Make this a seperate function at some point??
snapGas.pos   *= 1e3 #[kpc]

snapGas.vol *= 1e9 #[kpc^3]

meanweight = sum(snapGas.gmet[:,0:9], axis = 1) / ( sum(snapGas.gmet[:,0:9]/elements_mass[0:9], axis = 1) + snapGas.ne*snapGas.gmet[:,0] )
Tfac = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53
snapGas.data['T'] = snapGas.u / Tfac # K

###
##  Selection   ##
###
target=5.
delta = 0.25

#Select INDICES of T = 10^5+-delta K
Cond = np.where((snapGas.data['T']>=1.*10**(target-delta)) & (snapGas.data['T']<=1.*10**(target+delta)))






Tracers, CellsTFC, CellIDsTFC = GetTracersFromCells(snapGas, snapTracers,Cond)

CellsCFTinit, CellIDsCFTinit, Parents = GetCellsFromTracers(snapGas, snapTracers,Tracers)

# Single FvdV Projection:
# load in the subfind group files
snap_subfind = load_subfind(snapnum-1,dir=simfile)

# load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
snapGas     = gadget_readsnap(snapnum-1, simfile, hdf5=True, loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
snapTracers = gadget_readsnap(snapnum-1, simfile, hdf5=True, loadonlytype = [6], lazy_load=True)

print(f" SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

Snapper1 = Snapper()
snapGas     = Snapper1.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=0)

#Convert Units
## Make this a seperate function at some point??
snapGas.pos   *= 1e3 #[kpc]

snapGas.vol *= 1e9 #[kpc^3]

meanweight = sum(snapGas.gmet[:,0:9], axis = 1) / ( sum(snapGas.gmet[:,0:9]/elements_mass[0:9], axis = 1) + snapGas.ne*snapGas.gmet[:,0] )
Tfac = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53
snapGas.data['T'] = snapGas.u / Tfac # K

###
##  Selection   ##
###
target=5.
delta = 0.25

#Select INDICES of T = 10^5+-delta K
Cond = np.where((snapGas.data['T']>=1.*10**(target-delta)) & (snapGas.data['T']<=1.*10**(target+delta)))

CellsCFT, CellIDsCFT, Parents = GetCellsFromTracers(snapGas, snapTracers,Tracers)
#==============================================================================#
#
#                                  TESTING
#
#==============================================================================#

def test_CellIDs():
    """
    Test that the Cell IDs selected from tracers match the CellIDs containing tracers when tracers are selected
    """
    CellIDMatch = np.all(np.isin(CellIDsTFC,CellIDsCFTinit))

    assert CellIDMatch == True,"[@CellIDMatch:] Cell IDs not equal! TFC and CFT! Check tracer selections!"

def test_CellData():
    """
    Check all values of data from Cells selected from tracers matches data selected from selecting the tracers.
    """
    truthyList = []
    for ((k1,v1),(k2,v2)) in zip(CellsCFTinit.items(),CellsTFC.items()):
        #Do all entries if CellsCFT values and CellsTFC values match?
        truthyList.append(np.all(np.isin(v1,v2)))

    #Do the entries match for all keys?
    truthy = np.all(truthyList)

    assert truthy == True,"[@Cells data:] Cell data not equal from TFC and CFT! Check tracer selections!"

def test_IndividualTracer():
    """
    Test the correct number of Data points are selected by GetIndividualCellFromTracer.
    """
    total = 1000
    TracerNumber = np.arange(start=0,stop=total)

    CellIndex, SelectedTracers1 = GetIndividualCellFromTracer(Tracers,Parents,CellIDsCFT,TracerNumber)

    assert len(SelectedTracers1) == len(TracerNumber),"[@Individual Tracer:] Selected Tracers not same length as TracerNumber - Initial Selection"

    CellIndexPost, _ = GetIndividualCellFromTracer(Tracers,Parents,CellIDsCFT,TracerNumber,SelectedTracers=SelectedTracers1)

    assert len(CellIndex) == len(CellIndexPost),"[@Individual Tracer:] CellIDs on Pre not same as CellIDs on Post- Selected from SelectedTracers"

    data = CellsCFT['T'][CellIndexPost]

    assert len(data) == len(CellIndexPost),"[@Individual Tracer:] Cells selected not equal to indended size of subset!"

def test_ParentsTracers():
    """
    Test that parents has the same shape as tracers
    """
    assert np.shape(Parents) == np.shape(Tracers),"[@Parents Tracers:] Parents not same shape as tracers. Check GetTracersFromCells and GetCellsFromTracers!"

def test_ParentsCellIDs():
    """
    Test that all parents are contained within CellIDsCFT
    """
    truthy = np.all(np.isin(CellIDsCFT,Parents))

    assert truthy == True,"[@ParentsCellIDs:] Not all Parents are contained in CellIDsCFT!"
