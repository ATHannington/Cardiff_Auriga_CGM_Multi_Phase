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
from random import sample

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


#------------------------------------------------------------------------------#


TracersTFC, CellsTFC, CellIDsTFC, ParentsTFC = GetTracersFromCells(snapGas, snapTracers,Cond)

TracersCFTinit, CellsCFTinit, CellIDsCFTinit, ParentsCFTinit = GetCellsFromTracers(snapGas, snapTracers,TracersTFC)

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

TracersCFT, CellsCFT, CellIDsCFT, ParentsCFT = GetCellsFromTracers(snapGas, snapTracers,TracersTFC)

#==============================================================================#
#
#                                  TESTING
#
#==============================================================================#

def test_SameSnapCellIDs():
    """
    Test that the Cell IDs selected from tracers match the CellIDs containing tracers when tracers are selected.
    """
    CellIDMatch = np.all(np.isin(CellIDsTFC,CellIDsCFTinit))

    assert CellIDMatch == True,"[@CellIDMatch:] Cell IDs not equal! TFC and CFT! Check tracer selections!"


def test_SameSnapCellData():
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


def test_SameSnapTracersParents():
    """
    Test whether Parents and Tracers returned from CFT match those from TFC when applied to same snapshot data.
    """
    truthy = np.isin(TracersCFTinit,TracersTFC)
    assert np.all(truthy) == True,"[@Same Snap Tracers Parents:] Not all Tracers CFT init found in TracersTFC!"

    truthy = np.isin(ParentsCFTinit,ParentsTFC)
    assert np.all(truthy) == True,"[@Same Snap Tracers Parents:] Not all Parents CFT init found in ParentsTFC!"


def test_ParentsMatchTracers():
    """
    Test that there are the same number of prids (parent ids) as trids (tracer ids).
    """
    assert np.shape(ParentsTFC) == np.shape(TracersTFC),"[@Parents Match Tracers:] ParentsTFC different shape to TracersTFC!"
    assert np.shape(ParentsCFTinit) == np.shape(TracersCFTinit),"[@Parents Match Tracers:] ParentsCFT init different shape to TracersCFT init!"
    assert np.shape(ParentsCFT) == np.shape(TracersCFT),"[@Parents Match Tracers:] ParentsCFT different shape to TracersCFT!"


def test_DwindlingParentsAndTracers():
    """
    Test that we are losing or maintaining trid and prid number, but not increasing. Also test that all TracersCFT are a subset of Tracers TFC.
    We should be finding that this subset is the same size or smaller, but never bigger or including a new value.
    """
    assert np.shape(TracersCFT)[0] <= np.shape(TracersTFC)[0],"[@Dwindling Parents and Tracers:] TracersCFT not <= in shape than TracersTFC!"
    assert np.shape(ParentsCFT)[0] <= np.shape(ParentsTFC)[0],"[@Dwindling Parents and Tracers:] ParentsCFT not <= in shape than ParentsTFC!"
    assert np.all(np.isin(TracersCFT,TracersTFC)) == True,"[@Dwindling Parents and Tracers:] TracersCFT not a subset of TracersTFC!"

def test_ParentsInCellIDs():
    """
    Test that all Parent IDs, prids, are contained in the CellIDs data. This should be a many-to-one super-set.
    i.e. there may be duplicate CellIDs in Parents but every Parent should match at least one Cell ID.
    """
    truthy = np.all(np.isin(ParentsTFC,CellIDsTFC))
    assert truthy == True,"[@Parents in Cell IDs:] ParentsTFC not many-to-one super-set of CellIDsTFC!"

    truthy = np.all(np.isin(ParentsCFTinit,CellIDsCFTinit))
    assert truthy == True,"[@Parents in Cell IDs:] ParentsCFT init not many-to-one super-set of CellIDsCFT init!"

    truthy = np.all(np.isin(ParentsCFT,CellIDsCFT))
    assert truthy == True,"[@Parents in Cell IDs:] ParentsCFT not many-to-one super-set of CellIDsCFT!"

def test_CellsShapes():
    """
    Test that Cells Data has consistent shape with the number of Cell IDs. This ensures all data has been correctly selected.
    """
    truthyList =[]
    for key, values in CellsTFC.items():
        truthyList.append(np.shape(values)[0] == np.shape(CellIDsTFC)[0])

    truthy = np.all(truthyList)
    assert truthy == True,"[@Cells Shapes:] values of Cells TFC not consistent shape to CellIDsTFC! Some data may be missing!"

    truthyList =[]
    for key, values in CellsCFTinit.items():
        truthyList.append(np.shape(values)[0] == np.shape(CellIDsCFTinit)[0])

    truthy = np.all(truthyList)
    assert truthy == True,"[@Cells Shapes:] values of Cells CFT init not consistent shape to CellIDsCFTinit! Some data may be missing!"

    truthyList =[]
    for key, values in CellsCFT.items():
        truthyList.append(np.shape(values)[0] == np.shape(CellIDsCFT)[0])

    truthy = np.all(truthyList)
    assert truthy == True,"[@Cells Shapes:] values of Cells CFT not consistent shape to CellIDsCFT! Some data may be missing!"


def test_IndividualTracer():
    """
    Test that the returned tracers from GetIndividualCellFromTracer are a subset of the SelectedTracers. Also that the data
    returned is of shape == subset. There are NaNs where the SelectedTracer is no longer present in the data.
    """
    subset = 1000

    rangeMin = 0
    rangeMax = len(snapGas.data['T'])
    TracerNumberSelect = np.arange(start=rangeMin, stop = rangeMax, step = 1 )
    TracerNumberSelect = sample(TracerNumberSelect.tolist(),min(subset,rangeMax))

    SelectedTracers1 = snapTracers.data['trid'][TracerNumberSelect]

    data, massData, TracersReturned = GetIndividualCellFromTracer(Tracers=snapTracers.data['trid'],\
    Parents=snapTracers.data['prid'],CellIDs=snapGas.data['id'],SelectedTracers=SelectedTracers1,\
    Data=snapGas.data['T'],mass=snapGas.data['mass'])

    assert np.shape(data)[0] == subset,"[@Individual Tracer:] returned data not size == subset! Some data/NaNs may be missing!"
    assert np.shape(massData)[0] == subset,"[@Individual Tracer:] returned mass data not size == subset! Some data/NaNs may be missing!"

    assert np.all(np.isin(TracersReturned,SelectedTracers1)) == True,"[@Individual Tracer:] Tracers Returned is not a subset of Selected Tracers! Some Tracers Returned have been mis-selected!"
    assert np.shape(TracersReturned)[0] <= subset,"[@Individual Tracer:] Tracers Returned is not of size <= subset! There may be too many Returned Tracers!"
