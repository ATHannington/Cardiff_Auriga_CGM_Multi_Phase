"""
Author: A. T. Hannington
Created: 12/03/2020
Known Bugs:
    None
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
from Snapper import *


def GetTracersFromCells(snapGas, snapTracers,Cond):
    print("GetTracersFromCells")

    #Select Cell IDs for cells which meet condition
    CellIDs = snapGas.id[Cond]

    #Select Parent IDs in Cond list
    #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
    ParentsIndices = np.where(np.isin(snapTracers.prid,CellIDs))

    #Select Tracers and Parent IDs from cells that meet condition and contain tracers
    Tracers = snapTracers.trid[ParentsIndices]
    Parents = snapTracers.prid[ParentsIndices]

    #Get CellIDs for cells which meet condition AND contain tracers
    CellsIndices = np.where(np.isin(snapGas.id,Parents))
    CellIDs = snapGas.id[CellsIndices]

    #Select the data for Cells that meet Cond which contain tracers
    #   Does this by creating new dict from old data.
    #       Only selects values at index where Cell meets cond and contains tracers
    Cells={}
    for key, value in snapGas.data.items():
        if value is not None:
                Cells.update({key: value[CellsIndices]})

    return Tracers, Cells, CellIDs, Parents

#------------------------------------------------------------------------------#
def GetCellsFromTracers(snapGas, snapTracers,Tracers):
    print("GetCellsFromTracers")

    #Select indices (positions in array) of Tracer IDs which are in the Tracers list
    TracersIndices = np.where(np.isin(snapTracers.trid,Tracers))

    #Select the matching parent cell IDs for tracers which are in Tracers list
    Parents = snapTracers.prid[TracersIndices]

    #Select Tracers which are in the original tracers list (thus their original cells met condition and contained tracers)
    TracersCFT = snapTracers.trid[TracersIndices]

    #Select Cell IDs which are in Parents
    #   NOTE:   This selection causes trouble. Selecting only Halo=HaloID means some Parents now aren't associated with Halo
    #           This means some parents and tracers need to be dropped as they are no longer in desired halo.
    CellsIndices = np.where(np.isin(snapGas.id,Parents))
    CellIDs = snapGas.id[CellsIndices]

    #So, from above issue: Select Parents and Tracers which are associated with Desired Halo ONLY!
    ParentsIndices = np.where(np.isin(Parents,snapGas.id))
    Parents = Parents[ParentsIndices]
    TracersCFT = TracersCFT[ParentsIndices]


    #Select data from cells which contain tracers
    #   Does this by making a new dictionary from old data. Only selects values
    #       At indices of Cells which contain tracers in tracers list.
    Cells={}
    for key, value in snapGas.data.items():
        if value is not None:
                Cells.update({key: value[CellsIndices]})

    return TracersCFT, Cells, CellIDs, Parents

#------------------------------------------------------------------------------#
##  FvdV weighted percentile code:
#------------------------------------------------------------------------------#
def weightedperc(data, weights, perc):
    #percentage to decimal
    perc /= 100.

    #Indices of data array in sorted form
    ind_sorted = np.argsort(data)

    #Sort the data
    sorted_data = np.array(data)[ind_sorted]

    #Sort the weights by the sorted data sorting
    sorted_weights = np.array(weights)[ind_sorted]

    #Find the cumulative sum of the weights
    cm = np.cumsum(sorted_weights)

    #Find indices where cumulative some as a fraction of final cumulative sum value is greater than percentage
    whereperc = np.where(cm/float(cm[-1]) >= perc)

    #Reurn the first data value where above is true
    return sorted_data[whereperc[0][0]]

#------------------------------------------------------------------------------#
def GetIndividualCellFromTracer(Tracers,Parents,CellIDs,SelectedTracers):


    Selection = np.where(np.isin(Tracers,SelectedTracers))
    CellIDNumber = Parents[Selection]
    SubsetSelectedTracers = Tracers[Selection]

    CellIndex = np.array([])
    for ID in CellIDNumber:
        value = np.where(np.isin(CellIDs,ID))
        CellIndex = np.append(CellIndex, value)

    CellIndex = list(map(int, CellIndex))

    return CellIndex, SubsetSelectedTracers
#------------------------------------------------------------------------------#
def ConvertUnits(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0):
    #Density is rho/ <rho> where <rho> is average baryonic density
    rhocrit = 3. * (snapGas.omega0 * (1+snapGas.redshift)**3. + snapGas.omegalambda) * (snapGas.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)
    rhomean = 3. * (snapGas.omega0 * (1+snapGas.redshift)**3.) * (snapGas.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)

    meanweight = sum(snapGas.gmet[:,0:9], axis = 1) / ( sum(snapGas.gmet[:,0:9]/elements_mass[0:9], axis = 1) + snapGas.ne*snapGas.gmet[:,0] )
    Tfac = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53

    gasdens = snapGas.rho / (c.parsec*1e6)**3. * c.msol * 1e10
    gasX = snapGas.gmet[:,0]

    snapGas.data['T'] = snapGas.u / Tfac # K
    snapGas.data['n_H'] = gasdens / c.amu * gasX # cm^-3
    snapGas.data['dens'] = gasdens / (rhomean * omegabaryon0/snapGas.omega0) # rho / <rho>
    snapGas.data['Tdens'] = snapGas.data['T'] *snapGas.data['dens']

    bfactor = 1e6*(np.sqrt(1e10 * c.msol) / np.sqrt(c.parsec * 1e6)) * (1e5 / (c.parsec * 1e6)) #[microGauss]
    snapGas.data['B'] = np.linalg.norm((snapGas.data['bfld'] * bfactor), axis=1)

    snapGas.data['R'] =  np.linalg.norm(snapGas.data['pos'], axis=1)

    return snapGas
#------------------------------------------------------------------------------#
def HaloOnlyGasSelect(snapGas,snap_subfind,Halo=0):
    #Find length of the first n entries of particle type 0 that are associated with HaloID 0: ['HaloID', 'particle type']
    gaslength = snap_subfind.data['slty'][Halo,0]
    # tracerlength = snap_subfind.data['slty'][6,0]
    # print(tracerlength)
    # print(foo)
    #Take only data from above HaloID
    for key, value in snapGas.data.items():
        if (snapGas.data[key] is not None):
            snapGas.data[key] = snapGas.data[key][:gaslength]

    return snapGas
#------------------------------------------------------------------------------#
def LoadTracersParameters(TracersParamsPath):

    TRACERSPARAMS = pd.read_csv(TracersParamsPath, delimiter=" ", header=None, \
    usecols=[0,1],skipinitialspace=True, index_col=0, comment="#").to_dict()[1]

    #Convert Dictionary items to (mostly) floats
    for key, value in TRACERSPARAMS.items():
        if ((key != 'targetTLst') & (key != 'simfile')):
            #Convert values to floats
            TRACERSPARAMS.update({key:float(value)})
        elif (key == 'targetTLst'):
            #Convert targetTLst to list of floats
            lst = value.split(",")
            lst2 = [float(item) for item in lst]
            TRACERSPARAMS.update({key:lst2})
        elif (key == 'simfile'):
            #Keep simfile as a string
            TRACERSPARAMS.update({key:value})

    #Get Temperatures as strings in a list so as to form "4-5-6" for savepath.
    Tlst = [str(int(item)) for item in TRACERSPARAMS['targetTLst']]
    Tstr = '-'.join(Tlst)

    #This rather horrible savepath ensures the data can only be combined with the right input file, TracersParams.csv, to be plotted/manipulated
    DataSavepath = f"Data_snap{int(TRACERSPARAMS['snapnum'])}_min{int(TRACERSPARAMS['snapMin'])}_max{int(TRACERSPARAMS['snapMax'])}" +\
        f"_{int(TRACERSPARAMS['Rinner'])}R{int(TRACERSPARAMS['Router'])}_targetT{Tstr}"+\
        f"_deltaT{int(TRACERSPARAMS['deltaT'])}"

    return TRACERSPARAMS, DataSavepath, Tlst
