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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import const as c
from gadget import *
from gadget_subfind import *
import h5py
import sys
import logging
import math
import random

#==============================================================================#
#       MAIN ANALYSIS CODE - IN FUNC FOR MULTIPROCESSING
#==============================================================================#
def snap_analysis(snapNumber,targetT,TRACERSPARAMS,HaloID,TracersTFC,\
elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,lazyLoadBool=True):
    print("")
    print(f"Starting Snap {snapNumber}")

    # load in the subfind group files
    snap_subfind = load_subfind(snapNumber,dir=TRACERSPARAMS['simfile'])

    # load in the gas particles mass and position only for HaloID 0.
    #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    snapGas     = gadget_readsnap(snapNumber, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0,4], lazy_load=lazyLoadBool, subfind = snap_subfind)
    # load tracers data
    snapTracers = gadget_readsnap(snapNumber, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [6], lazy_load=lazyLoadBool)

    #Load Cell IDs - avoids having to turn lazy_load off...
    # But ensures 'id' is loaded into memory before HaloOnlyGasSelect is called
    #  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
    #   Be in memory so taking the subset would be skipped.
    tmp = snapGas.data['id']
    tmp = snapGas.data['age']
    del tmp

    print(f"[@{snapNumber}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

    #Centre the simulation on HaloID 0
    snapGas  = SetCentre(snap=snapGas,snap_subfind=snap_subfind,HaloID=HaloID)

    #--------------------------#
    ##    Units Conversion    ##
    #--------------------------#

    print(f"[@{snapNumber}]: Calculating Tracked Parameters!")

    #Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos *= 1e3 #[kpc]
    snapGas.vol *= 1e9 #[kpc^3]

    #Calculate New Parameters and Load into memory others we want to track
    snapGas = CalculateTrackedParameters(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0)

    #Pad stars and gas data with Nones so that all keys have values of same first dimension shape
    snapGas = PadNonEntries(snapGas)
    #Find Halo=HaloID data for only selection snapshot. This ensures the
    #selected tracers are originally in the Halo, but allows for tracers
    #to leave (outflow) or move inwards (inflow) from Halo.

    if (snapNumber == int(TRACERSPARAMS['selectSnap'])):
        print(f"[@{snapNumber}]:Finding Halo 0 Only Data!")

        snapGas = HaloOnlyGasSelect(snapGas,snap_subfind,Halo=HaloID)

    ###
    ##  Selection   ##
    ###

    #Select Cells which have the tracers from the selection snap in them
    TracersCFT, CellsCFT, CellIDsCFT, ParentsCFT = GetCellsFromTracers(snapGas, snapTracers,TracersTFC,saveParams,saveTracersOnly,snapNumber)

    # #Add snap data to temperature specific dictionary
    # print(f"Adding (T{int(targetT)},{int(snap)}) to Dict")
    # FullDict.update({(f"T{int(targetT)}",f"{int(snap)}"): CellsCFT})
    out = {(f"T{int(targetT)}",f"{int(snapNumber)}"): CellsCFT}

    savePath = DataSavepath + f"_T{int(targetT)}_{int(snapNumber)}"+ FullDataPathSuffix

    print("\n" + f"[@{snapNumber} @T{int(targetT)}]: Saving Tracers data as: "+ savePath)

    hdf5_save(savePath,out)

    save_statistics(CellsCFT, targetT, snapNumber, TRACERSPARAMS, saveParams, DataSavepath, MiniDataPathSuffix)

    if (TRACERSPARAMS['TracerPlotBool'] == True) or (TRACERSPARAMS['QuadPlotBool'] == True):
        #Maximise Threads in SERIAL
        if (TRACERSPARAMS['TracerPlotBool'] == True) :
            nThreads = 16
        else:
            nThreads = 2

        #Only perform Quad plot for one temperature as is same for each temp
        if(int(targetT) == int(TRACERSPARAMS['targetTLst'][0])):
            quadBool = TRACERSPARAMS['QuadPlotBool']
        else:
            quadBool = False

        PlotProjections(snapGas,out,snapNumber,targetT,TRACERSPARAMS, DataSavepath,\
        FullDataPathSuffix, Axes=[0,2], zAxis=[1], QuadPlotBool=quadBool,\
        TracerPlotBool=TRACERSPARAMS['TracerPlotBool'], numThreads=nThreads)

    sys.stdout.flush()
    return {"out": out, "TracersCFT": TracersCFT, "CellsCFT": CellsCFT, "CellIDsCFT": CellIDsCFT, "ParentsCFT" : ParentsCFT}

#==============================================================================#
#       PRE-MAIN ANALYSIS CODE
#==============================================================================#
def tracer_selection_snap_analysis(targetT,TRACERSPARAMS,HaloID,\
elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,\
lazyLoadBool=True,SUBSET=None,snapNumber=None,saveTracers=True,loadonlyhalo=True):

    if snapNumber is None:
        snapNumber = TRACERSPARAMS['selectSnap']

    print(f"Starting T = {targetT} Analysis!")

    # load in the subfind group files
    snap_subfind = load_subfind(snapNumber,dir=TRACERSPARAMS['simfile'])

    # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
    snapGas     = gadget_readsnap(snapNumber, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0,4], loadonlyhalo = HaloID, lazy_load=lazyLoadBool, subfind = snap_subfind)
    snapTracers = gadget_readsnap(snapNumber, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [6], lazy_load=lazyLoadBool)

    #Load Cell IDs - avoids having to turn lazy_load off...
    # But ensures 'id' is loaded into memory before HaloOnlyGasSelect is called
    #  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
    #   Be in memory so taking the subset would be skipped.
    tmp = snapGas.data['id']
    tmp = snapGas.data['age']
    del tmp

    print(f"SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

    #Centre the simulation on HaloID 0
    snapGas  = SetCentre(snap=snapGas,snap_subfind=snap_subfind,HaloID=HaloID)

    #--------------------------#
    ##    Units Conversion    ##
    #--------------------------#
    print("Calculating Tracked Parameters!")

    #Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos *= 1e3 #[kpc]
    snapGas.vol *= 1e9 #[kpc^3]

    #Calculate New Parameters and Load into memory others we want to track
    snapGas = CalculateTrackedParameters(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0)

    #Pad stars and gas data with Nones so that all keys have values of same first dimension shape
    snapGas = PadNonEntries(snapGas)

    ### Exclude values outside halo 0 ###
    if (loadonlyhalo is True):
        print(f"Finding Halo {int(HaloID)} Only Data!")

        snapGas = HaloOnlyGasSelect(snapGas,snap_subfind,Halo=HaloID)

    #--------------------------------------------------------------------------#
    ####                    SELECTION                                        ###
    #--------------------------------------------------------------------------#
    print("Setting Selection Condition!")

    #Set condition for Tracer selection
    whereGas = np.where(snapGas.type==0)
    whereStars = np.where(snapGas.type==4)
    NGas = len(snapGas.type[whereGas])

    StarsSelect = np.where((snapGas.data['R']>=TRACERSPARAMS['Rinner']) & \
                        (snapGas.data['R']<=TRACERSPARAMS['Router']) &\
                        (snapGas.type == 4) )

    GasSelect = np.where((snapGas.data['T'][whereGas]>=1.*10**(targetT-TRACERSPARAMS['deltaT'])) & \
                    (snapGas.data['T'][whereGas]<=1.*10**(targetT+TRACERSPARAMS['deltaT'])) & \
                    (snapGas.data['R'][whereGas]>=TRACERSPARAMS['Rinner']) & \
                    (snapGas.data['R'][whereGas]<=TRACERSPARAMS['Router']) &\
                    (snapGas.data['sfr'][whereGas]<=0))

    Cond =np.array(StarsSelect[0].tolist() + GasSelect[0].tolist())

    #Get Cell data and Cell IDs from tracers based on condition
    TracersTFC, CellsTFC, CellIDsTFC, ParentsTFC = GetTracersFromCells(snapGas, snapTracers,Cond,saveParams,saveTracersOnly,snapNumber=snapNumber)

    # #Add snap data to temperature specific dictionary
    # print(f"Adding (T{int(targetT)},{int(snap)}) to Dict")
    # FullDict.update({(f"T{int(targetT)}",f"{int(snap)}"): CellsCFT})
    if (saveTracers is True):
        out = {(f"T{int(targetT)}",f"{int(snapNumber)}"): {'trid': TracersTFC}}

        savePath = DataSavepath + f"_T{int(targetT)}_{int(snapNumber)}_Tracers"+ FullDataPathSuffix

        print("\n" + f"[@{int(snapNumber)} @T{int(targetT)}]: Saving Tracers ID ('trid') data as: "+ savePath)

        hdf5_save(savePath,out)

    #SUBSET
    if (SUBSET is not None):
        print(f"*** TRACER SUBSET OF {SUBSET} TAKEN! ***")
        TracersTFC = TracersTFC[:SUBSET]

    return TracersTFC, CellsTFC, CellIDsTFC, ParentsTFC, snapGas, snapTracers
#------------------------------------------------------------------------------#

#==============================================================================#
#       t3000 MAIN ANALYSIS CODE - IN FUNC FOR MULTIPROCESSING
#==============================================================================#
def t3000_snap_analysis(snapNumber,targetT,TRACERSPARAMS,HaloID,CellIDsTFC,\
elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,lazyLoadBool=True):
    print("")
    print(f"Starting Snap {snapNumber}")

    # load in the subfind group files
    snap_subfind = load_subfind(snapNumber,dir=TRACERSPARAMS['simfile'])

    # load in the gas particles mass and position only for HaloID 0.
    #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    snapGas     = gadget_readsnap(snapNumber, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0,4], lazy_load=lazyLoadBool, subfind = snap_subfind)

    #Load Cell IDs - avoids having to turn lazy_load off...
    # But ensures 'id' is loaded into memory before HaloOnlyGasSelect is called
    #  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
    #   Be in memory so taking the subset would be skipped.
    tmp = snapGas.data['id']
    tmp = snapGas.data['age']
    del tmp

    print(f"[@{snapNumber}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

    #Centre the simulation on HaloID 0
    snapGas  = SetCentre(snap=snapGas,snap_subfind=snap_subfind,HaloID=HaloID)

    #--------------------------#
    ##    Units Conversion    ##
    #--------------------------#

    print(f"[@{snapNumber}]: Calculating Tracked Parameters!")

    #Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos *= 1e3 #[kpc]
    snapGas.vol *= 1e9 #[kpc^3]

    #Calculate New Parameters and Load into memory others we want to track
    snapGas = CalculateTrackedParameters(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0)

    #Pad stars and gas data with Nones so that all keys have values of same first dimension shape
    snapGas = PadNonEntries(snapGas)
    #Find Halo=HaloID data for only selection snapshot. This ensures the
    #selected tracers are originally in the Halo, but allows for tracers
    #to leave (outflow) or move inwards (inflow) from Halo.

    if (snapNumber == int(TRACERSPARAMS['selectSnap'])):
        print(f"[@{snapNumber}]:Finding Halo 0 Only Data!")

        snapGas = HaloOnlyGasSelect(snapGas,snap_subfind,Halo=HaloID)

    ###
    ##  Selection   ##
    ###

    whereCellsSelected = np.where(np.isin(snapGas.data['id'],CellIDsTFC))

    for key, value in snapGas.data.items():
        if (value is not None):
            snapGas.data[key] = value[whereCellsSelected]

    Rcrit = 500.

    print(f"Select approx HaloID = {int(HaloID)} by R<={Rcrit:0.02f} kpc")
    Cond = np.where(snapGas.data['R']<=Rcrit)

    for key, value in snapGas.data.items():
        if (value is not None):
            snapGas.data[key] = value[Cond]

    CellIDsCFT = snapGas.data['id']

    print("Selected!")
    print("Entering save Cells...")

    CellsCFT = t3000_saveCellsData(snapGas,snapNumber,saveParams,saveTracersOnly)
    # #Add snap data to temperature specific dictionary
    # print(f"Adding (T{int(targetT)},{int(snap)}) to Dict")
    # FullDict.update({(f"T{int(targetT)}",f"{int(snap)}"): CellsCFT})
    out = {(f"T{int(targetT)}",f"{int(snapNumber)}"): CellsCFT}

    savePath = DataSavepath + f"_T{int(targetT)}_{int(snapNumber)}"+ FullDataPathSuffix

    print("\n" + f"[@{snapNumber} @T{int(targetT)}]: Saving Cells data as: "+ savePath)

    hdf5_save(savePath,out)

    save_statistics(CellsCFT, targetT, snapNumber, TRACERSPARAMS, saveParams, DataSavepath, MiniDataPathSuffix)

    sys.stdout.flush()
    return {"out": out, "CellsCFT": CellsCFT, "CellIDsCFT": CellIDsCFT}

#==============================================================================#
#       PRE-MAIN ANALYSIS CODE
#==============================================================================#
def t3000_cell_selection_snap_analysis(targetT,TRACERSPARAMS,HaloID,\
elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,\
lazyLoadBool=True,SUBSET=None,snapNumber=None,saveCells=True,loadonlyhalo=True):

    if snapNumber is None:
        snapNumber = TRACERSPARAMS['selectSnap']

    print(f"Starting T = {targetT} Analysis!")

    # load in the subfind group files
    snap_subfind = load_subfind(snapNumber,dir=TRACERSPARAMS['simfile'])

    # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
    snapGas     = gadget_readsnap(snapNumber, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0,4], loadonlyhalo = HaloID, lazy_load=lazyLoadBool, subfind = snap_subfind)

    #Load Cell IDs - avoids having to turn lazy_load off...
    # But ensures 'id' is loaded into memory before HaloOnlyGasSelect is called
    #  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
    #   Be in memory so taking the subset would be skipped.
    tmp = snapGas.data['id']
    tmp = snapGas.data['age']
    del tmp

    print(f"SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

    #Centre the simulation on HaloID 0
    snapGas  = SetCentre(snap=snapGas,snap_subfind=snap_subfind,HaloID=HaloID)

    #--------------------------#
    ##    Units Conversion    ##
    #--------------------------#
    print("Calculating Tracked Parameters!")

    #Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos *= 1e3 #[kpc]
    snapGas.vol *= 1e9 #[kpc^3]

    #Calculate New Parameters and Load into memory others we want to track
    snapGas = CalculateTrackedParameters(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0)

    #Pad stars and gas data with Nones so that all keys have values of same first dimension shape
    snapGas = PadNonEntries(snapGas)

    ### Exclude values outside halo 0 ###
    if (loadonlyhalo is True):
        print(f"Finding Halo {int(HaloID)} Only Data!")

        snapGas = HaloOnlyGasSelect(snapGas,snap_subfind,Halo=HaloID)

    #--------------------------------------------------------------------------#
    ####                    SELECTION                                        ###
    #--------------------------------------------------------------------------#
    print("Setting Selection Condition!")

    #Set condition for Tracer selection
    whereGas = np.where(snapGas.type==0)
    whereStars = np.where(snapGas.type==4)
    NGas = len(snapGas.type[whereGas])

    StarsSelect = np.where((snapGas.data['R']>=TRACERSPARAMS['Rinner']) & \
                        (snapGas.data['R']<=TRACERSPARAMS['Router']) &\
                        (snapGas.type == 4) )

    GasSelect = np.where((snapGas.data['T'][whereGas]>=1.*10**(targetT-TRACERSPARAMS['deltaT'])) & \
                    (snapGas.data['T'][whereGas]<=1.*10**(targetT+TRACERSPARAMS['deltaT'])) & \
                    (snapGas.data['R'][whereGas]>=TRACERSPARAMS['Rinner']) & \
                    (snapGas.data['R'][whereGas]<=TRACERSPARAMS['Router']) &\
                    (snapGas.data['sfr'][whereGas]<=0))

    Cond =np.array(StarsSelect[0].tolist() + GasSelect[0].tolist())

    for key, value in snapGas.data.items():
        if (value is not None):
            snapGas.data[key] = value[Cond]

    CellIDsTFC = snapGas.data['id']
    # #Add snap data to temperature specific dictionary
    # print(f"Adding (T{int(targetT)},{int(snap)}) to Dict")
    # FullDict.update({(f"T{int(targetT)}",f"{int(snap)}"): CellsCFT})
    if (saveCells is True):
        out = {(f"T{int(targetT)}",f"{int(snapNumber)}"): {'id': CellIDsTFC}}

        savePath = DataSavepath + f"_T{int(targetT)}_{int(snapNumber)}_CellIDs"+ FullDataPathSuffix

        print("\n" + f"[@{int(snapNumber)} @T{int(targetT)}]: Saving Cell ID ('id') data as: "+ savePath)

        hdf5_save(savePath,out)

    #SUBSET
    if (SUBSET is not None):
        print(f"*** TRACER SUBSET OF {SUBSET} TAKEN! ***")
        TracersTFC = TracersTFC[:SUBSET]

    return CellIDsTFC, snapGas

#------------------------------------------------------------------------------#

def GetTracersFromCells(snapGas, snapTracers,Cond,saveParams,saveTracersOnly,snapNumber):
    print(f"[@{snapNumber}]: Get Tracers From Cells!")

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

    # Save number of tracers
    Ntracers = int(len(Tracers))
    print(f"[@{snapNumber}]: Number of tracers = {Ntracers}")

    Cells = saveTracerData(snapGas,Tracers,Parents,CellIDs,CellsIndices,Ntracers,snapNumber,saveParams,saveTracersOnly)

    return Tracers, Cells, CellIDs, Parents

#------------------------------------------------------------------------------#
def GetCellsFromTracers(snapGas, snapTracers,Tracers,saveParams,saveTracersOnly,snapNumber):
    print(f"[@{snapNumber}]: Get Cells From Tracers!")

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

    # Save number of tracers
    Ntracers = int(len(TracersCFT))
    print(f"[@{snapNumber}]: Number of tracers = {Ntracers}")

    Cells = saveTracerData(snapGas,TracersCFT,Parents,CellIDs,CellsIndices,Ntracers,snapNumber,saveParams,saveTracersOnly)

    return TracersCFT, Cells, CellIDs, Parents

#------------------------------------------------------------------------------#
def saveTracerData(snapGas,Tracers,Parents,CellIDs,CellsIndices,Ntracers,snapNumber,saveParams,saveTracersOnly):

    print(f"[@{snapNumber}]: Saving Tracer Data!")

    #Select the data for Cells that meet Cond which contain tracers
    #   Does this by creating new dict from old data.
    #       Only selects values at index where Cell meets cond and contains tracers
    Cells={}
    for key in saveParams:
        Cells.update({key: snapGas.data[key][CellsIndices]})

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    #   Now perform save of parameters not tracked in stats (saveTracersOnly params)#
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    #Redshift
    redshift = snapGas.redshift        #z
    aConst = 1. / (1. + redshift)   #[/]

    #Get lookback time in Gyrs
    #[0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[0] #[Gyrs]

    for TracerSaveParameter in saveTracersOnly:
        if (TracerSaveParameter == 'Lookback'):
            Cells.update({'Lookback' : np.array([lookback]) })
        elif (TracerSaveParameter == 'Ntracers'):
            Cells.update({'Ntracers' : np.array([Ntracers])})
        elif (TracerSaveParameter == 'Snap'):
            Cells.update({'Snap' : np.array([snapNumber])})
        elif (TracerSaveParameter == 'trid'):
            #Save Tracer IDs
            Cells.update({'trid':Tracers})
        elif (TracerSaveParameter == 'prid'):
            #Save Parent Cell IDs
            Cells.update({'prid':Parents})
        elif (TracerSaveParameter == 'id'):
            #Save Cell IDs
            Cells.update({'id':CellIDs})
        else:
            Cells.update({f'{TracerSaveParameter}' : snapGas.data[TracerSaveParameter][CellsIndices]})

    return Cells

def t3000_saveCellsData(snapGas,snapNumber,saveParams,saveTracersOnly):

    print(f"[@{snapNumber}]: Saving Cell Data!")

    Ncells = len(snapGas.data['id'])

    print(f"[@{snapNumber}]: Ncells = {int(Ncells)}")

    #Select the data for Cells that meet Cond which contain tracers
    #   Does this by creating new dict from old data.
    #       Only selects values at index where Cell meets cond and contains tracers
    Cells={}
    for key in saveParams:
        Cells.update({key: snapGas.data[key]})

    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    #   Now perform save of parameters not tracked in stats (saveTracersOnly params)#
    #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    #Redshift
    redshift = snapGas.redshift        #z
    aConst = 1. / (1. + redshift)   #[/]

    #Get lookback time in Gyrs
    #[0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[0] #[Gyrs]

    for TracerSaveParameter in saveTracersOnly:
        if (TracerSaveParameter == 'Lookback'):
            Cells.update({'Lookback' : np.array([lookback]) })
        elif (TracerSaveParameter == 'Ncells'):
            Cells.update({'Ncells' : np.array([Ncells])})
        elif (TracerSaveParameter == 'Snap'):
            Cells.update({'Snap' : np.array([snapNumber])})
        elif (TracerSaveParameter == 'id'):
            #Save Cell IDs
            Cells.update({'id':snapGas.data['id']})
        else:
            Cells.update({f'{TracerSaveParameter}' : snapGas.data[TracerSaveParameter]})

    return Cells
#------------------------------------------------------------------------------#
##  FvdV weighted percentile code:
#------------------------------------------------------------------------------#
def weightedperc(data, weights, perc,key):
    #percentage to decimal
    perc /= 100.

    #Indices of data array in sorted form
    ind_sorted = np.argsort(data)

    #Sort the data
    sorted_data = np.array(data)[ind_sorted]

    #Sort the weights by the sorted data sorting
    sorted_weights = np.array(weights)[ind_sorted]

    #Remove nan entries
    whereDataIsNotNAN = np.where(np.isnan(sorted_data)==False)

    sorted_data = sorted_data[whereDataIsNotNAN]
    sorted_weights = sorted_weights[whereDataIsNotNAN]

    whereWeightsIsNotNAN = np.where(np.isnan(sorted_weights)==False)
    sorted_weights = sorted_weights[whereWeightsIsNotNAN]

    nDataNotNan = len(sorted_data)
    nWeightsNotNan = len(sorted_weights)

    if (nDataNotNan>0):
        #Find the cumulative sum of the weights
        cm = np.cumsum(sorted_weights)

        #Find indices where cumulative some as a fraction of final cumulative sum value is greater than percentage
        whereperc = np.where(cm/float(cm[-1]) >= perc)

        #Reurn the first data value where above is true
        out = sorted_data[whereperc[0][0]]
    else:
        print(key)
        print("[@WeightPercent:] Warning! Data all nan! Returning 0 value!")
        out = np.array([0.])

    return out

#------------------------------------------------------------------------------#

def SetCentre(snap,snap_subfind,HaloID):
    print('Centering!')

    # subfind has calculated its centre of mass for you
    HaloCentre = snap_subfind.data['fpos'][HaloID,:]
    # use the subfind COM to centre the coordinates on the galaxy
    snap.data['pos'] = (snap.data['pos'] - np.array(HaloCentre))

    snap.data['R'] =  (np.linalg.norm(snap.data['pos'], axis=1))

    whereGas = np.where(snap.type==0)
    #Adjust to galaxy centred velocity
    wheredisc, = np.where((snap.data['R'][whereGas] < 20.) & (snap.data['sfr'] > 0.))
    snap.vel = snap.vel - np.median(snap.vel[wheredisc], axis = 0)
    return snap

#------------------------------------------------------------------------------#
def CalculateTrackedParameters(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0):
    whereGas = np.where(snapGas.type==0)
    #Density is rho/ <rho> where <rho> is average baryonic density
    rhocrit = 3. * (snapGas.omega0 * (1+snapGas.redshift)**3 + snapGas.omegalambda) * (snapGas.hubbleparam * 100.*1e5/(c.parsec*1e6))**2 / ( 8. * pi * c.G)
    rhomean = 3. * (snapGas.omega0 * (1+snapGas.redshift)**3) * (snapGas.hubbleparam * 100.*1e5/(c.parsec*1e6))**2 / ( 8. * pi * c.G)

    #Mean weight [amu]
    meanweight = sum(snapGas.gmet[whereGas,0:9][0], axis = 1) / ( sum(snapGas.gmet[whereGas,0:9][0]/elements_mass[0:9], axis = 1) + snapGas.ne*snapGas.gmet[whereGas,0][0] )

    #3./2. N KB
    Tfac = ((3./2.) * c.KB) / (meanweight * c.amu)

    snapGas.data['dens'] = (snapGas.rho / (c.parsec*1e6)**3) * c.msol * 1e10 #[g cm^-3]
    gasX = snapGas.gmet[whereGas,0][0]

    #Temperature = U / (3/2 * N KB) [K]
    snapGas.data['T'] = (snapGas.u*1e10) / (Tfac) # K
    snapGas.data['n_H'] = snapGas.data['dens']/ c.amu * gasX # cm^-3
    snapGas.data['rho_rhomean']  = snapGas.data['dens']/ (rhomean * omegabaryon0/snapGas.omega0) # rho / <rho>
    snapGas.data['Tdens'] = snapGas.data['T'] * snapGas.data['rho_rhomean']

    bfactor = 1e6*(np.sqrt(1e10 * c.msol) / np.sqrt(c.parsec * 1e6)) * (1e5 / (c.parsec * 1e6)) #[microGauss]

    #Magnitude of Magnetic Field [micro Guass]
    snapGas.data['B'] = np.linalg.norm((snapGas.data['bfld'] * bfactor), axis=1)

    #Radius [kpc]
    snapGas.data['R'] =  (np.linalg.norm(snapGas.data['pos'], axis=1)) #[Kpc]

    #Radial Velocity [km s^-1]
    KpcTokm = 1e3*c.parsec*1e-5
    snapGas.data['vrad'] = (snapGas.pos*KpcTokm*snapGas.vel).sum(axis=1)
    snapGas.data['vrad'] /= snapGas.data['R']*KpcTokm

    #Cooling time [Gyrs]
    GyrToSeconds = 365.25*24.*60.*60.*1e9
    snapGas.data['tcool'] = (snapGas.data['u'] * 1e10 * snapGas.data['dens']) / (GyrToSeconds * snapGas.data['gcol'] * snapGas.data['n_H']**2) #[Gyrs]
    snapGas.data['theat'] = snapGas.data['tcool'].copy()

    coolingGas = np.where(snapGas.data['tcool']<0.0)
    heatingGas = np.where(snapGas.data['tcool']>0.0)
    zeroChangeGas = np.where(snapGas.data['tcool']==0.0)

    snapGas.data['tcool'][coolingGas] = abs(snapGas.data['tcool'][coolingGas])
    snapGas.data['tcool'][heatingGas] = np.nan
    snapGas.data['tcool'][zeroChangeGas] = np.nan


    snapGas.data['theat'][coolingGas] = np.nan
    snapGas.data['theat'][heatingGas] = abs(snapGas.data['theat'][heatingGas])
    snapGas.data['theat'][zeroChangeGas] = np.nan


    #Load in metallicity
    snapGas.data['gz'] = snapGas.data['gz']/Zsolar
    #Load in Metals
    tmp = snapGas.data['gmet']
    #Load in Star Formation Rate
    tmp = snapGas.data['sfr']

    #Specific Angular Momentum [kpc km s^-1]
    snapGas.data['L'] = sqrt((cross(snapGas.data['pos'], snapGas.data['vel'])**2.).sum(axis=1))

    ndens = snapGas.data['dens']/ (meanweight * c.amu)
    #Thermal Pressure : P/k_B = n T [$ # K cm^-3]
    snapGas.data['P_thermal'] = ndens*snapGas.T

    #Magnetic Pressure [P/k_B K cm^-3]
    snapGas.data['P_magnetic'] = ((snapGas.data['B']*1e-6) **2)/( 8. * pi * c.KB)

    #Kinetic "Pressure" [P/k_B K cm^-3]
    snapGas.data['P_kinetic'] = (snapGas.rho / (c.parsec*1e6)**3) * 1e10 * c.msol *(1./c.KB) * (np.linalg.norm(snapGas.data['vel'][whereGas]*1e5, axis=1))**2

    #Sound Speed [(erg K^-1 K ??? g^-1)^1/2 = (g cm^2 s^-2 g^-1)^(1/2) = km s^-1]
    snapGas.data['csound'] = sqrt(((5./3.)*c.KB * snapGas.data['T'])/(meanweight*c.amu*1e5))

    # [cm kpc^-1 kpc cm^-1 s^1 = s / GyrToSeconds = Gyr]
    snapGas.data['tcross'] = (KpcTokm*1e3/GyrToSeconds) * (snapGas.data['vol'])**(1./3.) /snapGas.data['csound']

    #Free Fall time [Gyrs]
    snapGas.data['tff'] = sqrt(( 3. * pi )/(32.* c.G  * snapGas.data['dens'])) * (1./GyrToSeconds)

    #Cooling time over free fall time
    snapGas.data['tcool_tff'] = snapGas.data['tcool']/snapGas.data['tff']

    del tmp

    return snapGas
#------------------------------------------------------------------------------#
def HaloOnlyGasSelect(snapGas,snap_subfind,Halo=0):
    #Find length of the first n entries of particle type 0 that are associated with HaloID 0: ['HaloID', 'particle type']
    gaslength = snap_subfind.data['slty'][Halo,0]

    whereGas = np.where(snapGas.type==0)[0]
    whereStars = np.where(snapGas.type==4)[0]
    NGas = len(snapGas.type[whereGas])
    NStars = len(snapGas.type[whereStars])

    selectGas = [ii for ii in range(0,gaslength)]
    selectStars = [ii for ii in range(0,NStars)]

    selected = selectGas + selectStars

    #Take only data from above HaloID
    for key, value in snapGas.data.items():
        if (value is not None):
            snapGas.data[key] = value[selected]

    return snapGas
#------------------------------------------------------------------------------#
def LoadTracersParameters(TracersParamsPath):

    TRACERSPARAMS = pd.read_csv(TracersParamsPath, delimiter=" ", header=None, \
    usecols=[0,1],skipinitialspace=True, index_col=0, comment="#").to_dict()[1]

    #Convert Dictionary items to (mostly) floats
    for key, value in TRACERSPARAMS.items():
        if ((key == 'targetTLst')or(key == 'phasesSnaps')):
            #Convert targetTLst to list of floats
            lst = value.split(",")
            lst2 = [float(item) for item in lst]
            TRACERSPARAMS.update({key:lst2})
        elif ((key == 'simfile') or (key == 'savepath')):
            #Keep simfile as a string
            TRACERSPARAMS.update({key:value})
        else:
            #Convert values to floats
            TRACERSPARAMS.update({key:float(value)})

    if (TRACERSPARAMS['QuadPlotBool'] == 1.):
        TRACERSPARAMS['QuadPlotBool'] = True
    else:
        TRACERSPARAMS['QuadPlotBool'] = False

    if (TRACERSPARAMS['TracerPlotBool'] == 1.):
        TRACERSPARAMS['TracerPlotBool'] = True
    else:
        TRACERSPARAMS['TracerPlotBool'] = False

    #Get Temperatures as strings in a list so as to form "4-5-6" for savepath.
    Tlst = [str(int(item)) for item in TRACERSPARAMS['targetTLst']]
    Tstr = '-'.join(Tlst)

    #This rather horrible savepath ensures the data can only be combined with the right input file, TracersParams.csv, to be plotted/manipulated
    DataSavepath = TRACERSPARAMS['savepath'] + f"Data_selectSnap{int(TRACERSPARAMS['selectSnap'])}_min{int(TRACERSPARAMS['snapMin'])}_max{int(TRACERSPARAMS['snapMax'])}" +\
        f"_{int(TRACERSPARAMS['Rinner'])}R{int(TRACERSPARAMS['Router'])}_targetT{Tstr}"

    return TRACERSPARAMS, DataSavepath, Tlst

#------------------------------------------------------------------------------#

def GetIndividualCellFromTracer(Tracers,Parents,CellIDs,SelectedTracers,Data,mass,NullEntry=np.nan):

    #Select which of the SelectedTracers are in Tracers from this snap
    TracersTruthy = np.isin(SelectedTracers,Tracers)

    #Grab the indices of the trid in Tracers if it is contained in SelectedTracers
    #   Also add list of Tracer IDs trids to list for debugging purposes
    TracersIndices = []
    TracersReturned = []
    for ind, tracer in enumerate(SelectedTracers):
        truthy = np.isin(Tracers,tracer)
        if np.any(truthy) == True:
            TracersIndices.append(np.where(truthy)[0])
            TracersReturned.append(Tracers[np.where(truthy)])
        else:
            TracersIndices.append([np.nan])

    #If the tracer from SelectedTracers is in tracers, use the above indices to select its
    #   parent, and from there its cell, then grab data.
    #   If the tracer from SelectedTracers is not in tracers, put a nan value.
    #   This will allow plotting of all tracers, keeping saveData a fixed shape == subset/SelectedTracers
    saveData = []
    massData = []
    for (ind, element) in zip(TracersIndices,TracersTruthy):
        if element == True:
            parent = Parents[ind]
            dataIndex = np.where(np.isin(CellIDs,parent))

            dataEntry = Data[dataIndex].tolist()
            if (np.shape(dataEntry)[0] == 0):
                dataEntry = NullEntry
            else:
                dataEntry = dataEntry[0]

            massEntry = mass[dataIndex].tolist()
            if (np.shape(massEntry)[0] == 0):
                massEntry = NullEntry
            else:
                massEntry = massEntry[0]

            saveData.append(dataEntry)
            massData.append(massEntry)
        else:
            saveData.append(NullEntry)
            massData.append(NullEntry)

    return saveData, massData, TracersReturned

#------------------------------------------------------------------------------#

def hdf5_save(path,data):
    """
        Save nested dictionary as hdf5 file.
        Dictionary must have form:
            {(Meta-Key1 , Meta-Key2):{key1:... , key2: ...}}
        and will be saved in the form:
            {Meta-key1_Meta-key2:{key1:... , key2: ...}}
    """
    with h5py.File(path,"w") as f:
        for key, value in data.items():
            saveKey = None
            #Loop over Metakeys in tuple key of met-dictionary
            # Save this new metakey as one string, separated by '_'
            for entry in key:
                if saveKey is None:
                    saveKey = entry
                else:
                    saveKey = saveKey + "_"  + str(entry)
            #Create meta-dictionary entry with above saveKey
            #   Add to this dictionary entry a dictionary with keys from sub-dict
            #   and values from sub dict. Gzip for memory saving.
            grp = f.create_group(saveKey)
            for k, v in value.items():
                grp.create_dataset(k, data=v)

    return

#------------------------------------------------------------------------------#
def hdf5_load(path):
    """
        Load nested dictionary from hdf5 file.
        Dictionary will be saved in the form:
            {Meta-key1_Meta-key2:{key1:... , key2: ...}}
        and output in the following form:
            {(Meta-Key1 , Meta-Key2):{key1:... , key2: ...}}

    """
    loaded = h5py.File(path,'r')

    dataDict={}
    for key,value in loaded.items():
        #Split the meta-key back into a tuple format
        saveKey = tuple(key.split("_"))
        #Take the sub-dict out from hdf5 format and save as new temporary dictionary
        tmpDict = {}
        for k,v in value.items():
            tmpDict.update({k:v.value})
        #Add the sub-dictionary to the meta-dictionary
        dataDict.update({saveKey:tmpDict})

    return dataDict

#------------------------------------------------------------------------------#

def FullDict_hdf5_load(path,TRACERSPARAMS,FullDataPathSuffix):

    FullDict = {}
    for snap in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['snapMax']+1), int(TRACERSPARAMS['finalSnap']+1)),1):
        for targetT in TRACERSPARAMS['targetTLst']:
            loadPath = path + f"_T{int(targetT)}_{int(snap)}"+ FullDataPathSuffix
            data = hdf5_load(loadPath)
            FullDict.update(data)

    return FullDict

#------------------------------------------------------------------------------#

def Statistics_hdf5_load(targetT,path,TRACERSPARAMS,MiniDataPathSuffix):

    #Load data in {(T#, snap#):{k:v}} form
    nested = {}
    for snap in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['snapMax']+1), int(TRACERSPARAMS['finalSnap']+1)),1):
            #Temperature specific load path
            loadPath = path + f"_T{int(targetT)}_{int(snap)}_Statistics" + MiniDataPathSuffix
            data = hdf5_load(loadPath)
            nested.update(data)

    #flatten data to {k:[v1,v2,v3...]} form
    plotData = {}
    for key, value in nested.items():
        for k, v in value.items():
            if k not in plotData.keys():
                plotData.update({k : v})
            else:
                plotData[k] = np.append(plotData[k], v)
    return plotData

#------------------------------------------------------------------------------#

def PadNonEntries(snapGas):
    """
        Subroutine to pad all stars and gas entries in snapGas to have same first dimension size.
        So stars only data -> stars data + NGas x None
        So Gas only data -> Gas data + Nstars x None
        So all data first dimension == NTot

        Sanity checks and error messages in place.
    """

    print("Padding None Entries!")

    NGas =   len(snapGas.type[np.where(snapGas.type==0)])
    NStars = len(snapGas.type[np.where(snapGas.type==4)])
    NTot =   len(snapGas.type)


    GasNone_nx1 = np.full((NGas),np.nan).tolist() #[np.nan for ii in range(0,NGas)]
    StarsNone_nx1 = np.full((NStars),np.nan).tolist() #[np.nan for ii in range(0,NStars)]

    GasNone_nx3 =  np.full((NGas,3),np.nan).tolist() #[entryx3 for ii in range(0,NGas)]
    StarsNone_nx3 = np.full((NStars,3),np.nan).tolist() #[entryx3 for ii in range(0,NStars)]

    for key,value in snapGas.data.items():
        if (value is not None):
            #If shape indicates 1D give 1D lists nx1
            #Else list will be 2D so give lists nx3
            if (np.shape(np.shape(value))[0] == 1):
                GasNone = GasNone_nx1
                StarsNone = StarsNone_nx1
            else:
                GasNone = GasNone_nx3
                StarsNone = StarsNone_nx3

            if (np.shape(value)[0] == NGas):
                listValues = value.tolist()
                paddedList = listValues + StarsNone
                if (len(paddedList) != NTot):
                    print("[@ GAS @PadNonEntries:] Padded List not of length NTot. Data Does not have non-entries for STARS!")
                paddedValues = np.array(paddedList)
                snapGas.data[key] = paddedValues

                del listValues,paddedList,paddedValues

            elif(np.shape(value)[0] == NStars):
                listValues = value.tolist()
                #Opposite addition order to maintain sensible ordering.
                paddedList = GasNone + listValues
                if (len(paddedList) != NTot):
                    print("[@ STARS @PadNonEntries:] Padded List not of length NTot. Data Does not have non-entries for GAS!")
                paddedValues = np.array(paddedList)
                snapGas.data[key] = paddedValues

                del listValues,paddedList,paddedValues

            elif(np.shape(value)[0] != (NStars+NGas)):
                print("[@ ELSE @PadNonEntries:] Warning! Rule Exception! Original Data does not have shape consistent with number of stars or number of gas as defined by NGas NStars!")
                print(f"Key: {key}")

    return snapGas

#------------------------------------------------------------------------------#

def save_statistics(Cells, targetT, snapNumber, TRACERSPARAMS, saveParams, DataSavepath,MiniDataPathSuffix=".csv"):

    #------------------------------------------------------------------------------#
    #       Flatten dict and take subset
    #------------------------------------------------------------------------------#
    print("")
    print(f"[@{snapNumber} @T{int(targetT)}]: Analysing Statistics!")

    statsData = {}

    for k, v in Cells.items():
        if (k in saveParams):
            whereErrorKey = f"{k}"
            weightedperc(data=v, weights=Cells['mass'],perc=50.,key=whereErrorKey)
            #For the data keys we wanted saving (saveParams), this is where we generate the data to match the
            #   combined keys in saveKeys.
            #       We are saving the key (k) + median, UP, or LO in a new dict, statsData
            #           This effectively flattens and processes the data dict in one go
            #
            #   We have separate statements key not in keys and else.
            #       This is because if key does not exist yet in statsData, we want to create a new entry in statsData
            #           else we want to append to it, not create a new entry or overwrite the old one
            # whereGas = np.where(FullDict[key]['type'] == 0)
            if ((f"{k}median" not in statsData.keys()) or (f"{k}UP" not in statsData.keys()) or (f"{k}LO" not in statsData.keys())):
                statsData.update({f"{k}median": \
                weightedperc(data=v, weights=Cells['mass'],perc=50.,key=whereErrorKey)})
                statsData.update({f"{k}UP": \
                weightedperc(data=v, weights=Cells['mass'],perc=TRACERSPARAMS['percentileUP'],key=whereErrorKey)})
                statsData.update({f"{k}LO": \
                weightedperc(data=v, weights=Cells['mass'],perc=TRACERSPARAMS['percentileLO'],key=whereErrorKey)})
            else:
                statsData[f"{k}median"] = np.append(statsData[f"{k}median"],\
                weightedperc(data=v, weights=Cells['mass'],perc=50.,key=whereErrorKey))
                statsData[f"{k}UP"] = np.append(statsData[f"{k}UP"],\
                weightedperc(data=v, weights=Cells['mass'],perc=TRACERSPARAMS['percentileUP'],key=whereErrorKey))
                statsData[f"{k}LO"] = np.append(statsData[f"{k}LO"],\
                weightedperc(data=v, weights=Cells['mass'],perc=TRACERSPARAMS['percentileLO'],key=whereErrorKey))
    #------------------------------------------------------------------------------#
    #       Save stats as .csv files for a given temperature
    #------------------------------------------------------------------------------#


    #Generate our savepath
    savePath = DataSavepath + f"_T{int(targetT)}_{int(snapNumber)}_Statistics" + MiniDataPathSuffix
    print("\n" + f"[@{snapNumber} @T{int(targetT)}]: Saving Statistics as: " + savePath)

    out = {(f"T{int(targetT)}",f"{int(snapNumber)}"): statsData}

    hdf5_save(savePath,out)

    return

#------------------------------------------------------------------------------#

def flatten_wrt_T(dataDict,TRACERSPARAMS):
    flattened_dict = {}
    for snap in TRACERSPARAMS['phasesSnaps']:
        tmp = {}
        newkey = f"{int(snap)}"
        for T in TRACERSPARAMS['targetTLst']:
            key = (f"T{int(T)}",f"{int(snap)}")

            for k, v in dataDict[key].items():
                if (k in tmp.keys()):
                    tmp[k] = np.append(tmp[k] , v)
                else:
                    tmp.update({k : v})

        flattened_dict.update({newkey: tmp})

    return flattened_dict

#------------------------------------------------------------------------------#

def PlotProjections(snapGas,Cells,snapNumber,targetT,TRACERSPARAMS, DataSavepath,\
FullDataPathSuffix, Axes=[0,1],zAxis=[2],\
boxsize = 400., boxlos = 20.,pixres = 0.2,pixreslos = 4, DPI = 200,\
CMAP=None,QuadPlotBool=False,TracerPlotBool=True, numThreads=2):

    print(f"[@T{int(targetT)} @{int(snapNumber)}]: Starting Projections Video Plots!")

    if(CMAP == None):
        cmap = plt.get_cmap("inferno")
    else:
        cmap=CMAP

    #Axes Labels to allow for adaptive axis selection
    AxesLabels = ['x','y','z']

    #Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in SetCentre)
    imgcent =[0.,0.,0.]

    subset = int(TRACERSPARAMS['subset'])

    #--------------------------#
    ## Slices and Projections ##
    #--------------------------#
    print(f"[@T{int(targetT)} @{int(snapNumber)}]: Slices and Projections!")

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # slice_nH    = snap.get_Aslice("n_H", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    #
    # slice_B   = snap.get_Aslice("B", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    if(QuadPlotBool is True):
        nprojections = 5
    else:
        nprojections = 2

    print("\n"+f"[@T{int(targetT)} @{int(snapNumber)}]: Projection 1 of {nprojections}")

    proj_T = snapGas.get_Aslice("Tdens", box = [boxsize,boxsize],\
     center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
     nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=numThreads)

    print("\n"+f"[@T{int(targetT)} @{int(snapNumber)}]: Projection 2 of {nprojections}")

    proj_dens = snapGas.get_Aslice("rho_rhomean", box = [boxsize,boxsize],\
     center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
     nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=numThreads)

    if(QuadPlotBool is True):

        print("\n"+f"[@T{int(targetT)} @{int(snapNumber)}]: Projection 3 of {nprojections}")

        proj_nH = snapGas.get_Aslice("n_H", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=numThreads)

        print("\n"+f"[@T{int(targetT)} @{int(snapNumber)}]: Projection 4 of {nprojections}")

        proj_B = snapGas.get_Aslice("B", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=numThreads)

        print("\n"+f"[@T{int(targetT)} @{int(snapNumber)}]: Projection 5 of {nprojections}")

        proj_gz = snapGas.get_Aslice("gz", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=numThreads)


#==============================================================================#
#
#           Grab positions of Tracer Subset
#
#==============================================================================#
    if(TracerPlotBool is True):
        print("\n" + f"[@{int(snapNumber)} @T{int(targetT)}]: Selecting {int(subset)} subset of Tracers Positions...")
        #Select new subset for first snap
        #   Use old subset for all others
        if (int(snapNumber) == int(TRACERSPARAMS['snapMin'])):
            key = (f"T{int(targetT)}",f"{int(snapNumber)}")

            inRangeIDsIndices = np.where((Cells[key]['pos'][:,zAxis[0]]<=(float(boxlos)/2.))&(Cells[key]['pos'][:,zAxis[0]]>=(-1.*float(boxlos)/2.)))
            inRangeIDs = Cells[key]['id'][inRangeIDsIndices]
            inRangePridIndices = np.where(np.isin(Cells[key]['prid'],inRangeIDs))
            inRangeTrids = Cells[key]['trid'][inRangePridIndices]

            SelectedTracers1 = random.sample(inRangeTrids.tolist(),subset)
            SelectedTracers1 = np.array(SelectedTracers1)
        else:
            LoadPathTracers = DataSavepath + f"_T{int(targetT)}_{int(snapNumber-1)}_Projection_Tracers_{int(subset)}_Subset"+ FullDataPathSuffix
            oldData = hdf5_load(LoadPathTracers)
            key = (f"T{int(targetT)}",f"{int(snapNumber-1)}")
            SelectedTracers1 = oldData[key]['trid']

        key = (f"T{int(targetT)}",f"{int(snapNumber)}")

        whereGas = np.where(Cells[key]['type']==0)[0]

        nullEntry = [np.nan,np.nan,np.nan]

        posData, _, _ = GetIndividualCellFromTracer(Tracers=Cells[key]['trid'],\
            Parents=Cells[key]['prid'],CellIDs=Cells[key]['id'][whereGas],SelectedTracers=SelectedTracers1,\
            Data=Cells[key]['pos'][whereGas],mass=Cells[key]['mass'][whereGas],NullEntry=nullEntry)

        posData = np.array(posData)

        TracersSaveData = {(f"T{int(targetT)}",f"{int(snapNumber)}"): {'trid': SelectedTracers1, 'pos' : posData}}

        savePathTracers = DataSavepath + f"_T{int(targetT)}_{int(snapNumber)}_Projection_Tracers_{int(subset)}_Subset"+ FullDataPathSuffix

        print("\n" + f"[@{int(snapNumber)} @T{int(targetT)}]: Saving {int(subset)} Subset Tracers ID ('trid') data as: "+ savePathTracers)

        hdf5_save(savePathTracers,TracersSaveData)


#------------------------------------------------------------------------------#
    #PLOTTING TIME
    #Set plot figure sizes
    xsize = 10.
    ysize = 10.

    #Define halfsize for histogram ranges which are +/-
    halfbox = boxsize/2.

    #Redshift
    redshift = snapGas.redshift        #z
    aConst = 1. / (1. + redshift)   #[/]

    #[0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[0] #[Gyrs]

#==============================================================================#
#
#           Quad Plot for standard video
#
#==============================================================================#

    if (QuadPlotBool is True):
        print(f"[@T{int(targetT)} @{int(snapNumber)}]: Quad Plot...")

        aspect = "equal"
        fontsize = 12
        fontsizeTitle = 20

        #DPI Controlled by user as lower res needed for videos #
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize = (xsize,ysize), dpi = DPI, sharex="col", sharey="row")

        #Add overall figure plot
        TITLE = r"Redshift $(z) =$" + f"{redshift:0.03f} " + " " + r"$t_{Lookback}=$" + f"{lookback:0.03f} Gyrs" +\
        "\n" + f"Projections within {-1.*float(boxlos)/2.}"+r"<"+f"{AxesLabels[zAxis[0]]}-axis"+r"<"+f"{float(boxlos)/2.} kpc"
        fig.suptitle(TITLE, fontsize=fontsizeTitle)

        # cmap = plt.get_cmap(CMAP)
        cmap.set_bad(color="grey")

        #-----------#
        # Plot Temperature #
        #-----------#
        # print("pcm1")
        ax1 = axes[0,0]

        pcm1 = ax1.pcolormesh(proj_T['x'], proj_T['y'], np.transpose(proj_T['grid']/proj_dens['grid']),\
        vmin=1e4,vmax=1e7,\
        norm =  matplotlib.colors.LogNorm(), cmap = cmap, rasterized = True)

        ax1.set_title(f'Temperature Projection',fontsize=fontsize)
        cax1 = inset_axes(ax1,width="5%",height="95%",loc='right')
        fig.colorbar(pcm1, cax = cax1, orientation = 'vertical').set_label(label=r'$T$ [$K$]',size=fontsize, weight="bold")
        cax1.yaxis.set_ticks_position("left")
        cax1.yaxis.set_label_position("left")
        cax1.yaxis.label.set_color("white")
        cax1.tick_params(axis="y", colors="white",labelsize=fontsize)

        ax1.set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize = fontsize)
        # ax1.set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize = fontsize)
        ax1.set_aspect(aspect)

        #Fudge the tick labels...
        plt.sca(ax1)
        plt.yticks([-150, -100, -50, 0, 50, 100, 150, 200])
        #-----------#
        # Plot n_H Projection #
        #-----------#
        # print("pcm2")
        ax2 = axes[0,1]

        pcm2 = ax2.pcolormesh(proj_nH['x'], proj_nH['y'], np.transpose(proj_nH['grid'])/int(boxlos/pixreslos),\
        vmin=1e-6,vmax=1e-1,\
        norm =  matplotlib.colors.LogNorm(), cmap = cmap, rasterized = True)

        ax2.set_title(r'Hydrogen Number Density Projection', fontsize=fontsize)

        cax2 = inset_axes(ax2,width="5%",height="95%",loc='right')
        fig.colorbar(pcm2, cax=cax2, orientation = 'vertical').set_label(label=r'$n_H$ [$cm^{-3}$]',size=fontsize, weight="bold")
        cax2.yaxis.set_ticks_position("left")
        cax2.yaxis.set_label_position("left")
        cax2.yaxis.label.set_color("white")
        cax2.tick_params(axis="y", colors="white", labelsize=fontsize)
        # ax2.set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize=fontsize)
        # ax2.set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize=fontsize)
        ax2.set_aspect(aspect)

        #-----------#
        # Plot Metallicity #
        #-----------#
        # print("pcm3")
        ax3 = axes[1,0]

        pcm3 = ax3.pcolormesh(proj_gz['x'], proj_gz['y'], np.transpose(proj_gz['grid'])/int(boxlos/pixreslos),\
        vmin=1e-2,vmax=1e1,\
        norm =  matplotlib.colors.LogNorm(), cmap = cmap, rasterized = True)

        ax3.set_title(f'Metallicity Projection', y=-0.2, fontsize=fontsize)

        cax3 = inset_axes(ax3,width="5%",height="95%",loc='right')
        fig.colorbar(pcm3, cax = cax3, orientation = 'vertical').set_label(label=r'$Z/Z_{\odot}$',size=fontsize, weight="bold")
        cax3.yaxis.set_ticks_position("left")
        cax3.yaxis.set_label_position("left")
        cax3.yaxis.label.set_color("white")
        cax3.tick_params(axis="y", colors="white",labelsize=fontsize)

        ax3.set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize=fontsize)
        ax3.set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize=fontsize)

        ax3.set_aspect(aspect)

        #-----------#
        # Plot Magnetic Field Projection #
        #-----------#
        # print("pcm4")
        ax4 = axes[1,1]

        pcm4 = ax4.pcolormesh(proj_B['x'], proj_B['y'], np.transpose(proj_B['grid'])/int(boxlos/pixreslos),\
        vmin=1e-3,vmax=1e1,\
        norm =  matplotlib.colors.LogNorm(), cmap = cmap, rasterized = True)

        ax4.set_title(r'Magnetic Field Strength Projection', y=-0.2, fontsize=fontsize)

        cax4 = inset_axes(ax4,width="5%",height="95%",loc='right')
        fig.colorbar(pcm4, cax = cax4, orientation = 'vertical').set_label(label=r'$B$ [$\mu G$]',size=fontsize, weight="bold")
        cax4.yaxis.set_ticks_position("left")
        cax4.yaxis.set_label_position("left")
        cax4.yaxis.label.set_color("white")
        cax4.tick_params(axis="y", colors="white",labelsize=fontsize)

        # ax4.set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize=fontsize)
        ax4.set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize=fontsize)
        ax4.set_aspect(aspect)

        #Fudge the tick labels...
        plt.sca(ax4)
        plt.xticks([-150, -100, -50, 0, 50, 100, 150, 200])

        # print("snapnum")
        #Pad snapnum with zeroes to enable easier video making
        fig.subplots_adjust(wspace=0.0,hspace=0.0)
        # fig.tight_layout()

        SaveSnapNumber = str(snapNumber).zfill(4);
        savePath = DataSavepath + f"_T{int(targetT)}_Quad_Plot_{int(SaveSnapNumber)}.png"

        print(f"[@T{int(targetT)} @{int(snapNumber)}]: Save {savePath}")
        plt.savefig(savePath, dpi = DPI, transparent = False)
        plt.close()

        print(f"[@T{int(targetT)} @{int(snapNumber)}]: ...done!")


#==============================================================================#
#
#           Tracers overlayed on temperature for Tracer Subset Video
#
#==============================================================================#


    if(TracerPlotBool is True):
        print(f"[@T{int(targetT)} @{int(snapNumber)}]: Tracer Plot...")

        aspect = "equal"
        fontsize = 12
        fontsizeTitle = 16

        print(f"[@T{int(targetT)} @{int(snapNumber)}]: Loading Old Tracer Subset Data...")

        nOldSnaps = int(snapNumber) - int(TRACERSPARAMS['snapMin'])

        OldPosDict = {}
        for snap in range(int(TRACERSPARAMS['snapMin']),int(snapNumber)):
            print(f"[@T{int(targetT)} @{int(snapNumber)}]: Loading Old Tracer T{int(targetT)} {int(snap)}")

            LoadPathTracers = DataSavepath + f"_T{int(targetT)}_{int(snap)}_Projection_Tracers_{int(subset)}_Subset"+ FullDataPathSuffix
            data = hdf5_load(LoadPathTracers)
            OldPosDict.update(data)

        NullEntry= [np.nan,np.nan,np.nan]
        tmpOldPosDict= {}
        for key, dict in OldPosDict.items():
            tmp = {}
            for k, v in dict.items():
                if (k=="pos"):
                    vOutOfRange = np.where((v[:,zAxis[0]]>(float(boxlos)/2.))&(v[:,zAxis[0]]<(-1.*float(boxlos)/2.)))
                    v[vOutOfRange]  =NullEntry

                tmp.update({k : v})

            tmpOldPosDict.update({key : tmp})

        OldPosDict = tmpOldPosDict
        print(f"[@T{int(targetT)} @{int(snapNumber)}]: ...finished Loading Old Tracer Subset Data!")

        print(f"[@T{int(targetT)} @{int(snapNumber)}]: Plot...")
        #DPI Controlled by user as lower res needed for videos #
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize = (xsize,ysize), dpi = DPI)

        #Add overall figure plot
        TITLE = r"Redshift $(z) =$" + f"{redshift:0.03f} " + " " + r"$t_{Lookback}=$" + f"{lookback:0.03f} Gyrs" +\
        "\n" + f"Projections within {-1.*float(boxlos)/2.}"+r"<"+f"{AxesLabels[zAxis[0]]}-axis"+r"<"+f"{float(boxlos)/2.} kpc" +\
        "\n" + f"Subset of {int(subset)} Tracers selected at {int(TRACERSPARAMS['selectSnap'])} as being at " +\
        r"$T = 10^{%5.2f \pm %5.2f} K$"%(targetT,TRACERSPARAMS['deltaT'])

        fig.suptitle(TITLE, fontsize=fontsizeTitle)

        # cmap = plt.get_cmap(CMAP)
        cmap.set_bad(color="grey")

        #-----------#
        # Plot Temperature #
        #-----------#

        ###
        #   Select 10% of subset to have a colour, the rest to be white
        ###
        # cmapTracers = matplotlib.cm.get_cmap("nipy_spectral")
        colourTracers = []
        cwhite = (1.,1.,1.,1.)
        cblack = (0.,0.,0.,1.)
        for ii in range(0,int(subset+1)):
            if (ii % 5 == 0):
                colour = cblack
            else:
                colour = cwhite
            colourTracers.append(colour)

        colourTracers = np.array(colourTracers)

        ax1 = axes

        pcm1 = ax1.pcolormesh(proj_T['x'], proj_T['y'], np.transpose(proj_T['grid']/proj_dens['grid']),\
        vmin=1e4,vmax=1e7,\
        norm =  matplotlib.colors.LogNorm(), cmap = cmap, rasterized = True)

        sizeMultiply= 25
        sizeConst = 10

        whereInRange = np.where((posData[:,zAxis[0]]<=(float(boxlos)/2.))&(posData[:,zAxis[0]]>=(-1.*float(boxlos)/2.)))
        posDataInRange = posData[whereInRange]
        colourInRange = colourTracers[whereInRange]

        colourTracers = colourTracers.tolist()
        colourInRange = colourInRange.tolist()

        sizeData = np.array([(sizeMultiply*(xx+(float(boxlos)/2.))/float(boxlos))+sizeConst  for xx in posDataInRange[:,zAxis[0]]])

        ax1.scatter(posDataInRange[:,Axes[0]],posDataInRange[:,Axes[1]],s=sizeData,c='white',marker='o')#colourInRange,marker='o')

        minSnap = int(snapNumber) - min(int(nOldSnaps),3)




        print(f"[@T{int(targetT)} @{int(snapNumber)}]: Plot Tails...")
        jj=1
        for snap in range(int(minSnap+1),snapNumber+1):
            key1 = (f"T{int(targetT)}",f"{int(snap-1)}")
            key2 = (f"T{int(targetT)}",f"{int(snap)}")
            if (snap != int(snapNumber)):
                pos1 = OldPosDict[key1]['pos']
                pos2 = OldPosDict[key2]['pos']
            else:
                pos1 = OldPosDict[key1]['pos']
                pos2 = posData

            pathData = np.array([pos1,pos2])
            alph = float(jj)/float(max(1,min(int(nOldSnaps),3))+1.)
            jj +=1

            for ii in range(0,int(subset)):
                ax1.plot(pathData[:,ii,Axes[0]],pathData[:,ii,Axes[1]],c='white',alpha=alph)#colourTracers[ii],alpha=alph)


        print(f"[@T{int(targetT)} @{int(snapNumber)}]: ...finished Plot Tails!")

        xmin = np.nanmin(proj_T['x'])
        xmax = np.nanmax(proj_T['x'])
        ymin = np.nanmin(proj_T['y'])
        ymax = np.nanmax(proj_T['y'])

        ax1.set_ylim(ymin=ymin,ymax=ymax)
        ax1.set_xlim(xmin=xmin,xmax=xmax)

        ax1.set_title(f'Temperature Projection',fontsize=fontsize)
        cax1 = inset_axes(ax1,width="5%",height="95%",loc='right')
        fig.colorbar(pcm1, cax = cax1, orientation = 'vertical').set_label(label=r'$T$ [$K$]',size=fontsize, weight="bold")
        cax1.yaxis.set_ticks_position("left")
        cax1.yaxis.set_label_position("left")
        cax1.yaxis.label.set_color("white")
        cax1.tick_params(axis="y", colors="white",labelsize=fontsize)

        ax1.set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize = fontsize)
        ax1.set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize = fontsize)
        ax1.set_aspect(aspect)



        #Pad snapnum with zeroes to enable easier video making
        fig.subplots_adjust(wspace=0.0,hspace=0.0)
        # fig.tight_layout()

        SaveSnapNumber = str(snapNumber).zfill(4);
        savePath = DataSavepath + f"_T{int(targetT)}_Tracer_Subset_Plot_{int(SaveSnapNumber)}.png"

        print(f"[@T{int(targetT)} @{int(snapNumber)}]: Save {savePath}")
        plt.savefig(savePath, dpi = DPI, transparent = False)
        plt.close()

        print(f"[@T{int(targetT)} @{int(snapNumber)}]: ...Tracer Plot done!")

    return
