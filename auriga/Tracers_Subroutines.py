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
import h5py
import sys
import logging

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
    rhocrit = 3. * (snapGas.omega0 * (1+snapGas.redshift)**3. + snapGas.omegalambda) * (snapGas.hubbleparam * 100.*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)
    rhomean = 3. * (snapGas.omega0 * (1+snapGas.redshift)**3.) * (snapGas.hubbleparam * 100.*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)

    #Mean weight [amu]
    meanweight = sum(snapGas.gmet[whereGas,0:9][0], axis = 1) / ( sum(snapGas.gmet[whereGas,0:9][0]/elements_mass[0:9], axis = 1) + snapGas.ne*snapGas.gmet[whereGas,0][0] )

    #3./2. N KB
    Tfac = ((3./2.) * c.KB) / (meanweight * c.amu)

    gasdens= (snapGas.rho / (c.parsec*1e6)**3.) * c.msol * 1e10 #[g cm^-3]
    gasX = snapGas.gmet[whereGas,0][0]

    #Temperature = U / (3/2 * N KB) [K]
    snapGas.data['T'] = (snapGas.u*1e10) / (Tfac) # K
    snapGas.data['n_H'] = gasdens/ c.amu * gasX # cm^-3
    snapGas.data['dens']  = gasdens/ (rhomean * omegabaryon0/snapGas.omega0) # rho / <rho>
    snapGas.data['Tdens'] = snapGas.data['T'] * snapGas.data['dens']

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
    snapGas.data['tcool'] = (snapGas.data['u'] * 1e10 * snapGas.data['dens']) / (GyrToSeconds * snapGas.data['gcol'] * snapGas.data['n_H']**2.) #[Gyrs]
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

    ndens = gasdens/ (meanweight * c.amu)
    #Thermal Pressure : P/k_B = n T [$ # K cm^-3]
    snapGas.data['P_thermal'] = ndens*snapGas.T

    #Magnetic Pressure [P/k_B K cm^-3]
    snapGas.data['P_magnetic'] = ((snapGas.data['B']*1e-6) **2)/( 8. * pi * c.KB)

    #Kinetic "Pressure" [P/k_B K cm^-3]
    snapGas.data['P_kinetic'] = (snapGas.rho / (c.parsec*1e6)**3.) * 1e10 * c.msol *(1./c.KB) * (np.linalg.norm(snapGas.data['vel'][whereGas]*1e5, axis=1))**2

    #Sound Speed [(erg K^-1 K ??? g^-1)^1/2 = (g cm^2 s^-2 g^-1)^(1/2) = km s^-1]
    snapGas.data['csound'] = sqrt(((5./3.)*c.KB * snapGas.data['T'])/(meanweight*c.amu*1e5))

    # [cm kpc^-1 kpc cm^-1 s^1 = s / GyrToSeconds = Gyr]
    snapGas.data['tcross'] = (KpcTokm*1e3/GyrToSeconds) * (snapGas.data['vol'])**(1./3.) /snapGas.data['csound']

    #Free Fall time [Gyrs]
    snapGas.data['tff'] = sqrt(( 3. * pi )/(32.* c.G  * snapGas.data['dens']) ) * (1./GyrToSeconds)

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

    #Get Temperatures as strings in a list so as to form "4-5-6" for savepath.
    Tlst = [str(int(item)) for item in TRACERSPARAMS['targetTLst']]
    Tstr = '-'.join(Tlst)

    #This rather horrible savepath ensures the data can only be combined with the right input file, TracersParams.csv, to be plotted/manipulated
    DataSavepath = TRACERSPARAMS['savepath'] + f"Data_selectSnap{int(TRACERSPARAMS['selectSnap'])}_min{int(TRACERSPARAMS['snapMin'])}_max{int(TRACERSPARAMS['snapMax'])}" +\
        f"_{int(TRACERSPARAMS['Rinner'])}R{int(TRACERSPARAMS['Router'])}_targetT{Tstr}"

    return TRACERSPARAMS, DataSavepath, Tlst

#------------------------------------------------------------------------------#

def GetIndividualCellFromTracer(Tracers,Parents,CellIDs,SelectedTracers,Data,mass):

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
                dataEntry =np.nan
            else:
                dataEntry = dataEntry[0]

            massEntry = mass[dataIndex].tolist()
            if (np.shape(massEntry)[0] == 0):
                massEntry =np.nan
            else:
                massEntry = massEntry[0]

            saveData.append(dataEntry)
            massData.append(massEntry)
        else:
            saveData.append([np.nan])
            massData.append([np.nan])

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
        for T in TRACERSPARAMS['targetTLst']:
            key = (f"T{int(T)}",f"{int(snap)}")
            newkey = f"{int(snap)}"

            for k, v in dataDict[key].items():
                if (k in tmp.keys()):
                    tmp[k] = np.append(tmp[k] , v)
                else:
                    tmp.update({k : v})

        flattened_dict.update({f"{int(snap)}": tmp})

    return flattened_dict
