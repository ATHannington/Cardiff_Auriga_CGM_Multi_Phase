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
import h5py


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
        if (value is not None):
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
        if (value is not None):
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
    rhocrit = 3. * (snapGas.omega0 * (1+snapGas.redshift)**3. + snapGas.omegalambda) * (snapGas.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)
    rhomean = 3. * (snapGas.omega0 * (1+snapGas.redshift)**3.) * (snapGas.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)

    meanweight = sum(snapGas.gmet[whereGas,0:9][0], axis = 1) / ( sum(snapGas.gmet[whereGas,0:9][0]/elements_mass[0:9], axis = 1) + snapGas.ne*snapGas.gmet[whereGas,0][0] )
    Tfac = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53

    gasdens = (snapGas.rho / (c.parsec*1e6)**3.) * c.msol * 1e10 #[g cm^-3]
    gasX = snapGas.gmet[whereGas,0][0]

    snapGas.data['T'] = snapGas.u / Tfac # K
    snapGas.data['n_H'] = gasdens / c.amu * gasX # cm^-3
    snapGas.data['dens'] = gasdens / (rhomean * omegabaryon0/snapGas.omega0) # rho / <rho>
    snapGas.data['Tdens'] = snapGas.data['T'] *snapGas.data['dens']

    bfactor = 1e6*(np.sqrt(1e10 * c.msol) / np.sqrt(c.parsec * 1e6)) * (1e5 / (c.parsec * 1e6)) #[microGauss]

    #Magnitude of Magnetic Field [micro Guass]
    snapGas.data['B'] = np.linalg.norm((snapGas.data['bfld'] * bfactor), axis=1)

    #Radius [kpc]
    snapGas.data['R'] =  (np.linalg.norm(snapGas.data['pos'], axis=1)) #[Kpc]

    #Radial Velocity [km s^-1]
    snapGas.data['vrad'] = (snapGas.pos*snapGas.vel).sum(axis=1)
    snapGas.data['vrad'] /= snapGas.data['R']

    #Cooling time [s]
    snapGas.data['tcool'] = snapGas.data['u'] * 1e10 * gasdens / (snapGas.data['gcol'] * snapGas.data['n_H']**2.) #[s]

    #Load in metallicity
    tmp = snapGas.data['gz']
    #Load in Metals
    tmp = snapGas.data['gmet']
    #Load in Star Formation Rate
    tmp = snapGas.data['sfr']

    #Specific Angular Momentum [kpc km s^-1]
    snapGas.data['L'] = sqrt((cross(snapGas.data['pos'], snapGas.data['vel'])**2.).sum(axis=1))

    snapGas.data['ndens'] = snapGas.data['dens']/(meanweight*c.amu)

    #Thermal Pressure : P = n Kb T
    snapGas.data['P_thermal'] = snapGas.ndens * c.KB *snapGas.T

    #Magnetic Pressure [microGauss ^ 2]
    snapGas.data['P_magnetic'] = (snapGas.data['B'] **2)/( 8. * pi)

    #Kinetic "Pressure" [10^10 Msol km^2 ^s^-2]
    snapGas.data['P_kinetic'] = snapGas.rho * (np.linalg.norm(snapGas.data['vel'][whereGas], axis=1))**2

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
            saveData.append(Data[dataIndex].tolist())
            massData.append(mass[dataIndex].tolist())
        else:
            saveData.append([np.nan])
            massData.append([np.nan])

    return saveData, massData, TracersReturned

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
                grp.create_dataset(k, data=v, compression='gzip')

    return

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
            tmpDict.update({k:v[:]})
        #Add the sub-dictionary to the meta-dictionary
        dataDict.update({saveKey:tmpDict})

    return dataDict

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

    GasNone = [None for ii in range(0,NGas)]
    StarsNone = [None for ii in range(0,NStars)]

    for key,value in snapGas.data.items():
        if (value is not None):
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
                paddedList = listValues + GasNone
                if (len(paddedList) != NTot):
                    print("[@ STARS @PadNonEntries:] Padded List not of length NTot. Data Does not have non-entries for GAS!")
                paddedValues = np.array(paddedList)
                snapGas.data[key] = paddedValues

                del listValues,paddedList,paddedValues

            elif(np.shape(value)[0] != (NStars+NGas)):
                print("[@ ELSE @PadNonEntries:] Warning! Rule Exception! Original Data does not have shape consistent with number of stars or number of gas as defined by NGas NStars!")
                print(f"Key: {key}")

    return snapGas
