"""
Author: A. T. Hannington
Created: 19/03/2020
Known Bugs:
    pandas read_csv is loading nested dicts. Have implemented temporary fix.
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
from Tracers_Subroutines import *

#==============================================================================#
#       USER DEFINED PARAMETERS
#==============================================================================#
#Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = ['T','R','n_H','B']

#Essential Save Parameters, that will NOT be combined with savetypes
saveEssentials =['Lookback','Ntracers']

#Save types, which when combined with saveparams define what data is saved.
#   This is intended to be 'median', 'UP' (upper quartile), and 'LO' (lower quartile)
saveTypes= ['median','UP','LO']

#==============================================================================#
#       Prepare for analysis
#==============================================================================#

#Combine saveParams and saveTypes to form each combination for saving data types
saveKeys =[]
for param in saveParams:
    for type in saveTypes:
        saveKeys.append(param+type)

#Add saveEssentials to saveKeys so as to save these without the median and quartiles
#   being taken.
for key in saveEssentials:
    saveKeys.append(key)

# Load in parameters from csv. This ensures reproducability!
#   We save as a DataFrame, then convert to a dictionary, and take out nesting...
TRACERSPARAMS = pd.read_csv('TracersParams.csv', delimiter=" ", header=None, \
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

#Save as .csv
DataSavepathSuffix = f".csv"

#==============================================================================#
#       Chemical Properties
#==============================================================================#
#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
elements_mass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

Zsolar = 0.0127

omegabaryon0 = 0.048
#==============================================================================#
#       MAIN PROGRAM
#==============================================================================#

kk = 0
#Loop over target temperatures
for targetT in TRACERSPARAMS['targetTLst']:

    #Store number of target temperatures
    NTemps = float(len(TRACERSPARAMS['targetTLst']))
    #Calculate percentage complete as a function of Temperatures
    #   Aside: I did try to implement a total percentage complete, but combinatorix ='(
    percentage = (float(kk)/NTemps)*100.0

    #Increment percentage complete counter
    kk+=1

    print("")
    print(f"{percentage:0.02f}%")
    print("Setting Condition!")

    # load in the subfind group files
    snap_subfind = load_subfind(TRACERSPARAMS['snapnum'],dir=TRACERSPARAMS['simfile'])

    # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
    snapGas     = gadget_readsnap(TRACERSPARAMS['snapnum'], TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
    snapTracers = gadget_readsnap(TRACERSPARAMS['snapnum'], TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [6], lazy_load=True)

    print(f" SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

    #Centre the simulation on HaloID 0
    Snapper1 = Snapper()
    snapGas  = Snapper1.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=0)

    #Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos   *= 1e3 #[kpc]

    snapGas.vol *= 1e9 #[kpc^3]

    #--------------------------#
    ##    Units Conversion    ##
    #--------------------------#

    print("Converting Units!")

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


    ### Exclude values outside halo 0 ###

    print("Finding Halo 0 Only Data!")

    #Find length of the first n entries of particle type 0 that are associated with HaloID 0: ['HaloID', 'particle type']
    gaslength = snap_subfind.data['slty'][0,0]

    #Take only data from above HaloID
    for key, value in snapGas.data.items():
        if (snapGas.data[key] is not None):
            snapGas.data[key] = snapGas.data[key][:gaslength]

    #Take onlt tracers for above HaloID
    for key, value in snapTracers.data.items():
        if (snapTracers.data[key] is not None):
            snapTracers.data[key] = snapTracers.data[key][:gaslength]

    #--------------------------------------------------------------------------#
    ####                    SELECTION                                        ###
    #--------------------------------------------------------------------------#

    #Set condition for Tracer selection
    Cond = np.where((snapGas.data['T']>=1.*10**(targetT-TRACERSPARAMS['deltaT'])) & \
                    (snapGas.data['T']<=1.*10**(targetT+TRACERSPARAMS['deltaT'])) & \
                    (snapGas.data['R']>=TRACERSPARAMS['Rinner']) & \
                    (snapGas.data['R']<=TRACERSPARAMS['Router']) &
                    (snapGas.data['sfr']<=0)\
                   )

    #Get Cell data and Cell IDs from tracers based on condition
    Tracers, CellsTFC, CellIDsTFC = GetTracersFromCells(snapGas, snapTracers,Cond)

    #Set Blank Data dictionary for this temperature
    dataDict = {}
    #Set Blank ID dictionary for this temperature
    IDDict = {}

    #Loop over snaps from snapMin to snapmax, taking the snapnumMAX (the final snap) as the endpoint if snapMax is greater
    for ii in range(int(TRACERSPARAMS['snapMin']), int(min(TRACERSPARAMS['snapnumMAX']+1, TRACERSPARAMS['snapMax']+1))):
        print("")
        print(f"Starting Snap {ii}")

        # load in the subfind group files
        snap_subfind = load_subfind(ii,dir=TRACERSPARAMS['simfile'])

        # load in the gas particles mass and position only for HaloID 0.
        #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
        snapGas     = gadget_readsnap(ii, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
        # load tracers data
        snapTracers = gadget_readsnap(ii, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [6], lazy_load=True)

        print(f"SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

        #Set centre of simulation
        Snapper1 = Snapper()
        snapGas  = Snapper1.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=0)

        #Convert Units
        ## Make this a seperate function at some point??
        snapGas.pos   *= 1e3 #[kpc]

        snapGas.vol *= 1e9 #[kpc^3]

        #--------------------------#
        ##    Units Conversion    ##
        #--------------------------#

        print("Converting Units!")

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


        #Find length of the first n entries of particle type 0 that are associated with HaloID 0: ['HaloID', 'particle type']
        gaslength = snap_subfind.data['slty'][0,0]

        #Take only data from above HaloID
        for key, value in snapGas.data.items():
            if (snapGas.data[key] is not None):
                snapGas.data[key] = snapGas.data[key][:gaslength]

        #Take onlt tracers for above HaloID
        for key, value in snapTracers.data.items():
            if (snapTracers.data[key] is not None):
                snapTracers.data[key] = snapTracers.data[key][:gaslength]

        ###
        ##  Selection   ##
        ###

        #Select Cells which have the tracers from the selection snap in them
        CellsCFT, CellIDsCFT = GetCellsFromTracers(snapGas, snapTracers,Tracers)

        # Save number of tracers
        CellsCFT['Ntracers'] = [int(len(Tracers))]
        print(f"Number of tracers = {CellsCFT['Ntracers']}")

        print("Lookback")
        #Redshift
        redshift = snapGas.redshift        #z
        aConst = 1. / (1. + redshift)   #[/]

        #Get lookback time in Gyrs
        #[0] to remove from numpy array for purposes of plot title
        lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[0] #[Gyrs]
        CellsCFT['Lookback']=[lookback for jj in range(0,len(CellsCFT['T']))]

        #Add snap data to temperature specific dictionary
        print("Adding to Dict")
        dataDict.update({f"{ii}":CellsCFT})
        IDDict.update({f"ID{ii}":CellIDsCFT})

        # Delete unecessary
        del CellsCFT, CellIDsCFT, snapGas, snapTracers, Snapper1, snap_subfind

#==============================================================================#
#       Prepare data and save
#==============================================================================#

    #------------------------------------------------------------------------------#
    #       Flatten dict and take subset
    #------------------------------------------------------------------------------#

    plotData = {}

    # For every data key in the temperature specific dictionary, loop over index, key, and values
    for ind, (key, value) in enumerate(dataDict.items()):
        #For the nested dictionary for a given snap in the given temperature meta-dict, loop over key and values
        for k, v in value.items():
            if ((k != 'Lookback') & (k in saveParams)):
                #For the data keys we wanted saving (saveParams), this is where we generate the data to match the
                #   combined keys in saveKeys.
                #       We are saving the key (k) + median, UP, or LO in a new dict, plotData
                #           This effectively flattens and processes the data dict in one go
                #
                #   We have separate statements for ind ==0 and else.
                #       This is because if ind == 0 we want to create a new entry in plotData
                #           else we want to append to it, not create a new entry or overwrite the old one
                if ind == 0:
                    plotData.update({f"{k}median": \
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=50.)})
                    plotData.update({f"{k}UP": \
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=TRACERSPARAMS['percentileUP'])})
                    plotData.update({f"{k}LO": \
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=TRACERSPARAMS['percentileLO'])})
                else:
                    plotData[f"{k}median"] = np.append(plotData[f"{k}median"],\
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=50.))
                    plotData[f"{k}UP"] = np.append(plotData[f"{k}UP"],\
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=TRACERSPARAMS['percentileUP']))
                    plotData[f"{k}LO"] = np.append(plotData[f"{k}LO"],\
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=TRACERSPARAMS['percentileLO']))
            elif (k=='Lookback'):
                #Separate handling of lookback time so as to not take percentiles etc.
                if ind == 0:
                    plotData.update({f"{k}": np.median(v)})
                else:
                    plotData[f"{k}"] = np.append(plotData[f"{k}"],\
                     np.median(v))
            else:
                #
                #
                #   !!! NOT TESTED  !!!
                #
                #This takes the data not in saveParams and adds it to the dict anyway.
                if ind == 0 :
                    plotData.update({f"{k}": v})
                else:
                    plotData[f"{k}"] = np.append(plotData[f"{k}"], v)

    #Generate our savepath
    tmpSave = DataSavepath + f"_T{int(targetT)}" + DataSavepathSuffix
    print(tmpSave)

    #Take only the data we selected to save from plotData and save to a new temporary dict for saving
    tmpData = {}
    for key in saveKeys:
        tmpData.update({key : plotData[key]})

    #Create a new DataFrame from the temporary saving dictionary
    df = pd.DataFrame.from_dict(tmpData, orient="index")
    #Save data as csv!
    df.to_csv(tmpSave)

#End!
