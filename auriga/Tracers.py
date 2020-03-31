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
import pickle

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

#Select Halo of interest:
#   0 is the most massive:
HaloID = 0

#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

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
    #Save as .csv
DataSavepathSuffix = f".csv"
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

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


#Set Blank Data dictionary ALL data is stored here!
FullDict = {}



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
    snapGas     = gadget_readsnap(TRACERSPARAMS['snapnum'], TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0], loadonlyhalo = HaloID, lazy_load=True, subfind = snap_subfind)
    snapTracers = gadget_readsnap(TRACERSPARAMS['snapnum'], TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [6], lazy_load=True)

    print(f" SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

    #Centre the simulation on HaloID 0
    Snapper1 = Snapper()
    snapGas  = Snapper1.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=HaloID)

    #Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos   *= 1e3 #[kpc]

    snapGas.vol *= 1e9 #[kpc^3]

    #--------------------------#
    ##    Units Conversion    ##
    #--------------------------#

    print("Converting Units!")

    snapGas = ConvertUnits(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0)


    ### Exclude values outside halo 0 ###

    print("Finding Halo 0 Only Data!")

    snapGas = HaloOnlyGasSelect(snapGas,snap_subfind,Halo=HaloID)

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
    TracersTFC, CellsTFC, CellIDsTFC, ParentsTFC = GetTracersFromCells(snapGas, snapTracers,Cond)

    TracersOld = TracersTFC

    #Make a list of snaps to analyse. We want two sets of ranges, both starting at the selection snap, snapnum, end ending at snapMin, and snapMax/snapnumMAX (whichever is smaller)
    #   This should ensure that, as the selected tracers decrease, we are selecting all tracers at the selection snap, and a subset of those going outwards in time.
    snapRangeMin = [zz for zz in range(int(TRACERSPARAMS['snapnum']),(int(TRACERSPARAMS['snapMin'])-1), -1)]
    snapRangeMax = [zz for zz in range(int(TRACERSPARAMS['snapnum']),min(int(TRACERSPARAMS['snapnumMAX'])+1,int(TRACERSPARAMS['snapMax'])+1), 1)]
    snapRange = [snapRangeMin,snapRangeMax]

    #Loop over snaps from snapMin to snapmax, taking the snapnumMAX (the final snap) as the endpoint if snapMax is greater
    for snapSet in snapRange:
        for snap in snapSet:
            print("")
            print(f"Starting Snap {snap}")

            # load in the subfind group files
            snap_subfind = load_subfind(snap,dir=TRACERSPARAMS['simfile'])

            # load in the gas particles mass and position only for HaloID 0.
            #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
            snapGas     = gadget_readsnap(snap, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0], lazy_load=True, subfind = snap_subfind)
            # load tracers data
            snapTracers = gadget_readsnap(snap, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [6], lazy_load=True)

            print(f"SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

            #Set centre of simulation
            Snapper1 = Snapper()
            snapGas  = Snapper1.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=HaloID)

            #Convert Units
            ## Make this a seperate function at some point??
            snapGas.pos   *= 1e3 #[kpc]

            snapGas.vol *= 1e9 #[kpc^3]

            #--------------------------#
            ##    Units Conversion    ##
            #--------------------------#

            print("Converting Units!")

            snapGas = ConvertUnits(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0)

            #Find length of the first n entries of particle type 0 that are associated with HaloID 0: ['HaloID', 'particle type']

            if (snap == int(TRACERSPARAMS['snapnum'])):
                print("Finding Halo 0 Only Data!")

                snapGas = HaloOnlyGasSelect(snapGas,snap_subfind,Halo=HaloID)

            ###
            ##  Selection   ##
            ###

            #Select Cells which have the tracers from the selection snap in them
            #   Returns new tracers CFT as some of the tracers may no longer be associated
            #       With Halo=HaloID. Thus, our list of tracers should diminish over time.
            TracersCFT, CellsCFT, CellIDsCFT, ParentsCFT = GetCellsFromTracers(snapGas, snapTracers,TracersOld)

            #Update old tracers list to new, smaller subset associated with Halo=HaloID
            TracersOld = TracersCFT

            # Save number of tracers
            CellsCFT['Ntracers'] = [int(len(TracersCFT))]
            print(f"Number of tracers = {CellsCFT['Ntracers']}")

            print("Lookback")
            #Redshift
            redshift = snapGas.redshift        #z
            aConst = 1. / (1. + redshift)   #[/]

            #Get lookback time in Gyrs
            #[0] to remove from numpy array for purposes of plot title
            lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[0] #[Gyrs]

            CellsCFT['Lookback']= np.array([lookback for jj in range(0,len(CellsCFT['T']))])

            #Save snap number
            CellsCFT['Snap'] = np.array([int(snap) for jj in range(0,len(CellsCFT['T']))])

            #Save Tracer IDs
            CellsCFT['trid'] = TracersCFT

            #Save Parent Cell IDs
            CellsCFT['prid'] = ParentsCFT

            #Save Cell IDs
            CellsCFT['id'] = CellIDsCFT

            #Add snap data to temperature specific dictionary
            print("Adding to Dict")
            FullDict.update({(f"T{int(targetT)}",f"{int(snap)}"): CellsCFT})

            del snapGas, snapTracers, Snapper1, snap_subfind

#==============================================================================#
#       Prepare data and save
#==============================================================================#

    #------------------------------------------------------------------------------#
    #       Flatten dict and take subset
    #------------------------------------------------------------------------------#

    plotData = {}

    # For every data key in the temperature specific dictionary, loop over index, key, and values
    for key, value in FullDict.items():
        Tkey = key[0]
        #For the nested dictionary for a given snap in the given temperature meta-dict, loop over key and values
        if (Tkey == f"T{int(targetT)}"):
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
                    if ((f"{k}median" not in plotData.keys()) or (f"{k}UP" not in plotData.keys()) or (f"{k}LO" not in plotData.keys())):
                        plotData.update({f"{k}median": \
                        weightedperc(data=v, weights=FullDict[key]['mass'],perc=50.)})
                        plotData.update({f"{k}UP": \
                        weightedperc(data=v, weights=FullDict[key]['mass'],perc=TRACERSPARAMS['percentileUP'])})
                        plotData.update({f"{k}LO": \
                        weightedperc(data=v, weights=FullDict[key]['mass'],perc=TRACERSPARAMS['percentileLO'])})
                    else:
                        plotData[f"{k}median"] = np.append(plotData[f"{k}median"],\
                        weightedperc(data=v, weights=FullDict[key]['mass'],perc=50.))
                        plotData[f"{k}UP"] = np.append(plotData[f"{k}UP"],\
                        weightedperc(data=v, weights=FullDict[key]['mass'],perc=TRACERSPARAMS['percentileUP']))
                        plotData[f"{k}LO"] = np.append(plotData[f"{k}LO"],\
                        weightedperc(data=v, weights=FullDict[key]['mass'],perc=TRACERSPARAMS['percentileLO']))
                elif (k=='Lookback'):
                    #Separate handling of lookback time so as to not take percentiles etc.
                    if f"Lookback" not in plotData.keys():
                        plotData.update({f"{k}": np.median(v)})
                    else:
                        plotData[f"{k}"] = np.append(plotData[f"{k}"],\
                         np.median(v))
                else:
                    #
                    #   !!! NOT TESTED  !!!
                    #
                    #This takes the data not in saveParams and adds it to the dict anyway.
                    if f"{k}" not in plotData.keys():
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



save = DataSavepath + ".pickle"
print(save)
#Save CellsCFT as pickle compressed object
with open(save,"wb") as f:
    Cells = pickle.dump(FullDict,f)
