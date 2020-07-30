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
from Tracers_Subroutines import *
import h5py
import multiprocessing as mp
import sys
import logging

#==============================================================================#
#       USER DEFINED PARAMETERS
#==============================================================================#
#Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = ['dens','T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','tcool','theat','csound','tcross','tff']

print("")
print("Saved Parameters in this Analysis:")
print(saveParams)

#Optional Tracer only (no stats in .csv) parameters to be saved
#   Cannot guarantee that all Plotting and post-processing are independent of these
#       Will attempt to ensure any necessary parameters are stored in ESSENTIALS
saveTracersOnly = ['sfr','age']

print("")
print("Tracers ONLY (no stats) Saved Parameters in this Analysis:")
print(saveTracersOnly)

#SAVE ESSENTIALS : The data required to be tracked in order for the analysis to work
saveEssentials = ['Lookback','Ncells','Snap','id','type','mass']

print("")
print("ESSENTIAL Saved Parameters in this Analysis:")
print(saveEssentials)

saveTracersOnly = saveTracersOnly + saveEssentials

#Save types, which when combined with saveparams define what data is saved.
#   This is intended to be 'median', 'UP' (upper quartile), and 'LO' (lower quartile)
saveTypes= ['median','UP','LO']

#Select Halo of interest:
#   0 is the most massive:
HaloID = 0

#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

#File types for data save.
#   Mini: small median and percentiles data
#   Full: full FullDict data
MiniDataPathSuffix = f".h5"
FullDataPathSuffix = f".h5"

#Lazy Load switch. Set to False to save all data (warning, pickle file may explode)
lazyLoadBool = True

#Number of cores to run on:
n_processes = 4
#==============================================================================#
#       Prepare for analysis
#==============================================================================#

#Combine saveParams and saveTypes to form each combination for saving data types
saveKeys =[]
for param in saveParams:
    for TYPE in saveTypes:
        saveKeys.append(param+TYPE)

# #Add saveEssentials to saveKeys so as to save these without the median and quartiles
# #   being taken.
# for key in saveEssentials:
#     saveKeys.append(key)

# Load in parameters from csv. This ensures reproducability!
#   We save as a DataFrame, then convert to a dictionary, and take out nesting...
    #Save as .csv
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

print("")
print("Loaded Analysis Parameters:")
for key,value in TRACERSPARAMS.items():
    print(f"{key}: {value}")

print("")
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


if __name__=="__main__":
    kk = 0
    #Loop over target temperatures
    for targetT in TRACERSPARAMS['targetTLst']:

        #Store number of target temperatures
        NTemps = float(len(TRACERSPARAMS['targetTLst']))
        #Calculate percentage complete as a function of Temperatures
        #   Aside: I did try to implement a total percentage complete, but combinatorix ='(
        percentage = (float(kk)/NTemps)*100.0
        print("")
        print(f"{percentage:0.02f}%")
        #Increment percentage complete counter
        kk+=1

        CellIDsTFC, _ = \
        t3000_cell_selection_snap_analysis(targetT,TRACERSPARAMS,HaloID,\
        elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
        saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,lazyLoadBool,SUBSET=None)


        snapRange = [zz for zz in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['finalSnap'])+1,int(TRACERSPARAMS['snapMax'])+1), 1)]

        #Loop over snaps from snapMin to snapmax, taking the finalSnap (the final snap) as the endpoint if snapMax is greater

        #Setup arguments combinations for parallel processing pool
        print("\n" + f"Sorting multi-core arguments!")

        args_default = [targetT,TRACERSPARAMS,HaloID,CellIDsTFC,\
        elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
        saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,lazyLoadBool]

        args_list = [[snap]+args_default for snap in snapRange]

        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
        #   MAIN ANALYSIS
        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
        # Open multiprocesssing pool
        print("\n" + f"Opening {n_processes} core Pool!")
        mp.log_to_stderr(logging.DEBUG)
        pool = mp.Pool(processes=n_processes)

        #Compute Snap analysis
        output_list = [pool.apply_async(t3000_snap_analysis,args=args) for args in args_list]

        pool.close()
        pool.join()
        #Close multiprocesssing pool
        print(f"Closing core Pool!")
        # for snap in snapRange:
        #     out = t3000_snap_analysis(snap,targetT,TRACERSPARAMS,HaloID,CellIDsTFC,\
        #     elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
        #     saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,lazyLoadBool)
        #-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
