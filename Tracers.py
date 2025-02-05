"""
Author: A. T. Hannington
Created: 19/03/2020

Main analysis file for CGM multiphase analysis with Monte Carlo Tracer Particles. This script performs the inital analysis of selection the tracer particle for a single simulations (controlled in TracerParams.csv) before selecting the relevant cell data for those tracer particles at each simulation output ('snapshot', in Arepo terminology).
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *
import h5py
import multiprocessing as mp
import sys
import logging

# ==============================================================================#
#       USER DEFINED PARAMETERS
# ==============================================================================#
# Input parameters path:
TracersParamsPath = "TracersParams.csv"

# File types for data save.
#   Mini: small median and percentiles data
#   Full: full FullDict data
MiniDataPathSuffix = f".h5"
FullDataPathSuffix = f".h5"

# Lazy Load switch. Set to False to save all data (warning, pickle file may explode)
lazyLoadBool = True

# Number of cores to run on:
n_processes = 8

# ==============================================================================#
#       Prepare for analysis
# ==============================================================================#
# Load in parameters from csv. This ensures reproducability!
#   We save as a DataFrame, then convert to a dictionary, and take out nesting...
# Save as .csv
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersParamsPath)

print("")
print("Loaded Analysis Parameters:")
for key, value in TRACERSPARAMS.items():
    print(f"{key}: {value}")

print("")

# Save types, which when combined with saveparams define what data is saved.
#   This is intended to be the string equivalent of the percentiles.
saveTypes = ["_" + str(percentile) + "%" for percentile in TRACERSPARAMS["percentiles"]]

# Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['rho_rhomean','dens','T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','csound','tcross','tff','tcool_tff']

print("")
print("Saved Parameters in this Analysis:")
print(saveParams)

# Optional Tracer only (no stats in .csv) parameters to be saved
#   Cannot guarantee that all Plotting and post-processing are independent of these
#       Will attempt to ensure any necessary parameters are stored in ESSENTIALS
saveTracersOnly = TRACERSPARAMS["saveTracersOnly"]  # ['sfr','age']

print("")
print("Tracers ONLY (no stats) Saved Parameters in this Analysis:")
print(saveTracersOnly)

# SAVE ESSENTIALS : The data required to be tracked in order for the analysis to work
saveEssentials = TRACERSPARAMS[
    "saveEssentials"
]  # ['halo','subhalo','Lookback','Ntracers','Snap','id','prid','trid','type','mass','pos']

print("")
print("ESSENTIAL Saved Parameters in this Analysis:")
print(saveEssentials)

saveTracersOnly = saveTracersOnly + saveEssentials

# Combine saveParams and saveTypes to form each combination for saving data types
saveKeys = []
for param in saveParams:
    for TYPE in saveTypes:
        saveKeys.append(param + TYPE)

# Select Halo of interest:
#   0 is the most massive:
HaloID = int(TRACERSPARAMS["haloID"])
# ==============================================================================#
#       Chemical Properties
# ==============================================================================#
# element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements = [
    "H",
    "He",
    "C",
    "N",
    "O",
    "Ne",
    "Mg",
    "Si",
    "Fe",
    "Y",
    "Sr",
    "Zr",
    "Ba",
    "Pb",
]
elements_Z = [1, 2, 6, 7, 8, 10, 12, 14, 26, 39, 38, 40, 56, 82]
elements_mass = [
    1.01,
    4.00,
    12.01,
    14.01,
    16.00,
    20.18,
    24.30,
    28.08,
    55.85,
    88.91,
    87.62,
    91.22,
    137.33,
    207.2,
]
elements_solar = [
    12.0,
    10.93,
    8.43,
    7.83,
    8.69,
    7.93,
    7.60,
    7.51,
    7.50,
    2.21,
    2.87,
    2.58,
    2.18,
    1.75,
]

Zsolar = 0.0127

omegabaryon0 = 0.048


# ==============================================================================#
#       MAIN PROGRAM
# ==============================================================================#
def err_catcher(arg):
    """ Should allow Child processes of multiprocessing package to properly Raise (or, at the very least, print to stdout) error messages that otherwise might not appear while multiprocessing methods are running.
    """
    raise Exception(f"Child Process died and gave error: {arg}")
    return


if __name__ == "__main__":
    (
        TracersTFC,
        CellsTFC,
        CellIDsTFC,
        ParentsTFC,
        _,
        _,
        rotation_matrix,
    ) = tracer_selection_snap_analysis(
        TRACERSPARAMS,
        HaloID,
        elements,
        elements_Z,
        elements_mass,
        elements_solar,
        Zsolar,
        omegabaryon0,
        saveParams,
        saveTracersOnly,
        DataSavepath,
        FullDataPathSuffix,
        MiniDataPathSuffix,
        lazyLoadBool,
        SUBSET=None,
    )

    snapRange = [
        zz
        for zz in range(
            int(TRACERSPARAMS["snapMin"]),
            min(int(TRACERSPARAMS["finalSnap"]) + 1, int(TRACERSPARAMS["snapMax"]) + 1),
            1,
        )
    ]

    # Loop over snaps from snapMin to snapmax, taking the finalSnap (the final snap) as the endpoint if snapMax is greater

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
    #   MAIN ANALYSIS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
    print("\n" + f"Starting MULTIPROCESSING type Analysis!")
    # Setup arguments combinations for parallel processing pool
    print("\n" + f"Sorting multi-core arguments!")

    args_default = [
        TRACERSPARAMS,
        HaloID,
        TracersTFC,
        elements,
        elements_Z,
        elements_mass,
        elements_solar,
        Zsolar,
        omegabaryon0,
        saveParams,
        saveTracersOnly,
        DataSavepath,
        FullDataPathSuffix,
        MiniDataPathSuffix,
        rotation_matrix,
        lazyLoadBool,
        n_processes,
    ]

    args_list = [[snap] + args_default for snap in snapRange]

    # Open multiprocesssing pool

    print("\n" + f"Opening {n_processes} core Pool!")
    pool = mp.Pool(processes=n_processes)

    # Compute Snap analysis
    output_list = [
        pool.apply_async(snap_analysis, args=args, error_callback=err_catcher)
        for args in args_list
    ]

    pool.close()
    pool.join()
    # Close multiprocesssing pool
    print(f"Closing core Pool!")
    print(f"Final Error checks")
    success = [result.successful() for result in output_list]
    assert all(success) == True, "WARNING: CRITICAL: Child Process Returned Error!"
    print("Done! End of Analysis :)")
    #
    #
    ## Serial variant of analysis that is particularly useful when error messages from child processes (via multiprocessing package) are not indicating simple fixes. Use of this serial variant will allow for use of interactive python debugger to properly delve into, and resolve, bugs in the main snap_analysis subroutine.
    ##
    # print("\n" + f"Starting SERIAL type Analysis!")
    # for snap in snapRange:
    #     out = snap_analysis(snap,TRACERSPARAMS,HaloID,TracersTFC,\
    #     elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
    #     saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,rotation_matrix,lazyLoadBool,n_processes)
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
