"""
Author: A. T. Hannington
Created: 27/07/2021

Known Bugs:
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from Tracers_MultiHaloPlottingTools import *
from random import sample
import math

xsize = 10.0
ysize = 12.0
DPI = 100

# Set style options
opacityPercentiles = 0.25
lineStyleMedian = "solid"
lineStylePercentiles = "-."

ageUniverse = 13.77  # [Gyr]

colourmapMain = "plasma"
# Input parameters path:
TracersParamsPath = "TracersParams.csv"
TracersMasterParamsPath = "TracersParamsMaster.csv"
SelectedHaloesPath = "TracersSelectedHaloes.csv"

logParameters = [
    "dens",
    "rho_rhomean",
    "csound",
    "T",
    "n_H",
    "B",
    "gz",
    "L",
    "P_thermal",
    "P_magnetic",
    "P_kinetic",
    "P_tot",
    "Pthermal_Pmagnetic",
    "tcool",
    "theat",
    "tcross",
    "tff",
    "tcool_tff",
]
# "rho_rhomean,dens,T,R,n_H,B,vrad,gz,L,P_thermal,P_magnetic,P_kinetic,P_tot,tcool,theat,csound,tcross,tff,tcool_tff"
ylabel = {
    "T": r"Temperature [$K$]",
    "R": r"Radius [$kpc$]",
    "n_H": r"$n_H$ [$cm^{-3}$]",
    "B": r"|B| [$\mu G$]",
    "vrad": r"Radial Velocity [$km$ $s^{-1}$]",
    "gz": r"Average Metallicity $Z/Z_{\odot}$",
    "L": r"Specific Angular Momentum[$kpc$ $km$ $s^{-1}$]",
    "P_thermal": r"$P_{Thermal} / k_B$ [$K$ $cm^{-3}$]",
    "P_magnetic": r"$P_{Magnetic} / k_B$ [$K$ $cm^{-3}$]",
    "P_kinetic": r"$P_{Kinetic} / k_B$ [$K$ $cm^{-3}$]",
    "P_tot": r"$P_{tot} = P_{thermal} + P_{magnetic} / k_B$ [$K$ $cm^{-3}$]",
    "Pthermal_Pmagnetic": r"$P_{thermal}/P_{magnetic}$",
    "tcool": r"Cooling Time [$Gyr$]",
    "theat": r"Heating Time [$Gyr$]",
    "tcross": r"Sound Crossing Cell Time [$Gyr$]",
    "tff": r"Free Fall Time [$Gyr$]",
    "tcool_tff": r"Cooling Time over Free Fall Time",
    "csound": r"Sound Speed",
    "rho_rhomean": r"Density over Average Universe Density",
    "dens": r"Density [$g$ $cm^{-3}$]",
    "ndens": r"Number density [# $cm^{-3}$]",
}

for entry in logParameters:
    ylabel[entry] = r"Log10 " + ylabel[entry]

# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersMasterParamsPath)

# Load Halo Selection Data
SELECTEDHALOES, HALOPATHS = load_haloes_selected(HaloPathBase = TRACERSPARAMS['savepath'] ,SelectedHaloesPath=SelectedHaloesPath)

saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

DataSavepathSuffix = f".h5"

snapRange = [snap for snap in range(
        int(TRACERSPARAMS["snapMin"]),
        min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
        1)]


# ==============================================================================#
print("Load Non Time Flattened Data!")
mergedDict, saveParams =  multi_halo_merge(SELECTEDHALOES,
                            HALOPATHS,
                            DataSavepathSuffix,
                            snapRange,
                            Tlst,
                            TracersParamsPath
                            )
print("Done!")

selectionSnap = np.array(snapRange)[np.where(np.array(snapRange)== int(TRACERSPARAMS["selectSnap"]))[0]]
selectTime = abs(
        mergedDict[
            (
                f"T{Tlst[0]}",
                f"{TRACERSPARAMS['Rinner'][0]}R{TRACERSPARAMS['Router'][0]}",
                f"{int(selectionSnap)}",
            )
        ]["Lookback"][0]
    )
# ==============================================================================#
#           PLOT!!
# ==============================================================================#
tlookback = []
for snap in range(
        int(TRACERSPARAMS["snapMin"]),
        min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
        1,
):
    minTemp = TRACERSPARAMS["targetTLst"][0]
    minrin = TRACERSPARAMS["Rinner"][0]
    minrout = TRACERSPARAMS["Router"][0]
    key = (f"T{minTemp}", f"{minrin}R{minrout}", f"{int(snap)}")

    tlookback.append(mergedDict[key]["Lookback"][0])

tlookback = np.array(tlookback)


statsData = multi_halo_stats(mergedDict,TRACERSPARAMS,saveParams,snapRange,Tlst)

#==============================================================================#
#                   Medians PLOT                                               #
#==============================================================================#

medians_plot(mergedDict,statsData,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst)

#==============================================================================#
#                   Persistent Temperature PLOT                                #
#==============================================================================#

persistant_temperature_plot(mergedDict,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst)

#==============================================================================#
#                   Within Temperature PLOT                                    #
#==============================================================================#

within_temperature_plot(mergedDict,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst)

#==============================================================================#
#                   Stacked PDF PLOT                                           #
#==============================================================================#

stacked_pdf_plot(mergedDict,statsData,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst)

#==============================================================================#
#                   Phase Diagrams PLOT                                        #
#==============================================================================#

phases_plot(mergedDict,TRACERSPARAMS,saveParams,snapRange,Tlst)
#==============================================================================#
#                   Load Flattened Data                                        #
#==============================================================================#

del mergedDict

print("Load Time Flattened Data!")
flatMergedDict , _ = multi_halo_merge_flat_wrt_time(SELECTEDHALOES,
                            HALOPATHS,
                            DataSavepathSuffix,
                            snapRange,
                            Tlst,
                            TracersParamsPath
                            )
print("Done!")
#==============================================================================#
#                   Bar Chart PLOT                                             #
#==============================================================================#
bars_plot(flatMergedDict,TRACERSPARAMS,saveParams,tlookback,selectTime,snapRange,Tlst,DataSavepath)
