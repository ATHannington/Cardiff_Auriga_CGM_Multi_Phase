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
import os
from Tracers_Subroutines import *
from Tracers_MultiHalo_Plotting_Tools import *
from random import sample
import math


DPI = 150
xsize = 7.0
ysize = 4.0
epsilon = 0.25
epsilonRadial = 50.0
opacityPercentiles = 0.25
lineStyleMedian = "solid"
lineStylePercentiles = "-."
colourmapMain = "plasma"

# Toggle Titles
titleBool = False
# Toggle separate legend output pdf
separateLegend = True

# Input parameters path:
TracersParamsPath = "TracersParams.csv"
TracersMasterParamsPath = "TracersParamsMaster.csv"
SelectedHaloesPath = "TracersSelectedHaloes.csv"

logParameters = [
    "dens",
    "ndens",
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
    "P_CR",
    "PCR_Pthermal",
    "gah",
    "Grad_T",
    "Grad_n_H",
    "Grad_bfld",
    "Grad_P_CR",
    "gima",
    "Grad_T",
    "Grad_n_H",
    "Grad_bfld",
    "Grad_P_CR",
    "tcool",
    "theat",
    "tcross",
    "tff",
    "tcool_tff",
    "mass",
]
ylabel = {
    "T": r"Temperature (K)",
    "R": r"Radius (kpc)",
    "n_H": r"n$_H$ (cm$^{-3}$)",
    "B": r"|B| ($ \mu $G)",
    "vrad": r"Radial Velocity (km s$^{-1}$)",
    "gz": r"Metallicity Z$_{\odot}$",
    "L": r"Specific Angular Momentum" + "\n" + r"(kpc km s$^{-1}$)",
    "P_thermal": r"P$_{Thermal}$ / k$_B$ (K cm$^{-3}$)",
    "P_magnetic": r"P$_{Magnetic}$ / k$_B$ (K cm$^{-3}$)",
    "P_kinetic": r"P$_{Kinetic}$ / k$_B$ (K cm$^{-3}$)",
    "P_tot": r"P$_{tot}$ = (P$_{thermal}$ + P$_{magnetic}$)/ k$_B$"
    + "\n"
    + r"(K cm$^{-3}$)",
    "Pthermal_Pmagnetic": r"P$_{thermal}$/P$_{magnetic}$",
    "P_CR": r"P$_{CR}$ (K cm$^{-3}$)",
    "PCR_Pthermal": r"(X$_{CR}$ = P$_{CR}$/P$_{Thermal}$)",
    "gah": r"Alfven Gas Heating (erg s$^{-1}$)",
    "bfld": r"||B-Field|| ($ \mu $G)",
    "Grad_T": r"||Temperature Gradient|| (K kpc$^{-1}$)",
    "Grad_n_H": r"||n$_H$ Gradient|| (cm$^{-3}$ kpc$^{-1}$)",
    "Grad_bfld": r"||B-Field Gradient|| ($ \mu $G kpc$^{-1}$)",
    "Grad_P_CR": r"||P$_{CR}$ Gradient|| (K kpc$^{-4}$)",
    "gima" : r"Star Formation Rate (M$_{\odot}$ yr$^{-1}$)",
    # "crac" : r"Alfven CR Cooling (erg s$^{-1}$)",
    "tcool": r"Cooling Time (Gyr)",
    "theat": r"Heating Time (Gyr)",
    "tcross": r"Sound Crossing Cell Time (Gyr)",
    "tff": r"Free Fall Time (Gyr)",
    "tcool_tff": r"t$_{Cool}$/t$_{FreeFall}$",
    "csound": r"Sound Speed (km s$^{-1}$)",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "dens": r"Density (g cm$^{-3}$)",
    "ndens": r"Number density (cm$^{-3}$)",
    "mass": r"Mass (M$_{\odot}$)",
}

for entry in logParameters:
    ylabel[entry] = r"$Log_{10}$" + ylabel[entry]

#   Perform forbidden log of Grad check
deleteParams = []
for entry in logParameters:
    entrySplit = entry.split("_")
    if (
        ("Grad" in entrySplit) &
        (np.any(np.isin(np.array(logParameters), np.array(
            "_".join(entrySplit[1:])))))
    ):
        deleteParams.append(entry)

for entry in deleteParams:
    logParameters.remove(entry)
# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersMasterParamsPath)

# Load Halo Selection Data
SELECTEDHALOES, HALOPATHS = load_haloes_selected(
    HaloPathBase=TRACERSPARAMS["savepath"], SelectedHaloesPath=SelectedHaloesPath
)

saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

DataSavepathSuffix = f".h5"

snapRange = [
    snap
    for snap in range(
        int(TRACERSPARAMS["snapMin"]),
        min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
        1,
    )
]


# ==============================================================================#
print("Load Non Time Flattened Data 1st Halo ONLY!")
mergedDict, _ = multi_halo_merge(
    SELECTEDHALOES[:1],
    HALOPATHS[:1],
    DataSavepathSuffix,
    snapRange,
    Tlst,
    TracersParamsPath,
)
print("Done!")

selectionSnap = np.array(snapRange)[
    np.where(np.array(snapRange) == int(TRACERSPARAMS["selectSnap"]))[0]
]
selectTimeKey = (
    f"T{Tlst[0]}",
    f"{TRACERSPARAMS['Rinner'][0]}R{TRACERSPARAMS['Router'][0]}",
    f"{int(selectionSnap)}",
)
selectTime = abs(mergedDict[selectTimeKey]["Lookback"][0])

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

# =============================================================================#
#                   Load Flattened Data                                       #
# =============================================================================#

# del mergedDict

print("Load Time Flattened Data!")
flatMergedDict, _ = multi_halo_merge_flat_wrt_time(
    SELECTEDHALOES, HALOPATHS, DataSavepathSuffix, snapRange, Tlst, TracersParamsPath
)
print("Done!")

selectTimeKey = (
    f"T{Tlst[0]}",
    f"{TRACERSPARAMS['Rinner'][0]}R{TRACERSPARAMS['Router'][0]}",
)

for rin, rout in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
    savePath = f"./MultiHalo/{int(rin)}R{int(rout)}/"
    tmp = "./"
    for savePathChunk in savePath.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass


# Debug test
# for snap in snapRange:
#     timeIndex =  np.where(np.array(snapRange) == snap)[0]
#     print(flatMergedDict[selectTimeKey]['pos'][timeIndex])

# =============================================================================#
#                     Stats!                                                   # 
# =============================================================================#
print("Calculate multi halo statistics")
statsData = multi_halo_statistics(
    flatMergedDict, TRACERSPARAMS, saveParams, snapRange, Tlst
)

print("Save multi halo statistics")
save_statistics_csv(statsData, TRACERSPARAMS, Tlst, snapRange)
# ============================================================================#
# #                   Medians PLOT                                              #
# #=============================================================================#
matplotlib.rc_file_defaults()
plt.close("all")
medians_plot(
    flatMergedDict,
    statsData,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    logParameters,
    ylabel,
    titleBool=titleBool,
    separateLegend=separateLegend,
    radialSummaryBool=False,
    DPI=DPI,
    xsize=xsize,
    ysize=ysize,
    opacityPercentiles=opacityPercentiles,
    lineStyleMedian=lineStyleMedian,
    lineStylePercentiles=lineStylePercentiles,
    colourmapMain=colourmapMain,
)
matplotlib.rc_file_defaults()
plt.close("all")

matplotlib.rc_file_defaults()
plt.close("all")
medians_plot(
    flatMergedDict,
    statsData,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    logParameters,
    ylabel,
    titleBool=titleBool,
    separateLegend=separateLegend,
    radialSummaryBool=True,
    DPI=DPI,
    xsize=xsize,
    ysize=ysize,
    opacityPercentiles=opacityPercentiles,
    lineStyleMedian=lineStyleMedian,
    lineStylePercentiles=lineStylePercentiles,
    colourmapMain=colourmapMain,
)
matplotlib.rc_file_defaults()
plt.close("all")

# ==============================================================================#
# #         Persistently Currently Random Draw Temperature PLOT               #
# # ============================================================================#
matplotlib.rc_file_defaults()
plt.close("all")
currently_or_persistently_at_temperature_plot(
    flatMergedDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    DataSavepath,
    titleBool=titleBool,
    DPI=DPI,
    xsize=xsize,
    ysize=ysize,
    opacityPercentiles=opacityPercentiles,
    lineStyleMedian=lineStyleMedian,
    lineStylePercentiles=lineStylePercentiles,
    colourmapMain=colourmapMain,
)
matplotlib.rc_file_defaults()
plt.close("all")


# ============================================================================#
#       Temperature Variation PLOT                                              
#=============================================================================#
matplotlib.rc_file_defaults()
plt.close("all")
temperature_variation_plot(
    flatMergedDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    logParameters,
    ylabel,
    titleBool=titleBool,
    DPI=DPI,
    xsize=xsize,
    ysize=ysize,
    opacityPercentiles=opacityPercentiles,
    lineStyleMedian=lineStyleMedian,
    lineStylePercentiles=lineStylePercentiles,
    colourmapMain=colourmapMain,
)

matplotlib.rc_file_defaults()
plt.close("all")

# ============================================================================#
# # #                   Bar Chart PLOT                                            #
# # # =============================================================================#
matplotlib.rc_file_defaults()
plt.close("all")
bars_plot(
    flatMergedDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    selectTime,
    snapRange,
    Tlst,
    DataSavepath,
    titleBool=titleBool,
    separateLegend=separateLegend,
    DPI=DPI,
    opacityPercentiles=opacityPercentiles,
    lineStyleMedian=lineStyleMedian,
    lineStylePercentiles=lineStylePercentiles,
    colourmapMain=colourmapMain,
    epsilon = epsilon,
    epsilonRadial = epsilonRadial,
)
matplotlib.rc_file_defaults()
plt.close("all")

bars_plot(
    flatMergedDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    selectTime,
    snapRange,
    Tlst,
    DataSavepath,
    shortSnapRangeBool=True,
    shortSnapRangeNumber=len(snapRange)//4,
    titleBool=titleBool,
    separateLegend=separateLegend,
    DPI=DPI,
    opacityPercentiles=opacityPercentiles,
    lineStyleMedian=lineStyleMedian,
    lineStylePercentiles=lineStylePercentiles,
    colourmapMain=colourmapMain,
    epsilon = epsilon,
    epsilonRadial = epsilonRadial,
)
matplotlib.rc_file_defaults()
plt.close("all")
# ============================================================================#
# # ============================================================================#
# # #                         Non-Paper Plots
# # #
# # ============================================================================#
# # ============================================================================#
#
#
# # # ============================================================================#
# # # #                   Stacked PDF PLOT                                          #
# # # # =============================================================================#
# matplotlib.rc_file_defaults()
# plt.close("all")
# stacked_pdf_plot(
#     flatMergedDict,
#     TRACERSPARAMS,
#     saveParams,
#     tlookback,
#     snapRange,
#     Tlst,
#     logParameters,
#     ylabel,
#     titleBool,
#     DPI,
# )
# matplotlib.rc_file_defaults()
# plt.close("all")
# # #

# =============================================================================#
# #                Medians and Phases Combo                                     #
# # =============================================================================#
#
# for param in saveParams:
#     print("")
#     print("---")
#     print(f"medians_phases_plot : for {param} param")
#     print("---")
#     matplotlib.rc_file_defaults()
#     plt.close("all")
#     medians_phases_plot(
#         flatMergedDict,
#         statsData,
#         TRACERSPARAMS,
#         saveParams,
#         tlookback,
#         selectTime,
#         snapRange,
#         Tlst,
#         logParameters,
#         ylabel,
#         SELECTEDHALOES,
#         DPI,
#         weightKey="mass",
#         analysisParam=param,
#         Nbins=100,
#         titleBool = titleBool
#     )
#     matplotlib.rc_file_defaults()
#     plt.close("all")

# print("Load Non Time Flattened Data!")
# mergedDict, _ = multi_halo_merge(
#     SELECTEDHALOES,
#     HALOPATHS,
#     DataSavepathSuffix,
#     snapRange,
#     Tlst,
#     TracersParamsPath,
# )
# print("Done!")
#
# matplotlib.rc_file_defaults()
# plt.close("all")
# phases_plot(
#     dataDict = mergedDict,
#     TRACERSPARAMS = TRACERSPARAMS,
#     saveParams=saveParams,
#     snapRange=snapRange,
#     Tlst = [4.0,5.0,6.0],
#     titleBool = True,
#     ylabel = ylabel,
#     DPI=100,
#     xsize=20.0,
#     ysize=5.0,
#     opacityPercentiles=0.25,
#     lineStyleMedian="solid",
#     lineStylePercentiles="-.",
#     colourmapMain="plasma",
#     DataSavepathSuffix=f".h5",
#     TracersParamsPath="TracersParams.csv",
#     TracersMasterParamsPath="TracersParamsMaster.csv",
#     SelectedHaloesPath="TracersSelectedHaloes.csv",
#     Nbins=250)
# matplotlib.rc_file_defaults()
# plt.close("all")
