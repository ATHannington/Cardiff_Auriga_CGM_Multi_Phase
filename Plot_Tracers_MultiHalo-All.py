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
radialLimit = 200.0
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

logParameters = [ "nh", "dens", "ndens", "rho_rhomean", "rho", "vol", "csound", "T", "n_H", "n_H_col", "n_HI", "n_HI_col", "B", "gz", "L", "P_thermal", "P_magnetic", "P_kinetic", "P_tot", "Pthermal_Pmagnetic", "P_CR", "PCR_Pthermal", "gah", "Grad_T", "Grad_n_H", "Grad_bfld", "Grad_P_CR", "tcool", "theat", "tcross", "tff", "tcool_tff", "mass", "gima", "count"]

ylabel = {
    "T": r"Temperature (K)",
    "R": r"Radius (kpc)",
    "n_H": r"n$_H$ (cm$^{-3}$)",
    "n_H_col": r"n$_H$ (cm$^{-2}$)",
    "n_HI": r"n$_{HI}$ (cm$^{-3}$)",
    "n_HI_col": r"n$_{HI}$ (cm$^{-2}$)",
    "nh": r"Neutral Hydrogen Fraction",
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
    "rho": r"Density (M$_{\odot}$ kpc$^{-3}$)",
    "dens": r"Density (g cm$^{-3}$)",
    "ndens": r"Number density (cm$^{-3}$)",
    "mass": r"Mass (M$_{\odot}$)",
    "vol": r"Volume (kpc$^{3}$)",
    "age": "Lookback Time (Gyr)",
    "cool_length" : "Cooling Length (kpc)",
    "halo" : "FoF Halo",
    "subhalo" : "SubFind Halo",
    "x": r"x (kpc)",
    "y": r"y (kpc)",
    "z": r"z (kpc)",
    "count": r"Number of data points per pixel",
}

# ==============================================================================#
#   Save Property keys and associated Labels/definitions
# ==============================================================================#

labeldf = pd.DataFrame.from_dict(ylabel, orient='index')    
labeldf = labeldf.reset_index() 
labeldf.columns = ["Property Key", "Property Definition"]
labeldf.to_csv("Tracers_Property_Legend_Dictionary.csv",index=False)
print("\n"+f"Saved Property Symbol to Label Map as 'Tracers_Property_Legend_Dictionary.csv' !"+"\n")
# ==============================================================================#


for entry in logParameters:
    ylabel[entry] = r"$\mathrm{Log_{10}}$" + ylabel[entry]

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
    hush = True
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

# # matplotlib.rc_file_defaults()
# # plt.close("all")
# # medians_plot(
# #     flatMergedDict,
# #     statsData,
# #     TRACERSPARAMS,
# #     saveParams,
# #     tlookback,
# #     snapRange,
# #     [Tlst[0]],
# #     logParameters,
# #     ylabel,
# #     titleBool=titleBool,
# #     separateLegend=separateLegend,
# #     radialSummaryBool=True,
# #     radialSummaryFirstLastBool= True,
# #     DPI=DPI,
# #     xsize=xsize,
# #     ysize=ysize,
# #     opacityPercentiles=opacityPercentiles,
# #     lineStyleMedian=lineStyleMedian,
# #     lineStylePercentiles=lineStylePercentiles,
# #     colourmapMain=colourmapMain,
# # )
# # matplotlib.rc_file_defaults()
# # plt.close("all")


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
    radialSummaryFirstLastBool= True,
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
    radialLimit = radialLimit,
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
    shortSnapRangeNumber=1,#len(snapRange)//4,
    titleBool=titleBool,
    separateLegend=separateLegend,
    DPI=DPI,
    opacityPercentiles=opacityPercentiles,
    lineStyleMedian=lineStyleMedian,
    lineStylePercentiles=lineStylePercentiles,
    colourmapMain=colourmapMain,
    epsilon = epsilon,
    epsilonRadial = epsilonRadial,
    radialLimit = radialLimit,
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

# # datList = [vals for vals in flatMergedDict.values()]

# # allData = {}
# # for dd in datList:
# #     for key, value in dd.items():
# #         if key in list(allData.keys()):
# #             val = allData[key]
# #             val = np.concatenate((val,value),axis=1)
# #             allData.update({key: val})
# #         else:
# #             allData.update({key: value})

# # del datList, flatMergedDict

# # allData = {key: val[0] for key,val in allData.items()}

# # for ax, ind in zip(["x","y","z"],[0,1,2]):
# #     allData[ax]=allData["pos"][:,:,ind]

# # import Plotting_tools as apt
# # print("plot")
# # apt.phase_plot(
# #     allData,
# #     ylabel,
# #     xlimDict = {},
# #     logParameters = logParameters,
# #     snapNumber = "",
# #     yParams = ["x","y","z"],
# #     xParams = ["x","y","z"],
# #     weightKeys = ["halo","subhalo"],
# #     DPI=350,
# #     Nbins = 350,
# #     savePathBase = "./",
# #     savePathBaseFigureData = "./",
# #     saveFigureData = True,
# #     inplace = True,
# # )

# # # Make normal dictionary form of snap
# # out = {}
# # for key, value in snapGas.data.items():
# #     if value is not None:
# #         out.update({key: copy.deepcopy(value)})

# # whereNotDM = out["type"] != 1

# # import CR_Subroutines as cr 

# # out = cr.remove_selection(
# #     out,
# #     removalConditionMask = whereNotDM,
# #     errorString = f"Remove Not DM",
# #     verbose = False,
# #     )

# # whereBeyond500kpc= np.abs(out["pos"]) > 500.0

# # out = cr.remove_selection(
# #     out,
# #     removalConditionMask = whereBeyond500kpc,
# #     errorString = f"Remove Beyond 500kpc",
# #     verbose = False,
# #     )

# # for ax, ind in zip(["x","y","z"],[0,1,2]):
# #     out[ax]=out["pos"][:,ind]

# # out["halo"][np.where(np.isin(out["halo"],np.asarray([-1,0]))==False)[0]] = 1
# # out["subhalo"][np.where(np.isin(out["subhalo"],np.asarray([-1,0]))==False)[0]] = 1

# # import Plotting_tools as apt
# # print("plot")
# # apt.phase_plot(
# #     out,
# #     ylabel,
# #     xlimDict = {},
# #     logParameters = logParameters,
# #     snapNumber = "",
# #     yParams = ["x","y","z"],
# #     xParams = ["x","y","z"],
# #     weightKeys = ["halo","subhalo"],
# #     xsize = 12.0,
# #     ysize = 12.0,
# #     DPI=400,
# #     Nbins = 500,
# #     savePathBase = "./",
# #     savePathBaseFigureData = "./",
# #     saveFigureData = True,
# #     inplace = True,
# # )
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
