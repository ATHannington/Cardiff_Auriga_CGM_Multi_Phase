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
from Tracers_MultiHalo_Plotting_Tools import *
from random import sample
import math

xsize = 5.0
ysize = 6.0
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
    "tcool",
    "theat",
    "tcross",
    "tff",
    "tcool_tff",
    "mass",
]
# "rho_rhomean,dens,T,R,n_H,B,vrad,gz,L,P_thermal,P_magnetic,P_kinetic,P_tot,tcool,theat,csound,tcross,tff,tcool_tff"
ylabel = {
    "T": r"Temperature [K]",
    "R": r"Radius [kpc]",
    "n_H": r"n$_H$ [cm$^{-3}$]",
    "B": r"|B| [$ \mu $G]",
    "vrad": r"Radial Velocity [km s$^{-1}$]",
    "gz": r"Average Metallicity Z/Z$_{\odot}$",
    "L": r"Specific Angular Momentum[kpc km s$^{-1}$]",
    "P_thermal": r"P_${Thermal}$ / k$_B$ [K cm$^{-3}$]",
    "P_magnetic": r"P$_{Magnetic}$ / k$_B$ [K cm$^{-3}$]",
    "P_kinetic": r"P$_{Kinetic}$ / k$_B$ [K cm$^{-3}$]",
    "P_tot": r"P$_{tot}$ = (P$_{thermal}$ + P$_{magnetic}$)/ k$_B$ [K cm$^{-3}$]",
    "Pthermal_Pmagnetic": r"P$_{thermal}$/P$_{magnetic}$",
    "tcool": r"Cooling Time [Gyr]",
    "theat": r"Heating Time [Gyr]",
    "tcross": r"Sound Crossing Cell Time [Gyr]",
    "tff": r"Free Fall Time [Gyr]",
    "tcool_tff": r"t$_{Cool}$/t$_{FreeFall}$",
    "csound": r"Sound Speed [km s$^{-1}$]",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "dens": r"Density [g cm$^{-3}$]",
    "ndens": r"Number density [cm$^{-3}$]",
    "mass": r"Log10 Mass per pixel [M/M$_{\odot}$]",
}

for entry in logParameters:
    ylabel[entry] = r"$Log_{10}$" + ylabel[entry]

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
mergedDict, saveParams = multi_halo_merge(
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

# Debug test
# for snap in snapRange:
#     timeIndex =  np.where(np.array(snapRange) == snap)[0]
#     print(flatMergedDict[selectTimeKey]['pos'][timeIndex])

# =============================================================================#
#                     Stats!                                                  # ==============================================================================#

# statsData = multi_halo_statistics(flatMergedDict, TRACERSPARAMS, saveParams, snapRange, Tlst)
#
# save_statistics_csv(
#     statsData, TRACERSPARAMS, SELECTEDHALOES, HALOPATHS, Tlst, snapRange
# )
# # ============================================================================#
# #                   Medians PLOT                                              # #=============================================================================#
# matplotlib.rc_file_defaults()
# plt.close("all")
# medians_plot(
#     flatMergedDict,
#     statsData,
#     TRACERSPARAMS,
#     saveParams,
#     tlookback,
#     snapRange,
#     Tlst,
#     logParameters,
#     ylabel,
# )
# matplotlib.rc_file_defaults()
# plt.close("all")
#
# ==============================================================================#
#                   Persistently Temperature PLOT                             #
# ============================================================================#

currently_or_persistently_at_temperature_plot(
    flatMergedDict, TRACERSPARAMS, saveParams, tlookback, snapRange, Tlst,
    persistenceBool = True)
matplotlib.rc_file_defaults()
plt.close("all")
# ============================================================================#
#                   Currently Temperature PLOT                                #
# ============================================================================#

currently_or_persistently_at_temperature_plot(
    flatMergedDict, TRACERSPARAMS, saveParams, tlookback, snapRange, Tlst,
    persistenceBool = False)
matplotlib.rc_file_defaults()
plt.close("all")
# ============================================================================#
# #                   Stacked PDF PLOT                                          #
# # =============================================================================#
#
# stacked_pdf_plot(
#     flatMergedDict,
#     TRACERSPARAMS,
#     saveParams,
#     tlookback,
#     snapRange,
#     Tlst,
#     logParameters,
#     ylabel,
# )
# matplotlib.rc_file_defaults()
# plt.close("all")
# # ============================================================================#
# #                   Bar Chart PLOT                                            #
# # =============================================================================#
#
# bars_plot(
#     flatMergedDict,
#     TRACERSPARAMS,
#     saveParams,
#     tlookback,
#     selectTime,
#     snapRange,
#     Tlst,
#     DataSavepath,
# )
# matplotlib.rc_file_defaults()
# plt.close("all")
#
# bars_plot(
#     flatMergedDict,
#     TRACERSPARAMS,
#     saveParams,
#     tlookback,
#     selectTime,
#     snapRange,
#     Tlst,
#     DataSavepath,
#     shortSnapRangeBool=True,
#     shortSnapRangeNumber=1,
# )
# matplotlib.rc_file_defaults()
# plt.close("all")

# =============================================================================#
#                Medians and Phases Combo                                     #
# =============================================================================#
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
#         weightKey="mass",
#         analysisParam=param,
#         Nbins=100,
#     )
#     matplotlib.rc_file_defaults()
#     plt.close("all")
