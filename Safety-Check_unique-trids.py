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

# Input parameters path:
TracersParamsPath = "TracersParams.csv"
TracersMasterParamsPath = "TracersParamsMaster.csv"
SelectedHaloesPath = "TracersSelectedHaloes.csv"
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
# print("Load Non Time Flattened Data!")
# mergedDict, saveParams = multi_halo_merge(
#     SELECTEDHALOES,
#     HALOPATHS,
#     DataSavepathSuffix,
#     snapRange,
#     Tlst,
#     TracersParamsPath,
# )
# print("Done!")
#
# print("Check trids are unique!")
# for key,values in mergedDict.items():
#   u,c = np.unique(values['trid'],return_counts=True)
#   assert np.shape(np.where(c>1)[0])[0]<=0, f"[Not time flattened] {key} Duplicate Trids Detected! Fatal! \n {np.shape(u[c>1])} \n {u[c>1]} "
# print("Done!")

# =============================================================================#
#                   Load Flattened Data                                       #
# =============================================================================#

# del mergedDict

print("Load Time Flattened Data!")
flatMergedDict, _ = multi_halo_merge_flat_wrt_time(
    SELECTEDHALOES, HALOPATHS, DataSavepathSuffix, snapRange, Tlst, TracersParamsPath
)
print("Done!")

print("Check trids are unique!")
for key, values in flatMergedDict.items():
    u, c = np.unique(values["trid"][0, :], return_counts=True)
    assert (
        np.shape(np.where(c > 1)[0])[0] <= 0
    ), f"[Time flattened] {key} Duplicate Trids Detected! Fatal! \n {np.shape(u[c>1])} \n {u[c>1]} "
print("Done!")
