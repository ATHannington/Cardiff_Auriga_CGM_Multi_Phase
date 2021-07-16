"""
Author: A. T. Hannington
Created: 26/03/2020

Known Bugs:
    pandas read_csv loading data as nested dict. . Have added flattening to fix

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
from random import sample
import math
from functools import reduce

subset = 100
ageUniverse = 13.77  # [Gyr]

TracersParamsPath = "TracersParams.csv"
singleValueParams = ["Lookback", "Ntracers", "Snap"]
DataSavepathSuffix = f".h5"
# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersParamsPath)

dataParams = (
        TRACERSPARAMS["saveParams"]
        + TRACERSPARAMS["saveTracersOnly"]
        + TRACERSPARAMS["saveEssentials"]
)

for param in singleValueParams:
    dataParams.remove(param)

print("Loading data!")

dataDict = full_dict_hdf5_load(DataSavepath, TRACERSPARAMS, DataSavepathSuffix)

# ==============================================================================#
#   Get Data within range of z-axis LOS
# ==============================================================================#
print("Getting Tracer Data!")

boxlos = TRACERSPARAMS["boxlos"]
zAxis = int(TRACERSPARAMS["zAxis"][0])
# Loop over temperatures in targetTLst and grab Temperature specific subset of tracers in zAxis LOS range at selectionSnap
tridData = {}
for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
    print(f"{rin}R{rout}")
    for T in TRACERSPARAMS["targetTLst"]:
        print("")
        print(f"T{T}")
        # Grab the data for tracers in projection LOS volume
        key = (f"T{T}", f"{rin}R{rout}", f"{int(TRACERSPARAMS['selectSnap'])}")
        whereGas = np.where(dataDict[key]["type"] == 0)
        whereInRange = np.where(
            (dataDict[key]["pos"][whereGas][:, zAxis] <= (float(boxlos) / 2.0))
            & (dataDict[key]["pos"][whereGas][:, zAxis] >= (-1.0 * float(boxlos) / 2.0))
        )
        pridsIndices = np.where(
            np.isin(
                dataDict[key]["prid"][whereGas],
                dataDict[key]["id"][whereGas][whereInRange],
            )
        )[0]
        trids = dataDict[key]["trid"][whereGas][pridsIndices]

        tridData.update({key: trids})

# ==============================================================================#
#   Get Data within range of z-axis LOS common between ALL time-steps
# ==============================================================================#
#
# trid_list = []
# for subDict in dataDict.values():
#     trid_list.append(subDict['trid'])
#
# intersect = reduce(np.intersect1d,trid_list)
#
# intersectList =[]
# for outerkey, subDict in dataDict.items():
#     trids = subDict['trid']
#     tmp = {}
#     for key, value in subDict.items():
#
#         entry, a_ind, b_ind = np.intersect1d(trids,intersect,return_indices=True)
#         prids = subDict['prid'][a_ind]
#
#         _, id_indices, _ = np.intersect1d(subDict['id'],prids,return_indices=True)
#
#         tmp.update({key : value[id_indices]})
#         intersectList.append(a_ind)
#     dataDict.update({outerkey : tmp})
#
# oldIntersect = intersectList[0]
# for entry in intersectList:
#     assert np.shape(entry) == np.shape(oldIntersect)

# ==============================================================================#
#   Get Data within range of z-axis LOS common between ALL time-steps
# ==============================================================================#

TracerPlot(
    dataDict,
    tridData,
    TRACERSPARAMS,
    DataSavepath,
    FullDataPathSuffix=f".h5",
    Axes=TRACERSPARAMS["Axes"],
    zAxis=TRACERSPARAMS["zAxis"],
    boxsize=TRACERSPARAMS["boxsize"],
    boxlos=TRACERSPARAMS["boxlos"],
    pixres=TRACERSPARAMS["pixres"],
    pixreslos=TRACERSPARAMS["pixreslos"],
    numThreads=2,
    MaxSubset=subset,
)
