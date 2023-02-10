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

# Toggle Trio titles
trioTitleBool = False
titleBool = False

subset = 200
Ntails = 6
numThreads = 8
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

# load in the subfind group files
snap_subfind = load_subfind(TRACERSPARAMS["selectSnap"], dir=TRACERSPARAMS["simfile"])

# load in the gas particles mass and position only for HaloID 0.
#   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
#       gas and stars (type 0 and 4) MUST be loaded first!!
snapGas = gadget_readsnap(
    TRACERSPARAMS["selectSnap"],
    TRACERSPARAMS["simfile"],
    hdf5=True,
    loadonlytype=[4],
    lazy_load=True,
    subfind=snap_subfind,
)

print(
    f"[@{int(TRACERSPARAMS['selectSnap'])}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
)


snapGas.calc_sf_indizes(snap_subfind, halolist=[int(TRACERSPARAMS["haloID"])])
rotation_matrix = snapGas.select_halo(snap_subfind, do_rotation=True)


tracer_plot(
    dataDict,
    tridData,
    TRACERSPARAMS,
    rotation_matrix,
    DataSavepath,
    FullDataPathSuffix=f".h5",
    Axes=TRACERSPARAMS["Axes"],
    zAxis=TRACERSPARAMS["zAxis"],
    boxsize=TRACERSPARAMS["boxsize"],
    boxlos=TRACERSPARAMS["boxlos"],
    pixres=TRACERSPARAMS["pixres"],
    pixreslos=TRACERSPARAMS["pixreslos"],
    numThreads=numThreads,
    MaxSubset=subset,
    tailsLength=Ntails,
    trioTitleBool=trioTitleBool,
    titleBool=titleBool,
)
