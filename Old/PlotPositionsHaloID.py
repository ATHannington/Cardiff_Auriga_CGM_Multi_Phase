import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

TracersParamsPath = "TracersParams.csv"

# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".h5"


print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath, TRACERSPARAMS, DataSavepathSuffix)

snapRange = [
    xx
    for xx in range(
        int(TRACERSPARAMS["snapMin"]),
        min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"] + 1)),
        1,
    )
]

for T in Tlst:
    for snap in snapRange:
        fig = plt.figure()
        key = (f"T{int(T)}", f"{int(snap)}")

        x = dataDict[key]["pos"][:, 0]
        y = dataDict[key]["pos"][:, 1]
        z = dataDict[key]["pos"][:, 2]
        subHalo = dataDict[key]["SubHaloID"]

        plt.scatter(x, y, c=subHalo)
        plt.savefig(f"T{int(T)}-{int(snap)}.png")
