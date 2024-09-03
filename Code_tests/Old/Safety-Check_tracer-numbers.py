import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import const as c
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *

baseLoadPath = "/home/universe/spxfv/Auriga/level4_MHD_reruns/"

haloes = [
    "L10",
    "L3",
    "L2",
    "L5",
    "L6",
    "L7",
    "L1",
    "L8",
    "5",
    "6",
    "9",
    "13",
    "17",
    "23",
    "24",
    "26",
    "28",
]

snapNumber = 251

print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)

print\((.*?),flush=True\)
nTracersOuter = []
for halo in haloes:
    loadPath = baseLoadPath + f"halo_{halo}" + "/output/"
    print\((.*?),flush=True\)

    snap_subfind = load_subfind(snapNumber, dir=loadPath)

    snap = gadget_readsnap(
        snapNumber,
        loadPath,
        hdf5=True,
        loadonlytype=[6],
        lazy_load=True,
        subfind=snap_subfind,
    )

    nTracers = np.shape(snap.data["type"])[0]
    print\((.*?),flush=True\)

    nTracersOuter.append(nTracers)

alltest = np.all(
    np.where(np.array(nTracersOuter) == int(np.median(nTracersOuter)), True, False)
)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
if alltest == False:
    print\((.*?),flush=True\)
    for (ind, halo) in enumerate(haloes):
        print\((.*?),flush=True\)
else:
    print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)")
