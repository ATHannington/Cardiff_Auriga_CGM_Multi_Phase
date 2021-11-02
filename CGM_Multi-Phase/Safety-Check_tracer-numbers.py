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

print("----------------------------")
print("|                          |")
print("| !!!   Safety Check   !!! |")
print("| Number of Tracers Consistent |")
print("|                            |")
print(f"|    Snap Number:         |")
print(f"|       {snapNumber}       |")
print("|                          |")
print("----------------------------")

print("")
nTracersOuter = []
for halo in haloes:
    loadPath = baseLoadPath + f"halo_{halo}" + "/output/"
    print(f"Starting halo {halo}")

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
    print(f"For halo {halo} at snap {snapNumber} number of tracers = ", nTracers)

    nTracersOuter.append(nTracers)

alltest = np.all(
    np.where(np.array(nTracersOuter) == int(np.median(nTracersOuter)), True, False)
)
print("***")
print(f"All Haloes Safe? = {alltest}")
if alltest == False:
    print(f"WARNING! Haloes have inconsistent Tracer Numbers!")
    for (ind, halo) in enumerate(haloes):
        print(f"Halo_{halo} NTracers = {nTracersOuter[ind]}")
else:
    print("Phew! All haloes are safe.")
print("***")
print("Done! :)")
