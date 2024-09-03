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
    "L9"
]  # ['L10','L3','L2','L5','L6','L7','L9']#['L1','L8','5','6','9','13','17','23','24','26','28']

snapMin = 200
snapMax = 251

print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)} - {int(snapMax)}        |")
print\((.*?),flush=True\)
print\((.*?),flush=True\)
r_vir = 200  # 250
print\((.*?),flush=True\)
truthy = []
for halo in haloes:
    loadPath = baseLoadPath + f"halo_{halo}" + "/output/"
    print\((.*?),flush=True\)
    truthyInner = []
    for snapNumber in range(snapMin, snapMax + 1, 1):
        snap_subfind = load_subfind(snapNumber, dir=loadPath)

        snap = gadget_readsnap(
            snapNumber,
            loadPath,
            hdf5=True,
            loadonlytype=[0, 2, 3],
            lazy_load=True,
            subfind=snap_subfind,
        )

        snap = set_centre(snap, snap_subfind, 0, snapNumber)

        whereLowResDM = np.where(np.isin(snap.data["type"], [2, 3]))[0]
        snap.pos *= 1e3  # [kpc]
        snap.data["R"] = np.linalg.norm(snap.data["pos"], axis=1)

        test = np.all(snap.data["R"][whereLowResDM] > r_vir)
        truthyInner.append(test)
        print(
            f"For halo {halo} at snap {snapNumber} all low res DM Radii > R_vir?", test
        )

    halotest = np.all(truthyInner)
    print\((.*?),flush=True\)
    print\((.*?),flush=True\)
    print\((.*?),flush=True\)
    print\((.*?),flush=True\)
    truthy.append(halotest)

alltest = np.all(truthy)
print\((.*?),flush=True\)
print\((.*?),flush=True\)
if alltest == False:
    locbadHaloes = np.where(np.array(truthy) == False)[0]
    badHaloes = np.array(haloes)[locbadHaloes]
    print\((.*?),flush=True\)
else:
    print\((.*?),flush=True\)
print\((.*?),flush=True\)
print\((.*?),flush=True\)")
