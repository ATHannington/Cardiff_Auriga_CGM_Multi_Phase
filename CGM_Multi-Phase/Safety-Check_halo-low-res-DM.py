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

print("----------------------------")
print("|                          |")
print("| !!!   Safety Check   !!! |")
print("| Halo data not low res DM |")
print("| contaminated within R_vir|")
print(f"|    Snap Numbers:         |")
print(f"|         {int(snapMin)} - {int(snapMax)}        |")
print("|                          |")
print("----------------------------")
r_vir = 200  # 250
print("")
truthy = []
for halo in haloes:
    loadPath = baseLoadPath + f"halo_{halo}" + "/output/"
    print(f"Starting halo {halo}")
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

        snap = SetCentre(snap, snap_subfind, 0, snapNumber)

        whereLowResDM = np.where(np.isin(snap.data["type"], [2, 3]))[0]
        snap.pos *= 1e3  # [kpc]
        snap.data["R"] = np.linalg.norm(snap.data["pos"], axis=1)

        test = np.all(snap.data["R"][whereLowResDM] > r_vir)
        truthyInner.append(test)
        print(
            f"For halo {halo} at snap {snapNumber} all low res DM Radii > R_vir?", test
        )

    halotest = np.all(truthyInner)
    print("---")
    print(f"Halo {halo} Safe? = {halotest}")
    print("---")
    print("")
    truthy.append(halotest)

alltest = np.all(truthy)
print("***")
print(f"All Haloes Safe? = {alltest}")
if alltest == False:
    locbadHaloes = np.where(np.array(truthy) == False)[0]
    badHaloes = np.array(haloes)[locbadHaloes]
    print(f"WARNING! Haloes {badHaloes} are contaminated!")
else:
    print("Phew! All haloes are safe.")
print("***")
print("Done! :)")
