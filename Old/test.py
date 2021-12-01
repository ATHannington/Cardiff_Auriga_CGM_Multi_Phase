import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import const as c
from gadget import *
from gadget_subfind import *
import h5py
import sys
import logging
import math
import random
from Tracers_Subroutines import *

snapNumber = 127
simfile = "/home/universe/spxtd1-shared/ISOTOPES/output/"
HaloID = 0

# ==============================================================================#
#       Chemical Properties
# ==============================================================================#
# element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements = [
    "H",
    "He",
    "C",
    "N",
    "O",
    "Ne",
    "Mg",
    "Si",
    "Fe",
    "Y",
    "Sr",
    "Zr",
    "Ba",
    "Pb",
]
elements_Z = [1, 2, 6, 7, 8, 10, 12, 14, 26, 39, 38, 40, 56, 82]
elements_mass = [
    1.01,
    4.00,
    12.01,
    14.01,
    16.00,
    20.18,
    24.30,
    28.08,
    55.85,
    88.91,
    87.62,
    91.22,
    137.33,
    207.2,
]
elements_solar = [
    12.0,
    10.93,
    8.43,
    7.83,
    8.69,
    7.93,
    7.60,
    7.51,
    7.50,
    2.21,
    2.87,
    2.58,
    2.18,
    1.75,
]

Zsolar = 0.0127

omegabaryon0 = 0.048
# ==============================================================================#


# load in the subfind group files
snap_subfind = load_subfind(snapNumber, dir=simfile)

# load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
snapGas = gadget_readsnap(
    snapNumber,
    simfile,
    hdf5=True,
    loadonlytype=[0, 4],
    lazy_load=True,
    subfind=snap_subfind,
)

# Load Cell IDs - avoids having to turn lazy_load off...
# But ensures 'id' is loaded into memory before HaloOnlyGasSelect is called
#  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
#   Be in memory so taking the subset would be skipped.
tmp = snapGas.data["id"]
tmp = snapGas.data["age"]
tmp = snapGas.data["hrgm"]
tmp = snapGas.data["mass"]
del tmp

# Centre the simulation on HaloID 0
snapGas = SetCentre(
    snap=snapGas, snap_subfind=snap_subfind, HaloID=HaloID, snapNumber=snapNumber
)


# --------------------------#
##    Units Conversion    ##
# --------------------------#

# Convert Units
## Make this a seperate function at some point??
snapGas.pos *= 1e3  # [kpc]
snapGas.vol *= 1e9  # [kpc^3]

# Calculate New Parameters and Load into memory others we want to track
snapGas = CalculateTrackedParameters(
    snapGas,
    elements,
    elements_Z,
    elements_mass,
    elements_solar,
    Zsolar,
    omegabaryon0,
    snapNumber,
)

# Pad stars and gas data with Nones so that all keys have values of same first dimension shape
snapGas = PadNonEntries(snapGas, snapNumber)

# Select only gas in High Res Zoom Region
snapGas = HighResOnlyGasSelect(snapGas, snapNumber)

# Find Halo=HaloID data for only selection snapshot. This ensures the
# selected tracers are originally in the Halo, but allows for tracers
# to leave (outflow) or move inwards (inflow) from Halo.

# Assign SubHaloID and FoFHaloIDs
snapGas = HaloIDfinder(snapGas, snap_subfind, snapNumber)

snapGas = HaloOnlyGasSelect(snapGas, snap_subfind, HaloID, snapNumber)

# Pad stars and gas data with Nones so that all keys have values of same first dimension shape
snapGas = PadNonEntries(snapGas, snapNumber)
# Input parameters path:
TracersParamsPath = "TracersParams.csv"

# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

DataSavepathSuffix = f".h5"


print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath, TRACERSPARAMS, DataSavepathSuffix)
