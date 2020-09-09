import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
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

# load in the subfind group files
snap_subfind = load_subfind(snapNumber,dir=simfile)

# load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
snapGas     = gadget_readsnap(snapNumber, simfile, hdf5=True, loadonlytype = [0,4], lazy_load=True, subfind = snap_subfind)

#Pad stars and gas data with Nones so that all keys have values of same first dimension shape
snapGas = PadNonEntries(snapGas)

#Select only gas in High Res Zoom Region
snapGas = HighResOnlyGasSelect(snapGas)
