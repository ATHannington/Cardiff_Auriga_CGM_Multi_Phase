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
import os
from Tracers_Subroutines import *
from Tracers_MultiHalo_Plotting_Tools import *
from random import sample
import math

flatMergedDict, _ = multi_halo_merge_flat_wrt_time(
    ["halo_6"],
    ["/home/universe/c1838736/Tracers/V11-1/halo_6/"],
    ".h5",
    snapRange=[snap for snap in range(221, 251+1,1)],
    Tlst=["4.0","5.0","6.0"], 
    TracersParamsPath = "TracersParams.csv",
)

mergedDict, _ = multi_halo_merge(
    ["halo_6"],
    ["/home/universe/c1838736/Tracers/V11-1/halo_6/"],
    ".h5",
    snapRange=[snap for snap in range(221, 251+1,1)],
    Tlst=["4.0","5.0","6.0"], 
    TracersParamsPath = "TracersParams.csv",
)

singleDict = hdf5_load("/home/universe/c1838736/Tracers/V11-1/halo_6/Data_selectSnap235_targetT4.0-5.0-6.0_T6.0_125.0R175.0_225.h5")