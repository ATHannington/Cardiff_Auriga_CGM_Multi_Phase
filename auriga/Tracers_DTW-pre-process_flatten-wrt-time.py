import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

TracersParamsPath = 'TracersParams.csv'

#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".h5"

print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

print("Flattening wrt time!")

flatDict = flatten_wrt_time(dataDict,TRACERSPARAMS,saveParams=TRACERSPARAMS['saveParams'])

savePath = DataSavepath + f"_flat-wrt-time"+ DataSavepathSuffix

print("\n" + f": Saving flat data as: "+ savePath)

hdf5_save(savePath,flatDict)
