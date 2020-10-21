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

import multiprocessing as mp
import sys
import logging

TracersParamsPath = 'TracersParams.csv'

singleValueParams = ['Lookback','Ntracers','Snap']
#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

saveParams = TRACERSPARAMS['saveParams'] + TRACERSPARAMS['saveTracersOnly'] + TRACERSPARAMS['saveEssentials']

for param in singleValueParams:
    saveParams.remove(param)

DataSavepathSuffix = f".h5"

print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)


flatDict = {}
print("Flattening wrt time!")
for targetT in TRACERSPARAMS['targetTLst']:
    out = flatten_wrt_time(targetT,dataDict,TRACERSPARAMS,saveParams)
    flatDict.update(out)

savePath = DataSavepath + f"_flat-wrt-time"+ DataSavepathSuffix

print("\n" + f": Saving flat data as: "+ savePath)

hdf5_save(savePath,flatDict)
