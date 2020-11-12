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

#Number of cores to run on:
n_processes = 4

DataSavepathSuffix = f".h5"
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

print("Flattening wrt time!")

print("\n" + f"Sorting multi-core arguments!")

args_default = [dataDict,TRACERSPARAMS,saveParams,DataSavepath,DataSavepathSuffix]
args_list = [[T]+args_default for T in TRACERSPARAMS['targetTLst']]

#Open multiprocesssing pool

print("\n" + f"Opening {n_processes} core Pool!")
pool = mp.Pool(processes=n_processes)

#Compute Snap analysis
_ = [pool.apply_async(flatten_wrt_time,args=args) for args in args_list]

pool.close()
pool.join()
#Close multiprocesssing pool
print(f"Closing core Pool!")
