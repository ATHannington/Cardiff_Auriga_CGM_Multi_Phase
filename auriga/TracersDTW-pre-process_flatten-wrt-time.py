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
n_processes = 4
#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

saveParams = TRACERSPARAMS['saveParams'] + TRACERSPARAMS['saveTracersOnly'] + TRACERSPARAMS['saveEssentials']

saveParams.remove('id')

DataSavepathSuffix = f".h5"

print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

print("Flattening wrt time!")
if __name__=="__main__":
    #Loop over target temperatures
    args_default = [dataDict,TRACERSPARAMS,saveParams]

    args_list = [[targetT]+args_default for targetT in TRACERSPARAMS['targetTLst']]

    #Open multiprocesssing pool

    print("\n" + f"Opening {n_processes} core Pool!")
    mp.log_to_stderr(logging.DEBUG)
    pool = mp.Pool(processes=n_processes)

    #Compute Snap analysis
    output_list = [pool.apply_async(flatten_wrt_time,args=args) for args in args_list]

    pool.close()
    pool.join()

flatDict = {}

for (targetT,output) in zip(TRACERSPARAMS['targetTLst'],output_list):
    tempOut = output.get()
    flatDict.update({f"T{targetT}" : tempOut[f"T{targetT}"]})

savePath = DataSavepath + f"_flat-wrt-time"+ DataSavepathSuffix

print("\n" + f": Saving flat data as: "+ savePath)

hdf5_save(savePath,flatDict)
