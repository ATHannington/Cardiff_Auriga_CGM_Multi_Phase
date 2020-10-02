import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from itertools import combinations
import time
import torch
import torch.nn as nn

from soft_dtw_cuda import SoftDTW

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

dtwParams = ['T','R','dens','gz','P_tot','B','vrad']
TracersParamsPath = 'TracersParams.csv'
batch_limit = 1e5
printpercent = 1.
#------------------------------------------------------------------------------#
def DTW_prep(M):
    """
    Function to obtain unique combinations of time series.
    Will return time series index in Matrix m (0,1,2....) and unique partner (0,1,2...) ignoring (1,2)==(2,1) etc
    Returns pairs as list of tuples
    """
    elements = range(np.shape(M)[0])
    iterator = combinations(elements,r=2)

    return iterator

#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".h5"


print("Loading data!")

dataDict = {}

loadPath = DataSavepath + f"_flat-wrt-time"+ DataSavepathSuffix

dataDict = hdf5_load(loadPath)

print(torch.__version__)

cuda = torch.device('cuda')

n_gpus = torch.cuda.device_count()
print(f"Running on {n_gpus} GPUs")
multi_batch_limit = n_gpus * batch_limit
n_pairs = int(batch_limit**3)
while True:
  last_batch_size = n_pairs%multi_batch_limit
  if last_batch_size > 1:
    break
  else:
    multi_batch_limit -= 1

print("last_batch_size",last_batch_size)
print("multi_batch_limit",multi_batch_limit)

for T in Tlst:
    print(f"\n ***Starting T{T} Analyis!***")
    for analysisParam in dtwParams:
        print(f"Starting T{T} {analysisParam} Analysis!")

        print("Load M matrix...")
        key = f"T{T}"
        M = np.log10(dataDict[key][analysisParam].T.copy())
        print("...Loaded M matrix!")

        print("Prep iterator!")
        iterator = DTW_prep(M)

        print("Load DTW instance!")
        dtw = nn.DataParallel(SoftDTW(use_cuda=True,gamma=1e-10,normalize=True))

        print("Send M to Mtens cuda!")
        Mtens = torch.tensor(M, device=cuda).view(np.shape(M)[0],np.shape(M)[1],1)
        print("Make blank cuda tens!")
        out = torch.empty((0), dtype=torch.float64, device=cuda)
        print("Let's do the Time Warp...")
        start = time.time()

        percent = 0.
        start = time.time()
        xlist =[]
        ylist =[]

        for (xx,yy) in iterator:
          xlist.append(xx)
          ylist.append(yy)
          percentage = float(xx)/float(np.shape(M)[0]) * 100.
          if percentage >= percent :
            print(f"{percentage:2.0f}%")
            percent += printpercent
          if (len(xlist)>=multi_batch_limit):
            # print("Time Warping!")
            x = Mtens[xlist].view(len(xlist),np.shape(M)[1],1)
            y = Mtens[ylist].view(len(ylist),np.shape(M)[1],1)
            out_tmp = dtw.forward(x,y)
            out = torch.cat((out,out_tmp),dim=0)
            xlist =[]
            ylist =[]

        print("Finishing up...")
        x = Mtens[xlist].view(len(xlist),np.shape(M)[1],1)
        y = Mtens[ylist].view(len(ylist),np.shape(M)[1],1)
        out_tmp = dtw.forward(x,y)
        out = torch.cat((out,out_tmp),dim=0)

        end = time.time()
        elapsed = end - start
        print(f"Elapsed time in DTW = {elapsed}s")

        saveKey = (f"T{T}",f"log10{analysisParam}")
        D = out.cpu().detach().numpy()
        saveSubDict = {'distance_matrix': D ,'trid': dataDict[f"T{T}"]['trid'].copy(),'prid': dataDict[f"T{T}"]['prid'].copy()}
        saveDict = {saveKey : saveSubDict}

        savePath = DataSavepath + f"_T{T}_log10{analysisParam}_DTW-distance"+DataSavepathSuffix

        print("\n" + f"[@T{T} log10{analysisParam}]: Saving Distance Matrix data as: "+ savePath)

        hdf5_save(savePath,saveDict)
