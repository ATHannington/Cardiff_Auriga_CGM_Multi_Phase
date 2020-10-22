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
logParams = ['T','dens','P_tot','B']

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

dtwSubset = int(TRACERSPARAMS['dtwSubset'])

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
tmpanalysisDict = {}
tmptridDict = {}
tmppridDict = {}
for T in Tlst:
    for analysisParam in dtwParams:
        key = f"T{T}"
        if (analysisParam in logParams):
            newkey = (f"T{T}",f"log10{analysisParam}")
            tmpanalysisDict.update({newkey : np.log10(dataDict[key][analysisParam].T.copy())})
        else:
            newkey = (f"T{T}",f"{analysisParam}")
            tmpanalysisDict.update({newkey : dataDict[key][analysisParam].T.copy()})

        tmptridDict.update({newkey : dataDict[key]['trid']})
        tmppridDict.update({newkey : dataDict[key]['prid']})

analysisDict, whereDict = delete_nan_inf_axis(tmpanalysisDict,axis=1)

del tmpanalysisDict

tridDict = {}
pridDict = {}
for key, values in whereDict.items():
    tridDict.update({key : tmptridDict[key][:,values].T})
    pridDict.update({key : tmppridDict[key][:,values].T})

del whereDict, tmppridDict, tmptridDict

for T in Tlst:
    print(f"\n ***Starting T{T} Analyis!***")
    for analysisParam in dtwParams:
        print(f"Starting T{T} {analysisParam} Analysis!")

        print("Load M matrix...")
        if (analysisParam in logParams):
            key = (f"T{T}",f"log10{analysisParam}")
        else:
            key = (f"T{T}",f"{analysisParam}")
        Mtmp = analysisDict[key]
        print("...Loaded M matrix!")

        maxSize = min(np.shape(Mtmp)[0],dtwSubset)
        if (maxSize<dtwSubset):
            print("Taking full set of Tracers! No RAM issues!")
        elif(maxSize==dtwSubset):
            print(f"Taking subset {maxSize} number of Tracers to prevent RAM overload!")


        subset = tridDict[key][:maxSize,0]
        tridData = []
        pridData = []
        M = []
        for (tracers,parents,Mrow) in zip(tridDict[key].T,pridDict[key].T,Mtmp.T):
            whereSubset = np.where(np.isin(tracers,subset))[0]
            tridData.append(tracers[whereSubset].T)
            pridData.append(parents[whereSubset].T)
            M.append(Mrow[whereSubset].T)

        M = np.array(M).T
        tridData = np.array(tridData).T
        pridData = np.array(pridData).T
        del Mtmp

        print(f"Shape of M : {np.shape(M)}")
        print(f"Shape of tridData : {np.shape(tridData)}")
        print(f"Shape of pridData : {np.shape(pridData)}")

        print("Prep iterator!")
        iterator = DTW_prep(M)

        print("Load DTW instance!")
        dtw = nn.DataParallel(SoftDTW(use_cuda=True,gamma=1e-10,normalize=True))

        print("Send M to Mtens cuda!")
        Mtens = torch.tensor(M, device=cuda).view(np.shape(M)[0],np.shape(M)[1],1)
        print("Make blank list!")
        out = []
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
            out_tmp = out_tmp.cpu().detach().numpy().tolist()
            out += out_tmp
            xlist =[]
            ylist =[]

        print("Finishing up...")
        x = Mtens[xlist].view(len(xlist),np.shape(M)[1],1)
        y = Mtens[ylist].view(len(ylist),np.shape(M)[1],1)
        out_tmp = dtw.forward(x,y)
        out_tmp = out_tmp.cpu().detach().numpy().tolist()
        out += out_tmp

        end = time.time()
        elapsed = end - start
        print(f"Elapsed time in DTW = {elapsed}s")

        D = np.array(out)
        saveSubDict = {'distance_matrix': D ,'trid':tridData ,'prid': pridData,'data' : M}
        saveDict = {key : saveSubDict}

        if (analysisParam in logParams):
            savePath = DataSavepath + f"_T{T}_log10{analysisParam}_DTW-distance"+DataSavepathSuffix
            print("\n" + f"[@T{T} log10{analysisParam}]: Saving Distance Matrix + Sampled Raw Data as: "+ savePath)

        else:
            savePath = DataSavepath + f"_T{T}_{analysisParam}_DTW-distance"+DataSavepathSuffix
            print("\n" + f"[@T{T} {analysisParam}]: Saving Distance Matrix + Sampled Raw Data as: "+ savePath)

        hdf5_save(savePath,saveDict)
