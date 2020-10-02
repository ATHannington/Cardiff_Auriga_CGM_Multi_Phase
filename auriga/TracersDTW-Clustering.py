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

sort_level = 0
maxmimally_distinct_bool = True
method = 'ward'
DPI = 250
xsize =50
ysize =20
#==============================================================================#
def get_d_crit(Z,sort_level,maxmimally_distinct_bool):
    distance_levels = np.array(abs(Z[:-1,2] - Z[1:,2]))

    if (maxmimally_distinct_bool == True) :
      sorted_distance_levels_index = np.argsort(distance_levels)

      level = -1 - sort_level

      biggest_diff_loc = sorted_distance_levels_index[level]

      d_crit = Z[biggest_diff_loc,2] + (0.01  * distance_levels[biggest_diff_loc])
    else:
      level = -1 - sort_level
      d_crit = Z[level,2] - 1e-5*Z[level-1,2]

    print(f"d_crit = {d_crit:0.010f}")
    return d_crit
#==============================================================================#
#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".h5"


print("Loading data!")

dataDict = {}

loadPath = DataSavepath + f"_flat-wrt-time"+ DataSavepathSuffix

dataDict = hdf5_load(loadPath)

for T in Tlst:
    for analysisParam in dtwParams:
        loadPath = DataSavepath + f"_T{T}_log10{analysisParam}_DTW-distance"+DataSavepathSuffix

        print("\n" + f"[@T{T} log10{analysisParam}]: Loading Distance Matrix data as: "+ loadPath)

        loadKey = (f"T{T}",f"log10{analysisParam}")

        dtwDict = hdf5_load(loadPath)

        D = dtwDict[loadKey]['distance_matrix'].copy()

        M = np.log10(dataDict[f"T{T}"][f"{analysisParam}"].copy())

        xData = dataDict[f"T{T}"][f"Lookback"].copy()

        Z= linkage(D,method=method)
        dendo_plot = plt.figure(figsize=(xsize,ysize))
        plt.title(f'T{T} log10{analysisParam} Hierarchical Clustering Dendrogram using "{method}" method')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        ddata = dendrogram(Z, color_threshold=1.)

        prefixList = TRACERSPARAMS['savepath'].split('/')
        prefixList = prefixList[(-4):-1]
        prefix = "-".join(prefixList)

        opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"T{T}_log10{analysisParam}"+"_"+f"_Dendrogram.pdf"
        plt.savefig(opslaan, dpi = DPI, transparent = False)
        print(opslaan)

        d_crit = get_d_crit(Z,sort_level,maxmimally_distinct_bool)
        clusters = fcluster(Z, t=d_crit,criterion='distance')

        uniqueClusters =np.unique(clusters)

        for clusterID in uniqueClusters:
            cluster = M[np.where(clusters==clusterID)]
            cluster_plot = plt.figure()
            plt.title(f'Cluster {clusterIDT} for {T} log10{analysisParam} Hierarchical Clustering using "{method}" method')
            plt.xlabel('Lookback [Gyr]')
            plt.ylabel(f'Log10{analysisParam}')
            for path in cluster:
                plt.plot(xData,path)

            opslaan2 = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"Cluster{clusterID}_T{T}_log10{analysisParam}"+"_"+f"_Clustered-Individuals.pdf"
            plt.savefig(opslaan2, dpi = DPI, transparent = False)
            print(opslaan2)

        dtwDict[loadKey].update({"clusters" : clusters})

        savePath = DataSavepath + f"_T{T}_log10{analysisParam}_DTW-clusters"+DataSavepathSuffix

        print("\n" + f"[@T{T} log10{analysisParam}]: Saving Distance Matrix + Clusters data as: "+ savePath)

        hdf5_save(savePath,dtwDict)
