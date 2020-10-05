import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from itertools import combinations
import time

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
colourmapIndividuals = "nipy_spectral"
rgbcolour = mcolors.to_rgb(colour)
opacity = 0.01
subset = 500
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

for T in ['4.0','5.0']:
    dtwT_MDict = {}
    dtwT_DDict = {}
    for analysisParam in dtwParams:
        loadPath = DataSavepath + f"_T{T}_log10{analysisParam}_DTW-distance"+DataSavepathSuffix

        print("\n" + f"[@T{T} log10{analysisParam}]: Loading Distance Matrix data as: "+ loadPath)

        loadKey = (f"T{T}",f"log10{analysisParam}")

        dtwDict = hdf5_load(loadPath)

        D = dtwDict[loadKey]['distance_matrix'].copy()
        maxD = np.nanmax(D)
        D = D/maxD

        dtwT_DDict.update({f"log10{analysisParam}" : D})

        M = np.log10(dataDict[f"T{T}"][f"{analysisParam}"].T.copy())

        dtwT_MDict.update({f"log10{analysisParam}" : M})

        xData = dataDict[f"T{T}"][f"Lookback"].copy()

        print("Linkage!")
        Z= linkage(D,method=method)
        dendo_plot = plt.figure(figsize=(xsize,ysize))
        plt.title(f'T{T} log10{analysisParam} Hierarchical Clustering Dendrogram using "{method}" method')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        print("Dendrogram!")
        ddata = dendrogram(Z, color_threshold=1.)

        prefixList = TRACERSPARAMS['savepath'].split('/')
        prefixList = prefixList[(-4):-1]
        prefix = "-".join(prefixList)

        opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"T{T}_log10{analysisParam}"+f"_Dendrogram.pdf"
        plt.savefig(opslaan, dpi = DPI, transparent = False)
        print(opslaan)

        print(f"Check Dendrogram and input d_crit params:")
        maxmimally_distinct_bool_input = input("Enter maxmimally_distinct_bool true=1, false=0 : ")
        maxmimally_distinct_bool = bool(int(maxmimally_distinct_bool_input))

        sort_level_input = input("Enter sort_level 0 top/most distinct, 1 second to top/second most distinct : ")
        sort_level = int(sort_level_input)

        print("D_crit!")
        d_crit = get_d_crit(Z,sort_level,maxmimally_distinct_bool)
        print("Fcluster!")
        clusters = fcluster(Z, t=d_crit,criterion='distance')

        uniqueClusters =np.unique(clusters)
        plotYdata = []
        clusterIDdata = []
        for clusterID in uniqueClusters:
            cluster = M[np.where(clusters==clusterID)]
            clusterIndices = [xx for xx in range(len(cluster))]
            subsetClusterIndices = sample(clusterIndices,min(subset,len(cluster)))
            subsetCluster = cluster[subsetClusterIndices]
            plotYdata += subsetCluster.tolist()
            clusterIDtmp = [clusterID for xx in range(len(subsetCluster))]
            clusterIDdata += clusterIDtmp

        plotYdata = np.array(plotYdata)
        clusterIDdata = np.array(clusterIDdata)

        print("Cluster Plots!")

        cluster_plot, ax = plt.subplots()

        plt.title(f'Clusters for {T} log10{analysisParam} Hierarchical Clustering using "{method}" method')
        plt.xlabel('Lookback [Gyr]')
        plt.ylabel(f'Log10{analysisParam}')


        plotXdata = np.array([xData[:,0] for path in range(len(plotYdata))])

        points = np.array([plotXdata.T, plotYdata.T]).T.reshape(-1,1,2)
        segments = np.concatenate([points[:-1],points[1:]], axis=1)
        # colourList = [rgbcolour for ii in range(len(segments))]
        lc = LineCollection(segments,cmap = colourmapIndividuals ,alpha=opacity)
        line = ax.add_collection(lc)
        lc.set_array(clusterIDdata.flatten())
        ax.autoscale()
        ax.set_xlim(np.min(xData),np.max(xData))
        ax.set_ylim(np.min(M),np.max(M))
        # for path in cluster:
        #     plt.plot(xData,path,color=colour,alpha=opacity)

        opslaan2 = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"Cluster{clusterID}_T{T}_log10{analysisParam}"+f"_Clustered-Individuals.pdf"
        plt.savefig(opslaan2, dpi = DPI, transparent = False)
        print(opslaan2)
        plt.close()
        # for clusterID in uniqueClusters:
        #     cluster = M[np.where(clusters==clusterID)]
        #     cluster_plot, ax = plt.subplots()
        #     plt.title(f'Cluster {clusterID} for {T} log10{analysisParam} Hierarchical Clustering using "{method}" method')
        #     plt.xlabel('Lookback [Gyr]')
        #     plt.ylabel(f'Log10{analysisParam}')
        #
        #
        #     plotXdata = np.array([xData[:,0] for path in range(len(cluster))])
        #
        #     points = np.array([plotXdata.T, cluster.T]).T.reshape(-1,1,2)
        #     segments = np.concatenate([points[:-1],points[1:]], axis=1)
        #     # colourList = [rgbcolour for ii in range(len(segments))]
        #     lc = LineCollection(segments,color = colour,alpha=opacity)
        #     line = ax.add_collection(lc)
        #     ax.autoscale()
        #     ax.set_xlim(np.min(xData),np.max(xData))
        #     ax.set_ylim(np.min(M),np.max(M))
        #     # for path in cluster:
        #     #     plt.plot(xData,path,color=colour,alpha=opacity)
        #
        #     opslaan2 = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"Cluster{clusterID}_T{T}_log10{analysisParam}"+f"_Clustered-Individuals.pdf"
        #     plt.savefig(opslaan2, dpi = DPI, transparent = False)
        #     print(opslaan2)
        #     plt.close()
        dtwDict[loadKey].update({"clusters" : clusters})

        savePath = DataSavepath + f"_T{T}_log10{analysisParam}_DTW-clusters"+DataSavepathSuffix

        print("\n" + f"[@T{T} log10{analysisParam}]: Saving Distance Matrix + Clusters data as: "+ savePath)

        hdf5_save(savePath,dtwDict)

#------------------------------------------------------------------------------#
    paramstring = "+".join(dtwParams)

    #Normalise and then sum the distance vectors for each dtwParams
    print("Djoint!")
    Djoint = np.zeros(np.shape(dtwT_DDict[f"log10{dtwParams[0]}"]))
    for key,value in dtwT_DDict.items():
        Djoint += value

    print("Joint Linkage!")
    Zjoint= linkage(Djoint,method=method)

    dendo_plot = plt.figure(figsize=(xsize,ysize))
    plt.title(f'T{T} log10{paramstring} Hierarchical Clustering Dendrogram using "{method}" method')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    print("Joint dendrogram!")
    ddata = dendrogram(Zjoint, color_threshold=1.)

    prefixList = TRACERSPARAMS['savepath'].split('/')
    prefixList = prefixList[(-4):-1]
    prefix = "-".join(prefixList)

    opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"T{T}_log10{paramstring}"+f"_Joint-Dendrogram.pdf"
    plt.savefig(opslaan, dpi = DPI, transparent = False)
    print(opslaan)

    print(f"Check Dendrogram and input d_crit params:")
    maxmimally_distinct_bool_input = input("Enter maxmimally_distinct_bool true=1, false=0 : ")
    maxmimally_distinct_bool = bool(int(maxmimally_distinct_bool_input))

    sort_level_input = input("Enter sort_level 0 top/most distinct, 1 second to top/second most distinct : ")
    sort_level = int(sort_level_input)

    print("Joint d_crit!")
    d_crit = get_d_crit(Zjoint,sort_level,maxmimally_distinct_bool)
    print("Joint Fcluster!")
    clusters = fcluster(Zjoint, t=d_crit,criterion='distance')

    uniqueClusters =np.unique(clusters)

    print("Joint clusters!")
    for clusterID in uniqueClusters:
        for key, value in dtwT_MDict.items():
            cluster = value[np.where(clusters==clusterID)]
            cluster_plot, ax = plt.subplots()
            plt.title(f'Cluster {clusterID} for {T} log10{paramstring} Hierarchical Clustering using "{method}" method')
            plt.xlabel('Lookback [Gyr]')
            plt.ylabel(f'Log10{key}')

            plotXdata = np.array([xData[:,0] for path in range(len(cluster))])

            points = np.array([plotXdata.T, cluster.T]).T.reshape(-1,1,2)
            segments = np.concatenate([points[:-1],points[1:]], axis=1)
            # colourList = [rgbcolour for ii in range(len(segments))]
            lc = LineCollection(segments,color = colour,alpha=opacity)
            line = ax.add_collection(lc)
            ax.autoscale()
            ax.set_xlim(np.min(xData),np.max(xData))
            ax.set_ylim(np.min(M),np.max(M))

            opslaan2 = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"Cluster{clusterID}_T{T}_log10{key}_Joint-log10{paramstring}"+f"_Joint-Clustered-Individuals.pdf"
            plt.savefig(opslaan2, dpi = DPI, transparent = False)
            print(opslaan2)

    newFullSaveDict = {f"T{T}" : T, "clusters" : clusters, "distance_matrices" : dtwT_DDict,\
     "joint_distance_matrix" : Djoint, "data" : dtwT_MDict, "trid" : dtwDict[loadKey]['trid'].copy()\
     , "prid" : dtwDict[loadKey]['prid'].copy(), "Lookback" : xData}

    savePath = DataSavepath + f"_T{T}_log10{paramstring}_Joint-DTW-clusters"+DataSavepathSuffix

    print("\n" + f"[@T{T} log10{paramstring}]: Saving Joint Distance Matrix + Clusters data as: "+ savePath)

    hdf5_save(savePath,newFullSaveDict)
