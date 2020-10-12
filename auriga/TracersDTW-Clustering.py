import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
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

logParams = ['T','dens','P_tot','B']

TracersParamsPath = 'TracersParams.csv'

sort_level = 3
maxmimally_distinct_bool = False

method = 'ward'
DPI = 250
xsize =50
ysize =20
colourmapIndividuals = "nipy_spectral"
colour = "tab:gray"
rgbcolour = mcolors.to_rgb(colour)
opacity = 0.5
subset = 50
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


tmpanalysisDict = {}
for T in Tlst:
    for analysisParam in dtwParams:
        key = f"T{T}"
        if (analysisParam in logParams):
            newkey = (f"T{T}",f"log10{analysisParam}")
            tmpanalysisDict.update({newkey : np.log10(dataDict[key][analysisParam].T.copy())})
        else:
            newkey = (f"T{T}",f"{analysisParam}")
            tmpanalysisDict.update({newkey : dataDict[key][analysisParam].T.copy()})

analysisDict, whereDict = delete_nan_inf_axis(tmpanalysisDict,axis=1)


for T in Tlst:
    dtwT_MDict = {}
    dtwT_DDict = {}
    dtwT_PridDict = {}
    dtwT_TridDict = {}
    for analysisParam in dtwParams:
        if (analysisParam in logParams):
            loadKey = (f"T{T}",f"log10{analysisParam}")
            loadPath = DataSavepath + f"_T{T}_log10{analysisParam}_DTW-distance"+DataSavepathSuffix
            print("\n" + f"[@T{T} log10{analysisParam}]: Loading Distance Matrix data as: "+ loadPath)
        else:
            loadKey = (f"T{T}",f"{analysisParam}")
            loadPath = DataSavepath + f"_T{T}_{analysisParam}_DTW-distance"+DataSavepathSuffix
            print("\n" + f"[@T{T} {analysisParam}]: Loading Distance Matrix data as: "+ loadPath)

        dtwDict = hdf5_load(loadPath)

        D = dtwDict[loadKey]['distance_matrix'].copy()
        maxD = np.nanmax(D)
        D = D/maxD

        M = analysisDict[loadKey].copy()

        if (analysisParam in logParams):
            dtwT_MDict.update({f"log10{analysisParam}" : M})
            dtwT_DDict.update({f"log10{analysisParam}" : D})
            dtwT_PridDict.update({f"log10{analysisParam}" : dtwDict[loadKey]['prid'].copy()})
            dtwT_TridDict.update({f"log10{analysisParam}" : dtwDict[loadKey]['trid'].copy()})
        else:
            dtwT_MDict.update({f"{analysisParam}" : M})
            dtwT_DDict.update({f"{analysisParam}" : D})
            dtwT_PridDict.update({f"{analysisParam}" : dtwDict[loadKey]['prid'].copy()})
            dtwT_TridDict.update({f"{analysisParam}" : dtwDict[loadKey]['trid'].copy()})

        xData = dataDict[f"T{T}"][f"Lookback"].copy()

        print("Linkage!")
        Z= linkage(D,method=method)
        dendo_plot = plt.figure(figsize=(xsize,ysize))

        if (analysisParam in logParams):
            plt.title(f'T{T} log10{analysisParam} Hierarchical Clustering Dendrogram using "{method}" method')
        else:
            plt.title(f'T{T} {analysisParam} Hierarchical Clustering Dendrogram using "{method}" method')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        print("Dendrogram!")
        ddata = dendrogram(Z, color_threshold=1.)

        prefixList = TRACERSPARAMS['savepath'].split('/')
        prefixList = prefixList[(-4):-2]
        prefix = "-".join(prefixList)

        if (analysisParam in logParams):
            opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"_T{T}_log10{analysisParam}"+f"_Dendrogram.pdf"
        else:
            opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"_T{T}_{analysisParam}"+f"_Dendrogram.pdf"
        plt.savefig(opslaan, dpi = DPI, transparent = False)
        print(opslaan)

        # print(f"Check Dendrogram and input d_crit params:")
        # maxmimally_distinct_bool_input = input("Enter maxmimally_distinct_bool true=1, false=0 : ")
        # maxmimally_distinct_bool = bool(int(maxmimally_distinct_bool_input))
        #
        # sort_level_input = input("Enter sort_level 0 top/most distinct, 1 second to top/second most distinct : ")
        # sort_level = int(sort_level_input)

        print("D_crit!")
        d_crit = get_d_crit(Z,sort_level,maxmimally_distinct_bool)
        print("Fcluster!")
        clusters = fcluster(Z, t=d_crit,criterion='distance')

        uniqueClusters =np.unique(clusters)

        print("Cluster Plots!")
        for clusterID in uniqueClusters:
            cluster = M[np.where(clusters==clusterID)]
            clusterIndices = [xx for xx in range(len(cluster))]
            subsetClusterIndices = sample(clusterIndices,min(subset,len(cluster)))
            plotYdata = cluster[subsetClusterIndices]

            cluster_plot, ax = plt.subplots()
            if (analysisParam in logParams):
                plt.title(f'Cluster {clusterID} for {T} log10{analysisParam} Hierarchical Clustering using "{method}" method')
                plt.xlabel('Lookback [Gyr]')
                plt.ylabel(f'Log10{analysisParam}')
            else:
                plt.title(f'Cluster {clusterID} for {T} {analysisParam} Hierarchical Clustering using "{method}" method')
                plt.xlabel('Lookback [Gyr]')
                plt.ylabel(f'{analysisParam}')

            plotXdata = np.array([xData[:,0] for path in range(len(plotYdata))])

            paths = np.array([plotXdata.T, plotYdata.T]).T.reshape(-1,len(xData[:,0]),2)
            # colourList = [rgbcolour for ii in range(len(segments))]
            lc = LineCollection(paths,color = colour,alpha=opacity)
            line = ax.add_collection(lc)
            ax.autoscale()
            ax.set_xlim(np.nanmin(xData),np.nanmax(xData))
            ax.set_ylim(np.nanmin(M),np.nanmax(M))
            # for path in cluster:
            #     plt.plot(xData,path,color=colour,alpha=opacity)

            if (analysisParam in logParams):
                opslaan2 = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"_Cluster{clusterID}_T{T}_log10{analysisParam}"+f"_Clustered-Individuals.pdf"
            else:
                opslaan2 = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"_Cluster{clusterID}_T{T}_{analysisParam}"+f"_Clustered-Individuals.pdf"

            plt.savefig(opslaan2, dpi = DPI, transparent = False)
            print(opslaan2)
            plt.close()

        saveDict = {}
        saveDict.update({"clusters" : clusters})
        saveDict.update({"prid" : dtwDict[loadKey]['prid'].copy()})
        saveDict.update({"trid" : dtwDict[loadKey]['prid'].copy()})
        if (analysisParam in logParams):
            saveDict.update({f"log10{analysisParam}" : M})
        else:
            saveDict.update({f"{analysisParam}" : M})

        finalDict = {f"T{T}" : saveDict}

        if (analysisParam in logParams):
            savePath = DataSavepath + f"_T{T}_log10{analysisParam}_DTW-clusters"+DataSavepathSuffix
            print("\n" + f"[@T{T} log10{analysisParam}]: Saving Clusters data as: "+ savePath)
        else:
            savePath = DataSavepath + f"_T{T}_{analysisParam}_DTW-clusters"+DataSavepathSuffix
            print("\n" + f"[@T{T} {analysisParam}]: Saving Clusters data as: "+ savePath)

        hdf5_save(savePath,finalDict)

#------------------------------------------------------------------------------#
    del dtwDict, M, D, saveDict, lc, line, paths, plotXdata, plotYdata, cluster, clusters,Z,ddata,xData,dendo_plot,d_crit,uniqueClusters,clusterIndices,subsetClusterIndices
    paramstring = "+".join(dtwParams)
    plt.close('all')

    xData = dataDict[f"T{T}"][f"Lookback"].copy()

    print(f"Get intersection of trids!")
    dtwT_TridDictkeys = list(dtwT_TridDict.keys())
    dtwT_TridDict_list = list(dtwT_TridDict.values())
    trid_list = []
    for entry in dtwT_TridDict.values():
        trid_list.append(entry[:,0].tolist())
    tridSet = [set(tuple(xx)) for xx in trid_list]
    whereTracers =  np.where(np.isin(dtwT_TridDict_list[0][:,0],np.array(list(set.intersection(*tridSet)))))[0]
    print(f"Shape whereTracers {np.shape(whereTracers)}")
    del trid_list, tridSet

    #Normalise and then sum the distance vectors for each dtwParams
    print("Djoint! This may take a while...")

    kk = 0
    for key,value in dtwT_DDict.items():
        print(f"{key}")
        if (kk ==0 ):
            Djoint = np.zeros(shape=(np.shape(value)))
        entry = squareform(value)[whereTracers[:,np.newaxis],whereTracers]
        Djoint += squareform(entry)
        kk+=1

    print("Joint Linkage!")
    Zjoint= linkage(Djoint,method=method)

    dendo_plot = plt.figure(figsize=(xsize,ysize))
    if (analysisParam in logParams):
        plt.title(f'T{T} log10{paramstring} Hierarchical Clustering Dendrogram using "{method}" method')
    else:
        plt.title(f'T{T} {paramstring} Hierarchical Clustering Dendrogram using "{method}" method')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    print("Joint dendrogram!")
    ddata = dendrogram(Zjoint, color_threshold=1.)

    prefixList = TRACERSPARAMS['savepath'].split('/')
    prefixList = prefixList[(-4):-2]
    prefix = "-".join(prefixList)

    if (analysisParam in logParams):
        opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"_T{T}_log10{paramstring}"+f"_Joint-Dendrogram.pdf"
    else:
        opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"_T{T}_{paramstring}"+f"_Joint-Dendrogram.pdf"

    plt.savefig(opslaan, dpi = DPI, transparent = False)
    print(opslaan)

    # print(f"Check Dendrogram and input d_crit params:")
    # maxmimally_distinct_bool_input = input("Enter maxmimally_distinct_bool true=1, false=0 : ")
    # maxmimally_distinct_bool = bool(int(maxmimally_distinct_bool_input))
    #
    # sort_level_input = input("Enter sort_level 0 top/most distinct, 1 second to top/second most distinct : ")
    # sort_level = int(sort_level_input)

    print("Joint d_crit!")
    d_crit = get_d_crit(Zjoint,sort_level,maxmimally_distinct_bool)
    print("Joint Fcluster!")
    clusters = fcluster(Zjoint, t=d_crit,criterion='distance')

    uniqueClusters =np.unique(clusters)

    print("Joint clusters!")
    for clusterID in uniqueClusters:
        for key, value in dtwT_MDict.items():
            cluster = value[np.where(clusters==clusterID)]
            ymin = np.nanmin(value)
            ymax = np.nanmax(value)

            clusterIndices = [xx for xx in range(len(cluster))]
            subsetClusterIndices = sample(clusterIndices,min(subset,len(cluster)))
            plotYdata = cluster[subsetClusterIndices]

            cluster_plot, ax = plt.subplots()
            if (analysisParam in logParams):
                plt.title(f'Cluster {clusterID} for {T} log10{paramstring} Hierarchical Clustering using "{method}" method')
                plt.xlabel('Lookback [Gyr]')
                plt.ylabel(f'Log10{key}')
            else:
                plt.title(f'Cluster {clusterID} for {T} {paramstring} Hierarchical Clustering using "{method}" method')
                plt.xlabel('Lookback [Gyr]')
                plt.ylabel(f'{key}')
            plotXdata = np.array([xData[:,0] for path in range(len(plotYdata))])

            paths = np.array([plotXdata.T, plotYdata.T]).T.reshape(-1,len(xData[:,0]),2)

            lc = LineCollection(paths, color = colour ,alpha=opacity)
            line = ax.add_collection(lc)
            ax.autoscale()
            ax.set_xlim(np.nanmin(xData),np.nanmax(xData))
            ax.set_ylim(ymin,ymax)
            if (analysisParam in logParams):
                opslaan2 = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"_Cluster{clusterID}_T{T}_log10{key}_Joint-{paramstring}"+f"_Joint-Clustered-Individuals.pdf"
            else:
                opslaan2 = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+prefix+f"_Cluster{clusterID}_T{T}_{key}_Joint-{paramstring}"+f"_Joint-Clustered-Individuals.pdf"

            plt.savefig(opslaan2, dpi = DPI, transparent = False)
            print(opslaan2)

        tridData = dtwT_TridDict[dtwT_TridDictkeys[0]][whereTracers]
        pridData = dtwT_PridDict[dtwT_TridDictkeys[0]][whereTracers]

        saveDict = {}
        saveDict.update({"clusters" : clusters})
        saveDict.update({"prid" : pridData})
        saveDict.update({"trid" : tridData})
        finalDict = {f"T{T}" : saveDict}
        for param in dtwParams:
            if (param in logParams):
                saveDict.update({f"log10{param}" : dtwT_MDict[f"log10{param}"][whereTracers]})
            else:
                saveDict.update({f"{param}" : dtwT_MDict[f"{param}"][whereTracers]})

        if (analysisParam in logParams):
            savePath = DataSavepath + f"_T{T}_log10{paramstring}_Joint-DTW-clusters"+DataSavepathSuffix
            print("\n" + f"[@T{T} log10{paramstring}]: Saving Joint Clusters data as: "+ savePath)
        else:
            savePath = DataSavepath + f"_T{T}_{paramstring}_Joint-DTW-clusters"+DataSavepathSuffix
            print("\n" + f"[@T{T} {paramstring}]: Saving Joint Clusters data as: "+ savePath)

        hdf5_save(savePath,finalDict)

        plt.close('all')

        del finalDict, saveDict, Djoint, Z, cluster, clusters, clusterIndices, subsetClusterIndices, plotYdata, plotXdata, paths, whereTracers, d_crit
