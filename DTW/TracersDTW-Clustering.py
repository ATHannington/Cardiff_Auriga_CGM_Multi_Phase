import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
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

TracersParamsPath = "TracersParams.csv"

sort_level = 4
maxmimally_distinct_bool = False

method = "ward"
DPI = 250
xsize = 50
ysize = 20
colourmapIndividuals = "nipy_spectral"
colour = "tab:gray"
rgbcolour = mcolors.to_rgb(colour)
opacity = 0.5
subset = 50
# ==============================================================================#
def get_d_crit(Z, sort_level, maxmimally_distinct_bool):
    distance_levels = np.array(abs(Z[:-1, 2] - Z[1:, 2]))

    if maxmimally_distinct_bool == True:
        sorted_distance_levels_index = np.argsort(distance_levels)

        level = -1 - sort_level

        biggest_diff_loc = sorted_distance_levels_index[level]

        d_crit = Z[biggest_diff_loc, 2] + (0.01 * distance_levels[biggest_diff_loc])
    else:
        level = -1 - sort_level
        d_crit = Z[level, 2] - 1e-5 * Z[level - 1, 2]

    print(f"d_crit = {d_crit:0.010f}")
    return d_crit


# ==============================================================================#
# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

dtwParams = TRACERSPARAMS["dtwParams"]
logParams = TRACERSPARAMS["dtwlogParams"]

DataSavepathSuffix = f".h5"

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath, TRACERSPARAMS, DataSavepathSuffix)

xDataDict = {}
snapRange = range(
    int(TRACERSPARAMS["snapMin"]),
    min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"] + 1)),
)

for T in Tlst:
    tmp = []
    for snap in snapRange:
        key = (f"T{T}", f"{int(snap)}")
        tmp.append(dataDict[key][f"Lookback"].copy())
    newKey = f"T{T}"
    tmp = np.array(tmp).flatten()
    xDataDict.update({newKey: tmp})

del dataDict

for T in Tlst:
    for analysisParam in dtwParams:
        if analysisParam in logParams:
            loadKey = (f"T{T}", f"log10{analysisParam}")
            loadPath = (
                DataSavepath
                + f"_T{T}_log10{analysisParam}_DTW-distance"
                + DataSavepathSuffix
            )
            print(
                "\n"
                + f"[@T{T} log10{analysisParam}]: Loading Distance Matrix data as: "
                + loadPath
            )
        else:
            loadKey = (f"T{T}", f"{analysisParam}")
            loadPath = (
                DataSavepath
                + f"_T{T}_{analysisParam}_DTW-distance"
                + DataSavepathSuffix
            )
            print(
                "\n"
                + f"[@T{T} {analysisParam}]: Loading Distance Matrix data as: "
                + loadPath
            )

        dtwDict = hdf5_load(loadPath)

        D = dtwDict[loadKey]["distance_matrix"].copy()
        maxD = np.nanmax(D)
        D = D / maxD

        M = dtwDict[loadKey]["data"].copy()

        xData = xDataDict[f"T{T}"]

        print("Linkage!")
        Z = linkage(D, method=method)
        dendo_plot = plt.figure(figsize=(xsize, ysize))

        if analysisParam in logParams:
            plt.title(
                f'T{T} log10{analysisParam} Hierarchical Clustering Dendrogram using "{method}" method'
            )
        else:
            plt.title(
                f'T{T} {analysisParam} Hierarchical Clustering Dendrogram using "{method}" method'
            )
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")
        print("Dendrogram!")
        ddata = dendrogram(Z, color_threshold=1.0)

        prefixList = TRACERSPARAMS["savepath"].split("/")
        prefixList = prefixList[(-4):-2]
        prefix = "-".join(prefixList)

        if analysisParam in logParams:
            opslaan = (
                f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                + prefix
                + f"_T{T}_log10{analysisParam}"
                + f"_Dendrogram.pdf"
            )
        else:
            opslaan = (
                f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                + prefix
                + f"_T{T}_{analysisParam}"
                + f"_Dendrogram.pdf"
            )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)

        # print(f"Check Dendrogram and input d_crit params:")
        # maxmimally_distinct_bool_input = input("Enter maxmimally_distinct_bool true=1, false=0 : ")
        # maxmimally_distinct_bool = bool(int(maxmimally_distinct_bool_input))
        #
        # sort_level_input = input("Enter sort_level 0 top/most distinct, 1 second to top/second most distinct : ")
        # sort_level = int(sort_level_input)

        print("D_crit!")
        d_crit = get_d_crit(Z, sort_level, maxmimally_distinct_bool)
        print("Fcluster!")
        clusters = fcluster(Z, t=d_crit, criterion="distance")

        uniqueClusters = np.unique(clusters)

        print("Cluster Plots!")
        for clusterID in uniqueClusters:
            cluster = M[np.where(clusters == clusterID)]
            clusterIndices = [xx for xx in range(len(cluster))]
            subsetClusterIndices = sample(clusterIndices, min(subset, len(cluster)))
            plotYdata = cluster[subsetClusterIndices]

            cluster_plot, ax = plt.subplots()
            if analysisParam in logParams:
                plt.title(
                    f'Cluster {clusterID} for {T} log10{analysisParam} Hierarchical Clustering using "{method}" method'
                )
                plt.xlabel("Lookback [Gyr]")
                plt.ylabel(f"Log10{analysisParam}")
            else:
                plt.title(
                    f'Cluster {clusterID} for {T} {analysisParam} Hierarchical Clustering using "{method}" method'
                )
                plt.xlabel("Lookback [Gyr]")
                plt.ylabel(f"{analysisParam}")

            plotXdata = np.array([xData for path in range(len(plotYdata))])

            paths = np.array([plotXdata.T, plotYdata.T]).T.reshape(-1, len(xData), 2)
            # colourList = [rgbcolour for ii in range(len(segments))]
            lc = LineCollection(paths, color=colour, alpha=opacity)
            line = ax.add_collection(lc)
            ax.autoscale()
            ax.set_xlim(np.nanmin(xData), np.nanmax(xData))
            ax.set_ylim(np.nanmin(M), np.nanmax(M))
            # for path in cluster:
            #     plt.plot(xData,path,color=colour,alpha=opacity)

            if analysisParam in logParams:
                opslaan2 = (
                    f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                    + prefix
                    + f"_Cluster{clusterID}_T{T}_log10{analysisParam}"
                    + f"_Clustered-Individuals.pdf"
                )
            else:
                opslaan2 = (
                    f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                    + prefix
                    + f"_Cluster{clusterID}_T{T}_{analysisParam}"
                    + f"_Clustered-Individuals.pdf"
                )

            plt.savefig(opslaan2, dpi=DPI, transparent=False)
            print(opslaan2)
            plt.close()

        saveDict = {}
        saveDict.update({"clusters": clusters})
        saveDict.update({"prid": dtwDict[loadKey]["prid"].copy()})
        saveDict.update({"trid": dtwDict[loadKey]["prid"].copy()})
        saveDict.update({"d_crit": np.array([d_crit])})
        saveDict.update(
            {"maxmimally_distinct_bool": np.array([maxmimally_distinct_bool])}
        )
        saveDict.update({"sort_level": np.array([sort_level])})

        if analysisParam in logParams:
            saveDict.update({f"log10{analysisParam}": M})
        else:
            saveDict.update({f"{analysisParam}": M})

        finalDict = {f"T{T}": saveDict}

        if analysisParam in logParams:
            savePath = (
                DataSavepath
                + f"_T{T}_log10{analysisParam}_DTW-clusters"
                + DataSavepathSuffix
            )
            print(
                "\n"
                + f"[@T{T} log10{analysisParam}]: Saving Clusters data as: "
                + savePath
            )
        else:
            savePath = (
                DataSavepath
                + f"_T{T}_{analysisParam}_DTW-clusters"
                + DataSavepathSuffix
            )
            print(
                "\n" + f"[@T{T} {analysisParam}]: Saving Clusters data as: " + savePath
            )

        hdf5_save(savePath, finalDict)
