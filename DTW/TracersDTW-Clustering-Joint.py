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
from functools import reduce

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
    dtwT_MDict = {}
    dtwT_DDict = {}
    dtwT_PridDict = {}
    dtwT_TridDict = {}
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

        M = dtwDict[loadKey]["data"].copy()

        if analysisParam in logParams:
            dtwT_MDict.update({f"log10{analysisParam}": M})
            dtwT_DDict.update({f"log10{analysisParam}": D})
            dtwT_PridDict.update(
                {f"log10{analysisParam}": dtwDict[loadKey]["prid"].copy()}
            )
            dtwT_TridDict.update(
                {f"log10{analysisParam}": dtwDict[loadKey]["trid"].copy()}
            )
        else:
            dtwT_MDict.update({f"{analysisParam}": M})
            dtwT_DDict.update({f"{analysisParam}": D})
            dtwT_PridDict.update({f"{analysisParam}": dtwDict[loadKey]["prid"].copy()})
            dtwT_TridDict.update({f"{analysisParam}": dtwDict[loadKey]["trid"].copy()})

    paramstring = "+".join(dtwParams)
    plt.close("all")

    xData = xDataDict[f"T{T}"]

    print(f"Get intersection of trids!")
    dtwT_TridDictkeys = list(dtwT_TridDict.keys())
    trid_list = []
    for entry in dtwT_TridDict.values():
        trid_list.append(entry[:, 0])

    intersect = reduce(np.intersect1d, trid_list)
    intersectDict = {}
    for analysisParam in dtwParams:
        if analysisParam in logParams:
            key = f"log10{analysisParam}"
        else:
            key = f"{analysisParam}"
        trids = dtwT_TridDict[key][:, 0]
        entry, a_ind, b_ind = np.intersect1d(trids, intersect, return_indices=True)
        dtwT_MDict.update({key: dtwT_MDict[key][a_ind]})
        intersectDict.update({key: a_ind})

    intersectDictList = list(intersectDict.values())
    oldIntersect = intersectDictList[0]
    for key, value in intersectDict.items():
        assert np.shape(value) == np.shape(oldIntersect)

    # Normalise and then sum the distance vectors for each dtwParams
    print("Djoint! This may take a while...")

    kk = 0
    for key, value in dtwT_DDict.items():
        print(f"{key}")
        whereTracers = intersectDict[key]
        print(f"Shape whereTracers {np.shape(whereTracers)}")
        entry = squareform(value)[whereTracers[:, np.newaxis], whereTracers]
        maxD = np.nanmax(entry)
        entry = entry / maxD
        if kk == 0:
            Djoint = np.zeros(shape=np.shape(entry))
        Djoint += entry
        kk += 1

    Djoint = squareform(Djoint)

    print("Joint Linkage!")
    Zjoint = linkage(Djoint, method=method)

    dendo_plot = plt.figure(figsize=(xsize, ysize))
    if analysisParam in logParams:
        plt.title(
            f'T{T} log10{paramstring} Hierarchical Clustering Dendrogram using "{method}" method'
        )
    else:
        plt.title(
            f'T{T} {paramstring} Hierarchical Clustering Dendrogram using "{method}" method'
        )
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    print("Joint dendrogram!")
    ddata = dendrogram(Zjoint, color_threshold=1.0)

    prefixList = TRACERSPARAMS["savepath"].split("/")
    prefixList = prefixList[(-4):-2]
    prefix = "-".join(prefixList)

    if analysisParam in logParams:
        opslaan = (
            f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
            + prefix
            + f"_T{T}_log10{paramstring}"
            + f"_Joint-Dendrogram.pdf"
        )
    else:
        opslaan = (
            f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
            + prefix
            + f"_T{T}_{paramstring}"
            + f"_Joint-Dendrogram.pdf"
        )

    plt.savefig(opslaan, dpi=DPI, transparent=False)
    print(opslaan)

    # print(f"Check Dendrogram and input d_crit params:")
    # maxmimally_distinct_bool_input = input("Enter maxmimally_distinct_bool true=1, false=0 : ")
    # maxmimally_distinct_bool = bool(int(maxmimally_distinct_bool_input))
    #
    # sort_level_input = input("Enter sort_level 0 top/most distinct, 1 second to top/second most distinct : ")
    # sort_level = int(sort_level_input)

    print("Joint d_crit!")
    d_crit = get_d_crit(Zjoint, sort_level, maxmimally_distinct_bool)
    print("Joint Fcluster!")
    clusters = fcluster(Zjoint, t=d_crit, criterion="distance")

    uniqueClusters = np.unique(clusters)

    print("Joint clusters!")
    for clusterID in uniqueClusters:
        for key, value in dtwT_MDict.items():
            cluster = value[np.where(clusters == clusterID)]
            ymin = np.nanmin(value)
            ymax = np.nanmax(value)

            clusterIndices = [xx for xx in range(len(cluster))]
            subsetClusterIndices = sample(clusterIndices, min(subset, len(cluster)))
            plotYdata = cluster[subsetClusterIndices]

            cluster_plot, ax = plt.subplots()
            if analysisParam in logParams:
                plt.title(
                    f'Cluster {clusterID} for {T} log10{paramstring} Hierarchical Clustering using "{method}" method'
                )
                plt.xlabel("Lookback [Gyr]")
                plt.ylabel(f"Log10{key}")
            else:
                plt.title(
                    f'Cluster {clusterID} for {T} {paramstring} Hierarchical Clustering using "{method}" method'
                )
                plt.xlabel("Lookback [Gyr]")
                plt.ylabel(f"{key}")

            plotXdata = np.array([xData for path in range(len(plotYdata))])

            paths = np.array([plotXdata.T, plotYdata.T]).T.reshape(-1, len(xData), 2)

            lc = LineCollection(paths, color=colour, alpha=opacity)
            line = ax.add_collection(lc)
            ax.autoscale()
            ax.set_xlim(np.nanmin(xData), np.nanmax(xData))
            ax.set_ylim(ymin, ymax)
            if analysisParam in logParams:
                opslaan2 = (
                    f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                    + prefix
                    + f"_Cluster{clusterID}_T{T}_log10{key}_Joint-{paramstring}"
                    + f"_Joint-Clustered-Individuals.pdf"
                )
            else:
                opslaan2 = (
                    f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                    + prefix
                    + f"_Cluster{clusterID}_T{T}_{key}_Joint-{paramstring}"
                    + f"_Joint-Clustered-Individuals.pdf"
                )

            plt.savefig(opslaan2, dpi=DPI, transparent=False)
            print(opslaan2)

    whereTracers = intersectDict[dtwT_TridDictkeys[0]]
    tridData = dtwT_TridDict[dtwT_TridDictkeys[0]][whereTracers]
    pridData = dtwT_PridDict[dtwT_TridDictkeys[0]][whereTracers]

    saveDict = {}
    saveDict.update({"clusters": clusters})
    saveDict.update({"prid": pridData})
    saveDict.update({"trid": tridData})
    saveDict.update({"d_crit": np.array([d_crit])})
    saveDict.update({"maxmimally_distinct_bool": np.array([maxmimally_distinct_bool])})
    saveDict.update({"sort_level": np.array([sort_level])})

    for param in dtwParams:
        if param in logParams:
            saveDict.update({f"log10{param}": dtwT_MDict[f"log10{param}"]})
        else:
            saveDict.update({f"{param}": dtwT_MDict[f"{param}"]})

    savePath = (
        DataSavepath + f"_T{T}_{paramstring}_Joint-DTW-clusters" + DataSavepathSuffix
    )
    print("\n" + f"[@T{T} {paramstring}]: Saving Joint Clusters data as: " + savePath)

    finalDict = {f"T{T}": saveDict}

    hdf5_save(savePath, finalDict)

    plt.close("all")

    del (
        finalDict,
        saveDict,
        Djoint,
        Zjoint,
        cluster,
        clusters,
        clusterIndices,
        subsetClusterIndices,
        plotYdata,
        plotXdata,
        paths,
        whereTracers,
        d_crit,
    )
