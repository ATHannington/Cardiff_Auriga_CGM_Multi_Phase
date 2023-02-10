import math
from random import sample
from Tracers_Subroutines import *
import h5py
from gadget_subfind import *
from gadget import *
import const as c
import sys
from functools import reduce
import time
from itertools import combinations
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters


# # Input parameters path:
TracersParamsPath = "TracersParams.csv"
TracersMasterParamsPath = "TracersParamsMaster.csv"
SelectedHaloesPath = "TracersSelectedHaloes.csv"

sort_level = 2
maxmimally_distinct_bool = True

method = "average"
DPI = 50
xsize = 50
ysize = 20
colourmapIndividuals = "nipy_spectral"
colour = "tab:gray"
rgbcolour = mcolors.to_rgb(colour)
opacity = 0.5

subset = 100
# ==============================================================================#


def get_d_crit(Z, sort_level, maxmimally_distinct_bool):
    distance_levels = np.array(abs(Z[:-1, 2] - Z[1:, 2]))

    if maxmimally_distinct_bool == True:
        sorted_distance_levels_index = np.argsort(distance_levels)

        level = -1 - sort_level

        biggest_diff_loc = sorted_distance_levels_index[level]

        d_crit = Z[biggest_diff_loc, 2] + \
            (0.01 * distance_levels[biggest_diff_loc])
    else:
        level = -1 - sort_level
        d_crit = Z[level, 2] - 1e-5 * Z[level - 1, 2]

    print(f"d_crit = {d_crit:0.010f}")
    return d_crit


# ==============================================================================#


# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(
    TracersMasterParamsPath)

# Load Halo Selection Data
SELECTEDHALOES, HALOPATHS = load_haloes_selected(
    HaloPathBase=TRACERSPARAMS["savepath"], SelectedHaloesPath=SelectedHaloesPath
)

DataSavepathSuffix = f".h5"

tmp = DataSavepath.split("/")
tmp2 = tmp[:-1] + ["DTW"] + tmp[-1:]
DataSavepath = "/".join(tmp2)

print("")
print(f"In this programme we will be saving as:")
print(DataSavepath)
print("")
snapRange = [
    snap
    for snap in range(
        int(TRACERSPARAMS["snapMin"]),
        min(int(TRACERSPARAMS["snapMax"] + 1),
            int(TRACERSPARAMS["finalSnap"]) + 1),
        1,
    )
]

dtwParams = TRACERSPARAMS["dtwParams"]
logParams = TRACERSPARAMS["dtwlogParams"]

dtwSubset = int(TRACERSPARAMS["dtwSubset"])

loadParams = dtwParams + TRACERSPARAMS["saveEssentials"]

print("Load Non Time Flattened Data 1st Halo ONLY!")
mergedDict, saveParams = multi_halo_merge(
    SELECTEDHALOES[:1],
    HALOPATHS[:1],
    DataSavepathSuffix,
    snapRange,
    Tlst,
    TracersParamsPath,
)
print("Done!")

selectionSnap = np.array(snapRange)[
    np.where(np.array(snapRange) == int(TRACERSPARAMS["selectSnap"]))[0]
]
selectTimeKey = (
    f"T{Tlst[0]}",
    f"{TRACERSPARAMS['Rinner'][0]}R{TRACERSPARAMS['Router'][0]}",
    f"{int(selectionSnap)}",
)
selectTime = abs(mergedDict[selectTimeKey]["Lookback"][0])

tlookback = []
for snap in range(
    int(TRACERSPARAMS["snapMin"]),
    min(int(TRACERSPARAMS["snapMax"] + 1),
        int(TRACERSPARAMS["finalSnap"]) + 1),
    1,
):
    minTemp = TRACERSPARAMS["targetTLst"][0]
    minrin = TRACERSPARAMS["Rinner"][0]
    minrout = TRACERSPARAMS["Router"][0]
    key = (f"T{minTemp}", f"{minrin}R{minrout}", f"{int(snap)}")

    tlookback.append(mergedDict[key]["Lookback"][0])

tlookback = np.array(tlookback)

del mergedDict

for T in Tlst:
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"\n *  {rin}R{rout}!  *")
        dtw_MDict = {}
        dtw_DDict = {}
        dtw_PridDict = {}
        dtw_TridDict = {}
        for analysisParam in dtwParams:
            if analysisParam in logParams:
                loadKey = (f"T{T}", f"{rin}R{rout}", f"log10{analysisParam}")
                loadPath = (
                    DataSavepath
                    + f"_T{T}_{rin}R{rout}_log10{analysisParam}_DTW-distance"
                    + DataSavepathSuffix
                )
                print(
                    "\n"
                    + f"[@T{T} {rin}R{rout} log10{analysisParam}]: Loading Distance Matrix data as: "
                    + loadPath
                )
            else:
                loadKey = (f"T{T}", f"{rin}R{rout}", f"{analysisParam}")
                loadPath = (
                    DataSavepath
                    + f"_T{T}_{rin}R{rout}_{analysisParam}_DTW-distance"
                    + DataSavepathSuffix
                )
                print(
                    "\n"
                    + f"[@T{T} {rin}R{rout} {analysisParam}]: Loading Distance Matrix data as: "
                    + loadPath
                )

            dtwDict = hdf5_load(loadPath)

            D = dtwDict[loadKey]["distance_matrix"]

            M = dtwDict[loadKey]["data"]

            if analysisParam in logParams:
                dtw_MDict.update({f"log10{analysisParam}": M})
                dtw_DDict.update({f"log10{analysisParam}": D})
                dtw_PridDict.update(
                    {f"log10{analysisParam}": dtwDict[loadKey]["prid"]})
                dtw_TridDict.update(
                    {f"log10{analysisParam}": dtwDict[loadKey]["trid"]})
            else:
                dtw_MDict.update({f"{analysisParam}": M})
                dtw_DDict.update({f"{analysisParam}": D})
                dtw_PridDict.update(
                    {f"{analysisParam}": dtwDict[loadKey]["prid"]})
                dtw_TridDict.update(
                    {f"{analysisParam}": dtwDict[loadKey]["trid"]})

        paramstring = "+".join(dtwParams)
        plt.close("all")

        xData = tlookback

        # print(f"Get intersection of trids!")
        # dtw_TridDictkeys = list(dtw_TridDict.keys())
        # trid_list = []
        # for entry in dtw_TridDict.values():
        #     trid_list.append(entry[:, 0])

        # intersect = reduce(np.intersect1d,trid_list)

        # intersectDict = {}
        # notInDict = {}
        # for ii in range(0,len(trid_list)):
        #     _, _, new_ind = np.intersect1d(intersect,trid_list[ii],return_indices=True)
        #     notIn = np.where(np.isin(trid_list[ii],intersect)==False)[0]
        #     intersectDict.update({dtw_TridDictkeys[ii]: new_ind})
        #     notInDict.update({dtw_TridDictkeys[ii]: notIn})

        # MDict_nn = {}
        # for analysisParam in dtwParams:
        #     if analysisParam in logParams:
        #         key = f"log10{analysisParam}"
        #     else:
        #         key = f"{analysisParam}"
        #     MDict_nn.update({key: np.shape(dtw_MDict[key])[0]})

        #     not_in_Dict.update({key: trids[np.where(np.isin(trids,entry)==False)[0]]})
        #     dtw_MDict.update({key: dtw_MDict[key][a_ind]})
        #     intersectDict.update({key: a_ind})

        # intersectDictList = list(intersectDict.values())
        # oldIntersect = intersectDictList[0]
        # for key, value in intersectDict.items():
        #     assert np.shape(value) == np.shape(oldIntersect)

        # Normalise and then sum the distance vectors for each dtwParams
        print("Djoint!")
        kk = 0
        valueShapeOld = None
        for key, value in dtw_DDict.items():
            print(f"{key}")
            entry = value.copy()
            # whereTracers = intersectDict[key]
            # notIn = notInDict[key]
            # print(f"Shape whereTracers {np.shape(whereTracers)}")
            # print(f"Shape notIn {np.shape(notIn)}")
            # nn = MDict_nn[key]
            # d_size = np.prod((np.arange(nn-2, nn)+1))//2
            # # drange = range(0,d_size,1)
            #
            # assert d_size == value.shape[0], f"[@Djoint]: d_size != value.shape[0] ! {d_size} != {value.shape[0]} ! Check logic..."
            #
            # assert whereTracers.shape[0] <= value.shape[0], f"[@Djoint]: whereTracers.shape[0] > value.shape[0] ! {whereTracers.shape[0]} > {value.shape[0]} ! should be <= ! Check logic..."
            #
            # # chunkrange = range(0,d_size,nchunks)
            # # drop_indices =  [np.ravel_multi_index([[inter],chunk],(nn,nn)).tolist() for inter in intersectDict[key] for chunk in [drange[ii:jj] for (ii,jj) in zip(list(chunkrange)[:-1],list(chunkrange)[1:]) ] + [drange[chunkrange[-1]:]] ]
            # #
            # # offset = [np.prod((np.arange(nn-int(ii)-2, nn-int(ii))+1))//2 for ii in whereTracers]
            # #
            # # drop_indices = [offset + (int(jj) - 1) for jj in range(0,nn)]
            # entry = value.copy()
            # offset = lambda ii,nn,d_size: d_size - np.prod(np.arange(nn-int(ii)-2, nn-int(ii))+1)//2
            # drop_indices = lambda offset,entry,where,nn,d_size: np.delete(entry,[offset(ii,nn,d_size) + jj for ii in where for jj in range(0,nn)],axis=0)
            #
            # entry = drop_indices(offset,entry,notIn,nn,d_size)
            # # cols = list(range(1,nn+1,1))
            # # drop_indices_func = lambda ni: np.delete(value, [ni*col for col in cols], axis=0)
            # # drop_indices_func(notIn)
            #
            # if valueShapeOld is not None:
            #     valueShapeNew = entry.shape[0]
            #     assert valueShapeNew == valueShapeOld, f"[@Djoint]: valueShapeNew != valueShapeOld ! {valueShapeNew} != {valueShapeOld} ! Check logic..."
            #     valueShapeOld = valueShapeNew.copy()
            # else:
            #     valueShapeOld = entry.shape[0]

            maxD = np.nanmax(entry)
            entry = entry / maxD
            if kk == 0:
                Djoint = np.zeros(shape=np.shape(entry))
            Djoint += entry
            kk += 1

        # Djoint = squareform(Djoint)

        print("Joint Linkage! This may take a while...")
        Zjoint = linkage(Djoint, method=method)

        dendo_plot = plt.figure(figsize=(xsize, ysize))
        if analysisParam in logParams:
            plt.title(
                f'T{T} {rin}R{rout} log10{paramstring} Hierarchical Clustering Dendrogram using "{method}" method'
            )
        else:
            plt.title(
                f'T{T} {rin}R{rout} {paramstring} Hierarchical Clustering Dendrogram using "{method}" method'
            )
        plt.xlabel("Sample Index")
        plt.ylabel("Distance")

        print("Joint dendrogram! This may take a while...")

        ddata = dendrogram(Zjoint, color_threshold=0.0)

        # prefixList = TRACERSPARAMS["savepath"].split("/")
        # prefixList = prefixList[(-4):-2]
        # prefix = "-".join(prefixList)

        if analysisParam in logParams:
            opslaan = (
                f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                + f"_T{T}_{rin}R{rout}_log10{paramstring}"
                + f"_Joint-Dendrogram.png"
            )
        else:
            opslaan = (
                f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                + f"_T{T}_{rin}R{rout}_{paramstring}"
                + f"_Joint-Dendrogram.png"
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
        #
        # print("Joint clusters!")
        # for clusterID in uniqueClusters:
        #     for key, value in dtw_MDict.items():
        #         cluster = value[np.where(clusters == clusterID)]
        #         ymin = np.nanmin(value)
        #         ymax = np.nanmax(value)
        #
        #         clusterIndices = [xx for xx in range(len(cluster))]
        #         subsetClusterIndices = sample(clusterIndices, min(subset, len(cluster)))
        #         plotYdata = cluster[subsetClusterIndices]
        #
        #         cluster_plot, ax = plt.subplots()
        #         ax.tick_params(top=True, right=True)
        #         if analysisParam in logParams:
        #             plt.title(
        #                 f'Cluster {clusterID} for {T} {rin}R{rout} log10{paramstring} Hierarchical Clustering using "{method}" method'
        #             )
        #             plt.xlabel("Lookback [Gyr]")
        #             plt.ylabel(f"Log10{key}")
        #         else:
        #             plt.title(
        #                 f'Cluster {clusterID} for {T} {rin}R{rout} {paramstring} Hierarchical Clustering using "{method}" method'
        #             )
        #             plt.xlabel("Lookback [Gyr]")
        #             plt.ylabel(f"{key}")
        #
        #         plotXdata = np.array([xData for path in range(len(plotYdata))])
        #
        #         paths = np.array([plotXdata.T, plotYdata.T]).T.reshape(
        #             -1, len(xData), 2
        #         )
        #
        #         lc = LineCollection(paths, color=colour, alpha=opacity)
        #         line = ax.add_collection(lc)
        #         ax.autoscale()
        #         ax.set_xlim(np.nanmin(xData), np.nanmax(xData))
        #         ax.set_ylim(ymin, ymax)
        #         if analysisParam in logParams:
        #             opslaan2 = (
        #                 f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
        #                 + f"_Cluster{clusterID}_T{T}_{rin}R{rout}_log10{key}_Joint-{paramstring}"
        #                 + f"_Joint-Clustered-Individuals.pdf"
        #             )
        #         else:
        #             opslaan2 = (
        #                 f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
        #                 + f"_Cluster{clusterID}_T{T}_{rin}R{rout}_{key}_Joint-{paramstring}"
        #                 + f"_Joint-Clustered-Individuals.pdf"
        #             )
        #
        #         plt.savefig(opslaan2, dpi=DPI, transparent=False)
        #         print(opslaan2)

        saveDict = {}
        saveDict.update({"clusters": clusters})
        saveDict.update({"prid": dtw_PridDict[dtwParams[-1]]})
        saveDict.update({"trid": dtw_TridDict[dtwParams[-1]]})
        saveDict.update({"d_crit": np.array([d_crit])})
        saveDict.update(
            {"maxmimally_distinct_bool": np.array([maxmimally_distinct_bool])}
        )
        saveDict.update({"sort_level": np.array([sort_level])})

        for param in dtwParams:
            if param in logParams:
                saveDict.update({f"log10{param}": dtw_MDict[f"log10{param}"]})
            else:
                saveDict.update({f"{param}": dtw_MDict[f"{param}"]})

        savePath = (
            DataSavepath
            + f"_T{T}_{rin}R{rout}_{paramstring}_Joint-DTW-clusters"
            + DataSavepathSuffix
        )
        print(
            "\n"
            + f"[@T{T} {rin}R{rout} {paramstring}]: Saving Joint Clusters data as: "
            + savePath
        )

        finalDict = {(f"T{T}", f"{rin}R{rout}"): saveDict}

        hdf5_save(savePath, finalDict)

        plt.close("all")

        del (
            finalDict,
            saveDict,
            Djoint,
            Zjoint,
            # cluster,
            clusters,
            # clusterIndices,
            # subsetClusterIndices,
            # plotYdata,
            # plotXdata,
            # paths,
            # whereTracers,
            d_crit,
        )
