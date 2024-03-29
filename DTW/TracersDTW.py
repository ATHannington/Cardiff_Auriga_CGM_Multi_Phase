import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import itertools
import time
import torch
import torch.nn as nn

from soft_dtw_cuda import SoftDTW

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
import random
import math
from functools import reduce

# Input parameters path:
TracersParamsPath = "TracersParams.csv"
TracersMasterParamsPath = "TracersParamsMaster.csv"
SelectedHaloesPath = "TracersSelectedHaloes.csv"

batch_limit = 1e4
printpercent = 10.0

random.seed(1234)
# ------------------------------------------------------------------------------#
def DTW_prep(M):
    """
    Function to obtain unique combinations of time series.
    Will return time series index in Matrix m (0,1,2....) and unique partner (0,1,2...) ignoring (1,2)==(2,1) etc
    Returns pairs as list of tuples in iterator
    """

    elements = range(np.shape(M)[0])
    iterator = itertools.combinations(elements, r=2)

    return iterator


#
# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersMasterParamsPath)

# Load Halo Selection Data
SELECTEDHALOES, HALOPATHS = load_haloes_selected(
    HaloPathBase=TRACERSPARAMS["savepath"], SelectedHaloesPath=SelectedHaloesPath
)

DataSavepathSuffix = f".h5"

snapRange = [
    snap
    for snap in range(
        int(TRACERSPARAMS["snapMin"]),
        min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
        1,
    )
]

tmp = DataSavepath.split("/")
tmp2 = tmp[:-1] + ["DTW"] + tmp[-1:]
DataSavepath = "/".join(tmp2)

print("")
print(f"In this programme we will be saving as:")
print(DataSavepath)
print("")

dtwParams = TRACERSPARAMS["dtwParams"]
logParams = TRACERSPARAMS["dtwlogParams"]

dtwSubset = int(TRACERSPARAMS["dtwSubset"])

loadParams = dtwParams + TRACERSPARAMS["saveEssentials"]

print("Load Time Flattened Data!")
dataDict, saveParams = multi_halo_merge_flat_wrt_time(
    SELECTEDHALOES,
    HALOPATHS,
    DataSavepathSuffix,
    snapRange,
    Tlst,
    TracersParamsPath,
    loadParams=loadParams,
    dtwSubset=dtwSubset,
)
print("Done!")

print(torch.__version__)

cuda = torch.device("cuda")

n_gpus = torch.cuda.device_count()
print(f"Running on {n_gpus} GPUs")

multi_batch_limit = int(n_gpus * batch_limit)
n_pairs = int(batch_limit**3)
while True:
    last_batch_size = n_pairs % multi_batch_limit
    if (last_batch_size > 1) and (multi_batch_limit % 32 == 0):
        break
    else:
        multi_batch_limit -= 1

print("last_batch_size", last_batch_size)
print("multi_batch_limit", multi_batch_limit)
analysisDict = {}
tridDict = {}
pridDict = {}


for T in Tlst:
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        for analysisParam in dtwParams:
            key = (f"T{T}", f"{rin}R{rout}")
            if analysisParam == "subhalo":
                dat = dataDict[key][analysisParam]
                dataDict[key][analysisParam][np.where(np.isnan(dat) == True)[0]] = -2
            if analysisParam in logParams:
                newkey = (f"T{T}", f"{rin}R{rout}", f"log10{analysisParam}")
                analysisDict.update(
                    {newkey: np.log10(dataDict[key][analysisParam].T.copy())}
                )
            else:
                newkey = (f"T{T}", f"{rin}R{rout}", f"{analysisParam}")
                analysisDict.update({newkey: dataDict[key][analysisParam].T.copy()})

            tridDict.update({newkey: dataDict[key]["trid"]})
            pridDict.update({newkey: dataDict[key]["prid"]})

analysisDict, whereDict = delete_nan_inf_axis(analysisDict, axis=1)

del dataDict

for key, value in tridDict.items():
    if value is not None:
        whereEntry = np.where(whereDict[key])[0]
        tridDict.update({key: tridDict[key][:, whereEntry]})

print("Gathering Selection Specific Intersect of Trids")
for T in Tlst:
    print(f"\n T{T} ")
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"\n {rin}R{rout} ")
        trid_list = []
        for analysisParam in dtwParams:
            # print(f"T{T} {rin}R{rout} {analysisParam}")
            if analysisParam in logParams:
                key = (f"T{T}", f"{rin}R{rout}", f"log10{analysisParam}")
            else:
                key = (f"T{T}", f"{rin}R{rout}", f"{analysisParam}")
            trid_list.append(tridDict[key])

        intersect = reduce(np.intersect1d, trid_list)
        for analysisParam in dtwParams:
            print(f"T{T} {rin}R{rout} {analysisParam} intersect!")
            if analysisParam in logParams:
                key = (f"T{T}", f"{rin}R{rout}", f"log10{analysisParam}")
            else:
                key = (f"T{T}", f"{rin}R{rout}", f"{analysisParam}")

            whereEntry = np.where(np.isin(tridDict[key][0,:], intersect) == True)[0]

            data = analysisDict[key][whereEntry,:]
            print(f"Final shape == {np.shape(data)}")
            analysisDict.update({key: data})
            tridDict.update({key: tridDict[key][:, whereEntry]})
            pridDict.update({key: pridDict[key][:, whereEntry]})

for T in Tlst:
    print(f"\n ***Starting T{T} Analyis!***")
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"\n *  {rin}R{rout}!  *")
        for analysisParam in dtwParams:
            print(f"Starting T{T} {rin}R{rout} {analysisParam} Analysis!")

            print("Load M matrix...")
            if analysisParam in logParams:
                key = (f"T{T}", f"{rin}R{rout}", f"log10{analysisParam}")
            else:
                key = (f"T{T}", f"{rin}R{rout}", f"{analysisParam}")

            M = analysisDict[key]
            tridData = tridDict[key].T
            pridData = pridDict[key].T

            print("...Loaded M matrix!")

            print(f"Shape of M : {np.shape(M)}")
            print(f"Shape of tridData : {np.shape(tridData)}")
            print(f"Shape of pridData : {np.shape(pridData)}")

            print("Prep iterator!")
            iterator = DTW_prep(M)

            print("Load DTW instance!")
            dtw = SoftDTW(use_cuda=True, gamma=1e-10, normalize=True)

            # print("Send M to Mtens cuda!")
            # Mtens = torch.tensor(M, device=cuda).view(np.shape(M)[0], np.shape(M)[1], 1)
            print("Make blank list!")
            out_device = torch.empty((0), dtype=torch.float64, device=cuda)
            print("Let's do the Time Warp...")
            start = time.time()

            percent = 0.0
            start = time.time()
            xlist = []
            ylist = []

            nn = np.shape(M)[0]
            n_xy = np.prod((np.arange(nn - 2, nn) + 1)) // 2
            endofiter = False
            ii = 0
            while endofiter == False:
                select = np.array(list(itertools.islice(iterator, multi_batch_limit)))
                if select.size == 0:
                    endofiter == True
                    break
                ii += select.shape[0]
                xlist = select[:, 0].tolist()
                ylist = select[:, 1].tolist()
                percentage = float(ii) / float(n_xy) * 100.0
                if percentage >= percent:
                    print(f"{percentage:2.0f}%")
                    percent += printpercent
                if len(xlist) >= multi_batch_limit:
                    # print("x y!")
                    x = torch.tensor(M[xlist], device=cuda).view(
                        len(xlist), np.shape(M)[1], 1
                    )
                    y = torch.tensor(M[ylist], device=cuda).view(
                        len(ylist), np.shape(M)[1], 1
                    )
                    # print("Time Warping!")
                    out_tmp = dtw(x, y)
                    # print("Done!")
                    out_device = torch.cat((out_device, out_tmp), 0)
                    xlist = []
                    ylist = []

            # for ii, (xx, yy) in enumerate(iterator):
            #     xlist.append(xx)
            #     ylist.append(yy)
            #     percentage = float(ii) / float(n_xy) * 100.0
            #     if percentage >= percent:
            #         print(f"{percentage:2.0f}%")
            #         percent += printpercent
            #     if len(xlist) >= multi_batch_limit:
            #         # print("x y!")
            #         x = torch.tensor(M[xlist], device=cuda).view(len(xlist), np.shape(M)[1], 1)
            #         y = torch.tensor(M[ylist], device=cuda).view(len(ylist), np.shape(M)[1], 1)
            #         # print("Time Warping!")
            #         out_tmp = dtw(x, y)
            #         # print("Done!")
            #         out_device = torch.cat((out_device, out_tmp), 0)
            #         xlist = []
            #         ylist = []

            print("Finishing up...")
            x = torch.tensor(M[xlist], device=cuda).view(len(xlist), np.shape(M)[1], 1)
            y = torch.tensor(M[ylist], device=cuda).view(len(ylist), np.shape(M)[1], 1)
            out_tmp = dtw(x, y)
            out_device = torch.cat((out_device, out_tmp), 0)

            del xlist, ylist, x, y, out_tmp

            out = out_device.cpu().detach().numpy()

            end = time.time()
            elapsed = end - start
            print(f"Elapsed time in DTW = {elapsed}s")

            saveSubDict = {
                "distance_matrix": out,
                "trid": tridData,
                "prid": pridData,
                "data": M,
            }
            saveDict = {key: saveSubDict}

            if analysisParam in logParams:
                savePath = (
                    DataSavepath
                    + f"_T{T}_{rin}R{rout}_log10{analysisParam}_DTW-distance"
                    + DataSavepathSuffix
                )
                print(
                    "\n"
                    + f"[@T{T} {rin}R{rout} log10{analysisParam}]: Saving Distance Matrix + Sampled Raw Data as: "
                    + savePath
                )

            else:
                savePath = (
                    DataSavepath
                    + f"_T{T}_{rin}R{rout}_{analysisParam}_DTW-distance"
                    + DataSavepathSuffix
                )
                print(
                    "\n"
                    + f"[@T{T} {rin}R{rout} {analysisParam}]: Saving Distance Matrix + Sampled Raw Data as: "
                    + savePath
                )

            hdf5_save(savePath, saveDict)
