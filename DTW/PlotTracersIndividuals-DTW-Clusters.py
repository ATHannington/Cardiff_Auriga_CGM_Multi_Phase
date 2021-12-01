"""
Author: A. T. Hannington
Created: 26/03/2020

Known Bugs:
    pandas read_csv loading data as nested dict. . Have added flattening to fix

"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

xsize = 20.0
ysize = 10.0
DPI = 250

ageUniverse = 13.77  # [Gyr]

colourmapMain = "viridis"
colourmapIndividuals = "Dark2"  # "nipy_spectral"
# Set style options

lineStyleMedian = "-"
lineWidthMedian = 2


opacityPercentiles = 0.1
lineStylePercentiles = "-."
lineWidthPercentiles = 1

# Input parameters path:
TracersParamsPath = "TracersParams.csv"

dtwJoint = True

# "rho_rhomean,dens,T,R,n_H,B,vrad,gz,L,P_thermal,P_magnetic,P_kinetic,P_tot,tcool,theat,csound,tcross,tff,tcool_tff"
ylabel = {
    "T": r"Temperature [$K$]",
    "R": r"Radius [$kpc$]",
    "n_H": r"$n_H$ [$cm^{-3}$]",
    "B": r"|B| [$\mu G$]",
    "vrad": r"Radial Velocity [$km$ $s^{-1}$]",
    "gz": r"Average Metallicity $Z/Z_{\odot}$",
    "L": r"Specific Angular Momentum[$kpc$ $km$ $s^{-1}$]",
    "P_thermal": r"$P_{Thermal} / k_B$ [$K$ $cm^{-3}$]",
    "P_magnetic": r"$P_{Magnetic} / k_B$ [$K$ $cm^{-3}$]",
    "P_kinetic": r"$P_{Kinetic} / k_B$ [$K$ $cm^{-3}$]",
    "P_tot": r"$P_{tot} = P_{thermal} + P_{magnetic} / k_B$ [$K$ $cm^{-3}$]",
    "tcool": r"Cooling Time [$Gyr$]",
    "theat": r"Heating Time [$Gyr$]",
    "tcross": r"Sound Crossing Cell Time [$Gyr$]",
    "tff": r"Free Fall Time [$Gyr$]",
    "tcool_tff": r"Cooling Time over Free Fall Time",
    "csound": r"Sound Speed",
    "rho_rhomean": r"Density over Average Universe Density",
    "dens": r"Density [$g$ $cm^{-3}$]",
}

# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

saveParams = TRACERSPARAMS["dtwParams"]
logParams = TRACERSPARAMS["dtwlogParams"]


for entry in logParams:
    ylabel[entry] = r"Log10 " + ylabel[entry]

DataSavepathSuffix = f".h5"


paramstring = "+".join(saveParams)

print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath, TRACERSPARAMS, DataSavepathSuffix)

loadPath = DataSavepath + f"_flat-wrt-time" + DataSavepathSuffix

FlatDataDict = hdf5_load(loadPath)

print("Getting Tracer Data!")
Ydata = {}
Xdata = {}
ViolinDict = {}
ClusterDict = {}
MassDict = {}

tage = []
for snap in range(
    int(TRACERSPARAMS["snapMin"]),
    min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
    1,
):
    minTemp = TRACERSPARAMS["targetTLst"][0]
    key = (f"T{minTemp}", f"{int(snap)}")

    tage.append(dataDict[key]["Lookback"][0])

tage = np.array(tage)
tage = abs(tage - ageUniverse)


# Loop over temperatures in targetTLst and grab Temperature specific subset of tracers and relevant data
for T in TRACERSPARAMS["targetTLst"]:
    print("")
    print(f"Starting T{T} analysis")
    # Select tracers from those present at data selection snapshot, snapnum

    if dtwJoint == True:
        loadPath = (
            DataSavepath
            + f"_T{T}_{paramstring}_Joint-DTW-clusters"
            + DataSavepathSuffix
        )
        print(
            "\n" + f"[@T{T} {paramstring}]: Loading Joint Clusters data : " + loadPath
        )

        dtwDict = hdf5_load(loadPath)

        Tkey = f"T{T}"
        # rangeMin = 0
        # rangeMax = len(dtwDict[Tkey]['trid'])
        # TracerNumberSelect = np.arange(start=rangeMin, stop = rangeMax, step = 1 )
        # #Take Random sample of Tracers size min(subset, len(data))
        # TracerNumberSelect = sample(TracerNumberSelect.tolist(),min(subset,rangeMax))
        #
        # # selectMin = min(subset,rangeMax)
        # # select = math.floor(float(rangeMax)/float(subset))
        # # TracerNumberSelect = TracerNumberSelect[::select]
        #
        # SelectedTracers1 = dtwDict[Tkey]['trid'][TracerNumberSelect]

    # XScatterSubDict = {}
    XSubDict = {}
    YSubDict = {}
    MassSubDict = {}
    ViolinSubDict = {}
    ClusterSubDict = {}
    # SubHaloIDSubDict = {}
    for analysisParam in saveParams:
        print("")
        print(f"Starting {analysisParam} analysis")

        if dtwJoint == False:
            if analysisParam in logParams:
                loadPath = (
                    DataSavepath
                    + f"_T{T}_log10{analysisParam}_DTW-clusters"
                    + DataSavepathSuffix
                )
                print(
                    "\n"
                    + f"[@T{T} log10{analysisParam}]: Saving Clusters data as: "
                    + loadPath
                )
            else:
                loadPath = (
                    DataSavepath
                    + f"_T{T}_{analysisParam}_DTW-clusters"
                    + DataSavepathSuffix
                )
                print(
                    "\n"
                    + f"[@T{T} {analysisParam}]: Saving Clusters data as: "
                    + loadPath
                )
            dtwDict = hdf5_load(loadPath)
            Tkey = f"T{T}"
            # rangeMin = 0
            # rangeMax = len(dtwDict[Tkey]['trid'])
            # TracerNumberSelect = np.arange(start=rangeMin, stop = rangeMax, step = 1 )
            # #Take Random sample of Tracers size min(subset, len(data))
            # TracerNumberSelect = sample(TracerNumberSelect.tolist(),min(subset,rangeMax))
            #
            # # selectMin = min(subset,rangeMax)
            # # select = math.floor(float(rangeMax)/float(subset))
            # # TracerNumberSelect = TracerNumberSelect[::select]
            #
            # SelectedTracers1 = dtwDict[Tkey]['trid'][TracerNumberSelect]
        # Loop over snaps from and gather data for the SelectedTracers1.
        #   This should be the same tracers for all time points due to the above selection, and thus data and massdata should always have the same shape.

        tmpXdata = []
        tmpClusterData = []
        if analysisParam in logParams:
            paramKey = f"log10{analysisParam}"
        else:
            paramKey = f"{analysisParam}"

        snapRange = range(
            int(TRACERSPARAMS["snapMin"]),
            min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"] + 1)),
        )
        for snap in snapRange:
            fullkey = (f"T{T}", f"{int(snap)}")
            whereGas = np.where(dataDict[fullkey]["type"] == 0)[0]
            tmpXdata.append(dataDict[fullkey]["Lookback"][0])

            # Violin Data
            # weightedData = weightedperc(data=dataDict[fullkey][analysisParam][whereGas], weights=dataDict[fullkey]['mass'][whereGas], perc=50,key='mass')
            # tmpViolinData.append(weightedData)

        # Get Data
        tmpYSubSubDict = {}
        tmpMassSubSubDict = {}

        tmpClusterData = np.unique(dtwDict[Tkey]["clusters"])
        for cluster in tmpClusterData:
            whereCluster = np.where(dtwDict[Tkey]["clusters"] == cluster)
            data = dtwDict[Tkey][paramKey][whereCluster]
            tmpYSubSubDict.update({f"{cluster}": data})

            massData = []
            for jj in range(len(snapRange)):
                whereTrid = np.where(
                    np.isin(
                        FlatDataDict[Tkey]["trid"][jj, :],
                        dtwDict[Tkey]["trid"][:, jj][whereCluster],
                    )
                )[0]
                massData.append(FlatDataDict[Tkey]["mass"][jj, :][whereTrid])
            tmpMassSubSubDict.update({f"{cluster}": np.array(massData)})

        # Append the data from this parameters to a sub dictionary
        XSubDict.update({analysisParam: np.array(tmpXdata)})
        YSubDict.update({analysisParam: tmpYSubSubDict})
        MassSubDict.update({analysisParam: tmpMassSubSubDict})
        # ViolinSubDict.update({analysisParam : np.array(tmpViolinData)})
        ClusterSubDict.update({analysisParam: np.array(tmpClusterData)})

    # Add the full list of snaps data to temperature dependent dictionary.
    Xdata.update({f"T{T}": XSubDict})
    Ydata.update({f"T{T}": YSubDict})
    MassDict.update({f"T{T}": MassSubDict})
    # ViolinDict.update({f"T{T}" : ViolinSubDict})
    ClusterDict.update({f"T{T}": ClusterSubDict})
# ==============================================================================#

# ==============================================================================#
#           PLOT!!
# ==============================================================================#
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value


#
# unboundFracStartDict = {}
# unboundFracEndDict = {}
# haloFracStartDict = {}
# haloFracEndDict = {}
# otherHaloFracStartDict = {}
# otherHaloFracEndDict = {}
# unassignedFracStartDict = {}
# unassignedFracEndDict = {}

# print("")
# for temp in Tlst:
#     print(f"T{temp} : HaloID Analyis!")
#     startkey = (f"T{temp}", f"{int(TRACERSPARAMS['snapMin'])}")
#     endkey = (f"T{temp}", f"{min(int(TRACERSPARAMS['finalSnap']),int(TRACERSPARAMS['snapMax']))}")
#     startNtracers = dataDict[startkey]['Ntracers'][0]
#     endNtracers = dataDict[endkey]['Ntracers'][0]
#
#     startSubHaloIDDataFull, _ , _ = GetIndividualCellFromTracer(Tracers=dataDict[startkey]['trid'],\
#         Parents=dataDict[startkey]['prid'],CellIDs=dataDict[startkey]['id'],SelectedTracers=dataDict[startkey]['trid'],\
#         Data=dataDict[startkey]['SubHaloID'])
#     endSubHaloIDDataFull, _ , _ = GetIndividualCellFromTracer(Tracers=dataDict[endkey]['trid'],\
#         Parents=dataDict[endkey]['prid'],CellIDs=dataDict[endkey]['id'],SelectedTracers=dataDict[endkey]['trid'],\
#         Data=dataDict[endkey]['SubHaloID'])
#     unboundFracStart = float(np.shape(np.where(startSubHaloIDDataFull==-1)[0])[0])/float(startNtracers)
#     unboundFracEnd = float(np.shape(np.where(endSubHaloIDDataFull==-1)[0])[0])/float(endNtracers)
#     haloFracStart = float(np.shape(np.where(startSubHaloIDDataFull==int(TRACERSPARAMS['haloID']))[0])[0])/float(startNtracers)
#     haloFracEnd = float(np.shape(np.where(endSubHaloIDDataFull==int(TRACERSPARAMS['haloID']))[0])[0])/float(endNtracers)
#
#     otherHaloFracStart = float(np.shape(np.where((startSubHaloIDDataFull!=int(TRACERSPARAMS['haloID']))\
#     &(startSubHaloIDDataFull!=-1)&(np.isnan(startSubHaloIDDataFull)==False))[0])[0])/float(startNtracers)
#
#     otherHaloFracEnd = float(np.shape(np.where((endSubHaloIDDataFull!=int(TRACERSPARAMS['haloID']))\
#     &(endSubHaloIDDataFull!=-1)&(np.isnan(endSubHaloIDDataFull)==False))[0])[0])/float(endNtracers)
#
#     unassignedFracStart = float(np.shape(np.where(np.isnan(startSubHaloIDDataFull)==True)[0]) [0])/float(startNtracers)
#     unassignedFracEnd = float(np.shape(np.where(np.isnan(endSubHaloIDDataFull)==True)[0])[0])/float(endNtracers)
#
#     unboundFracStartDict.update({f"T{temp}" : unboundFracStart})
#     unboundFracEndDict.update({f"T{temp}" : unboundFracEnd})
#     haloFracStartDict.update({f"T{temp}" : haloFracStart})
#     haloFracEndDict.update({f"T{temp}" : haloFracEnd})
#     otherHaloFracStartDict.update({f"T{temp}" : otherHaloFracStart})
#     otherHaloFracEndDict.update({f"T{temp}" : otherHaloFracEnd})
#     unassignedFracStartDict.update({f"T{temp}" : unassignedFracStart})
#     unassignedFracEndDict.update({f"T{temp}" : unassignedFracEnd})
# print("")

for analysisParam in saveParams:
    print("")
    print(f"Starting {analysisParam} Sub-plots!")

    nClusters = len(ClusterDict[f"T{Tlst[0]}"][analysisParam].copy())
    fig = plt.figure(constrained_layout=True, figsize=(xsize, ysize), dpi=DPI)
    gs = GridSpec(3, nClusters * len(Tlst) / 2, figure=fig)

    # Create a plot for each Temperature
    for ii in range(len(Tlst)):

        # Temperature specific load path
        plotData = Statistics_hdf5_load(
            Tlst[ii], DataSavepath, TRACERSPARAMS, DataSavepathSuffix
        )

        snapsRange = np.array(
            [
                xx
                for xx in range(
                    int(TRACERSPARAMS["snapMin"]),
                    min(
                        int(TRACERSPARAMS["snapMax"]) + 1,
                        int(TRACERSPARAMS["finalSnap"]) + 1,
                    ),
                    1,
                )
            ]
        )
        selectionSnap = np.where(snapsRange == int(TRACERSPARAMS["selectSnap"]))

        vline = tage[selectionSnap]

        # Get number of temperatures
        NTemps = float(len(Tlst))

        # Get temperature
        temp = TRACERSPARAMS["targetTLst"][ii]

        # plotXScatterdata = XScatterDict[f"T{temp}"][analysisParam].copy()
        plotYdata = Ydata[f"T{temp}"][analysisParam].copy()
        plotXdata = Xdata[f"T{temp}"][analysisParam].copy()
        MassData = MassDict[f"T{temp}"][analysisParam].copy()
        clusterData = ClusterDict[f"T{temp}"][analysisParam].copy()
        # SubHaloIDData = SubHaloIDDict[f"T{temp}"][analysisParam].astype('int16').copy()

        # uniqueSubHalo = np.unique(SubHaloIDData)
        #
        # normedSubHaloIDData = SubHaloIDData.copy()
        # for (kk,halo) in enumerate(uniqueSubHalo):
        #     whereHalo = np.where(SubHaloIDData==halo)
        #     if((halo==int(TRACERSPARAMS['haloID']))or(halo==-1)):
        #         normedSubHaloIDData[whereHalo] = halo
        #     else:
        #         normedSubHaloIDData[whereHalo] = int(TRACERSPARAMS['haloID']) + 1
        #
        # normedUniqueSubHalo = np.unique(normedSubHaloIDData)

        # Convert lookback time to universe age
        # t0 = np.nanmax(plotXdata)
        plotXdata = abs(plotXdata - ageUniverse)
        # plotXScatterdata = abs(plotXScatterdata - ageUniverse)

        # Select a Temperature specific colour from colourmap
        maxCluster = np.nanmax(np.unique(clusterData))
        cmap = matplotlib.cm.get_cmap(colourmapIndividuals)
        cmap2 = matplotlib.cm.get_cmap(colourmapMain)
        colour = cmap2(float(ii + 1) / float(len(Tlst)))
        colourTracers = [cmap(float(jj) / float(maxCluster)) for jj in clusterData]

        LO = analysisParam + "LO"
        UP = analysisParam + "UP"
        median = analysisParam + "median"

        # YDataisNOTinf = np.where(np.isinf(plotYdata)==False)
        #
        # datamin = np.nanmin(plotYdata[YDataisNOTinf])
        # datamax = np.nanmax(plotYdata[YDataisNOTinf])

        if analysisParam in logParams:
            tmp = []
            # for (ind, array) in enumerate(violinData):
            #     tmpData = np.log10(array)
            #     whereNOTnan = np.where(np.isnan(tmpData)==False)
            #     wherenan = np.where(np.isnan(tmpData)==True)
            #     tmp.append(tmpData[whereNOTnan])

            # violinData = np.array(tmp)

            # plotYdata = np.log10(plotYdata)

            for k, v in plotData.items():
                plotData.update({k: np.log10(v)})

        ##
        #   If all entries of data are nan, and thus dataset len == 0
        #   add a nan and zero array to omit violin but continue plotting
        #   without errors.
        ##
        # tmp = []
        # for dataset in violinData:
        #     if (len(dataset)==0):
        #         tmp.append(np.array([np.nan,0,np.nan]))
        #     else:
        #         tmp.append(dataset)
        #
        # violinData = tmp

        print("")
        print("Sub-Plot!")

        # UPisINF = np.where(np.isinf(plotData[UP]) == True)
        # LOisINF = np.where(np.isinf(plotData[LO]) == True)
        # medianisINF = np.where(np.isinf(plotData[median]) == True)
        #
        # print("")
        # print(f"before {median} {plotData[median][medianisINF] }")
        # plotData[UP][UPisINF] = np.array([0.])
        # plotData[median][medianisINF] = np.array([0.])
        # plotData[LO][LOisINF] = np.array([0.])
        # print(f"after {median} {plotData[median][medianisINF] }")

        currentAx = fig.add_subplot(
            gs[1, int((nClusters / 2) * ii) : int((nClusters / 2) * (ii + 1))]
        )

        currentAx.fill_between(
            tage,
            plotData[UP],
            plotData[LO],
            facecolor=colour,
            alpha=opacityPercentiles,
            interpolate=False,
        )
        currentAx.plot(
            plotXdata,
            plotData[median],
            color=colour,
            lineStyle=lineStyleMedian,
            linewidth=lineStyleMedian,
        )
        currentAx.plot(
            plotXdata,
            plotData[LO],
            color=colour,
            lineStyle=lineStylePercentiles,
            linewidth=lineWidthPercentiles,
        )
        currentAx.plot(
            plotXdata,
            plotData[UP],
            color=colour,
            lineStyle=lineStylePercentiles,
            linewidth=lineWidthPercentiles,
        )

        tmpdatamin = []
        tmpdatamax = []
        for (col, clusterID) in zip(colourTracers, clusterData):
            startcol = int((clusterID - 1) % (nClusters / 2)) * ii
            endcol = int((clusterID - 1) % (nClusters / 2)) * (ii + 1)
            row = int(math.floor((clusterID - 1) / (3)))

            tmpAx = fig.add_subplot(gs[row, startcol:endcol])
            data = plotYdata[f"{clusterID}"].T
            mass = MassData[f"{clusterID}"]
            medianData = []
            upData = []
            loData = []
            for snap in range(np.shape(data)[0]):
                medianData.append(
                    weightedperc(
                        data=data[snap, :], weights=mass[snap, :], perc=50, key="median"
                    )
                )
                upData.append(
                    weightedperc(
                        data=data[snap, :],
                        weights=mass[snap, :],
                        perc=int(TRACERSPARAMS["percentileUP"]),
                        key="up",
                    )
                )
                loData.append(
                    weightedperc(
                        data=data[snap, :],
                        weights=mass[snap, :],
                        perc=int(TRACERSPARAMS["percentileLO"]),
                        key="lo",
                    )
                )
            # currentAx.plot(plotXdata,tracer,color = col, alpha = opacity, label = f"Cluster {clusterID}")
            medianData = np.array(medianData)
            upData = np.array(upData)
            loData = np.array(loData)

            tmpAx.fill_between(
                plotXdata,
                upData,
                loData,
                facecolor=col,
                alpha=opacityPercentiles,
                interpolate=False,
            )
            tmpAx.plot(
                plotXdata,
                medianData,
                color=col,
                lineStyle=lineStyleMedian,
                linewidth=lineWidthMedian,
                label=f"Cluster {int(clusterID)}",
            )
            tmpAx.plot(
                plotXdata,
                upData,
                color=col,
                lineStyle=lineStylePercentiles,
                linewidth=lineWidthPercentiles,
            )
            tmpAx.plot(
                plotXdata,
                loData,
                color=col,
                lineStyle=lineStylePercentiles,
                linewidth=lineWidthPercentiles,
            )

            tmpdatamin.append(np.nanmin(loData))
            tmpdatamax.append(np.nanmax(upData))

            tmpAx.axvline(x=vline, c="red")

            # whereDataIsNOTnan = np.where(np.isnan(plotYdata)==False)
            # paths = np.array([plotXScatterdata, plotYdata]).T.reshape(-1,len(plotXdata),2)

            # line = currentAx.add_collection(lc)

            tmpAx.xaxis.set_minor_locator(AutoMinorLocator())
            tmpAx.yaxis.set_minor_locator(AutoMinorLocator())
            tmpAx.tick_params(which="both")

            tmpAx.set_ylabel(ylabel[analysisParam], fontsize=10)
            tmpAx.set_ylim(ymin=np.nanmin(loData), ymax=np.nanmax(upData))

            tmpAx.legend(loc="upper right")

            tmpAx.set_xlabel(r"Age of Universe [$Gyrs$]", fontsize=10)

        datamin = np.nanmin(tmpdatamin)
        datamax = np.nanmax(tmpdatamax)
        # if ((np.isnan(datamax)==True) or (np.isnan(datamin)==True)):
        #     print("NaN datamin/datamax. Skipping Entry!")
        #     continue
        #
        # if ((np.isinf(datamax)==True) or (np.isinf(datamin)==True)):
        #     print("Inf datamin/datamax. Skipping Entry!")
        #     continue

        # unboundFracStart = unboundFracStartDict[f"T{temp}"]
        # unboundFracEnd = unboundFracEndDict[f"T{temp}"]
        # haloFracStart = haloFracStartDict[f"T{temp}"]
        # haloFracEnd = haloFracEndDict[f"T{temp}"]
        # otherHaloFracStart = otherHaloFracStartDict[f"T{temp}"]
        # otherHaloFracEnd = otherHaloFracEndDict[f"T{temp}"]
        # unassignedFracStart = unassignedFracStartDict[f"T{temp}"]
        # unassignedFracEnd = unassignedFracEndDict[f"T{temp}"]

        # HaloString = f"Of Tracer Subset: \n {haloFracStart:3.3%} start in Halo {int(TRACERSPARAMS['haloID'])},"\
        # +f" {unboundFracStart:3.3%} start 'unbound',{otherHaloFracStart:3.3%} start in other Haloes, {unassignedFracStart:3.3%} start unassigned."\
        # +f"\n {haloFracEnd:3.3%} end in Halo {int(TRACERSPARAMS['haloID'])}, {unboundFracEnd:3.3%} end 'unbound',{otherHaloFracEnd:3.3%} end in other Haloes, {unassignedFracEnd:3.3%} end unassigned."
        #
        # currentAx.text(1.02, 0.5, HaloString, horizontalalignment='left',verticalalignment='center',\
        # transform=currentAx.transAxes, wrap=True,bbox=dict(facecolor='tab:gray', alpha=0.25))

        plot_label = r"$T = 10^{%3.2f} K$" % (float(temp))
        currentAx.text(
            0.05,
            0.95,
            plot_label,
            horizontalalignment="left",
            verticalalignment="center",
            transform=currentAx.transAxes,
            wrap=True,
            bbox=dict(facecolor=colour, alpha=0.125),
        )

        currentAx.transAxes

        # parts = currentAx.violinplot(violinData,positions=plotXdata,showmeans=False,showmedians=False,showextrema=False)#label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour, lineStyle=lineStyleMedian)
        #
        # for pc in parts['bodies']:
        #     pc.set_facecolor(colour)
        #     pc.set_edgecolor('black')
        #     pc.set_alpha(opacityPercentiles)
        #
        # quartile1 = []
        # medians = []
        # quartile3 = []
        # for dataset in violinData:
        #     q1,med,q3 = np.percentile(dataset, [int(TRACERSPARAMS['percentileLO']), 50, int(TRACERSPARAMS['percentileUP'])], axis=0)
        #     quartile1.append(q1)
        #     medians.append(med)
        #     quartile3.append(q3)
        #
        # sorted_violinData = []
        # for dataset in violinData:
        #     ind_sorted = np.argsort(dataset)
        #     dataset = dataset[ind_sorted]
        #     sorted_violinData.append(dataset)
        #
        # whiskers = np.array([
        #     adjacent_values(sorted_array, q1, q3)
        #     for sorted_array, q1, q3 in zip(sorted_violinData, quartile1, quartile3)])
        # whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
        #
        # currentAx.scatter(plotXdata, medians, marker='o', color='white', s=30, zorder=3)
        # currentAx.vlines(plotXdata, quartile1, quartile3, color='k', linestyle='-', lw=3)
        # currentAx.vlines(plotXdata, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

        currentAx.axvline(x=vline, c="red")

        # whereDataIsNOTnan = np.where(np.isnan(plotYdata)==False)
        # paths = np.array([plotXScatterdata, plotYdata]).T.reshape(-1,len(plotXdata),2)

        # line = currentAx.add_collection(lc)

        currentAx.xaxis.set_minor_locator(AutoMinorLocator())
        currentAx.yaxis.set_minor_locator(AutoMinorLocator())
        currentAx.tick_params(which="both")

        currentAx.set_ylabel(ylabel[analysisParam], fontsize=10)
        currentAx.set_ylim(ymin=datamin, ymax=datamax)

        currentAx.legend(loc="upper right")
        fig.suptitle(
            f"Cells Containing Tracers selected by: "
            + "\n"
            + r"$T = 10^{n \pm %05.2f} K$" % (TRACERSPARAMS["deltaT"])
            + r" and $%05.2f \leq R \leq %05.2f kpc $"
            % (TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"])
            + "\n"
            + f" and selected at {vline[0]:3.2f} Gyr"
            + f" weighted by mass",
            fontsize=12,
        )

        currentAx.set_xlabel(r"Age of Universe [$Gyrs$]", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.80)
    if dtwJoint == True:
        opslaan = (
            f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
            + analysisParam
            + f"_IndividualsMedians-DTW-Joint-Clusters.pdf"
        )
    else:
        opslaan = (
            f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
            + analysisParam
            + f"_IndividualsMedians-DTW-Clusters.pdf"
        )
    plt.savefig(opslaan, dpi=DPI, transparent=False)
    print(opslaan)
    plt.close()
