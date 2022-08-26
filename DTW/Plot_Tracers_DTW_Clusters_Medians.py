"""
Author: A. T. Hannington
Created: 11/02/2022

Known Bugs:
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math
import numpy as np
import pandas as pd
import matplotlib

# # Input parameters path:
TracersParamsPath = "TracersParams.csv"
TracersMasterParamsPath = "TracersParamsMaster.csv"
SelectedHaloesPath = "TracersSelectedHaloes.csv"

fontsize = 13
fontsizeTitle = 14
xsize=7.0
ysize=8.0
tmpxsize = xsize + 2.0
DPI = 200

# Set style options
opacityPercentiles = 0.15
lineStyleMedian = "solid"
lineStylePercentiles = "-."


colourmapMain = "plasma"

# ==============================================================================#


# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersMasterParamsPath)

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
        min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
        1,
    )
]

dtwParams = TRACERSPARAMS["dtwParams"]
logParams = TRACERSPARAMS["dtwlogParams"]

dtwSubset = int(TRACERSPARAMS["dtwSubset"])

loadParams = dtwParams + TRACERSPARAMS["saveEssentials"]

ylabel = {
    "T": r"Temperature [K]",
    "R": r"Radius [kpc]",
    "n_H": r"n$_H$ [cm$^{-3}$]",
    "B": r"|B| [$ \mu $G]",
    "vrad": r"Radial Velocity [km s$^{-1}$]",
    "gz": r"Average Metallicity Z/Z$_{\odot}$",
    "L": r"Specific Angular Momentum[kpc km s$^{-1}$]",
    "P_thermal": r"P_${Thermal}$ / k$_B$ [K cm$^{-3}$]",
    "P_magnetic": r"P$_{Magnetic}$ / k$_B$ [K cm$^{-3}$]",
    "P_kinetic": r"P$_{Kinetic}$ / k$_B$ [K cm$^{-3}$]",
    "P_tot": r"P$_{tot}$ = (P$_{thermal}$ + P$_{magnetic}$)/ k$_B$ [K cm$^{-3}$]",
    "Pthermal_Pmagnetic": r"P$_{thermal}$/P$_{magnetic}$",
    "tcool": r"Cooling Time [Gyr]",
    "theat": r"Heating Time [Gyr]",
    "tcross": r"Sound Crossing Cell Time [Gyr]",
    "tff": r"Free Fall Time [Gyr]",
    "tcool_tff": r"t$_{Cool}$/t$_{FreeFall}$",
    "csound": r"Sound Speed [km s$^{-1}$]",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "dens": r"Density [g cm$^{-3}$]",
    "ndens": r"Number density [cm$^{-3}$]",
    "mass": r"Log10 Mass per pixel [M/M$_{\odot}$]",
    "subhalo" : "Subhalo Number"
}

for entry in logParams:
    ylabel[entry] = r"$Log_{10}$" + ylabel[entry]

# ==============================================================================#

saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

snapRange = [
    snap
    for snap in range(
        int(TRACERSPARAMS["snapMin"]),
        min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
        1,
    )
]


# ==============================================================================#
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
    min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
    1,
):
    minTemp = TRACERSPARAMS["targetTLst"][0]
    minrin = TRACERSPARAMS["Rinner"][0]
    minrout = TRACERSPARAMS["Router"][0]
    key = (f"T{minTemp}", f"{minrin}R{minrout}", f"{int(snap)}")

    tlookback.append(mergedDict[key]["Lookback"][0])

tlookback = np.array(tlookback)

paramstring = "+".join(dtwParams)

for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
    for analysisParam in dtwParams:
        analysisDict = {}
        fig, ax = plt.subplots(
            nrows=len(Tlst),
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(tmpxsize, ysize),
            dpi=DPI,
        )
        yminlist = []
        ymaxlist = []
        labelList = []
        for ii, T in enumerate(Tlst):
            # =============================================================================#
            #                   Load Clusters Data                                       #
            # =============================================================================#
            loadPath = (
                DataSavepath
                + f"_T{T}_{rin}R{rout}_{paramstring}_Joint-DTW-clusters"
                + DataSavepathSuffix
            )
            tmp = hdf5_load(loadPath)
            dtwDict = tmp[(f"T{T}", f"{rin}R{rout}")]

            if analysisParam in logParams:
                analysisDict.update(
                    {f"log10{analysisParam}": dtwDict[f"log10{analysisParam}"].T}
                )
            else:
                analysisDict.update({f"{analysisParam}": dtwDict[f"{analysisParam}"].T})

            clusters = dtwDict["clusters"]
            # saveDict.update({"prid": pridData})
            # saveDict.update({"trid": tridData})
            # saveDict.update({"d_crit": np.array([d_crit])})
            # saveDict.update({"maxmimally_distinct_bool": np.array([maxmimally_distinct_bool])})
            # saveDict.update({"sort_level": np.array([sort_level])})
            # ============================================================================#
            #                   Cluster by cluster analysis!                              # #=============================================================================#

            # Select a Temperature specific colour from colourmapMain
            uniqueClusters = np.unique(clusters)

            clusterDict = {}
            print("Joint clusters!")
            for jj, clusterID in enumerate(uniqueClusters):
                whereInCluster = np.where(clusters == clusterID)[0]

                for key, value in analysisDict.items():
                    clusterDict.update({key: value[:, whereInCluster]})
                # =============================================================================#
                #                     Stats!                                                  # ==============================================================================#
                tmp = {(f"T{T}", f"{rin}R{rout}"): clusterDict}
                statsData = {}
                for snap in snapRange:
                    selectKey = (f"T{Tlst[ii]}", f"{rin}R{rout}")
                    timeIndex = np.where(np.array(snapRange) == snap)[0]
                    # print(f"Taking {snap} temporal Subset...")
                    timeDat = {}
                    for param, values in tmp[selectKey].items():
                        if np.shape(np.shape(values))[0] > 1:
                            timeDat.update({param: values[timeIndex].flatten()})
                        else:
                            timeDat.update({param: values})
                    # print(f"...done!")
                    # print(f"Calculating {snap} Statistics!")
                    dat = calculate_statistics(
                        timeDat,
                        TRACERSPARAMS=TRACERSPARAMS,
                        saveParams=list(clusterDict.keys())
                    )
                    # Fix values to arrays to remove concat error of 0D arrays
                    for k, val in dat.items():
                        dat[k] = np.array([val]).flatten()

                    if selectKey in list(statsData.keys()):
                        for subkey, vals in dat.items():
                            if subkey in list(statsData[selectKey].keys()):

                                statsData[selectKey][subkey] = np.concatenate(
                                    (statsData[selectKey][subkey], dat[subkey]), axis=0
                                )
                            else:
                                statsData[selectKey].update({subkey: dat[subkey]})
                    else:
                        statsData.update({selectKey: dat})

                # ============================================================================#
                #                   PLOTTING!                                                 # #=============================================================================#
                selectionSnap = np.where(
                    np.array(snapRange) == int(TRACERSPARAMS["selectSnap"])
                )

                vline = tlookback[selectionSnap]

                # Get number of temperatures
                NTemps = float(len(Tlst))

                # Get temperature
                temp = TRACERSPARAMS["targetTLst"][ii]

                selectKey = (f"T{Tlst[ii]}", f"{rin}R{rout}")
                plotData = statsData[selectKey].copy()

                # Get a colour for median and percentiles for a given temperature
                #   Have fiddled to move colours away from extremes of the colormap
                cmap = matplotlib.cm.get_cmap(colourmapMain)
                colour = cmap(float(jj) / float(len(uniqueClusters)))

                if analysisParam in logParams:
                    loadPercentilesTypes = [
                        "log10" + analysisParam + "_" + str(percentile) + "%"
                        for percentile in TRACERSPARAMS["percentiles"]
                    ]
                    LO = (
                        "log10"
                        + analysisParam
                        + "_"
                        + str(min(TRACERSPARAMS["percentiles"]))
                        + "%"
                    )
                    UP = (
                        "log10"
                        + analysisParam
                        + "_"
                        + str(max(TRACERSPARAMS["percentiles"]))
                        + "%"
                    )
                    median = "log10" + analysisParam + "_" + "50.00%"
                else:
                    loadPercentilesTypes = [
                        analysisParam + "_" + str(percentile) + "%"
                        for percentile in TRACERSPARAMS["percentiles"]
                    ]
                    LO = (
                        analysisParam
                        + "_"
                        + str(min(TRACERSPARAMS["percentiles"]))
                        + "%"
                    )
                    UP = (
                        analysisParam
                        + "_"
                        + str(max(TRACERSPARAMS["percentiles"]))
                        + "%"
                    )
                    median = analysisParam + "_" + "50.00%"

                # if analysisParam in logParams:
                #     for k, v in plotData.items():
                #         plotData.update({k: np.log10(v)})

                ymin = np.nanmin(plotData[LO])
                ymax = np.nanmax(plotData[UP])
                yminlist.append(ymin)
                ymaxlist.append(ymax)

                if (
                    (np.isinf(ymin) == True)
                    or (np.isinf(ymax) == True)
                    or (np.isnan(ymin) == True)
                    or (np.isnan(ymax) == True)
                ):
                    print("Data All Inf/NaN! Skipping entry!")
                    continue
                print("")
                print("Sub-Plot!")

                if len(Tlst) == 1:
                    currentAx = ax
                else:
                    currentAx = ax[ii]

                midPercentile = math.floor(len(loadPercentilesTypes) / 2.0)

                # edited to only take 1 sigma percentiles to minimise plot noise
                percentilesPairs = zip(
                    loadPercentilesTypes[:midPercentile],
                    loadPercentilesTypes[midPercentile + 1 :],
                )

                for (LO, UP) in percentilesPairs:
                    currentAx.fill_between(
                        tlookback,
                        plotData[UP],
                        plotData[LO],
                        facecolor=colour,
                        alpha=opacityPercentiles,
                        interpolate=False,
                    )
                    # currentAx.plot(
                    #     tlookback,
                    #     plotData[UP],
                    #     color=colour,
                    #     lineStyle=lineStylePercentiles,
                    # )
                    # currentAx.plot(
                    #     tlookback,
                    #     plotData[LO],
                    #     color=colour,
                    #     lineStyle=lineStylePercentiles,
                    # )
                currentAx.plot(
                    tlookback,
                    plotData[median],
                    label=f"{int(jj)} (n={(whereInCluster.shape[0]/value.shape[1]):.3%})",
                    color=colour,
                    lineStyle=lineStyleMedian,
                )

            currentAx.axvline(x=vline, c="red")

            currentAx.xaxis.set_minor_locator(AutoMinorLocator())
            currentAx.yaxis.set_minor_locator(AutoMinorLocator())
            currentAx.tick_params(which="both")
            currentAx.legend(
                loc="center right",
                facecolor="white",
                framealpha=1,
                bbox_to_anchor = (1.275,0.5)
            )

            currentAx.set_title(
                r"$ 10^{%03.2f \pm %3.2f} K $ Tracers Data"
                % (float(T), TRACERSPARAMS["deltaT"]),
                fontsize=fontsize,
            )
            # fig.suptitle(
            #     f"Cells Containing Tracers selected by: "
            #     + "\n"
            #     + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
            #     + r" and $%3.0f \leq R \leq %3.0f $ kpc " % (rin, rout)
            #     + "\n"
            #     + f" and selected at {vline[0]:3.2f} Gyr"
            #     + "\n"
            #     + f"Clustered by Dynamic Time Warping",
            #     fontsize=fontsizeTitle,
            # )

        # Only give 1 x-axis a label, as they sharex
        if len(Tlst) == 1:
            axis0 = ax
            midax = ax
        else:
            axis0 = ax[len(Tlst) - 1]
            midax = ax[(len(Tlst) - 1) // 2]

        axis0.set_xlabel("Lookback Time [Gyrs]", fontsize=fontsize)
        midax.set_ylabel(ylabel[analysisParam], fontsize=fontsize)
        finalymin = np.nanmin(yminlist)
        finalymax = np.nanmax(ymaxlist)
        if (
            (np.isinf(finalymin) == True)
            or (np.isinf(finalymax) == True)
            or (np.isnan(finalymin) == True)
            or (np.isnan(finalymax) == True)
        ):
            print("Data All Inf/NaN! Skipping entry!")
            continue
        finalymin = math.floor(finalymin)
        finalymax = math.ceil(finalymax)
        custom_ylim = (finalymin, finalymax)
        plt.setp(
            ax,
            ylim=custom_ylim,
            xlim=(round(max(tlookback), 1), round(min(tlookback), 1)),
        )

        # legendList = np.unique(np.array(labelList))
        # legendList = legendList.tolist()
        # legendPatches = []
        # for val in legendList:
        #     kk = int(val.split(" ")[-1])
        #     cmap = matplotlib.cm.get_cmap(colourmapMain)
        #     colour = cmap(float(kk) / float(len(uniqueClusters)))
        #     plot_patch = matplotlib.patches.Patch(color=colour)
        #     legendPatches.append(plot_patch)
        #
        # fig.legend(
        #     handles=legendPatches,
        #     labels=legendList,
        #     loc="center right",
        #     facecolor="white",
        #     framealpha=1,
        # )
        # plt.subplots_adjust(top=0.80, hspace=0.50,right = 0.60)
        plt.subplots_adjust(hspace=0.50,right=0.95)
        plt.tight_layout()

        opslaan = (
            f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
            + f"_{rin}R{rout}_"
            + analysisParam
            + f"_DTW_Clusters_Medians.pdf"
        )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()
