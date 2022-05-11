"""
Author: A. T. Hannington
Created: 29/07/2021

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

fontsize = 13
fontsizeTitle = 14

def medians_plot(
    dataDict,
    statsData,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    logParameters,
    ylabel,
    titleBool=False,
    separateLegend = False,
    radialSummaryBool=False,
    radialSummaryFirstLastBool = True,
    DPI=100,
    xsize = 7.0,
    ysize = 6.0,
    opacityPercentiles = 0.25,
    lineStyleMedian = "solid",
    lineStylePercentiles = "-.",
    colourmapMain = "plasma",
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",):

    xlimDict = {
        "mass": {"xmin": 5.0, "xmax": 9.0},
        "L": {"xmin": 3.0, "xmax": 4.5},
        "T": {"xmin": 3.75, "xmax": 6.5},
        "R": {"xmin": 0, "xmax": 250},
        "n_H": {"xmin": -5.0, "xmax": 0.0},
        "B": {"xmin": -2.0, "xmax": 1.0},
        "vrad": {"xmin": -150.0, "xmax": 150.0},
        "gz": {"xmin": -1.5, "xmax": 0.5},
        "P_thermal": {"xmin": 1.0, "xmax": 4.0},
        "P_magnetic": {"xmin": -1.5, "xmax": 5.0},
        "P_kinetic": {"xmin": -1.0, "xmax": 8.0},
        "P_tot": {"xmin": -1.0, "xmax": 7.0},
        "Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 3.0},
        "tcool": {"xmin": -5.0, "xmax": 2.0},
        "theat": {"xmin": -4.0, "xmax": 4.0},
        "tff": {"xmin": -1.5, "xmax": 0.5},
        "tcool_tff": {"xmin": -4.0, "xmax": 2.0},
        "rho_rhomean": {"xmin": 0.0, "xmax": 8.0},
        "dens": {"xmin": -30.0, "xmax": -22.0},
        "ndens": {"xmin": -6.0, "xmax": 2.0},
    }

    from itertools import cycle
    lines = ["--","-.",":"]
    linecycler = cycle(lines)

    for analysisParam in saveParams:
        print("")
        print(f"Starting {analysisParam} Sub-plots!")

        print("")
        print("Loading Data!")

        radialPlotData = {}
        labelList = []

        lineStyleList =[]
        for (jj,(rin, rout)) in enumerate(zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"])):
            print(f"{rin}R{rout}")

            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                sharex=True,
                sharey=True,
                figsize=(xsize, ysize),
                dpi=DPI,
            )
            yminlist = []
            ymaxlist = []
            patchList = []

            rLineStyle = next(linecycler)
            lineStyleList.append(rLineStyle)

            if  (radialSummaryBool is True):
                plot_line = matplotlib.lines.Line2D([0],[0],color="black", lineStyle = rLineStyle, label = f"{rin}<R<{rout}")
                labelList.append(plot_line)


            radialPlotData.update({f"{rin}R{rout}":{}})
            for ii in range(len(Tlst)):
                print(f"T{Tlst[ii]}")
                T = float(Tlst[ii])

                selectKey = (f"T{Tlst[ii]}", f"{rin}R{rout}")
                plotData = statsData[selectKey].copy()

                selectionSnap = np.where(
                    np.array(snapRange) == int(TRACERSPARAMS["selectSnap"])
                )

                vline = tlookback[selectionSnap]

                # Get number of temperatures
                NTemps = float(len(Tlst))

                # Get temperature
                temp = TRACERSPARAMS["targetTLst"][ii]

                # Select a Temperature specific colour from colourmapMain

                # Get a colour for median and percentiles for a given temperature
                #   Have fiddled to move colours away from extremes of the colormap
                cmap = matplotlib.cm.get_cmap(colourmapMain)
                colour = cmap(float(ii) / float(len(Tlst)))

                loadPercentilesTypes = [
                    analysisParam + "_" + str(percentile) + "%"
                    for percentile in TRACERSPARAMS["percentiles"]
                ]
                LO = analysisParam + "_" + str(min(TRACERSPARAMS["percentiles"])) + "%"
                UP = analysisParam + "_" + str(max(TRACERSPARAMS["percentiles"])) + "%"
                median = analysisParam + "_" + "50.00%"

                if analysisParam in logParameters:
                    for k, v in plotData.items():
                        plotData.update({k: np.log10(v)})

                if (jj < len(TRACERSPARAMS["Rinner"])) & (radialSummaryBool is True):
                    radialPlotData[f"{rin}R{rout}"].update({f"T{Tlst[ii]}": plotData})

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


                currentAx = ax


                midPercentile = math.floor(len(loadPercentilesTypes) / 2.0)
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
                currentAx.plot(
                    tlookback,
                    plotData[median],
                    label=r"$T = 10^{%3.0f} K$" % (float(temp)),
                    color=colour,
                    lineStyle=lineStyleMedian,
                )
                if (jj>0)&(radialSummaryBool is True):
                    nRadialData = len(list(radialPlotData.keys()))
                    for (kk,(key, rData)) in enumerate(radialPlotData.items()):
                        if (radialSummaryFirstLastBool is True)&((kk == 0)|(kk==nRadialData)):
                            data = rData[f"T{Tlst[ii]}"]
                            currentAx.plot(
                                tlookback,
                                data[median],
                                color=colour,
                                lineStyle=lineStyleList[kk],
                            )
                        elif radialSummaryFirstLastBool is False:
                            data = rData[f"T{Tlst[ii]}"]
                            currentAx.plot(
                                tlookback,
                                data[median],
                                color=colour,
                                lineStyle=lineStyleList[kk],
                            )
                        else:
                            pass
                currentAx.axvline(x=vline, c="red")

                currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                currentAx.tick_params(axis="both",which="both",labelsize=fontsize)
                #
                #Delete text string for first y_axis label for all but last panel
                # plt.gcf().canvas.draw()
                # if (int(ii)<len(Tlst)-1):
                #     plt.setp(currentAx.get_xticklabels(),visible = False)
                #     plt.gcf().canvas.draw()
                #     # STOP160IF

                plot_patch = matplotlib.patches.Patch(color=colour, label = r"$T = 10^{%3.0f} K$" % (float(temp)))
                patchList.append(plot_patch)

                if titleBool is True:
                    fig.suptitle(
                        f"Cells Containing Tracers selected by: "
                        + "\n"
                        + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
                        + r" and $%3.0f \leq R \leq %3.0f $ kpc " % (rin, rout)
                        + "\n"
                        + f" and selected at {vline[0]:3.2f} Gyr",
                        fontsize=fontsizeTitle,
                    )

            # Only give 1 x-axis a label, as they sharex
            axis0 = ax
            midax = ax

            axis0.set_xlabel("Lookback Time (Gyr)", fontsize=fontsize)
            midax.set_ylabel(ylabel[analysisParam], fontsize=fontsize)
            finalymin = min(np.nanmin(yminlist),xlimDict[analysisParam]['xmin'])
            finalymax = max(np.nanmax(ymaxlist),xlimDict[analysisParam]['xmax'])
            if (
                (np.isinf(finalymin) == True)
                or (np.isinf(finalymax) == True)
                or (np.isnan(finalymin) == True)
                or (np.isnan(finalymax) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                continue
            finalymin = numpy.round_(finalymin, decimals = 1)
            finalymax = numpy.round_(finalymax, decimals = 1)

            custom_ylim = (finalymin, finalymax)
            plt.setp(
                ax,
                ylim=custom_ylim,
                xlim=(round(max(tlookback), 1), round(min(tlookback), 1)),
            )
            if (radialSummaryBool is True):
                if jj > 0 :
                    currentLabel = matplotlib.lines.Line2D([0],[0],color="black", lineStyle = lineStyleMedian, label = f"{rin}<R<{rout}")
                    if radialSummaryFirstLastBool is True:
                        handles = patchList+labelList[:1]+[currentLabel]
                    else:
                        handles = patchList+labelList[:jj]+[currentLabel]
                    lcol = len(Tlst) + 2
                    axis0.legend(handles = handles, loc="upper right",fontsize=fontsize, ncol=lcol)
                else:
                    lcol = len(Tlst) + 1
                    handles = patchList
                    axis0.legend(handles = handles, loc="upper right",fontsize=fontsize, ncol=lcol)

            else:
                lcol = len(Tlst) + 1
                handles = patchList
                axis0.legend(handles = handles, loc="upper right",fontsize=fontsize,ncol=lcol)

            plt.tight_layout()
            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1,left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1,left=0.15)

            if (radialSummaryBool is True)&(jj>0):
                opslaan = (
                    "./"
                    + "MultiHalo"
                    + "/"
                    + f"{int(rin)}R{int(rout)}"
                    + "/"
                    + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                    + analysisParam
                    + f"_Medians_Radial_Summary.pdf"
                )
            else:
                opslaan = (
                    "./"
                    + "MultiHalo"
                    + "/"
                    + f"{int(rin)}R{int(rout)}"
                    + "/"
                    + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                    + analysisParam
                    + f"_Medians.pdf"
                )


            if (separateLegend == True):
                axis0.get_legend().remove()

                figl, axl = plt.subplots(figsize = (lcol*2.5,1))
                axl.axis(False)
                axl.legend(handles = handles, ncol=lcol, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=fontsize)
                plt.tight_layout()
                if (radialSummaryBool is True)&(jj>0):
                    figl.savefig("./"
                        + "MultiHalo"
                        + "/"
                        + f"{int(rin)}R{int(rout)}"
                        + "/"
                        + f"Medians_Radial_Summary_Legend.pdf"
                    )
                else:
                    figl.savefig("./"
                        + "MultiHalo"
                        + "/"
                        + f"{int(rin)}R{int(rout)}"
                        + "/"
                        + f"Medians_Legend.pdf"
                    )

            fig.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

    return

def currently_or_persistently_at_temperature_plot(
    dataDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    persistenceBool,
    titleBool,
    DPI=100,
    xsize = 5.0,
    ysize = 6.0,
    opacityPercentiles = 0.25,
    lineStyleMedian = "solid",
    lineStylePercentiles = "-.",
    colourmapMain = "plasma",
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
):
    Ydata = {}
    Xdata = {}
    deltaT = float(TRACERSPARAMS["deltaT"])*2.
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")
        # Loop over temperatures in targetTLst and grab Temperature specific subset of tracers and relevant data
        for T in TRACERSPARAMS["targetTLst"]:
            print("")
            print(f"Starting T{T} analysis")

            tmpXdata = []
            tmpYdata = []
            snapRangeLow = range(
                int(TRACERSPARAMS["selectSnap"]), int(TRACERSPARAMS["snapMin"] - 1), -1
            )
            snapRangeHi = range(
                int(TRACERSPARAMS["selectSnap"] + 1),
                min(
                    int(TRACERSPARAMS["snapMax"] + 1),
                    int(TRACERSPARAMS["finalSnap"] + 1),
                ),
            )

            rangeSet = [snapRangeLow, snapRangeHi]

            key = (f"T{T}", f"{rin}R{rout}")
            timeIndexSelect = np.where(np.array(snapRange) == int(TRACERSPARAMS["selectSnap"]))[0]
            whereGas = np.where(dataDict[key]["type"][timeIndexSelect][0] == 0)[0]

            data = dataDict[key]["T"][timeIndexSelect][0][whereGas]

            selectedAtSelection = np.where(
                (data >= 1.0 * 10 ** (T - TRACERSPARAMS["deltaT"]))
                & (data <= 1.0 * 10 ** (T + TRACERSPARAMS["deltaT"]))) [0]

            for tmpsnapRange in rangeSet:
                currentSelection = selectedAtSelection
                for snap in tmpsnapRange:
                    key = (f"T{T}", f"{rin}R{rout}")
                    timeIndex = np.where(np.array(snapRange) == int(snap))[0]
                    whereGas = np.where(dataDict[key]["type"][timeIndex][0] == 0)[0]

                    if persistenceBool is True:
                        data = dataDict[key]["T"][timeIndex][0][whereGas]
                        selected = np.where(
                            (data >= 1.0 * 10 ** (T - deltaT))
                            & (data <= 1.0 * 10 ** (T + deltaT)))[0]
                        currentSelection = np.intersect1d(selected,currentSelection)
                        nTracers = int(np.shape(currentSelection)[0])

                    else:
                        data = dataDict[key]["T"][timeIndex][0][whereGas]

                        selected = np.where(
                            (data >= 1.0 * 10 ** (T - deltaT))
                            & (data <= 1.0 * 10 ** (T + deltaT)))[0]
                        nTracers = int(np.shape(selected)[0])

                    # print("nTracers",nTracers)
                    # Append the data from this snapshot to a temporary list
                    tmpXdata.append(tlookback[timeIndex][0])
                    tmpYdata.append(nTracers)

            ind_sorted = np.argsort(tmpXdata)
            maxN = np.nanmax(tmpYdata)
            tmpYarray = [(float(xx) / float(maxN)) * 100.0 for xx in tmpYdata]
            tmpYarray = np.array(tmpYarray)
            tmpXarray = np.array(tmpXdata)
            tmpYarray = np.flip(
                np.take_along_axis(tmpYarray, ind_sorted, axis=0), axis=0
            )
            tmpXarray = np.flip(
                np.take_along_axis(tmpXarray, ind_sorted, axis=0), axis=0
            )

            # Add the full list of snaps data to temperature dependent dictionary.
            Xdata.update({f"T{T}": tmpXarray})
            Ydata.update({f"T{T}": tmpYarray})

        # ==============================================================================#
        #           PLOT!!
        # ==============================================================================#

        fig, ax = plt.subplots(
            nrows=1, #len(Tlst),
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(xsize, ysize),
            dpi=DPI,
        )

        # Create a plot for each Temperature
        for ii in range(len(Tlst)):
            timeIndex = np.where(
                np.array(snapRange) == int(TRACERSPARAMS["selectSnap"])
            )[0]

            vline = tlookback[timeIndex]

            T = TRACERSPARAMS["targetTLst"][ii]

            # Get number of temperatures
            NTemps = float(len(Tlst))

            # Get temperature
            temp = TRACERSPARAMS["targetTLst"][ii]

            plotYdata = Ydata[f"T{temp}"]
            plotXdata = Xdata[f"T{temp}"]

            cmap = matplotlib.cm.get_cmap(colourmapMain)
            colour = cmap(float(ii) / float(len(Tlst)))
            colourTracers = "tab:gray"

            datamin = 0.0
            datamax = np.nanmax(plotYdata)

            print("")
            print("Sub-Plot!")

            currentAx = ax


            tmpMinData = np.array([0.0 for xx in range(len(plotXdata))])

            currentAx.fill_between(
                tlookback,
                tmpMinData,
                plotYdata,
                facecolor=colour,
                alpha=0.25,
                interpolate=False,
            )

            currentAx.plot(
                tlookback,
                plotYdata,
                label=r"$T = 10^{%3.0f} K$" % (float(temp)),
                color=colour,
                lineStyle="-",
            )

            currentAx.axvline(x=vline, c="red")
            currentAx.xaxis.set_minor_locator(AutoMinorLocator())
            currentAx.yaxis.set_minor_locator(AutoMinorLocator())
            currentAx.tick_params(axis="both",which="both",labelsize=fontsize)

            currentAx.set_ylim(ymin=datamin, ymax=datamax)
            currentAx.set_xlim(
                xmin=round(max(tlookback), 1), xmax=round(min(tlookback), 1)
            )
            if titleBool is True:
                if persistenceBool is True:
                    fig.suptitle(
                        f"Percentage Tracers Persistently Within Temperature Range "
                        + r"$T = 10^{n \pm %3.2f} K$" % (deltaT)
                        + "\n"
                        + r" selected at $%3.0f \leq R \leq %3.0f $ kpc" % (rin, rout)
                        + f" and selected at {vline[0]:3.2f} Gyr",
                        fontsize=fontsizeTitle,
                    )
                else:
                    fig.suptitle(
                        f"Percentage Tracers Currently Within Temperature Range "
                        + r"$T = 10^{n \pm %3.2f} K$" % (deltaT)
                        + "\n"
                        + r" selected at $%3.0f \leq R \leq %3.0f $ kpc" % (rin, rout)
                        + f" and selected at {vline[0]:3.2f} Gyr",
                        fontsize=fontsizeTitle,
                    )

        # Only give 1 x-axis a label, as they sharex
        # if len(Tlst) == 1:
        axis0 = ax
        midax = ax
        # else:
        #     axis0 = ax[len(Tlst) - 1]
        #     midax = ax[(len(Tlst) - 1) // 2]

        axis0.set_xlabel("Lookback Time (Gyr)", fontsize=fontsize)
        midax.set_ylabel(
            f"Percentage Tracers Within"+f"\n"+f"Temperature Range",
            fontsize=fontsize,
        )
        axis0.legend(loc="upper right",fontsize=fontsize)
        plt.tight_layout(h_pad=0.0)
        plt.subplots_adjust(left=0.15)
        if persistenceBool is True:
            opslaan = (
                "./"
                + "MultiHalo"
                + "/"
                + f"{int(rin)}R{int(rout)}"
                + "/"
                + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_T"
                + f"_Persistently_within_Temperature.pdf"
            )
        else:
            opslaan = (
                "./"
                + "MultiHalo"
                + "/"
                + f"{int(rin)}R{int(rout)}"
                + "/"
                + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_T"
                + f"_Currently_within_Temperature.pdf"
            )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()
    return


def stacked_pdf_plot(
    dataDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    logParameters,
    ylabel,
    titleBool,
    DPI=100,
    xsize = 5.0,
    ysize = 10.0,
    opacityPercentiles = 0.25,
    lineStyleMedian = "solid",
    lineStylePercentiles = "-.",
    colourmapMain = "plasma",
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
):

    opacity = 0.75
    selectColour = "red"
    selectStyle = "-."
    selectWidth = 4
    percentileLO = 1.0
    percentileUP = 99.0

    xlimDict = {
        "T": {"xmin": 3.75, "xmax": 6.5},
        "R": {"xmin": 0, "xmax": 250},
        "n_H": {"xmin": -6.0, "xmax": 0.0},
        "B": {"xmin": -6.0, "xmax": 2.0},
        "vrad": {"xmin": -250.0, "xmax": 250.0},
        "gz": {"xmin": -4.0, "xmax": 1.0},
        "L": {"xmin": 0.0, "xmax": 5.0},
        "P_thermal": {"xmin": -1.0, "xmax": 7.0},
        "P_magnetic": {"xmin": -7.0, "xmax": 7.0},
        "P_kinetic": {"xmin": -1.0, "xmax": 8.0},
        "P_tot": {"xmin": -1.0, "xmax": 7.0},
        "Pthermal_Pmagnetic": {"xmin": -3.0, "xmax": 10.0},
        "tcool": {"xmin": -6.0, "xmax": 3.0},
        "theat": {"xmin": -4.0, "xmax": 4.0},
        "tff": {"xmin": -3.0, "xmax": 1.0},
        "tcool_tff": {"xmin": -6.0, "xmax": 3.0},
        "rho_rhomean": {"xmin": 0.0, "xmax": 8.0},
        "dens": {"xmin": -30.0, "xmax": -22.0},
        "ndens": {"xmin": -6.0, "xmax": 2.0},
    }
    import seaborn as sns
    import scipy.stats as stats

    xlabel = ylabel

    outofrangeNbins = 10

    for dataKey in saveParams:
        print(f"{dataKey}")
        # Create a plot for each Temperature
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            print(f"{rin}R{rout}")
            for ii in range(len(Tlst)):
                print(f"T{Tlst[ii]}")

                selectKey = (f"T{Tlst[ii]}", f"{rin}R{rout}")

                # Get number of temperatures
                NTemps = float(len(Tlst))

                # Get temperature
                T = TRACERSPARAMS["targetTLst"][ii]

                selectKey = (
                    f"T{T}",
                    f"{rin}R{rout}",
                )
                timeIndex = np.where(
                    np.array(snapRange) == int(TRACERSPARAMS["selectSnap"])
                )[0]

                selectTime = tlookback[timeIndex][0]

                xmaxlist = []
                xminlist = []
                dataList = []
                weightsList = []

                sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
                fig, ax = plt.subplots(
                    nrows=len(snapRange),
                    ncols=1,
                    figsize=(xsize, ysize),
                    dpi=DPI,
                    frameon=False,
                    sharex=True,
                )
                # Loop over snaps from snapMin to snapmax, taking the snapnumMAX (the final snap) as the endpoint if snapMax is greater
                for (jj, snap) in enumerate(snapRange):
                    currentAx = ax[jj]
                    dictkey = (f"T{T}", f"{rin}R{rout}")
                    timeIndex = np.where(np.array(snapRange) == snap)[0]
                    whereGas = np.where(dataDict[dictkey]["type"][timeIndex][0] == 0)

                    data = dataDict[dictkey][dataKey][timeIndex][0][whereGas]
                    weights = dataDict[dictkey]["mass"][timeIndex][0][whereGas]

                    if dataKey in logParameters:
                        data = np.log10(data)

                    wheredata = np.where(
                        (np.isinf(data) == False) & ((np.isnan(data) == False))
                    )[0]
                    whereweights = np.where(
                        (np.isinf(weights) == False) & ((np.isnan(weights) == False))
                    )[0]
                    whereFull = wheredata[np.where(np.isin(wheredata, whereweights))]
                    data = data[whereFull]
                    weights = weights[whereFull]

                    dataList.append(data)
                    weightsList.append(weights)

                    if np.shape(data)[0] == 0:
                        print("No Data! Skipping Entry!")
                        continue

                    # Select a Temperature specific colour from colourmapMain
                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    if int(snap) == int(TRACERSPARAMS["selectSnap"]):
                        colour = selectColour
                        lineStyle = selectStyle
                        linewidth = selectWidth
                    else:
                        sRange = (
                            int(
                                min(
                                    TRACERSPARAMS["finalSnap"] + 1,
                                    TRACERSPARAMS["snapMax"] + 1,
                                )
                            )
                            - int(TRACERSPARAMS["snapMin"])
                        )
                        colour = cmap(((float(jj)) / (sRange)))
                        lineStyle = "-"
                        linewidth = 2

                    tmpdict = {"x": data, "y": weights}
                    df = pd.DataFrame(tmpdict)

                    xmin = xlimDict[dataKey]["xmin"]
                    xmax = xlimDict[dataKey]["xmax"]

                    ###############################
                    ##### aggregate endpoints #####
                    ###############################

                    belowrangedf = df.loc[(df["x"] < xmin)]
                    inrangedf = df.loc[(df["x"] >= xmin) & (df["x"] <= xmax)]
                    aboverangedf = df.loc[(df["x"] > xmax)]


                    belowrangedf = belowrangedf.assign(x=xmin)
                    aboverangedf = aboverangedf.assign(x=xmax)

                    df = pd.concat(
                        [belowrangedf, inrangedf, aboverangedf],
                        axis=0,
                        ignore_index=True,
                    )

                    xminlist.append(xmin)
                    xmaxlist.append(xmax)
                    # Draw the densities in a few steps
                    ## MAIN KDE ##
                    mainplot = sns.kdeplot(
                        df["x"],
                        weights=df["y"],
                        ax=currentAx,
                        bw_adjust=0.1,
                        clip=(xmin, xmax),
                        alpha=opacity,
                        fill=True,
                        lw=linewidth,
                        color=colour,
                        linestyle=lineStyle,
                        shade=True,
                        common_norm=True,
                    )
                    # mainplotylim = mainplot.get_ylim()
                    currentAx.axhline(
                        y=0,
                        lw=linewidth,
                        linestyle=lineStyle,
                        color=colour,
                        clip_on=False,
                    )

                    currentAx.set_yticks([])
                    currentAx.set_ylabel("")
                    # currentAx.set_ylim(mainplotylim)
                    currentAx.set_xlabel(xlabel[dataKey], fontsize=fontsize)
                    currentAx.tick_params(axis="both",which="both",labelsize=fontsize)
                    sns.despine(bottom=True, left=True)

                xmin = np.nanmin(xminlist)
                xmax = np.nanmax(xmaxlist)

                plt.xlim(xmin, xmax)
                #
                # plot_label = r"$T = 10^{%3.0f} K$" % (float(T))
                # plt.text(
                #     0.75,
                #     0.95,
                #     plot_label,
                #     horizontalalignment="left",
                #     verticalalignment="center",
                #     transform=fig.transFigure,
                #     wrap=True,
                #     bbox=dict(facecolor="blue", alpha=0.2),
                #     fontsize=fontsize,
                # )

                time_label = r"Time (Gyr)"
                plt.text(
                    0.08,
                    0.525,
                    time_label,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=fig.transFigure,
                    wrap=True,
                    fontsize=fontsize,
                )
                plt.arrow(
                    0.08,
                    0.500,
                    0.00,
                    -0.15,
                    fc="black",
                    ec="black",
                    width=0.005,
                    transform=fig.transFigure,
                    clip_on=False,
                )
                fig.transFigure

                if titleBool is True:
                    fig.suptitle(
                        f"PDF of Cells Containing Tracers selected by: "
                        + "\n"
                        + r"$T = 10^{%3.2f \pm %3.2f} K$" % (T, TRACERSPARAMS["deltaT"])
                        + r" and $%3.0f \leq R \leq %3.0f $ kpc" % (rin, rout)
                        + "\n"
                        + f" and selected at {selectTime:3.2f} Gyr",
                        fontsize=fontsizeTitle,
                    )
                # ax.axvline(x=vline, c='red')

                plt.tight_layout()
                if titleBool is True:
                    plt.subplots_adjust(top=0.90, bottom=0.075, hspace=-0.25)
                else:
                    plt.subplots_adjust(top=0.98, bottom=0.075, hspace=-0.25)
                opslaan = (
                    "./"
                    + "MultiHalo"
                    + "/"
                    + f"{int(rin)}R{int(rout)}"
                    + "/"
                    + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_snap{int(snap)}_T{T}_{dataKey}_PDF.pdf"
                )
                plt.savefig(opslaan, dpi=DPI, transparent=False)
                print(opslaan)
                plt.close()

    return


def phases_plot(
    dataDict,
    TRACERSPARAMS,
    saveParams,
    snapRange,
    Tlst,
    titleBool,
    ylabel,
    DPI=100,
    xsize = 20.0,
    ysize = 5.0,
    opacityPercentiles = 0.25,
    lineStyleMedian = "solid",
    lineStylePercentiles = "-.",
    colourmapMain = "plasma",
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
    Nbins=250,
):
    """
    Author: A. T. Hannington
    Created: 21/07/2020

    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    # Paramters to weight the 2D hist by
    weightKeys = ["mass", "tcool", "gz", "tcool_tff"]
    logparams = ["mass", "tcool", "gz", "tcool_tff"]
    zlimDict = {
        "mass": {"zmin": 4.0, "zmax": 9.0},
        "tcool": {"zmin": -5.0, "zmax": 4.0},
        "gz": {"zmin": -2.0, "zmax": 2.0},
        "tcool_tff": {"zmin": -6.0, "zmax": 4.0},
    }
    ymin = 3.5  # [Log10 T]
    ymax = 7.5  # [Log10 T]
    xmin = 1.0  # [Log10 rho_rhomean]
    xmax = 7.0  # [Log10 rho_rhomean]
    labelDict = ylabel

    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")
        print("Flatten Tracers Data (snapData).")

        TracersFinalDict = flatten_wrt_T(dataDict, snapRange, TRACERSPARAMS, rin, rout)
        # ------------------------------------------------------------------------------#
        #               PLOTTING
        #
        # ------------------------------------------------------------------------------#
        for snap in snapRange:
            print("\n" + f"Starting Snap {int(snap)}")
            for weightKey in weightKeys:
                print("\n" + f"Starting weightKey {weightKey}")
                key = f"{int(snap)}"
                tkey = (f"{rin}R{rout}", key)
                selectTime = abs(
                    dataDict[
                        (
                            f"T{float(Tlst[0])}",
                            f"{rin}R{rout}",
                            f"{int(TRACERSPARAMS['selectSnap'])}",
                        )
                    ]["Lookback"][0]
                )
                currentTime = abs(
                    dataDict[(f"T{float(Tlst[0])}", f"{rin}R{rout}", f"{int(snap)}")][
                        "Lookback"
                    ][0]
                )

                zmin = zlimDict[weightKey]["zmin"]
                zmax = zlimDict[weightKey]["zmax"]

                fig, ax = plt.subplots(
                    nrows=1,
                    ncols=int(len(Tlst)),
                    figsize=(xsize, ysize),
                    dpi=DPI,
                    sharey=True,
                    sharex=True,
                )

                for (ii, T) in enumerate(Tlst):
                    FullDictKey = (f"T{float(T)}", f"{rin}R{rout}", f"{int(snap)}")

                    if len(Tlst) == 1:
                        currentAx = ax
                    else:
                        currentAx = ax[ii]

                    whereGas = np.where(dataDict[FullDictKey]["type"] == 0)[0]

                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                    #   Figure 1: Full Cells Data
                    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                    print(f"T{T} Sub-Plot!")

                    xdataCells = np.log10(
                        dataDict[FullDictKey]["rho_rhomean"][whereGas]
                    )
                    ydataCells = np.log10(dataDict[FullDictKey]["T"][whereGas])
                    massCells = dataDict[FullDictKey]["mass"][whereGas]
                    weightDataCells = (
                        dataDict[FullDictKey][weightKey][whereGas] * massCells
                    )

                    if weightKey == "mass":
                        finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                            xdataCells, ydataCells, bins=Nbins, weights=massCells
                        )
                    else:
                        mhistCells, _, _ = np.histogram2d(
                            xdataCells, ydataCells, bins=Nbins, weights=massCells
                        )
                        histCells, xedgeCells, yedgeCells = np.histogram2d(
                            xdataCells, ydataCells, bins=Nbins, weights=weightDataCells
                        )

                        finalHistCells = histCells / mhistCells

                    finalHistCells[finalHistCells == 0.0] = np.nan
                    if weightKey in logparams:
                        finalHistCells = np.log10(finalHistCells)
                    finalHistCells = finalHistCells.T

                    xcells, ycells = np.meshgrid(xedgeCells, yedgeCells)

                    img1 = currentAx.pcolormesh(
                        xcells,
                        ycells,
                        finalHistCells,
                        cmap=colourmapMain,
                        vmin=zmin,
                        vmax=zmax,
                        rasterized=True,
                    )
                    #
                    # img1 = currentAx.imshow(finalHistCells,cmap=colourmapMain,vmin=zmin,vmax=zmax \
                    # ,extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')

                    currentAx.set_xlabel(
                        r"Log10 Density ($ \rho / \langle \rho \rangle $)",
                        fontsize=fontsize,
                    )
                    currentAx.set_ylabel("Log10 Temperatures (K)", fontsize=fontsize)

                    currentAx.set_ylim(ymin, ymax)
                    currentAx.set_xlim(xmin, xmax)
                    currentAx.tick_params(axis="both",which="both",labelsize=fontsize)

                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    colour = cmap(float(ii) / float(len(Tlst)))

                    plot_patch = matplotlib.patches.Patch(color=colour)
                    plot_label = r"$T = 10^{%3.0f} K$" % (float(T))
                    currentAx.legend(
                        handles=[plot_patch], labels=[plot_label], loc="upper right", fontsize=fontsize
                    )

                    cax1 = inset_axes(currentAx, width="5%", height="95%", loc="right")
                    fig.colorbar(img1, cax=cax1, orientation="vertical").set_label(
                        label=labelDict[weightKey], size=fontsize
                    )
                    cax1.yaxis.set_ticks_position("left")
                    cax1.yaxis.set_label_position("left")
                    cax1.yaxis.label.set_color("black")
                    cax1.tick_params(axis="y", colors="black", labelsize=fontsize)

                    currentAx.set_title(
                        r"$ 10^{%03.2f \pm %3.2f} K $ Tracers Data"
                        % (float(T), TRACERSPARAMS["deltaT"]),
                        fontsize=fontsizeTitle,
                    )
                    currentAx.set_aspect("auto")

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                #   Temperature Figure: Finishing up
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                if titleBool is True:
                    fig.suptitle(
                        f"Temperature Density Diagram, weighted by {weightKey}"
                        + f" at {currentTime:3.2f} Gyr"
                        + "\n"
                        + f"Tracers Data, selected at {selectTime:3.2f} Gyr with"
                        + r" $%3.0f \leq R \leq %3.0f $ kpc" % (rin, rout)
                        + r" and temperatures "
                        + r"$ 10^{n \pm %3.2f} K $" % (TRACERSPARAMS["deltaT"]),
                        fontsize=fontsizeTitle,
                    )

                plt.tight_layout()

                opslaan = (
                    "./"
                    + "MultiHalo"
                    + "/"
                    + f"{int(rin)}R{int(rout)}"
                    + "/"
                    + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_snap{int(snap)}_{weightKey}_PhaseDiagram_Individual-Temps.pdf"
                )
                plt.savefig(opslaan, dpi=DPI, transparent=False)
                print(opslaan)

    return


################################################################################
##                           Bars Plot Tools                                ####
################################################################################
def _get_id_prid_trid_where(dataDict, whereEntries):

    id = dataDict["id"][whereEntries]

    _, prid_ind, _ = np.intersect1d(dataDict["prid"], id, return_indices=True)

    prid = dataDict["prid"][prid_ind]
    trid = dataDict["trid"][prid_ind]

    return {"id": id, "prid": prid, "trid": trid}


def flat_analyse_time_averages(
    FlatDataDict,
    Tlst,
    snapRange,
    tlookback,
    TRACERSPARAMS,
    shortSnapRangeBool=False,
    shortSnapRangeNumber=None,
):

    epsilon = float(TRACERSPARAMS['deltaT'])

    gas = []
    halo0 = []
    unbound = []
    otherHalo = []
    noHalo = []
    stars = []
    wind = []
    ism =[]
    disk = []
    cgm = []
    igm = []
    cooling = []
    stableT = []
    heating = []
    dispersing = []
    stabledensity = []
    condensing = []
    inflow = []
    statflow = []
    outflow = []
    bdecreasing = []
    bstable = []
    bincreasing = []
    belowZ = []
    aboveZ = []
    tcool_tff_LO = []
    tcool_tff_UP = []
    ptherm_pmag_LO = []
    ptherm_pmag_UP = []


    out = {}
    preselectInd = np.where(snapRange < int(TRACERSPARAMS["selectSnap"]))[0]
    postselectInd = np.where(snapRange > int(TRACERSPARAMS["selectSnap"]))[0]
    selectInd = np.where(snapRange == int(TRACERSPARAMS["selectSnap"]))[0][0]

    rmin = min(TRACERSPARAMS["Rinner"])
    rmax = max(TRACERSPARAMS["Router"])
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        dfdat = {}
        for T in Tlst:
            Tkey = (f"T{T}", f"{rin}R{rout}")
            print(Tkey)

            if len(dfdat.keys()) > 0:
                val = dfdat["T"]
                Tval = val + [T]

                val = dfdat["Rinner"]
                rinval = val + [rin]

                val = dfdat["Router"]
                routval = val + [rout]

                dfdat.update({"T": Tval, "Rinner": rinval, "Router": routval})
            else:
                dfdat.update({"T": [T], "Rinner": [rin], "Router": [rout]})

            data = FlatDataDict[Tkey]
            ntracersAll = FlatDataDict[Tkey]["Ntracers"]

            # Select only the tracers which were ALWAYS gas
            whereGas = np.where((FlatDataDict[Tkey]["type"] == 0).all(axis=0))[0]
            ntracers = int(np.shape(whereGas)[0])

            print("Gas")

            if (shortSnapRangeBool is False) & (shortSnapRangeNumber is None):
                pre = preselectInd
                post = postselectInd
            else:
                pre = preselectInd[-1 * int(shortSnapRangeNumber) :]
                post = postselectInd[: int(shortSnapRangeNumber)]

            # Select where ANY tracer (gas or stars) meets condition PRIOR TO selection
            rowspre, colspre = np.where(FlatDataDict[Tkey]["type"][pre, :] == 0)
            # Calculate the number of these unique tracers compared to the total number
            gaspre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            rowspost, colspost = np.where(FlatDataDict[Tkey]["type"][post, :] == 0)
            gaspost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracersAll)
            )
            # Add data to internal database lists
            gas.append([gaspre, gaspost])

            print("Halo0")
            rowspre, colspre = np.where(
                FlatDataDict[Tkey]["subhalo"][:, whereGas][pre, :]
                == int(TRACERSPARAMS["haloID"])
            )
            halo0pre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            rowspost, colspost = np.where(
                FlatDataDict[Tkey]["subhalo"][:, whereGas][post, :]
                == int(TRACERSPARAMS["haloID"])
            )
            halo0post = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            halo0.append([halo0pre, halo0post])

            print("Unbound")
            rowspre, colspre = np.where(
                FlatDataDict[Tkey]["subhalo"][:, whereGas][pre, :] == -1
            )
            unboundpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            rowspost, colspost = np.where(
                FlatDataDict[Tkey]["subhalo"][:, whereGas][post, :] == -1
            )
            unboundpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            unbound.append([unboundpre, unboundpost])

            print("OtherHalo")
            rowspre, colspre = np.where(
                (
                    FlatDataDict[Tkey]["subhalo"][:, whereGas][pre, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["subhalo"][:, whereGas][pre, :] != -1)
                & (
                    np.isnan(FlatDataDict[Tkey]["subhalo"][:, whereGas][pre, :])
                    == False
                )
            )
            otherHalopre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            rowspost, colspost = np.where(
                (
                    FlatDataDict[Tkey]["subhalo"][:, whereGas][post, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["subhalo"][:, whereGas][post, :] != -1)
                & (
                    np.isnan(FlatDataDict[Tkey]["subhalo"][:, whereGas][post, :])
                    == False
                )
            )
            otherHalopost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            otherHalo.append([otherHalopre, otherHalopost])

            print("NoHalo")
            rowspre, colspre = np.where(
                (
                    FlatDataDict[Tkey]["subhalo"][:, whereGas][pre, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["subhalo"][:, whereGas][pre, :] != -1)
                & (
                    np.isnan(FlatDataDict[Tkey]["subhalo"][:, whereGas][pre, :])
                    == True
                )
            )
            noHalopre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            rowspost, colspost = np.where(
                (
                    FlatDataDict[Tkey]["subhalo"][:, whereGas][post, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["subhalo"][:, whereGas][post, :] != -1)
                & (
                    np.isnan(FlatDataDict[Tkey]["subhalo"][:, whereGas][post, :])
                    == True
                )
            )
            noHalopost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            noHalo.append([noHalopre, noHalopost])

            print("Stars")
            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["type"][pre, :] == 4)
                & (FlatDataDict[Tkey]["age"][pre, :] >= 0.0)
            )
            starspre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            )
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["type"][post, :] == 4)
                & (FlatDataDict[Tkey]["age"][post, :] >= 0.0)
            )
            starspost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracersAll)
            )
            stars.append([starspre, starspost])

            print("Wind")
            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["type"][pre, :] == 4)
                & (FlatDataDict[Tkey]["age"][pre, :] < 0.0)
            )
            windpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            )
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["type"][post, :] == 4)
                & (FlatDataDict[Tkey]["age"][post, :] < 0.0)
            )
            windpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracersAll)
            )
            wind.append([windpre, windpost])

            print("ISM")
            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["R"][pre, :] <= rmin)
                & (FlatDataDict[Tkey]["sfr"][pre, :] > 0.0)
            )
            ismpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["R"][post, :] <= rmin)
                & (FlatDataDict[Tkey]["sfr"][post, :] > 0.0)
            )
            ismpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracersAll)
            )
            ism.append([ismpre, ismpost])

            print("Wind")
            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["type"][pre, :] == 4)
                & (FlatDataDict[Tkey]["age"][pre, :] < 0.0)
            )
            windpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            )
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["type"][post, :] == 4)
                & (FlatDataDict[Tkey]["age"][post, :] < 0.0)
            )
            windpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracersAll)
            )
            wind.append([windpre, windpost])

            print("Radius")

            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["R"][:, whereGas][pre, :] < rmin)
            )
            diskpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["R"][:, whereGas][post, :] < rmin)
            )
            diskpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            disk.append([diskpre, diskpost])

            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["R"][:, whereGas][pre, :] >= rmin) &         (FlatDataDict[Tkey]["R"][:, whereGas][pre, :] <= rmax)
            )
            cgmpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["R"][:, whereGas][post, :] >= rmin) & (FlatDataDict[Tkey]["R"][:, whereGas][post, :] <= rmax)
            )
            cgmpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            cgm.append([cgmpre, cgmpost])

            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["R"][:, whereGas][pre, :] > rmax)
            )
            igmpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["R"][:, whereGas][post, :] > rmax)
            )
            igmpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            igm.append([igmpre, igmpost])

            print("Heating & Cooling")

            TPreDat =  np.log10(FlatDataDict[Tkey]["T"][:, whereGas][pre, :][-1,:]) - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])

            TPostDat = np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :]) - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][post, :][0,:])

            colspre = np.where(TPreDat < (-1. * epsilon))[0]
            coolingPre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(TPostDat < (-1. * epsilon))[0]
            coolingPost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            cooling.append([coolingPre, coolingPost])

            colspre = np.where((TPreDat >= (-1. * epsilon))&(TPreDat <= (epsilon))) [0]
            stableTPre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where((TPostDat >= (-1. * epsilon))&(TPostDat <= (epsilon)))[0]
            stableTPost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            stableT.append([stableTPre, stableTPost])

            colspre = np.where(TPreDat > (epsilon))[0]
            heatingPre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(TPostDat > (epsilon))[0]
            heatingPost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            heating.append([heatingPre, heatingPost])


            #
            # # Select where GAS FOREVER ONLY tracers meet condition FOR THE LAST 2 SNAPSHOTS PRIOR TO SELECTION
            # rowspre, colspre = np.where(
            #     np.log10(FlatDataDict[Tkey]["T"][:, whereGas][pre, :][-1:])
            #     - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
            #     > (epsilonT)
            # )
            # # Calculate the number of these unique tracers compared to the total number
            # bdecreasingpre = (
            #     100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # )
            #
            # rowspost, colspost = np.where(
            #     np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
            #     - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][post, :][:1])
            #     > (epsilonT)
            # )
            #
            # bdecreasingpost = (
            #     100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # )
            #
            # rowspre, colspre = np.where(
            #     np.log10(FlatDataDict[Tkey]["T"][:, whereGas][pre, :][-1:])
            #     - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
            #     < (-1.0 * epsilonT)
            # )
            # bincreasingpre = (
            #     100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # )
            #
            # rowspost, colspost = np.where(
            #     np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
            #     - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][post, :][:1])
            #     < (-1.0 * epsilonT)
            # )
            # bincreasingpost = (
            #     100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # )
            #
            # rowspre, colspre = np.where(
            #     (
            #         np.log10(FlatDataDict[Tkey]["T"][:, whereGas][pre, :][-1:])
            #         - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
            #         <= (0 + epsilonT)
            #     )
            #     & (
            #         np.log10(FlatDataDict[Tkey]["T"][:, whereGas][pre, :][-1:])
            #         - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
            #         >= (0 - epsilonT)
            #     )
            # )
            # smallTpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            #
            # rowspost, colspost = np.where(
            #     (
            #         np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
            #         - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][post, :][:1])
            #         <= (0.0 + epsilonT)
            #     )
            #     & (
            #         np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
            #         - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][post, :][:1])
            #         >= (0.0 - epsilonT)
            #     )
            # )
            # smallTpost = (
            #     100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # )
            # # Add data to internal database lists
            #
            # bdecreasing.append([bdecreasingpre, bdecreasingpost])
            # bincreasing.append([bincreasingpre, bincreasingpost])
            # smallTchange.append([smallTpre, smallTpost])

            print("Density")
              # (K)

            nHPreDat =  np.log10(FlatDataDict[Tkey]["n_H"][:, whereGas][pre, :][-1,:]) - np.log10(FlatDataDict[Tkey]["n_H"][:, whereGas][selectInd, :])

            nHPostDat = np.log10(FlatDataDict[Tkey]["n_H"][:, whereGas][selectInd, :]) - np.log10(FlatDataDict[Tkey]["n_H"][:, whereGas][post, :][0,:])

            colspre = np.where(nHPreDat < (-1. * epsilon))[0]
            dispersingPre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(nHPostDat < (-1. * epsilon))[0]
            dispersingPost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            dispersing.append([dispersingPre, dispersingPost])

            colspre = np.where((nHPreDat >= (-1. * epsilon))&(nHPreDat <= (epsilon))) [0]
            stablenHPre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where((nHPostDat >= (-1. * epsilon))&(nHPostDat <= (epsilon)))[0]
            stablenHPost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            stabledensity.append([stablenHPre, stablenHPost])

            colspre = np.where(nHPreDat > (epsilon))[0]
            condensingPre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(nHPostDat > (epsilon))[0]
            condensingPost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            condensing.append([condensingPre, condensingPost])


            print("Radial-Flow")
            data = FlatDataDict[Tkey]["vrad"][:, whereGas][pre, :]
            vradPreDat = np.nanmedian(data,axis=0)
            data = FlatDataDict[Tkey]["vrad"][:, whereGas][post, :]
            vradPostDat = np.nanmedian(data,axis=0)

            epsilonRadial = 50.0

            colspre = np.where(vradPreDat < 0.0 - epsilonRadial)[0]
            inflowpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(vradPostDat < 0.0 - epsilonRadial)[0]
            inflowpost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            inflow.append([inflowpre, inflowpost])

            colspre = np.where(
                (vradPreDat >= 0.0 - epsilonRadial) & (vradPreDat <= 0.0 + epsilonRadial)
            )[0]
            statflowpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            colspost = np.where(
                (vradPostDat >= 0.0 - epsilonRadial) & (vradPostDat <= 0.0 + epsilonRadial)
            )[0]
            statflowpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            statflow.append([statflowpre, statflowpost])

            colspre = np.where(vradPreDat > 0.0 + epsilonRadial)[0]
            outflowpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            colspost = np.where(vradPostDat > 0.0 + epsilonRadial)[0]
            outflowpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            outflow.append([outflowpre, outflowpost])


            print("|B|")
              # (K)

            BPreDat =  np.log10(FlatDataDict[Tkey]["B"][:, whereGas][pre, :][-1,:]) - np.log10(FlatDataDict[Tkey]["B"][:, whereGas][selectInd, :])

            BPostDat = np.log10(FlatDataDict[Tkey]["B"][:, whereGas][selectInd, :]) - np.log10(FlatDataDict[Tkey]["B"][:, whereGas][post, :][0,:])

            colspre = np.where(BPreDat < (-1. * epsilon))[0]
            bdecreasingPre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(BPostDat < (-1. * epsilon))[0]
            bdecreasingPost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            bdecreasing.append([bdecreasingPre, bdecreasingPost])

            colspre = np.where((BPreDat >= (-1. * epsilon))&(BPreDat <= (epsilon))) [0]
            bstablePre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where((BPostDat >= (-1. * epsilon))&(BPostDat <= (epsilon))) [0]
            bstablePost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            bstable.append([bstablePre, bstablePost])

            colspre = np.where(BPreDat > (epsilon))[0]
            bincreasingPre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(BPostDat > (epsilon))[0]
            bincreasingPost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            bincreasing.append([bincreasingPre, bincreasingPost])
             #
            print("Z")
            # Select FOREVER GAS ONLY tracers' specific parameter data and mass weights PRIOR TO SELECTION

            data = FlatDataDict[Tkey]["gz"][:, whereGas][pre, :]
            zPreDat = np.nanmedian(data,axis=0)

            data = FlatDataDict[Tkey]["gz"][:, whereGas][post, :]
            zPostDat = np.nanmedian(data,axis=0)

            colspre = np.where(zPreDat > 0.75)[0]
            aboveZpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(zPostDat > 0.75)[0]
            aboveZpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            aboveZ.append([aboveZpre, aboveZpost])

            colspre = np.where(zPreDat < 0.75)[0]
            belowZpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(zPostDat < 0.75)[0]
            belowZpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            belowZ.append([belowZpre, belowZpost])

            print("tcool_tff")
            # Select FOREVER GAS ONLY tracers' specific parameter data and mass weights PRIOR TO SELECTION

            data = FlatDataDict[Tkey]["tcool_tff"][:, whereGas][pre, :]
            tcool_tffPreDat = np.nanmedian(data,axis=0)

            data = FlatDataDict[Tkey]["tcool_tff"][:, whereGas][post, :]
            tcool_tffPostDat = np.nanmedian(data,axis=0)

            colspre = np.where(tcool_tffPreDat < 10.)[0]
            belowtcool_tffpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(tcool_tffPostDat < 10.)[0]
            belowtcool_tffpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            tcool_tff_LO.append([belowtcool_tffpre, belowtcool_tffpost])


            colspre = np.where(tcool_tffPreDat > 10.)[0]
            abovetcool_tffpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(tcool_tffPostDat > 10.)[0]
            abovetcool_tffpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            tcool_tff_UP.append([abovetcool_tffpre, abovetcool_tffpost])


            print("ptherm_pmag")
            # Select FOREVER GAS ONLY tracers' specific parameter data and mass weights PRIOR TO SELECTION

            data = FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:, whereGas][pre, :]
            ptherm_pmagPreDat = np.nanmedian(data,axis=0)

            data = FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:, whereGas][post, :]
            ptherm_pmagPostDat = np.nanmedian(data,axis=0)
            colspre = np.where(ptherm_pmagPreDat < 1.)[0]
            belowptherm_pmagpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(ptherm_pmagPostDat < 1.)[0]
            belowptherm_pmagpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            ptherm_pmag_LO.append([belowptherm_pmagpre, belowptherm_pmagpost])


            colspre = np.where(ptherm_pmagPreDat > 1.)[0]
            aboveptherm_pmagpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(ptherm_pmagPostDat > 1.)[0]
            aboveptherm_pmagpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            ptherm_pmag_UP.append([aboveptherm_pmagpre, aboveptherm_pmagpost])

        outinner = {
            "Rinner": dfdat["Rinner"],
            "Router": dfdat["Router"],
            "T": dfdat[
                "T"
            ],  # "%Gas": {"Pre-Selection" : np.array(gas)[:,0],"Post-Selection" : np.array(gas)[:,1]} , \
            # "Halo0": {
            #     "Pre-Selection": np.array(halo0)[:, 0],
            #     "Post-Selection": np.array(halo0)[:, 1],
            # },
            "Unbound": {
                "Pre-Selection": np.array(unbound)[:, 0],
                "Post-Selection": np.array(unbound)[:, 1],
            },
            "OtherHalo": {
                "Pre-Selection": np.array(otherHalo)[:, 0],
                "Post-Selection": np.array(otherHalo)[:, 1],
            },
            "IGM": {
                "Pre-Selection": np.array(noHalo)[:, 0],
                "Post-Selection": np.array(noHalo)[:, 1],
            },
            "ISM": {
                "Pre-Selection": np.array(ism)[:, 0],
                "Post-Selection": np.array(ism)[:, 1],
            },
            "Stars": {
                "Pre-Selection": np.array(stars)[:, 0],
                "Post-Selection": np.array(stars)[:, 1],
            },
            # "Wind": {
            #     "Pre-Selection": np.array(wind)[:, 0],
            #     "Post-Selection": np.array(wind)[:, 1],
            # },

            # "Disk": {
            #     "Pre-Selection": np.array(disk)[:, 0],
            #     "Post-Selection": np.array(disk)[:, 1],
            # },
            # "CGM": {
            #     "Pre-Selection": np.array(cgm)[:, 0],
            #     "Post-Selection": np.array(cgm)[:, 1],
            # },
            # "IGM": {
            #     "Pre-Selection": np.array(igm)[:, 0],
            #     "Post-Selection": np.array(igm)[:, 1],
            # },
            "Inflow": {
                "Pre-Selection": np.array(inflow)[:, 0],
                "Post-Selection": np.array(inflow)[:, 1],
            },
            "Radially-Static": {
                "Pre-Selection": np.array(statflow)[:, 0],
                "Post-Selection": np.array(statflow)[:, 1],
            },
            "Outflow": {
                "Pre-Selection": np.array(outflow)[:, 0],
                "Post-Selection": np.array(outflow)[:, 1],
            },
            # "<3/4(Z_Solar)": {
            #     "Pre-Selection": np.array(belowZ)[:, 0],
            #     "Post-Selection": np.array(belowZ)[:, 1],
            # },
            # ">3/4(Z_Solar)": {
            #     "Pre-Selection": np.array(aboveZ)[:, 0],
            #     "Post-Selection": np.array(aboveZ)[:, 1],
            # },
            #
            # "(tCool/tFF)<10": {
            #     "Pre-Selection": np.array(tcool_tff_LO)[:, 0],
            #     "Post-Selection": np.array(tcool_tff_LO)[:, 1],
            # },
            # "(tCool/tFF)>10": {
            #     "Pre-Selection": np.array(tcool_tff_UP)[:, 0],
            #     "Post-Selection": np.array(tcool_tff_UP)[:, 1],
            # },
            # "(PTherm/PMag)<1": {
            #     "Pre-Selection": np.array(ptherm_pmag_LO)[:, 0],
            #     "Post-Selection": np.array(ptherm_pmag_LO)[:, 1],
            # },
            # "(PTherm/PMag)>1": {
            #     "Pre-Selection": np.array(ptherm_pmag_UP)[:, 0],
            #     "Post-Selection": np.array(ptherm_pmag_UP)[:, 1],
            # },
            "Cooling": {
                "Pre-Selection": np.array(cooling)[:, 0],
                "Post-Selection": np.array(cooling)[:, 1],
            },
            "StableTemperature": {
                "Pre-Selection": np.array(stableT)[:, 0],
                "Post-Selection": np.array(stableT)[:, 1],
            },
            "Heating": {
                "Pre-Selection": np.array(heating)[:, 0],
                "Post-Selection": np.array(heating)[:, 1],
            },
            "Condensing": {
                "Pre-Selection": np.array(condensing)[:, 0],
                "Post-Selection": np.array(condensing)[:, 1],
            },
            "StableDensity": {
                "Pre-Selection": np.array(stabledensity)[:, 0],
                "Post-Selection": np.array(stabledensity)[:, 1],
            },
            "Dispersing": {
                "Pre-Selection": np.array(dispersing)[:, 0],
                "Post-Selection": np.array(dispersing)[:, 1],
            },
            # "|B|Decreasing": {
            #     "Pre-Selection": np.array(bdecreasing)[:, 0],
            #     "Post-Selection": np.array(bdecreasing)[:, 1],
            # },
            # "Stable|B|": {
            #     "Pre-Selection": np.array(bstable)[:, 0],
            #     "Post-Selection": np.array(bstable)[:, 1],
            # },
            # "|B|Increasing": {
            #     "Pre-Selection": np.array(bincreasing)[:, 0],
            #     "Post-Selection": np.array(bincreasing)[:, 1],
            # },
        }

        for key, value in outinner.items():
            if (key == "T") or (key == "Rinner") or (key == "Router"):
                if key in list(out.keys()):
                    val = out[key]
                    val = val + value
                    out.update({key: val})
                else:
                    out.update({key: value})
            else:
                tmp = {}
                for k, v in value.items():
                    tmp.update({k: v})
                out.update({key: tmp})

    dict_of_df = {k: pd.DataFrame(v) for k, v in out.items()}
    df1 = pd.concat(dict_of_df, axis=1)

    df = df1.set_index("T")
    return df


def bars_plot(
    FlatDataDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    selectTime,
    snapRange,
    Tlst,
    DataSavepath,
    DPI=100,
    xsize = 7.0,
    ysize = 6.0,
    bottomParam = 0.35,
    barwidth=0.80,
    opacityPercentiles = 0.25,
    lineStyleMedian = "solid",
    lineStylePercentiles = "-.",
    colourmapMain = "plasma",
    shortSnapRangeBool=False,
    shortSnapRangeNumber=None,
    titleBool = True,
    separateLegend = False,
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
):
    colourmapMain = "plasma"
    # Input parameters path:
    TracersParamsPath = "TracersParams.csv"
    DataSavepathSuffix = f".h5"
    singleVals = ["Rinner", "Router", "T", "Snap", "Lookback"]

    snapRange = np.array(snapRange)
    for key in FlatDataDict.keys():
        FlatDataDict[key].update({"Ntracers": np.shape(FlatDataDict[key]["type"])[1]})
    print("Analyse Data!")

    timeAvDF = flat_analyse_time_averages(
        FlatDataDict,
        Tlst,
        snapRange,
        tlookback,
        TRACERSPARAMS,
        shortSnapRangeBool=shortSnapRangeBool,
        shortSnapRangeNumber=shortSnapRangeNumber,
    )

    # Save
    if (shortSnapRangeBool is False) & (shortSnapRangeNumber is None):
        savePath = DataSavepath + "_Time-Averages-Statistics-Table.csv"
    else:
        savePath = (
            DataSavepath
            + f"_Time-Averages-Statistics-Table_shortSnapRange-{int(shortSnapRangeNumber)}snaps.csv"
        )
    print("\n" + f"Saving Stats table .csv as {savePath}")

    timeAvDF.to_csv(savePath, index=False)

    # -------------------------------------------------------------------------------#
    #       Plot!!
    # -------------------------------------------------------------------------------#
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):

        plotDF = timeAvDF.loc[
            (timeAvDF["Rinner"] == rin)[0] & (timeAvDF["Router"] == rout)[0]
        ]

        cmap = matplotlib.cm.get_cmap(colourmapMain)
        colour = [cmap(float(ii) / float(len(Tlst))) for ii in range(len(Tlst))]

        ################################################################################
        #       split Plot
        ###############################################################################

        cols = plotDF.columns.values
        preDF = plotDF[cols[::2].tolist()]
        postDF = plotDF[cols[1::2].tolist()]

        newcols = {}
        for name in cols[::2]:
            newcols.update({name: name[0]})

        preDF = preDF.rename(columns=newcols)
        preDF = preDF.drop(columns="Rinner")
        preDF.columns = preDF.columns.droplevel(1)

        newcols = {}
        for name in cols[1::2]:
            newcols.update({name: name[0]})

        postDF = postDF.rename(columns=newcols)
        postDF = postDF.drop(columns="Router")
        postDF.columns = postDF.columns.droplevel(1)

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(xsize, ysize), sharey=True
        )

        preDF.T.plot.bar(width=barwidth,rot=0, ax=ax, color=colour, align='center')

        cmap = matplotlib.cm.get_cmap(colourmapMain)
        patchList = [matplotlib.patches.Patch(color = cmap(float(ii) / float(len(Tlst))), label = r"$T = 10^{%3.0f} K$" % (float(temp))) for ii, temp in enumerate(Tlst)]
        ax.legend(handles = patchList, loc="upper left", fontsize=fontsize)
        plt.xticks(rotation=90, ha="right", fontsize=fontsize)
        ax.tick_params(axis="both",which="both",labelsize=fontsize)
        ax.set_ylim(0.,100.)

        if titleBool is True:
            plt.title(
                r"Percentage of Tracers Ever Meeting Criterion Pre Selection at $t_{Lookback}$"
                + f"={selectTime:3.2f} Gyr"
                + "\n"
                + r"selected by $T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
                + r" and $%3.0f \leq R \leq %3.0f $ kpc " % (rin, rout),
                fontsize=fontsizeTitle,
            )

        plt.annotate(
            text="Ever Matched"+"\n"+"Feature",
            xy=(0.15, 0.02),
            xytext=(0.15, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=fontsize,
        )
        plt.annotate(
            text="",
            xy=(0.10, 0.01),
            xytext=(0.385, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.39, bottomParam),
            xytext=(0.39, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        plt.annotate(
            text="On Average"+"\n"+"Feature",
            xy=(0.40, 0.02),
            xytext=(0.40, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=fontsize,
        )
        plt.annotate(
            text="",
            xy=(0.395, 0.01),
            xytext=(0.570, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.575, bottomParam),
            xytext=(0.575, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        plt.annotate(
            text="-1 Snapshot"+"\n"+"Feature",
            xy=(0.70, 0.02),
            xytext=(0.70, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=fontsize,
        )
        plt.annotate(
            text="",
            xy=(0.580, 0.01),
            xytext=(0.95, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        fig.transFigure

        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="both",which="both",labelsize=fontsize)


        plt.grid(which="both", axis="y")
        plt.ylabel("% of Tracers Selected Following Feature", fontsize=fontsize)
        plt.tight_layout()

        if titleBool is True:
            plt.subplots_adjust(top=0.90, bottom=bottomParam, left=0.10,right=0.95)
        else:
            plt.subplots_adjust(bottom=bottomParam, left=0.10,right=0.95)
        if (shortSnapRangeBool is False) & (shortSnapRangeNumber is None):
            opslaan = (
                "./"
                + "MultiHalo"
                + "/"
                + f"{int(rin)}R{int(rout)}"
                + "/"
                + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{rin}R{rout}_Pre-Stats-Bars.pdf"
            )
        else:
            opslaan = (
                "./"
                + "MultiHalo"
                + "/"
                + f"{int(rin)}R{int(rout)}"
                + "/"
                + f"Tracers_MultiHalo_shortSnapRange-{int(shortSnapRangeNumber)}snaps_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{rin}R{rout}_Pre-Stats-Bars.pdf"
            )


        if (separateLegend == True):
            ax.get_legend().remove()

        fig.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()



        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(xsize, ysize), sharey=True
        )

        postDF.T.plot.bar(width=barwidth, rot=0, ax=ax, color=colour, align='center')

        cmap = matplotlib.cm.get_cmap(colourmapMain)
        patchList = [matplotlib.patches.Patch(color = cmap(float(ii) / float(len(Tlst))), label = r"$T = 10^{%3.0f} K$" % (float(temp))) for ii, temp in enumerate(Tlst)]
        ax.legend(handles = patchList, loc="upper left", fontsize=fontsize)

        plt.xticks(rotation=90, ha="right", fontsize=fontsize)
        ax.tick_params(axis="both",which="both",labelsize=fontsize)
        ax.set_ylim(0.,100.)

        if titleBool is True:
            plt.title(
                r"Percentage of Tracers Ever Meeting Criterion Post Selection at $t_{Lookback}$"
                + f"={selectTime:3.2f} Gyr"
                + "\n"
                + r"selected by $T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
                + r" and $%3.0f \leq R \leq %3.0f $ kpc " % (rin, rout),
                fontsize=fontsizeTitle,
            )

        plt.annotate(
            text="Ever Matched"+"\n"+"Feature",
            xy=(0.15, 0.02),
            xytext=(0.15, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=fontsize,
        )
        plt.annotate(
            text="",
            xy=(0.10, 0.01),
            xytext=(0.385, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.39, bottomParam),
            xytext=(0.39, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        plt.annotate(
            text="On Average"+"\n"+"Feature",
            xy=(0.40, 0.02),
            xytext=(0.40, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=fontsize,
        )
        plt.annotate(
            text="",
            xy=(0.395, 0.01),
            xytext=(0.570, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.575, bottomParam),
            xytext=(0.575, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        plt.annotate(
            text="+1 Snapshot"+"\n"+"Feature",
            xy=(0.70, 0.02),
            xytext=(0.70, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=fontsize,
        )
        plt.annotate(
            text="",
            xy=(0.580, 0.01),
            xytext=(0.95, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        fig.transFigure
        #     text="",
        #     xy=(0.10, bottomParam),
        #     xytext=(0.10, 0.05),
        #     arrowprops=dict(arrowstyle="-"),
        #     xycoords=fig.transFigure,
        #     annotation_clip=False,
        # )
        # plt.annotate(
        #     text="Ever Matched Feature",
        #     xy=(0.17, 0.02),
        #     xytext=(0.17, 0.02),
        #     textcoords=fig.transFigure,
        #     annotation_clip=False,
        #     fontsize=fontsize,
        # )
        # plt.annotate(
        #     text="",
        #     xy=(0.10, 0.01),
        #     xytext=(0.45, 0.01),
        #     arrowprops=dict(arrowstyle="<->"),
        #     xycoords=fig.transFigure,
        #     annotation_clip=False,
        # )
        # plt.annotate(
        #     text="",
        #     xy=(0.47, bottomParam),
        #     xytext=(0.47, 0.05),
        #     arrowprops=dict(arrowstyle="-"),
        #     xycoords=fig.transFigure,
        #     annotation_clip=False,
        # )
        #
        # plt.annotate(
        #     text="On Average Feature",
        #     xy=(0.48, 0.02),
        #     xytext=(0.48, 0.02),
        #     textcoords=fig.transFigure,
        #     annotation_clip=False,
        #     fontsize=fontsize,
        # )
        # plt.annotate(
        #     text="",
        #     xy=(0.47, 0.01),
        #     xytext=(0.73, 0.01),
        #     arrowprops=dict(arrowstyle="<->"),
        #     xycoords=fig.transFigure,
        #     annotation_clip=False,
        # )
        # plt.annotate(
        #     text="",
        #     xy=(0.74, bottomParam),
        #     xytext=(0.74, 0.05),
        #     arrowprops=dict(arrowstyle="-"),
        #     xycoords=fig.transFigure,
        #     annotation_clip=False,
        # )
        #
        # plt.annotate(
        #     text="+1 Snapshot Feature",
        #     xy=(0.75, 0.02),
        #     xytext=(0.75, 0.02),
        #     textcoords=fig.transFigure,
        #     annotation_clip=False,
        #     fontsize=fontsize,
        # )
        # plt.annotate(
        #     text="",
        #     xy=(0.74, 0.01),
        #     xytext=(0.95, 0.01),
        #     arrowprops=dict(arrowstyle="<->"),
        #     xycoords=fig.transFigure,
        #     annotation_clip=False,
        # )
        # plt.annotate(
        #     text="",
        #     xy=(0.95, bottomParam),
        #     xytext=(0.95, 0.05),
        #     arrowprops=dict(arrowstyle="-"),
        #     xycoords=fig.transFigure,
        #     annotation_clip=False,
        # )

        # fig.transFigure

        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(axis="both",which="both",labelsize=fontsize)
        plt.grid(which="both", axis="y")
        plt.ylabel("% of Tracers Selected Following Feature", fontsize=fontsize)
        plt.tight_layout()
        if titleBool is True:
            plt.subplots_adjust(top=0.90, bottom=bottomParam, left=0.10, right=0.95)
        else:
            plt.subplots_adjust(bottom=bottomParam, left=0.10, right=0.95)

        if (shortSnapRangeBool is False) & (shortSnapRangeNumber is None):
            opslaan = (
                "./"
                + "MultiHalo"
                + "/"
                + f"{int(rin)}R{int(rout)}"
                + "/"
                + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{rin}R{rout}_Post-Stats-Bars.pdf"
            )
        else:
            opslaan = (
                "./"
                + "MultiHalo"
                + "/"
                + f"{int(rin)}R{int(rout)}"
                + "/"
                + f"Tracers_MultiHalo_shortSnapRange-{int(shortSnapRangeNumber)}snaps_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{rin}R{rout}_Post-Stats-Bars.pdf"
            )

        if (separateLegend == True):
            # get handles and labels for reuse
            label_params = ax.get_legend_handles_labels()

            ax.get_legend().remove()

            lcol = len(Tlst)
            figl, axl = plt.subplots(figsize = (lcol*2.5,1))
            axl.axis(False)
            axl.legend(handles = patchList, ncol = lcol, loc="center", bbox_to_anchor=(0.5, 0.5), fontsize=fontsize)
            plt.tight_layout()
            figl.savefig("./"
                + "MultiHalo"
                + "/"
                + f"{int(rin)}R{int(rout)}"
                + "/"
                + f"Stats-Bars_Legend.pdf"
            )

        fig.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()
    return


################################################################################
##                  EXPERIMENTAL                                              ##
################################################################################


def hist_plot(
    dataDict,
    statsData,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    selectTime,
    snapRange,
    Tlst,
    logParameters,
    ylabel,
    titleBool,
    DPI = 75,
    xsize = 10.0,
    ysize = 24.0,
    opacityPercentiles = 0.25,
    lineStyleMedian = "solid",
    lineStylePercentiles = "-.",
    colourmapMain = "plasma",
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
    Nbins=150
):

    weightKey = "mass"
    xanalysisParam = "L"
    yanalysisParam = "R"

    xlimDict = {
        "T": {"xmin": 3.75, "xmax": 6.5},
        "R": {"xmin": 0, "xmax": 250},
        "n_H": {"xmin": -6.0, "xmax": 0.0},
        "B": {"xmin": -6.0, "xmax": 2.0},
        "vrad": {"xmin": -250.0, "xmax": 250.0},
        "gz": {"xmin": -4.0, "xmax": 1.0},
        "L": {"xmin": 0.0, "xmax": 5.0},
        "P_thermal": {"xmin": -1.0, "xmax": 7.0},
        "P_magnetic": {"xmin": -7.0, "xmax": 7.0},
        "P_kinetic": {"xmin": -1.0, "xmax": 8.0},
        "P_tot": {"xmin": -1.0, "xmax": 7.0},
        "Pthermal_Pmagnetic": {"xmin": -3.0, "xmax": 10.0},
        "tcool": {"xmin": -6.0, "xmax": 3.0},
        "theat": {"xmin": -4.0, "xmax": 4.0},
        "tff": {"xmin": -3.0, "xmax": 1.0},
        "tcool_tff": {"xmin": -6.0, "xmax": 3.0},
        "rho_rhomean": {"xmin": 0.0, "xmax": 8.0},
        "dens": {"xmin": -30.0, "xmax": -22.0},
        "ndens": {"xmin": -6.0, "xmax": 2.0},
    }

    whereSelect = int(
        np.where(np.array(snapRange) == TRACERSPARAMS["selectSnap"])[0][0]
    )

    selectSnaps = (
        snapRange[0:whereSelect:2]
        + [snapRange[whereSelect]]
        + snapRange[whereSelect + 2 :: 2]
    )

    tlookbackSelect = (
        tlookback.tolist()[0:whereSelect:2]
        + [tlookback.tolist()[whereSelect]]
        + tlookback.tolist()[whereSelect + 2 :: 2]
    )

    print("")
    print("Loading Data!")
    # Create a plot for each Temperature
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")

        fig, ax = plt.subplots(
            nrows=len(selectSnaps),
            ncols=len(Tlst),
            sharex=True,
            sharey=True,
            figsize=(xsize, ysize),
            dpi=DPI,
        )
        for (jj, snap) in enumerate(selectSnaps):
            print(f"Snap {snap}")
            yminlist = []
            ymaxlist = []
            patchList = []
            labelList = []

            for (ii, T) in enumerate(Tlst):
                FullDictKey = (f"T{float(T)}", f"{rin}R{rout}", f"{snap}")

                if len(Tlst) == 1:
                    currentAx = ax[jj]
                else:
                    currentAx = ax[jj, ii]

                whereGas = np.where(dataDict[FullDictKey]["type"] == 0)[0]

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                #   Figure 1: Full Cells Data
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                print(f"T{T} Sub-Plot!")

                ydataCells = dataDict[FullDictKey][yanalysisParam][whereGas]
                if yanalysisParam in logParameters:
                    ydataCells = np.log10(ydataCells)
                yminlist.append(np.min(ydataCells))
                ymaxlist.append(np.max(ydataCells))

                # Set lookback time array for each data point in y
                xdataCells = dataDict[FullDictKey][xanalysisParam][whereGas]
                if xanalysisParam in logParameters:
                    xdataCells = np.log10(xdataCells)

                massCells = dataDict[FullDictKey]["mass"][whereGas]
                weightDataCells = dataDict[FullDictKey][weightKey][whereGas] * massCells

                if weightKey == "mass":
                    finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                        xdataCells, ydataCells, bins=Nbins, weights=massCells
                    )
                else:
                    mhistCells, _, _ = np.histogram2d(
                        xdataCells, ydataCells, bins=Nbins, weights=massCells
                    )
                    histCells, xedgeCells, yedgeCells = np.histogram2d(
                        xdataCells, ydataCells, bins=Nbins, weights=weightDataCells
                    )

                    finalHistCells = histCells / mhistCells

                finalHistCells[finalHistCells == 0.0] = np.nan
                if weightKey in logParameters:
                    finalHistCells = np.log10(finalHistCells)
                finalHistCells = finalHistCells.T

                xcells, ycells = np.meshgrid(xedgeCells, yedgeCells)

                img1 = currentAx.pcolormesh(
                    xcells,
                    ycells,
                    finalHistCells,
                    cmap=colourmapMain,
                    rasterized=True,
                )

                if jj == 0:
                    currentAx.set_title(
                        r"$ 10^{%03.2f \pm %3.2f} K $"
                        % (float(T), TRACERSPARAMS["deltaT"]),
                        fontsize=fontsizeTitle,
                    )

                currentAx.annotate(
                    text=f"{tlookbackSelect[jj]:3.2f} Gyr",
                    xy=(0.10, 0.90),
                    xytext=(0.10, 0.90),
                    textcoords=currentAx.transAxes,
                    annotation_clip=False,
                    fontsize=fontsize,
                )
                currentAx.transAxes
        if titleBool is True:
            fig.suptitle(
                f"Cells Containing Tracers selected by: "
                + "\n"
                + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
                + r" and $%3.0f \leq R \leq %3.0f $ kpc " % (rin, rout)
                + "\n"
                + f" and selected at {selectTime:3.2f} Gyr",
                fontsize=fontsizeTitle,
            )

        # Only give 1 x-axis a label, as they sharex
        if len(Tlst) == 1:
            axis0 = ax[(len(selectSnaps) - 1) // 2]
            midax = ax[(len(selectSnaps) - 1) // 2]
        else:
            axis0 = ax[len(selectSnaps) - 1, (len(Tlst) - 1) // 2]
            midax = ax[(len(selectSnaps) - 1) // 2, 0]

        axis0.set_xlabel(ylabel[xanalysisParam], fontsize=fontsize)
        midax.set_ylabel(ylabel[yanalysisParam], fontsize=fontsize)

        plt.colorbar(img1, ax=ax.ravel().tolist(), orientation="vertical").set_label(
            label=ylabel[weightKey], size=fontsize
        )

        plt.setp(
            ax,
            ylim=(xlimDict[yanalysisParam]["xmin"], xlimDict[yanalysisParam]["xmax"]),
            xlim=(xlimDict[xanalysisParam]["xmin"], xlimDict[xanalysisParam]["xmax"]),
        )
        plt.tight_layout()
        if titleBool is True:
            plt.subplots_adjust(
                top=0.90, bottom=0.05, left=0.10, right=0.75, hspace=0.1, wspace=0.1
            )
        else:
            plt.subplots_adjust(bottom=0.05, left=0.10, right=0.75, hspace=0.1, wspace=0.1
            )

        opslaan = (
            "./"
            + "MultiHalo"
            + "/"
            + f"{int(rin)}R{int(rout)}"
            + "/"
            + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
            + f"{xanalysisParam}-{yanalysisParam}"
            + f"_Hist.pdf"
        )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()

    return


################################################################################


def medians_phases_plot(
    FlatDataDict,
    statsData,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    selectTime,
    snapRange,
    Tlst,
    logParameters,
    ylabel,
    SELECTEDHALOES,
    DPI = 75,
    xsize = 7.0,
    ysize = 6.0,
    opacityPercentiles = 0.25,
    lineStyleMedian = "solid",
    lineStylePercentiles = "-.",
    colourmapMain = "plasma",
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
    Nbins=100,
    titleBool = True,
    weightKey="mass",
    analysisParam="R",
):

    labelDict = ylabel

    xlimDict = {
        "mass": {"xmin": 5.0, "xmax": 9.0},
        "L": {"xmin": 3.5, "xmax": 4.5},
        "T": {"xmin": 3.75, "xmax": 6.5},
        "R": {"xmin": 0, "xmax": 400},
        "n_H": {"xmin": -6.0, "xmax": 1.0},
        "B": {"xmin": -6.0, "xmax": 2.0},
        "vrad": {"xmin": -400.0, "xmax": 400.0},
        "gz": {"xmin": -4.0, "xmax": 1.0},
        "P_thermal": {"xmin": -1.0, "xmax": 7.0},
        "P_magnetic": {"xmin": -7.0, "xmax": 7.0},
        "P_kinetic": {"xmin": -1.0, "xmax": 8.0},
        "P_tot": {"xmin": -1.0, "xmax": 7.0},
        "Pthermal_Pmagnetic": {"xmin": -3.0, "xmax": 10.0},
        "tcool": {"xmin": -6.0, "xmax": 3.0},
        "theat": {"xmin": -4.0, "xmax": 4.0},
        "tff": {"xmin": -3.0, "xmax": 1.0},
        "tcool_tff": {"xmin": -6.0, "xmax": 3.0},
        "rho_rhomean": {"xmin": 0.0, "xmax": 8.0},
        "dens": {"xmin": -30.0, "xmax": -22.0},
        "ndens": {"xmin": -6.0, "xmax": 2.0},
    }

    nHaloes = float(len(SELECTEDHALOES))

    # Create a plot for each Temperature
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")

        fig, ax = plt.subplots(
            nrows=len(Tlst),
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(xsize, ysize),
            dpi=DPI,
        )
        yminlist = []
        ymaxlist = []
        patchList = []
        labelList = []

        breakFlag = False
        for (ii, T) in enumerate(Tlst):
            FullDictKey = (f"T{float(T)}", f"{rin}R{rout}")

            if len(Tlst) == 1:
                currentAx = ax
            else:
                currentAx = ax[ii]

            whereGas = np.where(
                np.where(FlatDataDict[FullDictKey]["type"] == 0, True, False).all(
                    axis=0
                )
            )[0]

            selectKey = (f"T{Tlst[ii]}", f"{rin}R{rout}")
            plotData = statsData[selectKey].copy()
            # Temperature specific load path

            snapRange = np.array(
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
            selectionSnap = np.where(snapRange == int(TRACERSPARAMS["selectSnap"]))

            vline = tlookback[selectionSnap]

            # Get number of temperatures
            NTemps = float(len(Tlst))

            # Get temperature
            temp = TRACERSPARAMS["targetTLst"][ii]

            # Select a Temperature specific colour from colourmapMain

            # Get a colour for median and percentiles for a given temperature
            #   Have fiddled to move colours away from extremes of the colormap
            cmap = matplotlib.cm.get_cmap(colourmapMain)
            colour = cmap(float(ii) / float(len(Tlst)))

            loadPercentilesTypes = [
                analysisParam + "_" + str(percentile) + "%"
                for percentile in TRACERSPARAMS["percentiles"]
            ]
            LO = analysisParam + "_" + str(min(TRACERSPARAMS["percentiles"])) + "%"
            UP = analysisParam + "_" + str(max(TRACERSPARAMS["percentiles"])) + "%"
            median = analysisParam + "_" + "50.00%"

            if analysisParam in logParameters:
                for k, v in plotData.items():
                    plotData.update({k: np.log10(v)})

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

            midPercentile = math.floor(len(loadPercentilesTypes) / 2.0)
            percentilesPairs = zip(
                loadPercentilesTypes[:midPercentile],
                loadPercentilesTypes[midPercentile + 1 :],
            )
            for (LOO, UPP) in percentilesPairs:
                currentAx.plot(
                    tlookback,
                    plotData[UPP],
                    color="black",
                    lineStyle=lineStylePercentiles,
                )
                currentAx.plot(
                    tlookback,
                    plotData[LOO],
                    color="black",
                    lineStyle=lineStylePercentiles,
                )
            currentAx.plot(
                tlookback,
                plotData[median],
                label=r"$T = 10^{%3.0f} K$" % (float(temp)),
                color="black",
                lineStyle=lineStyleMedian,
            )

            currentAx.axvline(x=vline, c="red")

            currentAx.xaxis.set_minor_locator(AutoMinorLocator())
            currentAx.yaxis.set_minor_locator(AutoMinorLocator())
            currentAx.tick_params(axis="both",which="both",labelsize=fontsize)

            #

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            #   Figure 1: Full Cells Data
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            print(f"T{T} Sub-Plot!")

            ydat = {
                analysisParam: FlatDataDict[FullDictKey][analysisParam][:, whereGas]
            }
            if analysisParam in logParameters:
                ydat[analysisParam] = np.log10(ydat[analysisParam])

            tmp, whereReal = delete_nan_inf_axis(ydat, axis=0)

            # Set y data points.
            # Flip x and y and weightings' temporal ordering to match medians.
            ydataCells = tmp[analysisParam]
            whereReal = whereGas[whereReal[analysisParam]]
            ydataCells = np.flip(ydataCells, axis=0)
            if np.any(np.array(np.shape(ydataCells)) == 0):
                print("No Data! Skipping entry!")
                breakFlag = True
                continue

            ymin = np.nanmin(ydataCells)
            ymax = np.nanmax(ydataCells)
            if (
                (np.isinf(ymin) == True)
                or (np.isinf(ymax) == True)
                or (np.isnan(ymin) == True)
                or (np.isnan(ymax) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                breakFlag = True
                continue

            nDat = np.shape(ydataCells)[1]

            # Set lookback time array for each data point in y
            xdataCells = np.flip(
                np.tile(np.array(tlookback), nDat).reshape(nDat, -1).T, axis=0
            )

            massCells = np.flip(FlatDataDict[FullDictKey]["mass"][:, whereReal], axis=0) / nHaloes
            weightDataCells = np.flip(
                FlatDataDict[FullDictKey][weightKey][:, whereReal] * massCells, axis=0
            )

            if np.any(np.array(np.shape(xdataCells)) == 0):
                print("No Data! Skipping entry!")
                breakFlag = True
                continue

            xmin = np.nanmin(xdataCells)
            xmax = np.nanmax(xdataCells)
            if (
                (np.isinf(xmin) == True)
                or (np.isinf(xmax) == True)
                or (np.isnan(xmin) == True)
                or (np.isnan(xmax) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                breakFlag = True
                continue

            if np.any(np.array(np.shape(weightDataCells)) == 0):
                print("No Data! Skipping entry!")
                breakFlag = True
                continue
            wmin = np.nanmin(weightDataCells)
            wmax = np.nanmax(weightDataCells)
            if (
                (np.isinf(wmin) == True)
                or (np.isinf(wmax) == True)
                or (np.isnan(wmin) == True)
                or (np.isnan(wmax) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                breakFlag = True
                continue

            if np.any(np.array(np.shape(massCells)) == 0):
                print("No Data! Skipping entry!")
                breakFlag = True
                continue

            mmin = np.nanmin(massCells)
            mmax = np.nanmax(massCells)
            if (
                (np.isinf(mmin) == True)
                or (np.isinf(mmax) == True)
                or (np.isnan(mmin) == True)
                or (np.isnan(mmax) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                breakFlag = True
                continue

            xstep = np.array(
                [np.absolute(x1 - x2) for x1, x2 in zip(tlookback[1:], tlookback[:-1])]
            )
            xstep = np.append(xstep[0], xstep)

            xedges = np.flip(np.append(tlookback + xstep / 2.0, [0]), axis=0)

            if weightKey in list(xlimDict.keys()):
                yedges = np.linspace(
                    xlimDict[analysisParam]["xmin"], xlimDict[analysisParam]["xmax"], Nbins
                )
            else:
                yedges = np.linspace(
                    np.nanmin(ydataCells), np.nanmax(ydataCells), Nbins
                )

            if weightKey == "mass":
                finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                    xdataCells.flatten(),
                    ydataCells.flatten(),
                    bins=(xedges, yedges),
                    weights=massCells.flatten(),
                )
            else:
                mhistCells, _, _ = np.histogram2d(
                    xdataCells.flatten(),
                    ydataCells.flatten(),
                    bins=(xedges, yedges),
                    weights=massCells.flatten(),
                )
                histCells, xedgeCells, yedgeCells = np.histogram2d(
                    xdataCells.flatten(),
                    ydataCells.flatten(),
                    bins=(xedges, yedges),
                    weights=weightDataCells.flatten(),
                )

                finalHistCells = histCells / mhistCells

            finalHistCells[finalHistCells == 0.0] = np.nan
            if weightKey in logParameters:
                finalHistCells = np.log10(finalHistCells)
            finalHistCells = finalHistCells.T

            xcells, ycells = np.meshgrid(xedgeCells, yedgeCells)

            if weightKey in list(xlimDict.keys()):
                zmin = xlimDict[weightKey]["xmin"]
                zmax = xlimDict[weightKey]["xmax"]
                img1 = currentAx.pcolormesh(
                    xcells,
                    ycells,
                    finalHistCells,
                    cmap=colourmapMain,
                    vmin=zmin,
                    vmax=zmax,
                    rasterized=True,
                )
            else:
                img1 = currentAx.pcolormesh(
                    xcells,
                    ycells,
                    finalHistCells,
                    cmap=colourmapMain,
                    rasterized=True,
                )

            currentAx.set_title(
                r"$ 10^{%03.2f \pm %3.2f} K $ Tracers Data"
                % (float(T), TRACERSPARAMS["deltaT"]),
                fontsize=fontsizeTitle,
            )
        if breakFlag == True:
            print("Missing sub-plot! Skipping entry!")
            continue

        if titleBool is True:
            fig.suptitle(
                f"Cells Containing Tracers selected by: "
                + "\n"
                + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
                + r" and $%3.0f \leq R \leq %3.0f $ kpc " % (rin, rout)
                + "\n"
                + f" and selected at {vline[0]:3.2f} Gyr",
                fontsize=fontsizeTitle,
            )
        fig.colorbar(img1, ax=ax.ravel().tolist(), orientation="vertical").set_label(
            label=labelDict[weightKey], size=fontsize
        )

        # Only give 1 x-axis a label, as they sharex
        if len(Tlst) == 1:
            axis0 = ax
            midax = ax
        else:
            axis0 = ax[len(Tlst) - 1]
            midax = ax[(len(Tlst) - 1) // 2]

        axis0.set_xlabel(r"Lookback Time (Gyr)", fontsize=fontsize)
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

        custom_ylim = (xlimDict[analysisParam]["xmin"], xlimDict[analysisParam]["xmax"])
        plt.setp(
            ax,
            ylim=custom_ylim,
            xlim=(round(max(tlookback), 1), round(min(tlookback), 1)),
        )
        plt.tight_layout()
        if titleBool is True:
            plt.subplots_adjust(top=0.80, right=0.75, hspace=0.25)
        else:
            plt.subplots_adjust(right=0.75, hspace=0.25)

        opslaan = (
            "./"
            + "MultiHalo"
            + "/"
            + f"{int(rin)}R{int(rout)}"
            + "/"
            + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
            + f"{weightKey}-{analysisParam}"
            + f"_Medians+Phases.pdf"
        )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()

    return

def temperature_variation_plot(
    dataDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    logParameters,
    ylabel,
    titleBool,
    DPI=100,
    xsize = 7.0,
    ysize = 6.0,
    opacityPercentiles = 0.25,
    lineStyleMedian = "solid",
    lineStylePercentiles = "-.",
    colourmapMain = "plasma",
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
    StatsDataPathSuffix = ".csv"):

    statsData = {}

    print("")
    print("Loading Data!")
    # Create a plot for each Temperature
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")

        fig, ax = plt.subplots(
            nrows=1,#len(Tlst),
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(xsize, ysize),
            dpi=DPI,
        )
        yminlist = []
        ymaxlist = []
        patchList = []
        labelList = []

        analysisParam = "T"
        plotData = {}

        for ii in range(len(Tlst)):
            print(f"T{Tlst[ii]}")
            T = float(Tlst[ii])

            selectKey = (f"T{Tlst[ii]}", f"{rin}R{rout}")


            tempDiff = np.abs(np.log10(dataDict[selectKey][analysisParam][1:,:]) - np.log10(dataDict[selectKey][analysisParam][:-1,:]))

            xData = tlookback[1:]
            # Temperature specific load path

            selectionSnap = np.where(
                np.array(snapRange) == int(TRACERSPARAMS["selectSnap"])
            )

            vline = tlookback[selectionSnap]

            # Get number of temperatures
            NTemps = float(len(Tlst))

            # Get temperature
            temp = TRACERSPARAMS["targetTLst"][ii]

            # Select a Temperature specific colour from colourmapMain

            # Get a colour for median and percentiles for a given temperature
            #   Have fiddled to move colours away from extremes of the colormap
            cmap = matplotlib.cm.get_cmap(colourmapMain)
            colour = cmap(float(ii) / float(len(Tlst)))

            loadPercentilesTypes = [
                analysisParam + "_" + f"{percentile:4.2f}" + "%"
                for percentile in TRACERSPARAMS['percentiles']
            ]
            LO = analysisParam + "_" + f"{min(TRACERSPARAMS['percentiles']):4.2f}" + "%"
            UP = analysisParam + "_" + f"{max(TRACERSPARAMS['percentiles']):4.2f}"  + "%"
            median = analysisParam + "_" + "50.00%"

            for perc_key, percentile in zip(loadPercentilesTypes,TRACERSPARAMS["percentiles"]):
                plotData.update({perc_key: np.nanpercentile(tempDiff,percentile,axis=1)})

            # if analysisParam in logParameters:
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

            currentAx = ax


            midPercentile = math.floor(len(loadPercentilesTypes) / 2.0)
            percentilesPairs = zip(
                loadPercentilesTypes[:midPercentile],
                loadPercentilesTypes[midPercentile + 1 :],
            )
            for (LO, UP) in percentilesPairs:
                currentAx.fill_between(
                    xData,
                    plotData[UP],
                    plotData[LO],
                    facecolor=colour,
                    alpha=opacityPercentiles,
                    interpolate=False,
                )
            currentAx.plot(
                xData,
                plotData[median],
                label=r"$T = 10^{%3.0f} K$" % (float(temp)),
                color=colour,
                lineStyle=lineStyleMedian,
            )

            currentAx.axvline(x=vline, c="red")

            currentAx.xaxis.set_minor_locator(AutoMinorLocator())
            currentAx.yaxis.set_minor_locator(AutoMinorLocator())
            currentAx.tick_params(axis="both",which="both",labelsize=fontsize)
            #
            # #Delete text string for first y_axis label for all but last panel
            # plt.gcf().canvas.draw()
            # if (int(ii)<len(Tlst)-1):
            #     plt.setp(currentAx.get_xticklabels(),visible = False)
            #     plt.gcf().canvas.draw()
            #     # STOP160IF

            # plot_patch = matplotlib.patches.Patch(color=colour)
            # plot_label = r"$T = 10^{%3.0f} K$" % (float(temp))
            # patchList.append(plot_patch)
            # labelList.append(plot_label)

            if titleBool is True:
                fig.suptitle(
                    f"Cells Containing Tracers selected by: "
                    + "\n"
                    + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
                    + r" and $%3.0f \leq R \leq %3.0f $ kpc " % (rin, rout)
                    + "\n"
                    + f" and selected at {vline[0]:3.2f} Gyr",
                    fontsize=fontsizeTitle,
                )

            saveKey = (f"T{T}", f"{rin}R{rout}")
            statsData.update({saveKey:plotData.copy()})



        # Only give 1 x-axis a label, as they sharex
        axis0 = ax
        midax = ax


        ylabelhere = r"$|\Delta \left(Log_{10}(\mathrm{T})\right)|$" + "\n" + "Temperature "+ "Variation (K)"
        axis0.set_xlabel("Lookback Time (Gyr)", fontsize=fontsize)
        midax.set_ylabel(ylabelhere, fontsize=fontsize)
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
            xlim=(round(max(xData), 1), round(min(xData), 1)),
        )
        axis0.legend(loc="upper right",fontsize=fontsize)

        plt.tight_layout()
        if titleBool is True:
            plt.subplots_adjust(top=0.875, hspace=0.1,left=0.15)
        else:
            plt.subplots_adjust(hspace=0.1,left=0.15)

        opslaan = (
            "./"
            + "MultiHalo"
            + "/"
            + f"{int(rin)}R{int(rout)}"
            + "/"
            + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
            + f"Temperature_Variation.pdf"
        )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()


    #### Output statsDF as .csv ####

    save_statistics_csv(
        statsData, TRACERSPARAMS, Tlst, snapRange[1:], savePathInsert = "Temperature_Variation_"
    )

    return
