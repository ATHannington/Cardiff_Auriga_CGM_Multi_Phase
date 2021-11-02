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

xsize = 5.0
ysize = 6.0
DPI = 100

# Set style options
opacityPercentiles = 0.25
lineStyleMedian = "solid"
lineStylePercentiles = "-."

ageUniverse = 13.77  # [Gyr]

colourmapMain = "plasma"


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
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
):

    tmpxsize = xsize + 2.0

    for analysisParam in saveParams:
        print("")
        print(f"Starting {analysisParam} Sub-plots!")

        print("")
        print("Loading Data!")
        # Create a plot for each Temperature
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            print(f"{rin}R{rout}")

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
            patchList = []
            labelList = []
            for ii in range(len(Tlst)):
                print(f"T{Tlst[ii]}")
                T = float(Tlst[ii])

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

                if len(Tlst) == 1:
                    currentAx = ax
                else:
                    currentAx = ax[ii]

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
                    label=r"$T = 10^{%3.2f} K$" % (float(temp)),
                    color=colour,
                    lineStyle=lineStyleMedian,
                )

                currentAx.axvline(x=vline, c="red")

                currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                currentAx.tick_params(which="both")
                #
                # #Delete text string for first y_axis label for all but last panel
                # plt.gcf().canvas.draw()
                # if (int(ii)<len(Tlst)-1):
                #     plt.setp(currentAx.get_xticklabels(),visible = False)
                #     plt.gcf().canvas.draw()
                #     # STOP160IF

                plot_patch = matplotlib.patches.Patch(color=colour)
                plot_label = r"$T = 10^{%3.2f} K$" % (float(temp))
                patchList.append(plot_patch)
                labelList.append(plot_label)

                fig.suptitle(
                    f"Cells Containing Tracers selected by: "
                    + "\n"
                    + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
                    + r" and $%3.2f \leq R \leq %3.2f kpc $" % (rin, rout)
                    + "\n"
                    + f" and selected at {vline[0]:3.2f} Gyr",
                    fontsize=12,
                )

            # Only give 1 x-axis a label, as they sharex
            if len(Tlst) == 1:
                axis0 = ax
                midax = ax
            else:
                axis0 = ax[len(Tlst) - 1]
                midax = ax[(len(Tlst) - 1) // 2]

            axis0.set_xlabel(r"Lookback Time [$Gyrs$]", fontsize=10)
            midax.set_ylabel(ylabel[analysisParam], fontsize=10)
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
                xlim=(round(max(tlookback)), round(min(tlookback))),
            )
            fig.legend(
                handles=patchList,
                labels=labelList,
                loc="center right",
                facecolor="white",
                framealpha=1,
            )
            plt.tight_layout()
            plt.subplots_adjust(top=0.875, right=0.80, hspace=0.1)

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
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

    return


def persistant_temperature_plot(
    dataDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
):
    Ydata = {}
    Xdata = {}
    # Loop over temperatures in targetTLst and grab Temperature specific subset of tracers and relevant data
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")
        for T in TRACERSPARAMS["targetTLst"]:
            print("")
            print(f"Starting T{T} analysis")
            key = (f"T{T}", f"{rin}R{rout}", f"{int(TRACERSPARAMS['selectSnap'])}")

            whereGas = np.where(dataDict[key]["type"] == 0)[0]
            data = dataDict[key]["T"][whereGas]

            whereSelect = np.where(
                (data >= 1.0 * 10 ** (T - TRACERSPARAMS["deltaT"]))
                & (data <= 1.0 * 10 ** (T + TRACERSPARAMS["deltaT"]))
            )

            selectedCells = dataDict[key]["id"][whereSelect]

            ParentsIndices = np.where(np.isin(dataDict[key]["prid"], selectedCells))

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

            for snapRange in rangeSet:
                key = (f"T{T}", f"{rin}R{rout}", f"{int(TRACERSPARAMS['selectSnap'])}")
                SelectedTracers = dataDict[key]["trid"][ParentsIndices]

                for snap in snapRange:
                    key = (f"T{T}", f"{rin}R{rout}", f"{int(snap)}")

                    whereGas = np.where(dataDict[key]["type"] == 0)[0]

                    data = dataDict[key]["T"][whereGas]

                    whereTrids = np.where(
                        np.isin(dataDict[key]["trid"], SelectedTracers)
                    )
                    Parents = dataDict[key]["prid"][whereTrids]

                    whereCells = np.where(
                        np.isin(dataDict[key]["id"][whereGas], Parents)
                    )

                    data = data[whereCells]

                    selected = np.where(
                        (data >= 1.0 * 10 ** (T - TRACERSPARAMS["deltaT"]))
                        & (data <= 1.0 * 10 ** (T + TRACERSPARAMS["deltaT"]))
                    )

                    selectedData = data[selected]

                    selectedIDs = dataDict[key]["id"][whereGas]
                    selectedIDs = selectedIDs[selected]

                    selectedCellsIndices = np.where(
                        np.isin(dataDict[key]["prid"], selectedIDs)
                    )

                    finalTrids = dataDict[key]["trid"][selectedCellsIndices]

                    SelectedTracers = finalTrids

                    nTracers = len(finalTrids)

                    # Append the data from this snapshot to a temporary list
                    tmpXdata.append(dataDict[key]["Lookback"][0])
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
            nrows=len(Tlst),
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(xsize, ysize),
            dpi=DPI,
        )

        # Create a plot for each Temperature
        for ii in range(len(Tlst)):
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

            if len(Tlst) == 1:
                currentAx = ax
            else:
                currentAx = ax[ii]

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
            currentAx.tick_params(which="both")

            currentAx.set_ylim(ymin=datamin, ymax=datamax)
            currentAx.set_xlim(
                xmin=round(max(tlookback)), xmax=round(min(tlookback))
            )

            fig.suptitle(
                f"Percentage Tracers Still at \n Selection Temperature "
                + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
                + "\n"
                + r" selected at $%3.2f \leq R \leq %3.2f kpc $" % (rin, rout)
                + f" and selected at {vline[0]:3.2f} Gyr",
                fontsize=12,
            )
            currentAx.legend(loc="upper right")

        # Only give 1 x-axis a label, as they sharex
        if len(Tlst) == 1:
            axis0 = ax
            midax = ax
        else:
            axis0 = ax[len(Tlst) - 1]
            midax = ax[(len(Tlst) - 1) // 2]

        axis0.set_xlabel(r"Lookback Time [$Gyrs$]", fontsize=10)
        midax.set_ylabel(
            r"Percentage Tracers Still at Selection Temperature",
            fontsize=10,
        )
        plt.tight_layout(h_pad=0.0)
        opslaan = (
            "./"
            + "MultiHalo"
            + "/"
            + f"{int(rin)}R{int(rout)}"
            + "/"
            + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_T"
            + f"_PersistantTemperature.pdf"
        )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()

    return


def within_temperature_plot(
    dataDict,
    TRACERSPARAMS,
    saveParams,
    tlookback,
    snapRange,
    Tlst,
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
):
    Ydata = {}
    Xdata = {}

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

            for snapRange in rangeSet:
                for snap in snapRange:
                    key = (f"T{T}", f"{rin}R{rout}", f"{int(snap)}")

                    whereGas = np.where(dataDict[key]["type"] == 0)[0]

                    data = dataDict[key]["T"][whereGas]

                    selected = np.where(
                        (data >= 1.0 * 10 ** (T - TRACERSPARAMS["deltaT"]))
                        & (data <= 1.0 * 10 ** (T + TRACERSPARAMS["deltaT"]))
                    )

                    ParentsIndices = np.where(
                        np.isin(dataDict[key]["prid"], dataDict[key]["id"][selected])
                    )

                    trids = dataDict[key]["trid"][ParentsIndices]

                    nTracers = len(trids)

                    # Append the data from this snapshot to a temporary list
                    tmpXdata.append(dataDict[key]["Lookback"][0])
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
            nrows=len(Tlst),
            ncols=1,
            sharex=True,
            sharey=True,
            figsize=(xsize, ysize),
            dpi=DPI,
        )

        # Create a plot for each Temperature
        for ii in range(len(Tlst)):
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

            if len(Tlst) == 1:
                currentAx = ax
            else:
                currentAx = ax[ii]

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
                label=r"$T = 10^{%3.2f} K$" % (float(temp)),
                color=colour,
                lineStyle="-",
            )

            currentAx.axvline(x=vline, c="red")
            currentAx.xaxis.set_minor_locator(AutoMinorLocator())
            currentAx.yaxis.set_minor_locator(AutoMinorLocator())
            currentAx.tick_params(which="both")

            currentAx.set_ylim(ymin=datamin, ymax=datamax)
            currentAx.set_xlim(
                xmin=round(max(tlookback)), xmax=round(min(tlookback))
            )

            fig.suptitle(
                f"Percentage Tracers Within \n Selection Temperature Range "
                + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
                + "\n"
                + r" selected at $%3.2f \leq R \leq %3.2f kpc $" % (rin, rout)
                + f" and selected at {vline[0]:3.2f} Gyr",
                fontsize=12,
            )
            currentAx.legend(loc="upper right")

        # Only give 1 x-axis a label, as they sharex
        if len(Tlst) == 1:
            axis0 = ax
            midax = ax
        else:
            axis0 = ax[len(Tlst) - 1]
            midax = ax[(len(Tlst) - 1) // 2]

        axis0.set_xlabel(r"Lookback Time [$Gyrs$]", fontsize=10)
        midax.set_ylabel(
            r"Percentage Tracers Still Within Selection Temperature Range",
            fontsize=10,
        )

        plt.tight_layout(h_pad=0.0)
        opslaan = (
            "./"
            + "MultiHalo"
            + "/"
            + f"{int(rin)}R{int(rout)}"
            + "/"
            + f"Tracers_MultiHalo_selectSnap{int(TRACERSPARAMS['selectSnap'])}_T"
            + f"_WithinTemperature.pdf"
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
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
):
    xsize = 5.0
    ysize = 10.0
    ageUniverse = 13.77  # [Gyr]
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
                    f"{int(TRACERSPARAMS['selectSnap'])}",
                )
                selectTime = abs(dataDict[selectKey]["Lookback"][0])

                xmaxlist = []
                xminlist = []
                dataList = []
                weightsList = []
                snapRange = [
                    xx
                    for xx in range(
                        int(TRACERSPARAMS["snapMin"]),
                        int(
                            min(
                                TRACERSPARAMS["finalSnap"] + 1,
                                TRACERSPARAMS["snapMax"] + 1,
                            )
                        ),
                    )
                ]
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
                    dictkey = (f"T{T}", f"{rin}R{rout}", f"{int(snap)}")

                    whereGas = np.where(dataDict[dictkey]["type"] == 0)

                    dataDict[dictkey]["age"][
                        np.where(np.isnan(dataDict[dictkey]["age"]) == True)
                    ] = 0.0

                    whereStars = np.where(
                        (dataDict[dictkey]["type"] == 4)
                        & (dataDict[dictkey]["age"] >= 0.0)
                    )

                    NGas = len(dataDict[dictkey]["type"][whereGas])
                    NStars = len(dataDict[dictkey]["type"][whereStars])
                    Ntot = NGas + NStars

                    # Percentage in stars
                    percentage = (float(NStars) / (float(Ntot))) * 100.0

                    data = dataDict[dictkey][dataKey][whereGas]
                    weights = dataDict[dictkey]["mass"][whereGas]

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

                    ### Use a rolling window over the last outofrangePercentOfxRange % of data to attempt to create peaks in kdeplot at ends ###

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
                    currentAx.set_xlabel(xlabel[dataKey], fontsize=10)
                    sns.despine(bottom=True, left=True)

                xmin = np.nanmin(xminlist)
                xmax = np.nanmax(xmaxlist)

                plt.xlim(xmin, xmax)
                #
                # plot_label = r"$T = 10^{%3.2f} K$" % (float(T))
                # plt.text(
                #     0.75,
                #     0.95,
                #     plot_label,
                #     horizontalalignment="left",
                #     verticalalignment="center",
                #     transform=fig.transFigure,
                #     wrap=True,
                #     bbox=dict(facecolor="blue", alpha=0.2),
                #     fontsize=10,
                # )

                time_label = r"Time [Gyr]"
                plt.text(
                    0.08,
                    0.525,
                    time_label,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=fig.transFigure,
                    wrap=True,
                    fontsize=10,
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

                fig.suptitle(
                    f"PDF of Cells Containing Tracers selected by: "
                    + "\n"
                    + r"$T = 10^{%3.2f \pm %3.2f} K$" % (T, TRACERSPARAMS["deltaT"])
                    + r" and $%3.2f \leq R \leq %3.2f kpc $" % (rin, rout)
                    + "\n"
                    + f" and selected at {selectTime:3.2f} Gyr",
                    fontsize=12,
                )
                # ax.axvline(x=vline, c='red')

                plt.tight_layout()
                plt.subplots_adjust(top=0.90, bottom=0.05, hspace=-0.25)

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
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
    Nbins=250,
):
    """
    Author: A. T. Hannington
    Created: 21/07/2020

    Known Bugs:
        pandas read_csv loading data as nested dict. . Have added flattening to fix

    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    xsize = 10.0
    ysize = 5.0
    fontsize = 10

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
    labelDict = {
        "mass": r"Log10 Mass per pixel [$M/M_{\odot}$]",
        "gz": r"Log10 Average Metallicity per pixel [$Z/Z_{\odot}$]",
        "tcool": r"Log10 Cooling Time per pixel [$Gyr$]",
        "tcool_tff": r"Cooling Time over Free Fall Time",
    }

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
                    figsize=(xsize * 2, ysize),
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
                        r"Log10 Density [$\rho / \langle \rho \rangle $]",
                        fontsize=fontsize,
                    )
                    currentAx.set_ylabel(r"Log10 Temperatures [$K$]", fontsize=fontsize)

                    currentAx.set_ylim(ymin, ymax)
                    currentAx.set_xlim(xmin, xmax)

                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    colour = cmap(float(ii) / float(len(Tlst)))

                    plot_patch = matplotlib.patches.Patch(color=colour)
                    plot_label = r"$T = 10^{%3.2f} K$" % (float(T))
                    currentAx.legend(
                        handles=[plot_patch], labels=[plot_label], loc="upper left"
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
                        fontsize=12,
                    )
                    currentAx.set_aspect("auto")

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                #   Temperature Figure: Finishing up
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                fig.suptitle(
                    f"Temperature Density Diagram, weighted by {weightKey}"
                    + f" at {currentTime:3.2f} Gyr"
                    + "\n"
                    + f"Tracers Data, selected at {selectTime:3.2f} Gyr as being"
                    + r" $%3.2f \leq R \leq %3.2f kpc $" % (rin, rout)
                    + r" and temperatures "
                    + r"$ 10^{n \pm %3.2f} K $" % (TRACERSPARAMS["deltaT"]),
                    fontsize=12,
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

    gas = []
    heating = []
    cooling = []
    smallTchange = []
    aboveZ = []
    belowZ = []
    inflow = []
    statflow = []
    outflow = []
    halo0 = []
    unbound = []
    otherHalo = []
    noHalo = []
    stars = []
    wind = []
    ism = []
    ptherm = []
    pmag = []
    tcool = []
    tff = []

    out = {}
    preselectInd = np.where(snapRange < int(TRACERSPARAMS["selectSnap"]))[0]
    postselectInd = np.where(snapRange > int(TRACERSPARAMS["selectSnap"]))[0]
    selectInd = np.where(snapRange == int(TRACERSPARAMS["selectSnap"]))[0]
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

            print("Heating & Cooling")
            epsilonT = float(TRACERSPARAMS["deltaT"])  # [k]

            # Select where GAS FOREVER ONLY tracers meet condition FOR THE LAST 2 SNAPSHOTS PRIOR TO SELECTION
            rowspre, colspre = np.where(
                np.log10(FlatDataDict[Tkey]["T"][:, whereGas][pre, :][-1:])
                - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                > (epsilonT)
            )
            # Calculate the number of these unique tracers compared to the total number
            coolingpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )

            rowspost, colspost = np.where(
                np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][post, :][:1])
                > (epsilonT)
            )

            coolingpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )

            rowspre, colspre = np.where(
                np.log10(FlatDataDict[Tkey]["T"][:, whereGas][pre, :][-1:])
                - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                < (-1.0 * epsilonT)
            )
            heatingpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )

            rowspost, colspost = np.where(
                np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][post, :][:1])
                < (-1.0 * epsilonT)
            )
            heatingpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )

            rowspre, colspre = np.where(
                (
                    np.log10(FlatDataDict[Tkey]["T"][:, whereGas][pre, :][-1:])
                    - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                    <= (0 + epsilonT)
                )
                & (
                    np.log10(FlatDataDict[Tkey]["T"][:, whereGas][pre, :][-1:])
                    - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                    >= (0 - epsilonT)
                )
            )
            smallTpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)

            rowspost, colspost = np.where(
                (
                    np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                    - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][post, :][:1])
                    <= (0.0 + epsilonT)
                )
                & (
                    np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                    - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][post, :][:1])
                    >= (0.0 - epsilonT)
                )
            )
            smallTpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            # Add data to internal database lists

            cooling.append([coolingpre, coolingpost])
            heating.append([heatingpre, heatingpost])
            smallTchange.append([smallTpre, smallTpost])
            #
            # print("Pthermal_Pmagnetic 1 ")
            # rowspre, colspre = np.where(
            #     FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:,whereGas][pre, :][-2:] > 1.0
            # )
            # pthermpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # rowspost, colspost = np.where(
            #     FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:,whereGas][post, :][:2] > 1.0
            # )
            # pthermpost = (
            #     100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # )
            # rowspre, colspre = np.where(
            #     FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:,whereGas][pre, :][-2:] < 1.0
            # )
            # pmagpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # rowspost, colspost = np.where(
            #     FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:,whereGas][post, :][:2] < 1.0
            # )
            # pmagpost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # ptherm.append([pthermpre, pthermpost])
            # pmag.append([pmagpre, pmagpost])

            # print("tcool_tff 10 ")
            # rowspre, colspre = np.where(
            #     FlatDataDict[Tkey]["tcool_tff"][:,whereGas][pre, :][-2:] > 10.0
            # )
            # tcoolpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # rowspost, colspost = np.where(
            #     FlatDataDict[Tkey]["tcool_tff"][:,whereGas][post, :][:2] > 10.0
            # )
            # tcoolpost = (
            #     100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # )
            # rowspre, colspre = np.where(
            #     FlatDataDict[Tkey]["tcool_tff"][:,whereGas][pre, :][-2:] < 10.0
            # )
            # tffpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # rowspost, colspost = np.where(
            #     FlatDataDict[Tkey]["tcool_tff"][:,whereGas][post, :][:2] < 10.0
            # )
            # tffpost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # tcool.append([pthermpre, pthermpost])
            # tff.append([tffpre, tffpost])
            #
            print("Z")
            # Select FOREVER GAS ONLY tracers' specific parameter data and mass weights PRIOR TO SELECTION

            data = FlatDataDict[Tkey]["gz"][:, whereGas][pre, :]
            weights = FlatDataDict[Tkey]["mass"][:, whereGas][pre, :]
            zPreDat = []
            # For each tracers, calculate the mass weighted average of specific parameter for all selected snapshots
            for (dat, wei) in zip(data.T, weights.T):
                zPreDat.append(np.nanmedian(dat))
            zPreDat = np.array(zPreDat)

            data = FlatDataDict[Tkey]["gz"][:, whereGas][post, :]
            weights = FlatDataDict[Tkey]["mass"][:, whereGas][post, :]
            zPostDat = []
            for (dat, wei) in zip(data.T, weights.T):
                zPostDat.append(np.nanmedian(dat))
            zPostDat = np.array(zPostDat)

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

            print("Radial-Flow")
            data = FlatDataDict[Tkey]["vrad"][:, whereGas][pre, :]
            weights = FlatDataDict[Tkey]["mass"][:, whereGas][pre, :]
            vradPreDat = []
            for (dat, wei) in zip(data.T, weights.T):
                vradPreDat.append(np.nanmedian(dat))
            vradPreDat = np.array(vradPreDat)

            data = FlatDataDict[Tkey]["vrad"][:, whereGas][post, :]
            weights = FlatDataDict[Tkey]["mass"][:, whereGas][post, :]
            vradPostDat = []
            for (dat, wei) in zip(data.T, weights.T):
                vradPostDat.append(np.nanmedian(dat))
            vradPostDat = np.array(vradPostDat)

            epsilon = 50.0

            colspre = np.where(vradPreDat < 0.0 - epsilon)[0]
            inflowpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            colspost = np.where(vradPostDat < 0.0 - epsilon)[0]
            inflowpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            inflow.append([inflowpre, inflowpost])

            colspre = np.where(
                (vradPreDat >= 0.0 - epsilon) & (vradPreDat <= 0.0 + epsilon)
            )[0]
            statflowpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            colspost = np.where(
                (vradPostDat >= 0.0 - epsilon) & (vradPostDat <= 0.0 + epsilon)
            )[0]
            statflowpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            statflow.append([statflowpre, statflowpost])

            colspre = np.where(vradPreDat > 0.0 + epsilon)[0]
            outflowpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            colspost = np.where(vradPostDat > 0.0 + epsilon)[0]
            outflowpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            outflow.append([outflowpre, outflowpost])

            print("Halo0")
            rowspre, colspre = np.where(
                FlatDataDict[Tkey]["SubHaloID"][:, whereGas][pre, :]
                == int(TRACERSPARAMS["haloID"])
            )
            halo0pre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            rowspost, colspost = np.where(
                FlatDataDict[Tkey]["SubHaloID"][:, whereGas][post, :]
                == int(TRACERSPARAMS["haloID"])
            )
            halo0post = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            halo0.append([halo0pre, halo0post])

            print("Unbound")
            rowspre, colspre = np.where(
                FlatDataDict[Tkey]["SubHaloID"][:, whereGas][pre, :] == -1
            )
            unboundpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            rowspost, colspost = np.where(
                FlatDataDict[Tkey]["SubHaloID"][:, whereGas][post, :] == -1
            )
            unboundpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            unbound.append([unboundpre, unboundpost])

            print("OtherHalo")
            rowspre, colspre = np.where(
                (
                    FlatDataDict[Tkey]["SubHaloID"][:, whereGas][pre, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["SubHaloID"][:, whereGas][pre, :] != -1)
                & (
                    np.isnan(FlatDataDict[Tkey]["SubHaloID"][:, whereGas][pre, :])
                    == False
                )
            )
            otherHalopre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            rowspost, colspost = np.where(
                (
                    FlatDataDict[Tkey]["SubHaloID"][:, whereGas][post, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["SubHaloID"][:, whereGas][post, :] != -1)
                & (
                    np.isnan(FlatDataDict[Tkey]["SubHaloID"][:, whereGas][post, :])
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
                    FlatDataDict[Tkey]["SubHaloID"][:, whereGas][pre, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["SubHaloID"][:, whereGas][pre, :] != -1)
                & (
                    np.isnan(FlatDataDict[Tkey]["SubHaloID"][:, whereGas][pre, :])
                    == True
                )
            )
            noHalopre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            rowspost, colspost = np.where(
                (
                    FlatDataDict[Tkey]["SubHaloID"][:, whereGas][post, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["SubHaloID"][:, whereGas][post, :] != -1)
                & (
                    np.isnan(FlatDataDict[Tkey]["SubHaloID"][:, whereGas][post, :])
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
                (FlatDataDict[Tkey]["R"][pre, :] <= 25.0)
                & (FlatDataDict[Tkey]["sfr"][pre, :] > 0.0)
            )
            ismpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["R"][post, :] <= 25.0)
                & (FlatDataDict[Tkey]["sfr"][post, :] > 0.0)
            )
            ismpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracersAll)
            )
            ism.append([ismpre, ismpost])

        outinner = {
            "Rinner": dfdat["Rinner"],
            "Router": dfdat["Router"],
            "T": dfdat[
                "T"
            ],  # "%Gas": {"Pre-Selection" : np.array(gas)[:,0],"Post-Selection" : np.array(gas)[:,1]} , \
            "%Halo0": {
                "Pre-Selection": np.array(halo0)[:, 0],
                "Post-Selection": np.array(halo0)[:, 1],
            },
            "%Unbound": {
                "Pre-Selection": np.array(unbound)[:, 0],
                "Post-Selection": np.array(unbound)[:, 1],
            },
            "%OtherHalo": {
                "Pre-Selection": np.array(otherHalo)[:, 0],
                "Post-Selection": np.array(otherHalo)[:, 1],
            },
            "%NoHalo": {
                "Pre-Selection": np.array(noHalo)[:, 0],
                "Post-Selection": np.array(noHalo)[:, 1],
            },
            "%Stars": {
                "Pre-Selection": np.array(stars)[:, 0],
                "Post-Selection": np.array(stars)[:, 1],
            },
            "%Wind": {
                "Pre-Selection": np.array(wind)[:, 0],
                "Post-Selection": np.array(wind)[:, 1],
            },
            "%ISM": {
                "Pre-Selection": np.array(ism)[:, 0],
                "Post-Selection": np.array(ism)[:, 1],
            },
            "%Inflow": {
                "Pre-Selection": np.array(inflow)[:, 0],
                "Post-Selection": np.array(inflow)[:, 1],
            },
            "%Radially-Static": {
                "Pre-Selection": np.array(statflow)[:, 0],
                "Post-Selection": np.array(statflow)[:, 1],
            },
            "%Outflow": {
                "Pre-Selection": np.array(outflow)[:, 0],
                "Post-Selection": np.array(outflow)[:, 1],
            },
            "%>3/4(Z_solar)": {
                "Pre-Selection": np.array(aboveZ)[:, 0],
                "Post-Selection": np.array(aboveZ)[:, 1],
            },
            "%<3/4(Z_solar)": {
                "Pre-Selection": np.array(belowZ)[:, 0],
                "Post-Selection": np.array(belowZ)[:, 1],
            },
            "%Heating": {
                "Pre-Selection": np.array(heating)[:, 0],
                "Post-Selection": np.array(heating)[:, 1],
            },
            "%Cooling": {
                "Pre-Selection": np.array(cooling)[:, 0],
                "Post-Selection": np.array(cooling)[:, 1],
            },
            "%SmallDelta(T)": {
                "Pre-Selection": np.array(smallTchange)[:, 0],
                "Post-Selection": np.array(smallTchange)[:, 1],
            },
            # "%(Ptherm_Pmagn)Above1": {
            #     "Pre-Selection": np.array(ptherm)[:, 0],
            #     "Post-Selection": np.array(ptherm)[:, 1],
            # },
            # "%(Ptherm_Pmagn)Below1": {
            #     "Pre-Selection": np.array(pmag)[:, 0],
            #     "Post-Selection": np.array(pmag)[:, 1],
            # },
            # "%(tcool_tff)Above10": {
            #     "Pre-Selection": np.array(tcool)[:, 0],
            #     "Post-Selection": np.array(tcool)[:, 1],
            # },
            # "%(tcool_tff)Below10": {
            #     "Pre-Selection": np.array(tff)[:, 0],
            #     "Post-Selection": np.array(tff)[:, 1],
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
    shortSnapRangeBool=False,
    shortSnapRangeNumber=None,
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
):
    xsize = 15.0
    ysize = 5.0
    DPI = 100
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
            nrows=1, ncols=1, figsize=(int(xsize / 2.0), ysize), sharey=True
        )

        preDF.T.plot.bar(rot=0, ax=ax, color=colour)

        legendLabels = [r"$10^{%3.2f}$" % (float(temp)) for temp in Tlst]
        ax.legend(legendLabels, loc="center left", title="T [K]", fontsize=10)
        plt.xticks(rotation=90, ha="right", fontsize=10)
        plt.title(
            r"Percentage of Tracers Ever Meeting Criterion Pre Selection at $t_{Lookback}$"
            + f"={selectTime:3.2f} Gyr"
            + "\n"
            + r"selected by $T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
            + r" and $%3.2f \leq R \leq %3.2f kpc $" % (rin, rout),
            fontsize=12,
        )

        plt.annotate(
            text="",
            xy=(0.10, 0.30),
            xytext=(0.10, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="Ever Matched Feature",
            xy=(0.17, 0.02),
            xytext=(0.17, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=10,
        )
        plt.annotate(
            text="",
            xy=(0.10, 0.01),
            xytext=(0.45, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.47, 0.30),
            xytext=(0.47, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        plt.annotate(
            text="Median Matched Feature",
            xy=(0.48, 0.02),
            xytext=(0.48, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=10,
        )
        plt.annotate(
            text="",
            xy=(0.47, 0.01),
            xytext=(0.73, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.74, 0.30),
            xytext=(0.74, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        plt.annotate(
            text="-1 Snapshot Feature",
            xy=(0.75, 0.02),
            xytext=(0.75, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=10,
        )
        plt.annotate(
            text="",
            xy=(0.74, 0.01),
            xytext=(0.90, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.90, 0.30),
            xytext=(0.90, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        fig.transFigure

        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both")
        plt.grid(which="both", axis="y")
        plt.ylabel("% of Tracers Selected Following Feature")
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.30, left=0.10, right=0.90)
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
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()

        fig, ax = plt.subplots(
            nrows=1, ncols=1, figsize=(int(xsize / 2.0), ysize), sharey=True
        )

        postDF.T.plot.bar(rot=0, ax=ax, color=colour)

        legendLabels = [r"$10^{%3.2f}$" % (float(temp)) for temp in Tlst]
        ax.legend(legendLabels, loc="center left", title="T [K]", fontsize=10)
        plt.xticks(rotation=90, ha="right", fontsize=10)
        plt.title(
            r"Percentage of Tracers Ever Meeting Criterion Post Selection at $t_{Lookback}$"
            + f"={selectTime:3.2f} Gyr"
            + "\n"
            + r"selected by $T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
            + r" and $%3.2f \leq R \leq %3.2f kpc $" % (rin, rout),
            fontsize=12,
        )

        plt.annotate(
            text="",
            xy=(0.10, 0.30),
            xytext=(0.10, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="Ever Matched Feature",
            xy=(0.17, 0.02),
            xytext=(0.17, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=10,
        )
        plt.annotate(
            text="",
            xy=(0.10, 0.01),
            xytext=(0.45, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.47, 0.30),
            xytext=(0.47, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        plt.annotate(
            text="Median Matched Feature",
            xy=(0.48, 0.02),
            xytext=(0.48, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=10,
        )
        plt.annotate(
            text="",
            xy=(0.47, 0.01),
            xytext=(0.73, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.74, 0.30),
            xytext=(0.74, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        plt.annotate(
            text="+1 Snapshot Feature",
            xy=(0.75, 0.02),
            xytext=(0.75, 0.02),
            textcoords=fig.transFigure,
            annotation_clip=False,
            fontsize=10,
        )
        plt.annotate(
            text="",
            xy=(0.74, 0.01),
            xytext=(0.90, 0.01),
            arrowprops=dict(arrowstyle="<->"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )
        plt.annotate(
            text="",
            xy=(0.90, 0.30),
            xytext=(0.90, 0.05),
            arrowprops=dict(arrowstyle="-"),
            xycoords=fig.transFigure,
            annotation_clip=False,
        )

        fig.transFigure

        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both")
        plt.grid(which="both", axis="y")
        plt.ylabel("% of Tracers Selected Following Feature")
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.30, left=0.10, right=0.90)

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
        plt.savefig(opslaan, dpi=DPI, transparent=False)
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
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
    Nbins=150,
    DPI=75,
):

    tmpysize = ysize * 4.0
    tmpxsize = xsize * 2.0
    weightKey = "mass"
    xanalysisParam = "L"
    yanalysisParam = "R"

    fontsize = 10
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
            figsize=(tmpxsize, tmpysize),
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
                        fontsize=fontsize + 2,
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

        fig.suptitle(
            f"Cells Containing Tracers selected by: "
            + "\n"
            + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
            + r" and $%3.2f \leq R \leq %3.2f kpc $" % (rin, rout)
            + "\n"
            + f" and selected at {selectTime:3.2f} Gyr",
            fontsize=12,
        )

        # Only give 1 x-axis a label, as they sharex
        if len(Tlst) == 1:
            axis0 = ax[(len(selectSnaps) - 1) // 2]
            midax = ax[(len(selectSnaps) - 1) // 2]
        else:
            axis0 = ax[len(selectSnaps) - 1, (len(Tlst) - 1) // 2]
            midax = ax[(len(selectSnaps) - 1) // 2, 0]

        axis0.set_xlabel(ylabel[xanalysisParam], fontsize=10)
        midax.set_ylabel(ylabel[yanalysisParam], fontsize=10)

        plt.colorbar(img1, ax=ax.ravel().tolist(), orientation="vertical").set_label(
            label=ylabel[weightKey], size=fontsize
        )

        plt.setp(
            ax,
            ylim=(xlimDict[yanalysisParam]["xmin"], xlimDict[yanalysisParam]["xmax"]),
            xlim=(xlimDict[xanalysisParam]["xmin"], xlimDict[xanalysisParam]["xmax"]),
        )
        plt.tight_layout()
        plt.subplots_adjust(
            top=0.90, bottom=0.05, left=0.10, right=0.75, hspace=0.1, wspace=0.1
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
    DataSavepathSuffix=f".h5",
    TracersParamsPath="TracersParams.csv",
    TracersMasterParamsPath="TracersParamsMaster.csv",
    SelectedHaloesPath="TracersSelectedHaloes.csv",
    Nbins=150,
    DPI=75,
    weightKey="L",
    analysisParam="R",
):

    tmpxsize = xsize + 2.0
    fontsize = 10

    labelDict = {
        "mass": r"Log10 Mass per pixel [$M/M_{\odot}$]",
        "gz": r"Log10 Average Metallicity per pixel [$Z/Z_{\odot}$]",
        "tcool": r"Log10 Cooling Time per pixel [$Gyr$]",
        "tcool_tff": r"Cooling Time over Free Fall Time",
        "L": r"Specific Angular Momentum [$kpc$ $km$ $s^{-1}$]",
    }

    xlimDict = {
        "L": {"xmin": 3.5, "xmax": 4.5},
        "T": {"xmin": 3.75, "xmax": 6.5},
        "R": {"xmin": 0, "xmax": 400},
        "n_H": {"xmin": -6.0, "xmax": 0.0},
        "B": {"xmin": -6.0, "xmax": 2.0},
        "vrad": {"xmin": -250.0, "xmax": 250.0},
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
    print("")
    print("Loading Data!")
    # Create a plot for each Temperature
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")

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
        patchList = []
        labelList = []

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

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            #   Figure 1: Full Cells Data
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            print(f"T{T} Sub-Plot!")

            zmin = xlimDict[weightKey]["xmin"]
            zmax = xlimDict[weightKey]["xmax"]

            ydat = {
                analysisParam: FlatDataDict[FullDictKey][analysisParam][:, whereGas]
            }

            ydat, whereReal = delete_nan_inf_axis(ydat, axis=0)

            # Set y data points.
            # Flip x and y and weightings' temporal ordering to match medians.
            ydataCells = ydat[analysisParam]
            whereReal = whereGas[whereReal[analysisParam]]
            if analysisParam in logParameters:
                ydataCells = np.log10(ydataCells)
            ydataCells = np.flip(ydataCells, axis=0)

            nDat = np.shape(ydataCells)[1]

            # Set lookback time array for each data point in y
            xdataCells = np.flip(
                np.tile(np.array(tlookback), nDat).reshape(nDat, -1).T, axis=0
            )

            massCells = np.flip(FlatDataDict[FullDictKey]["mass"][:, whereReal], axis=0)
            weightDataCells = np.flip(
                FlatDataDict[FullDictKey][weightKey][:, whereReal] * massCells, axis=0
            )

            if weightKey == "mass":
                finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                    xdataCells.flatten(),
                    ydataCells.flatten(),
                    bins=[len(snapRange) - 1, Nbins],
                    weights=massCells.flatten(),
                )
            else:
                mhistCells, _, _ = np.histogram2d(
                    xdataCells.flatten(),
                    ydataCells.flatten(),
                    bins=[len(snapRange) - 1, Nbins],
                    weights=massCells.flatten(),
                )
                histCells, xedgeCells, yedgeCells = np.histogram2d(
                    xdataCells.flatten(),
                    ydataCells.flatten(),
                    bins=[len(snapRange) - 1, Nbins],
                    weights=weightDataCells.flatten(),
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
                vmin=zmin,
                vmax=zmax,
                rasterized=True,
            )

            currentAx.set_title(
                r"$ 10^{%03.2f \pm %3.2f} K $ Tracers Data"
                % (float(T), TRACERSPARAMS["deltaT"]),
                fontsize=12,
            )

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

            if len(Tlst) == 1:
                currentAx = ax
            else:
                currentAx = ax[ii]

            midPercentile = math.floor(len(loadPercentilesTypes) / 2.0)
            percentilesPairs = zip(
                loadPercentilesTypes[:midPercentile],
                loadPercentilesTypes[midPercentile + 1 :],
            )
            for (LO, UP) in percentilesPairs:
                currentAx.plot(
                    tlookback,
                    plotData[UP],
                    color="black",
                    lineStyle=lineStylePercentiles,
                )
                currentAx.plot(
                    tlookback,
                    plotData[LO],
                    color="black",
                    lineStyle=lineStylePercentiles,
                )
            currentAx.plot(
                tlookback,
                plotData[median],
                label=r"$T = 10^{%3.2f} K$" % (float(temp)),
                color="black",
                lineStyle=lineStyleMedian,
            )

            currentAx.axvline(x=vline, c="red")

            currentAx.xaxis.set_minor_locator(AutoMinorLocator())
            currentAx.yaxis.set_minor_locator(AutoMinorLocator())
            currentAx.tick_params(which="both")
            #

        fig.suptitle(
            f"Cells Containing Tracers selected by: "
            + "\n"
            + r"$T = 10^{n \pm %3.2f} K$" % (TRACERSPARAMS["deltaT"])
            + r" and $%3.2f \leq R \leq %3.2f kpc $" % (rin, rout)
            + "\n"
            + f" and selected at {vline[0]:3.2f} Gyr",
            fontsize=12,
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

        axis0.set_xlabel(r"Lookback Time [$Gyrs$]", fontsize=10)
        midax.set_ylabel(ylabel[analysisParam], fontsize=10)
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
        plt.setp(ax, ylim=custom_ylim)
        plt.tight_layout()
        plt.subplots_adjust(top=0.80, right=0.75, hspace=0.25)

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
