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

xsize = 10.0
ysize = 12.0
DPI = 100

# Set style options
opacityPercentiles = 0.25
lineStyleMedian = "solid"
lineStylePercentiles = "-."

ageUniverse = 13.77  # [Gyr]

colourmapMain = "plasma"

def medians_plot(dataDict,statsData,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst,DataSavepathSuffix = f".h5",TracersParamsPath = "TracersParams.csv",TracersMasterParamsPath ="TracersParamsMaster.csv",SelectedHaloesPath = "TracersSelectedHaloes.csv"):


    logParameters = [
        "dens",
        "rho_rhomean",
        "csound",
        "T",
        "n_H",
        "B",
        "gz",
        "L",
        "P_thermal",
        "P_magnetic",
        "P_kinetic",
        "P_tot",
        "Pthermal_Pmagnetic",
        "tcool",
        "theat",
        "tcross",
        "tff",
        "tcool_tff",
    ]
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
        "Pthermal_Pmagnetic": r"$P_{thermal}/P_{magnetic}$",
        "tcool": r"Cooling Time [$Gyr$]",
        "theat": r"Heating Time [$Gyr$]",
        "tcross": r"Sound Crossing Cell Time [$Gyr$]",
        "tff": r"Free Fall Time [$Gyr$]",
        "tcool_tff": r"Cooling Time over Free Fall Time",
        "csound": r"Sound Speed",
        "rho_rhomean": r"Density over Average Universe Density",
        "dens": r"Density [$g$ $cm^{-3}$]",
        "ndens": r"Number density [# $cm^{-3}$]",
    }

    for entry in logParameters:
        ylabel[entry] = r"Log10 " + ylabel[entry]

    # ==============================================================================#

    for analysisParam in saveParams:
        print("")
        print(f"Starting {analysisParam} Sub-plots!")


        print("")
        print("Loading Data!")
        # Create a plot for each Temperature
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            print(f"{rin}R{rout}")

            fig, ax = plt.subplots(
                nrows=len(Tlst), ncols=1, sharex=True, figsize=(xsize, ysize), dpi=DPI
            )
            yminlist = []
            ymaxlist = []
            for ii in range(len(Tlst)):
                print(f"T{Tlst[ii]}")
                T = float(Tlst[ii])

                selectKey = (f"T{Tlst[ii]}",f"{rin}R{rout}")
                plotData = statsData[selectKey].copy()
                # Temperature specific load path

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

                vline = tlookback[selectionSnap]

                # Get number of temperatures
                NTemps = float(len(Tlst))

                # Get temperature
                temp = TRACERSPARAMS["targetTLst"][ii]

                # Select a Temperature specific colour from colourmap

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
                    loadPercentilesTypes[midPercentile + 1:],
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

                currentAx.axvline(x=vline, c="red")

                currentAx.xaxis.set_minor_locator(AutoMinorLocator())
                currentAx.yaxis.set_minor_locator(AutoMinorLocator())
                currentAx.tick_params(which="both")

                currentAx.set_ylabel(ylabel[analysisParam], fontsize=10)

                plot_patch = matplotlib.patches.Patch(color=colour)
                plot_label = r"$T = 10^{%3.2f} K$" % (float(temp))
                currentAx.legend(
                    handles=[plot_patch], labels=[plot_label], loc="upper right"
                )

                fig.suptitle(
                    f"Cells Containing Tracers selected by: "
                    + "\n"
                    + r"$T = 10^{n \pm %05.2f} K$" % (TRACERSPARAMS["deltaT"])
                    + r" and $%05.2f \leq R \leq %05.2f kpc $" % (rin, rout)
                    + "\n"
                    + f" and selected at {vline[0]:3.2f} Gyr"
                    + f" weighted by mass",
                    fontsize=12,
                )

            # Only give 1 x-axis a label, as they sharex
            if len(Tlst) == 1:
                axis0 = ax
            else:
                axis0 = ax[len(Tlst) - 1]

            axis0.set_xlabel(r"Lookback Time [$Gyrs$]", fontsize=10)
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
            plt.setp(ax, ylim=custom_ylim)
            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace=0.0)
            opslaan = (
                    "./"
                    + 'MultiHalo'
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

def persistant_temperature_plot(dataDict,statsData,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst,DataSavepathSuffix = f".h5",TracersParamsPath = "TracersParams.csv",TracersMasterParamsPath ="TracersParamsMaster.csv",SelectedHaloesPath = "TracersSelectedHaloes.csv"):
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
                min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"] + 1)),
            )

            rangeSet = [snapRangeLow, snapRangeHi]

            for snapRange in rangeSet:
                key = (f"T{T}", f"{rin}R{rout}", f"{int(TRACERSPARAMS['selectSnap'])}")
                SelectedTracers = dataDict[key]["trid"][ParentsIndices]

                for snap in snapRange:
                    key = (f"T{T}", f"{rin}R{rout}", f"{int(snap)}")

                    whereGas = np.where(dataDict[key]["type"] == 0)[0]

                    data = dataDict[key]["T"][whereGas]

                    whereTrids = np.where(np.isin(dataDict[key]["trid"], SelectedTracers))
                    Parents = dataDict[key]["prid"][whereTrids]

                    whereCells = np.where(np.isin(dataDict[key]["id"][whereGas], Parents))

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
            tmpYarray = np.flip(np.take_along_axis(tmpYarray,ind_sorted,axis=0), axis=0)
            tmpXarray = np.flip(np.take_along_axis(tmpXarray,ind_sorted,axis=0), axis=0)


            # Add the full list of snaps data to temperature dependent dictionary.
            Xdata.update({f"T{T}": tmpXarray})
            Ydata.update({f"T{T}": tmpYarray})

        # ==============================================================================#
        #           PLOT!!
        # ==============================================================================#

        fig, ax = plt.subplots(
            nrows=len(Tlst), ncols=1, sharex=True, figsize=(xsize, ysize), dpi=DPI
        )

        # Create a plot for each Temperature
        for ii in range(len(Tlst)):
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
                tlookback, tmpMinData, plotYdata, facecolor=colour, alpha=0.25, interpolate=False
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

            currentAx.set_ylabel(
                r"Percentage Tracers Still at $ T = 10^{%05.2f \pm %05.2f} K$"
                % (T, TRACERSPARAMS["deltaT"]),
                fontsize=10,
            )
            currentAx.set_ylim(ymin=datamin, ymax=datamax)

            fig.suptitle(
                f"Percentage Tracers Still at Selection Temperature "
                + r"$T = 10^{n \pm %05.2f} K$" % (TRACERSPARAMS["deltaT"])
                + "\n"
                + r" selected at $%05.2f \leq R \leq %05.2f kpc $" % (rin, rout)
                + f" and selected at {vline[0]:3.2f} Gyr",
                fontsize=12,
            )
            currentAx.legend(loc="upper right")

        # Only give 1 x-axis a label, as they sharex
        if len(Tlst) == 1:
            axis0 = ax
        else:
            axis0 = ax[len(Tlst) - 1]

        axis0.set_xlabel(r"Lookback Time [$Gyrs$]", fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, wspace=0.005)
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

def within_temperature_plot(dataDict,statsData,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst,DataSavepathSuffix = f".h5",TracersParamsPath = "TracersParams.csv",TracersMasterParamsPath ="TracersParamsMaster.csv",SelectedHaloesPath = "TracersSelectedHaloes.csv"):
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
                min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"] + 1)),
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
            tmpYarray = np.flip(np.take_along_axis(tmpYarray,ind_sorted,axis=0), axis=0)
            tmpXarray = np.flip(np.take_along_axis(tmpXarray,ind_sorted,axis=0), axis=0)

            # Add the full list of snaps data to temperature dependent dictionary.
            Xdata.update({f"T{T}": tmpXarray})
            Ydata.update({f"T{T}": tmpYarray})

        # ==============================================================================#
        #           PLOT!!
        # ==============================================================================#

        fig, ax = plt.subplots(
            nrows=len(Tlst), ncols=1, sharex=True, figsize=(xsize, ysize), dpi=DPI
        )

        # Create a plot for each Temperature
        for ii in range(len(Tlst)):
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

            vline = tlookback [selectionSnap]

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
                tlookback , tmpMinData, plotYdata, facecolor=colour, alpha=0.25, interpolate=False
            )

            currentAx.plot(
                tlookback ,
                plotYdata,
                label=r"$T = 10^{%3.0f} K$" % (float(temp)),
                color=colour,
                lineStyle="-",
            )

            currentAx.axvline(x=vline, c="red")
            currentAx.xaxis.set_minor_locator(AutoMinorLocator())
            currentAx.yaxis.set_minor_locator(AutoMinorLocator())
            currentAx.tick_params(which="both")

            currentAx.set_ylabel(
                r"Percentage Tracers Still at $ T = 10^{%05.2f \pm %05.2f} K$"
                % (T, TRACERSPARAMS["deltaT"]),
                fontsize=10,
            )
            currentAx.set_ylim(ymin=datamin, ymax=datamax)

            fig.suptitle(
                f"Percentage Tracers Within Selection Temperature Range "
                + r"$T = 10^{n \pm %05.2f} K$" % (TRACERSPARAMS["deltaT"])
                + "\n"
                + r" selected at $%05.2f \leq R \leq %05.2f kpc $" % (rin, rout)
                + f" and selected at {vline[0]:3.2f} Gyr",
                fontsize=12,
            )
            currentAx.legend(loc="upper right")

        # Only give 1 x-axis a label, as they sharex
        if len(Tlst) == 1:
            axis0 = ax
        else:
            axis0 = ax[len(Tlst) - 1]

        axis0.set_xlabel(r"Lookback Time [$Gyrs$]", fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, wspace=0.005)
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

def stacked_pdf_plot(dataDict,statsData,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst,DataSavepathSuffix = f".h5",TracersParamsPath = "TracersParams.csv",TracersMasterParamsPath ="TracersParamsMaster.csv",SelectedHaloesPath = "TracersSelectedHaloes.csv",Nbins = 75):

    ageUniverse = 13.77  # [Gyr]
    opacity = 0.75
    selectColour = "red"
    selectStyle = "-."
    selectWidth = 4
    percentileLO = 1.0
    percentileUP = 99.0


    import seaborn as sns
    import scipy.stats as stats


    # Entered parameters to be saved from
    #   n_H, B, R, T
    #   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
    # saveParams = ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

    logParameters = [
        "dens",
        "rho_rhomean",
        "csound",
        "T",
        "n_H",
        "B",
        "gz",
        "L",
        "P_thermal",
        "P_magnetic",
        "P_kinetic",
        "P_tot",
        "Pthermal_Pmagnetic",
        "tcool",
        "theat",
        "tcross",
        "tff",
        "tcool_tff",
    ]
    # "rho_rhomean,dens,T,R,n_H,B,vrad,gz,L,P_thermal,P_magnetic,P_kinetic,P_tot,tcool,theat,csound,tcross,tff,tcool_tff"
    xlabel = {
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
        "Pthermal_Pmagnetic": r"$P_{thermal}/P_{magnetic}$",
        "tcool": r"Cooling Time [$Gyr$]",
        "theat": r"Heating Time [$Gyr$]",
        "tcross": r"Sound Crossing Cell Time [$Gyr$]",
        "tff": r"Free Fall Time [$Gyr$]",
        "tcool_tff": r"Cooling Time over Free Fall Time",
        "csound": r"Sound Speed",
        "rho_rhomean": r"Density over Average Universe Density",
        "dens": r"Density [$g$ $cm^{-3}$]",
        "ndens": r"Number density [# $cm^{-3}$]",
    }

    for entry in logParameters:
        xlabel[entry] = r"Log10 " + xlabel[entry]

    for dataKey in saveParams:
        print(f"{dataKey}")
        # Create a plot for each Temperature
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            print(f"{rin}R{rout}")
            for ii in range(len(Tlst)):
                print(f"T{Tlst[ii]}")

                selectKey = (f"T{Tlst[ii]}",f"{rin}R{rout}")
                # Temperature specific load path
                plotData = statsData[selectKey]

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
                                TRACERSPARAMS["finalSnap"] + 1, TRACERSPARAMS["snapMax"] + 1
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
                        (dataDict[dictkey]["type"] == 4) & (dataDict[dictkey]["age"] >= 0.0)
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

                    # Select a Temperature specific colour from colourmap
                    cmap = matplotlib.cm.get_cmap(colourmapMain)
                    if int(snap) == int(TRACERSPARAMS["selectSnap"]):
                        colour = selectColour
                        lineStyle = selectStyle
                        linewidth = selectWidth
                    else:
                        sRange = int(
                            min(
                                TRACERSPARAMS["finalSnap"] + 1, TRACERSPARAMS["snapMax"] + 1
                            )
                        ) - int(TRACERSPARAMS["snapMin"])
                        colour = cmap(((float(jj)) / (sRange)))
                        lineStyle = "-"
                        linewidth = 2

                    tmpdict = {"x": data, "y": weights}
                    df = pd.DataFrame(tmpdict)

                    LO = weighted_percentile(
                        data=data, weights=weights, perc=percentileLO, key="LO"
                    )
                    UP = weighted_percentile(
                        data=data, weights=weights, perc=percentileUP, key="UP"
                    )

                    xmin = xminlist.append(LO)  # np.nanmin(data)
                    xmax = xmaxlist.append(UP)
                    # Draw the densities in a few steps
                    # ,
                    sns.kdeplot(
                        df["x"],
                        weights=df["y"],
                        ax=currentAx,
                        bw_adjust=0.5,
                        clip = (xmin,xmax),
                        alpha=opacity,
                        fill=True,
                        lw=linewidth,
                        color=colour,
                        linestyle=lineStyle,
                        shade =True
                    )
                    currentAx.axhline(
                        y=0,
                        lw=linewidth,
                        linestyle=lineStyle,
                        color=colour,
                        clip_on=False,
                    )

                    currentAx.set_yticks([])
                    currentAx.set_ylabel("")
                    currentAx.set_xlabel(xlabel[dataKey], fontsize=15)
                    sns.despine(bottom=True, left=True)



                xmin = np.nanmin(xminlist)
                xmax = np.nanmax(xmaxlist)

                plt.xlim(xmin, xmax)
                #
                plot_label = r"$T = 10^{%3.2f} K$" % (float(T))
                plt.text(
                    0.75,
                    0.95,
                    plot_label,
                    horizontalalignment="left",
                    verticalalignment="center",
                    transform=fig.transFigure,
                    wrap=True,
                    bbox=dict(facecolor="blue", alpha=0.2),
                    fontsize=15,
                )

                time_label = r"Lookback Time [Gyr]"
                plt.text(
                    0.10,
                    0.475,
                    time_label,
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=fig.transFigure,
                    wrap=True,
                    fontsize=15,
                )
                plt.arrow(
                    0.10,
                    0.525,
                    0.00,
                    +0.225,
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
                    + r"$T = 10^{%05.2f \pm %05.2f} K$" % (T, TRACERSPARAMS["deltaT"])
                    + r" and $%05.2f \leq R \leq %05.2f kpc $" % (rin, rout)
                    + "\n"
                    + f" and selected at {selectTime:3.2f} Gyr"
                    + f" weighted by mass"
                    + "\n"
                    + f"{percentileLO:3.2f}% to {percentileUP:3.2f}% Mass Weighted Percentiles Shown",
                    fontsize=12,
                )
                # ax.axvline(x=vline, c='red')

                plt.tight_layout()
                plt.subplots_adjust(top=0.90, hspace=-0.25)

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
