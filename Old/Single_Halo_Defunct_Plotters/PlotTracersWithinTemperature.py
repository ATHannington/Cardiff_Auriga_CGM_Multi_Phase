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
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
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

ageUniverse = 13.77  # [Gyr]

colourmapMain = "plasma"

# Input parameters path:
TracersParamsPath = "TracersParams.csv"

# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersParamsPath)

DataSavepathSuffix = f".h5"

print("Loading data!")

dataDict = {}

dataDict = full_dict_hdf5_load(DataSavepath, TRACERSPARAMS, DataSavepathSuffix)

saveHalo = (TRACERSPARAMS["savepath"].split("/"))[-2]

print("Getting Tracer Data!")
Ydata = {}
Xdata = {}

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

    tlookback.append(dataDict[key]["Lookback"][0])

tlookback = np.array(tlookback)

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
        tmpYarray = np.flip(np.take_along_axis(tmpYarray, ind_sorted, axis=0), axis=0)
        tmpXarray = np.flip(np.take_along_axis(tmpXarray, ind_sorted, axis=0), axis=0)

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
            linestyle="-",
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
        + saveHalo
        + "/"
        + f"{int(rin)}R{int(rout)}"
        + "/"
        + f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_T"
        + f"_WithinTemperature.pdf"
    )
    plt.savefig(opslaan, dpi=DPI, transparent=False)
    print(opslaan)
    plt.close()
