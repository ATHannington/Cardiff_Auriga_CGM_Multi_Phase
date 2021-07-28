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
# Input parameters path:
TracersParamsPath = "TracersParams.csv"
TracersMasterParamsPath = "TracersParamsMaster.csv"
SelectedHaloesPath = "TracersSelectedHaloes.csv"

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

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersMasterParamsPath)

# Load Halo Selection Data
SELECTEDHALOES, HALOPATHS = load_haloes_selected(HaloPathBase = TRACERSPARAMS['savepath'] ,SelectedHaloesPath=SelectedHaloesPath)

saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

DataSavepathSuffix = f".h5"

snapRange = [snap for snap in range(
        int(TRACERSPARAMS["snapMin"]),
        min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
        1)]
# ==============================================================================#
mergedDict, saveParams =  multi_halo_merge(SELECTEDHALOES,
                            HALOPATHS,
                            DataSavepathSuffix,
                            snapRange,
                            Tlst,
                            TracersParamsPath
                            )
# ==============================================================================#
#           PLOT!!
# ==============================================================================#
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



tmpstatsData = {}
for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
    for ii in range(len(Tlst)):
        T = Tlst[ii]
        key = (f'T{Tlst[ii]}',f"{rin}R{rout}")
        for snap in snapRange:
            selectKey = (f'T{Tlst[ii]}',f"{rin}R{rout}",f"{snap}")
            dat = save_statistics(
                    v,
                    T,
                    rin,
                    rout,
                    snapNumber=snap,
                    TRACERSPARAMS=TRACERSPARAMS,
                    saveParams = saveParams,
                    DataSavepath=None,
                    MiniDataPathSuffix=".csv",
                    saveBool=False
            )
            if k in list(tmpstatsData.keys()) :

                for subkey,vals in dat.items():
                    if subkey in list(tmpstatsData.keys()):

                        tmpstatsData[k][subkey] = np.concatenate((tmpstatsData[k][subkey],dat[subkey]),axis=None)
                    else:
                        tmpstatsData[k].update(dat)
            else:
                tmpstatsData.update({k:dat})

statsData = {}
for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
    for ii in range(len(Tlst)):
        key = (f'T{Tlst[ii]}',f"{rin}R{rout}")
        for snap in snapRange:
            selectKey = (f'T{Tlst[ii]}',f"{rin}R{rout}",f"{snap}")
            if key in list(statsData.keys()):

                statsData[key] = np.concatenate((statsData[key],tmpstatsData[selectKey]),axis=None)
            else:
                statsData.update({key : tmpstatsData[selectKey]})


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

        axis0.set_xlabel(r"Age of Universe [$Gyrs$]", fontsize=10)
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
                + f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
                + analysisParam
                + f"_Medians.pdf"
        )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)
        plt.close()
