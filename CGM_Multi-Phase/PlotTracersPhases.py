"""
Author: A. T. Hannington
Created: 21/07/2020

Known Bugs:
    pandas read_csv loading data as nested dict. . Have added flattening to fix

"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *
import h5py
import multiprocessing as mp
from functools import reduce


Nbins = 250
xsize = 20.0
ysize = 10.0
fontsize = 15
DPI = 100

ageUniverse = 13.77  # [Gyr]

# Input parameters path:
TracersParamsPath = "TracersParams.csv"

colourmap = "inferno_r"
colourmapMain = "plasma"

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

# ==============================================================================#
#       USER DEFINED PARAMETERS
# ==============================================================================#
# Input parameters path:
TracersParamsPath = "TracersParams.csv"

# File types for data save.
#   Mini: small median and percentiles data
#   Full: full FullDict data
MiniDataPathSuffix = f".h5"
FullDataPathSuffix = f".h5"

# Lazy Load switch. Set to False to save all data (warning, pickle file may explode)
lazyLoadBool = True

# Number of cores to run on:
n_processes = 4

# ==============================================================================#
#       Prepare for analysis
# ==============================================================================#
# Load in parameters from csv. This ensures reproducability!
#   We save as a DataFrame, then convert to a dictionary, and take out nesting...
# Save as .csv
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

print("")
print("Loaded Analysis Parameters:")
for key, value in TRACERSPARAMS.items():
    print(f"{key}: {value}")

print("")

# Save types, which when combined with saveparams define what data is saved.
#   This is intended to be the string equivalent of the percentiles.
saveTypes = [str(percentile) for percentile in TRACERSPARAMS["percentiles"]]

# Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['rho_rhomean','dens','T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','csound','tcross','tff','tcool_tff']

saveHalo = (TRACERSPARAMS["savepath"].split("/"))[-2]


print("")
print("Saved Parameters in this Analysis:")
print(saveParams)

# Optional Tracer only (no stats in .csv) parameters to be saved
#   Cannot guarantee that all Plotting and post-processing are independent of these
#       Will attempt to ensure any necessary parameters are stored in ESSENTIALS
saveTracersOnly = TRACERSPARAMS["saveTracersOnly"]  # ['sfr','age']

print("")
print("Tracers ONLY (no stats) Saved Parameters in this Analysis:")
print(saveTracersOnly)

# SAVE ESSENTIALS : The data required to be tracked in order for the analysis to work
saveEssentials = TRACERSPARAMS[
    "saveEssentials"
]  # ['FoFHaloID','SubHaloID','Lookback','Ntracers','Snap','id','prid','trid','type','mass','pos']

print("")
print("ESSENTIAL Saved Parameters in this Analysis:")
print(saveEssentials)

saveTracersOnly = saveTracersOnly + saveEssentials

# Combine saveParams and saveTypes to form each combination for saving data types
saveKeys = []
for param in saveParams:
    for TYPE in saveTypes:
        saveKeys.append(param + TYPE)

# Select Halo of interest:
#   0 is the most massive:
HaloID = int(TRACERSPARAMS["haloID"])
# ==============================================================================#
#       Chemical Properties
# ==============================================================================#
# element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements = [
    "H",
    "He",
    "C",
    "N",
    "O",
    "Ne",
    "Mg",
    "Si",
    "Fe",
    "Y",
    "Sr",
    "Zr",
    "Ba",
    "Pb",
]
elements_Z = [1, 2, 6, 7, 8, 10, 12, 14, 26, 39, 38, 40, 56, 82]
elements_mass = [
    1.01,
    4.00,
    12.01,
    14.01,
    16.00,
    20.18,
    24.30,
    28.08,
    55.85,
    88.91,
    87.62,
    91.22,
    137.33,
    207.2,
]
elements_solar = [
    12.0,
    10.93,
    8.43,
    7.83,
    8.69,
    7.93,
    7.60,
    7.51,
    7.50,
    2.21,
    2.87,
    2.58,
    2.18,
    1.75,
]

Zsolar = 0.0127

omegabaryon0 = 0.048
# ==============================================================================#
#       MAIN PROGRAM
# ==============================================================================#


#
# ==============================================================================#
#       MAIN PROGRAM
# ==============================================================================#


snapGasFinalDict = {}

for snap in TRACERSPARAMS["phasesSnaps"]:
    _, _, _, _, snapGas, _ = tracer_selection_snap_analysis(
        TRACERSPARAMS,
        HaloID,
        elements,
        elements_Z,
        elements_mass,
        elements_solar,
        Zsolar,
        omegabaryon0,
        saveParams,
        saveTracersOnly,
        DataSavepath,
        FullDataPathSuffix,
        MiniDataPathSuffix,
        lazyLoadBool,
        SUBSET=None,
        snapNumber=snap,
        saveTracers=False,
        TFCbool=False,
        loadonlyhalo=False,
    )

    snapGasFinalDict.update({f"{int(snap)}": snapGas.data})

FullDict = FullDict_hdf5_load(DataSavepath, TRACERSPARAMS, FullDataPathSuffix)
snapRange = [
    snap
    for snap in range(
        int(TRACERSPARAMS["snapMin"]),
        min(int(TRACERSPARAMS["snapMax"]) + 1, int(TRACERSPARAMS["finalSnap"]) + 1),
    )
]

for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
    print(f"{rin}R{rout}")
    print("Flatten Tracers Data (snapData).")

    TracersFinalDict = flatten_wrt_T(FullDict, snapRange, TRACERSPARAMS, rin, rout)
    # ------------------------------------------------------------------------------#
    #               PLOTTING
    #
    # ------------------------------------------------------------------------------#
    for snap in TRACERSPARAMS["phasesSnaps"]:
        print("\n" + f"Starting Snap {int(snap)}")
        for weightKey in weightKeys:
            print("\n" + f"Starting weightKey {weightKey}")
            key = f"{int(snap)}"
            tkey = (f"{rin}R{rout}", key)
            selectTime = abs(
                FullDict[
                    (
                        f"T{float(Tlst[0])}",
                        f"{rin}R{rout}",
                        f"{int(TRACERSPARAMS['selectSnap'])}",
                    )
                ]["Lookback"][0]
                - ageUniverse
            )
            currentTime = abs(
                FullDict[(f"T{float(Tlst[0])}", f"{rin}R{rout}", f"{int(snap)}")][
                    "Lookback"
                ][0]
                - ageUniverse
            )

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(xsize, ysize), dpi=DPI)

            whereCellsGas = np.where(snapGasFinalDict[key]["type"] == 0)[0]

            whereTracersGas = np.where(TracersFinalDict[tkey]["type"] == 0)[0]

            zmin = zlimDict[weightKey]["zmin"]
            zmax = zlimDict[weightKey]["zmax"]

            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            #   Figure 1: Full Cells Data
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            print(f"snapData Plot!")

            xdataCells = np.log10(snapGasFinalDict[key]["rho_rhomean"][whereCellsGas])
            ydataCells = np.log10(snapGasFinalDict[key]["T"][whereCellsGas])
            massCells = snapGasFinalDict[key]["mass"][whereCellsGas]
            weightDataCells = (
                snapGasFinalDict[key][weightKey][whereCellsGas] * massCells
            )

            # xdataCellsNotNaNorInf = np.where((np.isinf(xdataCells)==False) & (np.isnan(xdataCells)==False))[0]
            # ydataCellsNotNaNorInf = np.where((np.isinf(ydataCells)==False) & (np.isnan(ydataCells)==False))[0]
            # weightDataCellsNotNaNorInf = np.where((np.isinf(weightDataCells)==False) & (np.isnan(weightDataCells)==False))[0]
            # massCellsNotNaNorInf = np.where((np.isinf(massCells)==False) & (np.isnan(massCells)==False))[0]
            #
            #
            # where_list = [xdataCellsNotNaNorInf.tolist(),ydataCellsNotNaNorInf.tolist(),\
            # weightDataCellsNotNaNorInf.tolist(),massCellsNotNaNorInf.tolist()]
            #
            # whereData = np.array(reduce(np.intersect1d,where_list))
            #
            # xdataCells = xdataCells[whereData]
            # ydataCells = ydataCells[whereData]
            # massCells  = massCells[whereData]
            # weightDataCells = weightDataCells[whereData]

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

            img1 = ax[0].pcolormesh(
                xcells,
                ycells,
                finalHistCells,
                cmap=colourmap,
                vmin=zmin,
                vmax=zmax,
                rasterized=True,
            )

            # img1 = ax[0].imshow(finalHistCells,cmap=colourmap,vmin=zmin,vmax=zmax \
            # ,extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')

            ax[0].set_xlabel(
                r"Log10 Density [$\rho / \langle \rho \rangle $]", fontsize=fontsize
            )
            ax[0].set_ylabel(r"Log10 Temperatures [$K$]", fontsize=fontsize)

            ax[0].set_ylim(ymin, ymax)
            ax[0].set_xlim(xmin, xmax)
            cax1 = inset_axes(ax[0], width="5%", height="95%", loc="right")
            fig.colorbar(img1, cax=cax1, orientation="vertical").set_label(
                label=labelDict[weightKey], size=fontsize
            )
            cax1.yaxis.set_ticks_position("left")
            cax1.yaxis.set_label_position("left")
            cax1.yaxis.label.set_color("black")
            cax1.tick_params(axis="y", colors="black", labelsize=fontsize)

            ax[0].set_title(f"Full Simulation Data", fontsize=fontsize)
            ax[0].set_aspect("auto")
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            #   Figure 2: Tracers Only  Data
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            print(f"Tracers Plot!")

            xdataTracers = np.log10(
                TracersFinalDict[tkey]["rho_rhomean"][whereTracersGas]
            )
            ydataTracers = np.log10(TracersFinalDict[tkey]["T"][whereTracersGas])
            massTracers = TracersFinalDict[tkey]["mass"][whereTracersGas]
            weightDataTracers = (
                TracersFinalDict[tkey][weightKey][whereTracersGas] * massTracers
            )

            # xdataTracersNotNaNorInf = np.where((np.isinf(xdataTracers)==False) & (np.isnan(xdataTracers)==False))[0]
            # ydataTracersNotNaNorInf = np.where((np.isinf(ydataTracers)==False) & (np.isnan(ydataTracers)==False))[0]
            # weightDataTracersNotNaNorInf = np.where((np.isinf(weightDataTracers)==False) & (np.isnan(weightDataTracers)==False))[0]
            # massTracersNotNaNorInf = np.where((np.isinf(massTracers)==False) & (np.isnan(massTracers)==False))[0]
            #
            # where_list = [xdataTracersNotNaNorInf.tolist(),ydataTracersNotNaNorInf.tolist(),\
            # weightDataTracersNotNaNorInf.tolist(),massTracersNotNaNorInf.tolist()]
            #
            # whereTracers = np.array(reduce(np.intersect1d,where_list))
            #
            # xdataTracers = xdataTracers[whereTracers]
            # ydataTracers = ydataTracers[whereTracers]
            # massTracers  = massTracers[whereTracers]
            # weightDataTracers = weightDataTracers[whereTracers]

            if weightKey == "mass":
                finalHistTracers, xedgeTracers, yedgeTracers = np.histogram2d(
                    xdataTracers, ydataTracers, bins=Nbins, weights=massTracers
                )
            else:
                mhistTracers, _, _ = np.histogram2d(
                    xdataTracers, ydataTracers, bins=Nbins, weights=massTracers
                )
                histTracers, xedgeTracers, yedgeTracers = np.histogram2d(
                    xdataTracers, ydataTracers, bins=Nbins, weights=weightDataTracers
                )

                finalHistTracers = histTracers / mhistTracers

            finalHistTracers[finalHistTracers == 0.0] = np.nan
            if weightKey in logparams:
                finalHistTracers = np.log10(finalHistTracers)
            finalHistTracers = finalHistTracers.T

            xtracers, ytracers = np.meshgrid(xedgeTracers, yedgeTracers)

            img2 = ax[1].pcolormesh(
                xtracers,
                ytracers,
                finalHistTracers,
                cmap=colourmap,
                vmin=zmin,
                vmax=zmax,
                rasterized=True,
            )

            # img2 = ax[1].imshow(finalHistTracers,cmap=colourmap,vmin=zmin,vmax=zmax \
            # ,extent=[np.min(xedgeTracers),np.max(xedgeTracers),np.min(yedgeTracers),np.max(yedgeTracers)],origin='lower')

            ax[1].set_xlabel(
                r"Log10 Density [$\rho / \langle \rho \rangle $]", fontsize=fontsize
            )
            ax[1].set_ylabel(r"Log10 Temperatures [$K$]", fontsize=fontsize)

            ax[1].set_ylim(ymin, ymax)
            ax[1].set_xlim(xmin, xmax)

            cax2 = inset_axes(ax[1], width="5%", height="95%", loc="right")
            fig.colorbar(img2, cax=cax2, orientation="vertical").set_label(
                label=labelDict[weightKey], size=fontsize
            )
            cax2.yaxis.set_ticks_position("left")
            cax2.yaxis.set_label_position("left")
            cax2.yaxis.label.set_color("black")
            cax2.tick_params(axis="y", colors="black", labelsize=fontsize)

            ax[1].set_title(
                f"Tracers Data, selected at {selectTime:3.2f} Gyr as being "
                r"$%05.2f \leq R \leq %05.2f kpc $" % (rin, rout)
                + "\n"
                + r" and temperatures "
                + r"$ 10^{n \pm %05.2f} K $" % (TRACERSPARAMS["deltaT"])
            )
            ax[1].set_aspect("auto")
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            #   Full Figure: Finishing up
            # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            fig.suptitle(
                f"Temperature Density Diagram, weighted by {weightKey}"
                + f" and selected at {currentTime:3.2f} Gyr",
                fontsize=fontsize,
            )

            plt.subplots_adjust(top=0.90, hspace=0.01)

            opslaan = (
                "./"
                + saveHalo
                + "/"
                + f"{int(rin)}R{int(rout)}"
                + "/"
                + f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_snap{int(snap)}_{weightKey}_PhaseDiagram.pdf"
            )
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

            fig, ax = plt.subplots(
                nrows=1, ncols=int(len(Tlst)), figsize=(xsize * 2, ysize), dpi=DPI
            )
            for (ii, T) in enumerate(Tlst):
                FullDictKey = (f"T{float(T)}", f"{rin}R{rout}", f"{int(snap)}")

                if len(Tlst) == 1:
                    currentAx = ax
                else:
                    currentAx = ax[ii]

                whereGas = np.where(FullDict[FullDictKey]["type"] == 0)[0]

                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                #   Figure 1: Full Cells Data
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
                print(f"T{T} Sub-Plot!")

                xdataCells = np.log10(FullDict[FullDictKey]["rho_rhomean"][whereGas])
                ydataCells = np.log10(FullDict[FullDictKey]["T"][whereGas])
                massCells = FullDict[FullDictKey]["mass"][whereGas]
                weightDataCells = FullDict[FullDictKey][weightKey][whereGas] * massCells

                # xdataCellsNotNaNorInf = np.where((np.isinf(xdataCells)==False) & (np.isnan(xdataCells)==False))[0]
                # ydataCellsNotNaNorInf = np.where((np.isinf(ydataCells)==False) & (np.isnan(ydataCells)==False))[0]
                # weightDataCellsNotNaNorInf = np.where((np.isinf(weightDataCells)==False) & (np.isnan(weightDataCells)==False))[0]
                # massCellsNotNaNorInf = np.where((np.isinf(massCells)==False) & (np.isnan(massCells)==False))[0]
                #
                # where_list = [xdataCellsNotNaNorInf.tolist(),ydataCellsNotNaNorInf.tolist(),\
                # weightDataCellsNotNaNorInf.tolist(),massCellsNotNaNorInf.tolist()]
                #
                # whereData = np.array(reduce(np.intersect1d,where_list))

                # xdataCells = xdataCells[whereData]
                # ydataCells = ydataCells[whereData]
                # massCells  = massCells[whereData]
                # weightDataCells = weightDataCells[whereData]

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
                    cmap=colourmap,
                    vmin=zmin,
                    vmax=zmax,
                    rasterized=True,
                )
                #
                # img1 = currentAx.imshow(finalHistCells,cmap=colourmap,vmin=zmin,vmax=zmax \
                # ,extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')

                currentAx.set_xlabel(
                    r"Log10 Density [$\rho / \langle \rho \rangle $]", fontsize=fontsize
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
                    r"$ 10^{%03.2f \pm %05.2f} K $ Tracers Data"
                    % (float(T), TRACERSPARAMS["deltaT"]),
                    fontsize=fontsize,
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
                + r" $%05.2f \leq R \leq %05.2f kpc $" % (rin, rout)
                + r" and temperatures "
                + r"$ 10^{n \pm %05.2f} K $" % (TRACERSPARAMS["deltaT"]),
                fontsize=fontsize,
            )

            plt.subplots_adjust(top=0.90, hspace=0.01)

            opslaan = (
                "./"
                + saveHalo
                + "/"
                + f"{int(rin)}R{int(rout)}"
                + "/"
                + f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_snap{int(snap)}_{weightKey}_PhaseDiagram_Individual-Temps.pdf"
            )
            plt.savefig(opslaan, dpi=DPI, transparent=False)
            print(opslaan)
