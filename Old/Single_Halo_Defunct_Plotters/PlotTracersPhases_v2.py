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


# ==============================================================================#
#       Prepare for analysis
# ==============================================================================#
# Load in parameters from csv. This ensures reproducability!
#   We save as a DataFrame, then convert to a dictionary, and take out nesting...
# Save as .csv
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersParamsPath)

saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

DataSavepathSuffix = f".h5"

saveHalo = (TRACERSPARAMS["savepath"].split("/"))[-2]

print("Loading data!")
FullDict = full_dict_hdf5_load(DataSavepath, TRACERSPARAMS, FullDataPathSuffix)
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
    for snap in snapRange:
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
            )
            currentTime = abs(
                FullDict[(f"T{float(Tlst[0])}", f"{rin}R{rout}", f"{int(snap)}")][
                    "Lookback"
                ][0]
            )

            zmin = zlimDict[weightKey]["zmin"]
            zmax = zlimDict[weightKey]["zmax"]

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
