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
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize
import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

subset = 1000  # 10#1000
xsize = 10.0
ysize = 12.0
DPI = 250
opacity = 0.03  # 0.5#0.03

ageUniverse = 13.77  # [Gyr]

colourmapMain = "viridis"
colourmapIndividuals = "nipy_spectral"
# Input parameters path:
TracersParamsPath = "TracersParams_t3000.csv"

logParameters = [
    "dens",
    "rho_rhomean",
    "csound",
    "T",
    "n_H",
    "B",
    "L",
    "P_thermal",
    "P_magnetic",
    "P_kinetic",
    "P_tot",
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
    "tcool": r"Cooling Time [$Gyr$]",
    "theat": r"Heating Time [$Gyr$]",
    "tcross": r"Sound Crossing Cell Time [$Gyr$]",
    "tff": r"Free Fall Time [$Gyr$]",
    "tcool_tff": r"Cooling Time over Free Fall Time",
    "csound": r"Sound Speed",
    "rho_rhomean": r"Density over Average Universe Density",
    "dens": r"Density [$g$ $cm^{-3}$]",
}

for entry in logParameters:
    ylabel[entry] = r"Log10 " + ylabel[entry]

if subset <= 20:
    ColourIndividuals = True
else:
    ColourIndividuals = False
# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

DataSavepathSuffix = f".h5"


print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath, TRACERSPARAMS, DataSavepathSuffix)

print("Getting Tracer Data!")
XScatterDict = {}
Ydata = {}
Xdata = {}
Massdata = {}
ViolinDict = {}
# FoFHaloIDDict = {}
# SubHaloIDDict = {}

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

    key = (f"T{T}", f"{int(TRACERSPARAMS['selectSnap'])}")

    Rcrit = 500.0

    print(
        f"Select approx HaloID = {int(TRACERSPARAMS['haloID'])} or unbound and by R<={Rcrit:0.02f} kpc"
    )

    targetHaloList = np.array([int(TRACERSPARAMS["haloID"]), -1])
    Cond = np.where(
        (dataDict[key]["R"] <= Rcrit)
        & (np.isin(dataDict[key]["SubHaloID"], targetHaloList))
    )

    rangeMin = 0
    rangeMax = len(dataDict[key]["id"][Cond])
    CellNumberSelect = np.arange(start=rangeMin, stop=rangeMax, step=1)
    # Take Random sample of Cells size min(subset, len(data))
    CellNumberSelect = sample(CellNumberSelect.tolist(), min(subset, rangeMax))

    # selectMin = min(subset,rangeMax)
    # select = math.floor(float(rangeMax)/float(subset))
    # CellNumberSelect = CellNumberSelect[::select]
    #
    SelectedCells1 = dataDict[key]["id"][Cond][CellNumberSelect]

    XScatterSubDict = {}
    XSubDict = {}
    YSubDict = {}
    MassSubDict = {}
    ViolinSubDict = {}
    # FoFHaloIDSubDict = {}
    # SubHaloIDSubDict = {}
    for analysisParam in saveParams:
        print("")
        print(f"Starting {analysisParam} analysis")

        # Loop over snaps from and gather data for the SelectedCells1.
        #   This should be the same Cells for all time points due to the above selection, and thus data and massdata should always have the same shape.
        tmpXScatterdata = []
        tmpXdata = []
        tmpYdata = []
        tmpMassdata = []
        tmpViolinData = []
        # tmpFoFHaloID = []
        # tmpSubHaloID = []
        for snap in range(
            int(TRACERSPARAMS["snapMin"]),
            min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"] + 1)),
        ):
            key = (f"T{T}", f"{int(snap)}")
            Cond = np.where(np.isin(dataDict[key]["SubHaloID"], targetHaloList))
            whereGas = np.where(dataDict[key]["type"][Cond] == 0)[0]
            whereCell = np.where(
                np.isin(dataDict[key]["id"][whereGas], SelectedCells1)
            )[0]
            data, _ = GetIndividualCell(
                CellIDs=dataDict[key]["id"][whereGas],
                SelectedCells=SelectedCells1,
                Data=dataDict[key][analysisParam][whereGas],
            )
            massData, _ = GetIndividualCell(
                CellIDs=dataDict[key]["id"][whereGas],
                SelectedCells=SelectedCells1,
                Data=dataDict[key]["mass"][whereGas],
            )

            # whereCellInv  = np.where(np.isin(SelectedCells1,dataDict[key]['id'][whereGas]))
            # data = np.where(np.isin(SelectedCells1[whereCellInv],dataDict[key]['id'][whereGas][whereCell]),dataDict[key][analysisParam][whereGas][whereCell],np.nan)
            # massData = np.where(np.isin(SelectedCells1[whereCellInv],dataDict[key]['id'][whereGas][whereCell]),dataDict[key]['mass'][whereGas][whereCell],np.nan)
            # Get Individual Cell Data from selected Tracers.
            #   Not all Tracers will be present at all snapshots, so we return a NaN value in that instance.
            #   This allows for plotting of all tracers for all snaps they exist.
            #   Grab data for analysisParam and mass.
            # data = dataDict[key][analysisParam][whereGas][whereCell]
            # massData = dataDict[key]['mass'][whereGas][whereCell]

            # FoFData, _ = GetIndividualCellFromTracer(Tracers=dataDict[key]['trid'],\
            #     Parents=dataDict[key]['prid'],CellIDs=dataDict[key]['id'],SelectedTracers=SelectedTracers1,\
            #     Data=dataDict[key]['FoFHaloID'])
            #
            # HaloData, _ = GetIndividualCellFromTracer(Tracers=dataDict[key]['trid'],\
            #     Parents=dataDict[key]['prid'],CellIDs=dataDict[key]['id'],SelectedTracers=SelectedTracers1,\
            #     Data=dataDict[key]['SubHaloID'])

            # Append the data from this snapshot to a temporary list
            lookbackList = [dataDict[key]["Lookback"][0] for kk in SelectedCells1]
            tmpXScatterdata.append(lookbackList)
            tmpXdata.append(dataDict[key]["Lookback"][0])
            tmpYdata.append(data)
            tmpMassdata.append(massData)

            # #Save HaloID data
            # tmpFoFHaloID.append(FoFData)
            # tmpSubHaloID.append(HaloData)

            # Violin Data
            massMean = np.mean(dataDict[key]["mass"][whereGas])
            weightedData = (
                dataDict[key][analysisParam][whereGas] * dataDict[key]["mass"][whereGas]
            ) / massMean
            whereNOTnan = np.where(np.isnan(weightedData) == False)
            weightedData = weightedData[whereNOTnan]
            tmpViolinData.append(weightedData)

        # Append the data from this parameters to a sub dictionary
        XScatterSubDict.update({analysisParam: np.array(tmpXScatterdata)})
        XSubDict.update({analysisParam: np.array(tmpXdata)})
        YSubDict.update({analysisParam: np.array(tmpYdata)})
        MassSubDict.update({analysisParam: np.array(tmpMassdata)})
        ViolinSubDict.update({analysisParam: np.array(tmpViolinData)})
        # FoFHaloIDSubDict.update({analysisParam: np.array(tmpFoFHaloID)})
        # SubHaloIDSubDict.update({analysisParam : np.array(tmpSubHaloID)})

    # Add the full list of snaps data to temperature dependent dictionary.
    XScatterDict.update({f"T{T}": XScatterSubDict})
    Xdata.update({f"T{T}": XSubDict})
    Ydata.update({f"T{T}": YSubDict})
    Massdata.update({f"T{T}": MassSubDict})
    ViolinDict.update({f"T{T}": ViolinSubDict})
    # FoFHaloIDDict.update({f"T{T}" : FoFHaloIDSubDict})
    # SubHaloIDDict.update({f"T{T}" : SubHaloIDSubDict})
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


for analysisParam in saveParams:
    print("")
    print(f"Starting {analysisParam} Sub-plots!")

    fig, ax = plt.subplots(
        nrows=len(Tlst), ncols=1, sharex=True, figsize=(xsize, ysize), dpi=DPI
    )

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

        plotXScatterdata = XScatterDict[f"T{temp}"][analysisParam].copy()
        plotYdata = Ydata[f"T{temp}"][analysisParam].copy()
        plotXdata = Xdata[f"T{temp}"][analysisParam].astype("float64").copy()
        violinData = ViolinDict[f"T{temp}"][analysisParam].copy()
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
        plotXScatterdata = abs(plotXScatterdata - ageUniverse)
        # Set style options
        opacityPercentiles = 0.25
        lineStyleMedian = "solid"
        lineStylePercentiles = "-."

        # Select a Temperature specific colour from colourmap

        if ColourIndividuals == True:
            cmap = matplotlib.cm.get_cmap(colourmapIndividuals)
            colour = "tab:gray"
            colourTracers = [cmap(float(jj) / float(subset)) for jj in range(0, subset)]
        else:
            # Get a colour for median and percentiles for a given temperature
            #   Have fiddled to move colours away from extremes of the colormap
            cmap = matplotlib.cm.get_cmap(colourmapMain)
            colour = cmap(float(ii + 1) / float(len(Tlst)))
            colourTracers = "tab:gray"

        LO = analysisParam + "LO"
        UP = analysisParam + "UP"
        median = analysisParam + "median"

        YDataisNOTinf = np.where(np.isinf(plotYdata) == False)

        datamin = np.nanmin(plotYdata[YDataisNOTinf])
        datamax = np.nanmax(plotYdata[YDataisNOTinf])

        if analysisParam in logParameters:
            tmp = []
            for (ind, array) in enumerate(violinData):
                tmpData = np.log10(array)
                whereNOTnan = np.where(np.isnan(tmpData) == False)
                wherenan = np.where(np.isnan(tmpData) == True)
                tmp.append(tmpData[whereNOTnan])

            violinData = np.array(tmp)

            for k, v in plotData.items():
                plotData.update({k: np.log10(v)})

            plotYdata = np.log10(plotYdata)

            YDataisNOTinf = np.where(np.isinf(plotYdata) == False)

            datamin = np.nanmin(plotYdata[YDataisNOTinf])
            datamax = np.nanmax(plotYdata[YDataisNOTinf])

        if (np.isnan(datamax) == True) or (np.isnan(datamin) == True):
            print("NaN datamin/datamax. Skipping Entry!")
            continue

        if (np.isinf(datamax) == True) or (np.isinf(datamin) == True):
            print("Inf datamin/datamax. Skipping Entry!")
            continue

        ##
        #   If all entries of data are nan, and thus dataset len == 0
        #   add a nan and zero array to omit violin but continue plotting
        #   without errors.
        ##
        tmp = []
        for dataset in violinData:
            if len(dataset) == 0:
                tmp.append(np.array([np.nan, 0, np.nan]))
            else:
                tmp.append(dataset)

        violinData = tmp

        print("")
        print("Sub-Plot!")

        if len(Tlst) == 1:
            currentAx = ax
        else:
            currentAx = ax[ii]

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

        currentAx.fill_between(
            tage,
            plotData[UP],
            plotData[LO],
            facecolor=colour,
            alpha=opacityPercentiles,
            interpolate=False,
        )
        currentAx.plot(
            tage,
            plotData[median],
            label=r"$T = 10^{%3.0f} K$" % (float(temp)),
            color=colour,
            linestyle=lineStyleMedian,
        )

        # for jj in range(1,len(plotXdata)):
        #     whereDataIsNOTnan = np.where((np.isnan(plotYdata[jj])==False)& (np.isnan(plotYdata[jj-1])==False))
        #
        #     for kk in range(len(plotYdata[jj-1][whereDataIsNOTnan])):
        #         currentAx.plot(np.array([plotXScatterdata[jj-1][whereDataIsNOTnan][kk],plotXScatterdata[jj][whereDataIsNOTnan][kk]]),\
        #         np.array([(plotYdata[jj-1][whereDataIsNOTnan][kk]),(plotYdata[jj][whereDataIsNOTnan][kk])]), color = colourTracersHalo[normedSubHaloIDData[jj][whereDataIsNOTnan]][kk], alpha = opacity )
        startkey = (f"T{temp}", f"{int(TRACERSPARAMS['snapMin'])}")
        endkey = (
            f"T{temp}",
            f"{min(int(TRACERSPARAMS['finalSnap']),int(TRACERSPARAMS['snapMax']))}",
        )
        startNCell = dataDict[startkey]["Ncells"][0]
        endNCell = dataDict[endkey]["Ncells"][0]

        startSubHaloIDDataFull = dataDict[startkey]["SubHaloID"].copy()
        endSubHaloIDDataFull = dataDict[endkey]["SubHaloID"].copy()
        unboundFracStart = float(
            np.shape(np.where(startSubHaloIDDataFull == -1)[0])[0]
        ) / float(startNCell)
        unboundFracEnd = float(
            np.shape(np.where(endSubHaloIDDataFull == -1)[0])[0]
        ) / float(endNCell)
        haloFracStart = float(
            np.shape(
                np.where(startSubHaloIDDataFull == int(TRACERSPARAMS["haloID"]))[0]
            )[0]
        ) / float(startNCell)
        haloFracEnd = float(
            np.shape(np.where(endSubHaloIDDataFull == int(TRACERSPARAMS["haloID"]))[0])[
                0
            ]
        ) / float(endNCell)

        otherHaloFracStart = float(
            np.shape(
                np.where(
                    (startSubHaloIDDataFull != int(TRACERSPARAMS["haloID"]))
                    & (startSubHaloIDDataFull != -1)
                    & (np.isnan(startSubHaloIDDataFull) == False)
                )[0]
            )[0]
        ) / float(startNCell)

        otherHaloFracEnd = float(
            np.shape(
                np.where(
                    (endSubHaloIDDataFull != int(TRACERSPARAMS["haloID"]))
                    & (endSubHaloIDDataFull != -1)
                    & (np.isnan(endSubHaloIDDataFull) == False)
                )[0]
            )[0]
        ) / float(endNCell)

        unassignedFracStart = float(
            np.shape(np.where(np.isnan(startSubHaloIDDataFull) == True)[0])[0]
        ) / float(startNCell)
        unassignedFracEnd = float(
            np.shape(np.where(np.isnan(endSubHaloIDDataFull) == True)[0])[0]
        ) / float(endNCell)

        HaloString = (
            f"Of Cell Subset: \n {haloFracStart:3.3%} start in Halo {int(TRACERSPARAMS['haloID'])},"
            + f" {unboundFracStart:3.3%} start 'unbound',{otherHaloFracStart:3.3%} start in other Haloes, {unassignedFracStart:3.3%} start unassigned."
            + f"\n {haloFracEnd:3.3%} end in Halo {int(TRACERSPARAMS['haloID'])}, {unboundFracEnd:3.3%} end 'unbound',{otherHaloFracEnd:3.3%} end in other Haloes, {unassignedFracEnd:3.3%} end unassigned."
        )

        currentAx.text(
            1.02,
            0.5,
            HaloString,
            horizontalalignment="left",
            verticalalignment="center",
            transform=currentAx.transAxes,
            wrap=True,
            bbox=dict(facecolor="tab:gray", alpha=0.25),
        )

        currentAx.transAxes

        # parts = currentAx.violinplot(violinData,positions=plotXdata,showmeans=False,showmedians=False,showextrema=False)#label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour, linestyle=lineStyleMedian)
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

        whereDataIsNOTnan = np.where(np.isnan(plotYdata) == False)
        paths = np.array([plotXScatterdata, plotYdata]).T.reshape(-1, len(plotXdata), 2)

        if ColourIndividuals == True:
            lc = LineCollection(paths, cmap=colourmapIndividuals, alpha=opacity)
            # for jj in range(0,subset):
            # whereDataIsNOTnan = np.where(np.isnan(plotYdata[:,jj])==False)
            # lenNOTnan = len(plotYdata[:,jj][whereDataIsNOTnan])
            # if (lenNOTnan>0):
            #     currentAx.plot(plotXdata,(plotYdata.T[jj]).T, color = colourTracers[jj], alpha = opacity )
        else:
            # for jj in range(0,subset):
            #     whereDataIsNOTnan = np.where(np.isnan(plotYdata[:,jj])==False)
            #     lenNOTnan = len(plotYdata[:,jj][whereDataIsNOTnan])
            #     if (lenNOTnan>0):
            #         currentAx.plot(plotXdata,(plotYdata.T[jj]).T, color = colourTracers, alpha = opacity )

            # segments = np.concatenate([points[:-1],points[1:]], axis=1)

            # Ncolours = len(uniqueSubHalo)
            # cmap2 = matplotlib.cm.get_cmap(colourmapIndividuals, 256)
            # newcolors = cmap2(np.linspace(0, 0.9, Ncolours))
            # cmap3 = ListedColormap(newcolors)
            # cmin = float(normedSubHaloIDData.min())-0.5
            # cmax = float(normedSubHaloIDData.max())+1.5
            # norm = BoundaryNorm([xx for xx in np.arange(cmin,cmax,1)], cmap3.N)

            lc = LineCollection(paths, color=colourTracers, alpha=opacity)

            # cbar = plt.colorbar(line,ticks=normedUniqueSubHalo,ax=currentAx, orientation = 'vertical')
            # cbar.set_label(label=r'Sub-Halo ID')
            # cbar.solids.set(alpha=1)

        line = currentAx.add_collection(lc)

        currentAx.xaxis.set_minor_locator(AutoMinorLocator())
        currentAx.yaxis.set_minor_locator(AutoMinorLocator())
        currentAx.tick_params(which="both")

        currentAx.set_ylabel(ylabel[analysisParam], fontsize=10)
        currentAx.set_ylim(ymin=datamin, ymax=datamax)

        plot_patch = matplotlib.patches.Patch(color=colour)
        plot_label = r"$T = 10^{%3.2f} K$" % (float(temp))
        currentAx.legend(handles=[plot_patch], labels=[plot_label], loc="upper right")

        fig.suptitle(
            f"Cells selected by: "
            + "\n"
            + r"$T = 10^{n \pm %05.2f} K$" % (TRACERSPARAMS["deltaT"])
            + r" and $%05.2f \leq R \leq %05.2f kpc $"
            % (TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"])
            + "\n"
            + f" and selected at {vline[0]:3.2f} Gyr"
            + f" weighted by mass"
            + "\n"
            + f"Subset of {int(subset)} Individual Cells at each Temperature Plotted",
            fontsize=12,
        )

    # Only give 1 x-axis a label, as they sharex
    if len(Tlst) == 1:
        axis0 = ax
    else:
        axis0 = ax[len(Tlst) - 1]

    axis0.set_xlabel(r"Age of Universe [$Gyrs$]", fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, right=0.80)
    opslaan = (
        f"Cells_t3000_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"
        + analysisParam
        + "_"
        + str(int(subset))
        + f"_IndividualsMedians.pdf"
    )
    plt.savefig(opslaan, dpi=DPI, transparent=False)
    print(opslaan)
    plt.close()
