import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

ageUniverse = 13.77  # [Gyr]
xsize = 30.0
ysize = 10.0
DPI = 100
colourmapMain = "plasma"
# Input parameters path:
TracersParamsPath = "TracersParams.csv"
DataSavepathSuffix = f".h5"
singleVals = ["Rinner", "Router", "T", "Snap", "Lookback"]

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
# ==============================================================================#

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersParamsPath)

saveHalo = (TRACERSPARAMS["savepath"].split("/"))[-2]

print("Loading data!")

dataDict = {}

dataDict = full_dict_hdf5_load(DataSavepath, TRACERSPARAMS, DataSavepathSuffix)

snapsRange = np.array(
    [
        xx
        for xx in range(
            int(TRACERSPARAMS["snapMin"]),
            min(int(TRACERSPARAMS["snapMax"]) + 1, int(TRACERSPARAMS["finalSnap"]) + 1),
            1,
        )
    ]
)

FlatDataDict = {}
for T in Tlst:
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        loadPath = (
            DataSavepath + f"_T{T}_{rin}R{rout}_flat-wrt-time" + DataSavepathSuffix
        )
        key = (f"T{T}", f"{rin}R{rout}")
        tmp = hdf5_load(loadPath)
        tmp[key].update({"Ntracers": np.shape(tmp[key]["type"])[1]})
        FlatDataDict.update(tmp)
################################################################################
##                           Definitions                                    ####
################################################################################
# ------------------------------------------------------------------------------#
#                    Get id trid prid data from where Cells
# ------------------------------------------------------------------------------#
def _get_id_prid_trid_where(dataDict, whereEntries):

    id = dataDict["id"][whereEntries]

    _, prid_ind, _ = np.intersect1d(dataDict["prid"], id, return_indices=True)

    prid = dataDict["prid"][prid_ind]
    trid = dataDict["trid"][prid_ind]

    return {"id": id, "prid": prid, "trid": trid}


# ------------------------------------------------------------------------------#
#                    Get id data from single trid
# ------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------#
#
# ------------------------------------------------------------------------------#


def flat_analyse_time_averages(
    FlatDataDict, Tlst, snapsRange, lookbackData, TRACERSPARAMS
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
    preselectInd = np.where(snapsRange < int(TRACERSPARAMS["selectSnap"]))[0]
    postselectInd = np.where(snapsRange > int(TRACERSPARAMS["selectSnap"]))[0]
    selectInd = np.where(snapsRange == int(TRACERSPARAMS["selectSnap"]))[0]
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
            # Select where ANY tracer (gas or stars) meets condition PRIOR TO selection
            rowspre, colspre = np.where(
                FlatDataDict[Tkey]["type"][preselectInd, :] == 0
            )
            # Calculate the number of these unique tracers compared to the total number
            gaspre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            rowspost, colspost = np.where(
                FlatDataDict[Tkey]["type"][postselectInd, :] == 0
            )
            gaspost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracersAll)
            )
            # Add data to internal database lists
            gas.append([gaspre, gaspost])

            print("Heating & Cooling")
            epsilonT = float(TRACERSPARAMS["deltaT"])  # [k]

            # Select where GAS FOREVER ONLY tracers meet condition FOR THE LAST 2 SNAPSHOTS PRIOR TO SELECTION
            rowspre, colspre = np.where(
                np.log10(FlatDataDict[Tkey]["T"][:, whereGas][preselectInd, :][-1:])
                - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                > (epsilonT)
            )
            # Calculate the number of these unique tracers compared to the total number
            coolingpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )

            rowspost, colspost = np.where(
                np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][postselectInd, :][:1])
                > (epsilonT)
            )

            coolingpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )

            rowspre, colspre = np.where(
                np.log10(FlatDataDict[Tkey]["T"][:, whereGas][preselectInd, :][-1:])
                - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                < (-1.0 * epsilonT)
            )
            heatingpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )

            rowspost, colspost = np.where(
                np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][postselectInd, :][:1])
                < (-1.0 * epsilonT)
            )
            heatingpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )

            rowspre, colspre = np.where(
                (
                    np.log10(FlatDataDict[Tkey]["T"][:, whereGas][preselectInd, :][-1:])
                    - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                    <= (0 + epsilonT)
                )
                & (
                    np.log10(FlatDataDict[Tkey]["T"][:, whereGas][preselectInd, :][-1:])
                    - np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                    >= (0 - epsilonT)
                )
            )
            smallTpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)

            rowspost, colspost = np.where(
                (
                    np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                    - np.log10(
                        FlatDataDict[Tkey]["T"][:, whereGas][postselectInd, :][:1]
                    )
                    <= (0.0 + epsilonT)
                )
                & (
                    np.log10(FlatDataDict[Tkey]["T"][:, whereGas][selectInd, :])
                    - np.log10(
                        FlatDataDict[Tkey]["T"][:, whereGas][postselectInd, :][:1]
                    )
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
            #     FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:,whereGas][preselectInd, :][-2:] > 1.0
            # )
            # pthermpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # rowspost, colspost = np.where(
            #     FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:,whereGas][postselectInd, :][:2] > 1.0
            # )
            # pthermpost = (
            #     100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # )
            # rowspre, colspre = np.where(
            #     FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:,whereGas][preselectInd, :][-2:] < 1.0
            # )
            # pmagpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # rowspost, colspost = np.where(
            #     FlatDataDict[Tkey]["Pthermal_Pmagnetic"][:,whereGas][postselectInd, :][:2] < 1.0
            # )
            # pmagpost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # ptherm.append([pthermpre, pthermpost])
            # pmag.append([pmagpre, pmagpost])

            # print("tcool_tff 10 ")
            # rowspre, colspre = np.where(
            #     FlatDataDict[Tkey]["tcool_tff"][:,whereGas][preselectInd, :][-2:] > 10.0
            # )
            # tcoolpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # rowspost, colspost = np.where(
            #     FlatDataDict[Tkey]["tcool_tff"][:,whereGas][postselectInd, :][:2] > 10.0
            # )
            # tcoolpost = (
            #     100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # )
            # rowspre, colspre = np.where(
            #     FlatDataDict[Tkey]["tcool_tff"][:,whereGas][preselectInd, :][-2:] < 10.0
            # )
            # tffpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            # rowspost, colspost = np.where(
            #     FlatDataDict[Tkey]["tcool_tff"][:,whereGas][postselectInd, :][:2] < 10.0
            # )
            # tffpost = 100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            # tcool.append([pthermpre, pthermpost])
            # tff.append([tffpre, tffpost])
            #
            print("Z")
            # Select FOREVER GAS ONLY tracers' specific parameter data and mass weights PRIOR TO SELECTION
            data = FlatDataDict[Tkey]["gz"][:, whereGas][preselectInd, :]
            weights = FlatDataDict[Tkey]["mass"][:, whereGas][preselectInd, :]
            zPreDat = []
            # For each tracers, calculate the mass weighted average of specific parameter for all selected snapshots
            for (dat, wei) in zip(data.T, weights.T):
                zPreDat.append(weighted_percentile(dat, wei, 50, "Z-Pre"))
            zPreDat = np.array(zPreDat)

            data = FlatDataDict[Tkey]["gz"][:, whereGas][postselectInd, :]
            weights = FlatDataDict[Tkey]["mass"][:, whereGas][postselectInd, :]
            zPostDat = []
            for (dat, wei) in zip(data.T, weights.T):
                zPostDat.append(weighted_percentile(dat, wei, 50, "Z-Post"))
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
            data = FlatDataDict[Tkey]["vrad"][:, whereGas][preselectInd, :]
            weights = FlatDataDict[Tkey]["mass"][:, whereGas][preselectInd, :]
            vradPreDat = []
            for (dat, wei) in zip(data.T, weights.T):
                vradPreDat.append(weighted_percentile(dat, wei, 50, "Vrad-Pre"))
            vradPreDat = np.array(vradPreDat)

            data = FlatDataDict[Tkey]["vrad"][:, whereGas][postselectInd, :]
            weights = FlatDataDict[Tkey]["mass"][:, whereGas][postselectInd, :]
            vradPostDat = []
            for (dat, wei) in zip(data.T, weights.T):
                vradPostDat.append(weighted_percentile(dat, wei, 50, "Vrad-Post"))
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
                FlatDataDict[Tkey]["SubHaloID"][:, whereGas][preselectInd, :]
                == int(TRACERSPARAMS["haloID"])
            )
            halo0pre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            rowspost, colspost = np.where(
                FlatDataDict[Tkey]["SubHaloID"][:, whereGas][postselectInd, :]
                == int(TRACERSPARAMS["haloID"])
            )
            halo0post = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            halo0.append([halo0pre, halo0post])

            print("Unbound")
            rowspre, colspre = np.where(
                FlatDataDict[Tkey]["SubHaloID"][:, whereGas][preselectInd, :] == -1
            )
            unboundpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            rowspost, colspost = np.where(
                FlatDataDict[Tkey]["SubHaloID"][:, whereGas][postselectInd, :] == -1
            )
            unboundpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            unbound.append([unboundpre, unboundpost])

            print("OtherHalo")
            rowspre, colspre = np.where(
                (
                    FlatDataDict[Tkey]["SubHaloID"][:, whereGas][preselectInd, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["SubHaloID"][:, whereGas][preselectInd, :] != -1)
                & (
                    np.isnan(
                        FlatDataDict[Tkey]["SubHaloID"][:, whereGas][preselectInd, :]
                    )
                    == False
                )
            )
            otherHalopre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            )
            rowspost, colspost = np.where(
                (
                    FlatDataDict[Tkey]["SubHaloID"][:, whereGas][postselectInd, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["SubHaloID"][:, whereGas][postselectInd, :] != -1)
                & (
                    np.isnan(
                        FlatDataDict[Tkey]["SubHaloID"][:, whereGas][postselectInd, :]
                    )
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
                    FlatDataDict[Tkey]["SubHaloID"][:, whereGas][preselectInd, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["SubHaloID"][:, whereGas][preselectInd, :] != -1)
                & (
                    np.isnan(
                        FlatDataDict[Tkey]["SubHaloID"][:, whereGas][preselectInd, :]
                    )
                    == True
                )
            )
            noHalopre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracers)
            rowspost, colspost = np.where(
                (
                    FlatDataDict[Tkey]["SubHaloID"][:, whereGas][postselectInd, :]
                    != int(TRACERSPARAMS["haloID"])
                )
                & (FlatDataDict[Tkey]["SubHaloID"][:, whereGas][postselectInd, :] != -1)
                & (
                    np.isnan(
                        FlatDataDict[Tkey]["SubHaloID"][:, whereGas][postselectInd, :]
                    )
                    == True
                )
            )
            noHalopost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracers)
            )
            noHalo.append([noHalopre, noHalopost])

            print("Stars")
            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["type"][preselectInd, :] == 4)
                & (FlatDataDict[Tkey]["age"][preselectInd, :] >= 0.0)
            )
            starspre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            )
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["type"][postselectInd, :] == 4)
                & (FlatDataDict[Tkey]["age"][postselectInd, :] >= 0.0)
            )
            starspost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracersAll)
            )
            stars.append([starspre, starspost])

            print("Wind")
            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["type"][preselectInd, :] == 4)
                & (FlatDataDict[Tkey]["age"][preselectInd, :] < 0.0)
            )
            windpre = (
                100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            )
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["type"][postselectInd, :] == 4)
                & (FlatDataDict[Tkey]["age"][postselectInd, :] < 0.0)
            )
            windpost = (
                100.0 * float(np.shape(np.unique(colspost))[0]) / float(ntracersAll)
            )
            wind.append([windpre, windpost])

            print("ISM")
            rowspre, colspre = np.where(
                (FlatDataDict[Tkey]["R"][preselectInd, :] <= 25.0)
                & (FlatDataDict[Tkey]["sfr"][preselectInd, :] > 0.0)
            )
            ismpre = 100.0 * float(np.shape(np.unique(colspre))[0]) / float(ntracersAll)
            rowspost, colspost = np.where(
                (FlatDataDict[Tkey]["R"][postselectInd, :] <= 25.0)
                & (FlatDataDict[Tkey]["sfr"][postselectInd, :] > 0.0)
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
            "%Above3/4(Z_solar)": {
                "Pre-Selection": np.array(aboveZ)[:, 0],
                "Post-Selection": np.array(aboveZ)[:, 1],
            },
            "%Below3/4(Z_solar)": {
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


# ------------------------------------------------------------------------------#
#               Analyse log10(T) and snap specfifc statistics
# ------------------------------------------------------------------------------#
def _inner_analysis(dataDict, TRACERSPARAMS):

    out = {}

    NtracersAll = dataDict["Ntracers"]

    id = dataDict["id"]
    prid = dataDict["prid"]
    trid = dataDict["trid"]

    tracers = {"data": NtracersAll, "id": id, "prid": prid, "trid": trid}
    out.update({"TracersFull": tracers})

    gas = {}
    whereGas = np.where(dataDict["type"] == 0)[0]
    tmp = _get_id_prid_trid_where(dataDict, whereGas)
    gas.update(tmp)
    gasdata = 100.0 * (np.shape(gas["trid"])[0] / NtracersAll)
    gas.update({"data": gasdata})
    out.update({"%Gas": gas})

    Ntracers = np.shape(whereGas)[0]

    SubHalo = dataDict["SubHaloID"]

    halo0 = {}
    wherehalo0 = np.where(SubHalo[whereGas] == int(TRACERSPARAMS["haloID"]))[0]
    tmp = _get_id_prid_trid_where(dataDict, wherehalo0)
    halo0.update(tmp)
    halo0data = 100.0 * (np.shape(halo0["trid"])[0] / Ntracers)
    halo0.update({"data": halo0data})
    out.update({"%Halo0": halo0})

    unbound = {}
    whereunbound = np.where(SubHalo[whereGas] == -1)[0]
    tmp = _get_id_prid_trid_where(dataDict, whereunbound)
    unbound.update(tmp)
    unbounddata = 100.0 * (np.shape(unbound["trid"])[0] / Ntracers)
    unbound.update({"data": unbounddata})
    out.update({"%Unbound": unbound})

    otherHalo = {}
    whereotherHalo = np.where(
        (SubHalo[whereGas] != int(TRACERSPARAMS["haloID"]))
        & (SubHalo[whereGas] != -1)
        & (np.isnan(SubHalo[whereGas]) == False)
    )[0]
    tmp = _get_id_prid_trid_where(dataDict, whereotherHalo)
    otherHalo.update(tmp)
    otherHalodata = 100.0 * (np.shape(otherHalo["trid"])[0] / Ntracers)
    otherHalo.update({"data": otherHalodata})
    out.update({"%OtherHalo": otherHalo})

    noHalo = {}
    wherenoHalo = np.where(
        (SubHalo[whereGas] != int(TRACERSPARAMS["haloID"]))
        & (SubHalo[whereGas] != -1)
        & (np.isnan(SubHalo[whereGas]) == True)
    )[0]
    tmp = _get_id_prid_trid_where(dataDict, wherenoHalo)
    noHalo.update(tmp)
    noHalodata = 100.0 * (np.shape(noHalo["trid"])[0] / Ntracers)
    noHalo.update({"data": noHalodata})
    out.update({"%NoHalo": noHalo})

    stars = {}
    wherestars = np.where((dataDict["age"] >= 0) & (dataDict["type"] == 4))[0]
    tmp = _get_id_prid_trid_where(dataDict, wherestars)
    stars.update(tmp)
    starsdata = 100.0 * (np.shape(stars["trid"])[0] / NtracersAll)
    stars.update({"data": starsdata})
    out.update({"%Stars": stars})

    wind = {}
    wherewind = np.where((dataDict["age"] < 0) & (dataDict["type"] == 4))[0]
    tmp = _get_id_prid_trid_where(dataDict, wherewind)
    wind.update(tmp)
    winddata = 100.0 * (np.shape(wind["trid"])[0] / NtracersAll)
    wind.update({"data": winddata})
    out.update({"%Wind": wind})

    ism = {}
    whereism = np.where((dataDict["sfr"] > 0) & (dataDict["R"] <= 25.0))[0]
    tmp = _get_id_prid_trid_where(dataDict, whereism)
    ism.update(tmp)
    ismdata = 100.0 * (np.shape(ism["trid"])[0] / Ntracers)
    ism.update({"data": ismdata})
    out.update({"%ISM": ism})

    inflow = {}
    whereinflow = np.where(dataDict["vrad"][whereGas] < 0.0)[0]
    tmp = _get_id_prid_trid_where(dataDict, whereinflow)
    inflow.update(tmp)
    inflowdata = 100.0 * (np.shape(inflow["trid"])[0] / Ntracers)
    inflow.update({"data": inflowdata})
    out.update({"%Inflow": inflow})

    outflow = {}
    whereoutflow = np.where(dataDict["vrad"][whereGas] > 0.0)[0]
    tmp = _get_id_prid_trid_where(dataDict, whereoutflow)
    outflow.update(tmp)
    outflowdata = 100.0 * (np.shape(outflow["trid"])[0] / Ntracers)
    outflow.update({"data": outflowdata})
    out.update({"%Outflow": outflow})

    aboveZ = {}
    whereaboveZ = np.where(dataDict["gz"][whereGas] > (0.75))[0]
    tmp = _get_id_prid_trid_where(dataDict, whereaboveZ)
    aboveZ.update(tmp)
    aboveZdata = 100.0 * (np.shape(aboveZ["trid"])[0] / Ntracers)
    aboveZ.update({"data": aboveZdata})
    out.update({"%Above3/4(Z_solar)": aboveZ})

    belowZ = {}
    wherebelowZ = np.where(dataDict["gz"][whereGas] < (0.75))[0]
    tmp = _get_id_prid_trid_where(dataDict, wherebelowZ)
    belowZ.update(tmp)
    belowZdata = 100.0 * (np.shape(belowZ["trid"])[0] / Ntracers)
    belowZ.update({"data": belowZdata})
    out.update({"%Below3/4(Z_solar)": belowZ})

    heating = {}
    whereheating = np.where(np.isnan(dataDict["theat"][whereGas]) == False)[0]
    tmp = _get_id_prid_trid_where(dataDict, whereheating)
    heating.update(tmp)
    heatingdata = 100.0 * (np.shape(heating["trid"])[0] / Ntracers)
    heating.update({"data": heatingdata})
    out.update({"%Heating": heating})

    cooling = {}
    wherecooling = np.where(np.isnan(dataDict["tcool"][whereGas]) == False)[0]
    tmp = _get_id_prid_trid_where(dataDict, wherecooling)
    cooling.update(tmp)
    coolingdata = 100.0 * (np.shape(cooling["trid"])[0] / Ntracers)
    cooling.update({"data": coolingdata})
    out.update({"%Cooling": cooling})

    ptherm = {}
    whereptherm = np.where(dataDict["Pthermal_Pmagnetic"][whereGas] > 1.0)[0]
    tmp = _get_id_prid_trid_where(dataDict, whereptherm)
    ptherm.update(tmp)
    pthermdata = 100.0 * (np.shape(ptherm["trid"])[0] / Ntracers)
    ptherm.update({"data": pthermdata})
    out.update({"%(Pthermal_Pmagnetic)Above1": ptherm})

    pmag = {}
    wherepmag = np.where(dataDict["Pthermal_Pmagnetic"][whereGas] < 1.0)[0]
    tmp = _get_id_prid_trid_where(dataDict, wherepmag)
    pmag.update(tmp)
    pmagdata = 100.0 * (np.shape(pmag["trid"])[0] / Ntracers)
    pmag.update({"data": pmagdata})
    out.update({"%(Pthermal_Pmagnetic)Below1": pmag})

    tcool = {}
    wheretcool = np.where(dataDict["tcool_tff"][whereGas] > 10.0)[0]
    tmp = _get_id_prid_trid_where(dataDict, wheretcool)
    tcool.update(tmp)
    tcooldata = 100.0 * (np.shape(tcool["trid"])[0] / Ntracers)
    tcool.update({"data": tcooldata})
    out.update({"%(tcool_tff)Above10": tcool})

    tff = {}
    wheretff = np.where(dataDict["tcool_tff"][whereGas] < 10.0)[0]
    tmp = _get_id_prid_trid_where(dataDict, wheretff)
    tff.update(tmp)
    tffdata = 100.0 * (np.shape(tff["trid"])[0] / Ntracers)
    tff.update({"data": tffdata})
    out.update({"%(tcool_tff)Below10": tff})

    out.update({"Lookback": dataDict["Lookback"]})

    return out


# ------------------------------------------------------------------------------#
#               Analyse statistics for all T and snaps
# ------------------------------------------------------------------------------#
def full_data_analyse_tracer_averages(dataDict, Tlst, snapsRange, TRACERSPARAMS):
    dflist = []
    out = {}
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        for T in Tlst:
            Tkey = (f"T{T}", f"{rin}R{rout}")
            print(Tkey)
            tmp = {}
            for ii, snap in enumerate(snapsRange):
                print(f"{int(snap)}")
                FullKey = (f"T{T}", f"{rin}R{rout}", f"{int(snap)}")
                snapdataDict = dataDict[FullKey]
                outinner = _inner_analysis(snapdataDict, TRACERSPARAMS)
                outinner.update({"T": T})
                outinner.update({"Snap": int(snap)})
                outinner.update({"Rinner": rin})
                outinner.update({"Router": rout})

                for key, value in outinner.items():
                    if key == "T":
                        if isinstance(value, np.ndarray):
                            value = value[0]
                        if isinstance(value, list):
                            value = value[0]
                        if key in list(tmp.keys()):
                            val = tmp[key]
                            val = val + [value]
                            tmp.update({"T": val})
                        else:
                            tmp.update({"T": [float(value)]})
                    elif key == "Lookback":
                        if isinstance(value, np.ndarray):
                            value = value[0]
                        if isinstance(value, list):
                            value = value[0]
                        if key in list(tmp.keys()):
                            val = tmp[key]
                            val = val + [value]
                            tmp.update({"Lookback": val})
                        else:
                            tmp.update({"Lookback": [float(value)]})
                    elif key == "Snap":
                        if isinstance(value, np.ndarray):
                            value = value[0]
                        if isinstance(value, list):
                            value = value[0]
                        if key in list(tmp.keys()):
                            val = tmp[key]
                            val = val + [value]
                            tmp.update({"Snap": val})
                        else:
                            tmp.update({"Snap": [float(value)]})
                    elif key == "Rinner":
                        if isinstance(value, np.ndarray):
                            value = value[0]
                        if isinstance(value, list):
                            value = value[0]
                        if key in list(tmp.keys()):
                            val = tmp[key]
                            val = val + [value]
                            tmp.update({"Rinner": val})
                        else:
                            tmp.update({"Rinner": [float(value)]})
                    elif key == "Router":
                        if isinstance(value, np.ndarray):
                            value = value[0]
                        if isinstance(value, list):
                            value = value[0]
                        if key in list(tmp.keys()):
                            val = tmp[key]
                            val = val + [value]
                            tmp.update({"Router": val})
                        else:
                            tmp.update({"Router": [float(value)]})
                    else:
                        value = value["data"]
                        if isinstance(value, np.ndarray):
                            value = value[0]
                        if isinstance(value, list):
                            value = value[0]
                        if key in list(tmp.keys()):
                            val = tmp[key]
                            val = val + [value]
                            tmp.update({key: val})
                        else:
                            tmp.update({key: [float(value)]})
                out.update({FullKey: outinner})
            dflist.append(pd.DataFrame.from_dict(tmp))

    outDF = pd.concat(dflist, axis=0, join="outer", sort=False, ignore_index=True)
    return out, outDF


################################################################################
##                           MAIN PROGRAM                                   ####
################################################################################
print("Analyse Data!")
# # ------------------------------------------------------------------------------#
# #               Analyse statistics for all T and snaps
# # ------------------------------------------------------------------------------#
# statsDict, statsDF = full_data_analyse_tracer_averages(
#     dataDict, Tlst, snapsRange, TRACERSPARAMS
# )
#
#
# # Add Stats of medians and percentiles
# tmplist = []
# for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
#     for T in Tlst:
#         tmp = statistics_hdf5_load(
#             T, rin, rout, DataSavepath, TRACERSPARAMS, DataSavepathSuffix
#         )
#         df = pd.DataFrame(tmp).astype("float64")
#         tmplist.append(df)
# tmpDF = pd.concat(tmplist, axis=0, join="outer", sort=False, ignore_index=True)
# finalDF = pd.concat([statsDF, tmpDF], axis=1, join="outer", sort=False)
#
# # Sort Column Order
# cols = list(finalDF.columns.values)
# for item in singleVals:
#     cols.remove(item)
# cols = singleVals + cols
# finalDF = finalDF[cols]
# finalDF = finalDF.astype(
#     {"Snap": int32, "T": float64, "Rinner": float64, "Router": float64}
# )
#
# # Save
# savePath = DataSavepath + "_Statistics-Table.csv"
#
# print("\n" + f"Saving Stats table .csv as {savePath}")
#
# finalDF.to_csv(savePath, index=False)

lookbackData = []

for snap in snapsRange:
    lookbackData.append(
        dataDict[
            (
                f"T{Tlst[0]}",
                f"{TRACERSPARAMS['Rinner'][0]}R{TRACERSPARAMS['Router'][0]}",
                f"{int(snap)}",
            )
        ]["Lookback"][0]
    )
    if int(snap) == int(TRACERSPARAMS["selectSnap"]):
        selectTime = abs(
            dataDict[
                (
                    f"T{Tlst[0]}",
                    f"{TRACERSPARAMS['Rinner'][0]}R{TRACERSPARAMS['Router'][0]}",
                    f"{int(snap)}",
                )
            ]["Lookback"][0]
        )
lookbackData = np.array(lookbackData)

# del statsDict,statsDF,tmplist,tmpDF,finalDF

timeAvDF = flat_analyse_time_averages(
    FlatDataDict, Tlst, snapsRange, lookbackData, TRACERSPARAMS
)

# Save
savePath = DataSavepath + "_Time-Averages-Statistics-Table.csv"

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

    # fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (xsize,ysize), sharey=True)
    #
    # ax = plotDF.T.plot.bar(rot=0,figsize = (xsize,ysize),color=colour)
    # ax.legend(loc='upper right',title="Log10(T) [K]",fontsize=13)
    # plt.xticks(rotation=30,ha='right',fontsize=13)
    # plt.title(f"Percentage of Tracers Ever Meeting Criterion Pre and Post Selection at {selectTime:3.2f} Gyr" +\
    # "\n"+ r"selected by $T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
    # r" and $%05.2f \leq R \leq %05.2f kpc $"%(rin, rout), fontsize=16)
    #
    #
    # plt.annotate(text="Ever Matched Feature", xy=(0.25,0.02), xytext=(0.25,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
    # plt.annotate(text="", xy=(0.05,0.01), xytext=(0.49,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
    #
    # plt.annotate(text="Median Matched Feature", xy=(0.60,0.02), xytext=(0.60,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
    # plt.annotate(text="", xy=(0.51,0.01), xytext=(0.825,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
    #
    # plt.annotate(text="+/-2 Time-steps Matched Feature", xy=(0.85,0.02), xytext=(0.85,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
    # plt.annotate(text="", xy=(0.835,0.01), xytext=(1.00,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
    # fig.transFigure
    #
    # ax.yaxis.set_minor_locator(AutoMinorLocator())
    # ax.tick_params(which='both')
    # plt.grid(which='both',axis='y')
    # plt.ylabel('% of Tracers Selected Following Feature')
    # plt.tight_layout()
    # plt.subplots_adjust(top=0.90, bottom = 0.25, left=0.10, right=0.95)
    #
    #
    # opslaan = "./" + saveHalo + "/" + f"{int(rin)}R{int(rout)}" + "/"+ f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{rin}R{rout}_Stats-Bars.pdf"
    # plt.savefig(opslaan, dpi = DPI, transparent = False)
    # print(opslaan)
    # plt.close()

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

    ax.legend(loc="upper left", title="Log10(T) [K]", fontsize=13)
    plt.xticks(rotation=90, ha="right", fontsize=13)
    plt.title(
        r"Percentage of Tracers Ever Meeting Criterion Pre Selection at $t_{Lookback}$"
        + f"={selectTime:3.2f} Gyr"
        + "\n"
        + r"selected by $T = 10^{n \pm %05.2f} K$" % (TRACERSPARAMS["deltaT"])
        + r" and $%05.2f \leq R \leq %05.2f kpc $" % (rin, rout),
        fontsize=16,
    )

    plt.annotate(
        text="",
        xy=(0.10, 0.25),
        xytext=(0.10, 0.05),
        arrowprops=dict(arrowstyle="-"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )
    plt.annotate(
        text="Ever Matched Feature",
        xy=(0.20, 0.02),
        xytext=(0.20, 0.02),
        textcoords=fig.transFigure,
        annotation_clip=False,
        fontsize=14,
    )
    plt.annotate(
        text="",
        xy=(0.10, 0.01),
        xytext=(0.54, 0.01),
        arrowprops=dict(arrowstyle="<->"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )
    plt.annotate(
        text="",
        xy=(0.56, 0.25),
        xytext=(0.56, 0.05),
        arrowprops=dict(arrowstyle="-"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )

    plt.annotate(
        text="Median Matched Feature",
        xy=(0.60, 0.02),
        xytext=(0.60, 0.02),
        textcoords=fig.transFigure,
        annotation_clip=False,
        fontsize=14,
    )
    plt.annotate(
        text="",
        xy=(0.58, 0.01),
        xytext=(0.76, 0.01),
        arrowprops=dict(arrowstyle="<->"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )
    plt.annotate(
        text="",
        xy=(0.78, 0.25),
        xytext=(0.78, 0.05),
        arrowprops=dict(arrowstyle="-"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )

    plt.annotate(
        text="-1 Snapshot Feature",
        xy=(0.80, 0.03),
        xytext=(0.80, 0.03),
        textcoords=fig.transFigure,
        annotation_clip=False,
        fontsize=14,
    )
    plt.annotate(
        text="",
        xy=(0.78, 0.01),
        xytext=(0.95, 0.01),
        arrowprops=dict(arrowstyle="<->"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )
    plt.annotate(
        text="",
        xy=(0.95, 0.25),
        xytext=(0.95, 0.05),
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
    plt.subplots_adjust(top=0.90, bottom=0.25, left=0.10, right=0.95)

    opslaan = (
        "./"
        + saveHalo
        + "/"
        + f"{int(rin)}R{int(rout)}"
        + "/"
        + f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{rin}R{rout}_Pre-Stats-Bars.pdf"
    )
    plt.savefig(opslaan, dpi=DPI, transparent=False)
    print(opslaan)
    plt.close()

    fig, ax = plt.subplots(
        nrows=1, ncols=1, figsize=(int(xsize / 2.0), ysize), sharey=True
    )

    postDF.T.plot.bar(rot=0, ax=ax, color=colour)

    ax.legend(loc="upper left", title="Log10(T) [K]", fontsize=13)
    plt.xticks(rotation=90, ha="right", fontsize=13)
    plt.title(
        r"Percentage of Tracers Ever Meeting Criterion Post Selection at $t_{Lookback}$"
        + f"={selectTime:3.2f} Gyr"
        + "\n"
        + r"selected by $T = 10^{n \pm %05.2f} K$" % (TRACERSPARAMS["deltaT"])
        + r" and $%05.2f \leq R \leq %05.2f kpc $" % (rin, rout),
        fontsize=16,
    )

    plt.annotate(
        text="",
        xy=(0.10, 0.25),
        xytext=(0.10, 0.05),
        arrowprops=dict(arrowstyle="-"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )
    plt.annotate(
        text="Ever Matched Feature",
        xy=(0.20, 0.02),
        xytext=(0.20, 0.02),
        textcoords=fig.transFigure,
        annotation_clip=False,
        fontsize=14,
    )
    plt.annotate(
        text="",
        xy=(0.10, 0.01),
        xytext=(0.54, 0.01),
        arrowprops=dict(arrowstyle="<->"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )
    plt.annotate(
        text="",
        xy=(0.56, 0.25),
        xytext=(0.56, 0.05),
        arrowprops=dict(arrowstyle="-"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )

    plt.annotate(
        text="Median Matched Feature",
        xy=(0.60, 0.02),
        xytext=(0.60, 0.02),
        textcoords=fig.transFigure,
        annotation_clip=False,
        fontsize=14,
    )
    plt.annotate(
        text="",
        xy=(0.58, 0.01),
        xytext=(0.76, 0.01),
        arrowprops=dict(arrowstyle="<->"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )
    plt.annotate(
        text="",
        xy=(0.78, 0.25),
        xytext=(0.78, 0.05),
        arrowprops=dict(arrowstyle="-"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )

    plt.annotate(
        text="+1 Snapshot Feature",
        xy=(0.80, 0.03),
        xytext=(0.80, 0.03),
        textcoords=fig.transFigure,
        annotation_clip=False,
        fontsize=14,
    )
    plt.annotate(
        text="",
        xy=(0.78, 0.01),
        xytext=(0.95, 0.01),
        arrowprops=dict(arrowstyle="<->"),
        xycoords=fig.transFigure,
        annotation_clip=False,
    )
    plt.annotate(
        text="",
        xy=(0.95, 0.25),
        xytext=(0.95, 0.05),
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
    plt.subplots_adjust(top=0.90, bottom=0.25, left=0.10, right=0.95)

    opslaan = (
        "./"
        + saveHalo
        + "/"
        + f"{int(rin)}R{int(rout)}"
        + "/"
        + f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{rin}R{rout}_Post-Stats-Bars.pdf"
    )
    plt.savefig(opslaan, dpi=DPI, transparent=False)
    print(opslaan)
    plt.close()
