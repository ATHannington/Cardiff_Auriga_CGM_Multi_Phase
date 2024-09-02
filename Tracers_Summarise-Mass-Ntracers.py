import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import const as c
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *
import copy

# Input parameters path:
TracersParamsPath = "TracersParams.csv"
TracersMasterParamsPath = "TracersParamsMaster.csv"
SelectedHaloesPath = "TracersSelectedHaloes.csv"
ISMpercentile = 99.5 # 3 sigma
timeAverageBool = False


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

# Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersMasterParamsPath)

# Load Halo Selection Data
SELECTEDHALOES, HALOPATHS = load_haloes_selected(
    HaloPathBase=TRACERSPARAMS["savepath"], SelectedHaloesPath=SelectedHaloesPath
)

saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

DataSavepathSuffix = f".h5"

if timeAverageBool is True:
    snapRange = [
        snap
        for snap in range(
            int(TRACERSPARAMS["snapMin"]),
            min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
            1,
        )
    ]
else:
    snapRange = [int(TRACERSPARAMS["selectSnap"])]
# int(TRACERSPARAMS["selectSnap"])


rinList = []
routList = []
fullTList = []
fullSnapRangeList = []
for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
    print(f"{rin}R{rout}")
    for (ii, T) in enumerate(Tlst):
        for snapNumber in snapRange:
            rinList.append(rin)
            routList.append(rout)
            fullTList.append(float(T))
            fullSnapRangeList.append(snapNumber)

blankList = np.array([0.0 for xx in range(len(rinList))])
summaryDictTemplate = {
    "R_inner [kpc]": np.array(rinList),
    "R_outer [kpc]": np.array(routList),
    "Log10(T) [K]": np.array(fullTList),
    "Snap Number": np.array(fullSnapRangeList),
    f"Average R200c (per halo) [kpc]": blankList.copy(),
    f"Average {ISMpercentile}th percentile R_ISM (per halo) [kpc]": blankList.copy(),
    f"Total N_tracers (all haloes) within R200c": blankList.copy(),
    f"Total gas mass (all haloes) within R200c [msol]": blankList.copy(),
    f"Average gas n_H density (per halo) within R200c [cm-3]": blankList.copy(),
    f"Total N_tracers (all haloes) available in spherical shell": blankList.copy(),
    f"Total gas mass (all haloes) available in spherical shell [msol]": blankList.copy(),
    f"Average gas n_H density (per halo) available in spherical shell [cm-3]": blankList.copy(),
    f"Total N_tracers (all haloes) per temperature in spherical shell": blankList.copy(),
    f"Total gas mass (all haloes) per temperature in spherical shell [msol]": blankList.copy(),
    "Average gas n_H density (per halo) per temperature in spherical shell [cm-3]": blankList.copy(),
}

TRACERSPARAMSCOMBI = TRACERSPARAMS
fullSummaryDict = {}

rotation_matrix = None
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
for halo, loadPath in zip(SELECTEDHALOES, HALOPATHS):
    haloPath = TRACERSPARAMSCOMBI["simfile"] + halo + "/output/"
    print("")
    print("")
    print(f"Starting {halo}")
    summaryDict = copy.deepcopy(summaryDictTemplate)
    for snapNumber in snapRange:
        print("")
        print(f"[@{int(snapNumber)}]: Starting Snap {snapNumber}")

        # load in the subfind group files
        snap_subfind = load_subfind(snapNumber, dir=haloPath)

        # load in the gas particles mass and position only for HaloID 0.
        #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
        #       gas and stars (type 0 and 4) MUST be loaded first!!
        snap = gadget_readsnap(
            snapNumber,
            haloPath,
            hdf5=True,
            loadonlytype=[0,4],
            lazy_load=True,
            subfind=snap_subfind,
        )

        snapTracers = gadget_readsnap(
            snapNumber,
            haloPath,
            hdf5=True,
            loadonlytype=[6],
            lazy_load=True,
            subfind=snap_subfind,
        )

        # Load Cell IDs - avoids having to turn lazy_load off...
        # But ensures 'id' is loaded into memory before halo_only_gas_select is called
        #  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
        #   Be in memory so taking the subset would be skipped.

        tmp = snap.data["id"]
        tmp = snap.data["age"]
        tmp = snap.data["hrgm"]
        tmp = snap.data["mass"]
        tmp = snap.data["pos"]
        tmp = snap.data["vol"]
        del tmp

        print(
            f"[@{int(snapNumber)}]: SnapShot loaded at RedShift z={snap.redshift:0.05e}"
        )

        # Centre the simulation on HaloID 0
        # snap = set_centre(
        #     snap=snap, snap_subfind=snap_subfind, HaloID=HaloID, snapNumber=snapNumber
        # )

        snap.calc_sf_indizes(snap_subfind)
        if rotation_matrix is None:
            _ = snap.select_halo(snap_subfind, do_rotation=True)
        else:
            snap.select_halo(snap_subfind, do_rotation=False)
            snap.rotateto(
                rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
            )

        # --------------------------#
        ##    Units Conversion    ##
        # --------------------------#

        # Convert Units
        # Make this a seperate function at some point??
        snap.pos *= 1e3  # [kpc]
        snap.vol *= 1e9  # [kpc^3]
        snap.mass *= 1e10  # [Msol]
        snap.hrgm *= 1e10  # [Msol]
        # [Kpc]

        snap = high_res_only_gas_select(snap, snapNumber)

        # Calculate New Parameters and Load into memory others we want to track
        snap = calculate_tracked_parameters(
            snap,
            elements,
            elements_Z,
            elements_mass,
            elements_solar,
            Zsolar,
            omegabaryon0,
            snapNumber,
            paramsOfInterest=["T","R","n_H"],
            mappingBool=True,
            numthreads=8,
            verbose = False,
        )

        whereGas = np.where(snap.type == 0)[0]

        rvir = (snap_subfind.data["frc2"] * 1e3)[int(0)]
        print(f"For {halo} at snap {snapNumber} R200c [kpc] = ", rvir)

        dictRowSnap = np.where((summaryDict["Snap Number"] == snapNumber))[0]

        summaryDict[f"Average R200c (per halo) [kpc]"][dictRowSnap] = np.full(shape=summaryDict[f"Average R200c (per halo) [kpc]"][dictRowSnap].shape, fill_value=rvir)

        whereCentralISM = np.where((snap.sfr > 0.0)&(snap.subhalo[whereGas] == 0)&(snap.halo[whereGas] == 0))[0]
        radiusISMproxy = np.nanpercentile(snap.data["R"][whereCentralISM], ISMpercentile, axis=0)
        summaryDict[f"Average {ISMpercentile}th percentile R_ISM (per halo) [kpc]"][dictRowSnap] = np.full(shape=summaryDict[f"Average {ISMpercentile}th percentile R_ISM (per halo) [kpc]"][dictRowSnap].shape, fill_value=radiusISMproxy)

        print(f"For {halo} at snap {snapNumber} {ISMpercentile}th percentile R_ISM [kpc] = ", radiusISMproxy)


        ###-------------------------------------------
        #   Find total number of tracers and gas mass in halo within rVir
        ###-------------------------------------------
        Cond = np.where(
            (snap.data["R"][whereGas] <= rvir)
            & (np.isin(snap.data["subhalo"][whereGas], np.array([-1.0, 0.0])))
        )[0]

        # Select Cell IDs for cells which meet condition
        CellIDs = snap.id[whereGas][Cond]

        # Select Parent IDs in Cond list
        #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
        ParentsIndices = np.where(np.isin(snapTracers.prid[whereGas], CellIDs))

        # Select Tracers and Parent IDs from cells that meet condition and contain tracers
        Tracers = snapTracers.trid[whereGas][ParentsIndices]
        # Parents = snapTracers.prid[whereGas][ParentsIndices]

        # # Get Gas meeting Cond AND with tracers
        # CellsIndices = np.where(np.isin(snap.id[whereGas], Parents))
        # CellIDs = snap.id[whereGas][CellsIndices]

        Ntracers = int(len(Tracers))
        print(f"[@{snapNumber}]: Number of tracers = {Ntracers}")

        NtracersTotalR = np.shape(Tracers)[0]

        print(
            f"For {halo} at snap {snapNumber} Total N_tracers (all haloes) within R200c = ",
            NtracersTotalR,
        )
        
        summaryDict[f"Total N_tracers (all haloes) within R200c"][dictRowSnap] = np.full(shape=summaryDict[f"Total N_tracers (all haloes) within R200c"][dictRowSnap].shape, fill_value=NtracersTotalR)

        massTotalR = np.sum(snap.data["mass"][whereGas][Cond])
        print(f"{halo} {snapNumber} Total gas mass (all haloes) within R200c [msol]", massTotalR)
        
        summaryDict[f"Total gas mass (all haloes) within R200c [msol]"][
            dictRowSnap
        ] = np.full(shape = summaryDict[f"Total gas mass (all haloes) within R200c [msol]"][dictRowSnap].shape, fill_value=massTotalR)

        nHRvir = np.nanmedian(snap.data["n_H"][whereGas][Cond])
        print(f"{halo} {snapNumber} Average gas n_H density (per halo) within R200c [cm-3]", nHRvir)
        
        summaryDict[f"Average gas n_H density (per halo) within R200c [cm-3]"][
            dictRowSnap
        ] = np.full(shape = summaryDict[f"Average gas n_H density (per halo) within R200c [cm-3]"][dictRowSnap].shape, fill_value=nHRvir)

        ###-------------------------------------------
        #   Load in analysed data
        ###-------------------------------------------
        loadPath += "/"
        TRACERSPARAMS, DataSavepath, _ = load_tracers_parameters(
            loadPath + TracersParamsPath
        )
        saveParams += TRACERSPARAMS["saveParams"]

        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            print(f"{rin}R{rout}")
            whereGas = np.where(snap.type == 0)[0]

            Cond = np.where(
                (snap.data["R"][whereGas] >= rin)
                & (snap.data["R"][whereGas] <= rout)
                & (np.isin(snap.data["subhalo"][whereGas], np.array([-1.0, 0.0])))
            )[0]

            massR = np.sum(snap.data["mass"][whereGas][Cond])
            dictRowSelectRonly = np.where(
                (summaryDict["R_inner [kpc]"] == rin)
                & (summaryDict["R_outer [kpc]"] == rout)
                & (summaryDict["Snap Number"] == snapNumber)
            )[0]

            print(f"Total mass (all haloes) in spherical shell [msol] = ", massR)

            summaryDict[
                f"Total gas mass (all haloes) available in spherical shell [msol]"
            ][dictRowSelectRonly] = np.full(shape=summaryDict[f"Total gas mass (all haloes) available in spherical shell [msol]"][dictRowSelectRonly].shape, fill_value=massR)

            # Select Cell IDs for cells which meet condition
            CellIDs = snap.id[whereGas][Cond]

            # Select Parent IDs in Cond list
            #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
            ParentsIndices = np.where(np.isin(snapTracers.prid[whereGas], CellIDs))

            # Select Tracers and Parent IDs from cells that meet condition and contain tracers
            Tracers = snapTracers.trid[whereGas][ParentsIndices]
            nTracersR = int(len(Tracers))

            dictRowSelectRonly = np.where(
                (summaryDict["R_inner [kpc]"] == rin)
                & (summaryDict["R_outer [kpc]"] == rout)
                & (summaryDict["Snap Number"] == snapNumber)
            )[0]

            print(f"Total N_tracers (all haloes) available in spherical shell = ", nTracersR)

            summaryDict[
                f"Total N_tracers (all haloes) available in spherical shell"
            ][dictRowSelectRonly] = np.full(shape=summaryDict[f"Total N_tracers (all haloes) available in spherical shell"][dictRowSelectRonly].shape, fill_value=nTracersR)

            nHR = np.nanmedian(snap.data["n_H"][whereGas][Cond])
            print(f"Average gas n_H density (per halo) available in spherical shell [cm-3] = ", nHR)

            summaryDict[
                f"Average gas n_H density (per halo) available in spherical shell [cm-3]"
            ][dictRowSelectRonly] = np.full(shape=summaryDict[f"Average gas n_H density (per halo) available in spherical shell [cm-3]"][dictRowSelectRonly].shape, fill_value=nHR)
        
            for (ii, T) in enumerate(Tlst):

                FullDictKey = (f"T{float(T)}", f"{rin}R{rout}", f"{snapNumber}")
                print(FullDictKey)


                dictRowSelect = np.where(
                    (summaryDict["R_inner [kpc]"] == rin)
                    & (summaryDict["R_outer [kpc]"] == rout)
                    & (summaryDict["Log10(T) [K]"] == float(T))
                    & (summaryDict["Snap Number"] == snapNumber)
                )[0]

                Cond = np.where(
                    (snap.data["R"][whereGas] >= rin)
                    & (snap.data["R"][whereGas] <= rout)
                    & (
                        snap.data["T"][whereGas]
                        >= 1.0 * 10 ** (float(T) - TRACERSPARAMS["deltaT"])
                    )
                    & (
                        snap.data["T"][whereGas]
                        <= 1.0 * 10 ** (float(T) + TRACERSPARAMS["deltaT"])
                    )
                    & (np.isin(snap.data["subhalo"][whereGas], np.array([-1.0, 0.0])))
                )[0]

                massRT = np.sum(snap.data["mass"][whereGas][Cond])
                summaryDict[f"Total gas mass (all haloes) per temperature in spherical shell [msol]"][dictRowSelect] = np.full(shape=summaryDict[f"Total gas mass (all haloes) per temperature in spherical shell [msol]"][dictRowSelect].shape, fill_value=massRT)
                print(
                    f"Total gas mass (all haloes) per temperature in spherical shell [msol]",
                    FullDictKey,
                    massRT,
                )

                # Select Cell IDs for cells which meet condition
                CellIDs = snap.id[whereGas][Cond]

                # Select Parent IDs in Cond list
                #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
                ParentsIndices = np.where(np.isin(snapTracers.prid[whereGas], CellIDs))

                # Select Tracers and Parent IDs from cells that meet condition and contain tracers
                Tracers = snapTracers.trid[whereGas][ParentsIndices]
                nTracersRT = int(len(Tracers))

                print(f"Total N_tracers (all haloes) per temperature in spherical shell = ", nTracersRT)

                summaryDict[
                    f"Total N_tracers (all haloes) per temperature in spherical shell"
                ][dictRowSelect] = np.full(shape=summaryDict[f"Total N_tracers (all haloes) per temperature in spherical shell"][dictRowSelect].shape, fill_value=nTracersRT)


                nHRT = np.median(snap.data["n_H"][whereGas][Cond])
                summaryDict["Average gas n_H density (per halo) per temperature in spherical shell [cm-3]"][dictRowSelect] = np.full(shape=summaryDict["Average gas n_H density (per halo) per temperature in spherical shell [cm-3]"][dictRowSelect].shape,fill_value=nHRT)
    fullSummaryDict[f"{halo}"] = copy.deepcopy(summaryDict)
#-------------------------------------------------------------------------------------#
perHaloSummaryDict = copy.deepcopy(fullSummaryDict[list(fullSummaryDict.keys())[0]])
# perHaloSummaryDict = {}
for key in summaryDictTemplate.keys():
    if key not in ["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"]:
        for (ii, T) in enumerate(Tlst):
            for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
                tmp = []
                for hKey, data in fullSummaryDict.items():
                    FullDictKey = (f"T{float(T)}", f"{rin}R{rout}")
                    print(FullDictKey)
                    
                    dictRowSelect = np.where(
                        (fullSummaryDict[hKey]["R_inner [kpc]"] == rin)
                        & (fullSummaryDict[hKey]["R_outer [kpc]"] == rout)
                        & (fullSummaryDict[hKey]["Log10(T) [K]"] == float(T))
                    )[0]

                    tmp.append(data[key][dictRowSelect])
                tmpArray = np.asarray(tmp)
                if tmpArray.ndim >2: raise Exception("nDim of data from fullSummaryDict >2. Expected 1 array len == nSnaps x nTemperatures x nRadialSelections . Check logic... ")

                tmpArray = tmpArray.reshape(-1,len(SELECTEDHALOES))

                splitKey = key.split(" ")
                keySummaryType = splitKey[0]

                if keySummaryType == "Average":
                    perHaloSummaryDict[key][dictRowSelect] = np.nanmedian(tmpArray,axis=-1)

                elif keySummaryType == "Total":
                    perHaloSummaryDict[key][dictRowSelect] = np.sum(tmpArray,axis=-1)
                    averageKey = key.replace("Total", "Average")
                    averageKey = averageKey.replace("all haloes", "per halo")
                    if averageKey not in list(perHaloSummaryDict.keys()):
                        perHaloSummaryDict[averageKey] = np.zeros(len(perHaloSummaryDict[key]))
                    perHaloSummaryDict[averageKey][dictRowSelect] = np.nanmedian(tmpArray,axis=-1)
                else:
                    raise Exception(f"Summary key {key} begins with {keySummaryType} which is not understood. Please begin key with 'Average' or 'Total' or alter logic")

#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#

if timeAverageBool is True:
    ntracersDict = {"halo": [], "Snap Number": [], "R_inner [kpc]": [], "R_outer [kpc]": [], "Log10(T) [K]": [], "N_tracers": [],"Gas mass": []}
else:
    ntracersDict = {"halo": [], "R_inner [kpc]": [], "R_outer [kpc]": [], "Log10(T) [K]": [], "N_tracers": [],"Gas mass": []}
nSnaps = float(len(snapRange))
fullTracerTotal = 0
fullMassTotal = 0
for halo, loadPath in zip(SELECTEDHALOES, HALOPATHS):
    haloPath = TRACERSPARAMSCOMBI["simfile"] + halo + "/output/"

    loadPath += "/"
    _, DataSavepath, _ = load_tracers_parameters(
            loadPath + TracersParamsPath
     )
    

    saveHalo = (halo.split("_"))[-1]
    if "L" in saveHalo:
        saveHalo = saveHalo.split("L")[-1]
        saveHalo = int(saveHalo) + 50
    else:
        saveHalo = int(saveHalo)

    print("")
    print(f"Loading {halo} Data!")
    for T in Tlst:
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            if timeAverageBool is True:
                dataLoadPath = (
                    DataSavepath
                    + f"_T{T}_{rin}R{rout}_flat-wrt-time"
                    + ".h5"
                )
            
                tmp = hdf5_load(dataLoadPath)


                for jj,snapNumber in enumerate(snapRange):

                    FullDictKey = (f"T{float(T)}", f"{rin}R{rout}")

                    print(FullDictKey,f" snapNumber {snapNumber}: #{(jj+1)} of {int(nSnaps)}")

                    whereGas = np.where(tmp[FullDictKey]["type"][jj,:]==0)[0]
                    ntr = float(np.shape(tmp[FullDictKey]["trid"][jj,:])[-1])
                    massGas = np.sum(tmp[FullDictKey]["mass"][jj,whereGas])
                    "Andy, do we want to be able to perform time-averaging of the `selected` mass? If so, need to iterate over snapNumber and change if timeAverageBool is True stuff below..."
                    ntracersDict["halo"].append(saveHalo)
                    ntracersDict["Snap Number"].append(snapNumber)
                    ntracersDict["R_inner [kpc]"].append(rin)
                    ntracersDict["R_outer [kpc]"].append(rout)
                    ntracersDict["Log10(T) [K]"].append(float(T))
                    ntracersDict["N_tracers"].append(ntr)
                    ntracersDict["Gas mass"].append(massGas)
            else:
                dataLoadPath = (
                    DataSavepath
                    + f"_T{T}_{rin}R{rout}_flat-wrt-time"
                    + ".h5"
                )
            
                tmp = hdf5_load(dataLoadPath)

                timeIndex = np.where(
                    np.array([
                            snap
                            for snap in range(
                                int(TRACERSPARAMS["snapMin"]),
                                min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"]) + 1),
                                1,
                            )
                        ]) == int(TRACERSPARAMS["selectSnap"])
                    )[0]
                

                FullDictKey = (f"T{float(T)}", f"{rin}R{rout}")

                print(FullDictKey,f" snapNumber {snapNumber}")

                whereGas = np.where(tmp[FullDictKey]["type"][timeIndex,:]==0)[0]
                ntr = float(np.shape(tmp[FullDictKey]["trid"][timeIndex,:])[-1])
                massGas = np.sum(tmp[FullDictKey]["mass"][timeIndex,whereGas])
                # "Andy, do we want to be able to perform time-averaging of the `selected` mass? If so, need to iterate over snapNumber and change if timeAverageBool is True stuff below..."
                ntracersDict["halo"].append(saveHalo)
                # # # ntracersDict["Snap Number"].append(snapNumber)
                ntracersDict["R_inner [kpc]"].append(rin)
                ntracersDict["R_outer [kpc]"].append(rout)
                ntracersDict["Log10(T) [K]"].append(float(T))
                ntracersDict["N_tracers"].append(ntr)
                ntracersDict["Gas mass"].append(massGas)                

# tracerTotalsDict ={"R_inner [kpc]": "All", "R_outer [kpc]": "All", "Log10(T) [K]": "All", "Snap Number": "All", "N_tracers" : fullTracerTotal, "Gas mass" : fullMassTotal}
ntracersDict = {key: np.asarray(val).astype(np.float64) if key not in ["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"] else np.asarray(val) for key, val in ntracersDict.items()}
# ntracersDict = {}
# fullTracerTotal = 0
# for snapNumber in snapRange:
#     for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
#         for (ii, T) in enumerate(Tlst):
#             FullDictKey = (f"T{float(T)}", f"{rin}R{rout}", f"{int(snapNumber)}")
#             print(FullDictKey)
#             tmpNtracers = float(np.shape(flatMergedDict[FullDictKey]["trid"])[0])
#             newKey = (f"T{float(T)}", f"{rin}R{rout}")
#             try:
#                 ntracersDict[newKey].append(tmpNtracers)
#             except:
#                 ntracersDict.update({newKey : copy.deepcopy([tmpNtracers])})
#             fullTracerTotal += tmpNtracers

# ntracersDict.update({("T_All", "R_All") : np.array(fullTracerTotal)})

if timeAverageBool is True:
    averagedSummaryDict = {"R_inner [kpc]": [], "R_outer [kpc]": [], "Log10(T) [K]": []}
    nSnaps = float(len(snapRange))
    for (ii, T) in enumerate(Tlst):
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            print(f"Data from selection Averaging",f"T{float(T)}", f"{rin}R{rout}")
            print(FullDictKey)
            
            dictRowSelect = np.where(
                (perHaloSummaryDict["R_inner [kpc]"] == rin)
                & (perHaloSummaryDict["R_outer [kpc]"] == rout)
                & (perHaloSummaryDict["Log10(T) [K]"] == float(T))
            )[0]

            averagedSummaryDict["R_inner [kpc]"].append(rin) 
            averagedSummaryDict["R_outer [kpc]"].append(rout)
            averagedSummaryDict["Log10(T) [K]"].append(float(T))

            for key in perHaloSummaryDict.keys():
                if key not in ["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"]:
                    if key not in averagedSummaryDict.keys():
                        averagedSummaryDict[key] = []

                    averagedSummaryDict[key].append(np.nanmedian(perHaloSummaryDict[key][dictRowSelect]))

    timeAveragedNtracersDict = {"halo": [],"R_inner [kpc]": [], "R_outer [kpc]": [], "Log10(T) [K]": [], "N_tracers": []}
    nSnaps = float(len(snapRange))
    for halo, loadPath in zip(SELECTEDHALOES, HALOPATHS):
        saveHalo = (halo.split("_"))[-1]
        if "L" in saveHalo:
            saveHalo = saveHalo.split("L")[-1]
            saveHalo = int(saveHalo) + 50
        else:
            saveHalo = int(saveHalo)

        for (ii, T) in enumerate(Tlst):
            for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
                
                print(f"N_tracers Averaging",f"T{float(T)}", f"{rin}R{rout}")
                
                dictRowSelect = np.where(
                    (ntracersDict["halo"] == saveHalo)
                    & (ntracersDict["R_inner [kpc]"] == rin)
                    & (ntracersDict["R_outer [kpc]"] == rout)
                    & (ntracersDict["Log10(T) [K]"] == float(T))
                )[0]

                timeAveragedNtracersDict["halo"].append(saveHalo) 
                timeAveragedNtracersDict["R_inner [kpc]"].append(rin) 
                timeAveragedNtracersDict["R_outer [kpc]"].append(rout)
                timeAveragedNtracersDict["Log10(T) [K]"].append(float(T))

                for key in ntracersDict.keys():
                    if key not in ["halo", "R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"]:
                        if key not in timeAveragedNtracersDict.keys():
                            timeAveragedNtracersDict[key] = []

                        timeAveragedNtracersDict[key].append(np.nanmedian(ntracersDict[key][dictRowSelect]))

try:
    tracersDictToAverage = copy.deepcopy(timeAveragedNtracersDict)
except:
    tracersDictToAverage = copy.deepcopy(ntracersDict)
#-------------------------------------------------------------------------------------#
print("Load Full Time Flattened Data!")
flatMergedDict, _ = multi_halo_merge_flat_wrt_time(
    SELECTEDHALOES, HALOPATHS, DataSavepathSuffix, snapRange, Tlst, TracersParamsPath
)
tmptotalNtracersDict = {"Snap Number": [],"R_inner [kpc]": [], "R_outer [kpc]": [], "Log10(T) [K]": [], "Total_N_tracers": []}
fullTracerTotal = 0
for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
    for T in Tlst:
        FullDictKey = (f"T{float(T)}", f"{rin}R{rout}")
        print(FullDictKey)
        tmpNtracers = float(np.shape(flatMergedDict[FullDictKey]["trid"][0,:])[-1])
        tmptotalNtracersDict["R_inner [kpc]"].append(rin) 
        tmptotalNtracersDict["R_outer [kpc]"].append(rout)
        tmptotalNtracersDict["Log10(T) [K]"].append(float(T))
        tmptotalNtracersDict["Total_N_tracers"].append(tmpNtracers)

tmptotalNtracersDict = {key: np.asarray(val).astype(np.float64) if key not in ["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"] else np.asarray(val) for key, val in tmptotalNtracersDict.items() }

totalNtracersDict = {"Snap Number": [],"R_inner [kpc]": [], "R_outer [kpc]": [], "Log10(T) [K]": [], "Total_N_tracers": []}
fullTracerTotal = 0
for ii,snapNumber in enumerate(snapRange):
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        for T in Tlst:
            FullDictKey = (f"T{float(T)}", f"{rin}R{rout}")
            print(FullDictKey)
            dictRowSelect = np.where(
                (tmptotalNtracersDict["R_inner [kpc]"] == rin)
                & (tmptotalNtracersDict["R_outer [kpc]"] == rout)
                & (tmptotalNtracersDict["Log10(T) [K]"] == float(T))
            )[0]
            totalNtracersDict["R_inner [kpc]"].append(rin) 
            totalNtracersDict["R_outer [kpc]"].append(rout)
            totalNtracersDict["Log10(T) [K]"].append(float(T))
            totalNtracersDict["Snap Number"].append(snapNumber) 
            totalNtracersDict["Total_N_tracers"].append(np.sum(tmptotalNtracersDict["Total_N_tracers"][dictRowSelect]))

totalNtracersDict = {key: np.asarray(val).astype(np.float64) if key not in ["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"] else np.asarray(val) for key, val in totalNtracersDict.items() }

timeAveragedTotalNtracersDict = {"R_inner [kpc]": [], "R_outer [kpc]": [], "Log10(T) [K]": [], "Total_N_tracers": []}
nSnaps = float(len(snapRange))
for (ii, T) in enumerate(Tlst):
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        
        print(f"N_tracers Averaging",f"T{float(T)}", f"{rin}R{rout}")
        
        dictRowSelect = np.where(
            (totalNtracersDict["R_inner [kpc]"] == rin)
            & (totalNtracersDict["R_outer [kpc]"] == rout)
            & (totalNtracersDict["Log10(T) [K]"] == float(T))
        )[0]

        timeAveragedTotalNtracersDict["R_inner [kpc]"].append(rin) 
        timeAveragedTotalNtracersDict["R_outer [kpc]"].append(rout)
        timeAveragedTotalNtracersDict["Log10(T) [K]"].append(float(T))

        for key in totalNtracersDict.keys():
            if key not in ["halo", "R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"]:
                if key not in timeAveragedTotalNtracersDict.keys():
                    timeAveragedTotalNtracersDict[key] = []

                timeAveragedTotalNtracersDict[key].append(np.nanmedian(totalNtracersDict[key][dictRowSelect]))


timeAveragedTotalNtracersDict = {key: np.asarray(val).astype(np.float64) if key not in ["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"] else np.asarray(val) for key, val in timeAveragedTotalNtracersDict.items() }


#-------------------------------------------------------------------------------------#
tracersDictToAverage = {key: np.asarray(val).astype(np.float64) if key not in ["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"] else np.asarray(val) for key, val in tracersDictToAverage.items() }

averagedNtracersDict = {"R_inner [kpc]": [], "R_outer [kpc]": [], "Log10(T) [K]": [], "N_tracers": []}
nSnaps = float(len(snapRange))
for (ii, T) in enumerate(Tlst):
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        
        print(f"N_tracers Averaging",f"T{float(T)}", f"{rin}R{rout}")
        
        dictRowSelect = np.where(
            (tracersDictToAverage["R_inner [kpc]"] == rin)
            & (tracersDictToAverage["R_outer [kpc]"] == rout)
            & (tracersDictToAverage["Log10(T) [K]"] == float(T))
        )[0]

        averagedNtracersDict["R_inner [kpc]"].append(rin) 
        averagedNtracersDict["R_outer [kpc]"].append(rout)
        averagedNtracersDict["Log10(T) [K]"].append(float(T))

        for key in tracersDictToAverage.keys():
            if key not in ["halo", "R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"]:
                if key not in averagedNtracersDict.keys():
                    averagedNtracersDict[key] = []

                averagedNtracersDict[key].append(np.nanmedian(tracersDictToAverage[key][dictRowSelect]))

averagedNtracersDict.update(timeAveragedTotalNtracersDict)

if timeAverageBool is True:
    summaryDf = pd.DataFrame(averagedSummaryDict, index=[ii for ii in range(len(averagedSummaryDict["Log10(T) [K]"]))])
    summaryDfAllSnaps = pd.DataFrame(perHaloSummaryDict, index=[ii for ii in range(len(perHaloSummaryDict["Log10(T) [K]"]))])
    # del tracerTotalsDict["Snap Number"]
    # averagedNtracersDict = {key: np.concatenate((val,np.asarray([tracerTotalsDict[key]]))) for key, val in averagedNtracersDict.items()}
    tracersDf = pd.DataFrame(averagedNtracersDict, index=[ii for ii in range(len(averagedNtracersDict["Log10(T) [K]"]))])
else:
    summaryDfAllSnaps = pd.DataFrame(perHaloSummaryDict, index=[ii for ii in range(len(perHaloSummaryDict["Log10(T) [K]"]))])
    summaryDf = copy.deepcopy(summaryDfAllSnaps)
    # ntracersDict = {key: np.concatenate((val,np.asarray([tracerTotalsDict[key]]))) for key, val in ntracersDict.items()}
    tracersDf = pd.DataFrame(averagedNtracersDict, index=[ii for ii in range(len(averagedNtracersDict["Log10(T) [K]"]))])

nHaloes = float(len(SELECTEDHALOES))
summaryDf["Number of Haloes"] = nHaloes

print(summaryDf.head(n=20))
print(tracersDf.head(n=20))
print(summaryDfAllSnaps.head(n=20))

if timeAverageBool is True:
    savePath = "Data_Tracers_MultiHalo_Mass-Ntracers-Summary_Time-Average.xlsx"
else:
    savePath = "Data_Tracers_MultiHalo_Mass-Ntracers-Summary.xlsx"

excel = pd.ExcelWriter(path=savePath,mode="w")
with excel as writer:
    summaryDf.to_excel(writer,sheet_name="Halo_summary", index=False)
    tracersDf.to_excel(writer,sheet_name="Selected_summary", index=False)
    summaryDfAllSnaps.to_excel(writer,sheet_name="Halo_all_snaps", index=False)