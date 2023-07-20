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
Rmin = 0.0#min(TRACERSPARAMSCOMBI["Rinner"])
Rmax = 200.0#max(TRACERSPARAMSCOMBI["Router"])


TracersParamsPath = "TracersParams.csv"
TracersMasterParamsPath = "TracersParamsMaster.csv"
SelectedHaloesPath = "TracersSelectedHaloes.csv"
timeAverageBool = True

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
summaryDict = {
    "R_inner [kpc]": np.array(rinList),
    "R_outer [kpc]": np.array(routList),
    "Log10(T) [K]": np.array(fullTList),
    "Snap Number": np.array(fullSnapRangeList),
    "Gas mass per temperature [msol]": blankList.copy(),
    "Gas n_H density per temperature [cm-3]": blankList.copy(),
    "Average Rvir [kpc]": blankList.copy(),
    "Total gas mass (all haloes) available in spherical shell [msol]": blankList.copy(),
    "Total N_tracers (all haloes) in spherical shell": blankList.copy(),
    f"Total N_tracers (all haloes) within {Rmax:2.2f} kpc": blankList.copy(),
    f"Total gas mass (all haloes) within {Rmax:2.2f} kpc [msol]": blankList.copy(),
}

TRACERSPARAMSCOMBI = TRACERSPARAMS

rotation_matrix = None
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
for snapNumber in snapRange:
    for halo, loadPath in zip(SELECTEDHALOES, HALOPATHS):
        haloPath = TRACERSPARAMSCOMBI["simfile"] + halo + "/output/"

        print("")
        print("")
        print(f"Starting {halo}")
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

        ###-------------------------------------------
        #   Find total number of tracers and gas mass in halo within rVir
        ###-------------------------------------------
        Cond = np.where(
            (snap.data["R"][whereGas] <= Rmax)
            & (snap.data["R"][whereGas] >= Rmin)
            & (snap.data["sfr"][whereGas] <= 0.0)
            & (np.isin(snap.data["subhalo"][whereGas], np.array([-1.0, 0.0])))
        )[0]

        # Select Cell IDs for cells which meet condition
        CellIDs = snap.id[whereGas][Cond]

        # Select Parent IDs in Cond list
        #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
        ParentsIndices = np.where(np.isin(snapTracers.prid[whereGas], CellIDs))

        # Select Tracers and Parent IDs from cells that meet condition and contain tracers
        Tracers = snapTracers.trid[whereGas][ParentsIndices]
        Parents = snapTracers.prid[whereGas][ParentsIndices]

        # Get Gas meeting Cond AND with tracers
        CellsIndices = np.where(np.isin(snap.id[whereGas], Parents))
        CellIDs = snap.id[whereGas][CellsIndices]

        Ntracers = int(len(Tracers))
        print(f"[@{snapNumber}]: Number of tracers = {Ntracers}")

        NtracersTotalR = np.shape(Tracers)[0]

        print(
            f"For {halo} at snap {snapNumber} Total N_tracers (all haloes) within {Rmax:2.2f} kpc = ",
            NtracersTotalR,
        )

        dictRowSelectSnaponly = np.where((summaryDict["Snap Number"] == snapNumber))[0]

        summaryDict[f"Total N_tracers (all haloes) within {Rmax:2.2f} kpc"][
            dictRowSelectSnaponly
        ] += NtracersTotalR

        massTotalR = np.sum(snap.data["mass"][whereGas][Cond])
        print(f"For {halo} at snap {snapNumber} total mass [msol] = ", massTotalR)
        summaryDict[f"Total gas mass (all haloes) within {Rmax:2.2f} kpc [msol]"][
            dictRowSelectSnaponly
        ] += massTotalR


        rvir = (snap_subfind.data["frc2"] * 1e3)[int(0)]
        print(f"For {halo} at snap {snapNumber} Rvir [kpc] = ", massTotalR)
        summaryDict["Average Rvir [kpc]"][
            dictRowSelectSnaponly
        ] += rvir
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
                & (snap.data["sfr"][whereGas] <= 0)
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
                "Total gas mass (all haloes) available in spherical shell [msol]"
            ][dictRowSelectRonly] += massR
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
                summaryDict["Gas mass per temperature [msol]"][dictRowSelect] += massRT
                print(
                    f"Total mass (all haloes) in spherical shell per temperature [msol] = ",
                    FullDictKey,
                    massRT,
                )

                n_H_RT = np.median(snap.data["n_H"][whereGas][Cond])
                summaryDict["Gas n_H density per temperature [cm-3]"][
                    dictRowSelect
                ] += n_H_RT
#-------------------------------------------------------------------------------------#
#    # print("summaryDict = ", summaryDict)
#
#-------------------------------------------------------------------------------------#
#-------------------------------------------------------------------------------------#
print("Load Time Flattened Data!")
flatMergedDict, _ = multi_halo_merge_flat_wrt_time(
    SELECTEDHALOES, HALOPATHS, DataSavepathSuffix, snapRange, Tlst, TracersParamsPath
)
#-------------------------------------------------------------------------------------#

radiusTracerTotal = {f"{rin}R{rout}": 0 for rin,rout in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"])}
temperatureTracerTotal = {f"T{float(T)}": 0 for T in Tlst}
fullTracerTotal = 0
for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
    for (ii, T) in enumerate(Tlst):
        fullDictKey = (f"T{float(T)}", f"{rin}R{rout}")
        print(FullDictKey)
        tmpNtracers = int(np.shape(flatMergedDict[fullDictKey]["trid"])[1])
        radiusTracerTotal[f"{rin}R{rout}"] += tmpNtracers
        temperatureTracerTotal[f"T{float(T)}"] += tmpNtracers  
        fullTracerTotal += tmpNtracers

nSnaps = float(len(snapRange))
for key, value in summaryDict.items():
    if key not in ["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]", "Snap Number"]:
        summaryDict[key] = value / nSnaps

tracersDict ={
    "R_inner [kpc]": np.array(rinList),
    "R_outer [kpc]": np.array(routList),
    "Log10(T) [K]": np.array(fullTList),
    "N_tracers selected per temperature" : np.asarray(list(list(temperatureTracerTotal.values())*len(snapRange))*len(TRACERSPARAMS["Rinner"])),
    "N_tracers selected per radius" : np.asarray(list(list(radiusTracerTotal.values())*len(Tlst))*len(snapRange)),
    "N_tracers selected" : np.asarray([fullTracerTotal]*len(fullTList)),
}

# combined = copy.deepcopy(summaryDict)
# combined.update(tracersDict)



df1 = pd.DataFrame(summaryDict, index=[ii for ii in range(len(blankList))])

df1 = df1.groupby(["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]"]).sum()

df2 = pd.DataFrame(tracersDict, index=[ii for ii in range(len(blankList))])

df2 = df2.groupby(["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]"]).median()


df = pd.concat([df1,df2],axis=1)

nHaloes = float(len(SELECTEDHALOES))
df["Number of Haloes"] = nHaloes



# summaryDict = {
#     "R_inner [kpc]": np.array(rinList),
#     "R_outer [kpc]": np.array(routList),
#     "Log10(T) [K]": np.array(fullTList),
#     "Snap Number": np.array(fullSnapRangeList),
#     "N_tracers selected": blankList.copy(),
#     "Gas mass per temperature [msol]": blankList.copy(),
#     "Gas n_H density per temperature [cm-3]": blankList.copy(),
#     "Average Rvir [kpc]": blankList.copy(),
#     "Total gas mass (all haloes) available in spherical shell [msol]": blankList.copy(),
#     "Total N_tracers (all haloes) in spherical shell": blankList.copy(),
#     f"Total N_tracers (all haloes) within {Rmax:2.2f} kpc": blankList.copy(),
#     f"Total gas mass (all haloes) within {Rmax:2.2f} kpc [msol]": blankList.copy(),
# }

df["Average gas mass (per halo) available in spherical shell [msol]"] = (
    df["Total gas mass (all haloes) available in spherical shell [msol]"].astype(
        "float64"
    )
    / nHaloes
)

df["Average N_tracers (per halo) in spherical shell"] = (
    df["Total N_tracers (all haloes) in spherical shell"].astype("float64") / nHaloes
)

df[f"Average N_tracers (per halo) within {Rmax:2.2f} kpc"] = (
    df[f"Total N_tracers (all haloes) within {Rmax:2.2f} kpc"].astype("float64")
    / nHaloes
)

df["Average N_tracers selected (per halo)"] = (
    df["N_tracers selected"].astype("float64") / nHaloes
)

df["Average N_tracers selected per temperature (per halo)"] = (
    df["N_tracers selected per temperature"].astype("float64") / nHaloes
)

df["Average N_tracers selected per radius (per halo)"] = (
    df["N_tracers selected per radius"].astype("float64") / nHaloes
)


#----#
#     "Gas mass per temperature [msol]": blankList.copy(),
#     "Gas n_H density per temperature [cm-3]": blankList.copy(),
#     "Average Rvir [kpc]": blankList.copy(),
#     "Total gas mass (all haloes) available in spherical shell [msol]": blankList.copy(),

df[f"Average gas mass (per halo) within {Rmax:2.2f} kpc [msol]"] = (
    df[f"Total gas mass (all haloes) within {Rmax:2.2f} kpc [msol]"].astype("float64")
    / nHaloes
)

df[f"Average gas mass per temperature (per halo) within {Rmax:2.2f} kpc [msol]"] = (
    df[f"Gas mass per temperature [msol]"].astype("float64")
    / nHaloes
)

df[f"Average gas n_H density (per halo) within [cm-3]"] = (
    df["Gas n_H density per temperature [cm-3]"].astype("float64")
    / nHaloes
)

df[f"Average Rvir (per halo) [kpc]"] = (
    df["Average Rvir [kpc]"].astype("float64")
    / nHaloes
)



print(df.head(n=20))
if timeAverageBool is True:
    df.to_csv(
        "Data_Tracers_MultiHalo_Mass-Ntracers-Summary_Time-Average.csv", index=False
    )
else:
    df.to_csv("Data_Tracers_MultiHalo_Mass-Ntracers-Summary.csv", index=False)
