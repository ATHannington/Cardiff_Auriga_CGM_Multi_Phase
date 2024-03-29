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

# Input parameters path:
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
    "N_tracers selected": blankList.copy(),
    "N_tracers per temperature": blankList.copy(),
    "Gas mass selected [msol]": blankList.copy(),
    "Gas mass per temperature [msol]": blankList.copy(),
    "Gas n_H density per temperature [cm-3]": blankList.copy(),
    "Total gas mass (all haloes) available in spherical shell [msol]": blankList.copy(),
    "Total N_tracers (all haloes) in spherical shell": blankList.copy(),
    "Total N_tracers (all haloes) within selection radii": blankList.copy(),
    "Total gas mass (all haloes) within selection radii [msol]": blankList.copy(),
}

TRACERSPARAMSCOMBI = TRACERSPARAMS

Rmin = min(TRACERSPARAMSCOMBI["Rinner"])
Rmax = max(TRACERSPARAMSCOMBI["Router"])

for snapNumber in snapRange:
    for halo, loadPath in zip(SELECTEDHALOES, HALOPATHS):
        haloPath = TRACERSPARAMSCOMBI["simfile"] + halo + "/output/"

        print("")
        print("")
        print(f"Starting {halo}")

        snap_subfind = load_subfind(snapNumber, dir=haloPath)

        snap = gadget_readsnap(
            snapNumber,
            haloPath,
            hdf5=True,
            loadonlytype=[0,4],
            lazy_load=True,
            subfind=snap_subfind,
        )

        snapTracers = gadget_readsnap(
            snapNumber, haloPath, hdf5=True, loadonlytype=[6], lazy_load=True
        )

        tmp = snap.data["id"]
        tmp = snap.data["hrgm"]
        tmp = snap.data["mass"]
        tmp = snap.data["pos"]
        tmp = snap.data["vol"]

        print(
            f"[@{int(snapNumber)}]: SnapShot {halo} loaded at RedShift z={snap.redshift:0.05e}"
        )

        snap.calc_sf_indizes(snap_subfind)
        snap.select_halo(snap_subfind, do_rotation=True)

        whereStarsGas = np.where(np.isin(snap.type, [0, 4]) == True)[0]
        whereDM = np.where(snap.type == 1)[0]
        whereGas = np.where(snap.type == 0)[0]
        whereStars = np.where(snap.type == 4)[0]

        NDM = len(whereDM)
        NGas = len(whereGas)
        NStars = len(whereStars)

        deleteKeys = []
        for key, value in snap.data.items():
            if value is not None:
                if np.shape(value)[0] == (NStars):
                    # print("Stars")
                    deleteKeys.append(key)
                elif np.shape(value)[0] == (NGas + NStars):
                    # print("Stars and Gas")
                    snap.data[key] = value.copy()[whereGas]
                    pass
                else:
                    pass
                # print(np.shape(snap.data[key]))

        for key in deleteKeys:
            del snap.data[key]

        # --------------------------#
        ##    Units Conversion    ##
        # --------------------------#

        # Convert Units
        ## Make this a seperate function at some point??
        snap.pos *= 1e3  # [kpc]
        snap.vol *= 1e9  # [kpc^3]
        snap.mass *= 1e10  # [Msol]
        snap.hrgm *= 1e10  # [Msol]

        # [Kpc]
        snap.data["R"] = np.linalg.norm(snap.data["pos"], axis=1)  # [Kpc]

        whereGas = np.where(snap.type == 0)[0]
        # Density is rho/ <rho> where <rho> is average baryonic density
        rhocrit = (
            3.0
            * (snap.omega0 * (1.0 + snap.redshift) ** 3 + snap.omegalambda)
            * (snap.hubbleparam * 100.0 * 1e5 / (c.parsec * 1e6)) ** 2
            / (8.0 * pi * c.G)
        )
        rhomean = (
            3.0
            * (snap.omega0 * (1.0 + snap.redshift) ** 3)
            * (snap.hubbleparam * 100.0 * 1e5 / (c.parsec * 1e6)) ** 2
            / (8.0 * pi * c.G)
        )

        # Mean weight [amu]
        meanweight = sum(snap.gmet[whereGas, 0:9], axis=1) / (
            sum(snap.gmet[whereGas, 0:9] / elements_mass[0:9], axis=1)
            + snap.ne[whereGas] * snap.gmet[whereGas, 0]
        )

        # 3./2. N KB
        Tfac = ((3.0 / 2.0) * c.KB) / (meanweight * c.amu)

        snap.data["dens"] = (
            (snap.rho[whereGas] / (c.parsec * 1e6) ** 3) * c.msol * 1e10
        )  # [g cm^-3]
        gasX = snap.gmet[whereGas, 0]

        snap.data["n_H"] = snap.data["dens"][whereGas] / c.amu * gasX  # cm^-3

        # Temperature = U / (3/2 * N KB) [K]
        snap.data["T"] = (snap.u[whereGas] * 1e10) / (Tfac)  # K

        ###-------------------------------------------
        #   Find total number of tracers and gas mass in halo within rVir
        ###-------------------------------------------

        Cond = np.where(
            (snap.data["R"] <= Rmax)
            & (snap.data["R"] >= Rmin)
            & (snap.data["sfr"] <= 0.0)
            & (np.isin(snap.data["subhalo"], np.array([-1.0, 0.0])))
        )[0]

        CellIDs = snap.id[Cond]

        # Select Parent IDs in Cond list
        #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
        ParentsIndices = np.where(np.isin(snapTracers.prid, CellIDs))

        # Select Tracers and Parent IDs from cells that meet condition and contain tracers
        Tracers = snapTracers.trid[ParentsIndices]
        Parents = snapTracers.prid[ParentsIndices]

        # Get Gas meeting Cond AND with tracers
        CellsIndices = np.where(np.isin(snap.id, Parents))
        CellIDs = snap.id[CellsIndices]

        NtracersTotalR = np.shape(Tracers)[0]

        print(
            f"For {halo} at snap {snapNumber} Total N_tracers (all haloes) within selection radii = ",
            NtracersTotalR,
        )

        dictRowSelectSnaponly = np.where((summaryDict["Snap Number"] == snapNumber))[0]

        summaryDict["Total N_tracers (all haloes) within selection radii"][
            dictRowSelectSnaponly
        ] += NtracersTotalR

        massTotalR = np.sum(snap.data["mass"][Cond])
        print(f"For {halo} at snap {snapNumber} total mass [msol] = ", massTotalR)
        summaryDict["Total gas mass (all haloes) within selection radii [msol]"][
            dictRowSelectSnaponly
        ] += massTotalR
        ###-------------------------------------------
        #   Load in analysed data
        ###-------------------------------------------
        loadPath += "/"
        TRACERSPARAMS, DataSavepath, _ = load_tracers_parameters(
            loadPath + TracersParamsPath
        )
        saveParams += TRACERSPARAMS["saveParams"]
        dataDict = {}
        print("")
        print(f"Loading {halo} Analysed Data!")

        print("LOAD")
        dataDict = {}
        for T in Tlst:
            for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
                loadPath = (
                    DataSavepath
                    + f"_T{T}_{rin}R{rout}_flat-wrt-time.h5"
                )
                key = (f"T{T}", f"{rin}R{rout}")
                try:
                    tmp = hdf5_load(loadPath)
                    dataDict.update(tmp)
                except Exception as e:
                    print(f"[Multi Halo Merge Time]: WARNING! {e}")
                    pass


        print("LOADED")
        
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            print(f"{rin}R{rout}")

            whereGas = np.where(snap.type == 0)[0]

            Cond = np.where(
                (snap.data["R"][whereGas] >= rin)
                & (snap.data["R"][whereGas] <= rout)
                & (snap.data["sfr"][whereGas] <= 0)
                & (np.isin(snap.data["subhalo"], np.array([-1.0, 0.0])))
            )[0]

            massR = np.sum(snap.data["mass"][Cond])
            dictRowSelectRonly = np.where(
                (summaryDict["R_inner [kpc]"] == rin)
                & (summaryDict["R_outer [kpc]"] == rout)
                & (summaryDict["Snap Number"] == snapNumber)
            )[0]

            print(f"Total mass (all haloes) in spherical shell [msol] = ", massR)

            summaryDict[
                "Total gas mass (all haloes) available in spherical shell [msol]"
            ][dictRowSelectRonly] += massR

            # ==================================================================#
            # Select Cell IDs for cells which meet condition
            CellIDs = snap.id[Cond]

            # Select Parent IDs in Cond list
            #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
            ParentsIndices = np.where(np.isin(snapTracers.prid, CellIDs))

            # Select Tracers and Parent IDs from cells that meet condition and contain tracers
            Tracers = snapTracers.trid[ParentsIndices]
            Parents = snapTracers.prid[ParentsIndices]

            # Get Gas meeting Cond AND with tracers
            CellsIndices = np.where(np.isin(snap.id, Parents))
            CellIDs = snap.id[CellsIndices]

            NtracersR = np.shape(Tracers)[0]

            print(f"Total N_tracers (all haloes) in spherical shell= ", NtracersR)
            summaryDict["Total N_tracers (all haloes) in spherical shell"][
                dictRowSelectRonly
            ] += NtracersR

            for (ii, T) in enumerate(Tlst):
                whereSnap = np.where(np.array(snapRange) == int(snapNumber))[0]
                FullDictKey = (f"T{float(T)}", f"{rin}R{rout}")
                print(FullDictKey)
                Ntracersselected = np.shape(dataDict[FullDictKey]["type"])[1]
                print(
                    f"Total N_tracers (all haloes) in spherical shell = ",
                    Ntracersselected,
                )
                massselected = np.sum(
                    dataDict[FullDictKey]["mass"][whereSnap,:][
                        np.where(dataDict[FullDictKey]["type"][whereSnap,:] == 0)[0]
                    ]
                )
                print(
                    f"Total mass (all haloes) in spherical shell [msol] = ",
                    massselected,
                )

                dictRowSelect = np.where(
                    (summaryDict["R_inner [kpc]"] == rin)
                    & (summaryDict["R_outer [kpc]"] == rout)
                    & (summaryDict["Log10(T) [K]"] == float(T))
                    & (summaryDict["Snap Number"] == snapNumber)
                )[0]

                summaryDict["N_tracers selected"][dictRowSelect] += Ntracersselected
                summaryDict["Gas mass selected [msol]"][dictRowSelect] += massselected

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
                    & (np.isin(snap.data["subhalo"], np.array([-1.0, 0.0])))
                )[0]

                massRT = np.sum(snap.data["mass"][Cond])
                summaryDict["Gas mass per temperature [msol]"][dictRowSelect] += massRT
                print(
                    f"Total mass (all haloes) in spherical shell per temperature [msol] = ",
                    FullDictKey,
                    massRT,
                )

                CellIDs = snap.id[Cond]

                # Select Parent IDs in Cond list
                #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
                ParentsIndices = np.where(np.isin(snapTracers.prid, CellIDs))

                # Select Tracers and Parent IDs from cells that meet condition and contain tracers
                Tracers = snapTracers.trid[ParentsIndices]
                Parents = snapTracers.prid[ParentsIndices]

                # Get Gas meeting Cond AND with tracers
                CellsIndices = np.where(np.isin(snap.id, Parents))
                CellIDs = snap.id[CellsIndices]

                nTracersRT = np.shape(Tracers)[0]
                summaryDict["N_tracers per temperature"][dictRowSelect] += nTracersRT
                print(
                    f"Total N_tracers (all haloes) per temperature = ",
                    FullDictKey,
                    nTracersRT,
                )

                n_H_RT = np.median(snap.data["n_H"][Cond])
                summaryDict["Gas n_H density per temperature [cm-3]"][
                    dictRowSelect
                ] += n_H_RT
    # print("summaryDict = ", summaryDict)
#

nSnaps = float(len(snapRange))
for key, value in summaryDict.items():
    if key not in ["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]"]:
        summaryDict[key] = value / nSnaps


df = pd.DataFrame(summaryDict, index=[ii for ii in range(len(blankList))])

df = df.groupby(["R_inner [kpc]", "R_outer [kpc]", "Log10(T) [K]"]).sum()

df["%Available tracers in spherical shell selected"] = (
    df["N_tracers selected"].astype("float64")
    / df["Total N_tracers (all haloes) in spherical shell"].astype("float64")
) * 100.0

df["%Available mass of spherical shell selected"] = (
    df["Gas mass selected [msol]"].astype("float64")
    / df["Total gas mass (all haloes) available in spherical shell [msol]"].astype(
        "float64"
    )
) * 100.0

nHaloes = float(len(SELECTEDHALOES))
df["Number of Haloes"] = nHaloes

df["Average N_tracers selected (per halo)"] = (
    df["N_tracers selected"].astype("float64") / nHaloes
)
df["Average N_tracers per temperature (per halo)"] = (
    df["N_tracers per temperature"].astype("float64") / nHaloes
)
df["Average gas mass selected (per halo) [msol]"] = (
    df["Gas mass selected [msol]"].astype("float64") / nHaloes
)
df["Average gas mass available (per halo) [msol]"] = (
    df["Gas mass per temperature [msol]"].astype("float64") / nHaloes
)

# summaryDict = {'R_inner':
# 'R_outer':
# "Log10(T) [K]"
# 'N_tracers selected'
# 'Gas mass selected [msol]'
# 'Total gas mass (all haloes) available in spherical shell [msol]'
# 'Total N_tracers (all haloes) in spherical shell'
# 'Total N_tracers (all haloes) within selection radii'
# 'Total gas mass (all haloes) within selection radii [msol]'}

df["Average gas mass (per halo) available in spherical shell [msol]"] = (
    df["Total gas mass (all haloes) available in spherical shell [msol]"].astype(
        "float64"
    )
    / nHaloes
)

df["Average N_tracers (per halo) in spherical shell"] = (
    df["Total N_tracers (all haloes) in spherical shell"].astype("float64") / nHaloes
)

df["Average N_tracers (per halo) within selection radii"] = (
    df["Total N_tracers (all haloes) within selection radii"].astype("float64")
    / nHaloes
)

df["Average gas mass (per halo) within selection radii [msol]"] = (
    df["Total gas mass (all haloes) within selection radii [msol]"].astype("float64")
    / nHaloes
)

print(df.head(n=20))
if timeAverageBool is True:
    df.to_csv(
        "Data_Tracers_MultiHalo_Mass-Ntracers-Summary_Time-Average.csv", index=False
    )
else:
    df.to_csv("Data_Tracers_MultiHalo_Mass-Ntracers-Summary_flat-wrt-time-version.csv", index=False)
