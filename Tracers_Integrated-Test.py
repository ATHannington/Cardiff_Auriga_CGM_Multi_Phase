"""
Author: A. T. Hannington
Created: 19/03/2020
Known Bugs:
    None
"""
import numpy as np
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import gadget_readsnap
from gadget_subfind import load_subfind, sum
from Tracers_Subroutines import *
from random import sample
import pytest

SUBSET = None
individual_tracerSubset = 500

# Parameters where shape should be (1,)
singleValueParams = ["Lookback", "Ntracers", "Snap"]

# Params where shape should be >= shape('id')
exceptionsParams = ["trid", "prid"]

# ==============================================================================#
#       USER DEFINED PARAMETERS
# ==============================================================================#
# Input parameters path:
TracersParamsPath = "TracersParams_Test.csv"

# File types for data save.
#   Mini: small median and percentiles data
#   Full: full FullDict data
MiniDataPathSuffix = f".h5"
FullDataPathSuffix = f".h5"

# Lazy Load switch. Set to False to save all data (warning, pickle file may explode)
lazyLoadBool = True

# Number of cores to run on:
n_processes = 2

# ==============================================================================#
#       Prepare for analysis
# ==============================================================================#
# Load in parameters from csv. This ensures reproducability!
#   We save as a DataFrame, then convert to a dictionary, and take out nesting...
# Save as .csv
TRACERSPARAMS, DataSavepath, Tlst = load_tracers_parameters(TracersParamsPath)

print("")
print("Loaded Analysis Parameters:")
for key, value in TRACERSPARAMS.items():
    print(f"{key}: {value}")

print("")

# Save types, which when combined with saveparams define what data is saved.
#   This is intended to be the string equivalent of the percentiles.
saveTypes = ["_" + str(percentile) + "%" for percentile in TRACERSPARAMS["percentiles"]]

# Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = TRACERSPARAMS[
    "saveParams"
]  # ['rho_rhomean','dens','T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','csound','tcross','tff','tcool_tff']

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
]  # ['halo','subhalo','Lookback','Ntracers','Snap','id','prid','trid','type','mass','pos']

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
def err_catcher(arg):
    raise Exception(f"Child Process died and gave error: {arg}")
    return


# ==============================================================================#
#       MAIN PROGRAM
# ==============================================================================#
# ==============================================================================#
#                                                                              #
#                       PREPARE SAMPLE DATA                                    #
#                                                                              #
# ==============================================================================#
FullDict = {}

if len(TRACERSPARAMS["targetTLst"]) > 1:
    print(
        f"[@TRACERSPARAMS :] len(targetTLst) > 1 ! Only 1 first temperature is utilised."
    )

targetT = TRACERSPARAMS["targetTLst"][0]

(
    TracersTFC,
    CellsTFC,
    CellIDsTFC,
    ParentsTFC,
    snapGas,
    snapTracers,
    rotation_matrix
) = tracer_selection_snap_analysis(
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
    lazyLoadBool=lazyLoadBool,
    SUBSET=None,
    snapNumber=TRACERSPARAMS["selectSnap"],
    saveTracers=False,
    loadonlyhalo=True,
)

TFCkey0 = list(TracersTFC.keys())[0]
CellsTFC, CellIDsTFC, ParentsTFC = (
    CellsTFC[TFCkey0],
    CellIDsTFC[TFCkey0],
    ParentsTFC[TFCkey0],
)

output_dict = snap_analysis(
    TRACERSPARAMS["snapMin"],
    TRACERSPARAMS,
    HaloID,
    TracersTFC,
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
    rotation_matrix,
    lazyLoadBool,
)

out, TracersCFT, CellsCFT, CellIDsCFT, ParentsCFT = (
    output_dict["out"],
    output_dict["TracersCFT"],
    output_dict["CellsCFT"],
    output_dict["CellIDsCFT"],
    output_dict["ParentsCFT"],
)

TracersTFC = TracersTFC[TFCkey0]

TracersCFTinit, CellsCFTinit, CellIDsCFTinit, ParentsCFTinit = get_cells_from_tracers(
    snapGas,
    snapTracers,
    TracersTFC,
    saveParams,
    saveTracersOnly,
    snapNumber=TRACERSPARAMS["selectSnap"],
)

CFTkey0 = list(TracersCFT.keys())[0]
out, TracersCFT, CellsCFT, CellIDsCFT, ParentsCFT = (
    out[CFTkey0],
    TracersCFT[CFTkey0],
    CellsCFT[CFTkey0],
    CellIDsCFT[CFTkey0],
    ParentsCFT[CFTkey0],
)

FullDict.update(out)


# ==============================================================================#
#
#                                  TESTING
#
# ==============================================================================#


def test_same_snap_cell_IDs():
    """
    Test that the Cell IDs selected from tracers match the CellIDs containing tracers when tracers are selected.
    """
    CellIDMatch = np.all(np.isin(CellIDsTFC, CellIDsCFTinit))

    assert (
        CellIDMatch == True
    ), "[@CellIDMatch:] Cell IDs not equal! TFC and CFT! Check tracer selections!"


def test_same_snap_cell_data():
    """
    Check all values of data from Cells selected from tracers matches data selected from selecting the tracers.
    """
    truthyList = []
    for ((k1, v1), (k2, v2)) in zip(CellsCFTinit.items(), CellsTFC.items()):
        # Do all entries if CellsCFT values and CellsTFC values match?
        v1_notnan = np.where(np.isnan(v1) == False)
        v2_notnan = np.where(np.isnan(v2) == False)
        truthyList.append(np.all(np.isin(v1[v1_notnan], v2[v2_notnan])))

    # Do the entries match for all keys?
    truthy = np.all(truthyList)

    assert (
        truthy == True
    ), "[@Cells data:] Cell data not equal from TFC and CFT! Check tracer selections!"


def test_same_snap_tracers_parents():
    """
    Test whether Parents and Tracers returned from CFT match those from TFC when applied to same snapshot data.
    """
    truthy = np.isin(TracersCFTinit, TracersTFC)
    assert (
        np.all(truthy) == True
    ), "[@Same Snap Tracers Parents:] Not all Tracers CFT init found in TracersTFC!"

    truthy = np.isin(ParentsCFTinit, ParentsTFC)
    assert (
        np.all(truthy) == True
    ), "[@Same Snap Tracers Parents:] Not all Parents CFT init found in ParentsTFC!"


def test_parents_match_tracers():
    """
    Test that there are the same number of prids (parent ids) as trids (tracer ids).
    """
    assert np.shape(ParentsTFC) == np.shape(
        TracersTFC
    ), "[@Parents Match Tracers:] ParentsTFC different shape to TracersTFC!"
    assert np.shape(ParentsCFTinit) == np.shape(
        TracersCFTinit
    ), "[@Parents Match Tracers:] ParentsCFT init different shape to TracersCFT init!"
    assert np.shape(ParentsCFT) == np.shape(
        TracersCFT
    ), "[@Parents Match Tracers:] ParentsCFT different shape to TracersCFT!"


def test_dwindling_parents_and_tracers():
    """
    Test that we are losing or maintaining trid and prid number, but not increasing. Also test that all TracersCFT are a subset of Tracers TFC.
    We should be finding that this subset is the same size or smaller, but never bigger or including a new value.
    """
    assert (
        np.shape(TracersCFT)[0] <= np.shape(TracersTFC)[0]
    ), "[@Dwindling Parents and Tracers:] TracersCFT not <= in shape than TracersTFC!"
    assert (
        np.shape(ParentsCFT)[0] <= np.shape(ParentsTFC)[0]
    ), "[@Dwindling Parents and Tracers:] ParentsCFT not <= in shape than ParentsTFC!"
    assert (
        np.all(np.isin(TracersCFT, TracersTFC)) == True
    ), "[@Dwindling Parents and Tracers:] TracersCFT not a subset of TracersTFC!"


def test_parents_in_cell_IDs():
    """
    Test that all Parent IDs, prids, are contained in the CellIDs data. This should be a many-to-one super-set.
    i.e. there may be duplicate CellIDs in Parents but every Parent should match at least one Cell ID.
    """
    truthy = np.all(np.isin(ParentsTFC, CellIDsTFC))
    assert (
        truthy == True
    ), "[@Parents in Cell IDs:] ParentsTFC not many-to-one super-set of CellIDsTFC!"

    truthy = np.all(np.isin(ParentsCFTinit, CellIDsCFTinit))
    assert (
        truthy == True
    ), "[@Parents in Cell IDs:] ParentsCFT init not many-to-one super-set of CellIDsCFT init!"

    truthy = np.all(np.isin(ParentsCFT, CellIDsCFT))
    assert (
        truthy == True
    ), "[@Parents in Cell IDs:] ParentsCFT not many-to-one super-set of CellIDsCFT!"


def test_cell_shapes():
    """
    Test that Cells Data has consistent shape with the number of Cell IDs. This ensures all data has been correctly selected.
    """
    truthyList = []
    for key, values in CellsTFC.items():
        if key in singleValueParams:
            truthyList.append(np.shape(values)[0] == 1)
        elif key in exceptionsParams:
            truthyList.append(np.shape(values)[0] >= np.shape(CellIDsTFC)[0])
        else:
            truthyList.append(np.shape(values)[0] == np.shape(CellIDsTFC)[0])

    truthy = np.all(truthyList)
    assert (
        truthy == True
    ), "[@Cells Shapes:] values of Cells TFC not consistent shape to CellIDsTFC! Some data may be missing!"

    truthyList = []
    for key, values in CellsCFTinit.items():
        if key in singleValueParams:
            truthyList.append(np.shape(values)[0] == 1)
        elif key in exceptionsParams:
            truthyList.append(np.shape(values)[0] >= np.shape(CellIDsCFTinit)[0])
        else:
            truthyList.append(np.shape(values)[0] == np.shape(CellIDsCFTinit)[0])

    truthy = np.all(truthyList)
    assert (
        truthy == True
    ), "[@Cells Shapes:] values of Cells CFT init not consistent shape to CellIDsCFTinit! Some data may be missing!"

    truthyList = []
    for key, values in CellsCFT.items():
        if key in singleValueParams:
            truthyList.append(np.shape(values)[0] == 1)
        elif key in exceptionsParams:
            truthyList.append(np.shape(values)[0] >= np.shape(CellIDsCFT)[0])
        else:
            truthyList.append(np.shape(values)[0] == np.shape(CellIDsCFT)[0])

    truthy = np.all(truthyList)
    assert (
        truthy == True
    ), "[@Cells Shapes:] values of Cells CFT not consistent shape to CellIDsCFT! Some data may be missing!"


def test_individual_tracer_fake_data():
    startTrid = 200
    lenTrid = 10
    trid = np.arange(start=startTrid, stop=startTrid + lenTrid, step=1)

    startID = 0
    lenID = 5
    id = np.arange(start=startID, stop=startID + lenID, step=1)

    lenPrid = lenTrid
    prid = np.array([])
    while len(prid) <= lenPrid:
        prid = np.append(prid, id)
        prid.flatten()
        if len(prid) >= lenPrid:
            break
    prid = np.array(list(map(int, prid)))

    # Ignore mass, but give value
    mass1 = id

    tempData = {"T": np.array([float(ii) * 0.1 for ii in id])}

    subset = 5

    rangeMin = 0
    rangeMax = len(trid)
    TracerNumberSelect = np.arange(start=rangeMin, stop=rangeMax, step=1)
    randomSample = sample(TracerNumberSelect.tolist(), min(subset, rangeMax))

    assert len(randomSample) == min(
        subset, rangeMax
    ), "[@individual_tracerFakeData:], Random Sample not correct shape!"
    assert (
        np.all(np.isin(randomSample, TracerNumberSelect)) == True
    ), "[@individual_tracerFakeData:], Random Sample comtains non-TracerNumberSelect entries!"

    """
        Full set of Tracers Tests!
    """
    SelectedTracers1 = trid[TracerNumberSelect]
    SelectedParents1 = prid[TracerNumberSelect]
    assert (
        len(SelectedTracers1) == rangeMax
    ), "[@individual_tracerFakeData Full Set:], SelectedTracers1 not correct shape!"
    assert (
        np.all(np.isin(SelectedTracers1, trid)) == True
    ), "[@individual_tracerFakeData Full Set:], SelectedTracers1 contains non-trid entries!"

    (
        data,
        TracersReturned,
        ParentsReturned,
    ) = get_individual_cell_from_tracer_all_param_v2(
        Tracers=trid,
        Parents=prid,
        CellIDs=id,
        SelectedTracers=SelectedTracers1,
        Data=tempData,
    )

    expectedData = np.array([tempData["T"], tempData["T"]]).flatten()
    assert (
        np.all(data["T"] == expectedData) == True
    ), "[@individual_tracerFakeData Full Set:] dataReturned != expectedData ! Some data is false or re-ordering has occurred!"
    assert (
        np.shape(data["T"])[0] == rangeMax
    ), "[@individual_tracerFakeData Full Set:] returned data not size == rangeMax! Some data/NaNs may be missing!"
    assert (
        np.all(TracersReturned == SelectedTracers1) == True
    ), "[@individual_tracerFakeData Full Set:] Tracers Returned is not equal to Selected Tracers! Some Tracers Returned have been mis-selected!"
    assert (
        np.shape(TracersReturned)[0] <= rangeMax
    ), "[@individual_tracerFakeData Full Set:] Tracers Returned is not of size <= rangeMax! There may be too many Returned Tracers!"
    assert (
        np.all(np.isin(data["T"], tempData["T"])) == True
    ), "[@individual_tracerFakeData Full Set:] dataReturned not a subset of fakeData ! Some data is false or re-ordering has occurred!"
    assert (
        np.all(np.isin(TracersReturned, trid)) == True
    ), "[@individual_tracerFakeData Full Set:] Tracers Returned not subset of trid! Selection error!"
    assert (
        np.all(np.isin(ParentsReturned, prid)) == True
    ), "[@individual_tracerFakeData Full Set:] Tracers Returned not subset of trid! Selection error!"
    assert (
        np.all(TracersReturned == SelectedTracers1) == True
    ), "[@individual_tracerFakeData Full Set:] Trid test : ordered trids not equal to SelectedTracers! Ordering failure!"
    assert (
        np.all(ParentsReturned == SelectedParents1) == True
    ), "[@individual_tracerFakeData Full Set:] Prid test : ordered prids not equal to SelectedParents! Ordering failure!"

    truthyList = []
    for ind, value in enumerate(data["T"]):
        truthyList.append(np.isin(value, tempData["T"]))

    truthy = np.all(truthyList)

    assert (
        truthy == True
    ), "[@individual_tracerFakeData Full Set:] Data has incorrect values! Selection error!"

    """
        Subset of Tracers Selected Tests!
    """
    SelectedTracers1 = trid[randomSample]
    SelectedParents1 = prid[randomSample]

    assert (
        len(SelectedTracers1) == subset
    ), "[@individual_tracerFakeData Random Subset of Tracers:], SelectedTracers1 not correct shape!"
    assert (
        np.all(np.isin(SelectedTracers1, trid)) == True
    ), "[@individual_tracerFakeData Random Subset of Tracers:], SelectedTracers1 contains non-trid entries!"

    (
        data,
        TracersReturned,
        ParentsReturned,
    ) = get_individual_cell_from_tracer_all_param_v2(
        Tracers=trid,
        Parents=prid,
        CellIDs=id,
        SelectedTracers=SelectedTracers1,
        Data=tempData,
    )

    whereTracersReturnedNotNaN = np.where(np.isnan(TracersReturned) == False)
    whereParentsReturnedNotNaN = np.where(ParentsReturned != -1)

    assert (
        np.shape(data["T"])[0] == subset
    ), "[@individual_tracerFakeData Random Subset of Tracers:] returned data not size == subset! Some data/NaNs may be missing!"
    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], SelectedTracers1))
        == True
    ), "[@individual_tracerFakeData Random Subset of Tracers:] Tracers Returned is not a subset of Selected Tracers! Some Tracers Returned have been mis-selected!"
    assert (
        np.shape(
            TracersReturned[whereTracersReturnedNotNaN],
        )[0]
        <= subset
    ), "[@individual_tracerFakeData Random Subset of Tracers:] Tracers Returned is not of size <= subset! There may be too many Returned Tracers!"

    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], trid)) == True
    ), "[@individual_tracerFakeData Random Subset of Tracers:] Tracers Returned not subset of trid! Selection error!"
    assert (
        np.all(np.isin(ParentsReturned[whereParentsReturnedNotNaN], prid)) == True
    ), "[@individual_tracerFakeData Random Subset of Tracers:] Tracers Returned not subset of trid! Selection error!"
    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], SelectedTracers1))
        == True
    ), "[@individual_tracerFakeData Random Subset of Tracers:] Trid test : ordered trids not equal to SelectedTracers! Ordering failure!"
    assert (
        np.all(np.isin(ParentsReturned[whereParentsReturnedNotNaN], SelectedParents1))
        == True
        == True
    ), "[@individual_tracerFakeData Random Subset of Tracers:] Prid test : ordered prids not equal to SelectedParents! Ordering failure!"

    truthyList = []
    for ind, value in enumerate(data["T"]):
        if np.isnan(value) == False:
            truthyList.append(np.isin(value, tempData["T"]))
        else:
            truthyList.append(np.isnan(value))

    truthy = np.all(truthyList)

    assert (
        truthy == True
    ), "[@individual_tracerFakeData Random Subset of Tracers:] Data has incorrect values! Selection error!"

    """
        Subset of Tracers Selected are present in Tracers, tests!
    """

    SelectedTracers1 = trid[randomSample]
    SelectedParents1 = prid[randomSample]

    startTrid2 = startTrid - int(float(lenTrid) / 2.0)
    lenTrid2 = 10
    trid2 = np.arange(start=startTrid2, stop=startTrid2 + lenTrid2, step=1)

    assert (
        len(SelectedTracers1) == subset
    ), "[@individual_tracerFakeData Subset of Selected Tracers Present in Tracers:], SelectedTracers1 not correct shape!"
    assert (
        np.all(np.isin(SelectedTracers1, trid)) == True
    ), "[@individual_tracerFakeData Subset of Selected Tracers Present in Tracers:], SelectedTracers1 contains non-trid2 entries!"

    (
        data,
        TracersReturned,
        ParentsReturned,
    ) = get_individual_cell_from_tracer_all_param_v2(
        Tracers=trid2,
        Parents=prid,
        CellIDs=id,
        SelectedTracers=SelectedTracers1,
        Data=tempData,
    )

    whereTracersReturnedNotNaN = np.where(np.isnan(TracersReturned) == False)
    whereParentsReturnedNotNaN = np.where(ParentsReturned != -1)

    assert (
        np.shape(data["T"])[0] == subset
    ), "[@individual_tracerFakeData Subset of Selected Tracers Present in Tracers:] returned data not size == subset! Some data/NaNs may be missing!"
    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], SelectedTracers1))
        == True
    ), "[@individual_tracerFakeData Subset of Selected Tracers Present in Tracers:] Tracers Returned is not a subset of Selected Tracers! Some Tracers Returned have been mis-selected!"
    assert (
        np.shape(TracersReturned)[0] <= subset
    ), "[@individual_tracerFakeData Subset of Selected Tracers Present in Tracers:] Tracers Returned is not of size <= subset! There may be too many Returned Tracers!"

    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], trid2)) == True
    ), "[@individual_tracerFakeData Subset of Selected Tracers Present in Tracers:] Tracers Returned not subset of trid2! Selection error!"

    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], SelectedTracers1))
        == True
    ), "[@individual_tracerFakeData Random Subset of Tracers:] Trid test : ordered trids not equal to SelectedTracers! Ordering failure!"
    assert (
        np.all(np.isin(ParentsReturned[whereParentsReturnedNotNaN], SelectedParents1))
        == True
    ), "[@individual_tracerFakeData Random Subset of Tracers:] Prid test : ordered prids not equal to SelectedParents! Ordering failure!"

    truthyList = []
    for ind, value in enumerate(data["T"]):
        if np.isnan(value) == False:
            truthyList.append(np.isin(value, tempData["T"]))
        else:
            truthyList.append(np.isnan(value))

    truthy = np.all(truthyList)

    assert (
        truthy == True
    ), "[@individual_tracerFakeData Subset of Selected Tracers Present in Tracers:] Data has incorrect values! Selection error!"


def test_individual_tracer():
    """
    Test that the returned tracers from get_individual_cell_from_tracer are a subset of the SelectedTracers. Also that the data
    returned is of shape == subset. There are NaNs where the SelectedTracer is no longer present in the data.
    """

    rangeMin = 0
    rangeMax = len(snapGas.data["T"])
    tempData = {"T": snapGas.data["T"]}
    TracerNumberSelect = np.arange(start=rangeMin, stop=rangeMax, step=1)
    TracerNumberSelect = sample(
        TracerNumberSelect.tolist(), min(individual_tracerSubset, rangeMax)
    )

    SelectedTracers1 = snapTracers.data["trid"][TracerNumberSelect]
    SelectedParents1 = snapTracers.data["prid"][TracerNumberSelect]

    (
        data,
        TracersReturned,
        ParentsReturned,
    ) = get_individual_cell_from_tracer_all_param_v2(
        Tracers=snapTracers.data["trid"],
        Parents=snapTracers.data["prid"],
        CellIDs=snapGas.data["id"],
        SelectedTracers=SelectedTracers1,
        Data=tempData,
    )

    # Select indices (positions in array) of Tracer IDs which are in the Tracers list
    TracersIndices = np.where(np.isin(snapTracers.trid, SelectedTracers1))

    # Select the matching parent cell IDs for tracers which are in Tracers list
    Parents = snapTracers.prid[TracersIndices]

    # Select Tracers which are in the original tracers list (thus their original cells met condition and contained tracers)
    TracersCFT = snapTracers.trid[TracersIndices]

    # Select Cell IDs which are in Parents
    #   NOTE:   This selection causes trouble. Selecting only Halo=HaloID means some Parents now aren't associated with Halo
    #           This means some parents and tracers need to be dropped as they are no longer in desired halo.
    # CellsIndices = np.where(np.isin(snapGas.id,Parents))
    # CellIDs = snapGas.id[CellsIndices]

    # So, from above issue: Select Parents and Tracers which are associated with Desired Halo ONLY!
    ParentsIndices = np.where(np.isin(Parents, snapGas.id))
    Parents = Parents[ParentsIndices]
    TracersCFT = TracersCFT[ParentsIndices]

    # Select IDs for Cells with Tracers with no duplicates
    CellIndicesShort = np.where(np.isin(snapGas.id, Parents))[0]
    CellIDs = snapGas.id[CellIndicesShort]
    expectedData = tempData["T"][CellIndicesShort]

    whereDataNotNaN = np.where(np.isnan(data["T"]) == False)
    whereTracersReturnedNotNaN = np.where(np.isnan(TracersReturned) == False)
    whereParentsReturnedNotNaN = np.where(ParentsReturned != -1)

    assert (
        np.all(np.isin(data["T"][whereDataNotNaN], expectedData)) == True
    ), "[@Individual Tracer:] returned data not all contained in expectedData! Some data/NaNs may be missing!"

    assert (
        np.shape(data["T"])[0] == individual_tracerSubset
    ), "[@Individual Tracer:] returned data not size == individual_tracerSubset! Some data/NaNs may be missing!"

    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], SelectedTracers1))
        == True
    ), "[@Individual Tracer:] Tracers Returned is not a individual_tracerSubset of Selected Tracers! Some Tracers Returned have been mis-selected!"
    assert (
        np.shape(TracersReturned)[0] <= individual_tracerSubset
    ), "[@Individual Tracer:] Tracers Returned is not of size <= individual_tracerSubset! There may be too many Returned Tracers!"
    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], SelectedTracers1))
        == True
    ), "[@Individual Tracer:] Trid test : ordered trids not equal to SelectedTracers! Ordering failure!"
    assert (
        np.all(np.isin(ParentsReturned[whereParentsReturnedNotNaN], SelectedParents1))
        == True
    ), "[@Individual Tracer:] Prid test : ordered prids not equal to SelectedParents! Ordering failure!"

    tempData = {"pos": snapGas.data["pos"]}

    (
        data,
        TracersReturned,
        ParentsReturned,
    ) = get_individual_cell_from_tracer_all_param_v2(
        Tracers=snapTracers.data["trid"],
        Parents=snapTracers.data["prid"],
        CellIDs=snapGas.data["id"],
        SelectedTracers=SelectedTracers1,
        Data=tempData,
    )

    assert (
        np.shape(data["pos"])[0] == individual_tracerSubset
    ), "[@Individual Tracer:] returned data not size == individual_tracerSubset! Some data/NaNs may be missing!"

    whereTracersReturnedNotNaN = np.where(np.isnan(TracersReturned) == False)
    whereParentsReturnedNotNaN = np.where(ParentsReturned != -1)
    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], SelectedTracers1))
        == True
    ), "[@Individual Tracer:] Tracers Returned is not a individual_tracerSubset of Selected Tracers! Some Tracers Returned have been mis-selected!"
    assert (
        np.shape(TracersReturned)[0] <= individual_tracerSubset
    ), "[@Individual Tracer:] Tracers Returned is not of size <= individual_tracerSubset! There may be too many Returned Tracers!"
    assert (
        np.all(np.isin(TracersReturned[whereTracersReturnedNotNaN], SelectedTracers1))
        == True
    ), "[@Individual Tracer:] Trid test : ordered trids not equal to SelectedTracers! Ordering failure!"
    assert (
        np.all(np.isin(ParentsReturned[whereParentsReturnedNotNaN], SelectedParents1))
        == True
    ), "[@Individual Tracer:] Prid test : ordered prids not equal to SelectedParents! Ordering failure!"
