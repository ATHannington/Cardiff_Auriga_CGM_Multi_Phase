"""
Author: A. T. Hannington
Created: 12/03/2020
Known Bugs:
    None
"""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import const as c
from gadget import *
from gadget_subfind import *
import h5py
import sys
import logging
import math
import random
from itertools import combinations, chain


# ==============================================================================#
#       MAIN ANALYSIS CODE - IN FUNC FOR MULTIPROCESSING
# ==============================================================================#
def snap_analysis(
        snapNumber,
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
        lazyLoadBool=True,
):
    print("")
    print(f"[@{int(snapNumber)}]: Starting Snap {snapNumber}")

    # load in the subfind group files
    snap_subfind = load_subfind(snapNumber, dir=TRACERSPARAMS["simfile"])

    # load in the gas particles mass and position only for HaloID 0.
    #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    snap = gadget_readsnap(
        snapNumber,
        TRACERSPARAMS["simfile"],
        hdf5=True,
        loadonlytype=[0, 1, 4],
        lazy_load=lazyLoadBool,
        subfind=snap_subfind,
    )
    snapGas = gadget_readsnap(
        snapNumber,
        TRACERSPARAMS["simfile"],
        hdf5=True,
        loadonlytype=[0, 4],
        lazy_load=lazyLoadBool,
        subfind=snap_subfind,
    )

    # load tracers data
    snapTracers = gadget_readsnap(
        snapNumber,
        TRACERSPARAMS["simfile"],
        hdf5=True,
        loadonlytype=[6],
        lazy_load=lazyLoadBool,
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

    tmp = snapGas.data["id"]
    tmp = snapGas.data["age"]
    tmp = snapGas.data["hrgm"]
    tmp = snapGas.data["mass"]
    tmp = snapGas.data["pos"]
    tmp = snapGas.data["vol"]
    del tmp

    print(f"[@{int(snapNumber)}]: SnapShot loaded at RedShift z={snap.redshift:0.05e}")

    # Centre the simulation on HaloID 0
    snap = set_centre(
        snap=snap, snap_subfind=snap_subfind, HaloID=HaloID, snapNumber=snapNumber
    )
    snapGas = set_centre(
        snap=snapGas, snap_subfind=snap_subfind, HaloID=HaloID, snapNumber=snapNumber
    )

    # --------------------------#
    ##    Units Conversion    ##
    # --------------------------#

    # Convert Units
    ## Make this a seperate function at some point??
    snap.pos *= 1e3  # [kpc]
    snap.vol *= 1e9  # [kpc^3]
    snap.mass *= 1e10  # [Msol]
    snap.hrgm *= 1e10  # [Msol]

    snapGas.pos *= 1e3  # [kpc]
    snapGas.vol *= 1e9  # [kpc^3]
    snapGas.mass *= 1e10  # [Msol]
    snapGas.hrgm *= 1e10  # [Msol]

    # Calculate New Parameters and Load into memory others we want to track
    snapGas = calculate_tracked_parameters(
        snapGas,
        snap,
        elements,
        elements_Z,
        elements_mass,
        elements_solar,
        Zsolar,
        omegabaryon0,
        snapNumber,
    )

    del snap

    # Pad stars and gas data with Nones so that all keys have values of same first dimension shape
    snapGas = pad_non_entries(snapGas, snapNumber)

    # Select only gas in High Res Zoom Region
    snapGas = high_res_only_gas_select(snapGas, snapNumber)

    # Find Halo=HaloID data for only selection snapshot. This ensures the
    # selected tracers are originally in the Halo, but allows for tracers
    # to leave (outflow) or move inwards (inflow) from Halo.

    # Assign SubHaloID and FoFHaloIDs
    snapGas = halo_id_finder(snapGas, snap_subfind, snapNumber)

    if snapNumber == int(TRACERSPARAMS["selectSnap"]):
        snapGas = halo_only_gas_select(snapGas, snap_subfind, HaloID, snapNumber)

    # Pad stars and gas data with Nones so that all keys have values of same first dimension shape
    snapGas = pad_non_entries(snapGas, snapNumber)
    ###
    ##  Selection   ##
    ###
    TracersCFTFinal = {}
    CellsCFTFinal = {}
    CellIDsCFTFinal = {}
    ParentsCFTFinal = {}
    for targetT in TRACERSPARAMS["targetTLst"]:
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            key = (f"T{targetT}", f"{rin}R{rout}")
            TracersSelect = TracersTFC[key]

            print(
                f"[@{snapNumber} @{rin}R{rout} @T{targetT}]:  Get Cells From Tracers!"
            )
            # Select Cells which have the tracers from the selection snap in them
            TracersCFT, CellsCFT, CellIDsCFT, ParentsCFT = get_cells_from_tracers(
                snapGas,
                snapTracers,
                TracersSelect,
                saveParams,
                saveTracersOnly,
                snapNumber,
            )

            # #Add snap data to temperature specific dictionary
            # print(f"Adding (T{targetT},{int(snap)}) to Dict")
            # FullDict.update({(f"T{targetT}",f"{int(snap)}"): CellsCFT})
            out = {(f"T{targetT}", f"{rin}R{rout}", f"{int(snapNumber)}"): CellsCFT}

            savePath = (
                    DataSavepath
                    + f"_T{targetT}_{rin}R{rout}_{int(snapNumber)}"
                    + FullDataPathSuffix
            )

            print(
                "\n"
                + f"[@{snapNumber} @{rin}R{rout} @T{targetT}]: Saving Tracers data as: "
                + savePath
            )

            hdf5_save(savePath, out)

            save_statistics(
                CellsCFT,
                targetT,
                rin,
                rout,
                snapNumber,
                TRACERSPARAMS,
                saveParams,
                DataSavepath,
                MiniDataPathSuffix,
            )

            if (
                    (TRACERSPARAMS["QuadPlotBool"] == True)
                    & (targetT == int(TRACERSPARAMS["targetTLst"][0]))
                    & (rin == TRACERSPARAMS["Rinner"][0])
            ):
                plot_projections(
                    snapGas,
                    snapNumber,
                    targetT,
                    rin,
                    rout,
                    TRACERSPARAMS,
                    DataSavepath,
                    FullDataPathSuffix,
                    Axes=TRACERSPARAMS["Axes"],
                    zAxis=TRACERSPARAMS["zAxis"],
                    boxsize=TRACERSPARAMS["boxsize"],
                    boxlos=TRACERSPARAMS["boxlos"],
                    pixres=TRACERSPARAMS["pixres"],
                    pixreslos=TRACERSPARAMS["pixreslos"],
                )

            TracersCFTFinal.update(
                {(f"T{targetT}", f"{rin}R{rout}", f"{int(snapNumber)}"): TracersCFT}
            )
            CellsCFTFinal.update(
                {(f"T{targetT}", f"{rin}R{rout}", f"{int(snapNumber)}"): CellsCFT}
            )
            CellIDsCFTFinal.update(
                {(f"T{targetT}", f"{rin}R{rout}", f"{int(snapNumber)}"): CellIDsCFT}
            )
            ParentsCFTFinal.update(
                {(f"T{targetT}", f"{rin}R{rout}", f"{int(snapNumber)}"): ParentsCFT}
            )

    return {
        "out": out,
        "TracersCFT": TracersCFTFinal,
        "CellsCFT": CellsCFTFinal,
        "CellIDsCFT": CellIDsCFTFinal,
        "ParentsCFT": ParentsCFTFinal,
    }


# ==============================================================================#
#       PRE-MAIN ANALYSIS CODE
# ==============================================================================#
def tracer_selection_snap_analysis(
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
        lazyLoadBool=True,
        SUBSET=None,
        snapNumber=None,
        saveTracers=True,
        TFCbool=True,
        loadonlyhalo=True,
):
    print("")
    print("***")
    print(f"From {TRACERSPARAMS['simfile']} :")
    print(f"Tracer Selection")
    print(f"     log10T={TRACERSPARAMS['targetTLst']}")
    print(f"     {TRACERSPARAMS['Rinner']}R{TRACERSPARAMS['Router']}")
    print("***")

    if snapNumber is None:
        snapNumber = TRACERSPARAMS["selectSnap"]

    # load in the subfind group files
    snap_subfind = load_subfind(snapNumber, dir=TRACERSPARAMS["simfile"])

    # load in the gas particles mass and position only for HaloID 0.
    #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
    snap = gadget_readsnap(
        snapNumber,
        TRACERSPARAMS["simfile"],
        hdf5=True,
        loadonlytype=[0, 1, 4],
        lazy_load=lazyLoadBool,
        subfind=snap_subfind,
    )
    snapGas = gadget_readsnap(
        snapNumber,
        TRACERSPARAMS["simfile"],
        hdf5=True,
        loadonlytype=[0, 4],
        lazy_load=lazyLoadBool,
        subfind=snap_subfind,
    )

    # load tracers data
    snapTracers = gadget_readsnap(
        snapNumber,
        TRACERSPARAMS["simfile"],
        hdf5=True,
        loadonlytype=[6],
        lazy_load=lazyLoadBool,
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

    tmp = snapGas.data["id"]
    tmp = snapGas.data["age"]
    tmp = snapGas.data["hrgm"]
    tmp = snapGas.data["mass"]
    tmp = snapGas.data["pos"]
    tmp = snapGas.data["vol"]
    del tmp

    print(f"[@{int(snapNumber)}]: SnapShot loaded at RedShift z={snap.redshift:0.05e}")

    # Centre the simulation on HaloID 0
    snap = set_centre(
        snap=snap, snap_subfind=snap_subfind, HaloID=HaloID, snapNumber=snapNumber
    )
    snapGas = set_centre(
        snap=snapGas, snap_subfind=snap_subfind, HaloID=HaloID, snapNumber=snapNumber
    )

    # --------------------------#
    ##    Units Conversion    ##
    # --------------------------#

    # Convert Units
    ## Make this a seperate function at some point??
    snap.pos *= 1e3  # [kpc]
    snap.vol *= 1e9  # [kpc^3]
    snap.mass *= 1e10  # [Msol]
    snap.hrgm *= 1e10  # [Msol]

    snapGas.pos *= 1e3  # [kpc]
    snapGas.vol *= 1e9  # [kpc^3]
    snapGas.mass *= 1e10  # [Msol]
    snapGas.hrgm *= 1e10  # [Msol]

    # Calculate New Parameters and Load into memory others we want to track
    snapGas = calculate_tracked_parameters(
        snapGas,
        snap,
        elements,
        elements_Z,
        elements_mass,
        elements_solar,
        Zsolar,
        omegabaryon0,
        snapNumber,
    )

    del snap

    # Pad stars and gas data with Nones so that all keys have values of same first dimension shape
    snapGas = pad_non_entries(snapGas, snapNumber)

    snapGas = high_res_only_gas_select(snapGas, snapNumber)

    # Assign SubHaloID and FoFHaloIDs
    snapGas = halo_id_finder(snapGas, snap_subfind, snapNumber, OnlyHalo=HaloID)

    ### Exclude values outside halo 0 ###
    if loadonlyhalo is True:
        snapGas = halo_only_gas_select(snapGas, snap_subfind, HaloID, snapNumber)

    # Pad stars and gas data with Nones so that all keys have values of same first dimension shape
    snapGas = pad_non_entries(snapGas, snapNumber)

    if TFCbool == True:
        # --------------------------------------------------------------------------#
        ####                    SELECTION                                        ###
        # --------------------------------------------------------------------------#
        print(f"[@{int(snapNumber)}]: Setting Selection Condition!")

        # Get Cell data and Cell IDs from tracers based on condition
        TracersTFC, CellsTFC, CellIDsTFC, ParentsTFC = get_tracers_from_cells(
            snapGas,
            snapTracers,
            TRACERSPARAMS,
            saveParams,
            saveTracersOnly,
            snapNumber=snapNumber,
        )

        # #Add snap data to temperature specific dictionary
        # print(f"Adding (T{targetT},{int(snap)}) to Dict")
        # FullDict.update({(f"T{targetT}",f"{int(snap)}"): CellsCFT})
        if saveTracers is True:
            for targetT in TRACERSPARAMS["targetTLst"]:
                for (rin, rout) in zip(
                        TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]
                ):
                    key = (f"T{targetT}", f"{rin}R{rout}")

                    out = {
                        (f"T{targetT}", f"{rin}R{rout}", f"{int(snapNumber)}"): {
                            "trid": TracersTFC[key]
                        }
                    }

                    savePath = (
                            DataSavepath
                            + f"_T{targetT}_{rin}R{rout}_{int(snapNumber)}_Tracers"
                            + FullDataPathSuffix
                    )

                    print(
                        "\n"
                        + f"[@{int(snapNumber)} @T{targetT} @{rin}R{rout}]: Saving Tracers ID ('trid') data as: "
                        + savePath
                    )

                    hdf5_save(savePath, out)
    else:
        TracersTFC = None
        CellsTFC = None
        CellIDsTFC = None
        ParentsTFC = None
        # #SUBSET
        # if (SUBSET is not None):
        #     print(f"[@{int(snapNumber)} @T{targetT}]: *** TRACER SUBSET OF {SUBSET} TAKEN! ***")
        #     TracersTFC = TracersTFC[:SUBSET]

    return TracersTFC, CellsTFC, CellIDsTFC, ParentsTFC, snapGas, snapTracers


# ------------------------------------------------------------------------------#

# #==============================================================================#
# #       t3000 MAIN ANALYSIS CODE - IN FUNC FOR MULTIPROCESSING
# #==============================================================================#
# def t3000_snap_analysis(snapNumber,targetT,TRACERSPARAMS,HaloID,CellIDsTFC,\
# elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
# saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,lazyLoadBool=True):
#     print("")
#     print(f"[@{int(snapNumber)} @T{targetT}]: Starting Snap {snapNumber}")
#
#     # load in the subfind group files
#     snap_subfind = load_subfind(snapNumber,dir=TRACERSPARAMS['simfile'])
#
#     # load in the gas particles mass and position only for HaloID 0.
#     #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
#     snapGas     = gadget_readsnap(snapNumber, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0,4], lazy_load=lazyLoadBool, subfind = snap_subfind)
#     # load tracers data
#
#     #Load Cell IDs - avoids having to turn lazy_load off...
#     # But ensures 'id' is loaded into memory before halo_only_gas_select is called
#     #  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
#     #   Be in memory so taking the subset would be skipped.
#     tmp = snapGas.data['id']
#     tmp = snapGas.data['age']
#     tmp = snapGas.data['hrgm']
#     tmp = snapGas.data['mass']
#     tmp = snapGas.data['pos']
#     tmp = snapGas.data['vol']
#     del tmp
#
#     print(f"[@{int(snapNumber)} @T{targetT}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")
#
#     #Centre the simulation on HaloID 0
#     snapGas  = set_centre(snap=snapGas,snap_subfind=snap_subfind,HaloID=HaloID,snapNumber = snapNumber)
#
#     #--------------------------#
#     ##    Units Conversion    ##
#     #--------------------------#
#
#     #Convert Units
#     ## Make this a seperate function at some point??
#     snapGas.pos *= 1e3 #[kpc]
#     snapGas.vol *= 1e9 #[kpc^3]
#     snapGas.mass *= 1e10 #[Msol]
#     snapGas.hrgm *= 1e10 #[Msol]
#
#     #Calculate New Parameters and Load into memory others we want to track
#     snapGas = calculate_tracked_parameters(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0, snapNumber)
#
#     #Pad stars and gas data with Nones so that all keys have values of same first dimension shape
#     snapGas = pad_non_entries(snapGas,snapNumber)
#
#     #Select only gas in High Res Zoom Region
#     snapGas = high_res_only_gas_select(snapGas,snapNumber)
#
#     #Find Halo=HaloID data for only selection snapshot. This ensures the
#     #selected tracers are originally in the Halo, but allows for tracers
#     #to leave (outflow) or move inwards (inflow) from Halo.
#
#     #Assign SubHaloID and FoFHaloIDs
#     snapGas = halo_id_finder(snapGas,snap_subfind,snapNumber)
#
#     if (snapNumber == int(TRACERSPARAMS['selectSnap'])):
#
#         snapGas = halo_only_gas_select(snapGas,snap_subfind,HaloID,snapNumber)
#
#     #Pad stars and gas data with Nones so that all keys have values of same first dimension shape
#     snapGas = pad_non_entries(snapGas,snapNumber)
#     ###
#     ##  Selection   ##
#     ###
#
#     whereCellsSelected = np.where(np.isin(snapGas.data['id'],CellIDsTFC))
#
#     for key, value in snapGas.data.items():
#         if (value is not None):
#             snapGas.data[key] = value[whereCellsSelected]
#
#     Rcrit = 500.
#
#     print(f"[@{int(snapNumber)} @T{targetT}]: Select approx HaloID = {int(HaloID)} by R<={Rcrit:0.02f} kpc")
#     Cond = np.where(snapGas.data['R']<=Rcrit)
#
#     NCells = len(snapGas.data['type'])
#     print(f"[@{int(snapNumber)} @T{targetT}]: Number of Cells Pre-Selection Condition : {NCells}")
#
#     for key, value in snapGas.data.items():
#         if (value is not None):
#             snapGas.data[key] = value[Cond]
#
#     NCells = len(snapGas.data['type'])
#     print(f"[@{int(snapNumber)} @T{targetT}]: Number of Cells Post-Selection Condition : {NCells}")
#
#     CellIDsCFT = snapGas.data['id']
#
#     print(f"[@{int(snapNumber)} @T{targetT}]: Selected!")
#     print(f"[@{int(snapNumber)} @T{targetT}]: Entering save Cells...")
#
#     CellsCFT = t3000_save_cells_data(snapGas,snapNumber,saveParams,saveTracersOnly)
#     # #Add snap data to temperature specific dictionary
#     # print(f"Adding (T{targetT},{int(snap)}) to Dict")
#     # FullDict.update({(f"T{targetT}",f"{int(snap)}"): CellsCFT})
#     out = {(f"T{targetT}",f"{int(snapNumber)}"): CellsCFT}
#
#     savePath = DataSavepath + f"_T{targetT}_{int(snapNumber)}"+ FullDataPathSuffix
#
#     print("\n" + f"[@{snapNumber} @T{targetT}]: Saving Cells data as: "+ savePath)
#
#     hdf5_save(savePath,out)
#
#     save_statistics(CellsCFT, targetT, snapNumber, TRACERSPARAMS, saveParams, DataSavepath, MiniDataPathSuffix)
#
#     sys.stdout.flush()
#     return {"out": out, "CellsCFT": CellsCFT, "CellIDsCFT": CellIDsCFT}
#
# #==============================================================================#
# #       PRE-MAIN ANALYSIS CODE
# #==============================================================================#
# """
#     The t3000 versions below are focussed on cell selection as opposed to a Tracer
#     analysis.
# """
# def t3000_cell_selection_snap_analysis(targetT,TRACERSPARAMS,HaloID,\
# elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
# saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,\
# lazyLoadBool=True,SUBSET=None,snapNumber=None,saveCells=True,loadonlyhalo=True):
#
#     if snapNumber is None:
#         snapNumber = TRACERSPARAMS['selectSnap']
#
#     print(f"[@{int(snapNumber)} @T{targetT}]: Starting T = {targetT} Analysis!")
#
#     # load in the subfind group files
#     snap_subfind = load_subfind(snapNumber,dir=TRACERSPARAMS['simfile'])
#
#     # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
#     snapGas     = gadget_readsnap(snapNumber, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0,4], loadonlyhalo=HaloID, lazy_load=lazyLoadBool, subfind = snap_subfind)
#     #Load Cell IDs - avoids having to turn lazy_load off...
#     # But ensures 'id' is loaded into memory before halo_only_gas_select is called
#     #  Else we wouldn't limit the IDs to the nearest Halo for that step as they wouldn't
#     #   Be in memory so taking the subset would be skipped.
#     tmp = snapGas.data['id']
#     tmp = snapGas.data['age']
#     tmp = snapGas.data['hrgm']
#     tmp = snapGas.data['mass']
#     tmp = snapGas.data['pos']
#     tmp = snapGas.data['vol']
#     del tmp
#
#     print(f"[@{int(snapNumber)} @T{targetT}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")
#
#     #Centre the simulation on HaloID 0
#     snapGas  = set_centre(snap=snapGas,snap_subfind=snap_subfind,HaloID=HaloID,snapNumber=snapNumber)
#
#     #--------------------------#
#     ##    Units Conversion    ##
#     #--------------------------#
#
#     #Convert Units
#     ## Make this a seperate function at some point??
#     snapGas.pos *= 1e3 #[kpc]
#     snapGas.vol *= 1e9 #[kpc^3]
#     snapGas.mass *= 1e10 #[Msol]
#     snapGas.hrgm *= 1e10 #[Msol]
#
#     #Calculate New Parameters and Load into memory others we want to track
#     snapGas = calculate_tracked_parameters(snapGas,elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,snapNumber)
#
#     #Pad stars and gas data with Nones so that all keys have values of same first dimension shape
#     snapGas = pad_non_entries(snapGas,snapNumber)
#
#     #Select only gas in High Res Zoom Region
#     snapGas = high_res_only_gas_select(snapGas,snapNumber)
#
#     #Assign SubHaloID and FoFHaloIDs
#     snapGas = halo_id_finder(snapGas,snap_subfind,snapNumber,OnlyHalo=HaloID)
#
#     ### Exclude values outside halo 0 ###
#     if (loadonlyhalo is True):
#
#         snapGas = halo_only_gas_select(snapGas,snap_subfind,HaloID,snapNumber)
#
#     #Pad stars and gas data with Nones so that all keys have values of same first dimension shape
#     snapGas = pad_non_entries(snapGas,snapNumber)
#
#     #--------------------------------------------------------------------------#
#     ####                    SELECTION                                        ###
#     #--------------------------------------------------------------------------#
#     print(f"[@{int(snapNumber)} @T{targetT}]: Setting Selection Condition!")
#
#     #Set condition for Tracer selection
#     whereGas = np.where(snapGas.type==0)[0]
#     whereStars = np.where(snapGas.type==4)[0]
#     NGas = len(snapGas.type[whereGas])
#
#     Cond = np.where((snapGas.data['T'][whereGas]>=1.*10**(targetT-TRACERSPARAMS['deltaT'])) & \
#                     (snapGas.data['T'][whereGas]<=1.*10**(targetT+TRACERSPARAMS['deltaT'])) & \
#                     (snapGas.data['R'][whereGas]>=TRACERSPARAMS['Rinner']) & \
#                     (snapGas.data['R'][whereGas]<=TRACERSPARAMS['Router']) &\
#                     (snapGas.data['sfr'][whereGas]<=0))[0]
#
#     NCells = len(snapGas.data['type'])
#     print(f"[@{int(snapNumber)} @T{targetT}]: Number of Cells Pre-Selection Condition : {NCells}")
#
#     for key, value in snapGas.data.items():
#         if (value is not None):
#             snapGas.data[key] = value[Cond]
#
#     NCells = len(snapGas.data['type'])
#     print(f"[@{int(snapNumber)} @T{targetT}]: Number of Cells Post-Selection Condition : {NCells}")
#
#     CellIDsTFC = snapGas.data['id']
#     # #Add snap data to temperature specific dictionary
#     # print(f"Adding (T{targetT},{int(snap)}) to Dict")
#     # FullDict.update({(f"T{targetT}",f"{int(snap)}"): CellsCFT})
#     if (saveCells is True):
#         out = {(f"T{targetT}",f"{int(snapNumber)}"): {'id': CellIDsTFC}}
#
#         savePath = DataSavepath + f"_T{targetT}_{int(snapNumber)}_CellIDs"+ FullDataPathSuffix
#
#         print("\n" + f"[@{int(snapNumber)} @T{targetT}]: Saving Cell ID ('id') data as: "+ savePath)
#
#         hdf5_save(savePath,out)
#
#     #SUBSET
#     if (SUBSET is not None):
#         print(f"[@{int(snapNumber)} @T{targetT}]: *** TRACER SUBSET OF {SUBSET} TAKEN! ***")
#         TracersTFC = TracersTFC[:SUBSET]
#
#     return CellIDsTFC, snapGas
#
# #------------------------------------------------------------------------------#


def get_tracers_from_cells(
        snapGas, snapTracers, TRACERSPARAMS, saveParams, saveTracersOnly, snapNumber
):
    """
    Select the Cells which meet the conditional where Cond. Select from these cells
    those which ALSO contain tracers. Pass this to save_tracer_data to select the
    data from these cells (as defined by which data is requested to be saved in
    saveParams and saveTracersOnly).

        A reminder that saveTracersOnly includes all data considered essential
        for the code to run, and whatever other parameters the user requests to
        track that we do not wish statistics for.
    """
    TracersFinal = {}
    CellsFinal = {}
    CellIDsFinal = {}
    ParentsFinal = {}
    for targetT in TRACERSPARAMS["targetTLst"]:
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            print(f"[@T{targetT} @{rin}R{rout} @{snapNumber}]: Get Tracers From Cells!")
            # Set condition for Tracer selection
            whereGas = np.where(snapGas.type == 0)[0]
            whereStars = np.where(snapGas.type == 4)[0]
            NGas = len(snapGas.type[whereGas])

            Cond = np.where(
                (
                        snapGas.data["T"][whereGas]
                        >= 1.0 * 10 ** (targetT - TRACERSPARAMS["deltaT"])
                )
                & (
                        snapGas.data["T"][whereGas]
                        <= 1.0 * 10 ** (targetT + TRACERSPARAMS["deltaT"])
                )
                & (snapGas.data["R"][whereGas] >= rin)
                & (snapGas.data["R"][whereGas] <= rout)
                & (snapGas.data["sfr"][whereGas] <= 0)
            )[0]
            # Select Cell IDs for cells which meet condition
            CellIDs = snapGas.id[Cond]

            # Select Parent IDs in Cond list
            #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
            ParentsIndices = np.where(np.isin(snapTracers.prid, CellIDs))

            # Select Tracers and Parent IDs from cells that meet condition and contain tracers
            Tracers = snapTracers.trid[ParentsIndices]
            Parents = snapTracers.prid[ParentsIndices]

            # Get Gas meeting Cond AND with tracers
            CellsIndices = np.where(np.isin(snapGas.id, Parents))
            CellIDs = snapGas.id[CellsIndices]

            Ntracers = int(len(Tracers))
            print(f"[@{snapNumber}]: Number of tracers = {Ntracers}")

            Cells = save_tracer_data(
                snapGas,
                Tracers,
                Parents,
                CellIDs,
                CellsIndices,
                Ntracers,
                snapNumber,
                saveParams,
                saveTracersOnly,
            )

            key = (f"T{targetT}", f"{rin}R{rout}")
            TracersFinal.update({key: Tracers})
            CellsFinal.update({key: Cells})
            CellIDsFinal.update({key: CellIDs})
            ParentsFinal.update({key: Parents})

    return TracersFinal, CellsFinal, CellIDsFinal, ParentsFinal


# ------------------------------------------------------------------------------#
def get_cells_from_tracers(
        snapGas, snapTracers, Tracers, saveParams, saveTracersOnly, snapNumber
):
    """
    Get the IDs and data from cells containing the Tracers passed in in Tracers.
    Pass the indices of these cells to save_tracer_data for adjusting the entries of Cells
    by which cells contain tracers.
    Will return an entry for EACH tracer, which will include duplicates of certain
    Cells where more than one tracer is contained.
    """

    # Select indices (positions in array) of Tracer IDs which are in the Tracers list
    TracersIndices = np.where(np.isin(snapTracers.trid, Tracers))

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

    # #Create a list of indices of CellIDs that contains a ParentID:
    # #   This WILL include duplicates!
    # CellIndex = np.array([])
    # for ID in Parents:
    #     value = np.where(np.isin(CellIDs,ID))
    #     CellIndex = np.append(CellIndex, value)
    #
    # #Make sure CellIndex is of the right type
    # CellIndex = np.array(list(map(int, CellIndex)))
    #
    # #Grab the Cell IDs for cells with Tracers,
    # #   Using CelolIndex here will return duplicate entries s.t. every
    # #   there is a cell for every tracer including duplicates
    # CellIDs = CellIDs[CellIndex]
    # #Grabe The Indices of the snapGas.id's  with parents in. Using
    # #   CellIndex here returns duplicates such that there is a cell for every tracers.
    # CellsIndices = CellIndicesShort[CellIndex]

    # Save number of tracers
    Ntracers = int(len(TracersCFT))
    print(f"[@{snapNumber}]: Number of tracers = {Ntracers}")

    Cells = save_cells_data(
        snapGas,
        TracersCFT,
        Parents,
        CellIDs,
        CellIndicesShort,
        Ntracers,
        snapNumber,
        saveParams,
        saveTracersOnly,
    )

    return TracersCFT, Cells, CellIDs, Parents


# ------------------------------------------------------------------------------#
def save_tracer_data(
        snapGas,
        Tracers,
        Parents,
        CellIDs,
        CellsIndices,
        Ntracers,
        snapNumber,
        saveParams,
        saveTracersOnly,
):
    """
    Save the requested data from the Tracers' Cells data. Only saves the cells
    assoicated with a Tracer, as determined by CellsIndices.
    """
    print(f"[@{snapNumber}]: Saving Tracer Data!")

    # assert np.shape(CellsIndices)[0] == Ntracers,"[@save_cells_data]: Fewer CellsIndices than Tracers!!"
    # Select the data for Cells that meet Cond which contain tracers
    #   Does this by creating new dict from old data.
    #       Only selects values at index where Cell meets cond and contains tracers
    Cells = {}
    for key in saveParams:
        Cells.update({key: snapGas.data[key][CellsIndices]})

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    #   Now perform save of parameters not tracked in stats (saveTracersOnly params)#
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # Redshift
    redshift = snapGas.redshift  # z
    aConst = 1.0 / (1.0 + redshift)  # [/]

    # Get lookback time in Gyrs
    # [0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
        0
    ]  # [Gyrs]

    for TracerSaveParameter in saveTracersOnly:
        if TracerSaveParameter == "Lookback":
            Cells.update({"Lookback": np.array([lookback])})
        elif TracerSaveParameter == "Ntracers":
            Cells.update({"Ntracers": np.array([Ntracers])})
        elif TracerSaveParameter == "Snap":
            Cells.update({"Snap": np.array([snapNumber])})
        elif TracerSaveParameter == "trid":
            # Save Tracer IDs
            Cells.update({"trid": Tracers})
        elif TracerSaveParameter == "prid":
            # Save Parent Cell IDs
            Cells.update({"prid": Parents})
        elif TracerSaveParameter == "id":
            # Save Cell IDs
            Cells.update({"id": CellIDs})
        else:
            Cells.update(
                {
                    f"{TracerSaveParameter}": snapGas.data[TracerSaveParameter][
                        CellsIndices
                    ]
                }
            )

    return Cells


# ------------------------------------------------------------------------------#


def save_cells_data(
        snapGas,
        Tracers,
        Parents,
        CellIDs,
        CellsIndices,
        Ntracers,
        snapNumber,
        saveParams,
        saveTracersOnly,
):
    """
    Save the requested data from the Tracers' Cells data. Should save an entry for every Tracer,
    duplicating some cells.
    """
    print(f"[@{snapNumber}]: Saving Cells Data!")

    # Select the data for Cells that meet Cond which contain tracers
    #   Does this by creating new dict from old data.
    #       Only selects values at index where Cell meets cond and contains tracers

    # assert np.shape(CellsIndices)[0] == Ntracers,"[@save_cells_data]: Fewer CellsIndices than Tracers!!"

    Cells = {}
    for key in saveParams:
        Cells.update({key: snapGas.data[key][CellsIndices]})

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    #   Now perform save of parameters not tracked in stats (saveTracersOnly params)#
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # Redshift
    redshift = snapGas.redshift  # z
    aConst = 1.0 / (1.0 + redshift)  # [/]

    # Get lookback time in Gyrs
    # [0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
        0
    ]  # [Gyrs]

    for TracerSaveParameter in saveTracersOnly:
        if TracerSaveParameter == "Lookback":
            Cells.update({"Lookback": np.array([lookback])})
        elif TracerSaveParameter == "Ntracers":
            Cells.update({"Ntracers": np.array([Ntracers])})
        elif TracerSaveParameter == "Snap":
            Cells.update({"Snap": np.array([snapNumber])})
        elif TracerSaveParameter == "trid":
            # Save Tracer IDs
            Cells.update({"trid": Tracers})
        elif TracerSaveParameter == "prid":
            # Save Parent Cell IDs
            Cells.update({"prid": Parents})
        elif TracerSaveParameter == "id":
            # Save Cell IDs
            Cells.update({"id": CellIDs})
        else:
            Cells.update(
                {
                    f"{TracerSaveParameter}": snapGas.data[TracerSaveParameter][
                        CellsIndices
                    ]
                }
            )

    return Cells


def t3000_save_cells_data(snapGas, snapNumber, saveParams, saveTracersOnly):
    print(f"[@{snapNumber}]: Saving Cell Data!")

    Ncells = len(snapGas.data["id"])

    print(f"[@{snapNumber}]: Ncells = {int(Ncells)}")

    # Select the data for Cells that meet Cond which contain tracers
    #   Does this by creating new dict from old data.
    #       Only selects values at index where Cell meets cond and contains tracers
    Cells = {}
    for key in saveParams:
        Cells.update({key: snapGas.data[key]})

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    #   Now perform save of parameters not tracked in stats (saveTracersOnly params)#
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    # Redshift
    redshift = snapGas.redshift  # z
    aConst = 1.0 / (1.0 + redshift)  # [/]

    # Get lookback time in Gyrs
    # [0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
        0
    ]  # [Gyrs]

    for TracerSaveParameter in saveTracersOnly:
        if TracerSaveParameter == "Lookback":
            Cells.update({"Lookback": np.array([lookback])})
        elif TracerSaveParameter == "Ncells":
            Cells.update({"Ncells": np.array([Ncells])})
        elif TracerSaveParameter == "Snap":
            Cells.update({"Snap": np.array([snapNumber])})
        elif TracerSaveParameter == "id":
            # Save Cell IDs
            Cells.update({"id": snapGas.data["id"]})
        else:
            Cells.update({f"{TracerSaveParameter}": snapGas.data[TracerSaveParameter]})

    return Cells


# ------------------------------------------------------------------------------#
##  FvdV weighted percentile code:
# ------------------------------------------------------------------------------#
def weighted_percentile(data, weights, perc, key):
    """
    Find the weighted Percentile of the data.
    Returns a zero value and warning if all Data (or all weights) are NaN
    """

    # percentage to decimal
    perc /= 100.0

    # Indices of data array in sorted form
    ind_sorted = np.argsort(data)

    # Sort the data
    sorted_data = np.array(data)[ind_sorted]

    # Sort the weights by the sorted data sorting
    sorted_weights = np.array(weights)[ind_sorted]

    # Remove nan entries
    whereDataIsNotNAN = np.where(np.isnan(sorted_data) == False)

    sorted_data = sorted_data[whereDataIsNotNAN]
    sorted_weights = sorted_weights[whereDataIsNotNAN]

    whereWeightsIsNotNAN = np.where(np.isnan(sorted_weights) == False)
    sorted_weights = sorted_weights[whereWeightsIsNotNAN]

    nDataNotNan = len(sorted_data)
    nWeightsNotNan = len(sorted_weights)

    if nDataNotNan > 0:
        # Find the cumulative sum of the weights
        cm = np.cumsum(sorted_weights)

        # Find indices where cumulative some as a fraction of final cumulative sum value is greater than percentage
        whereperc = np.where(cm / float(cm[-1]) >= perc)

        # Reurn the first data value where above is true
        out = sorted_data[whereperc[0][0]]
    else:
        print(key)
        print("[@WeightPercent:] Warning! Data all nan! Returning 0 value!")
        out = np.array([0.0])

    return out


# ------------------------------------------------------------------------------#


def set_centre(snap, snap_subfind, HaloID, snapNumber):
    """
    Set centre of simulation box to centre on Halo HaloID.
    Set velocities to be centred on the median velocity of this halo.
    """
    print(f"[@{snapNumber}]: Centering!")

    # subfind has calculated its centre of mass for you
    HaloCentre = snap_subfind.data["fpos"][HaloID, :]
    # use the subfind COM to centre the coordinates on the galaxy
    snap.data["pos"] = snap.data["pos"] - np.array(HaloCentre)

    snap.data["R"] = np.linalg.norm(snap.data["pos"], axis=1)

    whereGas = np.where(snap.type == 0)
    # Adjust to galaxy centred velocity
    (wheredisc,) = np.where(
        (snap.data["R"][whereGas] < 20.0) & (snap.data["sfr"] > 0.0)
    )
    snap.vel = snap.vel - np.median(snap.vel[wheredisc], axis=0)
    return snap


# ------------------------------------------------------------------------------#
def calculate_tracked_parameters(
        snapGas,
        snap,
        elements,
        elements_Z,
        elements_mass,
        elements_solar,
        Zsolar,
        omegabaryon0,
        snapNumber,
):
    """
    Calculate the physical properties of all cells, or gas only where necessary
    """
    print(f"[@{snapNumber}]: Calculate Tracked Parameters!")

    whereGas = np.where(snapGas.type == 0)
    # Density is rho/ <rho> where <rho> is average baryonic density
    rhocrit = (
            3.0
            * (snapGas.omega0 * (1.0 + snapGas.redshift) ** 3 + snapGas.omegalambda)
            * (snapGas.hubbleparam * 100.0 * 1e5 / (c.parsec * 1e6)) ** 2
            / (8.0 * pi * c.G)
    )
    rhomean = (
            3.0
            * (snapGas.omega0 * (1.0 + snapGas.redshift) ** 3)
            * (snapGas.hubbleparam * 100.0 * 1e5 / (c.parsec * 1e6)) ** 2
            / (8.0 * pi * c.G)
    )

    # Mean weight [amu]
    meanweight = sum(snapGas.gmet[whereGas, 0:9][0], axis=1) / (
            sum(snapGas.gmet[whereGas, 0:9][0] / elements_mass[0:9], axis=1)
            + snapGas.ne[whereGas] * snapGas.gmet[whereGas, 0][0]
    )

    # 3./2. N KB
    Tfac = ((3.0 / 2.0) * c.KB) / (meanweight * c.amu)

    snapGas.data["dens"] = (
            (snapGas.rho[whereGas] / (c.parsec * 1e6) ** 3) * c.msol * 1e10
    )  # [g cm^-3]
    gasX = snapGas.gmet[whereGas, 0][0]

    # Temperature = U / (3/2 * N KB) [K]
    snapGas.data["T"] = (snapGas.u[whereGas] * 1e10) / (Tfac)  # K
    snapGas.data["n_H"] = snapGas.data["dens"][whereGas] / c.amu * gasX  # cm^-3
    snapGas.data["rho_rhomean"] = snapGas.data["dens"][whereGas] / (
            rhomean * omegabaryon0 / snapGas.omega0
    )  # rho / <rho>
    snapGas.data["Tdens"] = (
            snapGas.data["T"][whereGas] * snapGas.data["rho_rhomean"][whereGas]
    )

    bfactor = (
            1e6
            * (np.sqrt(1e10 * c.msol) / np.sqrt(c.parsec * 1e6))
            * (1e5 / (c.parsec * 1e6))
    )  # [microGauss]

    # Magnitude of Magnetic Field [micro Guass]
    snapGas.data["B"] = np.linalg.norm(
        (snapGas.data["bfld"][whereGas] * bfactor), axis=1
    )

    # Radius [kpc]
    snapGas.data["R"] = np.linalg.norm(snapGas.data["pos"][whereGas], axis=1)  # [Kpc]
    snap.data["R"] = np.linalg.norm(snap.data["pos"], axis=1)  # [Kpc]

    KpcTokm = 1e3 * c.parsec * 1e-5
    # Radial Velocity [km s^-1]
    snapGas.data["vrad"] = (
            snapGas.pos[whereGas] * KpcTokm * snapGas.vel[whereGas]
    ).sum(axis=1)
    snapGas.data["vrad"] /= snapGas.data["R"][whereGas] * KpcTokm

    # Cooling time [Gyrs]
    GyrToSeconds = 365.25 * 24.0 * 60.0 * 60.0 * 1e9

    snapGas.data["tcool"] = (
                                    snapGas.data["u"][whereGas] * 1e10 * snapGas.data["dens"][whereGas]
                            ) / (
                                    GyrToSeconds
                                    * snapGas.data["gcol"][whereGas]
                                    * snapGas.data["n_H"][whereGas] ** 2
                            )  # [Gyrs]
    snapGas.data["theat"] = snapGas.data["tcool"].copy()

    coolingGas = np.where(snapGas.data["tcool"] < 0.0)
    heatingGas = np.where(snapGas.data["tcool"] > 0.0)
    zeroChangeGas = np.where(snapGas.data["tcool"] == 0.0)

    snapGas.data["tcool"][coolingGas] = abs(snapGas.data["tcool"][coolingGas])
    snapGas.data["tcool"][heatingGas] = np.nan
    snapGas.data["tcool"][zeroChangeGas] = np.nan

    snapGas.data["theat"][coolingGas] = np.nan
    snapGas.data["theat"][heatingGas] = abs(snapGas.data["theat"][heatingGas])
    snapGas.data["theat"][zeroChangeGas] = np.nan

    # Load in metallicity
    snapGas.data["gz"] = snapGas.data["gz"][whereGas] / Zsolar
    # Load in Metals
    tmp = snapGas.data["gmet"]
    # Load in Star Formation Rate
    tmp = snapGas.data["sfr"]

    # Specific Angular Momentum [kpc km s^-1]
    snapGas.data["L"] = sqrt(
        (
                cross(snapGas.data["pos"][whereGas], snapGas.data["vel"][whereGas]) ** 2.0
        ).sum(axis=1)
    )

    snapGas.data["ndens"] = snapGas.data["dens"][whereGas] / (meanweight * c.amu)
    # Thermal Pressure : P/k_B = n T [$ # K cm^-3]
    snapGas.data["P_thermal"] = snapGas.data["ndens"] * snapGas.T

    # Magnetic Pressure [P/k_B K cm^-3]
    snapGas.data["P_magnetic"] = ((snapGas.data["B"][whereGas] * 1e-6) ** 2) / (
            8.0 * pi * c.KB
    )

    snapGas.data["P_tot"] = (
            snapGas.data["P_thermal"][whereGas] + snapGas.data["P_magnetic"][whereGas]
    )

    snapGas.data["Pthermal_Pmagnetic"] = (
            snapGas.data["P_thermal"][whereGas] / snapGas.data["P_magnetic"][whereGas]
    )

    # Kinetic "Pressure" [P/k_B K cm^-3]
    snapGas.data["P_kinetic"] = (
            (snapGas.rho[whereGas] / (c.parsec * 1e6) ** 3)
            * 1e10
            * c.msol
            * (1.0 / c.KB)
            * (np.linalg.norm(snapGas.data["vel"][whereGas] * 1e5, axis=1)) ** 2
    )

    # Sound Speed [(erg K^-1 K ??? g^-1)^1/2 = (g cm^2 s^-2 g^-1)^(1/2) = km s^-1]
    snapGas.data["csound"] = sqrt(
        ((5.0 / 3.0) * c.KB * snapGas.data["T"][whereGas]) / (meanweight * c.amu * 1e5)
    )

    # [cm kpc^-1 kpc cm^-1 s^1 = s / GyrToSeconds = Gyr]
    snapGas.data["tcross"] = (
            (KpcTokm * 1e3 / GyrToSeconds)
            * (snapGas.data["vol"][whereGas]) ** (1.0 / 3.0)
            / snapGas.data["csound"][whereGas]
    )

    rsort = np.argsort(snap.data["R"])
    runsort = np.argsort(rsort)

    rhosorted = (3.0 * np.cumsum(snap.data["mass"][rsort])) / (
            4.0 * pi * (snap.data["R"][rsort]) ** 3
    )
    rho = rhosorted[runsort]

    # Free Fall time [Gyrs]
    snap.data["tff"] = sqrt(
        (3.0 * pi) / (32.0 * ((c.G * c.msol) / ((1e3 * c.parsec) ** 3) * rho))
    ) * (1.0 / GyrToSeconds)

    whereGasStars = np.where(np.isin(snap.type, [0, 4]) == True)
    snapGas.data["tff"] = snap.data["tff"][whereGasStars].copy()
    snapGas.data["tff"] = snapGas.data["tff"][whereGas]

    # Cooling time over free fall time
    snapGas.data["tcool_tff"] = (
            snapGas.data["tcool"][whereGas] / snapGas.data["tff"][whereGas]
    )
    del tmp

    return snapGas


# ------------------------------------------------------------------------------#
def halo_only_gas_select(snapGas, snap_subfind, Halo=0, snapNumber=None):
    """
    Select only the snapGas entries associated with Sub Halo number Halo
    and unbound (-1).
    """
    print(f"[@{snapNumber}]: Select only Halo {Halo} and 'unbound' Gas!")

    HaloList = [float(Halo), -1.0]
    whereHalo = np.where(np.isin(snapGas.data["SubHaloID"], HaloList))[0]

    # #Find length of the first n entries of particle type 0 that are associated with HaloID 0: ['HaloID', 'particle type']
    # gaslength = snap_subfind.data['slty'][Halo,0]
    #
    # whereGas = np.where(snapGas.type==0)[0]
    # whereStars = np.where(snapGas.type==4)[0]
    # NGas = len(snapGas.type[whereGas])
    # NStars = len(snapGas.type[whereStars])
    #
    # selectGas = [ii for ii in range(0,gaslength)]
    # selectStars = [ii for ii in range(NGas,NStars)]
    #
    # selected = selectGas + selectStars

    # Take only data from above HaloID
    for key, value in snapGas.data.items():
        if value is not None:
            snapGas.data[key] = value[whereHalo]

    return snapGas


# ------------------------------------------------------------------------------#
def high_res_only_gas_select(snapGas, snapNumber):
    """
    Grab only snapGas entries for gas where high res gas mass (hrgm)
    is greater than 90% of the cell mass. This defines the cosmological
    Zoom region.
    """
    print(f"[@{snapNumber}]: Select High Res Gas Only!")

    whereGas = np.where(snapGas.data["type"] == 0)
    whereStars = np.where(snapGas.data["type"] == 4)

    whereHighRes = np.where(
        snapGas.data["hrgm"][whereGas] >= 0.90 * snapGas.data["mass"][whereGas]
    )

    selected = np.array(whereHighRes[0].tolist() + whereStars[0].tolist())

    for key, value in snapGas.data.items():
        if value is not None:
            snapGas.data[key] = value[selected]

    return snapGas


# ------------------------------------------------------------------------------#
def halo_id_finder(snapGas, snap_subfind, snapNumber, OnlyHalo=None):
    """
    Assign a unique ID value to each SubFind SubHalo --> SubHaloID
    Assign a unique ID value to each FoF Halo --> FoFHaloID
    Assign -1 to SubHaloID for unbound matter
    Assign NaN to unclassified (no halo) gas and stars

    Inputs: snapGas, snap_subfind
    OutPuts: snapGas
    """

    print(f"[@{snapNumber}]: HaloID Finder!")

    types = [0, 4]

    # Make a pre-computed list for these where type = 0 or 4
    #   This adds a speed advantage to the rest of this function =)
    whereTypeList = []
    for tp in types:
        whereType = np.where(snapGas.data["type"] == tp)
        whereTypeList.append(whereType)

    # Create some blank ID arrays, and set NaN to all values.

    snapGas.data["FoFHaloID"] = np.full(
        shape=np.shape(snapGas.data["type"]), fill_value=np.nan
    )
    snapGas.data["SubHaloID"] = np.full(
        shape=np.shape(snapGas.data["type"]), fill_value=np.nan
    )

    fnsh = snap_subfind.data["fnsh"]
    flty = snap_subfind.data["flty"]
    slty = snap_subfind.data["slty"]

    # Select only Halo == OnlyHalo
    if OnlyHalo != None:
        fnsh = np.array(fnsh[OnlyHalo])
        flty = np.array(flty[OnlyHalo, :])

    cumsumfnsh = np.cumsum(fnsh)
    cumsumflty = np.cumsum(flty, axis=0)
    cumsumslty = np.cumsum(slty, axis=0)

    # Loop over particle types
    for (ii, tp) in enumerate(types):
        # print(f"Haloes for particle type {tp}")
        printpercent = 5.0
        printcount = 0.0
        subhalo = 0
        fofhalo = 0

        whereType = whereTypeList[ii]

        # if cumsumflty is 2D (has more than one halo) make iterator full list
        #   else make iterator single halo
        if np.shape(np.shape(cumsumflty))[0] == 1:
            cumsumfltyIterator = np.array([cumsumflty[tp]])
        else:
            cumsumfltyIterator = cumsumflty[:, tp]

        # Loop over FoF Haloes as identified by an entry in flty
        for (fofhalo, csflty) in enumerate(cumsumfltyIterator):

            percentage = float(fofhalo) / float(len(cumsumfltyIterator)) * 100.0
            if percentage >= printcount:
                # print(f"{percentage:0.02f}% Halo IDs assigned!")
                printcount += printpercent

            if fofhalo == 0:
                # Start from beginning of data for fofhalo == 0
                nshLO = 0
                nshUP = cumsumfnsh[fofhalo]
                # No offset from flty at start
                lowest = 0
            else:
                # Grab entries from end of last FoFhalo to end of new FoFhalo
                nshLO = cumsumfnsh[fofhalo - 1]
                nshUP = cumsumfnsh[fofhalo]
                # Offset the indices of the data to be attributed to the new FoFhalo by the end of the last FoFhalo
                lowest = cumsumflty[fofhalo - 1, tp]

            # Find the cumulative sum (and thus index ranges) of the subhaloes for THIS FoFhalo ONLY!
            cslty = np.cumsum(snap_subfind.data["slty"][nshLO:nshUP, tp], axis=0)

            # Start the data selection from end of previous FoFHalo and continue lower bound to last slty entry
            lower = np.append(np.array(lowest), cslty + lowest)
            # Start upper bound from first slty entry (+ offset) to end on cumulative flty for "ubound" material
            upper = np.append(cslty + lowest, csflty)
            # print(f"lower[0] {lower[0]} : lower[-1] {lower[-1]}")
            # print(f"upper[0] {upper[0]} : upper[-1] {upper[-1]}")

            # Some Sanity checks. There should be 1 index pair for each subhalo, +1 for upper and lower bounds...
            assert len(lower) == (
                    nshUP + 1 - nshLO
            ), "[@halo_id_finder]: Lower selection list has fewer entries than number of subhaloes!"
            assert len(upper) == (
                    nshUP + 1 - nshLO
            ), "[@halo_id_finder]: Upper selection list has fewer entries than number of subhaloes!"

            # Loop over the index pairs, and assign all bound material (that is, all material apart from the end of slty to flty final pair)
            #  a subhalo number
            #       In the case where only 1 index is returned we opt to assign this single gas cell its own subhalo number
            for (lo, up) in zip(lower[:-1], upper[:-1]):
                # print(f"lo {lo} : up {up} --> subhalo {subhalo}")

                if lo == up:
                    whereSelectSH = whereType[0][lo]
                else:
                    # This notation allows us to select the entries for the subhalo, from the particle type tp list.
                    #   Double slicing [whereType][lo:up] fails as it modifies the outer copy but doesn't alter original
                    whereSelectSH = whereType[0][lo:up]
                snapGas.data["SubHaloID"][whereSelectSH] = subhalo
                subhalo += 1

            # Assign the whole csflty range a FoFhalo number
            if lower[0] == upper[-1]:
                whereSelectSHFoF = whereType[0][lower[0]]
            else:
                whereSelectSHFoF = whereType[0][lower[0]: upper[-1]]

            snapGas.data["FoFHaloID"][whereSelectSHFoF] = fofhalo

            # Provided there exists more than one entry, assign the difference between slty and flty indices a "-1"
            #   This will effectively discriminate between unbound gas (-1) and unassigned gas (NaN).
            if lower[-1] == upper[-1]:
                continue
            else:
                whereSelectSHunassigned = whereType[0][lower[-1]: upper[-1]]

            snapGas.data["SubHaloID"][whereSelectSHunassigned] = -1

    return snapGas


# ------------------------------------------------------------------------------#
def load_tracers_parameters(TracersParamsPath):
    TRACERSPARAMS = pd.read_csv(
        TracersParamsPath,
        delimiter=" ",
        header=None,
        usecols=[0, 1],
        skipinitialspace=True,
        index_col=0,
        comment="#",
    ).to_dict()[1]

    # Convert Dictionary items to (mostly) floats
    for key, value in TRACERSPARAMS.items():
        if (
                (key == "targetTLst")
                or (key == "phasesSnaps")
                or (key == "Axes")
                or (key == "percentiles")
                or (key == "Rinner")
                or (key == "Router")
        ):
            # Convert targetTLst to list of floats
            lst = value.split(",")
            lst2 = [float(item) for item in lst]
            TRACERSPARAMS.update({key: lst2})
        elif (
                (key == "saveParams")
                or (key == "saveTracersOnly")
                or (key == "saveEssentials")
                or (key == "dtwParams")
                or (key == "dtwlogParams")
        ):
            # Convert targetTLst to list of strings
            lst = value.split(",")
            strlst = [str(item) for item in lst]
            TRACERSPARAMS.update({key: strlst})
        elif (key == "simfile") or (key == "savepath"):
            # Keep simfile as a string
            TRACERSPARAMS.update({key: value})
        else:
            # Convert values to floats
            TRACERSPARAMS.update({key: float(value)})

    TRACERSPARAMS["Axes"] = [int(axis) for axis in TRACERSPARAMS["Axes"]]

    possibleAxes = [0, 1, 2]
    for axis in possibleAxes:
        if axis not in TRACERSPARAMS["Axes"]:
            TRACERSPARAMS.update({"zAxis": [axis]})

    if TRACERSPARAMS["QuadPlotBool"] == 1.0:
        TRACERSPARAMS["QuadPlotBool"] = True
    else:
        TRACERSPARAMS["QuadPlotBool"] = False

    # Get Temperatures as strings in a list so as to form "4-5-6" for savepath.
    Tlst = [str(item) for item in TRACERSPARAMS["targetTLst"]]
    Tstr = "-".join(Tlst)

    # This rather horrible savepath ensures the data can only be combined with the right input file, TracersParams.csv, to be plotted/manipulated
    DataSavepath = (
            TRACERSPARAMS["savepath"]
            + f"Data_selectSnap{int(TRACERSPARAMS['selectSnap'])}_targetT{Tstr}"
    )

    return TRACERSPARAMS, DataSavepath, Tlst


# ------------------------------------------------------------------------------#
def load_haloes_selected(HaloPathBase,SelectedHaloesPath):
    SELECTEDHALOES = pd.read_csv(
        SelectedHaloesPath,
        delimiter=" ",
        header=None,
        usecols=[0, 1],
        skipinitialspace=True,
        index_col=0,
        comment="#",
    ).to_dict()[1]

    HALOPATHS = {}
    # Convert Dictionary items to (mostly) floats
    for key, value in SELECTEDHALOES.items():
        if (
                (key == "selectedHaloes")
        ):
            # Convert targetTLst to list of floats
            lst = value.split(",")
            strlst = [str(item) for item in lst]
            pathlst = [HaloPathBase + item  for item in strlst]
            SELECTEDHALOES.update({key: strlst})
            HALOPATHS.update({'haloPaths' :pathlst })
        else:
            raise Exception('None "selectedHaloes" data fields detected in ' + SelectedHaloesPath)

    return SELECTEDHALOES['selectedHaloes'], HALOPATHS['haloPaths']


# ------------------------------------------------------------------------------#
#
# def get_individual_cell_from_tracer(
#         Tracers, Parents, CellIDs, SelectedTracers, Data, NullEntry=np.nan
# ):
#     if np.shape(np.shape(Data))[0] == 1:
#         dimension = 1
#         NullEntry = np.nan
#     else:
#         dimension = 3
#         NullEntry = [np.nan for dd in range(dimension)]
#
#     # Select which of the SelectedTracers are in Tracers from this snap
#     SelectedTrids = np.where(np.isin(SelectedTracers, Tracers), SelectedTracers, np.nan)
#     # whereTrids = np.where(np.isin(Tracers,SelectedTracers)==True)[0]
#     _, SelectedIndices, TridIndices = np.intersect1d(    SelectedTracers, Tracers, return_indices=True)
#
#     pridBlank = np.full(shape=np.shape(SelectedTracers), fill_value=-1)
#     pridBlank[SelectedIndices] = Parents[TridIndices]
#
#     SelectedPrids = pridBlank
#
#     if dimension == 1:
#         dataBlank = np.full(shape=np.shape(SelectedTracers), fill_value=NullEntry)
#     elif dimension == 3:
#         dataBlank = np.full(
#             shape=(np.shape(SelectedTracers)[0], dimension), fill_value=NullEntry
#         )
#     else:
#         print(
#             f"[@get_individual_cell_from_tracer]: dimension not 1 or 3! dataBlank Failure! Data neither 3D vector or 1D scalar!"
#         )
#         assert True == False
#
#     parentsEqualsCellIDs = np.array_equal(Parents,CellIDs)
#     if parentsEqualsCellIDs == False:
#         for (ind, ID) in enumerate(SelectedPrids):
#             if ID != -1:
#                 value = np.where(np.isin(CellIDs, ID))[0]
#                 if np.shape(value)[0] > 0:
#                     dataBlank[ind] = Data[value]
#     else:
#         dataBlank = Data[np.where(np.isin(CellIDs,Parents))[0]]
#     SelectedData = dataBlank
#
#     assert np.shape(SelectedTrids) == np.shape(SelectedTracers)
#     assert np.shape(SelectedPrids) == np.shape(SelectedTracers)
#     assert np.shape(SelectedData)[0] == np.shape(SelectedTracers)[0]
#     print("np.shape(SelectedTracers)",np.shape(SelectedTracers))
#     print("np.shape(SelectedTrids)",np.shape(SelectedTrids))
#     print("np.shape(SelectedPrids)",np.shape(SelectedPrids))
#     print("np.shape(SelectedData)[0]",np.shape(SelectedData)[0])
#     # #Select which of the SelectedTracers are in Tracers from this snap
#     # TracersTruthy = np.isin(SelectedTracers,Tracers)
#     # #Grab the indices of the trid in Tracers if it is contained in SelectedTracers
#     # #   Also add list of Tracer IDs trids to list for debugging purposes
#     # TracersIndices = []
#     # TracersReturned = []
#     # ParentsReturned = []
#     # for ind, tracer in enumerate(SelectedTracers):
#     #     truthy = np.isin(Tracers,tracer)
#     #     if np.any(truthy) == True:
#     #         TracersIndices.append(np.where(truthy)[0])
#     #         TracersReturned.append(Tracers[np.where(truthy)])
#     #         ParentsReturned.append(Parents[np.where(truthy)])
#     #     else:
#     #         TracersIndices.append([np.nan])
#     #
#     # #If the tracer from SelectedTracers is in tracers, use the above indices to select its
#     # #   parent, and from there its cell, then grab data.
#     # #   If the tracer from SelectedTracers is not in tracers, put a nan value.
#     # #   This will allow plotting of all tracers, keeping saveData a fixed shape == subset/SelectedTracers
#     #
#     # saveData = []
#     # massData = []
#     # for (ind, element) in zip(TracersIndices,TracersTruthy):
#     #     if element == True:
#     #         parent = Parents[ind]
#     #         dataIndex = np.where(np.isin(CellIDs,parent))
#     #         dataEntry = Data[dataIndex].tolist()
#     #         if (np.shape(dataEntry)[0] == 0):
#     #             dataEntry = NullEntry
#     #         else:
#     #             dataEntry = dataEntry[0]
#     #         saveData.append(dataEntry)
#     #     else:
#     #         saveData.append(NullEntry)
#     #
#     # saveData = np.array(saveData).flatten()
#     # TracersReturned = np.array(TracersReturned).flatten()
#     # ParentsReturned = np.array(ParentsReturned).flatten()
#     #
#     # return saveData, TracersReturned, ParentsReturned
#     return SelectedData, SelectedTrids, SelectedPrids
# def get_individual_cell_from_tracer(
#         Tracers, Parents, CellIDs, SelectedTracers, Data, NullEntry=np.nan
# ):
#     if np.shape(np.shape(Data))[0] == 1:
#         dimension = 1
#         NullEntry = np.nan
#     elif ((np.shape(np.shape(Data))[0] == 2)&((np.shape(Data)[0]==3)|(np.shape(Data)[1]==3))):
#         dimension = 3
#         NullEntry = [np.nan for dd in range(dimension)]
#     else:
#         raise Exception(f"[@get_individual_cell_from_tracer]: dimension not 1 or 3! dataBlank Failure! Data neither 3D vector or 1D scalar!")
#     # Select which of the SelectedTracers are in Tracers from this snap
#     SelectedTrids = np.where(np.isin(SelectedTracers, Tracers), SelectedTracers, np.nan)
#
#     # whereTrids = np.where(np.isin(Tracers,SelectedTracers)==True)[0]
#     _, SelectedIndices, TridIndices = np.intersect1d(    SelectedTracers, Tracers, return_indices=True)
#
#     pridBlank = np.full(shape=np.shape(SelectedTracers), fill_value=-1)
#     pridBlank[SelectedIndices] = Parents[TridIndices]
#
#     SelectedPrids = pridBlank
#     parentsEqualsCellIDs = np.array_equal(Parents,CellIDs)
#     if parentsEqualsCellIDs == False:
#         if dimension != 1:
#             sig = f"(m),(n),(n,{dimension}),({dimension}),()->(m,{dimension})"
#         else:
#             sig = f"(m),(n),(n),(),()->(m)"
#         vectorSelect = np.vectorize(_select_data_from_selected_prids,signature=sig)
#
#         dataBlank = vectorSelect(SelectedPrids,CellIDs,Data,NullEntry=NullEntry,dimension=dimension)
#     else:
#         dataBlank = Data[np.where(np.isin(CellIDs,Parents))[0]]
#
#     SelectedData = dataBlank
#
#     assert np.shape(SelectedTrids) == np.shape(SelectedTracers)
#     assert np.shape(SelectedPrids) == np.shape(SelectedTracers)
#     assert np.shape(SelectedData)[0] == np.shape(SelectedTracers)[0]
#
#     return SelectedData, SelectedTrids, SelectedPrids

def get_individual_cell_from_tracer(
        Tracers, Parents, CellIDs, SelectedTracers, Data, NullEntry=np.nan
):
    """
        Function to go from Tracers, Parents, CellIDs, Data
        to selectedData (Len(Tracers)) with NullEntry of [np.nan] or
        [np.nan,np.nan,np.nan] (depending on Data shape) where the Tracer from
        Selected Tracers is not in the CellIDs.
        This should return a consistently ordered data set where we always Have
        the data in the order of SelectedTracers and NaN's where that tracer
        has been lost. This allows for a look over the individual tracer's
        behaviour over time.

        We use a FORTRAN90 numpy.f2py compiled script called where_duplicates
        in this function. This function accepts two 1D arrays, a & b. a ought
        to be of the same shape as SelectedTracers, and contain the Parent IDs
        (prids). b ought to be of shape Data and contain the CellIDs.

        The intention is that we return the intersect of a & b, WITH DUPLICATES.
        That is, if a value is in a multiple times, it should return the
        corresponding index and value of b for each of those instances of the
        matching value. This is similar to numpy.intersect1d, but we include
        duplicates. Hence the name, 'where_duplicates'.
    """
    # Import FORTRAN90 function
    from where_duplicates import where_duplicates

    # Set up a blank data array of shape and order SelectedTracers.
    # Fill this with the relevant sized entry of NaN as if selecting from
    # true data.
    #
    # E.G. Temperature is scaler => NullEntry == np.nan
    # E.G. Position is vector => NullEntry == [np.nan, np.nan, np.nan]
    if np.shape(np.shape(Data))[0] == 1:
        dimension = 1
        NullEntry = np.nan
        dataBlank = np.full(shape=np.shape(SelectedTracers), fill_value=NullEntry)
    elif ((np.shape(np.shape(Data))[0] == 2)&((np.shape(Data)[0]==3)|(np.shape(Data)[1]==3))):
        dimension = 3
        NullEntry = [np.nan for dd in range(dimension)]
        dataBlank = np.full(shape=(np.shape(SelectedTracers)[0],dimension), fill_value=NullEntry)
    else:
        raise Exception(f"[@get_individual_cell_from_tracer]: dimension not 1 or 3! dataBlank Failure! Data neither 3D vector or 1D scalar!")

    # Select which of the SelectedTracers are in Tracers from this snap
    SelectedTrids = np.where(np.isin(SelectedTracers, Tracers), SelectedTracers, np.nan)

    # Find the indices of Tracers included in SelectedTracers in this snap
    # in the order, and shape, of SelectedTracers
    _, SelectedIndices, TridIndices = np.intersect1d(    SelectedTracers, Tracers, return_indices=True)

    # Set up a blank set of Parent IDs. Then set the corresponding pridBlank
    # values (order SelectedTracers) to have the corresponding Parent ID from
    # Parents (order Tracers)
    pridBlank = np.full(shape=np.shape(SelectedTracers), fill_value=-1)
    pridBlank[SelectedIndices] = Parents[TridIndices]

    #Rename for clarity
    SelectedPrids = pridBlank

    # Use our FORTRAN90 function as described above to return the
    # selectedCellIDs (order SelectedTracers), with -1 where the Tracer's cell
    # was not found in this snap. Return the Index of the CellID
    # to selectedCellIDs match, -1 where Tracer's cell not in snap.
    # This will allow for selection of Data with duplicates by
    # Data[selectedDataIndices[np.where(selectedDataIndices!=-1.)[0]]
    selectedCellIDs, selectedDataIndices = where_duplicates(SelectedPrids,CellIDs)

    #Grab location of index of match of SelectedPrids with CellIDs.
    whereIndexData = np.where(selectedDataIndices!=-1.)[0]

    # Assign the non-blank data to the prepared NullEntry populated array
    # of shape SelectedTracers. Again, this step is designed to
    # copy duplicates of the data where a cell contains more than one tracer.
    dataBlank[whereIndexData] = Data[selectedDataIndices[whereIndexData]]

    #Rename for clarity
    SelectedData = dataBlank

    assert np.shape(SelectedTrids) == np.shape(SelectedTracers)
    assert np.shape(SelectedPrids) == np.shape(SelectedTracers)
    assert np.shape(SelectedData)[0] == np.shape(SelectedTracers)[0]

    return SelectedData, SelectedTrids, SelectedPrids

# ------------------------------------------------------------------------------#
# def _select_data_from_selected_prids(SelectedPrids,CellIDs,Data,NullEntry=np.nan,dimension=1):
#     SelectedData = np.array([Data[np.where(CellIDs==ID)[0]].tolist()[0] if ((ID != -1)&(ID in CellIDs)) else NullEntry for (ind, ID) in enumerate(SelectedPrids)])
#     if dimension != 1:
#         SelectedData = SelectedData.reshape(-1,dimension)
#     return SelectedData

# ------------------------------------------------------------------------------#


def get_individual_cell(CellIDs, SelectedCells, Data, NullEntry=np.nan):
    if np.shape(np.shape(Data))[0] == 1:
        dimension = 1
        NullEntry = np.nan
        dataBlank = np.full(shape=np.shape(SelectedTracers), fill_value=NullEntry)
    elif ((np.shape(np.shape(Data))[0] == 2)&((np.shape(Data)[0]==3)|(np.shape(Data)[1]==3))):
        dimension = 3
        NullEntry = [np.nan for dd in range(dimension)]
        dataBlank = np.full(shape=(np.shape(SelectedTracers)[0],dimension), fill_value=NullEntry)
    else:
        raise Exception(f"[@get_individual_cell_from_tracer]: dimension not 1 or 3! dataBlank Failure! Data neither 3D vector or 1D scalar!")

        # Select which of the SelectedTracers are in Tracers from this snap
    SelectedCellsReturned = np.where(np.isin(SelectedCells, CellIDs), SelectedCells, -1)
    #
    # for (ind, ID) in enumerate(SelectedCellsReturned):
    #     if ID != -1:
    #         value = np.where(np.isin(CellIDs, ID))[0]
    #         if np.shape(value)[0] > 0:
    #             dataBlank[ind] = Data[value]
    #
    # SelectedData = dataBlank
    # assert np.shape(SelectedData)[0] == np.shape(SelectedCells)[0]
    # assert np.shape(SelectedCellsReturned) == np.shape(SelectedCells)


    parentsEqualsCellIDs = np.array_equal(SelectedCells,CellIDs)
    if parentsEqualsCellIDs == False:
        selectedCellIDs, selectedDataIndices = where_duplicates(SelectedCells,CellIDs)

        whereIndexData = np.where(selectedDataIndices!=-1.)[0]

        dataBlank[whereIndexData] = Data[selectedDataIndices[whereIndexData]]

    else:
        dataBlank = Data[np.where(np.isin(CellIDs,Parents))[0]]

    SelectedData = dataBlank

    assert np.shape(SelectedData)[0] == np.shape(SelectedCells)[0]
    assert np.shape(SelectedCellsReturned) == np.shape(SelectedCells)

    return SelectedData, SelectedCellsReturned


# ------------------------------------------------------------------------------#


def hdf5_save(path, data):
    """
    Save nested dictionary as hdf5 file.
    Dictionary must have form:
        {(Meta-Key1 , Meta-Key2):{key1:... , key2: ...}}
    and will be saved in the form:
        {Meta-key1_Meta-key2:{key1:... , key2: ...}}
    """
    with h5py.File(path, "w") as f:
        for key, value in data.items():
            saveKey = None
            # Loop over Metakeys in tuple key of met-dictionary
            # Save this new metakey as one string, separated by '_'
            if isinstance(key, tuple) == True:
                for entry in key:
                    if saveKey is None:
                        saveKey = entry
                    else:
                        saveKey = saveKey + "-" + str(entry)
            else:
                saveKey = key
            # Create meta-dictionary entry with above saveKey
            #   Add to this dictionary entry a dictionary with keys from sub-dict
            #   and values from sub dict. Gzip for memory saving.
            grp = f.create_group(saveKey)
            for k, v in value.items():
                grp.create_dataset(k, data=v)

    return


# ------------------------------------------------------------------------------#
def hdf5_load(path):
    """
    Load nested dictionary from hdf5 file.
    Dictionary will be saved in the form:
        {Meta-key1_Meta-key2:{key1:... , key2: ...}}
    and output in the following form:
        {(Meta-Key1 , Meta-Key2):{key1:... , key2: ...}}

    """
    loaded = h5py.File(path, "r")

    dataDict = {}
    for key, value in loaded.items():

        loadKey = None
        for entry in key.split("-"):
            if loadKey is None:
                loadKey = entry
            else:
                loadKey = tuple(key.split("-"))
        # Take the sub-dict out from hdf5 format and save as new temporary dictionary
        tmpDict = {}
        for k, v in value.items():
            tmpDict.update({k: v.value})
        # Add the sub-dictionary to the meta-dictionary
        dataDict.update({loadKey: tmpDict})

    return dataDict


# ------------------------------------------------------------------------------#


def full_dict_hdf5_load(path, TRACERSPARAMS, FullDataPathSuffix):
    FullDict = {}
    for snap in range(
            int(TRACERSPARAMS["snapMin"]),
            min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"] + 1)),
            1,
    ):
        for targetT in TRACERSPARAMS["targetTLst"]:
            for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
                loadPath = (
                        path + f"_T{targetT}_{rin}R{rout}_{int(snap)}" + FullDataPathSuffix
                )
                data = hdf5_load(loadPath)
                FullDict.update(data)

    return FullDict


# ------------------------------------------------------------------------------#


def statistics_hdf5_load(targetT, rin, rout, path, TRACERSPARAMS, MiniDataPathSuffix):
    # Load data in {(T#, snap#):{k:v}} form
    nested = {}
    for snap in range(
            int(TRACERSPARAMS["snapMin"]),
            min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"] + 1)),
            1,
    ):
        # Temperature specific load path
        loadPath = (
                path
                + f"_T{targetT}_{rin}R{rout}_{int(snap)}_Statistics"
                + MiniDataPathSuffix
        )
        data = hdf5_load(loadPath)
        nested.update(data)

    # flatten data to {k:[v1,v2,v3...]} form
    plotData = {}
    for key, value in nested.items():
        for k, v in value.items():
            if k not in plotData.keys():
                plotData.update({k: v})
            else:
                plotData[k] = np.append(plotData[k], v)
    return plotData


# ------------------------------------------------------------------------------#


def pad_non_entries(snapGas, snapNumber):
    """
    Subroutine to pad all stars and gas entries in snapGas to have same first dimension size.
    So stars only data -> stars data + NGas x None
    So Gas only data -> Gas data + Nstars x None
    So all data first dimension == NTot

    Sanity checks and error messages in place.
    """

    print(f"[@{snapNumber}]: Padding None Entries!")

    NGas = len(snapGas.type[np.where(snapGas.type == 0)])
    NStars = len(snapGas.type[np.where(snapGas.type == 4)])
    NTot = len(snapGas.type)

    for key, value in snapGas.data.items():
        if value is not None:
            # If shape indicates 1D give 1D lists nx1
            # Else list will be 2D so give lists nx3
            if np.shape(np.shape(value))[0] == 1:
                if np.shape(value)[0] == NGas:
                    paddedValues = np.pad(
                        value, (0, NStars), "constant", constant_values=(np.nan)
                    )
                    snapGas.data[key] = paddedValues
                    if np.shape(paddedValues)[0] != NTot:
                        print(
                            "[@ GAS @pad_non_entries 1D:] Padded List not of length NTot. Data Does not have non-entries for STARS!"
                        )
                        print(f"Key: {key}")
                        print(f"shape: {np.shape(paddedValues)}")

                elif np.shape(value)[0] == NStars:
                    # Opposite addition order to maintain sensible ordering.
                    paddedValues = np.pad(
                        value, (NGas, 0), "constant", constant_values=(np.nan)
                    )
                    snapGas.data[key] = paddedValues
                    if np.shape(paddedValues)[0] != NTot:
                        print(
                            "[@ STARS @pad_non_entries 1D:] Padded List not of length NTot. Data Does not have non-entries for GAS!"
                        )
                        print(f"Key: {key}")
                        print(f"shape: {np.shape(paddedValues)}")

                elif np.shape(value)[0] != (NTot):
                    print(
                        "[@ ELSE @pad_non_entries 1D:] Warning! Rule Exception! Original Data does not have shape consistent with number of stars or number of gas as defined by NGas NStars!"
                    )
                    print(f"Key: {key}")
                    print(f"shape: {np.shape(value)}")
            else:
                if np.shape(value)[0] == NGas:
                    paddedValues = np.pad(
                        value,
                        ((0, NStars), (0, 0)),
                        "constant",
                        constant_values=(np.nan),
                    )
                    snapGas.data[key] = paddedValues
                    if np.shape(paddedValues)[0] != NTot:
                        print(
                            "[@ GAS @pad_non_entries 2D:] Padded List not of length NTot. Data Does not have non-entries for STARS!"
                        )
                        print(f"Key: {key}")
                        print(f"shape: {np.shape(paddedValues)}")

                elif np.shape(value)[0] == NStars:
                    # Opposite addition order to maintain sensible ordering.
                    paddedValues = np.pad(
                        value, ((NGas, 0), (0, 0)), "constant", constant_values=(np.nan)
                    )
                    snapGas.data[key] = paddedValues
                    if np.shape(paddedValues)[0] != NTot:
                        print(
                            "[@ STARS @pad_non_entries 2D:] Padded List not of length NTot. Data Does not have non-entries for GAS!"
                        )
                        print(f"Key: {key}")
                        print(f"shape: {np.shape(paddedValues)}")

                elif np.shape(value)[0] != NTot:
                    print(
                        "[@ ELSE @pad_non_entries 2D:] Warning! Rule Exception! Original Data does not have shape consistent with number of stars or number of gas as defined by NGas NStars!"
                    )
                    print(f"Key: {key}")
                    print(f"shape: {np.shape(value)}")

    return snapGas


# ------------------------------------------------------------------------------#


def save_statistics(
        Cells,
        targetT,
        rin,
        rout,
        snapNumber,
        TRACERSPARAMS,
        saveParams,
        DataSavepath=None,
        MiniDataPathSuffix=".csv",
        saveBool=True
):
    # ------------------------------------------------------------------------------#
    #       Flatten dict and take subset
    # ------------------------------------------------------------------------------#
    print("")
    print(f"[@{snapNumber} @{rin}R{rout} @T{targetT}]: Analysing Statistics!")

    statsData = {}

    for k, v in Cells.items():
        if k in saveParams:
            whereErrorKey = f"{k}"
            # For the data keys we wanted saving (saveParams), this is where we generate the data to match the
            #   combined keys in saveKeys.
            #       We are saving the key "{k}{percentile:2.2%}" in a new dict, statsData
            #           This effectively flattens and processes the data dict in one go
            #
            #   We have separate statements key not in keys and else.
            #       This is because if key does not exist yet in statsData, we want to create a new entry in statsData
            #           else we want to append to it, not create a new entry or overwrite the old one
            # whereGas = np.where(FullDict[key]['type'] == 0)
            for percentile in TRACERSPARAMS["percentiles"]:
                saveKey = f"{k}_{percentile:2.2f}%"
                stat = weighted_percentile(
                    data=v, weights=Cells["mass"], perc=percentile, key=whereErrorKey
                )
                if saveKey not in statsData.keys():
                    statsData.update({saveKey: stat})
                else:
                    statsData[saveKey] = np.append(statsData[saveKey], stat)
        # ------------------------------------------------------------------------------#
    #       Save stats as .csv files for a given temperature
    # ------------------------------------------------------------------------------#

    if saveBool == True:
        # Generate our savepath
        savePath = (
                DataSavepath
                + f"_T{targetT}_{rin}R{rout}_{int(snapNumber)}_Statistics"
                + MiniDataPathSuffix
        )
        print(
            "\n"
            + f"[@{snapNumber} @{rin}R{rout} @T{targetT}]: Saving Statistics as: "
            + savePath
        )

        out = {(f"T{targetT}", f"{rin}R{rout}", f"{int(snapNumber)}"): statsData}

        hdf5_save(savePath, out)

    return statsData


# ------------------------------------------------------------------------------#


def flatten_wrt_T(dataDict, snapRange, TRACERSPARAMS, rin, rout):
    flattened_dict = {}
    for snap in snapRange:
        tmp = {}
        newkey = (f"{rin}R{rout}", f"{int(snap)}")
        for T in TRACERSPARAMS["targetTLst"]:
            key = (f"T{T}", f"{rin}R{rout}", f"{int(snap)}")

            for k, v in dataDict[key].items():
                if k in tmp.keys():
                    tmp[k] = np.append(tmp[k], v)
                else:
                    tmp.update({k: v})

        flattened_dict.update({newkey: tmp})

    return flattened_dict


# ------------------------------------------------------------------------------#
def flatten_wrt_time_speed_test(
        targetT,
        dataDict,
        rin,
        rout,
        TRACERSPARAMS,
        saveParams,
        snapRange,
        DataSavepath,
        DataSavepathSuffix,
        saveBool = True
):
    import time

    flattened_dict = {}


    tmp = {}
    newkey = (f"T{targetT}", f"{rin}R{rout}")
    key = (f"T{targetT}", f"{rin}R{rout}", f"{int(TRACERSPARAMS['selectSnap'])}")
    print(f"Starting {newkey} analysis!")
    TracerOrder = dataDict[key]["trid"]
    snap = snapRange[0]
    print(f"T{targetT} {rin}R{rout} Snap {snap}!")
    key = (f"T{targetT}", f"{rin}R{rout}", f"{int(snap)}")
    for k, v in dataDict[key].items():
        if k in saveParams:
            print('Start V1')

            start = time.time()
            (
                tracerData,
                TracersReturned,
                ParentsReturned,
            ) = get_individual_cell_from_tracer(
                Tracers=dataDict[key]["trid"][:50000],
                Parents=dataDict[key]["prid"][:50000],
                CellIDs=dataDict[key]["id"],
                SelectedTracers=TracerOrder[:50000],
                Data=dataDict[key][k],
            )
            stop = time.time()
            print(f'Elapsed = {stop-start} s')


            print('Start V2 - F90')

            start = time.time()
            (
                tracerData,
                TracersReturned,
                ParentsReturned,
            ) = get_individual_cell_from_tracer_v2(
                Tracers=dataDict[key]["trid"][:50000],
                Parents=dataDict[key]["prid"][:50000],
                CellIDs=dataDict[key]["id"],
                SelectedTracers=TracerOrder[:50000],
                Data=dataDict[key][k],
            )
            stop = time.time()
            print(f'Elapsed = {stop-start} s')

            if k == "trid":
                tracerData = TracersReturned
            elif k == "prid":
                tracerData = ParentsReturned

            if k in tmp.keys():
                entry = tmp[k]
                entry.append(tracerData)
                tmp.update({k: entry})
            else:
                tmp.update({k: [tracerData]})
                    # print(f"k : {k} --> type(tracerData) : {type(tracerData)} --> np.shape(tracerData) : {np.shape(tracerData)} --> type(tmp[k]) {type(tmp[k])} --> np.shape(tmp[k]) {np.shape(tmp[k])}")

    # print("To array!")
    for k, v in tmp.items():
        tmp.update({k: np.array(v)})
        # print(f"k : {k} --> type(np.array(v)) : {type(np.array(v))} --> type(tmp[k]) {type(tmp[k])}")

    flattened_dict.update({newkey: tmp})

    # final_dict = {}
    #
    # for key,dict in flattened_dict.items():
    #     tmp = delete_nan_inf_axis(dict,axis=0)
    #     final_dict.update({key : tmp})

    savePath = (
            DataSavepath + f"_T{targetT}_{rin}R{rout}_flat-wrt-time" + DataSavepathSuffix
    )

    if saveBool == True:
            print("\n" + f": Saving flat data as: " + savePath)

            hdf5_save(savePath, flattened_dict)

            return None
    else:
        return flattened_dict

def flatten_wrt_time(
        targetT,
        dataDict,
        rin,
        rout,
        TRACERSPARAMS,
        saveParams,
        snapRange,
        DataSavepath,
        DataSavepathSuffix,
        saveBool = True
):
    flattened_dict = {}
    snapRange = [
        xx
        for xx in range(
            int(TRACERSPARAMS["snapMin"]),
            min(int(TRACERSPARAMS["snapMax"]) + 1, int(TRACERSPARAMS["finalSnap"]) + 1),
            1,
        )
    ]
    tmp = {}

    tmp = {}
    newkey = (f"T{targetT}", f"{rin}R{rout}")
    key = (f"T{targetT}", f"{rin}R{rout}", f"{int(TRACERSPARAMS['selectSnap'])}")
    print(f"Starting {newkey} analysis!")
    TracerOrder = dataDict[key]["trid"]
    for snap in snapRange:
        print(f"T{targetT} {rin}R{rout} Snap {snap}!")
        key = (f"T{targetT}", f"{rin}R{rout}", f"{int(snap)}")
        for k, v in dataDict[key].items():
            if k in saveParams:
                (
                    tracerData,
                    TracersReturned,
                    ParentsReturned,
                ) = get_individual_cell_from_tracer(
                    Tracers=dataDict[key]["trid"],
                    Parents=dataDict[key]["prid"],
                    CellIDs=dataDict[key]["id"],
                    SelectedTracers=TracerOrder,
                    Data=dataDict[key][k],
                )

                if k == "trid":
                    tracerData = TracersReturned
                elif k == "prid":
                    tracerData = ParentsReturned
                if k in tmp.keys():
                    entry = tmp[k]
                    entry.append(tracerData)
                    tmp.update({k: entry})
                else:
                    tmp.update({k: [tracerData]})
                # print(f"k : {k} --> type(tracerData) : {type(tracerData)} --> np.shape(tracerData) : {np.shape(tracerData)} --> type(tmp[k]) {type(tmp[k])} --> np.shape(tmp[k]) {np.shape(tmp[k])}")


    for k, v in tmp.items():
        tmp.update({k: np.array(v)})
        # print(f"k : {k} --> type(np.array(v)) : {type(np.array(v))} -->
    flattened_dict.update({newkey: tmp})

    # final_dict = {}
    #
    # for key,dict in flattened_dict.items():
    #     tmp = delete_nan_inf_axis(dict,axis=0)
    #     final_dict.update({key : tmp})

    savePath = (
            DataSavepath + f"_T{targetT}_{rin}R{rout}_flat-wrt-time" + DataSavepathSuffix
            )
    print('Flat Keys = ', flattened_dict.keys())
    if saveBool == True:
            print("\n" + f": Saving flat data as: " + savePath)

            hdf5_save(savePath, flattened_dict)

            return None
    else:
        return flattened_dict

# ------------------------------------------------------------------------------#

def multi_halo_flatten_wrt_time_speed_test(dataDict,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst,DataSavepath,DataSavepathSuffix = f".h5",TracersParamsPath= "TracersParams.csv",TracersMasterParamsPath="TracersParamsMaster.csv",SelectedHaloesPath = "TracersSelectedHaloes.csv"):


    singleValueParams = ["Lookback", "Ntracers", "Snap"]

    # Number of cores to run on:

    flattenParams = saveParams.copy()
    flattenParams += TRACERSPARAMS["saveTracersOnly"] + TRACERSPARAMS["saveEssentials"]
    for param in singleValueParams:
        flattenParams.remove(param)

    flattenedDict = {}
    print("Flattening wrt time!")
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")

        for targetT in Tlst:
            key = (f'T{targetT}',f'{rin}R{rout}')
            print(key)
            #Disable saving of dict and return flattened to parent process

            out = flatten_wrt_time_speed_test(targetT,dataDict,
            rin,
            rout,
            TRACERSPARAMS,
            flattenParams,
            snapRange,
            DataSavepath,
            DataSavepathSuffix,
            saveBool = False)
            flattenedDict.update({key:out})

    print("Done! End of Flattening wrt Time Post-Processing :)")
    return flattenedDict


def multi_halo_flatten_wrt_time(dataDict,TRACERSPARAMS,saveParams,tlookback,snapRange,Tlst,DataSavepath,loadParams=None,DataSavepathSuffix = f".h5",TracersParamsPath = "TracersParams.csv",TracersMasterParamsPath ="TracersParamsMaster.csv",SelectedHaloesPath = "TracersSelectedHaloes.csv"):


    singleValueParams = ["Lookback", "Ntracers", "Snap"]

    # Number of cores to run on:

    flattenParams = saveParams.copy()
    flattenParams += TRACERSPARAMS["saveTracersOnly"] + TRACERSPARAMS["saveEssentials"]
    for param in singleValueParams:
        flattenParams.remove(param)

    flattenedDict = {}
    print("Flattening wrt time!")
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        print(f"{rin}R{rout}")

        for targetT in Tlst:
            key = (f'T{targetT}',f'{rin}R{rout}')
            print(key)
            #Disable saving of dict and return flattened to parent process

            out = flatten_wrt_time(targetT,dataDict,
            rin,
            rout,
            TRACERSPARAMS,
            flattenParams,
            snapRange,
            DataSavepath,
            DataSavepathSuffix,
            saveBool = False)
            flattenedDict.update({key:out})

    print("Done! End of Flattening wrt Time Post-Processing :)")
    return flattenedDict


#------------------------------------------------------------------------------#


def delete_nan_inf_axis(dict, axis=0):
    """
    Delete any column of dict with entry NaN or Inf in row (axis).
    """

    new_dict = {}
    where_dict = {}
    for key, value in dict.items():
        if value is not None:
            if axis == 0:
                whereEntry = ~np.isnan(value).any(axis=0) & ~np.isinf(value).any(axis=0)
                value = np.array(value)
                data = value[:, whereEntry]
            elif axis == 1:
                whereEntry = ~np.isnan(value).any(axis=1) & ~np.isinf(value).any(axis=1)
                value = np.array(value)
                data = value[whereEntry]
            else:
                print(
                    "[@delete_nan_inf_axis]: Greater than 2D dimensions of data in dict. Check logic!"
                )
                assert True == False
            new_dict.update({key: data})
            where_dict.update({key: whereEntry})

    return new_dict, where_dict


# ------------------------------------------------------------------------------#
def plot_projections(
        snapGas,
        snapNumber,
        targetT,
        rin,
        rout,
        TRACERSPARAMS,
        DataSavepath,
        FullDataPathSuffix,
        Axes=[0, 1],
        zAxis=[2],
        boxsize=400.0,
        boxlos=20.0,
        pixres=0.2,
        pixreslos=4,
        DPI=100,
        CMAP=None,
        numThreads=2,
):
    print(
        f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Starting Projections Video Plots!"
    )

    if CMAP == None:
        cmap = plt.get_cmap("inferno")
    else:
        cmap = CMAP

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["x", "y", "z"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]

    # --------------------------#
    ## Slices and Projections ##
    # --------------------------#
    print(f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Slices and Projections!")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # slice_nH    = snap.get_Aslice("n_H", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    #
    # slice_B   = snap.get_Aslice("B", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    nprojections = 5

    print(
        "\n"
        + f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Projection 1 of {nprojections}"
    )

    proj_T = snapGas.get_Aslice(
        "Tdens",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    print(
        "\n"
        + f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Projection 2 of {nprojections}"
    )

    proj_dens = snapGas.get_Aslice(
        "rho_rhomean",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    print(
        "\n"
        + f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Projection 3 of {nprojections}"
    )

    proj_nH = snapGas.get_Aslice(
        "n_H",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    print(
        "\n"
        + f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Projection 4 of {nprojections}"
    )

    proj_B = snapGas.get_Aslice(
        "B",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    print(
        "\n"
        + f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Projection 5 of {nprojections}"
    )

    proj_gz = snapGas.get_Aslice(
        "gz",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numThreads,
    )

    # ------------------------------------------------------------------------------#
    # PLOTTING TIME
    # Set plot figure sizes
    xsize = 10.0
    ysize = 10.0
    ageUniverse = 13.77
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    # Redshift
    redshift = snapGas.redshift  # z
    aConst = 1.0 / (1.0 + redshift)  # [/]

    # [0] to remove from numpy array for purposes of plot title
    tlookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
        0
    ]  # [Gyrs]
    # ==============================================================================#
    #
    #           Quad Plot for standard video
    #
    # ==============================================================================#
    print(f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Quad Plot...")

    fullTicks = [xx for xx in np.linspace(-1.0 * halfbox, halfbox, 9)]
    fudgeTicks = fullTicks[1:]

    aspect = "equal"
    fontsize = 12
    fontsizeTitle = 18

    # DPI Controlled by user as lower res needed for videos #
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
    )

    # Add overall figure plot
    TITLE = (
            r"Redshift $(z) =$"
            + f"{redshift:0.03f} "
            + " "
            + r"$t_{Lookback Time}=$"
            + f"{tlookback :0.03f} Gyrs"
            + "\n"
            + f"Projections within {-1. * float(boxlos) / 2.}"
            + r"<"
            + f"{AxesLabels[zAxis[0]]}-axis"
            + r"<"
            + f"{float(boxlos) / 2.} kpc"
    )
    fig.suptitle(TITLE, fontsize=fontsizeTitle)

    # cmap = plt.get_cmap(CMAP)
    cmap.set_bad(color="grey")

    # -----------#
    # Plot Temperature #
    # -----------#
    # print("pcm1")
    ax1 = axes[0, 0]

    pcm1 = ax1.pcolormesh(
        proj_T["x"],
        proj_T["y"],
        np.transpose(proj_T["grid"] / proj_dens["grid"]),
        vmin=1e4,
        vmax=1e7,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax1.set_title(f"Temperature Projection", fontsize=fontsize)
    cax1 = inset_axes(ax1, width="5%", height="95%", loc="right")
    fig.colorbar(pcm1, cax=cax1, orientation="vertical").set_label(
        label=r"$T$ [$K$]", size=fontsize, weight="bold"
    )
    cax1.yaxis.set_ticks_position("left")
    cax1.yaxis.set_label_position("left")
    cax1.yaxis.label.set_color("white")
    cax1.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax1.set_ylabel(f"{AxesLabels[Axes[1]]} (kpc)", fontsize=fontsize)
    # ax1.set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize = fontsize)
    # ax1.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax1)
    plt.xticks(fullTicks)
    plt.yticks(fudgeTicks)

    # -----------#
    # Plot n_H Projection #
    # -----------#
    # print("pcm2")
    ax2 = axes[0, 1]

    pcm2 = ax2.pcolormesh(
        proj_nH["x"],
        proj_nH["y"],
        np.transpose(proj_nH["grid"]) / int(boxlos / pixreslos),
        vmin=1e-6,
        vmax=1e-1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax2.set_title(r"Hydrogen Number Density Projection", fontsize=fontsize)

    cax2 = inset_axes(ax2, width="5%", height="95%", loc="right")
    fig.colorbar(pcm2, cax=cax2, orientation="vertical").set_label(
        label=r"$n_H$ [$cm^{-3}$]", size=fontsize, weight="bold"
    )
    cax2.yaxis.set_ticks_position("left")
    cax2.yaxis.set_label_position("left")
    cax2.yaxis.label.set_color("white")
    cax2.tick_params(axis="y", colors="white", labelsize=fontsize)
    # ax2.set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize=fontsize)
    # ax2.set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize=fontsize)
    # ax2.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax2)
    plt.xticks(fullTicks)
    plt.yticks(fullTicks)

    # -----------#
    # Plot Metallicity #
    # -----------#
    # print("pcm3")
    ax3 = axes[1, 0]

    pcm3 = ax3.pcolormesh(
        proj_gz["x"],
        proj_gz["y"],
        np.transpose(proj_gz["grid"]) / int(boxlos / pixreslos),
        vmin=1e-2,
        vmax=1e1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax3.set_title(f"Metallicity Projection", y=-0.2, fontsize=fontsize)

    cax3 = inset_axes(ax3, width="5%", height="95%", loc="right")
    fig.colorbar(pcm3, cax=cax3, orientation="vertical").set_label(
        label=r"$Z/Z_{\odot}$", size=fontsize, weight="bold"
    )
    cax3.yaxis.set_ticks_position("left")
    cax3.yaxis.set_label_position("left")
    cax3.yaxis.label.set_color("white")
    cax3.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax3.set_ylabel(f"{AxesLabels[Axes[1]]} (kpc)", fontsize=fontsize)
    ax3.set_xlabel(f"{AxesLabels[Axes[0]]} (kpc)", fontsize=fontsize)

    # ax3.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax3)
    plt.xticks(fullTicks)
    plt.yticks(fullTicks)

    # -----------#
    # Plot Magnetic Field Projection #
    # -----------#
    # print("pcm4")
    ax4 = axes[1, 1]

    pcm4 = ax4.pcolormesh(
        proj_B["x"],
        proj_B["y"],
        np.transpose(proj_B["grid"]) / int(boxlos / pixreslos),
        vmin=1e-3,
        vmax=1e1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax4.set_title(r"Magnetic Field Strength Projection", y=-0.2, fontsize=fontsize)

    cax4 = inset_axes(ax4, width="5%", height="95%", loc="right")
    fig.colorbar(pcm4, cax=cax4, orientation="vertical").set_label(
        label=r"$B$ [$\mu G$]", size=fontsize, weight="bold"
    )
    cax4.yaxis.set_ticks_position("left")
    cax4.yaxis.set_label_position("left")
    cax4.yaxis.label.set_color("white")
    cax4.tick_params(axis="y", colors="white", labelsize=fontsize)

    # ax4.set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize=fontsize)
    ax4.set_xlabel(f"{AxesLabels[Axes[0]]} (kpc)", fontsize=fontsize)
    # ax4.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax4)
    plt.xticks(fudgeTicks)
    plt.yticks(fullTicks)

    # print("snapnum")
    # Pad snapnum with zeroes to enable easier video making
    fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.90)
    # fig.tight_layout()

    SaveSnapNumber = str(snapNumber).zfill(4)
    savePath = DataSavepath + f"_Quad_Plot_{int(SaveSnapNumber)}.png"

    print(f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Save {savePath}")
    plt.savefig(savePath, transparent=False)
    plt.close()

    print(f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: ...done!")

    return


# ------------------------------------------------------------------------------#

# ------------------------------------------------------------------------------#
def tracer_plot(
        Cells,
        tridDict,
        TRACERSPARAMS,
        DataSavepath,
        FullDataPathSuffix,
        Axes=[0, 1],
        zAxis=[2],
        boxsize=400.0,
        boxlos=20.0,
        pixres=0.2,
        pixreslos=4,
        DPI=100,
        CMAP=None,
        numThreads=4,
        MaxSubset=100,
        lazyLoadBool=True,
):
    if CMAP == None:
        cmap = plt.get_cmap("inferno")
    else:
        cmap = CMAP

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["x", "y", "z"]

    # Set plot figure sizes
    xsize = 10.0
    ysize = 10.0
    ageUniverse = 13.77
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    nullEntry = [np.nan, np.nan, np.nan]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]

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

    HaloID = int(TRACERSPARAMS["haloID"])

    for snapNumber in range(
            int(TRACERSPARAMS["snapMin"]),
            min(int(TRACERSPARAMS["snapMax"] + 1), int(TRACERSPARAMS["finalSnap"] + 1)),
    ):
        # --------------------------#
        ## Slices and Projections ##
        # --------------------------#
        print(f"[@{int(snapNumber)}]: Load snap!")

        # load in the subfind group files
        snap_subfind = load_subfind(snapNumber, dir=TRACERSPARAMS["simfile"])

        # load in the gas particles mass and position only for HaloID 0.
        #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
        snap = gadget_readsnap(
            snapNumber,
            TRACERSPARAMS["simfile"],
            hdf5=True,
            loadonlytype=[0, 1, 4],
            lazy_load=lazyLoadBool,
            subfind=snap_subfind,
        )
        snapGas = gadget_readsnap(
            snapNumber,
            TRACERSPARAMS["simfile"],
            hdf5=True,
            loadonlytype=[0, 4],
            lazy_load=lazyLoadBool,
            subfind=snap_subfind,
        )

        # load tracers data
        snapTracers = gadget_readsnap(
            snapNumber,
            TRACERSPARAMS["simfile"],
            hdf5=True,
            loadonlytype=[6],
            lazy_load=lazyLoadBool,
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

        tmp = snapGas.data["id"]
        tmp = snapGas.data["age"]
        tmp = snapGas.data["hrgm"]
        tmp = snapGas.data["mass"]
        tmp = snapGas.data["pos"]
        tmp = snapGas.data["vol"]
        del tmp

        print(
            f"[@{int(snapNumber)}]: SnapShot loaded at RedShift z={snap.redshift:0.05e}"
        )

        # Centre the simulation on HaloID 0
        snap = set_centre(
            snap=snap, snap_subfind=snap_subfind, HaloID=HaloID, snapNumber=snapNumber
        )
        snapGas = set_centre(
            snap=snapGas,
            snap_subfind=snap_subfind,
            HaloID=HaloID,
            snapNumber=snapNumber,
        )

        # --------------------------#
        ##    Units Conversion    ##
        # --------------------------#

        # Convert Units
        ## Make this a seperate function at some point??
        snap.pos *= 1e3  # [kpc]
        snap.vol *= 1e9  # [kpc^3]
        snap.mass *= 1e10  # [Msol]
        snap.hrgm *= 1e10  # [Msol]

        snapGas.pos *= 1e3  # [kpc]
        snapGas.vol *= 1e9  # [kpc^3]
        snapGas.mass *= 1e10  # [Msol]
        snapGas.hrgm *= 1e10  # [Msol]

        # Calculate New Parameters and Load into memory others we want to track
        snapGas = calculate_tracked_parameters(
            snapGas,
            snap,
            elements,
            elements_Z,
            elements_mass,
            elements_solar,
            Zsolar,
            omegabaryon0,
            snapNumber,
        )

        # Pad stars and gas data with Nones so that all keys have values of same first dimension shape
        snapGas = pad_non_entries(snapGas, snapNumber)

        # Redshift
        redshift = snapGas.redshift  # z
        aConst = 1.0 / (1.0 + redshift)  # [/]

        # [0] to remove from numpy array for purposes of plot title
        tlookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
            0
        ]  # [Gyrs]
        print(f"[@{int(snapNumber)}]: Slices and Projections!")
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        # slice_nH    = snap.get_Aslice("n_H", box = [boxsize,boxsize],\
        #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
        #  axes = Axes, proj = False, numthreads=16)
        #
        # slice_B   = snap.get_Aslice("B", box = [boxsize,boxsize],\
        #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
        #  axes = Axes, proj = False, numthreads=16)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
        nprojections = 2

        print("\n" + f"[@{int(snapNumber)}]: Projection 1 of {nprojections}")

        proj_T = snapGas.get_Aslice(
            "Tdens",
            box=[boxsize, boxsize],
            center=imgcent,
            nx=int(boxsize / pixres),
            ny=int(boxsize / pixres),
            nz=int(boxlos / pixreslos),
            boxz=boxlos,
            axes=Axes,
            proj=True,
            numthreads=numThreads,
        )

        print("\n" + f"[@{int(snapNumber)}]: Projection 2 of {nprojections}")

        proj_dens = snapGas.get_Aslice(
            "rho_rhomean",
            box=[boxsize, boxsize],
            center=imgcent,
            nx=int(boxsize / pixres),
            ny=int(boxsize / pixres),
            nz=int(boxlos / pixreslos),
            boxz=boxlos,
            axes=Axes,
            proj=True,
            numthreads=numThreads,
        )

        # ==============================================================================#
        #
        #           Grab positions of Tracer Subset
        #
        # ==============================================================================#
        for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
            print(f"{rin}R{rout}")
            for targetT in TRACERSPARAMS["targetTLst"]:
                print("")
                print(f"Starting T{targetT} {rin}R{rout} analysis")

                selectkey = (
                    f"T{targetT}",
                    f"{rin}R{rout}",
                    f"{int(TRACERSPARAMS['selectSnap'])}",
                )

                whereGas = np.where(Cells[selectkey]["type"] == 0)[0]
                subset = min(len(tridDict[selectkey]), MaxSubset)
                SelectedTracers1 = random.sample(tridDict[selectkey].tolist(), subset)
                SelectedTracers1 = np.array(SelectedTracers1)

                print(
                    "\n"
                    + f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Selecting {int(subset)} subset of Tracers Positions..."
                )
                # Select new subset for first snap
                #   Use old subset for all others
                key = (f"T{targetT}", f"{rin}R{rout}", f"{int(snapNumber)}")

                whereGas = np.where(Cells[key]["type"] == 0)[0]

                posData, _, _ = get_individual_cell_from_tracer(
                    Tracers=Cells[key]["trid"],
                    Parents=Cells[key]["prid"],
                    CellIDs=Cells[key]["id"][whereGas],
                    SelectedTracers=SelectedTracers1,
                    Data=Cells[key]["pos"][whereGas],
                    NullEntry=nullEntry,
                )

                posData = np.array(posData)

                # ------------------------------------------------------------------------------#
                # PLOTTING TIME

                print(f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Tracer Plot...")

                # load in the subfind group files
                tmpsnap_subfind = load_subfind(
                    int(TRACERSPARAMS["selectSnap"]), dir=TRACERSPARAMS["simfile"]
                )

                # load in the gas particles mass and position only for HaloID 0.
                #   0 is gas, 1 is DM, 4 is stars, 5 is BHs, 6 is tracers
                tmpsnapGas = gadget_readsnap(
                    int(TRACERSPARAMS["selectSnap"]),
                    TRACERSPARAMS["simfile"],
                    hdf5=True,
                    loadonlytype=[0],
                    lazy_load=True,
                    subfind=tmpsnap_subfind,
                )

                # Redshift
                tmpredshift = tmpsnapGas.redshift  # z
                aConst = 1.0 / (1.0 + tmpredshift)  # [/]

                # [0] to remove from numpy array for purposes of plot title
                selectlookback = tmpsnapGas.cosmology_get_lookback_time_from_a(
                    np.array([aConst])
                )[
                    0
                ]  # [Gyrs]

                print(
                    f"[@T{targetT}] @{rin}R{rout} @{int(snapNumber)} : SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
                )

                aspect = "equal"
                fontsize = 12
                fontsizeTitle = 16

                print(
                    f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Loading Old Tracer Subset Data..."
                )

                nOldSnaps = int(snapNumber) - int(TRACERSPARAMS["snapMin"])

                OldPosDict = {}
                for snap in range(int(TRACERSPARAMS["snapMin"]), int(snapNumber)):
                    key = (f"T{targetT}", f"{rin}R{rout}", f"{int(snap)}")

                    whereGas = np.where(Cells[key]["type"] == 0)[0]

                    posData, _, _ = get_individual_cell_from_tracer(
                        Tracers=Cells[key]["trid"],
                        Parents=Cells[key]["prid"],
                        CellIDs=Cells[key]["id"][whereGas],
                        SelectedTracers=SelectedTracers1,
                        Data=Cells[key]["pos"][whereGas],
                        NullEntry=nullEntry,
                    )

                    data = {key: posData}
                    OldPosDict.update(data)

                # NullEntry= [np.nan,np.nan,np.nan]
                # tmpOldPosDict= {}
                # for key, dict in OldPosDict.items():
                #     tmp = {}
                #     for k, v in dict.items():
                #         if (k=="pos"):
                #             vOutOfRange = np.where((v[:,zAxis[0]]>(float(boxlos)/2.))&(v[:,zAxis[0]]<(-1.*float(boxlos)/2.)))
                #             v[vOutOfRange] = NullEntry
                #
                #         tmp.update({k : v})
                #
                #     tmpOldPosDict.update({key : tmp})
                #
                # OldPosDict = tmpOldPosDict
                print(
                    f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: ...finished Loading Old Tracer Subset Data!"
                )

                print(f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Plot...")
                # DPI Controlled by user as lower res needed for videos #
                fig, axes = plt.subplots(
                    nrows=1, ncols=1, figsize=(xsize, ysize), dpi=DPI
                )

                # Add overall figure plot
                TITLE = (
                        r"Redshift $(z) =$"
                        + f"{redshift:0.03f} "
                        + " "
                        + r"$t_{Lookback Time}=$"
                        + f"{tlookback :0.03f} Gyrs"
                        + "\n"
                        + f"Projections within {-1. * float(boxlos) / 2.}"
                        + r"<"
                        + f"{AxesLabels[zAxis[0]]}-axis"
                        + r"<"
                        + f"{float(boxlos) / 2.} kpc"
                        + "\n"
                        + f"Subset of {int(subset)} Tracers selected at "
                        + r"$t_{Lookback Time}=$"
                        + f"{selectlookback :0.03f} Gyrs"
                        + " as being at "
                        + "\n"
                        + r"$T = 10^{%5.2f \pm %5.2f} K$"
                        % (targetT, TRACERSPARAMS["deltaT"])
                        + r" and $ %5.2f < R < %5.2f Kpc$" % (rin, rout)
                )

                fig.suptitle(TITLE, fontsize=fontsizeTitle)

                # cmap = plt.get_cmap(CMAP)
                cmap.set_bad(color="grey")

                # -----------#
                # Plot Temperature #
                # -----------#

                ###
                #   Select 10% of subset to have a colour, the rest to be white
                ###
                # cmapTracers = matplotlib.cm.get_cmap("nipy_spectral")
                # colourTracers = []
                # cwhite = (1.,1.,1.,1.)
                # cblack = (0.,0.,0.,1.)
                # for ii in range(0,int(subset+1)):
                #     if (ii % 5 == 0):
                #         colour = cblack
                #     else:
                #         colour = cwhite
                #     colourTracers.append(colour)
                #
                # colourTracers = np.array(colourTracers)

                colour = "white"

                ax1 = axes

                pcm1 = ax1.pcolormesh(
                    proj_T["x"],
                    proj_T["y"],
                    np.transpose(proj_T["grid"] / proj_dens["grid"]),
                    vmin=1e4,
                    vmax=1e7,
                    norm=matplotlib.colors.LogNorm(),
                    cmap=cmap,
                    rasterized=True,
                )

                sizeMultiply = 25
                sizeConst = 10

                whereInRange = np.where(
                    (posData[:, zAxis[0]] <= (float(boxlos) / 2.0))
                    & (posData[:, zAxis[0]] >= (-1.0 * float(boxlos) / 2.0))
                )
                posDataInRange = posData[whereInRange]
                # colourInRange = colourTracers[whereInRange]
                #
                # colourTracers = colourTracers.tolist()
                # colourInRange = colourInRange.tolist()

                sizeData = np.array(
                    [
                        (sizeMultiply * (xx + (float(boxlos) / 2.0)) / float(boxlos))
                        + sizeConst
                        for xx in posDataInRange[:, zAxis[0]]
                    ]
                )

                ax1.scatter(
                    posDataInRange[:, Axes[0]],
                    posDataInRange[:, Axes[1]],
                    s=sizeData,
                    c=colour,
                    marker="o",
                )  # colourInRange,marker='o')

                if int(snapNumber) == int(TRACERSPARAMS["selectSnap"]):
                    innerCircle = matplotlib.patches.Circle(
                        xy=(0, 0),
                        radius=float(rin),
                        facecolor="none",
                        edgecolor="green",
                        linewidth=5,
                        linestyle="-.",
                    )
                    outerCircle = matplotlib.patches.Circle(
                        xy=(0, 0),
                        radius=float(rout),
                        facecolor="none",
                        edgecolor="green",
                        linewidth=5,
                        linestyle="-.",
                    )
                    ax1.add_patch(innerCircle)
                    ax1.add_patch(outerCircle)

                minSnap = int(snapNumber) - min(int(nOldSnaps), 3)

                print(f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Plot Tails...")
                jj = 1
                for snap in range(int(minSnap + 1), snapNumber + 1):
                    key1 = (f"T{targetT}", f"{rin}R{rout}", f"{int(snap - 1)}")
                    key2 = (f"T{targetT}", f"{rin}R{rout}", f"{int(snap)}")
                    if snap != int(snapNumber):
                        pos1 = OldPosDict[key1]
                        pos2 = OldPosDict[key2]
                    else:
                        pos1 = OldPosDict[key1]
                        pos2 = posData

                    whereInRange = np.where(
                        (pos1[:, zAxis[0]] <= (float(boxlos) / 2.0))
                        & (pos1[:, zAxis[0]] >= (-1.0 * float(boxlos) / 2.0))
                        & (pos2[:, zAxis[0]] <= (float(boxlos) / 2.0))
                        & (pos2[:, zAxis[0]] >= (-1.0 * float(boxlos) / 2.0))
                    )

                    pathData = np.array([pos1[whereInRange], pos2[whereInRange]])
                    ntails = np.shape(pos1[whereInRange])[0]
                    alph = float(jj) / float(max(1, min(int(nOldSnaps), 3)) + 1.0)
                    jj += 1

                    for ii in range(0, int(ntails)):
                        ax1.plot(
                            pathData[:, ii, Axes[0]],
                            pathData[:, ii, Axes[1]],
                            c=colour,
                            alpha=alph,
                        )  # colourTracers[ii],alpha=alph)

                print(
                    f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: ...finished Plot Tails!"
                )

                xmin = np.nanmin(proj_T["x"])
                xmax = np.nanmax(proj_T["x"])
                ymin = np.nanmin(proj_T["y"])
                ymax = np.nanmax(proj_T["y"])

                ax1.set_ylim(ymin=ymin, ymax=ymax)
                ax1.set_xlim(xmin=xmin, xmax=xmax)

                ax1.set_title(f"Temperature Projection", fontsize=fontsize)
                cax1 = inset_axes(ax1, width="5%", height="95%", loc="right")
                fig.colorbar(pcm1, cax=cax1, orientation="vertical").set_label(
                    label=r"$T$ [$K$]", size=fontsize, weight="bold"
                )
                cax1.yaxis.set_ticks_position("left")
                cax1.yaxis.set_label_position("left")
                cax1.yaxis.label.set_color("white")
                cax1.tick_params(axis="y", colors="white", labelsize=fontsize)

                ax1.set_ylabel(f"{AxesLabels[Axes[1]]} (kpc)", fontsize=fontsize)
                ax1.set_xlabel(f"{AxesLabels[Axes[0]]} (kpc)", fontsize=fontsize)
                ax1.set_aspect(aspect)

                # Pad snapnum with zeroes to enable easier video making
                fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.80)
                # fig.tight_layout()

                SaveSnapNumber = str(snapNumber).zfill(4)
                savePath = (
                        DataSavepath
                        + f"_T{targetT}_{rin}R{rout}_Tracer_Subset_Plot_{int(SaveSnapNumber)}.png"
                )

                print(
                    f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: Save {savePath}"
                )
                plt.savefig(savePath, transparent=False)
                plt.close()

                print(
                    f"[@T{targetT} @{rin}R{rout} @{int(snapNumber)}]: ...Tracer Plot done!"
                )

    return

def multi_halo_merge(  simList,
                        haloPathList,
                        FullDataPathSuffix,
                        snapRange,
                        Tlst,
                        TracersParamsPath = "TracersParams.csv"
                        ):
    """
        This function is designed to combine the data sets for multiple
        Auriga simulation datasets from Tracer.py analysis.
        NOTE: This is NOT the flatten_wrt_time version!

        inputs:
            simList: list [dtype = 'str']
            haloPathList: list [dtype = 'str']
            FullDataPathSuffix: str
            snapRange: list [dtype = 'int']
            Tlst: list [dtype = 'str']
            TracersParamsPath: str

        outputs:
            mergedDict: dictionary
                        keys = (
                        f"T{T}",
                        f"{rin}R{rout}",
                        f"{int(snap)}",
                        )
            saveParams: list [dtype = 'str']
    """
    import collections

    mergedDict = {}
    saveParams = []
    loadedParams = []
    for sim,loadPath in zip(simList,haloPathList):
        loadPath += '/'

        TRACERSPARAMS, DataSavepath , _ = load_tracers_parameters(loadPath+TracersParamsPath)
        saveParams += TRACERSPARAMS["saveParams"]

        saveHalo = (sim.split("_"))[-1]
        if 'L' in saveHalo:
            saveHalo = saveHalo.split('L')[-1]
            padFlag = True
        else:
            padFlag = False

        print("")
        print(f"Loading {sim} Data!")

        dataDict = {}
        print("LOAD")
        dataDict = full_dict_hdf5_load(DataSavepath, TRACERSPARAMS, FullDataPathSuffix)

        print("LOADED")

        # Pad id, prid, and trid, with unique Auriga halo      #
        # prefix. This should ensure there are no repeat id    #
        # numbers.
        print('PAD')
        for selectKey in dataDict.keys():
            for key in ['id','prid','trid']:
                ## Add Halo Number plus one zero to start of every number ##
                if padFlag is False:
                    index = math.ceil(np.log10(np.nanmax(dataDict[selectKey][key])))

                    dataDict[selectKey][key] = dataDict[selectKey][key] + int(int(saveHalo) * 10 ** (1 + index))
                else:
                    index = math.ceil(np.log10(np.nanmax(dataDict[selectKey][key])))

                    dataDict[selectKey][key] = dataDict[selectKey][key] +  int(int(saveHalo) * 10 ** (1 + index)) + int(9*10**(3+index))
                # np.array([
                #int(str(saveHalo)+'0'+str(v)) for v in dataDict[selectKey][key]
                #])
        print('PADDED')
        selectKey0 = list(dataDict.keys())[0]
        loadedParams += list(dataDict[selectKey0].keys())
        print('MERGE')

        for selectKey in dataDict.keys():
            for key in dataDict[selectKey].keys():
                if selectKey in list(mergedDict.keys()):
                    if key in list(mergedDict[selectKey].keys()):

                        tmp = np.concatenate((mergedDict[selectKey][key],dataDict[selectKey][key]),axis=0)

                        mergedDict[selectKey].update({key:  tmp})

                    else:

                        mergedDict[selectKey].update({key : dataDict[selectKey][key]})
                else:

                    mergedDict.update({selectKey : {key : dataDict[selectKey][key]}})

        print('MERGED')
        print('debug',"mergedDict[selectKey]['id']",mergedDict[selectKey]['id'])

    ### Check all sims contained same params ###
    paramFreqDict = collections.Counter(saveParams)
    counts = list(paramFreqDict.values())
    truthy = np.all(np.array([el == len(simList) for el in counts]))

    if truthy == False:
        print("")
        print(f"Param Counts Dict: {paramFreqDict}")
        raise Exception("[@ multi_halo_merge]: WARNING! CRITICAL! Simulations do not contain same Save Parameters (saveParams)! Check TracersParams.csv!")

    ### Check all LOADED DATA contained same params ###
    paramFreqDict = collections.Counter(loadedParams)
    counts = list(paramFreqDict.values())
    truthy = np.all(np.array([el == len(simList) for el in counts]))
    if truthy == False:
        print("")
        print(f"Param Counts Dict: {paramFreqDict}")
        raise Exception("[@ multi_halo_merge]: WARNING! CRITICAL! Flattened Data do not contain same Save Parameters (saveParams)! Check TracersParams.csv BEFORE flatten_wrt_time contained same Save Parameters (saveParams)!")

    saveParams = np.unique(np.array(saveParams)).tolist()
    return mergedDict,saveParams

def multi_halo_merge_flat_wrt_time(  simList,
                        haloPathList,
                        FullDataPathSuffix,
                        snapRange,
                        Tlst,
                        TracersParamsPath = "TracersParams.csv",
                        loadParams = None,
                        dtwSubset = None
                        ):
    """
        This function is designed to combine the data sets for multiple
        Auriga simulation datasets from Tracer.py analysis.
        NOTE: THIS IS the flatten_wrt_time version!

        inputs:
            simList: list [dtype = 'str']
            haloPathList: list [dtype = 'str']
            FullDataPathSuffix: str
            snapRange: list [dtype = 'int']
            Tlst: list [dtype = 'str']
            TracersParamsPath: str

        outputs:
            mergedDict: dictionary
                        keys = (
                        f"T{T}",
                        f"{rin}R{rout}",
                        f"{int(snap)}",
                        )
            saveParams: list [dtype = 'str']
    """
    import collections
    random.seed(1234)

    mergedDict = {}
    saveParams = []
    loadedParams = []
    for sim,loadPath in zip(simList,haloPathList):
        loadPath += '/'

        TRACERSPARAMS, DataSavepath , _ = load_tracers_parameters(loadPath+TracersParamsPath)
        saveParams += TRACERSPARAMS["saveParams"]

        saveHalo = (sim.split("_"))[-1]
        if 'L' in saveHalo:
            saveHalo = saveHalo.split('L')[-1]
            padFlag = True
        else:
            padFlag = False

        print("")
        print(f"Loading {sim} Data!")


        print("LOAD")
        dataDict = {}
        for T in Tlst:
            for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
                loadPath = (
                    DataSavepath + f"_T{T}_{rin}R{rout}_flat-wrt-time" + FullDataPathSuffix
                )
                key = (f"T{T}", f"{rin}R{rout}")
                tmp = hdf5_load(loadPath)
                dataDict.update(tmp)


        print("LOADED")

        # Pad id, prid, and trid, with unique Auriga halo      #
        # prefix. This should ensure there are no repeat id    #
        # numbers.
        print('PAD')
        for selectKey in dataDict.keys():
            for key in ['id','prid','trid']:
                ## Add Halo Number plus one zero to start of every number ##
                if padFlag is False:
                    index = math.ceil(np.log10(np.nanmax(dataDict[selectKey][key])))

                    dataDict[selectKey][key] = dataDict[selectKey][key] + int(int(saveHalo) * 10 ** (1 + index))
                else:
                    index = math.ceil(np.log10(np.nanmax(dataDict[selectKey][key])))

                    dataDict[selectKey][key] = dataDict[selectKey][key] +  int(int(saveHalo) * 10 ** (1 + index)) + int(9*10**(3+index))
                # np.array([
                #int(str(saveHalo)+'0'+str(v)) for v in dataDict[selectKey][key]
                #])
        print('PADDED')
        selectKey0 = list(dataDict.keys())[0]
        loadedParams += list(dataDict[selectKey0].keys())
        print('MERGE')
        if loadParams is not None:
            print('Loading loadParams ONLY! Discarding the rest of data')
            print(f'loadParams = {loadParams}')
            if dtwSubset is not None:
                print('Loading dtwSubset ONLY! Discarding the rest of data')
                print(f'dtwSubset = {dtwSubset} Data Points per halo, per time, per temperature, per radius.')
                for selectKey in dataDict.keys():

                    typeLen = np.shape(dataDict[selectKey]['type'])[1]
                    subset = min(typeLen,dtwSubset)

                    dtwSelect = random.sample([ ii for ii in range(0,typeLen,1)], k=subset)

                    for key in dataDict[selectKey].keys():
                        if key in loadParams:
                            if selectKey in list(mergedDict.keys()):
                                if key in list(mergedDict[selectKey].keys()):

                                    #AXIS 0 now temporal axis, so concat on axis 1
                                    tmp = np.concatenate((mergedDict[selectKey][key],dataDict[selectKey][key][:, dtwSelect]),axis=1)

                                    mergedDict[selectKey].update({key:  tmp})

                                else:

                                    mergedDict[selectKey].update({key : dataDict[selectKey][key][:, dtwSelect]})
                            else:

                                mergedDict.update({selectKey : {key : dataDict[selectKey][key][:, dtwSelect]}})
            else:
                for selectKey in dataDict.keys():
                    for key in dataDict[selectKey].keys():
                        if key in loadParams:
                            if selectKey in list(mergedDict.keys()):
                                if key in list(mergedDict[selectKey].keys()):

                                    #AXIS 0 now temporal axis, so concat on axis 1
                                    tmp = np.concatenate((mergedDict[selectKey][key],dataDict[selectKey][key]),axis=1)

                                    mergedDict[selectKey].update({key:  tmp})

                                else:

                                    mergedDict[selectKey].update({key : dataDict[selectKey][key]})
                            else:

                                mergedDict.update({selectKey : {key : dataDict[selectKey][key]}})
        else:
            for selectKey in dataDict.keys():
                for key in dataDict[selectKey].keys():

                    if selectKey in list(mergedDict.keys()):
                        if key in list(mergedDict[selectKey].keys()):

                            #AXIS 0 now temporal axis, so concat on axis 1
                            tmp = np.concatenate((mergedDict[selectKey][key],dataDict[selectKey][key]),axis=1)

                            mergedDict[selectKey].update({key:  tmp})

                        else:

                            mergedDict[selectKey].update({key : dataDict[selectKey][key]})
                    else:

                        mergedDict.update({selectKey : {key : dataDict[selectKey][key]}})

        print('MERGED')
        print('debug',"mergedDict[selectKey]['trid']",mergedDict[selectKey]['trid'])

    ### Check all sims contained same params ###
    paramFreqDict = collections.Counter(saveParams)
    counts = list(paramFreqDict.values())
    truthy = np.all(np.array([el == len(simList) for el in counts]))
    if truthy == False:
        print("")
        print(f"Param Counts Dict: {paramFreqDict}")
        raise Exception("[@ multi_halo_merge]: WARNING! CRITICAL! Simulations do not contain same Save Parameters (saveParams)! Check TracersParams.csv!")

    ### Check all LOADED DATA contained same params ###
    paramFreqDict = collections.Counter(loadedParams)
    counts = list(paramFreqDict.values())
    truthy = np.all(np.array([el == len(simList) for el in counts]))
    if truthy == False:
        print("")
        print(f"Param Counts Dict: {paramFreqDict}")
        raise Exception("[@ multi_halo_merge]: WARNING! CRITICAL! Flattened Data do not contain same Save Parameters (saveParams)! Check TracersParams.csv BEFORE flatten_wrt_time contained same Save Parameters (saveParams)!")


    saveParams = np.unique(np.array(saveParams)).tolist()
    return mergedDict,saveParams


def multi_halo_stats(dataDict,TRACERSPARAMS,saveParams,snapRange,Tlst,MiniDataPathSuffix = f".csv",TracersParamsPath = "TracersParams.csv",TracersMasterParamsPath ="TracersParamsMaster.csv",SelectedHaloesPath = "TracersSelectedHaloes.csv"):
    statsData = {}
    for (rin, rout) in zip(TRACERSPARAMS["Rinner"], TRACERSPARAMS["Router"]):
        for ii in range(len(Tlst)):
            T = Tlst[ii]
            key = (f'T{Tlst[ii]}',f"{rin}R{rout}")
            for snap in snapRange:
                selectKey = (f'T{Tlst[ii]}',f"{rin}R{rout}",f"{snap}")
                dat = save_statistics(
                        dataDict[selectKey],
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
                #Fix values to arrays to remove concat error of 0D arrays
                for k, val in dat.items():
                    dat[k] = np.array([val]).flatten()

                if key in list(statsData.keys()) :
                    for subkey,vals in dat.items():
                        if subkey in list(statsData[key].keys()):

                            statsData[key][subkey] = np.concatenate((statsData[key][subkey],dat[subkey]),axis=0)
                        else:
                            statsData[key].update({subkey : dat[subkey]})
                else:
                    statsData.update({key: dat})

    return statsData
