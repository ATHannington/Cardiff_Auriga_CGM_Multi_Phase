import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
from Snapper import *


def GetTracersFromCells(snapGas, snapTracers,Cond):
    print("GetTracersFromCells")

    # print("CellIDs initial from Cond")
    #Select CellIDs for Cond
    CellIDs = snapGas.id[Cond]
    # print(CellIDs)
    # print(np.shape(CellIDs))

    #Select Parent IDs in Cond list
    #   Selecting from cells with a tracer, which cell IDs are have cell IDs that meet Cond
    # print("Parents")
    Parents = np.isin(snapTracers.prid,CellIDs)
    # print(Parents)
    # print(np.shape(Parents))

    #Select INDICES of the tracers who are in a cell which meets Cond
    # print("ParentsIndices")
    ParentsIndices = np.where(Parents)
    # print(ParentsIndices)
    # print(np.shape(ParentsIndices))

    # print("CellIDs final Containing Tracers Only")
    CellsIndices = np.where(np.isin(snapGas.id,snapTracers.prid[ParentsIndices]))
    CellIDs = snapGas.id[CellsIndices]
    # print(CellIDs)
    # print(np.shape(CellIDs))

    #Select the data for Cells that meet Cond which contain tracers
    #   Does this by making a new dictionary from old data
    #       but only selects values where Parent ID is True
    #           i.e. where Cond is met and Cell ID contains tracers
    # print("Cells with tracers")
    Cells={}
    for key, value in snapGas.data.items():
        if value is not None:
                Cells.update({key: value[CellsIndices]})
    # print(Cells)

    #Finally, select TRACER IDs who are in cells that meet Cond
    # print("Tracers")
    Tracers = snapTracers.trid[ParentsIndices]
    # print(Tracers)
    # print(np.shape(Tracers))


    return Tracers, Cells, CellIDs

#------------------------------------------------------------------------------#
def GetCellsFromTracers(snapGas, snapTracers,Tracers):
    print("GetCellsFromTracers")

    # print(Tracers)
    # print(np.shape(Tracers))
    # print("Tracers Indices")
    TracersIndices = np.where(np.isin(snapTracers.trid,Tracers))
    # print(TracersIndices)
    # print(np.shape(TracersIndices))

    # print("Parents")
    Parents = snapTracers.prid[TracersIndices]
    # print(Parents)
    # print(np.shape(Parents))

    # print("CellsIndices")
    CellsIndices = np.isin(snapGas.id,Parents)
    # print(CellsIndices)
    # print(np.shape(CellsIndices))
    # print("Cells")
    Cells={}
    for key, value in snapGas.data.items():
        if value is not None:
                Cells.update({key: value[np.where(CellsIndices)]})

    # Cells = {key: value for ind, (key, value) in enumerate(snapGas.data.items()) if CellsIndices[ind] == True}
    # print(Cells)

    # print("CellIDs")
    CellIDs = snapGas.id[np.where(CellsIndices)]
    # print(CellIDs)
    # print(np.shape(CellIDs))
    return Cells, CellIDs
#------------------------------------------------------------------------------#
##  FvdV weighted percentile code:
#------------------------------------------------------------------------------#
def weightedperc(data, weights, perc):
    perc /= 100.
    ind_sorted = np.argsort(data)
    sorted_data = np.array(data)[ind_sorted]
    sorted_weights = np.array(weights)[ind_sorted]
    cum = np.cumsum(sorted_weights)
    whereperc, = np.where(cum/float(cum[-1]) >= perc)

    return sorted_data[whereperc[0]]
