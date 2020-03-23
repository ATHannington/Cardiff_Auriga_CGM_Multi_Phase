"""
Author: A. T. Hannington
Created: 12/03/2020
Known Bugs:
    None
"""
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

    #Select Cell IDs for cells which meet condition
    CellIDs = snapGas.id[Cond]

    #Select Parent IDs in Cond list
    #   Select parent IDs of cells which contain tracers and have IDs from selection of meeting condition
    Parents = np.isin(snapTracers.prid,CellIDs)

    #Select Indices (positions in array) of these parent IDs
    ParentsIndices = np.where(Parents)

    #Find Index of Cell IDs. These Cell IDs now match cond AND contain tracers
    CellsIndices = np.where(np.isin(snapGas.id,Parents))

    #Find the updated Cell IDs - match cond AND contain tracers
    CellIDs = snapGas.id[CellsIndices]

    #Select the data for Cells that meet Cond which contain tracers
    #   Does this by creating new dict from old data.
    #       Only selects values at index where Cell meets cond and contains tracers
    Cells={}
    for key, value in snapGas.data.items():
        if value is not None:
                Cells.update({key: value[CellsIndices]})


    #Finally, select TRACER IDs who are in cells that meet Cond
    Tracers = snapTracers.trid[ParentsIndices]

    return Tracers, Cells, CellIDs

#------------------------------------------------------------------------------#
def GetCellsFromTracers(snapGas, snapTracers,Tracers):
    print("GetCellsFromTracers")

    #Select indices (positions in array) of Tracer IDs which are in the Tracers list
    TracersIndices = np.where(np.isin(snapTracers.trid,Tracers))

    #Select the matching parent cell IDs for tracers which are in Tracers list
    Parents = snapTracers.prid[TracersIndices]

    #Select Indices of cell IDs which are in the parents list, the list of cell IDs
    #   which contain tracers in tracers list.
    CellsIndices = np.where(np.isin(snapGas.id,Parents))

    #Select data from cells which contain tracers
    #   Does this by making a new dictionary from old data. Only selects values
    #       At indices of Cells which contain tracers in tracers list.
    Cells={}
    for key, value in snapGas.data.items():
        if value is not None:
                Cells.update({key: value[CellsIndices]})

    #Select IDs of Cells which contain tracers in tracers list.
    CellIDs = snapGas.id[CellsIndices]

    return Cells, CellIDs

#------------------------------------------------------------------------------#
##  FvdV weighted percentile code:
#------------------------------------------------------------------------------#
def weightedperc(data, weights, perc):
    #percentage to decimal
    perc /= 100.

    #Indices of data array in sorted form
    ind_sorted = np.argsort(data)

    #Sort the data
    sorted_data = np.array(data)[ind_sorted]

    #Sort the weights by the sorted data sorting
    sorted_weights = np.array(weights)[ind_sorted]

    #Find the cumulative sum of the weights
    cm = np.cumsum(sorted_weights)

    #Find indices where cumulative some as a fraction of final cumulative sum value is greater than percentage
    whereperc = np.where(cm/float(cm[-1]) >= perc)

    #Reurn the first data value where above is true
    return sorted_data[whereperc[0]]
