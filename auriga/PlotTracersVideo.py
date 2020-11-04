"""
Author: A. T. Hannington
Created: 26/03/2020

Known Bugs:
    pandas read_csv loading data as nested dict. . Have added flattening to fix

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

subset = 100
ageUniverse = 13.77 #[Gyr]

TracersParamsPath = 'TracersParams.csv'
singleValueParams = ['Lookback','Ntracers','Snap']
#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

dataParams = TRACERSPARAMS['saveParams'] + TRACERSPARAMS['saveTracersOnly'] + TRACERSPARAMS['saveEssentials']

for param in singleValueParams:
    dataParams.remove(param)

print("Loading data!")

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

#==============================================================================#
#   Get Data within range of z-axis LOS
#==============================================================================#
print("Getting Tracer Data!")

boxlos = TRACERSPARAMS['boxlos']
zAxis = int(TRACERSPARAMS['zAxis'])
#Loop over temperatures in targetTLst and grab Temperature specific subset of tracers and relevant data
for T in TRACERSPARAMS['targetTLst']:
    print("")
    print(f"Starting T{T} analysis")

    for snap in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['snapMax']+1),int(TRACERSPARAMS['finalSnap']+1))):
        #Grab the data for tracers in projection LOS volume
        data = {}
        for analysisParam in dataParams:
            key = (f"T{T}",f"{int(snap)}")
            whereInRange = np.where((dataDict[key]['pos'][:,zAxis]<=(float(boxlos)/2.))&(dataDict[key]['pos'][:,zAxis]>=(-1.*float(boxlos)/2.)))
            if (analysisParam == 'trid'):
                pridsIndices = np.where(np.isin(dataDict[key]['prid'],dataDict[key]['id'][whereInRange]]))[0]
                trids = dataDict[key]['trid'][pridsIndices]
                tmpdata = trids
            elif (analysisParam == 'prid'):
                pridsIndices = np.where(np.isin(dataDict[key]['prid'],dataDict[key]['id'][whereInRange]]))[0]
                prids = dataDict[key]['prid'][pridsIndices]
                tmpdata = prids
            else:
                tmpdata = dataDict[key][analysisParam][whereInRange]

            data.update({analysisParam : np.array(tmpdata)})
        #Add the full list of snaps data to temperature dependent dictionary.
        dataDict.update({(f"T{T}", f"{int(snap)}"): data})

#==============================================================================#
#   Get Data within range of z-axis LOS common between ALL time-steps
#==============================================================================#

trid_list = []
for subDict in dataDict.values():
    trid_list.append(subDict['trid'])

intersect = reduce(np.intersect1d,trid_list)

intersectList =[]
for outerkey, subDict in dataDict.items():
    trids = subDict['trid']
    tmp = {}
    for key, value in subDict.items():

        entry, a_ind, b_ind = np.intersect1d(trids,intersect,return_indices=True)
        prids = subDict['prid'][a_ind]

        _, id_indices, _ = np.intersect1d(subDict['id'],prids,return_indices=True)

        tmp.update({key : value[id_indices]})
        intersectList.append(a_ind)
    dataDict.update({outerkey : tmp})

oldIntersect = intersectList[0]
for entry in intersectList:
    assert np.shape(entry) == np.shape(oldIntersect)

#==============================================================================#
#   Get Data within range of z-axis LOS common between ALL time-steps
#==============================================================================#

TracerPlot(dataDict,TRACERSPARAMS, DataSavepath,\
FullDataPathSuffix, Axes=TRACERSPARAMS['Axes'], zAxis=TRACERSPARAMS['zAxis'],\
boxsize = TRACERSPARAMS['boxsize'], boxlos = TRACERSPARAMS['boxlos'],\
pixres = TRACERSPARAMS['pixres'], pixreslos = TRACERSPARAMS['pixreslos'])
