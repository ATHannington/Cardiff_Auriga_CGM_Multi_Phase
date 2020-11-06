import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

ageUniverse = 13.77 #[Gyr]

#Input parameters path:
TracersParamsPath = 'TracersParams.csv'
DataSavepathSuffix = f".h5"

#==============================================================================#
#       Chemical Properties
#==============================================================================#
#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
elements_mass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

Zsolar = 0.0127

omegabaryon0 = 0.048
#==============================================================================#
#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

snapsRange = np.array([ xx for xx in range(int(TRACERSPARAMS['snapMin']), min(int(TRACERSPARAMS['snapMax'])+1,int(TRACERSPARAMS['finalSnap'])+1),1)])

################################################################################
##                           Definitions                                    ####
################################################################################
#------------------------------------------------------------------------------#
#                        Get id trid prid data
#------------------------------------------------------------------------------#
def _get_id(dataDict,whereEntries):

    id = dataDict['id'][whereEntries]

    _,prid_ind,_ = np.intersect1d(dataDict['prid'],id,return_indices=True)

    prid = dataDict['prid'][prid_ind]
    trid = dataDict['trid'][prid_ind]

    return {'id': id, 'prid': prid, 'trid' : trid}
#------------------------------------------------------------------------------#
#               Analyse log10(T) and snap specfifc statistics
#------------------------------------------------------------------------------#
def _inner_analysis(dataDict):

    out = {}

    NtracersAll = dataDict['Ntracers']

    id = dataDict['id']
    prid = dataDict['prid']
    trid = dataDict['trid']

    tracers = {'data' : NtracersAll , 'id' : id, 'prid': prid, 'trid': trid}
    out.update({"TracersFull" : tracers})

    gas = {}
    whereGas = np.where(dataDict['type']==0)[0]
    tmp = _get_id(dataDict,whereGas)
    gas.update(tmp)
    gasdata = (100.*(np.shape(gas['trid'])[0]/NtracersAll))
    gas.update({'data' : gasdata})
    out.update({"%Gas": gas})

    Ntracers = np.shape(whereGas)[0]

    SubHalo = dataDict['SubHaloID']

    halo0 = {}
    wherehalo0 = np.where(SubHalo[whereGas]==int(TRACERSPARAMS['haloID']))[0]
    tmp = _get_id(dataDict,wherehalo0)
    halo0.update(tmp)
    halo0data = (100.*(np.shape(halo0['trid'])[0]/Ntracers))
    halo0.update({'data' : halo0data})
    out.update({"%Halo0": halo0})

    unbound = {}
    whereunbound = np.where(SubHalo[whereGas]==-1)[0]
    tmp = _get_id(dataDict,whereunbound)
    unbound.update(tmp)
    unbounddata = (100.*(np.shape(unbound['trid'])[0]/Ntracers))
    unbound.update({'data' : unbounddata})
    out.update({"%Unbound": unbound})

    otherHalo = {}
    whereotherHalo = np.where(SubHalo[whereGas]!=int(TRACERSPARAMS['haloID']))\
        &(SubHalo[whereGas]!=-1)&(np.isnan(SubHalo[whereGas])==False))[0]
    tmp = _get_id(dataDict,whereotherHalo)
    otherHalo.update(tmp)
    otherHalodata = (100.*(np.shape(otherHalo['trid'])[0]/Ntracers))
    otherHalo.update({'data' : otherHalodata})
    out.update({"%OtherHalo": otherHalo})

    noHalo = {}
    wherenoHalo = np.where((SubHalo[whereGas]!=int(TRACERSPARAMS['haloID']))\
        &(SubHalo[whereGas]!=-1)&(np.isnan(SubHalo[whereGas])==True))[0]
    tmp = _get_id(dataDict,wherenoHalo)
    noHalo.update(tmp)
    noHalodata = (100.*(np.shape(noHalo['trid'])[0]/Ntracers))
    noHalo.update({'data' : noHalodata})
    out.update({"%NoHalo": noHalo})

    stars = {}
    wherestars = np.where((dataDict['age']>=0)&(dataDict['type']==4))[0]
    tmp = _get_id(dataDict,wherestars)
    stars.update(tmp)
    starsdata = (100.*(np.shape(stars['trid'])[0]/NtracersAll))
    stars.update({'data' : starsdata})
    out.update({"%Stars": stars})

    wind = {}
    wherewind = np.where((dataDict['age']<0)&(dataDict['type']==4))[0]
    tmp = _get_id(dataDict,wherewind)
    wind.update(tmp)
    winddata = (100.*(np.shape(wind['trid'])[0]/NtracersAll))
    wind.update({'data' : winddata})
    out.update({"%Wind": wind})

    ism = {}
    whereism = np.where((dataDict['sfr']>0)&(dataDict['R']<=25.))[0]
    tmp = _get_id(dataDict,whereism)
    ism.update(tmp)
    ismdata = (100.*(np.shape(ism['trid'])[0]/Ntracers))
    ism.update({'data' : ismdata})
    out.update({"%ISM": ism})


    inflow = {}
    whereinflow = np.where((dataDict['sfr']>0)&(dataDict['R']<=25.))[0]
    tmp = _get_id(dataDict,whereinflow)
    inflow.update(tmp)
    inflowdata = (100.*(np.shape(inflow['trid'])[0]/Ntracers))
    inflow.update({'data' : inflowdata})
    out.update({"%Inflow": inflow})

    outflow = {}
    whereoutflow = np.where(dataDict['vrad'][whereGas]>0.)[0]
    tmp = _get_id(dataDict,whereoutflow)
    outflow.update(tmp)
    outflowdata = (100.*(np.shape(outflow['trid'])[0]/Ntracers))
    outflow.update({'data' : outflowdata})
    out.update({"%Outflow": outflow})

    aboveZ = {}
    whereaboveZ = np.where(dataDict['gz'][whereGas]>=(1./3.)*Zsolar)[0]
    tmp = _get_id(dataDict,whereaboveZ)
    aboveZ.update(tmp)
    aboveZdata = (100.*(np.shape(aboveZ['trid'])[0]/Ntracers))
    aboveZ.update({'data' : aboveZdata})
    out.update({"%Above1/3(Z_solar)": aboveZ})

    belowZ = {}
    wherebelowZ = np.where(dataDict['gz'][whereGas]<=(1./3.)*Zsolar)[0]
    tmp = _get_id(dataDict,wherebelowZ)
    belowZ.update(tmp)
    belowZdata = (100.*(np.shape(belowZ['trid'])[0]/Ntracers))
    belowZ.update({'data' : belowZdata})
    out.update({"%Below1/3(Z_solar)": belowZ})

    heating = {}
    whereheating = np.where(np.isnan(dataDict['theat'][whereGas])==False)[0]
    tmp = _get_id(dataDict,whereheating)
    heating.update(tmp)
    heatingdata = (100.*(np.shape(heating['trid'])[0]/Ntracers))
    heating.update({'data' : heatingdata})
    out.update({"%Heating": heating})

    cooling = {}
    wherecooling = np.where(np.isnan(dataDict['tcool'][whereGas])==False)[0]
    tmp = _get_id(dataDict,wherecooling)
    cooling.update(tmp)
    coolingdata = (100.*(np.shape(cooling['trid'])[0]/Ntracers))
    cooling.update({'data' : coolingdata})
    out.update({"%Cooling": cooling})

    out.update({"T": T})
    out.update({'Lookback' : dataDict['Lookback']})

    return  out
#------------------------------------------------------------------------------#
#               Analyse statistics for all T and snaps
#------------------------------------------------------------------------------#
def fullData_analyse(dataDict,Tlst,snapsRange):
    for T in Tlst:
        Tkey = f"T{T}"
        for ii, snap in enumerate(snapsRange):
            FullKey = (Tkey , f"{int(snap)})
            snapdataDict = dataDict[FullKey]
            outinner  = _inner_analysis(snapdataDict)
        out.update({FullKey : outinner})
    return out
################################################################################
##                           MAIN PROGRAM                                   ####
################################################################################
print("Analyse Data!")
statsDict = fullData_analyse(dataDict,Tlst,snapsRange)
