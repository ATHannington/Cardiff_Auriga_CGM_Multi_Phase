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
singleVals = ["T","Snap","Lookback"]
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

loadPath = DataSavepath + f"_flat-wrt-time"+ DataSavepathSuffix

FlatDataDict = hdf5_load(loadPath)
################################################################################
##                           Definitions                                    ####
################################################################################
#------------------------------------------------------------------------------#
#                    Get id trid prid data from where Cells
#------------------------------------------------------------------------------#
def _get_id_prid_trid_where(dataDict,whereEntries):

    id = dataDict['id'][whereEntries]

    _,prid_ind,_ = np.intersect1d(dataDict['prid'],id,return_indices=True)

    prid = dataDict['prid'][prid_ind]
    trid = dataDict['trid'][prid_ind]

    return {'id': id, 'prid': prid, 'trid' : trid}
#------------------------------------------------------------------------------#
#                    Get id data from single trid
#------------------------------------------------------------------------------#

def _get_id_from_single_trid(dataDict,trid):

    prid_ind = np.where(dataDict['trid']==trid)
    prid = dataDict['prid'][prid_ind]

    id = np.where(dataDict['id']==prid)

    return id
#------------------------------------------------------------------------------#
#
#------------------------------------------------------------------------------#

def flat_analyse(FlatDataDict):

    return
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
    tmp = _get_id_prid_trid_where(dataDict,whereGas)
    gas.update(tmp)
    gasdata = (100.*(np.shape(gas['trid'])[0]/NtracersAll))
    gas.update({'data' : gasdata})
    out.update({"%Gas": gas})

    Ntracers = np.shape(whereGas)[0]

    SubHalo = dataDict['SubHaloID']

    halo0 = {}
    wherehalo0 = np.where(SubHalo[whereGas]==int(TRACERSPARAMS['haloID']))[0]
    tmp = _get_id_prid_trid_where(dataDict,wherehalo0)
    halo0.update(tmp)
    halo0data = (100.*(np.shape(halo0['trid'])[0]/Ntracers))
    halo0.update({'data' : halo0data})
    out.update({"%Halo0": halo0})

    unbound = {}
    whereunbound = np.where(SubHalo[whereGas]==-1)[0]
    tmp = _get_id_prid_trid_where(dataDict,whereunbound)
    unbound.update(tmp)
    unbounddata = (100.*(np.shape(unbound['trid'])[0]/Ntracers))
    unbound.update({'data' : unbounddata})
    out.update({"%Unbound": unbound})

    otherHalo = {}
    whereotherHalo = np.where((SubHalo[whereGas]!=int(TRACERSPARAMS['haloID']))\
        &(SubHalo[whereGas]!=-1)&(np.isnan(SubHalo[whereGas])==False)) [0]
    tmp = _get_id_prid_trid_where(dataDict,whereotherHalo)
    otherHalo.update(tmp)
    otherHalodata = (100.*(np.shape(otherHalo['trid'])[0]/Ntracers))
    otherHalo.update({'data' : otherHalodata})
    out.update({"%OtherHalo": otherHalo})

    noHalo = {}
    wherenoHalo = np.where((SubHalo[whereGas]!=int(TRACERSPARAMS['haloID']))\
        &(SubHalo[whereGas]!=-1)&(np.isnan(SubHalo[whereGas])==True))[0]
    tmp = _get_id_prid_trid_where(dataDict,wherenoHalo)
    noHalo.update(tmp)
    noHalodata = (100.*(np.shape(noHalo['trid'])[0]/Ntracers))
    noHalo.update({'data' : noHalodata})
    out.update({"%NoHalo": noHalo})

    stars = {}
    wherestars = np.where((dataDict['age']>=0)&(dataDict['type']==4))[0]
    tmp = _get_id_prid_trid_where(dataDict,wherestars)
    stars.update(tmp)
    starsdata = (100.*(np.shape(stars['trid'])[0]/NtracersAll))
    stars.update({'data' : starsdata})
    out.update({"%Stars": stars})

    wind = {}
    wherewind = np.where((dataDict['age']<0)&(dataDict['type']==4))[0]
    tmp = _get_id_prid_trid_where(dataDict,wherewind)
    wind.update(tmp)
    winddata = (100.*(np.shape(wind['trid'])[0]/NtracersAll))
    wind.update({'data' : winddata})
    out.update({"%Wind": wind})

    ism = {}
    whereism = np.where((dataDict['sfr']>0)&(dataDict['R']<=25.))[0]
    tmp = _get_id_prid_trid_where(dataDict,whereism)
    ism.update(tmp)
    ismdata = (100.*(np.shape(ism['trid'])[0]/Ntracers))
    ism.update({'data' : ismdata})
    out.update({"%ISM": ism})


    inflow = {}
    whereinflow = np.where(dataDict['vrad'][whereGas]<0.)[0]
    tmp = _get_id_prid_trid_where(dataDict,whereinflow)
    inflow.update(tmp)
    inflowdata = (100.*(np.shape(inflow['trid'])[0]/Ntracers))
    inflow.update({'data' : inflowdata})
    out.update({"%Inflow": inflow})

    outflow = {}
    whereoutflow = np.where(dataDict['vrad'][whereGas]>0.)[0]
    tmp = _get_id_prid_trid_where(dataDict,whereoutflow)
    outflow.update(tmp)
    outflowdata = (100.*(np.shape(outflow['trid'])[0]/Ntracers))
    outflow.update({'data' : outflowdata})
    out.update({"%Outflow": outflow})

    aboveZ = {}
    whereaboveZ = np.where(dataDict['gz'][whereGas]>(0.75))[0]
    tmp = _get_id_prid_trid_where(dataDict,whereaboveZ)
    aboveZ.update(tmp)
    aboveZdata = (100.*(np.shape(aboveZ['trid'])[0]/Ntracers))
    aboveZ.update({'data' : aboveZdata})
    out.update({"%Above3/4(Z_solar)": aboveZ})

    belowZ = {}
    wherebelowZ = np.where(dataDict['gz'][whereGas]<(0.75))[0]
    tmp = _get_id_prid_trid_where(dataDict,wherebelowZ)
    belowZ.update(tmp)
    belowZdata = (100.*(np.shape(belowZ['trid'])[0]/Ntracers))
    belowZ.update({'data' : belowZdata})
    out.update({"%Below3/4(Z_solar)": belowZ})

    heating = {}
    whereheating = np.where(np.isnan(dataDict['theat'][whereGas])==False)[0]
    tmp = _get_id_prid_trid_where(dataDict,whereheating)
    heating.update(tmp)
    heatingdata = (100.*(np.shape(heating['trid'])[0]/Ntracers))
    heating.update({'data' : heatingdata})
    out.update({"%Heating": heating})

    cooling = {}
    wherecooling = np.where(np.isnan(dataDict['tcool'][whereGas])==False)[0]
    tmp = _get_id_prid_trid_where(dataDict,wherecooling)
    cooling.update(tmp)
    coolingdata = (100.*(np.shape(cooling['trid'])[0]/Ntracers))
    cooling.update({'data' : coolingdata})
    out.update({"%Cooling": cooling})

    out.update({'Lookback' : dataDict['Lookback']})

    return  out
#------------------------------------------------------------------------------#
#               Analyse statistics for all T and snaps
#------------------------------------------------------------------------------#
def fullData_analyse(dataDict,Tlst,snapsRange):
    dflist = []
    out ={}
    for T in Tlst:
        Tkey = f"T{T}"
        print(Tkey)
        tmp = {}
        for ii, snap in enumerate(snapsRange):
            print(f"{int(snap)}")
            FullKey = (Tkey , f"{int(snap)}")
            snapdataDict = dataDict[FullKey]
            outinner  = _inner_analysis(snapdataDict)
            outinner.update({"T": T})
            outinner.update({"Snap": int(snap)})

            for key, value in outinner.items():
                if (key == "T"):
                    tmp.update({"T" : value})
                elif (key == "Lookback"):
                    tmp.update({"Lookback" : value})
                elif (key == "Snap"):
                    tmp.update({"Snap" : value})
                else:
                    tmp.update({key : value["data"]})
            dflist.append(pd.DataFrame.from_dict(tmp))
            out.update({FullKey : outinner})

    outDF = pd.concat(dflist, axis=0, join='outer',sort=False, ignore_index=True)
    return out, outDF
#------------------------------------------------------------------------------#
#                        Get id trid prid data
#------------------------------------------------------------------------------#


    prid_ind = np.where(dataDict[key]['trid'] == trid)
    prid = dataDict[key]['prid'][prid_ind]
    cell_ind = np.where(dataDict[key]['id'] == prid)

    dataDict[key][''][cell_ind] >=<! cond

################################################################################
##                           MAIN PROGRAM                                   ####
################################################################################
print("Analyse Data!")
#------------------------------------------------------------------------------#
#               Analyse statistics for all T and snaps
#------------------------------------------------------------------------------#
statsDict, statsDF = fullData_analyse(dataDict,Tlst,snapsRange)

#Add Stats of medians and percentiles
tmplist = []
for T in Tlst:
     tmp = Statistics_hdf5_load(T,DataSavepath,TRACERSPARAMS,DataSavepathSuffix)
     df = pd.DataFrame(tmp).astype('float64')
     tmplist.append(df)
tmpDF = pd.concat(tmplist, axis=0, join='outer',sort=False, ignore_index=True)
finalDF = pd.concat([statsDF,tmpDF], axis=1, join='outer',sort=False)

#Sort Column Order
cols = list(finalDF.columns.values)
for item in singleVals:
    cols.remove(item)
cols = singleVals + cols
finalDF = finalDF[cols]
finalDF = finalDF.astype({"Snap" : int32, "T": float64})

#Save
savePath = DataSavepath + "_Statistics-Table.csv"

print("\n"+f"Saving Stats table .csv as {savePath}")

finalDF.to_csv(savePath,index=False)

# #------------------------------------------------------------------------------#
# #               Analyse Tracers Continuously Exhibiting Feature
# #                   Since SnapMin
# #------------------------------------------------------------------------------#
# final = []
# for param in statsDict[(f"T{Tlst[0]}", f"{int(TRACERSPARAMS['snapMin'])}")].keys():
#     if param not in singleVals:
#         outer = []
#         for T in Tlst:
#             key = (f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")
#             trids = statsDict[key][param]['trid']
#             ntracers = np.shape(trids)[0]
#             tmp = []
#             lookback = []
#             for ii, snap in enumerate(snapsRange):
#                 key = (f"T{T}", f"{int(snap)}")
#                 currentTrids = statsDict[key][param]['trid']
#
#                 intersectTrids = np.intersect1d(currentTrids,trids)
#                 if (ntracers >0 ):
#                     percentage = (100.*np.shape(intersectTrids)[0]/ntracers)
#                 else:
#                     percentage = 0.
#                 tmp.append(percentage)
#                 lookback.append(statsDict[key]['Lookback'][0])
#
#             Tdat = [float(T) for snap in snapsRange]
#             outer.append(pd.DataFrame({f"T" : Tdat, "Snap" : snapsRange, "Lookback" : np.array(lookback), param : np.array(tmp)}))
#         final.append(pd.concat(outer,axis=0, join='outer', ignore_index=True))
# continuousDF = pd.concat(final, axis=1, join='outer',sort=False)
# continuousDF = continuousDF.loc[:,~continuousDF.columns.duplicated()]
#
# savePath = DataSavepath + f"_Continuous-snap{int(TRACERSPARAMS['snapMin'])}" +"_Statistics-Table.csv"
#
# print("\n"+f"Saving Stats table .csv as {savePath}")
#
# continuousDF.to_csv(savePath,index=False)
#
# #------------------------------------------------------------------------------#
# #               Analyse Tracers Moving from Gas to Gas, ISM, Wind, Stars
# #------------------------------------------------------------------------------#
# final = []
# for T in Tlst:
#     key = (f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")
#     trids = statsDict[key]['%Gas']['trid']
#     ntracers = np.shape(trids)[0]
#     outer = []
#     for param in ['%Gas','%Stars','%ISM', '%Wind']:
#         tmp = []
#         lookback = []
#         for ii, snap in enumerate(snapsRange):
#             key = (f"T{T}", f"{int(snap)}")
#             currentTrids = statsDict[key][param]['trid']
#
#             intersectTrids = np.intersect1d(currentTrids,trids)
#             if (ntracers >0 ):
#                 percentage = (100.*np.shape(intersectTrids)[0]/ntracers)
#             else:
#                 percentage = 0.
#             tmp.append(percentage)
#             lookback.append(statsDict[key]['Lookback'][0])
#
#         Tdat = [float(T) for snap in snapsRange]
#         outer.append(pd.DataFrame({f"T" : Tdat, "Snap" : snapsRange, "Lookback" : np.array(lookback), param : np.array(tmp)}))
#     final.append(pd.concat(outer,axis=1, join='outer'))
# gasDF = pd.concat(final, axis=0, join='outer',sort=False, ignore_index=True)
# gasDF = gasDF.loc[:,~gasDF.columns.duplicated()]
#
# savePath = DataSavepath + f"_Gas-phase-change-snap{int(TRACERSPARAMS['snapMin'])}" +"_Statistics-Table.csv"
#
# print("\n"+f"Saving Stats table .csv as {savePath}")
#
# gasDF.to_csv(savePath,index=False)
#
# #------------------------------------------------------------------------------#
# #               Analyse Tracers moving from OtherHalo/Unbound/IGM Gas to ISM
# #------------------------------------------------------------------------------#
#
# otherHalo = []
# noHalo = []
# unbound = []
# halo0ISM = []
# halo0NonISM = []
# for T in Tlst:
#     tridsGas = statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['%Gas']['trid']
#     tridsISM = statsDict[(f"T{T}", f"{int(snapsRange[-1])}")]['%ISM']['trid']
#     gastoISMintersectTrids = np.intersect1d(tridsGas,tridsISM)
#     # tridsStart = dataDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['trid']
#     # _, trid_ind, _ = np.intersect1d(tridsStart,gastoISMintersectTrids, return_indices = True)
#     # pridsStart = dataDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['prid'][trid_ind]
#     # idsStart = dataDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['id']
#     # _, id_ind, _ = np.intersect1d(pridsStart,idsStart, return_indices = True)
#     # RStart = dataDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['R'][id_ind]
#     # SubHaloIDStart = dataDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['SubHaloID'][id_ind]
#         #------------------------------------------------------------------------------#
#         #               Analyse Tracers moving from otherHalo Gas to ISM
#         #------------------------------------------------------------------------------#
#
#     otherHaloStart = np.intersect1d(statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['%OtherHalo']['trid'],gastoISMintersectTrids)
#     otherHaloToISMPercentage = 100.* (np.shape(otherHaloStart)[0]/statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['TracersFull']['data'][0])
#     otherHalo.append(float(otherHaloToISMPercentage))
#         #------------------------------------------------------------------------------#
#         #               Analyse Tracers moving from IGM Gas to ISM
#         #------------------------------------------------------------------------------#
#
#     noHaloStart = np.intersect1d(statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['%NoHalo']['trid'],gastoISMintersectTrids)
#     noHaloToISMPercentage = 100.* (np.shape(noHaloStart)[0]/statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['TracersFull']['data'][0])
#     noHalo.append(float(noHaloToISMPercentage))
#
#         #------------------------------------------------------------------------------#
#         #               Analyse Tracers moving from Unbound Gas to ISM
#         #------------------------------------------------------------------------------#
#
#     UnboundStart = np.intersect1d(statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['%Unbound']['trid'],gastoISMintersectTrids)
#     UnboundToISMPercentage = 100.* (np.shape(UnboundStart)[0]/statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['TracersFull']['data'][0])
#     unbound.append(float(UnboundToISMPercentage))
#
#         #------------------------------------------------------------------------------#
#         #               Analyse Tracers moving from Halo 0 ISM Gas back to ISM
#         #------------------------------------------------------------------------------#
#
#     Halo0Start = np.intersect1d(statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['%Halo0']['trid'],gastoISMintersectTrids)
#     Halo0ISMStart = np.intersect1d(statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['%ISM']['trid'],Halo0Start)
#     Halo0ISMToISMPercentage = 100.* (np.shape(Halo0ISMStart)[0]/statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['TracersFull']['data'][0])
#     halo0ISM.append(float(Halo0ISMToISMPercentage))
#         #------------------------------------------------------------------------------#
#
#         #               Analyse Tracers moving from Halo 0 NON-ISM Gas back to ISM
#         #------------------------------------------------------------------------------#
#     Halo0NonISMStart = np.where(np.isin(gastoISMintersectTrids,statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['%ISM']['trid'])==False)[0]
#     Halo0NonISMToISMPercentage = 100.* (np.shape(Halo0NonISMStart)[0]/statsDict[(f"T{T}", f"{int(TRACERSPARAMS['snapMin'])}")]['TracersFull']['data'][0])
#     halo0NonISM.append(float(Halo0NonISMToISMPercentage))
# gasToismDF = pd.DataFrame({"T" : np.array(Tlst).astype('float64'), "%OtherHalo-To-ISM" : otherHalo, "%NoHalo-To-ISM" : noHalo,\
#  "%Unbound-To-ISM" : unbound, "%Halo0-ISM-To-ISM" : halo0ISM, "%Halo0-NonISM-To-ISM" : halo0NonISM })
#
# savePath = DataSavepath + f"_Gas-to-ISM-snap{int(TRACERSPARAMS['snapMin'])}" +"_Statistics-Table.csv"
#
# print("\n"+f"Saving Stats table .csv as {savePath}")
#
# gasToismDF.to_csv(savePath,index=False)
