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

dtwJoint = True

ignoreParams = ['clusters', 'd_crit', 'maxmimally_distinct_bool', 'sort_level']

#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

dtwParams = TRACERSPARAMS['dtwParams']
saveParams = TRACERSPARAMS['saveParams']
logParams = ['dens','rho_rhomean','csound','T','n_H','B','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

paramstring = "+".join(dtwParams)

print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

loadPath = DataSavepath + f"_flat-wrt-time"+ DataSavepathSuffix

FlatDataDict = hdf5_load(loadPath)

paramStatsDict = {}
dtwDict ={}
for T in Tlst:
    paramStatsDict.update({ f"T{T}": Statistics_hdf5_load(T,DataSavepath,TRACERSPARAMS,DataSavepathSuffix)})

    if (dtwJoint == True):
        loadPath = DataSavepath + f"_T{T}_{paramstring}_Joint-DTW-clusters"+DataSavepathSuffix

        dtwDict.update(hdf5_load(loadPath))

print("Analyse Data!")
snapsRange = np.array([ xx for xx in range(int(TRACERSPARAMS['snapMin']), min(int(TRACERSPARAMS['snapMax'])+1,int(TRACERSPARAMS['finalSnap'])+1),1)])

################################################################################
##              Full (not clustered) Data                                   ####
################################################################################


StatsEntries = ['ClusterID','Snap','Lookback',"%Tracers",'%Halo0','%Unbound','%OtherHalo',"%NoHalo",'%Stars',"%ISM",'%Wind',"%Inflow","%Outflow","%AboveZsolar","%BelowZsolar","%Heating","%Cooling"]

def _inner_analysis(dataDict,T,ClusterID,FullKey):

    lookback = []
    temp = []
    cluster = []
    tracers = []
    halo0 = []
    unbound =[]
    otherHalo =[]
    noHalo = []
    gas = []
    stars = []
    ism = []
    wind = []
    inflow = []
    outflow = []

    NtracersFull = dataDict[FullKey]['Ntracers'][0]
    tracers.append(100.)

    SubHalo = dataDict[FullKey]['SubHaloID']

    whereGas = np.where(dataDict[FullKey]['type']==0)[0]

    Ntracers = np.shape(whereGas)[0]

    gas.append(100.*(Ntracers/NtracersFull))

    halo0.append(100.*(np.shape(np.where(SubHalo[whereGas]==int(TRACERSPARAMS['haloID']))[0])[0]/Ntracers))

    unbound.append(100.*(np.shape(np.where(SubHalo[whereGas]==-1)[0])[0]/Ntracers))

    otherHalo.append(100.*(np.shape(np.where((SubHalo[whereGas]!=int(TRACERSPARAMS['haloID']))\
        &(SubHalo[whereGas]!=-1)&(np.isnan(SubHalo[whereGas])==False))[0])[0]/Ntracers))

    noHalo.append(100.*(np.shape(np.where((SubHalo[whereGas]!=int(TRACERSPARAMS['haloID']))\
        &(SubHalo[whereGas]!=-1)&(np.isnan(SubHalo[whereGas])==True))[0])[0]/Ntracers))

    stars.append(100.*(np.shape(np.where((dataDict[FullKey]['age']>0)&(dataDict[FullKey]['type']==4))[0])[0]/NtracersFull))

    wind.append(100.*(np.shape(np.where((dataDict[FullKey]['age']<0)&(dataDict[FullKey]['type']==4))[0])[0]/NtracersFull))

    ism.append(100.*(np.shape(np.where((dataDict[FullKey]['sfr']>0)&(dataDict[FullKey]['R']<=25.))[0])[0]/NtracersFull))

    inflow.append(100.*(np.shape(np.where(dataDict[FullKey]['vrad'][whereGas]<0.)[0])[0]/Ntracers))
    outflow.append(100.*(np.shape(np.where(dataDict[FullKey]['vrad'][whereGas]>0.)[0])[0]/Ntracers))

    cluster.append(ClusterID)
    temp.append(T)
    lookback.append(dataDict[FullKey]['Lookback'][0])

    return  lookback, temp, cluster ,tracers, halo0 , unbound, otherHalo ,noHalo , gas, stars, ism , wind ,inflow, outflow

def fullData_analyse_cluster(dataDict,paramStatsDict,Tlst,snapsRange,ClusterID=-1):
    dfList = []
    for T in Tlst:
        Tkey = f"T{T}"
        innerDF = pd.DataFrame({})
        lookback = []
        temp = []
        cluster = []
        tracers = []
        halo0 = []
        unbound =[]
        otherHalo =[]
        noHalo = []
        gas = []
        stars = []
        ism = []
        wind = []
        inflow = []
        outflow = []
        for snap in snapsRange:
            FullKey = (f"T{T}",f"{int(snap)}")
            tmplookback, tmptemp, tmpcluster ,tmptracers,\
             tmphalo0 , tmpunbound, tmpotherHalo ,tmpnoHalo ,\
              tmpgas, tmpstars, tmpism , tmpwind ,tmpinflow, tmpoutflow \
              = _inner_analysis(dataDict,T,ClusterID, FullKey = FullKey)
            lookback.append(tmplookback[0])
            temp.append(tmptemp[0])
            cluster.append(tmpcluster[0])
            tracers.append(tmptracers[0])
            halo0.append(tmphalo0[0])
            unbound.append(tmpunbound[0])
            otherHalo.append(tmpotherHalo[0])
            noHalo.append(tmpnoHalo[0])
            gas.append(tmpgas[0])
            stars.append(tmpstars[0])
            ism.append(tmpism[0])
            wind.append(tmpwind[0])
            inflow.append(tmpinflow[0])
            outflow.append(tmpoutflow[0])

        tmp = np.array([cluster,snapsRange.tolist(),lookback,temp,tracers,halo0,unbound,otherHalo,noHalo,gas,stars,wind,ism,inflow,outflow]).T
        df = pd.DataFrame(tmp,columns=['Cluster','Snap','Lookback','T','%Tracers',\
        '%Halo0','%Unbound','%OtherHalo','%NoHalo','%Gas',\
        '%Stars','%Wind','%ISM','%Inflow','%Outflow'])
        df = df.astype('float64')
        df = df.astype({'Cluster' : 'int32', 'Snap': 'int32'})
        innerDF= pd.concat([innerDF,df], axis=1, join='outer',sort=False)

        df = pd.DataFrame(paramStatsDict[Tkey]).astype('float64')
        innerDF = pd.concat([innerDF,df], axis=1, join='outer',sort=False)

        dfList.append(innerDF)

    StatsDF = pd.concat(dfList, axis=0, join='outer',sort=False, ignore_index=True)
    return StatsDF

def clusteredData_analyse_cluster(dataDict,dtwparamStatsDict,Tlst,snapsRange,ignoreParams):

    dfList = []
    for T in Tlst:
        Tkey = f"T{T}"
        clusters =  np.unique(dataDict[Tkey]["clusters"])
        for clusterID in clusters:
            whereCluster = np.where(dataDict[Tkey]["clusters"]==clusterID)[0]

            innerDF = pd.DataFrame({})
            lookback = []
            temp = []
            cluster = []
            tracers = []
            halo0 = []
            unbound =[]
            otherHalo =[]
            noHalo = []
            gas = []
            stars = []
            ism = []
            wind = []
            inflow = []
            outflow = []
            for ii, snap in enumerate(snapsRange):
                tmpsnapdataDict = {}
                for k, v in dataDict[Tkey].items():
                    if(np.shape(np.shape(v))[0]>1):
                        if(np.shape(v)[1]>1):
                            tmpsnapdataDict.update({k : v[ii,:][whereCluster]})
                        else:
                            tmpsnapdataDict.update({k : v[ii]})
                    else:
                        tmpsnapdataDict.update({k : v})
                snapdataDict = {Tkey : tmpsnapdataDict}
                tmplookback, tmptemp, tmpcluster ,tmptracers,\
                 tmphalo0 , tmpunbound, tmpotherHalo ,tmpnoHalo ,\
                  tmpgas, tmpstars, tmpism , tmpwind ,tmpinflow, tmpoutflow \
                  = _inner_analysis(snapdataDict,T,ClusterID=clusterID, FullKey = Tkey)
                lookback.append(tmplookback[0])
                temp.append(tmptemp[0])
                cluster.append(tmpcluster[0])
                tracers.append(tmptracers[0])
                halo0.append(tmphalo0[0])
                unbound.append(tmpunbound[0])
                otherHalo.append(tmpotherHalo[0])
                noHalo.append(tmpnoHalo[0])
                gas.append(tmpgas[0])
                stars.append(tmpstars[0])
                ism.append(tmpism[0])
                wind.append(tmpwind[0])
                inflow.append(tmpinflow[0])
                outflow.append(tmpoutflow[0])

            tmp = np.array([cluster,snapsRange.tolist(),lookback,temp,tracers,halo0,unbound,otherHalo,noHalo,gas,stars,wind,ism,inflow,outflow]).T
            df = pd.DataFrame(tmp,columns=['Cluster','Snap','Lookback','T','%Tracers',\
            '%Halo0','%Unbound','%OtherHalo','%NoHalo','%Gas',\
            '%Stars','%Wind','%ISM','%Inflow','%Outflow'])
            df = df.astype('float64')
            df = df.astype({'Cluster' : 'int32', 'Snap': 'int32'})
            innerDF= pd.concat([innerDF,df], axis=1, join='outer',sort=False)

            df = pd.DataFrame(dtwparamStatsDict[Tkey]).astype('float64')
            innerDF = pd.concat([innerDF,df], axis=1, join='outer',sort=False)

            dfList.append(innerDF)

    StatsDF = pd.concat(dfList, axis=0, join='outer',sort=False, ignore_index=True)
    return StatsDF

StatsDF = fullData_analyse_cluster(dataDict,paramStatsDict,Tlst,snapsRange,ClusterID=-1)

clusteredDict = {}
for T in Tlst:
    key = f"T{T}"
    print(key)
    dictlist = []
    for ii, snap in enumerate(snapsRange):
        FullKey = (f"T{T}",f"{int(snap)}")
        _, tridIndices,dtwIndices =  np.intersect1d(FlatDataDict[key]['trid'][ii,:],dtwDict[key]['trid'][:,ii],return_indices=True)
        tmp = {}
        for k,v in FlatDataDict[key].items():
            tmp.update({k : v[ii,:][tridIndices]})
        for k,v in dtwDict[key].items():
            if(np.shape(np.shape(v))[0]>1):
                tmp.update({k : v[:,ii][dtwIndices]})
            else:
                tmp.update({k : v})
        FullKey = (f"T{T}",f"{int(snap)}")
        tmp.update({'Lookback' : dataDict[FullKey]['Lookback']})
        tmp.update({'Ntracers' : [np.shape(tmp['type'])[0]] })
        dictlist.append(tmp)
    final = {}
    for d in dictlist:
        for k,v in d.items():
            if k in final:
                entry = final[k]
                entry.append(v)
                final.update({k : entry})
            else:
                final.update({k : [v]})
    for k,v in final.items():
        final.update({k : np.array(v)})
    clusteredDict.update({key : final})

dtwparamStatsDict = {}
for T in Tlst:
    key = f"T{T}"
    print(key)
    dictlist = []
    for ii, snap in enumerate(snapsRange):
        FullKey = (f"T{T}",f"{int(snap)}")
        _, tridIndices,_ =  np.intersect1d(dataDict[FullKey]['trid'],dtwDict[key]['trid'][:,ii],return_indices=True)
        prids = dataDict[FullKey]['prid'][tridIndices]
        _, dataIndices,_ =  np.intersect1d(prids,dataDict[FullKey]['id'],return_indices=True)
        data = {}
        for k, v in dataDict[FullKey].items():
            if (k in saveParams+['mass']):
                data.update({k : v[dataIndices]})
        statsData = save_statistics(data, T, snap, TRACERSPARAMS, saveParams, DataSavepath=DataSavepath +"_DTW-Clusters" ,MiniDataPathSuffix=".csv",saveBool=True)
        dictlist.append(statsData)
    final = {}
    for d in dictlist:
        for k,v in d.items():
            if k in final:
                entry = final[k]
                entry.append(v)
                final.update({k : entry})
            else:
                final.update({k : [v]})
    dtwparamStatsDict.update({key : final})


savePath = DataSavepath + f"_DTW-Flat_Full-Data"+ DataSavepathSuffix
print("\n" + f" Saving Merged Data as: "+ savePath)
hdf5_save(savePath,clusteredDict)

clusterStatsDF = clusteredData_analyse_cluster(clusteredDict,dtwparamStatsDict,Tlst,snapsRange,ignoreParams=ignoreParams)

finalStatsDF = pd.concat([StatsDF,clusterStatsDF], axis=0, join='outer',sort=False, ignore_index=True)

savePath = DataSavepath + "_DTW-Cluster-Statistics-Table.csv"

print("\n"+f"Saving Stats table .csv as {savePath}")

finalStatsDF.to_csv(savePath, index=False)
