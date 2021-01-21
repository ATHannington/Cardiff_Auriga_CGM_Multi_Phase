import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import AutoMinorLocator

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

ageUniverse = 13.77 #[Gyr]
xsize = 30.
ysize = 10.
DPI = 250
colourmapMain = "plasma"
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

FlatDataDict = {}
for T in Tlst:
    loadPath = DataSavepath + f"_T{T}_flat-wrt-time"+ DataSavepathSuffix

    tmp = hdf5_load(loadPath)
    tmp[f"T{T}"].update({'Ntracers' : np.shape(tmp[f"T{T}"]['type'])[1]})
    FlatDataDict.update(tmp)
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

#------------------------------------------------------------------------------#
#
#------------------------------------------------------------------------------#

def flat_analyse_time_averages(FlatDataDict,Tlst,snapsRange,lookbackData,TRACERSPARAMS):

    gas = []
    heating = []
    cooling = []
    aboveZ = []
    belowZ = []
    inflow = []
    statflow = []
    outflow = []
    halo0 = []
    unbound = []
    otherHalo = []
    noHalo = []
    stars = []
    wind = []
    ism = []

    preselectInd = np.where(snapsRange < int(TRACERSPARAMS['selectSnap']))[0]
    postselectInd = np.where(snapsRange > int(TRACERSPARAMS['selectSnap']))[0]
    for T in Tlst:
        Tkey = f"T{T}"
        print(Tkey)


        data = FlatDataDict[Tkey]
        ntracers = FlatDataDict[Tkey]['Ntracers']

        print("Gas")
        rowspre, colspre = np.where(FlatDataDict[Tkey]['type'][preselectInd,:] == 0)
        gaspre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where(FlatDataDict[Tkey]['type'][postselectInd,:] == 0)
        gaspost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        gas.append([gaspre,gaspost])

        print("Heating & Cooling")
        rowspre, colspre = np.where(FlatDataDict[Tkey]['T'][preselectInd[:-2],:]> 10.**(float(T) + float(TRACERSPARAMS['deltaT'])))
        coolingpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where(FlatDataDict[Tkey]['T'][postselectInd,:][:2]> 10.**(float(T) + float(TRACERSPARAMS['deltaT'])))
        heatingpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        rowspre, colspre = np.where(FlatDataDict[Tkey]['T'][preselectInd,:][:-2]< 10.**(float(T) - float(TRACERSPARAMS['deltaT'])))
        heatingpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where(FlatDataDict[Tkey]['T'][postselectInd,:][:2]< 10.**(float(T) - float(TRACERSPARAMS['deltaT'])))
        coolingpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        cooling.append([coolingpre,coolingpost])
        heating.append([heatingpre,heatingpost])

        print("Z")
        data = FlatDataDict[Tkey]['gz'][preselectInd,:]
        weights = FlatDataDict[Tkey]['mass'][preselectInd,:]
        zPreDat = []
        for (dat,wei) in zip(data.T,weights.T):
            zPreDat.append(weightedperc(dat, wei, 50,"Z-Pre"))
        zPreDat = np.array(zPreDat)

        data = FlatDataDict[Tkey]['gz'][postselectInd,:]
        weights = FlatDataDict[Tkey]['mass'][postselectInd,:]
        zPostDat = []
        for (dat,wei) in zip(data.T,weights.T):
            zPostDat.append(weightedperc(dat, wei, 50,"Z-Post"))
        zPostDat = np.array(zPostDat)

        colspre = np.where(zPreDat>  0.75)[0]
        aboveZpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        colspost = np.where(zPostDat >  0.75)[0]
        aboveZpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        aboveZ.append([aboveZpre,aboveZpost])

        colspre = np.where(zPreDat<  0.75)[0]
        belowZpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        colspost = np.where(zPostDat <  0.75)[0]
        belowZpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        belowZ.append([belowZpre,belowZpost])

        print("Radial-Flow")
        data = FlatDataDict[Tkey]['vrad'][preselectInd,:]
        weights = FlatDataDict[Tkey]['mass'][preselectInd,:]
        vradPreDat = []
        for (dat,wei) in zip(data.T,weights.T):
            vradPreDat.append(weightedperc(dat, wei, 50,"Vrad-Pre"))
        vradPreDat = np.array(vradPreDat)

        data = FlatDataDict[Tkey]['vrad'][postselectInd,:]
        weights = FlatDataDict[Tkey]['mass'][postselectInd,:]
        vradPostDat = []
        for (dat,wei) in zip(data.T,weights.T):
            vradPostDat.append(weightedperc(dat, wei, 50,"Vrad-Post"))
        vradPostDat = np.array(vradPostDat)

        epsilon = 50.

        colspre = np.where(vradPreDat < 0.0 - epsilon)[0]
        inflowpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        colspost = np.where(vradPostDat <  0.0 - epsilon)[0]
        inflowpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        inflow.append([inflowpre,inflowpost])

        colspre = np.where((vradPreDat >= 0.0 - epsilon)&(vradPreDat <= 0.0 + epsilon))[0]
        statflowpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        colspost = np.where((vradPostDat >= 0.0 - epsilon)&(vradPostDat <= 0.0 + epsilon))[0]
        statflowpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        statflow.append([statflowpre,statflowpost])

        colspre = np.where(vradPreDat > 0.0 + epsilon)[0]
        outflowpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        colspost = np.where(vradPostDat >  0.0 + epsilon)[0]
        outflowpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        outflow.append([outflowpre,outflowpost])

        print("Halo0")
        rowspre, colspre = np.where(FlatDataDict[Tkey]['SubHaloID'][preselectInd,:] == int(TRACERSPARAMS['haloID']))
        halo0pre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where(FlatDataDict[Tkey]['SubHaloID'][postselectInd,:] ==int(TRACERSPARAMS['haloID']))
        halo0post = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        halo0.append([halo0pre,halo0post])

        print("Unbound")
        rowspre, colspre = np.where(FlatDataDict[Tkey]['SubHaloID'][preselectInd,:] == -1)
        unboundpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where(FlatDataDict[Tkey]['SubHaloID'][postselectInd,:] == -1)
        unboundpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        unbound.append([unboundpre,unboundpost])

        print("OtherHalo")
        rowspre, colspre = np.where((FlatDataDict[Tkey]['SubHaloID'][preselectInd,:] !=int(TRACERSPARAMS['haloID']))&\
        (FlatDataDict[Tkey]['SubHaloID'][preselectInd,:] != -1 ) &\
        (np.isnan(FlatDataDict[Tkey]['SubHaloID'][preselectInd,:]) == False))
        otherHalopre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where((FlatDataDict[Tkey]['SubHaloID'][postselectInd,:] !=int(TRACERSPARAMS['haloID']))&\
        (FlatDataDict[Tkey]['SubHaloID'][postselectInd,:] != -1 ) &\
        (np.isnan(FlatDataDict[Tkey]['SubHaloID'][postselectInd,:]) == False))
        otherHalopost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        otherHalo.append([otherHalopre,otherHalopost])

        print("NoHalo")
        rowspre, colspre = np.where((FlatDataDict[Tkey]['SubHaloID'][preselectInd,:] !=int(TRACERSPARAMS['haloID']))&\
        (FlatDataDict[Tkey]['SubHaloID'][preselectInd,:] != -1 ) &\
        (np.isnan(FlatDataDict[Tkey]['SubHaloID'][preselectInd,:]) == True))
        noHalopre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where((FlatDataDict[Tkey]['SubHaloID'][postselectInd,:] !=int(TRACERSPARAMS['haloID']))&\
        (FlatDataDict[Tkey]['SubHaloID'][postselectInd,:] != -1 ) &\
        (np.isnan(FlatDataDict[Tkey]['SubHaloID'][postselectInd,:]) == True))
        noHalopost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        noHalo.append([noHalopre,noHalopost])

        print("Stars")
        rowspre, colspre = np.where((FlatDataDict[Tkey]['type'][preselectInd,:] == 4)&\
        (FlatDataDict[Tkey]['age'][preselectInd,:] >= 0.))
        starspre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where((FlatDataDict[Tkey]['type'][postselectInd,:] == 4)&\
        (FlatDataDict[Tkey]['age'][postselectInd,:] >= 0.))
        starspost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        stars.append([starspre,starspost])

        print("Wind")
        rowspre, colspre = np.where((FlatDataDict[Tkey]['type'][preselectInd,:] == 4)&\
        (FlatDataDict[Tkey]['age'][preselectInd,:] < 0.))
        windpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where((FlatDataDict[Tkey]['type'][postselectInd,:] == 4)&\
        (FlatDataDict[Tkey]['age'][postselectInd,:] < 0.))
        windpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        wind.append([windpre,windpost])

        print("ISM")
        rowspre, colspre = np.where((FlatDataDict[Tkey]['R'][preselectInd,:] <= 25.0)&\
        (FlatDataDict[Tkey]['sfr'][preselectInd,:] > 0.))
        ismpre = 100.* float(np.shape(np.unique(colspre))[0])/float(ntracers)
        rowspost, colspost = np.where((FlatDataDict[Tkey]['R'][postselectInd,:] <= 25.0)&\
        (FlatDataDict[Tkey]['sfr'][postselectInd,:] > 0.))
        ismpost = 100.* float(np.shape(np.unique(colspost))[0])/float(ntracers)
        ism.append([ismpre,ismpost])

    out = {"%Gas": {"Pre-Selection" : np.array(gas)[:,0],"Post-Selection" : np.array(gas)[:,1]} , \
    "%Halo0": {"Pre-Selection" : np.array(halo0)[:,0],"Post-Selection" : np.array(halo0)[:,1]} , \
    "%Unbound": {"Pre-Selection" : np.array(unbound)[:,0],"Post-Selection" : np.array(unbound)[:,1]} , \
    "%OtherHalo": {"Pre-Selection" : np.array(otherHalo)[:,0],"Post-Selection" : np.array(otherHalo)[:,1]} , \
    "%NoHalo": {"Pre-Selection" : np.array(noHalo)[:,0],"Post-Selection" : np.array(noHalo)[:,1]} , \
    "%Stars": {"Pre-Selection" : np.array(stars)[:,0],"Post-Selection" : np.array(stars)[:,1]} , \
    "%Wind": {"Pre-Selection" : np.array(wind)[:,0],"Post-Selection" : np.array(wind)[:,1]} , \
    "%ISM": {"Pre-Selection" : np.array(ism)[:,0],"Post-Selection" : np.array(ism)[:,1]} , \
    "%Inflow": {"Pre-Selection" : np.array(inflow)[:,0],"Post-Selection" : np.array(inflow)[:,1]} , \
    "%Radially-Static": {"Pre-Selection" : np.array(statflow)[:,0],"Post-Selection" : np.array(statflow)[:,1]} , \
    "%Outflow": {"Pre-Selection" : np.array(outflow)[:,0],"Post-Selection" : np.array(outflow)[:,1]} , \
    "%Above3/4(Z_solar)": {"Pre-Selection" : np.array(aboveZ)[:,0],"Post-Selection" : np.array(aboveZ)[:,1]} , \
    "%Below3/4(Z_solar)": {"Pre-Selection" : np.array(belowZ)[:,0],"Post-Selection" : np.array(belowZ)[:,1]} , \
    "%Heating": {"Pre-Selection" : np.array(heating)[:,0],"Post-Selection" : np.array(heating)[:,1]} , \
    "%Cooling": {"Pre-Selection" : np.array(cooling)[:,0],"Post-Selection" : np.array(cooling)[:,1]} }

    dict_of_df = {k: pd.DataFrame(v) for k,v in out.items()}
    df1 = pd.concat(dict_of_df, axis=1)

    df2 = pd.DataFrame({"T" : Tlst})
    df = pd.concat([df2,df1], axis=1, join='outer',sort=False)
    df = df.set_index('T')
    return df
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
def fullData_analyse_tracer_averages(dataDict,Tlst,snapsRange):
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

################################################################################
##                           MAIN PROGRAM                                   ####
################################################################################
print("Analyse Data!")
#------------------------------------------------------------------------------#
#               Analyse statistics for all T and snaps
#------------------------------------------------------------------------------#
statsDict, statsDF = fullData_analyse_tracer_averages(dataDict,Tlst,snapsRange)

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

lookbackData = []

for snap in snapsRange:
    lookbackData.append(dataDict[(f"T{Tlst[0]}", f"{int(snap)}")]['Lookback'][0])
    if (int(snap) == int(TRACERSPARAMS['selectSnap'])):
        selectTime = abs( dataDict[(f"T{Tlst[0]}", f"{int(snap)}")]['Lookback'][0] - ageUniverse)
lookbackData = np.array(lookbackData)

timeAvDF = flat_analyse_time_averages(FlatDataDict,Tlst,snapsRange,lookbackData,TRACERSPARAMS)



#Save
savePath = DataSavepath + "_Flat-Statistics-Table.csv"

print("\n"+f"Saving Stats table .csv as {savePath}")

timeAvDF.to_csv(savePath,index=False)



#-------------------------------------------------------------------------------#
#       Plot!!
#-------------------------------------------------------------------------------#
cmap = matplotlib.cm.get_cmap(colourmapMain)
colour = [cmap(float(ii)/float(len(Tlst))) for ii in range(len(Tlst))]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (xsize,ysize), sharey=True)

ax = timeAvDF.T.plot.bar(rot=0,figsize = (xsize,ysize),color=colour)
ax.legend(loc='upper right',title="Log10(T) [K]",fontsize=13)
plt.xticks(rotation=30,ha='right',fontsize=13)
plt.title(f"Percentage of Tracers Ever Meeting Criterion Pre and Post Selection at {selectTime:3.2f} Gyr" +\
"\n"+ r"selected by $T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']), fontsize=16)


plt.annotate(text="Ever Matched Feature", xy=(0.25,0.02), xytext=(0.25,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
plt.annotate(text="", xy=(0.05,0.01), xytext=(0.49,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)

plt.annotate(text="Median Matched Feature", xy=(0.60,0.02), xytext=(0.60,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
plt.annotate(text="", xy=(0.51,0.01), xytext=(0.825,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)

plt.annotate(text="+/-2 Time-steps Matched Feature", xy=(0.85,0.02), xytext=(0.85,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
plt.annotate(text="", xy=(0.835,0.01), xytext=(1.00,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
fig.transFigure

ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both')
plt.grid(which='both',axis='y')
plt.ylabel('% of Tracers Selected Following Feature')
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom = 0.25, left=0.10, right=0.95)


opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{TRACERSPARAMS['Rinner']}R{TRACERSPARAMS['Router']}_Stats-Bars.pdf"
plt.savefig(opslaan, dpi = DPI, transparent = False)
print(opslaan)
plt.close()

################################################################################
#       split Plot
###############################################################################

cols = timeAvDF.columns.values
preDF = timeAvDF[cols[::2].tolist()]
postDF = timeAvDF[cols[1::2].tolist()]

newcols = {}
for name in cols[::2]:
    newcols.update({name : name[0]})

preDF = preDF.rename(columns=newcols)

newcols = {}
for name in cols[1::2]:
    newcols.update({name : name[0]})
postDF = postDF.rename(columns=newcols)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (int(xsize/2.),ysize), sharey=True)

preDF.T.plot.bar(rot=0,ax=ax,color=colour)

ax.legend(loc='upper right',title="Log10(T) [K]",fontsize=13)
plt.xticks(rotation=90,ha='right',fontsize=13)
plt.title(f"Percentage of Tracers Ever Meeting Criterion Pre Selection at {selectTime:3.2f} Gyr" +\
"\n"+ r"selected by $T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']), fontsize=16)


plt.annotate(text="", xy=(0.10,0.25), xytext=(0.10,0.05), arrowprops=dict(arrowstyle='-'), xycoords=fig.transFigure, annotation_clip =False)
plt.annotate(text="Ever Matched Feature", xy=(0.20,0.02), xytext=(0.20,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
plt.annotate(text="", xy=(0.10,0.01), xytext=(0.55,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
plt.annotate(text="", xy=(0.55,0.25), xytext=(0.55,0.05), arrowprops=dict(arrowstyle='-'), xycoords=fig.transFigure, annotation_clip =False)

plt.annotate(text="Median Matched Feature", xy=(0.62,0.02), xytext=(0.62,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
plt.annotate(text="", xy=(0.56,0.01), xytext=(0.85,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
plt.annotate(text="", xy=(0.85,0.25), xytext=(0.85,0.05), arrowprops=dict(arrowstyle='-'), xycoords=fig.transFigure, annotation_clip =False)

plt.annotate(text="+/-2 Time-steps \n Matched Feature", xy=(0.85,0.03), xytext=(0.85,0.03), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
plt.annotate(text="", xy=(0.86,0.01), xytext=(0.95,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
plt.annotate(text="", xy=(0.95,0.25), xytext=(0.95,0.05), arrowprops=dict(arrowstyle='-'), xycoords=fig.transFigure, annotation_clip =False)

fig.transFigure

ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both')
plt.grid(which='both',axis='y')
plt.ylabel('% of Tracers Selected Following Feature')
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom = 0.25, left=0.10, right=0.95)

opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{TRACERSPARAMS['Rinner']}R{TRACERSPARAMS['Router']}_Pre-Stats-Bars.pdf"
plt.savefig(opslaan, dpi = DPI, transparent = False)
print(opslaan)
plt.close()




fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (int(xsize/2.),ysize), sharey=True)

postDF.T.plot.bar(rot=0,ax=ax,color=colour)

ax.legend(loc='upper right',title="Log10(T) [K]",fontsize=13)
plt.xticks(rotation=90,ha='right',fontsize=13)
plt.title(f"Percentage of Tracers Ever Meeting Criterion Post Selection at {selectTime:3.2f} Gyr" +\
"\n"+ r"selected by $T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']), fontsize=16)

plt.annotate(text="", xy=(0.10,0.25), xytext=(0.10,0.05), arrowprops=dict(arrowstyle='-'), xycoords=fig.transFigure, annotation_clip =False)
plt.annotate(text="Ever Matched Feature", xy=(0.20,0.02), xytext=(0.20,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
plt.annotate(text="", xy=(0.10,0.01), xytext=(0.55,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
plt.annotate(text="", xy=(0.55,0.25), xytext=(0.55,0.05), arrowprops=dict(arrowstyle='-'), xycoords=fig.transFigure, annotation_clip =False)

plt.annotate(text="Median Matched Feature", xy=(0.62,0.02), xytext=(0.62,0.02), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
plt.annotate(text="", xy=(0.56,0.01), xytext=(0.85,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
plt.annotate(text="", xy=(0.85,0.25), xytext=(0.85,0.05), arrowprops=dict(arrowstyle='-'), xycoords=fig.transFigure, annotation_clip =False)

plt.annotate(text="+/-2 Time-steps \n Matched Feature", xy=(0.85,0.03), xytext=(0.85,0.03), textcoords=fig.transFigure, annotation_clip =False, fontsize=14)
plt.annotate(text="", xy=(0.86,0.01), xytext=(0.95,0.01), arrowprops=dict(arrowstyle='<->'), xycoords=fig.transFigure, annotation_clip =False)
plt.annotate(text="", xy=(0.95,0.25), xytext=(0.95,0.05), arrowprops=dict(arrowstyle='-'), xycoords=fig.transFigure, annotation_clip =False)

fig.transFigure

ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both')
plt.grid(which='both',axis='y')
plt.ylabel('% of Tracers Selected Following Feature')
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom = 0.25, left=0.10, right=0.95)

opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_{TRACERSPARAMS['Rinner']}R{TRACERSPARAMS['Router']}_Post-Stats-Bars.pdf"
plt.savefig(opslaan, dpi = DPI, transparent = False)
print(opslaan)
plt.close()
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
