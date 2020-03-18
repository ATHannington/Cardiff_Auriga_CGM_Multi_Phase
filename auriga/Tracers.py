import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
from Snapper import *
from Tracers_Subroutines import *

saveKeys = ['Tmedian','TUP','TLO','Lookbackmedian']


TRACERSPARAMS = pd.read_csv('TracersParams.csv', delimiter=" ", header=None, \
usecols=[0,1],skipinitialspace=True, index_col=0, comment="#").to_dict()[1]

for key, value in TRACERSPARAMS.items():
    if ((key != 'targetTLst') & (key != 'simfile')):
        TRACERSPARAMS.update({key:float(value)})
    elif ((key == 'targetTLst') & (key != 'simfile')):
        lst = value.split(",")
        lst2 = [float(item) for item in lst]
        TRACERSPARAMS.update({key:lst2})
    elif ((key != 'targetTLst') & (key == 'simfile')):
        TRACERSPARAMS.update({key:value})

Tlst = [str(int(item)) for item in TRACERSPARAMS['targetTLst']]
Tstr = '-'.join(Tlst)

DataSavepath = f"Data_snap{int(TRACERSPARAMS['snapnum'])}_min{int(TRACERSPARAMS['snapMin'])}_max{int(TRACERSPARAMS['snapMax'])}" +\
    f"_{int(TRACERSPARAMS['Rinner'])}R{int(TRACERSPARAMS['Router'])}_targetT{Tstr}"+\
    f"_deltaT{int(TRACERSPARAMS['deltaT'])}"

DataSavepathSuffix = f".csv"

#==============================================================================#
#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
elements_mass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

Zsolar = 0.0127

omegabaryon0 = 0.048
#==============================================================================#


fig = plt.figure()
ax = plt.gca()

kk = 0
for targetT in TRACERSPARAMS['targetTLst']:


    NTemps = float(len(TRACERSPARAMS['targetTLst']))
    percentage = (float(kk)/NTemps)*100.0

    kk+=1

    print("")
    print(f"{percentage:0.02f}%")
    print("Setting Condition!")
    # load in the subfind group files
    snap_subfind = load_subfind(TRACERSPARAMS['snapnum'],dir=TRACERSPARAMS['simfile'])

    # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
    snapGas     = gadget_readsnap(TRACERSPARAMS['snapnum'], TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
    snapTracers = gadget_readsnap(TRACERSPARAMS['snapnum'], TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [6], lazy_load=True)

    print(f" SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

    Snapper1 = Snapper()
    snapGas     = Snapper1.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=0)

    #Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos   *= 1e3 #[kpc]

    snapGas.vol *= 1e9 #[kpc^3]

    meanweight = sum(snapGas.gmet[:,0:9], axis = 1) / ( sum(snapGas.gmet[:,0:9]/elements_mass[0:9], axis = 1) + snapGas.ne*snapGas.gmet[:,0] )
    Tfac = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53
    snapGas.data['T'] = snapGas.u / Tfac # K

    #--------------------------------------------------------------------------#
    ####                    SELECTION                                        ###
    #--------------------------------------------------------------------------#

    snapGas.data['R'] =  np.linalg.norm(snapGas.data['pos'], axis=1)
    Cond = np.where((snapGas.data['T']>=1.*10**(targetT-TRACERSPARAMS['deltaT'])) & \
                    (snapGas.data['T']<=1.*10**(targetT+TRACERSPARAMS['deltaT'])) & \
                    (snapGas.data['R']>=TRACERSPARAMS['Rinner']) & \
                    (snapGas.data['R']<=TRACERSPARAMS['Router']) &
                    (snapGas.data['sfr']<=0)\
                   )

    Tracers, CellsTFC, CellIDsTFC = GetTracersFromCells(snapGas, snapTracers,Cond)
    print(f"min T = {np.min(CellsTFC['T']):0.02e}")
    print(f"max T = {np.max(CellsTFC['T']):0.02e}")
    dataDict = {}
    IDDict = {}

    for ii in range(int(TRACERSPARAMS['snapMin']), int(min(TRACERSPARAMS['snapnumMAX']+1, TRACERSPARAMS['snapMax']+1))):
        print("")
        print(f"Starting Snap {ii}")

        # Single FvdV Projection:
        # load in the subfind group files
        snap_subfind = load_subfind(ii,dir=TRACERSPARAMS['simfile'])

        # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
        snapGas     = gadget_readsnap(ii, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
        snapTracers = gadget_readsnap(ii, TRACERSPARAMS['simfile'], hdf5=True, loadonlytype = [6], lazy_load=True)

        print(f"SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

        Snapper1 = Snapper()
        snapGas  = Snapper1.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=0)

        #Convert Units
        ## Make this a seperate function at some point??
        snapGas.pos   *= 1e3 #[kpc]

        snapGas.vol *= 1e9 #[kpc^3]

        meanweight = sum(snapGas.gmet[:,0:9], axis = 1) / ( sum(snapGas.gmet[:,0:9]/elements_mass[0:9], axis = 1) + snapGas.ne*snapGas.gmet[:,0] )
        Tfac = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53
        snapGas.data['T'] = snapGas.u / Tfac # K


        ###
        ##  Selection   ##
        ###

        CellsCFT, CellIDsCFT = GetCellsFromTracers(snapGas, snapTracers,Tracers)

        print("Lookback")
        #Redshift
        redshift = snapGas.redshift        #z
        aConst = 1. / (1. + redshift)   #[/]

        #[0] to remove from numpy array for purposes of plot title
        lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[0] #[Gyrs]
        CellsCFT['Lookback']=[lookback for jj in range(0,len(CellsCFT['T']))]

        print("Adding to Dict")
        dataDict.update({f"{ii}":CellsCFT})
        IDDict.update({f"ID{ii}":CellIDsCFT})

        del CellsCFT, CellIDsCFT, snapGas, snapTracers, Snapper1, snap_subfind

    #------------------------------------------------------------------------------#
    #       Flatten dict and take subset
    #------------------------------------------------------------------------------#

    plotData = {}

    for ind, (key, value) in enumerate(dataDict.items()):
        for k, v in value.items():
            if (k == 'T'):
                if ind == 0:
                    plotData.update({f"{k}median": \
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=50.)})
                    plotData.update({f"{k}UP": \
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=TRACERSPARAMS['percentileUP'])})
                    plotData.update({f"{k}LO": \
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=TRACERSPARAMS['percentileLO'])})
                else:
                    plotData[f"{k}median"] = np.append(plotData[f"{k}median"],\
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=50.))
                    plotData[f"{k}UP"] = np.append(plotData[f"{k}UP"],\
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=TRACERSPARAMS['percentileUP']))
                    plotData[f"{k}LO"] = np.append(plotData[f"{k}LO"],\
                    weightedperc(data=v, weights=dataDict[key]['mass'],perc=TRACERSPARAMS['percentileLO']))
            elif (k=='Lookback'):
                if ind == 0:
                    plotData.update({f"{k}median": np.median(v)})
                else:
                    plotData[f"{k}median"] = np.append(plotData[f"{k}median"],\
                     np.median(v))
            else:
                if ind == 0 :
                    plotData.update({f"{k}": v})
                else:
                    plotData[f"{k}"] = np.append(plotData[f"{k}"], v)

    tmpSave = DataSavepath + f"_T{int(targetT)}" + DataSavepathSuffix
    print(tmpSave)

    tmpData = {}
    for key in saveKeys:
        tmpData.update({key : plotData[key]})

    df = pd.DataFrame.from_dict(tmpData, orient="index")
    df.to_csv(tmpSave)
