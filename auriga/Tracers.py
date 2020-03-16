import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
from Snapper import *


simfile='/home/universe/spxtd1-shared/ISOTOPES/output/' # set paths to simulation
snapnumMAX = 127

snapnum=127 # set snapshot to look at
snapnumDelta =  2

targetTLst= [4.,5.,6.] # [10^{target +- delta}]
deltaT = 0.25

Rinner = 25.0 #[kpc]
Router = 75.0 #[kpc]


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
for targetT in targetTLst:


    NTemps = float(len(targetTLst))
    percentage = (float(kk)/NTemps)*100.0

    kk+=1

    print("")
    print(f"{percentage:0.02f}%")
    print("Setting Condition!")
    # load in the subfind group files
    snap_subfind = load_subfind(snapnum,dir=simfile)

    # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
    snapGas     = gadget_readsnap(snapnum, simfile, hdf5=True, loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
    snapTracers = gadget_readsnap(snapnum, simfile, hdf5=True, loadonlytype = [6], lazy_load=True)

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
    Cond = np.where((snapGas.data['T']>=1.*10**(targetT-deltaT)) & \
                    (snapGas.data['T']<=1.*10**(targetT+deltaT)) & \
                    (snapGas.data['R']>=Rinner) & \
                    (snapGas.data['R']<=Router) &
                    (snapGas.data['sfr']<=0)\
                   )

    Tracers, CellsTFC, CellIDsTFC = GetTracersFromCells(snapGas, snapTracers,Cond)
    print(f"min T = {np.min(CellsTFC['T']):0.02e}")
    print(f"max T = {np.max(CellsTFC['T']):0.02e}")
    dataDict = {}
    IDDict = {}

    for ii in range(snapnum-snapnumDelta, min(snapnumMAX+1, snapnum+snapnumDelta+1)):
        print("")
        print(f"Starting Snap {ii}")

        # Single FvdV Projection:
        # load in the subfind group files
        snap_subfind = load_subfind(ii,dir=simfile)

        # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
        snapGas     = gadget_readsnap(ii, simfile, hdf5=True, loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
        snapTracers = gadget_readsnap(ii, simfile, hdf5=True, loadonlytype = [6], lazy_load=True)

        print(f" SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

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
            if ind == 0:
                plotData.update({f"{k}mean": np.mean(v)})
                plotData.update({f"{k}std": np.std(v)})
            else:
                plotData[f"{k}mean"] = np.append(plotData[f"{k}mean"],np.mean(v))
                plotData[f"{k}std"] = np.append(plotData[f"{k}std"],np.std(v))


    #Set style options
    opacity = 0.25

    #Select a Temperature specific colour from colourmap
    cmap = matplotlib.cm.get_cmap('viridis')
    colour = cmap((float(kk)/NTemps)-0.05)

    print(f"TMean = {plotData['Tmean']}")
    print(f"Tstd = {plotData['Tstd']}")

    print("")
    print("Temperature Sub-Plot!")
    ax.fill_between(plotData['Lookbackmean'], plotData['Tmean']+plotData['Tstd'], plotData['Tmean']-plotData['Tstd'],\
     facecolor=colour,alpha=opacity,interpolate=True)
    ax.plot(plotData['Lookbackmean'],plotData['Tmean'],label=r"$T = 10^{%05.02f} K$"%targetTLst[kk-1], color = colour)
    # ax.set_yscale('log')
    ax.set_xlabel(r"Lookback Time [$Gyrs$]")
    ax.set_ylabel(r"Temperature [$K$]")
    ax.set_title(f"Cells Containing Tracers selected by: " +\
     "\n"+ r"$T = 10^{n \pm %05.2f} K$"%(deltaT) +\
     r" and $%05.2f \leq R \leq %05.2f kpc $"%(Rinner, Router))
    plt.legend()



opslaan = f'Tracers.png'
plt.savefig(opslaan, dpi = 500, transparent = False)
print(opslaan)
plt.close()
