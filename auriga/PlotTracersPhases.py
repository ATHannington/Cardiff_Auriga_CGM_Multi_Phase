"""
Author: A. T. Hannington
Created: 21/07/2020

Known Bugs:
    pandas read_csv loading data as nested dict. . Have added flattening to fix

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import *
import h5py
import multiprocessing as mp

Nbins = 1000
xsize = 20.
ysize = 10.
fontsize = 15
DPI = 20

ageUniverse = 13.77 #[Gyr]

#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

colourmap="inferno_r"

#Paramters to weight the 2D hist by
weightKeys = ['mass','tcool','gz']

labelDict={'mass' : r'Log10 Mass per pixel [$M/M_{\odot}$]',\
 'gz':r'Log10 Average Metallicity per pixel $Z/Z_{\odot}$',\
 'tcool': r'Log10 Cooling Time per pixel [$Gyr$]',\
 }

#==============================================================================#
#       USER DEFINED PARAMETERS
#==============================================================================#
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

#File types for data save.
#   Mini: small median and percentiles data
#   Full: full FullDict data
MiniDataPathSuffix = f".h5"
FullDataPathSuffix = f".h5"

#Lazy Load switch. Set to False to save all data (warning, pickle file may explode)
lazyLoadBool = True

#Number of cores to run on:
n_processes = 4

#Save types, which when combined with saveparams define what data is saved.
#   This is intended to be 'median', 'UP' (upper quartile), and 'LO' (lower quartile)
saveTypes= ['median','UP','LO']
#==============================================================================#
#       Prepare for analysis
#==============================================================================#
# Load in parameters from csv. This ensures reproducability!
#   We save as a DataFrame, then convert to a dictionary, and take out nesting...
    #Save as .csv
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

print("")
print("Loaded Analysis Parameters:")
for key,value in TRACERSPARAMS.items():
    print(f"{key}: {value}")

print("")

#Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = TRACERSPARAMS['saveParams']#['rho_rhomean','dens','T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','csound','tcross','tff','tcool_tff']

print("")
print("Saved Parameters in this Analysis:")
print(saveParams)

#Optional Tracer only (no stats in .csv) parameters to be saved
#   Cannot guarantee that all Plotting and post-processing are independent of these
#       Will attempt to ensure any necessary parameters are stored in ESSENTIALS
saveTracersOnly = TRACERSPARAMS['saveTracersOnly']#['sfr','age']

print("")
print("Tracers ONLY (no stats) Saved Parameters in this Analysis:")
print(saveTracersOnly)

#SAVE ESSENTIALS : The data required to be tracked in order for the analysis to work
saveEssentials = TRACERSPARAMS['saveEssentials']#['FoFHaloID','SubHaloID','Lookback','Ntracers','Snap','id','prid','trid','type','mass','pos']

print("")
print("ESSENTIAL Saved Parameters in this Analysis:")
print(saveEssentials)

saveTracersOnly = saveTracersOnly + saveEssentials

#Combine saveParams and saveTypes to form each combination for saving data types
saveKeys =[]
for param in saveParams:
    for TYPE in saveTypes:
        saveKeys.append(param+TYPE)

#Select Halo of interest:
#   0 is the most massive:
HaloID = int(TRACERSPARAMS['haloID'])
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
#       MAIN PROGRAM
#==============================================================================#


#==============================================================================#
#       Prepare for analysis
#==============================================================================#

#Combine saveParams and saveTypes to form each combination for saving data types
saveKeys =[]
for param in saveParams:
    for TYPE in saveTypes:
        saveKeys.append(param+TYPE)

# #Add saveEssentials to saveKeys so as to save these without the median and quartiles
# #   being taken.
# for key in saveEssentials:
#     saveKeys.append(key)

# Load in parameters from csv. This ensures reproducability!
#   We save as a DataFrame, then convert to a dictionary, and take out nesting...
    #Save as .csv
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

print("")
print("Loaded Analysis Parameters:")
for key,value in TRACERSPARAMS.items():
    print(f"{key}: {value}")

print("")
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
#       MAIN PROGRAM
#==============================================================================#



snapGasFinalDict = {}

for snap in TRACERSPARAMS['phasesSnaps']:
    _, _, _, _, snapGas, _ = \
    tracer_selection_snap_analysis(4.0,TRACERSPARAMS,HaloID,\
    elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
    saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,\
    lazyLoadBool,SUBSET=None,snapNumber=snap,saveTracers=False,loadonlyhalo=False)

    snapGasFinalDict.update({f"{int(snap)}" : snapGas.data})

FullDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,FullDataPathSuffix)
print("Flatten Tracers Data (snapData).")
TracersFinalDict = flatten_wrt_T(FullDict,TRACERSPARAMS)

#------------------------------------------------------------------------------#
#               PLOTTING
#
#------------------------------------------------------------------------------#
for snap in TRACERSPARAMS['phasesSnaps']:
    print("\n"+f"Starting Snap {int(snap)}")
    for weightKey in weightKeys:
        print("\n"+f"Starting weightKey {weightKey}")
        key = f"{int(snap)}"

        FullDictKey = (f"T{float(Tlst[0])}", f"{int(snap)}")
        selectTime = abs(FullDict[FullDictKey]['Lookback'][0] - ageUniverse)

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (xsize,ysize), dpi = DPI)

        whereCellsGas = np.where(snapGasFinalDict[key]['type'] == 0)
        whereTracersGas = np.where(TracersFinalDict[key]['type'] == 0)

        snapGasFinalDict[key]['age'][np.where(np.isnan(snapGasFinalDict[key]['age']) == True)] = 0.
        TracersFinalDict[key]['age'][np.where(np.isnan(TracersFinalDict[key]['age']) == True)] = 0.

        whereCellsStars = np.where((snapGasFinalDict[key]['type'] == 4)&\
                              (snapGasFinalDict[key]['age'] >= 0.))

        whereTracersStars = np.where((TracersFinalDict[key]['type'] == 4)&\
                  (TracersFinalDict[key]['age'] >= 0.))

        whereCellsGas = np.where(snapGasFinalDict[key]['type'] == 0)

        whereTracersGas = np.where(TracersFinalDict[key]['type'] == 0)

        NGasCells = len(snapGasFinalDict[key]['type'][whereCellsGas])
        NStarsCells = len(snapGasFinalDict[key]['type'][whereCellsStars])
        NtotCells = NGasCells + NStarsCells

        #Percentage in stars
        percentageCells = (float(NStarsCells)/(float(NtotCells)))*100.

        NGasTracers = len(TracersFinalDict[key]['type'][whereTracersGas])
        NStarsTracers = len(TracersFinalDict[key]['type'][whereTracersStars])
        NtotTracers = NGasTracers + NStarsTracers

        #Percentage in stars
        percentageTracers = (float(NStarsTracers)/(float(NtotTracers)))*100.



#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#   Figure 1: Full Cells Data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        print(f"snapData Plot!")

        xdataCells = np.log10(snapGasFinalDict[key]['rho_rhomean'][whereCellsGas])
        ydataCells = np.log10(snapGasFinalDict[key]['T'][whereCellsGas])
        massCells = np.log10(snapGasFinalDict[key]['mass'][whereCellsGas]*1e10) #10^10Msol -> Msol
        weightDataCells = np.log10(snapGasFinalDict[key][weightKey][whereCellsGas]) * massCells

        whereweightDataCellsNotNaNorInf = np.where((np.isinf(weightDataCells)==False) & (np.isnan(weightDataCells)==False))

        xdataCells = xdataCells[whereweightDataCellsNotNaNorInf]
        ydataCells = ydataCells[whereweightDataCellsNotNaNorInf]
        massCells  = massCells[whereweightDataCellsNotNaNorInf]
        weightDataCells = weightDataCells[whereweightDataCellsNotNaNorInf]

        mhistCells,_,_=np.histogram2d(xdataCells,ydataCells,bins=Nbins,weights=massCells,normed=False)
        histCells,xedgeCells,yedgeCells=np.histogram2d(xdataCells,ydataCells,bins=Nbins,weights=weightDataCells,normed=False)

        finalHistCells = histCells/mhistCells

        finalHistCells = finalHistCells.T

        if (weightKey != 'gz'):
            zmin = np.nanmin(finalHistCells)
        else:
            zmin = -2.0

        img1 = ax[0].imshow(finalHistCells,cmap=colourmap,vmin=zmin,vmax=np.nanmax(finalHistCells) \
        ,extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')

        ax[0].set_xlabel(r"Log10 Density [$\rho / \langle \rho \rangle $]",fontsize=fontsize)
        ax[0].set_ylabel(r"Log10 Temperatures [$K$]",fontsize=fontsize)

        cax1 = inset_axes(ax[0],width="5%",height="95%",loc='right')
        fig.colorbar(img1, cax = cax1, orientation = 'vertical').set_label(label=labelDict[weightKey],size=fontsize)
        cax1.yaxis.set_ticks_position("left")
        cax1.yaxis.set_label_position("left")
        cax1.yaxis.label.set_color("black")
        cax1.tick_params(axis="y", colors="black",labelsize=fontsize)

        ax[0].set_title(f"Full Simulation Data",fontsize=fontsize)
        ax[0].set_aspect("auto")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#   Figure 2: Tracers Only  Data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        print(f"Tracers Plot!")


        xdataTracers = np.log10(TracersFinalDict[key]['rho_rhomean'][whereTracersGas])
        ydataTracers = np.log10(TracersFinalDict[key]['T'][whereTracersGas])
        massTracers = np.log10(TracersFinalDict[key]['mass'][whereTracersGas]*1e10) #10^10Msol -> Msol
        weightDataTracers = np.log10(TracersFinalDict[key][weightKey][whereTracersGas]) * massTracers

        whereweightDataTracersNotNaNorInf = np.where((np.isinf(weightDataTracers)==False) & (np.isnan(weightDataTracers)==False))

        xdataTracers = xdataTracers[whereweightDataTracersNotNaNorInf]
        ydataTracers = ydataTracers[whereweightDataTracersNotNaNorInf]
        massTracers  = massTracers[whereweightDataTracersNotNaNorInf]
        weightDataTracers = weightDataTracers[whereweightDataTracersNotNaNorInf]

        mhistTracers,_,_=np.histogram2d(xdataTracers,ydataTracers,bins=Nbins,weights=massTracers,normed=False)
        histTracers,xedgeTracers,yedgeTracers=np.histogram2d(xdataTracers,ydataTracers,bins=Nbins,weights=weightDataTracers,normed=False)

        finalHistTracers = histTracers/mhistTracers

        finalHistTracers = finalHistTracers.T

        img2 = ax[1].imshow(finalHistTracers,cmap=colourmap,vmin=np.nanmin(finalHistTracers),vmax=np.nanmax(finalHistTracers) \
        ,extent=[np.min(xedgeTracers),np.max(xedgeTracers),np.min(yedgeTracers),np.max(yedgeTracers)],origin='lower')

        ax[1].set_xlabel(r"Log10 Density [$\rho / \langle \rho \rangle $]",fontsize=fontsize)
        ax[1].set_ylabel(r"Log10 Temperatures [$K$]",fontsize=fontsize)

        cax2 = inset_axes(ax[1],width="5%",height="95%",loc='right')
        fig.colorbar(img2, cax = cax2, orientation = 'vertical').set_label(label=labelDict[weightKey],size=fontsize)
        cax2.yaxis.set_ticks_position("left")
        cax2.yaxis.set_label_position("left")
        cax2.yaxis.label.set_color("black")
        cax2.tick_params(axis="y", colors="black",labelsize=fontsize)

        ax[1].set_title(f"Selected Tracers Data" +\
        r" - Selected at $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
        "\n"+ r" and temperatures " + r"$ 10^{n \pm %05.2f} K $"%(TRACERSPARAMS['deltaT']))
        ax[1].set_aspect("auto")
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#   Full Figure: Finishing up
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        fig.suptitle(f"Temperature Density Diagram, weighted by {weightKey}" +\
        f" and selected at {selectTime:3.2f} Gyr",fontsize=fontsize)

        plt.subplots_adjust(top=0.90, hspace = 0.005)

        opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_snap{int(snap)}_{weightKey}_PhaseDiagram.pdf"
        plt.savefig(opslaan, dpi = DPI, transparent = False)
        print(opslaan)
        plt.close()
