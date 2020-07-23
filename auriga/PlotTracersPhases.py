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

Nbins = 500
xsize = 20.
ysize = 10.
fontsize = 15
DPI = 20
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

colourmap="inferno_r"

#Paramters to weight the 2D hist by
weightKeys = ['mass','tcool','gz']

labelDict={'mass' : r'Log10 Mass per pixel [$M/M_{\odot}$]',\
 'gz':r'Log10 Average Metallicity per pixel $Z/Z_{\odot}$',\
 'tcool': r'Log10 Cooling Time per pixel [$Gyr$]',\
 }

#File types for data save.
#   Mini: small median and percentiles data
#   Full: full FullDict data
MiniDataPathSuffix = f".h5"
FullDataPathSuffix = f".h5"

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


#Set Blank Data dictionary ALL data is stored here!
FullDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,FullDataPathSuffix)
CellsFinalDict = snapData_hdf5_load(DataSavepath,TRACERSPARAMS,FullDataPathSuffix)

print("Flatten Tracers Data (snapData).")
TracersFinalDict = flatten_wrt_T(FullDict, selectedSnaps,TRACERSPARAMS)

#------------------------------------------------------------------------------#
#               PLOTTING
#
#------------------------------------------------------------------------------#
for snap in selectedSnaps: #TRACERSPARAMS['phasesSnaps']
    print("\n"+f"Starting Snap {int(snap)}")
    for weightKey in weightKeys:
        print("\n"+f"Starting weightKey {weightKey}")
        key = f"{int(snap)}"

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (xsize,ysize), dpi = DPI)

        whereCellsGas = np.where(CellsFinalDict[key]['type'] == 0)
        whereTracersGas = np.where(TracersFinalDict[key]['type'] == 0)

        CellsFinalDict[key]['age'][np.where(np.isnan(CellsFinalDict[key]['age']) == True)] = 0.
        TracersFinalDict[key]['age'][np.where(np.isnan(TracersFinalDict[key]['age']) == True)] = 0.

        whereCellsStars = np.where((CellsFinalDict[key]['type'] == 4)&\
                              (CellsFinalDict[key]['age'] >= 0.))

        whereTracersStars = np.where((TracersFinalDict[key]['type'] == 4)&\
                  (TracersFinalDict[key]['age'] >= 0.))

        whereCellsGas = np.where(CellsFinalDict[key]['type'] == 0)

        whereTracersGas = np.where(TracersFinalDict[key]['type'] == 0)

        NGasCells = len(CellsFinalDict[key]['type'][whereCellsGas])
        NStarsCells = len(CellsFinalDict[key]['type'][whereCellsStars])
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

        xdataCells = np.log10(CellsFinalDict[key]['dens'][whereCellsGas])
        ydataCells = np.log10(CellsFinalDict[key]['T'][whereCellsGas])
        massCells = np.log10(CellsFinalDict[key]['mass'][whereCellsGas]*1e10) #10^10Msol -> Msol
        weightDataCells = np.log10(CellsFinalDict[key][weightKey][whereCellsGas]) * massCells

        whereweightDataCellsNotNaNorInf = np.where((np.isinf(weightDataCells)==False) & (np.isnan(weightDataCells)==False))

        xdataCells = xdataCells[whereweightDataCellsNotNaNorInf]
        ydataCells = ydataCells[whereweightDataCellsNotNaNorInf]
        massCells  = massCells[whereweightDataCellsNotNaNorInf]
        weightDataCells = weightDataCells[whereweightDataCellsNotNaNorInf]

        mhistCells,_,_=np.histogram2d(xdataCells,ydataCells,bins=Nbins,weights=massCells,normed=False)
        histCells,xedgeCells,yedgeCells=np.histogram2d(xdataCells,ydataCells,bins=Nbins,weights=weightDataCells,normed=False)

        finalHistCells = histCells/mhistCells

        finalHistCells = finalHistCells.T

        img1 = ax[0].imshow(finalHistCells,cmap=colourmap,vmin=np.nanmin(finalHistCells),vmax=np.nanmax(finalHistCells) \
        ,extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')

        ax[0].set_xlabel(r"Log10 Density [$g$ $cm^{-3}$]",fontsize=fontsize)
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


        xdataTracers = np.log10(TracersFinalDict[key]['dens'][whereTracersGas])
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

        ax[1].set_xlabel(r"Log10 Density [$g$ $cm^{-3}$]",fontsize=fontsize)
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
        f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}",fontsize=fontsize)

        plt.subplots_adjust(top=0.90, hspace = 0.005)

        opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['snapnum'])}_snap{int(snap)}_{weightKey}_PhaseDiagram.pdf"
        plt.savefig(opslaan, dpi = DPI, transparent = False)
        print(opslaan)
        plt.close()
