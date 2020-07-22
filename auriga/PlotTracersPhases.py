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
xsize = 12.
ysize = 10.
fontsize = 15
DPI = 100
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

colourmap="inferno_r"

selectedSnaps = [112,119,127]

#Paramters to weight the 2D hist by
weightKeys = ['mass','tcool','gz']

labelDict={'mass' : r'Log10 Mass per pixel [$M/M_{\odot}$]',\
 'gz':r'Log10 Average Metallicity per pixel $Z/Z_{\odot}$',\
 'tcool': r'Log10 Cooling Time per pixel [$Gyr$]',\
 }

#==============================================================================#
#       USER DEFINED PARAMETERS
#==============================================================================#
#Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = ['dens','T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','tcool','theat','csound','tcross','tff']

print("")
print("Saved Parameters in this Analysis:")
print(saveParams)

#Optional Tracer only (no stats in .csv) parameters to be saved
#   Cannot guarantee that all Plotting and post-processing are independent of these
#       Will attempt to ensure any necessary parameters are stored in ESSENTIALS
saveTracersOnly = ['sfr','age']

print("")
print("Tracers ONLY (no stats) Saved Parameters in this Analysis:")
print(saveTracersOnly)

#SAVE ESSENTIALS : The data required to be tracked in order for the analysis to work
saveEssentials = ['Lookback','Ntracers','Snap','id','prid','trid','type','mass']

print("")
print("ESSENTIAL Saved Parameters in this Analysis:")
print(saveEssentials)

saveTracersOnly = saveTracersOnly + saveEssentials

#Save types, which when combined with saveparams define what data is saved.
#   This is intended to be 'median', 'UP' (upper quartile), and 'LO' (lower quartile)
saveTypes= ['median','UP','LO']

#Select Halo of interest:
#   0 is the most massive:
HaloID = 0

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
FullDict = {}
CellsFullDict = {}

if __name__=="__main__":
    kk = 0
    #Loop over target temperatures
    for targetT in TRACERSPARAMS['targetTLst']:

        #Store number of target temperatures
        NTemps = float(len(TRACERSPARAMS['targetTLst']))
        #Calculate percentage complete as a function of Temperatures
        #   Aside: I did try to implement a total percentage complete, but combinatorix ='(
        percentage = (float(kk)/NTemps)*100.0
        print("")
        print(f"{percentage:0.02f}%")
        #Increment percentage complete counter
        kk+=1

        TracersTFC, CellsTFC, CellIDsTFC, ParentsTFC, _, _ = \
        tracer_selection_snap_analysis(targetT,TRACERSPARAMS,HaloID,\
        elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
        saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,lazyLoadBool,SUBSET=None)


        for snap in selectedSnaps:
            out, snapData, TracersCFT, CellsCFT, CellIDsCFT, ParentsCFT = SERIAL_snap_analysis_PLUS_snapData(snap,targetT,TRACERSPARAMS,HaloID,TracersTFC,\
            elements,elements_Z,elements_mass,elements_solar,Zsolar,omegabaryon0,\
            saveParams,saveTracersOnly,DataSavepath,FullDataPathSuffix,MiniDataPathSuffix,lazyLoadBool)
            FullDict.update(out)
            CellsFullDict.update(snapData)
#------------------------------------------------------------------------------#
#               PLOTTING
#
#------------------------------------------------------------------------------#
    for snap in selectedSnaps:
        for weightKey in weightKeys:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (xsize,ysize), dpi = DPI)

            tmpCellsFullDict = {}
            tmpFullDict = {}
            ii=0
            for T in TRACERSPARAMS['targetTLst']:
                dictkey = (f"T{int(T)}",f"{int(snap)}")
                if (ii==0):
                    tmpFullDict.update(FullDict[dictkey])
                    tmpCellsFullDict.update(CellsFullDict[dictkey])
                else:
                    for key in tmpFullDict.keys():
                        tmpFullDict[key] = np.append(tmpFullDict[key], FullDict[dictkey][key])
                    for key in tmpCellsFullDict.keys():
                        tmpCellsFullDict[key] = np.append(tmpCellsFullDict[key], CellsFullDict[dictkey][key])
                ii+=1

            whereCellsGas = np.where(tmpCellsFullDict['type'] == 0)
            whereTracersGas = np.where(tmpFullDict['type'] == 0)

            tmpCellsFullDict['age'][np.where(np.isnan(tmpCellsFullDict['age']) == True)] = 0.
            tmpFullDict['age'][np.where(np.isnan(tmpFullDict['age']) == True)] = 0.

            whereCellsStars = np.where((tmpCellsFullDict['type'] == 4)&\
                                  (tmpCellsFullDict['age'] >= 0.))

            whereTracersStars = np.where((tmpFullDict['type'] == 4)&\
                      (tmpFullDict['age'] >= 0.))

            NGasCells = len(tmpCellsFullDict['type'][whereCellsGas])
            NStarsCells = len(tmpCellsFullDict['type'][whereCellsStars])
            NtotCells = NGasCells + NStarsCells

            #Percentage in stars
            percentageCells = (float(NStarsCells)/(float(NtotCells)))*100.

            NGasTracers = len(tmpFullDict['type'][whereTracersGas])
            NStarsTracers = len(tmpFullDict['type'][whereTracersStars])
            NtotTracers = NGasTracers + NStarsTracers

            #Percentage in stars
            percentageTracers = (float(NStarsTracers)/(float(NtotTracers)))*100.



#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#   Figure 1: Full Cells Data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

            xdataCells = np.log10(tmpCellsFullDict['dens'])
            ydataCells = np.log10(tmpCellsFullDict['T'])

            massCells = np.log10(tmpCellsFullDict['mass']*1e10) #10^10Msol -> Msol
            weightDataCells = np.log10(tmpCellsFullDict[weightKey]) * massCells

            mhistCells,_,_=np.histogram2d(xdataCells,ydataCells,bins=Nbins,weights=massCells,normed=False)
            histCells,xedgeCells,yedgeCells=np.histogram2d(xdataCells,ydataCells,bins=Nbins,weights=weightDataCells,normed=False)

            finalHistCells = histCells/mhistCells

            img1 = plt.imshow(finalHistCells,cmap=colourmap,vmin=np.nanmin(finalHistCells),vmax=np.nanmax(finalHistCells),\
            extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')
            ax[0] = plt.gca()

            ax[0].set_xlabel(r"Log10 Density [$g$ $cm^{-3}$]",fontsize=fontsize)
            ax[0].set_ylabel(r"Log10 Temperatures [$K$]",fontsize=fontsize)

            cax1 = inset_axes(ax[0],width="5%",height="95%",loc='right')
            fig.colorbar(img1, cax=cax1, orientation = 'vertical').set_label(label=labelDict[weightKey],size=fontsize, weight="bold")
            cax1.yaxis.set_ticks_position("left")
            cax1.yaxis.set_label_position("left")
            cax1.yaxis.label.set_color("black")
            cax1.tick_params(axis="y", colors="black", labelsize=fontsize)

            ax[0].set_title(f"Full Simulation Data - {percentageCells:0.02f}% in Stars",fontsize=12)

#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#   Figure 2: Tracers Only  Data
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

            xdataTracers = np.log10(tmpFullDict['dens'])
            ydataTracers = np.log10(tmpFullDict['T'])

            massTracers = np.log10(tmpFullDict['mass']*1e10) #10^10Msol -> Msol
            weightDataTracers = np.log10(tmpFullDict[weightKey]) * massTracers

            mhistTracers,_,_=np.histogram2d(xdataTracers,ydataTracers,bins=Nbins,weights=massTracers,normed=False)
            histTracers,xedgeTracers,yedgeTracers=np.histogram2d(xdataTracers,ydataTracers,bins=Nbins,weights=weightDataTracers,normed=False)

            finalHistTracers = histTracers/mhistTracers

            img2 = plt.imshow(finalHistTracers,cmap=colourmap,vmin=np.nanmin(finalHistTracers),vmax=np.nanmax(finalHistTracers),\
            extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')
            ax[1] = plt.gca()

            ax[1].set_xlabel(r"Log10 Density [$g$ $cm^{-3}$]",fontsize=fontsize)
            ax[1].set_ylabel(r"Log10 Temperatures [$K$]",fontsize=fontsize)

            cax2 = inset_axes(ax[1],width="5%",height="95%",loc='right')
            fig.colorbar(img2, cax=cax2, orientation = 'vertical').set_label(label=labelDict[weightKey],size=fontsize, weight="bold")
            cax2.yaxis.set_ticks_position("left")
            cax2.yaxis.set_label_position("left")
            cax2.yaxis.label.set_color("black")
            cax2.tick_params(axis="y", colors="black", labelsize=fontsize)

            ax[1].set_title(f"Selected Tracers Data - {percentageTracers :0.02f}% in Stars",fontsize=12)
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
#   Full Figure: Finishing up
#- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
            fig.suptitle(f"Temperature Density Diagram, weighted by {weightKey} for gas selected at" +\
            "\n"+ r"$T = 10^{%05.2f \pm %05.2f} K$"%(T,TRACERSPARAMS['deltaT']) +\
            r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
            "\n" + f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}"+\
            f" weighted by mass",fontsize=12)

            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace = 0.005)

            opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['snapnum'])}_snap{int(snap)}_{weightKey}_PhaseDiagram.pdf"
            plt.savefig(opslaan, dpi = DPI, transparent = False)
            print(opslaan)
            plt.close()
