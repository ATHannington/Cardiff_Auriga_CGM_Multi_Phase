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

xsize = 10.
ysize = 12.
DPI = 250

#Set style options
opacityPercentiles = 0.25
lineStyleMedian = "solid"
lineStylePercentiles = "-."

ageUniverse = 13.77 #[Gyr]

colourmapMain = "viridis"
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

logParameters = ['dens','rho_rhomean','csound','T','n_H','B','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']
# "rho_rhomean,dens,T,R,n_H,B,vrad,gz,L,P_thermal,P_magnetic,P_kinetic,P_tot,tcool,theat,csound,tcross,tff,tcool_tff"
ylabel={'T': r'Temperature [$K$]', 'R': r'Radius [$kpc$]',\
 'n_H':r'$n_H$ [$cm^{-3}$]', 'B':r'|B| [$\mu G$]',\
 'vrad':r'Radial Velocity [$km$ $s^{-1}$]',\
 'gz':r'Average Metallicity $Z/Z_{\odot}$', 'L':r'Specific Angular Momentum[$kpc$ $km$ $s^{-1}$]',\
 'P_thermal': r'$P_{Thermal} / k_B$ [$K$ $cm^{-3}$]',\
 'P_magnetic':r'$P_{Magnetic} / k_B$ [$K$ $cm^{-3}$]',\
 'P_kinetic': r'$P_{Kinetic} / k_B$ [$K$ $cm^{-3}$]',\
 'P_tot': r'$P_{tot} = P_{thermal} + P_{magnetic} / k_B$ [$K$ $cm^{-3}$]',\
 'tcool': r'Cooling Time [$Gyr$]',\
 'theat': r'Heating Time [$Gyr$]',\
 'tcross': r'Sound Crossing Cell Time [$Gyr$]',\
 'tff': r'Free Fall Time [$Gyr$]',\
 'tcool_tff' : r'Cooling Time over Free Fall Time',\
 'csound' : r'Sound Speed',\
 'rho_rhomean': r'Density over Average Universe Density',\
 'dens' : r'Density [$g$ $cm^{-3}$]'}

for entry in logParameters:
    ylabel[entry] = r'Log10 '+ ylabel[entry]

#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

saveParams = TRACERSPARAMS['saveParams'] #['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

DataSavepathSuffix = f".h5"


print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)


tage = []
for snap in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['snapMax']+1),int(TRACERSPARAMS['finalSnap'])+1),1):
    minTemp = TRACERSPARAMS['targetTLst'][0]
    key = (f"T{minTemp}", f"{int(snap)}")

    tage.append(dataDict[key]['Lookback'][0])

tage = np.array(tage)
tage = abs(tage - ageUniverse)

#==============================================================================#

#==============================================================================#
#           PLOT!!
#==============================================================================#

for analysisParam in saveParams:
    print("")
    print(f"Starting {analysisParam} Sub-plots!")

    fig, ax = plt.subplots(nrows=len(Tlst), ncols=1 ,sharex=True, figsize = (xsize,ysize), dpi = DPI)

    #Create a plot for each Temperature
    for ii in range(len(Tlst)):

        #Temperature specific load path
        plotData = Statistics_hdf5_load(Tlst[ii],DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

        snapsRange = np.array([ xx for xx in range(int(TRACERSPARAMS['snapMin']), min(int(TRACERSPARAMS['snapMax'])+1,int(TRACERSPARAMS['finalSnap'])+1),1)])
        selectionSnap = np.where(snapsRange==int(TRACERSPARAMS['selectSnap']))

        vline = tage[selectionSnap]

        #Get number of temperatures
        NTemps = float(len(Tlst))

        #Get temperature
        temp = TRACERSPARAMS['targetTLst'][ii]

        #Select a Temperature specific colour from colourmap

        #Get a colour for median and percentiles for a given temperature
        #   Have fiddled to move colours away from extremes of the colormap
        cmap = matplotlib.cm.get_cmap(colourmapMain)
        colour = cmap(float(ii+1)/float(len(Tlst)))

        LO = analysisParam + 'LO'
        UP = analysisParam + 'UP'
        median = analysisParam +'median'

        if (analysisParam in logParameters):
            for k, v in plotData.items():
                plotData.update({k : np.log10(v)})

        print("")
        print("Sub-Plot!")


        if (len(Tlst)==1):
            currentAx = ax
        else:
            currentAx = ax[ii]

        currentAx.fill_between(tage,plotData[UP],plotData[LO],\
        facecolor=colour,alpha=opacityPercentiles,interpolate=False)
        currentAx.plot(tage,plotData[median],label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour, lineStyle=lineStyleMedian)

        currentAx.axvline(x=vline, c='red')

        currentAx.xaxis.set_minor_locator(AutoMinorLocator())
        currentAx.yaxis.set_minor_locator(AutoMinorLocator())
        currentAx.tick_params(which='both')

        currentAx.set_ylabel(ylabel[analysisParam],fontsize=10)
        currentAx.set_ylim(ymin=np.nanmin(plotData[median]), ymax=np.nanmax(plotData[median]))

        plot_patch = matplotlib.patches.Patch(color=colour)
        plot_label = r"$T = 10^{%3.2f} K$"%(float(temp))
        currentAx.legend(handles=[plot_patch], labels=[plot_label],loc='upper right')

        fig.suptitle(f"Cells Containing Tracers selected by: " +\
        "\n"+ r"$T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
        r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
        "\n" + f" and selected at {vline[0]:3.2f} Gyr"+\
        f" weighted by mass" \
        , fontsize=12)



    #Only give 1 x-axis a label, as they sharex
    if (len(Tlst)==1):
        axis0 = ax
    else:
        axis0 = ax[len(Tlst)-1]

    axis0.set_xlabel(r"Age of Universe [$Gyrs$]",fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.0)
    opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+analysisParam+f"_IndividualsMedians.pdf"
    plt.savefig(opslaan, dpi = DPI, transparent = False)
    print(opslaan)
    plt.close()
