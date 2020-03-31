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
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,AutoMinorLocator)
import const as c
from gadget import *
from gadget_subfind import *
from Snapper import *
import pickle
from Tracers_Subroutines import *
from random import sample

subset = 2500
xsize = 10.
ysize = 12.
DPI = 250
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'


TracerNumberSelect = np.arange(0,subset)

TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".pickle"


print("Loading data!")

dataDict = {}

load = DataSavepath + DataSavepathSuffix

with open(load,"rb") as f:
    dataDict = pickle.load(f)

# CellIndex = GetIndividualCellFromTracer(dataDict[('T4','127')]['trid'],dataDict[('T4','127')]['prid'],dataDict[('T4','127')]['id'],TracerNumber=TracerNumberSelect)

print("Getting Tracer Data!")
Tdata = {}
Xdata = {}
for T in TRACERSPARAMS['targetTLst']:

    key = (f"T{int(T)}",f"{int(TRACERSPARAMS['snapnum'])}")
    rangeMin = 0
    rangeMax = len(dataDict[key]['T'])
    TracerNumberSelect = np.arange(start=rangeMin, stop = rangeMax, step = 1 )
    TracerNumberSelect = sample(TracerNumberSelect.tolist(),subset)
    CellIndex, SelectedTracersOld = GetIndividualCellFromTracer(dataDict[key]['trid'],dataDict[key]['prid'],dataDict[key]['id'],TracerNumber=TracerNumberSelect)

    #Make a list of snaps to analyse. We want two sets of ranges, both starting at the selection snap, snapnum, end ending at snapMin, and snapMax/snapnumMAX (whichever is smaller)
    #   This should ensure that, as the selected tracers decrease, we are selecting all tracers at the selection snap, and a subset of those going outwards in time.
    snapRangeMin = [zz for zz in range(int(TRACERSPARAMS['snapnum']),(int(TRACERSPARAMS['snapMin'])+1), -1)]
    snapRangeMax = [zz for zz in range(int(TRACERSPARAMS['snapnum']),min(int(TRACERSPARAMS['snapnumMAX'])+1,int(TRACERSPARAMS['snapMax'])+1), 1)]
    snapRange = [snapRangeMin,snapRangeMax]

    TracersList = []
    #Loop over snaps and find a list of which tracers are selected
    for snapSet in snapRange:
        for snap in snapSet:
            key = (f"T{int(T)}",f"{int(snap)}")
            CellIndex, SelectedTracersNew = GetIndividualCellFromTracer(dataDict[key]['trid'],dataDict[key]['prid'],dataDict[key]['id'],TracerNumber=TracerNumberSelect, SelectedTracers=SelectedTracersOld)

            TracersList.append(SelectedTracersNew)
            SelectedTracersOld = SelectedTracersNew

    #The end points of the ranges in snapRange will have the minimum number of tracers from the initial snapnum set.
    #   Thus, if we want a set of tracers that maps over all time we need to find the set of tracers that these two end-points have in common.
    #       zeroPoint is the snapMin tracers, because of how iterating through the ranges works this is not at TracersList[0].
    zeroPoint = int(int(TRACERSPARAMS['snapnum']) - (int(TRACERSPARAMS['snapMin'])+1))
    FinalTracers = TracersList[zeroPoint][np.where(np.isin(TracersList[zeroPoint],TracersList[-1]))]


    snapRangeMin = [zz for zz in range(int(TRACERSPARAMS['snapMin']),(int(TRACERSPARAMS['snapnum'])+1), 1)]
    #Plus one to stop adding two snapnum data entries
    snapRangeMax = [zz for zz in range(int(TRACERSPARAMS['snapnum']+1),min(int(TRACERSPARAMS['snapnumMAX'])+1,int(TRACERSPARAMS['snapMax'])+1), 1)]
    snapRange = [snapRangeMin,snapRangeMax]

    #Loop over snaps from and gather data for the FinalTracers.
    #   This should be the same tracers for all time points due to the above selection, and thus data and tmp should always have the same shape.
    tmpTdata = []
    tmpXdata = []
    ll = 0
    for snapSet in snapRange:
        for snap in snapSet:
            key = (f"T{int(T)}",f"{int(snap)}")
            CellIndex, _ = GetIndividualCellFromTracer(dataDict[key]['trid'],dataDict[key]['prid'],dataDict[key]['id'],TracerNumber=None, SelectedTracers=FinalTracers)

            data = dataDict[key]['T'][CellIndex].reshape((-1,1))
            xdata = dataDict[key]['Lookback'][CellIndex].reshape((-1,1))

            if (len(tmpTdata)==0):
                tmpTdata = data
                tmpXdata = xdata
            else:
                tmpTdata = np.concatenate((tmpTdata,data),axis=1)
                tmpXdata = np.concatenate((tmpXdata,data),axis=1)

    Tdata.update({f"T{int(T)}" : tmpTdata})
    Xdata.update({f"T{int(T)}" : tmpXdata})

print("Starting Sub-plots!")

fig, ax = plt.subplots(nrows=len(Tlst), ncols=1 ,sharex=True, figsize = (xsize,ysize), dpi = DPI)

#Create a plot for each Temperature
for ii in range(len(Tlst)):


    #Temperature specific load path
    load = DataSavepath + f"_T{Tlst[ii]}" + ".csv"

    #Load data as DataFrame and convert to dictionary
    tmpData = pd.read_csv(load, delimiter=",", header=None, \
     skipinitialspace=True, index_col=0, quotechar='"',comment="#").to_dict()

    #Can't seem to get pandas to load data in without making two dataframes nested
    #  this section flattens into one dictionary
    plotData = {}
    for k, v in tmpData.items():
        for key, value in tmpData[k].items():
             if k == 1 :
                 plotData.update({key: value})
             else:
                 plotData[key]= np.append(plotData[key], value)


    LookbackMulti = []
    for tracer in TracerNumberSelect:
        LookbackMulti.append(plotData['Lookback'].tolist())

    LookbackMulti = np.array(LookbackMulti)

    #Get number of temperatures
    NTemps = float(len(Tlst))

    #Get temperature
    temp = TRACERSPARAMS['targetTLst'][ii]

    #Set style options
    opacity = 10./float(subset)
    opacityPercentiles = 0.25
    lineStyleMedian = "solid"
    lineStylePercentiles = "-."

    #Select a Temperature specific colour from colourmap
    cmap = matplotlib.cm.get_cmap('viridis')
    colour = cmap(((float(ii)+1.0)/(NTemps)))

    colourTracers = "tab:gray"


    print("")
    print("Sub-Plot!")

    ax[ii].fill_between(plotData['Lookback'],plotData['TUP'],plotData['TLO'],\
    facecolor=colour,alpha=opacityPercentiles,interpolate=True)
    ax[ii].plot(plotData['Lookback'],plotData['Tmedian'],label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour, lineStyle=lineStyleMedian)
    ax[ii].plot(Xdata[f"T{int(temp)}"].T,Tdata[f"T{int(temp)}"].T, color = colourTracers, alpha = opacity )
    ax[ii].axvline(x=plotData['Lookback'][int(TRACERSPARAMS['snapnum']-TRACERSPARAMS['snapMin'])], c='red')

    ax[ii].xaxis.set_minor_locator(AutoMinorLocator())
    ax[ii].yaxis.set_minor_locator(AutoMinorLocator())
    ax[ii].tick_params(which='both')

    ax[ii].set_yscale('log')
    ax[ii].set_ylabel(r"Temperature [$K$]",fontsize=8)
    ax[ii].set_ylim(ymin=1e3, ymax=1e7)
    fig.suptitle(f"Cells Containing Tracers selected by: " +\
    "\n"+ r"$T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
    r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
    "\n" + f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}"+\
    f" weighted by mass" +\
    "\n" + f"Subset of Individual Tracers at each Temperature Plotted" )
    ax[ii].legend(loc='upper right')


ax[len(Tlst)-1].set_xlabel(r"Lookback Time [$Gyrs$]",fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.90, wspace = 0.005)
opslaan = f"Tracers{int(TRACERSPARAMS['snapnum'])}T_Individuals.png"
plt.savefig(opslaan, dpi = DPI, transparent = False)
print(opslaan)
plt.close()
