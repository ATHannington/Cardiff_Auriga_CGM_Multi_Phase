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
Massdata ={}
for T in TRACERSPARAMS['targetTLst']:

    key = (f"T{int(T)}",f"{int(TRACERSPARAMS['snapnum'])}")
    rangeMin = 0
    rangeMax = len(dataDict[key]['T'])
    TracerNumberSelect = np.arange(start=rangeMin, stop = rangeMax, step = 1 )
    TracerNumberSelect = sample(TracerNumberSelect.tolist(),subset)

    SelectedTracers = dataDict[key]['trid'][TracerNumberSelect]

    #Loop over snaps from and gather data for the FinalTracers.
    #   This should be the same tracers for all time points due to the above selection, and thus data and tmp should always have the same shape.
    tmpTdata = []
    tmpXdata = []
    tmpMassdata =[]
    for snap in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['snapMax']+1),int(TRACERSPARAMS['snapnumMAX']+1))):
        key = (f"T{int(T)}",f"{int(snap)}")

        TracersTruthy = np.isin(SelectedTracers,dataDict[key]['trid'])
        TracersIndices = []
        for ind, tracer in enumerate(SelectedTracers):
            truthy = np.isin(dataDict[key]['trid'],tracer)
            if np.any(truthy) == True:
                TracersIndices.append(np.where(truthy)[0])
            else:
                # print(f"No Tracer {key} {tracer} {ind}")
                TracersIndices.append([np.nan])

        data = []
        massData = []
        for (ind, element) in zip(TracersIndices,TracersTruthy):
            if element == True:
                parent = dataDict[key]['prid'][ind]
                dataIndex = np.where(np.isin(dataDict[key]['id'],parent))
                data.append(dataDict[key]['T'][dataIndex].tolist())
                massData.append(dataDict[key]['mass'][dataIndex].tolist())
            else:
                data.append([np.nan])
                massData.append([np.nan])

        tmpXdata.append(dataDict[key]['Lookback'][0])
        tmpTdata.append(data)
        tmpMassdata.append(massData)
            # if (len(tmpTdata)==0):
            #     tmpTdata = data
            #     tmpXdata = xval
            # else:
            #     tmpTdata = np.concatenate((tmpTdata,data),axis=1)
            #     tmpXdata = np.concatenate((tmpXdata,xval),axis=1)

    Tdata.update({f"T{int(T)}" : np.array(tmpTdata)})
    Xdata.update({f"T{int(T)}" : np.array(tmpXdata)})
    Massdata.update({f"T{int(T)}" : np.array(tmpMassdata)})

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
             if ((type(value) == list) or (type(value) == numpy.ndarray)):
                value = [value[int(zz)] for zz in flatRangeIndices]
             if k == 1 :
                 plotData.update({key: value})
             else:
                 plotData[key]= np.append(plotData[key], value)

    vline = plotData['Lookback'][0]
    #Sort data by smallest Lookback time
    ind_sorted = np.argsort(plotData['Lookback'])
    for key, value in plotData.items():
        #Sort the data
        sorted_data = np.array(value)[ind_sorted]
        plotData.update({key: sorted_data})

    #Get number of temperatures
    NTemps = float(len(Tlst))

    #Get temperature
    temp = TRACERSPARAMS['targetTLst'][ii]

    plotTdata = Tdata[f"T{int(temp)}"][:,:,0]

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
    ax[ii].plot(Xdata[f"T{int(temp)}"],plotTdata, color = colourTracers, alpha = opacity )
    ax[ii].axvline(x=vline, c='red')

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
    "\n" + f"Subset of {int(subset)} Individual Tracers at each Temperature Plotted" )
    ax[ii].legend(loc='upper right')


ax[len(Tlst)-1].set_xlabel(r"Lookback Time [$Gyrs$]",fontsize=8)

plt.tight_layout()
plt.subplots_adjust(top=0.90, wspace = 0.005)
opslaan = f"Tracers{int(TRACERSPARAMS['snapnum'])}T_Individuals.png"
plt.savefig(opslaan, dpi = DPI, transparent = False)
print(opslaan)
plt.close()
