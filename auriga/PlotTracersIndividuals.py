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

subset = 10
xsize = 10.
ysize = 12.
DPI = 250
opacity = 0.5#0.01
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

saveParams = ['T', 'R', 'n_H', 'B']

ylabel={'T': r'Temperature [$K$]', 'R': r'Radius [$kpc$]', 'n_H':r'$n_H$ [c$m^{-3}$]', 'B':r'|B| [$\mu G$]'}

#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".h5"


print("Loading data!")

dataDict = {}

loadPath = DataSavepath + DataSavepathSuffix

dataDict = hdf5_load(loadPath)


print("Getting Tracer Data!")
Ydata = {}
Xdata = {}
Massdata ={}

for analysisParam in saveParams:
    print("")
    print(f"Starting {analysisParam} analysis")

    #Loop over temperatures in targetTLst and grab Temperature specific subset of tracers and relevant data
    for T in TRACERSPARAMS['targetTLst']:

        #Select tracers from those present at data selection snapshot, snapnum

        key = (f"T{int(T)}",f"{int(TRACERSPARAMS['snapnum'])}")
        rangeMin = 0
        rangeMax = len(dataDict[key][analysisParam])
        TracerNumberSelect = np.arange(start=rangeMin, stop = rangeMax, step = 1 )

        #Take Random sample of Tracers size min(subset, len(data))
        TracerNumberSelect = sample(TracerNumberSelect.tolist(),min(subset,rangeMax))
        SelectedTracers1 = dataDict[key]['trid'][TracerNumberSelect]

        #Loop over snaps from and gather data for the SelectedTracers1.
        #   This should be the same tracers for all time points due to the above selection, and thus data and massdata should always have the same shape.
        tmpYdata = []
        tmpXdata = []
        tmpMassdata =[]
        for snap in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['snapMax']+1),int(TRACERSPARAMS['snapnumMAX']+1))):
            key = (f"T{int(T)}",f"{int(snap)}")

            #Get Individual Cell Data from selected Tracers.
            #   Not all Tracers will be present at all snapshots, so we return a NaN value in that instance.
            #   This allows for plotting of all tracers for all snaps they exist.
            #   Grab data for analysisParam and mass.
            data, massData, _ = GetIndividualCellFromTracer(Tracers=dataDict[key]['trid'],\
                Parents=dataDict[key]['prid'],CellIDs=dataDict[key]['id'],SelectedTracers=SelectedTracers1,\
                Data=dataDict[key][analysisParam],mass=dataDict[key]['mass'])

            #Append the data from this snapshot to a temporary list
            tmpXdata.append(dataDict[key]['Lookback'][0])
            tmpYdata.append(data)
            tmpMassdata.append(massData)

        #Add the full list of snaps data to temperature dependent dictionary.
        Ydata.update({f"T{int(T)}" : np.array(tmpYdata)})
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

        plotYdata = Ydata[f"T{int(temp)}"][:,:,0]

        #Set style options
        opacityPercentiles = 0.25
        lineStyleMedian = "solid"
        lineStylePercentiles = "-."

        #Select a Temperature specific colour from colourmap
        cmap = matplotlib.cm.get_cmap('viridis')
        colour = cmap(((float(ii)+1.0)/(NTemps)))

        colourTracers = "tab:gray"


        print("")
        print("Sub-Plot!")
        LO = analysisParam + 'LO'
        UP = analysisParam + 'UP'
        median = analysisParam +'median'
        ax[ii].fill_between(plotData['Lookback'],plotData[UP],plotData[LO],\
        facecolor=colour,alpha=opacityPercentiles,interpolate=True)
        ax[ii].plot(plotData['Lookback'],plotData[median],label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour, lineStyle=lineStyleMedian)
        ax[ii].plot(Xdata[f"T{int(temp)}"],plotYdata, color = colourTracers, alpha = opacity )
        ax[ii].axvline(x=vline, c='red')

        ax[ii].xaxis.set_minor_locator(AutoMinorLocator())
        ax[ii].yaxis.set_minor_locator(AutoMinorLocator())
        ax[ii].tick_params(which='both')
        if (analysisParam != 'R'):
            ax[ii].set_yscale('log')
        ax[ii].set_ylabel(ylabel[analysisParam],fontsize=8)
        if (analysisParam == 'T'):
            ax[ii].set_ylim(ymin=1e3, ymax=1e7)
        fig.suptitle(f"Cells Containing Tracers selected by: " +\
        "\n"+ r"$T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
        r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
        "\n" + f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}"+\
        f" weighted by mass" +\
        "\n" + f"Subset of {int(subset)} Individual Tracers at each Temperature Plotted" )
        ax[ii].legend(loc='upper right')


    #Only give 1 x-axis a label, as they sharex
    ax[len(Tlst)-1].set_xlabel(r"Lookback Time [$Gyrs$]",fontsize=8)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace = 0.005)
    opslaan = f"Tracers{int(TRACERSPARAMS['snapnum'])}"+analysisParam+str(int(subset))+f"_Individuals.pdf"
    plt.savefig(opslaan, dpi = DPI, transparent = False)
    print(opslaan)
    plt.close()
