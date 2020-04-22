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
import h5py
from Tracers_Subroutines import *
from random import sample

subset = 10
xsize = 10.
ysize = 12.
DPI = 250
opacity = 0.5#0.01
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

saveParams = ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic']

logParameters = ['T','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic']

ylabel={'T': r'Temperature [$K$]', 'R': r'Radius [$kpc$]',\
 'n_H':r'$n_H$ [$cm^{-3}$]', 'B':r'|B| [$\mu G$]',\
 'vrad':r'Radial Velocity [$km$ $s^{-1}$]',\
 'gz':r'Average Metallicity', 'L':r'Specific Angular Momentum[$km^{2}$ $s^{-2}$]',\
 'P_thermal':r'Thermal Pressure [$erg$ $cm^{-2}$]',\
 'P_magnetic':r'Magnetic Pressure [$\mu G$ $sr^{-1}$]',\
 'P_kinetic': r'Kinetic Pressure [$M_{\odot}$ $km^2$ $s^-2$]'\
 }

for entry in logParameters:
    ylabel[entry] = r'Log10 '+ ylabel[entry]

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
        rangeMax = len(dataDict[key]['trid'])
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
            whereGas = np.where(dataDict[key]['type']==0)[0]
            #Get Individual Cell Data from selected Tracers.
            #   Not all Tracers will be present at all snapshots, so we return a NaN value in that instance.
            #   This allows for plotting of all tracers for all snaps they exist.
            #   Grab data for analysisParam and mass.
            data, massData, _ = GetIndividualCellFromTracer(Tracers=dataDict[key]['trid'],\
                Parents=dataDict[key]['prid'],CellIDs=dataDict[key]['id'][whereGas],SelectedTracers=SelectedTracers1,\
                Data=dataDict[key][analysisParam][whereGas],mass=dataDict[key]['mass'][whereGas])

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
                     plotData.update({key: float(value)})
                 else:
                     plotData[key]= np.append(plotData[key], float(value))

        vline = plotData['Lookback']
        #Sort data by smallest Lookback time
        ind_sorted = np.argsort(plotData['Lookback'])
        for key, value in plotData.items():
            #Sort the data
            if isinstance(value,float)==True:
                entry = [value]
            else:
                entry = value
            sorted_data = np.array(entry)[ind_sorted]
            plotData.update({key: sorted_data})

        #Get number of temperatures
        NTemps = float(len(Tlst))

        #Get temperature
        temp = TRACERSPARAMS['targetTLst'][ii]

        plotYdata = Ydata[f"T{int(temp)}"]

        #Set style options
        opacityPercentiles = 0.25
        lineStyleMedian = "solid"
        lineStylePercentiles = "-."

        #Select a Temperature specific colour from colourmap
        cmap = matplotlib.cm.get_cmap('viridis')
        colour = cmap(((float(ii)+1.0)/(NTemps)))

        colourTracers = "tab:gray"

        LO = analysisParam + 'LO'
        UP = analysisParam + 'UP'
        median = analysisParam +'median'

        datamin = np.nanmin(plotYdata)
        datamax = np.nanmax(plotYdata)

        if (analysisParam in logParameters):
            plotYdata = np.log10(abs(plotYdata))
            for key in [LO, UP, median]:
                plotData[key] = np.log10(abs(plotData[key]))

            datamin = min(np.nanmin(plotYdata),np.nanmin(plotData[LO]))
            datamax = max(np.nanmax(plotYdata),np.nanmax(plotData[UP]))

        print("")
        print("Sub-Plot!")


        if (len(Tlst)==1):
            currentAx = ax
        else:
            currentAx = ax[ii]


        currentAx.fill_between(plotData['Lookback'],plotData[UP],plotData[LO],\
        facecolor=colour,alpha=opacityPercentiles,interpolate=True)
        currentAx.plot(plotData['Lookback'],plotData[median],label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour, lineStyle=lineStyleMedian)
        currentAx.plot(Xdata[f"T{int(temp)}"],plotYdata, color = colourTracers, alpha = opacity )
        currentAx.axvline(x=vline, c='red')

        currentAx.xaxis.set_minor_locator(AutoMinorLocator())
        currentAx.yaxis.set_minor_locator(AutoMinorLocator())
        currentAx.tick_params(which='both')

        currentAx.set_ylabel(ylabel[analysisParam],fontsize=15)
        currentAx.set_ylim(ymin=datamin, ymax=datamax)

        fig.suptitle(f"Cells Containing Tracers selected by: " +\
        "\n"+ r"$T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
        r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
        "\n" + f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}"+\
        f" weighted by mass" +\
        "\n" + f"Subset of {int(subset)} Individual Tracers at each Temperature Plotted" \
        , fontsize=12)
        currentAx.legend(loc='upper right')


    #Only give 1 x-axis a label, as they sharex
    if (len(Tlst)==1):
        axis0 = ax
    else:
        axis0 = ax[len(Tlst)-1]

    axis0.set_xlabel(r"Lookback Time [$Gyrs$]",fontsize=15)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace = 0.005)
    opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['snapnum'])}_"+analysisParam+"_"+str(int(subset))+f"_Individuals.pdf"
    plt.savefig(opslaan, dpi = DPI, transparent = False)
    print(opslaan)
    plt.close()
