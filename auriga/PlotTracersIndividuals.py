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
import math

subset = 10#10#1000
xsize = 10.
ysize = 12.
DPI = 250
opacity = 0.5#0.5#0.01

n_Hcrit = 1e-1

#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

saveParams = ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','tcool']

logParameters = ['T','n_H','B','gz','L','P_thermal','P_magnetic','P_kinetic']

ylabel={'T': r'Temperature [$K$]', 'R': r'Radius [$kpc$]',\
 'n_H':r'$n_H$ [$cm^{-3}$]', 'B':r'|B| [$\mu G$]',\
 'vrad':r'Radial Velocity [$km$ $s^{-1}$]',\
 'gz':r'Average Metallicity [\]', 'L':r'Specific Angular Momentum[$kpc$ $km$ $s^{-1}$]',\
 'P_thermal': r'$P_Thermal\k_B [K cm^{-3}]$',\
 'P_magnetic':r'$P_Magnetic\k_B [K cm^{-3}]$',\
 'P_kinetic': r'$P_Kinetic\k_B [K cm^{-3}]$',\
 'tcool': r'Cooling Time [$Gyr$]'\
 }

for entry in logParameters:
    ylabel[entry] = r'Log10 '+ ylabel[entry]

if (subset<=10):
    ColourIndividuals = True
else:
    ColourIndividuals = False
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



    #Loop over temperatures in targetTLst and grab Temperature specific subset of tracers and relevant data
for T in TRACERSPARAMS['targetTLst']:
    print("")
    print(f"Starting T{T} analysis")
    #Select tracers from those present at data selection snapshot, snapnum

    key = (f"T{int(T)}",f"{int(TRACERSPARAMS['snapnum'])}")
    rangeMin = 0
    rangeMax = len(dataDict[key]['trid'])
    TracerNumberSelect = np.arange(start=rangeMin, stop = rangeMax, step = 1 )
    #Take Random sample of Tracers size min(subset, len(data))
    # TracerNumberSelect = sample(TracerNumberSelect.tolist(),min(subset,rangeMax))
    selectMin = min(subset,rangeMax)
    select = math.ceil(float(rangeMax)/float(subset))

    TracerNumberSelect = TracerNumberSelect[::select]
    SelectedTracers1 = dataDict[key]['trid'][TracerNumberSelect]

    XSubDict = {}
    YSubDict = {}
    MassSubDict = {}
    for analysisParam in saveParams:
        print("")
        print(f"Starting {analysisParam} analysis")

        #Loop over snaps from and gather data for the SelectedTracers1.
        #   This should be the same tracers for all time points due to the above selection, and thus data and massdata should always have the same shape.
        tmpXdata = []
        tmpYdata = []
        tmpMassdata = []
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
        #Append the data from this parameters to a sub dictionary
        XSubDict.update({analysisParam: np.array(tmpXdata)})
        YSubDict.update({analysisParam:  np.array(tmpYdata)})
        MassSubDict.update({analysisParam: np.array(tmpMassdata)})
    #Add the full list of snaps data to temperature dependent dictionary.
    Xdata.update({f"T{int(T)}" : XSubDict})
    Ydata.update({f"T{int(T)}" : YSubDict})
    Massdata.update({f"T{int(T)}" : MassSubDict})
#==============================================================================#
#           Check n_H!!
#==============================================================================#
truthyListFinal =[]
IntoStarList = []
IntoWindList = []
for T in TRACERSPARAMS['targetTLst']:
    key = f"T{int(T)}"

    whereListNew = np.array([])
    Nsnaps = np.shape(Ydata[key]['n_H'])[0]
    flipped = np.flip(Ydata[key]['n_H'],axis=0)
    for ind, entry in enumerate(flipped):
        whereListOld = whereListNew

        whereNan = np.where(np.isnan(entry)==True)[0]
        whereListNew = whereNan
        for value in whereListOld:
            #Last time entry
            if (value not in whereListNew):
                if (ind<=Nsnaps):
                    # print(f"After [{key}] [{ind}] [{value}] = {Ydata[key]['n_H'][ind-1][value]:0.02e}")
                    # print(f"t_lookback = {Xdata[key]['n_H'][ind]} Gyr")
                    data = entry[value]
                    IntoWindList.append(data)
                    truthy = data>=n_Hcrit
                    truthyListFinal.append(truthy)

        if(np.shape(whereNan)[0]>0):
            for value in whereNan:
                #First time entry
                if value not in whereListOld.flatten():
                    if (ind>0):
                        # print(f"Before [{key}] [{ind}] [{value}] = {Ydata[key]['n_H'][ind-1][value]:0.02e}")
                        # print(f"t_lookback = {Xdata[key]['n_H'][ind]} Gyr")
                        data = flipped[ind-1][value]
                        IntoStarList.append(data)
                        truthy = data>=n_Hcrit
                        truthyListFinal.append(truthy)

truthy = np.all(truthyListFinal)
IntoStarMedian = np.median(IntoStarList)
IntoWindMedian = np.median(IntoWindList)

print("")
print("***")
print(f"All Tracer Path Breaks meet n_H>={n_Hcrit:0.02e} Criterion: {truthy}")
print(f"Median n_H before going Into Star: {IntoStarMedian:0.02e}")
print(f"Median n_H before going Into Wind: {IntoWindMedian:0.02e}")
print("***")
print("")

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

        vline = plotData['Lookback'][0]
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

        plotYdata = Ydata[f"T{int(temp)}"][analysisParam]
        plotXdata = Xdata[f"T{int(temp)}"][analysisParam]
        #Set style options
        opacityPercentiles = 0.25
        lineStyleMedian = "solid"
        lineStylePercentiles = "-."

        #Select a Temperature specific colour from colourmap
        cmap = matplotlib.cm.get_cmap('spectral')

        if (ColourIndividuals == True):
            colour = "tab:gray"
            colourTracers = [cmap(float(jj)/float(subset)) for jj in range(0,subset)]
        else:
            colour = cmap(float(ii+1)/float(len(TLst)))
            colourTracers = "tab:gray"

        LO = analysisParam + 'LO'
        UP = analysisParam + 'UP'
        median = analysisParam +'median'

        datamin = min(np.nanmin(plotYdata),np.nanmin(plotData[LO]))
        datamax = max(np.nanmax(plotYdata),np.nanmax(plotData[UP]))

        if (analysisParam in logParameters):
            plotYdata = np.log10(plotYdata)
            for key in [LO, UP, median]:
                plotData[key] = np.log10(plotData[key])

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
        for jj in range(0,subset):
            currentAx.plot(plotXdata,(plotYdata.T[jj]).T, color = colourTracers[jj], alpha = opacity )
        currentAx.axvline(x=vline, c='red')

        currentAx.xaxis.set_minor_locator(AutoMinorLocator())
        currentAx.yaxis.set_minor_locator(AutoMinorLocator())
        currentAx.tick_params(which='both')

        currentAx.set_ylabel(ylabel[analysisParam],fontsize=10)
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

    axis0.set_xlabel(r"Lookback Time [$Gyrs$]",fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace = 0.005)
    opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['snapnum'])}_"+analysisParam+"_"+str(int(subset))+f"_Individuals.pdf"
    plt.savefig(opslaan, dpi = DPI, transparent = False)
    print(opslaan)
    plt.close()
