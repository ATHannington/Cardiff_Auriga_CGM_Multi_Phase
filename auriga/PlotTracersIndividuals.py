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

subset = 1000#10#1000
xsize = 10.
ysize = 12.
DPI = 250
opacity = 0.02#0.5#0.02

n_Hcrit = 1e-1

colourmapMain = "viridis"
colourmapIndividuals = "plasma"
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

saveParams = ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','tcool','theat','tcross','tff']

logParameters = ['T','n_H','B','gz','L','P_thermal','P_magnetic','P_kinetic','tcool','theat','tcross','tff']

ylabel={'T': r'Temperature [$K$]', 'R': r'Radius [$kpc$]',\
 'n_H':r'$n_H$ [$cm^{-3}$]', 'B':r'|B| [$\mu G$]',\
 'vrad':r'Radial Velocity [$km$ $s^{-1}$]',\
 'gz':r'Average Metallicity $Z/Z_{\odot}$', 'L':r'Specific Angular Momentum[$kpc$ $km$ $s^{-1}$]',\
 'P_thermal': r'$P_{Thermal} / k_B$ [$K$ $cm^{-3}$]',\
 'P_magnetic':r'$P_{Magnetic} / k_B$ [$K$ $cm^{-3}$]',\
 'P_kinetic': r'$P_{Kinetic} / k_B$ [$K$ $cm^{-3}$]',\
 'tcool': r'Cooling Time [$Gyr$]',\
 'theat': r'Heating Time [$Gyr$]',\
 'tcross': r'Sound Crossing Cell Time [$Gyr$]',\
 'tff': r'Free Fall Time [$Gyr$]'\
 }

for entry in logParameters:
    ylabel[entry] = r'Log10 '+ ylabel[entry]

if (subset<=20):
    ColourIndividuals = True
else:
    ColourIndividuals = False
#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".h5"


print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)


print("Getting Tracer Data!")
Ydata = {}
Xdata = {}
Massdata ={}

tage = []
for snap in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['snapMax']+1),int(TRACERSPARAMS['snapnumMAX'])+1),1):
    minTemp = int(TRACERSPARAMS['targetTLst'][0])
    key = (f"T{int(minTemp)}", f"{int(snap)}")

    tage.append(dataDict[key]['Lookback'][0])

tage = np.array(tage)
t0 = np.nanmax(tage)
tage = abs(tage - t0)


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
    select = math.floor(float(rangeMax)/float(subset))

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
# truthyListFinal =[]
# IntoStarList = []
# IntoWindList = []
# for T in TRACERSPARAMS['targetTLst']:
#     key = f"T{int(T)}"
#
#     whereListNew = np.array([])
#     Nsnaps = np.shape(Ydata[key]['n_H'])[0]
#     flipped = np.flip(Ydata[key]['n_H'],axis=0)
#     for ind, entry in enumerate(flipped):
#         whereListOld = whereListNew
#
#         whereNan = np.where(np.isnan(entry)==True)[0]
#         whereListNew = whereNan
#         for value in whereListOld:
#             #Last time entry
#             if (value not in whereListNew):
#                 if (ind<=Nsnaps):
#                     # print(f"After [{key}] [{ind}] [{value}] = {Ydata[key]['n_H'][ind-1][value]:0.02e}")
#                     # print(f"t_lookback = {Xdata[key]['n_H'][ind]} Gyr")
#                     data = entry[value]
#                     IntoWindList.append(data)
#                     truthy = data>=n_Hcrit
#                     truthyListFinal.append(truthy)
#
#         if(np.shape(whereNan)[0]>0):
#             for value in whereNan:
#                 #First time entry
#                 if value not in whereListOld.flatten():
#                     if (ind>0):
#                         # print(f"Before [{key}] [{ind}] [{value}] = {Ydata[key]['n_H'][ind-1][value]:0.02e}")
#                         # print(f"t_lookback = {Xdata[key]['n_H'][ind]} Gyr")
#                         data = flipped[ind-1][value]
#                         IntoStarList.append(data)
#                         truthy = data>=n_Hcrit
#                         truthyListFinal.append(truthy)
#
# truthy = np.all(truthyListFinal)
# IntoStarMedian = np.median(IntoStarList)
# IntoWindMedian = np.median(IntoWindList)
#
# print("")
# print("***")
# print(f"All Tracer Path Breaks meet n_H>={n_Hcrit:0.02e} Criterion: {truthy}")
# print(f"Median n_H before going Into Star: {IntoStarMedian:0.02e}")
# print(f"Median n_H before going Into Wind: {IntoWindMedian:0.02e}")
# print("***")
# print("")
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

        snapsRange = np.array([ xx for xx in range(int(TRACERSPARAMS['snapMin']), min(int(TRACERSPARAMS['snapMax'])+1,int(TRACERSPARAMS['snapnumMAX'])+1),1)])
        selectionSnap = np.where(snapsRange==int(TRACERSPARAMS['snapnum']))


        # #Sort data by smallest Lookback time
        # ind_sorted = np.argsort(plotData['Lookback'])
        # for key, value in plotData.items():
        #     #Sort the data
        #     if isinstance(value,float)==True:
        #         entry = [value]
        #     else:
        #         entry = value
        #     sorted_data = np.array(entry)[ind_sorted]
        #     plotData.update({key: sorted_data})
        #
        # #Reverse Lookback time into universe age
        # t0 = np.max(plotData['Lookback'])
        # time_age = abs(plotData['Lookback'] - t0)
        # plotData.update({'tage': time_age})

        vline = tage[selectionSnap]

        #Get number of temperatures
        NTemps = float(len(Tlst))

        #Get temperature
        temp = TRACERSPARAMS['targetTLst'][ii]

        plotYdata = Ydata[f"T{int(temp)}"][analysisParam]
        plotXdata = Xdata[f"T{int(temp)}"][analysisParam]

        #Convert lookback time to universe age
        t0 = np.nanmax(plotXdata)
        plotXdata = abs(plotXdata - t0)

        #Set style options
        opacityPercentiles = 0.25
        lineStyleMedian = "solid"
        lineStylePercentiles = "-."

        #Select a Temperature specific colour from colourmap

        if (ColourIndividuals == True):
            cmap = matplotlib.cm.get_cmap(colourmapIndividuals)
            colour = "tab:gray"
            colourTracers = [cmap(float(jj)/float(subset)) for jj in range(0,subset)]
        else:
            #Get a colour for median and percentiles for a given temperature
            #   Have fiddled to move colours away from extremes of the colormap
            cmap = matplotlib.cm.get_cmap(colourmapMain)
            colour = cmap(float(ii+1)/float(len(Tlst)))
            colourTracers = "tab:gray"

        LO = analysisParam + 'LO'
        UP = analysisParam + 'UP'
        median = analysisParam +'median'

        LOisNOTinf = np.where(np.isinf(plotData[LO])==False)
        UPisNOTinf = np.where(np.isinf(plotData[UP])==False)
        YDataisNOTinf = np.where(np.isinf(plotYdata)==False)

        datamin = min(np.nanmin(plotYdata[YDataisNOTinf]),np.nanmin(plotData[LO][LOisNOTinf]))
        datamax = max(np.nanmax(plotYdata[YDataisNOTinf]),np.nanmax(plotData[UP][UPisNOTinf]))

        if (analysisParam in logParameters):
            plotYdata = np.log10(plotYdata)
            for key in [LO, UP, median]:
                plotData[key] = np.log10(plotData[key])

            LOisNOTinf = np.where(np.isinf(plotData[LO])==False)
            UPisNOTinf = np.where(np.isinf(plotData[UP])==False)
            YDataisNOTinf = np.where(np.isinf(plotYdata)==False)

            datamin = min(np.nanmin(plotYdata[YDataisNOTinf]),np.nanmin(plotData[LO][LOisNOTinf]))
            datamax = max(np.nanmax(plotYdata[YDataisNOTinf]),np.nanmax(plotData[UP][UPisNOTinf]))

        if ((np.isnan(datamax)==True) or (np.isnan(datamin)==True)):
            print("NaN datamin/datamax. Skipping Entry!")
            continue

        if ((np.isinf(datamax)==True) or (np.isinf(datamin)==True)):
            print("Inf datamin/datamax. Skipping Entry!")
            continue

        LOisNOTinf = np.where(np.isinf(plotData[LO])==False)
        UPisNOTinf = np.where(np.isinf(plotData[UP])==False)
        YDataisNOTinf = np.where(np.isinf(plotYdata)==False)
        LOisNOTnan = np.where(np.isnan(plotData[LO])==False)
        UPisNOTnan = np.where(np.isnan(plotData[UP])==False)
        YDataisNOTnan = np.where(np.isnan(plotYdata)==False)
        print("")
        print("Sub-Plot!")


        if (len(Tlst)==1):
            currentAx = ax
        else:
            currentAx = ax[ii]

        # UPisINF = np.where(np.isinf(plotData[UP]) == True)
        # LOisINF = np.where(np.isinf(plotData[LO]) == True)
        # medianisINF = np.where(np.isinf(plotData[median]) == True)
        #
        # print("")
        # print(f"before {median} {plotData[median][medianisINF] }")
        # plotData[UP][UPisINF] = np.array([0.])
        # plotData[median][medianisINF] = np.array([0.])
        # plotData[LO][LOisINF] = np.array([0.])
        # print(f"after {median} {plotData[median][medianisINF] }")

        currentAx.fill_between(tage,plotData[UP],plotData[LO],\
        facecolor=colour,alpha=opacityPercentiles,interpolate=False)
        currentAx.plot(tage,plotData[median],label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour, lineStyle=lineStyleMedian)

        if (ColourIndividuals == True):
            for jj in range(0,subset):
                whereDataIsNOTnan = np.where(np.isnan(plotYdata[:,jj])==False)
                lenNOTnan = len(plotYdata[:,jj][whereDataIsNOTnan])
                if (lenNOTnan>0):
                    currentAx.plot(plotXdata,(plotYdata.T[jj]).T, color = colourTracers[jj], alpha = opacity )
        else:
            for jj in range(0,subset):
                whereDataIsNOTnan = np.where(np.isnan(plotYdata[:,jj])==False)
                lenNOTnan = len(plotYdata[:,jj][whereDataIsNOTnan])
                if (lenNOTnan>0):
                    currentAx.plot(plotXdata,(plotYdata.T[jj]).T, color = colourTracers, alpha = opacity )


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

    axis0.set_xlabel(r"Age of Universe [$Gyrs$]",fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace = 0.005)
    opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['snapnum'])}_"+analysisParam+"_"+str(int(subset))+f"_Individuals.pdf"
    plt.savefig(opslaan, dpi = DPI, transparent = False)
    print(opslaan)
    plt.close()
