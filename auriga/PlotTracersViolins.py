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
opacity = 0.03#0.5#0.03

colourmapMain = "viridis"
colourmapIndividuals = "nipy_spectral"
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

saveParams = ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','tcool','theat','tcross','tff','tcool_tff']

logParameters = ['T','n_H','B','gz','L','P_thermal','P_magnetic','P_kinetic','tcool','theat','tcross','tff','tcool_tff']

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
 'tff': r'Free Fall Time [$Gyr$]',\
 'tcool_tff' : r'Cooling Time over Free Fall Time'}

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

if('tcool_tff' in saveParams):
    for key, dict in dataDict.items():
        tcool = dict['tcool']
        tff = dict['tff']
        tctf = tcool/tff
        dataDict[key].update({'tcool_tff' : tctf})


print("Getting Tracer Data!")
Ydata = {}
Xdata = {}
Massdata ={}
ViolinDict = {}

tage = []
for snap in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['snapMax']+1),int(TRACERSPARAMS['finalSnap'])+1),1):
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

    key = (f"T{int(T)}",f"{int(TRACERSPARAMS['selectSnap'])}")
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
    ViolinSubDict = {}
    for analysisParam in saveParams:
        print("")
        print(f"Starting {analysisParam} analysis")

        #Loop over snaps from and gather data for the SelectedTracers1.
        #   This should be the same tracers for all time points due to the above selection, and thus data and massdata should always have the same shape.
        tmpXdata = []
        tmpYdata = []
        tmpMassdata = []
        tmpViolinData =[]
        for snap in range(int(TRACERSPARAMS['snapMin']),min(int(TRACERSPARAMS['snapMax']+1),int(TRACERSPARAMS['finalSnap']+1))):
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

            #Violin Data
            massMean = np.mean(dataDict[key]['mass'][whereGas])
            weightedData = (dataDict[key][analysisParam] * dataDict[key]['mass'])/massMean
            whereNOTnan = np.where(np.isnan(weightedData)==False)
            weightedData=weightedData[whereNOTnan]
            tmpViolinData.append(weightedData)

        #Append the data from this parameters to a sub dictionary
        XSubDict.update({analysisParam: np.array(tmpXdata)})
        YSubDict.update({analysisParam:  np.array(tmpYdata)})
        MassSubDict.update({analysisParam: np.array(tmpMassdata)})
        ViolinSubDict.update({analysisParam : np.array(tmpViolinData)})
    #Add the full list of snaps data to temperature dependent dictionary.
    Xdata.update({f"T{int(T)}" : XSubDict})
    Ydata.update({f"T{int(T)}" : YSubDict})
    Massdata.update({f"T{int(T)}" : MassSubDict})
    ViolinDict.update({f"T{int(T)}" : ViolinSubDict})

#==============================================================================#

#==============================================================================#
#           PLOT!!
#==============================================================================#
def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

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

        plotYdata = Ydata[f"T{int(temp)}"][analysisParam]
        plotXdata = Xdata[f"T{int(temp)}"][analysisParam]
        violinData = ViolinDict[f"T{int(temp)}"][analysisParam]

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


        YDataisNOTinf = np.where(np.isinf(plotYdata)==False)

        datamin = np.nanmin(plotYdata[YDataisNOTinf])
        datamax = np.nanmax(plotYdata[YDataisNOTinf])

        if (analysisParam in logParameters):
            for (ind, array) in enumerate(violinData):
                tmpData = np.log10(array)
                whereNOTnan = np.where(np.isnan(tmpData)==False)
                violinData[ind] = tmpData[whereNOTnan]

            plotYdata = np.log10(plotYdata)

            YDataisNOTinf = np.where(np.isinf(plotYdata)==False)

            datamin = np.nanmin(plotYdata[YDataisNOTinf])
            datamax = np.nanmax(plotYdata[YDataisNOTinf])

        if ((np.isnan(datamax)==True) or (np.isnan(datamin)==True)):
            print("NaN datamin/datamax. Skipping Entry!")
            continue

        if ((np.isinf(datamax)==True) or (np.isinf(datamin)==True)):
            print("Inf datamin/datamax. Skipping Entry!")
            continue

        ##
        #   If all entries of data are nan, and thus dataset len == 0
        #   add a nan and zero array to omit violin but continue plotting
        #   without errors.
        ##
        tmp = []
        for dataset in violinData:
            if (len(dataset)==0):
                tmp.append(np.array([np.nan,0,np.nan]))
            else:
                tmp.append(dataset)

        violinData = tmp


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


        # currentAx.fill_between(tage,plotData[UP],plotData[LO],\
        # facecolor=colour,alpha=opacityPercentiles,interpolate=False)
        # currentAx.plot(tage,plotData[median],label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour, lineStyle=lineStyleMedian)

        parts = currentAx.violinplot(violinData,positions=plotXdata,showmeans=False,showmedians=False,showextrema=False)#label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour, lineStyle=lineStyleMedian)

        for pc in parts['bodies']:
            pc.set_facecolor(colour)
            pc.set_edgecolor('black')
            pc.set_alpha(opacityPercentiles)

        quartile1 = []
        medians = []
        quartile3 = []
        for dataset in violinData:
            q1,med,q3 = np.percentile(dataset, [int(TRACERSPARAMS['percentileLO']), 50, int(TRACERSPARAMS['percentileUP'])], axis=0)
            quartile1.append(q1)
            medians.append(med)
            quartile3.append(q3)

        sorted_violinData = []
        for dataset in violinData:
            ind_sorted = np.argsort(dataset)
            dataset = dataset[ind_sorted]
            sorted_violinData.append(dataset)

        whiskers = np.array([
            adjacent_values(sorted_array, q1, q3)
            for sorted_array, q1, q3 in zip(sorted_violinData, quartile1, quartile3)])
        whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]

        currentAx.scatter(plotXdata, medians, marker='o', color='white', s=30, zorder=3)
        currentAx.vlines(plotXdata, quartile1, quartile3, color='k', linestyle='-', lw=3)
        currentAx.vlines(plotXdata, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

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

        plot_patch = matplotlib.patches.Patch(color=colour)
        plot_label = r"$T = 10^{%3.0f} K$"%(float(temp))
        currentAx.legend(handles=[plot_patch], labels=[plot_label],loc='upper right')

        fig.suptitle(f"Cells Containing Tracers selected by: " +\
        "\n"+ r"$T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
        r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
        "\n" + f" and selected at snap {TRACERSPARAMS['selectSnap']:0.0f}"+\
        f" weighted by mass" +\
        "\n" + f"Subset of {int(subset)} Individual Tracers at each Temperature Plotted" \
        , fontsize=12)



    #Only give 1 x-axis a label, as they sharex
    if (len(Tlst)==1):
        axis0 = ax
    else:
        axis0 = ax[len(Tlst)-1]

    axis0.set_xlabel(r"Age of Universe [$Gyrs$]",fontsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.90, wspace = 0.005)
    opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+analysisParam+"_"+str(int(subset))+f"_IndividualsViolins.pdf"
    plt.savefig(opslaan, dpi = DPI, transparent = False)
    print(opslaan)
    plt.close()