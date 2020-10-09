"""
Author: A. T. Hannington
Created: 27/03/2020

Known Bugs:
    pandas read_csv loading data as nested dict. . Have added flattening to fix

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *

Nbins = 100
xsize = 12.
ysize = 10.
DPI = 100

ageUniverse = 13.77 #[Gyr]

#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

selectedSnaps = [112,119,127]

#Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','tcool','theat','tcross','tff','tcool_tff']

logParameters = ['T','n_H','B','gz','L','P_thermal','P_magnetic','P_kinetic','tcool','theat','tcross','tff','tcool_tff']

xlabel={'T': r'Temperature [$K$]', 'R': r'Radius [$kpc$]',\
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
    xlabel[entry] = r'Log10 '+ xlabel[entry]

TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".h5"

print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

for dataKey in saveParams:
    #Create a plot for each Temperature
    for ii in range(len(Tlst)):

        #Get number of temperatures
        NTemps = float(len(Tlst))

        #Get temperature
        T = TRACERSPARAMS['targetTLst'][ii]

        #Temperature specific load path
        plotData = Statistics_hdf5_load(T,DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

        median = dataKey + "median"
        vline = plotData[median][0]
        if dataKey in logParameters:
            vline = np.log10(vline)

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

        #Loop over snaps from snapMin to snapmax, taking the snapnumMAX (the final snap) as the endpoint if snapMax is greater
        for snap in selectedSnaps:#range(int(TRACERSPARAMS['snapMin']), int(min(TRACERSPARAMS['snapnumMAX']+1, TRACERSPARAMS['snapMax']+1))):

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (xsize,ysize), dpi = DPI)

            dictkey = (f"T{T}",f"{int(snap)}")

            whereGas = np.where(dataDict[dictkey]['type'] == 0)

            dataDict[dictkey]['age'][np.where(np.isnan(dataDict[dictkey]['age']) == True)] = 0.

            whereStars = np.where((dataDict[dictkey]['type'] == 4)&\
                                  (dataDict[dictkey]['age'] >= 0.))

            NGas = len(dataDict[dictkey]['type'][whereGas])
            NStars = len(dataDict[dictkey]['type'][whereStars])
            Ntot = NGas + NStars

            #Percentage in stars
            percentage = (float(NStars)/(float(Ntot)))*100.

            data = dataDict[dictkey][dataKey]
            weights = dataDict[dictkey]['mass']

            selectTime = abs(dataDict[dictkey]['Lookback'][0] - ageUniverse)

            if dataKey in logParameters:
                data = np.log10(data)

            wheredata = np.where((np.isinf(data)==False)&((np.isnan(data)==False)))[0]
            whereweights =  np.where((np.isinf(weights)==False)&((np.isnan(weights)==False)))[0]
            whereFull =  whereFull = wheredata[np.where(np.isin(wheredata,whereweights))]
            data = data[whereFull]
            weights = weights[whereFull]

            if (np.shape(data)[0]==0):
                print("No Data! Skipping Entry!")
                continue

            xmin = np.nanmin(data)
            xmax = np.nanmax(data)



            #
            # step = (xmax-xmin)/Nbins
            #
            # bins = 10**(np.arange(start=xmin,stop=xmax,step=step))

            #Select a Temperature specific colour from colourmap
            cmap = matplotlib.cm.get_cmap('viridis')
            colour = cmap(((float(ii)+1.0)/(NTemps)))

            print("Sub-plot!")

            print(f"Snap{snap} T{T} Type {dataKey}")

            ax.hist(data, bins = Nbins, range = [xmin,xmax], weights = weights, normed = True, color=colour)
            # ax[1].hist(np.log10(data), bins = Nbins, range = [xmin,xmax], cumulative=True, weights = weights, density = True, color=colour)
            # ax[0].hist(data,bins=bins,density=True, weights=weights, log=True, color=colour)
            # ax[1].hist(data,bins=bins,density=True, cumulative=True, weights=weights,color=colour)
            ax.set_xlabel(xlabel[dataKey],fontsize=15)
            # ax[1].set_xlabel(xlabel[dataKey],fontsize=8)
            ax.set_ylabel("Normalised Count",fontsize=15)
            # ax[1].set_ylabel("Cumulative Normalised Count",fontsize=8)
            fig.suptitle(f"PDF of Cells Containing Tracers selected by: " +\
            "\n"+ r"$T = 10^{%05.2f \pm %05.2f} K$"%(T,TRACERSPARAMS['deltaT']) +\
            r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
            "\n" + f" and selected at {selectTime:3.2f} Gyr"+\
            f" weighted by mass"+\
            "\n" + f"{percentage:0.03f}% of Tracers in Stars",fontsize=12)
            ax.axvline(x=vline, c='red')

            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace = 0.005)

            opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_snap{int(snap)}_T{T}_{dataKey}_PDF.pdf"
            plt.savefig(opslaan, dpi = DPI, transparent = False)
            print(opslaan)
            plt.close()
