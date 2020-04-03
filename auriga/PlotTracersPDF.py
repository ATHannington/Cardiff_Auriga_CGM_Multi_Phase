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
from Snapper import *
import pickle
from Tracers_Subroutines import *

Nbins = 100
xsize = 12.
ysize = 10.
DPI = 100
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

#Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = ['T']#,'R','n_H','B']

xlabel={'T': r'Temperature [$K$]', 'R': r'Radius [$kpc$]', 'n_H':r'$n_H$ [c$m^{-3}$]', 'B':r'|B| [$\mu G$]'}


TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".pickle"


print("Loading data!")

dataDict = {}

load = DataSavepath + DataSavepathSuffix

with open(load,"rb") as f:
    dataDict = pickle.load(f)

for dataKey in saveParams:
    #Create a plot for each Temperature
    for ii in range(len(Tlst)):

        #Get number of temperatures
        NTemps = float(len(Tlst))

        #Get temperature
        T = TRACERSPARAMS['targetTLst'][ii]

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

        #Loop over snaps from snapMin to snapmax, taking the snapnumMAX (the final snap) as the endpoint if snapMax is greater
        for snap in range(int(TRACERSPARAMS['snapMin']), int(min(TRACERSPARAMS['snapnumMAX']+1, TRACERSPARAMS['snapMax']+1))):

            fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (xsize,ysize), dpi = DPI)

            dictkey = (f"T{int(T)}",f"{int(snap)}")

            data = dataDict[dictkey][dataKey]
            weights = dataDict[dictkey]['mass']

            xmin = np.min(np.log10(data))
            xmax = np.max(np.log10(data))
            #
            # step = (xmax-xmin)/Nbins
            #
            # bins = 10**(np.arange(start=xmin,stop=xmax,step=step))

            #Select a Temperature specific colour from colourmap
            cmap = matplotlib.cm.get_cmap('viridis')
            colour = cmap(((float(ii)+1.0)/(NTemps)))

            print("Sub-plot!")

            print(f"Snap{snap} T{T} Type{dataKey}")

            if (dataKey != 'R'):
                ax.set_xscale('log')
                # ax[1].set_xscale('log')
            ax.hist(np.log10(data), bins = Nbins, range = [xmin,xmax], weights = weights, density = True, color=colour)
            # ax[1].hist(np.log10(data), bins = Nbins, range = [xmin,xmax], cumulative=True, weights = weights, density = True, color=colour)
            # ax[0].hist(data,bins=bins,density=True, weights=weights, log=True, color=colour)
            # ax[1].hist(data,bins=bins,density=True, cumulative=True, weights=weights,color=colour)
            ax.set_xlabel(xlabel[dataKey],fontsize=8)
            # ax[1].set_xlabel(xlabel[dataKey],fontsize=8)
            ax.set_ylabel("Normalised Count",fontsize=8)
            # ax[1].set_ylabel("Cumulative Normalised Count",fontsize=8)
            fig.suptitle(f"PDF of Cells Containing Tracers selected by: " +\
            "\n"+ r"$T = 10^{%05.2f \pm %05.2f} K$"%(T,TRACERSPARAMS['deltaT']) +\
            r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
            "\n" + f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}"+\
            f" weighted by mass")

            plt.tight_layout()
            plt.subplots_adjust(top=0.90, hspace = 0.005)

            opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['snapnum'])}_snap{int(snap)}_T{int(T)}_{dataKey}_PDF.png"
            plt.savefig(opslaan, dpi = DPI, transparent = False)
            print(opslaan)
            plt.close()
