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

Nbins = 100.
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

#Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
saveParams = ['T','R','n_H','B']

xlabel={'T': r'Temperature [$K$]', 'R': r'Radius [$kpc$]', 'n_H':r'$n_H$ [c$m^{-3}$]', 'B':r'|B| [$\mu G$]'}


TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".pickle"


print("Loading data!")

dataDict = {}

load = DataSavepath + DataSavepathSuffix

with open(load,"rb") as f:
    dataDict = pickle.load(f)


fig = plt.figure()
ax = plt.gca()

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

            dictkey = (f"T{int(T)}",f"{int(snap)}")

            data = dataDict[dictkey][dataKey]
            weights = dataDict[dictkey]['mass']

            xmin = np.min(np.log10(data))
            xmax = np.max(np.log10(data))

            step = (xmax-xmin)/Nbins

            bins = 10**(np.arange(start=xmin,stop=xmax,step=step))

            #Select a Temperature specific colour from colourmap
            cmap = matplotlib.cm.get_cmap('viridis')
            colour = cmap(((float(ii)+1.0)/(NTemps)))

            print("Sub-plot!")

            print(f"Snap{snap} T{T} Type{dataKey}")

            if (dataKey != 'R'):
                plt.xscale('log')

            plt.hist(data,bins=bins,normed=True, weights=weights, log=True,color=colour)
            plt.xlabel(xlabel[dataKey])
            plt.ylabel("Normalised Count")
            plt.title(f"PDF of Cells Containing Tracers selected by: " +\
            "\n"+ r"$T = 10^{%05.2f \pm %05.2f} K$"%(T,TRACERSPARAMS['deltaT']) +\
            r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
            "\n" + f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}"+\
            f" weighted by mass")

            plt.tight_layout()

            opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['snapnum'])}_snap{int(snap)}_T{int(T)}_{dataKey}_PDF.png"
            plt.savefig(opslaan, dpi = 500, transparent = False)
            print(opslaan)
            plt.close()
