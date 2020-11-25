"""
Author: A. T. Hannington
Created: 19/03/2020

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
import csv

# Read in Tracer Analysis setup data
TRACERSPARAMS = pd.read_csv('TracersParams.csv', delimiter=" ", header=None, \
usecols=[0,1],skipinitialspace=True, index_col=0, comment="#").to_dict()[1]

#Update Dictionary value dtypes
for key, value in TRACERSPARAMS.items():
    #For nearly all entries convert to float
    if ((key != 'targetTLst') & (key != 'simfile')):
        TRACERSPARAMS.update({key:float(value)})
    elif (key == 'targetTLst'):
        #For targetTLst split str by "," and convert to list of floats
        lst = value.split(",")
        lst2 = [float(item) for item in lst]
        TRACERSPARAMS.update({key:lst2})
    elif (key == 'simfile'):
        #Keep simfile directory path as string
        TRACERSPARAMS.update({key:value})

#Make a list of target Temperature strings
Tlst = [str(int(item)) for item in TRACERSPARAMS['targetTLst']]
#Convert these temperatures to a string of form e.g. "4-5-6" for savepath
Tstr = '-'.join(Tlst)

#Generate savepath string of same type as analysis data
DataSavepath = f"Data_snap{int(TRACERSPARAMS['snapnum'])}_min{int(TRACERSPARAMS['snapMin'])}_max{int(TRACERSPARAMS['snapMax'])}" +\
    f"_{int(TRACERSPARAMS['Rinner'])}R{int(TRACERSPARAMS['Router'])}_targetT{Tstr}"+\
    f"_deltaT{int(TRACERSPARAMS['deltaT'])}"

DataSavepathSuffix = f".csv"

fig = plt.figure()
ax = plt.gca()

#Create a plot for each Temperature
for ii in range(len(Tlst)):

    #Temperature specific load path
    load = DataSavepath + f"_T{Tlst[ii]}" + DataSavepathSuffix

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

    #Get number of temperatures
    NTemps = float(len(Tlst))

    #Get temperature
    temp = TRACERSPARAMS['targetTLst'][ii]

    #Set style options
    opacity = 0.25

    #Select a Temperature specific colour from colourmap
    cmap = matplotlib.cm.get_cmap('viridis')
    colour = cmap(((float(ii)+1.0)/(NTemps)))

    print("")
    print("Sub-Plot!")

    ax.fill_between(plotData['Lookback'],plotData['BUP'],plotData['BLO'],\
    facecolor=colour,alpha=opacity,interpolate=True)
    ax.plot(plotData['Lookback'],plotData['Bmedian'],label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour)
    ax.axvline(x=plotData['Lookback'][int(TRACERSPARAMS['snapnum']-TRACERSPARAMS['snapMin'])], c='red')

    ax.set_yscale('log')
    ax.set_xlabel(r"Lookback Time [$Gyrs$]")
    ax.set_ylabel(r"|B| [$\mu G$]")
    ax.set_title(f"Cells Containing Tracers selected by: " +\
    "\n"+ r"$T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
    r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
    "\n" + f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}"+\
    f" weighted by mass")
    plt.legend()
    plt.tight_layout()



opslaan = f"Tracers{int(TRACERSPARAMS['snapnum'])}B.png"
plt.savefig(opslaan, dpi = 500, transparent = False)
print(opslaan)
plt.close()
