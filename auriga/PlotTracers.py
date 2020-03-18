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

TRACERSPARAMS = pd.read_csv('TracersParams.csv', delimiter=" ", header=None, \
usecols=[0,1],skipinitialspace=True, index_col=0, comment="#").to_dict()[1]

for key, value in TRACERSPARAMS.items():
    if ((key != 'targetTLst') & (key != 'simfile')):
        TRACERSPARAMS.update({key:float(value)})
    elif ((key == 'targetTLst') & (key != 'simfile')):
        lst = value.split(",")
        lst2 = [float(item) for item in lst]
        TRACERSPARAMS.update({key:lst2})
    elif ((key != 'targetTLst') & (key == 'simfile')):
        TRACERSPARAMS.update({key:value})

Tlst = [str(int(item)) for item in TRACERSPARAMS['targetTLst']]
Tstr = '-'.join(Tlst)

DataSavepath = f"Data_snap{int(TRACERSPARAMS['snapnum'])}_min{int(TRACERSPARAMS['snapMin'])}_max{int(TRACERSPARAMS['snapMax'])}" +\
    f"_{int(TRACERSPARAMS['Rinner'])}R{int(TRACERSPARAMS['Router'])}_targetT{Tstr}"+\
    f"_deltaT{int(TRACERSPARAMS['deltaT'])}"

DataSavepathSuffix = f".csv"

fig = plt.figure()
ax = plt.gca()

for ii in range(len(Tlst)):
    load = DataSavepath + f"_T{Tlst[ii]}" + DataSavepathSuffix

    tmpData = pd.read_csv(load, delimiter=",", header=None, \
     skipinitialspace=True, index_col=0, quotechar='"',comment="#").to_dict()

    plotData = {}
    for k, v in tmpData.items():
        for key, value in tmpData[k].items():
             if k == 1 :
                 plotData.update({key: value})
             else:
                plotData[key] = np.append(plotData[key],value)


    NTemps = float(len(Tlst))

    temp = TRACERSPARAMS['targetTLst'][ii]

    #Set style options
    opacity = 0.25

    #Select a Temperature specific colour from colourmap
    cmap = matplotlib.cm.get_cmap('viridis')
    colour = cmap((float(ii)/(NTemps-1.0)))

    print("")
    print("Temperature Sub-Plot!")

    ax.fill_between(plotData['Lookbackmedian'],plotData['TUP'],plotData['TLO'],\
    facecolor=colour,alpha=opacity,interpolate=True)
    ax.plot(plotData['Lookbackmedian'],plotData['Tmedian'],label=r"$T = 10^{%3.0f} K$"%(float(temp)), color = colour)
    ax.axvline(x=plotData['Lookbackmedian'][int(TRACERSPARAMS['snapnum']-TRACERSPARAMS['snapMin'])], c='red')

    ax.set_yscale('log')
    ax.set_xlabel(r"Lookback Time [$Gyrs$]")
    ax.set_ylabel(r"Temperature [$K$]")
    ax.set_title(f"Cells Containing Tracers selected by: " +\
    "\n"+ r"$T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
    r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
    "\n" + f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}"+\
    f" weighted by mass")
    plt.legend()
    plt.tight_layout()



opslaan = f'Tracers.png'
plt.savefig(opslaan, dpi = 500, transparent = False)
print(opslaan)
plt.close()
