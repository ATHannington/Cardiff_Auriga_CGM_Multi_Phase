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
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec

import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
from random import sample
import math

xsize = 10.
ysize = 10.
DPI = 250

ageUniverse = 13.77 #[Gyr]

colourmapMain = "viridis"
colourmapIndividuals = "Dark2"#"nipy_spectral"

lineStyleMedian = "-"
lineWidthMedian = 2


opacityPercentiles = 0.1
lineStylePercentiles = "-."
lineWidthPercentiles = 1

#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

dtwJoint = True
plotStatsEntries = ['%Halo0','%Unbound','%OtherHalo',"%NoHalo",'%Stars',"%ISM",'%Wind',"%Inflow","%Outflow"]
StatsEntries = ['Cluster','Snap','Lookback',"T","%Tracers",'%Halo0','%Unbound','%OtherHalo',"%NoHalo",'%Stars',"%ISM",'%Wind',"%Inflow","%Outflow"]

#==============================================================================#

#Load Analysis Setup Data
TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)
DataSavepathSuffix = f".h5"

print("Loading data!")

loadPath = DataSavepath + "_DTW-Cluster-Statistics-Table.csv"

StatsDF = pd.read_csv(loadPath)


for analysisParam in plotStatsEntries:
    print("")
    print(f"Starting {analysisParam} Sub-plots!")

    #Create a plot for each Temperature
    for ii in range(len(Tlst)):
        T = TRACERSPARAMS['targetTLst'][ii]


        plotData = StatsDF[analysisParam].loc[StatsDF['T'] == T].to_numpy()
        clusterData = StatsDF['Cluster'].loc[StatsDF['T'] == T].to_numpy()

        snapsRange = np.array([ xx for xx in range(int(TRACERSPARAMS['snapMin']), min(int(TRACERSPARAMS['snapMax'])+1,int(TRACERSPARAMS['finalSnap'])+1),1)])
        selectionSnap = np.where(snapsRange==int(TRACERSPARAMS['selectSnap']))

        xData = StatsDF['Lookback'].loc[(StatsDF['T'] == T) & ((StatsDF['Cluster'] == -1))].to_numpy()
        vline = xData[selectionSnap]

        xData = abs(xData - ageUniverse)
        vline = abs(vline - ageUniverse)

        #Get number of temperatures
        NTemps = float(len(Tlst))

        uniqueClusters = np.unique(clusterData)
        uniqueClusters = uniqueClusters[uniqueClusters!=-1]


        nClusters = len(uniqueClusters)
        fig = plt.figure(constrained_layout=True, figsize = (xsize,ysize), dpi = DPI)
        gs = GridSpec(3, int(nClusters/2), figure=fig)

        #Select a Temperature specific colour from colourmap
        maxCluster = np.nanmax(np.unique(clusterData))
        cmap = matplotlib.cm.get_cmap(colourmapIndividuals)
        cmap2 = matplotlib.cm.get_cmap(colourmapMain)
        colour = cmap2(float(ii+1)/float(len(Tlst)))
        colourTracers = [cmap(float(jj)/float(maxCluster)) for jj in uniqueClusters]

        print("")
        print("Sub-Plot!")

        currentAx = fig.add_subplot(gs[0, 0:int(nClusters/2)])
        whereCluster = np.where(clusterData==-1)[0]
        plotYdata = plotData[whereCluster]
        currentAx.plot(xData, plotYdata, color = colour, lineStyle=lineStyleMedian, linewidth = lineWidthMedian)

        datamin = 0.
        datamax = 100.
        for (col,clusterID) in zip(colourTracers,uniqueClusters):
            startcol = int((clusterID-1)%3)
            endcol = startcol + 1

            row = int(math.floor((clusterID-1)/(3)))+1

            tmpAx = fig.add_subplot(gs[row, startcol:endcol])

            whereCluster = np.where(clusterData==clusterID)[0]
            plotYdata = plotData[whereCluster]

            tmpAx.plot(xData, plotYdata, color = col, lineStyle=lineStyleMedian, linewidth = lineWidthMedian, label=f"Cluster {int(clusterID)}")

            tmpAx.axvline(x=vline, c='red')

            if tmpAx.is_last_row():
                tmpAx.set_xlabel(r"Age of Universe [$Gyrs$]",fontsize=10)
            else:
                plt.setp(tmpAx.get_xticklabels(), visible=False)

            if tmpAx.is_first_col():
                tmpAx.set_ylabel(f"{analysisParam}",fontsize=10)
            else:
                plt.setp(tmpAx.get_yticklabels(), visible=False)
            tmpAx.xaxis.set_minor_locator(AutoMinorLocator())
            tmpAx.yaxis.set_minor_locator(AutoMinorLocator())
            tmpAx.tick_params(which='both')

            tmpAx.legend(loc='upper right')
            tmpAx.set_ylim(ymin=datamin, ymax=datamax)
            # tmpAx.set_xlabel(r"Age of Universe [$Gyrs$]",fontsize=10)

        plot_label = r"$T = 10^{%3.2f} K$"%(float(T))
        currentAx.text(0.05, 0.95, plot_label, horizontalalignment='left',verticalalignment='center',\
        transform=currentAx.transAxes, wrap=True,bbox=dict(facecolor=colour, alpha=0.125))

        currentAx.transAxes

        currentAx.axvline(x=vline, c='red')

        currentAx.xaxis.set_minor_locator(AutoMinorLocator())
        currentAx.yaxis.set_minor_locator(AutoMinorLocator())
        currentAx.tick_params(which='both')

        currentAx.set_ylabel(f"{analysisParam}",fontsize=10)
        currentAx.set_ylim(ymin=datamin, ymax=datamax)

        # currentAx.legend(loc='upper right')
        fig.suptitle(f"Cells Containing Tracers selected by: " +\
        "\n"+ r"$T = 10^{n \pm %05.2f} K$"%(TRACERSPARAMS['deltaT']) +\
        r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
        "\n" + f" and selected at {vline[0]:3.2f} Gyr"+\
        f" weighted by mass"\
        , fontsize=12)

        currentAx.set_xlabel(r"Age of Universe [$Gyrs$]",fontsize=10)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        if (dtwJoint == True):
            opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+analysisParam+f"_T{T}"+f"_Joint-DTW-Stats-Plot.pdf"
        else:
            opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_"+analysisParam+f"_T{T}"+f"_DTW-Stats-Plot.pdf"
        plt.savefig(opslaan, dpi = DPI, transparent = False)
        print(opslaan)
        plt.close()
