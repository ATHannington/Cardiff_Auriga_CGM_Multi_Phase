"""
Author: A. T. Hannington
Created: 21/07/2020

Known Bugs:
    pandas read_csv loading data as nested dict. . Have added flattening to fix

"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *

Nbins = 500
xsize = 12.
ysize = 10.
fontsize = 15
DPI = 100
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

colourmap="inferno_r"

selectedSnaps = [112,119,127]

#Paramters to weight the 2D hist by
weightKeys = ['mass','tcool','gz']

labelDict={'mass' : r'Log10 Mass per pixel [$M/M_{\odot}$]'
 'gz':r'Log10 Average Metallicity per pixel $Z/Z_{\odot}$',\
 'tcool': r'Log10 Cooling Time per pixel [$Gyr$]',\

 }
#------------------------------------------------------------------------------#

TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

DataSavepathSuffix = f".h5"

print("Loading data!")

dataDict = {}

loadPath = DataSavepath + DataSavepathSuffix

dataDict = hdf5_load(loadPath)

#Loop over snaps from snapMin to snapmax, taking the snapnumMAX (the final snap) as the endpoint if snapMax is greater
for snap in selectedSnaps:#range(int(TRACERSPARAMS['snapMin']), int(min(TRACERSPARAMS['snapnumMAX']+1, TRACERSPARAMS['snapMax']+1))):

    for weightKey in weightKeys:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize = (xsize,ysize), dpi = DPI)

        dictkey = (f"T{int(T)}",f"{int(snap)}")
        print(f"T{int(T)}: {int(snap)}")

        whereGas = np.where(dataDict[dictkey]['type'] == 0)

        dataDict[dictkey]['age'][np.where(np.isnan(dataDict[dictkey]['age']) == True)] = 0.

        whereStars = np.where((dataDict[dictkey]['type'] == 4)&\
                              (dataDict[dictkey]['age'] >= 0.))

        NGas = len(dataDict[dictkey]['type'][whereGas])
        NStars = len(dataDict[dictkey]['type'][whereStars])
        Ntot = NGas + NStars

        #Percentage in stars
        percentage = (float(NStars)/(float(Ntot)))*100.

        xdata = np.log10(dataDict[dictkey]['dens'])
        ydata = np.log10(dataDict[dictkey]['T'])

        mass = np.log10(dataDict[dictkey]['mass']*1e10) #10^10Msol -> Msol
        weightData = np.log10(dataDict[dictkey][weightKey]) * mass

        mhist,_,_=np.histogram2d(xdata,ydata,bins=Nbins,weights=mass,normed=False)
        hist,xedge,yedge=np.histogram2d(xdata,ydata,bins=Nbins,weights=weightData,normed=False)

        finalHist = hist/mhist

        img1 = plt.imshow(finalHist,cmap=colourmap,vmin=np.nanmin(finalHist),vmax=np.nanmax(finalHist),extent=[np.min(xedge),np.max(xedge),np.min(yedge),np.max(yedge)],origin='lower')
        ax = plt.gca()

        ax.set_xlabel(r"Log10 Density [$g$ $cm^{-3}$]",fontsize=fontsize)
        ax.set_ylabel(r"Log10 Temperatures [$K$]",fontsize=fontsize)

        cax = inset_axes(ax,width="5%",height="95%",loc='right')
        fig.colorbar(img1, cax=cax, orientation = 'vertical').set_label(label=labelDict[weightKey]},size=fontsize, weight="bold")
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
        cax.yaxis.label.set_color("black")
        cax.tick_params(axis="y", colors="black", labelsize=fontsize)

        fig.suptitle(f"Temperature Density Diagram, weighted by {weightKey} for gas selected at" +\
        "\n"+ r"$T = 10^{%05.2f \pm %05.2f} K$"%(T,TRACERSPARAMS['deltaT']) +\
        r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
        "\n" + f" and selected at snap {TRACERSPARAMS['snapnum']:0.0f}"+\
        f" weighted by mass"+\
        "\n" + f"{percentage:0.03f}% of Tracers in Stars",fontsize=12)

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace = 0.005)

        opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['snapnum'])}_snap{int(snap)}_{weightKey}_PhaseDiagram.pdf"
        plt.savefig(opslaan, dpi = DPI, transparent = False)
        print(opslaan)
        plt.close()
