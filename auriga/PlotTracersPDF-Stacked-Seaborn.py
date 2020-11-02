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
import seaborn as sns
import const as c
from gadget import *
from gadget_subfind import *
import h5py
from Tracers_Subroutines import *
import scipy.stats as stats

Nbins = 100
xsize = 12.
ysize = 10.
DPI = 100

ageUniverse = 13.77 #[Gyr]
selectColour= 'red'
selectStyle = '-.'
selectWidth = 4
percentileLO = 1.0
percentileUP = 99.0
#Input parameters path:
TracersParamsPath = 'TracersParams.csv'

# selectedSnaps = [112,119,127]

#Entered parameters to be saved from
#   n_H, B, R, T
#   Hydrogen number density, |B-field|, Radius [kpc], Temperature [K]
# saveParams = ['T','R','n_H','B','vrad','gz','L','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

logParameters = ['dens','rho_rhomean','csound','T','n_H','B','L','gz','P_thermal','P_magnetic','P_kinetic','P_tot','tcool','theat','tcross','tff','tcool_tff']

xlabel={'T': r'Temperature [$K$]', 'R': r'Radius [$kpc$]',\
 'n_H':r'$n_H$ [$cm^{-3}$]', 'B':r'|B| [$\mu G$]',\
 'vrad':r'Radial Velocity [$km$ $s^{-1}$]',\
 'gz':r'Average Metallicity $Z/Z_{\odot}$', 'L':r'Specific Angular Momentum[$kpc$ $km$ $s^{-1}$]',\
 'P_thermal': r'$P_{Thermal} / k_B$ [$K$ $cm^{-3}$]',\
 'P_magnetic':r'$P_{Magnetic} / k_B$ [$K$ $cm^{-3}$]',\
 'P_kinetic': r'$P_{Kinetic} / k_B$ [$K$ $cm^{-3}$]',\
 'P_tot': r'$P_{tot} = P_{thermal} + P_{magnetic} / k_B$ [$K$ $cm^{-3}$]',\
 'tcool': r'Cooling Time [$Gyr$]',\
 'theat': r'Heating Time [$Gyr$]',\
 'tcross': r'Sound Crossing Cell Time [$Gyr$]',\
 'tff': r'Free Fall Time [$Gyr$]',\
 'tcool_tff' : r'Cooling Time over Free Fall Time',\
 'csound' : r'Sound Speed',\
 'rho_rhomean': r'Density over Average Universe Density',\
 'dens' : r'Density [$g$ $cm^{-3}$]'}

for entry in logParameters:
    xlabel[entry] = r'Log10 '+ xlabel[entry]

TRACERSPARAMS, DataSavepath, Tlst = LoadTracersParameters(TracersParamsPath)

saveParams = TRACERSPARAMS['saveParams']

DataSavepathSuffix = f".h5"

print("Loading data!")

dataDict = {}

dataDict = FullDict_hdf5_load(DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

for dataKey in saveParams:
    print(f"{dataKey}")
    #Create a plot for each Temperature
    for ii in range(len(Tlst)):
        print(f"T{Tlst[ii]}")
        #Get number of temperatures
        NTemps = float(len(Tlst))

        #Get temperature
        T = TRACERSPARAMS['targetTLst'][ii]

        #Temperature specific load path
        plotData = Statistics_hdf5_load(T,DataSavepath,TRACERSPARAMS,DataSavepathSuffix)

        # median = dataKey + "median"
        # vline = plotData[median][0]
        # if dataKey in logParameters:
        #     vline = np.log10(vline)

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

        selectKey = (f"T{T}",f"{int(TRACERSPARAMS['selectSnap'])}")
        selectTime = abs(dataDict[selectKey]['Lookback'][0] - ageUniverse)

        xmaxlist = []
        xminlist = []
        dataList = []
        weightsList = []
        snapRange = [xx for xx in range(int(TRACERSPARAMS['snapMin']), int(min(TRACERSPARAMS['finalSnap']+1, TRACERSPARAMS['snapMax']+1)))]
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        fig, ax = plt.subplots(nrows=len(snapRange), ncols=1, figsize = (xsize,ysize), dpi = DPI, frameon=False, sharex=True)
        #Loop over snaps from snapMin to snapmax, taking the snapnumMAX (the final snap) as the endpoint if snapMax is greater
        for (jj, snap) in enumerate(snapRange):
            currentAx = ax[jj]
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

            if dataKey in logParameters:
                data = np.log10(data)

            wheredata = np.where((np.isinf(data)==False)&((np.isnan(data)==False)))[0]
            whereweights =  np.where((np.isinf(weights)==False)&((np.isnan(weights)==False)))[0]
            whereFull = wheredata[np.where(np.isin(wheredata,whereweights))]
            data = data[whereFull]
            weights = weights[whereFull]

            dataList.append(data)
            weightsList.append(weights)

            if (np.shape(data)[0]==0):
                print("No Data! Skipping Entry!")
                continue

            #Select a Temperature specific colour from colourmap
            cmap = matplotlib.cm.get_cmap('viridis')
            if (int(snap) == int(TRACERSPARAMS['selectSnap']) ):
                colour = selectColour
                lineStyle = selectStyle
                linewidth = selectWidth
            else:
                sRange = int(min(TRACERSPARAMS['finalSnap']+1, TRACERSPARAMS['snapMax']+1)) - int(TRACERSPARAMS['snapMin'])
                colour = cmap(((float(jj)+1.0)/(sRange)))
                lineStyle = "-"
                linewidth = 2

            tmpdict = {'x':data, 'y': weights}
            df = pd.DataFrame(tmpdict)
            # Draw the densities in a few steps
            sns.kdeplot(df["x"], weights= df['y'],
                  ax =currentAx, bw_adjust=.5, clip_on=False,
                  fill=True, alpha=1, linewidth=1.5,color=colour)
            sns.kdeplot(df["x"], weights= df['y'], ax =currentAx, clip_on=False, color="w", lw=linewidth, linestyle=lineStyle, bw_adjust=.5)
            currentAx.axhline( y=0, lw=linewidth, linestyle=lineStyle, clip_on=False)

            LO = weightedperc(data=data, weights=weights, perc=percentileLO,key='LO')
            UP = weightedperc(data=data, weights=weights, perc=percentileUP,key='UP')

            xmin = xminlist.append(LO)#np.nanmin(data)
            xmax = xmaxlist.append(UP)#np.nanmax(data)

            # #            # # step = (xmax-xmin)/Nbins
            # #
            # # bins = 10**(np.arange(start=xmin,stop=xmax,step=step))
            #

            #
            # # print("Sub-plot!")
            #
            # # Small reduction of the X extents to get a cheap perspective effect
            # # xscale = 1 - jj / 200.
            # # Same for linewidth (thicker strokes on bottom)
            # lw = 2.0 + jj / 100.0
            #
            # # print(f"Snap{snap} T{T} Type {dataKey}")
            # density = stats.gaussian_kde(data)
            # n,x,_ = plt.hist(data, bins = Nbins, range = [xmin,xmax], weights = weights, density = True, color=colour, alpha = 0.)
            #
            # tmpymax = np.nanmax(density(x))
            # if dataKey in logParameters:
            #     deltay = (jj/10.)
            # else:
            #     deltay = (jj/200.)
            #
            # ymax.append(tmpymax - deltay)
            # ax.plot(x, density(x) - deltay, lw=lw, color=colour)
            #
            # empty = np.zeros(shape=np.shape(x))
            # ax.plot(x, empty - deltay, lw=lw, color=colour, alpha = 0.1)
            # ymin.append(-1.*deltay)
            currentAx.set_yticks([])
            currentAx.set_ylabel("")
            currentAx.set_xlabel(xlabel[dataKey],fontsize=15)
            sns.despine(bottom=True, left=True)

            # ax[1].hist(np.log10(data), bins = Nbins, range = [xmin,xmax], cumulative=True, weights = weights, density = True, color=colour)
            # ax[0].hist(data,bins=bins,density=True, weights=weights, log=True, color=colour)
            # ax[1].hist(data,bins=bins,density=True, cumulative=True, weights=weights,color=colour)

        # # Define and use a simple function to label the plot in axes coordinates

        xmin = np.nanmin(xminlist)
        xmax = np.nanmax(xmaxlist)

        plt.xlim(xmin,xmax)
        #
        plot_label = r"$T = 10^{%3.2f} K$"%(float(T))
        plt.text(0.80, 0.90, plot_label, horizontalalignment='left',verticalalignment='center',\
        transform=fig.transFigure, wrap=True,bbox=dict(facecolor='blue', alpha=0.2),fontsize = 15)

        time_label = r"Age of Universe [Gyr]"
        plt.text(0.10, 0.52, time_label, horizontalalignment='center',verticalalignment='center',\
        transform=fig.transFigure, wrap=True, fontsize = 15)
        plt.arrow(0.10, 0.50, 0., -0.25, fc='black', ec='black', width=0.005, transform=fig.transFigure, clip_on=False)
        fig.transFigure

        # plt.xlabel(xlabel[dataKey],fontsize=15)
        # ax[1].set_xlabel(xlabel[dataKey],fontsize=8)
        # ax.set_ylabel("Normalised Count",fontsize=15)
        # ax[1].set_ylabel("Cumulative Normalised Count",fontsize=8)
        fig.suptitle(f"PDF of Cells Containing Tracers selected by: " +\
        "\n"+ r"$T = 10^{%05.2f \pm %05.2f} K$"%(T,TRACERSPARAMS['deltaT']) +\
        r" and $%05.2f \leq R \leq %05.2f kpc $"%(TRACERSPARAMS['Rinner'], TRACERSPARAMS['Router']) +\
        "\n" + f" and selected at {selectTime:3.2f} Gyr"+\
        f" weighted by mass"+\
        "\n"+f"{percentileLO:3.2f}% to {percentileUP:3.2f}% Mass Weighted Percentiles Shown", fontsize=12)
        # ax.axvline(x=vline, c='red')

        plt.tight_layout()
        plt.subplots_adjust(top=0.90, hspace =-.25)

        opslaan = f"Tracers_selectSnap{int(TRACERSPARAMS['selectSnap'])}_snap{int(snap)}_T{T}_{dataKey}_PDF.pdf"
        plt.savefig(opslaan, dpi = DPI, transparent = False)
        print(opslaan)
        plt.close()
