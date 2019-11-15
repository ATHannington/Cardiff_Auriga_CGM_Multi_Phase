"""
Title:              PlotCellTemperatures.py
Created by:         Andrew T. Hannington
Use with:           RadTrans_*.f90
                        Created by: A. P. Whitworth et al.

Date Created:       2019

Usage Notes:
            Generates plots of Cell temperatures versus cell number - Equivalent
            to radius in the 1D cylinder case.

Known bugs: //

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import math

#-----------------------------------------------------------------------------#
#                   Functions                                                 #
#-----------------------------------------------------------------------------#
def frac_error(obs,exp):
    """
        Fractional Error function!
            This function is related to the Chi Square Test but does not use
            the variance as we do not know the expected true spread of the data.
                    def frac_error(obs,exp)

        INPUTS:     Observations: Obs [list]
                    Expected values: exp [list]

        OUTPUT:     Fractional Error Value: fracError [float]
    """
    N = float(len(obs))
    deviations = [((o-e)**2)/(e**2) for o in obs for e in exp]
    fracError = math.sqrt(np.sum(deviations))
    fracRms = math.sqrt(np.sum(deviations)*(1./N))

    return fracError,fracRms

#-----------------------------------------------------------------------------#
### Setup ###

importstring = "detailed-balance-RT_cell_temperatures.csv"
delta = 10
### Read in Data ###
read_data = pd.read_csv(importstring,delimiter=" ", \
skipinitialspace =True, na_values=["-Infinity", "Infinity"])

### Read Equib Temp and LPpTOT ###
equibTemp = read_data['EquT'][0]
LPpTOT = read_data['PTOT'][0]

### Write String Generate ###
outputstring = f"T={equibTemp:.2f}_LPpTOT={LPpTOT:.2E}_py-analysis.csv"


### Get Fractional Error Values ###
expected = [equibTemp for i in range(len(read_data['temp']))]

fracErrStd,fracRmsStd=frac_error(obs=read_data['temp'],exp=expected)
fracErrLucy,fracRmsLucy=frac_error(obs=read_data['lucy'],exp=expected)


### Find percentage values, then mean and Stnd. Dev. ###
read_data['temp'] = read_data['temp']/equibTemp
read_data['lucy'] = read_data['lucy']/equibTemp
read_data['ErrS']=read_data['ErrS']/equibTemp
read_data['ErrL']=read_data['ErrL']/equibTemp

#Inverse weights to make worse errors have less affect on the mean
weightsStd = np.array([(1./(read_data['ErrS'][i])) for i in range(len(read_data['ErrS']))])
weightsLucy = np.array([(1./(read_data['ErrL'][i])) for i in range(len(read_data['ErrL']))])

#Average data with inverse weights as above
meanStd = np.average(read_data['temp'],weights=weightsStd)
meanLucy = np.average(read_data['lucy'],weights=weightsLucy)

#Get Standard Deviation
sdStd = read_data['temp'].std()
sdLucy = read_data['lucy'].std()


### Create Output DataFrame ###
results=pd.DataFrame({'Mean':[meanStd,meanLucy],'Std':[sdStd,sdLucy]\
,'frac_error':[fracErrStd,fracErrLucy],\
'frac_RMS':[fracRmsStd,fracRmsLucy],\
'T_B':[equibTemp,None],'LPpTOT':[LPpTOT,None]})

#Rename Rows to standard and Lucy
results.rename(index={0:'standard',1:'lucy'}, inplace=True)

### Plotting Lists ###
meanStdList = [meanStd for i in range(len(read_data['cell']))]
meanLucyList = [meanLucy for i in range(len(read_data['cell']))]
### Write analysis to file ###
print("Writing to file!")

results.to_csv(path_or_buf=outputstring)                                       # Write dataframe to file

### Plot! ###

print("Beginning Plots!")

fig, ax = plt.subplots()


#Set style options
opacity = 0.20
colourStd = "Blue"
colourLucy = "Red"
styleStd = "--"
styleLucy = ":"

#Plot cell by cell values of Temp.
ax.plot(read_data['cell'],read_data['temp'],color=colourStd,label="Standard method")
ax.plot(read_data['cell'],read_data['lucy'],color=colourLucy,label="Lucy(1999) method")

#Shade the error regions around the mean
ax.fill_between(read_data['cell'], read_data['temp']+read_data['ErrS'], read_data['temp']-read_data['ErrS'],\
 facecolor=colourStd,alpha=opacity,interpolate=True)
ax.fill_between(read_data['cell'], read_data['lucy']+read_data['ErrL'], read_data['lucy']-read_data['ErrL'],\
 facecolor=colourLucy,alpha=opacity,interpolate=True)

#Plot the overall mean
ax.plot(read_data['cell'],meanStdList,color=colourStd,linestyle=styleStd,label="Standard method mean")
ax.plot(read_data['cell'],meanLucyList,color=colourLucy,linestyle=styleLucy,label="Lucy(1999) method mean")

ax.set_xlabel("Cell Number")
ax.set_ylabel("Cell Temperature Fraction [dimensionless]")

#Set grid lines on both major and minor axis ticks in both x and y.
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(which="both")

ax.legend()

plt.title("Cell Temperature Fraction versus Cell Number from Detailed Balance MCRT" \
+"\n"+ "Equib Temperature = " + str(round(equibTemp,2)) + " [K]." + f" LPpTOT = {LPpTOT:.2E}")
plt.savefig("temp_DB-RT"+f"_LPpTOT={LPpTOT:.0E}_T={equibTemp:.0f}" +".png")


plt.show()
