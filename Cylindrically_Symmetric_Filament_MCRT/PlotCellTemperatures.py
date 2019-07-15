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
    deviations = [((o-e)**2)/(e**2) for o in obs for e in exp]
    fracError = math.sqrt(np.sum(deviations))

    return fracError

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

fracErrStd=frac_error(obs=read_data['temp'],exp=expected)
fracErrLucy=frac_error(obs=read_data['lucy'],exp=expected)


### Find percentage values, then mean and Stnd. Dev. ###
read_data['temp'] = read_data['temp']/equibTemp
read_data['lucy'] = read_data['lucy']/equibTemp


meanStd = read_data['temp'].mean()
meanLucy = read_data['lucy'].mean()
sdStd = read_data['temp'].std()
sdLucy = read_data['lucy'].std()


### Create Output DataFrame ###
results=pd.DataFrame({'Mean':[meanStd,meanLucy],'Std':[sdStd,sdLucy]\
,'frac_error':[fracErrStd,fracErrLucy],'T_B':[equibTemp,None],'LPpTOT':[LPpTOT,None]})

results.rename(index={0:'standard',1:'lucy'}, inplace=True)                    #Rename Rows to standard and Lucy

### Plotting Lists ###
meanStdList = [meanStd for i in range(len(read_data['cell']))]
meanLucyList = [meanLucy for i in range(len(read_data['cell']))]

upperStd = [(meanStd+sdStd) for i in range(len(read_data['cell']))]
lowerStd = [(meanStd-sdStd) for i in range(len(read_data['cell']))]

upperLucy = [(meanLucy+sdLucy) for i in range(len(read_data['cell']))]
lowerLucy = [(meanLucy-sdLucy) for i in range(len(read_data['cell']))]

### Write analysis to file ###
print("Writing to file!")

results.to_csv(path_or_buf=outputstring)                                       # Write dataframe to file

### Plot! ###

print("Beginning Plots!")

fig, ax = plt.subplots()

ax.plot(read_data['cell'],read_data['temp'],color='black',label="Standard method")
ax.plot(read_data['cell'],read_data['lucy'],color='red',label="Lucy(1999) method")

ax.plot(read_data['cell'],meanStdList,color='cyan',label="Standard method mean and 1 SD")
ax.plot(read_data['cell'],meanLucyList,color='green',label="Lucy(1999) method mean and 1 SD")

ax.fill_between(read_data['cell'], upperStd, lowerStd,\
facecolor='cyan',alpha=0.5,interpolate=True)
ax.fill_between(read_data['cell'],upperLucy, lowerLucy,\
facecolor='green',alpha=0.5,interpolate=True)


ax.set_xlabel("Cell Number")
ax.set_ylabel("Cell Temperature Fraction [dimensionless]")
ax.set_xlim(1,max(read_data['cell']))
ax.set_ylim(min(min(read_data['lucy']),min(read_data['temp'])),max(max(read_data['lucy']),max(read_data['temp'])))

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(which="both")

ax.legend()


plt.title("Cell Temperature Fraction versus Cell Number from Detailed Balance MCRT" \
+"\n"+ "Equib Temperature = " + str(round(equibTemp,2)) + " [K]." + f" LPpTOT = {LPpTOT:.2E}")
plt.savefig("cells-vs-temperature_detailed-balance-RT"+f"_LPpTOT={LPpTOT:.2E}_T={equibTemp:.2f}" +".png")
plt.show()
