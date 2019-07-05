import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

### Setup ###

importstring = "detailed-balance-RT_cell_temperatures.csv"

### Read in Data ###
read_data = pd.read_csv(importstring,delimiter=" ", \
skipinitialspace =True, na_values=["-Infinity", "Infinity"])

### Read Equib Temp and LPpTOT ###
equibTemp = read_data['EquT'][0]
LPpTOT = read_data['PTOT'][0]

### Write String Generate ###
outputstring = f"T={equibTemp:.2f}_LPpTOT={LPpTOT:.2E}_py-analysis.csv"


sd= [np.std(read_data['temp'][i:i+delta])








### Write analysis to file ###

print("Writing to file!")


d = {'Mean':[meanTemp,meanLucy],'Mean%':[meanPercentTemp,meanPercentLucy],\
"EquibT":[equibTemp,equibTemp],"LPpTOT":[LPpTOT,LPpTOT]}											# Add Data to temp dictionary	

outputDF = pd.DataFrame(data=d)																		# Add Data to DataFrame

outputDF.rename(index={0:'Standard',1:'Lucy'}, inplace=True)										# Add Row index names

outputDF.to_csv(path_or_buf=outputstring)															# Write dataframe to file



### Plot! ###

print("Beginning Plots!")

fig, ax = plt.subplots()	

ax.plot(read_data['cell'],read_data['temp'],color='black',label="Standard method")
ax.plot(read_data['cell'],read_data['lucy'],color='red',label="Lucy(1999) method")

ax.set_xlabel("Cell Number")
ax.set_ylabel("Cell Temperature Fraction [dimensionless]")
ax.set_xlim(1,max(read_data['cell']))
ax.set_ylim(min(min(read_data['lucy']),min(read_data['temp'])),max(max(read_data['lucy']),max(read_data['temp'])))

ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.grid(which="both")

ax.legend()


plt.title("Cell Temperature Fraction versus Cell Number"+"\n"+"from Detailed Balance RT" \
+"\n"+ "Equib Temperature = " + str(round(equibTemp,2)) + " [K]." + f" - LPpTOT = {LPpTOT:.2E}")
plt.savefig("cells-vs-temperature_detailed-balance-RT.png")
plt.show()