import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import math

NSTEPS = 1000
IMPORTSTRING = "timing_vs_tau.csv"

## Import the data from CSV ###
read_data = pd.read_csv(IMPORTSTRING,delimiter=",", \
skipinitialspace =True, na_values=["-Infinity", "Infinity"])

## Set evaluators for schuster p = 0 and BBkBB background temp 29 ###
p_0_bool =  read_data['p'] == 0
BBkBB_29_bool = read_data['BBkBB'] == 29

### Eliminate any NaNs ###
tmp_data = read_data[p_0_bool & BBkBB_29_bool]
not_nan_bool = tmp_data['Time_s'].notnull()

### Make final DataFrame ###
final_data = tmp_data[not_nan_bool]

del tmp_data

# Fit 2nd order polynomial to data##
p_coeff = np.polyfit(final_data['TAUconst'],final_data['Time_s'],2)

## Obtain best fit coefficients ##
p = np.poly1d(p_coeff)

## Create blank x data ##
x = np.linspace(min(final_data['TAUconst']),max(final_data['TAUconst']),NSTEPS)

plt.plot(x,p(x),label=(r'Best fit $ax + b{x^{2}} + c$ :' + f'a={p[0]:.1f}, b={p[1]:.1f}, c={p[2]:.1f}')\
,color="red")
plt.scatter(final_data['TAUconst'],final_data['Time_s'], label='Data',color="black")
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\tau$ $[/]$')
plt.ylabel(r'$t_{Run}$ $[s]$')
plt.title(r'Program run time ($t_{Run}$) vs. Optical Depth ($\tau$)' +'\n'+
r'For Plummer-like density index $p=0$ and Thermal Equib. Temp $T=10.36K$')
plt.legend()


plt.show()
