"""
Title:              PlotProbabilities.py
Created by:         Andrew T. Hannington
Use with:           RadTrans_*.f90
                        Created by: A. P. Whitworth et al.

Date Created:       01/10/2019

Usage Notes:
			Plots timings of simulations versus optical depth
            PLEASE NOTE:    There is *no* current script to generate the table
                            of data needed for this file. The file was written
                            manually into a MS Excel .csv file, and the sims
                            timed using the linux bash command "time".

Known bugs: //


"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import math

NSTEPS = 1000
EQUIBTEMP = 10.34 #[K]

#Select which settings used are being plotted
P = 0
BGKBB = 29

# Import file
IMPORTSTRING = "timing_vs_tau.csv"

## Import the data from CSV ###
read_data = pd.read_csv(IMPORTSTRING,delimiter=",", \
skipinitialspace =True, na_values=["-Infinity", "Infinity"])

## Set evaluators for schuster p = 0 and BBkBB background temp 29 ###
p_bool =  read_data['p'] == P
BBkBB_bool = read_data['BBkBB'] == BGKBB

### Eliminate any NaNs ###
tmp_data = read_data[p_bool & BBkBB_bool]
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

#Plot best fit
plt.plot(x,p(x),label=(r'Best fit $a{x^{2}} + bx + c$ :' + f'a={p[0]:.1f}, b={p[1]:.1f}, c={p[2]:.1f}')\
,color="red")

#Plot real data
plt.scatter(final_data['TAUconst'],final_data['Time_s'], label='Data',color="black")
plt.yscale('log')
plt.xscale('log')
plt.xlabel(r'$\tau$ $[/]$')
plt.ylabel(r'$t_{Run}$ $[s]$')
plt.title(r'Program run time ($t_{Run}$) vs. Optical Depth ($\tau$)' +'\n'+
f'For Plummer-like density index p={P:.0f} and Thermal Equib. Temp T={EQUIBTEMP:.2f}')
plt.legend()


plt.show()
