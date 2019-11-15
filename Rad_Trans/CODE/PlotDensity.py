"""
Title:              PlotDensity.py
Created by:         Andrew T. Hannington
Use with:           RadTrans_*.f90
                        Created by: A. P. Whitworth et al.

Date Created:       2019

Usage Notes:
            Generates plots of Cell density versus cell number - Equivalent
            to radius in the 1D cylinder case.
            Gives central density and Schuster Exponent P value on plot.

Known bugs: //

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Setup ###

importstring = "density_cell.csv"
### Read in Data ###
read_data = pd.read_csv(importstring,delimiter=" ", \
skipinitialspace =True, na_values=["-Infinity", "Infinity"])

print("***+++***")
print("Plot Denisty from Rad Trans")
print("***+++***")
print()
print(f"Schuster exponent p = {int(read_data['shp'][0]):d}")
print(r"$Rho_{0}$ density of Schuster filament in $g cm^{-3}$ = " + f"{read_data['rh0'][0]:.5E}")
print()

plt.scatter(read_data['pos'],read_data['rho'])
plt.ylim(min(read_data['rho']),max(read_data['rho']))
plt.xlabel("Cell Number [/]")
plt.ylabel(r"Cell Density $[g$ $cm^{-3}]$")
plt.title(f"Density vs. Cell number" + f" -- Schuster profile of p = {int(read_data['shp'][0]):d}"\
+"\n"+r"$Rho_{0}$ = " + f"{read_data['rh0'][0]:.5E}"+r" $[g$ $cm^{-3}]$")

plt.savefig(f"density_vs_cell_schuster_p={int(read_data['shp'][0]):d}.png")

plt.show()
