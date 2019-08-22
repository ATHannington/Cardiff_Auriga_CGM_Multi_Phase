import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

importstring = "db_N-a_S-i_M-c_V-c_T-e_P-t_W-f.csv"
subroutine ="RT_Cyl1DSchuster_DetailedBalance"

read_data = pd.read_csv(importstring,delimiter=" ", \
skipinitialspace =True, na_values=["-Infinity", "Infinity", "NaN"])#,nrows=numberRows)
LPpTOT = read_data["P-t"][0]
equibTemp = read_data["T-e"][0]
wavelength = read_data["W-f"][0]


#------------------------------------------------------------------------------#


fig = plt.figure()
ax = plt.gca()

ax.scatter(read_data['M-c'],read_data['N-a'],color="red")
ax.set_xlim(min(read_data['M-c']),max(read_data['M-c']))
ax.set_ylim(min(read_data['N-a']),max(read_data['N-a']))
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.set_xlabel(r'Line Mass [$M_{\odot}/pc$]')
ax.set_ylabel(r'Number of Absorptions [/]')
plt.title('Number of Absorptions per cell vs. Line Mass of a cell.' + '\n'\
 +f'{LPpTOT:.2E} LP packets during {subroutine}')

fig.savefig(f"N-a_vs_M-c_LPpTOT={LPpTOT:.2E}_T={equibTemp:.2f}K.jpg")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

fig2 = plt.figure()
ax2 = plt.gca()

ax2.scatter(read_data['V-c'],read_data['S-i'],color="red")
ax2.set_xlim(min(read_data['V-c']),max(read_data['V-c']))
ax2.set_ylim(min(read_data['S-i']),max(read_data['S-i']))
ax2.set_xlabel(r'Line Volume [$pc^{-2}$]')
ax2.set_ylabel(f'Total Path Length at {wavelength:.2f}'+r'${\mu}m$ [$m$]')
plt.title(f'Line Volume vs. Total Path Length at {wavelength:.2f}'+r'${\mu}m$' + '\n'\
 +f'{LPpTOT:.2E} LP packets during {subroutine}')

fig2.savefig(f"S-i_vs_V-c_LPpTOT={LPpTOT:.2E}_T={equibTemp:.2f}K.jpg")

plt.show()
