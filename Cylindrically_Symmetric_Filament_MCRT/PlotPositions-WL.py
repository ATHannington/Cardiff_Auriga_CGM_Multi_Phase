import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

importstring = "positions-WL.csv"
subroutine ="RT_Cyl1DSchuster_DetailedBalance"
LPpTOT = 1E6
numberRows = "ALL"#20000
read_data = pd.read_csv(importstring,delimiter=" ", \
skipinitialspace =True, na_values=["-Infinity", "Infinity"])#,nrows=numberRows)

mainplot = plt.scatter(read_data['x'],read_data['y'],c=read_data['l'] \
,cmap="Blues_r")
plt.xlim(min(read_data['x']),max(read_data['x']))
plt.ylim(min(read_data['y']),max(read_data['y']))
plt.xlabel('x coordinate [pc]')
plt.ylabel('y coordinate [pc]')
plt.title(f'Plot of X and Y coordinates for {LPpTOT:.2E} LP packets' + '\n'\
+f'during {subroutine}')
cbar = plt.colorbar(mainplot,label="Wavelength [Microns]")
plt.savefig(f"postions_vs._wl_{subroutine}_n={numberRows}_LPpTOT={LPpTOT:.2E}.png")
plt.show()
