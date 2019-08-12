import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

importstring = "positions.csv"
numberRows = 20000
read_data = pd.read_csv(importstring,delimiter=" ", \
skipinitialspace =True, na_values=["-Infinity", "Infinity"]\
,nrows=numberRows)

plt.scatter(read_data['x'],read_data['y'])
plt.xlim(min(read_data['x']),max(read_data['x']))
plt.ylim(min(read_data['y']),max(read_data['y']))
plt.xlabel('x coordinate [pc]')
plt.ylabel('y coordinate [pc]')
plt.title('Plot of X and Y coordinates for 1E5 LP packets' + '\n'\
+'during RT_Cyl1DSchuster_DetailedBalance')
plt.show()
