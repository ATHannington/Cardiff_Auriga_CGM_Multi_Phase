import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gadget import *
from gadget_subfind import *
from const import *


simfile='/home/universe/spxtd1-shared/ISOTOPES/output/' # set paths to simulation
snapnum=127 # set snapshot to look at


gas_subfind = load_subfind(snapnum, dir=simfile) # load in the subfind group files
gas = gadget_readsnap(snapnum, simfile, hdf5=True, loadonly=['mass','pos'], loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = gas_subfind) # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs

haloid = 0 # look only at the central galaxy - index zero
cent = gas_subfind.data['fpos'][haloid,:] # subfind has calculated its centre of mass for you
gas.data['pos'] = (gas.data['pos'] - np.array(cent))*1e3 # use the subfind COM to centre the coordinates on the galaxy


hist,xedge,yedge=np.histogram2d(gas.pos[:,0],gas.pos[:,1], bins=500,range=[[-50,50],[-50,50]],weights=gas.mass,normed=False)



img1 = plt.imshow(hist,cmap='nipy_spectral',vmin=np.nanmin(hist),vmax=np.nanmax(hist),extent=[np.min(xedge),np.max(xedge),np.min(yedge),np.max(yedge)],origin='lower')
ax1 = plt.gca()
ax1.tick_params(labelsize=20.)
ax1.set_ylabel('X (kpc)', fontsize = 20.0) # Y label
ax1.set_xlabel('Y (kpc)', fontsize = 20) # X label

plt.savefig('My_Figure.pdf', bbox_inches='tight')
plt.close()
