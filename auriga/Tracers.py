import numpy as np
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *
from Snapper import *
simfile='/home/universe/spxtd1-shared/ISOTOPES/output/' # set paths to simulation
snapnum=127 # set snapshot to look at

#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
elements_mass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

Zsolar = 0.0127

omegabaryon0 = 0.048


# Single FvdV Projection:
# load in the subfind group files
snap_subfind = load_subfind(snapnum,dir=simfile)

# load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
snapGas     = gadget_readsnap(snapnum, simfile, hdf5=True, loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
snapTracers = gadget_readsnap(snapnum, simfile, hdf5=True, loadonlytype = [6], lazy_load=True)

print(f" SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

Snapper1 = Snapper()
snapGas     = Snapper1.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=0)

#Convert Units
## Make this a seperate function at some point??
snapGas.pos   *= 1e3 #[kpc]

snapGas.vol *= 1e9 #[kpc^3]

meanweight = sum(snapGas.gmet[:,0:9], axis = 1) / ( sum(snapGas.gmet[:,0:9]/elements_mass[0:9], axis = 1) + snapGas.ne*snapGas.gmet[:,0] )
Tfac = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53
snapGas.data['T'] = snapGas.u / Tfac # K

###
##  Selection   ##
###
target=5.
delta = 0.25

print("cond")
#Select INDICES of T = 10^5+-delta K
Cond = np.where((snapGas.data['T']>=1.*10**(target-delta)) & (snapGas.data['T']<=1.*10**(target+delta)))

print("CellIDs")
#Select CellIDs for Cond
CellIDs = snapGas.id[Cond]

#Select the data for Cells that meet Cond
#   Does this by making a new dictionary from old data
#       but only selects values where index of entry (given by enumerate)
#           is found in Cond list (with [0] to access the list as array, not tuple)
print("Cells")
Cells = {key: value for ind, (key, value) in enumerate(snapGas.data.items()) if ind in Cond[0]}

#Select Parent IDs in Cond list
#   Selecting from cells with a tracer, which cell IDs are have cell IDs that meet Cond
print("Parents")
Parents = np.isin(CellIDs,snapTracers.prid)

print("Cells with tracers")
Cells = {key: value for ind, (key, value) in enumerate(snapGas.data.items()) if Parents[ind] == True}
#Select INDICES of the tracers who are in a cell which meets Cond
print("ParentsIndices")
ParentsIndices = np.where(Parents)

#Finally, select TRACER IDs who are in cells that meet Cond
print("Tracers")
Tracers = snapTracers.trid[ParentsIndices]




print("Snapnum")
Cells['snapnum']=[snapnum for i in range(0,len(Cells['T']))]
subset = 100
print("Plot!")
fig = plt.figure()
ax = plt.gca()
ax.scatter(Cells['snapnum'][:subset],Cells['T'][:subset])
ax.set_yscale('log')
ax.set_xlim(snapnum-1,snapnum+1)
ax.set_xlabel("Snap Number")
ax.set_ylabel("Temperature [K]")
ax.set_title(f"Subset of first {subset:0.0f} Cell Temperatures Containing Tracers")
opslaan = f'Tracers.png'
plt.savefig(opslaan, dpi = 500, transparent = True)
print(opslaan)
plt.close()
