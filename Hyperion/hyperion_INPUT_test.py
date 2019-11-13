import numpy as np
import pandas as pd
from hyperion.util.constants import *
from hyperion.model import Model
# from hyperion.dust import SphericalDust
from hyperion.dust import IsotropicDust
from hyperion.model import ModelOutput
import math
import sys

#------------------------------------------------------------------------------#
NPHOTONS = int(1e6)

R = 0.5*pc#0.1500e+19#0.5*pc
SIGMASB = 5.670515e-5
BGfBB = 1.
CFwB = R #0.1500e+19
TEMP = 10.34
RHO0 = 0.1000e-18

TAUconst = 1.e-3
Ncellsxyz = 101

m = Model()
x = np.linspace(-pc, pc, Ncellsxyz)
y = np.linspace(-pc, pc, Ncellsxyz)
z = np.linspace(-pc, pc, Ncellsxyz)
m.set_cartesian_grid(x, y, z)


#------------------------------------------------------------------------------#
DRAINESKIP=80
DRAINEMAXLINE=1143-DRAINESKIP                                                               #line number where dust properties end
DRAINEMINLINE=0                                                                 #line number where dust properties start

lam, albedo, chi, = np.genfromtxt("draine_rv3.1.dat",skip_header=DRAINESKIP,usecols=(0,1,4),unpack=True)

lam = lam[DRAINEMINLINE:DRAINEMAXLINE]
albedo = albedo[DRAINEMINLINE:DRAINEMAXLINE]
chi = chi[DRAINEMINLINE:DRAINEMAXLINE]

nu = np.array([c/(l*1e-4) for l in lam])
#
# HOSKIP = 3
# HOMAXLINE = 67
# lamHO, chiHO = np.genfromtxt("Henning_Ossenkopf_94.dat",skip_header=HOSKIP,usecols=(2,3),unpack=True)
# lamHO = lamHO[:HOMAXLINE]
# chiHO = chiHO[:HOMAXLINE]
#
# albedoHO = np.zeros(shape=len(lamHO))
#
# nuHO = np.array([c/(l*1e-4) for l in lamHO])
#
# nu = np.append(nuHO,nu, axis=0)
# albedo = np.append(albedoHO, albedo, axis=0)
# chi = np.append(chiHO,chi,axis=0)
### Extend the dataset to go to zero linearly after nEXTRA. Adds far-IR to model
nEXTRA = int(1e2)
YMIN = 0.

GRADnu = (nu[0] - YMIN)/float(nEXTRA)
GRADalb = (albedo[0] - YMIN)/float(nEXTRA)
GRADchi = (chi[0] - YMIN)/float(nEXTRA)


for i in range(0,nEXTRA-1):
    nuval = nu[0] - GRADnu
    albval = albedo[0] - GRADalb
    chival = chi[0] - GRADchi
    nu = np.insert(nu,0,nuval,axis=0)
    albedo = np.insert(albedo,0,albval,axis=0)
    chi = np.insert(chi,0,chival,axis=0)

d = IsotropicDust(nu, albedo, chi)

d.write('mydust.hdf5')
d.plot('mydust.png')

# sys.exit('Andy48')

#### This Section of Code should set a constant tau regardless on temp ###!!
lamMax = (0.288*(1.e4))/TEMP
# print("lambda max = ", lamMax)
# for i in range(1,len(lam)):
#   if(lam[i] >= lamMax):
#     LPlFixed = i
#     chiBar = chi[i]
#     break
#
# CFsig = RHO0*(2.*R)
#
# RHO0 = (RHO0*TAUconst)/(CFsig*chiBar*(1.0-albedo[LPlFixed]))
#
# print(RHO0)

# m.add_density_grid(np.ones(shape=(100,100,100)),d)#, 'kmh.hdf5')                             # Need density file
density = np.zeros(m.grid.shape)
density[(m.grid.gx ** 2 + m.grid.gy ** 2 + m.grid.gz ** 2) < R ** 2] = RHO0
m.add_density_grid(density, d)
#------------------------------------------------------------------------------#
source = m.add_point_source()
# source = m.add_map_source()
# source.map = np.ones(m.grid.shape)
LUMINOSITY = ((4.0*math.pi*SIGMASB)*(CFwB**2)*BGfBB*(TEMP**4))
print("LUMINOSITY/lsun = ")
print(LUMINOSITY/lsun)
source.luminosity = LUMINOSITY
source.temperature = TEMP
half = int(len(x)/2)
pos = (x[half],y[half],z[half])
source.position = pos

#------------------------------------------------------------------------------#
# Add multi-wavelength image for a single viewing angle
image = m.add_peeled_images(sed=False, image=True)
image.set_wavelength_range(20, 1., 1000.)
image.set_viewing_angles([60.], [80.])
image.set_image_size(400, 400)
image.set_image_limits(-1.*pc,pc,-1.*pc,pc)

#------------------------------------------------------------------------------#
# m.n_inter_max=int(1e10)

m.set_n_initial_iterations(2)
m.set_raytracing(True)
m.set_n_photons(initial=NPHOTONS, imaging=NPHOTONS, raytracing_sources=1, raytracing_dust=NPHOTONS)
m.set_convergence(True, percentile=99., absolute=2., relative=1.02)

#------------------------------------------------------------------------------#
n_processes =2
m.write('example.rtin')
m.run('model.rtout', mpi=True, n_processes=n_processes, overwrite=True)
