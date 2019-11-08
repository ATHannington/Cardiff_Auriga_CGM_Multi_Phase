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

R = 0.1500E+19#0.5*pc
SIGMASB = 5.670515e-5
BGfBB = 1.
CFwB = 0.1500E+19
TEMP = 6000
RHO0 = 0.1000E-18

TAUconst = 1

m = Model()
x = np.linspace(-pc, pc, 101)
y = np.linspace(-pc, pc, 101)
z = np.linspace(-pc, pc, 101)
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

d = IsotropicDust(nu, albedo, chi)

d.write('mydust.hdf5')
d.plot('mydust.png')


#### This Section of Code should set a constant tau regardless on temp ###!!
lamMax = (0.288*(1.e4))/TEMP
print("lambda max = ", lamMax)
for i in range(1,len(lam)):
  if(lam[i] >= lamMax):
    LPlFixed = i
    chiBar = chi[i]
    break

CFsig = RHO0*(2.*R)

RHO0 = (RHO0*TAUconst)/(CFsig*chiBar*(1.0-albedo[LPlFixed]))

print(RHO0)

# m.add_density_grid(np.ones(shape=(100,100,100)),d)#, 'kmh.hdf5')                             # Need density file
density = np.zeros(m.grid.shape)
density[(m.grid.gx ** 2 + m.grid.gy ** 2 + m.grid.gz ** 2) < R ** 2] = RHO0
m.add_density_grid(density, d)
#------------------------------------------------------------------------------#
source = m.add_point_source()
# source = m.add_map_source()
# source.map = np.ones(m.grid.shape)
LUMINOSITY = ((4.0*math.pi*SIGMASB)*(CFwB**2)*BGfBB*(TEMP**4))/float(NPHOTONS)
print("LUMINOSITY/lsun = ")
print(LUMINOSITY/lsun)
source.luminosity = 1000*lsun#LUMINOSITY
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
m.set_n_photons(initial=NPHOTONS, imaging=NPHOTONS, raytracing_sources=1, raytracing_dust=(NPHOTONS/10))
m.set_convergence(False)#True, percentile=99., absolute=2., relative=1.02)

#------------------------------------------------------------------------------#
n_processes =2
m.write('example.rtin')
m.run('model.rtout', mpi=True, n_processes=n_processes, overwrite=True)
