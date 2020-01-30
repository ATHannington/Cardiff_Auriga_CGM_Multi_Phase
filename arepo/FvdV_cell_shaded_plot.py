from gadget import *
from gadget_subfind import *
import const as c #from const import *
import matplotlib       #This MUST come first!!!! before plt
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

snap = 127

sname = "halo_6"
PATH = '/home/universe/spxtd1-shared/ISOTOPES/output/'
s = gadget_readsnap(snap, snappath=PATH, loadonlytype=[0], lazy_load=True)
sf = load_subfind(snap, dir=PATH)

haloid = 0
cent = sf.data['fpos'][haloid,:]
s.data['pos'] = s.data['pos'] - np.array(cent)

s.pos *= 1e3
s.vol *= 1e9

print(s.redshift)


#element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
elements_mass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

Zsolar = 0.0127

omegabaryon0 = 0.048

rhocrit = 3. * (s.omega0 * (1+s.redshift)**3. + s.omegalambda) * (s.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)
rhomean = 3. * (s.omega0 * (1+s.redshift)**3.) * (s.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)


boxsize = 200.
boxlos = 50.
pixres = 1. #0.2
pixreslos = 0.1 #0.1

imgcent = [0,0,0]
xy = [0,2] #[0,1]


meanweight = sum(s.gmet[:,0:9], axis = 1) / ( sum(s.gmet[:,0:9]/elements_mass[0:9], axis = 1) + s.ne*s.gmet[:,0] )
Tfac = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53

s.data['T'] = s.u / Tfac # K

gasdens = s.rho / (c.parsec*1e6)**3. * c.msol * 1e10
gasX = s.gmet[:,0]
s.data['n_H'] = gasdens / c.amu * gasX # cm^-3
s.data['dens'] = gasdens / (rhomean * omegabaryon0/s.omega0) # rho / <rho>

s.data['Tdens'] = s.data['T'] *s.data['dens']


slice_nH = s.get_Aslice("n_H", box = [boxsize,boxsize], center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres), axes = xy, proj = False, numthreads=16)

slice_T = s.get_Aslice("T", box = [boxsize,boxsize], center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres), axes = xy, proj = False, numthreads=16)

proj_nH = s.get_Aslice("n_H", box = [boxsize,boxsize], center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres), nz = int(boxlos/pixreslos), boxz = boxlos, axes = xy, proj = True, numthreads=16)

proj_dens = s.get_Aslice("dens", box = [boxsize,boxsize], center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres), nz = int(boxlos/pixreslos), boxz = boxlos, axes = xy, proj = True, numthreads=16)

proj_T = s.get_Aslice("Tdens", box = [boxsize,boxsize], center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres), nz = int(boxlos/pixreslos), boxz = boxlos, axes = xy, proj = True, numthreads=16)


#PLOTTING TIME

xsize = 10.
ysize = 10.
fig, axes = plt.subplots(2, 2, figsize = (xsize,ysize), dpi = 500)

cmap = plt.get_cmap('Spectral')

pcm1 = axes[0,0].pcolormesh(slice_nH['x'], slice_nH['y'], np.transpose(slice_nH['grid']), norm = matplotlib.colors.LogNorm(vmin=1e-5, vmax=1e-1), cmap = cmap, rasterized = True)

pcm2 = axes[0,1].pcolormesh(proj_nH['x'], proj_nH['y'], np.transpose(proj_nH['grid'])/int(boxlos/pixreslos), norm = matplotlib.colors.LogNorm(vmin=1e-5, vmax=1e-1), cmap = cmap, rasterized = True)

pcm3 = axes[1,0].pcolormesh(slice_T['x'], slice_T['y'], np.transpose(slice_T['grid']), norm = matplotlib.colors.LogNorm(vmin=1e4, vmax=1e7), cmap = cmap, rasterized = True)

pcm4 = axes[1,1].pcolormesh(proj_T['x'], proj_T['y'], np.transpose(proj_T['grid']/proj_dens['grid']), norm = matplotlib.colors.LogNorm(vmin=1e4, vmax=1e7), cmap = cmap, rasterized = True)

for ii in range(0,2):
    for jj in range(0,2):
        axes[ii,jj].set_ylabel('Y (kpc)')#, fontsize = 20.0)
        axes[ii,jj].set_xlabel('X (kpc)')#, fontsize = 20.0)
        # axes[ii,jj].tick_params(labelsize=20.)

axes[0,0].set_title(r'Slice $n_H$')
axes[0,1].set_title(r'Projection $n_H$')
axes[1,0].set_title(r'Slice $T$')
axes[1,1].set_title(r'Projection $T$')

fig.colorbar(pcm1, ax = axes[0,0], orientation = 'horizontal',label=r'$n_H$ [$cm^{-3}$]')
fig.colorbar(pcm2, ax = axes[0,1], orientation = 'horizontal',label=r'$n_H$ [$cm^{-3}$]')
fig.colorbar(pcm3, ax = axes[1,0], orientation = 'horizontal',label=r'$T$ [$K$]')
fig.colorbar(pcm4, ax = axes[1,1], orientation = 'horizontal',label=r'$T$ [$K$]')

axes[0,0].set_aspect(1.0)
axes[0,1].set_aspect(1.0)
axes[1,0].set_aspect(1.0)
axes[1,1].set_aspect(1.0)

opslaan = 'Shaded_Cell.pdf'
plt.savefig(opslaan, dpi = 500, transparent = True)
print(opslaan)
plt.close()
