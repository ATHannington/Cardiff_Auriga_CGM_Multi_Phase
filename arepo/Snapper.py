import numpy as np
import matplotlib
matplotlib.use('Agg')   #For suppressing plotting on clusters
import matplotlib.pyplot as plt
import const as c
from gadget import *
from gadget_subfind import *



simfile='/home/universe/spxtd1-shared/ISOTOPES/output/' # set paths to simulation
snapnum=127 # set snapshot to look at

class Snapper(object):
    """
    Class for interacting with Arepo Snaps using arepo-snap-utils.

    Default imports required:

        import numpy as np
        import matplotlib
        matplotlib.use('Agg')   #For suppressing plotting on clusters
        import matplotlib.pyplot as plt
        import const as c
        from gadget import *
        from gadget_subfind import *

    """
#------------------------------------------------------------------------------#
#          __init__
#------------------------------------------------------------------------------#
    def __init__(self):
        """Initialisation Defines some necessary Global parameters:
            elements        : Element key strings list
            elements_Z      : Element Proton Number
            elements_mass   : Element mass unit
            elements_solar  : Element Solar abundances
            Zsolar          : Solar metallicity
            omegabaryon0    : Omega Baryon 0 - Baryonic Fraction??
        """
        global elements, elements_Z, elements_mass, elements_solar, Zsolar, omegabaryon0

        #element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
        elements       = ['H',  'He',  'C',   'N',   'O',   'Ne',  'Mg',  'Si',  'Fe',  'Y',   'Sr',  'Zr',  'Ba',   'Pb']
        elements_Z     = [1,    2,     6,     7,     8,     10,    12,    14,    26,    39,    38,    40,    56,     82]
        elements_mass  = [1.01, 4.00,  12.01, 14.01, 16.00, 20.18, 24.30, 28.08, 55.85, 88.91, 87.62, 91.22, 137.33, 207.2]
        elements_solar = [12.0, 10.93, 8.43,  7.83,  8.69,  7.93,  7.60,  7.51,  7.50,  2.21,  2.87,  2.58,  2.18,   1.75]

        Zsolar = 0.0127

        omegabaryon0 = 0.048
        return

#------------------------------------------------------------------------------#
#          Set Centre to [0,0,0] centering on COM of Halo number HaloID
#                   Where HaloID=0 is the biggest halo.
#------------------------------------------------------------------------------#

    def SetCentre(self,Snap,Snap_SubFind,HaloID):

        HaloCentre = Snap_SubFind.data['fpos'][HaloID,:] # subfind has calculated its centre of mass for you
        Snap.data['pos'] = (Snap.data['pos'] - np.array(HaloCentre)) # use the subfind COM to centre the coordinates on the galaxy

        print(f" SnapShot loaded at RedShift z={Snap.redshift:0.05e}")

        return Snap

#------------------------------------------------------------------------------#
#           Plot 1x Histogram 2d
#------------------------------------------------------------------------------#

    def PlotHistogram(self, Snap, SnapNum, Nbins=500,Axes=[0,1],Range = [[-50,50],[-50,50]],WeightsLabel='mass',Normed=False):
        """
        Function for Histogram 2D plotting an individual SnapShot.
        Args:
            Nbins               : Opt. Number of Histogram bins : Default = 500
            Axes                : Opt. Axes Selection           : Default = [0,1] == ['X','Y']
            Range               : Opt. Axis Range in kpc        : Default = [[-50,50],[-50,50]]
            WeightsLabel        : Opt. Weight bins by param.    : Default = 'mass'
            Normed              : Opt. Normalise bins?          : Default = False. True is NOT recommended!
        """
        AxesLabels = ['X','Y','Z']
        if (WeightsLabel == 'mass'):
            Weights = Snap.mass
        else:
            print("Unknown Weights Flag! Setting to Default 'mass'!")
            Weights = Snap.mass

        hist,xedge,yedge=np.histogram2d(Snap.pos[:,Axes[0]],Snap.pos[:,Axes[1]] \
        , bins=Nbins,range=Range,weights=Weights,normed=Normed)

        img1 = plt.imshow(hist,cmap='nipy_spectral',vmin=np.nanmin(hist) \
        ,vmax=np.nanmax(hist)\
        ,extent=[np.min(xedge),np.max(xedge),np.min(yedge),np.max(yedge)],origin='lower')
        ax1 = plt.gca()
        ax1.tick_params(labelsize=20.)
        ax1.set_ylabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize = 20.0) # Y label
        ax1.set_xlabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize = 20.0) # X label

        cbar = plt.colorbar()
        cbar.set_label(f'{WeightsLabel}')

        SnapNum = str(SnapNum).zfill(3);
        plt.savefig(f'Histogram2d_{WeightsLabel}_{SnapNum}.png', bbox_inches='tight')
        plt.close()

        return img1,ax1

#------------------------------------------------------------------------------#
#           Plot (NSnaps - Start)x Histogram 2d
#
#               CURRENTLY ONLY DEFAULTS SUPPORTED
#------------------------------------------------------------------------------#

    def HistogramMovieLoop(self,SimDirectory,NSnaps=127,Start=10):
        """
        Function for Histogram 2D plotting of a loop (Start to NSnaps) Snaps.
        Args:
            SimDirectory        : REQUIRED! Directory path for simulation
            NSnaps              : Opt. Number of SnapShots      : Default = 127
            Start               : Opt. Start Snapshot           : Default = 10 . Below this Halo finder may fail
        """
        # import matplotlib.animation as manimation

        self.SimDirectory = SimDirectory

        print("Let's make a Movie!")
        snapper = Snapper()
        for ii in range(Start,NSnaps+1):
            #Create child Snapper for each histogram. Caution, don't add to self
            # or data will stack up and bug out.

            #Load Snap and plot histogram with default options
            #
            ###
            # Andy: generalise this later, please!
            ###
            #
            #


            # load in the subfind group files
            snap_subfind = load_subfind(ii,dir=simfile)

            # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
            snap = gadget_readsnap(ii, simfile, hdf5=True, loadonly=['mass','pos'], loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)


            snap = snapper.SetCentre(Snap=snap,Snap_SubFind=snap_subfind,HaloID=0)

            #Convert Units
            ## Make this a seperate function at some point??
            snap.pos *= 1e3 #[kpc]
            snap.vol *= 1e9 #[kpc^3]

            snapper.PlotHistogram(Snap=snap,SnapNum=ii)

            #Print percentage complete
            print(f"{(float(ii + 1 - Start)/float(NSnaps - Start))*100.0 : 0.03f}","% complete")
            #Delete child snapper, to free up space and prevent data leakage

        return

#------------------------------------------------------------------------------#
#         Plot Projections and Slices
#------------------------------------------------------------------------------#

    def PlotProjections(self,Snap,SnapNum,Axes=[0,2],boxsize = 200.,boxlos = 50.,pixres = 1.,pixreslos = 0.1, DPI = 500):
        AxesLabels = ['X','Y','Z']
        imgcent =[0.,0.,0.]

        rhocrit = 3. * (Snap.omega0 * (1+Snap.redshift)**3. + Snap.omegalambda) * (Snap.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)
        rhomean = 3. * (Snap.omega0 * (1+Snap.redshift)**3.) * (Snap.hubbleparam * 100*1e5/(c.parsec*1e6))**2. / ( 8. * pi * c.G)


        meanweight = sum(Snap.gmet[:,0:9], axis = 1) / ( sum(Snap.gmet[:,0:9]/elements_mass[0:9], axis = 1) + Snap.ne*Snap.gmet[:,0] )
        Tfac = 1. / meanweight * (1.0 / (5./3.-1.)) * c.KB / c.amu * 1e10 * c.msol / 1.989e53

        gasdens = Snap.rho / (c.parsec*1e6)**3. * c.msol * 1e10
        gasX = Snap.gmet[:,0]

        Snap.data['T'] = Snap.u / Tfac # K
        Snap.data['n_H'] = gasdens / c.amu * gasX # cm^-3
        Snap.data['dens'] = gasdens / (rhomean * omegabaryon0/Snap.omega0) # rho / <rho>
        Snap.data['Tdens'] = Snap.data['T'] *Snap.data['dens']


        slice_nH    = Snap.get_Aslice("n_H", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         axes = Axes, proj = False, numthreads=16)

        slice_T     = Snap.get_Aslice("T", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         axes = Axes, proj = False, numthreads=16)

        proj_nH     = Snap.get_Aslice("n_H", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=16)

        proj_dens   = Snap.get_Aslice("dens", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=16)

        proj_T      = Snap.get_Aslice("Tdens", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=16)


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
                axes[ii,jj].set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)')#, fontsize = 20.0)
                axes[ii,jj].set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)')#, fontsize = 20.0)
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

        opslaan = f'Shaded_Cell_{SnapNum}.png'
        plt.savefig(opslaan, dpi = DPI, transparent = True)
        print(opslaan)
        plt.close()
#------------------------------------------------------------------------------#
#           Plot (NSnaps - Start)x Projections
#
#               CURRENTLY ONLY DEFAULTS SUPPORTED
#------------------------------------------------------------------------------#

    def ProjectionMovieLoop(self,SimDirectory,NSnaps=127,Start=10):
        """
        Function for Histogram 2D plotting of a loop (Start to NSnaps) Snaps.
        Args:
            SimDirectory        : REQUIRED! Directory path for simulation
            NSnaps              : Opt. Number of SnapShots      : Default = 127
            Start               : Opt. Start Snapshot           : Default = 10 . Below this Halo finder may fail
        """
        # import matplotlib.animation as manimation

        self.SimDirectory = SimDirectory

        print("Let's make a Movie!")
        snapper = Snapper()
        for ii in range(Start,NSnaps+1):
            #Create child Snapper for each histogram. Caution, don't add to self
            # or data will stack up and bug out.

            #Load Snap and plot histogram with default options
            #
            ###
            # Andy: generalise this later, please!
            ###
            #
            #


            # load in the subfind group files
            snap_subfind = load_subfind(ii,dir=simfile)

            # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
            snap = gadget_readsnap(ii, simfile,loadonlytype = [0],lazy_load=True)

            snap = snapper.SetCentre(Snap=snap,Snap_SubFind=snap_subfind,HaloID=0)

            #Convert Units
            ## Make this a seperate function at some point??
            snap.pos *= 1e3 #[kpc]
            snap.vol *= 1e9 #[kpc^3]

            snapper.PlotProjections(Snap=snap,SnapNum=ii,DPI=200)

            #Print percentage complete
            print(f"{(float(ii + 1 - Start)/float(NSnaps - Start))*100.0 : 0.03f}","% complete")
            #Delete child snapper, to free up space and prevent data leakage

        return
#===============================================================================#

#===============================================================================#

#------------------------------------------------------------------------------#
#           Run Code!
#------------------------------------------------------------------------------#

#Histogram:
#
# # load in the subfind group files
# snap_subfind = load_subfind(snapnum,dir=simfile)
#
# # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
# snap = gadget_readsnap(snapnum, simfile, hdf5=True, loadonly=['mass','pos'], loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
#
# Snapper1 = Snapper()
# snap = Snapper1.SetCentre(Snap=snap,Snap_SubFind=snap_subfind,HaloID=0)
#
# #Convert Units
# ## Make this a seperate function at some point??
# snap.pos *= 1e3 #[kpc]
# snap.vol *= 1e9 #[kpc^3]
#
# Snapper1.PlotHistogram(Snap=snap,SnapNum=snapnum)

#------------------------------------------------------------------------------#

# Histogram 2D Movie:
# Snapper2 = Snapper()
# Snapper2.HistogramMovieLoop(simfile,NSnaps=127,Start=10)

#------------------------------------------------------------------------------#

#Single FvdV Projection:
# # load in the subfind group files
# snap_subfind = load_subfind(snapnum,dir=simfile)
#
# # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
# snap = gadget_readsnap(snapnum, simfile,loadonlytype = [0],lazy_load=True)
#
# Snapper3 = Snapper()
# snap = Snapper3.SetCentre(Snap=snap,Snap_SubFind=snap_subfind,HaloID=0)
#
# #Convert Units
# ## Make this a seperate function at some point??
# snap.pos *= 1e3 #[kpc]
# snap.vol *= 1e9 #[kpc^3]
#
# Snapper3.PlotProjections(Snap=snap,SnapNum=snapnum)

#------------------------------------------------------------------------------#

# Histogram 2D Movie:
Snapper4 = Snapper()
Snapper4.ProjectionMovieLoop(SimDirectory=simfile,NSnaps=127,Start=10)
