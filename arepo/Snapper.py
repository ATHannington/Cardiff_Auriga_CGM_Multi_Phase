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
        print('Centering!')
        HaloCentre = Snap_SubFind.data['fpos'][HaloID,:] # subfind has calculated its centre of mass for you
        Snap.data['pos'] = (Snap.data['pos'] - np.array(HaloCentre)) # use the subfind COM to centre the coordinates on the galaxy

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

    def PlotProjections(self,Snap,SnapNum,SnapDM=None,SnapStars=None,Axes=[0,1],boxsize = 200.,boxlos = 50.,pixres = 1.,pixreslos = 0.1, Nbins=500, DPI = 500,CMAP='inferno'):

        print("Projection!")

        #Axes Labels to allow for adaptive axis selection
        AxesLabels = ['x','y','z']

        #Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in SetCentre)
        imgcent =[0.,0.,0.]

        #--------------------------#
        ##    Units Conversion    ##
        #--------------------------#

        #Density is rho/ <rho> where <rho> is average baryonic density
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

        bfactor = 1e6*(np.sqrt(1e10 * c.msol) / np.sqrt(c.parsec * 1e6)) * (1e5 / (c.parsec * 1e6)) #[microGauss]
        Snap.data['B'] = np.linalg.norm((Snap.data['bfld'] * bfactor), axis=1)

        #--------------------------#
        ## Slices and Projections ##
        #--------------------------#

        # slice_nH    = Snap.get_Aslice("n_H", box = [boxsize,boxsize],\
        #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
        #  axes = Axes, proj = False, numthreads=16)

        # slice_T     = Snap.get_Aslice("T", box = [boxsize,boxsize],\
        #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
        #  axes = Axes, proj = False, numthreads=16)

        # proj_T      = Snap.get_Aslice("Tdens", box = [boxsize,boxsize],\
        #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
        #  nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=16)
        #
        # proj_dens   = Snap.get_Aslice("dens", box = [boxsize,boxsize],\
        #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
        #  nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=16)

        proj_nH     = Snap.get_Aslice("n_H", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=16)

        proj_B     = Snap.get_Aslice("B", box = [boxsize,boxsize],\
         center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
         nz = int(boxlos/pixreslos), boxz = boxlos, axes = Axes, proj = True, numthreads=16)

        #   DM mass stored separately as singular value and needs to be
        #   converted into array for weighting of hist2D
        massDM = np.array([SnapDM.masses[1] for i in range(0,len(SnapDM.pos))])
        #snapStars.mass *= 1e10

        #PLOTTING TIME

        #Set plot figure sizes
        xsize = 10.
        ysize = 10.

        #Define halfsize for histogram ranges which are +/-
        halfbox = boxsize/2.

        #Halve Nbins for stars as they're hard to spot otherwise!
        NbinsStars = float(Nbins)/2.0

        #Redshift
        redshift = Snap.redshift        #z
        aConst = 1. / (1. + redshift)   #[/]

        #[0] to remove from numpy array for purposes of plot title
        lookback = Snap.cosmology_get_lookback_time_from_a(np.array([aConst]))[0] #[Gyrs]

        #DPI Controlled by user as lower res needed for videos #
        fig, axes = plt.subplots(2, 2, figsize = (xsize,ysize), dpi = DPI)

        #Add overall figure plot
        TITLE = r"$z=$" + f"{redshift:0.05e} " + " " + r"$t_{Lookback}=$" + f"{lookback:0.05e} Gyrs"
        fig.suptitle(TITLE, fontsize=16)
        cmap = plt.get_cmap(CMAP)
        cmap.set_bad(color="grey")

        #-----------#
        # Plot DM Histogram 2D #
        #-----------#

        pcm1 = axes[0,0].hist2d(SnapDM.pos[:,Axes[0]],SnapDM.pos[:,Axes[1]],bins=Nbins,range=[[-1.*halfbox,halfbox],[-1.*halfbox,halfbox]],
        norm = matplotlib.colors.LogNorm(), weights=massDM , cmap = cmap)

        axes[0,0].set_title(f'Dark Matter Column Density - #Bins={Nbins:0.01f}')
        fig.colorbar(pcm1[3], ax = axes[0,0], orientation = 'horizontal',label=r'$n_{DM}$ [$10^{10} M_{\odot}$ $pixel^{-2}$]')

        #-----------#
        # Plot n_H Projection #
        #-----------#

        pcm2 = axes[0,1].pcolormesh(proj_nH['x'], proj_nH['y'], np.transpose(proj_nH['grid'])/int(boxlos/pixreslos),\
         norm = matplotlib.colors.LogNorm(vmin=1e-6,vmax=1e-1), cmap = cmap, rasterized = True)
        axes[0,1].set_title(r'Projection $n_H$')
        fig.colorbar(pcm2, ax = axes[0,1], orientation = 'horizontal',label=r'$n_H$ [$cm^{-3}$]')

        #-----------#
        # Plot Stars Histogram 2D #
        #-----------#

        pcm3 = axes[1,0].hist2d(SnapStars.pos[:,Axes[0]],SnapStars.pos[:,Axes[1]],bins=NbinsStars,\
        range=[[-1.*halfbox,halfbox],[-1.*halfbox,halfbox]], weights=SnapStars.mass, norm = matplotlib.colors.LogNorm(), cmap = cmap)

        axes[1,0].set_title(f'Stars Column Density - #Bins={NbinsStars:0.01f}')
        fig.colorbar(pcm3[3], ax = axes[1,0], orientation = 'horizontal',label=r'$n_*$ [$10^{10} M_{\odot}$ $pixel^{-2}$]')

        #-----------#
        # Plot Magnetic Field Projection #
        #-----------#

        pcm4 = axes[1,1].pcolormesh(proj_B['x'], proj_B['y'], np.transpose(proj_B['grid']),\
        norm = matplotlib.colors.SymLogNorm(linthresh=0.1),\
         cmap = cmap, rasterized = True)

        axes[1,1].set_title(r'Projection $B$')
        fig.colorbar(pcm4, ax = axes[1,1], orientation = 'horizontal',label=r'$B$ [$\mu G$]')

        #-----------#

        #Add Axes labels to each plot, adaptive to which axes are selected
        for ii in range(0,2):
            for jj in range(0,2):
                axes[ii,jj].set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)')#, fontsize = 20.0)
                axes[ii,jj].set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)')#, fontsize = 20.0)

        axes[0,0].set_aspect(1.0)
        axes[0,1].set_aspect(1.0)
        axes[1,0].set_aspect(1.0)
        axes[1,1].set_aspect(1.0)

        #Pad snapnum with zeroes to enable easier video making
        SnapNum = str(SnapNum).zfill(3);
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
            snapGas   = gadget_readsnap(ii, simfile,loadonlytype = [0],lazy_load=True)
            snapDM    = gadget_readsnap(ii, simfile,loadonlytype = [1],lazy_load=True)
            snapStars = gadget_readsnap(ii, simfile,loadonlytype = [4],lazy_load=True)

            print(f" SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")

            snapGas   = snapper.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=0)
            snapDM    = snapper.SetCentre(Snap=snapDM,Snap_SubFind=snap_subfind,HaloID=0)
            snapStars = snapper.SetCentre(Snap=snapStars,Snap_SubFind=snap_subfind,HaloID=0)

            #Convert Units
            ## Make this a seperate function at some point??
            snapGas.pos   *= 1e3 #[kpc]
            snapDM.pos    *= 1e3 #[kpc]
            snapStars.pos *= 1e3 #[kpc]

            snapGas.vol *= 1e9 #[kpc^3]

            snapper.PlotProjections(Snap=snapGas,SnapNum=ii,SnapDM=snapDM,SnapStars=snapStars,DPI=200)

            #Print percentage complete
            print(f"{(float(ii + 1 - Start)/float(NSnaps - Start))*100.0 : 0.03f}","% complete")

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

# Single FvdV Projection:
# load in the subfind group files
# snap_subfind = load_subfind(snapnum,dir=simfile)
#
# # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
# snapGas   = gadget_readsnap(snapnum, simfile,loadonlytype = [0],lazy_load=True)
# snapDM    = gadget_readsnap(snapnum, simfile,loadonlytype = [1],lazy_load=True)
# snapStars = gadget_readsnap(snapnum, simfile,loadonlytype = [4],lazy_load=True)
#
# print(f" SnapShot loaded at RedShift z={snapGas.redshift:0.05e}")
#
# Snapper3 = Snapper()
# snapGas   = Snapper3.SetCentre(Snap=snapGas,Snap_SubFind=snap_subfind,HaloID=0)
# snapDM    = Snapper3.SetCentre(Snap=snapDM,Snap_SubFind=snap_subfind,HaloID=0)
# snapStars = Snapper3.SetCentre(Snap=snapStars,Snap_SubFind=snap_subfind,HaloID=0)
#
# #Convert Units
# ## Make this a seperate function at some point??
# snapGas.pos   *= 1e3 #[kpc]
# snapDM.pos    *= 1e3 #[kpc]
# snapStars.pos *= 1e3 #[kpc]
#
# snapGas.vol *= 1e9 #[kpc^3]
#
# Snapper3.PlotProjections(Snap=snapGas,SnapNum=snapnum,SnapDM=snapDM,SnapStars=snapStars)

#------------------------------------------------------------------------------#

# Histogram 2D Movie:
Snapper4 = Snapper()
Snapper4.ProjectionMovieLoop(SimDirectory=simfile,NSnaps=127,Start=10)
