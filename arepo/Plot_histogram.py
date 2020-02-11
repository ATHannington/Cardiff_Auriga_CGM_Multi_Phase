import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gadget import *
from gadget_subfind import *
from const import *


simfile='/home/universe/spxtd1-shared/ISOTOPES/output/' # set paths to simulation
snapnum=127 # set snapshot to look at

class Snapper(object):
    """
    Class Object for interacting with Arepo Snaps using arepo-snap-utils.

    Default imports required:

        import numpy as np
        import matplotlib
        matplotlib.use('Agg') #For suppressing plotting on clusters
        import matplotlib.pyplot as plt
        from gadget import *
        from gadget_subfind import *
        from const import

    """
#------------------------------------------------------------------------------#
#          __init__
#------------------------------------------------------------------------------#
    def __init__(self):
        """
        Nothing is required to setup a Snapper object!
        """
        pass

#------------------------------------------------------------------------------#
#           Load single SnapShot
#------------------------------------------------------------------------------#

    def LoadSnap(self,SimDirectory,SnapNum,hdf5Bool=True,LoadOnly=['mass','pos'] \
    ,LoadType=[0],LoadHalo=0,LazyLoad=True):
        """
        Function for loading an individual SnapShot.
        Args:
            SimDirectory    : REQUIRED! Directory path of simulation data
            SnapNum         : REQUIRED! Individual snap number
            hdf5Bool=None   : Opt. Default = True              Use HDF5 data
            LoadOnly=None   : Opt. Default = ['mass','pos']    Load only mass and position
            LoadType=None   : Opt. Default = [0]               Load only particle type 0 - gas
                                                               0 is gas, 1 is DM, 4 is stars, 5 is BHs
            LoadHalo=None   : Opt. Default = 0                 Load largest Halo
            LazyLoad=None   : Opt. Default = True              Load only required data into RAM
        """

        self.SnapNum = SnapNum
        self.SimDirectory = SimDirectory
        self.hdf5Bool = hdf5Bool
        self.LoadOnly = LoadOnly
        self.LoadType = LoadType
        self.LoadHalo = LoadHalo
        self.LazyLoad = LazyLoad

        # load in the subfind group files
        Snap_SubFind = load_subfind(self.SnapNum, dir=self.SimDirectory)

        # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
        Snap = gadget_readsnap(self.SnapNum, self.SimDirectory, hdf5=self.hdf5Bool \
        , loadonly=self.LoadOnly, loadonlytype = self.LoadType \
        , loadonlyhalo = self.LoadHalo, lazy_load=self.LazyLoad \
        , subfind = Snap_SubFind)

        haloid = 0 # look only at the central galaxy - index zero
        ImgCent = Snap_SubFind.data['fpos'][self.LoadHalo,:] # subfind has calculated its centre of mass for you
        Snap.data['pos'] = (Snap.data['pos'] - np.array(ImgCent))*1e3 # use the subfind COM to centre the coordinates on the galaxy [km]

        return Snap

#------------------------------------------------------------------------------#
#           Plot 1x Histogram 2d
#------------------------------------------------------------------------------#

    def PlotHistogram(self, Snap, Nbins=1000,Axes=[0,1],Range = [[-50,50],[-50,50]],WeightsLabel='mass',Normed=False):
        """
        Function for Histogram 2D plotting an individual SnapShot.
        Args:
            Nbins               : Opt. Number of Histogram bins : Default = 1000
            Axes                : Opt. Axes Selection           : Default = [0,1] == ['X','Y']
            Range               : Opt. Axis Range in kpc        : Default = [[-50,50],[-50,50]]
            WeightsLabel        : Opt. Weight bins by param.    : Default = 'mass'
            Normed              : Opt. Normalise bins?          : Default = False. True is NOT recommended!
        """

        self.Nbins = Nbins
        self.Axes  = Axes
        self.AxesLabels = ['X','Y','Z']
        self.Range = Range
        self.WeightsLabel = WeightsLabel
        if (WeightsLabel == 'mass'):
            self.Weights = Snap.mass
        else:
            print("Unknown Weights Flag! Setting to Default 'mass'!")
            self.Weights = Snap.mass

        self.Normed = Normed

        hist,xedge,yedge=np.histogram2d(Snap.pos[:,self.Axes[0]],Snap.pos[:,self.Axes[1]] \
        , bins=self.Nbins,range=self.Range,weights=self.Weights,normed=self.Normed)

        img1 = plt.imshow(hist,cmap='nipy_spectral',vmin=np.nanmin(hist) \
        ,vmax=np.nanmax(hist)\
        ,extent=[np.min(xedge),np.max(xedge),np.min(yedge),np.max(yedge)],origin='lower')
        ax1 = plt.gca()
        ax1.tick_params(labelsize=20.)
        ax1.set_ylabel(f'{self.AxesLabels[self.Axes[0]]} (kpc)', fontsize = 20.0) # Y label
        ax1.set_xlabel(f'{self.AxesLabels[self.Axes[1]]} (kpc)', fontsize = 20.0) # X label

        cbar = plt.colorbar()
        cbar.set_label(f'{self.WeightsLabel}')

        self.SnapNum = str(self.SnapNum).zfill(3);
        plt.savefig(f'Histogram2d_{self.WeightsLabel}_{self.SnapNum}.png', bbox_inches='tight')
        plt.close()

        return img1,ax1

#------------------------------------------------------------------------------#
#           Plot (NSnaps - Start)x Histogram 2d
#
#               CURRENTLY ONLY DEFAULTS SUPPORTED
#------------------------------------------------------------------------------#

    def HistogramMovieLoop(self,SimDirectory,NSnaps=127,Start=10):
        """
        Function for Histogram 2D plotting of a loop (Start to NSnaps) SnapShots.
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
            snap = snapper.LoadSnap(SimDirectory=self.SimDirectory,SnapNum=ii)
            snapper.PlotHistogram(Snap=snap)
            #Print percentage complete
            print((float(ii + 1 - Start)/float(NSnaps - Start))*100.0,"% complete")
            #Delete child snapper, to free up space and prevent data leakage



#===============================================================================#

#===============================================================================#

#------------------------------------------------------------------------------#
#           Run Code!
#------------------------------------------------------------------------------#

#Histogram:
Snapper1 = Snapper()
snap = Snapper1.LoadSnap(SimDirectory=simfile,SnapNum=snapnum)
Snapper1.PlotHistogram(Snap=snap)

#Movie:
# Snapper2 = Snapper()
# Snapper2.HistogramMovieLoop(simfile,NSnaps=127,Start=10)
