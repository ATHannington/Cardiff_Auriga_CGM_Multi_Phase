import numpy as np
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import const as c
from gadget import *
from gadget_subfind import *
from Tracers_Subroutines import CalculateTrackedParameters
import multiprocessing as mp


simfile = "/home/universe/spxfv/Auriga/level4_MHD_variants/halo_18_3000snaps/output/"  #'/home/universe/spxtd1-shared/ISOTOPES/output/' # set paths to simulation
snapnum = 2500  # 127 # set snapshot to look at
snap_start = 2500
snap_end = 2999
n_processes = 6

Axes = [0, 2]  # [i,j] where i & j range from 0 - 2, for x,y,z axes

boxsize = 400.0  # split between either side of x&y=0.[kpc]
boxlos = 20  # split either side of z = 0 will be boxlos/2 in size [kpc]
pixres = 0.2
pixreslos = 4

Nbins = 0
DPI = 200
CMAP = "inferno"
ColourBins = 256

# ==============================================================================#
posAxes = [0, 1, 2]

zAxis = [x for x in posAxes if x not in Axes]

# ------------------------------------------------------------------------------#
#          Initialisation
# ------------------------------------------------------------------------------#

"""Initialisation Defines some necessary Global parameters:
    elements        : Element key strings list
    elements_Z      : Element Proton Number
    elements_mass   : Element mass unit
    elements_solar  : Element Solar abundances
    Zsolar          : Solar metallicity
    omegabaryon0    : Omega Baryon 0 - Baryonic Fraction??
"""


# element number   0     1      2      3      4      5      6      7      8      9      10     11     12      13
elements = [
    "H",
    "He",
    "C",
    "N",
    "O",
    "Ne",
    "Mg",
    "Si",
    "Fe",
    "Y",
    "Sr",
    "Zr",
    "Ba",
    "Pb",
]
elements_Z = [1, 2, 6, 7, 8, 10, 12, 14, 26, 39, 38, 40, 56, 82]
elements_mass = [
    1.01,
    4.00,
    12.01,
    14.01,
    16.00,
    20.18,
    24.30,
    28.08,
    55.85,
    88.91,
    87.62,
    91.22,
    137.33,
    207.2,
]
elements_solar = [
    12.0,
    10.93,
    8.43,
    7.83,
    8.69,
    7.93,
    7.60,
    7.51,
    7.50,
    2.21,
    2.87,
    2.58,
    2.18,
    1.75,
]

Zsolar = 0.0127

omegabaryon0 = 0.048


def ColourMap(ColourBins=256):
    """
    Creates a diverging colourmap with dark colours in middle
    Input: int Number of colour Bins
    Output: Colormap
    """
    from matplotlib.colors import ListedColormap

    print("Defining ATH Custom Colourmap!")

    color1 = plt.get_cmap("Purples", ColourBins)
    color2 = plt.get_cmap("hot", ColourBins)

    c1_list = color1(np.linspace(0, 1, ColourBins))
    c2_list = color2(np.linspace(0, 1, ColourBins))

    full_list = np.concatenate((c1_list, c2_list))

    ATH_cmap = ListedColormap(full_list)

    return ATH_cmap


if (CMAP == "ATH") or (CMAP == "ath"):
    CMAP = ATH_ColourMap(ColourBins)
else:
    CMAP = plt.get_cmap(CMAP, ColourBins)
# ------------------------------------------------------------------------------#
#          Set Centre to [0,0,0] centering on COM of Halo number HaloID
#                   Where HaloID=0 is the biggest halo.
# ------------------------------------------------------------------------------#


def SetCentre(snap, snap_subfind, HaloID):
    print("Centering!")
    HaloCentre = snap_subfind.data["fpos"][
        HaloID, :
    ]  # subfind has calculated its centre of mass for you
    snap.data["pos"] = snap.data["pos"] - np.array(
        HaloCentre
    )  # use the subfind COM to centre the coordinates on the galaxy

    return snap


# ------------------------------------------------------------------------------#
#           Plot 1x Histogram 2d
# ------------------------------------------------------------------------------#


def PlotHistogram(
    snap,
    snapnum,
    Nbins=500,
    Axes=[0, 1],
    Range=[[-50, 50], [-50, 50]],
    WeightsLabel="mass",
    Normed=False,
):
    """
    Function for Histogram 2D plotting an individual SnapShot.
    Args:
        Nbins               : Opt. Number of Histogram bins : Default = 500
        Axes                : Opt. Axes Selection           : Default = [0,1] == ['X','Y']
        Range               : Opt. Axis Range in kpc        : Default = [[-50,50],[-50,50]]
        WeightsLabel        : Opt. Weight bins by param.    : Default = 'mass'
        Normed              : Opt. Normalise bins?          : Default = False. True is NOT recommended!
    """
    AxesLabels = ["X", "Y", "Z"]
    if WeightsLabel == "mass":
        Weights = snap.mass
    else:
        print("Unknown Weights Flag! Setting to Default 'mass'!")
        Weights = snap.mass

    hist, xedge, yedge = np.histogram2d(
        snap.pos[:, Axes[0]],
        snap.pos[:, Axes[1]],
        bins=Nbins,
        range=Range,
        weights=Weights,
        normed=Normed,
    )

    img1 = plt.imshow(
        hist,
        cmap="nipy_spectral",
        vmin=np.nanmin(hist),
        vmax=np.nanmax(hist),
        extent=[np.min(xedge), np.max(xedge), np.min(yedge), np.max(yedge)],
        origin="lower",
    )
    ax1 = plt.gca()
    ax1.tick_params(labelsize=20.0)
    ax1.set_ylabel(f"{AxesLabels[Axes[0]]} (kpc)", fontsize=20.0)  # Y label
    ax1.set_xlabel(f"{AxesLabels[Axes[1]]} (kpc)", fontsize=20.0)  # X label

    cbar = plt.colorbar()
    cbar.set_label(f"{WeightsLabel}")

    snapnum = str(snapnum).zfill(3)
    plt.savefig(f"Histogram2d_{WeightsLabel}_{snapnum}.png", bbox_inches="tight")
    plt.close()

    return img1, ax1


# ------------------------------------------------------------------------------#
#           Plot (NSnaps - Start)x Histogram 2d
#
#               CURRENTLY ONLY DEFAULTS SUPPORTED
# ------------------------------------------------------------------------------#


def HistogramMovieLoop(SimDirectory, NSnaps=127, Start=10):
    """
    Function for Histogram 2D plotting of a loop (Start to NSnaps) Snaps.
    Args:
        SimDirectory        : REQUIRED! Directory path for simulation
        NSnaps              : Opt. Number of SnapShots      : Default = 127
        Start               : Opt. Start Snapshot           : Default = 10 . Below this Halo finder may fail
    """
    # import matplotlib.animation as manimation

    print("Let's make a Movie!")
    for ii in range(Start, NSnaps + 1):
        # Create child Snapper for each histogram. Caution, don't add to self
        # or data will stack up and bug out.

        # Load Snap and plot histogram with default options
        #
        ###
        # Andy: generalise this later, please!
        ###
        #
        #

        # load in the subfind group files
        snap_subfind = load_subfind(ii, dir=simfile)

        # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
        snap = gadget_readsnap(
            ii,
            simfile,
            hdf5=True,
            loadonly=["mass", "pos"],
            loadonlytype=[0],
            loadonlyhalo=0,
            lazy_load=True,
            subfind=snap_subfind,
        )

        snap = SetCentre(snap=snap, snap_subfind=snap_subfind, HaloID=0)

        # Convert Units
        ## Make this a seperate function at some point??
        snap.pos *= 1e3  # [kpc]
        snap.vol *= 1e9  # [kpc^3]

        PlotHistogram(snap=snap, snapnum=ii)

        # Print percentage complete
        print(
            f"{(float(ii + 1 - Start)/float(NSnaps - Start))*100.0 : 0.03f}",
            "% complete",
        )
        # Delete child snapper, to free up space and prevent data leakage

    return


# ------------------------------------------------------------------------------#
#         Plot Projections and Slices
# ------------------------------------------------------------------------------#


def PlotProjections(
    snapGas,
    snapnum,
    snapDM=None,
    snapStars=None,
    Axes=[0, 1],
    zAxis=[2],
    boxsize=400.0,
    boxlos=50.0,
    pixres=1.0,
    pixreslos=0.1,
    Nbins=500,
    DPI=500,
    CMAP=None,
):

    if CMAP == None:
        cmap = plt.get_cmap("inferno")
    else:
        cmap = CMAP

    print("Quad Plot...")
    print("Calculating Tracked Parameters!")

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["x", "y", "z"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in SetCentre)
    imgcent = [0.0, 0.0, 0.0]

    # --------------------------#
    ##    Units Conversion    ##
    # --------------------------#
    whereGas = np.where(snapGas.type == 0)
    # Density is rho/ <rho> where <rho> is average baryonic density
    rhocrit = (
        3.0
        * (snapGas.omega0 * (1 + snapGas.redshift) ** 3.0 + snapGas.omegalambda)
        * (snapGas.hubbleparam * 100.0 * 1e5 / (c.parsec * 1e6)) ** 2.0
        / (8.0 * pi * c.G)
    )
    rhomean = (
        3.0
        * (snapGas.omega0 * (1 + snapGas.redshift) ** 3.0)
        * (snapGas.hubbleparam * 100.0 * 1e5 / (c.parsec * 1e6)) ** 2.0
        / (8.0 * pi * c.G)
    )

    # Mean weight [amu]
    meanweight = sum(snapGas.gmet[whereGas, 0:9][0], axis=1) / (
        sum(snapGas.gmet[whereGas, 0:9][0] / elements_mass[0:9], axis=1)
        + snapGas.ne * snapGas.gmet[whereGas, 0][0]
    )

    # 3./2. R == 2./3. NA KB
    Tfac = ((3.0 / 2.0) * c.KB * 1e10 * c.msol) / (meanweight * c.amu * 1.989e53)

    gasdens = (snapGas.rho / (c.parsec * 1e6) ** 3.0) * c.msol * 1e10  # [g cm^-3]
    gasX = snapGas.gmet[whereGas, 0][0]

    # Temperature = U / (3/2 * NA KB) [K]
    snapGas.data["T"] = snapGas.u / Tfac  # K
    snapGas.data["n_H"] = gasdens / c.amu * gasX  # cm^-3
    snapGas.data["dens"] = gasdens / (
        rhomean * omegabaryon0 / snapGas.omega0
    )  # rho / <rho>
    snapGas.data["Tdens"] = snapGas.data["T"] * snapGas.data["dens"]

    bfactor = (
        1e6
        * (np.sqrt(1e10 * c.msol) / np.sqrt(c.parsec * 1e6))
        * (1e5 / (c.parsec * 1e6))
    )  # [microGauss]

    # Magnitude of Magnetic Field [micro Guass]
    snapGas.data["B"] = np.linalg.norm((snapGas.data["bfld"] * bfactor), axis=1)

    # Load in metallicity
    snapGas.data["gz"] = snapGas.data["gz"] / Zsolar
    # --------------------------#
    ## Slices and Projections ##
    # --------------------------#
    print("Slices and Projections!")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # slice_nH    = snap.get_Aslice("n_H", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    #
    # slice_B   = snap.get_Aslice("B", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

    proj_nH = snapGas.get_Aslice(
        "n_H",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=16,
    )

    proj_B = snapGas.get_Aslice(
        "B",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=16,
    )

    proj_dens = snapGas.get_Aslice(
        "dens",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=16,
    )

    proj_T = snapGas.get_Aslice(
        "Tdens",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=16,
    )

    proj_gz = snapGas.get_Aslice(
        "gz",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=16,
    )

    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # DMwithinLoSVol = snapDM.pos[np.where((snapDM.pos[:,zAxis[0]]>=(-1.*float(boxlos)/2.))&(snapDM.pos[:,zAxis[0]]<=(float(boxlos)/2.)))]
    #
    # StarswithinLoSVol = snapStars.pos[np.where((snapStars.pos[:,zAxis[0]]>=(-1.*float(boxlos)/2.))&(snapStars.pos[:,zAxis[0]]<=(float(boxlos)/2.)))]
    # massStarswithinLoSVol = snapStars.mass[np.where((snapStars.pos[:,zAxis[0]]>=(-1.*float(boxlos)/2.))&(snapStars.pos[:,zAxis[0]]<=(float(boxlos)/2.)))]
    # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # #   DM mass stored separately as singular value and needs to be
    # #   converted into array for weighting of hist2D
    # if (snapDM is not None):
    #     massDM = np.array([snapDM.masses[1] for i in range(0,len(DMwithinLoSVol))])

    # PLOTTING TIME
    # Set plot figure sizes
    xsize = 10.0
    ysize = 10.0

    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    # Halve Nbins for stars as they're hard to spot otherwise!
    NbinsStars = float(Nbins) / 2.0

    # Redshift
    redshift = snapGas.redshift  # z
    aConst = 1.0 / (1.0 + redshift)  # [/]

    # [0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
        0
    ]  # [Gyrs]

    aspect = "equal"
    fontsize = 12
    fontsizeTitle = 20
    print("Plot!")

    # DPI Controlled by user as lower res needed for videos #
    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(xsize, ysize), dpi=DPI, sharex="col", sharey="row"
    )

    # Add overall figure plot
    TITLE = (
        r"Redshift $(z) =$"
        + f"{redshift:0.03f} "
        + " "
        + r"$t_{Lookback}=$"
        + f"{lookback:0.03f} Gyrs"
        + "\n"
        + f"Projections within {-1.*float(boxlos)/2.}"
        + r"<"
        + f"{AxesLabels[zAxis[0]]}-axis"
        + r"<"
        + f"{float(boxlos)/2.} kpc"
    )
    fig.suptitle(TITLE, fontsize=fontsizeTitle)

    # cmap = plt.get_cmap(CMAP)
    cmap.set_bad(color="grey")

    # -----------#
    # Plot Temperature #
    # -----------#
    # print("pcm1")
    ax1 = axes[0, 0]

    pcm1 = ax1.pcolormesh(
        proj_T["x"],
        proj_T["y"],
        np.transpose(proj_T["grid"] / proj_dens["grid"]),
        vmin=1e4,
        vmax=1e7,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax1.set_title(f"Temperature Projection", fontsize=fontsize)
    cax1 = inset_axes(ax1, width="5%", height="95%", loc="right")
    fig.colorbar(pcm1, cax=cax1, orientation="vertical").set_label(
        label=r"$T$ [$K$]", size=fontsize, weight="bold"
    )
    cax1.yaxis.set_ticks_position("left")
    cax1.yaxis.set_label_position("left")
    cax1.yaxis.label.set_color("white")
    cax1.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax1.set_ylabel(f"{AxesLabels[Axes[1]]} (kpc)", fontsize=fontsize)
    # ax1.set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize = fontsize)
    ax1.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax1)
    plt.yticks([-150, -100, -50, 0, 50, 100, 150, 200])
    # -----------#
    # Plot n_H Projection #
    # -----------#
    # print("pcm2")
    ax2 = axes[0, 1]

    pcm2 = ax2.pcolormesh(
        proj_nH["x"],
        proj_nH["y"],
        np.transpose(proj_nH["grid"]) / int(boxlos / pixreslos),
        vmin=1e-6,
        vmax=1e-1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax2.set_title(r"Hydrogen Number Density Projection", fontsize=fontsize)

    cax2 = inset_axes(ax2, width="5%", height="95%", loc="right")
    fig.colorbar(pcm2, cax=cax2, orientation="vertical").set_label(
        label=r"$n_H$ [$cm^{-3}$]", size=fontsize, weight="bold"
    )
    cax2.yaxis.set_ticks_position("left")
    cax2.yaxis.set_label_position("left")
    cax2.yaxis.label.set_color("white")
    cax2.tick_params(axis="y", colors="white", labelsize=fontsize)
    # ax2.set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize=fontsize)
    # ax2.set_xlabel(f'{AxesLabels[Axes[0]]} (kpc)', fontsize=fontsize)
    ax2.set_aspect(aspect)

    # -----------#
    # Plot Metallicity #
    # -----------#
    # print("pcm3")
    ax3 = axes[1, 0]

    pcm3 = ax3.pcolormesh(
        proj_gz["x"],
        proj_gz["y"],
        np.transpose(proj_gz["grid"]) / int(boxlos / pixreslos),
        vmin=1e-2,
        vmax=1e1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax3.set_title(f"Metallicity Projection", y=-0.2, fontsize=fontsize)

    cax3 = inset_axes(ax3, width="5%", height="95%", loc="right")
    fig.colorbar(pcm3, cax=cax3, orientation="vertical").set_label(
        label=r"$Z/Z_{\odot}$", size=fontsize, weight="bold"
    )
    cax3.yaxis.set_ticks_position("left")
    cax3.yaxis.set_label_position("left")
    cax3.yaxis.label.set_color("white")
    cax3.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax3.set_ylabel(f"{AxesLabels[Axes[1]]} (kpc)", fontsize=fontsize)
    ax3.set_xlabel(f"{AxesLabels[Axes[0]]} (kpc)", fontsize=fontsize)

    ax3.set_aspect(aspect)

    # -----------#
    # Plot Magnetic Field Projection #
    # -----------#
    # print("pcm4")
    ax4 = axes[1, 1]

    pcm4 = ax4.pcolormesh(
        proj_B["x"],
        proj_B["y"],
        np.transpose(proj_B["grid"]) / int(boxlos / pixreslos),
        vmin=1e-3,
        vmax=1e1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax4.set_title(r"Magnetic Field Strength Projection", y=-0.2, fontsize=fontsize)

    cax4 = inset_axes(ax4, width="5%", height="95%", loc="right")
    fig.colorbar(pcm4, cax=cax4, orientation="vertical").set_label(
        label=r"$B$ [$\mu G$]", size=fontsize, weight="bold"
    )
    cax4.yaxis.set_ticks_position("left")
    cax4.yaxis.set_label_position("left")
    cax4.yaxis.label.set_color("white")
    cax4.tick_params(axis="y", colors="white", labelsize=fontsize)

    # ax4.set_ylabel(f'{AxesLabels[Axes[1]]} (kpc)', fontsize=fontsize)
    ax4.set_xlabel(f"{AxesLabels[Axes[0]]} (kpc)", fontsize=fontsize)
    ax4.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax4)
    plt.xticks([-150, -100, -50, 0, 50, 100, 150, 200])

    # print("snapnum")
    # Pad snapnum with zeroes to enable easier video making
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    # fig.tight_layout()

    snapnum = str(snapnum).zfill(3)
    opslaan = f"Shaded_Cell_{snapnum}.png"
    print(f"Save {opslaan}")
    plt.savefig(opslaan, dpi=DPI, transparent=False)
    plt.close()

    print("...done!")


# ------------------------------------------------------------------------------#
#           Plot (Nsnaps - Start)x Projections
#
#               CURRENTLY ONLY DEFAULTS SUPPORTED
# ------------------------------------------------------------------------------#
def _projection_movie(
    snapnum,
    SimDirectory,
    Axes,
    zAxis,
    boxsize,
    boxlos,
    pixres,
    pixreslos,
    Nbins,
    DPI,
    CMAP,
):

    print(f"Starting {snapnum}")
    # load in the subfind group files
    snap_subfind = load_subfind(snapnum, dir=SimDirectory)

    # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
    snapGas = gadget_readsnap(snapnum, SimDirectory, loadonlytype=[0], lazy_load=True)

    print(f" snapShot loaded at RedShift z={snapGas.redshift:0.05e}")

    snapGas = SetCentre(snap=snapGas, snap_subfind=snap_subfind, HaloID=0)

    # Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos *= 1e3  # [kpc]

    snapGas.vol *= 1e9  # [kpc^3]

    PlotProjections(
        snapGas=snapGas,
        snapnum=snapnum,
        snapDM=None,
        snapStars=None,
        Axes=Axes,
        zAxis=zAxis,
        boxsize=boxsize,
        boxlos=boxlos,
        pixres=pixres,
        pixreslos=pixreslos,
        Nbins=Nbins,
        DPI=DPI,
        CMAP=CMAP,
    )
    return


def ProjectionMovieLoop(
    SimDirectory,
    Start,
    End,
    n_processes,
    Axes,
    zAxis,
    boxsize,
    boxlos,
    pixres,
    pixreslos,
    Nbins,
    DPI,
    CMAP,
):
    """
    Function for Histogram 2D plotting of a loop (Start to Nsnaps) snaps.
    Args:
        SimDirectory        : REQUIRED! Directory path for simulation
        Nsnaps              : Opt. Number of snapShots      : Default = 127
        Start               : Opt. Start snapshot           : Default = 10 . Below this Halo finder may fail
    """

    args_default = [
        SimDirectory,
        Axes,
        zAxis,
        boxsize,
        boxlos,
        pixres,
        pixreslos,
        Nbins,
        DPI,
        CMAP,
    ]
    args_list = [[snap] + args_default for snap in range(Start, End + 1)]
    print("\n" + f"Opening {n_processes} core Pool!")
    print("Let's make a Movie!")
    pool = mp.Pool(processes=n_processes)

    # Compute movie projections!
    [pool.apply_async(_projection_movie, args=args) for args in args_list]

    pool.close()
    pool.join()

    return


# ===============================================================================#

# ===============================================================================#

# ------------------------------------------------------------------------------#
#           Run Code!
# ------------------------------------------------------------------------------#

# Histogram:
#
# load in the subfind group files
# snap_subfind = load_subfind(snapnum,dir=simfile)
#
# # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
# snap = gadget_readsnap(snapnum, simfile, hdf5=True, loadonly=['mass','pos'], loadonlytype = [0], loadonlyhalo = 0, lazy_load=True, subfind = snap_subfind)
#
# snap = SetCentre(snap=snap,snap_SubFind=snap_subfind,HaloID=0)
#
# #Convert Units
# ## Make this a seperate function at some point??
# snap.pos *= 1e3 #[kpc]
# snap.vol *= 1e9 #[kpc^3]
#
# PlotHistogram(snap=snap,snapnum=snapnum)

# ------------------------------------------------------------------------------#

# Histogram 2D Movie:
# HistogramMovieLoop(simfile,Nsnaps=127,Start=10)

# #------------------------------------------------------------------------------#
# #
# # # Single FvdV Projection:
# # # load in the subfind group files
# snap_subfind = load_subfind(snapnum,dir=simfile)
#
# # load in the gas particles mass and position. 0 is gas, 1 is DM, 4 is stars, 5 is BHs
# snapGas   = gadget_readsnap(snapnum, simfile,loadonlytype = [0],lazy_load=True)
# # snapDM    = gadget_readsnap(snapnum, simfile,loadonlytype = [1],lazy_load=True)
# # snapStars = gadget_readsnap(snapnum, simfile,loadonlytype = [4],lazy_load=True)
#
# print(f" snapShot loaded at RedShift z={snapGas.redshift:0.05e}")
#
# snapGas   = SetCentre(snap=snapGas,snap_subfind=snap_subfind,HaloID=0)
# # snapDM    = SetCentre(snap=snapDM,snap_subfind=snap_subfind,HaloID=0)
# # snapStars = SetCentre(snap=snapStars,snap_subfind=snap_subfind,HaloID=0)
#
# #Convert Units
# ## Make this a seperate function at some point??
# snapGas.pos   *= 1e3 #[kpc]
# # snapDM.pos    *= 1e3 #[kpc]
# # snapStars.pos *= 1e3 #[kpc]
#
# snapGas.vol *= 1e9 #[kpc^3]
#
# PlotProjections(snapGas=snapGas,snapnum=snapnum,snapDM=None,snapStars=None,\
# Axes=Axes,zAxis=zAxis,boxsize=boxsize,boxlos=boxlos,pixres=pixres,pixreslos=pixreslos,\
# Nbins=Nbins,DPI=DPI,CMAP=CMAP)
# ------------------------------------------------------------------------------#

# Histogram 2D Movie:
ProjectionMovieLoop(
    SimDirectory=simfile,
    Start=snap_start,
    End=snap_end,
    n_processes=n_processes,
    Axes=Axes,
    zAxis=zAxis,
    boxsize=boxsize,
    boxlos=boxlos,
    pixres=pixres,
    pixreslos=pixreslos,
    Nbins=Nbins,
    DPI=DPI,
    CMAP=CMAP,
)
