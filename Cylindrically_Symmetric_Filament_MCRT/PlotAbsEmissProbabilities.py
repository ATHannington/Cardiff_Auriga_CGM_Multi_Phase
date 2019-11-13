"""
Title:              PlotAbsEmissProbabilities.py
Created by:         Andrew T. Hannington
Use with:           RadTrans_*.f90
                        Created by: A. P. Whitworth et al.

Date Created:       16/10/2019

Usage Notes:
    "Purpose: to generate an explanatory plot illustrating how we converge on a wavelength
             from a randomly generated probability in the RT_LumPack_** routines in
             RadTrans."

Known bugs:
            Not generalised to centering of vlines on different lambda value.
            Some tweaking of "FUDGE" factor will be needed, and possibly further
            corrections.

"""
import matplotlib.pyplot as plt
import numpy as np
import math

#Set number of steps overwhich to sample curve
NSTEPS = 1000

# Shift and scaling factor of sigmoid curve to bring towards varying between 0
#  and 1 in range x=[0,1]
XSHIFT = 0.5
XSCALE = 15.

# Set x and y min and max values for range
XMIN = 0.
XMAX = 1.
YMIN = 0.
YMAX = 1.

#XY value for vertical line and horizontal line
YLINEXYVAL=0.5
# xand y line minimum values
YLINEYMIN = 0.
XLINEXMIN = 0.
#Value to fudge/make work the vlines landing on the sigmoid function.
FUDGE = 0.2*YLINEXYVAL

#Set the numeber of ticks and number of the subset of those ticks to be labelled
# for both x and y
NTICKS = {"y":50,"x":25,"xlabeled":5,"ylabeled":10}

#Set line positions in x and start and stop in y
YLINE = {"x":YLINEXYVAL,"ymin":YLINEYMIN,"ymax":YLINEXYVAL}
xminval = YLINEXYVAL - 0.5/float(NTICKS['x'])
xmaxval = YLINEXYVAL + 0.5/float(NTICKS['x'])
YMINLINE = {"x":xminval,"ymin":YLINEYMIN,"ymax":xminval*(1.0 - FUDGE)}
YMAXLINE = {"x":xmaxval,"ymin":YLINEYMIN,"ymax":xmaxval*(1.0 + FUDGE)}

#Horizontal line
XLINE = {"y":YLINEXYVAL,"xmin":XLINEXMIN,"xmax":YLINEXYVAL}
#------------------------------------------------------------------------------#
#   Functions
#
#------------------------------------------------------------------------------#

def shifted_sigmoid(x):
    """
    Produces a scaled sigmoid, shifted in x by XSHIFT and compressed in "frequency"
    in x by "XSCALE". This produces a logistic sigmoid between y=0 and y=1 in the range
    x=0 x=1 using XSHIFT = 0.5 XSCALE = 15.
    """
    s = np.array([(1./(1. + math.exp(XSCALE*(-z+XSHIFT)) )) for z in x])
    return s
#------------------------------------------------------------------------------#
def get_ticks(nticks,nlabeledticks,valmin=0.,valmax=1.):
    """
    Produces nticks of ticks in range valmin to valmax. Of those ticks,
    nlabeledticks have labels with correct values, the remainder are blank.
    """
    ticks = np.linspace(valmin,valmax,num=nticks+1)
    labels = []
    for i in range(0,nticks+1):
        val = ticks[i]
        if i % nlabeledticks != 0:
            labels.append(" ")
        else:
            labels.append(val)
    return ticks, labels
#------------------------------------------------------------------------------#

# Set up X and Y values
x = np.linspace(XMIN,XMAX,NSTEPS)
y = shifted_sigmoid(x)

# Get x and y ticks and labels
xticks, xlabels = get_ticks(NTICKS['x'],NTICKS['xlabeled'],valmin=XMIN, valmax=XMAX)
yticks, ylabels = get_ticks(NTICKS['y'],NTICKS['ylabeled'],valmin=YMIN, valmax=YMAX)

# Set new labels on ticks at the line positions. i.e. have x intercept read lambda0
# and y intercept of lines read LRD.
ylabels[int(int(NTICKS['y'])/2)] = "LRD"
xlabels[int(math.ceil(int(NTICKS['x'])/2))] = r"$\lambda_{_0}$"

# Set up plot axes
ax = plt.gca()

# Line plot
plt.plot(x,y)

#Add vertical lines
plt.vlines(x=[YLINE["x"], YMINLINE["x"], YMAXLINE["x"]], \
ymin=[YLINE["ymin"], YMINLINE["ymin"], YMAXLINE["ymin"]], \
ymax=[YLINE["ymax"], YMINLINE["ymax"], YMAXLINE["ymax"]], \
colors=["black","red","red"],linestyles=["dashed","solid","solid"])

#Add horizontal lines
plt.hlines(y=XLINE["y"],xmin=XLINE['xmin'],xmax=XLINE['xmax'], \
colors="black", linestyles="dashed")

# Set x and y axis labels
ax.set_xlabel(r"$\frac{\lambda}{\lambda_{_{\rm MAX}}} \; [/]$")
ax.set_ylabel(r"$P_{_{{\rm BB},\lambda}}(T) \; [/]$")

#Set ticks and labels
ax.set(xticks=xticks)
ax.set_xticklabels(xlabels)
ax.set(yticks=yticks)
ax.set_yticklabels(ylabels)
#limits
plt.xlim(XMIN,XMAX)
plt.ylim(YMIN,YMAX)
#show on screen
plt.show()
