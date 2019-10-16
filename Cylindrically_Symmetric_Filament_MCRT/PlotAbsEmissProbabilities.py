import matplotlib.pyplot as plt
import numpy as np
import math
"""
Author: A. T. Hannington
Date: 16/10/2019
Purpose: to generate an explanatory plot illustrating how we converge on a wavelength
         from a randomly generated probability in the RT_LumPack_** routines in
         RadTrans.
"""
NSTEPS = 1000
XSHIFT = 0.5
XSCALE = 15.

XMIN = 0.
XMAX = 1.
YMIN = 0.
YMAX = 1.

YLINEXYVAL=0.5
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
x = np.linspace(XMIN,XMAX,NSTEPS)
y = shifted_sigmoid(x)

xticks, xlabels = get_ticks(NTICKS['x'],NTICKS['xlabeled'],valmin=XMIN, valmax=XMAX)
yticks, ylabels = get_ticks(NTICKS['y'],NTICKS['ylabeled'],valmin=YMIN, valmax=YMAX)

ylabels[int(int(NTICKS['y'])/2)] = "LRD"
xlabels[int(math.ceil(int(NTICKS['x'])/2))] = r"$\lambda_{_0}$"

ax = plt.gca()
plt.plot(x,y)

plt.vlines(x=[YLINE["x"], YMINLINE["x"], YMAXLINE["x"]], \
ymin=[YLINE["ymin"], YMINLINE["ymin"], YMAXLINE["ymin"]], \
ymax=[YLINE["ymax"], YMINLINE["ymax"], YMAXLINE["ymax"]], \
colors=["black","red","red"],linestyles=["dashed","solid","solid"])

plt.hlines(y=XLINE["y"],xmin=XLINE['xmin'],xmax=XLINE['xmax'], \
colors="black", linestyles="dashed")

ax.set_xlabel(r"$\frac{\lambda}{\lambda_{_{\rm MAX}}} \; [/]$")
ax.set_ylabel(r"$P_{_{{\rm BB},\lambda}}(T) \; [/]$")
ax.set(xticks=xticks)
ax.set_xticklabels(xlabels)
ax.set(yticks=yticks)
ax.set_yticklabels(ylabels)
plt.xlim(XMIN,XMAX)
plt.ylim(YMIN,YMAX)
plt.show()
