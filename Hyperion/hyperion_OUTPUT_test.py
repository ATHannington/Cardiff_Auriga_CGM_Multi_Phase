import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hyperion.model import Model
from hyperion.util.constants import *
from hyperion.model import ModelOutput


m = ModelOutput('model.rtout')
# m.get_sed()
# m.get_image()



# Extract the image for the first inclination, and scale to 300pc. We
# have to specify group=1 as there is no image in group 0.
image = m.get_image(group=0,inclination=0, distance=pc, units='MJy/sr')

# Open figure and create axes
fig = plt.figure(figsize=(8, 8))

# Pre-set maximum for colorscales
VMAX = {}
VMAX[100] = np.nanpercentile(image.val[:, :, np.argmin(np.abs(100 - image.wav))],99.)
VMAX[300] = np.nanpercentile(image.val[:, :, np.argmin(np.abs(300 - image.wav))],99.)
VMAX[400] = np.nanpercentile(image.val[:, :, np.argmin(np.abs(400 - image.wav))],99.)
VMAX[500] = np.nanpercentile(image.val[:, :, np.argmin(np.abs(500 - image.wav))],99.)
# We will now show four sub-plots, each one for a different wavelength
for i, wav in enumerate([100, 300, 400, 500]):

    ax = fig.add_subplot(2, 2, i + 1)

    # Find the closest wavelength
    iwav = np.argmin(np.abs(wav - image.wav))

    # Calculate the image width in arcseconds given the distance used above
    w = np.degrees((1 * pc) / image.distance) * 60.

    # This is the command to show the image. The parameters vmin and vmax are
    # the min and max levels for the colorscale (remove for default values).
    ax.imshow(np.sqrt(image.val[:, :, iwav]), vmin=0, vmax=np.sqrt(VMAX[wav]),\
    cmap=plt.cm.gist_heat, origin='lower', extent=[-w, w, -w, w])
    # ax.imshow(np.sqrt(image.val[:, :, iwav]), vmin=0, vmax=np.sqrt(VMAX[wav]),
    #           cmap=plt.cm.gist_heat, origin='lower', extent=[-w, w, -w, w])

    # Finalize the plot
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.set_xlabel('x (arcmin)')
    ax.set_ylabel('y (arcmin)')
    ax.set_title(str(wav) + ' microns', y=0.88, x=0.5, color='white')

fig.savefig('test_plot.png', bbox_inches='tight')
