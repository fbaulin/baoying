""" This file contains use cases
"""
import baoying as by
import numpy as np

xaxis = np.linspace(-4,4)
Xmesh = np.meshgrid(xaxis, xaxis)
ydata = xaxis**2
Ydata = Xmesh**2+Xmesh**2


#%% Running functions for quick plotting

# Run a single curve plot for only Y-data nd-array
by.curves(ydata)

# Run a multi curve plot for Y-data nd-matrix
by.curves(Ydata)

# Run a single curve plot for Y nd-matrix and single X nd-array
by.curves(Ydata, xaxis)


# Run a multi curve plot for 
by.curves(Ydata, xaxis)

#%% Run a single curve plot multiple times
# Creatng a figure
fig = by.fig()
# Data is accumulated in the fig
fig.add(ydata)
fig.add(ydata+1)
fig.add(ydata+2, color=[0.1, 0.3, 0.6])     # set custom color
# Combining data and drawing the fig
fig.color[1] = 'y'  # change color
fig.curves()
fig.mesh()
fig.animate()

