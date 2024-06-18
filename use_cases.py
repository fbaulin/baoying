""" This file contains use cases
"""
import baoying as by
import numpy as np

xaxis = np.linspace(-4,4)
xaxis_alt = 3e8/xaxis
Xmesh = np.meshgrid(xaxis, xaxis)
ydata = xaxis**2
Ydata = Xmesh**2+Xmesh**2


#%% 
"""Quick plotting of curves"""

# Run a single curve plot for only Y-data nd-array
by.curves(ydata)

# Run a multi curve plot for Y-data nd-matrix
by.curves(Ydata)

# Run a single curve plot for Y nd-matrix and single X nd-array
by.curves(Ydata, xaxis)

# Redundant as it repeats Data framework?
# # Run multiple curves on-the-go - basically a wrapper around Data class
# f = by.figure() #
# # list in the figure all preferences that could be common for the plots
# by.curves(ydata)
# by.curves(ydata+1)
# f.show()        # mode=('curves','subplots')


#%%
"""Storage & rendering of multiple curves"""
# Creatng a figure
data = by.Data()
# Data is accumulated in the data
data.add(ydata)
data.add(ydata+1, order='b')                # add to the back (any word starting with b)
data.add(ydata+2, color=[0.1, 0.3, 0.6])    # set custom color
# Combining data and drawing the fig
data.color[1] = 'y'  # change color using matlab short names
data.color[2] = 'y0.1'  # change color using matlab short names + brightness values

data.set_axis(xaxis, alt_axis=xaxis_alt)   # adding x axis in the beginning (or the end)
data.curves()    # curve plots
data.animate(fps=0.1)   # single curve but animated (args: filename)

data.set_axes(Xmesh)   # adding x-y axes in the beginning (or the end)
# data.set_axes(X_mesh)   # adding x-y axes in the beginning (or the end)
data.mesh()      # mesh plot 
data.contour()   # contour map
data.heatmap()   # heat map (only when shared axis)

#%%
""" Multiple 2D pictures """
# Creating a figure
data = by.data()
# Data is accumulated in the data
data.add(Ydata)
data.add(Ydata**(1/2))
data.add(Ydata**(1/3))
data.animate()                      # default multi curve animation
data.animate(mode='mesh')           # mesh animation
data.animate(mode='heatmap')        # heatmap animation

data.save(r'{UTC}.pkl')        # save results of the processing
# with preset tags 
#   {UTC} - UTC format local time
#   {ylabel} - label on y axis
#   {title} - title of data
by.Data('filename.pkl')
# by.load('filename.pkl')

#%% Adding 4D data on-the-go
# Simultaneous add of the X-Y mesh
x = np.linspace(-3, 3, 40)      # axis of some quasicont. variable
y = np.linspace(1,2,5)          # axis of a discrete parameter
data = by.data(x_axes=[x,y])
for i in range(24):
    for j in len(y):
        z = y[j]*x**2
        data.add(z)         # add curve instance
    data.frame()            # 

# X- Axis is added first, Y is added in a cycle
x = np.linspace(-3, 3, 40)
data = by.data(X=[x])
for i in range(24):
    y = i
    for j in len(y):
        z = y*x**2
        data.add(z)
    data.frame()

# X & Y are added in a cycle
x = np.linspace(-3, 3, 40)
y = np.linspace(1,2,5)
data = by.data(X=[x])
for i in range(24):
    for j in len(y):
        z = y*x**2
        data.add(z)
    data.frame()


# matrix-wise
data = by.data()
for i in range(24):
    data.add(Ydata)
    

#%%
"""Extra features"""
assert by.si([2e-9, 1e9], name='Hz')==('2nHz', '1GHz')  # converts value to SI


#%%
import numpy as np
import pandas as pd
from matplotlib import use, pyplot as plt

#%%
# use('Qt5Agg')
use('WebAgg')
df = pd.DataFrame(np.random.normal(0,1,(200,10))+np.linspace(0,10,200)[:,np.newaxis])

df.plot()
plt.plot(df.values[:,0:2])
plt.show()

