#%%
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import use
from matplotlib.animation import FuncAnimation
#%%
use('Qt5Agg')
#FIXME When Qt5Agg is used getting error 

def run_animation(x, y, file=None, x_label='x', y_label='y'):
    fig = plt.figure()
    ax = plt.axes(xlabel=x_label, ylabel=y_label, xlim=(np.min(x), np.max(x)), ylim=(np.min(y), np.max(y)))
    line, = ax.plot([], [], lw=3)
    def init():
        line.set_data([], [])
        return line,
    def animate(i):
        line.set_data(x,y[:,i])
        return line,
    anim = FuncAnimation(fig, animate, init_func=init,
                         frames=y.shape[1], interval=20, blit=True, 
                         repeat=True, repeat_delay=0)
    if file is not None: anim.save(file, writer='PillowWriter', dpi=96, fps=24, bitrate=256)
    plt.show()
    return anim

print('Hi')
x=np.linspace(0,8*np.pi,10**3)

y=np.cos(x[:,np.newaxis]+np.transpose(np.linspace(0,2*np.pi,10)))


run_animation(x, y)
# plt.plot(x, y)
# plt.show()

#%%
ax = plt.axes()
ax.plot(np.linspace(0,10,100),np.linspace(0,10,100)**2)
