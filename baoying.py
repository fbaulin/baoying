""" Main file with the library
"""
"""
Main features different from matbplotlib and plotly
- labels on hovers
- markers on clicks
- horizontal and vertical lines
- simplified use of animation
- data based visualization improvement
    - smart colors (based on the number of curves)
    - smart ad-hoc scaling
- easily axesed double-valued axes
- ? interactive plot control

- saving data

Data is contained in a container,
then operator makes a decision which plot he whants to adjust
and select different kinds of plots


Notes:
- use plotting in a newly spawned processes for interactive?

"""

from matplotlib import plot as plt
import numpy as np
from  plotly import graph_objects as go
from  plotly.subplots import make_subplots
import pandas as pd


class Data:


    def __init__(self,
                 renderer=None,
                 graph_type=None,
                 animate=False,
                 x_axes=None,
                 xlabels=None,
                 ylabels=None,
                 share_axis=False
                 ) -> None:
        # data and curve properties
        self.Y = []       # data is stored in list or lols and rendered to np only when drawing
        self.X = []                 if x_axes is None else x_axes
        self.names = []
        self.xlabels = []           if xlabels is None else xlabels
        self.ylabels = []           if ylabels is None else ylabels
        # plotting properties
        self.share_ax = share_axis      # share axes to reduce amount of data
        self.renderer = 'browser'   if renderer is None else renderer
        self.graph_type = graph_type    # default type of graph
        self.animate = animate          # if frame mode true, the data is treated as 3d
        self.fig = None
        self.graph_type = dict()        # graph type sets the axis type
        self.df = None                  # dataframe with the y data, x data and settings
        self.preframe = None            # flag of framing before data or after


    def add(self, Y,
            X=None,
            name=None,
            color=None,
            width=None,
            opacity=None
            ):
        """
        add new dataset
        supplies plotting parameters and stores those for transfering to later
            
        """
        if df is None:
            self.preframe=False
            df = pd.DataFrame(columns=['axis0'])
        if color is None: self.colors.append(None)
        elif isinstance(color, str): name2rgb(color)
        elif isinstance(color, list): color
        self.widths.append(width)
        self.opacities.append(opacity)


    def update(self, id, **kwargs):
        pass


    def remove(self, id):
        if isinstance(id, int): pass    # id - number of the data
        elif isinstance(id, str): pass  # id - name of the curve

    # Runs building curves
    def curves(self):
        self.show({0})     # run show with


    def mesh(self):
        pass


    def heatmap(self):
        pass


    def animate(self, mode='curves'):
        pass


    def hist(self, axis=None):
        pass


    def imag(self):
        pass


    def export(self, form='pickle', name=None):
        """pickles, xls, csv, png"""
        pass


    def horzline(self):
        pass


    def vertline(self):
        pass

    # Start new frame of data
    def case(self, axis=None, name=None):
        """
        case method (should work whether in beginning or an end)
        #NOTE! when cases accessed the index starts with the outter most
        axis - name of the axis cases (for instance axis='Linewidth')
        name - name of the case 
                if case was supplied before any data, 
                than the name is associated with the following data
                if case was supplied after data add, 
                than the name is associated with the preciding following data
        """
        if self.df is None:     # set the flag of preframing
            self.preframe=True
        if self.preframe:       # engage frame before data add
            pass                # create dataframe
        else:                   # engage frame after data add
            pass                # add column & modify dataframes rows if parameters are supplied

    # render data
    def show(structure):
        pass


    @staticmethod
    def load():
        pass


def name2rgb(name):
    pass


def rgb2name(r,g,b):
    pass

# This method operates via data structure
def curves():
    pass


def mesh():
    pass


def heatmap():
    pass


def export():
    pass


def plotit( y, x=None, title:str=None, legend=None, 
            x_label=None, y_label=None, x_lims:list=None, y_lims:list=None,
            x_scale="linear", y_scale="linear", 
            error_x=None, error_y=None, error_thickness=1.0, dashes=None, widths=None,
            opacity=None, mode='lines', colors=None, renderer=None, file_name=None, figure=None, row_col=[None, None] 
        ):
    pass


def export():
    pass
