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
        self.colors = []
        self.widths = []
        self.lines = []
        self.opacities = []
        # plotting properties
        self.share_ax = share_axis      # share axes to reduce amount of data
        self.renderer = 'browser'   if renderer is None else renderer
        self.graph_type = graph_type    # default type of graph
        self.animate = animate          # if frame mode true, the data is treated as 3d
        self.fig = None


    def add(self, Y,
            X=None,
            name=None,
            color=None,
            width=None,
            opacity=None
            ):
        """
        add new dataset
        """
        self.Y.append(Y) # solve how to distinguish
        self.X.append(X)
        if color is None: self.colors.append(None)
        elif isinstance(color, str): self.colors.append(name2rgb(color))
        elif isinstance(color, list): self.colors.append(color)
        self.widths.append(width)
        self.opacities.append(opacity)


    def frame(self, **kwargs):
        """ add new frame, if  data supplied in kwargs, add curves through add """
        pass


    def update(self, id, **kwargs):
        pass


    def remove(self, id):
        if isinstance(id, int): pass    # id - number of the data
        elif isinstance(id, str): pass  # id - name of the curve


    def curves(self):
        pass


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
    def case(self, name=None):
        """
        name - name of the case
                if case was supplied before any data, 
                than the name is associated with the following data
                if case was supplied after data add, 
                than the name is associated with the preciding following data
        """
        pass


    @staticmethod
    def load():
        pass


def name2rgb(name):
    pass


def rgb2name(r,g,b):
    pass


def curves():
    pass


def mesh():
    pass


def heatmap():
    pass


def export():
    pass


def plotit():
    """ Plotting method """
    pass

def export():
    pass

