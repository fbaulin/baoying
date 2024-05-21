""" Main file with the library
"""
"""
Main features different from matbplotlib and plotly
- labels on hovers
- markers on clicks
- simplified use of animation
- data based visualization improvement
    - smart colors (based on the number of curves)
    - smart ad-hoc scaling
- ? interactive plot control
    
Data is contained in a container,
then operator makes a decision which plot he whants to adjust 
and select different kinds of plots

"""

from matplotlib import plot as plt


class Data:

    def __init__(self) -> None:
        Y=None
        X=None
        fig=None


    def add(self, Y, X=None):
        """
        add new dataset
        """
        pass


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


    @staticmethod
    def load():
        pass


def plotit():
    """ Plotting method """
    pass

def export():
    pass

