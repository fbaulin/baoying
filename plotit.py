import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
from typing import Iterable
import numpy as np

# Make a line plot using plotly
def plotit_legacy(y, x=None,
           title: str = None, legend=None,
           x_label=None, y_label=None,
           x_lims: list = None, y_lims:list=None,
           x_scale="linear", y_scale="linear",
           error_x=None, error_y=None, error_thickness=1.0,
           dashes=None, widths=None, opacity:Iterable=None, mode='lines',
           colors=None,
           renderer=None,
           file_name=None,
           figure=None, row_col=[None, None]):
    """
    Plot data using plotly
    Plot x and y data. Extra options:
        - error wiskers plotting is avalilable;
        - export to file when file_name is specified.
    Args:
        y: (ndarray): y axis.
        x, (ndarray, optional): x axis values, if None - automatic x axis generation. Defaults to None.
        title (str, optional): Title of the plot (sets title only for the figure initiation). Defaults to None.
        legend (list, optional): List of curve titles, if False is supplied, than legend item is hidden. Defaults to None.
        x_label (str, optional): Label for the X axis. Column-wise curves. Defaults to None.
        y_label (str, optional): Label for the Y axis. Column-wise curves. Defaults to None.
        x_lims (tuple, optional): Limits of the range for the X axis. Defaults to None.
        y_lims (tuple, optional): Limits of the range for the Y axis. Defaults to None.
        x_scale (str, optional): Scale of the x-axis. Defaults to "linear".
        y_scale (str, optional): Scale of the y-axis. Defaults to "linear". Valid values: "linear", "log".
        error_x (ndarray, optional): Offset for the error wiskers. Defaults to None.
            Cases
                None - do not draw horizontal error bars
                ndarray - use supplied array for error bars.
        error_y (ndarray, optional): Offset for the error wiskers. Defaults to None.
            Cases
                None - no error bars are drawn & no statistical preprocessing is performed.
                False - extract mean values along the horz axis of array, draw mean.
                True -  extract both mean & std values along the horz axis of array, draw mean with error bars.
                ndarray - use supplied array for error bars.
        error_thickness (float): Thickness of the error whiskers. Defaults to 1.
        dashes (str, optional): List of line types. Defaults to None.
            Valid values: 'solid' (=None), 'dash', 'dot', 'dashdot'.
        widths (optional): List of line widths, or a single value. Defaults to None.
        opacity (optional): List or value of opacity, that is transfered to the go.Scatter.
        colors (optional): List of color names/value redirected to the go.Scatter object.
            Supports integer input, takes it as a number in D3 color sequence.
        renderer (optional): Defines how to render the plot. Defaults to None.
            Cases
                None - renders to the default place set in the package. Suppresed if figure supplied.
                False - Doesn't show plot.
                True - renders to browser (equivalent to 'browser', but suppresed if figure supplied).
                str - string value is used to define renderer parameter in show().
        figure (optional): Returning figure for future use or modifying existing. Defaults to None.
            Cases
                None - creating new figure. Nothing is returned.
                plotly.graphic_objects.Figure - modifying the supplyed figure & returning None.
                empty list - creating new figure appending it to the list & returning None.
                list ending with None - replacing None with a newly created Figure object.
                list with Figure object - extracting the Figure, modifying it & returning None.
        row_col (list, optional): If subplots are used, select row and column.
            To use make_subplot from plotly.subplot:
                make_subplots(rows=4, cols=1, shared_xaxes=True).update_layout(template='none')
        file_name (str, optional): Name of the file to export the plot image. Defaults to None
    Raises:
        ValueError: Error caused by inappropriate
    Returns:
        plotly.graph_objects.Scatter: Returns Figure object, that can be altered.
    """
    if isinstance(y, np.ndarray):
        if y.ndim == 1:
            y = y[:, np.newaxis]  # if 1D array is supplied, make it 2D

    if isinstance(error_y, bool):
        if error_y is True:  # matrices with error_y parameter true - prepare variance plot
            error_y = np.concatenate([np.std(darray, axis=1, keepdims=True) for darray in y], axis=1)
        y = np.concatenate([np.mean(darray, axis=1, keepdims=True) for darray in y], axis=1)

    if isinstance(y, (list, tuple)):  # if list of values glue it together
        if y[0].ndim == 1:
            y = np.stack(y, axis=1)  # one-dimensional vector supplied
        elif y[0].shape[1] == 1:
            y = np.concatenate(y, axis=1)  # two-dimensional vector supplied
        else:
            raise ValueError("Can't combine the data - check format")

    if np.any(np.iscomplex(y)):
        Warning('Complex data is supplied - ignoring imaginary parts')
        y = np.real(y)

    n_plot_lines = y.shape[1]
    if n_plot_lines > 10:
        n_plot_lines = 10  # if more than 20 reduce the number of curves

    showlegend = legend is not None
    if legend is None:
        legend = [None, ] * n_plot_lines

    line_properties = [dict() for i in range(n_plot_lines)]
    if widths is not None:
        if isinstance(widths, (float, int)):
            widths = [widths] * n_plot_lines
        [p.update(width=v) for p, v in zip(line_properties, widths)]

    if dashes is not None:
        if isinstance(dashes, str):
            dashes = [dashes] * n_plot_lines
        [p.update(dash=v) for p, v in zip(line_properties, dashes)]

    if colors is not None:
        if isinstance(colors, (int, str)):
            colors = [colors] * n_plot_lines
        if isinstance(colors, list):
            colors = [qualitative.D3[c] if isinstance(c, int) else c for c in colors]
        [p.update(color=v) for p, v in zip(line_properties, colors)]

    if isinstance(opacity, (type(None), float, int)):
        opacity = [opacity] * n_plot_lines

    if x is None:
        x = np.arange(y.shape[0])  # create missing x axis values

    plots = []
    for i_t in range(n_plot_lines):
        if isinstance(error_x, np.ndarray):  # repack errorbar for the plot
            err_x = dict(type='data', array=error_x[:, i_t], visible=True, thickness=error_thickness)
        else:
            err_x = None
        if isinstance(error_y, np.ndarray):  # repack errorbar for the plot
            err_y = dict(type='data', array=error_y[:, i_t], visible=True, thickness=error_thickness)
        else:
            err_y = None

        plots.append(go.Scatter(x=x, y=y[:, i_t], mode=mode, showlegend=bool(legend[i_t]), name=legend[i_t],
                                line=line_properties[i_t], opacity=opacity[i_t], error_x=err_x, error_y=err_y))

    if isinstance(figure, list) and len(figure) == 0:  # empty list - create new figure & store
        figure.append(go.Figure(layout=go.Layout(title=title)))
    if isinstance(figure, list) and len(figure) >= 1:  # non-empty list
        if figure[-1] is None:  # if last cell is empty, store new figure instead of none
            figure[-1] = go.Figure(layout=go.Layout(title=title))
        elif isinstance(figure[-1], (int, tuple)):  # create subplots
            if isinstance(figure[-1], tuple):
                r, c = figure[-1]
            elif figure[-1] < 10:
                r, c = (figure[-1], 1)
            else:
                r, c = divmod(figure[-1], 10)
            figure[-1] = make_subplots(rows=r, cols=c, shared_xaxes='columns', shared_yaxes='rows')
        fig = figure[-1]  # load last object from the list
    elif isinstance(figure, go.Figure):
        fig = figure
    elif figure is None:
        fig = go.Figure(layout=go.Layout(showlegend=showlegend, title=title))
    else:
        raise TypeError('Improper figure object: use go.Figure or list')

    if isinstance(row_col, int):
        row_col = divmod(row_col, 10)

    for p in plots:  fig.add_trace(p, row=row_col[0], col=row_col[1])    
    fig.update_layout(showlegend=showlegend, template='none')   
    fig.update_xaxes(title_text=x_label, range=x_lims, type=x_scale, tickformat='s')    
    fig.update_yaxes(title_text=y_label, range=y_lims, type=y_scale, tickformat='s')    
    
    if   renderer is None:        
        if figure is None:             fig.show()        
        else:                          pass   
    elif renderer is False:            pass    
    elif renderer is True:                     
        if figure is None:             fig.show(renderer='browser')        
        else:                          pass    
    elif isinstance(renderer, str):    fig.show(renderer=renderer)    
    else:raise TypeError('Renderer type error')    
    if file_name is not None: fig.write_image(file_name)    
    if figure is not None: return None  #  supplied figure var is changed -> function must retuen None



def plotit_llm_update(y, x=None, 
           title: str = None, legend=None, 
           x_label=None, y_label=None, 
           x_lims: list = None, y_lims: list = None, 
           x_scale="linear", y_scale="linear", 
           error_x=None, error_y=None, error_thickness=1.0, 
           dashes=None, widths=None, opacity: Iterable = None, mode='lines', 
           colors=None, 
           renderer=None, 
           file_name=None, 
           figure=None, row_col=[None, None]):
    """
    Plot data using plotly.
    Args:
        y (ndarray): y axis.
        x (ndarray, optional): x axis values, if None - automatic x axis generation.
        title (str, optional): Title of the plot (sets title only for the figure initiation).
        legend (list, optional): List of curve titles, if False is supplied, then legend item is hidden.
        x_label (str, optional): Label for the X axis.
        y_label (str, optional): Label for the Y axis.
        x_lims (list, optional): Limits of the range for the X axis.
        y_lims (list, optional): Limits of the range for the Y axis.
        x_scale (str, optional): Scale of the x-axis. Defaults to "linear".
        y_scale (str, optional): Scale of the y-axis. Defaults to "linear". Valid values: "linear", "log".
        error_x (ndarray, optional): Offset for the error whiskers.
        error_y (ndarray, optional): Offset for the error whiskers.
        error_thickness (float): Thickness of the error whiskers.
        dashes (str, optional): List of line types. Valid values: 'solid' (=None), 'dash', 'dot', 'dashdot'.
        widths (optional): List of line widths, or a single value.
        opacity (Iterable, optional): List or value of opacity, transferred to the go.Scatter.
        colors (optional): List of color names/values redirected to the go.Scatter object.
        renderer (optional): Defines how to render the plot.
        figure (optional): Returning figure for future use or modifying existing.
        row_col (list, optional): If subplots are used, select row and column.
        file_name (str, optional): Name of the file to export the plot image.
    Returns:
        plotly.graph_objects.Scatter: Returns Figure object, that can be altered.
    """
    # Handle 1D input by expanding to 2D
    if isinstance(y, np.ndarray) and y.ndim == 1:
        y = y[:, np.newaxis]

    # Handle list/tuple inputs by combining into ndarray
    if isinstance(y, (list, tuple)):
        if y[0].ndim == 1:
            y = np.stack(y, axis=1)
        else:
            y = np.concatenate(y, axis=1)
    
    if isinstance(error_y, bool) and error_y:
        error_y = np.concatenate([np.std(arr, axis=1, keepdims=True) for arr in y], axis=1)
        y = np.concatenate([np.mean(arr, axis=1, keepdims=True) for arr in y], axis=1)

    if np.any(np.iscomplex(y)):
        print("Warning: Complex data supplied - ignoring imaginary parts")
        y = np.real(y)

    n_plot_lines = min(y.shape[1], 10)  # Limit to 10 lines for clarity
    if legend is None:
        legend = [None] * n_plot_lines
    line_properties = [{'dash': dashes[i] if dashes else None,
                        'width': widths[i] if widths else None,
                        'color': colors[i] if colors else None}
                       for i in range(n_plot_lines)]
    if x is None:
        x = np.arange(y.shape[0])
    plots = [
        go.Scatter(
            x=x, y=y[:, i], mode=mode, name=legend[i], 
            line=line_properties[i], 
            opacity=opacity[i] if opacity else None,
            error_x=dict(type='data', array=error_x[:, i], visible=True, thickness=error_thickness) if isinstance(error_x, np.ndarray) else None,
            error_y=dict(type='data', array=error_y[:, i], visible=True, thickness=error_thickness) if isinstance(error_y, np.ndarray) else None
        ) 
        for i in range(n_plot_lines)
    ]
    if figure is None:
        fig = go.Figure(layout=go.Layout(showlegend=True, title=title))
    elif isinstance(figure, list):
        if not figure:
            figure.append(go.Figure(layout=go.Layout(title=title)))
        if figure[-1] is None:
            figure[-1] = go.Figure(layout=go.Layout(title=title))
        fig = figure[-1]
    else:
        fig = figure

    if isinstance(row_col, int):
        row_col = divmod(row_col, 10)
    for p in plots:
        fig.add_trace(p, row=row_col[0], col=row_col[1])
    fig.update_layout(template='none')
    fig.update_xaxes(title_text=x_label, range=x_lims, type=x_scale, tickformat='s')
    fig.update_yaxes(title_text=y_label, range=y_lims, type=y_scale, tickformat='s')
    if renderer is None:
        if figure is None:
            fig.show()
    elif renderer:
        if isinstance(renderer, str):
            fig.show(renderer=renderer)
        elif renderer is True:
            fig.show(renderer='browser')
    if file_name:
        fig.write_image(file_name)
    if figure is not None:
        return None  # If figure is modified in place
    # return fig