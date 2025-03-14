""" Plotit module
Модуль содержит отдельные функции для быстрой интеграции в произвольный код через 
- копирование модуля в папку проекта или 
- текста функции в текст пользовательской программы.

"""

# общие библиотеки
from typing import Iterable
import numpy as np
# библиотеки для plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import qualitative
# библиотеки для matplotlib
import matplotlib
from matplotlib import pyplot as plt

from matplotlib.animation import FuncAnimation  # библиотека для анимации


# Вывод графиков с использованием plotly
def plotit_plotly(
        y, x=None,
        names=None,
        lines=None, widths=None, colors=None, opacities=None,
        x_label=None, y_label=None,
        x_scale="linear", y_scale="linear",
        x_lims: list = None,    y_lims: list = None,
        title: str = None, 
        ax=None, row_col=[None, None],
        error_x=None, error_y=None, error_thickness=1.0,
        mode='lines',
        renderer=None,
        file_name=None,
        ):
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
        colors (optional): List of color names/values redirected to the go.Scatter object. Int values are supported.
        renderer (optional): Defines how to render the plot.
        ax (optional): Returning ax for future use or modifying existing.
        row_col (list, optional): If subplots are used, select row and column.
        file_name (str, optional): Name of the file to export the plot image.
    Returns:
        plotly.graph_objects.Scatter: Returns Figure object, that can be altered.

    Example 1. Plot single ndarray:
        >>> import numpy as np
        >>> from plotit import plotit_plotly as plot
        >>> y = np.arange(10)**2
        >>> plot(y)

    Example 2. Plot list:
        >>> import numpy as np
        >>> from plotit import plotit_plotly as plot
        >>> y = [np.arange(10)**2, np.arange(10)]
        >>> plot(y, names=['quad', 'lin'])

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
    
    # If colors are integers, select by order 
    if colors is not None:
        if isinstance(colors, (int, str)):  colors = [colors] * n_plot_lines
        if isinstance(colors, list):        colors = [qualitative.D3[c] if isinstance(c, int) else c for c in colors]

    if names is None:
        names = [None] * n_plot_lines
    line_properties = [{'dash': lines[i] if lines else None,
                        'width': widths[i] if widths else None,
                        'color': colors[i] if colors else None}
                       for i in range(n_plot_lines)]
    if x is None:
        x = np.arange(y.shape[0])
    plots = [
        go.Scatter(
            x=x, y=y[:, i], mode=mode, name=names[i],
            line=line_properties[i],
            opacity=opacities[i] if opacities else None,
            error_x=dict(type='data', array=error_x[:, i], visible=True, thickness=error_thickness) if isinstance(error_x, np.ndarray) else None,
            error_y=dict(type='data', array=error_y[:, i], visible=True, thickness=error_thickness) if isinstance(error_y, np.ndarray) else None
        )
        for i in range(n_plot_lines)
    ]
    if ax is None:
        fig = go.Figure(layout=go.Layout(showlegend=True, title=title))
    elif isinstance(ax, list):
        if len(ax)==0:
            ax.append(go.Figure(layout=go.Layout(title=title)))
        elif ax[-1] is None:
            ax[-1] = go.Figure(layout=go.Layout(title=title))
        elif isinstance(ax[-1], int):
            f = ax[-1]
            f = (f,1) if f<10 else divmod(f, 10)
            ax[-1] = make_subplots(*f, shared_xaxes='columns')
        fig = ax[-1]
    else:
        fig = ax

    if isinstance(row_col, int):
        if row_col<=10:
            row_col = (row_col, 1)
        else:		   row_col = divmod(row_col, 10)
    for p in plots:
        fig.add_trace(p, row=row_col[0], col=row_col[1])
    fig.update_layout(template='none')
    fig.update_xaxes(title_text=x_label, range=x_lims, type=x_scale, tickformat='s')
    fig.update_yaxes(title_text=y_label, range=y_lims, type=y_scale, tickformat='s')
    if renderer is None:
        if ax is None:
            fig.show()
    elif renderer:
        if isinstance(renderer, str):
            fig.show(renderer=renderer)
        elif renderer is True:
            fig.show(renderer='browser')
    if file_name:
        fig.write_image(file_name)
    if ax is not None:
        return None  # If figure is modified in place
    # return fig


# Вывод графиков с использованием matplotlib
def plotit_pyplot(
        y, x=None,
        names=None,
        lines=None, widths=None, colors=None, opacities=None,
        x_label=None,   y_label=None,
        x_scale='linear',   y_scale='linear',
        x_lims: list = None, y_lims: list = None,
        title: str = None,  
        ax=None,
        grid=True,
        ):
    """Рисование графиков

    Args:
        y (_type_): ординаты
        x (_type_, optional): абсциссы. Defaults to None.
        names (list, optional): имена кривых для легенды. Defaults to None.
        lines (list, optional): типы линий. Defaults to None.
        widths (list, optional): толщины линий. Defaults to None.
        colors (list, optional): цвета линий (целые значения - порядковые номера). Defaults to None.
        opacities (list, optional): прозрачности. Defaults to None.
        y_scale (str, optional): тип оси (линейная, логарифм и проч.). Defaults to None.
        y_lims (list, optional): границы по ординатам. Defaults to None.
        x_lims (list, optional): границы по абсциссам. Defaults to None.
        y_label (str, optional): название оси ординат. Defaults to None.
        x_label (str, optional): название оси абсцисс. Defaults to None.
        title (str, optional): заголовок графика. Defaults to None.
        grid (bool, optional): рисовать сетку. Defaults to True.
        ax (_type_, optional): ось на которую добаляется график. Defaults to None.

    Returns:
        _type_: _description_


    Example 1. Plot single ndarray:
        >>> import numpy as np
        >>> from plotit import plotit_pyplot as plot
        >>> y = np.arange(10)**2
        >>> plot(y)

    Example 2. Plot list:
        >>> import numpy as np
        >>> from plotit import plotit_pyplot as plot
        >>> y = [np.arange(10)**2, np.arange(10)]
        >>> plot(y, names=['quad', 'lin'])
    """
    if not isinstance(y,(list,tuple)): 
        y = [y]
        if isinstance(names,str): names = [names]
    if names is None:       names=[None]*len(y)
    if lines is None:       lines = ['solid'] * len(y)
    if widths is None:      widths = [1.5] * len(y)
    if opacities is None:	opacities = [1.0] * len(y)
    
    if colors is None:
        colors = [None] * len(y)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    color_cycle = prop_cycle.by_key()['color']
    
    if not isinstance(x,(list,tuple)):
        if x is None:   x = [np.arange(_.shape[-1]) for _ in y]
        else:		   x = [x]*len(y)
    if ax is None:		figure, ax = plt.subplots()
    else: figure = None
    for i in range(len(y)):
        color = color_cycle[colors[i] % len(color_cycle)] if colors[i] is not None else None
        ax.plot(x[i], y[i], label=names[i], color=color,
                linestyle=lines[i], linewidth=widths[i], alpha=opacities[i])
    # if names[0] is not None: ax.legend()
    if names[0] is not None:
        # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.legend(loc='upper right', framealpha=0.5)  # framealpha sets transparency (0 = fully transparent, 1 = fully opaque)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    ax.set_title(title)
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.grid(grid)
    # if matplotlib.get_backend() == "Qt5Agg":
    if 'Qt'.lower() in matplotlib.get_backend().lower():
        ax.figure.canvas.manager.window.raise_()  # Only runs if Qt5Agg is used
    # ax.figure.tight_layout()
    if figure is not None: 	return figure, ax


# Анимация
def run_animation(y, x=None, save_file=None, x_label='x', y_label='y', titles=None):
    """_summary_

    Args:
        y (_type_): _description_
        x (_type_, optional): _description_. Defaults to None.
        save_file (_type_, optional): _description_. Defaults to None.
        x_label (str, optional): _description_. Defaults to 'x'.
        y_label (str, optional): _description_. Defaults to 'y'.
        titles (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_

    Example 1. Animate list of single plots
        >>> import numpy as np
        >>> import matplotlib
        >>> matplotlib.use('Qt5Agg')
        >>> from matplotlib import pyplot as plt
        >>> plt.ion()
        >>> from plotit import run_animation as animate
        >>> t = np.linspace(0,1,100)
        >>> phis = np.linspace(0,2*np.pi,10)
        >>> y = [np.sin(10*t+phi) for phi in phis]
        >>> animate(y, t, titles=[f'φ={phi:0.1f} rad' for phi in phis])

    """
    
    # Check if the current backend is non-interactive and switch if needed
    if matplotlib.get_backend() not in ['Qt4Agg', 'Qt5Agg', 'TkAgg']: raise RuntimeError(f"Only 'Qt4Agg', 'Qt5Agg', 'TkAgg' are supported, but {matplotlib.get_backend()} is used. Try running matplotlib.use('Qt5Agg').")
    if isinstance(y, (list, tuple)): 
        y = np.stack(y, axis=0)
        
    fig, ax = plt.subplots()
    ax.set(xlabel=x_label, ylabel=y_label, xlim=(np.min(x), np.max(x)), ylim=(1.1 * np.min(y), 1.1 * np.max(y)))
    lines = []
    if x is None: x = np.arange(y[0].shape[-1])

    if y.ndim == 3:  # If y is a 3D array, create multiple lines
        for i in range(y.shape[0]):
            line, = ax.plot([], [], lw=1.5)
            lines.append(line)
    else:  # If y is a 2D array, create a single line
        line, = ax.plot([], [], lw=1.5)
        lines.append(line)

    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def animate(i):
        # Update each line's data based on the frame index i
        if y.ndim == 3:  # Animate multiple lines
            for j, line in enumerate(lines):
                line.set_data(x, y[j, i, :])
        else:  # Animate a single line
            lines[0].set_data(x, y[i, :])
        
        # Set the title for the current frame, if titles are provided
        if titles is not None and i < len(titles):
            ax.set_title(titles[i])
        return lines

    blit = titles is None   # if no titles than redraw only lines
    anim = FuncAnimation(fig, animate, init_func=init, frames=y.shape[-2], interval=100, blit=blit, repeat=True, repeat_delay=100)
    if save_file is not None:
        anim.save(save_file, writer='PillowWriter', dpi=96, fps=12, bitrate=256)

    plt.show()
    return anim


# Отображение созвездия
def plot_constellation(complex_values, names=None, ax=None):
    """
    Plots a constellation diagram from an array of complex values.
    Parameters:
        complex_values (array-like): An array of complex numbers representing the signal points.
    """
    # Extract real and imaginary parts
    real_parts = np.real(complex_values)
    imaginary_parts = np.imag(complex_values)

    # Create a scatter plot using matplotlib
    if ax is None:
        _, ax = plt.subplots()

    scatter = ax.scatter(real_parts, imaginary_parts, s=8, label=names, alpha=0.75)

    # Optional: label points with complex values
    # for i, val in enumerate(complex_values):
        # ax.text(real_parts[i], imaginary_parts[i], f'{val:.2f}', fontsize=10)

    # Add axis labels and title
    ax.set_title('Созвездие')
    ax.set_xlabel('Действ.')
    ax.set_ylabel('Мним.')
    ax.grid(True)

    lim = 1.1*np.max(np.abs(complex_values))
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])

    if names is not None:
        ax.legend()

    plt.gca().set_aspect('equal', adjustable='box')
    
    if names is not None:
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1))

    # ax.figure.tight_layout()
    # ax.figure.canvas.manager.window.raise_()
    if ax is None:
        plt.show()
    elif isinstance(ax, list):
        if len(ax) == 0:
            ax.append(ax)
        else:
            ax[-1] = ax

