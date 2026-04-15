import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import FigureCanvasBase
from matplotlib.rcsetup import validate_bool
import importlib

def print_ipe(self, filename, *args, **kwargs):
    backend_ipe = importlib.import_module(f"{__package__}.ipe-tools.matplotlib.backend_ipe")
    canvas = backend_ipe.FigureCanvasIpe(self.figure)
    return canvas.print_ipe(filename, *args, **kwargs)

def use_ipe():
    FigureCanvasBase.print_ipe = print_ipe
    FigureCanvasBase.filetypes["ipe"] = "Ipe 7 file format"

def use_science_style():
    importlib.import_module("scienceplots")
    plt.style.use(['science', 'no-latex'])
    params = {
            'figure.dpi': 250,
            'legend.borderaxespad': 0.,
            'patch.linewidth': .4,
            'lines.linewidth': .4,
            'hatch.linewidth': .4,
            'grid.linewidth': 0.4,
            'axes.linewidth': 0.4,
            'legend.edgecolor': '0.',
            'legend.fancybox': False,
            'legend.framealpha': 1.,
            'legend.frameon': True,
            'font.size': 6,
            'axes.titlesize': 7,
            'axes.labelsize': 5,
            'legend.fontsize': 6,
            'legend.title_fontsize': 7,
            'xtick.labelsize': 5,
            'ytick.labelsize': 5,
            'xtick.minor.width': 0.4,
            'xtick.major.width': 0.4,
            'ytick.minor.width': 0.4,
            'ytick.major.width': 0.4,
            'axes.formatter.use_mathtext': False,
            'ipe.textsize': True,
            }
    if 'ipe.textsize' not in plt.rcParams:
        plt.rcParams.validate['ipe.textsize'] = validate_bool
        
    plt.rcParams.update(params)


def cascade_plot(data, x=None, angle=10, spacing=1.0, scale=1.0,
                 ax=None, **kwargs):
    """
    Cascade (waterfall) plot.

    Parameters
    ----------
    data : np.ndarray or list of 1D arrays
        Shape (n_lines, n_points) or list of equal-length arrays.
    x : array-like, optional
        X values. Defaults to np.arange(n_points).
    angle : float
        Angle in degrees controlling horizontal shear.
    spacing : float
        Vertical offset between lines (along y-axis).
    scale : float
        Scales amplitude of each trace (around its baseline).
    ax : matplotlib.axes.Axes, optional
        Axis to plot on.
    **kwargs :
        Passed directly to ax.plot().

    Returns
    -------
    ax : matplotlib.axes.Axes
    """

    # Normalize input
    if isinstance(data, list):
        data = np.array(data)
    data = np.asarray(data)

    if data.ndim != 2:
        raise ValueError("data must be 2D or list of 1D arrays")

    n_lines, n_points = data.shape

    if x is None:
        x = np.arange(n_points)
    x = np.asarray(x)

    if x.shape[0] != n_points:
        raise ValueError("x must match data length")

    if ax is None:
        fig, ax = plt.subplots()

    # Horizontal shear factor from angle
    theta = np.deg2rad(angle)
    shear = np.tan(theta)

    for i in range(n_lines):
        y_offset = i * spacing

        # Scale around baseline (0 before offset)
        y = data[i] * scale + y_offset

        # Shear in x proportional to vertical offset
        x_i = x + shear * y_offset

        line, = ax.plot(x_i, y, **kwargs)

    ax.set_xlabel("X")
    ax.set_ylabel("Value")
    ax.set_title("Cascade Plot")

    return ax
