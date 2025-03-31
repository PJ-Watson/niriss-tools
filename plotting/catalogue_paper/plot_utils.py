"""Generic plotting functions."""

import astropy.visualization as astrovis
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from numpy.typing import ArrayLike

aanda_columnwidth = 256.0748 / 72.27
aanda_textwidth = 523.5307 / 72.27


def sky_plot(
    cat: Table,
    idx_arr: ArrayLike | None = None,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    ax: plt.Axes | None = None,
    s: int = 5,
    **kwargs,
):
    """
    Plot points on sky, using world coordinates.

    Parameters
    ----------
    cat : Table
        The catalogue from which to draw points.
    idx_arr : ArrayLike
        A boolean array selecting points from the catalogue.
    ra_col : str, optional
        The name of the column containing the right ascension, by default
        ``"ra"``.
    dec_col : str, optional
        The name of the column containing the declination, by default
        ``"dec"``.
    ax : plt.Axes | None, optional
        The axes on which the galaxies will be plotted, by default
        ``None``.
    s : int, optional
        The size of the markers, by default 5.
    **kwargs : dict, optional
        Any additional keywords to pass through to `~plt.Axes.scatter`.

    Returns
    -------
    (`~plt.Figure`, `~plt.Axes`, `~matplotlib.collections.PathCollection`)
        The figure, axes, and plotted points.
    """

    if ax is None:
        print("Creating figure")
        fig, ax = plt.subplots(figsize=(aanda_columnwidth, aanda_columnwidth / 1.62))
    else:
        fig = ax.get_figure()

    if idx_arr is not None:
        plot_cat = cat[idx_arr]
    else:
        plot_cat = cat

    print(len(plot_cat[ra_col]))
    print(np.nansum(np.isfinite(plot_cat[ra_col])))

    sc = ax.scatter(
        plot_cat[ra_col],
        plot_cat[dec_col],
        s=s,
        transform=ax.get_transform("world"),
        **kwargs,
    )
    return fig, ax, sc


def setup_aanda_style():
    """
    A helper function to setup the A&A style.
    """
    rc_fonts = {
        "font.family": "serif",
        "font.size": 7,
        "figure.figsize": (aanda_columnwidth, 3),
        "text.usetex": True,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.labelsize": 7,
        "xtick.top": True,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.labelsize": 7,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "lines.linewidth": 1,
        "text.latex.preamble": (
            r"""
        \usepackage{amsmath}
        \usepackage{txfonts}
        %
        \DeclareMathAlphabet{\mathsc}{OT1}{cmr}{m}{sc}
        \def\testbx{bx}%
        \DeclareRobustCommand{\ion}[2]{%
        \relax\ifmmode
        \ifx\testbx\f@series
        {\mathbf{#1\,\mathsc{#2}}}\else
        {\mathrm{#1\,\mathsc{#2}}}\fi
        \else\textup{#1\,{\mdseries\textsc{#2}}}%
        \fi}
        %
        """
        ),
    }
    mpl.rcParams.update(rc_fonts)

    return


def plot_default(
    img,
    axs=None,
    vmin=None,
    vmax=None,
    percentile=99,
    stretch="log",
    cmap="plasma",
    contours=None,
    linthresh=1e-6,
    **kwargs,
):
    """
    A generic function to plot image data in the correct orientation and scale.

    Parameters
    ----------
    img : _type_
        _description_.
    axs : _type_, optional
        _description_, by default None.
    vmin : _type_, optional
        _description_, by default None.
    vmax : _type_, optional
        _description_, by default None.
    percentile : int, optional
        _description_, by default 99.
    stretch : str, optional
        _description_, by default "linear".
    cmap : str, optional
        _description_, by default "plasma".
    contours : _type_, optional
        _description_, by default None.
    linthresh : _type_, optional
        _description_, by default 1e-6.
    **kwargs : dict, optional
        Keyword arguments are passed through to `~matplotlib.axes.Axes.imshow`.

    Returns
    -------
    _type_
        _description_.
    """
    if axs == None:
        fig, axs = plt.subplots()

    if stretch == "symlog":

        im = axs.imshow(
            img,
            origin="lower",
            norm=colors.SymLogNorm(linthresh, vmin=vmin, vmax=vmax),
            cmap=cmap,
            **kwargs,
        )
    else:

        match stretch:
            case "log":
                plot_stretch = astrovis.LogStretch()
            case "sqrt":
                plot_stretch = astrovis.SqrtStretch()
            case "linear":
                plot_stretch = astrovis.LinearStretch()
            case _:
                plot_stretch = stretch

        if (vmin is not None) or (vmax is not None):
            norm = astrovis.ImageNormalize(
                img,
                interval=astrovis.ManualInterval(
                    vmin=vmin,
                    vmax=vmax,
                ),
                stretch=plot_stretch,
            )
        else:
            if np.nanmax(img) == 1 and np.nanmin(img) == 0:
                norm = astrovis.ImageNormalize(
                    img,
                    interval=astrovis.ManualInterval(vmin=0, vmax=1),
                    stretch=plot_stretch,
                )
            else:
                norm = astrovis.ImageNormalize(
                    img,
                    interval=astrovis.PercentileInterval(percentile=percentile),
                    stretch=plot_stretch,
                )
        im = axs.imshow(img, origin="lower", norm=norm, cmap=cmap, **kwargs)

    im.format_cursor_data = format_cursor_data

    if contours is not None:
        axs.contour(img, levels=contours)

    return axs, im


def format_cursor_data(data):
    """
    A workaround to prevent character overflow in some plotted data.

    Parameters
    ----------
    data : _type_
        _description_.

    Returns
    -------
    _type_
        _description_.
    """
    return "[" + str(data) + "]"


def plot_hist(
    data: ArrayLike,
    ax: plt.Axes | None = None,
    alpha: float = 1.0,
    bins: int = 20,
    **kwargs,
):
    """
    Generate a histogram with some useful defaults.

    Parameters
    ----------
    data : ArrayLike
        The data to be plotted.
    ax : plt.Axes | None, optional
        The axes on which the histogram will be plotted, by default
        ``None``.
    alpha : float, optional
        The transparency of the histogram, by default 1.0.
    bins : int, optional
        The number of bins to divide the data into, by default 20.
    **kwargs : dict, optional
        Any additional keyword arguments to pass through to `
        `~plt.axes.hist``.

    Returns
    -------
    (`~plt.Figure`, `~plt.Axes`, `~matplotlib.collections.PathCollection`)
        The figure, axes, and plotted histogram.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(aanda_columnwidth, aanda_columnwidth / 1.62))
    else:
        fig = ax.get_figure()

    hist = ax.hist(data, alpha=alpha, bins=bins, **kwargs)

    return fig, ax, hist


def table_to_array(table: Table) -> ArrayLike:
    """
    Convert an astropy table to a numpy array.

    Parameters
    ----------
    table : Table
        Original astropy table.

    Returns
    -------
    ArrayLike
        Numpy array.
    """
    return np.lib.recfunctions.structured_to_unstructured(table.as_array())
