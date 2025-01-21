"""Generic plotting functions."""

import astropy.visualization as astrovis
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

aanda_columnwidth = 256.0748 / 72.27
aanda_textwidth = 523.5307 / 72.27


def setup_aanda_style():
    """
    A helper function to setup the A&A style.
    """
    rc_fonts = {
        "font.family": "serif",
        "font.size": 9,
        "figure.figsize": (aanda_columnwidth, 3),
        "text.usetex": True,
        "ytick.right": True,
        "ytick.direction": "in",
        "ytick.minor.visible": True,
        "ytick.labelsize": 8,
        "xtick.top": True,
        "xtick.direction": "in",
        "xtick.minor.visible": True,
        "xtick.labelsize": 8,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 9,
        "text.latex.preamble": (
            r"""
        \usepackage{txfonts}
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
    data,
    ax=None,
    alpha=1.0,
    bins=20,
    **kwargs,
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(aanda_columnwidth, aanda_columnwidth / 1.62))
    else:
        fig = ax.get_figure()

    hist = ax.hist(data, alpha=alpha, bins=bins, **kwargs)

    return fig, ax, hist
