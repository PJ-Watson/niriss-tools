"""
Functions to bin data to a specified signal/noise.
"""

import astropy.visualization as astrovis
import matplotlib.pyplot as plt
import numpy as np
from vorbin.voronoi_2d_binning import voronoi_2d_binning

__all__ = ["hexbin", "constrained_adaptive"]


def hexbin(
    x: ArrayLike,
    y: ArrayLike,
    signal: ArrayLike,
    noise: ArrayLike,
    bin_diameter: float = 10,
    centre: tuple[float, float] | None = None,
) -> tuple[ArrayLike, int, ArrayLike, ArrayLike]:
    """
    Hexagonally bin data, returning the binned S/N.

    Parameters
    ----------
    x : ArrayLike
        A 1D array containing the x-positions of the data.
    y : ArrayLike
        A 1D array containing the y-positions of the data.
    signal : ArrayLike
        A 1D array of the signal.
    noise : ArrayLike
        A 1D array of the noise. All of ``signal``, ``noise``, ``x``, and
        ``y`` must have the same shape.
    bin_diameter : float, optional
        The minimum diameter of each bin (subject to pixel discretisation
        effects), i.e. the long diameter of each hexagon. By default 10.
    centre : tuple[float, float] | None, optional
        If given, all bins will be aligned such that a hexagon is centred
        on these coordinates. If not given, the highest S/N pixel will be
        used instead.

    Returns
    -------
    bin_labels : ArrayLike
        The bin label assigned to each element of the input arrays.
    nbins : int
        The number of bins.
    binned_s_n : ArrayLike
        The S/N in each bin.
    bin_inv : ArrayLike
        A 1D array performing the inverse binning operation, i.e.
        ``binned_s_n[inv]`` gives an array of the same shape as ``x``.
    """

    if centre is None:
        centre_idx = np.argmax(signal / noise)
        centre = np.array([x[centre_idx], y[centre_idx]])

    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)

    x = np.asarray(x, float)
    y = np.asarray(y, float)

    sx = bin_diameter * 0.5 * np.sqrt(3)
    sy = bin_diameter * 3 / 2

    ix = ((x - xmin) + ((centre[0] - xmin) % sx)) / sx - 0.5
    iy = ((y - ymin) + ((centre[1] - ymin) % sy)) / sy

    nx = (np.max(ix) - np.min(ix)) + 1
    ny = (np.max(iy) - np.min(iy)) + 1
    nx1 = nx + 1
    ny1 = ny + 1
    nx2 = nx
    ny2 = ny

    ix1 = np.round(ix).astype(int)
    iy1 = np.round(iy).astype(int)
    ix2 = np.floor(ix).astype(int)
    iy2 = np.floor(iy).astype(int)

    # flat indices, plus one so that out-of-range points go to position 0.
    i1 = np.where(
        (0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1), ix1 * ny1 + iy1 + 1, 0
    )
    i2 = np.where(
        (0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2), ix2 * ny2 + iy2 + 1, 0
    )

    d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
    d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
    bdist = d1 < d2

    idxs = bdist * i1 + (~bdist * (i2 + np.nanmax(i1)))
    u, bin_inv = np.unique(idxs, return_inverse=True)
    nbins = len(u)
    bin_labels = np.arange(len(u), dtype=int)[bin_inv]

    binned_signal = np.bincount(bin_labels, weights=signal)
    binned_noise = np.sqrt(np.bincount(bin_labels, weights=noise**2))

    binned_s_n = binned_signal / binned_noise

    return bin_labels, nbins, binned_s_n, bin_inv


def constrained_adaptive(
    signal: ArrayLike,
    noise: ArrayLike,
    bin_diameter: float = 10,
    target_sn: float = 100,
    plot: bool = True,
    quiet: bool = False,
    cvt: bool = True,
    wvt: bool = True,
) -> tuple[ArrayLike, int, ArrayLike, ArrayLike]:
    """
    Perform a constrained adaptive binning method to reach a target S/N.

    This function performs a 2-stage binning procedure, subject to a
    minimum bin diameter. The initial stage is a hexagonal binning scheme,
    which selects any bins that achieve the target S/N. This is followed
    by a Voronoi tesselation on all remaining pixels, using the method of
    `Cappellari & Copin 2003\
<https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C>`_.

    Parameters
    ----------
    signal : ArrayLike
        A 2D array containing the signal to be binned.
    noise : ArrayLike
        A 2D array containing the associated noise.
    bin_diameter : float, optional
        The minimum diameter of each bin (subject to pixel discretisation
        effects), by default 10.
    target_sn : float, optional
        The desired S/N in each output bin, by default 100. This is only
        guaranteed to be achieved for the hexagonal bins, as there is some
        scatter on the S/N achieved through Voronoi binning.
    plot : bool, optional
        Show a comparison of the binned and unbinned data, alongside the
        S/N and bin maps. By default ``False``.
    quiet : bool, optional
        Print the output of the Voronoi binning procedure, by default
        ``False``.
    cvt : bool, optional
        Attempt to achieve a centroidal Voronoi tessellation, using the
        method of `Cappellari & Copin 2003 \
            <https://ui.adsabs.harvard.edu/abs/2003MNRAS.342..345C>`_,
        by default ``True``.
    wvt : bool, optional
        Use the weighted Voronoi tessellation method by
        `Diehl & Statler 2006 \
            <https://ui.adsabs.harvard.edu/abs/2006MNRAS.368..497D>`_.
        By default True.

    Returns
    -------
    bin_labels : ArrayLike
        A 2D ``int`` array, containing the bin label assigned to each
        element of the input arrays.
    nbins : int
        The number of bins.
    binned_s_n : ArrayLike
        A 2D array of the same shape as ``signal``, containing the S/N in
        each bin.
    bin_inv : ArrayLike
        A 2D array performing the inverse binning operation, i.e.
        ``binned_s_n[inv]`` gives an array of the same shape as ``x``.
    """

    Y, X = np.mgrid[0 : signal.shape[0], 0 : signal.shape[1]]

    f_signal = signal.ravel()
    f_noise = noise.ravel()

    hex_idxs, hex_bin_n, hex_s_n, hex_inv = hexbin(
        X.ravel(),
        Y.ravel(),
        f_signal,
        f_noise,
        bin_diameter=bin_diameter,
    )

    good_hex_data = (hex_s_n >= target_sn)[hex_inv]

    num_hex_bins = len(np.unique(hex_idxs[good_hex_data]))

    vorbin_idx = ~good_hex_data & np.isfinite(f_noise) & (f_noise > 0) & (f_signal > 0)

    vorbin_output = voronoi_2d_binning(
        x=X.ravel()[vorbin_idx],
        y=Y.ravel()[vorbin_idx],
        signal=f_signal[vorbin_idx],
        noise=f_noise[vorbin_idx],
        target_sn=target_sn,
        pixelsize=1,
        plot=False,
        quiet=quiet,
        cvt=cvt,
        wvt=wvt,
    )

    vor_idxs, vor_inv = np.unique(vorbin_output[0], return_inverse=True)
    vor_idxs += np.max(hex_idxs) + 1

    num_vor_bins = len(np.unique(vor_idxs))

    hex_idxs[vorbin_idx != ~good_hex_data] = -1
    hex_idxs[vorbin_idx] = vor_idxs[vor_inv]

    _all_unq, all_inv = np.unique(hex_idxs, return_inverse=True)
    nbins = len(_all_unq)
    all_idxs = np.arange(len(_all_unq))[all_inv]

    binned_signal = np.bincount(all_idxs, weights=f_signal)
    binned_noise = np.sqrt(np.bincount(all_idxs, weights=f_noise**2))

    binned_s_n = (binned_signal / binned_noise).reshape(signal.shape)
    bin_labels = all_idxs.reshape(signal.shape)
    bin_inv = all_inv.reshape(signal.shape)
    print(f"Total bins: {len(_all_unq)} (Hex: {num_hex_bins}, Voronoi: {num_vor_bins})")

    if plot:
        fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

        img_norm = astrovis.ImageNormalize(
            data=signal,
            interval=astrovis.PercentileInterval(99.9),
            stretch=astrovis.LogStretch(),
        )
        axs[0, 0].imshow(signal, norm=img_norm, origin="lower", cmap="plasma")
        bin_sig_plot = (
            np.bincount(all_idxs, weights=f_signal) / np.bincount(all_idxs)
        )[all_inv]
        bin_sig_plot[all_idxs == 0] = np.nan
        axs[0, 1].imshow(
            bin_sig_plot.reshape(signal.shape),
            origin="lower",
            norm=img_norm,
            cmap="plasma",
        )

        sn_plot = (
            np.bincount(all_idxs, weights=f_signal)
            / np.sqrt(np.bincount(all_idxs, weights=f_noise**2))
        )[all_inv]
        sn_plot[all_idxs == 0] = np.nan
        axs[1, 0].imshow(
            sn_plot.reshape(signal.shape),
            origin="lower",
            # norm=img_norm,
            cmap="plasma",
        )

        rng = np.random.default_rng()
        bin_plot = rng.random(size=len(_all_unq))[all_inv]
        bin_plot[all_idxs == 0] = np.nan

        axs[1, 1].imshow(
            bin_plot.reshape(signal.shape),
            origin="lower",
            # norm=img_norm,
            cmap="jet",
            interpolation="none",
        )

        for a in axs.flatten():
            a.set_facecolor("k")

        # plt.imshow(hex_idxs.reshape(signal.shape), origin="lower")

        plt.show()

    return bin_labels, nbins, binned_s_n, bin_inv
