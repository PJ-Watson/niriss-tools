"""
A module containing tools relating to the construction of PSFs.
"""

import matplotlib.pyplot as plt
import numpy as np
from astropy.nddata import Cutout2D, NDData
from astropy.stats import mad_std
from numpy.typing import ArrayLike
from photutils.aperture import CircularAperture, aperture_photometry
from photutils.centroids import centroid_com
from photutils.detection import find_peaks

__all__ = ["measure_curve_of_growth", "radial_apertures", "find_stars"]


def measure_curve_of_growth(
    image,
    position=None,
    radii=None,
    rnorm="auto",
    nradii=30,
    verbose=False,
    showme=False,
    rbg="auto",
):
    """
    Measure a curve of growth from cumulative circular aperture photometry on a list of radii centered on the center of mass of a source in a 2D image.

    Parameters
    ----------
    image : `~numpy.ndarray`
        2D image array.
    position : `~astropy.coordinates.SkyCoord`
        Position of the source.
    radii : `~astropy.units.Quantity` array
        Array of aperture radii.

    Returns
    -------
    `~astropy.table.Table`
        Table of photometry results, with columns 'aperture_radius' and 'aperture_flux'.
    """

    if type(radii) is type(None):
        radii = powspace(0.5, image.shape[1] / 2, nradii)

    # Calculate the centroid of the source in the image
    if type(position) is type(None):
        #        x0, y0 = centroid_2dg(image)
        position = centroid_com(image)

    if rnorm == "auto":
        rnorm = image.shape[1] / 2.0
    if rbg == "auto":
        rbg = image.shape[1] / 2.0

    apertures = [CircularAperture(position, r=r) for r in radii]

    if rbg:
        bg_mask = apertures[-1].to_mask().to_image(image.shape) == 0
        bg = np.nanmedian(image[bg_mask])
        if verbose:
            print("background", bg)
    else:
        bg = 0.0

    # Perform aperture photometry for each aperture
    phot_table = aperture_photometry(image - bg, apertures)
    # Calculate cumulative aperture fluxes
    cog = np.array([phot_table["aperture_sum_" + str(i)][0] for i in range(len(radii))])

    if rnorm:
        rnorm_indx = np.searchsorted(radii, rnorm)
        cog /= cog[rnorm_indx]

    area = np.pi * radii**2
    area_cog = np.insert(np.diff(area), 0, area[0])
    profile = np.insert(np.diff(cog), 0, cog[0]) / area_cog
    profile /= profile.max()

    if showme:
        plt.plot(radii, cog, marker="o")
        plt.plot(radii, profile / profile.max())
        plt.xlabel("radius pix")
        plt.ylabel("curve of growth")

    # Create output table of aperture radii and cumulative fluxes
    return radii, cog, profile


def radial_apertures(
    image: ArrayLike,
    positions: ArrayLike,
    radii: ArrayLike = [2, 4],
    # radii: ArrayLike = [4, 8],
):
    print(positions)
    apertures = [CircularAperture(positions=positions, r=r_i) for r_i in radii]
    phot_table = aperture_photometry(image, apertures)

    plt.scatter(
        -2.5 * np.log10(phot_table["aperture_sum_1"]),
        phot_table["aperture_sum_1"] / phot_table["aperture_sum_0"],
    )
    plt.show()


def find_stars(
    image: ArrayLike,
    background_rms: ArrayLike | None = None,
    threshold: float = 10.0,
    cutout_size: int = 10,
    radii: ArrayLike = [2, 4],
    zeropoint: float = 28.9,
):

    if background_rms is not None:
        threshold_level = background_rms * threshold
    else:
        threshold_level = threshold * mad_std(image)
    peak_table = find_peaks(
        data=image, threshold=threshold_level, npeaks=1000, centroid_func=centroid_com
    )

    print(peak_table)
    peak_table["x0"] = 0.0
    peak_table["y0"] = 0.0
    peak_table["minv"] = 0.0
    for ir in np.arange(len(radii)):
        peak_table["r" + str(ir)] = 0.0
    for ir in np.arange(len(radii)):
        peak_table["p" + str(ir)] = 0.0

    stars = []
    for ip, p in enumerate(peak_table):
        co = Cutout2D(
            image, (p["x_centroid"], p["y_centroid"]), cutout_size, mode="partial"
        )
        # measure offset, feed it to measure cog
        # if offset > 3 pixels -> skip
        position = centroid_com(co.data)
        peak_table["x0"][ip] = position[0] - cutout_size // 2
        peak_table["y0"][ip] = position[1] - cutout_size // 2
        peak_table["minv"][ip] = np.nanmin(co.data)
        _, cog, profile = measure_curve_of_growth(
            co.data, radii=np.array(radii), position=position, rnorm=None, rbg=None
        )
        for ir in np.arange(len(radii)):
            peak_table["r" + str(ir)][ip] = cog[ir]
        for ir in np.arange(len(radii)):
            peak_table["p" + str(ir)][ip] = profile[ir]
        co.radii = np.array(radii)
        co.cog = cog
        co.profile = profile
        stars.append(co)

    stars = np.array(stars)

    peak_table["mag"] = 28.9 - 2.5 * np.log10(peak_table["r1"])
    shift_lim = 2
    r = peak_table["r1"] / peak_table["r0"]
    shift_lim_root = np.sqrt(shift_lim)

    ok_mag = peak_table["mag"] < 28
    ok_min = peak_table["minv"] > -0.5
    ok_phot = (
        np.isfinite(peak_table["r" + str(len(radii) - 1)])
        & np.isfinite(peak_table["r1"])
        & np.isfinite(peak_table["p1"])
    )
    ok_shift = (
        (np.sqrt(peak_table["x0"] ** 2 + peak_table["y0"] ** 2) < shift_lim)
        & (np.abs(peak_table["x0"]) < shift_lim_root)
        & (np.abs(peak_table["y0"]) < shift_lim_root)
    )

    # r = peaks['r4']/peaks['r2']
    # shift_lim_root = np.sqrt(shift_lim)
    # ratio apertures @@@ hardcoded
    threshold_mode = [-0.2, 0.2]
    range = [0, 4]
    h = np.histogram(
        r[(r < 1.5) & ok_mag],
        bins=np.arange(0, range[1], threshold_mode[1] / 2.0),
        range=range,
    )
    ih = np.argmax(h[0])
    rmode = h[1][ih]
    print(rmode)
    ok_mode = ((r / rmode - 1) > threshold_mode[0]) & (
        (r / rmode - 1) < threshold_mode[1]
    )

    ok = ok_phot & ok_mode & ok_min & ok_shift & ok_mag
    # peak_table = peak_table[ok_shift & ok_mag & ok_min]
    peak_table = peak_table[ok]

    # plt.scatter(peak_table["mag"],peak_table["r1"]/peak_table["r0"])
    # plt.show()

    radial_apertures(
        image, np.array([peak_table["x_centroid"], peak_table["y_centroid"]]).T
    )

    from astropy.visualization import simple_norm
    from astropy.visualization.mpl_normalize import ImageNormalize
    from photutils.aperture import CircularAperture

    positions = np.transpose((peak_table["x_peak"], peak_table["y_peak"]))

    apertures = CircularAperture(positions, r=5.0)

    norm = simple_norm(image, "sqrt", percent=99.9)

    plt.imshow(
        image, cmap="Greys_r", origin="lower", norm=norm, interpolation="nearest"
    )

    apertures.plot(color="#0547f9", lw=1.5)

    plt.xlim(0, image.shape[1] - 1)

    plt.ylim(0, image.shape[0] - 1)

    plt.show()

    peak_table.rename_columns(["x_centroid", "y_centroid"], ["x", "y"])

    from photutils.psf import extract_stars

    data = NDData(image)
    stars = extract_stars(data, peak_table, size=51)
    # import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm

    nrows = 5
    ncols = 5
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20, 20), squeeze=True)
    ax = ax.ravel()
    for i in np.arange(nrows * ncols):
        norm = simple_norm(stars[i], "log", percent=99.0)
        ax[i].imshow(stars[i], norm=norm, origin="lower", cmap="viridis")
    plt.show()
    return
