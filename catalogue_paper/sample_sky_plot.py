"""
Show the distribution of galaxies on sky.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()


def sky_plot(
    cat: Table,
    idx_arr: ArrayLike,
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
        ``"RA"``.
    dec_col : str, optional
        The name of the column containing the declination, by default
        ``"DEC"``.
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
        fig, ax = plt.subplots(figsize=(aanda_columnwidth, aanda_columnwidth / 1.62))
    else:
        fig = ax.get_figure()

    plot_cat = cat[idx_arr]

    sc = ax.scatter(
        plot_cat[ra_col],
        plot_cat[dec_col],
        s=s,
        transform=ax.get_transform("world"),
        **kwargs,
    )
    return fig, ax, sc


if __name__ == "__main__":

    catalogue_dir = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "glass_niriss"
        / "match_catalogues"
        / "classification_v1"
    )

    v2_out_dir = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "classification-stage-2"
        / "catalogues"
        / "compiled"
    )

    new_cat_name = "internal_full_1.fits"
    v2_cat = Table.read(v2_out_dir / new_cat_name)

    img_path = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Prep"
        / "glass-a2744-ir_drc_sci.fits"
    )

    v1_cat = Table.read(catalogue_dir / "compiled_catalogue_v1.fits")

    with fits.open(img_path) as img_hdul:
        img_wcs = WCS(img_hdul[0])

        fig, ax = plt.subplots(
            figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth),
            constrained_layout=True,
            subplot_kw={"projection": img_wcs},
        )
        ax.imshow(
            img_hdul[0].data,
            norm=astrovis.ImageNormalize(
                img_hdul[0].data,
                # interval=astrovis.PercentileInterval(99.9),
                interval=astrovis.ManualInterval(-0.001, 10),
                stretch=astrovis.LogStretch(),
            ),
            cmap="binary",
            origin="lower",
        )
    # sky = partial(sky.plot_hist, bins=np.arange(16.5,33.5,0.5), ax=ax)
    # print (np.nanmin(v1_cat["MAG_AUTO"]), np.nanmax(v1_cat["MAG_AUTO"])

    # sky_plot(v1_cat, v1_cat["V1_CLASS"] == 0, ax=ax, label="Full Sample")
    # sky_plot(
    #     v1_cat,
    #     (v1_cat["V1_CLASS"] > 0) & (v1_cat["V1_CLASS"] < 4),
    #     ax=ax,
    #     label="Extracted",
    # )
    # sky_plot(v1_cat, (v1_cat["V1_CLASS"] == 4), ax=ax, label="First Pass")
    # sky_plot(v1_cat, v1_cat["V1_CLASS"] >= 5, ax=ax, label="Placeholder")

    # sky_plot(
    #     v2_cat,
    #     (v2_cat["Z_FLAG_ALL"] >= 9) & (~np.isfinite(v2_cat["zspec"])),
    #     ax=ax,
    #     label="Secure",
    # )

    for i, (label, z_range, icon, size) in enumerate(
        zip(
            ["1.1", "1.34", "1.9"],
            [[1.09, 1.11], [1.32, 1.37], [1.85, 1.9]],
            ["x", "*", "o"],
            [15, 30, 15],
        )
    ):
        sky_plot(
            v2_cat,
            (v2_cat["Z_FLAG_ALL"] >= 9)
            # & (~np.isfinite(v2_cat["zspec"]))
            & (v2_cat["Z_EST_ALL"] >= z_range[0]) & (v2_cat["Z_EST_ALL"] < z_range[-1]),
            ax=ax,
            label=label,
            s=size,
            marker=icon,
            edgecolor="none",
        )

    ax.set_xlabel(r"R.A.")
    ax.set_ylabel(r"Dec.")
    ax.legend()

    # plt.savefig(save_dir / "sample_sky_plot.pdf", dpi=600)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "sample_sky_plot.pdf", dpi=600)
    plt.savefig(save_dir / "svg" / "sample_sky_plot.svg", dpi=600)

    # plt.show()

    z_range = [1.32, 1.37]
    # z_range = [1.85,1.9]

    for r in v2_cat["ID", "RA", "DEC"][
        (v2_cat["Z_FLAG_ALL"] >= 9)  # & (~np.isfinite(v2_cat["zspec"])
        & (v2_cat["Z_EST_ALL"] >= z_range[0])
        & (v2_cat["Z_EST_ALL"] < z_range[-1])
    ]:
        print(r[0], r[1], r[2])
