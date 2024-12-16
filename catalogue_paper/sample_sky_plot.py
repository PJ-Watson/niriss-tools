import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()


def sky_plot(
    cat: Table,
    idx_arr: ArrayLike,
    ra_col: str = "RA",
    dec_col: str = "DEC",
    ax=None,
    s=2,
    **kwargs,
):
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
                interval=astrovis.PercentileInterval(99.9),
                stretch=astrovis.LogStretch(),
            ),
            cmap="binary",
            origin="lower",
        )
    # sky = partial(sky.plot_hist, bins=np.arange(16.5,33.5,0.5), ax=ax)
    # print (np.nanmin(v1_cat["MAG_AUTO"]), np.nanmax(v1_cat["MAG_AUTO"])

    sky_plot(v1_cat, v1_cat["V1_CLASS"] == 0, ax=ax, label="Full Sample")
    sky_plot(
        v1_cat,
        (v1_cat["V1_CLASS"] > 0) & (v1_cat["V1_CLASS"] < 4),
        ax=ax,
        label="Extracted",
    )
    sky_plot(v1_cat, (v1_cat["V1_CLASS"] == 4), ax=ax, label="First Pass")
    sky_plot(v1_cat, v1_cat["V1_CLASS"] >= 5, ax=ax, label="Placeholder")

    ax.set_xlabel(r"R.A.")
    ax.set_ylabel(r"Dec.")
    ax.legend()

    plt.savefig(save_dir / "sample_sky_plot.pdf")

    plt.show()
