"""
Merge existing redshift catalogues.
"""

from astropy.table import vstack
from default_imports import *

if __name__ == "__main__":

    prev_cat_dir = niriss_dir / "match_catalogues"

    sara_cat_path = prev_cat_dir / "hlsp_glass-jwst_jwst_nirspec_abell2744_v1_cat.fits"

    sara_cat = vstack([Table.read(sara_cat_path, 1), Table.read(sara_cat_path, 2)])
    sara_cat.rename_columns(sara_cat.colnames, [s.lower() for s in sara_cat.colnames])

    alt_cat = Table.read(prev_cat_dir / "ALT_DR1_public.fits")
    alt_cat = alt_cat[alt_cat["n_lines_detected"] >= 2]
    alt_cat.rename_columns(alt_cat.colnames, [s.lower() for s in alt_cat.colnames])

    uncover_cat = Table.read(prev_cat_dir / "uncover-msa-default_drz-DR4.1-zspec.fits")
    uncover_cat.rename_columns(
        uncover_cat.colnames, [s.lower() for s in uncover_cat.colnames]
    )

    img_path = grizli_dir / "Prep" / "glass-a2744-ir_drc_sci.fits"

    v2_cat = Table.read(
        grizli_dir
        / "classification-stage-2"
        / "catalogues"
        / "compiled"
        / "internal_full_1.fits"
    )

    with fits.open(img_path) as img_hdul:
        img_wcs = WCS(img_hdul[0])

        fig, ax = plt.subplots(
            figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth),
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
            cmap="binary_r",
            origin="lower",
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    # for i, (cat) in enumerate([sara_cat, uncover_cat, alt_cat]):
    #     plot_utils.sky_plot(cat, ax=ax, label="Placeholder")

    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)

    nirspec_cats = vstack(
        [sara_cat["ra", "dec", "z_spec"], uncover_cat["ra", "dec", "z_spec"]]
    )

    plot_kwargs = {"vmin": 0, "vmax": 10, "cmap": "rainbow"}
    # plot_utils.sky_plot(
    #     nirspec_cats, ax=ax, label="NIRSpec", c=nirspec_cats["z_spec"], **plot_kwargs
    # )
    # plot_utils.sky_plot(
    #     alt_cat, ax=ax, label="NIRCam WFSS", c=alt_cat["z_alt"], **plot_kwargs
    # )
    # _, _, sc = plot_utils.sky_plot(
    #     v2_cat[v2_cat["Z_FLAG_ALL"] >= 9],
    #     ax=ax,
    #     label="NIRISS",
    #     c=v2_cat["Z_EST_ALL"][v2_cat["Z_FLAG_ALL"] >= 9],
    #     **plot_kwargs,
    # )

    # ax.set_xlim(xlim)
    # ax.set_ylim(ylim)
    # ax.legend()
    # plt.colorbar(sc)
    # ax.set_facecolor("k")

    fig, ax = plt.subplots(
        figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth / 1.62),
        constrained_layout=True,
    )

    z_bins = np.arange(0, 10, 0.05)
    ax.hist(
        [
            v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["Z_EST_ALL"],
            nirspec_cats["z_spec"],
            alt_cat["z_alt"],
        ],
        stacked=True,
        bins=z_bins,
    )
    ax.set_xlim(z_bins[0], z_bins[-1])
    # ax_scatter.set_ylabel(r"$m_{\rm{F200W}}$")
    ax.set_xlabel(r"$z$")

    plt.show()

    fig, ax = plt.subplots(
        figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth / 1.62),
        constrained_layout=True,
        subplot_kw={"projection": "3d"},
    )

    # for cat in
    ax.scatter(
        v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["RA"],
        v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["DEC"],
        v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["Z_EST_ALL"],
    )
    ax.scatter(nirspec_cats["ra"], nirspec_cats["dec"], nirspec_cats["z_spec"])
    ax.scatter(alt_cat["ra"], alt_cat["dec"], alt_cat["z_alt"])

    plt.show()
