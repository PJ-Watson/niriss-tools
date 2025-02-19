"""
Show the distribution of galaxies on sky.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()


if __name__ == "__main__":

    img_path = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Prep"
        / "glass-a2744-ir_drc_sci.fits"
    )

    # v1_cat = Table.read(catalogue_dir / "compiled_catalogue_v1.fits")

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

    secure = (
        (np.abs(full_cat["Z_EST_ALL"] - 0.3064) <= 0.025)
        & (full_cat["Z_FLAG_ALL"] >= 9)
        & np.isfinite(full_cat["zspec"])
        & (
            (np.abs(full_cat["Z_EST_ALL"] - full_cat["zspec"]) / full_cat["zspec"])
            <= 0.05
        )
    )
    _, _, sc = plot_utils.sky_plot(
        full_cat,
        secure,
        ax=ax,
        label="Full Sample",
        c=((full_cat["Z_EST_ALL"] - full_cat["zspec"]) / full_cat["zspec"])[secure],
        cmap="rainbow",
    )
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

    # for i, (label, z_range, icon, size) in enumerate(
    #     zip(
    #         ["1.10", "1.34", "1.87"],
    #         [[1.09, 1.11], [1.32, 1.37], [1.85, 1.9]],
    #         ["x", "*", "o"],
    #         [15, 30, 15],
    #     )
    # ):
    #     plot_utils.sky_plot(
    #         v2_cat,
    #         (
    #             (v2_cat["Z_FLAG_ALL"] >= 9)
    #             # & (~np.isfinite(v2_cat["zspec"]))
    #             & (v2_cat["Z_EST_ALL"] >= z_range[0])
    #             & (v2_cat["Z_EST_ALL"] < z_range[-1])
    #         ),
    #         ax=ax,
    #         label=label,
    #         s=size,
    #         marker=icon,
    #         edgecolor="none",
    #     )
    #     print(
    #         z_range,
    #         np.nanmedian(
    #             v2_cat["Z_EST_ALL"][
    #                 (v2_cat["Z_FLAG_ALL"] >= 9)
    #                 # & (~np.isfinite(v2_cat["zspec"]))
    #                 & (v2_cat["Z_EST_ALL"] >= z_range[0])
    #                 & (v2_cat["Z_EST_ALL"] < z_range[-1])
    #             ]
    #         ),
    #     )

    #     # test_cat = v2_cat

    plt.colorbar(sc, ax=ax)

    ax.set_xlabel(r"R.A.")
    ax.set_ylabel(r"Dec.")
    ax.legend()

    # plt.savefig(save_dir / "sample_sky_plot.pdf", dpi=600)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "redshift_offset_sky_plot.pdf", dpi=600)
    # plt.savefig(save_dir / "svg" / "redshift_offset_sky_plot.svg", dpi=600)

    plt.show()
