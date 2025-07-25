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

    # ext_cat = ext_cat[~np.isfinite(ext_cat["zmed_prev"])]
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
    #     ext_cat,
    #     (ext_cat["Z_FLAG"] >= 9) & (~np.isfinite(ext_cat["zspec"])),
    #     ax=ax,
    #     label="Secure",
    # )

    multiple_cat = Table.read(
        niriss_dir / "match_catalogues" / "cha_multiple.txt", format="ascii"
    )
    print(multiple_cat)
    multiple_cat.sort("z")

    test_cat = ext_cat  # [ext_cat["Z_FLAG"] >= 3]
    test_cat.sort("Z_NIRISS")

    niriss_coords = SkyCoord(test_cat["RA"], test_cat["DEC"], unit="deg")
    mult_coords = SkyCoord(multiple_cat["RAdeg"], multiple_cat["DEdeg"])
    idx, d2d, _ = match_coordinates_sky(niriss_coords, mult_coords)
    sep_constraint = d2d < 0.5 * u.arcsec
    niriss_matches = test_cat[sep_constraint]
    niriss_matches["d2d"] = d2d[sep_constraint].to(u.arcsec)
    from astropy.table import hstack

    matched_multiple = hstack([multiple_cat[idx[sep_constraint]], niriss_matches])
    # with np.printoptions(threshold=np.inf):
    #     print (matched_multiple)
    #     print (multiple_cat[idx[sep_constraint]][multiple_cat.colnames])
    #     print (niriss_matches["ID_NIRISS","RA","DEC","Z_NIRISS","Z_FLAG"])
    matched_multiple.sort("z")
    # matched_multiple.group_by(["z","ID_NIRISS")]
    matched_multiple[
        multiple_cat.colnames + ["ID_NIRISS", "Z_NIRISS", "Z_FLAG", "d2d"]
    ].pprint_all()
    # print (ext_cat["ID_NIRISS"].astype(int)==998)
    # print (niriss_coords[ext_cat["ID_NIRISS"]=="998"])
    print(len(matched_multiple))
    # exit()

    multi_img_ids = np.array([[345, 475, 936]])

    print(np.isin(ext_cat["ID_NIRISS"], multi_img_ids))

    for i, (label, z_range, icon, size) in enumerate(
        zip(
            ["1.10", "1.34", "1.87", "2.65"],
            [[1.09, 1.11], [1.32, 1.37], [1.85, 1.9], [2.63, 2.67]],
            ["x", "*", "o", "s"],
            [15, 30, 15, 15],
        )
    ):
        plot_utils.sky_plot(
            ext_cat,
            (
                (
                    (
                        (ext_cat["Z_FLAG"] >= 4)
                        # & (~np.isfinite(ext_cat["zspec"]))
                        & (ext_cat["Z_NIRISS"] >= z_range[0])
                        & (ext_cat["Z_NIRISS"] < z_range[-1])
                    )
                    | (
                        (ext_cat["zmed_prev"] >= z_range[0])
                        & (ext_cat["zmed_prev"] < z_range[-1])
                    )
                )
                & (~np.isin(ext_cat["ID_NIRISS"], multi_img_ids.ravel()))
            ),
            ax=ax,
            label=rf"$\overline{{z}}={label}$",
            s=size,
            marker=icon,
            edgecolor="none",
        )
        print(
            z_range,
            np.nanmedian(
                ext_cat["Z_NIRISS"][
                    (ext_cat["Z_FLAG"] >= 4)
                    # & (~np.isfinite(ext_cat["zspec"]))
                    & (ext_cat["Z_NIRISS"] >= z_range[0])
                    & (ext_cat["Z_NIRISS"] < z_range[-1])
                ]
            ),
        )

    plot_utils.sky_plot(
        ext_cat,
        ((np.isin(ext_cat["ID_NIRISS"], multi_img_ids.ravel()))),
        ax=ax,
        # label=rf"$\overline{{z}}={label}$",
        s=15,
        marker="s",
        facecolor="none",
        edgecolor="C3",
    )

    # test_cat = ext_cat

    # plot_utils.sky_plot(
    #     ext_cat,
    #     (
    #         (ext_cat["Z_FLAG"] >= 4)
    #         # & (~np.isfinite(ext_cat["zspec"]))
    #         & (ext_cat["Z_NIRISS"] >= 0.28)
    #         & (ext_cat["Z_NIRISS"] < 0.33)
    #     ),
    #     ax=ax,
    #     label=label,
    #     s=size,
    #     marker=icon,
    #     edgecolor="none",
    # )

    ax.set_xlabel(r"R.A.")
    ax.set_ylabel(r"Dec.")
    ax.coords[0].set_major_formatter("d.dd")
    ax.coords[1].set_major_formatter("d.dd")
    ax.legend()

    # plt.savefig(save_dir / "sample_sky_plot.pdf", dpi=600)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "overdensities_sky_plot_rev1.pdf", dpi=600)
    # plt.savefig(save_dir / "svg" / "overdensities_sky_plot_2.svg", dpi=600)

    plt.show()

    # z_range = [1.32, 1.37]
    # # z_range = [1.85,1.9]

    # for r in ext_cat["ID", "RA", "DEC"][
    #     (ext_cat["Z_FLAG"] >= 9)  # & (~np.isfinite(ext_cat["zspec"])
    #     & (ext_cat["Z_NIRISS"] >= z_range[0])
    #     & (ext_cat["Z_NIRISS"] < z_range[-1])
    # ]:
    #     print(r[0], r[1], r[2])
