"""
Merge existing redshift catalogues.
"""

from astropy.table import Column, join, vstack
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    z_cat_name = "merged_z_cat_2.fits"
    if not (compiled_dir / z_cat_name).is_file():
        prev_cat_dir = niriss_dir / "match_catalogues"

        sara_cat_path = (
            prev_cat_dir / "hlsp_glass-jwst_jwst_nirspec_abell2744_v1_cat.fits"
        )

        _sara_cat_1 = Table.read(sara_cat_path, 1)
        _sara_cat_1.rename_column("MSA_ID", "id_msa_1324")
        _sara_cat_2 = Table.read(sara_cat_path, 2)
        _sara_cat_2.rename_column("MSA_ID", "id_msa_2756")
        sara_cat = vstack([_sara_cat_1, _sara_cat_2])
        sara_cat.rename_columns(
            sara_cat.colnames, [s.lower() for s in sara_cat.colnames]
        )
        # Remove low-quality redshifts
        sara_cat = sara_cat[sara_cat["flag"] != 1]

        alt_cat = Table.read(prev_cat_dir / "ALT_DR1_public.fits")
        # Remove single-line detections
        alt_cat = alt_cat[alt_cat["n_lines_detected"] >= 2]
        alt_cat.rename_columns(alt_cat.colnames, [s.lower() for s in alt_cat.colnames])
        alt_cat.rename_columns(
            ["id", "ra", "dec", "n_lines_detected"],
            ["id_alt", "ra_alt", "dec_alt", "nlines_alt"],
        )
        alt_coords = SkyCoord(ra=alt_cat["ra_alt"], dec=alt_cat["dec_alt"], unit="deg")

        uncover_cat = Table.read(
            prev_cat_dir / "uncover-msa-default_drz-DR4.1-zspec.fits"
        )
        uncover_cat.rename_columns(
            uncover_cat.colnames, [s.lower() for s in uncover_cat.colnames]
        )
        uncover_cat.rename_column("id_msa", "id_msa_2561")
        # Remove low-quality redshifts
        uncover_cat = uncover_cat[uncover_cat["flag_zspec_qual"] >= 2]
        # uncover_cat["id_msa_2561")
        # print (uncover_cat)

        ground_cat = Table.read(
            prev_cat_dir / "A2744_zcat_v10.0_edit.cat", format="ascii"
        )
        ground_cat.rename_columns(
            ground_cat.colnames, [s.lower() for s in ground_cat.colnames]
        )
        print(ground_cat.colnames)
        ground_cat.rename_columns(
            ["id", "ra", "dec", "specz", "qf"],
            ["id_ground", "ra_ground", "dec_ground", "z_ground", "flag_ground"],
        )

        # img_path = grizli_dir / "Prep" / "glass-a2744-ir_drc_sci.fits"

        # with fits.open(img_path) as img_hdul:
        #     img_wcs = WCS(img_hdul[0])

        #     fig, ax = plt.subplots(
        #         figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth),
        #         constrained_layout=True,
        #         subplot_kw={"projection": img_wcs},
        #     )
        #     ax.imshow(
        #         img_hdul[0].data,
        #         norm=astrovis.ImageNormalize(
        #             img_hdul[0].data,
        #             # interval=astrovis.PercentileInterval(99.9),
        #             interval=astrovis.ManualInterval(-0.001, 10),
        #             stretch=astrovis.LogStretch(),
        #         ),
        #         cmap="binary_r",
        #         origin="lower",
        #     )
        #     xlim = ax.get_xlim()
        #     ylim = ax.get_ylim()

        # for i, (cat) in enumerate([sara_cat, uncover_cat, alt_cat]):
        #     plot_utils.sky_plot(cat, ax=ax, label="Placeholder")

        # ax.set_xlim(xlim)
        # ax.set_ylim(ylim)

        sara_cat.rename_columns(
            ["ra", "dec", "z_spec", "flag"],
            ["ra_nirspec", "dec_nirspec", "z_nirspec", "flag_nirspec"],
        )
        uncover_cat.rename_columns(
            ["ra", "dec", "z_spec", "flag_zspec_qual"],
            ["ra_nirspec", "dec_nirspec", "z_nirspec", "flag_nirspec"],
        )
        nirspec_col_names = [
            "ra_nirspec",
            "dec_nirspec",
            "z_nirspec",
            "flag_nirspec",
        ]
        nirspec_cats = vstack(
            [
                sara_cat["id_msa_1324", "id_msa_2756", *nirspec_col_names],
                uncover_cat["id_msa_2561", *nirspec_col_names],
            ]
        )
        nirspec_cats["id_nirspec"] = np.arange(len(nirspec_cats))
        # print(nirspec_cats)
        nirspec_coords = SkyCoord(
            ra=nirspec_cats["ra_nirspec"], dec=nirspec_cats["dec_nirspec"], unit="deg"
        )

        # print (np.unique(nirspec_cats["id_msa"]))

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

        # combined_cat =
        # full_cat.add_columns(
        #     Column(name="id_msa_1324"),
        #     Column(name="id_msa_2756"),
        #     Column(name="id_msa_2561"),
        #     Column(name="id_alt"),
        # )
        extra_ids = ["id_ground", "id_msa_1324", "id_msa_2756", "id_msa_2561", "id_alt"]
        full_cat.rename_column("id_alt", "orig_id_alt")
        for s in ["id_nirspec", "id_alt"]:
            full_cat.add_column(
                np.full(len(full_cat), np.nan),
                name=s,
                index=-1,
            )
        # print (full_cat[]
        glass_niriss_cat = full_cat[:0]
        sep_limit = 0.5 * u.arcsec
        test_list = []
        for row in full_cat[:]:
            # print (row)
            coord = SkyCoord(ra=row["RA"], dec=row["DEC"], unit="deg")
            idx_nirspec, sep_nirspec, _ = match_coordinates_sky(coord, nirspec_coords)
            # print (sep.to(u.arcsec))
            if sep_nirspec.to(u.arcsec) <= sep_limit:
                # print (row["NUMBER"], "NIRSpec", nirspec_cats["z_nirspec"][idx_nirspec], row["Z_EST_ALL"], row["Z_FLAG_ALL"])
                # row["id_msa_1324","id_msa_2756","id_msa_2561"] = nirspec_cats["id_msa_1324","id_msa_2756","id_msa_2561"][idx_nirspec]
                row["id_nirspec"] = nirspec_cats["id_nirspec"][idx_nirspec]

            idx_alt, sep_alt, _ = match_coordinates_sky(coord, alt_coords)
            # print (sep.to(u.arcsec))
            if sep_alt.to(u.arcsec) <= sep_limit:
                # print (row["NUMBER"],"ALT", alt_cat["z_alt"][idx_alt], row["Z_EST_ALL"],row["Z_FLAG_ALL"])
                row["id_alt"] = alt_cat["id_alt"][idx_alt]

            # glass_niriss_cat.add_row(row)
            test_list.append(np.array(row))

        glass_niriss_cat = Table(np.asarray(test_list))

        # nirspec_coords_added = nirspec_coords[np.argsort(full)]

        full_z_cat = join(
            glass_niriss_cat, nirspec_cats, keys="id_nirspec", join_type="left"
        )
        full_z_cat = join(full_z_cat, alt_cat, keys="id_alt", join_type="left")

        # full_z_cat = vstack([
        #     full_z_cat,
        #     nirspec_cats[~np.isin(nirspec_cats["id_nirspec"],full_z_cat["id_nirspec"])]["id_nirspec"]
        # ])

        nirspec_not_added = nirspec_cats[
            ~np.isin(nirspec_cats["id_nirspec"], full_z_cat["id_nirspec"])
        ]
        alt_not_added = alt_cat[~np.isin(alt_cat["id_alt"], full_z_cat["id_alt"])]

        print(
            len(full_z_cat),
            len(nirspec_cats),
            len(nirspec_not_added),
            len(alt_cat),
            len(alt_not_added),
        )

        nirspec_coords = SkyCoord(
            ra=nirspec_not_added["ra_nirspec"],
            dec=nirspec_not_added["dec_nirspec"],
            unit="deg",
        )
        alt_coords = SkyCoord(
            ra=alt_not_added["ra_alt"], dec=alt_not_added["dec_alt"], unit="deg"
        )
        nirspec_not_added.add_column(
            np.full(len(nirspec_not_added), np.nan),
            name="id_alt",
            index=-1,
        )
        test_list = []
        for row, coord in zip(nirspec_not_added, nirspec_coords):
            # print (row)
            # coord = SkyCoord(ra=row["RA"],dec=row["DEC"], unit="deg")
            # idx_nirspec, sep_nirspec, _  = match_coordinates_sky(coord, nirspec_coords)
            # # print (sep.to(u.arcsec))
            # if sep_nirspec.to(u.arcsec)<=sep_limit:
            #     # print (row["NUMBER"], "NIRSpec", nirspec_cats["z_nirspec"][idx_nirspec], row["Z_EST_ALL"], row["Z_FLAG_ALL"])
            #     # row["id_msa_1324","id_msa_2756","id_msa_2561"] = nirspec_cats["id_msa_1324","id_msa_2756","id_msa_2561"][idx_nirspec]
            #     row["id_nirspec"] = nirspec_cats["id_nirspec"][idx_nirspec]

            idx_alt, sep_alt, _ = match_coordinates_sky(coord, alt_coords)
            # print (sep.to(u.arcsec))
            if sep_alt.to(u.arcsec) <= sep_limit:
                # print (row["NUMBER"],"ALT", alt_cat["z_alt"][idx_alt], row["Z_EST_ALL"],row["Z_FLAG_ALL"])
                row["id_alt"] = alt_not_added["id_alt"][idx_alt]

            # glass_niriss_cat.add_row(row)
            test_list.append(np.array(row))

        nirspec_matched = Table(np.asarray(test_list))

        nirspec_matched = join(
            nirspec_matched, alt_not_added, keys="id_alt", join_type="left"
        )
        print(len(nirspec_matched))
        full_z_cat = vstack([full_z_cat, nirspec_matched])

        print(len(alt_cat[~np.isin(alt_cat["id_alt"], full_z_cat["id_alt"])]))

        full_z_cat = vstack(
            [full_z_cat, alt_cat[~np.isin(alt_cat["id_alt"], full_z_cat["id_alt"])]]
        )
        # print (plot_utils.table_to_array(full_z_cat["Z_EST_ALL","z_nirspec","z_alt"]))
        # print (np.nansum(np.isfinite(plot_utils.table_to_array(full_z_cat["Z_EST_ALL","z_nirspec","z_alt"])),axis=0))

        full_z_cat.sort(["NUMBER", "id_nirspec"])

        exit()
        full_z_cat.write(compiled_dir / z_cat_name)
    else:
        full_z_cat = Table.read(compiled_dir / z_cat_name)

    wht_path = (
        grizli_dir / "RAW" / "abell2744-01324-001-251.0-nis-f200w-gr150r_drz_sci.fits"
    )

    # from scipy.ndimage import binary_dilation, generate_binary_structure
    # struct = generate_binary_structure(2, connectivity=2)

    wht_img = fits.getdata(wht_path)
    sky_wcs = WCS(fits.getheader(wht_path))
    # print (struct.shape)

    # sky_mask = binary_dilation(wht_img, struct, iterations=10)

    combined_ra = np.zeros(len(full_z_cat))
    combined_dec = np.zeros(len(full_z_cat))

    import numpy.ma as ma

    for i, row in enumerate(full_z_cat[:]):
        # print (dir(row["RA"]))
        # print (ma.ismas)
        # print (row["RA"] is ma.masked)
        if not row["RA"] is ma.masked:
            combined_ra[i] = row["RA"]
            combined_dec[i] = row["DEC"]
        elif not row["ra_nirspec"] is ma.masked:
            combined_ra[i] = row["ra_nirspec"]
            combined_dec[i] = row["dec_nirspec"]
        # elif not row["ra_nirspec"].mask:
        # else:
        elif not row["ra_alt"] is ma.masked:
            combined_ra[i] = row["ra_alt"]
            combined_dec[i] = row["dec_alt"]
        else:
            pass

    all_coords = SkyCoord(ra=combined_ra, dec=combined_dec, unit="deg")
    # x, y = wcs_utils.skycoord_to_pixel(all_coords, sky_wcs)
    # all_coords_int = np.round(np.array([y,x])).astype(int)
    # print (np.nansum(sky_mask[tuple(all_coords_int[:,:])]))
    # # print ("contains", sky_wcs.footprint_contains(all_coords))
    # print (all_coords_int)

    contained_idx = sky_wcs.footprint_contains(all_coords)

    wht_path = (
        grizli_dir / "RAW" / "abell2744-01324-001-251.0-nis-f200w-gr150c_drz_sci.fits"
    )
    wht_img = fits.getdata(wht_path)
    sky_wcs = WCS(fits.getheader(wht_path))

    contained_idx = contained_idx | sky_wcs.footprint_contains(all_coords)

    full_z_cat = full_z_cat[contained_idx]
    print(full_z_cat)

    # exit()

    # for c in full_z_cat.colnames:
    #     print (full_z_cat[c][0].dtype)
    #     if np.issubdtype(full_z_cat[c][0], np.integer):
    #         print ("int")
    #     if full_z_cat[c].dtype is str:
    #         print ("s")

    # print (full_z_cat)
    # print (glass_niriss_cat[extra_ids][20:25])
    # print (np.asarray(test_list))
    # with np.printoptions(threshold=np.inf):
    #     # print (np.array(glass_niriss_cat))
    #     for row in glass_niriss_cat[20:25]:
    #         print (row)

    fig, ax = plt.subplots(
        figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth / 3),
        constrained_layout=True,
    )

    # niriss_idx = (full_z_cat["Z_FLAG_ALL"] >= 9).astype(bool)

    print(len(np.unique(full_z_cat["id_nirspec"])))

    niriss_idx = np.zeros(len(full_z_cat), dtype=bool)
    niriss_idx[(full_z_cat["Z_FLAG_ALL"] >= 9) & (~full_z_cat["Z_FLAG_ALL"].mask)] = (
        True
    )
    # _ns = np.where((~(full_z_cat["Z_FLAG_ALL"] >= 9)) & (np.isfinite(full_z_cat["z_nirspec"])) & (~full_z_cat["z_nirspec"].mask))
    nirspec_idx = np.zeros(len(full_z_cat), dtype=bool)
    nirspec_idx[
        np.where(
            (~niriss_idx)
            & (np.isfinite(full_z_cat["z_nirspec"]))
            & (~full_z_cat["z_nirspec"].mask)
        )
    ] = True
    nircam_idx = np.zeros(len(full_z_cat), dtype=bool)
    nircam_idx[
        np.where(
            (~niriss_idx)
            & (~nirspec_idx)
            & (np.isfinite(full_z_cat["z_alt"]))
            & (~full_z_cat["z_alt"].mask)
        )
    ] = True
    # print (dir(nirspec_idx))
    # nirspec_idx.fill(False)
    # print (nirspec_idx)
    # print (len(np.unique(full_z_cat["id_nirspec"][nirspec_idx])))

    # print (len(nirspec_cats))
    # nircam_idx = (~(full_z_cat["Z_FLAG_ALL"] >= 9)) | (~np.isfinite(full_z_cat["z_nirspec"]))
    # nircam_idx = np.isfinite(full_z_cat["z_alt"]) & (~np.isfinite(full_z_cat["z_nirspec"]))

    print(np.isfinite(full_z_cat["z_nirspec"][-5:]))
    #  & np.isfinite(full_z_cat["z_alt"])
    # print (full_z_cat["z_alt"].mask)

    print(nirspec_idx)

    print(
        np.nansum(niriss_idx),
        np.nansum(nirspec_idx),
        np.nansum(nircam_idx),
        np.sum(~niriss_idx),
        np.sum(~nirspec_idx),
    )

    z_bins = np.arange(0, 10.1, 0.1)
    # ax.hist(
    #     [
    #         full_z_cat[niriss_idx]["Z_EST_ALL"],
    #         full_z_cat[nirspec_idx]["z_nirspec"],
    #         full_z_cat[nircam_idx]["z_alt"],
    #         # nirspec_cats["z_nirspec"],
    #         # alt_cat["z_alt"],
    #     ],
    #     stacked=True,
    #     bins=z_bins,
    #     label=["NIRISS", "NIRSpec","NIRCam-WFSS", ]
    # )

    z_niriss = np.array(full_z_cat[niriss_idx]["Z_EST_ALL"])
    z_nirspec = np.array(full_z_cat[nirspec_idx]["z_nirspec"])
    z_nircam = np.array(full_z_cat[nircam_idx]["z_alt"])

    _, _, barlist = ax.hist(
        [z_niriss, z_nircam],
        bins=z_bins,
        stacked=True,
        histtype="stepfilled",
        color=["dodgerblue", "orange"],
        edgecolor=["blue", "chocolate"],
        hatch=[None, "//"],
        linewidth=1,
        zorder=2,
        alpha=1.0,
        label=[
            "NIRISS",
            "NIRCam-WFSS",
        ],
    )
    ax.hist(
        np.concatenate([z_niriss, z_nircam, z_nirspec]),
        bins=z_bins,
        histtype="stepfilled",
        hatch="X",
        color="yellowgreen",
        edgecolor="darkgreen",
        ls="--",
        linewidth=1,
        zorder=1,
        label="NIRSpec",
    )
    # for bars in barlist:
    #     for bar in bars:
    #         bar.set_edgecolor(bar.get_facecolor())
    # plt.show()

    ax.legend(loc=1)
    from matplotlib.ticker import MultipleLocator

    ax.xaxis.set_major_locator(MultipleLocator(1.0))

    ax.set_xlim(z_bins[0], z_bins[-1])
    # ax_scatter.set_ylabel(r"$m_{\rm{F200W}}$")
    ax.set_xlabel(r"$z$")

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "catalogue_merging.pdf", dpi=600)

    plt.show()
    exit()
    # fig, ax = plt.subplots(
    #     figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth / 1.62),
    #     constrained_layout=True,
    #     subplot_kw={"projection": "3d"},
    # )

    # # for cat in
    # ax.scatter(
    #     v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["RA"],
    #     v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["DEC"],
    #     v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["Z_EST_ALL"],
    # )
    # ax.scatter(nirspec_cats["ra"], nirspec_cats["dec"], nirspec_cats["z_spec"])
    # ax.scatter(alt_cat["ra"], alt_cat["dec"], alt_cat["z_alt"])

    # plt.show()

    img_path = grizli_dir / "Prep" / "glass-a2744-ir_drc_sci.fits"

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
            zorder=-1,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plot_utils.sky_plot(
        full_z_cat,
        niriss_idx,
        ax=ax,
        label="NIRISS",  # **plot_kwargs
    )
    plot_utils.sky_plot(
        full_z_cat,
        nircam_idx,
        ax=ax,
        label="NIRCam WFSS",
        ra_col="ra_alt",
        dec_col="dec_alt",  # **plot_kwargs
    )
    plot_utils.sky_plot(
        full_z_cat,
        nirspec_idx,
        ax=ax,
        label="NIRSpec",
        ra_col="ra_nirspec",
        dec_col="dec_nirspec",  # **plot_kwargs
    )
    ax.legend()
    plt.show()
