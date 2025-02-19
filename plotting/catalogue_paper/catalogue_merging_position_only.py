"""
Merge existing redshift catalogues.
"""

from astropy.table import Column, join, vstack
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    # z_cat_name = "stage_3_z_cat_input.fits"
    # stage_3_cat_dir = grizli_dir / "Extractions_v3" / "classification"

    # full_cat = Table.read(
    #     grizli_dir / "ForcedExtractions" / "glass-a2744-ir.cat.fits"
    # )
    # full_cat = Table.read(
    #     grizli_dir / "Extractions_v4" / "catalogues" / "stage_4_input.fits"
    # )
    full_cat = Table.read(
        grizli_dir
        / "Extractions_v4"
        / "catalogues"
        / "stage_5_output_internal_phot.fits"
    )

    z_cat_name = "stage_5_output_internal_phot_zprev.fits"
    stage_cat_dir = grizli_dir / "Extractions_v4" / "catalogues"

    if not (stage_cat_dir / z_cat_name).is_file():
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
        ground_cat = ground_cat[
            np.isin(ground_cat["qf"], [3, 5, 9, 10, 13, 14, 15, 24, 33])
        ]
        print(ground_cat.colnames)
        ground_cat.rename_columns(
            ["ra", "dec", "specz", "qf"],
            ["ra_ground", "dec_ground", "z_ground", "flag_ground"],
        )
        ground_cat["id_ground"] = np.arange(len(ground_cat))

        ground_coords = SkyCoord(
            ra=ground_cat["ra_ground"], dec=ground_cat["dec_ground"], unit="deg"
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

        astrodeep_cat = Table.read(
            prev_cat_dir / "ASTRODEEP-JWST-ABELL2744_photoz.fits"
        )
        astrodeep_cat.rename_columns(
            astrodeep_cat.colnames, [s.lower() for s in astrodeep_cat.colnames]
        )
        astrodeep_cols = ["id", "ra", "dec", "zphot", "flag"]
        astrodeep_cat = astrodeep_cat[astrodeep_cols]
        astrodeep_coords = SkyCoord(
            ra=astrodeep_cat["ra"], dec=astrodeep_cat["dec"], unit="deg"
        )
        astrodeep_cat.rename_columns(
            astrodeep_cols, [s + "_astrodeep" for s in astrodeep_cols]
        )

        # astrodeep_cat[astrodeep_cat["zphot_astrodeep"]==19.85] = np.nan

        full_cat.add_column(
            np.full(len(full_cat), np.nan),
            name="id_astrodeep",
            index=-1,
        )
        sep_limit = 0.5 * u.arcsec
        test_list = []
        for row in full_cat[:]:

            coord = SkyCoord(ra=row["RA"], dec=row["DEC"], unit="deg")
            idx_astrodeep, sep_astrodeep, _ = match_coordinates_sky(
                coord, astrodeep_coords
            )
            # print (sep.to(u.arcsec))
            if sep_astrodeep.to(u.arcsec) <= sep_limit:
                # print (row["NUMBER"], "NIRSpec", nirspec_cats["z_nirspec"][idx_nirspec], row["Z_EST_ALL"], row["Z_FLAG_ALL"])
                # row["id_msa_1324","id_msa_2756","id_msa_2561"] = nirspec_cats["id_msa_1324","id_msa_2756","id_msa_2561"][idx_nirspec]
                row["id_astrodeep"] = astrodeep_cat["id_astrodeep"][idx_astrodeep]
                # print ("Matched to ASTRODEEP!")

            test_list.append(np.array(row))

        full_cat = Table(np.asarray(test_list))

        # nirspec_coords_added = nirspec_coords[np.argsort(full)]

        full_cat = join(full_cat, astrodeep_cat, keys="id_astrodeep", join_type="left")

        extra_ids = ["id_ground", "id_msa_1324", "id_msa_2756", "id_msa_2561", "id_alt"]
        # full_cat.rename_column("id_alt", "orig_id_alt")
        for s in ["id_ground", "id_nirspec", "id_alt"]:
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
            idx_ground, sep_ground, _ = match_coordinates_sky(coord, ground_coords)
            # print (sep.to(u.arcsec))
            if sep_ground.to(u.arcsec) <= sep_limit:
                # print (row["NUMBER"], "NIRSpec", nirspec_cats["z_nirspec"][idx_nirspec], row["Z_EST_ALL"], row["Z_FLAG_ALL"])
                # row["id_msa_1324","id_msa_2756","id_msa_2561"] = nirspec_cats["id_msa_1324","id_msa_2756","id_msa_2561"][idx_nirspec]
                row["id_ground"] = ground_cat["id_ground"][idx_ground]
                # print ("Matched to ground!")

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
            glass_niriss_cat, ground_cat, keys="id_ground", join_type="left"
        )
        full_z_cat = join(full_z_cat, nirspec_cats, keys="id_nirspec", join_type="left")
        full_z_cat = join(full_z_cat, alt_cat, keys="id_alt", join_type="left")

        # full_z_cat = vstack([
        #     full_z_cat,
        #     nirspec_cats[~np.isin(nirspec_cats["id_nirspec"],full_z_cat["id_nirspec"])]["id_nirspec"]
        # ])

        # print (np.nansum(~(prev_cat[z_spec_name].mask)))

        # exit()

        ground_not_added = ground_cat[
            ~np.isin(ground_cat["id_ground"], full_z_cat["id_ground"])
        ]

        nirspec_not_added = nirspec_cats[
            ~np.isin(nirspec_cats["id_nirspec"], full_z_cat["id_nirspec"])
        ]
        alt_not_added = alt_cat[~np.isin(alt_cat["id_alt"], full_z_cat["id_alt"])]

        # print(
        #     len(full_z_cat),
        #     len(nirspec_cats),
        #     len(nirspec_not_added),
        #     len(alt_cat),
        #     len(alt_not_added),
        # )

        ground_coords = SkyCoord(
            ra=ground_not_added["ra_ground"],
            dec=ground_not_added["dec_ground"],
            unit="deg",
        )
        nirspec_coords = SkyCoord(
            ra=nirspec_not_added["ra_nirspec"],
            dec=nirspec_not_added["dec_nirspec"],
            unit="deg",
        )
        alt_coords = SkyCoord(
            ra=alt_not_added["ra_alt"], dec=alt_not_added["dec_alt"], unit="deg"
        )
        ground_not_added.add_column(
            np.full(len(ground_not_added), np.nan),
            name="id_nirspec",
            index=-1,
        )
        ground_not_added.add_column(
            np.full(len(ground_not_added), np.nan),
            name="id_alt",
            index=-1,
        )
        test_list = []
        for row, coord in zip(ground_not_added, ground_coords):
            # print (row)
            # coord = SkyCoord(ra=row["RA"],dec=row["DEC"], unit="deg")
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
                row["id_alt"] = alt_not_added["id_alt"][idx_alt]

            # glass_niriss_cat.add_row(row)
            test_list.append(np.array(row))

        ground_matched = Table(np.asarray(test_list))

        ground_matched = join(
            ground_matched, nirspec_not_added, keys="id_nirspec", join_type="left"
        )
        ground_matched = join(
            ground_matched, alt_not_added, keys="id_alt", join_type="left"
        )

        full_z_cat = vstack([full_z_cat, ground_matched])

        nirspec_not_added = nirspec_cats[
            ~np.isin(nirspec_cats["id_nirspec"], full_z_cat["id_nirspec"])
        ]
        alt_not_added = alt_cat[~np.isin(alt_cat["id_alt"], full_z_cat["id_alt"])]

        print(
            len(full_z_cat),
            len(ground_cat),
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

        zmed = plot_utils.table_to_array(
            full_z_cat["z_ground", "z_nirspec", "z_alt"]
        ).filled(np.nan)

        full_z_cat["zmed_prev"] = np.nanmedian(zmed, axis=1)

        # exit()
        full_z_cat.write(stage_cat_dir / z_cat_name)
    else:
        full_z_cat = Table.read(stage_cat_dir / z_cat_name)
