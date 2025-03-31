"""
Merge existing redshift catalogues.
"""

import os
from pathlib import Path

import astropy.units as u
import numpy as np
import plot_utils
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.table import Table, join, vstack

root_dir = Path(os.getenv("ROOT_DIR"))
root_name = "glass-a2744"
save_dir = root_dir / "2024_08_16_A2744_v4" / "catalogue_paper" / "plots"
niriss_dir = root_dir / "2024_08_16_A2744_v4" / "glass_niriss"
grizli_dir = root_dir / "2024_08_16_A2744_v4" / "grizli_home"

if __name__ == "__main__":

    init_cat = Table.read(
        grizli_dir
        / "Extractions_v4"
        / "catalogues"
        / "stage_6_output_internal_phot.fits"
    )

    z_cat_name = "stage_6_output_internal_phot_zprev.fits"
    stage_cat_dir = grizli_dir / "Extractions_v4" / "catalogues" / "stage_6"
    stage_cat_dir.mkdir(exist_ok=True, parents=True)

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

        ground_cat = Table.read(
            prev_cat_dir / "A2744_zcat_v10.0_edit.cat", format="ascii"
        )
        ground_cat.rename_columns(
            ground_cat.colnames, [s.lower() for s in ground_cat.colnames]
        )
        ground_cat = ground_cat[
            np.isin(ground_cat["qf"], [3, 5, 9, 10, 13, 14, 15, 24, 33])
        ]
        ground_cat.rename_columns(
            ["ra", "dec", "specz", "qf"],
            ["ra_ground", "dec_ground", "z_ground", "flag_ground"],
        )
        ground_cat["id_ground"] = np.arange(len(ground_cat))

        ground_coords = SkyCoord(
            ra=ground_cat["ra_ground"], dec=ground_cat["dec_ground"], unit="deg"
        )

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

        megascience_cat = Table.read(prev_cat_dir / "UNCOVER_DR3_LW_SUPER_catalog.fits")
        megascience_cat.rename_columns(
            megascience_cat.colnames, [s.lower() for s in megascience_cat.colnames]
        )
        megascience_cols = ["id", "ra", "dec", "z_phot", "flag_eazy"]
        megascience_cat = megascience_cat[megascience_cols]
        megascience_cat = megascience_cat[megascience_cat["flag_eazy"] == 1]
        megascience_coords = SkyCoord(
            ra=megascience_cat["ra"], dec=megascience_cat["dec"], unit="deg"
        )
        megascience_cat.rename_columns(
            megascience_cols, [s + "_megascience" for s in megascience_cols]
        )

        for l in ["id_astrodeep", "id_megascience"]:
            init_cat.add_column(
                np.full(len(init_cat), np.nan),
                name=l,
                index=-1,
            )
        sep_limit = 0.5 * u.arcsec
        test_list = []
        for row in init_cat[:]:

            coord = SkyCoord(ra=row["RA"], dec=row["DEC"], unit="deg")
            idx_astrodeep, sep_astrodeep, _ = match_coordinates_sky(
                coord, astrodeep_coords
            )
            if sep_astrodeep.to(u.arcsec) <= sep_limit:
                row["id_astrodeep"] = astrodeep_cat["id_astrodeep"][idx_astrodeep]
            idx_megascience, sep_megascience, _ = match_coordinates_sky(
                coord, megascience_coords
            )
            if sep_megascience.to(u.arcsec) <= sep_limit:
                row["id_megascience"] = megascience_cat["id_megascience"][
                    idx_megascience
                ]

            test_list.append(np.array(row))

        full_cat = Table(np.asarray(test_list))

        full_cat = join(full_cat, astrodeep_cat, keys="id_astrodeep", join_type="left")
        full_cat = join(
            full_cat, megascience_cat, keys="id_megascience", join_type="left"
        )

        extra_ids = ["id_ground", "id_msa_1324", "id_msa_2756", "id_msa_2561", "id_alt"]
        for s in ["id_ground", "id_nirspec", "id_alt"]:
            full_cat.add_column(
                np.full(len(full_cat), np.nan),
                name=s,
                index=-1,
            )
        glass_niriss_cat = full_cat[:0]
        sep_limit = 0.5 * u.arcsec
        test_list = []
        for row in full_cat[:]:
            coord = SkyCoord(ra=row["RA"], dec=row["DEC"], unit="deg")
            idx_ground, sep_ground, _ = match_coordinates_sky(coord, ground_coords)
            if sep_ground.to(u.arcsec) <= sep_limit:
                row["id_ground"] = ground_cat["id_ground"][idx_ground]

            idx_nirspec, sep_nirspec, _ = match_coordinates_sky(coord, nirspec_coords)
            if sep_nirspec.to(u.arcsec) <= sep_limit:
                row["id_nirspec"] = nirspec_cats["id_nirspec"][idx_nirspec]

            idx_alt, sep_alt, _ = match_coordinates_sky(coord, alt_coords)
            if sep_alt.to(u.arcsec) <= sep_limit:
                row["id_alt"] = alt_cat["id_alt"][idx_alt]

            test_list.append(np.array(row))

        glass_niriss_cat = Table(np.asarray(test_list))

        full_z_cat = join(
            glass_niriss_cat, ground_cat, keys="id_ground", join_type="left"
        )
        full_z_cat = join(full_z_cat, nirspec_cats, keys="id_nirspec", join_type="left")
        full_z_cat = join(full_z_cat, alt_cat, keys="id_alt", join_type="left")

        ground_not_added = ground_cat[
            ~np.isin(ground_cat["id_ground"], full_z_cat["id_ground"])
        ]

        nirspec_not_added = nirspec_cats[
            ~np.isin(nirspec_cats["id_nirspec"], full_z_cat["id_nirspec"])
        ]
        alt_not_added = alt_cat[~np.isin(alt_cat["id_alt"], full_z_cat["id_alt"])]

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
            idx_nirspec, sep_nirspec, _ = match_coordinates_sky(coord, nirspec_coords)
            if sep_nirspec.to(u.arcsec) <= sep_limit:
                row["id_nirspec"] = nirspec_cats["id_nirspec"][idx_nirspec]

            idx_alt, sep_alt, _ = match_coordinates_sky(coord, alt_coords)
            if sep_alt.to(u.arcsec) <= sep_limit:
                row["id_alt"] = alt_not_added["id_alt"][idx_alt]

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

            idx_alt, sep_alt, _ = match_coordinates_sky(coord, alt_coords)
            if sep_alt.to(u.arcsec) <= sep_limit:
                row["id_alt"] = alt_not_added["id_alt"][idx_alt]

            test_list.append(np.array(row))

        nirspec_matched = Table(np.asarray(test_list))

        nirspec_matched = join(
            nirspec_matched, alt_not_added, keys="id_alt", join_type="left"
        )
        full_z_cat = vstack([full_z_cat, nirspec_matched])

        full_z_cat = vstack(
            [full_z_cat, alt_cat[~np.isin(alt_cat["id_alt"], full_z_cat["id_alt"])]]
        )

        full_z_cat.sort(["NUMBER", "id_nirspec"])

        zmed = plot_utils.table_to_array(
            full_z_cat["z_ground", "z_nirspec", "z_alt"]
        ).filled(np.nan)

        full_z_cat["zmed_prev"] = np.nanmedian(zmed, axis=1)

        full_z_cat.meta["EXTNAME"] = "GLASS-JWST NIRISS"

        full_z_cat.write(stage_cat_dir / z_cat_name)
    else:
        full_z_cat = Table.read(stage_cat_dir / z_cat_name)

    stage_dir = grizli_dir / "Extractions_v4"

    try:
        full_line_table = Table.read(stage_cat_dir / "full_internal_em_line_data.fits")
    except:
        try:
            line_table = Table.read(stage_cat_dir / "em_line_data.fits")
        except:
            line_data = []

            for f in [*(stage_dir / "row").glob("*row.fits")][:]:
                print(f)

                # obj_id =
                # obj_id = f.stem.split("_")[-1].strip(".row")
                if (
                    grizli_dir / "for_others" / "Peter" / "60mas" / "row" / f.name
                ).is_file():
                    print("Yes")
                    print(f.name)
                    line_data.append(
                        Table.read(
                            (
                                grizli_dir
                                / "for_others"
                                / "Peter"
                                / "60mas"
                                / "row"
                                / f.name
                            )
                        )[0]
                    )
                else:
                    # if grizli_dir /
                    # print(f)
                    line_data.append(Table.read(f)[0])

            line_table = Table(line_data)
            line_table.write(stage_cat_dir / "em_line_data.fits")
        full_line_table = join(
            full_z_cat[~full_z_cat["ID"].mask],
            line_table,
            keys_left="ID",
            keys_right="id",
            join_type="left",
        )
        full_line_table.rename_column("ID", "ID_NIRISS")
        full_line_table.write(stage_cat_dir / "full_internal_em_line_data.fits")
    # print (full_line_table.colnames)

    # full_line_table.rename_columns(full_line_table.colnames, [s.upper() for s in full_line_table.colnames])
    try:
        niriss_ext_cat = Table.read(stage_cat_dir / "full_external_em_lines.fits")
    except:
        lines = {
            "OII": np.array([3727.092, 3729.875]),
            "Hb": 4862.738,
            "OIII": np.array([5008.240, 4960.295]),
            "Ha": 6564.697,
            "SII": np.array([6718.29, 6732.67]),
            "SIII": np.array([9071.1, 9533.2]),
        }

        full_line_table.rename_columns(
            [
                "F115W,72.0_QUALITY",
                "F150W,72.0_QUALITY",
                "F200W,72.0_QUALITY",
                "F115W,341.0_QUALITY",
                "F150W,341.0_QUALITY",
                "F200W,341.0_QUALITY",
            ],
            [
                "F115W_072.0_QUALITY",
                "F150W_072.0_QUALITY",
                "F200W_072.0_QUALITY",
                "F115W_341.0_QUALITY",
                "F150W_341.0_QUALITY",
                "F200W_341.0_QUALITY",
            ],
        )

        ext_cols = [
            "ID_NIRISS",
            "RA",
            "DEC",
            "Z_NIRISS",
            "Z_FLAG",
            "F115W_072.0_QUALITY",
            "F115W_341.0_QUALITY",
            "F150W_072.0_QUALITY",
            "F150W_341.0_QUALITY",
            "F200W_072.0_QUALITY",
            "F200W_341.0_QUALITY",
        ]
        for l in lines:
            ext_cols.extend([f"flux_{l}", f"err_{l}"])

        niriss_ext_cat = Table()
        for e in ext_cols:
            niriss_ext_cat[e] = full_line_table[e][full_line_table["ID_NIRISS"] < 4000]
            if ("flux" in e) or ("err" in e):
                niriss_ext_cat[e] = niriss_ext_cat[e].filled(0.0)

        from glass_niriss.grism import NIRISS_FILTER_LIMITS, check_coverage

        filt_names = {
            "F115W": [
                "F115W_072.0_QUALITY",
                "F115W_341.0_QUALITY",
            ],
            "F150W": [
                "F150W_072.0_QUALITY",
                "F150W_341.0_QUALITY",
            ],
            "F200W": [
                "F200W_072.0_QUALITY",
                "F200W_341.0_QUALITY",
            ],
        }

        for i, row in enumerate(niriss_ext_cat[:]):
            if np.isfinite(row["Z_NIRISS"]):
                use_filts = {}
                for f_name, f_q_names in filt_names.items():
                    if not (
                        np.lib.recfunctions.structured_to_unstructured(
                            np.array(row[f_q_names])
                        )
                        == 0
                    ).all():
                        use_filts[f_name] = NIRISS_FILTER_LIMITS[f_name]
                # print (row["ID_NIRISS","Z_NIRISS"])
                for l, w_rf in lines.items():
                    # print (l, check_coverage(w_rf*(1+row["Z_NIRISS"]), filter_limits=use_filts))
                    if not check_coverage(
                        w_rf * (1 + row["Z_NIRISS"]), filter_limits=use_filts
                    ):
                        niriss_ext_cat[f"flux_{l}"][i] = 0.0
                        niriss_ext_cat[f"err_{l}"][i] = 0.0
            if row["Z_FLAG"] <= 2:
                for l, w_rf in lines.items():
                    niriss_ext_cat[f"flux_{l}"][i] = 0.0
                    niriss_ext_cat[f"err_{l}"][i] = 0.0
                niriss_ext_cat["Z_NIRISS"][i] = np.nan

        niriss_ext_cat.write(
            stage_cat_dir / "full_external_em_lines.fits", overwrite=True
        )

    # # print (niriss_ext_cat)
    # niriss_ext_cat.write(stage_cat_dir / "full_external_em_lines.fits", overwrite=True)
    # # print (full_z_cat.colnames)

    overdens_dir = stage_cat_dir / "overdens"
    overdens_dir.mkdir(exist_ok=True, parents=True)

    import matplotlib.pyplot as plt

    fig, axs = plt.subplots()

    z_groups = [[1.09, 1.11], [1.32, 1.37], [1.85, 1.90]]
    for z_low, z_high in z_groups:
        # print (z_low)

        idx = (
            (niriss_ext_cat["Z_NIRISS"] >= z_low)
            & (niriss_ext_cat["Z_NIRISS"] < z_high)
            & (niriss_ext_cat["Z_FLAG"] >= 4)
        )
        print(len(niriss_ext_cat[idx]))
        niriss_ext_cat[idx].write(
            overdens_dir / f"overdens_z_{z_low}_{z_high}.fits", overwrite=True
        )
        axs.scatter(niriss_ext_cat[idx]["RA"], niriss_ext_cat[idx]["DEC"])

    cluster_low = 0.28
    cluster_high = 0.33

    secure = (niriss_ext_cat["Z_FLAG"] >= 4) & np.isfinite(niriss_ext_cat["Z_NIRISS"])
    tentative = (niriss_ext_cat["Z_FLAG"] == 3) & np.isfinite(
        niriss_ext_cat["Z_NIRISS"]
    )
    idx = (
        (secure | tentative)
        & (niriss_ext_cat["Z_NIRISS"] >= cluster_low)
        & (niriss_ext_cat["Z_NIRISS"] < cluster_high)
        # & ~spec
    )
    axs.scatter(niriss_ext_cat[idx]["RA"], niriss_ext_cat[idx]["DEC"])
    axs.invert_xaxis()

    # plt.show()

    niriss_ext_cat = full_line_table

    niriss_ext_cat = Table.read(stage_cat_dir / "LCE_candidates_NIRISS.fits")
    niriss_ext_cat["zmed_prev"] = niriss_ext_cat["Z_SPEC_VALUE"]

    for row in niriss_ext_cat:
        print(f"{row["RA"]:.5f} {row["DEC"]:.5f}")

    print(niriss_ext_cat.colnames)
    # niriss_ext_cat["Z_NIRISS"] = niriss_ext_cat["Z_NIRISS"].filled(np.nan)
    # niriss_ext_cat["zmed_prev"] = niriss_ext_cat["zmed_prev"].filled(np.nan)

    cat_coords = SkyCoord(
        ra=niriss_ext_cat["RA"], dec=niriss_ext_cat["DEC"], unit="deg"
    )

    # import astropy.units as u
    # boyett+22
    boyett_tab = Table.read(niriss_dir / "match_catalogues" / "Boyett+22.fits")
    boyett_tab.sort("z")

    boyett_coords = SkyCoord(
        ra=boyett_tab["RAdeg"], dec=boyett_tab["DEdeg"], unit="deg"
    )

    idxs, sep, _ = match_coordinates_sky(boyett_coords, cat_coords)

    for row, idx, s in zip(boyett_tab, idxs, sep):
        glass = niriss_ext_cat[idx]
        # print (row["RAdeg","DEdeg"][0])
        extra_flag = ""
        dz = (row["z"] - glass["Z_NIRISS"]) / (1 + row["z"])
        if s > 1 * u.arcsec:
            extra_flag = "\t[distant, different object?]"
        elif np.abs(dz) > 1e-3:
            if np.abs(dz) > 0.02:
                # print ("!!")
                extra_flag = "\t[z not equal]"
        elif not np.isfinite(dz):
            # if np.abs(dz)>0.02:
            extra_flag = "\t[no z in catalogue]"
            # print ("!!")
        print(
            # f"Boyett: {row["ID"]: >5}, z={row["z"]:.3f}\tGLASS: {glass["ID_NIRISS"]: >5}, ",
            # f"z={glass["Z_NIRISS"]:.3f}\tsep={s.to(u.arcsec):.3f},\t",
            f"{glass["RA"]:.4f}, {glass["DEC"]:.4f}\t"
            # f"z_flag={glass["Z_FLAG"]}\tz_prev={glass["zmed_prev"]:.3f}"+extra_flag
        )

    # 3112
