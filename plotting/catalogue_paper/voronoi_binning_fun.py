"""
Compare the NIRISS redshifts to existing spectroscopic redshifts.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    import os
    from pathlib import Path

    import numpy as np
    import yaml

    # Or similar
    root_dir = Path(os.getenv("ROOT_DIR"))
    out_base_dir = root_dir / "2024_08_16_A2744_v4" / "glass_niriss"

    # Setup a separate directory for the binned data
    bin_data_dir = out_base_dir / "binned_data"
    root_name = "glass-a2744"

    # Load the file containing all the info on photometric data
    with open(out_base_dir / "conv_ancillary_data.yaml", "r") as file:
        info_dict = yaml.safe_load(file)

    # # Or load the file containing only the SW data
    # with open(out_base_dir / "conv_ancillary_data.yaml", "r") as file:
    #     info_dict = yaml.safe_load(file)

    # Where the grism data can be found
    grizli_extraction_dir = out_base_dir.parent / "grizli_home" / "Extractions"

    # If it doesn't already exist, we make sure that the segmentation map used by `grizli` has been reprojected to the same WCS as the photometric data. Note that there is no constraint on pixel scale, or alignment between the grism and photometric data.

    from glass_niriss.isophotal import reproject_and_convolve

    repr_seg_path = out_base_dir / "PSF_matched_data" / f"{root_name}_seg_map.fits"

    if not repr_seg_path.is_file():
        # Whichever mosaic we used as a reference for the photometry
        ref_mosaic = (
            grizli_extraction_dir.parent / "Prep" / f"{root_name}-ir_drc_sci.fits"
        )

        # The path of the original segmentation map
        orig_seg = grizli_extraction_dir / f"{root_name}-ir_seg.fits"

        reproject_and_convolve(
            ref_path=ref_mosaic,
            orig_images=orig_seg,
            psfs=None,
            psf_target=None,
            out_dir=out_base_dir / "PSF_matched_data",
            new_names=f"{root_name}_seg_map.fits",
            reproject_image_kw={"method": "interp", "order": 0, "compress": False},
        )

    pipes_dir = out_base_dir / "sed_fitting" / "pipes"
    pipes_dir.mkdir(exist_ok=True, parents=True)

    # Create the filter directory; populate as needed
    filter_dir = pipes_dir / "filter_throughputs"
    filter_dir.mkdir(exist_ok=True, parents=True)

    # Create a list of the filters used in our data
    filter_list = []
    for key in info_dict.keys():
        filter_list.append(str(filter_dir / f"{key}.txt"))

    # Create the atlas directory
    atlas_dir = pipes_dir / "atlases"
    atlas_dir.mkdir(exist_ok=True, parents=True)

    obj_id = 2074
    # # # obj_z = 1.369

    # # # for obj_z in np.arange(1.360,1.37, 0.001):
    # # for obj_z in [1.364]:
    obj_z = 1.364
    agn = False

    use_hex = False
    bin_diameter = 3
    target_sn = 10
    sn_filter = "jwst-nircam-f115w"

    from glass_niriss.sed import bin_and_save

    binned_name = f"{obj_id}_{"hexbin" if use_hex else "vorbin"}_{bin_diameter}_{target_sn}_{sn_filter}"
    binned_data_path = bin_data_dir / f"{binned_name}_data.fits"

    # if not binned_data_path.is_file():
    bin_and_save(
        obj_id=obj_id,
        out_dir=bin_data_dir,
        seg_map=repr_seg_path,
        info_dict=info_dict,
        sn_filter=sn_filter,
        target_sn=target_sn,
        bin_diameter=bin_diameter,
        use_hex=use_hex,
        overwrite=False,
        plot=True,
    )

    # with fits.open(
    #     info_dict[sn_filter]["sci"]
    # ) as img_hdul:
    #     with fits.open(repr_seg_path) as seg_hdul:

    #         from glass_niriss.pipeline import seg_slice
    #         obj_img_idxs = seg_slice(seg_hdul[0].data, obj_id)
    #         norm = astrovis.ImageNormalize(
    #             img_hdul[0].data[obj_img_idxs],
    #             interval=astrovis.PercentileInterval(99.9),
    #             stretch=astrovis.LogStretch(),
    #         )

    #         fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    #         seg_hdul[0].data[seg_hdul[0].data==0] = np.nan
    #         axs[0,1].imshow(seg_hdul[0].data[obj_img_idxs]%13, cmap="jet", interpolation="none")
    #         axs[0,1].set_facecolor("k")
    #         axs[0,0].imshow(img_hdul[0].data[obj_img_idxs], norm=norm, origin="lower", cmap="plasma")

    #     with fits.open(bin_data_dir / f"{obj_id}_{"hexbin" if use_hex else "vorbin"}_{bin_diameter}_{target_sn}_{sn_filter}_data.fits") as bin_hdul:
    #         bin_hdul.info()
    #     plt.show()

    from glass_niriss.pipeline import generate_fit_params

    # bagpipes_atlas_params = generate_fit_params(obj_z=obj_z, z_range=0.001)

    bagpipes_atlas_params = generate_fit_params(obj_z=obj_z, z_range=0.001)

    from glass_niriss.sed import AtlasGenerator

    n_samples = 1e5
    n_cores = 16

    remake_atlas = False

    if agn:
        bagpipes_atlas_params["agn"] = {
            "alphalam": -0.5,  # 0.44, Vanden Berk+01
            "betalam": 1,
            "f5100A": (0, 2e-18),
            "sigma": (250, 2500),
            "hanorm": (1e-19, 1e-17),
        }
        run_name = (
            f"z_{bagpipes_atlas_params["redshift"][0]}_"
            f"{bagpipes_atlas_params["redshift"][1]}_"
            f"{n_samples:.2E}_agn_test1"
        )
    else:
        run_name = (
            f"z_{bagpipes_atlas_params["redshift"][0]}_"
            f"{bagpipes_atlas_params["redshift"][1]}_"
            f"{n_samples:.2E}"
        )

    print(bagpipes_atlas_params)
    # exit()

    atlas_path = atlas_dir / f"{run_name}.hdf5"

    if not atlas_path.is_file() or remake_atlas:

        atlas_gen = AtlasGenerator(
            fit_instructions=bagpipes_atlas_params,
            filt_list=filter_list,
            phot_units="ergscma",
        )

        atlas_gen.gen_samples(n_samples=n_samples, parallel=n_cores)

        atlas_gen.write_samples(filepath=atlas_path)

    import os
    from functools import partial

    import numpy as np
    from astropy.table import Table

    from glass_niriss.pipeline import load_photom_bagpipes
    from glass_niriss.sed import AtlasFitter

    os.chdir(pipes_dir)

    overwrite = False

    load_fn = partial(
        load_photom_bagpipes, phot_cat=binned_data_path, cat_hdu_index="PHOT_CAT"
    )

    fit = AtlasFitter(
        fit_instructions=bagpipes_atlas_params,
        atlas_path=atlas_path,
        out_path=pipes_dir.parent,
        overwrite=overwrite,
    )

    obs_table = Table.read(binned_data_path, hdu="PHOT_CAT")
    cat_IDs = np.arange(len(obs_table))[:]

    catalogue_out_path = fit.out_path / Path(f"{binned_name}_{run_name}.fits")
    if (not catalogue_out_path.is_file()) or overwrite:

        fit.fit_catalogue(
            IDs=cat_IDs,
            load_data=load_fn,
            spectrum_exists=False,
            make_plots=False,
            cat_filt_list=filter_list,
            run=f"{binned_name}_{run_name}",
            parallel=8,
            # redshifts = np.zeros(11)+3.06,
            # redshift_range=0.01
        )
        print(fit.cat)
    else:
        fit.cat = Table.read(catalogue_out_path)

    import cmcrameri.cm as cmc
    import matplotlib.pyplot as plt
    from astropy.io import fits

    # fig, axs = plt.subplots(1, 1)
    # seg_map = fits.getdata(binned_data_path, hdu="SEG_MAP")
    # plot_map = np.full_like(seg_map, np.nan, dtype=float)
    # for row in fit.cat:
    #     plot_map[seg_map == int(row["#ID"])] = (
    #         row[
    #             # "continuity:massformed_50"
    #             # "stellar_mass_50"
    #             # "ssfr_50"
    #             "sfr_50"
    #             # "continuity:metallicity_50"
    #             # "mass_weighted_age_50"
    #             # "dust:Av_50"
    #             # "dust:eta"
    #             # "nebular:logU_50"
    #             # "redshift_50"
    #         ]
    #         # *row[
    #         #     "dust:eta_50"
    #         # ]
    #         /
    #         # -
    #         # np.log10(
    #         (len((seg_map == int(row["#ID"])).nonzero()[0])
    #         * ((0.04 * 8.54) ** 2))
    #         # )
    #     )
    # plot_map[seg_map==0] = np.nan
    # im = axs.imshow(
    #     # plot_map,
    #     np.log10(plot_map),
    #     origin="lower",
    #     # vmin=3,
    #     # vmax=9,
    #     # vmin=-4,
    #     # vmax=1,
    #     # vmin=-12,
    #     # vmax=-8,
    #     # vmin=-8,
    #     # vmax=-3,
    #     # cmap="plasma",
    #     # vmin=0,
    #     cmap="rainbow"
    #     # cmap = cmc.lajolla
    # )
    # axs.set_facecolor("k")
    # plt.colorbar(im)

    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, constrained_layout=True)

    seg_map = fits.getdata(binned_data_path, hdu="SEG_MAP")

    mass_map = np.full_like(seg_map, np.nan, dtype=float)
    sfr_map = np.full_like(seg_map, np.nan, dtype=float)
    ssfr_map = np.full_like(seg_map, np.nan, dtype=float)
    av_map = np.full_like(seg_map, np.nan, dtype=float)
    for row in fit.cat:
        mass_map[seg_map == int(row["#ID"])] = row["stellar_mass_50"] - np.log10(
            (len((seg_map == int(row["#ID"])).nonzero()[0]) * ((0.04 * 8.54) ** 2))
        )
        sfr_map[seg_map == int(row["#ID"])] = (
            row["sfr_50"]
            /
            # np.log10(
            (len((seg_map == int(row["#ID"])).nonzero()[0]) * ((0.04 * 8.54) ** 2))
            # )
        )
        ssfr_map[seg_map == int(row["#ID"])] = row["ssfr_50"] - np.log10(
            (len((seg_map == int(row["#ID"])).nonzero()[0]) * ((0.04 * 8.54) ** 2))
        )
        av_map[seg_map == int(row["#ID"])] = (
            row["dust:Av_50"]
            # -
            # np.log10(
            # (len((seg_map == int(row["#ID"])).nonzero()[0])
            # * ((0.04 * 8.54) ** 2))
            # )
        )
    mass_map[seg_map == 0] = np.nan
    sfr_map[seg_map == 0] = np.nan
    ssfr_map[seg_map == 0] = np.nan
    av_map[seg_map == 0] = np.nan
    im_mass = axs[0, 0].imshow(mass_map, origin="lower", cmap="plasma")
    plt.colorbar(im_mass, ax=axs[0, 0], label="M*/kpc2")
    im_sfr = axs[0, 1].imshow(np.log10(sfr_map), origin="lower", cmap="plasma")
    plt.colorbar(im_sfr, ax=axs[0, 1], label="SFR/kpc2")
    im_ssfr = axs[1, 0].imshow(ssfr_map, origin="lower", cmap="plasma")
    plt.colorbar(im_ssfr, ax=axs[1, 0], label="sSFR/kpc2")
    im_av = axs[1, 1].imshow(av_map, origin="lower", cmap="plasma")
    plt.colorbar(im_av, ax=axs[1, 1], label="Av")
    for a in axs.flatten():
        # print (a)
        a.set_facecolor("k")
        a.set_xlabel("")
        a.set_ylabel("")
        a.set_xticklabels([])
        a.set_yticklabels([])

    plt.show()
