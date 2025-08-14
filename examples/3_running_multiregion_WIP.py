"""An example workflow for performing a multi-region fit to NIRISS data."""

import os
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from bagpipes_extended.pipeline import generate_fit_params
from bagpipes_extended.sed import AtlasFitter, AtlasGenerator

import niriss_tools
from niriss_tools.grism import MultiRegionFit, align_direct_images, gen_stacked_beams
from niriss_tools.pipeline import load_photom_bagpipes
from niriss_tools.sed import bin_and_save

default_filter_dir = Path(niriss_tools.__file__).parent / "data" / "filter_throughputs"

# Or similar
root_dir = Path(os.getenv("ROOT_DIR"))
field_name = "glass-a2744"
proposal_ID = 1324
reduction_dir = Path(root_dir) / f"2025_08_05_{field_name}"

if __name__ == "__main__":
    out_base_dir = reduction_dir / "glass_niriss_bcgs"

    # Setup a separate directory for the binned data
    bin_data_dir = out_base_dir / "binned_data"

    # Where the grism data can be found
    grizli_extraction_dir = reduction_dir / "grizli_home" / "Extractions"

    # Create the [bag]pipes directory
    pipes_dir = out_base_dir / "sed_fitting" / "pipes"
    pipes_dir.mkdir(exist_ok=True, parents=True)

    # Load the file containing all the info on photometric data
    with open(out_base_dir / "PSF_matched_photometry.yaml", "r") as file:
        info_dict = yaml.safe_load(file)

    # Create the filter directory; populate as needed
    filter_dir = pipes_dir / "filter_throughputs"
    filter_dir.mkdir(exist_ok=True, parents=True)
    for file in default_filter_dir.glob("*.txt"):
        shutil.copy(file, filter_dir)

    # Create a list of the filters used in our data
    filter_list = []
    for key in info_dict.keys():
        filter_list.append(str(filter_dir / f"{key}.txt"))

    # Create the atlas directory
    atlas_dir = pipes_dir / "atlases"
    atlas_dir.mkdir(exist_ok=True, parents=True)

    obj_id = 2074
    # Load the redshift catalogue? Or just set it manually
    # redshift_cat = Table.read(
    #     root_dir
    #     / "2024_08_16_A2744_v4"
    #     / "grizli_home"
    #     / "Extractions_v4"
    #     / "catalogues"
    #     / "stage_7"
    #     / "hlsp_glass-jwst_jwst_niriss_abell2744_v1.0_cat.fits"
    # )
    # obj_z = np.round(
    #     float(redshift_cat[redshift_cat["ID_NIRISS"] == obj_id]["Z_NIRISS"]), 3
    # )
    obj_z = 1.368

    bin_scheme = "colour"
    bin_diameter = 3
    target_sn = 10
    sn_filter = "jwst-nircam-f150w"

    bagpipes_atlas_params = generate_fit_params(
        obj_z=obj_z,
        z_range=0.005,
        num_age_bins=5,
        sfh_type="continuity",
        min_age_bin=20,
    )

    # On a 2023 16-core laptop, generating 1e6 samples will take ~5 minutes
    n_samples = 1e5
    n_cores = 16

    remake_atlas = False
    run_name = (
        f"z_{bagpipes_atlas_params["redshift"][0]:.3f}_"
        f"{bagpipes_atlas_params["redshift"][1]:.3f}_"
        f"{n_samples:.2E}"
    )
    atlas_path = atlas_dir / f"{run_name}.hdf5"

    if not atlas_path.is_file() or remake_atlas:

        atlas_gen = AtlasGenerator(
            fit_instructions=bagpipes_atlas_params,
            filt_list=filter_list,
            phot_units="ergscma",
        )

        atlas_gen.gen_samples(n_samples=n_samples, parallel=n_cores)

        atlas_gen.write_samples(filepath=atlas_path)

    # exit()
    new_beam_loc = (
        bin_data_dir / f"{obj_id:0>5}" / f"{field_name}_{obj_id:0>5}.beams.fits"
    )
    new_beam_loc.parent.mkdir(exist_ok=True, parents=True)

    # Give a descriptive name for the binned data
    binned_name = f"{obj_id}_{bin_scheme}_{bin_diameter}_{target_sn}_{sn_filter}"
    binned_data_path = bin_data_dir / f"{obj_id:0>5}" / f"{binned_name}_data.fits"
    if not (binned_data_path.is_file() and new_beam_loc.is_file()):

        beams_path = [*grizli_extraction_dir.glob(f"**/*{obj_id:0>5}.beams.fits")]
        if len(beams_path) >= 1:
            beams_path = beams_path[0]
        else:
            raise IOError("Original beams file does not exist.")

        multib = gen_stacked_beams(
            str(beams_path),
            min_mask=0.0,
            min_sens=0.0,
            mask_resid=False,
            verbose=False,
            fcontam=0.2,
            group_name="glass-a2744",
        )
        multib.fit_trace_shift()

        # Write the realigned and stacked beam to a file
        stacked_beam_hdul = multib.write_master_fits(get_hdu=True)
        stacked_beam_hdul.writeto(new_beam_loc, overwrite=True)

        # Align all images to the new stacked beam
        aligned_info_dict = align_direct_images(
            multib.beams[0],
            info_dict=info_dict,
            out_dir=bin_data_dir / f"{obj_id:0>5}",
            overwrite=False,
            cutout=500,
        )

        _seg = fits.getdata(new_beam_loc, "SEG")
        bin_and_save(
            obj_id=obj_id,
            out_dir=bin_data_dir / f"{obj_id:0>5}",
            # seg_map=multib.beams[0].beam.seg,
            seg_map=_seg,
            info_dict=aligned_info_dict,
            sn_filter=sn_filter,
            target_sn=target_sn,
            bin_diameter=bin_diameter,
            bin_scheme=bin_scheme,
            overwrite=True,
            plot=True,
        )

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

    run_name = (
        f"z_{bagpipes_atlas_params["redshift"][0]}_"
        f"{bagpipes_atlas_params["redshift"][1]}_"
        f"{n_samples:.2E}_aligned"
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
            redshifts=obj_z,
            redshift_range=0.0025,
            n_posterior=500,
        )
        print(fit.cat)
    else:
        fit.cat = Table.read(catalogue_out_path)

    multireg = MultiRegionFit(
        binned_data=binned_data_path,
        pipes_dir=pipes_dir,
        # f"bcg_{obj_id}_{bin_mode}_{bin_size}_{sn_target}_z_{obj_z}_{obj_z}_{atlas_size:.2E}",
        run_name=f"{binned_name}_{run_name}",
        beams=str(new_beam_loc),
        # beams=multib.beams,
        min_mask=0.0,
        min_sens=0.0,
        mask_resid=False,
        verbose=False,
        fcontam=0.2,
    )
    multireg.MB.fit_trace_shift()

    multireg.fit_at_z(
        z=obj_z,
        n_samples=5,
        veldisp=500,
        oversamp_factor=1,
        n_iters=10,
        cpu_count=16,
        # memmap=True,
        nnls_iters=[1e3, 1e5],
        nnls_tol=[1e-6, 1e-10],
        nnls_method="adelie",
        # overwrite=True,
        save_lines=True,
        save_stacks=True,
        overwrite=False,
        out_dir=out_base_dir / "multiregion" / "tests_v2" / f"{obj_id}",
        n_shifted=3,
        # use_lines=[
        #     {
        #         "cloudy": [
        #             "H  1  6562.81A",
        #             "N  2  6583.45A",
        #             "N  2  6548.05A",
        #         ],
        #         "grizli": "Ha",
        #         "wave": 6564.697,
        #     },
        #     {
        #         "cloudy": ["H  1  4861.33A"],
        #         "grizli": "Hb",
        #         "wave": 4862.738,
        #     },
        # ],
    )
