from pathlib import Path

import yaml

root_dir = Path("/media") / "sharedData" / "data"
out_base_dir = root_dir / "2024_08_16_A2744_v4" / "glass_niriss"
bin_data_dir = out_base_dir / "binned_data"

with open(out_base_dir / "conv_ancillary_data.yaml", "r") as file:
    info_dict = yaml.safe_load(file)

from glass_niriss.isophotal import reproject_and_convolve

ref_mosaic = (
    root_dir / "2023_11_07_spectral_orders" / "Prep" / "nis-wfss-ir_drz_sci.fits"
)

# The path of the original segmentation map
orig_seg = (
    out_base_dir.parent
    / "grizli_home"
    / "tests"
    / "Prep_tests"
    / "glass-a2744-ir_seg.fits"
)

reproject_and_convolve(
    ref_path=ref_mosaic,
    orig_images=orig_seg,
    psfs=None,
    psf_target=None,
    out_dir=out_base_dir / "PSF_matched_data",
    new_names="glass-a2744_seg_map.fits",
    reproject_image_kw={"method": "interp", "order": 0, "compress": False},
)

repr_seg_path = out_base_dir / "PSF_matched_data" / "glass-a2744_seg_map.fits"

# Bagpipes directory
# Will include filters and atlases
pipes_dir = out_base_dir / "sed_fitting" / "pipes"
pipes_dir.mkdir(exist_ok=True, parents=True)

filter_dir = pipes_dir / "filter_throughputs"
filter_dir.mkdir(exist_ok=True, parents=True)

filter_list = []
for key in info_dict.keys():
    filter_list.append(str(filter_dir / f"{key}.txt"))

atlas_dir = pipes_dir / "atlases"
atlas_dir.mkdir(exist_ok=True, parents=True)


obj_id = 1761
obj_z = 3.06
obj_id = 1742
obj_z = 3.06
# obj_id = 3311
# obj_z = 1.34
# obj_id = 1597
# obj_z = 2.6724
# obj_id = 886
# obj_z = 0.3033

use_hex = False
bin_diameter = 4
target_sn = 50
sn_filter = "jwst-nircam-f200w"

from glass_niriss.sed import bin_and_save

binned_name = f"{obj_id}_{"hexbin" if use_hex else "vorbin"}_{bin_diameter}_{target_sn}"
binned_data_path = bin_data_dir / f"{binned_name}_data.fits"

if not binned_data_path.is_file():
    bin_and_save(
        obj_id=obj_id,
        out_dir=bin_data_dir,
        seg_map=repr_seg_path,
        info_dict=info_dict,
        sn_filter=sn_filter,
        target_sn=target_sn,
        bin_diameter=bin_diameter,
        use_hex=use_hex,
        overwrite=True,
    )

from glass_niriss.pipeline import generate_fit_params

bagpipes_atlas_params = generate_fit_params(obj_z=obj_z)

print(bagpipes_atlas_params)

from glass_niriss.sed import AtlasGenerator

n_samples = 1e5
n_cores = 16

remake_atlas = False
run_name = (
    f"z_{bagpipes_atlas_params["redshift"][0]}_"
    f"{bagpipes_atlas_params["redshift"][1]}_"
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

import os
from functools import partial

import numpy as np
from astropy.table import Table

from glass_niriss.pipeline import load_photom_bagpipes
from glass_niriss.sed import AtlasFitter

os.chdir(pipes_dir)

load_fn = partial(
    load_photom_bagpipes, phot_cat=binned_data_path, cat_hdu_index="PHOT_CAT"
)

fit = AtlasFitter(
    fit_instructions=bagpipes_atlas_params,
    atlas_path=atlas_path,
    out_path=pipes_dir.parent,
    overwrite=False,
)

obs_table = Table.read(binned_data_path, hdu="PHOT_CAT")
cat_IDs = np.arange(len(obs_table))[:]

catalogue_out_path = fit.out_path / Path(f"{binned_name}_{run_name}.fits")
print(fit.overwrite)
if (not catalogue_out_path.is_file()) or (fit.overwrite):

    fit.fit_catalogue(
        IDs=cat_IDs,
        load_data=load_fn,
        spectrum_exists=False,
        make_plots=False,
        cat_filt_list=filter_list,
        run=f"{binned_name}_{run_name}",
        parallel=8,
    )
    print(fit.cat)
else:
    fit.cat = Table.read(catalogue_out_path)

import logging

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
from astropy.io import fits
from grizli import jwst_utils

from glass_niriss.grism import RegionsMultiBeam

jwst_utils.QUIET_LEVEL = logging.WARNING
jwst_utils.set_quiet_logging(jwst_utils.QUIET_LEVEL)

grizli_extraction_dir = (
    root_dir / "2024_08_16_A2744_v4" / "grizli_home" / "tests" / "Extractions"
)

beams_path = [*grizli_extraction_dir.glob(f"*{obj_id}.beams.fits")]
if len(beams_path) >= 1:
    beams_path = beams_path[0]
else:
    raise IOError("Beams file does not exist.")

if __name__ == "__main__":
    multib = RegionsMultiBeam(
        binned_data=binned_data_path,
        pipes_dir=pipes_dir,
        # f"bcg_{obj_id}_{bin_mode}_{bin_size}_{sn_target}_z_{obj_z}_{obj_z}_{atlas_size:.2E}",
        run_name=f"{binned_name}_{run_name}",
        beams=str(beams_path),
        min_mask=0.0,
        min_sens=0.0,
        mask_resid=False,
        verbose=False,
    )

    multib.fit_at_z(
        z=obj_z,
        n_samples=3,
        veldisp=500,
        oversamp_factor=3,
        fit_stacks=True,
        temp_dir=Path("/media/sharedData/data/2024_08_16_A2744_v4/tests"),
        # direct_images=direct_images, poly_order=3
        cpu_count=16,
    )
