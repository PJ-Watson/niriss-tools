import os
from pathlib import Path

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

# Where the grism data can be found
grizli_extraction_dir = out_base_dir.parent / "grizli_home" / "Extractions"

from glass_niriss.isophotal import reproject_and_convolve

repr_seg_path = out_base_dir / "PSF_matched_data" / f"{root_name}_seg_map.fits"

if not repr_seg_path.is_file():
    # Whichever mosaic we used as a reference for the photometry
    ref_mosaic = grizli_extraction_dir.parent / "Prep" / f"{root_name}-ir_drc_sci.fits"

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


# Create the [bag]pipes directory
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


obj_id = 1761
obj_z = 3.06
# obj_id = 497
# obj_z = 0.3033
obj_id = 1597
obj_z = 2.6724

# obj_id = 3311
# obj_z = 1.34
obj_id = 2606
obj_z = 0.296

# # obj_id = 497
# # obj_z = 0.30
# # obj_id = 2224
# # obj_z = 0.3064
# obj_id = 1742
# obj_z = 3.06
# obj_id = 908
# obj_z = 0.3033
# obj_id = 3278
# obj_z = 0.296
# obj_id = 2328
# obj_z = 1.363
# obj_id = 2720
# obj_z = 3.04
# obj_id = 5021
# obj_z = 1.8868
# obj_id = 3137
# obj_z = 0.9384

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

n_samples = 1e6
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
