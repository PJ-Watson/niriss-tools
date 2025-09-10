"""
Functions and classes related to multi-region grism fitting.
"""

import multiprocessing
import sys
import traceback
from copy import deepcopy
from functools import partial
from itertools import product
from multiprocessing import Manager, Pool, cpu_count, shared_memory
from multiprocessing.managers import SharedMemoryManager
import os
from os import PathLike
from pathlib import Path
from time import time
import shutil
from collections.abc import Callable


import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.table import Table
from astropy.wcs import WCS
from grizli import utils as grizli_utils
from grizli.model import GrismDisperser
from grizli.multifit import MultiBeam, drizzle_to_wavelength
from numpy.typing import ArrayLike
from reproject import reproject_interp

import niriss_tools
from niriss_tools.grism.fitting_tools import CDNNLS
from niriss_tools.grism.specgen import (
    CLOUDY_LINE_MAP,
    BagpipesSampler,
    ExtendedModelGalaxy,
    check_coverage,
)
from niriss_tools.grism.utils import align_direct_images, gen_stacked_beams
from niriss_tools.pipeline.reduction import recursive_merge
from niriss_tools.sed.binning import bin_and_save

from bagpipes_extended.pipeline import generate_fit_params, load_photom_bagpipes
from bagpipes_extended import AtlasGenerator, AtlasFitter

__all__ = ["MultiRegionFit", "DEFAULT_PLINE"]

pipes_sampler = None
beams_object = None

DEFAULT_PLINE = {
    "pixscale": 0.06,
    "pixfrac": 1.0,
    "size": 5,
    "kernel": "lanczos3",
}


def _init_pipes_sampler(fit_instructions, veldisp, beams):
    global pipes_sampler
    pipes_sampler = BagpipesSampler(fit_instructions=fit_instructions, veldisp=veldisp)
    global beams_object
    beams_object = beams.copy()


class MultiRegionFit:
    """
    A multi-region version of `grizli.multifit.MultiBeam`.

    Parameters
    ----------
    binned_data : PathLike
        The file containing the binned photometric catalogue.
    pipes_dir : PathLike
        The directory in which the `bagpipes` posterior distributions are
        saved.
    run_name : str
        The name of the bagpipes run used to fit the galaxy.
    beams : list
        List of `~grizli.model.BeamCutout` objects.
    **multibeam_kwargs : dict, optional
        Any additional parameters to pass through to
        `grizli.multifit.MultiBeam`.
    """

    def __init__(
        self,
        config_path: PathLike,
        obj_id: int,
        obj_z: float,
        run_all: bool = True,
        # binned_data: PathLike,
        # pipes_dir: PathLike,
        # run_name: str,
        # beams: list,
        # **multibeam_kwargs,
    ):

        self.obj_id = obj_id
        self.obj_z = obj_z

        self.import_config(config_path)

        if run_all:
            self.beam_path, self.binned_data_path = self.gen_aligned_photometry(
                binning_kwargs=self.binning_kwargs, use_stacks=self.use_stacks, **self.multibeam_kwargs
            )
            self.atlas_path = self.gen_atlas(**self.sed_fit_kwargs)
            self.fit_atlas(
                self.atlas_path,
                self.binned_data_path,
                overwrite_fit=self.overwrite_atlas_fit,
                n_cores=self.n_cores_atlas_fit,
                z_range=self.sed_fit_kwargs["z_range"],
            )
            
            self.MB = MultiBeam(beams = str(self.beam_path), **self.multibeam_kwargs)
            self.ra, self.dec = self.MB.ra, self.MB.dec

            self.regions_phot_cat = Table.read(self.binned_data_path, "PHOT_CAT")
            self.regions_phot_cat = self.regions_phot_cat[
                self.regions_phot_cat["bin_id"].astype(int) != 0
            ]
            self.n_regions = len(self.regions_phot_cat)

            with fits.open(self.binned_data_path) as hdul:
                self.regions_seg_map = hdul["SEG_MAP"].data.copy()
                self.regions_seg_hdr = hdul["SEG_MAP"].header.copy()
                self.regions_seg_wcs = WCS(self.regions_seg_hdr)

            self.regions_seg_ids = np.asarray(self.regions_phot_cat["bin_id"], dtype=int)

            self.fit_at_z(self.obj_z, **self.grism_fit_kwargs)

        # self.MB = MultiBeam(beams=beams, **multibeam_kwargs)
        # self.id, self.ra, self.dec = self.MB.id, self.MB.ra, self.MB.dec

        # self.binned_data = binned_data
        # self.regions_phot_cat = Table.read(self.binned_data, "PHOT_CAT")
        # self.regions_phot_cat = self.regions_phot_cat[
        #     self.regions_phot_cat["bin_id"].astype(int) != 0
        # ]
        # self.n_regions = len(self.regions_phot_cat)

        # with fits.open(binned_data) as hdul:
        #     self.regions_seg_map = hdul["SEG_MAP"].data.copy()
        #     self.regions_seg_hdr = hdul["SEG_MAP"].header.copy()
        #     self.regions_seg_wcs = WCS(self.regions_seg_hdr)

        # self.regions_seg_ids = np.asarray(self.regions_phot_cat["bin_id"], dtype=int)

        # self.run_name = run_name
        # self.pipes_dir = pipes_dir

    def import_config(self, config_path: PathLike):

        import tomllib
        import warnings

        import yaml

        with open(config_path, "rb") as f:
            config = tomllib.load(f)

        self.root_dir = Path(config["files"].get("root_dir", "/"))
        self.out_dir = self.root_dir / config["files"].get("out_dir", ".")
        self.out_dir.mkdir(exist_ok=True, parents=True)
        self.extractions_dir = self.root_dir / config["files"].get(
            "extractions_dir", "."
        )

        _info_dict = config["files"].get("info_dict", "")
        if not _info_dict:
            raise ValueError("`info_dict` is not present in the supplied config file.")

        with open(self.root_dir / _info_dict, "r") as file:
            self.info_dict = yaml.safe_load(file)

        self.pipes_dir = self.out_dir / "sed_fitting" / "pipes"
        self.pipes_dir.mkdir(exist_ok=True, parents=True)
        if not config["files"].get("atlas_dir", ""):
            self.atlas_dir = self.pipes_dir / "atlases"
        else:
            self.atlas_dir = self.root_dir / config["files"]["atlas_dir"]
        self.atlas_dir.mkdir(exist_ok=True, parents=True)

        self.binning_kwargs = {
            "bin_scheme": config["SED"].get("bin_scheme", "colour"),
            "target_sn": config["SED"].get("target_sn", 10),
            "bin_diameter": config["SED"].get("bin_diameter", 3),
            "sn_filter": config["SED"]["sn_filter"],
            **config["SED"].get("bin_kwargs", {}),
        }

        self.sed_fit_kwargs = {
            "z_range": config["SED"].get("z_range", 0.005),
            "sfh_type": config["SED"].get("sfh_type", "continuity"),
            "min_age_bin": config["SED"].get("min_age_bin", 20),
            "num_age_bins": config["SED"].get("num_age_bins", 5),
            "n_samples": config["SED"].get("n_samples", 1e6),
            "remake_atlas": config["SED"].get("remake_atlas", False),
            "n_cores_atlas": config["SED"].get("n_cores_atlas", 4),
            # "bin_kwargs": config["SED"].get("bin_kwargs", {}),
        }

        self.n_cores_atlas_fit = config["SED"].get("n_cores_fit", 4)
        self.overwrite_atlas_fit = config["SED"].get("overwrite_fit", False)

        self.field_name = config["grism"]["field_name"]

        self.use_stacks = config["grism"].get("use_stacks", True)

        multibeam_kwargs = config["grism"].get("multibeam_kwargs", {})

        default_multib_kwargs = {
            "min_mask": 0.0,
            "min_sens": 0.0,
            "mask_resid": False,
            "verbose": False,
            "fcontam": 0.2,
            "group_name": self.field_name,
        }
        self.multibeam_kwargs = recursive_merge(default_multib_kwargs, multibeam_kwargs)

        self.grism_fit_kwargs = {
            "fit_background": config["grism"].get("fit_background", True),
            "poly_order": config["grism"].get("poly_order", 0),
            "n_samples": config["grism"].get("n_region_samples", 3),
            "n_iters": config["grism"].get("n_iters", 10),
            "spec_wavs": config["grism"].get("spec_wavs", None),
            "oversamp_factor": config["grism"].get("oversamp_factor", 1),
            "veldisp": config["grism"].get("veldisp", 500),
            "out_dir": config["grism"].get("out_dir", "multiregion"),
            "temp_dir": config["grism"].get("temp_dir", None),
            "memmap": config["grism"].get("memmap", False),
            "cpu_count": config["grism"].get("cpu_count", -1),
            "overwrite": config["grism"].get("overwrite", False),
            "use_lines": config["grism"].get("use_lines",CLOUDY_LINE_MAP),
            "save_lines": config["grism"].get("save_lines", True),
            "save_stacks": config["grism"].get("save_stacks", True),
            "pline": config["grism"].get("pline", DEFAULT_PLINE),
            "seed": config["grism"].get("seed", 2744),
            "nnls_method": config["grism"].get("nnls_method", "scipy"),
            "nnls_iters": config["grism"].get("nnls_iters", 10),
            "nnls_tol": config["grism"].get("nnls_tol", 1e-5),
            "n_shifted": config["grism"].get("n_shifted", 2),
            "n_shifted_samples": config["grism"].get("n_shifted_samples", 1),
        }

    def fit_atlas(
        self,
        atlas_path: PathLike,
        binned_data_path: PathLike,
        binned_data_hdu: str | int | None = "PHOT_CAT",
        bagpipes_atlas_params: dict | None = None,
        load_fn: Callable | None = None,
        overwrite_fit: bool = False,
        id_colname: str = "bin_id",
        n_cores: int = 4,
        obj_z: float | ArrayLike | None = None,
        z_range: float = 0.005,
    ):

        if bagpipes_atlas_params is None:
            bagpipes_atlas_params = self.bagpipes_atlas_params

        os.chdir(self.pipes_dir)

        if load_fn is None:

            load_fn = partial(
                load_photom_bagpipes,
                phot_cat=binned_data_path,
                cat_hdu_index=binned_data_hdu,
            )

        fit = AtlasFitter(
            fit_instructions=bagpipes_atlas_params,
            atlas_path=atlas_path,
            out_path=self.pipes_dir.parent,
            overwrite=overwrite_fit,
        )

        # run_name = (
        #     f"z_{bagpipes_atlas_params["redshift"][0]}_"
        #     f"{bagpipes_atlas_params["redshift"][1]}_"
        #     f"{n_samples:.2E}_aligned"
        # )
        self.run_name = str(Path(binned_data_path).stem).removesuffix("_data")

        obs_table = Table.read(binned_data_path, hdu=binned_data_hdu)
        cat_IDs = np.array(obs_table[id_colname])

        # catalogue_out_path = fit.out_path / Path(f"{binned_name}_{run_name}.fits")
        catalogue_out_path = fit.out_path / f"{self.run_name}.fits"
        if (not catalogue_out_path.is_file()) or overwrite_fit:

            fit.fit_catalogue(
                IDs=cat_IDs,
                load_data=load_fn,
                spectrum_exists=False,
                make_plots=False,
                cat_filt_list=self.filter_list,
                run=self.run_name,
                parallel=n_cores,
                redshifts=self.obj_z if obj_z is None else obj_z,
                redshift_range=z_range,
                n_posterior=500,
            )
            # print(fit.cat)
        else:
            fit.cat = Table.read(catalogue_out_path)

        print(fit.cat)

    def gen_atlas(
        self,
        obj_z: float | ArrayLike | None = None,
        z_range: float = 0.005,
        num_age_bins: int = 5,
        min_age_bin: float = 20,
        sfh_type: str = "continuity",
        n_samples: int | float = 1e6,
        remake_atlas: bool = False,
        n_cores_atlas: int = 4,
    ):

        default_filter_dir = (
            Path(niriss_tools.__file__).parent / "data" / "filter_throughputs"
        )

        # Create the filter directory; populate as needed
        filter_dir = self.pipes_dir / "filter_throughputs"
        filter_dir.mkdir(exist_ok=True, parents=True)
        for file in default_filter_dir.glob("*.txt"):
            shutil.copy(file, filter_dir)

        # Create a list of the filters used in our data
        self.filter_list = []
        for key in self.info_dict.keys():
            self.filter_list.append(str(filter_dir / f"{key}.txt"))

        self.bagpipes_atlas_params = generate_fit_params(
            obj_z=self.obj_z if obj_z is None else obj_z,
            z_range=z_range,
            num_age_bins=num_age_bins,
            sfh_type=sfh_type,
            min_age_bin=min_age_bin,
        )

        self.atlas_run_name = (
            f"z_{self.bagpipes_atlas_params["redshift"][0]:.4f}_"
            f"{self.bagpipes_atlas_params["redshift"][1]:.4f}_"
            f"{n_samples:.2E}"
        )
        atlas_path = self.atlas_dir / f"{self.atlas_run_name}.hdf5"

        if not atlas_path.is_file() or remake_atlas:

            atlas_gen = AtlasGenerator(
                fit_instructions=self.bagpipes_atlas_params,
                filt_list=self.filter_list,
                phot_units="ergscma",
            )

            atlas_gen.gen_samples(n_samples=n_samples, parallel=n_cores_atlas)

            atlas_gen.write_samples(filepath=atlas_path)

        return atlas_path

    def gen_aligned_photometry(
        self,
        binning_kwargs={},
        use_stacks: bool = True,
        beams_path=None,
        img_cutout=500,
        **multibeam_kwargs,
    ):

        binned_data_dir = self.out_dir / "binned_data"
        binned_data_dir.mkdir(exist_ok=True, parents=True)

        new_beam_loc = (
            binned_data_dir / f"{self.field_name}_{self.obj_id:0>5}.beams.fits"
        )

        # Give a descriptive name for the binned data
        binned_name = (
            f"{self.obj_id}_{binning_kwargs["bin_scheme"]}_"
            f"{binning_kwargs["bin_diameter"]}_{binning_kwargs["target_sn"]}"
            f"_{binning_kwargs["sn_filter"]}"
        )
        binned_data_path = binned_data_dir / f"{binned_name}_data.fits"

        if not (binned_data_path.is_file() and new_beam_loc.is_file()):

            if beams_path is None:

                beams_path = [
                    *self.extractions_dir.glob(f"**/*{self.obj_id:0>5}.beams.fits")
                ]
                if len(beams_path) >= 1:
                    beams_path = str(beams_path[0])
                else:
                    raise IOError(
                        f"Original beams file does not exist in {self.extractions_dir}"
                    )

            default_multib_kwargs = {
                "min_mask": 0.0,
                "min_sens": 0.0,
                "mask_resid": False,
                "verbose": False,
                "fcontam": 0.2,
                "group_name": self.field_name,
            }
            new_kwargs = recursive_merge(default_multib_kwargs, multibeam_kwargs)

            if use_stacks:
                multib = gen_stacked_beams(
                    beams_path,
                    **new_kwargs,
                )
            else:
                multib = MultiBeam(
                    beams_path,
                    **new_kwargs,
                )

            # Write the realigned and (stacked?) beam to a file
            beam_hdul = multib.write_master_fits(get_hdu=True)
            beam_hdul.writeto(new_beam_loc, overwrite=True)

            # Align all images to the new beam
            aligned_info_dict = align_direct_images(
                multib.beams[0],
                info_dict=self.info_dict,
                out_dir=binned_data_dir / f"{self.obj_id:0>5}",
                overwrite=False,
                cutout=img_cutout,
            )

            _seg = fits.getdata(new_beam_loc, "SEG")
            bin_and_save(
                obj_id=self.obj_id,
                out_dir=binned_data_dir,  # / f"{obj_id:0>5}",
                # seg_map=multib.beams[0].beam.seg,
                seg_map=_seg,
                info_dict=aligned_info_dict,
                binned_name=binned_name,
                **binning_kwargs,
            )

        return new_beam_loc, binned_data_path

    def add_pipes_info(self, header: fits.Header) -> fits.Header:
        """
        Update a header with information about the 2D SED fitting.

        Parameters
        ----------
        header : fits.Header
            The original header.

        Returns
        -------
        fits.Header
            The updated header.
        """
        header["MRBPRUN"] = (
            str(self.run_name),
            "The name of the bagpipes run used to generate the prior templates.",
        )
        header["MRBPPCAT"] = (
            str(self.binned_data_path),
            "The binned photometric catalogue and segmentation map used as input for bagpipes.",
        )
        return header

    @staticmethod
    def _gen_stacked_templates_from_pipes(
        seg_idx=None,
        seg_id=None,
        shared_seg_name=None,
        seg_maps_shape=None,
        shared_models_name=None,
        models_shape=None,
        posterior_dir=None,
        n_samples=None,
        spec_wavs=None,
        beam_info=None,
        temp_offset=None,
        cont_only=False,
        rm_line=None,
        rows=None,
        memmap: bool = False,
        id_shifts=None,
        n_shifted_rows=None,
    ):
        seg_id = int(seg_id)

        try:

            if memmap:
                seg_maps = np.memmap(
                    shared_seg_name, dtype=float, shape=seg_maps_shape, mode="r+"
                )
                temps_arr = np.memmap(
                    shared_models_name, dtype=float, shape=models_shape, mode="r+"
                )
            else:
                shm_seg_maps = shared_memory.SharedMemory(name=shared_seg_name)
                seg_maps = np.ndarray(
                    seg_maps_shape, dtype=float, buffer=shm_seg_maps.buf
                )
                shm_temps = shared_memory.SharedMemory(name=shared_models_name)
                temps_arr = np.ndarray(models_shape, dtype=float, buffer=shm_temps.buf)

            with h5py.File(Path(posterior_dir) / f"{seg_id}.h5", "r") as post_file:
                samples2d = np.array(post_file["samples2d"])[rows]

            # Just make one set of posterior samples from all possible seg ids
            if (id_shifts is not None) and (len(id_shifts) > 0):
                for s in id_shifts:
                    shifted_id = int((seg_id + s) % np.nanmax(seg_maps[0]))
                    with h5py.File(
                        Path(posterior_dir) / f"{shifted_id}.h5", "r"
                    ) as post_file:
                        samples2d = np.concatenate(
                            (
                                samples2d,
                                np.array(post_file["samples2d"])[
                                    rows[:n_shifted_rows]
                                ].reshape(n_shifted_rows, -1),
                            ),
                            axis=0,
                        )

            for sample_i, sample in enumerate(samples2d):

                temp_resamp_1d = pipes_sampler.sample(
                    sample,
                    spec_wavs=spec_wavs,
                    cont_only=cont_only,
                    rm_line=rm_line,
                )

                i0 = 0
                for k_i, (k, v) in enumerate(beam_info.items()):
                    for ib in v["list_idx"]:
                        beam = beams_object[ib]
                        if seg_maps.ndim == 3:
                            beam_seg = seg_maps[ib] == seg_id
                        else:
                            beam_seg = seg_maps[seg_idx][ib]
                        tmodel = beam.compute_model(
                            spectrum_1d=temp_resamp_1d,
                            thumb=beam.beam.direct * beam_seg,
                            in_place=False,
                            is_cgs=True,
                        )
                        temps_arr[
                            (n_samples * seg_idx) + sample_i + temp_offset,
                            i0 : i0 + np.prod(v["2d_shape"]),
                        ] += tmodel

                    temps_arr[
                        (n_samples * seg_idx) + sample_i + temp_offset,
                        i0 : i0 + np.prod(v["2d_shape"]),
                    ] /= len(v["list_idx"])

                    i0 += np.prod(v["2d_shape"])

            if memmap:
                temps_arr.flush()

        except:
            raise Exception(
                shared_models_name
                + "".join(traceback.format_exception(*sys.exc_info()))
            )

    @staticmethod
    def _gen_beam_templates_from_pipes(
        seg_idx=None,
        seg_id=None,
        shared_seg_name=None,
        seg_maps_shape=None,
        shared_models_name=None,
        models_shape=None,
        posterior_dir=None,
        spec_wavs=None,
        beam_info=None,
        cont_only=False,
        rm_line=None,
        rows=None,
        coeffs=None,
        memmap: bool = False,
        id_shifts=None,
        n_shifted_rows=None,
    ):
        seg_id = int(seg_id)

        try:

            if memmap:
                seg_maps = np.memmap(
                    shared_seg_name, dtype=float, shape=seg_maps_shape, mode="r+"
                )
                models_arr = np.memmap(
                    shared_models_name, dtype=float, shape=models_shape, mode="r+"
                )
            else:
                shm_seg_maps = shared_memory.SharedMemory(name=shared_seg_name)
                seg_maps = np.ndarray(
                    seg_maps_shape, dtype=float, buffer=shm_seg_maps.buf
                )
                shm_models = shared_memory.SharedMemory(name=shared_models_name)
                models_arr = np.ndarray(
                    models_shape, dtype=float, buffer=shm_models.buf
                )

            with h5py.File(Path(posterior_dir) / f"{seg_id}.h5", "r") as post_file:
                samples2d = np.array(post_file["samples2d"])[rows]

            # Just make one set of posterior samples from all possible seg ids
            if (id_shifts is not None) and (len(id_shifts) > 0):
                for s in id_shifts:
                    shifted_id = int((seg_id + s) % np.nanmax(seg_maps[0]))
                    with h5py.File(
                        Path(posterior_dir) / f"{shifted_id}.h5", "r"
                    ) as post_file:
                        samples2d = np.concatenate(
                            (
                                samples2d,
                                np.array(post_file["samples2d"])[
                                    rows[:n_shifted_rows]
                                ].reshape(n_shifted_rows, -1),
                            ),
                            axis=0,
                        )

            for sample_i, sample in enumerate(samples2d):

                temp_resamp_1d = pipes_sampler.sample(
                    sample,
                    spec_wavs=spec_wavs,
                    cont_only=cont_only,
                    rm_line=rm_line,
                )

                i0 = 0
                for k_i, (k, v) in enumerate(beam_info.items()):
                    for ib in v["list_idx"]:
                        if seg_maps.ndim == 3:
                            beam_seg = seg_maps[ib] == seg_id
                        else:
                            beam_seg = seg_maps[seg_idx][ib]
                        tmodel = beams_object[ib].compute_model(
                            spectrum_1d=temp_resamp_1d,
                            thumb=beams_object[ib].beam.direct * beam_seg,
                            in_place=False,
                            is_cgs=True,
                        )
                        models_arr[i0 : i0 + np.prod(v["2d_shape"])] += (
                            coeffs[f"bin_{seg_id}"][sample_i] * tmodel
                        )

                        i0 += np.prod(v["2d_shape"])

            if memmap:
                models_arr.flush()

        except:
            raise Exception(
                shared_models_name
                + "".join(traceback.format_exception(*sys.exc_info()))
            )

    @staticmethod
    def _reduce_seg_map(
        seg_idx=None,
        seg_id=None,
        shared_input=None,
        init_shape=None,
        shared_output=None,
        output_shape=None,
        beam_idx=None,
        oversamp_factor=None,
        memmap: bool = False,
    ):
        if memmap:
            init_arr = np.memmap(shared_input, dtype=float, shape=init_shape, mode="r+")
            output_arr = np.memmap(
                shared_output, dtype=float, shape=output_shape, mode="r+"
            )
        else:
            shm_init = shared_memory.SharedMemory(name=shared_input)
            init_arr = np.ndarray(init_shape, dtype=float, buffer=shm_init.buf)
            shm_output = shared_memory.SharedMemory(name=shared_output)
            output_arr = np.ndarray(output_shape, dtype=float, buffer=shm_output.buf)

        output_arr[seg_idx, beam_idx, :, :] = block_reduce(
            init_arr == seg_id,
            oversamp_factor,
            func=np.mean,
        )
        if memmap:
            output_arr.flush()

    def fit_at_z(
        self,
        z: float = 0.0,
        fit_background: bool = True,
        poly_order: int = 0,
        n_samples: int = 3,
        n_iters: int = 10,
        spec_wavs: ArrayLike | None = None,
        oversamp_factor: int = 1,
        veldisp: float = 500,
        direct_images: None = None,
        out_dir: PathLike | None = None,
        temp_dir: PathLike | None = None,
        memmap: bool = False,
        cpu_count: int = -1,
        overwrite: bool = False,
        use_lines: dict = CLOUDY_LINE_MAP,
        save_lines: bool = True,
        save_stacks: bool = True,
        pline: dict = DEFAULT_PLINE,
        seed: int = 2744,
        nnls_method: str = "scipy",
        nnls_iters: int = 100,
        nnls_tol: float = 1e-5,
        n_shifted: int = 2,
        n_shifted_samples: int = 1,
    ):
        """
        Fit the object at a specified redshift.

        Parameters
        ----------
        z : float, optional
            The redshift at which the object will be fitted, by default 0.
        fit_background : bool, optional
            Fit a constant background level, by default ``True``.
        poly_order : int, optional
            Fit a polynomial function to the spectrum, with a default
            order of ``0``.
        n_samples : int, optional
            The number of samples to draw from the joint posterior
            distributions in each region, by default ``3``.
        n_iters : int, optional
            The number of iterations to perform when fitting, by default
            ``10``.
        spec_wavs : ArrayLike | None, optional
            The wavelength sampling to use when generating the template
            spectra from the `bagpipes` posterior distributions. The
            default value of ``None`` sets this to the
            :math:`0.96 - 2.3\\mu\\rm{m}` range in :math:`45\\mathring{A}`
            steps.
        oversamp_factor : int, optional
            The factor by which the region segmentation map is oversampled
            before reprojecting to the beam coordinate system. This
            significantly slows down the model generation, but is
            essential to ensure that the template spectra correspond to
            the correct pixels in the NIRISS detector frame, and defaults
            to a factor of ``1``.
        veldisp : float, optional
            The velocity dispersion of the template spectra in km/s, by
            default ``500``.
        direct_images : _type_, optional
            WIP, may allow for changing beam direct images at some point.
            By default ``None``.
        out_dir : PathLike | None, optional
            Where the output files will be written. If ``None`` (default),
            files will be written to ``self.out_dir/multiregion``.
        temp_dir : PathLike | None, optional
            The temporary directory to use for memmapped files (if
            ``memmap==True``). If ``None`` (default), the current working
            directory will be used.
        memmap : bool, optional
            Whether to use a memmap to store large files. By default
            ``False``. If ``True``, the large model array will be written
            to a temporary array on disk.
        cpu_count : int, optional
            The number of CPUs to use for multiprocessing, by default -1.
        overwrite : bool, optional
            If ``True``, overwrite any existing fit. By default ``False``,
            and will attempt to load a previous fit.
        use_lines : ArrayLike, optional
            A list of lines, for which a 2D map will be generated based on
            the multi-region fit. Each item in the list should be a
            ``dict``, containing the following keys:

            * ``"cloudy"`` : The name of one or more lines following the
              ``Cloudy`` nomenclature
              (`Ferland+17 <https://ui.adsabs.harvard.edu/abs/2017RMxAA..53..385F/abstract>`__).
            * ``"grizli"`` : The name of the emission line in `grizli`, or
              any other desired name.
            * ``"wave"`` : The rest-frame vacuum wavelength of the line.

        save_lines : bool, optional
            Save the drizzled emission line maps, matching the `grizli`
            output format. By default ``True``.
        save_stacks : bool, optional
            Save the stacked beams, full models, and continuum models. The
            output format differs from grizli in that these stacks are not
            drizzled to account for the subpixel shifts. By default
            ``True``.
        pline : dict, optional
            Parameters for generating the drizzled emission line maps.
            Defaults to `~niriss_tools.grism.DEFAULT_PLINE`.
        seed : int | None, optional
            The seed for the random sampling, by default 2744. If None,
            then a new seed will be generated each time this method is
            called.
        nnls_method : str, optional
            The method to use for finding the best-fit coefficients for
            the set of templates. Must be one of "scipy", "numba", or
            "adelie" (ordered in increasing speed). By default, "scipy"
            will be used.
        nnls_iters : int or ArrayLike, optional
            The maximum number of iterations to attempt if the NNLS
            solution has not converged to the specified tolerance. If more
            than one value is passed, a two-stage fit will be run, whereby
            the best-fit solution from the first ``n_iters`` attempts will
            be re-fit with ``nnls_iters[0]`` iterations. This is only
            relevant if ``nnls_method != "scipy"``.
        nnls_tol : float or ArrayLike, optional
            The desired tolerance for the NNLS solver. As with
            ``nnls_iters``, the second value can be used to perform a more
            precise fit after the initial ``n_iters`` attempts.
        n_shifted : int or None, optional
            This allows for drawing additional posterior samples from
            other regions of the object. Regions will be selected
            randomly, with no preference as to spatial or spectral
            coherence (they are selected by shifting each segmentation map
            id when generating the models, hence the name). This reduces
            the chance of template mismatch based on the SED fit. By
            default, 2 additional regions will be used.
        n_shifted_samples : int or None, optional
            This determines the number of samples drawn from each of the
            additional regions (i.e. the total number of samples is given
            by ``n_samples + n_shifted * n_shifted_samples``). By default,
            only 1 sample is drawn from each extra region.

        Returns
        -------
        tuple
            Exact form of return still WIP.
        """

        try:
            import adelie

            HAS_ADELIE = True
        except:
            HAS_ADELIE = False

        nnls_iters = np.atleast_1d(nnls_iters).astype(int)
        nnls_tol = np.atleast_1d(nnls_tol)
        TWO_STAGE = (len(nnls_iters) > 1) | (len(nnls_tol) > 1)

        if spec_wavs is None:
            spec_wavs = np.arange(10000.0, 23000.0, 45.0)

        if memmap:
            if temp_dir is None:
                temp_dir = Path.cwd()
            else:
                temp_dir = Path(temp_dir)
                temp_dir.mkdir(exist_ok=True, parents=True)

        if not out_dir:
            multireg_out_dir = self.out_dir / "multiregion"
        else:
            multireg_out_dir = Path(out_dir)

        multireg_out_dir.mkdir(exist_ok=True, parents=True)

        self.MB.init_poly_coeffs(poly_order=poly_order)

        if fit_background:
            self.fit_bg = True
            A = np.vstack((self.MB.A_bg, self.MB.A_poly))
        else:
            self.fit_bg = False
            A = self.MB.A_poly * 1

        # Two RNG, so can compare with and without extra shift samples
        rng = np.random.default_rng(seed=seed)
        rng_shifts = np.random.default_rng(seed=seed)

        with h5py.File(
            self.pipes_dir
            / "posterior"
            / self.run_name
            / f"{int(self.regions_phot_cat["bin_id"][0])}.h5",
            "r",
        ) as test_post:

            fit_info_str = test_post.attrs["fit_instructions"]
            fit_info_str = fit_info_str.replace("array", "np.array")
            fit_info_str = fit_info_str.replace("float", "np.float")
            fit_info_str = fit_info_str.replace("np.np.", "np.")
            fit_instructions = eval(fit_info_str)
            n_post_samples = test_post["samples2d"].shape[0]

        # Try to allow for both memory and file-backed multiprocessing of
        # large arrays

        smm = SharedMemoryManager()
        smm.start()

        # If we are not oversampling, we can drastically reduce the memory usage
        # and only store the reprojected (multiregion) seg map for each beam.
        # If the segmentation map is not aligned to the beam direct images, and
        # we are oversampling to find the exact pixel overlap, the fastest
        # method for generating forward-modelled spectra is to pre-calculate
        # the segmentation map overlap for each region, and store this in a
        # shared memory array.
        if oversamp_factor == 1:
            oversamp_seg_maps_shape = (
                self.MB.N,
                self.MB.beams[0].beam.sh[0],
                self.MB.beams[0].beam.sh[1],
            )
        else:
            oversamp_seg_maps_shape = (
                self.n_regions,
                self.MB.N,
                self.MB.beams[0].beam.sh[0],
                self.MB.beams[0].beam.sh[1],
            )

        if memmap:
            oversamp_seg_maps = np.memmap(
                temp_dir / "memmap_oversamp_seg_maps.dat",
                dtype=float,
                mode="w+",
                shape=oversamp_seg_maps_shape,
            )
        else:
            shm_seg_maps = smm.SharedMemory(
                size=np.dtype(float).itemsize * np.prod(oversamp_seg_maps_shape)
            )
            oversamp_seg_maps = np.ndarray(
                oversamp_seg_maps_shape,
                dtype=float,
                buffer=shm_seg_maps.buf,
            )

        oversamp_seg_maps.fill(0.0)

        beam_info = {}
        start_idx = 0

        oversampled_shape = (
            oversamp_factor * self.MB.beams[0].beam.sh[0],
            oversamp_factor * self.MB.beams[0].beam.sh[1],
        )

        if memmap:
            oversampled = np.memmap(
                temp_dir / "memmap_oversampled.dat",
                dtype=float,
                mode="w+",
                shape=oversampled_shape,
            )
        else:
            shm_oversampled = smm.SharedMemory(
                size=np.dtype(float).itemsize * np.prod(oversampled_shape)
            )
            oversampled = np.ndarray(
                oversampled_shape,
                dtype=float,
                buffer=shm_oversampled.buf,
            )
        oversampled.fill(np.nan)

        start_idx = 0
        for i, (beam_cutout, cutout_shape) in enumerate(
            zip(self.MB.beams, self.MB.Nflat)
        ):
            beam_name = f"{beam_cutout.grism.pupil}-{beam_cutout.grism.filter}"
            if not beam_name in beam_info:
                beam_info[beam_name] = {}
                beam_info[beam_name]["2d_shape"] = beam_cutout.sh
                beam_info[beam_name]["list_idx"] = []
                beam_info[beam_name]["flat_slice"] = []
            beam_info[beam_name]["list_idx"].append(i)
            beam_info[beam_name]["flat_slice"].append(
                slice(int(start_idx), int(start_idx + cutout_shape))
            )
            start_idx += cutout_shape

            beam_wcs = deepcopy(beam_cutout.direct.wcs)
            beam_wcs = grizli_utils.transform_wcs(beam_wcs, scale=oversamp_factor)

            oversampled[:] = reproject_interp(
                (self.regions_seg_map, self.regions_seg_wcs),
                beam_wcs,
                oversampled_shape,
                return_footprint=False,
                order=0,
            )

            if oversamp_factor == 1:
                oversamp_seg_maps[i] = oversampled
                continue

            with Pool(processes=cpu_count) as pool:
                multi_fn = partial(
                    self._reduce_seg_map,
                    shared_input=(
                        temp_dir / "memmap_oversampled.dat"
                        if memmap
                        else shm_oversampled.name
                    ),
                    init_shape=oversampled_shape,
                    shared_output=(
                        temp_dir / "memmap_oversamp_seg_maps.dat"
                        if memmap
                        else shm_seg_maps.name
                    ),
                    output_shape=oversamp_seg_maps_shape,
                    beam_idx=i,
                    oversamp_factor=oversamp_factor,
                    memmap=memmap,
                )
                pool.starmap(multi_fn, enumerate(self.regions_seg_ids))

        # Avoid any nan-related problems later
        oversamp_seg_maps[~np.isfinite(oversamp_seg_maps)] = 0

        # The total number of templates
        NTEMP = self.n_regions * n_samples
        if n_shifted > 0:
            NTEMP += self.n_regions * n_shifted * n_shifted_samples
        num_stacks = len([*beam_info.keys()])
        stacked_shape = np.nansum([np.prod(v["2d_shape"]) for v in beam_info.values()])
        temp_offset = A.shape[0] - self.MB.N + num_stacks

        # This is the large array of models. Each row corresponds to a
        # (probably) unique template, forward-modelled across all beams,
        # and flattened.
        stacked_A_shape = (temp_offset + NTEMP, stacked_shape)

        if memmap:
            stacked_A = np.memmap(
                temp_dir / "memmap_stacked_A.dat",
                dtype=float,
                mode="w+",
                shape=stacked_A_shape,
            )
        else:
            shm_stacked_A = smm.SharedMemory(
                size=np.dtype(float).itemsize * np.prod(stacked_A_shape)
            )
            stacked_A = np.ndarray(
                stacked_A_shape,
                dtype=float,
                buffer=shm_stacked_A.buf,
            )

        print(f"{memmap=}")

        stacked_A.fill(0.0)

        # Offset for background
        if fit_background:
            off = 0.04
        else:
            off = 0.0

        # There are very few scenarios in which it makes sense to use the
        # individual beams for fitting. The computational requirements are
        # already considerable, and for GLASS-JWST, not stacking would mean ~6x
        # more memory, and at least that in computation time.
        stacked_scif = np.zeros(stacked_shape)
        stacked_ivarf = np.zeros(stacked_shape)
        stacked_weightf = np.zeros(stacked_shape)
        stacked_fit_mask = np.zeros(stacked_shape, dtype=bool)

        start_idx = 0
        for k_i, (k, v) in enumerate(beam_info.items()):
            stack_idxs = np.r_["0,2", *v["flat_slice"]]

            stacked_scif[start_idx : start_idx + np.prod(v["2d_shape"])] = np.nanmedian(
                self.MB.scif[stack_idxs],
                axis=0,
            )
            stacked_weightf[start_idx : start_idx + np.prod(v["2d_shape"])] = (
                np.nanmedian(
                    self.MB.weightf[stack_idxs],
                    axis=0,
                )
            )
            stacked_fit_mask[start_idx : start_idx + np.prod(v["2d_shape"])] = np.any(
                self.MB.fit_mask[stack_idxs],
                axis=0,
            )
            if fit_background:
                stacked_A[k_i, start_idx : start_idx + np.prod(v["2d_shape"])] = 1.0
                stacked_A[
                    num_stacks : num_stacks + self.MB.A_poly.shape[0],
                    start_idx : start_idx + np.prod(v["2d_shape"]),
                ] = np.nanmean(self.MB.A_poly[:, stack_idxs], axis=1)
            else:
                stacked_A[
                    num_stacks : num_stacks + self.MB.A_poly.shape[0],
                    start_idx : start_idx + np.prod(v["2d_shape"]),
                ] = np.nanmean(self.MB.A_poly[:, stack_idxs], axis=1)

            stacked_ivarf[start_idx : start_idx + np.prod(v["2d_shape"])] = (
                np.nanmedian(self.MB.ivarf[stack_idxs], axis=0)
            )

            start_idx += np.prod(v["2d_shape"])

        stacked_fit_mask &= np.isfinite(stacked_scif)

        # TODO: make the output name a parameter?
        self.output_table_path = multireg_out_dir / (
            f"{self.obj_id}_{len(np.unique(self.regions_phot_cat["bin_id"]))}"
            f"bins_{n_iters}iters_{n_samples}samples_z_{z}_sig_{veldisp}.fits"
        )

        # The function to produce the templates - only a couple of
        # parameters change on each iteration.
        stacked_fn = partial(
            self._gen_stacked_templates_from_pipes,
            shared_seg_name=(
                temp_dir / "memmap_oversamp_seg_maps.dat"
                if memmap
                else shm_seg_maps.name
            ),
            seg_maps_shape=oversamp_seg_maps_shape,
            shared_models_name=(
                temp_dir / "memmap_stacked_A.dat" if memmap else shm_stacked_A.name
            ),
            models_shape=stacked_A_shape,
            posterior_dir=str(self.pipes_dir / "posterior" / self.run_name),
            n_samples=n_samples + (n_shifted * n_shifted_samples),
            spec_wavs=spec_wavs,
            beam_info=beam_info,
            temp_offset=temp_offset,
            cont_only=False,
            rm_line=None,
            memmap=memmap,
            n_shifted_rows=n_shifted_samples,
        )

        # These column names should be fixed for all objects
        init_col_names = [
            "iteration",
            "chi2",
            "max_nnls_iters",
            "solve_nnls_iters",
            "nnls_tol",
            "rows",
            "id_shifts",
            "n_shifted_samples",
        ]

        # Construct a zero-length table if it doesn't already exist
        if self.output_table_path.is_file() and not overwrite:
            output_table = Table.read(self.output_table_path, format="fits")
        else:
            output_table = Table(
                [
                    [0],
                    *np.zeros((len(init_col_names) - 4, 1)),
                    np.zeros((1, n_samples), dtype=int),
                    np.zeros((1, n_shifted), dtype=int),
                    [0],
                    *np.zeros((temp_offset, 1)),
                    *np.zeros(
                        (self.n_regions, 1, n_samples + (n_shifted * n_shifted_samples))
                    ),
                ],
                names=init_col_names
                + [f"base_coeffs_{b}" for b in np.arange(temp_offset)]
                + [f"bin_{p}" for p in self.regions_phot_cat["bin_id"]],
                dtype=[int, float, int, int, float, int, int, int]
                + [float] * temp_offset
                + [float] * self.n_regions,
            )
            output_table = output_table[:0]

        # Check if the table length matches the expected number of iterations
        prev_iters = len(output_table)
        if prev_iters < n_iters + int(TWO_STAGE):

            # If there are previous iterations, sample from the RNGs so the
            # seed order is preserved
            for x in np.arange(prev_iters):
                rows = rng.choice(
                    np.arange(n_post_samples, dtype=int),
                    size=n_samples,
                    replace=False,
                )
                id_shifts = rng_shifts.choice(
                    np.arange(self.n_regions, dtype=int), size=n_shifted, replace=False
                )

            remaining_iters = n_iters + int(TWO_STAGE) - prev_iters

            output_table.write(self.output_table_path, overwrite=True, format="fits")

            iterations = np.arange(prev_iters, n_iters + int(TWO_STAGE))

            for iteration in iterations:

                # On the final iteration, reuse the samples from the current
                # best-fit solution
                if TWO_STAGE and (iteration == iterations[-1]):

                    best_iter = np.argmin(output_table["chi2"])

                    rows = [int(s) for s in output_table["rows"][best_iter]]
                    id_shifts = [int(s) for s in output_table["id_shifts"][best_iter]]
                else:
                    rows = rng.choice(
                        np.arange(n_post_samples, dtype=int),
                        size=n_samples,
                        replace=False,
                    )
                    id_shifts = rng_shifts.choice(
                        np.arange(self.n_regions, dtype=int),
                        size=n_shifted,
                        replace=False,
                    )

                print(f"Iteration {iteration}, {rows=}")
                t0 = time()

                # Generate the forward-modelled spectra
                print("Generating models...")
                with multiprocessing.Pool(
                    processes=cpu_count,
                    initializer=_init_pipes_sampler,
                    initargs=(
                        fit_instructions,
                        veldisp,
                        self.MB.beams,
                    ),
                ) as pool:
                    for s_i, s in enumerate(self.regions_seg_ids):
                        pool.apply_async(
                            stacked_fn,
                            (s_i, s),
                            kwds={"rows": rows, "id_shifts": id_shifts},
                        )
                    pool.close()
                    pool.join()

                t1 = time()
                print(f"Generating models... DONE {t1-t0:.3f}s")

                # Remove any negative or zero templates
                ok_temp = np.sum(stacked_A, axis=1) > 0

                # We need to remove duplicate templates so the NNLS solvers can
                # actually converge. Until such time as np.unique implements
                # a hash map to allow for speeding up `return_index`, we just
                # skip over every 99 pixels
                stacked_A_contig = np.ascontiguousarray(stacked_A[:, ::99])
                # np.unique() finds identical items in a raveled array. To make it
                # see each row as a single item, we create a view of each row as a
                # byte string of length itemsize times number of columns in `ar`
                ar_row_view = stacked_A_contig.view(
                    "|S%d" % (stacked_A_contig.itemsize * stacked_A_contig.shape[1])
                )
                _, unique_idxs = np.unique(ar_row_view, return_index=True)
                unique_temp = np.isin(np.arange(stacked_A.shape[0]), unique_idxs)
                del stacked_A_contig

                # Select only the unique templates
                ok_temp &= unique_temp

                out_coeffs = np.zeros(stacked_A.shape[0])

                # Transpose the template array
                stacked_Ax = stacked_A[:, stacked_fit_mask][ok_temp, :].T

                stacked_Ax *= np.sqrt(stacked_ivarf[stacked_fit_mask][:, np.newaxis])
                print(
                    f"Array size reduced from {stacked_A.shape} to "
                    f"{stacked_Ax.shape[::-1]} ("
                    f"{np.prod(stacked_Ax.shape)/np.prod(stacked_A.shape):.1%} of original)"
                )
                print("NNLS fitting...")

                # Change the max iters and tolerance for the final iteration
                if TWO_STAGE and (iteration == iterations[-1]):
                    print("Final iteration")
                    _nnls_i = nnls_iters[1]
                    _nnls_t = nnls_tol[1]
                else:
                    _nnls_i = nnls_iters[0]
                    _nnls_t = nnls_tol[0]

                # Allow for background fitting by including an offset
                y = stacked_scif[stacked_fit_mask] + off
                y *= np.sqrt(stacked_ivarf[stacked_fit_mask])

                # Three different methods of fitting, each with different call
                # signatures and return values
                if nnls_method == "scipy":

                    coeffs, rnorm, info = scipy.optimize._nnls._nnls(
                        stacked_Ax, y + off, _nnls_i
                    )
                    coeffs[:num_stacks] -= off

                elif nnls_method == "numba":

                    nnls_solver = CDNNLS(stacked_Ax, y + off)
                    nnls_solver.run(n_iter=_nnls_i, epsilon=_nnls_t)
                    coeffs = nnls_solver.w
                    coeffs[:num_stacks] -= off

                elif nnls_method == "adelie" and HAS_ADELIE:
                    state = adelie.solver.bvls(
                        stacked_Ax,
                        y + off,
                        lower=np.zeros(stacked_Ax.shape[-1]),
                        upper=np.full(stacked_Ax.shape[-1], np.inf),
                        max_iters=_nnls_i,
                        tol=_nnls_t,
                        # n_threads=20 # Inter-thread communication is actually slower
                    )
                    state.solve()
                    state_iters = deepcopy(state.iters)
                    coeffs = deepcopy(state.beta)
                    coeffs[:num_stacks] -= off
                    del state

                print(f"NNLS fitting... DONE {time()-t1:.3f}s")

                out_coeffs[ok_temp] = coeffs
                stacked_modelf = np.dot(out_coeffs, stacked_A)
                chi2 = np.nansum(
                    (
                        stacked_weightf
                        * (stacked_scif - stacked_modelf) ** 2
                        * stacked_ivarf
                    )[stacked_fit_mask]
                )
                output_table.add_row(
                    [
                        iteration,
                        chi2,
                        _nnls_i,
                        state_iters if (nnls_method == "adelie" and HAS_ADELIE) else 0,
                        _nnls_t,
                        rows,
                        id_shifts,
                        n_shifted_samples,
                        *out_coeffs[:temp_offset],
                        *out_coeffs[temp_offset:].reshape(self.n_regions, -1),
                    ]
                )
                output_table.write(self.output_table_path, overwrite=True)
                print(f"Iteration {iteration}: chi2={chi2:.3f}\n")

                # Reset the template array
                stacked_A[temp_offset:].fill(0.0)

            del stacked_Ax

        # There must be a better way to obtain the coefficients, but
        # slicing tables is not entirely straightforward
        best_iter = np.argmin(output_table["chi2"])
        out_coeffs = np.asarray(
            [
                i
                for d in output_table[best_iter][len(init_col_names) :]
                for i in np.atleast_1d(d)
            ]
        ).ravel()

        # Repopulate the background parameters for MultiBeam
        if fit_background:
            for k_i, (k, v) in enumerate(beam_info.items()):
                for ib in v["list_idx"]:
                    self.MB.beams[ib].background = out_coeffs[k_i]

        best_rows = [int(s) for s in output_table["rows"][best_iter]]
        best_id_shifts = [int(s) for s in output_table["id_shifts"][best_iter]]

        # Largely unmodified from the original grizli code. Included within
        # this particular class method to avoid dealing with SharedMemory
        if save_stacks:
            print("Generating models...")
            with multiprocessing.Pool(
                processes=cpu_count,
                initializer=_init_pipes_sampler,
                initargs=(
                    fit_instructions,
                    veldisp,
                    self.MB.beams,
                ),
            ) as pool:
                for s_i, s in enumerate(self.regions_seg_ids):
                    pool.apply_async(
                        stacked_fn,
                        (s_i, s),
                        kwds={"rows": best_rows, "id_shifts": best_id_shifts},
                        error_callback=print,
                        # error_callback=traceback.format_exc,
                    )
                pool.close()
                pool.join()

            stacked_modelf = np.dot(out_coeffs, stacked_A)  # .copy()

            stacked_A[temp_offset:].fill(0.0)

            print("Generating continuum...")
            with multiprocessing.Pool(
                processes=cpu_count,
                initializer=_init_pipes_sampler,
                initargs=(
                    fit_instructions,
                    veldisp,
                    self.MB.beams,
                ),
            ) as pool:
                for s_i, s in enumerate(self.regions_seg_ids):
                    pool.apply_async(
                        stacked_fn,
                        (s_i, s),
                        kwds={
                            "rows": best_rows,
                            "id_shifts": best_id_shifts,
                            "cont_only": True,
                        },
                        # error_callback=print,
                    )
                pool.close()
                pool.join()

            stacked_contf = np.dot(out_coeffs, stacked_A)  # .copy()

            stacked_hdul = fits.HDUList(fits.PrimaryHDU())

            start_idx = 0
            for i, (k, v) in enumerate(beam_info.items()):

                slice_plot = slice(start_idx, start_idx + np.prod(v["2d_shape"]))

                hdus = [
                    fits.ImageHDU(
                        data=stacked_scif[slice_plot].reshape(v["2d_shape"]),
                        name="SCI",
                    ),
                    fits.ImageHDU(
                        data=stacked_weightf[slice_plot].reshape(v["2d_shape"]),
                        name="WHT",
                    ),
                    fits.ImageHDU(
                        data=stacked_ivarf[slice_plot].reshape(v["2d_shape"]),
                        name="IVAR",
                    ),
                    fits.ImageHDU(
                        data=(stacked_fit_mask[slice_plot] * 1.0).reshape(
                            v["2d_shape"]
                        ),
                        name="MASK",
                    ),
                    fits.ImageHDU(
                        data=stacked_modelf[slice_plot].reshape(v["2d_shape"]),
                        name="MODEL",
                    ),
                    fits.ImageHDU(
                        data=stacked_contf[slice_plot].reshape(v["2d_shape"]),
                        name="CONT",
                    ),
                ]
                for h in hdus:
                    h.header["EXTVER"] = k
                    h.header["RA"] = (self.ra, "Right ascension")
                    h.header["DEC"] = (self.dec, "Declination")
                    h.header["GRISM"] = (k.split("_")[0], "Grism")
                    # h.header['ISFLAM'] = (flambda, 'Pixels in f-lam units')
                    h.header["CONF"] = (
                        self.MB.beams[0].beam.conf.conf_file,
                        "Configuration file",
                    )
                    h.header["REDSHIFT"] = (z, "Redshift used")
                    h.header["CHI2"] = output_table["chi2"][best_iter]
                    h.header = self.add_pipes_info(h.header)
                stacked_hdul.extend(hdus)

                start_idx += np.prod(v["2d_shape"])

            stacked_hdul.writeto(
                multireg_out_dir / f"regions_{self.obj_id:05d}_z_{z}_stacked.fits",
                output_verify="silentfix",
                overwrite=True,
            )

        if save_lines:
            beam_models_len = 0
            for k_i, (k, v) in enumerate(beam_info.items()):
                beam_models_len += np.prod(v["2d_shape"]) * len(v["list_idx"])

            if memmap:
                flat_beam_models = np.memmap(
                    temp_dir / "memmap_beams_model.dat",
                    dtype=float,
                    mode="w+",
                    shape=(beam_models_len),
                )
            else:

                shm_beam_models = smm.SharedMemory(
                    size=np.dtype(float).itemsize * beam_models_len
                )
                flat_beam_models = np.ndarray(
                    (beam_models_len),
                    dtype=float,
                    buffer=shm_beam_models.buf,
                )

            line_hdu = None
            saved_lines = []

            beams_fn = partial(
                self._gen_beam_templates_from_pipes,
                shared_seg_name=(
                    temp_dir / "memmap_oversamp_seg_maps.dat"
                    if memmap
                    else shm_seg_maps.name
                ),
                seg_maps_shape=oversamp_seg_maps_shape,
                shared_models_name=(
                    temp_dir / "memmap_beams_model.dat"
                    if memmap
                    else shm_beam_models.name
                ),
                models_shape=flat_beam_models.shape,
                posterior_dir=str(self.pipes_dir / "posterior" / self.run_name),
                spec_wavs=spec_wavs,
                beam_info=beam_info,
                cont_only=False,
                rows=best_rows,
                coeffs=output_table[best_iter],
                memmap=memmap,
                n_shifted_rows=n_shifted_samples,
            )

            for l_i, l_v in enumerate(use_lines):

                if not check_coverage(l_v["wave"] * (1 + z)):
                    continue

                print(f"Generating map for {l_v["grizli"]}...")

                flat_beam_models.fill(0.0)

                with multiprocessing.Pool(
                    processes=cpu_count,
                    initializer=_init_pipes_sampler,
                    initargs=(fit_instructions, veldisp, self.MB.beams),
                ) as pool:
                    for s_i, s in enumerate(self.regions_phot_cat["bin_id"]):
                        pool.apply_async(
                            beams_fn,
                            args=(s_i, s),
                            kwds={
                                "rm_line": l_v["cloudy"],
                                "id_shifts": best_id_shifts,
                            },
                            error_callback=print,
                        )
                    pool.close()
                    pool.join()

                i0 = 0
                start_idx = 0
                for k_i, (k, v) in enumerate(beam_info.items()):
                    for ib in v["list_idx"]:
                        self.MB.beams[ib].beam.model = flat_beam_models[
                            i0 : i0 + np.prod(v["2d_shape"])
                        ].reshape(v["2d_shape"])
                        i0 += np.prod(v["2d_shape"])
                    start_idx += np.prod(v["2d_shape"])

                hdu = drizzle_to_wavelength(
                    self.MB.beams,
                    ra=self.ra,
                    dec=self.dec,
                    wave=l_v["wave"] * (1 + z),
                    fcontam=self.MB.fcontam,
                    **pline,
                )

                hdu[0].header["REDSHIFT"] = (z, "Redshift used")
                hdu[0].header["CHI2"] = output_table["chi2"][best_iter]
                hdu[0].header = self.add_pipes_info(hdu[0].header)
                for e in [-4, -3, -2, -1]:
                    hdu[e].header["EXTVER"] = l_v["grizli"]
                    hdu[e].header["REDSHIFT"] = (z, "Redshift used")
                    hdu[e].header["RESTWAVE"] = (
                        l_v["wave"],
                        "Line rest wavelength",
                    )

                saved_lines.append(l_v["grizli"])

                if line_hdu is None:
                    line_hdu = hdu
                    line_hdu[0].header["NUMLINES"] = (
                        1,
                        "Number of lines in this file",
                    )
                else:
                    line_hdu.extend(hdu[-4:])
                    line_hdu[0].header["NUMLINES"] += 1

                    # Make sure DSCI extension is filled.  Can be empty for
                    # lines at the edge of the grism throughput
                    for f_i in range(hdu[0].header["NDFILT"]):
                        filt_i = hdu[0].header["DFILT{0:02d}".format(f_i + 1)]
                        if hdu["DWHT", filt_i].data.max() != 0:
                            line_hdu["DSCI", filt_i] = hdu["DSCI", filt_i]
                            line_hdu["DWHT", filt_i] = hdu["DWHT", filt_i]

                li = line_hdu[0].header["NUMLINES"]
                line_hdu[0].header["LINE{0:03d}".format(li)] = l_v["grizli"]

            if line_hdu is not None:
                line_hdu[0].header["HASLINES"] = (
                    " ".join(saved_lines),
                    "Lines in this file",
                )

                line_wcs = WCS(line_hdu[1].header)
                segm = self.MB.drizzle_segmentation(wcsobj=line_wcs)
                seg_hdu = fits.ImageHDU(data=segm.astype(np.int32), name="SEG")
                line_hdu.insert(1, seg_hdu)

                line_hdu.writeto(
                    multireg_out_dir
                    / f"regions_{self.obj_id:05d}_z_{z}_{pline.get("pixscale", 0.06)}arcsec.line.fits",
                    output_verify="silentfix",
                    overwrite=True,
                )

        smm.shutdown()

        # chi2 = np.nansum(
        #     (stacked_weightf * (stacked_scif - stacked_modelf) ** 2 * stacked_ivarf)[
        #         stacked_fit_mask
        #     ]
        # )

        if fit_background:
            poly_coeffs = out_coeffs[num_stacks : num_stacks + self.MB.n_poly]
        else:
            poly_coeffs = out_coeffs[: self.MB.n_poly]

        # print(self.n_poly, self.N)
        # print(num_stacks)
        # print(f"{out_coeffs=}")
        # print(poly_coeffs, self.x_poly)
        self.MB.y_poly = np.dot(poly_coeffs, self.MB.x_poly)
        # x_poly = self.x_poly[1,:]+1 = self.beams[0].beam.lam/1.e4

        # return A, out_coeffs, chi2, stacked_modelf
        return
