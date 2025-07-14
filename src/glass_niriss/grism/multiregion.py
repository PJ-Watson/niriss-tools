"""
Functions and classing related to multi-region grism fitting.
"""

import multiprocessing
import sys
import traceback
from copy import deepcopy
from functools import partial
from itertools import product
from multiprocessing import Manager, Pool, cpu_count, shared_memory
from multiprocessing.managers import SharedMemoryManager
from os import PathLike
from pathlib import Path

import bagpipes
import h5py
import matplotlib.pyplot as plt
import numpy as np

# import fnnlsEigen as fe
# from matrix_free_nnls import solve_nnls
import pandas
from astropy.io import fits
from astropy.nddata import block_reduce
from astropy.table import Table
from astropy.wcs import WCS
from bagpipes import model_galaxy
from grizli.multifit import MultiBeam, drizzle_to_wavelength
from numpy.typing import ArrayLike
from pandas._libs.hashtable import SIZE_HINT_LIMIT, duplicated
from pandas.core import algorithms
from pandas.core.sorting import get_group_index
from reproject import reproject_interp
from grizli import utils as grizli_utils
from grizli.model import GrismDisperser

from time import time

from glass_niriss.grism.specgen import (
    CLOUDY_LINE_MAP,
    BagpipesSampler,
    ExtendedModelGalaxy,
    check_coverage,
)

# def duplicated_rows(a, keep="first"):
#     size_hint = min(a.shape[0], SIZE_HINT_LIMIT)

#     def f(vals):
#         labels, shape = algorithms.factorize(vals, size_hint=size_hint)
#         return labels.astype("i8", copy=False), len(shape)

#     vals = (col for col in a.T)
#     labels, shape = map(list, zip(*map(f, vals)))

#     ids = get_group_index(labels, shape, sort=False, xnull=False)
#     return duplicated(ids, keep)


# from glass_niriss.pytntnn import tntnn

__all__ = ["RegionsMultiBeam", "CDNNLS", "DEFAULT_PLINE"]

pipes_sampler = None
beams_object = None
# scaled_seg_maps = None

import numba
import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from scipy import sparse

DEFAULT_PLINE = {
    "pixscale": 0.06,
    "pixfrac": 1.0,
    "size": 5,
    "kernel": "lanczos3",
}


class CDNNLS:
    """
    Coordinate descent NNLS.

    Parameters
    ----------
    X : ArrayLike
        The coefficient array.
    y : ArrayLike
        The RHS vector.
    """

    def __init__(self, X: ArrayLike, y: ArrayLike):
        self._set_objective(X, y)

    def _set_objective(self, X, y):
        # use Fortran order to compute gradient on contiguous columns
        # self.X, self.y = np.asfortranarray(X), y
        self.X, self.y = X, y

        # Make sure we cache the numba compilation.
        self.run(1)

    def run(
        self,
        method: str = "classic",
        n_iter: int | None = None,
        epsilon: float = 1e-6,
        **kwargs,
    ):
        """
        Run a non-negative least squares solver.

        Parameters
        ----------
        method : str, optional
            The method to use, either the default "classic", or "antilop".
        n_iter : int | None, optional
            How many iterations to run. If ``None``, then this is set to
            ``3*X.shape[1]``.
        epsilon : float, optional
            The tolerance to use for the stopping condition, by default
            ``1e-6``.
        **kwargs : dict, optional
            Other arguments to pass to the individual methods.
        """

        if n_iter is None:
            n_iter = 3 * self.X.shape[1]
        match method:
            case "classic":
                # print("classic")
                self._run_classic(n_iter, epsilon, **kwargs)
            case "antilop":
                # print("antilop")
                self.w = self._run_antilop(self.X, self.y, n_iter, epsilon, **kwargs)

    # @staticmethod
    # @njit('double[:,:](double[:,:], double[:,:],double,int_,double[:,:])')
    # def _solve_NQP(Q, q, epsilon, max_n_iter, x):

    #     # Initialize
    #     x_diff     = np.zeros_like(x)
    #     grad_f     = q.copy()
    #     grad_f_bar = q.copy()

    #     # Loop over iterations
    #     for i in range(max_n_iter):

    #         # Get passive set information
    #         passive_set = np.logical_or(x > 0, grad_f < 0)
    #         # print (f"{passive_set.shape=}")
    #         n_passive = np.sum(passive_set)

    #         # Calculate gradient
    #         grad_f_bar[:] = grad_f
    #         # orig_shape = grad_f_bar
    #         grad_f_bar = grad_f_bar.flatten()
    #         grad_f_bar[~passive_set.flatten()] = 0
    #         grad_f_bar = grad_f_bar.reshape(grad_f.shape)
    #         grad_norm = np.dot(grad_f_bar.flatten(), grad_f_bar.flatten())
    #         # print (grad_norm)

    #         # Abort?
    #         if (n_passive == 0 or grad_norm < epsilon):
    #             break

    #         # Exact line search
    #         # print (Q.dot(grad_f_bar).shape, grad_f_bar.shape)
    #         alpha_den = np.vdot(grad_f_bar.flatten(), Q.dot(grad_f_bar).flatten())
    #         alpha = grad_norm / alpha_den

    #         # Update x
    #         x_diff = -x.copy()
    #         x -= alpha * grad_f_bar
    #         x = np.maximum(x, 0.)
    #         x_diff += x

    #         # Update gradient
    #         grad_f += Q.dot(x_diff)
    #         # print (grad_f)

    #     # # Return
    #     return x

    @staticmethod
    @njit(
        numba.double[:, ::1](
            numba.double[:, ::1], numba.double[::1], numba.int_, numba.double
        )
    )
    def _run_antilop(X, Y, n_iter, epsilon):

        # if not n_iter:
        #     n_iter = 3*Y.shape[0]

        XTX = np.dot(X.T, X)
        XTY = np.dot(X.T, Y)

        # Normalization factors
        H_diag = np.diag(XTX).reshape(-1, 1)
        Q_den = np.sqrt(np.outer(H_diag, H_diag))
        q_den = np.sqrt(H_diag)

        # Normalize
        Q = XTX / Q_den
        q = -XTY.reshape(-1, 1) / q_den
        # q = -XTY / q_den.flatten()

        # print (XTX.shape, XTY.shape, q_den.flatten().shape)

        # Solve NQP
        y = np.zeros((X.shape[1], 1))
        # print (q.shape)
        # y = self.solveNQP_NUMBA(Q, q, epsilon, max_n_iter, y)
        y_diff = np.zeros_like(y)
        grad_f = q.copy()
        grad_f_bar = q.copy()

        # Loop over iterations
        for i in range(n_iter):

            # Get passive set information
            passive_set = np.logical_or(y > 0, grad_f < 0)
            # print (f"{passive_set.shape=}")
            n_passive = np.sum(passive_set)

            # Calculate gradient
            grad_f_bar[:] = grad_f
            # orig_shape = grad_f_bar
            grad_f_bar = grad_f_bar.flatten()
            grad_f_bar[~passive_set.flatten()] = 0
            grad_f_bar = grad_f_bar.reshape(grad_f.shape)
            grad_norm = np.dot(grad_f_bar.flatten(), grad_f_bar.flatten())
            # print (grad_norm)

            # Abort?
            if n_passive == 0 or grad_norm < epsilon:
                print("Finished, iter=", i)
                break

            # Exact line search
            # print (Q.dot(grad_f_bar).shape, grad_f_bar.shape)
            alpha_den = np.vdot(grad_f_bar.flatten(), np.dot(Q, grad_f_bar).flatten())
            alpha = grad_norm / alpha_den

            # Update y
            y_diff = -y.copy()
            y -= alpha * grad_f_bar
            y = np.maximum(y, 0.0)
            y_diff += y

            # Update gradient
            grad_f += np.dot(Q, y_diff)

        # Undo normalization
        x = y / q_den

        # Return
        return x

    def _run_classic(self, n_iter: int, epsilon: float = 1e-6):
        X = np.asfortranarray(self.X)
        L = (X**2).sum(axis=0)
        if sparse.issparse(self.X):
            self.w = self._sparse_cd_classic(
                self.X.data, self.X.indices, self.X.indptr, self.y, L, n_iter
            )
        else:
            self.w = self._cd_classic(X, self.y, L, n_iter, epsilon)

    @staticmethod
    @njit
    def _cd_classic(X, y, L, n_iter, epsilon):
        n_features = X.shape[1]
        R = np.copy(y)
        w = np.zeros(n_features)
        XTX = X.T.dot(X)
        f = -X.T.dot(y)
        for _ in range(n_iter):
            Hxf = np.dot(XTX, w) + f
            # print (_, np.sum(np.abs(Hxf) <= epsilon)/n_features)
            criterion_1 = np.all(Hxf >= -epsilon)
            criterion_2 = np.all(Hxf[w > 0] <= epsilon)
            if criterion_1 and criterion_2:
                print("Finished, iter=", _)
                break
            for j in range(n_features):
                if L[j] == 0.0:
                    continue
                old = w[j]
                w[j] = max(w[j] + X[:, j] @ R / L[j], 0)
                diff = old - w[j]
                if diff != 0:
                    R += diff * X[:, j]
        return w

    @staticmethod
    @njit
    def _sparse_cd_classic(X_data, X_indices, X_indptr, y, L, n_iter):
        n_features = len(X_indptr) - 1
        w = np.zeros(n_features)
        R = np.copy(y)
        for _ in range(n_iter):
            for j in range(n_features):
                if L[j] == 0.0:
                    continue
                old = w[j]
                start, end = X_indptr[j : j + 2]
                scal = 0.0
                for ind in range(start, end):
                    scal += X_data[ind] * R[X_indices[ind]]
                w[j] = max(w[j] + scal / L[j], 0)
                diff = old - w[j]
                if diff != 0:
                    for ind in range(start, end):
                        R[X_indices[ind]] += diff * X_data[ind]
        return w


def _init_pipes_sampler(fit_instructions, veldisp, beams):
    global pipes_sampler
    # print (f"{core_id} initialised")
    pipes_sampler = BagpipesSampler(fit_instructions=fit_instructions, veldisp=veldisp)
    global beams_object
    beams_object = beams.copy()


class RegionsMultiBeam(MultiBeam):
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
        binned_data: PathLike,
        pipes_dir: PathLike,
        run_name: str,
        beams: list,
        **multibeam_kwargs,
    ):

        super().__init__(beams, **multibeam_kwargs)

        self.regions_phot_cat = Table.read(binned_data, "PHOT_CAT")
        self.regions_phot_cat = self.regions_phot_cat[
            self.regions_phot_cat["bin_id"].astype(int) != 0
        ]
        self.n_regions = len(self.regions_phot_cat)

        # print(self.regions_phot_cat)

        with fits.open(binned_data) as hdul:
            self.regions_seg_map = hdul["SEG_MAP"].data.copy()
            self.regions_seg_hdr = hdul["SEG_MAP"].header.copy()
            self.regions_seg_wcs = WCS(self.regions_seg_hdr)

        self.regions_seg_ids = np.asarray(self.regions_phot_cat["bin_id"], dtype=int)

        self.run_name = run_name
        self.pipes_dir = pipes_dir

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
    ):
        seg_id = int(seg_id)

        try:

            if memmap:
                # init_arr = np.memmap(shared_input, dtype=float, shape=init_shape, mode="r+")
                # output_arr = np.memmap(
                #     shared_output, dtype=float, shape=output_shape, mode="r+"
                # )
                seg_maps = np.memmap(
                    shared_seg_name, dtype=float, shape=seg_maps_shape, mode="r+"
                )
                temps_arr = np.memmap(
                    shared_models_name, dtype=float, shape=models_shape, mode="r+"
                )
            else:
                # # print (f"{seg_idx=}, {seg_id=}")
                # shm_init = shared_memory.SharedMemory(name=shared_input)
                # init_arr = np.ndarray(init_shape, dtype=float, buffer=shm_init.buf)
                # shm_output = shared_memory.SharedMemory(name=shared_output)
                # output_arr = np.ndarray(output_shape, dtype=float, buffer=shm_output.buf)

                shm_seg_maps = shared_memory.SharedMemory(name=shared_seg_name)
                seg_maps = np.ndarray(
                    seg_maps_shape, dtype=float, buffer=shm_seg_maps.buf
                )
                shm_temps = shared_memory.SharedMemory(name=shared_models_name)
                temps_arr = np.ndarray(models_shape, dtype=float, buffer=shm_temps.buf)

            with h5py.File(Path(posterior_dir) / f"{seg_id}.h5", "r") as post_file:
                samples2d = np.array(post_file["samples2d"])

            for sample_i, row in enumerate(rows):

                temp_resamp_1d = pipes_sampler.sample(
                    samples2d[row],
                    spec_wavs=spec_wavs,
                    cont_only=cont_only,
                    rm_line=rm_line,
                )

                i0 = 0
                for k_i, (k, v) in enumerate(beam_info.items()):
                    # stack_idxs = np.r_["0,2", *v["flat_slice"]]
                    for ib in v["list_idx"]:
                        beam = beams_object[ib]

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
        n_samples=None,
        spec_wavs=None,
        beam_info=None,
        temp_offset=None,
        cont_only=False,
        rm_line=None,
        rows=None,
        coeffs=None,
        memmap: bool = False,
    ):
        seg_id = int(seg_id)

        try:

            if memmap:
                # init_arr = np.memmap(shared_input, dtype=float, shape=init_shape, mode="r+")
                # output_arr = np.memmap(
                #     shared_output, dtype=float, shape=output_shape, mode="r+"
                # )
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
                samples2d = np.array(post_file["samples2d"])

            for sample_i, row in enumerate(rows):

                temp_resamp_1d = pipes_sampler.sample(
                    samples2d[row],
                    spec_wavs=spec_wavs,
                    cont_only=cont_only,
                    rm_line=rm_line,
                )

                i0 = 0
                for k_i, (k, v) in enumerate(beam_info.items()):
                    # stack_idxs = np.r_["0,2", *v["flat_slice"]]
                    for ib in v["list_idx"]:
                        # beam = beams_object[ib]

                        # beam_seg = seg_maps[seg_idx][ib]
                        tmodel = beams_object[ib].compute_model(
                            spectrum_1d=temp_resamp_1d,
                            thumb=beams_object[ib].beam.direct * seg_maps[seg_idx][ib],
                            in_place=False,
                            is_cgs=True,
                        )
                        # beam.beam.model += coeffs[f"{seg_id}_{sample_i}"]*tmodel.reshape(beam.beam.model.shape)
                        models_arr[i0 : i0 + np.prod(v["2d_shape"])] += (
                            coeffs[f"{seg_id}_{sample_i}"] * tmodel
                        )  # .reshape(beam.beam.model.shape)
                        # print (beam.model.shape)
                        # temps_arr[
                        #     (n_samples * seg_idx) + sample_i + temp_offset,
                        #     i0 : i0 + np.prod(v["2d_shape"]),
                        # ] += tmodel

                        # temps_arr[
                        #     (n_samples * seg_idx) + sample_i + temp_offset,
                        #     i0 : i0 + np.prod(v["2d_shape"]),
                        # ] /= len(v["list_idx"])

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
        # self,
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
            # print (f"{seg_idx=}, {seg_id=}")
            shm_init = shared_memory.SharedMemory(name=shared_input)
            init_arr = np.ndarray(init_shape, dtype=float, buffer=shm_init.buf)
            shm_output = shared_memory.SharedMemory(name=shared_output)
            output_arr = np.ndarray(output_shape, dtype=float, buffer=shm_output.buf)

        output_arr[seg_idx, beam_idx, :, :] = block_reduce(
            init_arr == seg_id,
            oversamp_factor,
            func=np.mean,
        )
        # print (np.nansum(output_arr[seg_idx, beam_idx, :, :]))
        if memmap:
            output_arr.flush()

    def _gen_stacked_beams(
        self, pixfrac=1.0, kernel="square", dfillval=0, spatial_scale=1.0
    ):
        # for i, beam in enumerate(self.beams):
        #     print(beam)
        # print(self.PA)
        print(dir(self))
        # print(self.ra, self.dec)
        # print(dir(self.beams[0].beam))
        # return
        test_beam = self.beams[14]
        # print (test_beam.full_2d_wcs)
        # print (test_beam.trace_table)
        # print (test_beam.get_2d_wcs)
        # print(dir(test_beam.direct))
        print(dir(test_beam))
        print(dir(test_beam.beam))
        # print(test_beam.direct.wcs)
        # print(test_beam.grism.wcs)
        print(
            test_beam.beam.xc,
            test_beam.beam.xcenter,
            test_beam.beam.x0,
            test_beam.beam.xoffset,
            test_beam.beam.dx,
            test_beam.beam.dxpix,
            test_beam.beam.yc,
            test_beam.beam.ycenter,
            test_beam.beam.yoffset,
        )
        print(
            test_beam.grism.wcs.all_world2pix(
                [[self.ra, self.dec]],
                1,
                # [test_beam.grism.wcs.wcs.crpix], 1
            ).flatten()
        )
        test_dir = np.zeros((2, *test_beam.beam.direct.shape))
        # return
        # fig, axs = plt.subplots(2,2)
        # for a, d in zip(axs.ravel(), test_beam.direct.data.values()):
        #     a.imshow(d)
        # plt.show()

        from drizzlepac import adrizzle
        from reproject.mosaicking import find_optimal_celestial_wcs, reproject_and_coadd
        from reproject import reproject_adaptive

        adrizzle.log.setLevel("ERROR")
        drizzler = adrizzle.do_driz

        new_multibeam = []

        # data
        for filt, pa_info in self.PA.items():
            # print (pa_info)
            for pa, beam_idxs in pa_info.items():
                # print(pa, beam_idxs)
                # if filt != "F200W":
                #     continue
                # if float(pa) != 341.0:
                #     continue
                # fig, axs = plt.subplots(3, 3, sharex=True, sharey=True)
                # for ax_i, xmult in enumerate([-1, 0, 1]):
                #     for ax_j, ymult in enumerate([-1, 0, 1]):
                #         # for ymult
                #         # print(float(pa), 341.0)
                #         # use_beam = None
                #         # min_off = 1e10
                #         # for i in beam_idxs:
                #         #     if self.beams[i].beam.ycenter < min_off:
                #         #         use_beam = i

                # new_beam = deepcopy(self.beams[beam_idxs[0]])
                # # print(dir(new_beam.direct.wcs))
                # # print(new_beam.direct.wcs)
                # # print(new_beam.grism.wcs)
                # shift_crpix = (
                #     (np.asarray(new_beam.direct.data["REF"].shape)) / 2
                #     - np.array(
                #         test_beam.direct.wcs.all_world2pix(
                #             [[self.ra, self.dec]],
                #             0,
                #             # [test_beam.grism.wcs.wcs.crpix], 1
                #         ).flatten()
                #     )
                #     + 1
                # )

                # # Set centre of direct image to the actual coordinates
                # direct_cen = (np.asarray(new_beam.direct.data["REF"].shape)) / 2
                # # print(cen)
                # new_beam.direct.wcs.wcs.crpix = direct_cen
                # new_beam.direct.wcs.wcs.crval = [self.ra, self.dec]

                # # Align beam to direct
                # direct_cen = (np.asarray(new_beam.direct.data["REF"].shape)) / 2
                # # print(cen)
                # # print ("SHIFT", shift_crpix)
                # new_beam.grism.wcs = grizli_utils.transform_wcs(
                #     new_beam.grism.wcs, translation=shift_crpix
                # )
                #         # new_beam.grism.wcs.wcs.crpix = [
                #         #     # direct_cen[0] + 12,
                #         #     direct_cen[0]-45,
                #         #     direct_cen[1],
                #         # ]
                #         # new_beam.grism.wcs.wcs.crval = [self.ra, self.dec]
                #         # print(new_beam.grism.wcs)

                #         # print("XCenter", [self.beams[i].beam.yoffset for i in beam_idxs])
                #         # print("XCenter", [self.beams[i].beam.xoffset for i in beam_idxs])

                #         sh = new_beam.sh
                #         outsci = np.zeros(sh, dtype=np.float32)
                #         outwht = np.zeros(sh, dtype=np.float32)
                #         outctx = np.zeros(sh, dtype=np.int32)

                #         outvar = np.zeros(sh, dtype=np.float32)
                #         outwv = np.zeros(sh, dtype=np.float32)
                #         outcv = np.zeros(sh, dtype=np.int32)

                #         outdir = np.zeros(
                #             new_beam.direct.data["REF"].shape, dtype=np.float32
                #         )
                #         outwd = np.zeros(
                #             new_beam.direct.data["REF"].shape, dtype=np.float32
                #         )
                #         outcd = np.zeros(
                #             new_beam.direct.data["REF"].shape, dtype=np.int32
                #         )

                #         grism_data = [
                #             self.beams[i].grism.data["SCI"] for i in beam_idxs
                #         ]
                #         direct_data = [
                #             self.beams[i].direct.data["REF"] for i in beam_idxs
                #         ]

                #         # for i, beam in enumerate([self.beams[i] for i in beam_idxs]):
                #         # beam_header, beam_wcs = beam.full_2d_wcs()
                #         # print (beam_wcs, beam.grism.wcs)
                #         # if not hasattr(beam_wcs, "pixel_shape"):
                #         #     beam_wcs.pixel_shape = beam_wcs._naxis1, beam_wcs._naxis2

                #         # if not hasattr(beam_wcs, "_naxis1"):
                #         #     beam_wcs._naxis1, beam_wcs._naxis2 = beam_wcs._naxis
                #         for i, idx in enumerate(beam_idxs):

                #             beam = self.beams[idx]
                #             direct_wcs_i = beam.direct.wcs
                #             grism_wcs_i = beam.grism.wcs.copy()

                #             grism_wcs_i.wcs.crpix = [
                #                 grism_wcs_i.wcs.crpix[0] + xmult * beam.beam.xcenter,
                #                 grism_wcs_i.wcs.crpix[1] + ymult * beam.beam.ycenter,
                #             ]
                #             # print (direct_wcs_i, grism_wcs_i)
                #             # print(beam.beam.ycenter)
                #             # if not hasattr(grism_wcs_i, "_naxis1"):
                #             # grism_wcs_i._naxis1, grism_wcs_i._naxis2 = grism_wcs_i._naxis[::-1]
                #             # grism_wcs_i._naxis =grism_wcs_i._naxis[::-1]

                #             grism_wht = beam.ivar
                #             grism_wht[~np.isfinite(grism_wht)] = 0.0
                #             wht_mask = grism_wht == 0
                #             med_wht = np.median(grism_wht[~wht_mask])
                #             grism_wht[wht_mask] = med_wht
                #             # print (dir(beam.grism.data))
                #             # print (beam.grism.data.keys())
                #             # print (dir(beam.grism))
                #             # plt.imshow(grism_data[i])
                #             # plt.imshow(beam.grism.data["ERR"])
                #             # plt.show()

                #             drizzler(
                #                 direct_data[i],
                #                 direct_wcs_i,
                #                 np.ones_like(direct_data[i]),
                #                 new_beam.direct.wcs,
                #                 outdir,
                #                 outwd,
                #                 outcd,
                #                 1.0,
                #                 "cps",
                #                 1,
                #                 wcslin_pscale=1.0,
                #                 uniqid=1,
                #                 pixfrac=pixfrac,
                #                 kernel=kernel,
                #                 fillval=dfillval,
                #             )
                #             drizzler(
                #                 grism_data[i],
                #                 # beam.grism.wcs,
                #                 grism_wcs_i,
                #                 # grism_wht,
                #                 np.ones_like(grism_data[i]),
                #                 new_beam.grism.wcs,
                #                 outsci,
                #                 outwht,
                #                 outctx,
                #                 1.0,
                #                 "cps",
                #                 1,
                #                 wcslin_pscale=1.0,
                #                 uniqid=1,
                #                 pixfrac=pixfrac,
                #                 kernel=kernel,
                #                 fillval=dfillval,
                #             )

                #         # Correct for drizzle scaling
                #         area_ratio = 1.0 / new_beam.grism.wcs.pscale ** 2

                #         # Preserve flux (has to preserve aperture flux along spatial axis but
                #         # average in spectral axis).
                #         # area_ratio *= spatial_scale

                #         # preserve flux density
                #         flux_density_scale = spatial_scale ** 2

                #         # science
                #         outsci *= area_ratio * flux_density_scale

                #         axs[ax_i, ax_j].imshow((outsci), vmax=0.5, origin="lower")
                #         # axs[ax_i, ax_j].imshow((outdir), vmax=0.0012, origin="lower")
                #         # plt.show()
                #         # plt.imshow(np.sqrt(outdir), origin="lower")
                # plt.show()

                # print("PSCALE", new_beam.grism.wcs.pscale)

                new_beam = deepcopy(self.beams[beam_idxs[0]])
                # print(dir(new_beam.direct.wcs))
                # print(new_beam.direct.wcs)
                # print(new_beam.grism.wcs)

                # Set centre of direct image to the actual coordinates
                direct_cen = (np.asarray(new_beam.direct.data["REF"].shape) + 1) / 2
                print (f"{direct_cen=}")

                shift_crpix = (
                    # (np.asarray(new_beam.direct.data["REF"].shape) + 1) / 2
                    direct_cen
                    - np.array(
                        new_beam.direct.wcs.all_world2pix(
                            [[self.ra, self.dec]],
                            1,
                            # [test_beam.grism.wcs.wcs.crpix], 1
                        ).flatten()
                    )
                    # -1
                )

                # # print(cen)
                # new_beam.direct.wcs.wcs.crpix = direct_cen
                # new_beam.direct.wcs.wcs.crval = [self.ra, self.dec]

                # Align beam to direct
                # direct_cen = (np.asarray(new_beam.direct.data["REF"].shape)+1) / 2
                # print(cen)
                print("SHIFT", shift_crpix)

                print(new_beam.grism.wcs.wcs.crval)
                print(self.ra, self.dec)

                new_beam.grism.wcs = grizli_utils.transform_wcs(
                    new_beam.grism.wcs, translation=shift_crpix
                )
                new_beam.direct.wcs = grizli_utils.transform_wcs(
                    new_beam.direct.wcs, translation=shift_crpix
                )

                # print (new_beam.grism.wcs.wcs.crval)
                # print (self.ra, self.dec)
                # exit()

                sh = new_beam.sh
                outsci = np.zeros(sh, dtype=np.float32)
                outwht = np.zeros(sh, dtype=np.float32)
                outctx = np.zeros(sh, dtype=np.int32)

                outvar = np.zeros(sh, dtype=np.float32)
                outwv = np.zeros(sh, dtype=np.float32)
                outcv = np.zeros(sh, dtype=np.int32)

                outcon = np.zeros(sh, dtype=np.float32)
                outwc = np.zeros(sh, dtype=np.float32)
                outcc = np.zeros(sh, dtype=np.int32)

                outdir = np.zeros(new_beam.direct.data["REF"].shape, dtype=np.float32)
                outwd = np.zeros(new_beam.direct.data["REF"].shape, dtype=np.float32)
                outcd = np.zeros(new_beam.direct.data["REF"].shape, dtype=np.int32)

                grism_data = [self.beams[i].grism.data["SCI"] for i in beam_idxs]
                direct_data = [self.beams[i].beam.direct for i in beam_idxs]
                # print ("SEG", new_beam.beam.seg)

                new_seg = grizli_utils.blot_nearest_exact(
                    new_beam.beam.seg,
                    self.beams[beam_idxs[0]].direct.wcs,
                    new_beam.direct.wcs,
                    verbose=False,
                    stepsize=-1,
                    scale_by_pixel_area=False,
                )

                for i, idx in enumerate(beam_idxs):

                    beam = self.beams[idx]
                    direct_wcs_i = beam.direct.wcs
                    grism_wcs_i = beam.grism.wcs.copy()

                    # grism_wcs_i.wcs.crpix = [
                    #     grism_wcs_i.wcs.crpix[0] - beam.beam.xcenter,
                    #     grism_wcs_i.wcs.crpix[1] - beam.beam.ycenter,
                    # ]
                    # grism_wcs_i.wcs.crpix = [
                    #     grism_wcs_i.wcs.crpix[0],
                    #     grism_wcs_i.wcs.crpix[1],
                    # ]
                    # plt.imshow(beam.ivar)
                    # plt.show()
                    # grism_wht = beam.ivar
                    contam_weight = np.exp(
                        -(self.fcontam * np.abs(beam.contam) * np.sqrt(beam.ivar))
                    )
                    grism_wht = beam.ivar * contam_weight
                    grism_wht[~np.isfinite(grism_wht)] = 0.0
                    contam_wht = beam.ivar
                    contam_wht[~np.isfinite(contam_wht)] = 0.0
                    # wht_mask = grism_wht == 0
                    # med_wht = np.median(grism_wht[~wht_mask])
                    # grism_wht[wht_mask] = med_wht

                    drizzler(
                        direct_data[i],
                        direct_wcs_i,
                        np.ones_like(direct_data[i]),
                        new_beam.direct.wcs,
                        outdir,
                        outwd,
                        outcd,
                        1.0,
                        "cps",
                        1,
                        wcslin_pscale=1.0,
                        uniqid=1,
                        pixfrac=pixfrac,
                        kernel=kernel,
                        fillval=dfillval,
                    )
                    drizzler(
                        grism_data[i],
                        # beam.grism.wcs,
                        grism_wcs_i,
                        grism_wht,
                        # np.ones_like(grism_data[i]),
                        new_beam.grism.wcs,
                        outsci,
                        outwht,
                        outctx,
                        1.0,
                        "cps",
                        1,
                        wcslin_pscale=1.0,
                        uniqid=1,
                        pixfrac=pixfrac,
                        kernel=kernel,
                        fillval=dfillval,
                    )
                    drizzler(
                        beam.contam,
                        # beam.grism.wcs,
                        grism_wcs_i,
                        contam_wht,
                        # np.ones_like(grism_data[i]),
                        new_beam.grism.wcs,
                        outcon,
                        outwc,
                        outcc,
                        1.0,
                        "cps",
                        1,
                        wcslin_pscale=1.0,
                        uniqid=1,
                        pixfrac=pixfrac,
                        kernel=kernel,
                        fillval=dfillval,
                    )

                    drizzler(
                        contam_weight,
                        grism_wcs_i,
                        grism_wht,
                        new_beam.grism.wcs,
                        outvar,
                        outwv,
                        outcv,
                        1.0,
                        "cps",
                        1,
                        wcslin_pscale=1.0,
                        uniqid=1,
                        pixfrac=pixfrac,
                        kernel=kernel,
                        fillval=dfillval,
                    )

                # print(
                #     self.id,
                #     # new_direct
                #     # new_seg,
                #     np.nanmedian(
                #         np.asarray([self.beams[i].direct.origin for i in beam_idxs]),
                #         axis=0,
                #     ),
                #     np.nanmedian(
                #         np.asarray([self.beams[i].direct.pad for i in beam_idxs]),
                #         axis=0,
                #     ),
                #     self.beams[beam_idxs[0]].beam.beam,
                #     self.beams[beam_idxs[0]].beam.conf,
                #     0,
                #     0,
                #     self.beams[beam_idxs[0]].beam.fwcpos,
                #     self.beams[beam_idxs[0]].beam.MW_EBV,
                #     0.0,
                #     0.0,
                # )

                # # data_list = [self.beams[i].grism.data["SCI"] for i in beam_idxs]
                # # wcs_list = [self.beams[i].grism.wcs for i in beam_idxs]
                # data_list = [self.beams[i].direct.data["REF"] for i in beam_idxs]
                # wcs_list = [self.beams[i].direct.wcs for i in beam_idxs]
                # shape_list = [d.shape for d in data_list]
                # print([self.beams[i].direct.pad for i in beam_idxs])

                # stack_wcs, stack_shape = find_optimal_celestial_wcs(
                #     [*zip(shape_list, wcs_list)]
                # )
                # array, footprint = reproject_and_coadd(
                #     [*zip(data_list, wcs_list)],
                #     #    stack_wcs,
                #     new_beam.direct.wcs,
                #     shape_out=new_beam.direct.data["REF"].shape,
                #     reproject_function=reproject_adaptive,
                # )
                # # plt.imshow(np.sqrt(array))
                # # plt.show()

                # Correct for drizzle scaling
                area_ratio = 1.0 / new_beam.grism.wcs.pscale**2

                # Preserve flux (has to preserve aperture flux along spatial axis but
                # average in spectral axis).
                # area_ratio *= spatial_scale

                # preserve flux density
                flux_density_scale = spatial_scale**2

                # science
                outsci *= area_ratio * flux_density_scale
                # Direct
                outdir *= area_ratio * flux_density_scale
                # Variance
                outvar *= area_ratio / outwv * flux_density_scale**2
                outwht = 1 / outvar
                outwht[(outvar == 0) | (~np.isfinite(outwht))] = 0
                # Contam
                outcon *= area_ratio * flux_density_scale

                new_beam.grism.data["SCI"] = outsci
                new_beam.grism.data["ERR"] = np.sqrt(outvar)
                new_beam.grism.data["DQ"] = np.zeros_like(outsci)
                new_beam.contam = outcon
                new_beam.direct.data["REF"] = outdir

                # flam_scale = np.nanmedian(new_beam.direct.data["REF"]/new_beam.beam.direct)
                # print (f"{flam_scale=}")
                # fig, axs = plt.subplots(1,3, sharex=True,sharey=True)
                # axs[0].imshow(np.log10(new_beam.beam.direct))
                # axs[1].imshow(np.log10(outdir))
                # axs[2].imshow(np.log10(new_beam.direct.data["REF"]))
                # # axs[2].imshow(new_beam.beam.direct-outdir)
                # plt.show()
                # return

                new_beam.beam = GrismDisperser(
                    id=self.id,
                    direct=outdir,
                    segmentation=new_seg,
                    origin=np.nanmedian(
                        np.asarray([self.beams[i].direct.origin for i in beam_idxs]),
                        axis=0,
                    ),
                    pad=np.nanmedian(
                        np.asarray([self.beams[i].direct.pad for i in beam_idxs]),
                        axis=0,
                    ),
                    grow=np.nanmedian(
                        np.asarray([self.beams[i].direct.grow for i in beam_idxs]),
                        axis=0,
                    ),
                    beam=self.beams[beam_idxs[0]].beam.beam,
                    xcenter=0,
                    ycenter=0,
                    conf=self.beams[beam_idxs[0]].beam.conf,
                    fwcpos=self.beams[beam_idxs[0]].beam.fwcpos,
                    MW_EBV=self.beams[beam_idxs[0]].beam.MW_EBV,
                    xoffset=0.0,
                    yoffset=0.0,
                )
                # print(f"{self.min_sens=}")
                new_beam._parse_from_data(
                    isJWST=True,
                    contam_sn_mask=[10, 3],
                    min_mask=self.min_mask,
                    min_sens=self.min_sens,
                    mask_resid=self.mask_resid,
                )
                new_beam.compute_model()

                # fig, axs = plt.subplots(4, 1, sharex=True, sharey=True)
                # axs[0].imshow(np.sqrt(new_beam.grism.data["SCI"]), origin="lower")
                # axs[2].imshow(np.sqrt(new_beam.model), origin="lower")
                # axs[1].imshow(np.sqrt(outsci-outcon), origin="lower")
                # axs[3].imshow(np.sqrt(np.sqrt(outvar)))
                # # axs
                # # scale = np.nansum(new_beam.model)/np.nansum(outsci)
                # # scale *= 5
                # # axs[3].imshow(np.sqrt(np.sqrt(outvar)))
                # # axs[2].imshow(new_beam.grism.data["DQ"], origin="lower")
                # # axs[2].imshow(new_beam.ivar, origin="lower")
                # # axs[3].imshow(new_beam.contam, origin="lower")
                # # axs[3].imshow(outsci - new_beam.model/scale, origin="lower")
                # # axs[0].imshow(new_beam.beam.direct)
                # # axs[1].imshow(self.beams[beam_idxs[0]].beam.direct)
                # plt.show()

                # print(new_beam.model)

                # return
                new_multibeam.append(new_beam)
                if filt=="F115W":
                    if float(pa)==341.0:
                        test_dir[0]= outdir
                    else:
                        test_dir[1]= outdir
            
        # fig, axs = plt.subplots(1,3,sharex=True,sharey=True)
        # axs[0].imshow(test_dir[0])
        # axs[1].imshow(np.rot90(test_dir[1], k=3))
        # axs[2].imshow(test_dir[0]-np.rot90(test_dir[1],k=3))
        # print (test_dir[0].shape, new_multibeam[0].direct.wcs)
        # plt.show()
        # exit()


        return new_multibeam

    def fit_at_z(
        self,
        z: float = 0.0,
        fit_background: bool = True,
        poly_order: int = 0,
        n_samples: int = 3,
        num_iters: int = 10,
        spec_wavs: ArrayLike | None = None,
        oversamp_factor: int = 3,
        veldisp: float = 500,
        fit_stacks: bool = True,
        direct_images: None = None,
        temp_dir: PathLike | None = None,
        memmap: bool = False,
        out_dir: PathLike | None = None,
        cpu_count: int = -1,
        overwrite: bool = False,
        use_lines: dict = CLOUDY_LINE_MAP,
        save_lines: bool = True,
        save_stacks: bool = True,
        pline: dict = DEFAULT_PLINE,
        nnls_iters: int = 100,
        seed: int = 2744,
        nnls_method: str = "scipy",
        **grizli_kwargs,
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
        num_iters : int, optional
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
            to a factor of ``3``.
        veldisp : float, optional
            The velocity dispersion of the template spectra in km/s, by
            default ``500``.
        fit_stacks : bool, optional
            Fit stacked spectra, by default ``True``. May be redundant.
        direct_images : _type_, optional
            WIP, may allow for changing beam direct images at some point.
            By default ``None``.
        temp_dir : PathLike | None, optional
            The temporary directory to use for memmapped files (if
            ``memmap==True``). If ``None`` (default), the current working
            directory will be used.
        memmap : bool, optional
            Whether to use a memmap to store large files. By default
            ``True`` (``False`` is still WIP).
        out_dir : PathLike | None, optional
            Where the output files will be written. If ``None`` (default),
            the current working directory will be used.
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
            Defaults to `~glass_niriss.grism.DEFAULT_PLINE`.
        **grizli_kwargs : dict, optional
            A catch-all for previous `grizli` keywords. These are
            redundant in the multiregion method, and are kept only to
            avoid keyword argument conflicts with the standard `grizli`
            function.

        Returns
        -------
        tuple
            Exact form of return still WIP.
        """
        try:
            import sklearn.linear_model

            HAS_SKLEARN = True
        except:
            HAS_SKLEARN = False

        try:
            import adelie

            HAS_ADELIE = True
        except:
            HAS_ADELIE = False

        import numpy.linalg
        import scipy.optimize
        # from scipy.optimize import _cython_nnls
        # from scipy.optimize import _nnls
        # print (dir(_nnls))
        # exit()

        if spec_wavs is None:
            spec_wavs = np.arange(10000.0, 23000.0, 45.0)

        if fit_stacks:
            new_beams = self._gen_stacked_beams()
            mb_obj = MultiBeam(
                new_beams,
                group_name=self.group_name,
                fcontam=self.fcontam,
                min_mask=self.min_mask,
                min_sens=self.min_sens,
                mask_resid=self.mask_resid,
            )
            mb_obj.fit_trace_shift()
        else:
            mb_obj = self

        # print(self.group_name, self.id)
        if memmap:
            if temp_dir is None:
                temp_dir = Path.cwd()
            temp_dir.mkdir(exist_ok=True, parents=True)

        # print 'xxx Init poly'
        mb_obj.init_poly_coeffs(poly_order=poly_order)

        # print 'xxx Init bg'
        if fit_background:
            self.fit_bg = True
            A = np.vstack((mb_obj.A_bg, mb_obj.A_poly))
        else:
            self.fit_bg = False
            A = mb_obj.A_poly * 1

        rng = np.random.default_rng(seed=seed)

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
            n_fit_comps = test_post["samples2d"].shape[1]
            n_post_samples = test_post["samples2d"].shape[0]

        if out_dir is None:
            out_dir = Path.cwd()
        out_dir.mkdir(exist_ok=True, parents=True)
        if memmap:
            if temp_dir is None:
                temp_dir = Path.cwd()
            temp_dir.mkdir(exist_ok=True, parents=True)

        # print("Exiting code now")
        # exit()

        # with SharedMemoryManager() as smm:
        smm = SharedMemoryManager()
        smm.start()

        # Try to allow for both memory and file-backed multiprocessing of
        # large arrays
        oversamp_seg_maps_shape = (
            self.n_regions,
            mb_obj.N,
            mb_obj.beams[0].beam.sh[0],
            mb_obj.beams[0].beam.sh[1],
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
            oversamp_factor * mb_obj.beams[0].beam.sh[0],
            oversamp_factor * mb_obj.beams[0].beam.sh[1],
        )
        # print (np.prod(oversampled_shape))
        # exit()
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
            zip(mb_obj.beams, mb_obj.Nflat)
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
            beam_wcs.wcs.cd /= oversamp_factor
            beam_wcs.wcs.crpix *= oversamp_factor

            oversampled[:] = reproject_interp(
                (self.regions_seg_map, self.regions_seg_wcs),
                beam_wcs,
                oversampled_shape,
                return_footprint=False,
                order=0,
            )
            # print (test.dtype)
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

        # print(np.nansum(oversamp_seg_maps))

        oversamp_seg_maps[~np.isfinite(oversamp_seg_maps)] = 0

        # print(np.nansum(oversamp_seg_maps))

        # plt.imshow(oversamp_seg_maps[0, 0])
        # plt.show()

        NTEMP = self.n_regions * n_samples
        num_stacks = len([*beam_info.keys()])
        stacked_shape = np.nansum([np.prod(v["2d_shape"]) for v in beam_info.values()])
        temp_offset = A.shape[0] - mb_obj.N + num_stacks

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

        print("MEMMAP", memmap)

        # shm_stacked_A = smm.SharedMemory(
        #     size=np.dtype(float).itemsize * np.prod(stacked_A_shape)
        # )
        # stacked_A = np.ndarray(
        #     stacked_A_shape,
        #     dtype=float,
        #     buffer=shm_stacked_A.buf,
        # )
        stacked_A.fill(0.0)

        print(stacked_A.shape, stacked_A_shape)

        # Offset for background
        if fit_background:
            off = 0.04
        else:
            off = 0.0
        stacked_scif = np.zeros(stacked_shape)

        stacked_ivarf = np.zeros(stacked_shape)
        stacked_weightf = np.zeros(stacked_shape)
        stacked_fit_mask = np.zeros(stacked_shape, dtype=bool)
        start_idx = 0
        for k_i, (k, v) in enumerate(beam_info.items()):
            stack_idxs = np.r_["0,2", *v["flat_slice"]]
            # stacked_scif[start_idx:start_idx+np.prod(v["2d_shape"])] = np.nansum(
            #     ((self.scif[stack_idxs])*self.ivarf[stack_idxs])*self.fit_mask[stack_idxs], axis=0
            # )/np.nansum(
            #     (self.ivarf[stack_idxs])*self.fit_mask[stack_idxs], axis=0
            # )
            _scif = mb_obj.scif[stack_idxs]
            # _scif[~self.fit_mask[stack_idxs]] = np.nan
            stacked_scif[start_idx : start_idx + np.prod(v["2d_shape"])] = np.nanmedian(
                # (self.scif[stack_idxs]),
                _scif,
                axis=0,
            )
            stacked_weightf[start_idx : start_idx + np.prod(v["2d_shape"])] = (
                np.nanmedian(
                    mb_obj.weightf[stack_idxs],
                    axis=0,
                )
            )
            stacked_fit_mask[start_idx : start_idx + np.prod(v["2d_shape"])] = np.any(
                mb_obj.fit_mask[stack_idxs],
                axis=0,
                # dtype=bool
            )
            if fit_background:
                stacked_A[k_i, start_idx : start_idx + np.prod(v["2d_shape"])] = 1.0
                stacked_A[
                    num_stacks : num_stacks + mb_obj.A_poly.shape[0],
                    start_idx : start_idx + np.prod(v["2d_shape"]),
                ] = np.nanmean(mb_obj.A_poly[:, stack_idxs], axis=1)
            else:
                stacked_A[
                    num_stacks : num_stacks + mb_obj.A_poly.shape[0],
                    start_idx : start_idx + np.prod(v["2d_shape"]),
                ] = np.nanmean(mb_obj.A_poly[:, stack_idxs], axis=1)

            stacked_ivarf[start_idx : start_idx + np.prod(v["2d_shape"])] = (
                np.nanmedian(mb_obj.ivarf[stack_idxs], axis=0)
            )

            start_idx += np.prod(v["2d_shape"])

        stacked_fit_mask &= np.isfinite(stacked_scif)

        coeffs_tracker = []
        chi2_tracker = []

        output_table = Table(
            names=["iteration", "chi2", "rows"]
            + [f"base_coeffs_{b}" for b in np.arange(temp_offset)]
            + [
                f"{p[0]}_{p[1]}"
                for p in product(self.regions_phot_cat["bin_id"], np.arange(n_samples))
            ],
            dtype=[int, float, str] + [float] * stacked_A.shape[0],
        )

        output_table_path = out_dir / (
            f"{self.id}_{len(np.unique(self.regions_phot_cat["bin_id"]))}"
            f"bins_{num_iters}iters_{n_samples}samples_z_{z}_sig_{veldisp}.csv"
        )

        # !! This can be placed outside the iteration loop
        stacked_fn = partial(
            self._gen_stacked_templates_from_pipes,
            shared_seg_name=(
                temp_dir / "memmap_oversamp_seg_maps.dat"
                if memmap
                else shm_seg_maps.name
            ),
            seg_maps_shape=oversamp_seg_maps_shape,
            # shared_seg_name=shm_seg_maps.name,
            # seg_maps_shape=oversamp_seg_maps_shape,
            shared_models_name=(
                temp_dir / "memmap_stacked_A.dat" if memmap else shm_stacked_A.name
            ),
            models_shape=stacked_A_shape,
            # shared_models_name= shm_stacked_A.name,
            # models_shape=stacked_A_shape,
            posterior_dir=str(self.pipes_dir / "posterior" / self.run_name),
            n_samples=n_samples,
            spec_wavs=spec_wavs,
            beam_info=beam_info,
            temp_offset=temp_offset,
            cont_only=False,
            rm_line=None,
            memmap=memmap,
        )

        if (not output_table_path.is_file()) or (overwrite):

            output_table.write(output_table_path, overwrite=True, format="pandas.csv")

            iterations = np.arange(num_iters)+1
            for iteration in iterations:
                if iteration==iterations[-1]:

                    best_iter = np.argmin(output_table["chi2"])

                    rows = [int(s) for s in output_table["rows"][best_iter].split(";")]
                else:
                    rows = rng.choice(
                        np.arange(n_post_samples, dtype=int), size=n_samples, replace=False
                    )
                print(f"Iteration {iteration}, {rows=}")

                print("Generating models...")
                with multiprocessing.Pool(
                    processes=cpu_count,
                    initializer=_init_pipes_sampler,
                    initargs=(
                        fit_instructions,
                        veldisp,
                        mb_obj.beams,
                    ),
                ) as pool:
                    for s_i, s in enumerate(self.regions_seg_ids):
                        pool.apply_async(
                            stacked_fn,
                            (s_i, s),
                            kwds={"rows": rows},
                        )
                    pool.close()
                    pool.join()

                print("Generating models... DONE")
                ok_temp = np.sum(stacked_A, axis=1) > 0

                stacked_A_contig = np.ascontiguousarray(stacked_A[:, ::100])
                # np.unique() finds identical items in a raveled array. To make it
                # see each row as a single item, we create a view of each row as a
                # byte string of length itemsize times number of columns in `ar`
                ar_row_view = stacked_A_contig.view(
                    "|S%d" % (stacked_A_contig.itemsize * stacked_A_contig.shape[1])
                )
                _, unique_idxs = np.unique(ar_row_view, return_index=True)
                unique_temp = np.isin(np.arange(stacked_A.shape[0]), unique_idxs)
                del stacked_A_contig

                # print (np.nansum(ok_temp), np.nansum(unique_temp), np.nansum(stacked_fit_mask&np.isfinite(stacked_scif)))

                ok_temp &= unique_temp

                out_coeffs = np.zeros(stacked_A.shape[0])
                # stacked_fit_mask &= np.isfinite(stacked_scif)
                stacked_Ax = stacked_A[:, stacked_fit_mask][ok_temp, :].T
                # print(stacked_A[:, stacked_fit_mask].base.shape)
                # print (stacked_A[:, stacked_fit_mask][ok_temp, :].base.shape) is a copy
                # print(stacked_A[:, stacked_fit_mask][ok_temp, :].T.base.shape)
                # Weight by ivar
                stacked_Ax *= np.sqrt(stacked_ivarf[stacked_fit_mask][:, np.newaxis])
                print(
                    f"Array size reduced from {stacked_A.shape} to "
                    f"{stacked_Ax.shape[::-1]} ("
                    f"{stacked_Ax.shape[0]/stacked_A.shape[-1]:.1%} of original)"
                )
                print("NNLS fitting...")

                t0 = time()

                if fit_background:
                    # Proof of concept, this can be improved with a 
                    # proper nnls_kwargs dict, and passing tolerance/iters
                    # as a double-valued list
                    if iteration==iterations[-1]:
                        nnls_iters = 1e5
                        nnls_tol = 1e-10
                    else:
                        nnls_tol = 1e-6
                    
                    y = stacked_scif[stacked_fit_mask] + off
                    y *= np.sqrt(stacked_ivarf[stacked_fit_mask])

                    if nnls_method == "scipy":

                        coeffs, rnorm, info = scipy.optimize._nnls._nnls(
                            stacked_Ax, y + off, nnls_iters
                        )
                        # coeffs, rnorm, info = _cython_nnls._nnls(
                        #     stacked_Ax, y + off, nnls_iters
                        # )
                        coeffs[:num_stacks] -= off

                    elif nnls_method == "numba":

                        nnls_solver = CDNNLS(stacked_Ax, y + off)
                        # nnls_solver._run(stacked_Ax.shape[1]*3)
                        nnls_solver.run(n_iter=nnls_iters)
                        coeffs = nnls_solver.w
                        coeffs[:num_stacks] -= off

                    elif nnls_method == "adelie" and HAS_ADELIE:
                        state = adelie.solver.bvls(
                            stacked_Ax,
                            y + off,
                            lower=np.zeros(stacked_Ax.shape[-1]),
                            upper=np.full(stacked_Ax.shape[-1], np.inf),
                            max_iters=int(nnls_iters),
                            tol=nnls_tol
                            # n_threads=20 # Inter-thread communication is actually slower
                        )
                        state.solve()
                        print (state.iters)
                        coeffs = state.beta
                        coeffs[:num_stacks] -= off


                else:
                    y = mb_obj.scif[mb_obj.fit_mask]
                    y *= np.sqrt(mb_obj.ivarf[mb_obj.fit_mask])

                    coeffs, rnorm = scipy.optimize.nnls(Ax, y)

                    Ax = A[:, mb_obj.fit_mask][ok_temp, :].T
                    # Weight by ivar
                    Ax *= np.sqrt(mb_obj.ivarf[mb_obj.fit_mask][:, np.newaxis])

                print("NNLS fitting... DONE")
                print(f"Time taken {time()-t0}s")

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
                        ";".join(f"{r:0>3}" for r in rows),
                        *out_coeffs,
                    ]
                )
                output_table.write(output_table_path, overwrite=True)
                print(f"Iteration {iteration}: chi2={chi2:.3f}\n")
                stacked_A[temp_offset:].fill(0.0)

            del stacked_Ax

        else:
            print("Loading previous fit...")
            output_table = Table.read(output_table_path, format="pandas.csv")

        best_iter = np.argmin(output_table["chi2"])

        out_coeffs = np.array(output_table[best_iter][3:])
        if fit_background:
            for k_i, (k, v) in enumerate(beam_info.items()):
                for ib in v["list_idx"]:
                    mb_obj.beams[ib].background = out_coeffs[k_i]

        best_rows = [int(s) for s in output_table["rows"][best_iter].split(";")]

        if save_stacks:
            print("Generating models...")
            # import traceback
            with multiprocessing.Pool(
                processes=cpu_count,
                initializer=_init_pipes_sampler,
                initargs=(
                    fit_instructions,
                    veldisp,
                    mb_obj.beams,
                ),
            ) as pool:
                for s_i, s in enumerate(self.regions_seg_ids):
                    pool.apply_async(
                        stacked_fn,
                        (s_i, s),
                        kwds={"rows": best_rows},
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
                    mb_obj.beams,
                ),
            ) as pool:
                for s_i, s in enumerate(self.regions_seg_ids):
                    pool.apply_async(
                        stacked_fn,
                        (s_i, s),
                        kwds={"rows": best_rows, "cont_only": True},
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
                        mb_obj.beams[0].beam.conf.conf_file,
                        "Configuration file",
                    )
                stacked_hdul.extend(hdus)

                start_idx += np.prod(v["2d_shape"])

            stacked_hdul.writeto(
                f"regions_{self.id:05d}_z_{z}_stacked.fits",
                output_verify="silentfix",
                overwrite=True,
            )

        if save_lines:
            beam_models_len = 0
            for k_i, (k, v) in enumerate(beam_info.items()):
                beam_models_len += np.prod(v["2d_shape"]) * len(v["list_idx"])

            # shm_beam_models = smm.SharedMemory(
            #     size=np.dtype(float).itemsize * beam_models_len
            # )
            # flat_beam_models = np.ndarray(
            #     (beam_models_len),
            #     dtype=float,
            #     buffer=shm_beam_models.buf,
            # )

            if memmap:
                flat_beam_models = np.memmap(
                    temp_dir / "memmap_beams_model.dat",
                    dtype=float,
                    mode="w+",
                    shape=(beam_models_len),
                )
            else:
                # shm_stacked_A = smm.SharedMemory(
                #     size=np.dtype(float).itemsize * np.prod(stacked_A_shape)
                # )
                # stacked_A = np.ndarray(
                #     stacked_A_shape,
                #     dtype=float,
                #     buffer=shm_stacked_A.buf,
                # )

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
                # shared_seg_name=shm_seg_maps.name,
                # seg_maps_shape=oversamp_seg_maps.shape,
                # shared_models_name=shm_beam_models.name,
                # models_shape=flat_beam_models.shape,
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
                n_samples=n_samples,
                spec_wavs=spec_wavs,
                beam_info=beam_info,
                temp_offset=temp_offset,
                cont_only=False,
                rows=[int(s) for s in output_table["rows"][best_iter].split(";")],
                coeffs=output_table[best_iter],
                memmap=memmap,
            )

            for l_i, l_v in enumerate(use_lines):

                if not check_coverage(l_v["wave"] * (1 + z)):
                    continue

                print(f"Generating map for {l_v["grizli"]}...")

                flat_beam_models.fill(0.0)

                with multiprocessing.Pool(
                    processes=cpu_count,
                    initializer=_init_pipes_sampler,
                    initargs=(fit_instructions, veldisp, mb_obj.beams),
                ) as pool:
                    for s_i, s in enumerate(self.regions_phot_cat["bin_id"]):
                        pool.apply_async(
                            beams_fn,
                            args=(s_i, s),
                            kwds={"rm_line": l_v["cloudy"]},
                            error_callback=print,
                        )
                    pool.close()
                    pool.join()

                i0 = 0
                start_idx = 0
                for k_i, (k, v) in enumerate(beam_info.items()):
                    for ib in v["list_idx"]:
                        mb_obj.beams[ib].beam.model = flat_beam_models[
                            i0 : i0 + np.prod(v["2d_shape"])
                        ].reshape(v["2d_shape"])
                        i0 += np.prod(v["2d_shape"])
                    start_idx += np.prod(v["2d_shape"])

                hdu = drizzle_to_wavelength(
                    mb_obj.beams,
                    ra=self.ra,
                    dec=self.dec,
                    # wave=line_wave_obs,
                    wave=l_v["wave"] * (1 + z),
                    fcontam=mb_obj.fcontam,
                    **pline,
                )

                hdu[0].header["REDSHIFT"] = (z, "Redshift used")
                hdu[0].header["CHI2"] = output_table["chi2"][best_iter]
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
                segm = mb_obj.drizzle_segmentation(wcsobj=line_wcs)
                seg_hdu = fits.ImageHDU(data=segm.astype(np.int32), name="SEG")
                line_hdu.insert(1, seg_hdu)

                line_hdu.writeto(
                    f"regions_{self.id:05d}_z_{z}_{pline.get("pixscale", 0.06)}arcsec.line.fits",
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
            poly_coeffs = out_coeffs[num_stacks : num_stacks + mb_obj.n_poly]
        else:
            poly_coeffs = out_coeffs[: mb_obj.n_poly]

        # print(self.n_poly, self.N)
        # print(num_stacks)
        # print(f"{out_coeffs=}")
        # print(poly_coeffs, self.x_poly)
        mb_obj.y_poly = np.dot(poly_coeffs, mb_obj.x_poly)
        # x_poly = self.x_poly[1,:]+1 = self.beams[0].beam.lam/1.e4

        # return A, out_coeffs, chi2, stacked_modelf
        return
