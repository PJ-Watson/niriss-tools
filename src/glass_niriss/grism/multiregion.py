"""
Functions and classing related to multi-region grism fitting.
"""

import multiprocessing
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
from grizli.multifit import MultiBeam
from numpy.typing import ArrayLike
from pandas._libs.hashtable import SIZE_HINT_LIMIT, duplicated
from pandas.core import algorithms
from pandas.core.sorting import get_group_index
from reproject import reproject_interp

from glass_niriss.grism.specgen import ExtendedModelGalaxy

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

__all__ = ["BagpipesSampler", "RegionsMultiBeam", "CDNNLS"]

pipes_sampler = None
beams_object = None
# scaled_seg_maps = None

import numba
import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from scipy import sparse


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
                print("classic")
                self._run_classic(n_iter, epsilon, **kwargs)
            case "antilop":
                print("antilop")
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


class BagpipesSampler(object):
    """
    Bagpipes model galaxy sampler.

    Parameters
    ----------
    fit_instructions : dict
        A dictionary containing information about the model to be
        generated.
    veldisp : float, optional
        The velocity dispersion of the model galaxy in km/s. By default
        ``veldisp=500``.
    """

    def __init__(self, fit_instructions: dict, veldisp: float = 500.0):

        self.fit_instructions = deepcopy(fit_instructions)
        self.model_components = deepcopy(fit_instructions)
        self.model_components["veldisp"] = veldisp

        # self._set_constants()
        self._process_fit_instructions()
        self.model_gal = None
        # print (fit_instructions)

    def _process_fit_instructions(self) -> None:
        """
        Load the fit instructions and populate the class attributes.

        Originally from `bagpipes.fitting.fitted_model`.
        """
        all_keys = []  # All keys in fit_instructions and subs
        all_vals = []  # All vals in fit_instructions and subs

        self.params = []  # Parameters to be fitted
        self.limits = []  # Limits for fitted parameter values
        self.pdfs = []  # Probability densities within lims
        self.hyper_params = []  # Hyperparameters of prior distributions
        self.mirror_pars = {}  # Params which mirror a fitted param

        # Flatten the input fit_instructions dictionary.
        for key in list(self.fit_instructions):
            if not isinstance(self.fit_instructions[key], dict):
                all_keys.append(key)
                all_vals.append(self.fit_instructions[key])

            else:
                for sub_key in list(self.fit_instructions[key]):
                    all_keys.append(key + ":" + sub_key)
                    all_vals.append(self.fit_instructions[key][sub_key])

        # Sort the resulting lists alphabetically by parameter name.
        indices = np.argsort(all_keys)
        all_vals = [all_vals[i] for i in indices]
        all_keys.sort()

        # Find parameters to be fitted and extract their priors.
        for i in range(len(all_vals)):
            # R_curve cannot be fitted and is either unset or must be a 2D numpy array
            if not all_keys[i] == "R_curve":
                if isinstance(all_vals[i], tuple):
                    self.params.append(all_keys[i])
                    self.limits.append(all_vals[i])  # Limits on prior.

                    # Prior probability densities between these limits.
                    prior_key = all_keys[i] + "_prior"
                    if prior_key in list(all_keys):
                        self.pdfs.append(all_vals[all_keys.index(prior_key)])

                    else:
                        self.pdfs.append("uniform")

                    # Any hyper-parameters of these prior distributions.
                    self.hyper_params.append({})
                    for i in range(len(all_keys)):
                        if all_keys[i].startswith(prior_key + "_"):
                            hyp_key = all_keys[i][len(prior_key) + 1 :]
                            self.hyper_params[-1][hyp_key] = all_vals[i]

                # Find any parameters which mirror the value of a fit param.
                if all_vals[i] in all_keys:
                    self.mirror_pars[all_keys[i]] = all_vals[i]

                if all_vals[i] == "dirichlet":
                    n = all_vals[all_keys.index(all_keys[i][:-6])]
                    comp = all_keys[i].split(":")[0]
                    for j in range(1, n):
                        self.params.append(comp + ":dirichletr" + str(j))
                        self.pdfs.append("uniform")
                        self.limits.append((0.0, 1.0))
                        self.hyper_params.append({})

        # Find the dimensionality of the fit
        self.ndim = len(self.params)

    def _update_model_components(
        self,
        param: ArrayLike,
    ) -> dict:
        """
        Generates a model object with the current parameters.

        Originally from `bagpipes.fitting.fitted_model`, modified to allow
        for running in parallel (no attributes overwritten).

        Returns
        -------
        new_components : dict
            The updated components for a model galaxy.
        """

        new_components = deepcopy(self.model_components)

        dirichlet_comps = []

        # Substitute values of fit params from param into model_comp.
        for i in range(len(self.params)):
            split = self.params[i].split(":")
            if len(split) == 1:
                new_components[self.params[i]] = param[i]

            elif len(split) == 2:
                if "dirichlet" in split[1]:
                    if split[0] not in dirichlet_comps:
                        dirichlet_comps.append(split[0])

                else:
                    new_components[split[0]][split[1]] = param[i]

        # Set any mirror params to the value of the relevant fit param.
        for key in list(self.mirror_pars):
            split_par = key.split(":")
            split_val = self.mirror_pars[key].split(":")
            fit_val = new_components[split_val[0]][split_val[1]]
            new_components[split_par[0]][split_par[1]] = fit_val

        # Deal with any Dirichlet distributed parameters.
        if len(dirichlet_comps) > 0:
            comp = dirichlet_comps[0]
            n_bins = 0
            for i in range(len(self.params)):
                split = self.params[i].split(":")
                if (split[0] == comp) and ("dirichlet" in split[1]):
                    n_bins += 1

            new_components[comp]["r"] = np.zeros(n_bins)

            j = 0
            for i in range(len(self.params)):
                split = self.params[i].split(":")
                if (split[0] == comp) and "dirichlet" in split[1]:
                    new_components[comp]["r"][j] = param[i]
                    j += 1

            tx = dirichlet(new_components[comp]["r"], new_components[comp]["alpha"])

            new_components[comp]["tx"] = tx

        # new_components["veldisp"] = 1100.

        return new_components

    def _sample(self, param_vector, cont_only=False, rm_line=None, **model_kwargs):

        new_comps = self._update_model_components(param_vector)
        # if "cont_only" not in model_kwargs:
        # model_kwargs["cont_only"] = model_kwargs.get("cont_only", False)
        # model_kwargs["rm_line"] = model_kwargs.get("rm_line", "H  1  6562.81A")

        if self.model_gal is None:
            self.model_gal = ExtendedModelGalaxy(new_comps, **model_kwargs)
        # else:
        self.model_gal.update(new_comps, cont_only=cont_only, rm_line=rm_line)

        return self.model_gal.spectrum.T


def _init_pipes_sampler(fit_instructions, veldisp, beams):
    global pipes_sampler
    # print (f"{core_id} initialised")
    pipes_sampler = BagpipesSampler(fit_instructions=fit_instructions, veldisp=veldisp)
    global beams_object
    beams_object = beams.copy()
    # global scaled_seg_maps
    # scaled_seg_maps = seg_maps.copy()


def _calc_template_from_pipes(
    seg_idx=None,
    seg_id=None,
    shared_seg_name=None,
    seg_maps_shape=None,
    shared_temps_name=None,
    temps_shape=None,
    # oversamp_seg_maps=None,
    # seg_ids=None,
    posterior_dir=None,
    iteration=None,
    n_samples=None,
    # oversamp_factor=None,
    spec_wavs=None,
    beam_info=None,
    temp_offset=None,
    # beams=None,
    # stacked_shape=None,
    cont_only=False,
    rm_line=None,
):
    # import sys
    # seg_id = seg_ids[use_idx]
    seg_id = int(seg_id)
    # print (seg_idx, seg_id)

    shm_seg_maps = shared_memory.SharedMemory(name=shared_seg_name)
    seg_maps = np.ndarray(seg_maps_shape, dtype=float, buffer=shm_seg_maps.buf)
    shm_temps = shared_memory.SharedMemory(name=shared_temps_name)
    temps_arr = np.ndarray(temps_shape, dtype=float, buffer=shm_temps.buf)
    # print (temps_arr.shape, stacked_shape)
    # print(seg_id, flush=True)
    # sys.stdout.flush()
    file = h5py.File(Path(posterior_dir) / f"{seg_id}.h5", "r")
    samples2d = np.array(file["samples2d"])

    # template_section = np.zeros((n_samples, temps_arr.shape[1]))

    for sample_i, sample_vec in enumerate(
        samples2d[iteration * n_samples : (iteration + 1) * n_samples]
    ):
        # print (seg_id, sample_vec)
        # print (seg_idx, seg_id, sample_i, flush=True)

        temp_resamp_1d = pipes_sampler._sample(
            sample_vec,
            spec_wavs=spec_wavs,
            cont_only=cont_only,
            rm_line=rm_line,
        )

        i0 = 0
        for k_i, (k, v) in enumerate(beam_info.items()):
            stack_idxs = np.r_["0,2", *v["flat_slice"]]
            for ib in v["list_idx"]:
                beam = beams_object[ib]
                # beam_seg = block_reduce(
                #     # oversamp_seg_maps[np.argmin() == int(seg_id),
                #     oversamp_factor,
                #     func=np.mean,
                # )
                beam_seg = seg_maps[seg_idx][ib]
                tmodel = beam.compute_model(
                    spectrum_1d=temp_resamp_1d,
                    thumb=beam.beam.direct * beam_seg,
                    # thumb=new_direct[ib]*beam_seg,
                    in_place=False,
                    is_cgs=True,
                )
                #     template_section[
                #         sample_i,
                #         i0 : i0 + np.prod(v["2d_shape"]),
                #     ] += tmodel
                # template_section[
                #     sample_i,
                #     i0 : i0 + np.prod(v["2d_shape"]),
                # ] /= len(v["list_idx"])
                temps_arr[
                    (n_samples * seg_idx) + sample_i + temp_offset,
                    i0 : i0 + np.prod(v["2d_shape"]),
                ] += tmodel
            temps_arr[
                (n_samples * seg_idx) + sample_i + temp_offset,
                i0 : i0 + np.prod(v["2d_shape"]),
            ] /= len(v["list_idx"])

            i0 += np.prod(v["2d_shape"])

    # print (np.nansum(template_section), flush=True)
    # temps_arr[seg_idx * n_samples : (seg_idx + 1) * n_samples] = template_section

    # return {"seg_id":seg_id, "template_section":template_section}
    # return (seg_id, template_section)


# def _format_results(result_tuple, shared_arr, n_samples, seg_ids):

#     # print (shared_arr, flush=True)

#     seg_id, template_section = result_tuple
#     s_i = np.argwhere(seg_ids == int(seg_id))[0][0]
#     # temp_offset + s_i * n_samples + sample_i

#     shared_arr[s_i * n_samples : (s_i + 1) * n_samples] = template_section
#     # shared_arr.flush()


def _print_id(test_var, dummy_var):

    print(test_var, dummy_var, flush=True)


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
    ):
        # print (f"{seg_idx=}, {seg_id=}")
        shm_init = shared_memory.SharedMemory(name=shared_input)
        init_arr = np.ndarray(init_shape, dtype=float, buffer=shm_init.buf)
        shm_output = shared_memory.SharedMemory(name=shared_output)
        output_arr = np.ndarray(output_shape, dtype=float, buffer=shm_output.buf)

        output_arr[seg_idx][beam_idx] = block_reduce(
            init_arr == seg_id,
            oversamp_factor,
            func=np.mean,
        )

    def fit_at_z(
        self,
        z=0.0,
        fit_background=True,
        poly_order=0,
        n_samples=3,
        num_iters=10,
        spec_wavs: ArrayLike | None = None,
        oversamp_factor=3,
        veldisp=500,
        fit_stacks=True,
        direct_images=None,
        temp_dir: PathLike | None = None,
        memmap: bool = False,
        out_dir: PathLike | None = None,
        cpu_count=-1,
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
        veldisp : int, optional
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

        import numpy.linalg
        import scipy.optimize

        if spec_wavs is None:
            spec_wavs = np.arange(9600.0, 23000.0, 45.0)

        # print 'xxx Init poly'
        self.init_poly_coeffs(poly_order=poly_order)

        # print 'xxx Init bg'
        if fit_background:
            self.fit_bg = True
            A = np.vstack((self.A_bg, self.A_poly))
        else:
            self.fit_bg = False
            A = self.A_poly * 1

        # print (self.regions_seg_ids)
        # exit()

        # print(A.shape, self.A_bg.shape, self.A_poly.shape)

        test_post = h5py.File(
            self.pipes_dir
            / "posterior"
            / self.run_name
            / f"{int(self.regions_phot_cat["bin_id"][0])}.h5",
            "r",
        )

        fit_info_str = test_post.attrs["fit_instructions"]
        fit_info_str = fit_info_str.replace("array", "np.array")
        fit_info_str = fit_info_str.replace("float", "np.float")
        fit_info_str = fit_info_str.replace("np.np.", "np.")
        fit_instructions = eval(fit_info_str)

        if out_dir is None:
            out_dir = Path.cwd()
        out_dir.mkdir(exist_ok=True, parents=True)
        if memmap:
            if temp_dir is None:
                temp_dir = Path.cwd()
            temp_dir.mkdir(exist_ok=True, parents=True)

            # oversamp_seg_maps = np.memmap(
            #     temp_dir / "oversamp_seg.dat",
            #     # dtype=int,
            #     mode="w+",
            #     shape=(
            #         self.n_regions,
            #         self.N,
            #         # oversamp_factor * self.beams[0].beam.sh[0],
            #         # oversamp_factor * self.beams[0].beam.sh[1],
            #         self.beams[0].beam.sh[0],
            #         self.beams[0].beam.sh[1],
            #     ),
            # )
        # else:
        with SharedMemoryManager() as smm:

            shm_seg_maps = smm.SharedMemory(
                size=np.dtype(float).itemsize
                * self.n_regions
                * self.N
                * self.beams[0].beam.sh[0]
                * self.beams[0].beam.sh[1]
            )
            oversamp_seg_maps = np.ndarray(
                (
                    self.n_regions,
                    self.N,
                    self.beams[0].beam.sh[0],
                    self.beams[0].beam.sh[1],
                ),
                dtype=float,
                buffer=shm_seg_maps.buf,
            )
            oversamp_seg_maps.fill(0.0)
            # shared_init_params[:] = cubes[:]
            # oversamp_seg_maps = np.zeros(
            #     (
            #         self.n_regions,
            #         self.N,
            #         # oversamp_factor * self.beams[0].beam.sh[0],
            #         # oversamp_factor * self.beams[0].beam.sh[1],
            #         self.beams[0].beam.sh[0],
            #         self.beams[0].beam.sh[1],
            #     ),
            #     # dtype=int,
            # )
            beam_info = {}
            start_idx = 0

            shm_oversampled = smm.SharedMemory(
                size=np.dtype(float).itemsize
                * oversamp_factor
                * self.beams[0].beam.sh[0]
                * oversamp_factor
                * self.beams[0].beam.sh[1],
            )
            oversampled = np.ndarray(
                (
                    oversamp_factor * self.beams[0].beam.sh[0],
                    oversamp_factor * self.beams[0].beam.sh[1],
                ),
                dtype=float,
                buffer=shm_oversampled.buf,
            )
            oversampled.fill(np.nan)

            start_idx = 0
            for i, (beam_cutout, cutout_shape) in enumerate(
                zip(self.beams, self.Nflat)
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

                # oversampled[:] =
                # print (oversampled.dtype)
                oversampled[:] = reproject_interp(
                    (self.regions_seg_map, self.regions_seg_wcs),
                    beam_wcs,
                    (
                        oversamp_factor * self.beams[0].beam.sh[0],
                        oversamp_factor * self.beams[0].beam.sh[1],
                    ),
                    return_footprint=False,
                    order=0,
                    # output_array=oversampled,
                )
                # print (test.dtype)
                with Pool(processes=cpu_count) as pool:
                    multi_fn = partial(
                        self._reduce_seg_map,
                        shared_input=shm_oversampled.name,
                        init_shape=oversampled.shape,
                        shared_output=shm_seg_maps.name,
                        output_shape=oversamp_seg_maps.shape,
                        beam_idx=i,
                        oversamp_factor=oversamp_factor,
                        # store_fn=store_fn,
                        # tqdm_list=shared_list,
                    )
                    pool.starmap(multi_fn, enumerate(self.regions_seg_ids))

            oversamp_seg_maps[~np.isfinite(oversamp_seg_maps)] = 0

            # exit()

            NTEMP = self.n_regions * n_samples
            num_stacks = len([*beam_info.keys()])
            stacked_shape = np.nansum(
                [np.prod(v["2d_shape"]) for v in beam_info.values()]
            )
            temp_offset = A.shape[0] - self.N + num_stacks
            # if memmap:
            #     stacked_A = np.memmap(
            #         temp_dir / "stacked_A.dat",
            #         dtype=float,
            #         mode="w+",
            #         shape=(temp_offset + NTEMP, stacked_shape),
            #     )
            # else:
            #     stacked_A = np.zeros(
            #         (temp_offset + NTEMP, stacked_shape),
            #         dtype=float,
            #     )

            shm_stacked_A = smm.SharedMemory(
                size=np.dtype(float).itemsize * (temp_offset + NTEMP) * stacked_shape
            )
            stacked_A = np.ndarray(
                (temp_offset + NTEMP, stacked_shape),
                dtype=float,
                buffer=shm_stacked_A.buf,
            )
            stacked_A.fill(0.0)

            # Offset for background
            if fit_background:
                off = 0.04
            else:
                off = 0.0
            stacked_scif = np.zeros(stacked_shape)
            # A = np.zeros((fit_background+3,stacked_shape))
            # stacked_A_temp = np.zeros((A_temp.shape[0],stacked_shape))
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
                _scif = self.scif[stack_idxs]
                # _scif[~self.fit_mask[stack_idxs]] = np.nan
                stacked_scif[start_idx : start_idx + np.prod(v["2d_shape"])] = (
                    np.nanmedian(
                        # (self.scif[stack_idxs]),
                        _scif,
                        axis=0,
                    )
                )
                stacked_weightf[start_idx : start_idx + np.prod(v["2d_shape"])] = (
                    np.nanmedian(
                        self.weightf[stack_idxs],
                        axis=0,
                    )
                )
                stacked_fit_mask[start_idx : start_idx + np.prod(v["2d_shape"])] = (
                    np.any(
                        self.fit_mask[stack_idxs],
                        axis=0,
                        # dtype=bool
                    )
                )
                # stacked_fit_mask[start_idx:start_idx+np.prod(v["2d_shape"])] = np.ones_like(stacked_scif[start_idx:start_idx+np.prod(v["2d_shape"])], dtype=bool)
                if fit_background:
                    stacked_A[k_i, start_idx : start_idx + np.prod(v["2d_shape"])] = 1.0
                    stacked_A[
                        num_stacks : num_stacks + self.A_poly.shape[0],
                        start_idx : start_idx + np.prod(v["2d_shape"]),
                    ] = np.nanmean(self.A_poly[:, stack_idxs], axis=1)
                    # np.nansum(test_arr[:,indexs], axis=1)
                    #  np.vstack((self.A_bg, self.A_poly))
                else:
                    stacked_A[
                        num_stacks : num_stacks + self.A_poly.shape[0],
                        start_idx : start_idx + np.prod(v["2d_shape"]),
                    ] = np.nanmean(self.A_poly[:, stack_idxs], axis=1)

                # stacked_A_temp[:,start_idx:start_idx+np.prod(v["2d_shape"])] = np.nanmean(
                #     A_temp[:,stack_idxs], axis=1
                # )
                stacked_ivarf[start_idx : start_idx + np.prod(v["2d_shape"])] = (
                    np.nanmedian(self.ivarf[stack_idxs], axis=0)
                )
                # stacked_fit_mask[start_idx:start_idx+np.prod(v["2d_shape"])] = np.nansum(
                #     multib.scif[np.r_['0,2', *v["flat_slice"]]], axis=0
                # )

                start_idx += np.prod(v["2d_shape"])

            # iteration = 0

            # print(stacked_weightf)

            coeffs_tracker = []
            chi2_tracker = []

            output_table = Table(
                names=["iteration", "chi2"]
                + [f"base_coeffs_{b}" for b in np.arange(temp_offset)]
                + [
                    f"{p[0]}_{p[1]}"
                    for p in product(
                        self.regions_phot_cat["bin_id"], np.arange(n_samples)
                    )
                ]
            )
            # print(output_table)
            # print(len(output_table.colnames))
            output_table_path = out_dir / (
                f"{self.id}_{len(np.unique(self.regions_phot_cat["bin_id"]))}"
                f"bins_{num_iters}iters_{n_samples}samples.csv"
            )
            print(output_table_path)
            output_table.write(output_table_path, overwrite=True, format="pandas.csv")

            iterations = np.arange(num_iters)
            for iteration in iterations:
                print(f"Iteration {iteration}")

                # result_callback = partial(
                #     _format_results,
                #     shared_arr=stacked_A[temp_offset:],
                #     # seg_ids=np.asarray(self.regions_phot_cat["bin_id"], dtype=int),
                #     seg_ids=self.regions_seg_ids,
                #     n_samples=n_samples,
                # )
                sub_fn = partial(
                    _calc_template_from_pipes,
                    # seg_ids = self.regions_seg_ids,
                    shared_seg_name=shm_seg_maps.name,
                    seg_maps_shape=oversamp_seg_maps.shape,
                    shared_temps_name=shm_stacked_A.name,
                    temps_shape=stacked_A.shape,
                    posterior_dir=str(self.pipes_dir / "posterior" / self.run_name),
                    iteration=iteration,
                    n_samples=n_samples,
                    spec_wavs=spec_wavs,
                    beam_info=beam_info,
                    temp_offset=temp_offset,
                    # beams=self.beams,
                    # stacked_shape=stacked_shape,
                    cont_only=False,
                    rm_line=None,
                )

                # result_callback = partial(_print_id, dummy_var="test")
                # init_fn = partial(_init_pipes_sampler, fit_instructions=fit_instructions, veldisp=veldisp)
                print("Generating models...")
                with multiprocessing.Pool(
                    processes=cpu_count,
                    # initializer=init_fn,
                    # initargs=np.arange(cpu_count),
                    # initargs=0,
                    initializer=_init_pipes_sampler,
                    initargs=(
                        fit_instructions,
                        veldisp,
                        self.beams,
                    ),  # oversamp_seg_maps),
                ) as pool:
                    # pool.map(sub_fn, self.regions_phot_cat["bin_id"])
                    # pool.map_async(sub_fn, self.regions_phot_cat["bin_id"], callback=result_callback, error_callback=print)

                    for s_i, s in enumerate(self.regions_seg_ids):
                        # print (s_i,s)
                        # print(pipes_sampler)
                        pool.apply_async(
                            sub_fn,
                            # (s, oversamp_seg_maps[s_i]),
                            (s_i, s),
                            # callback=result_callback,
                            error_callback=print,
                        )
                        # print (f"{s}_2")
                    pool.close()
                    # wait for all tasks to complete and processes to close
                    pool.join()

                # for s_i, seg_id in enumerate(self.regions_phot_cat["bin_id"]):
                #     print(s_i)
                #     # if s_i == 0:
                #     file = h5py.File(
                #         self.pipes_dir / "posterior" / self.run_name / f"{seg_id}.h5", "r"
                #     )
                #     samples2d = np.array(file["samples2d"])

                #     for sample_i, sample_vec in enumerate(
                #         samples2d[iteration * n_samples : (iteration + 1) * n_samples]
                #     ):
                #         # print (seg_id, sample_vec)

                #         temp_resamp_1d = pipes_sampler.sample(sample_vec, spec_wavs=spec_wavs)

                #         i0 = 0
                #         for k_i, (k, v) in enumerate(beam_info.items()):
                #             stack_idxs = np.r_["0,2", *v["flat_slice"]]
                #             for ib in v["list_idx"]:
                #                 beam = self.beams[ib]
                #                 beam_seg = block_reduce(
                #                     oversamp_seg_maps[ib] == int(seg_id),
                #                     oversamp_factor,
                #                     func=np.mean,
                #                 )
                #                 tmodel = beam.compute_model(
                #                     spectrum_1d=temp_resamp_1d,
                #                     thumb=beam.beam.direct * beam_seg,
                #                     # thumb=new_direct[ib]*beam_seg,
                #                     in_place=False,
                #                     is_cgs=True,
                #                 )
                #                 stacked_A[
                #                     temp_offset + s_i * n_samples + sample_i,
                #                     i0 : i0 + np.prod(v["2d_shape"]),
                #                 ] += tmodel
                #             stacked_A[
                #                 temp_offset + s_i * n_samples + sample_i,
                #                 i0 : i0 + np.prod(v["2d_shape"]),
                #             ] /= len(v["list_idx"])
                #             i0 += np.prod(v["2d_shape"])

                print("Generating models... DONE")
                ok_temp = np.sum(stacked_A, axis=1) > 0
                from time import time

                # print ("Done sum")
                # _, unique_idxs = np.unique(stacked_A, axis=0, return_index=True)
                # # print ("Done unique")
                # unique_temp = np.isin(np.arange(stacked_A.shape[0]), unique_idxs)
                # init_time = time()
                # unique_temp = ~duplicated_rows(stacked_A)
                # print (time()-init_time)
                # init_time = time()

                stacked_A_contig = np.ascontiguousarray(stacked_A[:, ::100])
                # np.unique() finds identical items in a raveled array. To make it
                # see each row as a single item, we create a view of each row as a
                # byte string of length itemsize times number of columns in `ar`
                ar_row_view = stacked_A_contig.view(
                    "|S%d" % (stacked_A_contig.itemsize * stacked_A_contig.shape[1])
                )
                _, unique_idxs = np.unique(ar_row_view, return_index=True)
                unique_temp = np.isin(np.arange(stacked_A.shape[0]), unique_idxs)
                # print (time()-init_time)
                # exit()

                # repeats = 5

                # init_time = time()
                # times_pandas = []
                # for i in range(repeats):
                #     duplicated_rows(stacked_A)
                #     times_pandas.append(time()-init_time)
                #     init_time = time()

                # init_time = time()
                # times_np = []
                # for i in range(repeats):
                #     numpy.unique(stacked_A, axis=0, return_index=True)
                #     times_np.append(time()-init_time)
                #     init_time = time()

                # # times_pandas = timeit.timeit(
                # #     "duplicated_rows(stacked_A)",
                # #     globals=globals(),
                # #     setup="from __main__ import stacked_A",
                # #     number=5,
                # # )
                # print(
                #     f"Pandas: {np.nanmedian(times_pandas):.3f} +/- {np.nanmedian(times_pandas):.3f}"
                # )
                # # times_np = timeit.timeit(
                # #     "numpy.unique(stacked_A, axis=0, return_index=True)",
                # #     setup="from __main__ import numpy, stacked_A",
                # #     number=5,
                # # )
                # print(
                #     f"Numpy: {np.nanmedian(times_np):.3f} +/- {np.nanmedian(times_np):.3f}"
                # )
                ok_temp &= unique_temp
                # print ("3")
                # print ("Unique,", np.nansum(unique_temp), stacked_A.shape)
                out_coeffs = np.zeros(stacked_A.shape[0])
                # print(ok_temp, ok_temp.shape, out_coeffs.shape)
                # print (stacked_fit_mask.shape,stacked_y.shape)
                # print ("Done others")
                stacked_fit_mask &= np.isfinite(stacked_scif)
                # print ("Done masking")

                # for row in stacked_A_temp:
                #     print (np.sum(row)>0)

                stacked_Ax = stacked_A[:, stacked_fit_mask][ok_temp, :].T
                # print ("Transposed")
                # print (stacked_A[:, stacked_fit_mask].shape)
                # print (stacked_A[:, stacked_fit_mask][ok_temp,:].shape)
                # print (stacked_Ax.shape, stacked_y.shape)
                print(stacked_Ax.shape)
                # Weight by ivar
                stacked_Ax *= np.sqrt(stacked_ivarf[stacked_fit_mask][:, np.newaxis])

                print("NNLS fitting...")
                # exit()
                # print (stacked_Ax.shape, stacked_y.shape)

                if fit_background:
                    y = stacked_scif[stacked_fit_mask] + off
                    y *= np.sqrt(stacked_ivarf[stacked_fit_mask])
                    # # y_fit = stacked_y[stacked_fit_mask] + off
                    # # y_fit = stacked_y[stacked_fit_mask] + off
                    # coeffs, rnorm = scipy.optimize.nnls(
                    #     stacked_Ax,
                    #     y + off,
                    # )  # , maxiter=1e4, atol=1e2)
                    # print (stacked_Ax.shape)
                    # exit()
                    nnls_solver = CDNNLS(stacked_Ax, y + off)
                    # nnls_solver._run(stacked_Ax.shape[1]*3)
                    nnls_solver.run(n_iter=100)
                    coeffs = nnls_solver.w
                    # coeffs, rnorm = scipy.optimize.nnls(
                    #     stacked_Ax,
                    #     y + off,
                    # )  # , maxiter=1e4, atol=1e2)
                    # _lsq_res = scipy.optimize.lsq_linear(stacked_Ax, y + off, bounds=(0, np.inf), method='bvls', verbose=1)
                    # coeffs = _lsq_res.x
                    # print (f"Status: {_lsq_res.status}")
                    # coeffs = fe.fnnls(numpy.ascontiguousarray(stacked_Ax), numpy.ascontiguousarray(y + off))
                    # coeffs = solve_nnls(stacked_Ax, y + off, gtol=1e-10, max_iter=stacked_Ax.shape[1]*3)[0]
                    # tnt_result = tntnn(stacked_Ax, y+off, verbose=True)
                    # coeffs = tnt_result.x
                    coeffs[:num_stacks] -= off
                else:
                    y = self.scif[self.fit_mask]
                    y *= np.sqrt(self.ivarf[self.fit_mask])

                    coeffs, rnorm = scipy.optimize.nnls(Ax, y)

                    Ax = A[:, self.fit_mask][ok_temp, :].T
                    # Weight by ivar
                    Ax *= np.sqrt(self.ivarf[self.fit_mask][:, np.newaxis])

                print("NNLS fitting... DONE")
                out_coeffs[ok_temp] = coeffs
                stacked_modelf = np.dot(out_coeffs, stacked_A)
                # chi2 = np.sum((self.weightf*(self.scif - modelf)**2*stacked_ivarf)[self.fit_mask])

                # print(stacked_modelf)
                # print(stacked_modelf.shape)
                # print (coeffs)

                chi2 = np.nansum(
                    (
                        stacked_weightf
                        * (stacked_scif - stacked_modelf) ** 2
                        * stacked_ivarf
                    )[stacked_fit_mask]
                )

                # coeffs_tracker.append(out_coeffs)
                # chi2_tracker.append(chi2)#

                output_table.add_row([iteration, chi2, *out_coeffs])
                output_table.write(output_table_path, overwrite=True)
                print(f"Iteration {iteration}: chi2={chi2:.3f}\n")
                stacked_A[temp_offset:].fill(0.0)

            del stacked_Ax

            best_iter = np.argmin(output_table["chi2"])

            # result_callback = partial(
            #     _format_results,
            #     shared_arr=stacked_A[temp_offset:],
            #     # seg_ids=np.asarray(self.regions_phot_cat["bin_id"], dtype=int),
            #     seg_ids=self.regions_seg_ids,
            #     n_samples=n_samples,
            # )

            sub_fn = partial(
                _calc_template_from_pipes,
                # seg_ids = self.regions_seg_ids,
                shared_seg_name=shm_seg_maps.name,
                seg_maps_shape=oversamp_seg_maps.shape,
                shared_temps_name=shm_stacked_A.name,
                temps_shape=stacked_A.shape,
                posterior_dir=str(self.pipes_dir / "posterior" / self.run_name),
                iteration=best_iter,
                n_samples=n_samples,
                spec_wavs=spec_wavs,
                beam_info=beam_info,
                temp_offset=temp_offset,
                # beams=self.beams,
                # stacked_shape=stacked_shape,
                cont_only=False,
                # rm_line=[
                #     "O  3  5006.84A",
                #     "O  3  4958.91A"
                # ]
                # rm_line=[
                #     "Blnd  3726.00A",
                #     "Blnd  3729.00A",
                # ],
                # rm_line = "H  1  6562.81A"
                rm_line=[
                    "H  1  6562.81A",
                    "N  2  6583.45A",
                    "N  2  6548.05A",
                ],
                # rm_line="H  1  4861.33A",
            )

            # result_callback = partial(_print_id, dummy_var="test")
            # init_fn = partial(_init_pipes_sampler, fit_instructions=fit_instructions, veldisp=veldisp)
            with multiprocessing.Pool(
                processes=cpu_count,
                # initializer=init_fn,
                # initargs=np.arange(cpu_count),
                # initargs=0,
                initializer=_init_pipes_sampler,
                initargs=(fit_instructions, veldisp, self.beams),
            ) as pool:
                # print("Initialised")
                # pool.map(sub_fn, self.regions_phot_cat["bin_id"])
                # pool.map_async(sub_fn, self.regions_phot_cat["bin_id"], callback=result_callback, error_callback=print)
                for s_i, s in enumerate(self.regions_phot_cat["bin_id"]):
                    # print (s)
                    # print(pipes_sampler)
                    pool.apply_async(
                        sub_fn,
                        # (s, oversamp_seg_maps[s_i]),
                        (s_i, s),
                        # callback=result_callback,
                        error_callback=print,
                    )
                    # print (f"{s}_2")
                pool.close()
                # wait for all tasks to complete and processes to close
                pool.join()

        out_coeffs = np.array(output_table[best_iter][2:])
        stacked_modelf = np.dot(out_coeffs, stacked_A)
        # print(chi2_tracker)
        print(best_iter, output_table[best_iter])

        # axs = axs.flatten()
        # filt_plot = "F150W-GR150R"
        for pupil in ["GR150C", "GR150R"]:
            fig, axs = plt.subplots(
                3, 3, sharex=True, sharey=True, dpi=600, figsize=(16, 10)
            )
            for f_i, filt in enumerate(["F115W", "F150W", "F200W"]):
                filt_plot = f"{filt}-{pupil}"
                start_idx = 0
                for i, k in enumerate(beam_info.keys()):

                    if k != filt_plot:
                        start_idx += np.prod(beam_info[k]["2d_shape"])
                    else:
                        import astropy.visualization as astrovis

                        init = start_idx

                        slice_plot = slice(
                            init, init + np.prod(beam_info[filt_plot]["2d_shape"])
                        )

                        wht_i = stacked_weightf[slice_plot]
                        clip = (wht_i > 0) & np.isfinite(wht_i > 0)
                        if clip.sum() == 0:
                            clip = np.isfinite(wht_i)

                        avg_rms = 1 / np.nanmedian(
                            np.sqrt(stacked_ivarf[slice_plot][clip])
                        )
                        print("rms", avg_rms)
                        print(np.nanmedian(np.sqrt(wht_i[clip])))
                        print(np.nanmedian(wht_i[clip]))
                        print(np.nanmedian(stacked_ivarf[slice_plot][clip]))
                        print(np.nanmedian(1 / stacked_ivarf[slice_plot][clip]))

                        data_list = [
                            stacked_scif[slice_plot],
                            stacked_modelf[slice_plot],
                            # stacked_weightf[slice_plot],
                            stacked_scif[slice_plot] - stacked_modelf[slice_plot],
                        ]

                        for w, data_w in enumerate(data_list):

                            # print ([1.1 * np.nanpercentile(data_w[clip], 98), 5 * avg_rms])
                            vmax = np.nanmax(
                                [1.1 * np.nanpercentile(data_w[clip], 98), 5 * avg_rms]
                            )
                            vmin = -0.1 * vmax
                            plot_kwargs = {
                                "origin": "lower",
                                "norm": astrovis.ImageNormalize(
                                    data_w,
                                    stretch=astrovis.SqrtStretch(),
                                    interval=astrovis.ManualInterval(
                                        vmin=vmin, vmax=vmax
                                    ),
                                    # interval=astrovis.PercentileInterval(99.9)
                                ),
                                "cmap": "plasma",
                            }

                            axs[f_i, w].imshow(
                                data_w.reshape(beam_info[filt_plot]["2d_shape"]),
                                **plot_kwargs,
                            )
                        # axs[f_i, 0].imshow(
                        #     stacked_modelf[slice_plot].reshape(
                        #         beam_info[filt_plot]["2d_shape"]
                        #     ),
                        #     **plot_kwargs,
                        # )
                        # axs[f_i, 2].imshow(
                        #     (
                        #         stacked_scif[slice_plot] - stacked_modelf[slice_plot]
                        #     ).reshape(beam_info[filt_plot]["2d_shape"]),
                        #     **plot_kwargs,
                        # )
            plt.show()

        chi2 = np.nansum(
            (stacked_weightf * (stacked_scif - stacked_modelf) ** 2 * stacked_ivarf)[
                stacked_fit_mask
            ]
        )

        if fit_background:
            poly_coeffs = out_coeffs[num_stacks : num_stacks + self.n_poly]
        else:
            poly_coeffs = out_coeffs[: self.n_poly]

        print(self.n_poly, self.N)
        print(num_stacks)
        print(f"{out_coeffs=}")
        print(poly_coeffs, self.x_poly)
        self.y_poly = np.dot(poly_coeffs, self.x_poly)
        # x_poly = self.x_poly[1,:]+1 = self.beams[0].beam.lam/1.e4

        return A, out_coeffs, chi2, stacked_modelf
