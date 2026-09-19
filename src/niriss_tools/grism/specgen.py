"""
Functions and classes for generating specific types of model spectra.
"""

import warnings
from copy import deepcopy
from functools import partial
from multiprocessing import Manager, Pool
from pathlib import Path

import bagpipes
import h5py
import numpy as np
from bagpipes import config, filters, utils
from bagpipes.input.spectral_indices import measure_index
from bagpipes.models import chemical_enrichment_history
from bagpipes.models import model_galaxy as BagpipesModelGalaxy
from bagpipes.models import star_formation_history as BagpipesSFH
from bagpipes.models.agn_model import agn
from bagpipes.models.dust_attenuation_model import dust_attenuation
from bagpipes.models.dust_emission_model import dust_emission
from bagpipes.models.igm_model import igm
from bagpipes.models.nebular_model import nebular
from bagpipes.models.stellar_model import stellar
from grizli.utils_numba.interp import interp_conserve_c
from numpy.typing import ArrayLike
from tqdm import tqdm

__all__ = [
    "ExtendedModelGalaxy",
    "CLOUDY_LINE_MAP",
    "check_coverage",
    "NIRISS_050_FILTER_LIMITS",
    "NIRISS_001_FILTER_LIMITS",
    "BagpipesSpecGenerator",
    "air_to_vac",
]

CLOUDY_LINE_MAP = [
    # Paschen series
    {
        "cloudy": [
            "H  1  1.87510m",
        ],
        "grizli": "PaA",
        "wave": 18756.3,
    },
    {
        "cloudy": [
            "H  1  1.28181m",
        ],
        "grizli": "PaB",
        "wave": 12821.7,
    },
    {
        "cloudy": [
            "H  1  1.09381m",
        ],
        "grizli": "PaG",
        "wave": 10941.2,
    },
    {
        "cloudy": [
            "H  1  1.00494m",
        ],
        "grizli": "PaD",
        "wave": 10052.2,
    },
    # Balmer Series
    {
        "cloudy": [
            "H  1  6562.80A",
            "N  2  6583.45A",
            "N  2  6548.05A",
        ],
        "grizli": "Ha",
        "wave": 6564.697,
    },
    {
        "cloudy": ["H  1  4861.32A"],
        "grizli": "Hb",
        "wave": 4862.738,
    },
    # {
    #     "cloudy": ["H  1  4340.46A"],
    #     "grizli": "Hg",
    #     "wave": 4341.731,
    # },
    # {
    #     "cloudy": ["H  1  4101.73A"],
    #     "grizli": "Hd",
    #     "wave": 4102.936,
    # },
    # # Oxygen
    {
        "cloudy": [
            "O  3  5006.84A",
        ],
        "grizli": "OIII-5007",
        "wave": 5008.240,
    },
    {
        "cloudy": [
            "O  3  4958.91A",
        ],
        "grizli": "OIII-4959",
        "wave": 4960.295,
    },
    {
        "cloudy": ["O  3  4363.21A"],
        "grizli": "OIII-4363",
        "wave": 4364.436,
    },
    {
        "cloudy": [
            "O  2  3726.03A",
            "O  2  3728.81A",
        ],
        "grizli": "OII",
        "wave": 3728.48,
    },
    # Sulphur
    {
        "cloudy": [
            "S  3  9530.62A",
        ],
        "grizli": "SIII-9530",
        "wave": 9530.62,
    },
    {
        "cloudy": [
            "S  3  9068.62A",
        ],
        "grizli": "SIII-9068",
        "wave": 9068.62,
    },
    {
        "cloudy": [
            "S  2  6730.82A",
            "S  2  6716.44A",
        ],
        "grizli": "SII",
        "wave": 6725.48,
    },
    {
        "cloudy": [
            "S  3  6312.06A",
        ],
        "grizli": "SIII-6314",
        "wave": 6313.81,
    },
    # Helium
    {
        "cloudy": [
            "Blnd  1.08302m",
        ],
        "grizli": "HeI-1083",
        "wave": 10830.3,
    },
    {
        "cloudy": [
            "Blnd  5875.66A",
        ],
        "grizli": "HeI-5877",
        "wave": 5877.249,
    },
    {
        "cloudy": [
            "He 1  3888.64A",
        ],
        "grizli": "HeI-3889",
        "wave": 3889.75,
    },
    {
        "cloudy": [
            "Ne 3  3868.76A",
        ],
        "grizli": "NeIII-3867",
        "wave": 3869.87,
    },
]

# In Angstroms
NIRISS_050_FILTER_LIMITS = {
    # "F090W": [7960, 10050],
    "F115W": [10130, 12830],
    "F150W": [13300, 16710],
    "F200W": [17510, 22260],
}

NIRISS_001_FILTER_LIMITS = {
    # "F090W": [7960, 10050],
    "F115W": [10010, 12930],
    "F150W": [13200, 16810],
    "F200W": [17380, 22420],
}


def check_coverage(
    obs_wavelength: float, filter_limits: dict = NIRISS_001_FILTER_LIMITS
):
    """
    Check if a line is covered by the NIRISS filters.

    Parameters
    ----------
    obs_wavelength : float
        The observed wavelength of the line.
    filter_limits : dict, optional
        A dictionary, where the keys are the names of the grism filters,
        and the values are array-like, containing
        ``[min_wavelength, max_wavelength]``. Defaults to
        ``NIRISS_FILTER_LIMITS``.

    Returns
    -------
    bool
        ``True`` if the line falls within the filter coverage, else
        ``False``.
    """

    obs_wavelength = np.atleast_1d(obs_wavelength)
    covered = np.zeros((len(filter_limits), *obs_wavelength.shape), dtype=bool)
    for i, (k, v) in enumerate(filter_limits.items()):
        covered[i] = (obs_wavelength >= v[0]) & (obs_wavelength <= v[-1])
    return np.bitwise_or.reduce(covered, axis=0)


def air_to_vac(wavelength: ArrayLike) -> ArrayLike:
    """
    Convert air to vacuum wavelengths.

    Implements the air to vacuum wavelength conversion described in eqn 65 of
    Griesen 2006.

    TODO: check against most recent specutils conversions.

    Parameters
    ----------
    wavelength : ArrayLike
        The wavelengths in Angstroms.
    """

    sigma2 = (1 / (wavelength / 1e4)) ** 2

    refr = 1 + 1e-6 * (287.6155 + 1.62887 * sigma2 + 0.01360 * sigma2**2)

    # Only convert above 2000A
    wavelength[wavelength > 2e3] *= refr[wavelength > 2e3]

    return wavelength


def create_spec_file(
    post_id: str,
    posterior_dir: Path | None = None,
    spec_dir: Path | None = None,
    spec_wavs: ArrayLike | None = None,
) -> None:
    """
    Generated resampled spectra from a bagpipes posterior output.

    Parameters
    ----------
    post_id : str
        The ID of the posterior ``"*.h5"`` file.
    posterior_dir : Path | None, optional
        The directory containing the posterior file, by default ``None``.
    spec_dir : Path | None, optional
        The directory containing the spectral file, by default ``None``.
    spec_wavs : ArrayLike | None, optional
        The wavelengths onto which the spectrum will be resampled, by
        default ``None``.
    """

    with h5py.File(posterior_dir / f"{post_id}.h5", "r") as post_file:
        samples2d = np.array(post_file["samples2d"])

    with h5py.File(spec_dir / f"{post_id}.h5", "a") as spec_file:

        if not spec_file.get("spec_data"):

            if not spec_file.get("spec_wavs"):

                spec_file.create_dataset("spec_wavs", data=spec_wavs)

            unique_vectors, unique_inv = np.unique(
                samples2d, axis=0, return_inverse=True
            )
            spec_data = np.zeros((unique_vectors.shape[0], spec_wavs.shape[0]))

            for s_i, param_vector in enumerate(unique_vectors):
                spec_data[s_i] = spec_sampler.sample(
                    param_vector,
                    spec_wavs=spec_wavs,
                )[1]

            spec_file.create_dataset("spec_data", data=spec_data[unique_inv])


def pre_gen_spec(
    pipes_dir: Path,
    fit_instructions: dict,
    spec_wavs: ArrayLike,
    veldisp: float = 500,
    run: str = "",
    cpu_count: int | None = None,
) -> None:
    """
    Generate resampled spectra for all posterior samples.

    Parameters
    ----------
    pipes_dir : Path
        The main bagpipes directory.
    fit_instructions : dict
        A dictionary containing the details of the model, as well as any
        constraints and priors on the parameters.
    spec_wavs : `ArrayLike`
        An array of wavelengths at which spectral fluxes should be
        returned.
    veldisp : float, optional
        The velocity dispersion of the model galaxy in km/s. By default
        ``veldisp=500``.
    run : str, optional
        The subfolder into which outputs will be saved, useful e.g.
        for fitting more than one model configuration to the same
        data, by default ``""``.
    cpu_count : int, optional
        The number of processes to use when generating the spectra. If
        ``None`` (default), this will run on the number of
        cores returned by  `multiprocessing.cpu_count`.
    """

    posterior_dir = pipes_dir / "posterior" / run

    spec_dir = pipes_dir / "spec" / run
    spec_dir.mkdir(exist_ok=True, parents=True)

    post_ids = [f.stem for f in posterior_dir.glob("*.h5")]

    if not len(post_ids) > 0:
        return

    # Generate the spectra
    print("Generating resampled spectra...")
    pbar = tqdm(total=len(post_ids))

    def _update(*a):
        pbar.update()

    with Pool(
        processes=cpu_count,
        initializer=init_bagpipes_spec_gen,
        initargs=(
            fit_instructions,
            veldisp,
        ),
    ) as pool:
        for p_i, p in enumerate(post_ids):
            pool.apply_async(
                create_spec_file,
                (p,),
                kwds=dict(
                    posterior_dir=posterior_dir,
                    spec_dir=spec_dir,
                    spec_wavs=spec_wavs,
                ),
                error_callback=print,
                callback=_update,
            )
        pool.close()
        pool.join()
        pbar.close()
