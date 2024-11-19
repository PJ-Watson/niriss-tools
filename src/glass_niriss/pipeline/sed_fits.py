"""
Functions and classes for fitting SEDs.
"""

from os import PathLike

import numpy as np
from astropy.table import Table
from numpy.typing import ArrayLike

__all__ = ["load_photom_bagpipes", "generate_fit_params"]


def load_photom_bagpipes(
    str_id: str,
    phot_cat: Table | PathLike,
    id_colname: str = "bin_id",
    zeropoint: float = 28.9,
    cat_hdu_index: int | str = 0,
):
    """
    Load photometry from a catalogue to bagpipes-formatted data.

    Parameters
    ----------
    str_id : str
        The ID of the object to fit.
    phot_cat : os.PathLike
        The location of the photometric catalogue.
    id_colname : str, optional
        The name of the column containing ``str_id``, by default ``"bin_id"``.

    Returns
    -------
    array-like
        An Nx2 array containing fluxes and errors in all photometric bands.
    """

    if not isinstance(phot_cat, Table):
        phot_cat = Table.read(phot_cat, hdu=cat_hdu_index)

    row_idx = (phot_cat[id_colname] == str_id).nonzero()[0][0]

    fluxes = []
    errs = []
    for c in phot_cat.colnames:
        if "_sci" in c.lower():
            fluxes.append(phot_cat[c][row_idx])
        elif "_var" in c.lower():
            errs.append(np.sqrt(phot_cat[c][row_idx]))
        elif "_err" in c.lower():
            errs.append(phot_cat[c][row_idx])
    flux = np.asarray(fluxes) * 1e-2
    flux_err = np.asarray(errs) * 1e-2
    flux_err = np.sqrt(flux_err**2 + (0.1 * flux) ** 2)
    flux = flux.copy()
    flux_err = flux_err.copy()
    bad_values = (
        ~np.isfinite(flux) | (flux <= 0) | ~np.isfinite(flux_err) | (flux_err <= 0)
    )
    flux[bad_values] = 0.0
    flux_err[bad_values] = 1e30
    return np.c_[flux, flux_err]


def bin_and_fit(
    convolved_image: ArrayLike | PathLike,
    atlas: PathLike,
):

    data_dir = Path("/media/sharedData/data/")

    filter_dir = data_dir / "2024_02_14_RPS/PSF_matching/bagpipes/filters_sedpy/"

    db_dir = data_dir / "2024_02_14_RPS/PSF_matching/bagpipes_atlas"
    atlas_dir = db_dir / "pregrids/"
    atlas_dir.mkdir(exist_ok=True, parents=True)

    filter_arr = np.load(filter_dir.parent / "filter_paths_arr.npy")
    # print(filter_arr)
    filter_list = []
    for i, f in enumerate(filter_arr):
        # print (f)
        filter_list.append(str(filter_dir.parent / f))
    # print(filter_list)
    # eff_wavs = np.zeros_like(filter_arr, dtype="float")
    # for i, f in enumerate(filter_arr):
    #     data = np.loadtxt(filter_dir / f.split("/")[-1])
    #     # print (data)
    #     # plt.plot(data[:,0],data[:,1])
    #     # print (np.nanmedian())
    #     dlambda = make_bins(data[:, 0])[1]
    #     filt_weights = dlambda * data[:, 1]
    #     # print (len(dlam), len(data[:,0]))
    #     eff_wavs[i] = np.sqrt(
    #         np.sum(filt_weights * data[:, 0]) / np.sum(filt_weights / data[:, 0])
    #     )
    #     # plt.scatter(eff_wavs[i], 1)
    #     # print (dlam)
    # # plt.show()
    # # exit()
    # # print(dir(bagpipes.model_galaxy))

    fit_instr = np.load(
        data_dir
        / "2024_02_14_RPS/PSF_matching/bagpipes/"
        / f"fit_info_F0083_leja_100_-200_200_-100_cvt_wvt_dust_corr_7_bins.npy",
        allow_pickle=True,
    ).item()
    # del fit_instr["metallicity"]
    # del fit_instr["metallicity_prior_mu"]
    # del fit_instr["metallicity_prior_sigma"]
    # fit_instr["metallicity"] = (0.,3.)
    # fit_instr["metallicity_prior"] = "Gaussian"
    # fit_instr["metallicity_prior_mu"] = 1.
    fit_instr["continuity"]["metallicity_prior"] = "Gaussian"
    # fit_instr["continuity"]["massformed"] = (3,11)
    # fit_instr["dust"]["eta"] = 2.
    print(fit_instr)

    n_samples = 3e6
    n_cores = 16

    remake_atlas = False

    if not (atlas_dir / f"atlas_{n_samples:.2E}.hdf5").is_file() or remake_atlas:

        atlas_gen = AtlasGenerator(
            # n_samples=100,
            fit_instructions=fit_instr,
            filt_list=filter_list,
            phot_units="ergscma",
        )

        atlas_gen.gen_samples(n_samples=n_samples, parallel=n_cores)

        atlas_gen.write_samples(filepath=atlas_dir / f"atlas_{n_samples:.2E}.hdf5")

        with h5py.File(atlas_dir / f"atlas_{n_samples:.2E}.hdf5", "r") as file:
            print(file)
            print([*file.keys()])

    # fig, axs = plt.subplots(5,3)
    # axs = axs.flatten()
    # n_samples = 1000
    # samples = np.zeros((n_samples,atlas_gen.ndim))
    # for i in np.arange(n_samples):
    #     samples[i] = atlas_gen.prior.transform(np.random.rand(atlas_gen.ndim))
    # for i in np.arange(atlas_gen.ndim):
    #     axs[i].hist(samples[:,i])
    # plt.show()
    # print (atlas_gen.prior.transform(cube))
    # print (atlas_gen.model_atlas.shape)
    # good_idx = np.nanmedian(atlas_gen.model_atlas, axis=1) > 2
    # plt.imshow(np.log10(atlas_gen.model_atlas[good_idx]), aspect="auto")
    # plt.show()

    # from misctools import SED_tools, generic_plotting
    obs_cat_path = Path(
        "/media/sharedData/data/2024_02_14_RPS/PSF_matching/"
        "voronoi/F0083_vorbin_cat_100_-200_200_-100_cvt_wvt.fits"
    )
    bin_size = "0"
    seg_map = np.load(
        "/media/sharedData/data/2024_02_14_RPS/PSF_matching/"
        "bagpipes/pipes/derived"
        "/bin_assignment_F0083_leja_100_-200_200_-100_cvt_wvt_dust_corr.npy"
    )

    # bin_size = 3
    # obs_cat_path = Path(
    #     "/media/sharedData/data/2024_02_14_RPS/PSF_matching/"
    #     f"voronoi/F0083_cat_100_-200_200_-100_block_{bin_size}_f480m_matched_dust_corr.fits"
    # )
    # seg_map = np.load(
    #     "/media/sharedData/data/2024_02_14_RPS/PSF_matching/"
    #     "bagpipes/pipes/derived"
    #     f"/bin_assignment_F0083_cat_100_-200_200_-100_block_{bin_size}_f480m_matched.npy"
    # )

    # obj_id = 1040
    load_fn = partial(load_photom_bagpipes, cat_path=obs_cat_path)

    # galaxy = bagpipes.galaxy(
    #     # 1152,
    #     # 969,
    #     # obj_id,
    #     # 14604,
    #     1040,
    #     # 1273,
    #     # 1223,
    #     # 1046,
    #     load_fn,
    #     spectrum_exists=False,
    #     filt_list=filter_list,
    # )

    # fig = galaxy.plot()
    # exit()
    fit = AtlasFitter(
        # galaxy=galaxy,
        fit_instructions=fit_instr,
        atlas_path=atlas_dir / f"atlas_{n_samples:.2E}.hdf5",
        out_path=db_dir,
        # run=f"{gal_id}_leja{append}",
    )
    # print("Atlas loaded.")
    # print(fit.fit_single(galaxy=galaxy))
    # # print (fit.cat)

    # exit()

    # plt.imshow(seg_map)
    # plt.show()

    plot_map = np.zeros_like(seg_map, dtype=float)

    obs_table = Table.read(obs_cat_path)
    # print(obs_table)

    cat_IDs = np.arange(len(obs_table))[:]
    redshifts = [0.3033] * len(cat_IDs)

    catalogue_out_path = fit.out_path / Path(
        f"F0083_block_{bin_size}_{n_samples:.2E}.fits"
    )
    if not catalogue_out_path.is_file():

        fit.fit_catalogue(
            IDs=cat_IDs,
            # fit_instructions = fit,
            load_data=load_fn,
            spectrum_exists=False,
            make_plots=False,
            cat_filt_list=filter_list,
            redshifts=redshifts,
            run=f"F0083_block_{bin_size}_{n_samples:.2E}",
            # run=f"{gal_id}_leja{append}_block_5",
            # full_catalogue=True,
            parallel=1,
        )
        print(fit.cat)
        fit.cat.write(catalogue_out_path)
    else:
        fit.cat = Table.read(catalogue_out_path)

    # new_table = Table(
    #     names=fit.params
    # )
    # print (new_table)
    print(fit.cat.colnames)
    print(fit.cat.dtype)


def generate_fit_params(
    obj_z: float | ArrayLike,
    z_range: float = 0.01,
    num_age_bins: int = 5,
    min_age_bin: float = 30,
):

    fit_params = {}
    if (z_range == 0.0) or (type(obj_z) is ArrayLike):
        fit_params["redshift"] = obj_z
    else:
        fit_params["redshift"] = (obj_z - z_range / 2, obj_z + z_range / 2)

    from astropy.cosmology import FlatLambdaCDM

    # Set up necessary variables for cosmological calculations.
    cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3)
    age_at_z = cosmo.age(np.nanmax(fit_params["redshift"])).value

    age_bins = np.geomspace(min_age_bin, age_at_z * 1e3, num=num_age_bins)

    age_bins = np.insert(age_bins, 0, 0.0)

    continuity = {
        "massformed": (3.0, 11.0),
        "metallicity": (0.0, 3.0),
        "metallicity_prior_mu": 1.0,
        "metallicity_prior_sigma": 0.5,
        "bin_edges": age_bins.tolist(),
    }

    for i in range(1, len(continuity["bin_edges"]) - 1):
        continuity["dsfr" + str(i)] = (-10.0, 10.0)
        continuity["dsfr" + str(i) + "_prior"] = "student_t"
        continuity["dsfr" + str(i) + "_prior_scale"] = (
            0.5  # Defaults to 0.3 (Leja19), we aim for a broader sample
        )
        continuity["dsfr" + str(i) + "_prior_df"] = (
            2  # Defaults to this value as in Leja19, but can be set
        )

    fit_params["continuity"] = continuity

    fit_params["dust"] = {
        "type": "Cardelli",
        "Av": (0.0, 2.0),
        "eta": 2.0,
    }
    fit_params["nebular"] = {"logU": (-3.5, -2.0)}
    fit_params["t_bc"] = 0.02

    return fit_params
