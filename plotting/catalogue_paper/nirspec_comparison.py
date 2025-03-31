"""
Compare NIRISS to NIRSpec.
"""

import plot_utils
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.nddata import block_reduce
from default_imports import *
from spectres import spectres

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    fig, axs = plt.subplots(
        1,
        1,
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        # hspace=0.,
        # wspace=0.,
    )
    # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    nirspec_dir = niriss_dir / "match_catalogues" / "nirspec_data"

    import msaexp
    from msaexp.resample_numba import resample_template_numba

    disp_dir = Path(msaexp.__file__).parent / "data"
    from astropy import constants
    from ppxf.ppxf_util import varsmooth

    # print (disp_dir)
    niriss_res = constants.c.to(u.km / u.s) / 150
    # print (niriss_res)
    # exit()
    obj_z_nirspec = 0.307
    for resolution, colour, offset, dispersion, label in zip(
        [0.001, 0.005],
        ["C0", "C2"],
        [-0.5e-18, -1.5e-18],
        [0, 1000],
        ["NIRSpec", "NIRSpec (R=150)"],
    ):
        wave_resampled = np.arange(0.805, 5.205, resolution) * u.um
        # obj_z = 0.0

        for i, f in enumerate(nirspec_dir.glob("**/*.fits")):
            # print (f)
            # with fits.open(f) as hdul:
            # hdul.info()
            dat = QTable.read(f, "SPEC1D")
            grating = dat.meta["GRATING"].lower()
            # print ((disp_dir / f"jwst_nirspec_{grating}_disp.fits").is_file())
            disp_tab = Table.read(disp_dir / f"jwst_nirspec_{grating}_disp.fits")
            print(disp_tab)
            eff_R = np.interp(dat["wave"], disp_tab["WAVELENGTH"], disp_tab["R"])
            eff_disp = constants.c.to(u.km / u.s) / eff_R
            print(eff_disp)
            print(
                np.sqrt(niriss_res.value**2 - eff_disp.value**2),
            )
            print(
                constants.c.value
                * 1e-6
                * dat["wave"].value
                / np.sqrt(niriss_res.value**2 - eff_disp.value**2)
            )

            # print (dat["wave.value"])
            # axs.plot(dat["wave"].value, smoothed_flux)
            # axs.plot(dat["wave"], dat["flux"])
            # axs.plot(wave_resampled, smoothed_flux)
            # plt.show()

            # resample_template_numba(wave_resampled, 140, dat["wave"],disp)

            # print (dat)
            kern_size = 5
            # wave = block_reduce(dat["wave"], kern_size, np.mean)
            # flux_jansky = block_reduce(dat["flux"], kern_size)
            # err = np.sqrt(block_reduce(dat["err"]**2, kern_size))
            mask = dat["err"] == 0

            dat["flux"] *= 2.1
            dat["err"] *= 2.1

            if dispersion > 0:
                xi = np.arange(len(dat["flux"]))
                dat["flux"][mask] = np.nanmedian(dat["flux"][~mask])
                # dat["flux"]  = np.interp(xi, xi[~mask], dat["flux"][~mask])
                # dat["err"]  = np.interp(xi, xi[~mask], dat["err"][~mask])
                dat["flux"] = (
                    varsmooth(
                        dat["wave"].value,
                        dat["flux"].value,
                        constants.c.to(u.km / u.s).value
                        * dat["wave"].value
                        / np.sqrt(niriss_res.value**2 - eff_disp.value**2)
                        / 1e4
                        / 2.35,
                        # 5,
                        # dat["wave"].value
                        xout=dat["wave"].value,
                    )
                    * u.uJy
                )
                dat["err"] = (
                    varsmooth(
                        dat["wave"].value,
                        dat["err"].value,
                        constants.c.to(u.km / u.s).value
                        * dat["wave"].value
                        / np.sqrt(niriss_res.value**2 - eff_disp.value**2)
                        / 1e4,
                        # 5,
                        # dat["wave"].value
                        xout=dat["wave"].value,
                    )
                    * u.uJy
                )

            dat["flux"][mask] = np.nan
            # print(dir(wave_resampled))
            flux_jansky, err = spectres(
                wave_resampled.value,
                dat["wave"].value,
                dat["flux"].value,
                dat["err"].value,
            )
            flux_jansky *= u.uJy
            err *= u.uJy
            # flux_jansky[err == 0] = np.nan
            # print (flux_jansky)
            flux_cgs = flux_jansky.to(
                u.erg / u.AA / u.cm**2 / u.s,
                equivalencies=u.spectral_density(wave_resampled),
            )
            flux_cgs_err = err.to(
                u.erg / u.AA / u.cm**2 / u.s,
                equivalencies=u.spectral_density(wave_resampled),
            )
            axs.plot(
                # dat["wave"] / (1 + obj_z),
                # wave,
                # wave_resampled,
                np.array(wave_resampled / (1 + obj_z_nirspec)),
                # convolve(flux_cgs, Gaussian1DKernel(5)),
                # convolve(flux_cgs, Gaussian1DKernel(kern_size)),
                flux_cgs.value + offset,
                drawstyle="steps-mid",
                c=colour,
                # label=f"NIRSpec, {resolution}" if i == 0 else None,
                label=label if i == 0 else None,
            )
            # print (np.array(convolve(flux_cgs-flux_cgs_err, Gaussian1DKernel(10))))
            axs.fill_between(
                # np.array(dat["wave"] / (1 + obj_z)),
                # np.array(wave),
                np.array(wave_resampled / (1 + obj_z_nirspec)),
                # convolve(flux_cgs, Gaussian1DKernel(5)),
                # np.array(convolve(flux_cgs - flux_cgs_err, Gaussian1DKernel(kern_size))),
                # np.array(convolve(flux_cgs + flux_cgs_err, Gaussian1DKernel(kern_size))),
                (flux_cgs - flux_cgs_err).value + offset,
                (flux_cgs + flux_cgs_err).value + offset,
                # flux_cgs
                step="mid",
                alpha=0.3,
                color=colour,
            )

    stage_4_dir = grizli_dir / "Extractions_v4"

    niriss_ids = [3514]

    z_off = 0.0115
    # z_off = 0
    obj_z = 0.299
    for obj_id in niriss_ids:
        for i, filt in enumerate(niriss_filter_sens.keys()):
            dat = Table.read(
                stage_4_dir / "1D" / f"{root_name}_{obj_id:0>5}.1D.fits", filt
            )
            dat["wave"] /= 1e4
            wave_lims = (dat["wave"] >= niriss_filter_sens[filt][0]) & (
                dat["wave"] <= niriss_filter_sens[filt][-1]
            )
            dat["wave"] /= 1 + obj_z
            axs.plot(
                # dat["wave"] / (1 + obj_z) / 1e4,
                dat["wave"][wave_lims],
                (dat["flux"] / dat["flat"])[wave_lims],
                drawstyle="steps-mid",
                label="NIRISS" if i == 0 else None,
                c="C1",
            )
            axs.fill_between(
                # dat["wave"] / (1 + obj_z) / 1e4,
                dat["wave"][wave_lims],
                ((dat["flux"] - dat["err"]) / dat["flat"])[wave_lims],
                ((dat["flux"] + dat["err"]) / dat["flat"])[wave_lims],
                step="mid",
                alpha=0.3,
                color="C1",
            )
            # axs.plot(
            #     # dat["wave"] / (1 + obj_z) / 1e4,
            #     dat["wave"] / 1e4,
            #     dat["cont"] / dat["flat"],
            #     drawstyle="steps-mid",
            # )
        # with fits.open(
        #     stage_4_dir / "full" / f"{root_name}_{obj_id:0>5}.full.fits"
        # ) as full_hdul:
        #     # full_hdul.info()
        #     temp_tab = Table(full_hdul["TEMPL"].data)
        #     print(temp_tab)
        #     obj_z = full_hdul[0].header["REDSHIFT"]
        #     temp_tab["wave"] /= 1e4

        #     wave_lims = (
        #         temp_tab["wave"] >= niriss_filter_sens["F115W"][0]  # / (1 + obj_z)
        #     ) & (temp_tab["wave"] <= niriss_filter_sens["F115W"][1])

        #     temp_tab["wave"] *= (1 + obj_z+z_off)/(1+obj_z)
        #     axs.plot(
        #         temp_tab["wave"][wave_lims],
        #         temp_tab["full"][wave_lims]-1e-18,
        #         drawstyle="steps-mid",
        #         c="forestgreen",
        #         label="Template",
        #     )

    fig.patch.set_alpha(0.0)

    # plt.savefig(save_dir / "test.pdf")

    # axs.set_xlim(xmax=2.3)
    axs.set_xlim([0.7, 3.5])

    # ca_lams = [8500.35, 8544.44, 8664.52]
    line_wavelengths = {}
    line_wavelengths["PaA"] = [18756.3]
    # line_ratios["PaA"] = [1.0]
    line_wavelengths["PaB"] = [12821.7]
    line_wavelengths["BrB"] = [26258.8]
    # line_wavelengths["BrA"] = [40522.8]
    # line_wavelengths["BrG"] = [21661.3]
    # line_wavelengths["BrD"] = [19451.0]
    # line_ratios["PaB"] = [1.0]
    # line_wavelengths["PaG"] = [10941.2]
    # line_ratios["PaG"] = [1.0]
    # line_wavelengths["PaD"] = [10052.2]
    # line_ratios["PaD"] = [1.0]
    # line_wavelengths["Pa8"] = [9548.65]
    line_wavelengths["CaII-8600"] = [8500.36, 8544.44, 8664.52]

    line_wavelengths["SIII-9531"] = [9533.2]
    line_wavelengths["SIII-9068"] = [9071.1]

    for lab, lams in line_wavelengths.items():
        for lam in np.atleast_1d(lams):
            axs.axvline(
                # lam * (1 + obj_z + z_off) / 1e4,
                lam / 1e4,
                linestyle=":",
                c="k",
                linewidth=1,
                zorder=-1,
            )

    axs.legend()
    axs.set_ylabel(r"$f_{\lambda}$ (Arbitrary Offset)")
    axs.set_xlabel(r"$\lambda_{\rm{restframe}}\,(\mu m)$")

    plt.show()
