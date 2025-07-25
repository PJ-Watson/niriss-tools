"""
Compare NIRISS to NIRSpec.
"""

import plot_utils
from astropy.convolution import Gaussian1DKernel, convolve
from astropy.nddata import block_reduce
from default_imports import *
from spectres import spectres

plot_utils.setup_aanda_style()


def convolve_to_niriss_res(orig_wavs, orig_flux, orig_R, new_wavs=None):

    niriss_sigma = constants.c.to(u.km / u.s) / np.interp(
        orig_wavs,
        np.arange(0.5, 5, 0.01),
        np.arange(0.5, 5, 0.01) / 9.4e-3,
    )
    orig_sigma = constants.c.to(u.km / u.s) / orig_R

    return varsmooth(
        orig_wavs,
        orig_flux,
        orig_wavs
        * np.sqrt(niriss_sigma.value**2 - orig_sigma.value**2)
        / constants.c.to(u.km / u.s).value,
        # 5,
        # dat["wave"].value
        # xout=dat["wave"].value,
        xout=new_wavs if new_wavs else orig_wavs,
    )


if __name__ == "__main__":

    # fig, axs = plt.subplots(
    #     1,
    #     1,
    #     figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_columnwidth * 2),
    #     constrained_layout=True,
    #     # sharex=True,
    #     # sharey=True,
    #     # hspace=0.,
    #     # wspace=0.,
    # )
    # fig = plt.figure(
    #     figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_columnwidth*2),
    #     constrained_layout=True,
    # )
    # gs = fig.add_gridspec(2, 3)
    # axs = fig.add_subplot(gs[0,sharey="row"
    fig, axes = plt.subplot_mosaic(
        "AAAA;BCDE",
        figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_columnwidth * 1.75),
        constrained_layout=True,
        # sharey="row",
        gridspec_kw={"height_ratios": [1, 0.6]},
    )
    axs = axes["A"]
    # gs_zoom = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec = gs[1], height_ratios=((z_max-z_cut_high)/((z_cut_low-z_min)/2), 2), hspace=0.025)
    # gs = fig.add_gridspec(3, 1, height_ratios=(1, (z_max-z_cut_high)/((z_cut_low-z_min)/2)*0.8, 2))
    # ax_top = fig.add_subplot(gs1[0, 0])

    # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    nirspec_dir = niriss_dir / "match_catalogues" / "nirspec_data"

    import msaexp
    from msaexp.resample_numba import resample_template_numba

    disp_dir = Path(msaexp.__file__).parent / "data"
    from astropy import constants
    from ppxf.ppxf_util import varsmooth

    # obj_z_nirspec = 0.3073
    obj_z_nirspec = 0
    for resolution, colour, offset, dispersion, label in zip(
        [0.000, 0.0045],
        ["C2", "C0"],
        [3e-18, 1e-18],
        [0, 1000],
        ["NIRSpec", r"NIRSpec ($R_{\rm{NIRISS}}$)"],
    ):

        for i, f in enumerate(nirspec_dir.glob("**/*.fits")):
            dat = QTable.read(f, "SPEC1D")

            if resolution != 0:
                wave_resampled = np.arange(0.805, 5.205, resolution) * u.um
            else:
                wave_resampled = dat["wave"]

            grating = dat.meta["GRATING"].lower()
            # print ((disp_dir / f"jwst_nirspec_{grating}_disp.fits").is_file())
            disp_tab = Table.read(disp_dir / f"jwst_nirspec_{grating}_disp.fits")
            # print(disp_tab)
            eff_R = np.interp(dat["wave"], disp_tab["WAVELENGTH"], disp_tab["R"])

            mask = dat["err"] == 0

            dat["flux"] *= 2.1
            dat["err"] *= 2.1

            if dispersion > 0:
                # xi = np.arange(len(dat["flux"]))
                dat["flux"][mask] = np.nanmedian(dat["flux"][~mask])
                dat["flux"] = (
                    convolve_to_niriss_res(
                        dat["wave"].value,
                        dat["flux"].value,
                        eff_R,
                    )
                    * u.uJy
                )
                dat["err"] = (
                    convolve_to_niriss_res(
                        dat["wave"].value,
                        dat["err"].value,
                        eff_R,
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
                np.array(wave_resampled / (1 + obj_z_nirspec)),
                flux_cgs.value + offset,
                drawstyle="steps-mid",
                c=colour,
                # label=f"NIRSpec, {resolution}" if i == 0 else None,
                label=label if i == 0 else None,
            )
            axs.fill_between(
                np.array(wave_resampled / (1 + obj_z_nirspec)),
                (flux_cgs - flux_cgs_err).value + offset,
                (flux_cgs + flux_cgs_err).value + offset,
                # flux_cgs
                step="mid",
                alpha=0.3,
                color=colour,
            )

    stage_4_dir = grizli_dir / "Extractions_v4"

    niriss_ids = [3514]

    # z_off = 0.0115
    # z_off = 0
    flux_off = 0e-18
    obj_z = 0.3072
    for obj_id in niriss_ids:
        for i, filt in enumerate(niriss_filter_sens.keys()):
            dat = Table.read(
                stage_4_dir / "1D" / f"{root_name}_{obj_id:0>5}.1D.fits", filt
            )
            dat["wave"] /= 1e4
            wave_lims = (dat["wave"] >= niriss_filter_sens[filt][0]) & (
                dat["wave"] <= niriss_filter_sens[filt][-1]
            )
            # dat["wave"] /= 1 + obj_z
            # dat["flux"] += flux_off
            axs.plot(
                # dat["wave"] / (1 + obj_z) / 1e4,
                dat["wave"][wave_lims],
                (dat["flux"] / dat["flat"])[wave_lims] + flux_off,
                drawstyle="steps-mid",
                label="NIRISS" if i == 0 else None,
                c="C1",
            )
            axs.fill_between(
                # dat["wave"] / (1 + obj_z) / 1e4,
                dat["wave"][wave_lims],
                ((dat["flux"] - dat["err"]) / dat["flat"])[wave_lims] + flux_off,
                ((dat["flux"] + dat["err"]) / dat["flat"])[wave_lims] + flux_off,
                step="mid",
                alpha=0.3,
                color="C1",
            )
    fig.patch.set_alpha(0.0)

    # axs.set_xlim([0.7, 3.5])
    axs.set_xlim([1.0, 4.5])

    line_dict = {
        r"$\left[\ion{S}{iii}\right]\lambda{9068}$": 9071.1,
        r"$\left[\ion{S}{iii}\right]\lambda{9531}$": 9533.2,
        r"\ion{Pa}{$\alpha$}": 18756.3,
        r"\ion{Br}{$\beta$}": 26258.8,
    }

    for a, (line_name, line_lam) in zip(
        [axes["B"], axes["C"], axes["D"], axes["E"]],
        line_dict.items(),
        # [[0.875, 0.975], [1.83, 1.93], [2.6, 2.64]]
    ):
        colour = "C2"
        line_lam /= 1e4
        spacing = 0.01
        xlims = np.array([line_lam - spacing, line_lam + spacing]) * (1 + obj_z)
        for i, f in enumerate(nirspec_dir.glob("**/*.fits")):
            dat = QTable.read(f, "SPEC1D")
            dat = dat[
                (dat["wave"].to(u.micron).value >= xlims[0])
                & (dat["wave"].to(u.micron).value <= xlims[-1])
            ]
            flux_cgs = dat["flux"].to(
                u.erg / u.AA / u.cm**2 / u.s,
                equivalencies=u.spectral_density(dat["wave"]),
            )
            flux_cgs_err = dat["err"].to(
                u.erg / u.AA / u.cm**2 / u.s,
                equivalencies=u.spectral_density(dat["wave"]),
            )
            a.plot(
                np.array(dat["wave"]),
                flux_cgs.value,
                drawstyle="steps-mid",
                c=colour,
                # label=f"NIRSpec, {resolution}" if i == 0 else None,
                label=label if i == 0 else None,
            )
            a.fill_between(
                np.array(dat["wave"]),
                (flux_cgs - flux_cgs_err).value,
                (flux_cgs + flux_cgs_err).value,
                # flux_cgs
                step="mid",
                alpha=0.3,
                color=colour,
            )

        a.set_xlim(xlims)
        # a.annotate(line_name, (0.1, 0.1), xycoords="axes fraction")
        a.set_title(line_name)

    for _, a in axes.items():

        # axs.set_xlim(xlims)
        # ca_lams = [8500.35, 8544.44, 8664.52]
        line_wavelengths = {}
        line_ratios = {}
        line_wavelengths["PaA"] = [18756.3]
        # line_ratios["PaA"] = [1.0]
        # line_wavelengths["PaB"] = [12821.7]
        line_wavelengths["BrB"] = [26258.8]
        # line_wavelengths["HeI-1083"] = [10832.057, 10833.306]
        # line_wavelengths["BrA"] = [40522.8]
        # line_wavelengths["BrG"] = [21661.3]
        # line_wavelengths["BrD"] = [19451.0]
        # line_ratios["PaB"] = [1.0]
        # line_wavelengths["PaG"] = [10941.2]
        # line_ratios["PaG"] = [1.0]
        # line_wavelengths["PaD"] = [10052.2]
        # line_ratios["PaD"] = [1.0]
        # line_wavelengths["Pa8"] = [9548.65]
        # line_wavelengths["HeII-16923"] = [1.69230e4]
        # line_wavelengths["Pa9"] = [9231.60]
        # line_wavelengths["CaII-8600"] = [8500.36, 8544.44, 8664.52]
        # line_wavelengths["FeII-11128"] = [1.11286e4]
        # line_ratios["FeII-11128"] = [1.0]
        # line_wavelengths["FeII-12570"] = [1.25702e4]
        # line_ratios["FeII-12570"] = [1.0]
        # line_wavelengths["FeII-16440"] = [1.64400e4]
        # line_ratios["FeII-16440"] = [1.0]
        # line_wavelengths["FeII-16877"] = [1.68778e4]
        # line_ratios["FeII-16877"] = [1.0]
        # line_wavelengths["FeII-17418"] = [1.74188e4]
        # line_ratios["FeII-17418"] = [1.0]
        # line_wavelengths["FeII-17418"] = [1.74188e4]
        # line_ratios["FeII-17418"] = [1.0]
        # line_wavelengths["FeII-18362"] = [1.83624e4]
        # line_ratios["FeII-18362"] = [1.0]

        line_wavelengths["SIII-9531"] = [9533.2]
        line_wavelengths["SIII-9068"] = [9071.1]

        for lab, lams in line_wavelengths.items():
            for lam in np.atleast_1d(lams):
                a.axvline(
                    lam * (1 + obj_z) / 1e4,
                    # lam / 1e4,
                    linestyle="--",
                    c="k",
                    linewidth=1,
                    zorder=-1,
                    label=r"$z=0.307$" if lab == "PaA" else None,
                )

        if a == axes["B"]:
            a.set_ylabel(r"$f_{\lambda}$ (Arbitrary Offset)")
        a.set_xlabel(r"$\lambda_{\rm{obs}}\,(\mu m)$")

    for _, a in axes.items():

        line_wavelengths = {}
        line_wavelengths["Ha"] = [6564.61]
        line_wavelengths["Pa9"] = [9231.60]

        for lab, lams in line_wavelengths.items():
            for lam in np.atleast_1d(lams):
                a.axvline(
                    lam * (1 + 2.73) / (1 + obj_z_nirspec) / 1e4,
                    # lam / 1e4,
                    linestyle=":",
                    c="r",
                    linewidth=1,
                    zorder=-1,
                    label=r"$z=2.736$" if lab == "Ha" else None,
                )
        a.grid(color="k", alpha=0.05, zorder=-1)

    axs.legend()
    axs.set_ylabel(r"$f_{\lambda}$ (Arbitrary Offset)")
    axs.set_xlabel(r"$\lambda_{\rm{obs}}\,(\mu m)$")
    axs.grid(color="k", alpha=0.05, zorder=-1)

    plt.savefig(save_dir / "nirspec_comparison_rev1.pdf", dpi=600)
    plt.show()
