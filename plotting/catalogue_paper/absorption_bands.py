"""
Plot the absorption bands in a cluster galaxy.
"""

import plot_utils
from astropy.convolution import Gaussian1DKernel, convolve
from default_imports import *

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
    ax = axs
    # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    obj_id = 3528
    # obj_id = 3677
    # obj_id = 3710
    # obj_id = 3822
    # obj_id = 3847
    # obj_id = 3859
    # obj_id = 3862

    extractions_dir = grizli_dir / "Extractions_v3"

    model_offset = 0.1
    spec_offset = 0.2
    norm_lam = [0.825, 0.875]

    with fits.open(
        extractions_dir / "full" / f"{root_name}_{obj_id:05}.full.fits"
    ) as full_hdul:
        # full_hdul.info()
        temp_tab = Table(full_hdul["TEMPL"].data)
        print(temp_tab)
        obj_z = full_hdul[0].header["REDSHIFT"]

    with fits.open(
        extractions_dir / "1D" / f"{root_name}_{obj_id:05}.1D.fits"
    ) as oned_hdul:
        # oned_hdul.info()
        filt_list = ["F115W"]
        for filt in filt_list:
            oned_tab = Table(oned_hdul[filt].data)
            print(oned_tab)

            oned_tab["wave"] /= 1e4 * (1 + obj_z)

            norm = np.nanmedian(
                (oned_tab["flux"] / oned_tab["flat"])[
                    (oned_tab["wave"] >= norm_lam[0])
                    & (oned_tab["wave"] <= norm_lam[1])
                ]
            )

            ax.plot(
                oned_tab["wave"],
                oned_tab["flux"] / oned_tab["flat"] / norm,
                drawstyle="steps-mid",
                c="darkorchid",
                label="Observed",
            )
            ax.plot(
                oned_tab["wave"],
                oned_tab["line"] / oned_tab["flat"] / norm - model_offset,
                drawstyle="steps-mid",
                c="lightskyblue",
                label="Model",
            )
            ax.fill_between(
                oned_tab["wave"],
                (oned_tab["flux"] + oned_tab["err"]) / oned_tab["flat"] / norm,
                (oned_tab["flux"] - oned_tab["err"]) / oned_tab["flat"] / norm,
                # drawstyle="steps-mid",
                color="darkorchid",
                step="mid",
                alpha=0.5,
            )

    ca_lams = [8500.35, 8544.44, 8664.52]

    for ca in ca_lams:
        ax.axvline(ca / 1e4, linestyle=":", c="k", linewidth=1, zorder=-1)

    with fits.open(
        extractions_dir / "full" / f"{root_name}_{obj_id:05}.full.fits"
    ) as full_hdul:
        # full_hdul.info()
        temp_tab = Table(full_hdul["TEMPL"].data)
        print(temp_tab)
        obj_z = full_hdul[0].header["REDSHIFT"]
        temp_tab["wave"] /= 1e4 * (1 + obj_z)

        wave_lims = (
            temp_tab["wave"] >= niriss_filter_sens["F115W"][0] / (1 + obj_z)
        ) & (temp_tab["wave"] <= niriss_filter_sens["F115W"][1] / (1 + obj_z))
        norm = np.nanmedian(
            temp_tab["full"][
                ((temp_tab["wave"]) >= norm_lam[0])
                & ((temp_tab["wave"]) <= norm_lam[1])
            ]
        )
        print(norm)
        ax.plot(
            temp_tab["wave"][wave_lims],
            temp_tab["full"][wave_lims] / norm - spec_offset,
            drawstyle="steps-mid",
            c="forestgreen",
            label="Template",
        )

    # import fsps

    # sp = fsps.StellarPopulation(
    #     add_neb_emission=True,
    #     # logzsol=1,
    #     cloudy_dust=True,
    #     sfh=0,
    #     # smooth_lsf=True
    #     zcontinuous=1,
    #     zred=obj_z,
    #     add_igm_absorption=True,
    #     dust_type=2,
    #     dust1=0.0,
    #     dust2=0.2,
    # )
    # print (sp.libraries)
    # print (sp.resolutions)
    # # sp.wavelengths = np.arange(1e4,2e4)
    # # for ax, flam in zip(axs, [True, False]):
    # # for flam in [True, False]:
    # for flam in [True]:
    #     for zsol in [0.5,1,1.5,2,2.5,3]:
    #         for age in np.arange(5,10):
    #             sp.params["logzsol"] = np.log10(zsol)
    #             # sp.params["logzsol"] = zsol
    #         # print (flam)
    #             waves, specs = sp.get_spectrum(
    #                 peraa=flam,
    #                 tage=age,
    #             )
    #             # sp.set_lsf(waves, sigma=2000*np.ones_like(waves))

    #             # waves, specs = sp.get_spectrum(
    #             #     peraa=flam,
    #             #     # tage=8.5,
    #             # )

    #             wave_min = 1e4
    #             wave_max = 2e4
    #             waves *= (1.+obj_z)
    #             wave_idx = (waves >= wave_min) & (waves <= wave_max)

    #             for s in np.atleast_2d(specs):
    #                 norm = np.nanmedian(s[(waves / 1e4 >= norm_lam[0]) & (waves /1e4 <= norm_lam[1])])
    #                 ax.plot(waves[wave_idx]/1e4, s[wave_idx]/norm)

    #                 ax.plot(waves[wave_idx]/1e4, convolve(s/norm - 0.2, Gaussian1DKernel(20))[wave_idx], linestyle=":")

    ylim = ax.get_ylim()[1]
    ax.annotate(
        "CaII+TiO+VO",
        xy=(0.855, ylim),
        xycoords="data",
        xytext=(0.855, ylim + 0.05),
        textcoords="data",
        arrowprops=dict(arrowstyle="-[,widthB=2.5", connectionstyle="arc3"),
        ha="center",
    )
    ax.annotate(
        "CN+TiO+ZrO",
        xy=(0.935, ylim),
        xycoords="data",
        xytext=(0.935, ylim + 0.05),
        textcoords="data",
        arrowprops=dict(arrowstyle="-[,widthB=2", connectionstyle="arc3"),
        ha="center",
    )

    ax.set_ylabel(r"$f_{\lambda}$ (Arbitrary Offset)")
    ax.set_xlabel(r"$\lambda_{\rm{restframe}}\,(\mu m)$")

    # ax.set_xlim(xmin=wave_min, xmax=wave_max)

    ax.legend()
    # ax.set_xlim(1,1.5)
    # ax.set_ylim(ymax=3)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "absorption_bands.pdf")

    plt.show()
