"""
Show the visibility of lines in the NIRISS filters.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    # print (full_cat["NUMBER"][
    #     (full_cat["Z_FLAG_ALL"]>=9)#
    #     & (full_cat["Z_NIRISS"]>=1.85)
    #     & (full_cat["Z_NIRISS"]<=1.9)
    # ])

    obj_id = 2938
    # 1983
    # 2938

    obj_z = full_cat[np.where(full_cat["NUMBER"] == obj_id)]["Z_NIRISS"]
    print(obj_z)

    pas = ["72.0", "341.0"]
    filts = ["F115W", "F150W", "F200W"]

    stack_path = [
        *(grizli_dir / "Extractions_v4").glob(
            f"**/*glass-a2744_{obj_id:0>5}.stack.fits"
        )
    ][0]

    widths = [0.15]

    with fits.open(stack_path) as stack_hdul:
        # stack_hdul.info()
        for i, f in enumerate(filts):
            for j, p in enumerate(pas[:-1]):
                # print (f+","+p)
                # wht_i = stack_hdul["WHT", f + "," + p].data
                sci_i = stack_hdul["SCI", f + "," + p].data
                # kern_i = stack_hdul["KERNEL", f + "," + p].data
                h_i = stack_hdul["SCI", f + "," + p].header

                sh = sci_i.shape
                extent = [h_i["WMIN"], h_i["WMAX"], 0, sh[0]]
                # print (sh[1])
                # print ((h_i["WMAX"]-h_i["WMIN"]) / sh[1])

                print(extent)
                # widths.append(0.15)
                widths.append(h_i["WMAX"] - h_i["WMIN"])
    print(widths)

    # exit()
    fig, axs = plt.subplots(
        3,
        4,
        figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_columnwidth / 1.1),
        constrained_layout=True,
        # use_gridspec=True,
        sharex="col",
        sharey="row",
        width_ratios=widths,
        height_ratios=[1, 1, 2],
    )

    hatch_colour = "k"
    hatch_kwargs = {
        "facecolor": "k",
        "edgecolor": hatch_colour,
        "step": "mid",
        "alpha": 0.3,
        "hatch": "//",
        "linewidth": 1,
        "zorder": 3,
    }

    # obj_z = None

    fig.delaxes(axs[2, 0])
    # fig.delaxes(axs[1,0])
    # fig.delaxes(axs[0,0])
    with fits.open(stack_path) as stack_hdul:
        # stack_hdul.info()
        for i, f in enumerate(filts):
            for j, p in enumerate(pas):
                if j == 0:
                    axs[j, i + 1].set_title(f)
                if i == 0:
                    axs[j, i].annotate(
                        rf"P.A.\,=\,{p}$^\circ$",
                        (-0.05, 0.5),
                        xycoords="axes fraction",
                        va="center",
                        ha="right",
                        # bbox=dict(boxstyle="round", fc="w"),
                        rotation=90,
                    )
                    kern_i = stack_hdul["KERNEL", f + "," + p].data
                    vmax_kern = 2 * np.percentile(kern_i.data, 99.5)
                    norm_kern = astrovis.ImageNormalize(
                        data=kern_i,
                        stretch=astrovis.SqrtStretch(),
                        interval=astrovis.ManualInterval(
                            vmin=0,
                            vmax=vmax_kern,
                        ),
                    )
                    axs[j, i].set_yticklabels([])

                    # Kernel
                    sh_kern = kern_i.data.shape
                    extent_kern = [0, sh_kern[1], 0, sh_kern[0]]

                    axs[j, i].imshow(
                        kern_i,
                        origin="lower",
                        interpolation="none",
                        norm=norm_kern,
                        # vmin=-0.1 * vmax_kern,
                        # vmax=vmax_kern,
                        cmap=cm.berlin,
                        extent=extent_kern,
                        aspect="equal",
                    )

                # print (f+","+p)
                wht_i = stack_hdul["WHT", f + "," + p].data
                sci_i = stack_hdul["SCI", f + "," + p].data
                contam_i = stack_hdul["CONTAM", f + "," + p].data
                h_i = stack_hdul["SCI", f + "," + p].header

                clip = wht_i > 0 & (contam_i < 0.1 * sci_i)
                if clip.sum() == 0:
                    clip = np.isfinite(wht_i)

                vmax = 0.5
                norm = astrovis.ImageNormalize(
                    data=sci_i,
                    stretch=astrovis.SqrtStretch(),
                    interval=astrovis.ManualInterval(
                        vmin=-0.005,
                        vmax=vmax,
                    ),
                )
                # Spectrum
                sh = sci_i.shape
                extent = [h_i["WMIN"], h_i["WMAX"], 0, sh[0]]
                print(extent)

                axs[j, i + 1].imshow(
                    # sci_i,
                    sci_i + contam_i,
                    origin="lower",
                    interpolation="none",
                    # vmin=-0.1 * vmax,
                    # vmax=vmax,
                    norm=norm,
                    cmap=cm.berlin,
                    extent=extent,
                    aspect="auto",
                )
                # axs[j, i].set_xticklabels([])
                # axs[j, i + 1].set_yticklabels([])

                axs[j, i + 1].fill_betweenx(
                    [0, sh[0]], h_i["WMIN"], niriss_filter_sens[f][0], **hatch_kwargs
                )
                axs[j, i + 1].fill_betweenx(
                    [0, sh[0]], niriss_filter_sens[f][-1], h_i["WMAX"], **hatch_kwargs
                )
                # axs[j, i].xaxis.set_tick_params(length=0)
                # axs[j, i].yaxis.set_tick_params(length=0)
                # from matplotlib.ticker import MultipleLocator
                # axs[j, (2*i)+1].xaxis.set_major_locator(MultipleLocator(0.1))

    oned_path = [
        *(grizli_dir / "Extractions_v2").glob(f"**/*glass-a2744_{obj_id:0>5}.1D.fits")
    ][0]

    line_dict = {
        r"Ly$\alpha$": 1215.24,
        # "MgII": 2799.12,
        "OII": 3728.4835,
        "OIII-4958": 4960.295,
        "OIII": 5008.24,
        # "OIII" : 4984.2675,
        r"H$\beta$": 4862.68,
        r"H$\alpha$": 6564.61,
        # "SII": 6731,
        "SIII": 9533.2,
        "PaB": 12821.7,
    }

    max_ylim = 0.0
    with fits.open(oned_path) as oned_hdul:
        # oned_hdul.info()
        for i, f in enumerate(filts):
            oned_tab = Table(oned_hdul[f].data)
            # print (oned_tab.colnames)
            print(len(oned_tab["wave"]))
            # axs[2, 2*i].remove()
            axs[2, i + 1].plot(
                oned_tab["wave"] / 1e4,
                oned_tab["flux"] / oned_tab["flat"] / 1e-19,
                drawstyle="steps-mid",
                c="#d55e00",
                label="Data" if i == 2 else "_",
            )
            axs[2, i + 1].fill_between(
                oned_tab["wave"] / 1e4,
                (oned_tab["flux"] + oned_tab["err"]) / oned_tab["flat"] / 1e-19,
                (oned_tab["flux"] - oned_tab["err"]) / oned_tab["flat"] / 1e-19,
                # drawstyle="steps-mid",
                color="#d55e00",
                step="mid",
                alpha=0.5,
            )
            if np.nanmax(oned_tab["flux"] / oned_tab["flat"] / 1e-19) > max_ylim:
                max_ylim = np.nanmax(oned_tab["flux"] / oned_tab["flat"] / 1e-19)
            axs[2, i + 1].plot(
                oned_tab["wave"] / 1e4,
                oned_tab["contam"] / oned_tab["flat"] / 1e-19,
                c="#0072b2",
                alpha=0.7,
                drawstyle="steps-mid",
                zorder=-1,
                label="Contamination" if i == 2 else "_",
            )
            axs[2, i + 1].fill_betweenx(
                [-1, 100],
                oned_tab["wave"][0] / 1e4,
                niriss_filter_sens[f][0],
                **hatch_kwargs,
            )
            axs[2, i + 1].fill_betweenx(
                [-1, 100],
                niriss_filter_sens[f][-1],
                oned_tab["wave"][-1] / 1e4,
                **hatch_kwargs,
                label=r"$<50\%$ Sensitivity" if i == 2 else "_",
            )
            axs[2, i + 1].set_ylim(ymin=0)
            from matplotlib.ticker import MultipleLocator

            if i == 0:
                axs[2, i + 1].yaxis.set_major_locator(MultipleLocator(2.0))
                axs[2, i + 1].set_yticklabels([])
                # print ((oned_tab["wave"][0] / 1e4 - 0.1,0))
                for s in np.arange(0, 12, 2):
                    an = axs[2, i + 1].annotate(
                        f"{int(s)}",
                        (oned_tab["wave"][0] / 1e4 - 0.0125, s - 0.1),
                        # xycoords
                        annotation_clip=False,
                        va="center",
                        ha="right",
                    )
                    an.set_in_layout(False)
            if i == 2:
                axs[2, i + 1].legend(fontsize=7, loc=1)

            ylim = axs[2, i + 1].get_ylim()[1]
            for l, v in line_dict.items():
                # print (v/1e4,  niriss_filter_sens[f][0])
                line_w = (1 + obj_z) * v / 1e4
                if (line_w >= niriss_filter_sens[f][0]) and (
                    line_w < niriss_filter_sens[f][-1]
                ):
                    axs[2, i + 1].axvline(line_w, c="k", linewidth=1, linestyle=":")
                    if l != "OIII-4958":
                        axs[2, i + 1].annotate(
                            rf"{l}",
                            xy=(line_w + 0.005 - (0.005 if l == "OIII" else 0), 1.1),
                            xycoords=axs[2, i + 1].get_xaxis_transform(),
                            # xytext=(0.855, ylim + 0.05),
                            # textcoords="data",
                            # arrowprops=dict(arrowstyle="-[,widthB=2.5", connectionstyle="arc3"),
                            ha="center",
                            va="center",
                            rotation=60,
                        )
    for i, f in enumerate(filts):
        axs[2, i + 1].set_ylim(ymax=1.05 * max_ylim)
        axs[2, i + 1].set_xlabel(r"Observed Wavelength ($\mu$m)")
        # axs[2, i + 1].set_ylabel(r"$f_{\lambda} [10^{-19}\,\rm{erg}\,\rm{s}^{-1}\,\rm{cm}^{-2}\,\AA^{-1}]$")

    # axs[2, (0) + 1].set_ylabel(r"$f_{\lambda} [10^{-19}\,\rm{erg}/\rm{s}/\rm{cm}^{2}/\AA]$")
    an = axs[2, 1].annotate(
        r"$f_{\lambda}\ [10^{-19}\,\rm{erg}/\rm{s}/\rm{cm}^{2}/\AA]$",
        (-0.125, 0.5),
        ha="right",
        va="center",
        xycoords="axes fraction",
        rotation="vertical",
    )
    an.set_in_layout(False)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "niriss_beams_example.pdf", dpi=600)

    # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0,
    #                         wspace=0)
    # fig.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.show()
    # # gs = fig.add_gridspec(3, 3)#, height_ratios=(1, 2))
    # # ax = fig.add_subplot(gs[1, 0])

    # z_min = 0
    # z_max = 3
    # z_range = np.linspace(z_min, z_max, 100)

    # # for k, v in line_dict.items():
    # #     ax.plot(z_range, (1 + z_range) * v)

    # # for k, v in niriss_info.items():
    # #     ax.fill_between(z_range, v[0], v[1], color="k", alpha=0.4, edgecolor="none")
    # for k, v in line_dict.items():
    #     ax.plot((1 + z_range) * v / 1e4, z_range, c="k", linewidth=0.5)

    # for k, v in niriss_info.items():
    #     ax.fill_betweenx(z_range, v[0], v[1], color="k", alpha=0.2, edgecolor="none")

    # ax.set_xlim(0.95, 2.300)
    # ax.set_ylim(z_min, z_max)
    # ax.axhline(1.1, c="k", linestyle=":")
    # ax.axhline(1.35, c="k", linestyle=":")
    # ax.axhline(1.9, c="k", linestyle=":")
    # ax.axhline(0.3064, c="k", linestyle=":", label="Cluster")

    # # hist = partial(plot_utils.plot_hist, bins=np.arange(16.5, 33.5, 0.5), ax=ax)
    # # # print (np.nanmin(v1_cat["MAG_AUTO"]), np.nanmax(v1_cat["MAG_AUTO"])

    # # hist(v1_cat["MAG_AUTO"], label="Full Sample")
    # # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] > 0], ax=ax, label="Extracted")
    # # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 4], ax=ax, label="First Pass")
    # # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 5], ax=ax, label="Placeholder")

    # ax.set_xlabel(r"Observed Wavelength ($\mu$m)")
    # ax.set_ylabel(r"$z$")
    # # ax.legend()

    # for k, v in line_dict.items():
    #     for k_n, v_n in niriss_info.items():
    #         # low = v_n[0]/v -1
    #         print(f"{k}: {k_n}={v_n[0]/v-1:.3f},{v_n[1]/v-1:.3f}")

    # ax_filts = fig.add_subplot(gs[0], sharex=ax)
    # plt.setp(ax_filts.get_xticklabels(), visible=False)
    # # for f in filter_dir.glob("*.txt"):
    # for c, f in zip(["blue", "green", "red"], niriss_info.keys()):
    #     print(f)
    #     filt_tab = Table.read(filter_dir / f"NIRISS_{f}.txt", format="ascii")
    #     # print (filt_tab)
    #     filt_tab["PCE"][filt_tab["PCE"] < 1e-3] = np.nan
    #     ax_filts.plot(filt_tab["Wavelength"], filt_tab["PCE"], c=c)
    #     # half_sens_idx = [
    #     #     np.nanargmax(filt_tab["PCE"] <= 0.5 * np.nanmax(filt_tab["PCE"])),
    #     #     np.nanargmax(filt_tab["PCE"] >= 0.5 * np.nanmax(filt_tab["PCE"]))
    #     # ]
    #     half_sens_idx = np.argwhere(filt_tab["PCE"] >= 0.5 * np.nanmax(filt_tab["PCE"]))
    #     #     np.nanargmax(filt_tab["PCE"] >= 0.5 * np.nanmax(filt_tab["PCE"]))
    #     # ]
    #     # print (half_sens_idx)

    #     ax_filts.fill_betweenx(
    #         [0, 1],
    #         filt_tab["Wavelength"][half_sens_idx[0]],
    #         filt_tab["Wavelength"][half_sens_idx[-1]],
    #         color=c,
    #         alpha=0.2,
    #         edgecolor="none",
    #     )
    #     ax.fill_betweenx(
    #         z_range,
    #         filt_tab["Wavelength"][half_sens_idx[0]],
    #         filt_tab["Wavelength"][half_sens_idx[-1]],
    #         color=c,
    #         alpha=0.2,
    #         edgecolor="none",
    #     )
    #     # print(
    #     #     filt_tab["Wavelength"][half_sens_idx[0]],
    #     #     filt_tab["Wavelength"][half_sens_idx[-1]],
    #     # )
    # ax_filts.set_ylabel(r"Throughput")
    # ax_filts.set_ylim(0, 1)

    # plt.savefig(save_dir / "svg" / "wavelength_coverage.svg", dpi=600)

    # plt.show()
