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

    if len([*save_dir.glob(f"glass-a2744_{obj_id:0>5}.beams_*.fits")]) == 0:
        beams_path = [
            *(grizli_dir / "Extractions_v4").glob(
                f"**/*glass-a2744_{obj_id:0>5}.beams.fits"
            )
        ][0]

        print(beams_path)

        from grizli.multifit import MultiBeam

        mb_full = MultiBeam(str(beams_path), min_mask=0.0, min_sens=0.0)
        print(dir(mb_full))

        beams_dict = {341.0: [], 72.0: []}
        for filt, filt_dict in mb_full.PA.items():
            # beams_dict
            for pa, beam_idxs in filt_dict.items():
                # print(pa)
                # print(pa.type())
                beams_dict[pa].extend(beam_idxs)
        # print(beams_dict)
        # print(mb_full.PA)

        for pa, beam_idxs in beams_dict.items():
            mb_single = MultiBeam(
                [b for i, b in enumerate(mb_full.beams) if i in beam_idxs],
                min_sens=0.0,
                min_mask=0.0,
            )
            mb_single.oned_spectrum_to_hdu(
                outputfile=save_dir / f"glass-a2744_{obj_id:0>5}.beams_{pa:.1f}.fits"
            )

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
    import matplotlib

    matplotlib.use("pdf")
    hatch_colour = "k"
    # hatch_colour = (0,0,0,0.3)
    hatch_kwargs = {
        "color": hatch_colour,
        # "color": "k",
        # "facecolor": hatch_colour,
        # "edgecolor": hatch_colour,
        "step": "mid",
        "alpha": 0.3,
        "hatch": "//",
        "linewidth": 0,
        "zorder": 3,
        # "hatchcolor" : hatch_colour
    }
    # hatch_colour = (0,0,0,0.3)

    # Hatch colours, only bugged for 7+ years in matplotlib
    hatch_only_kwargs = {
        "color": "none",
        # "edgecolor": hatch_colour,
        "step": "mid",
        "alpha": 0.3,
        "hatch": "//",
        # "linewidth": 1,
        "zorder": 3,
        # "hatchcolor" : hatch_colour
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
                # axs[j, i + 1].fill_betweenx(
                #     [0, sh[0]], h_i["WMIN"], niriss_filter_sens[f][0], **hatch_only_kwargs
                # )
                # axs[j, i + 1].fill_betweenx(
                #     [0, sh[0]], niriss_filter_sens[f][-1], h_i["WMAX"], **hatch_only_kwargs
                # )
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
        r"$\left[\ion{O}{ii}\right]$": 3728.4835,
        "OIII-4958": 4960.295,
        r"$\left[\ion{O}{iii}\right]$": 5008.24,
        # "OIII" : 4984.2675,
        r"H$\beta$": 4862.68,
        r"H$\alpha$": 6564.61,
        # "SII": 6731,
        "SIII": 9533.2,
        "PaB": 12821.7,
    }

    # single_pa_beams = [*save_dir.glob(f"glass-a2744_{obj_id:0>5}.beams*.fits")]
    # single_pa_beams.sort(reverse=True)
    # for p, c in zip(single_pa_beams, ["#009E73", "#CC79A7"]):
    #     with fits.open(p) as oned_hdul:
    #         print (p.stem.split("beams_"))
    #         # oned_hdul.info()
    #         for i, f in enumerate(filts):
    #             oned_tab = Table(oned_hdul[f].data)
    #             # print (oned_tab.colnames)
    #             print(len(oned_tab["wave"]))
    #             # axs[2, 2*i].remove()
    #             axs[2, i + 1].plot(
    #                 oned_tab["wave"] / 1e4,
    #                 oned_tab["flux"] / oned_tab["flat"] / 1e-19,
    #                 drawstyle="steps-mid",
    #                 c=c,
    #                 label=rf"$\rm{{P.A.}}={p.stem.split("beams_")[-1]}^\circ$" if i == 2 else "_",
    #                 # linestyle=":",
    #                 alpha=1.,
    #                 linewidth=0.5
    #             )

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
                c="#D55E00",
                label="Combined" if i == 2 else "_",
                linewidth=0.5,
            )
            axs[2, i + 1].fill_between(
                oned_tab["wave"] / 1e4,
                (oned_tab["flux"] + oned_tab["err"]) / oned_tab["flat"] / 1e-19,
                (oned_tab["flux"] - oned_tab["err"]) / oned_tab["flat"] / 1e-19,
                # drawstyle="steps-mid",
                color="#D55E00",
                step="mid",
                alpha=0.5,
            )
            if np.nanmax(oned_tab["flux"] / oned_tab["flat"] / 1e-19) > max_ylim:
                max_ylim = np.nanmax(oned_tab["flux"] / oned_tab["flat"] / 1e-19)
            axs[2, i + 1].plot(
                oned_tab["wave"] / 1e4,
                oned_tab["contam"] / oned_tab["flat"] / 1e-19,
                c="#0072B2",
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
            # axs[2, i + 1].fill_betweenx(
            #     [-1, 100],
            #     oned_tab["wave"][0] / 1e4,
            #     niriss_filter_sens[f][0],
            #     **hatch_only_kwargs,
            # )
            # axs[2, i + 1].fill_betweenx(
            #     [-1, 100],
            #     niriss_filter_sens[f][-1],
            #     oned_tab["wave"][-1] / 1e4,
            #     **hatch_only_kwargs,
            # )
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
                    axs[2, i + 1].axvline(
                        line_w, c="k", linewidth=1, linestyle=":", zorder=-10
                    )
                    if l != "OIII-4958":
                        axs[2, i + 1].annotate(
                            rf"{l}",
                            xy=(
                                line_w
                                + 0.005
                                + (0.005 if l == r"$\left[\ion{O}{ii}\right]$" else 0),
                                1.1 + (0.05 if "ion" in l else 0),
                            ),
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

    # plt.savefig(save_dir / "niriss_beams_example.svg", dpi=600)
    plt.savefig(save_dir / "niriss_beams_example_rev1.pdf", dpi=600)
    plt.show()
