"""
Examples of the redshift quality flags.
"""

import plot_utils
from astrocolour import ColourImage
from astrocolour.utils import RawAsinhTransform
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    # print (full_cat["NUMBER"][
    #     (full_cat["Z_FLAG_ALL"]>=9)#
    #     & (full_cat["Z_NIRISS"]>=1.85)
    #     & (full_cat["Z_NIRISS"]<=1.9)
    # ])

    # obj_id = 2938
    # 1983
    # 2938

    # obj_list = {2663: "F200W", 2185: "F115W"}
    print(niriss_cat.colnames)

    # check_filt = "F200W"
    # check_angle = "72.0"
    # # niriss_cat = niriss_cat[
    #     (niriss_cat["Z_FLAG"] >= 4)
    #     & (niriss_cat[f"{check_filt}_{check_angle:0>5}_QUALITY"] == 0)
    # ]
    # with np.printoptions(threshold=np.inf):
    #     # print(niriss_cat)
    #     niriss_cat.pprint(len(niriss_cat) + 2)

    # obj_list_full = [
    #     {2663: ["F200W", "341.0"], 2185: ["F115W", "72.0"]},
    #     {3028: ["F200W", "341.0"], 682: ["F200W", "341.0"]},
    #     {690: ["F200W", "341.0"], 882: ["F200W", "72.0"]},
    # ]

    obj_list_full = np.array(
        [
            [1316, 3062],
            [754, 345],
            [3394, 1223],
        ]
    )
    # obj_list_full = obj_list_full[0]
    # offset = 8
    # # obj_list_full = [
    # #     {niriss_cat["ID_NIRISS"][offset]: ["F200W","341.0"],niriss_cat["ID_NIRISS"][offset+1]: ["F200W","341.0"]},
    # #     {niriss_cat["ID_NIRISS"][offset+2]: ["F200W","341.0"],niriss_cat["ID_NIRISS"][offset+3]: ["F200W","341.0"]}
    # # ]
    # extver = [check_filt, check_angle]
    # obj_list_full = [
    #     {
    #         niriss_cat["ID_NIRISS"][offset]: extver,
    #         niriss_cat["ID_NIRISS"][offset + 1]: extver,
    #     },
    #     {
    #         niriss_cat["ID_NIRISS"][offset + 2]: extver,
    #         niriss_cat["ID_NIRISS"][offset + 3]: extver,
    #     },
    # ]
    # print(obj_list_full)
    # print (niriss_cat[niriss_cat["Z_NIRISS"]>=6])
    # obj_list_full = niriss_cat[niriss_cat["Z_NIRISS"]>=6]["ID_NIRISS"]
    fig = plt.figure(
        1,
        constrained_layout=True,
    )
    fig.set_size_inches(
        plot_utils.aanda_textwidth * 1.0,
        plot_utils.aanda_columnwidth * len(obj_list_full.ravel()) * 0.5,
    )
    # subfigs = fig.subfigures(*obj_list_full.shape)
    subfigs = fig.subfigures(len(obj_list_full.ravel()))

    hatch_colour = "k"
    hatch_kwargs = {
        "facecolor": "k",
        "edgecolor": hatch_colour,
        "step": "mid",
        "alpha": 0.2,
        "hatch": "//",
        "linewidth": 1,
        "zorder": 3,
    }
    #

    print(niriss_cat[niriss_cat["Z_NIRISS"] >= 6])
    # print (obj_list[0])
    # print (dict())
    # exit()
    border = 5
    bgr = np.array(
        [
            fits.getdata(
                grizli_dir / "Prep" / f"{root_name}-f{filt}wn-clear_drc_sci.fits"
            )
            for filt in [115, 150, 200]
        ]
    )

    for o, obj_id in enumerate(obj_list_full.ravel()):
        # print(dir(subfigs.ravel()[0]))
        # gs = subfigs.ravel()[o].add_gridspec(2, 2, width_ratios=[1, 3])
        gs = subfigs.ravel()[o].add_gridspec(1, 3, width_ratios=[1, 3, 1])

        # print(obj_id)
        row = ext_cat[ext_cat["ID_NIRISS"] == obj_id]
        width = int(
            np.nanmax([row["XMAX"] - row["XMIN"], row["YMAX"] - row["YMIN"]]) // 2
            + border
        )
        crop = (
            slice(None),
            slice(
                int(row["Y"] - width),
                int(row["Y"] + width),
            ),
            slice(
                int(row["X"] - width),
                int(row["X"] + width),
            ),
        )
        # print(bgr[crop].shape)
        cimg = ColourImage(
            bgr[crop],
            transformation_kwargs={
                "transformation": RawAsinhTransform(astrovis.ManualInterval(-0.001, 2))
            },
        )
        ax_img = subfigs.ravel()[o].add_subplot(gs[0])
        ax_img.imshow(cimg, origin="lower")
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        ax_img.set_ylabel(
            rf"({chr(97+o)})\ \ ID: {obj_id};\quad$z={row["Z_NIRISS"].value[0]:.3f}$"
        )
        # continue
        # print(dir(gs[:, 0]))
        # plt.subplots_from_gridspec()
        # axs = gs[:,0].subplots(
        #     2,
        #     1,
        #     # figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_columnwidth / 1.1),
        #     # constrained_layout=True,
        #     # use_gridspec=True,
        #     # sharex="col",
        #     # sharey="row",
        #     # width_ratios=[1, 3, 1, 3],
        #     # height_ratios=[1, 1, 2],
        # )
        ax_z = subfigs.ravel()[o].add_subplot(gs[-1])
        # print(obj_id)
        oned_path = [
            *(grizli_dir).glob(f"Extractions*/**/*glass-a2744_{obj_id:0>5}.1D.fits")
        ]
        oned_path.sort(reverse=True)
        # print(oned_path)
        if obj_id == 3062:
            oned_path = oned_path[0]
        else:
            oned_path = oned_path[-1]
        max_ylim = 0.0

        # print(str(oned_path).replace("1D", "full"))

        with fits.open(str(oned_path).replace("1D", "full")) as hdul_all:
            hdul = hdul_all["ZFIT_STACK"]
            # try:
            c2min = hdul.data["chi2"].min()
            # print (hdul.data["pdf"])
            scale_nu = c2min / hdul.header["DOF"]
            ax_z.plot(
                hdul.data["zgrid"],
                # hdul.data["chi2"] / hdul.header["DOF"],
                # ((hdul.data['chi2']-c2min)/scale_nu)
                np.log10(hdul.data["pdf"]),
            )
            pzmax = np.log10(hdul.data["pdf"]).max()
            ax_z.set_ylim(pzmax - 6, pzmax + 0.9)
            ax_z.set_xlabel(r"$z_{\textsc{grizli}}$")
            from matplotlib.ticker import MultipleLocator

            ax_z.yaxis.set_major_locator(MultipleLocator(base=1))
            # ax_z.set_ylabel(r"$\log_{10}\left(p(z)\right)$")
            ax_z.set_ylabel(r"$\log_{10}\,p(z)$")
        # continue
        axs_oned = subfigs.ravel()[o].add_subplot(gs[1])
        with fits.open(oned_path) as oned_hdul:
            # oned_hdul.info()
            for i, hdu in enumerate(oned_hdul[1:]):
                oned_tab = Table(hdu.data)
                f = hdu.name
                # print(f)
                # # print (oned_tab.colnames)
                # print(len(oned_tab["wave"]))
                # axs[2, 2*i].remove()
                axs_oned.plot(
                    oned_tab["wave"] / 1e4,
                    oned_tab["flux"] / oned_tab["flat"] / 1e-19,
                    drawstyle="steps-mid",
                    c="#d55e00",
                    label="Data" if i == 2 else "_",
                )
                axs_oned.fill_between(
                    oned_tab["wave"] / 1e4,
                    (oned_tab["flux"] + oned_tab["err"]) / oned_tab["flat"] / 1e-19,
                    (oned_tab["flux"] - oned_tab["err"]) / oned_tab["flat"] / 1e-19,
                    # drawstyle="steps-mid",
                    color="#d55e00",
                    step="mid",
                    alpha=0.5,
                )
                axs_oned.plot(
                    oned_tab["wave"] / 1e4,
                    oned_tab["line"] / oned_tab["flat"] / 1e-19,
                    # drawstyle="steps-mid",
                    c="k",
                    label="Data" if i == 2 else "_",
                )
                # if np.nanmax(oned_tab["flux"] / oned_tab["flat"] / 1e-19) > max_ylim:
                # flux_test =
                # print (dir(oned_tab["flux"]))
                # print (np.nanmax([oned_tab["flux"] / oned_tab["flat"] / 1e-19]+[max_ylim]))
                max_ylim = np.nanmax(
                    (
                        (oned_tab["flux"] / oned_tab["flat"] / 1e-19)[
                            ((oned_tab["wave"] / 1e4) > niriss_filter_sens[f][0])
                            & ((oned_tab["wave"] / 1e4) < niriss_filter_sens[f][-1])
                        ]
                    ).tolist()
                    + [max_ylim]
                    + (
                        (oned_tab["contam"] / oned_tab["flat"] / 1e-19)[
                            ((oned_tab["wave"] / 1e4) > niriss_filter_sens[f][0])
                            & ((oned_tab["wave"] / 1e4) < niriss_filter_sens[f][-1])
                        ]
                    ).tolist()
                )
                axs_oned.plot(
                    oned_tab["wave"] / 1e4,
                    oned_tab["contam"] / oned_tab["flat"] / 1e-19,
                    c="#0072b2",
                    alpha=0.7,
                    drawstyle="steps-mid",
                    zorder=-1,
                    label="Contamination" if i == 2 else "_",
                )
                axs_oned.set_ylim(ymin=0)
                from matplotlib.ticker import MultipleLocator

                # if i == 0:
                #     axs_oned.yaxis.set_major_locator(MultipleLocator(2.0))
                #     axs_oned.set_yticklabels([])
                #     # print ((oned_tab["wave"][0] / 1e4 - 0.1,0))
                #     for s in np.arange(0, 12, 2):
                #         an = axs_oned.annotate(
                #             f"{int(s)}",
                #             (oned_tab["wave"][0] / 1e4 - 0.0125, s - 0.1),
                #             # xycoords
                #             annotation_clip=False,
                #             va="center",
                #             ha="right",
                #         )
                #         an.set_in_layout(False)
                # if i == 2:
                #     axs_oned.legend(fontsize=7, loc=1)
                # ylim = axs_oned.get_ylim()[1]
                # for l, v in line_dict.items():
                #     # print (v/1e4,  niriss_filter_sens[f][0])
                #     line_w = (1 + obj_z) * v / 1e4
                #     if (line_w >= niriss_filter_sens[f][0]) and (
                #         line_w < niriss_filter_sens[f][-1]
                #     ):
                #         axs_oned.axvline(line_w, c="k", linewidth=1, linestyle=":")
                #         if l != "OIII-4958":
                #             axs_oned.annotate(
                #                 rf"{l}",
                #                 xy=(
                #                     line_w + 0.005 - (0.005 if l == "OIII" else 0),
                #                     1.1,
                #                 ),
                #                 xycoords=axs_oned.get_xaxis_transform(),
                #                 # xytext=(0.855, ylim + 0.05),
                #                 # textcoords="data",
                #                 # arrowprops=dict(arrowstyle="-[,widthB=2.5", connectionstyle="arc3"),
                #                 ha="center",
                #                 va="center",
                #                 rotation=60,
                # )
        axs_oned.set_ylim(ymax=1.1 * max_ylim)
        # xlim = axs_oned.get_xlim()
        axs_oned.set_xlim(0.95, 2.3)
        # for k,v in niriss_filter_sens.items():
        # print (k, v)
        # test = "F115W"
        # print
        # continue
        # axs_oned.fill_betweenx(
        #     [-1, 100],
        #     niriss_filter_sens["F115W"][-1],
        #     niriss_filter_sens["F150W"][0],
        #     # oned_tab["wave"][-1] / 1e4,
        #     **hatch_kwargs,
        #     label="Sens. Cutoff" if i == 2 else "_",
        # )
        for cutoff_edges in [
            [axs_oned.get_xlim()[0], niriss_filter_sens["F115W"][0]],
            [
                niriss_filter_sens["F115W"][-1],
                niriss_filter_sens["F150W"][0],
            ],
            [
                niriss_filter_sens["F150W"][-1],
                niriss_filter_sens["F200W"][0],
            ],
            [niriss_filter_sens["F200W"][-1], axs_oned.get_xlim()[-1]],
        ]:
            axs_oned.fill_betweenx(
                [-1, 1.25 * max_ylim],
                *cutoff_edges,
                **hatch_kwargs,
            )
        axs_oned.set_ylabel(
            r"$f_{\lambda}\ [10^{-19}\,\rm{erg}/\rm{s}/\rm{cm}^{2}/\AA]$",
        )
        axs_oned.set_xlabel(r"Observed Wavelength ($\mu$m)")
        # axs_oned.fill_betweenx(
        #     [-1, 100],
        #     niriss_filter_sens["F115W"][-1],
        #     niriss_filter_sens["F150W"][0],
        #     # oned_tab["wave"][-1] / 1e4,
        #     **hatch_kwargs,
        #     label="Sens. Cutoff" if i == 2 else "_",
        # )
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
    # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 5], ax=ax, label="Placeholder")

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

    # fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "redshift_quality_examples.pdf", dpi=600)

    # # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0,
    # #                         wspace=0)
    # # fig.subplots_adjust(wspace=0.3, hspace=0.3)
    # plt.show()
    # # plt.savefig(save_dir / "svg" / "wavelength_coverage.svg", dpi=600)

    plt.show()
