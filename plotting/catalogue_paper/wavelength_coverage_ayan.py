"""
Show the visibility of lines in the NIRISS filters.
"""

import plot_utils
from default_imports import *
from labellines import labelLine, labelLines

plot_utils.setup_aanda_style()

line_dict = {
    # "Ly$\alpha$": 1215.24,
    # r"$\ion{Mg}{ii}$": 2799.12,
    r"$\left[\ion{O}{ii}\right]$": 3728.4835,
    r"$\left[\ion{Ne}{iii}\right]$": 3869,
    r"$\ion{H}{\beta}$": 4861,
    r"$\left[\ion{O}{iii}\right]$": 5008.24,
    # r"$\left[\ion{O}{iii}\right]$" : 4984.2675,
    # "Hbeta": 4862.68,
    r"$\ion{H}{\alpha}$": 6564.61,
    r"$\left[\ion{S}{ii}\right]$": 6731,
    # r"$\left[\ion{S}{iii}\right]$": 9533.2,
    # r"$\ion{Pa}{\beta}$": 12821.7,
}

niriss_info = {
    "F115W": [10130, 12830],
    "F150W": [13300, 16710],
    "F200W": [17510, 22260],
}

filter_dir = root_dir / "archival" / "JWST" / "niriss_filter_tarball" / "DATA"

if __name__ == "__main__":

    catalogue_dir = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "glass_niriss"
        / "match_catalogues"
        / "classification_v1"
    )

    poss_wavelengths = np.arange(1e4, 3e4)
    niriss_coverage = np.zeros_like(poss_wavelengths)
    for k, v in niriss_info.items():
        niriss_coverage[(poss_wavelengths >= v[0]) & (poss_wavelengths <= v[-1])] = 1

    poss_wavelengths = poss_wavelengths[niriss_coverage > 0]
    # plt.plot(poss_wavelengths, niriss_coverage)
    # plt.show()
    # exit()
    # cat_names = ["grizli_photz_matched.fits"]

    # catalogue_path = catalogue_dir / "grizli_photz_matched.fits"

    v1_cat = Table.read(catalogue_dir / "compiled_catalogue_v1.fits")

    fig = plt.figure(
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.0),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(2, 1, height_ratios=(1, 2))
    ax = fig.add_subplot(gs[1, 0])

    z_min = 1.0
    z_max = 4
    z_range = np.linspace(z_min, z_max, 100)

    fill_kwargs = {
        # "color": "gold",
        "alpha": 0.5
    }

    line_sets = [
        # [r"$\ion{Pa}{\beta}$", r"$\left[\ion{S}{iii}\right]$", r"$\ion{H}{\alpha}$"],
        # [
        #     r"$\left[\ion{S}{iii}\right]$",
        #     r"$\ion{H}{\alpha}$",
        #     r"$\left[\ion{O}{iii}\right]$",
        # ],
        # [
        #     r"$\ion{H}{\alpha}$",
        #     r"$\left[\ion{O}{iii}\right]$",
        #     r"$\left[\ion{O}{ii}\right]$",
        # ],
        [
            r"$\left[\ion{O}{ii}\right]$",
            r"$\left[\ion{Ne}{iii}\right]$",
            r"$\ion{H}{\beta}$",
            r"$\left[\ion{O}{iii}\right]$",
        ],
        [
            r"$\left[\ion{O}{ii}\right]$",
            r"$\left[\ion{Ne}{iii}\right]$",
            r"$\ion{H}{\beta}$",
            r"$\left[\ion{O}{iii}\right]$",
            r"$\ion{H}{\alpha}$",
            # r"$\left[\ion{S}{ii}\right]$",
        ],
    ]
    for line_set, colour, x_lims in zip(
        line_sets, ["C0", "C1"], [[0.93, 0.96], [0.96, 0.99]]
    ):
        # print(a, b, c)
        redshifts = np.arange(0, 5, 0.001)
        redshift_availability = np.zeros((len(line_set), len(redshifts)), dtype=bool)
        for k, v in niriss_info.items():
            # poss_wavelengths = np.arange(1e4, 3e4)
            # niriss_coverage = np.zeros_like(poss_wavelengths)
            # niriss_coverage[(poss_wavelengths >= v[0]) & (poss_wavelengths <= v[-1])] = 1
            redshift_max = redshift_min = 0
            for i, l in enumerate(line_set):
                shifted_lam = (1 + redshifts) * line_dict[l]
                # print (shifted_lam)
                # # redshift_min = np.nanmin()
                # print (shifted_lam>=v[0])
                # print (redshifts[(shifted_lam>=v[0]).nonzero()][0])
                # redshift_min = np.nanmax(redshifts[(shifted_lam>=v[0])][0])
                # redshift_max = np.nanmin(redshifts[(shifted_lam<=v[-1])][-1])
                redshift_availability[i] |= (shifted_lam >= v[0]) & (
                    shifted_lam <= v[-1]
                )
            # print (redshift_min, redshift_max)
        redshift_availability = np.bitwise_and.reduce(redshift_availability, axis=0)
        # ax.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)
        # poss_wavelengths = poss_wavelengths[niriss_coverage>0]
        # plt.imshow(redshift_availability, aspect="auto", interpolation="none")
        # plt.plot(redshift_availability)
        # plt.show()
        # lam_coverage = np.zeros_like(poss_wavelengths)
        # redshifts = np.arange(0,5,0.001)
        # idx_all = np.arange(len(poss_wavelengths))
        # for l in line_set:
        #     shifted_lam = (1+redshifts)*line_dict[l]
        #     # idx_vis = np.clip(np.searchsorted(poss_wavelengths, shifted_lam), a_min=0, a_max=len(poss_wavelengths)-1)
        #     print (idx_vis)
        #     idx_all = [i for i in idx_all if i in idx_vis]
        #     plt.scatter(poss_wavelengths[idx_vis], (1+redshifts)*line_dict[l])
        # plt.scatter(poss_wavelengths[idx_vis], np.ones_like(idx_vis))
        # plt.show()
        import itertools
        import operator

        visible_idxs = [
            [i for i, value in it]
            for key, it in itertools.groupby(
                enumerate(redshift_availability), key=operator.itemgetter(1)
            )
            if key != 0
        ]
        for vis in visible_idxs:
            # print(redshifts[vis[0]], redshifts[vis[-1]])
            # ax.fill_between(
            #     x_lims,
            #     [redshifts[vis[0]], redshifts[vis[0]]],
            #     [redshifts[vis[-1]], redshifts[vis[-1]]],
            #     color=colour,
            #     **fill_kwargs,
            # )
            # ax.fill_between(
            #     [2.25, 2.28],
            #     [redshifts[vis[0]], redshifts[vis[0]]],
            #     [redshifts[vis[-1]], redshifts[vis[-1]]],
            #     color=colour,
            #     **fill_kwargs,
            # )
            ax.fill_between(
                [0.93, 2.3],
                [redshifts[vis[0]], redshifts[vis[0]]],
                [redshifts[vis[-1]], redshifts[vis[-1]]],
                color=colour,
                **fill_kwargs,
            )
        # exit()
        # high = np.nanmin(
        #     [
        #         niriss_info["F200W"][-1] / line_dict[a] - 1,
        #         niriss_info["F150W"][-1] / line_dict[b] - 1,
        #         niriss_info["F115W"][-1] / line_dict[c] - 1,
        #     ]
        # )
        # low = np.nanmax(
        #     [
        #         niriss_info["F200W"][0] / line_dict[a] - 1,
        #         niriss_info["F150W"][0] / line_dict[b] - 1,
        #         niriss_info["F115W"][0] / line_dict[c] - 1,
        #     ]
        # )
        # if [a, b, c] == [
        #     r"$\ion{H}{\alpha}$",
        #     r"$\left[\ion{O}{iii}\right]$",
        #     r"$\left[\ion{O}{ii}\right]$",
        # ]:
        #     print("yes")
        #     ax.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)

    # line_set = [
    #     [r"$\ion{H}{\alpha}$", r"$\left[\ion{O}{iii}\right]$"],
    #     [r"$\left[\ion{O}{iii}\right]$", r"$\left[\ion{O}{ii}\right]$"],
    # ]
    # for a, b in line_set:
    #     print(a, b)

    #     high = np.nanmin(
    #         [
    #             niriss_info["F200W"][-1] / line_dict[a] - 1,
    #             niriss_info["F150W"][-1] / line_dict[b] - 1,
    #             # niriss_info["F115W"][-1]/line_dict[c] - 1,
    #         ]
    #     )
    #     low = np.nanmax(
    #         [
    #             niriss_info["F200W"][0] / line_dict[a] - 1,
    #             niriss_info["F150W"][0] / line_dict[b] - 1,
    #             # niriss_info["F115W"][0]/line_dict[c] - 1,
    #         ]
    #     )
    #     if [a, b] == [r"$\left[\ion{O}{iii}\right]$", r"$\left[\ion{O}{ii}\right]$"]:
    #         #     print ("yes")
    #         ax.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)

    # line_set = [
    #     [r"$\ion{Pa}{\beta}$", r"$\ion{H}{\alpha}$"],
    # ]
    # for a, b in line_set:
    #     print(a, b)

    #     high = np.nanmin(
    #         [
    #             niriss_info["F200W"][-1] / line_dict[a] - 1,
    #             niriss_info["F115W"][-1] / line_dict[b] - 1,
    #             # niriss_info["F115W"][-1]/line_dict[c] - 1,
    #         ]
    #     )
    #     low = np.nanmax(
    #         [
    #             niriss_info["F200W"][0] / line_dict[a] - 1,
    #             niriss_info["F115W"][0] / line_dict[b] - 1,
    #             # niriss_info["F115W"][0]/line_dict[c] - 1,
    #         ]
    #     )
    #     # if [a,b]==[r"$\left[\ion{O}{iii}\right]$",r"$\left[\ion{O}{ii}\right]$"]:
    #     #     print ("yes")
    #     ax.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)
    #     print(low, high)

    # line_set = [
    #     [r"$\ion{H}{\alpha}$", r"$\left[\ion{O}{iii}\right]$"],
    #     # [r"$\ion{Pa}{\beta}$", r"$\ion{H}{\alpha}$"],
    # ]
    # for a, b in line_set:
    #     print(a, b)

    #     high = np.nanmin(
    #         [
    #             niriss_info["F150W"][-1] / line_dict[a] - 1,
    #             niriss_info["F115W"][-1] / line_dict[b] - 1,
    #             # niriss_info["F115W"][-1]/line_dict[c] - 1,
    #         ]
    #     )
    #     low = np.nanmax(
    #         [
    #             niriss_info["F150W"][0] / line_dict[a] - 1,
    #             niriss_info["F115W"][0] / line_dict[b] - 1,
    #             # niriss_info["F115W"][0]/line_dict[c] - 1,
    #         ]
    #     )
    #     # if [a,b]==[r"$\left[\ion{O}{iii}\right]$",r"$\left[\ion{O}{ii}\right]$"]:
    #     #     print ("yes")
    #     ax.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)
    #     print(low, high)
    # exit()
    # for k, v in line_dict.items():
    #     ax.plot(z_range, (1 + z_range) * v)

    # for k, v in niriss_info.items():
    #     ax.fill_between(z_range, v[0], v[1], color="k", alpha=0.4, edgecolor="none")

    line_dict = {
        # "Ly$\alpha$": 1215.24,
        # r"$\ion{Mg}{ii}$": 2799.12,
        r"$\left[\ion{O}{ii}\right]$": 3728.4835,
        r"$\left[\ion{Ne}{iii}\right]$": 3869,
        r"$\ion{H}{\beta}$": 4861,
        # r"$\left[\ion{O}{iii}\right]$": 5008.24,
        # r"$\left[\ion{O}{iii}\right]$" : 4984.2675,
        # "Hbeta": 4862.68,
        r"$\ion{H}{\alpha}$": 6564.61,
        # r"$\left[\ion{S}{ii}\right]$": 6717,
        # r"$\left[\ion{S}{iii}\right]$": 9533.2,
        # r"$\ion{Pa}{\beta}$": 12821.7,
    }
    for k, v in line_dict.items():
        ax.plot((1 + z_range) * v / 1e4, z_range, c="k", linewidth=0.5, label=k)
        # ax.annotate(k, (2.25, 2.3 / v * 1e4 - 1), ha="right", va="bottom")
        print(2.3 / v * 1e4 - 1)
    lower_line_dict = {
        # "Ly$\alpha$": 1215.24,
        # r"$\ion{Mg}{ii}$": 2799.12,
        # r"$\ion{Mg}{ii}$": 2799.12,
        # r"$\ion{Mg}{ii}$": 2799.12,
        # r"$\left[\ion{Ne}{iii}\right]$": 3869,
        # r"$\ion{H}{\beta}$": 4861,
        r"$\left[\ion{O}{iii}\right]$": 5008.24,
        # r"$\left[\ion{O}{iii}\right]$" : 4984.2675,
        # "Hbeta": 4862.68,
        # r"$\ion{H}{\alpha}$": 6564.61,
        r"$\left[\ion{S}{ii}\right]$": 6717,
    }
    for k, v in lower_line_dict.items():
        ax.plot((1 + z_range) * v / 1e4, z_range, c="k", linewidth=0.5, label=k)
        # ax.annotate(k, (1.05, 1.15 / v * 1e4 - 1), ha="left", va="bottom")

    # ax.plot((1 + z_range) *3728.4835 / 1e4, z_range, c="k", linewidth=0.5)
    # ax.annotate(
    #     r"$\left[\ion{O}{ii}\right]$",
    #     (1.05, 1.15 / 3728.4835 * 1e4 - 1),
    #     ha="left",
    #     va="bottom",
    # )

    lines = np.asarray(ax.get_lines())
    for idx, col, pos, out_width, y_off in zip(
        np.arange(6, dtype=int),
        ["#b38d68ff", "#80a8c4ff", "#80a8c4ff", "#b38d68ff", "#b38d68ff", "#e6e6e6ff"],
        [
            1.125,
            1.5,
            1.95,
            2.0,
            1.5,
            1.55,
        ],
        [3, 3.5, 3.5, 3.0, 3.0, 2.3],
        [0.01, 0, 0.01, 0.01, -0.03, -0.05],
    ):
        labelLines(
            [lines[idx]],
            zorder=2.5,
            xvals=pos,
            # xvals=[1.125, 1.125, 1.5, 1.5, 2.0,2.0, 1.125],
            # backgroundcolor="none",
            # bbox={'pad': -1, 'color': "green"}
            # bbox={'pad': -3, 'color': 'white'}
            # outline_color="none"
            outline_color=col,
            outline_width=out_width,
            yoffsets=y_off,
        )
    # labelLines(
    #     ax.get_lines(),
    #     zorder=2.5,
    #     xvals = 1.1,
    #     outline_color=(0.9,0.9,1.0),
    #     outline_width=5,
    # )

    for k, v in niriss_info.items():
        ax.fill_betweenx(z_range, v[0], v[1], color="k", alpha=0.2, edgecolor="none")

    ax.set_xlim(0.93, 2.300)
    ax.set_ylim(z_min, z_max)
    # ax.axhline(1.1, c="k", linestyle=":")
    # ax.axhline(1.35, c="k", linestyle=":")
    # ax.axhline(1.9, c="k", linestyle=":")
    ax.axhline(0.3064, c="k", linestyle=":", label="Cluster")

    # hist = partial(plot_utils.plot_hist, bins=np.arange(16.5, 33.5, 0.5), ax=ax)
    # # print (np.nanmin(v1_cat["MAG_AUTO"]), np.nanmax(v1_cat["MAG_AUTO"])

    # hist(v1_cat["MAG_AUTO"], label="Full Sample")
    # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] > 0], ax=ax, label="Extracted")
    # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 4], ax=ax, label="First Pass")
    # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 5], ax=ax, label="Placeholder")

    ax.set_xlabel(r"Observed Wavelength ($\mu$m)")
    ax.set_ylabel(r"$z$")
    # ax.legend()

    for k, v in line_dict.items():
        for k_n, v_n in niriss_info.items():
            # low = v_n[0]/v -1
            print(f"{k}: {k_n}={v_n[0]/v-1:.3f},{v_n[1]/v-1:.3f}")

    ax_filts = fig.add_subplot(gs[0], sharex=ax)
    plt.setp(ax_filts.get_xticklabels(), visible=False)
    # for f in filter_dir.glob("*.txt"):
    for c, f in zip(["blue", "green", "red"], niriss_info.keys()):
        print(f)
        filt_tab = Table.read(filter_dir / f"NIRISS_{f}.txt", format="ascii")
        # print (filt_tab)
        filt_tab["PCE"][filt_tab["PCE"] < 1e-3] = np.nan
        ax_filts.plot(filt_tab["Wavelength"], filt_tab["PCE"], c=c)

        ax_filts.annotate(
            rf"{f}",
            (np.nanmedian(niriss_info[f]) / 1e4, 0.3),
            ha="center",
            va="center",
            c=c,
        )
        # half_sens_idx = [
        #     np.nanargmax(filt_tab["PCE"] <= 0.5 * np.nanmax(filt_tab["PCE"])),
        #     np.nanargmax(filt_tab["PCE"] >= 0.5 * np.nanmax(filt_tab["PCE"]))
        # ]
        half_sens_idx = np.argwhere(filt_tab["PCE"] >= 0.5 * np.nanmax(filt_tab["PCE"]))
        #     np.nanargmax(filt_tab["PCE"] >= 0.5 * np.nanmax(filt_tab["PCE"]))
        # ]
        # print (half_sens_idx)
        shaded_c = "k"
        ax_filts.fill_betweenx(
            [0, 1],
            filt_tab["Wavelength"][half_sens_idx[0]],
            filt_tab["Wavelength"][half_sens_idx[-1]],
            color=shaded_c,
            alpha=0.1,
            edgecolor="none",
        )
        ax.fill_betweenx(
            z_range,
            filt_tab["Wavelength"][half_sens_idx[0]],
            filt_tab["Wavelength"][half_sens_idx[-1]],
            color=shaded_c,
            alpha=0.1,
            edgecolor="none",
        )
        # print(
        #     filt_tab["Wavelength"][half_sens_idx[0]],
        #     filt_tab["Wavelength"][half_sens_idx[-1]],
        # )
    ax_filts.set_ylabel(r"Throughput")
    ax_filts.set_ylim(0, 1)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "wavelength_coverage_Ayan.pdf", dpi=600)
    plt.savefig(save_dir / "svg" / "wavelength_coverage_Ayan.svg", dpi=600)

    # plt.show()
