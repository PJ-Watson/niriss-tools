"""
Show the visibility of lines in the NIRISS filters.
"""

import matplotlib.gridspec as gridspec
import plot_utils
from default_imports import *
from labellines import labelLine, labelLines

plot_utils.setup_aanda_style()

line_dict = {
    # "Ly$\alpha$": 1215.24,
    # r"$\ion{Mg}{ii}$": 2799.12,
    r"$\left[\ion{O}{ii}\right]$": 3728.4835,
    # r"$\left[\ion{O}{iii}\right]$": 4960.295,
    r"$\left[\ion{O}{iii}\right]$": 5008.24,
    # r"$\left[\ion{O}{iii}\right]$" : 4984.2675,
    # "Hbeta": 4862.68,
    r"$\ion{H}{\alpha}$": 6564.61,
    r"$\left[\ion{S}{iii}\right]$": 9533.2,
    r"$\ion{Pa}{\beta}$": 12821.7,
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

    # cat_names = ["grizli_photz_matched.fits"]

    # catalogue_path = catalogue_dir / "grizli_photz_matched.fits"

    v1_cat = Table.read(catalogue_dir / "compiled_catalogue_v1.fits")

    fig = plt.figure(
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth * 1.5),
        constrained_layout=True,
    )

    z_min = 0
    z_max = 8.55
    z_cut_low = 4.8
    z_cut_high = 7.25

    gs = fig.add_gridspec(2, 1, height_ratios=(1, 3))
    gs1 = gridspec.GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=gs[1],
        height_ratios=((z_max - z_cut_high) / ((z_cut_low - z_min) / 2), 2),
        hspace=0.025,
    )
    # gs = fig.add_gridspec(3, 1, height_ratios=(1, (z_max-z_cut_high)/((z_cut_low-z_min)/2)*0.8, 2))
    ax_top = fig.add_subplot(gs1[0, 0])
    ax_bottom = fig.add_subplot(gs1[1, 0], sharex=ax_top)

    z_range = np.linspace(z_min, z_max, 100)

    fill_kwargs = {
        "color": "gold",
    }

    line_set = [
        [r"$\ion{Pa}{\beta}$", r"$\left[\ion{S}{iii}\right]$", r"$\ion{H}{\alpha}$"],
        [
            r"$\left[\ion{S}{iii}\right]$",
            r"$\ion{H}{\alpha}$",
            r"$\left[\ion{O}{iii}\right]$",
        ],
        [
            r"$\ion{H}{\alpha}$",
            r"$\left[\ion{O}{iii}\right]$",
            r"$\left[\ion{O}{ii}\right]$",
        ],
    ]
    for a, b, c in line_set:
        print(a, b, c)

        high = np.nanmin(
            [
                niriss_info["F200W"][-1] / line_dict[a] - 1,
                niriss_info["F150W"][-1] / line_dict[b] - 1,
                niriss_info["F115W"][-1] / line_dict[c] - 1,
            ]
        )
        low = np.nanmax(
            [
                niriss_info["F200W"][0] / line_dict[a] - 1,
                niriss_info["F150W"][0] / line_dict[b] - 1,
                niriss_info["F115W"][0] / line_dict[c] - 1,
            ]
        )
        if [a, b, c] == [
            r"$\ion{H}{\alpha}$",
            r"$\left[\ion{O}{iii}\right]$",
            r"$\left[\ion{O}{ii}\right]$",
        ]:
            print("yes")
            ax_top.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)
            ax_bottom.fill_between(
                [0.95, 0.98], [low, low], [high, high], **fill_kwargs
            )

    line_set = [
        [r"$\ion{H}{\alpha}$", r"$\left[\ion{O}{iii}\right]$"],
        [r"$\left[\ion{O}{iii}\right]$", r"$\left[\ion{O}{ii}\right]$"],
    ]
    for a, b in line_set:
        print(a, b)

        high = np.nanmin(
            [
                niriss_info["F200W"][-1] / line_dict[a] - 1,
                niriss_info["F150W"][-1] / line_dict[b] - 1,
                # niriss_info["F115W"][-1]/line_dict[c] - 1,
            ]
        )
        low = np.nanmax(
            [
                niriss_info["F200W"][0] / line_dict[a] - 1,
                niriss_info["F150W"][0] / line_dict[b] - 1,
                # niriss_info["F115W"][0]/line_dict[c] - 1,
            ]
        )
        if [a, b] == [r"$\left[\ion{O}{iii}\right]$", r"$\left[\ion{O}{ii}\right]$"]:

            ax_top.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)
            ax_bottom.fill_between(
                [0.95, 0.98], [low, low], [high, high], **fill_kwargs
            )

    line_set = [
        [r"$\ion{Pa}{\beta}$", r"$\ion{H}{\alpha}$"],
    ]
    for a, b in line_set:
        print(a, b)

        high = np.nanmin(
            [
                niriss_info["F200W"][-1] / line_dict[a] - 1,
                niriss_info["F115W"][-1] / line_dict[b] - 1,
                # niriss_info["F115W"][-1]/line_dict[c] - 1,
            ]
        )
        low = np.nanmax(
            [
                niriss_info["F200W"][0] / line_dict[a] - 1,
                niriss_info["F115W"][0] / line_dict[b] - 1,
                # niriss_info["F115W"][0]/line_dict[c] - 1,
            ]
        )
        # if [a,b]==[r"$\left[\ion{O}{iii}\right]$",r"$\left[\ion{O}{ii}\right]$"]:
        #     print ("yes")

        ax_top.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)
        ax_bottom.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)

    line_set = [
        [r"$\ion{H}{\alpha}$", r"$\left[\ion{O}{iii}\right]$"],
        [r"$\ion{Pa}{\beta}$", r"$\ion{H}{\alpha}$"],
    ]
    for a, b in line_set:
        print(a, b)

        high = np.nanmin(
            [
                niriss_info["F150W"][-1] / line_dict[a] - 1,
                niriss_info["F115W"][-1] / line_dict[b] - 1,
                # niriss_info["F115W"][-1]/line_dict[c] - 1,
            ]
        )
        low = np.nanmax(
            [
                niriss_info["F150W"][0] / line_dict[a] - 1,
                niriss_info["F115W"][0] / line_dict[b] - 1,
                # niriss_info["F115W"][0]/line_dict[c] - 1,
            ]
        )
        # if [a,b]==[r"$\left[\ion{O}{iii}\right]$",r"$\left[\ion{O}{ii}\right]$"]:
        #     print ("yes")
        ax_top.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)
        ax_bottom.fill_between([0.95, 0.98], [low, low], [high, high], **fill_kwargs)
        print(low, high)
    # exit()
    # for k, v in line_dict.items():
    #     ax.plot(z_range, (1 + z_range) * v)

    # for k, v in niriss_info.items():
    #     ax.fill_between(z_range, v[0], v[1], color="k", alpha=0.4, edgecolor="none")
    for k, v in line_dict.items():
        ax_top.plot((1 + z_range) * v / 1e4, z_range, c="k", linewidth=0.5, label=k)
        ax_bottom.plot((1 + z_range) * v / 1e4, z_range, c="k", linewidth=0.5, label=k)
        # ax.annotate(k, (2.25, 2.3 / v * 1e4 - 1), ha="right", va="bottom")
        # print(2.3 / v * 1e4 - 1)
    uv_line_dict = {
        r"Ly$\alpha$": 1215.24,
        r"$\ion{Mg}{ii}$": 2799.12,
        # r"$\ion{C}{iv}$": 1549.48,
        # r"$\ion{He}{ii}$": 1640.4,
    }
    for k, v in uv_line_dict.items():
        ax_top.plot((1 + z_range) * v / 1e4, z_range, c="k", linewidth=0.5, label=k)
        ax_bottom.plot((1 + z_range) * v / 1e4, z_range, c="k", linewidth=0.5, label=k)

    abs_line_dict = {
        r"Balmer\ Break": 3645,
        r"4000\AA\ Break": 4000,
        r"$\ion{Ca}{ii}$": 8600,
        # r"TiO": 9400
    }
    for k, v in abs_line_dict.items():
        ax_top.plot(
            (1 + z_range) * v / 1e4,
            z_range,
            c="k",
            linestyle="-.",
            linewidth=0.5,
            label=k,
        )
        ax_bottom.plot(
            (1 + z_range) * v / 1e4,
            z_range,
            c="k",
            linestyle="-.",
            linewidth=0.5,
            label=k,
        )
        # ax.annotate(k, (1.05, 1.15 / v * 1e4 - 1), ha="left", va="bottom")

    # ax.plot((1 + z_range) *3728.4835 / 1e4, z_range, c="k", linewidth=0.5)
    # ax.annotate(
    #     r"$\left[\ion{O}{ii}\right]$",
    #     (1.05, 1.15 / 3728.4835 * 1e4 - 1),
    #     ha="left",
    #     va="bottom",
    # )

    for k, v in niriss_info.items():
        ax_top.fill_betweenx(
            z_range, v[0], v[1], color="k", alpha=0.2, edgecolor="none"
        )
        ax_bottom.fill_betweenx(
            z_range, v[0], v[1], color="k", alpha=0.2, edgecolor="none"
        )

    # print (ax.get_lines())
    lines = np.asarray(ax_bottom.get_lines())
    for idx, col, pos, yoff in zip(
        [[0, 1, 6], [2, 3, 8], [4, 5, 9], [7]],
        [(0.9, 0.9, 1.0), (0.9, 1, 0.9), (1.0, 0.9, 0.9), (1.0, 0.9, 0.9)],
        [1.125, 1.5, 2.0, 1.95],
        [0, 0, 0, 0.025],
    ):
        labelLines(
            lines[idx].tolist(),
            zorder=2.5,
            xvals=pos,
            # xvals=[1.125, 1.125, 1.5, 1.5, 2.0,2.0, 1.125],
            # backgroundcolor="none",
            # bbox={'pad': -1, 'color': "green"}
            # bbox={'pad': -3, 'color': 'white'}
            # outline_color="none"
            outline_color=col,
            outline_width=5,
            yoffsets=yoff,
        )
    labelLines(
        ax_top.get_lines(),
        zorder=2.5,
        xvals=1.1,
        outline_color=(0.9, 0.9, 1.0),
        outline_width=5,
    )

    ax_top.set_xlim(0.95, 2.300)
    ax_bottom.set_ylim(z_min, z_cut_low)
    ax_top.set_ylim(z_cut_high, z_max)
    # ax.axhline(1.1, c="k", linestyle=":")
    # ax.axhline(1.35, c="k", linestyle=":")
    # ax.axhline(1.9, c="k", linestyle=":")
    ax_bottom.axhline(0.3064, c="k", linestyle=":", label="Cluster")
    # hide the spines between ax and ax2
    ax_top.spines.bottom.set_visible(False)
    ax_bottom.spines.top.set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)  # don't put tick labels at the top
    ax_bottom.xaxis.tick_bottom()

    # hist = partial(plot_utils.plot_hist, bins=np.arange(16.5, 33.5, 0.5), ax=ax)
    # # print (np.nanmin(v1_cat["MAG_AUTO"]), np.nanmax(v1_cat["MAG_AUTO"])

    # hist(v1_cat["MAG_AUTO"], label="Full Sample")
    # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] > 0], ax=ax, label="Extracted")
    # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 4], ax=ax, label="First Pass")
    # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 5], ax=ax, label="Placeholder")

    ax_bottom.set_xlabel(r"Observed Wavelength ($\mu$m)")
    ax_bottom.set_ylabel(r"$z$")
    # ax_top.set_ylabel(r"$z$")
    # ax_bottom.set_xticks()
    # ax.legend()

    for k, v in line_dict.items():
        for k_n, v_n in niriss_info.items():
            # low = v_n[0]/v -1
            print(f"{k}: {k_n}={v_n[0]/v-1:.3f},{v_n[1]/v-1:.3f}")

    ax_filts = fig.add_subplot(gs[0], sharex=ax_top)
    plt.setp(ax_filts.get_xticklabels(), visible=False)
    # for f in filter_dir.glob("*.txt"):
    for c_1, c_2, f in zip(
        ["blue", "green", "red"],
        [(0.9, 0.9, 1.0), (0.9, 1, 0.9), (1.0, 0.9, 0.9)],
        niriss_info.keys(),
    ):
        print(f)
        filt_tab = Table.read(filter_dir / f"NIRISS_{f}.txt", format="ascii")
        # print (filt_tab)
        filt_tab["PCE"][filt_tab["PCE"] < 1e-3] = np.nan
        ax_filts.plot(filt_tab["Wavelength"], filt_tab["PCE"], c=c_1)

        ax_filts.annotate(
            rf"{f}",
            (np.nanmedian(niriss_info[f]) / 1e4, 0.3),
            ha="center",
            va="center",
            c=c_1,
        )
        # half_sens_idx = [
        #     np.nanargmax(filt_tab["PCE"] <= 0.5 * np.nanmax(filt_tab["PCE"])),
        #     np.nanargmax(filt_tab["PCE"] >= 0.5 * np.nanmax(filt_tab["PCE"]))
        # ]
        half_sens_idx = np.argwhere(filt_tab["PCE"] >= 0.5 * np.nanmax(filt_tab["PCE"]))
        #     np.nanargmax(filt_tab["PCE"] >= 0.5 * np.nanmax(filt_tab["PCE"]))
        # ]
        # print (half_sens_idx)

        ax_filts.fill_betweenx(
            [0, 1],
            filt_tab["Wavelength"][half_sens_idx[0]],
            filt_tab["Wavelength"][half_sens_idx[-1]],
            color=c_2,
            # alpha=0.1,
            edgecolor="none",
        )
        for a in [ax_bottom, ax_top]:
            a.fill_betweenx(
                z_range,
                filt_tab["Wavelength"][half_sens_idx[0]],
                filt_tab["Wavelength"][half_sens_idx[-1]],
                color=c_2,
                # alpha=0.1,
                edgecolor="none",
            )
        # print(
        #     filt_tab["Wavelength"][half_sens_idx[0]],
        #     filt_tab["Wavelength"][half_sens_idx[-1]],
        # )

    d = 0.5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=5,  # linewidth=1,
        linestyle="none",
        color="k",
        mec="k",
        mew=0.5,
        clip_on=False,
        in_layout=False,
    )
    ax_top.plot([0, 1], [0, 0], transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot([0, 1], [1, 1], transform=ax_bottom.transAxes, **kwargs)

    ax_filts.set_ylabel(r"Throughput")
    ax_filts.set_ylim(0, 1)

    fig.patch.set_alpha(0.0)

    ax_top.set_yticks([8.0], [8.0])
    ax_top.set_yticks(np.arange(7.4, 9, 0.2), minor=True)
    ax_bottom.set_yticks(np.arange(0, 4.8, 0.2), minor=True)
    ax_bottom.set_yticks(np.arange(0, 5), [0.0, 1.0, 2.0, 3.0, 4.0])
    ax_top.set_ylim(ymax=z_max)

    plt.savefig(save_dir / "wavelength_coverage_rev1.pdf", dpi=600)
    # plt.savefig(save_dir / "svg" / "wavelength_coverage.svg", dpi=600)

    plt.show()
