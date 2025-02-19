"""
Show galaxies in the magnitude-redshift plane.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

line_dict = {
    "Ly$\alpha$": 1215.24,
    "MgII": 2799.12,
    "OII": 3728.4835,
    # "OIII": 4960.295,
    # "OIII": 5008.24,
    "OIII": 4984.2675,
    "Hbeta": 4862.68,
    "Halpha": 6564.61,
}

niriss_info = {
    "F115W": [10130, 12830],
    "F150W": [13300, 16710],
    "F200W": [17510, 22260],
}

if __name__ == "__main__":

    # fig, ax = plt.subplots(
    fig = plt.figure(
        figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth / 1.62),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(2, 2, width_ratios=(3, 1), height_ratios=(1, 2))
    ax_scatter = fig.add_subplot(gs[1, 0])

    secure = full_cat["Z_FLAG_ALL"] >= 9
    tentative = (full_cat["Z_FLAG_ALL"] == 7) | (full_cat["Z_FLAG_ALL"] == 6)

    z_bins = np.arange(0, 8, 0.05)

    ax_scatter.scatter(
        full_cat["Z_EST_ALL"][secure],
        full_cat["MAG_AUTO"][secure],
        # color="k",
        alpha=0.7,
        # edgecolor="none",
        color="dodgerblue",
        edgecolor="blue",
        s=8,
        label="Secure",
    )
    ax_scatter.scatter(
        full_cat["Z_EST_ALL"][tentative],
        full_cat["MAG_AUTO"][tentative],
        color="k",
        facecolor="none",
        marker="o",
        alpha=0.7,
        s=8,
        linewidth=0.5,
        label="Tentative",
    )
    ax_scatter.set_xlim(z_bins[0], z_bins[-1])
    y_lims = ax_scatter.get_ylim()

    # for i, l in enumerate(h_alpha_lims):
    #     ax.fill_betweenx(y_lims, *l, color="r", alpha=0.15, edgecolor="none", label=r"H$\alpha$" if i==0 else None)
    # for i, l in enumerate(OIII_lims):
    #     ax.fill_betweenx(y_lims, *l, color="g", alpha=0.15, edgecolor="none", label=r"[O\,\textsc{iii}]" if i==0 else None)
    # for i, l in enumerate(OII_lims):
    #     ax.fill_betweenx(y_lims, *l, color="b", alpha=0.15, edgecolor="none", label=r"[O\,\textsc{ii}]" if i==0 else None)

    # ax.set_ylim(y_lims)

    # # cat_names = ["grizli_photz_matched.fits"]

    # # catalogue_path = catalogue_dir / "grizli_photz_matched.fits"

    # v1_cat = Table.read(catalogue_dir / "compiled_catalogue_v1.fits")

    # z_min = 0
    # z_max = 3
    # z_range = np.linspace(z_min, z_max, 100)

    # for k, v in line_dict.items():
    #     ax.plot(z_range, (1 + z_range) * v)

    # for k, v in niriss_info.items():
    #     ax.fill_between(z_range, v[0], v[1], color="k", alpha=0.4, edgecolor="none")

    # ax.set_ylim(9000, 23500)
    # ax.set_xlim(z_min, z_max)
    # ax.axvline(1.34)
    # ax.axvline(1.98)

    # # hist = partial(plot_utils.plot_hist, bins=np.arange(16.5, 33.5, 0.5), ax=ax)
    # # # print (np.nanmin(v1_cat["MAG_AUTO"]), np.nanmax(v1_cat["MAG_AUTO"])

    # # hist(v1_cat["MAG_AUTO"], label="Full Sample")
    # # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] > 0], ax=ax, label="Extracted")
    # # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 4], ax=ax, label="First Pass")
    # # hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 5], ax=ax, label="Placeholder")
    print(full_cat.colnames)

    ax_scatter.set_ylabel(r"$m_{\rm{F200W}}$")
    ax_scatter.set_xlabel(r"$z$")
    ax_scatter.legend(loc=4)

    ax_z_hist = fig.add_subplot(gs[0, 0], sharex=ax_scatter)
    plt.setp(ax_z_hist.get_xticklabels(), visible=False)

    print(len(full_cat["Z_EST_ALL"][secure & (full_cat["Z_EST_ALL"] >= 2)]))

    z_secure = np.array(full_cat["Z_EST_ALL"][secure])
    z_tentative = np.array(full_cat["Z_EST_ALL"][tentative])

    _, _, barlist = ax_z_hist.hist(
        [z_secure, z_tentative],
        bins=z_bins,
        histtype="stepfilled",
        color=["dodgerblue", "none"],
        edgecolor=["blue", "k"],
        linewidth=1,
        zorder=2,
        stacked=True,
    )
    # ax_z_hist.hist(
    #     np.concatenate([z_secure, z_tentative]),
    #     bins=z_bins,
    #     histtype="step",
    #     ec="k",
    #     # ls="--",
    #     linewidth=1,
    #     zorder=1,
    # )

    # ax_z_hist.hist(
    #     [full_cat["Z_EST_ALL"][secure],full_cat["Z_EST_ALL"][tentative]],
    #     # full_cat["MAG_AUTO"][secure],
    #     # color="purple",
    #     # alpha=.7,
    #     # edgecolor="none",
    #     # s=7,
    #     bins=z_bins,
    #     stacked=True
    # )
    ax_z_hist.set_ylabel(r"Number of Sources")

    ax_mag_hist = fig.add_subplot(gs[1, 1], sharey=ax_scatter)
    plt.setp(ax_mag_hist.get_yticklabels(), visible=False)

    mag_secure = np.array(full_cat["MAG_AUTO"][secure])
    mag_tentative = np.array(full_cat["MAG_AUTO"][tentative])
    print(len(mag_secure), len(mag_tentative))

    mag_bins = np.arange(15, 30, 0.5)
    _, _, barlist = ax_mag_hist.hist(
        [mag_secure, mag_tentative],
        bins=mag_bins,
        histtype="stepfilled",
        color=["dodgerblue", "none"],
        edgecolor=["blue", "k"],
        linewidth=1,
        zorder=2,
        orientation="horizontal",
        stacked=True,
    )
    # ax_mag_hist.hist(
    #     np.concatenate([mag_secure, mag_tentative]),
    #     bins=mag_bins,
    #     histtype="step",
    #     ec="k",
    #     # ls="--",
    #     linewidth=1,
    #     zorder=1,
    #     orientation="horizontal",
    # )

    # ax_mag_hist.hist(
    #     [full_cat["MAG_AUTO"][secure], full_cat["MAG_AUTO"][tentative]],
    #     # color="purple",
    #     # alpha=.7,
    #     # edgecolor="none",
    #     # s=7,
    #     bins=np.arange(15, 30, 0.5),
    #     orientation="horizontal",
    #     stacked=True,
    # )
    ax_mag_hist.set_xlabel(r"Number of Sources")

    fig.patch.set_alpha(0.0)

    # plt.savefig(save_dir / "mag_z_scatter.pdf")
    # plt.savefig(save_dir / "svg" / "mag_z_scatter.svg")
    # for k, v in line_dict.items():
    #     for k_n, v_n in niriss_info.items():
    #         # low = v_n[0]/v -1
    #         print(f"{k}: {k_n}={v_n[0]/v-1:.3f},{v_n[1]/v-1:.3f}")

    plt.show()
