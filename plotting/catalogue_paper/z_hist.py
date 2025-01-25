"""
Show the distribution of NIRISS redshifts.
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

    v2_out_dir = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "classification-stage-2"
        / "catalogues"
        / "compiled"
    )

    new_cat_name = "internal_full_1.fits"
    v2_cat = Table.read(v2_out_dir / new_cat_name)

    fig, ax = plt.subplots(
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.62),
        constrained_layout=True,
    )

    secure = v2_cat["Z_FLAG_ALL"] >= 9
    tentative = (v2_cat["Z_FLAG_ALL"] == 7) | (v2_cat["Z_FLAG_ALL"] == 6)

    z_bins = np.arange(0, 4, 0.05)

    ax.hist(
        v2_cat["Z_EST_ALL"][secure],
        # v2_cat["MAG_AUTO"][secure],
        color="purple",
        # alpha=.7,
        # edgecolor="none",
        # s=7,
        bins=z_bins,
    )
    # ax.scatter(
    #     v2_cat["Z_EST_ALL"][tentative],
    #     v2_cat["MAG_AUTO"][tentative],
    #     color="k",
    #     facecolor="none",
    #     marker="o",
    #     alpha=.7,
    #     s=4,
    #     linewidth=0.5,
    # )
    ax.set_xlim(z_bins[0], z_bins[-1])
    # y_lims = ax.get_ylim()

    # h_alpha_lims = [
    #     [0.543, 0.954],
    #     [1.026, 1.545],
    #     [1.667, 2.391],
    # ]
    # OIII_lims = [
    #     [1.032, 1.574],
    #     [1.668, 2.353],
    #     [2.513, 3.466],
    # ]
    # OII_lims = [
    #     [1.717,2.441],
    #     [2.567,3.482],
    #     [3.696,4.970],
    # ]
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

    # ax.set_ylabel(r"$m_{\rm{F200W}}$")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"Number of Objects")
    # ax.legend()

    plt.savefig(save_dir / "z_hist.pdf")
    # for k, v in line_dict.items():
    #     for k_n, v_n in niriss_info.items():
    #         # low = v_n[0]/v -1
    #         print(f"{k}: {k_n}={v_n[0]/v-1:.3f},{v_n[1]/v-1:.3f}")

    plt.show()
