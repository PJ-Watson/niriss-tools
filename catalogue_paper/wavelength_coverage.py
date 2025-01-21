"""
Show the visibility of lines in the NIRISS filters.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

line_dict = {
    "Ly$\alpha$": 1215.24,
    # "MgII": 2799.12,
    "OII": 3728.4835,
    # "OIII": 4960.295,
    "OIII": 5008.24,
    # "OIII" : 4984.2675,
    # "Hbeta": 4862.68,
    "Halpha": 6564.61,
    "SIII": 9533.2,
    "PaB": 12821.7,
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
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.2),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(2, 1, height_ratios=(1, 2))
    ax = fig.add_subplot(gs[1, 0])

    z_min = 0
    z_max = 3
    z_range = np.linspace(z_min, z_max, 100)

    # for k, v in line_dict.items():
    #     ax.plot(z_range, (1 + z_range) * v)

    # for k, v in niriss_info.items():
    #     ax.fill_between(z_range, v[0], v[1], color="k", alpha=0.4, edgecolor="none")
    for k, v in line_dict.items():
        ax.plot((1 + z_range) * v / 1e4, z_range, c="k", linewidth=0.5)

    for k, v in niriss_info.items():
        ax.fill_betweenx(z_range, v[0], v[1], color="k", alpha=0.2, edgecolor="none")

    ax.set_xlim(0.95, 2.300)
    ax.set_ylim(z_min, z_max)
    ax.axhline(1.1, c="k", linestyle=":")
    ax.axhline(1.35, c="k", linestyle=":")
    ax.axhline(1.9, c="k", linestyle=":")
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
            color=c,
            alpha=0.2,
            edgecolor="none",
        )
        ax.fill_betweenx(
            z_range,
            filt_tab["Wavelength"][half_sens_idx[0]],
            filt_tab["Wavelength"][half_sens_idx[-1]],
            color=c,
            alpha=0.2,
            edgecolor="none",
        )
        # print(
        #     filt_tab["Wavelength"][half_sens_idx[0]],
        #     filt_tab["Wavelength"][half_sens_idx[-1]],
        # )
    ax_filts.set_ylabel(r"Throughput")
    ax_filts.set_ylim(0, 1)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "wavelength_coverage.pdf", dpi=600)
    plt.savefig(save_dir / "svg" / "wavelength_coverage.svg", dpi=600)

    plt.show()
