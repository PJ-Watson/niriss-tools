"""
Compare the NIRISS redshifts to existing spectroscopic redshifts.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    fig, axs = plt.subplots(
        2,
        1,
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.1),
        constrained_layout=True,
        sharex="col",
        sharey="row",
        # hspace=0.,
        # wspace=0.,
    )
    fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    # secure = full_cat["Z_FLAG_ALL"] >= 9
    # tentative = (full_cat["Z_FLAG_ALL"] == 7) | (full_cat["Z_FLAG_ALL"] == 6)

    secure = (full_cat["Z_FLAG"] >= 4) & np.isfinite(full_cat["Z_NIRISS"])
    tentative = (full_cat["Z_FLAG"] == 3) & np.isfinite(full_cat["Z_NIRISS"])

    spec = np.zeros(len(full_cat), dtype=bool)
    # spec[np.isfinite(full_cat["zspec"])] = True
    spec[np.isfinite(full_cat["zmed_prev"])] = True
    phot = np.zeros(len(full_cat), dtype=bool)
    # phot[np.isfinite(full_cat["zphot"]) & ~spec] = True
    phot[
        np.isfinite(full_cat["zphot_astrodeep"])
        & (~spec)
        & (full_cat["flag_astrodeep"] != 100028)
    ] = True
    phot[
        np.isfinite(full_cat["z_phot_megascience"])
        & (~spec)
        # & (full_cat["flag_astrodeep"] !=)
    ] = True
    # print (np.nansum(phot))
    no_z = np.zeros(len(full_cat), dtype=bool)
    no_z[~phot & ~spec] = True
    # phot = (np.isfinite(full_cat["zphot"])) & (~(np.isfinite(full_cat["zspec"]) & full_cat["zspec"].mask))
    print(np.sum(spec), np.sum(phot))

    # print (full_cat["flag_astrodeep"][5:10])

    print(
        np.sum(secure),
        np.sum(secure & spec),
        np.sum(secure & phot),
        np.sum(secure & no_z),
    )
    print(
        np.sum(tentative),
        np.sum(tentative & spec),
        np.sum(tentative & phot),
        np.sum(tentative & no_z),
    )
    print(
        np.sum(secure | tentative),
        np.sum((secure | tentative) & spec),
        np.sum((secure | tentative) & phot),
        np.sum((secure | tentative) & no_z),
    )

    print()

    # exit()

    # z_range = np.linspace(
    #     0, np.nanmax([full_cat["zphot"][secure], full_cat["Z_EST_ALL"][secure]]) * 1.05, 2
    # )
    z_range = np.linspace(0, 8.5, 2)
    print(z_range)
    for a in axs.flatten()[0:1]:
        a.plot(z_range, z_range, c="k", linestyle=":", linewidth=1, zorder=-1)
    for a in axs.flatten()[1:]:
        a.axhline(0, c="k", linestyle=":", linewidth=1, zorder=-1)

    # for i, (z_name, z_cat_name, idx) in enumerate(
    #     zip(
    #         [r"$z_{\rm{spec}}$", r"$z_{\rm{phot}}$"],
    #         ["zmed_prev", "zphot_astrodeep"],
    #         [spec, phot],
    #     )
    # ):
    z_name = r"$z_{\rm{phot}}$"
    z_cat_name = "z_phot_megascience"
    idx = phot
    offset = np.abs(full_cat["Z_NIRISS"] - full_cat[z_cat_name]) > 1
    dz = (full_cat[z_cat_name] - full_cat["Z_NIRISS"]) / (1 + full_cat[z_cat_name])
    axs[1].scatter(
        full_cat[z_cat_name][secure & idx & ~offset],
        dz[secure & idx & ~offset],
        # full_cat["MAG_AUTO"][secure],
        color="purple",
        alpha=0.7,
        # edgecolor="none",
        s=7,
        # bins=z_bins,
        label=z_name,
    )
    # axs[0,i].scatter(
    #     full_cat[z_cat_name][secure & idx & offset],
    #     ((full_cat[z_cat_name]-full_cat["Z_NIRISS"])/(1+full_cat[z_cat_name]))[secure & idx & offset],
    #     # full_cat["MAG_AUTO"][secure],
    #     color="red",
    #     alpha=0.7,
    #     # edgecolor="none",
    #     s=7,
    #     # bins=z_bins,
    #     label=z_name,
    # )
    axs[0].scatter(
        full_cat[z_cat_name][secure & idx & ~offset],
        full_cat["Z_NIRISS"][secure & idx & ~offset],
        # full_cat["MAG_AUTO"][secure],
        color="purple",
        alpha=0.7,
        # edgecolor="none",
        s=7,
        # bins=z_bins,
        label=z_name,
    )
    axs[0].scatter(
        full_cat[z_cat_name][secure & idx & offset],
        full_cat["Z_NIRISS"][secure & idx & offset],
        # full_cat["MAG_AUTO"][secure],
        facecolor="none",
        alpha=0.7,
        edgecolor="purple",
        s=7,
        # bins=z_bins,
        label=z_name,
    )
    # outliers = full_cat["ID_NIRISS"]
    scatter = np.nanstd(
        (
            full_cat["Z_NIRISS"][secure & idx & ~offset]
            - full_cat[z_cat_name][secure & idx & ~offset]
        )
    )
    axs[0].text(
        0.1,
        0.9,
        rf"$\sigma_{{z}}={scatter:.2f}$",
        va="top",
        transform=axs[0].transAxes,
    )
    # shift = np.nanmean(
    #     dz,
    # )
    # shift_std = np.nanstd(
    #     ((full_cat[z_cat_name]-full_cat["Z_NIRISS"])/(1+full_cat[z_cat_name]))[secure & idx & ~offset],
    # )
    # print (shift_std)
    # axs[1,i].text(
    #     0.1,
    #     0.9,
    #     rf"$\overline{{\Delta z}}={np.nanmean(dz[secure & idx & ~offset]):.3f}$",
    #     va="top",
    #     transform=axs[1,i].transAxes,
    # )
    if z_cat_name == "zmed_prev":
        # str_print = f"{np.nanmean(dz[secure & idx & ~offset]):.1E}".split("E")
        # str_print = rf"{str_print[0]}\times 10^{{{int(str_print[-1])}}}"
        # print (str_print)
        # axs[1,i].text(
        #     0.1,
        #     0.9,
        #     rf"$\overline{{\Delta z}}={str_print}$",
        #     va="top",
        #     transform=axs[1,i].transAxes,
        # )
        text = r"\begin{align*} "  # + "\n"
        for j, (source, colname) in enumerate(
            zip(["NIRSpec", "NIRCam", "Ground"], ["z_nirspec", "z_alt", "z_ground"])
        ):
            selection = (~full_cat[colname].mask) & (np.isfinite(full_cat[colname]))
            # str_print = np.nanmean(dz[secure & idx & ~offset & selection])
            str_print = (
                f"{np.nanmean(dz[secure & idx & ~offset & selection]):.1E}".split("E")
            )
            str_print = rf"{str_print[0]}\times 10^{{{int(str_print[-1])}}}"
            # print (str_print)
            # axs[1,i].text(
            #     0.1,
            #     0.9-j*0.125,
            #     # rf"$\overline{{\Delta z}}={np.nanmean(dz[secure & idx & ~offset & selection]):.1E}$ ({source})",
            #     rf"${rf"\overline{{\Delta z}}_{{\rm{{{source}}}}}":<15}={str_print}$",
            #     va="top",
            #     transform=axs[1,i].transAxes,
            # )
            text += (
                rf"\overline{{\Delta z}}_{{\textsc{{{source.lower()}}}}} &= {str_print}"
                + ("\\\\" if j != 2 else "")
            )  # + "\n"
        text += r"\end{align*}"
        print(repr(text))
        # exit()
        axs[1].text(
            0.1,
            0.9,
            text,
            va="top",
            transform=axs[1].transAxes,
        )

    else:
        axs[1].text(
            0.1,
            0.9,
            rf"$\overline{{\Delta z}}={np.nanmean(dz[secure & idx & ~offset]):.3f}$",
            va="top",
            transform=axs[1].transAxes,
        )

    # print(
    #     full_cat["NUMBER", "Z_NIRISS", z_cat_name, "RA", "DEC","id_msa_1324"][
    #         (secure & idx)
    #         & (np.abs(full_cat["Z_NIRISS"] - full_cat[z_cat_name]) > 1)
    #     ]
    # )
    print(
        full_cat["NUMBER", "Z_NIRISS", z_cat_name, "RA", "DEC", "id_msa_1324"][
            (secure & idx) & (np.abs(dz) > 0.025)
        ]
    )

    # ax.hist(
    #     # full_cat["zspec"][secure],
    #     full_cat["zphot"][secure] - full_cat["Z_EST_ALL"][secure],
    #     # full_cat["MAG_AUTO"][secure],
    #     color="purple",
    #     # alpha=.7,
    #     # edgecolor="none",
    #     # s=7,
    #     bins=np.arange(-0.025, 0.025, 0.001),
    # )
    # print (full_cat["ID"][(full_cat["zspec"]>=7) & secure])
    # ax.scatter(
    #     full_cat["Z_EST_ALL"][tentative],
    #     full_cat["MAG_AUTO"][tentative],
    #     color="k",
    #     facecolor="none",
    #     marker="o",
    #     alpha=.7,
    #     s=4,
    #     linewidth=0.5,
    # )
    # ax.set_xlim(z_bins[0], z_bins[-1])
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

    # # ax.set_ylabel(r"$m_{\rm{F200W}}$")
    # axs[1].semilogy()
    # axs[1].semilogx()
    axs[0].set_xscale("log")
    # axs[0].set_xscale("log")
    axs[0].set_yscale("log")
    # axs[1].set_xticks(
    z_ticks = np.concatenate([np.arange(0.1, 1, 0.1), np.arange(1.0, 10.0, 1)])
    z_ticks_labels = [
        "0.1",
        "0.2",
        "",
        "",
        "0.5",
        "",
        "",
        "",
        "",
        "1.0",
        "2.0",
        "",
        "",
        "5.0",
        "",
        "",
        "",
        "10.0",
    ]
    # axs[1].set_xticklabels([])
    # axs[0].set_xticks(z_ticks[:-1], z_ticks_labels[:-1], minor=True)
    axs[0].set_xticks(z_ticks, z_ticks_labels, minor=True)
    axs[0].set_yticks(z_ticks, z_ticks_labels, minor=True)
    axs[0].set_xticks([0.1, 1.0], ["0.1", "1.0"], minor=False)
    # axs[0,1].set_xticks([0.1, 1.0], ["0.1", "1.0"], minor=False)
    axs[0].set_yticks([0.1, 1.0], ["0.1", "1.0"], minor=False)
    axs[1].set_xlabel(r"$z_{\rm{phot}}$")
    # axs[1].set_xlabel(r"$z_{\rm{spec}}$")
    axs[0].set_ylabel(r"$z_{\rm{NIRISS}}$")
    axs[1].set_ylabel(r"$\delta z / (1+z)$")
    # axs[1].set_ylabel(r"$z_{\rm{NIRISS}}$")
    # # ax.set_ylabel(r"Number of Objects")
    # # ax.legend()

    lims = [0.08, 10]

    axs[0].set_xlim(lims)
    # axs[0,1].set_xlim(lims)
    # ylims
    # axs[1].set_ylim(lims)
    #
    # plt.subplots_adjust(wspace=0, hspace=0)

    plt.savefig(save_dir / "dz_1pz_megascience.pdf")
    # for k, v in line_dict.items():
    #     for k_n, v_n in niriss_info.items():
    #         # low = v_n[0]/v -1
    #         print(f"{k}: {k_n}={v_n[0]/v-1:.3f},{v_n[1]/v-1:.3f}")

    plt.show()
