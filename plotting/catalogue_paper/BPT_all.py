"""
Check the location of sources in the BPT diagram.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    fig, axs = plt.subplots(
        1,
        1,
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        # hspace=0.,
        # wspace=0.,
    )

    lines = {
        "OII": np.array([3727.092, 3729.875]),
        "Hb": 4862.738,
        "OIII": np.array([5008.240, 4960.295]),
        "Ha": 6564.697,
        "SII": np.array([6718.29, 6732.67]),
        "SIII": np.array([9071.1, 9533.2]),
    }

    max_sn = []
    for row in ext_cat:
        max_sn.append(np.nanmax([row[f"flux_{l}"] / row[f"err_{l}"] for l in lines]))
    ext_cat["max_sn"] = max_sn

    sn_bins = np.arange(0, 200, 3)

    non_cluster = (ext_cat["Z_NIRISS"] <= 0.25) | (ext_cat["Z_NIRISS"] >= 0.35)

    # axs.hist(ext_cat["max_sn"][(ext_cat["Z_FLAG"] >= 4) & non_cluster], bins=sn_bins)
    # axs.hist(ext_cat["max_sn"][(ext_cat["Z_FLAG"] ==3 )& non_cluster], bins=sn_bins)

    print(
        ext_cat["ID_NIRISS"][
            (ext_cat["max_sn"] > 10) & (ext_cat["Z_FLAG"] == 3) & non_cluster
        ]
    )

    # plt.show()

    # exit()

    # print (e, np.unique(np.array(ext_cat[e].data), return_counts=True))
    print(np.unique(np.array(ext_cat["Z_FLAG"].data), return_counts=True))

    sn_lim = 3
    secure = ext_cat["Z_FLAG"] >= 4
    # for l in ["SII", "OIII", "Hb", "Ha"]:
    for l in ["OIII", "Hb", "OII"]:
        # for l in ["Hb", "Ha"]:
        secure &= (
            (ext_cat[f"flux_{l}"] > 0)
            & (ext_cat[f"flux_{l}"] / ext_cat[f"err_{l}"] > sn_lim)
            & (np.isfinite(ext_cat[f"flux_{l}"]))
        )
    print(np.nanmedian(ext_cat["flux_Ha"][secure]))

    print(
        np.nansum(
            (ext_cat["Z_FLAG"] >= 4)
            & (ext_cat[f"flux_Ha"] > 0)
            & (ext_cat[f"flux_Ha"] / ext_cat[f"err_Ha"] > 3)
            & (np.isfinite(ext_cat[f"flux_Ha"]))
            & (ext_cat[f"flux_Hb"] >= 0)
            # & (ext_cat[f"flux_Ha"] / ext_cat[f"err_Ha"] > 3)
            & (np.isfinite(ext_cat[f"flux_Hb"]))
        )
    )
    print(ext_cat["ID_NIRISS"][secure])

    # plt.scatter(
    #     ext_cat["flux_Ha"][secure],
    #     ext_cat["flux_Hb"][secure],

    # )
    # plt.semilogx()
    # plt.semilogy()
    # plt.show()
    # exit()
    from ppxf.ppxf import attenuation

    e_bv = 1.97 * np.log10((ext_cat["flux_Ha"] / ext_cat["flux_Hb"]) / 2.86)
    a_v = 4.05 * e_bv

    lines = {
        "OII": np.array([3727.092, 3729.875]),
        "Hb": 4862.738,
        "OIII": np.array([5008.240, 4960.295]),
        "Ha": 6564.697,
        "SII": np.array([6718.29, 6732.67]),
        "SIII": np.array([9071.1, 9533.2]),
    }

    for i, (row, a_v_i) in enumerate(zip(ext_cat, a_v)):
        for l, w_rf in lines.items():
            # print (a_v_i)
            w_rf = np.atleast_1d(w_rf)[-1]
            frac = attenuation(w_rf, a_v_i)
            ext_cat[f"flux_{l}"][i] *= frac
            ext_cat[f"err_{l}"][i] *= frac
    # print(
    #     ext_cat[
    #         secure
    #         & ( np.log10(ext_cat["flux_OIII"]*0.75 / ext_cat["flux_Hb"])> 0.9)
    #     ]
    # )

    # print(
    #     ext_cat[
    #         secure
    #         & (( np.log10(ext_cat["flux_SII"] / ext_cat["flux_Ha"]/1.2)> -0.6)
    #         | ( np.log10(ext_cat["flux_OIII"]*0.75 / ext_cat["flux_Hb"])> 0.6))
    #     ]
    # )

    # test_idx =
    for row in ext_cat[secure]:
        print(
            row["ID_NIRISS"],
            np.log10(row["flux_SII"] / row["flux_Ha"] / 1.2),
            np.log10(row["flux_OIII"] * 0.75 / row["flux_Hb"]),
        )

    x_lim = np.arange(-1.5, 0.3, 0.01)
    y_lim = np.arange(-1, 1.5, 0.01)

    print(sum(secure))
    axs.plot(x_lim, 1.3 + 0.72 / (x_lim - 0.32), linestyle=":", c="k", zorder=-1)

    # plt.hist(ext_cat["flux_Ha"][secure], bins=np.arange(0,1e-16,1e-18))
    plot_col = ext_cat["Z_NIRISS"]
    plot_col = ext_cat["flux_SII"] / ext_cat["err_SII"]
    # plot_col = np.log10(ext_cat["flux_OIII"] / ext_cat["err_OIII"])
    # plot_col = np.log10(ext_cat["flux_SII"] / ext_cat["err_SII"])
    # plot_col = np.log10(ext_cat["flux_Ha"] / ext_cat["err_Ha"])
    plot_col = np.log10(ext_cat["flux_Hb"] / ext_cat["err_Hb"])
    # plot_col = np.log10(np.nanmin([(ext_cat["flux_Hb"] / ext_cat["err_Hb"]),(ext_cat["flux_SII"] / ext_cat["err_SII"])], axis=0))
    sc = axs.scatter(
        # np.log10(ext_cat["flux_SII"] / (ext_cat["flux_Ha"] / 1.2))[secure],
        np.log10(ext_cat["flux_OII"] / (ext_cat["flux_OIII"] * 0.75))[secure],
        np.log10((ext_cat["flux_OIII"] * 0.75) / ext_cat["flux_Hb"])[secure],
        c=plot_col[secure],
        cmap="rainbow",
        s=8,
        vmax=1.6,
    )
    # x=
    # xerr = np.log10(((ext_cat["flux_SII"]+ext_cat["err_SII"]) / ((ext_cat["flux_Ha"]-ext_cat["err_Ha"]) / 1.2))) - np.log10(ext_cat["flux_SII"] / (ext_cat["flux_Ha"] / 1.2))
    # yerr = np.log10((((ext_cat["flux_OIII"]+ext_cat["err_OIII"])*0.75) / ((ext_cat["flux_Hb"]-ext_cat["err_Hb"])))) - np.log10((ext_cat["flux_OIII"] * 0.75) / ext_cat["flux_Hb"])
    # print (xerr[secure])
    # axs.errorbar(
    #     np.log10(ext_cat["flux_OII"] / (ext_cat["flux_OIII"]*0.75))[secure],
    #     # np.log10(ext_cat["flux_SII"] / (ext_cat["flux_Ha"] / 1.2))[secure],
    #     np.log10((ext_cat["flux_OIII"] * 0.75) / ext_cat["flux_Hb"])[secure],
    #     xerr=xerr[secure],
    #     yerr=yerr[secure],
    #     fmt=".",
    #     ecolor=(0,0,0,0.5),
    #     linewidth=1,
    #     # c=plot_col[secure],
    #     # cmap="plasma",
    #     markersize=1,
    #     zorder=-1
    # )
    # plt.colorbar(
    #     sc,
    #     # label="Redshift",
    #     label="log(S/N) [Hbeta]"
    # )
    axs.set_xlim(x_lim[0], x_lim[-1])
    axs.set_ylim(y_lim[0], y_lim[-1])
    axs.set_xlabel(r"$\rm{log}_{10}\left(\rm{SII}/\rm{H}\alpha\right)$")
    axs.set_ylabel(r"$\rm{log}_{10}\left(\rm{OIII}/\rm{H}\beta\right)$")
    # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    fig.patch.set_alpha(0.0)

    # plt.savefig(save_dir / "BPT_all.pdf")

    plt.show()
