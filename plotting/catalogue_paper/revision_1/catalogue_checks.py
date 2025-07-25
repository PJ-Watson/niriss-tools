"""
A default plotting script.
"""

import plot_utils
from astropy.table import join, vstack
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    # print (ext_cat)
    tentative = ext_cat["Z_FLAG"] == 3
    secure = ext_cat["Z_FLAG"] == 4
    disagree_cutoff = 1.6e-3
    disagree = (
        np.abs(
            (ext_cat["zmed_prev"] - ext_cat["Z_NIRISS"]) / (1 + ext_cat["zmed_prev"])
        )
        > disagree_cutoff
    )
    agree = (
        np.abs(
            (ext_cat["zmed_prev"] - ext_cat["Z_NIRISS"]) / (1 + ext_cat["zmed_prev"])
        )
        < disagree_cutoff
    )
    # disagree = (ext_cat["zmed_prev"] - ext_cat["Z_NIRISS"])>0.01
    zprev_exist = ext_cat["zmed_prev"] > 0
    zspec_exist = (
        (ext_cat["z_alt"].filled(np.nan) > 0)
        | (ext_cat["z_ground"].filled(np.nan) > 0)
        | (ext_cat["z_nirspec"].filled(np.nan) > 0)
    )
    zspec_not_exist = np.bitwise_not(
        (ext_cat["z_alt"].filled(np.nan) > 0)
        | (ext_cat["z_ground"].filled(np.nan) > 0)
        | (ext_cat["z_nirspec"].filled(np.nan) > 0)
    )
    print(zspec_not_exist)
    print(((ext_cat["Z_REFIT"] == True) & (ext_cat["Z_FLAG"] >= 3)).sum())
    print(((ext_cat["Z_REFIT"] == True) & (ext_cat["Z_FLAG"] == 4) & zspec_exist).sum())

    orig_cat = Table.read(
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "classification-stage-2"
        / "catalogues"
        / "input"
        / "compiled_catalogue_v1.fits"
    )["NUMBER", "REDSHIFT_USE"]
    orig_cat.rename_columns(["NUMBER", "REDSHIFT_USE"], ["ID_NIRISS", "Z_GRIZLI"])
    forced_cat = Table.read(
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Extractions_v3"
        / "classification"
        / "stage_3_PJW_class.fits"
    )["ID", "GRIZLI_REDSHIFT"]
    forced_cat.rename_columns(["ID", "GRIZLI_REDSHIFT"], ["ID_NIRISS", "Z_GRIZLI"])
    # print(forced_cat.colnames)
    forced_cat["ID_NIRISS"] = forced_cat["ID_NIRISS"].astype(int)
    for i, row in enumerate(orig_cat[:]):
        if np.isin(row["ID_NIRISS"], forced_cat["ID_NIRISS"]):
            print("y", row["ID_NIRISS"])
            print(
                [
                    np.where(
                        forced_cat["ID_NIRISS"].astype(int)
                        == row["ID_NIRISS"].astype(int)
                    )
                ]
            )
            orig_cat["Z_GRIZLI"][i] = forced_cat["Z_GRIZLI"][
                np.where(
                    forced_cat["ID_NIRISS"].astype(int) == row["ID_NIRISS"].astype(int)
                )
            ]
    orig_cat = vstack(
        [orig_cat, forced_cat[~np.isin(forced_cat["ID_NIRISS"], orig_cat["ID_NIRISS"])]]
    )
    orig_cat.rename_column("Z_GRIZLI", "Z_GRIZLI_OLD")

    overlap_cat = join(ext_cat, orig_cat, keys="ID_NIRISS")
    # plt.plot([0, 10], [0, 10], c="k", linestyle=":")
    # plt.scatter(
    #     overlap_cat["Z_GRIZLI_OLD"], overlap_cat["Z_NIRISS"], c=overlap_cat["Z_REFIT"]
    # )
    # print (overlap_cat[zspec_exist])
    print(
        (
            (overlap_cat["Z_GRIZLI_OLD"] - overlap_cat["Z_NIRISS"])[zspec_exist] >= 0.01
        ).sum()
    )
    print(
        (
            (overlap_cat["Z_GRIZLI_OLD"] - overlap_cat["Z_NIRISS"])[zspec_not_exist]
            >= 0.1
        ).sum()
    )
    print(
        (
            np.bitwise_not(
                (
                    ((overlap_cat["Z_GRIZLI_OLD"] - overlap_cat["Z_NIRISS"]) >= 0.1)
                    & (zspec_not_exist)
                )
                | (
                    ((overlap_cat["Z_GRIZLI_OLD"] - overlap_cat["Z_NIRISS"]) >= 0.01)
                    & zspec_exist
                )
            )
            & (overlap_cat["Z_REFIT"])
            & ((overlap_cat["Z_GRIZLI_OLD"] - overlap_cat["Z_NIRISS"]) > 1.6e-3)
        ).sum()
    )
    print(overlap_cat["Z_REFIT"].sum())
    # plt.show()

    # print (ext_cat["Z_GRIZLI"])
    # print (orig_cat)
    # exit()

    # plt.hist(ext_cat["Z_NIRISS"][((ext_cat["Z_REFIT"]==True) & (ext_cat["Z_FLAG"]>=3))], bins=np.arange(0,3,0.05))
    # plt.show()

    for data_idx, data_unc in zip(
        [
            (ext_cat["Z_REFIT"] == True) & (ext_cat["Z_FLAG"] >= 3) & zspec_exist,
            (ext_cat["Z_REFIT"] == True) & (ext_cat["Z_FLAG"] >= 3) & zspec_not_exist,
        ],
        [0.01, 0.1],
    ):
        data = ext_cat[data_idx].copy()
        if data_unc == 0.1:
            data["zmed_prev"] = data["zphot_astrodeep"]
        # plt.scatter(
        #     # data_unc * data["zmed_prev"],
        #     data["DZ_1PZ"],
        #     # (data["Z_NIRISS"]-data["zmed_prev"]) / (data["zmed_prev"]+1)
        #     np.abs(data["Z_NIRISS"] - data["zmed_prev"]),
        # )
        # # plt.plot([0.001, 0.1], [0.001, 0.1])

        if data_unc == 0.01:
            max_unc = np.nanmax(
                [
                    data_unc * data["zmed_prev"],
                    np.full_like(data["zmed_prev"], data_unc),
                ],
                axis=0,
            )
        else:
            max_unc = data_unc * data["zmed_prev"]
        data_outside = np.abs(data["Z_NIRISS"] - data["zmed_prev"]) > max_unc
        print("Outside bounds:", sum(data_outside))
        # plt.show()

    tent_and_dis = ext_cat[tentative & disagree]
    print(len(ext_cat[tentative & disagree]))
    # print (len(tent_and_dis))
    # print(ext_cat.colnames)
    print(len(ext_cat[tentative & zspec_exist & agree]))

    print(np.nanmedian(ext_cat["zwidth1"][tentative]))
    print(np.nanmedian(ext_cat["zwidth1"][secure][ext_cat["zwidth1"][secure] >= 0]))
    print(ext_cat["zwidth1"][4])

    fig, axs = plt.subplots(
        1,
        1,
        # figsize=(plot_utils.aanda_textwidth * 0.75, plot_utils.aanda_columnwidth),
        figsize=(plot_utils.aanda_textwidth * 0.75, plot_utils.aanda_columnwidth),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        # hspace=0.,
        # wspace=0.,
    )
    bins = np.geomspace(1e-4, 0.05, 30)
    # bins = np.linspace(0,0.05,50)
    print(secure.sum(), tentative.sum())
    # np.logspace
    all_data = ext_cat["zwidth1"][(tentative | secure)]
    kwargs = {"bins": bins, "alpha": 0.5}
    for data_idx, col, name in zip(
        [tentative, secure], ["C0", "C1"], ["Tentative", "Secure"]
    ):
        data = ext_cat["zwidth1"][data_idx]
        axs.hist(
            data,
            color=col,
            **kwargs,
            weights=np.ones(len(data)) / len(data),
            label=name,
        )

        axs.axvline(np.nanmedian(data[np.isfinite(data)]), linestyle=":", c=col)
    axs.hist(
        all_data,
        edgecolor="k",
        facecolor="none",
        histtype="step",
        **kwargs,
        weights=np.ones(len(all_data)) / len(all_data),
        label="All",
    )

    # plt.hist(
    #     ext_cat["zwidth1"][secure], **kwargs
    # )
    axs.set_xlabel(r"\textsc{grizli} ``\texttt{zwidth1}''")
    # axs.set_ylabel(r"Number of Galaxies")
    axs.set_ylabel(r"Fraction of Galaxies")
    plt.semilogx()

    axs.legend()
    # plt.show()
    # fig, axs = plt.subplots(
    #     1,
    #     1,
    #     figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.8),
    #     constrained_layout=True,
    #     sharex=True,
    #     sharey=True,
    #     # hspace=0.,
    #     # wspace=0.,
    # )
    # # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "grism_uncertainties.pdf")

    # plt.show()

    print(niriss_cat)

    # for c in niriss_cat.colnames:
    # if
    print(len(niriss_cat) * 2)
    for q in np.arange(6, -1, step=-1):
        q = np.round(q / 2, 1)
        # print (q)
        # print ([(niriss_cat[f"{f}_072.0_QUALITY"]==q).sum()+(niriss_cat[f"{f}_341.0_QUALITY"]==q).sum() for f in ["F115W","F150W","F200W"]])

        print(
            f"{q} & "
            + " & ".join(
                [
                    f"{(niriss_cat[f"{f}_072.0_QUALITY"]==q).sum()+(niriss_cat[f"{f}_341.0_QUALITY"]==q).sum()}"
                    for f in ["F115W", "F150W", "F200W"]
                ]
            )
            + f" & {np.nansum([
                    (niriss_cat[f"{f}_072.0_QUALITY"]==q).sum()+(niriss_cat[f"{f}_341.0_QUALITY"]==q).sum()
                    for f in ["F115W", "F150W", "F200W"]
                ])}"
            + " \\\\"
        )

    print(
        f"{q} & "
        + " & ".join(
            [
                f"{(niriss_cat[f"{f}_072.0_QUALITY"]>=0).sum()+(niriss_cat[f"{f}_341.0_QUALITY"]>=0).sum()}"
                for f in ["F115W", "F150W", "F200W"]
            ]
        )
        + f" & {np.nansum([
                (niriss_cat[f"{f}_072.0_QUALITY"]>=0).sum()+(niriss_cat[f"{f}_341.0_QUALITY"]>=0).sum()
                for f in ["F115W", "F150W", "F200W"]
            ])}"
        + " \\\\"
    )
    print(niriss_cat.colnames)
    plt.clf()
    fig, axs = plt.subplots()
    sn_cutoff = 10
    niriss_cat = niriss_cat[
        ((niriss_cat["flux_Hb"] / niriss_cat["err_Hb"]) >= sn_cutoff)
        & ((niriss_cat["flux_Ha"] / niriss_cat["err_Ha"]) >= sn_cutoff)
    ]
    for row in niriss_cat:
        print(row["ID_NIRISS"])
    sc = axs.scatter(
        np.log10(niriss_cat["flux_Ha"]),
        niriss_cat["flux_Ha"] / niriss_cat["flux_Hb"],
        c=niriss_cat["flux_Ha"] / niriss_cat["err_Ha"],
        # c=niriss_cat["Z_NIRISS"],
        vmax=50,
    )
    plt.colorbar(sc, ax=axs, label="Halpha S/N")
    axs.set_xlabel("log Ha flux")
    axs.axhline(2.86, zorder=-1, c="k", linestyle=":")
    axs.set_ylabel("Ha/Hb (uncorrected)")

    plt.show()
