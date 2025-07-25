"""
Show cutouts of the confirmed multiple images.
"""

import plot_utils
from astrocolour import ColourImage
from astrocolour.utils import RawAsinhTransform
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    z_groups = [[1.09, 1.11], [1.32, 1.37], [1.85, 1.90]]
    # z_groups = [[3.0,3.1]]
    z_groups = [[2.6, 2.8]]
    z_groups = [[2.63, 2.67]]
    # z_groups = [[2.,2.4]]

    bgr = np.array(
        [
            fits.getdata(
                grizli_dir / "Prep" / f"{root_name}-f{filt}wn-clear_drc_sci.fits"
            )
            for filt in [115, 150, 200]
        ]
    )

    border = 5
    arr_page_size = (5,)

    fig = plt.figure(
        1,
        constrained_layout=True,
    )
    multi_img_ids = np.array(
        [
            [380, 458, 998],
            [345, 475, 936],
        ]
    )

    fig.set_size_inches(
        plot_utils.aanda_columnwidth,
        plot_utils.aanda_columnwidth * len(multi_img_ids.ravel()) / 3.2,
    )
    subfigs = fig.subfigures(2, 1)

    for m_i, (m_ids, m_lab) in enumerate(
        zip(
            multi_img_ids,
            [
                r"(a)\ \ $\overline{z}=2.011$",
                r"(b)\ \ $\overline{z}=2.653$",
            ],
        )
    ):

        axs = subfigs[m_i].subplots(
            len(m_ids.ravel()),
            2,
            width_ratios=[2.5, 1],
            # gridspec_kw={"hspace": 0.01, "wspace": 0.01},
        )
        subfigs[m_i].suptitle(m_lab)
        subfigs[m_i].supylabel(
            r"$f_{\lambda}\ [10^{-19}\,\rm{erg}/\rm{s}/\rm{cm}^{2}/\AA]$"
        )

        idx = (ext_cat["Z_FLAG"] >= 4) & (np.isin(ext_cat["ID_NIRISS"], m_ids))
        # print (z_low)

        # import matplotlib.backends.backend_pdf

        # pdf = matplotlib.backends.backend_pdf.PdfPages(
        #     save_dir / f"overdensities_cutouts_z_{z_low}_{z_high}.pdf"
        # )

        # idx = (
        #     (ext_cat["Z_NIRISS"] >= z_low)
        #     & (ext_cat["Z_NIRISS"] < z_high)
        #     & (ext_cat["Z_FLAG"] >= 4)
        # )
        # print(len(ext_cat[idx]))
        overdens_cat = ext_cat[idx]  #
        overdens_cat.sort("ID_NIRISS")
        for i, (ax, row) in enumerate(zip(np.atleast_2d(axs), overdens_cat)):
            # print (row["X_MIN"])
            # print (row.colnames)
            # print (ax)
            width = int(
                np.nanmax([row["XMAX"] - row["XMIN"], row["YMAX"] - row["YMIN"]]) // 2
                + border
            )
            width = 25
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
                    "transformation": RawAsinhTransform(
                        astrovis.ManualInterval(-0.001, 2)
                    )
                },
            )
            ax[1].imshow(cimg, origin="lower")
            ax[1].set_xticks([])
            ax[1].set_yticks([])

            # print (row["ID"])
            with fits.open(
                grizli_dir
                / "Extractions_v4"
                / "1D"
                / f"{root_name}_{row["ID_NIRISS"]:0>5}.1D.fits"
            ) as oned_hdul:
                for hdu in oned_hdul[1:]:
                    dat = Table(hdu.data)
                    dat["wave"] /= 1e4
                    dat["flat"] *= 1e-18
                    wave_lims = (dat["wave"] >= niriss_filter_sens[hdu.name][0]) & (
                        dat["wave"] <= niriss_filter_sens[hdu.name][-1]
                    )
                    dat = dat[wave_lims]
                    ax[0].plot(
                        dat["wave"],
                        dat["flux"] / dat["flat"],
                        drawstyle="steps-mid",
                        c="#d55e00",
                        label="Data" if i == 0 else "_",
                    )
                    ax[0].fill_between(
                        # dat["wave"] / (1 + obj_z) / 1e4,
                        dat["wave"],
                        ((dat["flux"] - dat["err"]) / dat["flat"]),
                        ((dat["flux"] + dat["err"]) / dat["flat"]),
                        step="mid",
                        alpha=0.3,
                        color="#d55e00",
                    )

                    # ax[0].plot(
                    #     dat["wave"],
                    #     dat["contam"] / dat["flat"] ,
                    #     c="#0072b2",
                    #     alpha=0.7,
                    #     drawstyle="steps-mid",
                    #     zorder=-1,
                    #     label="Contamination" if i == 0 else "_",
                    # )
                ax[0].set_ylim(ymin=0)
                # axs[i,1].sharex(axs[0,1])
                # print(i, len(page_idxs))
                if (i + 1) != len(m_ids.ravel()):
                    ax[0].set_xticklabels([])
                else:
                    ax[0].set_xlabel(r"Observed Wavelength ($\mu$m)")

                an = ax[0].annotate(
                    rf"ID {row["ID_NIRISS"]}",
                    (0.95, 0.85),
                    ha="right",
                    va="top",
                    xycoords="axes fraction",
                    # rotation="vertical"
                )
                an = ax[0].annotate(
                    rf"$z={row["Z_NIRISS"]:.3f}$",
                    (0.95, 0.7),
                    ha="right",
                    va="top",
                    xycoords="axes fraction",
                    # rotation="vertical"
                )
                # an = ax[0].annotate(
                #     rf"$\rm{{H}}\alpha/\rm{{H}}\beta={row["flux_Ha"]/row["flux_Hb"]:.3f}$",
                #     (0.95, 0.7),
                #     ha="right",
                #     va="top",
                #     xycoords="axes fraction",
                #     # rotation="vertical"
                # )
                # an = ax[0].annotate(
                #     rf"$\rm{{O}}III/\rm{{H}}\alpha={row["flux_OIII"]/row["flux_Ha"]:.3f}$",
                #     (0.95, 0.6),
                #     ha="right",
                #     va="top",
                #     xycoords="axes fraction",
                #     # rotation="vertical"
                # )
                # an = ax[0].annotate(
                #     rf"$\rm{{O}}III/\rm{{O}}II={row["flux_OIII"]/row["flux_OII"]:.3f}$",
                #     (0.95, 0.5),
                #     ha="right",
                #     va="top",
                #     xycoords="axes fraction",
                #     # rotation="vertical"
                # )

                # an = ax[0].annotate(
                #     rf"{row["RA"]:.5f} {row["DEC"]:.5f}",
                #     (0.95, 0.4),
                #     ha="right",
                #     va="top",
                #     xycoords="axes fraction",
                #     # rotation="vertical"
                # )

            # plt.imshow(np.log(bgr[0][row["YMIN"]:row["YMAX"],row["XMIN"]:row["XMAX"]]))
            # plt.imshow(
            #     np.log(
            #         bgr[0][
            #             int(row["Y"] - width) : int(row["Y"] + width),
            #             int(row["X"] - width) : int(row["X"] + width),
            #         ]
            #     )
            # )
            # axs.scatter(niriss_ext_cat[idx]["RA"], niriss_ext_cat[idx]["DEC"])

    fig.patch.set_alpha(0.0)

    # print (ext_cat)

    plt.savefig(save_dir / "multiple_images_cutouts_rev1.pdf")
    # pdf.savefig(1)
    # plt.clf()
    plt.show()

    # pdf.close()
