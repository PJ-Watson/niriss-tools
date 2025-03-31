"""
Show cutouts at a particular redshift.
"""

import plot_utils
from astrocolour import ColourImage
from astrocolour.utils import RawAsinhTransform
from default_imports import *

plot_utils.setup_aanda_style()


def split_to_shape(a, chunk_shape, start_axis=0):
    """
    Split an array into chunks.

    Parameters
    ----------
    a : _type_
        The array to be divided.
    chunk_shape : _type_
        The chunk to use when dividing a.
    start_axis : int, optional
        The axis to start splitting on, by default 0.

    Returns
    -------
    array-like
        A nested array of values.
    """
    import math

    if len(chunk_shape) != len(a.shape):
        raise ValueError("chunk length does not match array number of axes")

    if start_axis == len(a.shape):
        return a

    split_indices = [
        chunk_shape[start_axis] * (i + 1)
        for i in range(math.floor(a.shape[start_axis] / chunk_shape[start_axis]))
    ]
    split = np.array_split(a, split_indices, axis=start_axis)
    return [split_to_shape(split_a, chunk_shape, start_axis + 1) for split_a in split]


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

    for z_low, z_high in z_groups[:]:
        # print (z_low)

        import matplotlib.backends.backend_pdf

        pdf = matplotlib.backends.backend_pdf.PdfPages(
            save_dir / f"overdensities_cutouts_z_{z_low}_{z_high}.pdf"
        )

        idx = (
            (ext_cat["Z_NIRISS"] >= z_low)
            & (ext_cat["Z_NIRISS"] < z_high)
            & (ext_cat["Z_FLAG"] >= 4)
        )
        print(len(ext_cat[idx]))
        overdens_cat = ext_cat[idx]  #
        overdens_cat.sort("Z_NIRISS")
        # fig, axs = plt.subplots(
        #     len(overdens_cat),
        #     2,
        #     figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth * len(overdens_cat)/1.5),
        #     constrained_layout=True,
        #     # sharex=True,
        #     # sharey=True,
        #     # hspace=0.,
        #     # wspace=0.,
        # )
        # print (len(overdens_cat)/5)
        # print (np.floor(len(overdens_cat) / arr_page_size))
        # print ([arr_page_size*(i+1) for i in range(int(np.floor(len(overdens_cat)/5)))])
        table_idxs = split_to_shape(np.arange(len(overdens_cat)), arr_page_size)
        for page_i, page_idxs in enumerate(table_idxs):
            print(page_idxs)
            if len(page_idxs) == 0:
                continue
            fig.set_size_inches(
                plot_utils.aanda_textwidth,
                plot_utils.aanda_columnwidth * len(page_idxs) / 2.0,
            )
            axs = fig.subplots(
                len(page_idxs),
                2,
                width_ratios=[1, 3],
            )
            # fig, axs = plt.subplots(
            #     len(page_idxs),
            #     2,
            #     figsize=(
            #         plot_utils.aanda_textwidth,
            #         plot_utils.aanda_columnwidth * len(page_idxs) / 2.0,
            #     ),
            #     constrained_layout=True,
            #     width_ratios=[1, 3],
            #     # sharex=True,
            #     # sharey=True,
            #     # hspace=0.,
            #     # wspace=0.,
            # )
            # # exit()
            for i, (ax, row) in enumerate(
                zip(np.atleast_2d(axs), overdens_cat[page_idxs])
            ):
                # print (row["X_MIN"])
                # print (row.colnames)
                # print (ax)
                width = int(
                    np.nanmax([row["XMAX"] - row["XMIN"], row["YMAX"] - row["YMIN"]])
                    // 2
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
                        "transformation": RawAsinhTransform(
                            astrovis.ManualInterval(-0.001, 2)
                        )
                    },
                )
                ax[0].imshow(cimg, origin="lower")
                ax[0].set_xticks([])
                ax[0].set_yticks([])

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
                        dat["flat"] /= 1e-18
                        wave_lims = (dat["wave"] >= niriss_filter_sens[hdu.name][0]) & (
                            dat["wave"] <= niriss_filter_sens[hdu.name][-1]
                        )
                        dat = dat[wave_lims]
                        ax[1].plot(
                            dat["wave"],
                            dat["flux"] / dat["flat"],
                            drawstyle="steps-mid",
                            c="purple",
                        )
                        ax[1].fill_between(
                            # dat["wave"] / (1 + obj_z) / 1e4,
                            dat["wave"],
                            ((dat["flux"] - dat["err"]) / dat["flat"]),
                            ((dat["flux"] + dat["err"]) / dat["flat"]),
                            step="mid",
                            alpha=0.3,
                            color="purple",
                        )
                    ax[1].set_ylim(ymin=0)
                    # axs[i,1].sharex(axs[0,1])
                    # print(i, len(page_idxs))
                    if (i + 1) != len(page_idxs):
                        ax[1].set_xticklabels([])
                    else:
                        ax[1].set_xlabel(r"Observed Wavelength ($\mu$m)")
                    ax[1].set_ylabel(
                        r"$f_{\lambda}\ [10^{-19}\,\rm{erg}/\rm{s}/\rm{cm}^{2}/\AA]$"
                    )

                    an = ax[1].annotate(
                        rf"ID {row["ID_NIRISS"]}",
                        (0.95, 0.9),
                        ha="right",
                        va="top",
                        xycoords="axes fraction",
                        # rotation="vertical"
                    )
                    an = ax[1].annotate(
                        rf"$z={row["Z_NIRISS"]:.3f}$",
                        (0.95, 0.8),
                        ha="right",
                        va="top",
                        xycoords="axes fraction",
                        # rotation="vertical"
                    )
                    an = ax[1].annotate(
                        rf"$\rm{{H}}\alpha/\rm{{H}}\beta={row["flux_Ha"]/row["flux_Hb"]:.3f}$",
                        (0.95, 0.7),
                        ha="right",
                        va="top",
                        xycoords="axes fraction",
                        # rotation="vertical"
                    )
                    an = ax[1].annotate(
                        rf"$\rm{{O}}III/\rm{{H}}\alpha={row["flux_OIII"]/row["flux_Ha"]:.3f}$",
                        (0.95, 0.6),
                        ha="right",
                        va="top",
                        xycoords="axes fraction",
                        # rotation="vertical"
                    )
                    an = ax[1].annotate(
                        rf"$\rm{{O}}III/\rm{{O}}II={row["flux_OIII"]/row["flux_OII"]:.3f}$",
                        (0.95, 0.5),
                        ha="right",
                        va="top",
                        xycoords="axes fraction",
                        # rotation="vertical"
                    )

                    an = ax[1].annotate(
                        rf"{row["RA"]:.5f} {row["DEC"]:.5f}",
                        (0.95, 0.4),
                        ha="right",
                        va="top",
                        xycoords="axes fraction",
                        # rotation="vertical"
                    )

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

            # plt.savefig(save_dir / "test.pdf")
            pdf.savefig(1)
            plt.clf()
            # plt.show()

        pdf.close()
