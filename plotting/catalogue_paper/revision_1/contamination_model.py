"""
A default plotting script.
"""

import cmcrameri.cm as cmc
import matplotlib as mpl
import plot_utils
from default_imports import *

mpl.use("TkAgg")

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    # print(niriss_cat)
    # print (grizli_dir)

    # fig, axs = plt.subplots(
    #     4,
    #     2,
    #     figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth),
    #     constrained_layout=True,
    #     sharex=True,
    #     sharey=True,
    #     # hspace=0.,
    #     # wspace=0.,
    # )
    fig = plt.figure(
        figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth * 0.925),
        constrained_layout=True,
    )
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 0.1, 2.5))
    left_sub = gs[0].subgridspec(18, 1)
    axs_direct = fig.add_subplot(left_sub[2:8])
    axs_grism = fig.add_subplot(left_sub[9:15], sharex=axs_direct, sharey=axs_direct)
    # axs = right_sub.subplots(sharex=True,sharey=True)
    right_sub = gs[2].subgridspec(3, 2)
    axs = right_sub.subplots(sharex=True, sharey=True)
    plot_kwargs = {"cmap": cmc.lajolla, "origin": "lower"}

    new_name = "grizli_home_refine_contam"
    old_name = "grizli_home_refine_contam_old_seg"
    refine_dir = grizli_dir.parent / new_name
    refine_dir = grizli_dir
    files = (refine_dir / "Prep").glob("*FLT.fits")
    flt_file_refined = sorted(files, key=lambda file: Path(file).lstat().st_mtime)[12]
    flt_file_orig = grizli_dir.parent / old_name / "Prep" / flt_file_refined.name

    # print (flt_file_orig, flt_file_refined)
    # exit()
    l_crop = 3648 // 2 - 2048 // 2
    r_crop = 3648 // 2 + 2048 // 2
    crop = (slice(l_crop, r_crop), slice(l_crop, r_crop))

    with fits.open(flt_file_refined) as flt_hdul:
        # plt.imshow(flt_hdul["GSCI"].data[crop]>0)
        # plt.show()
        # exit()
        flt_hdul.info()
        norm = astrovis.ImageNormalize(
            data=flt_hdul["GSCI"].data[crop],
            interval=astrovis.ManualInterval(-0.0, 10),
            stretch=astrovis.LogStretch(),
        )
        axs_direct.imshow(
            flt_hdul["DREF"].data[crop] / flt_hdul["DREF"].header["PHOTFLAM"],
            norm=norm,
            **plot_kwargs,
        )
        grism = flt_hdul["GSCI"].header["CONFFILE"].split("CONF/")[1].split(".")[0]
        filt = flt_hdul["DREF"].header["DFILTER"].split("-")[0]
        axs_direct.set_title(rf"Direct Imaging ({filt})")
        axs_grism.set_title(rf"Grism Data ({grism})")
        axs_grism.imshow(flt_hdul["GSCI"].data[crop], norm=norm, **plot_kwargs)
        axs_direct.set_xticklabels([])
        axs_direct.set_yticklabels([])

    for f_i, (flt_file, label) in enumerate(
        zip([flt_file_orig, flt_file_refined], ["Default Parameters", "This Analysis"])
    ):
        # try:
        with fits.open(flt_file) as flt_hdul:
            flux_scale_factor = 1e-20
            flux_scale = flt_hdul["DREF"].header["PHOTFLAM"] / flux_scale_factor
            norm = astrovis.ImageNormalize(
                data=flt_hdul["GSCI"].data[crop] * flux_scale,
                interval=astrovis.ManualInterval(-0.0 * flux_scale, 10 * flux_scale),
                stretch=astrovis.LogStretch(),
            )
            if f_i == 1:
                from grizli import multifit

                _flt = multifit.GroupFLT(
                    grism_files=[str(flt_file)],
                    # grism_files=flt_files,
                    catalog="/media/sharedData/data/2024_08_16_A2744_v4/grizli_home/Prep/glass-a2744-ir.cat.fits",
                    cpu_count=-1,
                    sci_extn=1,
                    pad=800,
                )
                print(dir(_flt.FLTs[0]))
                _flt.FLTs[0].process_seg_file(
                    "/media/sharedData/data/2024_08_16_A2744_v4/grizli_home/ForcedExtractions/glass-a2744-ir_seg_5_renumbered_test_2.fits",
                    # segmentation=True,
                )
                seg_data = np.rot90(_flt.FLTs[0].seg[crop])
                # seg_data = fits.getdata(ext="SEG")[crop]
            else:
                seg_data = flt_hdul["SEG"].data[crop]
            seg_plot = np.full_like(seg_data, np.nan)
            print(seg_data.shape, seg_plot.shape)
            seg_plot[seg_data > 0] = (seg_data % 7 + 1)[seg_data > 0]
            print(np.nansum(seg_data > 0))
            axs[0, f_i].imshow(seg_plot, cmap="jet", origin="lower")
            axs[0, f_i].set_facecolor("k")
            axs[0, f_i].set_title(label)
            fl = axs[1, f_i].imshow(
                flt_hdul["MODEL"].data[crop] * flux_scale, norm=norm, **plot_kwargs
            )
            # for i, (a, hdu_id) in enumerate(zip(axs[1:,0], [""SEG", "MODEL"])):
            #     a.imshow(flt_hdul[hdu_id].data[crop], norm=norm, **plot_kwargs)

            if f_i == 1:
                cb = plt.colorbar(
                    fl,
                    ax=axs[1, f_i],
                    label=rf"$10^{{{int(np.log10(flux_scale_factor))}}}\ \ \rm{{erg\,/\,s\,/\,cm^{{2}}\,/\,\AA}}$",
                    # ticks=[-1, 0, 0.1, 0.2, 0.5, 1, 2,5]
                )
                # log_range = np.log10(0), np.log10(norm.vmax)
                # log_range = 10**norm.vmin, 10**norm.vmax
                # print (log_range)
                # new_ticks = 10**(np.linspace(*log_range, 10))
                # print (new_ticks)
                # print (np.log10(np.linspace(norm.vmin, norm.vmax)))
                # print (cb.get_ticks())
                # print (cb.ax.get_yticklabels())
                # cb.set_ticks([0, 0.1, 0.2, 0.5, 1, 2])

                # # this part is essential for picking quasi-reasonable values.
                # rounded = [np.format_float_positional(x, 2, unique=False, fractional=False, trim='k') for x in new_ticks]
                # cb.set_ticks(list(map(float, rounded)))
                # cb.set_ticklabels([x.rstrip('.') for x in rounded])

            resid_scale_factor = 1e-22
            resid = (
                (flt_hdul["GSCI"].data - flt_hdul["MODEL"].data)
                * flt_hdul["DREF"].header["PHOTFLAM"]
                / resid_scale_factor
            )[crop]
            print(flt_hdul["DREF"].header["PHOTFLAM"])
            im = axs[2, f_i].imshow(
                resid,
                norm=astrovis.ImageNormalize(
                    data=resid,
                    interval=astrovis.ManualInterval(
                        -0.15
                        * flt_hdul["DREF"].header["PHOTFLAM"]
                        / resid_scale_factor,
                        0.15 * flt_hdul["DREF"].header["PHOTFLAM"] / resid_scale_factor,
                    ),
                    stretch=astrovis.LinearStretch(),
                ),
                origin="lower",
                cmap=cmc.vik,
            )

            axs[0, f_i].set_xticklabels([])
            axs[0, f_i].set_yticklabels([])
            if f_i == 1:
                # plt.colorbar(im, ax=axs[2, f_i], label=r"$\rm{Data}-\rm{Model}\ \left[\rm{erg\,/\,s\,/\,cm^{2}\,/\,\AA}\right]$")
                plt.colorbar(
                    im,
                    ax=axs[2, f_i],
                    label=rf"$10^{{{int(np.log10(resid_scale_factor))}}}\ \ \rm{{erg\,/\,s\,/\,cm^{{2}}\,/\,\AA}}$",
                )
                # plt.colorbar(im, ax=axs[2, f_i], label=rf"$\rm{{erg\,/\,s\,/\,cm^{{2}}\,/\,\AA}}$")
        axs[0, 0].set_ylabel(r"Segmentation Map")
        axs[1, 0].set_ylabel(r"Contamination Model")
        axs[2, 0].set_ylabel(r"Residuals")
    # except:
    #     pass

    # with fits.open(flt_file_orig) as flt_hdul:
    #     # plt.imshow(flt_hdul["GSCI"].data[crop]>0)
    #     # plt.show()
    #     # exit()
    #     # flt_hdul.info()
    #     norm = astrovis.ImageNormalize(
    #         data=flt_hdul["GSCI"].data[crop],
    #         interval=astrovis.ManualInterval(-0.01, 10),
    #         stretch=astrovis.LogStretch(),
    #     )
    #     for i, (a, hdu_id) in enumerate(zip(axs.ravel()[4:], ["GSCI", "SEG", "MODEL"])):
    #         a.imshow(flt_hdul[hdu_id].data[crop], norm=norm, **plot_kwargs)

    #     resid = (flt_hdul["GSCI"].data - flt_hdul["MODEL"].data)[crop]
    #     axs.ravel()[-1].imshow(
    #         resid,
    #         norm=astrovis.ImageNormalize(
    #             data=resid,
    #             interval=astrovis.ManualInterval(-0.2, 0.2),
    #             stretch=astrovis.LinearStretch(),
    #         ),
    #         origin="lower",
    #         cmap=cmc.vik,
    #     )

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

    plt.savefig(save_dir / "contamination_model_comparison_rev1.pdf", dpi=300)

    plt.show()
