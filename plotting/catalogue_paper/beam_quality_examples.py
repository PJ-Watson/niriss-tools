"""
Examples of the beam quality flags.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    check_filt = "F200W"
    check_angle = "72.0"
    niriss_cat = niriss_cat[
        (niriss_cat["Z_FLAG"] >= 4)
        & (niriss_cat[f"{check_filt}_{check_angle:0>5}_QUALITY"] == 0)
    ]
    with np.printoptions(threshold=np.inf):
        niriss_cat.pprint(len(niriss_cat) + 2)

    obj_list_full = [
        {2663: ["F200W", "341.0"], 2185: ["F115W", "72.0"]},
        {3028: ["F200W", "341.0"], 682: ["F200W", "341.0"]},
        {690: ["F200W", "341.0"], 882: ["F200W", "72.0"]},
    ]
    fig = plt.figure(
        1,
        constrained_layout=True,
    )
    fig.set_size_inches(
        plot_utils.aanda_textwidth * 0.8,
        plot_utils.aanda_columnwidth * len(obj_list_full) / 1.3,
    )
    subfigs = fig.subfigures(len(obj_list_full), 1)

    hatch_colour = "k"
    hatch_kwargs = {
        "facecolor": "k",
        "edgecolor": hatch_colour,
        "step": "mid",
        "alpha": 0.3,
        "hatch": "//",
        "linewidth": 1,
        "zorder": 3,
    }

    for o, obj_list in enumerate(obj_list_full):

        axs = subfigs[o].subplots(
            4,
            4,
            # figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_columnwidth / 1.1),
            # constrained_layout=True,
            # use_gridspec=True,
            sharex="col",
            # sharey="row",
            width_ratios=[1, 3, 1, 3],
            # height_ratios=[1, 1, 2],
        )
        for i, (obj_id, (obj_filter, pa)) in enumerate(obj_list.items()):
            try:
                obj_z = full_cat[np.where(full_cat["NUMBER"] == obj_id)]["Z_NIRISS"]
                # print(obj_z)
                # print ()
                # pas = ["72.0", "341.0"]
                # filts = ["F115W", "F150W", "F200W"]
                # pa = "72.0"
                # obj_filter = "F200W"
                # pa = "341.0"
                # print(pa, obj_filter)
                stack_path = [
                    *(grizli_dir).glob(
                        f"Extractions*/**/*glass-a2744_{obj_id:0>5}.stack.fits"
                    )
                ]
                stack_path.sort(reverse=True)
                print(stack_path)
                stack_path = stack_path[-1]

                with fits.open(stack_path) as stack_hdul:
                    extver = obj_filter + "," + pa
                    kern_i = stack_hdul["KERNEL", extver].data
                    vmax_kern = 1.1 * np.percentile(kern_i.data, 99.5)
                    norm_kern = astrovis.ImageNormalize(
                        data=kern_i,
                        stretch=astrovis.SqrtStretch(),
                        interval=astrovis.ManualInterval(
                            vmin=-0.1 * vmax_kern,
                            # vmin=0,
                            vmax=vmax_kern,
                        ),
                    )

                    # Kernel
                    sh_kern = kern_i.data.shape
                    extent_kern = [0, sh_kern[1], 0, sh_kern[0]]

                    for j in range(4):
                        axs[j, 2 * i].imshow(
                            kern_i,
                            origin="lower",
                            interpolation="none",
                            norm=norm_kern,
                            # vmin=-0.1 * vmax_kern,
                            # vmax=vmax_kern,
                            # cmap=cm.berlin,
                            cmap="plasma",
                            extent=extent_kern,
                            aspect="equal",
                        )
                        axs[j, 2 * i].set_xticklabels([])
                        axs[j, 2 * i].set_yticklabels([])

                    # print (f+","+p)
                    wht_i = stack_hdul["WHT", extver].data
                    sci_i = stack_hdul["SCI", extver].data
                    contam_i = stack_hdul["CONTAM", extver].data
                    model_i = stack_hdul["MODEL", extver].data
                    h_i = stack_hdul["SCI", extver].header

                    clip = wht_i > 0  # & (contam_i < 0.1 * sci_i)
                    if clip.sum() == 0:
                        clip = np.isfinite(wht_i)

                    avg_rms = 1 / np.median(np.sqrt(wht_i[clip]))

                    vmax = np.maximum(1.1 * np.percentile(sci_i[clip], 98), 5 * avg_rms)
                    vmin = -0.1 * vmax
                    norm = astrovis.ImageNormalize(
                        data=sci_i,
                        stretch=astrovis.SqrtStretch(),
                        interval=astrovis.ManualInterval(
                            vmin=vmin,
                            vmax=vmax,
                        ),
                    )
                    # vmin = -0.005
                    # vmax = 0.5
                    for k, (data, data_label) in enumerate(
                        zip(
                            [sci_i, contam_i, model_i, sci_i - model_i],
                            ["SCI", "CONTAM", "MODEL", "RESID"],
                        )
                    ):

                        # Spectrum
                        sh = sci_i.shape
                        extent = [h_i["WMIN"], h_i["WMAX"], 0, sh[0]]
                        # print(extent)

                        axs[k, (2 * i) + 1].imshow(
                            # sci_i,
                            data,
                            origin="lower",
                            interpolation="none",
                            # vmin=-0.1 * vmax,
                            # vmax=vmax,
                            norm=norm,
                            # cmap=cm.berlin,
                            cmap="plasma",
                            extent=extent,
                            aspect="auto",
                        )
                        # axs[j, i].set_xticklabels([])
                        axs[k, (2 * i) + 1].set_yticklabels([])
                        axs[k, (2 * i) + 1].annotate(
                            data_label,
                            (0.97, 0.85),
                            ha="right",
                            va="top",
                            xycoords="axes fraction",
                            bbox=dict(
                                facecolor=(1, 1, 1, 0.9),
                                edgecolor="k",
                                boxstyle="round",
                            ),
                        )
                    axs[3, (2 * i) + 1].set_xlabel(r"Observed Wavelength ($\mu$m)")
                    axs[0, (2 * i) + 1].set_title(
                        rf"({chr(i+97+2*o)})\ \ ID: {obj_id};\quad$z={obj_z.value[0]:.3f}$"
                    )
            except Exception as e:
                print(e)
                pass
    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "beam_quality_examples.pdf", dpi=600)

    plt.show()
