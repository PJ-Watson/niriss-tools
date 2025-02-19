"""
Show the distribution of galaxies on sky.
"""

import plot_utils
from default_imports import *
from regions import RectangleSkyRegion, Regions
from sregion import SRegion

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    # Import MAST
    # from astroquery.mast import Mast
    # from sregion import SRegion

    # # Enter one query parameter:
    # service = "Mast.Jwst.Filtered.Nirspec"
    # one_parameter = {"columns": "*",
    #             "filters": [{"paramName": "program",
    #                             "values": ['1324']
    #                             }]
    #             }
    # response_one_parameter = Mast.service_request_async(service, one_parameter)
    # results_one_parameter = response_one_parameter[0].json()['data']
    # # Print the first ten filenames
    # reg=None
    # for result_one_parameter in results_one_parameter[:10]:
    #     # print (result_one_parameter["gs_ra"])
    #     if "cal" in result_one_parameter["filename"]:
    #         print (result_one_parameter["s_region"])
    #         r = SRegion(result_one_parameter["s_region"])
    #         print (r.region)
    #         r = Regions.parse("ICRS\n"+r.region[0], format="ds9").regions
    #         # print (dir(r))
    #         # print (r)
    #         # pixel_region = r.to_pixel(new_wcs)
    #         # # pixel_region.plot(ax=ax, edgecolor="none", facecolor="tab:green", alpha=0.1, fill=True)
    #         # if reg is None:
    #         #     reg = pixel_region.to_mask().to_image(new_shape)  # data
    #         # else:
    #         #     reg += pixel_region.to_mask().to_image(new_shape)  # .data

    # exit()

    from grizli import utils as grizli_utils

    ## Abell2744
    ra, dec = 3.58641, -30.39997

    sep = 20  # search radius, arcmin

    QUERY_URL = "https://grizli-cutout.herokuapp.com/assoc?coord={ra},{dec}&arcmin={sep}&output=csv"

    assoc_query = grizli_utils.read_catalog(
        QUERY_URL.format(ra=ra, dec=dec, sep=sep), format="csv"
    )
    # print (np.unique(assoc_query["instrument_name"]))
    nir = assoc_query["instrument_name"] == "NIRCAM"
    # nir_3538 = (assoc_query['proposal_id'] == 3538) & (assoc_query["filter"] == "F300M-GRISMR")
    nir_3538 = (assoc_query["proposal_id"] == 3538) & (
        (assoc_query["filter"] == "F335M-GRISMR")
        | (assoc_query["filter"] == "F300M-GRISMR")
        | (assoc_query["filter"] == "F410M-GRISMR")
        | (assoc_query["filter"] == "F460M-GRISMR")
    )
    nir_2883 = (assoc_query["proposal_id"] == 2883) & (
        (assoc_query["filter"] == "F360M-GRISMC")
        | (assoc_query["filter"] == "F480M-GRISMC")
    )
    nir_3516 = (assoc_query["proposal_id"] == 3516) & (
        (assoc_query["filter"] == "F356W-GRISMR")
    )

    nir_1324 = (assoc_query["proposal_id"] == 1324) & (
        (assoc_query["filter"] == "GR150R-F115W")
        | (assoc_query["filter"] == "GR150R-F150W")
        | (assoc_query["filter"] == "GR150R-F200W")
    )

    img_path = (
        root_dir
        # / "2024_08_16_A2744_v4"
        # / "grizli_home"
        # / "Prep"
        # / "glass-a2744-ir_drc_sci.fits"
        / "archival"
        / "grizli-v2"
        / "JwstMosaics"
        / "v7"
        / "abell2744clu-grizli-v7.2-f200w-clear_drc_sci.fits"
    )

    with fits.open(img_path) as img_hdul:
        img_wcs = WCS(img_hdul[0])

        fig, ax = plt.subplots(
            figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth * 0.8),
            constrained_layout=True,
            subplot_kw={"projection": img_wcs},
        )
        # fig = plt.figure(
        #     figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth),
        #     constrained_layout=True,
        # )
        # fig.set_figwidth(plot_utils.aanda_columnwidth)
        # ax = fig.add_subplot(1, 1, 1, projection=img_wcs)
        # ax.imshow(
        #     img_hdul[0].data,
        #     norm=astrovis.ImageNormalize(
        #         img_hdul[0].data,
        #         # interval=astrovis.PercentileInterval(99.9),
        #         interval=astrovis.ManualInterval(-0.001, 10),
        #         stretch=astrovis.LogStretch(),
        #     ),
        #     cmap="binary",
        #     origin="lower",
        # )
        # img_shape = img_hdul[0].data.shape
        from reproject import reproject_interp

        from glass_niriss.isophotal import gen_new_wcs

        new_wcs, new_shape = gen_new_wcs(img_path, resolution=1.0, padding=0.0)

        new_img = reproject_interp(
            img_hdul[0], new_wcs, shape_out=new_shape, return_footprint=False
        )
        ax.imshow(
            new_img,
            norm=astrovis.ImageNormalize(
                new_img,
                # interval=astrovis.PercentileInterval(99.9),
                interval=astrovis.ManualInterval(-0.001, 10),
                stretch=astrovis.LogStretch(),
            ),
            cmap="binary",
            origin="lower",
        )
    # img_wcs
    # plt.show()

    import matplotlib

    legend_handles = []
    legend_labels = []

    for n_i, (n_idx, lab) in enumerate(
        zip(
            [nir_3516, nir_3538, nir_2883, nir_1324],
            ["ALT", "GO-3538", "MAGNIF", "GLASS"],
        )
    ):
        # if n_i!=3:
        #     continue
        reg = None
        for q in assoc_query["assoc_name"][n_idx][:]:
            assoc_info = grizli_utils.read_catalog(
                f"https://grizli-cutout.herokuapp.com/assoc_json?name={q}",
                format="pandas.json",
            )
            for a in assoc_info:
                r = RectangleSkyRegion(
                    SkyCoord(ra=a["ra"], dec=a["dec"], unit="deg"),
                    129 * u.arcsec,
                    129 * u.arcsec,
                    angle=a["orientat"] * u.deg,
                )
                pixel_region = r.to_pixel(new_wcs)
                # pixel_region.plot(ax=ax, edgecolor="none", facecolor="tab:green", alpha=0.1, fill=True)
                if reg is None:
                    reg = pixel_region.to_mask().to_image(new_shape)  # data
                else:
                    reg += pixel_region.to_mask().to_image(new_shape)  # .data

            # reg = np.clip(reg, a_min=0, a_max=3)
        reg = np.clip(reg, a_min=0, a_max=1)
        # reg[reg == 0] = np.nan

        # ax.imshow(reg, alpha=0.7, cmap=cmap)
        # print (reg.dtype)
        cs = ax.contour(
            reg,
            levels=[
                0.5,
            ],
            origin="lower",
            zorder=10,
            # cmap="RdYlBu"
            colors=f"C{n_i}",
            linestyles="-" if lab == "GLASS" else "--",
            linewidths=1.0,
            # label=lab
        )
        legend_handles.append(cs.legend_elements()[0][0])
        legend_labels.append(rf"{lab}")

    info_2756 = Table()
    # info_2756["ra"] = [3.618268,3.558250,3.606292,3.597144, 3.572696,3.604204,3.570270,3.581209]
    # info_2756["dec"] = [-30.429608,3.424366,3.418043,3.402225, 3.409550,3.385645,3.387376,3.421064]
    info_2756["ra"] = [3.5889112499999998, 3.5829014583333336]
    info_2756["dec"] = [-30.412497222222214, -30.413124444444463]
    info_2756["orientat"] = [168.19787476937464, 168.20090794192197]

    info_1324 = Table()
    info_1324["ra"] = [
        3.59532875,
    ]
    info_1324["dec"] = [-30.403127777777797]
    info_1324["orientat"] = 181.76845029046

    info_2561 = Table()
    info_2561["ra"] = [
        3.583912791666667,
        3.6084097916666664,
        3.5732805,
        3.5586419166666667,
        3.5808445,
        3.5803515833333335,
        3.5808445,
        3.584232083333333,
        3.5766530000000003,
    ]
    info_2561["dec"] = [
        -30.399861111111136,
        -30.391133611111115,
        -30.368674999999996,
        -30.356406666666658,
        -30.372304999999983,
        -30.37216361111109,
        -30.372304999999983,
        -30.404650833333335,
        -30.3700077777778,
    ]
    info_2561["orientat"] = [
        44.571144852715825,
        44.55877295571023,
        44.576587580973445,
        44.58401711024244,
        44.572754218239695,
        44.57300382601669,
        44.572754218239695,  #
        50.62685694972375,
        50.630761223240015,
    ]

    for i, (info, lab) in enumerate(
        zip([info_2756, info_1324, info_2561], ["DDT-2756", "GLASS", "UNCOVER"])
    ):
        reg = None
        for a in info:
            r = RectangleSkyRegion(
                SkyCoord(ra=a["ra"], dec=a["dec"], unit="deg"),
                216 * u.arcsec,
                204 * u.arcsec,
                angle=a["orientat"] * u.deg,
            )
            pixel_region = r.to_pixel(new_wcs)
            # pixel_region.plot(ax=ax, edgecolor="none", facecolor="tab:green", alpha=0.1, fill=True)
            if reg is None:
                reg = pixel_region.to_mask().to_image(new_shape)  # data
            else:
                reg += pixel_region.to_mask().to_image(new_shape)  # .data

            # reg = np.clip(reg, a_min=0, a_max=3)
        reg = np.clip(reg, a_min=0, a_max=1)
        cs = ax.contour(
            reg,
            levels=[
                0.5,
            ],
            origin="lower",
            zorder=10,
            # cmap="RdYlBu"
            colors=f"C{i+4}",
            linestyles=":",
            linewidths=1.0,
            # label=rf"{lab}"
        )
        legend_handles.append(cs.legend_elements()[0][0])
        legend_labels.append(rf"{lab}")

    # # Import MAST
    # from astroquery.mast import Mast

    # # Enter one query parameter:
    # service = "Mast.Jwst.Filtered.Nirspec"
    # one_parameter = {"columns": "*",
    #             "filters": [{"paramName": "program",
    #                             "values": ['1324']
    #                             }]
    #             }
    # response_one_parameter = Mast.service_request_async(service, one_parameter)
    # results_one_parameter = response_one_parameter[0].json()['data']
    # # Print the first ten filenames
    # reg=None
    # for result_one_parameter in results_one_parameter[:]:
    #     # print (result_one_parameter["gs_ra"])
    #     if result_one_parameter["productLevel"]!="2b":
    #         continue
    #     try:
    #         # print (result_one_parameter["s_region"])
    #         r = SRegion(result_one_parameter["s_region"])
    #         # print (r.region)
    #         r = Regions.parse("ICRS\n"+r.region[0], format="ds9").regions[0]
    #         # print (r)
    #         pixel_region = r.to_pixel(new_wcs)
    #         # print (pixel_region)
    #         # pixel_region.plot(ax=ax, edgecolor="none", facecolor="tab:green", alpha=0.1, fill=True)
    #         # print (np.nansum(pixel_region.to_mask().to_image(new_shape)))
    #         if np.nansum(pixel_region.to_mask().to_image(new_shape))>=1:
    #             print (result_one_parameter["filename"],np.nansum(pixel_region.to_mask().to_image(new_shape)) , result_one_parameter)
    #         if reg is None:
    #             reg = pixel_region.to_mask().to_image(new_shape)  # data
    #         else:
    #             reg += pixel_region.to_mask().to_image(new_shape)  # .data
    #     except:
    #         pass

    # reg = np.clip(reg, a_min=0, a_max=1)
    # ax.contour(
    #     reg,
    #     levels=[0.5,],
    #     origin="lower",
    #     zorder=10,
    #     # cmap="RdYlBu"
    #     # colors=cmap(1.)
    # )
    # # ax.imshow(reg, alpha=0.7)
    # # print (reg.dtype)

    # for a in ax.coords:
    ax.coords[0].set_major_formatter("d.ddd")
    ax.coords[1].set_major_formatter("d.ddd")
    # Why for the love of God does astropy override basic
    # matplotlib parameters?
    # I just want to rotate the tick labels and keep them aligned
    # ax.coords[1].set_ticklabel(
    #     rotation="vertical",
    #     rotation_mode="default",
    #     pad=0,
    #     verticalalignment="center",
    #     horizontalalignment="center",
    # )
    ax.set_xlabel(r"R.A.")
    ax.set_ylabel(r"Dec.")

    # import matplotlib.patches as mpatches

    # patches = []
    # for cmap, name in zip(
    #     [cmap_1324, cmap_2883, cmap_3538, cmap_3516],
    #     ["GLASS-JWST", "MAGNIF", "GO-3538", "ALT"],
    # ):

    #     patches.append(mpatches.Patch(color=[*cmap(1.0)[:-1], 0.5], label=name))
    # # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    # ax.legend(handles=patches, loc=2, fontsize=7)
    ax.legend(legend_handles, legend_labels, loc=4)

    fig.patch.set_alpha(0.0)

    plt.savefig(save_dir / "pointings.pdf", dpi=600)

    # plt.show()
