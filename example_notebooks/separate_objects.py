"""
Separate blended objects.
"""

import os
from functools import partial
from pathlib import Path

import astropy.units as u
import astropy.visualization as astrovis
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from cmcrameri import cm
from numpy.typing import ArrayLike

root_dir = Path(os.getenv("ROOT_DIR"))
save_dir = root_dir / "2024_08_16_A2744_v4" / "catalogue_paper" / "plots"
niriss_dir = root_dir / "2024_08_16_A2744_v4" / "glass_niriss"
grizli_dir = root_dir / "2024_08_16_A2744_v4" / "grizli_home"

compiled_dir = (
    root_dir
    / "2024_08_16_A2744_v4"
    / "grizli_home"
    / "classification-stage-2"
    / "catalogues"
    / "compiled"
)
full_cat_name = "internal_full_6.fits"
full_cat = Table.read(compiled_dir / full_cat_name)

niriss_filter_sens = {
    "F115W": [1.0130, 1.2830],
    "F150W": [1.3300, 1.6710],
    "F200W": [1.7510, 2.2260],
}

if __name__ == "__main__":

    force_extract_dir = grizli_dir / "ForcedExtractions"

    init_seg_map = fits.getdata(force_extract_dir / "glass-a2744-ir_seg_1_merged.fits")

    ids_to_deblend = [741, 1129, 3310, 3315]
    mask_others = ~np.isin(init_seg_map, ids_to_deblend)  # & (init_seg_map!=0)
    # plt.imshow(mask_others)
    # plt.show()

    detect_img = fits.getdata(
        grizli_dir / "Prep" / "glass-a2744-ir_drc_sci.fits"
    ).astype(float)
    # # plt.imshow(detect_img, origin="lower")
    # # plt.show()
    # print (detect_img.dtype)
    # detect_img = detect_img.view(detect_img.dtype.newbyteorder())
    # print (detect_img.dtype)
    # plt.imshow(detect_img, origin="lower")
    # plt.show()
    # exit()
    detect_wht = fits.getdata(
        grizli_dir / "Prep" / "glass-a2744-ir_drc_wht.fits"
    ).astype(float)
    # detect_wht = detect_wht.view(detect_wht.dtype.newbyteorder("="))
    err = 1 / detect_wht**0.5
    err[err <= 0] = 0.0

    import sep

    cat, new_seg = sep.extract(
        detect_img,
        thresh=5,
        err=err,
        mask=mask_others,
        segmentation_map=True,
        deblend_cont=1e-3,
    )

    overall_seg = np.zeros_like(new_seg, dtype=int)

    overall_seg[np.isin(new_seg, [1, 2])] = new_seg[np.isin(new_seg, [1, 2])]

    mask_others |= overall_seg > 1

    cat, new_seg = sep.extract(
        detect_img,
        thresh=5,
        err=err,
        mask=mask_others,
        segmentation_map=True,
        deblend_cont=1e-3,
        deblend_nthresh=128,
    )

    overall_seg[np.isin(new_seg, [2, 3, 4, 7])] = np.max(overall_seg) + 1
    overall_seg[np.isin(new_seg, [5, 6, 8])] = np.max(overall_seg) + 1

    mask_others |= overall_seg > 1

    cat, new_seg = sep.extract(
        detect_img,
        thresh=1,
        err=err,
        mask=mask_others,
        segmentation_map=True,
        deblend_cont=1e-4,
        deblend_nthresh=128,
        filter_kernel=None,
        minarea=9,
    )

    overall_seg[np.isin(new_seg, [27])] = np.max(overall_seg) + 1
    overall_seg[np.isin(new_seg, [28, 29])] = np.max(overall_seg) + 1

    # print (cat)
    # print (dir(cat))
    # print (Table(cat))
    # plt.imshow(overall_seg, origin="lower")
    # plt.show()

    from scipy import interpolate as it
    from scipy import ndimage as nd

    mask_orig = np.isin(init_seg_map, ids_to_deblend)
    mask_current = overall_seg > 0
    xy = np.where(mask_current)
    interp = it.NearestNDInterpolator(np.transpose(xy), overall_seg[xy])
    B = interp(*np.indices(overall_seg.shape))

    init_seg_map[mask_orig] = B[mask_orig] + np.max(init_seg_map)

    with fits.open(force_extract_dir / "glass-a2744-ir_seg_1_merged.fits") as seg_hdul:
        seg_hdul[0].data = init_seg_map.astype(seg_hdul[0].data.dtype)
        seg_hdul.writeto(force_extract_dir / "glass-a2744-ir_seg_2_deblended.fits")

    plt.imshow(init_seg_map, origin="lower")
    plt.show()

    # catalogue_dir = (
    #     root_dir
    #     / "2024_08_16_A2744_v4"
    #     / "glass_niriss"
    #     / "match_catalogues"
    #     / "classification_v1"
    # )

    # v2_out_dir = (
    #     root_dir
    #     / "2024_08_16_A2744_v4"
    #     / "grizli_home"
    #     / "classification-stage-2"
    #     / "catalogues"
    #     / "compiled"
    # )

    # new_cat_name = "internal_full_3.fits"
    # v2_cat = Table.read(v2_out_dir / new_cat_name)

    # img_path = (
    #     root_dir
    #     / "2024_08_16_A2744_v4"
    #     / "grizli_home"
    #     / "Prep"
    #     / "glass-a2744-ir_drc_sci.fits"
    # )

    # # v1_cat = Table.read(catalogue_dir / "compiled_catalogue_v1.fits")

    # with fits.open(img_path) as img_hdul:
    #     img_wcs = WCS(img_hdul[0])

    #     fig, ax = plt.subplots(
    #         figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth),
    #         constrained_layout=True,
    #         subplot_kw={"projection": img_wcs},
    #     )
    #     ax.imshow(
    #         img_hdul[0].data,
    #         norm=astrovis.ImageNormalize(
    #             img_hdul[0].data,
    #             # interval=astrovis.PercentileInterval(99.9),
    #             interval=astrovis.ManualInterval(-0.001, 10),
    #             stretch=astrovis.LogStretch(),
    #         ),
    #         cmap="binary",
    #         origin="lower",
    #     )
    # # sky = partial(sky.plot_hist, bins=np.arange(16.5,33.5,0.5), ax=ax)
    # # print (np.nanmin(v1_cat["MAG_AUTO"]), np.nanmax(v1_cat["MAG_AUTO"])

    # # sky_plot(v1_cat, v1_cat["V1_CLASS"] == 0, ax=ax, label="Full Sample")
    # # sky_plot(
    # #     v1_cat,
    # #     (v1_cat["V1_CLASS"] > 0) & (v1_cat["V1_CLASS"] < 4),
    # #     ax=ax,
    # #     label="Extracted",
    # # )
    # # sky_plot(v1_cat, (v1_cat["V1_CLASS"] == 4), ax=ax, label="First Pass")
    # # sky_plot(v1_cat, v1_cat["V1_CLASS"] >= 5, ax=ax, label="Placeholder")

    # # sky_plot(
    # #     v2_cat,
    # #     (v2_cat["Z_FLAG_ALL"] >= 9) & (~np.isfinite(v2_cat["zspec"])),
    # #     ax=ax,
    # #     label="Secure",
    # # )

    # for i, (label, z_range, icon, size) in enumerate(
    #     zip(
    #         ["1.10", "1.34", "1.87"],
    #         [[1.09, 1.11], [1.32, 1.37], [1.85, 1.9]],
    #         ["x", "*", "o"],
    #         [15, 30, 15],
    #     )
    # ):
    #     plot_utils.sky_plot(
    #         v2_cat,
    #         (
    #             (v2_cat["Z_FLAG_ALL"] >= 9)
    #             # & (~np.isfinite(v2_cat["zspec"]))
    #             & (v2_cat["Z_EST_ALL"] >= z_range[0])
    #             & (v2_cat["Z_EST_ALL"] < z_range[-1])
    #         ),
    #         ax=ax,
    #         label=label,
    #         s=size,
    #         marker=icon,
    #         edgecolor="none",
    #     )
    #     print(
    #         z_range,
    #         np.nanmedian(
    #             v2_cat["Z_EST_ALL"][
    #                 (v2_cat["Z_FLAG_ALL"] >= 9)
    #                 # & (~np.isfinite(v2_cat["zspec"]))
    #                 & (v2_cat["Z_EST_ALL"] >= z_range[0])
    #                 & (v2_cat["Z_EST_ALL"] < z_range[-1])
    #             ]
    #         ),
    #     )

    #     # test_cat = v2_cat

    # ax.set_xlabel(r"R.A.")
    # ax.set_ylabel(r"Dec.")
    # ax.legend()

    # # plt.savefig(save_dir / "sample_sky_plot.pdf", dpi=600)

    # fig.patch.set_alpha(0.0)

    # plt.savefig(save_dir / "overdensities_sky_plot.pdf", dpi=600)
    # plt.savefig(save_dir / "svg" / "overdensities_sky_plot.svg", dpi=600)

    # plt.show()

    # # z_range = [1.32, 1.37]
    # # # z_range = [1.85,1.9]

    # # for r in v2_cat["ID", "RA", "DEC"][
    # #     (v2_cat["Z_FLAG_ALL"] >= 9)  # & (~np.isfinite(v2_cat["zspec"])
    # #     & (v2_cat["Z_EST_ALL"] >= z_range[0])
    # #     & (v2_cat["Z_EST_ALL"] < z_range[-1])
    # # ]:
    # #     print(r[0], r[1], r[2])
