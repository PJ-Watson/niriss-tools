"""
Display the full spectroscopic distribution.
"""

from astropy.table import Column, join, vstack
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    wht_path = (
        grizli_dir / "RAW" / "abell2744-01324-001-251.0-nis-f200w-gr150r_drz_sci.fits"
    )

    wht_img = fits.getdata(wht_path)
    sky_wcs = WCS(fits.getheader(wht_path))

    combined_ra = np.zeros(len(ext_cat))
    combined_dec = np.zeros(len(ext_cat))

    import numpy.ma as ma

    for i, row in enumerate(ext_cat[:]):
        # print (dir(row["RA"]))
        # print (ma.ismas)
        # print (row["RA"] is ma.masked)
        # if not row["RA"] is ma.masked:
        #     combined_ra[i] = row["RA"]
        #     combined_dec[i] = row["DEC"]
        # elif not row["ra_ground"] is ma.masked:
        #     combined_ra[i] = row["ra_ground"]
        #     combined_dec[i] = row["dec_ground"]
        if (not row["z_ground"] is ma.masked) and np.isfinite(row["z_ground"]):
            combined_ra[i] = row["ra_ground"]
            combined_dec[i] = row["dec_ground"]
        elif (not row["z_nirspec"] is ma.masked) and np.isfinite(row["z_nirspec"]):
            combined_ra[i] = row["ra_nirspec"]
            combined_dec[i] = row["dec_nirspec"]
        # elif not row["ra_nirspec"].mask:
        # else:
        elif (not row["z_alt"] is ma.masked) and np.isfinite(row["z_alt"]):
            combined_ra[i] = row["ra_alt"]
            combined_dec[i] = row["dec_alt"]
        elif (not row["Z_NIRISS"] is ma.masked) and np.isfinite(row["Z_NIRISS"]):
            combined_ra[i] = row["RA"]
            combined_dec[i] = row["DEC"]
        else:
            pass

    all_coords = SkyCoord(ra=combined_ra, dec=combined_dec, unit="deg")
    # x, y = wcs_utils.skycoord_to_pixel(all_coords, sky_wcs)
    # all_coords_int = np.round(np.array([y,x])).astype(int)
    # print (np.nansum(sky_mask[tuple(all_coords_int[:,:])]))
    # # print ("contains", sky_wcs.footprint_contains(all_coords))
    # print (all_coords_int)

    contained_idx = sky_wcs.footprint_contains(all_coords)

    wht_path = (
        grizli_dir / "RAW" / "abell2744-01324-001-251.0-nis-f200w-gr150c_drz_sci.fits"
    )
    wht_img = fits.getdata(wht_path)
    sky_wcs = WCS(fits.getheader(wht_path))

    contained_idx = contained_idx | sky_wcs.footprint_contains(all_coords)

    ext_cat = ext_cat[contained_idx]
    print(ext_cat)

    # exit()

    # for c in ext_cat.colnames:
    #     print (ext_cat[c][0].dtype)
    #     if np.issubdtype(ext_cat[c][0], np.integer):
    #         print ("int")
    #     if ext_cat[c].dtype is str:
    #         print ("s")

    # print (ext_cat)
    # print (glass_niriss_cat[extra_ids][20:25])
    # print (np.asarray(test_list))
    # with np.printoptions(threshold=np.inf):
    #     # print (np.array(glass_niriss_cat))
    #     for row in glass_niriss_cat[20:25]:
    #         print (row)

    fig, ax = plt.subplots(
        figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth / 3),
        constrained_layout=True,
    )

    # niriss_idx = (ext_cat["Z_FLAG_ALL"] >= 9).astype(bool)

    print(len(np.unique(ext_cat["id_nirspec"])))

    # niriss_idx = np.zeros(len(ext_cat), dtype=bool)
    # niriss_idx[(ext_cat["Z_FLAG"] >= 4) & (~ext_cat["Z_FLAG"].mask)] = (
    #     True
    # )
    # # _ns = np.where((~(ext_cat["Z_FLAG_ALL"] >= 9)) & (np.isfinite(ext_cat["z_nirspec"])) & (~ext_cat["z_nirspec"].mask))
    # nirspec_idx = np.zeros(len(ext_cat), dtype=bool)
    # nirspec_idx[
    #     np.where(
    #         (~niriss_idx)
    #         & (np.isfinite(ext_cat["z_nirspec"]))
    #         & (~ext_cat["z_nirspec"].mask)
    #     )
    # ] = True
    # nircam_idx = np.zeros(len(ext_cat), dtype=bool)
    # nircam_idx[
    #     np.where(
    #         (~niriss_idx)
    #         & (~nirspec_idx)
    #         & (np.isfinite(ext_cat["z_alt"]))
    #         & (~ext_cat["z_alt"].mask)
    #     )
    # ] = True
    # ground_idx = np.zeros(len(ext_cat), dtype=bool)
    # ground_idx[
    #     np.where(
    #         (~niriss_idx)
    #         & (~nirspec_idx)
    #         & (~nircam_idx)
    #         & (np.isfinite(ext_cat["z_ground"]))
    #         & (~ext_cat["z_ground"].mask)
    #     )
    # ] = True

    ground_idx = np.zeros(len(ext_cat), dtype=bool)
    ground_idx[
        np.where((np.isfinite(ext_cat["z_ground"])) & (~ext_cat["z_ground"].mask))
    ] = True
    nirspec_idx = np.zeros(len(ext_cat), dtype=bool)
    nirspec_idx[
        np.where(
            (np.logical_not(ground_idx))
            & (np.isfinite(ext_cat["z_nirspec"]))
            & (np.logical_not(ext_cat["z_nirspec"].mask))
        )
    ] = True
    nircam_idx = np.zeros(len(ext_cat), dtype=bool)
    nircam_idx[
        np.where(
            (np.logical_not(ground_idx))
            & (np.logical_not(nirspec_idx))
            & (np.isfinite(ext_cat["z_alt"]))
            & np.logical_not(ext_cat["z_alt"].mask)
        )
    ] = True
    niriss_idx = np.zeros(len(ext_cat), dtype=bool)
    niriss_idx[
        np.logical_not(ground_idx)
        & np.logical_not(nirspec_idx)
        & np.logical_not(nircam_idx)
        & (ext_cat["Z_FLAG"] >= 4)
        & np.logical_not(ext_cat["Z_FLAG"].mask)
        # & (ext_cat["Z_FLAG"]==1)
    ] = True
    # _ns = np.where((~(ext_cat["Z_FLAG_ALL"] >= 9)) & (np.isfinite(ext_cat["z_nirspec"])) & (~ext_cat["z_nirspec"].mask))

    # print (dir(nirspec_idx))
    # nirspec_idx.fill(False)
    # print (nirspec_idx)
    # print (len(np.unique(ext_cat["id_nirspec"][nirspec_idx])))

    # print (len(nirspec_cats))
    # nircam_idx = (~(ext_cat["Z_FLAG_ALL"] >= 9)) | (~np.isfinite(ext_cat["z_nirspec"]))
    # nircam_idx = np.isfinite(ext_cat["z_alt"]) & (~np.isfinite(ext_cat["z_nirspec"]))

    print(np.isfinite(ext_cat["z_nirspec"][-5:]))
    #  & np.isfinite(ext_cat["z_alt"])
    # print (ext_cat["z_alt"].mask)

    print(nirspec_idx)

    print(
        np.nansum(niriss_idx),
        np.nansum(nirspec_idx),
        np.nansum(nircam_idx),
        np.nansum(ground_idx),
        np.sum(~niriss_idx),
        np.sum(~nirspec_idx),
    )
    # exit()
    z_bins = np.arange(0, 10.1, 0.1)
    # ax.hist(
    #     [
    #         ext_cat[niriss_idx]["Z_EST_ALL"],
    #         ext_cat[nirspec_idx]["z_nirspec"],
    #         ext_cat[nircam_idx]["z_alt"],
    #         # nirspec_cats["z_nirspec"],
    #         # alt_cat["z_alt"],
    #     ],
    #     stacked=True,
    #     bins=z_bins,
    #     label=["NIRISS", "NIRSpec","NIRCam-WFSS", ]
    # )

    z_niriss = np.array(ext_cat[niriss_idx]["Z_NIRISS"])
    z_nirspec = np.array(ext_cat[nirspec_idx]["z_nirspec"])
    z_nircam = np.array(ext_cat[nircam_idx]["z_alt"])
    z_ground = np.array(ext_cat[ground_idx]["z_ground"])

    _, _, barlist = ax.hist(
        [
            z_niriss,
            z_ground,
            z_nirspec,
            z_nircam,
        ],
        bins=z_bins,
        stacked=True,
        histtype="stepfilled",
        color=[
            "dodgerblue",
            "lightcoral",
            "yellowgreen",
            "orange",
        ],
        edgecolor=[
            "blue",
            "darkred",
            "darkgreen",
            "chocolate",
        ],
        hatch=[None, "//", "X", "\\\\"],
        linewidth=1,
        zorder=-1,
        alpha=1.0,
        label=[
            "NIRISS",
            "Ground",
            "NIRSpec",
            "NIRCam-WFSS",
        ],
    )
    # ax.hist(
    #     np.concatenate([z_niriss, z_nircam, z_nirspec]),
    #     bins=z_bins,
    #     histtype="stepfilled",
    #     hatch="X",
    #     color="yellowgreen",
    #     edgecolor="darkgreen",
    #     ls="--",
    #     linewidth=1,
    #     zorder=1,
    #     label= "NIRSpec"
    # )
    # for bars in barlist:
    #     for bar in bars:
    #         bar.set_edgecolor(bar.get_facecolor())
    # plt.show()

    ax.legend(loc=1, reverse=True)
    from matplotlib.ticker import MultipleLocator

    ax.xaxis.set_major_locator(MultipleLocator(1.0))

    ax.set_xlim(z_bins[0], z_bins[-1])
    # ax_scatter.set_ylabel(r"$m_{\rm{F200W}}$")
    ax.set_xlabel(r"$z_{\rm{spec}}$")
    ax.set_ylabel(r"Number of Galaxies")
    ax.grid(color="k", alpha=0.05, zorder=-1)

    fig.patch.set_alpha(0.0)
    # ax.semilogy()

    plt.savefig(save_dir / "full_z_hist_rev1.pdf", dpi=600)

    plt.show()
    # exit()
    # fig, ax = plt.subplots(
    #     figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth / 1.62),
    #     constrained_layout=True,
    #     subplot_kw={"projection": "3d"},
    # )

    # # for cat in
    # ax.scatter(
    #     v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["RA"],
    #     v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["DEC"],
    #     v2_cat[v2_cat["Z_FLAG_ALL"] >= 9]["Z_EST_ALL"],
    # )
    # ax.scatter(nirspec_cats["ra"], nirspec_cats["dec"], nirspec_cats["z_spec"])
    # ax.scatter(alt_cat["ra"], alt_cat["dec"], alt_cat["z_alt"])

    # plt.show()

    img_path = grizli_dir / "Prep" / "glass-a2744-ir_drc_sci.fits"

    with fits.open(img_path) as img_hdul:
        img_wcs = WCS(img_hdul[0])

        fig, ax = plt.subplots(
            figsize=(plot_utils.aanda_textwidth, plot_utils.aanda_textwidth),
            constrained_layout=True,
            subplot_kw={"projection": img_wcs},
        )
        ax.imshow(
            img_hdul[0].data,
            norm=astrovis.ImageNormalize(
                img_hdul[0].data,
                # interval=astrovis.PercentileInterval(99.9),
                interval=astrovis.ManualInterval(-0.001, 10),
                stretch=astrovis.LogStretch(),
            ),
            cmap="binary_r",
            origin="lower",
            zorder=-1,
        )
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    plot_utils.sky_plot(
        ext_cat,
        niriss_idx,
        ax=ax,
        label="NIRISS",  # **plot_kwargs
    )
    plot_utils.sky_plot(
        ext_cat,
        nircam_idx,
        ax=ax,
        label="NIRCam WFSS",
        ra_col="ra_alt",
        dec_col="dec_alt",  # **plot_kwargs
    )
    plot_utils.sky_plot(
        ext_cat,
        nirspec_idx,
        ax=ax,
        label="NIRSpec",
        ra_col="ra_nirspec",
        dec_col="dec_nirspec",  # **plot_kwargs
    )
    plot_utils.sky_plot(
        ext_cat,
        ground_idx,
        ax=ax,
        label="Ground",
        ra_col="ra_ground",
        dec_col="dec_ground",  # **plot_kwargs
    )
    ax.legend()
    plt.show()
