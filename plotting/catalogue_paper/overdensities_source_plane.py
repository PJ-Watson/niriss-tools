"""
Display the source plane distribution of galaxies in overdensities.
"""

import plot_utils
from astropy import cosmology
from default_imports import *
from scipy.stats import gaussian_kde

cosmo = cosmology.default_cosmology.get()
import matplotlib as mpl
from matplotlib import gridspec

mpl.use("TkAgg")

plot_utils.setup_aanda_style()

if __name__ == "__main__":

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

    img_path = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Prep"
        / "glass-a2744-ir_drc_sci.fits"
    )

    # v1_cat = Table.read(catalogue_dir / "compiled_catalogue_v1.fits")

    full_lens_tab = Table.read(
        niriss_dir / "match_catalogues" / "CG_PB_z_niriss_lensing.csv"
    )

    with fits.open(img_path) as img_hdul:
        img_wcs = WCS(img_hdul[0])
        fp = img_wcs.calc_footprint()
        img_extent = np.array(
            [fp[:, 0].min(), fp[:, 0].max(), fp[:, 1].min(), fp[:, 1].max()]
        )
        # print (img_extent)

        # print (img_wcs.calc_footprint().min())
        # exit()

        fig = plt.figure(
            figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth * 1.1),
            constrained_layout=True,
        )
        gs = gridspec.GridSpec(
            2, 2, width_ratios=[1, 8], height_ratios=[1, 30], figure=fig
        )

        gs1 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[1, :])

        axs = gs1.subplots(
            sharex=True,
            sharey=True,
            subplot_kw={"projection": img_wcs},
        )

        norm = mpl.colors.Normalize(vmin=10, vmax=100)
        # norm = mpl.colors.Normalize(vmin=cnt.levels[0]*scale_corr.value, vmax=cnt.levels[-1]*scale_corr.value)
        ax_cbar = fig.add_subplot(gs[0, 1])
        cb = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap="plasma"),  # aspect=1/30,
            cax=ax_cbar,
            label=r"N/Mpc$^2$",
            orientation="horizontal",
        )
        cb.ax.xaxis.set_ticks_position("top")
        cb.ax.xaxis.set_label_position("top")
        # fig.colorbar(cnt, ax=axs.ravel().tolist())

        # fig, axs = plt.subplots(
        #     2,
        #     2,
        #     figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth),
        #     constrained_layout=True,
        #     subplot_kw={"projection": img_wcs},
        #     sharex=True,
        #     sharey=True,
        # )
        for a_i, (ax, (z_low, z_high), icon, size, colour) in enumerate(
            zip(
                axs.ravel(),
                [[1.09, 1.11], [1.32, 1.37], [1.85, 1.9], [2.63, 2.67]],
                ["x", "*", "o", "s"],
                [15, 30, 15, 15],
                ["C0", "C1", "C2", "C3"],
            )
        ):
            # print (z_low)
            # continue
            # exit()
            # for a in axs:
            ax.imshow(
                img_hdul[0].data,
                norm=astrovis.ImageNormalize(
                    img_hdul[0].data,
                    # interval=astrovis.PercentileInterval(99.9),
                    interval=astrovis.ManualInterval(-0.001, 10),
                    stretch=astrovis.SqrtStretch(),
                ),
                cmap="binary",
                origin="lower",
                zorder=-1,
            )

            X, Y = np.meshgrid(
                np.arange(img_extent[0], img_extent[1], 1 / 3600),
                np.arange(img_extent[2], img_extent[3], 1 / 3600),
            )

            lens_tab = full_lens_tab[
                (full_lens_tab["z"] > z_low) & (full_lens_tab["z"] < z_high)
            ]
            lens_tab = lens_tab[np.unique(lens_tab["ID"], return_index=True)[1]]

            z_med = np.median(lens_tab["z"].data)
            ax.annotate(
                rf"$\overline{{z}}={z_med:.2f}$",
                (0.1, 0.85),
                xycoords="axes fraction",
            )
            d_A = cosmo.angular_diameter_distance(z_med)
            scale_radian = 1 * np.pi / 180 / 3600
            kpc_scale = d_A.to(u.kpc) * scale_radian / u.arcsec

            plot_utils.sky_plot(
                lens_tab,
                ax=ax,
                ra_col="RA_source",
                dec_col="Dec_source",
                color=colour,
                label="Source Plane",
                zorder=3,
                marker=icon,
                s=size,
                edgecolor="none",
            )
            # for a in axs:
            #     a.legend()

            ra_s = lens_tab["RA_source"].data
            dec_s = lens_tab["Dec_source"].data

            # xlims = axs[1].get_xlim()
            # ylims = axs[1].get_ylim()

            # print (xlims)
            # X, Y = np.mgrid[ra_s.min() : ra_s.max() : 100j, dec_s.min() : dec_s.max() : 100j]
            # X, Y = np.mgrid[xlims[0] : xlims[1] : 6000j, ylims[0] : ylims[1] : 6000j]
            positions = np.vstack([X.ravel(), Y.ravel()])
            values = np.vstack([ra_s, dec_s])
            kernel = gaussian_kde(values)
            Z = np.reshape(kernel(positions).T, X.shape)
            # axs[1].imshow(
            #     # np.rot90(Z),
            #     Z,
            #     cmap=plt.cm.gist_earth_r,
            #     # extent=[ra_s.min(), ra_s.max(), dec_s.min(), dec_s.max()],
            #     extent = [X.min(),X.max(),Y.min(),Y.max()],
            #     # transform=axs[1].get_transform("world"),
            #     # extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
            #     # zorder=-1
            # )
            # print (np.unique(Z))
            # print (np.nansum(Z), np.prod(np.diff(img_extent.reshape(-1, 2), axis=1).ravel()),np.nansum(Z/3600**2))
            # exit()
            Z *= (
                len(lens_tab)
                * (1 / ((1 * u.deg).to(u.arcsec) * kpc_scale) ** 2)
                .to(1 / u.Mpc**2)
                .value
            )
            # scale_corr = (1/((1*u.deg).to(u.arcsec)*kpc_scale)**2).to(1/u.Mpc**2)
            # print (np.geomspace(0.5/scale_corr.value,10/scale_corr.value,10),)
            vmax = np.nanmax(Z)
            cnt = ax.contour(
                X,
                Y,
                # np.rot90(Z),
                Z,
                # colors=["red"],
                # levels=np.linspace(0.1/scale_corr.value,3/scale_corr.value,10),
                # levels=np.linspace(0.1/scale_corr.value,vmax,10),
                # extent=[ra_s.min(), ra_s.max(), dec_s.min(), dec_s.max()],
                extent=[X.min(), X.max(), Y.min(), Y.max()],
                # extent=[xlims[0],xlims[1],ylims[0],ylims[1]],
                # zorder=10,
                transform=ax.get_transform("world"),
                # vmin = 0.1/scale_corr.value,
                # # vmax= 5/scale_corr.value
                vmin=10,
                vmax=100,
                # vmax=vmax
                cmap="plasma",
            )
            # print (scale_corr)
            print(cnt.levels)
            # import matplotlib as mpl
            # # cmap = mpl.cm.plasma
            # norm = mpl.colors.Normalize(vmin=cnt.levels[0], vmax=cnt.levels[-1])
            # # norm = mpl.colors.Normalize(vmin=cnt.levels[0]*scale_corr.value, vmax=cnt.levels[-1]*scale_corr.value)

            # fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="plasma"),aspect=30,
            #             ax=axs.ravel().tolist(), label=r'N/Mpc$^2$')
            # fig.colorbar(cnt, ax=axs.ravel().tolist())
            if a_i < 2:
                # ax.coords[0].set_ticks_visible(False)
                ax.coords[0].set_ticklabel_visible(False)
            else:
                ax.set_xlabel("R.A.")
            if a_i % 2 == 0:
                ax.set_ylabel("Dec.")
            else:
                # ax.coords[1].set_ticks_visible(False)
                ax.coords[1].set_ticklabel_visible(False)

            ax.coords[0].set_major_formatter("d.dd")
            ax.coords[1].set_major_formatter("d.dd")
            # # lat.set_ticks_visible(False)
            # # lat.set_ticklabel_visible(False)
            #     # axs[1].set_xlabel("")
            #     axs[1].set_xlabel("R.A.")
            #     axs[1].set_ylabel("Dec.")
            #     axs[0].set_ylabel("Dec.")
            # plt.colorbar(cnt)
            # print (cnt.levels/((1*u.deg).to(u.arcsec)*kpc_scale.to(u.Mpc/u.arcsec))**2)
            # print ((1*u.deg).to(u.arcsec)*kpc_scale)
            # exit()
            # ax.scatter(lens_tab["RA"], lens_tab["Dec"])
            # ax.scatter(lens_tab["RA_source"], lens_tab["Dec_source"])
            # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

            fig.patch.set_alpha(0.0)

            plt.savefig(save_dir / f"od_source_plane_subplots.pdf")

    # plt.show()
