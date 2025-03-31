"""
Extract catalogues for lensing analysis.
"""

import plot_utils
from default_imports import *

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
    # # fig.get_layout_engine().set(w_pad=0 / 72, h_pad=0 / 72, hspace=0, wspace=0)

    # fig.patch.set_alpha(0.0)

    # plt.savefig(save_dir / "test.pdf")

    # plt.show()

    ext_cat = Table.read(
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Extractions_v4"
        / "catalogues"
        / "full_external_em_lines.fits"
    )

    full_cat = full_cat[~full_cat["ID"].mask]

    _, idx_1, idx_2 = np.intersect1d(
        np.array(ext_cat["ID_NIRISS"]), np.array(full_cat["ID"]), return_indices=True
    )

    # ext_cat[""]
    # print (full_cat.colnames)

    print(np.nansum(idx_1 == idx_2))

    ext_cat["Z_SPEC_VALUE"] = full_cat["zmed_prev"].filled(np.nan)
    ext_cat["Z_NIRISS_NEW"] = ~np.isfinite(ext_cat["Z_SPEC_VALUE"])

    ext_cat.write(
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Extractions_v4"
        / "catalogues"
        / "CG_PB_full.fits",
        overwrite=True,
    )
    ext_cat = ext_cat[~ext_cat["Z_NIRISS"].mask]
    print(ext_cat)
    ext_cat.write(
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Extractions_v4"
        / "catalogues"
        / "CG_PB_z_niriss.fits",
        overwrite=True,
    )
    ext_cat = ext_cat[ext_cat["Z_NIRISS_NEW"]]
    ext_cat.write(
        root_dir
        / "2024_08_16_A2744_v4"
        / "grizli_home"
        / "Extractions_v4"
        / "catalogues"
        / "CG_PB_z_niriss_new.fits",
        overwrite=True,
    )

    # ext_cat["Z_SPEC_PREV"]

    print(ext_cat)
    # print (full_cat)
