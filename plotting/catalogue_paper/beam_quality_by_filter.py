"""
Show the beam quality as a function of orientation and grism filter.
"""

import plot_utils
from default_imports import *

plot_utils.setup_aanda_style()

if __name__ == "__main__":

    catalogue_dir = (
        root_dir
        / "2024_08_16_A2744_v4"
        / "glass_niriss"
        / "match_catalogues"
        / "classification_v1"
    )

    # cat_names = ["grizli_photz_matched.fits"]

    # catalogue_path = catalogue_dir / "grizli_photz_matched.fits"

    v1_cat = Table.read(catalogue_dir / "compiled_catalogue_v1.fits")

    fig, ax = plt.subplots(
        figsize=(plot_utils.aanda_columnwidth, plot_utils.aanda_columnwidth / 1.62),
        constrained_layout=True,
    )
    hist = partial(plot_utils.plot_hist, bins=np.arange(16.5, 33.5, 0.5), ax=ax)
    # print (np.nanmin(v1_cat["MAG_AUTO"]), np.nanmax(v1_cat["MAG_AUTO"])

    hist(v1_cat["MAG_AUTO"], label="Full Sample")
    hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] > 0], ax=ax, label="Extracted")
    hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 4], ax=ax, label="First Pass")
    hist(v1_cat["MAG_AUTO"][v1_cat["V1_CLASS"] >= 5], ax=ax, label="Placeholder")

    ax.set_xlabel(r"MAG_AUTO")
    ax.set_ylabel(r"Number of Objects")
    ax.legend()

    # plt.savefig(save_dir / "magnitude_hist.pdf")

    plt.show()
