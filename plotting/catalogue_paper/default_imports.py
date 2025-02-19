"""
Setup default imports and directories.
"""

import os
from functools import partial
from pathlib import Path

import astropy.units as u
import astropy.visualization as astrovis
import matplotlib.pyplot as plt
import numpy as np
import plot_utils
from astropy.coordinates import SkyCoord, match_coordinates_sky
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from cmcrameri import cm
from numpy.typing import ArrayLike

root_dir = Path(os.getenv("ROOT_DIR"))
root_name = "glass-a2744"
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

full_cat = Table.read(
    root_dir
    / "2024_08_16_A2744_v4"
    / "grizli_home"
    / "Extractions_v4"
    / "catalogues"
    / "stage_5_output_internal_phot_zprev.fits"
)

niriss_filter_sens = {
    "F115W": [1.0130, 1.2830],
    "F150W": [1.3300, 1.6710],
    "F200W": [1.7510, 2.2260],
}
