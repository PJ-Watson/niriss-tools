"""
Setup default imports and directories.
"""

import os
from functools import partial
from pathlib import Path

import astropy.visualization as astrovis
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs import utils as wcs_utils
from numpy.typing import ArrayLike

root_dir = Path(os.getenv("ROOT_DIR"))
save_dir = root_dir / "2024_08_16_A2744_v4" / "catalogue_paper" / "plots"
