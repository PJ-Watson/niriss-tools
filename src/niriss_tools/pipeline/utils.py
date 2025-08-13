"""Pipeline utility functions."""

import os

from astropy.io import fits
from grizli import utils as grizli_utils

__all__ = ["parse_images_from_pattern", "find_matches"]


def parse_images_from_pattern(img_dir: os.PathLike, pattern: str = "*.fits") -> dict:
    """
    Populate a dict with info and locations of images matching a pattern.

    Parameters
    ----------
    img_dir : os.PathLike
        The directory to search.
    pattern : str, optional
        The pattern to search with, by default ``"*.fits"``.

    Returns
    -------
    dict
        The keys of each entry are lower case, and in the format
        ``"{telescope}-{instrument}-{filter}"``. Each entry is itself a
        dictionary, containg basic information on the image and its
        location.
    """

    out_dict = {}
    for i, filepath in enumerate(img_dir.glob(f"{pattern}")):
        print(filepath.name)
        hdr = fits.getheader(filepath)

        key = f"{hdr["TELESCOP"]}-{hdr["INSTRUME"]}-"
        filt = grizli_utils.parse_filter_from_header(hdr)
        key += f"{filt.removeprefix("F150W2-").removesuffix("-CLEAR")}"
        key = key.lower()

        out_dict[key] = {
            "filt": grizli_utils.parse_filter_from_header(hdr, filter_only=True),
            "pupil": hdr.get("PUPIL", "UNKNOWN"),
            "detector": hdr["DETECTOR"],
            "instrument": hdr["INSTRUME"],
            "telescope": hdr["TELESCOP"],
            "sci": str(filepath),
        }

    return out_dict


def find_matches(
    img_dir: os.PathLike,
    info_dict: dict,
    pattern: str = "*{filt}-*_drc_var.fits*",
    key_name: str = "var",
    case_sensitive: bool = False,
) -> dict:
    """
    Update an info_dict with matches from a specified directory.

    Parameters
    ----------
    img_dir : os.PathLike
        The directory to search.
    info_dict : dict
        A nested dictionary, where each entry contains information about
        observed images.
    pattern : str, optional
        The filename pattern to match, by default
        ``"*{filt}-*_drc_var.fits*"``.
    key_name : str, optional
        The name to add to each dictionary item, by default ``"var"``.
    case_sensitive : bool, optional
        Whether to match on case, by default ``False``.

    Returns
    -------
    dict
        The modified input ``info_dict``.
    """

    # I can't wait for Python 3.14 to introduce template strings and clear this mess up
    import re

    kw_names = re.findall(r"\{(.*?)\}", pattern)
    print(kw_names)

    for key, info in info_dict.items():
        eval_kws = {}
        for kw in kw_names:
            eval_kws[kw] = info[kw]
        print(pattern.format(**eval_kws))
        for i, filepath in enumerate(
            img_dir.glob(
                pattern.format(**eval_kws),
                case_sensitive=case_sensitive,
            )
        ):
            info_dict[key][key_name] = str(filepath)

    return info_dict
