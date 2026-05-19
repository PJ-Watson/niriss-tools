"""Pipeline utility functions."""

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from grizli import utils as grizli_utils
from numpy.typing import ArrayLike

__all__ = [
    "parse_images_from_pattern",
    "find_matches",
    "getObsIdFromQuery",
    "getExpIdFromQuery",
    "queryMAST",
]


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


@np.vectorize
def getObsIdFromQuery(obsName: str) -> int:
    """
    Cutout the obs ID from the long, jumbled MAST obs ID.

    The original version of this function was written by VM and ZS for
    `passagepipe.utils`.

    Parameters
    ----------
    obsName : str
        The MAST observation name.

    Returns
    -------
    int
        The observation ID.
    """

    return int(obsName.split("_")[0][7:-3])


@np.vectorize
def getExpIdFromQuery(obsName: str) -> int:
    """
    Cutout the exp ID from the long, jumbled MAST obs ID.

    The original version of this function was written by VM and ZS for
    `passagepipe.utils`.

    Parameters
    ----------
    obsName : str
        The MAST observation name.

    Returns
    -------
    int
        The exposure ID.
    """

    return int(obsName.split("_")[1])


def queryMAST(
    pid: int, instrument: str = "NIRISS", use_filter: ArrayLike | None = None
) -> Table:
    """
    Query MAST for the full list of observations for specific PID.

    The original version of this function was written by VM and ZS for
    `passagepipe.utils`.

    Parameters
    ----------
    pid : int
        The JWST Proposal ID.
    instrument : str, optional
        Select the instrument to query observations. By default only NIRISS
        observations will be returned.
    use_filter : ArrayLike | None, optional
        Return observations only in a specific set of filters, by default
        `None`.

    Returns
    -------
    Table
        The set of observations requested.
    """

    from astropy.table import vstack
    from astroquery.mast import Observations
    from mastquery import query

    query.DEFAULT_QUERY["project"] = ["JWST"]
    query.DEFAULT_QUERY["obs_collection"] = ["JWST"]
    query.DEFAULT_QUERY["instrument_name"] = [f"{instrument.upper()}*"]

    queryList = query.run_query(
        box=None,
        proposal_id=[pid],
        base_query=query.DEFAULT_QUERY,
    )
    if use_filter is not None:
        queryList = queryList[np.isin(queryList["filter"], use_filter)]

    if "target_name" not in queryList.columns:
        queryList["target_name"] = queryList["target"]
    subqueryList = Observations.get_product_list(queryList)

    cond = (
        (subqueryList["calib_level"] == 1)
        # & (subqueryList["productType"] == "SCIENCE")
        & (subqueryList["productSubGroupDescription"] == "UNCAL")
    )

    uncalList = subqueryList[cond]
    _, idx = np.unique(uncalList["obs_id"], return_index=True)
    uncalList = uncalList[idx]

    uncalList["obs_id_num"] = getObsIdFromQuery(obsName=np.asarray(uncalList["obs_id"]))
    uncalList["exp_id_num"] = getExpIdFromQuery(obsName=np.asarray(uncalList["obs_id"]))
    return uncalList
