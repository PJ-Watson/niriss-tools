"""
Functions used to reduce the raw data.
"""

import os
from os import PathLike
from pathlib import Path

import numpy as np


def stsci_det1(
    uncal_dir: PathLike,
    raw_output_dir: PathLike,
    cpu_count: int = 1,
    joblib_backend: str = "loky",
    **kwargs,
):
    """
    Run the JWST level 1 detector pipeline.

    Parameters
    ----------
    uncal_dir : PathLike
        The location of the ``_uncal.fits`` files.
    raw_output_dir : PathLike
        Where the output files will be saved.
    cpu_count : int, optional
        The number of CPU cores to use for running the pipeline. Each core
        will process a single file, so note that the memory usage will
        increase substantially when running in parallel. By default only
        one core will be used.
    joblib_backend : str, optional
        The backend to use when running the pipeline in parallel, by
        default ``"loky"``.
    **kwargs : dict, optional
        Any additional keyword arguments, to be passed to
        `jwst.pipeline.Detector1Pipeline`.
    """

    if cpu_count <= 0:
        cpu_count = os.cpu_count()

    files_to_process = []

    for i, file in enumerate(uncal_dir.glob("*uncal.fits")):
        output_filename = (raw_output_dir / file.name).with_stem(
            file.stem.replace("_uncal", "_rate")
        )
        if output_filename.is_file():
            print(f"{output_filename} exists.")
            continue
        else:
            files_to_process.append(file)

    from joblib import Parallel, delayed

    Parallel(
        n_jobs=cpu_count,
        backend=joblib_backend,
        verbose=50,
    )(
        delayed(run_det1)(
            uncal_path=file,
            output_dir=raw_output_dir,
            **kwargs,
        )
        for file in files_to_process
    )


def run_det1(uncal_path: PathLike, output_dir: PathLike, **kwargs):
    """
    Run the level 1 detector pipeline on an uncalibrated image.

    Applies detector-level corrections.

    Parameters
    ----------
    uncal_path : PathLike
        The location of the file to be processed, with the extension
        ``_uncal.fits``.
    output_dir : PathLike
        The directory into which the output files will be saved.
    **kwargs : dict, optional
        Any additional keyword arguments, to be passed to
        `jwst.pipeline.Detector1Pipeline`.
    """

    from jwst.pipeline import Detector1Pipeline

    uncal_path = Path(uncal_path)

    Detector1Pipeline.call(
        str(uncal_path),
        output_dir=str(output_dir),
        save_results=True,
        output_file=uncal_path.stem.removesuffix("_uncal"),
        **kwargs,
    )


def gen_assocs(raw_output_dir: PathLike, field_name: str = "glass-a2744") -> dict:
    """
    Generate association tables for each group of filters and grisms.

    This is particularly useful for programmes such as PASSAGE, for which
    no existing associations are available in the Dawn JWST Archive.

    Parameters
    ----------
    raw_output_dir : PathLike
        Where the current ``_rate.fits`` files are located.
    field_name : str, optional
        The name of the field, by default "glass-a2744".

    Returns
    -------
    dict
        The keys of the dict are the name of each group, and the values
        are the association tables.
    """

    from astropy.io import fits
    from astropy.table import Table
    from astropy.wcs import WCS
    from grizli import utils as grizli_utils
    from shapely.geometry import Polygon

    assoc_dict = {}

    for filepath in raw_output_dir.glob("*_rate.fits"):
        main_hdr = fits.getheader(filepath)
        sci_hdr = fits.getheader(filepath, "SCI")
        filt_only = grizli_utils.parse_filter_from_header(main_hdr).split("-")[0]
        filt_all = grizli_utils.parse_filter_from_header(main_hdr)

        # There's surely a more intelligent way to do this, but grizli is happy for now
        s = np.asarray(
            sci_hdr["S_REGION"].removeprefix("POLYGON ICRS  ").split(" "), dtype=float
        )
        footprint = (
            f"(({s[0]}, {s[1]}), ({s[2]}, {s[3]}), ({s[4]}, {s[5]}), ({s[6]}, {s[7]}))"
        )
        name = filepath.stem.removesuffix("_rate")
        try:
            assoc_dict[f"{field_name}_assoc_{filt_only.lower()}"].add_row(
                [
                    name,
                    "rate",
                    name,
                    f"{field_name}_assoc_{filt_only.lower()}",
                    filt_all,
                    main_hdr.get("INSTRUME"),
                    main_hdr.get("PROGRAM"),
                    f"dummy/{filepath.name}",
                    footprint,
                ],
            )
        except:
            assoc_dict[f"{field_name}_assoc_{filt_only.lower()}"] = Table(
                data=[
                    [name],
                    ["rate"],
                    [name],
                    [f"{field_name}_assoc_{filt_only.lower()}"],
                    [filt_all],
                    [main_hdr.get("INSTRUME")],
                    [main_hdr.get("PROGRAM")],
                    [f"dummy/{filepath.name}"],
                    [footprint],
                ],
                names=[
                    "file",
                    "extension",
                    "dataset",
                    "assoc",
                    "filter",
                    "instrument_name",
                    "proposal_id",
                    "dataURL",
                    "footprint",
                ],
            )
    return assoc_dict
