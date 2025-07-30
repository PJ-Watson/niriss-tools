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
