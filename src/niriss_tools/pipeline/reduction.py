"""
Functions used to reduce the raw data.
"""

import os
from os import PathLike
from pathlib import Path

import astropy
import numpy as np

__all__ = [
    "stsci_det1",
    "run_det1",
    "gen_associations",
    "process_using_aws",
    "recursive_merge",
    "load_assoc"
]


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
            # print(f"{output_filename} exists.")
            continue
        else:
            files_to_process.append(file)

    if len(files_to_process) > 0:

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


def gen_associations(raw_output_dir: PathLike, field_name: str = "glass-a2744") -> dict:
    """
    Generate exposure tables for each group of filters and grisms.

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
        The keys of the dict are the name of each group (equivalent to
        ``"assoc"`` used by `grizli.aws`), and the values are the
        exposure tables.
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


def load_assoc(
    ra=3.58641,
    dec=-30.39997,
    radius=1,
    proposal_id=1324,
    instrument_name = "NIRISS"
):
    """
    Load exposure tables and association names from the DJA.

    Default parameters are those used for the GLASS-JWST ERS analysis.

    Parameters
    ----------
    ra : float, optional
        The right ascension of the centre of the field of interest, by
        default ``3.58641``.
    dec : float, optional
        The declination of the centre of the field of interest, by default
        ``-30.39997``.
    radius : int, optional
        The radius in arcminutes to query, by default ``1``.
    proposal_id : int, optional
        The JWST proposal ID for the observations of interest, by default
        ``1324``.

    Returns
    -------
    dict
        The keys of the dict are the name of each association, and the
        values are the exposure tables.
    """

    import shutil

    from grizli import utils
    from grizli.aws import visit_processor

    QUERY_URL = (
        "https://grizli-cutout.herokuapp.com/assoc"
        "?coord={ra},{dec}&arcmin={radius}&output=csv"
    )

    assoc_query = utils.read_catalog(
        QUERY_URL.format(ra=ra, dec=dec, radius=radius), format="csv"
    )

    nis = (assoc_query["instrument_name"] == instrument_name) & (
        assoc_query["proposal_id"] == proposal_id
    )

    EXPOSURE_API = "https://grizli-cutout.herokuapp.com/exposures?associations={assoc}"

    assoc_dict = {}
    for assoc_name in assoc_query["assoc_name"][nis]:
        exp = utils.read_catalog(EXPOSURE_API.format(assoc=assoc_name), format="csv")
        assoc_dict[assoc_name] = exp
        # assoc_dict[assoc_name] = None

    return assoc_dict


def process_using_aws(
    grizli_home_dir: PathLike,
    raw_output_dir: PathLike,
    assoc_tab_dict: dict,
    field_name: str = "glass-a2744",
    process_visit_kwargs: dict = {},
    ref_wcs: astropy.wcs.WCS | None = None,
    mosaic_pixel_scale: float = 0.03,
    mosaic_pad: float = 6,
    drizzle_kernel: str = "square",
    drizzle_pixfrac: float = 0.8,
    cutout_mosaic_kwargs: dict = {},
    proposal_id = 1324,
):
    """
    Process WFSS data using the functions in `grizli.aws`.

    Parameters
    ----------
    grizli_home_dir : PathLike
        Directory containing the usual grizli folders, e.g. ``"Prep"``,
        ``"visits"``.
    raw_output_dir : PathLike
        Where the current ``_rate.fits`` files are located.
    assoc_tab_dict : dict
        The keys of the dict are the name of each group of exposures,
        and the values should be an exposure table.
    field_name : str, optional
        The name of the field, by default ``"glass-a2744"``.
    process_visit_kwargs : dict, optional
        Any additional arguments to pass to
        `grizli.aws.visit_processor.process_visit`, by default ``{}``.
    ref_wcs : astropy.wcs.WCS | None, optional
        The reference WCS to be used when generating the drizzled mosaics.
        By default None, in which case it will be calculated automatically
        from the overlap of the individual filter images.
    mosaic_pixel_scale : float, optional
        The pixel scale (in arcseconds) to be used for the drizzled
        mosaics, by default 0.03.
    mosaic_pad : float, optional
        The padding for the drizzled mosaics in arcseconds, by default 6.
    drizzle_kernel : str, optional
        The kernel to use for drizzling the mosaics, by default
        ``"square"``.
    drizzle_pixfrac : float, optional
        The ``pixfrac`` used for the drizzled mosaics, by default 0.8.
    cutout_mosaic_kwargs : dict, optional
        Any additional arguments to pass to
        `grizli.aws.visit_processor.cutout_mosaic`, by default ``{}``.
    """

    visit_dir = grizli_home_dir / "visits"
    os.chdir(visit_dir)

    import shutil

    from grizli import utils as grizli_utils
    from grizli.aws import visit_processor

    visit_processor.ROOT_PATH = str(visit_dir)

    for assoc_name, exp in assoc_tab_dict.items():
        if not (visit_dir / assoc_name / "Prep").is_dir():

            # Make all the directories
            assoc_dir = visit_dir / assoc_name
            (assoc_dir / "RAW").mkdir(exist_ok=True, parents=True)
            (assoc_dir / "Persistence").mkdir(exist_ok=True, parents=True)
            (assoc_dir / "Extractions").mkdir(exist_ok=True, parents=True)
            (assoc_dir / "Prep").mkdir(exist_ok=True, parents=True)

            # Only copy files if this visit hasn't been processed yet
            if len([*(assoc_dir / "Prep").glob("*drz_sci.fits")]) == 0:
                for filename in exp["dataset"]:
                    try:
                        shutil.copy(
                            raw_output_dir / f"{filename}_rate.fits", assoc_dir / "RAW"
                        )
                    except Exception as e:
                        print(e)
                        print(f"{filename} not found?")

    for assoc_name, tab in assoc_tab_dict.items():
        if len([*(visit_dir / assoc_name / "Prep").glob("*drz_sci.fits")]) == 0:
            # By default, do not clean all files afterwards
            if not process_visit_kwargs.get("clean"):
                process_visit_kwargs["clean"] = False

            # Ensure the correct CRDS context is used, unless otherwise specified
            if not process_visit_kwargs.get("other_args"):
                process_visit_kwargs["other_args"] = {}
            if not process_visit_kwargs["other_args"].get("CRDS_CONTEXT"):
                process_visit_kwargs["other_args"]["CRDS_CONTEXT"] = os.environ[
                    "CRDS_CONTEXT"
                ]

            if not process_visit_kwargs["other_args"].get("mosaic_drizzle_args"):
                process_visit_kwargs["other_args"]["mosaic_drizzle_args"] = {}
            if not process_visit_kwargs["other_args"]["mosaic_drizzle_args"].get(
                "context"
            ):
                process_visit_kwargs["other_args"]["mosaic_drizzle_args"]["context"] = (
                    os.environ["CRDS_CONTEXT"]
                )

            if "instrume" in tab.colnames:
                tab["instrument_name"] = tab["instrume"]
                tab["proposal_id"] = proposal_id
                tab["dataURL"] = [f"dummy/{file}_rate.fits" for file in tab["file"]]

            _ = visit_processor.process_visit(
                assoc_name,
                sync=False,
                with_db=False,
                tab=tab,
                **process_visit_kwargs,
            )
        else:
            print(f"Directory {assoc_name} found, local preprocesing complete!")

    os.chdir(grizli_home_dir / "Prep")

    # Symlink preprocessed exposure files here
    import subprocess

    for assoc_name in assoc_tab_dict.keys():
        subprocess.run(f"ln -sf ../visits/{assoc_name}/Prep/*rate.fits .", shell=True)

    import numpy as np
    from astropy.wcs import WCS

    files = [str(s) for s in (grizli_home_dir / "Prep").glob("*rate.fits")]
    files.sort()
    res = visit_processor.res_query_from_local(files=files)
    is_grism = np.array(["GR" in filt for filt in res["filter"]])

    if not ref_wcs:
        # Mosaic WCS that contains all exposures
        hdu = grizli_utils.make_maximal_wcs(
            files=files,
            pixel_scale=mosaic_pixel_scale,
            pad=mosaic_pad,
            get_hdu=True,
            verbose=False,
        )

        ref_wcs = WCS(hdu.header)

    # Default set of parameters for drizzled mosaics
    _mosaic_kwargs = {
        "rootname": field_name,
        "res": res[
            ~is_grism
        ],  # Pass the exposure information table for the direct images
        "ir_wcs": ref_wcs,
        "half_optical": False,  # Otherwise will make JWST exposures at half pixel scale of ref_wcs
        "kernel": drizzle_kernel,
        "pixfrac": drizzle_pixfrac,
        "clean_flt": False,  # Otherwise removes "rate.fits" files from the working directory!
        "s3output": None,
        "make_exptime_map": False,
        "weight_type": "jwst",
        "skip_existing": False,
        "context": os.environ["CRDS_CONTEXT"],
    }
    # Make individual drizzled images for each of the filters
    _ = visit_processor.cutout_mosaic(
        **recursive_merge(_mosaic_kwargs, cutout_mosaic_kwargs)
    )

    from astropy.table import vstack
    from grizli.pipeline import auto_script

    # Create a combined visits.yaml
    visits, groups, info = [], [], None
    for assoc_name in assoc_tab_dict.keys():
        v, g, i = auto_script.load_visits_yaml(
            grizli_home_dir
            / "visits"
            / assoc_name
            / "Prep"
            / f"{assoc_name}_visits.yaml"
        )
        for j, v_j in enumerate(v):
            v[j]["footprints"] = [fp for fps in v_j["footprints"] for fp in fps]
        for j, g_j in enumerate(g):
            for img_type in g_j.keys():
                try:
                    g[j][img_type]["footprints"] = [
                        fp for fps in g_j[img_type]["footprints"] for fp in fps
                    ]
                except:
                    print(f"Problem with {g[j]}")

        visits.extend(v)
        groups.extend(g)
        if info is None:
            info = i
        else:
            info = vstack([info, i])

    auto_script.write_visit_info(
        visits, groups, info, field_name, path=str(grizli_home_dir / "Prep")
    )

    # Make a stacked mosaic using all three filters
    auto_script.make_filter_combinations(
        field_name,
        filter_combinations={"ir": ["F115WN-CLEAR", "F150WN-CLEAR", "F200WN-CLEAR"]},
    )


def recursive_merge(d1: dict, d2: dict) -> dict:
    """
    Recursively merge two dictionaries.

    Values from the second are prioritised in case of conflicts. This code
    was originally posted on stackoverflow, by Aaron Hall and Bobik.

    Parameters
    ----------
    d1 : dict
        The original dictionary.
    d2 : dict
        The new dictionary, which can overwrite values in `d1`.

    Returns
    -------
    dict
        The merged dictionary.
    """

    from collections.abc import MutableMapping

    for k, v in d1.items():
        if k in d2:
            # this next check is the only difference!
            if all(isinstance(e, MutableMapping) for e in (v, d2[k])):
                d2[k] = recursive_merge(v, d2[k])
            # we could further check types and merge as appropriate here.
    d3 = d1.copy()
    d3.update(d2)
    return d3
