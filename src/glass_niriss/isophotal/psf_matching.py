import os
from os import PathLike
from pathlib import Path
from typing import Callable

import numpy as np
from astropy.io import fits
from astropy.nddata import block_reduce
from numpy.typing import ArrayLike
from photutils.psf import (
    CosineBellWindow,
    TopHatWindow,
    TukeyWindow,
    create_matching_kernel,
)
from scipy.ndimage import zoom

from glass_niriss.isophotal import align

__all__ = ["match_photutils", "match_pypher", "reproject_and_convolve"]


def match_photutils(
    source_psf: ArrayLike | PathLike,
    target_psf: ArrayLike | PathLike,
    out_path: PathLike | None = None,
    window: Callable | None = CosineBellWindow(0.5),
    oversample: int = 3,
):
    if isinstance(source_psf, PathLike):
        source_psf = fits.getdata(source_psf)
    if oversample != 1:
        source_psf = zoom(source_psf, oversample)
    source_psf /= source_psf.sum()

    if isinstance(target_psf, PathLike):
        target_psf = fits.getdata(target_psf)
    if oversample != 1:
        target_psf = zoom(target_psf, oversample)
    target_psf /= target_psf.sum()

    phot_kernel = create_matching_kernel(source_psf, target_psf, window=window)

    if oversample != 1:
        phot_kernel = block_reduce(phot_kernel, oversample)

    phot_kernel /= phot_kernel.sum()
    if out_path is not None:
        fits.writeto(out_path, data=phot_kernel)
    return phot_kernel


def match_pypher(
    source_psf: ArrayLike | PathLike,
    target_psf: ArrayLike | PathLike,
    out_path: PathLike | None = None,
    pypher_r: float = 3e-3,
    oversample: int = 3,
    pixscale: float = 0.04,
    tmp_dir: PathLike | None = None,
):
    if tmp_dir is None:
        tmp_dir = Path.cwd()
    if isinstance(source_psf, PathLike) and oversample == 1:
        source_path = source_psf
    else:
        if isinstance(source_psf, PathLike):
            source_psf = fits.getdata(source_psf)

        source_path = tmp_dir / "psf_a.fits"
        header = fits.Header()
        header["PIXSCALE"] = pixscale / oversample
        if oversample != 1:
            source_psf = zoom(source_psf, oversample)
        source_psf /= source_psf.sum()
        fits.writeto(source_path, data=source_psf, header=header, overwrite=True)

    if isinstance(target_psf, PathLike) and oversample == 1:
        target_path = target_psf
    else:
        if isinstance(target_psf, PathLike):
            target_psf = fits.getdata(target_psf)
        target_path = tmp_dir / "psf_b.fits"
        header = fits.Header()
        header["PIXSCALE"] = pixscale / oversample
        if oversample != 1:
            target_psf = zoom(target_psf, oversample)
        target_psf /= target_psf.sum()
        fits.writeto(target_path, data=target_psf, header=header, overwrite=True)

        # fits.writeto(DIR_KERNELS+'psf_b.fits',target_psf,header=hdr,overwrite=True)
    if out_path is None:
        out_path = tmp_dir / "kernel_a_to_b.fits"

    out_path.unlink(missing_ok=True)

    os.system(f"pypher {source_path} {target_path} {out_path} -r {pypher_r:.3g}")
    with fits.open(out_path, mode="update") as kern_hdul:
        # kernel = kern_hdul[0].data
        if oversample != 1:
            kern_hdul[0].data = block_reduce(kern_hdul[0].data, oversample)

        kern_hdul[0].data /= kern_hdul[0].data.sum()
        kern_hdul.flush()

        kernel = kern_hdul[0].data.copy()

    for p in ["psf_a.fits", "psf_b.fits", "kernel_a_to_b.fits"]:
        (tmp_dir / p).unlink(missing_ok=True)

    return kernel


def reproject_and_convolve(
    ref_path: PathLike,
    orig_images: list[PathLike],
    psfs: list[ArrayLike | PathLike | None],
    out_dir: PathLike,
    psf_target: ArrayLike | PathLike,
    oversample: int = 3,
    save_unconvolved: bool = True,
    new_wcs_kw: dict | None = None,
    reproject_image_kw: dict | None = None,
    psf_method: str = "pypher",
    psf_match_kw: dict | None = None,
    convolve_method: str = "fft",
    new_names: list[str] | None = None,
):

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    new_wcs_kwargs = dict(new_wcs_kw or {})
    new_wcs, new_shape = align.gen_new_wcs(ref_path=ref_path, **new_wcs_kwargs)
    print(new_shape)
    reproject_image_kwargs = dict(reproject_image_kw or {})

    if "method" not in reproject_image_kwargs:
        reproject_image_kwargs["method"] = "adaptive"

    print(reproject_image_kwargs)
    if convolve_method == "fft":
        from astropy.convolution import convolve_fft as conv_fn
    elif convolve_method == "direct":
        from astropy.convolution import convolve as conv_fn

    orig_images = np.atleast_1d(orig_images).ravel()
    psfs = np.atleast_1d(psfs).ravel()
    if new_names is None:
        new_names = [None] * len(orig_images)
    else:
        new_names = np.atleast_1d(new_names).ravel()

    return_paths = []

    for i, (o, p, n) in enumerate(zip(orig_images, psfs, new_names)):
        out_path = out_dir
        if n is not None:
            out_path = out_path / n
        print(out_path)
        # if out_path.is_file()
        # exit()
        repr_path = align.reproject_image(
            o, out_path, new_wcs, new_shape, **reproject_image_kwargs
        )
        # exit()

        if p is None:
            return_paths.append(None)
            continue

        p = Path(p)
        psf_target = Path(psf_target)

        if p.stem == psf_target.stem:
            conv_path = repr_path.with_name(repr_path.name.replace("repr_", "conv_"))
            if conv_path.is_file():
                print(f"'{conv_path.name}' already exists. Skipping.")
            else:
                print(
                    "Target PSF is the same as source PSF. Copying file without convolving."
                )
                import shutil

                shutil.copy(repr_path, conv_path)

            return_paths.append(conv_path)
            continue

        if (out_path.parent / f"{p.stem}_matched_{psf_target.stem}.fits").is_file():
            kernel = fits.getdata(
                out_path.parent / f"{p.stem}_matched_{psf_target.stem}.fits"
            )
        else:
            psf_match_kwargs = dict(psf_match_kw or {})
            if psf_method == "pypher":
                kernel = match_pypher(
                    p,
                    psf_target,
                    pixscale=0.04,
                    out_path=out_path.parent
                    / f"{p.stem}_matched_{psf_target.stem}.fits",
                    **psf_match_kwargs,
                )
            # import pypher
            else:
                kernel = match_photutils(p, psf_target, **psf_match_kwargs)

        kernel /= kernel.sum()

        conv_path = repr_path.with_name(repr_path.name.replace("repr_", "conv_"))
        if conv_path.is_file():
            print(f"'{conv_path.name}' already exists. Skipping.")
        else:
            with fits.open(repr_path) as repr_hdul:

                # import time
                # t0 = time.time()
                # print (t0)
                # conv_data = convolve(repr_hdul[0].data, kernel)
                # t1 = time.time()
                # print (t1-t0)
                conv_data = conv_fn(repr_hdul[0].data, kernel)
                # print (time.time()-t1)
                fits.writeto(
                    conv_path,
                    data=conv_data,
                    header=repr_hdul[0].header,
                )
        return_paths.append(conv_path)

    return return_paths
