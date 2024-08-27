"""
Create isophotal models of galaxies using an iterative process.
"""

import os
from pathlib import Path

import astropy.units as u
import numpy as np
import numpy.ma as ma
from astropy.convolution import convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from numpy.typing import ArrayLike
from photutils.background import Background2D
from photutils.isophote import Ellipse, EllipseGeometry, build_ellipse_model
from photutils.segmentation import SourceCatalog, SourceFinder, make_2dgaussian_kernel

_bcg_attr_warn = (
    "No bCGs have been identified yet. Please run "
    "`ClusterModels.load_bcg_catalogue()` to specify the galaxies "
    "to be modelled."
)
_img_attr_warn = (
    "No image has been loaded. Please run `ClusterModels.load_image()` "
    "to initialise the `{}` attribute."
)
_seg_attr_warn = (
    "No segmentation map has been loaded. Please run "
    "`ClusterModels.load_segmentation()` to load an existing segmentation map, or "
    "`ClusterModels.gen_segmentation() to create a new one."
)

# _rm_props = [
#     "background",
#     "background_ma",
#     "bbox",
#     "convdata",
#     "convdata_ma",
#     "data",
#     "data_ma",
#     "error",
#     "error_ma",
#     "kron_aperture",
#     "local_background_aperture",
#     "segment",
#     "segment_ma",
#     "sky_bbox_ll",
#     "sky_bbox_lr",
#     "sky_bbox_ul",
#     "sky_bbox_ur",
#     "sky_centroid",
#     "sky_centroid_icrs",
#     "sky_centroid_quad",
#     "sky_centroid_win",
# ]
req_seg_cat_props = [
    # "area",
    # "background_centroid",
    # "background_mean",
    # "background_sum",
    "label",
    "bbox_xmin",
    "bbox_xmax",
    "bbox_ymin",
    "bbox_ymax",
    "xcentroid",
    "ycentroid",
    "ellipticity",
    "orientation",
    "semimajor_sigma",
    "semiminor_sigma",
    # "centroid"
    # "centroid"
]


class ClusterModels:
    """
    A class to create isophotal models of cluster galaxies.

    An implementation of the method of Martis+24.

    Parameters
    ----------
    output_dir : `~os.PathLike`
        The directory in which the output will be saved.
    base_name : str
        The name prepended to all output files.
    """

    def __init__(
        self,
        output_dir: os.PathLike,
        base_name: str = "A2744_cluster_models",
    ):

        self.out_dir = Path(output_dir)
        self.base_name = base_name

        self._img_data = self._img_hdr = self._img_wcs = None
        self._seg_map = self._seg_cat = self._seg_wcs = None
        self._background = self._extra_mask = None

    @property
    def bcg_cat(self) -> Table:
        """
        A `~astropy.table.Table` of the galaxies to be modelled.
        """
        if self._bcg_cat is None:
            raise AttributeError(
                _bcg_attr_warn,
                name=None,
            )
        return self._bcg_cat

    @property
    def bcg_coords(self) -> SkyCoord:
        """
        A `~astropy.coordinates.SkyCoord` array of the bCG locations.
        """
        if self._bcg_coords is None:
            raise AttributeError(
                _bcg_attr_warn,
                name=None,
            )
        return self._bcg_coords

    @property
    def img_data(self) -> ArrayLike:
        """
        A 2D array of the image, containing the galaxies to be modelled.
        """
        if self._img_data is None:
            raise AttributeError(
                _img_attr_warn.format("img_data"),
                name=None,
            )
        return self._img_data

    @property
    def img_hdr(self) -> fits.Header:
        """
        The FITS header of the image, containing any ancillary information.
        """
        if self._img_hdr is None:
            raise AttributeError(
                _img_attr_warn.format("img_hdr"),
                name=None,
            )
        return self._img_hdr

    @property
    def img_wcs(self) -> WCS:
        """
        The WCS pertaining to the image.
        """
        if self._img_wcs is None:
            raise AttributeError(
                _img_attr_warn.format("img_wcs"),
                name=None,
            )
        return self._img_wcs

    @property
    def background(self) -> Background2D:
        """
        A 2D estimate of the background and associated RMS noise.
        """
        if self._background is None:
            raise AttributeError(
                _img_attr_warn.format("background"),
                name=None,
            )
        return self._background

    @property
    def seg_map(self) -> ArrayLike:
        """
        A segmentation map corresponding to all detected sources.

        A 2D array where the extent of each source is described by a
        unique positive integer. A value of 0 corresponds to the
        background.
        """
        if self._seg_map is None:
            raise AttributeError(
                _seg_attr_warn,
                name=None,
            )
        return self._seg_map

    @property
    def seg_cat(self) -> Table:
        """
        A catalogue detailing all sources in the segmentation map.

        At a minimum, for each object in the segmentation map, this
        catalogue contains:
        * Its integer value (`label` or `obj_id`) in the segmentation map.
        * The corners of its rectangular bounding box (`bbox_xmin`,
          `bbox_xmax`, `bbox_ymin`, and `bbox_ymax`).
        * The location of its centre in both pixel and world coordinates
          (`xcentroid`, `ycentroid`, `ra`, and `dec`).
        * Its `ellipticity` and `orientation`.
        * Its semimajor and seminor axis lengths (`orientation`,
        `semimajor_sigma`, and `semiminor_sigma`).
        """
        if self._seg_cat is None:
            raise AttributeError(
                "No segmentation map has been loaded. Please run "
                "`ClusterModels.load_segmentation()` to initialise the "
                "`seg_cat` attribute.",
                name=None,
            )
        return self._seg_cat

    @property
    def seg_wcs(self) -> WCS:
        """
        The WCS pertaining to the segmentation map.

        If the segmentation map was loaded, rather than generated using
        `ClusterModels.gen_segmentation()`, this is checked for
        equivalency with `img_wcs`.
        """
        if self._seg_wcs is None:
            raise AttributeError(
                "No segmentation map has been loaded. Please run "
                "`ClusterModels.load_segmentation()` to initialise the "
                "`seg_wcs` attribute.",
                name=None,
            )
        return self._seg_wcs

    @property
    def extra_mask(self) -> ArrayLike:
        """
        A mask of additional sources not included in the segmentation map.

        A 2D array where the extent of each source is described by a
        unique positive integer. A value of 0 corresponds to the
        background.

        These sources can overlap with existing sources in the
        segmentation map, and typically correspond to fainter objects that
        can only be identified in the residual map.
        """
        if self._extra_mask is None:
            raise AttributeError(
                "No additional mask has been loaded.",
                name=None,
            )
        return self._extra_mask

    def load_bcg_catalogue(
        self,
        bcg_catalogue: os.PathLike | Table,
        ra_key: str = "ra",
        dec_key: str = "dec",
        unit: str | u.Unit | tuple[u.Unit] | tuple[str] = "deg",
        **skycoord_kwargs,
    ) -> None:
        """
        Load a catalogue of galaxies to be modelled.

        Parameters
        ----------
        bcg_catalogue : `~os.PathLike` | `~astropy.table.Table`
            Either the filepath of the catalogue, or the catalogue in the
            form of a `~astropy.table.Table`.
        ra_key : str
            The column name of the R.A. world coordinate, by default "ra".
        dec_key : str
            The column name of the Dec. coordinate, by default "dec".
        unit : str | u.Unit | tuple[u.Unit] | tuple[str], optional
            The units of the world coordinates. If heteregenous
            coordinates are used, these should be described using a tuple
            of different units. Degrees are assumed by default.
        **skycoord_kwargs : dict, optional
            Additional keyword arguments to pass to
            `~astropy.coords.SkyCoord`.
        """

        if isinstance(bcg_catalogue, Table):
            self._bcg_cat = bcg_catalogue
        else:
            self._bcg_cat = Table.read(bcg_catalogue)

        self._bcg_cat.rename_columns(
            self._bcg_cat.colnames, [n.lower() for n in self._bcg_cat.colnames]
        )

        if ra_key is None:
            ra_key = "ra"
        if dec_key is None:
            dec_key = "dec"

        try:
            self._bcg_coords = SkyCoord(
                ra=self._bcg_cat[ra_key],
                dec=self._bcg_cat[dec_key],
                unit=unit,
                **skycoord_kwargs,
            )
        except Exception as e:
            raise ValueError(
                "Failed to load bCG coordinates from supplied catalogue."
            ) from e
        return

    def load_image(
        self,
        image_path: os.PathLike,
        image_hdu_index: int = 0,
        **background_kwargs,
    ) -> None:
        """
        Load the image containing the galaxies of interest.

        Also calculates the background and background RMS.

        Parameters
        ----------
        image_path : `~os.PathLike`
            The path of the original image, containing the
            measured fluxes.
        image_hdu_index : int, optional
            The index of the HDU containing the science
            image, by default 0.
        **background_kwargs : dict, optional
            Additional keyword arguments to pass to
            `~photutils.background.Background2D`.
        """
        # Make modular, should be possible to reload image without redoing map

        # We assume the data can fit into memory - AstroPy does not respect
        # IO context managers with `memmap=True`, which makes resource management
        # difficult. (See https://github.com/astropy/astropy/issues/7404 )
        with fits.open(image_path, memmap=False) as img_hdul:
            self._img_data = img_hdul[image_hdu_index].data.copy()
            self._img_hdr = img_hdul[image_hdu_index].header.copy()
            self._img_wcs = WCS(self.img_hdr)

        default_bkg_params = {"box_size": 20, "filter_size": 11}
        for k, v in default_bkg_params.items():
            if k not in background_kwargs.keys():
                background_kwargs[k] = v

        self._background = Background2D(data=self.img_data, **background_kwargs)

        return

    # def _check_wcs(self):

    # def load_segmentation(
    #     seg_cat : os.PathLike | Table,
    #     seg_map : os.PathLike | ArrayLike,
    #     seg_map_hdu_index: int = 0,
    #     seg_wcs : WCS = None,
    # ):

    #     if isinstance(seg_cat, Table):
    #         self.seg_cat = seg_cat
    #     else:
    #         self.seg_cat = Table.read(seg_cat)

    #     if isinstance(seg_map, os.PathLike):
    #         with fits.open(seg_map, memmap=False) as seg_hdul:
    #             self.seg_map = seg_hdul[seg_map_hdu_index].data.copy()
    #             self.seg_wcs = WCS(self.seg_hdul[seg_map_hdu_index].header.copy())

    # # load into memory, match against bcg cat

    def gen_segmentation(
        self,
        threshold: float | ArrayLike = 3,
        npixels: int = 20,
        thresh_abs: bool = False,
        kern_fwhm: float = 3,
        kern_size: int = 21,
        mask: ArrayLike = None,
        overwrite=False,
        **seg_kwargs,
        # ) -> tuple[ArrayLike, Table, WCS]:
    ) -> fits.HDUList:
        """
        Generate a segmentation map from the existing image.

        Parameters
        ----------
        threshold : float | ArrayLike, optional
            _description_, by default 3.
        npixels : int, optional
            _description_, by default 20.
        thresh_abs : bool, optional
            _description_, by default False.
        kern_fwhm : float, optional
            _description_, by default 3.
        kern_size : int, optional
            _description_, by default 21.
        mask : ArrayLike, optional
            _description_, by default None.
        overwrite : bool, optional
            _description_, by default False.
        **seg_kwargs : dict, optional
            Additional keyword arguments to pass to the
            `~photutils.segmentation.SourceFinder`.

        Returns
        -------
        fits.HDUList
            _description_.
        """
        kernel = make_2dgaussian_kernel(kern_fwhm, size=kern_size)
        convolved_data = convolve(self._img_data - self._background.background, kernel)

        if not thresh_abs:
            threshold *= self._background.background_rms

        if "contrast" not in seg_kwargs.keys():
            seg_kwargs["contrast"] = 0.01

        finder = SourceFinder(npixels=npixels, **seg_kwargs)
        segm = finder(convolved_data, threshold=threshold, mask=mask)
        self._seg_map = segm.data
        _seg_cat = SourceCatalog(
            self._img_data - self._background.background,
            segm,
            background=self._background.background,
            wcs=self._img_wcs,
        )

        self._seg_cat = _seg_cat.to_table(
            columns=req_seg_cat_props,
        )
        self._seg_cat["ra"] = _seg_cat.sky_centroid.ra
        self._seg_cat["dec"] = _seg_cat.sky_centroid.dec

        self._seg_wcs = self._img_wcs.copy()
        _seg_hdr = fits.Header()
        _seg_hdr.update(self._seg_wcs.to_header())

        seg_hdul = fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=self._seg_map, header=_seg_hdr, name="SEG_MAP"),
                # seg_tbl_hdu,
                fits.BinTableHDU(data=self._seg_cat, name="SEG_CAT"),
            ]
        )
        seg_hdul.writeto(
            self.out_dir / f"{self.base_name}_seg.fits", overwrite=overwrite
        )

        return seg_hdul

    def cutout_slice(
        self,
        obj_id: int,
        scale_padding: float | None = 1.5,
        pix_padding: int | None = None,
    ) -> tuple:
        """
        _summary_.

        Parameters
        ----------
        obj_id : int
            _description_.
        scale_padding : float | None, optional
            _description_, by default 1.5.
        pix_padding : int | None, optional
            _description_, by default None.

        Returns
        -------
        tuple
            _description_.
        """

        seg_idx = self._seg_cat["label"] == obj_id

        if scale_padding is not None:
            y_pad = int(
                (
                    self._seg_cat["bbox_ymax"][seg_idx]
                    - self._seg_cat["bbox_ymin"][seg_idx]
                )
                * (scale_padding - 1)
                * 0.5
            )
            x_pad = int(
                (
                    self._seg_cat["bbox_xmax"][seg_idx]
                    - self._seg_cat["bbox_xmin"][seg_idx]
                )
                * (scale_padding - 1)
                * 0.5
            )

        elif pix_padding is not None:
            y_pad = x_pad = int(pix_padding)

        else:
            y_pad = x_pad = 0

        cutout_idxs = (
            slice(
                np.max([int(self._seg_cat["bbox_ymin"][seg_idx]) - y_pad, 0]),
                np.min(
                    [
                        int(self._seg_cat["bbox_ymax"][seg_idx]) + y_pad,
                        self._seg_map.shape[0] - 1,
                    ]
                ),
            ),
            slice(
                np.max([int(self._seg_cat["bbox_xmin"][seg_idx]) - x_pad, 0]),
                np.min(
                    [
                        int(self._seg_cat["bbox_xmax"][seg_idx]) + x_pad,
                        self._seg_map.shape[1] - 1,
                    ]
                ),
            ),
        )

        # print (self._seg_cat["ycentroid"][seg_idx], float(self._seg_cat["xcentroid"][seg_idx]))

        new_centroid = (
            float(self._seg_cat["ycentroid"][seg_idx]) - cutout_idxs[0].start,
            float(self._seg_cat["xcentroid"][seg_idx]) - cutout_idxs[1].start,
        )

        return cutout_idxs, new_centroid

    # def run_init()
    # # First run -> _model_galaxy()
    # # re-run segmentation, gen and check mask
    # # re-run _model_galaxy()

    @staticmethod
    def _model_galaxy(
        cutout,
        mask,
        x0,
        y0,
        sma,
        eps=0.3,
        pa=1.0,
        recentre=False,
        plot=False,
        **fit_kwargs,
    ):

        data = ma.masked_array(cutout, mask)
        import matplotlib.pyplot as plt

        # plt.imshow(data, origin="lower")
        # plt.scatter(x0,y0,c="red")
        # plt.show()

        g = EllipseGeometry(
            x0=x0,
            y0=y0,
            sma=sma,
            eps=eps,
            pa=pa,
        )
        if recentre:
            g.find_center(data)
        ellipse = Ellipse(data, geometry=g)

        isolist = ellipse.fit_image(**fit_kwargs)

        iso_tab = isolist.to_table(columns="all")
        # iso_tab = isolist.to_table()
        # print (iso_tab.colnames)

        model_image = build_ellipse_model(data.shape, isolist, high_harmonics=True)
        residual = cutout - model_image

        if plot:
            import astropy.visualization as astrovis

            norm = astrovis.ImageNormalize(
                data=cutout,
                # stretch=astrovis.
                stretch=astrovis.LogStretch(),
                # interval=astrovis.PercentileInterval(99.9),
                interval=astrovis.ManualInterval(-0.1, 50),
            )
            fig, axs = plt.subplots(1, 3)  # , sharex=True, sharey=True)
            for i, (a, im) in enumerate(zip(axs, [data, model_image, residual])):
                a.imshow(im, origin="lower", norm=norm)

            plt.show()

        return model_image, iso_tab

    # def iterate_models()
    # # Store progress in a multi-extension FITS file
    # # ext 0 : image
    # # ext 1 : background
    # # ext 2 : background.rms
    # # ext 3 : seg map
    # # ext 4 : additional mask
    # # ext 5 : all models
    # # ext 6 : models table (hdr: "ITERNUM"=iteration, "CURRMOD"=model about to be run)
    # # i.e. ITERNUM=3, CURRMOD=0 indicates 0, 1, and 2nd iterations fully complete
    # # OR: MODELHIST as historical tables

    # # Need to load PSF somewhere
    # # Keep PSF deconv as a flag?


# def cutout_galaxy(
#     data: ArrayLike,
#     seg_map: ArrayLike,
#     gal_id: int,
#     expand_factor: float = 1.5,
# ) -> tuple[ArrayLike, ArrayLike]:

#     # This function could (should) be cythonised

#     x_locs, y_locs = (seg_map == gal_id).nonzero()
#     # print (x_locs)
#     x_range = int((np.nanmax(x_locs) - np.nanmin(x_locs)) * (expand_factor - 1) * 0.5)
#     y_range = int((np.nanmax(y_locs) - np.nanmin(y_locs)) * (expand_factor - 1) * 0.5)

#     cutout_idxs = (
#         slice(
#             np.max([np.nanmin(x_locs) - x_range, 0]),
#             np.min([np.nanmax(x_locs) + x_range, data.shape[0] - 1]),
#         ),
#         slice(
#             np.max([np.nanmin(y_locs) - y_range, 0]),
#             np.min([np.nanmax(y_locs) + y_range, data.shape[1] - 1]),
#         ),
#     )
#     print(cutout_idxs)

#     cutout = data[cutout_idxs]
#     mask = (seg_map[cutout_idxs] > 0) & (seg_map[cutout_idxs] != gal_id)

#     # cutout = data[np.nanmin(x_locs)-x_range//2:np.nanmax(x_locs)+x_range//2,np.nanmin(y_locs)-y_range//2:np.nanmax(y_locs)+y_range//2]

#     return cutout, mask


# def cutout_galaxy_seg_cat(
#     data: ArrayLike,
#     seg_map: ArrayLike,
#     gal_id: int,
#     seg_cat: ArrayLike,
#     expand_factor: float = 1.5,
# ) -> tuple[ArrayLike, ArrayLike, tuple[float, float]]:

#     # This function could (should) be cythonised

#     # x_locs, y_locs = (seg_map == gal_id).nonzero()
#     # print (x_locs)
#     # import matplotlib.pyplot as plt
#     # plt.imshow(seg_map)
#     # plt.show()
#     # plt.imshow(seg_map == gal_id)
#     # plt.show()
#     # x_range = int((np.nanmax(x_locs)-np.nanmin(x_locs))*(expand_factor-1)*0.5)
#     # y_range = int((np.nanmax(y_locs)-np.nanmin(y_locs))*(expand_factor-1)*0.5)

#     x_range = int(
#         (seg_cat["bbox_ymax"] - seg_cat["bbox_ymin"]) * (expand_factor - 1) * 0.5
#     )
#     y_range = int(
#         (seg_cat["bbox_xmax"] - seg_cat["bbox_xmin"]) * (expand_factor - 1) * 0.5
#     )

#     cutout_idxs = (
#         slice(
#             np.max([seg_cat["bbox_ymin"] - x_range, 0]),
#             np.min([seg_cat["bbox_ymax"] + x_range, data.shape[0] - 1]),
#         ),
#         slice(
#             np.max([seg_cat["bbox_xmin"] - y_range, 0]),
#             np.min([seg_cat["bbox_xmax"] + y_range, data.shape[1] - 1]),
#         ),
#     )
#     print(cutout_idxs)

#     cutout = data[cutout_idxs]
#     mask = (seg_map[cutout_idxs] > 0) & (seg_map[cutout_idxs] != gal_id)

#     # cutout = data[np.nanmin(x_locs)-x_range//2:np.nanmax(x_locs)+x_range//2,np.nanmin(y_locs)-y_range//2:np.nanmax(y_locs)+y_range//2]
#     new_centroid = (
#         seg_cat["xcentroid"] - cutout_idxs[1].start,
#         seg_cat["ycentroid"] - cutout_idxs[0].start,
#     )
#     return cutout, mask, new_centroid, cutout_idxs
