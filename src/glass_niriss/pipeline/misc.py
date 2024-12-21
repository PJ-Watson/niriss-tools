"""Additional functions used in the pipeline."""

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["seg_slice"]


def seg_slice(
    seg_map: ArrayLike,
    seg_id: int,
    padding: int = 50,
) -> tuple:
    """
    Find the location of an object in a segmentation map.

    Parameters
    ----------
    seg_map : ArrayLike
        The segmentation map, where each pixel value indicates the ID of
        the object to which that pixel belongs.
    seg_id : int
        The ID of the object to find.
    padding : int, optional
        The extra padding to add to the object boundaries, by default 50.

    Returns
    -------
    tuple
        The indices of the object in the array.
    """

    x_idxs, y_idxs = (seg_map == seg_id).nonzero()

    x_min = np.nanmin(x_idxs)
    x_max = np.nanmax(x_idxs)
    y_min = np.nanmin(y_idxs)
    y_max = np.nanmax(y_idxs)

    new_idxs = (
        slice(
            int(np.nanmax([x_min - padding, 0])),
            int(np.nanmin([x_max + padding, seg_map.shape[0]])),
        ),
        slice(
            int(np.nanmax([y_min - padding, 0])),
            int(np.nanmin([y_max + padding, seg_map.shape[1]])),
        ),
    )
    return new_idxs
