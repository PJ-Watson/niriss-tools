# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module profiles tools for building a model elliptical galaxy image
from a list of isophotes.
"""


import numpy as np
from scipy.interpolate import LSQUnivariateSpline

# import ctypes as ct

cimport cython
cimport numpy as np

np.import_array()

cdef extern from "worker.c":
    pass

cdef extern from "worker.h":

    void worker(double* result, double* weight, int n_rows, int n_cols, int N, int high_harmonics,
		double* fss_arr, double* intens_arr, double* eps_arr, double* pa_arr,
		double* x0_arr, double* y0_arr, double* a3_arr, double* b3_arr, double* a4_arr, double* b4_arr,
		int nthreads)

@cython.boundscheck(False)
def build_ellipse_model(
    (int, int) shape,
    isolist,
    int nthreads=1,
    double fill=0.,
    bint high_harmonics=False
):
    """
    Build a model elliptical galaxy image from a list of isophotes.
    For each ellipse in the input isophote list the algorithm fills the
    output image array with the corresponding isophotal intensity.
    Pixels in the output array are in general only partially covered by
    the isophote "pixel".  The algorithm takes care of this partial
    pixel coverage by keeping track of how much intensity was added to
    each pixel by storing the partial area information in an auxiliary
    array.  The information in this array is then used to normalize the
    pixel intensities.
    Parameters
    ----------
    shape : 2-tuple
        The (ny, nx) shape of the array used to generate the input
        ``isolist``.
    isolist : `~photutils.isophote.IsophoteList` instance
        The isophote list created by the `~photutils.isophote.Ellipse`
        class.
    nthreads: float, optiomal
    	Number of threads to perform work. Default is 1 (serial code).
    fill : float, optional
        The constant value to fill empty pixels. If an output pixel has
        no contribution from any isophote, it will be assigned this
        value.  The default is 0.
    high_harmonics : bool, optional
        Whether to add the higher-order harmonics (i.e., ``a3``, ``b3``,
        ``a4``, and ``b4``; see `~photutils.isophote.Isophote` for
        details) to the result.
    Returns
    -------
    result : 2D `~numpy.ndarray`
        The image with the model galaxy.
    """

    # the target grid is spaced in 0.1 pixel intervals so as
    # to ensure no gaps will result on the output array.
    finely_spaced_sma = np.arange(isolist[0].sma, isolist[-1].sma, 0.1)

    # interpolate ellipse parameters

    # End points must be discarded, but how many?
    # This seems to work so far
    nodes = isolist.sma[2:-2]

    intens_array = LSQUnivariateSpline(
        isolist.sma, isolist.intens, nodes)(finely_spaced_sma)
    eps_array = LSQUnivariateSpline(
        isolist.sma, isolist.eps, nodes)(finely_spaced_sma)
    pa_array = LSQUnivariateSpline(
        isolist.sma, isolist.pa, nodes)(finely_spaced_sma)
    x0_array = LSQUnivariateSpline(
        isolist.sma, isolist.x0, nodes)(finely_spaced_sma)
    y0_array = LSQUnivariateSpline(
        isolist.sma, isolist.y0, nodes)(finely_spaced_sma)
    grad_array = LSQUnivariateSpline(
        isolist.sma, isolist.grad, nodes)(finely_spaced_sma)
    a3_array = LSQUnivariateSpline(
        isolist.sma, isolist.a3, nodes)(finely_spaced_sma)
    b3_array = LSQUnivariateSpline(
        isolist.sma, isolist.b3, nodes)(finely_spaced_sma)
    a4_array = LSQUnivariateSpline(
        isolist.sma, isolist.a4, nodes)(finely_spaced_sma)
    b4_array = LSQUnivariateSpline(
        isolist.sma, isolist.b4, nodes)(finely_spaced_sma)

    # Return deviations from ellipticity to their original amplitude meaning
    a3_array = -a3_array * grad_array * finely_spaced_sma
    b3_array = -b3_array * grad_array * finely_spaced_sma
    a4_array = -a4_array * grad_array * finely_spaced_sma
    b4_array = -b4_array * grad_array * finely_spaced_sma

    # correct deviations cased by fluctuations in spline solution
    eps_array[np.where(eps_array < 0.)] = 0.
    eps_array[np.where(eps_array < 0.)] = 0.05

    # convert everything to C-type array (pointers)
    cdef double[::1] c_fss_array = finely_spaced_sma
    cdef double[::1] c_intens_array = intens_array
    cdef double[::1] c_eps_array = eps_array
    cdef double[::1] c_pa_array = pa_array
    cdef double[::1] c_x0_array = x0_array
    cdef double[::1] c_y0_array = y0_array
    cdef double[::1] c_a3_array = a3_array
    cdef double[::1] c_b3_array = b3_array
    cdef double[::1] c_a4_array = a4_array
    cdef double[::1] c_b4_array = b4_array

    # initialize result and weight arrays, also as 1D ctype array
    result = np.zeros(shape=(shape[1]*shape[0],))
    cdef double[::1] result_memview = result
    weight = np.zeros(shape=(shape[1]*shape[0],))
    cdef double[::1] weight_memview = weight
    # cdef double* c_result = &result[0]
    # cdef double* c_weight = &weight[0]

    # convert high_harmnics bool flag to int,
    # convert all other ints to ctype
    cdef int c_high_harm = int(high_harmonics)
    cdef int c_N = len(finely_spaced_sma)
    cdef int c_nrows = shape[0]
    cdef int c_ncols = shape[1]

    # # load into C worker function (worker.so should be in same directory)
    # lib = ct.cdll.LoadLibrary('./worker.so')
    # lib.worker.restype = None
    # lib.
    worker(&result_memview[0], &weight_memview[0], c_nrows, c_ncols, c_N, c_high_harm,
               &c_fss_array[0], &c_intens_array[0], &c_eps_array[0], &c_pa_array[0],
               &c_x0_array[0], &c_y0_array[0], &c_a3_array[0], &c_b3_array[0],
               &c_a4_array[0], &c_b4_array[0],
               nthreads)

     # zero weight values must be set to 1.
    weight[np.where(weight <= 0.)] = 1.

    # normalize
    result /= weight

    # fill value
    result[np.where(result == 0.)] = fill

    # reshape
    result = result.reshape(shape[0], shape[1])

    return result
