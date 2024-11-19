"""
Array optimisations to reduce intermediate memory usage.
"""


import numpy as np

cimport cython
cimport numpy as np

np.import_array()

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

from libc.math cimport fabs, pow, sqrt


@cython.boundscheck(False)
@cython.wraparound(False)
def calc_chisq(
    # np.ndarray[DTYPE_t, ndim=2, mode='c'] models,
    # np.ndarray[DTYPE_t, ndim=1, mode='c'] obs,
    # np.ndarray[DTYPE_t, ndim=1, mode='c'] inv_sig_sq,
    DTYPE_t[:,::1] models,
    DTYPE_t[::1] obs,
    DTYPE_t[::1] inv_sig_sq,
):
    """
    Calculate the chi-squared values for an array of models.

    Parameters
    ----------
    models : 2D `~np.ndarray`
        An array of shape (N, M), for N models, and M observations.
    obs : 1D `~np.ndarray`
        A 1D array containing M observations.
    inv_sig_sq : 1D `~np.ndarray`
        A 1D array containing the inverse squared uncertainties for each
        of the M observations.

    Returns
    -------
    chisq : 1D `~np.ndarray`
        The calculated chi-squared values, of shape (N,).
    """

    cdef size_t i, j, I, J
    I = models.shape[0]
    J = models.shape[1]

    cdef DTYPE_t[::1] chisq = np.zeros(I, dtype=DTYPE)

    for i in range(I):
        for j in range(J):
            chisq[i] += pow(models[i,j]-obs[j], 2)*inv_sig_sq[j]

    return np.asarray(chisq)


@cython.boundscheck(False)
@cython.wraparound(False)
def assign_hex(
    DTYPE_t[:,::1] hex_centres,
    DTYPE_t diameter,
    tuple[int, int] shape,
):

    cdef size_t i, j, k, K
    K = hex_centres.shape[0]
    # J = signal.shape[1]
    cdef DTYPE_t radius = diameter * 0.5
    cdef DTYPE_t short_radius = 0.5*sqrt(3)*radius

    cdef np.int32_t[:,::1] labels = np.zeros(shape, dtype=np.int32)

    for i in range(shape[0]):
        for j in range(shape[1]):
            # hex_centres[i,j]
            for k in range(K):
                # print (i,j,k)
                # if k==0:
                # if j>=(hex_centres[k][0]-short_radius) and j<(hex_centres[k][0]+short_radius) and i>=(hex_centres[k][1]-radius) and i<(hex_centres[k][1]+radius):

                #     print (i, j, k, (hex_centres[k][0]-short_radius), (hex_centres[k][1]-radius),
                #         j>=(hex_centres[k][0]-short_radius),
                #         j<(hex_centres[k][0]+short_radius),
                #         i>=(hex_centres[k][1]-radius),
                #         i<(hex_centres[k][1]+radius),
                #         fabs(i-hex_centres[k][1]) < sqrt(3) * min(radius - fabs(j-hex_centres[k][0]), radius*0.5)
                #     )
                # print (
                #     i,
                #     j,
                #     k,
                #     hex_centres[k][1], hex_centres[k][0],
                #     fabs(i-hex_centres[k][1]),
                #     min(radius - fabs(j-hex_centres[k][0]), radius*0.5),
                # )
                # if nearly_less_equal(fabs(j-hex_centres[k][0]), sqrt(3) * min(radius - fabs(i-hex_centres[k][1]), radius*0.5)):
                if (fabs(j-hex_centres[k][0]) <= sqrt(3) * min(radius - fabs(i-hex_centres[k][1]), radius*0.5)):
                    labels[i][j]=k+1
                    break
    # for i in range(6):
    #     labels[0][i] = 21

    return np.asarray(labels)
