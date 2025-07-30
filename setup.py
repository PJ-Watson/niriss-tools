"""
Setup Cython modules.
"""

import os
from glob import glob

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

ext_kwargs = {
    "include_dirs": [numpy.get_include(), "src/niriss_tools/c_utils"],
    "define_macros": [
        ("_USE_MATH_DEFINES", "1"),
        ("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION"),
    ],
    "extra_compile_args": ["-O3", "-fopenmp"],
}
extensions = [
    Extension(
        "niriss_tools.c_utils.isophote_model",
        sources=["src/niriss_tools/c_utils/isophote_model.pyx"],
        depends=["src/niriss_tools/c_utils/worker.h"],
        extra_link_args=["-fopenmp"],
        **ext_kwargs,
    ),
    Extension(
        "niriss_tools.c_utils.array_ops",
        sources=["src/niriss_tools/c_utils/array_ops.pyx"],
        **ext_kwargs,
    ),
]
extensions = cythonize(
    extensions,
    language_level=3,
)
setup(ext_modules=extensions)
