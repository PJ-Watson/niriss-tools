"""
Setup Cython modules.
"""

import os
from glob import glob

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

sourcefiles = [
    os.path.join("src", "glass_niriss", "c_utils", "isophote_model.pyx")
]  # +glob(os.path.join("src", "glass_niriss","c_utils","*.c"))
headerfiles = glob(os.path.join("src", "glass_niriss", "c_utils", "*.h"))
include_dirs = [numpy.get_include(), os.path.join("src", "glass_niriss", "c_utils")]
extensions = [
    Extension(
        "glass_niriss.c_utils",
        sourcefiles,
        include_dirs=include_dirs,
        depends=headerfiles,
        define_macros=[
            ("_USE_MATH_DEFINES", "1"),
            ("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION"),
        ],
        extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
    )
]
extensions = cythonize(
    extensions,
    language_level=3,
)
setup(ext_modules=extensions)
