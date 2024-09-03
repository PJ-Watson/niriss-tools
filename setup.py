"""
Setup Cython modules.
"""

import os
from glob import glob

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

# sourcefiles = [
#     os.path.join("src", "glass_niriss", "c_utils", "isophote_model.pyx")
# ]  # +glob(os.path.join("src", "glass_niriss","c_utils","*.c"))
# sourcefiles = glob(os.path.join("src", "glass_niriss", "c_utils", "*.pyx"))
# headerfiles = glob(os.path.join("src", "glass_niriss", "c_utils", "*.h"))
ext_kwargs = {
    "include_dirs": [numpy.get_include(), "src/glass_niriss/c_utils"],
    "define_macros": [
        ("_USE_MATH_DEFINES", "1"),
        ("NPY_NO_DEPRECATED_API", "NPY_2_0_API_VERSION"),
    ],
    "extra_compile_args": ["-O3", "-fopenmp"],
}
extensions = [
    Extension(
        "glass_niriss.c_utils.isophote_model",
        sources=["src/glass_niriss/c_utils/isophote_model.pyx"],
        depends=["src/glass_niriss/c_utils/worker.h"],
        # extra_compile_args=["-fopenmp"],
        extra_link_args=["-fopenmp"],
        **ext_kwargs,
    ),
    Extension(
        "glass_niriss.c_utils.array_ops",
        sources=["src/glass_niriss/c_utils/array_ops.pyx"],
        **ext_kwargs,
    ),
]
extensions = cythonize(
    extensions,
    language_level=3,
)
setup(ext_modules=extensions)
