import os
import sys

import pybind11, sysconfig
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

__version__ = "1.0.0"

ext_modules = [
    Extension(
        "outlier_detector",
        ["outlier_detector_py.cpp"],
        include_dirs=[pybind11.get_include(), pybind11.get_include(user=True), "/usr/include/opencv4", "/usr/lib/python3.10/site-packages/numpy"],
        language="c++",
        extra_compile_args=["-std=c++11"],
        extra_link_args=["-lopencv_core", "-lstdc++", "-lpython3.10", "-L/usr/lib/python3.10/config-3.10-x86_64-linux-gnu"],
        libraries=['opencv_core', 'opencv_imgcodecs', 'opencv_highgui', 'opencv_imgproc']
    )
]

setup(
    name="outlier_detector",
    version='0.0.1',
    author="Semen Saydumarov",
    description="Python interface for outlier_detector C++ code",
    long_description="",
    ext_modules=ext_modules,
    install_requires=["pybind11>=2.6.0"],
    setup_requires=["pybind11>=2.6.0"],
    cmdclass={'build_ext': build_ext},
    zip_safe=False,
)
