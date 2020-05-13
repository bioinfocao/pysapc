"""
Setup sparseAP_cy Cython binary
Generate .so(linux/mac) or .pyd(windows) file by:
python setup_sparseAP.py build_ext --inplace
move the .so/.pyd file to pysapc folder or other python paths
"""
# Authors: Huojun Cao <bioinfocao at gmail.com>
# License: BSD 3 clause

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

try:
    # try with fopenmp
    ext = Extension("sparseAP_cy", ['sparseAP_cy.pyx'],
            extra_compile_args=['-fopenmp'],
            extra_link_args=['-fopenmp'],
            include_dirs=[numpy.get_include()],
            )
    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [ext],
    )
except:
    # if fopenmp is not installed
    ext = Extension("sparseAP_cy", ['sparseAP_cy.pyx'],
            include_dirs=[numpy.get_include()],
            )
    setup(
        cmdclass = {'build_ext': build_ext},
        ext_modules = [ext],
    )