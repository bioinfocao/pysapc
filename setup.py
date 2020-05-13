from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

try:
        # try with fopenmp
        ext = Extension("pysapc.sparseAP_cy", ['pysapc/sparseAP_cy.pyx'],
                extra_compile_args=['-fopenmp'],
                extra_link_args=['-fopenmp'],
                include_dirs=[numpy.get_include()],
                )
        setup(
        name="pysapc",
        version="1.2.0",
        description="Sparse Affinity Propagation Clustering",
        author="Huojun Cao",
        author_email="bioinfocao@gmail.com",
        url="https://github.com/bioinfocao/pysapc",
        license="BSD 3 clause",
        packages=["pysapc","pysapc.tests"],
        #packages = find_packages(), 
        package_data = {
                # If any package contains *.txt or *.rst files, include them:
                '': ['*.txt', '*.rst'],
                },
        include_package_data=True,
        install_requires=["numpy","scipy","pandas","cython"],
        cmdclass = {"build_ext": build_ext},
        ext_modules = [ext],
        setup_requires=['wheel']
        )
except:
    # if fopenmp is not installed
        ext = Extension("pysapc.sparseAP_cy", ['pysapc/sparseAP_cy.pyx'],
                include_dirs=[numpy.get_include()],
                )
        setup(
        name="pysapc",
        version="1.2.0",
        description="Sparse Affinity Propagation Clustering",
        author="Huojun Cao",
        author_email="bioinfocao@gmail.com",
        url="https://github.com/bioinfocao/pysapc",
        license="BSD 3 clause",
        packages=["pysapc","pysapc.tests"],
        #packages = find_packages(), 
        package_data = {
                # If any package contains *.txt or *.rst files, include them:
                '': ['*.txt', '*.rst'],
                },
        include_package_data=True,
        install_requires=["numpy","scipy","pandas","cython"],
        cmdclass = {"build_ext": build_ext},
        ext_modules = [ext],
        setup_requires=['wheel']
        )
