#!/usr/bin/python3
import os
from setuptools import setup, find_packages, Extension
import numpy as np

annotate = True
try:
    from Cython.Build import cythonize
    print('Compiling via cython')
except ImportError:
    cythonize = None

# https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#distributing-cython-modules
def no_cythonize(extensions, **_ignore):
    for extension in extensions:
        sources = []
        for sfile in extension.sources:
            path, ext = os.path.splitext(sfile)
            if ext in (".pyx", ".py"):
                if extension.language == "c++":
                    ext = ".cpp"
                else:
                    ext = ".c"
                sfile = path + ext
            sources.append(sfile)
        extension.sources[:] = sources
    return extensions

filename_list = ['blas_functions', 'exp_and_log', 'indexing', 'caching', 'symmetric_unitary_interpolation', 'unitary_interpolation', 'hamiltonian_system', 'trotter_system', 'sym_trotter_system']
include_dirs = [np.get_include()]
extra_compile_args = ["-O3", "-march=native"]
# extra_compile_args = ["/O2", "/arch:AVX512"]
# extra_link_args    = ["-O3", "-march=native"]
extensions = [Extension("unipolator."+filename, ["src/unipolator/"+filename+".pyx"], include_dirs=include_dirs, extra_compile_args=extra_compile_args) for filename in filename_list] #, extra_link_args=extra_link_args ),

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

CYTHONIZE = True if cythonize is not None else False
if CYTHONIZE:
    compiler_directives = {'initializedcheck': False, 'boundscheck': False, 'wraparound': False, 'language_level': 3, "embedsignature": True, "cdivision": True, "nonecheck" : False}
    extensions = cythonize(extensions, compiler_directives=compiler_directives, annotate=annotate)#, gdb_debug=True)
else:
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-dev.txt") as fp:
    dev_requires = fp.read().strip().split("\n")

setup(
    name = "unipolator",
    version = "0.1",
    author = "Michael Schilling",
    author_email = "michael@ntropic.de",
    description  = "Unipolator allows for n dimensional unitary interpolation, and the calculation of propagators using unitary interpolation. Speeds up your propagators for linear quantum systems.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Ntropic/unipolator/refs/tags/v0.1.tar.gz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules=extensions,
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "docs": ["sphinx", "sphinx-rtd-theme"]
    },
)
