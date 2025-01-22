#!/usr/bin/python3
import os
from setuptools import setup, find_packages, Extension
import numpy as np
from Cython.Build import cythonize

name = 'unipolator'
annotate = False


# make a list of the .pyx files in the os.join.path("src",name) directory
#pyx_files = [os.path.splitext(fn)[0] for fn in os.listdir(os.path.join('src', name)) if fn.endswith(".pyx")] 
pyx_files = ['blas_functions', 'blas_functions_vectors', 'exp_and_log', 'indexing', 'caching', 'unitary_interpolation', 'hamiltonian_system', 'symmetric_unitary_interpolation', 'trotter_system', 'sym_trotter_system']
pyx_files += ['unitary_interpolation_vector', 'symmetric_unitary_interpolation_vector', 'trotter_system_vector', 'sym_trotter_system_vector', 'hamiltonian_system_vector']
#pyx_files += ['autobinning']

# check which operating system we are on
# Linux?
if os.name == 'posix':   # Linux or Mac OSX
    extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-fopenmp"]
else:  # Windows?
    extra_compile_args = ["/O2", "/openmp"]
    
include_dirs = [np.get_include()]
extensions = [Extension(name+'.'+filename, [ os.path.join('src', name, filename+'.pyx')], include_dirs=include_dirs, extra_compile_args=extra_compile_args) for filename in pyx_files]

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

compiler_directives = {'emit_code_comments': False, 'initializedcheck': False, 'boundscheck': False, 'wraparound': False, 'language_level': 3, "embedsignature": True, "cdivision": True, "nonecheck" : False, 'profile': False}
extensions = cythonize(extensions, language_level = "3", annotate=annotate, compiler_directives=compiler_directives ) #gdb_debug=True, 

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-dev.txt") as fp:
    dev_requires = fp.read().strip().split("\n")

setup(   
    name = "unipolator",
    #zip_safe = False,
    version           = "0.3.7",
    author = "Michael Schilling",
    author_email = "michael@ntropic.de",
    description  = "Unipolator allows for n dimensional unitary interpolation, and the calculation of propagators using unitary interpolation. Speeds up your propagators for linear quantum systems.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Ntropic/unipolator",
    download_url = "https://github.com/Ntropic/unipolator/archive/refs/tags/v0.3.7.tar.gz",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    ext_modules = extensions,
    install_requires = install_requires,
    extras_require = {
        "dev": dev_requires,
        "docs": ["sphinx", "sphinx-rtd-theme"]
    },
)
