#!/usr/bin/python3
import os
from setuptools import setup, find_packages, Extension
import numpy as np

name = 'unipolator'
annotate = False

try:
    from Cython.Build import cythonize
    print('Compiling via cython')
except ImportError:
    print('Not using Cythonize, due to an ImportError')
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

# make a list of the .pyx files in the os.join.path("src",name) directory
#pyx_files = [os.path.splitext(fn)[0] for fn in os.listdir(os.path.join('src', name)) if fn.endswith(".pyx")] 
pyx_files = ['blas_functions', 'blas_functions_vectors', 'exp_and_log', 'indexing', 'caching', 'unitary_interpolation',  'symmetric_unitary_interpolation', 'hamiltonian_system', 'trotter_system', 'sym_trotter_system']
#pyx_files += ['autobinning']
pyx_files += ['unitary_interpolation_vector', 'symmetric_unitary_interpolation_vector', 'trotter_system_vector', 'sym_trotter_system_vector', 'hamiltonian_system_vector']

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

CYTHONIZE = True if cythonize is not None else False
if CYTHONIZE:
    compiler_directives = {'initializedcheck': False, 'boundscheck': False, 'wraparound': False, 'language_level': 3, "embedsignature": True, "cdivision": True, "nonecheck" : False, 'profile': False}
    extensions = cythonize(extensions, language_level = "3", annotate=annotate, compiler_directives=compiler_directives ) #gdb_debug=True, 
else:
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-dev.txt") as fp:
    dev_requires = fp.read().strip().split("\n")

setup(   
    name = "unipolator",
    #zip_safe = False,
    version         = "0.3.5",
    author = "Michael Schilling",
    author_email = "michael@ntropic.de",
    description  = "Unipolator allows for n dimensional unitary interpolation, and the calculation of propagators using unitary interpolation. Speeds up your propagators for linear quantum systems.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Ntropic/unipolator",
    download_url = "https://github.com/Ntropic/unipolator/archive/refs/tags/v0.3.5.tar.gz",
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
