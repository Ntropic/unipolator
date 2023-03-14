#!/usr/bin/python3
import numpy as np

import os
from setuptools import setup, find_packages, Extension
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
#pyx_files = [os.path.splitext(fn)[0] for fn in os.listdir(os.path.join("src",name)) if fn.endswith(".pyx")] #(".pyx", ".py"))]
pyx_files = ['blas_functions', 'exp_and_log', 'indexing', 'caching', 'symmetric_unitary_interpolation', 'unitary_interpolation', 'hamiltonian_system', 'trotter_system', 'sym_trotter_system']
extra_compile_args = ["-O3", "-march=native"]
# extra_compile_args = ["/O2", "/arch:AVX512"]
# extra_link_args    = ["-O3", "-march=native"]
include_dirs = [np.get_include()]
other_args = {"extra_compile_args": extra_compile_args, "include_dirs": include_dirs}
extensions = [Extension(name+"."+filename, ['src/'+name+'/'+filename+".pyx"], **other_args) for filename in pyx_files] #**other_args, extra_link_args=extra_link_args ),
#extensions = [Extension(name+"."+filename, [os.path.join("src",name,filename+".pyx")]) for filename in pyx_files] #**other_args, extra_link_args=extra_link_args ),
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

CYTHONIZE = True if cythonize is not None else False
#CYTHONIZE = bool(int(os.getenv("CYTHONIZE", 0))) and cythonize is not None
if CYTHONIZE:
    compiler_directives = {'initializedcheck': False, 'boundscheck': False, 'wraparound': False, 'language_level': 3, "embedsignature": True, "cdivision": True, "nonecheck" : False}
    extensions = cythonize(extensions, language_level = "3", annotate=annotate, compiler_directives=compiler_directives ) #gdb_debug=True, 
else:
    extensions = no_cythonize(extensions)

with open("requirements.txt") as fp:
    install_requires = fp.read().strip().split("\n")

with open("requirements-dev.txt") as fp:
    dev_requires = fp.read().strip().split("\n")

setup(   
    name = "unipolator",
    version             = "0.2.1",
    author = "Michael Schilling",
    author_email = "michael@ntropic.de",
    description  = "Unipolator allows for n dimensional unitary interpolation, and the calculation of propagators using unitary interpolation. Speeds up your propagators for linear quantum systems.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/Ntropic/unipolator/refs/tags/v0.2.1.tar.gz",
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
