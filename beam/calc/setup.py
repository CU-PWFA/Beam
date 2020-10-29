#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 14:46:39 2017

@author: robert

Run from the top folder with the command:
    'python calc/setup.py build_ext --inplace'
"""

import numpy
import venv
import os
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

# On windows the math library is already linked
if os.name == 'nt':
    ext_modules=[
        Extension('calc.*',
                  sources=['beam/calc/*.pyx'], #Compile entire module
                  #sources=['beam/calc/electron.pyx', 'beam/calc/ionization.pyx',
                  #         'beam/calc/plasma.pyx', 'beam/calc/laser.pyx'],
                  #libraries=["fftw3"],
                  include_dirs=[numpy.get_include(), 
                                os.path.join(venv.sys.base_prefix, 'include')],
                  extra_compile_args = [],
                  extra_link_args=['-fopenmp'],
                  #define_macros=[('CYTHON_TRACE', '1')],
                  language='c++'
        )
    ]
else:  
    ext_modules=[
        Extension('calc.*',
                  sources=['beam/calc/*.pyx'], #Compile entire module
                  #sources=['beam/calc/ionization.pyx'],
                  #sources=['beam/calc/laser.pyx'], #Compile specific files
                  libraries=["m"],
                  include_dirs=[numpy.get_include(), 
                                os.path.join(venv.sys.base_prefix, 'include')],
                  extra_compile_args = ['-march=native', '-fopenmp', '-O3'],
                  extra_link_args=['-fopenmp'],
                  #define_macros=[('CYTHON_TRACE', '1')]
        )
    ]

setup(ext_modules=cythonize(ext_modules))
