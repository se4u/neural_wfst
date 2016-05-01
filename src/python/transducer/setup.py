import os
import numpy as np
from setuptools import setup
from distutils.extension import Extension

from Cython.Build import cythonize


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext



setup(name='transducer',
     author='Ryan Cotterell',
     description='Fast CRF WFST',
     version='1.0',
     include_dirs=[np.get_include()],
     ext_modules = cythonize(['src/*.pyx'], language="c++"))

