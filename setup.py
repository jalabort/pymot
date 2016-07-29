from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
from numpy import get_include
from os import path


munkres_extension = Extension(
    'pymot.external.munkres.base',
    [path.join('pymot', 'external', 'munkres', 'base.pyx'),
     path.join('pymot', 'external', 'munkres', 'cpp', 'Munkres.cpp')],
    include_dirs = [get_include(),
                    path.join('pymot', 'external', 'munkres', 'cpp')],
    language='c++',
    pyrex_gdb=True
)


setup(
    name='pymot',
    version='0.0.0',
    ext_modules = cythonize([munkres_extension]),
    install_requires=['numpy', 'cython', 'tqdm'],
    packages=find_packages(),
    tests_require=['nose']
)
