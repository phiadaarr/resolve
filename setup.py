# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019-2020 Max-Planck-Society
# Copyright(C) 2022 Max-Planck-Society, Philipp Arras
# Author: Philipp Arras

import sys
import os.path
import itertools
from glob import iglob
import os

from setuptools import setup, Extension, find_packages
import pybind11

tmp = os.getenv("DUCC0_CFLAGS", "").split(" ")
user_cflags = [x for x in tmp if x != ""]
tmp = os.getenv("DUCC0_LFLAGS", "").split(" ")
user_lflags = [x for x in tmp if x != ""]
tmp = os.getenv("DUCC0_FLAGS", "").split(" ")
tmp = [x for x in tmp if x != ""]
user_cflags += tmp
user_lflags += tmp

compilation_strategy = os.getenv('DUCC0_OPTIMIZATION', 'native-strip')
if compilation_strategy not in ['none', 'none-debug', 'none-strip', 'portable', 'portable-debug', 'portable-strip', 'native', 'native-debug', 'native-strip']:
    raise RuntimeError('unknown compilation strategy')
do_debug = compilation_strategy in ['none-debug', 'portable-debug', 'native-debug']
do_strip = compilation_strategy in ['none-strip', 'portable-strip', 'native-strip']
do_optimize = compilation_strategy not in ['none', 'none-debug', 'none-strip']
do_native = compilation_strategy in ['native', 'native-debug', 'native-strip']

def _print_env():
    import platform
    print("")
    print("Build environment:")
    print("Platform:     ", platform.platform())
    print("Machine:      ", platform.machine())
    print("System:       ", platform.system())
    print("Architecture: ", platform.architecture())
    print("")

def _get_files_by_suffix(directory, suffix):
    path = directory
    iterable_sources = (iglob(os.path.join(root, '*.'+suffix))
                        for root, dirs, files in os.walk(path))
    return list(itertools.chain.from_iterable(iterable_sources))


include_dirs = ['./resolvelib/ducc/src',
                pybind11.get_include(True),
                pybind11.get_include(False)]

extra_compile_args = ['-std=c++17', '-fvisibility=hidden']

if do_debug:
    extra_compile_args += ['-g']
else:
    extra_compile_args += ['-g0']

if do_optimize:
    extra_compile_args += ['-ffast-math', '-O3']
else:
    extra_compile_args += ['-O0']

if do_native:
    extra_compile_args += ['-march=native']

python_module_link_args = []

if sys.platform == 'darwin':
    import distutils.sysconfig
    extra_compile_args += ['-mmacosx-version-min=10.14', '-pthread']
    python_module_link_args += ['-mmacosx-version-min=10.14', '-pthread']
elif sys.platform == 'win32':
    extra_compile_args = ['/EHsc', '/std:c++17']
    if do_optimize:
        extra_compile_args += ['/Ox']
else:
    if do_optimize:
        extra_compile_args += ['-Wfatal-errors',
                               '-Wfloat-conversion',
                               '-W',
                               '-Wall',
                               '-Wstrict-aliasing',
                               '-Wwrite-strings',
                               '-Wredundant-decls',
                               '-Woverloaded-virtual',
                               '-Wcast-qual',
                               '-Wcast-align',
                               '-Wpointer-arith']
    extra_compile_args += ['-pthread']
    python_module_link_args += ['-Wl,-rpath,$ORIGIN', '-pthread']
    if do_native:
        python_module_link_args += ['-march=native']
    if do_strip:
        python_module_link_args += ['-s']

extra_compile_args += user_cflags
python_module_link_args += user_lflags

depfiles = (_get_files_by_suffix('.', 'h') +
            _get_files_by_suffix('.', 'cc') +
            ['setup.py'])

extensions = [Extension("resolvelib",
                        sources=['resolvelib/resolvelib.cc',
                                 'resolvelib/ducc/src/ducc0/infra/threading.cc'],
                        depends=depfiles,
                        include_dirs=include_dirs,
                        extra_compile_args=extra_compile_args,
                        extra_link_args=python_module_link_args)]

_print_env()

setup(
    name="resolve",
    author="Philipp Arras",
    author_email="parras@mpa-garching.mpg.de",
    description="Radio imaging with information field theory",
    url="https://gitlab.mpcdf.mpg.de/ift/resolve",
    packages=find_packages(include=["resolve", "resolve.*"]),
    zip_safe=True,
    dependency_links=[],
    install_requires=["ducc0"],  # and nifty8
    ext_modules=extensions,
    entry_points={"console_scripts":
        [
            "resolve=resolve.command_line.resolve:main",
            "plot-sky=resolve.ubik_tools.plot_sky_hdf5:cmdline_visualize_sky_hdf5",
        ]},
    license="GPLv3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 " "or later (GPLv3+)",
    ],
)
