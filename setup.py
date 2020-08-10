# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright(C) 2019 Max-Planck-Society
# Author: Philipp Arras

from setuptools import find_packages, setup


setup(
    name="resolve",
    author="Philipp Arras",
    author_email="parras@mpa-garching.mpg.de",
    description="Radio imaging with information field theory",
    url="https://gitlab.mpcdf.mpg.de/ift/resolve",
    packages=find_packages(include=["resolve", "resolve.*"]),
    zip_safe=True,
    dependency_links=[],
    install_requires=["nifty7>=5.0"],
    license="GPLv3",
    classifiers=[
        "Development Status :: 3 - Alpha", "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 "
        "or later (GPLv3+)"
    ])
