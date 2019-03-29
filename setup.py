# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

from setuptools import find_packages, setup


def write_version():
    import subprocess
    p = subprocess.Popen(
        ["git", "describe", "--dirty", "--tags", "--always"],
        stdout=subprocess.PIPE)
    res = p.communicate()[0].strip().decode('utf-8')
    with open("resolve/git_version.py", "w") as file:
        file.write('gitversion = "{}"\n'.format(res))


write_version()
exec(open('resolve/version.py').read())

setup(
    name="resolve",
    author="Philipp Arras",
    author_email="parras@mpa-garching.mpg.de",
    description="Radio imaging with information field theory",
    url="https://gitlab.mpcdf.mpg.de/ift/resolve",
    packages=find_packages(include=["resolve", "resolve.*"]),
    zip_safe=True,
    dependency_links=[],
    install_requires=["nifty5>=5.0"],
    license="GPLv3",
    classifiers=[
        "Development Status :: 3 - Alpha", "Topic :: Utilities",
        "License :: OSI Approved :: GNU General Public License v3 "
        "or later (GPLv3+)"
    ])
