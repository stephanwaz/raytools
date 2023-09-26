#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""The setup script."""
import sys
import shutil
import os
from glob import glob

from setuptools import find_packages, setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['clasp', 'numpy', 'scipy', 'tqdm']

setup_requirements = ["setuptools", "wheel"]

test_requirements = ['pytest', 'pytest-cov']

if sys.platform.startswith('darwin'):
    radexec = glob('radiance_depend/macos/*')
else:
    radexec = glob('radiance_depend/linux/*')

radexec = [rade for rade in radexec if
           shutil.which(os.path.basename(rade)) is None]
if len(radexec) > 0:
    data_files = [('bin', radexec)]
else:
    data_files = []
package_data = {"raytools": ["cal/*.cal"]}

setup(
    author="Stephen Wasilewski",
    author_email='stephanwaz@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        ],
    description="working with hdr images, numpy and coordinate transformations "
                "for lighting simulation",
    python_requires=">=3.6.8",
    entry_points={
        'console_scripts': ['raytools=raytools.cli:main'],
        },
    install_requires=requirements,
    license="Mozilla Public License 2.0 (MPL 2.0)",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='raytools',
    name='raytools',
    packages=find_packages(),
    data_files=data_files,
    package_data=package_data,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/stephanwaz/raytools',
    version='0.1.3',
    zip_safe=False,
    )
