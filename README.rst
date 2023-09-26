====================
raytools (0.1.3)
====================

.. image:: https://img.shields.io/pypi/v/raytools?style=flat-square
    :target: https://pypi.org/project/raytools
    :alt: PyPI

.. image:: https://img.shields.io/pypi/l/raytools?style=flat-square
    :target: https://www.mozilla.org/en-US/MPL/2.0/
    :alt: PyPI - License

.. image:: https://img.shields.io/readthedocs/raytools/stable?style=flat-square
    :target: https://raytools.readthedocs.io/en/stable/
    :alt: Read the Docs (version)

.. image:: https://img.shields.io/coveralls/github/stephanwaz/raytools?style=flat-square
    :target: https://coveralls.io/github/stephanwaz/raytools
    :alt: Coveralls github

working with hdr images, numpy and coordinate transformations for lighting simulation (basic libraries for raytraverse).
Installs with a command line tool for calculating daylight metrics from angular fisheye images (similar to evalglare).

* Free software: Mozilla Public License 2.0 (MPL 2.0)
* Documentation: https://raytools.readthedocs.io/en/latest/.


Installation
------------

Raytools requires MacOS or Linux operating system with python >=3.7. For
windows you can consider using docker (https://www.docker.com/products/docker-desktop/ ).
raytools can be installed in the python:3.9 container.

The easiest way to install raytools is with pip::

    pip install --upgrade pip
    pip install raytools

or if you have cloned this repository::

    cd path/to/this/file
    pip install .

Note: raytools depends on a small subset of Radiance executables
(rcalc, vwrays, getinfo, pvalue). If these are not available in the $PATH at
the time of installation, then those precompiled (macos or linux) binaries will
be installed to the same bin/ as the current python (and raytools), which could be
in a virtualenv. This avoids overriding the existing install of these programs,
but it does require that you make sure that the enviroment at installation is the same as
when using raytools.

Getting Started
---------------
After installation, the raytools executable should be available from the command line.
See "raytools --help" and "raytools metric --help" for usage.