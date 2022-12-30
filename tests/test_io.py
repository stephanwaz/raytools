#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytools.io"""
import os
import re

import pytest
from raytools import io
import numpy as np


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    cpath = os.getcwd()
    os.chdir(data)
    yield data
    os.chdir(cpath)


def test_array2img(tmpdir):
    b, a = np.mgrid[0:600, 0:400]
    ar = a*b
    io.array2hdr(ar, 'mgrid.hdr')
    io.array2hdr(a, 'mgrida.hdr')
    io.array2hdr(b, 'mgridb.hdr')
    a2 = io.hdr2array('mgrida.hdr')
    b2 = io.hdr2array('mgridb.hdr')
    ar2 = io.hdr2array('mgrid.hdr')
    assert np.allclose(a.T, a2, atol=.25, rtol=.03)
    assert np.allclose(b.T, b2, atol=.25, rtol=.03)
    assert np.allclose(ar.T, ar2, atol=.25, rtol=.03)
    a3 = io.hdr2carray('mgrid.hdr')
    assert np.allclose(a3[0], a3[1])
    io.array2hdr(a3, 'cgrid.hdr')
    a4 = np.swapaxes(io.hdr2carray('cgrid.hdr'), 1, 2)
    assert np.allclose(a3, a4, atol=.25, rtol=.03)


def test_setproc():
    nproc = io.get_nproc(8)
    assert nproc == 8
    nproc = io.get_nproc()
    assert nproc == os.cpu_count()
    io.set_nproc(7)
    assert io.get_nproc() == 7
    io.unset_nproc()
    assert io.get_nproc() == os.cpu_count()
    io.unset_nproc()
    with pytest.raises(ValueError):
        io.set_nproc("7")
    assert io.get_nproc() == os.cpu_count()
    io.set_nproc(6)
    assert io.get_nproc() == 6
    io.set_nproc(None)
    assert io.get_nproc() == 6
    io.set_nproc(0)
    assert io.get_nproc() == os.cpu_count()


def test_npbytefile(tmpdir):
    a = np.arange(999).reshape(-1, 9)
    io.np2bytefile(a, "test_npbytefile")
    c = io.bytefile2np(open("test_npbytefile", 'rb'), (-1, 9))
    assert np.all(a == c)


def test_load_txt(tmpdir):
    np.savetxt("farray.txt", np.arange(100))
    f = open("bad.txt", 'w')
    f.write("a b, f, f\n")
    f.close()
    with pytest.raises(FileNotFoundError):
        a = io.load_txt("farray.tsv")
    with pytest.raises(ValueError):
        a = io.load_txt("bad.txt")
    with pytest.raises(TypeError):
        a = io.load_txt(123)
    a = io.load_txt("farray.txt")
    assert np.allclose(a, np.arange(100))
