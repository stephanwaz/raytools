#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytools.imagetools"""
import os
import shutil

import pytest
import numpy as np

from raytools import imagetools, io


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/testdir/', data + '/test')
    cpath = os.getcwd()
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/testdir'
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_imgmetric(tmpdir):
    metrics = ['illum', 'loggcr', 'dgp', 'ugp']
    img = "oct21_detail_glz_EW_desk.hdr"
    res = imagetools.imgmetric(img, metrics)
    res2 = imagetools.imgmetric(img, metrics, peakn=True)
    res3 = imagetools.imgmetric(img, metrics, peakn=True, blursun=True)
    res4 = imagetools.imgmetric(img, metrics, peakn=True, scale=1.79)
    res5 = imagetools.imgmetric(img, metrics, peakn=True, scale=1.79, blursun=True)
    assert np.allclose(res[0], [res2[0], res3[0], res4[0]*100, res5[0]*100])
    assert np.isclose(res2[1], res4[1])
    assert np.isclose(res3[1], res5[1])
    res6 = imagetools.imgmetric(img, metrics, peakn=True, blurtol=1)
    assert np.isclose(res2[0], res6[0])
    res7 = imagetools.imgmetric(img, metrics, peakn=True, lumrgb=(0.265, 0.670, 0.065))
    assert np.allclose(res2, res7)
    res7 = imagetools.imgmetric(img, metrics, peakn=False, lumrgb=(0.265, 0.670, 0.065))
    assert np.allclose(res, res7)
    res7 = imagetools.imgmetric(img, metrics, peakn=True, blurtol=1, lumrgb=(0.265, 0.670, 0.065))
    assert np.allclose(res6, res7)
    res7 = imagetools.imgmetric(img, metrics, peakn=True, blursun=True, lumrgb=(0.265, 0.670, 0.065))
    assert np.allclose(res3, res7)
    res7 = imagetools.imgmetric(img, metrics, peakt=1e18, peakn=True,
                                lumrgb=(0.265, 0.670, 0.065))
    res8 = imagetools.imgmetric(img, metrics, peakt=1e18, peakn=True)
    assert np.allclose(res8, res7)

    v, o, rgb = imagetools.hdr2vol(img, color=True)
    v, o, l = imagetools.hdr2vol(img)
    pvola, v, omega, lum = imagetools.normalize_peak(v, o, rgb, keepzeros=True, returnparts=True)
    pvolb, v, omega, lum = imagetools.normalize_peak(v, o, l, keepzeros=True, returnparts=True)
    assert np.allclose(pvola[0], pvolb[0])
    pvola, strays = imagetools.normalize_peak(v, o, rgb, findsecondary=True, peakt=1e3)
    pvolb, straysb = imagetools.normalize_peak(v, o, l, findsecondary=True, peakt=1e3)
    assert np.isclose(np.sum([s[1] for s in strays]), np.sum([s[1] for s in straysb]))


def test_imagetransform(tmpdir):
    imagetools.hdr_ang2uv("oct21_detail_glz_EW_desk.hdr")
    imagetools.hdr_uv2ang("oct21_detail_glz_EW_desk_uv.hdr")
    uv = io.hdr2array("oct21_detail_glz_EW_desk_uv.hdr")
    ang = io.hdr2array("oct21_detail_glz_EW_desk.hdr")
    uv2ang = imagetools.array_uv2ang(uv)
    ang2uv = imagetools.array_ang2uv(ang)
    vm = imagetools.hdr2vm("oct21_detail_glz_EW_desk.hdr")
    mask = vm.in_view(vm.pixelrays(ang.shape[0]))
    p = (1, 50, 99)
    assert np.allclose(np.percentile(ang[mask], p),
                       np.percentile(uv2ang[mask], p), rtol=.05)
    assert np.allclose(np.percentile(ang2uv, p),
                       np.percentile(uv, p), rtol=.05)
