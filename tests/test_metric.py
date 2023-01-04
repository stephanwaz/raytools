#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytools.evaluate.metric"""
import os
import re
import shutil

import pytest
from raytools import io, imagetools
from raytools.mapper import ViewMapper
from raytools.evaluate import PositionIndex, MetricSet, BaseMetricSet
from raytools import translate
import numpy as np


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


def test_get_pos_idx(tmpdir):
    exp = [[16.0, 16.0, 16.0, 8.076347019578511, 6.856663497343681, 8.968471163183743, 16.0, 16.0, 16.0, 16.0],
           [16.0, 9.477810403835639, 5.965841356086234, 4.2579984530618225, 3.548712152029083, 4.522287557454674, 8.626821872361402, 16.0, 16.0, 16.0],
           [16.0, 6.087587840512674, 3.743634068668167, 2.5905982171679907, 2.1066586423000793, 2.6158672551048934, 4.88801223548715, 11.250158366243358, 16.0, 16.0],
           [8.242109188672083, 4.5378327204826645, 2.729101697082527, 1.8268060895950733, 1.4364596439443964, 1.7403135007849175, 3.28638637322815, 7.818635653939338, 16.0, 16.0],
           [7.163371648976106, 3.918286200607336, 2.3229020762248958, 1.5151625926531762, 1.1368003104524078, 1.3622203177528458, 2.8170336179218136, 6.980479737328229, 16.0, 16.0],
           [7.163371648976106, 3.918286200607336, 2.3229020762248958, 1.5151625926531764, 1.1368003104524078, 1.3622203177528456, 2.817033617921815, 6.980479737328229, 16.0, 16.0],
           [8.242109188672083, 4.5378327204826645, 2.729101697082527, 1.8268060895950733, 1.4364596439443964, 1.7403135007849182, 3.286386373228152, 7.818635653939338, 16.0, 16.0],
           [16.0, 6.087587840512674, 3.743634068668167, 2.5905982171679907, 2.1066586423000793, 2.615867255104895, 4.888012235487156, 11.250158366243358, 16.0, 16.0],
           [16.0, 9.477810403835639, 5.965841356086234, 4.2579984530618225, 3.548712152029083, 4.522287557454676, 8.626821872361402, 16.0, 16.0, 16.0],
           [16.0, 16.0, 16.0, 8.076347019578508, 6.8566634973436775, 8.968471163183748, 16.0, 16.0, 16.0, 16.0]]
    vm = ViewMapper(viewangle=180)
    res = 10
    img = vm.pixelrays(res)
    fimg = img.reshape(-1, 3)
    posfinder = PositionIndex()
    posidx = posfinder.positions(vm, fimg).reshape(res, res)
    assert np.allclose(posidx, exp)

def test_guth(tmpdir):
    angs = np.linspace(0, 90, 19)
    angr = angs * np.pi / 180
    vecs = np.stack((np.cos(angr), np.zeros(19), np.sin(angr))).T
    viewvec = translate.rotate_elem(((1, 0, 0),), 45)[0]
    srcvecs = translate.rotate_elem(vecs, np.pi/4, degrees=False)
    ang2 = translate.degrees(viewvec, srcvecs)
    ang2r = translate.radians(viewvec, srcvecs)
    assert np.allclose(angs, ang2)
    assert np.allclose(angr, ang2r)
    pos = PositionIndex()
    posi = pos.positions_vec(viewvec, srcvecs)
    x = np.minimum(angs, 55)/55
    b = 2843.58*np.exp(x + 1.5*np.square(x))/179
    assert np.all(np.logical_and(b/posi > 12, b/posi < 16))


def test_position(tmpdir):
    vm = ViewMapper((.5, .5, -1), viewangle=180)
    res = 1000
    img = vm.pixelrays(res)
    fimg = img.reshape(-1, 3)
    cos = vm.ctheta(fimg)
    pc = cos.reshape(res, res)
    io.array2hdr(pc, "position_cos.hdr")
    posfinder = PositionIndex()
    posidx = posfinder.positions(vm, fimg).reshape(res, res)
    pg = 1/np.square(posidx).reshape(res, res)
    io.array2hdr(pg, "position_guth.hdr")
    posfinder = PositionIndex(guth=False)
    posidx = posfinder.positions(vm, fimg).reshape(res, res)
    pk = 1/posidx.reshape(res, res)
    io.array2hdr(pk, "position_kim.hdr")
    position_kim = io.hdr2array("position_kim.hdr")
    position_guth = io.hdr2array("position_guth.hdr")
    position_cos = io.hdr2array("position_cos.hdr")
    assert np.allclose(position_kim, pk, atol=4.8, rtol=.1)
    assert np.allclose(position_guth, pg, atol=.5, rtol=.1)
    assert np.allclose(position_cos, pc, atol=.6, rtol=.1)


def test_basemetric(tmpdir, capsys):
    with capsys.disabled():
        vm = ViewMapper()
        uv = vm.idx2uv(np.arange(128), (16, 8), False)
        v = vm.uv2xyz(uv)
        o = np.full(len(v), 2*np.pi/64)
        l = np.repeat([.1, 1, 20, 1], 32)
        ms = BaseMetricSet(v, o, l, vm, metricset=["avglum", "loggcr"])
        ref = [9.88975000e+02,  5.17512300e-01]
        assert np.allclose(ref, ms())
        vm = ViewMapper(viewangle=153)
        v = v[0:64]
        o = o[0:64]
        l = l[0:64]
        l[0] = 10000
        ms = BaseMetricSet(v, o, l, vm, omega_as_view_area=False)
        ref = [2.13098453e+04, 2.41039550e+04, 1.86901131e+00, 7.39624535e+01,
               5.23313325e+02, 2.71876179e+00, 1.03284540e+01, 4.84741676e+04,
               3.31925135e+03, 1.79000000e+06]
        assert np.allclose(ms(ms.allmetrics), ref)
    vm = ViewMapper(viewangle=180)
    uv = vm.idx2uv(np.arange(240), (16, 16), False)
    v = vm.uv2xyz(uv)
    o = np.full(len(v), 2*np.pi/256)
    l = np.repeat([.1, 1, 20, 1], 60)
    ms = BaseMetricSet(v, o, l, vm, warn=True, omega_as_view_area=False)
    captured = capsys.readouterr()
    assert re.match(r"Warning, large discrepancy between sum\(omega\) and view area: .*", captured.err)


def test_allmetrics(tmpdir):
    img = "oct21_detail_glz_EW_desk.hdr"
    vm = imagetools.hdr2vm(img)
    v, o, l = imagetools.hdr2vol("oct21_detail_glz_EW_desk.hdr")
    ms = MetricSet(v, o, l, vm, metricset=["illum", "avglum", "loggcr"])
    default = ms()
    refd = [2.81437911e+04, 6.48205594e+03, 4.66149959e+00]
    assert np.allclose(default, refd)
    allmetric = ms(ms.allmetrics)
    ref = [2.81437911e+04, 6.48205594e+03, 4.66149959e+00, 4.58669216e+04,
           5.72520326e+04, 4.75779091e+00, 1.25009205e+05, 5.88265640e+03,
           4.44421626e+03, 5.19627647e+08, 9.63467627e-01, 1.00000000e+00,
           1.20468250e+03, 3.81700814e+02, 1.65204054e+00, 3.66047908e+00,
           3.36031979e-01, 7.03755723e+01, 2.00000000e+03, 9.56598328e+11,
           6.28318531e+00, 4.27772913e+02, 2.69446426e+04, 6.05973883e-01,
           5.19627647e+08]
    assert np.allclose(allmetric, ref)

