#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytools.utility"""
import pytest
import re
from raytools import utility
import numpy as np


def do_stuff(a, b):
    return np.sum(a * b)


def test_poolcall(capsys):
    a = np.random.default_rng().random((10, 1, 10))
    b = np.arange(10)
    c1 = utility.pool_call(do_stuff, a, b, pbar=False, workers='t')
    c = utility.pool_call(do_stuff, a, b, desc="by10")
    captured = capsys.readouterr().err
    m = re.match(r"([\w\d]+):.* (\d+)/(\d+)", captured.strip())
    assert m.group(1) == "by10"
    assert m.group(2) == '10'
    a = a.reshape(1,10,10)
    d = utility.pool_call(do_stuff, a, b, expandarg=False)
    captured = capsys.readouterr().err
    m = re.match(r"([\w\d]+):.* (\d+)/(\d+)", captured.strip())
    assert m.group(1) == "processing"
    assert m.group(2) == '1'
    e = utility.pool_call(do_stuff, a, b, expandarg=False, workers=False)
    assert len(capsys.readouterr().err) == 0
    assert np.isclose(np.sum(c), d[0])



