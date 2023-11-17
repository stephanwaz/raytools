# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import warnings

import numpy as np
import functools

from raytools import io
from raytools.evaluate.metricset import MetricSet


class ColorMetricSet(MetricSet):
    """object for calculating metrics based on a view direction, and rays
    consisting on direction, solid angle and luminance information assumes lum
    has multi channel color information

    by encapsulating these calculations within a class, metrics with redundant
    calculations can take advantage of cached results, for example dgp does
    not need to recalculate illuminance when it has been directly requested.
    all metrics can be accessed as properties (and are calculated just in time)
    or the object can be called (no arguments) to return a np.array of all
    metrics defined in "metricset"

    Parameters
    ----------
    vm: raytools.mapper.ViewMapper
        the view direction
    vec: np.array
        (N, 3) directions of all rays in view
    omega: np.array
        (N,) solid angle of all rays in view
    lum: np.array
        (N, M) rgb (or spectral) of all rays in view (multiplied by "scale")
    vlambda: Union[tuple, list, np.array, optional
        (M,) channel weights for luminance conversion of input, length must
        match channel count of lum (lum is forced to this shape).
    metricset: list, optional
        keys of metrics to return, same as property names
    scale: float, optional
        scalefactor for luminance
    threshold: float, optional
        threshold for glaresource/background similar behavior to evalglare '-b'
        paramenter. if greater than 100 used as a fixed luminance threshold.
        otherwise used as a factor times the task luminance (defined by
        'tradius')
    guth: bool, optional
        if True, use Guth for the upper field of view and iwata for the lower
        if False, use Kim
    tradius: float, optional
        radius in degrees for task luminance calculation
    kwargs:
        additional arguments that may be required by additional properties
    """
    

    #: available metrics (and the default return set)
    defaultmetrics = ["illum", "irradiance"]

    allmetrics = MetricSet.allmetrics + ["illum", "irradiance"]

    safe2sum = MetricSet.safe2sum.union(["illum", "irradiance"])

    def __init__(self, vec, omega, lum, vm, metricset=None, scale=179.,
                 threshold=2000., guth=True, tradius=30.0,
                 omega_as_view_area=False, lowlight=False,
                 vlambda=(0.265, 0.670, 0.065), **kwargs):
        rgb = np.asarray(lum).reshape(-1, len(vlambda))
        lum = io.rgb2rad(rgb, vlambda)
        super().__init__(vec, omega, lum, vm, metricset=metricset, scale=scale,
                         threshold=threshold, tradius=tradius, lowlight=lowlight,
                         omega_as_view_area=omega_as_view_area, guth=guth,
                         **kwargs)
        self._rgb = rgb[self.view_mask]

    def __call__(self, metrics=None):
        """
        Returns
        -------
        result: np.array
            list of computed metrics

        """
        if metrics is None:
            metrics = self.defaultmetrics
        else:
            self.check_metrics(metrics, True)
        out = []
        if self._warn:
            for m in metrics:
                r = getattr(self, m)
                if hasattr(r, '__len__'):
                    out += list(r)
                else:
                    out.append(r)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for m in metrics:
                    r = getattr(self, m)
                    if hasattr(r, '__len__'):
                        out += list(r)
                    else:
                        out.append(r)
        return np.array(out)

    # -------------------metric dependencies (return array)--------------------

    @property
    def rgb(self):
        return self._rgb

    @property
    @functools.lru_cache(1)
    def sources(self):
        """vec, omega, lum of rays above threshold"""
        m = self.src_mask
        vec = self.vec[m]
        lum = self.lum[m]
        oga = self.omega[m]
        rgb = self.rgb[m]
        return vec, oga, lum, rgb

    @property
    @functools.lru_cache(1)
    def background(self):
        """vec, omega, lum of rays below threshold"""
        m = np.logical_not(self.src_mask)
        vec = self.vec[m]
        lum = self.lum[m]
        oga = self.omega[m]
        rgb = self.rgb[m]
        return vec, oga, lum, rgb

    # ----------------overrides just to handle 4th return value-----------------

    @property
    @functools.lru_cache(1)
    def pwsl2(self):
        """position weighted source luminance squared, used by dgp, ugr, etc
        sum(Ls^2*omega/Ps^2)"""
        _, soga, slum, _ = self.sources
        return np.sum(np.square(slum) * soga * self.scale ** 2 /
                      np.square(self.source_pos_idx))

    @property
    @functools.lru_cache(1)
    def srcillum(self):
        """source illuminance"""
        svec, soga, slum, _ = self.sources
        return np.einsum('i,i,i->', self.vm.ctheta(svec), slum,
                         soga) * self.scale

    @property
    @functools.lru_cache(1)
    def srcarea(self):
        """total source area"""
        _, soga, _, _ = self.sources
        return np.sum(soga)

    @property
    @functools.lru_cache(1)
    def backlum_true(self):
        """average background luminance mathematical"""
        bvec, boga, blum, _ = self.background
        return np.einsum('i,i->', blum, boga)*self.scale/np.sum(boga)

    @property
    @functools.lru_cache(1)
    def irradiance(self):
        """irradiance for each channel"""
        return np.einsum('i,ij,i->j', self.ctheta, self.rgb,
                         self.omega) * self.scale