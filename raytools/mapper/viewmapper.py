# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from raytools.mapper.mapper import Mapper
from raytools.mapper.angularmixin import AngularMixin
from raytools import translate


class ViewMapper(AngularMixin, Mapper):
    """translate between world direction vectors and normalized UV space for a
    given view angle. pixel projection yields equiangular projection

    Parameters
    ----------
    dxyz: tuple, optional
        central view direction
    viewangle: float, optional
        if < 180, the horizontal and vertical view angle, if greater, view
        becomes 360,180
    """

    def __init__(self, dxyz=(0.0, 1.0, 0.0), viewangle=360.0, name='view',
                 origin=(0, 0, 0), jitterrate=0.5):
        self._viewangle = viewangle
        if viewangle == 360:
            aspect = 2
            self._viewangle = 180
            sf = (1, 1)
            bbox = np.stack(((0, 0), (2, 1)))
        else:
            aspect = 1
            sf = np.array((self._viewangle/180, self._viewangle/180))
            bbox = np.stack((.5 - sf/2, .5 + sf/2))
        super().__init__(dxyz=dxyz, sf=sf, bbox=bbox, aspect=aspect, name=name,
                         origin=origin, jitterrate=jitterrate)

    @property
    def aspect(self):
        return self._aspect

    @aspect.setter
    def aspect(self, a):
        self.area = 2*np.pi*(1 - np.cos(self.viewangle*np.pi*a/360))
        self._aspect = a
        cl = translate.theta2chord(np.pi/2)/(np.pi/2)
        va = self.viewangle*np.pi/360
        clp = translate.theta2chord(va)/va
        self._chordfactor = cl/clp

    @property
    def dxyz(self):
        """(float, float, float) central view direction"""
        return self._dxyz

    @dxyz.setter
    def dxyz(self, xyz):
        """set view parameters"""
        self._dxyz = translate.norm1(np.asarray(xyz).ravel()[0:3])
        self._rmtx = translate.rmtx_yp(self.dxyz)
        if self.aspect == 2:
            self._ivm = ViewMapper(-self.dxyz, 180)
        else:
            self._ivm = None

    def idx2uv(self, idx, shape, jitter=True):
        """
        Parameters
        ----------
        idx: flattened index
        shape:
            the shape to unravel into
        jitter: bool, optional
            randomly offset coordinates within grid

        Returns
        -------
        uv: np.array
            uv coordinates
        """
        si = np.stack(np.unravel_index(idx, shape))
        if jitter and self.jitterrate > 0:
            rng = ((1 - self.jitterrate)/2, (1 + self.jitterrate)/2)
            offset = np.random.default_rng().uniform(*rng, si.shape).T
        else:
            offset = 0.5
        uv = (si.T + offset)/shape[1]
        return uv

    def transform_to(self, imarray, other, nearest=False):
        """take an image arrray with an assumed projection of self and
        transform it to other

        Parameters
        ----------
        imarray : np.array
            2-d or 3-d array with (features,x,y)
        other : raytools.ViewMapper
            the destination view
        nearest : bool
            disable linear interpolation and use nearest pixel
            sets the method= of scipy.interpolate.RegularGridInterpolator

        Returns
        -------
        np.array
        """
        res = imarray.shape[-1]
        if len(imarray.shape) == 3:
            features = imarray.shape[0]
        else:
            features = 1
        img = np.zeros_like(imarray)
        pxyz = self.pixelrays(res)
        imask = self.in_view(pxyz, indices=False)
        mask = other.in_view(pxyz, indices=False)

        pxyz = pxyz.reshape(-1, 3)
        pxy = other.ray2pixel(pxyz[mask], res, False)
        x = np.arange(res) + .5
        if nearest:
            method = "nearest"
        else:
            method = "linear"
        imarray = imarray.reshape(features, *imarray.shape[-2:])
        oshape = img.shape
        img = img.reshape(imarray.shape)
        for i in range(features):
            instance = RegularGridInterpolator((x, x), imarray[i],
                                               bounds_error=False,
                                               method=method,
                                               fill_value=None)
            img[i].flat[mask] = instance(pxy).T.ravel()
        return img.reshape(oshape)
