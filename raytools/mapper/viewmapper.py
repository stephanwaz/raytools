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

    def solid_xyz2vxy(self, xyz):
        """transform from world xyz to view image space (2d)"""
        # transform into equiangular image space
        pxy = self.xyz2vxy(xyz)
        # view size in radians
        vs = min(self.viewangle, 180) * np.pi / 180
        # center and scale coordinates to radians
        a_xy = (pxy - .5) * vs
        # radius in equiangular
        r_a = np.sqrt(np.sum(np.square(a_xy), axis=-1, keepdims=True))
        # radius in equisolid
        r = 2 * np.arcsin(r_a * np.sqrt(2) / np.pi)
        # equisolid coordinates
        so_xy = a_xy * r / (r_a + 1e-9)
        # scale back to 0,1
        pxy = so_xy / vs + .5
        return pxy

    def solid_vxy2xyz(self, xy, stackorigin=False):
        """transform from view image space (2d) to world xyz. xy is [0,1]"""
        # view size in radians
        vs = min(self.viewangle, 180) * np.pi / 180
        # center and scale coordinates to radians
        so_xy = (np.atleast_2d(xy) - .5) * vs
        # radius in equisolid
        r = np.sqrt(np.sum(np.square(so_xy), axis=-1, keepdims=True))
        # radius in equiangular
        r_a = np.sin(r/2) * np.pi/np.sqrt(2)
        # equiangular coordinates
        a_xy = so_xy * r_a / (r + 1e-9)
        # scale back to -.5,.5
        pxy = a_xy / vs
        # now the standard transform from angular to world
        pxy *= (np.array([self._xsign, 1]) *
                (min(self.viewangle, 180) * self._chordfactor) / 180)
        d = np.sqrt(np.sum(np.square(pxy), -1))
        z = np.cos(np.pi * d)
        nperr = np.seterr(all="ignore")
        d = np.where(d <= 0, np.pi, np.sqrt(1 - z * z) / d)
        np.seterr(**nperr)
        pxy *= d[..., None]
        xyz = np.concatenate((pxy, z[..., None]), -1)
        xyz = self.view2world(xyz.reshape(-1, 3)).reshape(xyz.shape)
        if stackorigin:
            xyz = np.hstack((np.broadcast_to(self.origin, xyz.shape), xyz))
        return xyz

    def solid_pixelrays(self, res, jitter=0.0):
        """world xyz coordinates for pixels in view image space"""
        pxy = self.pixels(res, jitter=jitter)
        return self.solid_pixel2ray(pxy, res)

    def solid_ray2pixel(self, xyz, res, integer=True):
        """world xyz to pixel coordinate"""
        pxy = self.solid_xyz2vxy(xyz) * np.array([[res, res]])
        if integer:
            pxy = np.floor(pxy).astype(int)
        return pxy

    def solid_pixel2ray(self, pxy, res):
        """pixel coordinate to world xyz vector"""
        return self.solid_vxy2xyz(pxy/np.broadcast_to((res, res), pxy.shape))

    @staticmethod
    def _process_viewproj(a):
        a = a.lower()
        if a[0:2] == 'vt':
            a = a[-1]
        a = a[0]
        if a in ('a', 'f'):  # angular fisheye
            return 'vta'
        elif a == 'e':  # equisolid fisheye
            return 'vte'
        else:  # Shirley-Chiu square/rectangle
            return 'vuv'

    def init_img(self, res=512, pt=(0, 0, 0), features=1, viewproj='vta',
                 indices=True,
                 **kwargs):
        """Initialize an image array with vectors and mask

        Parameters
        ----------
        res: int, optional
            image array resolution
        pt: tuple, optional
            view point for image header
        features: int, optional
            when greater than 1 initialize img array with N output channels
        viewproj: str, optional
            output view projection chose from::

                angular: '[Aa]*', 'vta', 'fish*'
                equisolid: '[Ee]*', 'vte'
                shirley chiu: '[Uu]*', 'square'

        Returns
        -------
        img: np.array
            zero array of shape (res*self.aspect, res)
        vecs: np.array
            direction vectors corresponding to each pixel (img.size, 3)
        mask: np.array
            indices of flattened img that are in view
        mask2: np.array None
            if features > 1, use mask 2 fro color images
        header: str
        """
        viewproj = self._process_viewproj(viewproj)
        if viewproj in ['vta', 'vuv']:
            return super().init_img(res, pt, features, viewproj, indices, **kwargs)
        if features > 1:
            img = np.zeros((features, res * self.aspect, res))
        else:
            img = np.zeros((res * self.aspect, res))
        header = self.header(pt, viewproj=viewproj, **kwargs)
        mask = None
        mask2 = None
        pxy = self.pixels(res, jitter=0.0)
        if viewproj == 'vte':
            vecs = self.solid_pixel2ray(pxy, res)
        mask = self.in_view(vecs, indices=indices)
        if features > 1 and indices:
            mask2 = (np.tile(np.arange(features), len(mask[0])),
                     np.repeat(mask[0], features),
                     np.repeat(mask[1], features))
        else:
            mask2 = mask
        return img, vecs, mask, mask2, header