# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytools.mapper.viewmapper import ViewMapper


class SolidViewMapper(ViewMapper):
    """translate between world direction vectors and normalized UV space for a
    given view angle. pixel projection yields equisolid projection

    NOTE: view angle is given in equiangular basis, so will not match when
    less than 180.

    Parameters
    ----------
    dxyz: tuple, optional
        central view direction
    viewangle: float, optional
        view angle limited to 180
    """

    def __init__(self, dxyz=(0.0, 1.0, 0.0), viewangle=180.0, name='solidview',
                 origin=(0, 0, 0), jitterrate=0.5):
        if viewangle > 180:
            raise ValueError("View angle for SolidViewMapper cannot exceed 180")
        super().__init__(dxyz=dxyz, viewangle=viewangle, name=name,
                         origin=origin, jitterrate=jitterrate)

    def xyz2vxy(self, xyz):
        """transform from world xyz to view image space (2d)"""
        # transform into equiangular image space
        pxy = super().xyz2vxy(xyz)
        # view size in radians
        vs = self.viewangle * np.pi / 180
        # center and scale coordinates to radians
        a_xy = (pxy - .5) * vs
        # radius in equiangular
        r_a = np.sqrt(np.sum(np.square(a_xy), axis=-1, keepdims=True))
        # radius in equisolid
        r = 2 * np.arcsin(r_a * np.sqrt(2) / np.pi)
        # equisolid coordinates
        so_xy = a_xy * r / r_a
        # scale back to 0,1
        pxy = so_xy / vs + .5
        return pxy

    def vxy2xyz(self, xy, stackorigin=False):
        """transform from view image space (2d) to world xyz. xy is [0,1]"""
        # view size in radians
        vs = self.viewangle * np.pi / 180
        # center and scale coordinates to radians
        so_xy = (np.atleast_2d(xy) - .5) * vs
        # radius in equisolid
        r = np.sqrt(np.sum(np.square(so_xy), axis=-1, keepdims=True))
        # radius in equiangular
        r_a = np.sin(r/2) * np.pi/np.sqrt(2)
        # equiangular coordinates
        a_xy = so_xy * r_a / r
        # scale back to -.5,.5
        pxy = a_xy / vs
        # now the standard transform from angular to world
        pxy *= (np.array([self._xsign, 1]) *
                (self.viewangle * self._chordfactor) / 180)
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

