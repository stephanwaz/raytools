# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for translating from mappers to hdr"""
import numpy as np
import clasp.script_tools as cst


from raytools import translate, io
from raytools.evaluate import MetricSet, ColorMetricSet, retina
from raytools.mapper import ViewMapper, SolidViewMapper
from scipy.interpolate import RegularGridInterpolator


def array_uv2ang(imarray):
    resx = imarray.shape[-2]
    resy = imarray.shape[-1]
    if resx > resy:
        vm = ViewMapper(viewangle=360)
    else:
        vm = ViewMapper(viewangle=180)
    img, pixelxyz, mask, _, _ = vm.init_img(resy, features=len(imarray.shape), indices=False)
    uv = vm.xyz2uv(pixelxyz.reshape(-1, 3))
    ij = translate.uv2ij(uv[mask], resy)
    if len(imarray.shape) == 3:
        img = np.zeros((3, resx*resy))
        img[:, mask] = imarray[:, ij[:, 0], ij[:, 1]]
        return img.reshape(3, resx, resy)
    else:
        img = np.zeros(resx*resy)
        img[mask] = imarray[ij[:, 0], ij[:, 1]]
        return img.reshape(resx, resy)


def hdr_uv2ang(imgf, outf=None, stdout=False, **kwargs):
    if outf is None and not stdout:
        outf = imgf.rsplit(".", 1)[0] + "_ang.hdr"
    imarray, header = io.hdr2carray(imgf, header=True)
    img = array_uv2ang(imarray)
    io.carray2hdr(img, outf, header=header, clean=True)
    return outf


def array_ang2uv(imarray, vm=None):
    if vm is None:
        vm = ViewMapper(viewangle=180)
    res = imarray.shape[-1]
    uv = translate.bin2uv(np.arange(res*res), res)
    xyz = vm.uv2xyz(uv)
    pxy = vm.ray2pixel(xyz, imarray.shape[-1])
    if len(imarray.shape) == 3:
        return imarray[:, pxy[:, 0], pxy[:, 1]].reshape(3, res, res)
    else:
        return imarray[pxy[:, 0], pxy[:, 1]].reshape(res, res)


def hdr_ang2uv(imgf, useview=True, outf=None, stdout=False, **kwargs):
    if outf is None and not stdout:
        outf = imgf.rsplit(".", 1)[0] + "_uv.hdr"
    vm = None
    if useview:
        vm = hdr2vm(imgf)
    if vm is None:
        vm = ViewMapper(viewangle=180)
    imarray, header = io.hdr2carray(imgf, header=True)
    img = array_ang2uv(imarray, vm)
    io.carray2hdr(img, outf, header=header, clean=True)
    return outf


def array_rotate(imarray, ang, center=None, rotate_first=True, fill_value=None, nearest=False):
    """rotate and center a angular fisheye image array

    Parameters
    ----------
    imarray : np.array
        ([optional 3 color], width, height)
    ang : float
        rotation angle in degrees
    center : tuple, optional
        new pixel center (in orginal coordinates)
    rotate_first : bool, optional
        order of rotate/center. true is rotate first.
    fill_value : optional
        passed to scipy.RegularGridInterpolator

    Returns
    -------

    corrected np.array

    """
    res = imarray.shape[-1]
    vm = ViewMapper(viewangle=180)
    features = 1
    if len(imarray.shape) == 3:
        features = 3
    img, pxyz, mask, mask2, _ = vm.init_img(res, features=features,
                                            indices=False)

    pxyz = pxyz.reshape(-1, 3)
    if rotate_first and ang != 0:
        pxyz = translate.rotate_elem(pxyz, ang, 1)
    if center is not None:
        if len(center) == 4:
            targets = vm.pixel2ray(np.reshape(center, (2, 2)), res)
            m1 = translate.rmtx_yp(targets[1])
            m2 = translate.rmtx_yp(targets[0])
            m3 = m2[0].T @ m2[1].T @ m1[1] @ m1[0]
            pxyz = (m3 @ pxyz.T).T
        else:
            cxyz = vm.pixel2ray(np.atleast_2d(center)[:, 0:2], res)
            vm2 = ViewMapper(dxyz=cxyz[0], viewangle=180)
            pxyz = vm2.view2world(vm.world2view(pxyz))

    pxy = vm.ray2pixel(pxyz, res, integer=False)
    x = np.arange(res)
    if nearest:
        method = "nearest"
    else:
        method = "linear"
    if len(imarray.shape) == 3:
        for i in range(3):
            instance = RegularGridInterpolator((x, x), imarray[i],
                                               bounds_error=False,
                                               method=method, fill_value=fill_value)
            img[i].flat = instance(pxy).T.ravel()
    else:
        instance = RegularGridInterpolator((x, x), imarray, bounds_error=False,
                                           method=method, fill_value=fill_value)
        img.flat = instance(pxy)

    if (not rotate_first) and ang != 0:
        img = array_rotate(img, ang, center=None, rotate_first=True, fill_value=fill_value, nearest=nearest)
    return img


def array_solid2ang(imarray, nearest=False, reverse=False, viewangle=180,
                    returnvm=True):
    if reverse:
        vmi = ViewMapper(viewangle=viewangle)
        vmo = SolidViewMapper(viewangle=viewangle, name="vts")
    else:
        vmi = SolidViewMapper(viewangle=viewangle)
        vmo = ViewMapper(viewangle=viewangle, name="vta")
    if returnvm:
        return vmi.transform_to(imarray, vmo, nearest=nearest), vmo
    return vmi.transform_to(imarray, vmo, nearest=nearest)


def hdr_solid2ang(imgf, outf=None, nearest=False, stdout=False, reverse=False,
                  viewangle=None, **kwargs):
    imarray, header = io.hdr2carray(imgf, header=True)
    if viewangle is None:
        vm = hdr2vm(imgf)
        if vm is None:
            viewangle = 180
        else:
            viewangle = vm.viewangle

    img, vmo = array_solid2ang(imarray, nearest, reverse, viewangle)

    if outf is None and not stdout:
        outf = imgf.rsplit(".", 1)[0] + f"_2{vmo.name}.hdr"

    header.append(vmo.header())
    io.carray2hdr(img, outf, header, clean=True)
    return outf


def hdr_rotate(imgf, outf=None, rotate=0.0, center=None, rotate_first=True,
               nearest=False, stdout=False, **kwargs):
    cl = ""
    rl = ""
    if rotate_first and rotate != 0:
        cl += f"_r{int(rotate):02d}"
    if center is not None:
        cl += f"_{center[0]}-{center[1]}"
    if (not rotate_first) and rotate != 0:
        cl += f"_r{int(rotate):02d}"
    if outf is None and not stdout:
        outf = imgf.rsplit(".", 1)[0] + f"{cl}.hdr"
    imarray, header = io.hdr2carray(imgf, header=True)
    header.append(f"hdr_rotate rotate:{rotate} center:{center}, rotate_first:{rotate_first}, nearest:{nearest}")
    img = array_rotate(imarray, rotate, center, rotate_first=rotate_first,
                       nearest=nearest)
    io.carray2hdr(img, outf, header=header, clean=True)
    return outf


def hdr2vol(imgf, vm=None, color=False, vlambda=None):
    if color:
        ar = io.hdr2carray(imgf)
    else:
        ar = io.hdr2array(imgf, vlambda=vlambda)
    if vm is None:
        vm = hdr2vm(imgf)
    vecs = vm.pixelrays(ar.shape[-1]).reshape(-1, 3)
    oga = vm.pixel2omega(vm.pixels(ar.shape[-1]), ar.shape[-1]).ravel()
    return vecs, oga, np.squeeze(ar.reshape(-1, ar.shape[-1]*ar.shape[-2])).T


def vf_to_vm(view):
    """view file to ViewMapper"""
    vl = [i for i in open(view).readlines() if "-vta" in i]
    if len(vl) == 0:
        raise ValueError(f"no valid -vta view in file {view}")
    vp = vl[-1].split()
    view_angle = float(vp[vp.index("-vh") + 1])
    vd = vp.index("-vd")
    view_dir = [float(vp[i]) for i in range(vd + 1, vd + 4)]
    return ViewMapper(view_dir, view_angle)


def img_size(imgf):
    hd = cst.pipeline([f"getinfo -d {imgf}"]).strip().split()
    x = 1
    y = 1
    for i in range(2, len(hd)):
        if 'X' in hd[i - 1]:
            x = int(hd[i])
        elif 'Y' in hd[i - 1]:
            y = int(hd[i])
    return x, y


def hdr2vm(imgf, vpt=False):
    """hdr to ViewMapper"""
    header, err = cst.pipeline([f"getinfo {imgf}"], caperr=True)
    try:
        err = err.decode("utf-8")
    except AttributeError:
        pass
    if "bad header!" in err:
        raise IOError(f"{err} - wrong file type?")
    elif "cannot open" in header:
        raise FileNotFoundError(f"{imgf} not found")
    vp = None
    if "VIEW= -vta" in header:
        vp = header.rsplit("VIEW= -vta", 1)[-1].splitlines()[0].split()
    elif "rvu -vta" in header:
        vp = header.rsplit("rvu -vta", 1)[-1].splitlines()[0].split()
    if vp is not None:
        view_angle = float(vp[vp.index("-vh") + 1])
        try:
            vd = vp.index("-vd")
        except ValueError:
            view_dir = [0.0, 1.0, 0.0]
        else:
            view_dir = [float(vp[i]) for i in range(vd + 1, vd + 4)]
        try:
            vpi = vp.index("-vp")
        except ValueError:
            view_pt = [0.0, 0.0, 0.0]
        else:
            view_pt = [float(vp[i]) for i in range(vpi + 1, vpi + 4)]
        x, y = img_size(imgf)
        vm = ViewMapper(view_dir, view_angle * x / y)
    else:
        view_pt = None
        vm = None
    if vpt:
        return vm, view_pt
    else:
        return vm


def gather_strays(pvol, peakt, cosrad):
    pc = np.nonzero(pvol[:, 4] > peakt)[0]
    # only do something if some pixels above peakt
    if pc.size > 3:
        # first sort descending by luminance
        pc = pc[np.argsort(-pvol[pc, 4])]
        pvols = pvol[pc]
        # calculate angular distance from peak ray and filter strays
        pd = np.einsum("i,ji->j", pvols[0, 0:3], pvols[:, 0:3])
        dm = pd > cosrad
        strays = [(np.average(pvols[dm, 0:3], axis=0), np.sum(pvols[dm, 3]), np.average(pvols[dm, 4:], axis=0, weights=pvols[dm, 3]))]
        return strays + gather_strays(pvols[np.logical_not(dm)], peakt, cosrad)
    else:
        return []


def find_peak(v, o, l, scale=179, peaka=6.7967e-05, peakt=1e5, peakr=4,
              blurtol=0.75, peakc=1.0, peakrad=4.0, findsecondary=False,
              blursun=False, vlambda=(0.265, 0.670, 0.065)):
    if len(l.shape) > 1:
        l = np.hstack((io.rgb2rad(l, vlambda)[:, None], l))
    else:
        l = l.reshape(-1, 1)
    pc = np.nonzero(l[:, 0] > peakt / scale)[0]
    # only do something if some pixels above peakt
    if pc.size == 0:
        return None, None, pc
    # first sort descending by luminance
    pc = pc[np.argsort(-l[pc, 0])]
    pvol = np.hstack((v[pc], o[pc, None], l[pc]))
    # establish maximum radius for grouping
    cosrad = np.cos((peaka / np.pi) ** .5 * peakrad)
    # calculate angular distance from peak ray and filter strays
    pd = np.einsum("i,ji->j", pvol[0, 0:3], pvol[:, 0:3])
    dm = pd > cosrad
    pc = pc[dm]

    if findsecondary:
        tol2 = np.cos((peaka / np.pi) ** .5 * peakrad * 8)
        strays = gather_strays(pvol[pd < tol2], peakt * 4 / scale, tol2)
        if len(strays) > 0 and len(strays[0][-1]) > 1:
            strays = [(s[0], s[1], s[2][1:]) for s in strays]
    else:
        strays = None

    pvol = pvol[dm]
    # this handles image filtering/blurring
    nearpeak = pvol[0, 4] * blurtol <= pvol[:, 4]
    # calculate expected energy assuming full visibility:
    esun = np.average(pvol[nearpeak, 4]) * peaka * peakc
    # sum up to peak energy
    cume = np.cumsum(pvol[:, 3:4] * pvol[:, 4:], axis=0)
    # when there is enough energy, treat as full sun
    if cume[-1, 0] > esun:
        stop = np.argmax(cume[:, 0] > esun)
        if stop == 0:
            stop = len(cume)
        peakl = cume[stop - 1] / peaka
    # otherwise treat as partial sun (needs to use peak ratio)
    else:
        stop = np.argmax(pvol[:, 4] < pvol[0, 4] / peakr)
        if stop == 0:
            stop = len(cume)
        if peakc > 1:
            peakl = cume[stop - 1] / peaka
        else:
            peakl = pvol[0, 4:]
            peaka = cume[stop - 1, 0] / peakl[0]
    pc = pc[:stop]
    pvol = pvol[:stop]
    # new source vector weight by L*omega of source rarys
    pv = translate.norm(np.average(pvol[:, 0:3], axis=0, weights=pvol[:, 3] *
                                                                 pvol[:, 4]))
    if blursun:
        cf = np.atleast_1d(retina.blur_sun(peaka, peakl[0]))[0]
    else:
        cf = 1
    if len(peakl) > 1:
        pvol = (pv.ravel(), peaka * cf, peakl[1:] / cf)
    else:
        pvol = (pv.ravel(), peaka * cf, peakl / cf)
    return pvol, strays, pc


def normalize_peak(v, o, l, scale=179, peaka=6.7967e-05, peakt=1e5, peakr=4,
                   blursun=False, blurtol=0.75, returnall=True, peakc=1.0,
                   peakrad=4.0, returnparts=False, keepzeros=False,
                   findsecondary=False, vlambda=(0.265, 0.670, 0.065)):
    """consolidate the brightest pixels represented by v, o, l up into a single
    source, correcting the area while maintaining equal energy

    Parameters
    ----------
    v: np.array
        shape (N, 3), direction vectors of pixels (x, y, z) normalized
    o: np.array
        shape (N,), solid angles of pixels (steradians)
    l: np.array
        shape (N,), luminance of pixels
    scale: Union[float, int], optional
        scale factor for l to convert to cd/m^2, default assumes radiance units
    peaka: float, optional
        area to aggregate to
    peakt: Union[float, int], optional
        lower threshold for possible bright pixels
    peakr: Union[float, int], optional
        ratio, from peak pixel value to lowest value to include when aggregating
        partially visible sun.
    peakc: float, optional
        correct peak value for expected energy (use with photos)
    peakrad: float, optional
        distance tolerance (as factor of radius) for peak pixel collection
    blursun: bool, optional
        whether to correct area and luminance according to human PSF
    blurtol: float, optional
        when checking for sun visibility this enables an averaging of near peak
        values whick could be an artifact from pfilt. set to 1 to disable.
    returnall: bool, optional
        if true, return complete v, o, l. if false, only return peak
    returnparts: bool, optional
        supercedes return all, return pvol, v, o, l
    keepzeros: bool, optional
        zero at lums of source rays, but keep in return value

    Returns
    -------
    pvol: tuple
    v: np.array
        shape (N, 3), direction vectors of pixels (x, y, z) normalized
    o: np.array
        shape (N,), solid angles of pixels (steradians)
    l: np.array
        shape (N,), luminance of pixels

    """
    pvol, strays, pc = find_peak(v, o, l, scale=scale, peaka=peaka, peakt=peakt,
                                 peakr=peakr, blurtol=blurtol, peakc=peakc,
                                 peakrad=peakrad, findsecondary=findsecondary,
                                 blursun=blursun, vlambda=vlambda)
    # only do something if some pixels above peakt
    if pc.size > 0:
        lum = np.atleast_2d(np.copy(l.T)).T
        omega = np.copy(o)
        if keepzeros:
            lum[pc] = 0
            omega[pc] = 0
            vol = np.hstack((v, omega[:, None], lum))
        else:
            vol = np.hstack((v, omega[:, None], lum))
            # filter out source rays
            vol = np.delete(vol, pc, axis=0)
        if returnparts:
            v = vol[:, 0:3]
            omega = vol[:, 3]
            lum = np.squeeze(vol[:, 4:])
        else:
            # then add new consolidated ray back to output v, o, l
            v = np.vstack((vol[:, 0:3], [pvol[0]]))
            omega = np.concatenate((vol[:, 3], [pvol[1]]))
            lum = np.squeeze(np.vstack((vol[:, 4:], np.atleast_2d(pvol[2].T))))
    else:
        if len(l.shape) > 1:
            pvol = np.zeros(4 + l.shape[-1])
        else:
            pvol = np.zeros(5)
        lum = l
        omega = o
    if findsecondary:
        return pvol, strays
    elif returnparts:
        return pvol, v, omega, lum
    elif returnall:
        return v, omega, lum
    else:
        return pvol


def imgmetric(imgf, metrics, peakn=False, scale=179, lumrgb=None, threshold=2000., lowlight=False,
              **peakwargs):
    vm = hdr2vm(imgf)
    if vm is None:
        vm = ViewMapper(viewangle=180)
    needscolor = False
    try:
        MetricSet.check_metrics(metrics, True)
    except AttributeError:
        needscolor = True
    if lumrgb is None:
        lumrgb = io.hdr_header(imgf, items=['luminancergb'])[-1]
        try:
            lumrgb = [float(i) for i in lumrgb.split()]
            if len(lumrgb) != 3:
                raise IndexError
        except IndexError:
            lumrgb = None
        else:
            if not needscolor and np.allclose((0.26507413, 0.67011463, 0.06481124), lumrgb, atol=.001):
                lumrgb = None
    v, o, l = hdr2vol(imgf, vm, needscolor, vlambda=lumrgb)
    if peakn:
        v, o, l = normalize_peak(v, o, l, scale, **peakwargs)
    if needscolor:
        return ColorMetricSet(v, o, l, vm, metrics, scale=scale,
                              threshold=threshold, lowlight=lowlight,
                              vlambda=lumrgb)()
    return MetricSet(v, o, l, vm, metrics, scale=scale, threshold=threshold,
                     lowlight=lowlight)()
