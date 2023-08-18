# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for reading and writing"""
import re
import shlex
import os
from subprocess import Popen, PIPE

import numpy as np


def get_nproc(nproc=None):
    if nproc is not None:
        return nproc
    env_nproc = os.getenv('RAYTOOLS_PROC_CAP')
    try:
        return int(env_nproc)
    except (ValueError, TypeError):
        return os.cpu_count()


def set_nproc(nproc):
    if nproc is None:
        return None
    if type(nproc) != int:
        raise ValueError('nproc must be an int')
    if nproc < 1:
        unset_nproc()
    else:
        os.environ['RAYTOOLS_PROC_CAP'] = str(nproc)


def unset_nproc():
    try:
        os.environ.pop('RAYTOOLS_PROC_CAP')
    except KeyError:
        pass


def np2bytes(ar, dtype='<f'):
    """format ar as bytestring

    Parameters
    ----------
    ar: np.array
    dtype: str
        argument to pass to np.dtype()

    Returns
    -------
    bytes
    """
    dt = np.dtype(dtype)
    return ar.astype(dt).tobytes()


def np2bytefile(ar, outf, dtype='<f', mode='wb'):
    """save vectors to file

    Parameters
    ----------
    ar: np.array
        array to write
    outf: str
        file to write to
    dtype: str
        argument to pass to np.dtype()
    """
    f = open(outf, mode)
    f.write(np2bytes(ar, dtype=dtype))
    f.close()


def bytes2np(buf, shape, dtype='<f'):
    """read ar from bytestring

    Parameters
    ----------
    buf: bytes, str
    shape: tuple
        array shape
    dtype: str
        argument to pass to np.dtype()

    Returns
    -------
    np.array
    """
    dt = np.dtype(dtype)
    return np.frombuffer(buf, dtype=dt).reshape(*shape)


def bytefile2np(f, shape, dtype='<f'):
    """read binary data from f

    Parameters
    ----------
    f: IOBase
        file object to read array from
    shape: tuple
        array shape
    dtype: str
        argument to pass to np.dtype()

    Returns
    -------
    ar.shape
        necessary for reconstruction
    """
    return bytes2np(f.read(), shape, dtype)


def _array2hdr(ar, imgf, header, pval):
    """write 2d np.array to hdr image format

        Parameters
        ----------
        ar: np.array
            image array
        imgf: str
            file path to write
        header: list
            list of header lines to append to image header
        pval: str
            pvalue command

        Returns
        -------
        imgf
        """
    if imgf is None:
        f = None
    else:
        f = open(imgf, 'wb')
    if header is not None:
        hdr = "' '".join(header)
        getinfo = shlex.split(f"getinfo -a '{hdr}'")
        p = Popen(pval.split(), stdin=PIPE, stdout=PIPE)
        q = Popen(getinfo, stdin=p.stdout, stdout=f)
    else:
        p = Popen(pval.split(), stdin=PIPE, stdout=f)
        q = p
    p.stdin.write(np2bytes(ar))
    p.stdin.flush()
    q.communicate()
    try:
        f.close()
    except AttributeError:
        pass
    return imgf


def array2hdr(ar, imgf, header=None):
    """write 2d np.array (x,y) to hdr image format

    Parameters
    ----------
    ar: np.array
            image array
    imgf: str
        file path to right
    header: list
        list of header lines to append to image header

    Returns
    -------
    imgf
    """
    if len(ar.shape) > 2:
        return carray2hdr(ar, imgf, header)
    pval = f'pvalue -r -b -h -H -df -o -y {ar.shape[-1]} +x {ar.shape[-2]}'
    return _array2hdr(ar.T[::-1], imgf, header, pval)


def carray2hdr(ar, imgf, header=None):
    """write color channel np.array (3, x, y) to hdr image format

    Parameters
    ----------
    ar: np.array
            image array
    imgf: str
        file path to right
    header: list
        list of header lines to append to image header

    Returns
    -------
    imgf
    """
    pval = f'pvalue -r -h -H -df -o -y {ar.shape[-1]} +x {ar.shape[-2]}'
    return _array2hdr(ar.T[::-1], imgf, header, pval)


def hdr2array(imgf, stdin=None, header=False):
    """read np.array from hdr image

    Parameters
    ----------
    imgf: file path of image
    stdin:
        passed to Popen (imgf should be "")

    Returns
    -------
    ar: np.array

    """
    pval = f'pvalue -b -h -df -o {imgf}'
    p = Popen(shlex.split(pval), stdin=stdin, stdout=PIPE)
    shape = p.stdout.readline().strip().split()
    shape = (int(shape[-3]), int(shape[-1]))
    imgd = bytes2np(p.stdout.read(), shape).T[:, ::-1]
    if header:
        return imgd, hdr_header(imgf)
    return imgd


def hdr2carray(imgf, stdin=None, header=False):
    """read np.array from color hdr image

    Parameters
    ----------
    imgf: file path of image
    stdin:
        passed to Popen (imgf should be "")

    Returns
    -------
    ar: np.array
    """
    pval = f'pvalue -n -h -df -o {imgf}'
    p = Popen(shlex.split(pval), stdin=stdin, stdout=PIPE)
    shape = p.stdout.readline().strip().split()
    shape = (3, int(shape[-3]), int(shape[-1]))
    imgd = np.transpose(bytes2np(p.stdout.read(), shape)[:, ::-1], (0, 2, 1))
    if header:
        return imgd, hdr_header(imgf)
    return imgd


def hdr_header(imgf):
    p = Popen(shlex.split(f"getinfo {imgf}"), stdout=PIPE, stderr=PIPE).communicate()
    err = p[1]
    try:
        err = err.decode("utf-8")
    except AttributeError:
        pass
    if "bad header!" in err:
        raise IOError(f"{err} - wrong file type?")
    try:
        header = p[0].decode("utf-8")
    except UnicodeDecodeError:
        raise IOError(f"{err} - wrong file type?")
    if "cannot open" in header:
        raise FileNotFoundError(f"{imgf} not found")
    return [i for i in header.strip().splitlines() if not re.match(r".*#?RADIANCE.*", i)]


def rgb2rad(rgb, vlambda=(0.265, 0.670, 0.065)):
    return np.einsum('...j,j->...', rgb, vlambda)


def rgb2lum(rgb, vlambda=(0.265, 0.670, 0.065)):
    return np.einsum('...j,j->...', rgb, np.array(vlambda) * 179)


def rgbe2lum(rgbe):
    """
    convert from Radiance hdr rgbe 4-byte data format to floating point
    luminance.

    Parameters
    ----------
    rgbe: np.array
        r,g,b,e unsigned integers according to:
        http://radsite.lbl.gov/radiance/refer/filefmts.pdf

    Returns
    -------
    lum: luminance in cd/m^2
    """
    v = np.power(2., rgbe[:, 3] - 128).reshape(-1, 1) / 256
    rgb = np.where(rgbe[:, 0:3] == 0, 0, (rgbe[:, 0:3] + 0.5) * v)
    # luminance = 179 * (0.265*R + 0.670*G + 0.065*B)
    return rgb2lum(rgb)


def load_txt(farray, **kwargs):
    """consistent error handing of np.loadtxt

    Parameters
    ----------
    farray: any
        candidate to load
    kwargs:
        passed to np.loadtxt

    Returns
    -------
    np.array

    Raises
    ------
    ValueError:
        file exists, but is not loadable
    FileNotFoundError:
        farray is str, but file does not exist
    TypeError:
        farray is not str or bytes.

    """
    if isinstance(farray, (str, bytes)):
        if os.path.isfile(farray):
            try:
                return np.loadtxt(farray, **kwargs)
            except (ValueError, AttributeError):
                raise ValueError
        else:
            raise FileNotFoundError(f"{farray}")
    else:
        raise TypeError

