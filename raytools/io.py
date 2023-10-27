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
import sys
import textwrap
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


def _array2hdr(ar, imgf, header, pval, clean=False):
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
        if clean:
            header = clean_header(header)
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


def array2hdr(ar, imgf, header=None, clean=False):
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
        return carray2hdr(ar, imgf, header, clean=clean)
    pval = f'pvalue -r -b -h -H -df -o -y {ar.shape[-1]} +x {ar.shape[-2]}'
    return _array2hdr(ar.T[::-1], imgf, header, pval, clean=clean)


def carray2hdr(ar, imgf, header=None, clean=False):
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
    return _array2hdr(ar.T[::-1], imgf, header, pval, clean=clean)


def _hdr_in(pval, imgf, stdin):
    p = Popen(shlex.split(pval), stdin=stdin, stdout=PIPE)
    shape = p.stdout.readline().strip().split()
    try:
        shape = (int(shape[-3]), int(shape[-1]))
    except IndexError:
        if imgf == "":
            imgf = "-"
        if os.path.isfile(imgf):
            raise ValueError(f"Bad HDR file '{imgf}'")
        else:
            raise ValueError(f"HDR image file '{imgf}' not found")
    return p.stdout.read(), shape


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
    imgd = bytes2np(*_hdr_in(pval, imgf, stdin)).T[:, ::-1]
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
    data, shape = _hdr_in(pval, imgf, stdin)
    shape = (3, *shape)
    imgd = np.transpose(bytes2np(data, shape)[:, ::-1], (0, 2, 1))
    if header:
        return imgd, hdr_header(imgf)
    return imgd


def header_items(header, items):
    items = [i.lower() for i in items]
    out = [""] * len(items)
    for line in header:
        cl = line.strip()
        if re.match(r".+:$", cl):
            continue
        sep = None
        if "=" in cl:
            sep = "="
        key, val = cl.split(sep, 1)
        key = key.lower()
        if key in items:
            out[items.index(key)] = val.strip()
    return out


def clean_header(header):
    """remove redundant entries from radiance image header, updating view,
    purging pvalue and clasp_tmp file names"""
    return CleanHeader(header).header



def hdr_header(imgf, clean=False, items=None):
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
    header = [i for i in header.strip().splitlines() if not re.match(r".*#?RADIANCE.*", i)]
    if items is not None:
        header = header_items(header, items=items)
    elif clean:
        header = clean_header(header)
    return header


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


class CleanHeader(object):

    def __init__(self, text, spacespertab=4, outtab="\t"):
        self.spacespertab = spacespertab
        self.outtab = outtab
        self._data = None
        self._header = None
        self.text = text

    @property
    def data(self):
        return self._data

    @property
    def header(self):
        return self._header

    @property
    def text(self):
        return self._text

    @text.setter
    def text(self, t):
        if type(t) != str:
            t = "\n".join(t)
        self._text = t
        self._data, view = self.preprocess()
        header = self.make_header(self._data)
        if view is not None:
            header.append(f"VIEW={view}")
        self._header = header

    def preprocess(self, text=None):
        if text is not None:
            self._text = text

        files = []
        view = None  # only store last found view
        vals = []  # use to check for redundant lines
        cs = []  # the current stack of file nesting
        for i, h in enumerate(self._text.splitlines()):
            cl = h.strip()
            cl = re.sub(r"\S*clasp_tmp\S*", "<stdin>", cl)

            inset = re.match(r"\s*", h).group()
            inset = inset.replace(" " * self.spacespertab,
                                  "\t").replace(" ", "")
            depth = len(inset)
            if re.match(r"(.+):$", cl):
                curf = cl.rsplit("/", 1)[-1][:-1]
                fi = dict(name=curf, depth=depth, content=[])
                if len(cs) == 0:
                    cs.append(fi)
                    files.append(fi)
                elif depth > cs[-1]['depth']:
                    cs[-1]['content'].append(fi)
                    cs.append(fi)
                else:
                    le = depth - cs[-1]['depth']
                    cs = cs[:-(le+1)]
                    if len(cs) == 0:
                        cs.append(fi)
                        files.append(fi)
                    else:
                        cs[-1]['content'].append(fi)
                        cs.append(fi)
                continue

            sep = None
            if "=" in cl:
                sep = "="
            try:
                key, val = cl.split(sep, 1)
            except ValueError:
                key = cl
                val = ''
            if key == "pvalue":
                continue
            if sep is None:
                sep = " "
            if key == "VIEW":
                view = val
                continue
            if key == "...":
                cs[-1]['content'][-1] += " " + val
            elif cl not in vals:
                vals.append(f"{key}{sep}{val}")
                if depth <= cs[-1]['depth']:
                    le = cs[-1]['depth'] - depth
                    cs = cs[:-(le + 1)]
                cs[-1]['content'].append(vals[-1])
        return files, view

    def _clean_up(self, data, depth=0, da=0):
        outdata = []
        for v in data:
            if type(v) == str:
                indent = self.outtab * (depth + 1 - da)
                outdata.append(f"{indent}{v}")
            elif len(v['content']) == 0:
                continue
            elif len(v['content']) == 1 and type(v['content'][0]) != str:
                outdata += self._clean_up(v['content'], v['depth'], da+1)
            elif np.all([type(i) != str for i in v['content']]):
                outdata += self._clean_up(v['content'], v['depth'], da + 1)
            else:
                indent = self.outtab * (v['depth'] - da)
                outdata.append(indent + v['name'] + ":")
                outdata += self._clean_up(v['content'], v['depth'], da)
        return outdata

    def make_header(self, data):
        outdata = self._clean_up(data)
        hdr = [textwrap.fill(i, expand_tabs=False, replace_whitespace=False,
                             drop_whitespace=False, width=150,
                             subsequent_indent=re.match(r"\s*", i).group()
                             + "   ...  ")
               for i in outdata]
        return "\n".join(hdr).splitlines()