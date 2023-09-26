# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""Console script for raytools."""
import sys

import numpy as np

from clasp import click
import clasp.click_ext as clk
from scipy import stats

import raytools
from raytools.utility import pool_call
from raytools import imagetools, translate
from raytools.evaluate import hvsgsm


@click.group()
@click.option('-config', '-c', type=click.Path(exists=True),
              help="path of config file to load")
@click.option('-n', default=None, type=int,
              help='sets the environment variable RAYTOOLS_PROC_CAP set to'
                   ' 0 to clear (parallel processes will use cpu_limit)')
@click.option('--opts', '-opts', is_flag=True,
              help="check parsed options")
@click.option('--debug', is_flag=True,
              help="show traceback on exceptions")
@click.version_option(version=raytools.__version__)
@click.pass_context
def main(ctx, config=None, n=None,  **kwargs):
    """the raytools executable is a command line interface to utility commands
    as part of the raytools python package.
    """
    raytools.io.set_nproc(n)
    ctx.info_name = 'raytools'
    clk.get_config(ctx, config, None, None, None)


@main.command()
@click.argument("imgs", callback=clk.are_files)
@click.option("-metrics", callback=clk.split_str, default="illum dgp ugp",
              help='metrics to compute, choices: ["illum", '
                   '"avglum", "gcr", "ugp", "dgp", "tasklum", "backlum", '
                   '"dgp_t1", "log_gc", "dgp_t2", "ugr", "threshold", "pwsl2", '
                   '"view_area", "backlum_true", "srcillum", "srcarea", '
                   '"maxlum", "gss", "gssnb"]')
@click.option("--parallel/--no-parallel", default=True,
              help="use available cores")
@click.option("--peakn/--no-peakn", default=True,
              help="corrrect aliasing and/or filtering artifacts for direct sun"
                   " by assigning up to expected energy to peakarea")
@click.option("-peaka", default=6.7967e-05,
              help="expected peak area over which peak energy is distributed")
@click.option("-peakt", default=1.0e5,
              help="include down to this threshold in possible peak, note that"
                   "once expected peak energy is satisfied remaining pixels are"
                   "maintained, so it is safe-ish to keep this value low")
@click.option("-peakr", default=4.0,
              help="for peaks that do not meet expected area (such as partial"
                   " suns, to determines the ratio of what counts as part of"
                   " the source (max/peakr)")
@click.option("-threshold", default=2000.,
              help="same as the evalglare -b option. if factor is larger than "
                   "100, it is used as constant threshold in cd/m2, else this "
                   "factor is multiplied by the average task luminance. task "
                   "position is center of image with a 30 degree field of view")
@click.option("-scale", default=179.,
              help="scale factor applied to pixel values to convert to cd/m^2")
@click.option("--blursun/--no-blursun", default=False,
              help="applies human PSF to peak glare source (only if peakn=True")
@click.option("--gssimage/--no-gssimage", default=False,
              help="save gss pre sum images")
@click.option("--lowlight/--no-lowlight", default=False,
              help="use lowlight correction for dgp")
@clk.shared_decs(clk.command_decs(raytools.__version__, wrap=True))
def metric(ctx, imgs, metrics=None, parallel=True, peakn=False,
              peaka=6.7967e-05, peakt=1e5, peakr=4.0, threshold=2000.,
              scale=179., blursun=False, lowlight=False, gssimage=False, **kwargs):
    """calculate metrics for hdr images. This similar to evalglare but without
    glare source grouping, which is equivalent to -r 0 in evalglare.
    This ensures that all glare source positions are  weighted by the metrics
    to which they are applied. Additional peak normalization reduces the
    deviation between images processed in different ways, for example pfilt
    with -r, rpict drawsource(), or an undersampled vwrays | rtrace run where
    the pixels give a coarse estimate of the actual sun area.

    arguments:

    imgs: hdr image files, must be angular fisheye projection,
        if no view in header, assumes 180 degree

    Notes
    -----

    when "gss" is given as a metric, note that none of the other parameters
    apply.

    """
    if parallel:
        cap = None
    else:
        cap = 1
    try:
        gss = metrics.pop(metrics.index("gss"))
    except ValueError:
        gss = False
    try:
        gssnb = metrics.pop(metrics.index("gssnb"))
    except ValueError:
        gssnb = False
    if len(metrics) > 0:
        results = pool_call(imagetools.imgmetric, list(zip(imgs)), metrics,
                            cap=cap, desc="processing images", peakn=peakn,
                            peaka=peaka, peakt=peakt, peakr=peakr,
                            threshold=threshold, scale=scale, blursun=blursun,
                            lowlight=lowlight)
        results = np.asarray(results)
    else:
        results = None
    if gss:
        gresult = np.asarray(hvsgsm.gss_compute(imgs, save=gssimage))[:, None]
        metrics.append("gss")
        if results is not None:
            results = np.hstack((results, gresult))
        else:
            results = gresult
    if gssnb:
        gresult = np.asarray(hvsgsm.gss_compute(imgs, psf=False, adaptmove=False, directmove=False, save=gssimage))[:, None]
        metrics.append("gssnb")
        if results is not None:
            results = np.hstack((results, gresult))
        else:
            results = gresult
    print("image\t" + "\t".join(metrics))
    for img, result in zip(imgs, results):
        print(img + "\t" + "\t".join([f"{i:.06g}" for i in result]))


@main.command()
@click.argument("img", callback=clk.are_files)
@click.option("--uv2ang/--ang2uv", default=False,
              help="direction of transform")
@click.option("--useview/--no-useview", default=True,
              help="use view direction for transform ang2uv to match standard"
                   "projection")
@click.option("-rotate", default=0.0,
              help="degrees to rotate img (overrides projection, "
                   "assumes angular fisheye input)")
@click.option("-center", default=None, callback=clk.split_int,
              help="new image center give as pixel 'x y' (overrides projection,"
                   "assumes angular fisheye input)")
@click.option("--rotate-first/--center-first", default=True,
              help="order to apply rotation and centering")
@click.option("--nearest/--no-nearest", default=False,
              help="use nearest interpolation (only for center/rotate")
@clk.shared_decs(clk.command_decs(raytools.__version__, wrap=True))
def project(ctx, img, uv2ang=False, useview=True, rotate=0.0, center=None, rotate_first=True, nearest=False, **kwargs):
    """project images between angular and shirley-chiu square coordinates"""
    if rotate != 0 or center is not None:
        func = imagetools.hdr_rotate
    elif uv2ang:
        func = imagetools.hdr_uv2ang
    else:
        func = imagetools.hdr_ang2uv
    results = pool_call(func, img, expandarg=False, useview=useview,
                        rotate=rotate, center=center, rotate_first=rotate_first, nearest=nearest)
    print("Wrote the Following image files:", file=sys.stderr)
    print("\n".join(results), file=sys.stderr)


@main.command()
@click.argument("img", callback=clk.are_files)
@click.option("--reverse/--forward", default=False,
              help="direction of transform")
@click.option("--nearest/--no-nearest", default=False,
              help="use nearest interpolation (only for center/rotate")
@click.option("--stdout/--no-stdout", default=True,
              help="if only a single image on input, use stdout")
@click.option("-viewangle", type=float,
              help="if given overrides view size in header, if not given, "
                   "and not in header, then 180 is used")
@clk.shared_decs(clk.command_decs(raytools.__version__, wrap=True))
def solid2ang(ctx, img, reverse=False, nearest=False, stdout=True,
              viewangle=None, **kwargs):
    """project images between equisolid and equiangular"""
    func = imagetools.hdr_solid2ang
    if len(img) == 1 and stdout:
        func(img[0], outf=None, nearest=nearest, stdout=True, reverse=reverse,
             viewangle=viewangle)
    else:
        results = pool_call(func, img, nearest=nearest, reverse=reverse,
                            viewangle=viewangle)
        print("Wrote the Following image files:", file=sys.stderr)
        print("\n".join(results), file=sys.stderr)

@main.command()
@click.argument("dataf", callback=clk.is_file)
@click.option("-x", default=0,
              help="column index for x coord")
@click.option("-x_out", default="1000", callback=clk.split_float,
              help="give as a single value to resample between max and min, "
                   "or specify own range")
@click.option("-y", default=None, callback=clk.split_int,
              help="column index(es) for y coord(s), if none, returns KDE of x")
@click.option("-sigma", default=None, type=float,
              help="if None applies scotts rule (x.size**0.2)")
@click.option("-sf", default=1.0,
              help="scale factor for sigma")
@clk.shared_decs(clk.command_decs(raytools.__version__, wrap=True))
def smooth(ctx, dataf, x=0, x_out=(1000.0,), y=(-1,), sigma=None, sf=1.0,
           **kwargs):
    if y is None:
        y = []
    data = np.loadtxt(dataf)
    if len(data.shape) == 1:
        data = data[:, None]
    dx = data[:, x]
    dy = data[:, y].T
    if len(x_out) == 1:
        x_out = np.linspace(np.min(dx), np.max(dx), int(x_out[0]))
    if sigma is None:
        sigma = np.sqrt(np.cov(dx) * len(dx) ** (-.4))
    sigma *= sf
    if dx.size > 10000:
        bw = 500
    else:
        bw = None
    if dy.size == 0:
        k = stats.gaussian_kde(dx, bw_method=sigma)
        y_out = k(x_out)[None]
    else:
        y_out = translate.non_uniform_gaussian_filter(dx, dy, x_out,
                                                      sigma=sigma,
                                                      bw=bw)
    for x, yo in zip(x_out, y_out.T):
        print(x, *yo)





@main.result_callback()
@click.pass_context
def printconfig(ctx, returnvalue, **kwargs):
    """callback to cleanup any temp files"""
    try:
        clk.tmp_clean(ctx)
    except Exception:
        pass


if __name__ == '__main__':
    main()
