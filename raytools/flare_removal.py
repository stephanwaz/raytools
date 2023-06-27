from raytools import imagetools as im
import numpy as np
from raytools import io
from scipy import interpolate
from scipy import signal, ndimage
from clasp.script_tools import pipeline


def odd(x):
    return (np.floor(x / 2) * 2 + 1).astype(int)


def even(x):
    return (np.round(x / 2) * 2).astype(int)


def tag_peaks(img, vm, pxyz):
    peaks = np.asarray(pipeline([f'ximage -op {img}']).split(),
                       dtype=int).reshape(-1, 2)
    res = im.img_size(img)[-1]
    peak_xy = vm.ray2pixel(pxyz, res, False)
    peaks = peaks - peak_xy
    peaks = np.sort(np.arctan2(peaks[:, 1], peaks[:, 0]) + np.pi)
    return peaks * 180 / np.pi


def flare_removal(img, pxyz, peaks, size=200, prewitt_smoothing=4,
                  solar_omega=6.7967e-05, color=False):
    """

    Parameters
    ----------
    img : np.array
        path to image
    pxyz : np.array
        xyz direction to peak src
    peaks : Union[int, list]
        either the number (autodetect), or azimuths of peaks
        (see tag_peaks)
    size : int, optional
        radius in pixels to include from peak source for flare removal
    prewitt_smoothing : Union[int, float], optional
        sigma for gaussian blurring to smooth slope detection
        with prewitt operator
    solar_omega : float, optional
        expected area of source
    color : bool. optional
        do color image

    Returns
    -------
    corrected : np.array
        image array with flare removed
    flare : np.array
        image array of removed flare
    add_to_sun : float
        luminance of solar disc
    check : dict
        dense_grid arrays of individual steps
    """
    vm = im.hdr2vm(img)
    v, omega, lum = im.hdr2vol(img, vm=vm, color=color)
    res = int(round(omega.size ** .5))

    channels = 1
    if color:
        channels = 3

    # operate in pixel space (lens flare is on picture plane)
    peak_xys = vm.ray2pixel(pxyz, res, False)

    checks = []
    flares = []
    peak_luminances = []
    for pj, peak_xy in enumerate(peak_xys):
        # initialize output array
        corrected = np.atleast_2d(np.copy(lum)).T
        # translate to polar centered on peak
        pxy_peak = vm.pixels(res).reshape(-1, 2) - peak_xy[None]
        srcmask, radius, angle = xy_to_polar(pxy_peak, size)

        # order pixels along spiral sorted by radius and phi
        order, ri_s, spiral = unroll(radius, angle)
        lumb_s = corrected[srcmask][order]
        # use the largest concentric circle to establish resolution and upsample
        period = np.sum(ri_s == size - 1)
        dense_spiral_x, dense_spiral_y, asize = fill_spiral(spiral, ri_s, lumb_s,
                                                            period, size)
        # shape back into 2d where each row is a concentric circle
        dense_grid = dense_spiral_y.reshape(asize, -1, channels)


        # mask peaks
        d_img, prewitt = find_downhill(dense_grid, prewitt_smoothing)
        p_img, npeaks, prewitt2, get_to_peak = find_peaks(dense_grid, period, peaks, prewitt_smoothing)

        check = dict(source=dense_grid, radial_slope=prewitt, radial_mask=d_img,
                     rotational_slope=np.abs(prewitt2), get_to_peak=get_to_peak, rotational_mask=p_img)
        # combine masks
        p_img[:] = np.max(p_img * d_img, axis=2, keepdims=True)
        check["combined_mask"] = p_img
        # get fill values for peak areas
        bw = odd(period/(npeaks*2))
        ahead = ndimage.minimum_filter1d(dense_grid, bw, axis=1, origin=-int(bw/2), mode="wrap")
        behind = ndimage.minimum_filter1d(dense_grid, bw, axis=1, origin=int(bw/2), mode="wrap")
        rolling = dense_grid - np.maximum(ahead, behind)
        # rolling = dense_grid - ndimage.percentile_filter(dense_grid, 1, (1, odd(period/(npeaks+2)), 1))
        rolling[rolling < 0] = 0
        # weighted average of original and replacement based on p_img
        roll_out = dense_grid - p_img * rolling
        check["rolling"] = rolling * p_img

        if pj == 0:
            # radius of sun in pixels
            solar_rad = (solar_omega / np.pi) ** .5 / np.average(omega[srcmask]) ** .5

            b_ext = 6
            e_ext = 10
            # isolate center and linearly interpolate
            center = int(solar_rad*b_ext)
            # value at 5x
            extrap_val = roll_out[int(solar_rad*e_ext)]
            # value at 4x
            base_val = roll_out[center]
            # slope along each radial arm
            slope = (base_val - extrap_val)/(solar_rad*(e_ext-b_ext))
            # extrapolate
            roll_out[:center] = base_val[None] + slope[None] * np.arange(center)[::-1, None, None]
            blurz = int(solar_rad*(e_ext + b_ext)/2)
            roll_out[:blurz] = ndimage.gaussian_filter1d(roll_out[:blurz], period/20, axis=1, mode='wrap')
            check["output"] = roll_out

        # interpolate back to img pixels
        roll_out = roll_out.reshape(-1, channels)

        lumb_r = []
        for i in range(channels):
            lumb_r.append(interpolate.interp1d(dense_spiral_x, roll_out[:, i],
                                          assume_sorted=True, fill_value=(roll_out[0, i], roll_out[-1, i]),
                                          bounds_error=False)(spiral))

        lumb_r = np.stack(lumb_r).T
        # reverse sort and apply
        inv_sort = np.argsort(order, kind='stable')
        corrected[srcmask] = lumb_r[inv_sort]

        # find total impact of operations
        flare = lum.reshape(channels, res, res) - np.transpose(corrected.reshape(res, res, channels), (2, 0, 1))
        # use this value to correct solar luminance
        solar_luminance = np.sum(flare.reshape(channels, -1) * omega[None], axis=1) / solar_omega
        for k in check.keys():
            check[k] = np.squeeze(np.transpose(check[k], (2, 0, 1)))
        flares.append(np.squeeze(flare))
        peak_luminances.append(solar_luminance)
        checks.append(check)

    corrected = lum.reshape(channels, res, res) - np.sum(flares, axis=0)
    # corrected = np.transpose(corrected, (2, 0, 1))
    return np.squeeze(corrected), flares, peak_luminances, checks


def xy_to_polar(xy, maxrad):
    radius = np.linalg.norm(xy, axis=1)
    rmask = radius < maxrad
    # translate to polar coordinates
    radius = radius[rmask]
    angle = np.arctan2(xy[rmask, 1], xy[rmask, 0]) + np.pi
    return rmask, radius, angle


def unroll(radius, angle):
    # sort pixels by angle within concentric bands of radius=x
    radius_i = np.floor(radius)
    order = np.lexsort([angle, radius_i])
    # makes sorted arrays
    ri_s = radius_i[order]
    a_s = angle[order]
    # unroll into 1-D
    spiral = ri_s + a_s / np.pi / 2
    return order, ri_s, spiral


def fill_spiral(spiral, ri_s, lumb_s, period, size):
    # interpolate into dense array
    dense_spiral_y = []
    start_r = int(ri_s[0])
    asize = size - start_r
    dense_spiral_x = np.linspace(start_r, size, period * asize, endpoint=False)
    for i in range(lumb_s.shape[1]):
        s_int = interpolate.interp1d(spiral, lumb_s[:, i], assume_sorted=True,
                                     fill_value=(lumb_s[0, i], lumb_s[-1, i]),
                                     bounds_error=False)
        dense_spiral_y.append(s_int(dense_spiral_x))
    dense_spiral_y = np.stack(dense_spiral_y).T
    return dense_spiral_x, dense_spiral_y, asize


def find_downhill(eq_img, prewitt_smoothing):
    # prewitt makes sure rays are going downhill
    prewitt_base = ndimage.gaussian_filter(eq_img, (prewitt_smoothing, 0, 0))
    prewitt = ndimage.correlate1d(prewitt_base, [1, 0, -1], mode='nearest', axis=0)
    # mask everything to the right of uphill
    d_img = np.cumprod(prewitt > 0, axis=0)
    d_img = ndimage.maximum_filter(d_img.astype(float), (3, 3, 0))
    return ndimage.gaussian_filter(d_img, (3, 3, 0)), prewitt


def find_peaks(eq_img, period, peaks, prewitt_smoothing):
    peaki = np.round(np.sort(peaks) * period / 360).astype(int)
    peaks = len(peaks)
    prewitt_base = ndimage.gaussian_filter(eq_img, (0, prewitt_smoothing, 0))
    prewitt_r = ndimage.correlate1d(prewitt_base, [-1, 0, 1], mode='wrap', axis=1)
    prewitt_f = ndimage.correlate1d(prewitt_base, [1, 0, -1], mode='wrap', axis=1)
    peakmask = np.full(eq_img.shape, False)
    peakmask2 = np.full(eq_img.shape, False)
    for pi in peaki:
        fr = np.mod(np.arange(pi, pi+int(period/peaks)), period)
        br = np.arange(pi, pi-int(period/peaks), -1)

        for r, prewitt in [(fr, prewitt_f), (br, prewitt_r)]:
            get_to_peak = np.minimum.accumulate(eq_img[:, r], axis=1) == eq_img[:, pi:pi+1]
            # limit large misses on peak
            get_to_peak[:, 40:] = False
            downhill = prewitt[:, r] > 0
            downhill = np.cumprod(np.logical_or(get_to_peak, downhill), axis=1)
            peakmask[:, r] = np.logical_or(peakmask[:, r], downhill)
            peakmask2[:, r] = np.logical_or(peakmask2[:, r], get_to_peak)
    peakmask = ndimage.maximum_filter(peakmask.astype(float), (3, 3, 0))
    return ndimage.gaussian_filter(peakmask, (3, 3, 0)), peaks, prewitt_f, peakmask2
