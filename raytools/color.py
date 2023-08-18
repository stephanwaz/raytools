from raytools import io, imagetools, translate
import colour
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx

colour.set_domain_range_scale("1")


def rgb_to_yuv(rgb):
    return colour.XYZ_to_Luv(rgb_to_xyz(rgb))


def xyz_to_rgb(xyz):
    irgb = colour.CCS_ILLUMINANTS['cie_10_1964']['D65']
    ixyz = colour.CCS_ILLUMINANTS['cie_10_1964']['D55']
    pri = colour.models.rgb.datasets.RGB_COLOURSPACE_sRGB.primaries
    cs = colour.RGB_Colourspace("sRGB", pri, irgb)
    return colour.XYZ_to_RGB(xyz, ixyz, irgb, cs.matrix_XYZ_to_RGB)


def rgb_to_xyz(rgb):
    irgb = colour.CCS_ILLUMINANTS['cie_10_1964']['D65']
    ixyz = colour.CCS_ILLUMINANTS['cie_10_1964']['D55']
    pri = colour.models.rgb.datasets.RGB_COLOURSPACE_sRGB.primaries
    cs = colour.RGB_Colourspace("sRGB", pri, irgb)
    return colour.RGB_to_XYZ(rgb, irgb, ixyz, cs.matrix_RGB_to_XYZ)


def rgb_to_cam16_ucs(rgb):
    cam = rgb_to_cam16(rgb)
    return cam16_to_cam16ucs(cam)


def cam16_to_cam16ucs(cam):
    JMh = np.stack((cam.J_HK, cam.M, cam.h)).T
    return colour.JMh_CAM16_to_CAM16UCS(JMh)


def rgb_samples():
    ca = 2**np.linspace(-2, 6, 64)
    rgb = np.stack(np.broadcast_arrays(ca[None, :, None], ca[None, None], ca[:, None, None])).reshape(3, -1)
    cam = rgb_to_cam16(rgb.T)
    ucs = cam16_to_cam16ucs(cam)
    print(ucs[:, 0].shape, cam.J.shape, cam.J_HK.shape, cam.Q.shape, cam.Q_HK.shape, cam.C.shape, cam.M.shape, cam.h.shape, cam.H.shape)
    # 0:J 1:a 2:b 3:ab 4:J(lightness) 5:J_HK 6:Q 7:Q_HK 8:C 9:M 10:s 11:h 12:H
    cama = np.stack((ucs[:, 0], ucs[:, 1], ucs[:, 2], np.sqrt(np.square(ucs[:, 1]) + np.square(ucs[:, 2])), cam.J, cam.J_HK, cam.Q, cam.Q_HK, cam.C, cam.M, cam.s, cam.h, cam.H)).T
    np.savetxt("rgb_samples_cam.txt", cama)
    # bright = (rgb[..., None] * np.array((1, 2, 4, 8))[None, None]).reshape(3, 1024, 1024)
    io.carray2hdr(rgb.reshape(3, 512, 512), "rgb_samples.hdr")
    # print(bright.shape)


def rgb_samples2():

    ca = np.linspace(0, 1, 512)
    cb = np.linspace(.1, 1, 512)
    hs = np.stack(np.broadcast_arrays(ca[None, :], cb[:, None])).reshape(2, -1)
    hsv = np.vstack((hs, 1*np.ones((1, hs.shape[1]))))
    rgb = colors.hsv_to_rgb(hsv.T).T
    # rgb = translate.norm(rgb.T).T
    cam = rgb_to_cam16(rgb.T)
    ucs = cam16_to_cam16ucs(cam)
    print(ucs[:, 0].shape, cam.J.shape, cam.J_HK.shape, cam.Q.shape, cam.Q_HK.shape, cam.C.shape, cam.M.shape, cam.h.shape, cam.H.shape)
    # 0:J 1:a 2:b 3:ab 4:J(lightness) 5:J_HK 6:Q 7:Q_HK 8:C 9:M 10:s 11:h 12:H
    cama = np.stack((ucs[:, 0], ucs[:, 1], ucs[:, 2], np.sqrt(np.square(ucs[:, 1]) + np.square(ucs[:, 2])), cam.J, cam.J_HK, cam.Q, cam.Q_HK, cam.C, cam.M, cam.s, cam.h, cam.H)).T
    np.savetxt("rgb_samples_cam2.txt", cama)
    # bright = (rgb[..., None] * np.array((1, 2, 4, 8))[None, None]).reshape(3, 1024, 1024)
    io.carray2hdr(rgb.reshape(3, 512, 512), "rgb_samples2.hdr")
    # print(bright.shape)


def rgb_samples3():
    ca = np.linspace(.01, 1, 64)
    xyY = np.stack(np.broadcast_arrays(ca[None, :, None], ca[:, None, None],
                                       ca[None, None])).reshape(3, -1)
    xyY = xyY[:, np.argsort(xyY[2], kind='stable')]
    XYZ = np.copy(xyY)
    XYZ[1] = xyY[2]
    XYZ[0] = xyY[2] * xyY[0] / xyY[1]
    XYZ[2] = xyY[2] / xyY[1] * (1 - xyY[0] - xyY[1])
    rgb = xyz_to_rgb(xyY.T).T
    cam = rgb_to_cam16(rgb.T)
    ucs = cam16_to_cam16ucs(cam)
    print(ucs[:, 0].shape, cam.J.shape, cam.J_HK.shape, cam.Q.shape,
          cam.Q_HK.shape, cam.C.shape, cam.M.shape, cam.h.shape, cam.H.shape)
    # 0:J 1:a 2:b 3:ab 4:J(lightness) 5:J_HK 6:Q 7:Q_HK 8:C 9:M 10:s 11:h 12:H
    cama = np.stack((ucs[:, 0], ucs[:, 1], ucs[:, 2],
                     np.sqrt(np.square(ucs[:, 1]) + np.square(ucs[:, 2])),
                     cam.J, cam.J_HK, cam.Q, cam.Q_HK, cam.C, cam.M, cam.s,
                     cam.h, cam.H)).T
    np.savetxt("rgb_samples_cam3.txt", cama)
    # bright = (rgb[..., None] * np.array((1, 2, 4, 8))[None, None]).reshape(3, 1024, 1024)
    io.carray2hdr(rgb.reshape(3, 512, 512), "rgb_samples3.hdr")
    # print(bright.shape)


def rgb_to_cam16(rgb):
    """

    Parameters
    ----------
    rgb :

    Returns
    -------
    cam :
        - J (float / NDArrayFloat | None) - Correlate of Lightness J.
        - C (float / NDArrayFloat / None) - Correlate of chroma C.
        - h (float / NDArrayFloat / None) - Hue angle h in degrees.
        - s (float / NDArrayFloat / None) - Correlate of saturation s.
        - Q (float / NDArrayFloat | None) - Correlate of brightness Q.
        - M (float | NDArrayFloat / None) - Correlate of colourfulness M.
        - H (float / NDArrayFloat | None) - Hue h quadrature H.
        - HC (float | NDArrayFloat | None) - Hue h composition H°.
        - J_HK (float / NDArrayFloat | None) - Correlate of Lightness Jwk
        accounting for Helmholtz-Kohlrausch effect.
        - Q_HK (float / NDArrayFloat | None) - Correlate of brightness Qk
        accounting for Helmholtz-Kohlrausch effect.
    """
    xyz = rgb_to_xyz(rgb)

    lum = io.rgb2rad(rgb)
    la = np.average(lum)
    xyz_a = rgb_to_xyz(np.array((np.max(lum), np.max(lum), np.max(lum))))
    print(xyz_a, xyz[np.argmax(lum)], np.max(lum), rgb[np.argmax(lum)])
    cam = colour.XYZ_to_Hellwig2022(xyz, xyz_a, la, la * .1)
    # cam = colour.XYZ_to_Hellwig2022(xyz, xyz[np.argmax(lum)], la, la * .1)
    return cam


def ap_bp_histo3(img):
    rgb = io.hdr2carray(img).reshape(3, -1).T * 179
    cam = rgb_to_cam16(rgb)
    ucs = cam16_to_cam16ucs(cam)
    print(np.percentile(cam.Q_HK, np.linspace(0,100,11)))
    H, _ = np.histogramdd(ucs, (10, 512, 512), ((0,100), (-1, 1), (-1, 1)))
    Hs = translate.resample(H, (10, 1024, 1024), gauss=True, radius=2)
    return Hs / np.sum(np.maximum(Hs,1), 0, keepdims=True)


def ap_bp_histo(img):
    rgb = io.hdr2carray(img).reshape(3, -1).T * 179
    cam = rgb_to_cam16(rgb)
    ucs = cam16_to_cam16ucs(cam)
    H, _ = np.histogramdd(ucs[:, 1:], (200, 200), ((-1, 1), (-1, 1)))
    Hs = translate.resample(H, (1000, 1000), gauss=True, radius=4)
    # print(np.percentile(Hs, (90, 95, 99, 100)))
    return Hs


def plot_ucs(img):
    """3d plot of uniform color scale, x and y are ap, bp and Z is Jp. color is determined by hue and then
    desaturated / darkened according to saturation and brightness. size is colourfulness"""
    rgb = io.hdr2carray(img).reshape(3, -1).T * 179
    cam = rgb_to_cam16(rgb)
    ucs = cam16_to_cam16ucs(cam)
    print(np.percentile(ucs, np.linspace(0, 100, 5), axis=0))
    print(np.percentile(cam.Q_HK, np.linspace(0, 100, 5), axis=0))
    ucs = np.hstack((ucs, cam.h[:, None], cam.s[:, None], cam.Q_HK[:, None], cam.C[:, None], cam.M[:, None]))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ss = np.random.default_rng(0).choice(ucs, 1000, axis=0).T
    norm = colors.Normalize(vmin=0, vmax=1)
    colormap = cmx.ScalarMappable(cmap='hsv', norm=norm)
    rgb0 = colormap.to_rgba(ss[3])[:, 0:3]
    hsv0 = colors.rgb_to_hsv(rgb0)
    hsv0[:, 1] *= (1 + ss[4]/np.max(ss[4]))/2
    hsv0[:, 2] *= (1 + ss[5]/np.max(ss[5]))/2
    si = ss[6]/ss[7]
    si = 1 + 10*si/np.max(si)
    rgb1 = colors.hsv_to_rgb(hsv0)
    ax.scatter(ss[2], ss[1], ss[0], c=rgb1, s=si)
    ax.set_xlabel(img)
    ax.set_xlim(-1,1)
    ax.set_ylim(-1,1)
    ax.set_zlim(0,1)
    plt.show()


def black_body_wp(ct):
    """CIE x,y coordinate from color temp."""
    return colour.XYZ_to_xy(colour.sd_to_XYZ(colour.sd_blackbody(ct)))


def xy_from_wpspec(wp1):
    if type(wp1) == str:
        wp1 = colour.CCS_ILLUMINANTS['cie_2_1931'][wp1]
    elif hasattr(wp1, "__len__"):
        if len(wp1) == 2:
            wp1 = wp1
        elif len(wp1) == 3:
            wp1 = colour.XYZ_to_xy(colour.sRGB_to_XYZ(wp1))
    else:
        wp1 = black_body_wp(wp1)
    return wp1


RADIANCE_PRIMARIES = (0.640, 0.330, 0.290, 0.600, 0.150, 0.060)


def srgb_wp(rgb, wp1, wp2=6500, p1=None, p2=None):
    """adjust RGB to new whitepoint specified by colortemp"""
    if p1 is None:
        p1 = colour.RGB_COLOURSPACES['sRGB'].primaries
    elif p1 == 'rad':
        p1 = RADIANCE_PRIMARIES
    if p2 is None:
        p2 = colour.RGB_COLOURSPACES['sRGB'].primaries
    elif p2 == 'rad':
        p2 = RADIANCE_PRIMARIES
    cs1 = colour.RGB_Colourspace('cs1', p1, xy_from_wpspec(wp1))
    cs2 = colour.RGB_Colourspace('cs2', p2, xy_from_wpspec(wp2))
    return colour.XYZ_to_RGB(colour.RGB_to_XYZ(rgb, cs1), cs2)


def srgb_lum(rgb):
    """luminance from srgb (must be D65, use srgb_wp)"""
    rgb = np.asarray(rgb).reshape(-1, 3)
    return 0.212656 * rgb[:, 0] + 0.715158 * rgb[:, 1] + 0.072186 * rgb[:, 2]


# def main():
#     cs_rad = colour.RGB_Colourspace('Radiance', (0.640,0.330, 0.290,0.600, 0.150,0.060), (1/3, 1/3))
#     cs_srgb2 = colour.RGB_Colourspace('test',
#                                     colour.RGB_COLOURSPACES['sRGB'].primaries,
#                                     xy_from_wpspec(5200))
#     # equivalent to CIE_709
#     cs_srgb = colour.RGB_COLOURSPACES['sRGB']
#     colour.XYZ_to_RGB()
#     npm52 = colour.CCS_ILLUMINANTS
#     # print(cs_rad.matrix_RGB_to_XYZ)
#     print(cs_srgb.matrix_XYZ_to_RGB)
    # print(colour.matrix_RGB_to_RGB(cs_rad, cs_srgb))
    # print(colour.RGB_to_RGB((1,.5,1), cs_srgb, cs_rad))
    # print(colour.RGB_to_)
    # print(list(colour.RGB_COLOURSPACES.keys()))
    # rgb_samples()
    # rgb_samples()
    # rgb_samples2()
    # for img in ["vtv", "vtv_frit2", "vtv_blue"]:
    #     rgb = io.hdr2carray(img + ".hdr").reshape(3, -1).T * 179
    #     cam = rgb_to_cam16(rgb)
    #     ucs = cam16_to_cam16ucs(cam)
    #     shp = imagetools.img_size(img + ".hdr")
    #     ucs = ucs.T.reshape(3, *shp)
    #     io.carray2hdr(ucs - np.min(ucs, axis=(1,2), keepdims=True), f"{img}_ucs.hdr")
    #     for k in ['Q', 'Q_HK']:
    #         io.array2hdr(cam.__getattribute__(k).reshape(shp), f"{img}_{k}.hdr")
    #
    # #     print(img)
    #     plot_ucs(f"{img}.hdr")
    #     Hs = ap_bp_histo(f"{img}.hdr")
    #     io.array2hdr(Hs, f"{img}_histo.hdr")