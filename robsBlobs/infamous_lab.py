import numpy as np

from .monitor import Monitor
from .cie_monitor_helpers import rgb2xyz, xyz2rgb
from .matlab import matlab_rgb2xyz

lab_delta = 6/29
def lab_f(t):
    if t > np.power(lab_delta, 3):
        return np.power(t, 0.3333333333333333)
    else:
        return t/(3 * (np.power(lab_delta, 2))) + 4/29


def xyz2lab(mon: Monitor, XYZ):
    xyz_s = XYZ / mon.monWP
    xsf = lab_f(xyz_s[0])
    ysf = lab_f(xyz_s[1])
    zsf = lab_f(xyz_s[2])

    l = 116 * ysf - 16
    a = 500 * (xsf - ysf)
    b = 200 * (ysf - zsf)

    return np.array([l, a, b])


def xyz2lab_wo_mon(WP, XYZ):
    xyz_s = XYZ / WP
    xsf = lab_f(xyz_s[0])
    ysf = lab_f(xyz_s[1])
    zsf = lab_f(xyz_s[2])

    l = 116 * ysf - 16
    a = 500 * (xsf - ysf)
    b = 200 * (ysf - zsf)

    return np.array([l, a, b])


def lab_inv_f(t):
    if t > lab_delta:
        return np.power(t, 3)
    else:
        return (3 * (np.power(lab_delta, 2))) * (t - 4/29)


def lab2xyz(mon: Monitor, LAB):
    l = LAB[0]
    a = LAB[1]
    b = LAB[2]

    x_pre_f = (l+16)/116 + (a/500)
    y_pre_f = (l+16)/116
    z_pre_f = (l+16)/116 - b/200

    X = mon.monWP[0] * lab_inv_f(x_pre_f)
    Y = mon.monWP[1] * lab_inv_f(y_pre_f)
    Z = mon.monWP[2] * lab_inv_f(z_pre_f)

    return np.array([X, Y, Z])


def rgb2lab(mon: Monitor, rgb):
    xyz = rgb2xyz(mon, rgb)
    return xyz2lab(mon, xyz)


def lab2rgb(mon: Monitor, lab):
    xyz = lab2xyz(mon, lab)
    return xyz2rgb(mon, xyz)


def matlab_rgb2lab(mon: Monitor, rgb):
    xyz = matlab_rgb2xyz(mon, rgb)
    return xyz2lab(mon, xyz)


def lstar(mon: Monitor, rgb):
    lab = rgb2lab(mon, rgb)
    return lab[0]


def chroma(mon: Monitor, rgb):
    lab = rgb2lab(mon, rgb)
    a = lab[1]
    b = lab[2]
    return np.sqrt(np.power(a, 2) + np.power(b, 2))


def hue(lab):
    a = lab[1]
    b = lab[2]

    h = np.arctan2(b, a)
    if h < 0:
        return h + 2*np.pi

    return h