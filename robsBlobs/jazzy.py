import numpy as np

from .cie_monitor_helpers import rgb2xyz
from .monitor import Monitor

b = 1.15
g = 0.66
c1 = 3424/np.power(2, 12)
c2 = 2413/np.power(2, 7)
c3 = 2392/np.power(2, 7)
n = 2610/np.power(2, 14)
p = 1.7 * 2523/np.power(2, 5)
d = -0.56
d0 = 1.6295499532821566e-11

def xyz_corr(XYZ):
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    x_prime = b*X - (b-1)*Z
    y_prime = g*Y - (g-1)*X

    return np.array([x_prime, y_prime, Z])


adapt = np.array([[0.41478972, 0.579999, 0.014648],
                  [-0.20151,   1.120649, 0.0531008],
                  [-0.0166008, 0.2648,   0.6684799]])


def xyz_to_lms(xyz_corr):
    return adapt @ xyz_corr


def lms_prime(lms):
    a = c1 + c2*np.power(lms/10000, n)
    DD = 1 + c3*np.power(lms/10000, n)
    c = a/DD
    return np.power(c, p)


to_iaz = np.array([[0.5, 0.5, 0],
                   [3.524000, -4.066708, 0.542708],
                   [0.199076, 1.096799, -1.295875]])

def iab(lms_prime):
    return to_iaz @ lms_prime


def to_jaz(iab):
    I_z = iab[0]
    a_z = iab[1]
    b_z = iab[2]

    a = (1 + d)*I_z
    DD = 1 + d*I_z
    J_z = a/DD - d0

    return np.array([J_z, a_z, b_z])


def xyz_to_jaz(XYZ):
    return to_jaz(iab(lms_prime(xyz_to_lms(xyz_corr(XYZ)))))


def jazzy(mon: Monitor, rgb):
    xyz = rgb2xyz(mon, rgb)
    return xyz_to_jaz(xyz)


def jaz_chroma(jaz):
    a = jaz[1]
    b = jaz[2]

    return np.sqrt(np.power(a, 2) + np.power(b, 2))


def jaz_hue(jaz):
    a = jaz[1]
    b = jaz[2]

    theta = np.arctan(b/a)
    if theta < 0:
        theta += 2*np.pi

    return theta


def jaz_lightness(jaz):
    return jaz[0]


def jaz_deltaE(jaz1, jaz2):
    J1 = jaz_lightness(jaz1)
    chroma1 = jaz_chroma(jaz1)
    hue1 = jaz_hue(jaz1)

    J2 = jaz_lightness(jaz2)
    chroma2 = jaz_chroma(jaz2)
    hue2 = jaz_hue(jaz2)

    deltaJ = J1 - J2
    deltaC = chroma1 - chroma2

    delta_little_h = hue1 - hue2
    deltaH = 2 * np.sqrt(chroma1*chroma2) * np.sin(delta_little_h/2)

    deltaE = np.sqrt(np.power(deltaJ, 2) + np.power(deltaC, 2) + np.power(deltaH, 2))

    return deltaE