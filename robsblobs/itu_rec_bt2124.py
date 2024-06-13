import numpy as np

from .monitor import Monitor
from .cie_monitor_helpers import rgb2xyz

bt2100_monxy = np.array([
    [0.708, 0.292],
    [0.170, 0.797],
    [0.131, 0.046]
])

bt2124_xyz2rgb = np.array([
    [ 1.7166511880, -0.3556707838, -0.2533662814],
    [-0.6666843518,  1.6164812366,  0.0157685458],
    [ 0.0176398574, -0.0427706133,  0.9421031212]
])

# step 0
def bt2124_xyz2rgb(xyz):
    return bt2124_xyz2rgb @ xyz


# step 1
def bt2124_rgb2lms(rgb):
    R = rgb[0]
    G = rgb[1]
    B = rgb[2]

    L = (1688*R + 2146*G + 262*B) / 4096
    M = (683*R + 2951*G + 462*B) / 4096
    S = (99*R + 309*G + 3688*B) / 4096

    return np.array([L, M, S])


# step 2
def eotf_inv_PQ(F):
    m1 = 2610/16384
    m2 = 2523/4096 * 128
    c1 = 3424/4096
    c2 = 2413/4096 * 32
    c3 = 2392/4096 * 32

    Y = F / 10000

    a = c1 + c2 * np.power(Y, m1)
    b = 1 + c3 * np.power(Y, m1)

    return np.power(a/b, m2)


def lms_prime_PQ(lms):
    Lp = eotf_inv_PQ(lms[0])
    Mp = eotf_inv_PQ(lms[1])
    Sp = eotf_inv_PQ(lms[2])

    return np.array([Lp, Mp, Sp])


# step 3
def lms2itp(lmsp):
    Lp = lmsp[0]
    Mp = lmsp[1]
    Sp = lmsp[2]

    I = 0.5*Lp + 0.5*Mp
    Ct = (6610*Lp - 13613*Mp + 7003*Sp) / 4096
    Cp = (17933*Lp - 17390*Mp - 543*Sp) / 4096

    return np.array([I, Ct, Cp])


# step 4
def rgb2itp_PQ(mon: Monitor, rgb):
    xyz = rgb2xyz(mon, rgb)
    bt2124_rgb = bt2124_xyz2rgb(xyz)
    bt2124_lms = bt2124_rgb2lms(bt2124_rgb)
    lmsp = lms_prime_PQ(bt2124_lms)
    itp = lms2itp(lmsp)

    # I = I
    # T = 0.5 * Ct
    # P = Cp

    return np.array([itp[0], 0.5*itp[1], itp[2]])


# step 5
def ITP_de_PQ(itp1, itp2):
    I1 = itp1[0]
    T1 = itp1[1]
    P1 = itp1[2]

    I2 = itp2[0]
    T2 = itp2[1]
    P2 = itp2[2]

    dI = I1 - I2
    dT = T1 - T2
    dP = P1 - P2

    return 720 * np.sqrt(dI**2 + dT**2 + dP**2)


# alternative step 5 - relative fidelity using HLG
# def rgb2itp_HLG(mon: Monitor, rgb):
    # xyz = rgb2xyz(mon, rgb)
    # bt2124_rgb = bt2124_xyz2rgb(xyz)
    # bt2124_lms = bt2124_rgb2lms(bt2124_rgb)
    # lmsp = lms_prime_HLG(bt2124_lms)
    # itp = lms2itp(lmsp)

    # I = I
    # T = 0.5 * 1.823698 * Ct
    # P = 1.887755 * Cp

    # return np.array([itp[0], 0.5 * 1.823698 * itp[1], 1.887755 * itp[2]])


def ITP_de_HLG(itp1, itp2):
    I1 = itp1[0]
    T1 = itp1[1]
    P1 = itp1[2]

    I2 = itp2[0]
    T2 = itp2[1]
    P2 = itp2[2]

    dI = I1 - I2
    dT = T1 - T2
    dP = P1 - P2

    return np.sqrt(dI**2 + dT**2 + dP**2)