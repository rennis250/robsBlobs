import numpy as np

def X(x, y, Y):
    return (x/y) * Y


def Z(x, y, Y):
    return ((1 - x - y)/y) * Y


def xyY2XYZ(xyY):
    return [X(xyY[0], xyY[1], xyY[2]), xyY[2], Z(xyY[0], xyY[1], xyY[2])]


def p1(y, s1):
    return v1(s1) - v1(y)


def p2(y, vy, s3):
    if s3 <= y:
        return v1(y) - v1(s3)
    else:
        return v2(y, vy) - v2(s3, vy)
    

def v1(x):
    if x >= 0.0043:
        return np.power(100 * x - 0.43, 0.51) / (np.power(100 * x - 0.43, 0.51) + 31.75)
    else:
        return 0


def v2(x, vy):
    if x > 0.001 * k(vy):
        return np.power(100 * x/k(vy) - 0.1, 0.86) / (np.power(100 * x/k(vy) - 0.1, 0.86) + 103.2)
    else:
        return 0


def k(vy):
    return 0.140 + 0.175*vy


def f1(y, vy, s1, s3):
    return 700*p1(y, s1) - 54*p2(y, vy, s3)


def f2(y, vy, s3):
    return 96.5*p2(y, vy, s3)


def svf(txyz, bxyz):
    y = txyz[1]

    xyz2lms = np.array([[ 0.520,  0.589, -0.102],
                        [-0.194,  0.562,  0.034],
                        [ 0.007, -0.015,  0.907]])

    bxyz = bxyz/bxyz[1]

    tlms = xyz2lms @ txyz
    blms = xyz2lms @ bxyz
    tlmsn = tlms/blms

    s1 = tlmsn[0]
    s2 = tlmsn[1]
    s3 = tlmsn[2]

    vy = 40*v1(y)
    
    return np.array([f1(y, vy, s1, s3), f2(y, vy, s3), vy])


def svfde(tc1, bc1, tc2, bc2):
    svf1 = svf(tc1, bc1)
    c1f1 = svf1[0]
    c1f2 = svf1[1]
    c1vy = svf1[2]

    svf2 = svf(tc2, bc2)
    c2f1 = svf2[0]
    c2f2 = svf2[1]
    c2vy = svf2[2]

    return np.sqrt(np.power(c1f1 - c2f1, 2) + np.power(c1f2 - c2f2, 2) + np.power(2.3*(c1vy - c2vy), 2))


def svfsat(f1, f2, vy):
    return np.sqrt(np.power(f1, 2) + np.power(f2, 2))/vy