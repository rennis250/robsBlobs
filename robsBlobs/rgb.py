import numpy as np

sRGB_xyY = np.array([[0.6400, 0.3300, 0.2126*100],
                    [0.3000, 0.6000, 0.7152*100],
                    [0.1500, 0.0600, 0.0722*100]])

sRGB_WP = np.array([0.3127, 0.3290, 1.0000*100])

sRGB2XYZ = np.array([[0.4124, 0.3576, 0.1805],
                     [0.2126, 0.7152, 0.0722],
                     [0.0193, 0.1192, 0.9505]])

XYZ2sRGB = np.linalg.inv(sRGB2XYZ)

def xyz2srgb_linear(XYZ):
    print("xyz2srgb_linear: linear srgb values!")
    return XYZ2sRGB @ XYZ


def linear_srgb2XYZ(srgb):
    print("linear_srgb2XYZ: needs linear srgb values!")
    return sRGB2XYZ @ srgb


def srgb_GammaCorrect(srgb):
    srgb_gc = np.zeros((3))
    cc = 0
    for c in srgb:
        if c <= 0.0031308:
            srgb_gc[cc] = 12.92*c
        else:
            srgb_gc[cc] = 1.055 * np.power(c, 1/2.4) - 0.055

        cc += 1
    
    return srgb_gc


def srgb_linearize(srgb_gc):
    srgb = np.zeros((3))
    cc = 0
    for c in srgb_gc:
        if c <= 0.04045:
            srgb[cc] = c/12.92
        else:
            srgb[cc] = np.power((c + 0.055)/1.055, 2.4)

        cc += 1

    return srgb


XYZ2CIERGB = np.array([[8041697, -3049000, -1591845],
                       [-1752003, 4851000, 301853],
                       [17697, -49000, 3432153]])
XYZ2CIERGB = (1/3400850) * XYZ2CIERGB

CIERGB2XYZ = np.linalg.inv(XYZ2CIERGB)

def xyz2ciergb(XYZ):
    return XYZ2CIERGB @ XYZ


def ciergb2xyz(rgb):
    return CIERGB2XYZ @ rgb