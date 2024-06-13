import numpy as np

from .monitor import Monitor
from .cie_monitor_helpers import rgb2xyz

sRGB_xyY = np.array([[0.6400, 0.3300, 0.2126*100],
                    [0.3000, 0.6000, 0.7152*100],
                    [0.1500, 0.0600, 0.0722*100]])

sRGB_WP_xyY = np.array([0.3127, 0.3290, 1.0000*100])

sRGB2XYZ = np.array([[41.24, 35.76, 18.05],
                     [21.26, 71.52,  7.22],
                     [ 1.93, 11.92, 95.05]])

XYZ2sRGB = np.linalg.inv(sRGB2XYZ)

sRGB_WP_XYZ = np.array([95.04151515, 100.0, 108.91606061])

def xyz2srgb_linear(XYZ):
    """
    Converts linear XYZ values to linear sRGB values using the XYZ2sRGB matrix.

    Args:
        XYZ (numpy.ndarray): A 3-element array containing the linear XYZ values.

    Returns:
        numpy.ndarray: A 3-element array containing the linear sRGB values.
    """
    # print("xyz2srgb_linear: linear srgb values!")
    return XYZ2sRGB @ XYZ


def linear_srgb2XYZ(srgb):
    """
    Convert linear sRGB values to XYZ color space using the sRGB2XYZ matrix.

    Args:
        srgb (numpy.ndarray): Linear sRGB values as a 3-element array.

    Returns:
        numpy.ndarray: XYZ color space values as a 3-element array.
    """
    # print("linear_srgb2XYZ: needs linear srgb values!")
    return sRGB2XYZ @ srgb


def srgb_GammaCorrect(srgb):
    """
    Applies gamma correction to an sRGB color.

    Args:
        srgb (numpy.ndarray): An array of sRGB color values.

    Returns:
        numpy.ndarray: An array of gamma-corrected sRGB color values.
    """
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
    """
    Linearize sRGB gamma-compressed values.

    Args:
        srgb_gc (numpy.ndarray): Array of sRGB gamma-compressed values.

    Returns:
        numpy.ndarray: Array of linearized sRGB values.
    """
    srgb = np.zeros((3))
    cc = 0
    for c in srgb_gc:
        if c <= 0.04045:
            srgb[cc] = c/12.92
        else:
            srgb[cc] = np.power((c + 0.055)/1.055, 2.4)

        cc += 1

    return srgb


# for this to work correctly, the XYZ values must be normalized
CIERGB2XYZ = np.array([[8041697, -3049000, -1591845],
                       [-1752003, 4851000, 301853],
                       [17697, -49000, 3432153]])
CIERGB2XYZ = (1/3400850) * CIERGB2XYZ

XYZ2CIERGB = np.linalg.inv(CIERGB2XYZ)

def xyz2ciergb(XYZ):
    """
    Convert XYZ color space to CIE RGB color space.

    Parameters:
    XYZ (numpy.ndarray): An array of shape (3,) representing the XYZ color space.

    Returns:
    numpy.ndarray: An array of shape (3,) representing the CIE RGB color space.
    """
    return XYZ2CIERGB @ XYZ


def ciergb2xyz(rgb):
    """
    Convert a color from CIE RGB color space to CIE XYZ color space.

    Args:
        rgb (tuple): A tuple of three values representing the red, green, and blue components of the color.

    Returns:
        tuple: A tuple of three values representing the X, Y, and Z components of the color in the CIE XYZ color space.
    """
    return CIERGB2XYZ @ rgb



# to make it easier to keep track of chips and their ranks,
# we can make a little function that converts RGB coordinates
# to an index
def rgbToCode(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    return (r << 16) + (g << 8) + b


def codeToRGB(code):
    r = (code >> 16) & 0xFF
    g = (code >> 8) & 0xFF
    b = code & 0xFF

    return (r, g, b)


# {'CIE RGB', 'CIE decision 5', 'Wyszecki & Styles', 'Schanda (p. 8)'}:
# Relationship between the two CIE 1931 primary systems of color specifications
# (R, G, B) and (X, Y, Z).
# R 700.0 nm
# G 546.1 nm
# B 435.8 nm
# [Wyszecke & Stiles (1982), Color Science, Table  1(3.3.3.) p. 139]
#
# <quote>
# Decision 5
#
# The four stimuli that define the colorimetric scales will consist of
# monochromatic radiations of the following wavelengths: 700.0 nm,
# 546.1 nm, and 435.8 nm, and the standard light source B.
# </quote>
# [Schanda, Janos () Colorimetry. p. 8]
Thorsten_CIERGB_XYZ2RGB = np.array([
    # X:         Y:       Z:
    [0.73467,   0.26533,  0],            # R
    [0.27376,   0.71741,  0.00883],      # G
    [0.16658,   0.00886,  0.82456]       # B
])
# Schanda:  standard light source B = 0.34842 X + 0.35161 Y + 0.29997 Z
# conversion_matrix = 'Schanda p. 30'


# 'CIE XYZ Schanda p. 30':
# Note that the matrix is given for the direction RGB -> XYZ. So we invert 
# the matrix using inv to get T_XYZ2RGB for XYZ -> RGB
# this takes normalized XYZ values
Schada_RGB2XYZ = np.array([
    # R:         G:         B:  
    [2.768892,  1.751748,   1.130160],    # X
    [1,         4.590700,   0.060100],    # Y
    [0,         0.056508,   5.594292]     # Z
])
Thorsten_Schanda = np.linalg.inv(Schada_RGB2XYZ)


# he normalizes by sum of Y for white
# Thorsten's matrix for XYZ -> sRGB
Thorsten_XYZ2sRGB = np.array([
    [3.24096994, -1.53738318, -0.49861076],
    [-0.96924364, 1.8759675, 0.04155506],
    [0.05563008, -0.20397696, 1.05697151]
])


# see technical report here:
# https://www.adobe.com/digitalimag/pdfs/AdobeRGB1998.pdf
Adobe_XYZ2RGB = np.array([
    [2.04159,  -0.56501,  -0.34473],
    [-0.96924,  1.87597,   0.04156],
    [0.01344,  -0.11836,  1.01517]
])


import warnings
# convert RGB of one monitor to the RGB of another monitor that produces the same color
# the input rgb value should already be linearized and in the range of [0, 1]
def rgb2rgb(mon: Monitor, rgb, target_space="sRGB"):
    if target_space == "sRGB":
        # This works, but I personally do not like this normalized approach.
        # It is extra unnecessary effort that increases the chances of problems.
        # return Thorsten_XYZ2sRGB @ mon.RGB2XYZ_normed @ rgb

        return XYZ2sRGB @ rgb2xyz(mon, rgb)
    elif target_space == "AdobeRGB":
        # following procedure specified in technical report (see link above)
        XYZ_a = rgb2xyz(mon, rgb)

        Xa = XYZ_a[0]
        Ya = XYZ_a[1]
        Za = XYZ_a[2]

        Xw = mon.monWP[0]
        Yw = mon.monWP[1]
        Zw = mon.monWP[2]

        Xk = mon.monBP[0]
        Yk = mon.monBP[1]
        Zk = mon.monBP[2]

        X = (Xa - Xk)/(Xw - Xk) * (Xw/Yw)
        Y = (Ya - Yk)/(Yw - Yk)
        Z = (Za - Zk)/(Zw - Zk) * (Zw/Yw)
        
        return Adobe_XYZ2RGB @ np.array([X, Y, Z])
    elif target_space == "CIERGB":
        # need to read a bit more to be sure that this
        # is being done correctly

        warnings.warn('This is using a "normalized" XYZ matrix, but I am not certain yet if this is the "correct" CIERGB transformation.')

        return Thorsten_Schanda @ mon.RGB2XYZ_normed @ rgb


def clipRGB(rgb):
    return np.clip(rgb, 0, 1)