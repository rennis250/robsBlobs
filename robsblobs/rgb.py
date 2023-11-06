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
    """
    Converts linear XYZ values to linear sRGB values using the XYZ2sRGB matrix.

    Args:
        XYZ (numpy.ndarray): A 3-element array containing the linear XYZ values.

    Returns:
        numpy.ndarray: A 3-element array containing the linear sRGB values.
    """
    print("xyz2srgb_linear: linear srgb values!")
    return XYZ2sRGB @ XYZ


def linear_srgb2XYZ(srgb):
    """
    Convert linear sRGB values to XYZ color space using the sRGB2XYZ matrix.

    Args:
        srgb (numpy.ndarray): Linear sRGB values as a 3-element array.

    Returns:
        numpy.ndarray: XYZ color space values as a 3-element array.
    """
    print("linear_srgb2XYZ: needs linear srgb values!")
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


XYZ2CIERGB = np.array([[8041697, -3049000, -1591845],
                       [-1752003, 4851000, 301853],
                       [17697, -49000, 3432153]])
XYZ2CIERGB = (1/3400850) * XYZ2CIERGB

CIERGB2XYZ = np.linalg.inv(XYZ2CIERGB)

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