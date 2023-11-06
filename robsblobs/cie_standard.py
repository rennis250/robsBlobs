import numpy as np

from .cmfs import ciexyz_1931

def XYZ2xyY(XYZ):
    """
    Convert CIE XYZ color space to CIE xyY color space.

    Parameters:
    XYZ (numpy.ndarray): An array of CIE XYZ values.

    Returns:
    numpy.ndarray: An array of CIE xyY values.
    """
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    s = X + Y + Z

    x = X/s
    y = Y/s

    return np.array([x, y, Y])


def XYZ2xy(XYZ):
    """
    Converts CIE 1931 XYZ tristimulus values to chromaticity coordinates (x, y).

    Parameters:
    XYZ (numpy.ndarray): Array of CIE 1931 XYZ tristimulus values.

    Returns:
    numpy.ndarray: Array of chromaticity coordinates (x, y).
    """
    xyY = XYZ2xyY(XYZ)
    x = xyY[0]
    y = xyY[1]

    return np.array([x, y])


def xyY2XYZ(xyY):
    """
    Convert xyY color space to XYZ color space.

    Args:
        xyY (numpy.ndarray): Array of xyY values.

    Returns:
        numpy.ndarray: Array of XYZ values.
    """
    x = xyY[0]
    y = xyY[1]
    Y = xyY[2]

    X = (Y/y) * x
    Z = (Y/y) * (1 - x - y)

    return np.array([X, Y, Z])


# generate the Planck-ian Locus (lasso) with the old school XYZ CMFs
plank_lasso = np.zeros((len(ciexyz_1931)+4, 2))
plank_colors = np.zeros((len(plank_lasso), 3))
def generatePlankLasso():
    """
    Generates a line of colors representing the Planckian locus in the CIE 1931 color space.

    Returns:
    None
    """
    for c in range(len(ciexyz_1931)):
        spect = np.zeros((len(ciexyz_1931), ))
        spect[c] = 1.0

        X = (np.array(list(ciexyz_1931[1])) * spect).sum()
        Y = (np.array(list(ciexyz_1931[2])) * spect).sum()
        Z = (np.array(list(ciexyz_1931[3])) * spect).sum()

        x = X/(X + Y + Z)
        y = Y/(X + Y + Z)

        plank_lasso[c, 0] = x
        plank_lasso[c, 1] = y

    # linearly interpolate between the first and last point to create the line of purples
    c = len(ciexyz_1931)
    for a in np.linspace(0, 1, 4):
        b = a*plank_lasso[0, :] + (1-a)*plank_lasso[len(ciexyz_1931)-1, :]
        plank_lasso[c, :] = b

        c += 1

    # r = 0
    # for c in range(len(plank_lasso)):
    #     xyY = np.array([plank_lasso[c, 0], plank_lasso[c, 1], 100])
    #     XYZ = xyY2XYZ(xyY)
    #     xyz2lab(mon: Monitor, XYZ)
    #     rgb = xyz2srgb(XYZ)
    #     plank_colors[c, :] = rgb