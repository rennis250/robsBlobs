import numpy as np

from .cie_standard import plank_lasso, XYZ2xyY
from .monitor import Monitor

def to_u(XYZ):
    """
    Converts a 3D point in XYZ space to its corresponding U coordinate in UV space.

    Args:
        XYZ (tuple): A tuple containing the X, Y, and Z coordinates of the point in XYZ space.

    Returns:
        float: The U coordinate of the point in UV space.
    """
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    return (4*X) / (X + 15*Y + 3*Z)


def to_v(XYZ):
    """
    Converts a 3D point in XYZ space to its corresponding V coordinate in UV space.

    Args:
        XYZ (tuple): A tuple containing the X, Y, and Z coordinates of the point in XYZ space.

    Returns:
        float: The V coordinate of the point in UV space.
    """
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    return (9*Y) / (X + 15*Y + 3*Z)


def xyY2uvY(xyY):
    """
    Convert xyY color space to uvY color space.

    Parameters:
    xyY (numpy.ndarray): Array of xyY values.

    Returns:
    numpy.ndarray: Array of uvY values.
    """
    x = xyY[0]
    y = xyY[1]
    Y = xyY[2]

    u = (4*x) / (-2*x + 12*y + 3)
    v = (9*y) / (-2*x + 12*y + 3)

    return np.array([u, v, Y])


def uvY2xyY(uvY):
    """
    Convert uvY color space to xyY color space.

    Args:
        uvY (np.array): Array of length 3 containing u, v, and Y values.

    Returns:
        np.array: Array of length 3 containing x, y, and Y values.
    """
    u = uvY[0]
    v = uvY[1]
    Y = uvY[2]

    x = (9*u) / (6*u - 16*v + 12)
    y = (4*v) / (6*u - 16*v + 12)
    
    return np.array([x, y, Y])


def XYZ2LUV(mon: Monitor, XYZ):
    """
    Convert XYZ color space to LUV color space.

    Args:
        mon (Monitor): The monitor object.
        XYZ (numpy.ndarray): The input color in XYZ color space.

    Returns:
        numpy.ndarray: The color in LUV color space.
    """
    Y = XYZ[1]
    Yn = mon.monWP[1]

    delta = 6/29
    L = 0
    if Y/Yn <= np.power(delta, 3):
        L = np.power(29/3, 3) * Y/Yn
    else:
        L = 116 * np.power(Y/Yn, 1/3) - 16

    un = to_u(mon.monWP)
    vn = to_v(mon.monWP)

    u = to_u(XYZ)
    v = to_v(XYZ)

    u_star = 13*L*(u - un)
    v_star = 13*L*(v - vn)

    return np.array([L, u_star, v_star])


def XYZ2LUV_wo_mon(XYZ, WP):
    Y = XYZ[1]
    Yn = WP[1]

    delta = 6/29
    L = 0
    if Y/Yn <= np.power(delta, 3):
        L = np.power(29/3, 3) * Y/Yn
    else:
        L = 116 * np.power(Y/Yn, 1/3) - 16

    un = to_u(WP)
    vn = to_v(WP)

    u = to_u(XYZ)
    v = to_v(XYZ)

    u_star = 13*L*(u - un)
    v_star = 13*L*(v - vn)

    return np.array([L, u_star, v_star])


def LUV_chroma_DE(LUV1, LUV2):
    du = np.power(LUV1[0] - LUV2[0], 2)
    dv = np.power(LUV1[1] - LUV2[1], 2)

    return np.sqrt(du + dv)


plank_lasso_uvY = np.zeros((len(plank_lasso), 2))
for c in range(len(plank_lasso)):
    uvY = xyY2uvY(np.array([plank_lasso[c, 0], plank_lasso[c, 1], 100.0]))
    plank_lasso_uvY[c, :] = np.array([uvY[0], uvY[1]])