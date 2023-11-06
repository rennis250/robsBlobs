import numpy as np

from .monitor import Monitor

def dkl2rgb(mon: Monitor, ldrgyv):
    """
    Convert a vector of DKL values to RGB values.

    Args:
        mon (Monitor): The monitor object.
        ldrgyv (numpy.ndarray): A vector of DKL values.

    Returns:
        numpy.ndarray: A vector of RGB values.
    """
    return mon.DKL2RGB @ (ldrgyv/2.0 + 0.5)


def rgb2dkl(mon: Monitor, rgb):
    """
    Convert an RGB color to DKL color space.

    Args:
        mon (Monitor): The monitor object containing the DKL2RGB matrix.
        rgb (numpy.ndarray): The RGB color to convert.

    Returns:
        numpy.ndarray: The DKL color space representation of the input RGB color.
    """
    rgbScaled = 2.0 * (rgb - 0.5)
    return np.linalg.inv(mon.DKL2RGB) @ rgbScaled