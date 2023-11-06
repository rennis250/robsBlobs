import numpy as np

from .monitor import Monitor
from .infamous_lab import rgb2lab

def lab2lch(lab):
    """
    Convert a color from CIELAB to CIELCH color space.

    Parameters:
    lab (numpy.ndarray): An array of shape (3,) representing the CIELAB color.

    Returns:
    numpy.ndarray: An array of shape (3,) representing the CIELCH color.
    """
    L = lab[0]
    a = lab[1]
    b = lab[2]

    C = np.sqrt(np.power(a, 2) + np.power(b, 2))

    H = np.arctan2(b, a)
    if H < 0:
        H = H + 2*np.pi

    return np.array([L, C, H])


def rgb2lch(mon: Monitor, rgb):
    """
    Convert RGB color space to LCH color space.

    Args:
        mon (Monitor): The monitor object.
        rgb (tuple): The RGB color values.

    Returns:
        tuple: The LCH color values.
    """
    lab = rgb2lab(mon, rgb)
    return lab2lch(lab)