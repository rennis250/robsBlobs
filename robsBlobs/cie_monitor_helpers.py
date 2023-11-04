import numpy as np

from .monitor import Monitor
from .cie_standard import xyY2XYZ

def rgb2xyz(mon: Monitor, rgb):
    """
    Convert RGB color space to XYZ color space using the given monitor's RGB2XYZ matrix.

    Args:
        mon (Monitor): The monitor object containing the RGB2XYZ matrix.
        rgb (tuple): The RGB color to be converted.

    Returns:
        tuple: The converted XYZ color.
    """
    return mon.RGB2XYZ @ rgb


def xyz2rgb(mon: Monitor, xyz):
    """
    Convert a color from CIE XYZ color space to RGB color space.

    Args:
        mon (Monitor): The monitor object containing the color space conversion matrix.
        xyz (tuple): A tuple containing the CIE XYZ color values.

    Returns:
        tuple: A tuple containing the RGB color values.
    """
    return mon.XYZ2RGB @ xyz


def xyY2rgb(mon: Monitor, xyY):
    """
    Convert xyY color space to RGB color space.

    Args:
        mon (Monitor): The monitor object.
        xyY (tuple): A tuple containing the x, y, and Y values.

    Returns:
        tuple: A tuple containing the R, G, and B values.
    """
    return xyz2rgb(mon, xyY2XYZ(xyY))


def luminance(mon: Monitor, rgb):
    """
    Calculates the luminance of a given RGB color on a given monitor.

    Args:
        mon (Monitor): The monitor object representing the monitor on which the color is displayed.
        rgb (tuple): A tuple representing the RGB color to calculate the luminance for.

    Returns:
        float: The calculated luminance value.
    """
    lums = rgb * [mon.monxyY[0, 2], mon.monxyY[1, 2], mon.monxyY[2, 2]]
    return np.array(lums).sum()/mon.maxLuminance
