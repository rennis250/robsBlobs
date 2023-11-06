import numpy as np

from .monitor import Monitor

def radiance(mon: Monitor, rgb):
    """
    Calculates the radiance of a given RGB value using the provided monitor object.

    Args:
        mon (Monitor): The monitor object containing the R, G, and B max spectra and max radiance values.
        rgb (tuple): A tuple containing the RGB values to calculate radiance for.

    Returns:
        float: The calculated radiance value.
    """
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    return (mon.R_max_spectrum*r + mon.G_max_spectrum*g + mon.B_max_spectrum*b).sum()/mon.maxRadiance


def visibleRadiance(mon: Monitor, rgb):
    """
    Calculates the visible radiance of a given RGB value on a given monitor.

    Args:
        mon (Monitor): The monitor object.
        rgb (tuple): The RGB value to calculate the visible radiance for.

    Returns:
        float: The visible radiance of the given RGB value on the given monitor.
    """
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    return (mon.visible_R_max_spectrum*r + mon.visible_G_max_spectrum*g + mon.visible_B_max_spectrum*b).sum()/mon.maxVisibleRadiance