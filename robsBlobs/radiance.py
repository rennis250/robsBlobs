import numpy as np

from .monitor import Monitor

def radiance(mon: Monitor, rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    return (mon.R_max_spectrum*r + mon.G_max_spectrum*g + mon.B_max_spectrum*b).sum()/mon.maxRadiance


def visibleRadiance(mon: Monitor, rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    return (mon.visible_R_max_spectrum*r + mon.visible_G_max_spectrum*g + mon.visible_B_max_spectrum*b).sum()/mon.maxVisibleRadiance