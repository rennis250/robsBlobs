import numpy as np

from .monitor import Monitor
from .cie_standard import xyY2XYZ

def rgb2xyz(mon: Monitor, rgb):
    return mon.RGB2XYZ @ rgb


def xyz2rgb(mon: Monitor, xyz):
    return mon.XYZ2RGB @ xyz


def xyY2rgb(mon: Monitor, xyY):
    return xyz2rgb(mon, xyY2XYZ(xyY))


def luminance(mon: Monitor, rgb):
        lums = rgb * [mon.monxyY[0, 2], mon.monxyY[1, 2], mon.monxyY[2, 2]]
        return np.array(lums).sum()/mon.maxLuminance
