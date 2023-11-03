import numpy as np

from .monitor import Monitor

def dkl2rgb(mon: Monitor, ldrgyv):
    return mon.DKL2RGB @ (ldrgyv/2.0 + 0.5)


def rgb2dkl(mon: Monitor, rgb):
    rgbScaled = 2.0 * (rgb - 0.5)
    return np.linalg.inv(mon.DKL2RGB) @ rgbScaled