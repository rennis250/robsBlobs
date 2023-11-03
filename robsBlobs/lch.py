import numpy as np

from .monitor import Monitor
from .infamous_lab import rgb2lab

def lab2lch(lab):
    L = lab[0]
    a = lab[1]
    b = lab[2]

    C = np.sqrt(np.power(a, 2) + np.power(b, 2))

    H = np.arctan2(b, a)
    if H < 0:
        H = H + 2*np.pi

    return np.array([L, C, H])


def rgb2lch(mon: Monitor, rgb):
    lab = rgb2lab(mon, rgb)
    return lab2lch(lab)