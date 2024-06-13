import numpy as np

from .monitor import Monitor

def dkl2MB(mon: Monitor, dkl):
    rg = dkl[1]
    yv = dkl[2]

    mb = np.array([
        rg*mon.redMB[0] + mon.grayMB[0],
        yv*mon.blueMB[1] + mon.grayMB[1]
    ])

    return mb