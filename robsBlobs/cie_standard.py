import numpy as np

from .cmfs import ciexyz_1931

def XYZ2xyY(XYZ):
    X = XYZ[0]
    Y = XYZ[1]
    Z = XYZ[2]

    s = X + Y + Z

    x = X/s
    y = Y/s

    return np.array([x, y, Y])


def XYZ2xy(XYZ):
    xyY = XYZ2xyY(XYZ)
    x = xyY[0]
    y = xyY[1]

    return np.array([x, y])


def xyY2XYZ(xyY):
    x = xyY[0]
    y = xyY[1]
    Y = xyY[2]

    X = (Y/y) * x
    Z = (Y/y) * (1 - x - y)

    return np.array([X, Y, Z])


# generate the Planck-ian Locus (lasso) with the old school XYZ CMFs
plank_lasso = np.zeros((len(ciexyz_1931)+4, 2))
plank_colors = np.zeros((len(plank_lasso), 3))
def generatePlankLasso():
    for c in range(len(ciexyz_1931)):
        spect = np.zeros((len(ciexyz_1931), ))
        spect[c] = 1.0

        X = (np.array(list(ciexyz_1931[1])) * spect).sum()
        Y = (np.array(list(ciexyz_1931[2])) * spect).sum()
        Z = (np.array(list(ciexyz_1931[3])) * spect).sum()

        x = X/(X + Y + Z)
        y = Y/(X + Y + Z)

        plank_lasso[c, 0] = x
        plank_lasso[c, 1] = y

    # linearly interpolate between the first and last point to create the line of purples
    c = len(ciexyz_1931)
    for a in np.linspace(0, 1, 4):
        b = a*plank_lasso[0, :] + (1-a)*plank_lasso[len(ciexyz_1931)-1, :]
        plank_lasso[c, :] = b

        c += 1

    # r = 0
    # for c in range(len(plank_lasso)):
    #     xyY = np.array([plank_lasso[c, 0], plank_lasso[c, 1], 100])
    #     XYZ = xyY2XYZ(xyY)
    #     xyz2lab(mon: Monitor, XYZ)
    #     rgb = xyz2srgb(XYZ)
    #     plank_colors[c, :] = rgb