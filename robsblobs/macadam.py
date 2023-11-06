import numpy as np
from scipy.interpolate import CubicSpline, LinearNDInterpolator

from .spectra import spect2xyz
from .illuminants import d65_xie_macadam
from .cie_standard import XYZ2xyY

minw = 380
maxw = 781
nwlns = 1000
wlns = np.linspace(minw, maxw, nwlns)
ncents = 100
nwidths = 50

cents = np.linspace(minw, maxw, ncents)
widths = np.linspace(1, maxw-minw, nwidths)

spect = np.zeros((ncents, nwidths, 2, len(wlns)))
for i in range(ncents):
    for j in range(nwidths):
        s = np.select(
            [
                np.logical_and(wlns >= cents[i]-widths[j]/2, wlns <= cents[i]+widths[j]/2), 
                np.logical_and(wlns <= cents[i]-widths[j]/2, wlns >= cents[i]+widths[j]/2)
            ],
            [
                1,
                0
            ]
        )

        spect[i, j, 0, :] = s
        spect[i, j, 1, :] = 1-s


konica_wlns = np.linspace(380, 781)
xyz = np.zeros((ncents, nwidths, 2, 3))
xyY = np.zeros((ncents, nwidths, 2, 3))
for i in range(ncents):
    for j in range(nwidths):
        s_interp = CubicSpline(wlns, spect[i, j, 0, :])
        s_new = s_interp(konica_wlns)
        xyz[i, j, 0, :] = spect2xyz(s_new * d65_xie_macadam)
        xyY[i, j, 0, :] = XYZ2xyY(xyz[i, j, 0, :])

        s_interp = CubicSpline(wlns, spect[i, j, 1, :])
        s_new = s_interp(konica_wlns)
        xyz[i, j, 1, :] = spect2xyz(s_new * d65_xie_macadam)
        xyY[i, j, 1, :] = XYZ2xyY(xyz[i, j, 1, :])


xs = xyY[:, :, :, 0].flatten()
ys = xyY[:, :, :, 1].flatten()
zs = xyY[:, :, :, 2].flatten()
macadam_interp = LinearNDInterpolator(list(zip(xs, ys)), zs)