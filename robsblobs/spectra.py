import numpy as np
from scipy.interpolate import CubicSpline

from .cmfs import ciexyz_1931

konica_wlns = np.linspace(380, 781)

wavelength_cmf = np.array(list(ciexyz_1931[0]))
x_bar = np.array(list(ciexyz_1931[1]))
y_bar = np.array(list(ciexyz_1931[2]))
z_bar = np.array(list(ciexyz_1931[3]))

x_interp = CubicSpline(wavelength_cmf, x_bar)
x_new_cmf = x_interp(konica_wlns)
x_new_cmf[x_new_cmf < 0] = 0

y_interp = CubicSpline(wavelength_cmf, y_bar)
y_new_cmf = y_interp(konica_wlns)
y_new_cmf[y_new_cmf < 0] = 0

z_interp = CubicSpline(wavelength_cmf, z_bar)
z_new_cmf = z_interp(konica_wlns)
z_new_cmf[z_new_cmf < 0] = 0

diffs = np.diff(konica_wlns)
stepSize = np.zeros((len(diffs)+1))
stepSize[:len(diffs)] = diffs

# assume always using konica wln range for now
def spect2xyz(spect):
    """
    Convert spectral data to CIE XYZ color space.

    Parameters:
    spect (numpy.ndarray): Spectral data to be converted.

    Returns:
    numpy.ndarray: The converted CIE XYZ color space data.
    """
    corrCMF = np.zeros((len(x_new_cmf), 3))
    corrCMF[:, 0] = 683 * x_new_cmf * stepSize
    corrCMF[:, 1] = 683 * y_new_cmf * stepSize
    corrCMF[:, 2] = 683 * z_new_cmf * stepSize

    return spect.T @ corrCMF