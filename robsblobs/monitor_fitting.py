import numpy as np
import scipy.io as sio
from scipy.optimize import curve_fit

inds = [0, 32, 64, 83, 128, 160, 192, 220, 255]

def gamma(x, g, a):
    """
    Calculate the gamma function for a given input x, power g, and coefficient a.

    Args:
        x (float): Input value.
        g (float): Power value.
        a (float): Coefficient value.

    Returns:
        float: Result of the gamma function.
    """
    return a*np.power(x, g)


def fit_gamma(fn):
    """
    Fits gamma curves to the red, green, and blue channels of an image.

    Args:
        fn (str): The filename of the image to fit gamma curves to.

    Returns:
        None
    """
    lum_mat = sio.loadmat(fn)
    
    rys = np.array(lum_mat["rys"])
    gys = np.array(lum_mat["gys"])
    bys = np.array(lum_mat["bys"])

    gams = []
    gains = []
    for fc in range(rys.shape[0]):
        params, pcov = curve_fit(gamma, np.array(inds)/255, rys[fc]/np.max(rys[fc]))
        rg = params[0]
        rga = params[1]
        params, pcov = curve_fit(gamma, np.array(inds)/255, gys[fc]/np.max(gys[fc]))
        gg = params[0]
        gga = params[1]
        params, pcov = curve_fit(gamma, np.array(inds)/255, bys[fc]/np.max(bys[fc]))
        bg = params[0]
        bga = params[1]

        gams.append([rg, gg, bg])
        gains.append([rga, gga, bga])

    sio.savemat(fn, {"gammas": np.array(gams), "gains": np.array(gains)})


# def konica_spect_to_xyz(spect):
#     wlns = np.arange(380, 780)
#     cmfData = csvread('ciexyz31_1.csv')

#     wavelength_cmf = cmfData[:, 0]
#     x_bar = cmfData[:, 1]
#     y_bar = cmfData[:, 2]
#     z_bar = cmfData[:, 3]

#     cmf = np.interp(wavelength_cmf, [x_bar, y_bar, z_bar], wlns, 'spline')
#     cmf[cmf < 0] = 0

#     XYZ = np.zeros((3, 1))

#     stepSize = [diff(wlns), 0]

#     corrCMF[:, 0] = 683 * cmf[:, 0] * stepSize
#     corrCMF[:, 1] = 683 * cmf[:, 1] * stepSize
#     corrCMF[:, 2] = 683 * cmf[:, 2] * stepSize

#     return spect.T @ corrCMF