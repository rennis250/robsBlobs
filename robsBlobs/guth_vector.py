import numpy as np
from scipy.interpolate import CubicSpline

from .monitor import Monitor

konica_wlns = np.arange(380, 781)

wlns = np.array([420,
                 430,
                 440,
                 450,
                 459,
                 460,
                 470,
                 480,
                 490,
                 500,
                 510,
                 520,
                 526,
                 530,
                 540,
                 550,
                 560,
                 570,
                 580,
                 590,
                 600,
                 610,
                 620,
                 630,
                 640,
                 645,
                 650,
                 660,
                 670,
                 680])

A = np.array([-0.140,
              -0.240,
              -0.360,
              -0.470,
              -0.520,
              -0.520,
              -0.500,
              -0.460,
              -0.400,
              -0.360,
              -0.310,
              -0.260,
              -0.230,
              -0.200,
              -0.140,
              -0.080,
               0.020,
               0.200,
               0.460,
               0.700,
               0.830,
               0.890,
               0.920,
               0.940,
               0.955,
               0.960,
               0.965,
               0.970,
               0.975,
               0.980])
A_interp = CubicSpline(wlns, A)
A_adj = A_interp(konica_wlns)

B = np.array([0.640,
              0.680,
              0.800,
              0.850,
              0.880,
              0.880,
              0.890,
              0.910,
              0.920,
              0.930,
              0.950,
              0.955,
              0.960,
              0.965,
              0.970,
              0.970,
              0.965,
              0.950,
              0.860,
              0.710,
              0.550,
              0.460,
              0.400,
              0.340,
              0.310,
              0.290,
              0.280,
              0.245,
              0.215,
              0.190])
B_interp = CubicSpline(wlns, B)
B_adj = B_interp(konica_wlns)

C = np.array([0.710,
              0.650,
              0.460,
              0.250,
              0.070,
              0.070,
             -0.080,
             -0.160,
             -0.220,
             -0.240,
             -0.250,
             -0.260,
             -0.260,
             -0.260,
             -0.260,
             -0.250,
             -0.235,
             -0.200,
             -0.150,
             -0.070,
             -0.010,
              0.030,
              0.060,
              0.075,
              0.080,
              0.085,
              0.090,
              0.095,
              0.100,
              0.110])
C_interp = CubicSpline(wlns, C)
C_adj = C_interp(konica_wlns)

from .cmfs import V_lambda

# assume for now that all spectra were recorded with the konica cs2000-a
def guth_lum_spect(spect):
    """
    Calculates the Guth luminosity of a given spectrum.

    Parameters:
    spect (numpy.ndarray): The input spectrum.

    Returns:
    float: The Guth luminosity of the input spectrum.
    """
    spect_lum = V_lambda * spect

    As = np.power(np.sum(spect * A_adj), 2.0)
    Bs = np.power(np.sum(spect * B_adj), 2.0)
    Cs = np.power(np.sum(spect * C_adj), 2.0)

    return np.sqrt(As + Bs + Cs)


def guth_lum_rgb(mon: Monitor, rgb):
    """
    Calculates the Guth luminance for a given RGB color value.

    Args:
        mon (Monitor): The monitor object containing the maximum spectral values for each color channel.
        rgb (tuple): A tuple containing the red, green, and blue values for the color.

    Returns:
        float: The Guth luminance value for the given RGB color.
    """
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    R_spect = r * mon.R_max_spectrum
    G_spect = g * mon.G_max_spectrum
    B_spect = b * mon.B_max_spectrum

    spect = R_spect + G_spect + B_spect

    return guth_lum_spect(spect)