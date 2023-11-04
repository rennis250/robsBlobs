import numpy as np

# what happens when using L+M light to stimulate
# a 2L+M cell.

from .monitor import Monitor

def rgb2lms(mon: Monitor, rgb):
    """
    Convert RGB values to LMS values using the given monitor calibration.

    Args:
        mon (Monitor): The monitor calibration to use for the conversion.
        rgb (array-like): The RGB values to convert.

    Returns:
        array-like: The LMS values corresponding to the input RGB values.
    """
    return mon.RGB2LMS @ rgb


# sources:
# http://cvrl.ucl.ac.uk/database/text/cienewxyz/cie2012xyz2.htm
# https://en.wikipedia.org/wiki/LMS_color_space#cite_note-13
def xyz2lms(xyz):
    """
    Convert from CIE XYZ color space to LMS color space using the CIE 2006 LMS to XYZ matrix.

    Parameters:
    -----------
    xyz : numpy.ndarray
        A 3-element array representing the XYZ color values.

    Returns:
    --------
    numpy.ndarray
        A 3-element array representing the LMS color values.
    """
    cie2006_lms2xyz = np.array([[1.94735469, -1.41445123, 0.36476327],
                                [0.68990272,  0.34832189, 0],
                                [0,           0,          1.93485343]])

    xyz2lms_mat = np.linalg.inv(cie2006_lms2xyz)
    return xyz2lms_mat @ xyz


def lms2dkl(lms):
    """
    Convert LMS color space to DKL color space.

    Parameters:
    -----------
    lms : numpy.ndarray
        Array of LMS color values.

    Returns:
    --------
    numpy.ndarray
        Array of DKL color values.
    """
    l = lms[:, 0].squeeze()
    m = lms[:, 1].squeeze()
    s = lms[:, 2].squeeze()

    ld = l + m
    rg = l - m
    yv = 2.0*s - ld

    dkl = np.zeros(lms.shape)
    dkl[:, 0] = ld
    dkl[:, 1] = rg
    dkl[:, 2] = yv
    
    return dkl


def maxLMS(lms):
    return np.max(lms)


def maxWeightLMS(lms_s, wL, wM, wS):
    w_lms_s = np.array([[wL], [wM], [wS]]) * lms_s
    return np.array(np.max(w_lms_s, axis=0)).flatten()


# TODO: get weights from Shuchen
def maxLMS_Shuchen(lms):
    ws = np.array([0.40, 0.45, 0.15])
    return np.max(ws * lms)


rand_weights = np.random.rand(1, 3)
def maxLMS_Random(lms):
    return np.max(rand_weights * lms)
