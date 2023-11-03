import numpy as np

from .monitor import Monitor

def matlab_adapt_xyz(mon: Monitor, xyz_in):
    wp_D65_matlab = np.array([0.9505, 1.0000, 1.0888])

    # Bradford cone response model matrix
    Ma = np.array([
        [0.8951, 0.2664, -0.1614],
        [-0.7502, 1.7135, 0.0367],
        [0.0389, -0.0685, 1.0296],
    ])

    # Source white point cone response
    CRs = Ma @ wp_D65_matlab.T
    
    # Destination white point cone response
    CRd = Ma @ mon.monWP.T
    
    # Cone response domain scaling matrix
    S = np.diag(CRd / CRs)

    # Linear adaptation transform matrix
    M = np.linalg.lstsq(Ma, S @ Ma)

    return xyz_in @ M[0].T


def matlab_rgb2xyz(mon, rgb):
    rgb2xyz_mat = np.array([
        [0.4125, 0.3576, 0.1804],
        [0.2127, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9503],
    ])

    xyz_pre = rgb @ rgb2xyz_mat.T
    return matlab_adapt_xyz(mon, xyz_pre)
