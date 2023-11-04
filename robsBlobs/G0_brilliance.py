import numpy as np
from scipy.interpolate import LinearNDInterpolator

from .monitor import Monitor
from .uv_space import uvY2xyY
from .cie_standard import XYZ2xyY
from .macadam import macadam_interp

G0_stims_uvY = np.array([
    [0.448, 0.5024, 72.94], # 1
    [0.3453, 0.4970, 179.31], # 2
    [0.2548, 0.4829, 338.67], # 3
    [0.2333, 0.5556, 361.72], # 4
    [0.2242, 0.5385, 435.82], # 5
    [0.2117, 0.5104, 431.28], # 6
    [0.1201, 0.5092, 334.39], # 7
    [0.1400, 0.5006, 387.97], # 8
    [0.1667, 0.4869, 466.44], # 9
    [0.1415, 0.4180, 310.11], # 10
    [0.1577, 0.4361, 379.57], # 11
    [0.1769, 0.4537, 448.53], # 12
    [0.2537, 0.3307, 181.88], # 13
    [0.2331, 0.3794, 275.61], # 14
    [0.2132, 0.4297, 411.25], # 15
    [0.1978, 0.4683, 530.00]  # 16
])

G0_stims_xyY = np.zeros(G0_stims_uvY.shape)
for c in range(len(G0_stims_xyY)):
    G0_stims_xyY[c, :] = uvY2xyY(G0_stims_uvY[c, :])


xs = G0_stims_xyY[:, 0]
ys = G0_stims_xyY[:, 1]
zs = G0_stims_xyY[:, 2]
brill_interp = LinearNDInterpolator(list(zip(xs, ys)), zs)

def local_rgb2xyz(mon: Monitor, rgb):
    return mon.monXYZ @ rgb

# brilliance according to the function in fig. 4.5 of Xie's thesis:
# https://scholarworks.rit.edu/cgi/viewcontent.cgi?article=12595&context=theses
# page 66 of the PDF file
def brilliance(mon: Monitor, rgb):
    """
    Calculates the brilliance of a given RGB color on a given monitor.

    Args:
        mon (Monitor): The monitor on which the color is displayed.
        rgb (tuple): The RGB color to calculate the brilliance for.

    Returns:
        float: The calculated brilliance value.
    """
    xyz = local_rgb2xyz(mon, rgb)
    xyY = XYZ2xyY(xyz)
    g0_boundary_lum = brill_interp(xyY[0], xyY[1])
    g0_normed_lum = xyY[2]/g0_boundary_lum
    return np.power(g0_normed_lum, 1/2.05)


def brilliance_macadam(mon: Monitor, rgb):
    """
    Calculates the brilliance of a color using the MacAdam method.

    Args:
        mon (Monitor): The monitor object used for calibration.
        rgb (tuple): The RGB values of the color.

    Returns:
        float: The brilliance of the color.
    """
    xyz = local_rgb2xyz(mon, rgb)
    xyY = XYZ2xyY(xyz)

    # in Xie's thesis, they related macadam optimal colors at 200 cd/m2 to G0
    macadam_boundary_lum = macadam_interp(xyY[0], xyY[1])
    g0_normed_lum = xyY[2]/macadam_boundary_lum
    return np.power(g0_normed_lum, 1/2.05)