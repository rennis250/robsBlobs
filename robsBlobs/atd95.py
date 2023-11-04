import numpy as np

atd_jnd_macadam = 0.002
atd_jnd = 0.005 # page 243 of fairchild book, guth - "wide variety of color discrim phenomena"

# small diffs are quadrature sum of del_A1, del_T1, and del_D1
# large diffs are quadrature sum of del_A2, del_T2, and del_D2

def vos_corr_xyz(x, y, z):
    a = 1.027*x - 0.00008*y - 0.0009
    b = 0.03845*x + 0.01496*y + 1
    xp = a/b

    c = 0.00376*x + 1.0072*y + 0.00764
    d = 0.03845*x + 0.0149*y + 1
    yp = c/d

    zp = 1 - xp - yp

    return np.array([xp, yp, zp])


def xyz_to_lms(Xp, Yp, Zp):
    """
    Convert from XYZ color space to LMS color space using the ATD95 transformation.

    Args:
        Xp (float): X component of the color in XYZ color space.
        Yp (float): Y component of the color in XYZ color space.
        Zp (float): Z component of the color in XYZ color space.

    Returns:
        numpy.ndarray: Array containing the L, M, and S components of the color in LMS color space.
    """
    l = np.power(0.66 * (0.2435 * Xp + 0.8524 * Yp - 0.0516 * Zp), 0.7) + 0.024
    m = np.power(1.0 * (-0.3954 * Xp + 1.1642 * Yp + 0.0837 * Zp), 0.7) + 0.036
    s = np.power(0.43 * (0.04 * Yp + 0.6225 * Zp), 0.7) + 0.31

    return np.array([l, m, s])


# XYZ values should be Judd modified values
# XYZ adapt is normally assumed to be perfect white under the illumination
# for unrelated colors, k1 = 1.0 and k2 = 0.0
# for related colors, k1 = 0.0 and k2 is btween 15 and 50
# if adapted to stimulus and white point, then k1 = 1.0 and k2 = 5.0
# if we want a "real" von Kries transform, then  k1 = 0.0 and k2 = 50
def guth_atd(xyz_stim, xyz_adapt, Y_illum, k1, k2):
    """
    Computes the ATD (Adaptation-Test-Distance) color space values for a given stimulus and adaptation condition.

    Args:
    - xyz_stim (list): A list of 3 floats representing the tristimulus values of the stimulus.
    - xyz_adapt (list): A list of 3 floats representing the tristimulus values of the adaptation condition.
    - Y_illum (float): The luminance of the illuminant.
    - k1 (float): A scaling factor for the stimulus tristimulus values.
    - k2 (float): A scaling factor for the adaptation tristimulus values.

    Returns:
    - A dictionary containing the following ATD color space values:
        - A1 (float): First stage A response.
        - T1 (float): First stage T response.
        - D1 (float): First stage D response.
        - A2 (float): Second stage A response.
        - T2 (float): Second stage T response.
        - D2 (float): Second stage D response.
        - Br (float): Brightness.
        - H (float): Hue.
        - C (float): Saturation.
    """
    X = xyz_stim[0]
    Y = xyz_stim[1]
    Z = xyz_stim[2]

    s = X + Y + Z
    x = X/s
    y = Y/s
    z = Z/s
    vos_xyz = vos_corr_xyz(x, y, z)

    r = Y/vos_xyz[1]
    Xp = 18 * np.power(vos_xyz[0]*r, 0.8)
    Yp = 18 * np.power(vos_xyz[1]*r, 0.8)
    Zp = 18 * np.power(vos_xyz[2]*r, 0.8)

    lms = xyz_to_lms(Xp, Yp, Zp)
    l = lms[0]
    m = lms[1]
    s = lms[2]

    Xo = xyz_adapt[0]
    Yo = xyz_adapt[1]
    Zo = xyz_adapt[2]

    so = Xo + Yo + Zo
    xo = Xo/so
    yo = Yo/so
    zo = Zo/so
    vos_xoyozo = vos_corr_xyz(xo, yo, zo)

    r = Yo/vos_xoyozo[1]
    Xop = 18 * np.power(vos_xoyozo[0]*r, 0.8)
    Yop = 18 * np.power(vos_xoyozo[1]*r, 0.8)
    Zop = 18 * np.power(vos_xoyozo[2]*r, 0.8)

    Xa = k1 * Xp + k2 * Xop
    Ya = k1 * Yp + k2 * Yop
    Za = k1 * Zp + k2 * Zop

    lms_a = xyz_to_lms(Xa, Ya, Za)
    la = lms_a[0]
    ma = lms_a[1]
    sa = lms_a[2]

    # sig is normally 300, but can be varied to fit data better
    sig = 300

    lg = l * (sig/(sig + la))
    mg = m * (sig/(sig + ma))
    sg = s * (sig/(sig + sa))

    # first stage initial signals
    A1i = 3.57 * lg + 2.64 * mg
    T1i = 7.18 * lg - 6.21 * mg
    D1i = -0.70 * lg + 0.085 * mg + 1 * sg

    # second stage initial signals
    A2i = 0.09 * A1i
    T2i = 0.43 * T1i + 0.76 * D1i
    D2i = D1i

    # final atd responses
    A1 = A1i/(200 + np.abs(A1i))
    T1 = T1i/(200 + np.abs(T1i))
    D1 = D1i/(200 + np.abs(D1i))

    A2 = A2i/(200 + np.abs(A2i))
    T2 = T2i/(200 + np.abs(T2i))
    D2 = D2i/(200 + np.abs(D2i))

    brightness = np.sqrt(np.power(A1, 2) + np.power(T1, 2) + np.power(D1, 2))
    saturation = np.sqrt(np.power(T2, 2) + np.power(D2, 2))/A2

    # in guth's original formulation hue = T2/D2, but this does not work in practice
    hue = np.arctan2(T2, D2)
    if hue < 0:
        hue += 2*np.pi
    
    hue = np.rad2deg(hue)
    # hue = T2/D2

    return {
        "A1": A1,
        "T1": T1,
        "D1": D1,
        "A2": A2,
        "T2": T2,
        "D2": D2,
        "Br": brightness,
        "H": hue,
        "C": saturation
    }