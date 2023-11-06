import numpy as np

def rgb2hsv(rgb):
    """
    Convert an RGB color value to HSV.

    Args:
        rgb (tuple): A tuple containing the red, green, and blue values (in that order) of the color to be converted.

    Returns:
        numpy.ndarray: A 1D numpy array containing the hue, saturation, and value (in that order) of the converted color.
    """
    R = rgb[0]
    G = rgb[1]
    B = rgb[2]

    M = np.max(rgb)
    m = np.min(rgb)
    C = M - m

    S = 0
    Hprime = 0
    if C == 0:
        return np.array([Hprime, S, M])
    elif M == R:
        Hprime = np.mod((G - B)/C, 6)
    elif M == G:
        Hprime = (B - R)/C + 2
    elif M == B:
        Hprime = (R - G)/C + 4
    
    H = 60 * Hprime
    
    alpha = R - G * np.cos(np.deg2rad(60)) - B * np.cos(np.deg2rad(60))
    beta = G * np.sin(np.deg2rad(60)) - B * np.sin(np.deg2rad(60))
    H2 = np.arctan2(beta, alpha)
    if H2 < 0:
        H2 += 2*np.pi

    C2 = np.sqrt(np.power(alpha, 2) + np.power(beta, 2))

    V = M

    if V != 0:
        # S = C2/V
        S = C/V

    return np.array([H, S, V])