import numpy as np

def hsv2rgb(hsv):
    H = hsv[0]
    S = hsv[1]
    V = hsv[2]

    C = V * S

    Hprime = H/60

    X = C * (1 - np.abs(np.mod(Hprime, 2) - 1))

    R1 = 0
    G1 = 0
    B1 = 0

    if Hprime >= 0 and Hprime < 1:
        R1 = C
        G1 = X
        B1 = 0
    elif Hprime >= 1 and Hprime < 2:
        R1 = X
        G1 = C
        B1 = 0
    elif Hprime >= 2 and Hprime < 3:
        R1 = 0
        G1 = C
        B1 = X
    elif Hprime >= 3 and Hprime < 4:
        R1 = 0
        G1 = X
        B1 = C
    elif Hprime >= 4 and Hprime < 5:
        R1 = X
        G1 = 0
        B1 = C
    elif Hprime >= 5 and Hprime < 6:
        R1 = C
        G1 = 0
        B1 = X

    m = V - C

    R = R1 + m
    G = G1 + m
    B = B1 + m

    return np.array([R, G, B])


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