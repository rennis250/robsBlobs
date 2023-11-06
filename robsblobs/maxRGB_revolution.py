import numpy as np

# estimated from slide 34 of Karl's "Luminance 100 years" talk.
bright_JND = 1/1.65 # just take the inverse of the slope of the psychometric function
bright_JND = bright_JND/100 # since it was in units of 100% and we are typically working in [0,1] units
bright_JND

def maxRGB(rgb):
    """
    Returns the maximum value of the input RGB array.

    Parameters:
    rgb (numpy.ndarray): An array of RGB values.

    Returns:
    float: The maximum value of the input RGB array.
    """
    return np.max(rgb)


def maxWeightRGB(rgbs, wR, wG, wB):
    w_rgbs = np.array([[wR], [wG], [wB]]) * rgbs
    return np.array(np.max(w_rgbs, axis=0)).flatten()


def maxRGB_Shuchen(rgb):
    ws = np.array([0.40, 0.45, 0.15])
    return np.max(ws * rgb)


rand_weights = np.random.rand(1, 3)
def maxRGB_Random(rgb):
    return np.max(rand_weights * rgb)


def maxRGB_G0(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    return np.max([0.36593642*r,  0.42763098*g,  0.33796619*b])