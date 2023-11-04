import numpy as np

def atan2pi_to_normal_pi(x):
    """
    Converts an angle in the range [-pi, pi] to the range [0, 2*pi].

    Args:
        x (float): Angle in radians.

    Returns:
        float: Angle in radians in the range [0, 2*pi].
    """
    if x < 0:
        return np.mod(x, 2*np.pi)
    else:
        return x
