import numpy as np

def atan2pi_to_normal_pi(x):
    if x < 0:
        return np.mod(x, 2*np.pi)
    else:
        return x
