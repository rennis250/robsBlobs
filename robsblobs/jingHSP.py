import numpy as np

newRW = 0.2667
newGW = 0.6267
newBW = 0.1067

def jingHSP(rgb, version='new'):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]

    N = 2.9

    t = 0
    if version == 'old':
        t = np.power(0.302*r, N) + np.power(0.550*g, N) + np.power(0.148*b, N)
    else:
        t = np.power(newRW*r, N) + np.power(newGW*g, N) + np.power(newBW*b, N)
    
    return np.power(t, 1/N)