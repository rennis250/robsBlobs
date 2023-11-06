import numpy as np

# Converts xyY to the Kaiser L** measure for perceived brightness
def ware_cowan(xyY):
    x = xyY[0]
    y = xyY[1]
    Y = xyY[2]

    a = 0.184 * y
    b = 2.527 * x * y
    c = 4.65 * np.power(x, 3) * y
    d = 4.657 * x * np.power(y, 4)

    F = 0.256 - a - b + c + d
    
    return np.log10(Y) + F