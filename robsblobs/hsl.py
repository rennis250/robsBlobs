import numpy as np

def hsl2rgb(hsl):
    H = hsl[0]
    S = hsl[1]
    L = hsl[2]

    C = (1 - np.abs(2*L - 1)) * S

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

    m = L - C/2

    R = R1 + m
    G = G1 + m
    B = B1 + m

    return np.array([R, G, B])