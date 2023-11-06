import numpy as np
import colour

from .hunt import hunt_brightness
from .cie_standard import XYZ2xy

# Table 12.3 Example Hunt color appearance model calculations
# Quantity Case 1 Case 2 Case 3 Case 4
Xs = [19.01, 57.06, 3.53, 19.01]
Ys = [20.00, 43.06, 6.56, 20.00]
Zs = [21.78, 31.96, 2.14, 21.78]
XWs = [95.05, 95.05, 109.85, 109.85]
YWs = [100.00, 100.00, 100.00, 100.00]
ZWs = [108.88, 108.88, 35.58, 35.58]
LAs = [318.31, 31.83, 318.31, 31.83]
Ncs = [1.0, 1.0, 1.0, 1.0]
Nbs = [75, 75, 75, 75]
Discounting = [True, True, True, True]
hSs = [269.3, 18.6, 178.3, 262.8]
Hs = [317.2, 398.8, 222.2, 313.4]
HCs = ["83B 17R", "99R 1B", "78G 22B", "87B 13R"]
ss = [0.03, 153.36, 245.40, 209.29]
Qs = [31.92, 31.22, 18.90, 22.15]
Js = [42.12, 66.76, 19.56, 40.27]
C94s = [0.16, 63.89, 74.58, 73.84]
M94s = [0.16, 58.28, 76.33, 67.35]

for tc in range(4):
    X = Xs[tc]
    Y = Ys[tc]
    Z = Zs[tc]

    XW = XWs[tc]
    YW = YWs[tc]
    ZW = ZWs[tc]

    Nc = Ncs[tc]
    Nb = Nbs[tc]

    xy_illum = XYZ2xy(np.array([XW, YW, ZW]))
    Y_illum = LAs[tc]
    
    xy_stim = XYZ2xy(np.array([X, Y, Z]))
    Yref_stim = Y/LAs[tc]
    
    # assumptions about background and adapt
    # from Fairchild 
    xy_adapt = XYZ2xy(np.array([XW, YW, ZW]))
    Yref_adapt = 0.20
    
    xy_bkgd = XYZ2xy(np.array([XW, YW, ZW]))
    Yref_bkgd = 0.20

    discount = Discounting[tc]

    XYZ = np.array([X, Y, Z]) 
    XYZ_w = np.array([XW, YW, ZW])
    XYZ_b = np.array([XW, 0.20*YW, ZW])
    L_A = Y_illum
    T = colour.temperature.xy_to_CCT(xy_illum, "Hernandez 1999")

    # res = hunt_brightness(xy_stim, Yref_stim, xy_illum, Y_illum, xy_adapt, Yref_adapt, xy_bkgd, Yref_bkgd, Nc, Nb, discount, False)
    res = hunt_brightness(XYZ, XYZ_w, L_A, np.array([0, 0, 0]), XYZ_b, Nc, Nb, discount, False)
    res_Q = res['Q']
    # res_WB = res['WB']

    res_col = colour.appearance.hunt.XYZ_to_Hunt(XYZ, XYZ_w, XYZ_b, L_A, CCT_w=T, discount_illuminant=discount)
    res_col_Q = res_col.Q

    if Qs[tc] != res_Q:
        print(tc, "Qs: ", Qs[tc], ", Rob: ", res_Q, "Colour: ", res_col_Q)
