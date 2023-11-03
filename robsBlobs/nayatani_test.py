import numpy as np
import colour

from .nayatani_insanity import robNT
from .cie_standard import XYZ2xy

# Table 11.1 Example Nayatani et al. color appearance model calculations
# Quantity Case 1 Case 2 Case 3 Case 4
Xs = [19.01, 57.06, 3.53, 19.01]
Ys = [20.00, 43.06, 6.56, 20.00]
Zs = [21.78, 31.96, 2.14, 21.78]
Xns = [95.05, 95.05, 109.85, 109.85]
Yns = [100.00, 100.00, 100.00, 100.00]
Zns = [108.88, 108.88, 35.58, 35.58]
Eos = [5000, 500, 5000, 500]
Eors = [1000, 1000, 1000, 1000]
Brs = [62.6, 67.3, 37.5, 44.2]
Lps = [50.0, 73.0, 24.5, 49.4]
Lns = [50.0, 75.9, 29.7, 49.4]
thetas = [257.5, 21.6, 190.6, 236.3]
Hs = [317.8, 2.1, 239.4, 303.6]
Hcs = ["82B 18R", "98R 2Y", "61G 39B", "96B 4R"]
Ss = [0.0, 37.1, 81.3, 40.2]
Cs = [0.0, 48.3, 49.3, 39.9]
Ms = [0.0, 42.9, 62.1, 35.8]

for tc in range(4):
    X = Xs[tc]
    Y = Ys[tc]
    Z = Zs[tc]

    Xn = Xns[tc]
    Yn = Yns[tc]
    Zn = Zns[tc]

    Eo = Eos[tc]
    Eor = Eors[tc]

    xy_stim = XYZ2xy(np.array([X, Y, Z]))
    Y_stim = Y/Eo
    # Y_stim = Y/Yn
    # Yo_bg = Yn/Eo
    Yo_bg = 0.18
    xy_illum = XYZ2xy(np.array([Xn, Yn, Zn]))

    res = robNT(xy_stim, Y_stim, Yo_bg, xy_illum, Eo, Eor)
    res_Br = res['Brightness']
    res_Lp = res['Lightness']
    res_Ln = res['Normalized Lightness']
    res_theta = res['Hue Angle']
    res_S = res['Saturation']
    res_C = res['Chroma']
    res_M = res['Colorfulness']

    XYZ = np.array([X, Y, Z])
    XYZ_n = np.array([Xn, Yn, Zn])
    Y_o = 0.18

    res_col = colour.appearance.nayatani95.XYZ_to_Nayatani95(XYZ, XYZ_n, Y_o, Eo, Eor)
    # res_col_Br = res_col.B_r
    res_col_Q = res_col.Q
    res_col_Lp = res_col.L_star_P
    res_col_Ln = res_col.L_star_N
    res_col_theta = res_col.h
    res_col_S = res_col.s
    res_col_C = res_col.C
    res_col_M = res_col.M

    if Brs[tc] != res_Br:
        print(tc, "Brs: ", Brs[tc], ", Rob: ", res_Br, "Colour: ", res_col_Q)

    if Lps[tc] != res_Lp:
        print(tc, "Lps: ", Lps[tc], ", Rob: ", res_Lp, "Colour: ", res_col_Lp)

    if Lns[tc] != res_Ln:
        print(tc, "Lns: ", Lns[tc], ", Rob: ", res_Ln, "Colour: ", res_col_Ln)

    if thetas[tc] != res_theta:
        print(tc, "thetas: ", thetas[tc], ", Rob: ", res_theta, "Colour: ", res_col_theta)

    if Ss[tc] != res_S:
        print(tc, "Ss: ", Ss[tc],  ", Rob: ",res_S, "Colour: ", res_col_S)

    if Cs[tc] != res_C:
        print(tc, "Cs: ", Cs[tc],  ", Rob: ",res_C, "Colour: ", res_col_C)

    if Ms[tc] != res_M:
        print(tc, "Ms: ", Ms[tc],  ", Rob: ",res_M, "Colour: ", res_col_M)