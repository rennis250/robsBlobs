import numpy as np
import colour

from .atd95 import guth_atd

# Table 14.1 Example ATD color vision model calculations
# Quantity Case 1 Case 2 Case 3 Case 4
Xs = [19.01, 57.06, 3.53, 19.01]
Ys = [20.00, 43.06, 6.56, 20.00]
Zs = [21.78, 31.96, 2.14, 21.78]
Xos = [95.05, 95.05, 109.85, 109.85]
Yos = [100.00, 100.00, 100.00, 100.00]
Zos = [108.88, 108.88, 35.58, 35.58]
Yos_abs = [318.31, 31.83, 318.31, 31.83] # (cd/m2)
sigmas = [300, 300, 300, 300]
k1s = [0.0, 0.0, 0.0, 0.0]
k2s = [50.0, 50.0, 50.0, 50.0]
A1s = [0.1788, 0.2031, 0.1068, 0.1460]
T1s = [0.0287, 0.0680, -0.0110, 0.0007]
D1s = [0.0108, 0.0005, 0.0044, 0.0130]
A2s = [0.0192, 0.0224, 0.0106, 0.0152]
T2s = [0.0205, 0.0308, -0.0014, 0.0102]
D2s = [0.0108, 0.0005, 0.0044, 0.0130]
Brs = [0.1814, 0.2142, 0.1075, 0.1466]
Cs = [1.206, 1.371, 0.436, 1.091]
Hs = [1.91, 63.96, -0.31, 0.79]

for tc in range(len(Xs)):
    X = Xs[tc]
    Y = Ys[tc]
    Z = Zs[tc]

    Xo = Xos[tc]
    Yo = Yos[tc]
    Zo = Zos[tc]

    Yo_abs = Yos_abs[tc]

    sigma = sigmas[tc]

    k1 = k1s[tc]
    k2 = k2s[tc]

    xyz_stim = np.array([X, Y, Z])
    # xyz_stim[1] = xyz_stim[1] * Yo_abs
    # xyz_adapt = np.array([Xo, Yo_abs, Zo])
    xyz_adapt = np.array([Xo, Yo, Zo])

    res = guth_atd(xyz_stim, xyz_adapt, Yo_abs, k1, k2)
    A1 = res["A1"]
    T1 = res["T1"]
    D1 = res["D1"]

    A2 = res["A2"]
    T2 = res["T2"]
    D2 = res["D2"]

    Br = res["Br"]
    C = res["C"]
    H = res["H"]

    xyz_stim = np.array([X, Y, Z])
    xyz_adapt = np.array([Xo, Yo, Zo])

    res_col = colour.appearance.atd95.XYZ_to_ATD95(xyz_stim, xyz_adapt, Yo_abs, k1, k2)
    A1_col = res_col.A_1
    T1_col = res_col.T_1
    D1_col = res_col.D_1

    A2_col = res_col.A_2
    T2_col = res_col.T_2
    D2_col = res_col.D_2

    Br_col = res_col.Q
    C_col = res_col.C
    H_col = res_col.h

    if A1s[tc] != A1:
        print("A1: ", A1s[tc], " rob: ", A1, " colour: ", A1_col)

    if A2s[tc] != A2:
        print("A2: ", A2s[tc], " rob: ", A2, " colour: ", A2_col)

    if T1s[tc] != T1:
        print("T1: ", T1s[tc], " rob: ", T1, " colour: ", T1_col)

    if T2s[tc] != T2:
        print("T2: ", T2s[tc], " rob: ", T2, " colour: ", T2_col)

    if D1s[tc] != D1:
        print("D1: ", D1s[tc], " rob: ", D1, " colour: ", D1_col)

    if D2s[tc] != D2:
        print("D2: ", D2s[tc], " rob: ", D2, " colour: ", D2_col)

    if Brs[tc] != Br:
        print("Br: ", Brs[tc], " rob: ", Br, " colour: ", Br_col)

    if Cs[tc] != C:
        print("C: ", Cs[tc], " rob: ", C, " colour: ", C_col)

    if Hs[tc] != H:
        print("H: ", Hs[tc], " rob: ", H, " colour: ", H_col)