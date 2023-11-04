import numpy as np
import colour

from .cie_standard import xyY2XYZ, XYZ2xy

to_lms = np.array([[0.38971, 0.68898, -0.07868],
                   [-0.22981, 1.18340, 0.04641],
                   [0.0, 0.0, 1.0]])

def xyz_to_cone(xyz):
    """
    Convert a set of XYZ coordinates to cone coordinates using the transformation matrix to_lms.

    Args:
        xyz (numpy.ndarray): A 3-element array representing the XYZ coordinates.

    Returns:
        numpy.ndarray: A 3-element array representing the cone coordinates.
    """
    return to_lms @ xyz


def f_n(I):
    a = np.power(I, 0.73)
    b = np.power(I, 0.73) + 2
    return 40*(a/b)


def adapt_cones(lms, lms_w, lms_p, La, Fl, Y_b, Y_white, discount, helson_judd):
    """
    Adapt cone responses to background illumination and discounting.

    Args:
    - lms (list): cone responses to the stimulus
    - lms_w (list): cone responses to the white point
    - lms_p (list): cone responses to the adapting stimulus
    - La (float): adapting luminance
    - Fl (float): luminance adaptation factor
    - Y_b (float): luminance of the background
    - Y_white (float): luminance of the white point
    - discount (bool): whether to discount the adapting stimulus
    - helson_judd (bool): whether to use the Helson-Judd effect

    Returns:
    - np.array: adapted cone responses
    """
    p = lms[0]
    g = lms[1]
    b = lms[2]

    p_w = lms_w[0]
    g_w = lms_w[1]
    b_w = lms_w[2]

    p_p = lms_p[0]
    g_p = lms_p[1]
    b_p = lms_p[2]

    t = p_w + g_w + b_w
    hp = 3 * p_w/t
    hg = 3 * g_w/t
    hb = 3 * b_w/t

    if not discount:
        Fp = (1 + np.power(La, 1/3) + hp)/(1 + np.power(La, 1/3) + 1/hp)
        Fg = (1 + np.power(La, 1/3) + hg)/(1 + np.power(La, 1/3) + 1/hg)
        Fb = (1 + np.power(La, 1/3) + hb)/(1 + np.power(La, 1/3) + 1/hb)
    else:
        Fp = 1.0
        Fg = 1.0
        Fb = 1.0

    fn_P = f_n(Fl * Fp * p/p_w)
    fn_G = f_n(Fl * Fg * g/g_w)
    fn_B = f_n(Fl * Fb * b/b_w)

    if helson_judd:
        r = Y_b/Y_white
        p_D = f_n(r * Fl * Fg) - f_n(r * Fl * Fp)
        g_D = 0.0
        b_D = f_n(r * Fl * Fg) - f_n(r * Fl * Fb)
    else:
        p_D = 0.0
        g_D = 0.0
        b_D = 0.0

    beta_p = np.power(10, 7)/(np.power(10, 7) + 5*La*(p_w/100))
    beta_g = np.power(10, 7)/(np.power(10, 7) + 5*La*(g_w/100))
    beta_b = np.power(10, 7)/(np.power(10, 7) + 5*La*(b_w/100))

    p_adapt = beta_p*(fn_P + p_D) + 1
    g_adapt = beta_g*(fn_G + g_D) + 1
    b_adapt = beta_b*(fn_B + b_D) + 1

    return np.array([p_adapt, g_adapt, b_adapt])


def achrom_cone(pgb):
    """
    Calculates the achromatic cone response for a given pixel.

    Args:
        pgb (tuple): A tuple containing the pixel values for the red, green, and blue channels.

    Returns:
        float: The achromatic cone response for the given pixel.
    """
    p = pgb[0]
    g = pgb[1]
    b = pgb[2]

    return 2*p + g + (1/20)*b - 3.05 + 1


def C(pgb):
    p = pgb[0]
    g = pgb[1]
    b = pgb[2]

    return np.array([
        p - g,
        g - b,
        b - p
    ])


def h_s(C):
    C1 = C[0]
    C2 = C[1]
    C3 = C[2]

    a = 0.5 * (C2 - C3)/4.5
    b = C1 - (C2/11)

    h = np.arctan2(a, b)
    if h < 0:
        h += 2*np.pi

    return np.rad2deg(h)


unique_red = {'hs': 20.14, 'es': 0.8}
unique_yellow = {'hs': 90.00, 'es': 0.7}
unique_green = {'hs': 164.25, 'es': 1.0}
unique_blue = {'hs': 237.53, 'es': 1.2}
def H(hs):
    # print("initial hs: " + str(hs))
    mhc = 0
    if hs > 0 and hs < unique_red['hs']:
        mhc = 3
        hs = hs + 360 # for proper es interpolation
    elif hs > unique_red['hs'] and hs < unique_yellow['hs']:
        mhc = 0
    elif hs > unique_yellow['hs'] and hs < unique_green['hs']:
        mhc = 1
    elif hs > unique_green['hs'] and hs < unique_blue['hs']:
        mhc = 2
    elif hs > unique_blue['hs'] and hs < unique_red['hs'] + 360:
        mhc = 3

    # rd = hs - unique_red['hs']
    # yd = hs - unique_yellow['hs']
    # gd = hs - unique_green['hs']
    # bd = hs - unique_blue['hs']

    # mhc = 0
    # hds = np.array([rd, yd, gd, bd])
    # print("hds: ", hds)
    # for hdc in range(len(hds)):
        # if hdc < 0:
            # mhc = hdc - 1
            # break
    
    # mhc = np.argmin(hds)
    # if hds[mhc] > 0:
        # mhc = mhc - 1
    
    # if mhc < 0:
        # mhc = 3

    if mhc == 0:
        H1 = 0

        h1 = unique_red['hs']
        e1 = unique_red['es']

        h2 = unique_yellow['hs']
        e2 = unique_yellow['es']
    elif mhc == 1:
        H1 = 100

        h1 = unique_yellow['hs']
        e1 = unique_yellow['es']

        h2 = unique_green['hs']
        e2 = unique_green['es']
    elif mhc == 2:
        H1 = 200

        h1 = unique_green['hs']
        e1 = unique_green['es']

        h2 = unique_blue['hs']
        e2 = unique_blue['es']
    elif mhc == 3:
        H1 = 300

        h1 = unique_blue['hs']
        e1 = unique_blue['es']

        h2 = unique_red['hs'] + 360 # for proper es interpolation
        e2 = unique_red['es']


    a = 100 * (hs - h1)/e1
    b = (hs - h1)/e1 + (h2 - hs)/e2

    hue = H1 + a/b

    a = (hs - h1)/(h2 - h1)
    es = (1 - a)*e1 + a*e2

    # uhues = np.array([
        # unique_red['hs'],
        # unique_yellow['hs'],
        # unique_green['hs'],
        # unique_blue['hs']
    # ])
    # ecens = np.array([
        # unique_red['es'],
        # unique_yellow['es'],
        # unique_green['es'],
        # unique_blue['es']
    # ])
    # es = np.interp(hs, uhues, ecens)

    # print("mhc: " + str(mhc))
    # print("hs: " + str(hs))
    # print("uhues: ", uhues)
    # print("ecens: ", ecens)

    # print("e1: " + str(e1))
    # print("e2: " + str(e2))
    # print("a: " + str(a))

    # print("es _ np: " + str(es))
    # print("es _ rob: " + str(es_rob))

    return hue, es


def M_YB(C, La, Nc, Ncb, es):
    C2 = C[1]
    C3 = C[2]

    Ft = La/(La + 0.1)

    a = 0.5 * (C2 - C3)/4.5
    b = es * (10/13) * Nc * Ncb * Ft
    
    return 100 * a * b


def M_RG(C, Nc, Ncb, es):
    C1 = C[0]
    C2 = C[1]

    a = C1 - (C2/11)
    b = es * (10/13) * Nc * Ncb

    return 100 * a * b


def M(M_YB, M_RG):
    return np.sqrt(np.power(M_YB, 2) + np.power(M_RG, 2))


def s(M, pgb):
    p = pgb[0]
    g = pgb[1]
    b = pgb[2]

    return 50 * M/(p + g + b)


def achrom_signal(Las, A, Nbb, S_S_w):
    j = 0.00001/(5*Las/2.26 + 0.00001)

    Fls = 3800 * np.power(j, 2) * (5*Las/2.26) + 0.2 * np.power(1 - np.power(j, 2), 4) * np.power(5*Las/2.26, 1/6)

    # should it be to power of 0.4? that is what is in the "colour" python package, but not the fairchild book
    # Fls = 3800 * np.power(j, 2) * (5*Las/2.26) + 0.2 * np.power(1 - np.power(j, 2), 0.4) * np.power(5*Las/2.26, 1/6)

    a = (5*Las/2.26)*S_S_w
    b = 1 + 5*(5*Las/2.26)
    c = 1 + 0.3*np.power(a, 0.3)
    Bs = 0.5/c + 0.5/b

    As = 3.05 * Bs * (f_n(Fls * S_S_w)) + 0.3

    return Nbb * (A - 1 + As - 0.3 + np.sqrt(1 + np.power(0.3, 2)))


def brightness(asig_a, asig_w, Nb, M):
    a = np.power(7*asig_w, 0.5)
    b = 5.33 * np.power(Nb, 0.13)
    N1 = a/b

    N2 = 7 * asig_w * np.power(Nb, 0.362)/200

    a = 7 * (asig_a + M/100)
    Q = np.power(a, 0.6) * N1 - N2
    
    return Q


def Q_wb(Q, Qb):
    return 20 * (np.power(Q, 0.7) - np.power(Qb, 0.7))


def lightness(Y_b, Y_w, Q, Q_w):
    Z = 1 + np.power(Y_b/Y_w, 0.5)
    return 100 * np.power(Q/Q_w, Z)
    

# values for Nc, Nb
# small area, uniform surround = 1.0, 300
# normal scenes = 1.0, 75
# television, dim surround = 1.0, 25
# transparencies, light box = 0.7, 25
# transparencies, dark surround = 0.7, 10
def hunt_brightness(XYZ_stim, XYZ_illum, La, XYZ_adapt, XYZ_bkgd, Nc, Nb, discount, helson_judd):
    """
    Calculates the brightness of a stimulus based on the Hunt model.

    Args:
    - XYZ_stim (tuple): tuple containing the tristimulus values of the stimulus in XYZ color space.
    - XYZ_illum (tuple): tuple containing the tristimulus values of the illuminant in XYZ color space.
    - La (float): adaptation luminance level in cd/m^2.
    - XYZ_adapt (tuple): tuple containing the tristimulus values of the adapting field in XYZ color space.
    - XYZ_bkgd (tuple): tuple containing the tristimulus values of the background in XYZ color space.
    - Nc (float): surround luminance level in cd/m^2.
    - Nb (float): background luminance level in cd/m^2.
    - discount (float): discounting factor.
    - helson_judd (bool): whether to use Helson-Judd effect or not.

    Returns:
    - dict: a dictionary containing the following keys:
        - 'Q': the brightness of the stimulus.
        - 'WB': the brightness of the stimulus with respect to the background.
        - 'J': the lightness of the stimulus.
    """
    xy_illum = XYZ2xy(XYZ_illum)

    xyz = XYZ_stim
    Y_stim = XYZ_stim[1]

    xyz_w = XYZ_illum
    Y_white = XYZ_illum[1]

    xyz_bkg = XYZ_bkgd
    Y_b = XYZ_bkgd[1]

    xyz_prox = XYZ_bkgd
    Y_p = xyz_prox[1]

    T = colour.temperature.xy_to_CCT(xy_illum, "Hernandez 1999")

    Las = 2.26*La*np.power(T/4000 - 0.4, 1/3)
    k = 1/(5*La + 1)
    Fl = 0.2 * np.power(k, 4) * (5*La) + 0.1*np.power(1 - np.power(k, 4), 2) * np.power(5*La, 1/3)

    Ncb = 0.725*np.power(Y_white/Y_b, 0.2)
    Nbb = 0.725*np.power(Y_white/Y_b, 0.2)

    # white (cones)
    pgb_w = xyz_to_cone(xyz_w)

    # proximal
    pgb_p = xyz_to_cone(xyz_prox)

    # stimulus
    pgb = xyz_to_cone(xyz)
    pgb_a = adapt_cones(pgb, pgb_w, pgb_p, La, Fl, Y_b, Y_white, discount, helson_judd)
    Aa = achrom_cone(pgb_a)
    Ca = C(pgb_a)
    hue, es = H(h_s(Ca))
    M_YBa = M_YB(Ca, La, Nc, Ncb, es)
    M_RGa = M_RG(Ca, Nc, Ncb, es)
    Ma = M(M_YBa, M_RGa)
    sa = s(Ma, pgb_a)
    S_S_w = Y_stim/Y_white

    asig_a = achrom_signal(Las, Aa, Nbb, S_S_w)

    # white
    pgb_aw = adapt_cones(pgb_w, pgb_w, pgb_p, La, Fl, Y_b, Y_white, discount, helson_judd)
    Aw = achrom_cone(pgb_aw)
    Caw = C(pgb_aw)
    # hue, es = H(h_s(Caw))
    M_YBaw = M_YB(Caw, La, Nc, Ncb, es)
    M_RGaw = M_RG(Caw, Nc, Ncb, es)
    Maw = M(M_YBaw, M_RGaw)
    S_w_S_w = Y_white/Y_white

    asig_w = achrom_signal(Las, Aw, Nbb, S_w_S_w)

    Qa = brightness(asig_a, asig_w, Nb, Ma)
    Qw = brightness(asig_w, asig_w, Nb, Maw)

    # background
    pgb_bkg = xyz_to_cone(xyz_bkg)
    pgb_bkg_a = adapt_cones(pgb_bkg, pgb_w, pgb_p, La, Fl, Y_b, Y_white, discount, helson_judd)
    Aa_bkg = achrom_cone(pgb_bkg_a)
    Ca_bkg = C(pgb_bkg_a)
    hue_bkg, es_bkg = H(h_s(Ca_bkg))
    M_YBa_bkg = M_YB(Ca_bkg, La, Nc, Ncb, es_bkg)
    M_RGa_bkg = M_RG(Ca_bkg, Nc, Ncb, es_bkg)
    Ma_bkg = M(M_YBa_bkg, M_RGa_bkg)
    sa_bkg = s(Ma_bkg, pgb_bkg_a)

    S_bkg_S_w = Y_b/Y_white
    
    asig_bkg = achrom_signal(Las, Aa_bkg, Nbb, S_bkg_S_w)
    Qa_bkg = brightness(asig_bkg, asig_w, Nb, Ma_bkg)

    Q_wba = Q_wb(Qa, Qa_bkg)

    J = lightness(Y_b, Y_white, Qa, Qw)

    return {
        'Q': Qa,
        'WB': Q_wba,
        'J': J
    }