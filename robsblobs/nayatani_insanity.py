import numpy as np

from .cie_standard import xyY2XYZ

n = 1

NT_unique_red = 20.14 # degrees
NT_unique_yellow = 90.00 # degrees
NT_unique_green = 164.25 # degrees
NT_unique_blue = 231.00 # degrees

# Hue Quadrature is 400-step hue scale, starting at unique red.
# Interpolate hue based on unique hue angles above to get
# the Hue Quadrature
quad_steps = np.linspace(0, 400)
quad_red = 0
quad_yellow = 100
quad_green = 200
quad_blue = 300

# Hue Composition is what percentage of the arc between two unique
# hues a given color occupies, such as 33Y 67R.
# Interpolate quadrature values to get the percentages for
# Hue Composition.

def eR(R, xi):
    if R >= 20 * xi:
        return 1.758
    else:
        return 1
    

def eG(G, eta):
    if G >= 20 * eta:
        return 1.758
    else:
        return 1


def beta_1(x):
    return (6.496 + 6.362 * np.power(x, 0.4495))/(6.469 + np.power(x, 0.4495))


def beta_2(x):
    return  0.7844 * (8.414 + 8.091 * np.power(x, 0.5128))/(8.414 + np.power(x, 0.5128)) # in Fairchild book


def Es(t):
    return 0.9394 - 0.2478 * np.sin(t) - 0.0743 * np.sin(2*t) + 0.0666 * np.sin(3*t) \
        - 0.0186 * np.sin(4*t) - 0.0055 * np.cos(t) - 0.0521 * np.cos(2*t) \
        - 0.0573 * np.cos(3*t) - 0.0061 * np.cos(4*t)


def robNT(xy_stim, Y_stim, Yo_bg, xy_illum, Eo, Eor):
    """
    Computes the Nayatani et al. (1995) color appearance model.

    Args:
        xy_stim (tuple): Tuple containing the chromaticity coordinates of the stimulus in CIE 1931 xyY color space.
        Y_stim (float): Luminance of the stimulus in cd/m^2.
        Yo_bg (float): Luminance of the background in cd/m^2.
        xy_illum (tuple): Tuple containing the chromaticity coordinates of the illuminant in CIE 1931 xyY color space.
        Eo (float): Illuminance of the stimulus in lux.
        Eor (float): Illuminance of the reference white in lux.

    Returns:
        dict: A dictionary containing the following keys:
            - 'Brightness': Brightness of the stimulus.
            - 'Lightness': Lightness of the stimulus.
            - 'Normalized Lightness': Normalized lightness of the stimulus.
            - 'Hue Angle': Hue angle of the stimulus in degrees.
            - 'Saturation': Saturation of the stimulus.
            - 'Chroma': Chroma of the stimulus.
            - 'Colorfulness': Colorfulness of the stimulus.
            - 'Achromatic Response': Achromatic response of the stimulus.
            - 'RG Response': Red-green response of the stimulus.
            - 'YB Response': Yellow-blue response of the stimulus.
    """
    xo = xy_illum[0]
    yo = xy_illum[1]
    Yo = Yo_bg

    Lo_adapt = (Yo * Eo)/(100 * np.pi)
    Lo_r = (Yo * Eor)/(100 * np.pi)

    xi = (0.48105*xo + 0.78841*yo - 0.08081)/yo
    eta = (-0.272*xo + 1.11962*yo + 0.0457)/yo
    zeta = 0.91822 * (1.0 - xo - yo)/yo

    cone_adapt_resp = Lo_adapt * np.array([xi, eta, zeta])
    R_o = cone_adapt_resp[0]
    G_o = cone_adapt_resp[1]
    B_o = cone_adapt_resp[2]

    xyY_stim = np.array([xy_stim[0], xy_stim[1], Y_stim*Eo])
    XYZ = xyY2XYZ(xyY_stim)
    M = np.array([
        [ 0.40024, 0.7076, -0.08081],
        [-0.2263,  1.16532, 0.0457],
        [0,        0,       0.91822],
    ])
    cone_resp = M @ XYZ
    R = cone_resp[0]
    G = cone_resp[1]
    B = cone_resp[2]

    e_R = eR(R, xi)
    e_G = eG(G, eta)

    b1_Lor = beta_1(Lo_r)
    b1_Ro = beta_1(R_o)
    b1_Go = beta_1(G_o)
    b2_Bo = beta_2(B_o)

    a = 41.69/b1_Lor
    b = (2/3) * b1_Ro * e_R * np.log10((R + n)/(20*xi + n))
    c = (1/3) * b1_Go * e_G * np.log10((G + n)/(20*eta + n))
    Q = a * (b + c)

    d = 50/b1_Lor
    e = (2/3) * b1_Ro
    f = (1/3) * b1_Go
    Br = Q + d*(e + f)

    h = (2/3) * b1_Ro * 1.758 * np.log10((100*xi + n)/(20*xi + n))
    i = (1/3) * b1_Go * 1.758 * np.log10((100*eta + n)/(20*eta + n))
    Brw = a*(h + i) + d*(e + f)

    Lp_star = Q + 50
    Ln_star = 100 * Br/Brw

    ta = b1_Ro * np.log10((R + n)/(20*xi + n))
    tb = (12/11) * b1_Go * np.log10((G + n)/(20*eta + n))
    tc = (1/11) * b2_Bo * np.log10((B + n)/(20*zeta + n))
    t = ta - tb + tc

    pa = (1/9) * b1_Ro * np.log10((R + n)/(20*xi + n))
    pb = (1/9) * b1_Go * np.log10((G + n)/(20*eta + n))
    pc = (2/9) * b2_Bo * np.log10((B + n)/(20*zeta + n))
    p = pa + pb - pc

    theta = np.arctan2(p, t)
    if theta < 0:
        theta += 2*np.pi
    
    thetad = np.rad2deg(theta)

    Es_theta = Es(theta)
    Srg = (488.93 * Es_theta * t)/b1_Lor
    Syb = (488.93 * Es_theta * p)/b1_Lor
    S = np.sqrt(np.power(Srg, 2) + np.power(Syb, 2))

    Lp_scaled = np.power(Lp_star/50, 0.7)
    Crg = Lp_scaled * Srg
    Cyb = Lp_scaled * Syb
    C = Lp_scaled * S

    Mrg = Crg * Brw/100
    Myb = Cyb * Brw/100
    M = C * Brw/100

    NT = {
        'Brightness': Br,
        'Lightness': Lp_star,
        'Normalized Lightness': Ln_star,
        'Hue Angle': thetad,
        'Saturation': S,
        'Chroma': C,
        'Colorfulness': M,
        'Achromatic Response': Q,
        'RG Response': t,
        'YB Response': p,
    }

    return NT