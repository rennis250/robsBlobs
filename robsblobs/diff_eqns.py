import numpy as np

def ciede2000(lab1, lab2, kl=1, kc=1, kh=1):
    l1 = lab1[0]
    a1 = lab1[1]
    b1 = lab1[2]

    l2 = lab2[0]
    a2 = lab2[1]
    b2 = lab2[2]

    c1 = np.sqrt(a1**2 + b1**2)
    c2 = np.sqrt(a2**2 + b2**2)

    cbar = (c1 + c2)/2

    g = 0.5 * (1 - np.sqrt(cbar**7/(cbar**7 + 25**7)))

    ap1 = (1 + g)*a1
    ap2 = (1 + g)*a2

    cp1 = np.sqrt(ap1**2 + b1**2)
    cp2 = np.sqrt(ap2**2 + b2**2)

    hp1 = np.arctan2(b1, ap1)
    if hp1 < 0:
        hp1 += 2*np.pi

    hp2 = np.arctan2(b2, ap2)
    if hp2 < 0:
        hp2 += 2*np.pi

    dlp = l2 - l1
    dcp = cp2 - cp1
    dhp = hp2 - hp1

    if dhp > np.pi:
        dhp -= 2*np.pi
    elif dhp < -np.pi:
        dhp += 2*np.pi

    dHp = 2*np.sqrt(cp1*cp2)*np.sin(dhp/2)

    lbar = (l1 + l2)/2
    cpbar = (cp1 + cp2)/2

    hpbar = (hp1 + hp2)/2
    if np.abs(hp1 - hp2) > np.pi:
        hpbar -= np.pi

    if hpbar < 0:
        hpbar += 2*np.pi

    ta = 0.17*np.cos(hpbar - np.pi/6)
    tb = 0.24*np.cos(2*hpbar)
    tc = 0.32*np.cos(3*hpbar + np.pi/30)
    td = 0.2*np.cos(4*hpbar - 63*np.pi/180)
    t = 1 - ta + tb + tc - td

    sl = 1 + (0.015*(lbar - 50)**2)/np.sqrt(20 + (lbar - 50)**2)
    sc = 1 + 0.045*cpbar
    sh = 1 + 0.015*cpbar*t

    dtha = (180/np.pi*hpbar - 275)/25
    dth = (30*np.pi/180)*np.exp(-(dtha**2))

    rc = 2*np.sqrt(cpbar**7/(cpbar**7 + 25**7))
    rt = -np.sin(2*dth)*rc

    klSl = kl*sl
    kcSc = kc*sc
    khSh = kh*sh

    de00 = np.sqrt((dlp/klSl)**2 + (dcp/kcSc)**2 + (dHp/khSh)**2 + rt*(dcp/kcSc)*(dHp/khSh))

    return de00


def uv_euclid(luv1, luv2):
    l1 = luv1[0]
    u1 = luv1[1]
    v1 = luv1[2]

    l2 = luv2[0]
    u2 = luv2[1]
    v2 = luv2[2]

    return np.sqrt((l1 - l2)**2 + (u1 - u2)**2 + (v1 - v2)**2)


def srgb_euclid(rgb1, rgb2):
    r1 = rgb1[0]
    g1 = rgb1[1]
    b1 = rgb1[2]

    r2 = rgb2[0]
    g2 = rgb2[1]
    b2 = rgb2[2]

    return np.sqrt((r2 - r1)**2 + (g2 - g1)**2 + (b2 - b1)**2)


def srgb_weighted_euclid(rgb1, rgb2):
    r1 = rgb1[0]
    g1 = rgb1[1]
    b1 = rgb1[2]

    r2 = rgb2[0]
    g2 = rgb2[1]
    b2 = rgb2[2]

    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2

    return np.sqrt(2*dr**2 + 4*dg**2 + 2*db**2)


def srgb_nonlin_cube_dist(rgb1, rgb2):
    r1 = rgb1[0]
    g1 = rgb1[1]
    b1 = rgb1[2]

    r2 = rgb2[0]
    g2 = rgb2[1]
    b2 = rgb2[2]

    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2

    rm = 0.5*(r1 + r2)

    if rm < 128:
        return np.sqrt(2*dr**2 + 4*dg**2 + 3*db**2)
    else:
        return np.sqrt(3*dr**2 + 4*dg**2 + 2*db**2)
    

def srgb_nonlin_redmean(rgb1, rgb2):
    r1 = rgb1[0]
    g1 = rgb1[1]
    b1 = rgb1[2]

    r2 = rgb2[0]
    g2 = rgb2[1]
    b2 = rgb2[2]

    dr = r1 - r2
    dg = g1 - g2
    db = b1 - b2

    rm = 0.5*(r1 + r2)

    a = (2 + rm/256)*dr**2
    b = 4*dg**2
    c = (2 + (255 - rm)/256)*db**2

    return np.sqrt(a + b + c)


def srgb_nonlin_redmean_compuphase(rgb1, rgb2):
    r1 = rgb1[0]
    g1 = rgb1[1]
    b1 = rgb1[2]

    r2 = rgb2[0]
    g2 = rgb2[1]
    b2 = rgb2[2]

    rmean = (r1 + r2)/2

    r = r1 - r2
    g = g1 - g2
    b = b1 - b2

    a = ((512+rmean)*r*r)>>8
    b = 4*g*g
    c = ((767-rmean)*b*b)>>8

    return np.sqrt(a + b + c)