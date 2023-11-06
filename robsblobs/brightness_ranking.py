import numpy as np

from .monitor import Monitor

def rank_lum(mon: Monitor, rgbs):
    """
    Ranks the brightness of RGB colors based on the luminance values of the monitor.

    Args:
        mon (Monitor): The monitor object containing the luminance values.
        rgbs (numpy.ndarray): An array of RGB colors to be ranked.

    Returns:
        numpy.ndarray: An array of indices that sorts the RGB colors from highest to lowest brightness.
    """
    rs = rgbs[:, 0].squeeze()
    gs = rgbs[:, 1].squeeze()
    bs = rgbs[:, 2].squeeze()

    rY = mon.monxyY[0, 2]
    gY = mon.monxyY[1, 2]
    bY = mon.monxyY[2, 2]

    lums = rs*rY + gs*gY + bs*bY
    rnks = np.argsort(lums) # sorts from lowest to highest
    return np.flip(rnks)


def rank_rad(mon: Monitor, rgbs):
    """
    Ranks the brightness of RGB values based on the maximum spectrum of the given monitor.

    Args:
        mon (Monitor): The monitor object containing the maximum spectrum values.
        rgbs (numpy.ndarray): An array of RGB values.

    Returns:
        numpy.ndarray: An array of indices that correspond to the brightness ranking of the RGB values.
    """
    rs = rgbs[:, 0].squeeze()
    gs = rgbs[:, 1].squeeze()
    bs = rgbs[:, 2].squeeze()

    rspec = mon.R_max_spectrum
    gspec = mon.G_max_spectrum
    bspec = mon.B_max_spectrum

    rads = np.zeros((len(rs), ))
    for c in range(len(rs)):
        r_rad = (rs[c]*rspec).sum()
        g_rad = (gs[c]*gspec).sum()
        b_rad = (bs[c]*bspec).sum()

        rads[c] = r_rad + g_rad + b_rad

    rnks = np.argsort(rads) # sorts from lowest to highest
    return np.flip(rnks)


def rank_sumrgb(rgbs):
    """
    Ranks the brightness of RGB values based on the sum of their components.

    Args:
        rgbs (numpy.ndarray): An array of RGB values, where each row represents a single RGB value.

    Returns:
        numpy.ndarray: An array of indices that represent the ranking of the RGB values based on their brightness.
    """
    rs = rgbs[:, 0].squeeze()
    gs = rgbs[:, 1].squeeze()
    bs = rgbs[:, 2].squeeze()

    ms = np.zeros((len(rs), ))
    for c in range(len(rs)):
        ms[c] = rs[c] + gs[c] + bs[c]

    rnks = np.argsort(ms) # sorts from lowest to highest
    return np.flip(rnks)


def rank_maxrgb(rgbs):
    """
    Ranks the RGB values based on their maximum value.

    Args:
        rgbs (numpy.ndarray): An array of RGB values.

    Returns:
        numpy.ndarray: An array of indices that correspond to the sorted RGB values in descending order.
    """
    rs = rgbs[:, 0].squeeze()
    gs = rgbs[:, 1].squeeze()
    bs = rgbs[:, 2].squeeze()

    ms = np.zeros((len(rs), ))
    for c in range(len(rs)):
        ms[c] = np.max(np.array([rs[c], gs[c], bs[c]]))

    rnks = np.argsort(ms) # sorts from lowest to highest
    return np.flip(rnks)


def normalised_kendall_tau_distance(values1, values2):
    """
    Compute the normalised Kendall tau distance between two lists of values.

    Args:
        values1 (list): First list of values.
        values2 (list): Second list of values.

    Returns:
        float: Normalised Kendall tau distance between the two lists of values.
    """
    n = len(values1)
    assert len(values2) == n, "Both lists have to be of equal length"
    i, j = np.meshgrid(np.arange(n), np.arange(n))
    a = np.argsort(values1)
    b = np.argsort(values2)
    ndisordered = np.logical_or(np.logical_and(a[i] < a[j], b[i] > b[j]), np.logical_and(a[i] > a[j], b[i] < b[j])).sum()
    return ndisordered / (n * (n - 1) / 2.0) # corrected based on wikipedia article and c code


def accuracyShuchen(ranks1, ranks2):
    """
    Computes the accuracy of the ranking given two sets of ranks.

    Parameters:
    ranks1 (list): A list of ranks.
    ranks2 (list): A list of ranks.

    Returns:
    float: The accuracy of the ranking.
    """
    nr1 = len(ranks1)
    pairRanks1 = np.zeros((nr1, nr1))
    for c1 in range(nr1):
        for cc1 in range(nr1):
            rl = ranks1[c1]
            rr = ranks1[cc1]
            if rl > rr:
                pairRanks1[c1, cc1] = 1
            else:
                pairRanks1[c1, cc1] = 0
    
    nr2 = len(ranks2)
    pairRanks2 = np.zeros((nr2, nr2))
    for c1 in range(nr2):
        for cc1 in range(nr2):
            rl = ranks2[c1]
            rr = ranks2[cc1]
            if rl > rr:
                pairRanks2[c1, cc1] = 1
            else:
                pairRanks2[c1, cc1] = 0

    acc = (pairRanks1 == pairRanks2).flatten().sum()

    return acc/(nr1*nr1)

from .maxRGB_revolution import maxRGB, maxRGB_Shuchen, maxRGB_G0
from .sumrgb import sumRGB_Shuchen, sumRGB
from .ware_cowan import ware_cowan
from .fairchildL import rgb2FairchildLstar
from .funny_dude_discovers_maxRGB_in_adobe import RGBtoHSP
from .G0_brilliance import brilliance, brilliance_macadam
from .radiance import radiance, visibleRadiance
from .cie_monitor_helpers import rgb2xyz, luminance, xyY2rgb
from .cie_standard import XYZ2xyY, XYZ2xy
from .lms import rgb2lms, maxLMS
from .infamous_lab import rgb2lab, chroma
from .guth_vector import guth_lum_rgb
from .jazzy import jaz_lightness, jazzy
from .nayatani_insanity import robNT
from .hunt import hunt_brightness
from .svf import svf
from .jingHSP import jingHSP

import colour

def computeAllBrightnessModels(mon: Monitor, rgbs):
    """
    Computes various brightness models for a given set of RGB values.

    Args:
    - mon: Monitor object containing information about the monitor used to display the RGB values.
    - rgbs: numpy array of shape (n, 3) containing n RGB values.

    Returns:
    - Dictionary containing the computed brightness values for each model.
    """
    d65_xy = XYZ2xy(mon.monWP)

    ncs = len(rgbs)

    brightnesses = {
        'maxRGB': np.zeros((ncs)),
        'maxRGB_Shuchen': np.zeros((ncs)),
        'maxRGB_G0': np.zeros((ncs)),
        'sumRGB': np.zeros((ncs)),
        'sumRGB_Shuchen': np.zeros((ncs)),
        'ware_cowan': np.zeros((ncs)),
        'Lstar': np.zeros((ncs)),
        'fairchildL': np.zeros((ncs)),
        'HSP': np.zeros((ncs)),
        'HSP_JingWeights': np.zeros((ncs)),
        'maxLMS': np.zeros((ncs)),
        'brilliance': np.zeros((ncs)),
        'brilliance_desat': np.zeros((ncs)),
        'brill_macadam': np.zeros((ncs)),
        'luminance': np.zeros((ncs)),
        'radiance': np.zeros((ncs)),
        'chroma': np.zeros((ncs)),
        'visibleRadiance': np.zeros((ncs)),
        'Nayatani Br': np.zeros((ncs)),
        'Nayatani Lp': np.zeros((ncs)),
        'Nayatani Ln': np.zeros((ncs)),
        'CAM16 Avg Q': np.zeros((ncs)),
        'CAM16 Avg J': np.zeros((ncs)),
        'CAM16 Dim Q': np.zeros((ncs)),
        'CAM16 Dim J': np.zeros((ncs)),
        'CAM16 Dark Q': np.zeros((ncs)),
        'CAM16 Dark J': np.zeros((ncs)),
        'Guth': np.zeros((ncs)),
        'JzAzBz': np.zeros((ncs)),
        'Hunt_Q Tele': np.zeros((ncs)),
        'Hunt_WB Tele': np.zeros((ncs)),
        'Hunt_Q Norm': np.zeros((ncs)),
        'Hunt_WB Norm': np.zeros((ncs)),
        'Hunt_Q Small': np.zeros((ncs)),
        'Hunt_WB Small': np.zeros((ncs)),
        'JingB': np.zeros((ncs)),
        'ATD': np.zeros((ncs)),
        'Hellwig Avg Q': np.zeros((ncs)),
        'Hellwig Avg J': np.zeros((ncs)),
        'Hellwig Dim Q': np.zeros((ncs)),
        'Hellwig Dim J': np.zeros((ncs)),
        'Hellwig Dark Q': np.zeros((ncs)),
        'Hellwig Dark J': np.zeros((ncs)),
        'SVF': np.zeros((ncs))
    }

    for c in range(ncs):
        rgb = rgbs[c, :].squeeze()

        r = rgb[0]
        g = rgb[1]
        b = rgb[2]

        xyz = rgb2xyz(mon, rgb)
        xyY = XYZ2xyY(xyz)

        xy_desat = 0.4*xyY[0:2] + 0.6*d65_xy

        lms = rgb2lms(mon, rgb)

        lab = rgb2lab(mon, rgb)
        brightnesses['Lstar'][c] = lab[0]

        brightnesses['maxRGB'][c] = maxRGB(rgb)
        brightnesses['maxRGB_Shuchen'][c] = maxRGB_Shuchen(rgb)
        brightnesses['maxRGB_G0'][c] = maxRGB_G0(rgb)

        brightnesses['sumRGB'][c] = sumRGB(rgb)
        brightnesses['sumRGB_Shuchen'][c] = sumRGB_Shuchen(rgb)

        brightnesses['ware_cowan'][c] = ware_cowan(xyY)
        brightnesses['fairchildL'][c] = rgb2FairchildLstar(mon, rgb)

        hsp = RGBtoHSP(r, g, b, "photoshop")
        brightnesses['HSP'][c] = hsp[2]

        hsp = RGBtoHSP(r, g, b, "jing")
        brightnesses['HSP_JingWeights'][c] = hsp[2]

        brightnesses['JingB'][c] = jingHSP(rgb)

        brightnesses['maxLMS'][c] = maxLMS(lms)

        brightnesses['brilliance'][c] = brilliance(mon, rgb)

        xyY_desat = np.array([xy_desat[0], xy_desat[1], xyY[2]])
        rgb_desat = xyY2rgb(mon, xyY_desat)
        brightnesses['brilliance_desat'][c] = brilliance(mon, rgb_desat)

        brightnesses['brill_macadam'][c] = brilliance_macadam(mon, rgb)

        brightnesses['luminance'][c] = luminance(mon, rgb)
        brightnesses['radiance'][c] = radiance(mon, rgb)
        brightnesses['chroma'][c] = chroma(mon, rgb)

        brightnesses['visibleRadiance'][c] = visibleRadiance(mon, rgb)

        brightnesses['Guth'][c] = guth_lum_rgb(mon, rgb)

        jaz = jazzy(mon, rgb)
        brightnesses['JzAzBz'][c] = jaz_lightness(jaz)

        Eo = mon.monWP[1]
        Eor = 1000
        Y_stim = xyY[2]/Eo
        Yo_bg = 0 # we always had a black background in all experiments
        xy_illum = XYZ2xy(mon.monWP)
        xy_stim = xyY[0:2]
        nt_res = robNT(xy_stim, Y_stim, Yo_bg, xy_illum, Eo, Eor)
        brightnesses['Nayatani Br'][c] = nt_res['Brightness']
        brightnesses['Nayatani Lp'][c] = nt_res['Lightness']
        brightnesses['Nayatani Ln'][c] = nt_res['Normalized Lightness']

        XYZ_w = mon.monWP
        L_A = 0.2*mon.monWP[1]
        Y_b = 0.001
        surround = colour.appearance.VIEWING_CONDITIONS_CAM16["Average"]
        cam16 = colour.appearance.XYZ_to_CAM16(xyz, XYZ_w, L_A, Y_b, surround) 
        brightnesses['CAM16 Avg Q'][c] = cam16.Q
        brightnesses['CAM16 Avg J'][c] = cam16.J

        surround = colour.appearance.VIEWING_CONDITIONS_CAM16["Dim"]
        cam16 = colour.appearance.XYZ_to_CAM16(xyz, XYZ_w, L_A, Y_b, surround) 
        brightnesses['CAM16 Dim Q'][c] = cam16.Q
        brightnesses['CAM16 Dim J'][c] = cam16.J

        surround = colour.appearance.VIEWING_CONDITIONS_CAM16["Dark"]
        cam16 = colour.appearance.XYZ_to_CAM16(xyz, XYZ_w, L_A, Y_b, surround) 
        brightnesses['CAM16 Dark Q'][c] = cam16.Q
        brightnesses['CAM16 Dark J'][c] = cam16.J

        # values for Nc, Nb
        # small area, uniform surround = 1.0, 300
        # normal scenes = 1.0, 75
        # television, dim surround = 1.0, 25
        # transparencies, light box = 0.7, 25
        # transparencies, dark surround = 0.7, 10
        xy_adapt = np.array([0.33, 0.33])
        Yref_adapt = 0.001
        xy_bkgd = np.array([0.33, 0.33])
        Yref_bkgd = 0.001
        Nc = 1.0
        Nb = 25
        hunt_res = hunt_brightness(xy_stim, Y_stim, xy_illum, Eo, xy_adapt, Yref_adapt, xy_bkgd, Yref_bkgd, Nc, Nb)
        brightnesses['Hunt_Q Tele'] = hunt_res['Q']
        brightnesses['Hunt_WB Tele'] = hunt_res['WB']

        Nc = 1.0
        Nb = 75
        hunt_res = hunt_brightness(xy_stim, Y_stim, xy_illum, Eo, xy_adapt, Yref_adapt, xy_bkgd, Yref_bkgd, Nc, Nb)
        brightnesses['Hunt_Q Norm'] = hunt_res['Q']
        brightnesses['Hunt_WB Norm'] = hunt_res['WB']

        Nc = 1.0
        Nb = 300
        hunt_res = hunt_brightness(xy_stim, Y_stim, xy_illum, Eo, xy_adapt, Yref_adapt, xy_bkgd, Yref_bkgd, Nc, Nb)
        brightnesses['Hunt_Q Small'] = hunt_res['Q']
        brightnesses['Hunt_WB Small'] = hunt_res['WB']

        # we were viewing in related colors mode
        atd_k_1 = 1.0
        atd_k_2 = 5.0
        atd_res = colour.appearance.atd95.XYZ_to_ATD95(xyz, mon.monWP, Yref_adapt, atd_k_1, atd_k_2)
        brightnesses['ATD'][c] = atd_res.Q

        surr = colour.appearance.hellwig2022.VIEWING_CONDITIONS_HELLWIG2022["Dim"]
        hf_res = colour.appearance.XYZ_to_Hellwig2022(xyz, mon.monWP, Yref_adapt, Yref_bkgd, surr)
        brightnesses['Hellwig Dim Q'][c] = hf_res.Q
        brightnesses['Hellwig Dim J'][c] = hf_res.J

        surr = colour.appearance.hellwig2022.VIEWING_CONDITIONS_HELLWIG2022["Average"]
        hf_res = colour.appearance.XYZ_to_Hellwig2022(xyz, mon.monWP, Yref_adapt, Yref_bkgd, surr)
        brightnesses['Hellwig Avg Q'][c] = hf_res.Q
        brightnesses['Hellwig Avg J'][c] = hf_res.J

        surr = colour.appearance.hellwig2022.VIEWING_CONDITIONS_HELLWIG2022["Dark"]
        hf_res = colour.appearance.XYZ_to_Hellwig2022(xyz, mon.monWP, Yref_adapt, Yref_bkgd, surr)
        brightnesses['Hellwig Dark Q'][c] = hf_res.Q
        brightnesses['Hellwig Dark J'][c] = hf_res.J

        svf_res = svf(xyz, mon.monWP)
        brightnesses['SVF'][c] = svf_res[2]

    return brightnesses