import numpy as np
from sklearn.cluster import KMeans

import pycircstat as cstat
import astropy.stats.circstats as circstats

def max_sat(labs):
    """
    Calculates the maximum saturation of a given set of LAB colors.

    Parameters:
    labs (numpy.ndarray): A numpy array of shape (n, 3) containing n LAB colors.

    Returns:
    numpy.ndarray: A numpy array of shape (2,) containing the mean a* and b* values of the most saturated colors.
    """
    ls = labs[:, 0]
    As = labs[:, 1]
    bs = labs[:, 2]

    chromas = np.sqrt(np.power(As, 2) + np.power(bs, 2))
    sats = chromas/ls

    idxs = np.argsort(sats)
    five_perc_idxs = idxs[int(len(idxs)*0.95):]

    hs = np.array([bs[five_perc_idxs], As[five_perc_idxs]])
    hs[hs < 0] += 2*np.pi
    msat = cstat.mean(hs)

    return msat


def max_sat_hue(labs):
    """
    Calculates the maximum saturation hue of the input LAB color space image.

    Parameters:
    labs (numpy.ndarray): A numpy array of shape (n, 3) containing the LAB color space image.

    Returns:
    float: The maximum saturation hue of the input image.
    """
    ls = labs[:, 0]
    As = labs[:, 1]
    bs = labs[:, 2]

    chromas = np.sqrt(np.power(As, 2) + np.power(bs, 2))
    sats = chromas/ls

    idxs = np.argsort(sats)
    five_perc_idxs = idxs[int(len(idxs)*0.95):]

    hues = np.arctan2(bs[five_perc_idxs], As[five_perc_idxs])
    hues[hues < 0] += 2*np.pi

    msat = cstat.mean(hues)
    return msat


def most_lum(labs):
    """
    Calculates the most luminous color in the given LAB color space.

    Parameters:
    labs (numpy.ndarray): An array of LAB color values.

    Returns:
    numpy.ndarray: The most luminous color in the given LAB color space.
    """
    ls = labs[:, 0]
    As = labs[:, 1]
    bs = labs[:, 2]

    idxs = np.argsort(ls)
    five_perc_idxs = idxs[int(len(idxs)*0.95):]

    hs = np.array([bs[five_perc_idxs], As[five_perc_idxs]])
    hs[hs < 0] += 2*np.pi
    mlum = cstat.mean(hs)

    return mlum


def mean_col(labs):
    """
    Calculates the mean color of an image in LAB color space.

    Parameters:
    labs (numpy.ndarray): A numpy array of shape (n, 3) containing the LAB color values of an image.

    Returns:
    numpy.ndarray: A numpy array of shape (3,) containing the mean LAB color values of the image.
    """
    ls = labs[:, 0]
    As = labs[:, 1]
    bs = labs[:, 2]

    mcol = np.array([np.mean(ls), np.mean(As), np.mean(bs)])
    return mcol


def mean_hue(labs):
    """
    Calculates the mean hue of an array of LAB color values.

    Parameters:
    labs (numpy.ndarray): An array of LAB color values.

    Returns:
    float: The mean hue value in radians.
    """
    As = labs[:, 1]
    bs = labs[:, 2]

    hues = np.arctan2(bs, As)
    hues[hues < 0] += 2*np.pi

    return cstat.mean(hues)


def max_cov(labs):
    ls = labs[:, 0]
    As = labs[:, 1]
    bs = labs[:, 2]

    hues = np.arctan2(bs, As)
    hues[hues < 0] += 2*np.pi

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(hues.reshape(-1, 1))
    cs = kmeans.labels_

    clust1 = np.where(cs == 0)[0]
    clust1_size = len(clust1)
    clust2 = np.where(cs == 1)[0]
    clust2_size = len(clust2)

    if clust1_size > clust2_size:
        lm = np.mean(ls[clust1])
        Am = np.mean(As[clust1])
        bm = np.mean(bs[clust1])

        return np.array([lm, Am, bm])
    else:
        lm = np.mean(ls[clust2])
        Am = np.mean(As[clust2])
        bm = np.mean(bs[clust2])

        return np.array([lm, Am, bm])


def max_cov_pixels(labs):
    ls = labs[:, 0]
    As = labs[:, 1]
    bs = labs[:, 2]

    hues = np.arctan2(bs, As)
    hues[hues < 0] += 2*np.pi

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(hues.reshape(-1, 1))
    cs = kmeans.labels_

    clust1 = np.where(cs == 0)[0]
    clust1_size = len(clust1)
    clust2 = np.where(cs == 1)[0]
    clust2_size = len(clust2)

    if clust1_size > clust2_size:
        return np.array([ls[clust1], As[clust1], bs[clust1]]), cs
    else:
        return np.array([ls[clust2], As[clust2], bs[clust2]]), cs


def most_freq(labs):
    """
    Finds the most frequently occurring color in a given array of colors.

    Parameters:
    labs (numpy.ndarray): An array of colors in the format of (R, G, B).

    Returns:
    numpy.ndarray: An array representing the most frequently occurring color in the input array.
    """
    xs = labs[:, 0]
    ys = labs[:, 1]
    zs = labs[:, 2]

    xmin = np.min(xs) - 0.02
    ymin = np.min(ys) - 0.02
    zmin = np.min(zs) - 0.02

    xmax = np.max(xs) + 0.02
    ymax = np.max(ys) + 0.02
    zmax = np.max(zs) + 0.02

    x_sps = np.linspace(xmin, xmax, 10)
    y_sps = np.linspace(ymin, ymax, 10)
    z_sps = np.linspace(zmin, zmax, 10)

    step_size = 10

    max_freq = np.NINF
    xc_mfreq = 1
    yc_mfreq = 1
    zc_mfreq = 1

    for xc in range(len(x_sps)):
        x = x_sps[xc] + step_size/2
        for yc in range(len(y_sps)):
            y = y_sps[yc] + step_size/2
            for zc in range(len(z_sps)):
                z = z_sps[zc] + step_size/2
            
                cond1 = np.logical_and(xs > (x - step_size/2), xs < (x + step_size/2))
                cond2 = np.logical_and(ys > (y - step_size/2), ys < (y + step_size/2)) 
                cond3 = np.logical_and(zs > (z - step_size/2), zs < (z + step_size/2))
                idxs = np.logical_and(cond1, np.logical_and(cond2, cond3))
            
                ncolors = len(np.where(idxs)[0])
                if ncolors > max_freq:
                    max_freq = ncolors
                    xc_mfreq_coarse = xc
                    yc_mfreq_coarse = yc
                    zc_mfreq_coarse = zc

    xmin = xs[xc_mfreq_coarse] - step_size/2
    ymin = ys[yc_mfreq_coarse] - step_size/2
    zmin = zs[zc_mfreq_coarse] - step_size/2

    xmax = xs[xc_mfreq_coarse] + step_size/2
    ymax = ys[yc_mfreq_coarse] + step_size/2
    zmax = zs[zc_mfreq_coarse] + step_size/2

    x_sps = np.arange(xmin, xmax, 1)
    y_sps = np.arange(ymin, ymax, 1)
    z_sps = np.arange(zmin, zmax, 1)

    step_size = 1

    max_freq = np.NINF
    xc_mfreq = 1
    yc_mfreq = 1
    zc_mfreq = 1

    for xc in range(len(x_sps)):
        x = x_sps[xc] + step_size/2
        for yc in range(len(y_sps)):
            y = y_sps[yc] + step_size/2
            for zc in range(len(z_sps)):
                z = z_sps[zc] + step_size/2
            
                cond1 = np.logical_and(xs > (x - step_size/2), xs < (x + step_size/2))
                cond2 = np.logical_and(ys > (y - step_size/2), ys < (y + step_size/2))
                cond3 = np.logical_and(zs > (z - step_size/2), zs < (z + step_size/2))
                idxs = np.logical_and(cond1, np.logical_and(cond2, cond3))

                ncolors = len(np.where(idxs)[0])
                if ncolors > max_freq:
                    max_freq = ncolors
                    xc_mfreq = xc
                    yc_mfreq = yc
                    zc_mfreq = zc

    mfreq_col = [xs[xc_mfreq] + step_size/2, ys[yc_mfreq] + step_size/2, zs[zc_mfreq] + step_size/2]
    return np.array(mfreq_col)


# written for CIELAB
def classify_color(lab, color):
    l = lab[0]
    a = lab[1]
    b = lab[2]

    h = np.arctan2(b, a)
    if h < 0:
        h += 2 * np.pi

    hd = np.rad2deg(h)
    if color == 1:
        # 'Purple Center, Green Edge'
        if hd >= 80 and hd < 275:
            return "purple", "center"
        else:
            return "green", "edge"
    
    elif color == 2:
        # 'Pink Center, Orange Edge'
        # if a < 12 and b > 10:
        if hd >= 62:
            return "pink", "center"
        else:
            return "orange", "edge"

    elif color == 3:
        # 'Green Center, Purple Edge'
        if hd >= 80 and hd < 330:
            return "purple", "edge"
        else:
            return "green", "center"


def iri_extract_edge_center(labs, color):
    """
    Extracts the edge and center colors from an image in LAB color space.

    Args:
        labs (numpy.ndarray): A numpy array of shape (n, 3) containing the LAB color values of an image.
        color (str): The color to extract, either "red", "green", or "blue".

    Returns:
        dict: A dictionary containing the center and edge colors of the specified color, along with their indices.
            The dictionary has the following structure:
            {
                'cent': {
                    'lab': numpy.ndarray,
                    'idx': list,
                },
                'edge': {
                    'lab': numpy.ndarray,
                    'idx': list,
                }
            }
    """
    ls = labs[:, 0]
    As = labs[:, 1]
    bs = labs[:, 2]

    edge_center = {
        'cent': {
            'lab': [],
            'idx': [],
        },
        'edge': {
            'lab': [],
            'idx': [],
        }
    }

    for c in range(len(labs)):
        cat, cent_edge = classify_color(labs[c, :].squeeze(), color)
        if cent_edge == "center":
            edge_center['cent']['lab'].append([ls[c], As[c], bs[c]])
            edge_center['cent']['idx'].append(c)
        else:
            edge_center['edge']['lab'].append([ls[c], As[c], bs[c]])
            edge_center['edge']['idx'].append(c)

    edge_center['cent']['lab'] = np.array(edge_center['cent']['lab'])
    edge_center['edge']['lab'] = np.array(edge_center['edge']['lab'])

    return edge_center


def yo_cent_edge_colors(img, mask, color):
    cent_yx = np.array([img.shape[0] / 2, img.shape[1] / 2])

    labs = np.zeros(img.shape)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if mask[y, x]:
                labs[y, x, :] = robsblobs.infamous_lab.rgb2lab(mon, img[y, x, :])

    cent_map = np.zeros(img.shape[:2], dtype=bool)
    edge_map = np.zeros(img.shape[:2], dtype=bool)

    h2 = img.shape[0]/2
    w2 = img.shape[1]/2
    for theta in np.linspace(0, 2*np.pi, 1000):
        for r in np.arange(0, h2, 0.5):
            y = r*np.sin(theta) + (h2-1)
            x = r*np.cos(theta) + (w2-1)
            if mask[int(y), int(x)]:
                cat, cent_edge = robsblobs.img_stats.classify_color(labs[int(y), int(x), :], color)
                if cent_edge == "center":
                    cent_map[int(y), int(x)] = True
                else:
                    edge_map[int(y), int(x)] = True

    return cent_map, edge_map