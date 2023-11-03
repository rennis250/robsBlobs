import numpy as np

def sumRGB(rgb):
    return np.sum(rgb)


def sumWeightRGB(rgbs, wR, wG, wB):
    w_rgbs = np.array([[wR], [wG], [wB]]) * rgbs
    return np.array(np.sum(w_rgbs, axis=0)).flatten()


def sumRGB_Shuchen(rgb):
    ws = np.array([0.37, 0.45, 0.18])
    return np.sum(ws * rgb)


rand_weights = np.random.rand(1, 3)
def sumRGB_Random(rgb):
    return np.sum(rand_weights * rgb)