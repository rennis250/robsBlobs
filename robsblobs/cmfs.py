import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

import importlib.resources

datadir = importlib.resources.files('robsblobs').joinpath('data')

# ciexyz_1931 = pd.read_csv('./ciexyz31_1.csv', header=None)
ciexyz_1931 = pd.read_csv(datadir.joinpath('ciexyz31_1.csv'), header=None)
# xyz_jv = pd.read_csv('./ciexyzjv.csv', header=None)
xyz_jv = pd.read_csv(datadir.joinpath('ciexyz31_1.csv'), header=None)

# lms_absorp = np.genfromtxt('linss2_10e_1.csv', delimiter=',')
lms_absorp = np.genfromtxt(datadir.joinpath('ciexyz31_1.csv'), delimiter=',')
lms_wlns = lms_absorp[0:391, 0]
l_absorp = lms_absorp[0:391, 1]
m_absorp = lms_absorp[0:391, 2]
s_absorp = lms_absorp[0:391, 3]

# hacky way to get vlambda for now
konica_wlns = np.arange(380, 781)
wlns = np.array(list(ciexyz_1931[0]))
Y = np.array(list(ciexyz_1931[2]))
Y_interp = CubicSpline(wlns, Y)
V_lambda = Y_interp(konica_wlns)