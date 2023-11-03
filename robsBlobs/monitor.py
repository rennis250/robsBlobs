import numpy as np
import scipy.io as sio
from scipy.interpolate import CubicSpline

from .cie_standard import xyY2XYZ
from .cmfs import lms_wlns, l_absorp, m_absorp, s_absorp
from .dkl_setup import cie2lms, lumchrm, solvex, solvey
# from .uv_space import xyY2uvY

class Monitor:
    def __init__(self, name):
        self.name = name

        self.monxyY_file = ""
        self.spectralData_file = ""

        self.monxyY = np.array([])
        self.monXYZ = np.array([])
        # self.monuvY = np.array([])
        self.monWP = np.array([])
        self.maxLuminance = np.array([])

        # default assumption is 2.2 gamma for all primaries
        self.monGamma = np.array([2.2, 2.2, 2.2])

        # our Konica CS2000-a measures in range of 380 to 780nm
        # in steps of 1nm
        self.wlns = np.arange(380, 781)

        self.spectralData = {}

        self.R_max_spectrum = np.array([])
        self.G_max_spectrum = np.array([])
        self.B_max_spectrum = np.array([])
        self.W_max_spectrum = np.array([])

        self.R_max_luminance = -1
        self.G_max_luminance = -1
        self.B_max_luminance = -1
        self.W_max_luminance = -1

        self.RGB2LMS = np.array([])
        self.LMS2RGB = np.array([])
        self.DKL2RGB = np.array([])
        self.RGB2DKL = np.array([])

        self.XYZ2RGB = np.array([])
        self.RGB2XYZ = np.array([])

        self.visibleRadianceWindow = np.zeros((len(self.wlns)))

    def load_monxyY(self, fname):
        self.monxyY_file = fname
        self.monxyY = np.genfromtxt(fname, delimiter=',')
        self.calc_monXYZ()

    def set_monxyY(self, monxyY):
        self.monxyY = monxyY
        self.calc_monXYZ()

    def calc_monXYZ(self):
        self.monXYZ = np.zeros((3, 3))
        for c in range(3):
            self.monXYZ[:, c] = xyY2XYZ(self.monxyY[c, :])

        self.monWP = self.monXYZ.sum(axis = 1)
        self.maxLuminance = self.monWP[1]

        # self.monuvY = np.zeros(self.monxyY.shape)
        # for c in range(3):
            # self.monuvY[c, :] = xyY2uvY(self.monxyY[c, :])

        self.RGB2XYZ = self.monXYZ
        self.XYZ2RGB = np.linalg.inv(self.RGB2XYZ)

        self.dkl2rgbFromCalib()

    def set_monGamma(self, monGamma):
        self.monGamma = monGamma

    def load_spectralData(self, fname):
        self.spectralData_file = fname

        self.spectralData = sio.loadmat(fname)

        self.R_max_spectrum = (self.spectralData['SPECTRA'][0][0]['spectralData'][0][0]).flatten()
        self.R_max_luminance = self.spectralData['SPECTRA'][0][0]['Y'][0][0].flatten()[0]

        self.G_max_spectrum = (self.spectralData['SPECTRA'][0][1]['spectralData'][0][0]).flatten()
        self.G_max_luminance = self.spectralData['SPECTRA'][0][1]['Y'][0][0].flatten()[0]

        self.B_max_spectrum = (self.spectralData['SPECTRA'][0][2]['spectralData'][0][0]).flatten()
        self.B_max_luminance = self.spectralData['SPECTRA'][0][2]['Y'][0][0].flatten()[0]

        self.W_max_spectrum = (self.spectralData['SPECTRA'][0][3]['spectralData'][0][0]).flatten()
        self.W_max_luminance = self.spectralData['SPECTRA'][0][3]['Y'][0][0].flatten()[0]

        self.calc_spectralQuants()

    def set_R_spectrum(self, wlns, R_max_spectrum, R_max_luminance):
        r_interp = CubicSpline(wlns, R_max_spectrum)
        self.R_max_spectrum = r_interp(self.wlns)
        self.R_max_spectrum[self.R_max_spectrum < 0] = 0

        self.R_max_luminance = R_max_luminance

    def set_G_spectrum(self, wlns, G_max_spectrum, G_max_luminance):
        g_interp = CubicSpline(wlns, G_max_spectrum)
        self.G_max_spectrum = g_interp(self.wlns)
        self.G_max_spectrum[self.G_max_spectrum < 0] = 0

        self.G_max_luminance = G_max_luminance

    def set_B_spectrum(self, wlns, B_max_spectrum, B_max_luminance):
        b_interp = CubicSpline(wlns, B_max_spectrum)
        self.B_max_spectrum = b_interp(self.wlns)
        self.B_max_spectrum[self.B_max_spectrum < 0] = 0

        self.B_max_luminance = B_max_luminance

    def set_W_spectrum(self, wlns, W_max_spectrum, W_max_luminance):
        w_interp = CubicSpline(wlns, W_max_spectrum)
        self.W_max_spectrum = w_interp(self.wlns)
        self.W_max_spectrum[self.W_max_spectrum < 0] = 0

        self.W_max_luminance = W_max_luminance

    def calc_spectralQuants(self):
        self.maxRadiance = self.W_max_spectrum.sum()

        self.rgb2lmsTransform()
        self.computeVisibleRadianceWindow()

        self.visible_R_max_spectrum = self.R_max_spectrum * self.visibleRadianceWindow
        self.visible_G_max_spectrum = self.G_max_spectrum * self.visibleRadianceWindow
        self.visible_B_max_spectrum = self.B_max_spectrum * self.visibleRadianceWindow

        self.visible_W_max_spectrum = self.W_max_spectrum * self.visibleRadianceWindow

        self.maxVisibleRadiance = self.visible_W_max_spectrum.sum()

    def rgb2lmsTransform(self):
        l_interp = CubicSpline(lms_wlns, l_absorp)
        self.l_adj = l_interp(self.wlns)
        self.l_adj[self.l_adj < 0] = 0

        m_interp = CubicSpline(lms_wlns, m_absorp)
        self.m_adj = m_interp(self.wlns)
        self.m_adj[self.m_adj < 0] = 0

        # s curve stops early, so correct all the NaNs to 0, as they should be
        s_absorp[np.isnan(s_absorp)] = 0

        s_interp = CubicSpline(lms_wlns, s_absorp)
        self.s_adj = s_interp(self.wlns)
        self.s_adj[self.s_adj < 0] = 0

        # offs1 = np.arange(11, len(rs))
        # offs2 = np.arange(1, len(rs) - 10)

        rs = self.R_max_spectrum
        gs = self.G_max_spectrum
        bs = self.B_max_spectrum

        rL = rs @ self.l_adj
        rM = rs @ self.m_adj
        rS = rs @ self.s_adj

        gL = gs @ self.l_adj
        gM = gs @ self.m_adj
        gS = gs @ self.s_adj

        bL = bs @ self.l_adj
        bM = bs @ self.m_adj
        bS = bs @ self.s_adj

        self.RGB2LMS = np.array([
            [rL, gL, bL],
            [rM, gM, bM],
            [rS, gS, bS]])
        self.LMS2RGB = np.linalg.inv(self.RGB2LMS)

    def computeVisibleRadianceWindow(self):
        l_max_idx = np.argmax(self.l_adj)
        s_max_idx = np.argmax(self.s_adj)

        for c in range(s_max_idx+1):
            self.visibleRadianceWindow[c] = self.s_adj[c]

        for c in range(s_max_idx+1, l_max_idx+1):
            self.visibleRadianceWindow[c] = 1.0

        for c in range(l_max_idx+1, len(self.visibleRadianceWindow)):
            self.visibleRadianceWindow[c] = self.l_adj[c]

    # from Qasim Zaidi
    def dkl2rgbFromCalib(self):
        white = np.zeros((3,))

        rx = self.monxyY[0, 0]
        ry = self.monxyY[0, 1]
        white[0] = self.monxyY[0, 2]/2

        gx = self.monxyY[1, 0]
        gy = self.monxyY[1, 1]
        white[1] = self.monxyY[1, 2]/2
        
        bx = self.monxyY[2, 0]
        by = self.monxyY[2, 1]
        white[2] = self.monxyY[2, 2]/2

        R = cie2lms(rx, ry)
        G = cie2lms(gx, gy)
        B = cie2lms(bx, by)

        L = np.array([R[0], G[0], B[0]])
        M = np.array([R[1], G[1], B[1]])
        S = np.array([R[2], G[2], B[2]])

        lumchrm(white[0], white[1], white[2], L, M, S)

        print('Red Green Axis')
        deltaGrg = solvex(white[0]*S[0], white[0]*(L[0]+M[0]), S[1], S[2], L[1]+M[1], L[2]+M[2])
        deltaBrg = solvey(white[0]*S[0], white[0]*(L[0]+M[0]), S[1], S[2], L[1]+M[1], L[2]+M[2])
        dGrg = -1*deltaGrg/white[1]
        dBrg = -1*deltaBrg/white[2]
        lumchrm(0.0, white[1]+deltaGrg, white[2]+deltaBrg, L, M, S)
        lumchrm(white[0]*2.0, white[1]-deltaGrg, white[2]-deltaBrg, L, M, S)

        print('Blue Yellow Axis')
        deltaRyv = solvex(white[2]*L[2], white[2]*M[2], L[0], L[1], M[0], M[1])
        deltaGyv = solvey(white[2]*L[2], white[2]*M[2], L[0], L[1], M[0], M[1])
        dRyv = -1*deltaRyv/white[0]
        dGyv = -1*deltaGyv/white[1]
        lumchrm(white[0]+deltaRyv, white[1]+deltaGyv, 0.0, L, M, S)
        lumchrm(white[0]-deltaRyv, white[1]-deltaGyv, white[2]*2.0, L, M, S)

        self.DKL2RGB = np.array([
                        [1, 1,    dRyv],
                        [1, dGrg, dGyv],
                        [1, dBrg, 1]])
        self.RGB2DKL = np.linalg.inv(self.DKL2RGB)