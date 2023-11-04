import numpy as np

# helper functions for computing DKL2RGB in monitor.py
# from Qasim Zaidi
def lumchrm(lumr, lumg, lumb, r, g, b):
  lum = np.array([lumr, lumg, lumb])

  bigl = 0.0
  bigm = 0.0
  bigs = 0.0

  for i in range(3):
     print('Luminances[', i, '] = ', lum[i])
     lu = lum[i]
     bigl = bigl + lu*r[i]
     bigm = bigm + lu*g[i]
     bigs = bigs + lu*b[i]

  denom = bigl+bigm
  print('bigl =', bigl ,' bigm =', bigm, ' bigs =', bigs)

  lout = bigl/denom
  sout = bigs/denom
  print('L/L+M =', lout, ' S/L+M =', sout)

  return


def solvex(a, b, c, d, e, f):
  return (a*f/d-b)/(c*f/d-e)


def solvey(a, b, c, d, e, f):
  return (a*e/c-b)/(d*e/c-f)
  

def cie2lms(x, y):
  """
  Convert CIE color space coordinates to LMS color space coordinates.

  Args:
    x (float): The x coordinate in CIE color space.
    y (float): The y coordinate in CIE color space.

  Returns:
    numpy.ndarray: A 1D array of length 3 containing the LMS color space coordinates.
  """
  z = 1-x-y
  cie = np.array([x, y, z]).T

  matrix = np.array([[ .15514, .54316, -.03286],
             [-.15514, .45684,  .03286],
             [  0,     0,       .01608]])

  lms = matrix @ cie

  l = lms[0]/(lms[0]+lms[1]) #   L/(L+M)
  m = lms[1]/(lms[0]+lms[1]) #   M/(L+M)
  s = lms[2]/(lms[0]+lms[1]) #   S/(L+M)
  norm = np.array([l, m, s])

  return norm