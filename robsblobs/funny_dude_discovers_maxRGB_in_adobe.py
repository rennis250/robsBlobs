import numpy as np

# found this by accident:
# https://alienryderflex.com/hsp.html
# i wonder what other surprises lie out there

Pr = 0.241
Pg = 0.691
Pb = 0.068

Pr_informed = 0.299
Pg_informed = 0.587
Pb_informed = 0.114

Pr_jing = 0.302
Pg_jing = 0.550
Pb_jing = 0.148

Pr_newJing = 0.2667
Pg_newJing = 0.6267
Pb_newJing = 0.1067

# public domain function by Darel Rex Finley, 2006
#
# This function expects the passed-in values to be on a scale
# of 0 to 1, and uses that same scale for the return values.
#
# See description/examples at alienryderflex.com/hsp.html
def RGBtoHSP(R, G, B, version):
  # Calculate the Perceived brightness.
  if version == "original":
    P = np.sqrt(R*R*Pr + G*G*Pg + B*B*Pb)
  elif version == "photoshop":
    P = np.sqrt(R*R*Pr_informed + G*G*Pg_informed + B*B*Pb_informed)
  elif version == "jing":
    P = np.sqrt(R*R*Pr_jing + G*G*Pg_jing + B*B*Pb_jing)
  elif version == "jing_new":
    P = np.sqrt(R*R*Pr_newJing + G*G*Pg_newJing + B*B*Pb_newJing)

  # Calculate the Hue and Saturation. (This part works
  # the same way as in the HSV/B and HSL systems???.)
  if R==G and R==B:
    H = 0.0
    S = 0.0
    return np.array([H, S, P])

  if R>=G and R>=B:              # R is largest
    if B>=G:
      H = 6/6 - 1/6*(B-G)/(R-G)
      S = 1 - G/R
    else:
      H = 0/6 + 1/6*(G-B)/(R-B)
      S = 1 - B/R
  elif G >= R and G >= B:        # G is largest
    if R>=B:
      H = 2/6 - 1/6*(R-B)/(G-B)
      S = 1 - B/G
    else:
      H = 2/6 + 1/6*(B-R)/(G-R)
      S = 1 - R/G
  else:                          # B is largest
    if G>=R:
      H = 4/6 - 1/6*(G-R)/(B-R)
      S = 1 - R/B
    else:
      H = 4/6 + 1/6*(R-G)/(B-G)
      S = 1 - G/B

  return np.array([H, S, P])


# public domain function by Darel Rex Finley, 2006
#
# This function expects the passed-in values to be on a scale
# of 0 to 1, and uses that same scale for the return values.
#
# Note that some combinations of HSP, even if in the scale
# 0-1, may return RGB values that exceed a value of 1.  For
# example, if you pass in the HSP color 0,1,1, the result
# will be the RGB color 2.037,0,0.
#
# See description/examples at alienryderflex.com/hsp.html
def HSPtoRGB(H, S, P):
  part = 0
  minOverMax = 1 - S

  if minOverMax>0.:
    if H < 1/6:                                                  # R>G>B
      H = 6*(H - 0/6)
      part = 1 + H*(1/minOverMax - 1)
      B = P/np.sqrt(Pr/minOverMax/minOverMax + Pg*part*part + Pb)
      R = B/minOverMax
      G = B + H*(R - B)
    elif H < 2/6:                                                # G>R>B
      H = 6*(-H + 2/6)
      part = 1 + H*(1/minOverMax - 1)
      B = P/np.sqrt(Pg/minOverMax/minOverMax + Pr*part*part + Pb)
      G = B/minOverMax
      R = B + H*(G - B)
    elif H < 3/6:                                                # G>B>R
      H = 6*(H - 2/6)
      part = 1 + H*(1/minOverMax - 1)
      R = P/np.sqrt(Pg/minOverMax/minOverMax + Pb*part*part + Pr)
      G = R/minOverMax
      B = R + H*(G - R)
    elif H < 4/6:                                                # B>G>R
      H = 6*(-H + 4/6)
      part = 1 + H*(1/minOverMax-1)
      R = P/np.sqrt(Pb/minOverMax/minOverMax + Pg*part*part + Pr)
      B = R/minOverMax
      G = R+H*(B-R)
    elif H < 5./6.:                                              # B>R>G
      H = 6*(H - 4/6)
      part = 1 + H*(1/minOverMax-1)
      G = P/np.sqrt(Pb/minOverMax/minOverMax + Pr*part*part + Pg)
      B = G/minOverMax
      R = G+H*(B-G)
    else:                                                        # R>B>G
      H = 6*(-H + 6/6)
      part = 1 + H*(1/minOverMax-1)
      G = P/np.sqrt(Pr/minOverMax/minOverMax + Pb*part*part + Pg)
      R = G/minOverMax
      B = G+H*(R-G)
  else:
    if H < 1/6:                              # R>G>B
      H = 6*(H - 0/6)
      R = np.sqrt(P*P/(Pr + Pg*H*H))
      G = R*H
      B = 0
    elif H < 2/6:                            # G>R>B
      H = 6*(-H + 2/6)
      G = np.sqrt(P*P/(Pg + Pr*H*H))
      R = G*H
      B = 0
    elif H < 3/6:                            # G>B>R
      H = 6*(H - 2/6)
      G = np.sqrt(P*P/(Pg + Pb*H*H))
      B = G*H
      R = 0
    elif H < 4/6:                            # B>G>R
      H = 6*(-H + 4/6)
      B = np.sqrt(P*P/(Pb + Pg*H*H))
      G = B*H
      R = 0
    elif H < 5/6:                            # B>R>G
      H = 6*(H - 4/6)
      B = np.sqrt(P*P/(Pb + Pr*H*H))
      R = B*H
      G = 0
    else:                                    # R>B>G
      H = 6*(-H + 6/6)
      R = np.sqrt(P*P/(Pr + Pb*H*H))
      B = R*H
      G = 0

  return np.array([R, G, B])