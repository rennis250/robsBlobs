import numpy as np

from .monitor import Monitor
from .infamous_lab import rgb2lab
from .lch import lab2lch

# https://stackoverflow.com/questions/596216/formula-to-determine-perceived-brightness-of-rgb-color
# /**
#  * Saturated colors appear brighter to human eye.
#  * That's called Helmholtz-Kohlrausch effect.
#  * Fairchild and Pirrotta came up with a formula to
#  * calculate a correction for that effect.
#  * "Color Quality of Semiconductor and Conventional Light Sources":
#  * https://books.google.ru/books?id=ptDJDQAAQBAJ&pg=PA45&lpg=PA45&dq=fairchild+pirrotta+correction&source=bl&ots=7gXR2MGJs7&sig=ACfU3U3uIHo0ZUdZB_Cz9F9NldKzBix0oQ&hl=ru&sa=X&ved=2ahUKEwi47LGivOvmAhUHEpoKHU_ICkIQ6AEwAXoECAkQAQ#v=onepage&q=fairchild%20pirrotta%20correction&f=false
#  * @return {number}
#  */
# function getLightnessUsingFairchildPirrottaCorrection([l, c, h]) {
#     const l_ = 2.5 - 0.025 * l
#     const g = 0.116 * Math.abs(Math.sin(degreesToRadians((h - 90) / 2))) + 0.085
#     return l + l_ * g * c
# }

# export function getPerceivedLightness(hex) {
#     return getLightnessUsingFairchildPirrottaCorrection(labToLch(xyzToLab(rgbToXyz(hex))))
# }

# Converts RGB values (between 0 and 1) first to LCh, then to the Fairchild L** measure for perceived brightness
def rgb2FairchildLstar(mon: Monitor, rgb):
    lab = rgb2lab(mon, rgb)
    l = lab[0]
    
    lch = lab2lch(lab)
    chrom = lch[1]
    H = lch[2]
    
    Hd = np.rad2deg(H)
    H_corr = np.deg2rad((Hd - 90)/2)
        
    # calculate the two functions: 
    # f1 provides the hue dependency, which looks like the absolute value of a
    # sin wave (half a cycle), i.e. the effect drops down at 90 degrees, i.e.
    # at yellow

    f1 = 0.116 * np.abs(np.sin(H_corr)) + 0.085

    # f2 allows the prediction of the HK effect to be much larger at low
    # luminance factors than at high luminance factors

    f2 = 2.5 - 0.025*l

    return l + (f2 * f1 * chrom)