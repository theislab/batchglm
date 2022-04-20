import math
import numpy as np


def ll(scale, loc, x):
    resid = loc - x
    ll = -.5 * np.log(2 * math.pi) - np.log(scale) - .5 * np.power(resid / scale, 2)
    return ll
