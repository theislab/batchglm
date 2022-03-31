import math
import numpy as np

def ll(scale, loc, x):
    resid = loc - x
    ll = -.5 * loc.shape[0] * np.log(2 * math.pi * scale) - .5 * np.linalg.norm(resid, axis=0) / np.power(scale, 2)
    return ll
