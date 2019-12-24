import numpy as np
from scipy.spatial.distance import minkowski

def dist_with_miss(a, b,p=1, l=0.0):
    if(len(a) != len(b)):
        return np.inf
    a = np.array(a)
    b = np.array(b)
    ls = l * np.ones(len(a))
    msk = ~ (np.isnan(a) | np.isnan(b))
    res = minkowski(a[msk],b[msk],p) + np.sum((ls[~msk]))**(1/p)
    return res