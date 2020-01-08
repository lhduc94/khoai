# coding=utf-8

"""Data Frame Tools."""
from scipy.spatial.distance import minkowski
import numpy as np


def dist_with_miss(a, b, p=1, l=0.0):
    """ A function reduce memory of DataFrame.
    Parameters:
                df: DataFrame
                    A table of data.
                veborse: bool
                    Show mem. usage decreased.
    Output:
                DataFrame
    """

    if(len(a) != len(b)):
        return np.inf
    a = np.array(a)
    b = np.array(b)
    ls = l * np.ones(len(a))
    msk = ~ (np.isnan(a) | np.isnan(b))
    res = minkowski(a[msk], b[msk], p) + np.sum((ls[~msk]))**(1 / p)
    return res
