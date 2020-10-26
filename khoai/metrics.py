# coding=utf-8
"""Metrics Tools."""
from scipy.spatial.distance import minkowski
import numpy as np


def dist_with_miss(a, b, p=1, l=0.0):
    """ A function compute a distance between 2 array with missing value.
    Parameters:
                a: array
                    Input Array.
                b: array
                    Input Array.
                p: int
                    The order of the norm of the difference in minkowski distance.
                l: float
                    The lambda value of missing value.
    Output:
                distance: float
                    The distance between 2 array.
    """

    if len(a) != len(b):
        return np.inf
    a = np.array(a)
    b = np.array(b)
    ls = l * np.ones(len(a))
    msk = ~ (np.isnan(a) | np.isnan(b))
    res = minkowski(a[msk], b[msk], p) + np.sum((ls[~msk])) ** (1 / p)
    return res


def mape(a, b):
    """ A function compute a MAPE between 2 array with missing value.
        Parameters:
                    a: array
                        Input Array.
                    b: array
                        Input Array.
        Output:
                    MAPE
        """
    a = np.array(a)
    b = np.array(b)
    mask = a != 0
    return (np.abs(a - b) / a)[mask].mean() * 100


def smape(a, b):
    """ A function compute a SMAPE between 2 array with missing value.
            Parameters:
                        a: array
                            Input Array.
                        b: array
                            Input Array.
            Output:
                        SMAPE
            """
    a = np.array(a)
    b = np.array(b)
    mask = a != 0
    return (np.abs(a - b) / (np.abs(a) + np.abs(b)))[mask].mean() * 100


def sigmoid(x):
    """Sigmoid
            Parameter:
                        x: array or a number
            Output:
                        sigmoid value
    """
    return 1 / (1 + np.exp(-x))
