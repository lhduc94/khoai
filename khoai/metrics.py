# coding=utf-8
"""Metrics Tools."""
from scipy.spatial.distance import minkowski
import numpy as np


def levenshtein(s1, s2, normalize=False):
    """ A function computes the levenshtein distance between two string.

        s1: String
        s2: String
    normalize: divide edit distance by maximum length if true
    """
    if len(s1) < len(s2):
        return levenshtein(s2, s1, normalize)
    if not s2:
        return len(s1)
    current_row = None
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i+1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j+1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    if current_row:
        if normalize:
            return current_row[-1] / len(s1)
        return current_row[-1]
    return -1


def dist_with_miss(a, b, p=1, l=0.0):
    """ A function computes a distance between 2 array with missing value.
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


def jaccard_similarity(A, B):
    """
        A function computes a Jaccard_e similarity of 2 set.
            Parameters:
                        A: Set or List.
                            Input
                        B: Set or List.
                            Input
            Output:
                        score: double
    """

    A = set(A)
    B = set(B)
    X = set(A) & set(B)
    Y = set(A) | set(B)
    score = 0.0
    if len(Y) > 0:
        score = len(X) / len(Y)
    return score


def mape(a, b):
    """
        A function computes a MAPE between 2 array with missing value.
                Parameters:
                            a: array
                                Input Array.
                            b: array
                                Input Array.
                Output:
                            MAPE: double
    """
    a = np.array(a)
    b = np.array(b)
    mask = a != 0
    return (np.abs(a - b) / a)[mask].mean() * 100


def smape(a, b):
    """
        A function computes a SMAPE between 2 array with missing value.
                Parameters:
                            a: array
                                Input Array.
                            b: array
                                Input Array.
                Output:
                            SMAPE: double
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


def pk(actual, predicted, k=10):
    """
        Computes the precision at k.
        This function computes the precision at k between two lists of items.
                Parameters:

                            actual: Array
                                     A Array of elements that are to be predicted (order doesn't matter)
                            predicted: Array
                                        A Array of predicted elements (order does matter)
                            k: int, optional
                                The maximum number of predicted elements
                Output:

                            score: double
                                The precision at k over the input lists
        """
    if not actual:
        return 0.0
    if len(predicted) > k:
        predicted = predicted[:k]
    predicted = set(predicted)
    actual = set(actual)
    hit = predicted & actual

    return len(hit) / len(predicted)


def mpk(actual, predicted, k=10):
    """
        Computes the mean precision at k.
        This function computes the mean average precision at k between two lists of lists of items.
                Parameters:

                            actual: Array
                                     A Array of elements that are to be predicted (order doesn't matter)
                            predicted: Array
                                        A Array of predicted elements (order does matter)
                            k: int, optional
                                The maximum number of predicted elements
                Output:
                            score: double
                The mean  precision at k over the input lists
    """
    return np.mean([pk(a, p, k) for a, p in zip(actual, predicted)])


def apk(actual, predicted, k=10):
    """
        Computes the average precision at k.
        This function computes the average precision at k between two lists of items.
                Parameters:

                            actual : Array
                                     A Array of elements that are to be predicted (order doesn't matter)
                            predicted : Array
                                        A Array of predicted elements (order does matter)
                            k : int, optional
                                The maximum number of predicted elements
                Output:

                            score : double
                                The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average precision at k between two lists of lists of items.
            Parameters:

                        actual : Array
                                 A Array of elements that are to be predicted (order doesn't matter)
                        predicted : Array
                                    A Array of predicted elements (order does matter)
                        k : int, optional
                            The maximum number of predicted elements
            Output:
                        score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])
