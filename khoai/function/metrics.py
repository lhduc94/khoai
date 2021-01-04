# coding=utf-8
"""Metrics Tools."""
from scipy.spatial.distance import minkowski
import numpy as np
import numbers


def _num_samples(x):
    """Return number of samples in array-like x."""
    message = 'Expected sequence or array-like, got %s' % type(x)
    if hasattr(x, 'fit') and callable(x.fit):
        # Don't get num_samples from an ensembles length!
        raise TypeError(message)

    if not hasattr(x, '__len__') and not hasattr(x, 'shape'):
        if hasattr(x, '__array__'):
            x = np.asarray(x)
        else:
            raise TypeError(message)

    if hasattr(x, 'shape') and x.shape is not None:
        if len(x.shape) == 0:
            raise TypeError("Singleton array %r cannot be considered"
                            " a valid collection." % x)
        if isinstance(x.shape[0], numbers.Integral):
            return x.shape[0]

    try:
        return len(x)
    except TypeError:
        raise TypeError(message)


def check_consistent_length(*arrays):
    """Check that all arrays have consistent first dimensions.
    Checks whether all objects in arrays have the same shape or length.
    Parameters
    ----------
    *arrays : list or tuple of input objects.
        Objects that will be checked for consistent length.
    """

    lengths = [_num_samples(X) for X in arrays if X is not None]
    uniques = np.unique(lengths)
    if len(uniques) > 1:
        raise ValueError("Found input variables with inconsistent numbers of"
                         " samples: %r" % [int(l) for l in lengths])


def levenshtein_distance(s1, s2, normalize=False):
    """
        A function computes the levenshtein distance between two string.
                Parameters:
                                s1: String
                                s2: String
                                normalize: bool
                                    divide edit distance by maximum length if true
                Returns:
                                The levenshtein distance
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1, normalize)
    if not s2:
        return len(s1)
    current_row = None
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
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
    """
        A function computes a distance between 2 array with missing value.
                Parameters:
                            a: array
                                Input Array.
                            b: array
                                Input Array.
                            p: int
                                The order of the norm of the difference in minkowski distance.
                            l: float
                                The lambda value of missing value.
                Returns:
                            distance: float
                                The distance between 2 array.
    """

    check_consistent_length(a, b)
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
                            A: Set or List Integer of String.
                                Input
                            B: Set or List Integer or Str.
                                Input
                Returns:
                            score: double
                                Jaccard similarity
    """
    if len(A) == 0 or len(B) == 0:
        return 0.0
    A = set(A)
    B = set(B)
    X = set(A) & set(B)
    Y = set(A) | set(B)
    score = 0.0
    if len(Y) > 0:
        score = len(X) / len(Y)
    return score


def mae(y_true, y_pred, sample_weight=None):
    """
        A function computes a mean absolute error between 2 array with missing value.
                Parameters:
                            y_true: array
                                Input Array.
                            y_pred: array
                                Input Array.
                            sample_weight: array
                                Sample weights
                Returns:
                            MAE
    """
    check_consistent_length(y_true, y_pred, sample_weight)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.average(np.abs(y_true - y_pred), weights=sample_weight, axis=0)


def mse(y_true, y_pred, sample_weight=None):
    """
        A function computes a mean squared error between 2 array with missing value.
                Parameters:
                            y_true: array
                                Input Array.
                            y_pred: array
                                Input Array.
                            sample_weight: array
                                Sample weights
                Returns:
                            MSE
    """
    check_consistent_length(y_true, y_pred, sample_weight)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.average((y_true - y_pred)**2, weights=sample_weight, axis=0)


def rmse(y_true, y_pred, sample_weight=None):
    """
        A function computes a mean squared error between 2 array with missing value.
                Parameters:
                            y_true: array
                                Input Array.
                            y_pred: array
                                Input Array.
                            sample_weight: array
                                Sample weights
                Returns:
                            RMSE
    """
    return np.sqrt(mse(y_true, y_pred, sample_weight))


def mape(y_true, y_pred):
    """
        A function computes a MAPE between 2 array with missing value.
                Parameters:
                            y_true: array
                                Input Array.
                            y_pred: array
                                Input Array.
                Returns:
                            MAPE: double
    """
    check_consistent_length(y_true, y_pred)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return (np.abs(y_true - y_pred) / y_true)[mask].mean() * 100


def smape(y_true, y_pred):
    """
        A function computes a SMAPE between 2 array with missing value.
                Parameters:
                            y_true: array
                                Input Array.
                            y_pred: array
                                Input Array.
                Returns:
                            SMAPE: double
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    return (np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))[mask].mean() * 100


def pk(y_true, y_pred, k=10):
    """
        Computes the precision at k.
        This function computes the precision at k between two lists of items.
                Parameters:

                            y_true: Array
                                    A Array of elements that are to be predicted (order doesn't matter)
                            y_pred: Array
                                    A Array of predicted elements (order does matter)
                            k: int, optional
                                The maximum number of predicted elements
                Returns:

                            score: double
                                The precision at k over the input lists
        Example :
        """
    if not y_pred or not y_true:
        return 0.0
    if len(y_pred) > k:
        y_pred = y_pred[:k]
    num_hits = 0.0
    for i, p in enumerate(y_pred):
        if p in y_true:
            num_hits += 1.0
    return num_hits / k


def apk(y_true, y_pred, k=10, normalize='min'):
    """
        Computes the average precision at k.
        This function computes the average precision at k between two lists of
        items.
                Parameters:
                            y_true: list
                                    A list of elements that are to be predicted (order doesn't matter)
                            y_pred: list
                                    A list of predicted elements (order does matter)
                            k: int, optional
                                The maximum number of predicted elements
                            normalize: type of normalize
                                'k': normalize by top k
                                'm': normalize by number of relevant documents (length of y_true)
                                'min': normalize by min(m,k)
                Returns
                            score : double
                                The average precision at k over the input lists
    """
    assert normalize in ['k', 'm', 'min'], "normalize should be in ['k', 'm', min']"
    if not y_true or not y_pred:
        return 0.0
    if len(y_pred) > k:
        y_pred = y_pred[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(y_pred):
        if p in y_true:
            num_hits += 1.0
            p_at_k = num_hits / (i + 1.0)
            score += p_at_k * 1  # (p@k * rel(k))

    if normalize == 'k':
        score = score / k
    elif normalize == 'm':
        score = score / len(y_true)
    else:
        score = score / min(len(y_true), k)
    return score

