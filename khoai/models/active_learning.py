import numpy as np


def least_confidence(y_pred_prob):
    y_pred_prob = np.array(y_pred_prob)
    assert y_pred_prob.shape[1] > 1, "least 2 probability"
    sorted_y_pred_prob = np.sort(y_pred_prob)
    argmax_y_pred = sorted_y_pred_prob[:, -1]
    return 1 - argmax_y_pred


def margin_sampling(y_pred_prob):
    y_pred_prob = np.array(y_pred_prob)
    assert y_pred_prob.shape[1] > 1, "least 2 probability"
    sorted_y_pred_prob = np.sort(y_pred_prob)
    return sorted_y_pred_prob[:, -2] - sorted_y_pred_prob[:, -1]


def entropy(y_pred_prob):
    y_pred_prob = np.array(y_pred_prob)
    assert y_pred_prob.shape[1] > 1, "least 2 probability"
    _entropy = np.log(y_pred_prob) * y_pred_prob
    _entropy = np.nan_to_num(_entropy, 1E-6)
    _entropy = -np.sum(_entropy, axis=1)
    return _entropy


