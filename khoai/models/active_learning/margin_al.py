from .base import SamplingMethod
import numpy as np


class MarginAL(SamplingMethod):
    """https://arxiv.org/pdf/1906.00025v1.pdf"""
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.min_margin = None

    def update_threshold(self, threshold):
        self.threshold = threshold

    def get_score(self):
        return self.min_margin

    def select_samples(self, y_pred_prob=None, N=None, already_selected=None, **kwargs):
        assert y_pred_prob is not None, "y_pred_prob must not None"
        y_pred_prob = np.array(y_pred_prob)
        if N is None:
            N = y_pred_prob.shape[0]
        if already_selected is None:
            already_selected = []
        assert y_pred_prob.shape[1] > 1, "least 2 probability"
        sorted_y_pred_prob = np.sort(y_pred_prob)
        min_margin = sorted_y_pred_prob[:, -1] - sorted_y_pred_prob[:, -2]
        rank_ind = np.argsort(min_margin)
        rank_ind = [i for i in rank_ind if i not in already_selected]
        uncertain_samples = rank_ind[:N]
        certain_samples = list(np.where(min_margin - self.threshold <= 1E-6)[0])
        certain_samples = [i for i in certain_samples if i not in already_selected]
        self.min_margin = min_margin
        return certain_samples, uncertain_samples
