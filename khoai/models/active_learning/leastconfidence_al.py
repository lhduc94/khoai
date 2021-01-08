from .base import SamplingMethod
import numpy as np


class LeastConfidenceAL(SamplingMethod):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold
        self.uncertainty = None

    def update_threshold(self, threshold):
        self.threshold = threshold

    def get_score(self):
        return self.uncertainty

    def select_samples(self, y_pred_prob=None, N=None, already_selected=None, **kwargs):
        assert y_pred_prob is not None, "y_pred_prob must not None"
        y_pred_prob = np.array(y_pred_prob)
        if N is None:
            N = y_pred_prob.shape[0]
        if already_selected is None:
            already_selected = []
        assert y_pred_prob.shape[1] > 1, "least 2 probability"
        sorted_y_pred_prob = np.sort(y_pred_prob)
        argmax_y_pred = sorted_y_pred_prob[:, -1]
        uncertainty = 1 - argmax_y_pred
        rank_ind = np.argsort(uncertainty)[::-1]
        rank_ind = [i for i in rank_ind if i not in already_selected]
        uncertain_samples = rank_ind[:N]
        certain_samples = list(np.where(uncertainty - self.threshold <= 1E-6)[0])
        certain_samples = [i for i in certain_samples if i not in already_selected]
        self.uncertainty = uncertainty
        return certain_samples, uncertain_samples
