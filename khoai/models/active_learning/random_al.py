import numpy as np
from .base import SamplingMethod


class RandomAL(SamplingMethod):
    def __init__(self, X, seed):
        super().__init__()
        self.X = X
        self.index = np.arange(X.shape[0])
        np.random.seed(seed)
        self.seed = seed

    def reverse(self):
        np.random.seed(self.seed)

    def select_samples(self, N=None, already_selected=None, **kwargs):
        if N is None:
            N = self.X.shape[0]
        if already_selected is None:
            already_selected = []
        index = list(set(self.index) - set(already_selected))
        N = min(len(index), N)
        samples = np.random.choice(index, N, replace=False)
        return [], list(samples)
