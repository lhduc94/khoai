from .base import SamplingMethod
import numpy as np


class UniformAL(SamplingMethod):

    def __init__(self, X, seed):
        super().__init__()
        self.X = X
        np.random.seed(seed)

    def select_samples(self, N=None, already_selected=None, **kwargs):
        if N is None:
            N = self.X.shape[0]
        if already_selected is None:
            already_selected = []
        samples = [i for i in range(self.X.shape[0]) if i not in already_selected]
        return [], samples[0:N]
