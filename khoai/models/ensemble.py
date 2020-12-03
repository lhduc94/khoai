import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from typing import List, Dict, Callable, NamedTuple, Optional, Tuple, Any
from tqdm import tqdm
import torch
import torch.nn as nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Blender(object):

    def __init__(self,
                 n_splits: List[int],
                 n_seeds: int,
                 loss_func: Callable):
        self.n_splits = n_splits
        self.n_seeds = n_seeds
        self.loss_func = loss_func

    def get_score(self, weights, train_idx, oofs, label):
        blend = np.zeros_like(oofs[0][train_idx, :])
        for oof, weight in zip(oofs[:-1], weights):
            blend += weight * oofs[train_idx, :]
        blend += (1 - np.sum(weights)) * oofs[-1][train_idx, :]
        return self.loss_func(label[train_idx, :], blend)

    def get_best_weights(self, oofs, label):
        weight_list = []
        weights = np.array([1 / len(oofs) for x in range(len(oofs) - 1)])

        for n_split in tqdm(self.n_splits):
            for seed in range(self.n_seeds):
                kfold = KFold(n_splits=n_split, random_state=seed, shuffle=True)
                for fold, (train_idx, valid_idx) in enumerate(kfold.split(X=oofs[0])):
                    res = minimize(self.get_score, weights, args=(train_idx, oofs, label),
                                   method="Nelder-Mead",
                                   tol=1e-6)
                    weight_list.append(res.x)
        mean_weight = np.mean(weight_list, axis=0)
        return mean_weight


class CNNStacking(nn.Module):
    def __init__(self, n_features, n_labels):
        super(CNNStacking, self).__init__()

        self.sq = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 1), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(2, 1), bias=False),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=16 * n_labels, out_features=4 * n_labels),
            nn.ReLU(),
            nn.Linear(in_features=4 * n_labels, out_features=n_labels),
        )

    def forward(self, x):
        return self.sq(x)
