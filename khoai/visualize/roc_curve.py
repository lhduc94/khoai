# coding=utf-8
"""Plot Tools."""
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import numpy as np


def plot_roc_curve(y_true, y_pred_prob, show_threshold=False, **params):
    """
        A function plot Roc AUC.
                Parameters:
                            y_true: Array
                                True label
                            y_pred_prob: Array
                                Probability predicted label
                            show_threshold: Bool
                                Show threshold
                Returns:
                            figure: Figure
                            roc_auc: AUC value
    """

    figure = plt.figure(figsize=params.get('figsize', (17, 10)))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    if show_threshold:
        ax2 = plt.gca().twinx()
        ax2.plot(fpr, thresholds, markeredgecolor='r',
                 linestyle='dashed', color='r')
        ax2.set_ylabel('Threshold', color='r')
        ax2.set_ylim([0.0, 1.0])
        ax2.set_xlim([0.0, 1.0])

    plt.show()

    return figure, roc_auc


def plot_multi_roc_curve(y_trues, y_pred_probs, labels, **params):
    """
        A function plot Roc AUC.
                Parameters:
                            y_trues: Array of Array
                                True label
                            y_pred_probs: Array of Array
                                Probability predicted label
                            labels: List
                                List of label
                Returns:
                            figure: Figure
                            roc_aucs: List AUC value
    """

    figure = plt.figure(figsize=params.get('figsize', (17, 10)))
    roc_aucs = []
    for y_true, y_pred_prob, label in zip(y_trues, y_pred_probs, labels):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        roc_aucs.append(roc_auc)
        plt.plot(fpr, tpr, label=f'{label} ROC curve (area = %0.5f)' % roc_auc)

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")

    plt.show()

    return figure, roc_aucs
