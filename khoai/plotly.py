# coding=utf-8
"""Plot Tools."""
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List


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


def heatmap(x, y, size, color, **params):
    n_colors = params.get('n_colors', 256)
    palette = sns.diverging_palette(20, 220, n=n_colors)
    color_min, color_max = [-1, 1]

    def value_to_color(val):
        val_position = float((val - color_min)) / (color_max - color_min)
        ind = int(val_position * (n_colors - 1))
        return palette[ind]

    figure = plt.figure(figsize=params.get('figsize', None))
    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1)
    ax = plt.subplot(plot_grid[:, :-1])

    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size * params.get('size_scale', 500),
        c=color.map(value_to_color),
        marker=params.get('marker', 's')
    )

    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=params.get('x_rotation', 45), horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    ax = plt.subplot(plot_grid[:, -1])

    col_x = [0] * len(palette)
    bar_y = np.linspace(color_min, color_max, n_colors)

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5] * len(palette),
        left=col_x,
        height=bar_height,
        color=palette,
        linewidth=0
    )
    ax.set_xlim(1, 2)
    ax.grid(False)
    ax.set_facecolor('white')
    ax.set_xticks([])
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3))
    ax.yaxis.tick_right()
    return figure


def plot_corr(df: pd.DataFrame, columns=None, **params):
    """
            A function plot correlation heatmap.
                    Parameters:
                                df: DataFrame
                                    True label
                                columns: List of columns
                                    If columns = None, get all columns of df
                                **params: The parameters of figure.
                                    figsize: Tuple of figure size. Default (17,10)
                                    n_colors: Number of color. Default 256
                                    size_scale: Scale point. Default 500

                    Returns:
                                figure: Figure
        """

    if columns is None:
        columns = df.columns
    corr = df[columns].corr()
    corr = pd.melt(corr.reset_index(),
                   id_vars='index')
    corr.columns = ['x', 'y', 'value']
    figure = heatmap(x=corr['x'], y=corr['y'], size=corr['value'].abs(), color=corr['value'], **params)

    return figure
