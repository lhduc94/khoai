# coding=utf-8

"""Plot Tools."""
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve


def plot_roc_curve(y_true, y_pred_prob):
    """A function plot Roc AUC.
    Parameters:
                y_true: Array
                    True label
                y_pred_prob: Array
                    Probability predicted label
    Output:
                roc_auc
        """

    plt.figure(figsize=(17, 10))
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)  # compute area under the curve
    plt.figure(figsize=(17, 10))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.5f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',
             linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold', color='r')
    ax2.set_ylim([thresholds[-1], thresholds[0]])
    ax2.set_xlim([fpr[0], fpr[-1]])
    plt.show()

    return roc_auc
