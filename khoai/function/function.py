import numpy as np


def sigmoid(x):
    """Sigmoid
            Parameter:
                        x: array or a number
            Returns:
                        sigmoid value
    """
    return 1 / (1 + np.exp(-x))