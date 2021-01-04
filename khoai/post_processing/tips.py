# coding=utf-8
"""Tips"""


def calibrate(data, train_pop, target_pop, sampled_train_pop, sampled_target_pop):
    """ Calibrate data after undersample data.
    Parameters:
                data: float array
                    probability prediction Array.
                train_pop: int
                    Number of data in training data.
                target_pop: int
                    Number of target class (minority class) in the training dataset.
                sampled_train_pop: int
                    Number of data in training dataset after undersampling.
                sampled_target_pop: int
                    Number of the target class(minority class) in the training dataset after undersampling.
    Return:
                data: float array
                    probability prediction Array after calibrate.
    """
    a = data * (target_pop / train_pop) / (sampled_target_pop / sampled_train_pop)
    b = (1 - data) * (1 - target_pop / train_pop) / (1 - sampled_target_pop / sampled_train_pop)
    calibrated_data = a / (a + b)
    return calibrated_data
