import abc


class SamplingMethod(object):
    @abc.abstractmethod
    def __init__(self):
        return

    @abc.abstractmethod
    def select_samples(self):
        return
