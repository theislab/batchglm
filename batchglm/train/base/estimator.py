import abc


class BaseEstimatorGlm(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def model_container(self):
        pass

    @abc.abstractmethod
    def initialize(self):
        pass

    @abc.abstractmethod
    def train_sequence(self):
        pass

    @abc.abstractmethod
    def finalize(self):
        pass
