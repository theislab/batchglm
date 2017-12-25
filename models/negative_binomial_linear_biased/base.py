import abc

from models.negative_binomial import NegativeBinomialModel
from models import BasicEstimator


class Model(NegativeBinomialModel, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def a(self):
        pass

    @property
    @abc.abstractmethod
    def b(self):
        pass


class AbstractEstimator(Model, BasicEstimator, metaclass=abc.ABCMeta):
    pass
