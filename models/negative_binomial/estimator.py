from models import BasicEstimator

from . import NegativeBinomialModel

__all__ = ['AbstractNegativeBinomialEstimator']


class AbstractNegativeBinomialEstimator(NegativeBinomialModel, BasicEstimator):
    pass
