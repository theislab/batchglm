import abc

import numpy as np

from ..base import BasicModel


class Model(BasicModel, metaclass=abc.ABCMeta):
    """
    Mixture Model base class.

    Every mixture model has the parameter `mixture_prob` which denotes the assignment probabilities of
    the samples to each mixture.
    They are linked to `location` and `scale` by a linker function, e.g. `location = exp(design_loc * link_loc)`.
    """

    @property
    def mixture_assignment(self):
        return np.squeeze(np.argmax(self.mixture_prob(), axis=-1))

    @property
    def mixture_weights(self):
        return np.exp(self.mixture_log_weights)

    @property
    @abc.abstractmethod
    def mixture_log_weights(self):
        pass

    @abc.abstractmethod
    def mixture_prob(self):
        pass

    @abc.abstractmethod
    def mixture_log_prob(self):
        pass

    @property
    @abc.abstractmethod
    def mixture_weight_constraints(self):
        pass
