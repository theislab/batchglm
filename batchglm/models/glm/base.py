import abc

from ..base import BasicModel


class Model(BasicModel, metaclass=abc.ABCMeta):
    """
    Generalized Linear Model base class.

    Every GLM has the parameters `link_loc` and `link_scale` and model-specific
    matrices `design_loc` and `design_scale`.
    They are linked to `location` and `scale` by a linker function, e.g. `location = exp(design_loc * link_loc)`.
    """

    @property
    @abc.abstractmethod
    def design_loc(self):
        pass

    @property
    @abc.abstractmethod
    def design_scale(self):
        pass

    @abc.abstractmethod
    def link_loc(self, data):
        pass

    @abc.abstractmethod
    def inverse_link_loc(self, data):
        pass

    @property
    def location(self):
        return self.inverse_link_loc(self.design_loc @ self.par_link_loc)

    @property
    @abc.abstractmethod
    def par_link_loc(self):
        pass

    @abc.abstractmethod
    def link_scale(self, data):
        pass

    @abc.abstractmethod
    def inverse_link_scale(self, data):
        pass

    @property
    def scale(self):
        return self.inverse_link_scale(self.design_scale @ self.par_link_scale)

    @property
    @abc.abstractmethod
    def par_link_scale(self):
        pass
