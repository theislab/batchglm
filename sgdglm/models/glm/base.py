import abc

from models.base import BasicModel


class Model(BasicModel, metaclass=abc.ABCMeta):

    @property
    @abc.abstractmethod
    def design_loc(self):
        pass

    @property
    @abc.abstractmethod
    def design_scale(self):
        pass

    @abc.abstractmethod
    def link_fn(self, data):
        pass

    @abc.abstractmethod
    def inverse_link_fn(self, data):
        pass

    @property
    def location(self):
        return self.inverse_link_fn(self.design_loc @ self.link_loc)

    @property
    @abc.abstractmethod
    def link_loc(self):
        pass

    @property
    def scale(self):
        return self.inverse_link_fn(self.design_scale @ self.link_scale)

    @property
    @abc.abstractmethod
    def link_scale(self):
        pass
