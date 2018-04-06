import abc

from models import BasicInputData


class InputData(BasicInputData):
    # same as BasicInputData
    @property
    def initial_mixture_assignment(self):
        return self['initial_mixture_assignment']


class Model(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def r(self):
        pass
    
    @property
    @abc.abstractmethod
    def p(self):
        pass

    @property
    @abc.abstractmethod
    def mu(self):
        pass

    @property
    @abc.abstractmethod
    def mixture_assignment(self):
        pass
