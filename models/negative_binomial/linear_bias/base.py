import abc

from . import NegativeBinomialModel, NegativeBinomialInputData


class InputData(NegativeBinomialInputData):
    
    def __init__(self, sample_data, design):
        super().__init__(sample_data)
        self.design = design
    
    @property
    def design(self):
        return self['design']
    
    @design.setter
    def design(self, value):
        self['design'] = value


class Model(NegativeBinomialModel, metaclass=abc.ABCMeta):
    
    @property
    @abc.abstractmethod
    def a(self):
        pass
    
    @property
    @abc.abstractmethod
    def b(self):
        pass
