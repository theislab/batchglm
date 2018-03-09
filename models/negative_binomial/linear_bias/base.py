from . import NegativeBinomialModel, NegativeBinomialInputData


__all__ = ['NegativeBinomialWithLinearBiasInputData', 'NegativeBinomialWithLinearBiasModel']


class NegativeBinomialWithLinearBiasInputData(NegativeBinomialInputData):
    
    def __init__(self, sample_data, design):
        super().__init__(sample_data)
        self.design = design
    
    @property
    def design(self):
        return self['design']
    
    @design.setter
    def design(self, value):
        self['design'] = value


class NegativeBinomialWithLinearBiasModel(NegativeBinomialModel):
    @property
    def bias_r(self):
        return self._r
    
    @property
    def bias_p(self):
        return self._p
    
    @bias_r.setter
    def bias_r(self, value):
        self._r = value
    
    @bias_p.setter
    def bias_p(self, value):
        self._p = value
