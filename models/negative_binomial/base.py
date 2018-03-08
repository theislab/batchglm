from models import BasicInputData


class NegativeBinomialInputData(BasicInputData):
    # same as BasicInputData
    pass


class NegativeBinomialModel:
    @property
    def r(self):
        return self._r
    
    @property
    def p(self):
        return self._p
    
    @property
    def mu(self):
        return self._mu
    
    @mu.setter
    def mu(self, value):
        self._mu = value
    
    @r.setter
    def r(self, value):
        self._r = value
    
    @p.setter
    def p(self, value):
        self._p = value
