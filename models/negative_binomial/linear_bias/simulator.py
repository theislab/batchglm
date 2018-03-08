#!/usr/bin/env python3

import numpy as np

from models.negative_binomial import NegativeBinomialSimulator

from . import NegativeBinomialWithLinearBiasModel, NegativeBinomialWithLinearBiasInputData


class NegativeBinomialWithLinearBiasSimulator(NegativeBinomialSimulator, NegativeBinomialWithLinearBiasModel):
    cfg = NegativeBinomialSimulator.cfg.copy()
    
    def __init__(self, *args, **kwargs):
        NegativeBinomialSimulator.__init__(self, *args, **kwargs)
        self.data = NegativeBinomialWithLinearBiasInputData(None, None)
    
    def generate_params(self, num_classes=4, min_bias=0.8, max_bias=1.2, *args):
        super().generate_params(*args)
        
        self.data.design = np.repeat(range(num_classes), self.num_distributions / num_classes)
        self.data.design = self.data.design[range(self.num_distributions)]
        
        self.params['bias_r'] = np.random.uniform(min_bias, max_bias, num_classes)
        self.params['bias_mu'] = np.random.uniform(min_bias, max_bias, num_classes)
    
    @property
    def r(self):
        return super().r * self.bias_r
    
    @property
    def mu(self):
        return super().mu * self.bias_mu
    
    @property
    def bias_r(self):
        bias = self.params['bias_r'][self.data.design]
        return np.tile(bias, (self.num_samples, 1))
    
    @property
    def bias_mu(self):
        bias = self.params['bias_mu'][self.data.design]
        return np.tile(bias, (self.num_samples, 1))


def main():
    sim = NegativeBinomialWithLinearBiasSimulator()
    sim.generate()
    sim.save("resources/")
    return sim
