import abc

import os
import math
import numpy as np
import pandas as pd
import patsy

from . import NegativeBinomialSimulator
from .base import Model, InputData

__all__ = ['Simulator']


class Simulator(NegativeBinomialSimulator, Model, metaclass=abc.ABCMeta):
    # static variables
    cfg = NegativeBinomialSimulator.cfg.copy()
    
    # type hinting
    data: InputData
    sample_description: pd.DataFrame
    
    def __init__(self, *args, **kwargs):
        NegativeBinomialSimulator.__init__(self, *args, **kwargs)
        
        self.data = InputData(None, None)
    
    def generate_design_matrix(self, num_batches=4, num_confounder=2):
        reps_batches = math.ceil(self.num_samples / num_batches)
        reps_confounder = math.ceil(self.num_samples / num_confounder)
        
        # batch column
        batches = np.repeat(range(num_batches), reps_batches)
        batches = batches[range(self.num_samples)]
        
        # confounder column
        confounders = np.squeeze(np.tile([np.arange(num_confounder)], reps_confounder))
        confounders = confounders[range(self.num_samples)]
        
        # build sample description
        sample_description = {
            "batch": batches,
            "confounder": confounders,
        }
        sample_description = pd.DataFrame(data=sample_description, dtype="category")
        self.data["sample_description"] = sample_description
    
    def generate_params(self, *args, sample_description: pd.DataFrame = None,
                        min_bias=0.5, max_bias=2, **kwargs):
        super().generate_params(*args, **kwargs)
        
        if sample_description is None:
            if self.data.get("sample_description", None) is None:
                self.generate_design_matrix()
            sample_description = self.data["sample_description"]
        
        formula = "~ 1 "
        for col in sample_description.columns:
            formula += " + %s" % col
        
        self.data.design = patsy.dmatrix(formula, sample_description)
        
        # num_classes = np.unique(self.data.design, axis=0).shape[0]
        
        self.params['bias_r'] = np.log(np.random.uniform(min_bias, max_bias, self.data.design.shape[1] - 1))
        self.params['bias_mu'] = np.log(np.random.uniform(min_bias, max_bias, self.data.design.shape[1] - 1))
    
    @property
    def r(self):
        return super().r * self.bias_r
    
    @property
    def mu(self):
        return super().mu * self.bias_mu
    
    @property
    def bias_r(self, expanded=True):
        # bias = exp(log_bias) = exp(
        #   design[samples, num_params] %*% t(params[num_params])
        # )
        bias = np.exp(np.matmul(self.data.design[:, 1:], self.params['bias_r']))
        if expanded:
            bias = np.expand_dims(bias, axis=1)
            return np.tile(bias, (1, self.num_distributions))
        else:
            return bias
    
    @property
    def bias_mu(self, expanded=True):
        # bias = exp(log_bias) = exp(
        #   design[samples, num_params] %*% t(params[num_params])
        # )
        bias = np.exp(np.matmul(self.data.design[:, 1:], self.params['bias_mu']))
        if expanded:
            bias = np.expand_dims(bias, axis=1)
            return np.tile(bias, (1, self.num_distributions))
        else:
            return bias
    
    def load(self, folder):
        super().load(folder)
        
        file = os.path.join(folder, "sample_description.tsv")
        if os.path.isfile(file):
            self.sample_description = pd.read_csv("resources/params.tsv", sep="\t", dtype="categorial")
    
    def save(self, folder):
        super().save()
        
        file = os.path.join(folder, "sample_description.tsv")
        self.sample_description.to_csv(file, sep="\t", index=False)


def main():
    sim = Simulator()
    sim.generate()
    sim.save("resources/")
    return sim
