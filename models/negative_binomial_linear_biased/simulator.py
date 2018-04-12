import abc

import os
import math
import numpy as np
import pandas as pd
import patsy

from models.negative_binomial import NegativeBinomialSimulator
from .base import Model, InputData


class Simulator(NegativeBinomialSimulator, Model, metaclass=abc.ABCMeta):
    # static variables
    cfg = NegativeBinomialSimulator.cfg.copy()
    
    # type hinting
    data: InputData
    sample_description: pd.DataFrame
    
    def __init__(self, *args, **kwargs):
        NegativeBinomialSimulator.__init__(self, *args, **kwargs)
        Model.__init__(self)
        
        self.data = InputData(None, None)
        self.sample_description = None
    
    def generate_sample_description(self, num_batches=4, num_confounder=2):
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
        self.sample_description = pd.DataFrame(data=sample_description, dtype="category")
    
    def generate_params(self, *args, min_bias=0.5, max_bias=2, **kwargs):
        super().generate_params(*args, **kwargs)
        
        if self.sample_description is None:
            self.generate_sample_description()
        
        self.formula = "~ 1 "
        for col in self.sample_description.columns:
            self.formula += " + %s" % col
        
        self.data.design = patsy.dmatrix(self.formula, self.sample_description)
        
        # num_classes = np.unique(self.data.design, axis=0).shape[0]
        
        self.params['a'] = np.log(
            np.concatenate([
                np.expand_dims(self.params["r"], 0),
                np.random.uniform(min_bias, max_bias, (self.data.design.shape[1] - 1, self.num_genes))
            ])
        )
        self.params['b'] = np.log(
            np.concatenate([
                np.expand_dims(self.params["mu"], 0),
                np.random.uniform(min_bias, max_bias, (self.data.design.shape[1] - 1, self.num_genes))
            ])
        )
    
    @property
    def r(self):
        return np.exp(np.matmul(self.data.design, self.params['a']))
    
    @property
    def mu(self):
        return np.exp(np.matmul(self.data.design, self.params['b']))
    
    @property
    def a(self):
        return self.params['a']
    
    @property
    def b(self):
        return self.params['b']
    
    def load(self, folder):
        super().load(folder)
        
        file = os.path.join(folder, "sample_description.tsv")
        if os.path.isfile(file):
            self.sample_description = pd.read_csv(file, sep="\t", dtype="category")
            
            self.formula = "~ 1 "
            for col in self.sample_description.columns:
                self.formula += " + %s" % col
    
    def save(self, folder):
        super().save(folder)
        
        file = os.path.join(folder, "sample_description.tsv")
        self.sample_description.to_csv(file, sep="\t", index=False)


def main():
    sim = Simulator()
    sim.generate()
    sim.save("resources/")
    return sim
