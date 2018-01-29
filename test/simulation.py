#!/usr/bin/env python3

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import NegativeBinomial as NBdist

import os

from abc import abstractclassmethod


class MatrixSimulator:
    """
    Classes implementing `MatrixSimulator` should be able to generate a
    2D-matrix of sample data, as well as a dict of corresponding parameters.

    convention: N distributions with M samples each => (M, N) matrix
    """
    cfg = {
        "data": "data.tsv",
        "param_folder": "params",
    }

    def __init__(self, num_samples=2000, num_distributions=10000):
        self.num_samples = num_samples
        self.num_distributions = num_distributions

        self.data = None
        self.params = {}

    def generate(self):
        self.generate_params()
        self.generate_data()

    @abstractclassmethod
    def generate_data(self, *args):
        pass

    @abstractclassmethod
    def generate_params(self, *args):
        pass

    def load(self, folder):
        self.data = np.loadtxt(
            os.path.join(folder, self.cfg["data"]), delimiter="\t")

        self.num_samples = self.data.shape[0]
        self.num_distributions = self.data.shape[1]

        param_folder = os.path.join(folder, self.cfg['param_folder'])
        if os.path.isdir(param_folder):
            for param_name in os.listdir(param_folder):
                file = os.path.join(param_folder, param_name)
                if os.path.isfile(file):
                    self.params[param_name] = np.loadtxt(
                        os.path.join(folder, self.cfg["data"]), delimiter="\t")

    def save(self, folder):
        np.savetxt(os.path.join(folder, self.cfg["data"]), self.data, delimiter="\t")

        param_folder = os.path.join(folder, self.cfg['param_folder'])
        os.makedirs(param_folder, exist_ok=True)

        for (param, val) in self.params.items():
            np.savetxt(os.path.join(param_folder, param), val, delimiter="\t")


class NegativeBinomial(MatrixSimulator):
    cfg = MatrixSimulator.cfg.copy()

    def __init__(self, *args):
        super().__init__(*args)

    def negative_binomial(self):
        # ugly hack using tensorflow, since parametrisation with `p`
        # does not work with np.random.negative_binomial

        with tf.Graph().as_default() as g:
            r = tf.constant(self.r)
            mu = tf.constant(self.mu)

            p = mu / (r + mu)

            dist = NBdist(total_count=r, probs=p)

            # run sampling session
            with tf.Session() as sess:
                return sess.run(tf.squeeze(
                    dist.sample(1)
                ))

        # random_data = np.random.negative_binomial(
        #     self.r,
        #     self.p,
        # )
        # return random_data

    @property
    def r(self):
        return np.tile(self.params['r'], (self.num_samples, 1))

    @property
    def p(self):
        # - p*(mu - 1) = p * r / (-p)
        return self.mu / (self.r + self.mu)

    @property
    def mu(self):
        return np.tile(self.params['mu'], (self.num_samples, 1))

    def generate_params(self, min_mean=20, max_mean=10000, min_r=10, max_r=100):
        self.params["mu"] = np.random.uniform(min_mean, max_mean, [self.num_distributions]),
        self.params["r"] = np.round(np.random.uniform(min_r, max_r, [self.num_distributions]))

    def generate_data(self):
        self.data = self.negative_binomial()


class NegativeBinomialWithLinearBias(NegativeBinomial):
    cfg = MatrixSimulator.cfg.copy()

    # cfg.update({
    #     "design": "design.tsv",
    # })

    def __init__(self):
        super().__init__()

    def generate_params(self, num_classes=4, min_bias=0.8, max_bias=1.2, *args):
        super().generate_params(*args)

        design = np.repeat(range(num_classes), self.num_distributions / num_classes)
        design = design[range(self.num_distributions)]
        self.params['design'] = design

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
        bias = self.params['bias_r'][self.params['design']]
        return np.tile(bias, (self.num_samples, 1))

    @property
    def bias_mu(self):
        bias = self.params['bias_mu'][self.params['design']]
        return np.tile(bias, (self.num_samples, 1))

    # def load(self, folder):
    #     super().load(folder)
    #     self.params = pd.read_csv("resources/params.tsv", sep="\t")
    #
    # def save(self, folder):
    #     super().save(folder)
    #
    #     self.params.to_csv(
    #         os.path.join(folder, self.cfg["params"]), sep="\t", index=False)
