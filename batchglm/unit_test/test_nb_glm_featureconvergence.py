from typing import List

import os
# import sys
import unittest
import tempfile
import logging

import numpy as np
import scipy.sparse

import batchglm.api as glm
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData

glm.setup_logging(verbosity="INFO", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.INFO)


def estimate_adam_full(input_data: InputData):

    estimator = Estimator(input_data, batch_size=500,
                          provide_optimizers={"gd": True, "adam": True, "adagrad": False, "rmsprop": False, "nr": True},
                          convergence_type="by_feature")
    estimator.initialize()

    estimator.train_sequence(training_strategy=[
            {
                "learning_rate": 0.5,
                "convergence_criteria": "all_converged",
                "stopping_criteria": 1e-1,
                "use_batching": False,
                "optim_algo": "ADAM",
            },
        ])

    return estimator

def estimate_adam_batched(input_data: InputData):

    estimator = Estimator(input_data, batch_size=500)
    estimator.initialize()

    estimator.train_sequence(training_strategy=[
            {
                "learning_rate": 0.5,
                "convergence_criteria": "all_converged",
                "stopping_criteria": 1e-1,
                "use_batching": True,
                "optim_algo": "ADAM",
            },
        ])

    return estimator

def estimate_nr_full(input_data: InputData):

    estimator = Estimator(input_data, batch_size=500)
    estimator.initialize()

    estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "all_converged",
                "stopping_criteria": 1e-1,
                "use_batching": False,
                "optim_algo": "newton",
            },
        ])

    return estimator

def estimate_nr_batched(input_data: InputData):

    estimator = Estimator(input_data, batch_size=500)
    estimator.initialize()

    estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "all_converged",
                "stopping_criteria": 1e-1,
                "use_batching": True,
                "optim_algo": "newton",
            },
        ])

    return estimator

class NB_GLM_Test(unittest.TestCase):
    sim: Simulator

    _estims: List[Estimator]

    def setUp(self):
        self.sim = Simulator(num_observations=1000, num_features=7)
        self.sim.generate()
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.close_session()

    def test_adam_full(self):
        sim = self.sim.__copy__()

        estimator = estimate_adam_full(sim.input_data)
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()
        return estimator, sim

    def test_adam_batch(self):
        sim = self.sim.__copy__()

        estimator = estimate_adam_batched(sim.input_data)
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()
        return estimator, sim

    def test_nr_full(self):
        sim = self.sim.__copy__()

        estimator = estimate_nr_full(sim.input_data)
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()
        return estimator, sim

    def test_nr_batch(self):
        sim = self.sim.__copy__()

        estimator = estimate_nr_batched(sim.input_data)
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()
        return estimator, sim


if __name__ == '__main__':
    unittest.main()
