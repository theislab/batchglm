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

glm.setup_logging(verbosity="DEBUG", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.INFO)


def estimate(
        input_data: InputData,
        algo,
        batched,
        quick_scale
):
    provide_optimizers = {"gd": False, "adam": False, "adagrad": False, "rmsprop": False, "nr": False}
    provide_optimizers[algo.lower()] = True

    estimator = Estimator(
        input_data,
        batch_size=50,
        quick_scale=quick_scale,
        provide_optimizers=provide_optimizers,
        termination_type="by_feature"
    )
    estimator.initialize()

    estimator.train_sequence(training_strategy=[
            {
                "learning_rate": 0.5,
                "convergence_criteria": "all_converged",
                "stopping_criteria": 1e-1,
                "use_batching": batched,
                "optim_algo": algo,
            },
        ])

    return estimator


class NB_GLM_Test(unittest.TestCase):
    """
    Test whether feature-wise termination works on all optimizer settings.
    The unit tests cover a and b traing, a-only traing and b-only training
    for both updated on the full data and on the batched data.
    For each scenario, all implemented optimizers are individually required
    and used once.
    """
    sim: Simulator

    _estims: List[Estimator]

    def setUp(self):
        self.sim1 = Simulator(num_observations=200, num_features=7)
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2)
        self.sim1.generate()

        self.sim2 = Simulator(num_observations=200, num_features=7)
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2)
        self.sim2.generate()

        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.close_session()

    def test_full_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            estimator = estimate(sim.input_data, algo=algo, batched=False, quick_scale=False)
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_full_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            estimator = estimate(sim.input_data, algo=algo, batched=False, quick_scale=True)
            estimator.finalize()
            self._estims.append(estimator)

        return True


    def test_full_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            estimator = estimate(sim.input_data, algo=algo, batched=False, quick_scale=False)
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_batched_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            estimator = estimate(sim.input_data, algo=algo, batched=True, quick_scale=False)
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_batched_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            estimator = estimate(sim.input_data, algo=algo, batched=True, quick_scale=True)
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_batched_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            estimator = estimate(sim.input_data, algo=algo, batched=True, quick_scale=False)
            estimator.finalize()
            self._estims.append(estimator)

        return True


if __name__ == '__main__':
    unittest.main()
