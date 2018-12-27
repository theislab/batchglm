from typing import List

import unittest
import logging

import numpy as np

import batchglm.api as glm
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData

glm.setup_logging(verbosity="ERROR", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def estimate_byfeature(input_data: InputData, quick_scale):
    estimator = Estimator(
        input_data,
        batch_size=500,
        quick_scale=quick_scale,
        termination_type="by_feature"
    )
    estimator.initialize()

    estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": 1e-4,
                "use_batching": False,
                "optim_algo": "Newton",
            },
        ])

    return estimator

def estimate_global(input_data: InputData, quick_scale):
    estimator = Estimator(
        input_data,
        batch_size=500,
        quick_scale=quick_scale,
        termination_type="global"
    )
    estimator.initialize()

    estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "scaled_moving_average",
                "stopping_criteria": 1e-4,
                "loss_window_size": 20,
                "use_batching": False,
                "optim_algo": "Newton",
            },
        ])

    return estimator


class NB_GLM_Test_ExtremValues(unittest.TestCase):
    """
    Test various input data types including outlier features.

    These unit tests check whether the model converges and is
    stable in numeric extremes. Each case is tested for the three
    main training graphs: a and b, a only and b only.
    Cases covered are:

        - Zero variance features:
            - Train a and b model: test_zero_variance_a_and_b()
            - Train a model only: test_zero_variance_a_only()
            - Train b model only: test_zero_variance_b_only()
        - Low mean features: test_low_values()
    """
    sim: Simulator
    _estims: List[Estimator]

    def setUp(self):
        self.sim1 = Simulator(num_observations=1000, num_features=10)
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2)
        self.sim1.generate()

        self.sim2 = Simulator(num_observations=1000, num_features=10)
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2)
        self.sim2.generate()

        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.close_session()

    def test_low_values_a_and_b(self):
        sim = self.sim1.__copy__()
        sim.data.X[:, 0] = 0
        estimator = estimate_byfeature(sim.input_data, quick_scale=False)
        self._estims.append(estimator)
        estimator = estimate_global(sim.input_data, quick_scale=False)
        self._estims.append(estimator)

        return estimator, sim

    def test_low_values_a_only(self):
        sim = self.sim1.__copy__()
        sim.data.X[:, 0] = 0
        estimator = estimate_byfeature(sim.input_data, quick_scale=True)
        self._estims.append(estimator)
        estimator = estimate_global(sim.input_data, quick_scale=True)
        self._estims.append(estimator)

        return estimator, sim

    def test_low_values_b_only(self):
        sim = self.sim2.__copy__()
        sim.data.X[:, 0] = 0
        estimator = estimate_byfeature(sim.input_data, quick_scale=False)
        self._estims.append(estimator)
        estimator = estimate_global(sim.input_data, quick_scale=False)
        self._estims.append(estimator)

        return estimator, sim

    def test_zero_variance_a_and_b(self):
        sim = self.sim1.__copy__()
        sim.data.X[:, 0] = np.exp(sim.a)[0, 0]
        estimator = estimate_byfeature(sim.input_data, quick_scale=False)
        self._estims.append(estimator)
        estimator = estimate_global(sim.input_data, quick_scale=False)
        self._estims.append(estimator)

        return estimator, sim

    def test_zero_variance_a_only(self):
        sim = self.sim1.__copy__()
        sim.data.X[:, 0] = np.exp(sim.a)[0, 0]
        estimator = estimate_byfeature(sim.input_data, quick_scale=True)
        self._estims.append(estimator)
        estimator = estimate_global(sim.input_data, quick_scale=True)
        self._estims.append(estimator)

        return estimator, sim

    def test_zero_variance_b_only(self):
        sim = self.sim2.__copy__()
        sim.data.X[:, 0] = np.exp(sim.a)[0, 0]
        estimator = estimate_byfeature(sim.input_data, quick_scale=False)
        self._estims.append(estimator)
        estimator = estimate_global(sim.input_data, quick_scale=False)
        self._estims.append(estimator)

        return estimator, sim


if __name__ == '__main__':
    unittest.main()
