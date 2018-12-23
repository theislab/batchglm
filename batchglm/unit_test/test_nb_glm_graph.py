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
        quick_scale,
        termination
):
    provide_optimizers = {"gd": False, "adam": False, "adagrad": False, "rmsprop": False, "nr": False}
    provide_optimizers[algo.lower()] = True

    estimator = Estimator(
        input_data,
        batch_size=10,
        quick_scale=quick_scale,
        provide_optimizers=provide_optimizers,
        termination_type=termination
    )
    estimator.initialize()

    estimator.train_sequence(training_strategy=[
            {
                "learning_rate": 0.5 if algo is not "nr" else 1,
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": 1e1,
                "use_batching": batched,
                "optim_algo": algo,
            },
        ])

    return estimator


class NB_GLM_Test_Graph(unittest.TestCase):
    """
    Test whether training graph work.

    Quick tests which simply passes small data sets through
    all possible training graphs to check whether there are graph
    bugs. This is all tested in test_nb_glm_acc.py but this
    set of unit_tests runs much faster and does not abort due
    to accuracy outliers. The training graphs covered are:

    - termination by feature
        - full data model
            - train a and b model: test_full_byfeature_a_and_b()
            - train a model only: test_full_byfeature_a_only()
            - train b model only: test_full_byfeature_b_only()
        - batched data model
            - train a and b model: test_batched_byfeature_a_and_b()
            - train a model only: test_batched_byfeature_a_only()
            - train b model only: test_batched_byfeature_b_only()
    - termination global
        - full data model
            - train a and b model: test_full_global_a_and_b()
            - train a model only: test_full_global_a_only()
            - train b model only: test_full_global_b_only()
        - batched data model
            - train a and b model: test_batched_global_a_and_b()
            - train a model only: test_batched_global_a_only()
            - train b model only: test_batched_global_b_only()
    """
    sim: Simulator
    _estims: List[Estimator]

    def setUp(self):
        self.sim1 = Simulator(num_observations=50, num_features=2)
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2)
        self.sim1.generate()

        self.sim2 = Simulator(num_observations=50, num_features=2)
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2)
        self.sim2.generate()

        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.close_session()

    def test_full_byfeature_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="by_feature"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_full_byfeature_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=True,
                termination="by_feature"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True


    def test_full_byfeature_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="by_feature"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_batched_byfeature_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="by_feature"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_batched_byfeature_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data, algo=algo,
                batched=True,
                quick_scale=True,
                termination="by_feature"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_batched_byfeature_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="by_feature"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_full_global_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="global"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_full_global_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=True,
                termination="global"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_full_global_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="global"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_batched_global_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="global"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_batched_global_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=True,
                termination="global"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True

    def test_batched_global_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="global"
            )
            estimator.finalize()
            self._estims.append(estimator)

        return True


if __name__ == '__main__':
    unittest.main()
