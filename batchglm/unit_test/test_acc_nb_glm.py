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
        termination,
        acc,
):
    provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True, "nr": True}

    estimator = Estimator(
        input_data,
        batch_size=900,
        quick_scale=quick_scale,
        provide_optimizers=provide_optimizers,
        termination_type=termination
    )
    estimator.initialize()

    # Choose learning rate based on optimizer
    if algo.lower() is "nr":
        lr = 1
    elif algo.lower() is "gd":
        lr = 0.05
    else:
        lr = 0.5
    estimator.train_sequence(training_strategy=[
            {
                "learning_rate": lr,
                "convergence_criteria": "all_converged",
                "stopping_criteria": acc,
                "use_batching": batched,
                "optim_algo": algo,
            },
        ])

    return estimator

def eval_estimation(
        estimator,
        sim,
        threshold_dev_a = 0.01,
        threshold_dev_b = 0.01,
        threshold_std_a = 0.5,
        threshold_std_b = 0.5
):
    mean_dev_a = np.mean(estimator.a.values - sim.a.values)
    std_dev_a = np.std(estimator.a.values - sim.a.values)
    mean_dev_b = np.mean(estimator.b.values - sim.b.values)
    std_dev_b = np.std(estimator.b.values - sim.b.values)

    print("\n")
    print("mean_dev_a %f" % mean_dev_a)
    print("std_dev_a %f" % std_dev_a)
    print("mean_dev_b %f" % mean_dev_b)
    print("std_dev_b %f" % std_dev_b)

    if np.abs(mean_dev_a) < threshold_dev_a and \
            np.abs(mean_dev_b) < threshold_dev_b and \
            std_dev_a < threshold_std_a and \
            std_dev_b < threshold_std_b:
        return True
    else:
        return False

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
        self.sim1 = Simulator(num_observations=1000, num_features=200)
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2)
        self.sim1.generate()

        self.sim2 = Simulator(num_observations=1000, num_features=200)
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2)
        self.sim2.generate()

        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.close_session()

    def test_full_byfeature_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="by_feature",
                acc=1e-4
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_batched_byfeature_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "RMSPROP", "NR"]: # GD does not work well yet
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="by_feature",
                acc=1e-1
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                threshold_dev_a=1,
                threshold_dev_b=1,
                threshold_std_a=1,
                threshold_std_b=1
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_full_global_a_and_b(self):
        sim = self.sim1.__copy__()

        all_exact = True
        for algo in ["ADAM", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="global",
                acc=1e-3
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_batched_global_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "RMSPROP", "NR"]: # GD does not work well yet
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="global",
                acc=1e-1
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                threshold_dev_a=1,
                threshold_dev_b=1,
                threshold_std_a=1,
                threshold_std_b=1
            )
            assert success, "%s did not yield exact results" % algo

        return True

if __name__ == '__main__':
    unittest.main()
