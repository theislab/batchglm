import abc
import logging
from typing import List
import unittest
import numpy as np

from batchglm.models.base_glm import _EstimatorGLM, _SimulatorGLM


class _Test_AccuracySizeFactors_GLM_Estim():

    def __init__(
            self,
            estimator: _EstimatorGLM,
            simulator: _SimulatorGLM
    ):
        self.estimator = estimator
        self.sim = simulator

    def estimate(
            self,
            algo,
            batched
        ):
        self.estimator.initialize()

        # Choose learning rate based on optimizer
        if algo.lower() in ["nr", "nr_tr", "irls", "irls_gd", "irls_tr", "irls_gd_tr"]:
            lr = 1
        elif algo.lower() == "gd":
            lr = 0.05
        else:
            lr = 0.5

        self.estimator.train_sequence(training_strategy=[
            {
                "learning_rate": lr,
                "convergence_criteria": "all_converged",
                "use_batching": batched,
                "optim_algo": algo,
            },
        ])

    def eval_estimation(
            self,
            estimator_store,
            batched
    ):
        if batched:
            threshold_dev_a = 0.5
            threshold_dev_b = 0.5
            threshold_std_a = 1
            threshold_std_b = 20
        else:
            threshold_dev_a = 0.3
            threshold_dev_b = 0.3
            threshold_std_a = 1
            threshold_std_b = 2

        mean_dev_a = np.mean(estimator_store.a_var - self.sim.a_var.values)
        std_dev_a = np.std(estimator_store.a_var - self.sim.a_var.values)
        mean_dev_b = np.mean(estimator_store.b_var - self.sim.b_var.values)
        std_dev_b = np.std(estimator_store.b_var - self.sim.b_var.values)

        logging.getLogger("batchglm").info("mean_dev_a %f" % mean_dev_a)
        logging.getLogger("batchglm").info("std_dev_a %f" % std_dev_a)
        logging.getLogger("batchglm").info("mean_dev_b %f" % mean_dev_b)
        logging.getLogger("batchglm").info("std_dev_b %f" % std_dev_b)

        if np.abs(mean_dev_a) < threshold_dev_a and \
                np.abs(mean_dev_b) < threshold_dev_b and \
                std_dev_a < threshold_std_a and \
                std_dev_b < threshold_std_b:
            return True
        else:
            return False


class Test_AccuracySizeFactors_GLM(unittest.TestCase, metaclass=abc.ABCMeta):

    def simulate(self):
        self._simulate()

    @abc.abstractmethod
    def get_simulator(self):
        pass

    def _simulate(self):
        sim = self.get_simulator()
        sim.generate_sample_description(num_batches=2, num_conditions=2)
        sim.generate_params()
        sim.size_factors = np.random.uniform(0.1, 2, size=sim.num_observations)
        sim.generate_data()
        logging.getLogger("batchglm").debug(" Size factor standard deviation % f" % np.std(sim.size_factors.data))
        self.sim = sim

    def _basic_test(
            self,
            estimator,
            batched,
            algos
    ):
        for algo in algos:
            estimator.estimate(
                algo=algo,
                batched=batched
            )
            estimator_store = estimator.estimator.finalize()
            success = estimator.eval_estimation(
                estimator_store=estimator_store,
                batched=False
            )
            assert success, "%s did not yield exact results" % algo

        return True

    @abc.abstractmethod
    def basic_test(
            self,
            batched,
            train_scale,
            sparse
    ):
        pass

    def _test_full_a_and_b(self, sparse):
        return self.basic_test(
            batched=False,
            train_scale=True,
            sparse=sparse
        )

    def _test_full_a_only(self, sparse):
        return self.basic_test(
            batched=False,
            train_scale=False,
            sparse=sparse
        )

    def _test_batched_a_and_b(self, sparse):
        return self.basic_test(
            batched=True,
            train_scale=True,
            sparse=sparse
        )

    def _test_batched_a_only(self, sparse):
        return self.basic_test(
            batched=True,
            train_scale=False,
            sparse=sparse
        )
