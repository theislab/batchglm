import abc
import logging
from typing import List
import unittest
import numpy as np

import batchglm.api as glm
from batchglm.models.base_glm import _EstimatorGLM, _SimulatorGLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_Accuracy_GLM_Estim():

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
            batched,
            acc,
        ):
        self.estimator.initialize()

        # Choose learning rate based on optimizer
        if algo.lower() in ["nr", "nr_tr", "irls", "irls_gd", "irls_tr", "irls_gd_tr"]:
            lr = 1
        else:
            lr = 0.05

        self.estimator.train_sequence(training_strategy=[
            {
                "learning_rate": lr,
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": acc,
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
            threshold_dev_a = 0.2
            threshold_dev_b = 0.2
            threshold_std_a = 1
            threshold_std_b = 2

        mean_dev_a = np.mean(estimator_store.a - self.sim.a.values)
        std_dev_a = np.std(estimator_store.a - self.sim.a.values)
        mean_dev_b = np.mean(estimator_store.b - self.sim.b.values)
        std_dev_b = np.std(estimator_store.b - self.sim.b.values)

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


class Test_Accuracy_GLM(unittest.TestCase, metaclass=abc.ABCMeta):
    _estims: List[_Test_Accuracy_GLM_Estim]

    def setUp(self):
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.estimator.close_session()

    def simulate(self):
        self.simulate1()
        self.simulate2()

    @abc.abstractmethod
    def get_simulator(self):
        pass

    def simulate1(self):
        self.sim1 = self.get_simulator()
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2)
        self.sim1.generate()

    def simulate2(self):
        self.sim2 = self.get_simulator()
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2)
        self.sim2.generate()

    def simulator(self, train_loc):
        if train_loc:
            return self.sim1
        else:
            return self.sim2

    def _basic_test(
            self,
            estimator,
            batched,
            algos
    ):
        for algo in algos:
            logger.info("algorithm: %s" % algo)
            estimator.estimate(
                algo=algo,
                batched=batched,
                acc=1e-6
            )
            estimator_store = estimator.estimator.finalize()
            self._estims.append(estimator)
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
            train_loc,
            train_scale,
            sparse
    ):
        pass

    def _test_full_a_and_b(self, sparse):
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=True,
            sparse=sparse
        )

    def _test_full_a_only(self, sparse):
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=False,
            sparse=sparse
        )

    def _test_full_b_only(self, sparse):
        return self.basic_test(
            batched=False,
            train_loc=False,
            train_scale=True,
            sparse=sparse
        )

    def _test_batched_a_and_b(self, sparse):
        return self.basic_test(
            batched=True,
            train_loc=True,
            train_scale=True,
            sparse=sparse
        )

    def _test_batched_a_only(self, sparse):
        return self.basic_test(
            batched=True,
            train_loc=True,
            train_scale=False,
            sparse=sparse
        )

    def _test_batched_b_only(self, sparse):
        return self.basic_test(
            batched=True,
            train_loc=False,
            train_scale=True,
            sparse=sparse
        )

if __name__ == '__main__':
    unittest.main()
