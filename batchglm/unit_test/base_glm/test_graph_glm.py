import abc
import logging
from typing import List
import unittest

import batchglm.api as glm
from batchglm.models.base_glm import _EstimatorGLM, _SimulatorGLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger("batchglm")


class _Test_Graph_GLM_Estim():

    def __init__(
            self,
            estimator: _EstimatorGLM,
            simulator: _SimulatorGLM,
            algo: str
    ):
        self.estimator = estimator
        self.sim = simulator
        self.algo = algo.lower()

    def estimate(
            self,
            batched
        ):
        self.estimator.initialize()

        self.estimator.train_sequence(training_strategy=[
            {
                "learning_rate": 1,
                "convergence_criteria": "all_converged",
                "stopping_criteria": 1e1,
                "use_batching": batched,
                "optim_algo": self.algo,
            },
        ])


class Test_Graph_GLM(unittest.TestCase, metaclass=abc.ABCMeta):
    _estims: List[_Test_Graph_GLM_Estim]

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

    @abc.abstractmethod
    def basic_test_one_algo(
            self,
            batched,
            train_loc,
            train_scale,
            algo,
            sparse
    ):
        pass

    def _basic_test_one_algo(
            self,
            estimator,
            batched
    ):
        estimator.estimate(batched=batched)
        estimator.estimator.finalize()
        self._estims.append(estimator)

        return True

    def basic_test(
            self,
            batched,
            train_loc,
            train_scale,
            sparse
    ):
        #algos = ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR", "NR_TR", "IRLS", "IRLS_GD", "IRLS_TR", "IRLS_GD_TR"]
        algos = ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR", "NR_TR"]
        for algo in algos:
            logger.info("algorithm: %s" % algo)
            self.basic_test_one_algo(
                batched=batched,
                train_loc=train_loc,
                train_scale=train_scale,
                algo=algo,
                sparse=sparse
            )

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
