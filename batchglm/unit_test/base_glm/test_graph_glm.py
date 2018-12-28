import abc
import logging
from typing import List
import unittest

import batchglm.api as glm
from batchglm.models.base_glm import _Estimator_GLM, _Simulator_GLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_Graph_GLM_Estim():

    def __init__(
            self,
            estimator: _Estimator_GLM,
            simulator: _Simulator_GLM,
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
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": 1e1,
                "use_batching": batched,
                "optim_algo": self.algo,
            },
        ])


class Test_Graph_GLM(unittest.TestCase, metaclass=abc.ABCMeta):
    """
    Test whether training graph work.

    Quick tests which simply passes small data sets through
    all possible training graphs to check whether there are graph
    bugs. This is all tested in test_acc_glm.py but this
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
            termination,
            train_loc,
            train_scale,
            algo
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

    @abc.abstractmethod
    def basic_test(
            self,
            batched,
            termination,
            train_loc,
            train_scale
    ):
        pass

    def _basic_test(
            self,
            batched,
            termination,
            train_loc,
            train_scale,
            algos
    ):
        for algo in algos:
            logger.debug("algorithm: %s" % algo)
            self.basic_test_one_algo(
                batched=batched,
                termination=termination,
                train_loc=train_loc,
                train_scale=train_scale,
                algo=algo
            )

    def _test_full_byfeature_a_and_b(self):
        return self.basic_test(
            batched=False,
            termination="by_feature",
            train_loc=True,
            train_scale=True
        )

    def _test_full_byfeature_a_only(self):
        return self.basic_test(
            batched=False,
            termination="by_feature",
            train_loc=True,
            train_scale=False
        )

    def _test_full_byfeature_b_only(self):
        return self.basic_test(
            batched=False,
            termination="by_feature",
            train_loc=False,
            train_scale=True
        )

    def _test_batched_byfeature_a_and_b(self):
        return self.basic_test(
            batched=True,
            termination="by_feature",
            train_loc=True,
            train_scale=True
        )

    def _test_batched_byfeature_a_only(self):
        return self.basic_test(
            batched=True,
            termination="by_feature",
            train_loc=True,
            train_scale=False
        )

    def _test_batched_byfeature_b_only(self):
        return self.basic_test(
            batched=True,
            termination="by_feature",
            train_loc=False,
            train_scale=True
        )

    def _test_full_global_a_and_b(self):
        return self.basic_test(
            batched=False,
            termination="global",
            train_loc=True,
            train_scale=True
        )

    def _test_full_global_a_only(self):
        return self.basic_test(
            batched=False,
            termination="global",
            train_loc=True,
            train_scale=False
        )

    def _test_full_global_b_only(self):
        return self.basic_test(
            batched=False,
            termination="global",
            train_loc=False,
            train_scale=True
        )

    def _test_batched_global_a_and_b(self):
        return self.basic_test(
            batched=True,
            termination="global",
            train_loc=True,
            train_scale=True
        )

    def _test_batched_global_a_only(self):
        return self.basic_test(
            batched=True,
            termination="global",
            train_loc=True,
            train_scale=False
        )

    def _test_batched_global_b_only(self):
        return self.basic_test(
            batched=True,
            termination="global",
            train_loc=False,
            train_scale=True
        )

if __name__ == '__main__':
    unittest.main()
