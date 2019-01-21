import abc
import logging
from typing import List
import unittest
import numpy as np

import batchglm.api as glm
from batchglm.models.base_glm import _Estimator_GLM, _Simulator_GLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_AccuracyAnalytic_GLM_Estim():

    def __init__(
            self,
            estimator: _Estimator_GLM,
            simulator: _Simulator_GLM
    ):
        self.estimator = estimator
        self.sim = simulator

    def estimate(self):
        self.estimator.initialize()
        self.estimator.train_sequence(training_strategy=[
            {
                "learning_rate": 1,
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": 1e-6,
                "use_batching": False,
                "optim_algo": "nr_tr",
            },
        ])

    def eval_estimation_a(
            self,
            estimator_store
    ):
        threshold_dev = 1e-2
        threshold_std = 1e-1

        mean_dev = np.mean(estimator_store.a.values - self.sim.a.values)
        std_dev = np.std(estimator_store.a.values - self.sim.a.values)

        logger.info("mean_dev_a %f" % mean_dev)
        logger.info("std_dev_a %f" % std_dev)

        if np.abs(mean_dev) < threshold_dev and \
                std_dev < threshold_std:
            return True
        else:
            return False

    def eval_estimation_b(
            self,
            estimator_store
    ):
        threshold_dev = 1e-2
        threshold_std = 1e-1

        mean_dev = np.mean(estimator_store.b.values - self.sim.b.values)
        std_dev = np.std(estimator_store.b.values - self.sim.b.values)

        logger.info("mean_dev_b %f" % mean_dev)
        logger.info("std_dev_b %f" % std_dev)

        if np.abs(mean_dev) < threshold_dev and \
                std_dev < threshold_std:
            return True
        else:
            return False


class Test_AccuracyAnalytic_GLM(unittest.TestCase, metaclass=abc.ABCMeta):
    """
    Test whether analytic solutions yield exact results.

    Accuracy is evaluted via deviation of simulated ground truth.
    The analytic solution is independent of the optimizer, batching and
    termination mode and therefore only tested for one example each.

    - termination by feature
        - full data model
            - train a model only: test_a_analytic()
            - train b model only: test_b_analytic()

    The unit tests throw an assertion error if the required accurcy is
    not met.
    """
    _estims: List[_Test_AccuracyAnalytic_GLM_Estim]

    def setUp(self):
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.estimator.close_session()

    @abc.abstractmethod
    def get_simulator(self):
        pass

    def simulate(self):
        self.sim = self.get_simulator()
        self.sim.generate_sample_description(num_batches=1, num_conditions=2)
        self.sim.generate_params(
            rand_fn_ave=lambda shape: np.random.uniform(1e5, 2*1e5, shape),
        )
        self.sim.generate_data()

    @abc.abstractmethod
    def get_estimator(self, train_scale, sparse):
        pass

    def _test_a_and_b_closed(self, sparse):
        estimator = self.get_estimator(train_scale=False, sparse=sparse)
        estimator.estimate()
        estimator_store = estimator.estimator.finalize()
        self._estims.append(estimator)
        success = estimator.eval_estimation_a(
            estimator_store=estimator_store
        )
        assert success, "closed form for a model was inaccurate"
        success = estimator.eval_estimation_b(
            estimator_store=estimator_store
        )
        assert success, "closed form for b model was inaccurate"
        return True


    def _test_a_closed(self, sparse):
        estimator = self.get_estimator(train_scale=False, sparse=sparse)
        estimator.estimate()
        estimator_store = estimator.estimator.finalize()
        self._estims.append(estimator)
        success = estimator.eval_estimation_a(
            estimator_store=estimator_store
        )
        assert success, "closed form for a model was inaccurate"
        return True

    def _test_b_closed(self, sparse):
        estimator = self.get_estimator(train_scale=False, sparse=sparse)
        estimator.estimate()
        estimator_store = estimator.estimator.finalize()
        self._estims.append(estimator)
        success = estimator.eval_estimation_b(
            estimator_store=estimator_store
        )
        assert success, "closed form for b model was inaccurate"
        return True


if __name__ == '__main__':
    unittest.main()
