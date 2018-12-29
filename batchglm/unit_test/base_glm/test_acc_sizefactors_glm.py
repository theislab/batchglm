import abc
import logging
from typing import List
import unittest
import numpy as np

import batchglm.api as glm
from batchglm.models.base_glm import _Estimator_GLM, _Simulator_GLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_AccuracySizeFactors_GLM_Estim():

    def __init__(
            self,
            estimator: _Estimator_GLM,
            simulator: _Simulator_GLM
    ):
        self.estimator = estimator
        self.sim = simulator

    def estimate(
            self,
            algo,
            batched,
            termination,
            acc,
        ):
        self.estimator.initialize()

        # Choose learning rate based on optimizer
        if algo.lower() == "nr":
            lr = 1
        elif algo.lower() == "gd":
            lr = 0.05
        else:
            lr = 0.5

        self.estimator.train_sequence(training_strategy=[
            {
                "learning_rate": lr,
                "convergence_criteria": "all_converged_ll" if termination == "by_feature" else "scaled_moving_average",
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
            threshold_dev_a = 0.3
            threshold_dev_b = 0.3
            threshold_std_a = 1
            threshold_std_b = 2

        mean_dev_a = np.mean(estimator_store.a.values - self.sim.a.values)
        std_dev_a = np.std(estimator_store.a.values - self.sim.a.values)
        mean_dev_b = np.mean(estimator_store.b.values - self.sim.b.values)
        std_dev_b = np.std(estimator_store.b.values - self.sim.b.values)

        logger.warning("mean_dev_a %f" % mean_dev_a)
        logger.warning("std_dev_a %f" % std_dev_a)
        logger.warning("mean_dev_b %f" % mean_dev_b)
        logger.warning("std_dev_b %f" % std_dev_b)

        if np.abs(mean_dev_a) < threshold_dev_a and \
                np.abs(mean_dev_b) < threshold_dev_b and \
                std_dev_a < threshold_std_a and \
                std_dev_b < threshold_std_b:
            return True
        else:
            return False


class Test_AccuracySizeFactors_GLM(unittest.TestCase, metaclass=abc.ABCMeta):
    """
    Test whether optimizers yield exact results if size factors are used.

    Accuracy is evaluted via deviation of simulated ground truth.
    The unit tests test individual training graphs and multiple optimizers
    (incl. one tensorflow internal optimizer and newton-rhapson)
    for each training graph. The training graphs tested are as follows:

    - termination by feature
        - full data model
            - train a and b model: test_full_a_and_b()
            - train a model only: test_full_a_only()
            - train b model only: test_full_b_only()
        - batched data model
            - train a and b model: test_batched_a_and_b()
            - train a model only: test_batched_a_only()
            - train b model only: test_batched_b_only()

    The unit tests throw an assertion error if the required accuracy is
    not met. Accuracy thresholds are fairly lenient so that unit_tests
    pass even with noise inherent in fast optimisation and random
    initialisation in simulation. Still, large biases (i.e. graph errors)
    should be discovered here.
    """
    _estims: List[_Test_AccuracySizeFactors_GLM_Estim]

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

    def _simulate(self, num_batches):
        sim = self.get_simulator()
        sim.generate_sample_description(num_batches=num_batches, num_conditions=2)
        sim.generate_params()
        sim.size_factors = np.random.uniform(0.1, 2, size=sim.num_observations)
        sim.generate_data()
        logger.debug(" Size factor standard deviation % f" % np.std(sim.size_factors.data))
        return sim

    def simulate1(self):
        self.sim1 = self._simulate(num_batches=2)

    def simulate2(self):
        self.sim2 = self._simulate(num_batches=0)

    def simulator(self, train_loc):
        if train_loc:
            return self.sim1
        else:
            return self.sim2

    def _basic_test(
            self,
            estimator,
            batched,
            termination,
            algos
    ):
        for algo in algos:
            logger.warning("algorithm: %s" % algo)
            estimator.estimate(
                algo=algo,
                batched=batched,
                termination=termination,
                acc=1e-6 if algo == "NR" else 1e-3
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
            termination,
            train_loc,
            train_scale
    ):
        pass

    def _test_full_a_and_b(self):
        return self.basic_test(
            batched=False,
            termination="by_feature",
            train_loc=True,
            train_scale=True
        )

    def _test_full_a_only(self):
        return self.basic_test(
            batched=False,
            termination="by_feature",
            train_loc=True,
            train_scale=False
        )

    def _test_full_b_only(self):
        return self.basic_test(
            batched=False,
            termination="by_feature",
            train_loc=False,
            train_scale=True
        )

    def _test_batched_a_and_b(self):
        return self.basic_test(
            batched=True,
            termination="by_feature",
            train_loc=True,
            train_scale=True
        )

    def _test_batched_a_only(self):
        return self.basic_test(
            batched=True,
            termination="by_feature",
            train_loc=True,
            train_scale=False
        )

    def _test_batched_b_only(self):
        return self.basic_test(
            batched=True,
            termination="by_feature",
            train_loc=False,
            train_scale=True
        )


if __name__ == '__main__':
    unittest.main()
