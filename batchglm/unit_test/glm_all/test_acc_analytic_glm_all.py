from typing import List
import unittest
import logging

import batchglm.api as glm
from batchglm.models.base_glm import _Estimator_GLM

from .external import Test_AccuracyAnalytic_GLM, _Test_AccuracyAnalytic_GLM_Estim

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_AccuracyAnalytic_GLM_ALL_Estim(_Test_AccuracyAnalytic_GLM_Estim):

    def __init__(
            self,
            simulator,
            train_scale,
            noise_model
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model=="nb":
                from batchglm.api.models.nb_glm import Estimator
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 500
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True, "nr": True}
        estimator = Estimator(
            input_data=simulator.input_data,
            batch_size=batch_size,
            quick_scale=train_scale,
            provide_optimizers=provide_optimizers,
            termination_type="by_feature"
        )
        super().__init__(
            estimator=estimator,
            simulator=simulator
        )

class Test_AccuracyAnalytic_GLM_ALL(
    Test_AccuracyAnalytic_GLM,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results.

    Accuracy is evaluted via deviation of simulated ground truth.
    The unit tests test individual training graphs and multiple optimizers
    (incl. one tensorflow internal optimizer and newton-rhapson)
    for each training graph. The training graphs tested are as follows:

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

    The unit tests throw an assertion error if the required accurcy is
    not met. Accuracy thresholds are fairly lenient so that unit_tests
    pass even with noise inherent in fast optimisation and random
    initialisation in simulation. Still, large biases (i.e. graph errors)
    should be discovered here.
    """
    noise_model: str
    _estims: List[_Estimator_GLM]

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.nb_glm import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(
            num_observations=100000,
            num_features=2
        )

    def get_estimator(self, train_scale):
        return _Test_AccuracyAnalytic_GLM_ALL_Estim(
            simulator=self.sim,
            train_scale=train_scale,
            noise_model=self.noise_model
        )

    def _test_a_and_b(self):
        self.simulate()
        logger.debug("* Running tests for closed form of a and b model")
        self._test_a_and_b_closed()

    def _test_a_only(self):
        self.simulate()
        logger.debug("* Running tests for closed form of a model")
        self._test_a_closed()

    def _test_b_only(self):
        self.simulate()
        logger.debug("* Running tests for closed form of b model")
        self._test_b_closed()


class Test_AccuracyAnalytic_GLM_NB(
    Test_AccuracyAnalytic_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial noise.
    """

    def test_a_and_b(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)

        logger.debug("* Checking closed form of MLE for a and MME for b model.")
        self.noise_model = "nb"
        self._test_a_and_b()


if __name__ == '__main__':
    unittest.main()
