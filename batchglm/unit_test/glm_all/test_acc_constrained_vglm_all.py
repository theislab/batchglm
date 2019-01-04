from typing import List
import unittest
import logging

import numpy as np

import batchglm.api as glm
from batchglm.models.base_glm import _Estimator_GLM

from .external import Test_AccuracyConstrained_VGLM, _Test_AccuracyConstrained_VGLM_Estim

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_AccuracyConstrained_VGLM_ALL_Estim(_Test_AccuracyConstrained_VGLM_Estim):

    def __init__(
            self,
            simulator,
            quick_scale,
            termination,
            noise_model
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model=="nb":
                from batchglm.api.models.nb_glm import Estimator
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 900
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True, "nr": True, "irls": True}

        # Encode equality constrained on overdetermined confounder coefficient.
        constraints = np.zeros([4,3])
        constraints[0, 0] = 1
        constraints[1, 1] = 1
        constraints[3, 3] = 1
        constraints[4, 3] = -1
        input_data = simulator.input_data
        input_data.design_loc = np.hstack([
            input_data.design_loc,
            input_data.design_loc[:,0]-input_data.design_loc[-1]
        ])
        input_data.design_scale = input_data.design_loc
        input_data.constraints_loc = constraints
        input_data.constraints_scale = constraints

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            termination_type=termination,
            init_a="standard",
            init_b="standard",
            noise_model=noise_model
        )
        super().__init__(
            estimator=estimator,
            simulator=simulator
        )

class Test_AccuracyConstrained_VGLM_ALL(
    Test_AccuracyConstrained_VGLM,
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

        return Simulator(num_observations=10000, num_features=10)

    def basic_test(
            self,
            batched,
            termination,
            train_loc,
            train_scale
    ):
        algos = ["ADAM", "ADAGRAD", "NR", "IRLS"]
        estimator = _Test_AccuracyConstrained_VGLM_ALL_Estim(
            simulator=self.simulator(train_loc=train_loc),
            quick_scale=False if train_scale else True,
            termination=termination,
            noise_model=self.noise_model
        )
        return self._basic_test(
            estimator=estimator,
            batched=batched,
            termination=termination,
            algos=algos
        )

    def _test_full(self):
        self.simulate()
        logger.debug("* Running tests for full data")
        logger.debug("** Running tests for a and b training")
        super()._test_full_a_and_b()
        logger.debug("** Running tests for a only training")
        super()._test_full_a_only()
        logger.debug("** Running tests for b only training")
        super()._test_full_b_only()

    def _test_batched(self):
        self.simulate()
        logger.debug("* Running tests for batched data")
        logger.debug("** Running tests for a and b training")
        super()._test_batched_a_and_b()
        logger.debug("** Running tests for a only training")
        super()._test_batched_a_only()
        logger.debug("** Running tests for b only training")
        super()._test_batched_b_only()


class Test_AccuracyConstrained_VGLM_NB(
    Test_AccuracyConstrained_VGLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial noise.
    """

    def test_full_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_AccuracySizeFactors_GLM_NB.test_full_nb()")

        self.noise_model = "nb"
        self._test_full()

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_AccuracySizeFactors_GLM_NB.test_batched_nb()")

        self.noise_model = "nb"
        self._test_batched()


if __name__ == '__main__':
    unittest.main()
