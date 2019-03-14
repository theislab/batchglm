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
            noise_model,
            constraints_loc,
            constraints_scale
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model=="nb":
                from batchglm.api.models.glm_nb import Estimator, InputData
            elif noise_model=="norm":
                from batchglm.api.models.glm_norm import Estimator, InputData
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 900
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": True, "nr_tr": True,
                              "irls": False, "irls_gd": False, "irls_tr": False, "irls_gd_tr": False}

        input_data = simulator.input_data
        design_loc = np.hstack([
            input_data.design_loc.values,
            np.expand_dims(input_data.design_loc.values[:,0]-input_data.design_loc.values[:,-1], axis=-1)
        ])
        design_scale = design_loc.copy()
        input_data = InputData.new(
            data=simulator.X,
            design_loc=design_loc,
            design_scale=design_scale,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale
        )

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            provide_batched=True,
            init_a="standard",
            init_b="standard"
        )
        super().__init__(
            estimator=estimator,
            simulator=simulator
        )

class Test_AccuracyConstrained_VGLM_ALL(
    Test_AccuracyConstrained_VGLM,
    unittest.TestCase
):
    noise_model: str
    _estims: List[_Estimator_GLM]

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.glm_nb import Simulator
            elif self.noise_model == "norm":
                from batchglm.api.models.glm_norm import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=10000, num_features=10)

    def basic_test(
            self,
            batched,
            train_loc,
            train_scale
    ):
        #algos = ["ADAM", "NR_TR", "IRLS_GD_TR"]
        algos = ["NR_TR"]
        # Encode equality constrained on overdetermined confounder coefficient.
        if train_loc:
            constraints = np.zeros([4, 3])
            constraints[0, 0] = 1
            constraints[1, 1] = 1
            constraints[2, 2] = 1
            constraints[3, 2] = -1
        else:
            constraints = np.zeros([3, 2])
            constraints[0, 0] = 1
            constraints[1, 1] = 1
            constraints[2, 1] = -1

        estimator = _Test_AccuracyConstrained_VGLM_ALL_Estim(
            simulator=self.simulator(train_loc=train_loc),
            quick_scale=False if train_scale else True,
            noise_model=self.noise_model,
            constraints_loc=constraints,
            constraints_scale=constraints,
        )
        return self._basic_test(
            estimator=estimator,
            batched=batched,
            algos=algos
        )

    def _test_full(self):
        self.simulate()
        logger.debug("* Running tests for full data")
        super()._test_full_a_and_b()
        super()._test_full_a_only()
        super()._test_full_b_only()

    def _test_batched(self):
        self.simulate()
        logger.debug("* Running tests for batched data")
        super()._test_batched_a_and_b()
        super()._test_batched_a_only()
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
        logger.error("Test_AccuracyConstrained_VGLM_NB.test_full_nb()")

        self.noise_model = "nb"
        self._test_full()

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_AccuracyConstrained_VGLM_NB.test_batched_nb()")

        self.noise_model = "nb"
        self._test_batched()

class Test_AccuracyConstrained_VGLM_NORM(
    Test_AccuracyConstrained_VGLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for normal distributed noise.
    """

    def test_full_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_AccuracyConstrained_VGLM_NORM.test_full_norm()")

        self.noise_model = "norm"
        self._test_full()

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_AccuracyConstrained_VGLM_NORM.test_batched_norm()")

        self.noise_model = "norm"
        self._test_batched()


if __name__ == '__main__':
    unittest.main()
