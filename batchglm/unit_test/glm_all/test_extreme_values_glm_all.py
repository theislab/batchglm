from typing import List
import unittest
import logging

import batchglm.api as glm
from batchglm.models.base_glm import _Estimator_GLM, _Simulator_GLM

from .external import Test_ExtremValues_GLM, _Test_ExtremValues_GLM_Estim

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_ExtremValues_GLM_ALL_Estim(_Test_ExtremValues_GLM_Estim):

    def __init__(
            self,
            input_data,
            termination,
            quick_scale,
            noise_model
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model == "nb":
                from batchglm.api.models.nb_glm import Estimator
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 10
        provide_optimizers = {"gd": False, "adam": False, "adagrad": False, "rmsprop": False, "nr": True, "irls": True}

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            termination_type=termination,
            noise_model=noise_model
        )
        super().__init__(
            estimator=estimator
        )


class Test_ExtremValues_GLM_ALL(Test_ExtremValues_GLM, unittest.TestCase):
    """
    Test various input data types including outlier features.

    These unit tests cover a range of input data and check whether
    the overall graph works with different inputs. Only one
    training strategy is tested here. The cases tested are:

        - Dense X matrix: test_numpy_dense()
        - Sparse X matrix: test_scipy_sparse()
        - Dense X in anndata: test_anndata_dense()
        - Sparse X in anndata: test_anndata_sparse()
    """
    noise_model: str
    sim: _Simulator_GLM
    _estims: List[_Estimator_GLM]

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.nb_glm import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=50, num_features=2)

    def get_estimator(
            self,
            input_data,
            termination,
            quick_scale
    ):
        return _Test_ExtremValues_GLM_ALL_Estim(
            input_data=input_data,
            termination=termination,
            quick_scale=quick_scale,
            noise_model=self.noise_model
        )

    def _test_low_values(self):
        self.simulate()
        logger.debug("** Running tests for low values")
        logger.debug("** Running a and b training test")
        self._test_low_values_a_and_b()
        logger.debug("** Running a only training test")
        self._test_low_values_a_only()
        logger.debug("** Running b only training test")
        self._test_low_values_b_only()

    def _test_zero_variance(self):
        self.simulate()
        logger.debug("** Running tests for zero-variance features")
        logger.debug("** Running a and b training test")
        self._test_zero_variance_a_and_b()
        logger.debug("** Running a only training test")
        self._test_zero_variance_a_only()
        logger.debug("** Running b only training test")
        self._test_zero_variance_b_only()


class Test_ExtremValues_GLM_NB(
    Test_ExtremValues_GLM_ALL,
    unittest.TestCase
):
    """
    Test various input data types including outlier features for negative binomial noise.
    """

    def test_low_values_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_ExtremValues_GLM_NB.test_low_values_nb()")


        self.noise_model = "nb"
        self._test_low_values()

    def test_zero_variance_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_ExtremValues_GLM_NB.test_zero_variance_nb()")

        self.noise_model = "nb"
        self._test_zero_variance()

if __name__ == '__main__':
    unittest.main()
