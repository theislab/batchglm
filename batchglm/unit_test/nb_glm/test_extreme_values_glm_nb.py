from typing import List

import unittest
import logging

import batchglm.api as glm
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData

from .external import Test_ExtremValues_GLM, _Test_ExtremValues_GLM_Estim

glm.setup_logging(verbosity="ERROR", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class _Test_ExtremValues_GLM_NB_Estim(_Test_ExtremValues_GLM_Estim):

    def __init__(
            self,
            input_data,
            termination,
            quick_scale
    ):
        batch_size = 10
        provide_optimizers = {"gd": False, "adam": False, "adagrad": False, "rmsprop": False, "nr": True}

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            termination_type=termination
        )
        super().__init__(
            estimator=estimator
        )


class Test_ExtremValues_GLM_NB(Test_ExtremValues_GLM, unittest.TestCase):
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
    sim: Simulator
    _estims: List[Estimator]

    def setUp(self):
        self.simulate1()
        self.simulate2()
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.estimator.close_session()

    def simulate1(self):
        self._simulate1(sim=Simulator(num_observations=50, num_features=2))

    def simulate2(self):
        self._simulate2(sim=Simulator(num_observations=50, num_features=2))

    def get_estimator(
            self,
            input_data,
            termination,
            quick_scale
    ):
        return _Test_ExtremValues_GLM_NB_Estim(
            input_data=input_data,
            termination=termination,
            quick_scale=quick_scale
        )

    def test_low_values_a_and_b(self):
        self._test_low_values_a_and_b()

    def test_low_values_a_only(self):
        self._test_low_values_a_only()

    def test_low_values_b_only(self):
        self._test_low_values_b_only()

    def test_zero_variance_a_and_b(self):
        self._test_zero_variance_a_and_b()

    def test_zero_variance_a_only(self):
        self._test_zero_variance_a_only()

    def test_zero_variance_b_only(self):
        self._test_zero_variance_b_only()

if __name__ == '__main__':
    unittest.main()
