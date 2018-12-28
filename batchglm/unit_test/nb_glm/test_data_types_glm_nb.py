from typing import List

import unittest
import logging

import batchglm.api as glm
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData

from .external import Test_DataTypes_GLM, _Test_DataTypes_GLM_Estim

glm.setup_logging(verbosity="ERROR", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class _Test_DataTypes_GLM_NB_Estim(_Test_DataTypes_GLM_Estim):

    def __init__(
            self,
            input_data
    ):
        batch_size = 10
        provide_optimizers = {"gd": False, "adam": False, "adagrad": False, "rmsprop": False, "nr": True}

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=True,
            provide_optimizers=provide_optimizers,
            termination_type="by_feature"
        )
        super().__init__(
            estimator=estimator
        )


class Test_DataTypes_GLM_NB(Test_DataTypes_GLM, unittest.TestCase):
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
        self.simulate()
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.estimator.close_session()

    def simulate(self):
        self._simulate(sim=Simulator(num_observations=50, num_features=2))

    def input_data(
            self,
            data,
            design_loc,
            design_scale
    ):
        return InputData.new(
            data=data,
            design_loc=design_loc,
            design_scale=design_scale,
        )

    def get_estimator(
            self,
            input_data
    ):
        return _Test_DataTypes_GLM_NB_Estim(input_data=input_data)

    def test_numpy_dense(self):
        self._test_numpy_dense()

    def test_scipy_sparse(self):
        self._test_scipy_sparse()

    def test_anndata_dense(self):
        self._test_anndata_dense()

    def test_anndata_sparse(self):
        self._test_anndata_sparse()


if __name__ == '__main__':
    unittest.main()
