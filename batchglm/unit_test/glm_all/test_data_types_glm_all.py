from typing import List
import unittest
import logging

import batchglm.api as glm
from batchglm.models.base_glm import _EstimatorGLM, _SimulatorGLM

from .external import Test_DataTypes_GLM, _Test_DataTypes_GLM_Estim

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_DataTypes_GLM_ALL_Estim(_Test_DataTypes_GLM_Estim):

    def __init__(
            self,
            input_data,
            noise_model
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model=="nb":
                from batchglm.api.models.glm_nb import Estimator
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 10
        provide_optimizers = {"gd": False, "adam": False, "adagrad": False, "rmsprop": False,
                              "nr": True, "nr_tr": True,
                              "irls": True, "irls_gd": True, "irls_tr": True, "irls_gd_tr": True}

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=True,
            provide_optimizers=provide_optimizers,
            provide_batched=True
        )
        super().__init__(
            estimator=estimator,
            algo="IRLS_GD_TR"
        )


class Test_DataTypes_GLM_ALL(Test_DataTypes_GLM, unittest.TestCase):
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
    sim: _SimulatorGLM
    _estims: List[_EstimatorGLM]

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.glm_nb import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=50, num_features=2)

    def input_data(
            self,
            data,
            design_loc,
            design_scale
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.glm_nb import InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        return InputDataGLM.new(
            data=data,
            design_loc=design_loc,
            design_scale=design_scale,
        )

    def get_estimator(
            self,
            input_data
    ):
        return _Test_DataTypes_GLM_ALL_Estim(
            input_data=input_data,
            noise_model=self.noise_model
        )


class Test_DataTypes_GLM_NB(
    Test_DataTypes_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether training graphs work for negative binomial noise.
    """

    def test_standard(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_DataTypes_GLM_NB.test_standard_nb()")

        self.noise_model = "nb"
        self.simulate()
        self._test_numpy(sparse=False)
        self._test_numpy(sparse=True)

        return True

    def test_anndata(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_DataTypes_GLM_NB.test_anndata_nb()")

        self.noise_model = "nb"
        self.simulate()
        self._test_anndata(sparse=False)
        self._test_anndata(sparse=True)
        self._test_anndata_raw(sparse=False)
        self._test_anndata_raw(sparse=True)

        return True

if __name__ == '__main__':
    unittest.main()
