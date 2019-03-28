from typing import List
import unittest
import logging
import scipy.sparse

import batchglm.api as glm
from batchglm.models.base_glm import _Estimator_GLM

from .external import Test_Graph_GLM, _Test_Graph_GLM_Estim

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_Graph_GLM_ALL_Estim(_Test_Graph_GLM_Estim):

    def __init__(
            self,
            simulator,
            quick_scale,
            algo,
            batched,
            noise_model,
            sparse
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model=="nb":
                from batchglm.api.models.glm_nb import Estimator, InputData
            elif noise_model=="norm":
                from batchglm.api.models.glm_norm import Estimator, InputData
            elif noise_model=="beta":
                from batchglm.api.models.glm_beta import Estimator, InputData
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 100
        provide_optimizers = {"gd": False, "adam": False, "adagrad": False, "rmsprop": False,
                              "nr": False, "nr_tr": False,
                              "irls": False, "irls_gd": False, "irls_tr": False, "irls_gd_tr": False}
        provide_optimizers[algo.lower()] = True

        if sparse:
            input_data = InputData.new(
                data=scipy.sparse.csr_matrix(simulator.input_data.X),
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale
            )
        else:
            input_data = InputData.new(
                data=simulator.input_data.X,
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale
            )

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            provide_batched=batched,
            optim_algos=[algo.lower()]
        )
        super().__init__(
            estimator=estimator,
            simulator=simulator,
            algo=algo
        )

class Test_Graph_GLM_ALL(
    Test_Graph_GLM,
    unittest.TestCase
):
    """
    Test whether training graphs work.

    Quick tests which simply passes small data sets through
    all possible training graphs to check whether there are graph
    bugs. This is all tested in test_acc_glm.py but this
    set of unit_tests runs much faster and does not abort due
    to accuracy outliers. The training graphs covered are:

     - full data model
        - train a and b model: test_full_global_a_and_b()
        - train a model only: test_full_global_a_only()
        - train b model only: test_full_global_b_only()
    - batched data model
        - train a and b model: test_batched_global_a_and_b()
        - train a model only: test_batched_global_a_only()
        - train b model only: test_batched_global_b_only()
    """
    noise_model: str
    _estims: List[_Estimator_GLM]

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.glm_nb import Simulator
            elif self.noise_model=="norm":
                from batchglm.api.models.glm_norm import Simulator
            elif self.noise_model=="beta":
                from batchglm.api.models.glm_beta import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=200, num_features=2)

    def basic_test_one_algo(
            self,
            batched,
            train_loc,
            train_scale,
            algo,
            sparse
    ):
        estimator = _Test_Graph_GLM_ALL_Estim(
            simulator=self.simulator(train_loc=train_loc),
            quick_scale=False if train_scale else True,
            algo=algo,
            batched=batched,
            noise_model=self.noise_model,
            sparse=sparse
        )
        return self._basic_test_one_algo(
            estimator=estimator,
            batched=batched
        )

    def _test_full(self, sparse):
        self.simulate()
        super()._test_full_a_and_b(sparse=sparse)
        super()._test_full_a_only(sparse=sparse)
        super()._test_full_b_only(sparse=sparse)

    def _test_batched(self, sparse):
        self.simulate()
        super()._test_batched_a_and_b(sparse=sparse)
        super()._test_batched_a_only(sparse=sparse)
        super()._test_batched_b_only(sparse=sparse)


class Test_Graph_GLM_NB(
    Test_Graph_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether training graphs work for negative binomial noise.
    """

    def test_full_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_Graph_GLM_NB.test_full_nb()")

        self.noise_model = "nb"
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_Graph_GLM_NB.test_batched_nb()")

        self.noise_model = "nb"
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)

class Test_Graph_GLM_NORM(
    Test_Graph_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether training graphs work for normally distributed noise.
    """

    def test_full_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_Graph_GLM_NORM.test_full_norm()")

        self.noise_model = "norm"
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_Graph_GLM_NORM.test_batched_norm()")

        self.noise_model = "norm"
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)

class Test_Graph_GLM_BETA(
    Test_Graph_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether training graphs work for beta distributed noise.
    """

    def test_full_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_Graph_GLM_BETA.test_full_beta()")

        self.noise_model = "beta"
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_Graph_GLM_BETA.test_batched_beta()")

        self.noise_model = "beta"
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


if __name__ == '__main__':
    unittest.main()
