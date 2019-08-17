from typing import List
import unittest
import logging
import scipy.sparse

from batchglm.models.base_glm import _EstimatorGLM

from .external import Test_AccuracySizeFactors_GLM, _Test_AccuracySizeFactors_GLM_Estim


class _Test_AccuracySizeFactors_GLM_ALL_Estim(_Test_AccuracySizeFactors_GLM_Estim):

    def __init__(
            self,
            simulator,
            quick_scale,
            noise_model,
            sparse
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model=="nb":
                from batchglm.api.models.glm_nb import Estimator, InputDataGLM
            elif noise_model=="norm":
                from batchglm.api.models.glm_norm import Estimator, InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 900
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": True, "nr_tr": True,
                              "irls": True, "irls_gd": True, "irls_tr": True, "irls_gd_tr": True}

        if sparse:
            input_data = InputDataGLM.new(
                data=scipy.sparse.csr_matrix(simulator.input_data.x),
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale
            )
        else:
            input_data = InputDataGLM.new(
                data=simulator.input_data.x,
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale
            )
        input_data.size_factors = simulator.size_factors

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            provide_batched=True,
            provide_fim=True,
            provide_hessian=True,
            init_a="standard",
            init_b="standard"
        )
        super().__init__(
            estimator=estimator,
            simulator=simulator
        )

class Test_AccuracySizeFactors_GLM_ALL(
    Test_AccuracySizeFactors_GLM,
    unittest.TestCase
):
    noise_model: str
    _estims: List[_EstimatorGLM]

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.glm_nb import Simulator
            elif self.noise_model=="norm":
                from batchglm.api.models.glm_norm import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=10000, num_features=10)

    def basic_test(
            self,
            batched,
            train_scale,
            sparse
    ):
        # Define algorithms to test depending on noise model..
        if self.noise_model == "nb":
            algos = ["IRLS_GD_TR"]
        elif self.noise_model == "norm":
            algos = ["IRLS_TR"]
        else:
            assert False

        estimator = _Test_AccuracySizeFactors_GLM_ALL_Estim(
            simulator=self.sim,
            quick_scale=False if train_scale else True,
            noise_model=self.noise_model,
            sparse=sparse
        )
        return self._basic_test(
            estimator=estimator,
            batched=batched,
            algos=algos
        )

    def _test_full(self, sparse):
        logging.getLogger("batchglm").debug("* Running tests for full data")
        self._test_full_a_and_b(sparse=sparse)
        self._test_full_a_only(sparse=sparse)

    def _test_batched(self, sparse):
        logging.getLogger("batchglm").debug("* Running tests for batched data")
        self._test_batched_a_and_b(sparse=sparse)
        self._test_batched_a_only(sparse=sparse)


class Test_AccuracySizeFactors_GLM_NB(
    Test_AccuracySizeFactors_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial noise.
    """

    def test_full_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_AccuracySizeFactors_GLM_NB.test_full_nb()")

        self.noise_model = "nb"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_AccuracySizeFactors_GLM_NB.test_batched_nb()")

        self.noise_model = "nb"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)

class Test_AccuracySizeFactors_GLM_NORM(
    Test_AccuracySizeFactors_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial noise.
    """

    def test_full_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.DEBUG)
        logging.getLogger("batchglm").error("Test_AccuracySizeFactors_GLM_NORM.test_full_norm()")

        self.noise_model = "norm"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_AccuracySizeFactors_GLM_NORM.test_batched_norm()")

        self.noise_model = "norm"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


if __name__ == '__main__':
    unittest.main()
