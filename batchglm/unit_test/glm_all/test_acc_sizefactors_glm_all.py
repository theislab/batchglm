from typing import List
import unittest
import logging
import scipy.sparse

import batchglm.api as glm
from batchglm.models.base_glm import _Estimator_GLM

from .external import Test_AccuracySizeFactors_GLM, _Test_AccuracySizeFactors_GLM_Estim

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_AccuracySizeFactors_GLM_ALL_Estim(_Test_AccuracySizeFactors_GLM_Estim):

    def __init__(
            self,
            simulator,
            quick_scale,
            termination,
            noise_model,
            sparse
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model=="nb":
                from batchglm.api.models.glm_nb import Estimator, InputData
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 900
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": True, "nr_tr": True, "irls": True, "irls_tr": True}

        if sparse:
            input_data = InputData.new(
                data=scipy.sparse.csr_matrix(simulator.input_data.X),
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale
            )
            input_data.size_factors = simulator.size_factors
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
            termination_type=termination,
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
                from batchglm.api.models.glm_nb import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=10000, num_features=10)

    def basic_test(
            self,
            batched,
            termination,
            train_scale,
            sparse
    ):
        algos = ["NR_TR"]  #["ADAM", "NR_TR", "IRLS_TR"]
        estimator = _Test_AccuracySizeFactors_GLM_ALL_Estim(
            simulator=self.sim,
            quick_scale=False if train_scale else True,
            termination=termination,
            noise_model=self.noise_model,
            sparse=sparse
        )
        return self._basic_test(
            estimator=estimator,
            batched=batched,
            termination=termination,
            algos=algos
        )

    def _test_full(self, sparse):
        logger.debug("* Running tests for full data")
        self._test_full_a_and_b(sparse=sparse)
        self._test_full_a_only(sparse=sparse)

    def _test_batched(self, sparse):
        logger.debug("* Running tests for batched data")
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
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("Test_AccuracySizeFactors_GLM_NB.test_full_nb()")

        self.noise_model = "nb"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("Test_AccuracySizeFactors_GLM_NB.test_batched_nb()")

        self.noise_model = "nb"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


if __name__ == '__main__':
    unittest.main()
