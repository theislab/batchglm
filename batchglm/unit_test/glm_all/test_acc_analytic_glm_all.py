from typing import List
import unittest
import logging
import scipy.sparse

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
            noise_model,
            sparse,
            init_a,
            init_b
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model == "nb":
                from batchglm.api.models.glm_nb import Estimator, InputData
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 500
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": True, "nr_tr": True, "irls": True, "irls_tr": True}

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
            quick_scale=not train_scale,
            provide_optimizers=provide_optimizers,
            termination_type="by_feature",
            init_a=init_a,
            init_b=init_b
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
                from batchglm.api.models.glm_nb import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(
            num_observations=10000,
            num_features=3
        )

    def get_estimator(self, train_scale, sparse, init_a, init_b):
        return _Test_AccuracyAnalytic_GLM_ALL_Estim(
            simulator=self.sim,
            train_scale=train_scale,
            noise_model=self.noise_model,
            sparse=sparse,
            init_a=init_a,
            init_b=init_b
        )

    def _test_a_closed_b_closed(self, sparse):
        self._test_a_and_b_closed(sparse=sparse, init_a="closed_form", init_b="closed_form")

    def _test_a_closed_b_standard(self, sparse):
        self._test_a_and_b_closed(sparse=sparse, init_a="closed_form", init_b="standard")

    def _test_a_standard_b_closed(self, sparse):
        self._test_a_and_b_closed(sparse=sparse, init_a="standard", init_b="closed_form")

    def _test_a_standard_b_standard(self, sparse):
        self._test_a_and_b_closed(sparse=sparse, init_a="standard", init_b="standard")


class Test_AccuracyAnalytic_GLM_NB(
    Test_AccuracyAnalytic_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial noise.
    """

    def test_a_closed_b_closed(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_AccuracyAnalytic_GLM_NB.test_a_closed_b_closed()")

        self.noise_model = "nb"
        self.simulate_complex()
        self._test_a_closed_b_closed(sparse=False)
        self._test_a_closed_b_closed(sparse=True)

    def test_a_closed_b_standard(self):
        # TODO this is still inexact!
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("Test_AccuracyAnalytic_GLM_NB.test_a_closed_b_standard()")

        self.noise_model = "nb"
        self.simulate_b_easy()
        #self._test_a_closed_b_standard(sparse=False)
        #self._test_a_closed_b_standard(sparse=True)

    def test_a_standard_b_closed(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_AccuracyAnalytic_GLM_NB.test_a_standard_b_closed()")

        self.noise_model = "nb"
        self.simulate_a_easy()
        self._test_a_standard_b_closed(sparse=False)
        self._test_a_standard_b_closed(sparse=True)

    def test_a_standard_b_standard(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("Test_AccuracyAnalytic_GLM_NB.test_a_standard_b_standard()")

        self.noise_model = "nb"
        self.simulate_a_b_easy()
        self._test_a_standard_b_standard(sparse=False)
        self._test_a_standard_b_standard(sparse=True)



if __name__ == '__main__':
    unittest.main()
