from typing import List
import unittest
import logging

import batchglm.api as glm
from batchglm.api.models.nb_glm import Simulator, Estimator

from .external import Test_Accuracy_GLM, _Test_Accuracy_GLM_Estim

glm.setup_logging(verbosity="ERROR", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class _Test_Accuracy_GLM_NB_Estim(_Test_Accuracy_GLM_Estim):

    def __init__(
            self,
            simulator,
            quick_scale,
            termination
    ):
        batch_size = 900
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True, "nr": True}
        estimator = Estimator(
            input_data=simulator.input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            termination_type=termination
        )
        super().__init__(
            estimator=estimator,
            simulator=simulator
        )

class Test_Accuracy_GLM_NB(
    Test_Accuracy_GLM,
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
    _estims: List[Estimator]

    def simulate1(self):
        self._simulate1(sim=Simulator(num_observations=1000, num_features=50))

    def simulate2(self):
        self._simulate2(sim=Simulator(num_observations=1000, num_features=50))

    def basic_test(
            self,
            batched,
            termination,
            train_loc,
            train_scale
    ):
        algos = ["ADAM", "ADAGRAD", "NR"]
        estimator = _Test_Accuracy_GLM_NB_Estim(
            simulator=self.simulator(train_loc=train_loc),
            quick_scale=False if train_scale else True,
            termination=termination,
        )
        return self._basic_test(
            estimator=estimator,
            batched=batched,
            termination=termination,
            algos=algos
        )

    def test_full_byfeature_a_and_b(self):
        super()._test_full_byfeature_a_and_b()

    def test_full_byfeature_a_only(self):
        super()._test_full_byfeature_a_only()

    def test_full_byfeature_b_only(self):
        super()._test_full_byfeature_b_only()

    def test_batched_byfeature_a_and_b(self):
        super()._test_batched_byfeature_a_and_b()

    def test_batched_byfeature_a_only(self):
        super()._test_batched_byfeature_a_only()

    def test_batched_byfeature_b_only(self):
        super()._test_batched_byfeature_b_only()

    def test_full_global_a_and_b(self):
        super()._test_full_global_a_and_b()

    def test_full_global_a_only(self):
        super()._test_full_global_a_only()

    def test_full_global_b_only(self):
        super()._test_full_global_b_only()

    def test_batched_global_a_and_b(self):
        super()._test_batched_global_a_and_b()

    def test_batched_global_a_only(self):
        super()._test_batched_global_a_only()

    def test_batched_global_b_only(self):
        super()._test_batched_global_b_only()

if __name__ == '__main__':
    unittest.main()
