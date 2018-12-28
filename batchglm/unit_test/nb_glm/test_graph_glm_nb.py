from typing import List
import unittest
import logging

import batchglm.api as glm
from batchglm.api.models.nb_glm import Simulator, Estimator

from .external import Test_Graph_GLM, _Test_Graph_GLM_Estim

glm.setup_logging(verbosity="ERROR", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


class _Test_Graph_GLM_NB_Estim(_Test_Graph_GLM_Estim):

    def __init__(
            self,
            simulator,
            quick_scale,
            termination,
            algo
    ):
        batch_size = 10
        provide_optimizers = {"gd": False, "adam": False, "adagrad": False, "rmsprop": False, "nr": False}
        provide_optimizers[algo.lower()] = True

        estimator = Estimator(
            input_data=simulator.input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            termination_type=termination
        )
        super().__init__(
            estimator=estimator,
            simulator=simulator,
            algo=algo
        )

class Test_Graph_GLM_NB(
    Test_Graph_GLM,
    unittest.TestCase
):
    """
    Test whether training graph work.

    Quick tests which simply passes small data sets through
    all possible training graphs to check whether there are graph
    bugs. This is all tested in test_acc_glm.py but this
    set of unit_tests runs much faster and does not abort due
    to accuracy outliers. The training graphs covered are:

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
    """
    _estims: List[Estimator]

    def simulate1(self):
        self._simulate1(sim=Simulator(num_observations=50, num_features=2))

    def simulate2(self):
        self._simulate2(sim=Simulator(num_observations=50, num_features=2))

    def basic_test_one_algo(
            self,
            batched,
            termination,
            train_loc,
            train_scale,
            algo
    ):
        estimator = _Test_Graph_GLM_NB_Estim(
            simulator=self.simulator(train_loc=train_loc),
            quick_scale=False if train_scale else True,
            termination=termination,
            algo=algo
        )
        return self._basic_test_one_algo(
            estimator=estimator,
            batched=batched,
            termination=termination
        )

    def basic_test(
            self,
            batched,
            termination,
            train_loc,
            train_scale
    ):
        algos = ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR"]
        self._basic_test(
            batched=batched,
            termination=termination,
            train_loc=train_loc,
            train_scale=train_scale,
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
