from typing import List
import unittest
import logging

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
            termination,
            algo,
            noise_model
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model=="nb":
                from batchglm.api.models.nb_glm import Estimator
            else:
                raise ValueError("noise_model not recognized")

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
    noise_model: str
    _estims: List[_Estimator_GLM]

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.nb_glm import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=50, num_features=2)

    def basic_test_one_algo(
            self,
            batched,
            termination,
            train_loc,
            train_scale,
            algo
    ):
        estimator = _Test_Graph_GLM_ALL_Estim(
            simulator=self.simulator(train_loc=train_loc),
            quick_scale=False if train_scale else True,
            termination=termination,
            algo=algo,
            noise_model=self.noise_model
        )
        return self._basic_test_one_algo(
            estimator=estimator,
            batched=batched
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

    def _test_full_byfeature(self):
        self.simulate()
        logger.debug("* Running tests for full data and feature-wise termination")
        logger.debug("** Running tests for a and b training")
        super()._test_full_byfeature_a_and_b()
        logger.debug("** Running tests for a only training")
        super()._test_full_byfeature_a_only()
        logger.debug("** Running tests for b only training")
        super()._test_full_byfeature_b_only()

    def _test_batched_byfeature(self):
        self.simulate()
        logger.debug("* Running tests for batched data and feature-wise termination")
        logger.debug("** Running tests for a and b training")
        super()._test_batched_byfeature_a_and_b()
        logger.debug("** Running tests for a only training")
        super()._test_batched_byfeature_a_only()
        logger.debug("** Running tests for b only training")
        super()._test_batched_byfeature_b_only()

    def _test_full_global(self):
        self.simulate()
        logger.debug("* Running tests for full data and global termination")
        logger.debug("** Running tests for a and b training")
        super()._test_full_global_a_and_b()
        logger.debug("** Running tests for a only training")
        super()._test_full_global_a_only()
        logger.debug("** Running tests for b only training")
        super()._test_full_global_b_only()

    def _test_batched_global(self):
        self.simulate()
        logger.debug("* Running tests for batched data and global termination")
        logger.debug("** Running tests for a and b training")
        super()._test_batched_global_a_and_b()
        logger.debug("** Running tests for a only training")
        super()._test_batched_global_a_only()
        logger.debug("** Running tests for b only training")
        super()._test_batched_global_b_only()


class Test_Graph_GLM_NB(
    Test_Graph_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether training graphs work for negative binomial noise.
    """

    def test_full_byfeature_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)

        self.noise_model = "nb"
        self._test_full_byfeature()

    def test_batched_byfeature_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)

        self.noise_model = "nb"
        self._test_batched_byfeature()

    def test_full_global_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)

        self.noise_model = "nb"
        self._test_full_global()

    def test_batched_global_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)

        self.noise_model = "nb"
        self._test_batched_global()

if __name__ == '__main__':
    unittest.main()
