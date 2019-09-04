import logging
import numpy as np
import unittest

import batchglm.api as glm
from batchglm.unit_test.test_acc_glm_all import _TestAccuracyGlmAll

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _TestAccuracyVglmAll(_TestAccuracyGlmAll):

    def simulate(self):
        super().simulate()
        # Override design matrix of simulation 1 to encode constraints
        dmat = np.hstack([
            self.sim1.input_data.design_loc,
            np.expand_dims(self.sim1.input_data.design_loc[:, 0] -
                           self.sim1.input_data.design_loc[:, -1], axis=-1)
        ])
        constraints = np.zeros([4, 3])
        constraints[0, 0] = 1
        constraints[1, 1] = 1
        constraints[2, 2] = 1
        constraints[3, 2] = -1
        new_coef_names = ['Intercept', 'condition[T.1]', 'batch[1]', 'batch[2]']
        self.sim1.input_data.design_loc = dmat
        self.sim1.input_data.design_scale = dmat
        self.sim1.input_data._design_loc_names = new_coef_names
        self.sim1.input_data._design_scale_names = new_coef_names
        self.sim1.input_data.constraints_loc = constraints
        self.sim1.input_data.constraints_scale = constraints

    def _test_full(self, sparse):
        self._test_full_a_and_b(sparse=sparse)
        self._test_full_a_only(sparse=sparse)

    def _test_batched(self, sparse):
        self._test_batched_a_and_b(sparse=sparse)
        self._test_batched_a_only(sparse=sparse)


class TestAccuracyVglmNb(
    _TestAccuracyVglmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial distributed data.
    """

    def test_full_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyVglmNb.test_full_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyVglmNb.test_batched_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


class TestAccuracyVglmNorm(
    _TestAccuracyGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for normal distributed data.
    # TODO not tested yet.
    """

    def test_full_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyVglmNorm.test_full_norm()")

        np.random.seed(1)
        self.noise_model = "norm"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyVglmNorm.test_batched_norm()")

        np.random.seed(1)
        self.noise_model = "norm"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


class TestAccuracyVglmBeta(
    _TestAccuracyGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for beta distributed data.
    TODO not working yet.
    """

    def test_full_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyVglmBeta.test_full_beta()")

        np.random.seed(1)
        self.noise_model = "beta"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyVglmBeta.test_batched_beta()")

        np.random.seed(1)
        self.noise_model = "beta"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


if __name__ == '__main__':
    unittest.main()
