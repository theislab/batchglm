import logging
import numpy as np
import unittest

import batchglm.api as glm
from batchglm.unit_test.test_acc_glm_all import _TestAccuracyGlmAll

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _TestAccuracyGlmAllSf(_TestAccuracyGlmAll):

    def simulate(self):
        super().simulate()
        # Add size factors into input data: Do not centre at 1 so that they bias MAD if something is off.
        self.sim1.input_data.size_factors = np.random.uniform(1.5, 2., size=self.sim1.input_data.num_observations)

    def _test_full(self, sparse):
        self._test_full_a_and_b(sparse=sparse)

    def _test_batched(self, sparse):
        self._test_batched_a_and_b(sparse=sparse)


class TestAccuracyGlmNbSf(
    _TestAccuracyGlmAllSf,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial distributed data.
    """

    def test_full_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNbSf.test_full_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNbSf.test_batched_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


class TestAccuracyGlmNormSf(
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
        logger.error("TestAccuracyGlmNormSf.test_full_norm()")

        np.random.seed(1)
        self.noise_model = "norm"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNormSf.test_batched_norm()")

        np.random.seed(1)
        self.noise_model = "norm"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


class TestAccuracyGlmBetaSf(
    _TestAccuracyGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for beta distributed data.
    Note: size factors are note implemented for beta distribution.
    """

    def test_dummy(self):
        return True


if __name__ == '__main__':
    unittest.main()
