import logging
import numpy as np
import unittest

import batchglm.api as glm
from batchglm.unit_test.test_graph_glm_all import _TestGraphGlmAll

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _TestAccuracyXtremeAll(_TestGraphGlmAll):
    """
    Test whether numerical extremes throw error in initialisation or during first training steps.
    """

    def _test_low_values_a_and_b(self):
        self.simulate1()
        self.sim1.input_data.x[:, 0] = 0.
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=True,
            sparse=False
        )

    def _test_low_values_a_only(self):
        self.simulate1()
        self.sim1.input_data.x[:, 0] = 0.
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=True,
            sparse=False
        )

    def _test_low_values_b_only(self):
        self.simulate1()
        self.sim1.input_data.x[:, 0] = 0.
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=True,
            sparse=False
        )

    def _test_zero_variance_a_and_b(self):
        self.simulate1()
        self.sim1.input_data.x[:, 0] = 5.
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=True,
            sparse=False
        )

    def _test_zero_variance_a_only(self):
        self.simulate1()
        self.sim1.input_data.x[:, 0] = 5.
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=True,
            sparse=False
    )

    def _test_zero_variance_b_only(self):
        self.simulate1()
        self.sim1.input_data.x[:, 0] = 5.
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=True,
            sparse=False
        )

    def _test_all(self):
        self._test_low_values_a_and_b()
        self._test_low_values_a_only()
        self._test_low_values_b_only()
        self._test_zero_variance_a_and_b()
        self._test_zero_variance_a_only()
        self._test_zero_variance_b_only()


class TestAccuracyXtremeNb(
    _TestAccuracyXtremeAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial distributed data.
    """

    def test_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestAccuracyXtremeNb.test_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self._test_all()


class TestAccuracyXtremeNorm(
    _TestAccuracyXtremeAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for normal distributed data.
    """

    def test_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestAccuracyXtremeNorm.test_norm()")

        np.random.seed(1)
        self.noise_model = "norm"
        self._test_all()


class TestAccuracyXtremeBeta(
    _TestAccuracyXtremeAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for beta distributed data.
    """

    def test_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestAccuracyXtremeBeta.test_beta()")

        np.random.seed(1)
        self.noise_model = "beta"
        self._test_all()


if __name__ == '__main__':
    unittest.main()
