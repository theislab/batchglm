import logging
import unittest
from typing import List, Optional, Union

import numpy as np
from test_accuracy import TestAccuracy
from utils import get_estimator, get_generated_model

logger = logging.getLogger("batchglm")
logging.getLogger("batchglm").setLevel(logging.WARNING)


class _TestAccuracyXtremeAll(TestAccuracy):
    """
    Test whether numerical extremes throw error in initialisation or during first training steps.
    """

    def _test_accuracy_extreme_values(self, idx: Union[List[int], int, np.ndarray], val: float, noise_model: str):
        model = get_generated_model(noise_model=noise_model, num_conditions=2, num_batches=4, sparse=False, mode=None)
        model._x[:, idx] = val
        estimator = get_estimator(noise_model=noise_model, model=model, init_location="standard", init_scale="standard")
        return self._test_accuracy(estimator)

    def _test_low_values(self, **kwargs):
        self._test_accuracy_extreme_values(idx=0, val=0.0, **kwargs)

    def _test_zero_variance(self, **kwargs):
        self._modify_sim(idx=0, val=5.0, **kwargs)
        return self.basic_test(batched=False, train_loc=True, train_scale=True, sparse=False)


class TestAccuracyXtremeNb(_TestAccuracyXtremeAll):
    """
    Test whether optimizers yield exact results for negative binomial distributed data.
    """

    def test_nb(self) -> bool:
        logger.error("TestAccuracyXtremeNb.test_nb()")

        np.random.seed(1)
        ret_val = self._test_low_values(noise_model="nb")
        np.random.seed(1)
        return ret_val and self._test_zero_variance(noise_model="nb")


class TestAccuracyXtremeNorm(_TestAccuracyXtremeAll):
    """
    Test whether optimizers yield exact results for normal distributed data.
    """

    def test_norm(self) -> bool:
        logger.error("TestAccuracyXtremeNorm.test_norm()")
        logger.info("Normal noise model not implemented for numpy")

        np.random.seed(1)
        self._test_low_values(noise_model="norm")
        # self._test_zero_variance(noise_model="norm")
        return True


class TestAccuracyXtremeBeta(_TestAccuracyXtremeAll):
    """
    Test whether optimizers yield exact results for beta distributed data.
    """

    def test_beta(self) -> bool:
        logger.error("TestAccuracyXtremeBeta.test_beta()")
        logger.info("Beta noise model not implemented for numpy")

        # np.random.seed(1)
        # self._test_low_values(noise_model="beta")
        # self._test_zero_variance(noise_model="beta")
        return True


if __name__ == "__main__":
    unittest.main()
