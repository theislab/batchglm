
import unittest
from tests.numpy.utils import getGeneratedModel
from test.numpy.test_accuracy import TestAccuracy

class _TestAccuracyXtremeAll(TestAccuracy):
    """
    Test whether numerical extremes throw error in initialisation or during first training steps.
    """

    def _test_accuracy_extreme_values(idx: Union[List[int], int, np.ndarray], val: float):
        if self.noise_model == 'nb':
            from batchglm.train.numpy.glm_nb import Estimator
        elif self.noise_model == 'norm':
            from batchglm.train.numpy.norm import Estimator
        elif self.noise_model == 'beta':
            from batchglm.train.numpy.beta import Estimator
        else:
            raise ValueError(f"Noise model {self.noise_model} not recognized.")
        
        model = getGeneratedModel(
            noise_model=self.noise_model,
            num_conditions=2,
            num_batches=4,
            sparse=False,
            mode=None
        )
        model._x[:, idx] = val
        estimator = Estimator(model=model, init_location="standard", init_scale="standard")
        return self._testAccuracy(Estimator(model=sparse_model, init_location="standard", init_scale="standard"))

    def _test_low_values(self):
        self._test_accuracy_extreme_values(idx=0, val=0.0)

    def _test_zero_variance(self):
        self._modify_sim(idx=0, val=5.0)
        return self.basic_test(batched=False, train_loc=True, train_scale=True, sparse=False)


class TestAccuracyXtremeNb(_TestAccuracyXtremeAll):
    """
    Test whether optimizers yield exact results for negative binomial distributed data.
    """

    def test_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestAccuracyXtremeNb.test_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self._test_low_values()
        self._test_zero_variance()


class TestAccuracyXtremeNorm(_TestAccuracyXtremeAll):
    """
    Test whether optimizers yield exact results for normal distributed data.
    """

    def test_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestAccuracyXtremeNorm.test_norm()")
        logger.info("Normal noise model not implemented for numpy")

        # np.random.seed(1)
        # self.noise_model = "norm"
        # self._test_low_values()
        # self._test_zero_variance()


class TestAccuracyXtremeBeta(_TestAccuracyXtremeAll):
    """
    Test whether optimizers yield exact results for beta distributed data.
    """

    def test_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestAccuracyXtremeBeta.test_beta()")
        logger.info("Beta noise model not implemented for numpy")

        # np.random.seed(1)
        # self.noise_model = "beta"
        # self._test_low_values()
        # self._test_zero_variance()


if __name__ == "__main__":
    unittest.main()
