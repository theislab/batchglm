import logging
import unittest

import numpy as np
from utils import get_estimator, get_generated_model

from batchglm import pkg_constants
from batchglm.models.base_glm import ModelGLM
from batchglm.train.numpy.base_glm import EstimatorGlm

logger = logging.getLogger("batchglm")

NB_OPTIMIZERS = ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR", "NR_TR", "IRLS", "IRLS_GD", "IRLS_TR", "IRLS_GD_TR"]
NORM_OPTIMIZERS = ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR", "NR_TR", "IRLS", "IRLS_TR"]
BETA_OPTIMIZERS = ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR", "NR_TR"]


pkg_constants.TRUST_REGION_T1 = 0.5
pkg_constants.TRUST_REGION_T2 = 1.5
pkg_constants.CHOLESKY_LSTSQS = True
pkg_constants.CHOLESKY_LSTSQS_BATCHED = True
pkg_constants.JACOBIAN_MODE = "analytic"


class TestAccuracy(unittest.TestCase):
    def eval_estimation(self, estimator: EstimatorGlm):
        mean_thres_location = 0.2
        mean_thres_scale = 0.2
        std_thres_location = 1
        std_thres_scale = 1

        def deviation_theta(true: np.ndarray, pred: np.ndarray, mean_thres: float, std_thres: float) -> bool:
            relative_deviation = (pred - true) / true
            mean = np.mean(relative_deviation)
            std = np.std(relative_deviation)
            logger.info(f"Relative deviation theta location: {mean} (mean), {std} (std)")
            return np.abs(mean) <= mean_thres and std <= std_thres

        success = True
        if estimator.train_loc:
            success = deviation_theta(
                true=estimator.model_container.model._theta_location,
                pred=estimator.model_container.theta_location,
                mean_thres=mean_thres_location,
                std_thres=std_thres_location,
            )
        if estimator.train_scale:
            success &= deviation_theta(
                true=estimator.model_container.model._theta_scale,
                pred=estimator.model_container.theta_scale,
                mean_thres=mean_thres_scale,
                std_thres=std_thres_scale,
            )
        return success

    def _test_accuracy(self, estimator: EstimatorGlm) -> bool:
        """Runs the estimator to fit the model and evaluates with respect to the simulated parameters."""
        estimator.initialize()
        estimator.train_sequence(training_strategy="DEFAULT")
        success = self.eval_estimation(estimator)
        if not success:
            logger.warning("Estimator did not yield exact results")
        return success


class TestAccuracyNB(TestAccuracy):
    def test_accuracy_rand_theta(self):
        """
        This tests randTheta simulated data with 2 conditions and 4 batches sparse and dense.
        """
        dense_model = get_generated_model(
            noise_model="nb", num_conditions=2, num_batches=4, sparse=False, mode="randTheta"
        )
        sparse_model = get_generated_model(
            noise_model="nb", num_conditions=2, num_batches=4, sparse=True, mode="randTheta"
        )
        dense_estimator = get_estimator(
            noise_model="nb", model=dense_model, init_location="standard", init_scale="standard"
        )
        assert self._test_accuracy(dense_estimator)

        sparse_estimator = get_estimator(
            noise_model="nb", model=sparse_model, init_location="standard", init_scale="standard"
        )
        assert self._test_accuracy(sparse_estimator)

    def test_accuracy_const_theta(self):
        """
        This tests constTheta simulated data with 2 conditions and 0 batches sparse and dense.
        """
        dense_model = get_generated_model(
            noise_model="nb", num_conditions=2, num_batches=0, sparse=False, mode="constTheta"
        )
        sparse_model = get_generated_model(
            noise_model="nb", num_conditions=2, num_batches=0, sparse=True, mode="constTheta"
        )

        dense_estimator = get_estimator(
            noise_model="nb", model=dense_model, init_location="standard", init_scale="standard"
        )
        assert self._test_accuracy(dense_estimator)

        sparse_estimator = get_estimator(
            noise_model="nb", model=sparse_model, init_location="standard", init_scale="standard"
        )
        assert self._test_accuracy(sparse_estimator)


class TestAccuracyPoisson(TestAccuracy):
    def test_accuracy_rand_theta(self):
        """
        This tests randTheta simulated data with 2 conditions and 4 batches sparse and dense.
        """
        dense_model = get_generated_model(
            noise_model="poisson", num_conditions=2, num_batches=4, sparse=False, mode="randTheta"
        )
        sparse_model = get_generated_model(
            noise_model="poisson", num_conditions=2, num_batches=4, sparse=True, mode="randTheta"
        )
        dense_estimator = get_estimator(
            noise_model="poisson", model=dense_model, init_location="standard", init_scale="standard"
        )
        assert self._test_accuracy(dense_estimator)

        sparse_estimator = get_estimator(
            noise_model="poisson", model=sparse_model, init_location="standard", init_scale="standard"
        )
        assert self._test_accuracy(sparse_estimator)

    def test_accuracy_const_theta(self):
        """
        This tests constTheta simulated data with 2 conditions and 0 batches sparse and dense.
        """
        dense_model = get_generated_model(
            noise_model="poisson", num_conditions=2, num_batches=0, sparse=False, mode="constTheta"
        )
        sparse_model = get_generated_model(
            noise_model="poisson", num_conditions=2, num_batches=0, sparse=True, mode="constTheta"
        )

        dense_estimator = get_estimator(
            noise_model="poisson", model=dense_model, init_location="standard", init_scale="standard"
        )
        assert self._test_accuracy(dense_estimator)

        sparse_estimator = get_estimator(
            noise_model="poisson", model=sparse_model, init_location="standard", init_scale="standard"
        )
        assert self._test_accuracy(sparse_estimator)


class TestAccuracyNorm(TestAccuracy):
    def test_accuracy_rand_theta(self):
        """
        This tests randTheta simulated data with 2 conditions and 4 batches sparse and dense.
        """
        dense_model = get_generated_model(
            noise_model="norm", num_conditions=2, num_batches=4, sparse=False, mode="randTheta"
        )
        sparse_model = get_generated_model(
            noise_model="norm", num_conditions=2, num_batches=4, sparse=True, mode="randTheta"
        )
        dense_estimator = get_estimator(noise_model="norm", model=dense_model)
        assert self._test_accuracy(dense_estimator)

        sparse_estimator = get_estimator(noise_model="norm", model=sparse_model)
        assert self._test_accuracy(sparse_estimator)

    def test_accuracy_const_theta(self):
        """
        This tests constTheta simulated data with 2 conditions and 0 batches sparse and dense.
        """
        dense_model = get_generated_model(
            noise_model="norm", num_conditions=2, num_batches=0, sparse=False, mode="constTheta"
        )
        sparse_model = get_generated_model(
            noise_model="norm", num_conditions=2, num_batches=0, sparse=True, mode="constTheta"
        )

        dense_estimator = get_estimator(noise_model="norm", model=dense_model)
        assert self._test_accuracy(dense_estimator)

        sparse_estimator = get_estimator(noise_model="norm", model=sparse_model)
        assert self._test_accuracy(sparse_estimator)


if __name__ == "__main__":
    unittest.main()
