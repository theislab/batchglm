import logging
import unittest

from utils import get_estimator, get_generated_model

from batchglm.train.numpy.base_glm import BaseModelContainer

logger = logging.getLogger("batchglm")

n_obs = 2000
n_vars = 100
num_batches = 4
num_conditions = 2

class TestShape(unittest.TestCase):

    def _test_shape(self, model_container: BaseModelContainer) -> bool:
        """Runs the estimator to fit the model and evaluates with respect to the simulated parameters."""
        assert model_container.theta_scale.shape == (model_container.model.num_scale_params, n_vars)
        assert model_container.theta_location.shape == (model_container.model.num_loc_params, n_vars)


class TestShapeNB(TestShape):
    def test_shape(self) -> bool:
        """
        This tests randTheta simulated data with 2 conditions and 4 batches sparse and dense.
        """
        dense_model = get_generated_model(
            noise_model="nb", num_conditions=num_conditions, num_batches=num_batches, sparse=False,
            n_obs=n_obs, n_vars=n_vars
        )
        sparse_model = get_generated_model(
            noise_model="nb", num_conditions=num_conditions, num_batches=num_batches, sparse=True,
            n_obs=n_obs, n_vars=n_vars
        )
        dense_estimator = get_estimator(
            noise_model="nb", model=dense_model, init_location="standard", init_scale="standard"
        )
        sparse_estimator = get_estimator(
            noise_model="nb", model=sparse_model, init_location="standard", init_scale="standard"
        )
        model_container_dense = dense_estimator.model_container
        model_container_sparse = sparse_estimator.model_container
        self._test_shape(model_container_dense)
        self._test_shape(model_container_sparse)

if __name__ == "__main__":
    unittest.main()
