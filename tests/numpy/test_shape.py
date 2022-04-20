import logging
import unittest

from utils import get_estimator, get_generated_model

from batchglm.train.numpy.base_glm import BaseModelContainer

logger = logging.getLogger("batchglm")

n_obs = 2000
n_vars = 100
num_batches = 4
num_conditions = 2


def _test_shape_of_model(model_container: BaseModelContainer) -> bool:
    """Check the shape of different fitted/parametrized values against what is epected"""
    assert model_container.theta_scale.shape == (model_container.model.num_scale_params, n_vars)
    assert model_container.theta_location.shape == (model_container.model.num_loc_params, n_vars)

    assert model_container.fim_weight_location_location.shape == (n_obs, n_vars)

    assert model_container.hessian_weight_location_location.shape == (n_obs, n_vars)
    assert model_container.hessian_weight_location_scale.shape == (n_obs, n_vars)
    assert model_container.hessian_weight_scale_scale.shape == (n_obs, n_vars)

    assert model_container.jac_scale.shape == (n_vars, model_container.model.num_scale_params)
    assert model_container.jac_location.shape == (n_vars, model_container.model.num_loc_params)


class TestShape(unittest.TestCase):

    def _test_shape(self) -> bool:
        dense_model = get_generated_model(
            noise_model=self._model_name, num_conditions=num_conditions, num_batches=num_batches, sparse=False,
            n_obs=n_obs, n_vars=n_vars
        )
        sparse_model = get_generated_model(
            noise_model=self._model_name, num_conditions=num_conditions, num_batches=num_batches, sparse=True,
            n_obs=n_obs, n_vars=n_vars
        )
        dense_estimator = get_estimator(
            noise_model=self._model_name, model=dense_model, init_location="standard", init_scale="standard"
        )
        sparse_estimator = get_estimator(
            noise_model=self._model_name, model=sparse_model, init_location="standard", init_scale="standard"
        )
        model_container_dense = dense_estimator.model_container
        model_container_sparse = sparse_estimator.model_container
        _test_shape_of_model(model_container_dense)
        _test_shape_of_model(model_container_sparse)
        return True


class TestShapeNB(TestShape):

    def __init__(self, *args, **kwargs):
        self._model_name = "nb"
        super(TestShapeNB, self).__init__(*args, **kwargs)

    def test_shape(self) -> bool:
        return self._test_shape()

class TestShapeNorm(TestShape):

    def __init__(self, *args, **kwargs):
        self._model_name = "norm"
        super(TestShapeNorm, self).__init__(*args, **kwargs)

    def test_shape(self) -> bool:
        return self._test_shape()

if __name__ == "__main__":
    unittest.main()
