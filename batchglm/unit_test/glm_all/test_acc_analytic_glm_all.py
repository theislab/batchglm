import logging
from typing import List
import unittest
import numpy as np
import scipy.sparse

import batchglm.api as glm
from batchglm.models.base_glm import _EstimatorGLM, _SimulatorGLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_AccuracyAnalytic_GLM_ALL_Estim():

    estimator: _EstimatorGLM
    simulator: _SimulatorGLM
    noise_model: str

    def __init__(
            self,
            simulator,
            train_scale,
            noise_model,
            sparse,
            init_a,
            init_b
    ):
        self.simulator = simulator
        self.noise_model = noise_model

        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model == "nb":
                from batchglm.api.models.glm_nb import Estimator, InputDataGLM
            elif noise_model=="norm":
                from batchglm.api.models.glm_norm import Estimator, InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 500
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": False, "nr_tr": False,
                              "irls": False, "irls_gd": False, "irls_tr": False, "irls_gd_tr": False}

        if sparse:
            input_data = InputDataGLM.new(
                data=scipy.sparse.csr_matrix(simulator.input_data.X),
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale
            )
        else:
            input_data = InputDataGLM.new(
                data=simulator.input_data.X,
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale
            )

        self.estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=not train_scale,
            provide_optimizers=provide_optimizers,
            provide_batched=True,
            provide_fim=False,
            provide_hessian=False,
            init_a=init_a,
            init_b=init_b
        )

    def estimate(self):
        self.estimator.initialize()

    def eval_estimation_a(
            self,
            estimator_store,
            init_a,
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                threshold_dev = 1e-2
                threshold_std = 1e-1
            elif self.noise_model == "norm":
                threshold_dev = 1e-2
                threshold_std = 1e-1
            else:
                raise ValueError("noise_model not recognized")

        if init_a == "standard":
            mean_dev = np.mean(estimator_store.a[0, :] - self.simulator.a[0, :])
            std_dev = np.std(estimator_store.a[0, :] - self.simulator.a[0, :])
        elif init_a == "closed_form":
            mean_dev = np.mean(estimator_store.a - self.simulator.a)
            std_dev = np.std(estimator_store.a - self.simulator.a)
        else:
            assert False

        logging.getLogger("batchglm").info("mean_dev_a %f" % mean_dev)
        logging.getLogger("batchglm").info("std_dev_a %f" % std_dev)

        if np.abs(mean_dev) < threshold_dev and \
                std_dev < threshold_std:
            return True
        else:
            return False

    def eval_estimation_b(
            self,
            estimator_store,
            init_b
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                threshold_dev = 1e-2
                threshold_std = 1e-1
            elif self.noise_model == "norm":
                threshold_dev = 1e-2
                threshold_std = 1e-1
            else:
                raise ValueError("noise_model not recognized")

        if init_b == "standard":
            mean_dev = np.mean(estimator_store.b[0, :] - self.simulator.b[0, :])
            std_dev = np.std(estimator_store.b[0, :] - self.simulator.b[0, :])
        elif init_b == "closed_form":
            mean_dev = np.mean(estimator_store.b - self.simulator.b)
            std_dev = np.std(estimator_store.b - self.simulator.b)
        else:
            assert False

        logging.getLogger("batchglm").info("mean_dev_b %f" % mean_dev)
        logging.getLogger("batchglm").info("std_dev_b %f" % std_dev)

        if np.abs(mean_dev) < threshold_dev and \
                std_dev < threshold_std:
            return True
        else:
            return False


class Test_AccuracyAnalytic_GLM_ALL(
    unittest.TestCase
):
    noise_model: str
    _estims: List[_EstimatorGLM]

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.glm_nb import Simulator
            elif self.noise_model=="norm":
                from batchglm.api.models.glm_norm import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(
            num_observations=10000,
            num_features=3
        )

    def get_estimator(self, train_scale, sparse, init_a, init_b):
        return _Test_AccuracyAnalytic_GLM_ALL_Estim(
            simulator=self.sim,
            train_scale=train_scale,
            noise_model=self.noise_model,
            sparse=sparse,
            init_a=init_a,
            init_b=init_b
        )

    def simulate_complex(self):
        self.sim = self.get_simulator()
        self.sim.generate_sample_description(num_batches=1, num_conditions=2)
        self.sim.generate_params(
            rand_fn_ave=lambda shape: np.random.uniform(1e5, 2 * 1e5, shape),
            rand_fn_loc=lambda shape: np.random.uniform(1, 3, shape),
            rand_fn_scale=lambda shape: np.random.uniform(1, 3, shape)
        )
        self.sim.generate_data()

    def simulate_easy(self):
        self.sim = self.get_simulator()
        self.sim.generate_sample_description(num_batches=1, num_conditions=2)

        def rand_fn_standard(shape):
            theta = np.ones(shape)
            theta[0, :] = np.random.uniform(5, 20, shape[1])
            return theta

        self.sim.generate_params(
            rand_fn_ave=lambda shape: np.random.uniform(1e5, 2 * 1e5, shape),
            rand_fn_loc=lambda shape: np.ones(shape),
            rand_fn_scale=lambda shape: rand_fn_standard(shape)
        )
        self.sim.generate_data()

    def setUp(self):
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.estimator.close_session()

    def _test_a_and_b(self, sparse, init_a, init_b):
        estimator = self.get_estimator(
            train_scale=False,
            sparse=sparse,
            init_a=init_a,
            init_b=init_b
        )
        estimator.estimate()
        estimator_store = estimator.estimator.finalize()
        self._estims.append(estimator)
        success = estimator.eval_estimation_a(
            estimator_store=estimator_store,
            init_a=init_a,

        )
        assert success, "estimation for a_model was inaccurate"
        success = estimator.eval_estimation_b(
            estimator_store=estimator_store,
            init_b=init_b
        )
        assert success, "estimation for b_model was inaccurate"
        return True


class Test_AccuracyAnalytic_GLM_NB(
    Test_AccuracyAnalytic_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial noise.
    """

    def test_a_closed_b_closed(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR),
        logging.getLogger("batchglm").setLevel(logging.DEBUG)
        logger.error("Test_AccuracyAnalytic_GLM_NB.test_a_closed_b_closed()")

        self.noise_model = "nb"
        self.simulate_complex()
        self._test_a_and_b(sparse=False, init_a="closed_form", init_b="closed_form")
        self._test_a_and_b(sparse=True, init_a="closed_form", init_b="closed_form")

    def test_a_standard_b_standard(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("Test_AccuracyAnalytic_GLM_NB.test_a_standard_b_standard()")

        self.noise_model = "nb"
        self.simulate_easy()
        self._test_a_and_b(sparse=False, init_a="standard", init_b="standard")
        self._test_a_and_b(sparse=True, init_a="standard", init_b="standard")


class Test_AccuracyAnalytic_GLM_NORM(
    Test_AccuracyAnalytic_GLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for normally distributed noise.
    """

    def test_a_closed_b_closed(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR),
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("Test_AccuracyAnalytic_GLM_NORM.test_a_closed_b_closed()")

        self.noise_model = "norm"
        self.simulate_complex()
        self._test_a_and_b(sparse=False, init_a="closed_form", init_b="closed_form")
        self._test_a_and_b(sparse=True, init_a="closed_form", init_b="closed_form")

    def test_a_standard_b_standard(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("Test_AccuracyAnalytic_GLM_NORM.test_a_standard_b_standard()")

        self.noise_model = "norm"
        self.simulate_easy()
        self._test_a_and_b(sparse=False, init_a="standard", init_b="standard")
        self._test_a_and_b(sparse=True, init_a="standard", init_b="standard")


if __name__ == '__main__':
    unittest.main()
