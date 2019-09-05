import logging
import unittest
import numpy as np
import scipy.sparse

import batchglm.api as glm
from batchglm.models.base_glm import _EstimatorGLM, _SimulatorGLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _TestAccuracyAnalyticGlmAllEstim():

    estimator: _EstimatorGLM
    sim: _SimulatorGLM
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
        self.sim = simulator
        self.noise_model = noise_model

        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model == "nb":
                from batchglm.api.models.tf1.glm_nb import Estimator, InputDataGLM
            elif noise_model == "norm":
                from batchglm.api.models import Estimator, InputDataGLM
            elif noise_model == "beta":
                from batchglm.api.models.tf1.glm_beta import Estimator, InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 500
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": False, "nr_tr": False,
                              "irls": False, "irls_gd": False, "irls_tr": False, "irls_gd_tr": False}

        if sparse:
            input_data = InputDataGLM(
                data=scipy.sparse.csr_matrix(simulator.input_data.x),
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale
            )
        else:
            input_data = InputDataGLM(
                data=simulator.input_data.x,
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

    def eval_estimation_a(
            self,
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
            elif self.noise_model == "beta":
                threshold_dev = 1e-2
                threshold_std = 1e-1
            else:
                raise ValueError("noise_model not recognized")

        if init_a == "standard":
            mean_dev = np.mean(self.estimator.model.a_var[0, :] - self.sim.a_var[0, :])
            std_dev = np.std(self.estimator.model.a_var[0, :] - self.sim.a_var[0, :])
        elif init_a == "closed_form":
            mean_dev = np.mean(self.estimator.model.a_var - self.sim.a_var)
            std_dev = np.std(self.estimator.model.a_var - self.sim.a_var)
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
            elif self.noise_model == "beta":
                threshold_dev = 1e-2
                threshold_std = 1e-1
            else:
                raise ValueError("noise_model not recognized")

        if init_b == "standard":
            mean_dev = np.mean(self.estimator.b_var[0, :] - self.sim.b[0, :])
            std_dev = np.std(self.estimator.b_var[0, :] - self.sim.b[0, :])
        elif init_b == "closed_form":
            mean_dev = np.mean(self.estimator.b_var - self.sim.b)
            std_dev = np.std(self.estimator.b_var - self.sim.b)
        else:
            assert False

        logging.getLogger("batchglm").info("mean_dev_b %f" % mean_dev)
        logging.getLogger("batchglm").info("std_dev_b %f" % std_dev)

        if np.abs(mean_dev) < threshold_dev and \
                std_dev < threshold_std:
            return True
        else:
            return False


class TestAccuracyAnalyticGlmAll(
    unittest.TestCase
):
    noise_model: str

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.tf1.glm_nb import Simulator
            elif self.noise_model == "norm":
                from batchglm.api.models import Simulator
            elif self.noise_model == "beta":
                from batchglm.api.models.tf1.glm_beta import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(
            num_observations=100000,
            num_features=3
        )

    def get_estimator(self, train_scale, sparse, init_a, init_b):
        return _TestAccuracyAnalyticGlmAllEstim(
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

        def rand_fn_ave(shape):
            if self.noise_model in ["nb", "norm"]:
                theta = np.random.uniform(10, 1000, shape)
            elif self.noise_model in ["beta"]:
                theta = np.random.uniform(0.1, 0.7, shape)
            else:
                raise ValueError("noise model not recognized")
            return theta

        def rand_fn_loc(shape):
            if self.noise_model in ["nb", "norm"]:
                theta = np.random.uniform(1, 3, shape)
            elif self.noise_model in ["beta"]:
                theta = np.random.uniform(0, 0.15, shape)
            else:
                raise ValueError("noise model not recognized")
            return theta

        def rand_fn_scale(shape):
            theta = np.zeros(shape)
            if self.noise_model in ["nb"]:
                theta[0, :] = np.random.uniform(1, 3, shape[1])
            elif self.noise_model in ["norm"]:
                theta[0, :] = np.random.uniform(1, 2, shape[1])
            elif self.noise_model in ["beta"]:
                theta[0, :] = np.random.uniform(0.2, 0.4, shape[1])
            else:
                raise ValueError("noise model not recognized")
            return theta

        self.sim.generate_params(
            rand_fn_ave=lambda shape: rand_fn_ave(shape),
            rand_fn_loc=lambda shape: rand_fn_loc(shape),
            rand_fn_scale=lambda shape: rand_fn_scale(shape)
        )
        self.sim.generate_data()

    def simulate_easy(self):
        self.sim = self.get_simulator()
        self.sim.generate_sample_description(num_batches=1, num_conditions=1)

        def rand_fn_ave(shape):
            if self.noise_model in ["nb", "norm"]:
                theta = np.random.uniform(10, 1000, shape)
            elif self.noise_model in ["beta"]:
                theta = np.random.uniform(0.1, 0.9, shape)
            else:
                raise ValueError("noise model not recognized")
            return theta

        def rand_fn_loc(shape):
            return np.ones(shape)

        def rand_fn_scale(shape):
            theta = np.zeros(shape)
            if self.noise_model in ["nb"]:
                theta[0, :] = np.random.uniform(1, 3, shape[1])
            elif self.noise_model in ["norm"]:
                theta[0, :] = np.random.uniform(1, 2, shape[1])
            elif self.noise_model in ["beta"]:
                theta[0, :] = np.random.uniform(0.2, 0.4, shape[1])
            else:
                raise ValueError("noise model not recognized")
            return theta

        self.sim.generate_params(
            rand_fn_ave=lambda shape: rand_fn_ave(shape),
            rand_fn_loc=lambda shape: rand_fn_loc(shape),
            rand_fn_scale=lambda shape: rand_fn_scale(shape)
        )
        self.sim.generate_data()
        assert self.sim.input_data.design_loc.shape[1] == 1, "confounders include in intercept-only simulation"
        assert self.sim.input_data.design_scale.shape[1] == 1, "confounders include in intercept-only simulation"

    def _test_a_and_b(self, sparse, init_a, init_b):
        estimator = self.get_estimator(
            train_scale=False,
            sparse=sparse,
            init_a=init_a,
            init_b=init_b
        )
        estimator.estimator.initialize()
        estimator.estimator.finalize()
        success = estimator.eval_estimation_a(
            init_a=init_a,
        )
        assert success, "estimation for a_model was inaccurate"
        success = estimator.eval_estimation_b(
            init_b=init_b
        )
        assert success, "estimation for b_model was inaccurate"
        return True


class TestAccuracyAnalyticGlmNb(
    TestAccuracyAnalyticGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial data.
    """

    def test_a_closed_b_closed(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyAnalyticGlmNb.test_a_closed_b_closed()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate_complex()
        self._test_a_and_b(sparse=False, init_a="closed_form", init_b="closed_form")
        self._test_a_and_b(sparse=True, init_a="closed_form", init_b="closed_form")

    def test_a_standard_b_standard(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyAnalyticGlmNb.test_a_standard_b_standard()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate_easy()
        self._test_a_and_b(sparse=False, init_a="standard", init_b="standard")
        self._test_a_and_b(sparse=True, init_a="standard", init_b="standard")


class TestAccuracyAnalyticGlmNorm(
    TestAccuracyAnalyticGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for normally distributed data.
    """

    def test_a_closed_b_closed(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyAnalyticGlmNorm.test_a_closed_b_closed()")

        np.random.seed(1)
        self.noise_model = "norm"
        self.simulate_complex()
        self._test_a_and_b(sparse=False, init_a="closed_form", init_b="closed_form")
        self._test_a_and_b(sparse=True, init_a="closed_form", init_b="closed_form")

    def test_a_standard_b_standard(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyAnalyticGlmNorm.test_a_standard_b_standard()")

        np.random.seed(1)
        self.noise_model = "norm"
        self.simulate_easy()
        self._test_a_and_b(sparse=False, init_a="standard", init_b="standard")
        self._test_a_and_b(sparse=True, init_a="standard", init_b="standard")


class TestAccuracyAnalyticGlmBeta(
    TestAccuracyAnalyticGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for beta distributed data.
    """

    def test_a_closed_b_closed(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyAnalyticGlmBeta.test_a_closed_b_closed()")

        np.random.seed(1)
        self.noise_model = "beta"
        self.simulate_complex()
        self._test_a_and_b(sparse=False, init_a="closed_form", init_b="closed_form")
        self._test_a_and_b(sparse=True, init_a="closed_form", init_b="closed_form")

    def test_a_standard_b_standard(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyAnalyticGlmBeta.test_a_standard_b_standard()")

        np.random.seed(1)
        self.noise_model = "beta"
        self.simulate_easy()
        self._test_a_and_b(sparse=False, init_a="standard", init_b="standard")
        self._test_a_and_b(sparse=True, init_a="standard", init_b="standard")


if __name__ == '__main__':
    unittest.main()
