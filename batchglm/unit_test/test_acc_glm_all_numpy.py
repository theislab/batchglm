import logging
import numpy as np
import scipy.sparse
import unittest

import batchglm.api as glm

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _TestAccuracyGlmAllEstim:

    def __init__(
            self,
            simulator,
            quick_scale,
            noise_model,
            sparse,
            init_mode
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model == "nb":
                from batchglm.api.models.numpy.glm_nb import Estimator, InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        if sparse:
            input_data = InputDataGLM(
                data=scipy.sparse.csr_matrix(simulator.input_data.x),
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale,
                design_loc_names=simulator.input_data.design_loc_names,
                design_scale_names=simulator.input_data.design_scale_names,
                constraints_loc=simulator.input_data.constraints_loc,
                constraints_scale=simulator.input_data.constraints_scale,
                size_factors=simulator.input_data.size_factors
            )
        else:
            input_data = InputDataGLM(
                data=simulator.input_data.x,
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale,
                design_loc_names=simulator.input_data.design_loc_names,
                design_scale_names=simulator.input_data.design_scale_names,
                constraints_loc=simulator.input_data.constraints_loc,
                constraints_scale=simulator.input_data.constraints_scale,
                size_factors=simulator.input_data.size_factors
            )

        self.estimator = Estimator(
            input_data=input_data,
            quick_scale=quick_scale,
            init_a=init_mode,
            init_b=init_mode
        )
        self.sim = simulator

    def estimate(
            self
    ):
        self.estimator.initialize()
        self.estimator.train_sequence(training_strategy="DEFAULT")

    def eval_estimation(
            self,
            train_loc,
            train_scale
    ):
        threshold_dev_a = 0.2
        threshold_dev_b = 0.2
        threshold_std_a = 1
        threshold_std_b = 1

        success = True
        if train_loc:
            mean_rel_dev_a = np.mean((self.estimator.model.a_var - self.sim.a_var) / self.sim.a_var)
            std_rel_dev_a = np.std((self.estimator.model.a_var - self.sim.a_var) / self.sim.a_var)

            logging.getLogger("batchglm").info("mean_rel_dev_a %f" % mean_rel_dev_a)
            logging.getLogger("batchglm").info("std_rel_dev_a %f" % std_rel_dev_a)

            if np.abs(mean_rel_dev_a) > threshold_dev_a or std_rel_dev_a > threshold_std_a:
                success = False
        if train_scale:
            mean_rel_dev_b = np.mean((self.estimator.model.b_var - self.sim.b_var) / self.sim.b_var)
            std_rel_dev_b = np.std((self.estimator.model.b_var - self.sim.b_var) / self.sim.b_var)

            logging.getLogger("batchglm").info("mean_rel_dev_b %f" % mean_rel_dev_b)
            logging.getLogger("batchglm").info("std_rel_dev_b %f" % std_rel_dev_b)

            if np.abs(mean_rel_dev_b) > threshold_dev_b or std_rel_dev_b > threshold_std_b:
                success = False

        return success


class _TestAccuracyGlmAll(
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results.

    Accuracy is evaluted via deviation of simulated ground truth.
    The unit tests test individual training graphs and multiple optimizers
    (incl. one tensorflow internal optimizer and newton-rhapson)
    for each training graph. The training graphs tested are as follows:

     - full data model
        - train a and b model: test_full_global_a_and_b()
        - train a model only: test_full_global_a_only()
        - train b model only: test_full_global_b_only()
    - batched data model
        - train a and b model: test_batched_global_a_and_b()
        - train a model only: test_batched_global_a_only()
        - train b model only: test_batched_global_b_only()

    The unit tests throw an assertion error if the required accurcy is
    not met. Accuracy thresholds are fairly lenient so that unit_tests
    pass even with noise inherent in fast optimisation and random
    initialisation in simulation. Still, large biases (i.e. graph errors)
    should be discovered here.

    Note on settings by optimised:

    IRLS_TR: Needs slow TR collapse to converge.
    """
    noise_model: str
    optims_tested: dict

    def simulate(self):
        self.simulate1()
        self.simulate2()

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.numpy.glm_nb import Simulator
            elif self.noise_model == "norm":
                from batchglm.api.models import Simulator
            elif self.noise_model == "beta":
                from batchglm.api.models.numpy.glm_beta import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=1000, num_features=10)

    def simulate1(self):
        self.sim1 = self.get_simulator()
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2, intercept_scale=True)

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
            if self.noise_model in ["nb"]:
                theta = np.random.uniform(1, 3, shape)
            elif self.noise_model in ["norm"]:
                theta = np.random.uniform(1, 3, shape)
            elif self.noise_model in ["beta"]:
                theta = np.random.uniform(0, 0.15, shape)
            else:
                raise ValueError("noise model not recognized")
            return theta

        self.sim1.generate_params(
            rand_fn_ave=lambda shape: rand_fn_ave(shape),
            rand_fn_loc=lambda shape: rand_fn_loc(shape),
            rand_fn_scale=lambda shape: rand_fn_scale(shape)
        )
        self.sim1.generate_data()

    def simulate2(self):
        self.sim2 = self.get_simulator()
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2, intercept_scale=True)

        def rand_fn_ave(shape):
            if self.noise_model in ["nb", "norm"]:
                theta = np.random.uniform(10, 1000, shape)
            elif self.noise_model in ["beta"]:
                theta = np.random.uniform(0.1, 0.9, shape)
            else:
                raise ValueError("noise model not recognized")
            return theta

        def rand_fn_loc(shape):
            if self.noise_model in ["nb", "norm"]:
                theta = np.ones(shape)
            elif self.noise_model in ["beta"]:
                theta = np.zeros(shape)+0.05
            else:
                raise ValueError("noise model not recognized")
            return theta

        def rand_fn_scale(shape):
            if self.noise_model in ["nb"]:
                theta = np.ones(shape)
            elif self.noise_model in ["norm"]:
                theta = np.ones(shape)
            elif self.noise_model in ["beta"]:
                theta = np.ones(shape) - 0.8
            else:
                raise ValueError("noise model not recognized")
            return theta

        self.sim2.generate_params(
            rand_fn_ave=lambda shape: rand_fn_ave(shape),
            rand_fn_loc=lambda shape: rand_fn_loc(shape),
            rand_fn_scale=lambda shape: rand_fn_scale(shape)
        )
        self.sim2.generate_data()

    def simulator(self, train_loc):
        if train_loc:
            return self.sim1
        else:
            return self.sim2

    def basic_test(
            self,
            batched,
            train_loc,
            train_scale,
            sparse
    ):
        self.optims_tested = {
            "nb": ["IRLS"],
            "beta": ["IRLS"],
            "norm": ["IRLS"]
        }
        init_mode = "standard"

        for algo in self.optims_tested[self.noise_model]:
            logger.info("algorithm: %s" % algo)

            acc = 1e-14
            glm.pkg_constants.TRUST_REGION_T1 = 0.5
            glm.pkg_constants.TRUST_REGION_T2 = 1.5
            glm.pkg_constants.CHOLESKY_LSTSQS = True
            glm.pkg_constants.CHOLESKY_LSTSQS_BATCHED = True
            glm.pkg_constants.JACOBIAN_MODE = "analytic"

            estimator = _TestAccuracyGlmAllEstim(
                simulator=self.simulator(train_loc=train_loc),
                quick_scale=False if train_scale else True,
                noise_model=self.noise_model,
                sparse=sparse,
                init_mode=init_mode
            )
            estimator.estimate()
            estimator.estimator.finalize()
            success = estimator.eval_estimation(
                train_loc=train_loc,
                train_scale=train_scale,
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def _test_full_a_and_b(self, sparse):
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=True,
            sparse=sparse
        )

    def _test_full_a_only(self, sparse):
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=False,
            sparse=sparse
        )

    def _test_full_b_only(self, sparse):
        return self.basic_test(
            batched=False,
            train_loc=False,
            train_scale=True,
            sparse=sparse
        )

    def _test_batched_a_and_b(self, sparse):
        return self.basic_test(
            batched=True,
            train_loc=True,
            train_scale=True,
            sparse=sparse
        )

    def _test_batched_a_only(self, sparse):
        return self.basic_test(
            batched=True,
            train_loc=True,
            train_scale=False,
            sparse=sparse
        )

    def _test_batched_b_only(self, sparse):
        return self.basic_test(
            batched=True,
            train_loc=False,
            train_scale=True,
            sparse=sparse
        )

    def _test_full(self, sparse):
        self._test_full_a_and_b(sparse=sparse)
        self._test_full_a_only(sparse=sparse)
        self._test_full_b_only(sparse=sparse)

    def _test_batched(self, sparse):
        self._test_batched_a_and_b(sparse=sparse)
        self._test_batched_a_only(sparse=sparse)
        self._test_batched_b_only(sparse=sparse)


class TestAccuracyGlmNb(
    _TestAccuracyGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial distributed data.
    """

    def test_full_nb(self):
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNb.test_full_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)


if __name__ == '__main__':
    unittest.main()
