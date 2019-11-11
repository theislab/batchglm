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
                from batchglm.api.models.tf1.glm_nb import Estimator, InputDataGLM
            elif noise_model == "norm":
                from batchglm.api.models import Estimator, InputDataGLM
            elif noise_model == "beta":
                from batchglm.api.models.tf1.glm_beta import Estimator, InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 2000
        provide_optimizers = {
            "gd": True,
            "adam": True,
            "adagrad": True,
            "rmsprop": True,
            "nr": True,
            "nr_tr": True,
            "irls": noise_model in ["nb", "norm"],
            "irls_gd": noise_model in ["nb", "norm"],
            "irls_tr": noise_model in ["nb", "norm"],
            "irls_gd_tr": noise_model in ["nb", "norm"]
        }

        if sparse:
            input_data = InputDataGLM(
                data=scipy.sparse.csr_matrix(simulator.input_data.x),
                design_loc=simulator.input_data.design_loc,
                design_scale=simulator.input_data.design_scale,
                design_loc_names=simulator.input_data.design_loc_names,
                design_scale_names=simulator.input_data.design_scale_names,
                constraints_loc=simulator.input_data.constraints_loc,
                constraints_scale=simulator.input_data.constraints_scale,
                size_factors=simulator.input_data.size_factors,
                as_dask=False
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
                size_factors=simulator.input_data.size_factors,
                as_dask=False
            )

        self.estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            provide_batched=True,
            provide_fim=noise_model in ["nb", "norm"],
            provide_hessian=True,
            init_a=init_mode,
            init_b=init_mode
        )
        self.sim = simulator

    def estimate(
            self,
            algo,
            batched,
            acc,
            lr
    ):
        self.estimator.initialize()
        self.estimator.train_sequence(training_strategy=[
            {
                "learning_rate": lr,
                "convergence_criteria": "all_converged",
                "stopping_criteria": acc,
                "use_batching": batched,
                "optim_algo": algo,
            },
        ])

    def eval_estimation(
            self,
            batched,
            train_loc,
            train_scale
    ):
        if batched:
            threshold_dev_a = 0.4
            threshold_dev_b = 0.4
            threshold_std_a = 2
            threshold_std_b = 2
        else:
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
                from batchglm.api.models.tf1.glm_nb import Simulator
            elif self.noise_model == "norm":
                from batchglm.api.models import Simulator
            elif self.noise_model == "beta":
                from batchglm.api.models.tf1.glm_beta import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=10000, num_features=10)

    def simulate1(self):
        self.sim1 = self.get_simulator()
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2)

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
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2)

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
            "nb": ["ADAM", "IRLS_GD_TR"],
            "beta": ["NR_TR"],
            "norm": ["IRLS_TR"]
        }
        if self.noise_model in ["norm"]:
            algos = self.optims_tested["norm"]
            init_mode = "all_zero"
            lr = {"ADAM": 1e-3, "NR_TR": 1, "IRLS_TR": 1}
        elif self.noise_model in ["beta"]:
            algos = self.optims_tested["beta"]
            init_mode = "all_zero"
            if batched:
                lr = {"ADAM": 0.1, "NR_TR": 1}
            else:
                lr = {"ADAM": 1e-5, "NR_TR": 1}
        elif self.noise_model in ["nb"]:
            algos = self.optims_tested["nb"]
            init_mode = "standard"
            if batched:
                lr = {"ADAM": 0.1, "IRLS_GD_TR": 1}
            else:
                lr = {"ADAM": 0.05, "IRLS_GD_TR": 1}
        else:
            raise ValueError("noise model %s not recognized" % self.noise_model)

        for algo in algos:
            logger.info("algorithm: %s" % algo)
            if algo in ["ADAM", "RMSPROP", "GD"]:
                if batched:
                    acc = 1e-4
                else:
                    acc = 1e-6
                glm.pkg_constants.JACOBIAN_MODE = "analytic"
            elif algo in ["NR", "NR_TR"]:
                if batched:
                    acc = 1e-12
                else:
                    acc = 1e-14
                if self.noise_model in ["beta"]:
                    glm.pkg_constants.TRUST_REGION_RADIUS_INIT = 1
                else:
                    glm.pkg_constants.TRUST_REGION_RADIUS_INIT = 100
                glm.pkg_constants.TRUST_REGION_T1 = 0.5
                glm.pkg_constants.TRUST_REGION_T2 = 1.5
                glm.pkg_constants.CHOLESKY_LSTSQS = True
                glm.pkg_constants.CHOLESKY_LSTSQS_BATCHED = True
                glm.pkg_constants.JACOBIAN_MODE = "analytic"
                glm.pkg_constants.HESSIAN_MODE = "analytic"
            elif algo in ["IRLS", "IRLS_TR", "IRLS_GD", "IRLS_GD_TR"]:
                if batched:
                    acc = 1e-12
                else:
                    acc = 1e-14
                glm.pkg_constants.TRUST_REGION_T1 = 0.5
                glm.pkg_constants.TRUST_REGION_T2 = 1.5
                glm.pkg_constants.CHOLESKY_LSTSQS = True
                glm.pkg_constants.CHOLESKY_LSTSQS_BATCHED = True
                glm.pkg_constants.JACOBIAN_MODE = "analytic"
            else:
                return ValueError("algo %s not recognized" % algo)
            estimator = _TestAccuracyGlmAllEstim(
                simulator=self.simulator(train_loc=train_loc),
                quick_scale=False if train_scale else True,
                noise_model=self.noise_model,
                sparse=sparse,
                init_mode=init_mode
            )
            estimator.estimate(
                algo=algo,
                batched=batched,
                acc=acc,
                lr=lr[algo]
            )
            estimator.estimator.finalize()
            success = estimator.eval_estimation(
                batched=batched,
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
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNb.test_full_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNb.test_batched_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


class TestAccuracyGlmNorm(
    _TestAccuracyGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for normal distributed data.
    """

    def test_full_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNorm.test_full_norm()")

        np.random.seed(1)
        self.noise_model = "norm"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNorm.test_batched_norm()")
        # TODO not working yet.

        np.random.seed(1)
        self.noise_model = "norm"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


class TestAccuracyGlmBeta(
    _TestAccuracyGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for beta distributed data.
    TODO not working yet.
    """

    def test_full_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmBeta.test_full_beta()")

        np.random.seed(1)
        self.noise_model = "beta"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmBeta.test_batched_beta()")

        np.random.seed(1)
        self.noise_model = "beta"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


if __name__ == '__main__':
    unittest.main()
