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
            sparse
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model == "nb":
                from batchglm.api.models.glm_nb import Estimator, InputDataGLM
            elif noise_model == "norm":
                from batchglm.api.models.glm_norm import Estimator, InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 200
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": True, "nr_tr": True,
                              "irls": True, "irls_gd": True, "irls_tr": True, "irls_gd_tr": True}

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
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            provide_batched=True,
            provide_fim=True,
            provide_hessian=True,
            init_a="standard",
            init_b="standard"
        )
        self.sim = simulator

    def estimate(
            self,
            algo,
            batched,
            acc,
    ):
        self.estimator.initialize()

        # Choose learning rate based on optimizer
        if algo.lower() in ["nr", "nr_tr", "irls", "irls_gd", "irls_tr", "irls_gd_tr"]:
            lr = 1
        else:
            lr = 0.05

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
            threshold_dev_a = 0.5
            threshold_dev_b = 0.5
            threshold_std_a = 1
            threshold_std_b = 20
        else:
            threshold_dev_a = 0.2
            threshold_dev_b = 0.2
            threshold_std_a = 1
            threshold_std_b = 2

        success = True
        if train_loc:
            mean_dev_a = np.mean(self.estimator.a_var - self.sim.a_var)
            std_dev_a = np.std(self.estimator.a_var - self.sim.a_var)

            logging.getLogger("batchglm").info("mean_dev_a %f" % mean_dev_a)
            logging.getLogger("batchglm").info("std_dev_a %f" % std_dev_a)

            if np.abs(mean_dev_a) > threshold_dev_a or std_dev_a > threshold_std_a:
                success = False
        if train_scale:
            mean_dev_b = np.mean(self.estimator.b_var - self.sim.b_var)
            std_dev_b = np.std(self.estimator.b_var - self.sim.b_var)

            logging.getLogger("batchglm").info("mean_dev_b %f" % mean_dev_b)
            logging.getLogger("batchglm").info("std_dev_b %f" % std_dev_b)

            if np.abs(mean_dev_b) > threshold_dev_b or std_dev_b > threshold_std_b:
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
    """
    noise_model: str

    def simulate(self):
        self.simulate1()
        self.simulate2()

    def get_simulator(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.glm_nb import Simulator
            elif self.noise_model == "norm":
                from batchglm.api.models.glm_norm import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=1000, num_features=500)

    def simulate1(self):
        self.sim1 = self.get_simulator()
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2)
        self.sim1.generate()

    def simulate2(self):
        self.sim2 = self.get_simulator()
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2)
        self.sim2.generate()

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
        if self.noise_model in ["norm", "beta"]:  # exponential family for which IRLS exists:
            algos = ["ADAM", "NR_TR", "IRLS_GD_TR"]
        elif self.noise_model in ["nb"]:
            algos = ["ADAM", "IRLS_GD_TR"]
        else:
            raise ValueError("noise model %s not recognized" % self.noise_model)
        estimator = _TestAccuracyGlmAllEstim(
            simulator=self.simulator(train_loc=train_loc),
            quick_scale=False if train_scale else True,
            noise_model=self.noise_model,
            sparse=sparse
        )
        for algo in algos:
            logger.info("algorithm: %s" % algo)
            if algo == "ADAM":
                acc = 1e-8
            elif algo == "NR_TR":
                acc = 1e-8
            elif algo == "IRLS_GD_TR":
                acc = 1e-10
            else:
                return ValueError("algo %s not recognized" % algo)
            estimator.estimate(
                algo=algo,
                batched=batched,
                acc=acc
            )
            estimator.estimator.finalize()
            success = estimator.eval_estimation(
                batched=False,
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
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNb.test_full_nb()")

        np.random.seed(1)
        self.noise_model = "nb"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
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
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNorm.test_full_norm()")

        np.random.seed(1)
        self.noise_model = "norm"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmNorm.test_batched_norm()")

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
    """

    def test_full_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmBeta.test_full_norm()")

        np.random.seed(1)
        self.noise_model = "beta"
        self.simulate()
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestAccuracyGlmBeta.test_batched_norm()")

        np.random.seed(1)
        self.noise_model = "beta"
        self.simulate()
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


if __name__ == '__main__':
    unittest.main()
