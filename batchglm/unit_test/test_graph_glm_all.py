import unittest
import logging
import scipy.sparse

import batchglm.api as glm

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _TestGraphGlmAllEstim:

    def __init__(
            self,
            simulator,
            quick_scale,
            algo,
            batched,
            noise_model,
            sparse
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model == "nb":
                from batchglm.api.models.numpy.glm_nb import Estimator, InputDataGLM
            elif noise_model == "norm":
                from batchglm.api.models import Estimator, InputDataGLM
            elif noise_model == "beta":
                from batchglm.api.models.numpy.glm_beta import Estimator, InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 200
        provide_optimizers = {
            "gd": False, "adam": False, "adagrad": False, "rmsprop": False,
            "nr": False, "nr_tr": False,
            "irls": False, "irls_gd": False, "irls_tr": False, "irls_gd_tr": False
        }
        provide_optimizers[algo.lower()] = True

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

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers,
            provide_batched=batched,
            optim_algos=[algo.lower()]
        )
        self.estimator = estimator
        self.sim = simulator
        self.algo = algo.lower()

    def estimate(
            self,
            batched
        ):
        self.estimator.initialize()

        self.estimator.train_sequence(training_strategy=[
            {
                "learning_rate": 1,
                "convergence_criteria": "step",
                "stopping_criteria": 1,
                "use_batching": batched,
                "optim_algo": self.algo,
            },
        ])


class _TestGraphGlmAll:
    """
    Test whether training graphs work.

    Quick tests which simply passes small data sets through
    all possible training graphs to check whether there are graph
    bugs. This is all tested in test_acc_glm.py but this
    set of unit_tests runs much faster and does not abort due
    to accuracy outliers. The training graphs covered are:

     - full data model
        - train a and b model: test_full_global_a_and_b()
        - train a model only: test_full_global_a_only()
        - train b model only: test_full_global_b_only()
    - batched data model
        - train a and b model: test_batched_global_a_and_b()
        - train a model only: test_batched_global_a_only()
        - train b model only: test_batched_global_b_only()
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
                from batchglm.api.models.numpy.glm_nb import Simulator
            elif self.noise_model == "norm":
                from batchglm.api.models import Simulator
            elif self.noise_model == "beta":
                from batchglm.api.models.numpy.glm_beta import Simulator
            else:
                raise ValueError("noise_model not recognized")

        return Simulator(num_observations=200, num_features=2)

    def simulate1(self):
        self.sim1 = self.get_simulator()
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2, intercept_scale=True)
        self.sim1.generate()

    def simulate2(self):
        self.sim2 = self.get_simulator()
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2, intercept_scale=True)
        self.sim2.generate()

    def simulator(self, train_loc):
        if train_loc:
            return self.sim1
        else:
            return self.sim2

    def basic_test_one_algo(
            self,
            batched,
            train_loc,
            train_scale,
            algo,
            sparse
    ):
        estimator = _TestGraphGlmAllEstim(
            simulator=self.simulator(train_loc=train_loc),
            quick_scale=False if train_scale else True,
            algo=algo,
            batched=batched,
            noise_model=self.noise_model,
            sparse=sparse
        )
        estimator.estimate(batched=batched)
        estimator.estimator.finalize()
        return True

    def basic_test(
            self,
            batched,
            train_loc,
            train_scale,
            sparse
    ):
        if self.noise_model == "nb":
            algos = ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR", "NR_TR", "IRLS", "IRLS_GD", "IRLS_TR", "IRLS_GD_TR"]
        elif self.noise_model == "norm":
            algos = ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR", "NR_TR", "IRLS", "IRLS_TR"]
        elif self.noise_model == "beta":
            algos = ["GD", "ADAM", "ADAGRAD", "RMSPROP", "NR", "NR_TR"]
        else:
            raise ValueError("noise model %s not recognized" % self.noise_model)
        for algo in algos:
            logger.info("algorithm: %s" % algo)
            self.basic_test_one_algo(
                batched=batched,
                train_loc=train_loc,
                train_scale=train_scale,
                algo=algo,
                sparse=sparse
            )

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
        self.simulate()
        self._test_full_a_and_b(sparse=sparse)
        self._test_full_a_only(sparse=sparse)
        self._test_full_b_only(sparse=sparse)

    def _test_batched(self, sparse):
        self.simulate()
        self._test_batched_a_and_b(sparse=sparse)
        self._test_batched_a_only(sparse=sparse)
        self._test_batched_b_only(sparse=sparse)


class TestGraphGlmNb(
    _TestGraphGlmAll,
    unittest.TestCase
):
    """
    Test whether training graphs work for negative binomial noise.
    """

    def test_full_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestGraphGlmNb.test_full_nb()")

        self.noise_model = "nb"
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestGraphGlmNb.test_batched_nb()")

        self.noise_model = "nb"
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


class TestGraphGlmNorm(
    _TestGraphGlmAll,
    unittest.TestCase
):
    """
    Test whether training graphs work for normally distributed noise.
    """

    def test_full_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestGraphGlmNorm.test_full_norm()")

        self.noise_model = "norm"
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestGraphGlmNorm.test_batched_norm()")

        self.noise_model = "norm"
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


class TestGraphGlmBeta(
    _TestGraphGlmAll,
    unittest.TestCase
):
    """
    Test whether training graphs work for beta distributed noise.
    """

    def test_full_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.ERROR)
        logger.error("TestGraphGlmBeta.test_full_beta()")

        self.noise_model = "beta"
        self._test_full(sparse=False)
        self._test_full(sparse=True)

    def test_batched_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logger.error("TestGraphGlmBeta.test_batched_beta()")

        self.noise_model = "beta"
        self._test_batched(sparse=False)
        self._test_batched(sparse=True)


if __name__ == '__main__':
    unittest.main()
