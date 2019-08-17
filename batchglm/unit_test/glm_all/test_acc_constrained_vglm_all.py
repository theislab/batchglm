import logging
import numpy as np
from typing import List
import unittest

import batchglm.api as glm
from batchglm.models.base_glm import _EstimatorGLM


class _Test_AccuracyConstrained_VGLM_ALL_Estim:

    def __init__(
            self,
            simulator,
            train_scale,
            noise_model,
            constraints_loc,
            constraints_scale
    ):
        if noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if noise_model=="nb":
                from batchglm.api.models.glm_nb import Estimator, InputDataGLM
            elif noise_model=="norm":
                from batchglm.api.models.glm_norm import Estimator, InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 500
        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": True, "nr_tr": True,
                              "irls": True, "irls_gd": True, "irls_tr": True, "irls_gd_tr": True}

        input_data = simulator.input_data
        design_loc = np.hstack([
            input_data.design_loc.values,
            np.expand_dims(input_data.design_loc.values[:, 0]-input_data.design_loc.values[:, -1], axis=-1)
        ])
        design_scale = design_loc.copy()
        input_data = InputDataGLM.new(
            data=simulator.X,
            design_loc=design_loc,
            design_scale=design_scale,
            constraints_loc=constraints_loc,
            constraints_scale=constraints_scale
        )

        estimator = Estimator(
            input_data=input_data,
            batch_size=batch_size,
            quick_scale=not train_scale,
            provide_optimizers=provide_optimizers,
            provide_batched=True,
            provide_fim=True,
            provide_hessian=True
        )

        self.estimator = estimator
        self.sim = simulator

    def eval_estimation(
            self,
            estimator_store,
            batched
    ):
        if batched:
            threshold_dev_a = 0.5
            threshold_dev_b = 0.5
            threshold_std_a = 1
            threshold_std_b = 20
        else:
            threshold_dev_a = 0.3
            threshold_dev_b = 0.3
            threshold_std_a = 1
            threshold_std_b = 2

        mean_dev_a = np.mean(estimator_store.a_var.values - self.sim.a_var.values)
        std_dev_a = np.std(estimator_store.a_var.values - self.sim.a_var.values)
        mean_dev_b = np.mean(estimator_store.b_var.values - self.sim.b_var.values)
        std_dev_b = np.std(estimator_store.b_var.values - self.sim.b_var.values)

        logging.getLogger("batchglm").info("mean_dev_a %f" % mean_dev_a)
        logging.getLogger("batchglm").info("std_dev_a %f" % std_dev_a)
        logging.getLogger("batchglm").info("mean_dev_b %f" % mean_dev_b)
        logging.getLogger("batchglm").info("std_dev_b %f" % std_dev_b)

        if np.abs(mean_dev_a) < threshold_dev_a and \
                np.abs(mean_dev_b) < threshold_dev_b and \
                std_dev_a < threshold_std_a and \
                std_dev_b < threshold_std_b:
            return True
        else:
            return False

class Test_AccuracyConstrained_VGLM_ALL(unittest.TestCase):
    noise_model: str
    _estims: List[_EstimatorGLM]

    def simulate(self):
        self.simulate1()
        self.simulate2()

    def _simulate(self, num_batches):
        sim = self.get_simulator()
        sim.generate_sample_description(num_batches=num_batches, num_conditions=2)
        sim.generate_params()
        sim.size_factors = np.random.uniform(0.1, 2, size=sim.nobs)
        sim.generate_data()
        logging.getLogger("batchglm").debug("Size factor standard deviation % f" %
                                            np.std(sim.size_factors.data))
        return sim

    def simulate1(self):
        self.sim1 = self._simulate(num_batches=2)

    def simulate2(self):
        self.sim2 = self._simulate(num_batches=0)

    def simulator(self, train_loc):
        if train_loc:
            return self.sim1
        else:
            return self.sim2

    def basic_test(
            self,
            batched,
            train_loc,
            train_scale
    ):
        if batched:
            ts = ["IRLS_BATCHED"]
        else:
            ts = ["IRLS"]

        # Encode equality constrained on overdetermined confounder coefficient.
        if train_loc:
            constraints = np.zeros([4, 3])
            constraints[0, 0] = 1
            constraints[1, 1] = 1
            constraints[2, 2] = 1
            constraints[3, 2] = -1
        else:
            constraints = np.zeros([3, 2])
            constraints[0, 0] = 1
            constraints[1, 1] = 1
            constraints[2, 1] = -1

        for x in ts:
            logging.getLogger("batchglm").debug("algorithm: %s" % x)
            estimator = _Test_AccuracyConstrained_VGLM_ALL_Estim(
                simulator=self.simulator(train_loc=train_loc),
                train_scale=train_scale,
                noise_model=self.noise_model,
                constraints_loc=constraints,
                constraints_scale=constraints,
            )
            estimator.estimator.initialize()
            estimator.estimator.train_sequence(training_strategy=ts)
            estimator_store = estimator.estimator.finalize()
            success = estimator.eval_estimation(
                estimator_store=estimator_store,
                batched=batched
            )
            assert success, "%s did not yield exact results" % x

        return True

    def _test_full_a_and_b(self):
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=True
        )

    def _test_full_a_only(self):
        return self.basic_test(
            batched=False,
            train_loc=True,
            train_scale=False
        )

    def _test_full_b_only(self):
        return self.basic_test(
            batched=False,
            train_loc=False,
            train_scale=True
        )

    def _test_batched_a_and_b(self):
        return self.basic_test(
            batched=True,
            train_loc=True,
            train_scale=True
        )

    def _test_batched_a_only(self):
        return self.basic_test(
            batched=True,
            train_loc=True,
            train_scale=False
        )

    def _test_batched_b_only(self):
        return self.basic_test(
            batched=True,
            train_loc=False,
            train_scale=True
        )

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

        return Simulator(num_observations=10000, num_features=10)


    def _test_full(self):
        self.simulate()
        logging.getLogger("batchglm").debug("* Running tests for full data")
        self._test_full_a_and_b()
        self._test_full_a_only()
        self._test_full_b_only()

    def _test_batched(self):
        self.simulate()
        logging.getLogger("batchglm").debug("* Running tests for batched data")
        self._test_batched_a_and_b()
        self._test_batched_a_only()
        self._test_batched_b_only()


class Test_AccuracyConstrained_VGLM_NB(
    Test_AccuracyConstrained_VGLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial noise.
    """

    def test_full_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_AccuracyConstrained_VGLM_NB.test_full_nb()")

        self.noise_model = "nb"
        self._test_full()

    def test_batched_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_AccuracyConstrained_VGLM_NB.test_batched_nb()")

        self.noise_model = "nb"
        self._test_batched()

class Test_AccuracyConstrained_VGLM_NORM(
    Test_AccuracyConstrained_VGLM_ALL,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for normal distributed noise.
    """

    def test_full_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_AccuracyConstrained_VGLM_NORM.test_full_norm()")

        self.noise_model = "norm"
        self._test_full()

    def test_batched_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_AccuracyConstrained_VGLM_NORM.test_batched_norm()")

        self.noise_model = "norm"
        self._test_batched()


if __name__ == '__main__':
    unittest.main()
