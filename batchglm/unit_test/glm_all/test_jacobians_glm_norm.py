import logging
import unittest
import time
import numpy as np
import scipy.sparse

import batchglm.api as glm
import batchglm.data as data_utils
import batchglm.pkg_constants as pkg_constants

from batchglm.models.base_glm import InputData

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class Test_Jacobians_GLM_ALL(unittest.TestCase):
    noise_model: str

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def simulate(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "norm":
                from batchglm.api.models.glm_norm import Simulator
            else:
                raise ValueError("noise_model not recognized")

        num_observations = 500
        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=2, num_batches=2)
        sim.generate()

        self.sim = sim

    def get_jacs(
            self,
            input_data: InputData
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "norm":
                from batchglm.api.models.glm_norm import Estimator
            else:
                raise ValueError("noise_model not recognized")

        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": True, "nr_tr": True,
                              "irls": True, "irls_gd": True, "irls_tr": True, "irls_gd_tr": True}

        estimator = Estimator(
            input_data=input_data,
            quick_scale=False,
            provide_optimizers=provide_optimizers,
            init_a="standard",
            init_b="standard"
        )
        estimator.initialize()
        # Do not train, evaluate at initialization!
        estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "step",
                "stopping_criteria": 0,
                "use_batching": False,
                "optim_algo": "gd",
                "train_mean": False,
                "train_sd": False
            },
        ])
        estimator_store = estimator.finalize()
        return estimator_store.gradients.values

    def compare_jacs(
            self,
            design,
            sparse
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="norm":
                from batchglm.api.models.glm_norm import InputData
            else:
                raise ValueError("noise_model not recognized")

        sample_description = data_utils.sample_description_from_xarray(self.sim.data, dim="observations")
        design_loc = data_utils.design_matrix(sample_description, formula=design)
        design_scale = data_utils.design_matrix(sample_description, formula=design)

        if sparse:
            input_data = InputData.new(
                data=scipy.sparse.csr_matrix(self.sim.X),
                design_loc=design_loc,
                design_scale=design_scale
            )
        else:
            input_data = InputData.new(
                data=self.sim.X,
                design_loc=design_loc,
                design_scale=design_scale
            )

        logger.debug("** Running analytic Jacobian test")
        pkg_constants.JACOBIAN_MODE = "analytic"
        t0_analytic = time.time()
        J_analytic = self.get_jacs(input_data)
        t1_analytic = time.time()
        t_analytic = t1_analytic - t0_analytic

        logger.debug("** Running tensorflow Jacobian test")
        pkg_constants.JACOBIAN_MODE = "tf"
        t0_tf = time.time()
        J_tf = self.get_jacs(input_data)
        t1_tf = time.time()
        t_tf = t1_tf - t0_tf

        # Make sure that jacobians are not all zero which might make evaluation of equality difficult.
        assert np.sum(np.abs(J_analytic)) > 1e-10, \
            "jacobians too small to perform test: %f" % np.sum(np.abs(J_analytic))

        logging.getLogger("batchglm").info("run time tensorflow solution: %f" % t_tf)
        logging.getLogger("batchglm").info("run time observation batch-wise analytic solution: %f" % t_analytic)

        #print(J_tf)
        #print(J_analytic)
        #print((J_tf - J_analytic) / J_tf)

        max_rel_dev = np.max(np.abs((J_tf - J_analytic) / J_tf))
        assert max_rel_dev < 1e-12
        return True

    def _test_compute_jacobians(self, sparse):
        self.simulate()
        self.compare_jacs(design="~ 1 + condition + batch", sparse=sparse)


class Test_Jacobians_GLM_NORM(Test_Jacobians_GLM_ALL, unittest.TestCase):

    def test_compute_jacobians_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("Test_Jacobians_GLM_NORM.test_compute_jacobians_norm()")

        self.noise_model = "norm"
        self._test_compute_jacobians(sparse=False)
        #self._test_compute_jacobians(sparse=True)  #TODO automatic differentiation does not seem to work here yet.


if __name__ == '__main__':
    unittest.main()
