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
            if self.noise_model == "nb":
                from batchglm.api.models.glm_nb import Simulator
            else:
                raise ValueError("noise_model not recognized")

        num_observations = 500
        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=2, num_batches=2)
        sim.generate()

        self.sim = sim

    def estimate(
            self,
            input_data: InputData,
            quick_scale
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.glm_nb import Estimator
            else:
                raise ValueError("noise_model not recognized")

        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": True, "nr_tr": True, "irls": True, "irls_tr": True}

        estimator = Estimator(
            input_data=input_data,
            quick_scale=quick_scale,
            provide_optimizers=provide_optimizers
        )
        estimator.initialize()
        # Do not train, evalute at initialization!
        return estimator

    def compare_jacs(
            self,
            design,
            quick_scale,
            sparse
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.glm_nb import InputData
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
        estimator_analytic = self.estimate(input_data, quick_scale)
        t0_analytic = time.time()
        J_analytic = estimator_analytic['gradients']
        a_analytic = estimator_analytic.a.values
        b_analytic = estimator_analytic.b.values
        t1_analytic = time.time()
        estimator_analytic.close_session()
        t_analytic = t1_analytic - t0_analytic

        logger.debug("** Running tensorflow Jacobian test")
        pkg_constants.JACOBIAN_MODE = "tf"
        estimator_tf = self.estimate(input_data, quick_scale)
        t0_tf = time.time()
        J_tf = estimator_tf['gradients']
        a_tf = estimator_tf.a.values
        b_tf = estimator_tf.b.values
        t1_tf = time.time()
        estimator_tf.close_session()
        t_tf = t1_tf - t0_tf

        i = 1
        logger.info("run time tensorflow solution: %f" % t_tf)
        logger.info("run time observation batch-wise analytic solution: %f" % t_analytic)
        logger.info("relative difference of mean estimates for analytic jacobian to observation-wise jacobian:")
        logger.info((a_analytic - a_tf) / a_tf)
        logger.info("relative difference of dispersion estimates for analytic jacobian to observation-wise jacobian:")
        logger.info((b_analytic - b_tf) / b_tf)
        logger.info("relative difference of analytic jacobian to analytic observation-wise jacobian:")
        logger.info((J_tf - J_analytic) / J_tf)

        max_rel_dev = np.max(np.abs((J_tf - J_analytic) / J_tf))
        assert max_rel_dev < 1e-10
        return True

    def _test_compute_jacobians_dense(self):
        logger.debug("* Running Jacobian tests dense data")
        self.simulate()
        self._test_compute_jacobians_a_and_b(sparse=False)
        self._test_compute_jacobians_a_only(sparse=False)
        self._test_compute_jacobians_b_only(sparse=False)

    def _test_compute_jacobians_sparse(self):
        logger.debug("* Running Jacobian tests sparse data")
        self.simulate()
        self._test_compute_jacobians_a_and_b(sparse=True)
        self._test_compute_jacobians_a_only(sparse=True)
        self._test_compute_jacobians_b_only(sparse=True)

    def _test_compute_jacobians_a_and_b(self, sparse):
        logger.debug("* Running Jacobian tests for a and b training")
        return self.compare_jacs(
            design="~ 1 + condition + batch",
            quick_scale=False,
            sparse=sparse
        )

    def _test_compute_jacobians_a_only(self, sparse):
        logger.debug("* Running Jacobian tests for a only training")
        return self.compare_jacs(
            design="~ 1 + condition + batch",
            quick_scale=True,
            sparse=sparse
        )

    def _test_compute_jacobians_b_only(self, sparse):
        logger.debug("* Running Jacobian tests for b only training")
        return self.compare_jacs(
            design="~ 1 + condition",
            quick_scale=False,
            sparse=sparse
        )


class Test_Jacobians_GLM_NB(Test_Jacobians_GLM_ALL, unittest.TestCase):

    def test_compute_jacobians_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("Test_Jacobians_GLM_NB.test_compute_jacobians_nb()")

        self.noise_model = "nb"
        #self._test_compute_jacobians_dense()
        self._test_compute_jacobians_sparse()


if __name__ == '__main__':
    unittest.main()
