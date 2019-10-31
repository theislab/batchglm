import logging
import unittest
import time
import numpy as np
import scipy.sparse

import batchglm.api as glm
import batchglm.data as data_utils
import batchglm.pkg_constants as pkg_constants

from batchglm.models.base_glm import InputDataGLM


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
            elif self.noise_model == "norm":
                from batchglm.api.models.glm_norm import Simulator
            elif self.noise_model == "beta":
                from batchglm.api.models.glm_beta import Simulator
            else:
                raise ValueError("noise_model not recognized")

        num_observations = 500
        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=2, num_batches=2)
        sim.generate()

        self.sim = sim

    def get_jacs(
            self,
            input_data: InputDataGLM
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.glm_nb import Estimator
            elif self.noise_model == "norm":
                from batchglm.api.models.glm_norm import Estimator
            elif self.noise_model == "beta":
                from batchglm.api.models.glm_beta import Estimator
            else:
                raise ValueError("noise_model not recognized")

        estimator = Estimator(
            input_data=input_data,
            init_a=self.sim.a_var,
            init_b=self.sim.b_var
        )
        estimator.initialize()
        # Do not train, evaluate at initialization!
        estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "step",
                "stopping_criteria": 1,
                "use_batching": False,
                "optim_algo": "gd",
                "train_mu": True,
                "train_r": True,
                "autograd": pkg_constants.JACOBIAN_MODE == "tf"
            },
        ])
        estimator.finalize()
        return estimator.jacobian

    def compare_jacs(
            self,
            design,
            sparse
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.glm_nb import InputDataGLM
            elif self.noise_model == "norm":
                from batchglm.api.models.glm_norm import InputDataGLM
            elif self.noise_model == "beta":
                from batchglm.api.models.glm_beta import InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        sample_description = self.sim.sample_description
        design_loc = data_utils.design_matrix(sample_description, formula=design)
        design_scale = data_utils.design_matrix(sample_description, formula=design)

        if sparse:
            input_data = InputDataGLM(
                data=scipy.sparse.csr_matrix(self.sim.x),
                design_loc=design_loc,
                design_scale=design_scale
            )
        else:
            input_data = InputDataGLM(
                data=self.sim.x,
                design_loc=design_loc,
                design_scale=design_scale
            )

        logging.getLogger("batchglm").debug("** Running analytic Jacobian test")
        pkg_constants.JACOBIAN_MODE = "analytic"
        t0_analytic = time.time()
        J_analytic = self.get_jacs(input_data)
        t1_analytic = time.time()
        t_analytic = t1_analytic - t0_analytic

        logging.getLogger("batchglm").debug("** Running tensorflow Jacobian test")
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
        logging.getLogger("batchglm").info("MAD: %f" % np.max(np.abs((J_tf - J_analytic))))
        logging.getLogger("batchglm").info("MRAD: %f" % np.max(np.abs((J_tf - J_analytic) / J_tf)))

        #print(J_tf)
        #print(J_analytic)
        #print((J_tf - J_analytic) / J_tf)

        mrad = np.max(np.abs((J_tf - J_analytic) / J_tf))
        assert mrad < 1e-10, mrad # changed 1e-12 to 1e-10
        return True

    def _test_compute_jacobians(self, sparse):
        self.simulate()
        self.compare_jacs(design="~ 1 + condition + batch", sparse=sparse)


class Test_Jacobians_GLM_NB(Test_Jacobians_GLM_ALL, unittest.TestCase):

    def test_compute_jacobians_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logging.getLogger("batchglm").error("Test_Jacobians_GLM_NB.test_compute_jacobians_nb()")

        self.noise_model = "nb"
        self._test_compute_jacobians(sparse=False)
        #self._test_compute_jacobians(sparse=True)  #TODO automatic differentiation does not seems to work here yet.

"""
class Test_Jacobians_GLM_NORM(Test_Jacobians_GLM_ALL, unittest.TestCase):

    def test_compute_jacobians_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logging.getLogger("batchglm").error("Test_Jacobians_GLM_NORM.test_compute_jacobians_norm()")

        self.noise_model = "norm"
        self._test_compute_jacobians(sparse=False)
        #self._test_compute_jacobians(sparse=True)  #TODO automatic differentiation does not seem to work here yet.

class Test_Jacobians_GLM_BETA(Test_Jacobians_GLM_ALL, unittest.TestCase):

    def test_compute_jacobians_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logging.getLogger("batchglm").error("Test_Jacobians_GLM_BETA.test_compute_jacobians_beta()")

        self.noise_model = "beta"
        self._test_compute_jacobians(sparse=False)
        #self._test_compute_jacobians(sparse=True)  #TODO automatic differentiation does not seem to work here yet.
"""

if __name__ == '__main__':
    unittest.main()
