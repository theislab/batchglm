import logging
import unittest
import time
import numpy as np
import scipy.sparse

import batchglm.data as data_utils
import batchglm.pkg_constants as pkg_constants

from batchglm.models.base_glm import InputDataGLM


class Test_Hessians_GLM_ALL(unittest.TestCase):
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
                from batchglm.api.models.tf1.glm_nb import Simulator
            elif self.noise_model == "norm":
                from batchglm.api.models import Simulator
            elif self.noise_model == "beta":
                from batchglm.api.models.tf1.glm_beta import Simulator
            else:
                raise ValueError("noise_model not recognized")

        num_observations = 500
        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=2, num_batches=2)
        sim.generate()

        self.sim = sim

    def get_hessians(
            self,
            input_data: InputDataGLM
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.tf1.glm_nb import Estimator
            elif self.noise_model == "norm":
                from batchglm.api.models import Estimator
            elif self.noise_model == "beta":
                from batchglm.api.models.tf1.glm_beta import Estimator
            else:
                raise ValueError("noise_model not recognized")

        provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True,
                              "nr": False, "nr_tr": False,
                              "irls": False, "irls_gd": False, "irls_tr": False, "irls_gd_tr": False}

        estimator = Estimator(
            input_data=input_data,
            quick_scale=False,
            provide_optimizers=provide_optimizers,
            provide_fim=False,
            provide_hessian=False,
            init_a="standard",
            init_b="standard"
        )
        estimator.initialize()
        estimator_store = estimator.finalize()

        return - estimator_store.fisher_inv

    def _test_compute_hessians(self, sparse):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.tf1.glm_nb import Simulator, InputDataGLM
            elif self.noise_model == "norm":
                from batchglm.api.models import Simulator, InputDataGLM
            elif self.noise_model == "beta":
                from batchglm.api.models.tf1.glm_beta import Simulator, InputDataGLM
            else:
                raise ValueError("noise_model not recognized")

        num_observations = 500
        num_conditions = 2

        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=num_conditions, num_batches=2)
        sim.generate()

        sample_description = data_utils.sample_description_from_xarray(sim.data, dim="observations")
        design_loc = data_utils.design_matrix(sample_description, formula="~ 1 + condition + batch")
        design_scale = data_utils.design_matrix(sample_description, formula="~ 1 + condition")

        if sparse:
            input_data = InputDataGLM(
                data=scipy.sparse.csr_matrix(sim.X),
                design_loc=design_loc,
                design_scale=design_scale
            )
        else:
            input_data = InputDataGLM(
                data=sim.X,
                design_loc=design_loc,
                design_scale=design_scale
            )

        # Compute hessian based on analytic solution.
        pkg_constants.HESSIAN_MODE = "analytic"
        t0_analytic = time.time()
        h_analytic = self.get_hessians(input_data)
        t1_analytic = time.time()
        t_analytic = t1_analytic - t0_analytic

        # Compute hessian based on tensorflow auto-differentiation.
        pkg_constants.HESSIAN_MODE = "tf1"
        t0_tf = time.time()
        h_tf = self.get_hessians(input_data)
        t1_tf = time.time()
        t_tf = t1_tf - t0_tf

        logging.getLogger("batchglm").info("run time observation batch-wise analytic solution: %f" % t_analytic)
        logging.getLogger("batchglm").info("run time tensorflow solution: %f" % t_tf)
        logging.getLogger("batchglm").info("MAD: %f" % np.max(np.abs((h_tf - h_analytic))))

        #i = 1
        #print(h_tf[i, :, :])
        #print(h_analytic[i, :, :])
        #print(h_tf[i, :, :] - h_analytic[i, :, :])

        # Make sure that hessians are not all zero which might make evaluation of equality difficult.
        assert np.sum(np.abs(h_analytic)) > 1e-10, \
            "hessians too small to perform test: %f" % np.sum(np.abs(h_analytic))
        mad = np.max(np.abs(h_tf - h_analytic))
        assert mad < 1e-15, mad
        return True


class Test_Hessians_GLM_NB(Test_Hessians_GLM_ALL, unittest.TestCase):

    def test_compute_hessians_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_Hessians_GLM_NB.test_compute_hessians_nb()")

        self.noise_model = "nb"
        self._test_compute_hessians(sparse=False)
        #self._test_compute_hessians(sparse=False)  # TODO tf1>=1.13 waiting for tf1.sparse.expand_dims to work

        return True


class Test_Hessians_GLM_NORM(Test_Hessians_GLM_ALL, unittest.TestCase):

    def test_compute_hessians_norm(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_Hessians_GLM_NORM.test_compute_hessians_norm()")

        self.noise_model = "norm"
        self._test_compute_hessians(sparse=False)
        #self._test_compute_hessians(sparse=False)  # TODO tf1>=1.13 waiting for tf1.sparse.expand_dims to work

        return True


class Test_Hessians_GLM_BETA(Test_Hessians_GLM_ALL, unittest.TestCase):

    def test_compute_hessians_beta(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)
        logging.getLogger("batchglm").error("Test_Hessians_GLM_BETA.test_compute_hessians_beta()")

        self.noise_model = "beta"
        self._test_compute_hessians(sparse=False)
        #self._test_compute_hessians(sparse=False)  # TODO tf1>=1.13 waiting for tf1.sparse.expand_dims to work

        return True


if __name__ == '__main__':
    unittest.main()
