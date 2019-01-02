import logging
import unittest
import time

import batchglm.api as glm
import batchglm.data as data_utils
import batchglm.pkg_constants as pkg_constants

from batchglm.models.base_glm import _Estimator_GLM, _InputData_GLM, _Simulator_GLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)

class Test_Hessians_GLM_ALL(unittest.TestCase):
    noise_model: str
    sim: _Simulator_GLM
    estimator_fw: _Estimator_GLM
    estimator_ow: _Estimator_GLM
    estimator_tf: _Estimator_GLM
    estimator: _Estimator_GLM

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def estimate(
            self,
            input_data: _InputData_GLM
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.nb_glm import Estimator
            else:
                raise ValueError("noise_model not recognized")

        estimator = Estimator(
            input_data=input_data,
            termination_type="by_feature",
            noise_model=self.noise_model
        )
        estimator.initialize()
        estimator.train_sequence(training_strategy=[
            {
                "learning_rate": 0.1,
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": 1e-4,
                "use_batching": False,
                "optim_algo": "ADAM"  # Newton is very slow if hessian is evaluated through tf
            },
        ])
        return estimator

    def _test_compute_hessians(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.nb_glm import Simulator, InputData
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

        input_data = InputData.new(sim.X, design_loc=design_loc, design_scale=design_scale)

        logger.debug("* Running analytic Hessian by observation tests")
        pkg_constants.HESSIAN_MODE = "obs_batched"
        self.estimator_ob = self.estimate(input_data)
        t0_ob = time.time()
        self.H_ob = self.estimator_ob.hessians
        t1_ob = time.time()
        self.estimator_ob.close_session()
        self.t_ob = t1_ob - t0_ob

        logger.debug("* Running analytic Hessian by feature tests")
        pkg_constants.HESSIAN_MODE = "feature"
        self.estimator_fw = self.estimate(input_data)
        t0_fw = time.time()
        self.H_fw = self.estimator_fw.hessians
        t1_fw = time.time()
        self.estimator_fw.close_session()
        self.t_fw = t1_fw - t0_fw

        logger.debug("* Running tensorflow Hessian by feature tests")
        pkg_constants.HESSIAN_MODE = "tf"
        self.estimator_tf = self.estimate(input_data)
        t0_tf = time.time()
        # tensorflow computes the negative hessian as the
        # objective is the negative log-likelihood.
        self.H_tf = self.estimator_tf.hessians
        t1_tf = time.time()
        self.estimator_tf.close_session()
        self.t_tf = t1_tf - t0_tf

        i = 1
        logger.warning("run time observation batch-wise analytic solution: %f" % self.t_ob)
        logger.warning("run time feature-wise analytic solution: %f" % self.t_fw)
        logger.warning("run time feature-wise tensorflow solution: %f" % self.t_tf)
        logger.warning("ratio of tensorflow feature-wise hessian to analytic observation batch-wise hessian:")
        logger.warning(self.H_tf.values[i, :, :] / self.H_ob.values[i, :, :])
        logger.warning("ratio of tensorflow feature-wise hessian to analytic feature-wise hessian:")
        logger.warning(self.H_tf.values[i, :, :] / self.H_fw.values[i, :, :])


class Test_Hessians_GLM_NB(Test_Hessians_GLM_ALL, unittest.TestCase):

    def test_compute_hessians_nb(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.WARNING)

        self.noise_model = "nb"
        self._test_compute_hessians()

if __name__ == '__main__':
    unittest.main()
