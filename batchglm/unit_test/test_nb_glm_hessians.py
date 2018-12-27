import logging
import unittest
import time

import batchglm.api as glm
import batchglm.data as data_utils
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData_NBGLM
import batchglm.pkg_constants as pkg_constants

glm.setup_logging(verbosity="INFO", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.INFO)


def estimate(input_data: InputData_NBGLM):
    estimator = Estimator(input_data, termination_type="by_feature")
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


class NB_GLM_Test_Hessians(unittest.TestCase):
    sim: Simulator
    estimator_fw: Estimator
    estimator_ow: Estimator
    estimator_tf: Estimator
    estimator: Estimator

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_compute_hessians(self):
        num_observations = 500
        num_conditions = 2

        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=num_conditions, num_batches=2)
        sim.generate()

        sample_description = data_utils.sample_description_from_xarray(sim.data, dim="observations")
        design_loc = data_utils.design_matrix(sample_description, formula="~ 1 + condition + batch")
        design_scale = data_utils.design_matrix(sample_description, formula="~ 1 + condition")

        input_data = InputData_NBGLM.new(sim.X, design_loc=design_loc, design_scale=design_scale)

        pkg_constants.HESSIAN_MODE = "obs_batched"
        self.estimator_ob = estimate(input_data)
        t0_ob = time.time()
        self.H_ob = self.estimator_ob.hessians
        t1_ob = time.time()
        self.estimator_ob.close_session()
        self.t_ob = t1_ob - t0_ob

        pkg_constants.HESSIAN_MODE = "feature"
        self.estimator_fw = estimate(input_data)
        t0_fw = time.time()
        self.H_fw = self.estimator_fw.hessians
        t1_fw = time.time()
        self.estimator_fw.close_session()
        self.t_fw = t1_fw - t0_fw

        pkg_constants.HESSIAN_MODE = "tf"
        self.estimator_tf = estimate(input_data)
        t0_tf = time.time()
        # tensorflow computes the negative hessian as the
        # objective is the negative log-likelihood.
        self.H_tf = self.estimator_tf.hessians
        t1_tf = time.time()
        self.estimator_tf.close_session()
        self.t_tf = t1_tf - t0_tf

        i = 1
        print("\n")
        print("run time observation batch-wise analytic solution: ", str(self.t_ob))
        print("run time feature-wise analytic solution: ", str(self.t_fw))
        print("run time feature-wise tensorflow solution: ", str(self.t_tf))
        print("ratio of tensorflow feature-wise hessian to analytic observation batch-wise hessian:")
        print(self.H_tf.values[i, :, :] / self.H_ob.values[i, :, :])
        print("ratio of tensorflow feature-wise hessian to analytic feature-wise hessian:")
        print(self.H_tf.values[i, :, :] / self.H_fw.values[i, :, :])


if __name__ == '__main__':
    unittest.main()
