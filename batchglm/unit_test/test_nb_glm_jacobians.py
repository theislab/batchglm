import logging
import unittest
import time

import numpy as np

import batchglm.api as glm
import batchglm.data as data_utils
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData
import batchglm.pkg_constants as pkg_constants

glm.setup_logging(verbosity="DEBUG", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.INFO)


def estimate(input_data: InputData, quick_scale):
    estimator = Estimator(input_data, quick_scale=quick_scale)
    estimator.initialize()
    # Do not train, evalute at initialization!
    return estimator

def compare_jacs(sim, design, quick_scale):
    sample_description = data_utils.sample_description_from_xarray(sim.data, dim="observations")
    design_loc = data_utils.design_matrix(sample_description, formula=design)
    design_scale = data_utils.design_matrix(sample_description, formula=design)

    input_data = InputData.new(sim.X, design_loc=design_loc, design_scale=design_scale)

    pkg_constants.JACOBIAN_MODE = "analytic"
    estimator_analytic = estimate(input_data, quick_scale)
    t0_analytic = time.time()
    J_analytic = estimator_analytic['full_gradient']
    a_analytic = estimator_analytic.a.values
    b_analytic = estimator_analytic.b.values
    t1_analytic = time.time()
    estimator_analytic.close_session()
    t_analytic = t1_analytic - t0_analytic

    pkg_constants.JACOBIAN_MODE = "tf"
    estimator_tf = estimate(input_data, quick_scale)
    t0_tf = time.time()
    J_tf = estimator_tf['full_gradient']
    a_tf = estimator_tf.a.values
    b_tf = estimator_tf.b.values
    t1_tf = time.time()
    estimator_tf.close_session()
    t_tf = t1_tf - t0_tf

    i = 1
    print("\n")
    print("run time tensorflow solution: ", str(t_tf))
    print("run time observation batch-wise analytic solution: ", str(t_analytic))
    print("relative difference of mean estimates for analytic jacobian to observation-wise jacobian:")
    print((a_analytic - a_tf) / a_tf)
    print("relative difference of dispersion estimates for analytic jacobian to observation-wise jacobian:")
    print((b_analytic - b_tf) / b_tf)
    print("relative difference of analytic jacobian to analytic observation-wise jacobian:")
    print((J_tf - J_analytic) / J_tf)

    max_rel_dev = np.max(np.abs((J_tf - J_analytic) / J_tf))
    assert max_rel_dev < 1e-10
    return True

class NB_GLM_jac_Test(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_compute_jacobians_a_and_b(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)

        num_observations = 500
        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=2, num_batches=2)
        sim.generate()

        return compare_jacs(sim, design="~ 1 + condition + batch", quick_scale=False)

    def test_compute_jacobians_a_only(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)

        num_observations = 500
        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=2, num_batches=2)
        sim.generate()

        return compare_jacs(sim, design="~ 1 + condition + batch", quick_scale=True)

    def test_compute_jacobians_b_only(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)

        num_observations = 500
        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=2, num_batches=0)
        sim.generate()

        return compare_jacs(sim, design="~ 1 + condition", quick_scale=False)

if __name__ == '__main__':
    unittest.main()
