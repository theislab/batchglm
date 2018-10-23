from typing import List

import logging
import os
import unittest
import tempfile
import time

import numpy as np
import scipy.sparse

import batchglm.data as data_utils
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData
import batchglm.pkg_constants as pkg_constants


# from utils.config import getConfig


def estimate(input_data: InputData):
    estimator = Estimator(input_data)
    estimator.initialize()
    # Do not train, evalute at initialization!
    return estimator


class NB_GLM_jac_Test(unittest.TestCase):
    sim: Simulator
    estimator_analytic: Estimator
    estimator_tf: Estimator

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_compute_hessians(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)

        num_observations = 500
        num_conditions = 2

        sim = Simulator(num_observations=num_observations, num_features=4)
        sim.generate_sample_description(num_conditions=num_conditions, num_batches=2)
        sim.generate()

        sample_description = data_utils.sample_description_from_xarray(sim.data, dim="observations")
        design_loc = data_utils.design_matrix(sample_description, formula="~ 1 + condition + batch")
        design_scale = data_utils.design_matrix(sample_description, formula="~ 1 + condition")

        input_data = InputData.new(sim.X, design_loc=design_loc, design_scale=design_scale)

        pkg_constants.JACOBIAN_MODE = "analytic"
        self.estimator_analytic = estimate(input_data)
        t0_analytic = time.time()
        self.J_analytic = self.estimator_analytic['full_gradient']
        self.a_analytic = self.estimator_analytic.a.values
        self.b_analytic = self.estimator_analytic.b.values
        t1_analytic = time.time()
        self.estimator_analytic.close_session()
        self.t_analytic = t1_analytic - t0_analytic

        pkg_constants.JACOBIAN_MODE = "tf"
        self.estimator_tf = estimate(input_data)
        t0_tf = time.time()
        self.J_tf = self.estimator_tf['full_gradient']
        self.a_tf = self.estimator_tf.a.values
        self.b_tf = self.estimator_tf.b.values
        t1_tf = time.time()
        self.estimator_tf.close_session()
        self.t_tf = t1_tf - t0_tf

        i = 1
        print("\n")
        print("run time tensorflow solution: ", str(self.t_tf))
        print("run time observation batch-wise analytic solution: ", str(self.t_analytic))
        print("relative difference of mean estimates for analytic jacobian to observation-wise jacobian:")
        print((self.a_analytic - self.a_tf) / self.a_tf)
        print("relative difference of dispersion estimates for analytic jacobian to observation-wise jacobian:")
        print((self.b_analytic - self.b_tf) / self.b_tf)
        print("relative difference of analytic jacobian to analytic observation-wise jacobian:")
        print((self.J_tf - self.J_analytic)/self.J_tf)


if __name__ == '__main__':
    unittest.main()
