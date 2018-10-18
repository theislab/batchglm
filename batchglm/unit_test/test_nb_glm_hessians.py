from typing import List

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
    estimator = Estimator(input_data, batch_size=None)
    estimator.initialize()
    estimator.train_sequence(training_strategy="QUICK")
    return estimator


class NB_GLM_hessian_Test(unittest.TestCase):
    sim: Simulator
    estimator_fw: Estimator
    estimator_ow: Estimator
    estimator_tf: Estimator
    H_fw
    H_ow
    H_tf
    t_fw
    t_ow
    t_tf
    
    def tearDown(self):
        self.estimator.close_session()


    def compute_hessians(self, num_observations=500, num_conditions=2):
        sim = Simulator(num_observations=num_observations, num_features=100)
        sim.generate_sample_description(num_conditions=num_conditions, num_batches=0)
        sim.generate()

        sample_description = data_utils.sample_description_from_xarray(sim.data, dim="observations")
        design_loc = data_utils.design_matrix(sample_description, formula="~ 1 + condition")
        design_scale = data_utils.design_matrix(sample_description, formula="~ 1 + condition")

        input_data = InputData.new(sim.X, design_loc=design_loc, design_scale=design_scale)
        
        pkg_constants.HESSIAN_MODE = "feature"
        self.estimator_fw = estimate(input_data)
        t0_fw = time.time()
        self.H_fw = self.estimator_fw.hessians
        t1_fw = time.time()
        self.estimator_fw.close_session()
        self.t_fw = t1_fw - t0_fw

        pkg_constants.HESSIAN_MODE = "obs"
        self.estimator_ow = estimate(input_data)
        t0_ow = time.time()
        self.H_ow = self.estimator_ow.hessians
        t1_ow = time.time()
        self.estimator_ow.close_session()
        self.t_ow = t1_ow - t0_ow

        pkg_constants.HESSIAN_MODE = "tf"
        self.estimator_tf = estimate(input_data)
        t0_tf = time.time()
        self.H_tf = self.estimator_tf.hessians
        t1_tf = time.time()
        self.estimator_tf.close_session()
        self.t_tf = t1_tf - t0_tf

    def compare_hessians(self, i):
        # test finalizing
        print("analytic feature-wise hessian in "+str(self.t_fw))
        print(H_fw[i,:,:])
        print("analytic observation-wise hessian in "+str(self.t_ow))
        print(H_ow[i,:,:])
        print("tensorflow feature-wise hessian in "+str(self.t_tf))
        print(H_tf[i,:,:])


if __name__ == '__main__':
    unittest.main()
