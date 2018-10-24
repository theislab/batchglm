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


def estimate_adam_full(input_data: InputData, working_dir: str):
    estimator = Estimator(input_data, batch_size=500)
    estimator.initialize(
        working_dir=working_dir,
        save_checkpoint_steps=20,
        save_summaries_steps=20,
        export=["a", "b", "mu", "r", "loss"],
        export_steps=20
    )
    input_data.save(os.path.join(working_dir, "input_data.h5"))

    estimator.train_sequence(training_strategy=[
        {'convergence_criteria': 't_test',
         'learning_rate': 0.1,
         'loss_window_size': 20,
         'optim_algo': 'ADAM',
         'stop_at_loss_change': 1e-8,
         'use_batching': False}
    ])

    return estimator


def estimate_nr_full(input_data: InputData, working_dir: str):
    estimator = Estimator(input_data, batch_size=500)
    estimator.initialize(
        working_dir=working_dir,
        save_checkpoint_steps=20,
        save_summaries_steps=20,
        export=["a", "b", "mu", "r", "loss"],
        export_steps=20
    )
    input_data.save(os.path.join(working_dir, "input_data.h5"))

    estimator.train_sequence(training_strategy="NEWTON_EXACT")

    return estimator

def estimate_nr_batched(input_data: InputData, working_dir: str):
    estimator = Estimator(input_data, batch_size=50)
    estimator.initialize(
        working_dir=working_dir,
        save_checkpoint_steps=20,
        save_summaries_steps=20,
        export=["a", "b", "mu", "r", "loss"],
        export_steps=20
    )
    input_data.save(os.path.join(working_dir, "input_data.h5"))

    estimator.train_sequence(training_strategy="NEWTON_BATCHED")

    return estimator

def estimate_nr_series(input_data: InputData, working_dir: str):
    estimator = Estimator(input_data, batch_size=500)
    estimator.initialize(
        working_dir=working_dir,
        save_checkpoint_steps=20,
        save_summaries_steps=20,
        export=["a", "b", "mu", "r", "loss"],
        export_steps=20
    )
    input_data.save(os.path.join(working_dir, "input_data.h5"))

    estimator.train_sequence(training_strategy="NEWTON_SERIES")

    return estimator


class NB_GLM__Newton_Test(unittest.TestCase):
    sim: Simulator
    working_dir: tempfile.TemporaryDirectory

    _estims: List[Estimator]

    def setUp(self):
        self.sim = Simulator(num_observations=1000, num_features=2)
        self.sim.generate()
        self.working_dir = tempfile.TemporaryDirectory()

        self._estims = []
        print("working_dir: %s" % self.working_dir)

    def tearDown(self):
        for e in self._estims:
            e.close_session()

        self.working_dir.cleanup()


    def test_adam_full(self):
        X = scipy.sparse.csr_matrix(self.sim.X)
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData.new(
            data=X,
            design_loc=design_loc,
            design_scale=design_scale,
        )
        wd = os.path.join(self.working_dir.name, "newton_full")
        os.makedirs(wd, exist_ok=True)

        t0 = time.time()
        estimator = estimate_adam_full(idata, wd)
        t1 = time.time()
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()
        print("\n")
        print("run time adam on full data: ", str(t1-t0))
        print((estimator.a.values-self.sim.a.values)/self.sim.a.values)
        print((estimator.b.values - self.sim.b.values) / self.sim.b.values)


    def test_newton_batched(self):
        X = scipy.sparse.csr_matrix(self.sim.X)
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData.new(
            data=X,
            design_loc=design_loc,
            design_scale=design_scale,
        )
        wd = os.path.join(self.working_dir.name, "newton_batched")
        os.makedirs(wd, exist_ok=True)

        t0 = time.time()
        pkg_constants.JACOBIAN_MODE = "analytic"
        estimator = estimate_nr_batched(idata, wd)
        t1 = time.time()
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()
        print("\n")
        print("run time newton-rhapson on batched data: ", str(t1 - t0))
        print((estimator.a.values - self.sim.a.values) / self.sim.a.values)
        print((estimator.b.values - self.sim.b.values) / self.sim.b.values)


    def test_newton_full(self):
        X = scipy.sparse.csr_matrix(self.sim.X)
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData.new(
            data=X,
            design_loc=design_loc,
            design_scale=design_scale,
        )
        wd = os.path.join(self.working_dir.name, "newton_full")
        os.makedirs(wd, exist_ok=True)

        t0 = time.time()
        pkg_constants.JACOBIAN_MODE = "analytic"
        estimator = estimate_nr_full(idata, wd)
        t1 = time.time()
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()
        print("\n")
        print("run time newton-rhapson on full data: ", str(t1-t0))
        print((estimator.a.values-self.sim.a.values)/self.sim.a.values)
        print((estimator.b.values - self.sim.b.values) / self.sim.b.values)

    def test_newton_series(self):
        X = scipy.sparse.csr_matrix(self.sim.X)
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData.new(
            data=X,
            design_loc=design_loc,
            design_scale=design_scale,
        )
        wd = os.path.join(self.working_dir.name, "newton_full")
        os.makedirs(wd, exist_ok=True)

        t0 = time.time()
        pkg_constants.JACOBIAN_MODE = "analytic"
        estimator = estimate_nr_series(idata, wd)
        t1 = time.time()
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()
        print("\n")
        print("run time newton-rhapson series: ", str(t1-t0))
        print((estimator.a.values-self.sim.a.values)/self.sim.a.values)
        print((estimator.b.values - self.sim.b.values) / self.sim.b.values)


if __name__ == '__main__':
    unittest.main()
