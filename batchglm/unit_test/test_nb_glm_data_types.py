from typing import List

import os
# import sys
import unittest
import tempfile
import logging

import numpy as np
import scipy.sparse

import batchglm.api as glm
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData

glm.setup_logging(verbosity="INFO", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.INFO)


def estimate(input_data: InputData, working_dir: str):
    print(input_data.X)

    estimator = Estimator(
        input_data,
        batch_size=500,
        init_a="standard",
        init_b="standard",
        termination_type="global"
    )
    estimator.initialize(
        working_dir=working_dir,
        save_checkpoint_steps=20,
        save_summaries_steps=20,
        export=["a", "b", "mu", "r", "loss"],
        export_steps=20
    )
    input_data.save(os.path.join(working_dir, "input_data.h5"))

    estimator.train_sequence(training_strategy="DEFAULT")

    return estimator


class NB_GLM_Test(unittest.TestCase):
    sim: Simulator
    working_dir: tempfile.TemporaryDirectory

    _estims: List[Estimator]

    def setUp(self):
        self.sim = Simulator(num_observations=2000, num_features=100)
        self.sim.generate()
        self.working_dir = tempfile.TemporaryDirectory()

        self._estims = []
        print("working_dir: %s" % self.working_dir)

    def tearDown(self):
        for e in self._estims:
            e.close_session()

        self.working_dir.cleanup()

    def test_default_fit(self):
        sim = self.sim.__copy__()
        print(sim.input_data[2:4, [5, 6, 7]])

        wd = os.path.join(self.working_dir.name, "default_fit")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim.input_data, wd)
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()

        return estimator, sim

    def test_sparse_fit(self):
        X = scipy.sparse.csr_matrix(self.sim.X)
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData.new(
            data=X,
            design_loc=design_loc,
            design_scale=design_scale,
        )

        wd = os.path.join(self.working_dir.name, "anndata")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(idata, wd)
        self._estims.append(estimator)

        return estimator, idata

    def test_anndata(self):
        adata = self.sim.data_to_anndata()
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData.new(
            data=adata,
            design_loc=design_loc,
            design_scale=design_scale,
        )

        wd = os.path.join(self.working_dir.name, "anndata")
        os.makedirs(wd, exist_ok=True)

        print(adata)
        estimator = estimate(idata, wd)
        self._estims.append(estimator)

        return estimator, adata

    def test_anndata_sparse(self):
        adata = self.sim.data_to_anndata()
        adata.X = scipy.sparse.csr_matrix(adata.X)
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData.new(
            data=adata,
            design_loc=design_loc,
            design_scale=design_scale,
        )

        wd = os.path.join(self.working_dir.name, "anndata")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(idata, wd)
        self._estims.append(estimator)

        return estimator, adata

    def test_zero_variance(self):
        sim = self.sim.__copy__()
        sim.data.X[:, 0] = np.exp(sim.a)[0, 0]

        wd = os.path.join(self.working_dir.name, "zero_variance")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim.input_data, wd)
        self._estims.append(estimator)

        return estimator, sim

    def test_low_values(self):
        sim = self.sim.__copy__()
        sim.data.X[:, 0] = 0

        wd = os.path.join(self.working_dir.name, "low_values")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim.input_data, wd)
        self._estims.append(estimator)

        return estimator, sim


if __name__ == '__main__':
    unittest.main()
