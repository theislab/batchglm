from typing import List

import unittest
import logging

import numpy as np
import scipy.sparse

import batchglm.api as glm
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData_NBGLM

glm.setup_logging(verbosity="INFO", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.INFO)


def estimate(input_data: InputData_NBGLM):
    estimator = Estimator(
        input_data,
        batch_size=500,
        termination_type="by_feature"
    )
    estimator.initialize()

    estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": 1e-1,
                "use_batching": False,
                "optim_algo": "Newton",
            },
        ])

    return estimator


class NB_GLM_Test_DataTypes(unittest.TestCase):
    """
    Test various input data types including outlier features.

    These unit tests cover a range of input data and check whether
    the overall graph works with different inputs. Only one
    training strategy is tested here. The cases tested are:

        - Dense X matrix: test_default_fit()
        - Sparse X matrix: test_sparse_fit()
        - Dense X in anndata: test_anndata()
        - Sparse X in anndata: test_anndata_sparse()
    """
    sim: Simulator
    _estims: List[Estimator]

    def setUp(self):
        self.sim = Simulator(num_observations=1000, num_features=2)
        self.sim.generate_sample_description(num_batches=0, num_conditions=2)
        self.sim.generate()
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.close_session()

    def test_default_fit(self):
        sim = self.sim.__copy__()
        estimator = estimate(sim.input_data)
        self._estims.append(estimator)
        estimator = estimator.finalize()

        return estimator, sim

    def test_sparse_fit(self):
        X = scipy.sparse.csr_matrix(self.sim.X)
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData_NBGLM.new(
            data=X,
            design_loc=design_loc,
            design_scale=design_scale,
        )
        estimator = estimate(idata)
        self._estims.append(estimator)

        return estimator, idata

    def test_anndata(self):
        adata = self.sim.data_to_anndata()
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData_NBGLM.new(
            data=adata,
            design_loc=design_loc,
            design_scale=design_scale,
        )
        estimator = estimate(idata)
        self._estims.append(estimator)
        return estimator, adata

    def test_anndata_sparse(self):
        adata = self.sim.data_to_anndata()
        adata.X = scipy.sparse.csr_matrix(adata.X)
        design_loc = self.sim.design_loc
        design_scale = self.sim.design_scale
        idata = InputData_NBGLM.new(
            data=adata,
            design_loc=design_loc,
            design_scale=design_scale,
        )
        estimator = estimate(idata)
        self._estims.append(estimator)

        return estimator, adata


if __name__ == '__main__':
    unittest.main()
