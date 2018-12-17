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


def estimate(input_data: InputData):

    estimator = Estimator(input_data, batch_size=500)
    estimator.initialize()

    estimator.train(
        convergence_criteria="all_converged",
        use_batching=False
    )

    return estimator


class NB_GLM_Test(unittest.TestCase):
    sim: Simulator

    _estims: List[Estimator]

    def setUp(self):
        self.sim = Simulator(num_observations=1000, num_features=20)
        self.sim.generate()
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.close_session()

    def test_default_fit(self):
        sim = self.sim.__copy__()

        estimator = estimate(sim.input_data)
        self._estims.append(estimator)

        # test finalizing
        estimator = estimator.finalize()
        return estimator, sim


if __name__ == '__main__':
    unittest.main()
