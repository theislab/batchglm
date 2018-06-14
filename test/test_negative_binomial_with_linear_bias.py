import os
import unittest
import tempfile

# import numpy as np
from typing import List

from models.negative_binomial_linear_biased import Simulator
from models.negative_binomial_linear_biased.estimator import Estimator


# from utils.config import getConfig


def estimate(sim: Simulator, working_dir: str):
    print(sim.data.sample_data)

    estimator = Estimator(sim.data, batch_size=500)
    estimator.initialize(
        working_dir=working_dir,
        save_checkpoint_steps=200,
        save_summaries_steps=200,
        stop_at_step=1000,
        stop_below_loss_change=1e-5,

        export=["a", "b", "mu", "r", "loss"],
        export_steps=200
    )
    sim.save(os.path.join(working_dir, "sim_data.h5"))

    estimator.train(learning_rate=0.05)

    return estimator


class NegativeBinomialWithLinearBiasTest(unittest.TestCase):
    sim: Simulator
    working_dir: tempfile.TemporaryDirectory

    _estims: List[Estimator]

    def setUp(self):
        self.sim = Simulator(num_samples=2000, num_genes=100)
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

        wd = os.path.join(self.working_dir.name, "default_fit")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim, wd)
        self._estims.append(estimator)

        return estimator, sim

    def test_zero_variance(self):
        sim = self.sim.__copy__()
        sim.data.sample_data[:, 0] = sim.params.mu[0]

        wd = os.path.join(self.working_dir.name, "zero_variance")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim, wd)
        self._estims.append(estimator)

        return estimator, sim

    def test_low_values(self):
        sim = self.sim.__copy__()
        sim.data.sample_data[1:, 0] = 0

        wd = os.path.join(self.working_dir.name, "low_values")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim, wd)
        self._estims.append(estimator)

        return estimator, sim

    def test_nonsense_data(self):
        sim = self.sim.__copy__()
        sim.data.sample_data[:, :] = 0
        sim.data.sample_data[0, 0] = 12

        wd = os.path.join(self.working_dir.name, "nonsense_data")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim, wd)
        self._estims.append(estimator)

        return estimator, sim


if __name__ == '__main__':
    unittest.main()
