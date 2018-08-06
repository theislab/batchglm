import os
import unittest
import tempfile

# import numpy as np

from batchglm.api.models.nb import Simulator, Estimator


# from utils.config import getConfig


def estimate(sim: Simulator, working_dir: str):
    print(sim.data.X)

    estimator = Estimator(sim.input_data)
    estimator.initialize(
        working_dir=working_dir,
        # stop_at_step=1000,
        # stop_below_loss_change=0.001,
        export=["mu", "r", "loss"],
        export_steps=200
    )
    sim.save(os.path.join(working_dir, "sim_data.h5"))

    estimator.train_sequence()

    return estimator


class NegativeBinomialTest(unittest.TestCase):
    sim: Simulator
    working_dir: tempfile.TemporaryDirectory

    def setUp(self):
        self.sim = Simulator(num_observations=2000, num_features=100)
        self.sim.generate()
        self.working_dir = tempfile.TemporaryDirectory()
        print("working_dir: %s" % self.working_dir)

    def tearDown(self):
        self.working_dir.cleanup()

    def test_default_fit(self):
        sim = self.sim.__copy__()
        print(sim.input_data[2:4, [5, 6, 7]])

        wd = os.path.join(self.working_dir.name, "default_fit")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim, wd)

        # test finalizing
        estimator = estimator.finalize()
        print(estimator.mu.values)
        print(estimator.gradient.values)
        print(estimator.hessians.values)
        print(estimator.probs().values)
        print(estimator.log_probs().values)

        return estimator, sim

    def test_zero_variance(self):
        sim = self.sim.__copy__()
        sim.data.X[:, 0] = sim.params.mu[0]

        wd = os.path.join(self.working_dir.name, "zero_variance")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim, wd)

        return estimator, sim

    def test_low_values(self):
        sim = self.sim.__copy__()
        sim.data.X[1:, 0] = 0

        wd = os.path.join(self.working_dir.name, "low_values")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim, wd)

        return estimator, sim

    def test_nonsense_data(self):
        sim = self.sim.__copy__()
        sim.data.X[:, :] = 0
        sim.data.X[0, 0] = 12

        wd = os.path.join(self.working_dir.name, "nonsense_data")
        os.makedirs(wd, exist_ok=True)

        estimator = estimate(sim, wd)

        return estimator, sim


if __name__ == '__main__':
    unittest.main()
