import logging
import unittest
import numpy as np

import batchglm.api as glm
from batchglm.models.base_glm import _SimulatorGLM, InputDataGLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class TestSimulationGlmAll:

    sim: _SimulatorGLM
    input_data: InputDataGLM
    noise_model: str

    def eval_simulation_mean(
            self
    ):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                threshold_dev = 1e-2
                threshold_std = 1e-1
            elif self.noise_model == "norm":
                threshold_dev = 1e-2
                threshold_std = 1e-1
            elif self.noise_model == "beta":
                threshold_dev = 1e-2
                threshold_std = 1e-1
            else:
                raise ValueError("noise_model not recognized")

        means_sim = self.sim.a_var[0, :]
        means_obs = self.sim.link_loc(np.mean(self.sim.input_data.x, axis=0))
        mean_dev = np.mean(means_sim - means_obs)
        std_dev = np.std(means_sim - means_obs)

        logging.getLogger("batchglm").info("mean_dev_a %f" % mean_dev)
        logging.getLogger("batchglm").info("std_dev_a %f" % std_dev)

        if np.abs(mean_dev) < threshold_dev and \
                std_dev < threshold_std:
            return True
        else:
            return False

    def _test_all_moments(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model == "nb":
                from batchglm.api.models.tf1.glm_nb import Simulator
            elif self.noise_model == "norm":
                from batchglm.api.models import Simulator
            elif self.noise_model == "beta":
                from batchglm.api.models.tf1.glm_beta import Simulator
            else:
                raise ValueError("noise_model not recognized")

        self.sim = Simulator(
            num_observations=100000,
            num_features=10
        )
        self.sim.generate_sample_description(num_batches=1, num_conditions=1)
        self.sim.generate_params()
        self.sim.generate_data()

        success = self.eval_simulation_mean()
        assert success, "mean of simulation was inaccurate"
        return True


class TestSimulationGlmNb(
    TestSimulationGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for negative binomial data.
    """

    def test(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestSimulationGlmNb.test()")

        self.noise_model = "nb"
        self._test_all_moments()


class TestSimulationGlmNorm(
    TestSimulationGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for normally distributed data.
    """

    def test(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestSimulationGlmNorm.test()")

        self.noise_model = "norm"
        self._test_all_moments()


class TestSimulationGlmBeta(
    TestSimulationGlmAll,
    unittest.TestCase
):
    """
    Test whether optimizers yield exact results for beta distributed data.
    """

    def test(self):
        logging.getLogger("tensorflow").setLevel(logging.ERROR)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("TestSimulationGlmBeta.test()")

        self.noise_model = "beta"
        self._test_all_moments()


if __name__ == '__main__':
    unittest.main()
