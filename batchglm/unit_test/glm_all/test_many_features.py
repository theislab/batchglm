import unittest
import logging

import batchglm.api as glm
from batchglm.models.base_glm import _Simulator_GLM


glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class Test_ManyFeatures_GLM_ALL(unittest.TestCase):
    """
    Test estimation with many genes.

    Use this to check memory laod.
    """
    noise_model: str
    sim: _Simulator_GLM

    def simulate(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.glm_nb import Simulator, InputData
            else:
                raise ValueError("noise_model not recognized")

        sim = Simulator(num_observations=50, num_features=1000)
        sim.generate_sample_description(num_batches=2, num_conditions=2)
        sim.generate()

        self.input = InputData.new(
            data=sim.X,
            design_loc=sim.design_loc,
            design_scale=sim.design_scale,
        )


    def estimate(self):
        if self.noise_model is None:
            raise ValueError("noise_model is None")
        else:
            if self.noise_model=="nb":
                from batchglm.api.models.glm_nb import Estimator
            else:
                raise ValueError("noise_model not recognized")

        batch_size = 1
        provide_optimizers = {"gd": False, "adam": False, "adagrad": False, "rmsprop": False,
                              "nr": True, "nr_tr": True, "irls": True, "irls_tr": True}

        estimator = Estimator(
            input_data=self.input,
            batch_size=batch_size,
            quick_scale=True,
            provide_optimizers=provide_optimizers,
            provide_batched=False,
            termination_type="global"
        )
        estimator.initialize()

        estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": 1e-6,
                "use_batching": False,
                "optim_algo": "nr_tr",
            },
        ])
        estimator_store = estimator.finalize()
        return estimator_store

    def test_many_features(self):
        logging.getLogger("tensorflow").setLevel(logging.INFO)
        logging.getLogger("batchglm").setLevel(logging.INFO)
        logger.error("Test_ManyFeatures_GLM_ALL.test_many_features()")

        self.noise_model = "nb"
        self.simulate()
        self.estimate()

        return True

if __name__ == '__main__':
    unittest.main()
