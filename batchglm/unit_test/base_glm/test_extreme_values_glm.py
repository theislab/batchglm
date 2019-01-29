import abc
import logging
from typing import List
import unittest
import numpy as np

import batchglm.api as glm
from batchglm.models.base_glm import _Estimator_GLM, InputData, _Simulator_GLM

glm.setup_logging(verbosity="WARNING", stream="STDOUT")
logger = logging.getLogger(__name__)


class _Test_ExtremValues_GLM_Estim():

    def __init__(
            self,
            estimator: _Estimator_GLM
    ):
        self.estimator = estimator

    def estimate(
            self,
        ):
        self.estimator.initialize()
        self.estimator.train_sequence(training_strategy=[
            {
                "convergence_criteria": "all_converged_ll",
                "stopping_criteria": 1e-8,
                "use_batching": False,
                "optim_algo": "NR_TR",
            },
        ])


class Test_ExtremValues_GLM(unittest.TestCase, metaclass=abc.ABCMeta):
    """
    Test various input data types including outlier features.

    These unit tests check whether the model converges and is
    stable in numeric extremes. Each case is tested for the three
    main training graphs: a and b, a only and b only.
    Cases covered are:

        - Zero variance features:
            - Train a and b model: test_zero_variance_a_and_b()
            - Train a model only: test_zero_variance_a_only()
            - Train b model only: test_zero_variance_b_only()
        - Low mean features: test_low_values()
    """
    sim: _Simulator_GLM
    _estims: List[_Test_ExtremValues_GLM_Estim]

    def setUp(self):
        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.estimator.close_session()

    def simulate(self):
        self.simulate1()
        self.simulate2()

    @abc.abstractmethod
    def get_simulator(self):
        pass

    def simulate1(self):
        self.sim1 = self.get_simulator()
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2)
        self.sim1.generate()

    def simulate2(self):
        self.sim2 = self.get_simulator()
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2)
        self.sim2.generate()

    def simulator(self, train_loc):
        if train_loc:
            return self.sim1
        else:
            return self.sim2

    @abc.abstractmethod
    def get_estimator(
            self,
            input_data: InputData,
            termination,
            quick_scale
    ) -> _Estimator_GLM:
        pass

    def _basic_test(
            self,
            input_data: InputData,
            quick_scale
    ):
        """
        Run estimation in all termination modes:

            - "by_feature"
            - "global"

        :param input_data:
        :param quick_scale:
        :return:
        """
        # Run estimation with termination by feature:
        estimator = self.get_estimator(
            input_data=input_data,
            termination="by_feature",
            quick_scale=quick_scale
        )
        self._estims.append(estimator)

        # Run estimation with termination globally:
        estimator = self.get_estimator(
            input_data=input_data,
            termination="global",
            quick_scale=quick_scale
        )
        self._estims.append(estimator)

        return True

    def _test_low_values_a_and_b(self):
        sim = self.sim1.__copy__()
        sim.data.X[:, 0] = 0
        return self._basic_test(
            input_data=sim.input_data,
            quick_scale=False
        )

    def _test_low_values_a_only(self):
        sim = self.sim1.__copy__()
        sim.data.X[:, 0] = 0
        return self._basic_test(
            input_data=sim.input_data,
            quick_scale=True
        )

    def _test_low_values_b_only(self):
        sim = self.sim2.__copy__()
        sim.data.X[:, 0] = 0
        return self._basic_test(
            input_data=sim.input_data,
            quick_scale=False
        )

    def _test_zero_variance_a_and_b(self):
        sim = self.sim1.__copy__()
        sim.data.X[:, 0] = np.exp(sim.a)[0, 0]
        return self._basic_test(
            input_data=sim.input_data,
            quick_scale=False
        )

    def _test_zero_variance_a_only(self):
        sim = self.sim1.__copy__()
        sim.data.X[:, 0] = np.exp(sim.a)[0, 0]
        return self._basic_test(
            input_data=sim.input_data,
            quick_scale=True
        )

    def _test_zero_variance_b_only(self):
        sim = self.sim2.__copy__()
        sim.data.X[:, 0] = np.exp(sim.a)[0, 0]
        return self._basic_test(
            input_data=sim.input_data,
            quick_scale=False
        )


if __name__ == '__main__':
    unittest.main()
