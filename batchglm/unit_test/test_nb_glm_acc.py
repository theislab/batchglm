from typing import List

import unittest
import logging

import numpy as np

import batchglm.api as glm
from batchglm.api.models.nb_glm import Simulator, Estimator, InputData_NBGLM

glm.setup_logging(verbosity="ERROR", stream="STDOUT")
logging.getLogger("tensorflow").setLevel(logging.ERROR)


def estimate(
        input_data: InputData_NBGLM,
        algo,
        batched,
        quick_scale,
        termination,
        acc,
):
    provide_optimizers = {"gd": True, "adam": True, "adagrad": True, "rmsprop": True, "nr": True}

    estimator = Estimator(
        input_data,
        batch_size=900,
        quick_scale=quick_scale,
        provide_optimizers=provide_optimizers,
        termination_type=termination
    )
    estimator.initialize()

    # Choose learning rate based on optimizer
    if algo.lower() is "nr":
        lr = 1
    elif algo.lower() is "gd":
        lr = 0.05
    else:
        lr = 0.5
    estimator.train_sequence(training_strategy=[
            {
                "learning_rate": lr,
                "convergence_criteria": "all_converged_ll" if termination == "by_feature" else "scaled_moving_average",
                "stopping_criteria": acc,
                "use_batching": batched,
                "optim_algo": algo,
            },
        ])

    return estimator

def eval_estimation(
        estimator,
        sim,
        batched
):
    if batched:
        threshold_dev_a = 0.5
        threshold_dev_b = 0.5
        threshold_std_a = 1
        threshold_std_b = 20
    else:
        threshold_dev_a = 0.2
        threshold_dev_b = 0.2
        threshold_std_a = 1
        threshold_std_b = 2

    mean_dev_a = np.mean(estimator.a.values - sim.a.values)
    std_dev_a = np.std(estimator.a.values - sim.a.values)
    mean_dev_b = np.mean(estimator.b.values - sim.b.values)
    std_dev_b = np.std(estimator.b.values - sim.b.values)

    print("\n")
    print("mean_dev_a %f" % mean_dev_a)
    print("std_dev_a %f" % std_dev_a)
    print("mean_dev_b %f" % mean_dev_b)
    print("std_dev_b %f" % std_dev_b)

    if np.abs(mean_dev_a) < threshold_dev_a and \
            np.abs(mean_dev_b) < threshold_dev_b and \
            std_dev_a < threshold_std_a and \
            std_dev_b < threshold_std_b:
        return True
    else:
        return False

class NB_GLM_Test_Accuracy(unittest.TestCase):
    """
    Test whether optimizers yield exact results.

    Accuracy is evaluted via deviation of simulated ground truth.
    The unit tests test individual training graphs and multiple optimizers
    (incl. one tensorflow internal optimizer and newton-rhapson)
    for each training graph. The training graphs tested are as follows:

    - termination by feature
        - full data model
            - train a and b model: test_full_byfeature_a_and_b()
            - train a model only: test_full_byfeature_a_only()
            - train b model only: test_full_byfeature_b_only()
        - batched data model
            - train a and b model: test_batched_byfeature_a_and_b()
            - train a model only: test_batched_byfeature_a_only()
            - train b model only: test_batched_byfeature_b_only()
    - termination global
        - full data model
            - train a and b model: test_full_global_a_and_b()
            - train a model only: test_full_global_a_only()
            - train b model only: test_full_global_b_only()
        - batched data model
            - train a and b model: test_batched_global_a_and_b()
            - train a model only: test_batched_global_a_only()
            - train b model only: test_batched_global_b_only()

    The unit tests throw an assertion error if the required accurcy is
    not met. Accuracy thresholds are fairly lenient so that unit_tests
    pass even with noise inherent in fast optimisation and random
    initialisation in simulation. Still, large biases (i.e. graph errors)
    should be discovered here.
    """
    sim: Simulator
    _estims: List[Estimator]

    def setUp(self):
        self.sim1 = Simulator(num_observations=1000, num_features=50)
        self.sim1.generate_sample_description(num_batches=2, num_conditions=2)
        self.sim1.generate()

        self.sim2 = Simulator(num_observations=1000, num_features=50)
        self.sim2.generate_sample_description(num_batches=0, num_conditions=2)
        self.sim2.generate()

        self._estims = []

    def tearDown(self):
        for e in self._estims:
            e.close_session()

    def test_full_byfeature_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="by_feature",
                acc=1e-6 if algo == "NR" else 1e-3
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=False
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_full_byfeature_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=True,
                termination="by_feature",
                acc=1e-6 if algo == "NR" else 1e-3
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=False
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_full_byfeature_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="by_feature",
                acc=1e-6 if algo == "NR" else 1e-3
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=False
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_batched_byfeature_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="by_feature",
                acc=1e-2
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=True
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_batched_byfeature_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=True,
                termination="by_feature",
                acc=1e-2
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=True
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_batched_byfeature_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="by_feature",
                acc=1e-2
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=True
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_full_global_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="global",
                acc=1e-3
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=False
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_full_global_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=True,
                termination="global",
                acc=1e-3
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=False
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_full_global_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=False,
                quick_scale=False,
                termination="global",
                acc=1e-3
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=False
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_batched_global_a_and_b(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="global",
                acc=1e-2
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=True
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_batched_global_a_only(self):
        sim = self.sim1.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=True,
                termination="global",
                acc=1e-2
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=True
            )
            assert success, "%s did not yield exact results" % algo

        return True

    def test_batched_global_b_only(self):
        sim = self.sim2.__copy__()

        for algo in ["ADAM", "ADAGRAD", "NR"]:
            print("algorithm: %s" % algo)
            estimator = estimate(
                sim.input_data,
                algo=algo,
                batched=True,
                quick_scale=False,
                termination="global",
                acc=1e-2
            )
            estimator_store = estimator.finalize()
            self._estims.append(estimator)
            success = eval_estimation(
                estimator=estimator_store,
                sim=sim,
                batched=True
            )
            assert success, "%s did not yield exact results" % algo

        return True

if __name__ == '__main__':
    unittest.main()
