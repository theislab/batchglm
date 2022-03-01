from typing import List, Optional, Union

import numpy as np

from batchglm.models.glm_beta import Model as BetaModel
from batchglm.models.glm_nb import Model as NBModel
from batchglm.models.glm_norm import Model as NormModel
from batchglm.train.numpy.glm_beta import Estimator as BetaEstimator
from batchglm.train.numpy.glm_nb import Estimator as NBEstimator
from batchglm.train.numpy.glm_norm import Estimator as NormEstimator


def get_estimator(noise_model: Optional[str] = None, **kwargs):
    if noise_model is None:
        raise ValueError("noise_model is None")
    else:
        if noise_model == "nb":
            estimator = NBEstimator(**kwargs)
        elif noise_model == "norm":
            estimator = NormEstimator(**kwargs)
        elif noise_model == "beta":
            estimator = BetaEstimator(**kwargs)
        else:
            raise ValueError("noise_model not recognized")
    return estimator


def get_generated_model(
    num_conditions: int, num_batches: int, sparse: bool, mode: Optional[str] = None, noise_model: Optional[str] = None
):
    if noise_model is None:
        raise ValueError("noise_model is None")
    else:
        if noise_model == "nb":
            model = NBModel()
        elif noise_model == "norm":
            model = NormModel()
        elif noise_model == "beta":
            model = BetaModel()
        else:
            raise ValueError("noise_model not recognized")

    def random_uniform(low: float, high: float):
        return lambda: lambda shape: np.random.uniform(low=low, high=high, size=shape)

    def const(offset: float):
        return lambda offset: lambda shape: np.zeros(shape) + offset

    if mode is None:
        """Sample loc and scale with default functions"""
        rand_fn_ave = None
        rand_fn_loc = None
        rand_fn_scale = None

    elif mode == "randTheta":

        if noise_model in ["nb", "norm"]:
            rand_fn_ave = random_uniform(10, 1000)
            rand_fn_loc = random_uniform(1, 3)
            rand_fn_scale = random_uniform(1, 3)
        elif noise_model == "beta":
            rand_fn_ave = random_uniform(0.1, 0.7)
            rand_fn_loc = random_uniform(0.0, 0.15)
            rand_fn_scale = random_uniform(0.0, 0.15)
        else:
            raise ValueError(f"Noise model {noise_model} not recognized.")

    elif mode == "constTheta":

        if noise_model in ["nb", "norm"]:
            rand_fn_ave = random_uniform(10, 1000)
            rand_fn_loc = const(1.0)
            rand_fn_scale = const(1.0)
        elif noise_model == "beta":
            rand_fn_ave = random_uniform(0.1, 0.9)
            rand_fn_loc = const(0.05)
            rand_fn_scale = const(0.2)
        else:
            raise ValueError(f"Noise model {noise_model} not recognized.")

    else:
        raise ValueError(f"Mode {mode} not recognized.")

    model.generate(
        n_obs=2000,
        n_vars=100,
        num_conditions=num_conditions,
        num_batches=num_batches,
        intercept_scale=True,
        sparse=sparse,
        rand_fn_ave=rand_fn_ave,
        rand_fn_loc=rand_fn_loc,
        rand_fn_scale=rand_fn_scale,
    )
    return model
