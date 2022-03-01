import numpy as np
from typing import Union, List, Optional


def getEstimator(noise_model: Optional[str] = None, **kwargs):
    if noise_model is None:
        raise ValueError("noise_model is None")
    else:
        if noise_model == "nb":
            estim_module = batchglm.train.numpy.glm_nb
        elif noise_model == "norm":
            estim_module = batchglm.train.numpy.glm_norm
        elif noise_model == "beta":
            estim_module = batchglm.train.numpy.glm_beta
        else:
            raise ValueError("noise_model not recognized")
    from estim_module import Estimator
    return Estimator(**kwargs)
  

def getGeneratedModel(num_conditions: int, num_batches: int, sparse: bool, mode: Optional[str] = None, noise_model: Optional[str] = None):
    if noise_model is None:
        raise ValueError("noise_model is None")
    else:
        if noise_model == "nb":
            model_module = batchglm.models.glm_nb
        elif noise_model == "norm":
            model_module = batchglm.models.glm_norm
        elif noise_model == "beta":
            model_module = batchglm.models.glm_beta
        else:
            raise ValueError("noise_model not recognized")
    from model_module import Model
    model = Model()
  
    randU = lambda low, high: lambda shape: np.random.uniform(low=low, high=high, size=shape)
    const = lambda offset: lambda shape: np.zeros(shape) + offset
    
    if mode is None:
        """Sample loc and scale with default functions"""
        rand_fn_ave = None
        rand_fn_loc = None
        rand_fn_scale = None
    
    elif mode == 'randTheta':
        
        if noise_model in ["nb", "norm"]:
            rand_fn_ave = randU(10, 1000)
            rand_fn_loc = randU(1, 3)
            rand_fn_scale = randU(1, 3)
        elif noise_model == "beta":
            rand_fn_ave = randU(0.1, 0.7)
            rand_fn_loc = randU(0.0, 0.15)
            rand_fn_scale = randU(0.0, 0.15)
        else:
            raise ValueError(f"Noise model {noise_model} not recognized.")
    
    elif mode == 'constTheta':
        
        if noise_model in ["nb", "norm"]:
            rand_fn_ave = randU(10, 1000)
            rand_fn_loc = const(1.0)
            rand_fn_scale = const(1.0)
        elif noise_model == "beta":
            rand_fn_ave = randU(0.1, 0.9)
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
        rand_fn_scale=rand_fn_scale
    )
    return model