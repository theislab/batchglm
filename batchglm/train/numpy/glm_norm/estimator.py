import abc
import logging
import multiprocessing
import pprint
import sys
import time
from enum import Enum
from typing import List, Tuple

import dask.array
import numpy as np
import scipy
import scipy.optimize
import scipy.sparse
import sparse

from .external import pkg_constants, EstimatorGlm, Model, closedform_norm_glm_logsd, closedform_norm_glm_mean
from .model_container import ModelContainer

logger = logging.getLogger("batchglm")


class Estimator(EstimatorGlm):
    

    def __init__(
        self,
        model: Model,
        init_location: str = "AUTO",
        init_scale: str = "AUTO",
        # batch_size: Optional[Union[Tuple[int, int], int]] = None,
        quick_scale: bool = False,
        dtype: str = "float64",
    ):
        """
        Performs initialisation and creates a new estimator.
        :param model:
            The IWLS model to be fit
        :param dtype:
            i.e float64
        """
        init_theta_location = np.zeros([model.num_loc_params, model.num_features])
        init_theta_location = init_theta_location.astype(dtype)
        init_theta_scale = np.zeros([model.num_scale_params, model.x.shape[1]])
        init_theta_scale = init_theta_scale.astype(dtype)
        _model_container = ModelContainer(
            model=model,
            init_theta_location=init_theta_location,
            init_theta_scale=init_theta_scale,
            chunk_size_genes=model.chunk_size_genes,
            dtype=dtype,
        )
        super(Estimator, self).__init__(model_container=_model_container, dtype=dtype)

    def train(
        self,
        **kwargs,
    ):
        model = self._model_container.model
        _, theta_location, _ = closedform_norm_glm_mean(
            x=model.x,
            design_loc=model.design_loc,
            constraints_loc=model.constraints_loc,
            size_factors=model.size_factors,
        )
        _, theta_scale, _ = closedform_norm_glm_logsd(
            x=model.x,
            design_scale=model.design_scale,
            constraints=model.constraints_scale,
            size_factors=model.size_factors,
        )
        self._model_container.theta_location = theta_location
        self._model_container.theta_scale = theta_scale
