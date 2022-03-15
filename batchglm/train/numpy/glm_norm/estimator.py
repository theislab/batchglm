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

from .external import pkg_constants, EstimatorGlm, Model, init_par
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
        init_theta_location, init_theta_scale, _, _ = init_par(
            model=model, init_location=init_location, init_scale=init_scale
        )
        init_theta_location = init_theta_location.astype(dtype)
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
        pass
