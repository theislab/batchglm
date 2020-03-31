import logging
from typing import Union

import numpy as np
try:
    import tensorflow as tf
except:
    tf = None

from .external import TFEstimatorGLM, InputDataGLM, Model
from .external import init_par
from .estimator_graph import EstimatorGraph
from .model import ProcessModel
from .training_strategies import TrainingStrategies


class Estimator(TFEstimatorGLM, ProcessModel):
    """
    Estimator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

    def __init__(
            self,
            input_data: InputDataGLM,
            batch_size: int = 512,
            graph: tf.Graph = None,
            init_model: Model = None,
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            quick_scale: bool = False,
            model: EstimatorGraph = None,
            provide_optimizers: dict = {
                "gd": True,
                "adam": True,
                "adagrad": True,
                "rmsprop": True,
                "nr": True,
                "nr_tr": True,
                "irls": True,
                "irls_gd": True,
                "irls_tr": True,
                "irls_gd_tr": True,
            },
            provide_batched: bool = False,
            provide_fim: bool = False,
            provide_hessian: bool = False,
            optim_algos: list = [],
            extended_summary=False,
            dtype="float64",
            **kwargs
    ):
        """
        Performs initialisation and creates a new estimator.

        :param input_data: InputData
            The input data
        :param batch_size: int
            Size of mini-batches used.
        :param graph: (optional) tf1.Graph
        :param init_model: (optional)
            If provided, this model will be used to initialize this Estimator.
        :param init_a: (Optional)
            Low-level initial values for a. Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "standard": initialize intercept with observed mean
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'a'
        :param init_b: (Optional)
            Low-level initial values for b. Can be:

            - str:
                * "auto": automatically choose best initialization
                * "random": initialize with random values
                * "standard": initialize with zeros
                * "init_model": initialize with another model (see `ìnit_model` parameter)
                * "closed_form": try to initialize with closed form
            - np.ndarray: direct initialization of 'b'
        :param quick_scale: bool
            Whether `scale` will be fitted faster and maybe less accurate.
            Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
        :param model: EstimatorGraph
            EstimatorGraph to use. Basically for debugging.
        :param provide_optimizers:

            E.g.    {"gd": False, "adam": False, "adagrad": False, "rmsprop": False,
                    "nr": False, "nr_tr": True,
                    "irls": False, "irls_gd": False, "irls_tr": False, "irls_gd_tr": False}
        :param provide_batched: bool
            Whether mini-batched optimizers should be provided.
        :param provide_fim: Whether to compute fisher information matrix during training
            Either supply provide_fim and provide_hessian or optim_algos.
        :param provide_hessian: Whether to compute hessians during training
            Either supply provide_fim and provide_hessian or optim_algos.
        :param optim_algos: Algorithms that you want to use on this object. Depending on that,
            the hessian and/or fisher information matrix are computed.
            Either supply provide_fim and provide_hessian or optim_algos.
        :param extended_summary: Include detailed information in the summaries.
            Will increase runtime of summary writer, use only for debugging.
        :param dtype: Precision used in tensorflow.
        """
        if tf is None:
            raise ValueError("tensorflow could not be imported." +
                             "Install tensorflow to use Estimators from the tf1 submodule")
        self.TrainingStrategies = TrainingStrategies

        self._input_data = input_data
        init_a, init_b, train_loc, train_scale = init_par(
            input_data=input_data,
            init_a=init_a,
            init_b=init_b,
            init_model=None
        )
        self._train_loc = train_loc
        self._train_scale = train_scale
        init_a = init_a.astype(dtype)
        init_b = init_b.astype(dtype)
        if quick_scale:
            self._train_scale = False

        if len(optim_algos) > 0:
            if np.any([x.lower() in ["nr", "nr_tr"] for x in optim_algos]):
                provide_hessian = True
            if np.any([x.lower() in ["irls", "irls_tr", "irls_gd", "irls_gd_tr"] for x in optim_algos]):
                provide_fim = True

        TFEstimatorGLM.__init__(
            self=self,
            input_data=input_data,
            batch_size=batch_size,
            graph=graph,
            init_a=init_a,
            init_b=init_b,
            model=model,
            provide_optimizers=provide_optimizers,
            provide_batched=provide_batched,
            provide_fim=provide_fim,
            provide_hessian=provide_hessian,
            extended_summary=extended_summary,
            noise_model="nb",
            dtype=dtype
        )

    def get_model_container(
            self,
            input_data
    ):
        return Model(input_data=input_data)
