import logging
from typing import Union

import numpy as np
import tensorflow as tf

from .external import AbstractEstimator, EstimatorAll, ESTIMATOR_PARAMS, InputData, Model
from .external import closedform_beta_glm_logp, closedform_beta_glm_logq
from .external import SparseXArrayDataArray
from .estimator_graph import EstimatorGraph
from .model import ProcessModel
from .training_strategies import TrainingStrategies

logger = logging.getLogger("batchglm")


class Estimator(EstimatorAll, AbstractEstimator, ProcessModel):
    """
    Estimator for Generalized Linear Models (GLMs) with beta noise.
    Uses the natural logarithm as linker function.
    """

    def __init__(
            self,
            input_data: InputData,
            batch_size: int = 500,
            graph: tf.Graph = None,
            init_model: Model = None,
            init_a: Union[np.ndarray, str] = "closed_form",
            init_b: Union[np.ndarray, str] = "closed_form",
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
            dtype="float64"
    ):
        """
        Performs initialisation and creates a new estimator.

        :param input_data: InputData
            The input data
        :param batch_size: int
            Size of mini-batches used.
        :param graph: (optional) tf.Graph
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
        self.TrainingStrategies = TrainingStrategies

        self._input_data = input_data
        self._train_loc = True
        self._train_scale = True

        (init_a, init_b) = self.init_par(
            input_data=input_data,
            init_a=init_a,
            init_b=init_b,
            init_model=init_model
        )
        init_a = init_a.astype(dtype)
        init_b = init_b.astype(dtype)

        print("init_a: \n", np.exp(init_a))
        print("init_b: \n", np.exp(init_b))

        if len(optim_algos) > 0:
            if np.any([x.lower() in ["nr", "nr_tr"] for x in optim_algos]):
                provide_hessian = True
            if np.any([x.lower() in ["irls", "irls_tr", "irls_gd", "irls_gd_tr"] for x in optim_algos]):
                assert False, "Irls not possible for beta GLM"

        EstimatorAll.__init__(
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
            noise_model="beta",
            dtype=dtype
        )

    @classmethod
    def param_shapes(cls) -> dict:
        return ESTIMATOR_PARAMS

    def init_par(
            self,
            input_data,
            init_a,
            init_b,
            init_model
    ):
        r"""
        standard:
        Only initialise intercept and keep other coefficients as zero.

        closed-form:
        Initialize with Maximum Likelihood / Maximum of Momentum estimators

        Idea:
        $$
            \theta &= f(x) \\
            \Rightarrow f^{-1}(\theta) &= x \\
                &= (D \cdot D^{+}) \cdot x \\
                &= D \cdot (D^{+} \cdot x) \\
                &= D \cdot x' = f^{-1}(\theta)
        $$
        """

        size_factors_init = input_data.size_factors

        if init_model is None:
            if isinstance(init_a, str):
                # Chose option if auto was chosen
                if init_a.lower() == "auto":
                    init_a = "closed_form"
                if init_a.lower() == "closed_form":
                    groupwise_means, init_a, rmsd_a = closedform_beta_glm_logp(
                        X=input_data.X,
                        design_loc=input_data.design_loc,
                        constraints_loc=input_data.constraints_loc.values,
                        design_scale=input_data.design_scale,
                        constraints=input_data.constraints_scale.values,
                        size_factors=size_factors_init,
                        link_fn=lambda p: np.log(self.np_clip_param(p, "p"))
                    )

                    # train p, if the closed-form solution is inaccurate
                    self._train_loc = not (np.all(rmsd_a == 0) or rmsd_a.size == 0)

                    logger.debug("Using closed-form MME initialization for p")
                    logger.debug("Should train p: %s", self._train_loc)
                elif init_a.lower() == "standard":
                    groupwise_means, init_a_intercept, rmsd_a = closedform_beta_glm_logp(
                        X=input_data.X,
                        design_loc=input_data.design_loc[:, [0]],
                        constraints_loc=input_data.constraints_loc[[0], [0]].values,
                        design_scale=input_data.design_scale[:, [0]],
                        constraints=input_data.constraints_scale[[0], [0]].values,
                        size_factors=size_factors_init,
                        link_fn=lambda p: np.log(self.np_clip_param(p, "p"))
                    )
                    init_a = np.zeros([input_data.num_loc_params, input_data.num_features])
                    init_a[0, :] = init_a_intercept
                    self._train_loc = True

                    logger.debug("Using standard initialization for p")
                    logger.debug("Should train p: %s", self._train_loc)
                elif init_a.lower() == "all_zero":
                    init_a = np.zeros([input_data.num_loc_params, input_data.num_features])
                    self._train_loc = True

                    logger.debug("Using all_zero initialization for p")
                    logger.debug("Should train p: %s", self._train_loc)
                else:
                    raise ValueError("init_a string %s not recognized" % init_a)

            if isinstance(init_b, str):
                if init_b.lower() == "auto":
                    init_b = "closed_form"

                if init_b.lower() == "standard":
                    groupwise_scales, init_b_intercept, rmsd_b = closedform_beta_glm_logq(
                        X=input_data.X,
                        design_loc=input_data.design_loc[:, [0]],
                        constraints_loc=input_data.constraints_loc[[0], [0]].values,
                        design_scale=input_data.design_scale[:, [0]],
                        constraints=input_data.constraints_scale[[0], [0]].values,
                        size_factors=size_factors_init,
                        link_fn=lambda q: np.log(self.np_clip_param(q, "q"))
                    )
                    init_b = np.zeros([input_data.num_loc_params, input_data.num_features])
                    init_b[0, :] = init_b_intercept
                    self._train_scale = True

                    logger.debug("Using standard initialization for q")
                    logger.debug("Should train q: %s", self._train_loc)
                elif init_b.lower() == "closed_form":
                    groupwise_scales, init_b, rmsd_b = closedform_beta_glm_logq(
                        X=input_data.X,
                        design_loc=input_data.design_loc,
                        constraints_loc=input_data.constraints_loc.values,
                        design_scale=input_data.design_scale,
                        constraints=input_data.constraints_scale.values,
                        size_factors=size_factors_init,
                        link_fn=lambda q: np.log(self.np_clip_param(q, "q"))
                    )
                    # train q, if the closed-form solution is inaccurate
                    self._train_scale = not (np.all(rmsd_b == 0) or rmsd_b.size == 0)

                    logger.debug("Using closed-form MME initialization for q")
                    logger.debug("Should train q: %s", self._train_scale)
                elif init_b.lower() == "all_zero":
                    init_b = np.zeros([input_data.num_scale_params, input_data.X.shape[1]])

                    logger.debug("Using all_zero initialization for q")
                    logger.debug("Should train r: %s", self._train_scale)
                else:
                    raise ValueError("init_b string %s not recognized" % init_b)
        else:
            # Locations model:
            if isinstance(init_a, str) and (init_a.lower() == "auto" or init_a.lower() == "init_model"):
                my_loc_names = set(input_data.loc_names.values)
                my_loc_names = my_loc_names.intersection(set(init_model.input_data.loc_names.values))

                init_loc = np.zeros([input_data.num_loc_params, input_data.num_features])
                for parm in my_loc_names:
                    init_idx = np.where(init_model.input_data.loc_names == parm)[0]
                    my_idx = np.where(input_data.loc_names == parm)[0]
                    init_loc[my_idx] = init_model.a_var[init_idx]

                init_a = init_loc
                logger.debug("Using initialization based on input model for mean")

            # Scale model:
            if isinstance(init_b, str) and (init_b.lower() == "auto" or init_b.lower() == "init_model"):
                my_scale_names = set(input_data.scale_names.values)
                my_scale_names = my_scale_names.intersection(init_model.input_data.scale_names.values)

                init_scale = np.zeros([input_data.num_scale_params, input_data.num_features])
                for parm in my_scale_names:
                    init_idx = np.where(init_model.input_data.scale_names == parm)[0]
                    my_idx = np.where(input_data.scale_names == parm)[0]
                    init_scale[my_idx] = init_model.b_var[init_idx]

                init_b = init_scale
                logger.debug("Using initialization based on input model for dispersion")

        return init_a, init_b

    @property
    def input_data(self) -> InputData:
        return self._input_data
