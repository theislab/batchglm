import logging
from typing import Union

import numpy as np

from .external import closedform_beta_glm_logitmean, closedform_beta_glm_logsamplesize
from .external import InputDataGLM, Model
from .external import Estimator as GLMEstimator
from .model import BetaGLM, LossGLMBeta
from .processModel import ProcessModel
from .vars import ModelVars
from .training_strategies import TrainingStrategies


class Estimator(GLMEstimator, ProcessModel):
    """
    Estimator for Generalized Linear Models (GLMs) with beta distributed noise.
    Uses a logit linker function for loc and log linker function for scale.
    """

    model: BetaGLM

    def __init__(
            self,
            input_data: InputDataGLM,
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            quick_scale: bool = False,
            dtype="float64",
    ):
        """
        Performs initialisation and creates a new estimator.

        :param input_data: InputDataGLM
            The input data
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
        :param dtype: Precision used in tensorflow.
        """
        self.TrainingStrategies = TrainingStrategies

        self._train_loc = True
        self._train_scale = True

        (init_a, init_b) = self.init_par(
            input_data=input_data,
            init_a=init_a,
            init_b=init_b,
            init_model=None
        )
        init_a = init_a.astype(dtype)
        init_b = init_b.astype(dtype)
        if quick_scale:
            self._train_scale = False

        self.model_vars = ModelVars(
            init_a=init_a,
            init_b=init_b,
            constraints_loc=input_data.constraints_loc,
            constraints_scale=input_data.constraints_scale,
            dtype=dtype
        )

        super(Estimator, self).__init__(
            input_data=input_data,
            dtype=dtype
        )

    def train(
        self,
        use_batching: bool = True,
        batch_size: int = 500,
        optimizer: str = "adam",
        learning_rate: float = 1e-2,
        convergence_criteria: str = "step",
        stopping_criteria: int = 1000,
        autograd: bool = False,
        featurewise: bool = True,
        benchmark: bool = False
    ):
        self.model = BetaGLM(
            model_vars=self.model_vars,
            dtype=self.model_vars.dtype,
            compute_a=self._train_loc,
            compute_b=self._train_scale,
            use_gradient_tape=autograd,
            optimizer=optimizer
        )
        self._loss = LossGLMBeta()

        optimizer_object = self.get_optimizer_object(optimizer, learning_rate)

        super(Estimator, self)._train(
            noise_model="beta",
            use_batching=use_batching,
            batch_size=batch_size,
            optimizer_object=optimizer_object,
            convergence_criteria=convergence_criteria,
            stopping_criteria=stopping_criteria,
            autograd=autograd,
            benchmark=benchmark,
            optimizer=optimizer
        )

    def get_model_container(
            self,
            input_data
    ):
        return Model(input_data=input_data)

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
        """

        size_factors_init = input_data.size_factors

        if init_model is None:
            groupwise_means = None
            init_a_str = None
            if isinstance(init_a, str):
                init_a_str = init_a.lower()
                # Chose option if auto was chosen
                if init_a.lower() == "auto":
                    init_a = "closed_form"

                if init_a.lower() == "closed_form":
                    groupwise_means, init_a, rmsd_a = closedform_beta_glm_logitmean(
                        x=input_data.x,
                        design_loc=input_data.design_loc,
                        constraints_loc=input_data.constraints_loc,
                        size_factors=size_factors_init,
                        link_fn=lambda mean: np.log(
                            1/(1/self.np_clip_param(mean, "mean")-1)
                        )
                    )

                    # train mu, if the closed-form solution is inaccurate
                    self._train_loc = not (np.all(rmsd_a == 0) or rmsd_a.size == 0)

                    logging.getLogger("batchglm").debug("Using closed-form MME initialization for mean")
                elif init_a.lower() == "standard":
                    overall_means = np.mean(input_data.x, axis=0)
                    overall_means = self.np_clip_param(overall_means, "mean")

                    init_a = np.zeros([input_data.num_loc_params, input_data.num_features])
                    init_a[0, :] = np.log(overall_means/(1-overall_means))
                    self._train_loc = True

                    logging.getLogger("batchglm").debug("Using standard initialization for mean")
                elif init_a.lower() == "all_zero":
                    init_a = np.zeros([input_data.num_loc_params, input_data.num_features])
                    self._train_loc = True

                    logging.getLogger("batchglm").debug("Using all_zero initialization for mean")
                else:
                    raise ValueError("init_a string %s not recognized" % init_a)
                logging.getLogger("batchglm").debug("Should train mean: %s", self._train_loc)
            if isinstance(init_b, str):
                if init_b.lower() == "auto":
                    init_b = "standard"

                if init_b.lower() == "standard":
                    groupwise_scales, init_b_intercept, rmsd_b = closedform_beta_glm_logsamplesize(
                        x=input_data.x,
                        design_scale=input_data.design_scale[:, [0]],
                        constraints=input_data.constraints_scale[[0], :][:, [0]],
                        size_factors=size_factors_init,
                        groupwise_means=None,
                        link_fn=lambda samplesize: np.log(self.np_clip_param(samplesize, "samplesize"))
                    )
                    init_b = np.zeros([input_data.num_scale_params, input_data.num_features])
                    init_b[0, :] = init_b_intercept

                    logging.getLogger("batchglm").debug("Using standard-form MME initialization for dispersion")
                elif init_b.lower() == "closed_form":
                    dmats_unequal = False
                    if input_data.num_design_loc_params == input_data.num_design_scale_params:
                        if np.any(input_data.design_loc != input_data.design_scale):
                            dmats_unequal = True

                    inits_unequal = False
                    if init_a_str is not None:
                        if init_a_str != init_b:
                            inits_unequal = True

                    if inits_unequal or dmats_unequal:
                        raise ValueError(
                            "cannot use closed_form init for scale model if scale model differs from loc model"
                        )

                    groupwise_scales, init_b, rmsd_b = closedform_beta_glm_logsamplesize(
                        x=input_data.x,
                        design_scale=input_data.design_scale,
                        constraints=input_data.constraints_scale,
                        size_factors=size_factors_init,
                        groupwise_means=groupwise_means,
                        link_fn=lambda samplesize: np.log(self.np_clip_param(samplesize, "samplesize"))
                    )

                    logging.getLogger("batchglm").debug("Using closed-form MME initialization for dispersion")
                elif init_b.lower() == "all_zero":
                    init_b = np.zeros([input_data.num_scale_params, input_data.num_features])

                    logging.getLogger("batchglm").debug("Using standard initialization for dispersion")
                else:
                    raise ValueError("init_b string %s not recognized" % init_b)
                logging.getLogger("batchglm").debug("Should train r: %s", self._train_scale)
        else:
            init_a, init_b = self.get_init_from_model(init_a=init_a,
                                                      init_b=init_b,
                                                      input_data=input_data,
                                                      init_model=init_model)

        return init_a, init_b
