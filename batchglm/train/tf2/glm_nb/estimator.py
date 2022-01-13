import logging
from typing import Union
import numpy as np

from .external import InputDataGLM, Model
from .external import closedform_nb_glm_logmu, closedform_nb_glm_logphi
from .model import NBGLM
from .vars import ModelVars
from .processModel import ProcessModel
from .external import Estimator as GLMEstimator
from .training_strategies import TrainingStrategies

# needed for train_irls_ls_tr
from .optim import IRLS_LS


class Estimator(GLMEstimator, ProcessModel):
    """
    Estimator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """
    model: NBGLM

    def __init__(
            self,
            input_data: InputDataGLM,
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            quick_scale: bool = False,
            dtype="float32",
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
            batch_size: int = 5000,
            optim_algo: str = "adam",
            learning_rate: float = 1e-2,
            convergence_criteria: str = "step",
            stopping_criteria: int = 1000,
            autograd: bool = False,
            featurewise: bool = True,
            benchmark: bool = False,
            maxiter: int = 1,
            b_update_freq = 1
    ):
        if self.model is None:
            self.model = NBGLM(
                model_vars=self.model_vars,
                dtype=self.model_vars.dtype,
                compute_a=self._train_loc,
                compute_b=self._train_scale,
                use_gradient_tape=autograd,
                optimizer=optim_algo
            )
        else:
            self.model.setMethod(optim_algo)

        intercept_scale = len(self.model.model_vars.idx_train_scale) == 1
        optimizer_object = self.get_optimizer_object(optim_algo, learning_rate, intercept_scale)
        self.optimizer = optimizer_object
        if optim_algo.lower() in ['irls_gd_tr', 'irls_ar_tr', 'irls_tr_gd_tr']:
            self.update = self.update_separated
            self.maxiter = maxiter

        super(Estimator, self)._train(
            noise_model="nb",
            is_batched=use_batching,
            batch_size=batch_size,
            optimizer_object=optimizer_object,
            convergence_criteria=convergence_criteria,
            stopping_criteria=stopping_criteria,
            autograd=autograd,
            featurewise=featurewise,
            benchmark=benchmark,
            optim_algo=optim_algo,
            b_update_freq=b_update_freq
        )

    def get_optimizer_object(self, optimizer, learning_rate, intercept_scale):
        optim = optimizer.lower()
        if optim in ['irls_gd_tr', 'irls_gd', 'irls_ar_tr', 'irls_tr_gd_tr']:
            return IRLS_LS(
                dtype=self.dtype,
                tr_mode=optim.endswith('tr'),
                model=self.model,
                name=optim,
                n_obs=self.input_data.num_observations,
                intercept_scale=intercept_scale)
        return super().get_optimizer_object(optimizer, learning_rate)

    def update_separated(self, results, batches, batch_features, compute_b):

        self.optimizer.perform_parameter_update(
            inputs=[batches, *results],
            compute_a=True,
            compute_b=False,
            batch_features=batch_features,
            is_batched=False
        )
        if compute_b:
            self.optimizer.perform_parameter_update(
                inputs=[batches, *results],
                compute_a=False,
                compute_b=True,
                batch_features=batch_features,
                is_batched=False,
                maxiter=self.maxiter
            )
        else:
            self.model.model_vars.updated_b = np.zeros_like(self.model.model_vars.updated_b)


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

        Idea:
        $$
            \theta &= f(x) \\
            \Rightarrow f^{-1}(\theta) &= x \\
                &= (D \cdot D^{+}) \cdot x \\
                &= D \cdot (D^{+} \cdot x) \\
                &= D \cdot x' = f^{-1}(\theta)
        $$
        """

        if init_model is None:
            groupwise_means = None
            init_a_str = None
            if isinstance(init_a, str):
                init_a_str = init_a.lower()
                # Chose option if auto was chosen
                if init_a.lower() == "auto":
                    init_a = "standard"

                if init_a.lower() == "closed_form":
                    groupwise_means, init_a, rmsd_a = closedform_nb_glm_logmu(
                        x=input_data.x,
                        design_loc=input_data.design_loc,
                        constraints_loc=input_data.constraints_loc,
                        size_factors=input_data.size_factors,
                        link_fn=lambda loc: np.log(self.np_clip_param(loc, "loc"))
                    )

                    # train mu, if the closed-form solution is inaccurate
                    self._train_loc = not (np.all(rmsd_a == 0) or rmsd_a.size == 0)

                    if input_data.size_factors is not None:
                        if np.any(input_data.size_factors != 1):
                            self._train_loc = True

                    logging.getLogger("batchglm").debug("Using closed-form MLE initialization for mean")
                    logging.getLogger("batchglm").debug("Should train loc: %s", self._train_loc)
                elif init_a.lower() == "standard":
                    overall_means = np.mean(input_data.x, axis=0)  # directly calculate the mean
                    overall_means = self.np_clip_param(overall_means, "loc")

                    init_a = np.zeros([input_data.num_loc_params, input_data.num_features])
                    init_a[0, :] = np.log(overall_means)
                    self._train_loc = True

                    logging.getLogger("batchglm").debug("Using standard initialization for mean")
                    logging.getLogger("batchglm").debug("Should train loc: %s", self._train_loc)
                elif init_a.lower() == "all_zero":
                    init_a = np.zeros([input_data.num_loc_params, input_data.num_features])
                    self._train_loc = True

                    logging.getLogger("batchglm").debug("Using all_zero initialization for mean")
                    logging.getLogger("batchglm").debug("Should train loc: %s", self._train_loc)
                else:
                    raise ValueError("init_a string %s not recognized" % init_a)

            if isinstance(init_b, str):
                if init_b.lower() == "auto":
                    init_b = "standard"

                if init_b.lower() == "standard":
                    groupwise_scales, init_b_intercept, rmsd_b = closedform_nb_glm_logphi(
                        x=input_data.x,
                        design_scale=input_data.design_scale[:, [0]],
                        constraints=input_data.constraints_scale[[0], :][:, [0]],
                        size_factors=input_data.size_factors,
                        groupwise_means=None,
                        link_fn=lambda scale: np.log(self.np_clip_param(scale, "scale"))
                    )
                    init_b = np.zeros([input_data.num_scale_params, input_data.num_features])
                    init_b[0, :] = init_b_intercept

                    logging.getLogger("batchglm").debug("Using standard-form MME initialization for dispersion")
                    logging.getLogger("batchglm").debug("Should train scale: %s", self._train_scale)
                elif init_b.lower() == "closed_form":
                    dmats_unequal = False
                    if input_data.design_loc.shape[1] == input_data.design_scale.shape[1]:
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

                    groupwise_scales, init_b, rmsd_b = closedform_nb_glm_logphi(
                        x=input_data.x,
                        design_scale=input_data.design_scale,
                        constraints=input_data.constraints_scale,
                        size_factors=input_data.size_factors,
                        groupwise_means=groupwise_means,
                        link_fn=lambda scale: np.log(self.np_clip_param(scale, "scale"))
                    )

                    logging.getLogger("batchglm").debug("Using closed-form MME initialization for dispersion")
                    logging.getLogger("batchglm").debug("Should train scale: %s", self._train_scale)
                elif init_b.lower() == "all_zero":
                    init_b = np.zeros([input_data.num_scale_params, input_data.x.shape[1]])

                    logging.getLogger("batchglm").debug("Using standard initialization for dispersion")
                    logging.getLogger("batchglm").debug("Should train scale: %s", self._train_scale)
                else:
                    raise ValueError("init_b string %s not recognized" % init_b)
        else:
            init_a, init_b = self.get_init_from_model(init_a=init_a,
                                                      init_b=init_b,
                                                      input_data=input_data,
                                                      init_model=init_model)

        return init_a, init_b
