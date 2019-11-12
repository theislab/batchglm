import logging
import numpy as np
import scipy.sparse
from typing import Union

from .external import closedform_norm_glm_logsd
from .external import InputDataGLM, Model
from .external import Estimator as GLMEstimator
from .model import NormGLM, LossGLMNorm
from .processModel import ProcessModel
from .vars import ModelVars
from .training_strategies import TrainingStrategies


logger = logging.getLogger("batchglm")


class Estimator(GLMEstimator, ProcessModel):
    """
    Estimator for Generalized Linear Models (GLMs) with normal distributed noise.
    Uses the identity function as linker function for loc and a log-linker function for scale.
    """

    model: NormGLM
    loss: LossGLMNorm

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
                * "all zero": initialize with zeros
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

        self.model = NormGLM(
            model_vars=self.model_vars,
            dtype=self.model_vars.dtype,
            compute_a=self._train_loc,
            compute_b=self._train_scale,
            use_gradient_tape=autograd,
            optimizer=optimizer
        )

        self._loss = LossGLMNorm()

        optimizer_object = self.get_optimizer_object(optimizer, learning_rate)

        super(Estimator, self)._train(
            noise_model="norm",
            use_batching=use_batching,
            batch_size=batch_size,
            optimizer_object=optimizer_object,
            convergence_criteria=convergence_criteria,
            stopping_criteria=stopping_criteria,
            autograd=autograd,
            featurewise=featurewise,
            benchmark=benchmark

        )

    def get_model_container(
            self,
            input_data
    ):
        return Model(input_data=input_data)

    def init_par(self, input_data, init_a, init_b, init_model):
        r"""
        standard:
        Only initialise intercept and keep other coefficients as zero.

        closed-form:
        Initialize with Maximum Likelihood / Maximum of Momentum estimators
        """

        size_factors_init = input_data.size_factors
        if size_factors_init is not None:
            size_factors_init = np.expand_dims(size_factors_init, axis=1)
            size_factors_init = np.broadcast_to(
                array=size_factors_init,
                shape=[input_data.num_observations, input_data.num_features]
            )

        sf_given = False
        if input_data.size_factors is not None:
            if np.any(np.abs(input_data.size_factors - 1.) > 1e-8):
                sf_given = True

        is_ols_model = input_data.design_scale.shape[1] == 1 and \
            np.all(np.abs(input_data.design_scale - 1.) < 1e-8) and not sf_given

        if init_model is None:
            groupwise_means = None
            init_a_str = None
            if isinstance(init_a, str):
                init_a_str = init_a.lower()
                # Chose option if auto was chosen
                if init_a.lower() == "auto":
                    init_a = "closed_form"

                if init_a.lower() == "closed_form" or init_a.lower() == "standard":
                    design_constr = np.matmul(input_data.design_loc, input_data.constraints_loc)
                    # Iterate over genes if X is sparse to avoid large sparse tensor.
                    # If X is dense, the least square problem can be vectorised easily.
                    if isinstance(input_data.x, scipy.sparse.csr_matrix):
                        init_a, rmsd_a, _, _ = np.linalg.lstsq(
                            np.matmul(design_constr.T, design_constr),
                            input_data.x.T.dot(design_constr).T,  # need double .T because of dot product on sparse.
                            rcond=None
                        )
                    else:
                        init_a, rmsd_a, _, _ = np.linalg.lstsq(
                            np.matmul(design_constr.T, design_constr),
                            np.matmul(design_constr.T, input_data.x),
                            rcond=None
                        )
                    groupwise_means = None
                    if is_ols_model:
                        self._train_loc = False

                    logger.debug("Using OLS initialization for location model")
                elif init_a.lower() == "all_zero":
                    init_a = np.zeros([input_data.num_loc_params, input_data.num_features])
                    self._train_loc = True

                    logger.debug("Using all_zero initialization for mean")
                else:
                    raise ValueError("init_a string %s not recognized" % init_a)
                logger.debug("Should train location model: %s", self._train_loc)

            if isinstance(init_b, str):
                if init_b.lower() == "auto":
                    init_b = "standard"

                if is_ols_model:
                    # Calculated variance via E(x)^2 or directly depending on whether `mu` was specified.
                    if isinstance(input_data.x, scipy.sparse.csr_matrix):
                        expect_xsq = np.asarray(np.mean(input_data.x.power(2), axis=0))
                    else:
                        expect_xsq = np.expand_dims(np.mean(np.square(input_data.x), axis=0), axis=0)
                    mean_model = np.matmul(
                        np.matmul(input_data.design_loc, input_data.constraints_loc),
                        init_a
                    )
                    expect_x_sq = np.mean(np.square(mean_model), axis=0)
                    variance = (expect_xsq - expect_x_sq)
                    init_b = np.log(np.sqrt(variance))
                    self._train_scale = False

                    logger.debug("Using residuals from OLS estimate for variance estimate")
                elif init_b.lower() == "closed_form":
                    dmats_unequal = False
                    if input_data.design_loc.shape[1] == input_data.design_scale.shape[1]:
                        if np.any(input_data.design_loc != input_data.design_scale):
                            dmats_unequal = True

                    inits_unequal = False
                    if init_a_str is not None:
                        if init_a_str != init_b:
                            inits_unequal = True

                    # Watch out: init_mean is full obs x features matrix and is very large in many cases.
                    if inits_unequal or dmats_unequal:
                        raise ValueError(
                            "cannot use closed_form init for scale model \
                            if scale model differs from loc model"
                        )

                    groupwise_scales, init_b, rmsd_b = closedform_norm_glm_logsd(
                        x=input_data.x,
                        design_scale=input_data.design_scale,
                        constraints=input_data.constraints_scale,
                        size_factors=size_factors_init,
                        groupwise_means=groupwise_means,
                        link_fn=lambda sd: np.log(self.np_clip_param(sd, "sd"))
                    )

                    # train scale, if the closed-form solution is inaccurate
                    self._train_scale = not (np.all(rmsd_b == 0) or rmsd_b.size == 0)

                    logger.debug("Using closed-form MME initialization for standard deviation")
                elif init_b.lower() == "standard":
                    groupwise_scales, init_b_intercept, rmsd_b = closedform_norm_glm_logsd(
                        x=input_data.x,
                        design_scale=input_data.design_scale[:, [0]],
                        constraints=input_data.constraints_scale[[0], :][:, [0]],
                        size_factors=size_factors_init,
                        groupwise_means=None,
                        link_fn=lambda sd: np.log(self.np_clip_param(sd, "sd"))
                    )
                    init_b = np.zeros([input_data.num_scale_params, input_data.num_features])
                    init_b[0, :] = init_b_intercept

                    # train scale, if the closed-form solution is inaccurate
                    self._train_scale = not (np.all(rmsd_b == 0) or rmsd_b.size == 0)

                    logger.debug("Using closed-form MME initialization for standard deviation")
                    logger.debug("Should train sd: %s", self._train_scale)
                elif init_b.lower() == "all_zero":
                    init_b = np.zeros([input_data.num_scale_params, input_data.num_features])

                    logger.debug("Using standard initialization for standard deviation")
                else:
                    raise ValueError("init_b string %s not recognized" % init_b)
                logger.debug("Should train sd: %s", self._train_scale)
        else:
            init_a, init_b = self.get_init_from_model(init_a=init_a,
                                                      init_b=init_b,
                                                      input_data=input_data,
                                                      init_model=init_model)

        return init_a, init_b
