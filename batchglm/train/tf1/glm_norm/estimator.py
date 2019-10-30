import logging
import numpy as np
import scipy.sparse
import tensorflow as tf
from typing import Union

from .external import TFEstimatorGLM, InputDataGLM, Model
from .external import closedform_norm_glm_mean, closedform_norm_glm_logsd
from .estimator_graph import EstimatorGraph
from .model import ProcessModel
from .training_strategies import TrainingStrategies

logger = logging.getLogger("batchglm")


class Estimator(TFEstimatorGLM, ProcessModel):
    """
    Estimator for Generalized Linear Models (GLMs) with normal distributed noise.
    Uses the identity function as linker function for loc and a log-linker function for scale.
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
            dtype="float64"
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
        if quick_scale:
            self._train_scale = False

        if len(optim_algos) > 0:
            if np.any([x.lower() in ["nr", "nr_tr"] for x in optim_algos]):
                provide_hessian = True
            if np.any([x.lower() in ["irls", "irls_tr"] for x in optim_algos]):
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
            noise_model="norm",
            dtype=dtype
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
                       np.all(np.abs(input_data.design_scale - 1.) < 1e-8) and \
                       not sf_given

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
                        raise ValueError("cannot use closed_form init for scale model " +
                                         "if scale model differs from loc model")

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
            # Locations model:
            if isinstance(init_a, str) and (init_a.lower() == "auto" or init_a.lower() == "init_model"):
                my_loc_names = set(input_data.loc_names)
                my_loc_names = my_loc_names.intersection(set(init_model.input_data.loc_names))

                init_loc = np.zeros([input_data.num_loc_params, input_data.num_features])
                for parm in my_loc_names:
                    init_idx = np.where(init_model.input_data.loc_names == parm)[0]
                    my_idx = np.where(input_data.loc_names == parm)[0]
                    init_loc[my_idx] = init_model.a_var[init_idx]

                init_a = init_loc
                logger.debug("Using initialization based on input model for mean")

            # Scale model:
            if isinstance(init_b, str) and (init_b.lower() == "auto" or init_b.lower() == "init_model"):
                my_scale_names = set(input_data.scale_names)
                my_scale_names = my_scale_names.intersection(init_model.input_data.scale_names)

                init_scale = np.zeros([input_data.num_scale_params, input_data.num_features])
                for parm in my_scale_names:
                    init_idx = np.where(init_model.input_data.scale_names == parm)[0]
                    my_idx = np.where(input_data.scale_names == parm)[0]
                    init_scale[my_idx] = init_model.b_var[init_idx]

                init_b = init_scale
                logger.debug("Using initialization based on input model for dispersion")

        return init_a, init_b

