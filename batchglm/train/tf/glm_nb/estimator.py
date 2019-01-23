import logging
from typing import Union

import numpy as np
import tensorflow as tf

from .external import AbstractEstimator, EstimatorAll, ESTIMATOR_PARAMS, InputData, Model
from .external import data_utils
from .external import closedform_nb_glm_logmu, closedform_nb_glm_logphi
from .external import SparseXArrayDataArray
from .estimator_graph import EstimatorGraph
from .model import ProcessModel
from .training_strategies import TrainingStrategies

logger = logging.getLogger(__name__)


class Estimator(EstimatorAll, AbstractEstimator, ProcessModel):
    """
    Estimator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

    def __init__(
            self,
            input_data: InputData,
            batch_size: int = 500,
            graph: tf.Graph = None,
            init_model: Model = None,
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            quick_scale: bool = False,
            model: EstimatorGraph = None,
            provide_optimizers: dict = None,
            termination_type: str = "by_feature",
            extended_summary=False,
            dtype="float64",
    ):
        self.TrainingStrategies = TrainingStrategies
        EstimatorAll.__init__(
            self=self,
            input_data=input_data,
            batch_size=batch_size,
            graph=graph,
            init_model=init_model,
            init_a=init_a,
            init_b=init_b,
            quick_scale=quick_scale,
            model=model,
            provide_optimizers=provide_optimizers,
            termination_type=termination_type,
            extended_summary=extended_summary,
            noise_model="nb",
            dtype=dtype
        )

    @classmethod
    def param_shapes(cls) -> dict:
        return ESTIMATOR_PARAMS

    def init_par(
            self,
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

        size_factors_init = self.input_data.size_factors
        if size_factors_init is not None:
            size_factors_init = np.expand_dims(size_factors_init, axis=1)
            size_factors_init = np.broadcast_to(
                array=size_factors_init,
                shape=[self.input_data.num_observations, self.input_data.num_features]
            )

        groupwise_means = None
        init_a_str = None
        if isinstance(init_a, str):
            init_a_str = init_a.lower()
            # Chose option if auto was chosen
            if init_a.lower() == "auto":
                init_a = "closed_form"

            if init_a.lower() == "closed_form":
                #try:
                groupwise_means, init_a, rmsd_a = closedform_nb_glm_logmu(
                    X=self.input_data.X,
                    design_loc=self.input_data.design_loc,
                    constraints_loc=self.input_data.constraints_loc.values,
                    size_factors=size_factors_init,
                    link_fn=lambda mu: np.log(self.np_clip_param(mu, "mu"))
                )

                # train mu, if the closed-form solution is inaccurate
                self._train_loc = not np.all(rmsd_a == 0)

                # Temporal fix: train mu if size factors are given as closed form may be different:
                if self.input_data.size_factors is not None:
                    self._train_loc = True

                logger.debug("Using closed-form MLE initialization for mean")
                logger.debug("Should train mu: %s", self._train_loc)
                #except np.linalg.LinAlgError:
                #    logger.warning("Closed form initialization failed!")
            elif init_a.lower() == "standard":
                if isinstance(self.input_data.X, SparseXArrayDataArray):
                    overall_means = self.input_data.X.mean(dim="observations")
                else:
                    overall_means = self.input_data.X.mean(dim="observations").values  # directly calculate the mean
                overall_means = self.np_clip_param(overall_means, "mu")

                init_a = np.zeros([self.input_data.num_loc_params, self.input_data.num_features])
                init_a[0, :] = np.log(overall_means)
                self._train_loc = True

                logger.debug("Using standard initialization for mean")
                logger.debug("Should train mu: %s", self._train_loc)
            elif init_a.lower() == "all_zero":
                init_a = np.zeros([self.input_data.num_loc_params, self.input_data.num_features])
                self._train_loc = True

                logger.debug("Using all_zero initialization for mean")
                logger.debug("Should train mu: %s", self._train_loc)
            else:
                raise ValueError("init_a string %s not recognized" % init_a)

        if isinstance(init_b, str):
            if init_b.lower() == "auto":
                init_b = "standard"

            if init_b.lower() == "closed_form" or init_b.lower() == "standard":
                #try:
                # Check whether it is necessary to recompute group-wise means.
                dmats_unequal = False
                if self.input_data.design_loc.shape[1] == self.input_data.design_scale.shape[1]:
                    if np.any(self.input_data.design_loc.values != self.input_data.design_scale.values):
                        dmats_unequal = True

                inits_unequal = False
                if init_a_str is not None:
                    if init_a_str != init_b:
                        inits_unequal = True

                if inits_unequal or dmats_unequal:
                    groupwise_means = None

                # Watch out: init_mu is full obs x features matrix and is very large in many cases.
                if inits_unequal or dmats_unequal:
                    if isinstance(self.input_data.X, SparseXArrayDataArray):
                        init_mu = np.exp(
                            np.matmul(
                                self.input_data.design_loc.values,
                                np.matmul(self.input_data.constraints_loc.values, init_a)
                            ) + size_factors_init
                        )
                    else:
                        init_a_xr = data_utils.xarray_from_data(init_a, dims=("loc_params", "features"))
                        init_a_xr.coords["loc_params"] = self.input_data.constraints_loc.coords["loc_params"]
                        init_mu = np.exp(
                            self.input_data.design_loc.dot(self.input_data.constraints_loc.dot(init_a_xr)) +
                            size_factors_init
                        )
                else:
                    init_mu = None

                if init_b.lower() == "closed_form":
                    groupwise_scales, init_b, rmsd_b = closedform_nb_glm_logphi(
                        X=self.input_data.X,
                        mu=init_mu,
                        design_scale=self.input_data.design_scale,
                        constraints=self.input_data.constraints_scale.values,
                        size_factors=size_factors_init,
                        groupwise_means=groupwise_means,
                        link_fn=lambda r: np.log(self.np_clip_param(r, "r"))
                    )

                    logger.debug("Using closed-form MME initialization for dispersion")
                    logger.debug("Should train r: %s", self._train_scale)
                elif init_b.lower() == "standard":
                    groupwise_scales, init_b_intercept, rmsd_b = closedform_nb_glm_logphi(
                        X=self.input_data.X,
                        mu=init_mu,
                        design_scale=self.input_data.design_scale[:,[0]],
                        constraints=self.input_data.constraints_scale[[0], [0]].values,
                        size_factors=size_factors_init,
                        groupwise_means=None,
                        link_fn=lambda r: np.log(self.np_clip_param(r, "r"))
                    )
                    init_b = np.zeros([self.input_data.num_scale_params, self.input_data.X.shape[1]])
                    init_b[0, :] = init_b_intercept

                    logger.debug("Using closed-form MME initialization for dispersion")
                    logger.debug("Should train r: %s", self._train_scale)
                #except np.linalg.LinAlgError:
                #    logger.warning("Closed form initialization failed!")
            elif init_b.lower() == "all_zero":
                init_b = np.zeros([self.input_data.num_scale_params, self.input_data.X.shape[1]])

                logger.debug("Using standard initialization for dispersion")
                logger.debug("Should train r: %s", self._train_scale)
            else:
                raise ValueError("init_b string %s not recognized" % init_b)

        if init_model is not None:
            # Locations model:
            if isinstance(init_a, str) and (init_a.lower() == "auto" or init_a.lower() == "init_model"):
                my_loc_names = set(self.input_data.design_loc_names.values)
                my_loc_names = my_loc_names.intersection(init_model.input_data.design_loc_names.values)

                init_loc = np.zeros([self.input_data.num_loc_params, self.input_data.num_features])
                for parm in my_loc_names:
                    init_idx = np.where(init_model.input_data.design_loc_names == parm)
                    my_idx = np.where(self.input_data.design_loc_names == parm)
                    init_loc[my_idx] = init_model.par_link_loc[init_idx]

                init_a = init_loc
                logger.debug("Using initialization based on input model for mean")

            # Scale model:
            if isinstance(init_b, str) and (init_b.lower() == "auto" or init_b.lower() == "init_model"):
                my_scale_names = set(self.input_data.design_scale_names.values)
                my_scale_names = my_scale_names.intersection(init_model.input_data.design_scale_names.values)

                init_scale = np.zeros([self.input_data.num_scale_params, self.input_data.num_features])
                for parm in my_scale_names:
                    init_idx = np.where(init_model.input_data.design_scale_names == parm)
                    my_idx = np.where(self.input_data.design_scale_names == parm)
                    init_scale[my_idx] = init_model.par_link_scale[init_idx]

                init_b = init_scale
                logger.debug("Using initialization based on input model for dispersion")

        return init_a, init_b

    @property
    def input_data(self) -> InputData:
        return self._input_data
