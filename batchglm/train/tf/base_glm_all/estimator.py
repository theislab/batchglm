import abc
from copy import copy, deepcopy
from typing import Union
import logging
import pprint
from enum import Enum

import tensorflow as tf

import numpy as np

from .estimator_graph import EstimatorGraphAll
from .external import MonitoredTFEstimator, InputData, _Model_GLM, SparseXArrayDataArray

logger = logging.getLogger(__name__)


class EstimatorAll(MonitoredTFEstimator, metaclass=abc.ABCMeta):
    """
    Estimator for Generalized Linear Models (GLMs) with negative binomial noise.
    Uses the natural logarithm as linker function.
    """

    class TrainingStrategy(Enum):
        pass

    model: EstimatorGraphAll
    _train_loc: bool
    _train_scale: bool

    def __init__(
            self,
            input_data: InputData,
            batch_size: int = 500,
            graph: tf.Graph = None,
            init_model: _Model_GLM = None,
            init_a: Union[np.ndarray, str] = "AUTO",
            init_b: Union[np.ndarray, str] = "AUTO",
            quick_scale: bool = False,
            model: EstimatorGraphAll = None,
            provide_optimizers: dict = None,
            termination_type: str = "by_feature",
            extended_summary=False,
            noise_model: str = None,
            dtype=tf.float64
    ):
        """
        Create a new Estimator

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
        :param model: EstimatorGraph
            EstimatorGraph to use. Basically for debugging.
        :param provide_optimizers:

            E.g. {"gd": True, "adam": True, "adagrad": True, "rmsprop": True, "nr": True, "irls": True}
        :param termination_type:
        :param extended_summary:
        :param dtype: Precision used in tensorflow.

        Useful in scenarios where fitting the exact `scale` is not absolutely necessary.
        :param extended_summary: Include detailed information in the summaries.
            Will drastically increase runtime of summary writer, use only for debugging.
        """
        if noise_model == "nb":
            from .external_nb import EstimatorGraph
        else:
            raise ValueError("noise model %s was not recognized" % noise_model)
        self.noise_model = noise_model

        # validate design matrix:
        if np.linalg.matrix_rank(input_data.design_loc) != np.linalg.matrix_rank(input_data.design_loc.T):
            raise ValueError("design_loc matrix is not full rank")
        if np.linalg.matrix_rank(input_data.design_scale) != np.linalg.matrix_rank(input_data.design_scale.T):
            raise ValueError("design_scale matrix is not full rank")

        # ### initialization
        if model is None:
            if graph is None:
                graph = tf.Graph()

        # ### prepare fetch_fn:
        def fetch_fn(idx):
            r"""
            Documentation of tensorflow coding style in this function:
            tf.py_func defines a python function (the getters of the InputData object slots)
            as a tensorflow operation. Here, the shape of the tensor is lost and
            has to be set with set_shape. For size factors, we use explicit broadcasting
            as explained below.
            """
            # Catch dimension collapse error if idx is only one element long, ie. 0D:
            if len(idx.shape) == 0:
                idx = tf.expand_dims(idx, axis=0)

            if isinstance(input_data.X, SparseXArrayDataArray):
                X_tensor_idx, X_tensor_val, X_shape = tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                    func=input_data.fetch_X_sparse,
                    inp=[idx],
                    Tout=[np.int64, np.float64, np.int64],
                    stateful=False  #  TODO: remove with tf>=v1.13
                )
                X_tensor_idx = tf.cast(X_tensor_idx, dtype=tf.int64)
                X_shape = tf.cast(X_shape, dtype=tf.int64)
                X_tensor_val = tf.cast(X_tensor_val, dtype=dtype)
                X_tensor = (X_tensor_idx, X_tensor_val, X_shape)
            else:
                X_tensor = tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                    func=input_data.fetch_X_dense,
                    inp=[idx],
                    Tout=input_data.X.dtype,
                    stateful=False  #  TODO: remove with tf>=v1.13
                )
                X_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_features])
                X_tensor = (tf.cast(X_tensor, dtype=dtype),)

            design_loc_tensor = tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                func=input_data.fetch_design_loc,
                inp=[idx],
                Tout=input_data.design_loc.dtype,
                stateful=False  #  TODO: remove with tf>=v1.13
            )
            design_loc_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_design_loc_params])
            design_loc_tensor = tf.cast(design_loc_tensor, dtype=dtype)

            design_scale_tensor = tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                func=input_data.fetch_design_scale,
                inp=[idx],
                Tout=input_data.design_scale.dtype,
                stateful=False  #  TODO: remove with tf>=v1.13
            )
            design_scale_tensor.set_shape(idx.get_shape().as_list() + [input_data.num_design_scale_params])
            design_scale_tensor = tf.cast(design_scale_tensor, dtype=dtype)

            if input_data.size_factors is not None:
                size_factors_tensor = tf.log(tf.py_func(  #tf.py_function( TODO: replace with tf>=v1.13
                    func=input_data.fetch_size_factors,
                    inp=[idx],
                    Tout=input_data.size_factors.dtype,
                    stateful=False  #  TODO: remove with tf>=v1.13
                ))
                size_factors_tensor.set_shape(idx.get_shape())
                size_factors_tensor = tf.expand_dims(size_factors_tensor, axis=-1)
                size_factors_tensor = tf.cast(size_factors_tensor, dtype=dtype)
            else:
                size_factors_tensor = tf.constant(0, shape=[1, 1], dtype=dtype)
            size_factors_tensor = tf.broadcast_to(size_factors_tensor,
                                                  shape=[tf.size(idx), input_data.num_features])

            # return idx, data
            return idx, (X_tensor, design_loc_tensor, design_scale_tensor, size_factors_tensor)

        with graph.as_default():
            # create model
            model = EstimatorGraph(
                fetch_fn=fetch_fn,
                feature_isnonzero=self._input_data.feature_isnonzero,
                num_observations=self._input_data.num_observations,
                num_features=self._input_data.num_features,
                num_design_loc_params=self._input_data.num_design_loc_params,
                num_design_scale_params=self._input_data.num_design_scale_params,
                num_loc_params=self._input_data.num_loc_params,
                num_scale_params=self._input_data.num_scale_params,
                batch_size=batch_size,
                graph=graph,
                init_a=init_a,
                init_b=init_b,
                constraints_loc=self._input_data.constraints_loc,
                constraints_scale=self._input_data.constraints_scale,
                provide_optimizers=provide_optimizers,
                train_loc=self._train_loc,
                train_scale=self._train_scale,
                termination_type=termination_type,
                extended_summary=extended_summary,
                noise_model=self.noise_model,
                dtype=dtype
            )

        MonitoredTFEstimator.__init__(self, model)
        model.session = self.session

    def _scaffold(self):
        with self.model.graph.as_default():
            scaffold = tf.train.Scaffold(
                init_op=self.model.init_op,
                summary_op=self.model.merged_summary,
                saver=self.model.saver,
            )
        return scaffold

    def train(self, *args,
              learning_rate=None,
              convergence_criteria="t_test",
              loss_window_size=100,
              stopping_criteria=0.05,
              train_mu: bool = None,
              train_r: bool = None,
              use_batching=True,
              optim_algo="gradient_descent",
              **kwargs):
        r"""
        Starts training of the model

        :param feed_dict: dict of values which will be feeded each `session.run()`

            See also feed_dict parameter of `session.run()`.
        :param learning_rate: learning rate used for optimization
        :param convergence_criteria: criteria after which the training will be interrupted.
            Currently implemented criterias:

            - "step":
              stop, when the step counter reaches `stopping_criteria`
            - "difference":
              stop, when `loss(step=i) - loss(step=i-1)` < `stopping_criteria`
            - "moving_average":
              stop, when `mean_loss(steps=[i-2N..i-N) - mean_loss(steps=[i-N..i)` < `stopping_criteria`
            - "absolute_moving_average":
              stop, when `|mean_loss(steps=[i-2N..i-N) - mean_loss(steps=[i-N..i)|` < `stopping_criteria`
            - "t_test" (recommended):
              Perform t-Test between the last [i-2N..i-N] and [i-N..i] losses.
              Stop if P("both distributions are equal") > `stopping_criteria`.
        :param stopping_criteria: Additional parameter for convergence criteria.

            See parameter `convergence_criteria` for exact meaning
        :param loss_window_size: specifies `N` in `convergence_criteria`.
        :param train_mu: Set to True/False in order to enable/disable training of mu
        :param train_r: Set to True/False in order to enable/disable training of r
        :param use_batching: If True, will use mini-batches with the batch size defined in the constructor.
            Otherwise, the gradient of the full dataset will be used.
        :param optim_algo: name of the requested train op. Can be:

            - "Adam"
            - "Adagrad"
            - "RMSprop"
            - "GradientDescent" or "GD"

            See :func:train_utils.MultiTrainer.train_op_by_name for further details.
        """
        if train_mu is None:
            # check if mu was initialized with MLE
            train_mu = self._train_loc
        if train_r is None:
            # check if r was initialized with MLE
            train_r = self._train_scale

        # Check whether newton-rhapson is desired:
        newton_type_mode = False
        if optim_algo.lower() == "newton" or \
                optim_algo.lower() == "newton-raphson" or \
                optim_algo.lower() == "newton_raphson" or \
                optim_algo.lower() == "nr" or \
                optim_algo.lower() == "irls" or \
                optim_algo.lower() == "iwls" or \
                optim_algo.lower() == "newton-trust-region" or \
                optim_algo.lower() == "newton_trust_region" or \
                optim_algo.lower() == "newton-raphson-trust-region" or \
                optim_algo.lower() == "newton_raphson_trust_region" or \
                optim_algo.lower() == "newton_tr" or \
                optim_algo.lower() == "nr_tr" or \
                optim_algo.lower() == "irls_tr" or \
                optim_algo.lower() == "iwls_tr" or \
                optim_algo.lower() == "irls_trust_region" or \
                optim_algo.lower() == "iwls_trust_region" or \
                optim_algo.lower() == "irls-trust-region" or \
                optim_algo.lower() == "iwls-trust-region":
            newton_type_mode = True
        # Set learning rae defaults if not set by user.
        if learning_rate is None:
            if newton_type_mode:
                learning_rate = 1
            else:
                learning_rate = 0.5

        # Check that newton-rhapson is called properly:
        if newton_type_mode:
            if learning_rate != 1:
                logger.warning(
                    "Newton-rhapson or IRLS in base_glm_all is used with learning rate " +
                    str(learning_rate) +
                    ". Newton-rhapson and IRLS should only be used with learning rate = 1."
                )

        # Report all parameters after all defaults were imputed in settings:
        logger.debug("Optimizer settings in base_glm_all Estimator.train():")
        logger.debug("learning_rate " + str(learning_rate))
        logger.debug("convergence_criteria " + str(convergence_criteria))
        logger.debug("loss_window_size " + str(loss_window_size))
        logger.debug("stopping_criteria " + str(stopping_criteria))
        logger.debug("train_mu " + str(train_mu))
        logger.debug("train_r " + str(train_r))
        logger.debug("use_batching " + str(use_batching))
        logger.debug("optim_algo " + str(optim_algo))
        if len(kwargs) > 0:
            logger.debug("**kwargs: ")
            logger.debug(kwargs)

        if train_mu or train_r:
            if use_batching:
                loss = self.model.batched_data_model.loss
                train_op = self.model.trainer_batch.train_op_by_name(optim_algo)
            else:
                loss = self.model.full_data_model.loss
                train_op = self.model.trainer_full.train_op_by_name(optim_algo)

            super().train(*args,
                          feed_dict={"learning_rate:0": learning_rate},
                          convergence_criteria=convergence_criteria,
                          loss_window_size=loss_window_size,
                          stopping_criteria=stopping_criteria,
                          loss=loss,
                          train_op=train_op,
                          **kwargs)

    def train_sequence(self, training_strategy):
        if isinstance(training_strategy, Enum):
            training_strategy = training_strategy.value
        elif isinstance(training_strategy, str):
            training_strategy = self.TrainingStrategies[training_strategy].value

        if training_strategy is None:
            training_strategy = self.TrainingStrategies.DEFAULT.value

        logger.info("training strategy:\n%s", pprint.pformat(training_strategy))

        for idx, d in enumerate(training_strategy):
            self.model.model_vars.converged = False
            logger.info("Beginning with training sequence #%d", idx + 1)
            self.train(**d)
            logger.info("Training sequence #%d complete", idx + 1)

    @property
    def input_data(self):
        return self._input_data

    @property
    def a_var(self):
        return self.to_xarray("a_var", coords=self.input_data.data.coords)

    @property
    def b_var(self):
        return self.to_xarray("b_var", coords=self.input_data.data.coords)

    @property
    def loss(self):
        return self.to_xarray("loss")

    @property
    def log_likelihood(self):
        return self.to_xarray("log_likelihood", coords=self.input_data.data.coords)

    @property
    def gradients(self):
        return self.to_xarray("gradients", coords=self.input_data.data.coords)

    @property
    def hessians(self):
        return self.to_xarray("hessians", coords=self.input_data.data.coords)

    @property
    def fisher_inv(self):
        return self.to_xarray("fisher_inv", coords=self.input_data.data.coords)

    def finalize(self):
        if self.noise_model == "nb":
            from .external_nb import EstimatorStoreXArray
        else:
            raise ValueError("noise model not recognized")

        store = EstimatorStoreXArray(self)
        logger.debug("Closing session")
        self.close_session()
        return store

    @abc.abstractmethod
    def init_par(
            self,
            input_data,
            init_a,
            init_b,
            init_model
    ):
        pass
